from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import random
import statistics
import sys
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from householder_rope.attention import HouseholderSelfAttention  # noqa: E402
from householder_rope.core import BlockDiagonalRoPECore, HouseholderRoPE, HouseholderRoPEConfig  # noqa: E402


LOGGER = logging.getLogger("real_data_scale_harness")


@dataclass(frozen=True)
class RopeVariant:
    """One RoPE ablation point for the realistic-data harness."""

    label: str
    num_reflectors: int
    init: str


@dataclass(frozen=True)
class RealDataHarnessConfig:
    """Serializable config for the realistic-data single-device harness."""

    backend: str
    dataset_name: str
    dataset_config: str
    tokenizer_name: str
    train_text_limit: int
    eval_text_limit: int
    seq_len: int
    batch_size: int
    eval_batch_size: int
    gradient_accumulation_steps: int
    train_steps: int
    eval_every: int
    eval_batches: int
    num_layers: int
    embed_dim: int
    num_heads: int
    mlp_ratio: float
    learning_rate: float
    weight_decay: float
    seed: int
    use_compile: bool
    use_bf16: bool
    householder_init: str
    reflector_sweep: tuple[int, ...]
    output_dir: Path
    output_stem: str

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["output_dir"] = str(self.output_dir)
        return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark Householder-RoPE on a realistic language-modeling workload."
    )
    parser.add_argument(
        "--backend",
        default="auto",
        choices=["auto", "torch", "flax"],
        help="Execution backend. 'auto' uses Flax on TPU and PyTorch otherwise.",
    )
    parser.add_argument("--dataset-name", default="wikitext", help="Hugging Face dataset name.")
    parser.add_argument(
        "--dataset-config",
        default="wikitext-103-raw-v1",
        help="Hugging Face dataset configuration.",
    )
    parser.add_argument(
        "--tokenizer-name",
        default="gpt2",
        help="Tokenizer used to build causal LM batches.",
    )
    parser.add_argument(
        "--train-text-limit",
        type=int,
        default=12000,
        help="How many raw training records to keep before tokenization.",
    )
    parser.add_argument(
        "--eval-text-limit",
        type=int,
        default=1500,
        help="How many raw validation records to keep before tokenization.",
    )
    parser.add_argument("--seq-len", type=int, default=256, help="Token sequence length.")
    parser.add_argument("--batch-size", type=int, default=8, help="Training batch size.")
    parser.add_argument("--eval-batch-size", type=int, default=8, help="Evaluation batch size.")
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=1,
        help="Number of microsteps per optimizer step.",
    )
    parser.add_argument("--train-steps", type=int, default=60, help="Optimizer steps per variant.")
    parser.add_argument("--eval-every", type=int, default=10, help="Evaluation frequency in steps.")
    parser.add_argument("--eval-batches", type=int, default=8, help="Validation batches per eval.")
    parser.add_argument("--num-layers", type=int, default=2, help="Number of transformer blocks.")
    parser.add_argument("--embed-dim", type=int, default=512, help="Embedding width.")
    parser.add_argument("--num-heads", type=int, default=8, help="Attention heads.")
    parser.add_argument("--mlp-ratio", type=float, default=4.0, help="Feed-forward hidden ratio.")
    parser.add_argument("--learning-rate", type=float, default=3.0e-4, help="AdamW learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1.0e-2, help="AdamW weight decay.")
    parser.add_argument("--seed", type=int, default=17, help="Random seed.")
    parser.add_argument(
        "--use-compile",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable torch.compile on CUDA when available.",
    )
    parser.add_argument(
        "--use-bf16",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable bf16 autocast or bf16 params when the backend supports it.",
    )
    parser.add_argument(
        "--householder-init",
        default="jittered_pairs",
        choices=["paired_identity", "jittered_pairs", "random"],
        help="Initialization for non-zero reflector variants.",
    )
    parser.add_argument(
        "--reflector-sweep",
        type=int,
        nargs="+",
        default=[0, 8],
        help="Reflector counts to benchmark. Zero is the standard RoPE baseline.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "artifacts",
        help="Directory for JSON, CSV, and plot outputs.",
    )
    parser.add_argument(
        "--output-stem",
        default="colab_real_data_scale_harness",
        help="Shared stem for the output artifact names.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )
    return parser.parse_args()


def configure_logging(level: str) -> None:
    logging.basicConfig(level=getattr(logging, level), format="%(asctime)s | %(levelname)s | %(message)s")


def import_dataset_runtime():
    try:
        from datasets import load_dataset
        from transformers import AutoTokenizer
    except ImportError as exc:  # pragma: no cover - environment dependent.
        raise RuntimeError(
            "This harness needs `datasets` and `transformers`. Install the Colab runtime extras first."
        ) from exc
    return load_dataset, AutoTokenizer


def import_plot_runtime():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - environment dependent.
        raise RuntimeError("This harness needs `matplotlib` for plot generation.") from exc
    return plt


def resolve_backend(requested: str) -> str:
    if requested != "auto":
        return requested
    return "flax" if "COLAB_TPU_ADDR" in os.environ else "torch"


def seed_everything(seed: int, *, backend: str) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if backend == "flax":
        import jax  # noqa: PLC0415

        _ = jax.random.PRNGKey(seed)


def to_serializable(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: to_serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_serializable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def collect_environment(backend: str) -> dict[str, Any]:
    environment: dict[str, Any] = {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "backend": backend,
        "torch_version": torch.__version__,
        "torch_cuda_available": torch.cuda.is_available(),
        "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    if backend == "flax":
        try:
            import flax  # noqa: PLC0415
            import jax  # noqa: PLC0415

            environment["flax_version"] = flax.__version__
            environment["jax_version"] = jax.__version__
            environment["jax_backend"] = jax.default_backend()
            environment["jax_devices"] = [str(device) for device in jax.devices()]
        except ImportError as exc:  # pragma: no cover - environment dependent.
            environment["jax_import_error"] = str(exc)
    return environment


def build_variants(reflector_sweep: tuple[int, ...], householder_init: str) -> list[RopeVariant]:
    variants: list[RopeVariant] = []
    seen_counts: set[int] = set()
    for count in reflector_sweep:
        if count < 0:
            raise ValueError(f"Reflector counts must be non-negative, received {count}.")
        if count in seen_counts:
            continue
        seen_counts.add(count)
        if count == 0:
            variants.append(RopeVariant(label="standard_rope", num_reflectors=0, init="paired_identity"))
        else:
            variants.append(
                RopeVariant(
                    label=f"householder_m{count}",
                    num_reflectors=count,
                    init=householder_init,
                )
            )
    return variants

def build_lm_splits(config: RealDataHarnessConfig):
    load_dataset, AutoTokenizer = import_dataset_runtime()
    raw = load_dataset(config.dataset_name, config.dataset_config)
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    trimmed = {
        "train": raw["train"].select(range(min(config.train_text_limit, len(raw["train"])))),
        "validation": raw["validation"].select(range(min(config.eval_text_limit, len(raw["validation"])))),
    }

    def tokenize_batch(batch: dict[str, list[str]]) -> dict[str, Any]:
        return tokenizer(batch["text"], return_attention_mask=False)

    tokenized = {
        name: split.map(tokenize_batch, batched=True, remove_columns=split.column_names)
        for name, split in trimmed.items()
    }

    block_size = config.seq_len + 1
    eos_token_id = tokenizer.eos_token_id

    def group_texts(examples: dict[str, list[list[int]]]) -> dict[str, list[list[int]]]:
        concatenated: list[int] = []
        for ids in examples["input_ids"]:
            concatenated.extend(ids)
            if eos_token_id is not None:
                concatenated.append(eos_token_id)
        usable = (len(concatenated) // block_size) * block_size
        if usable == 0:
            return {"input_ids": [], "labels": []}
        chunks = [concatenated[index : index + block_size] for index in range(0, usable, block_size)]
        return {
            "input_ids": [chunk[:-1] for chunk in chunks],
            "labels": [chunk[1:] for chunk in chunks],
        }

    grouped = {
        name: split.map(group_texts, batched=True, batch_size=512, remove_columns=split.column_names)
        for name, split in tokenized.items()
    }
    for split in grouped.values():
        split.set_format(type="numpy", columns=["input_ids", "labels"])

    summary = {
        "train_sequences": len(grouped["train"]),
        "validation_sequences": len(grouped["validation"]),
        "vocab_size": int(tokenizer.vocab_size),
    }
    return tokenizer, grouped, summary


def ensure_split_sizes(config: RealDataHarnessConfig, split_summary: dict[str, Any]) -> None:
    if split_summary["train_sequences"] < config.batch_size:
        raise ValueError(
            f"Train split only has {split_summary['train_sequences']} sequences, "
            f"which is smaller than batch_size={config.batch_size}."
        )
    if split_summary["validation_sequences"] < config.eval_batch_size:
        raise ValueError(
            f"Validation split only has {split_summary['validation_sequences']} sequences, "
            f"which is smaller than eval_batch_size={config.eval_batch_size}."
        )


def batch_iterator(split, batch_size: int, *, shuffle: bool, seed: int) -> Iterator[dict[str, np.ndarray]]:
    rng = np.random.default_rng(seed)
    indices = np.arange(len(split))
    while True:
        if shuffle:
            rng.shuffle(indices)
        for start in range(0, len(indices) - batch_size + 1, batch_size):
            batch_indices = indices[start : start + batch_size]
            batch = split[batch_indices.tolist()]
            yield {
                "input_ids": np.asarray(batch["input_ids"], dtype=np.int64),
                "labels": np.asarray(batch["labels"], dtype=np.int64),
            }
        if not shuffle:
            break


class TorchFeedForward(nn.Module):
    """Minimal feed-forward block for the realistic-data LM harness."""

    def __init__(self, embed_dim: int, mlp_ratio: float) -> None:
        super().__init__()
        hidden_dim = int(embed_dim * mlp_ratio)
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TorchTransformerBlock(nn.Module):
    """One causal self-attention block using Householder-RoPE."""

    def __init__(self, *, embed_dim: int, num_heads: int, mlp_ratio: float, variant: RopeVariant) -> None:
        super().__init__()
        head_dim = embed_dim // num_heads
        rope = HouseholderRoPE(
            num_heads=num_heads,
            head_dim=head_dim,
            config=HouseholderRoPEConfig(
                mode="per_head",
                num_reflectors=variant.num_reflectors,
                init=variant.init,
                rope_ndim=1,
                enforce_SO=variant.num_reflectors % 2 == 0,
            ),
            rope_core=BlockDiagonalRoPECore(dim=head_dim, ndim=1),
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = HouseholderSelfAttention(embed_dim=embed_dim, num_heads=num_heads, rope=rope, dropout_p=0.0)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ff = TorchFeedForward(embed_dim, mlp_ratio)

    def forward(self, x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), pos, is_causal=True)
        x = x + self.ff(self.norm2(x))
        return x


class TorchHouseholderLM(nn.Module):
    """Small causal LM used for realistic-data scaling tests."""

    def __init__(
        self,
        *,
        vocab_size: int,
        embed_dim: int,
        num_heads: int,
        num_layers: int,
        mlp_ratio: float,
        variant: RopeVariant,
    ) -> None:
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, embed_dim)
        self.blocks = nn.ModuleList(
            [
                TorchTransformerBlock(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    variant=variant,
                )
                for _ in range(num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        pos = torch.arange(input_ids.shape[1], device=input_ids.device, dtype=torch.float32)
        x = self.token_embed(input_ids)
        for block in self.blocks:
            x = block(x, pos)
        x = self.final_norm(x)
        return self.lm_head(x)


def count_torch_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def evaluate_torch(
    model: nn.Module,
    split,
    *,
    batch_size: int,
    eval_batches: int,
    seed: int,
    device: torch.device,
) -> float:
    iterator = batch_iterator(split, batch_size, shuffle=False, seed=seed)
    losses: list[float] = []
    model.eval()
    with torch.no_grad():
        for batch_index, batch in enumerate(iterator):
            if batch_index >= eval_batches:
                break
            input_ids = torch.from_numpy(batch["input_ids"]).to(device)
            labels = torch.from_numpy(batch["labels"]).to(device)
            logits = model(input_ids)
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1))
            losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses))


def run_torch_variant(
    config: RealDataHarnessConfig,
    variant: RopeVariant,
    splits,
    *,
    vocab_size: int,
) -> dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_model = TorchHouseholderLM(
        vocab_size=vocab_size,
        embed_dim=config.embed_dim,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
        mlp_ratio=config.mlp_ratio,
        variant=variant,
    ).to(device)
    parameter_count = count_torch_parameters(base_model)
    model = base_model
    if config.use_compile and hasattr(torch, "compile") and device.type == "cuda":
        model = torch.compile(base_model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    train_iter = batch_iterator(splits["train"], config.batch_size, shuffle=True, seed=config.seed)
    use_bf16 = bool(config.use_bf16 and device.type == "cuda" and torch.cuda.is_bf16_supported())
    history: list[dict[str, float]] = []
    tokens_per_step = config.batch_size * config.seq_len * config.gradient_accumulation_steps
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    for step in range(1, config.train_steps + 1):
        step_start = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        train_loss_total = 0.0
        for _ in range(config.gradient_accumulation_steps):
            batch = next(train_iter)
            input_ids = torch.from_numpy(batch["input_ids"]).to(device)
            labels = torch.from_numpy(batch["labels"]).to(device)
            amp_context = (
                torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                if use_bf16
                else nullcontext()
            )
            with amp_context:
                logits = model(input_ids)
                loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1))
            (loss / config.gradient_accumulation_steps).backward()
            train_loss_total += float(loss.detach().cpu())
        optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        step_ms = (time.perf_counter() - step_start) * 1000.0
        record = {
            "step": step,
            "train_loss": train_loss_total / config.gradient_accumulation_steps,
            "step_ms": step_ms,
            "tokens_per_second": tokens_per_step / max(step_ms / 1000.0, 1.0e-9),
        }
        if step % config.eval_every == 0 or step == config.train_steps:
            record["eval_loss"] = evaluate_torch(
                model,
                splits["validation"],
                batch_size=config.eval_batch_size,
                eval_batches=config.eval_batches,
                seed=config.seed,
                device=device,
            )
        history.append(record)

    eval_history = [entry["eval_loss"] for entry in history if "eval_loss" in entry]
    return {
        "backend": "torch",
        "variant": variant.label,
        "num_reflectors": variant.num_reflectors,
        "parameter_count": parameter_count,
        "final_train_loss": history[-1]["train_loss"],
        "final_eval_loss": eval_history[-1] if eval_history else None,
        "mean_step_ms": float(statistics.mean(entry["step_ms"] for entry in history)),
        "mean_tokens_per_second": float(statistics.mean(entry["tokens_per_second"] for entry in history)),
        "peak_memory_gb": (
            float(torch.cuda.max_memory_allocated(device) / (1024**3))
            if device.type == "cuda"
            else None
        ),
        "history": history,
    }

def build_flax_runtime():
    try:
        import flax.linen as flax_nn
        import jax
        import jax.numpy as jnp
        import optax
    except ImportError as exc:  # pragma: no cover - environment dependent.
        raise RuntimeError(
            "The Flax backend needs `jax`, `flax`, and `optax`. Install the Colab runtime extras first."
        ) from exc

    from householder_rope.flax_modules import FlaxHouseholderSelfAttention
    from householder_rope.jax_core import JaxHouseholderRoPEConfig

    class FlaxFeedForward(flax_nn.Module):
        embed_dim: int
        mlp_ratio: float
        param_dtype: Any = jnp.float32

        @flax_nn.compact
        def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
            hidden_dim = int(self.embed_dim * self.mlp_ratio)
            x = flax_nn.Dense(hidden_dim, param_dtype=self.param_dtype)(x)
            x = flax_nn.gelu(x)
            x = flax_nn.Dense(self.embed_dim, param_dtype=self.param_dtype)(x)
            return x

    class FlaxTransformerBlock(flax_nn.Module):
        embed_dim: int
        num_heads: int
        mlp_ratio: float
        variant: RopeVariant
        param_dtype: Any = jnp.float32

        @flax_nn.compact
        def __call__(self, x: jnp.ndarray, pos: jnp.ndarray) -> jnp.ndarray:
            tokens = x.shape[1]
            causal_mask = jnp.triu(
                jnp.full((1, 1, tokens, tokens), fill_value=-1.0e30, dtype=self.param_dtype),
                k=1,
            )
            rope_config = JaxHouseholderRoPEConfig(
                mode="per_head",
                num_reflectors=self.variant.num_reflectors,
                init=self.variant.init,
                rope_ndim=1,
                enforce_SO=self.variant.num_reflectors % 2 == 0,
            )
            y = flax_nn.LayerNorm(param_dtype=self.param_dtype)(x)
            y = FlaxHouseholderSelfAttention(
                embed_dim=self.embed_dim,
                num_heads=self.num_heads,
                rope_config=rope_config,
                dropout_rate=0.0,
                param_dtype=self.param_dtype,
            )(y, pos, deterministic=True, attn_mask=causal_mask)
            x = x + y
            y = flax_nn.LayerNorm(param_dtype=self.param_dtype)(x)
            y = FlaxFeedForward(self.embed_dim, self.mlp_ratio, param_dtype=self.param_dtype)(y)
            return x + y

    class FlaxHouseholderLM(flax_nn.Module):
        vocab_size: int
        embed_dim: int
        num_heads: int
        num_layers: int
        mlp_ratio: float
        variant: RopeVariant
        param_dtype: Any = jnp.float32

        @flax_nn.compact
        def __call__(self, input_ids: jnp.ndarray) -> jnp.ndarray:
            pos = jnp.arange(input_ids.shape[1], dtype=self.param_dtype)
            x = flax_nn.Embed(
                self.vocab_size,
                self.embed_dim,
                embedding_init=flax_nn.initializers.normal(stddev=0.02),
                param_dtype=self.param_dtype,
            )(input_ids)
            for index in range(self.num_layers):
                x = FlaxTransformerBlock(
                    embed_dim=self.embed_dim,
                    num_heads=self.num_heads,
                    mlp_ratio=self.mlp_ratio,
                    variant=self.variant,
                    param_dtype=self.param_dtype,
                    name=f"block_{index}",
                )(x, pos)
            x = flax_nn.LayerNorm(param_dtype=self.param_dtype, name="final_norm")(x)
            return flax_nn.Dense(
                self.vocab_size,
                use_bias=False,
                param_dtype=self.param_dtype,
                name="lm_head",
            )(x)

    def count_jax_parameters(tree: Any) -> int:
        return int(sum(int(np.prod(np.asarray(leaf.shape))) for leaf in jax.tree_util.tree_leaves(tree)))

    def maybe_jax_memory_gb() -> float | None:
        try:
            stats = jax.devices()[0].memory_stats()
        except Exception:  # pragma: no cover - backend dependent.
            return None
        if not stats:
            return None
        for key in ("bytes_in_use", "peak_bytes_in_use", "bytes_reserved"):
            if key in stats:
                return float(stats[key]) / (1024**3)
        return None

    def run_flax_variant(
        config: RealDataHarnessConfig,
        variant: RopeVariant,
        splits,
        *,
        vocab_size: int,
    ) -> dict[str, Any]:
        dtype = jnp.bfloat16 if config.use_bf16 else jnp.float32
        model = FlaxHouseholderLM(
            vocab_size=vocab_size,
            embed_dim=config.embed_dim,
            num_heads=config.num_heads,
            num_layers=config.num_layers,
            mlp_ratio=config.mlp_ratio,
            variant=variant,
            param_dtype=dtype,
        )
        init_batch = jnp.asarray(splits["train"][: config.batch_size]["input_ids"])
        params = model.init(jax.random.PRNGKey(config.seed), init_batch)["params"]
        parameter_count = count_jax_parameters(params)
        optimizer = optax.adamw(learning_rate=config.learning_rate, weight_decay=config.weight_decay)
        opt_state = optimizer.init(params)

        def loss_fn(current_params, input_ids, labels):
            logits = model.apply({"params": current_params}, input_ids)
            return optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean()

        @jax.jit
        def train_step(current_params, current_opt_state, input_ids, labels):
            loss, grads = jax.value_and_grad(loss_fn)(current_params, input_ids, labels)
            updates, next_opt_state = optimizer.update(grads, current_opt_state, current_params)
            next_params = optax.apply_updates(current_params, updates)
            return next_params, next_opt_state, loss

        @jax.jit
        def eval_step(current_params, input_ids, labels):
            return loss_fn(current_params, input_ids, labels)

        history: list[dict[str, float]] = []
        train_iter = batch_iterator(splits["train"], config.batch_size, shuffle=True, seed=config.seed)
        tokens_per_step = config.batch_size * config.seq_len * config.gradient_accumulation_steps

        for step in range(1, config.train_steps + 1):
            step_start = time.perf_counter()
            train_loss_total = 0.0
            for _ in range(config.gradient_accumulation_steps):
                batch = next(train_iter)
                input_ids = jnp.asarray(batch["input_ids"])
                labels = jnp.asarray(batch["labels"])
                params, opt_state, loss = train_step(params, opt_state, input_ids, labels)
                train_loss_total += float(jax.device_get(loss))
            step_ms = (time.perf_counter() - step_start) * 1000.0
            record = {
                "step": step,
                "train_loss": train_loss_total / config.gradient_accumulation_steps,
                "step_ms": step_ms,
                "tokens_per_second": tokens_per_step / max(step_ms / 1000.0, 1.0e-9),
            }
            if step % config.eval_every == 0 or step == config.train_steps:
                eval_losses: list[float] = []
                eval_iter = batch_iterator(
                    splits["validation"],
                    config.eval_batch_size,
                    shuffle=False,
                    seed=config.seed,
                )
                for batch_index, batch in enumerate(eval_iter):
                    if batch_index >= config.eval_batches:
                        break
                    loss = eval_step(params, jnp.asarray(batch["input_ids"]), jnp.asarray(batch["labels"]))
                    eval_losses.append(float(jax.device_get(loss)))
                record["eval_loss"] = float(np.mean(eval_losses))
            history.append(record)

        eval_history = [entry["eval_loss"] for entry in history if "eval_loss" in entry]
        return {
            "backend": "flax",
            "variant": variant.label,
            "num_reflectors": variant.num_reflectors,
            "parameter_count": parameter_count,
            "final_train_loss": history[-1]["train_loss"],
            "final_eval_loss": eval_history[-1] if eval_history else None,
            "mean_step_ms": float(statistics.mean(entry["step_ms"] for entry in history)),
            "mean_tokens_per_second": float(statistics.mean(entry["tokens_per_second"] for entry in history)),
            "peak_memory_gb": maybe_jax_memory_gb(),
            "history": history,
        }

    return run_flax_variant, jax


def plot_results(results: list[dict[str, Any]], *, loss_curve_path: Path, throughput_path: Path) -> None:
    plt = import_plot_runtime()

    plt.figure(figsize=(10, 4.5))
    for result in results:
        steps = [entry["step"] for entry in result["history"]]
        train_loss = [entry["train_loss"] for entry in result["history"]]
        plt.plot(steps, train_loss, label=f"{result['variant']} train")
        eval_steps = [entry["step"] for entry in result["history"] if "eval_loss" in entry]
        eval_loss = [entry["eval_loss"] for entry in result["history"] if "eval_loss" in entry]
        if eval_steps:
            plt.plot(eval_steps, eval_loss, linestyle="--", marker="o", label=f"{result['variant']} eval")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("Householder-RoPE realistic-data loss curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(loss_curve_path, dpi=160)
    plt.close()

    labels = [result["variant"] for result in results]
    values = [result["mean_tokens_per_second"] for result in results]
    plt.figure(figsize=(8, 4.5))
    plt.bar(labels, values)
    plt.ylabel("Tokens / second")
    plt.title("Mean training throughput")
    plt.tight_layout()
    plt.savefig(throughput_path, dpi=160)
    plt.close()


def write_summary_csv(results: list[dict[str, Any]], path: Path) -> None:
    header = [
        "variant",
        "backend",
        "num_reflectors",
        "parameter_count",
        "final_train_loss",
        "final_eval_loss",
        "mean_step_ms",
        "mean_tokens_per_second",
        "peak_memory_gb",
    ]
    rows = [
        [
            result["variant"],
            result["backend"],
            result["num_reflectors"],
            result["parameter_count"],
            result["final_train_loss"],
            result["final_eval_loss"],
            result["mean_step_ms"],
            result["mean_tokens_per_second"],
            result["peak_memory_gb"],
        ]
        for result in results
    ]
    lines = [",".join(header)] + [",".join(str(item) for item in row) for row in rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

def print_summary(results: list[dict[str, Any]]) -> None:
    for result in results:
        LOGGER.info(
            "%s | backend=%s | final_eval_loss=%s | mean_step_ms=%.3f | tokens_per_second=%.1f",
            result["variant"],
            result["backend"],
            "n/a" if result["final_eval_loss"] is None else f"{result['final_eval_loss']:.4f}",
            result["mean_step_ms"],
            result["mean_tokens_per_second"],
        )


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)

    backend = resolve_backend(args.backend)
    config = RealDataHarnessConfig(
        backend=backend,
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        tokenizer_name=args.tokenizer_name,
        train_text_limit=args.train_text_limit,
        eval_text_limit=args.eval_text_limit,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        train_steps=args.train_steps,
        eval_every=args.eval_every,
        eval_batches=args.eval_batches,
        num_layers=args.num_layers,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        mlp_ratio=args.mlp_ratio,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        seed=args.seed,
        use_compile=args.use_compile,
        use_bf16=args.use_bf16,
        householder_init=args.householder_init,
        reflector_sweep=tuple(args.reflector_sweep),
        output_dir=args.output_dir,
        output_stem=args.output_stem,
    )
    if config.embed_dim % config.num_heads != 0:
        raise ValueError(
            f"embed_dim={config.embed_dim} must be divisible by num_heads={config.num_heads}."
        )

    seed_everything(config.seed, backend=backend)
    LOGGER.info(
        "Building realistic-data dataset pipeline for %s/%s",
        config.dataset_name,
        config.dataset_config,
    )
    _, splits, dataset_summary = build_lm_splits(config)
    ensure_split_sizes(config, dataset_summary)
    environment = collect_environment(backend)
    variants = build_variants(config.reflector_sweep, config.householder_init)

    LOGGER.info("Running %d variants on backend=%s", len(variants), backend)
    if backend == "flax":
        run_flax_variant, jax = build_flax_runtime()
        if jax.default_backend() not in {"gpu", "tpu"}:
            LOGGER.warning("Flax backend is running on %s, not TPU/GPU.", jax.default_backend())
        run_variant = lambda variant: run_flax_variant(
            config,
            variant,
            splits,
            vocab_size=dataset_summary["vocab_size"],
        )
    else:
        run_variant = lambda variant: run_torch_variant(
            config,
            variant,
            splits,
            vocab_size=dataset_summary["vocab_size"],
        )

    results = [run_variant(variant) for variant in variants]

    config.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = config.output_dir / f"{config.output_stem}_metrics.json"
    summary_path = config.output_dir / f"{config.output_stem}_summary.csv"
    loss_curve_path = config.output_dir / f"{config.output_stem}_loss_curves.png"
    throughput_path = config.output_dir / f"{config.output_stem}_throughput.png"

    payload = {
        "config": config.to_dict(),
        "environment": environment,
        "dataset_summary": dataset_summary,
        "variants": [to_serializable(asdict(variant)) for variant in variants],
        "results": results,
    }
    metrics_path.write_text(json.dumps(to_serializable(payload), indent=2), encoding="utf-8")
    write_summary_csv(results, summary_path)
    plot_results(results, loss_curve_path=loss_curve_path, throughput_path=throughput_path)
    print_summary(results)

    LOGGER.info("Wrote metrics to %s", metrics_path)
    LOGGER.info("Wrote summary to %s", summary_path)
    LOGGER.info("Wrote loss curves to %s", loss_curve_path)
    LOGGER.info("Wrote throughput plot to %s", throughput_path)


if __name__ == "__main__":
    main()
