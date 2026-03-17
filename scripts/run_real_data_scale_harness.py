from __future__ import annotations

import argparse
import csv
import json
import logging
import math
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
from householder_rope.diagnostics import summarize_householder_rope_diagnostics  # noqa: E402


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
    log_every: int
    diagnostics_every: int
    diagnostic_token_limit: int
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
    parser.add_argument(
        "--log-every",
        type=int,
        default=5,
        help="Emit an INFO training summary every N optimizer steps.",
    )
    parser.add_argument(
        "--diagnostics-every",
        type=int,
        default=10,
        help="Run the deeper probe diagnostics every N optimizer steps.",
    )
    parser.add_argument(
        "--diagnostic-token-limit",
        type=int,
        default=128,
        help="Maximum sequence length used by the dense diagnostics probe.",
    )
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
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s | %(levelname)s | %(message)s",
        stream=sys.stdout,
        force=True,
    )
    LOGGER.setLevel(getattr(logging, level))


def should_log_record(config: RealDataHarnessConfig, record: dict[str, Any]) -> bool:
    return (
        record["step"] == 1
        or record["step"] == config.train_steps
        or record["step"] % config.log_every == 0
        or "eval_loss" in record
        or "diagnostics" in record
    )


def log_variant_start(config: RealDataHarnessConfig, variant: RopeVariant, *, backend: str) -> None:
    LOGGER.info(
        "Starting %s | backend=%s | reflectors=%d | steps=%d | batch=%d | seq_len=%d | log_every=%d | diagnostics_every=%d",
        variant.label,
        backend,
        variant.num_reflectors,
        config.train_steps,
        config.batch_size,
        config.seq_len,
        config.log_every,
        config.diagnostics_every,
    )


def log_variant_paths(variant: RopeVariant, *, history_jsonl_path: Path, history_csv_path: Path) -> None:
    LOGGER.info(
        "%s | history_jsonl=%s | history_csv=%s",
        variant.label,
        history_jsonl_path,
        history_csv_path,
    )


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


def exp_clamped(value: float, *, limit: float = 20.0) -> float:
    return float(math.exp(min(value, limit)))


def tensor_rms(value: torch.Tensor) -> float:
    value = value.detach().float()
    return float(torch.sqrt(torch.mean(value.square())).cpu())


def tensor_std(value: torch.Tensor) -> float:
    return float(value.detach().float().std(unbiased=False).cpu())


def tensor_max_abs(value: torch.Tensor) -> float:
    return float(value.detach().float().abs().max().cpu())


def parameter_global_norm(parameters: Iterator[torch.Tensor]) -> float:
    total = 0.0
    for parameter in parameters:
        value = parameter.detach().float()
        total += float(torch.sum(value.square()).cpu())
    return math.sqrt(total)


def gradient_global_norm(parameters: Iterator[torch.Tensor]) -> float:
    total = 0.0
    for parameter in parameters:
        if parameter.grad is None:
            continue
        grad = parameter.grad.detach().float()
        total += float(torch.sum(grad.square()).cpu())
    return math.sqrt(total)


def reduce_numeric_metric(value: Any) -> dict[str, float]:
    array = np.asarray(to_serializable(value), dtype=np.float64)
    if array.size == 0:
        return {}
    flat = array.reshape(-1)
    return {
        "mean": float(np.mean(flat)),
        "max": float(np.max(flat)),
        "min": float(np.min(flat)),
    }


def block_mixing_offdiag_mean(value: Any) -> float | None:
    array = np.asarray(to_serializable(value), dtype=np.float64)
    if array.size == 0:
        return None
    if array.ndim == 2:
        array = array[None, ...]
    if array.ndim != 3 or array.shape[-1] != array.shape[-2]:
        return None
    mask = ~np.eye(array.shape[-1], dtype=bool)
    offdiag = array[:, mask]
    if offdiag.size == 0:
        return 0.0
    return float(np.mean(offdiag))


def reduce_rope_diagnostics(summary: dict[str, Any]) -> dict[str, float]:
    reduced: dict[str, float] = {}
    for key in (
        "orthogonality_defect",
        "relativity_defect",
        "reversibility_defect",
        "commutator_defect",
        "attention_logit_path_error",
    ):
        if key not in summary:
            continue
        stats = reduce_numeric_metric(summary[key])
        if not stats:
            continue
        reduced[f"{key}_mean"] = stats["mean"]
        reduced[f"{key}_max"] = stats["max"]

    if "block_mixing_energy" in summary:
        stats = reduce_numeric_metric(summary["block_mixing_energy"])
        if stats:
            reduced["block_mixing_energy_mean"] = stats["mean"]
            reduced["block_mixing_energy_max"] = stats["max"]
        offdiag_mean = block_mixing_offdiag_mean(summary["block_mixing_energy"])
        if offdiag_mean is not None:
            reduced["block_mixing_offdiag_mean"] = offdiag_mean

    utilization = summary.get("reflector_utilization", {})
    field_map = {
        "raw_norms": "raw_reflector_norm",
        "pair_cosine_similarity": "pair_cosine_similarity",
        "identity_deviation": "identity_deviation",
        "orthogonality_defect": "utilization_orthogonality_defect",
        "gradient_norms": "reflector_gradient_norm",
    }
    for key, prefix in field_map.items():
        if key not in utilization:
            continue
        stats = reduce_numeric_metric(utilization[key])
        if not stats:
            continue
        reduced[f"{prefix}_mean"] = stats["mean"]
        reduced[f"{prefix}_max"] = stats["max"]
    return reduced


def flatten_metrics(value: Any, *, prefix: str = "") -> dict[str, Any]:
    flat: dict[str, Any] = {}
    if isinstance(value, dict):
        for key, item in value.items():
            next_prefix = f"{prefix}_{key}" if prefix else str(key)
            flat.update(flatten_metrics(item, prefix=next_prefix))
        return flat
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            next_prefix = f"{prefix}_{index}" if prefix else str(index)
            flat.update(flatten_metrics(item, prefix=next_prefix))
        return flat
    flat[prefix] = to_serializable(value)
    return flat


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(to_serializable(record)) + "\n")


def write_history_csv(history: list[dict[str, Any]], path: Path) -> None:
    rows = [flatten_metrics(record) for record in history]
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key in seen:
                continue
            fieldnames.append(key)
            seen.add(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def history_series(history: list[dict[str, Any]], key: str) -> tuple[list[int], list[float]]:
    steps: list[int] = []
    values: list[float] = []
    for record in history:
        if key not in record:
            continue
        steps.append(int(record["step"]))
        values.append(float(record[key]))
    return steps, values


def mean_from_history(history: list[dict[str, Any]], key: str) -> float | None:
    values = [float(record[key]) for record in history if key in record]
    if not values:
        return None
    return float(statistics.mean(values))


def last_from_history(history: list[dict[str, Any]], key: str) -> Any:
    for record in reversed(history):
        if key in record:
            return record[key]
    return None


def latest_diagnostics(history: list[dict[str, Any]]) -> dict[str, Any] | None:
    for record in reversed(history):
        if "diagnostics" in record:
            return record["diagnostics"]
    return None


def summarize_variant_result(
    *,
    backend: str,
    variant: RopeVariant,
    parameter_count: int,
    history: list[dict[str, Any]],
    peak_memory_gb: float | None,
    history_jsonl_path: Path,
    history_csv_path: Path,
) -> dict[str, Any]:
    eval_history = [entry["eval_loss"] for entry in history if "eval_loss" in entry]
    latest_probe = latest_diagnostics(history)
    result = {
        "backend": backend,
        "variant": variant.label,
        "num_reflectors": variant.num_reflectors,
        "parameter_count": parameter_count,
        "final_train_loss": history[-1]["train_loss"],
        "final_train_perplexity": history[-1].get("train_perplexity"),
        "final_eval_loss": eval_history[-1] if eval_history else None,
        "mean_step_ms": float(statistics.mean(entry["step_ms"] for entry in history)),
        "mean_tokens_per_second": float(statistics.mean(entry["tokens_per_second"] for entry in history)),
        "mean_grad_global_norm": mean_from_history(history, "grad_global_norm"),
        "mean_parameter_global_norm": mean_from_history(history, "parameter_global_norm"),
        "peak_memory_gb": peak_memory_gb,
        "latest_probe_summary": None if latest_probe is None else latest_probe.get("summary"),
        "history_jsonl_path": str(history_jsonl_path),
        "history_csv_path": str(history_csv_path),
        "history": history,
    }
    return result


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
    use_bf16: bool,
) -> float:
    iterator = batch_iterator(split, batch_size, shuffle=False, seed=seed)
    losses: list[float] = []
    training_state = model.training
    model.eval()
    amp_context = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if use_bf16 and device.type == "cuda"
        else nullcontext()
    )
    with torch.no_grad(), amp_context:
        for batch_index, batch in enumerate(iterator):
            if batch_index >= eval_batches:
                break
            input_ids = torch.from_numpy(batch["input_ids"]).to(device)
            labels = torch.from_numpy(batch["labels"]).to(device)
            logits = model(input_ids)
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1))
            losses.append(float(loss.detach().cpu()))
    model.train(training_state)
    return float(np.mean(losses))


def collect_torch_probe_metrics(
    model: TorchHouseholderLM,
    batch: dict[str, np.ndarray],
    *,
    device: torch.device,
    use_bf16: bool,
    diagnostic_token_limit: int,
) -> dict[str, Any]:
    probe_tokens = min(int(batch["input_ids"].shape[1]), diagnostic_token_limit)
    input_ids = torch.from_numpy(batch["input_ids"][:, :probe_tokens]).to(device)
    labels = torch.from_numpy(batch["labels"][:, :probe_tokens]).to(device)
    captured: dict[str, torch.Tensor] = {}
    handles: list[Any] = []

    def capture_tensor(name: str):
        def hook(_module, inputs, output):
            tensor = output[0] if isinstance(output, tuple) else output
            captured[name] = tensor.detach().float()
        return hook

    def capture_block(name: str):
        def hook(_module, inputs, output):
            captured[f"{name}.input"] = inputs[0].detach().float()
            captured[f"{name}.output"] = output.detach().float()
        return hook

    handles.append(model.token_embed.register_forward_hook(capture_tensor("token_embed")))
    handles.append(model.final_norm.register_forward_hook(capture_tensor("final_norm")))
    handles.append(model.lm_head.register_forward_hook(capture_tensor("lm_head")))
    for index, block in enumerate(model.blocks):
        prefix = f"block_{index}"
        handles.append(block.register_forward_hook(capture_block(prefix)))
        handles.append(block.norm1.register_forward_hook(capture_tensor(f"{prefix}.norm1")))
        handles.append(block.attn.register_forward_hook(capture_tensor(f"{prefix}.attn")))
        handles.append(block.norm2.register_forward_hook(capture_tensor(f"{prefix}.norm2")))
        handles.append(block.ff.register_forward_hook(capture_tensor(f"{prefix}.ff")))
        handles.append(block.attn.q_proj.register_forward_hook(capture_tensor(f"{prefix}.q_proj")))
        handles.append(block.attn.k_proj.register_forward_hook(capture_tensor(f"{prefix}.k_proj")))

    amp_context = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if use_bf16 and device.type == "cuda"
        else nullcontext()
    )
    training_state = model.training
    model.eval()
    with torch.no_grad(), amp_context:
        logits = model(input_ids)
    if training_state:
        model.train()
    for handle in handles:
        handle.remove()

    log_probs = logits.float().log_softmax(dim=-1)
    probs = log_probs.exp()
    entropy = -(probs * log_probs).sum(dim=-1).mean()
    confidence = probs.max(dim=-1).values.mean()
    top2 = torch.topk(probs, k=2, dim=-1).values
    top2_margin = (top2[..., 0] - top2[..., 1]).mean()
    probe_loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1))

    layer_metrics: dict[str, dict[str, float]] = {}
    rope_summaries: list[dict[str, float]] = []
    pos = torch.arange(probe_tokens, device=device, dtype=torch.float32)
    for index, block in enumerate(model.blocks):
        prefix = f"block_{index}"
        block_metrics: dict[str, float] = {
            "block_input_rms": tensor_rms(captured[f"{prefix}.input"]),
            "block_output_rms": tensor_rms(captured[f"{prefix}.output"]),
            "block_delta_rms": tensor_rms(captured[f"{prefix}.output"] - captured[f"{prefix}.input"]),
            "norm1_output_rms": tensor_rms(captured[f"{prefix}.norm1"]),
            "attention_output_rms": tensor_rms(captured[f"{prefix}.attn"]),
            "norm2_output_rms": tensor_rms(captured[f"{prefix}.norm2"]),
            "feedforward_output_rms": tensor_rms(captured[f"{prefix}.ff"]),
            "q_proj_rms": tensor_rms(captured[f"{prefix}.q_proj"]),
            "k_proj_rms": tensor_rms(captured[f"{prefix}.k_proj"]),
        }
        q = block.attn._split_heads(captured[f"{prefix}.q_proj"].to(device))
        k = block.attn._split_heads(captured[f"{prefix}.k_proj"].to(device))
        rope_summary = summarize_householder_rope_diagnostics(
            block.attn.rope,
            pos_a=pos,
            pos_b=pos + 1.0,
            q=q,
            k=k,
        )
        reduced_rope = {f"rope_{key}": value for key, value in reduce_rope_diagnostics(rope_summary).items()}
        block_metrics.update(reduced_rope)
        rope_summaries.append(reduced_rope)
        layer_metrics[prefix] = block_metrics

    def layer_mean(key: str) -> float | None:
        values = [metrics[key] for metrics in layer_metrics.values() if key in metrics]
        if not values:
            return None
        return float(statistics.mean(values))

    summary: dict[str, float | int] = {
        "probe_tokens": probe_tokens,
        "probe_batch_size": int(input_ids.shape[0]),
        "probe_loss": float(probe_loss.detach().cpu()),
        "probe_perplexity": exp_clamped(float(probe_loss.detach().cpu())),
        "probe_logits_mean": float(logits.detach().float().mean().cpu()),
        "probe_logits_std": tensor_std(logits),
        "probe_logits_max_abs": tensor_max_abs(logits),
        "probe_logit_entropy": float(entropy.detach().cpu()),
        "probe_confidence": float(confidence.detach().cpu()),
        "probe_top2_margin": float(top2_margin.detach().cpu()),
        "probe_token_embed_rms": tensor_rms(captured["token_embed"]),
        "probe_final_hidden_rms": tensor_rms(captured["final_norm"]),
        "probe_attention_output_rms_mean": layer_mean("attention_output_rms"),
        "probe_feedforward_output_rms_mean": layer_mean("feedforward_output_rms"),
        "probe_block_delta_rms_mean": layer_mean("block_delta_rms"),
    }
    for key in (
        "rope_orthogonality_defect_mean",
        "rope_relativity_defect_mean",
        "rope_reversibility_defect_mean",
        "rope_commutator_defect_mean",
        "rope_attention_logit_path_error_mean",
        "rope_block_mixing_offdiag_mean",
        "rope_identity_deviation_mean",
        "rope_pair_cosine_similarity_mean",
        "rope_raw_reflector_norm_mean",
        "rope_reflector_gradient_norm_mean",
    ):
        value = layer_mean(key)
        if value is not None:
            summary[key] = value
    return {
        "summary": summary,
        "layers": layer_metrics,
    }


def log_training_record(variant_label: str, record: dict[str, Any], train_steps: int) -> None:
    probe_entropy = "n/a" if "probe_logit_entropy" not in record else f"{record['probe_logit_entropy']:.3f}"
    rope_orth = "n/a" if "rope_orthogonality_defect_mean" not in record else f"{record['rope_orthogonality_defect_mean']:.2e}"
    rope_identity = (
        "n/a"
        if "rope_identity_deviation_mean" not in record
        else f"{record['rope_identity_deviation_mean']:.4f}"
    )
    eval_loss = "n/a" if "eval_loss" not in record else f"{record['eval_loss']:.4f}"
    memory = "n/a"
    if "memory_allocated_gb" in record:
        peak = record.get("peak_memory_gb", record["memory_allocated_gb"])
        memory = f"{record['memory_allocated_gb']:.2f}/{peak:.2f}GB"
    LOGGER.info(
        "%s | step %d/%d | train_loss=%.4f | train_ppl=%.2f | eval_loss=%s | grad_norm=%.4f | tok/s=%.1f | mem=%s | probe_entropy=%s | rope_orth=%s | rope_id=%s",
        variant_label,
        record["step"],
        train_steps,
        record["train_loss"],
        record["train_perplexity"],
        eval_loss,
        record["grad_global_norm"],
        record["tokens_per_second"],
        memory,
        probe_entropy,
        rope_orth,
        rope_identity,
    )


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
    train_model: nn.Module = base_model
    if config.use_compile and hasattr(torch, "compile") and device.type == "cuda":
        train_model = torch.compile(base_model)
    optimizer = torch.optim.AdamW(base_model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    train_iter = batch_iterator(splits["train"], config.batch_size, shuffle=True, seed=config.seed)
    use_bf16 = bool(config.use_bf16 and device.type == "cuda" and torch.cuda.is_bf16_supported())
    history: list[dict[str, Any]] = []
    tokens_per_step = config.batch_size * config.seq_len * config.gradient_accumulation_steps
    history_jsonl_path = config.output_dir / f"{config.output_stem}_{variant.label}_history.jsonl"
    history_csv_path = config.output_dir / f"{config.output_stem}_{variant.label}_history.csv"
    if history_jsonl_path.exists():
        history_jsonl_path.unlink()
    if history_csv_path.exists():
        history_csv_path.unlink()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    log_variant_start(config, variant, backend="torch")
    log_variant_paths(variant, history_jsonl_path=history_jsonl_path, history_csv_path=history_csv_path)

    for step in range(1, config.train_steps + 1):
        train_model.train()
        step_start = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        train_loss_total = 0.0
        last_batch: dict[str, np.ndarray] | None = None
        for _ in range(config.gradient_accumulation_steps):
            batch = next(train_iter)
            last_batch = batch
            input_ids = torch.from_numpy(batch["input_ids"]).to(device)
            labels = torch.from_numpy(batch["labels"]).to(device)
            amp_context = (
                torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                if use_bf16
                else nullcontext()
            )
            with amp_context:
                logits = train_model(input_ids)
                loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1))
            (loss / config.gradient_accumulation_steps).backward()
            train_loss_total += float(loss.detach().cpu())
        grad_norm = gradient_global_norm(base_model.parameters())
        optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        step_ms = (time.perf_counter() - step_start) * 1000.0
        train_loss = train_loss_total / config.gradient_accumulation_steps
        record: dict[str, Any] = {
            "step": step,
            "train_loss": train_loss,
            "train_perplexity": exp_clamped(train_loss),
            "step_ms": step_ms,
            "tokens_per_second": tokens_per_step / max(step_ms / 1000.0, 1.0e-9),
            "grad_global_norm": grad_norm,
            "parameter_global_norm": parameter_global_norm(base_model.parameters()),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
        }
        if device.type == "cuda":
            record["memory_allocated_gb"] = float(torch.cuda.memory_allocated(device) / (1024**3))
            record["peak_memory_gb"] = float(torch.cuda.max_memory_allocated(device) / (1024**3))
        if step % config.eval_every == 0 or step == config.train_steps:
            record["eval_loss"] = evaluate_torch(
                train_model,
                splits["validation"],
                batch_size=config.eval_batch_size,
                eval_batches=config.eval_batches,
                seed=config.seed,
                device=device,
                use_bf16=use_bf16,
            )
        if (step % config.diagnostics_every == 0 or step == 1 or step == config.train_steps) and last_batch is not None:
            diagnostics = collect_torch_probe_metrics(
                base_model,
                last_batch,
                device=device,
                use_bf16=use_bf16,
                diagnostic_token_limit=config.diagnostic_token_limit,
            )
            record["diagnostics"] = diagnostics
            record.update(diagnostics["summary"])
        history.append(record)
        append_jsonl(history_jsonl_path, record)
        if should_log_record(config, record):
            log_training_record(variant.label, record, config.train_steps)

    write_history_csv(history, history_csv_path)
    return summarize_variant_result(
        backend="torch",
        variant=variant,
        parameter_count=parameter_count,
        history=history,
        peak_memory_gb=float(torch.cuda.max_memory_allocated(device) / (1024**3)) if device.type == "cuda" else None,
        history_jsonl_path=history_jsonl_path,
        history_csv_path=history_csv_path,
    )


def build_flax_runtime():
    try:
        import flax.linen as flax_nn
        from flax.traverse_util import flatten_dict as flax_flatten_dict
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
        for key in ("peak_bytes_in_use", "bytes_in_use", "bytes_reserved"):
            if key in stats:
                return float(stats[key]) / (1024**3)
        return None

    def summarize_flax_reflectors(params: Any) -> dict[str, float]:
        flat = flax_flatten_dict(params, sep="/")
        arrays = [leaf for key, leaf in flat.items() if key.endswith("reflectors")]
        if not arrays:
            return {}
        norms = [np.asarray(jnp.linalg.norm(array, axis=-1)).reshape(-1) for array in arrays]
        merged = np.concatenate(norms)
        return {
            "rope_raw_reflector_norm_mean": float(np.mean(merged)),
            "rope_raw_reflector_norm_max": float(np.max(merged)),
        }

    def probe_flax_metrics(model: Any, params: Any, batch: dict[str, np.ndarray], diagnostic_token_limit: int) -> dict[str, Any]:
        input_ids = jnp.asarray(batch["input_ids"][:, :diagnostic_token_limit])
        labels = jnp.asarray(batch["labels"][:, :diagnostic_token_limit])
        logits = model.apply({"params": params}, input_ids)
        log_probs = jax.nn.log_softmax(logits, axis=-1)
        probs = jnp.exp(log_probs)
        entropy = -jnp.sum(probs * log_probs, axis=-1).mean()
        confidence = jnp.max(probs, axis=-1).mean()
        top2 = jax.lax.top_k(probs, k=2)[0]
        margin = (top2[..., 0] - top2[..., 1]).mean()
        loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean()
        summary = {
            "probe_tokens": int(input_ids.shape[1]),
            "probe_batch_size": int(input_ids.shape[0]),
            "probe_loss": float(loss),
            "probe_perplexity": exp_clamped(float(loss)),
            "probe_logits_mean": float(jnp.mean(logits)),
            "probe_logits_std": float(jnp.std(logits)),
            "probe_logits_max_abs": float(jnp.max(jnp.abs(logits))),
            "probe_logit_entropy": float(entropy),
            "probe_confidence": float(confidence),
            "probe_top2_margin": float(margin),
        }
        summary.update(summarize_flax_reflectors(params))
        return {"summary": summary, "layers": {}}

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
        history: list[dict[str, Any]] = []
        train_iter = batch_iterator(splits["train"], config.batch_size, shuffle=True, seed=config.seed)
        tokens_per_step = config.batch_size * config.seq_len * config.gradient_accumulation_steps
        history_jsonl_path = config.output_dir / f"{config.output_stem}_{variant.label}_history.jsonl"
        history_csv_path = config.output_dir / f"{config.output_stem}_{variant.label}_history.csv"
        if history_jsonl_path.exists():
            history_jsonl_path.unlink()
        if history_csv_path.exists():
            history_csv_path.unlink()

        log_variant_start(config, variant, backend="flax")
        log_variant_paths(variant, history_jsonl_path=history_jsonl_path, history_csv_path=history_csv_path)

        def loss_fn(current_params, input_ids, labels):
            logits = model.apply({"params": current_params}, input_ids)
            return optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean()

        @jax.jit
        def train_step(current_params, current_opt_state, input_ids, labels):
            loss, grads = jax.value_and_grad(loss_fn)(current_params, input_ids, labels)
            updates, next_opt_state = optimizer.update(grads, current_opt_state, current_params)
            next_params = optax.apply_updates(current_params, updates)
            return (
                next_params,
                next_opt_state,
                loss,
                optax.global_norm(grads),
                optax.global_norm(current_params),
                optax.global_norm(updates),
            )

        @jax.jit
        def eval_step(current_params, input_ids, labels):
            return loss_fn(current_params, input_ids, labels)

        for step in range(1, config.train_steps + 1):
            step_start = time.perf_counter()
            train_loss_total = 0.0
            grad_norm = 0.0
            param_norm = 0.0
            update_norm = 0.0
            last_batch: dict[str, np.ndarray] | None = None
            for _ in range(config.gradient_accumulation_steps):
                batch = next(train_iter)
                last_batch = batch
                params, opt_state, loss, grad_norm_value, param_norm_value, update_norm_value = train_step(
                    params,
                    opt_state,
                    jnp.asarray(batch["input_ids"]),
                    jnp.asarray(batch["labels"]),
                )
                train_loss_total += float(jax.device_get(loss))
                grad_norm = float(jax.device_get(grad_norm_value))
                param_norm = float(jax.device_get(param_norm_value))
                update_norm = float(jax.device_get(update_norm_value))
            step_ms = (time.perf_counter() - step_start) * 1000.0
            train_loss = train_loss_total / config.gradient_accumulation_steps
            record: dict[str, Any] = {
                "step": step,
                "train_loss": train_loss,
                "train_perplexity": exp_clamped(train_loss),
                "step_ms": step_ms,
                "tokens_per_second": tokens_per_step / max(step_ms / 1000.0, 1.0e-9),
                "grad_global_norm": grad_norm,
                "parameter_global_norm": param_norm,
                "update_global_norm": update_norm,
                "learning_rate": config.learning_rate,
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
            if (step % config.diagnostics_every == 0 or step == 1 or step == config.train_steps) and last_batch is not None:
                diagnostics = probe_flax_metrics(model, params, last_batch, config.diagnostic_token_limit)
                record["diagnostics"] = diagnostics
                record.update(diagnostics["summary"])
            history.append(record)
            append_jsonl(history_jsonl_path, record)
            if should_log_record(config, record):
                log_training_record(variant.label, record, config.train_steps)

        write_history_csv(history, history_csv_path)
        return summarize_variant_result(
            backend="flax",
            variant=variant,
            parameter_count=parameter_count,
            history=history,
            peak_memory_gb=maybe_jax_memory_gb(),
            history_jsonl_path=history_jsonl_path,
            history_csv_path=history_csv_path,
        )

    return run_flax_variant, jax


def plot_metric_grid(
    results: list[dict[str, Any]],
    *,
    specs: list[tuple[str, str, str]],
    path: Path,
    title: str,
) -> None:
    plt = import_plot_runtime()
    figure, axes = plt.subplots(2, 2, figsize=(12, 8))
    for axis, (key, panel_title, ylabel) in zip(axes.flat, specs):
        any_series = False
        for result in results:
            steps, values = history_series(result["history"], key)
            if not values:
                continue
            axis.plot(steps, values, label=result["variant"])
            any_series = True
        axis.set_title(panel_title)
        axis.set_xlabel("Step")
        axis.set_ylabel(ylabel)
        if any_series:
            axis.legend(fontsize=8)
        else:
            axis.text(0.5, 0.5, "No data", ha="center", va="center", transform=axis.transAxes)
    figure.suptitle(title)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def plot_results(
    results: list[dict[str, Any]],
    *,
    loss_curve_path: Path,
    throughput_path: Path,
    dynamics_path: Path,
    component_path: Path,
    rope_path: Path,
) -> None:
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

    plot_metric_grid(
        results,
        specs=[
            ("grad_global_norm", "Gradient norm", "L2 norm"),
            ("parameter_global_norm", "Parameter norm", "L2 norm"),
            ("probe_logit_entropy", "Probe logit entropy", "Entropy"),
            ("probe_confidence", "Probe confidence", "Probability"),
        ],
        path=dynamics_path,
        title="Householder-RoPE training dynamics",
    )
    plot_metric_grid(
        results,
        specs=[
            ("probe_token_embed_rms", "Token embedding RMS", "RMS"),
            ("probe_attention_output_rms_mean", "Attention output RMS", "RMS"),
            ("probe_feedforward_output_rms_mean", "Feed-forward output RMS", "RMS"),
            ("probe_final_hidden_rms", "Final hidden RMS", "RMS"),
        ],
        path=component_path,
        title="Householder-RoPE component diagnostics",
    )
    plot_metric_grid(
        results,
        specs=[
            ("rope_orthogonality_defect_mean", "RoPE orthogonality defect", "Relative error"),
            ("rope_identity_deviation_mean", "Identity deviation", "Relative error"),
            ("rope_block_mixing_offdiag_mean", "Off-block mixing energy", "Energy"),
            ("rope_attention_logit_path_error_mean", "Dense vs matrix-free logit error", "Relative error"),
        ],
        path=rope_path,
        title="Householder-RoPE transport diagnostics",
    )


def write_summary_csv(results: list[dict[str, Any]], path: Path) -> None:
    header = [
        "variant",
        "backend",
        "num_reflectors",
        "parameter_count",
        "final_train_loss",
        "final_train_perplexity",
        "final_eval_loss",
        "mean_step_ms",
        "mean_tokens_per_second",
        "mean_grad_global_norm",
        "mean_parameter_global_norm",
        "peak_memory_gb",
        "latest_probe_loss",
        "latest_probe_logit_entropy",
        "latest_probe_confidence",
        "latest_rope_orthogonality_defect_mean",
        "latest_rope_identity_deviation_mean",
        "history_jsonl_path",
        "history_csv_path",
    ]
    rows = []
    for result in results:
        latest_probe_summary = result.get("latest_probe_summary") or {}
        rows.append(
            [
                result["variant"],
                result["backend"],
                result["num_reflectors"],
                result["parameter_count"],
                result["final_train_loss"],
                result["final_train_perplexity"],
                result["final_eval_loss"],
                result["mean_step_ms"],
                result["mean_tokens_per_second"],
                result["mean_grad_global_norm"],
                result["mean_parameter_global_norm"],
                result["peak_memory_gb"],
                latest_probe_summary.get("probe_loss"),
                latest_probe_summary.get("probe_logit_entropy"),
                latest_probe_summary.get("probe_confidence"),
                latest_probe_summary.get("rope_orthogonality_defect_mean"),
                latest_probe_summary.get("rope_identity_deviation_mean"),
                result["history_jsonl_path"],
                result["history_csv_path"],
            ]
        )
    lines = [",".join(header)] + [",".join("" if item is None else str(item) for item in row) for row in rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def print_summary(results: list[dict[str, Any]]) -> None:
    for result in results:
        latest_probe_summary = result.get("latest_probe_summary") or {}
        LOGGER.info(
            "%s | backend=%s | final_eval_loss=%s | mean_step_ms=%.3f | tokens_per_second=%.1f | probe_entropy=%s | rope_orth=%s",
            result["variant"],
            result["backend"],
            "n/a" if result["final_eval_loss"] is None else f"{result['final_eval_loss']:.4f}",
            result["mean_step_ms"],
            result["mean_tokens_per_second"],
            "n/a"
            if latest_probe_summary.get("probe_logit_entropy") is None
            else f"{latest_probe_summary['probe_logit_entropy']:.3f}",
            "n/a"
            if latest_probe_summary.get("rope_orthogonality_defect_mean") is None
            else f"{latest_probe_summary['rope_orthogonality_defect_mean']:.2e}",
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
        log_every=args.log_every,
        diagnostics_every=args.diagnostics_every,
        diagnostic_token_limit=args.diagnostic_token_limit,
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

    config.output_dir.mkdir(parents=True, exist_ok=True)
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

    metrics_path = config.output_dir / f"{config.output_stem}_metrics.json"
    summary_path = config.output_dir / f"{config.output_stem}_summary.csv"
    loss_curve_path = config.output_dir / f"{config.output_stem}_loss_curves.png"
    throughput_path = config.output_dir / f"{config.output_stem}_throughput.png"
    dynamics_path = config.output_dir / f"{config.output_stem}_training_dynamics.png"
    component_path = config.output_dir / f"{config.output_stem}_component_diagnostics.png"
    rope_path = config.output_dir / f"{config.output_stem}_rope_diagnostics.png"

    payload = {
        "config": config.to_dict(),
        "environment": environment,
        "dataset_summary": dataset_summary,
        "variants": [to_serializable(asdict(variant)) for variant in variants],
        "artifact_paths": {
            "metrics": str(metrics_path),
            "summary": str(summary_path),
            "loss_curves": str(loss_curve_path),
            "throughput": str(throughput_path),
            "training_dynamics": str(dynamics_path),
            "component_diagnostics": str(component_path),
            "rope_diagnostics": str(rope_path),
        },
        "results": results,
    }
    metrics_path.write_text(json.dumps(to_serializable(payload), indent=2), encoding="utf-8")
    write_summary_csv(results, summary_path)
    plot_results(
        results,
        loss_curve_path=loss_curve_path,
        throughput_path=throughput_path,
        dynamics_path=dynamics_path,
        component_path=component_path,
        rope_path=rope_path,
    )
    print_summary(results)

    LOGGER.info("Wrote metrics to %s", metrics_path)
    LOGGER.info("Wrote summary to %s", summary_path)
    LOGGER.info("Wrote loss curves to %s", loss_curve_path)
    LOGGER.info("Wrote throughput plot to %s", throughput_path)
    LOGGER.info("Wrote training dynamics plot to %s", dynamics_path)
    LOGGER.info("Wrote component diagnostics plot to %s", component_path)
    LOGGER.info("Wrote RoPE diagnostics plot to %s", rope_path)


if __name__ == "__main__":
    main()
