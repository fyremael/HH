from __future__ import annotations

import argparse
import json
import logging
import math
import platform
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from householder_rope import BlockDiagonalRoPECore, HouseholderRoPE, HouseholderRoPEConfig  # noqa: E402
from householder_rope.diagnostics import (  # noqa: E402
    attention_logit_path_error as torch_attention_logit_path_error,
    orthogonality_defect as torch_orthogonality_defect,
    relativity_defect as torch_relativity_defect,
    reversibility_defect as torch_reversibility_defect,
)
from householder_rope.jax_core import (  # noqa: E402
    JaxBlockDiagonalRoPECore,
    JaxHouseholderRoPE,
    JaxHouseholderRoPEConfig,
)
from householder_rope.jax_diagnostics import (  # noqa: E402
    attention_logit_path_error as jax_attention_logit_path_error,
    orthogonality_defect as jax_orthogonality_defect,
    relativity_defect as jax_relativity_defect,
    reversibility_defect as jax_reversibility_defect,
)

try:  # noqa: E402
    from householder_rope.flax_modules import FlaxHouseholderRoPE
except Exception as exc:  # pragma: no cover - optional dependency path.
    FlaxHouseholderRoPE = None
    FLAX_IMPORT_ERROR = str(exc)
else:
    FLAX_IMPORT_ERROR = None


LOGGER = logging.getLogger("backend_compare")
SCENARIO_SETS: dict[str, list[dict[str, Any]]] = {
    "default": [
        {
            "name": "per_head_1d_small",
            "mode": "per_head",
            "group_size": 2,
            "batch": 2,
            "num_heads": 4,
            "tokens": 32,
            "head_dim": 32,
            "num_reflectors": 4,
            "rope_ndim": 1,
            "seed": 101,
        },
        {
            "name": "group_shared_2d_medium",
            "mode": "group_shared",
            "group_size": 2,
            "batch": 2,
            "num_heads": 4,
            "tokens": 64,
            "head_dim": 64,
            "num_reflectors": 8,
            "rope_ndim": 2,
            "seed": 202,
        },
    ],
    "large_gpu": [
        {
            "name": "per_head_1d_large",
            "mode": "per_head",
            "group_size": 2,
            "batch": 4,
            "num_heads": 8,
            "tokens": 256,
            "head_dim": 64,
            "num_reflectors": 8,
            "rope_ndim": 1,
            "seed": 303,
        },
        {
            "name": "group_shared_2d_large",
            "mode": "group_shared",
            "group_size": 2,
            "batch": 2,
            "num_heads": 8,
            "tokens": 512,
            "head_dim": 128,
            "num_reflectors": 16,
            "rope_ndim": 2,
            "seed": 404,
        },
    ],
    "long_context_gpu": [
        {
            "name": "per_head_1d_long_context",
            "mode": "per_head",
            "group_size": 2,
            "batch": 4,
            "num_heads": 8,
            "tokens": 1024,
            "head_dim": 64,
            "num_reflectors": 8,
            "rope_ndim": 1,
            "seed": 505,
        },
        {
            "name": "group_shared_2d_long_context",
            "mode": "group_shared",
            "group_size": 2,
            "batch": 2,
            "num_heads": 8,
            "tokens": 768,
            "head_dim": 128,
            "num_reflectors": 16,
            "rope_ndim": 2,
            "seed": 606,
        },
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare PyTorch, pure-JAX, and Flax Householder-RoPE backends.")
    parser.add_argument("--repeats", type=int, default=20, help="Benchmark repetitions per backend.")
    parser.add_argument("--warmup", type=int, default=5, help="Warmup iterations per backend.")
    parser.add_argument(
        "--scenario-set",
        default="default",
        choices=["default", "large_gpu", "long_context_gpu", "all"],
        help="Which benchmark scenario preset to run.",
    )
    parser.add_argument(
        "--benchmark-mode",
        default="all",
        choices=["all", "gpu"],
        help="Whether to record all runtime paths or only GPU-focused timings.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts" / "backend_comparison.json",
        help="Where to write the comparison JSON artifact.",
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
    )


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


def resolve_scenarios(name: str) -> list[dict[str, Any]]:
    if name == "all":
        return [scenario for scenarios in SCENARIO_SETS.values() for scenario in scenarios]
    return SCENARIO_SETS[name]


def collect_environment_metadata() -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "numpy_version": np.__version__,
        "torch_version": torch.__version__,
        "jax_version": jax.__version__,
        "jax_backend": jax.default_backend(),
        "jax_devices": [str(device) for device in jax.devices()],
        "torch_cuda_available": torch.cuda.is_available(),
        "flax_available": FlaxHouseholderRoPE is not None,
    }
    if torch.cuda.is_available():
        metadata["torch_cuda_device_name"] = torch.cuda.get_device_name(0)
    if FLAX_IMPORT_ERROR is not None:
        metadata["flax_import_error"] = FLAX_IMPORT_ERROR
    return metadata


def build_frequency_matrix_np(dim: int, ndim: int, base: float = 10000.0) -> np.ndarray:
    if dim % 2 != 0:
        raise ValueError(f"dim must be even, received {dim}.")
    num_pairs = dim // 2
    frequency_matrix = np.zeros((num_pairs, ndim), dtype=np.float32)
    base_block, remainder = divmod(num_pairs, ndim)
    start = 0
    for axis in range(ndim):
        block_size = base_block + int(axis < remainder)
        stop = start + block_size
        if stop > start:
            local_index = np.arange(stop - start, dtype=np.float32)
            frequency_matrix[start:stop, axis] = base ** (-local_index / max(stop - start, 1))
        start = stop
    return frequency_matrix


def build_positions_np(tokens: int, ndim: int) -> np.ndarray:
    if ndim == 1:
        return np.arange(tokens, dtype=np.float32)
    side = int(math.ceil(tokens ** (1.0 / ndim)))
    shape = (side,) * ndim
    coords = np.stack(np.unravel_index(np.arange(np.prod(shape)), shape), axis=-1)
    return coords[:tokens].astype(np.float32)


def reflector_shape(config: dict[str, Any]) -> tuple[int, ...]:
    if config["mode"] == "shared":
        return (config["num_reflectors"], config["head_dim"])
    if config["mode"] == "per_head":
        return (config["num_heads"], config["num_reflectors"], config["head_dim"])
    return (config["num_heads"] // config["group_size"], config["num_reflectors"], config["head_dim"])


def max_abs_diff(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.max(np.abs(left - right)))


def rms_diff(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(left - right))))


def benchmark_torch(
    rope: HouseholderRoPE,
    q: torch.Tensor,
    k: torch.Tensor,
    pos: torch.Tensor,
    *,
    repeats: int,
    warmup: int,
) -> dict[str, float]:
    times: list[float] = []
    device_type = q.device.type
    with torch.no_grad():
        for _ in range(warmup):
            rope(q, k, pos)
            if device_type == "cuda":
                torch.cuda.synchronize(q.device)
        for _ in range(repeats):
            start = time.perf_counter()
            rope(q, k, pos)
            if device_type == "cuda":
                torch.cuda.synchronize(q.device)
            times.append((time.perf_counter() - start) * 1000.0)
    return {
        "mean_ms": float(statistics.mean(times)),
        "std_ms": float(statistics.pstdev(times) if len(times) > 1 else 0.0),
    }


def benchmark_jax(fn, q: jax.Array, k: jax.Array, *, repeats: int, warmup: int) -> dict[str, float]:
    compiled = jax.jit(fn)
    warm = compiled(q, k)
    warm[0].block_until_ready()
    for _ in range(max(warmup - 1, 0)):
        compiled(q, k)[0].block_until_ready()
    times: list[float] = []
    for _ in range(repeats):
        start = time.perf_counter()
        compiled(q, k)[0].block_until_ready()
        times.append((time.perf_counter() - start) * 1000.0)
    return {
        "mean_ms": float(statistics.mean(times)),
        "std_ms": float(statistics.pstdev(times) if len(times) > 1 else 0.0),
    }


def jax_runtime_label(prefix: str) -> str:
    return f"{prefix}_{jax.default_backend()}_jit"


def summarize_tensor_metric(value: Any) -> dict[str, float]:
    if isinstance(value, torch.Tensor):
        array = value.detach().cpu().numpy()
    else:
        array = np.asarray(value)
    return {
        "max": float(np.max(array)),
        "mean": float(np.mean(array)),
    }


def run_scenario(
    config: dict[str, Any],
    *,
    repeats: int,
    warmup: int,
    benchmark_mode: str,
    environment: dict[str, Any],
) -> dict[str, Any]:
    LOGGER.info("Running scenario %s", config["name"])
    rng = np.random.default_rng(config["seed"])

    batch = config["batch"]
    heads = config["num_heads"]
    tokens = config["tokens"]
    dim = config["head_dim"]
    ndim = config["rope_ndim"]

    q_np = rng.standard_normal((batch, heads, tokens, dim), dtype=np.float32)
    k_np = rng.standard_normal((batch, heads, tokens, dim), dtype=np.float32)
    pos_np = build_positions_np(tokens, ndim)
    pos_offset_np = pos_np + 1.0
    reflectors_np = rng.standard_normal(reflector_shape(config), dtype=np.float32)
    frequency_matrix_np = build_frequency_matrix_np(dim, ndim)

    torch_config = HouseholderRoPEConfig(
        mode=config["mode"],
        group_size=config["group_size"],
        num_reflectors=config["num_reflectors"],
        init="random",
        rope_ndim=ndim,
        enforce_SO=False,
    )
    torch_core = BlockDiagonalRoPECore(dim=dim, ndim=ndim, frequency_matrix=torch.from_numpy(frequency_matrix_np))
    torch_rope = HouseholderRoPE(num_heads=heads, head_dim=dim, config=torch_config, rope_core=torch_core)
    with torch.no_grad():
        torch_rope.reflectors.copy_(torch.from_numpy(reflectors_np))
    torch_rope.eval()

    q_torch = torch.from_numpy(q_np)
    k_torch = torch.from_numpy(k_np)
    pos_torch = torch.from_numpy(pos_np)
    pos_offset_torch = torch.from_numpy(pos_offset_np)
    q_torch_out, k_torch_out = torch_rope(q_torch, k_torch, pos_torch)

    jax_config = JaxHouseholderRoPEConfig(
        mode=config["mode"],
        group_size=config["group_size"],
        num_reflectors=config["num_reflectors"],
        init="random",
        rope_ndim=ndim,
        enforce_SO=False,
    )
    jax_core = JaxBlockDiagonalRoPECore(dim=dim, ndim=ndim, frequency_matrix=jnp.asarray(frequency_matrix_np))
    jax_rope = JaxHouseholderRoPE(
        num_heads=heads,
        head_dim=dim,
        config=jax_config,
        rope_core=jax_core,
        reflectors=jnp.asarray(reflectors_np),
    )

    q_jax = jnp.asarray(q_np)
    k_jax = jnp.asarray(k_np)
    pos_jax = jnp.asarray(pos_np)
    pos_offset_jax = jnp.asarray(pos_offset_np)
    q_jax_out, k_jax_out = jax_rope(q_jax, k_jax, pos_jax)

    q_torch_np = q_torch_out.detach().cpu().numpy()
    k_torch_np = k_torch_out.detach().cpu().numpy()
    q_jax_np = np.asarray(q_jax_out)
    k_jax_np = np.asarray(k_jax_out)

    torch_logits = torch.einsum("bhti,bhsi->bhts", q_torch_out, k_torch_out).detach().cpu().numpy()
    jax_logits = np.asarray(jnp.einsum("bhti,bhsi->bhts", q_jax_out, k_jax_out))

    jax_label = jax_runtime_label("jax")
    runtime_ms: dict[str, Any] = {}
    if benchmark_mode == "all":
        runtime_ms["torch_cpu"] = benchmark_torch(
            torch_rope,
            q_torch,
            k_torch,
            pos_torch,
            repeats=repeats,
            warmup=warmup,
        )
    runtime_ms[jax_label] = benchmark_jax(
        lambda q, k: jax_rope(q, k, pos_jax),
        q_jax,
        k_jax,
        repeats=repeats,
        warmup=warmup,
    )

    results: dict[str, Any] = {
        "scenario": config,
        "environment": environment,
        "numerical_agreement": {
            "torch_vs_jax": {
                "q_max_abs_diff": max_abs_diff(q_torch_np, q_jax_np),
                "k_max_abs_diff": max_abs_diff(k_torch_np, k_jax_np),
                "q_rms_diff": rms_diff(q_torch_np, q_jax_np),
                "k_rms_diff": rms_diff(k_torch_np, k_jax_np),
                "logit_max_abs_diff": max_abs_diff(torch_logits, jax_logits),
                "logit_rms_diff": rms_diff(torch_logits, jax_logits),
            },
        },
        "diagnostics": {
            "torch": {
                "orthogonality_defect": summarize_tensor_metric(torch_orthogonality_defect(torch_rope.materialize_Q(expand_heads=False))),
                "relativity_defect": summarize_tensor_metric(torch_relativity_defect(torch_rope, pos_torch, pos_offset_torch)),
                "reversibility_defect": summarize_tensor_metric(torch_reversibility_defect(torch_rope, pos_torch)),
                "attention_logit_path_error": summarize_tensor_metric(torch_attention_logit_path_error(q_torch, k_torch, pos_torch, torch_rope)),
            },
            "jax": {
                "orthogonality_defect": summarize_tensor_metric(jax_orthogonality_defect(jax_rope.materialize_Q(expand_heads=False))),
                "relativity_defect": summarize_tensor_metric(jax_relativity_defect(jax_rope, pos_jax, pos_offset_jax)),
                "reversibility_defect": summarize_tensor_metric(jax_reversibility_defect(jax_rope, pos_jax)),
                "attention_logit_path_error": summarize_tensor_metric(jax_attention_logit_path_error(q_jax, k_jax, pos_jax, jax_rope)),
            },
        },
        "runtime_ms": runtime_ms,
    }

    if FlaxHouseholderRoPE is None:
        results["environment"]["flax_import_error"] = FLAX_IMPORT_ERROR
    else:
        flax_module = FlaxHouseholderRoPE(
            num_heads=heads,
            head_dim=dim,
            config=jax_config,
            rope_core=jax_core,
        )
        flax_variables = {"params": {"reflectors": jnp.asarray(reflectors_np)}}
        q_flax_out, k_flax_out = flax_module.apply(flax_variables, q_jax, k_jax, pos_jax)
        q_flax_np = np.asarray(q_flax_out)
        k_flax_np = np.asarray(k_flax_out)
        flax_logits = np.asarray(jnp.einsum("bhti,bhsi->bhts", q_flax_out, k_flax_out))
        results["numerical_agreement"]["jax_vs_flax"] = {
            "q_max_abs_diff": max_abs_diff(q_jax_np, q_flax_np),
            "k_max_abs_diff": max_abs_diff(k_jax_np, k_flax_np),
            "q_rms_diff": rms_diff(q_jax_np, q_flax_np),
            "k_rms_diff": rms_diff(k_jax_np, k_flax_np),
            "logit_max_abs_diff": max_abs_diff(jax_logits, flax_logits),
            "logit_rms_diff": rms_diff(jax_logits, flax_logits),
        }
        runtime_ms[jax_runtime_label("flax")] = benchmark_jax(
            lambda q, k: flax_module.apply(flax_variables, q, k, pos_jax),
            q_jax,
            k_jax,
            repeats=repeats,
            warmup=warmup,
        )

    if torch.cuda.is_available():
        torch_core_cuda = BlockDiagonalRoPECore(
            dim=dim,
            ndim=ndim,
            frequency_matrix=torch.from_numpy(frequency_matrix_np).cuda(),
        )
        torch_rope_cuda = HouseholderRoPE(
            num_heads=heads,
            head_dim=dim,
            config=torch_config,
            rope_core=torch_core_cuda,
        ).cuda()
        with torch.no_grad():
            torch_rope_cuda.reflectors.copy_(torch.from_numpy(reflectors_np).cuda())
        torch_rope_cuda.eval()
        q_cuda = torch.from_numpy(q_np).cuda()
        k_cuda = torch.from_numpy(k_np).cuda()
        pos_cuda = torch.from_numpy(pos_np).cuda()
        runtime_ms["torch_cuda"] = benchmark_torch(
            torch_rope_cuda,
            q_cuda,
            k_cuda,
            pos_cuda,
            repeats=repeats,
            warmup=warmup,
        )
    elif benchmark_mode == "gpu":
        raise RuntimeError("benchmark_mode='gpu' requires torch.cuda.is_available().")

    return results


def print_summary(results: list[dict[str, Any]]) -> None:
    for result in results:
        name = result["scenario"]["name"]
        torch_vs_jax = result["numerical_agreement"]["torch_vs_jax"]
        jax_vs_flax = result["numerical_agreement"].get("jax_vs_flax")
        runtimes = result["runtime_ms"]
        print(f"[{name}]")
        print(
            "  torch vs jax: "
            f"q_max={torch_vs_jax['q_max_abs_diff']:.3e}, "
            f"k_max={torch_vs_jax['k_max_abs_diff']:.3e}, "
            f"logit_max={torch_vs_jax['logit_max_abs_diff']:.3e}"
        )
        if jax_vs_flax is not None:
            print(
                "  jax vs flax: "
                f"q_max={jax_vs_flax['q_max_abs_diff']:.3e}, "
                f"k_max={jax_vs_flax['k_max_abs_diff']:.3e}, "
                f"logit_max={jax_vs_flax['logit_max_abs_diff']:.3e}"
            )
        elif "flax_import_error" in result["environment"]:
            print(f"  jax vs flax: skipped ({result['environment']['flax_import_error']})")

        runtime_parts = [f"{key}={value['mean_ms']:.3f}" for key, value in runtimes.items()]
        print("  runtime ms: " + ", ".join(runtime_parts))


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)

    environment = collect_environment_metadata()
    if args.benchmark_mode == "gpu" and environment["jax_backend"] != "gpu":
        raise RuntimeError("benchmark_mode='gpu' requires JAX to use a GPU backend.")

    scenarios = resolve_scenarios(args.scenario_set)
    results = [
        run_scenario(
            config,
            repeats=args.repeats,
            warmup=args.warmup,
            benchmark_mode=args.benchmark_mode,
            environment=dict(environment),
        )
        for config in scenarios
    ]
    payload = {
        "metadata": {
            "scenario_set": args.scenario_set,
            "benchmark_mode": args.benchmark_mode,
            "repeats": args.repeats,
            "warmup": args.warmup,
            "environment": environment,
        },
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(to_serializable(payload), indent=2), encoding="utf-8")
    LOGGER.info("Wrote backend comparison artifact to %s", args.output)
    print_summary(results)


if __name__ == "__main__":
    main()



