from __future__ import annotations

import argparse
import json
import logging
import platform
import statistics
import sys
import time
from copy import deepcopy
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

from householder_rope.attention_benchmark import (  # noqa: E402
    AttentionBenchmarkScenario,
    build_attention_case,
    build_flax_attention_block,
    build_jax_context,
    build_torch_attention_block,
    flax_variables_from_case,
    jax_attention_block_forward,
    jax_sgd_step,
    make_flax_loss_fn,
    make_jax_loss_fn,
    torch_single_step_losses,
)


LOGGER = logging.getLogger("attention_block_benchmark")
SCENARIO_SETS: dict[str, list[AttentionBenchmarkScenario]] = {
    "default": [
        AttentionBenchmarkScenario(
            name="per_head_1d_block_small",
            mode="per_head",
            group_size=2,
            batch=2,
            num_heads=4,
            tokens=32,
            head_dim=32,
            num_reflectors=4,
            rope_ndim=1,
            seed=707,
        ),
        AttentionBenchmarkScenario(
            name="group_shared_2d_block_small",
            mode="group_shared",
            group_size=2,
            batch=2,
            num_heads=4,
            tokens=64,
            head_dim=64,
            num_reflectors=8,
            rope_ndim=2,
            seed=808,
        ),
    ],
    "large_gpu": [
        AttentionBenchmarkScenario(
            name="per_head_1d_block_large",
            mode="per_head",
            group_size=2,
            batch=4,
            num_heads=8,
            tokens=256,
            head_dim=64,
            num_reflectors=8,
            rope_ndim=1,
            seed=909,
        ),
        AttentionBenchmarkScenario(
            name="group_shared_2d_block_large",
            mode="group_shared",
            group_size=2,
            batch=2,
            num_heads=8,
            tokens=512,
            head_dim=128,
            num_reflectors=16,
            rope_ndim=2,
            seed=1001,
        ),
    ],
    "long_context_gpu": [
        AttentionBenchmarkScenario(
            name="per_head_1d_block_long_context",
            mode="per_head",
            group_size=2,
            batch=4,
            num_heads=8,
            tokens=1024,
            head_dim=64,
            num_reflectors=8,
            rope_ndim=1,
            seed=1101,
        ),
        AttentionBenchmarkScenario(
            name="group_shared_2d_block_long_context",
            mode="group_shared",
            group_size=2,
            batch=2,
            num_heads=8,
            tokens=768,
            head_dim=128,
            num_reflectors=16,
            rope_ndim=2,
            seed=1202,
        ),
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark aligned full attention blocks and single-step training updates.")
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
    parser.add_argument("--forward-repeats", type=int, default=20, help="Forward-pass benchmark repetitions.")
    parser.add_argument("--forward-warmup", type=int, default=5, help="Forward-pass warmup iterations.")
    parser.add_argument("--train-repeats", type=int, default=10, help="Training-step benchmark repetitions.")
    parser.add_argument("--train-warmup", type=int, default=3, help="Training-step warmup iterations.")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts" / "attention_block_benchmark.json",
        help="Where to write the benchmark JSON artifact.",
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


def resolve_scenarios(name: str) -> list[AttentionBenchmarkScenario]:
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
    }
    if torch.cuda.is_available():
        metadata["torch_cuda_device_name"] = torch.cuda.get_device_name(0)
    try:
        import flax  # noqa: PLC0415

        metadata["flax_version"] = flax.__version__
    except Exception as exc:  # pragma: no cover - environment dependent.
        metadata["flax_import_error"] = str(exc)
    return metadata


def max_abs_diff(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.max(np.abs(left - right)))


def rms_diff(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(left - right))))


def block_until_ready(value: Any) -> Any:
    leaves = jax.tree_util.tree_leaves(value)
    for leaf in leaves:
        if hasattr(leaf, "block_until_ready"):
            leaf.block_until_ready()
    return value


def benchmark_torch_forward(
    module: torch.nn.Module,
    x: torch.Tensor,
    pos: torch.Tensor,
    *,
    repeats: int,
    warmup: int,
) -> dict[str, float]:
    times: list[float] = []
    module.eval()
    device_type = x.device.type
    with torch.no_grad():
        for _ in range(warmup):
            module(x, pos)
            if device_type == "cuda":
                torch.cuda.synchronize(x.device)
        for _ in range(repeats):
            start = time.perf_counter()
            module(x, pos)
            if device_type == "cuda":
                torch.cuda.synchronize(x.device)
            times.append((time.perf_counter() - start) * 1000.0)
    return {
        "mean_ms": float(statistics.mean(times)),
        "std_ms": float(statistics.pstdev(times) if len(times) > 1 else 0.0),
    }


def benchmark_torch_train_step(
    module: torch.nn.Module,
    x: torch.Tensor,
    pos: torch.Tensor,
    target: torch.Tensor,
    *,
    lr: float,
    repeats: int,
    warmup: int,
) -> dict[str, float]:
    def step() -> None:
        module.train()
        for parameter in module.parameters():
            parameter.grad = None
        output = module(x, pos)
        loss = torch.nn.functional.mse_loss(output, target)
        loss.backward()
        with torch.no_grad():
            for parameter in module.parameters():
                parameter -= lr * parameter.grad

    times: list[float] = []
    device_type = x.device.type
    for _ in range(warmup):
        step()
        if device_type == "cuda":
            torch.cuda.synchronize(x.device)
    for _ in range(repeats):
        start = time.perf_counter()
        step()
        if device_type == "cuda":
            torch.cuda.synchronize(x.device)
        times.append((time.perf_counter() - start) * 1000.0)
    return {
        "mean_ms": float(statistics.mean(times)),
        "std_ms": float(statistics.pstdev(times) if len(times) > 1 else 0.0),
    }


def benchmark_jax_forward(fn, state: Any, *, repeats: int, warmup: int) -> dict[str, float]:
    compiled = jax.jit(fn)
    block_until_ready(compiled(state))
    for _ in range(max(warmup - 1, 0)):
        block_until_ready(compiled(state))
    times: list[float] = []
    for _ in range(repeats):
        start = time.perf_counter()
        block_until_ready(compiled(state))
        times.append((time.perf_counter() - start) * 1000.0)
    return {
        "mean_ms": float(statistics.mean(times)),
        "std_ms": float(statistics.pstdev(times) if len(times) > 1 else 0.0),
    }


def benchmark_jax_train_step(fn, state: Any, *, repeats: int, warmup: int) -> dict[str, float]:
    compiled = jax.jit(fn)
    current_state = state
    for _ in range(warmup):
        current_state, loss = compiled(current_state)
        block_until_ready(loss)
    times: list[float] = []
    for _ in range(repeats):
        start = time.perf_counter()
        current_state, loss = compiled(current_state)
        block_until_ready(loss)
        times.append((time.perf_counter() - start) * 1000.0)
    return {
        "mean_ms": float(statistics.mean(times)),
        "std_ms": float(statistics.pstdev(times) if len(times) > 1 else 0.0),
    }


def jax_runtime_label(prefix: str) -> str:
    return f"{prefix}_{jax.default_backend()}_jit"


def run_scenario(
    scenario: AttentionBenchmarkScenario,
    *,
    benchmark_mode: str,
    forward_repeats: int,
    forward_warmup: int,
    train_repeats: int,
    train_warmup: int,
    environment: dict[str, Any],
) -> dict[str, Any]:
    LOGGER.info("Running attention benchmark scenario %s", scenario.name)
    case = build_attention_case(scenario)
    x_jax = jnp.asarray(case.x)
    pos_jax = jnp.asarray(case.pos)
    target_jax = jnp.asarray(case.target)

    jax_params, rope_core, rope_config, head_to_group = build_jax_context(case)
    jax_forward = lambda params: jax_attention_block_forward(
        params,
        x_jax,
        pos_jax,
        num_heads=scenario.num_heads,
        head_dim=scenario.head_dim,
        rope_core=rope_core,
        rope_config=rope_config,
        head_to_group=head_to_group,
    )
    jax_output = jax_forward(jax_params)

    flax_module = build_flax_attention_block(case)
    flax_params = flax_variables_from_case(case)["params"]
    flax_output = flax_module.apply({"params": flax_params}, x_jax, pos_jax, deterministic=True)

    compare_device = torch.device("cuda" if benchmark_mode == "gpu" and torch.cuda.is_available() else "cpu")
    torch_module, x_torch, pos_torch, target_torch = build_torch_attention_block(case, device=compare_device)
    torch_module.eval()
    with torch.no_grad():
        torch_output = torch_module(x_torch, pos_torch)

    torch_output_np = torch_output.detach().cpu().numpy()
    jax_output_np = np.asarray(jax_output)
    flax_output_np = np.asarray(flax_output)

    jax_loss_fn = make_jax_loss_fn(
        num_heads=scenario.num_heads,
        head_dim=scenario.head_dim,
        rope_core=rope_core,
        rope_config=rope_config,
        head_to_group=head_to_group,
    )
    flax_loss_fn = make_flax_loss_fn(flax_module)

    torch_loss_before, torch_loss_after = torch_single_step_losses(
        deepcopy(torch_module),
        x_torch,
        pos_torch,
        target_torch,
        lr=scenario.learning_rate,
    )
    jax_updated_params, jax_loss_before = jax_sgd_step(
        jax_params,
        x_jax,
        pos_jax,
        target_jax,
        lr=scenario.learning_rate,
        loss_fn=jax_loss_fn,
    )
    jax_loss_after = float(jax_loss_fn(jax_updated_params, x_jax, pos_jax, target_jax))
    flax_updated_params, flax_loss_before = jax_sgd_step(
        flax_params,
        x_jax,
        pos_jax,
        target_jax,
        lr=scenario.learning_rate,
        loss_fn=flax_loss_fn,
    )
    flax_loss_after = float(flax_loss_fn(flax_updated_params, x_jax, pos_jax, target_jax))

    runtime_ms: dict[str, Any] = {}
    if benchmark_mode == "all":
        torch_cpu_module, x_cpu, pos_cpu, target_cpu = build_torch_attention_block(case, device=torch.device("cpu"))
        runtime_ms["torch_forward_cpu"] = benchmark_torch_forward(
            torch_cpu_module,
            x_cpu,
            pos_cpu,
            repeats=forward_repeats,
            warmup=forward_warmup,
        )
        runtime_ms["torch_train_step_cpu"] = benchmark_torch_train_step(
            deepcopy(torch_cpu_module),
            x_cpu,
            pos_cpu,
            target_cpu,
            lr=scenario.learning_rate,
            repeats=train_repeats,
            warmup=train_warmup,
        )

    jax_forward_label = jax_runtime_label("jax_forward")
    jax_train_label = jax_runtime_label("jax_train_step")
    flax_forward_label = jax_runtime_label("flax_forward")
    flax_train_label = jax_runtime_label("flax_train_step")

    runtime_ms[jax_forward_label] = benchmark_jax_forward(jax_forward, jax_params, repeats=forward_repeats, warmup=forward_warmup)
    runtime_ms[jax_train_label] = benchmark_jax_train_step(
        lambda params: jax_sgd_step(
            params,
            x_jax,
            pos_jax,
            target_jax,
            lr=scenario.learning_rate,
            loss_fn=jax_loss_fn,
        ),
        jax_params,
        repeats=train_repeats,
        warmup=train_warmup,
    )
    runtime_ms[flax_forward_label] = benchmark_jax_forward(
        lambda params: flax_module.apply({"params": params}, x_jax, pos_jax, deterministic=True),
        flax_params,
        repeats=forward_repeats,
        warmup=forward_warmup,
    )
    runtime_ms[flax_train_label] = benchmark_jax_train_step(
        lambda params: jax_sgd_step(
            params,
            x_jax,
            pos_jax,
            target_jax,
            lr=scenario.learning_rate,
            loss_fn=flax_loss_fn,
        ),
        flax_params,
        repeats=train_repeats,
        warmup=train_warmup,
    )

    if torch.cuda.is_available():
        torch_cuda_module, x_cuda, pos_cuda, target_cuda = build_torch_attention_block(case, device=torch.device("cuda"))
        runtime_ms["torch_forward_cuda"] = benchmark_torch_forward(
            torch_cuda_module,
            x_cuda,
            pos_cuda,
            repeats=forward_repeats,
            warmup=forward_warmup,
        )
        runtime_ms["torch_train_step_cuda"] = benchmark_torch_train_step(
            deepcopy(torch_cuda_module),
            x_cuda,
            pos_cuda,
            target_cuda,
            lr=scenario.learning_rate,
            repeats=train_repeats,
            warmup=train_warmup,
        )
    elif benchmark_mode == "gpu":
        raise RuntimeError("benchmark_mode='gpu' requires torch.cuda.is_available().")

    return {
        "scenario": scenario.to_dict(),
        "environment": dict(environment),
        "forward_agreement": {
            "torch_vs_jax": {
                "output_max_abs_diff": max_abs_diff(torch_output_np, jax_output_np),
                "output_rms_diff": rms_diff(torch_output_np, jax_output_np),
            },
            "jax_vs_flax": {
                "output_max_abs_diff": max_abs_diff(jax_output_np, flax_output_np),
                "output_rms_diff": rms_diff(jax_output_np, flax_output_np),
            },
        },
        "train_step_agreement": {
            "losses": {
                "torch_initial_loss": torch_loss_before,
                "torch_post_step_loss": torch_loss_after,
                "jax_initial_loss": float(jax_loss_before),
                "jax_post_step_loss": jax_loss_after,
                "flax_initial_loss": float(flax_loss_before),
                "flax_post_step_loss": flax_loss_after,
            },
            "torch_vs_jax": {
                "initial_loss_abs_diff": abs(torch_loss_before - float(jax_loss_before)),
                "post_step_loss_abs_diff": abs(torch_loss_after - jax_loss_after),
            },
            "jax_vs_flax": {
                "initial_loss_abs_diff": abs(float(jax_loss_before) - float(flax_loss_before)),
                "post_step_loss_abs_diff": abs(jax_loss_after - flax_loss_after),
            },
        },
        "runtime_ms": runtime_ms,
    }


def print_summary(results: list[dict[str, Any]]) -> None:
    for result in results:
        scenario = result["scenario"]
        forward = result["forward_agreement"]
        train = result["train_step_agreement"]
        runtimes = result["runtime_ms"]
        print(f"[{scenario['name']}]")
        print(
            "  forward agreement: "
            f"torch_vs_jax_out_max={forward['torch_vs_jax']['output_max_abs_diff']:.3e}, "
            f"jax_vs_flax_out_max={forward['jax_vs_flax']['output_max_abs_diff']:.3e}"
        )
        print(
            "  train agreement: "
            f"torch_vs_jax_loss_post={train['torch_vs_jax']['post_step_loss_abs_diff']:.3e}, "
            f"jax_vs_flax_loss_post={train['jax_vs_flax']['post_step_loss_abs_diff']:.3e}"
        )
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
            scenario,
            benchmark_mode=args.benchmark_mode,
            forward_repeats=args.forward_repeats,
            forward_warmup=args.forward_warmup,
            train_repeats=args.train_repeats,
            train_warmup=args.train_warmup,
            environment=environment,
        )
        for scenario in scenarios
    ]
    payload = {
        "metadata": {
            "scenario_set": args.scenario_set,
            "benchmark_mode": args.benchmark_mode,
            "forward_repeats": args.forward_repeats,
            "forward_warmup": args.forward_warmup,
            "train_repeats": args.train_repeats,
            "train_warmup": args.train_warmup,
            "environment": environment,
        },
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(to_serializable(payload), indent=2), encoding="utf-8")
    LOGGER.info("Wrote attention benchmark artifact to %s", args.output)
    print_summary(results)


if __name__ == "__main__":
    main()

