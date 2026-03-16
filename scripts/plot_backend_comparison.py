from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"

RUNTIME_BACKENDS = ["jax_gpu_jit", "flax_gpu_jit", "torch_cuda"]
RUNTIME_COLORS = {
    "jax_gpu_jit": "#1f77b4",
    "flax_gpu_jit": "#ff7f0e",
    "torch_cuda": "#2ca02c",
}
AGREEMENT_SERIES = [
    ("torch_vs_jax", "logit_max_abs_diff", "Torch vs JAX logit max", "#d62728"),
    ("torch_vs_jax", "q_max_abs_diff", "Torch vs JAX q max", "#9467bd"),
    ("jax_vs_flax", "logit_max_abs_diff", "JAX vs Flax logit max", "#8c564b"),
]
DIAGNOSTIC_SERIES = [
    ("torch", "relativity_defect", "Torch relativity max", "#17becf"),
    ("jax", "relativity_defect", "JAX relativity max", "#bcbd22"),
    ("torch", "attention_logit_path_error", "Torch path max", "#7f7f7f"),
    ("jax", "attention_logit_path_error", "JAX path max", "#e377c2"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot backend comparison artifacts as runtime and stability figures.")
    parser.add_argument(
        "--inputs",
        nargs="+",
        type=Path,
        default=[
            ARTIFACTS / "backend_comparison_wsl_uv_pinned_gpu.json",
            ARTIFACTS / "backend_comparison_wsl_uv_long_context_gpu.json",
        ],
        help="Comparison JSON artifacts to visualize.",
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=ARTIFACTS / "backend_comparison_sequence_sweep",
        help="Prefix for the generated PNG files.",
    )
    return parser.parse_args()


def load_payload(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return {
            "metadata": {
                "scenario_set": path.stem,
                "environment": {},
            },
            "results": payload,
        }
    return payload


def family_key(result: dict[str, Any]) -> str:
    scenario = result["scenario"]
    return (
        f"{scenario['mode']}|ndim={scenario['rope_ndim']}|batch={scenario['batch']}|"
        f"heads={scenario['num_heads']}|dim={scenario['head_dim']}|reflectors={scenario['num_reflectors']}"
    )


def family_title(result: dict[str, Any]) -> str:
    scenario = result["scenario"]
    mode = scenario["mode"].replace("_", " ")
    return (
        f"{mode}, {scenario['rope_ndim']}D, B={scenario['batch']}, H={scenario['num_heads']}, "
        f"D={scenario['head_dim']}, M={scenario['num_reflectors']}"
    )


def collect_records(paths: list[Path]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for path in paths:
        payload = load_payload(path)
        label = payload.get("metadata", {}).get("scenario_set", path.stem)
        for result in payload["results"]:
            record = {
                "artifact_label": label,
                "path": path,
                "result": result,
            }
            grouped.setdefault(family_key(result), []).append(record)
    for records in grouped.values():
        records.sort(key=lambda item: item["result"]["scenario"]["tokens"])
    return grouped


def plot_runtime(grouped: dict[str, list[dict[str, Any]]], output_path: Path) -> None:
    families = list(grouped.items())
    fig, axes = plt.subplots(1, len(families), figsize=(7 * len(families), 5), squeeze=False)
    for axis, (_, records) in zip(axes[0], families):
        title = family_title(records[0]["result"])
        tokens = [item["result"]["scenario"]["tokens"] for item in records]
        for backend in RUNTIME_BACKENDS:
            token_points = []
            values = []
            for item in records:
                runtime = item["result"]["runtime_ms"].get(backend)
                if runtime is None:
                    continue
                token_points.append(item["result"]["scenario"]["tokens"])
                values.append(runtime["mean_ms"])
            if values:
                axis.plot(
                    token_points,
                    values,
                    marker="o",
                    linewidth=2.0,
                    label=backend,
                    color=RUNTIME_COLORS[backend],
                )
        axis.set_title(title)
        axis.set_xlabel("Sequence length")
        axis.set_ylabel("Runtime (ms)")
        axis.set_yscale("log")
        axis.set_xticks(tokens)
        axis.grid(True, linestyle=":", alpha=0.35)
        axis.legend(frameon=False)
    fig.suptitle("Householder-RoPE Backend Runtime Scaling", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_stability(grouped: dict[str, list[dict[str, Any]]], output_path: Path) -> None:
    families = list(grouped.items())
    fig, axes = plt.subplots(2, len(families), figsize=(7 * len(families), 9), squeeze=False)
    for column, (_, records) in enumerate(families):
        title = family_title(records[0]["result"])
        tokens = [item["result"]["scenario"]["tokens"] for item in records]

        axis_agreement = axes[0, column]
        for comparison_key, metric_key, label, color in AGREEMENT_SERIES:
            token_points = []
            values = []
            for item in records:
                comparison = item["result"]["numerical_agreement"].get(comparison_key)
                if comparison is None:
                    continue
                token_points.append(item["result"]["scenario"]["tokens"])
                values.append(comparison[metric_key])
            if values:
                axis_agreement.plot(token_points, values, marker="o", linewidth=2.0, label=label, color=color)
        axis_agreement.set_title(f"Agreement: {title}")
        axis_agreement.set_xlabel("Sequence length")
        axis_agreement.set_ylabel("Max absolute diff")
        axis_agreement.set_yscale("log")
        axis_agreement.set_xticks(tokens)
        axis_agreement.grid(True, linestyle=":", alpha=0.35)
        axis_agreement.legend(frameon=False)

        axis_diag = axes[1, column]
        for backend_key, metric_key, label, color in DIAGNOSTIC_SERIES:
            token_points = []
            values = []
            for item in records:
                metric = item["result"]["diagnostics"][backend_key].get(metric_key)
                if metric is None:
                    continue
                token_points.append(item["result"]["scenario"]["tokens"])
                values.append(metric["max"])
            if values:
                axis_diag.plot(token_points, values, marker="o", linewidth=2.0, label=label, color=color)
        axis_diag.set_title(f"Stability: {title}")
        axis_diag.set_xlabel("Sequence length")
        axis_diag.set_ylabel("Relative error")
        axis_diag.set_yscale("log")
        axis_diag.set_xticks(tokens)
        axis_diag.grid(True, linestyle=":", alpha=0.35)
        axis_diag.legend(frameon=False)
    fig.suptitle("Householder-RoPE Agreement and Stability vs Sequence Length", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    grouped = collect_records(args.inputs)
    if not grouped:
        raise ValueError("No comparison results found in the provided artifacts.")

    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    runtime_path = args.output_prefix.with_name(args.output_prefix.name + "_runtime.png")
    stability_path = args.output_prefix.with_name(args.output_prefix.name + "_stability.png")
    plot_runtime(grouped, runtime_path)
    plot_stability(grouped, stability_path)
    print(runtime_path)
    print(stability_path)


if __name__ == "__main__":
    main()
