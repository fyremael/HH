from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
FORWARD_COLORS = {
    "jax_forward": "#1f77b4",
    "flax_forward": "#ff7f0e",
    "torch_forward": "#2ca02c",
}
TRAIN_COLORS = {
    "jax_train_step": "#1f77b4",
    "flax_train_step": "#ff7f0e",
    "torch_train_step": "#2ca02c",
}
AGREEMENT_SERIES = [
    ("forward_agreement", "torch_vs_jax", "output_max_abs_diff", "Torch vs JAX output max", "#d62728"),
    ("forward_agreement", "jax_vs_flax", "output_max_abs_diff", "JAX vs Flax output max", "#9467bd"),
    ("train_step_agreement", "torch_vs_jax", "post_step_loss_abs_diff", "Torch vs JAX post-step loss", "#8c564b"),
    ("train_step_agreement", "jax_vs_flax", "post_step_loss_abs_diff", "JAX vs Flax post-step loss", "#e377c2"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot attention block benchmark artifacts.")
    parser.add_argument(
        "--inputs",
        nargs="+",
        type=Path,
        default=[
            ARTIFACTS / "attention_block_benchmark_wsl_pinned_large.json",
            ARTIFACTS / "attention_block_benchmark_wsl_pinned_long_context.json",
        ],
        help="Attention benchmark JSON artifacts to visualize.",
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=ARTIFACTS / "attention_block_sequence_sweep",
        help="Prefix for the generated PNG files.",
    )
    return parser.parse_args()


def load_payload(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


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




def safe_positive(value: float, eps: float = 1.0e-12) -> float:
    return max(float(value), eps)

def collect_records(paths: list[Path]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for path in paths:
        payload = load_payload(path)
        label = payload.get("metadata", {}).get("scenario_set", path.stem)
        for result in payload["results"]:
            grouped.setdefault(family_key(result), []).append({
                "artifact_label": label,
                "result": result,
            })
    for records in grouped.values():
        records.sort(key=lambda item: item["result"]["scenario"]["tokens"])
    return grouped


def find_runtime_key(runtime_ms: dict[str, Any], prefix: str) -> str | None:
    for key in runtime_ms:
        if key.startswith(prefix):
            return key
    return None


def plot_runtime(grouped: dict[str, list[dict[str, Any]]], output_path: Path) -> None:
    families = list(grouped.items())
    fig, axes = plt.subplots(2, len(families), figsize=(7 * len(families), 9), squeeze=False)
    for column, (_, records) in enumerate(families):
        title = family_title(records[0]["result"])
        tokens = [item["result"]["scenario"]["tokens"] for item in records]

        forward_axis = axes[0, column]
        for prefix, color in FORWARD_COLORS.items():
            key_points = []
            values = []
            for item in records:
                runtime_key = find_runtime_key(item["result"]["runtime_ms"], prefix)
                if runtime_key is None:
                    continue
                key_points.append(item["result"]["scenario"]["tokens"])
                values.append(item["result"]["runtime_ms"][runtime_key]["mean_ms"])
            if values:
                forward_axis.plot(key_points, values, marker="o", linewidth=2.0, label=prefix, color=color)
        forward_axis.set_title(f"Forward: {title}")
        forward_axis.set_xlabel("Sequence length")
        forward_axis.set_ylabel("Runtime (ms)")
        forward_axis.set_yscale("log")
        forward_axis.set_xticks(tokens)
        forward_axis.grid(True, linestyle=":", alpha=0.35)
        forward_axis.legend(frameon=False)

        train_axis = axes[1, column]
        for prefix, color in TRAIN_COLORS.items():
            key_points = []
            values = []
            for item in records:
                runtime_key = find_runtime_key(item["result"]["runtime_ms"], prefix)
                if runtime_key is None:
                    continue
                key_points.append(item["result"]["scenario"]["tokens"])
                values.append(item["result"]["runtime_ms"][runtime_key]["mean_ms"])
            if values:
                train_axis.plot(key_points, values, marker="o", linewidth=2.0, label=prefix, color=color)
        train_axis.set_title(f"Train step: {title}")
        train_axis.set_xlabel("Sequence length")
        train_axis.set_ylabel("Runtime (ms)")
        train_axis.set_yscale("log")
        train_axis.set_xticks(tokens)
        train_axis.grid(True, linestyle=":", alpha=0.35)
        train_axis.legend(frameon=False)
    fig.suptitle("Full Attention-Block Runtime Scaling", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_agreement(grouped: dict[str, list[dict[str, Any]]], output_path: Path) -> None:
    families = list(grouped.items())
    fig, axes = plt.subplots(2, len(families), figsize=(7 * len(families), 9), squeeze=False)
    for column, (_, records) in enumerate(families):
        title = family_title(records[0]["result"])
        tokens = [item["result"]["scenario"]["tokens"] for item in records]

        forward_axis = axes[0, column]
        for section_key, backend_key, metric_key, label, color in AGREEMENT_SERIES[:2]:
            key_points = []
            values = []
            for item in records:
                key_points.append(item["result"]["scenario"]["tokens"])
                values.append(safe_positive(item["result"][section_key][backend_key][metric_key]))
            forward_axis.plot(key_points, values, marker="o", linewidth=2.0, label=label, color=color)
        forward_axis.set_title(f"Forward agreement: {title}")
        forward_axis.set_xlabel("Sequence length")
        forward_axis.set_ylabel("Max absolute diff")
        forward_axis.set_yscale("log")
        forward_axis.set_xticks(tokens)
        forward_axis.grid(True, linestyle=":", alpha=0.35)
        forward_axis.legend(frameon=False)

        train_axis = axes[1, column]
        for section_key, backend_key, metric_key, label, color in AGREEMENT_SERIES[2:]:
            key_points = []
            values = []
            for item in records:
                key_points.append(item["result"]["scenario"]["tokens"])
                values.append(safe_positive(item["result"][section_key][backend_key][metric_key]))
            train_axis.plot(key_points, values, marker="o", linewidth=2.0, label=label, color=color)
        train_axis.set_title(f"Train-step agreement: {title}")
        train_axis.set_xlabel("Sequence length")
        train_axis.set_ylabel("Absolute loss diff")
        train_axis.set_yscale("log")
        train_axis.set_xticks(tokens)
        train_axis.grid(True, linestyle=":", alpha=0.35)
        train_axis.legend(frameon=False)
    fig.suptitle("Full Attention-Block Agreement vs Sequence Length", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    grouped = collect_records(args.inputs)
    if not grouped:
        raise ValueError("No attention benchmark results found in the provided artifacts.")

    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    runtime_path = args.output_prefix.with_name(args.output_prefix.name + "_runtime.png")
    agreement_path = args.output_prefix.with_name(args.output_prefix.name + "_agreement.png")
    plot_runtime(grouped, runtime_path)
    plot_agreement(grouped, agreement_path)
    print(runtime_path)
    print(agreement_path)


if __name__ == "__main__":
    main()

