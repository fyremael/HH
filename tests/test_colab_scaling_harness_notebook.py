import json
from pathlib import Path


def test_colab_scaling_harness_notebook_structure() -> None:
    notebook_path = Path(__file__).resolve().parents[1] / "colab_householder_rope_scaling_harness.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))

    assert notebook["nbformat"] == 4
    cells = notebook["cells"]
    assert cells, "Notebook should contain cells."
    for index, cell in enumerate(cells):
        if cell["cell_type"] == "code":
            compile(cell["source"], f"colab_householder_rope_scaling_harness.ipynb cell {index}", "exec")

    first_cell = cells[0]
    assert first_cell["cell_type"] == "markdown"
    intro = first_cell["source"]
    assert "## What this is" in intro
    assert "## Why it matters" in intro
    assert "## How to run / use it" in intro
    assert "## Validation plan" in intro
    assert "## Known failure modes" in intro
    assert "## Next steps" in intro
    assert "40GB A100" in intro
    assert "stream" in intro
    assert "4096-token" in intro
    assert "VRAM" in intro
    assert "backs off the batch size" in intro
    assert "all the way down to 1" in intro
    assert "gradient accumulation" in intro

    joined_sources = "\n".join(cell["source"] for cell in cells)
    assert 'run_command("git", "clone"' in joined_sources
    assert "uv" in joined_sources
    assert "--system" in joined_sources
    assert "wikitext-103-raw-v1" in joined_sources
    assert "run_real_data_scale_harness.py" in joined_sources
    assert "A100_PRESETS" in joined_sources
    assert "fast_sanity" in joined_sources
    assert "serious_comparison" in joined_sources
    assert "geometry_signal" in joined_sources
    assert "capacity_limit" in joined_sources
    assert "capacity_max" in joined_sources
    assert "long_context_stress" in joined_sources
    assert 'ACTIVE_PROFILE = "capacity_max"' in joined_sources
    assert '"seq_len": 4096' in joined_sources
    assert '"batch_size": 8' in joined_sources
    assert '"train_steps": 300' in joined_sources
    assert '"num_layers": 24' in joined_sources
    assert '"embed_dim": 2048' in joined_sources
    assert '"num_heads": 32' in joined_sources
    assert '"reflector_sweep": [0, 16, 32]' in joined_sources
    assert "log_every" in joined_sources
    assert "diagnostics_every" in joined_sources
    assert "diagnostic_token_limit" in joined_sources
    assert "training_dynamics.png" in joined_sources
    assert "component_diagnostics.png" in joined_sources
    assert "rope_diagnostics.png" in joined_sources
    assert "history_jsonl_path" in joined_sources
    assert 'PYTHONUNBUFFERED="1"' in joined_sources
    assert 'PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"' in joined_sources
    assert '"-u"' in joined_sources
    assert '"--log-level"' in joined_sources
    assert '"INFO"' in joined_sources
    assert "Live harness logs will stream below." in joined_sources
    assert "subprocess.Popen(" in joined_sources
    assert "stdout=subprocess.PIPE" in joined_sources
    assert "stderr=subprocess.STDOUT" in joined_sources
    assert "for line in process.stdout" in joined_sources
    assert "candidate_batches = list(range" in joined_sources
    assert "base_effective_batch" in joined_sources
    assert "gradient_accumulation_steps" in joined_sources
    assert "run_command_stream" in joined_sources
    assert "Attempting batch_size=" in joined_sources
    assert "gradient_accumulation_steps=" in joined_sources
    assert "CUDA OOM at batch_size=" in joined_sources
    assert "Run succeeded with batch_size=" in joined_sources
    assert "_ga{run_config['gradient_accumulation_steps']}" in joined_sources
    assert 'print("+", " ".join(command), flush=True)' in joined_sources
    assert r' \".join(command)' not in joined_sources
    assert "RUN_OUTPUT_STEM" in joined_sources
    assert "attempted_batches" in joined_sources
    assert "Effective batch size:" in joined_sources
    assert "Effective gradient accumulation:" in joined_sources
    assert "Effective tokens per optimizer step:" in joined_sources
    assert "mean_tokens_per_second" in joined_sources
    assert "peak_memory_gb" in joined_sources
    assert "eval_loss_delta_vs_standard" in joined_sources
    assert "throughput_ratio_vs_standard" in joined_sources
