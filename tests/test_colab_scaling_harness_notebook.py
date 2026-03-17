import json
from pathlib import Path


def test_colab_scaling_harness_notebook_structure() -> None:
    notebook_path = Path(__file__).resolve().parents[1] / "colab_householder_rope_scaling_harness.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))

    assert notebook["nbformat"] == 4
    cells = notebook["cells"]
    assert cells, "Notebook should contain cells."

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
    assert "long_context_stress" in joined_sources
    assert 'ACTIVE_PROFILE = "capacity_limit"' in joined_sources
    assert '"seq_len": 4096' in joined_sources
    assert '"batch_size": 4' in joined_sources
    assert '"train_steps": 400' in joined_sources
    assert '"num_layers": 12' in joined_sources
    assert '"embed_dim": 1536' in joined_sources
    assert '"num_heads": 24' in joined_sources
    assert '"reflector_sweep": [0, 16, 32]' in joined_sources
    assert "log_every" in joined_sources
    assert "diagnostics_every" in joined_sources
    assert "diagnostic_token_limit" in joined_sources
    assert "training_dynamics.png" in joined_sources
    assert "component_diagnostics.png" in joined_sources
    assert "rope_diagnostics.png" in joined_sources
    assert "history_jsonl_path" in joined_sources
    assert 'PYTHONUNBUFFERED="1"' in joined_sources
    assert '"-u"' in joined_sources
    assert '"--log-level"' in joined_sources
    assert '"INFO"' in joined_sources
    assert "Live harness logs will stream below." in joined_sources
    assert "subprocess.Popen(" in joined_sources
    assert "stdout=subprocess.PIPE" in joined_sources
    assert "stderr=subprocess.STDOUT" in joined_sources
    assert "for line in process.stdout" in joined_sources
    assert "If this profile OOMs, switch ACTIVE_PROFILE to 'geometry_signal'." in joined_sources
