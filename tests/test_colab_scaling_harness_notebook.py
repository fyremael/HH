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
    assert "logs stream live" in intro or "stream live" in intro

    joined_sources = "\n".join(cell["source"] for cell in cells)
    assert 'run_command("git", "clone"' in joined_sources
    assert "uv" in joined_sources
    assert "--system" in joined_sources
    assert "wikitext-103-raw-v1" in joined_sources
    assert "run_real_data_scale_harness.py" in joined_sources
    assert "A100_PRESETS" in joined_sources
    assert "fast_sanity" in joined_sources
    assert "serious_comparison" in joined_sources
    assert "long_context_stress" in joined_sources
    assert 'ACTIVE_PROFILE = "serious_comparison"' in joined_sources
    assert '"seq_len": 1024' in joined_sources
    assert '"batch_size": 6' in joined_sources
    assert '"num_layers": 6' in joined_sources
    assert '"embed_dim": 1024' in joined_sources
    assert '"num_heads": 16' in joined_sources
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
