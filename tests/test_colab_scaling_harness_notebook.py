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

    joined_sources = "\n".join(cell["source"] for cell in cells)
    assert 'run_command("git", "clone"' in joined_sources
    assert "uv pip install --system" in joined_sources
    assert "wikitext-103-raw-v1" in joined_sources
    assert "run_real_data_scale_harness.py" in joined_sources
    assert "A100_PRESETS" in joined_sources
    assert "fast_sanity" in joined_sources
    assert "serious_comparison" in joined_sources
    assert "long_context_stress" in joined_sources
    assert 'ACTIVE_PROFILE = "serious_comparison"' in joined_sources
