import importlib.util
import sys
from pathlib import Path

import pytest


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_real_data_scale_harness.py"
SPEC = importlib.util.spec_from_file_location("run_real_data_scale_harness", SCRIPT_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_build_variants_deduplicates_and_preserves_baseline() -> None:
    variants = MODULE.build_variants((0, 8, 8, 16), "jittered_pairs")
    assert [variant.label for variant in variants] == ["standard_rope", "householder_m8", "householder_m16"]
    assert [variant.init for variant in variants] == ["paired_identity", "jittered_pairs", "jittered_pairs"]


def test_build_variants_rejects_negative_counts() -> None:
    with pytest.raises(ValueError):
        MODULE.build_variants((0, -1), "jittered_pairs")


def test_resolve_backend_auto_prefers_tpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("COLAB_TPU_ADDR", "grpc://10.0.0.2:8470")
    assert MODULE.resolve_backend("auto") == "flax"
    monkeypatch.delenv("COLAB_TPU_ADDR", raising=False)
    assert MODULE.resolve_backend("auto") == "torch"


def test_config_to_dict_serializes_output_dir() -> None:
    output_dir = Path("artifacts") / "unit-test"
    config = MODULE.RealDataHarnessConfig(
        backend="torch",
        dataset_name="wikitext",
        dataset_config="wikitext-103-raw-v1",
        tokenizer_name="gpt2",
        train_text_limit=8,
        eval_text_limit=4,
        seq_len=16,
        batch_size=2,
        eval_batch_size=2,
        gradient_accumulation_steps=1,
        train_steps=1,
        eval_every=1,
        eval_batches=1,
        num_layers=1,
        embed_dim=32,
        num_heads=4,
        mlp_ratio=2.0,
        learning_rate=1.0e-3,
        weight_decay=0.0,
        seed=0,
        use_compile=False,
        use_bf16=False,
        householder_init="jittered_pairs",
        reflector_sweep=(0, 8),
        output_dir=output_dir,
        output_stem="smoke",
    )
    payload = config.to_dict()
    assert payload["output_dir"] == str(output_dir)
    assert payload["reflector_sweep"] == (0, 8)
