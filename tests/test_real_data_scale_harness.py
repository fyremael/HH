import importlib.util
import sys
from pathlib import Path

import pytest
import torch


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
        train_text_limit=None,
        eval_text_limit=None,
        seq_len=16,
        batch_size=2,
        eval_batch_size=2,
        gradient_accumulation_steps=1,
        train_steps=1,
        eval_every=1,
        eval_batches=None,
        log_every=1,
        diagnostics_every=1,
        diagnostic_token_limit=8,
        num_layers=1,
        embed_dim=32,
        num_heads=4,
        mlp_ratio=2.0,
        learning_rate=1.0e-3,
        weight_decay=0.0,
        seed=0,
        use_compile=False,
        use_bf16=False,
        intervention_eval=True,
        householder_init="jittered_pairs",
        householder_mixing_sweep=("global",),
        householder_local_band_pairs=2,
        freeze_qk_after_warmup_steps=None,
        reflector_sweep=(0, 8),
        output_dir=output_dir,
        output_stem="smoke",
    )
    payload = config.to_dict()
    assert payload["output_dir"] == str(output_dir)
    assert payload["reflector_sweep"] == (0, 8)
    assert payload["diagnostics_every"] == 1
    assert payload["train_text_limit"] is None
    assert payload["eval_batches"] is None
    assert payload["intervention_eval"] is True
    assert payload["householder_mixing_sweep"] == ("global",)
    assert payload["householder_local_band_pairs"] == 2
    assert payload["freeze_qk_after_warmup_steps"] is None


def test_parse_optional_record_limit_accepts_full_keywords() -> None:
    assert MODULE.parse_optional_record_limit("full") is None
    assert MODULE.parse_optional_record_limit("ALL") is None
    assert MODULE.parse_optional_record_limit("12") == 12


def test_parse_optional_eval_batches_accepts_full_keywords_and_zero() -> None:
    assert MODULE.parse_optional_eval_batches("full") is None
    assert MODULE.parse_optional_eval_batches("0") is None
    assert MODULE.parse_optional_eval_batches("7") == 7


def test_parse_optional_nonnegative_int_accepts_none_keyword() -> None:
    assert MODULE.parse_optional_nonnegative_int("none") is None
    assert MODULE.parse_optional_nonnegative_int("0") == 0
    assert MODULE.parse_optional_nonnegative_int("9") == 9


def test_override_torch_householder_enabled_restores_state() -> None:
    model = MODULE.TorchHouseholderLM(
        vocab_size=64,
        embed_dim=32,
        num_heads=4,
        num_layers=2,
        mlp_ratio=2.0,
        variant=MODULE.RopeVariant(label="householder_m4", num_reflectors=4, init="jittered_pairs"),
    )
    ropes = list(MODULE.iter_torch_householder_ropes(model))
    assert ropes
    assert all(rope.config.enabled for rope in ropes)
    with MODULE.override_torch_householder_enabled(model, False):
        assert all(not rope.config.enabled for rope in ropes)
    assert all(rope.config.enabled for rope in ropes)


def test_evaluate_torch_householder_intervention_reports_disable_delta(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = MODULE.TorchHouseholderLM(
        vocab_size=64,
        embed_dim=32,
        num_heads=4,
        num_layers=1,
        mlp_ratio=2.0,
        variant=MODULE.RopeVariant(label="householder_m4", num_reflectors=4, init="jittered_pairs"),
    )

    def fake_evaluate_torch(*args, **kwargs):
        ropes = list(MODULE.iter_torch_householder_ropes(model))
        return 4.0 if all(rope.config.enabled for rope in ropes) else 4.5

    monkeypatch.setattr(MODULE, "evaluate_torch", fake_evaluate_torch)
    summary = MODULE.evaluate_torch_householder_intervention(
        model,
        split=[],
        variant=MODULE.RopeVariant(label="householder_m4", num_reflectors=4, init="jittered_pairs"),
        batch_size=2,
        eval_batches=None,
        seed=0,
        device=torch.device("cpu"),
        use_bf16=False,
    )
    assert summary is not None
    assert summary["active_eval_loss"] == pytest.approx(4.0)
    assert summary["disabled_eval_loss"] == pytest.approx(4.5)
    assert summary["disabled_minus_active_eval_loss"] == pytest.approx(0.5)


def test_build_variants_supports_frequency_banded_labels() -> None:
    variants = MODULE.build_variants(
        (0, 8),
        "jittered_pairs",
        householder_mixing_sweep=("global", "frequency_banded"),
        householder_local_band_pairs=3,
    )
    assert [variant.label for variant in variants] == [
        "standard_rope",
        "householder_m8",
        "householder_local_p3_m8",
    ]
    assert variants[-1].mixing_strategy == "frequency_banded"
    assert variants[-1].local_band_pairs == 3


def test_set_torch_qk_projection_trainable_disables_qk_updates() -> None:
    model = MODULE.TorchHouseholderLM(
        vocab_size=64,
        embed_dim=32,
        num_heads=4,
        num_layers=1,
        mlp_ratio=2.0,
        variant=MODULE.RopeVariant(label="householder_m4", num_reflectors=4, init="jittered_pairs"),
    )
    parameters = list(MODULE.iter_torch_qk_projection_parameters(model))
    assert parameters
    assert all(parameter.requires_grad for parameter in parameters)
    MODULE.set_torch_qk_projection_trainable(model, False)
    assert all(not parameter.requires_grad for parameter in parameters)
    assert MODULE.count_trainable_parameters(iter(parameters)) == 0


def test_fold_torch_householder_into_projections_preserves_logits() -> None:
    torch.manual_seed(0)
    model = MODULE.TorchHouseholderLM(
        vocab_size=64,
        embed_dim=32,
        num_heads=4,
        num_layers=2,
        mlp_ratio=2.0,
        variant=MODULE.RopeVariant(label="householder_m4", num_reflectors=4, init="jittered_pairs"),
    )
    input_ids = torch.randint(0, 64, (2, 12))
    with torch.no_grad():
        active_logits = model(input_ids)
        with MODULE.fold_torch_householder_into_projections(model):
            folded_logits = model(input_ids)
        restored_logits = model(input_ids)
    assert torch.allclose(active_logits, folded_logits, atol=1.0e-5)
    assert torch.allclose(active_logits, restored_logits, atol=1.0e-6)


def test_reduce_rope_diagnostics_summarizes_nested_metrics() -> None:
    reduced = MODULE.reduce_rope_diagnostics(
        {
            "orthogonality_defect": torch.tensor([1.0e-6, 2.0e-6]),
            "relativity_defect": torch.tensor([[3.0e-6, 5.0e-6]]),
            "block_mixing_energy": torch.tensor(
                [
                    [1.0, 0.1, 0.2],
                    [0.3, 1.1, 0.4],
                    [0.5, 0.6, 1.2],
                ]
            ),
            "reflector_utilization": {
                "raw_norms": torch.tensor([[2.0, 4.0]]),
                "pair_cosine_similarity": torch.tensor([0.9, 0.95]),
                "identity_deviation": torch.tensor([0.01, 0.02]),
            },
        }
    )
    assert reduced["orthogonality_defect_mean"] == pytest.approx(1.5e-6)
    assert reduced["relativity_defect_max"] == pytest.approx(5.0e-6)
    assert reduced["block_mixing_offdiag_mean"] == pytest.approx((0.1 + 0.2 + 0.3 + 0.4 + 0.5 + 0.6) / 6.0)
    assert reduced["raw_reflector_norm_mean"] == pytest.approx(3.0)
    assert reduced["pair_cosine_similarity_mean"] == pytest.approx(0.925)


def test_flatten_metrics_flattens_nested_diagnostics() -> None:
    flat = MODULE.flatten_metrics(
        {
            "step": 3,
            "diagnostics": {
                "summary": {"probe_loss": 6.5},
                "layers": {"block_0": {"rope_orthogonality_defect_mean": 1.0e-6}},
            },
        }
    )
    assert flat["step"] == 3
    assert flat["diagnostics_summary_probe_loss"] == 6.5
    assert flat["diagnostics_layers_block_0_rope_orthogonality_defect_mean"] == 1.0e-6
