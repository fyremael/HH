from __future__ import annotations

import torch

from householder_rope import (
    BlockDiagonalRoPECore,
    HouseholderRoPE,
    HouseholderRoPEConfig,
    apply_householder_stack,
    attention_logit_path_error,
    commutator_defect,
    householder_normalize,
    materialize_Q,
    orthogonality_defect,
    premix_qk,
    relativity_defect,
    reversibility_defect,
)


def _dense_apply_headwise(x: torch.Tensor, matrices: torch.Tensor) -> torch.Tensor:
    return torch.einsum("bhti,hji->bhtj", x, matrices)


def test_single_reflector_is_symmetric_orthogonal_and_involutory() -> None:
    torch.manual_seed(0)
    u = householder_normalize(torch.randn(8), eps=1.0e-8)
    H = torch.eye(8) - 2.0 * torch.outer(u, u)
    identity = torch.eye(8)
    assert torch.allclose(H, H.transpose(-1, -2), atol=1.0e-6)
    assert torch.allclose(H.transpose(-1, -2) @ H, identity, atol=1.0e-6)
    assert torch.allclose(H @ H, identity, atol=1.0e-6)


def test_stack_materialization_is_orthogonal() -> None:
    torch.manual_seed(1)
    V = torch.randn(3, 4, 16)
    Q = materialize_Q(V)
    defect = orthogonality_defect(Q)
    assert defect.max().item() < 1.0e-6


def test_determinant_parity_tracks_reflector_count() -> None:
    torch.manual_seed(2)
    odd_Q = materialize_Q(torch.randn(3, 8))
    even_Q = materialize_Q(torch.randn(4, 8))
    assert torch.det(odd_Q).item() < 0.0
    assert torch.det(even_Q).item() > 0.0


def test_premix_matches_dense_q_transpose_application() -> None:
    torch.manual_seed(3)
    q = torch.randn(2, 3, 5, 8)
    k = torch.randn(2, 3, 5, 8)
    V = torch.randn(3, 4, 8)
    q_bar, k_bar = premix_qk(q, k, V)
    Q = materialize_Q(V)
    dense_q = _dense_apply_headwise(q, Q.transpose(-1, -2))
    dense_k = _dense_apply_headwise(k, Q.transpose(-1, -2))
    assert torch.allclose(q_bar, dense_q, atol=1.0e-5)
    assert torch.allclose(k_bar, dense_k, atol=1.0e-5)


def test_relativity_and_reversibility_hold_for_nd_rope() -> None:
    torch.manual_seed(4)
    config = HouseholderRoPEConfig(
        mode="per_head",
        num_reflectors=4,
        init="jittered_pairs",
        rope_ndim=2,
    )
    rope = HouseholderRoPE(
        num_heads=2,
        head_dim=8,
        config=config,
        rope_core=BlockDiagonalRoPECore(dim=8, ndim=2),
    )
    pos_a = torch.tensor([1.5, -2.0])
    pos_b = torch.tensor([3.0, 4.0])
    rel = relativity_defect(rope, pos_a, pos_b)
    rev = reversibility_defect(rope, pos_b)
    assert rel.max().item() < 1.0e-6
    assert rev.max().item() < 1.0e-6


def test_transported_generators_continue_to_commute() -> None:
    torch.manual_seed(5)
    rope_core = BlockDiagonalRoPECore(dim=16, ndim=3)
    V = torch.randn(4, 16)
    Q = materialize_Q(V)
    defect = commutator_defect(rope_core, Q)
    assert defect.max().item() < 1.0e-6


def test_single_reflector_rank_two_transport_identity() -> None:
    torch.manual_seed(6)
    u = householder_normalize(torch.randn(12), eps=1.0e-8)
    H = torch.eye(12) - 2.0 * torch.outer(u, u)
    A = torch.randn(12, 12)
    B = A - A.transpose(-1, -2)
    left = H @ B @ H
    uuT = torch.outer(u, u)
    right = B - 2.0 * (uuT @ B + B @ uuT)
    assert torch.allclose(left, right, atol=1.0e-6)


@torch.no_grad()
def test_property_sweep_keeps_dense_and_matrix_free_paths_aligned() -> None:
    torch.manual_seed(7)
    dims = [8, 16, 32]
    reflector_counts = [0, 2, 4, 8]
    for dim in dims:
        for reflector_count in reflector_counts:
            q = torch.randn(1, 2, 4, dim)
            V = torch.randn(2, reflector_count, dim)
            stacked = apply_householder_stack(q, V, order="reverse")
            Q = materialize_Q(V)
            dense = _dense_apply_headwise(q, Q.transpose(-1, -2))
            assert torch.allclose(stacked, dense, atol=1.0e-5)


def test_paired_identity_initialization_is_exact_identity() -> None:
    torch.manual_seed(8)
    config = HouseholderRoPEConfig(
        mode="shared",
        num_reflectors=4,
        init="paired_identity",
        rope_ndim=1,
    )
    rope = HouseholderRoPE(
        num_heads=2,
        head_dim=8,
        config=config,
        rope_core=BlockDiagonalRoPECore(dim=8, ndim=1),
    )
    Q = rope.materialize_Q().to(torch.float64)
    identity = torch.eye(8, dtype=Q.dtype)
    assert torch.allclose(Q, identity, atol=1.0e-6)


def test_dense_and_matrix_free_attention_logits_match() -> None:
    torch.manual_seed(9)
    config = HouseholderRoPEConfig(
        mode="group_shared",
        group_size=2,
        num_reflectors=4,
        init="jittered_pairs",
        rope_ndim=2,
    )
    rope = HouseholderRoPE(
        num_heads=4,
        head_dim=8,
        config=config,
        rope_core=BlockDiagonalRoPECore(dim=8, ndim=2),
    )
    q = torch.randn(2, 4, 6, 8)
    k = torch.randn(2, 4, 6, 8)
    pos = torch.tensor(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [2.0, 0.0],
            [2.0, 1.0],
        ]
    )
    error = attention_logit_path_error(q, k, pos, rope)
    assert error.max().item() < 1.0e-6

