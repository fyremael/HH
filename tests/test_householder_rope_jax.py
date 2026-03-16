from __future__ import annotations

import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
from jax import random

from householder_rope.jax_attention import householder_attention
from householder_rope.jax_core import (
    JaxBlockDiagonalRoPECore,
    JaxHouseholderRoPE,
    JaxHouseholderRoPEConfig,
    apply_householder_stack,
    householder_normalize,
    materialize_Q,
    premix_qk,
)
from householder_rope.jax_diagnostics import (
    attention_logit_path_error,
    commutator_defect,
    orthogonality_defect,
    relativity_defect,
    reversibility_defect,
)


def _dense_apply_headwise(x: jnp.ndarray, matrices: jnp.ndarray) -> jnp.ndarray:
    return jnp.einsum("bhti,hji->bhtj", x, matrices)


def test_jax_single_reflector_is_symmetric_orthogonal_and_involutory() -> None:
    key = random.PRNGKey(0)
    u = householder_normalize(random.normal(key, (8,)), eps=1.0e-8)
    H = jnp.eye(8) - 2.0 * jnp.outer(u, u)
    identity = jnp.eye(8)
    assert bool(jnp.allclose(H, H.T, atol=1.0e-6))
    assert bool(jnp.allclose(H.T @ H, identity, atol=1.0e-6))
    assert bool(jnp.allclose(H @ H, identity, atol=1.0e-6))


def test_jax_stack_materialization_is_orthogonal() -> None:
    key = random.PRNGKey(1)
    V = random.normal(key, (3, 4, 16))
    Q = materialize_Q(V)
    defect = orthogonality_defect(Q)
    assert float(jnp.max(defect)) < 1.0e-6


def test_jax_determinant_parity_tracks_reflector_count() -> None:
    odd_Q = materialize_Q(random.normal(random.PRNGKey(2), (3, 8)))
    even_Q = materialize_Q(random.normal(random.PRNGKey(3), (4, 8)))
    assert float(jnp.linalg.det(odd_Q)) < 0.0
    assert float(jnp.linalg.det(even_Q)) > 0.0


def test_jax_premix_matches_dense_q_transpose_application() -> None:
    q = random.normal(random.PRNGKey(4), (2, 3, 5, 8))
    k = random.normal(random.PRNGKey(5), (2, 3, 5, 8))
    V = random.normal(random.PRNGKey(6), (3, 4, 8))
    q_bar, k_bar = premix_qk(q, k, V)
    Q = materialize_Q(V)
    dense_q = _dense_apply_headwise(q, jnp.swapaxes(Q, -1, -2))
    dense_k = _dense_apply_headwise(k, jnp.swapaxes(Q, -1, -2))
    assert bool(jnp.allclose(q_bar, dense_q, atol=1.0e-5))
    assert bool(jnp.allclose(k_bar, dense_k, atol=1.0e-5))


def test_jax_relativity_and_reversibility_hold_for_nd_rope() -> None:
    config = JaxHouseholderRoPEConfig(
        mode="per_head",
        num_reflectors=4,
        init="jittered_pairs",
        rope_ndim=2,
    )
    rope = JaxHouseholderRoPE(
        num_heads=2,
        head_dim=8,
        config=config,
        rope_core=JaxBlockDiagonalRoPECore(dim=8, ndim=2),
        key=random.PRNGKey(7),
    )
    pos_a = jnp.asarray([1.5, -2.0])
    pos_b = jnp.asarray([3.0, 4.0])
    rel = relativity_defect(rope, pos_a, pos_b)
    rev = reversibility_defect(rope, pos_b)
    assert float(jnp.max(rel)) < 1.0e-5
    assert float(jnp.max(rev)) < 1.0e-5


def test_jax_transported_generators_continue_to_commute() -> None:
    rope_core = JaxBlockDiagonalRoPECore(dim=16, ndim=3)
    V = random.normal(random.PRNGKey(8), (4, 16))
    Q = materialize_Q(V)
    defect = commutator_defect(rope_core, Q)
    assert float(jnp.max(defect)) < 1.0e-5


def test_jax_single_reflector_rank_two_transport_identity() -> None:
    u = householder_normalize(random.normal(random.PRNGKey(9), (12,)), eps=1.0e-8)
    H = jnp.eye(12) - 2.0 * jnp.outer(u, u)
    A = random.normal(random.PRNGKey(10), (12, 12))
    B = A - A.T
    left = H @ B @ H
    uuT = jnp.outer(u, u)
    right = B - 2.0 * (uuT @ B + B @ uuT)
    assert bool(jnp.allclose(left, right, atol=1.0e-5))


def test_jax_property_sweep_keeps_dense_and_matrix_free_paths_aligned() -> None:
    dims = [8, 16, 32]
    reflector_counts = [0, 2, 4, 8]
    key = random.PRNGKey(11)
    for dim in dims:
        for reflector_count in reflector_counts:
            key, q_key, v_key = random.split(key, 3)
            q = random.normal(q_key, (1, 2, 4, dim))
            V = random.normal(v_key, (2, reflector_count, dim))
            stacked = apply_householder_stack(q, V, order="reverse")
            Q = materialize_Q(V)
            dense = _dense_apply_headwise(q, jnp.swapaxes(Q, -1, -2))
            assert bool(jnp.allclose(stacked, dense, atol=1.0e-5))


def test_jax_paired_identity_initialization_is_exact_identity() -> None:
    config = JaxHouseholderRoPEConfig(
        mode="shared",
        num_reflectors=4,
        init="paired_identity",
        rope_ndim=1,
    )
    rope = JaxHouseholderRoPE(
        num_heads=2,
        head_dim=8,
        config=config,
        rope_core=JaxBlockDiagonalRoPECore(dim=8, ndim=1),
        key=random.PRNGKey(12),
    )
    Q = rope.materialize_Q().astype(jnp.float64)
    identity = jnp.eye(8, dtype=Q.dtype)
    assert bool(jnp.allclose(Q, identity, atol=1.0e-6))


def test_jax_dense_and_matrix_free_attention_logits_match() -> None:
    config = JaxHouseholderRoPEConfig(
        mode="group_shared",
        group_size=2,
        num_reflectors=4,
        init="jittered_pairs",
        rope_ndim=2,
    )
    rope = JaxHouseholderRoPE(
        num_heads=4,
        head_dim=8,
        config=config,
        rope_core=JaxBlockDiagonalRoPECore(dim=8, ndim=2),
        key=random.PRNGKey(13),
    )
    q = random.normal(random.PRNGKey(14), (2, 4, 6, 8))
    k = random.normal(random.PRNGKey(15), (2, 4, 6, 8))
    pos = jnp.asarray(
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
    assert float(jnp.max(error)) < 1.0e-5


def test_jax_training_smoke_runs_without_nans() -> None:
    config = JaxHouseholderRoPEConfig(
        mode="per_head",
        num_reflectors=4,
        init="jittered_pairs",
        rope_ndim=1,
    )
    rope = JaxHouseholderRoPE(
        num_heads=2,
        head_dim=8,
        config=config,
        rope_core=JaxBlockDiagonalRoPECore(dim=8, ndim=1),
        key=random.PRNGKey(16),
    )
    q = random.normal(random.PRNGKey(17), (2, 2, 5, 8))
    k = random.normal(random.PRNGKey(18), (2, 2, 5, 8))
    v = random.normal(random.PRNGKey(19), (2, 2, 5, 8))
    target = random.normal(random.PRNGKey(20), (2, 2, 5, 8))
    pos = jnp.arange(5, dtype=jnp.float32)

    reflectors = rope.reflectors

    def loss_fn(current_reflectors: jnp.ndarray) -> jnp.ndarray:
        current_rope = rope.replace_reflectors(current_reflectors)
        output, _ = householder_attention(q, k, v, pos, current_rope)
        return jnp.mean((output - target) ** 2)

    for _ in range(3):
        loss, grad = jax.value_and_grad(loss_fn)(reflectors)
        assert bool(jnp.isfinite(loss))
        reflectors = reflectors - 1.0e-2 * grad

    trained_rope = rope.replace_reflectors(reflectors)
    defect = orthogonality_defect(trained_rope.materialize_Q(expand_heads=True))
    assert float(jnp.max(defect)) < 1.0e-6
