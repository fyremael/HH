from __future__ import annotations

from typing import Any

import jax.numpy as jnp

from .jax_core import JaxBlockDiagonalRoPECore, JaxHouseholderRoPE, materialize_Q


def _frobenius_norm(x: jnp.ndarray) -> jnp.ndarray:
    return jnp.linalg.norm(x, axis=(-2, -1))


def _relative_matrix_error(reference: jnp.ndarray, estimate: jnp.ndarray) -> jnp.ndarray:
    return _frobenius_norm(reference - estimate) / jnp.maximum(_frobenius_norm(reference), 1.0e-12)


def orthogonality_defect(Q: jnp.ndarray) -> jnp.ndarray:
    if Q.ndim == 2:
        identity = jnp.eye(Q.shape[-1], dtype=Q.dtype)
        return _relative_matrix_error(identity, jnp.swapaxes(Q, -1, -2) @ Q)
    identity = jnp.broadcast_to(jnp.eye(Q.shape[-1], dtype=Q.dtype), Q.shape)
    product = jnp.matmul(jnp.swapaxes(Q, -1, -2), Q)
    return _relative_matrix_error(identity, product)


def relativity_defect(
    rope: JaxHouseholderRoPE,
    pos_a: jnp.ndarray | float | int,
    pos_b: jnp.ndarray | float | int,
) -> jnp.ndarray:
    R_a = rope.materialize_rope(pos_a).astype(jnp.float32)
    R_b = rope.materialize_rope(pos_b).astype(jnp.float32)
    delta = jnp.asarray(pos_b, dtype=jnp.float32) - jnp.asarray(pos_a, dtype=jnp.float32)
    R_delta = rope.materialize_rope(delta).astype(jnp.float32)
    lhs = jnp.matmul(jnp.swapaxes(R_a, -1, -2), R_b)
    return _relative_matrix_error(R_delta, lhs)


def reversibility_defect(rope: JaxHouseholderRoPE, pos: jnp.ndarray | float | int) -> jnp.ndarray:
    pos_array = jnp.asarray(pos, dtype=jnp.float32)
    R_pos = rope.materialize_rope(pos_array).astype(jnp.float32)
    R_neg = rope.materialize_rope(-pos_array).astype(jnp.float32)
    return _relative_matrix_error(R_neg, jnp.swapaxes(R_pos, -1, -2))


def commutator_defect(rope_core: JaxBlockDiagonalRoPECore, Q: jnp.ndarray) -> jnp.ndarray:
    generators = rope_core.generators().astype(Q.dtype)
    if Q.ndim == 2:
        Q = Q[None, ...]
    bank_defects = []
    for q_matrix in Q:
        transported = jnp.stack([q_matrix @ generator @ jnp.swapaxes(q_matrix, -1, -2) for generator in generators])
        defects = []
        for left in range(transported.shape[0]):
            for right in range(left + 1, transported.shape[0]):
                commutator = transported[left] @ transported[right] - transported[right] @ transported[left]
                defects.append(_frobenius_norm(commutator))
        if defects:
            bank_defects.append(jnp.stack(defects))
        else:
            bank_defects.append(jnp.zeros((1,), dtype=Q.dtype))
    stacked = jnp.stack(bank_defects)
    return jnp.squeeze(stacked, axis=0) if stacked.shape[0] == 1 else stacked


def block_mixing_energy(Q: jnp.ndarray) -> jnp.ndarray:
    if Q.ndim == 2:
        Q = Q[None, ...]
    block_count = Q.shape[-1] // 2
    energies = []
    for q_matrix in Q:
        energy = jnp.zeros((block_count, block_count), dtype=q_matrix.dtype)
        for row_block in range(block_count):
            row_slice = slice(2 * row_block, 2 * row_block + 2)
            for col_block in range(block_count):
                col_slice = slice(2 * col_block, 2 * col_block + 2)
                block = q_matrix[row_slice, col_slice]
                energy = energy.at[row_block, col_block].set(_frobenius_norm(block) ** 2)
        energies.append(energy)
    stacked = jnp.stack(energies)
    return jnp.squeeze(stacked, axis=0) if stacked.shape[0] == 1 else stacked


def reflector_utilization(
    V: jnp.ndarray,
    *,
    grad: jnp.ndarray | None = None,
    eps: float = 1.0e-8,
) -> dict[str, jnp.ndarray]:
    summary: dict[str, jnp.ndarray] = {
        "raw_norms": jnp.linalg.norm(V, axis=-1),
    }
    if V.shape[-2] >= 2:
        pair_count = V.shape[-2] // 2
        left = V[..., 0 : 2 * pair_count : 2, :]
        right = V[..., 1 : 2 * pair_count : 2, :]
        numerator = jnp.sum(left * right, axis=-1)
        denominator = jnp.maximum(
            jnp.linalg.norm(left, axis=-1) * jnp.linalg.norm(right, axis=-1),
            eps,
        )
        summary["pair_cosine_similarity"] = numerator / denominator
    if grad is not None:
        summary["gradient_norms"] = jnp.linalg.norm(grad, axis=-1)

    Q = materialize_Q(V, eps=eps).astype(jnp.float32)
    identity = jnp.eye(Q.shape[-1], dtype=Q.dtype)
    if Q.ndim == 3:
        identity = jnp.broadcast_to(identity, Q.shape)
    summary["identity_deviation"] = _relative_matrix_error(identity, Q)
    summary["orthogonality_defect"] = orthogonality_defect(Q)
    return summary


def apply_dense_rope_sequence(
    x: jnp.ndarray,
    rope: JaxHouseholderRoPE,
    pos: jnp.ndarray | float | int,
) -> jnp.ndarray:
    matrices = rope.materialize_rope(pos, expand_heads=True).astype(x.dtype)
    if matrices.ndim == 3:
        return jnp.einsum("bhti,tji->bhtj", x, matrices)
    if matrices.ndim == 4:
        return jnp.einsum("bhti,htji->bhtj", x, matrices)
    raise ValueError(f"Unsupported dense RoPE matrix shape {tuple(matrices.shape)}.")


def attention_logit_path_error(
    q: jnp.ndarray,
    k: jnp.ndarray,
    pos: jnp.ndarray | float | int,
    rope: JaxHouseholderRoPE,
) -> jnp.ndarray:
    q_matrix_free, k_matrix_free = rope(q, k, pos)
    q_dense = apply_dense_rope_sequence(q, rope, pos)
    k_dense = apply_dense_rope_sequence(k, rope, pos)
    dense_logits = jnp.einsum("bhti,bhsi->bhts", q_dense, k_dense)
    matrix_free_logits = jnp.einsum("bhti,bhsi->bhts", q_matrix_free, k_matrix_free)
    return _relative_matrix_error(dense_logits.astype(jnp.float32), matrix_free_logits.astype(jnp.float32))


def summarize_householder_rope_diagnostics(
    rope: JaxHouseholderRoPE,
    *,
    pos_a: jnp.ndarray | float | int,
    pos_b: jnp.ndarray | float | int,
    q: jnp.ndarray | None = None,
    k: jnp.ndarray | None = None,
    grad: jnp.ndarray | None = None,
) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    Q = rope.materialize_Q(expand_heads=False).astype(jnp.float32)
    summary["orthogonality_defect"] = orthogonality_defect(Q)
    summary["relativity_defect"] = relativity_defect(rope, pos_a=pos_a, pos_b=pos_b)
    summary["reversibility_defect"] = reversibility_defect(rope, pos=pos_a)
    summary["commutator_defect"] = commutator_defect(rope.rope_core, Q)
    summary["block_mixing_energy"] = block_mixing_energy(Q)
    summary["reflector_utilization"] = reflector_utilization(
        rope.reflectors,
        grad=grad,
        eps=rope.config.eps,
    )
    if q is not None and k is not None:
        summary["attention_logit_path_error"] = attention_logit_path_error(q, k, pos_a, rope)
    return summary



