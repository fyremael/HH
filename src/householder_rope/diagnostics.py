from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor

from .core import BlockDiagonalRoPECore, HouseholderRoPE, materialize_Q


def _frobenius_norm(x: Tensor) -> Tensor:
    return torch.linalg.matrix_norm(x, ord="fro")


def _relative_matrix_error(reference: Tensor, estimate: Tensor) -> Tensor:
    return _frobenius_norm(reference - estimate) / _frobenius_norm(reference).clamp_min(1.0e-12)


def orthogonality_defect(Q: Tensor) -> Tensor:
    if Q.dim() == 2:
        identity = torch.eye(Q.shape[-1], dtype=Q.dtype, device=Q.device)
        return _relative_matrix_error(identity, Q.transpose(-1, -2) @ Q)
    identity = torch.eye(Q.shape[-1], dtype=Q.dtype, device=Q.device).expand_as(Q)
    product = torch.matmul(Q.transpose(-1, -2), Q)
    return _relative_matrix_error(identity, product)


def relativity_defect(rope: HouseholderRoPE, pos_a: Tensor | float | int, pos_b: Tensor | float | int) -> Tensor:
    R_a = rope.materialize_rope(pos_a).to(torch.float64)
    R_b = rope.materialize_rope(pos_b).to(torch.float64)
    delta = torch.as_tensor(pos_b, dtype=torch.float64) - torch.as_tensor(pos_a, dtype=torch.float64)
    R_delta = rope.materialize_rope(delta).to(torch.float64)
    lhs = torch.matmul(R_a.transpose(-1, -2), R_b)
    return _relative_matrix_error(R_delta, lhs)


def reversibility_defect(rope: HouseholderRoPE, pos: Tensor | float | int) -> Tensor:
    R_pos = rope.materialize_rope(pos).to(torch.float64)
    R_neg = rope.materialize_rope(-torch.as_tensor(pos, dtype=torch.float64)).to(torch.float64)
    return _relative_matrix_error(R_neg, R_pos.transpose(-1, -2))


def commutator_defect(rope_core: BlockDiagonalRoPECore, Q: Tensor) -> Tensor:
    generators = rope_core.generators().to(device=Q.device, dtype=Q.dtype)
    if Q.dim() == 2:
        Q = Q.unsqueeze(0)
    bank_defects = []
    for q_matrix in Q:
        transported = torch.stack([q_matrix @ generator @ q_matrix.transpose(-1, -2) for generator in generators])
        defects = []
        for left in range(transported.shape[0]):
            for right in range(left + 1, transported.shape[0]):
                commutator = transported[left] @ transported[right] - transported[right] @ transported[left]
                defects.append(_frobenius_norm(commutator))
        if defects:
            bank_defects.append(torch.stack(defects))
        else:
            bank_defects.append(torch.zeros(1, dtype=Q.dtype, device=Q.device))
    stacked = torch.stack(bank_defects)
    return stacked.squeeze(0) if stacked.shape[0] == 1 else stacked


def block_mixing_energy(Q: Tensor) -> Tensor:
    if Q.dim() == 2:
        Q = Q.unsqueeze(0)
    block_count = Q.shape[-1] // 2
    energies = []
    for q_matrix in Q:
        energy = torch.zeros(
            block_count,
            block_count,
            dtype=q_matrix.dtype,
            device=q_matrix.device,
        )
        for row_block in range(block_count):
            row_slice = slice(2 * row_block, 2 * row_block + 2)
            for col_block in range(block_count):
                col_slice = slice(2 * col_block, 2 * col_block + 2)
                block = q_matrix[row_slice, col_slice]
                energy[row_block, col_block] = _frobenius_norm(block).square()
        energies.append(energy)
    stacked = torch.stack(energies)
    return stacked.squeeze(0) if stacked.shape[0] == 1 else stacked


def reflector_utilization(
    V: Tensor,
    *,
    grad: Tensor | None = None,
    eps: float = 1.0e-8,
) -> dict[str, Tensor]:
    summary: dict[str, Tensor] = {
        "raw_norms": torch.linalg.vector_norm(V, dim=-1),
        "support_fraction": V.ne(0.0).float().mean(dim=-1),
    }
    if V.shape[-2] >= 2:
        pair_count = V.shape[-2] // 2
        left = V[..., 0 : 2 * pair_count : 2, :]
        right = V[..., 1 : 2 * pair_count : 2, :]
        summary["pair_cosine_similarity"] = F.cosine_similarity(left, right, dim=-1, eps=eps)
    if grad is not None:
        summary["gradient_norms"] = torch.linalg.vector_norm(grad, dim=-1)

    Q = materialize_Q(V, eps=eps).to(dtype=torch.float64)
    identity = torch.eye(Q.shape[-1], dtype=Q.dtype, device=Q.device)
    if Q.dim() == 3:
        identity = identity.unsqueeze(0).expand_as(Q)
    summary["identity_deviation"] = _relative_matrix_error(identity, Q)
    summary["orthogonality_defect"] = orthogonality_defect(Q)
    return summary


def apply_dense_rope_sequence(x: Tensor, rope: HouseholderRoPE, pos: Tensor | float | int) -> Tensor:
    matrices = rope.materialize_rope(pos, expand_heads=True).to(device=x.device, dtype=x.dtype)
    if matrices.dim() == 3:
        return torch.einsum("bhti,tji->bhtj", x, matrices)
    if matrices.dim() == 4:
        return torch.einsum("bhti,htji->bhtj", x, matrices)
    raise ValueError(f"Unsupported dense RoPE matrix shape {tuple(matrices.shape)}.")


def attention_logit_path_error(
    q: Tensor,
    k: Tensor,
    pos: Tensor | float | int,
    rope: HouseholderRoPE,
) -> Tensor:
    q_matrix_free, k_matrix_free = rope(q, k, pos)
    q_dense = apply_dense_rope_sequence(q, rope, pos)
    k_dense = apply_dense_rope_sequence(k, rope, pos)
    dense_logits = torch.einsum("bhti,bhsi->bhts", q_dense, k_dense)
    matrix_free_logits = torch.einsum("bhti,bhsi->bhts", q_matrix_free, k_matrix_free)
    return _relative_matrix_error(dense_logits.to(torch.float64), matrix_free_logits.to(torch.float64))


def summarize_householder_rope_diagnostics(
    rope: HouseholderRoPE,
    *,
    pos_a: Tensor | float | int,
    pos_b: Tensor | float | int,
    q: Tensor | None = None,
    k: Tensor | None = None,
) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    Q = rope.materialize_Q(expand_heads=False).to(torch.float64)
    summary["orthogonality_defect"] = orthogonality_defect(Q)
    summary["relativity_defect"] = relativity_defect(rope, pos_a=pos_a, pos_b=pos_b)
    summary["reversibility_defect"] = reversibility_defect(rope, pos=pos_a)
    summary["commutator_defect"] = commutator_defect(rope.rope_core, Q)
    summary["block_mixing_energy"] = block_mixing_energy(Q)
    summary["reflector_utilization"] = reflector_utilization(
        rope.effective_reflectors(detach=True),
        grad=(
            None
            if rope.reflectors.grad is None
            else rope.reflectors.grad.detach() * rope.reflector_support_mask.to(rope.reflectors.grad)
        ),
        eps=rope.config.eps,
    )
    if q is not None and k is not None:
        summary["attention_logit_path_error"] = attention_logit_path_error(q, k, pos_a, rope)
    return summary
