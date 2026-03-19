from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

import torch
from torch import Tensor, nn

Mode = Literal["shared", "per_head", "group_shared"]
Order = Literal["forward", "reverse"]


def _compute_dtype(dtype: torch.dtype, fp32_norm_accumulation: bool) -> torch.dtype:
    if fp32_norm_accumulation and dtype in {torch.float16, torch.bfloat16}:
        return torch.float32
    return dtype


def householder_normalize(
    v: Tensor,
    eps: float = 1.0e-8,
    fp32_norm_accumulation: bool = True,
) -> Tensor:
    """Return a smoothly normalized reflector direction."""

    compute_dtype = _compute_dtype(v.dtype, fp32_norm_accumulation)
    v_compute = v.to(compute_dtype)
    norm = torch.linalg.vector_norm(v_compute, dim=-1, keepdim=True)
    safe_norm = torch.sqrt(norm.square() + eps * eps)
    return (v_compute / safe_norm).to(v.dtype)


def _householder_tau(
    v: Tensor,
    eps: float,
    fp32_norm_accumulation: bool,
) -> Tensor:
    compute_dtype = _compute_dtype(v.dtype, fp32_norm_accumulation)
    v_compute = v.to(compute_dtype)
    denom = v_compute.square().sum(dim=-1, keepdim=True) + eps * eps
    return (2.0 / denom).to(v.dtype)


def _broadcast_head_bank(v: Tensor, target_ndim: int) -> Tensor:
    shape = [1] * target_ndim
    shape[-3] = v.shape[0]
    shape[-2] = 1
    shape[-1] = v.shape[1]
    return v.reshape(shape)


def build_default_frequency_matrix(
    dim: int,
    ndim: int,
    base: float = 10000.0,
    axis_allocation: Literal["contiguous", "round_robin"] = "contiguous",
) -> Tensor:
    if dim % 2 != 0:
        raise ValueError(f"RoPE head dimension must be even, received {dim}.")
    if ndim < 1:
        raise ValueError(f"RoPE ndim must be positive, received {ndim}.")

    num_pairs = dim // 2
    frequency_matrix = torch.zeros(num_pairs, ndim, dtype=torch.float32)
    if num_pairs == 0:
        return frequency_matrix

    axis_buckets: list[list[int]] = [[] for _ in range(ndim)]
    if axis_allocation == "round_robin":
        for pair_index in range(num_pairs):
            axis_buckets[pair_index % ndim].append(pair_index)
    else:
        base_block, remainder = divmod(num_pairs, ndim)
        start = 0
        for axis in range(ndim):
            block_size = base_block + int(axis < remainder)
            stop = start + block_size
            axis_buckets[axis].extend(range(start, stop))
            start = stop

    for axis, pair_indices in enumerate(axis_buckets):
        if not pair_indices:
            continue
        local_index = torch.arange(len(pair_indices), dtype=torch.float32)
        inverse_frequency = base ** (-local_index / max(len(pair_indices), 1))
        frequency_matrix[pair_indices, axis] = inverse_frequency

    return frequency_matrix


class BlockDiagonalRoPECore(nn.Module):
    """Canonical 1D or ND RoPE in a block-diagonal toral basis."""

    def __init__(
        self,
        dim: int,
        ndim: int = 1,
        *,
        base: float = 10000.0,
        frequency_matrix: Tensor | None = None,
        axis_allocation: Literal["contiguous", "round_robin"] = "contiguous",
    ) -> None:
        super().__init__()
        self.dim = dim
        self.ndim = ndim
        if frequency_matrix is None:
            frequency_matrix = build_default_frequency_matrix(
                dim=dim,
                ndim=ndim,
                base=base,
                axis_allocation=axis_allocation,
            )
        frequency_matrix = torch.as_tensor(frequency_matrix, dtype=torch.float32)
        expected_shape = (dim // 2, ndim)
        if tuple(frequency_matrix.shape) != expected_shape:
            raise ValueError(
                "frequency_matrix must have shape "
                f"{expected_shape}, received {tuple(frequency_matrix.shape)}."
            )
        self.register_buffer("frequency_matrix", frequency_matrix, persistent=True)

    def _canonicalize_positions(self, pos: Tensor | float | int) -> Tensor:
        pos_tensor = torch.as_tensor(
            pos,
            dtype=self.frequency_matrix.dtype,
            device=self.frequency_matrix.device,
        )
        if self.ndim == 1:
            if pos_tensor.ndim == 0:
                pos_tensor = pos_tensor.reshape(1, 1)
            elif pos_tensor.shape[-1] == 1:
                pos_tensor = pos_tensor.reshape(*pos_tensor.shape[:-1], 1)
            else:
                pos_tensor = pos_tensor.unsqueeze(-1)
            return pos_tensor

        if pos_tensor.ndim == 1:
            if pos_tensor.numel() != self.ndim:
                raise ValueError(
                    f"Expected {self.ndim} coordinates, received {pos_tensor.numel()}."
                )
            pos_tensor = pos_tensor.reshape(1, self.ndim)

        if pos_tensor.shape[-1] != self.ndim:
            raise ValueError(
                f"Expected trailing position dimension {self.ndim}, "
                f"received {pos_tensor.shape[-1]}."
            )
        return pos_tensor

    def angles(self, pos: Tensor | float | int) -> Tensor:
        pos_tensor = self._canonicalize_positions(pos)
        frequency_matrix = self.frequency_matrix.to(
            device=pos_tensor.device,
            dtype=pos_tensor.dtype,
        )
        return torch.einsum("...n,pn->...p", pos_tensor, frequency_matrix)

    def apply(self, x: Tensor, pos: Tensor | float | int) -> Tensor:
        if x.shape[-1] != self.dim:
            raise ValueError(
                f"Expected trailing feature dimension {self.dim}, "
                f"received {x.shape[-1]}."
            )

        angles = self.angles(pos).to(device=x.device)
        x_pairs = x.reshape(*x.shape[:-1], self.dim // 2, 2)
        while angles.ndim < x_pairs.ndim - 1:
            angles = angles.unsqueeze(-3)

        cos = torch.cos(angles).to(dtype=x.dtype)
        sin = torch.sin(angles).to(dtype=x.dtype)
        x0 = x_pairs[..., 0]
        x1 = x_pairs[..., 1]
        y0 = x0 * cos - x1 * sin
        y1 = x0 * sin + x1 * cos
        return torch.stack((y0, y1), dim=-1).reshape_as(x)

    def forward(
        self,
        q: Tensor,
        k: Tensor,
        pos: Tensor | float | int,
    ) -> tuple[Tensor, Tensor]:
        return self.apply(q, pos), self.apply(k, pos)

    def materialize(self, pos: Tensor | float | int) -> Tensor:
        angles = self.angles(pos)
        output_shape = (*angles.shape[:-1], self.dim, self.dim)
        matrix = torch.zeros(
            output_shape,
            dtype=angles.dtype,
            device=angles.device,
        )
        cos = torch.cos(angles)
        sin = torch.sin(angles)
        for pair_index in range(self.dim // 2):
            row = 2 * pair_index
            col = row + 1
            matrix[..., row, row] = cos[..., pair_index]
            matrix[..., row, col] = -sin[..., pair_index]
            matrix[..., col, row] = sin[..., pair_index]
            matrix[..., col, col] = cos[..., pair_index]
        return matrix.squeeze(0) if matrix.ndim > 2 and matrix.shape[0] == 1 else matrix

    def generators(self) -> Tensor:
        generator_family = torch.zeros(
            self.ndim,
            self.dim,
            self.dim,
            dtype=self.frequency_matrix.dtype,
            device=self.frequency_matrix.device,
        )
        for axis in range(self.ndim):
            for pair_index in range(self.dim // 2):
                scale = self.frequency_matrix[pair_index, axis]
                row = 2 * pair_index
                col = row + 1
                generator_family[axis, row, col] = -scale
                generator_family[axis, col, row] = scale
        return generator_family


def apply_householder_stack(
    x: Tensor,
    V: Tensor,
    order: Order = "forward",
    *,
    eps: float = 1.0e-8,
    head_to_group: Tensor | None = None,
    fp32_norm_accumulation: bool = True,
    use_tau_parameterization: bool = False,
) -> Tensor:
    """Apply a reflector stack to vectors stored along the trailing dimension.

    `forward` applies reflectors in storage order `H_1, ..., H_M`, which realizes
    the dense product `Q = H_M ... H_1` under left-acting column-vector semantics.
    `reverse` applies `H_M, ..., H_1`, which realizes `Q^T`.
    """

    if order not in {"forward", "reverse"}:
        raise ValueError(f"Unsupported order '{order}'.")
    if V.dim() not in {2, 3}:
        raise ValueError(f"Expected V to have rank 2 or 3, received rank {V.dim()}.")
    if V.shape[-1] != x.shape[-1]:
        raise ValueError(
            f"Reflector dimension {V.shape[-1]} must match feature dimension {x.shape[-1]}."
        )
    if V.shape[-2] == 0:
        return x

    compute_dtype = _compute_dtype(x.dtype, fp32_norm_accumulation)
    z = x.to(compute_dtype)
    reflector_count = V.shape[-2]
    indices = range(reflector_count) if order == "forward" else range(reflector_count - 1, -1, -1)

    V_device = V.to(device=x.device, dtype=compute_dtype)
    if use_tau_parameterization:
        work_vectors = V_device
        tau = _householder_tau(
            V_device,
            eps=eps,
            fp32_norm_accumulation=fp32_norm_accumulation,
        ).to(dtype=compute_dtype)
    else:
        work_vectors = householder_normalize(
            V_device,
            eps=eps,
            fp32_norm_accumulation=fp32_norm_accumulation,
        ).to(dtype=compute_dtype)
        tau = torch.full(
            (*work_vectors.shape[:-1], 1),
            2.0,
            dtype=compute_dtype,
            device=x.device,
        )

    if work_vectors.dim() == 2:
        for index in indices:
            vector = work_vectors[index]
            scale = tau[index]
            projection = (z * vector).sum(dim=-1, keepdim=True)
            z = z - scale * projection * vector
        return z.to(x.dtype)

    if z.ndim < 3:
        raise ValueError(
            "Banked reflector stacks require tensors with a head axis at -3, "
            f"received shape {tuple(x.shape)}."
        )

    if head_to_group is not None:
        gather_index = head_to_group.to(device=x.device)
        work_vectors = work_vectors.index_select(0, gather_index)
        tau = tau.index_select(0, gather_index)
    elif z.shape[-3] != work_vectors.shape[0]:
        raise ValueError(
            "Banked reflector stacks require the head axis to match the reflector bank count. "
            f"Received head axis {z.shape[-3]} and bank count {work_vectors.shape[0]}."
        )

    for index in indices:
        vector = _broadcast_head_bank(work_vectors[:, index, :], z.ndim)
        scale = _broadcast_head_bank(tau[:, index, :], z.ndim)
        projection = (z * vector).sum(dim=-1, keepdim=True)
        z = z - scale * projection * vector
    return z.to(x.dtype)


def premix_qk(
    q: Tensor,
    k: Tensor,
    V: Tensor,
    *,
    eps: float = 1.0e-8,
    head_to_group: Tensor | None = None,
    fp32_norm_accumulation: bool = True,
    use_tau_parameterization: bool = False,
) -> tuple[Tensor, Tensor]:
    q_bar = apply_householder_stack(
        q,
        V,
        order="reverse",
        eps=eps,
        head_to_group=head_to_group,
        fp32_norm_accumulation=fp32_norm_accumulation,
        use_tau_parameterization=use_tau_parameterization,
    )
    k_bar = apply_householder_stack(
        k,
        V,
        order="reverse",
        eps=eps,
        head_to_group=head_to_group,
        fp32_norm_accumulation=fp32_norm_accumulation,
        use_tau_parameterization=use_tau_parameterization,
    )
    return q_bar, k_bar


def apply_householder_rope(
    q: Tensor,
    k: Tensor,
    pos: Tensor | float | int,
    rope_core: BlockDiagonalRoPECore,
    V: Tensor,
    *,
    eps: float = 1.0e-8,
    head_to_group: Tensor | None = None,
    fp32_norm_accumulation: bool = True,
    use_tau_parameterization: bool = False,
) -> tuple[Tensor, Tensor]:
    q_bar, k_bar = premix_qk(
        q,
        k,
        V,
        eps=eps,
        head_to_group=head_to_group,
        fp32_norm_accumulation=fp32_norm_accumulation,
        use_tau_parameterization=use_tau_parameterization,
    )
    return rope_core(q_bar, k_bar, pos)


def _materialize_single_stack(V: Tensor, eps: float) -> Tensor:
    dim = V.shape[-1]
    Q = torch.eye(dim, dtype=V.dtype, device=V.device)
    if V.shape[-2] == 0:
        return Q
    for vector in V:
        u = householder_normalize(vector, eps=eps, fp32_norm_accumulation=True).to(V.dtype)
        H = torch.eye(dim, dtype=V.dtype, device=V.device) - 2.0 * torch.outer(u, u)
        Q = H @ Q
    return Q


def materialize_Q(
    V: Tensor,
    *,
    eps: float = 1.0e-8,
    head_to_group: Tensor | None = None,
) -> Tensor:
    """Materialize dense orthogonal matrices for diagnostics and tests only."""

    if V.dim() == 2:
        return _materialize_single_stack(V, eps=eps)
    if V.dim() != 3:
        raise ValueError(f"Expected V to have rank 2 or 3, received rank {V.dim()}.")

    matrices = torch.stack(
        [_materialize_single_stack(V[index], eps=eps) for index in range(V.shape[0])],
        dim=0,
    )
    if head_to_group is None:
        return matrices
    return matrices.index_select(0, head_to_group.to(device=V.device))


def _conjugate_block_diagonal(Q: Tensor, D: Tensor) -> Tensor:
    if Q.dim() == 2 and D.dim() == 2:
        return Q @ D @ Q.transpose(-1, -2)
    if Q.dim() == 2 and D.dim() == 3:
        return torch.einsum("ai,tij,bj->tab", Q, D, Q)
    if Q.dim() == 3 and D.dim() == 2:
        return torch.einsum("hai,ij,hbj->hab", Q, D, Q)
    if Q.dim() == 3 and D.dim() == 3:
        return torch.einsum("hai,tij,hbj->htab", Q, D, Q)
    raise ValueError(
        f"Unsupported conjugation shapes Q={tuple(Q.shape)} and D={tuple(D.shape)}."
    )


@dataclass
class HouseholderRoPEConfig:
    enabled: bool = True
    mode: Mode = "per_head"
    num_reflectors: int = 4
    enforce_SO: bool = True
    init: Literal["paired_identity", "jittered_pairs", "random"] = "paired_identity"
    eps: float = 1.0e-8
    rope_ndim: int = 1
    base: float = 10000.0
    mixing_strategy: Literal["global", "frequency_banded"] = "global"
    local_band_pairs: int = 2
    materialize_q_for_debug: bool = False
    group_size: int = 2
    use_tau_parameterization: bool = False
    fp32_norm_accumulation: bool = True
    diagnostics: dict[str, bool] = field(
        default_factory=lambda: {
            "log_orthogonality_defect": True,
            "log_relativity_defect": True,
            "log_reversibility_defect": True,
            "log_commutator_defect": True,
            "log_block_mixing_energy": True,
        }
    )

    def validate(self, *, num_heads: int, head_dim: int) -> None:
        if head_dim % 2 != 0:
            raise ValueError(f"head_dim must be even for RoPE, received {head_dim}.")
        if self.enforce_SO and self.num_reflectors % 2 != 0:
            raise ValueError(
                "enforce_SO=True requires an even number of reflectors, "
                f"received {self.num_reflectors}."
            )
        if self.mode not in {"shared", "per_head", "group_shared"}:
            raise ValueError(f"Unsupported mode '{self.mode}'.")
        if self.mode == "group_shared":
            if self.group_size < 1:
                raise ValueError(f"group_size must be positive, received {self.group_size}.")
            if num_heads % self.group_size != 0:
                raise ValueError(
                    f"num_heads={num_heads} must be divisible by group_size={self.group_size}."
                )
        if self.init in {"paired_identity", "jittered_pairs"} and self.num_reflectors % 2 != 0:
            raise ValueError(
                f"Initialization '{self.init}' requires an even number of reflectors."
            )
        if self.rope_ndim < 1:
            raise ValueError(f"rope_ndim must be positive, received {self.rope_ndim}.")
        if self.mixing_strategy not in {"global", "frequency_banded"}:
            raise ValueError(f"Unsupported mixing_strategy '{self.mixing_strategy}'.")
        if self.local_band_pairs < 1:
            raise ValueError(f"local_band_pairs must be positive, received {self.local_band_pairs}.")

    def num_banks(self, num_heads: int) -> int:
        if self.mode == "shared":
            return 1
        if self.mode == "per_head":
            return num_heads
        return num_heads // self.group_size

    def build_head_to_bank(self, num_heads: int) -> Tensor:
        if self.mode == "shared":
            return torch.zeros(num_heads, dtype=torch.long)
        if self.mode == "per_head":
            return torch.arange(num_heads, dtype=torch.long)
        return torch.div(torch.arange(num_heads), self.group_size, rounding_mode="floor")


class HouseholderRoPE(nn.Module):
    """Householder basis transport wrapped around a standard RoPE core."""

    def __init__(
        self,
        *,
        num_heads: int,
        head_dim: int,
        config: HouseholderRoPEConfig | None = None,
        rope_core: BlockDiagonalRoPECore | None = None,
        frequency_matrix: Tensor | None = None,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.config = config or HouseholderRoPEConfig()
        self.config.validate(num_heads=num_heads, head_dim=head_dim)
        self.rope_core = rope_core or BlockDiagonalRoPECore(
            dim=head_dim,
            ndim=self.config.rope_ndim,
            base=self.config.base,
            frequency_matrix=frequency_matrix,
        )
        self.register_buffer(
            "head_to_bank",
            self.config.build_head_to_bank(num_heads),
            persistent=False,
        )

        reflector_shape = (
            (self.config.num_reflectors, head_dim)
            if self.config.mode == "shared"
            else (self.config.num_banks(num_heads), self.config.num_reflectors, head_dim)
        )
        self.reflectors = nn.Parameter(torch.empty(reflector_shape, dtype=torch.float32))
        self.register_buffer(
            "reflector_support_mask",
            self._build_reflector_support_mask(reflector_shape),
            persistent=False,
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        with torch.no_grad():
            if self.reflectors.numel() == 0:
                return
            if self.config.init == "random":
                self.reflectors.normal_()
                return

            pair_count = self.config.num_reflectors // 2
            base_shape = (*self.reflectors.shape[:-2], pair_count, self.head_dim)
            base_vectors = torch.randn(
                base_shape,
                dtype=self.reflectors.dtype,
                device=self.reflectors.device,
            )
            paired = torch.repeat_interleave(base_vectors, repeats=2, dim=-2)
            if self.config.init == "jittered_pairs":
                paired[..., 1::2, :] += 1.0e-2 * torch.randn_like(paired[..., 1::2, :])
            self.reflectors.copy_(paired)
            self.reflectors.mul_(self.reflector_support_mask.to(dtype=self.reflectors.dtype))

    def _build_reflector_support_mask(self, reflector_shape: tuple[int, ...]) -> Tensor:
        mask = torch.ones(reflector_shape, dtype=torch.float32)
        if self.config.mixing_strategy == "global" or self.config.num_reflectors == 0:
            return mask

        mask.zero_()
        num_pairs = self.head_dim // 2
        band_pairs = min(self.config.local_band_pairs, num_pairs)
        band_count = max(1, math.ceil(num_pairs / band_pairs))
        support_slot_count = max(1, math.ceil(self.config.num_reflectors / 2))

        for reflector_index in range(self.config.num_reflectors):
            support_slot = reflector_index // 2
            band_index = min((support_slot * band_count) // support_slot_count, band_count - 1)
            pair_start = band_index * band_pairs
            pair_stop = min(num_pairs, pair_start + band_pairs)
            dim_start = 2 * pair_start
            dim_stop = 2 * pair_stop
            if mask.dim() == 2:
                mask[reflector_index, dim_start:dim_stop] = 1.0
            else:
                mask[:, reflector_index, dim_start:dim_stop] = 1.0
        return mask

    def effective_reflectors(self, *, detach: bool = False) -> Tensor:
        reflectors = self.reflectors.detach() if detach else self.reflectors
        mask = self.reflector_support_mask.to(device=reflectors.device, dtype=reflectors.dtype)
        return reflectors * mask

    def premix_qk(self, q: Tensor, k: Tensor) -> tuple[Tensor, Tensor]:
        if not self.config.enabled:
            return q, k
        reflectors = self.effective_reflectors()
        return premix_qk(
            q,
            k,
            reflectors,
            eps=self.config.eps,
            head_to_group=None if self.config.mode == "per_head" else self.head_to_bank,
            fp32_norm_accumulation=self.config.fp32_norm_accumulation,
            use_tau_parameterization=self.config.use_tau_parameterization,
        )

    def forward(
        self,
        q: Tensor,
        k: Tensor,
        pos: Tensor | float | int,
    ) -> tuple[Tensor, Tensor]:
        if not self.config.enabled:
            return self.rope_core(q, k, pos)
        reflectors = self.effective_reflectors()
        return apply_householder_rope(
            q,
            k,
            pos,
            self.rope_core,
            reflectors,
            eps=self.config.eps,
            head_to_group=None if self.config.mode == "per_head" else self.head_to_bank,
            fp32_norm_accumulation=self.config.fp32_norm_accumulation,
            use_tau_parameterization=self.config.use_tau_parameterization,
        )

    def materialize_Q(self, *, expand_heads: bool = False) -> Tensor:
        Q = materialize_Q(self.effective_reflectors(detach=True), eps=self.config.eps)
        if not expand_heads:
            return Q
        if Q.dim() == 2:
            return Q.unsqueeze(0).expand(self.num_heads, -1, -1)
        if Q.shape[0] == self.num_heads:
            return Q
        return Q.index_select(0, self.head_to_bank)

    def materialize_rope(
        self,
        pos: Tensor | float | int,
        *,
        expand_heads: bool = False,
    ) -> Tensor:
        Q = self.materialize_Q(expand_heads=expand_heads)
        D = self.rope_core.materialize(pos).to(device=Q.device, dtype=Q.dtype)
        return _conjugate_block_diagonal(Q, D)
