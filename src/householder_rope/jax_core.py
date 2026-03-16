from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import jax
import jax.numpy as jnp

Array = jax.Array
Mode = Literal["shared", "per_head", "group_shared"]
Order = Literal["forward", "reverse"]


def _compute_dtype(dtype: jnp.dtype, fp32_norm_accumulation: bool) -> jnp.dtype:
    dtype = jnp.dtype(dtype)
    if fp32_norm_accumulation and dtype in (jnp.dtype(jnp.float16), jnp.dtype(jnp.bfloat16)):
        return jnp.float32
    return dtype


def householder_normalize(
    v: Array,
    eps: float = 1.0e-8,
    fp32_norm_accumulation: bool = True,
) -> Array:
    """Return a smoothly normalized reflector direction."""

    v = jnp.asarray(v)
    compute_dtype = _compute_dtype(v.dtype, fp32_norm_accumulation)
    v_compute = v.astype(compute_dtype)
    norm = jnp.linalg.norm(v_compute, axis=-1, keepdims=True)
    safe_norm = jnp.sqrt(jnp.square(norm) + eps * eps)
    return (v_compute / safe_norm).astype(v.dtype)


def _householder_tau(
    v: Array,
    eps: float,
    fp32_norm_accumulation: bool,
) -> Array:
    v = jnp.asarray(v)
    compute_dtype = _compute_dtype(v.dtype, fp32_norm_accumulation)
    v_compute = v.astype(compute_dtype)
    denom = jnp.sum(jnp.square(v_compute), axis=-1, keepdims=True) + eps * eps
    return (2.0 / denom).astype(v.dtype)


def _broadcast_head_bank(v: Array, target_ndim: int) -> Array:
    shape = [1] * target_ndim
    shape[-3] = v.shape[0]
    shape[-2] = 1
    shape[-1] = v.shape[1]
    return jnp.reshape(v, shape)


def build_default_frequency_matrix(
    dim: int,
    ndim: int,
    base: float = 10000.0,
    axis_allocation: Literal["contiguous", "round_robin"] = "contiguous",
) -> Array:
    if dim % 2 != 0:
        raise ValueError(f"RoPE head dimension must be even, received {dim}.")
    if ndim < 1:
        raise ValueError(f"RoPE ndim must be positive, received {ndim}.")

    num_pairs = dim // 2
    frequency_matrix = jnp.zeros((num_pairs, ndim), dtype=jnp.float32)
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
        local_index = jnp.arange(len(pair_indices), dtype=jnp.float32)
        inverse_frequency = base ** (-local_index / max(len(pair_indices), 1))
        frequency_matrix = frequency_matrix.at[jnp.asarray(pair_indices), axis].set(inverse_frequency)

    return frequency_matrix


class JaxBlockDiagonalRoPECore:
    """Canonical 1D or ND RoPE in a block-diagonal toral basis for JAX."""

    def __init__(
        self,
        dim: int,
        ndim: int = 1,
        *,
        base: float = 10000.0,
        frequency_matrix: Array | None = None,
        axis_allocation: Literal["contiguous", "round_robin"] = "contiguous",
    ) -> None:
        self.dim = dim
        self.ndim = ndim
        if frequency_matrix is None:
            frequency_matrix = build_default_frequency_matrix(
                dim=dim,
                ndim=ndim,
                base=base,
                axis_allocation=axis_allocation,
            )
        frequency_matrix = jnp.asarray(frequency_matrix, dtype=jnp.float32)
        expected_shape = (dim // 2, ndim)
        if tuple(frequency_matrix.shape) != expected_shape:
            raise ValueError(
                "frequency_matrix must have shape "
                f"{expected_shape}, received {tuple(frequency_matrix.shape)}."
            )
        self.frequency_matrix = frequency_matrix

    def _canonicalize_positions(self, pos: Array | float | int) -> Array:
        pos_tensor = jnp.asarray(pos, dtype=self.frequency_matrix.dtype)
        if self.ndim == 1:
            if pos_tensor.ndim == 0:
                pos_tensor = jnp.reshape(pos_tensor, (1, 1))
            elif pos_tensor.shape[-1] == 1:
                pos_tensor = jnp.reshape(pos_tensor, (*pos_tensor.shape[:-1], 1))
            else:
                pos_tensor = jnp.expand_dims(pos_tensor, axis=-1)
            return pos_tensor

        if pos_tensor.ndim == 1:
            if pos_tensor.size != self.ndim:
                raise ValueError(
                    f"Expected {self.ndim} coordinates, received {pos_tensor.size}."
                )
            pos_tensor = jnp.reshape(pos_tensor, (1, self.ndim))

        if pos_tensor.shape[-1] != self.ndim:
            raise ValueError(
                f"Expected trailing position dimension {self.ndim}, "
                f"received {pos_tensor.shape[-1]}."
            )
        return pos_tensor

    def angles(self, pos: Array | float | int) -> Array:
        pos_tensor = self._canonicalize_positions(pos)
        frequency_matrix = self.frequency_matrix.astype(pos_tensor.dtype)
        return jnp.einsum("...n,pn->...p", pos_tensor, frequency_matrix)

    def apply(self, x: Array, pos: Array | float | int) -> Array:
        x = jnp.asarray(x)
        if x.shape[-1] != self.dim:
            raise ValueError(
                f"Expected trailing feature dimension {self.dim}, "
                f"received {x.shape[-1]}."
            )

        angles = self.angles(pos).astype(x.dtype)
        x_pairs = jnp.reshape(x, (*x.shape[:-1], self.dim // 2, 2))
        while angles.ndim < x_pairs.ndim - 1:
            angles = jnp.expand_dims(angles, axis=-3)

        cos = jnp.cos(angles).astype(x.dtype)
        sin = jnp.sin(angles).astype(x.dtype)
        x0 = x_pairs[..., 0]
        x1 = x_pairs[..., 1]
        y0 = x0 * cos - x1 * sin
        y1 = x0 * sin + x1 * cos
        return jnp.reshape(jnp.stack((y0, y1), axis=-1), x.shape)

    def __call__(
        self,
        q: Array,
        k: Array,
        pos: Array | float | int,
    ) -> tuple[Array, Array]:
        return self.apply(q, pos), self.apply(k, pos)

    def materialize(self, pos: Array | float | int) -> Array:
        angles = self.angles(pos)
        matrix = jnp.zeros((*angles.shape[:-1], self.dim, self.dim), dtype=angles.dtype)
        cos = jnp.cos(angles)
        sin = jnp.sin(angles)
        for pair_index in range(self.dim // 2):
            row = 2 * pair_index
            col = row + 1
            matrix = matrix.at[..., row, row].set(cos[..., pair_index])
            matrix = matrix.at[..., row, col].set(-sin[..., pair_index])
            matrix = matrix.at[..., col, row].set(sin[..., pair_index])
            matrix = matrix.at[..., col, col].set(cos[..., pair_index])
        if matrix.ndim > 2 and matrix.shape[0] == 1:
            return jnp.squeeze(matrix, axis=0)
        return matrix

    def generators(self) -> Array:
        generator_family = jnp.zeros(
            (self.ndim, self.dim, self.dim),
            dtype=self.frequency_matrix.dtype,
        )
        for axis in range(self.ndim):
            for pair_index in range(self.dim // 2):
                scale = self.frequency_matrix[pair_index, axis]
                row = 2 * pair_index
                col = row + 1
                generator_family = generator_family.at[axis, row, col].set(-scale)
                generator_family = generator_family.at[axis, col, row].set(scale)
        return generator_family


def apply_householder_stack(
    x: Array,
    V: Array,
    order: Order = "forward",
    *,
    eps: float = 1.0e-8,
    head_to_group: Array | None = None,
    fp32_norm_accumulation: bool = True,
    use_tau_parameterization: bool = False,
) -> Array:
    """Apply a reflector stack to vectors stored along the trailing dimension."""

    x = jnp.asarray(x)
    V = jnp.asarray(V)
    if order not in {"forward", "reverse"}:
        raise ValueError(f"Unsupported order '{order}'.")
    if V.ndim not in {2, 3}:
        raise ValueError(f"Expected V to have rank 2 or 3, received rank {V.ndim}.")
    if V.shape[-1] != x.shape[-1]:
        raise ValueError(
            f"Reflector dimension {V.shape[-1]} must match feature dimension {x.shape[-1]}."
        )
    if V.shape[-2] == 0:
        return x

    compute_dtype = _compute_dtype(x.dtype, fp32_norm_accumulation)
    z = x.astype(compute_dtype)
    reflector_count = V.shape[-2]
    indices = range(reflector_count) if order == "forward" else range(reflector_count - 1, -1, -1)

    V_device = V.astype(compute_dtype)
    if use_tau_parameterization:
        work_vectors = V_device
        tau = _householder_tau(
            V_device,
            eps=eps,
            fp32_norm_accumulation=fp32_norm_accumulation,
        ).astype(compute_dtype)
    else:
        work_vectors = householder_normalize(
            V_device,
            eps=eps,
            fp32_norm_accumulation=fp32_norm_accumulation,
        ).astype(compute_dtype)
        tau = jnp.full((*work_vectors.shape[:-1], 1), 2.0, dtype=compute_dtype)

    if work_vectors.ndim == 2:
        for index in indices:
            vector = work_vectors[index]
            scale = tau[index]
            projection = jnp.sum(z * vector, axis=-1, keepdims=True)
            z = z - scale * projection * vector
        return z.astype(x.dtype)

    if z.ndim < 3:
        raise ValueError(
            "Banked reflector stacks require tensors with a head axis at -3, "
            f"received shape {tuple(x.shape)}."
        )

    if head_to_group is not None:
        gather_index = jnp.asarray(head_to_group)
        work_vectors = work_vectors[gather_index]
        tau = tau[gather_index]
    elif z.shape[-3] != work_vectors.shape[0]:
        raise ValueError(
            "Banked reflector stacks require the head axis to match the reflector bank count. "
            f"Received head axis {z.shape[-3]} and bank count {work_vectors.shape[0]}."
        )

    for index in indices:
        vector = _broadcast_head_bank(work_vectors[:, index, :], z.ndim)
        scale = _broadcast_head_bank(tau[:, index, :], z.ndim)
        projection = jnp.sum(z * vector, axis=-1, keepdims=True)
        z = z - scale * projection * vector
    return z.astype(x.dtype)


def premix_qk(
    q: Array,
    k: Array,
    V: Array,
    *,
    eps: float = 1.0e-8,
    head_to_group: Array | None = None,
    fp32_norm_accumulation: bool = True,
    use_tau_parameterization: bool = False,
) -> tuple[Array, Array]:
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
    q: Array,
    k: Array,
    pos: Array | float | int,
    rope_core: JaxBlockDiagonalRoPECore,
    V: Array,
    *,
    eps: float = 1.0e-8,
    head_to_group: Array | None = None,
    fp32_norm_accumulation: bool = True,
    use_tau_parameterization: bool = False,
) -> tuple[Array, Array]:
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


def _materialize_single_stack(V: Array, eps: float) -> Array:
    dim = V.shape[-1]
    Q = jnp.eye(dim, dtype=V.dtype)
    if V.shape[-2] == 0:
        return Q
    for vector in V:
        u = householder_normalize(vector, eps=eps, fp32_norm_accumulation=True).astype(V.dtype)
        H = jnp.eye(dim, dtype=V.dtype) - 2.0 * jnp.outer(u, u)
        Q = H @ Q
    return Q


def materialize_Q(
    V: Array,
    *,
    eps: float = 1.0e-8,
    head_to_group: Array | None = None,
) -> Array:
    """Materialize dense orthogonal matrices for diagnostics and tests only."""

    V = jnp.asarray(V)
    if V.ndim == 2:
        return _materialize_single_stack(V, eps=eps)
    if V.ndim != 3:
        raise ValueError(f"Expected V to have rank 2 or 3, received rank {V.ndim}.")

    matrices = jnp.stack(
        [_materialize_single_stack(V[index], eps=eps) for index in range(V.shape[0])],
        axis=0,
    )
    if head_to_group is None:
        return matrices
    return matrices[jnp.asarray(head_to_group)]


def _conjugate_block_diagonal(Q: Array, D: Array) -> Array:
    if Q.ndim == 2 and D.ndim == 2:
        return Q @ D @ jnp.swapaxes(Q, -1, -2)
    if Q.ndim == 2 and D.ndim == 3:
        return jnp.einsum("ai,tij,bj->tab", Q, D, Q)
    if Q.ndim == 3 and D.ndim == 2:
        return jnp.einsum("hai,ij,hbj->hab", Q, D, Q)
    if Q.ndim == 3 and D.ndim == 3:
        return jnp.einsum("hai,tij,hbj->htab", Q, D, Q)
    raise ValueError(
        f"Unsupported conjugation shapes Q={tuple(Q.shape)} and D={tuple(D.shape)}."
    )


@dataclass
class JaxHouseholderRoPEConfig:
    enabled: bool = True
    mode: Mode = "per_head"
    num_reflectors: int = 4
    enforce_SO: bool = True
    init: Literal["paired_identity", "jittered_pairs", "random"] = "paired_identity"
    eps: float = 1.0e-8
    rope_ndim: int = 1
    base: float = 10000.0
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

    def num_banks(self, num_heads: int) -> int:
        if self.mode == "shared":
            return 1
        if self.mode == "per_head":
            return num_heads
        return num_heads // self.group_size

    def build_head_to_bank(self, num_heads: int) -> Array:
        if self.mode == "shared":
            return jnp.zeros((num_heads,), dtype=jnp.int32)
        if self.mode == "per_head":
            return jnp.arange(num_heads, dtype=jnp.int32)
        return jnp.floor_divide(jnp.arange(num_heads, dtype=jnp.int32), self.group_size)


def initialize_reflectors(
    *,
    key: Array,
    shape: tuple[int, ...],
    init: Literal["paired_identity", "jittered_pairs", "random"],
    dtype: jnp.dtype = jnp.float32,
) -> Array:
    if init == "random":
        return jax.random.normal(key, shape, dtype=dtype)

    pair_count = shape[-2] // 2
    base_shape = (*shape[:-2], pair_count, shape[-1])
    base_key, jitter_key = jax.random.split(key)
    base_vectors = jax.random.normal(base_key, base_shape, dtype=dtype)
    paired = jnp.repeat(base_vectors, repeats=2, axis=-2)
    if init == "jittered_pairs":
        jitter = 1.0e-2 * jax.random.normal(jitter_key, paired[..., 1::2, :].shape, dtype=dtype)
        paired = paired.at[..., 1::2, :].add(jitter)
    return paired


class JaxHouseholderRoPE:
    """JAX companion implementation for Householder basis transport."""

    def __init__(
        self,
        *,
        num_heads: int,
        head_dim: int,
        config: JaxHouseholderRoPEConfig | None = None,
        rope_core: JaxBlockDiagonalRoPECore | None = None,
        frequency_matrix: Array | None = None,
        reflectors: Array | None = None,
        key: Array | None = None,
    ) -> None:
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.config = config or JaxHouseholderRoPEConfig()
        self.config.validate(num_heads=num_heads, head_dim=head_dim)
        self.rope_core = rope_core or JaxBlockDiagonalRoPECore(
            dim=head_dim,
            ndim=self.config.rope_ndim,
            base=self.config.base,
            frequency_matrix=frequency_matrix,
        )
        self.head_to_bank = self.config.build_head_to_bank(num_heads)

        reflector_shape = (
            (self.config.num_reflectors, head_dim)
            if self.config.mode == "shared"
            else (self.config.num_banks(num_heads), self.config.num_reflectors, head_dim)
        )
        if reflectors is None:
            key = jax.random.PRNGKey(0) if key is None else key
            reflectors = initialize_reflectors(
                key=key,
                shape=reflector_shape,
                init=self.config.init,
                dtype=jnp.float32,
            )
        self.reflectors = jnp.asarray(reflectors, dtype=jnp.float32)

    def replace_reflectors(self, reflectors: Array) -> JaxHouseholderRoPE:
        return JaxHouseholderRoPE(
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            config=self.config,
            rope_core=self.rope_core,
            reflectors=jnp.asarray(reflectors, dtype=jnp.float32),
        )

    def premix_qk(self, q: Array, k: Array) -> tuple[Array, Array]:
        if not self.config.enabled:
            return jnp.asarray(q), jnp.asarray(k)
        return premix_qk(
            q,
            k,
            self.reflectors,
            eps=self.config.eps,
            head_to_group=None if self.config.mode == "per_head" else self.head_to_bank,
            fp32_norm_accumulation=self.config.fp32_norm_accumulation,
            use_tau_parameterization=self.config.use_tau_parameterization,
        )

    def __call__(
        self,
        q: Array,
        k: Array,
        pos: Array | float | int,
    ) -> tuple[Array, Array]:
        if not self.config.enabled:
            return self.rope_core(q, k, pos)
        return apply_householder_rope(
            q,
            k,
            pos,
            self.rope_core,
            self.reflectors,
            eps=self.config.eps,
            head_to_group=None if self.config.mode == "per_head" else self.head_to_bank,
            fp32_norm_accumulation=self.config.fp32_norm_accumulation,
            use_tau_parameterization=self.config.use_tau_parameterization,
        )

    def materialize_Q(self, *, expand_heads: bool = False) -> Array:
        Q = materialize_Q(self.reflectors, eps=self.config.eps)
        if not expand_heads:
            return Q
        if Q.ndim == 2:
            return jnp.broadcast_to(Q[None, ...], (self.num_heads, *Q.shape))
        if Q.shape[0] == self.num_heads:
            return Q
        return Q[self.head_to_bank]

    def materialize_rope(
        self,
        pos: Array | float | int,
        *,
        expand_heads: bool = False,
    ) -> Array:
        Q = self.materialize_Q(expand_heads=expand_heads)
        D = self.rope_core.materialize(pos).astype(Q.dtype)
        return _conjugate_block_diagonal(Q, D)
