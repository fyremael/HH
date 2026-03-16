from __future__ import annotations

import jax
import jax.numpy as jnp

from .jax_core import JaxHouseholderRoPE


def scaled_dot_product_attention(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    *,
    mask: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array]:
    scale = jnp.asarray(q.shape[-1], dtype=q.dtype) ** -0.5
    logits = jnp.einsum("bhti,bhsi->bhts", q, k) * scale
    if mask is not None:
        logits = logits + mask
    weights = jax.nn.softmax(logits, axis=-1)
    output = jnp.einsum("bhts,bhsi->bhti", weights, v)
    return output, weights


def householder_attention(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    pos: jax.Array | float | int,
    rope: JaxHouseholderRoPE,
    *,
    mask: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array]:
    q_rot, k_rot = rope(q, k, pos)
    return scaled_dot_product_attention(q_rot, k_rot, v, mask=mask)
