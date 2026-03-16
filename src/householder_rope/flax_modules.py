from __future__ import annotations

from dataclasses import field

import flax.linen as nn
import jax.numpy as jnp

from .jax_core import (
    JaxBlockDiagonalRoPECore,
    JaxHouseholderRoPEConfig,
    apply_householder_rope,
    initialize_reflectors,
)


class FlaxHouseholderRoPE(nn.Module):
    """Flax-native Householder-RoPE wrapper."""

    num_heads: int
    head_dim: int
    config: JaxHouseholderRoPEConfig = field(default_factory=JaxHouseholderRoPEConfig)
    rope_core: JaxBlockDiagonalRoPECore | None = None
    frequency_matrix: jnp.ndarray | None = None
    param_dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(
        self,
        q: jnp.ndarray,
        k: jnp.ndarray,
        pos: jnp.ndarray | float | int,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        self.config.validate(num_heads=self.num_heads, head_dim=self.head_dim)
        rope_core = self.rope_core or JaxBlockDiagonalRoPECore(
            dim=self.head_dim,
            ndim=self.config.rope_ndim,
            base=self.config.base,
            frequency_matrix=self.frequency_matrix,
        )

        reflector_shape = (
            (self.config.num_reflectors, self.head_dim)
            if self.config.mode == "shared"
            else (self.config.num_banks(self.num_heads), self.config.num_reflectors, self.head_dim)
        )
        reflectors = self.param(
            "reflectors",
            lambda key, shape: initialize_reflectors(
                key=key,
                shape=shape,
                init=self.config.init,
                dtype=self.param_dtype,
            ),
            reflector_shape,
        )

        if not self.config.enabled:
            return rope_core(q, k, pos)

        head_to_group = None if self.config.mode == "per_head" else self.config.build_head_to_bank(self.num_heads)
        return apply_householder_rope(
            q,
            k,
            pos,
            rope_core,
            reflectors,
            eps=self.config.eps,
            head_to_group=head_to_group,
            fp32_norm_accumulation=self.config.fp32_norm_accumulation,
            use_tau_parameterization=self.config.use_tau_parameterization,
        )


class FlaxHouseholderSelfAttention(nn.Module):
    """Minimal Flax self-attention block with Householder-RoPE."""

    embed_dim: int
    num_heads: int
    rope_config: JaxHouseholderRoPEConfig = field(default_factory=JaxHouseholderRoPEConfig)
    dropout_rate: float = 0.0
    use_bias: bool = True
    param_dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        pos: jnp.ndarray | float | int,
        *,
        deterministic: bool = True,
        attn_mask: jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        if self.embed_dim % self.num_heads != 0:
            raise ValueError(
                f"embed_dim={self.embed_dim} must be divisible by num_heads={self.num_heads}."
            )

        head_dim = self.embed_dim // self.num_heads
        q = nn.Dense(self.embed_dim, use_bias=self.use_bias, param_dtype=self.param_dtype, name="q_proj")(x)
        k = nn.Dense(self.embed_dim, use_bias=self.use_bias, param_dtype=self.param_dtype, name="k_proj")(x)
        v = nn.Dense(self.embed_dim, use_bias=self.use_bias, param_dtype=self.param_dtype, name="v_proj")(x)

        batch, tokens, _ = x.shape
        q = jnp.transpose(jnp.reshape(q, (batch, tokens, self.num_heads, head_dim)), (0, 2, 1, 3))
        k = jnp.transpose(jnp.reshape(k, (batch, tokens, self.num_heads, head_dim)), (0, 2, 1, 3))
        v = jnp.transpose(jnp.reshape(v, (batch, tokens, self.num_heads, head_dim)), (0, 2, 1, 3))

        q, k = FlaxHouseholderRoPE(
            num_heads=self.num_heads,
            head_dim=head_dim,
            config=self.rope_config,
            param_dtype=self.param_dtype,
            name="rope",
        )(q, k, pos)

        scale = jnp.asarray(head_dim, dtype=q.dtype) ** -0.5
        logits = jnp.einsum("bhti,bhsi->bhts", q, k) * scale
        if attn_mask is not None:
            logits = logits + attn_mask
        weights = nn.softmax(logits, axis=-1)
        weights = nn.Dropout(rate=self.dropout_rate)(weights, deterministic=deterministic)
        attn_output = jnp.einsum("bhts,bhsi->bhti", weights, v)
        attn_output = jnp.reshape(jnp.transpose(attn_output, (0, 2, 1, 3)), (batch, tokens, self.embed_dim))
        return nn.Dense(self.embed_dim, use_bias=self.use_bias, param_dtype=self.param_dtype, name="out_proj")(attn_output)
