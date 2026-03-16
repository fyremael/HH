from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .core import HouseholderRoPE


class HouseholderSelfAttention(nn.Module):
    """Minimal self-attention block with a Householder-RoPE hook."""

    def __init__(
        self,
        *,
        embed_dim: int,
        num_heads: int,
        rope: HouseholderRoPE | None = None,
        dropout_p: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim={embed_dim} must be divisible by num_heads={num_heads}."
            )
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout_p = dropout_p
        self.rope = rope

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def _split_heads(self, x: Tensor) -> Tensor:
        batch, tokens, _ = x.shape
        return x.reshape(batch, tokens, self.num_heads, self.head_dim).transpose(1, 2)

    def _merge_heads(self, x: Tensor) -> Tensor:
        batch, heads, tokens, head_dim = x.shape
        return x.transpose(1, 2).reshape(batch, tokens, heads * head_dim)

    def forward(
        self,
        x: Tensor,
        pos: Tensor | float | int,
        *,
        attn_mask: Tensor | None = None,
        is_causal: bool = False,
    ) -> Tensor:
        q = self._split_heads(self.q_proj(x))
        k = self._split_heads(self.k_proj(x))
        v = self._split_heads(self.v_proj(x))
        if self.rope is not None:
            q, k = self.rope(q, k, pos)
        if hasattr(F, "scaled_dot_product_attention"):
            attn_output = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attn_mask,
                dropout_p=self.dropout_p if self.training else 0.0,
                is_causal=is_causal,
            )
        else:
            attn_output = _fallback_scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attn_mask,
                dropout_p=self.dropout_p if self.training else 0.0,
                is_causal=is_causal,
                training=self.training,
            )
        return self.out_proj(self._merge_heads(attn_output))


def _fallback_scaled_dot_product_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    *,
    attn_mask: Tensor | None,
    dropout_p: float,
    is_causal: bool,
    training: bool,
) -> Tensor:
    """Compatibility path for older Torch builds without fused SDPA."""

    scale = 1.0 / math.sqrt(q.shape[-1])
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale

    if is_causal:
        query_len = q.shape[-2]
        key_len = k.shape[-2]
        causal_mask = torch.ones(
            (query_len, key_len),
            dtype=torch.bool,
            device=q.device,
        ).triu(diagonal=1)
        scores = scores.masked_fill(causal_mask, float("-inf"))

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            scores = scores.masked_fill(~attn_mask, float("-inf"))
        else:
            scores = scores + attn_mask.to(device=scores.device, dtype=scores.dtype)

    weights = torch.softmax(scores, dim=-1)
    if dropout_p > 0.0:
        weights = F.dropout(weights, p=dropout_p, training=training)
    return torch.matmul(weights, v)
