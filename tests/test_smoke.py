from __future__ import annotations

import random

import torch

from householder_rope import BlockDiagonalRoPECore, HouseholderRoPE, HouseholderRoPEConfig, HouseholderSelfAttention
from householder_rope.diagnostics import orthogonality_defect


def test_short_training_smoke_runs_without_nans() -> None:
    random.seed(0)
    torch.manual_seed(0)

    config = HouseholderRoPEConfig(
        mode="per_head",
        num_reflectors=4,
        init="jittered_pairs",
        rope_ndim=1,
    )
    rope = HouseholderRoPE(
        num_heads=2,
        head_dim=8,
        config=config,
        rope_core=BlockDiagonalRoPECore(dim=8, ndim=1),
    )
    attention = HouseholderSelfAttention(embed_dim=16, num_heads=2, rope=rope)
    optimizer = torch.optim.Adam(attention.parameters(), lr=1.0e-3)

    x = torch.randn(2, 5, 16)
    pos = torch.arange(5, dtype=torch.float32)
    target = torch.randn_like(x)

    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        output = attention(x, pos)
        loss = torch.nn.functional.mse_loss(output, target)
        assert torch.isfinite(loss)
        loss.backward()
        optimizer.step()

    Q = rope.materialize_Q(expand_heads=True)
    defect = orthogonality_defect(Q)
    assert defect.max().item() < 1.0e-6
