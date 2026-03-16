from __future__ import annotations

import json
import random
from typing import Any

import torch

from .core import BlockDiagonalRoPECore, HouseholderRoPE, HouseholderRoPEConfig
from .diagnostics import summarize_householder_rope_diagnostics


def _to_serializable(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.tolist()
    if isinstance(value, dict):
        return {key: _to_serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_serializable(item) for item in value]
    return value


def main() -> None:
    random.seed(0)
    torch.manual_seed(0)

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

    q = torch.randn(2, 2, 4, 8)
    k = torch.randn(2, 2, 4, 8)
    pos = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    summary = summarize_householder_rope_diagnostics(rope, pos_a=pos, pos_b=pos + 1.0, q=q, k=k)
    print(json.dumps(_to_serializable(summary), indent=2))


if __name__ == "__main__":
    main()
