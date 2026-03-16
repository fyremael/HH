from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from householder_rope import (  # noqa: E402
    BlockDiagonalRoPECore,
    HouseholderRoPE,
    HouseholderRoPEConfig,
    summarize_householder_rope_diagnostics,
)


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

    config = HouseholderRoPEConfig(mode="shared", num_reflectors=4, init="jittered_pairs", rope_ndim=2)
    rope = HouseholderRoPE(
        num_heads=2,
        head_dim=8,
        config=config,
        rope_core=BlockDiagonalRoPECore(dim=8, ndim=2),
    )

    q = torch.randn(1, 2, 4, 8)
    k = torch.randn(1, 2, 4, 8)
    pos = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])

    q_rot, k_rot = rope(q, k, pos)
    diagnostics = summarize_householder_rope_diagnostics(
        rope,
        pos_a=pos,
        pos_b=pos + 1.0,
        q=q,
        k=k,
    )
    print("q_rot shape:", tuple(q_rot.shape))
    print("k_rot shape:", tuple(k_rot.shape))
    print(json.dumps(_to_serializable(diagnostics), indent=2))


if __name__ == "__main__":
    main()
