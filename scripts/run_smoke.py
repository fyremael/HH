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
    HouseholderSelfAttention,
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
    attention = HouseholderSelfAttention(embed_dim=16, num_heads=2, rope=rope)
    optimizer = torch.optim.Adam(attention.parameters(), lr=1.0e-3)

    x = torch.randn(2, 4, 16)
    target = torch.randn_like(x)
    pos = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])

    loss_history = []
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        output = attention(x, pos)
        loss = torch.nn.functional.mse_loss(output, target)
        loss.backward()
        optimizer.step()
        loss_history.append(float(loss.item()))

    q = torch.randn(2, 2, 4, 8)
    k = torch.randn(2, 2, 4, 8)
    diagnostics = summarize_householder_rope_diagnostics(
        rope,
        pos_a=pos,
        pos_b=pos + 1.0,
        q=q,
        k=k,
    )
    serializable = _to_serializable(diagnostics)
    serializable["loss_history"] = loss_history

    output_dir = ROOT / "artifacts"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "householder_rope_smoke_metrics.json"
    output_path.write_text(json.dumps(serializable, indent=2), encoding="utf-8")
    print(f"Wrote smoke metrics to {output_path}")


if __name__ == "__main__":
    main()
