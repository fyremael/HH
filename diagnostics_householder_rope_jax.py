from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from householder_rope.jax_core import (  # noqa: E402
    JaxBlockDiagonalRoPECore,
    JaxHouseholderRoPE,
    JaxHouseholderRoPEConfig,
)
from householder_rope.jax_diagnostics import summarize_householder_rope_diagnostics  # noqa: E402


def _to_serializable(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _to_serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_serializable(item) for item in value]
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def main() -> None:
    key = jax.random.PRNGKey(0)
    key, rope_key, q_key, k_key = jax.random.split(key, 4)

    config = JaxHouseholderRoPEConfig(
        mode="group_shared",
        group_size=2,
        num_reflectors=4,
        init="jittered_pairs",
        rope_ndim=2,
    )
    rope = JaxHouseholderRoPE(
        num_heads=4,
        head_dim=8,
        config=config,
        rope_core=JaxBlockDiagonalRoPECore(dim=8, ndim=2),
        key=rope_key,
    )

    q = jax.random.normal(q_key, (1, 4, 4, 8))
    k = jax.random.normal(k_key, (1, 4, 4, 8))
    pos = jnp.asarray([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    summary = summarize_householder_rope_diagnostics(rope, pos_a=pos, pos_b=pos + 1.0, q=q, k=k)
    print(json.dumps(_to_serializable(summary), indent=2))


if __name__ == "__main__":
    main()
