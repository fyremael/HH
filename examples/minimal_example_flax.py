from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from householder_rope.flax_modules import FlaxHouseholderRoPE  # noqa: E402
from householder_rope.jax_core import JaxBlockDiagonalRoPECore, JaxHouseholderRoPE, JaxHouseholderRoPEConfig  # noqa: E402
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
    config = JaxHouseholderRoPEConfig(mode="per_head", num_reflectors=4, init="jittered_pairs", rope_ndim=2)
    module = FlaxHouseholderRoPE(
        num_heads=2,
        head_dim=8,
        config=config,
        rope_core=JaxBlockDiagonalRoPECore(dim=8, ndim=2),
    )

    q = jax.random.normal(jax.random.PRNGKey(0), (1, 2, 4, 8))
    k = jax.random.normal(jax.random.PRNGKey(1), (1, 2, 4, 8))
    pos = jnp.asarray([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])

    variables = module.init(jax.random.PRNGKey(2), q, k, pos)
    q_rot, k_rot = module.apply(variables, q, k, pos)
    pure = JaxHouseholderRoPE(
        num_heads=2,
        head_dim=8,
        config=config,
        rope_core=JaxBlockDiagonalRoPECore(dim=8, ndim=2),
        reflectors=variables["params"]["reflectors"],
    )
    diagnostics = summarize_householder_rope_diagnostics(
        pure,
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
