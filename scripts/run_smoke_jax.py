from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from householder_rope.jax_attention import householder_attention  # noqa: E402
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
    random.seed(0)
    key = jax.random.PRNGKey(0)

    config = JaxHouseholderRoPEConfig(
        mode="per_head",
        num_reflectors=4,
        init="jittered_pairs",
        rope_ndim=2,
    )
    key, rope_key, q_key, k_key, v_key, target_key = jax.random.split(key, 6)
    rope = JaxHouseholderRoPE(
        num_heads=2,
        head_dim=8,
        config=config,
        rope_core=JaxBlockDiagonalRoPECore(dim=8, ndim=2),
        key=rope_key,
    )

    q = jax.random.normal(q_key, (2, 2, 4, 8))
    k = jax.random.normal(k_key, (2, 2, 4, 8))
    v = jax.random.normal(v_key, (2, 2, 4, 8))
    target = jax.random.normal(target_key, (2, 2, 4, 8))
    pos = jnp.asarray([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])

    reflectors = rope.reflectors
    loss_history = []

    def loss_fn(current_reflectors: jnp.ndarray) -> jnp.ndarray:
        current_rope = rope.replace_reflectors(current_reflectors)
        output, _ = householder_attention(q, k, v, pos, current_rope)
        return jnp.mean((output - target) ** 2)

    for _ in range(3):
        loss, grad = jax.value_and_grad(loss_fn)(reflectors)
        reflectors = reflectors - 1.0e-2 * grad
        loss_history.append(float(loss))

    trained_rope = rope.replace_reflectors(reflectors)
    diagnostics = summarize_householder_rope_diagnostics(
        trained_rope,
        pos_a=pos,
        pos_b=pos + 1.0,
        q=q,
        k=k,
        grad=grad,
    )
    serializable = _to_serializable(diagnostics)
    serializable["loss_history"] = loss_history

    output_dir = ROOT / "artifacts"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "householder_rope_jax_smoke_metrics.json"
    output_path.write_text(json.dumps(serializable, indent=2), encoding="utf-8")
    print(f"Wrote JAX smoke metrics to {output_path}")


if __name__ == "__main__":
    main()
