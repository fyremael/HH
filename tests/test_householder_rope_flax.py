from __future__ import annotations

import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
flax = pytest.importorskip("flax")
optax = pytest.importorskip("optax")

import jax.numpy as jnp
from jax import random

from householder_rope.flax_modules import FlaxHouseholderRoPE, FlaxHouseholderSelfAttention
from householder_rope.jax_core import JaxBlockDiagonalRoPECore, JaxHouseholderRoPE, JaxHouseholderRoPEConfig, materialize_Q
from householder_rope.jax_diagnostics import orthogonality_defect


def test_flax_householder_rope_matches_pure_jax_wrapper() -> None:
    config = JaxHouseholderRoPEConfig(
        mode="per_head",
        num_reflectors=4,
        init="jittered_pairs",
        rope_ndim=2,
    )
    module = FlaxHouseholderRoPE(
        num_heads=2,
        head_dim=8,
        config=config,
        rope_core=JaxBlockDiagonalRoPECore(dim=8, ndim=2),
    )
    q = random.normal(random.PRNGKey(0), (2, 2, 4, 8))
    k = random.normal(random.PRNGKey(1), (2, 2, 4, 8))
    pos = jnp.asarray([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])

    variables = module.init(random.PRNGKey(2), q, k, pos)
    q_flax, k_flax = module.apply(variables, q, k, pos)

    reflectors = variables["params"]["reflectors"]
    pure = JaxHouseholderRoPE(
        num_heads=2,
        head_dim=8,
        config=config,
        rope_core=JaxBlockDiagonalRoPECore(dim=8, ndim=2),
        reflectors=reflectors,
    )
    q_jax, k_jax = pure(q, k, pos)

    assert bool(jnp.allclose(q_flax, q_jax, atol=1.0e-6))
    assert bool(jnp.allclose(k_flax, k_jax, atol=1.0e-6))


def test_flax_paired_identity_initialization_materializes_identity() -> None:
    config = JaxHouseholderRoPEConfig(
        mode="shared",
        num_reflectors=4,
        init="paired_identity",
        rope_ndim=1,
    )
    module = FlaxHouseholderRoPE(
        num_heads=2,
        head_dim=8,
        config=config,
        rope_core=JaxBlockDiagonalRoPECore(dim=8, ndim=1),
    )
    q = random.normal(random.PRNGKey(3), (1, 2, 4, 8))
    k = random.normal(random.PRNGKey(4), (1, 2, 4, 8))
    pos = jnp.arange(4, dtype=jnp.float32)

    variables = module.init(random.PRNGKey(5), q, k, pos)
    reflectors = variables["params"]["reflectors"]
    Q = materialize_Q(reflectors).astype(jnp.float64)
    identity = jnp.eye(8, dtype=Q.dtype)
    assert bool(jnp.allclose(Q, identity, atol=1.0e-6))


def test_flax_attention_smoke_runs_without_nans() -> None:
    config = JaxHouseholderRoPEConfig(
        mode="per_head",
        num_reflectors=4,
        init="jittered_pairs",
        rope_ndim=1,
    )
    module = FlaxHouseholderSelfAttention(
        embed_dim=16,
        num_heads=2,
        rope_config=config,
        dropout_rate=0.0,
    )

    x = random.normal(random.PRNGKey(6), (2, 5, 16))
    pos = jnp.arange(5, dtype=jnp.float32)
    target = random.normal(random.PRNGKey(7), (2, 5, 16))

    variables = module.init(random.PRNGKey(8), x, pos, deterministic=True)
    params = variables["params"]
    optimizer = optax.adam(1.0e-3)
    opt_state = optimizer.init(params)

    def loss_fn(current_params):
        output = module.apply({"params": current_params}, x, pos, deterministic=True)
        return jnp.mean((output - target) ** 2)

    for _ in range(3):
        loss, grads = jax.value_and_grad(loss_fn)(params)
        assert bool(jnp.isfinite(loss))
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)

    output = module.apply({"params": params}, x, pos, deterministic=True)
    assert output.shape == x.shape
    defect = orthogonality_defect(materialize_Q(params["rope"]["reflectors"]))
    assert float(jnp.max(defect)) < 1.0e-6
