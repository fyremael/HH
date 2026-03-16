from __future__ import annotations

import numpy as np
import pytest
import torch

jax = pytest.importorskip("jax")
flax = pytest.importorskip("flax")

import jax.numpy as jnp

from householder_rope.attention_benchmark import (
    AttentionBenchmarkScenario,
    build_attention_case,
    build_flax_attention_block,
    build_jax_context,
    build_torch_attention_block,
    flax_variables_from_case,
    jax_attention_block_forward,
    jax_sgd_step,
    make_flax_loss_fn,
    make_jax_loss_fn,
    torch_single_step_losses,
)


def test_attention_block_forward_alignment_across_backends() -> None:
    scenario = AttentionBenchmarkScenario(
        name="forward_alignment",
        mode="per_head",
        group_size=2,
        batch=2,
        num_heads=2,
        tokens=8,
        head_dim=8,
        num_reflectors=4,
        rope_ndim=1,
        seed=123,
    )
    case = build_attention_case(scenario)

    torch_module, x_torch, pos_torch, _ = build_torch_attention_block(case, device=torch.device("cpu"))
    with torch.no_grad():
        torch_output = torch_module(x_torch, pos_torch).detach().cpu().numpy()

    jax_params, rope_core, rope_config, head_to_group = build_jax_context(case)
    x_jax = jnp.asarray(case.x)
    pos_jax = jnp.asarray(case.pos)
    jax_output = jax_attention_block_forward(
        jax_params,
        x_jax,
        pos_jax,
        num_heads=scenario.num_heads,
        head_dim=scenario.head_dim,
        rope_core=rope_core,
        rope_config=rope_config,
        head_to_group=head_to_group,
    )

    flax_module = build_flax_attention_block(case)
    flax_output = flax_module.apply(flax_variables_from_case(case), x_jax, pos_jax, deterministic=True)

    assert np.allclose(torch_output, np.asarray(jax_output), atol=1.0e-5)
    assert np.allclose(np.asarray(jax_output), np.asarray(flax_output), atol=1.0e-6)


def test_attention_block_single_step_loss_alignment_across_backends() -> None:
    scenario = AttentionBenchmarkScenario(
        name="step_alignment",
        mode="group_shared",
        group_size=2,
        batch=2,
        num_heads=4,
        tokens=8,
        head_dim=8,
        num_reflectors=4,
        rope_ndim=2,
        seed=456,
        learning_rate=1.0e-3,
    )
    case = build_attention_case(scenario)

    torch_module, x_torch, pos_torch, target_torch = build_torch_attention_block(case, device=torch.device("cpu"))
    torch_loss_before, torch_loss_after = torch_single_step_losses(
        torch_module,
        x_torch,
        pos_torch,
        target_torch,
        lr=scenario.learning_rate,
    )

    jax_params, rope_core, rope_config, head_to_group = build_jax_context(case)
    x_jax = jnp.asarray(case.x)
    pos_jax = jnp.asarray(case.pos)
    target_jax = jnp.asarray(case.target)
    jax_loss_fn = make_jax_loss_fn(
        num_heads=scenario.num_heads,
        head_dim=scenario.head_dim,
        rope_core=rope_core,
        rope_config=rope_config,
        head_to_group=head_to_group,
    )
    updated_jax_params, jax_loss_before = jax_sgd_step(
        jax_params,
        x_jax,
        pos_jax,
        target_jax,
        lr=scenario.learning_rate,
        loss_fn=jax_loss_fn,
    )
    jax_loss_after = float(jax_loss_fn(updated_jax_params, x_jax, pos_jax, target_jax))

    flax_module = build_flax_attention_block(case)
    flax_params = flax_variables_from_case(case)["params"]
    flax_loss_fn = make_flax_loss_fn(flax_module)
    updated_flax_params, flax_loss_before = jax_sgd_step(
        flax_params,
        x_jax,
        pos_jax,
        target_jax,
        lr=scenario.learning_rate,
        loss_fn=flax_loss_fn,
    )
    flax_loss_after = float(flax_loss_fn(updated_flax_params, x_jax, pos_jax, target_jax))

    assert abs(torch_loss_before - float(jax_loss_before)) < 1.0e-5
    assert abs(torch_loss_after - jax_loss_after) < 1.0e-5
    assert abs(float(jax_loss_before) - float(flax_loss_before)) < 1.0e-6
    assert abs(jax_loss_after - flax_loss_after) < 1.0e-6
