from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import torch
from flax.core import FrozenDict, freeze

from .attention import HouseholderSelfAttention
from .core import BlockDiagonalRoPECore, HouseholderRoPE, HouseholderRoPEConfig
from .flax_modules import FlaxHouseholderSelfAttention
from .jax_core import JaxBlockDiagonalRoPECore, JaxHouseholderRoPEConfig, apply_householder_rope


@dataclass(frozen=True)
class AttentionBenchmarkScenario:
    """Scenario description for full attention-block benchmarking."""

    name: str
    mode: str
    group_size: int
    batch: int
    num_heads: int
    tokens: int
    head_dim: int
    num_reflectors: int
    rope_ndim: int
    seed: int
    learning_rate: float = 1.0e-3

    @property
    def embed_dim(self) -> int:
        return self.num_heads * self.head_dim

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["embed_dim"] = self.embed_dim
        return payload


@dataclass(frozen=True)
class AttentionBenchmarkCase:
    """Shared inputs and parameters for aligned backend benchmarks."""

    scenario: AttentionBenchmarkScenario
    x: np.ndarray
    target: np.ndarray
    pos: np.ndarray
    reflectors: np.ndarray
    frequency_matrix: np.ndarray
    q_kernel: np.ndarray
    q_bias: np.ndarray
    k_kernel: np.ndarray
    k_bias: np.ndarray
    v_kernel: np.ndarray
    v_bias: np.ndarray
    out_kernel: np.ndarray
    out_bias: np.ndarray


def build_frequency_matrix_np(dim: int, ndim: int, base: float = 10000.0) -> np.ndarray:
    if dim % 2 != 0:
        raise ValueError(f"dim must be even, received {dim}.")
    num_pairs = dim // 2
    frequency_matrix = np.zeros((num_pairs, ndim), dtype=np.float32)
    base_block, remainder = divmod(num_pairs, ndim)
    start = 0
    for axis in range(ndim):
        block_size = base_block + int(axis < remainder)
        stop = start + block_size
        if stop > start:
            local_index = np.arange(stop - start, dtype=np.float32)
            frequency_matrix[start:stop, axis] = base ** (-local_index / max(stop - start, 1))
        start = stop
    return frequency_matrix


def build_positions_np(tokens: int, ndim: int) -> np.ndarray:
    if ndim == 1:
        return np.arange(tokens, dtype=np.float32)
    side = int(math.ceil(tokens ** (1.0 / ndim)))
    shape = (side,) * ndim
    coords = np.stack(np.unravel_index(np.arange(np.prod(shape)), shape), axis=-1)
    return coords[:tokens].astype(np.float32)


def reflector_shape(scenario: AttentionBenchmarkScenario) -> tuple[int, ...]:
    if scenario.mode == "shared":
        return (scenario.num_reflectors, scenario.head_dim)
    if scenario.mode == "per_head":
        return (scenario.num_heads, scenario.num_reflectors, scenario.head_dim)
    return (scenario.num_heads // scenario.group_size, scenario.num_reflectors, scenario.head_dim)


def _sample_projection(rng: np.random.Generator, embed_dim: int) -> tuple[np.ndarray, np.ndarray]:
    scale = 1.0 / math.sqrt(embed_dim)
    kernel = rng.standard_normal((embed_dim, embed_dim), dtype=np.float32) * scale
    bias = rng.standard_normal((embed_dim,), dtype=np.float32) * scale
    return kernel, bias


def build_attention_case(scenario: AttentionBenchmarkScenario) -> AttentionBenchmarkCase:
    rng = np.random.default_rng(scenario.seed)
    x = rng.standard_normal((scenario.batch, scenario.tokens, scenario.embed_dim), dtype=np.float32)
    target = rng.standard_normal((scenario.batch, scenario.tokens, scenario.embed_dim), dtype=np.float32)
    reflectors = rng.standard_normal(reflector_shape(scenario), dtype=np.float32)
    q_kernel, q_bias = _sample_projection(rng, scenario.embed_dim)
    k_kernel, k_bias = _sample_projection(rng, scenario.embed_dim)
    v_kernel, v_bias = _sample_projection(rng, scenario.embed_dim)
    out_kernel, out_bias = _sample_projection(rng, scenario.embed_dim)
    return AttentionBenchmarkCase(
        scenario=scenario,
        x=x,
        target=target,
        pos=build_positions_np(scenario.tokens, scenario.rope_ndim),
        reflectors=reflectors,
        frequency_matrix=build_frequency_matrix_np(scenario.head_dim, scenario.rope_ndim),
        q_kernel=q_kernel,
        q_bias=q_bias,
        k_kernel=k_kernel,
        k_bias=k_bias,
        v_kernel=v_kernel,
        v_bias=v_bias,
        out_kernel=out_kernel,
        out_bias=out_bias,
    )


def torch_rope_config(scenario: AttentionBenchmarkScenario) -> HouseholderRoPEConfig:
    return HouseholderRoPEConfig(
        mode=scenario.mode,
        group_size=scenario.group_size,
        num_reflectors=scenario.num_reflectors,
        init="random",
        rope_ndim=scenario.rope_ndim,
        enforce_SO=False,
    )


def jax_rope_config(scenario: AttentionBenchmarkScenario) -> JaxHouseholderRoPEConfig:
    return JaxHouseholderRoPEConfig(
        mode=scenario.mode,
        group_size=scenario.group_size,
        num_reflectors=scenario.num_reflectors,
        init="random",
        rope_ndim=scenario.rope_ndim,
        enforce_SO=False,
    )


def _copy_torch_linear(linear: torch.nn.Linear, kernel: np.ndarray, bias: np.ndarray) -> None:
    with torch.no_grad():
        linear.weight.copy_(torch.from_numpy(kernel.T).to(device=linear.weight.device, dtype=linear.weight.dtype))
        linear.bias.copy_(torch.from_numpy(bias).to(device=linear.bias.device, dtype=linear.bias.dtype))


def build_torch_attention_block(
    case: AttentionBenchmarkCase,
    *,
    device: torch.device,
) -> tuple[HouseholderSelfAttention, torch.Tensor, torch.Tensor, torch.Tensor]:
    scenario = case.scenario
    rope_core = BlockDiagonalRoPECore(
        dim=scenario.head_dim,
        ndim=scenario.rope_ndim,
        frequency_matrix=torch.from_numpy(case.frequency_matrix).to(device=device),
    )
    rope = HouseholderRoPE(
        num_heads=scenario.num_heads,
        head_dim=scenario.head_dim,
        config=torch_rope_config(scenario),
        rope_core=rope_core,
    ).to(device)
    with torch.no_grad():
        rope.reflectors.copy_(torch.from_numpy(case.reflectors).to(device=device, dtype=rope.reflectors.dtype))

    module = HouseholderSelfAttention(
        embed_dim=scenario.embed_dim,
        num_heads=scenario.num_heads,
        rope=rope,
        dropout_p=0.0,
    ).to(device)
    _copy_torch_linear(module.q_proj, case.q_kernel, case.q_bias)
    _copy_torch_linear(module.k_proj, case.k_kernel, case.k_bias)
    _copy_torch_linear(module.v_proj, case.v_kernel, case.v_bias)
    _copy_torch_linear(module.out_proj, case.out_kernel, case.out_bias)

    x = torch.from_numpy(case.x).to(device=device)
    pos = torch.from_numpy(case.pos).to(device=device)
    target = torch.from_numpy(case.target).to(device=device)
    return module, x, pos, target


def build_jax_context(
    case: AttentionBenchmarkCase,
) -> tuple[dict[str, Any], JaxBlockDiagonalRoPECore, JaxHouseholderRoPEConfig, jnp.ndarray | None]:
    scenario = case.scenario
    config = jax_rope_config(scenario)
    rope_core = JaxBlockDiagonalRoPECore(
        dim=scenario.head_dim,
        ndim=scenario.rope_ndim,
        frequency_matrix=jnp.asarray(case.frequency_matrix),
    )
    head_to_group = None if scenario.mode == "per_head" else config.build_head_to_bank(scenario.num_heads)
    params = {
        "q_proj": {
            "kernel": jnp.asarray(case.q_kernel),
            "bias": jnp.asarray(case.q_bias),
        },
        "k_proj": {
            "kernel": jnp.asarray(case.k_kernel),
            "bias": jnp.asarray(case.k_bias),
        },
        "v_proj": {
            "kernel": jnp.asarray(case.v_kernel),
            "bias": jnp.asarray(case.v_bias),
        },
        "out_proj": {
            "kernel": jnp.asarray(case.out_kernel),
            "bias": jnp.asarray(case.out_bias),
        },
        "rope": {
            "reflectors": jnp.asarray(case.reflectors),
        },
    }
    return params, rope_core, config, head_to_group


def flax_variables_from_case(case: AttentionBenchmarkCase) -> FrozenDict[str, Any]:
    params, _, _, _ = build_jax_context(case)
    return freeze({"params": params})


def build_flax_attention_block(case: AttentionBenchmarkCase) -> FlaxHouseholderSelfAttention:
    scenario = case.scenario
    return FlaxHouseholderSelfAttention(
        embed_dim=scenario.embed_dim,
        num_heads=scenario.num_heads,
        rope_config=jax_rope_config(scenario),
        dropout_rate=0.0,
    )


def split_heads(x: jnp.ndarray, *, num_heads: int, head_dim: int) -> jnp.ndarray:
    batch, tokens, _ = x.shape
    return jnp.transpose(jnp.reshape(x, (batch, tokens, num_heads, head_dim)), (0, 2, 1, 3))


def merge_heads(x: jnp.ndarray) -> jnp.ndarray:
    batch, heads, tokens, head_dim = x.shape
    return jnp.reshape(jnp.transpose(x, (0, 2, 1, 3)), (batch, tokens, heads * head_dim))


def dense_apply(x: jnp.ndarray, kernel: jnp.ndarray, bias: jnp.ndarray) -> jnp.ndarray:
    return jnp.einsum("btd,df->btf", x, kernel) + bias


def jax_attention_block_forward(
    params: dict[str, Any],
    x: jnp.ndarray,
    pos: jnp.ndarray,
    *,
    num_heads: int,
    head_dim: int,
    rope_core: JaxBlockDiagonalRoPECore,
    rope_config: JaxHouseholderRoPEConfig,
    head_to_group: jnp.ndarray | None,
) -> jnp.ndarray:
    q = dense_apply(x, params["q_proj"]["kernel"], params["q_proj"]["bias"])
    k = dense_apply(x, params["k_proj"]["kernel"], params["k_proj"]["bias"])
    v = dense_apply(x, params["v_proj"]["kernel"], params["v_proj"]["bias"])

    q = split_heads(q, num_heads=num_heads, head_dim=head_dim)
    k = split_heads(k, num_heads=num_heads, head_dim=head_dim)
    v = split_heads(v, num_heads=num_heads, head_dim=head_dim)

    q, k = apply_householder_rope(
        q,
        k,
        pos,
        rope_core,
        params["rope"]["reflectors"],
        eps=rope_config.eps,
        head_to_group=head_to_group,
        fp32_norm_accumulation=rope_config.fp32_norm_accumulation,
        use_tau_parameterization=rope_config.use_tau_parameterization,
    )

    scale = jnp.asarray(head_dim, dtype=q.dtype) ** -0.5
    logits = jnp.einsum("bhti,bhsi->bhts", q, k) * scale
    weights = jax.nn.softmax(logits, axis=-1)
    attn_output = jnp.einsum("bhts,bhsi->bhti", weights, v)
    merged = merge_heads(attn_output)
    return dense_apply(merged, params["out_proj"]["kernel"], params["out_proj"]["bias"])


def mse_loss(output: jnp.ndarray, target: jnp.ndarray) -> jnp.ndarray:
    return jnp.mean((output - target) ** 2)


def make_jax_loss_fn(
    *,
    num_heads: int,
    head_dim: int,
    rope_core: JaxBlockDiagonalRoPECore,
    rope_config: JaxHouseholderRoPEConfig,
    head_to_group: jnp.ndarray | None,
):
    def loss_fn(params: dict[str, Any], x: jnp.ndarray, pos: jnp.ndarray, target: jnp.ndarray) -> jnp.ndarray:
        output = jax_attention_block_forward(
            params,
            x,
            pos,
            num_heads=num_heads,
            head_dim=head_dim,
            rope_core=rope_core,
            rope_config=rope_config,
            head_to_group=head_to_group,
        )
        return mse_loss(output, target)

    return loss_fn


def make_flax_loss_fn(module: FlaxHouseholderSelfAttention):
    def loss_fn(params: dict[str, Any], x: jnp.ndarray, pos: jnp.ndarray, target: jnp.ndarray) -> jnp.ndarray:
        output = module.apply({"params": params}, x, pos, deterministic=True)
        return mse_loss(output, target)

    return loss_fn


def jax_sgd_step(
    params: Any,
    x: jnp.ndarray,
    pos: jnp.ndarray,
    target: jnp.ndarray,
    *,
    lr: float,
    loss_fn,
) -> tuple[Any, jnp.ndarray]:
    loss, grads = jax.value_and_grad(loss_fn)(params, x, pos, target)
    updated = jax.tree_util.tree_map(lambda p, g: p - lr * g, params, grads)
    return updated, loss


def torch_single_step_losses(
    module: HouseholderSelfAttention,
    x: torch.Tensor,
    pos: torch.Tensor,
    target: torch.Tensor,
    *,
    lr: float,
) -> tuple[float, float]:
    module.train()
    for parameter in module.parameters():
        parameter.grad = None
    initial_output = module(x, pos)
    initial_loss = torch.nn.functional.mse_loss(initial_output, target)
    initial_loss.backward()
    with torch.no_grad():
        for parameter in module.parameters():
            parameter -= lr * parameter.grad
    for parameter in module.parameters():
        parameter.grad = None
    updated_output = module(x, pos)
    updated_loss = torch.nn.functional.mse_loss(updated_output, target)
    return float(initial_loss.detach().cpu()), float(updated_loss.detach().cpu())
