# Householder-RoPE

## What this is

This repository implements the `SPEC.md` design for Householder-RoPE: a Householder-reflector parameterization of learnable orthogonal basis transport for 1D and ND rotary positional encoding. The implementation now includes a PyTorch path, a pure JAX companion path, and a Flax-native wrapper around the JAX backend.

The core contract is the same across backends:

1. Build an orthogonal matrix `Q = H_M ... H_1` from Householder reflectors.
2. Premix queries and keys with `Q^T` by applying the reflector stack in reverse order.
3. Apply a standard block-diagonal RoPE kernel in the transported basis.
4. Materialize dense matrices only in diagnostics and tests.

## Why it matters

Householder reflectors give an exact orthogonal parameterization with a useful depth knob. When the reflector depth `M` is much smaller than the head dimension `D`, the premix cost is `O(MD)` per vector instead of `O(D^2)` for dense orthogonal mixing. The construction also keeps the ND RoPE algebra intact: relativity, reversibility, skew-symmetry, and transported-generator commutativity are preserved up to numerical precision.

The implementation keeps the product-order convention explicit because that is the easiest place to introduce a silent bug. Stored reflectors define `Q = H_M ... H_1`. The premix path needs `Q^T`, so the matrix-free forward pass applies the stored reflectors in reverse order.

## How to run / use it

Install the package in editable mode:

```bash
python -m pip install -e .[dev]
```

If you want the explicit JAX/Flax extras recorded in packaging metadata, use:

```bash
python -m pip install -e .[dev,jax,flax]
```

Run the full unit and integration suite:

```bash
python -m pytest
```

Run the PyTorch smoke script:

```bash
python scripts/run_smoke.py
```

Run the JAX smoke script:

```bash
python scripts/run_smoke_jax.py
```

Run the backend comparison harness:

```bash
python scripts/run_backend_comparison.py
```

The harness also supports larger GPU-focused presets:

```bash
python scripts/run_backend_comparison.py --scenario-set large_gpu --benchmark-mode gpu --repeats 10 --warmup 3
python scripts/run_backend_comparison.py --scenario-set long_context_gpu --benchmark-mode gpu --repeats 8 --warmup 3 --output artifacts/backend_comparison_wsl_uv_long_context_gpu.json
python scripts/plot_backend_comparison.py --inputs artifacts/backend_comparison_wsl_uv_pinned_gpu.json artifacts/backend_comparison_wsl_uv_long_context_gpu.json --output-prefix artifacts/backend_comparison_sequence_sweep
```

For complete attention blocks and single-step training updates, use the dedicated benchmark harness:

```bash
python scripts/run_attention_block_benchmark.py --scenario-set large_gpu --benchmark-mode gpu --forward-repeats 10 --forward-warmup 3 --train-repeats 6 --train-warmup 2 --output artifacts/attention_block_benchmark_wsl_pinned_large.json
python scripts/run_attention_block_benchmark.py --scenario-set long_context_gpu --benchmark-mode gpu --forward-repeats 8 --forward-warmup 3 --train-repeats 4 --train-warmup 2 --output artifacts/attention_block_benchmark_wsl_pinned_long_context.json
python scripts/plot_attention_block_benchmark.py --inputs artifacts/attention_block_benchmark_wsl_pinned_large.json artifacts/attention_block_benchmark_wsl_pinned_long_context.json --output-prefix artifacts/attention_block_sequence_sweep
```

For a realistic-data single-device run outside Colab, use the reusable harness. It now streams ongoing training metrics to the console during the run, writes per-variant JSONL and CSV histories, and saves dedicated diagnostics plots alongside the loss and throughput summaries:

```bash
python scripts/run_real_data_scale_harness.py --train-steps 20 --reflector-sweep 0 8 --log-every 5 --diagnostics-every 10 --diagnostic-token-limit 96 --output-stem realistic_data_smoke
```

For a WSL GPU comparison run with `uv`, where Ubuntu already has working CUDA-enabled JAX and PyTorch in the system Python, use a shared-package `uv` environment and only overlay the minimal safe packages:

```bash
uv venv .venv-wsl-compare --python 3.10 --system-site-packages
uv pip install --python .venv-wsl-compare/bin/python 'numpy<2' pytest
uv pip install --python .venv-wsl-compare/bin/python --no-deps -e /mnt/f/_codex/HOUSEHOLDER
.venv-wsl-compare/bin/python -m pytest tests/test_householder_rope.py tests/test_householder_rope_jax.py tests/test_smoke.py
.venv-wsl-compare/bin/python scripts/run_backend_comparison.py --repeats 15 --warmup 3 --output artifacts/backend_comparison_wsl_uv.json
```

This path keeps the system CUDA JAX plugin intact, fixes `torch.from_numpy(...)` for older CUDA PyTorch builds by overlaying `numpy<2`, and lets the comparison harness skip Flax automatically if the inherited WSL Flax install is incompatible.

For a fully pinned modern WSL GPU stack with newer Torch and a matching Flax/JAX combo, use the pinned requirements file on Ubuntu's native filesystem and then restore the JAX-compatible CuDNN package after the Torch install:

```bash
uv venv /home/$USER/.uv-envs/householder-pinned-gpu-py311-cu12 --python 3.11
uv pip install --python /home/$USER/.uv-envs/householder-pinned-gpu-py311-cu12/bin/python -r requirements/wsl_uv_pinned_gpu.txt
uv pip install --python /home/$USER/.uv-envs/householder-pinned-gpu-py311-cu12/bin/python torch==2.7.0 --index-url https://download.pytorch.org/whl/cu128
uv pip install --python /home/$USER/.uv-envs/householder-pinned-gpu-py311-cu12/bin/python nvidia-cudnn-cu12==9.20.0.48
uv pip install --python /home/$USER/.uv-envs/householder-pinned-gpu-py311-cu12/bin/python --no-deps -e /mnt/f/_codex/HOUSEHOLDER
/home/$USER/.uv-envs/householder-pinned-gpu-py311-cu12/bin/python -m pytest
/home/$USER/.uv-envs/householder-pinned-gpu-py311-cu12/bin/python scripts/run_backend_comparison.py --scenario-set large_gpu --benchmark-mode gpu --repeats 10 --warmup 3 --output artifacts/backend_comparison_wsl_uv_pinned_gpu.json
```

This modern pinned path is the clean shared-process stack verified in this repo: `jax==0.9.1`, `flax==0.12.5`, `optax==0.2.6`, `torch==2.7.0+cu128`, and `nvidia-cudnn-cu12==9.20.0.48`.

Run the PyTorch diagnostics entry point:

```bash
python diagnostics_householder_rope.py
```

Run the JAX diagnostics entry point:

```bash
python diagnostics_householder_rope_jax.py
```

Run the PyTorch minimal example:

```bash
python examples/minimal_example.py
```

Run the JAX minimal example:

```bash
python examples/minimal_example_jax.py
```

Open the notebooks:

```text
demo_householder_rope.ipynb
demo_householder_rope_jax.ipynb
colab_householder_rope_scaling_harness.ipynb
```

For a realistic A100 or TPU Colab run, open `colab_householder_rope_scaling_harness.ipynb`. Its first code cell will `git clone` the repo if needed and install the runtime with `uv`, then the notebook offers six A100-ready profiles: `fast_sanity`, `serious_comparison`, `geometry_signal`, `capacity_limit`, `capacity_max`, and `long_context_stress`. The default `capacity_max` profile is the current recommendation when the goal is to push a 40GB A100 close to its limit: it uses a 24-layer, 2048-width, 4096-token configuration with `batch_size=8`, runs for 300 steps, and compares `num_reflectors in {0, 16, 32}` so the experiment pressures VRAM and runtime far more than the earlier small runs. The notebook now automatically backs off the batch size on CUDA OOM all the way down to `1`, raises gradient accumulation to stay near the original effective batch when it has to shrink the microbatch, surfaces a compact loss-throughput-memory tradeoff table after the run, and streams the harness output line by line through `subprocess.Popen(...)`. It still writes JSON, JSONL, CSV, and PNG artifacts into `artifacts/`, including dedicated training-dynamics, component-diagnostics, and RoPE-health plots. If the batch-size backoff still fails, the recommended manual fallbacks are `capacity_limit` and then `geometry_signal`.

Minimal PyTorch usage:

```python
import torch

from householder_rope import BlockDiagonalRoPECore, HouseholderRoPE, HouseholderRoPEConfig

config = HouseholderRoPEConfig(
    mode="per_head",
    num_reflectors=4,
    init="paired_identity",
    rope_ndim=2,
)

rope = HouseholderRoPE(
    num_heads=4,
    head_dim=16,
    config=config,
    rope_core=BlockDiagonalRoPECore(dim=16, ndim=2),
)

q = torch.randn(2, 4, 8, 16)
k = torch.randn(2, 4, 8, 16)
pos = torch.stack(torch.meshgrid(torch.arange(2), torch.arange(4), indexing="ij"), dim=-1).reshape(-1, 2)

q_rot, k_rot = rope(q, k, pos)
```

Minimal pure-JAX usage:

```python
import jax
import jax.numpy as jnp

from householder_rope.jax_core import JaxBlockDiagonalRoPECore, JaxHouseholderRoPE, JaxHouseholderRoPEConfig

key = jax.random.PRNGKey(0)
config = JaxHouseholderRoPEConfig(
    mode="per_head",
    num_reflectors=4,
    init="paired_identity",
    rope_ndim=2,
)

rope = JaxHouseholderRoPE(
    num_heads=4,
    head_dim=16,
    config=config,
    rope_core=JaxBlockDiagonalRoPECore(dim=16, ndim=2),
    key=key,
)

q = jax.random.normal(jax.random.PRNGKey(1), (2, 4, 8, 16))
k = jax.random.normal(jax.random.PRNGKey(2), (2, 4, 8, 16))
pos = jnp.stack(jnp.meshgrid(jnp.arange(2), jnp.arange(4), indexing="ij"), axis=-1).reshape(-1, 2)

q_rot, k_rot = rope(q, k, pos)
```

Minimal Flax usage:

```python
import jax
import jax.numpy as jnp

from householder_rope.flax_modules import FlaxHouseholderRoPE
from householder_rope.jax_core import JaxBlockDiagonalRoPECore, JaxHouseholderRoPEConfig

module = FlaxHouseholderRoPE(
    num_heads=2,
    head_dim=8,
    config=JaxHouseholderRoPEConfig(mode="per_head", num_reflectors=4, rope_ndim=2),
    rope_core=JaxBlockDiagonalRoPECore(dim=8, ndim=2),
)

q = jax.random.normal(jax.random.PRNGKey(0), (1, 2, 4, 8))
k = jax.random.normal(jax.random.PRNGKey(1), (1, 2, 4, 8))
pos = jnp.asarray([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
variables = module.init(jax.random.PRNGKey(2), q, k, pos)
q_rot, k_rot = module.apply(variables, q, k, pos)
```

## Validation plan

The test suite covers the required mathematical and integration checks from `SPEC.md` for all backends:

1. Single-reflector symmetry, orthogonality, and involution.
2. Stack orthogonality and determinant parity.
3. Matrix-free premix equivalence against dense `Q^T`.
4. Relativity and reversibility defects.
5. Transported-generator commutativity.
6. The rank-2 single-reflector transport identity.
7. Dense-vs-matrix-free attention logit equivalence.
8. Short smoke training loops without NaNs.
9. Flax module equivalence against the pure-JAX wrapper.

The diagnostics modules expose orthogonality defect, block mixing energy, reflector utilization, and dense logit-path error.

## Known failure modes

If the reflector order is applied incorrectly, orthogonality can still look perfect while the dense and matrix-free RoPE paths disagree. The tests explicitly guard against this.

If reflector depth is much larger than needed, the learned basis can become too diffuse. The diagnostics surface block mixing energy so this is visible rather than guessed.

If the user requests paired-identity initialization with an odd number of reflectors, exact identity is impossible. The config rejects that case instead of silently doing something approximate.

On WSL, a system Python that mixes an older CUDA PyTorch build with `numpy>=2` will fail on `torch.from_numpy(...)`. The documented `uv` path overlays `numpy<2` specifically to avoid that breakage while leaving the working GPU JAX installation alone.

On the same WSL stack, Flax may be present but version-incompatible with the inherited JAX install. The package root and comparison harness treat Flax as optional in that case so the PyTorch-vs-JAX comparison still runs cleanly.

In the fully pinned modern WSL env, the `torch==2.7.0+cu128` install downgrades `nvidia-cudnn-cu12` below JAX''s runtime floor unless it is re-pinned afterward. The documented final step restores `nvidia-cudnn-cu12==9.20.0.48`, and both JAX GPU and PyTorch CUDA were verified after that repair.

## Next steps

The next engineering steps are a fused kernel path, richer ND frequency schedules that are loaded directly from host-model configs, an ablation CLI for depth and initialization sweeps, and optional regularizers on off-block mixing energy.

## Repro checklist

- Fix seeds for `random`, `numpy`, `torch`, and `jax`.
- Log Python, PyTorch, JAX, and Flax versions for smoke and benchmark runs.
- Save the exact config object next to any experiment outputs.
- Keep dense materialization out of the production hot path.
- Validate dense-vs-matrix-free equivalence whenever the product-order logic changes.







