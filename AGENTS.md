# AGENTS.md — Operating Manual for Agents (Grand Challenge Technologies Ltd.)

This file defines **how to work** in this workspace: how to think, how to write, how to code, what to ship, and how to review.
It is intended for **human collaborators and code agents** building across research, engineering, and product pipelines.

**Grand Challenge Technologies Ltd. (GCT)** is the core motivating entity behind all initiatives, artifacts, and solutions described here.
Everything shipped should be compatible with professional delivery: reproducible, auditable, and deployable.

---

## 0) North Star

We build systems that are:

- **Rigorous** enough to publish and reproduce.
- **Practical** enough to run on real hardware, on real datasets, with real constraints.
- **Modular** enough to remix into new projects without rewrites.
- **Readable** enough that a new contributor can onboard quickly.
- **Composable** across research prototypes and production pathways.

**Tone:** elegant pragmatism. Feet on the ground, eyes to the stars, ears to the wind.

**Avoid:** buzzwords, hype, and clichés (e.g., “pushing the boundaries…”).

---

## 1) The Project Map (GCT Ecosystem)

This workspace spans multiple strands that frequently interlock.

### 1.1 Architecture & Optimization (Core)
- **MODULUS** — *Measure-Optimised DUaLity Unified Solver.* A modular duality optimization and analysis framework emphasizing:
  - modular layers
  - Lipschitz controls
  - dual norms & constraint geometry
  - measure concentration diagnostics
  - stability-first training dynamics
- **RUNT / nGPT-style normalized Transformers**
  - hyperspherical embeddings and weight parameterizations
  - Haar-random / hypersphere-consistent initialization
  - well-conditioned matrix composition as a first-class objective
- **Hyperball / Riemannian-adjacent wrappers**
  - constrained parameter geometries on spheres/balls
  - Euclidean-compatible implementation with manifold-aware behavior

### 1.2 Representation & Routing
- **Mixture-of-Experts (MoE) routing**
  - adaptive routing via Rényi entropy / free-energy / partition-function lenses
  - top-k routing alternatives and stability metrics
- **Independence & factorization**
  - MMD-based independence objectives for disentangling learned streams
  - modular “independence engine” that can be attached to models

### 1.3 Coordination & Memory Substrates
- **AETHER**
  - a tuplespace / coordination fabric for distributed agentic memory and retrieval
  - interfaces designed for composability with training and inference systems
- **Retrieval, alignment, and memory mechanisms**
  - modern Hopfield flavors, contrastive retrieval, InfoLOOB-like objectives
  - diagnostics for memory capacity and interference

### 1.4 Harnesses & Diagnostics
- **RoPE-Test**
  - analysis harness for RoPE instantiations via attention-map structure diagnostics
  - PyTorch + JAX reference implementations where applicable
- **Layerwise representation collapse / recovery tooling**
  - hidden-state similarity, spectral diagnostics, entropy measures, token diversity
- **Graph / diffusion-inspired structure probes**
  - stability reasoning (e.g., Davis–Kahan) for noisy finite approximations of idealized operators

### 1.5 Product & Experience
- **BOTRAD.IO**
  - “cosmic boombox” AI radio: interactive generative worldbuilding + music/radio experience
  - narrative-first, compelling, clever, retro-futuristic without cringe

---

## 2) Behavioral Rules (How Agents Should Work)

### 2.1 Work Like a Senior Collaborator
- Do not hand-wave. Make claims testable.
- Prefer simple baselines first, then add sophistication behind flags.
- Use explicit interfaces and stable naming.
- If uncertain, state assumptions explicitly and proceed with a best effort.
- When you discover a better approach mid-stream, pivot cleanly and document the rationale.

### 2.2 “Spec → Build → Verify → Package”
Every meaningful deliverable follows this pipeline:

1. **SPEC**: A clear design doc in Markdown.
2. **BUILD**: A working implementation with composable modules.
3. **VERIFY**: Tests, smoke runs, sanity metrics, and invariants.
4. **PACKAGE**: A runnable artifact (pip module, zip, notebook, or CLI) with instructions.

**No orphan output.** If code exists, it must run. If a spec exists, it must be implementable.

### 2.3 Default to Complete Sentences
- Write in complete sentences rather than bullet fragments.
- Bullets are fine, but they must remain grammatical and concrete.

---

## 3) Communication Style & Documentation

### 3.1 Writing Style
- Use precise language, minimal jargon, and clear definitions.
- Teach as you go: short, well-placed explanations that clarify *why the design is shaped this way*.
- Use genderless language in manifestos and broad addresses.
- Prefer short paragraphs and clear headers.
- When making novelty claims, be careful, specific, and falsifiable.

### 3.2 Markdown Standards
Every doc should begin with:
1. **What this is**
2. **Why it matters**
3. **How to run / use it**
4. **Validation plan**
5. **Known failure modes**
6. **Next steps**

Where relevant, also include:
- “Interfaces” (shapes, types, invariants)
- “Ablations” (feature flags and schedules)
- “Complexity” (time, memory, scaling limits)

---

## 4) Engineering Standards

### 4.1 Code Quality
- Prefer **clarity** over cleverness.
- Use type hints in Python where reasonable.
- Add docstrings that explain **what** and **why**, not just **how**.
- Keep functions short and single-purpose.
- Keep side-effects contained and explicit.
- Avoid hidden global state.

### 4.2 Repository Shape (Recommended)
Use this skeleton for new tools/harnesses/modules:

```
project_name/
  README.md
  SPECIFICATIONS.md
  pyproject.toml
  src/project_name/
    __init__.py
    core.py
    modules/
    utils/
  tests/
    test_smoke.py
    test_shapes.py
    test_determinism.py
  scripts/
    run_smoke.py
  examples/
    minimal_example.py
    advanced_example.py
  assets/
  LICENSE
```

### 4.3 Reproducibility (Hard Requirement)
- Fix seeds (PyTorch, NumPy, random, and JAX).
- Log exact versions of:
  - Python, CUDA, PyTorch/JAX, transformers, flash-attn, etc.
- Store configs used in each run next to outputs.
- Provide deterministic toggles where possible.

Include a **Repro Checklist** in every README.

### 4.4 Instrumentation (Required for Research Builds)
Where relevant, include:
- W&B hooks (optional but supported)
- CSV/JSON metric export
- per-step/per-epoch logging with human-readable summaries
- profiling hooks (timing, memory, throughput)
- structured evaluation output suitable for plotting/regression tests

### 4.5 Performance & Hardware Awareness
Assume target GPUs range from **T4 (16GB)** to **A100 (40GB)**.

Provide config knobs for:
- microbatch size
- gradient accumulation
- activation checkpointing
- precision (fp16/bf16/fp32)
- compile flags (torch.compile / XLA)
- attention kernel selection (flash-attn vs baseline)

Avoid hidden quadratic behaviors. State complexity where relevant.

### 4.6 Distributed Training
If building training code:
- Support `torchrun` and multi-node DDP cleanly.
- Provide explicit networking assumptions.
- Fail fast with actionable errors when NCCL or rendezvous fails.
- Include `--dry-run` and `--verbose` options where destructive operations exist.
- Use safe path handling (quotes, spaces, unicode).

---

## 5) Mathematical & Research Standards

### 5.1 Rigor as a Design Constraint
- Define terms before using them.
- Prefer short “proof skeletons” over vague reasoning.
- If a result is conjectural, label it clearly and propose experiments.
- State assumptions (smoothness, boundedness, spectral gaps, etc.) explicitly.

### 5.2 Stability, Conditioning, and Geometry (First-Class)
A recurring theme is that **training stability is geometry + conditioning**.

Agents should:
- treat conditioning as a measurable property, not a vibe
- include metrics (spectral norms, condition numbers, Lipschitz proxies)
- interpret dynamics via a stability lens (edge-of-stability, spectral radius, etc.)
- design parameterizations that maintain invariants (unit-norm/hypersphere constraints)

### 5.3 “Free Energy Lens” as a Translation Layer
We often reframe info-theory and optimization using thermodynamic language:
- partition sums ↔ tilted cumulants
- shuffling / doubly-stochastic maps ↔ monotonic free-energy movement
- Rényi entropies ↔ temperature-indexed free energies

This is not for vibes; it is for:
- convexity intuition
- monotonicity guarantees
- clean inequality reasoning
- principled routing / selection objectives

### 5.4 Diagnostics-First Mindset
When evaluating a new mechanism (RoPE variant, routing trick, manifold wrapper):
- Define what “correct behavior” looks like.
- Add automated checks and visualizations.
- Build a harness so the idea becomes **repeatable science**, not a one-off notebook.

### 5.5 Ablations Are Non-Negotiable
Every meaningful mechanism must be disable-able via flags/config:

- `--use_feature_x true|false`
- `--feature_x_strength <float>`
- `--feature_x_schedule <enum>`

Include an ablation matrix in the spec.

---

## 6) Deliverable Types & Requirements

### 6.1 CLI Tools
Must have:
- `--help` describing flags clearly
- `--dry-run` for file operations
- `--verbose` / `--log-level`
- structured logging (timestamps + levels)
- safe path handling (spaces, unicode, quotes)
- exit codes suitable for CI pipelines

### 6.2 Notebooks (Strong Preference: Self-Contained)
User preference: **ready-to-run notebooks with all code included**.
- Do not rely on external GitHub repos for core logic.
- Annotate GPU requirements and fallback options.
- Save plots to disk (`.png`) and metrics to disk (`.csv`).
- Include a “Run All” path that succeeds from a clean kernel.

### 6.3 Packages (Downloadable Artifacts)
If asked for a “downloadable package”:
- Provide a zip with a single top-level folder.
- Ensure `pip install -e .` works.
- Provide `python -m project_name` as an entrypoint when appropriate.
- Include:
  - README.md
  - SPECIFICATIONS.md
  - LICENSE
  - tests/
  - examples/

### 6.4 Papers / Monographs (LaTeX)
- Use compile-ready LaTeX with theorem environments.
- Add pedagogy and annotations, but keep the main line of argument clean.
- Cite sources appropriately.
- State novelty claims cautiously; propose experiments to validate.

---

## 7) Review Checklist (Before You Call It “Done”)

### 7.1 For Code
- [ ] Runs end-to-end from a clean environment.
- [ ] Has smoke tests.
- [ ] Has clear error messages.
- [ ] Has config defaults that make sense.
- [ ] Has ablations and toggles.
- [ ] Has logging and metrics export.
- [ ] Has README “How to run” and “Troubleshooting”.
- [ ] Does not assume a specific GPU; supports scale-down mode.

### 7.2 For Specs
- [ ] Objective is unambiguous and testable.
- [ ] Interfaces are defined (inputs/outputs, shapes, expected types).
- [ ] Dependencies are listed.
- [ ] Ablations are listed.
- [ ] Validation plan includes unit tests + behavior tests.
- [ ] Failure modes and mitigations are documented.

### 7.3 For Research Claims
- [ ] Claims map to metrics or experiments.
- [ ] “Expected outcomes” are stated before results.
- [ ] Negative results are acceptable and documented.
- [ ] Baselines are included.

---

## 8) Default Conventions

### 8.1 Naming
- Python: `snake_case` for functions, `PascalCase` for classes.
- Config: prefer YAML or JSON with explicit field names.
- Files: avoid spaces where possible; if unavoidable, handle safely.

### 8.2 Logging
- Use Python’s `logging`.
- Prefer `INFO` for progress, `DEBUG` for deep detail.
- Avoid print() in libraries.

### 8.3 Testing
- Use `pytest`.
- Keep tests fast (seconds, not minutes).
- Include:
  - shape checks
  - determinism checks
  - gradient sanity checks (when relevant)

---

## 9) Safety, Legality, and Boundaries

- Do not produce harmful instructions (weapons, self-harm, illegal activity).
- Do not include copyrighted text beyond short quotes.
- When content is medical/legal/financial: cite sources and separate facts from advice.

---

## 10) Handoff Format (Agent → Maintainer)

When completing a task, provide:

1. **What shipped** (files, entry points, what it does).
2. **How to run** (exact commands).
3. **How to validate** (tests, expected outputs).
4. **Known limitations** (what is missing or fragile).
5. **Next steps** (high-leverage improvements).

---

## 11) “Core Clarity” Aesthetic (When Branding Matters)

When output has a design/branding component:
- Minimal, structured, refined.
- White space is allowed.
- Visual hierarchy matters more than decoration.
- Striking and memorable beats “generic professional”.

---

### End

If you follow this file, you will ship work that is easy to trust, easy to run, and easy to build on — in a way that reflects the standard and mission of **Grand Challenge Technologies Ltd.**
