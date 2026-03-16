# SPEC.md

## Project

**Householder-RoPE: A Householder-reflector formulation of learnable orthogonal basis transport for ND RoPE**

## Status

Draft specification for implementation.

## Purpose

Extend the learnable orthogonal-basis formulation of ND RoPE into an explicit **Householder-reflector parameterization** of the orthogonal mixing matrix \(Q\), preserving the algebraic RoPE guarantees while enabling a matrix-free, depth-controlled, and implementation-friendly realization.

This spec is intended to drive an engineering build of:

1. a mathematically faithful Householder-RoPE module,
2. a diagnostic and validation suite,
3. ablation-ready training hooks,
4. a reference implementation path for Transformer attention blocks.

The starting point is the ND RoPE formulation in *Rethinking RoPE: A Mathematical Blueprint for N-Dimensional Positional Encoding*, which shows that valid ND RoPEs arise from commuting skew-symmetric generators in a MASA of \(\mathfrak{so}(d)\), and that inter-dimensional interactions can be introduced by learning an orthogonal change of basis \(Q\). The paper discusses Cayley, matrix exponential, and Givens parameterizations of \(Q\), and explicitly mentions Householder transformations as a future direction. The present spec makes that direction constructive. See Eq. (25)–(30) and the surrounding discussion, especially the orthogonal-basis formulation and the note that Householder transformations are a viable orthogonal parameterization not developed in the paper. citeturn321740view1turn321740view2

---

## 1. Executive summary

We define a per-head or shared orthogonal matrix

\[
Q = H_m H_{m-1}\cdots H_1,
\qquad
H_r = I - 2u_r u_r^\top,
\qquad
u_r = \frac{v_r}{\lVert v_r\rVert_2},
\]

where each \(H_r\) is a Householder reflector and \(m\) is a user-chosen reflector depth. We then set

\[
R_x = Q D_x Q^\top,
\]

where \(D_x\) is the standard block-diagonal 1D or ND RoPE transform in the canonical toral basis. This preserves the RoPE validity conditions because orthogonal conjugation preserves skew-symmetry, commutativity, and linear independence of the generator family. The result is a learnable basis transport that remains exact, orthogonal, and efficient.

Operationally, one does **not** materialize \(R_x\) as a dense matrix. Instead, for queries and keys one computes

\[
\bar q = Q^\top q,
\qquad
\bar k = Q^\top k,
\]

then applies the usual fast RoPE kernel in the transformed basis. The added cost is only the application of the reflector stack, which is linear in \(m d_h\) per vector instead of quadratic in \(d_h^2\).

---

## 2. Goals and non-goals

### 2.1 Goals

The system SHALL:

- implement a Householder parameterization of the orthogonal matrix \(Q\) in the paper’s learnable-basis ND RoPE construction, preserving the structure of Eq. (25)–(27); citeturn321740view1
- preserve RoPE relativity and reversibility up to numerical precision;
- support 1D and ND block-diagonal RoPE cores;
- avoid dense \(d_h \times d_h\) matrix construction in the hot path;
- expose reflector depth \(m\) as an expressivity/computation knob;
- support both per-head and shared-\(Q\) modes;
- provide initialization schemes, diagnostics, and tests tailored to orthogonal basis transport;
- make ablations against identity, Cayley, matrix exponential, and Givens feasible.

### 2.2 Non-goals

The initial system SHALL NOT:

- replace the inner RoPE block kernel itself with a different positional encoding family;
- require full manifold optimization on \(SO(d_h)\) in v1;
- assume the best reflector schedule a priori;
- guarantee improved downstream quality before empirical validation.

---

## 3. Mathematical contract

### 3.1 Base ND RoPE structure

Let \(\{B_i\}_{i=1}^N \subset \mathfrak{so}(d_h)\) be a commuting, linearly independent family of skew-symmetric generators. The paper shows that valid ND RoPE can be written in the form

\[
R_x = \exp\!\Big(\sum_{i=1}^N x^{(i)} B_i\Big),
\]

with the valid design space tied to a maximal abelian subalgebra of \(\mathfrak{so}(d_h)\). Standard 1D and 2D RoPE correspond to the maximal toral basis, and inter-dimensional interactions are introduced by learning an orthogonal matrix \(Q\) that block-diagonalizes the basis. citeturn994649view0turn321740view1

Concretely, the paper writes

\[
\forall B_i \in \mathcal B,\qquad Q^\top B_i Q = \operatorname{diag}(J_{i,1},\dots,J_{i,N}),
\]

and therefore

\[
\exp(X\cdot \mathcal B)
=
Q\Big(\bigoplus_{j=1}^N \exp\big(\sum_{i=1}^N x^{(i)}J_{i,j}\big)\Big)Q^\top.
\]

This is the exact insertion point for the Householder formulation. citeturn321740view1

### 3.2 Householder parameterization

We define

\[
Q(V) = H_mH_{m-1}\cdots H_1,
\qquad
H_r = I - 2u_ru_r^\top,
\qquad
u_r = \frac{v_r}{\|v_r\|_2}.
\]

Each reflector is symmetric, orthogonal, and involutory, and every orthogonal matrix can be represented as a product of at most \(d_h\) reflections. An orthogonal matrix has determinant \(+1\) if and only if it is the product of an even number of reflections, so choosing even \(m\) keeps \(Q\) in \(SO(d_h)\). citeturn972104search4turn972104search2

### 3.3 Transported generator family

Define the transported generators

\[
\widetilde B_i = Q B_i Q^\top.
\]

Then

\[
R_x^{\mathrm{HH}} = \exp\!\Big(\sum_{i=1}^N x^{(i)}\widetilde B_i\Big)
= Q\,\exp\!\Big(\sum_{i=1}^N x^{(i)}B_i\Big)Q^\top
= Q D_x Q^\top.
\]

The following invariants MUST hold analytically:

1. **Skew-symmetry preserved**
   \[
   \widetilde B_i^\top = -\widetilde B_i.
   \]

2. **Commutativity preserved**
   \[
   [\widetilde B_i,\widetilde B_j] = Q[B_i,B_j]Q^\top = 0.
   \]

3. **Linear independence preserved**
   because conjugation by \(Q\) is invertible.

4. **Relativity preserved**
   \[
   R_{x_1}^\top R_{x_2} = R_{x_2-x_1}.
   \]

5. **Reversibility preserved**
   \[
   R_x^{-1} = R_x^\top = R_{-x}.
   \]

These are direct consequences of the paper’s characterization of valid RoPEs and of orthogonal conjugation. citeturn994649view0turn321740view1

### 3.4 Rank-2 transport identity for a single reflector

For a single reflector \(H = I - 2uu^\top\) and any skew-symmetric \(B\),

\[
HBH = B - 2(uu^\top B + Buu^\top) + 4uu^\top Buu^\top.
\]

Since \(u^\top B u = 0\) for real skew-symmetric \(B\), the last term vanishes and we obtain

\[
HBH = B - 2(uu^\top B + Buu^\top).
\]

Therefore a single reflector modifies a generator by a skew-symmetric correction of rank at most two. This yields a useful interpretation:

- reflector depth \(m\) is a controllable hierarchy of low-rank generator transports;
- Householder-RoPE can be viewed as a sequence of structured low-rank basis deformations of canonical RoPE.

This identity is a required part of the conceptual documentation and SHOULD appear in developer notes and tests.

---

## 4. System design

### 4.1 Modes

The implementation SHALL support these modes:

#### Mode A: Shared-Q

A single reflector stack \(Q\) is shared across all attention heads in a layer.

Use when:

- parameter budget is tight,
- a global basis transport is desired,
- interpretability across heads is important.

#### Mode B: Per-head-Q

Each head has its own reflector stack \(Q_h\).

Use when:

- heads are expected to specialize,
- cross-channel geometry is head-specific,
- maximum expressivity is desired.

#### Mode C: Group-shared-Q

Heads are partitioned into groups and each group shares one reflector stack.

Use when:

- a midpoint is needed between A and B.

### 4.2 RoPE core compatibility

The implementation SHALL support:

- standard 1D RoPE,
- 2D RoPE,
- general ND RoPE in the block-diagonal paper-style basis,
- frequency schedules already present in the hosting model.

The Householder layer SHALL operate as a **basis transport wrapper** around the RoPE core rather than rewriting the inner trigonometric block kernel.

### 4.3 Hot-path principle

The system SHALL NOT explicitly form dense \(Q\) or \(R_x\) in the forward hot path except in diagnostics or tests.

The hot path SHALL use:

1. reflector application to premix \(q\) and \(k\),
2. standard RoPE on premixed \(q\) and \(k\),
3. standard attention thereafter.

---

## 5. Tensor and API specification

### 5.1 Canonical tensor shapes

Assume the attention input tensors use the canonical shape

\[
q,k \in \mathbb R^{B \times H \times T \times D},
\]

with:

- \(B\): batch,
- \(H\): number of heads,
- \(T\): sequence or token axis,
- \(D=d_h\): head dimension.

For ND spatial settings, \(T\) may be flattened over a Cartesian product index and position coordinates supplied separately.

### 5.2 Reflector parameter tensor

#### Per-head mode

\[
V \in \mathbb R^{H \times M \times D}
\]

where \(M=m\) is the number of reflectors per head.

#### Shared mode

\[
V \in \mathbb R^{M \times D}
\]

broadcast across heads.

### 5.3 Required public interfaces

#### `householder_normalize(v, eps)`

Input:
- `v[..., D]`

Output:
- normalized `u[..., D]`

Contract:
- returns \(u = v / \max(\|v\|_2, \varepsilon)\).

#### `apply_householder_stack(x, V, order)`

Input:
- `x`: tensor with trailing feature dim `D`
- `V`: reflector vectors
- `order`: `forward` or `reverse`

Output:
- transformed tensor of same shape as `x`

Contract:
- applies the sequence
  \[
  z \leftarrow z - 2u_r(u_r^\top z)
  \]
  in the specified order.

#### `premix_qk(q, k, V)`

Output:
- premixed `q_bar, k_bar`

Contract:
- computes the basis change appropriate to the implementation convention.
- In a right-acting row-vector implementation, care MUST be taken with order so that the result corresponds to \(Q^\top q\) and \(Q^\top k\) in the column-vector derivation.

#### `apply_householder_rope(q, k, pos, rope_core, V)`

Output:
- transformed `q_rope, k_rope`

Contract:
- applies Householder premix, then the standard RoPE core.

#### `materialize_Q(V)`

Diagnostic only.

Output:
- dense orthogonal matrix or matrices.

Contract:
- allowed only in validation, visualization, or slow-path tests.

---

## 6. Forward-pass specification

### 6.1 Column-vector mathematical view

The intended semantics are

\[
\bar q = Q^\top q,
\qquad
\bar k = Q^\top k,
\]

\[
\hat q_x = D_x \bar q,
\qquad
\hat k_y = D_y \bar k,
\]

\[
\langle R_x q, R_y k \rangle = \langle \hat q_x, \hat k_y \rangle.
\]

### 6.2 Implementation note on order

Because each reflector is symmetric, \(H_r^\top = H_r\), but the full product reverses under transposition:

\[
Q = H_m\cdots H_1,
\qquad
Q^\top = H_1\cdots H_m.
\]

Therefore:

- if the stored convention is \(Q = H_m\cdots H_1\),
- then the premix path for \(Q^\top\) MUST apply reflectors in reverse construction order relative to the dense left-product convention.

The codebase MUST document this carefully, since silent order mistakes preserve orthogonality but alter the learned basis.

### 6.3 Complexity

Applying one reflector to one length-\(D\) vector costs \(O(D)\).
Applying \(M\) reflectors costs \(O(MD)\).

For tensors \([B,H,T,D]\), the asymptotic premix cost is

\[
O(BHTMD).
\]

Dense application of \(Q\) would cost

\[
O(BHTD^2),
\]

so the reflector formulation is preferable when \(M \ll D\).

---

## 7. Initialization specification

The implementation SHALL provide these initialization strategies.

### 7.1 Identity initialization via paired reflectors

Use even \(M\), grouped into pairs.
Initialize each pair with identical vectors:

\[
H(u)H(u)=I.
\]

This yields an exact identity map at initialization while preserving learnable parameters.

This SHALL be the default initialization for training stability.

### 7.2 Near-identity jittered pairs

Start from paired identity initialization, then add small perturbations to one vector in each pair.

Use when:
- symmetry breaking is desired from step zero,
- exact identity is too conservative.

### 7.3 Random orthogonal warm-start by reflector composition

Initialize vectors randomly and rely on the reflector stack to define a random orthogonal matrix.

Use only in experiments; not recommended as the default.

### 7.4 Determinant policy

The configuration SHALL expose:

- `enforce_SO = true|false`

If `true`, \(M\) MUST be even. Since an orthogonal matrix has determinant \(+1\) iff it is a product of an even number of reflections, this is sufficient. citeturn972104search4

---

## 8. Numerical stability requirements

The implementation MUST:

- normalize reflector vectors with a configurable epsilon,
- avoid branching that produces discontinuous gradients except where explicitly documented,
- support fp32 and bf16/fp16 forward paths with fp32 accumulation for normalization and dot products,
- expose a safe fallback for zero or near-zero reflector vectors.

Suggested safe normalization:

\[
\mathrm{normalize}(v) = \frac{v}{\sqrt{\lVert v \rVert_2^2 + \varepsilon^2}}.
\]

The implementation SHOULD optionally expose a `tau` form,

\[
H = I - \tau v v^\top,
\]

for compatibility with numerical linear algebra libraries that exploit the rank-1 structure of Householder operations. The GNU Scientific Library documentation notes that Householder transforms can be applied efficiently using this rank-1 structure. citeturn105544search4

---

## 9. Diagnostics and observability

A complete v1 implementation SHALL include the following diagnostics.

### 9.1 Orthogonality defect

For diagnostic materialization of \(Q\):

\[
\delta_{\mathrm{orth}} = \frac{\lVert Q^\top Q - I \rVert_F}{\lVert I \rVert_F}.
\]

Expected: near machine precision in fp32.

### 9.2 Relativity defect

For sampled positions \(x_1,x_2\):

\[
\delta_{\mathrm{rel}} = \frac{\lVert R_{x_1}^\top R_{x_2} - R_{x_2-x_1} \rVert_F}{\lVert R_{x_2-x_1} \rVert_F}.
\]

This is a core correctness metric because relativity is identified in the paper as one of the defining RoPE properties. citeturn994649view0

### 9.3 Reversibility defect

\[
\delta_{\mathrm{rev}} = \frac{\lVert R_x^\top - R_{-x} \rVert_F}{\lVert R_{-x} \rVert_F}.
\]

### 9.4 Commutator defect

For transported generators:

\[
\delta_{ij}^{\mathrm{comm}} = \lVert [\widetilde B_i,\widetilde B_j] \rVert_F.
\]

Expected: numerical noise.

### 9.5 Block mixing energy

Let \(P_a\) project onto canonical 2D RoPE block \(a\). Define

\[
M_{ab} = \lVert P_a Q P_b \rVert_F^2.
\]

This reveals whether the learned basis remains mostly local or becomes globally mixed.

### 9.6 Reflector utilization diagnostics

Track per reflector:

- norm of raw vector \(\|v_r\|\),
- cosine similarity between paired reflectors at initialization and over training,
- effective deviation from identity,
- gradient norm.

### 9.7 Logit-path invariance sanity check

Compare attention logits produced by:

1. dense path using explicit \(R_x = QD_xQ^\top\), and
2. premix path using \(Q^\top\) followed by standard RoPE.

These MUST match up to numerical tolerance.

---

## 10. Test plan

### 10.1 Unit tests

#### Test U1: reflector orthogonality

For random normalized \(u\):

- verify \(H^\top H = I\),
- verify \(H^2 = I\),
- verify symmetry.

#### Test U2: stack orthogonality

For random \(V\):

- materialize \(Q\),
- verify \(Q^\top Q \approx I\).

#### Test U3: determinant parity

- odd number of reflectors should give \(\det(Q) \approx -1\),
- even number should give \(\det(Q) \approx +1\).

#### Test U4: premix equivalence

- compare matrix-free and dense application paths.

#### Test U5: relativity

- verify \(R_{x_1}^\top R_{x_2} \approx R_{x_2-x_1}\).

#### Test U6: reversibility

- verify \(R_x^\top \approx R_{-x}\).

#### Test U7: commutativity transport

- verify transported generators still commute.

#### Test U8: rank-2 single-reflector identity

- numerically verify
  \[
  HBH = B - 2(uu^\top B + Buu^\top)
  \]
  for skew-symmetric \(B\).

### 10.2 Property tests

- random seeds over \(D\in\{8,16,32,64,128\}\),
- random \(M\in\{0,2,4,8,16\}\),
- random 1D and ND positions,
- random batch/head sizes.

### 10.3 Integration tests

- replace RoPE in a small Transformer block,
- verify output/logit equivalence between dense and matrix-free paths,
- run a short training smoke test without NaNs or orthogonality drift.

---

## 11. Experiment plan

### 11.1 Mandatory ablations

The first empirical sweep SHALL compare:

1. standard RoPE,
2. Householder-RoPE with \(M=0\) or identity init,
3. Householder-RoPE with \(M=2,4,8\),
4. shared-Q vs per-head-Q,
5. even-\(M\) constrained vs unconstrained parity,
6. identity init vs jittered-pair init,
7. optional comparison against Cayley or Givens baselines.

The paper positions Cayley, matrix exponential, and Givens as candidate orthogonal parameterizations with distinct tradeoffs, making those the natural comparison set. citeturn321740view2turn321740view3

### 11.2 Metrics

At minimum report:

- training loss,
- validation perplexity or task metric,
- throughput,
- memory overhead,
- orthogonality defect,
- relativity defect,
- block mixing energy statistics,
- reflector depth versus gain.

### 11.3 Interpretation questions

The experiment suite SHOULD answer:

- Does light reflector depth already help?
- Is per-head specialization useful?
- Does learned mixing remain local or become global?
- Are gains primarily optimization-related or representational?
- Does the reflector stack saturate quickly with depth?

---

## 12. Recommended implementation order

### Phase 1: math-faithful core

Implement:

- reflector normalization,
- matrix-free stack application,
- dense materialization for tests,
- 1D RoPE wrapper,
- correctness tests U1–U8.

### Phase 2: ND support

Implement:

- ND block kernel interface,
- flattened-token coordinate support,
- relativity/reversibility test suite for ND cases.

### Phase 3: attention integration

Implement:

- drop-in wrapper for a standard attention module,
- shared/per-head/group-shared modes,
- training smoke tests.

### Phase 4: diagnostics and ablations

Implement:

- orthogonality and relativity dashboards,
- block mixing energy visualization,
- depth sweeps and initialization sweeps.

---

## 13. Configuration specification

A v1 config object SHALL contain:

```yaml
householder_rope:
  enabled: true
  mode: per_head          # shared | per_head | group_shared
  num_reflectors: 4
  enforce_SO: true
  init: paired_identity   # paired_identity | jittered_pairs | random
  eps: 1.0e-8
  rope_ndim: 1            # 1 | 2 | N
  materialize_q_for_debug: false
  diagnostics:
    log_orthogonality_defect: true
    log_relativity_defect: true
    log_reversibility_defect: true
    log_commutator_defect: true
    log_block_mixing_energy: true
```

Optional fields:

```yaml
  group_size: 2
  use_tau_parameterization: false
  dense_debug_frequency: 0
  fp32_norm_accumulation: true
```

---

## 14. Reference pseudocode

```text
function householder_rope_qk(q, k, pos, V, rope_core):
    # q, k: [B, H, T, D]
    q_bar = apply_householder_stack_for_QT(q, V)
    k_bar = apply_householder_stack_for_QT(k, V)
    q_rot, k_rot = rope_core(q_bar, k_bar, pos)
    return q_rot, k_rot
```

One reflector application is

```text
function apply_reflector(z, u):
    alpha = dot(z, u)
    return z - 2 * alpha * u
```

For batched tensors the dot product is taken over the trailing feature dimension.

---

## 15. Failure modes and mitigations

### 15.1 Order bug in reflector application

Symptom:
- orthogonality checks pass,
- dense and premix paths disagree.

Mitigation:
- add explicit dense-vs-matrix-free equivalence tests.

### 15.2 Dead reflectors

Symptom:
- some reflectors remain close to identity and never receive gradient.

Mitigation:
- inspect gradient norms,
- use jittered-pair initialization,
- try smaller group sharing or per-head mode.

### 15.3 Excessive mixing

Symptom:
- block mixing becomes globally diffuse,
- downstream quality degrades.

Mitigation:
- reduce \(M\),
- add regularizer on off-block mixing energy,
- share \(Q\) across heads.

### 15.4 Added latency

Symptom:
- reflector premix dominates time.

Mitigation:
- reduce \(M\),
- fuse reflector applications,
- cache premixed keys where appropriate in autoregressive decoding.

---

## 16. Acceptance criteria

A v1 implementation is accepted only if all of the following hold:

1. `Q^T Q - I` remains at numerical noise in fp32 diagnostics.
2. Relativity and reversibility defects remain near numerical noise.
3. Dense and matrix-free logit paths match to tolerance.
4. No dense \(D \times D\) matrices are formed in the production hot path.
5. The implementation supports at least standard 1D RoPE and one ND case.
6. Reflector depth sweeps run end-to-end.
7. The code documents the product-order convention clearly.

---

## 17. Rationale for Householder over the other orthogonal parameterizations

Relative to the paper’s other discussed choices, Householder offers a compelling v1 engineering tradeoff.

- **Versus Cayley:** avoids matrix inversion in the hot path and gives a more local, compositional interpretation of basis transport. The paper notes Cayley can suffer from numerical instability due to inversion and is global in character. citeturn321740view3
- **Versus matrix exponential:** avoids a full exponential map while remaining exactly orthogonal at every step. The paper notes higher computational cost for matrix-exponential parameterization. citeturn321740view2
- **Versus Givens:** provides exact orthogonality and a depth knob with simple matrix-free application, while sacrificing some of Givens’ pairwise interpretability. The paper highlights Givens as localized and interpretable but order-sensitive. citeturn321740view2

The most distinctive advantage of Householder in this setting is that it reveals basis transport as a composition of low-rank skew-generator updates, which is mathematically sharper than treating \(Q\) as an opaque orthogonal matrix.

---

## 18. Deliverables

The first engineering milestone SHALL produce:

1. `SPEC.md` — this document
2. `householder_rope.py` or `householder_rope.py`-equivalent module
3. `test_householder_rope.py`
4. `diagnostics_householder_rope.py`
5. `demo_householder_rope.ipynb` or minimal notebook
6. `README.md` with usage and design notes

Optional later deliverables:

- JAX version,
- fused kernel path,
- regularized mixing penalties,
- direct comparison suite against Cayley/Givens.

---

## 19. Source notes

Primary source for the ND RoPE formulation and the orthogonal-basis insertion point:

- Liu, H. and Zhou, H., *Rethinking RoPE: A Mathematical Blueprint for N-Dimensional Positional Encoding*, arXiv:2504.06308v1. In particular, see the discussion of relativity/reversibility, the MASA-based characterization, Eq. (25)–(27), and the orthogonal-matrix parameterization section that discusses Cayley, matrix exponential, Givens, and mentions Householder as future work. citeturn994649view0turn321740view1turn321740view2

Supporting source for Householder facts used in this spec:

- Klain, D., *Orthogonal matrices* lecture notes. Used for the facts that Householder reflections are orthogonal and that every orthogonal matrix is a product of at most \(n\) reflections; also, determinant parity tracks the parity of the number of reflections. citeturn972104search4
- Cornell CS 6210 Householder notes. Used for the reflector form \(H = I - 2vv^\top\) and the geometric reflection interpretation. citeturn972104search2
- GNU Scientific Library documentation. Used for the note that Householder operations can exploit rank-1 structure efficiently. citeturn105544search4

