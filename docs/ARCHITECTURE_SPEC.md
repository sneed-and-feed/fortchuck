# `fortchuck` Architecture & Mathematical Specification

> **Document Role**: Authoritative mathematical, algorithmic, systems-engineering, verification, and blind-benchmarking blueprint for **fortchuck** (`https://github.com/sneed-and-feed/fortchuck`, license `CC0-1.0 Universal`). This specification reconciles and hardens the Phase 1 research syntheses (`docs/research/ARXIV_MANIFOLD_SURVEY.md`, `docs/research/ARXIV_COUPLED_BOX_MANIFOLD_SURVEY.md`, and `docs/research/TOOLCHAIN_AND_STANDARDS.md`) into a zero-allocation, OpenMP-reentrant Modern Fortran (`F2018`) + ANSI C99 ABI + zero-copy Python library.

---

## Table of Contents

- [1. Problem Class, Adversarial Audit, and Upstream Synthesis](#1-problem-class-adversarial-audit-and-upstream-synthesis)
- [2. Complete Mathematical Algorithm Specification](#2-complete-mathematical-algorithm-specification)
- [3. Fortran 2018 Module Architecture and Zero-Allocation Workspace](#3-fortran-2018-module-architecture-and-zero-allocation-workspace)
- [4. ANSI C99/C11 ABI Header and Zero-Copy Python Package](#4-ansi-c99c11-abi-header-and-zero-copy-python-package)
- [5. Verification, Testing, Lean 4 Formalization, and CI Gates](#5-verification-testing-lean-4-formalization-and-ci-gates)
- [6. Strictly Blind Multi-Seed Benchmark Architecture](#6-strictly-blind-multi-seed-benchmark-architecture)
- [7. Phase 2 Disjoint Swarm Implementation Partition](#7-phase-2-disjoint-swarm-implementation-partition)

---

## 1. Problem Class, Adversarial Audit, and Upstream Synthesis

### 1.1 Canonical Optimization Problem and Unified Regime Coverage

`fortchuck` solves smooth ($C^1$, typically $C^2$ nonconvex and ill-conditioned) optimization problems over the Cartesian product of a $d$-dimensional Euclidean hyperrectangle (box) and $R$ compact real Stiefel manifolds:

```math
\min_{z \in \mathcal B, \; X_1 \in \mathrm{St}(n_1, p_1), \; \dots, \; X_R \in \mathrm{St}(n_R, p_R)} \; f(z, X_1, \dots, X_R)
```

where the factor domains are:

```math
\mathcal B = \left\lbrace z \in \mathbb R^d : l_i \le z_i \le u_i, \; i = 1, \dots, d \right\rbrace, \qquad \mathrm{St}(n_r, p_r) = \left\lbrace X_r \in \mathbb R^{n_r \times p_r} : X_r^\top X_r = I_{p_r} \right\rbrace
```

with $-\infty \le l_i \le u_i \le +\infty$ (supporting two-sided bounds, one-sided bounds, unconstrained coordinates, and fixed variables `l_i = u_i`)
and $1 \le p_r \le n_r$ for each $r = 1, \dots, R$
(recovering unit spheres at `p_r = 1` and orthogonal / special orthogonal groups $\mathrm{O}(n_r), \mathrm{SO}(n_r)$ when `p_r = n_r`).
A single solver engine seamlessly covers all three structural regimes ($d + R \ge 1$):

1. **Pure Euclidean Box** $(R = 0, \; d \ge 1)$: Active-set masked `L-BFGS-B` on $\mathcal B$.
2. **Pure Multi-Stiefel** $(d = 0, \; R \ge 1)$: Lock-free Riemannian `LRBFGS` (or Alternating Barzilai--Borwein when $m = 0$) on $\prod_{r=1}^R \mathrm{St}(n_r, p_r)$.
3. **Coupled Box x Multi-Stiefel** $(d \ge 1, \; R \ge 1)$: Joint quasi-Newton capturing off-diagonal cross-Hessian blocks $\nabla^2_{z, X_r} f \neq 0$ while maintaining strict primal feasibility at every objective evaluation.

### 1.2 Critical Audit and Reconciliation of Phase 1 Findings

Auditing the independent Stiefel (`ARXIV_MANIFOLD_SURVEY.md`) and Coupled Box-Manifold (`ARXIV_COUPLED_BOX_MANIFOLD_SURVEY.md`) surveys alongside compiler/runtime realities (`TOOLCHAIN_AND_STANDARDS.md`) resolves six subtle numerical and systems interactions:

1. **Exact `2 * p_r < n_r` vs `2 * p_r >= n_r` Crossover and Schur-Reduced Woodbury Identity**:
   In classical Wen--Yin (2013) rank-`2*p_r` Sherman--Morrison--Woodbury (SMW), factoring the capacitance matrix `M_{2p_r}(tau)`
   costs $\tfrac{16}{3} p_r^3$ FLOPs and suffers small-step off-diagonal block imbalance (`arXiv:2608.28073`).
   Block-eliminating `M_{2p_r}(tau)` reduces the exact same rank-`2*p_r` Woodbury Cayley curve to a single $p_r \times p_r$ strictly dissipative Schur complement:

```math
J_r(\tau) = I_{p_r} - \frac{\tau}{2} A_r + \frac{\tau^2}{4} S_r, \qquad \kappa_2\bigl(J_r(\tau)\bigr) \le \frac{5 + \tau^2 \lVert \xi_r \rVert_{\mathrm F}^2}{4}
```

   (`arXiv:1301.0172` Prop. 3.2; `arXiv:2608.28073` §3.2). At the exact boundary $2 p_r = n_r$ with default memory $m = 7$,
   total per-iteration linear-algebra cost (setup + trial step + $2m$-pair transport) is $24.58\,n_r^3$ FLOPs for Schur-Woodbury
   versus $17.17\,n_r^3$ FLOPs for Direct LU with resolvent doubling `2 * M_{n_r}(tau)^{-1} - I_{n_r}`.
   Dispatching **Branch A** (`Schur-Woodbury`) when `2 * p_r < n_r` and **Branch B** (`Direct LU`) when `2 * p_r >= n_r` is therefore strictly FLOP-optimal at `2 * p_r == n_r` and matches the architectural specification verbatim.
2. **Pre-Callback Sign-Preserving Householder-QR Manifold Polish**:
   Polishing `X_{r,k+1}` *after* evaluating $f$ and $\nabla f$ at a trial point introduces an $O(10^{-13} \lVert \nabla f \rVert)$ discrepancy between stored `f_val`/`grad` and `X`, which can stall monotone line search near machine precision.
   Instead, `fortchuck` checks $\delta_r = \lVert X_r^\top X_r - I_{p_r} \rVert_{\mathrm F} > 10^{-13}$ immediately after Cayley retraction and *before* calling the user callback,
   applying thin Householder QR `fc_geqrf_pos` with enforced positive diagonal $R_{ii} > 0$.
   Enforcing $R_{ii} > 0$ prevents reflection sign flips,
   moves `X_r` by at most $\delta_r / \sqrt{2} \approx 7 \times 10^{-14}$ when drift triggers,
   preserves $\det(X_r) = +1$ on `SO(n_r)`, and guarantees the user callback never observes orthogonality drift above $10^{-13}$.
3. **Zero-`GEMM` Product Two-Loop Recursion via Cached Isometric Invariants**:
   Under Isometric Cayley Vector Transport $\mathcal T_{\tau_k \eta_{r,k}}^{\mathrm{Cay}}(\zeta) = Q_{\eta_{r,k}}(\tau_k)\zeta$,
   both canonical Stiefel inner products (`st_sy`, `st_ss`, `st_yy`) and skew projections $A_{S,j,r} = X_{r,k}^\top S_{j,r}, \; A_{Y,j,r} = X_{r,k}^\top Y_{j,r} \in \mathfrak{so}(p_r)$
   are algebraically invariant across steps since $((Q_r X_r)^\top (Q_r \zeta) = X_r^\top \zeta)$.
   Caching them in the ring buffer eliminates 100% of matrix-matrix multiplications (`GEMM`) from both the dynamic free-subspace curvature filter and the two-loop recursion, reducing the entire two-loop pass to $O(m(d + \sum_r (n_r p_r + p_r^2)))$ Level-1 SIMD vector sweeps.
4. **Cosine-Guarded Persistent Per-Block Barzilai--Borwein Scaling**:
   Raw per-block BB2 (`ARXIV_COUPLED_BOX_MANIFOLD_SURVEY.md` §4.5) can jump by orders of magnitude near convergence if strong cross-coupling $\nabla^2_{z, X_r} f$ makes one block's secant inner product transiently negative or near-orthogonal, and resetting to $1.0$ destroys asymptotic scaling.
   Instead, each block $b \in \lbrace \text{box}, 1, \dots, R \rbrace$ updates $\gamma_k^{(b)}$ only when its block secant cosine satisfies
   $\sigma_b^{sy} \ge 10^{-3} \sqrt{\sigma_b^{ss} \sigma_b^{yy}}$ with `sigma_b^{yy} > 1e-30`,
   otherwise persisting $\gamma_k^{(b)} = \gamma_{k-1}^{(b)}$.
   When $R = 0$ or $(d = 0, R = 1)$, this reduces identically to classical scalar BB2.
5. **Dimension-Independent Active Band + Secant-Scaled Active Snapping**:
   Using $\epsilon_k = \min(\bar\epsilon, \lVert r_{\text{box}}(z_k) \rVert_\infty)$ with $\bar\epsilon = 10^{-4}$ prevents $\sqrt{d}$ band inflation at large $d$,
   while scaling $\mathcal A_{\epsilon_k}(z_k)$ by safeguarded 1D coordinate secants (falling back to `gamma_box`) avoids stiff-coordinate starvation inside the active band.
6. **IEEE-754 `-ffpe-trap=overflow` Bound-Arithmetic Safety**:
   Because `fpm test` runs under `-ffpe-trap=invalid,zero,overflow`, computing `u(i) - l(i)` when a caller passes `l(i) = -huge(1.0_dp)` and `u(i) = +huge(1.0_dp)` raises `FE_OVERFLOW`. All box checks use overflow-free half-widths `0.5_dp * u(i) - 0.5_dp * l(i)` and inward comparisons `z(i) <= l(i) + eps_ki` and `z(i) >= u(i) - eps_ki`.

### 1.3 Adversarial Failure-Mode Mitigation Matrix

| ID | Failure Mode | Root Mathematical / Systems Cause | Mandatory Mitigation in `fortchuck` |
| :--- | :--- | :--- | :--- |
| FM-1 | Rosen Boundary Jamming | Naive tangent Armijo $\phi(\tau) \le \phi(0) + c_1 \tau \phi'(0^+)$ rejects unit steps when `z_i - l_i ~ 1e-14` clamps early. | Bertsekas active band `A_eps(z_k)` + Generalized Arc-Armijo realized chord `Delta_k(tau)`. |
| FM-2 | Lagrange-Multiplier Stiefel Bias | Clamped `z_i = l_i` has positive multiplier $\lambda_i^\ast > 0$ and cross-coupled `y_{j,box,i} /= 0`, biasing Stiefel steps (`arXiv:2605.10573`). | Dynamic projection `P_{free,k}` masking `q`, `s_j`, and `y_j` on `F_eps(z_k)` inside every two-loop call. |
| FM-3 | Indefinite Masked Curvature | Full pair $\langle s_j, y_j \rangle > 0$ can turn nonpositive after positive-curvature box coordinates clamp into `A_eps(z_k)`. | Per-iteration dynamic filter `sigma_tilde_j >= 1e-10 * ||s_tilde_j|| * ||y_tilde_j||` before using slot `j`. |
| FM-4 | Pre-Partition Ascent Bug | Applying $H_k$ before active-set partitioning couples binding and free coordinates, violating descent (`Bertsekas 1982`). | Partition `A_fixed`, `A_eps`, and `F_eps` before applying positive-definite `H_tilde_k`. |
| FM-5 | Cross-Block Scale Starvation | Single scalar $\gamma_k$ starves softer blocks under $10^5$ box-vs-Stiefel Hessian mismatch (e.g., ML-CPC). | Cosine-guarded persistent per-block BB2 center scaling `gamma_blk(1:R+1)`. |
| FM-6 | Discontinuous Stopping Near Bounds | Tangent-cone norm jumps from $\lambda_i^\ast$ to $0$ at `z_i = l_i`. | 1-Lipschitz `L-BFGS-B` residual `||z - P_B(z - gz)||_inf` paired with Stiefel Riemannian gradient norm. |
| FM-7 | Antipodal / Ill-Conditioned Cayley Step | Large trial velocity approaches $\det(I + X^\top Y) = 0$ or ill-conditioned `GETRF` pivot. | Velocity cap `tau_0 * max_r ||eta_{r,k}||_can <= 2*pi` + `GETRF` reciprocal pivot guard (`1e-12`) halving `tau`. |
| FM-8 | QR Column Sign Flips & Callback Drift | Naive `GEQRF` flips columns by $-1$; post-callback polish invalidates cached $f, \nabla f$. | Pre-callback `fc_geqrf_pos` ($R_{ii} > 0$) whenever `||X_r^T X_r - I||_F > 1e-13`. |
| FM-9 | `-ffpe-trap` Overflow on `±DBL_MAX` | `u_i - l_i` overflows `real64` when `l_i = -huge`, `u_i = +huge`. | Compute `half_w = 0.5_dp*u_i - 0.5_dp*l_i`; test `z_i <= l_i + eps_ki` and `z_i >= u_i - eps_ki`. |
| FM-10 | Premature `ftol` Stop on Curved Valleys | Single flat step triggers loose `ftol` while $\mathcal E_{\text{KKT}} \gg \text{gtol}$ on Rosenbrock or quartic valleys. | Default `ftol = 1e-14` gated by 2 consecutive small $\Delta f$ steps or `E_KKT <= 100 * gtol`. |

---

## 2. Complete Mathematical Algorithm Specification

### 2.1 Product Manifold Geometry and Stiefel Metric

Let $p = (z, X_1, \dots, X_R)$ denote a primal feasible point in the product domain.
On each Stiefel factor $\mathrm{St}(n_r, p_r)$, the tangent space at `X_r` is:

```math
T_{X_r} \mathrm{St}(n_r, p_r) = \left\lbrace \xi_r \in \mathbb R^{n_r \times p_r} : X_r^\top \xi_r + \xi_r^\top X_r = 0 \right\rbrace
```

with ambient orthogonal tangent projection $P_{T_{X_r}}(Z) = Z - X_r\,\mathrm{sym}(X_r^\top Z)$, where $\mathrm{sym}(M) = \tfrac{1}{2}(M + M^\top)$ and $\mathrm{skew}(M) = \tfrac{1}{2}(M - M^\top)$.
Parameterized by $\beta \in \lbrace 1/2, 1 \rbrace$ (default $\beta = 1/2$ canonical metric; optional $\beta = 1$ Euclidean metric), the factor inner product and Riemannian gradient from Euclidean partial derivative $G_r = \nabla_{X_r} f(p) \in \mathbb R^{n_r \times p_r}$ are:

```math
\langle \xi_r, \eta_r \rangle_{X_r} = \mathrm{tr}\!\left(\xi_r^\top \left(I_{n_r} - (1 - \beta) X_r X_r^\top\right) \eta_r\right) = \langle \xi_r, \eta_r \rangle_{\mathrm F} - (1 - \beta)\langle X_r^\top \xi_r, X_r^\top \eta_r \rangle_{\mathrm F}
```

```math
\mathrm{grad}\,f_r(p) = \begin{cases} G_r - X_r G_r^\top X_r & \text{if } \beta = 1/2 \;\; (\text{canonical, default}), \\ G_r - \tfrac{1}{2} X_r (X_r^\top G_r + G_r^\top X_r) & \text{if } \beta = 1 \;\; (\text{Euclidean}). \end{cases}
```

On the product tangent space $\mathcal T_p = \mathbb R^d \oplus \bigoplus_{r=1}^R T_{X_r}\mathrm{St}(n_r, p_r)$, the joint inner product is:

```math
\left\langle (\delta z, \xi), \, (\widetilde{\delta z}, \eta) \right\rangle_p = \delta z^\top \widetilde{\delta z} + \sum_{r=1}^R \langle \xi_r, \eta_r \rangle_{X_r}.
```

Under $\beta = 1/2$, each of the $\tfrac{1}{2}p_r(p_r - 1)$ vertical Lie-algebra coordinates has unit metric weight matching box coordinates `z_i`, and the Cayley curve has zero intrinsic Riemannian acceleration at $\tau = 0$.

### 2.2 Overflow-Safe Bertsekas Active-Band Partition and Joint Search Curve

At iterate $p_k = (z_k, X_{1,k}, \dots, X_{R,k})$,
let `gz` denote $\nabla_z f(p_k) \in \mathbb R^d$
and let $g_{r,k} = \mathrm{grad}\,f_r(p_k) \in T_{X_{r,k}}\mathrm{St}(n_r, p_r)$:

1. **Continuous Box KKT Residual**: For $i = 1, \dots, d$, define coordinatewise clamping $\lbrack P_{\mathcal B}(v) \rbrack_i = \min(\max(v_i, l_i), u_i)$ and:

```math
r_{\text{box},i}(z_k) = z_{k,i} - \min\!\bigl(\max(z_{k,i} - g_{z,k,i}, \, l_i), \, u_i\bigr).
```

2. **Overflow-Safe Safeguarded Active Band**: With $\bar\epsilon$ given by `opts%eps_band` (default $10^{-4}$), set:

```math
\epsilon_k = \min\!\bigl(\bar\epsilon, \; \lVert r_{\text{box}}(z_k) \rVert_\infty\bigr), \qquad \epsilon_{k,i} = \begin{cases} \min\bigl(\epsilon_k, \, 0.25 u_i - 0.25 l_i\bigr) & \text{if } l_i, u_i \text{ are both finite}, \\ \epsilon_k & \text{otherwise.} \end{cases}
```

   For each $i \in \lbrace 1, \dots, d \rbrace$:
   - If `l_i == u_i`, assign `mask_box(i) = 2` (fixed set $\mathcal A_{\text{fixed}}$).
   - Else assign `mask_box(i) = 1` (active-band set $\mathcal A_{\epsilon_k}(z_k)$) iff:

```math
\bigl(z_{k,i} \le l_i + \epsilon_{k,i} \;\text{and}\; g_{z,k,i} > 0\bigr) \quad\text{or}\quad \bigl(z_{k,i} \ge u_i - \epsilon_{k,i} \;\text{and}\; g_{z,k,i} \lt 0\bigr).
```

   - Else assign `mask_box(i) = 0` (free set $\mathcal F_{\epsilon_k}(z_k)$), defining the joint free tangent subspace:

```math
\mathcal S_{\text{free},k} = \mathbb R^{\mathcal F_{\epsilon_k}(z_k)} \oplus \bigoplus_{r=1}^R T_{X_{r,k}}\mathrm{St}(n_r, p_r).
```

3. **Two-Metric Coupled Search Direction**:
   - On $\mathcal A_{\text{fixed}}$: set `dz(i) = 0`.
   - On `A_eps(z_k)` (`mask_box(i) == 1`): set $d_{z,k,i} = -\gamma_{k,i}^{\text{act}} g_{z,k,i}$, where the 1D active-band scale is:

```math
\gamma_{k,i}^{\text{act}} = \begin{cases} \mathrm{clip}\bigl(s_{k-1,\text{box},i} / y_{k-1,\text{box},i}, \, \gamma_{\min}, \, \gamma_{\max}\bigr) & \text{if } k \ge 1 \text{ and } s_{k-1,\text{box},i} y_{k-1,\text{box},i} > 10^{-16} s_{k-1,\text{box},i}^2, \\ \gamma_k^{\text{box}} & \text{otherwise.} \end{cases}
```

   - On the free subspace `S_{free,k}`: apply the positive-definite masked product L-BFGS operator $\widetilde H_k \succ 0$ (§2.4):

```math
\begin{bmatrix} d_{z,k,\mathcal F_{\epsilon_k}} \\ \eta_{1,k} \\ \vdots \\ \eta_{R,k} \end{bmatrix} = -\widetilde H_k \begin{bmatrix} g_{z,k,\mathcal F_{\epsilon_k}} \\ g_{1,k} \\ \vdots \\ g_{R,k} \end{bmatrix}.
```

4. **Feasible Joint Search Curve**: Parameterize the primal feasible trial curve for $\tau \ge 0$ (where $\Pi_{\mathrm{orth}}$ denotes pre-callback Householder-QR polish whenever drift exceeds `opts%orth_tol`):

```math
z(\tau) = P_{\mathcal B}(z_k + \tau d_{z,k}), \qquad X_r(\tau) = \Pi_{\mathrm{orth}}\!\left(\mathcal R_{X_{r,k}}^{\mathrm{Cay}}(\tau \eta_{r,k})\right), \qquad \phi(\tau) = f\bigl(z(\tau), X_1(\tau), \dots, X_R(\tau)\bigr).
```

### 2.3 Strict Directional Descent Theorem and Fallback Guarantee

**Theorem 2.1 (Unconditional Joint Strict Descent and Exact KKT Characterization).**
Let $\widetilde H_k \succ 0$ have eigenvalues in `[lambda_min, lambda_max] subset (0, inf)` on the joint free tangent subspace,
and let every active-band scale satisfy $\gamma_{k,i}^{\text{act}} \ge \gamma_{\min} > 0$. Define the right-hand box velocity:

```math
\dot z_i(0^+) = \begin{cases} 0 & \text{if } (z_{k,i} = l_i \text{ and } d_{z,k,i} \le 0) \text{ or } (z_{k,i} = u_i \text{ and } d_{z,k,i} \ge 0), \\ d_{z,k,i} & \text{otherwise.} \end{cases}
```

Then the right-hand directional derivative $\phi'(0^+) = \lim_{\tau \downarrow 0} (\phi(\tau) - \phi(0))/\tau$ satisfies:

```math
\phi'(0^+) = \sum_{i=1}^d g_{z,k,i} \dot z_i(0^+) + \sum_{r=1}^R \langle g_{r,k}, \eta_{r,k} \rangle_{X_{r,k}} \le -\gamma_{\min} \sum_{i \in \mathcal A_{\epsilon_k}^{\text{int}}(z_k)} g_{z,k,i}^2 - \lambda_{\min}\!\left(\left\lVert g_{z,k,\mathcal F_{\epsilon_k}}\right\rVert_2^2 + \sum_{r=1}^R \lVert g_{r,k} \rVert_{X_{r,k}}^2\right) \le 0
```

where $\mathcal A_{\epsilon_k}^{\text{int}}(z_k) = \left\lbrace i \in \mathcal A_{\epsilon_k}(z_k) : l_i \lt z_{k,i} \lt u_i \right\rbrace$.
Furthermore, $\phi'(0^+) = 0$ if and only if $p_k$ is a first-order KKT stationary point of $f$.

*Proof Summary.*
On `A_fixed`, $\dot z_i(0^+) = 0$.
On `A_eps(z_k)`, boundary indices have outward `dz(i)` so $\dot z_i(0^+) = 0$,
whereas interior indices $i \in \mathcal A_{\epsilon_k}^{\text{int}}(z_k)$
satisfy $\dot z_i(0^+) = -\gamma_{k,i}^{\text{act}} g_{z,k,i}$,
contributing at most $-\gamma_{\min} g_{z,k,i}^2$.
On $\mathcal F_{\epsilon_k}(z_k)$, if clamping occurs (`dot_z(i) /= dz(i)`),
then either $(z_{k,i} = l_i \;\text{and}\; d_{z,k,i} \lt 0)$
or $(z_{k,i} = u_i \;\text{and}\; d_{z,k,i} > 0)$.
In both cases, $i \notin \mathcal A_{\epsilon_k}(z_k)$ forces `gz(i)` and `dz(i)` to have the same sign, so:

```math
g_{z,k,i} \dot z_i(0^+) = 0 \le g_{z,k,i} d_{z,k,i} \qquad \forall \, i \in \mathcal F_{\epsilon_k}(z_k).
```

Summing over free box indices and Stiefel factors yields $\phi'(0^+) \le -\gamma_{\min} \sum_{i \in \mathcal A_{\epsilon_k}^{\text{int}}} g_{z,k,i}^2 - \langle \tilde g_k, \widetilde H_k \tilde g_k \rangle_{p_k}$, establishing both strict descent and KKT characterization. $\square$

#### Mandatory Floating-Point Sufficient-Descent Guard & ABB Steepest-Descent Fallback

Before entering line search at any non-KKT iterate, compute $\phi'(0^+)$ together with realized velocity and gradient norms:

```math
v_0 = \left(\lVert \dot z(0^+) \rVert_2^2 + \sum_{r=1}^R \lVert \eta_{r,k} \rVert_{X_{r,k}}^2\right)^{1/2}, \qquad g_0 = \left(\sum_{i : \dot z_i(0^+) \neq 0} g_{z,k,i}^2 + \sum_{r=1}^R \lVert g_{r,k} \rVert_{X_{r,k}}^2\right)^{1/2}.
```

If roundoff ever produces `v_0 == 0` or $\phi'(0^+) > -10^{-12} v_0 g_0$:
1. Increment `res%n_fallbacks`, flush the L-BFGS ring buffer (`m_cur = 0`), and switch immediately to **Alternating Barzilai--Borwein (ABB) Projected-Riemannian Steepest Descent**:

```math
d_{z,k,i} = \begin{cases} 0 & i \in \mathcal A_{\text{fixed}}, \\ -\alpha_k^{\text{ABB,box}} g_{z,k,i} & i \notin \mathcal A_{\text{fixed}}, \end{cases} \qquad \eta_{r,k} = -\alpha_k^{\text{ABB},(r)} g_{r,k}
```

   where each block's $\alpha_k^{\text{ABB}} \in [10^{-10}, 10^{10}]$ alternates
   BB1 ($\sigma_{ss}/\lvert\sigma_{sy}\rvert$ on odd $k$) and
   BB2 ($\lvert\sigma_{sy}\rvert/\sigma_{yy}$ on even $k$),
   or $1/\max(1, \lVert g_{\text{blk}} \rVert)$ when no prior step exists.
2. **Two-Chance Line-Search Recovery**: Even when $\phi'(0^+) \le -10^{-12} v_0 g_0$ holds,
   if backtracking along a quasi-Newton step (`used_qn == .true.`) exhausts `max_ls_steps`, `fortchuck` flushes `m_cur = 0`, increments `res%n_fallbacks`, recomputes the ABB steepest-descent direction at $p_k$, and retries line search once before returning `FC_ERR_LS_FAILED`.

### 2.4 Joint Product-Space Masked L-BFGS Two-Loop Recursion and Damping

`fortchuck` maintains an eager circular ring buffer of capacity $m \ge 0$ (`default m = 7`) storing unmasked product tangent vectors $(s_j, y_j)$
in the current tangent space $\mathcal T_{p_k}$ (preserving genuine cross-curvature if a clamped coordinate later leaves its bound),
alongside cached Stiefel isometric scalars (`st_sy(j)`, `st_ss(j)`, `st_yy(j)`) and cached skew matrices $A_{S,j,r} = X_{r,k}^\top S_{j,r}, \; A_{Y,j,r} = X_{r,k}^\top Y_{j,r}$.

#### Step 1: Dynamic Free-Subspace Curvature Filter (FM-2 & FM-3 Guard)

At iteration $k$, let `P_{free,k}` zero box coordinates with `mask_box(i) /= 0` virtually (skipping active indices in loops with zero array copies). For each stored pair $j \in \lbrace 1, \dots, m_{\mathrm{cur}} \rbrace$:

```math
\tilde\sigma_j = \sum_{i \in \mathcal F_{\epsilon_k}} s_{j,\text{box},i} y_{j,\text{box},i} + \sigma_{j,\mathrm{St}}^{sy}, \quad \tilde\nu_{s,j} = \sum_{i \in \mathcal F_{\epsilon_k}} s_{j,\text{box},i}^2 + \sigma_{j,\mathrm{St}}^{ss}, \quad \tilde\nu_{y,j} = \sum_{i \in \mathcal F_{\epsilon_k}} y_{j,\text{box},i}^2 + \sigma_{j,\mathrm{St}}^{yy}.
```

Mark slot $j$ active (`active_pair_flag(j) = 1`, setting $\tilde\rho_j = 1 / \tilde\sigma_j$) iff:

```math
\tilde\sigma_j \ge 10^{-10} \sqrt{\tilde\nu_{s,j} \tilde\nu_{y,j}} \quad\text{and}\quad \tilde\nu_{s,j} > 10^{-60}.
```

#### Step 2: `GEMM`-Free Masked Product Two-Loop Recursion

Initialize `q_box(i) = gz(i)` for $i \in \mathcal F_{\epsilon_k}$ ($0$ otherwise), `q_r = g_{r,k}`, and `Aq_r = X_{r,k}^T g_{r,k}`
(already computed inside `fc_stiefel_rgrad`: $2\,\mathrm{skew}(X_{r,k}^\top G_{r,k})$ when $\beta = 1/2$,
or $\mathrm{skew}(X_{r,k}^\top G_{r,k})$ when $\beta = 1$).

In the **backward loop** ($j = m_{\mathrm{cur}}, \dots, 1$, over `active_pair_flag(j) == 1`), compute:

```math
\alpha_j = \tilde\rho_j \left(\sum_{i \in \mathcal F_{\epsilon_k}} s_{j,\text{box},i} q_{\text{box},i} + \sum_{r=1}^R \Bigl(\langle S_{j,r}, q_r \rangle_{\mathrm F} - (1 - \beta)\langle A_{S,j,r}, A_{q,r} \rangle_{\mathrm F}\Bigr)\right)
```

and apply the in-place Level-1 updates:

```math
q_{\text{box},\mathcal F_{\epsilon_k}} \leftarrow q_{\text{box},\mathcal F_{\epsilon_k}} - \alpha_j y_{j,\text{box},\mathcal F_{\epsilon_k}}, \qquad q_r \leftarrow q_r - \alpha_j Y_{j,r}, \qquad A_{q,r} \leftarrow A_{q,r} - \alpha_j A_{Y,j,r}.
```

In the **center scaling step**, equilibrate each block via cosine-guarded persistent BB2 scales $\gamma_k^{(b)} \in [10^{-10}, 10^{10}]$:

```math
q_{\text{box},\mathcal F_{\epsilon_k}} \leftarrow \gamma_k^{\text{box}} q_{\text{box},\mathcal F_{\epsilon_k}}, \qquad q_r \leftarrow \gamma_k^{(r)} q_r, \qquad A_{q,r} \leftarrow \gamma_k^{(r)} A_{q,r}
```

where each $\gamma_k^{(b)}$ starts at $1.0$ ($k = 0$) and updates to
$\mathrm{clip}(\sigma_{k-1,b}^{sy} / \sigma_{k-1,b}^{yy}, 10^{-10}, 10^{10})$
whenever $\sigma_{k-1,b}^{sy} \ge 10^{-3} \sqrt{\sigma_{k-1,b}^{ss} \sigma_{k-1,b}^{yy}}$
and $\sigma_{k-1,b}^{yy} > 10^{-30}$,
otherwise retaining $\gamma_{k-1}^{(b)}$.

In the **forward loop** ($j = 1, \dots, m_{\mathrm{cur}}$, over `active_pair_flag(j) == 1`), compute:

```math
\beta_{\text{loop}} = \tilde\rho_j \left(\sum_{i \in \mathcal F_{\epsilon_k}} y_{j,\text{box},i} q_{\text{box},i} + \sum_{r=1}^R \Bigl(\langle Y_{j,r}, q_r \rangle_{\mathrm F} - (1 - \beta)\langle A_{Y,j,r}, A_{q,r} \rangle_{\mathrm F}\Bigr)\right)
```

and apply the in-place Level-1 updates, followed by negation on the free subspace:

```math
q_{\text{box},\mathcal F_{\epsilon_k}} \leftarrow q_{\text{box},\mathcal F_{\epsilon_k}} + (\alpha_j - \beta_{\text{loop}}) s_{j,\text{box},\mathcal F_{\epsilon_k}}, \qquad q_r \leftarrow q_r + (\alpha_j - \beta_{\text{loop}}) S_{j,r}, \qquad A_{q,r} \leftarrow A_{q,r} + (\alpha_j - \beta_{\text{loop}}) A_{S,j,r},
```

```math
d_{z,k,\mathcal F_{\epsilon_k}} = -q_{\text{box},\mathcal F_{\epsilon_k}}, \qquad \eta_{r,k} = -q_r.
```

#### Step 3: Post-Step Transport and Curvature Positivity Damping

Once $p_{k+1}$ is accepted at step size `tau_k`, transport all stored Stiefel blocks (`S_{j,r}`, `Y_{j,r}`)
together with $\tau_k \eta_{r,k}$ and `g_{r,k}` to the new tangent space via $\mathcal T^{\mathrm{Cay}}$, forming:

```math
s_k = \Bigl(z_{k+1} - z_k, \; \mathcal T_{\tau_k \eta_{1,k}}^{\mathrm{Cay}}(\tau_k \eta_{1,k}), \; \dots\Bigr), \qquad y_k = \Bigl(g_{z,k+1} - g_{z,k}, \; g_{1,k+1} - \mathcal T_{\tau_k \eta_{1,k}}^{\mathrm{Cay}}(g_{1,k}), \; \dots\Bigr).
```

Let $\tilde s_k, \tilde y_k$ denote their projections onto the updated free subspace at `p_{k+1}`,
and let $B_{k,0} s_k$ scale each block of `s_k` by `1 / gamma_k^{(b)}`:
- **Mode `FC_DAMP_CAUTIOUS = 0` (Default, Huang--Absil--Gallivan 2018)**: Push `(s_k, y_k)` into the ring buffer iff $\lVert \tilde s_k \rVert_{p_{k+1}} > 10^{-30}$ and:

```math
\langle \tilde s_k, \tilde y_k \rangle_{p_{k+1}} \ge \max\!\left(10^{-8} \lVert \tilde s_k \rVert_{p_{k+1}}^2 \min\!\bigl(1, \mathcal E_{\text{KKT}}(p_k)\bigr), \; 10^{-12} \lVert \tilde s_k \rVert_{p_{k+1}} \lVert \tilde y_k \rVert_{p_{k+1}}\right).
```

- **Mode `FC_DAMP_POWELL = 1` (Block-Prior Powell Damping)**: With $\omega = 0.2$ and $\mu_k = \langle \tilde s_k, P_{\text{free},k+1} B_{k,0} s_k \rangle_{p_{k+1}} > 0$,
  whenever $\langle \tilde s_k, \tilde y_k \rangle_{p_{k+1}} \lt \omega \mu_k$, damp `y_k` via:

```math
\theta_k = \frac{(1 - \omega)\mu_k}{\mu_k - \langle \tilde s_k, \tilde y_k \rangle_{p_{k+1}}} \in (0, 1), \qquad y_k \leftarrow \theta_k y_k + (1 - \theta_k) B_{k,0} s_k
```

which enforces $\langle \tilde s_k, \tilde y_k \rangle_{p_{k+1}} = \omega \mu_k > 0$ identically prior to insertion.

### 2.5 Dual-Regime Cayley Retraction, Isometric Transport, and Pre-Callback Polish

For each Stiefel factor $r \in \lbrace 1, \dots, R \rbrace$, project $\eta_{r,k} \leftarrow P_{T_{X_{r,k}}}(\eta_{r,k})$ and define:

```math
\tilde\eta_r = \eta_{r,k} - \frac{1}{2} X_{r,k}(X_{r,k}^\top \eta_{r,k}), \qquad W_{\eta_r} = \tilde\eta_r X_{r,k}^\top - X_{r,k} \tilde\eta_r^\top \in \mathfrak{so}(n_r), \qquad Q_{\eta_r}(\tau) = \left(I_{n_r} - \frac{\tau}{2} W_{\eta_r}\right)^{-1}\!\left(I_{n_r} + \frac{\tau}{2} W_{\eta_r}\right).
```

Both linear-algebra regimes evaluate the exact same retraction $Q_{\eta_r}(\tau) X_{r,k}$
and isometric transport $Q_{\eta_r}(\tau_k) \zeta$:

#### Branch A (`2 * p_r < n_r`): Rank $2p_r$ Woodbury / Schur-Cayley Engine

1. **Outer Setup** ($5 n_r p_r^2$ FLOPs once per outer iteration $k$):

```math
A_r = \mathrm{skew}(X_{r,k}^\top \eta_{r,k}) \in \mathfrak{so}(p_r), \qquad K_r = \eta_{r,k} - X_{r,k} A_r, \qquad \tilde\eta_r = \eta_{r,k} - \frac{1}{2} X_{r,k} A_r, \qquad S_r = K_r^\top K_r.
```

   (Scalar fast-path when `p_r == 1`: `A_r = 0`, `K_r = eta_{r,k}`, and $J_r(\tau) = 1 + \tfrac{\tau^2}{4}\lVert \eta_{r,k} \rVert_2^2$ requires zero matrix factorizations.)
2. **Trial Retraction** (at step $\tau$, costing $2 n_r p_r^2 + \tfrac{2}{3} p_r^3$ FLOPs):
   Factor $J_r(\tau)^\top = I_{p_r} + \tfrac{\tau}{2} A_r + \tfrac{\tau^2}{4} S_r = P_r^\top L_r U_r$ via `fc_lu_factor`,
   solve $Z_r J_r(\tau) = 2 X_{r,k} + \tau K_r$ in-place on the column-major panel via unit-stride `fc_lu_solve_right`,
   and set $Y_r(\tau) = Z_r - X_{r,k}$.
3. **Division-Free Isometric Cayley Transport** (at accepted step `tau_k`, reusing factored `J_lu` at $8 n_r p_r^2 + O(p_r^3)$ FLOPs/vector):
   With $C_r = I_{p_r} - \tfrac{\tau_k}{4} A_r$, for each tangent matrix $\zeta$ compute:

```math
R_1 = X_{r,k}^\top \zeta, \quad R_2 = -\tilde\eta_r^\top \zeta, \quad J_r(\tau_k) Z_1 = C_r R_1 + \frac{\tau_k}{2} R_2, \quad \mathcal T_{\tau_k \eta_{r,k}}^{\mathrm{Cay}}(\zeta) = \zeta + \tau_k \tilde\eta_r Z_1 + 2 X_{r,k}(C_r Z_1 - R_1).
```

#### Branch B (`2 * p_r >= n_r`, incl. Orthogonal Groups): Direct LU with Resolvent Doubling

1. **Outer Setup** ($2 n_r p_r^2 + 2 n_r^2 p_r$ FLOPs once per outer iteration $k$):
   Compute $P_r = \tilde\eta_r X_{r,k}^\top$ from `eta_tilde_r`,
   and form $W_{\eta_r} = P_r - P_r^\top \in \mathfrak{so}(n_r)$.
2. **Trial Retraction** (at step $\tau$, costing $\tfrac{2}{3} n_r^3 + 2 n_r^2 p_r$ FLOPs):
   Using $Q_{\eta_r}(\tau) = 2 M_{n_r}(\tau)^{-1} - I_{n_r}$
   where $M_{n_r}(\tau) = I_{n_r} - \tfrac{\tau}{2} W_{\eta_r}$, factor `M_{n_r}(tau)` via `fc_lu_factor`,
   solve $M_{n_r}(\tau) Z_r = 2 X_{r,k}$ via `fc_lu_solve`,
   and set $Y_r(\tau) = Z_r - X_{r,k}$.
3. **Isometric Cayley Transport** (at accepted step `tau_k`, reusing factored `M_lu` at $2 n_r^2 p_r$ FLOPs/vector):
   Solve $M_{n_r}(\tau_k) Z_\zeta = 2 \zeta$
   and set $\zeta \leftarrow Z_\zeta - \zeta$.

#### Singularity, Velocity, and Pre-Callback Drift-Polish Guards

1. **Antipodal Initial Velocity Cap**: Before backtracking, cap initial step size via:

```math
\tau_{\mathrm{init}} \leftarrow \min\!\left(\tau_{\mathrm{init}}, \; \frac{\Delta_{\max}}{\max\bigl(10^{-30}, \, \max_{1 \le r \le R} \lVert \eta_{r,k} \rVert_{\mathrm{can}}\bigr)}\right), \qquad \Delta_{\max} = 2\pi
```

   bounding Schur condition numbers by $\kappa_2(J_r) \le (5 + 8\pi^2)/4 \approx 21$.
2. **Singular / Ill-Conditioned Pivot Halving Guard**: Inside `fc_lu_factor`, if any pivot is zero, `NaN`, or violates $\min_i \lvert U_{ii} \rvert \lt 10^{-12} \max_i \lvert U_{ii} \rvert$, reject $\tau$ immediately (`info > 0`) and halve $\tau \leftarrow 0.5\tau$ without calling the user callback.
3. **Pre-Callback Positive-Diagonal Householder-QR Polish**: Immediately after forming `Y_r(tau)`, compute $\delta_r = \lVert Y_r(\tau)^\top Y_r(\tau) - I_{p_r} \rVert_{\mathrm F}$.
   Whenever $\delta_r > \varepsilon_{\mathrm{orth}}$ (default `1.0e-13`),
   apply `fc_geqrf_pos` ($R_{ii} > 0$, incrementing `res%n_reorth`) *before* invoking `eval_fg`.
   If polish ran on the accepted trial step, re-project transported history pairs onto $T_{X_{r,k+1}}\mathrm{St}(n_r, p_r)$ and refresh `As_hist`, `Ay_hist`
   (flushing `m_cur = 0` only if extreme drift $\delta_r > 10^{-4}$ occurred).

### 2.6 Non-Jamming Generalized Arc-Armijo / Zhang--Hager Globalization

At trial step $\tau = \tau_{\mathrm{init}} \cdot 0.5^j$ for `j = 0 .. opts%max_ls_steps - 1`
(with $\tau_{\mathrm{init}} = 1.0$ for $k \ge 1$
and $\tau_{\mathrm{init}} = \min(1.0, 1.0 / \max(1.0, \lVert g_0 \rVert_{p_0}))$ at $k = 0$),
compute the coupled realized box-chord + Stiefel tangent reference variation:

```math
\Delta_k(\tau) = g_{z,k}^\top \bigl(z(\tau) - z_k\bigr) + \tau \sum_{r=1}^R \langle g_{r,k}, \eta_{r,k} \rangle_{X_{r,k}}.
```

Accept $\tau$ iff all three conditions hold:
1. **Sufficient Negative Chord Slope** (prevents free-set clamping overshoot without Rosen jamming):

```math
\Delta_k(\tau) \le -10^{-12}\left(\lVert z(\tau) - z_k \rVert_2^2 + \tau^2 \sum_{r=1}^R \lVert \eta_{r,k} \rVert_{X_{r,k}}^2\right) \lt 0.
```

2. **Callback Evaluation & Domain Guard**: `n_eval < opts%max_eval`, callback returns `rc == 0`, and `ieee_is_finite(f_trial)` holds. (If `rc > 0` or `NaN`/`Inf`, halve $\tau \leftarrow 0.5\tau$; if `rc < 0`, abort immediately with `FC_ERR_CALLBACK_ABORT`.)
3. **Zhang--Hager / Generalized Arc-Armijo Decrease** ($c_1 = 10^{-4}$):

```math
f\bigl(z(\tau), X_1(\tau), \dots, X_R(\tau)\bigr) \le C_k + c_1 \Delta_k(\tau), \qquad Q_{k+1} = \eta_{\mathrm{ZH}} Q_k + 1, \qquad C_{k+1} = \frac{\eta_{\mathrm{ZH}} Q_k C_k + f(p_{k+1})}{Q_{k+1}}
```

initialized at $Q_0 = 1, C_0 = f(p_0)$ (auto-resolving `opts%zh_eta < 0` universally to Zhang--Hager nonmonotone $0.85$, eliminating the Maratos unit-step rejection effect along curved Stiefel + coupled Box valleys while preserving global KKT convergence).

### 2.7 Exact KKT Stopping Criteria, Budgets, and Deterministic Status Codes

At $k = 0$ and after every accepted step $k \ge 1$, evaluate the continuous KKT residual:

```math
\mathcal E_{\text{box}}(p_k) = \lVert r_{\text{box}}(z_k) \rVert_\infty, \qquad \mathcal E_{\mathrm{St}}(p_k) = \max_{1 \le r \le R} \lVert \mathrm{grad}\,f_r(p_k) \rVert_{X_{r,k}}, \qquad \mathcal E_{\text{KKT}}(p_k) = \max\!\bigl(\mathcal E_{\text{box}}(p_k), \, \mathcal E_{\mathrm{St}}(p_k)\bigr)
```

taking box residual zero when $d = 0$ and Stiefel residual zero when $R = 0$.
Because canonical gradient norm dominates Euclidean Frobenius gradient norm ($\lVert \mathrm{grad}^{\mathrm{can}} f_r \rVert_{\mathrm{can}} \ge \lVert \mathrm{grad}^{\mathrm{Euc}} f_r \rVert_{\mathrm F}$),
achieving $\mathcal E_{\text{KKT}}(p_k) \le \varepsilon_{\text{gtol}}$
strictly implies Euclidean tangent residual $\le \varepsilon_{\text{gtol}}$. All exit pathways return deterministic integer status codes:

| Symbolic Constant | Code | Trigger Condition |
| :--- | :---: | :--- |
| `FC_SUCCESS_QUERY` | `0` | Workspace query (`lwork == -1` or `liwork == -1`) completed cleanly without touching iterates. |
| `FC_CONVERGED_GRAD` | `1` | $\mathcal E_{\text{KKT}}(p_k) \le \varepsilon_{\text{gtol}}$ via `opts%gtol` (primary KKT optimality certificate; default `1.0e-6`). |
| `FC_CONVERGED_FTOL` | `2` | $\lvert f_k - f_{k-1} \rvert \le \varepsilon_{\text{ftol,eff}} \max(1, \lvert f_k \rvert, \lvert f_{k-1} \rvert)$ (honoring explicit `opts%ftol` when `gtol == 0` or `ftol > 1e-13`; sub-2-ULP IEEE-754 roundoff safeguard `2 * FC_EPS` over 16 stalled steps when `gtol > 0` and default `ftol = 1e-14`). |
| `FC_CONVERGED_XTOL` | `3` | $\max\bigl(\lVert z_k - z_{k-1} \rVert_\infty, \max_r \lVert X_{r,k} - X_{r,k-1} \rVert_{\mathrm F}\bigr) \le \varepsilon_{\text{xtol,eff}} \max(1, \lVert z_k \rVert_\infty, \sqrt{\max_r p_r})$ (explicit `opts%xtol` when `gtol == 0` or `xtol > 1e-13`; sub-2-ULP safeguard `2 * FC_EPS` over 16 stalled steps at default `1e-14`). |
| `FC_MAX_ITER_REACHED` | `4` | `k == opts%max_iter` reached with valid feasible iterate $p_k$. |
| `FC_MAX_EVAL_REACHED` | `5` | `n_eval == opts%max_eval` hard budget hit (`n_eval` never exceeds `max_eval`; returns best accepted $p_k$). |
| `FC_ERR_INVALID_DIM` | `-1` | Invalid dimensions: `d < 0`, `R < 0`, `d + R == 0`, `memory < 0`, or any $p_r \notin [1, n_r]$. |
| `FC_ERR_NULL_PTR` | `-2` | Required C pointer is `NULL` when corresponding dimension $> 0$. |
| `FC_ERR_INVALID_BOUNDS` | `-3` | Any box coordinate has $l_i > u_i$, `NaN` bound, or `l_i == +Inf` / `u_i == -Inf`. |
| `FC_ERR_INVALID_X0` | `-4` | Initial `X0` contains `NaN`/`Inf`, is rank-deficient ($\lvert R_{ii} \rvert \le 10^{-14}$), or fails feasibility when `project_initial == 0`. |
| `FC_ERR_WORK_TOO_SMALL` | `-5` | `lwork < lwork_min` or `liwork < liwork_min` (with `lwork /= -1` and `liwork /= -1`). |
| `FC_ERR_CALLBACK_NAN` | `-6` | Initial objective or gradient at $p_0$ returned `rc > 0` or non-finite `NaN`/`Inf`. |
| `FC_ERR_CALLBACK_ABORT` | `-7` | User callback returned negative status `rc < 0` requesting immediate abort. |
| `FC_ERR_LS_FAILED` | `-8` | Backtracking failed down to `max_ls_steps` on both QN direction and ABB steepest-descent restart. |
| `FC_ERR_INVALID_OPTS` | `-9` | Invalid hyperparameter (`c1_armijo <= 0` or `>= 1`, `max_iter < 0`, `max_eval < 1`, unknown enum). |

### 2.8 Master Solver Execution Pseudocode (`fc_solve`)

```text
1. Validate dimensions, options, pointers; compute exact lwork_min, liwork_min.
   If lwork == -1 or liwork == -1: write mins to work(1), iwork(1), res; return FC_SUCCESS_QUERY (0).
   If lwork < lwork_min or liwork < liwork_min: return FC_ERR_WORK_TOO_SMALL (-5).
2. Slice work(1:lwork) and iwork(1:liwork) via deterministic integer offsets (0 heap allocations).
3. Validate bounds l <= u; clamp z <- P_[l,u](z) if project_initial == 1 (else verify feasibility).
   For r = 1..R: compute orth_err_r; if project_initial == 1 and orth_err_r > orth_tol, polish via fc_geqrf_pos
   (returning FC_ERR_INVALID_X0 if rank-deficient); if project_initial == 0 and orth_err_r > 1e-8, error.
4. Evaluate initial callback (n_eval = 1) -> f_k, gz, GX; verify finite & rc == 0.
   Compute rbox, Stiefel rgrad_X, E_KKT(p_0). If E_KKT(p_0) <= gtol: return FC_CONVERGED_GRAD (1).
   Initialize Q_zh = 1, C_zh = f_k, m_cur = 0, ring_head = 0, gamma_blk(1:R+1) = 1.0, tiny_df_streak = 0.
5. For k = 0 .. max_iter - 1:
   a. Partition box indices into mask_box(1:d) in {FREE=0, ACTIVE=1, FIXED=2} using eps_k = min(eps_band, ||rbox||_inf).
   b. If m_cur >= 1 and memory >= 1:
        Compute (dz, eta_X) via fc_lbfgs_direction (dynamic free-subspace filter + GEMM-free two-loop + active-band step).
        used_qn = .true.
      Else:
        Compute ABB steepest-descent direction (dz, eta_X); used_qn = .false.
   c. Check sufficient descent phi'(0^+) <= -1e-12 * v_0 * g_0.
      If violated: flush m_cur = 0, increment n_fallbacks, recompute ABB steepest-descent direction, used_qn = .false.
   d. Line-search loop (up to 2 attempts: QN first, then 1 ABB restart if QN exhausts max_ls_steps):
      - Run fc_cayley_setup for r = 1..R; set tau = min(tau_init, max_step_vel / max(1e-30, max_r ||eta_{r,k}||_can)).
      - For ls = 1 .. max_ls_steps:
          * z_trial = P_[l,u](z + tau * dz); check chord slope Delta_k(tau) <= -1e-12 * step_norm_sq < 0; if not, tau *= 0.5, cycle.
          * Retract X_trial_r = R^Cay(tau * eta_{r,k}) for r = 1..R; if any LU pivot guard triggers, tau *= 0.5, cycle.
          * For r = 1..R: if ||X_trial_r^T X_trial_r - I||_F > orth_tol, apply fc_geqrf_pos(X_trial_r) (polished_step = .true.).
          * If n_eval >= max_eval: populate res at accepted p_k and return FC_MAX_EVAL_REACHED (5).
          * Call eval_fg(z_trial, X_trial) -> f_trial, gz_trial, GX_trial (n_eval += 1).
          * If rc < 0: return FC_ERR_CALLBACK_ABORT (-7). If rc > 0 or .not. finite(f_trial, grads): tau *= 0.5, cycle.
          * If f_trial <= C_zh + c1_armijo * Delta_k(tau): step ACCEPTED, exit ls loop.
          * tau *= 0.5.
      - If not accepted and used_qn: flush m_cur = 0, n_fallbacks += 1, set ABB steepest-descent direction, used_qn = .false., retry d.
      - If not accepted and .not. used_qn: return FC_ERR_LS_FAILED (-8).
   e. Compute step_diff, f_diff = |f_trial - f_k|; update Zhang-Hager (Q_zh, C_zh).
   f. Transport stored Stiefel history + (tau * eta_X, rgrad_X) to T_{X_trial} reusing accepted LU factors (plus tangent re-proj if polished_step).
   g. Commit z = z_trial, X = X_trial, f_k = f_trial, gz = gz_trial, GX = GX_trial; compute new rbox, rgrad_X, E_KKT(p_{k+1}).
   h. Update gamma_blk(1:R+1) (cosine-guarded persistent BB2) and push (s_k, y_k) into ring buffer if memory >= 1 (Cautious or Powell).
   i. Check stopping criteria in strict order: FC_CONVERGED_GRAD (1), FC_CONVERGED_FTOL (2), FC_CONVERGED_XTOL (3).
6. Return FC_MAX_ITER_REACHED (4).
```

---

## 3. Fortran 2018 Module Architecture and Zero-Allocation Workspace

### 3.1 Repository Tree and Strict Disjoint File Boundaries

```text
fortchuck/
├── fpm.toml                          # Fortran Package Manager manifest (F2018 + OpenMP)
├── pyproject.toml                    # Zero-copy Python package & pytest/benchmark metadata
├── CITATION.cff                      # Academic citation metadata (CC0-1.0)
├── LICENSE                           # CC0-1.0 Universal
├── README.md                         # Badges, math formulation, quickstart, benchmark table
├── include/
│   └── fortchuck.h                   # ANSI C99 / C++11 interoperable header
├── src/
│   ├── fortchuck_kinds.f90           # Kinds, status enums, bind(C) structs & callback interface
│   ├── fortchuck_linalg.f90          # Pure F2018 gemm, syrk, getrf/getrs/right-solve, geqrf_pos
│   ├── fortchuck_box.f90             # Pure overflow-safe box projection, KKT, Bertsekas active band
│   ├── fortchuck_stiefel.f90         # Pure Stiefel rgrad, dual-regime Cayley, transport, polish
│   ├── fortchuck_lbfgs.f90           # Pure masked product two-loop, block-BB2, cautious/Powell ring
│   ├── fortchuck_solver.f90          # Workspace slicer, query, and master thread-safe fc_solve loop
│   ├── fortchuck_c_abi.f90           # Null-safe iso_c_binding entry points matching fortchuck.h
│   └── fortchuck.f90                 # Public Fortran 2018 facade module
├── test/
│   ├── test_linalg_and_stiefel.f90   # Unit/property tests: LU, QR sign, Schur vs Direct Cayley parity
│   ├── test_coupled_solver.f90       # Pure-box, pure-Stiefel, multi-factor coupled KKT & edge cases
│   └── test_openmp_reentrancy.f90    # 16-thread OpenMP parallel multi-start bit-reproducibility test
├── python/
│   └── fortchuck/
│       ├── __init__.py               # Public Python API exports
│       ├── _native.py                # Shared library builder/loader + ctypes struct/fn bindings
│       └── solver.py                 # Zero-copy column-major minimize_box_stiefel() wrapper
├── tests/
│   ├── test_c_abi_and_python.py      # ctypes workspace query, zero-copy verification, error codes
│   └── test_pathological_guards.py   # Antipodal velocity, NaN callbacks, clamped multiplier FM-2 test
├── formalization/
│   ├── lean-toolchain                # leanprover/lean4:v4.34.0-rc2 (matching warm Mathlib cache)
│   ├── lakefile.lean                 # Lake build configuration
│   └── Fortchuck.lean                # Machine-checked proofs (0 sorry): Cayley, Schur-SMW, Descent
├── benchmarks/
│   ├── problems.py                   # Closed-form objectives & gradients for Benchmarks B1..B7B
│   ├── referee.py                    # Independent external KKT, feasibility, check_grad & eval counter
│   ├── baselines.py                  # SciPy (L-BFGS-B, SLSQP, trust-constr), pymanopt, Alternating BCD
│   └── run_blind_suite.py            # Multi-seed blind runner emitting JSON/Markdown receipts
├── scripts/
│   ├── audit_gfm_math.py             # 22-rule KaTeX/GFM Markdown compliance linter
│   ├── build_shared.py               # Cross-platform fortchuck.dll / libfortchuck.so builder
│   └── verify_zero_alloc.py          # nm -u symbol gate verifying 0 heap/pack symbols across src/*.o
└── .github/
    └── workflows/
        └── ci.yml                    # 4-job GitHub Actions verification matrix
```

### 3.2 `fpm.toml` Configuration

```toml
name = "fortchuck"
version = "0.1.0"
license = "CC0-1.0"
author = "Noah Cohen"
maintainer = "Noah Cohen"
copyright = "CC0-1.0 Universal Public Domain Dedication"
description = "Zero-allocation, thread-safe Modern Fortran Coupled Box x Stiefel Quasi-Newton Optimizer"

[library]
source-dir = "src"

[build]
auto-executables = false
auto-tests = true
openmp = true

[fortran]
implicit-typing = false
implicit-external = false
source-form = "free"
```

### 3.3 Fortran 2018 Module Specifications and `pure` Contracts

Every computational routine in `fortchuck_linalg`, `fortchuck_box`, `fortchuck_stiefel`, `fortchuck_lbfgs`, and `fc_workspace_query` **must carry the `pure` attribute**, have `implicit none (type, external)`, declare zero `save` or module-mutable variables, declare zero runtime-sized stack automatic arrays (`real(dp) :: tmp(n)` is strictly forbidden!), and perform zero I/O or `stop` statements:

1. **`fortchuck_kinds.f90`**:
   - Precision parameters `dp = c_double`, `i32 = c_int`, `i64 = c_int64_t`, `FC_EPS = epsilon(1.0_dp)`, `FC_HUGE = huge(1.0_dp)`, `FC_PI = 3.14159265358979323846264338327950288_dp`.
   - Status constants (`FC_SUCCESS_QUERY = 0` .. `FC_ERR_INVALID_OPTS = -9`), metric enums (`FC_METRIC_CANONICAL = 0`, `FC_METRIC_EUCLIDEAN = 1`), damping enums (`FC_DAMP_CAUTIOUS = 0`, `FC_DAMP_POWELL = 1`), box mask enums (`FC_BOX_FREE = 0`, `FC_BOX_ACTIVE = 1`, `FC_BOX_FIXED = 2`).
   - `type, bind(C) :: fc_options_t` (96 bytes, zero padding) and `type, bind(C) :: fc_result_t` (88 bytes, zero padding).
   - `pure subroutine fc_init_default_options(opts)` and `abstract interface` for `fc_eval_fg_cb`.

2. **`fortchuck_linalg.f90`** (`100% pure`):
   - `pure subroutine fc_gemm_nn(m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)` (plus `fc_gemm_tn`, `fc_gemm_nt`: cache-friendly `j-l-i` column-major loops so inner `i = 1..m` loop is unit-stride SIMD FMA; optional `#ifdef FORTCHUCK_USE_BLAS` dispatch when `m*n*k > 32768`).
   - `pure subroutine fc_syrk_tn(n, p, X, S)` computing $S = X^\top X \in \mathbb R^{p \times p}$, plus `fc_frob_inner` and `fc_frob_norm`.
   - `pure subroutine fc_lu_factor(n, A, ipiv, piv_tol, info)`: In-place partial-pivoted $P A = L U$ with IEEE `NaN` check and reciprocal pivot guard $\min_i \lvert U_{ii} \rvert \lt \varepsilon_{\mathrm{piv}} \max_i \lvert U_{ii} \rvert$ (`piv_tol = 1.0e-12_dp`, setting `info = i` on failure).
   - `pure subroutine fc_lu_solve(n, nrhs, LU, ipiv, B)` solving $A X = B$; `pure subroutine fc_lu_solve_trans(n, nrhs, LU, ipiv, B)` solving $A^\top X = B$; and `pure subroutine fc_lu_solve_right(m, p, LU_Jt, ipiv, Panel)` solving $Z J = \mathrm{Panel} \in \mathbb R^{m \times p}$ directly via unit-stride column operations on `Panel` given `LU_Jt` = LU factors of $J^\top$.
   - `pure subroutine fc_geqrf_pos(n, p, X, tau_qr, work_qr, info)`: In-place thin Householder QR replacing $X \in \mathbb R^{n \times p}$ with $Q \in \mathrm{St}(n, p)$ such that $R_{ii} > 0$ for all $i = 1, \dots, p$ (setting `info = i` if `abs(R(i,i)) <= 1.0e-14_dp`).

3. **`fortchuck_box.f90`** (`100% pure`):
   - `pure subroutine fc_box_validate(d, l, u, status)`
   - `pure subroutine fc_box_project(d, l, u, z)`
   - `pure subroutine fc_box_kkt_residual(d, z, l, u, gz, rbox, box_inf, box_l2)`
   - `pure subroutine fc_box_partition(d, z, l, u, gz, box_inf, eps_band, mask_box, n_active)`
   - `pure subroutine fc_box_active_step(d, z, l, u, gz, mask_box, gamma_box, has_prev, s_prev, y_prev, dz)`
   - `pure function fc_box_dir_deriv0(d, z, l, u, gz, dz, v_box_sq, g_box_sq) result(slope0)`
   - `pure subroutine fc_box_trial_step(d, z, l, u, gz, dz, tau, z_trial, chord_gz, step_norm_sq)`

4. **`fortchuck_stiefel.f90`** (`100% pure`):
   - `pure subroutine fc_stiefel_offsets(R, n_arr, p_arr, x_off, skew_off, cayley_off, ipiv_off, n_st_tot, n_skew_tot, n_cayley_tot, n_ipiv_tot, status)`
   - `pure subroutine fc_stiefel_rgrad(n, p, X, GX, metric, rgrad, A_skew, scratch_p2)`
   - `pure subroutine fc_stiefel_proj_tangent(n, p, X, Z, A_skew_out, scratch_p2)`
   - `pure function fc_stiefel_inner(n, p, X, U, V, metric, scratch_2p2) result(val)`
   - `pure function fc_stiefel_orth_err(n, p, X, E_out) result(err_frob)`
   - `pure subroutine fc_cayley_setup(n, p, X, eta, cay_buf, scratch)` (auto-dispatching `2*p < n` to Schur-Woodbury setup and `2*p >= n` to Direct skew $W_\eta$ setup)
   - `pure subroutine fc_cayley_retract(n, p, X, eta, tau, cay_buf, ipiv, Y_out, scratch, info)` (auto-dispatching Branch A vs Branch B + scalar $p = 1$ fast-path)
   - `pure subroutine fc_cayley_transport_vec(n, p, X_old, eta, tau, cay_buf, ipiv, zeta_inout, scratch)` (in-place isometric $\mathcal T_{\tau \eta}^{\mathrm{Cay}}(\zeta)$ reusing factored `cay_buf` from the accepted retraction)

5. **`fortchuck_lbfgs.f90`** (`100% pure`):
   - `pure subroutine fc_lbfgs_direction(...)`: Executes dynamic free-subspace curvature filter, `GEMM`-free backward/center/forward product sweeps, and active-band box assignment (`fc_box_active_step`).
   - `pure subroutine fc_abb_direction(...)`: Computes per-block Alternating Barzilai--Borwein projected/Riemannian steepest-descent step.
   - `pure subroutine fc_lbfgs_transport_and_push(...)`: Transports stored Stiefel slots `1..m_cur` via `fc_cayley_transport_vec` (plus tangent re-projection if `polished_step`), updates persistent cosine-guarded per-block BB2 scales `gamma_blk(1:R+1)`, applies Cautious or Powell damping, and inserts `(s_k, y_k)` into the circular ring buffer.

6. **`fortchuck_solver.f90`**, **`fortchuck_c_abi.f90`**, and **`fortchuck.f90`**:
   - `pure subroutine fc_workspace_query(d, R, n_arr, p_arr, m, lwork, liwork, status)`
   - `subroutine fc_solve(...)` (non-`pure` solely because it invokes the user procedure pointer `eval_fg`; otherwise `100%` stateless and reentrant).

### 3.4 Exact Caller-Owned `work` and `iwork` Layout

Define the padded block counts and workspace totals:

```math
d_{\mathrm{pad}} = \max(d, 1), \quad N_{\mathrm{St}} = \sum_{r=1}^R n_r p_r, \quad N_{\mathrm{St,pad}} = \max(N_{\mathrm{St}}, 1), \quad N_{\mathrm{tot,pad}} = \max(d + N_{\mathrm{St}}, 1), \quad N_{\mathrm{skew,pad}} = \max\!\left(1, \sum_{r=1}^R p_r^2\right), \quad m_{\mathrm{pad}} = \max(m, 1)
```

```math
c_r = \begin{cases} 2 n_r p_r + 3 p_r^2 & \text{if } 2 p_r \lt n_r, \\ 2 n_r^2 & \text{if } 2 p_r \ge n_r, \end{cases} \qquad q_r = \begin{cases} p_r & \text{if } 2 p_r \lt n_r, \\ n_r & \text{if } 2 p_r \ge n_r, \end{cases}
```

```math
N_{\mathrm{cay}} = \max\!\left(1, \sum_{r=1}^R c_r\right), \qquad N_{\mathrm{piv}} = \max\!\left(1, \sum_{r=1}^R q_r\right), \qquad N_{\mathrm{scr}} = \max\!\left(64, \; \max_{1 \le r \le R} (2 n_r p_r + 6 p_r^2 + 4 p_r)\right).
```

| Buffer Segment | Target Array | Exact Length (`real64` or `int32`) | Role |
| :--- | :--- | :--- | :--- |
| `work` `[1]` | `z_trial`, `gz`, `gz_trial`, `dz`, `rbox` | $5 \, d_{\mathrm{pad}}$ | Box trial iterate, gradients, direction, KKT residual |
| `work` `[2]` | `X_trial`, `GX`, `GX_trial`, `rgrad_X`, `rgrad_old`, `eta_X` | $6 \, N_{\mathrm{St,pad}}$ | Stiefel trial frames, Euclidean/Riemannian grads, tangent step |
| `work` `[3]` | `q_buf`, `s_cur`, `y_cur` | $3 \, N_{\mathrm{tot,pad}}$ | Two-loop accumulator and candidate secant buffers |
| `work` `[4]` | `Aq_buf` | $N_{\mathrm{skew,pad}}$ | Working `p_r x p_r` skew accumulator in two-loop |
| `work` `[5]` | `gamma_blk`, `abb_ss`, `abb_sy`, `abb_yy` | $4 \, (R + 1)$ | Persistent per-block BB2 scales & latest block secant stats |
| `work` `[6]` | `s_hist`, `y_hist` | $2 \, m \, N_{\mathrm{tot,pad}}$ | Circular L-BFGS ring buffer (0 words when `m = 0`) |
| `work` `[7]` | `As_hist`, `Ay_hist` | $2 \, m \, N_{\mathrm{skew,pad}}$ | Cached invariant `X_r^T S_{j,r}` and `X_r^T Y_{j,r}` (0 when `m = 0`) |
| `work` `[8]` | `st_sy`, `st_ss`, `st_yy`, `rho_m`, `alpha_m` | $5 \, m_{\mathrm{pad}}$ | Cached isometric Stiefel norms & two-loop scalars |
| `work` `[9]` | `cayley_persist` | $N_{\mathrm{cay}}$ | Per-factor `A_r, S_r, K_r, eta_tilde_r, J_lu_r` or `W_eta_r, M_lu_r` |
| `work` `[10]` | `scratch` | $N_{\mathrm{scr}}$ | Shared retraction, transport, inner-product, and QR scratch |
| `iwork` `[1]` | `mask_box` | $d_{\mathrm{pad}}$ | Box partition tags (`FREE=0`, `ACTIVE=1`, `FIXED=2`) |
| `iwork` `[2]` | `active_pair_flag`, `ring_order` | $2 \, m_{\mathrm{pad}}$ | Dynamic subspace filter flags and chronological slot map |
| `iwork` `[3]` | `x_off`, `skew_off`, `cay_off`, `ipiv_off` | $4 \, (R + 1)$ | Packed factor offset tables |
| `iwork` `[4]` | `ipiv_all` | $N_{\mathrm{piv}}$ | Per-factor `GETRF` pivot indices |

**Workspace Query Protocol (`lwork = -1` / `liwork = -1`)**:
- Calling `fortchuck_workspace_query(d, R, n_arr, p_arr, m, &lwork, &liwork)` returns `FC_SUCCESS_QUERY (0)` with exact minimum sizes in `int64_t`.
- Calling `fortchuck_solve` with `lwork == -1` or `liwork == -1` writes `real(lwork_min, c_double)` to `work[0]` (if `work != NULL`), `int(min(liwork_min, huge(0_c_int)), c_int)` to `iwork[0]` (if `iwork != NULL`), populates `res->lwork_min` and `res->liwork_min`, and returns `FC_SUCCESS_QUERY (0)`.

### 3.5 Zero-Allocation and OpenMP Reentrancy Contract

1. **Zero Heap Allocations**: Every `*.o` compiled from `src/*.f90` must pass `scripts/verify_zero_alloc.py` (`nm -u`), referencing **zero** allocator (`malloc`, `free`, `realloc`, `calloc`, `_gfortran_allocate*`, `_gfortran_deallocate*`) or hidden temporary packing (`_gfortran_internal_pack*`, `_gfortran_internal_unpack*`) symbols, and compiling cleanly with `-Warray-temporaries -Werror`.
2. **100% OpenMP Reentrancy**: Zero mutable module globals, zero `save` state, and zero internal static buffers exist anywhere in `fortchuck`. Any number of OpenMP parallel-do threads (`-fopenmp`) or POSIX threads may invoke `fortchuck_solve` concurrently on separate caller-owned `(z, X, work, iwork, udata)` buffers with bit-identical deterministic output independent of thread schedule.

---

## 4. ANSI C99/C11 ABI Header and Zero-Copy Python Package

### 4.1 Complete `include/fortchuck.h` Specification

```c
#ifndef FORTCHUCK_H
#define FORTCHUCK_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Status and error codes */
enum fc_status_e {
    FC_SUCCESS_QUERY       =  0,
    FC_CONVERGED_GRAD      =  1,
    FC_CONVERGED_FTOL      =  2,
    FC_CONVERGED_XTOL      =  3,
    FC_MAX_ITER_REACHED    =  4,
    FC_MAX_EVAL_REACHED    =  5,
    FC_ERR_INVALID_DIM     = -1,
    FC_ERR_NULL_PTR        = -2,
    FC_ERR_INVALID_BOUNDS  = -3,
    FC_ERR_INVALID_X0      = -4,
    FC_ERR_WORK_TOO_SMALL  = -5,
    FC_ERR_CALLBACK_NAN    = -6,
    FC_ERR_CALLBACK_ABORT  = -7,
    FC_ERR_LS_FAILED       = -8,
    FC_ERR_INVALID_OPTS    = -9
};

enum fc_metric_e {
    FC_METRIC_CANONICAL = 0,  /* beta = 1/2 (default) */
    FC_METRIC_EUCLIDEAN = 1   /* beta = 1             */
};

enum fc_damping_e {
    FC_DAMP_CAUTIOUS = 0,     /* Huang-Absil-Gallivan cautious skip (default) */
    FC_DAMP_POWELL   = 1      /* Block-prior Riemannian Powell damping        */
};

/* 96-byte naturally aligned options struct (8 doubles + 8 int32s, 0 padding) */
typedef struct fc_options_t {
    double  gtol;             /* Default: 1.0e-6  */
    double  ftol;             /* Default: 1.0e-14 */
    double  xtol;             /* Default: 1.0e-14 */
    double  c1_armijo;        /* Default: 1.0e-4  */
    double  zh_eta;           /* Default: -1.0 (auto: 0.85 Zhang-Hager) */
    double  eps_band;         /* Default: 1.0e-4  */
    double  orth_tol;         /* Default: 1.0e-13 */
    double  max_step_vel;     /* Default: 2*pi    */
    int32_t max_iter;         /* Default: 15000   */
    int32_t max_eval;         /* Default: 15000   */
    int32_t memory;           /* Default: 7 (0 = pure ABB steepest descent) */
    int32_t metric;           /* Default: FC_METRIC_CANONICAL (0) */
    int32_t damping_mode;     /* Default: FC_DAMP_CAUTIOUS (0)    */
    int32_t project_initial;  /* Default: 1 (project z0, polish full-rank X0) */
    int32_t max_ls_steps;     /* Default: 40      */
    int32_t reserved_flags;   /* Default: 0       */
} fc_options_t;

/* 88-byte naturally aligned result struct (6 doubles + 2 int64s + 6 int32s, 0 padding) */
typedef struct fc_result_t {
    double  f_val;
    double  kkt_residual;
    double  box_residual;
    double  stiefel_residual;
    double  orth_error;
    double  last_step_size;
    int64_t lwork_min;
    int64_t liwork_min;
    int32_t status;
    int32_t n_iter;
    int32_t n_eval;
    int32_t n_reorth;
    int32_t n_fallbacks;
    int32_t n_active_box;
} fc_result_t;

/* Objective + Euclidean partial gradient callback:
 * Return 0 on success, >0 on recoverable domain error (triggers step halving),
 * <0 to abort optimization immediately with FC_ERR_CALLBACK_ABORT. */
typedef int32_t (*fc_eval_fg_cb)(
    int32_t d,
    int32_t R,
    const int32_t *n_arr,
    const int32_t *p_arr,
    const double *z,
    const double *X_packed,
    double *f_out,
    double *gz_out,
    double *GX_packed_out,
    void *udata
);

void fortchuck_default_options(fc_options_t *opts);

int32_t fortchuck_workspace_query(
    int32_t d,
    int32_t R,
    const int32_t *n_arr,
    const int32_t *p_arr,
    int32_t memory,
    int64_t *lwork_out,
    int64_t *liwork_out
);

int32_t fortchuck_solve(
    int32_t d,
    int32_t R,
    const int32_t *n_arr,
    const int32_t *p_arr,
    double *z,
    const double *l,
    const double *u,
    double *X_packed,
    fc_eval_fg_cb eval_fg,
    void *udata,
    const fc_options_t *opts,
    double *work,
    int64_t lwork,
    int32_t *iwork,
    int64_t liwork,
    fc_result_t *res
);

int32_t fortchuck_project_stiefel(
    int32_t n,
    int32_t p,
    double *X,
    double *work,
    int64_t lwork
);

int32_t fortchuck_kkt_residual(
    int32_t d,
    int32_t R,
    const int32_t *n_arr,
    const int32_t *p_arr,
    const double *z,
    const double *l,
    const double *u,
    const double *X_packed,
    const double *gz,
    const double *GX_packed,
    int32_t metric,
    double *work,
    int64_t lwork,
    double *kkt_inf_out,
    double *orth_err_out
);

#ifdef __cplusplus
}
#endif

#endif /* FORTCHUCK_H */
```

### 4.2 Zero-Copy Python `ctypes` Package (`python/fortchuck/`)

1. **`python/fortchuck/_native.py`**:
   - Registers MinGW/OpenMP DLL directories on Windows (`os.add_dll_directory` over `PATH`), locates prebuilt `fortchuck.dll` / `libfortchuck.so` (or auto-invokes `scripts/build_shared.py` if missing), and defines `ctypes.Structure` classes `FCOptions` (`ctypes.sizeof(FCOptions) == 96`) and `FCResult` (`ctypes.sizeof(FCResult) == 88`) verified by an import-time `assert` on struct sizes and field offsets.
2. **`python/fortchuck/solver.py`**:
   - Exposes `minimize_box_stiefel(fun_and_grad, z0=None, bounds=None, X0_list=None, options=None, work=None, iwork=None)` plus reusable `FortchuckWorkspace(d, n_list, p_list, memory=7)` for zero-allocation repeated solves in Python loops.
   - **Zero-Copy Column-Major (`order="F"`) Fast Path**:
     - When `R == 1` and `X0` is already a contiguous `float64` Fortran-ordered array (`X0.flags.f_contiguous`), `X_packed` wraps a single buffer with zero transpose copies; inside the `@CFUNCTYPE` callback, `np.ctypeslib.as_array` reshaped with `order="F"` creates zero-copy NumPy views into `z`, `X_r`, `gz_out`, and `GX_r_out` so user Python code can write gradients directly in-place (`out_gz[:] = ...`) or return `(f, gz, [GX_1, ..., GX_R])`.
     - Python exceptions raised inside `fun_and_grad` are caught inside the `@CFUNCTYPE` wrapper, stashed on the call context, and return `-1` (`FC_ERR_CALLBACK_ABORT`) cleanly across the Fortran stack frame before re-raising in Python (never unwinding exceptions through Fortran!).

---

## 5. Verification, Testing, Lean 4 Formalization, and CI Gates

### 5.1 Native Fortran 2018 Test Suite (`test/*.f90`)

Executed under strict sanitizers (`fpm test --compiler gfortran --flag "-std=f2018 -O2 -g -Wall -Wextra -Werror -pedantic -Warray-temporaries -fimplicit-none -fcheck=all -fbacktrace -ffpe-trap=invalid,zero,overflow -fopenmp"`):

1. **`test/test_linalg_and_stiefel.f90`**:
   - Verifies `fc_lu_factor`, `fc_lu_solve`, `fc_lu_solve_trans`, and `fc_lu_solve_right` against analytical inverses and near-singular pivot rejection (`piv_tol = 1e-12`).
   - Verifies `fc_geqrf_pos` produces $\lVert Q^\top Q - I_p \rVert_{\mathrm F} \le 10^{-15}$ with strictly positive diagonal entries, $\det(Q) = +1$ for perturbed $\mathrm{SO}(n)$ matrices,
   and $\lVert Q - X \rVert_{\mathrm F} \le \delta$ for small drift $\delta = 10^{-12}$.
   - **Algebraic Equivalence Property Test**: Across $(n, p) \in \lbrace (10, 1), (12, 3), (8, 4), (7, 5), (6, 6) \rbrace$, directly compares Branch A (`Schur-Woodbury`) and Branch B (`Direct LU Resolvent`) on identical $(X, \eta, \tau, \zeta)$, asserting retraction agreement $\le 10^{-14}$, transport agreement $\le 10^{-14}$, and exact canonical metric isometry:

```math
\left\lvert \langle \mathcal T \zeta_1, \mathcal T \zeta_2 \rangle_Y - \langle \zeta_1, \zeta_2 \rangle_X \right\rvert \le 10^{-14}.
```

2. **`test/test_coupled_solver.f90`**:
   - **Regime Coverage**: Pure Box ($R = 0$, including `l_i = -huge(1.0_dp), u_i = +huge(1.0_dp)` under `-ffpe-trap=overflow` and fixed `l_i == u_i`), Pure Stiefel ($d = 0$, both $2p \lt n$ on $\mathrm{St}(12, 3)$ and $2p \ge n$ on $\mathrm{SO}(5)$ preserving $\det(X^\ast) = +1$), and Coupled Multi-Factor ($d = 6, R = 2$ on $\mathrm{St}(10, 2) \times \mathrm{St}(4, 4)$).
   - **Ground-Truth Optimality**: Box-coupled Brockett problem verified to match exact Ky Fan QP global minimum $f^\ast$ to $\le 10^{-11}$ with $\mathcal E_{\text{KKT}} \le 10^{-8}$ and orthogonality drift `<= 1e-13`.
   - **Both Damping & Metric Modes**: Tests `FC_DAMP_CAUTIOUS` vs `FC_DAMP_POWELL`, `FC_METRIC_CANONICAL` vs `FC_METRIC_EUCLIDEAN`, and `memory = 0` (ABB) vs `memory = 7` (`LRBFGS`).
3. **`test/test_openmp_reentrancy.f90`**:
   - Spawns `64` independent coupled `[l, u]^8 x St(12, 3) x St(4, 4)` optimization instances across `16` OpenMP parallel-do threads (`schedule(dynamic, 1)`), using per-thread pre-allocated `work`/`iwork` buffers, and asserts **bit-for-bit identical** `f_val`, `n_iter`, `n_eval`, and final `(z, X)` against sequential single-thread execution (`0.0_dp` max difference!).

### 5.2 Python FFI & Adversarial Edge-Case Tests (`tests/*.py`)

1. **`tests/test_c_abi_and_python.py`**:
   - Verifies struct binary layout (`sizeof` and field offsets), `lwork = -1` query parity with `fortchuck_workspace_query`, `FC_ERR_*` codes on invalid bounds/dims/null pointers/undersized `work`, hard `max_eval` enforcement (`res.n_eval <= max_eval` strictly), and clean Python exception propagation across C callbacks.
2. **`tests/test_pathological_guards.py`**:
   - **FM-2 Clamped Multiplier Injection Test**: Constructs the strongly cross-coupled boundary problem on $[0, 1] \times \mathrm{St}(8, 3)$:

```math
f(z, X) = 10^4 z_1 (1 + x_{11}) + \frac{1}{2}\mathrm{tr}(X^\top D X)
```

   where `z_1* = 0` clamps on step 1 with huge active multiplier $\lambda_1^\ast = 10^4(1 + x_{11}^\ast) \gg 1$ and large cross-Hessian `1e4`,
   verifying rapid superlinear convergence ($\le 35$ iterations to $\mathcal E_{\text{KKT}} \le 10^{-10}$).
   - **Callback Domain Barrier Recovery**: Tests an objective returning `rc = +1` (or `NaN`) when step $\tau = 1$ overshoots a log-barrier interior, confirming automatic step halving and clean convergence.

### 5.3 Lean 4 Formal Verification (`formalization/Fortchuck.lean`)

Using the cached `leanprover/lean4:v4.34.0-rc2` + Mathlib4 (`de2ef68216c6074f338c8e61890ee0a379ddfb9b`) toolchain via interactive `lean-lsp` MCP verification (`0 sorry`, `0` warnings), formalize four self-contained algebraic theorems over `Matrix`:

1. **`skew_generator_properties`**: For orthonormal `X` ($X^\top X = 1$) and skew tangent `xi` (`X^T xi + xi^T X = 0`), setting `xi_tilde = xi - (1/2) X (X^T xi)` and `W_xi = xi_tilde X^T - X xi_tilde^T` satisfies $W_\xi^\top = -W_\xi$ and `W_xi X = xi`.
2. **`cayley_rotor_orthogonality_and_isometry`**: Given $W^\top = -W$ and inverse witness `M * (1 - (tau/2) W) = (1 - (tau/2) W) * M = 1`, the Cayley rotor `Q = M * (1 + (tau/2) W)` satisfies `Q = 2 M - 1`, $Q^\top Q = 1$, `(Q X)^T (Q X) = X^T X`, and preserves tangent skew-symmetry and canonical inner products.
3. **`schur_woodbury_cayley_orthogonality`**: Given $X^\top X = 1$, `A^T = -A`, `X^T K = 0`, `J = 1 - (tau/2) A + (tau^2/4) K^T K`, and `J W = W J = 1`, the rank $2p$ Schur-Woodbury retraction `Y = (2 X + tau K) W - X` satisfies $Y^\top Y = 1$ identically.
4. **`coupled_two_metric_strict_descent`**: Formalizes Theorem 2.1's coordinatewise free-set clamping inequality `g_i * dot_z_i <= g_i * d_i` under the Bertsekas active-band sign hypothesis, proving:

```math
\phi'(0^+) \le -\gamma_{\min} \sum_{i \in \mathcal A^{\text{int}}} g_i^2 - \lambda_{\min} \lVert \tilde g \rVert^2 \le 0.
```

### 5.4 Automated Repository & CI Gates

1. **`scripts/audit_gfm_math.py`**: Must pass with `0` violations across `README.md`, `docs/ARCHITECTURE_SPEC.md`, and `docs/research/*.md`.
2. **`scripts/verify_zero_alloc.py`**: Compiles `src/*.f90` to object files and inspects `nm -u *.o` (and `libfortchuck.a`), failing if any heap allocation, deallocation, array-pack, or Fortran I/O/stop runtime symbol appears.
3. **`.github/workflows/ci.yml`**: Four jobs (`gfm-math-audit`, `fortran-fpm-and-zero-alloc`, `python-ffi-and-pytest`, `blind-benchmark-smoke`) mirroring `triad-core` CI standards.

---

## 6. Strictly Blind Multi-Seed Benchmark Architecture

### 6.1 Blind Evaluation Firewall and Independent Referee (`benchmarks/referee.py`)

To guarantee zero data leakage and publication-grade empirical rigor:
1. **Single Frozen Default Configuration**: `fortchuck` is called across **all** benchmark classes and **all** random seeds using `FCOptions()` out-of-the-box (`memory=7`, `metric=FC_METRIC_CANONICAL`, `damping_mode=FC_DAMP_CAUTIOUS`, `eps_band=1e-4`), setting **only** the shared problem-agnostic stopping tolerance (`gtol = 1.0e-6`) and evaluation cap (`max_eval = 5000`). Zero problem-specific parameters, warm-starts, or custom scalings are permitted.
2. **Independent External Referee**: `benchmarks/referee.py` wraps every problem's `(eval_f, eval_grad)` in an opaque evaluation counter, verifies analytical gradients via central finite differences ($\le 10^{-7}$) prior to benchmarking, generates identical feasible `(z0, X0_list)` initializations per seed via `np.random.Generator(np.random.PCG64(seed))` across `10` deterministic seeds (`101..110`), and independently computes on every solver's returned solution:

```math
\mathcal E_{\text{KKT,ref}}(z, X) = \max\!\left(\left\lVert z - P_{[l,u]}\bigl(z - \nabla_z f(z, X)\bigr) \right\rVert_\infty, \; \max_{1 \le r \le R} \left\lVert G_r - X_r\,\mathrm{sym}(X_r^\top G_r) \right\rVert_{\mathrm F}\right),
```

```math
\mathcal V_{\text{box}} = \max_{1 \le i \le d} \max(0, \, l_i - z_i, \, z_i - u_i), \qquad \mathcal V_{\text{orth}} = \max_{1 \le r \le R} \left\lVert X_r^\top X_r - I_{p_r} \right\rVert_{\mathrm F},
```

   marking a run successful iff $\mathcal E_{\text{KKT,ref}} \le 10^{-6}$
   and $\max(\mathcal V_{\text{box}}, \mathcal V_{\text{orth}}) \le 10^{-8}$.

### 6.2 Canonical 9-Instance Blind Benchmark Matrix (`benchmarks/problems.py`)

Exact closed-form objectives and Euclidean gradients follow `docs/research/ARXIV_COUPLED_BOX_MANIFOLD_SURVEY.md` §6:

| ID | Benchmark Problem | Domain Dimensions | Cayley & Coupling Regime Tested |
| :--- | :--- | :--- | :--- |
| `B1` | Ensemble Kohn--Sham DFT Fractional Occupations | $z \in [0, 1]^{10}, \; X \in \mathrm{St}(120, 10)$ | Branch A ($2p \lt n$), small-gap metallic orbital-occupation sloshing |
| `B2A` | Amplitude-Limited Blind Source Separation (ICA) | $Z \in [-2, 2]^{5 \times 150}, \; W \in \mathrm{St}(20, 5)$ | Branch A ($d = 750$), nonconvex $\ln\cosh$ with $d \gg np$ active bounds |
| `B2B` | Smooth Penalty-Split Non-Negative Sparse PCA | $Z \in [0, 5]^{120 \times 6}, \; X \in \mathrm{St}(120, 6)$ | Branch A ($d = 720$), $80\%$ clamped `Z_ij = 0` with large multipliers (FM-2) |
| `B3` | Multi-Camera Anisotropic Bounded-Scale Rig Wahba | $z \in \mathcal B^{15}, \; X_r \in \mathrm{St}(3, 3)^4$ | Branch B ($2p \ge n$, $\mathrm{SO}(3)^4$), $R = 4$ multi-factor $\kappa = 10^3$ |
| `B4` | Bounded-Variance ML Common Principal Components | $z \in [0.05, 20]^{150}, \; X \in \mathrm{St}(30, 30)$ | Branch B $(\mathrm{O}(30))$, log-variance domain barrier, $1.6 \times 10^5$ scale ratio |
| `B5` | Bounded Factor Correlation Matrix Calibration | $z \in \mathcal B^{110}, \; X \in \mathrm{St}(100, 10)$ | Branch A, quartic curved valley + clamped idiosyncratic variances |
| `B6` | Box-Coupled Brockett with Exact Ky Fan QP Truth | $z \in [0.1, 5]^{10}, \; X \in \mathrm{St}(150, 10)$ | Branch A, $\kappa(M) = 10^4$, certified global minimum $f - f^\ast$ check |
| `B7A` | Pure-Box Limit: Active-Bound Chained Rosenbrock + Quadratic | $z \in [l, u]^{500}, \; R = 0$ | Pure Box ($R = 0$, incl. fixed `l_i == u_i`) vs `SciPy L-BFGS-B` |
| `B7B` | Pure-Stiefel Limit: Tall Brockett & Orthogonal Procrustes | $d = 0, \; \mathrm{St}(200, 8) \text{ and } \mathrm{St}(24, 24)$ | Pure Stiefel ($d = 0$) Branch A & B head-to-head vs `pymanopt` |

### 6.3 Head-to-Head Baseline Solvers (`benchmarks/baselines.py`)

1. **`fortchuck` (Default `m = 7`)**: Primary coupled quasi-Newton solver under test.
2. **`fortchuck-ABB` (`m = 0` ablation)**: Isolates the speedup of masked product quasi-Newton curvature over first-order spectral steepest descent.
3. **`Alternating-BCD` (`SciPy L-BFGS-B` + `pymanopt CG`)**: Strictly feasible alternating baseline running 10 inner `L-BFGS-B` iterations on $z$ and 10 inner `pymanopt` `ConjugateGradient` iterations on `X_{1..R}` per cycle under the shared `max_eval = 5000` cap.
4. **`pymanopt-Warped-CG` & `pymanopt-TrustRegions`**: `pymanopt 2.2.1` on `Product([Euclidean(d), Stiefel(...)])` via smooth `tanh` / `exp` box warping (and native `Stiefel` on `B7B`).
5. **`SciPy-L-BFGS-B`**: Native Fortran `L-BFGS-B` on `B7A`, and quadratic Stiefel penalty continuation ($\rho \in \lbrace 10^2, 10^4, 10^6 \rbrace$) on `B1..B6, B7B`.
6. **`SciPy-SLSQP` & `SciPy-trust-constr`**: Constrained Euclidean NLP with explicit `Bounds(l, u)` and $\mathrm{vech}(X_r^\top X_r - I_{p_r}) = 0$ equality constraints (evaluated on instances with `N_tot <= 450` under a `20s` per-seed wall-clock timeout).

---

## 7. Phase 2 Disjoint Swarm Implementation Partition

To maximize parallel velocity with **zero file contention**, Phase 2 splits cleanly across five disjoint ownership slices:

| Worker Role | Owned Files (Strict Write Boundary) | Deliverable & Verification Gate |
| :--- | :--- | :--- |
| **`fortran-core`** | `fpm.toml`, `src/fortchuck_kinds.f90`, `src/fortchuck_linalg.f90`, `src/fortchuck_box.f90`, `src/fortchuck_stiefel.f90`, `test/test_linalg_and_stiefel.f90` | `100% pure` linalg, overflow-safe box engine, dual-regime Schur-Woodbury & Direct LU Cayley + `geqrf_pos`; passes `fpm test` with sanitizers + `0` `nm -u` alloc symbols. |
| **`fortran-solver`** | `include/fortchuck.h`, `src/fortchuck_lbfgs.f90`, `src/fortchuck_solver.f90`, `src/fortchuck_c_abi.f90`, `src/fortchuck.f90`, `test/test_coupled_solver.f90`, `test/test_openmp_reentrancy.f90` | Masked `GEMM`-free product L-BFGS, `fc_solve` master loop, C-ABI, coupled & 16-thread OpenMP bit-reproducibility tests; passes `fpm test` + `0` `nm -u` alloc symbols. |
| **`python-ffi`** | `pyproject.toml`, `python/fortchuck/__init__.py`, `python/fortchuck/_native.py`, `python/fortchuck/solver.py`, `scripts/build_shared.py`, `scripts/verify_zero_alloc.py`, `scripts/audit_gfm_math.py`, `tests/test_c_abi_and_python.py`, `tests/test_pathological_guards.py` | Shared `.dll`/`.so` build, zero-copy `order="F"` Python bindings, `nm -u` zero-alloc verifier, and `pytest` FFI/guard suite (`100%` pass). |
| **`lean-prover`** | `formalization/lean-toolchain`, `formalization/lakefile.lean`, `formalization/Fortchuck.lean` | Interactive `lean-lsp` formalization of all 4 core theorems (`0 sorry`, `0` warnings, `lake build` clean). |
| **`bench-docs`** | `benchmarks/problems.py`, `benchmarks/referee.py`, `benchmarks/baselines.py`, `benchmarks/run_blind_suite.py`, `.github/workflows/ci.yml`, `CITATION.cff`, `README.md` | Blind 10-seed benchmark execution across `B1..B7B` vs all baselines (`results/blind_benchmark_summary.json`), CI workflow, and publication-grade `README.md` (`0` GFM math violations). |
