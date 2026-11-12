# `fortchuck` Architecture & Mathematical Specification (`v0.2.0`)

> **Document Role**: Authoritative mathematical, algorithmic, systems-engineering, verification, and blind-benchmarking blueprint for **fortchuck** (`https://github.com/sneed-and-feed/fortchuck`, license `CC0-1.0 Universal`, version `0.2.0`). This specification reconciles and hardens the Phase 1 research syntheses (`docs/research/ARXIV_MANIFOLD_SURVEY.md`, `docs/research/ARXIV_COUPLED_BOX_MANIFOLD_SURVEY.md`, and `docs/research/TOOLCHAIN_AND_STANDARDS.md`) together with the `v0.2.0` four-pillar upgrade (batched Level-3 panel Cayley transport, coordinatewise diagonal box preconditioning + Hermite arc interpolation, coupled singly-linearly-constrained box/simplex optimization, and Lean 4 Theorems 6--7 + Benchmark `B8` + blind holdout `Seeds 301..310`) into a zero-allocation, OpenMP-reentrant Modern Fortran (`F2018`) + ANSI C99 ABI + zero-copy Python library.

---

## Table of Contents

- [1. Problem Class, Adversarial Audit, and Upstream Synthesis](#1-problem-class-adversarial-audit-and-upstream-synthesis)
- [2. Complete Mathematical Algorithm Specification](#2-complete-mathematical-algorithm-specification)
- [3. Fortran 2018 Module Architecture and Zero-Allocation Workspace](#3-fortran-2018-module-architecture-and-zero-allocation-workspace)
- [4. ANSI C99/C11 ABI Header and Zero-Copy Python Package](#4-ansi-c99c11-abi-header-and-zero-copy-python-package)
- [5. Verification, Testing, Lean 4 Formalization, and CI Gates](#5-verification-testing-lean-4-formalization-and-ci-gates)
- [6. Strictly Blind Multi-Seed Benchmark Architecture](#6-strictly-blind-multi-seed-benchmark-architecture)
- [7. Phase 2 Disjoint Swarm Implementation Partition (`v0.1.0` Baseline)](#7-phase-2-disjoint-swarm-implementation-partition-v010-baseline)
- [8. Phase 3 Disjoint Swarm Implementation Partition (`v0.2.0` Upgrade)](#8-phase-3-disjoint-swarm-implementation-partition-v020-upgrade)

---

## 1. Problem Class, Adversarial Audit, and Upstream Synthesis

### 1.1 Canonical Optimization Problem and Unified Regime Coverage

`fortchuck` solves smooth ($C^1$, typically $C^2$ nonconvex and ill-conditioned) optimization problems over the Cartesian product of a $d$-dimensional Euclidean box (or singly-linearly-constrained box / simplex polytope) and $R$ compact real Stiefel manifolds:

```math
\min_{z \in \mathcal D, \; X_1 \in \mathrm{St}(n_1, p_1), \; \dots, \; X_R \in \mathrm{St}(n_R, p_R)} \; f(z, X_1, \dots, X_R)
```

where the Euclidean factor domain $\mathcal D$ is either the hyperrectangle $\mathcal B$ (`fortchuck_solve`) or the singly-linearly-constrained box / generalized simplex $\mathcal C$ (`fortchuck_solve_constrained`), paired with Stiefel factors:

```math
\mathcal B = \left\lbrace z \in \mathbb R^d : l_i \le z_i \le u_i \right\rbrace, \qquad \mathcal C = \left\lbrace z \in \mathbb R^d : l_i \le z_i \le u_i, \; a^\top z = b \right\rbrace, \qquad \mathrm{St}(n_r, p_r) = \left\lbrace X_r \in \mathbb R^{n_r \times p_r} : X_r^\top X_r = I_{p_r} \right\rbrace
```

with $-\infty \le l_i \le u_i \le +\infty$ (supporting two-sided bounds, one-sided bounds, unconstrained coordinates, fixed variables `l_i = u_i`,
unit simplices $\sum_i z_i = 1$ on $z \ge 0$, capped portfolio simplices,
and Fermi--Dirac trace constraints $\sum_i z_i = N_e$ on $[0, 1]^d$)
and $1 \le p_r \le n_r$ for each $r = 1, \dots, R$
(recovering unit spheres at `p_r = 1` and orthogonal / special orthogonal groups $\mathrm{O}(n_r), \mathrm{SO}(n_r)$ when `p_r = n_r`).
A single zero-allocation solver engine seamlessly covers all four structural regimes ($d + R \ge 1$):

1. **Pure Euclidean Box** $(R = 0, \; d \ge 1)$: Coordinatewise-preconditioned active-set masked `L-BFGS-B` on $\mathcal B$ (or singly-linearly-constrained `SLBQP` quasi-Newton on $\mathcal C$).
2. **Pure Multi-Stiefel** $(d = 0, \; R \ge 1)$: Lock-free batched-panel Riemannian `LRBFGS` (or Alternating Barzilai--Borwein when $m = 0$) on $\prod_{r=1}^R \mathrm{St}(n_r, p_r)$.
3. **Coupled Box x Multi-Stiefel** $(d \ge 1, \; R \ge 1, \; \mathcal D = \mathcal B)$: Joint quasi-Newton capturing off-diagonal cross-Hessian blocks $\nabla^2_{z, X_r} f \neq 0$ while maintaining strict primal feasibility at every objective evaluation.
4. **Coupled Linear-Box / Simplex x Multi-Stiefel** $(d \ge 1, \; R \ge 0, \; \mathcal D = \mathcal C)$: Rank-1 KKT Nullspace product quasi-Newton paired with exact Dai--Fletcher / Cominetti--Mascarenhas projection $P_{\mathcal C}$ and isotropic Spectral Projected Gradient (`SPG`) fallback (`v0.2.0`).

### 1.2 Critical Audit and Reconciliation of Phase 1 & `v0.2.0` Findings

Auditing the Phase 1 Stiefel (`ARXIV_MANIFOLD_SURVEY.md`) and Coupled Box-Manifold (`ARXIV_COUPLED_BOX_MANIFOLD_SURVEY.md`) surveys alongside the `v0.1.0` blind empirical telemetry (`results/blind_benchmark_report.md`) resolves nine subtle numerical and systems interactions:

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
7. **Batched Level-3 Panel Cayley Transport & Division-Hoisted Register Micro-Kernels (`v0.2.0` Pillar 1)**:
   In `v0.1.0`, `B7B` (`St(200,8) x St(24,24)`) required `619 ms` at `m = 7` (`500` evals) vs `308 ms` at `m = 0` (`627` evals) because `fc_lbfgs_transport_and_push` invoked `fc_cayley_transport_vec` $2 m_{\mathrm{cur}} + 2 = 16$ separate times per factor, executing `80` unblocked micro-`GEMM`s in Branch A and `9,216` scalar hardware divisions inside RHS-outer `fc_lu_solve` in Branch B.
   Packing all $K_{\mathrm{batch}} = 2 m_{\mathrm{cur}} + 2$ tangent matrices into a single contiguous
   $(n_r, K_{\mathrm{batch}} p_r)$ panel inside `fc_cayley_transport_batch` — backed by `4x4` / `1x4` register-blocked `fc_gemm_*` and division-hoisted `4`-RHS `fc_lu_solve*` — collapses `16` transport calls per factor into **1** Level-3 panel operation with zero heap allocation (`3x--5x` faster inner transport).
8. **Coordinatewise Diagonal Secant Box Preconditioning & Safeguarded Hermite Arc Interpolation (`v0.2.0` Pillar 2)**:
   In `v0.1.0`, the only remaining missed runs (`8/180` across `B2A` at `d = 750` and `B5` at `d = 110`) stemmed from (i) applying a single scalar $\gamma_k^{\text{box}}$ to `d - m` unpreconditioned box directions when eigenvalues and uniquenesses differ by `1e2--1e3x` across coordinates, and (ii) naive `tau <- 0.5 * tau` backtracking wasting 2--3 evaluations per rejected curved-valley step.
   Maintaining a persistent coordinatewise diagonal inverse-curvature estimate `diag_box(1:d)` clamped into $[\max(\gamma_{\min}, 10^{-3}\gamma_k^{\text{box}}), \min(\gamma_{\max}, 10^3\gamma_k^{\text{box}})]$, paired with safeguarded quadratic/cubic Hermite backtracking (`tau_new in [0.1*tau, 0.5*tau]`) and Cauchy free-breakpoint probing, resolves intra-box anisotropy with $O(d)$ work.
9. **Rank-1 KKT Nullspace Product Two-Loop for Non-Isotropic Linear-Box Optimization (`v0.2.0` Pillar 3)**:
   When $\mathcal C = \lbrace z \in [l,u]^d : a^\top z = b \rbrace$, applying Euclidean projection
   $P_{\mathcal C}(z_k - \widetilde H_k g_k)$ directly to a non-isotropically preconditioned direction
   ($D_k \neq \gamma I$ plus L-BFGS rank $2m$ updates) can produce strict **ascent** (`phi'(0^+) > 0`)
   because $\widetilde H_k$ rotates `gz_k` across the hyperplane $a^\top \delta z = 0$.
   Enforcing the KKT hyperplane condition inside the $\widetilde H_k^{-1}$ metric via the **Rank-1 KKT Nullspace Product Two-Loop** (`u = -H_k g_tilde`, `w = H_k a_tilde`, `d_free = u - ((<a_tilde, u> - delta_b) / <a_tilde, w>) * w`, costing just 1 extra `GEMM`-free two-loop sweep and automatically adjusting Stiefel cross-curvature!)
   plus isotropic `SPG` fallback `P_C(z_k - alpha_box * gz_k) - z_k` guarantees exact primal feasibility $z(\tau) \in \mathcal C$ and unconditional strict descent $\phi'(0^+) \lt 0$ (Lean 4 Theorems 6 & 7).

### 1.3 Adversarial Failure-Mode Mitigation Matrix

| ID | Failure Mode | Root Mathematical / Systems Cause | Mandatory Mitigation in `fortchuck` |
| :--- | :--- | :--- | :--- |
| FM-1 | Rosen Boundary Jamming | Naive tangent Armijo $\phi(\tau) \le \phi(0) + c_1 \tau \phi'(0^+)$ rejects unit steps when `z_i - l_i ~ 1e-14` clamps early. | Bertsekas active band `A_eps(z_k)` + Generalized Arc-Armijo realized chord `Delta_k(tau)`. |
| FM-2 | Lagrange-Multiplier Stiefel Bias | Clamped `z_i = l_i` has positive multiplier $\lambda_i^\ast > 0$ and cross-coupled `y_{j,box,i} /= 0`, biasing Stiefel steps (`arXiv:2605.10573`). | Dynamic projection `P_{free,k}` masking `q`, `s_j`, and `y_j` on `F_eps(z_k)` inside every two-loop call. |
| FM-3 | Indefinite Masked Curvature | Full pair $\langle s_j, y_j \rangle > 0$ can turn nonpositive after positive-curvature box coordinates clamp into `A_eps(z_k)`. | Per-iteration dynamic filter `sigma_tilde_j >= 1e-10 * ||s_tilde_j|| * ||y_tilde_j||` before using slot `j`. |
| FM-4 | Pre-Partition Ascent Bug | Applying $H_k$ before active-set partitioning couples binding and free coordinates, violating descent (`Bertsekas 1982`). | Partition `A_fixed`, `A_eps`, and `F_eps` before applying positive-definite `H_tilde_k`. |
| FM-5 | Cross-Block Scale Starvation | Single scalar $\gamma_k$ starves softer blocks under $10^5$ box-vs-Stiefel Hessian mismatch (e.g., ML-CPC). | Cosine-guarded persistent per-block BB2 center scaling `gamma_blk(1:R+1)`. |
| FM-6 | Discontinuous Stopping Near Bounds | Tangent-cone norm jumps from $\lambda_i^\ast$ to $0$ at `z_i = l_i`. | 1-Lipschitz residual `||z - P_D(z - gz)||_inf` ($\mathcal D \in \lbrace \mathcal B, \mathcal C \rbrace$) paired with Stiefel Riemannian gradient norm. |
| FM-7 | Antipodal / Ill-Conditioned Cayley Step | Large trial velocity approaches $\det(I + X^\top Y) = 0$ or ill-conditioned `GETRF` pivot. | Velocity cap `tau_0 * max_r ||eta_{r,k}||_can <= 2*pi` + `GETRF` reciprocal pivot guard (`1e-12`) halving `tau`. |
| FM-8 | QR Column Sign Flips & Callback Drift | Naive `GEQRF` flips columns by $-1$; post-callback polish invalidates cached $f, \nabla f$. | Pre-callback `fc_geqrf_pos` ($R_{ii} > 0$) whenever `||X_r^T X_r - I||_F > 1e-13`. |
| FM-9 | `-ffpe-trap` Overflow on `±DBL_MAX` | `u_i - l_i` or `a_i * l_i` overflows `real64` when bounds are `±huge`. | Half-widths `0.5_dp*u_i - 0.5_dp*l_i`, `a_i /= 0` infinity flags, and $\lVert a \rVert_\infty$-normalized root brackets. |
| FM-10 | Premature `ftol` Stop on Curved Valleys | Single flat step triggers loose `ftol` while $\mathcal E_{\text{KKT}} \gg \text{gtol}$ on Rosenbrock or quartic valleys. | Default `ftol = 1e-14` gated by 2 consecutive small $\Delta f$ steps or `E_KKT <= 100 * gtol`. |
| FM-11 | Fragmented `m = 7` Transport Overhead | `2*m_cur + 2` per-vector Cayley transport calls thrash L1 cache and issue $O(n^2 m)$ scalar divisions (`B7B`). | Contiguous `(n_r, K_batch*p_r)` `fc_cayley_transport_batch` + `4x4` `GEMM` + division-hoisted `4`-RHS `LU`. |
| FM-12 | Intra-Box Coordinate Anisotropy & Halving Waste | Scalar `gamma_box` under-scales `d - m` heterogeneous coordinates (`B2A`, `B5`); `0.5*tau` wastes backtracks. | Persistent coordinatewise `diag_box(1:d)` in $[10^{-3}\gamma_{\text{box}}, 10^3\gamma_{\text{box}}]$ + safeguarded quadratic/cubic Hermite arc interpolation. |
| FM-13 | Non-Isotropic Hyperplane Projection Ascent | Euclidean $P_{\mathcal C}(z_k - \widetilde H_k g_k)$ ascends when non-isotropic `H_tilde_k` tilts `-g_k` across `a^T dz = 0`. | Rank-1 KKT Nullspace Two-Loop `H_proj,k` (`<a_tilde, d_free> = delta_b`) + boundary sign guard + isotropic `SPG` fallback. |
| FM-14 | Degenerate / Infeasible Knapsack Kinks | Unreachable $a^\top z = b$, singleton boundary faces, or flat kinks $\psi'(\nu) = 0$ stall pure Newton root-finders. | Exact $\underline b \le b \le \overline b$ certificate (`FC_ERR_INFEASIBLE_LIN`), singleton fast-path, and Newton-secant + safeguarded bisection in $\le 60$ steps. |

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

3. **Two-Metric Coupled Search Direction (`v0.2.0` Coordinatewise Diagonal Box Preconditioner)**:
   - On $\mathcal A_{\text{fixed}}$: set `dz(i) = 0`.
   - On `A_eps(z_k)` (`mask_box(i) == 1`): set $d_{z,k,i} = -\gamma_{k,i}^{\text{act}} g_{z,k,i}$,
     where the 1D active-band scale uses the persistent safeguarded coordinatewise diagonal preconditioner `diag_box(i)` ($D_{k,i}$ from §2.4):

```math
\gamma_{k,i}^{\text{act}} = \begin{cases} \mathrm{clip}\bigl(s_{k-1,\text{box},i} / y_{k-1,\text{box},i}, \, d_{\min,k}, \, d_{\max,k}\bigr) & \text{if } k \ge 1 \text{ and } s_{k-1,\text{box},i} y_{k-1,\text{box},i} \ge 10^{-12} s_{k-1,\text{box},i}^2, \\ D_{k,i} & \text{otherwise.} \end{cases}
```

   - On the free subspace `S_{free,k}`: apply the positive-definite coordinatewise-preconditioned masked product L-BFGS operator $\widetilde H_k \succ 0$ (§2.4):

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

In the **center scaling step**, equilibrate Stiefel blocks via cosine-guarded persistent BB2 scales $\gamma_k^{(r)} \in [10^{-10}, 10^{10}]$
and precondition free box coordinates individually via the safeguarded coordinatewise diagonal inverse-curvature vector `diag_box(1:d)` ($D_{k,i}$):

```math
q_{\text{box},i} \leftarrow D_{k,i} \, q_{\text{box},i} \;\; (i \in \mathcal F_{\epsilon_k}), \qquad q_r \leftarrow \gamma_k^{(r)} q_r, \qquad A_{q,r} \leftarrow \gamma_k^{(r)} A_{q,r}
```

where each block baseline $\gamma_k^{(b)}$ ($b \in \lbrace \text{box}, 1, \dots, R \rbrace$) starts at $1.0$ ($k = 0$) and updates to
$\mathrm{clip}(\sigma_{k-1,b}^{sy} / \sigma_{k-1,b}^{yy}, 10^{-10}, 10^{10})$
whenever $\sigma_{k-1,b}^{sy} \ge 10^{-3} \sqrt{\sigma_{k-1,b}^{ss} \sigma_{k-1,b}^{yy}}$
and $\sigma_{k-1,b}^{yy} > 10^{-30}$,
otherwise retaining $\gamma_{k-1}^{(b)}$.
Around the updated box BB2 baseline $\gamma_k^{\text{box}}$, define the condition-bounded coordinatewise trust interval:

```math
d_{\min,k} = \max\!\bigl(\gamma_{\min}, \, 10^{-3} \gamma_k^{\text{box}}\bigr), \qquad d_{\max,k} = \min\!\bigl(\gamma_{\max}, \, 10^{3} \gamma_k^{\text{box}}\bigr), \qquad \gamma_{\min} = 10^{-10}, \quad \gamma_{\max} = 10^{10}.
```

At $k = 0$, initialize $D_{0,i} = 1.0$ for $i = 1, \dots, d$.
After step $k \ge 0$ is accepted with box displacement `s_{k,box}` and gradient difference `y_{k,box}`, update each coordinate $i \in \lbrace 1, \dots, d \rbrace$ (Gilbert--Lemaréchal / L-BFGS-B diagonal secant smoothing):
- If $s_{k,\text{box},i} \, y_{k,\text{box},i} \ge 10^{-12} s_{k,\text{box},i}^2$
  and $\lvert y_{k,\text{box},i} \rvert > 10^{-30}$,
  set $d_{\mathrm{sec},i} = \mathrm{clip}(s_{k,\text{box},i} / y_{k,\text{box},i}, \, d_{\min,k+1}, \, d_{\max,k+1})$,
  updating $D_{1,i} = d_{\mathrm{sec},i}$ on $k = 0$
  and $D_{k+1,i} = \mathrm{clip}(0.5 D_{k,i} + 0.5 d_{\mathrm{sec},i}, \, d_{\min,k+1}, \, d_{\max,k+1})$ for $k \ge 1$.
- Otherwise (e.g., coordinate $i$ remained clamped or had nonpositive 1D curvature),
  set $D_{1,i} = \gamma_1^{\text{box}}$ on $k = 0$ so initially clamped coordinates inherit the problem's global box scale before release,
  and set $D_{k+1,i} = \mathrm{clip}(D_{k,i}, \, d_{\min,k+1}, \, d_{\max,k+1})$ for $k \ge 1$.

Because $H_{k,0} = \mathrm{diag}(D_k, \gamma_k^{(1)} I, \dots, \gamma_k^{(R)} I) \succ 0$ has condition number bounded on the box block by $\le 10^6$,
the masked product operator $\widetilde H_k \succ 0$ preserves strict positive-definiteness unconditionally while preconditioning all $d$ heterogeneous coordinates even when $d \gg m$.
Modularly, the backward/center/forward sweep is factored into a reusable `pure subroutine fc_lbfgs_apply_H` applying $+\widetilde H_k$ to any free-subspace vector `(v_box, V_St, Av_St)` (invoked once for unconstrained box steps and twice for the Rank-1 KKT Nullspace Two-Loop in §2.9).

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

#### Step 3: Batched Post-Step Panel Transport and Curvature Positivity Damping

Once $p_{k+1}$ is accepted at step size `tau_k`,
pack all $K_{\mathrm{batch}} = 2 m_{\mathrm{cur}} + 2$ tangent matrices for factor $r$ (`S_{1..m_cur,r}`, `Y_{1..m_cur,r}`, `tau_k * eta_{r,k}`, and `g_{r,k}`)
into a single contiguous column-major panel $\Pi_r \in \mathbb R^{n_r \times (p_r K_{\mathrm{batch}})}$ inside `scratch`
and transport the entire panel in **one** Level-3 call to `fc_cayley_transport_batch` (§2.5), forming:

```math
s_k = \Bigl(z_{k+1} - z_k, \; \mathcal T_{\tau_k \eta_{1,k}}^{\mathrm{Cay}}(\tau_k \eta_{1,k}), \; \dots\Bigr), \qquad y_k = \Bigl(g_{z,k+1} - g_{z,k}, \; g_{1,k+1} - \mathcal T_{\tau_k \eta_{1,k}}^{\mathrm{Cay}}(g_{1,k}), \; \dots\Bigr).
```

Let $\tilde s_k, \tilde y_k$ denote their projections onto the updated free subspace at `p_{k+1}`,
and let $B_{k,0} s_k$ scale box coordinate $i$ by `1 / diag_box(i)`
and Stiefel block $r$ by $1 / \gamma_k^{(r)}$:
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

### 2.5 Dual-Regime Cayley Retraction, Batched Panel Transport, and Pre-Callback Polish

For each Stiefel factor $r \in \lbrace 1, \dots, R \rbrace$, project $\eta_{r,k} \leftarrow P_{T_{X_{r,k}}}(\eta_{r,k})$ and define:

```math
\tilde\eta_r = \eta_{r,k} - \frac{1}{2} X_{r,k}(X_{r,k}^\top \eta_{r,k}), \qquad W_{\eta_r} = \tilde\eta_r X_{r,k}^\top - X_{r,k} \tilde\eta_r^\top \in \mathfrak{so}(n_r), \qquad Q_{\eta_r}(\tau) = \left(I_{n_r} - \frac{\tau}{2} W_{\eta_r}\right)^{-1}\!\left(I_{n_r} + \frac{\tau}{2} W_{\eta_r}\right).
```

Both linear-algebra regimes evaluate the exact same retraction $Q_{\eta_r}(\tau) X_{r,k}$
and isometric batched panel transport $Q_{\eta_r}(\tau_k) \Pi_r$
for $\Pi_r = \lbrack \zeta^{(1)}, \dots, \zeta^{(K)} \rbrack \in \mathbb R^{n_r \times N_{\mathrm{col}}}$
with `N_col = p_r * K` and $K = 2 m_{\mathrm{cur}} + 2$:

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
3. **Batched Level-3 Isometric Cayley Panel Transport (`fc_cayley_transport_batch`)**:
   Instead of $K = 2 m_{\mathrm{cur}} + 2$ fragmented single-vector calls,
   form $C_r = I_{p_r} - \tfrac{\tau_k}{4} A_r$ **once**
   and execute a single contiguous $n_r \times N_{\mathrm{col}}$ panel pass
   (`N_col = p_r * K`, reusing factored `J_lu` with $p_r$ divisions total):

```math
R_1 = X_{r,k}^\top \Pi_r, \quad Z_1 \leftarrow C_r R_1 - \frac{\tau_k}{2} \tilde\eta_r^\top \Pi_r, \quad J_r(\tau_k) Z_1 \leftarrow Z_1, \quad R_1 \leftarrow C_r Z_1 - R_1, \quad \Pi_r \leftarrow \Pi_r + \tau_k \tilde\eta_r Z_1 + 2 X_{r,k} R_1.
```

   (When `p_r == 1`, hoist `inv_j11 = 1.0_dp / J_lu(1,1)` once and fuse all $K$ columns in a single unit-stride loop requiring `0` scratch words beyond `Panel`; when `p_r > 1`, requires only $2 p_r N_{\mathrm{col}} + p_r^2$ scratch words beyond `Panel`.)

#### Branch B (`2 * p_r >= n_r`, incl. Orthogonal Groups): Direct LU with Resolvent Doubling

1. **Outer Setup** ($2 n_r p_r^2 + 2 n_r^2 p_r$ FLOPs once per outer iteration $k$):
   Compute $P_r = \tilde\eta_r X_{r,k}^\top$ from `eta_tilde_r`,
   and form $W_{\eta_r} = P_r - P_r^\top \in \mathfrak{so}(n_r)$.
2. **Trial Retraction** (at step $\tau$, costing $\tfrac{2}{3} n_r^3 + 2 n_r^2 p_r$ FLOPs):
   Using $Q_{\eta_r}(\tau) = 2 M_{n_r}(\tau)^{-1} - I_{n_r}$
   where $M_{n_r}(\tau) = I_{n_r} - \tfrac{\tau}{2} W_{\eta_r}$, factor `M_{n_r}(tau)` via `fc_lu_factor`,
   solve $M_{n_r}(\tau) Z_r = 2 X_{r,k}$ via `fc_lu_solve`,
   and set $Y_r(\tau) = Z_r - X_{r,k}$.
3. **Batched Level-3 Isometric Cayley Panel Transport (`fc_cayley_transport_batch`)**:
   Set $Z_{\Pi} = 2 \Pi_r \in \mathbb R^{n_r \times N_{\mathrm{col}}}$,
   execute **one** division-hoisted `4`-RHS blocked `fc_lu_solve` for $M_{n_r}(\tau_k) Z_{\Pi} = 2 \Pi_r$
   (streaming `M_lu` in `4`-column register tiles with $n_r$ divisions total instead of `n_r * N_col`),
   and set $\Pi_r \leftarrow Z_{\Pi} - \Pi_r$.

#### Cache-Tiled SIMD `GEMM` & Division-Hoisted Multi-RHS `LU` Micro-Kernels (`v0.2.0` Pillar 1)

Inside `fortchuck_linalg.f90` (`100% pure`, zero automatic stack arrays):
1. **`fc_gemm_tn`**: L1/L2 cache-tiled (`MC = 64, NC = 64, KC = 128`) with an explicit `4 x 4` register micro-kernel holding `16` scalar accumulators `c00..c33` across unit-stride `l = l0 .. min(l0+KC-1, k)` (`8` vector loads feeding `16` independent FMAs per step, saturating dual AVX2/AVX-512 FMA pipelines).
2. **`fc_gemm_nn` & `fc_gemm_nt`**: Cache-tiled (`NC = 64, KC = 128, MC = 128`) with a `1 x 4` column-unrolled register micro-kernel streaming each contiguous `A(i, l)` slice once across `4` output columns `C(i, j..j+3)` (`4x` reduction in `A` memory traffic); optional `#ifdef FORTCHUCK_USE_BLAS` threshold lowered to `m*n*k > 8192`.
3. **`fc_lu_solve` & `fc_lu_solve_trans`**: Hoist `inv_ukk = 1.0_dp / LU(k, k)` outside all RHS loops ($n$ reciprocals total regardless of `nrhs`) and process RHS columns in `JB = 4` register panels with unit-stride `i` FMA inner loops.

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

### 2.6 Non-Jamming Generalized Arc-Armijo / Zhang--Hager Globalization with Safeguarded Hermite Interpolation

Starting from $\tau = \tau_{\mathrm{init}}$
(setting `tau_init = 1.0` for $k \ge 1$
and $\tau_{\mathrm{init}} = \min(1.0, 1.0 / \max(1.0, \lVert g_0 \rVert_{p_0}))$ at $k = 0$, after applying the antipodal Stiefel velocity cap),
compute the coupled realized box-chord + Stiefel tangent reference variation along $z(\tau) = P_{\mathcal D}(z_k + \tau d_{z,k})$ ($\mathcal D \in \lbrace \mathcal B, \mathcal C \rbrace$):

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

#### Safeguarded Quadratic / Cubic Hermite Backtracking & Cauchy Breakpoint Probe (`v0.2.0` Pillar 2)

Prior to entering the backtracking loop (`d > 0`), `fc_box_breakpoint_cap` scans free box coordinates (`mask_box(i) == 0`, $d_{z,k,i} \neq 0$)
to locate the earliest positive bound-hit breakpoint $\tau_{\mathrm{brk,min}} > 0$ (if any finite free wall is encountered).
Whenever a trial step $\tau$ produces a finite `f_trial` (`rc == 0`) that **rejects** the Zhang--Hager Armijo test, replace naive `tau <- 0.5 * tau` halving with safeguarded polynomial Hermite interpolation over $[0.1\tau, \, 0.5\tau]$
using $f_0 = f(p_k)$
and realized effective initial slope $s_0(\tau) = \Delta_k(\tau) / \tau \lt 0$:

1. **First Rejected Trial (`ls_step == 1` or no prior finite trial) — Quadratic Hermite + Breakpoint Probe**:
   If $D_q = 2\bigl(f_{\mathrm{trial}} - f_0 - \Delta_k(\tau)\bigr) > 10^{-30}$, form the exact quadratic minimizer:

```math
\tau_{\mathrm{quad}} = \frac{-\Delta_k(\tau) \, \tau}{2\bigl(f_{\mathrm{trial}} - f_0 - \Delta_k(\tau)\bigr)}
```

   (otherwise $\tau_{\mathrm{quad}} = 0.5\tau$).
   If free box coordinates slammed into bounds before $0.5\tau$ (`n_free_hit > 0` and $0.1\tau \le \tau_{\mathrm{brk,min}} \lt \min(0.5\tau, \tau_{\mathrm{quad}})$),
   blend $\tau_{\mathrm{cand}} = \tau_{\mathrm{brk,min}}$ so the next probe tests the exact kink-free interior segment;
   else set `tau_cand = tau_quad`, and clamp $\tau_{\mathrm{new}} = \mathrm{clip}(\tau_{\mathrm{cand}}, \, 0.1\tau, \, 0.5\tau)$.
2. **Subsequent Rejected Trials (`ls_step >= 2` with finite `(tau_prev, f_prev)`) — Cubic Hermite Minimizer**:
   When $f_{\mathrm{trial}} > f_0$,
   fit the unique cubic $P(\alpha) = f_0 + s_0 \alpha + c_2 \alpha^2 + c_3 \alpha^3$
   interpolating `(0, f_0, s_0)`, `(tau_prev, f_prev)`, and $(\tau, f_{\mathrm{trial}})$ (Dennis--Schnabel §6.3.2):

```math
\begin{bmatrix} c_3 \\ c_2 \end{bmatrix} = \frac{1}{\tau^2 \tau_{\mathrm{prev}}^2 (\tau - \tau_{\mathrm{prev}})} \begin{bmatrix} \tau_{\mathrm{prev}}^2 & -\tau^2 \\ -\tau_{\mathrm{prev}}^3 & \tau^3 \end{bmatrix} \begin{bmatrix} f_{\mathrm{trial}} - f_0 - s_0 \tau \\ f_{\mathrm{prev}} - f_0 - s_0 \tau_{\mathrm{prev}} \end{bmatrix}.
```

   If discriminant $d_{\mathrm{cub}} = c_2^2 - 3 c_3 s_0 > 0$
   and $\lvert c_3 \rvert > 10^{-30}$,
   set $\tau_{\mathrm{cand}} = (-c_2 + \sqrt{d_{\mathrm{cub}}}) / (3 c_3)$;
   else fall back to `tau_quad`. Update `(tau_prev, f_prev) <- (tau, f_trial)`
   and set $\tau \leftarrow \mathrm{clip}(\tau_{\mathrm{cand}}, \, 0.1\tau, \, 0.5\tau)$.

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
| `FC_ERR_LS_FAILED` | `-8` | Backtracking failed down to `max_ls_steps` on both QN direction and ABB/SPG steepest-descent restart. |
| `FC_ERR_INVALID_OPTS` | `-9` | Invalid hyperparameter (`c1_armijo <= 0` or `>= 1`, `max_iter < 0`, `max_eval < 1`, unknown enum). |
| `FC_ERR_INFEASIBLE_LIN` | `-10` | Linear-box constraint $a^\top z = b$ has `NaN`/`Inf`, $\lVert a \rVert_\infty = 0$, or is infeasible on $[l, u]^d$ ($\underline b > b$ or $\overline b \lt b$). |

### 2.8 Master Solver Execution Pseudocode (`fc_solve` & `fc_solve_constrained`)

```text
1. Validate dimensions, options, pointers; compute exact unified lwork_min, liwork_min.
   If lwork == -1 or liwork == -1: write mins to work(1), iwork(1), res; return FC_SUCCESS_QUERY (0).
   If lwork < lwork_min or liwork < liwork_min: return FC_ERR_WORK_TOO_SMALL (-5).
2. Slice work(1:lwork) and iwork(1:liwork) via deterministic integer offsets (0 heap allocations).
3. Validate bounds l <= u (and if has_lin_eq: validate finite a, b, ||a||_inf > 0, and b_min <= b <= b_max
   via fc_box_lin_validate, returning FC_ERR_INFEASIBLE_LIN (-10) if infeasible).
   Project z <- P_D(z) if project_initial == 1 (P_[l,u] or fc_box_lin_project; else verify feasibility <= 1e-8).
   For r = 1..R: compute orth_err_r; if project_initial == 1 and orth_err_r > orth_tol, polish via fc_geqrf_pos
   (returning FC_ERR_INVALID_X0 if rank-deficient); if project_initial == 0 and orth_err_r > 1e-8, error.
4. Evaluate initial callback (n_eval = 1) -> f_k, gz, GX; verify finite & rc == 0.
   Compute rbox (via fc_box_kkt_residual or fc_box_lin_kkt_residual -> nu_k), Stiefel rgrad_X, E_KKT(p_0).
   If E_KKT(p_0) <= gtol: return FC_CONVERGED_GRAD (1).
   Initialize Q_zh = 1, C_zh = f_k, m_cur = 0, ring_head = 0, gamma_blk(1:R+1) = 1.0, diag_box(1:d) = 1.0.
5. For k = 0 .. max_iter - 1:
   a. Partition mask_box(1:d) in {FREE=0, ACTIVE=1, FIXED=2} (fc_box_partition if box, fc_box_lin_partition if has_lin_eq).
   b. If m_cur >= 1 and memory >= 1:
        If .not. has_lin_eq: compute (dz, eta_X) via fc_lbfgs_direction (diag_box + GEMM-free two-loop + active step).
        Else: compute (dz, eta_X) via Rank-1 KKT Nullspace Two-Loop (u = -H_k g_tilde, w = H_k a_tilde, 1-pass sign guard).
        used_qn = .true.
      Else:
        Compute ABB / Isotropic SPG steepest-descent direction (dz, eta_X); used_qn = .false.
   c. Check sufficient descent phi'(0^+) <= -1e-12 * v_0 * g_0.
      If violated: flush m_cur = 0, increment n_fallbacks, recompute ABB / Isotropic SPG direction, used_qn = .false.
   d. Line-search loop (up to 2 attempts: QN first, then 1 ABB/SPG restart if QN exhausts max_ls_steps):
      - Run fc_cayley_setup for r = 1..R; scan fc_box_breakpoint_cap; cap tau_init by 2*pi / max_r ||eta_{r,k}||_can.
      - For ls = 1 .. max_ls_steps:
          * z_trial = P_D(z + tau * dz); check chord slope Delta_k(tau) <= -1e-12 * step_norm_sq < 0; if not, tau *= 0.5, cycle.
          * Retract X_trial_r = R^Cay(tau * eta_{r,k}) for r = 1..R; if any LU pivot guard triggers, tau *= 0.5, cycle.
          * For r = 1..R: if ||X_trial_r^T X_trial_r - I||_F > orth_tol, apply fc_geqrf_pos(X_trial_r) (polished_step = .true.).
          * If n_eval >= max_eval: populate res at accepted p_k and return FC_MAX_EVAL_REACHED (5).
          * Call eval_fg(z_trial, X_trial) -> f_trial, gz_trial, GX_trial (n_eval += 1).
          * If rc < 0: return FC_ERR_CALLBACK_ABORT (-7). If rc > 0 or .not. finite(f_trial, grads): tau *= 0.5, cycle.
          * If f_trial <= C_zh + c1_armijo * Delta_k(tau): step ACCEPTED, exit ls loop.
          * Safeguarded Quadratic/Cubic Hermite + Cauchy breakpoint update: tau <- clip(tau_hermite, 0.1*tau, 0.5*tau).
      - If not accepted and used_qn: flush m_cur = 0, n_fallbacks += 1, set ABB/SPG direction, used_qn = .false., retry d.
      - If not accepted and .not. used_qn: return FC_ERR_LS_FAILED (-8).
   e. Compute step_diff, f_diff = |f_trial - f_k|; update Zhang-Hager (Q_zh, C_zh).
   f. Batch-transport stored Stiefel history + (tau * eta_X, rgrad_X) via fc_cayley_transport_batch (plus re-proj if polished_step).
   g. Commit z = z_trial, X = X_trial, f_k = f_trial, gz = gz_trial, GX = GX_trial; compute new rbox, rgrad_X, E_KKT(p_{k+1}).
   h. Update gamma_blk(1:R+1), diag_box(1:d), and push (s_k, y_k) into ring buffer if memory >= 1 (Cautious or Powell).
   i. Check stopping criteria in strict order: FC_CONVERGED_GRAD (1), FC_CONVERGED_FTOL (2), FC_CONVERGED_XTOL (3).
6. Return FC_MAX_ITER_REACHED (4).
```

### 2.9 Coupled Singly-Linearly-Constrained Box and Simplex Engine (`v0.2.0` Pillar 3)

When invoked via `fc_solve_constrained` / `fortchuck_solve_constrained` (`has_lin_eq == .true.`, requiring $d \ge 1$),
`fortchuck` optimizes over $\mathcal C \times \prod_{r=1}^R \mathrm{St}(n_r, p_r)$
where $\mathcal C = \left\lbrace z \in \mathbb R^d : l_i \le z_i \le u_i, \; a^\top z = b \right\rbrace$:

1. **Overflow-Safe Primal Feasibility Certificate (`fc_box_lin_validate`)**:
   Reject `NaN`/`Inf` in $a$ or $b$, or $\alpha_{\max} = \max_{1 \le i \le d} \lvert a_i \rvert = 0$, with `FC_ERR_INFEASIBLE_LIN (-10)`.
   Skipping coordinates with $a_i = 0$ (preventing `0 * Inf` traps under `-ffpe-trap=invalid,overflow`),
   compute attainable bounds $\underline b = \sum_{a_i > 0} a_i l_i + \sum_{a_i \lt 0} a_i u_i$
   and $\overline b = \sum_{a_i > 0} a_i u_i + \sum_{a_i \lt 0} a_i l_i$
   (tracking `has_unbounded_lo` / `has_unbounded_hi` flags whenever a contributing bound has magnitude $\ge 0.5 \cdot \text{huge}$ via `FC_HUGE / 2`).
   With $\tau_{\mathrm{feas}} = 10^{-14}(1 + \lvert b \rvert)$, return `FC_ERR_INFEASIBLE_LIN (-10)` iff `(not has_unbounded_lo and b_min > b + tau_feas)` or `(not has_unbounded_hi and b_max < b - tau_feas)`.
   If $b - \underline b \le \tau_{\mathrm{feas}}$ or `b_max - b <= tau_feas`, `fc_box_lin_project` resolves the singleton boundary face analytically in $O(d)$ flops.

2. **Scalar-Memory Semismooth Newton-Secant + Safeguarded Bisection Projection (`fc_box_lin_project`)**:
   To solve $\min_{z \in \mathcal C} \tfrac{1}{2} \lVert z - y \rVert_2^2$ with zero array workspace,
   normalize $\hat a_i = a_i / \alpha_{\max} \in [-1, 1]$
   and $\hat b = b / \alpha_{\max}$.
   By KKT optimality (Lean 4 Theorem 6), the unique minimizer is $z^\ast(\hat\nu^\ast) = P_{[l, u]}(y - \hat\nu^\ast \hat a)$
   where $\hat\nu^\ast = \alpha_{\max} \nu^\ast$ solves the monotone nonincreasing piecewise-linear root equation:

```math
\psi(\hat\nu) = \sum_{i=1}^d \hat a_i \min\!\bigl(u_i, \max(l_i, y_i - \hat\nu \hat a_i)\bigr) - \hat b = 0, \qquad \psi'(\hat\nu) = -\sum_{i : l_i \lt y_i - \hat\nu \hat a_i \lt u_i} \hat a_i^2 \le 0.
```

   - **Initial Bracket** $[\hat\nu_{\mathrm{lo}}, \hat\nu_{\mathrm{hi}}]$:
     Test the unconstrained hyperplane root $\hat\nu_0 = \bigl(\sum_i \hat a_i P_{[l_i,u_i]}(y_i) - \hat b\bigr) / \sum_i \hat a_i^2$ first; if already converged, exit in 1 pass.
     Otherwise bracket $[\hat\nu_{\mathrm{lo}}, \hat\nu_{\mathrm{hi}}]$ from finite breakpoint ratios
     `((y_i - u_i)/a_hat_i, (y_i - l_i)/a_hat_i)` (falling back to geometric step doubling from $\hat\nu_0$ whenever any relevant bound is unbounded or a tiny `|a_hat_i|` risks exponent overflow),
     guaranteeing $\psi(\hat\nu_{\mathrm{lo}}) \ge 0 \ge \psi(\hat\nu_{\mathrm{hi}})$.
   - **Dai--Fletcher / Cominetti--Mascarenhas Iteration** ($\le 60$ scalar passes):
     Evaluate $\psi(\hat\nu)$ (using Neumaier compensated summation) and exact active slope $\psi'(\hat\nu)$ in one $O(d)$ pass,
     tighten the bracket endpoint matching $\mathrm{sign}(\psi(\hat\nu))$,
     and form the exact affine semismooth Newton step $\hat\nu_{\mathrm{N}} = \hat\nu - \psi(\hat\nu)/\psi'(\hat\nu)$
     when $\psi'(\hat\nu) \lt -10^{-30}$ (which hits the exact root in **1** step once inside the optimal active face!),
     falling back to secant interpolation across $[\hat\nu_{\mathrm{lo}}, \hat\nu_{\mathrm{hi}}]$ at flat kinks,
     and safeguarding inside $[\hat\nu_{\mathrm{lo}} + 0.05\Delta\hat\nu, \, \hat\nu_{\mathrm{hi}} - 0.05\Delta\hat\nu]$ via midpoint bisection whenever $\lvert \psi \rvert$ fails to halve.
     Terminates with $\nu_{\mathrm{out}} = \hat\nu / \alpha_{\max}$ as soon as:

```math
\lvert a^\top z - b \rvert \le 10^{-14}\bigl(1 + \lvert b \rvert + \lVert a \rVert_1 \lVert z \rVert_\infty\bigr) \quad\text{or}\quad \lvert \hat\nu_{\mathrm{hi}} - \hat\nu_{\mathrm{lo}} \rvert \le 2 \, \varepsilon_{\mathrm{mach}}\bigl(1 + \lvert \hat\nu_{\mathrm{lo}} \rvert + \lvert \hat\nu_{\mathrm{hi}} \rvert\bigr).
```

3. **Exact 1-Lipschitz Constrained KKT Residual & Multiplier Sign (`fc_box_lin_kkt_residual`)**:
   Projecting $y = z_k - g_{z,k}$ onto $\mathcal C$ yields
   $z_{\mathrm{proj}} = P_{\mathcal C}(z_k - g_{z,k}) = P_{[l, u]}\bigl(z_k - (g_{z,k} + \nu_k a)\bigr)$
   with multiplier $\nu_k$.
   Define the constrained residual $r_{\mathcal C}(z_k) = z_k - z_{\mathrm{proj}}$,
   infinity norm $\mathcal E_{\mathcal C}(p_k) = \lVert r_{\mathcal C}(z_k) \rVert_\infty$,
   coupled residual $\mathcal E_{\text{KKT}}(p_k) = \max\bigl(\mathcal E_{\mathcal C}(p_k), \mathcal E_{\mathrm{St}}(p_k)\bigr)$,
   and reduced Lagrangian gradient $\hat g_{z,k} = g_{z,k} + \nu_k a$.
   Note the exact sign: at `z_{k,i} = l_i`, clamping $z_{k,i} - \hat g_{z,k,i} \le l_i$ holds
   iff $\hat g_{z,k,i} \ge 0$;
   at `z_{k,i} = u_i`, clamping holds iff $\hat g_{z,k,i} \le 0$.

4. **Guaranteed Strict Descent via Rank-1 KKT Nullspace Product Two-Loop + Isotropic SPG Fallback**:
   - **Boundary Active-Face Partition (`fc_box_lin_partition`)**:
     With boundary tolerance $\tau_{\mathrm{bnd},i} = 10^{-12}(1 + \lvert z_{k,i} \rvert)$,
     tag `mask_box(i) = FC_BOX_FIXED (2)` if `l_i == u_i` ($d_{z,k,i} = 0$);
     tag `mask_box(i) = FC_BOX_ACTIVE (1)` iff $(z_{k,i} \le l_i + \tau_{\mathrm{bnd},i} \text{ and } \hat g_{z,k,i} \ge 0)$
     or $(z_{k,i} \ge u_i - \tau_{\mathrm{bnd},i} \text{ and } \hat g_{z,k,i} \le 0)$
     (snapping any tiny `<= 1e-12` boundary residual `dz_i = l_i - z_i` or `u_i - z_i`
     and accumulating $\delta_b = -\sum_{i \in \mathcal A_k} a_i d_{z,k,i}$); else tag `mask_box(i) = FC_BOX_FREE (0)`.
   - **Rank-1 KKT Nullspace Product Two-Loop**:
     Let $\tilde g = (g_{z,k,\mathcal F_k}, g_{1,k}, \dots, g_{R,k})$
     and $\tilde a = (a_{\mathcal F_k}, 0_{\mathrm{St}})$
     on the joint free subspace $\mathcal S_{\text{free},k}$. Invoke `fc_lbfgs_apply_H` twice (`0` dense matrices, `0` `GEMM`s):

```math
\tilde u = -\widetilde H_k \tilde g, \qquad \tilde w = \widetilde H_k \tilde a, \qquad \tilde d_{\mathrm{free}} = \tilde u - \frac{\langle \tilde a, \tilde u \rangle_{p_k} - \delta_b}{\langle \tilde a, \tilde w \rangle_{p_k}} \tilde w
```

   whenever $\langle \tilde a, \tilde w \rangle_{p_k} > 10^{-30} \lVert \tilde a \rVert_2^2$
   (using $\tilde d_{\mathrm{free}} = \tilde u$ if `a_{F_k} == 0` and `delta_b == 0`).
   Because $\tilde w = \widetilde H_k \tilde a$ propagates through stored cross-secant pairs
   $(s_{j,\text{box}}, y_{j,\mathrm{St}})$,
   the Stiefel search velocities $\eta_{r,k}$ automatically compensate for the linear box equality constraint!
   Furthermore, $\widetilde H_{\mathrm{proj},k} = \widetilde H_k - \tilde w \tilde w^\top / \langle \tilde a, \tilde w \rangle$ annihilates $\tilde a$,
   making $\tilde d_{\mathrm{free}}$ **100% invariant** to shifts
   $g_{z,k} \mapsto g_{z,k} + \mu a$
   so L-BFGS secants retain raw smooth $y_{k,\text{box}} = g_{z,k+1} - g_{z,k}$.
   - **One-Pass Boundary Sign Guard**:
     If any boundary coordinate in $\mathcal F_k$ points outward
     ($z_{k,i} \le l_i + \tau_{\mathrm{bnd},i}$ with `dz_i < 0`,
     or $z_{k,i} \ge u_i - \tau_{\mathrm{bnd},i}$ with `dz_i > 0`),
     freeze all such outward boundary indices into `FC_BOX_ACTIVE` (`dz(i) = 0`)
     and recompute $\tilde d_{\mathrm{free}}$ once
     so $d_{z,k} \in T_{\mathcal C}(z_k)$
     and $\dot z(0^+) = d_{z,k}$ holds exactly on `(0, tau_brk]`.
   - **Isotropic Spectral Projected Gradient (`SPG`) + Stiefel ABB Fallback**:
     Whenever `m_cur == 0`, `memory == 0`, $\langle \tilde a, \tilde w \rangle_{p_k} \le 10^{-30} \lVert \tilde a \rVert_2^2$,
     or $\phi'(0^+) > -10^{-12} v_0 g_0$ (e.g., releasing from a stationary face or simplex corner), or on the second-chance line-search restart, compute:

```math
d_{z,k}^{\mathrm{SPG}} = P_{\mathcal C}\!\bigl(z_k - \alpha_k^{\text{ABB,box}} g_{z,k}\bigr) - z_k, \qquad \eta_{r,k}^{\mathrm{ABB}} = -\alpha_k^{\text{ABB},(r)} g_{r,k}.
```

   By Lean 4 Theorem 6, convexity of $\mathcal C$ guarantees $z_k + \tau d_{z,k}^{\mathrm{SPG}} \in \mathcal C$ for all $\tau \in [0, 1]$
   and $g_{z,k}^\top d_{z,k}^{\mathrm{SPG}} \le -\tfrac{1}{\alpha_k^{\text{ABB,box}}} \lVert d_{z,k}^{\mathrm{SPG}} \rVert_2^2 \lt 0$
   unconditionally whenever $\mathcal E_{\mathcal C}(p_k) > 0$.

---

## 3. Fortran 2018 Module Architecture and Zero-Allocation Workspace

### 3.1 Repository Tree and Strict Disjoint File Boundaries

```text
fortchuck/
├── fpm.toml                          # Fortran Package Manager manifest (F2018 + OpenMP + CLI)
├── pyproject.toml                    # Zero-copy Python package & pytest/benchmark metadata
├── CITATION.cff                      # Academic citation metadata (CC0-1.0)
├── LICENSE                           # CC0-1.0 Universal
├── README.md                         # Badges, math formulation, quickstart, benchmark table
├── app/
│   └── fortchuck_bench.f90           # Standalone native Fortran-to-Fortran benchmark CLI (v0.2.0)
├── include/
│   └── fortchuck.h                   # ANSI C99 / C++11 interoperable header
├── src/
│   ├── fortchuck_kinds.f90           # Kinds, status enums, bind(C) structs & callback interface
│   ├── fortchuck_linalg.f90          # Pure F2018 4x4/1x4 tiled gemm, syrk, 4-RHS getrf/getrs, geqrf_pos
│   ├── fortchuck_box.f90             # Pure overflow-safe box + Dai-Fletcher linear-box projection & KKT
│   ├── fortchuck_stiefel.f90         # Pure Stiefel rgrad, dual-regime Cayley, batched panel transport, polish
│   ├── fortchuck_lbfgs.f90           # Pure diag-box + Rank-1 KKT nullspace product two-loop, ring buffer
│   ├── fortchuck_solver.f90          # Workspace slicer, query, fc_solve & fc_solve_constrained loops
│   ├── fortchuck_c_abi.f90           # Null-safe iso_c_binding entry points matching fortchuck.h
│   └── fortchuck.f90                 # Public Fortran 2018 facade module
├── test/
│   ├── test_linalg_and_stiefel.f90   # Unit/property tests: LU, QR, batched transport, Dai-Fletcher SLBQP
│   ├── test_coupled_solver.f90       # Pure-box, pure-Stiefel, simplex/DFT coupled KKT & edge cases
│   └── test_openmp_reentrancy.f90    # 16-thread OpenMP parallel multi-start bit-reproducibility test
├── python/
│   └── fortchuck/
│       ├── __init__.py               # Public Python API exports (v0.2.0)
│       ├── _native.py                # Shared library builder/loader + ctypes struct/fn bindings
│       └── solver.py                 # Zero-copy minimize_box_stiefel(..., linear_constraint=...) wrapper
├── tests/
│   ├── test_c_abi_and_python.py      # ctypes workspace query, constrained C-ABI, zero-copy & error codes
│   └── test_pathological_guards.py   # Antipodal velocity, NaN callbacks, FM-2 & non-isotropic KKT guards
├── formalization/
│   ├── lean-toolchain                # leanprover/lean4:v4.34.0-rc2 (matching warm Mathlib cache)
│   ├── lakefile.lean                 # Lake build configuration
│   └── Fortchuck.lean                # Machine-checked proofs (7 theorems, 0 sorry): Cayley, SLBQP, KKT QN
├── benchmarks/
│   ├── problems.py                   # Closed-form objectives & gradients for Benchmarks B1..B8B
│   ├── referee.py                    # Independent external KKT, linear-box feasibility, check_grad & oracle
│   ├── baselines.py                  # SciPy (L-BFGS-B, SLSQP, trust-constr), pymanopt, Alternating BCD
│   └── run_blind_suite.py            # Multi-seed blind runner (Seeds 101..110, 201..210, 301..310)
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
version = "0.2.0"
license = "CC0-1.0"
author = "Noah Cohen"
maintainer = "Noah Cohen"
copyright = "CC0-1.0 Universal Public Domain Dedication"
description = "Zero-allocation, thread-safe Modern Fortran Coupled Box/Simplex x Stiefel Quasi-Newton Optimizer"

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

[[executable]]
name = "fortchuck-bench"
source-dir = "app"
main = "fortchuck_bench.f90"
```

### 3.3 Fortran 2018 Module Specifications and `pure` Contracts

Every computational routine in `fortchuck_linalg`, `fortchuck_box`, `fortchuck_stiefel`, `fortchuck_lbfgs`, and `fc_workspace_query` **must carry the `pure` attribute**, have `implicit none (type, external)`, declare zero `save` or module-mutable variables, declare zero runtime-sized stack automatic arrays (`real(dp) :: tmp(n)` is strictly forbidden!), and perform zero I/O or `stop` statements:

1. **`fortchuck_kinds.f90`**:
   - Precision parameters `dp = c_double`, `i32 = c_int`, `i64 = c_int64_t`, `FC_EPS = epsilon(1.0_dp)`, `FC_HUGE = huge(1.0_dp)`, `FC_PI = 3.14159265358979323846264338327950288_dp`.
   - Status constants (`FC_SUCCESS_QUERY = 0` .. `FC_ERR_INVALID_OPTS = -9`, plus `FC_ERR_INFEASIBLE_LIN = -10`), metric enums (`FC_METRIC_CANONICAL = 0`, `FC_METRIC_EUCLIDEAN = 1`), damping enums (`FC_DAMP_CAUTIOUS = 0`, `FC_DAMP_POWELL = 1`), box mask enums (`FC_BOX_FREE = 0`, `FC_BOX_ACTIVE = 1`, `FC_BOX_FIXED = 2`).
   - `type, bind(C) :: fc_options_t` (96 bytes, zero padding, 100% binary-unchanged from `v0.1.0`) and `type, bind(C) :: fc_result_t` (88 bytes, zero padding, 100% binary-unchanged from `v0.1.0`).
   - `pure subroutine fc_init_default_options(opts)` and `abstract interface` for `fc_eval_fg_cb`.

2. **`fortchuck_linalg.f90`** (`100% pure`, signatures 100% preserved from `v0.1.0`):
   - `pure subroutine fc_gemm_nn(m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)`, `fc_gemm_tn`, and `fc_gemm_nt`: L1/L2 cache-tiled (`64 x 64 x 128` / `128 x 64 x 128`) with explicit `4 x 4` (`fc_gemm_tn`) and `1 x 4` (`fc_gemm_nn`, `fc_gemm_nt`) register micro-kernels saturating SIMD FMA pipelines; optional `#ifdef FORTCHUCK_USE_BLAS` dispatch when `m*n*k > 8192`.
   - `pure subroutine fc_syrk_tn(n, p, X, S)` computing $S = X^\top X \in \mathbb R^{p \times p}$, plus `fc_frob_inner` and `fc_frob_norm`.
   - `pure subroutine fc_lu_factor(n, A, ipiv, piv_tol, info)`: In-place partial-pivoted $P A = L U$ with IEEE `NaN` check and reciprocal pivot guard $\min_i \lvert U_{ii} \rvert \lt \varepsilon_{\mathrm{piv}} \max_i \lvert U_{ii} \rvert$ (`piv_tol = 1.0e-12_dp`, setting `info = i` on failure).
   - `pure subroutine fc_lu_solve(n, nrhs, LU, ipiv, B)` solving $A X = B$; `pure subroutine fc_lu_solve_trans(n, nrhs, LU, ipiv, B)` solving $A^\top X = B$ (both upgraded with hoisted `inv_ukk = 1.0_dp / LU(k,k)` reciprocals and `JB = 4` RHS register panels); and `pure subroutine fc_lu_solve_right(m, p, LU_Jt, ipiv, Panel)` solving $Z J = \mathrm{Panel} \in \mathbb R^{m \times p}$.
   - `pure subroutine fc_geqrf_pos(n, p, X, tau_qr, work_qr, info)`: In-place thin Householder QR replacing $X \in \mathbb R^{n \times p}$ with $Q \in \mathrm{St}(n, p)$ such that $R_{ii} > 0$ for all $i = 1, \dots, p$ (setting `info = i` if `abs(R(i,i)) <= 1.0e-14_dp`).

3. **`fortchuck_box.f90`** (`100% pure`, preserving all `v0.1.0` routines + adding `v0.2.0` diagonal, breakpoint, and linear-box routines):
   - Existing `v0.1.0` routines (signatures preserved): `fc_box_validate`, `fc_box_project`, `fc_box_kkt_residual`, `fc_box_partition`, `fc_box_active_step`, `fc_box_dir_deriv0`, `fc_box_trial_step`.
   - `pure subroutine fc_box_active_step_diag(d, z, l, u, gz, mask_box, diag_box, d_min, d_max, has_prev, s_prev, y_prev, dz)`: Active-band coordinatewise step using persistent `diag_box(1:d)` fallback clamped to `[d_min, d_max]`.
   - `pure subroutine fc_box_breakpoint_cap(d, z, l, u, mask_box, dz, tau_brk_min, tau_brk_max, n_free_hit)`: Overflow-safe scan of earliest and latest positive free-coordinate bound-hit breakpoints along `dz`.
   - `pure subroutine fc_box_lin_validate(d, l, u, a, b, status)`: Validates $d \ge 1$, bounds $l \le u$, finite $(a, b)$, $\lVert a \rVert_\infty > 0$, and primal feasibility $\underline b \le b \le \overline b$ (`FC_ERR_INFEASIBLE_LIN = -10` on failure).
   - `pure subroutine fc_box_lin_project(d, l, u, a, b, z, nu_out, status)`: Scalar-memory overflow-safe Dai--Fletcher / Cominetti--Mascarenhas semismooth Newton-secant + safeguarded bisection projection $z \leftarrow P_{\mathcal C}(z)$ in $\le 60$ passes
     (`0` array workspace, accuracy $\lvert a^\top z - b \rvert \le 10^{-14}(1 + \lvert b \rvert + \lVert a \rVert_1 \lVert z \rVert_\infty)$).
   - `pure subroutine fc_box_lin_kkt_residual(d, z, l, u, a, b, gz, z_proj_tmp, rbox, nu_out, box_inf, box_l2, status)`: Evaluates $r_{\mathcal C}(z) = z - P_{\mathcal C}(z - g_z)$
     and multiplier estimate $\nu_{\mathrm{out}}$.
   - `pure subroutine fc_box_lin_partition(d, z, l, u, a, gz, nu_k, mask_box, dz, delta_b, n_active)`: Boundary active-face partition relative to $\hat g_z = g_z + \nu_k a$,
     snapping tiny `<= 1e-12` boundary gaps and returning $\delta_b = -\sum_{i \in \mathcal A_k} a_i d_{z,i}$.
   - `pure subroutine fc_box_lin_trial_step(d, z, l, u, a, b, gz, dz, tau, z_trial, chord_gz, step_norm_sq, status)`: Forms $z_{\mathrm{trial}} = P_{\mathcal C}(z + \tau d_z)$
     and evaluates realized chord `gz^T (z_trial - z)` and $\lVert z_{\mathrm{trial}} - z \rVert_2^2$.

4. **`fortchuck_stiefel.f90`** (`100% pure`, preserving all `v0.1.0` routines + adding batched panel transport):
   - Existing `v0.1.0` routines (signatures preserved): `fc_stiefel_offsets`, `fc_stiefel_rgrad`, `fc_stiefel_proj_tangent`, `fc_stiefel_inner`, `fc_stiefel_orth_err`, `fc_cayley_setup`, `fc_cayley_retract`, `fc_cayley_transport_vec`.
   - `pure subroutine fc_cayley_transport_batch(n, p, k_batch, X_old, eta, tau, cay_buf, ipiv, Panel, scratch)`: Auto-dispatching Level-3 batched isometric Cayley transport on `Panel(n, p * k_batch)` (`2*p < n` Schur-Woodbury panel + scalar $p = 1$ fast-path vs `2*p >= n` Direct LU multi-RHS panel), matching `k_batch` calls to `fc_cayley_transport_vec` to $\le 10^{-14}$.

5. **`fortchuck_lbfgs.f90`** (`100% pure`):
   - `pure subroutine fc_lbfgs_apply_H(...)`: Reusable `GEMM`-free backward/center/forward product sweep applying $+\widetilde H_k$
     with `diag_box(1:d)` on $\mathcal F_k$ and `gamma_blk(2:R+1)` on Stiefel blocks.
   - `pure subroutine fc_lbfgs_direction(...)`: Unconstrained-box masked product L-BFGS step (`fc_lbfgs_apply_H` + `fc_box_active_step_diag`).
   - `pure subroutine fc_lbfgs_direction_kkt(...)`: Rank-1 KKT Nullspace Product Two-Loop (`u = -H_k g_tilde`, `w = H_k a_tilde`, `d_free = u - ((<a_tilde, u> - delta_b)/<a_tilde, w>) * w`, plus 1-pass outward boundary freeze guard; sets `fallback_spg = .true.` if degenerate).
   - `pure subroutine fc_abb_direction(...)` & `pure subroutine fc_spg_lin_direction(...)`: Per-block Alternating Barzilai--Borwein step (and isotropic `SPG` step $P_{\mathcal C}(z_k - \alpha_{\text{box}} g_{z,k}) - z_k$ when linearly constrained).
   - `pure subroutine fc_lbfgs_transport_and_push(...)`: Packs all $K_{\mathrm{batch}} = 2 m_{\mathrm{cur}} + 2$ tangent blocks per factor into `scratch`, invokes `fc_cayley_transport_batch` **once** per factor $r$, updates `gamma_blk(1:R+1)` and safeguarded `diag_box(1:d)`,
     applies Cautious or Powell damping (using $1/D_{k,i}$ on box coordinates), and pushes `(s_k, y_k)`.

6. **`fortchuck_solver.f90`**, **`fortchuck_c_abi.f90`**, and **`fortchuck.f90`**:
   - `pure subroutine fc_workspace_query(d, R, n_arr, p_arr, m, lwork, liwork, status)`: Unified exact workspace query sufficient for both `fc_solve` and `fc_solve_constrained`.
   - `subroutine fc_solve(...)` (100% signature-compatible with `v0.1.0`) and `subroutine fc_solve_constrained(..., a, b, ...)` sharing a single internal stateless core loop with `has_lin_eq` dispatch and safeguarded Quadratic/Cubic Hermite + Cauchy breakpoint line search.

### 3.4 Exact Unified Caller-Owned `work` and `iwork` Layout (`v0.2.0`)

Define the padded block counts, maximum batch width $K_{\max} = 2m + 2$, and per-factor scratch sizes:

```math
d_{\mathrm{pad}} = \max(d, 1), \quad N_{\mathrm{St}} = \sum_{r=1}^R n_r p_r, \quad N_{\mathrm{St,pad}} = \max(N_{\mathrm{St}}, 1), \quad N_{\mathrm{tot,pad}} = \max(d + N_{\mathrm{St}}, 1), \quad N_{\mathrm{skew,pad}} = \max\!\left(1, \sum_{r=1}^R p_r^2\right), \quad m_{\mathrm{pad}} = \max(m, 1)
```

```math
c_r = \begin{cases} 2 n_r p_r + 3 p_r^2 & \text{if } 2 p_r \lt n_r, \\ 2 n_r^2 & \text{if } 2 p_r \ge n_r, \end{cases} \qquad q_r = \begin{cases} p_r & \text{if } 2 p_r \lt n_r, \\ n_r & \text{if } 2 p_r \ge n_r, \end{cases}
```

```math
s_r = \begin{cases} \max\bigl(2 n_r p_r + 6 p_r^2 + 4 p_r, \; (n_r + 2 p_r) p_r (2m + 2) + p_r^2\bigr) & \text{if } 2 p_r \lt n_r, \\ \max\bigl(2 n_r p_r + 6 p_r^2 + 4 p_r, \; 2 n_r p_r (2m + 2)\bigr) & \text{if } 2 p_r \ge n_r, \end{cases}
```

```math
N_{\mathrm{cay}} = \max\!\left(1, \sum_{r=1}^R c_r\right), \qquad N_{\mathrm{piv}} = \max\!\left(1, \sum_{r=1}^R q_r\right), \qquad N_{\mathrm{scr}} = \max\!\left(64, \; \max_{1 \le r \le R} s_r\right).
```

| Buffer Segment | Target Array | Exact Length (`real64` or `int32`) | Role |
| :--- | :--- | :--- | :--- |
| `work` `[1]` | `z_trial`, `gz`, `gz_trial`, `dz`, `rbox`, `diag_box`, `z_spg` | $7 \, d_{\mathrm{pad}}$ | Box trial iterate, grads, direction, KKT, diagonal secant, SPG temp |
| `work` `[2]` | `X_trial`, `GX`, `GX_trial`, `rgrad_X`, `rgrad_old`, `eta_X` | $6 \, N_{\mathrm{St,pad}}$ | Stiefel trial frames, Euclidean/Riemannian grads, tangent step |
| `work` `[3]` | `q_buf`, `w_buf`, `s_cur`, `y_cur` | $4 \, N_{\mathrm{tot,pad}}$ | Primary & KKT-nullspace two-loop accumulators + secant buffers |
| `work` `[4]` | `Aq_buf` | $N_{\mathrm{skew,pad}}$ | Working `p_r x p_r` skew accumulator (reused across sweeps) |
| `work` `[5]` | `gamma_blk`, `abb_ss`, `abb_sy`, `abb_yy` | $4 \, (R + 1)$ | Persistent per-block BB2 scales & latest block secant stats |
| `work` `[6]` | `s_hist`, `y_hist` | $2 \, m \, N_{\mathrm{tot,pad}}$ | Circular L-BFGS ring buffer (0 words when `m = 0`) |
| `work` `[7]` | `As_hist`, `Ay_hist` | $2 \, m \, N_{\mathrm{skew,pad}}$ | Cached invariant `X_r^T S_{j,r}` and `X_r^T Y_{j,r}` (0 when `m = 0`) |
| `work` `[8]` | `st_sy`, `st_ss`, `st_yy`, `rho_m`, `alpha_m` | $5 \, m_{\mathrm{pad}}$ | Cached isometric Stiefel norms & two-loop scalars |
| `work` `[9]` | `cayley_persist` | $N_{\mathrm{cay}}$ | Per-factor `A_r, S_r, K_r, eta_tilde_r, J_lu_r` or `W_eta_r, M_lu_r` |
| `work` `[10]` | `scratch` | $N_{\mathrm{scr}}$ | Shared retraction, batched `(n_r, p_r*K_max)` panel transport, & QR scratch |
| `iwork` `[1]` | `mask_box` | $d_{\mathrm{pad}}$ | Box partition tags (`FREE=0`, `ACTIVE=1`, `FIXED=2`) |
| `iwork` `[2]` | `active_pair_flag`, `ring_order` | $2 \, m_{\mathrm{pad}}$ | Dynamic subspace filter flags and chronological slot map |
| `iwork` `[3]` | `x_off`, `skew_off`, `cay_off`, `ipiv_off` | $4 \, (R + 1)$ | Packed factor offset tables |
| `iwork` `[4]` | `ipiv_all` | $N_{\mathrm{piv}}$ | Per-factor `GETRF` pivot indices |

**Workspace Query Protocol (`lwork = -1` / `liwork = -1`)**:
- Calling `fortchuck_workspace_query(d, R, n_arr, p_arr, m, &lwork, &liwork)` returns `FC_SUCCESS_QUERY (0)` with exact unified minimum sizes in `int64_t` valid for both `fortchuck_solve` and `fortchuck_solve_constrained`.
- Calling `fortchuck_solve` or `fortchuck_solve_constrained` with `lwork == -1` or `liwork == -1` writes `real(lwork_min, c_double)` to `work[0]` (if `work != NULL`), `int(min(liwork_min, huge(0_c_int)), c_int)` to `iwork[0]` (if `iwork != NULL`), populates `res->lwork_min` and `res->liwork_min`, and returns `FC_SUCCESS_QUERY (0)`.

### 3.5 Zero-Allocation and OpenMP Reentrancy Contract

1. **Zero Heap Allocations**: Every `*.o` compiled from `src/*.f90` must pass `scripts/verify_zero_alloc.py` (`nm -u`), referencing **zero** allocator (`malloc`, `free`, `realloc`, `calloc`, `_gfortran_allocate*`, `_gfortran_deallocate*`) or hidden temporary packing (`_gfortran_internal_pack*`, `_gfortran_internal_unpack*`) symbols, and compiling cleanly with `-Warray-temporaries -Werror`.
2. **100% OpenMP Reentrancy**: Zero mutable module globals, zero `save` state, and zero internal static buffers exist anywhere in `fortchuck`. Any number of OpenMP parallel-do threads (`-fopenmp`) or POSIX threads may invoke `fortchuck_solve` or `fortchuck_solve_constrained` concurrently on separate caller-owned `(z, X, work, iwork, udata)` buffers with bit-identical deterministic output independent of thread schedule.

---

## 4. ANSI C99/C11 ABI Header and Zero-Copy Python Package

### 4.1 Complete `include/fortchuck.h` Specification (`100%` Backward-Compatible `v0.2.0` ABI)

```c
#ifndef FORTCHUCK_H
#define FORTCHUCK_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Status and error codes */
enum fc_status_e {
    FC_SUCCESS_QUERY       =   0,
    FC_CONVERGED_GRAD      =   1,
    FC_CONVERGED_FTOL      =   2,
    FC_CONVERGED_XTOL      =   3,
    FC_MAX_ITER_REACHED    =   4,
    FC_MAX_EVAL_REACHED    =   5,
    FC_ERR_INVALID_DIM     =  -1,
    FC_ERR_NULL_PTR        =  -2,
    FC_ERR_INVALID_BOUNDS  =  -3,
    FC_ERR_INVALID_X0      =  -4,
    FC_ERR_WORK_TOO_SMALL  =  -5,
    FC_ERR_CALLBACK_NAN    =  -6,
    FC_ERR_CALLBACK_ABORT  =  -7,
    FC_ERR_LS_FAILED       =  -8,
    FC_ERR_INVALID_OPTS    =  -9,
    FC_ERR_INFEASIBLE_LIN  = -10
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
    int32_t memory;           /* Default: 7 (0 = pure ABB/SPG steepest descent) */
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

/* Coupled singly-linearly-constrained Box / Simplex x Multi-Stiefel solver (v0.2.0):
 * Minimizes f(z, X_1..X_R) subject to l_i <= z_i <= u_i, a^T z = b, and X_r^T X_r = I_{p_r}. */
int32_t fortchuck_solve_constrained(
    int32_t d,
    int32_t R,
    const int32_t *n_arr,
    const int32_t *p_arr,
    double *z,
    const double *l,
    const double *u,
    const double *a,
    double b,
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

/* Zero-workspace Euclidean projection onto {z in [l, u]^d : a^T z = b} (v0.2.0):
 * Projects z[0..d-1] in-place; writes dual multiplier to *nu_out if nu_out != NULL. */
int32_t fortchuck_project_linear_box(
    int32_t d,
    const double *l,
    const double *u,
    const double *a,
    double b,
    double *z,
    double *nu_out
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

/* Constrained KKT & linear/orthogonality feasibility certificate (v0.2.0):
 * Supports lwork == -1 workspace query (minimum lwork = max(1, d + max_r(n_r*p_r + 3*p_r^2))). */
int32_t fortchuck_kkt_residual_constrained(
    int32_t d,
    int32_t R,
    const int32_t *n_arr,
    const int32_t *p_arr,
    const double *z,
    const double *l,
    const double *u,
    const double *a,
    double b,
    const double *X_packed,
    const double *gz,
    const double *GX_packed,
    int32_t metric,
    double *work,
    int64_t lwork,
    double *kkt_inf_out,
    double *lin_err_out,
    double *orth_err_out
);

#ifdef __cplusplus
}
#endif

#endif /* FORTCHUCK_H */
```

### 4.2 Zero-Copy Python `ctypes` Package (`python/fortchuck/`)

1. **`python/fortchuck/_native.py`**:
   - Registers MinGW/OpenMP DLL directories on Windows (`os.add_dll_directory` over `PATH`), locates prebuilt `fortchuck.dll` / `libfortchuck.so` (or auto-invokes `scripts/build_shared.py` if missing), defines `FCOptions` (`96` bytes) and `FCResult` (`88` bytes) verified by import-time `assert`s, exports `FC_ERR_INFEASIBLE_LIN = -10`, and binds all 8 C99 entry points (`fortchuck_default_options`, `fortchuck_workspace_query`, `fortchuck_solve`, `fortchuck_solve_constrained`, `fortchuck_project_stiefel`, `fortchuck_project_linear_box`, `fortchuck_kkt_residual`, and `fortchuck_kkt_residual_constrained`).
2. **`python/fortchuck/solver.py` & `python/fortchuck/__init__.py`**:
   - Exposes `minimize_box_stiefel(fun_and_grad, z0=None, bounds=None, X0_list=None, options=None, work=None, iwork=None, linear_constraint=None)` plus `project_linear_box(z, lower, upper, a, b) -> tuple[np.ndarray, float]`, `kkt_residual(..., linear_constraint=None)`, and reusable `FortchuckWorkspace(d, n_list, p_list, memory=7)` (`__version__ = "0.2.0"`).
   - When `linear_constraint=(a, b)` is supplied (`a` a 1D `float64` array-like of length `d` and `b` a finite `float`), `minimize_box_stiefel` dispatches directly to `fortchuck_solve_constrained` while keeping `linear_constraint=None` on the exact `fortchuck_solve` codepath.
   - **Zero-Copy Column-Major (`order="F"`) Fast Path**:
     - When `R == 1` and `X0` is already a contiguous `float64` Fortran-ordered array (`X0.flags.f_contiguous`), `X_packed` wraps a single buffer with zero transpose copies; inside the `@CFUNCTYPE` callback, `np.ctypeslib.as_array` reshaped with `order="F"` creates zero-copy NumPy views into `z`, `X_r`, `gz_out`, and `GX_r_out` so user Python code can write gradients directly in-place (`out_gz[:] = ...`) or return `(f, gz, [GX_1, ..., GX_R])`.
     - Python exceptions raised inside `fun_and_grad` are caught inside the `@CFUNCTYPE` wrapper, stashed on the call context, and return `-1` (`FC_ERR_CALLBACK_ABORT`) cleanly across the Fortran stack frame before re-raising in Python (never unwinding exceptions through Fortran!).

---

## 5. Verification, Testing, Lean 4 Formalization, and CI Gates

### 5.1 Native Fortran 2018 Test Suite (`test/*.f90`) and Benchmark CLI (`app/fortchuck_bench.f90`)

Executed under strict sanitizers (`fpm test --compiler gfortran --flag "-std=f2018 -O2 -g -Wall -Wextra -Werror -pedantic -Warray-temporaries -fimplicit-none -fcheck=all -fbacktrace -ffpe-trap=invalid,zero,overflow -fopenmp"`):

1. **`test/test_linalg_and_stiefel.f90`**:
   - Verifies `4x4`/`1x4` register-tiled `fc_gemm_*`, division-hoisted `4`-RHS `fc_lu_solve` and `fc_lu_solve_trans`, `fc_lu_factor`, and `fc_lu_solve_right` across odd/prime dimensions (`m, n, k, nrhs in {1, 2, 3, 4, 5, 7, 16, 65, 131}`) against reference implementations to $\le 10^{-14}$.
   - Verifies `fc_geqrf_pos` produces $\lVert Q^\top Q - I_p \rVert_{\mathrm F} \le 10^{-15}$ with strictly positive diagonal entries, $\det(Q) = +1$ for perturbed $\mathrm{SO}(n)$ matrices,
   and $\lVert Q - X \rVert_{\mathrm F} \le \delta$ for small drift $\delta = 10^{-12}$.
   - **Batched Panel vs Single-Vector Equivalence & Isometry Property Test**: Across $(n, p) \in \lbrace (10, 1), (12, 3), (8, 4), (7, 5), (6, 6), (40, 6), (16, 16) \rbrace$ and $K_{\mathrm{batch}} \in \lbrace 1, 5, 16 \rbrace$, asserts `fc_cayley_transport_batch` matches `K_batch` calls to `fc_cayley_transport_vec` to $\le 10^{-14}$ across both Branch A and Branch B, while preserving canonical inner products:

```math
\left\lvert \langle \mathcal T \zeta_1, \mathcal T \zeta_2 \rangle_Y - \langle \zeta_1, \zeta_2 \rangle_X \right\rvert \le 10^{-14}.
```

   - **Dai--Fletcher / Cominetti--Mascarenhas Linear-Box Projection Suite**: Verifies `fc_box_lin_project` on unit simplices, general weighted knapsacks ($a_i \gtrless 0$), unbounded one-sided coordinates (`±FC_HUGE` under `-ffpe-trap=overflow`), singleton boundary faces ($b = \underline b$ and $b = \overline b$), and infeasible hyperplanes (`FC_ERR_INFEASIBLE_LIN = -10`), checking variational optimality $\langle y - z^\ast, z - z^\ast \rangle \le 10^{-14}$ across random feasible $z \in \mathcal C$.
2. **`test/test_coupled_solver.f90`**:
   - **Regime Coverage**: Pure Box ($R = 0$, including `±huge(1.0_dp)` under `-ffpe-trap=overflow` and fixed `l_i == u_i`), Pure Stiefel ($d = 0$, both $2p \lt n$ on $\mathrm{St}(12, 3)$ and $2p \ge n$ on $\mathrm{SO}(5)$), Coupled Multi-Factor ($d = 6, R = 2$ on $\mathrm{St}(10, 2) \times \mathrm{St}(4, 4)$), Pure Simplex/Knapsack ($R = 0, \mathcal D = \mathcal C$), and Coupled Simplex/Trace-Constrained DFT x Multi-Stiefel (`fc_solve_constrained` with both $a = \mathbf{1}$ and $a \neq \mathbf{1}$, verifying $\lvert a^\top z^\ast - b \rvert \le 10^{-14}$ and $\mathcal E_{\text{KKT}} \le 10^{-8}$).
   - **Ground-Truth Optimality & Modes**: Box-coupled Brockett problem verified against exact Ky Fan QP global minimum $f^\ast$ to $\le 10^{-11}$, plus both damping modes (`CAUTIOUS`, `POWELL`), metrics (`CANONICAL`, `EUCLIDEAN`), and `memory = 0` (`ABB`/`SPG`) vs `memory = 7` (`LRBFGS`).
3. **`test/test_openmp_reentrancy.f90`**:
   - Spawns `64` independent optimization instances (covering both unconstrained-box `fc_solve` and linearly-constrained `fc_solve_constrained` on `[l, u]^8 x St(12, 3) x St(4, 4)`) across `16` OpenMP parallel-do threads (`schedule(dynamic, 1)`), asserting **bit-for-bit identical** `f_val`, `n_iter`, `n_eval`, and final `(z, X)` against sequential single-thread execution (`0.0_dp` max difference!).
4. **`app/fortchuck_bench.f90` (`fpm run fortchuck-bench --profile release`)**:
   - Standalone native Fortran-to-Fortran benchmark executable (`0` C/Python callback overhead) reporting `system_clock(count_rate)` sub-millisecond telemetry for:
     (a) `fc_cayley_transport_batch` vs `16x` `fc_cayley_transport_vec` throughput speedup on `St(200, 8)` (Branch A) and `St(24, 24)` (Branch B);
     (b) 10-seed native `B6` (`d = 10, St(150, 10)`), `B7B` (`St(200, 8) x St(24, 24)` at `m = 7` vs `m = 0`), and `B8A` (`{z in [0,1]^12 : sum(z) = 5} x St(120, 12)` via `fc_solve_constrained`).

### 5.2 Python FFI & Adversarial Edge-Case Tests (`tests/*.py`)

1. **`tests/test_c_abi_and_python.py`**:
   - Verifies `96B`/`88B` struct binary layout, `lwork = -1` query parity with unified §3.4 closed-form sizes across both `fortchuck_solve` and `fortchuck_solve_constrained`, `fortchuck_project_linear_box` & `fortchuck_kkt_residual_constrained` accuracy, `FC_ERR_INFEASIBLE_LIN = -10` error handling, `minimize_box_stiefel(..., linear_constraint=(a, b))`, hard `max_eval` enforcement, and clean Python exception propagation.
2. **`tests/test_pathological_guards.py`**:
   - **FM-2 Clamped Multiplier Injection Test**: Strongly cross-coupled boundary problem on $[0, 1] \times \mathrm{St}(8, 3)$:

```math
f(z, X) = 10^4 z_1 (1 + x_{11}) + \frac{1}{2}\mathrm{tr}(X^\top D X)
```

   where `z_1* = 0` clamps on step 1 with huge active multiplier $\lambda_1^\ast = 10^4(1 + x_{11}^\ast) \gg 1$,
   verifying rapid superlinear convergence (at most $35$ iterations to $\mathcal E_{\text{KKT}} \le 10^{-10}$).
   - **FM-13 Non-Isotropic Hyperplane Ascent & Simplex Corner Release Test**: Constructs an ill-conditioned diagonal/coupled quadratic + Brockett problem on $\Delta_8 \times \mathrm{St}(10, 3)$
   initialized at a suboptimal simplex vertex $e_1 = (1, 0, \dots, 0)$ with condition number $10^4$,
   verifying immediate strict-descent corner release and superlinear convergence to $\mathcal E_{\text{KKT}} \le 10^{-9}$ with $\lvert \mathbf{1}^\top z^\ast - 1 \rvert \le 10^{-14}$.
   - **Callback Domain Barrier Recovery**: Confirms automatic step halving and clean convergence when `eval_fg` returns `rc = +1` (or `NaN`) on trial overshoot.

### 5.3 Lean 4 Formal Verification (`formalization/Fortchuck.lean`)

Using the cached `leanprover/lean4:v4.34.0-rc2` + Mathlib4 (`de2ef68216c6074f338c8e61890ee0a379ddfb9b`) toolchain via interactive `lean-lsp` MCP verification (`0 sorry`, `0` warnings), `formalization/Fortchuck.lean` proves **seven** machine-checked theorems:

1. **Theorem 1 (`skew_generator_properties`)**: For orthonormal `X` ($X^\top X = 1$) and skew tangent `xi` (`X^T xi + xi^T X = 0`), setting `xi_tilde = xi - (1/2) X (X^T xi)` and `W_xi = xi_tilde X^T - X xi_tilde^T` satisfies $W_\xi^\top = -W_\xi$ and `W_xi X = xi`.
2. **Theorem 2 (`cayley_rotor_orthogonality_and_isometry`)**: Given $W^\top = -W$ and inverse witness `M * (1 - (tau/2) W) = (1 - (tau/2) W) * M = 1`, the Cayley rotor `Q = M * (1 + (tau/2) W)` satisfies `Q = 2 M - 1`, $Q^\top Q = 1$, `(Q X)^T (Q X) = X^T X`, and preserves tangent skew-symmetry and canonical inner products.
3. **Theorem 3 (`schur_woodbury_cayley_orthogonality`)**: Given $X^\top X = 1$, `A^T = -A`, `X^T K = 0`, `J = 1 - (tau/2) A + (tau^2/4) K^T K`, and `J W = W J = 1`, the rank $2p$ Schur-Woodbury retraction `Y = (2 X + tau K) W - X` satisfies $Y^\top Y = 1$ identically.
4. **Theorem 4 (`cayley_branch_a_eq_branch_b`)**: Proves exact algebraic equivalence between the Branch A Schur-Woodbury retraction/transport and the Branch B Direct LU resolvent Cayley operator.
5. **Theorem 5 (`coupled_two_metric_strict_descent`)**: Formalizes Theorem 2.1's coordinatewise free-set clamping inequality `g_i * dot_z_i <= g_i * d_i` under the Bertsekas active-band sign hypothesis, proving:

```math
\phi'(0^+) \le -\gamma_{\min} \sum_{i \in \mathcal A^{\text{int}}} g_i^2 - \lambda_{\min} \lVert \tilde g \rVert^2 \le 0.
```

6. **Theorem 6 (`simplex_box_projection_variational`, `v0.2.0` Pillar 4)**: Over `Fin d -> ℝ`, if `∀ i, l i ≤ u i`,
   $z^\ast_i = \min\bigl(u_i, \max(l_i, y_i - \nu a_i)\bigr)$,
   and $\sum_{i=1}^d a_i z^\ast_i = b$,
   then for every feasible $z \in \mathcal C$ satisfying $l_i \le z_i \le u_i$
   and $\sum_i a_i z_i = b$:
   (a) $l_i \le z^\ast_i \le u_i$;
   (b) $\sum_{i=1}^d (y_i - z^\ast_i)(z_i - z^\ast_i) \le 0$;
   (c) $\sum_{i=1}^d (z^\ast_i - y_i)^2 + \sum_{i=1}^d (z_i - z^\ast_i)^2 \le \sum_{i=1}^d (z_i - y_i)^2$ (unique Euclidean projection); and
   (d) when $y = z_k - \gamma g$ with $\gamma > 0$
   and feasible $z_k \in \mathcal C$,
   the isotropic `SPG` direction $d_{\mathrm{SPG}} = z^\ast - z_k$
   satisfies $\sum_i a_i d_{\mathrm{SPG},i} = 0$
   and $\sum_i g_i d_{\mathrm{SPG},i} \le -\tfrac{1}{\gamma} \sum_i d_{\mathrm{SPG},i}^2 \le 0$
   (strictly $\lt 0$ whenever $z^\ast \neq z_k$).
7. **Theorem 7 (`rank1_kkt_nullspace_quasinewton_descent`, `v0.2.0` Pillar 4)**: Over `Matrix (Fin n) (Fin n) ℝ`, for any symmetric positive-definite matrix $H$ ($H^\top = H$ and $\forall v \neq 0, \; v^\top H v > 0$) and nonzero $a \neq 0$, defining $u = -H g$, $w = H a$, and $d = u - \frac{a^\top u}{a^\top w} w = -H_{\mathrm{proj}} g$ (via witness $v = g - \frac{a^\top H g}{a^\top H a} a$ satisfying $v^\top H v = -g^\top d$):
   (a) $a^\top w > 0$;
   (b) $a^\top d = 0$ (exact hyperplane tangency);
   (c) multiplier shift invariance $-H_{\mathrm{proj}}(g + \mu a) = d$ for all $\mu \in \mathbb R$; and
   (d) $g^\top d \le 0$, with strict descent $g^\top d \lt 0$ whenever $\forall \mu \in \mathbb R, \; g \neq \mu a$.

### 5.4 Automated Repository & CI Gates

1. **`scripts/audit_gfm_math.py`**: Must pass with `0` violations across `README.md`, `docs/ARCHITECTURE_SPEC.md`, `results/blind_benchmark_report.md`, and `docs/research/*.md`.
2. **`scripts/verify_zero_alloc.py`**: Compiles `src/*.f90` to object files and inspects `nm -u *.o` (and `libfortchuck.a`), failing if any heap allocation, deallocation, array-pack, or Fortran I/O/stop runtime symbol appears.
3. **`.github/workflows/ci.yml`**: Four jobs (`gfm-math-audit`, `fortran-fpm-and-zero-alloc`, `python-ffi-and-pytest`, `blind-benchmark-smoke`) mirroring `triad-core` CI standards.

---

## 6. Strictly Blind Multi-Seed Benchmark Architecture

### 6.1 Blind Evaluation Firewall, Three-Suite Seed Protocol (`101..110`, `201..210`, `301..310`), and Referee (`benchmarks/referee.py`)

To guarantee zero data leakage and publication-grade empirical rigor across both `v0.1.0` and `v0.2.0`:
1. **Single Frozen Default Configuration**: `fortchuck` is called across **all** 11 benchmark instances (`B1..B8B`) and **all** 30 random seeds (`330` runs total: Primary `101..110`, Second Holdout `201..210`, and untouched **Third Blind Holdout `301..310`**) using `FCOptions()` out-of-the-box (`memory=7`, `metric=FC_METRIC_CANONICAL`, `damping_mode=FC_DAMP_CAUTIOUS`, `eps_band=1e-4`), setting **only** `gtol = 1.0e-6` and `max_eval = 5000` (passing `linear_constraint=(problem.lin_a, problem.lin_b)` automatically when `problem.lin_a is not None`). Zero problem-specific parameters, warm-starts, or custom scalings are permitted.
2. **Bit-Identical Seed Preservation + Constrained Referee**: For `B1..B7B` (`lin_a is None`), `generate_initial_point(problem, seed)` preserves the exact `v0.1.0` `PCG64(seed)` RNG draw sequence bit-for-bit. When `lin_a is not None` (`B8A`, `B8B`), `z0` is projected onto the central $80\%$ sub-box $\lbrack l + 0.1(u-l), \, u - 0.1(u-l) \rbrack \cap \lbrace a^\top z = b \rbrace$ via an **independent pure-NumPy** bisection/secant routine `project_linear_box_ref` (never calling `fortchuck`!) so both projection and interior/warping baselines start from the exact same strictly interior feasible point. On every returned `(z, X)`, the independent referee evaluates ($\mathcal D = \mathcal B$ when `lin_a is None`, $\mathcal D = \mathcal C$ when `lin_a is not None`):

```math
\mathcal E_{\text{KKT,ref}}(z, X) = \max\!\left(\left\lVert z - P_{\mathcal D,\mathrm{ref}}\bigl(z - \nabla_z f(z, X)\bigr) \right\rVert_\infty, \; \max_{1 \le r \le R} \left\lVert G_r - X_r\,\mathrm{sym}(X_r^\top G_r) \right\rVert_{\mathrm F}\right),
```

```math
\mathcal V_{\text{box}} = \max_{1 \le i \le d} \max(0, \, l_i - z_i, \, z_i - u_i), \qquad \mathcal V_{\text{lin}} = \frac{\lvert a^\top z - b \rvert}{1 + \lvert b \rvert}, \qquad \mathcal V_{\text{orth}} = \max_{1 \le r \le R} \left\lVert X_r^\top X_r - I_{p_r} \right\rVert_{\mathrm F},
```

   (taking $\mathcal V_{\text{lin}} = 0$ when `lin_a is None`),
   marking a run successful iff $\mathcal E_{\text{KKT,ref}} \le 10^{-6}$
   and $\max(\mathcal V_{\text{box}}, \mathcal V_{\text{lin}}, \mathcal V_{\text{orth}}) \le 10^{-8}$.

### 6.2 Canonical 11-Instance Blind Benchmark Matrix (`B1..B8B` in `benchmarks/problems.py`)

Exact closed-form objectives and Euclidean gradients for `B1..B7B` follow `docs/research/ARXIV_COUPLED_BOX_MANIFOLD_SURVEY.md` §6; `v0.2.0` adds Benchmark `B8` (`B8A` and `B8B`, adding optional `lin_a: np.ndarray | None = None` and `lin_b: float | None = None` to `BenchmarkProblem`):

| ID | Benchmark Problem | Domain Dimensions | Cayley & Coupling Regime Tested |
| :--- | :--- | :--- | :--- |
| `B1` | Ensemble Kohn--Sham DFT Fractional Occupations (Soft Penalty) | $z \in [0, 1]^{10}, \; X \in \mathrm{St}(120, 10)$ | Branch A ($2p \lt n$), small-gap metallic orbital-occupation sloshing |
| `B2A` | Amplitude-Limited Blind Source Separation (ICA) | $Z \in [-2, 2]^{5 \times 150}, \; W \in \mathrm{St}(20, 5)$ | Branch A ($d = 750$), nonconvex $\ln\cosh$ with $d \gg np$ active bounds |
| `B2B` | Smooth Penalty-Split Non-Negative Sparse PCA | $Z \in [0, 5]^{120 \times 6}, \; X \in \mathrm{St}(120, 6)$ | Branch A ($d = 720$), $80\%$ clamped `Z_ij = 0` with large multipliers (FM-2) |
| `B3` | Multi-Camera Anisotropic Bounded-Scale Rig Wahba | $z \in \mathcal B^{15}, \; X_r \in \mathrm{St}(3, 3)^4$ | Branch B ($2p \ge n$, $\mathrm{SO}(3)^4$), $R = 4$ multi-factor $\kappa = 10^3$ |
| `B4` | Bounded-Variance ML Common Principal Components | $z \in [0.05, 20]^{150}, \; X \in \mathrm{St}(30, 30)$ | Branch B $(\mathrm{O}(30))$, log-variance domain barrier, $1.6 \times 10^5$ scale ratio |
| `B5` | Bounded Factor Correlation Matrix Calibration | $z \in \mathcal B^{110}, \; X \in \mathrm{St}(100, 10)$ | Branch A, quartic curved valley + clamped idiosyncratic variances |
| `B6` | Box-Coupled Brockett with Exact Ky Fan QP Truth | $z \in [0.1, 5]^{10}, \; X \in \mathrm{St}(150, 10)$ | Branch A, $\kappa(M) = 10^4$, certified global minimum $f - f^\ast$ check |
| `B7A` | Pure-Box Limit: Active-Bound Chained Rosenbrock + Quadratic | $z \in [l, u]^{500}, \; R = 0$ | Pure Box ($R = 0$, incl. fixed `l_i == u_i`) vs `SciPy L-BFGS-B` |
| `B7B` | Pure-Stiefel Limit: Tall Brockett & Orthogonal Procrustes | $d = 0, \; \mathrm{St}(200, 8) \text{ and } \mathrm{St}(24, 24)$ | Pure Stiefel ($d = 0$) Branch A & B head-to-head vs `pymanopt` |
| `B8A` | Trace-Constrained Ensemble Kohn--Sham DFT (`a = 1`) | $z \in [0, 1]^{12}, \; \mathbf{1}^\top z = 5, \; X \in \mathrm{St}(120, 12)$ | Hard electron trace conservation (`SLBQP` + Branch A, Marzari--Vanderbilt--Payne) |
| `B8B` | Long-Only Weighted Orthogonal Risk-Factor Parity (`a != 1`) | $z \in [0, 0.15]^{24}, \; a^\top z = 1, \; \mathrm{St}(32, 6) \times \mathrm{St}(6, 6)$ | General $a \neq \mathbf{1}$ capped simplex + multi-factor Branch A & B (`N_tot = 252`) |

#### Exact Closed-Form Definitions for `B8A` and `B8B` (`PCG64` Instance Seeds `20260801`, `20260802`)

1. **`B8A` — Trace-Constrained Ensemble Kohn--Sham DFT** (`d = 12`, $l = 0, u = 1$,
   $a = \mathbf{1}, b = N_e = 5.0$,
   and $X \in \mathrm{St}(120, 12)$):
   Construct metallic 1D periodic Kohn--Sham Hamiltonian $H_0 \in \mathbb R^{120 \times 120}$
   and SPD Coulomb Hartree kernel $K_{\mathrm{Coul}} \in \mathbb R^{120 \times 120}$ identically to `B1` (`seed = 20260801`,
   $\alpha_{\mathrm{H}} = 0.35, k_{\mathrm{B}} T = 0.02, \delta_{\mathrm{FD}} = 10^{-12}$),
   omitting `B1`'s soft penalty $\tfrac{\mu_N}{2}(\mathbf{1}^\top z - N_e)^2$ because electron conservation $\mathbf{1}^\top z = 5$ holds identically on $\mathcal C$:

```math
f(z, X) = \sum_{i=1}^{12} z_i \, x_i^\top H_0 x_i + \frac{\alpha_{\mathrm{H}}}{2} \rho(z, X)^\top K_{\mathrm{Coul}} \rho(z, X) + k_{\mathrm{B}} T \sum_{i=1}^{12} h_{\mathrm{FD}}(z_i), \qquad \rho_s(z, X) = \sum_{i=1}^{12} z_i X_{s,i}^2.
```

2. **`B8B` — Long-Only Weighted Orthogonal Risk-Factor Parity** (`d = 24`, $l_i = 0, u_i = 0.15$,
   non-unit exposure weights $a_i = 0.7 + 0.7\tfrac{i-1}{23}$ with $a^\top z = 1.0$,
   $X_1 \in \mathrm{St}(32, 6)$,
   $X_2 \in \mathrm{St}(6, 6)$,
   and total dimension $N_{\mathrm{tot}} = 252 \le 450$ so all 8 solvers including `SLSQP` and `trust-constr` execute without size skip):
   With SPD asset covariance $\Sigma \in \mathbb R^{24 \times 24}$ (condition number $\kappa(\Sigma) = 500$), return drift $\mu \in \mathbb R^{24}$, cross-coupling matrix $M \in \mathbb R^{32 \times 24}$,
   alignment matrices $B_{\mathrm{tgt}} \in \mathbb R^{32 \times 6}, C_{\mathrm{tgt}} \in \mathbb R^{6 \times 6}$ drawn from `PCG64(20260802)`,
   $\lambda_{\mathrm{fac}} = 2.0$,
   and $D_{\mathrm{tgt}} = \mathrm{diag}(0.30, 0.22, 0.16, 0.12, 0.08, 0.05)$,
   let $U = X_1 X_2 \in \mathrm{St}(32, 6)$, $V = M^\top U \in \mathbb R^{24 \times 6}$,
   and $E = V^\top \mathrm{diag}(z) V - D_{\mathrm{tgt}}$:

```math
f(z, X_1, X_2) = \frac{1}{2} z^\top \Sigma z - \mu^\top z + \frac{\lambda_{\mathrm{fac}}}{2} \lVert E \rVert_{\mathrm F}^2 - \mathrm{tr}\!\bigl(B_{\mathrm{tgt}}^\top X_1 X_2 C_{\mathrm{tgt}}^\top\bigr),
```

```math
\nabla_z f = \Sigma z - \mu + \lambda_{\mathrm{fac}} \, \mathrm{diag}(V E V^\top), \quad \nabla_U f = 2 \lambda_{\mathrm{fac}} M \mathrm{diag}(z) V E - B_{\mathrm{tgt}} C_{\mathrm{tgt}}, \quad \nabla_{X_1} f = (\nabla_U f) X_2^\top, \quad \nabla_{X_2} f = X_1^\top (\nabla_U f).
```

### 6.3 Head-to-Head Baseline Solvers (`benchmarks/baselines.py`)

1. **`fortchuck` (Default `m = 7`)**: Primary coupled quasi-Newton solver under test (`fortchuck_solve` on `B1..B7B`, `fortchuck_solve_constrained` on `B8A..B8B`).
2. **`fortchuck-ABB` (`m = 0` ablation)**: First-order spectral steepest-descent ablation (`ABB` on `B1..B7B`, isotropic `SPG` + Stiefel `ABB` on `B8A..B8B`).
3. **`Alternating-BCD`**: Runs 10 inner `SciPy L-BFGS-B` steps on $z \in \mathcal B$ (or 10 inner `SciPy SLSQP` equality-constrained steps + projection onto $\mathcal C$ when `lin_a is not None`) alternating with 10 inner `pymanopt` `ConjugateGradient` steps on `X_{1..R}` per cycle under `max_eval = 5000`.
4. **`pymanopt-Warped-CG` & `pymanopt-TrustRegions`**: `pymanopt 2.2.1` on `Product([Euclidean(d), Stiefel(...)])` via smooth `tanh` / `exp` box warping (augmented on `B8A..B8B` with quadratic linear-equality penalty continuation $\rho \in \lbrace 10^2, 10^4, 10^6 \rbrace$ on $\tfrac{\rho}{2}(a^\top z(\theta) - b)^2$).
5. **`SciPy-L-BFGS-B`**: Native Fortran `L-BFGS-B` on `B7A`, and quadratic penalty continuation ($\rho \in \lbrace 10^2, 10^4, 10^6 \rbrace$) on Stiefel orthogonality (plus $\tfrac{\rho}{2}(a^\top z - b)^2$ on `B8A..B8B`).
6. **`SciPy-SLSQP` & `SciPy-trust-constr`**: Constrained Euclidean NLP with explicit `Bounds(l, u)`, linear constraint $a^\top z = b$ (when `lin_a is not None`), and $\mathrm{vech}(X_r^\top X_r - I_{p_r}) = 0$ equality constraints (`N_tot <= 450`, `20s` per-seed wall timeout — executing directly on `B3` and `B8B`).

---

## 7. Phase 2 Disjoint Swarm Implementation Partition (`v0.1.0` Baseline)

Historical Phase 2 ownership table that delivered the verified `v0.1.0` baseline tag:

| Worker Role | Owned Files (Strict Write Boundary) | Deliverable & Verification Gate |
| :--- | :--- | :--- |
| **`fortran-core`** | `fpm.toml`, `src/fortchuck_kinds.f90`, `src/fortchuck_linalg.f90`, `src/fortchuck_box.f90`, `src/fortchuck_stiefel.f90`, `test/test_linalg_and_stiefel.f90` | `100% pure` linalg, overflow-safe box engine, dual-regime Schur-Woodbury & Direct LU Cayley + `geqrf_pos`; passes `fpm test` with sanitizers + `0` `nm -u` alloc symbols. |
| **`fortran-solver`** | `include/fortchuck.h`, `src/fortchuck_lbfgs.f90`, `src/fortchuck_solver.f90`, `src/fortchuck_c_abi.f90`, `src/fortchuck.f90`, `test/test_coupled_solver.f90`, `test/test_openmp_reentrancy.f90` | Masked `GEMM`-free product L-BFGS, `fc_solve` master loop, C-ABI, coupled & 16-thread OpenMP bit-reproducibility tests; passes `fpm test` + `0` `nm -u` alloc symbols. |
| **`python-ffi`** | `pyproject.toml`, `python/fortchuck/__init__.py`, `python/fortchuck/_native.py`, `python/fortchuck/solver.py`, `scripts/build_shared.py`, `scripts/verify_zero_alloc.py`, `scripts/audit_gfm_math.py`, `tests/test_c_abi_and_python.py`, `tests/test_pathological_guards.py` | Shared `.dll`/`.so` build, zero-copy `order="F"` Python bindings, `nm -u` zero-alloc verifier, and `pytest` FFI/guard suite (`100%` pass). |
| **`lean-prover`** | `formalization/lean-toolchain`, `formalization/lakefile.lean`, `formalization/Fortchuck.lean` | Interactive `lean-lsp` formalization of core Stiefel Cayley & coupled descent theorems (`0 sorry`, `0` warnings, `lake build` clean). |
| **`bench-docs`** | `benchmarks/problems.py`, `benchmarks/referee.py`, `benchmarks/baselines.py`, `benchmarks/run_blind_suite.py`, `.github/workflows/ci.yml`, `CITATION.cff`, `README.md` | Blind 10-seed benchmark execution across `B1..B7B` vs all baselines (`results/blind_benchmark_summary.json`), CI workflow, and publication-grade `README.md` (`0` GFM math violations). |

---

## 8. Phase 3 Disjoint Swarm Implementation Partition (`v0.2.0` Upgrade)

Phase 3 executes across **five strictly disjoint file-ownership slices** with zero write overlap (Wave A: `fortran-core-v2` + `lean-prover-v2` in parallel; Wave B: `fortran-solver-v2` building atop `fortran-core-v2`; Wave C: `python-ffi-v2` + `bench-docs-v2` running shared-library FFI tests and the full 30-seed `B1..B8B` blind evaluation across `101..110`, `201..210`, and untouched `301..310`):

| Worker Role | Owned Files (Strict Write Boundary) | `v0.2.0` Deliverable & Verification Gate |
| :--- | :--- | :--- |
| **`fortran-core-v2`** | `src/fortchuck_kinds.f90`, `src/fortchuck_linalg.f90`, `src/fortchuck_box.f90`, `src/fortchuck_stiefel.f90`, `test/test_linalg_and_stiefel.f90` | Add `FC_ERR_INFEASIBLE_LIN = -10`; upgrade `fc_gemm_*` (`4x4`/`1x4` register tiling) & `fc_lu_solve*` (hoisted reciprocals + `4`-RHS panels); add `fc_cayley_transport_batch`, `fc_box_active_step_diag`, `fc_box_breakpoint_cap`, and Dai--Fletcher `fc_box_lin_*` while keeping all `v0.1.0` signatures intact; pass `fpm test` with sanitizers + `verify_zero_alloc.py`. |
| **`fortran-solver-v2`** | `fpm.toml`, `app/fortchuck_bench.f90`, `include/fortchuck.h`, `src/fortchuck_lbfgs.f90`, `src/fortchuck_solver.f90`, `src/fortchuck_c_abi.f90`, `src/fortchuck.f90`, `test/test_coupled_solver.f90`, `test/test_openmp_reentrancy.f90` | Wire batched panel transport, `diag_box`, Quadratic/Cubic Hermite + breakpoint line search, Rank-1 KKT Nullspace Two-Loop + `SPG` fallback, `fc_solve_constrained`, constrained C-ABI, and native CLI `fortchuck-bench` (`v0.2.0`); pass `fpm test` (incl. 16-thread OpenMP bit-identity), `fpm run fortchuck-bench --profile release`, and `verify_zero_alloc.py`. |
| **`python-ffi-v2`** | `pyproject.toml`, `python/fortchuck/__init__.py`, `python/fortchuck/_native.py`, `python/fortchuck/solver.py`, `scripts/build_shared.py`, `tests/test_c_abi_and_python.py`, `tests/test_pathological_guards.py` | Rebuild shared `.dll`/`.so`, bind `fortchuck_solve_constrained`, `fortchuck_project_linear_box`, `fortchuck_kkt_residual_constrained` & `FC_ERR_INFEASIBLE_LIN`, add `linear_constraint=(a, b)` + `project_linear_box`, sync §3.4 workspace test, and pass `pytest` (`100%` pass incl. FM-13 simplex vertex release). |
| **`lean-prover-v2`** | `formalization/Fortchuck.lean` | Prove **Theorem 6 (`simplex_box_projection_variational`)** and **Theorem 7 (`rank1_kkt_nullspace_quasinewton_descent`)** interactively via `lean-lsp` MCP (`7` total theorems, `0 sorry`, `0` warnings, `lake build` clean). |
| **`bench-docs-v2`** | `benchmarks/problems.py`, `benchmarks/referee.py`, `benchmarks/baselines.py`, `benchmarks/run_blind_suite.py`, `.github/workflows/ci.yml`, `CITATION.cff`, `README.md`, `results/*` | Implement `B8A` & `B8B`, independent referee `project_linear_box_ref` + `check_grad`, constrained baseline adapters, run blind 30-seed evaluation (`101..110`, `201..210`, `301..310`) across `B1..B8B`, update `README.md`, `CITATION.cff` (`0.2.0`), `ci.yml`, and `results/blind_benchmark_report.md`, and pass `audit_gfm_math.py` (`0` issues across all `.md`). |
