# fortchuck: Zero-Allocation Coupled Box/Simplex × Stiefel Product-Manifold Quasi-Newton Optimizer

[![CI Matrix](https://github.com/sneed-and-feed/fortchuck/actions/workflows/ci.yml/badge.svg)](https://github.com/sneed-and-feed/fortchuck/actions/workflows/ci.yml)
[![Lean 4 Formalization](https://img.shields.io/badge/Lean_4-15_theorems_0_sorry-00f0ff.svg?style=flat-square)](formalization/Fortchuck.lean)
[![Fortran 2018 Zero-Alloc](https://img.shields.io/badge/Fortran_2018-0_heap_alloc_pure-b844ff.svg?style=flat-square)](scripts/verify_zero_alloc.py)
[![OpenMP Reentrancy](https://img.shields.io/badge/OpenMP-16--thread_bit--identical-ffb800.svg?style=flat-square)](test/test_openmp_reentrancy.f90)
[![License](https://img.shields.io/badge/License-CC0--1.0_Universal-lightgrey.svg?style=flat-square)](LICENSE)

A standalone, zero-heap-allocation, `100%` OpenMP-reentrant **Modern Fortran 2018 + ANSI C99 ABI + Zero-Copy Python (`ctypes`)** solver (`v0.2.1`) for smooth nonconvex optimization over heterogeneous products of bound-constrained Euclidean boxes, singly-linearly-constrained simplices/knapsacks, and multiple Stiefel manifolds, paired with **15 machine-checked Lean 4** algebraic, invertibility, L-BFGS positive-definiteness, and variational theorems (`0 sorry`, `0` custom axioms, `0` compiler warnings) and a strictly blind 40-seed (`3,520`-run, incl. virgin one-shot `blind4` `401..410`) four-suite referee benchmark archive.

---

## Table of Contents

- [1. Executive Overview](#1-executive-overview)
- [2. Coupled Product Geometry and v0.2.0 Four-Pillar Architecture](#2-coupled-product-geometry-and-v020-four-pillar-architecture)
- [3. Lean 4 Formal Verification](#3-lean-4-formal-verification)
- [4. Quickstart: Zero-Copy Python, Modern Fortran 2018, and ANSI C99](#4-quickstart-zero-copy-python-modern-fortran-2018-and-ansi-c99)
- [5. Blind 40-Seed Benchmark Suite and Native CLI Telemetry](#5-blind-40-seed-benchmark-suite-and-native-cli-telemetry)
- [6. Repository Structure and Verification Gates](#6-repository-structure-and-verification-gates)
- [7. Citation](#7-citation)
- [8. License](#8-license)

---

## 1. Executive Overview

`sneed-and-feed/fortchuck` (`v0.2.1`, `CC0-1.0` Public Domain) solves smooth coupled problems over arbitrary Cartesian products of box- or simplex/knapsack-constrained coordinates and orthogonal Stiefel frames:

```math
\min_{z \in \mathcal D, \; X_1, \dots, X_R} \; f(z, X_1, \dots, X_R)
```

where the Euclidean factor $\mathcal D$ is either an extended-real box $\mathcal B$ (`fc_solve`) or a singly-linearly-constrained box/simplex polytope $\mathcal C$ (`fc_solve_constrained` / `linear_constraint=(a, b)`), coupled to $R$ Stiefel manifolds:

```math
\mathcal B = \left\lbrace z \in \mathbb R^d : l_i \le z_i \le u_i \right\rbrace, \qquad \mathcal C = \left\lbrace z \in \mathcal B : a^\top z = b \right\rbrace, \qquad X_r \in \mathrm{St}(n_r, p_r) = \left\lbrace X \in \mathbb R^{n_r \times p_r} : X^\top X = I_{p_r} \right\rbrace
```

with $1 \le p_r \le n_r$, extended-real bounds
$(l_i, u_i) \in [-\infty, +\infty) \times (-\infty, +\infty]$,
and $l_i \le u_i$ (including fixed variables `l_i == u_i`, unbounded sentinel limits, unit simplices, and weighted budget hyperplanes).

| Optimization Regime | Configuration | Representative Scientific & Engineering Workloads |
| :--- | :---: | :--- |
| **Pure Box / Simplex Limit** | $d \ge 1, \; R = 0$ | Active-bound NLP, non-negative least squares, simplex/knapsack portfolio allocation |
| **Pure Multi-Stiefel Limit** | $d = 0, \; R \ge 1$ | Brockett subspace diagonalization, $\mathrm{SO}(n)$ Procrustes & multi-camera Wahba alignment |
| **Coupled Box × Multi-Stiefel** | $z \in \mathcal B, \; R \ge 1$ | Soft-penalized DFT, sparse PCA, bounded-variance ML-CPC, amplitude-limited ICA |
| **Coupled Simplex/Knapsack × Multi-Stiefel** | $z \in \mathcal C, \; R \ge 1$ | Hard electron-conserving ensemble Kohn--Sham DFT, long-only orthogonal risk-factor parity |

### Core Engineering Guarantees

1. **Strict Zero Heap Allocation & Zero Stack VLAs (`nm -u` Verified)**: Every library kernel in `src/*.f90` executes exclusively inside caller-supplied `work(1:lwork)` and `iwork(1:liwork)` scratch buffers queried via `lwork = -1` or `fortchuck_workspace_query`. Automated symbol auditing ([`scripts/verify_zero_alloc.py`](scripts/verify_zero_alloc.py)) confirms **zero** `malloc`, `free`, `_gfortran_allocate*`, `_gfortran_internal_pack*`, or Fortran I/O/stop symbols across all compiled object files.
2. **Bit-Identical Multi-Threaded OpenMP Reentrancy**: Zero `save` variables, zero mutable module state, and `100% pure` computational kernels allow concurrent `fc_solve` and `fc_solve_constrained` calls across `16+` OpenMP threads with `0.0` byte-level discrepancy against sequential execution ([`test/test_openmp_reentrancy.f90`](test/test_openmp_reentrancy.f90)).
3. **Binary-Compatible Zero-Copy C99 & Python ABI**: Naturally aligned `fc_options_t` (`96` bytes) and `fc_result_t` (`88` bytes) structs preserve `100%` `v0.1.0` ABI compatibility while exposing 8 C99 entry points ([`include/fortchuck.h`](include/fortchuck.h)) and zero-copy NumPy `order="F"` views ([`python/fortchuck/`](python/fortchuck/)).

---

## 2. Coupled Product Geometry and v0.2.0 Four-Pillar Architecture

Existing optimization toolboxes force practitioners to choose between smooth manifold solvers without native bound or linear-conservation constraints (where `tanh` warps and $\tfrac{\rho}{2}(a^\top z - b)^2$ penalties stall at $\mathcal O(1/\rho)$ feasibility errors) and alternating block-coordinate schemes (which destroy joint quasi-Newton cross-curvature). `fortchuck` unifies active-set projection, singly-linearly-constrained KKT nullspace geometry, and Riemannian quasi-Newton transport from first principles ([`docs/ARCHITECTURE_SPEC.md`](docs/ARCHITECTURE_SPEC.md)):

### 2.1 Safeguarded Bertsekas Active Band, Rotation-Invariant BB2 Seeding, and Scale-Invariant Arc Search

At iterate $p_k = (z_k, X_{1,k}, \dots, X_{R,k})$, `fortchuck` partitions $\{1, \dots, d\}$ into fixed coordinates (`l_i == u_i`),
safeguarded outward-pointing active-band coordinates $\mathcal A_{\epsilon_k}(z_k)$,
and free coordinates $\mathcal F_{\epsilon_k}(z_k)$. Inside every L-BFGS two-loop recursion, dynamic virtual masking retains secant slot $j$ if and only if its free-subspace curvature satisfies:

```math
\tilde\sigma_j = \sum_{i \in \mathcal F_{\epsilon_k}} s_{j,\mathrm{box},i} y_{j,\mathrm{box},i} + \sigma_{j,\mathrm{St}}^{sy} \ge 10^{-10} \sqrt{\tilde\nu_{s,j} \tilde\nu_{y,j}}
```

eliminating clamped-multiplier cross-contamination (`FM-2`). In `v0.2.1`, free Euclidean coordinates use rotation-invariant scalar Oren--Luenberger / BB2 seeding $\gamma_{\mathrm{box}} \in [10^{-20}, 10^{20}]$ (eliminating coordinatewise seed distortion on rotated valleys while preserving exact `v0.2.0` workspace ABI),
paired with a dimensionless cosine chord-descent check ($\Delta_\tau \le -10^{-12} \lVert s(\tau) \rVert_2 g_0$, invariant under objective rescaling $f \mapsto \alpha f$ across $\alpha \in [10^{-14}, 10^{12}]$), cancellation-free delta-space KKT residuals, and safeguarded Quadratic/Cubic Hermite backtracking over $[0.1\tau, \, 0.5\tau]$ along the Zhang--Hager nonmonotone reference arc.

### 2.2 Dual-Regime Exact Cayley Retraction and Batched Level-3 Panel Transport

Under the canonical Stiefel metric ($\beta = 1/2$), each tangent step $\eta_r \in T_{X_r}\mathrm{St}(n_r, p_r)$ defines the skew-symmetric Lie generator
$W_{\eta_r} = \tilde\eta_r X_r^\top - X_r \tilde\eta_r^\top \in \mathfrak{so}(n_r)$,
where $\tilde\eta_r = \eta_r - \tfrac{1}{2} X_r (X_r^\top \eta_r)$. `fortchuck` dispatches automatically per factor:

- **Branch A** ($2 p_r \lt n_r$, Tall Stiefel — Schur-Woodbury):
  Solves the strictly dissipative $p_r \times p_r$ Schur complement:

```math
J_r(\tau) = I_{p_r} - \frac{\tau}{2} A_r + \frac{\tau^2}{4} K_r^\top K_r, \qquad \mathcal R_{X_r}^{\mathrm{Cay}}(\tau \eta_r) = (2 X_r + \tau K_r) J_r(\tau)^{-1} - X_r
```

- **Branch B** ($2 p_r \ge n_r$, Wide/Square Orthogonal Group — Direct Skew-Resolvent LU):
  Factors $N_r(\tau) = I_{n_r} - \tfrac{\tau}{2} W_{\eta_r}$ directly
  in $\mathbb R^{n_r \times n_r}$:

```math
M_r(\tau) = \left(I_{n_r} - \frac{\tau}{2} W_{\eta_r}\right)^{-1}, \qquad Q_r(\tau) = 2 M_r(\tau) - I_{n_r}, \qquad \mathcal R_{X_r}^{\mathrm{Cay}}(\tau \eta_r) = Q_r(\tau) X_r
```

In `v0.2.0`, `fc_lbfgs_transport_and_push` packs all $K_{\mathrm{batch}} = 2 m_{\mathrm{cur}} + 2$ historical tangent matrices into a single contiguous `(n_r, p_r * K_batch)` panel in workspace and invokes `fc_cayley_transport_batch` **once** per factor—replacing $2m+2$ separate matrix-vector sweeps with Level-3 `GEMM` (`4x4`/`1x4` register-tiled SIMD FMA micro-kernels or optional OpenBLAS) and `4`-RHS division-hoisted LU solves while preserving canonical inner products to $\le 10^{-14}$.

### 2.3 Singly-Linearly-Constrained Box/Simplex (`SLBQP`) and Rank-1 KKT Nullspace Two-Loop

When $z \in \mathcal C = \left\lbrace z \in [l, u] : a^\top z = b \right\rbrace$, `fc_box_lin_project` computes the exact Euclidean projection
$P_{\mathcal C}(y)$ via
$\ell_\infty$-rescaled Dai--Fletcher / Cominetti--Mascarenhas semismooth Newton-secant + safeguarded bisection with Neumaier compensated summation (`|a^T z - b| <= 10^{-14} * (1 + |b|)` in $\le 60$ passes).

On the free coupled subspace, applying the masked positive-definite product L-BFGS inverse $H_k$ via **two** matrix-free two-loop calls
($u = -H_k \tilde g$ and
$w = H_k \tilde a$) solves the exact equality-constrained quasi-Newton KKT system:

```math
d_{\mathrm{free}} = u - \frac{\langle \tilde a, u \rangle - \delta_b}{\langle \tilde a, w \rangle} w, \qquad \langle \tilde a, d_{\mathrm{free}} \rangle = \delta_b
```

which is provably invariant under arbitrary gauge shifts $g_z \mapsto g_z + \mu a$ and strictly descending for homogeneous steps whenever $\tilde g \notin \mathrm{span}(\tilde a)$,
paired with active-bound outward hold during trial projection (preventing multiplier-shift lift-off on tightly capped simplices), Stiefel L-BFGS history preservation across box-only active-face degeneracy, a runtime directional-derivative guard, and a non-jamming isotropic Spectral Projected Gradient (`SPG`) fallback $d_z^{\mathrm{SPG}} = P_{\mathcal C}(z_k - \alpha_k \nabla_z f) - z_k$ that releases simplex corners in a single step (`FM-13`).

---

## 3. Lean 4 Formal Verification

All **15** foundational algebraic, unconditional invertibility, Branch A batched transport, filtered L-BFGS positive-definiteness, variational projection, and KKT nullspace theorems are machine-checked in [`formalization/Fortchuck.lean`](formalization/Fortchuck.lean) (`leanprover/lean4:v4.34.0-rc2` + Mathlib4, `2,074` lines, `0 sorry`, `0` custom axioms, `0` compiler warnings, `0` orphan definitions — every theorem individually audited via `lean_verify` to depend solely on `[propext, Classical.choice, Quot.sound]`):

| Pillar | Theorem Identifier | Mathematical Guarantee | Machine-Checked Statement |
| :---: | :--- | :--- | :--- |
| 1 | `skew_generator_properties` | $\mathfrak{so}(n)$ Lie Generator & Tangent Reproduction | $X^\top X = I_p \land X^\top \xi + \xi^\top X = 0 \implies W_\xi^\top = -W_\xi \land W_\xi X = \xi$ |
| 2a | `cayley_resolvent_isUnit` | Unconditional Resolvent Invertibility | $W^\top = -W \implies \mathrm{IsUnit}(I_n - \tfrac{\tau}{2} W)$ for all $\tau \in \mathbb R$ |
| 2b | `cayley_rotor_orthogonality_and_isometry` | Two-Sided Rotor Orthogonality & Canonical Isometry | $Q = 2 M - I_n, \; Q^\top Q = Q Q^\top = I_n, \; \langle Q \zeta_1, Q \zeta_2 \rangle_{Q X} = \langle \zeta_1, \zeta_2 \rangle_X$ |
| 3a | `schur_woodbury_isUnit` | Unconditional Schur-Woodbury Invertibility | $A^\top = -A \implies \mathrm{IsUnit}(I_p - \tfrac{\tau}{2} A + \tfrac{\tau^2}{4} K^\top K)$ for all $\tau \in \mathbb R$ |
| 3b | `schur_woodbury_cayley_orthogonality` | Rank $2p$ Schur-Woodbury Exact Feasibility | $Y = (2 X + \tau K) J(\tau)^{-1} - X \implies Y^\top Y = I_p$ |
| 4a | `cayley_branch_a_eq_branch_b` | Retraction Equivalence of Branch A and Branch B | $Q(\tau) X = (2 X + \tau K) J(\tau)^{-1} - X$ identically for all $(n, p, \tau)$ |
| 4b | `cayley_branch_a_transport_eq_branch_b` | Single & Batched Panel Transport Equivalence | $Q(\tau) \zeta = \mathcal T_A(X, \xi, \tau, \zeta)$ for arbitrary panel $\zeta \in \mathbb R^{n \times q}$ |
| 4c | `schur_woodbury_transport_isometry` | Branch A Canonical Transport Isometry | $Y^\top \mathcal T_A(\zeta_1) = X^\top \zeta_1, \; \langle \mathcal T_A(\zeta_1), \mathcal T_A(\zeta_2) \rangle_Y = \langle \zeta_1, \zeta_2 \rangle_X$ |
| 5a | `lbfgs_two_loop_symm_posDef_bound` | Inductive Filtered Two-Loop Symmetry & PD Bound | $\forall j, \langle s_j, y_j \rangle > 0 \implies \exists \lambda_{\min} > 0, \; \langle v, H v \rangle \ge \lambda_{\min} \lVert v \rVert^2$ |
| 5b | `coupled_two_metric_strict_descent` | Unconditional Box × Stiefel Strict Descent & KKT | $\phi'(0^+) \le -\gamma_{\min} \lVert g_{\mathcal A^{\mathrm{int}}} \rVert_2^2 - \lambda_{\min} \lVert \tilde g \rVert^2 \le 0$, with $\phi'(0^+) = 0 \iff \mathrm{KKT}(z, X)$ |
| 5c | `masked_lbfgs_coupled_strict_descent` | End-to-End Masked Product L-BFGS Descent | Discharges `h_free_pd` directly from `lbfgsTwoLoop` + `filter_free_subspace_pairs` |
| 6a | `simplex_box_dual_root_exists` | Feasibility Certificate & IVT Dual Root Existence | $\mathcal C \neq \emptyset \iff \underline b \le b \le \overline b \implies \exists \nu \in \mathbb R, \; \sum_i a_i z^\ast_i(\nu) = b$ |
| 6b | `simplex_box_projection_variational` | Variational `SLBQP` Optimality, Shift Invariance & `SPG` | $\langle y - z^\ast, z - z^\ast \rangle \le 0, \; P_{\mathcal C}(y + \mu a) = P_{\mathcal C}(y), \; \langle g, d_{\mathrm{spg}} \rangle \le -\gamma^{-1} \lVert d_{\mathrm{spg}} \rVert_2^2$ |
| 6c | `coupled_spg_stiefel_strict_descent` | Coupled $\mathcal C \times \prod \mathrm{St}$ Fallback Descent | Unconditional feasibility & strict descent of `fc_spg_lin_direction` |
| 7 | `rank1_kkt_nullspace_quasinewton_descent` | Rank-1 KKT Nullspace Tangency, Shift & Inhomogeneous Formula | $a^\top d = 0, \; d(g + \mu a) = d(g), \; g^\top d_{\mathrm{inhom}} = -v^\top H v + \tfrac{\delta_b}{a^\top H a}(g^\top H a)$ |

---

## 4. Quickstart: Zero-Copy Python, Modern Fortran 2018, and ANSI C99

### 4.1 Zero-Copy Python Interface (`minimize_box_stiefel` & `project_linear_box`)

```bash
# Build native shared library (.dll / .so) and install editable package inside a virtualenv
python scripts/build_shared.py
pip install -e ".[dev,bench]"
```

```python
import numpy as np
from fortchuck import (
    FCOptions,
    FortchuckWorkspace,
    kkt_residual,
    minimize_box_stiefel,
    project_linear_box,
)

n, p = 60, 6
H = np.diag(np.linspace(-4.0, 2.0, n))
lower, upper = np.zeros(p), np.ones(p)
lin_a, lin_b = np.ones(p), 3.0  # Exact trace / particle-number conservation: 1^T z = 3.0

def dft_toy_fg(z: np.ndarray, X_list: list[np.ndarray]):
    X = X_list[0]  # Zero-copy read-only Fortran-contiguous (n, p) view
    HX = H @ X
    eps_orb = np.sum(X * HX, axis=0)
    f = float(np.dot(z, eps_orb) + 0.25 * np.sum(z * z))
    gz = eps_orb + 0.5 * z
    GX = np.asfortranarray(2.0 * HX * z[None, :])
    return f, gz, [GX]

z0, _ = project_linear_box(np.full(p, 0.5), lower, upper, lin_a, lin_b)
X0 = np.asfortranarray(np.eye(n, p, dtype=np.float64))
ws = FortchuckWorkspace(d=p, n_list=[n], p_list=[p], memory=7)

# Pass linear_constraint=(lin_a, lin_b) for z in C, or omit for unconstrained box z in B
res = minimize_box_stiefel(
    dft_toy_fg,
    z0=z0,
    bounds=(lower, upper),
    X0_list=[X0],
    options=FCOptions(gtol=1e-8, max_eval=2000),
    workspace=ws,
    linear_constraint=(lin_a, lin_b),
)
_, gz_fin, gx_fin = dft_toy_fg(res.z, res.X_list)
kkt_inf, lin_err, orth_err = kkt_residual(
    res.z, (lower, upper), res.X_list, gz_fin, gx_fin, linear_constraint=(lin_a, lin_b)
)
assert res.converged and kkt_inf <= 1e-8 and abs(np.dot(lin_a, res.z) - lin_b) <= 1e-13
```

### 4.2 Modern Fortran 2018 (`fc_solve` & `fc_solve_constrained`)

Add `fortchuck` to `fpm.toml` and invoke `fc_workspace_query` + `fc_solve` (or `fc_solve_constrained` when $a^\top z = b$ is active) with caller-owned `work` and `iwork` buffers:

```fortran
program demo_fortchuck_v2
  use, intrinsic :: iso_c_binding, only: c_int32_t, c_double, c_ptr, c_null_ptr
  use fortchuck, only: dp, i32, i64, fc_options_t, fc_result_t, &
    fc_init_default_options, fc_workspace_query, fc_solve_constrained, FC_CONVERGED_GRAD
  implicit none (type, external)

  integer(i32), parameter :: d = 3_i32, R = 1_i32
  integer(i32)            :: n_arr(1) = [4_i32], p_arr(1) = [2_i32], status
  integer(i64)            :: lwork, liwork
  real(dp)                :: z(3) = [0.5_dp, 0.3_dp, 0.2_dp]
  real(dp)                :: l(3) = [0.0_dp, 0.0_dp, 0.0_dp], u(3) = [1.0_dp, 1.0_dp, 1.0_dp]
  real(dp)                :: a(3) = [1.0_dp, 1.0_dp, 1.0_dp], b = 1.0_dp
  real(dp)                :: X(8) = [1.0_dp, 0.0_dp, 0.0_dp, 0.0_dp, 0.0_dp, 1.0_dp, 0.0_dp, 0.0_dp]
  real(dp)                :: work(8192)
  integer(i32)            :: iwork(256)
  type(fc_options_t)      :: opts
  type(fc_result_t)       :: res

  call fc_init_default_options(opts)
  call fc_workspace_query(d, R, n_arr, p_arr, opts%memory, lwork, liwork, status)
  call fc_solve_constrained(d, R, n_arr, p_arr, z, l, u, a, b, X, eval_demo, c_null_ptr, &
    opts, work, lwork, iwork, liwork, res)

contains

  function eval_demo(d_in, R_in, na, pa, z_in, X_in, f_out, gz_out, GX_out, udata) &
      bind(C) result(rc)
    integer(c_int32_t), value, intent(in) :: d_in, R_in
    integer(c_int32_t), intent(in)        :: na(*), pa(*)
    real(c_double), intent(in)            :: z_in(*), X_in(*)
    real(c_double), intent(out)           :: f_out
    real(c_double), intent(inout)         :: gz_out(*), GX_out(*)
    type(c_ptr), value, intent(in)        :: udata
    integer(c_int32_t)                    :: rc
    integer(i32)                          :: k

    f_out = 0.5_dp * ((z_in(1) - 0.7_dp)**2 + (z_in(2) - 0.2_dp)**2 + (z_in(3) - 0.1_dp)**2)
    gz_out(1) = z_in(1) - 0.7_dp
    gz_out(2) = z_in(2) - 0.2_dp
    gz_out(3) = z_in(3) - 0.1_dp
    do k = 1_i32, 8_i32
      f_out     = f_out + 0.5_dp * real(k, dp) * X_in(k)**2
      GX_out(k) = real(k, dp) * X_in(k)
    end do
    rc = 0_i32
  end function eval_demo
end program demo_fortchuck_v2
```

### 4.3 ANSI C99 / C++11 (`include/fortchuck.h`)

```c
#include "fortchuck.h"
#include <stdlib.h>

/* Query unified caller-owned buffer sizes and run zero-allocation constrained solve */
int run_c99_constrained_example(fc_eval_fg_cb cb, void *udata) {
    int32_t n_arr[1] = {30}, p_arr[1] = {5};
    double z[5] = {0.2, 0.2, 0.2, 0.2, 0.2}, lambda_star = 0.0;
    double l[5] = {0.0, 0.0, 0.0, 0.0, 0.0};
    double u[5] = {1.0, 1.0, 1.0, 1.0, 1.0};
    double a[5] = {1.0, 1.0, 1.0, 1.0, 1.0}, b = 1.0;
    double X[150] = {0.0};
    for (int j = 0; j < 5; ++j) X[j * 30 + j] = 1.0; /* Column-major I_{30x5} */

    fortchuck_project_linear_box(5, l, u, a, b, z, &lambda_star);

    fc_options_t opts;
    fortchuck_default_options(&opts);

    int64_t lwork = 0, liwork = 0;
    fortchuck_workspace_query(5, 1, n_arr, p_arr, opts.memory, &lwork, &liwork);

    double  *work  = (double *)malloc((size_t)lwork * sizeof(double));
    int32_t *iwork = (int32_t *)malloc((size_t)liwork * sizeof(int32_t));
    fc_result_t res;

    int32_t status = fortchuck_solve_constrained(
        5, 1, n_arr, p_arr, z, l, u, a, b, X, cb, udata, &opts, work, lwork, iwork, liwork, &res
    );
    free(iwork);
    free(work);
    return (int)status;
}
```

---

## 5. Blind 40-Seed Benchmark Suite and Native CLI Telemetry

To prevent hyperparameter overfitting, [`benchmarks/`](benchmarks) enforces a strict blind firewall from `src/*.f90`. Every solver is initialized from bit-identical `PCG64` feasible points across **40 deterministic seeds**: `10` primary (`101..110`), `10` holdout (`201..210`), `10` third holdout (`301..310`), and `10` **virgin fourth blind holdout (`401..410`)** held completely untouched until all `v0.2.1` solver fixes were frozen and passed every sanitizer and unit gate. All `440` initial analytical Euclidean gradients are pre-verified against central finite differences to relative error $\le 1.98 \times 10^{-8} \le 10^{-7}$ ([`benchmarks/referee.py`](benchmarks/referee.py)), and `fortchuck` runs with **strictly frozen out-of-the-box** `FCOptions(gtol=1e-6, max_eval=5000)` (`memory = 7`, `max_iter = 15000`, `zh_eta` auto `0.85`, canonical metric, cautious skip) across every problem and seed.

A run is certified **Solved** by the independent pure-NumPy referee (`project_linear_box_ref`, zero `fortchuck` calls) if and only if:

```math
\mathcal E_{\mathrm{KKT,ref}}(z, X) = \max\!\left(\left\lVert z - P_{\mathcal C}\bigl(z - \nabla_z f(z, X)\bigr) \right\rVert_\infty, \; \max_{1 \le r \le R} \left\lVert G_r - X_r\,\mathrm{sym}(X_r^\top G_r) \right\rVert_{\mathrm F}\right) \le 10^{-6}
```

```math
\max\!\left(\mathcal V_{\mathrm{box}}, \, \mathcal V_{\mathrm{lin}}, \, \mathcal V_{\mathrm{orth}}\right) \le 10^{-8}, \qquad \mathcal V_{\mathrm{lin}} = \frac{\lvert a^\top z - b \rvert}{1 + \lvert b \rvert}
```

Full machine-readable `3,520`-run (`40` seeds × `11` problems × `8` solvers) JSON archives and per-problem breakdowns are stored in [`results/blind_benchmark_summary.json`](results/blind_benchmark_summary.json) and [`results/blind_benchmark_report.md`](results/blind_benchmark_report.md).

### 5.1 Canonical 11-Instance Benchmark Matrix (`B1` through `B8B`)

| ID | Benchmark Problem | Product Manifold Domain | Primary Coupling & Numerical Stress Tested |
| :--- | :--- | :--- | :--- |
| `B1` | Ensemble Kohn--Sham DFT Fractional Occupations | $[0, 1]^{10} \times \mathrm{St}(120, 10)$ | Branch A ($2p \lt n$), soft-penalty metallic orbital-occupation sloshing |
| `B2A` | Amplitude-Limited Blind Source Separation (ICA) | $[-2, 2]^{5 \times 150} \times \mathrm{St}(20, 5)$ | Branch A ($d = 750$), multi-modal nonconvex $\ln\cosh$ with $d \gg np$ active bounds |
| `B2B` | Smooth Penalty-Split Non-Negative Sparse PCA | $[0, 5]^{120 \times 6} \times \mathrm{St}(120, 6)$ | Branch A ($d = 720$), $\sim 80\%$ clamped $Z = 0$ with large multipliers (`FM-2`) |
| `B3` | Multi-Camera Anisotropic Bounded-Scale Rig Wahba | $\mathcal B^{15} \times \mathrm{St}(3, 3)^4$ | Branch B $(\mathrm{SO}(3)^4)$, $R = 4$ multi-factor $\kappa(W) = 10^3$ |
| `B4` | Bounded-Variance ML Common Principal Components | $[0.05, 20]^{150} \times \mathrm{St}(30, 30)$ | Branch B $(\mathrm{O}(30))$, log-variance barrier, $1.6 \times 10^5$ Hessian scale ratio |
| `B5` | Bounded Factor Correlation Matrix Calibration | $\mathcal B^{110} \times \mathrm{St}(100, 10)$ | Branch A, quartic factor valley + clamped idiosyncratic variances |
| `B6` | Box-Coupled Brockett with Certified Ky Fan QP Truth | $[0.1, 5]^{10} \times \mathrm{St}(150, 10)$ | Branch A, $\kappa(M) = 10^4$, certified analytical global optimum $f^\ast$ |
| `B7A` | Pure-Box Active-Bound Chained Rosenbrock + Quadratic | $[l, u]^{500} \; (R = 0)$ | Pure Box ($5$ fixed $l_i = u_i$, $\sim 60\%$ active bounds) vs `SciPy L-BFGS-B` |
| `B7B` | Pure-Stiefel Tall Brockett & Orthogonal Procrustes | $\mathrm{St}(200, 8) \times \mathrm{St}(24, 24) \; (d = 0)$ | Pure Stiefel Branch A & Branch B $(\mathrm{SO}(24))$ vs native `pymanopt` |
| `B8A` | Trace-Constrained Ensemble Kohn--Sham DFT (`v0.2.0+`) | $\mathcal C^{12} \times \mathrm{St}(120, 12) \; (\mathbf{1}^\top z = 5)$ | Hard Fermi--Dirac electron conservation (`SLBQP` + Rank-1 KKT Nullspace) |
| `B8B` | Long-Only Weighted Orthogonal Risk-Factor Parity (`v0.2.0+`) | $\mathcal C^{24} \times \mathrm{St}(32, 6) \times \mathrm{St}(6, 6)$ | Weighted budget $a^\top z = 1$ ($N_{\mathrm{tot}} = 252 \le 450$, direct `SLSQP`/`trust-constr` comparison) |

### 5.2 Pooled 40-Seed 8-Solver Aggregate Summary (`Seeds 101..410`, `440` Slots/Solver, `3,520` Runs)

Pooled metrics across all four blind suites (`101..110`, `201..210`, `301..310`, and virgin one-shot `401..410`), with explicit directional two-sided Wilcoxon polarity (`FC better` means the signed-rank sum favors lower/better values for `fortchuck`, `FC worse` favors the baseline):

| Solver | Evaluated Runs | Solved Runs | Solve Rate | Penalized Evals Med (IQR) | Wall Time ms Med (IQR) | Median KKT Ref | Max Lin Viol | Median Orth Viol | Wilcoxon Evals p (2-sided) | Wilcoxon KKT p (2-sided) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `fortchuck` (`v0.2.1`) | 440 | **440** | **100.0%** | **472.0 (1505.2)** | **206.77 (1130.73)** | `7.91e-07` | `2.96e-16` | `2.68e-14` | `—` | `—` |
| `fortchuck-ABB` (`v0.2.1`) | 440 | 330 | 75.0% | 1016.5 (4905.2) | 337.29 (1735.68) | `8.17e-07` | `1.78e-15` | `3.15e-14` | `2.20e-70 (FC better)` | `3.28e-18 (FC better)` |
| `Alternating-BCD` | 440 | 213 | 48.4% | 5000.0 (4643.0) | 2424.09 (3233.56) | `1.17e-05` | `5.55e-16` | `1.09e-15` | `3.68e-65 (FC better)` | `2.26e-57 (FC better)` |
| `pymanopt-Warped-CG` | 440 | 98 | 22.3% | 5000.0 (0.0) | 4950.47 (4332.76) | `1.70e-03` | `9.80e-05` | `1.13e-15` | `7.94e-74 (FC better)` | `1.96e-72 (FC better)` |
| `pymanopt-TrustRegions` | 440 | 106 | 24.1% | 5000.0 (0.0) | 3201.50 (2637.97) | `3.77e-03` | `8.32e-05` | `2.65e-15` | `7.94e-74 (FC better)` | `9.41e-63 (FC better)` |
| `SciPy-L-BFGS-B` | 440 | 40 | 9.1% | 5000.0 (0.0) | 2715.65 (3652.59) | `1.30e-04` | `1.02e-06` | `5.29e-06` | `9.69e-72 (FC better)` | `5.29e-73 (FC better)` |
| `SciPy-SLSQP` | 80 (`N_tot <= 450`) | 80 | 100.0% (`18.2%` all) | 58.0 (41.5) | 261.06 (395.50) | `6.31e-07` | `1.44e-15` | `1.17e-12` | `5.09e-10 (FC better)` | `7.05e-01 (FC worse)` |
| `SciPy-trust-constr` | 80 (`N_tot <= 450`) | 40 | 50.0% (`9.1%` all) | 2572.0 (4913.0) | 1898.06 (2960.78) | `1.29e-06` | `5.55e-16` | `7.55e-15` | `7.53e-15 (FC better)` | `1.87e-13 (FC better)` |

### 5.3 Four-Suite Generalization Across Primary (`101..110`), Holdout (`201..210`), Blind3 (`301..310`), and Virgin One-Shot `Blind4` (`401..410`)

| Solver | Primary (`101..110`) | Holdout (`201..210`) | Blind3 (`301..310`) | **Virgin `Blind4` (`401..410`)** | `Blind4` Pen. Evals Med | `Blind4` Wilcoxon Evals p |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `fortchuck` (`v0.2.1`) | **110/110 (100.0%)** | **110/110 (100.0%)** | **110/110 (100.0%)** | **110/110 (100.0%)** | **472.5** | `—` |
| `fortchuck-ABB` (`v0.2.1`) | 82/110 (74.5%) | 83/110 (75.5%) | 84/110 (76.4%) | 81/110 (73.6%) | 775.5 | `1.57e-18 (FC better)` |
| `Alternating-BCD` | 54/110 (49.1%) | 53/110 (48.2%) | 53/110 (48.2%) | 53/110 (48.2%) | 5000.0 | `3.39e-17 (FC better)` |
| `pymanopt-Warped-CG` | 23/110 (20.9%) | 25/110 (22.7%) | 25/110 (22.7%) | 25/110 (22.7%) | 5000.0 | `8.74e-20 (FC better)` |
| `pymanopt-TrustRegions` | 24/110 (21.8%) | 27/110 (24.5%) | 28/110 (25.5%) | 27/110 (24.5%) | 5000.0 | `8.74e-20 (FC better)` |
| `SciPy-L-BFGS-B` | 10/110 (9.1%) | 10/110 (9.1%) | 10/110 (9.1%) | 10/110 (9.1%) | 5000.0 | `2.15e-19 (FC better)` |
| `SciPy-SLSQP` | 20/20 (100.0%) | 20/20 (100.0%) | 20/20 (100.0%) | 20/20 (100.0%) | 59.5 | `5.69e-03 (FC better)` |
| `SciPy-trust-constr` | 10/20 (50.0%) | 10/20 (50.0%) | 10/20 (50.0%) | 10/20 (50.0%) | 2553.5 | `8.78e-05 (FC better)` |

### 5.4 Zero-FFI Native Fortran 2018 CLI Telemetry (`app/fortchuck_bench.f90`)

Pure Fortran 2018 release timings (`gfortran -cpp -O3 -march=native -funroll-loops -fopenmp`, `0` Python FFI overhead, `Seeds 101..110`) recorded by `fpm run fortchuck-bench`:

| Native Workload (`fortchuck-bench`) | Windows Pure F2018 | WSL2 Pure F2018 | WSL2 OpenBLAS (`-DFORTCHUCK_USE_BLAS`) | Exact Accuracy / Feasibility Certificate |
| :--- | :---: | :---: | :---: | :--- |
| `[1]` Batched Panel Transport (`St(200,8)`, `K_batch=16`, `4000` reps) | `872.00 ms` vs `857.31 ms` (`0.98x`) | `1064.39 ms` vs `1113.94 ms` (`1.05x`) | **`358.99 ms` vs `454.35 ms` (`1.27x`)** | `max_err = 0.00e+00` vs `16` single-vector calls |
| `[1]` Batched Panel Transport (`St(24,24)`, `K_batch=16`, `4000` reps) | `280.74 ms` vs `281.08 ms` (`1.00x`) | `324.28 ms` vs `327.19 ms` (`1.01x`) | `297.23 ms` vs `282.66 ms` (`0.95x`) | `max_err = 0.00e+00` vs `16` single-vector calls |
| `[2]` `B6` Box-Coupled Ky Fan QP (`10/10` solved) | `973.56 ms` (`1839.5` evals `v0.2.1`) | `1215.56 ms` | **`659.38 ms`** | `max |f - f*| <= 5.45e-12`, `max_KKT <= 9.96e-07` |
| `[3]` `B7B` Tall Brockett + Procrustes (`10/10` solved) | `374.51 ms` (`m=7`) / `212.91 ms` (`m=0`) | `1395.26 ms` (`m=7`) / `728.05 ms` (`m=0`) | **`249.95 ms` (`m=7`) / `150.92 ms` (`m=0`)** | `max_KKT <= 9.99e-07`, `max_orth <= 9.67e-14` |
| `[4]` `B8A` Hard-Trace Ensemble DFT (`10/10` solved) | `119.23 ms` (`214.0` evals) | `146.99 ms` (`213.0` iters) | **`68.19 ms` (`208.0` iters, `214.0` evals)** | `max |1^T z - 5| <= 1.78e-15` (`v0.2.1`), `max_KKT <= 1.00e-06` |

*(Tradeoff note: on cheap pure-Stiefel native callbacks like `B7B`, `m = 0` (`ABB`) can achieve lower native wall time than `m = 7` despite needing ~33% more evaluations because the `2m + 2` Cayley panel transport dominates callback cost; when objective/gradient evaluations are nontrivial, `m = 7` dominates in both evaluations and wall time.)*

### 5.5 Key `v0.2.1` Improvements, Certified Global Optima & Nonconvex Stationarity Scope

- **100% 40-Seed Convergence (`440/440`, incl. `110/110` on Virgin `Blind4`)**: Fixing the premature `FC_CONVERGED_FTOL` noise-floor exit lifted `B2A` from `16/30 (53%)` to **`40/40 (100%)`** (`585.0` med evals, `607.0` on virgin `blind4`) and `B5` from `26/30 (87%)` to **`40/40 (100%)`** (`1192.0` med evals, `1187.5` on virgin `blind4`), with `fortchuck` (`m = 7`) strictly beating `fortchuck-ABB` (`m = 0`) on all **11/11** problems (`p = 2.20e-70 (FC better)`).
- **Hard Linear Conservation on `B8A` & `B8B` (`80/80 = 100%` Solved, `max V_lin <= 2.96e-16`)**: Eliminating active-bound projection lift-off slashed `B8B` from `250.0` evals (`466.0` Holdout) down to **`36.0` evals (`37.0` on virgin `blind4`, `8.66 ms`, `max V_lin = 1.11e-16`)**—beating `fortchuck-ABB` (`64.0` evals, `p = 3.55e-08 (FC better)`) and outperforming dense `SciPy-SLSQP` (`37.5` evals, `488.74 ms`) in both evaluations (`p = 5.09e-10 (FC better)` across shared 40-seed `B3 + B8B`, `p = 5.69e-03 (FC better)` on virgin `blind4`) and ~56x wall time. `B8A` solves `40/40` in `151.5` evals (`150.5` virgin `blind4`, `max V_lin = 2.96e-16`).
- **Pure-Box Recovery on `B7A` (`40/40 = 100%`)**: Restoring rotation-invariant scalar BB2 seeding eliminated the `v0.2.0` `67.5 -> 126.0` regression, restoring `B7A` to **`68.0` evals (`67.5` on the shared 20-seed cohort, `68.0` on virgin `blind4`, `11.05 ms`)**—matching `v0.1.0` and `SciPy-L-BFGS-B` (`68.0` evals, `39.22 ms`) while running ~3.5x faster and beating `fortchuck-ABB` (`88.5` evals, `p = 3.51e-08 (FC better)`).
- **Certified Global Optimality (`B6` & `B7B`) vs Nonconvex Local Stationarity (`B1`, `B2B`)**: On `B6` ($f^\ast = -10.445509834730$) and `B7B` ($f^\ast = -66.949417348256$), `fortchuck` solves `40/40` each to `max |f - f*| <= 3.60e-12` (`2.67e-12` virgin `blind4`) and `<= 1.31e-12` (`9.38e-13` virgin `blind4`), respectively. On multi-modal nonconvex benchmarks (`B1`, `B2A`, `B2B`, `B5`), `fortchuck` is a local first-order KKT solver: different algorithms can converge to distinct stationary basins (e.g., on `B2B`, `fortchuck` converges in `104.5` evals to median `8.517` while `fortchuck-ABB` takes `735.5` evals to reach median `8.301`), so global optimality is claimed strictly on `B6` and `B7B`.

---

## 6. Repository Structure and Verification Gates

```text
fortchuck/
├── fpm.toml                          # Fortran Package Manager manifest (v0.2.0, F2018 + OpenMP)
├── pyproject.toml                    # Zero-copy Python package & pytest/benchmark metadata
├── CITATION.cff                      # Academic citation metadata (v0.2.0, CC0-1.0)
├── LICENSE                           # CC0-1.0 Universal Public Domain Dedication
├── README.md                         # Architecture, 7 Lean 4 proofs, quickstart & 30-seed benchmarks
├── app/
│   └── fortchuck_bench.f90           # Zero-FFI native Fortran 2018 CLI benchmark suite
├── docs/
│   ├── ARCHITECTURE_SPEC.md          # Authoritative v0.2.0 mathematical, F2018/C99 ABI & Lean spec
│   └── research/
│       ├── ARXIV_MANIFOLD_SURVEY.md              # Stiefel Cayley, vector transport & L-BFGS survey
│       ├── ARXIV_COUPLED_BOX_MANIFOLD_SURVEY.md  # Coupled Box x Stiefel & benchmark synthesis
│       └── TOOLCHAIN_AND_STANDARDS.md            # F2018, OpenMP, C99 ABI & Lean 4 standards
├── include/
│   └── fortchuck.h                   # ANSI C99 / C++11 interoperable header (8 entry points)
├── src/
│   ├── fortchuck_kinds.f90           # Precision kinds, status enums & 96B/88B bind(C) structs
│   ├── fortchuck_linalg.f90          # Cache-blocked 4x4/1x4 FMA gemm, 4-RHS LU, geqrf_pos & BLAS
│   ├── fortchuck_box.f90             # Box/SLBQP Dai-Fletcher projection, KKT, diag & breakpoints
│   ├── fortchuck_stiefel.f90         # Dual-regime Schur-Woodbury & Direct LU Cayley + panel transport
│   ├── fortchuck_lbfgs.f90           # Masked product two-loop, Rank-1 KKT nullspace & SPG directions
│   ├── fortchuck_solver.f90          # Unified stateless fc_solve / fc_solve_constrained & Hermite LS
│   ├── fortchuck_c_abi.f90           # Null-safe iso_c_binding entry points matching fortchuck.h
│   └── fortchuck.f90                 # Public Fortran 2018 facade module
├── test/
│   ├── test_linalg_and_stiefel.f90   # Tiled GEMM/LU, panel transport parity & Dai-Fletcher tests
│   ├── test_coupled_solver.f90       # Box, Stiefel, SLBQP, shift-invariance & FM-2/FM-13 guards
│   └── test_openmp_reentrancy.f90    # 64-instance / 16-thread mixed-mode OpenMP bit-reproducibility
├── python/
│   └── fortchuck/
│       ├── __init__.py               # Public Python API exports (v0.2.0)
│       ├── _native.py                # Shared library loader + 8-symbol ctypes ABI assertions
│       └── solver.py                 # Zero-copy minimize_box_stiefel(), project_linear_box() & KKT
├── tests/
│   ├── test_c_abi_and_python.py      # ctypes ABI layout, v0.2.0 workspace & linear_constraint tests
│   └── test_pathological_guards.py   # Clamped multiplier FM-2, corner release FM-13 & barrier guards
├── formalization/
│   ├── lean-toolchain                # leanprover/lean4:v4.34.0-rc2
│   ├── lakefile.lean                 # Lake build configuration
│   └── Fortchuck.lean                # 15 machine-checked theorems (2,074 lines, 0 sorry, 0 warnings)
├── benchmarks/
│   ├── problems.py                   # Canonical 11-instance benchmark suite (B1..B8B)
│   ├── referee.py                    # Pure-NumPy project_linear_box_ref, KKT referee & FD verifier
│   ├── baselines.py                  # Adapters for fortchuck, SciPy, pymanopt 2.2.1 & Alternating-BCD
│   └── run_blind_suite.py            # 40-seed 4-suite blind runner & directional Wilcoxon aggregator
├── results/
│   ├── blind_benchmark_summary.json  # Unified 3,520-run (40-seed x 11-prob x 8-solver, incl. virgin blind4) JSON
│   └── blind_benchmark_report.md     # Full 40-seed 8-solver Markdown report + progression & CLI telemetry
├── scripts/
│   ├── audit_gfm_math.py             # 22-rule GitHub KaTeX/GFM Markdown compliance linter
│   ├── build_shared.py               # Cross-platform fortchuck.dll / libfortchuck.so builder
│   └── verify_zero_alloc.py          # nm -u symbol gate verifying 0 heap/pack/IO symbols
└── .github/
    └── workflows/
        └── ci.yml                    # 4-job GitHub Actions CI verification matrix
```

### Synchronous Local Verification Commands

```bash
# 1. Native Fortran 2018 sanitizer, 16-thread OpenMP reentrancy & release CLI benchmark
fpm test --compiler gfortran --flag "-cpp -std=f2018 -O2 -g -Wall -Wextra -Werror -pedantic -Warray-temporaries -fimplicit-none -fcheck=all -fbacktrace -ffpe-trap=invalid,zero,overflow -fopenmp"
fpm run fortchuck-bench --compiler gfortran --flag "-cpp -std=f2018 -O3 -march=native -funroll-loops -fopenmp"

# 2. Build shared C-ABI library & verify zero heap allocations / array temporaries via nm -u
python scripts/build_shared.py
python scripts/verify_zero_alloc.py

# 3. Run Python ctypes FFI & pathological guard suite (54 tests)
pytest -v tests/

# 4. Verify central FD gradients across all 40 seeds & run blind 4-suite benchmark (or --smoke)
python -m benchmarks.referee --seeds 101..110
python -m benchmarks.referee --seeds 201..210
python -m benchmarks.referee --seeds 301..310
python -m benchmarks.referee --seeds 401..410
python benchmarks/run_blind_suite.py --smoke

# 5. Verify Lean 4 formal proofs (15 theorems, 0 sorry, 0 warnings) and 22-rule GFM/KaTeX math compliance
cd formalization && lake build && cd ..
python scripts/audit_gfm_math.py README.md results/blind_benchmark_report.md docs/ARCHITECTURE_SPEC.md docs/research/TOOLCHAIN_AND_STANDARDS.md docs/research/ARXIV_MANIFOLD_SURVEY.md docs/research/ARXIV_COUPLED_BOX_MANIFOLD_SURVEY.md
```

---

## 7. Citation

If you use `fortchuck` in academic research or production scientific computing, please cite:

```bibtex
@software{cohen2026fortchuck,
  author       = {Cohen, Noah},
  title        = {{fortchuck: Zero-Allocation Coupled Box/Simplex $\times$ Stiefel Product-Manifold Quasi-Newton Optimizer}},
  version      = {0.2.1},
  year         = {2026},
  license      = {CC0-1.0},
  url          = {https://github.com/sneed-and-feed/fortchuck}
}
```

---

## 8. License

Dedicated to the public domain under [**CC0 1.0 Universal**](LICENSE). No rights reserved.
