# fortchuck: Zero-Allocation Coupled Box × Stiefel Product-Manifold Quasi-Newton Optimizer

[![CI Matrix](https://github.com/sneed-and-feed/fortchuck/actions/workflows/ci.yml/badge.svg)](https://github.com/sneed-and-feed/fortchuck/actions/workflows/ci.yml)
[![Lean 4 Formalization](https://img.shields.io/badge/Lean_4-0_sorry_0_warnings-00f0ff.svg?style=flat-square)](formalization/Fortchuck.lean)
[![Fortran 2018 Zero-Alloc](https://img.shields.io/badge/Fortran_2018-0_heap_alloc_pure-b844ff.svg?style=flat-square)](scripts/verify_zero_alloc.py)
[![OpenMP Reentrancy](https://img.shields.io/badge/OpenMP-16--thread_bit--identical-ffb800.svg?style=flat-square)](test/test_openmp_reentrancy.f90)
[![License](https://img.shields.io/badge/License-CC0--1.0_Universal-lightgrey.svg?style=flat-square)](LICENSE)

A standalone, zero-heap-allocation, `100%` OpenMP-reentrant **Modern Fortran 2018 + ANSI C99 ABI + Zero-Copy Python (`ctypes`)** solver for smooth nonconvex optimization over heterogeneous products of bound-constrained Euclidean boxes and multiple Stiefel manifolds, paired with machine-checked **Lean 4** algebraic and variational proofs (`0 sorry`, `0` custom axioms, `0` compiler warnings) and a blind 20-seed (`10` primary + `10` out-of-sample holdout) referee benchmark suite.

---

## Table of Contents

- [1. Executive Overview](#1-executive-overview)
- [2. Coupled Product Geometry and Mathematical Architecture](#2-coupled-product-geometry-and-mathematical-architecture)
- [3. Lean 4 Formal Verification](#3-lean-4-formal-verification)
- [4. Quickstart: Fortran 2018, ANSI C99, and Zero-Copy Python](#4-quickstart-fortran-2018-ansi-c99-and-zero-copy-python)
- [5. Blind Multi-Seed Benchmark Suite and Out-of-Sample Holdout Evaluation](#5-blind-multi-seed-benchmark-suite-and-out-of-sample-holdout-evaluation)
- [6. Repository Structure and Verification Gates](#6-repository-structure-and-verification-gates)
- [7. Citation](#7-citation)
- [8. License](#8-license)

---

## 1. Executive Overview

`sneed-and-feed/fortchuck` (`v0.1.0`, `CC0-1.0` Public Domain) solves smooth coupled problems over arbitrary Cartesian products of box-constrained coordinates and orthogonal Stiefel frames:

```math
\min_{z \in \mathcal B, \; X_1, \dots, X_R} \; f(z, X_1, \dots, X_R)
```

where the feasible domain combines extended-real box bounds and $R$ Stiefel factors:

```math
\mathcal B = \left\lbrace z \in \mathbb R^d : l_i \le z_i \le u_i, \; i = 1, \dots, d \right\rbrace, \qquad X_r \in \mathrm{St}(n_r, p_r) = \left\lbrace X \in \mathbb R^{n_r \times p_r} : X^\top X = I_{p_r} \right\rbrace
```

with $1 \le p_r \le n_r$, extended-real bounds
$(l_i, u_i) \in [-\infty, +\infty) \times (-\infty, +\infty]$,
and $l_i \le u_i$ (including fixed variables `l_i == u_i` and unbounded sentinel limits).

| Optimization Regime | Configuration | Representative Scientific & Engineering Workloads |
| :--- | :---: | :--- |
| **Pure Box Limit** | $d \ge 1, \; R = 0$ | Active-bound nonlinear programming, non-negative least squares, bounded calibration |
| **Pure Multi-Stiefel Limit** | $d = 0, \; R \ge 1$ | Brockett subspace diagonalization, $\mathrm{SO}(n)$ Procrustes & multi-camera Wahba alignment |
| **Coupled Box × Multi-Stiefel** | $d \ge 1, \; R \ge 1$ | Ensemble Kohn--Sham DFT fractional occupations, sparse PCA, bounded-variance ML-CPC, ICA |

### Core Engineering Guarantees

1. **Strict Zero Heap Allocation (`nm -u` Verified)**: Every library kernel in `src/*.f90` executes exclusively inside caller-supplied `work(1:lwork)` and `iwork(1:liwork)` scratch buffers queried via `lwork = -1` or `fortchuck_workspace_query`. Automated symbol auditing ([`scripts/verify_zero_alloc.py`](scripts/verify_zero_alloc.py)) confirms **zero** `malloc`, `free`, `_gfortran_allocate*`, `_gfortran_internal_pack*`, or Fortran I/O/stop symbols across all compiled object files.
2. **Bit-Identical Multi-Threaded OpenMP Reentrancy**: Zero `save` variables, zero mutable module state, and `100% pure` computational kernels allow concurrent solves across `16+` OpenMP threads with `0.0` byte-level discrepancy against single-threaded execution ([`test/test_openmp_reentrancy.f90`](test/test_openmp_reentrancy.f90)).
3. **Zero-Copy Column-Major (`order="F"`) C99 & Python ABI**: Naturally aligned `fc_options_t` (`96` bytes) and `fc_result_t` (`88` bytes) structs interoperate directly across Fortran `iso_c_binding`, ANSI C99 ([`include/fortchuck.h`](include/fortchuck.h)), and NumPy `order="F"` buffers ([`python/fortchuck/`](python/fortchuck/)).

---

## 2. Coupled Product Geometry and Mathematical Architecture

Existing optimization toolboxes force practitioners to choose between smooth manifold solvers without native bound constraints (where `tanh`/`exp` variable warps drive gradients to zero at active bounds) and alternating block-coordinate schemes (which destroy joint quasi-Newton cross-curvature across tightly coupled blocks). `fortchuck` unifies projected active-set methods and Riemannian quasi-Newton geometry from first principles ([`docs/ARCHITECTURE_SPEC.md`](docs/ARCHITECTURE_SPEC.md)):

### 2.1 Safeguarded Bertsekas Active Band and Dynamic `FM-2` Subspace Masking

At iterate $p_k = (z_k, X_{1,k}, \dots, X_{R,k})$, `fortchuck` computes the overflow-safe 1-Lipschitz box KKT residual
$r_{\text{box}}(z_k) = z_k - P_{\mathcal B}(z_k - \nabla_z f(p_k))$
and partitions $\{1, \dots, d\}$ into fixed coordinates $\mathcal A_{\text{fixed}}$ (`l_i == u_i`),
safeguarded outward-pointing active-band coordinates $\mathcal A_{\epsilon_k}(z_k)$,
and free coordinates $\mathcal F_{\epsilon_k}(z_k)$.

When a box coordinate clamps at a bound (`z_{k,i} = l_i`) with a strictly positive Lagrange multiplier $\lambda_i^\ast = \nabla_{z_i} f(p^\ast) > 0$,
historical secant vectors carry nonzero box gradient differences coupled to Stiefel steps. Applying an unmasked product L-BFGS operator injects permanent active-multiplier bias into the Stiefel search direction (`FM-2`). Inside every two-loop recursion, `fortchuck` applies dynamic virtual masking $P_{\text{free},k}$ across stored secant slots without copying memory, retaining slot $j$ if and only if its free-subspace curvature satisfies:

```math
\tilde\sigma_j = \sum_{i \in \mathcal F_{\epsilon_k}} s_{j,\mathrm{box},i} y_{j,\mathrm{box},i} + \sigma_{j,\mathrm{St}}^{sy} \ge 10^{-10} \sqrt{\tilde\nu_{s,j} \tilde\nu_{y,j}}
```

paired with cosine-guarded persistent per-block Barzilai--Borwein (`BB2`) center scaling across the $R + 1$ product blocks (`FM-5`).

### 2.2 Dual-Regime Exact Cayley Retraction and Isometric Transport

Under the canonical Stiefel metric ($\beta = 1/2$), each tangent step $\eta_r \in T_{X_r}\mathrm{St}(n_r, p_r)$
defines the skew-symmetric Lie generator
$W_{\eta_r} = \tilde\eta_r X_r^\top - X_r \tilde\eta_r^\top \in \mathfrak{so}(n_r)$,
where $\tilde\eta_r = \eta_r - \tfrac{1}{2} X_r (X_r^\top \eta_r)$. `fortchuck` dispatches automatically per factor to minimize dense linear algebra complexity:

- **Branch A** ($2 p_r \lt n_r$, Tall Stiefel — Schur-Woodbury):
  Decomposes $A_r = X_r^\top \eta_r \in \mathfrak{so}(p_r)$
  and $K_r = \eta_r - X_r A_r$,
  solving the strictly dissipative $p_r \times p_r$ Schur complement:

```math
J_r(\tau) = I_{p_r} - \frac{\tau}{2} A_r + \frac{\tau^2}{4} K_r^\top K_r, \qquad \mathcal R_{X_r}^{\mathrm{Cay}}(\tau \eta_r) = (2 X_r + \tau K_r) J_r(\tau)^{-1} - X_r
```

- **Branch B** ($2 p_r \ge n_r$, Wide/Square Orthogonal Group — Direct Skew-Resolvent LU):
  Factors $N_r(\tau) = I_{n_r} - \tfrac{\tau}{2} W_{\eta_r}$ directly
  in $\mathbb R^{n_r \times n_r}$ ($3.4\times$ fewer LU flops than double-width Woodbury at `p_r == n_r`):

```math
M_r(\tau) = \left(I_{n_r} - \frac{\tau}{2} W_{\eta_r}\right)^{-1}, \qquad Q_r(\tau) = 2 M_r(\tau) - I_{n_r}, \qquad \mathcal R_{X_r}^{\mathrm{Cay}}(\tau \eta_r) = Q_r(\tau) X_r
```

In both regimes, the accepted LU factorization is reused across all $m$ stored L-BFGS history pairs to apply exact isometric Cayley vector transport $\mathcal T_{\tau \eta_r}^{\mathrm{Cay}}(\zeta_r) = Q_r(\tau) \zeta_r$,
preserving canonical inner products to machine precision while pre-callback positive-diagonal Householder QR (`fc_geqrf_pos`, $R_{ii} > 0$) prevents floating-point drift from exceeding $10^{-13}$. Globalization combines non-jamming generalized Arc-Armijo chord projection with Zhang--Hager nonmonotone reference tracking (`zh_eta` auto `0.85`), eliminating curved-valley Maratos unit-step rejections while guaranteeing global KKT stationarity.

---

## 3. Lean 4 Formal Verification

All core algebraic identities, branch equivalences, and two-metric variational descent guarantees are machine-checked in [`formalization/Fortchuck.lean`](formalization/Fortchuck.lean) (`leanprover/lean4:v4.34.0-rc2` + Mathlib4, `0 sorry`, `0` custom axioms, `0` warnings):

| Theorem Identifier | Mathematical Guarantee | Machine-Checked Statement |
| :--- | :--- | :--- |
| `skew_generator_properties` | $\mathfrak{so}(n)$ Lie Generator & Tangent Reproduction | $X^\top X = I_p \land X^\top \xi + \xi^\top X = 0 \implies W_\xi^\top = -W_\xi \land W_\xi X = \xi$ |
| `cayley_rotor_orthogonality_and_isometry` | Two-Sided Rotor Orthogonality & Canonical Isometry | $Q = 2 M - I_n, \; Q^\top Q = Q Q^\top = I_n, \; \langle Q \zeta_1, Q \zeta_2 \rangle_{Q X} = \langle \zeta_1, \zeta_2 \rangle_X$ |
| `schur_woodbury_cayley_orthogonality` | Rank $2p$ Schur-Woodbury Exact Feasibility | $Y = (2 X + \tau K) J(\tau)^{-1} - X \implies Y^\top Y = I_p$ |
| `cayley_branch_a_eq_branch_b` | Algebraic Equivalence of Branch A and Branch B | $Q(\tau) X = (2 X + \tau K) J(\tau)^{-1} - X$ identically for all $(n, p, \tau)$ |
| `coupled_two_metric_strict_descent` | Unconditional Joint Strict Descent & KKT Characterization | $\phi'(0^+) \le -\gamma_{\min} \lVert g_{\mathcal A^{\mathrm{int}}} \rVert_2^2 - \lambda_{\min} \lVert \tilde g \rVert^2 \le 0$, with $\phi'(0^+) = 0 \iff \mathrm{KKT}(z, X)$ |

---

## 4. Quickstart: Fortran 2018, ANSI C99, and Zero-Copy Python

### 4.1 Zero-Copy Python Interface (`minimize_box_stiefel`)

```bash
# Build native shared library (.dll / .so) and install editable package inside a virtualenv
python scripts/build_shared.py
pip install -e ".[dev,bench]"
```

```python
import numpy as np
from fortchuck import FCOptions, FortchuckWorkspace, minimize_box_stiefel

n, p = 50, 4
A = np.diag(np.linspace(1.0, 10.0, n))
bounds = (np.full(p, 0.2), np.full(p, 3.0))

def fun_and_grad(z: np.ndarray, X_list: list[np.ndarray]):
    X = X_list[0]  # Zero-copy read-only Fortran-contiguous (n, p) view
    AX = A @ X
    rayleigh = np.sum(X * AX, axis=0)
    f = float(np.dot(rayleigh, z) + 0.5 * np.sum((z - 1.5) ** 2))
    gz = rayleigh + (z - 1.5)
    GX = np.asfortranarray(2.0 * AX * z[None, :])
    return f, gz, [GX]

z0 = np.ones(p, dtype=np.float64)
X0 = np.asfortranarray(np.eye(n, p, dtype=np.float64))
ws = FortchuckWorkspace(d=p, n_list=[n], p_list=[p], memory=7)

res = minimize_box_stiefel(
    fun_and_grad,
    z0=z0,
    bounds=bounds,
    X0_list=[X0],
    options=FCOptions(gtol=1e-8, max_eval=2000),
    workspace=ws,
)
assert res.converged and res.kkt_residual <= 1e-8 and res.orth_error <= 1e-13
```

### 4.2 Modern Fortran 2018 (`fpm`)

Add `fortchuck` to `fpm.toml` and invoke `fc_workspace_query` + `fc_solve` with caller-owned `work` and `iwork` arrays:

```fortran
program demo_fortchuck
  use, intrinsic :: iso_c_binding, only: c_int32_t, c_double, c_ptr, c_null_ptr
  use fortchuck, only: dp, i32, i64, fc_options_t, fc_result_t, &
    fc_init_default_options, fc_workspace_query, fc_solve, FC_CONVERGED_GRAD
  implicit none (type, external)

  integer(i32), parameter :: d = 2_i32, R = 1_i32
  integer(i32)            :: n_arr(1) = [4_i32], p_arr(1) = [2_i32], status
  integer(i64)            :: lwork, liwork
  real(dp)                :: z(2) = [0.5_dp, 0.5_dp], l(2) = [0.1_dp, 0.1_dp], u(2) = [2.0_dp, 2.0_dp]
  real(dp)                :: X(8) = [1.0_dp, 0.0_dp, 0.0_dp, 0.0_dp, 0.0_dp, 1.0_dp, 0.0_dp, 0.0_dp]
  real(dp)                :: work(4096)
  integer(i32)            :: iwork(256)
  type(fc_options_t)      :: opts
  type(fc_result_t)       :: res

  call fc_init_default_options(opts)
  call fc_workspace_query(d, R, n_arr, p_arr, opts%memory, lwork, liwork, status)
  call fc_solve(d, R, n_arr, p_arr, z, l, u, X, eval_demo, c_null_ptr, &
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

    f_out = 0.5_dp * ((z_in(1) - 0.8_dp)**2 + (z_in(2) - 1.5_dp)**2)
    gz_out(1) = z_in(1) - 0.8_dp
    gz_out(2) = z_in(2) - 1.5_dp
    do k = 1_i32, 8_i32
      f_out     = f_out + 0.5_dp * real(k, dp) * X_in(k)**2
      GX_out(k) = real(k, dp) * X_in(k)
    end do
    rc = 0_i32
  end function eval_demo
end program demo_fortchuck
```

### 4.3 ANSI C99 / C++11 (`include/fortchuck.h`)

```c
#include "fortchuck.h"
#include <stdlib.h>

/* Query exact caller-owned buffer sizes and run zero-allocation solve */
int run_c99_example(fc_eval_fg_cb cb, void *udata) {
    int32_t n_arr[1] = {30}, p_arr[1] = {5};
    double z[5] = {1.0, 1.0, 1.0, 1.0, 1.0};
    double l[5] = {0.0, 0.0, 0.0, 0.0, 0.0};
    double u[5] = {2.0, 2.0, 2.0, 2.0, 2.0};
    double X[150] = {0.0};
    for (int j = 0; j < 5; ++j) X[j * 30 + j] = 1.0; /* Column-major I_{30x5} */

    fc_options_t opts;
    fortchuck_default_options(&opts);

    int64_t lwork = 0, liwork = 0;
    fortchuck_workspace_query(5, 1, n_arr, p_arr, opts.memory, &lwork, &liwork);

    double  *work  = (double *)malloc((size_t)lwork * sizeof(double));
    int32_t *iwork = (int32_t *)malloc((size_t)liwork * sizeof(int32_t));
    fc_result_t res;

    int32_t status = fortchuck_solve(
        5, 1, n_arr, p_arr, z, l, u, X, cb, udata, &opts, work, lwork, iwork, liwork, &res
    );
    free(iwork);
    free(work);
    return (int)status;
}
```

---

## 5. Blind Multi-Seed Benchmark Suite and Out-of-Sample Holdout Evaluation

To prevent hyperparameter overfitting, [`benchmarks/`](benchmarks/) was constructed behind a strict blind firewall from `src/*.f90`. Every solver is initialized from bit-identical `PCG64` feasible points across `10` deterministic primary seeds (`101..110`) and `10` fresh out-of-sample holdout seeds (`201..210`). All `9` analytical Euclidean gradients are pre-verified against central finite differences to relative error $\le 1.48 \times 10^{-8} \le 10^{-7}$ ([`benchmarks/referee.py`](benchmarks/referee.py)), and `fortchuck` is evaluated using **strictly frozen out-of-the-box** `FCOptions(gtol=1e-6, max_eval=5000)` (`memory = 7`, `max_iter = 15000`, `zh_eta` auto `0.85`, canonical metric, cautious skip) across every problem and seed.

A run is certified **Solved** by the independent external referee if and only if:

```math
\mathcal E_{\mathrm{KKT,ref}}(z, X) = \max\!\left(\left\lVert z - P_{[l,u]}\bigl(z - \nabla_z f(z, X)\bigr) \right\rVert_\infty, \; \max_{1 \le r \le R} \left\lVert G_r - X_r\,\mathrm{sym}(X_r^\top G_r) \right\rVert_{\mathrm F}\right) \le 10^{-6}
```

and $\max(\mathcal V_{\mathrm{box}}, \mathcal V_{\mathrm{orth}}) \le 10^{-8}$. Full per-seed JSON data and per-problem breakdowns are archived in [`results/blind_benchmark_summary.json`](results/blind_benchmark_summary.json) and [`results/blind_benchmark_report.md`](results/blind_benchmark_report.md).

### 5.1 Canonical 9-Instance Benchmark Matrix

| ID | Benchmark Problem | Product Manifold Domain | Primary Coupling & Numerical Stress Tested |
| :--- | :--- | :--- | :--- |
| `B1` | Ensemble Kohn--Sham DFT Fractional Occupations | $[0, 1]^{10} \times \mathrm{St}(120, 10)$ | Branch A ($2p \lt n$), small-gap metallic orbital-occupation sloshing |
| `B2A` | Amplitude-Limited Blind Source Separation (ICA) | $[-2, 2]^{5 \times 150} \times \mathrm{St}(20, 5)$ | Branch A ($d = 750$), nonconvex $\ln\cosh$ with $d \gg np$ active bounds |
| `B2B` | Smooth Penalty-Split Non-Negative Sparse PCA | $[0, 5]^{120 \times 6} \times \mathrm{St}(120, 6)$ | Branch A ($d = 720$), $\sim 80\%$ clamped $Z = 0$ with large multipliers (`FM-2`) |
| `B3` | Multi-Camera Anisotropic Bounded-Scale Rig Wahba | $\mathcal B^{15} \times \mathrm{St}(3, 3)^4$ | Branch B $(\mathrm{SO}(3)^4)$, $R = 4$ multi-factor $\kappa(W) = 10^3$ |
| `B4` | Bounded-Variance ML Common Principal Components | $[0.05, 20]^{150} \times \mathrm{St}(30, 30)$ | Branch B $(\mathrm{O}(30))$, log-variance barrier, $1.6 \times 10^5$ Hessian scale ratio |
| `B5` | Bounded Factor Correlation Matrix Calibration | $\mathcal B^{110} \times \mathrm{St}(100, 10)$ | Branch A, quartic factor valley + clamped idiosyncratic variances |
| `B6` | Box-Coupled Brockett with Certified Ky Fan QP Truth | $[0.1, 5]^{10} \times \mathrm{St}(150, 10)$ | Branch A, $\kappa(M) = 10^4$, certified analytical global optimum $f^\ast$ |
| `B7A` | Pure-Box Active-Bound Chained Rosenbrock + Quadratic | $[l, u]^{500} \; (R = 0)$ | Pure Box ($5$ fixed $l_i = u_i$, $\sim 60\%$ active bounds) vs `SciPy L-BFGS-B` |
| `B7B` | Pure-Stiefel Tall Brockett & Orthogonal Procrustes | $\mathrm{St}(200, 8) \times \mathrm{St}(24, 24) \; (d = 0)$ | Pure Stiefel Branch A & Branch B $(\mathrm{SO}(24))$ vs native `pymanopt` |

### 5.2 Primary Blind Suite (`Seeds 101..110`, `90` Problem-Seed Pairs, Frozen Default `FCOptions`)

| Solver | Evaluated Runs | Solved Runs | Solve Rate | Penalized Evals Med (IQR) | Wall Time ms Med (IQR) | Median KKT Ref | Median Orth Viol | Wilcoxon Evals p (2-sided) | Wilcoxon KKT p (2-sided) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `fortchuck` | 90 | 88 | **97.8%** | 602.0 (1662.0) | 619.29 (1437.09) | `8.00e-07` | `2.67e-14` | `—` | `—` |
| `fortchuck-ABB` | 90 | 59 | 65.6% | 3945.5 (4518.5) | 605.28 (1662.13) | `8.89e-07` | `3.88e-14` | `3.73e-16` | `1.18e-07` |
| `Alternating-BCD` | 90 | 34 | 37.8% | 5000.0 (4477.8) | 3006.91 (3210.31) | `1.63e-04` | `1.01e-15` | `7.82e-14` | `2.01e-15` |
| `pymanopt-Warped-CG` | 90 | 23 | 25.6% | 5000.0 (36.8) | 4757.55 (3771.80) | `2.77e-03` | `1.13e-15` | `3.73e-16` | `2.98e-16` |
| `pymanopt-TrustRegions` | 90 | 24 | 26.7% | 5000.0 (610.2) | 3108.43 (1997.64) | `4.45e-03` | `4.08e-14` | `3.73e-16` | `2.61e-13` |
| `SciPy-L-BFGS-B` | 90 | 10 | 11.1% | 5000.0 (0.0) | 2630.73 (3772.81) | `7.97e-04` | `5.56e-06` | `1.44e-15` | `3.41e-16` |
| `SciPy-SLSQP` | 10 (`N_tot <= 450`) | 10 | 100.0% | 75.0 (10.5) | 71.65 (10.18) | `5.69e-07` | `6.82e-14` | `1.95e-03` | `3.22e-01` |
| `SciPy-trust-constr` | 10 (`N_tot <= 450`) | 10 | 100.0% | 90.0 (17.8) | 466.20 (123.67) | `7.85e-07` | `2.46e-15` | `1.95e-03` | `1.95e-03` |

### 5.3 Out-of-Sample Holdout Suite (`Seeds 201..210`, `90` Pairs) and Pooled `180`-Run Summary

| Solver | Holdout Solved (`201..210`) | Holdout Pen. Evals Med (IQR) | Holdout Wilcoxon Evals p | Pooled Solved (`101..110` + `201..210`) | Pooled Pen. Evals Med (IQR) | Pooled Wall ms Med (IQR) | Pooled Wilcoxon Evals p |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `fortchuck` | **84/90 (93.3%)** | 576.5 (1668.8) | `—` | **172/180 (95.6%)** | 588.0 (1676.2) | 619.29 (1390.41) | `—` |
| `fortchuck-ABB` | 57/90 (63.3%) | 3869.5 (4422.2) | `1.06e-14` | 116/180 (64.4%) | 3869.5 (4462.5) | 672.76 (1650.08) | `3.64e-29` |
| `Alternating-BCD` | 33/90 (36.7%) | 5000.0 (4068.5) | `6.01e-14` | 67/180 (37.2%) | 5000.0 (4218.5) | 2998.90 (3109.00) | `3.41e-26` |
| `pymanopt-Warped-CG` | 25/90 (27.8%) | 5000.0 (2697.2) | `1.71e-15` | 48/180 (26.7%) | 5000.0 (797.5) | 4713.49 (3359.76) | `5.61e-30` |
| `pymanopt-TrustRegions` | 27/90 (30.0%) | 5000.0 (2945.5) | `1.71e-15` | 51/180 (28.3%) | 5000.0 (2321.5) | 3048.72 (1900.33) | `5.61e-30` |
| `SciPy-L-BFGS-B` | 10/90 (11.1%) | 5000.0 (0.0) | `7.49e-15` | 20/180 (11.1%) | 5000.0 (0.0) | 2551.18 (3702.90) | `7.97e-29` |
| `SciPy-SLSQP` | 10/10 (100.0%) | 86.5 (18.5) | `1.95e-03` | 20/20 (100.0%) | 80.0 (17.5) | 77.15 (20.23) | `8.66e-05` |
| `SciPy-trust-constr` | 10/10 (100.0%) | 95.0 (28.2) | `1.95e-03` | 20/20 (100.0%) | 92.0 (19.8) | 473.29 (154.96) | `8.82e-05` |

### 5.4 Per-Problem and Certified Global Optimality Highlights

- **Certified Global Optimality (`B6` Ky Fan QP & `B7B` Procrustes)**: On `B6` ($f^\ast = -10.445509834730$, $\kappa(M) = 10^4$), `fortchuck` solves `10/10 (100%)` on both primary (`1912.0` evals) and holdout (`1863.0` evals) suites to median global objective gap `1.91e-12` (primary, max `4.46e-12`) and `1.39e-12` (holdout, max `2.54e-12`), whereas **every** external baseline solves `0/10` (`|f - f_star| >= 7.27e-05`, over seven orders of magnitude larger). On `B7B` ($f^\ast = -66.949417348256$), `fortchuck` solves `10/10` on both suites (`499.5` and `520.0` evals, median gap `6.32e-13` and `8.24e-13`), requiring `2.1x` fewer evaluations than native `pymanopt` CG (`1062.5` and `1096.5` evals) and `3.4x` fewer than `pymanopt-TrustRegions` (`1700.5` and `1737.5` evals, `p = 1.95e-03`).
- **Tightly Coupled DFT, CPC & Quartic Factor Calibration (`B1`, `B4`, `B5`)**: Joint masked quasi-Newton cross-curvature solves `20/20 (100%)` on `B1` (`2161.0` / `1963.5` evals vs `0/20` for `fortchuck-ABB` and `pymanopt`), `20/20 (100%)` on `B4` (`1765.5` / `1749.5` evals across a $1.6 \times 10^5$ Hessian scale ratio vs `0/20` for all external baselines), and `17/20 (85%)` on `B5` (`10/10` primary, `7/10` holdout vs `0/20` for all baselines).
- **Active-Multiplier Clamped Sparse PCA (`B2B`, `N_tot = 1440`)**: Dynamic `FM-2` free-subspace masking solves `20/20 (100%)` in median `126.0` (primary, `47.19 ms`) and `100.5` (holdout, `36.36 ms`) evaluations, achieving a `7.0x–14.4x` evaluation speedup over `fortchuck-ABB` (`887.0` / `1447.0` evals, `p = 1.95e-03`) and outperforming `Alternating-BCD` (`0/20`), `pymanopt-Warped-CG` (`9/20`), and `pymanopt-TrustRegions` (`11/20`).
- **Multi-Camera Rig Self-Calibration (`B3`, `R = 4`)** ($\mathrm{SO}(3)^4$, $\kappa = 10^3$) **and Pure-Box Limit (`B7A`, `d = 500`)**: On `B3`, `fortchuck` solves `20/20 (100%)` in `42.0` / `46.0` evaluations (`12.27 ms` / `11.14 ms`), outperforming `Alternating-BCD` (`319.5` / `316.5` evals), `pymanopt-Warped-CG` (`874.5` / `854.0` evals), `SciPy-SLSQP` (`75.0` / `86.5` evals), and `SciPy-trust-constr` (`90.0` / `95.0` evals, `466.20 ms`). On `B7A`, `fortchuck` matches native Fortran `SciPy-L-BFGS-B` evaluation counts (`67.5` vs `68.0` / `66.5` evals, `20/20` solved) while executing `4.4x–4.5x` faster in wall-clock latency (`8.75 ms` / `7.96 ms` vs `38.58 ms` / `35.86 ms`).

---

## 6. Repository Structure and Verification Gates

```text
fortchuck/
├── fpm.toml                          # Fortran Package Manager manifest (F2018 + OpenMP)
├── pyproject.toml                    # Zero-copy Python package & pytest/benchmark metadata
├── CITATION.cff                      # Academic citation metadata (CC0-1.0)
├── LICENSE                           # CC0-1.0 Universal Public Domain Dedication
├── README.md                         # Architecture, Lean 4 proofs, quickstart & blind benchmarks
├── docs/
│   ├── ARCHITECTURE_SPEC.md          # Complete mathematical, Fortran/C99 ABI & verification spec
│   └── research/
│       ├── ARXIV_MANIFOLD_SURVEY.md              # Stiefel Cayley, vector transport & L-BFGS survey
│       ├── ARXIV_COUPLED_BOX_MANIFOLD_SURVEY.md  # Coupled Box x Stiefel & benchmark synthesis
│       └── TOOLCHAIN_AND_STANDARDS.md            # F2018, OpenMP, C99 ABI & Lean 4 standards
├── include/
│   └── fortchuck.h                   # ANSI C99 / C++11 interoperable header
├── src/
│   ├── fortchuck_kinds.f90           # Precision kinds, status enums & bind(C) structs
│   ├── fortchuck_linalg.f90          # Pure F2018 gemm, syrk, LU w/ pivot guard, geqrf_pos
│   ├── fortchuck_box.f90             # Overflow-safe box projection, KKT & Bertsekas active band
│   ├── fortchuck_stiefel.f90         # Dual-regime Schur-Woodbury & Direct LU Cayley + transport
│   ├── fortchuck_lbfgs.f90           # Dynamic free-subspace masked two-loop & block-BB2 ring
│   ├── fortchuck_solver.f90          # Deterministic workspace slicer & thread-safe fc_solve loop
│   ├── fortchuck_c_abi.f90           # Null-safe iso_c_binding entry points matching fortchuck.h
│   └── fortchuck.f90                 # Public Fortran 2018 facade module
├── test/
│   ├── test_linalg_and_stiefel.f90   # LU, positive-R QR, and Branch A == Branch B parity tests
│   ├── test_coupled_solver.f90       # Pure-box, pure-Stiefel, multi-factor KKT & FM-2 guards
│   └── test_openmp_reentrancy.f90    # 16-thread OpenMP parallel multi-start bit-reproducibility
├── python/
│   └── fortchuck/
│       ├── __init__.py               # Public Python API exports
│       ├── _native.py                # Shared library loader + ctypes struct/offset assertions
│       └── solver.py                 # Zero-copy order="F" minimize_box_stiefel() & workspace
├── tests/
│   ├── test_c_abi_and_python.py      # ctypes ABI layout, workspace query & zero-copy tests
│   └── test_pathological_guards.py   # Clamped multiplier FM-2 & callback barrier recovery tests
├── formalization/
│   ├── lean-toolchain                # leanprover/lean4:v4.34.0-rc2
│   ├── lakefile.lean                 # Lake build configuration
│   └── Fortchuck.lean                # Machine-checked proofs (0 sorry, 0 warnings)
├── benchmarks/
│   ├── problems.py                   # Canonical 9-instance benchmark suite (B1..B7B)
│   ├── referee.py                    # Independent KKT/feasibility referee & central FD verifier
│   ├── baselines.py                  # Fair wrappers for SciPy, pymanopt 2.2.1 & Alternating-BCD
│   └── run_blind_suite.py            # Multi-seed blind runner & Wilcoxon statistical aggregator
├── results/
│   ├── blind_benchmark_summary.json  # Machine-readable primary + holdout blind archive
│   └── blind_benchmark_report.md     # Full primary + holdout Markdown report
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
# 1. Native Fortran 2018 sanitizer & 16-thread OpenMP reentrancy suite
fpm test --compiler gfortran --flag "-std=f2018 -O2 -g -Wall -Wextra -Werror -pedantic -Warray-temporaries -fimplicit-none -fcheck=all -fbacktrace -ffpe-trap=invalid,zero,overflow -fopenmp"

# 2. Build shared C-ABI library & verify zero heap allocations / array temporaries via nm -u
python scripts/build_shared.py
python scripts/verify_zero_alloc.py

# 3. Run Python ctypes FFI & pathological guard suite
pytest -v tests/

# 4. Verify central finite-difference benchmark gradients & run blind + holdout benchmark suite
python -m benchmarks.referee --seeds 101..110
python -m benchmarks.referee --seeds 201..210
python benchmarks/run_blind_suite.py

# 5. Verify Lean 4 formal proofs (0 sorry, 0 warnings) and 22-rule GFM/KaTeX math compliance across all 6 docs
cd formalization && lake build && cd ..
python scripts/audit_gfm_math.py README.md results/blind_benchmark_report.md docs/ARCHITECTURE_SPEC.md docs/research/TOOLCHAIN_AND_STANDARDS.md docs/research/ARXIV_MANIFOLD_SURVEY.md docs/research/ARXIV_COUPLED_BOX_MANIFOLD_SURVEY.md
```

---

## 7. Citation

If you use `fortchuck` in academic research or production scientific computing, please cite:

```bibtex
@software{cohen2026fortchuck,
  author       = {Cohen, Noah},
  title        = {{fortchuck: Zero-Allocation Coupled Box $\times$ Stiefel Product-Manifold Quasi-Newton Optimizer}},
  version      = {0.1.0},
  year         = {2026},
  license      = {CC0-1.0},
  url          = {https://github.com/sneed-and-feed/fortchuck}
}
```

---

## 8. License

Dedicated to the public domain under [**CC0 1.0 Universal**](LICENSE). No rights reserved.
