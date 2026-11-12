# fortchuck: Zero-Allocation Coupled Box/Simplex × Stiefel Product-Manifold Quasi-Newton Optimizer

[![CI Matrix](https://github.com/sneed-and-feed/fortchuck/actions/workflows/ci.yml/badge.svg)](https://github.com/sneed-and-feed/fortchuck/actions/workflows/ci.yml)
[![Lean 4 Formalization](https://img.shields.io/badge/Lean_4-7_theorems_0_sorry-00f0ff.svg?style=flat-square)](formalization/Fortchuck.lean)
[![Fortran 2018 Zero-Alloc](https://img.shields.io/badge/Fortran_2018-0_heap_alloc_pure-b844ff.svg?style=flat-square)](scripts/verify_zero_alloc.py)
[![OpenMP Reentrancy](https://img.shields.io/badge/OpenMP-16--thread_bit--identical-ffb800.svg?style=flat-square)](test/test_openmp_reentrancy.f90)
[![License](https://img.shields.io/badge/License-CC0--1.0_Universal-lightgrey.svg?style=flat-square)](LICENSE)

A standalone, zero-heap-allocation, `100%` OpenMP-reentrant **Modern Fortran 2018 + ANSI C99 ABI + Zero-Copy Python (`ctypes`)** solver (`v0.2.0`) for smooth nonconvex optimization over heterogeneous products of bound-constrained Euclidean boxes, singly-linearly-constrained simplices/knapsacks, and multiple Stiefel manifolds, paired with **7 machine-checked Lean 4** algebraic and variational theorems (`0 sorry`, `0` custom axioms, `0` compiler warnings) and a strictly blind 30-seed (`2,640`-run) three-suite referee benchmark archive.

---

## Table of Contents

- [1. Executive Overview](#1-executive-overview)
- [2. Coupled Product Geometry and v0.2.0 Four-Pillar Architecture](#2-coupled-product-geometry-and-v020-four-pillar-architecture)
- [3. Lean 4 Formal Verification](#3-lean-4-formal-verification)
- [4. Quickstart: Zero-Copy Python, Modern Fortran 2018, and ANSI C99](#4-quickstart-zero-copy-python-modern-fortran-2018-and-ansi-c99)
- [5. Blind 30-Seed Benchmark Suite and Native CLI Telemetry](#5-blind-30-seed-benchmark-suite-and-native-cli-telemetry)
- [6. Repository Structure and Verification Gates](#6-repository-structure-and-verification-gates)
- [7. Citation](#7-citation)
- [8. License](#8-license)

---

## 1. Executive Overview

`sneed-and-feed/fortchuck` (`v0.2.0`, `CC0-1.0` Public Domain) solves smooth coupled problems over arbitrary Cartesian products of box- or simplex/knapsack-constrained coordinates and orthogonal Stiefel frames:

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

### 2.1 Safeguarded Bertsekas Active Band, Coordinatewise `diag_box`, and Hermite Arc Interpolation

At iterate $p_k = (z_k, X_{1,k}, \dots, X_{R,k})$, `fortchuck` partitions $\{1, \dots, d\}$ into fixed coordinates (`l_i == u_i`),
safeguarded outward-pointing active-band coordinates $\mathcal A_{\epsilon_k}(z_k)$,
and free coordinates $\mathcal F_{\epsilon_k}(z_k)$. Inside every L-BFGS two-loop recursion, dynamic virtual masking retains secant slot $j$ if and only if its free-subspace curvature satisfies:

```math
\tilde\sigma_j = \sum_{i \in \mathcal F_{\epsilon_k}} s_{j,\mathrm{box},i} y_{j,\mathrm{box},i} + \sigma_{j,\mathrm{St}}^{sy} \ge 10^{-10} \sqrt{\tilde\nu_{s,j} \tilde\nu_{y,j}}
```

eliminating clamped-multiplier cross-contamination (`FM-2`). In `v0.2.0`, free Euclidean coordinates are preconditioned by a safeguarded coordinatewise secant diagonal `diag_box(1:d)` clamped around the persistent block Barzilai--Borwein scalar $\gamma_{\mathrm{box}}$, while backtracking combines exact free-wall breakpoint detection (`fc_box_breakpoint_cap`) with safeguarded Quadratic and Cubic Hermite interpolation over $[0.1\tau, \, 0.5\tau]$ along the Zhang--Hager nonmonotone reference arc.

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

which is provably invariant under arbitrary gauge shifts $g_z \mapsto g_z + \mu a$ and strictly descending whenever $\tilde g \notin \mathrm{span}(\tilde a)$,
paired with a non-jamming isotropic Spectral Projected Gradient (`SPG`) fallback $d_z^{\mathrm{SPG}} = P_{\mathcal C}(z_k - \alpha_k \nabla_z f) - z_k$ that releases simplex corners in a single step (`FM-13`).

---

## 3. Lean 4 Formal Verification

All 7 core algebraic identities, branch equivalences, variational projection inequalities, and KKT nullspace quasi-Newton descent theorems are machine-checked in [`formalization/Fortchuck.lean`](formalization/Fortchuck.lean) (`leanprover/lean4:v4.34.0-rc2` + Mathlib4, `1,090` lines, `0 sorry`, `0` custom axioms, `0` warnings):

| # | Theorem Identifier | Mathematical Guarantee | Machine-Checked Statement |
| :---: | :--- | :--- | :--- |
| 1 | `skew_generator_properties` | $\mathfrak{so}(n)$ Lie Generator & Tangent Reproduction | $X^\top X = I_p \land X^\top \xi + \xi^\top X = 0 \implies W_\xi^\top = -W_\xi \land W_\xi X = \xi$ |
| 2 | `cayley_rotor_orthogonality_and_isometry` | Two-Sided Rotor Orthogonality & Canonical Isometry | $Q = 2 M - I_n, \; Q^\top Q = Q Q^\top = I_n, \; \langle Q \zeta_1, Q \zeta_2 \rangle_{Q X} = \langle \zeta_1, \zeta_2 \rangle_X$ |
| 3 | `schur_woodbury_cayley_orthogonality` | Rank $2p$ Schur-Woodbury Exact Feasibility | $Y = (2 X + \tau K) J(\tau)^{-1} - X \implies Y^\top Y = I_p$ |
| 4 | `cayley_branch_a_eq_branch_b` | Algebraic Equivalence of Branch A and Branch B | $Q(\tau) X = (2 X + \tau K) J(\tau)^{-1} - X$ identically for all $(n, p, \tau)$ |
| 5 | `coupled_two_metric_strict_descent` | Unconditional Box × Stiefel Strict Descent & KKT | $\phi'(0^+) \le -\gamma_{\min} \lVert g_{\mathcal A^{\mathrm{int}}} \rVert_2^2 - \lambda_{\min} \lVert \tilde g \rVert^2 \le 0$, with $\phi'(0^+) = 0 \iff \mathrm{KKT}(z, X)$ |
| 6 | `simplex_box_projection_variational` | Variational `SLBQP` Optimality & `SPG` Tangency/Descent | $\langle y - z^\ast, z - z^\ast \rangle \le 0, \; a^\top d_{\mathrm{spg}} = 0, \; \langle g, d_{\mathrm{spg}} \rangle \le -\gamma^{-1} \lVert d_{\mathrm{spg}} \rVert_2^2$ |
| 7 | `rank1_kkt_nullspace_quasinewton_descent` | Rank-1 KKT Nullspace Tangency, Shift Invariance & Descent | $a^\top d_{\mathrm{kkt}}(g) = 0, \; d_{\mathrm{kkt}}(g + \mu a) = d_{\mathrm{kkt}}(g), \; \langle g, d_{\mathrm{kkt}}(g) \rangle = -\langle v, H v \rangle \le 0$ |

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

## 5. Blind 30-Seed Benchmark Suite and Native CLI Telemetry

To prevent hyperparameter overfitting, [`benchmarks/`](benchmarks) enforces a strict blind firewall from `src/*.f90`. Every solver is initialized from bit-identical `PCG64` feasible points across **30 deterministic seeds** (`10` primary `101..110`, `10` out-of-sample holdout `201..210`, and `10` untouched third blind holdout `301..310`, totaling **`2,640` runs** across `11` problems × `8` solvers). All `330` initial analytical Euclidean gradients are pre-verified against central finite differences to relative error $\le 1.98 \times 10^{-8} \le 10^{-7}$ ([`benchmarks/referee.py`](benchmarks/referee.py)), and `fortchuck` runs with **strictly frozen out-of-the-box** `FCOptions(gtol=1e-6, max_eval=5000)` (`memory = 7`, `max_iter = 15000`, `zh_eta` auto `0.85`, canonical metric, cautious skip) across every problem and seed.

A run is certified **Solved** by the independent pure-NumPy referee (`project_linear_box_ref`, zero `fortchuck` calls) if and only if:

```math
\mathcal E_{\mathrm{KKT,ref}}(z, X) = \max\!\left(\left\lVert z - P_{\mathcal C}\bigl(z - \nabla_z f(z, X)\bigr) \right\rVert_\infty, \; \max_{1 \le r \le R} \left\lVert G_r - X_r\,\mathrm{sym}(X_r^\top G_r) \right\rVert_{\mathrm F}\right) \le 10^{-6}
```

```math
\max\!\left(\mathcal V_{\mathrm{box}}, \, \mathcal V_{\mathrm{lin}}, \, \mathcal V_{\mathrm{orth}}\right) \le 10^{-8}, \qquad \mathcal V_{\mathrm{lin}} = \frac{\lvert a^\top z - b \rvert}{1 + \lvert b \rvert}
```

Full per-seed JSON archives and per-problem breakdowns are stored in [`results/blind_benchmark_summary.json`](results/blind_benchmark_summary.json) and [`results/blind_benchmark_report.md`](results/blind_benchmark_report.md).

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
| `B8A` | Trace-Constrained Ensemble Kohn--Sham DFT (`v0.2.0`) | $\mathcal C^{12} \times \mathrm{St}(120, 12) \; (\mathbf{1}^\top z = 5)$ | Hard Fermi--Dirac electron conservation (`SLBQP` + Rank-1 KKT Nullspace) |
| `B8B` | Long-Only Weighted Orthogonal Risk-Factor Parity (`v0.2.0`) | $\mathcal C^{24} \times \mathrm{St}(32, 6) \times \mathrm{St}(6, 6)$ | Weighted budget $a^\top z = 1$ ($N_{\mathrm{tot}} = 252 \le 450$, direct `SLSQP`/`trust-constr` comparison) |

### 5.2 Pooled 30-Seed Aggregate Summary (`Seeds 101..110 + 201..210 + 301..310`, `330` Slots/Solver)

| Solver | Evaluated Runs | Solved Runs | Solve Rate | Penalized Evals Med (IQR) | Wall Time ms Med (IQR) | Median KKT Ref | Max Lin Viol | Median Orth Viol | Wilcoxon Evals p (2-sided) | Wilcoxon KKT p (2-sided) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `fortchuck` | 330 | **312** | **94.5%** | **524.5 (1651.5)** | **298.35 (1021.87)** | `8.34e-07` | `1.90e-14` | `3.28e-14` | `—` | `—` |
| `fortchuck-ABB` | 330 | 232 | 70.3% | 1262.0 (4906.0) | 295.36 (1575.94) | `8.32e-07` | `8.59e-15` | `3.01e-14` | `3.27e-34` | `1.30e-13` |
| `Alternating-BCD` | 330 | 160 | 48.5% | 5000.0 (4650.8) | 2436.93 (3434.81) | `1.17e-05` | `5.55e-16` | `1.10e-15` | `7.77e-42` | `1.03e-40` |
| `pymanopt-Warped-CG` | 330 | 73 | 22.1% | 5000.0 (0.0) | 5406.79 (4480.17) | `1.68e-03` | `9.80e-05` | `1.13e-15` | `6.62e-53` | `2.00e-54` |
| `pymanopt-TrustRegions` | 330 | 79 | 23.9% | 5000.0 (0.0) | 3507.79 (2741.58) | `3.87e-03` | `8.32e-05` | `2.71e-15` | `6.62e-53` | `3.99e-47` |
| `SciPy-L-BFGS-B` | 330 | 30 | 9.1% | 5000.0 (0.0) | 2759.82 (3658.59) | `1.30e-04` | `1.02e-06` | `5.20e-06` | `5.62e-51` | `5.17e-55` |
| `SciPy-SLSQP` | 60 (`N_tot <= 450`) | 60 | 100.0% (`18.2%` all) | 58.0 (41.8) | 261.06 (404.25) | `6.17e-07` | `1.44e-15` | `1.17e-12` | `2.80e-02` | `7.40e-01` |
| `SciPy-trust-constr` | 60 (`N_tot <= 450`) | 30 | 50.0% (`9.1%` all) | 2572.0 (4911.5) | 1920.18 (3107.86) | `1.29e-06` | `5.55e-16` | `1.42e-14` | `1.63e-11` | `9.28e-11` |

### 5.3 Multi-Suite Generalization Across Primary (`101..110`), Holdout (`201..210`), and Untouched Blind3 (`301..310`)

| Solver | Primary Solved (`101..110`) | Primary Pen. Evals Med | Holdout Solved (`201..210`) | Holdout Pen. Evals Med | Blind3 Solved (`301..310`) | Blind3 Pen. Evals Med | Blind3 Wilcoxon Evals p |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `fortchuck` | **106/110 (96.4%)** | **506.5** | **103/110 (93.6%)** | **527.0** | **103/110 (93.6%)** | **531.0** | `—` |
| `fortchuck-ABB` | 75/110 (68.2%) | 1303.0 | 80/110 (72.7%) | 1324.0 | 77/110 (70.0%) | 975.0 | `9.73e-13` |
| `Alternating-BCD` | 54/110 (49.1%) | 5000.0 | 53/110 (48.2%) | 5000.0 | 53/110 (48.2%) | 5000.0 | `1.98e-15` |
| `pymanopt-Warped-CG` | 23/110 (20.9%) | 5000.0 | 25/110 (22.7%) | 5000.0 | 25/110 (22.7%) | 5000.0 | `1.25e-18` |
| `pymanopt-TrustRegions` | 24/110 (21.8%) | 5000.0 | 27/110 (24.5%) | 5000.0 | 28/110 (25.5%) | 5000.0 | `1.25e-18` |
| `SciPy-L-BFGS-B` | 10/110 (9.1%) | 5000.0 | 10/110 (9.1%) | 5000.0 | 10/110 (9.1%) | 5000.0 | `6.16e-18` |
| `SciPy-SLSQP` | 20/20 (100.0%) | 59.0 | 20/20 (100.0%) | 54.0 | 20/20 (100.0%) | 58.0 | `4.78e-01` |
| `SciPy-trust-constr` | 10/20 (50.0%) | 2564.5 | 10/20 (50.0%) | 2572.0 | 10/20 (50.0%) | 2564.0 | `1.91e-06` |

### 5.4 Zero-FFI Native Fortran 2018 CLI Telemetry (`app/fortchuck_bench.f90`)

Pure Fortran 2018 release timings (`gfortran -cpp -O3 -march=native -funroll-loops -fopenmp`, `0` Python FFI overhead, `Seeds 101..110`) recorded by `fpm run fortchuck-bench`:

| Native Workload (`fortchuck-bench`) | Windows Pure F2018 | WSL2 Pure F2018 | WSL2 OpenBLAS (`-DFORTCHUCK_USE_BLAS`) | Exact Accuracy / Feasibility Certificate |
| :--- | :---: | :---: | :---: | :--- |
| `[1]` Batched Panel Transport (`St(200,8)`, `K_batch=16`, `4000` reps) | `872.00 ms` vs `857.31 ms` (`0.98x`) | `1064.39 ms` vs `1113.94 ms` (`1.05x`) | **`358.99 ms` vs `454.35 ms` (`1.27x`)** | `max_err = 0.00e+00` vs `16` single-vector calls |
| `[1]` Batched Panel Transport (`St(24,24)`, `K_batch=16`, `4000` reps) | `280.74 ms` vs `281.08 ms` (`1.00x`) | `324.28 ms` vs `327.19 ms` (`1.01x`) | `297.23 ms` vs `282.66 ms` (`0.95x`) | `max_err = 0.00e+00` vs `16` single-vector calls |
| `[2]` `B6` Box-Coupled Ky Fan QP (`10/10` solved) | `973.56 ms` (`1976.0` evals) | `1215.56 ms` (`1982.0` iters) | **`659.38 ms`** | `max |f - f*| <= 5.98e-12`, `max_KKT <= 9.96e-07` |
| `[3]` `B7B` Tall Brockett + Procrustes (`10/10` solved) | `374.51 ms` (`m=7`) / `212.91 ms` (`m=0`) | `1395.26 ms` (`m=7`) / `728.05 ms` (`m=0`) | **`249.95 ms` (`m=7`) / `150.92 ms` (`m=0`)** | `max_KKT <= 9.99e-07`, `max_orth <= 9.67e-14` |
| `[4]` `B8A` Hard-Trace Ensemble DFT (`10/10` solved) | `119.23 ms` (`214.0` evals) | `146.99 ms` (`213.0` iters) | **`68.19 ms` (`208.0` iters, `214.0` evals)** | `max |1^T z - 5| <= 1.52e-13` (`1.78e-15` WSL2), `max_KKT <= 1.00e-06` |

### 5.5 Per-Problem, `v0.1.0 -> v0.2.0` Progression, and Global Optimality Highlights

- **Hard Linear Conservation on `B8A` & `B8B` (`60/60 = 100%` Solved)**: On `B8A` (Trace-Constrained Ensemble Kohn--Sham DFT, `N_tot = 1452`), `fortchuck` solves `30/30` seeds in median **`151.0` evaluations** (`122.10 ms`, max `V_lin = 0.00e+00`), outperforming `Alternating-BCD` (`395.5` evals, `419.30 ms`, `p = 1.86e-09`) by **`2.62x` in evaluations and `3.43x` in wall time** while eliminating the soft trace penalty of `B1` (`2013.0` evals, `1384.87 ms` — a **`13.3x` evaluation reduction**). On `B8B` (`N_tot = 252 <= 450`), `fortchuck` solves `30/30` in `82.22 ms` (`max V_lin = 1.90e-14`), executing **`5.94x` faster** than dense `SciPy-SLSQP` (`488.74 ms`) and **`5.71x` faster** than `Alternating-BCD` (`469.59 ms`), whereas `SciPy-trust-constr` (`0/30`), `pymanopt-Warped-CG` (`0/30`, `V_lin ~ 5.2e-05`), `pymanopt-TrustRegions` (`0/30`), and `SciPy-L-BFGS-B` (`0/30`) solve **`0%`**.
- **Certified Global Optimality (`B6` Ky Fan QP & `B7B` Procrustes, `60/60 = 100%` Solved)**: On `B6` ($f^\ast = -10.445509834730$), `fortchuck` solves `30/30` to median global gap `1.95e-12` (max `4.81e-12` across all 30 seeds) in `1346.82 ms` (`1.32x` faster than `v0.1.0` on the shared 20-seed cohort), while **every** external baseline solves `0/30`. On `B7B` ($f^\ast = -66.949417348256$), `fortchuck` solves `30/30` to median gap `6.25e-13` (max `1.31e-12`) in `496.0` evaluations and `430.78 ms` (`1.41x` wall-clock speedup over `v0.1.0`), outperforming `pymanopt` CG (`1069.5` evals, `1405.68 ms`, `29/30`) and `pymanopt-TrustRegions` (`1683.0` evals, `1218.61 ms`, `p = 1.86e-09`).
- **Stiefel-Dominated Speedups & `30`-Seed Robustness (`B1`, `B2B`, `B3`, `B4`, `B5`, `B7A`)**: Across the 20 shared seeds from `v0.1.0`, batched panel transport and register-tiled kernels accelerate `B7B` (`1.41x`), `B6` (`1.32x`), `B1` (`1.31x`, with fewer evals `2084.5 -> 1959.0`), `B4` (`1.17x`), `B5` (`1.06x`, reaching `26/30 = 86.7%` solved across 30 seeds vs `0/30` for all baselines), and `B2B` (`1.05x`, `30/30` in `39.63 ms` — **`9.3x` fewer evaluations** than `fortchuck-ABB`'s `1203.5`). Coordinatewise `diag_box` preconditioning solves `30/30` on `B7A` in `17.48 ms` (**`2.24x` faster** than Fortran `SciPy-L-BFGS-B` `39.22 ms`) and `16/30` on multi-modal `B2A` (`vs 6/30` `Alternating-BCD`, `3/30` `ABB`, `0/30` `pymanopt`/`SciPy`).

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
│   └── Fortchuck.lean                # 7 machine-checked theorems (1,090 lines, 0 sorry, 0 warnings)
├── benchmarks/
│   ├── problems.py                   # Canonical 11-instance benchmark suite (B1..B8B)
│   ├── referee.py                    # Pure-NumPy project_linear_box_ref, KKT referee & FD verifier
│   ├── baselines.py                  # Adapters for fortchuck, SciPy, pymanopt 2.2.1 & Alternating-BCD
│   └── run_blind_suite.py            # 30-seed 3-suite blind runner & Wilcoxon statistical aggregator
├── results/
│   ├── blind_benchmark_summary.json  # Machine-readable 2,640-run (primary + holdout + blind3) archive
│   └── blind_benchmark_report.md     # Full 30-seed Markdown report + v0.1.0->v0.2.0 & CLI telemetry
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

# 3. Run Python ctypes FFI & pathological guard suite (41 tests)
pytest -v tests/

# 4. Verify central FD gradients across all 30 seeds & run blind 3-suite benchmark (or --smoke)
python -m benchmarks.referee --seeds 101..110
python -m benchmarks.referee --seeds 201..210
python -m benchmarks.referee --seeds 301..310
python benchmarks/run_blind_suite.py --smoke

# 5. Verify Lean 4 formal proofs (7 theorems, 0 sorry, 0 warnings) and 22-rule GFM/KaTeX math compliance
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
  version      = {0.2.0},
  year         = {2026},
  license      = {CC0-1.0},
  url          = {https://github.com/sneed-and-feed/fortchuck}
}
```

---

## 8. License

Dedicated to the public domain under [**CC0 1.0 Universal**](LICENSE). No rights reserved.
