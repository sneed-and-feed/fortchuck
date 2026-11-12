# Blind Multi-Seed Benchmark Report (`fortchuck` v0.1.0)

## 1. Blind Evaluation Protocol and Referee Specification

All benchmark runs enforce a strict blind firewall between `benchmarks/*.py` and `src/*.f90`:

- **Frozen Out-of-the-Box Configuration**: `fortchuck` is evaluated across every benchmark instance and every random seed with `FCOptions(gtol=1.0e-6, max_eval=5000)` (`memory = 7`, `max_iter = 15000`, Zhang--Hager nonmonotone `zh_eta` auto `0.85`, canonical metric, cautious skip, `eps_band = 1.0e-4`) and **zero** per-problem hyperparameter adjustments.
- **Dual Blind & Out-of-Sample Holdout Seeds**: Initial feasible iterates are generated via `PCG64` across `10` deterministic primary seeds (`101..110`) and `10` fresh out-of-sample holdout seeds (`201..210`).
- **Pre-Flight Central Finite-Difference Verification**: Every coordinate of every analytical gradient is verified against central finite differences (`fd_step = 1.0e-5`) to relative error at most $10^{-7}$ prior to solver execution.
- **Independent Post-Solve KKT and Feasibility Certificate**: A run is certified **Solved** by `benchmarks/referee.py` if and only if both optimality and feasibility tolerances hold simultaneously:

```math
\mathcal E_{\mathrm{KKT,ref}}(z, X) = \max\!\left(\left\lVert z - P_{[l,u]}\bigl(z - \nabla_z f(z, X)\bigr) \right\rVert_\infty, \; \max_{1 \le r \le R} \left\lVert G_r - X_r\,\mathrm{sym}(X_r^\top G_r) \right\rVert_{\mathrm F}\right) \le 10^{-6}
```

```math
\max\!\left(\mathcal V_{\mathrm{box}}, \, \mathcal V_{\mathrm{orth}}\right) = \max\!\left(\max_{1 \le i \le d} \max(0, \, l_i - z_i, \, z_i - u_i), \; \max_{1 \le r \le R} \left\lVert X_r^\top X_r - I_{p_r} \right\rVert_{\mathrm F}\right) \le 10^{-8}
```

## 2. Primary Blind Suite Aggregate Summary (`Seeds 101..110`)

Aggregate metrics across all evaluated `(problem, seed)` pairs in the primary blind suite (`Penalized Evals` assigns `max_eval = 5000` to unsolved runs; paired two-sided Wilcoxon signed-rank `p`-values compare `fortchuck` against each baseline across shared non-skipped runs):

| Solver | Evaluated Runs | Solved Runs | Solve Rate | Penalized Evals Med (IQR) | Wall Time ms Med (IQR) | Median KKT Ref | Median Orth Viol | Wilcoxon Evals p (2-sided) | Wilcoxon KKT p (2-sided) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `fortchuck` | 90 | 88 | 97.8% | 602.0 (1662.0) | 619.29 (1437.09) | `8.00e-07` | `2.67e-14` | `—` | `—` |
| `fortchuck-ABB` | 90 | 59 | 65.6% | 3945.5 (4518.5) | 605.28 (1662.13) | `8.89e-07` | `3.88e-14` | `3.73e-16` | `1.18e-07` |
| `Alternating-BCD` | 90 | 34 | 37.8% | 5000.0 (4477.8) | 3006.91 (3210.31) | `1.63e-04` | `1.01e-15` | `7.82e-14` | `2.01e-15` |
| `pymanopt-Warped-CG` | 90 | 23 | 25.6% | 5000.0 (36.8) | 4757.55 (3771.80) | `2.77e-03` | `1.13e-15` | `3.73e-16` | `2.98e-16` |
| `pymanopt-TrustRegions` | 90 | 24 | 26.7% | 5000.0 (610.2) | 3108.43 (1997.64) | `4.45e-03` | `4.08e-14` | `3.73e-16` | `2.61e-13` |
| `SciPy-L-BFGS-B` | 90 | 10 | 11.1% | 5000.0 (0.0) | 2630.73 (3772.81) | `7.97e-04` | `5.56e-06` | `1.44e-15` | `3.41e-16` |
| `SciPy-SLSQP` | 10 | 10 | 100.0% | 75.0 (10.5) | 71.65 (10.18) | `5.69e-07` | `6.82e-14` | `1.95e-03` | `3.22e-01` |
| `SciPy-trust-constr` | 10 | 10 | 100.0% | 90.0 (17.8) | 466.20 (123.67) | `7.85e-07` | `2.46e-15` | `1.95e-03` | `1.95e-03` |

## 3. Out-of-Sample Holdout Suite Aggregate Summary (`Seeds 201..210`)

Aggregate metrics across fresh out-of-sample holdout seeds (`201..210`) under the identical frozen `FCOptions(gtol=1.0e-6, max_eval=5000)` configuration:

| Solver | Evaluated Runs | Solved Runs | Solve Rate | Penalized Evals Med (IQR) | Wall Time ms Med (IQR) | Median KKT Ref | Median Orth Viol | Wilcoxon Evals p (2-sided) | Wilcoxon KKT p (2-sided) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `fortchuck` | 90 | 84 | 93.3% | 576.5 (1668.8) | 626.86 (1352.80) | `7.98e-07` | `2.95e-14` | `—` | `—` |
| `fortchuck-ABB` | 90 | 57 | 63.3% | 3869.5 (4422.2) | 756.07 (1636.34) | `8.44e-07` | `4.16e-14` | `1.06e-14` | `1.34e-06` |
| `Alternating-BCD` | 90 | 33 | 36.7% | 5000.0 (4068.5) | 2993.06 (2450.08) | `3.67e-04` | `1.04e-15` | `6.01e-14` | `1.37e-14` |
| `pymanopt-Warped-CG` | 90 | 25 | 27.8% | 5000.0 (2697.2) | 4566.30 (3227.34) | `1.83e-03` | `1.12e-15` | `1.71e-15` | `2.79e-16` |
| `pymanopt-TrustRegions` | 90 | 27 | 30.0% | 5000.0 (2945.5) | 3011.97 (1885.08) | `3.95e-03` | `3.57e-15` | `1.71e-15` | `6.96e-13` |
| `SciPy-L-BFGS-B` | 90 | 10 | 11.1% | 5000.0 (0.0) | 2444.23 (3687.79) | `4.09e-04` | `6.34e-06` | `7.49e-15` | `3.08e-16` |
| `SciPy-SLSQP` | 10 | 10 | 100.0% | 86.5 (18.5) | 85.05 (20.11) | `4.44e-07` | `4.69e-14` | `1.95e-03` | `9.22e-01` |
| `SciPy-trust-constr` | 10 | 10 | 100.0% | 95.0 (28.2) | 479.05 (177.81) | `7.81e-07` | `5.27e-15` | `1.95e-03` | `1.31e-01` |

### Pooled Primary + Holdout Aggregate (`20` Seeds Total)

| Solver | Evaluated Runs | Solved Runs | Solve Rate | Penalized Evals Med (IQR) | Wall Time ms Med (IQR) | Median KKT Ref | Median Orth Viol | Wilcoxon Evals p (2-sided) | Wilcoxon KKT p (2-sided) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `fortchuck` | 180 | 172 | 95.6% | 588.0 (1676.2) | 619.29 (1390.41) | `8.00e-07` | `2.82e-14` | `—` | `—` |
| `fortchuck-ABB` | 180 | 116 | 64.4% | 3869.5 (4462.5) | 672.76 (1650.08) | `8.78e-07` | `4.08e-14` | `3.64e-29` | `9.46e-13` |
| `Alternating-BCD` | 180 | 67 | 37.2% | 5000.0 (4218.5) | 2998.90 (3109.00) | `1.85e-04` | `1.02e-15` | `3.41e-26` | `3.34e-28` |
| `pymanopt-Warped-CG` | 180 | 48 | 26.7% | 5000.0 (797.5) | 4713.49 (3359.76) | `2.04e-03` | `1.13e-15` | `5.61e-30` | `6.96e-31` |
| `pymanopt-TrustRegions` | 180 | 51 | 28.3% | 5000.0 (2321.5) | 3048.72 (1900.33) | `4.15e-03` | `2.77e-14` | `5.61e-30` | `1.01e-24` |
| `SciPy-L-BFGS-B` | 180 | 20 | 11.1% | 5000.0 (0.0) | 2551.18 (3702.90) | `4.61e-04` | `5.69e-06` | `7.97e-29` | `9.54e-31` |
| `SciPy-SLSQP` | 20 | 20 | 100.0% | 80.0 (17.5) | 77.15 (20.23) | `5.31e-07` | `6.60e-14` | `8.66e-05` | `4.52e-01` |
| `SciPy-trust-constr` | 20 | 20 | 100.0% | 92.0 (19.8) | 473.29 (154.96) | `7.85e-07` | `3.20e-15` | `8.82e-05` | `5.86e-04` |

## 4. Exact Global Optimality Certificate (`B6` Ky Fan QP & `B7B` Procrustes)

Problems `B6` and `B7B` possess exact analytical global minima $f^\ast$ certified by Ky Fan's extremal trace inequality and Kabsch singular value decomposition:

| Suite | Problem | Certified Global Optimum | Solver | Solve Rate | Median Abs Gap | Max Abs Gap | Median Signed Gap | Median KKT Ref |
| :--- | :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| Primary (`101..110`) | `B6` | `-10.445509834730` | `fortchuck` | 10/10 (100%) | `1.91e-12` | `4.46e-12` | `1.91e-12` | `9.29e-07` |
| Primary (`101..110`) | `B6` | `-10.445509834730` | `fortchuck-ABB` | 10/10 (100%) | `9.36e-11` | `1.14e-10` | `9.36e-11` | `7.09e-07` |
| Primary (`101..110`) | `B6` | `-10.445509834730` | `Alternating-BCD` | 0/10 (0%) | `2.41e-03` | `6.52e-03` | `2.41e-03` | `2.09e-02` |
| Primary (`101..110`) | `B6` | `-10.445509834730` | `pymanopt-Warped-CG` | 0/10 (0%) | `5.18e-04` | `2.31e-03` | `5.18e-04` | `5.17e-03` |
| Primary (`101..110`) | `B6` | `-10.445509834730` | `pymanopt-TrustRegions` | 0/10 (0%) | `7.77e-05` | `2.93e-04` | `7.77e-05` | `1.80e-03` |
| Primary (`101..110`) | `B6` | `-10.445509834730` | `SciPy-L-BFGS-B` | 0/10 (0%) | `1.07e-04` | `2.26e-03` | `-1.03e-04` | `2.65e-03` |
| Primary (`101..110`) | `B6` | `-10.445509834730` | `SciPy-SLSQP` | SKIPPED | N/A | N/A | N/A | N/A |
| Primary (`101..110`) | `B6` | `-10.445509834730` | `SciPy-trust-constr` | SKIPPED | N/A | N/A | N/A | N/A |
| Primary (`101..110`) | `B7B` | `-66.949417348256` | `fortchuck` | 10/10 (100%) | `6.32e-13` | `1.04e-12` | `-6.32e-13` | `6.57e-07` |
| Primary (`101..110`) | `B7B` | `-66.949417348256` | `fortchuck-ABB` | 10/10 (100%) | `8.31e-13` | `4.08e-12` | `-2.34e-13` | `6.44e-07` |
| Primary (`101..110`) | `B7B` | `-66.949417348256` | `Alternating-BCD` | 10/10 (100%) | `2.98e-13` | `2.15e-12` | `2.98e-13` | `8.59e-07` |
| Primary (`101..110`) | `B7B` | `-66.949417348256` | `pymanopt-Warped-CG` | 10/10 (100%) | `2.98e-13` | `2.15e-12` | `2.98e-13` | `8.59e-07` |
| Primary (`101..110`) | `B7B` | `-66.949417348256` | `pymanopt-TrustRegions` | 10/10 (100%) | `2.13e-13` | `2.27e-13` | `-2.13e-13` | `1.17e-08` |
| Primary (`101..110`) | `B7B` | `-66.949417348256` | `SciPy-L-BFGS-B` | 0/10 (0%) | `1.31e-01` | `1.31e-01` | `1.31e-01` | `1.42e-03` |
| Primary (`101..110`) | `B7B` | `-66.949417348256` | `SciPy-SLSQP` | SKIPPED | N/A | N/A | N/A | N/A |
| Primary (`101..110`) | `B7B` | `-66.949417348256` | `SciPy-trust-constr` | SKIPPED | N/A | N/A | N/A | N/A |
| Holdout (`201..210`) | `B6` | `-10.445509834730` | `fortchuck` | 10/10 (100%) | `1.39e-12` | `2.54e-12` | `1.39e-12` | `9.16e-07` |
| Holdout (`201..210`) | `B6` | `-10.445509834730` | `fortchuck-ABB` | 8/10 (80%) | `1.03e-10` | `5.47e-09` | `1.03e-10` | `7.50e-07` |
| Holdout (`201..210`) | `B6` | `-10.445509834730` | `Alternating-BCD` | 0/10 (0%) | `1.88e-03` | `4.10e-03` | `1.88e-03` | `9.64e-03` |
| Holdout (`201..210`) | `B6` | `-10.445509834730` | `pymanopt-Warped-CG` | 0/10 (0%) | `2.52e-04` | `2.40e-01` | `2.52e-04` | `1.65e-03` |
| Holdout (`201..210`) | `B6` | `-10.445509834730` | `pymanopt-TrustRegions` | 0/10 (0%) | `7.27e-05` | `1.26e-04` | `7.27e-05` | `1.47e-03` |
| Holdout (`201..210`) | `B6` | `-10.445509834730` | `SciPy-L-BFGS-B` | 0/10 (0%) | `1.01e-04` | `9.56e-04` | `-8.34e-05` | `3.25e-03` |
| Holdout (`201..210`) | `B6` | `-10.445509834730` | `SciPy-SLSQP` | SKIPPED | N/A | N/A | N/A | N/A |
| Holdout (`201..210`) | `B6` | `-10.445509834730` | `SciPy-trust-constr` | SKIPPED | N/A | N/A | N/A | N/A |
| Holdout (`201..210`) | `B7B` | `-66.949417348256` | `fortchuck` | 10/10 (100%) | `8.24e-13` | `1.32e-12` | `-8.24e-13` | `6.82e-07` |
| Holdout (`201..210`) | `B7B` | `-66.949417348256` | `fortchuck-ABB` | 10/10 (100%) | `1.09e-12` | `3.85e-12` | `6.96e-13` | `6.22e-07` |
| Holdout (`201..210`) | `B7B` | `-66.949417348256` | `Alternating-BCD` | 9/10 (90%) | `2.49e-13` | `1.39e-12` | `2.49e-13` | `9.00e-07` |
| Holdout (`201..210`) | `B7B` | `-66.949417348256` | `pymanopt-Warped-CG` | 9/10 (90%) | `2.49e-13` | `1.39e-12` | `2.49e-13` | `9.00e-07` |
| Holdout (`201..210`) | `B7B` | `-66.949417348256` | `pymanopt-TrustRegions` | 10/10 (100%) | `2.13e-13` | `2.42e-13` | `-2.13e-13` | `2.65e-09` |
| Holdout (`201..210`) | `B7B` | `-66.949417348256` | `SciPy-L-BFGS-B` | 0/10 (0%) | `1.50e-03` | `1.31e-01` | `-1.50e-03` | `1.42e-03` |
| Holdout (`201..210`) | `B7B` | `-66.949417348256` | `SciPy-SLSQP` | SKIPPED | N/A | N/A | N/A | N/A |
| Holdout (`201..210`) | `B7B` | `-66.949417348256` | `SciPy-trust-constr` | SKIPPED | N/A | N/A | N/A | N/A |

## 5. Per-Problem Breakdown — Primary Suite (`Seeds 101..110`)

### `B1`: Ensemble Kohn-Sham DFT Fractional Occupations

- **Regime**: `z in [0,1]^10 x St(120,10) (Branch A, metallic orbital-occupation coupling)` (`d = 10`, `R = 1`, `N_tot = 1210`, intrinsic dim `1155`, FD gradient max rel err `2.104e-10`)

| Solver | Solve Rate | Raw Evals Med (IQR) | Penalized Evals Med (IQR) | Wall ms Med (IQR) | Median KKT Ref | Median Box Viol | Median Orth Viol | Median Objective | Wilcoxon Evals p |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `fortchuck` | 10/10 (100%) | 2161.0 (746.0) | 2161.0 (746.0) | 1935.35 (648.39) | `9.12e-07` | `0.00e+00` | `2.87e-14` | `-4.377192` | `—` |
| `fortchuck-ABB` | 0/10 (0%) | 5000.0 (0.0) | 5000.0 (0.0) | 2056.95 (74.04) | `2.14e-05` | `0.00e+00` | `3.04e-14` | `-4.368467` | `1.95e-03` |
| `Alternating-BCD` | 4/10 (40%) | 5000.0 (4646.0) | 5000.0 (4646.0) | 3515.23 (3413.99) | `3.74e-04` | `0.00e+00` | `1.21e-15` | `-4.349253` | `1.31e-01` |
| `pymanopt-Warped-CG` | 0/10 (0%) | 5000.0 (0.0) | 5000.0 (0.0) | 5735.49 (260.76) | `5.64e-04` | `0.00e+00` | `1.29e-15` | `-4.386121` | `1.95e-03` |
| `pymanopt-TrustRegions` | 0/10 (0%) | 5000.0 (0.0) | 5000.0 (0.0) | 3750.34 (100.63) | `7.64e-03` | `0.00e+00` | `1.54e-13` | `-4.386094` | `1.95e-03` |
| `SciPy-L-BFGS-B` | 0/10 (0%) | 5000.0 (0.0) | 5000.0 (0.0) | 4942.57 (247.49) | `7.97e-04` | `0.00e+00` | `3.20e-06` | `-4.368138` | `1.95e-03` |
| `SciPy-SLSQP` | SKIP (`N_tot > 450`) | — | — | — | — | — | — | — | — |
| `SciPy-trust-constr` | SKIP (`N_tot > 450`) | — | — | — | — | — | — | — | — |

### `B2A`: Amplitude-Limited Blind Source Separation (ICA)

- **Regime**: `Z in [-2,2]^{5x150} (d=750) x St(20,5) (Branch A, nonconvex log-cosh)` (`d = 750`, `R = 1`, `N_tot = 850`, intrinsic dim `835`, FD gradient max rel err `2.877e-09`)

| Solver | Solve Rate | Raw Evals Med (IQR) | Penalized Evals Med (IQR) | Wall ms Med (IQR) | Median KKT Ref | Median Box Viol | Median Orth Viol | Median Objective | Wilcoxon Evals p |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `fortchuck` | 8/10 (80%) | 593.0 (145.2) | 602.0 (432.2) | 179.41 (38.18) | `8.68e-07` | `0.00e+00` | `3.63e-14` | `-306.924479` | `—` |
| `fortchuck-ABB` | 1/10 (10%) | 3041.0 (1598.2) | 5000.0 (0.0) | 573.98 (348.06) | `3.36e-06` | `0.00e+00` | `4.09e-14` | `-307.238333` | `7.81e-03` |
| `Alternating-BCD` | 1/10 (10%) | 5000.0 (0.0) | 5000.0 (0.0) | 4309.60 (377.92) | `1.89e-03` | `0.00e+00` | `5.73e-16` | `-307.397095` | `7.81e-03` |
| `pymanopt-Warped-CG` | 0/10 (0%) | 5000.0 (0.0) | 5000.0 (0.0) | 5494.74 (277.97) | `5.66e-01` | `0.00e+00` | `6.53e-16` | `-304.173011` | `7.81e-03` |
| `pymanopt-TrustRegions` | 0/10 (0%) | 5000.0 (0.0) | 5000.0 (0.0) | 3516.54 (277.77) | `3.04e-01` | `0.00e+00` | `2.68e-11` | `-304.608793` | `7.81e-03` |
| `SciPy-L-BFGS-B` | 0/10 (0%) | 5000.0 (0.0) | 5000.0 (0.0) | 4245.38 (365.10) | `9.24e-02` | `0.00e+00` | `2.05e-05` | `-306.311418` | `7.81e-03` |
| `SciPy-SLSQP` | SKIP (`N_tot > 450`) | — | — | — | — | — | — | — | — |
| `SciPy-trust-constr` | SKIP (`N_tot > 450`) | — | — | — | — | — | — | — | — |

### `B2B`: Smooth Penalty-Split Non-Negative Sparse PCA

- **Regime**: `Z in [0,5]^{120x6} (d=720) x St(120,6) (Branch A, ~80% clamped Z=0, FM-2)` (`d = 720`, `R = 1`, `N_tot = 1440`, intrinsic dim `1419`, FD gradient max rel err `1.484e-08`)

| Solver | Solve Rate | Raw Evals Med (IQR) | Penalized Evals Med (IQR) | Wall ms Med (IQR) | Median KKT Ref | Median Box Viol | Median Orth Viol | Median Objective | Wilcoxon Evals p |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `fortchuck` | 10/10 (100%) | 126.0 (40.5) | 126.0 (40.5) | 47.19 (14.47) | `7.14e-07` | `0.00e+00` | `1.28e-14` | `8.643606` | `—` |
| `fortchuck-ABB` | 10/10 (100%) | 887.0 (889.2) | 887.0 (889.2) | 159.21 (158.66) | `9.48e-07` | `0.00e+00` | `6.70e-14` | `8.200998` | `1.95e-03` |
| `Alternating-BCD` | 0/10 (0%) | 3521.0 (1630.8) | 5000.0 (0.0) | 2827.98 (1421.06) | `1.68e-04` | `0.00e+00` | `9.37e-16` | `8.069873` | `1.95e-03` |
| `pymanopt-Warped-CG` | 3/10 (30%) | 4764.0 (3196.0) | 5000.0 (354.0) | 4433.54 (3083.58) | `4.25e-04` | `0.00e+00` | `1.11e-15` | `13.886985` | `1.95e-03` |
| `pymanopt-TrustRegions` | 4/10 (40%) | 5000.0 (610.2) | 5000.0 (610.2) | 3108.43 (453.04) | `1.28e-02` | `0.00e+00` | `1.10e-10` | `8.460341` | `1.95e-03` |
| `SciPy-L-BFGS-B` | 0/10 (0%) | 413.5 (538.0) | 5000.0 (0.0) | 319.97 (408.39) | `2.54e-05` | `0.00e+00` | `5.91e-06` | `10.943772` | `1.95e-03` |
| `SciPy-SLSQP` | SKIP (`N_tot > 450`) | — | — | — | — | — | — | — | — |
| `SciPy-trust-constr` | SKIP (`N_tot > 450`) | — | — | — | — | — | — | — | — |

### `B3`: Multi-Camera Anisotropic Bounded-Scale Rig Wahba

- **Regime**: `z in B^15 x St(3,3)^4 (Branch B SO(3)^4, R=4, kappa(W)=10^3)` (`d = 15`, `R = 4`, `N_tot = 51`, intrinsic dim `27`, FD gradient max rel err `7.213e-11`)

| Solver | Solve Rate | Raw Evals Med (IQR) | Penalized Evals Med (IQR) | Wall ms Med (IQR) | Median KKT Ref | Median Box Viol | Median Orth Viol | Median Objective | Wilcoxon Evals p |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `fortchuck` | 10/10 (100%) | 42.0 (12.5) | 42.0 (12.5) | 12.27 (2.24) | `4.18e-07` | `0.00e+00` | `5.77e-15` | `2.176801` | `—` |
| `fortchuck-ABB` | 10/10 (100%) | 89.0 (19.2) | 89.0 (19.2) | 21.62 (5.28) | `5.47e-07` | `0.00e+00` | `6.33e-15` | `2.176801` | `1.95e-03` |
| `Alternating-BCD` | 9/10 (90%) | 319.5 (12.8) | 323.5 (11.2) | 446.67 (28.85) | `8.77e-07` | `0.00e+00` | `8.26e-16` | `2.176801` | `1.95e-03` |
| `pymanopt-Warped-CG` | 10/10 (100%) | 874.5 (304.5) | 874.5 (304.5) | 1683.11 (677.98) | `7.22e-07` | `0.00e+00` | `8.45e-16` | `2.176801` | `1.95e-03` |
| `pymanopt-TrustRegions` | 10/10 (100%) | 658.0 (52.8) | 658.0 (52.8) | 736.61 (68.56) | `6.27e-07` | `0.00e+00` | `7.28e-16` | `2.176801` | `1.95e-03` |
| `SciPy-L-BFGS-B` | 0/10 (0%) | 801.5 (66.2) | 5000.0 (0.0) | 770.76 (90.03) | `2.29e-05` | `0.00e+00` | `4.83e-06` | `2.176772` | `1.95e-03` |
| `SciPy-SLSQP` | 10/10 (100%) | 75.0 (10.5) | 75.0 (10.5) | 71.65 (10.18) | `5.69e-07` | `0.00e+00` | `6.82e-14` | `2.176801` | `1.95e-03` |
| `SciPy-trust-constr` | 10/10 (100%) | 90.0 (17.8) | 90.0 (17.8) | 466.20 (123.67) | `7.85e-07` | `0.00e+00` | `2.46e-15` | `2.176802` | `1.95e-03` |

### `B4`: Bounded-Variance ML Common Principal Components

- **Regime**: `z in [0.05,20]^150 x St(30,30) (Branch B O(30), log barrier, 1.6e5 scale ratio)` (`d = 150`, `R = 1`, `N_tot = 1050`, intrinsic dim `585`, FD gradient max rel err `6.833e-10`)

| Solver | Solve Rate | Raw Evals Med (IQR) | Penalized Evals Med (IQR) | Wall ms Med (IQR) | Median KKT Ref | Median Box Viol | Median Orth Viol | Median Objective | Wilcoxon Evals p |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `fortchuck` | 10/10 (100%) | 1765.5 (107.2) | 1765.5 (107.2) | 1445.40 (72.96) | `6.63e-07` | `0.00e+00` | `4.11e-14` | `17.596323` | `—` |
| `fortchuck-ABB` | 8/10 (80%) | 4267.0 (584.0) | 4267.0 (584.0) | 2091.44 (394.02) | `7.00e-07` | `0.00e+00` | `4.95e-14` | `17.596323` | `1.95e-03` |
| `Alternating-BCD` | 0/10 (0%) | 5000.0 (0.0) | 5000.0 (0.0) | 3913.42 (28.19) | `1.85e-04` | `0.00e+00` | `3.07e-15` | `17.596323` | `1.95e-03` |
| `pymanopt-Warped-CG` | 0/10 (0%) | 5000.0 (0.0) | 5000.0 (0.0) | 5494.91 (46.19) | `8.56e-03` | `0.00e+00` | `3.16e-15` | `17.599052` | `1.95e-03` |
| `pymanopt-TrustRegions` | 0/10 (0%) | 5000.0 (0.0) | 5000.0 (0.0) | 3548.59 (57.88) | `4.70e-03` | `0.00e+00` | `9.85e-14` | `17.597364` | `1.95e-03` |
| `SciPy-L-BFGS-B` | 0/10 (0%) | 5000.0 (0.0) | 5000.0 (0.0) | 4698.54 (122.38) | `6.40e-01` | `0.00e+00` | `1.03e-05` | `21.044210` | `1.95e-03` |
| `SciPy-SLSQP` | SKIP (`N_tot > 450`) | — | — | — | — | — | — | — | — |
| `SciPy-trust-constr` | SKIP (`N_tot > 450`) | — | — | — | — | — | — | — | — |

### `B5`: Bounded Factor Correlation Matrix Calibration

- **Regime**: `z in B^110 x St(100,10) (Branch A, quartic valley + clamped psi)` (`d = 110`, `R = 1`, `N_tot = 1110`, intrinsic dim `1055`, FD gradient max rel err `8.432e-10`)

| Solver | Solve Rate | Raw Evals Med (IQR) | Penalized Evals Med (IQR) | Wall ms Med (IQR) | Median KKT Ref | Median Box Viol | Median Orth Viol | Median Objective | Wilcoxon Evals p |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `fortchuck` | 10/10 (100%) | 1165.0 (178.5) | 1165.0 (178.5) | 894.84 (117.67) | `9.00e-07` | `0.00e+00` | `3.73e-14` | `34.668813` | `—` |
| `fortchuck-ABB` | 0/10 (0%) | 5000.0 (0.0) | 5000.0 (0.0) | 1731.83 (28.74) | `1.95e-03` | `0.00e+00` | `4.38e-14` | `34.668964` | `1.95e-03` |
| `Alternating-BCD` | 0/10 (0%) | 5000.0 (0.0) | 5000.0 (0.0) | 3442.93 (53.16) | `4.55e-03` | `0.00e+00` | `1.22e-15` | `34.669225` | `1.95e-03` |
| `pymanopt-Warped-CG` | 0/10 (0%) | 5000.0 (0.0) | 5000.0 (0.0) | 5029.09 (101.01) | `1.50e-02` | `0.00e+00` | `1.30e-15` | `34.671020` | `1.95e-03` |
| `pymanopt-TrustRegions` | 0/10 (0%) | 5000.0 (0.0) | 5000.0 (0.0) | 3331.17 (53.76) | `2.28e-02` | `0.00e+00` | `4.02e-12` | `34.669569` | `1.95e-03` |
| `SciPy-L-BFGS-B` | 0/10 (0%) | 2495.5 (595.8) | 5000.0 (0.0) | 2257.59 (403.33) | `3.64e-05` | `0.00e+00` | `1.88e-10` | `34.668813` | `1.95e-03` |
| `SciPy-SLSQP` | SKIP (`N_tot > 450`) | — | — | — | — | — | — | — | — |
| `SciPy-trust-constr` | SKIP (`N_tot > 450`) | — | — | — | — | — | — | — | — |

### `B6`: Box-Coupled Brockett with Exact Ky Fan QP Truth

- **Regime**: `z in [0.1,5]^10 x St(150,10) (Branch A, kappa(M)=10^4, certified f_star)` (`d = 10`, `R = 1`, `N_tot = 1510`, intrinsic dim `1455`, FD gradient max rel err `3.584e-10`)

| Solver | Solve Rate | Raw Evals Med (IQR) | Penalized Evals Med (IQR) | Wall ms Med (IQR) | Median KKT Ref | Median Box Viol | Median Orth Viol | Median Objective | Wilcoxon Evals p |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `fortchuck` | 10/10 (100%) | 1912.0 (179.5) | 1912.0 (179.5) | 1828.88 (175.36) | `9.29e-07` | `0.00e+00` | `4.93e-14` | `-10.445510` | `—` |
| `fortchuck-ABB` | 10/10 (100%) | 4656.5 (443.2) | 4656.5 (443.2) | 1659.85 (177.58) | `7.09e-07` | `0.00e+00` | `4.56e-14` | `-10.445510` | `1.95e-03` |
| `Alternating-BCD` | 0/10 (0%) | 5000.0 (0.0) | 5000.0 (0.0) | 3018.80 (50.54) | `2.09e-02` | `0.00e+00` | `1.05e-15` | `-10.443098` | `1.95e-03` |
| `pymanopt-Warped-CG` | 0/10 (0%) | 5000.0 (0.0) | 5000.0 (0.0) | 4535.92 (173.10) | `5.17e-03` | `0.00e+00` | `1.06e-15` | `-10.444992` | `1.95e-03` |
| `pymanopt-TrustRegions` | 0/10 (0%) | 5000.0 (0.0) | 5000.0 (0.0) | 3050.82 (49.35) | `1.80e-03` | `0.00e+00` | `3.99e-15` | `-10.445432` | `1.95e-03` |
| `SciPy-L-BFGS-B` | 0/10 (0%) | 5000.0 (0.0) | 5000.0 (0.0) | 4464.43 (119.64) | `2.65e-03` | `0.00e+00` | `1.50e-05` | `-10.445612` | `1.95e-03` |
| `SciPy-SLSQP` | SKIP (`N_tot > 450`) | — | — | — | — | — | — | — | — |
| `SciPy-trust-constr` | SKIP (`N_tot > 450`) | — | — | — | — | — | — | — | — |

### `B7A`: Pure-Box Active-Bound Chained Rosenbrock + Quadratic

- **Regime**: `z in [l,u]^500, R=0 (incl. 5 fixed l_i==u_i, ~60% active bounds)` (`d = 500`, `R = 0`, `N_tot = 500`, intrinsic dim `500`, FD gradient max rel err `9.086e-10`)

| Solver | Solve Rate | Raw Evals Med (IQR) | Penalized Evals Med (IQR) | Wall ms Med (IQR) | Median KKT Ref | Median Box Viol | Median Orth Viol | Median Objective | Wilcoxon Evals p |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `fortchuck` | 10/10 (100%) | 67.5 (2.0) | 67.5 (2.0) | 8.75 (2.10) | `8.21e-07` | `0.00e+00` | `0.00e+00` | `87.046277` | `—` |
| `fortchuck-ABB` | 10/10 (100%) | 85.0 (11.0) | 85.0 (11.0) | 8.68 (1.52) | `8.60e-07` | `0.00e+00` | `0.00e+00` | `87.046277` | `1.95e-03` |
| `Alternating-BCD` | 10/10 (100%) | 68.0 (0.8) | 68.0 (0.8) | 37.63 (3.00) | `8.47e-07` | `0.00e+00` | `0.00e+00` | `87.046277` | `2.66e-01` |
| `pymanopt-Warped-CG` | 0/10 (0%) | 5000.0 (0.0) | 5000.0 (0.0) | 1745.73 (104.22) | `1.28e-03` | `0.00e+00` | `0.00e+00` | `87.047726` | `1.95e-03` |
| `pymanopt-TrustRegions` | 0/10 (0%) | 5000.0 (0.0) | 5000.0 (0.0) | 1539.69 (43.33) | `2.15e-04` | `0.00e+00` | `0.00e+00` | `87.046452` | `1.95e-03` |
| `SciPy-L-BFGS-B` | 10/10 (100%) | 68.0 (0.8) | 68.0 (0.8) | 38.58 (3.52) | `8.47e-07` | `0.00e+00` | `0.00e+00` | `87.046277` | `2.66e-01` |
| `SciPy-SLSQP` | SKIP (`N_tot > 450`) | — | — | — | — | — | — | — | — |
| `SciPy-trust-constr` | SKIP (`N_tot > 450`) | — | — | — | — | — | — | — | — |

### `B7B`: Pure-Stiefel Tall Brockett & Orthogonal Procrustes

- **Regime**: `d=0, St(200,8) x St(24,24) (Pure Stiefel Branch A & Branch B)` (`d = 0`, `R = 2`, `N_tot = 2176`, intrinsic dim `1840`, FD gradient max rel err `5.084e-10`)

| Solver | Solve Rate | Raw Evals Med (IQR) | Penalized Evals Med (IQR) | Wall ms Med (IQR) | Median KKT Ref | Median Box Viol | Median Orth Viol | Median Objective | Wilcoxon Evals p |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `fortchuck` | 10/10 (100%) | 499.5 (32.8) | 499.5 (32.8) | 619.29 (52.33) | `6.57e-07` | `0.00e+00` | `6.54e-14` | `-66.949417` | `—` |
| `fortchuck-ABB` | 10/10 (100%) | 627.0 (74.8) | 627.0 (74.8) | 308.35 (43.31) | `6.44e-07` | `0.00e+00` | `7.27e-14` | `-66.949417` | `1.95e-03` |
| `Alternating-BCD` | 10/10 (100%) | 1062.5 (86.8) | 1062.5 (86.8) | 1246.05 (121.98) | `8.59e-07` | `0.00e+00` | `2.46e-15` | `-66.949417` | `1.95e-03` |
| `pymanopt-Warped-CG` | 10/10 (100%) | 1062.5 (86.8) | 1062.5 (86.8) | 1244.34 (110.67) | `8.59e-07` | `0.00e+00` | `2.46e-15` | `-66.949417` | `1.95e-03` |
| `pymanopt-TrustRegions` | 10/10 (100%) | 1700.5 (263.0) | 1700.5 (263.0) | 1122.45 (156.97) | `1.17e-08` | `0.00e+00` | `2.93e-15` | `-66.949417` | `1.95e-03` |
| `SciPy-L-BFGS-B` | 0/10 (0%) | 2877.0 (831.8) | 5000.0 (0.0) | 2454.12 (718.14) | `1.42e-03` | `0.00e+00` | `5.46e-05` | `-66.818295` | `1.95e-03` |
| `SciPy-SLSQP` | SKIP (`N_tot > 450`) | — | — | — | — | — | — | — | — |
| `SciPy-trust-constr` | SKIP (`N_tot > 450`) | — | — | — | — | — | — | — | — |

## 6. Per-Problem Breakdown — Out-of-Sample Holdout Suite (`Seeds 201..210`)

### `B1`: Ensemble Kohn-Sham DFT Fractional Occupations

- **Regime**: `z in [0,1]^10 x St(120,10) (Branch A, metallic orbital-occupation coupling)` (`d = 10`, `R = 1`, `N_tot = 1210`, intrinsic dim `1155`, FD gradient max rel err `2.091e-10`)

| Solver | Solve Rate | Raw Evals Med (IQR) | Penalized Evals Med (IQR) | Wall ms Med (IQR) | Median KKT Ref | Median Box Viol | Median Orth Viol | Median Objective | Wilcoxon Evals p |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `fortchuck` | 10/10 (100%) | 1963.5 (672.5) | 1963.5 (672.5) | 1585.30 (493.61) | `9.05e-07` | `0.00e+00` | `4.81e-14` | `-4.372830` | `—` |
| `fortchuck-ABB` | 0/10 (0%) | 5000.0 (0.0) | 5000.0 (0.0) | 1764.53 (22.69) | `1.06e-04` | `0.00e+00` | `5.14e-14` | `-4.368467` | `1.95e-03` |
| `Alternating-BCD` | 2/10 (20%) | 5000.0 (0.0) | 5000.0 (0.0) | 3217.01 (105.12) | `1.37e-03` | `0.00e+00` | `1.20e-15` | `-4.349173` | `1.37e-02` |
| `pymanopt-Warped-CG` | 0/10 (0%) | 5000.0 (0.0) | 5000.0 (0.0) | 4857.62 (65.66) | `8.43e-04` | `0.00e+00` | `1.32e-15` | `-4.386120` | `1.95e-03` |
| `pymanopt-TrustRegions` | 0/10 (0%) | 5000.0 (0.0) | 5000.0 (0.0) | 3171.16 (32.14) | `9.59e-03` | `0.00e+00` | `1.35e-13` | `-4.386097` | `1.95e-03` |
| `SciPy-L-BFGS-B` | 0/10 (0%) | 5000.0 (0.0) | 5000.0 (0.0) | 4421.93 (59.63) | `4.09e-04` | `0.00e+00` | `3.20e-06` | `-4.358742` | `1.95e-03` |
| `SciPy-SLSQP` | SKIP (`N_tot > 450`) | — | — | — | — | — | — | — | — |
| `SciPy-trust-constr` | SKIP (`N_tot > 450`) | — | — | — | — | — | — | — | — |

### `B2A`: Amplitude-Limited Blind Source Separation (ICA)

- **Regime**: `Z in [-2,2]^{5x150} (d=750) x St(20,5) (Branch A, nonconvex log-cosh)` (`d = 750`, `R = 1`, `N_tot = 850`, intrinsic dim `835`, FD gradient max rel err `1.935e-09`)

| Solver | Solve Rate | Raw Evals Med (IQR) | Penalized Evals Med (IQR) | Wall ms Med (IQR) | Median KKT Ref | Median Box Viol | Median Orth Viol | Median Objective | Wilcoxon Evals p |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `fortchuck` | 7/10 (70%) | 549.0 (373.0) | 549.0 (3512.8) | 141.39 (102.86) | `8.91e-07` | `0.00e+00` | `4.52e-14` | `-309.118287` | `—` |
| `fortchuck-ABB` | 1/10 (10%) | 4565.5 (1008.8) | 5000.0 (0.0) | 756.07 (181.66) | `2.14e-05` | `0.00e+00` | `4.61e-14` | `-308.979735` | `1.56e-02` |
| `Alternating-BCD` | 2/10 (20%) | 5000.0 (0.0) | 5000.0 (0.0) | 3782.02 (85.89) | `1.17e-02` | `0.00e+00` | `6.98e-16` | `-306.937441` | `1.56e-02` |
| `pymanopt-Warped-CG` | 0/10 (0%) | 5000.0 (0.0) | 5000.0 (0.0) | 4759.49 (146.30) | `4.67e-01` | `0.00e+00` | `6.53e-16` | `-309.091804` | `1.56e-02` |
| `pymanopt-TrustRegions` | 0/10 (0%) | 5000.0 (0.0) | 5000.0 (0.0) | 3044.81 (34.47) | `6.25e-01` | `0.00e+00` | `3.22e-11` | `-307.525867` | `1.56e-02` |
| `SciPy-L-BFGS-B` | 0/10 (0%) | 5000.0 (0.0) | 5000.0 (0.0) | 3769.62 (169.44) | `4.53e-02` | `0.00e+00` | `3.04e-05` | `-309.847765` | `1.56e-02` |
| `SciPy-SLSQP` | SKIP (`N_tot > 450`) | — | — | — | — | — | — | — | — |
| `SciPy-trust-constr` | SKIP (`N_tot > 450`) | — | — | — | — | — | — | — | — |

### `B2B`: Smooth Penalty-Split Non-Negative Sparse PCA

- **Regime**: `Z in [0,5]^{120x6} (d=720) x St(120,6) (Branch A, ~80% clamped Z=0, FM-2)` (`d = 720`, `R = 1`, `N_tot = 1440`, intrinsic dim `1419`, FD gradient max rel err `1.028e-08`)

| Solver | Solve Rate | Raw Evals Med (IQR) | Penalized Evals Med (IQR) | Wall ms Med (IQR) | Median KKT Ref | Median Box Viol | Median Orth Viol | Median Objective | Wilcoxon Evals p |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `fortchuck` | 10/10 (100%) | 100.5 (26.0) | 100.5 (26.0) | 36.36 (9.26) | `6.41e-07` | `0.00e+00` | `1.06e-14` | `8.606887` | `—` |
| `fortchuck-ABB` | 10/10 (100%) | 1447.0 (886.2) | 1447.0 (886.2) | 260.99 (148.27) | `9.34e-07` | `0.00e+00` | `3.02e-14` | `8.026764` | `1.95e-03` |
| `Alternating-BCD` | 0/10 (0%) | 3651.0 (2740.5) | 5000.0 (0.0) | 2801.24 (2033.33) | `2.14e-04` | `0.00e+00` | `1.15e-15` | `8.188735` | `1.95e-03` |
| `pymanopt-Warped-CG` | 6/10 (60%) | 4045.0 (3606.5) | 4045.0 (3606.5) | 3810.59 (3432.04) | `9.27e-07` | `0.00e+00` | `1.00e-15` | `13.121072` | `1.95e-03` |
| `pymanopt-TrustRegions` | 7/10 (70%) | 2930.5 (2722.0) | 2930.5 (2722.0) | 1835.12 (1700.28) | `8.56e-07` | `0.00e+00` | `1.18e-15` | `8.282000` | `1.95e-03` |
| `SciPy-L-BFGS-B` | 0/10 (0%) | 358.5 (359.8) | 5000.0 (0.0) | 283.13 (281.84) | `1.20e-04` | `0.00e+00` | `1.15e-05` | `13.161188` | `1.95e-03` |
| `SciPy-SLSQP` | SKIP (`N_tot > 450`) | — | — | — | — | — | — | — | — |
| `SciPy-trust-constr` | SKIP (`N_tot > 450`) | — | — | — | — | — | — | — | — |

### `B3`: Multi-Camera Anisotropic Bounded-Scale Rig Wahba

- **Regime**: `z in B^15 x St(3,3)^4 (Branch B SO(3)^4, R=4, kappa(W)=10^3)` (`d = 15`, `R = 4`, `N_tot = 51`, intrinsic dim `27`, FD gradient max rel err `9.877e-11`)

| Solver | Solve Rate | Raw Evals Med (IQR) | Penalized Evals Med (IQR) | Wall ms Med (IQR) | Median KKT Ref | Median Box Viol | Median Orth Viol | Median Objective | Wilcoxon Evals p |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `fortchuck` | 10/10 (100%) | 46.0 (10.8) | 46.0 (10.8) | 11.14 (2.90) | `5.30e-07` | `0.00e+00` | `5.74e-15` | `2.176801` | `—` |
| `fortchuck-ABB` | 10/10 (100%) | 90.0 (14.0) | 90.0 (14.0) | 20.12 (4.48) | `6.11e-07` | `0.00e+00` | `7.51e-15` | `2.176801` | `1.95e-03` |
| `Alternating-BCD` | 10/10 (100%) | 316.5 (25.2) | 316.5 (25.2) | 424.06 (33.22) | `7.93e-07` | `0.00e+00` | `6.11e-16` | `2.176801` | `1.95e-03` |
| `pymanopt-Warped-CG` | 10/10 (100%) | 854.0 (368.2) | 854.0 (368.2) | 1580.69 (691.35) | `8.58e-07` | `0.00e+00` | `5.68e-16` | `2.176801` | `1.95e-03` |
| `pymanopt-TrustRegions` | 10/10 (100%) | 651.5 (77.5) | 651.5 (77.5) | 691.34 (91.39) | `5.39e-07` | `0.00e+00` | `6.81e-16` | `2.176801` | `1.95e-03` |
| `SciPy-L-BFGS-B` | 0/10 (0%) | 805.0 (153.5) | 5000.0 (0.0) | 740.99 (119.73) | `2.28e-05` | `0.00e+00` | `4.83e-06` | `2.176772` | `1.95e-03` |
| `SciPy-SLSQP` | 10/10 (100%) | 86.5 (18.5) | 86.5 (18.5) | 85.05 (20.11) | `4.44e-07` | `0.00e+00` | `4.69e-14` | `2.176801` | `1.95e-03` |
| `SciPy-trust-constr` | 10/10 (100%) | 95.0 (28.2) | 95.0 (28.2) | 479.05 (177.81) | `7.81e-07` | `0.00e+00` | `5.27e-15` | `2.176802` | `1.95e-03` |

### `B4`: Bounded-Variance ML Common Principal Components

- **Regime**: `z in [0.05,20]^150 x St(30,30) (Branch B O(30), log barrier, 1.6e5 scale ratio)` (`d = 150`, `R = 1`, `N_tot = 1050`, intrinsic dim `585`, FD gradient max rel err `4.202e-10`)

| Solver | Solve Rate | Raw Evals Med (IQR) | Penalized Evals Med (IQR) | Wall ms Med (IQR) | Median KKT Ref | Median Box Viol | Median Orth Viol | Median Objective | Wilcoxon Evals p |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `fortchuck` | 10/10 (100%) | 1749.5 (93.8) | 1749.5 (93.8) | 1400.37 (89.03) | `6.55e-07` | `0.00e+00` | `4.52e-14` | `17.596323` | `—` |
| `fortchuck-ABB` | 8/10 (80%) | 4281.0 (920.8) | 4281.0 (920.8) | 2031.52 (409.62) | `7.04e-07` | `0.00e+00` | `6.81e-14` | `17.596323` | `1.95e-03` |
| `Alternating-BCD` | 0/10 (0%) | 5000.0 (0.0) | 5000.0 (0.0) | 3840.67 (73.37) | `9.87e-04` | `0.00e+00` | `3.13e-15` | `17.596323` | `1.95e-03` |
| `pymanopt-Warped-CG` | 0/10 (0%) | 5000.0 (0.0) | 5000.0 (0.0) | 5383.44 (161.79) | `1.19e-02` | `0.00e+00` | `3.02e-15` | `17.599161` | `1.95e-03` |
| `pymanopt-TrustRegions` | 0/10 (0%) | 5000.0 (0.0) | 5000.0 (0.0) | 3498.80 (103.76) | `5.17e-03` | `0.00e+00` | `7.26e-14` | `17.597603` | `1.95e-03` |
| `SciPy-L-BFGS-B` | 0/10 (0%) | 5000.0 (0.0) | 5000.0 (0.0) | 4612.58 (47.25) | `5.91e-01` | `0.00e+00` | `7.85e-06` | `20.832811` | `1.95e-03` |
| `SciPy-SLSQP` | SKIP (`N_tot > 450`) | — | — | — | — | — | — | — | — |
| `SciPy-trust-constr` | SKIP (`N_tot > 450`) | — | — | — | — | — | — | — | — |

### `B5`: Bounded Factor Correlation Matrix Calibration

- **Regime**: `z in B^110 x St(100,10) (Branch A, quartic valley + clamped psi)` (`d = 110`, `R = 1`, `N_tot = 1110`, intrinsic dim `1055`, FD gradient max rel err `9.454e-10`)

| Solver | Solve Rate | Raw Evals Med (IQR) | Penalized Evals Med (IQR) | Wall ms Med (IQR) | Median KKT Ref | Median Box Viol | Median Orth Viol | Median Objective | Wilcoxon Evals p |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `fortchuck` | 7/10 (70%) | 1258.5 (202.2) | 1347.5 (2846.8) | 937.34 (140.85) | `9.60e-07` | `0.00e+00` | `5.30e-14` | `34.668813` | `—` |
| `fortchuck-ABB` | 0/10 (0%) | 5000.0 (0.0) | 5000.0 (0.0) | 1702.56 (13.52) | `1.77e-03` | `0.00e+00` | `3.95e-14` | `34.668954` | `1.56e-02` |
| `Alternating-BCD` | 0/10 (0%) | 5000.0 (0.0) | 5000.0 (0.0) | 3395.43 (58.31) | `5.73e-03` | `0.00e+00` | `9.96e-16` | `34.669074` | `1.56e-02` |
| `pymanopt-Warped-CG` | 0/10 (0%) | 5000.0 (0.0) | 5000.0 (0.0) | 5008.48 (112.21) | `2.43e-02` | `0.00e+00` | `1.28e-15` | `34.671823` | `1.56e-02` |
| `pymanopt-TrustRegions` | 0/10 (0%) | 5000.0 (0.0) | 5000.0 (0.0) | 3266.41 (64.93) | `2.36e-02` | `0.00e+00` | `3.87e-12` | `34.669714` | `1.56e-02` |
| `SciPy-L-BFGS-B` | 0/10 (0%) | 2508.5 (849.2) | 5000.0 (0.0) | 2303.85 (708.62) | `3.08e-05` | `0.00e+00` | `1.28e-10` | `34.668813` | `1.56e-02` |
| `SciPy-SLSQP` | SKIP (`N_tot > 450`) | — | — | — | — | — | — | — | — |
| `SciPy-trust-constr` | SKIP (`N_tot > 450`) | — | — | — | — | — | — | — | — |

### `B6`: Box-Coupled Brockett with Exact Ky Fan QP Truth

- **Regime**: `z in [0.1,5]^10 x St(150,10) (Branch A, kappa(M)=10^4, certified f_star)` (`d = 10`, `R = 1`, `N_tot = 1510`, intrinsic dim `1455`, FD gradient max rel err `3.113e-10`)

| Solver | Solve Rate | Raw Evals Med (IQR) | Penalized Evals Med (IQR) | Wall ms Med (IQR) | Median KKT Ref | Median Box Viol | Median Orth Viol | Median Objective | Wilcoxon Evals p |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `fortchuck` | 10/10 (100%) | 1863.0 (297.5) | 1863.0 (297.5) | 1773.78 (292.13) | `9.16e-07` | `0.00e+00` | `3.88e-14` | `-10.445510` | `—` |
| `fortchuck-ABB` | 8/10 (80%) | 4538.0 (954.0) | 4538.0 (954.0) | 1603.34 (331.76) | `7.50e-07` | `0.00e+00` | `5.11e-14` | `-10.445510` | `1.95e-03` |
| `Alternating-BCD` | 0/10 (0%) | 5000.0 (0.0) | 5000.0 (0.0) | 2984.20 (39.14) | `9.64e-03` | `0.00e+00` | `1.06e-15` | `-10.443625` | `1.95e-03` |
| `pymanopt-Warped-CG` | 0/10 (0%) | 5000.0 (0.0) | 5000.0 (0.0) | 4497.09 (65.84) | `1.65e-03` | `0.00e+00` | `1.16e-15` | `-10.445257` | `1.95e-03` |
| `pymanopt-TrustRegions` | 0/10 (0%) | 5000.0 (0.0) | 5000.0 (0.0) | 2993.22 (48.08) | `1.47e-03` | `0.00e+00` | `6.34e-15` | `-10.445437` | `1.95e-03` |
| `SciPy-L-BFGS-B` | 0/10 (0%) | 5000.0 (0.0) | 5000.0 (0.0) | 4478.06 (69.57) | `3.25e-03` | `0.00e+00` | `1.50e-05` | `-10.445593` | `1.95e-03` |
| `SciPy-SLSQP` | SKIP (`N_tot > 450`) | — | — | — | — | — | — | — | — |
| `SciPy-trust-constr` | SKIP (`N_tot > 450`) | — | — | — | — | — | — | — | — |

### `B7A`: Pure-Box Active-Bound Chained Rosenbrock + Quadratic

- **Regime**: `z in [l,u]^500, R=0 (incl. 5 fixed l_i==u_i, ~60% active bounds)` (`d = 500`, `R = 0`, `N_tot = 500`, intrinsic dim `500`, FD gradient max rel err `8.252e-10`)

| Solver | Solve Rate | Raw Evals Med (IQR) | Penalized Evals Med (IQR) | Wall ms Med (IQR) | Median KKT Ref | Median Box Viol | Median Orth Viol | Median Objective | Wilcoxon Evals p |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `fortchuck` | 10/10 (100%) | 67.5 (1.0) | 67.5 (1.0) | 7.96 (0.81) | `8.45e-07` | `0.00e+00` | `0.00e+00` | `87.046277` | `—` |
| `fortchuck-ABB` | 10/10 (100%) | 88.5 (2.8) | 88.5 (2.8) | 8.62 (1.41) | `7.14e-07` | `0.00e+00` | `0.00e+00` | `87.046277` | `1.95e-03` |
| `Alternating-BCD` | 10/10 (100%) | 66.5 (2.0) | 66.5 (2.0) | 37.03 (3.23) | `8.99e-07` | `0.00e+00` | `0.00e+00` | `87.046277` | `8.36e-01` |
| `pymanopt-Warped-CG` | 0/10 (0%) | 5000.0 (0.0) | 5000.0 (0.0) | 1697.02 (47.35) | `2.38e-02` | `0.00e+00` | `0.00e+00` | `87.048625` | `1.95e-03` |
| `pymanopt-TrustRegions` | 0/10 (0%) | 5000.0 (0.0) | 5000.0 (0.0) | 1495.16 (35.58) | `1.88e-04` | `0.00e+00` | `0.00e+00` | `87.046492` | `1.95e-03` |
| `SciPy-L-BFGS-B` | 10/10 (100%) | 66.5 (2.0) | 66.5 (2.0) | 35.86 (2.95) | `8.99e-07` | `0.00e+00` | `0.00e+00` | `87.046277` | `8.36e-01` |
| `SciPy-SLSQP` | SKIP (`N_tot > 450`) | — | — | — | — | — | — | — | — |
| `SciPy-trust-constr` | SKIP (`N_tot > 450`) | — | — | — | — | — | — | — | — |

### `B7B`: Pure-Stiefel Tall Brockett & Orthogonal Procrustes

- **Regime**: `d=0, St(200,8) x St(24,24) (Pure Stiefel Branch A & Branch B)` (`d = 0`, `R = 2`, `N_tot = 2176`, intrinsic dim `1840`, FD gradient max rel err `4.872e-10`)

| Solver | Solve Rate | Raw Evals Med (IQR) | Penalized Evals Med (IQR) | Wall ms Med (IQR) | Median KKT Ref | Median Box Viol | Median Orth Viol | Median Objective | Wilcoxon Evals p |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `fortchuck` | 10/10 (100%) | 520.0 (58.2) | 520.0 (58.2) | 626.86 (83.78) | `6.82e-07` | `0.00e+00` | `6.99e-14` | `-66.949417` | `—` |
| `fortchuck-ABB` | 10/10 (100%) | 693.5 (78.2) | 693.5 (78.2) | 331.91 (45.38) | `6.22e-07` | `0.00e+00` | `6.11e-14` | `-66.949417` | `1.95e-03` |
| `Alternating-BCD` | 9/10 (90%) | 1096.5 (126.2) | 1114.0 (191.2) | 1254.56 (167.78) | `9.00e-07` | `0.00e+00` | `2.57e-15` | `-66.949417` | `1.95e-03` |
| `pymanopt-Warped-CG` | 9/10 (90%) | 1096.5 (126.2) | 1114.0 (191.2) | 1264.13 (184.93) | `9.00e-07` | `0.00e+00` | `2.57e-15` | `-66.949417` | `1.95e-03` |
| `pymanopt-TrustRegions` | 10/10 (100%) | 1737.5 (210.0) | 1737.5 (210.0) | 1200.82 (192.46) | `2.65e-09` | `0.00e+00` | `2.76e-15` | `-66.949417` | `1.95e-03` |
| `SciPy-L-BFGS-B` | 0/10 (0%) | 2210.0 (353.2) | 5000.0 (0.0) | 1906.46 (319.46) | `1.42e-03` | `0.00e+00` | `5.46e-05` | `-66.950913` | `1.95e-03` |
| `SciPy-SLSQP` | SKIP (`N_tot > 450`) | — | — | — | — | — | — | — | — |
| `SciPy-trust-constr` | SKIP (`N_tot > 450`) | — | — | — | — | — | — | — | — |

