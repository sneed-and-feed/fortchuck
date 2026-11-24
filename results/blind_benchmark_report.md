# Blind Multi-Seed Benchmark Report (`fortchuck` v0.2.1)

## 1. Blind Evaluation Protocol and Referee Specification

All benchmark runs enforce a strict blind firewall between `benchmarks/*.py` and `src/*.f90`:

- **Frozen Out-of-the-Box Configuration**: `fortchuck` is evaluated across every benchmark instance (`B1` through `B8B`) and every random seed with `FCOptions(gtol=1.0e-6, max_eval=5000)` (`memory = 7`, `max_iter = 15000`, Zhang--Hager nonmonotone `zh_eta` auto `0.85`, canonical metric, cautious skip, `eps_band = 1.0e-4`), passing `linear_constraint=(a, b)` when `lin_a` is present and **zero** per-problem hyperparameter adjustments.
- **Multi-Suite Deterministic Initializations**: Initial feasible iterates are generated via `PCG64` across `10` deterministic primary seeds (`101..110`), `10` out-of-sample holdout seeds (`201..210`), `10` third blind holdout seeds (`301..310`), `10` virgin fourth blind holdout seeds (`401..410`). On linearly constrained problems (`B8A`, `B8B`), initial Euclidean draws are projected onto the central 80% sub-box intersected with $a^\top z = b$ via `project_linear_box_ref` so every solver starts from the exact same strictly interior feasible point.
- **Pre-Flight Central Finite-Difference Verification**: Every coordinate of every analytical gradient is verified against central finite differences (`fd_step = 1.0e-5`) to relative error at most $10^{-7}$ prior to solver execution.
- **Independent Post-Solve KKT and Feasibility Certificate**: A run is certified **Solved** by `benchmarks/referee.py` if and only if both optimality and feasibility tolerances hold simultaneously (using the pure-NumPy safeguarded root-finder `project_linear_box_ref` with zero calls to `fortchuck` when `lin_a` is present):

```math
\mathcal E_{\mathrm{KKT,ref}}(z, X) = \max\!\left(\left\lVert z - P_{\mathcal C}\bigl(z - \nabla_z f(z, X)\bigr) \right\rVert_\infty, \; \max_{1 \le r \le R} \left\lVert G_r - X_r\,\mathrm{sym}(X_r^\top G_r) \right\rVert_{\mathrm F}\right) \le 10^{-6}
```

```math
\max\!\left(\mathcal V_{\mathrm{box}}, \, \mathcal V_{\mathrm{lin}}, \, \mathcal V_{\mathrm{orth}}\right) \le 10^{-8}, \qquad \mathcal V_{\mathrm{lin}} = \frac{\lvert a^\top z - b \rvert}{1 + \lvert b \rvert}
```

## 2. Pooled Multi-Suite Aggregate Summary (`40` Seeds Total)

Pooled aggregate metrics across all executed blind seed suites (`Penalized Evals` assigns `max_eval = 5000` to unsolved runs; directional two-sided paired Wilcoxon signed-rank `p`-values compare `fortchuck` against each baseline across shared non-skipped runs, where `FC better` means the signed-rank sum favors lower/better values for `fortchuck`, `FC worse` favors the baseline, and `tie` indicates equal signed-rank sums):

| Solver | Evaluated Runs | Solved Runs | Solve Rate | Penalized Evals Med (IQR) | Wall Time ms Med (IQR) | Median KKT Ref | Max Lin Viol | Median Orth Viol | Wilcoxon Evals p (2-sided) | Wilcoxon KKT p (2-sided) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `fortchuck` | 440 | 440 | 100.0% | 472.0 (1505.2) | 206.77 (1130.73) | `7.91e-07` | `2.96e-16` | `2.68e-14` | `—` | `—` |
| `fortchuck-ABB` | 440 | 330 | 75.0% | 1016.5 (4905.2) | 337.29 (1735.68) | `8.17e-07` | `1.78e-15` | `3.15e-14` | `2.20e-70 (FC better)` | `3.28e-18 (FC better)` |
| `Alternating-BCD` | 440 | 213 | 48.4% | 5000.0 (4643.0) | 2424.09 (3233.56) | `1.17e-05` | `5.55e-16` | `1.09e-15` | `3.68e-65 (FC better)` | `2.26e-57 (FC better)` |
| `pymanopt-Warped-CG` | 440 | 98 | 22.3% | 5000.0 (0.0) | 4950.47 (4332.76) | `1.70e-03` | `9.80e-05` | `1.13e-15` | `7.94e-74 (FC better)` | `1.96e-72 (FC better)` |
| `pymanopt-TrustRegions` | 440 | 106 | 24.1% | 5000.0 (0.0) | 3201.50 (2637.97) | `3.77e-03` | `8.32e-05` | `2.65e-15` | `7.94e-74 (FC better)` | `9.41e-63 (FC better)` |
| `SciPy-L-BFGS-B` | 440 | 40 | 9.1% | 5000.0 (0.0) | 2715.65 (3652.59) | `1.30e-04` | `1.02e-06` | `5.29e-06` | `9.69e-72 (FC better)` | `5.29e-73 (FC better)` |
| `SciPy-SLSQP` | 80 | 80 | 100.0% | 58.0 (41.5) | 261.06 (395.50) | `6.31e-07` | `1.44e-15` | `1.17e-12` | `5.09e-10 (FC better)` | `7.05e-01 (FC worse)` |
| `SciPy-trust-constr` | 80 | 40 | 50.0% | 2572.0 (4913.0) | 1898.06 (2960.78) | `1.29e-06` | `5.55e-16` | `7.55e-15` | `7.53e-15 (FC better)` | `1.87e-13 (FC better)` |

## 3. Primary Blind Suite Aggregate Summary (`Seeds 101..110`)

Aggregate metrics across all evaluated `(problem, seed)` pairs in the primary blind suite:

| Solver | Evaluated Runs | Solved Runs | Solve Rate | Penalized Evals Med (IQR) | Wall Time ms Med (IQR) | Median KKT Ref | Max Lin Viol | Median Orth Viol | Wilcoxon Evals p (2-sided) | Wilcoxon KKT p (2-sided) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `fortchuck` | 110 | 110 | 100.0% | 481.0 (1552.0) | 187.40 (1103.74) | `8.07e-07` | `1.48e-16` | `2.86e-14` | `—` | `—` |
| `fortchuck-ABB` | 110 | 82 | 74.5% | 1303.0 (4864.5) | 316.06 (1681.29) | `8.57e-07` | `1.48e-16` | `3.06e-14` | `3.23e-19 (FC better)` | `1.61e-06 (FC better)` |
| `Alternating-BCD` | 110 | 54 | 49.1% | 5000.0 (4664.2) | 1763.24 (3260.69) | `8.11e-06` | `3.33e-16` | `1.06e-15` | `1.44e-16 (FC better)` | `4.99e-15 (FC better)` |
| `pymanopt-Warped-CG` | 110 | 23 | 20.9% | 5000.0 (0.0) | 5028.55 (4102.04) | `1.52e-03` | `8.50e-05` | `1.14e-15` | `8.74e-20 (FC better)` | `1.78e-19 (FC better)` |
| `pymanopt-TrustRegions` | 110 | 24 | 21.8% | 5000.0 (0.0) | 3278.26 (2239.10) | `4.17e-03` | `8.32e-05` | `3.36e-15` | `8.74e-20 (FC better)` | `3.84e-17 (FC better)` |
| `SciPy-L-BFGS-B` | 110 | 10 | 9.1% | 5000.0 (0.0) | 2790.81 (3643.59) | `9.71e-05` | `1.26e-07` | `5.56e-06` | `3.23e-19 (FC better)` | `1.51e-19 (FC better)` |
| `SciPy-SLSQP` | 20 | 20 | 100.0% | 59.0 (34.0) | 286.70 (453.63) | `6.48e-07` | `1.11e-15` | `1.40e-12` | `4.79e-04 (FC better)` | `8.12e-01 (FC better)` |
| `SciPy-trust-constr` | 20 | 10 | 50.0% | 2564.5 (4909.5) | 1845.38 (3841.97) | `1.42e-05` | `5.55e-16` | `6.13e-14` | `8.78e-05 (FC better)` | `1.91e-06 (FC better)` |

## 4. Out-of-Sample Holdout Suite Aggregate Summary (`Seeds 201..210`)

Aggregate metrics across out-of-sample holdout seeds (`201..210`) under the identical frozen `FCOptions(gtol=1.0e-6, max_eval=5000)` configuration:

| Solver | Evaluated Runs | Solved Runs | Solve Rate | Penalized Evals Med (IQR) | Wall Time ms Med (IQR) | Median KKT Ref | Max Lin Viol | Median Orth Viol | Wilcoxon Evals p (2-sided) | Wilcoxon KKT p (2-sided) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `fortchuck` | 110 | 110 | 100.0% | 469.0 (1451.8) | 222.16 (1067.50) | `7.94e-07` | `2.96e-16` | `2.62e-14` | `—` | `—` |
| `fortchuck-ABB` | 110 | 83 | 75.5% | 1324.0 (4892.5) | 332.71 (1669.86) | `7.41e-07` | `1.78e-15` | `2.83e-14` | `9.73e-19 (FC better)` | `2.54e-04 (FC better)` |
| `Alternating-BCD` | 110 | 53 | 48.2% | 5000.0 (4641.5) | 2238.97 (3807.73) | `4.10e-05` | `5.55e-16` | `1.09e-15` | `4.69e-18 (FC better)` | `1.16e-15 (FC better)` |
| `pymanopt-Warped-CG` | 110 | 25 | 22.7% | 5000.0 (0.0) | 5936.10 (4522.30) | `1.68e-03` | `9.49e-05` | `1.12e-15` | `8.74e-20 (FC better)` | `1.32e-19 (FC better)` |
| `pymanopt-TrustRegions` | 110 | 27 | 24.5% | 5000.0 (0.0) | 3762.86 (2933.98) | `3.63e-03` | `5.87e-05` | `2.45e-15` | `8.74e-20 (FC better)` | `8.43e-17 (FC better)` |
| `SciPy-L-BFGS-B` | 110 | 10 | 9.1% | 5000.0 (0.0) | 2472.08 (3547.09) | `1.52e-04` | `1.26e-07` | `6.34e-06` | `3.21e-19 (FC better)` | `1.39e-19 (FC better)` |
| `SciPy-SLSQP` | 20 | 20 | 100.0% | 54.0 (47.0) | 292.10 (379.25) | `6.75e-07` | `9.99e-16` | `1.10e-12` | `2.46e-03 (FC better)` | `9.27e-01 (FC better)` |
| `SciPy-trust-constr` | 20 | 10 | 50.0% | 2572.0 (4904.0) | 1920.18 (2714.34) | `1.42e-05` | `1.11e-16` | `6.18e-15` | `8.61e-05 (FC better)` | `1.34e-04 (FC better)` |

## 5. Third Blind Holdout Suite Aggregate Summary (`Seeds 301..310`)

Aggregate metrics across the third blind holdout suite (`301..310`) under the identical frozen `FCOptions(gtol=1.0e-6, max_eval=5000)` configuration:

| Solver | Evaluated Runs | Solved Runs | Solve Rate | Penalized Evals Med (IQR) | Wall Time ms Med (IQR) | Median KKT Ref | Max Lin Viol | Median Orth Viol | Wilcoxon Evals p (2-sided) | Wilcoxon KKT p (2-sided) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `fortchuck` | 110 | 110 | 100.0% | 469.5 (1488.5) | 173.26 (1127.03) | `7.76e-07` | `2.96e-16` | `3.03e-14` | `—` | `—` |
| `fortchuck-ABB` | 110 | 84 | 76.4% | 975.0 (4832.5) | 331.68 (1767.21) | `8.44e-07` | `1.11e-16` | `3.61e-14` | `3.80e-19 (FC better)` | `4.10e-06 (FC better)` |
| `Alternating-BCD` | 110 | 53 | 48.2% | 5000.0 (4648.0) | 2998.09 (3434.13) | `1.66e-05` | `3.33e-16` | `1.15e-15` | `1.84e-18 (FC better)` | `3.95e-16 (FC better)` |
| `pymanopt-Warped-CG` | 110 | 25 | 22.7% | 5000.0 (0.0) | 5365.50 (4614.83) | `1.75e-03` | `9.80e-05` | `1.13e-15` | `8.74e-20 (FC better)` | `4.13e-19 (FC better)` |
| `pymanopt-TrustRegions` | 110 | 28 | 25.5% | 5000.0 (511.5) | 3426.45 (2841.60) | `3.76e-03` | `5.74e-05` | `2.64e-15` | `8.74e-20 (FC better)` | `5.77e-17 (FC better)` |
| `SciPy-L-BFGS-B` | 110 | 10 | 9.1% | 5000.0 (0.0) | 3012.95 (3673.92) | `2.01e-04` | `1.02e-06` | `4.87e-06` | `3.85e-19 (FC better)` | `1.47e-19 (FC better)` |
| `SciPy-SLSQP` | 20 | 20 | 100.0% | 58.0 (43.5) | 258.50 (393.37) | `5.45e-07` | `1.44e-15` | `1.03e-12` | `3.57e-03 (FC better)` | `9.56e-01 (FC better)` |
| `SciPy-trust-constr` | 20 | 10 | 50.0% | 2564.0 (4915.8) | 2010.10 (2971.19) | `1.28e-06` | `2.22e-16` | `1.70e-14` | `8.76e-05 (FC better)` | `1.34e-05 (FC better)` |

## 6. Virgin Fourth Blind Holdout Suite Aggregate Summary (`Seeds 401..410`)

Aggregate metrics across the virgin fourth blind holdout suite (`401..410`), held completely untouched until all `src/*.f90` edits were frozen, under the identical `FCOptions(gtol=1.0e-6, max_eval=5000)` configuration:

| Solver | Evaluated Runs | Solved Runs | Solve Rate | Penalized Evals Med (IQR) | Wall Time ms Med (IQR) | Median KKT Ref | Max Lin Viol | Median Orth Viol | Wilcoxon Evals p (2-sided) | Wilcoxon KKT p (2-sided) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `fortchuck` | 110 | 110 | 100.0% | 472.5 (1552.5) | 240.84 (1258.38) | `7.75e-07` | `1.48e-16` | `2.60e-14` | `—` | `—` |
| `fortchuck-ABB` | 110 | 81 | 73.6% | 775.5 (4901.0) | 356.97 (1993.04) | `8.14e-07` | `1.11e-16` | `3.25e-14` | `1.57e-18 (FC better)` | `1.10e-05 (FC better)` |
| `Alternating-BCD` | 110 | 53 | 48.2% | 5000.0 (4629.2) | 2082.69 (3119.98) | `2.44e-05` | `2.22e-16` | `1.06e-15` | `3.39e-17 (FC better)` | `1.34e-15 (FC better)` |
| `pymanopt-Warped-CG` | 110 | 25 | 22.7% | 5000.0 (0.0) | 4496.25 (3352.68) | `1.70e-03` | `8.62e-05` | `1.12e-15` | `8.74e-20 (FC better)` | `2.21e-19 (FC better)` |
| `pymanopt-TrustRegions` | 110 | 27 | 24.5% | 5000.0 (0.0) | 2938.07 (1800.21) | `1.93e-03` | `4.31e-05` | `2.45e-15` | `8.74e-20 (FC better)` | `7.06e-17 (FC better)` |
| `SciPy-L-BFGS-B` | 110 | 10 | 9.1% | 5000.0 (0.0) | 2299.17 (3663.54) | `1.13e-04` | `1.26e-07` | `5.47e-06` | `2.15e-19 (FC better)` | `1.28e-19 (FC better)` |
| `SciPy-SLSQP` | 20 | 20 | 100.0% | 59.5 (42.2) | 258.87 (348.08) | `6.79e-07` | `1.11e-15` | `1.76e-12` | `5.69e-03 (FC better)` | `4.09e-01 (FC worse)` |
| `SciPy-trust-constr` | 20 | 10 | 50.0% | 2553.5 (4914.2) | 1634.09 (2595.63) | `1.42e-05` | `1.11e-16` | `4.71e-15` | `8.78e-05 (FC better)` | `6.29e-05 (FC better)` |

## 7. Exact Global Optimality Certificate (`B6` Ky Fan QP & `B7B` Procrustes)

Problems `B6` and `B7B` possess exact analytical global minima $f^\ast$ certified by Ky Fan's extremal trace inequality and Kabsch singular value decomposition:

| Suite | Problem | Certified Global Optimum | Solver | Solve Rate | Median Abs Gap | Max Abs Gap | Median Signed Gap | Median KKT Ref |
| :--- | :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| Primary (`101..110`) | `B6` | `-10.445509834730` | `fortchuck` | 10/10 (100%) | `9.73e-13` | `3.02e-12` | `9.73e-13` | `9.40e-07` |
| Primary (`101..110`) | `B6` | `-10.445509834730` | `fortchuck-ABB` | 6/10 (60%) | `1.13e-10` | `7.98e-10` | `1.13e-10` | `7.63e-07` |
| Primary (`101..110`) | `B6` | `-10.445509834730` | `Alternating-BCD` | 0/10 (0%) | `2.41e-03` | `6.52e-03` | `2.41e-03` | `2.09e-02` |
| Primary (`101..110`) | `B6` | `-10.445509834730` | `pymanopt-Warped-CG` | 0/10 (0%) | `5.18e-04` | `2.31e-03` | `5.18e-04` | `5.17e-03` |
| Primary (`101..110`) | `B6` | `-10.445509834730` | `pymanopt-TrustRegions` | 0/10 (0%) | `7.77e-05` | `2.93e-04` | `7.77e-05` | `1.80e-03` |
| Primary (`101..110`) | `B6` | `-10.445509834730` | `SciPy-L-BFGS-B` | 0/10 (0%) | `1.07e-04` | `2.26e-03` | `-1.03e-04` | `2.65e-03` |
| Primary (`101..110`) | `B6` | `-10.445509834730` | `SciPy-SLSQP` | SKIPPED | N/A | N/A | N/A | N/A |
| Primary (`101..110`) | `B6` | `-10.445509834730` | `SciPy-trust-constr` | SKIPPED | N/A | N/A | N/A | N/A |
| Primary (`101..110`) | `B7B` | `-66.949417348256` | `fortchuck` | 10/10 (100%) | `7.60e-13` | `1.31e-12` | `-7.60e-13` | `6.88e-07` |
| Primary (`101..110`) | `B7B` | `-66.949417348256` | `fortchuck-ABB` | 10/10 (100%) | `1.95e-12` | `4.16e-12` | `1.95e-12` | `6.98e-07` |
| Primary (`101..110`) | `B7B` | `-66.949417348256` | `Alternating-BCD` | 10/10 (100%) | `2.98e-13` | `2.15e-12` | `2.98e-13` | `8.59e-07` |
| Primary (`101..110`) | `B7B` | `-66.949417348256` | `pymanopt-Warped-CG` | 10/10 (100%) | `2.98e-13` | `2.15e-12` | `2.98e-13` | `8.59e-07` |
| Primary (`101..110`) | `B7B` | `-66.949417348256` | `pymanopt-TrustRegions` | 10/10 (100%) | `2.13e-13` | `2.27e-13` | `-2.13e-13` | `1.17e-08` |
| Primary (`101..110`) | `B7B` | `-66.949417348256` | `SciPy-L-BFGS-B` | 0/10 (0%) | `1.31e-01` | `1.31e-01` | `1.31e-01` | `1.42e-03` |
| Primary (`101..110`) | `B7B` | `-66.949417348256` | `SciPy-SLSQP` | SKIPPED | N/A | N/A | N/A | N/A |
| Primary (`101..110`) | `B7B` | `-66.949417348256` | `SciPy-trust-constr` | SKIPPED | N/A | N/A | N/A | N/A |
| Holdout (`201..210`) | `B6` | `-10.445509834730` | `fortchuck` | 10/10 (100%) | `1.24e-12` | `3.39e-12` | `1.24e-12` | `9.03e-07` |
| Holdout (`201..210`) | `B6` | `-10.445509834730` | `fortchuck-ABB` | 8/10 (80%) | `1.09e-10` | `6.94e-10` | `1.09e-10` | `7.11e-07` |
| Holdout (`201..210`) | `B6` | `-10.445509834730` | `Alternating-BCD` | 0/10 (0%) | `1.88e-03` | `4.10e-03` | `1.88e-03` | `9.64e-03` |
| Holdout (`201..210`) | `B6` | `-10.445509834730` | `pymanopt-Warped-CG` | 0/10 (0%) | `2.52e-04` | `2.40e-01` | `2.52e-04` | `1.65e-03` |
| Holdout (`201..210`) | `B6` | `-10.445509834730` | `pymanopt-TrustRegions` | 0/10 (0%) | `7.27e-05` | `1.26e-04` | `7.27e-05` | `1.47e-03` |
| Holdout (`201..210`) | `B6` | `-10.445509834730` | `SciPy-L-BFGS-B` | 0/10 (0%) | `1.01e-04` | `9.56e-04` | `-8.34e-05` | `3.25e-03` |
| Holdout (`201..210`) | `B6` | `-10.445509834730` | `SciPy-SLSQP` | SKIPPED | N/A | N/A | N/A | N/A |
| Holdout (`201..210`) | `B6` | `-10.445509834730` | `SciPy-trust-constr` | SKIPPED | N/A | N/A | N/A | N/A |
| Holdout (`201..210`) | `B7B` | `-66.949417348256` | `fortchuck` | 10/10 (100%) | `6.25e-13` | `1.05e-12` | `-6.25e-13` | `6.40e-07` |
| Holdout (`201..210`) | `B7B` | `-66.949417348256` | `fortchuck-ABB` | 10/10 (100%) | `1.48e-12` | `3.37e-12` | `1.48e-12` | `6.79e-07` |
| Holdout (`201..210`) | `B7B` | `-66.949417348256` | `Alternating-BCD` | 9/10 (90%) | `2.49e-13` | `1.39e-12` | `2.49e-13` | `9.00e-07` |
| Holdout (`201..210`) | `B7B` | `-66.949417348256` | `pymanopt-Warped-CG` | 9/10 (90%) | `2.49e-13` | `1.39e-12` | `2.49e-13` | `9.00e-07` |
| Holdout (`201..210`) | `B7B` | `-66.949417348256` | `pymanopt-TrustRegions` | 10/10 (100%) | `2.13e-13` | `2.42e-13` | `-2.13e-13` | `2.65e-09` |
| Holdout (`201..210`) | `B7B` | `-66.949417348256` | `SciPy-L-BFGS-B` | 0/10 (0%) | `1.50e-03` | `1.31e-01` | `-1.50e-03` | `1.42e-03` |
| Holdout (`201..210`) | `B7B` | `-66.949417348256` | `SciPy-SLSQP` | SKIPPED | N/A | N/A | N/A | N/A |
| Holdout (`201..210`) | `B7B` | `-66.949417348256` | `SciPy-trust-constr` | SKIPPED | N/A | N/A | N/A | N/A |
| Blind3 (`301..310`) | `B6` | `-10.445509834730` | `fortchuck` | 10/10 (100%) | `1.48e-12` | `3.60e-12` | `1.48e-12` | `9.02e-07` |
| Blind3 (`301..310`) | `B6` | `-10.445509834730` | `fortchuck-ABB` | 9/10 (90%) | `1.09e-10` | `5.30e-10` | `1.09e-10` | `7.22e-07` |
| Blind3 (`301..310`) | `B6` | `-10.445509834730` | `Alternating-BCD` | 0/10 (0%) | `1.79e-03` | `5.71e-03` | `1.79e-03` | `1.26e-02` |
| Blind3 (`301..310`) | `B6` | `-10.445509834730` | `pymanopt-Warped-CG` | 0/10 (0%) | `2.08e-04` | `6.25e+00` | `2.08e-04` | `1.60e-03` |
| Blind3 (`301..310`) | `B6` | `-10.445509834730` | `pymanopt-TrustRegions` | 0/10 (0%) | `8.20e-05` | `3.76e-04` | `8.20e-05` | `2.53e-03` |
| Blind3 (`301..310`) | `B6` | `-10.445509834730` | `SciPy-L-BFGS-B` | 0/10 (0%) | `9.94e-05` | `1.93e-03` | `-5.81e-05` | `3.67e-03` |
| Blind3 (`301..310`) | `B6` | `-10.445509834730` | `SciPy-SLSQP` | SKIPPED | N/A | N/A | N/A | N/A |
| Blind3 (`301..310`) | `B6` | `-10.445509834730` | `SciPy-trust-constr` | SKIPPED | N/A | N/A | N/A | N/A |
| Blind3 (`301..310`) | `B7B` | `-66.949417348256` | `fortchuck` | 10/10 (100%) | `5.61e-13` | `1.22e-12` | `-5.61e-13` | `6.92e-07` |
| Blind3 (`301..310`) | `B7B` | `-66.949417348256` | `fortchuck-ABB` | 10/10 (100%) | `1.49e-12` | `3.92e-12` | `1.49e-12` | `7.26e-07` |
| Blind3 (`301..310`) | `B7B` | `-66.949417348256` | `Alternating-BCD` | 10/10 (100%) | `8.60e-13` | `2.02e-12` | `8.60e-13` | `9.06e-07` |
| Blind3 (`301..310`) | `B7B` | `-66.949417348256` | `pymanopt-Warped-CG` | 10/10 (100%) | `8.60e-13` | `2.02e-12` | `8.60e-13` | `9.06e-07` |
| Blind3 (`301..310`) | `B7B` | `-66.949417348256` | `pymanopt-TrustRegions` | 10/10 (100%) | `2.13e-13` | `2.27e-13` | `-2.13e-13` | `2.06e-08` |
| Blind3 (`301..310`) | `B7B` | `-66.949417348256` | `SciPy-L-BFGS-B` | 0/10 (0%) | `6.63e-02` | `1.31e-01` | `6.48e-02` | `1.42e-03` |
| Blind3 (`301..310`) | `B7B` | `-66.949417348256` | `SciPy-SLSQP` | SKIPPED | N/A | N/A | N/A | N/A |
| Blind3 (`301..310`) | `B7B` | `-66.949417348256` | `SciPy-trust-constr` | SKIPPED | N/A | N/A | N/A | N/A |
| Blind4 (`401..410`) | `B6` | `-10.445509834730` | `fortchuck` | 10/10 (100%) | `1.37e-12` | `2.67e-12` | `1.37e-12` | `9.32e-07` |
| Blind4 (`401..410`) | `B6` | `-10.445509834730` | `fortchuck-ABB` | 8/10 (80%) | `1.10e-10` | `3.44e-09` | `1.10e-10` | `7.06e-07` |
| Blind4 (`401..410`) | `B6` | `-10.445509834730` | `Alternating-BCD` | 0/10 (0%) | `9.02e-04` | `4.19e-03` | `9.02e-04` | `1.63e-02` |
| Blind4 (`401..410`) | `B6` | `-10.445509834730` | `pymanopt-Warped-CG` | 0/10 (0%) | `2.71e-04` | `8.74e-04` | `2.71e-04` | `2.09e-03` |
| Blind4 (`401..410`) | `B6` | `-10.445509834730` | `pymanopt-TrustRegions` | 0/10 (0%) | `5.94e-05` | `3.58e-04` | `5.94e-05` | `1.52e-03` |
| Blind4 (`401..410`) | `B6` | `-10.445509834730` | `SciPy-L-BFGS-B` | 0/10 (0%) | `1.07e-04` | `1.96e-03` | `-1.49e-05` | `4.91e-03` |
| Blind4 (`401..410`) | `B6` | `-10.445509834730` | `SciPy-SLSQP` | SKIPPED | N/A | N/A | N/A | N/A |
| Blind4 (`401..410`) | `B6` | `-10.445509834730` | `SciPy-trust-constr` | SKIPPED | N/A | N/A | N/A | N/A |
| Blind4 (`401..410`) | `B7B` | `-66.949417348256` | `fortchuck` | 10/10 (100%) | `6.04e-13` | `9.38e-13` | `-6.04e-13` | `6.84e-07` |
| Blind4 (`401..410`) | `B7B` | `-66.949417348256` | `fortchuck-ABB` | 10/10 (100%) | `1.25e-12` | `2.22e-12` | `1.25e-12` | `6.66e-07` |
| Blind4 (`401..410`) | `B7B` | `-66.949417348256` | `Alternating-BCD` | 10/10 (100%) | `2.42e-13` | `2.80e-12` | `2.06e-13` | `8.06e-07` |
| Blind4 (`401..410`) | `B7B` | `-66.949417348256` | `pymanopt-Warped-CG` | 10/10 (100%) | `2.42e-13` | `2.80e-12` | `2.06e-13` | `8.06e-07` |
| Blind4 (`401..410`) | `B7B` | `-66.949417348256` | `pymanopt-TrustRegions` | 10/10 (100%) | `2.13e-13` | `2.27e-13` | `-2.13e-13` | `6.43e-10` |
| Blind4 (`401..410`) | `B7B` | `-66.949417348256` | `SciPy-L-BFGS-B` | 0/10 (0%) | `1.50e-03` | `1.31e-01` | `-1.50e-03` | `1.42e-03` |
| Blind4 (`401..410`) | `B7B` | `-66.949417348256` | `SciPy-SLSQP` | SKIPPED | N/A | N/A | N/A | N/A |
| Blind4 (`401..410`) | `B7B` | `-66.949417348256` | `SciPy-trust-constr` | SKIPPED | N/A | N/A | N/A | N/A |

## 8. `v0.1.0` to `v0.2.1` Progression and Zero-FFI Native CLI Telemetry

### `v0.1.0` vs `v0.2.1` Progression (`Seeds 101..110 + 201..210` Shared Cohort & `40`-Seed Pool)

Comparison of frozen out-of-the-box `fortchuck` (`FCOptions(gtol=1.0e-6, max_eval=5000)`) across the 20 shared seeds (`101..110` + `201..210`) from `v0.1.0` alongside the full `40`-seed `v0.2.1` pool:

| Problem | `v0.1.0` Solved (20) | `v0.1.0` Raw Evals Med | `v0.1.0` Wall ms Med | `v0.2.1` Solved (20) | `v0.2.1` Raw Evals Med (20) | `v0.2.1` Wall ms Med (20) | `v0.2.1` Wall Speedup | `v0.2.1` Solved (40 Pool) | `v0.2.1` Max Lin Viol (40) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `B1` | 20/20 (100%) | 2084.5 | 1813.88 | 20/20 (100%) | 2241.5 | 1517.16 | `1.20x` | **40/40 (100%)** | `0.00e+00` |
| `B2A` | 15/20 (75%) | 580.0 | 171.27 | 20/20 (100%) | 593.0 | 164.47 | `1.04x` | **40/40 (100%)** | `0.00e+00` |
| `B2B` | 20/20 (100%) | 111.5 | 41.50 | 20/20 (100%) | 107.0 | 35.42 | `1.17x` | **40/40 (100%)** | `0.00e+00` |
| `B3` | 20/20 (100%) | 44.0 | 11.99 | 20/20 (100%) | 45.5 | 12.95 | `0.93x` | **40/40 (100%)** | `0.00e+00` |
| `B4` | 20/20 (100%) | 1762.0 | 1424.73 | 20/20 (100%) | 1718.0 | 1267.76 | `1.12x` | **40/40 (100%)** | `0.00e+00` |
| `B5` | 17/20 (85%) | 1233.0 | 918.65 | 20/20 (100%) | 1219.5 | 736.51 | `1.25x` | **40/40 (100%)** | `0.00e+00` |
| `B6` | 20/20 (100%) | 1902.0 | 1801.48 | 20/20 (100%) | 1940.0 | 1398.63 | `1.29x` | **40/40 (100%)** | `0.00e+00` |
| `B7A` | 20/20 (100%) | 67.5 | 8.02 | 20/20 (100%) | 67.5 | 10.95 | `0.73x` | **40/40 (100%)** | `0.00e+00` |
| `B7B` | 20/20 (100%) | 504.0 | 619.29 | 20/20 (100%) | 496.0 | 439.90 | `1.41x` | **40/40 (100%)** | `0.00e+00` |
| `B8A` | — (`v0.2.0` new) | — | — | 20/20 (100%) | 152.5 | 133.42 | — | **40/40 (100%)** | `2.96e-16` |
| `B8B` | — (`v0.2.0` new) | — | — | 20/20 (100%) | 36.0 | 8.21 | — | **40/40 (100%)** | `1.11e-16` |

### Zero-FFI Native Fortran 2018 CLI Telemetry (`app/fortchuck_bench.f90`)

Pure Fortran 2018 release timings (`gfortran -cpp -O3 -march=native -funroll-loops -fopenmp`, `0` Python `ctypes` callback overhead, `Seeds 101..110`) recorded by `fpm run fortchuck-bench` across Windows (MinGW-w64 UCRT) and WSL2 Ubuntu 26.04 (Pure F2018 & OpenBLAS `-DFORTCHUCK_USE_BLAS`):

| Native Workload (`fortchuck-bench`) | Windows Pure F2018 | WSL2 Pure F2018 | WSL2 OpenBLAS (`-DFORTCHUCK_USE_BLAS`) | Exact Accuracy / Feasibility Certificate |
| :--- | :---: | :---: | :---: | :--- |
| `[1]` Batched Panel Transport (`St(200,8)`, `K_batch=16`, `4000` reps) | `872.00 ms` vs `857.31 ms` (`0.98x`) | `1064.39 ms` vs `1113.94 ms` (`1.05x`) | **`358.99 ms` vs `454.35 ms` (`1.27x`)** | `max_err = 0.00e+00` vs `16` single-vector calls |
| `[1]` Batched Panel Transport (`St(24,24)`, `K_batch=16`, `4000` reps) | `280.74 ms` vs `281.08 ms` (`1.00x`) | `324.28 ms` vs `327.19 ms` (`1.01x`) | `297.23 ms` vs `282.66 ms` (`0.95x`) | `max_err = 0.00e+00` vs `16` single-vector calls |
| `[2]` `B6` Box-Coupled Ky Fan QP (`10/10` solved) | `973.56 ms` (`1976.0` evals) | `1215.56 ms` (`1982.0` iters) | **`659.38 ms`** | `max |f - f*| <= 5.98e-12`, `max_KKT <= 9.96e-07` |
| `[3]` `B7B` Tall Brockett + Procrustes (`10/10` solved) | `374.51 ms` (`m=7`) / `212.91 ms` (`m=0`) | `1395.26 ms` (`m=7`) / `728.05 ms` (`m=0`) | **`249.95 ms` (`m=7`) / `150.92 ms` (`m=0`)** | `max_KKT <= 9.99e-07`, `max_orth <= 9.67e-14` |
| `[4]` `B8A` Hard-Trace Ensemble DFT (`10/10` solved) | `119.23 ms` (`214.0` evals) | `146.99 ms` (`213.0` iters) | **`68.19 ms` (`208.0` iters, `214.0` evals)** | `max |1^T z - 5| <= 1.52e-13` (`1.78e-15` WSL2), `max_KKT <= 1.00e-06` |

## 9. Per-Problem Breakdown — Pooled Multi-Suite (`40` Seeds)

### `B1`: Ensemble Kohn-Sham DFT Fractional Occupations

- **Regime**: `z in [0,1]^10 x St(120,10) (Branch A, metallic orbital-occupation coupling)` (`d = 10`, `R = 1`, `N_tot = 1210`, intrinsic dim `1155`, FD gradient max rel err `2.104e-10`)

| Solver | Solve Rate | Raw Evals Med (IQR) | Penalized Evals Med (IQR) | Wall ms Med (IQR) | Median KKT Ref | Median Box Viol | Median Lin Viol | Median Orth Viol | Median Objective | Wilcoxon Evals p |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `fortchuck` | 40/40 (100%) | 2176.5 (734.8) | 2176.5 (734.8) | 1517.16 (492.17) | `8.84e-07` | `0.00e+00` | `0.00e+00` | `3.40e-14` | `-4.377192` | `—` |
| `fortchuck-ABB` | 1/40 (2%) | 5000.0 (0.0) | 5000.0 (0.0) | 1813.44 (237.95) | `1.82e-05` | `0.00e+00` | `0.00e+00` | `5.20e-14` | `-4.368467` | `1.82e-12 (FC better)` |
| `Alternating-BCD` | 10/40 (25%) | 5000.0 (1132.5) | 5000.0 (1132.5) | 3247.29 (1224.28) | `5.29e-04` | `0.00e+00` | `0.00e+00` | `1.27e-15` | `-4.349272` | `2.45e-05 (FC better)` |
| `pymanopt-Warped-CG` | 0/40 (0%) | 5000.0 (0.0) | 5000.0 (0.0) | 5045.87 (1432.14) | `5.76e-04` | `0.00e+00` | `0.00e+00` | `1.32e-15` | `-4.386121` | `1.82e-12 (FC better)` |
| `pymanopt-TrustRegions` | 0/40 (0%) | 5000.0 (0.0) | 5000.0 (0.0) | 3282.14 (612.02) | `7.60e-03` | `0.00e+00` | `0.00e+00` | `1.28e-13` | `-4.386093` | `1.82e-12 (FC better)` |
| `SciPy-L-BFGS-B` | 0/40 (0%) | 5000.0 (0.0) | 5000.0 (0.0) | 4664.35 (449.99) | `6.07e-04` | `0.00e+00` | `0.00e+00` | `3.21e-06` | `-4.368302` | `1.82e-12 (FC better)` |
| `SciPy-SLSQP` | SKIP (`N_tot > 450`) | — | — | — | — | — | — | — | — | — |
| `SciPy-trust-constr` | SKIP (`N_tot > 450`) | — | — | — | — | — | — | — | — | — |

### `B2A`: Amplitude-Limited Blind Source Separation (ICA)

- **Regime**: `Z in [-2,2]^{5x150} (d=750) x St(20,5) (Branch A, nonconvex log-cosh)` (`d = 750`, `R = 1`, `N_tot = 850`, intrinsic dim `835`, FD gradient max rel err `2.877e-09`)

| Solver | Solve Rate | Raw Evals Med (IQR) | Penalized Evals Med (IQR) | Wall ms Med (IQR) | Median KKT Ref | Median Box Viol | Median Lin Viol | Median Orth Viol | Median Objective | Wilcoxon Evals p |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `fortchuck` | 40/40 (100%) | 585.0 (272.2) | 585.0 (272.2) | 172.03 (79.80) | `8.52e-07` | `0.00e+00` | `0.00e+00` | `4.05e-14` | `-307.697350` | `—` |
| `fortchuck-ABB` | 25/40 (62%) | 3903.5 (1966.8) | 4018.5 (1834.2) | 836.84 (384.48) | `9.60e-07` | `0.00e+00` | `0.00e+00` | `5.72e-14` | `-307.433520` | `1.82e-12 (FC better)` |
| `Alternating-BCD` | 6/40 (15%) | 5000.0 (0.0) | 5000.0 (0.0) | 3919.56 (1024.61) | `1.71e-03` | `0.00e+00` | `0.00e+00` | `6.75e-16` | `-306.622148` | `3.57e-08 (FC better)` |
| `pymanopt-Warped-CG` | 0/40 (0%) | 5000.0 (0.0) | 5000.0 (0.0) | 5066.85 (1420.52) | `4.79e-01` | `0.00e+00` | `0.00e+00` | `6.85e-16` | `-305.867967` | `3.57e-08 (FC better)` |
| `pymanopt-TrustRegions` | 0/40 (0%) | 5000.0 (0.0) | 5000.0 (0.0) | 3197.20 (749.52) | `5.76e-01` | `0.00e+00` | `0.00e+00` | `3.11e-11` | `-306.387353` | `3.57e-08 (FC better)` |
| `SciPy-L-BFGS-B` | 0/40 (0%) | 5000.0 (0.0) | 5000.0 (0.0) | 3888.54 (626.03) | `5.47e-02` | `0.00e+00` | `0.00e+00` | `2.44e-05` | `-308.161036` | `3.57e-08 (FC better)` |
| `SciPy-SLSQP` | SKIP (`N_tot > 450`) | — | — | — | — | — | — | — | — | — |
| `SciPy-trust-constr` | SKIP (`N_tot > 450`) | — | — | — | — | — | — | — | — | — |

### `B2B`: Smooth Penalty-Split Non-Negative Sparse PCA

- **Regime**: `Z in [0,5]^{120x6} (d=720) x St(120,6) (Branch A, ~80% clamped Z=0, FM-2)` (`d = 720`, `R = 1`, `N_tot = 1440`, intrinsic dim `1419`, FD gradient max rel err `1.484e-08`)

| Solver | Solve Rate | Raw Evals Med (IQR) | Penalized Evals Med (IQR) | Wall ms Med (IQR) | Median KKT Ref | Median Box Viol | Median Lin Viol | Median Orth Viol | Median Objective | Wilcoxon Evals p |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `fortchuck` | 40/40 (100%) | 104.5 (29.8) | 104.5 (29.8) | 35.56 (12.51) | `6.40e-07` | `0.00e+00` | `0.00e+00` | `1.11e-14` | `8.516674` | `—` |
| `fortchuck-ABB` | 40/40 (100%) | 735.5 (999.5) | 735.5 (999.5) | 144.39 (194.34) | `9.13e-07` | `0.00e+00` | `0.00e+00` | `3.22e-14` | `8.301033` | `1.82e-12 (FC better)` |
| `Alternating-BCD` | 0/40 (0%) | 4089.0 (1960.0) | 5000.0 (0.0) | 3414.21 (1461.90) | `2.71e-04` | `0.00e+00` | `0.00e+00` | `1.11e-15` | `8.134383` | `3.55e-08 (FC better)` |
| `pymanopt-Warped-CG` | 19/40 (48%) | 3897.5 (3861.0) | 5000.0 (3240.8) | 3917.97 (3665.62) | `7.84e-05` | `0.00e+00` | `0.00e+00` | `1.08e-15` | `13.816678` | `3.56e-08 (FC better)` |
| `pymanopt-TrustRegions` | 26/40 (65%) | 3413.5 (2331.5) | 3413.5 (2331.5) | 2260.01 (1308.29) | `8.59e-07` | `0.00e+00` | `0.00e+00` | `1.03e-15` | `8.313161` | `3.56e-08 (FC better)` |
| `SciPy-L-BFGS-B` | 0/40 (0%) | 490.5 (522.5) | 5000.0 (0.0) | 404.44 (399.63) | `3.29e-05` | `0.00e+00` | `0.00e+00` | `5.57e-06` | `10.442436` | `3.55e-08 (FC better)` |
| `SciPy-SLSQP` | SKIP (`N_tot > 450`) | — | — | — | — | — | — | — | — | — |
| `SciPy-trust-constr` | SKIP (`N_tot > 450`) | — | — | — | — | — | — | — | — | — |

### `B3`: Multi-Camera Anisotropic Bounded-Scale Rig Wahba

- **Regime**: `z in B^15 x St(3,3)^4 (Branch B SO(3)^4, R=4, kappa(W)=10^3)` (`d = 15`, `R = 4`, `N_tot = 51`, intrinsic dim `27`, FD gradient max rel err `7.213e-11`)

| Solver | Solve Rate | Raw Evals Med (IQR) | Penalized Evals Med (IQR) | Wall ms Med (IQR) | Median KKT Ref | Median Box Viol | Median Lin Viol | Median Orth Viol | Median Objective | Wilcoxon Evals p |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `fortchuck` | 40/40 (100%) | 46.5 (11.0) | 46.5 (11.0) | 13.19 (4.58) | `5.01e-07` | `0.00e+00` | `0.00e+00` | `5.35e-15` | `2.176801` | `—` |
| `fortchuck-ABB` | 40/40 (100%) | 82.0 (13.5) | 82.0 (13.5) | 23.55 (5.29) | `5.88e-07` | `0.00e+00` | `0.00e+00` | `6.82e-15` | `2.176801` | `3.54e-08 (FC better)` |
| `Alternating-BCD` | 38/40 (95%) | 315.0 (22.0) | 315.5 (23.2) | 455.35 (116.17) | `8.71e-07` | `0.00e+00` | `0.00e+00` | `6.64e-16` | `2.176801` | `3.55e-08 (FC better)` |
| `pymanopt-Warped-CG` | 40/40 (100%) | 865.5 (294.8) | 865.5 (294.8) | 1573.38 (736.69) | `8.18e-07` | `0.00e+00` | `0.00e+00` | `7.29e-16` | `2.176801` | `3.57e-08 (FC better)` |
| `pymanopt-TrustRegions` | 40/40 (100%) | 651.0 (91.5) | 651.0 (91.5) | 742.48 (165.53) | `6.39e-07` | `0.00e+00` | `0.00e+00` | `6.81e-16` | `2.176801` | `3.56e-08 (FC better)` |
| `SciPy-L-BFGS-B` | 0/40 (0%) | 810.0 (107.0) | 5000.0 (0.0) | 803.43 (173.50) | `2.27e-05` | `0.00e+00` | `0.00e+00` | `4.83e-06` | `2.176772` | `3.49e-08 (FC better)` |
| `SciPy-SLSQP` | 40/40 (100%) | 78.5 (13.8) | 78.5 (13.8) | 81.19 (20.40) | `3.49e-07` | `0.00e+00` | `0.00e+00` | `2.77e-14` | `2.176801` | `3.52e-08 (FC better)` |
| `SciPy-trust-constr` | 40/40 (100%) | 87.0 (21.5) | 87.0 (21.5) | 469.22 (188.33) | `8.32e-07` | `0.00e+00` | `0.00e+00` | `4.01e-15` | `2.176802` | `3.55e-08 (FC better)` |

### `B4`: Bounded-Variance ML Common Principal Components

- **Regime**: `z in [0.05,20]^150 x St(30,30) (Branch B O(30), log barrier, 1.6e5 scale ratio)` (`d = 150`, `R = 1`, `N_tot = 1050`, intrinsic dim `585`, FD gradient max rel err `6.833e-10`)

| Solver | Solve Rate | Raw Evals Med (IQR) | Penalized Evals Med (IQR) | Wall ms Med (IQR) | Median KKT Ref | Median Box Viol | Median Lin Viol | Median Orth Viol | Median Objective | Wilcoxon Evals p |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `fortchuck` | 40/40 (100%) | 1760.0 (208.5) | 1760.0 (208.5) | 1366.51 (218.61) | `6.58e-07` | `0.00e+00` | `0.00e+00` | `5.51e-14` | `17.596323` | `—` |
| `fortchuck-ABB` | 35/40 (88%) | 4391.0 (712.5) | 4391.0 (712.5) | 2261.89 (279.87) | `6.99e-07` | `0.00e+00` | `0.00e+00` | `5.57e-14` | `17.596323` | `3.57e-08 (FC better)` |
| `Alternating-BCD` | 0/40 (0%) | 5000.0 (0.0) | 5000.0 (0.0) | 4209.16 (998.90) | `5.31e-04` | `0.00e+00` | `0.00e+00` | `3.15e-15` | `17.596323` | `3.57e-08 (FC better)` |
| `pymanopt-Warped-CG` | 0/40 (0%) | 5000.0 (0.0) | 5000.0 (0.0) | 6019.23 (1589.91) | `7.22e-03` | `0.00e+00` | `0.00e+00` | `3.09e-15` | `17.598992` | `3.57e-08 (FC better)` |
| `pymanopt-TrustRegions` | 0/40 (0%) | 5000.0 (0.0) | 5000.0 (0.0) | 3932.39 (1161.60) | `4.78e-03` | `0.00e+00` | `0.00e+00` | `8.90e-14` | `17.597396` | `3.57e-08 (FC better)` |
| `SciPy-L-BFGS-B` | 0/40 (0%) | 5000.0 (0.0) | 5000.0 (0.0) | 5102.82 (638.05) | `5.66e-01` | `0.00e+00` | `0.00e+00` | `7.08e-06` | `20.711426` | `3.57e-08 (FC better)` |
| `SciPy-SLSQP` | SKIP (`N_tot > 450`) | — | — | — | — | — | — | — | — | — |
| `SciPy-trust-constr` | SKIP (`N_tot > 450`) | — | — | — | — | — | — | — | — | — |

### `B5`: Bounded Factor Correlation Matrix Calibration

- **Regime**: `z in B^110 x St(100,10) (Branch A, quartic valley + clamped psi)` (`d = 110`, `R = 1`, `N_tot = 1110`, intrinsic dim `1055`, FD gradient max rel err `8.432e-10`)

| Solver | Solve Rate | Raw Evals Med (IQR) | Penalized Evals Med (IQR) | Wall ms Med (IQR) | Median KKT Ref | Median Box Viol | Median Lin Viol | Median Orth Viol | Median Objective | Wilcoxon Evals p |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `fortchuck` | 40/40 (100%) | 1192.0 (207.5) | 1192.0 (207.5) | 780.72 (129.90) | `9.35e-07` | `0.00e+00` | `0.00e+00` | `5.07e-14` | `34.668813` | `—` |
| `fortchuck-ABB` | 0/40 (0%) | 5000.0 (0.0) | 5000.0 (0.0) | 1898.82 (283.14) | `1.17e-03` | `0.00e+00` | `0.00e+00` | `3.84e-14` | `34.668874` | `1.82e-12 (FC better)` |
| `Alternating-BCD` | 0/40 (0%) | 5000.0 (0.0) | 5000.0 (0.0) | 3874.72 (385.43) | `5.71e-03` | `0.00e+00` | `0.00e+00` | `1.10e-15` | `34.669135` | `1.82e-12 (FC better)` |
| `pymanopt-Warped-CG` | 0/40 (0%) | 5000.0 (0.0) | 5000.0 (0.0) | 5691.08 (925.53) | `2.46e-02` | `0.00e+00` | `0.00e+00` | `1.28e-15` | `34.671835` | `1.82e-12 (FC better)` |
| `pymanopt-TrustRegions` | 0/40 (0%) | 5000.0 (0.0) | 5000.0 (0.0) | 3778.95 (630.10) | `2.41e-02` | `0.00e+00` | `0.00e+00` | `3.80e-12` | `34.669702` | `1.82e-12 (FC better)` |
| `SciPy-L-BFGS-B` | 0/40 (0%) | 2482.0 (864.8) | 5000.0 (0.0) | 2426.71 (813.35) | `3.82e-05` | `0.00e+00` | `0.00e+00` | `1.59e-10` | `34.668813` | `1.82e-12 (FC better)` |
| `SciPy-SLSQP` | SKIP (`N_tot > 450`) | — | — | — | — | — | — | — | — | — |
| `SciPy-trust-constr` | SKIP (`N_tot > 450`) | — | — | — | — | — | — | — | — | — |

### `B6`: Box-Coupled Brockett with Exact Ky Fan QP Truth

- **Regime**: `z in [0.1,5]^10 x St(150,10) (Branch A, kappa(M)=10^4, certified f_star)` (`d = 10`, `R = 1`, `N_tot = 1510`, intrinsic dim `1455`, FD gradient max rel err `3.584e-10`)

| Solver | Solve Rate | Raw Evals Med (IQR) | Penalized Evals Med (IQR) | Wall ms Med (IQR) | Median KKT Ref | Median Box Viol | Median Lin Viol | Median Orth Viol | Median Objective | Wilcoxon Evals p |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `fortchuck` | 40/40 (100%) | 1947.0 (296.2) | 1947.0 (296.2) | 1475.46 (246.95) | `9.28e-07` | `0.00e+00` | `0.00e+00` | `5.91e-14` | `-10.445510` | `—` |
| `fortchuck-ABB` | 31/40 (78%) | 4562.5 (800.2) | 4562.5 (800.2) | 1564.01 (280.55) | `7.17e-07` | `0.00e+00` | `0.00e+00` | `4.43e-14` | `-10.445510` | `3.57e-08 (FC better)` |
| `Alternating-BCD` | 0/40 (0%) | 5000.0 (0.0) | 5000.0 (0.0) | 3220.71 (257.05) | `1.41e-02` | `0.00e+00` | `0.00e+00` | `1.07e-15` | `-10.443734` | `1.82e-12 (FC better)` |
| `pymanopt-Warped-CG` | 0/40 (0%) | 5000.0 (0.0) | 5000.0 (0.0) | 4897.11 (658.80) | `2.53e-03` | `0.00e+00` | `0.00e+00` | `1.07e-15` | `-10.445240` | `1.82e-12 (FC better)` |
| `pymanopt-TrustRegions` | 0/40 (0%) | 5000.0 (0.0) | 5000.0 (0.0) | 3154.58 (589.62) | `1.60e-03` | `0.00e+00` | `0.00e+00` | `2.41e-15` | `-10.445439` | `1.82e-12 (FC better)` |
| `SciPy-L-BFGS-B` | 0/40 (0%) | 5000.0 (0.0) | 5000.0 (0.0) | 4878.01 (552.32) | `3.55e-03` | `0.00e+00` | `0.00e+00` | `1.50e-05` | `-10.445583` | `1.82e-12 (FC better)` |
| `SciPy-SLSQP` | SKIP (`N_tot > 450`) | — | — | — | — | — | — | — | — | — |
| `SciPy-trust-constr` | SKIP (`N_tot > 450`) | — | — | — | — | — | — | — | — | — |

### `B7A`: Pure-Box Active-Bound Chained Rosenbrock + Quadratic

- **Regime**: `z in [l,u]^500, R=0 (incl. 5 fixed l_i==u_i, ~60% active bounds)` (`d = 500`, `R = 0`, `N_tot = 500`, intrinsic dim `500`, FD gradient max rel err `9.086e-10`)

| Solver | Solve Rate | Raw Evals Med (IQR) | Penalized Evals Med (IQR) | Wall ms Med (IQR) | Median KKT Ref | Median Box Viol | Median Lin Viol | Median Orth Viol | Median Objective | Wilcoxon Evals p |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `fortchuck` | 40/40 (100%) | 68.0 (2.0) | 68.0 (2.0) | 11.05 (2.19) | `8.33e-07` | `0.00e+00` | `0.00e+00` | `0.00e+00` | `87.046277` | `—` |
| `fortchuck-ABB` | 40/40 (100%) | 88.5 (7.5) | 88.5 (7.5) | 11.28 (1.90) | `7.65e-07` | `0.00e+00` | `0.00e+00` | `0.00e+00` | `87.046277` | `3.51e-08 (FC better)` |
| `Alternating-BCD` | 40/40 (100%) | 68.0 (1.5) | 68.0 (1.5) | 41.24 (8.58) | `8.85e-07` | `0.00e+00` | `0.00e+00` | `0.00e+00` | `87.046277` | `3.82e-01 (FC better)` |
| `pymanopt-Warped-CG` | 0/40 (0%) | 5000.0 (0.0) | 5000.0 (0.0) | 1778.68 (283.53) | `2.30e-02` | `0.00e+00` | `0.00e+00` | `0.00e+00` | `87.048579` | `3.03e-08 (FC better)` |
| `pymanopt-TrustRegions` | 0/40 (0%) | 5000.0 (0.0) | 5000.0 (0.0) | 1579.10 (243.00) | `1.58e-04` | `0.00e+00` | `0.00e+00` | `0.00e+00` | `87.046429` | `3.03e-08 (FC better)` |
| `SciPy-L-BFGS-B` | 40/40 (100%) | 68.0 (1.5) | 68.0 (1.5) | 41.56 (9.81) | `8.85e-07` | `0.00e+00` | `0.00e+00` | `0.00e+00` | `87.046277` | `3.82e-01 (FC better)` |
| `SciPy-SLSQP` | SKIP (`N_tot > 450`) | — | — | — | — | — | — | — | — | — |
| `SciPy-trust-constr` | SKIP (`N_tot > 450`) | — | — | — | — | — | — | — | — | — |

### `B7B`: Pure-Stiefel Tall Brockett & Orthogonal Procrustes

- **Regime**: `d=0, St(200,8) x St(24,24) (Pure Stiefel Branch A & Branch B)` (`d = 0`, `R = 2`, `N_tot = 2176`, intrinsic dim `1840`, FD gradient max rel err `5.084e-10`)

| Solver | Solve Rate | Raw Evals Med (IQR) | Penalized Evals Med (IQR) | Wall ms Med (IQR) | Median KKT Ref | Median Box Viol | Median Lin Viol | Median Orth Viol | Median Objective | Wilcoxon Evals p |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `fortchuck` | 40/40 (100%) | 497.0 (63.8) | 497.0 (63.8) | 469.29 (87.91) | `6.77e-07` | `0.00e+00` | `0.00e+00` | `6.71e-14` | `-66.949417` | `—` |
| `fortchuck-ABB` | 40/40 (100%) | 662.0 (94.8) | 662.0 (94.8) | 317.49 (52.50) | `6.87e-07` | `0.00e+00` | `0.00e+00` | `6.92e-14` | `-66.949417` | `3.57e-08 (FC better)` |
| `Alternating-BCD` | 39/40 (98%) | 1091.0 (105.0) | 1096.5 (107.0) | 1305.60 (325.38) | `8.86e-07` | `0.00e+00` | `0.00e+00` | `2.58e-15` | `-66.949417` | `3.57e-08 (FC better)` |
| `pymanopt-Warped-CG` | 39/40 (98%) | 1091.0 (105.0) | 1096.5 (107.0) | 1351.64 (327.72) | `8.86e-07` | `0.00e+00` | `0.00e+00` | `2.58e-15` | `-66.949417` | `3.57e-08 (FC better)` |
| `pymanopt-TrustRegions` | 40/40 (100%) | 1697.0 (276.8) | 1697.0 (276.8) | 1167.28 (287.19) | `7.36e-09` | `0.00e+00` | `0.00e+00` | `2.69e-15` | `-66.949417` | `1.82e-12 (FC better)` |
| `SciPy-L-BFGS-B` | 0/40 (0%) | 2346.0 (987.2) | 5000.0 (0.0) | 2295.19 (977.91) | `1.42e-03` | `0.00e+00` | `0.00e+00` | `5.46e-05` | `-66.950913` | `3.57e-08 (FC better)` |
| `SciPy-SLSQP` | SKIP (`N_tot > 450`) | — | — | — | — | — | — | — | — | — |
| `SciPy-trust-constr` | SKIP (`N_tot > 450`) | — | — | — | — | — | — | — | — | — |

### `B8A`: Trace-Constrained Ensemble Kohn-Sham DFT

- **Regime**: `z in [0,1]^12 (1^T z = 5) x St(120,12) (SLBQP + Branch A, hard trace)` (`d = 12`, `R = 1`, `N_tot = 1452`, intrinsic dim `1373`, linear constraint `a^T z = 5.0`, FD gradient max rel err `1.745e-10`)

| Solver | Solve Rate | Raw Evals Med (IQR) | Penalized Evals Med (IQR) | Wall ms Med (IQR) | Median KKT Ref | Median Box Viol | Median Lin Viol | Median Orth Viol | Median Objective | Wilcoxon Evals p |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `fortchuck` | 40/40 (100%) | 151.5 (28.5) | 151.5 (28.5) | 143.86 (44.50) | `8.22e-07` | `0.00e+00` | `0.00e+00` | `3.15e-14` | `-8.904018` | `—` |
| `fortchuck-ABB` | 38/40 (95%) | 188.5 (72.2) | 188.5 (72.2) | 90.30 (29.89) | `7.52e-07` | `0.00e+00` | `0.00e+00` | `2.85e-14` | `-8.904018` | `1.42e-02 (FC better)` |
| `Alternating-BCD` | 40/40 (100%) | 389.0 (131.2) | 389.0 (131.2) | 399.14 (135.41) | `7.96e-07` | `0.00e+00` | `0.00e+00` | `1.29e-15` | `-8.904018` | `9.49e-08 (FC better)` |
| `pymanopt-Warped-CG` | 0/40 (0%) | 5000.0 (0.0) | 5000.0 (0.0) | 6960.20 (1211.90) | `3.80e-03` | `0.00e+00` | `5.59e-05` | `1.46e-15` | `-8.908933` | `3.56e-08 (FC better)` |
| `pymanopt-TrustRegions` | 0/40 (0%) | 5000.0 (0.0) | 5000.0 (0.0) | 5077.80 (685.17) | `8.47e-03` | `0.00e+00` | `4.59e-06` | `1.55e-15` | `-8.909287` | `3.56e-08 (FC better)` |
| `SciPy-L-BFGS-B` | 0/40 (0%) | 2980.0 (2580.0) | 5000.0 (0.0) | 4172.78 (3634.93) | `5.46e-05` | `0.00e+00` | `6.58e-08` | `7.34e-06` | `-8.904045` | `3.56e-08 (FC better)` |
| `SciPy-SLSQP` | SKIP (`N_tot > 450`) | — | — | — | — | — | — | — | — | — |
| `SciPy-trust-constr` | SKIP (`N_tot > 450`) | — | — | — | — | — | — | — | — | — |

### `B8B`: Long-Only Weighted Orthogonal Risk-Factor Parity

- **Regime**: `z in [0,0.15]^24 (a^T z = 1) x St(32,6) x St(6,6) (N_tot=252, Branch A & B)` (`d = 24`, `R = 2`, `N_tot = 252`, intrinsic dim `209`, linear constraint `a^T z = 1.0`, FD gradient max rel err `1.682e-11`)

| Solver | Solve Rate | Raw Evals Med (IQR) | Penalized Evals Med (IQR) | Wall ms Med (IQR) | Median KKT Ref | Median Box Viol | Median Lin Viol | Median Orth Viol | Median Objective | Wilcoxon Evals p |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `fortchuck` | 40/40 (100%) | 36.0 (3.0) | 36.0 (3.0) | 8.66 (2.14) | `6.92e-07` | `0.00e+00` | `2.78e-17` | `8.26e-15` | `-4.990125` | `—` |
| `fortchuck-ABB` | 40/40 (100%) | 64.0 (10.0) | 64.0 (10.0) | 11.94 (3.66) | `7.70e-07` | `0.00e+00` | `5.55e-17` | `8.96e-15` | `-4.990125` | `3.55e-08 (FC better)` |
| `Alternating-BCD` | 40/40 (100%) | 377.5 (86.8) | 377.5 (86.8) | 454.53 (71.47) | `7.93e-07` | `0.00e+00` | `1.11e-16` | `1.00e-15` | `-4.990125` | `3.56e-08 (FC better)` |
| `pymanopt-Warped-CG` | 0/40 (0%) | 5000.0 (0.0) | 5000.0 (0.0) | 8567.99 (2386.37) | `9.12e-04` | `0.00e+00` | `5.17e-05` | `9.53e-16` | `-4.990095` | `3.04e-08 (FC better)` |
| `pymanopt-TrustRegions` | 0/40 (0%) | 5000.0 (0.0) | 5000.0 (0.0) | 5893.07 (1275.11) | `2.31e-04` | `0.00e+00` | `2.57e-05` | `9.78e-16` | `-4.990135` | `3.04e-08 (FC better)` |
| `SciPy-L-BFGS-B` | 0/40 (0%) | 1027.0 (125.5) | 5000.0 (0.0) | 1225.70 (168.03) | `2.87e-05` | `0.00e+00` | `1.26e-07` | `2.19e-06` | `-4.990130` | `3.04e-08 (FC better)` |
| `SciPy-SLSQP` | 40/40 (100%) | 36.5 (6.2) | 36.5 (6.2) | 477.43 (62.54) | `7.13e-07` | `0.00e+00` | `6.66e-16` | `5.17e-12` | `-4.990125` | `2.49e-01 (FC better)` |
| `SciPy-trust-constr` | 0/40 (0%) | 195.5 (25.2) | 5000.0 (0.0) | 3432.99 (757.54) | `2.95e-05` | `0.00e+00` | `5.55e-17` | `7.07e-14` | `-4.990124` | `3.04e-08 (FC better)` |

## 10. Per-Problem Breakdown — Primary Suite (`Seeds 101..110`)

### `B1`: Ensemble Kohn-Sham DFT Fractional Occupations

- **Regime**: `z in [0,1]^10 x St(120,10) (Branch A, metallic orbital-occupation coupling)` (`d = 10`, `R = 1`, `N_tot = 1210`, intrinsic dim `1155`, FD gradient max rel err `2.104e-10`)

| Solver | Solve Rate | Raw Evals Med (IQR) | Penalized Evals Med (IQR) | Wall ms Med (IQR) | Median KKT Ref | Median Box Viol | Median Lin Viol | Median Orth Viol | Median Objective | Wilcoxon Evals p |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `fortchuck` | 10/10 (100%) | 2247.0 (401.0) | 2247.0 (401.0) | 1525.70 (264.63) | `8.37e-07` | `0.00e+00` | `0.00e+00` | `3.94e-14` | `-4.377192` | `—` |
| `fortchuck-ABB` | 0/10 (0%) | 5000.0 (0.0) | 5000.0 (0.0) | 1798.27 (82.14) | `1.94e-05` | `0.00e+00` | `0.00e+00` | `7.42e-14` | `-4.368460` | `1.95e-03 (FC better)` |
| `Alternating-BCD` | 4/10 (40%) | 5000.0 (4646.0) | 5000.0 (4646.0) | 3100.14 (2962.18) | `3.74e-04` | `0.00e+00` | `0.00e+00` | `1.21e-15` | `-4.349253` | `2.75e-01 (FC better)` |
| `pymanopt-Warped-CG` | 0/10 (0%) | 5000.0 (0.0) | 5000.0 (0.0) | 4728.90 (425.49) | `5.64e-04` | `0.00e+00` | `0.00e+00` | `1.29e-15` | `-4.386121` | `1.95e-03 (FC better)` |
| `pymanopt-TrustRegions` | 0/10 (0%) | 5000.0 (0.0) | 5000.0 (0.0) | 3212.65 (229.87) | `7.64e-03` | `0.00e+00` | `0.00e+00` | `1.54e-13` | `-4.386094` | `1.95e-03 (FC better)` |
| `SciPy-L-BFGS-B` | 0/10 (0%) | 5000.0 (0.0) | 5000.0 (0.0) | 4427.55 (192.08) | `7.97e-04` | `0.00e+00` | `0.00e+00` | `3.20e-06` | `-4.368138` | `1.95e-03 (FC better)` |
| `SciPy-SLSQP` | SKIP (`N_tot > 450`) | — | — | — | — | — | — | — | — | — |
| `SciPy-trust-constr` | SKIP (`N_tot > 450`) | — | — | — | — | — | — | — | — | — |

### `B2A`: Amplitude-Limited Blind Source Separation (ICA)

- **Regime**: `Z in [-2,2]^{5x150} (d=750) x St(20,5) (Branch A, nonconvex log-cosh)` (`d = 750`, `R = 1`, `N_tot = 850`, intrinsic dim `835`, FD gradient max rel err `2.877e-09`)

| Solver | Solve Rate | Raw Evals Med (IQR) | Penalized Evals Med (IQR) | Wall ms Med (IQR) | Median KKT Ref | Median Box Viol | Median Lin Viol | Median Orth Viol | Median Objective | Wilcoxon Evals p |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `fortchuck` | 10/10 (100%) | 597.5 (126.5) | 597.5 (126.5) | 164.47 (59.17) | `8.51e-07` | `0.00e+00` | `0.00e+00` | `3.49e-14` | `-306.924479` | `—` |
| `fortchuck-ABB` | 8/10 (80%) | 3379.0 (1599.8) | 3379.0 (1599.8) | 691.65 (317.84) | `9.20e-07` | `0.00e+00` | `0.00e+00` | `3.81e-14` | `-307.238333` | `1.95e-03 (FC better)` |
| `Alternating-BCD` | 1/10 (10%) | 5000.0 (0.0) | 5000.0 (0.0) | 3774.72 (340.76) | `1.89e-03` | `0.00e+00` | `0.00e+00` | `5.73e-16` | `-307.397095` | `1.95e-03 (FC better)` |
| `pymanopt-Warped-CG` | 0/10 (0%) | 5000.0 (0.0) | 5000.0 (0.0) | 4799.62 (315.65) | `5.66e-01` | `0.00e+00` | `0.00e+00` | `6.53e-16` | `-304.173011` | `1.95e-03 (FC better)` |
| `pymanopt-TrustRegions` | 0/10 (0%) | 5000.0 (0.0) | 5000.0 (0.0) | 3054.23 (86.01) | `3.04e-01` | `0.00e+00` | `0.00e+00` | `2.68e-11` | `-304.608793` | `1.95e-03 (FC better)` |
| `SciPy-L-BFGS-B` | 0/10 (0%) | 5000.0 (0.0) | 5000.0 (0.0) | 3779.71 (62.66) | `9.24e-02` | `0.00e+00` | `0.00e+00` | `2.05e-05` | `-306.311418` | `1.95e-03 (FC better)` |
| `SciPy-SLSQP` | SKIP (`N_tot > 450`) | — | — | — | — | — | — | — | — | — |
| `SciPy-trust-constr` | SKIP (`N_tot > 450`) | — | — | — | — | — | — | — | — | — |

### `B2B`: Smooth Penalty-Split Non-Negative Sparse PCA

- **Regime**: `Z in [0,5]^{120x6} (d=720) x St(120,6) (Branch A, ~80% clamped Z=0, FM-2)` (`d = 720`, `R = 1`, `N_tot = 1440`, intrinsic dim `1419`, FD gradient max rel err `1.484e-08`)

| Solver | Solve Rate | Raw Evals Med (IQR) | Penalized Evals Med (IQR) | Wall ms Med (IQR) | Median KKT Ref | Median Box Viol | Median Lin Viol | Median Orth Viol | Median Objective | Wilcoxon Evals p |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `fortchuck` | 10/10 (100%) | 122.0 (30.0) | 122.0 (30.0) | 38.85 (11.75) | `6.70e-07` | `0.00e+00` | `0.00e+00` | `1.14e-14` | `8.643606` | `—` |
| `fortchuck-ABB` | 10/10 (100%) | 989.5 (941.8) | 989.5 (941.8) | 190.25 (183.03) | `9.30e-07` | `0.00e+00` | `0.00e+00` | `2.74e-14` | `8.200998` | `1.95e-03 (FC better)` |
| `Alternating-BCD` | 0/10 (0%) | 3521.0 (1630.8) | 5000.0 (0.0) | 2840.31 (1274.28) | `1.68e-04` | `0.00e+00` | `0.00e+00` | `9.37e-16` | `8.069873` | `1.95e-03 (FC better)` |
| `pymanopt-Warped-CG` | 3/10 (30%) | 4764.0 (3196.0) | 5000.0 (354.0) | 4443.59 (2971.71) | `4.25e-04` | `0.00e+00` | `0.00e+00` | `1.11e-15` | `13.886985` | `1.95e-03 (FC better)` |
| `pymanopt-TrustRegions` | 4/10 (40%) | 5000.0 (610.2) | 5000.0 (610.2) | 2967.06 (649.97) | `1.28e-02` | `0.00e+00` | `0.00e+00` | `1.10e-10` | `8.460341` | `1.95e-03 (FC better)` |
| `SciPy-L-BFGS-B` | 0/10 (0%) | 413.5 (538.0) | 5000.0 (0.0) | 336.72 (403.17) | `2.54e-05` | `0.00e+00` | `0.00e+00` | `5.91e-06` | `10.943772` | `1.95e-03 (FC better)` |
| `SciPy-SLSQP` | SKIP (`N_tot > 450`) | — | — | — | — | — | — | — | — | — |
| `SciPy-trust-constr` | SKIP (`N_tot > 450`) | — | — | — | — | — | — | — | — | — |

### `B3`: Multi-Camera Anisotropic Bounded-Scale Rig Wahba

- **Regime**: `z in B^15 x St(3,3)^4 (Branch B SO(3)^4, R=4, kappa(W)=10^3)` (`d = 15`, `R = 4`, `N_tot = 51`, intrinsic dim `27`, FD gradient max rel err `7.213e-11`)

| Solver | Solve Rate | Raw Evals Med (IQR) | Penalized Evals Med (IQR) | Wall ms Med (IQR) | Median KKT Ref | Median Box Viol | Median Lin Viol | Median Orth Viol | Median Objective | Wilcoxon Evals p |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `fortchuck` | 10/10 (100%) | 42.0 (13.0) | 42.0 (13.0) | 12.06 (5.61) | `4.98e-07` | `0.00e+00` | `0.00e+00` | `5.86e-15` | `2.176801` | `—` |
| `fortchuck-ABB` | 10/10 (100%) | 84.0 (7.2) | 84.0 (7.2) | 24.09 (3.43) | `5.06e-07` | `0.00e+00` | `0.00e+00` | `7.20e-15` | `2.176801` | `1.95e-03 (FC better)` |
| `Alternating-BCD` | 9/10 (90%) | 319.5 (12.8) | 323.5 (11.2) | 425.73 (30.51) | `8.77e-07` | `0.00e+00` | `0.00e+00` | `8.26e-16` | `2.176801` | `1.95e-03 (FC better)` |
| `pymanopt-Warped-CG` | 10/10 (100%) | 874.5 (304.5) | 874.5 (304.5) | 1607.06 (620.87) | `7.22e-07` | `0.00e+00` | `0.00e+00` | `8.45e-16` | `2.176801` | `1.95e-03 (FC better)` |
| `pymanopt-TrustRegions` | 10/10 (100%) | 658.0 (52.8) | 658.0 (52.8) | 675.10 (91.87) | `6.27e-07` | `0.00e+00` | `0.00e+00` | `7.28e-16` | `2.176801` | `1.95e-03 (FC better)` |
| `SciPy-L-BFGS-B` | 0/10 (0%) | 801.5 (66.2) | 5000.0 (0.0) | 753.68 (46.72) | `2.29e-05` | `0.00e+00` | `0.00e+00` | `4.83e-06` | `2.176772` | `1.95e-03 (FC better)` |
| `SciPy-SLSQP` | 10/10 (100%) | 75.0 (10.5) | 75.0 (10.5) | 74.74 (12.61) | `5.69e-07` | `0.00e+00` | `0.00e+00` | `6.82e-14` | `2.176801` | `1.95e-03 (FC better)` |
| `SciPy-trust-constr` | 10/10 (100%) | 90.0 (17.8) | 90.0 (17.8) | 445.71 (110.13) | `7.85e-07` | `0.00e+00` | `0.00e+00` | `2.46e-15` | `2.176802` | `1.95e-03 (FC better)` |

### `B4`: Bounded-Variance ML Common Principal Components

- **Regime**: `z in [0.05,20]^150 x St(30,30) (Branch B O(30), log barrier, 1.6e5 scale ratio)` (`d = 150`, `R = 1`, `N_tot = 1050`, intrinsic dim `585`, FD gradient max rel err `6.833e-10`)

| Solver | Solve Rate | Raw Evals Med (IQR) | Penalized Evals Med (IQR) | Wall ms Med (IQR) | Median KKT Ref | Median Box Viol | Median Lin Viol | Median Orth Viol | Median Objective | Wilcoxon Evals p |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `fortchuck` | 10/10 (100%) | 1741.5 (104.2) | 1741.5 (104.2) | 1276.11 (140.75) | `6.64e-07` | `0.00e+00` | `0.00e+00` | `5.55e-14` | `17.596323` | `—` |
| `fortchuck-ABB` | 9/10 (90%) | 4391.0 (314.0) | 4391.0 (314.0) | 2137.18 (211.45) | `7.00e-07` | `0.00e+00` | `0.00e+00` | `4.92e-14` | `17.596323` | `1.95e-03 (FC better)` |
| `Alternating-BCD` | 0/10 (0%) | 5000.0 (0.0) | 5000.0 (0.0) | 3953.62 (710.71) | `1.85e-04` | `0.00e+00` | `0.00e+00` | `3.07e-15` | `17.596323` | `1.95e-03 (FC better)` |
| `pymanopt-Warped-CG` | 0/10 (0%) | 5000.0 (0.0) | 5000.0 (0.0) | 5989.59 (859.56) | `8.56e-03` | `0.00e+00` | `0.00e+00` | `3.16e-15` | `17.599052` | `1.95e-03 (FC better)` |
| `pymanopt-TrustRegions` | 0/10 (0%) | 5000.0 (0.0) | 5000.0 (0.0) | 3716.99 (837.50) | `4.70e-03` | `0.00e+00` | `0.00e+00` | `9.85e-14` | `17.597364` | `1.95e-03 (FC better)` |
| `SciPy-L-BFGS-B` | 0/10 (0%) | 5000.0 (0.0) | 5000.0 (0.0) | 4772.65 (499.06) | `6.40e-01` | `0.00e+00` | `0.00e+00` | `1.03e-05` | `21.044210` | `1.95e-03 (FC better)` |
| `SciPy-SLSQP` | SKIP (`N_tot > 450`) | — | — | — | — | — | — | — | — | — |
| `SciPy-trust-constr` | SKIP (`N_tot > 450`) | — | — | — | — | — | — | — | — | — |

### `B5`: Bounded Factor Correlation Matrix Calibration

- **Regime**: `z in B^110 x St(100,10) (Branch A, quartic valley + clamped psi)` (`d = 110`, `R = 1`, `N_tot = 1110`, intrinsic dim `1055`, FD gradient max rel err `8.432e-10`)

| Solver | Solve Rate | Raw Evals Med (IQR) | Penalized Evals Med (IQR) | Wall ms Med (IQR) | Median KKT Ref | Median Box Viol | Median Lin Viol | Median Orth Viol | Median Objective | Wilcoxon Evals p |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `fortchuck` | 10/10 (100%) | 1219.5 (272.8) | 1219.5 (272.8) | 736.51 (228.96) | `9.28e-07` | `0.00e+00` | `0.00e+00` | `4.73e-14` | `34.668813` | `—` |
| `fortchuck-ABB` | 0/10 (0%) | 5000.0 (0.0) | 5000.0 (0.0) | 1756.15 (100.99) | `1.58e-03` | `0.00e+00` | `0.00e+00` | `3.98e-14` | `34.669022` | `1.95e-03 (FC better)` |
| `Alternating-BCD` | 0/10 (0%) | 5000.0 (0.0) | 5000.0 (0.0) | 3822.40 (343.45) | `4.55e-03` | `0.00e+00` | `0.00e+00` | `1.22e-15` | `34.669225` | `1.95e-03 (FC better)` |
| `pymanopt-Warped-CG` | 0/10 (0%) | 5000.0 (0.0) | 5000.0 (0.0) | 5608.24 (779.05) | `1.50e-02` | `0.00e+00` | `0.00e+00` | `1.30e-15` | `34.671020` | `1.95e-03 (FC better)` |
| `pymanopt-TrustRegions` | 0/10 (0%) | 5000.0 (0.0) | 5000.0 (0.0) | 3717.55 (578.86) | `2.28e-02` | `0.00e+00` | `0.00e+00` | `4.02e-12` | `34.669569` | `1.95e-03 (FC better)` |
| `SciPy-L-BFGS-B` | 0/10 (0%) | 2495.5 (595.8) | 5000.0 (0.0) | 2408.18 (436.03) | `3.64e-05` | `0.00e+00` | `0.00e+00` | `1.88e-10` | `34.668813` | `1.95e-03 (FC better)` |
| `SciPy-SLSQP` | SKIP (`N_tot > 450`) | — | — | — | — | — | — | — | — | — |
| `SciPy-trust-constr` | SKIP (`N_tot > 450`) | — | — | — | — | — | — | — | — | — |

### `B6`: Box-Coupled Brockett with Exact Ky Fan QP Truth

- **Regime**: `z in [0.1,5]^10 x St(150,10) (Branch A, kappa(M)=10^4, certified f_star)` (`d = 10`, `R = 1`, `N_tot = 1510`, intrinsic dim `1455`, FD gradient max rel err `3.584e-10`)

| Solver | Solve Rate | Raw Evals Med (IQR) | Penalized Evals Med (IQR) | Wall ms Med (IQR) | Median KKT Ref | Median Box Viol | Median Lin Viol | Median Orth Viol | Median Objective | Wilcoxon Evals p |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `fortchuck` | 10/10 (100%) | 1947.0 (386.8) | 1947.0 (386.8) | 1399.92 (233.66) | `9.40e-07` | `0.00e+00` | `0.00e+00` | `7.41e-14` | `-10.445510` | `—` |
| `fortchuck-ABB` | 6/10 (60%) | 4519.5 (823.2) | 4519.5 (823.2) | 1505.64 (250.35) | `7.63e-07` | `0.00e+00` | `0.00e+00` | `6.01e-14` | `-10.445510` | `1.95e-03 (FC better)` |
| `Alternating-BCD` | 0/10 (0%) | 5000.0 (0.0) | 5000.0 (0.0) | 3501.53 (251.68) | `2.09e-02` | `0.00e+00` | `0.00e+00` | `1.05e-15` | `-10.443098` | `1.95e-03 (FC better)` |
| `pymanopt-Warped-CG` | 0/10 (0%) | 5000.0 (0.0) | 5000.0 (0.0) | 5504.36 (396.95) | `5.17e-03` | `0.00e+00` | `0.00e+00` | `1.06e-15` | `-10.444992` | `1.95e-03 (FC better)` |
| `pymanopt-TrustRegions` | 0/10 (0%) | 5000.0 (0.0) | 5000.0 (0.0) | 3888.68 (375.86) | `1.80e-03` | `0.00e+00` | `0.00e+00` | `3.99e-15` | `-10.445432` | `1.95e-03 (FC better)` |
| `SciPy-L-BFGS-B` | 0/10 (0%) | 5000.0 (0.0) | 5000.0 (0.0) | 5127.43 (300.89) | `2.65e-03` | `0.00e+00` | `0.00e+00` | `1.50e-05` | `-10.445612` | `1.95e-03 (FC better)` |
| `SciPy-SLSQP` | SKIP (`N_tot > 450`) | — | — | — | — | — | — | — | — | — |
| `SciPy-trust-constr` | SKIP (`N_tot > 450`) | — | — | — | — | — | — | — | — | — |

### `B7A`: Pure-Box Active-Bound Chained Rosenbrock + Quadratic

- **Regime**: `z in [l,u]^500, R=0 (incl. 5 fixed l_i==u_i, ~60% active bounds)` (`d = 500`, `R = 0`, `N_tot = 500`, intrinsic dim `500`, FD gradient max rel err `9.086e-10`)

| Solver | Solve Rate | Raw Evals Med (IQR) | Penalized Evals Med (IQR) | Wall ms Med (IQR) | Median KKT Ref | Median Box Viol | Median Lin Viol | Median Orth Viol | Median Objective | Wilcoxon Evals p |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `fortchuck` | 10/10 (100%) | 67.5 (2.0) | 67.5 (2.0) | 10.58 (2.01) | `8.21e-07` | `0.00e+00` | `0.00e+00` | `0.00e+00` | `87.046277` | `—` |
| `fortchuck-ABB` | 10/10 (100%) | 85.0 (11.0) | 85.0 (11.0) | 10.70 (0.70) | `8.60e-07` | `0.00e+00` | `0.00e+00` | `0.00e+00` | `87.046277` | `1.95e-03 (FC better)` |
| `Alternating-BCD` | 10/10 (100%) | 68.0 (0.8) | 68.0 (0.8) | 49.07 (11.34) | `8.47e-07` | `0.00e+00` | `0.00e+00` | `0.00e+00` | `87.046277` | `2.66e-01 (FC better)` |
| `pymanopt-Warped-CG` | 0/10 (0%) | 5000.0 (0.0) | 5000.0 (0.0) | 2222.84 (144.03) | `1.28e-03` | `0.00e+00` | `0.00e+00` | `0.00e+00` | `87.047726` | `1.95e-03 (FC better)` |
| `pymanopt-TrustRegions` | 0/10 (0%) | 5000.0 (0.0) | 5000.0 (0.0) | 1928.81 (102.99) | `2.15e-04` | `0.00e+00` | `0.00e+00` | `0.00e+00` | `87.046452` | `1.95e-03 (FC better)` |
| `SciPy-L-BFGS-B` | 10/10 (100%) | 68.0 (0.8) | 68.0 (0.8) | 51.45 (7.97) | `8.47e-07` | `0.00e+00` | `0.00e+00` | `0.00e+00` | `87.046277` | `2.66e-01 (FC better)` |
| `SciPy-SLSQP` | SKIP (`N_tot > 450`) | — | — | — | — | — | — | — | — | — |
| `SciPy-trust-constr` | SKIP (`N_tot > 450`) | — | — | — | — | — | — | — | — | — |

### `B7B`: Pure-Stiefel Tall Brockett & Orthogonal Procrustes

- **Regime**: `d=0, St(200,8) x St(24,24) (Pure Stiefel Branch A & Branch B)` (`d = 0`, `R = 2`, `N_tot = 2176`, intrinsic dim `1840`, FD gradient max rel err `5.084e-10`)

| Solver | Solve Rate | Raw Evals Med (IQR) | Penalized Evals Med (IQR) | Wall ms Med (IQR) | Median KKT Ref | Median Box Viol | Median Lin Viol | Median Orth Viol | Median Objective | Wilcoxon Evals p |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `fortchuck` | 10/10 (100%) | 493.5 (53.2) | 493.5 (53.2) | 436.73 (49.18) | `6.88e-07` | `0.00e+00` | `0.00e+00` | `6.71e-14` | `-66.949417` | `—` |
| `fortchuck-ABB` | 10/10 (100%) | 662.0 (159.0) | 662.0 (159.0) | 287.41 (66.82) | `6.98e-07` | `0.00e+00` | `0.00e+00` | `7.31e-14` | `-66.949417` | `1.95e-03 (FC better)` |
| `Alternating-BCD` | 10/10 (100%) | 1062.5 (86.8) | 1062.5 (86.8) | 1580.81 (184.18) | `8.59e-07` | `0.00e+00` | `0.00e+00` | `2.46e-15` | `-66.949417` | `1.95e-03 (FC better)` |
| `pymanopt-Warped-CG` | 10/10 (100%) | 1062.5 (86.8) | 1062.5 (86.8) | 1584.33 (297.91) | `8.59e-07` | `0.00e+00` | `0.00e+00` | `2.46e-15` | `-66.949417` | `1.95e-03 (FC better)` |
| `pymanopt-TrustRegions` | 10/10 (100%) | 1700.5 (263.0) | 1700.5 (263.0) | 1407.55 (202.98) | `1.17e-08` | `0.00e+00` | `0.00e+00` | `2.93e-15` | `-66.949417` | `1.95e-03 (FC better)` |
| `SciPy-L-BFGS-B` | 0/10 (0%) | 2877.0 (831.8) | 5000.0 (0.0) | 2953.87 (820.27) | `1.42e-03` | `0.00e+00` | `0.00e+00` | `5.46e-05` | `-66.818295` | `1.95e-03 (FC better)` |
| `SciPy-SLSQP` | SKIP (`N_tot > 450`) | — | — | — | — | — | — | — | — | — |
| `SciPy-trust-constr` | SKIP (`N_tot > 450`) | — | — | — | — | — | — | — | — | — |

### `B8A`: Trace-Constrained Ensemble Kohn-Sham DFT

- **Regime**: `z in [0,1]^12 (1^T z = 5) x St(120,12) (SLBQP + Branch A, hard trace)` (`d = 12`, `R = 1`, `N_tot = 1452`, intrinsic dim `1373`, linear constraint `a^T z = 5.0`, FD gradient max rel err `1.745e-10`)

| Solver | Solve Rate | Raw Evals Med (IQR) | Penalized Evals Med (IQR) | Wall ms Med (IQR) | Median KKT Ref | Median Box Viol | Median Lin Viol | Median Orth Viol | Median Objective | Wilcoxon Evals p |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `fortchuck` | 10/10 (100%) | 154.0 (24.5) | 154.0 (24.5) | 136.04 (35.08) | `8.94e-07` | `0.00e+00` | `0.00e+00` | `3.39e-14` | `-8.904018` | `—` |
| `fortchuck-ABB` | 9/10 (90%) | 176.0 (47.8) | 176.0 (47.8) | 79.24 (21.08) | `7.83e-07` | `0.00e+00` | `0.00e+00` | `2.97e-14` | `-8.904018` | `1.25e-01 (FC better)` |
| `Alternating-BCD` | 10/10 (100%) | 353.0 (117.5) | 353.0 (117.5) | 419.30 (118.90) | `8.44e-07` | `0.00e+00` | `0.00e+00` | `1.27e-15` | `-8.904018` | `5.86e-03 (FC better)` |
| `pymanopt-Warped-CG` | 0/10 (0%) | 5000.0 (0.0) | 5000.0 (0.0) | 8140.29 (531.00) | `3.35e-03` | `0.00e+00` | `5.45e-05` | `1.41e-15` | `-8.908985` | `1.95e-03 (FC better)` |
| `pymanopt-TrustRegions` | 0/10 (0%) | 5000.0 (0.0) | 5000.0 (0.0) | 5886.07 (456.88) | `6.71e-03` | `0.00e+00` | `4.00e-06` | `1.76e-15` | `-8.909272` | `1.95e-03 (FC better)` |
| `SciPy-L-BFGS-B` | 0/10 (0%) | 2346.0 (1623.0) | 5000.0 (0.0) | 3591.28 (2493.65) | `5.15e-05` | `0.00e+00` | `5.03e-09` | `7.34e-06` | `-8.904045` | `1.95e-03 (FC better)` |
| `SciPy-SLSQP` | SKIP (`N_tot > 450`) | — | — | — | — | — | — | — | — | — |
| `SciPy-trust-constr` | SKIP (`N_tot > 450`) | — | — | — | — | — | — | — | — | — |

### `B8B`: Long-Only Weighted Orthogonal Risk-Factor Parity

- **Regime**: `z in [0,0.15]^24 (a^T z = 1) x St(32,6) x St(6,6) (N_tot=252, Branch A & B)` (`d = 24`, `R = 2`, `N_tot = 252`, intrinsic dim `209`, linear constraint `a^T z = 1.0`, FD gradient max rel err `1.682e-11`)

| Solver | Solve Rate | Raw Evals Med (IQR) | Penalized Evals Med (IQR) | Wall ms Med (IQR) | Median KKT Ref | Median Box Viol | Median Lin Viol | Median Orth Viol | Median Objective | Wilcoxon Evals p |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `fortchuck` | 10/10 (100%) | 36.5 (3.0) | 36.5 (3.0) | 8.36 (1.29) | `7.41e-07` | `0.00e+00` | `2.78e-17` | `8.44e-15` | `-4.990125` | `—` |
| `fortchuck-ABB` | 10/10 (100%) | 63.0 (7.0) | 63.0 (7.0) | 10.44 (1.78) | `8.62e-07` | `0.00e+00` | `5.55e-17` | `8.54e-15` | `-4.990125` | `1.95e-03 (FC better)` |
| `Alternating-BCD` | 10/10 (100%) | 359.5 (53.8) | 359.5 (53.8) | 480.54 (92.92) | `7.42e-07` | `0.00e+00` | `1.11e-16` | `9.85e-16` | `-4.990125` | `1.95e-03 (FC better)` |
| `pymanopt-Warped-CG` | 0/10 (0%) | 5000.0 (0.0) | 5000.0 (0.0) | 10134.91 (1998.83) | `9.15e-04` | `0.00e+00` | `5.16e-05` | `1.03e-15` | `-4.990102` | `1.95e-03 (FC better)` |
| `pymanopt-TrustRegions` | 0/10 (0%) | 5000.0 (0.0) | 5000.0 (0.0) | 6898.25 (902.39) | `2.52e-04` | `0.00e+00` | `2.56e-05` | `9.78e-16` | `-4.990136` | `1.95e-03 (FC better)` |
| `SciPy-L-BFGS-B` | 0/10 (0%) | 1014.5 (104.8) | 5000.0 (0.0) | 1321.31 (215.85) | `3.02e-05` | `0.00e+00` | `1.26e-07` | `2.19e-06` | `-4.990130` | `1.95e-03 (FC better)` |
| `SciPy-SLSQP` | 10/10 (100%) | 40.0 (10.5) | 40.0 (10.5) | 531.59 (53.82) | `6.99e-07` | `0.00e+00` | `6.66e-16` | `4.98e-12` | `-4.990125` | `1.13e-01 (FC better)` |
| `SciPy-trust-constr` | 0/10 (0%) | 199.0 (25.2) | 5000.0 (0.0) | 4338.92 (1083.11) | `3.14e-05` | `0.00e+00` | `0.00e+00` | `2.78e-13` | `-4.990124` | `1.95e-03 (FC better)` |

## 11. Per-Problem Breakdown — Out-of-Sample Holdout Suite (`Seeds 201..210`)

### `B1`: Ensemble Kohn-Sham DFT Fractional Occupations

- **Regime**: `z in [0,1]^10 x St(120,10) (Branch A, metallic orbital-occupation coupling)` (`d = 10`, `R = 1`, `N_tot = 1210`, intrinsic dim `1155`, FD gradient max rel err `2.091e-10`)

| Solver | Solve Rate | Raw Evals Med (IQR) | Penalized Evals Med (IQR) | Wall ms Med (IQR) | Median KKT Ref | Median Box Viol | Median Lin Viol | Median Orth Viol | Median Objective | Wilcoxon Evals p |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `fortchuck` | 10/10 (100%) | 2233.0 (873.0) | 2233.0 (873.0) | 1496.05 (602.25) | `8.76e-07` | `0.00e+00` | `0.00e+00` | `3.08e-14` | `-4.372830` | `—` |
| `fortchuck-ABB` | 1/10 (10%) | 5000.0 (0.0) | 5000.0 (0.0) | 1771.47 (51.53) | `2.63e-05` | `0.00e+00` | `0.00e+00` | `3.90e-14` | `-4.368467` | `1.95e-03 (FC better)` |
| `Alternating-BCD` | 2/10 (20%) | 5000.0 (0.0) | 5000.0 (0.0) | 4187.06 (612.51) | `1.37e-03` | `0.00e+00` | `0.00e+00` | `1.20e-15` | `-4.349173` | `1.37e-02 (FC better)` |
| `pymanopt-Warped-CG` | 0/10 (0%) | 5000.0 (0.0) | 5000.0 (0.0) | 6698.04 (450.08) | `8.43e-04` | `0.00e+00` | `0.00e+00` | `1.32e-15` | `-4.386120` | `1.95e-03 (FC better)` |
| `pymanopt-TrustRegions` | 0/10 (0%) | 5000.0 (0.0) | 5000.0 (0.0) | 4125.10 (482.54) | `9.59e-03` | `0.00e+00` | `0.00e+00` | `1.35e-13` | `-4.386097` | `1.95e-03 (FC better)` |
| `SciPy-L-BFGS-B` | 0/10 (0%) | 5000.0 (0.0) | 5000.0 (0.0) | 5300.89 (392.30) | `4.09e-04` | `0.00e+00` | `0.00e+00` | `3.20e-06` | `-4.358742` | `1.95e-03 (FC better)` |
| `SciPy-SLSQP` | SKIP (`N_tot > 450`) | — | — | — | — | — | — | — | — | — |
| `SciPy-trust-constr` | SKIP (`N_tot > 450`) | — | — | — | — | — | — | — | — | — |

### `B2A`: Amplitude-Limited Blind Source Separation (ICA)

- **Regime**: `Z in [-2,2]^{5x150} (d=750) x St(20,5) (Branch A, nonconvex log-cosh)` (`d = 750`, `R = 1`, `N_tot = 850`, intrinsic dim `835`, FD gradient max rel err `1.935e-09`)

| Solver | Solve Rate | Raw Evals Med (IQR) | Penalized Evals Med (IQR) | Wall ms Med (IQR) | Median KKT Ref | Median Box Viol | Median Lin Viol | Median Orth Viol | Median Objective | Wilcoxon Evals p |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `fortchuck` | 10/10 (100%) | 568.5 (381.5) | 568.5 (381.5) | 179.63 (82.55) | `8.12e-07` | `0.00e+00` | `0.00e+00` | `3.90e-14` | `-309.118287` | `—` |
| `fortchuck-ABB` | 5/10 (50%) | 4834.0 (858.2) | 4834.0 (858.2) | 981.67 (184.45) | `2.50e-05` | `0.00e+00` | `0.00e+00` | `7.33e-14` | `-307.661972` | `1.95e-03 (FC better)` |
| `Alternating-BCD` | 2/10 (20%) | 5000.0 (0.0) | 5000.0 (0.0) | 4707.83 (337.84) | `1.17e-02` | `0.00e+00` | `0.00e+00` | `6.98e-16` | `-306.937441` | `1.95e-03 (FC better)` |
| `pymanopt-Warped-CG` | 0/10 (0%) | 5000.0 (0.0) | 5000.0 (0.0) | 6193.33 (306.29) | `4.67e-01` | `0.00e+00` | `0.00e+00` | `6.53e-16` | `-309.091804` | `1.95e-03 (FC better)` |
| `pymanopt-TrustRegions` | 0/10 (0%) | 5000.0 (0.0) | 5000.0 (0.0) | 3807.69 (231.54) | `6.25e-01` | `0.00e+00` | `0.00e+00` | `3.22e-11` | `-307.525867` | `1.95e-03 (FC better)` |
| `SciPy-L-BFGS-B` | 0/10 (0%) | 5000.0 (0.0) | 5000.0 (0.0) | 4433.42 (73.31) | `4.53e-02` | `0.00e+00` | `0.00e+00` | `3.04e-05` | `-309.847765` | `1.95e-03 (FC better)` |
| `SciPy-SLSQP` | SKIP (`N_tot > 450`) | — | — | — | — | — | — | — | — | — |
| `SciPy-trust-constr` | SKIP (`N_tot > 450`) | — | — | — | — | — | — | — | — | — |

### `B2B`: Smooth Penalty-Split Non-Negative Sparse PCA

- **Regime**: `Z in [0,5]^{120x6} (d=720) x St(120,6) (Branch A, ~80% clamped Z=0, FM-2)` (`d = 720`, `R = 1`, `N_tot = 1440`, intrinsic dim `1419`, FD gradient max rel err `1.028e-08`)

| Solver | Solve Rate | Raw Evals Med (IQR) | Penalized Evals Med (IQR) | Wall ms Med (IQR) | Median KKT Ref | Median Box Viol | Median Lin Viol | Median Orth Viol | Median Objective | Wilcoxon Evals p |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `fortchuck` | 10/10 (100%) | 95.5 (31.8) | 95.5 (31.8) | 32.05 (8.03) | `7.36e-07` | `0.00e+00` | `0.00e+00` | `9.49e-15` | `8.606887` | `—` |
| `fortchuck-ABB` | 10/10 (100%) | 1324.0 (931.2) | 1324.0 (931.2) | 269.17 (220.34) | `9.31e-07` | `0.00e+00` | `0.00e+00` | `3.18e-14` | `8.026764` | `1.95e-03 (FC better)` |
| `Alternating-BCD` | 0/10 (0%) | 3651.0 (2740.5) | 5000.0 (0.0) | 3379.63 (2486.22) | `2.14e-04` | `0.00e+00` | `0.00e+00` | `1.15e-15` | `8.188735` | `1.95e-03 (FC better)` |
| `pymanopt-Warped-CG` | 6/10 (60%) | 4045.0 (3606.5) | 4045.0 (3606.5) | 4469.22 (4312.87) | `9.27e-07` | `0.00e+00` | `0.00e+00` | `1.00e-15` | `13.121072` | `1.95e-03 (FC better)` |
| `pymanopt-TrustRegions` | 7/10 (70%) | 2930.5 (2722.0) | 2930.5 (2722.0) | 2229.88 (1813.15) | `8.56e-07` | `0.00e+00` | `0.00e+00` | `1.18e-15` | `8.282000` | `1.95e-03 (FC better)` |
| `SciPy-L-BFGS-B` | 0/10 (0%) | 358.5 (359.8) | 5000.0 (0.0) | 342.75 (315.34) | `1.20e-04` | `0.00e+00` | `0.00e+00` | `1.15e-05` | `13.161188` | `1.95e-03 (FC better)` |
| `SciPy-SLSQP` | SKIP (`N_tot > 450`) | — | — | — | — | — | — | — | — | — |
| `SciPy-trust-constr` | SKIP (`N_tot > 450`) | — | — | — | — | — | — | — | — | — |

### `B3`: Multi-Camera Anisotropic Bounded-Scale Rig Wahba

- **Regime**: `z in B^15 x St(3,3)^4 (Branch B SO(3)^4, R=4, kappa(W)=10^3)` (`d = 15`, `R = 4`, `N_tot = 51`, intrinsic dim `27`, FD gradient max rel err `9.877e-11`)

| Solver | Solve Rate | Raw Evals Med (IQR) | Penalized Evals Med (IQR) | Wall ms Med (IQR) | Median KKT Ref | Median Box Viol | Median Lin Viol | Median Orth Viol | Median Objective | Wilcoxon Evals p |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `fortchuck` | 10/10 (100%) | 46.5 (10.2) | 46.5 (10.2) | 12.95 (2.54) | `5.34e-07` | `0.00e+00` | `0.00e+00` | `5.01e-15` | `2.176801` | `—` |
| `fortchuck-ABB` | 10/10 (100%) | 82.5 (15.5) | 82.5 (15.5) | 21.74 (4.79) | `6.19e-07` | `0.00e+00` | `0.00e+00` | `6.72e-15` | `2.176801` | `1.95e-03 (FC better)` |
| `Alternating-BCD` | 10/10 (100%) | 316.5 (25.2) | 316.5 (25.2) | 554.42 (34.01) | `7.93e-07` | `0.00e+00` | `0.00e+00` | `6.11e-16` | `2.176801` | `1.95e-03 (FC better)` |
| `pymanopt-Warped-CG` | 10/10 (100%) | 854.0 (368.2) | 854.0 (368.2) | 2146.65 (921.44) | `8.58e-07` | `0.00e+00` | `0.00e+00` | `5.68e-16` | `2.176801` | `1.95e-03 (FC better)` |
| `pymanopt-TrustRegions` | 10/10 (100%) | 651.5 (77.5) | 651.5 (77.5) | 850.17 (144.72) | `5.39e-07` | `0.00e+00` | `0.00e+00` | `6.81e-16` | `2.176801` | `1.95e-03 (FC better)` |
| `SciPy-L-BFGS-B` | 0/10 (0%) | 805.0 (153.5) | 5000.0 (0.0) | 895.50 (193.70) | `2.28e-05` | `0.00e+00` | `0.00e+00` | `4.83e-06` | `2.176772` | `1.95e-03 (FC better)` |
| `SciPy-SLSQP` | 10/10 (100%) | 86.5 (18.5) | 86.5 (18.5) | 96.38 (24.81) | `4.44e-07` | `0.00e+00` | `0.00e+00` | `4.69e-14` | `2.176801` | `1.95e-03 (FC better)` |
| `SciPy-trust-constr` | 10/10 (100%) | 95.0 (28.2) | 95.0 (28.2) | 653.41 (222.00) | `7.81e-07` | `0.00e+00` | `0.00e+00` | `5.27e-15` | `2.176802` | `1.95e-03 (FC better)` |

### `B4`: Bounded-Variance ML Common Principal Components

- **Regime**: `z in [0.05,20]^150 x St(30,30) (Branch B O(30), log barrier, 1.6e5 scale ratio)` (`d = 150`, `R = 1`, `N_tot = 1050`, intrinsic dim `585`, FD gradient max rel err `4.202e-10`)

| Solver | Solve Rate | Raw Evals Med (IQR) | Penalized Evals Med (IQR) | Wall ms Med (IQR) | Median KKT Ref | Median Box Viol | Median Lin Viol | Median Orth Viol | Median Objective | Wilcoxon Evals p |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `fortchuck` | 10/10 (100%) | 1662.0 (217.0) | 1662.0 (217.0) | 1228.93 (185.90) | `6.58e-07` | `0.00e+00` | `0.00e+00` | `5.23e-14` | `17.596323` | `—` |
| `fortchuck-ABB` | 9/10 (90%) | 4523.0 (669.0) | 4523.0 (669.0) | 2232.64 (445.82) | `6.87e-07` | `0.00e+00` | `0.00e+00` | `7.15e-14` | `17.596323` | `1.95e-03 (FC better)` |
| `Alternating-BCD` | 0/10 (0%) | 5000.0 (0.0) | 5000.0 (0.0) | 4566.23 (553.09) | `9.87e-04` | `0.00e+00` | `0.00e+00` | `3.13e-15` | `17.596323` | `1.95e-03 (FC better)` |
| `pymanopt-Warped-CG` | 0/10 (0%) | 5000.0 (0.0) | 5000.0 (0.0) | 6480.33 (815.45) | `1.19e-02` | `0.00e+00` | `0.00e+00` | `3.02e-15` | `17.599161` | `1.95e-03 (FC better)` |
| `pymanopt-TrustRegions` | 0/10 (0%) | 5000.0 (0.0) | 5000.0 (0.0) | 4225.88 (243.14) | `5.17e-03` | `0.00e+00` | `0.00e+00` | `7.26e-14` | `17.597603` | `1.95e-03 (FC better)` |
| `SciPy-L-BFGS-B` | 0/10 (0%) | 5000.0 (0.0) | 5000.0 (0.0) | 5333.04 (324.77) | `5.91e-01` | `0.00e+00` | `0.00e+00` | `7.85e-06` | `20.832811` | `1.95e-03 (FC better)` |
| `SciPy-SLSQP` | SKIP (`N_tot > 450`) | — | — | — | — | — | — | — | — | — |
| `SciPy-trust-constr` | SKIP (`N_tot > 450`) | — | — | — | — | — | — | — | — | — |

### `B5`: Bounded Factor Correlation Matrix Calibration

- **Regime**: `z in B^110 x St(100,10) (Branch A, quartic valley + clamped psi)` (`d = 110`, `R = 1`, `N_tot = 1110`, intrinsic dim `1055`, FD gradient max rel err `9.454e-10`)

| Solver | Solve Rate | Raw Evals Med (IQR) | Penalized Evals Med (IQR) | Wall ms Med (IQR) | Median KKT Ref | Median Box Viol | Median Lin Viol | Median Orth Viol | Median Objective | Wilcoxon Evals p |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `fortchuck` | 10/10 (100%) | 1192.0 (236.5) | 1192.0 (236.5) | 754.13 (158.41) | `9.57e-07` | `0.00e+00` | `0.00e+00` | `5.54e-14` | `34.668813` | `—` |
| `fortchuck-ABB` | 0/10 (0%) | 5000.0 (0.0) | 5000.0 (0.0) | 1755.11 (41.68) | `1.63e-03` | `0.00e+00` | `0.00e+00` | `3.27e-14` | `34.668915` | `1.95e-03 (FC better)` |
| `Alternating-BCD` | 0/10 (0%) | 5000.0 (0.0) | 5000.0 (0.0) | 4211.69 (393.85) | `5.73e-03` | `0.00e+00` | `0.00e+00` | `9.96e-16` | `34.669074` | `1.95e-03 (FC better)` |
| `pymanopt-Warped-CG` | 0/10 (0%) | 5000.0 (0.0) | 5000.0 (0.0) | 6122.56 (849.62) | `2.43e-02` | `0.00e+00` | `0.00e+00` | `1.28e-15` | `34.671823` | `1.95e-03 (FC better)` |
| `pymanopt-TrustRegions` | 0/10 (0%) | 5000.0 (0.0) | 5000.0 (0.0) | 4039.39 (404.42) | `2.36e-02` | `0.00e+00` | `0.00e+00` | `3.87e-12` | `34.669714` | `1.95e-03 (FC better)` |
| `SciPy-L-BFGS-B` | 0/10 (0%) | 2508.5 (849.2) | 5000.0 (0.0) | 2472.08 (851.41) | `3.08e-05` | `0.00e+00` | `0.00e+00` | `1.28e-10` | `34.668813` | `1.95e-03 (FC better)` |
| `SciPy-SLSQP` | SKIP (`N_tot > 450`) | — | — | — | — | — | — | — | — | — |
| `SciPy-trust-constr` | SKIP (`N_tot > 450`) | — | — | — | — | — | — | — | — | — |

### `B6`: Box-Coupled Brockett with Exact Ky Fan QP Truth

- **Regime**: `z in [0.1,5]^10 x St(150,10) (Branch A, kappa(M)=10^4, certified f_star)` (`d = 10`, `R = 1`, `N_tot = 1510`, intrinsic dim `1455`, FD gradient max rel err `3.113e-10`)

| Solver | Solve Rate | Raw Evals Med (IQR) | Penalized Evals Med (IQR) | Wall ms Med (IQR) | Median KKT Ref | Median Box Viol | Median Lin Viol | Median Orth Viol | Median Objective | Wilcoxon Evals p |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `fortchuck` | 10/10 (100%) | 1887.5 (402.5) | 1887.5 (402.5) | 1358.20 (280.08) | `9.03e-07` | `0.00e+00` | `0.00e+00` | `4.96e-14` | `-10.445510` | `—` |
| `fortchuck-ABB` | 8/10 (80%) | 4688.5 (628.8) | 4688.5 (628.8) | 1478.90 (203.29) | `7.11e-07` | `0.00e+00` | `0.00e+00` | `3.29e-14` | `-10.445510` | `1.95e-03 (FC better)` |
| `Alternating-BCD` | 0/10 (0%) | 5000.0 (0.0) | 5000.0 (0.0) | 3190.03 (172.14) | `9.64e-03` | `0.00e+00` | `0.00e+00` | `1.06e-15` | `-10.443625` | `1.95e-03 (FC better)` |
| `pymanopt-Warped-CG` | 0/10 (0%) | 5000.0 (0.0) | 5000.0 (0.0) | 4846.70 (263.71) | `1.65e-03` | `0.00e+00` | `0.00e+00` | `1.16e-15` | `-10.445257` | `1.95e-03 (FC better)` |
| `pymanopt-TrustRegions` | 0/10 (0%) | 5000.0 (0.0) | 5000.0 (0.0) | 3153.64 (158.77) | `1.47e-03` | `0.00e+00` | `0.00e+00` | `6.34e-15` | `-10.445437` | `1.95e-03 (FC better)` |
| `SciPy-L-BFGS-B` | 0/10 (0%) | 5000.0 (0.0) | 5000.0 (0.0) | 4520.76 (127.35) | `3.25e-03` | `0.00e+00` | `0.00e+00` | `1.50e-05` | `-10.445593` | `1.95e-03 (FC better)` |
| `SciPy-SLSQP` | SKIP (`N_tot > 450`) | — | — | — | — | — | — | — | — | — |
| `SciPy-trust-constr` | SKIP (`N_tot > 450`) | — | — | — | — | — | — | — | — | — |

### `B7A`: Pure-Box Active-Bound Chained Rosenbrock + Quadratic

- **Regime**: `z in [l,u]^500, R=0 (incl. 5 fixed l_i==u_i, ~60% active bounds)` (`d = 500`, `R = 0`, `N_tot = 500`, intrinsic dim `500`, FD gradient max rel err `8.252e-10`)

| Solver | Solve Rate | Raw Evals Med (IQR) | Penalized Evals Med (IQR) | Wall ms Med (IQR) | Median KKT Ref | Median Box Viol | Median Lin Viol | Median Orth Viol | Median Objective | Wilcoxon Evals p |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `fortchuck` | 10/10 (100%) | 67.5 (1.0) | 67.5 (1.0) | 11.61 (2.42) | `8.45e-07` | `0.00e+00` | `0.00e+00` | `0.00e+00` | `87.046277` | `—` |
| `fortchuck-ABB` | 10/10 (100%) | 88.5 (2.8) | 88.5 (2.8) | 11.92 (1.73) | `7.14e-07` | `0.00e+00` | `0.00e+00` | `0.00e+00` | `87.046277` | `1.95e-03 (FC better)` |
| `Alternating-BCD` | 10/10 (100%) | 66.5 (2.0) | 66.5 (2.0) | 36.57 (3.66) | `8.99e-07` | `0.00e+00` | `0.00e+00` | `0.00e+00` | `87.046277` | `8.36e-01 (FC worse)` |
| `pymanopt-Warped-CG` | 0/10 (0%) | 5000.0 (0.0) | 5000.0 (0.0) | 1714.06 (38.61) | `2.38e-02` | `0.00e+00` | `0.00e+00` | `0.00e+00` | `87.048625` | `1.95e-03 (FC better)` |
| `pymanopt-TrustRegions` | 0/10 (0%) | 5000.0 (0.0) | 5000.0 (0.0) | 1503.69 (32.02) | `1.88e-04` | `0.00e+00` | `0.00e+00` | `0.00e+00` | `87.046492` | `1.95e-03 (FC better)` |
| `SciPy-L-BFGS-B` | 10/10 (100%) | 66.5 (2.0) | 66.5 (2.0) | 37.95 (4.98) | `8.99e-07` | `0.00e+00` | `0.00e+00` | `0.00e+00` | `87.046277` | `8.36e-01 (FC worse)` |
| `SciPy-SLSQP` | SKIP (`N_tot > 450`) | — | — | — | — | — | — | — | — | — |
| `SciPy-trust-constr` | SKIP (`N_tot > 450`) | — | — | — | — | — | — | — | — | — |

### `B7B`: Pure-Stiefel Tall Brockett & Orthogonal Procrustes

- **Regime**: `d=0, St(200,8) x St(24,24) (Pure Stiefel Branch A & Branch B)` (`d = 0`, `R = 2`, `N_tot = 2176`, intrinsic dim `1840`, FD gradient max rel err `4.872e-10`)

| Solver | Solve Rate | Raw Evals Med (IQR) | Penalized Evals Med (IQR) | Wall ms Med (IQR) | Median KKT Ref | Median Box Viol | Median Lin Viol | Median Orth Viol | Median Objective | Wilcoxon Evals p |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `fortchuck` | 10/10 (100%) | 501.5 (56.5) | 501.5 (56.5) | 441.71 (41.31) | `6.40e-07` | `0.00e+00` | `0.00e+00` | `6.60e-14` | `-66.949417` | `—` |
| `fortchuck-ABB` | 10/10 (100%) | 690.0 (75.2) | 690.0 (75.2) | 292.51 (50.79) | `6.79e-07` | `0.00e+00` | `0.00e+00` | `7.02e-14` | `-66.949417` | `1.95e-03 (FC better)` |
| `Alternating-BCD` | 9/10 (90%) | 1096.5 (126.2) | 1114.0 (191.2) | 1301.93 (105.04) | `9.00e-07` | `0.00e+00` | `0.00e+00` | `2.57e-15` | `-66.949417` | `1.95e-03 (FC better)` |
| `pymanopt-Warped-CG` | 9/10 (90%) | 1096.5 (126.2) | 1114.0 (191.2) | 1350.04 (184.21) | `9.00e-07` | `0.00e+00` | `0.00e+00` | `2.57e-15` | `-66.949417` | `1.95e-03 (FC better)` |
| `pymanopt-TrustRegions` | 10/10 (100%) | 1737.5 (210.0) | 1737.5 (210.0) | 1192.48 (87.22) | `2.65e-09` | `0.00e+00` | `0.00e+00` | `2.76e-15` | `-66.949417` | `1.95e-03 (FC better)` |
| `SciPy-L-BFGS-B` | 0/10 (0%) | 2210.0 (353.2) | 5000.0 (0.0) | 2024.72 (312.69) | `1.42e-03` | `0.00e+00` | `0.00e+00` | `5.46e-05` | `-66.950913` | `1.95e-03 (FC better)` |
| `SciPy-SLSQP` | SKIP (`N_tot > 450`) | — | — | — | — | — | — | — | — | — |
| `SciPy-trust-constr` | SKIP (`N_tot > 450`) | — | — | — | — | — | — | — | — | — |

### `B8A`: Trace-Constrained Ensemble Kohn-Sham DFT

- **Regime**: `z in [0,1]^12 (1^T z = 5) x St(120,12) (SLBQP + Branch A, hard trace)` (`d = 12`, `R = 1`, `N_tot = 1452`, intrinsic dim `1373`, linear constraint `a^T z = 5.0`, FD gradient max rel err `1.486e-10`)

| Solver | Solve Rate | Raw Evals Med (IQR) | Penalized Evals Med (IQR) | Wall ms Med (IQR) | Median KKT Ref | Median Box Viol | Median Lin Viol | Median Orth Viol | Median Objective | Wilcoxon Evals p |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `fortchuck` | 10/10 (100%) | 150.5 (61.8) | 150.5 (61.8) | 132.08 (84.19) | `7.87e-07` | `0.00e+00` | `0.00e+00` | `3.26e-14` | `-8.904018` | `—` |
| `fortchuck-ABB` | 10/10 (100%) | 209.5 (52.0) | 209.5 (52.0) | 91.75 (15.74) | `6.58e-07` | `0.00e+00` | `0.00e+00` | `2.64e-14` | `-8.904018` | `3.22e-01 (FC better)` |
| `Alternating-BCD` | 10/10 (100%) | 410.0 (110.5) | 410.0 (110.5) | 448.38 (95.32) | `8.01e-07` | `0.00e+00` | `0.00e+00` | `1.30e-15` | `-8.904018` | `1.95e-03 (FC better)` |
| `pymanopt-Warped-CG` | 0/10 (0%) | 5000.0 (0.0) | 5000.0 (0.0) | 7038.82 (609.81) | `7.87e-03` | `0.00e+00` | `5.61e-05` | `1.31e-15` | `-8.908933` | `1.95e-03 (FC better)` |
| `pymanopt-TrustRegions` | 0/10 (0%) | 5000.0 (0.0) | 5000.0 (0.0) | 5136.18 (273.24) | `8.47e-03` | `0.00e+00` | `4.48e-06` | `1.49e-15` | `-8.909349` | `1.95e-03 (FC better)` |
| `SciPy-L-BFGS-B` | 0/10 (0%) | 2717.5 (1812.2) | 5000.0 (0.0) | 3688.06 (2219.00) | `5.89e-05` | `0.00e+00` | `6.58e-08` | `7.34e-06` | `-8.904085` | `1.95e-03 (FC better)` |
| `SciPy-SLSQP` | SKIP (`N_tot > 450`) | — | — | — | — | — | — | — | — | — |
| `SciPy-trust-constr` | SKIP (`N_tot > 450`) | — | — | — | — | — | — | — | — | — |

### `B8B`: Long-Only Weighted Orthogonal Risk-Factor Parity

- **Regime**: `z in [0,0.15]^24 (a^T z = 1) x St(32,6) x St(6,6) (N_tot=252, Branch A & B)` (`d = 24`, `R = 2`, `N_tot = 252`, intrinsic dim `209`, linear constraint `a^T z = 1.0`, FD gradient max rel err `1.181e-11`)

| Solver | Solve Rate | Raw Evals Med (IQR) | Penalized Evals Med (IQR) | Wall ms Med (IQR) | Median KKT Ref | Median Box Viol | Median Lin Viol | Median Orth Viol | Median Objective | Wilcoxon Evals p |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `fortchuck` | 10/10 (100%) | 36.0 (1.0) | 36.0 (1.0) | 7.93 (1.09) | `7.32e-07` | `0.00e+00` | `0.00e+00` | `7.55e-15` | `-4.990125` | `—` |
| `fortchuck-ABB` | 10/10 (100%) | 63.5 (7.0) | 63.5 (7.0) | 11.53 (0.89) | `7.10e-07` | `0.00e+00` | `5.55e-17` | `8.79e-15` | `-4.990125` | `1.95e-03 (FC better)` |
| `Alternating-BCD` | 10/10 (100%) | 367.0 (65.8) | 367.0 (65.8) | 462.81 (62.18) | `8.88e-07` | `0.00e+00` | `1.67e-16` | `1.11e-15` | `-4.990125` | `1.95e-03 (FC better)` |
| `pymanopt-Warped-CG` | 0/10 (0%) | 5000.0 (0.0) | 5000.0 (0.0) | 8786.56 (2036.03) | `9.83e-04` | `0.00e+00` | `5.15e-05` | `9.63e-16` | `-4.990070` | `1.95e-03 (FC better)` |
| `pymanopt-TrustRegions` | 0/10 (0%) | 5000.0 (0.0) | 5000.0 (0.0) | 5930.38 (522.23) | `1.98e-04` | `0.00e+00` | `2.21e-05` | `9.29e-16` | `-4.990134` | `1.95e-03 (FC better)` |
| `SciPy-L-BFGS-B` | 0/10 (0%) | 1100.0 (169.2) | 5000.0 (0.0) | 1284.50 (191.38) | `1.79e-05` | `0.00e+00` | `1.26e-07` | `2.19e-06` | `-4.990130` | `1.95e-03 (FC better)` |
| `SciPy-SLSQP` | 10/10 (100%) | 37.5 (5.8) | 37.5 (5.8) | 479.05 (28.25) | `7.53e-07` | `0.00e+00` | `6.66e-16` | `5.00e-12` | `-4.990125` | `7.75e-01 (FC better)` |
| `SciPy-trust-constr` | 0/10 (0%) | 183.5 (21.5) | 5000.0 (0.0) | 3479.97 (807.23) | `2.88e-05` | `0.00e+00` | `2.78e-17` | `2.47e-14` | `-4.990124` | `1.95e-03 (FC better)` |

## 12. Per-Problem Breakdown — Third Blind Holdout Suite (`Seeds 301..310`)

### `B1`: Ensemble Kohn-Sham DFT Fractional Occupations

- **Regime**: `z in [0,1]^10 x St(120,10) (Branch A, metallic orbital-occupation coupling)` (`d = 10`, `R = 1`, `N_tot = 1210`, intrinsic dim `1155`, FD gradient max rel err `2.275e-10`)

| Solver | Solve Rate | Raw Evals Med (IQR) | Penalized Evals Med (IQR) | Wall ms Med (IQR) | Median KKT Ref | Median Box Viol | Median Lin Viol | Median Orth Viol | Median Objective | Wilcoxon Evals p |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `fortchuck` | 10/10 (100%) | 1909.5 (617.2) | 1909.5 (617.2) | 1316.67 (369.98) | `9.20e-07` | `0.00e+00` | `0.00e+00` | `6.26e-14` | `-4.372830` | `—` |
| `fortchuck-ABB` | 0/10 (0%) | 5000.0 (0.0) | 5000.0 (0.0) | 1814.43 (63.55) | `2.59e-05` | `0.00e+00` | `0.00e+00` | `5.66e-14` | `-4.377185` | `1.95e-03 (FC better)` |
| `Alternating-BCD` | 1/10 (10%) | 5000.0 (0.0) | 5000.0 (0.0) | 3341.34 (187.93) | `8.37e-04` | `0.00e+00` | `0.00e+00` | `1.33e-15` | `-4.349318` | `3.91e-03 (FC better)` |
| `pymanopt-Warped-CG` | 0/10 (0%) | 5000.0 (0.0) | 5000.0 (0.0) | 5263.93 (396.98) | `4.25e-04` | `0.00e+00` | `0.00e+00` | `1.19e-15` | `-4.386121` | `1.95e-03 (FC better)` |
| `pymanopt-TrustRegions` | 0/10 (0%) | 5000.0 (0.0) | 5000.0 (0.0) | 3396.10 (133.47) | `7.50e-03` | `0.00e+00` | `0.00e+00` | `1.49e-13` | `-4.386074` | `1.95e-03 (FC better)` |
| `SciPy-L-BFGS-B` | 0/10 (0%) | 5000.0 (0.0) | 5000.0 (0.0) | 4593.80 (218.36) | `9.15e-04` | `0.00e+00` | `0.00e+00` | `3.19e-06` | `-4.368334` | `1.95e-03 (FC better)` |
| `SciPy-SLSQP` | SKIP (`N_tot > 450`) | — | — | — | — | — | — | — | — | — |
| `SciPy-trust-constr` | SKIP (`N_tot > 450`) | — | — | — | — | — | — | — | — | — |

### `B2A`: Amplitude-Limited Blind Source Separation (ICA)

- **Regime**: `Z in [-2,2]^{5x150} (d=750) x St(20,5) (Branch A, nonconvex log-cosh)` (`d = 750`, `R = 1`, `N_tot = 850`, intrinsic dim `835`, FD gradient max rel err `1.702e-09`)

| Solver | Solve Rate | Raw Evals Med (IQR) | Penalized Evals Med (IQR) | Wall ms Med (IQR) | Median KKT Ref | Median Box Viol | Median Lin Viol | Median Orth Viol | Median Objective | Wilcoxon Evals p |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `fortchuck` | 10/10 (100%) | 536.0 (196.0) | 536.0 (196.0) | 154.94 (48.68) | `8.88e-07` | `0.00e+00` | `0.00e+00` | `5.30e-14` | `-309.928001` | `—` |
| `fortchuck-ABB` | 7/10 (70%) | 3488.5 (1798.2) | 3488.5 (1798.2) | 745.66 (385.03) | `9.84e-07` | `0.00e+00` | `0.00e+00` | `6.34e-14` | `-308.599616` | `1.95e-03 (FC better)` |
| `Alternating-BCD` | 3/10 (30%) | 5000.0 (345.0) | 5000.0 (345.0) | 4301.17 (761.73) | `1.07e-03` | `0.00e+00` | `0.00e+00` | `7.67e-16` | `-307.215217` | `1.95e-03 (FC better)` |
| `pymanopt-Warped-CG` | 0/10 (0%) | 5000.0 (0.0) | 5000.0 (0.0) | 5673.81 (1262.20) | `5.00e-01` | `0.00e+00` | `0.00e+00` | `7.00e-16` | `-306.743575` | `1.95e-03 (FC better)` |
| `pymanopt-TrustRegions` | 0/10 (0%) | 5000.0 (0.0) | 5000.0 (0.0) | 3518.02 (465.16) | `7.23e-01` | `0.00e+00` | `0.00e+00` | `3.03e-11` | `-307.111514` | `1.95e-03 (FC better)` |
| `SciPy-L-BFGS-B` | 0/10 (0%) | 5000.0 (0.0) | 5000.0 (0.0) | 3941.67 (602.58) | `3.95e-02` | `0.00e+00` | `0.00e+00` | `3.45e-05` | `-308.408565` | `1.95e-03 (FC better)` |
| `SciPy-SLSQP` | SKIP (`N_tot > 450`) | — | — | — | — | — | — | — | — | — |
| `SciPy-trust-constr` | SKIP (`N_tot > 450`) | — | — | — | — | — | — | — | — | — |

### `B2B`: Smooth Penalty-Split Non-Negative Sparse PCA

- **Regime**: `Z in [0,5]^{120x6} (d=720) x St(120,6) (Branch A, ~80% clamped Z=0, FM-2)` (`d = 720`, `R = 1`, `N_tot = 1440`, intrinsic dim `1419`, FD gradient max rel err `1.215e-08`)

| Solver | Solve Rate | Raw Evals Med (IQR) | Penalized Evals Med (IQR) | Wall ms Med (IQR) | Median KKT Ref | Median Box Viol | Median Lin Viol | Median Orth Viol | Median Objective | Wilcoxon Evals p |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `fortchuck` | 10/10 (100%) | 100.0 (13.2) | 100.0 (13.2) | 32.03 (6.53) | `6.11e-07` | `0.00e+00` | `0.00e+00` | `1.09e-14` | `8.722752` | `—` |
| `fortchuck-ABB` | 10/10 (100%) | 887.0 (926.5) | 887.0 (926.5) | 167.87 (169.37) | `9.16e-07` | `0.00e+00` | `0.00e+00` | `3.05e-14` | `8.485785` | `1.95e-03 (FC better)` |
| `Alternating-BCD` | 0/10 (0%) | 4325.5 (1129.5) | 5000.0 (0.0) | 4038.10 (751.32) | `8.46e-05` | `0.00e+00` | `0.00e+00` | `1.15e-15` | `8.264379` | `1.95e-03 (FC better)` |
| `pymanopt-Warped-CG` | 5/10 (50%) | 1796.0 (3645.8) | 4706.5 (2226.8) | 2032.46 (4026.01) | `8.22e-03` | `0.00e+00` | `0.00e+00` | `1.19e-15` | `14.206347` | `1.95e-03 (FC better)` |
| `pymanopt-TrustRegions` | 8/10 (80%) | 2960.5 (1695.2) | 2960.5 (1695.2) | 2133.59 (817.99) | `8.01e-07` | `0.00e+00` | `0.00e+00` | `1.08e-15` | `8.231148` | `1.95e-03 (FC better)` |
| `SciPy-L-BFGS-B` | 0/10 (0%) | 700.5 (478.8) | 5000.0 (0.0) | 592.63 (382.00) | `1.96e-05` | `0.00e+00` | `0.00e+00` | `5.00e-06` | `9.289392` | `1.95e-03 (FC better)` |
| `SciPy-SLSQP` | SKIP (`N_tot > 450`) | — | — | — | — | — | — | — | — | — |
| `SciPy-trust-constr` | SKIP (`N_tot > 450`) | — | — | — | — | — | — | — | — | — |

### `B3`: Multi-Camera Anisotropic Bounded-Scale Rig Wahba

- **Regime**: `z in B^15 x St(3,3)^4 (Branch B SO(3)^4, R=4, kappa(W)=10^3)` (`d = 15`, `R = 4`, `N_tot = 51`, intrinsic dim `27`, FD gradient max rel err `1.384e-10`)

| Solver | Solve Rate | Raw Evals Med (IQR) | Penalized Evals Med (IQR) | Wall ms Med (IQR) | Median KKT Ref | Median Box Viol | Median Lin Viol | Median Orth Viol | Median Objective | Wilcoxon Evals p |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `fortchuck` | 10/10 (100%) | 45.5 (6.8) | 45.5 (6.8) | 14.08 (4.08) | `4.05e-07` | `0.00e+00` | `0.00e+00` | `5.07e-15` | `2.176801` | `—` |
| `fortchuck-ABB` | 10/10 (100%) | 81.5 (13.2) | 81.5 (13.2) | 23.42 (6.48) | `6.63e-07` | `0.00e+00` | `0.00e+00` | `7.31e-15` | `2.176801` | `1.95e-03 (FC better)` |
| `Alternating-BCD` | 9/10 (90%) | 310.0 (14.0) | 310.0 (14.0) | 497.68 (50.36) | `8.23e-07` | `0.00e+00` | `0.00e+00` | `6.90e-16` | `2.176801` | `1.95e-03 (FC better)` |
| `pymanopt-Warped-CG` | 10/10 (100%) | 735.0 (206.8) | 735.0 (206.8) | 1598.59 (603.58) | `8.37e-07` | `0.00e+00` | `0.00e+00` | `7.52e-16` | `2.176801` | `1.95e-03 (FC better)` |
| `pymanopt-TrustRegions` | 10/10 (100%) | 639.0 (119.2) | 639.0 (119.2) | 789.29 (118.53) | `6.40e-07` | `0.00e+00` | `0.00e+00` | `7.01e-16` | `2.176801` | `1.95e-03 (FC better)` |
| `SciPy-L-BFGS-B` | 0/10 (0%) | 813.0 (65.0) | 5000.0 (0.0) | 846.44 (94.91) | `2.24e-05` | `0.00e+00` | `0.00e+00` | `4.83e-06` | `2.176772` | `1.95e-03 (FC better)` |
| `SciPy-SLSQP` | 10/10 (100%) | 80.0 (8.2) | 80.0 (8.2) | 85.41 (16.28) | `2.32e-07` | `0.00e+00` | `0.00e+00` | `1.93e-14` | `2.176801` | `1.95e-03 (FC better)` |
| `SciPy-trust-constr` | 10/10 (100%) | 83.5 (21.8) | 83.5 (21.8) | 494.87 (211.90) | `8.10e-07` | `0.00e+00` | `0.00e+00` | `5.66e-15` | `2.176802` | `1.95e-03 (FC better)` |

### `B4`: Bounded-Variance ML Common Principal Components

- **Regime**: `z in [0.05,20]^150 x St(30,30) (Branch B O(30), log barrier, 1.6e5 scale ratio)` (`d = 150`, `R = 1`, `N_tot = 1050`, intrinsic dim `585`, FD gradient max rel err `7.163e-10`)

| Solver | Solve Rate | Raw Evals Med (IQR) | Penalized Evals Med (IQR) | Wall ms Med (IQR) | Median KKT Ref | Median Box Viol | Median Lin Viol | Median Orth Viol | Median Objective | Wilcoxon Evals p |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `fortchuck` | 10/10 (100%) | 1750.5 (163.0) | 1750.5 (163.0) | 1339.38 (180.90) | `6.64e-07` | `0.00e+00` | `0.00e+00` | `5.01e-14` | `17.596323` | `—` |
| `fortchuck-ABB` | 8/10 (80%) | 4379.5 (813.0) | 4379.5 (813.0) | 2337.44 (269.31) | `7.02e-07` | `0.00e+00` | `0.00e+00` | `5.50e-14` | `17.596323` | `1.95e-03 (FC better)` |
| `Alternating-BCD` | 0/10 (0%) | 5000.0 (0.0) | 5000.0 (0.0) | 4646.57 (1094.44) | `7.83e-04` | `0.00e+00` | `0.00e+00` | `3.34e-15` | `17.596324` | `1.95e-03 (FC better)` |
| `pymanopt-Warped-CG` | 0/10 (0%) | 5000.0 (0.0) | 5000.0 (0.0) | 7146.86 (1284.70) | `6.23e-03` | `0.00e+00` | `0.00e+00` | `3.04e-15` | `17.599196` | `1.95e-03 (FC better)` |
| `pymanopt-TrustRegions` | 0/10 (0%) | 5000.0 (0.0) | 5000.0 (0.0) | 4667.13 (525.76) | `4.15e-03` | `0.00e+00` | `0.00e+00` | `1.40e-13` | `17.597122` | `1.95e-03 (FC better)` |
| `SciPy-L-BFGS-B` | 0/10 (0%) | 5000.0 (0.0) | 5000.0 (0.0) | 5359.46 (392.96) | `5.19e-01` | `0.00e+00` | `0.00e+00` | `5.74e-06` | `20.590865` | `1.95e-03 (FC better)` |
| `SciPy-SLSQP` | SKIP (`N_tot > 450`) | — | — | — | — | — | — | — | — | — |
| `SciPy-trust-constr` | SKIP (`N_tot > 450`) | — | — | — | — | — | — | — | — | — |

### `B5`: Bounded Factor Correlation Matrix Calibration

- **Regime**: `z in B^110 x St(100,10) (Branch A, quartic valley + clamped psi)` (`d = 110`, `R = 1`, `N_tot = 1110`, intrinsic dim `1055`, FD gradient max rel err `7.298e-10`)

| Solver | Solve Rate | Raw Evals Med (IQR) | Penalized Evals Med (IQR) | Wall ms Med (IQR) | Median KKT Ref | Median Box Viol | Median Lin Viol | Median Orth Viol | Median Objective | Wilcoxon Evals p |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `fortchuck` | 10/10 (100%) | 1144.5 (210.0) | 1144.5 (210.0) | 772.10 (97.42) | `9.36e-07` | `0.00e+00` | `0.00e+00` | `6.46e-14` | `34.668813` | `—` |
| `fortchuck-ABB` | 0/10 (0%) | 5000.0 (0.0) | 5000.0 (0.0) | 2013.87 (53.26) | `3.78e-04` | `0.00e+00` | `0.00e+00` | `4.61e-14` | `34.668819` | `1.95e-03 (FC better)` |
| `Alternating-BCD` | 0/10 (0%) | 5000.0 (0.0) | 5000.0 (0.0) | 3899.44 (120.82) | `6.51e-03` | `0.00e+00` | `0.00e+00` | `1.15e-15` | `34.669151` | `1.95e-03 (FC better)` |
| `pymanopt-Warped-CG` | 0/10 (0%) | 5000.0 (0.0) | 5000.0 (0.0) | 5968.92 (380.28) | `3.22e-02` | `0.00e+00` | `0.00e+00` | `1.37e-15` | `34.671898` | `1.95e-03 (FC better)` |
| `pymanopt-TrustRegions` | 0/10 (0%) | 5000.0 (0.0) | 5000.0 (0.0) | 3874.64 (169.40) | `2.78e-02` | `0.00e+00` | `0.00e+00` | `3.55e-12` | `34.669870` | `1.95e-03 (FC better)` |
| `SciPy-L-BFGS-B` | 0/10 (0%) | 3007.5 (1162.5) | 5000.0 (0.0) | 3012.95 (1009.36) | `4.58e-05` | `0.00e+00` | `0.00e+00` | `2.16e-10` | `34.668813` | `1.95e-03 (FC better)` |
| `SciPy-SLSQP` | SKIP (`N_tot > 450`) | — | — | — | — | — | — | — | — | — |
| `SciPy-trust-constr` | SKIP (`N_tot > 450`) | — | — | — | — | — | — | — | — | — |

### `B6`: Box-Coupled Brockett with Exact Ky Fan QP Truth

- **Regime**: `z in [0.1,5]^10 x St(150,10) (Branch A, kappa(M)=10^4, certified f_star)` (`d = 10`, `R = 1`, `N_tot = 1510`, intrinsic dim `1455`, FD gradient max rel err `2.405e-10`)

| Solver | Solve Rate | Raw Evals Med (IQR) | Penalized Evals Med (IQR) | Wall ms Med (IQR) | Median KKT Ref | Median Box Viol | Median Lin Viol | Median Orth Viol | Median Objective | Wilcoxon Evals p |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `fortchuck` | 10/10 (100%) | 1942.5 (265.0) | 1942.5 (265.0) | 1536.62 (224.32) | `9.02e-07` | `0.00e+00` | `0.00e+00` | `6.25e-14` | `-10.445510` | `—` |
| `fortchuck-ABB` | 9/10 (90%) | 4563.0 (855.0) | 4563.0 (855.0) | 1651.60 (359.52) | `7.22e-07` | `0.00e+00` | `0.00e+00` | `4.76e-14` | `-10.445510` | `1.95e-03 (FC better)` |
| `Alternating-BCD` | 0/10 (0%) | 5000.0 (0.0) | 5000.0 (0.0) | 3139.84 (231.89) | `1.26e-02` | `0.00e+00` | `0.00e+00` | `9.63e-16` | `-10.443719` | `1.95e-03 (FC better)` |
| `pymanopt-Warped-CG` | 0/10 (0%) | 5000.0 (0.0) | 5000.0 (0.0) | 4844.96 (815.23) | `1.60e-03` | `0.00e+00` | `0.00e+00` | `9.81e-16` | `-10.445302` | `1.95e-03 (FC better)` |
| `pymanopt-TrustRegions` | 0/10 (0%) | 5000.0 (0.0) | 5000.0 (0.0) | 3125.18 (792.84) | `2.53e-03` | `0.00e+00` | `0.00e+00` | `1.83e-14` | `-10.445428` | `1.95e-03 (FC better)` |
| `SciPy-L-BFGS-B` | 0/10 (0%) | 5000.0 (0.0) | 5000.0 (0.0) | 4538.45 (144.32) | `3.67e-03` | `0.00e+00` | `0.00e+00` | `1.50e-05` | `-10.445568` | `1.95e-03 (FC better)` |
| `SciPy-SLSQP` | SKIP (`N_tot > 450`) | — | — | — | — | — | — | — | — | — |
| `SciPy-trust-constr` | SKIP (`N_tot > 450`) | — | — | — | — | — | — | — | — | — |

### `B7A`: Pure-Box Active-Bound Chained Rosenbrock + Quadratic

- **Regime**: `z in [l,u]^500, R=0 (incl. 5 fixed l_i==u_i, ~60% active bounds)` (`d = 500`, `R = 0`, `N_tot = 500`, intrinsic dim `500`, FD gradient max rel err `7.747e-10`)

| Solver | Solve Rate | Raw Evals Med (IQR) | Penalized Evals Med (IQR) | Wall ms Med (IQR) | Median KKT Ref | Median Box Viol | Median Lin Viol | Median Orth Viol | Median Objective | Wilcoxon Evals p |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `fortchuck` | 10/10 (100%) | 68.0 (3.5) | 68.0 (3.5) | 10.98 (2.29) | `8.30e-07` | `0.00e+00` | `0.00e+00` | `0.00e+00` | `87.046277` | `—` |
| `fortchuck-ABB` | 10/10 (100%) | 90.0 (4.8) | 90.0 (4.8) | 11.94 (2.13) | `7.35e-07` | `0.00e+00` | `0.00e+00` | `0.00e+00` | `87.046277` | `1.95e-03 (FC better)` |
| `Alternating-BCD` | 10/10 (100%) | 67.0 (1.8) | 67.0 (1.8) | 36.65 (1.63) | `9.05e-07` | `0.00e+00` | `0.00e+00` | `0.00e+00` | `87.046277` | `3.44e-01 (FC worse)` |
| `pymanopt-Warped-CG` | 0/10 (0%) | 5000.0 (0.0) | 5000.0 (0.0) | 1792.26 (70.20) | `3.07e-01` | `0.00e+00` | `0.00e+00` | `0.00e+00` | `87.064904` | `1.95e-03 (FC better)` |
| `pymanopt-TrustRegions` | 0/10 (0%) | 5000.0 (0.0) | 5000.0 (0.0) | 1608.25 (37.35) | `1.16e-04` | `0.00e+00` | `0.00e+00` | `0.00e+00` | `87.046429` | `1.95e-03 (FC better)` |
| `SciPy-L-BFGS-B` | 10/10 (100%) | 67.0 (1.8) | 67.0 (1.8) | 36.38 (1.90) | `9.05e-07` | `0.00e+00` | `0.00e+00` | `0.00e+00` | `87.046277` | `3.44e-01 (FC worse)` |
| `SciPy-SLSQP` | SKIP (`N_tot > 450`) | — | — | — | — | — | — | — | — | — |
| `SciPy-trust-constr` | SKIP (`N_tot > 450`) | — | — | — | — | — | — | — | — | — |

### `B7B`: Pure-Stiefel Tall Brockett & Orthogonal Procrustes

- **Regime**: `d=0, St(200,8) x St(24,24) (Pure Stiefel Branch A & Branch B)` (`d = 0`, `R = 2`, `N_tot = 2176`, intrinsic dim `1840`, FD gradient max rel err `4.765e-10`)

| Solver | Solve Rate | Raw Evals Med (IQR) | Penalized Evals Med (IQR) | Wall ms Med (IQR) | Median KKT Ref | Median Box Viol | Median Lin Viol | Median Orth Viol | Median Objective | Wilcoxon Evals p |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `fortchuck` | 10/10 (100%) | 490.5 (59.8) | 490.5 (59.8) | 516.77 (76.54) | `6.92e-07` | `0.00e+00` | `0.00e+00` | `5.54e-14` | `-66.949417` | `—` |
| `fortchuck-ABB` | 10/10 (100%) | 637.5 (31.8) | 637.5 (31.8) | 317.49 (30.83) | `7.26e-07` | `0.00e+00` | `0.00e+00` | `6.12e-14` | `-66.949417` | `1.95e-03 (FC better)` |
| `Alternating-BCD` | 10/10 (100%) | 1064.0 (142.2) | 1064.0 (142.2) | 1281.05 (218.27) | `9.06e-07` | `0.00e+00` | `0.00e+00` | `2.65e-15` | `-66.949417` | `1.95e-03 (FC better)` |
| `pymanopt-Warped-CG` | 10/10 (100%) | 1064.0 (142.2) | 1064.0 (142.2) | 1295.12 (225.78) | `9.06e-07` | `0.00e+00` | `0.00e+00` | `2.65e-15` | `-66.949417` | `1.95e-03 (FC better)` |
| `pymanopt-TrustRegions` | 10/10 (100%) | 1504.0 (235.2) | 1504.0 (235.2) | 1044.80 (203.31) | `2.06e-08` | `0.00e+00` | `0.00e+00` | `2.60e-15` | `-66.949417` | `1.95e-03 (FC better)` |
| `SciPy-L-BFGS-B` | 0/10 (0%) | 2636.5 (954.8) | 5000.0 (0.0) | 2361.48 (906.76) | `1.42e-03` | `0.00e+00` | `0.00e+00` | `5.46e-05` | `-66.884604` | `1.95e-03 (FC better)` |
| `SciPy-SLSQP` | SKIP (`N_tot > 450`) | — | — | — | — | — | — | — | — | — |
| `SciPy-trust-constr` | SKIP (`N_tot > 450`) | — | — | — | — | — | — | — | — | — |

### `B8A`: Trace-Constrained Ensemble Kohn-Sham DFT

- **Regime**: `z in [0,1]^12 (1^T z = 5) x St(120,12) (SLBQP + Branch A, hard trace)` (`d = 12`, `R = 1`, `N_tot = 1452`, intrinsic dim `1373`, linear constraint `a^T z = 5.0`, FD gradient max rel err `1.275e-10`)

| Solver | Solve Rate | Raw Evals Med (IQR) | Penalized Evals Med (IQR) | Wall ms Med (IQR) | Median KKT Ref | Median Box Viol | Median Lin Viol | Median Orth Viol | Median Objective | Wilcoxon Evals p |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `fortchuck` | 10/10 (100%) | 146.5 (30.0) | 146.5 (30.0) | 148.94 (29.85) | `8.22e-07` | `0.00e+00` | `0.00e+00` | `3.15e-14` | `-8.904018` | `—` |
| `fortchuck-ABB` | 10/10 (100%) | 225.0 (95.0) | 225.0 (95.0) | 106.72 (32.05) | `8.32e-07` | `0.00e+00` | `0.00e+00` | `2.75e-14` | `-8.904018` | `1.31e-01 (FC better)` |
| `Alternating-BCD` | 10/10 (100%) | 400.5 (159.0) | 400.5 (159.0) | 397.47 (153.31) | `7.17e-07` | `0.00e+00` | `0.00e+00` | `1.28e-15` | `-8.904018` | `3.91e-03 (FC better)` |
| `pymanopt-Warped-CG` | 0/10 (0%) | 5000.0 (0.0) | 5000.0 (0.0) | 6913.63 (333.99) | `2.98e-03` | `0.00e+00` | `6.29e-05` | `1.64e-15` | `-8.908905` | `1.95e-03 (FC better)` |
| `pymanopt-TrustRegions` | 0/10 (0%) | 5000.0 (0.0) | 5000.0 (0.0) | 5053.51 (106.55) | `7.88e-03` | `0.00e+00` | `4.35e-06` | `1.55e-15` | `-8.909342` | `1.95e-03 (FC better)` |
| `SciPy-L-BFGS-B` | 0/10 (0%) | 4509.5 (2054.5) | 5000.0 (0.0) | 6701.92 (3690.48) | `1.76e-04` | `0.00e+00` | `9.72e-08` | `7.34e-06` | `-8.904045` | `1.95e-03 (FC better)` |
| `SciPy-SLSQP` | SKIP (`N_tot > 450`) | — | — | — | — | — | — | — | — | — |
| `SciPy-trust-constr` | SKIP (`N_tot > 450`) | — | — | — | — | — | — | — | — | — |

### `B8B`: Long-Only Weighted Orthogonal Risk-Factor Parity

- **Regime**: `z in [0,0.15]^24 (a^T z = 1) x St(32,6) x St(6,6) (N_tot=252, Branch A & B)` (`d = 24`, `R = 2`, `N_tot = 252`, intrinsic dim `209`, linear constraint `a^T z = 1.0`, FD gradient max rel err `1.301e-11`)

| Solver | Solve Rate | Raw Evals Med (IQR) | Penalized Evals Med (IQR) | Wall ms Med (IQR) | Median KKT Ref | Median Box Viol | Median Lin Viol | Median Orth Viol | Median Objective | Wilcoxon Evals p |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `fortchuck` | 10/10 (100%) | 36.5 (3.0) | 36.5 (3.0) | 9.15 (1.59) | `6.21e-07` | `0.00e+00` | `5.55e-17` | `8.73e-15` | `-4.990125` | `—` |
| `fortchuck-ABB` | 10/10 (100%) | 60.5 (8.5) | 60.5 (8.5) | 12.02 (4.56) | `7.42e-07` | `0.00e+00` | `5.55e-17` | `9.47e-15` | `-4.990125` | `1.95e-03 (FC better)` |
| `Alternating-BCD` | 10/10 (100%) | 320.5 (118.2) | 320.5 (118.2) | 449.88 (97.11) | `8.51e-07` | `0.00e+00` | `2.22e-16` | `1.07e-15` | `-4.990125` | `1.95e-03 (FC better)` |
| `pymanopt-Warped-CG` | 0/10 (0%) | 5000.0 (0.0) | 5000.0 (0.0) | 8429.40 (919.51) | `8.18e-04` | `0.00e+00` | `5.17e-05` | `9.81e-16` | `-4.990097` | `1.95e-03 (FC better)` |
| `pymanopt-TrustRegions` | 0/10 (0%) | 5000.0 (0.0) | 5000.0 (0.0) | 5964.37 (679.88) | `3.91e-04` | `0.00e+00` | `3.06e-05` | `9.57e-16` | `-4.990137` | `1.95e-03 (FC better)` |
| `SciPy-L-BFGS-B` | 0/10 (0%) | 994.0 (68.2) | 5000.0 (0.0) | 1208.13 (153.75) | `2.38e-05` | `0.00e+00` | `1.26e-07` | `2.19e-06` | `-4.990130` | `1.95e-03 (FC better)` |
| `SciPy-SLSQP` | 10/10 (100%) | 36.0 (5.8) | 36.0 (5.8) | 480.74 (46.18) | `6.70e-07` | `0.00e+00` | `7.77e-16` | `4.72e-12` | `-4.990125` | `9.80e-01 (FC better)` |
| `SciPy-trust-constr` | 0/10 (0%) | 205.5 (14.2) | 5000.0 (0.0) | 3509.94 (362.49) | `2.79e-05` | `0.00e+00` | `5.55e-17` | `1.06e-13` | `-4.990124` | `1.95e-03 (FC better)` |

## 13. Per-Problem Breakdown — Virgin Fourth Blind Holdout Suite (`Seeds 401..410`)

### `B1`: Ensemble Kohn-Sham DFT Fractional Occupations

- **Regime**: `z in [0,1]^10 x St(120,10) (Branch A, metallic orbital-occupation coupling)` (`d = 10`, `R = 1`, `N_tot = 1210`, intrinsic dim `1155`, FD gradient max rel err `2.281e-10`)

| Solver | Solve Rate | Raw Evals Med (IQR) | Penalized Evals Med (IQR) | Wall ms Med (IQR) | Median KKT Ref | Median Box Viol | Median Lin Viol | Median Orth Viol | Median Objective | Wilcoxon Evals p |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `fortchuck` | 10/10 (100%) | 2339.0 (914.2) | 2339.0 (914.2) | 1838.31 (650.40) | `8.78e-07` | `0.00e+00` | `0.00e+00` | `3.26e-14` | `-4.377192` | `—` |
| `fortchuck-ABB` | 0/10 (0%) | 5000.0 (0.0) | 5000.0 (0.0) | 2053.99 (38.20) | `1.09e-05` | `0.00e+00` | `0.00e+00` | `4.71e-14` | `-4.377192` | `1.95e-03 (FC better)` |
| `Alternating-BCD` | 3/10 (30%) | 5000.0 (3397.5) | 5000.0 (3397.5) | 3073.88 (2124.67) | `1.50e-04` | `0.00e+00` | `0.00e+00` | `1.27e-15` | `-4.349293` | `1.93e-01 (FC better)` |
| `pymanopt-Warped-CG` | 0/10 (0%) | 5000.0 (0.0) | 5000.0 (0.0) | 4470.92 (109.16) | `5.01e-04` | `0.00e+00` | `0.00e+00` | `1.37e-15` | `-4.386124` | `1.95e-03 (FC better)` |
| `pymanopt-TrustRegions` | 0/10 (0%) | 5000.0 (0.0) | 5000.0 (0.0) | 2920.47 (44.81) | `6.86e-03` | `0.00e+00` | `0.00e+00` | `7.69e-14` | `-4.386098` | `1.95e-03 (FC better)` |
| `SciPy-L-BFGS-B` | 0/10 (0%) | 5000.0 (0.0) | 5000.0 (0.0) | 4636.32 (109.70) | `6.99e-04` | `0.00e+00` | `0.00e+00` | `3.22e-06` | `-4.368460` | `1.95e-03 (FC better)` |
| `SciPy-SLSQP` | SKIP (`N_tot > 450`) | — | — | — | — | — | — | — | — | — |
| `SciPy-trust-constr` | SKIP (`N_tot > 450`) | — | — | — | — | — | — | — | — | — |

### `B2A`: Amplitude-Limited Blind Source Separation (ICA)

- **Regime**: `Z in [-2,2]^{5x150} (d=750) x St(20,5) (Branch A, nonconvex log-cosh)` (`d = 750`, `R = 1`, `N_tot = 850`, intrinsic dim `835`, FD gradient max rel err `1.846e-09`)

| Solver | Solve Rate | Raw Evals Med (IQR) | Penalized Evals Med (IQR) | Wall ms Med (IQR) | Median KKT Ref | Median Box Viol | Median Lin Viol | Median Orth Viol | Median Objective | Wilcoxon Evals p |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `fortchuck` | 10/10 (100%) | 607.0 (272.8) | 607.0 (272.8) | 192.77 (85.25) | `8.59e-07` | `0.00e+00` | `0.00e+00` | `3.69e-14` | `-307.220668` | `—` |
| `fortchuck-ABB` | 5/10 (50%) | 4450.0 (1639.0) | 4979.0 (1115.8) | 1025.61 (366.16) | `1.32e-06` | `0.00e+00` | `0.00e+00` | `5.67e-14` | `-305.114902` | `1.95e-03 (FC better)` |
| `Alternating-BCD` | 0/10 (0%) | 5000.0 (0.0) | 5000.0 (0.0) | 3586.60 (36.70) | `2.50e-04` | `0.00e+00` | `0.00e+00` | `5.82e-16` | `-305.917650` | `1.95e-03 (FC better)` |
| `pymanopt-Warped-CG` | 0/10 (0%) | 5000.0 (0.0) | 5000.0 (0.0) | 4362.96 (65.13) | `4.14e-01` | `0.00e+00` | `0.00e+00` | `8.06e-16` | `-303.337352` | `1.95e-03 (FC better)` |
| `pymanopt-TrustRegions` | 0/10 (0%) | 5000.0 (0.0) | 5000.0 (0.0) | 2791.07 (56.04) | `5.55e-01` | `0.00e+00` | `0.00e+00` | `3.09e-11` | `-306.299844` | `1.95e-03 (FC better)` |
| `SciPy-L-BFGS-B` | 0/10 (0%) | 5000.0 (0.0) | 5000.0 (0.0) | 3862.53 (84.68) | `1.07e-01` | `0.00e+00` | `0.00e+00` | `2.37e-05` | `-305.823734` | `1.95e-03 (FC better)` |
| `SciPy-SLSQP` | SKIP (`N_tot > 450`) | — | — | — | — | — | — | — | — | — |
| `SciPy-trust-constr` | SKIP (`N_tot > 450`) | — | — | — | — | — | — | — | — | — |

### `B2B`: Smooth Penalty-Split Non-Negative Sparse PCA

- **Regime**: `Z in [0,5]^{120x6} (d=720) x St(120,6) (Branch A, ~80% clamped Z=0, FM-2)` (`d = 720`, `R = 1`, `N_tot = 1440`, intrinsic dim `1419`, FD gradient max rel err `1.582e-08`)

| Solver | Solve Rate | Raw Evals Med (IQR) | Penalized Evals Med (IQR) | Wall ms Med (IQR) | Median KKT Ref | Median Box Viol | Median Lin Viol | Median Orth Viol | Median Objective | Wilcoxon Evals p |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `fortchuck` | 10/10 (100%) | 119.5 (34.8) | 119.5 (34.8) | 44.44 (14.84) | `5.19e-07` | `0.00e+00` | `0.00e+00` | `1.16e-14` | `8.169200` | `—` |
| `fortchuck-ABB` | 10/10 (100%) | 556.0 (355.2) | 556.0 (355.2) | 121.76 (82.72) | `8.56e-07` | `0.00e+00` | `0.00e+00` | `3.34e-14` | `8.267101` | `1.95e-03 (FC better)` |
| `Alternating-BCD` | 0/10 (0%) | 4583.0 (1493.2) | 5000.0 (0.0) | 3264.20 (972.69) | `8.45e-04` | `0.00e+00` | `0.00e+00` | `8.81e-16` | `8.013479` | `1.95e-03 (FC better)` |
| `pymanopt-Warped-CG` | 5/10 (50%) | 4322.0 (3939.0) | 4322.0 (3939.0) | 3679.31 (3361.80) | `6.17e-05` | `0.00e+00` | `0.00e+00` | `1.05e-15` | `13.839006` | `1.95e-03 (FC better)` |
| `pymanopt-TrustRegions` | 7/10 (70%) | 2920.0 (1929.2) | 2920.0 (1929.2) | 1666.19 (1038.93) | `8.33e-07` | `0.00e+00` | `0.00e+00` | `7.77e-16` | `8.302527` | `1.95e-03 (FC better)` |
| `SciPy-L-BFGS-B` | 0/10 (0%) | 668.0 (391.0) | 5000.0 (0.0) | 544.86 (310.01) | `4.74e-05` | `0.00e+00` | `0.00e+00` | `6.27e-06` | `10.745824` | `1.95e-03 (FC better)` |
| `SciPy-SLSQP` | SKIP (`N_tot > 450`) | — | — | — | — | — | — | — | — | — |
| `SciPy-trust-constr` | SKIP (`N_tot > 450`) | — | — | — | — | — | — | — | — | — |

### `B3`: Multi-Camera Anisotropic Bounded-Scale Rig Wahba

- **Regime**: `z in B^15 x St(3,3)^4 (Branch B SO(3)^4, R=4, kappa(W)=10^3)` (`d = 15`, `R = 4`, `N_tot = 51`, intrinsic dim `27`, FD gradient max rel err `1.367e-10`)

| Solver | Solve Rate | Raw Evals Med (IQR) | Penalized Evals Med (IQR) | Wall ms Med (IQR) | Median KKT Ref | Median Box Viol | Median Lin Viol | Median Orth Viol | Median Objective | Wilcoxon Evals p |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `fortchuck` | 10/10 (100%) | 50.5 (8.2) | 50.5 (8.2) | 15.81 (4.63) | `5.74e-07` | `0.00e+00` | `0.00e+00` | `5.46e-15` | `2.176801` | `—` |
| `fortchuck-ABB` | 10/10 (100%) | 78.5 (12.0) | 78.5 (12.0) | 24.41 (7.95) | `5.88e-07` | `0.00e+00` | `0.00e+00` | `6.63e-15` | `2.176801` | `1.95e-03 (FC better)` |
| `Alternating-BCD` | 10/10 (100%) | 309.5 (23.5) | 309.5 (23.5) | 389.23 (30.02) | `9.47e-07` | `0.00e+00` | `0.00e+00` | `7.40e-16` | `2.176801` | `1.95e-03 (FC better)` |
| `pymanopt-Warped-CG` | 10/10 (100%) | 879.5 (300.0) | 879.5 (300.0) | 1477.86 (515.63) | `8.39e-07` | `0.00e+00` | `0.00e+00` | `7.54e-16` | `2.176801` | `1.95e-03 (FC better)` |
| `pymanopt-TrustRegions` | 10/10 (100%) | 655.0 (115.0) | 655.0 (115.0) | 625.27 (73.19) | `6.72e-07` | `0.00e+00` | `0.00e+00` | `6.41e-16` | `2.176801` | `1.95e-03 (FC better)` |
| `SciPy-L-BFGS-B` | 0/10 (0%) | 815.5 (242.0) | 5000.0 (0.0) | 790.79 (217.69) | `2.27e-05` | `0.00e+00` | `0.00e+00` | `4.83e-06` | `2.176772` | `1.95e-03 (FC better)` |
| `SciPy-SLSQP` | 10/10 (100%) | 77.0 (10.0) | 77.0 (10.0) | 78.03 (10.56) | `2.86e-07` | `0.00e+00` | `0.00e+00` | `1.78e-14` | `2.176801` | `1.95e-03 (FC better)` |
| `SciPy-trust-constr` | 10/10 (100%) | 84.5 (21.8) | 84.5 (21.8) | 419.99 (81.11) | `8.96e-07` | `0.00e+00` | `0.00e+00` | `4.71e-15` | `2.176802` | `1.95e-03 (FC better)` |

### `B4`: Bounded-Variance ML Common Principal Components

- **Regime**: `z in [0.05,20]^150 x St(30,30) (Branch B O(30), log barrier, 1.6e5 scale ratio)` (`d = 150`, `R = 1`, `N_tot = 1050`, intrinsic dim `585`, FD gradient max rel err `5.658e-10`)

| Solver | Solve Rate | Raw Evals Med (IQR) | Penalized Evals Med (IQR) | Wall ms Med (IQR) | Median KKT Ref | Median Box Viol | Median Lin Viol | Median Orth Viol | Median Objective | Wilcoxon Evals p |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `fortchuck` | 10/10 (100%) | 1853.0 (241.2) | 1853.0 (241.2) | 1535.91 (174.90) | `6.54e-07` | `0.00e+00` | `0.00e+00` | `6.94e-14` | `17.596323` | `—` |
| `fortchuck-ABB` | 9/10 (90%) | 4352.0 (617.0) | 4352.0 (617.0) | 2371.99 (401.11) | `6.73e-07` | `0.00e+00` | `0.00e+00` | `6.34e-14` | `17.596323` | `1.95e-03 (FC better)` |
| `Alternating-BCD` | 0/10 (0%) | 5000.0 (0.0) | 5000.0 (0.0) | 3566.46 (88.10) | `1.06e-03` | `0.00e+00` | `0.00e+00` | `3.18e-15` | `17.596323` | `1.95e-03 (FC better)` |
| `pymanopt-Warped-CG` | 0/10 (0%) | 5000.0 (0.0) | 5000.0 (0.0) | 4908.01 (49.16) | `6.23e-03` | `0.00e+00` | `0.00e+00` | `3.08e-15` | `17.598686` | `1.95e-03 (FC better)` |
| `pymanopt-TrustRegions` | 0/10 (0%) | 5000.0 (0.0) | 5000.0 (0.0) | 3149.28 (45.20) | `5.15e-03` | `0.00e+00` | `0.00e+00` | `6.65e-14` | `17.597529` | `1.95e-03 (FC better)` |
| `SciPy-L-BFGS-B` | 0/10 (0%) | 5000.0 (0.0) | 5000.0 (0.0) | 4702.69 (142.04) | `5.66e-01` | `0.00e+00` | `0.00e+00` | `7.08e-06` | `20.422174` | `1.95e-03 (FC better)` |
| `SciPy-SLSQP` | SKIP (`N_tot > 450`) | — | — | — | — | — | — | — | — | — |
| `SciPy-trust-constr` | SKIP (`N_tot > 450`) | — | — | — | — | — | — | — | — | — |

### `B5`: Bounded Factor Correlation Matrix Calibration

- **Regime**: `z in B^110 x St(100,10) (Branch A, quartic valley + clamped psi)` (`d = 110`, `R = 1`, `N_tot = 1110`, intrinsic dim `1055`, FD gradient max rel err `9.975e-10`)

| Solver | Solve Rate | Raw Evals Med (IQR) | Penalized Evals Med (IQR) | Wall ms Med (IQR) | Median KKT Ref | Median Box Viol | Median Lin Viol | Median Orth Viol | Median Objective | Wilcoxon Evals p |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `fortchuck` | 10/10 (100%) | 1187.5 (94.8) | 1187.5 (94.8) | 800.03 (71.96) | `9.19e-07` | `0.00e+00` | `0.00e+00` | `4.39e-14` | `34.668813` | `—` |
| `fortchuck-ABB` | 0/10 (0%) | 5000.0 (0.0) | 5000.0 (0.0) | 2045.96 (34.83) | `1.14e-03` | `0.00e+00` | `0.00e+00` | `3.15e-14` | `34.668859` | `1.95e-03 (FC better)` |
| `Alternating-BCD` | 0/10 (0%) | 5000.0 (0.0) | 5000.0 (0.0) | 3579.36 (448.09) | `7.03e-03` | `0.00e+00` | `0.00e+00` | `1.09e-15` | `34.669169` | `1.95e-03 (FC better)` |
| `pymanopt-Warped-CG` | 0/10 (0%) | 5000.0 (0.0) | 5000.0 (0.0) | 5127.60 (409.29) | `2.99e-02` | `0.00e+00` | `0.00e+00` | `1.11e-15` | `34.672052` | `1.95e-03 (FC better)` |
| `pymanopt-TrustRegions` | 0/10 (0%) | 5000.0 (0.0) | 5000.0 (0.0) | 3364.50 (288.75) | `2.41e-02` | `0.00e+00` | `0.00e+00` | `3.59e-12` | `34.669727` | `1.95e-03 (FC better)` |
| `SciPy-L-BFGS-B` | 0/10 (0%) | 2263.0 (861.8) | 5000.0 (0.0) | 2299.17 (792.73) | `3.77e-05` | `0.00e+00` | `0.00e+00` | `1.10e-10` | `34.668813` | `1.95e-03 (FC better)` |
| `SciPy-SLSQP` | SKIP (`N_tot > 450`) | — | — | — | — | — | — | — | — | — |
| `SciPy-trust-constr` | SKIP (`N_tot > 450`) | — | — | — | — | — | — | — | — | — |

### `B6`: Box-Coupled Brockett with Exact Ky Fan QP Truth

- **Regime**: `z in [0.1,5]^10 x St(150,10) (Branch A, kappa(M)=10^4, certified f_star)` (`d = 10`, `R = 1`, `N_tot = 1510`, intrinsic dim `1455`, FD gradient max rel err `4.334e-10`)

| Solver | Solve Rate | Raw Evals Med (IQR) | Penalized Evals Med (IQR) | Wall ms Med (IQR) | Median KKT Ref | Median Box Viol | Median Lin Viol | Median Orth Viol | Median Objective | Wilcoxon Evals p |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `fortchuck` | 10/10 (100%) | 1970.5 (273.8) | 1970.5 (273.8) | 1552.43 (227.06) | `9.32e-07` | `0.00e+00` | `0.00e+00` | `5.03e-14` | `-10.445510` | `—` |
| `fortchuck-ABB` | 8/10 (80%) | 4529.5 (496.0) | 4529.5 (496.0) | 1688.70 (187.60) | `7.06e-07` | `0.00e+00` | `0.00e+00` | `5.83e-14` | `-10.445510` | `1.95e-03 (FC better)` |
| `Alternating-BCD` | 0/10 (0%) | 5000.0 (0.0) | 5000.0 (0.0) | 3198.91 (48.52) | `1.63e-02` | `0.00e+00` | `0.00e+00` | `1.16e-15` | `-10.444608` | `1.95e-03 (FC better)` |
| `pymanopt-Warped-CG` | 0/10 (0%) | 5000.0 (0.0) | 5000.0 (0.0) | 4737.28 (39.48) | `2.09e-03` | `0.00e+00` | `0.00e+00` | `1.18e-15` | `-10.445239` | `1.95e-03 (FC better)` |
| `pymanopt-TrustRegions` | 0/10 (0%) | 5000.0 (0.0) | 5000.0 (0.0) | 3090.43 (28.37) | `1.52e-03` | `0.00e+00` | `0.00e+00` | `1.44e-15` | `-10.445450` | `1.95e-03 (FC better)` |
| `SciPy-L-BFGS-B` | 0/10 (0%) | 5000.0 (0.0) | 5000.0 (0.0) | 5041.60 (65.27) | `4.91e-03` | `0.00e+00` | `0.00e+00` | `1.50e-05` | `-10.445525` | `1.95e-03 (FC better)` |
| `SciPy-SLSQP` | SKIP (`N_tot > 450`) | — | — | — | — | — | — | — | — | — |
| `SciPy-trust-constr` | SKIP (`N_tot > 450`) | — | — | — | — | — | — | — | — | — |

### `B7A`: Pure-Box Active-Bound Chained Rosenbrock + Quadratic

- **Regime**: `z in [l,u]^500, R=0 (incl. 5 fixed l_i==u_i, ~60% active bounds)` (`d = 500`, `R = 0`, `N_tot = 500`, intrinsic dim `500`, FD gradient max rel err `7.307e-10`)

| Solver | Solve Rate | Raw Evals Med (IQR) | Penalized Evals Med (IQR) | Wall ms Med (IQR) | Median KKT Ref | Median Box Viol | Median Lin Viol | Median Orth Viol | Median Objective | Wilcoxon Evals p |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `fortchuck` | 10/10 (100%) | 68.0 (1.8) | 68.0 (1.8) | 11.38 (1.66) | `8.51e-07` | `0.00e+00` | `0.00e+00` | `0.00e+00` | `87.046277` | `—` |
| `fortchuck-ABB` | 10/10 (100%) | 88.0 (9.5) | 88.0 (9.5) | 11.36 (2.10) | `8.16e-07` | `0.00e+00` | `0.00e+00` | `0.00e+00` | `87.046277` | `1.95e-03 (FC better)` |
| `Alternating-BCD` | 10/10 (100%) | 68.5 (2.0) | 68.5 (2.0) | 44.25 (2.44) | `9.06e-07` | `0.00e+00` | `0.00e+00` | `0.00e+00` | `87.046277` | `8.59e-02 (FC better)` |
| `pymanopt-Warped-CG` | 0/10 (0%) | 5000.0 (0.0) | 5000.0 (0.0) | 1757.49 (30.92) | `1.13e-01` | `0.00e+00` | `0.00e+00` | `0.00e+00` | `87.051344` | `1.95e-03 (FC better)` |
| `pymanopt-TrustRegions` | 0/10 (0%) | 5000.0 (0.0) | 5000.0 (0.0) | 1557.92 (29.22) | `1.02e-04` | `0.00e+00` | `0.00e+00` | `0.00e+00` | `87.046347` | `1.95e-03 (FC better)` |
| `SciPy-L-BFGS-B` | 10/10 (100%) | 68.5 (2.0) | 68.5 (2.0) | 45.89 (2.46) | `9.06e-07` | `0.00e+00` | `0.00e+00` | `0.00e+00` | `87.046277` | `8.59e-02 (FC better)` |
| `SciPy-SLSQP` | SKIP (`N_tot > 450`) | — | — | — | — | — | — | — | — | — |
| `SciPy-trust-constr` | SKIP (`N_tot > 450`) | — | — | — | — | — | — | — | — | — |

### `B7B`: Pure-Stiefel Tall Brockett & Orthogonal Procrustes

- **Regime**: `d=0, St(200,8) x St(24,24) (Pure Stiefel Branch A & Branch B)` (`d = 0`, `R = 2`, `N_tot = 2176`, intrinsic dim `1840`, FD gradient max rel err `3.760e-10`)

| Solver | Solve Rate | Raw Evals Med (IQR) | Penalized Evals Med (IQR) | Wall ms Med (IQR) | Median KKT Ref | Median Box Viol | Median Lin Viol | Median Orth Viol | Median Objective | Wilcoxon Evals p |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `fortchuck` | 10/10 (100%) | 516.0 (103.2) | 516.0 (103.2) | 530.21 (90.76) | `6.84e-07` | `0.00e+00` | `0.00e+00` | `8.03e-14` | `-66.949417` | `—` |
| `fortchuck-ABB` | 10/10 (100%) | 663.0 (72.0) | 663.0 (72.0) | 338.51 (42.18) | `6.66e-07` | `0.00e+00` | `0.00e+00` | `7.20e-14` | `-66.949417` | `1.95e-03 (FC better)` |
| `Alternating-BCD` | 10/10 (100%) | 1109.0 (107.8) | 1109.0 (107.8) | 1156.71 (70.50) | `8.06e-07` | `0.00e+00` | `0.00e+00` | `2.66e-15` | `-66.949417` | `1.95e-03 (FC better)` |
| `pymanopt-Warped-CG` | 10/10 (100%) | 1109.0 (107.8) | 1109.0 (107.8) | 1158.60 (173.93) | `8.06e-07` | `0.00e+00` | `0.00e+00` | `2.66e-15` | `-66.949417` | `1.95e-03 (FC better)` |
| `pymanopt-TrustRegions` | 10/10 (100%) | 1720.5 (241.0) | 1720.5 (241.0) | 995.53 (152.42) | `6.43e-10` | `0.00e+00` | `0.00e+00` | `2.62e-15` | `-66.949417` | `1.95e-03 (FC better)` |
| `SciPy-L-BFGS-B` | 0/10 (0%) | 2145.0 (1206.8) | 5000.0 (0.0) | 1916.36 (1020.46) | `1.42e-03` | `0.00e+00` | `0.00e+00` | `5.46e-05` | `-66.950913` | `1.95e-03 (FC better)` |
| `SciPy-SLSQP` | SKIP (`N_tot > 450`) | — | — | — | — | — | — | — | — | — |
| `SciPy-trust-constr` | SKIP (`N_tot > 450`) | — | — | — | — | — | — | — | — | — |

### `B8A`: Trace-Constrained Ensemble Kohn-Sham DFT

- **Regime**: `z in [0,1]^12 (1^T z = 5) x St(120,12) (SLBQP + Branch A, hard trace)` (`d = 12`, `R = 1`, `N_tot = 1452`, intrinsic dim `1373`, linear constraint `a^T z = 5.0`, FD gradient max rel err `1.665e-10`)

| Solver | Solve Rate | Raw Evals Med (IQR) | Penalized Evals Med (IQR) | Wall ms Med (IQR) | Median KKT Ref | Median Box Viol | Median Lin Viol | Median Orth Viol | Median Objective | Wilcoxon Evals p |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `fortchuck` | 10/10 (100%) | 150.5 (38.5) | 150.5 (38.5) | 151.19 (38.68) | `7.05e-07` | `0.00e+00` | `0.00e+00` | `2.81e-14` | `-8.904018` | `—` |
| `fortchuck-ABB` | 9/10 (90%) | 170.0 (30.0) | 170.0 (30.0) | 86.29 (23.90) | `7.46e-07` | `0.00e+00` | `0.00e+00` | `2.85e-14` | `-8.904018` | `5.39e-01 (FC better)` |
| `Alternating-BCD` | 10/10 (100%) | 367.5 (73.5) | 367.5 (73.5) | 306.33 (93.09) | `8.07e-07` | `0.00e+00` | `0.00e+00` | `1.33e-15` | `-8.904018` | `3.91e-03 (FC better)` |
| `pymanopt-Warped-CG` | 0/10 (0%) | 5000.0 (0.0) | 5000.0 (0.0) | 5757.04 (62.17) | `2.17e-03` | `0.00e+00` | `4.62e-05` | `1.46e-15` | `-8.908815` | `1.95e-03 (FC better)` |
| `pymanopt-TrustRegions` | 0/10 (0%) | 5000.0 (0.0) | 5000.0 (0.0) | 4307.36 (60.69) | `1.04e-02` | `0.00e+00` | `6.46e-06` | `1.48e-15` | `-8.909153` | `1.95e-03 (FC better)` |
| `SciPy-L-BFGS-B` | 0/10 (0%) | 3252.5 (2820.2) | 5000.0 (0.0) | 4031.23 (3927.28) | `5.19e-05` | `0.00e+00` | `3.30e-08` | `7.34e-06` | `-8.904045` | `1.95e-03 (FC better)` |
| `SciPy-SLSQP` | SKIP (`N_tot > 450`) | — | — | — | — | — | — | — | — | — |
| `SciPy-trust-constr` | SKIP (`N_tot > 450`) | — | — | — | — | — | — | — | — | — |

### `B8B`: Long-Only Weighted Orthogonal Risk-Factor Parity

- **Regime**: `z in [0,0.15]^24 (a^T z = 1) x St(32,6) x St(6,6) (N_tot=252, Branch A & B)` (`d = 24`, `R = 2`, `N_tot = 252`, intrinsic dim `209`, linear constraint `a^T z = 1.0`, FD gradient max rel err `1.406e-11`)

| Solver | Solve Rate | Raw Evals Med (IQR) | Penalized Evals Med (IQR) | Wall ms Med (IQR) | Median KKT Ref | Median Box Viol | Median Lin Viol | Median Orth Viol | Median Objective | Wilcoxon Evals p |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `fortchuck` | 10/10 (100%) | 37.0 (1.8) | 37.0 (1.8) | 10.86 (2.44) | `7.47e-07` | `0.00e+00` | `2.78e-17` | `8.31e-15` | `-4.990125` | `—` |
| `fortchuck-ABB` | 10/10 (100%) | 66.5 (16.0) | 66.5 (16.0) | 14.56 (4.41) | `7.88e-07` | `0.00e+00` | `2.78e-17` | `1.04e-14` | `-4.990125` | `1.95e-03 (FC better)` |
| `Alternating-BCD` | 10/10 (100%) | 408.5 (58.5) | 408.5 (58.5) | 434.44 (33.37) | `7.70e-07` | `0.00e+00` | `1.11e-16` | `9.63e-16` | `-4.990125` | `1.95e-03 (FC better)` |
| `pymanopt-Warped-CG` | 0/10 (0%) | 5000.0 (0.0) | 5000.0 (0.0) | 6864.35 (82.17) | `7.97e-04` | `0.00e+00` | `5.18e-05` | `8.64e-16` | `-4.990098` | `1.95e-03 (FC better)` |
| `pymanopt-TrustRegions` | 0/10 (0%) | 5000.0 (0.0) | 5000.0 (0.0) | 4665.20 (26.06) | `2.38e-04` | `0.00e+00` | `2.90e-05` | `1.03e-15` | `-4.990137` | `1.95e-03 (FC better)` |
| `SciPy-L-BFGS-B` | 0/10 (0%) | 1049.0 (139.0) | 5000.0 (0.0) | 1188.04 (142.52) | `3.19e-05` | `0.00e+00` | `1.26e-07` | `2.19e-06` | `-4.990130` | `1.95e-03 (FC better)` |
| `SciPy-SLSQP` | 10/10 (100%) | 34.5 (7.2) | 34.5 (7.2) | 427.62 (22.60) | `7.52e-07` | `0.00e+00` | `7.22e-16` | `5.73e-12` | `-4.990125` | `7.64e-01 (FC worse)` |
| `SciPy-trust-constr` | 0/10 (0%) | 191.5 (11.5) | 5000.0 (0.0) | 3023.54 (203.29) | `3.36e-05` | `0.00e+00` | `5.55e-17` | `1.24e-14` | `-4.990124` | `1.95e-03 (FC better)` |

