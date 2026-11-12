# Blind Multi-Seed Benchmark Report (`fortchuck` v0.2.0)

## 1. Blind Evaluation Protocol and Referee Specification

All benchmark runs enforce a strict blind firewall between `benchmarks/*.py` and `src/*.f90`:

- **Frozen Out-of-the-Box Configuration**: `fortchuck` is evaluated across every benchmark instance (`B1` through `B8B`) and every random seed with `FCOptions(gtol=1.0e-6, max_eval=5000)` (`memory = 7`, `max_iter = 15000`, Zhang--Hager nonmonotone `zh_eta` auto `0.85`, canonical metric, cautious skip, `eps_band = 1.0e-4`), passing `linear_constraint=(a, b)` when `lin_a` is present and **zero** per-problem hyperparameter adjustments.
- **Multi-Suite Deterministic Initializations**: Initial feasible iterates are generated via `PCG64` across `10` deterministic primary seeds (`101..110`), `10` out-of-sample holdout seeds (`201..210`), `10` untouched third blind holdout seeds (`301..310`). On linearly constrained problems (`B8A`, `B8B`), initial Euclidean draws are projected onto the central 80% sub-box intersected with $a^\top z = b$ via `project_linear_box_ref` so every solver starts from the exact same strictly interior feasible point.
- **Pre-Flight Central Finite-Difference Verification**: Every coordinate of every analytical gradient is verified against central finite differences (`fd_step = 1.0e-5`) to relative error at most $10^{-7}$ prior to solver execution.
- **Independent Post-Solve KKT and Feasibility Certificate**: A run is certified **Solved** by `benchmarks/referee.py` if and only if both optimality and feasibility tolerances hold simultaneously (using the pure-NumPy safeguarded root-finder `project_linear_box_ref` with zero calls to `fortchuck` when `lin_a` is present):

```math
\mathcal E_{\mathrm{KKT,ref}}(z, X) = \max\!\left(\left\lVert z - P_{\mathcal C}\bigl(z - \nabla_z f(z, X)\bigr) \right\rVert_\infty, \; \max_{1 \le r \le R} \left\lVert G_r - X_r\,\mathrm{sym}(X_r^\top G_r) \right\rVert_{\mathrm F}\right) \le 10^{-6}
```

```math
\max\!\left(\mathcal V_{\mathrm{box}}, \, \mathcal V_{\mathrm{lin}}, \, \mathcal V_{\mathrm{orth}}\right) \le 10^{-8}, \qquad \mathcal V_{\mathrm{lin}} = \frac{\lvert a^\top z - b \rvert}{1 + \lvert b \rvert}
```

## 2. Pooled Multi-Suite Aggregate Summary (`30` Seeds Total)

Pooled aggregate metrics across all executed blind seed suites (`Penalized Evals` assigns `max_eval = 5000` to unsolved runs; paired two-sided Wilcoxon signed-rank `p`-values compare `fortchuck` against each baseline across shared non-skipped runs):

| Solver | Evaluated Runs | Solved Runs | Solve Rate | Penalized Evals Med (IQR) | Wall Time ms Med (IQR) | Median KKT Ref | Max Lin Viol | Median Orth Viol | Wilcoxon Evals p (2-sided) | Wilcoxon KKT p (2-sided) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `fortchuck` | 330 | 312 | 94.5% | 524.5 (1651.5) | 298.35 (1021.87) | `8.34e-07` | `1.90e-14` | `3.28e-14` | `—` | `—` |
| `fortchuck-ABB` | 330 | 232 | 70.3% | 1262.0 (4906.0) | 295.36 (1575.94) | `8.32e-07` | `8.59e-15` | `3.01e-14` | `3.27e-34` | `1.30e-13` |
| `Alternating-BCD` | 330 | 160 | 48.5% | 5000.0 (4650.8) | 2436.93 (3434.81) | `1.17e-05` | `5.55e-16` | `1.10e-15` | `7.77e-42` | `1.03e-40` |
| `pymanopt-Warped-CG` | 330 | 73 | 22.1% | 5000.0 (0.0) | 5406.79 (4480.17) | `1.68e-03` | `9.80e-05` | `1.13e-15` | `6.62e-53` | `2.00e-54` |
| `pymanopt-TrustRegions` | 330 | 79 | 23.9% | 5000.0 (0.0) | 3507.79 (2741.58) | `3.87e-03` | `8.32e-05` | `2.71e-15` | `6.62e-53` | `3.99e-47` |
| `SciPy-L-BFGS-B` | 330 | 30 | 9.1% | 5000.0 (0.0) | 2759.82 (3658.59) | `1.30e-04` | `1.02e-06` | `5.20e-06` | `5.62e-51` | `5.17e-55` |
| `SciPy-SLSQP` | 60 | 60 | 100.0% | 58.0 (41.8) | 261.06 (404.25) | `6.17e-07` | `1.44e-15` | `1.17e-12` | `2.80e-02` | `7.40e-01` |
| `SciPy-trust-constr` | 60 | 30 | 50.0% | 2572.0 (4911.5) | 1920.18 (3107.86) | `1.29e-06` | `5.55e-16` | `1.42e-14` | `1.63e-11` | `9.28e-11` |

## 3. Primary Blind Suite Aggregate Summary (`Seeds 101..110`)

Aggregate metrics across all evaluated `(problem, seed)` pairs in the primary blind suite:

| Solver | Evaluated Runs | Solved Runs | Solve Rate | Penalized Evals Med (IQR) | Wall Time ms Med (IQR) | Median KKT Ref | Max Lin Viol | Median Orth Viol | Wilcoxon Evals p (2-sided) | Wilcoxon KKT p (2-sided) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `fortchuck` | 110 | 106 | 96.4% | 506.5 (1628.8) | 212.04 (1029.08) | `8.57e-07` | `1.90e-14` | `3.10e-14` | `—` | `—` |
| `fortchuck-ABB` | 110 | 75 | 68.2% | 1303.0 (4904.8) | 327.30 (1574.05) | `8.71e-07` | `8.59e-15` | `2.65e-14` | `4.00e-13` | `2.47e-06` |
| `Alternating-BCD` | 110 | 54 | 49.1% | 5000.0 (4664.2) | 1763.24 (3260.69) | `8.11e-06` | `3.33e-16` | `1.06e-15` | `2.71e-15` | `6.43e-14` |
| `pymanopt-Warped-CG` | 110 | 23 | 20.9% | 5000.0 (0.0) | 5028.55 (4102.04) | `1.52e-03` | `8.50e-05` | `1.14e-15` | `3.99e-19` | `1.88e-19` |
| `pymanopt-TrustRegions` | 110 | 24 | 21.8% | 5000.0 (0.0) | 3278.26 (2239.10) | `4.17e-03` | `8.32e-05` | `3.36e-15` | `3.99e-19` | `3.22e-17` |
| `SciPy-L-BFGS-B` | 110 | 10 | 9.1% | 5000.0 (0.0) | 2790.81 (3643.59) | `9.71e-05` | `1.26e-07` | `5.56e-06` | `1.89e-18` | `2.83e-19` |
| `SciPy-SLSQP` | 20 | 20 | 100.0% | 59.0 (34.0) | 286.70 (453.63) | `6.48e-07` | `1.11e-15` | `1.40e-12` | `2.79e-01` | `5.22e-01` |
| `SciPy-trust-constr` | 20 | 10 | 50.0% | 2564.5 (4909.5) | 1845.38 (3841.97) | `1.42e-05` | `5.55e-16` | `6.13e-14` | `8.82e-05` | `1.91e-06` |

## 4. Out-of-Sample Holdout Suite Aggregate Summary (`Seeds 201..210`)

Aggregate metrics across out-of-sample holdout seeds (`201..210`) under the identical frozen `FCOptions(gtol=1.0e-6, max_eval=5000)` configuration:

| Solver | Evaluated Runs | Solved Runs | Solve Rate | Penalized Evals Med (IQR) | Wall Time ms Med (IQR) | Median KKT Ref | Max Lin Viol | Median Orth Viol | Wilcoxon Evals p (2-sided) | Wilcoxon KKT p (2-sided) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `fortchuck` | 110 | 103 | 93.6% | 527.0 (1689.5) | 356.28 (1056.49) | `8.14e-07` | `1.11e-16` | `3.45e-14` | `—` | `—` |
| `fortchuck-ABB` | 110 | 80 | 72.7% | 1324.0 (4904.0) | 297.53 (1613.01) | `7.49e-07` | `2.22e-16` | `2.95e-14` | `1.45e-11` | `8.98e-04` |
| `Alternating-BCD` | 110 | 53 | 48.2% | 5000.0 (4641.5) | 2238.97 (3807.73) | `4.10e-05` | `5.55e-16` | `1.09e-15` | `2.23e-14` | `2.38e-15` |
| `pymanopt-Warped-CG` | 110 | 25 | 22.7% | 5000.0 (0.0) | 5936.10 (4522.30) | `1.68e-03` | `9.49e-05` | `1.12e-15` | `1.25e-18` | `2.83e-19` |
| `pymanopt-TrustRegions` | 110 | 27 | 24.5% | 5000.0 (0.0) | 3762.86 (2933.98) | `3.63e-03` | `5.87e-05` | `2.45e-15` | `1.25e-18` | `1.17e-16` |
| `SciPy-L-BFGS-B` | 110 | 10 | 9.1% | 5000.0 (0.0) | 2472.08 (3547.09) | `1.52e-04` | `1.26e-07` | `6.34e-06` | `6.16e-18` | `1.43e-19` |
| `SciPy-SLSQP` | 20 | 20 | 100.0% | 54.0 (47.0) | 292.10 (379.25) | `6.75e-07` | `9.99e-16` | `1.10e-12` | `6.37e-02` | `2.02e-01` |
| `SciPy-trust-constr` | 20 | 10 | 50.0% | 2572.0 (4904.0) | 1920.18 (2714.34) | `1.42e-05` | `1.11e-16` | `6.18e-15` | `1.91e-06` | `1.91e-05` |

## 5. Untouched Third Blind Holdout Suite Aggregate Summary (`Seeds 301..310`)

Aggregate metrics across the untouched third blind holdout suite (`301..310`) under the identical frozen `FCOptions(gtol=1.0e-6, max_eval=5000)` configuration:

| Solver | Evaluated Runs | Solved Runs | Solve Rate | Penalized Evals Med (IQR) | Wall Time ms Med (IQR) | Median KKT Ref | Max Lin Viol | Median Orth Viol | Wilcoxon Evals p (2-sided) | Wilcoxon KKT p (2-sided) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `fortchuck` | 110 | 103 | 93.6% | 531.0 (1707.2) | 289.44 (970.43) | `8.00e-07` | `1.11e-16` | `3.81e-14` | `—` | `—` |
| `fortchuck-ABB` | 110 | 77 | 70.0% | 975.0 (4906.8) | 276.99 (1547.07) | `8.74e-07` | `1.11e-16` | `3.61e-14` | `9.73e-13` | `1.18e-06` |
| `Alternating-BCD` | 110 | 53 | 48.2% | 5000.0 (4648.0) | 2998.09 (3434.13) | `1.66e-05` | `3.33e-16` | `1.15e-15` | `1.98e-15` | `1.73e-14` |
| `pymanopt-Warped-CG` | 110 | 25 | 22.7% | 5000.0 (0.0) | 5365.50 (4614.83) | `1.75e-03` | `9.80e-05` | `1.13e-15` | `1.25e-18` | `4.13e-19` |
| `pymanopt-TrustRegions` | 110 | 28 | 25.5% | 5000.0 (511.5) | 3426.45 (2841.60) | `3.76e-03` | `5.74e-05` | `2.64e-15` | `1.25e-18` | `1.83e-16` |
| `SciPy-L-BFGS-B` | 110 | 10 | 9.1% | 5000.0 (0.0) | 3012.95 (3673.92) | `2.01e-04` | `1.02e-06` | `4.87e-06` | `6.16e-18` | `1.36e-19` |
| `SciPy-SLSQP` | 20 | 20 | 100.0% | 58.0 (43.5) | 258.50 (393.37) | `5.45e-07` | `1.44e-15` | `1.03e-12` | `4.78e-01` | `1.33e-01` |
| `SciPy-trust-constr` | 20 | 10 | 50.0% | 2564.0 (4915.8) | 2010.10 (2971.19) | `1.28e-06` | `2.22e-16` | `1.70e-14` | `1.91e-06` | `4.77e-05` |

## 6. Exact Global Optimality Certificate (`B6` Ky Fan QP & `B7B` Procrustes)

Problems `B6` and `B7B` possess exact analytical global minima $f^\ast$ certified by Ky Fan's extremal trace inequality and Kabsch singular value decomposition:

| Suite | Problem | Certified Global Optimum | Solver | Solve Rate | Median Abs Gap | Max Abs Gap | Median Signed Gap | Median KKT Ref |
| :--- | :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| Primary (`101..110`) | `B6` | `-10.445509834730` | `fortchuck` | 10/10 (100%) | `1.64e-12` | `4.39e-12` | `1.64e-12` | `9.47e-07` |
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
| Holdout (`201..210`) | `B6` | `-10.445509834730` | `fortchuck` | 10/10 (100%) | `2.27e-12` | `3.56e-12` | `2.27e-12` | `8.77e-07` |
| Holdout (`201..210`) | `B6` | `-10.445509834730` | `fortchuck-ABB` | 8/10 (80%) | `1.06e-10` | `6.94e-10` | `1.06e-10` | `7.11e-07` |
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
| Blind3 (`301..310`) | `B6` | `-10.445509834730` | `fortchuck` | 10/10 (100%) | `2.15e-12` | `4.81e-12` | `2.15e-12` | `9.08e-07` |
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

## 7. `v0.1.0` to `v0.2.0` Progression and Zero-FFI Native CLI Telemetry

### `v0.1.0` vs `v0.2.0` Progression (`Seeds 101..110 + 201..210` Shared Cohort & `30`-Seed Pool)

Comparison of frozen out-of-the-box `fortchuck` (`FCOptions(gtol=1.0e-6, max_eval=5000)`) across the 20 shared seeds (`101..110` + `201..210`) from `v0.1.0` alongside the full 30-seed (`+ 301..310`) `v0.2.0` pool:

| Problem | `v0.1.0` Solved (20) | `v0.1.0` Raw Evals Med | `v0.1.0` Wall ms Med | `v0.2.0` Solved (20) | `v0.2.0` Raw Evals Med (20) | `v0.2.0` Wall ms Med (20) | `v0.2.0` Wall Speedup | `v0.2.0` Solved (30 Pool) | `v0.2.0` Max Lin Viol (30) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `B1` | 20/20 (100%) | 2084.5 | 1813.88 | 20/20 (100%) | 1959.0 | 1389.50 | `1.31x` | **30/30 (100%)** | `0.00e+00` |
| `B2A` | 15/20 (75%) | 580.0 | 171.27 | 12/20 (60%) | 909.5 | 288.27 | `0.59x` | **16/30 (53%)** | `0.00e+00` |
| `B2B` | 20/20 (100%) | 111.5 | 41.50 | 20/20 (100%) | 128.5 | 39.59 | `1.05x` | **30/30 (100%)** | `0.00e+00` |
| `B3` | 20/20 (100%) | 44.0 | 11.99 | 20/20 (100%) | 44.5 | 12.19 | `0.98x` | **30/30 (100%)** | `0.00e+00` |
| `B4` | 20/20 (100%) | 1762.0 | 1424.73 | 20/20 (100%) | 1810.5 | 1213.28 | `1.17x` | **30/30 (100%)** | `0.00e+00` |
| `B5` | 17/20 (85%) | 1233.0 | 918.65 | 17/20 (85%) | 1490.5 | 869.20 | `1.06x` | **26/30 (87%)** | `0.00e+00` |
| `B6` | 20/20 (100%) | 1902.0 | 1801.48 | 20/20 (100%) | 1970.0 | 1361.89 | `1.32x` | **30/30 (100%)** | `0.00e+00` |
| `B7A` | 20/20 (100%) | 67.5 | 8.02 | 20/20 (100%) | 126.0 | 18.33 | `0.44x` | **30/30 (100%)** | `0.00e+00` |
| `B7B` | 20/20 (100%) | 504.0 | 619.29 | 20/20 (100%) | 496.0 | 438.36 | `1.41x` | **30/30 (100%)** | `0.00e+00` |
| `B8A` | — (`v0.2.0` new) | — | — | 20/20 (100%) | 153.0 | 125.00 | — | **30/30 (100%)** | `0.00e+00` |
| `B8B` | — (`v0.2.0` new) | — | — | 20/20 (100%) | 277.0 | 84.10 | — | **30/30 (100%)** | `1.90e-14` |

### Zero-FFI Native Fortran 2018 CLI Telemetry (`app/fortchuck_bench.f90`)

Pure Fortran 2018 release timings (`gfortran -cpp -O3 -march=native -funroll-loops -fopenmp`, `0` Python `ctypes` callback overhead, `Seeds 101..110`) recorded by `fpm run fortchuck-bench` across Windows (MinGW-w64 UCRT) and WSL2 Ubuntu 26.04 (Pure F2018 & OpenBLAS `-DFORTCHUCK_USE_BLAS`):

| Native Workload (`fortchuck-bench`) | Windows Pure F2018 | WSL2 Pure F2018 | WSL2 OpenBLAS (`-DFORTCHUCK_USE_BLAS`) | Exact Accuracy / Feasibility Certificate |
| :--- | :---: | :---: | :---: | :--- |
| `[1]` Batched Panel Transport (`St(200,8)`, `K_batch=16`, `4000` reps) | `872.00 ms` vs `857.31 ms` (`0.98x`) | `1064.39 ms` vs `1113.94 ms` (`1.05x`) | **`358.99 ms` vs `454.35 ms` (`1.27x`)** | `max_err = 0.00e+00` vs `16` single-vector calls |
| `[1]` Batched Panel Transport (`St(24,24)`, `K_batch=16`, `4000` reps) | `280.74 ms` vs `281.08 ms` (`1.00x`) | `324.28 ms` vs `327.19 ms` (`1.01x`) | `297.23 ms` vs `282.66 ms` (`0.95x`) | `max_err = 0.00e+00` vs `16` single-vector calls |
| `[2]` `B6` Box-Coupled Ky Fan QP (`10/10` solved) | `973.56 ms` (`1976.0` evals) | `1215.56 ms` (`1982.0` iters) | **`659.38 ms`** | `max |f - f*| <= 5.98e-12`, `max_KKT <= 9.96e-07` |
| `[3]` `B7B` Tall Brockett + Procrustes (`10/10` solved) | `374.51 ms` (`m=7`) / `212.91 ms` (`m=0`) | `1395.26 ms` (`m=7`) / `728.05 ms` (`m=0`) | **`249.95 ms` (`m=7`) / `150.92 ms` (`m=0`)** | `max_KKT <= 9.99e-07`, `max_orth <= 9.67e-14` |
| `[4]` `B8A` Hard-Trace Ensemble DFT (`10/10` solved) | `119.23 ms` (`214.0` evals) | `146.99 ms` (`213.0` iters) | **`68.19 ms` (`208.0` iters, `214.0` evals)** | `max |1^T z - 5| <= 1.52e-13` (`1.78e-15` WSL2), `max_KKT <= 1.00e-06` |

## 8. Per-Problem Breakdown — Pooled Multi-Suite (`30` Seeds)

### `B1`: Ensemble Kohn-Sham DFT Fractional Occupations

- **Regime**: `z in [0,1]^10 x St(120,10) (Branch A, metallic orbital-occupation coupling)` (`d = 10`, `R = 1`, `N_tot = 1210`, intrinsic dim `1155`, FD gradient max rel err `2.104e-10`)

| Solver | Solve Rate | Raw Evals Med (IQR) | Penalized Evals Med (IQR) | Wall ms Med (IQR) | Median KKT Ref | Median Box Viol | Median Lin Viol | Median Orth Viol | Median Objective | Wilcoxon Evals p |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `fortchuck` | 30/30 (100%) | 2013.0 (809.8) | 2013.0 (809.8) | 1384.87 (564.70) | `8.95e-07` | `0.00e+00` | `0.00e+00` | `4.76e-14` | `-4.377192` | `—` |
| `fortchuck-ABB` | 1/30 (3%) | 5000.0 (0.0) | 5000.0 (0.0) | 1673.76 (241.06) | `2.38e-05` | `0.00e+00` | `0.00e+00` | `5.20e-14` | `-4.368467` | `1.86e-09` |
| `Alternating-BCD` | 7/30 (23%) | 5000.0 (0.0) | 5000.0 (0.0) | 3341.34 (641.80) | `8.20e-04` | `0.00e+00` | `0.00e+00` | `1.26e-15` | `-4.349258` | `7.91e-05` |
| `pymanopt-Warped-CG` | 0/30 (0%) | 5000.0 (0.0) | 5000.0 (0.0) | 5406.79 (1385.30) | `6.15e-04` | `0.00e+00` | `0.00e+00` | `1.27e-15` | `-4.386120` | `1.86e-09` |
| `pymanopt-TrustRegions` | 0/30 (0%) | 5000.0 (0.0) | 5000.0 (0.0) | 3417.18 (622.95) | `7.66e-03` | `0.00e+00` | `0.00e+00` | `1.49e-13` | `-4.386092` | `1.86e-09` |
| `SciPy-L-BFGS-B` | 0/30 (0%) | 5000.0 (0.0) | 5000.0 (0.0) | 4689.71 (754.45) | `5.93e-04` | `0.00e+00` | `0.00e+00` | `3.19e-06` | `-4.368183` | `1.86e-09` |
| `SciPy-SLSQP` | SKIP (`N_tot > 450`) | — | — | — | — | — | — | — | — | — |
| `SciPy-trust-constr` | SKIP (`N_tot > 450`) | — | — | — | — | — | — | — | — | — |

### `B2A`: Amplitude-Limited Blind Source Separation (ICA)

- **Regime**: `Z in [-2,2]^{5x150} (d=750) x St(20,5) (Branch A, nonconvex log-cosh)` (`d = 750`, `R = 1`, `N_tot = 850`, intrinsic dim `835`, FD gradient max rel err `2.877e-09`)

| Solver | Solve Rate | Raw Evals Med (IQR) | Penalized Evals Med (IQR) | Wall ms Med (IQR) | Median KKT Ref | Median Box Viol | Median Lin Viol | Median Orth Viol | Median Objective | Wilcoxon Evals p |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `fortchuck` | 16/30 (53%) | 938.5 (564.8) | 1371.0 (4160.5) | 289.44 (162.31) | `9.68e-07` | `0.00e+00` | `0.00e+00` | `5.75e-14` | `-309.730662` | `—` |
| `fortchuck-ABB` | 3/30 (10%) | 3566.0 (2250.5) | 5000.0 (0.0) | 820.93 (517.86) | `3.90e-06` | `0.00e+00` | `0.00e+00` | `5.32e-14` | `-307.938152` | `4.38e-04` |
| `Alternating-BCD` | 6/30 (20%) | 5000.0 (0.0) | 5000.0 (0.0) | 4097.38 (1003.40) | `5.44e-03` | `0.00e+00` | `0.00e+00` | `6.98e-16` | `-307.018354` | `3.90e-04` |
| `pymanopt-Warped-CG` | 0/30 (0%) | 5000.0 (0.0) | 5000.0 (0.0) | 5803.31 (1273.58) | `5.12e-01` | `0.00e+00` | `0.00e+00` | `6.73e-16` | `-306.955213` | `4.38e-04` |
| `pymanopt-TrustRegions` | 0/30 (0%) | 5000.0 (0.0) | 5000.0 (0.0) | 3518.02 (694.95) | `5.80e-01` | `0.00e+00` | `0.00e+00` | `3.11e-11` | `-306.387353` | `4.38e-04` |
| `SciPy-L-BFGS-B` | 0/30 (0%) | 5000.0 (0.0) | 5000.0 (0.0) | 4005.09 (636.62) | `5.05e-02` | `0.00e+00` | `0.00e+00` | `2.44e-05` | `-308.240922` | `4.38e-04` |
| `SciPy-SLSQP` | SKIP (`N_tot > 450`) | — | — | — | — | — | — | — | — | — |
| `SciPy-trust-constr` | SKIP (`N_tot > 450`) | — | — | — | — | — | — | — | — | — |

### `B2B`: Smooth Penalty-Split Non-Negative Sparse PCA

- **Regime**: `Z in [0,5]^{120x6} (d=720) x St(120,6) (Branch A, ~80% clamped Z=0, FM-2)` (`d = 720`, `R = 1`, `N_tot = 1440`, intrinsic dim `1419`, FD gradient max rel err `1.484e-08`)

| Solver | Solve Rate | Raw Evals Med (IQR) | Penalized Evals Med (IQR) | Wall ms Med (IQR) | Median KKT Ref | Median Box Viol | Median Lin Viol | Median Orth Viol | Median Objective | Wilcoxon Evals p |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `fortchuck` | 30/30 (100%) | 129.5 (57.5) | 129.5 (57.5) | 39.63 (28.00) | `7.12e-07` | `0.00e+00` | `0.00e+00` | `1.43e-14` | `8.542968` | `—` |
| `fortchuck-ABB` | 30/30 (100%) | 1203.5 (984.5) | 1203.5 (984.5) | 211.38 (203.11) | `9.28e-07` | `0.00e+00` | `0.00e+00` | `3.05e-14` | `8.312845` | `2.13e-06` |
| `Alternating-BCD` | 0/30 (0%) | 3926.0 (2072.0) | 5000.0 (0.0) | 3486.55 (1588.81) | `1.09e-04` | `0.00e+00` | `0.00e+00` | `1.13e-15` | `8.188735` | `1.73e-06` |
| `pymanopt-Warped-CG` | 14/30 (47%) | 3688.5 (3817.5) | 5000.0 (2925.2) | 3978.93 (3756.41) | `1.91e-04` | `0.00e+00` | `0.00e+00` | `1.09e-15` | `13.816678` | `1.86e-09` |
| `pymanopt-TrustRegions` | 19/30 (63%) | 3568.5 (2620.5) | 3568.5 (2620.5) | 2426.46 (1389.98) | `8.97e-07` | `0.00e+00` | `0.00e+00` | `1.22e-15` | `8.324789` | `1.86e-09` |
| `SciPy-L-BFGS-B` | 0/30 (0%) | 406.5 (518.2) | 5000.0 (0.0) | 368.90 (403.83) | `2.75e-05` | `0.00e+00` | `0.00e+00` | `5.47e-06` | `10.442436` | `1.73e-06` |
| `SciPy-SLSQP` | SKIP (`N_tot > 450`) | — | — | — | — | — | — | — | — | — |
| `SciPy-trust-constr` | SKIP (`N_tot > 450`) | — | — | — | — | — | — | — | — | — |

### `B3`: Multi-Camera Anisotropic Bounded-Scale Rig Wahba

- **Regime**: `z in B^15 x St(3,3)^4 (Branch B SO(3)^4, R=4, kappa(W)=10^3)` (`d = 15`, `R = 4`, `N_tot = 51`, intrinsic dim `27`, FD gradient max rel err `7.213e-11`)

| Solver | Solve Rate | Raw Evals Med (IQR) | Penalized Evals Med (IQR) | Wall ms Med (IQR) | Median KKT Ref | Median Box Viol | Median Lin Viol | Median Orth Viol | Median Objective | Wilcoxon Evals p |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `fortchuck` | 30/30 (100%) | 42.5 (10.8) | 42.5 (10.8) | 11.94 (4.39) | `5.36e-07` | `0.00e+00` | `0.00e+00` | `4.92e-15` | `2.176801` | `—` |
| `fortchuck-ABB` | 30/30 (100%) | 84.0 (11.8) | 84.0 (11.8) | 22.12 (5.44) | `5.78e-07` | `0.00e+00` | `0.00e+00` | `7.02e-15` | `2.176801` | `1.72e-06` |
| `Alternating-BCD` | 28/30 (93%) | 315.5 (20.2) | 316.0 (22.2) | 497.06 (107.02) | `8.48e-07` | `0.00e+00` | `0.00e+00` | `6.57e-16` | `2.176801` | `1.73e-06` |
| `pymanopt-Warped-CG` | 30/30 (100%) | 842.5 (291.0) | 842.5 (291.0) | 1684.00 (823.92) | `8.18e-07` | `0.00e+00` | `0.00e+00` | `7.29e-16` | `2.176801` | `1.73e-06` |
| `pymanopt-TrustRegions` | 30/30 (100%) | 651.0 (82.2) | 651.0 (82.2) | 785.08 (152.10) | `5.69e-07` | `0.00e+00` | `0.00e+00` | `6.87e-16` | `2.176801` | `1.73e-06` |
| `SciPy-L-BFGS-B` | 0/30 (0%) | 810.0 (74.5) | 5000.0 (0.0) | 803.43 (161.05) | `2.27e-05` | `0.00e+00` | `0.00e+00` | `4.83e-06` | `2.176772` | `1.71e-06` |
| `SciPy-SLSQP` | 30/30 (100%) | 80.0 (16.0) | 80.0 (16.0) | 83.63 (22.23) | `3.75e-07` | `0.00e+00` | `0.00e+00` | `3.36e-14` | `2.176801` | `1.69e-06` |
| `SciPy-trust-constr` | 30/30 (100%) | 88.0 (20.8) | 88.0 (20.8) | 496.43 (247.28) | `8.05e-07` | `0.00e+00` | `0.00e+00` | `3.78e-15` | `2.176802` | `1.73e-06` |

### `B4`: Bounded-Variance ML Common Principal Components

- **Regime**: `z in [0.05,20]^150 x St(30,30) (Branch B O(30), log barrier, 1.6e5 scale ratio)` (`d = 150`, `R = 1`, `N_tot = 1050`, intrinsic dim `585`, FD gradient max rel err `6.833e-10`)

| Solver | Solve Rate | Raw Evals Med (IQR) | Penalized Evals Med (IQR) | Wall ms Med (IQR) | Median KKT Ref | Median Box Viol | Median Lin Viol | Median Orth Viol | Median Objective | Wilcoxon Evals p |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `fortchuck` | 30/30 (100%) | 1846.5 (153.2) | 1846.5 (153.2) | 1255.62 (350.45) | `6.52e-07` | `0.00e+00` | `0.00e+00` | `6.45e-14` | `17.596323` | `—` |
| `fortchuck-ABB` | 26/30 (87%) | 4403.0 (723.2) | 4403.0 (723.2) | 2223.67 (586.43) | `6.99e-07` | `0.00e+00` | `0.00e+00` | `5.23e-14` | `17.596323` | `1.86e-09` |
| `Alternating-BCD` | 0/30 (0%) | 5000.0 (0.0) | 5000.0 (0.0) | 4488.53 (709.47) | `4.53e-04` | `0.00e+00` | `0.00e+00` | `3.13e-15` | `17.596323` | `1.73e-06` |
| `pymanopt-Warped-CG` | 0/30 (0%) | 5000.0 (0.0) | 5000.0 (0.0) | 6480.33 (1082.82) | `8.05e-03` | `0.00e+00` | `0.00e+00` | `3.09e-15` | `17.599196` | `1.73e-06` |
| `pymanopt-TrustRegions` | 0/30 (0%) | 5000.0 (0.0) | 5000.0 (0.0) | 4238.19 (841.86) | `4.70e-03` | `0.00e+00` | `0.00e+00` | `9.41e-14` | `17.597364` | `1.73e-06` |
| `SciPy-L-BFGS-B` | 0/30 (0%) | 5000.0 (0.0) | 5000.0 (0.0) | 5285.11 (473.05) | `5.66e-01` | `0.00e+00` | `0.00e+00` | `7.25e-06` | `20.878153` | `1.73e-06` |
| `SciPy-SLSQP` | SKIP (`N_tot > 450`) | — | — | — | — | — | — | — | — | — |
| `SciPy-trust-constr` | SKIP (`N_tot > 450`) | — | — | — | — | — | — | — | — | — |

### `B5`: Bounded Factor Correlation Matrix Calibration

- **Regime**: `z in B^110 x St(100,10) (Branch A, quartic valley + clamped psi)` (`d = 110`, `R = 1`, `N_tot = 1110`, intrinsic dim `1055`, FD gradient max rel err `8.432e-10`)

| Solver | Solve Rate | Raw Evals Med (IQR) | Penalized Evals Med (IQR) | Wall ms Med (IQR) | Median KKT Ref | Median Box Viol | Median Lin Viol | Median Orth Viol | Median Objective | Wilcoxon Evals p |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `fortchuck` | 26/30 (87%) | 1466.5 (282.0) | 1534.0 (316.5) | 869.20 (191.27) | `9.48e-07` | `0.00e+00` | `0.00e+00` | `4.64e-14` | `34.668813` | `—` |
| `fortchuck-ABB` | 0/30 (0%) | 5000.0 (0.0) | 5000.0 (0.0) | 1752.60 (98.42) | `1.17e-03` | `0.00e+00` | `0.00e+00` | `3.89e-14` | `34.668887` | `8.30e-06` |
| `Alternating-BCD` | 0/30 (0%) | 5000.0 (0.0) | 5000.0 (0.0) | 4006.50 (292.62) | `4.93e-03` | `0.00e+00` | `0.00e+00` | `1.11e-15` | `34.669121` | `8.30e-06` |
| `pymanopt-Warped-CG` | 0/30 (0%) | 5000.0 (0.0) | 5000.0 (0.0) | 5901.62 (648.93) | `2.30e-02` | `0.00e+00` | `0.00e+00` | `1.32e-15` | `34.671816` | `8.30e-06` |
| `pymanopt-TrustRegions` | 0/30 (0%) | 5000.0 (0.0) | 5000.0 (0.0) | 3881.70 (371.27) | `2.40e-02` | `0.00e+00` | `0.00e+00` | `3.87e-12` | `34.669691` | `8.30e-06` |
| `SciPy-L-BFGS-B` | 0/30 (0%) | 2561.0 (815.5) | 5000.0 (0.0) | 2453.37 (778.10) | `3.83e-05` | `0.00e+00` | `0.00e+00` | `1.65e-10` | `34.668813` | `8.30e-06` |
| `SciPy-SLSQP` | SKIP (`N_tot > 450`) | — | — | — | — | — | — | — | — | — |
| `SciPy-trust-constr` | SKIP (`N_tot > 450`) | — | — | — | — | — | — | — | — | — |

### `B6`: Box-Coupled Brockett with Exact Ky Fan QP Truth

- **Regime**: `z in [0.1,5]^10 x St(150,10) (Branch A, kappa(M)=10^4, certified f_star)` (`d = 10`, `R = 1`, `N_tot = 1510`, intrinsic dim `1455`, FD gradient max rel err `3.584e-10`)

| Solver | Solve Rate | Raw Evals Med (IQR) | Penalized Evals Med (IQR) | Wall ms Med (IQR) | Median KKT Ref | Median Box Viol | Median Lin Viol | Median Orth Viol | Median Objective | Wilcoxon Evals p |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `fortchuck` | 30/30 (100%) | 1920.5 (355.5) | 1920.5 (355.5) | 1346.82 (337.07) | `9.22e-07` | `0.00e+00` | `0.00e+00` | `6.59e-14` | `-10.445510` | `—` |
| `fortchuck-ABB` | 23/30 (77%) | 4569.0 (796.0) | 4569.0 (796.0) | 1433.09 (182.43) | `7.27e-07` | `0.00e+00` | `0.00e+00` | `4.20e-14` | `-10.445510` | `1.73e-06` |
| `Alternating-BCD` | 0/30 (0%) | 5000.0 (0.0) | 5000.0 (0.0) | 3251.95 (338.92) | `1.38e-02` | `0.00e+00` | `0.00e+00` | `1.01e-15` | `-10.443611` | `1.86e-09` |
| `pymanopt-Warped-CG` | 0/30 (0%) | 5000.0 (0.0) | 5000.0 (0.0) | 5087.28 (769.81) | `3.27e-03` | `0.00e+00` | `0.00e+00` | `1.06e-15` | `-10.445247` | `1.86e-09` |
| `pymanopt-TrustRegions` | 0/30 (0%) | 5000.0 (0.0) | 5000.0 (0.0) | 3406.45 (797.53) | `1.72e-03` | `0.00e+00` | `0.00e+00` | `4.84e-15` | `-10.445435` | `1.86e-09` |
| `SciPy-L-BFGS-B` | 0/30 (0%) | 5000.0 (0.0) | 5000.0 (0.0) | 4632.71 (451.03) | `3.03e-03` | `0.00e+00` | `0.00e+00` | `1.50e-05` | `-10.445584` | `1.86e-09` |
| `SciPy-SLSQP` | SKIP (`N_tot > 450`) | — | — | — | — | — | — | — | — | — |
| `SciPy-trust-constr` | SKIP (`N_tot > 450`) | — | — | — | — | — | — | — | — | — |

### `B7A`: Pure-Box Active-Bound Chained Rosenbrock + Quadratic

- **Regime**: `z in [l,u]^500, R=0 (incl. 5 fixed l_i==u_i, ~60% active bounds)` (`d = 500`, `R = 0`, `N_tot = 500`, intrinsic dim `500`, FD gradient max rel err `9.086e-10`)

| Solver | Solve Rate | Raw Evals Med (IQR) | Penalized Evals Med (IQR) | Wall ms Med (IQR) | Median KKT Ref | Median Box Viol | Median Lin Viol | Median Orth Viol | Median Objective | Wilcoxon Evals p |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `fortchuck` | 30/30 (100%) | 126.0 (8.8) | 126.0 (8.8) | 17.48 (3.29) | `8.45e-07` | `0.00e+00` | `0.00e+00` | `0.00e+00` | `87.046277` | `—` |
| `fortchuck-ABB` | 30/30 (100%) | 88.5 (7.0) | 88.5 (7.0) | 10.58 (2.12) | `7.64e-07` | `0.00e+00` | `0.00e+00` | `0.00e+00` | `87.046277` | `1.72e-06` |
| `Alternating-BCD` | 30/30 (100%) | 68.0 (2.0) | 68.0 (2.0) | 38.62 (9.58) | `8.71e-07` | `0.00e+00` | `0.00e+00` | `0.00e+00` | `87.046277` | `1.70e-06` |
| `pymanopt-Warped-CG` | 0/30 (0%) | 5000.0 (0.0) | 5000.0 (0.0) | 1829.84 (340.01) | `1.91e-02` | `0.00e+00` | `0.00e+00` | `0.00e+00` | `87.048579` | `1.69e-06` |
| `pymanopt-TrustRegions` | 0/30 (0%) | 5000.0 (0.0) | 5000.0 (0.0) | 1608.25 (352.39) | `1.88e-04` | `0.00e+00` | `0.00e+00` | `0.00e+00` | `87.046453` | `1.69e-06` |
| `SciPy-L-BFGS-B` | 30/30 (100%) | 68.0 (2.0) | 68.0 (2.0) | 39.22 (11.62) | `8.71e-07` | `0.00e+00` | `0.00e+00` | `0.00e+00` | `87.046277` | `1.70e-06` |
| `SciPy-SLSQP` | SKIP (`N_tot > 450`) | — | — | — | — | — | — | — | — | — |
| `SciPy-trust-constr` | SKIP (`N_tot > 450`) | — | — | — | — | — | — | — | — | — |

### `B7B`: Pure-Stiefel Tall Brockett & Orthogonal Procrustes

- **Regime**: `d=0, St(200,8) x St(24,24) (Pure Stiefel Branch A & Branch B)` (`d = 0`, `R = 2`, `N_tot = 2176`, intrinsic dim `1840`, FD gradient max rel err `5.084e-10`)

| Solver | Solve Rate | Raw Evals Med (IQR) | Penalized Evals Med (IQR) | Wall ms Med (IQR) | Median KKT Ref | Median Box Viol | Median Lin Viol | Median Orth Viol | Median Objective | Wilcoxon Evals p |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `fortchuck` | 30/30 (100%) | 496.0 (61.2) | 496.0 (61.2) | 430.78 (72.33) | `6.67e-07` | `0.00e+00` | `0.00e+00` | `6.30e-14` | `-66.949417` | `—` |
| `fortchuck-ABB` | 30/30 (100%) | 659.0 (105.8) | 659.0 (105.8) | 273.36 (51.05) | `6.92e-07` | `0.00e+00` | `0.00e+00` | `6.90e-14` | `-66.949417` | `1.73e-06` |
| `Alternating-BCD` | 29/30 (97%) | 1069.5 (92.5) | 1080.5 (100.5) | 1377.46 (258.13) | `8.97e-07` | `0.00e+00` | `0.00e+00` | `2.57e-15` | `-66.949417` | `1.73e-06` |
| `pymanopt-Warped-CG` | 29/30 (97%) | 1069.5 (92.5) | 1080.5 (100.5) | 1405.68 (258.88) | `8.97e-07` | `0.00e+00` | `0.00e+00` | `2.57e-15` | `-66.949417` | `1.73e-06` |
| `pymanopt-TrustRegions` | 30/30 (100%) | 1683.0 (270.0) | 1683.0 (270.0) | 1218.61 (300.31) | `1.09e-08` | `0.00e+00` | `0.00e+00` | `2.69e-15` | `-66.949417` | `1.86e-09` |
| `SciPy-L-BFGS-B` | 0/30 (0%) | 2401.5 (934.0) | 5000.0 (0.0) | 2423.13 (792.62) | `1.42e-03` | `0.00e+00` | `0.00e+00` | `5.46e-05` | `-66.884604` | `1.73e-06` |
| `SciPy-SLSQP` | SKIP (`N_tot > 450`) | — | — | — | — | — | — | — | — | — |
| `SciPy-trust-constr` | SKIP (`N_tot > 450`) | — | — | — | — | — | — | — | — | — |

### `B8A`: Trace-Constrained Ensemble Kohn-Sham DFT

- **Regime**: `z in [0,1]^12 (1^T z = 5) x St(120,12) (SLBQP + Branch A, hard trace)` (`d = 12`, `R = 1`, `N_tot = 1452`, intrinsic dim `1373`, linear constraint `a^T z = 5.0`, FD gradient max rel err `1.745e-10`)

| Solver | Solve Rate | Raw Evals Med (IQR) | Penalized Evals Med (IQR) | Wall ms Med (IQR) | Median KKT Ref | Median Box Viol | Median Lin Viol | Median Orth Viol | Median Objective | Wilcoxon Evals p |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `fortchuck` | 30/30 (100%) | 151.0 (26.0) | 151.0 (26.0) | 122.10 (31.59) | `8.56e-07` | `0.00e+00` | `0.00e+00` | `2.54e-14` | `-8.904018` | `—` |
| `fortchuck-ABB` | 29/30 (97%) | 210.0 (70.2) | 210.0 (70.2) | 89.91 (39.92) | `7.45e-07` | `0.00e+00` | `0.00e+00` | `2.65e-14` | `-8.904018` | `1.56e-05` |
| `Alternating-BCD` | 30/30 (100%) | 395.5 (154.0) | 395.5 (154.0) | 419.30 (125.77) | `7.96e-07` | `0.00e+00` | `0.00e+00` | `1.28e-15` | `-8.904018` | `1.86e-09` |
| `pymanopt-Warped-CG` | 0/30 (0%) | 5000.0 (0.0) | 5000.0 (0.0) | 7153.92 (1103.26) | `4.16e-03` | `0.00e+00` | `5.82e-05` | `1.46e-15` | `-8.908960` | `1.73e-06` |
| `pymanopt-TrustRegions` | 0/30 (0%) | 5000.0 (0.0) | 5000.0 (0.0) | 5256.52 (694.28) | `7.88e-03` | `0.00e+00` | `4.12e-06` | `1.55e-15` | `-8.909331` | `1.73e-06` |
| `SciPy-L-BFGS-B` | 0/30 (0%) | 2950.0 (2843.2) | 5000.0 (0.0) | 4172.78 (4136.99) | `5.60e-05` | `0.00e+00` | `6.58e-08` | `7.34e-06` | `-8.904045` | `1.73e-06` |
| `SciPy-SLSQP` | SKIP (`N_tot > 450`) | — | — | — | — | — | — | — | — | — |
| `SciPy-trust-constr` | SKIP (`N_tot > 450`) | — | — | — | — | — | — | — | — | — |

### `B8B`: Long-Only Weighted Orthogonal Risk-Factor Parity

- **Regime**: `z in [0,0.15]^24 (a^T z = 1) x St(32,6) x St(6,6) (N_tot=252, Branch A & B)` (`d = 24`, `R = 2`, `N_tot = 252`, intrinsic dim `209`, linear constraint `a^T z = 1.0`, FD gradient max rel err `1.682e-11`)

| Solver | Solve Rate | Raw Evals Med (IQR) | Penalized Evals Med (IQR) | Wall ms Med (IQR) | Median KKT Ref | Median Box Viol | Median Lin Viol | Median Orth Viol | Median Objective | Wilcoxon Evals p |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `fortchuck` | 30/30 (100%) | 250.0 (310.2) | 250.0 (310.2) | 82.22 (72.91) | `6.00e-07` | `0.00e+00` | `5.55e-17` | `2.84e-14` | `-4.990125` | `—` |
| `fortchuck-ABB` | 30/30 (100%) | 61.5 (9.5) | 61.5 (9.5) | 11.76 (2.45) | `7.65e-07` | `0.00e+00` | `5.55e-17` | `9.27e-15` | `-4.990125` | `1.64e-07` |
| `Alternating-BCD` | 30/30 (100%) | 358.0 (95.2) | 358.0 (95.2) | 469.59 (80.55) | `8.15e-07` | `0.00e+00` | `1.11e-16` | `1.03e-15` | `-4.990125` | `2.99e-01` |
| `pymanopt-Warped-CG` | 0/30 (0%) | 5000.0 (0.0) | 5000.0 (0.0) | 9108.27 (2032.60) | `9.22e-04` | `0.00e+00` | `5.17e-05` | `1.01e-15` | `-4.990095` | `1.73e-06` |
| `pymanopt-TrustRegions` | 0/30 (0%) | 5000.0 (0.0) | 5000.0 (0.0) | 6160.68 (918.46) | `2.10e-04` | `0.00e+00` | `2.42e-05` | `9.61e-16` | `-4.990135` | `1.73e-06` |
| `SciPy-L-BFGS-B` | 0/30 (0%) | 1020.5 (107.5) | 5000.0 (0.0) | 1262.62 (184.79) | `2.79e-05` | `0.00e+00` | `1.26e-07` | `2.19e-06` | `-4.990130` | `1.73e-06` |
| `SciPy-SLSQP` | 30/30 (100%) | 37.5 (6.0) | 37.5 (6.0) | 488.74 (62.88) | `6.99e-07` | `0.00e+00` | `6.66e-16` | `4.82e-12` | `-4.990125` | `2.12e-06` |
| `SciPy-trust-constr` | 0/30 (0%) | 199.0 (28.2) | 5000.0 (0.0) | 3611.10 (700.84) | `2.87e-05` | `0.00e+00` | `2.78e-17` | `8.19e-14` | `-4.990124` | `1.73e-06` |

## 9. Per-Problem Breakdown — Primary Suite (`Seeds 101..110`)

### `B1`: Ensemble Kohn-Sham DFT Fractional Occupations

- **Regime**: `z in [0,1]^10 x St(120,10) (Branch A, metallic orbital-occupation coupling)` (`d = 10`, `R = 1`, `N_tot = 1210`, intrinsic dim `1155`, FD gradient max rel err `2.104e-10`)

| Solver | Solve Rate | Raw Evals Med (IQR) | Penalized Evals Med (IQR) | Wall ms Med (IQR) | Median KKT Ref | Median Box Viol | Median Lin Viol | Median Orth Viol | Median Objective | Wilcoxon Evals p |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `fortchuck` | 10/10 (100%) | 1883.0 (1129.8) | 1883.0 (1129.8) | 1153.93 (663.97) | `8.99e-07` | `0.00e+00` | `0.00e+00` | `3.67e-14` | `-4.372830` | `—` |
| `fortchuck-ABB` | 0/10 (0%) | 5000.0 (0.0) | 5000.0 (0.0) | 1613.83 (233.47) | `1.94e-05` | `0.00e+00` | `0.00e+00` | `7.42e-14` | `-4.368460` | `1.95e-03` |
| `Alternating-BCD` | 4/10 (40%) | 5000.0 (4646.0) | 5000.0 (4646.0) | 3100.14 (2962.18) | `3.74e-04` | `0.00e+00` | `0.00e+00` | `1.21e-15` | `-4.349253` | `1.93e-01` |
| `pymanopt-Warped-CG` | 0/10 (0%) | 5000.0 (0.0) | 5000.0 (0.0) | 4728.90 (425.49) | `5.64e-04` | `0.00e+00` | `0.00e+00` | `1.29e-15` | `-4.386121` | `1.95e-03` |
| `pymanopt-TrustRegions` | 0/10 (0%) | 5000.0 (0.0) | 5000.0 (0.0) | 3212.65 (229.87) | `7.64e-03` | `0.00e+00` | `0.00e+00` | `1.54e-13` | `-4.386094` | `1.95e-03` |
| `SciPy-L-BFGS-B` | 0/10 (0%) | 5000.0 (0.0) | 5000.0 (0.0) | 4427.55 (192.08) | `7.97e-04` | `0.00e+00` | `0.00e+00` | `3.20e-06` | `-4.368138` | `1.95e-03` |
| `SciPy-SLSQP` | SKIP (`N_tot > 450`) | — | — | — | — | — | — | — | — | — |
| `SciPy-trust-constr` | SKIP (`N_tot > 450`) | — | — | — | — | — | — | — | — | — |

### `B2A`: Amplitude-Limited Blind Source Separation (ICA)

- **Regime**: `Z in [-2,2]^{5x150} (d=750) x St(20,5) (Branch A, nonconvex log-cosh)` (`d = 750`, `R = 1`, `N_tot = 850`, intrinsic dim `835`, FD gradient max rel err `2.877e-09`)

| Solver | Solve Rate | Raw Evals Med (IQR) | Penalized Evals Med (IQR) | Wall ms Med (IQR) | Median KKT Ref | Median Box Viol | Median Lin Viol | Median Orth Viol | Median Objective | Wilcoxon Evals p |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `fortchuck` | 7/10 (70%) | 849.0 (501.5) | 849.0 (3344.8) | 212.04 (99.64) | `9.52e-07` | `0.00e+00` | `0.00e+00` | `5.31e-14` | `-306.710037` | `—` |
| `fortchuck-ABB` | 1/10 (10%) | 3094.5 (1321.0) | 5000.0 (0.0) | 538.44 (305.56) | `4.19e-06` | `0.00e+00` | `0.00e+00` | `2.49e-14` | `-307.238333` | `1.56e-02` |
| `Alternating-BCD` | 1/10 (10%) | 5000.0 (0.0) | 5000.0 (0.0) | 3774.72 (340.76) | `1.89e-03` | `0.00e+00` | `0.00e+00` | `5.73e-16` | `-307.397095` | `1.56e-02` |
| `pymanopt-Warped-CG` | 0/10 (0%) | 5000.0 (0.0) | 5000.0 (0.0) | 4799.62 (315.65) | `5.66e-01` | `0.00e+00` | `0.00e+00` | `6.53e-16` | `-304.173011` | `1.56e-02` |
| `pymanopt-TrustRegions` | 0/10 (0%) | 5000.0 (0.0) | 5000.0 (0.0) | 3054.23 (86.01) | `3.04e-01` | `0.00e+00` | `0.00e+00` | `2.68e-11` | `-304.608793` | `1.56e-02` |
| `SciPy-L-BFGS-B` | 0/10 (0%) | 5000.0 (0.0) | 5000.0 (0.0) | 3779.71 (62.66) | `9.24e-02` | `0.00e+00` | `0.00e+00` | `2.05e-05` | `-306.311418` | `1.56e-02` |
| `SciPy-SLSQP` | SKIP (`N_tot > 450`) | — | — | — | — | — | — | — | — | — |
| `SciPy-trust-constr` | SKIP (`N_tot > 450`) | — | — | — | — | — | — | — | — | — |

### `B2B`: Smooth Penalty-Split Non-Negative Sparse PCA

- **Regime**: `Z in [0,5]^{120x6} (d=720) x St(120,6) (Branch A, ~80% clamped Z=0, FM-2)` (`d = 720`, `R = 1`, `N_tot = 1440`, intrinsic dim `1419`, FD gradient max rel err `1.484e-08`)

| Solver | Solve Rate | Raw Evals Med (IQR) | Penalized Evals Med (IQR) | Wall ms Med (IQR) | Median KKT Ref | Median Box Viol | Median Lin Viol | Median Orth Viol | Median Objective | Wilcoxon Evals p |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `fortchuck` | 10/10 (100%) | 111.5 (54.2) | 111.5 (54.2) | 33.06 (16.19) | `6.96e-07` | `0.00e+00` | `0.00e+00` | `1.30e-14` | `8.784987` | `—` |
| `fortchuck-ABB` | 10/10 (100%) | 989.5 (941.8) | 989.5 (941.8) | 163.11 (157.89) | `9.30e-07` | `0.00e+00` | `0.00e+00` | `2.74e-14` | `8.200998` | `1.95e-03` |
| `Alternating-BCD` | 0/10 (0%) | 3521.0 (1630.8) | 5000.0 (0.0) | 2840.31 (1274.28) | `1.68e-04` | `0.00e+00` | `0.00e+00` | `9.37e-16` | `8.069873` | `1.95e-03` |
| `pymanopt-Warped-CG` | 3/10 (30%) | 4764.0 (3196.0) | 5000.0 (354.0) | 4443.59 (2971.71) | `4.25e-04` | `0.00e+00` | `0.00e+00` | `1.11e-15` | `13.886985` | `1.95e-03` |
| `pymanopt-TrustRegions` | 4/10 (40%) | 5000.0 (610.2) | 5000.0 (610.2) | 2967.06 (649.97) | `1.28e-02` | `0.00e+00` | `0.00e+00` | `1.10e-10` | `8.460341` | `1.95e-03` |
| `SciPy-L-BFGS-B` | 0/10 (0%) | 413.5 (538.0) | 5000.0 (0.0) | 336.72 (403.17) | `2.54e-05` | `0.00e+00` | `0.00e+00` | `5.91e-06` | `10.943772` | `1.95e-03` |
| `SciPy-SLSQP` | SKIP (`N_tot > 450`) | — | — | — | — | — | — | — | — | — |
| `SciPy-trust-constr` | SKIP (`N_tot > 450`) | — | — | — | — | — | — | — | — | — |

### `B3`: Multi-Camera Anisotropic Bounded-Scale Rig Wahba

- **Regime**: `z in B^15 x St(3,3)^4 (Branch B SO(3)^4, R=4, kappa(W)=10^3)` (`d = 15`, `R = 4`, `N_tot = 51`, intrinsic dim `27`, FD gradient max rel err `7.213e-11`)

| Solver | Solve Rate | Raw Evals Med (IQR) | Penalized Evals Med (IQR) | Wall ms Med (IQR) | Median KKT Ref | Median Box Viol | Median Lin Viol | Median Orth Viol | Median Objective | Wilcoxon Evals p |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `fortchuck` | 10/10 (100%) | 40.5 (8.2) | 40.5 (8.2) | 10.59 (3.26) | `4.78e-07` | `0.00e+00` | `0.00e+00` | `5.03e-15` | `2.176801` | `—` |
| `fortchuck-ABB` | 10/10 (100%) | 84.0 (7.2) | 84.0 (7.2) | 19.18 (1.33) | `5.06e-07` | `0.00e+00` | `0.00e+00` | `7.20e-15` | `2.176801` | `1.95e-03` |
| `Alternating-BCD` | 9/10 (90%) | 319.5 (12.8) | 323.5 (11.2) | 425.73 (30.51) | `8.77e-07` | `0.00e+00` | `0.00e+00` | `8.26e-16` | `2.176801` | `1.95e-03` |
| `pymanopt-Warped-CG` | 10/10 (100%) | 874.5 (304.5) | 874.5 (304.5) | 1607.06 (620.87) | `7.22e-07` | `0.00e+00` | `0.00e+00` | `8.45e-16` | `2.176801` | `1.95e-03` |
| `pymanopt-TrustRegions` | 10/10 (100%) | 658.0 (52.8) | 658.0 (52.8) | 675.10 (91.87) | `6.27e-07` | `0.00e+00` | `0.00e+00` | `7.28e-16` | `2.176801` | `1.95e-03` |
| `SciPy-L-BFGS-B` | 0/10 (0%) | 801.5 (66.2) | 5000.0 (0.0) | 753.68 (46.72) | `2.29e-05` | `0.00e+00` | `0.00e+00` | `4.83e-06` | `2.176772` | `1.95e-03` |
| `SciPy-SLSQP` | 10/10 (100%) | 75.0 (10.5) | 75.0 (10.5) | 74.74 (12.61) | `5.69e-07` | `0.00e+00` | `0.00e+00` | `6.82e-14` | `2.176801` | `1.95e-03` |
| `SciPy-trust-constr` | 10/10 (100%) | 90.0 (17.8) | 90.0 (17.8) | 445.71 (110.13) | `7.85e-07` | `0.00e+00` | `0.00e+00` | `2.46e-15` | `2.176802` | `1.95e-03` |

### `B4`: Bounded-Variance ML Common Principal Components

- **Regime**: `z in [0.05,20]^150 x St(30,30) (Branch B O(30), log barrier, 1.6e5 scale ratio)` (`d = 150`, `R = 1`, `N_tot = 1050`, intrinsic dim `585`, FD gradient max rel err `6.833e-10`)

| Solver | Solve Rate | Raw Evals Med (IQR) | Penalized Evals Med (IQR) | Wall ms Med (IQR) | Median KKT Ref | Median Box Viol | Median Lin Viol | Median Orth Viol | Median Objective | Wilcoxon Evals p |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `fortchuck` | 10/10 (100%) | 1810.5 (76.0) | 1810.5 (76.0) | 1179.24 (113.50) | `6.58e-07` | `0.00e+00` | `0.00e+00` | `6.23e-14` | `17.596323` | `—` |
| `fortchuck-ABB` | 9/10 (90%) | 4391.0 (314.0) | 4391.0 (314.0) | 1981.80 (348.59) | `7.00e-07` | `0.00e+00` | `0.00e+00` | `4.92e-14` | `17.596323` | `1.95e-03` |
| `Alternating-BCD` | 0/10 (0%) | 5000.0 (0.0) | 5000.0 (0.0) | 3953.62 (710.71) | `1.85e-04` | `0.00e+00` | `0.00e+00` | `3.07e-15` | `17.596323` | `1.95e-03` |
| `pymanopt-Warped-CG` | 0/10 (0%) | 5000.0 (0.0) | 5000.0 (0.0) | 5989.59 (859.56) | `8.56e-03` | `0.00e+00` | `0.00e+00` | `3.16e-15` | `17.599052` | `1.95e-03` |
| `pymanopt-TrustRegions` | 0/10 (0%) | 5000.0 (0.0) | 5000.0 (0.0) | 3716.99 (837.50) | `4.70e-03` | `0.00e+00` | `0.00e+00` | `9.85e-14` | `17.597364` | `1.95e-03` |
| `SciPy-L-BFGS-B` | 0/10 (0%) | 5000.0 (0.0) | 5000.0 (0.0) | 4772.65 (499.06) | `6.40e-01` | `0.00e+00` | `0.00e+00` | `1.03e-05` | `21.044210` | `1.95e-03` |
| `SciPy-SLSQP` | SKIP (`N_tot > 450`) | — | — | — | — | — | — | — | — | — |
| `SciPy-trust-constr` | SKIP (`N_tot > 450`) | — | — | — | — | — | — | — | — | — |

### `B5`: Bounded Factor Correlation Matrix Calibration

- **Regime**: `z in B^110 x St(100,10) (Branch A, quartic valley + clamped psi)` (`d = 110`, `R = 1`, `N_tot = 1110`, intrinsic dim `1055`, FD gradient max rel err `8.432e-10`)

| Solver | Solve Rate | Raw Evals Med (IQR) | Penalized Evals Med (IQR) | Wall ms Med (IQR) | Median KKT Ref | Median Box Viol | Median Lin Viol | Median Orth Viol | Median Objective | Wilcoxon Evals p |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `fortchuck` | 9/10 (90%) | 1534.0 (174.8) | 1570.0 (205.0) | 886.85 (235.05) | `9.45e-07` | `0.00e+00` | `0.00e+00` | `4.64e-14` | `34.668813` | `—` |
| `fortchuck-ABB` | 0/10 (0%) | 5000.0 (0.0) | 5000.0 (0.0) | 1720.30 (107.00) | `1.58e-03` | `0.00e+00` | `0.00e+00` | `3.98e-14` | `34.669022` | `3.91e-03` |
| `Alternating-BCD` | 0/10 (0%) | 5000.0 (0.0) | 5000.0 (0.0) | 3822.40 (343.45) | `4.55e-03` | `0.00e+00` | `0.00e+00` | `1.22e-15` | `34.669225` | `3.91e-03` |
| `pymanopt-Warped-CG` | 0/10 (0%) | 5000.0 (0.0) | 5000.0 (0.0) | 5608.24 (779.05) | `1.50e-02` | `0.00e+00` | `0.00e+00` | `1.30e-15` | `34.671020` | `3.91e-03` |
| `pymanopt-TrustRegions` | 0/10 (0%) | 5000.0 (0.0) | 5000.0 (0.0) | 3717.55 (578.86) | `2.28e-02` | `0.00e+00` | `0.00e+00` | `4.02e-12` | `34.669569` | `3.91e-03` |
| `SciPy-L-BFGS-B` | 0/10 (0%) | 2495.5 (595.8) | 5000.0 (0.0) | 2408.18 (436.03) | `3.64e-05` | `0.00e+00` | `0.00e+00` | `1.88e-10` | `34.668813` | `3.91e-03` |
| `SciPy-SLSQP` | SKIP (`N_tot > 450`) | — | — | — | — | — | — | — | — | — |
| `SciPy-trust-constr` | SKIP (`N_tot > 450`) | — | — | — | — | — | — | — | — | — |

### `B6`: Box-Coupled Brockett with Exact Ky Fan QP Truth

- **Regime**: `z in [0.1,5]^10 x St(150,10) (Branch A, kappa(M)=10^4, certified f_star)` (`d = 10`, `R = 1`, `N_tot = 1510`, intrinsic dim `1455`, FD gradient max rel err `3.584e-10`)

| Solver | Solve Rate | Raw Evals Med (IQR) | Penalized Evals Med (IQR) | Wall ms Med (IQR) | Median KKT Ref | Median Box Viol | Median Lin Viol | Median Orth Viol | Median Objective | Wilcoxon Evals p |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `fortchuck` | 10/10 (100%) | 2067.0 (286.0) | 2067.0 (286.0) | 1495.04 (258.48) | `9.47e-07` | `0.00e+00` | `0.00e+00` | `6.82e-14` | `-10.445510` | `—` |
| `fortchuck-ABB` | 6/10 (60%) | 4519.5 (823.2) | 4519.5 (823.2) | 1513.80 (162.20) | `7.63e-07` | `0.00e+00` | `0.00e+00` | `6.01e-14` | `-10.445510` | `1.95e-03` |
| `Alternating-BCD` | 0/10 (0%) | 5000.0 (0.0) | 5000.0 (0.0) | 3501.53 (251.68) | `2.09e-02` | `0.00e+00` | `0.00e+00` | `1.05e-15` | `-10.443098` | `1.95e-03` |
| `pymanopt-Warped-CG` | 0/10 (0%) | 5000.0 (0.0) | 5000.0 (0.0) | 5504.36 (396.95) | `5.17e-03` | `0.00e+00` | `0.00e+00` | `1.06e-15` | `-10.444992` | `1.95e-03` |
| `pymanopt-TrustRegions` | 0/10 (0%) | 5000.0 (0.0) | 5000.0 (0.0) | 3888.68 (375.86) | `1.80e-03` | `0.00e+00` | `0.00e+00` | `3.99e-15` | `-10.445432` | `1.95e-03` |
| `SciPy-L-BFGS-B` | 0/10 (0%) | 5000.0 (0.0) | 5000.0 (0.0) | 5127.43 (300.89) | `2.65e-03` | `0.00e+00` | `0.00e+00` | `1.50e-05` | `-10.445612` | `1.95e-03` |
| `SciPy-SLSQP` | SKIP (`N_tot > 450`) | — | — | — | — | — | — | — | — | — |
| `SciPy-trust-constr` | SKIP (`N_tot > 450`) | — | — | — | — | — | — | — | — | — |

### `B7A`: Pure-Box Active-Bound Chained Rosenbrock + Quadratic

- **Regime**: `z in [l,u]^500, R=0 (incl. 5 fixed l_i==u_i, ~60% active bounds)` (`d = 500`, `R = 0`, `N_tot = 500`, intrinsic dim `500`, FD gradient max rel err `9.086e-10`)

| Solver | Solve Rate | Raw Evals Med (IQR) | Penalized Evals Med (IQR) | Wall ms Med (IQR) | Median KKT Ref | Median Box Viol | Median Lin Viol | Median Orth Viol | Median Objective | Wilcoxon Evals p |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `fortchuck` | 10/10 (100%) | 124.0 (3.8) | 124.0 (3.8) | 20.25 (4.07) | `8.84e-07` | `0.00e+00` | `0.00e+00` | `0.00e+00` | `87.046277` | `—` |
| `fortchuck-ABB` | 10/10 (100%) | 85.0 (11.0) | 85.0 (11.0) | 12.43 (3.30) | `8.60e-07` | `0.00e+00` | `0.00e+00` | `0.00e+00` | `87.046277` | `1.95e-03` |
| `Alternating-BCD` | 10/10 (100%) | 68.0 (0.8) | 68.0 (0.8) | 49.07 (11.34) | `8.47e-07` | `0.00e+00` | `0.00e+00` | `0.00e+00` | `87.046277` | `1.95e-03` |
| `pymanopt-Warped-CG` | 0/10 (0%) | 5000.0 (0.0) | 5000.0 (0.0) | 2222.84 (144.03) | `1.28e-03` | `0.00e+00` | `0.00e+00` | `0.00e+00` | `87.047726` | `1.95e-03` |
| `pymanopt-TrustRegions` | 0/10 (0%) | 5000.0 (0.0) | 5000.0 (0.0) | 1928.81 (102.99) | `2.15e-04` | `0.00e+00` | `0.00e+00` | `0.00e+00` | `87.046452` | `1.95e-03` |
| `SciPy-L-BFGS-B` | 10/10 (100%) | 68.0 (0.8) | 68.0 (0.8) | 51.45 (7.97) | `8.47e-07` | `0.00e+00` | `0.00e+00` | `0.00e+00` | `87.046277` | `1.95e-03` |
| `SciPy-SLSQP` | SKIP (`N_tot > 450`) | — | — | — | — | — | — | — | — | — |
| `SciPy-trust-constr` | SKIP (`N_tot > 450`) | — | — | — | — | — | — | — | — | — |

### `B7B`: Pure-Stiefel Tall Brockett & Orthogonal Procrustes

- **Regime**: `d=0, St(200,8) x St(24,24) (Pure Stiefel Branch A & Branch B)` (`d = 0`, `R = 2`, `N_tot = 2176`, intrinsic dim `1840`, FD gradient max rel err `5.084e-10`)

| Solver | Solve Rate | Raw Evals Med (IQR) | Penalized Evals Med (IQR) | Wall ms Med (IQR) | Median KKT Ref | Median Box Viol | Median Lin Viol | Median Orth Viol | Median Objective | Wilcoxon Evals p |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `fortchuck` | 10/10 (100%) | 493.5 (53.2) | 493.5 (53.2) | 488.43 (107.53) | `6.88e-07` | `0.00e+00` | `0.00e+00` | `6.71e-14` | `-66.949417` | `—` |
| `fortchuck-ABB` | 10/10 (100%) | 662.0 (159.0) | 662.0 (159.0) | 297.61 (93.53) | `6.98e-07` | `0.00e+00` | `0.00e+00` | `7.31e-14` | `-66.949417` | `1.95e-03` |
| `Alternating-BCD` | 10/10 (100%) | 1062.5 (86.8) | 1062.5 (86.8) | 1580.81 (184.18) | `8.59e-07` | `0.00e+00` | `0.00e+00` | `2.46e-15` | `-66.949417` | `1.95e-03` |
| `pymanopt-Warped-CG` | 10/10 (100%) | 1062.5 (86.8) | 1062.5 (86.8) | 1584.33 (297.91) | `8.59e-07` | `0.00e+00` | `0.00e+00` | `2.46e-15` | `-66.949417` | `1.95e-03` |
| `pymanopt-TrustRegions` | 10/10 (100%) | 1700.5 (263.0) | 1700.5 (263.0) | 1407.55 (202.98) | `1.17e-08` | `0.00e+00` | `0.00e+00` | `2.93e-15` | `-66.949417` | `1.95e-03` |
| `SciPy-L-BFGS-B` | 0/10 (0%) | 2877.0 (831.8) | 5000.0 (0.0) | 2953.87 (820.27) | `1.42e-03` | `0.00e+00` | `0.00e+00` | `5.46e-05` | `-66.818295` | `1.95e-03` |
| `SciPy-SLSQP` | SKIP (`N_tot > 450`) | — | — | — | — | — | — | — | — | — |
| `SciPy-trust-constr` | SKIP (`N_tot > 450`) | — | — | — | — | — | — | — | — | — |

### `B8A`: Trace-Constrained Ensemble Kohn-Sham DFT

- **Regime**: `z in [0,1]^12 (1^T z = 5) x St(120,12) (SLBQP + Branch A, hard trace)` (`d = 12`, `R = 1`, `N_tot = 1452`, intrinsic dim `1373`, linear constraint `a^T z = 5.0`, FD gradient max rel err `1.745e-10`)

| Solver | Solve Rate | Raw Evals Med (IQR) | Penalized Evals Med (IQR) | Wall ms Med (IQR) | Median KKT Ref | Median Box Viol | Median Lin Viol | Median Orth Viol | Median Objective | Wilcoxon Evals p |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `fortchuck` | 10/10 (100%) | 150.0 (25.8) | 150.0 (25.8) | 123.95 (35.05) | `8.56e-07` | `0.00e+00` | `0.00e+00` | `2.59e-14` | `-8.904018` | `—` |
| `fortchuck-ABB` | 9/10 (90%) | 176.0 (47.8) | 176.0 (47.8) | 90.29 (35.17) | `7.47e-07` | `0.00e+00` | `0.00e+00` | `2.64e-14` | `-8.904018` | `1.76e-02` |
| `Alternating-BCD` | 10/10 (100%) | 353.0 (117.5) | 353.0 (117.5) | 419.30 (118.90) | `8.44e-07` | `0.00e+00` | `0.00e+00` | `1.27e-15` | `-8.904018` | `1.95e-03` |
| `pymanopt-Warped-CG` | 0/10 (0%) | 5000.0 (0.0) | 5000.0 (0.0) | 8140.29 (531.00) | `3.35e-03` | `0.00e+00` | `5.45e-05` | `1.41e-15` | `-8.908985` | `1.95e-03` |
| `pymanopt-TrustRegions` | 0/10 (0%) | 5000.0 (0.0) | 5000.0 (0.0) | 5886.07 (456.88) | `6.71e-03` | `0.00e+00` | `4.00e-06` | `1.76e-15` | `-8.909272` | `1.95e-03` |
| `SciPy-L-BFGS-B` | 0/10 (0%) | 2346.0 (1623.0) | 5000.0 (0.0) | 3591.28 (2493.65) | `5.15e-05` | `0.00e+00` | `5.03e-09` | `7.34e-06` | `-8.904045` | `1.95e-03` |
| `SciPy-SLSQP` | SKIP (`N_tot > 450`) | — | — | — | — | — | — | — | — | — |
| `SciPy-trust-constr` | SKIP (`N_tot > 450`) | — | — | — | — | — | — | — | — | — |

### `B8B`: Long-Only Weighted Orthogonal Risk-Factor Parity

- **Regime**: `z in [0,0.15]^24 (a^T z = 1) x St(32,6) x St(6,6) (N_tot=252, Branch A & B)` (`d = 24`, `R = 2`, `N_tot = 252`, intrinsic dim `209`, linear constraint `a^T z = 1.0`, FD gradient max rel err `1.682e-11`)

| Solver | Solve Rate | Raw Evals Med (IQR) | Penalized Evals Med (IQR) | Wall ms Med (IQR) | Median KKT Ref | Median Box Viol | Median Lin Viol | Median Orth Viol | Median Objective | Wilcoxon Evals p |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `fortchuck` | 10/10 (100%) | 241.0 (147.5) | 241.0 (147.5) | 74.83 (38.60) | `8.05e-07` | `0.00e+00` | `8.33e-17` | `2.78e-14` | `-4.990125` | `—` |
| `fortchuck-ABB` | 10/10 (100%) | 62.0 (9.5) | 62.0 (9.5) | 12.34 (3.80) | `8.46e-07` | `0.00e+00` | `5.55e-17` | `8.81e-15` | `-4.990125` | `5.86e-03` |
| `Alternating-BCD` | 10/10 (100%) | 359.5 (53.8) | 359.5 (53.8) | 480.54 (92.92) | `7.42e-07` | `0.00e+00` | `1.11e-16` | `9.85e-16` | `-4.990125` | `1.60e-01` |
| `pymanopt-Warped-CG` | 0/10 (0%) | 5000.0 (0.0) | 5000.0 (0.0) | 10134.91 (1998.83) | `9.15e-04` | `0.00e+00` | `5.16e-05` | `1.03e-15` | `-4.990102` | `1.95e-03` |
| `pymanopt-TrustRegions` | 0/10 (0%) | 5000.0 (0.0) | 5000.0 (0.0) | 6898.25 (902.39) | `2.52e-04` | `0.00e+00` | `2.56e-05` | `9.78e-16` | `-4.990136` | `1.95e-03` |
| `SciPy-L-BFGS-B` | 0/10 (0%) | 1014.5 (104.8) | 5000.0 (0.0) | 1321.31 (215.85) | `3.02e-05` | `0.00e+00` | `1.26e-07` | `2.19e-06` | `-4.990130` | `1.95e-03` |
| `SciPy-SLSQP` | 10/10 (100%) | 40.0 (10.5) | 40.0 (10.5) | 531.59 (53.82) | `6.99e-07` | `0.00e+00` | `6.66e-16` | `4.98e-12` | `-4.990125` | `3.91e-03` |
| `SciPy-trust-constr` | 0/10 (0%) | 199.0 (25.2) | 5000.0 (0.0) | 4338.92 (1083.11) | `3.14e-05` | `0.00e+00` | `0.00e+00` | `2.78e-13` | `-4.990124` | `1.95e-03` |

## 10. Per-Problem Breakdown — Out-of-Sample Holdout Suite (`Seeds 201..210`)

### `B1`: Ensemble Kohn-Sham DFT Fractional Occupations

- **Regime**: `z in [0,1]^10 x St(120,10) (Branch A, metallic orbital-occupation coupling)` (`d = 10`, `R = 1`, `N_tot = 1210`, intrinsic dim `1155`, FD gradient max rel err `2.091e-10`)

| Solver | Solve Rate | Raw Evals Med (IQR) | Penalized Evals Med (IQR) | Wall ms Med (IQR) | Median KKT Ref | Median Box Viol | Median Lin Viol | Median Orth Viol | Median Objective | Wilcoxon Evals p |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `fortchuck` | 10/10 (100%) | 2034.5 (671.2) | 2034.5 (671.2) | 1503.27 (400.12) | `8.98e-07` | `0.00e+00` | `0.00e+00` | `5.78e-14` | `-4.368467` | `—` |
| `fortchuck-ABB` | 1/10 (10%) | 5000.0 (0.0) | 5000.0 (0.0) | 1948.89 (307.18) | `2.63e-05` | `0.00e+00` | `0.00e+00` | `3.90e-14` | `-4.368467` | `1.95e-03` |
| `Alternating-BCD` | 2/10 (20%) | 5000.0 (0.0) | 5000.0 (0.0) | 4187.06 (612.51) | `1.37e-03` | `0.00e+00` | `0.00e+00` | `1.20e-15` | `-4.349173` | `2.73e-02` |
| `pymanopt-Warped-CG` | 0/10 (0%) | 5000.0 (0.0) | 5000.0 (0.0) | 6698.04 (450.08) | `8.43e-04` | `0.00e+00` | `0.00e+00` | `1.32e-15` | `-4.386120` | `1.95e-03` |
| `pymanopt-TrustRegions` | 0/10 (0%) | 5000.0 (0.0) | 5000.0 (0.0) | 4125.10 (482.54) | `9.59e-03` | `0.00e+00` | `0.00e+00` | `1.35e-13` | `-4.386097` | `1.95e-03` |
| `SciPy-L-BFGS-B` | 0/10 (0%) | 5000.0 (0.0) | 5000.0 (0.0) | 5300.89 (392.30) | `4.09e-04` | `0.00e+00` | `0.00e+00` | `3.20e-06` | `-4.358742` | `1.95e-03` |
| `SciPy-SLSQP` | SKIP (`N_tot > 450`) | — | — | — | — | — | — | — | — | — |
| `SciPy-trust-constr` | SKIP (`N_tot > 450`) | — | — | — | — | — | — | — | — | — |

### `B2A`: Amplitude-Limited Blind Source Separation (ICA)

- **Regime**: `Z in [-2,2]^{5x150} (d=750) x St(20,5) (Branch A, nonconvex log-cosh)` (`d = 750`, `R = 1`, `N_tot = 850`, intrinsic dim `835`, FD gradient max rel err `1.935e-09`)

| Solver | Solve Rate | Raw Evals Med (IQR) | Penalized Evals Med (IQR) | Wall ms Med (IQR) | Median KKT Ref | Median Box Viol | Median Lin Viol | Median Orth Viol | Median Objective | Wilcoxon Evals p |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `fortchuck` | 5/10 (50%) | 1067.5 (515.5) | 3188.5 (3840.8) | 336.50 (129.64) | `1.58e-06` | `0.00e+00` | `0.00e+00` | `5.75e-14` | `-311.252698` | `—` |
| `fortchuck-ABB` | 2/10 (20%) | 4781.0 (1149.2) | 5000.0 (0.0) | 1039.23 (270.31) | `2.63e-05` | `0.00e+00` | `0.00e+00` | `6.20e-14` | `-307.661972` | `6.25e-02` |
| `Alternating-BCD` | 2/10 (20%) | 5000.0 (0.0) | 5000.0 (0.0) | 4707.83 (337.84) | `1.17e-02` | `0.00e+00` | `0.00e+00` | `6.98e-16` | `-306.937441` | `6.25e-02` |
| `pymanopt-Warped-CG` | 0/10 (0%) | 5000.0 (0.0) | 5000.0 (0.0) | 6193.33 (306.29) | `4.67e-01` | `0.00e+00` | `0.00e+00` | `6.53e-16` | `-309.091804` | `6.25e-02` |
| `pymanopt-TrustRegions` | 0/10 (0%) | 5000.0 (0.0) | 5000.0 (0.0) | 3807.69 (231.54) | `6.25e-01` | `0.00e+00` | `0.00e+00` | `3.22e-11` | `-307.525867` | `6.25e-02` |
| `SciPy-L-BFGS-B` | 0/10 (0%) | 5000.0 (0.0) | 5000.0 (0.0) | 4433.42 (73.31) | `4.53e-02` | `0.00e+00` | `0.00e+00` | `3.04e-05` | `-309.847765` | `6.25e-02` |
| `SciPy-SLSQP` | SKIP (`N_tot > 450`) | — | — | — | — | — | — | — | — | — |
| `SciPy-trust-constr` | SKIP (`N_tot > 450`) | — | — | — | — | — | — | — | — | — |

### `B2B`: Smooth Penalty-Split Non-Negative Sparse PCA

- **Regime**: `Z in [0,5]^{120x6} (d=720) x St(120,6) (Branch A, ~80% clamped Z=0, FM-2)` (`d = 720`, `R = 1`, `N_tot = 1440`, intrinsic dim `1419`, FD gradient max rel err `1.028e-08`)

| Solver | Solve Rate | Raw Evals Med (IQR) | Penalized Evals Med (IQR) | Wall ms Med (IQR) | Median KKT Ref | Median Box Viol | Median Lin Viol | Median Orth Viol | Median Objective | Wilcoxon Evals p |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `fortchuck` | 10/10 (100%) | 163.0 (80.8) | 163.0 (80.8) | 53.46 (28.38) | `8.54e-07` | `0.00e+00` | `0.00e+00` | `1.41e-14` | `8.485380` | `—` |
| `fortchuck-ABB` | 10/10 (100%) | 1324.0 (931.2) | 1324.0 (931.2) | 257.47 (171.56) | `9.31e-07` | `0.00e+00` | `0.00e+00` | `3.18e-14` | `8.026764` | `1.95e-03` |
| `Alternating-BCD` | 0/10 (0%) | 3651.0 (2740.5) | 5000.0 (0.0) | 3379.63 (2486.22) | `2.14e-04` | `0.00e+00` | `0.00e+00` | `1.15e-15` | `8.188735` | `1.95e-03` |
| `pymanopt-Warped-CG` | 6/10 (60%) | 4045.0 (3606.5) | 4045.0 (3606.5) | 4469.22 (4312.87) | `9.27e-07` | `0.00e+00` | `0.00e+00` | `1.00e-15` | `13.121072` | `1.95e-03` |
| `pymanopt-TrustRegions` | 7/10 (70%) | 2930.5 (2722.0) | 2930.5 (2722.0) | 2229.88 (1813.15) | `8.56e-07` | `0.00e+00` | `0.00e+00` | `1.18e-15` | `8.282000` | `1.95e-03` |
| `SciPy-L-BFGS-B` | 0/10 (0%) | 358.5 (359.8) | 5000.0 (0.0) | 342.75 (315.34) | `1.20e-04` | `0.00e+00` | `0.00e+00` | `1.15e-05` | `13.161188` | `1.95e-03` |
| `SciPy-SLSQP` | SKIP (`N_tot > 450`) | — | — | — | — | — | — | — | — | — |
| `SciPy-trust-constr` | SKIP (`N_tot > 450`) | — | — | — | — | — | — | — | — | — |

### `B3`: Multi-Camera Anisotropic Bounded-Scale Rig Wahba

- **Regime**: `z in B^15 x St(3,3)^4 (Branch B SO(3)^4, R=4, kappa(W)=10^3)` (`d = 15`, `R = 4`, `N_tot = 51`, intrinsic dim `27`, FD gradient max rel err `9.877e-11`)

| Solver | Solve Rate | Raw Evals Med (IQR) | Penalized Evals Med (IQR) | Wall ms Med (IQR) | Median KKT Ref | Median Box Viol | Median Lin Viol | Median Orth Viol | Median Objective | Wilcoxon Evals p |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `fortchuck` | 10/10 (100%) | 51.0 (18.2) | 51.0 (18.2) | 15.32 (5.54) | `3.14e-07` | `0.00e+00` | `0.00e+00` | `5.86e-15` | `2.176801` | `—` |
| `fortchuck-ABB` | 10/10 (100%) | 82.5 (15.5) | 82.5 (15.5) | 24.90 (3.75) | `6.19e-07` | `0.00e+00` | `0.00e+00` | `6.72e-15` | `2.176801` | `1.95e-03` |
| `Alternating-BCD` | 10/10 (100%) | 316.5 (25.2) | 316.5 (25.2) | 554.42 (34.01) | `7.93e-07` | `0.00e+00` | `0.00e+00` | `6.11e-16` | `2.176801` | `1.95e-03` |
| `pymanopt-Warped-CG` | 10/10 (100%) | 854.0 (368.2) | 854.0 (368.2) | 2146.65 (921.44) | `8.58e-07` | `0.00e+00` | `0.00e+00` | `5.68e-16` | `2.176801` | `1.95e-03` |
| `pymanopt-TrustRegions` | 10/10 (100%) | 651.5 (77.5) | 651.5 (77.5) | 850.17 (144.72) | `5.39e-07` | `0.00e+00` | `0.00e+00` | `6.81e-16` | `2.176801` | `1.95e-03` |
| `SciPy-L-BFGS-B` | 0/10 (0%) | 805.0 (153.5) | 5000.0 (0.0) | 895.50 (193.70) | `2.28e-05` | `0.00e+00` | `0.00e+00` | `4.83e-06` | `2.176772` | `1.95e-03` |
| `SciPy-SLSQP` | 10/10 (100%) | 86.5 (18.5) | 86.5 (18.5) | 96.38 (24.81) | `4.44e-07` | `0.00e+00` | `0.00e+00` | `4.69e-14` | `2.176801` | `1.95e-03` |
| `SciPy-trust-constr` | 10/10 (100%) | 95.0 (28.2) | 95.0 (28.2) | 653.41 (222.00) | `7.81e-07` | `0.00e+00` | `0.00e+00` | `5.27e-15` | `2.176802` | `1.95e-03` |

### `B4`: Bounded-Variance ML Common Principal Components

- **Regime**: `z in [0.05,20]^150 x St(30,30) (Branch B O(30), log barrier, 1.6e5 scale ratio)` (`d = 150`, `R = 1`, `N_tot = 1050`, intrinsic dim `585`, FD gradient max rel err `4.202e-10`)

| Solver | Solve Rate | Raw Evals Med (IQR) | Penalized Evals Med (IQR) | Wall ms Med (IQR) | Median KKT Ref | Median Box Viol | Median Lin Viol | Median Orth Viol | Median Objective | Wilcoxon Evals p |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `fortchuck` | 10/10 (100%) | 1817.5 (183.8) | 1817.5 (183.8) | 1273.49 (172.24) | `6.77e-07` | `0.00e+00` | `0.00e+00` | `6.60e-14` | `17.596323` | `—` |
| `fortchuck-ABB` | 9/10 (90%) | 4523.0 (669.0) | 4523.0 (669.0) | 2239.17 (336.82) | `6.87e-07` | `0.00e+00` | `0.00e+00` | `7.15e-14` | `17.596323` | `1.95e-03` |
| `Alternating-BCD` | 0/10 (0%) | 5000.0 (0.0) | 5000.0 (0.0) | 4566.23 (553.09) | `9.87e-04` | `0.00e+00` | `0.00e+00` | `3.13e-15` | `17.596323` | `1.95e-03` |
| `pymanopt-Warped-CG` | 0/10 (0%) | 5000.0 (0.0) | 5000.0 (0.0) | 6480.33 (815.45) | `1.19e-02` | `0.00e+00` | `0.00e+00` | `3.02e-15` | `17.599161` | `1.95e-03` |
| `pymanopt-TrustRegions` | 0/10 (0%) | 5000.0 (0.0) | 5000.0 (0.0) | 4225.88 (243.14) | `5.17e-03` | `0.00e+00` | `0.00e+00` | `7.26e-14` | `17.597603` | `1.95e-03` |
| `SciPy-L-BFGS-B` | 0/10 (0%) | 5000.0 (0.0) | 5000.0 (0.0) | 5333.04 (324.77) | `5.91e-01` | `0.00e+00` | `0.00e+00` | `7.85e-06` | `20.832811` | `1.95e-03` |
| `SciPy-SLSQP` | SKIP (`N_tot > 450`) | — | — | — | — | — | — | — | — | — |
| `SciPy-trust-constr` | SKIP (`N_tot > 450`) | — | — | — | — | — | — | — | — | — |

### `B5`: Bounded Factor Correlation Matrix Calibration

- **Regime**: `z in B^110 x St(100,10) (Branch A, quartic valley + clamped psi)` (`d = 110`, `R = 1`, `N_tot = 1110`, intrinsic dim `1055`, FD gradient max rel err `9.454e-10`)

| Solver | Solve Rate | Raw Evals Med (IQR) | Penalized Evals Med (IQR) | Wall ms Med (IQR) | Median KKT Ref | Median Box Viol | Median Lin Viol | Median Orth Viol | Median Objective | Wilcoxon Evals p |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `fortchuck` | 8/10 (80%) | 1366.5 (255.5) | 1471.0 (298.5) | 860.46 (140.30) | `9.65e-07` | `0.00e+00` | `0.00e+00` | `4.43e-14` | `34.668813` | `—` |
| `fortchuck-ABB` | 0/10 (0%) | 5000.0 (0.0) | 5000.0 (0.0) | 1738.95 (119.02) | `1.63e-03` | `0.00e+00` | `0.00e+00` | `3.27e-14` | `34.668915` | `7.81e-03` |
| `Alternating-BCD` | 0/10 (0%) | 5000.0 (0.0) | 5000.0 (0.0) | 4211.69 (393.85) | `5.73e-03` | `0.00e+00` | `0.00e+00` | `9.96e-16` | `34.669074` | `7.81e-03` |
| `pymanopt-Warped-CG` | 0/10 (0%) | 5000.0 (0.0) | 5000.0 (0.0) | 6122.56 (849.62) | `2.43e-02` | `0.00e+00` | `0.00e+00` | `1.28e-15` | `34.671823` | `7.81e-03` |
| `pymanopt-TrustRegions` | 0/10 (0%) | 5000.0 (0.0) | 5000.0 (0.0) | 4039.39 (404.42) | `2.36e-02` | `0.00e+00` | `0.00e+00` | `3.87e-12` | `34.669714` | `7.81e-03` |
| `SciPy-L-BFGS-B` | 0/10 (0%) | 2508.5 (849.2) | 5000.0 (0.0) | 2472.08 (851.41) | `3.08e-05` | `0.00e+00` | `0.00e+00` | `1.28e-10` | `34.668813` | `7.81e-03` |
| `SciPy-SLSQP` | SKIP (`N_tot > 450`) | — | — | — | — | — | — | — | — | — |
| `SciPy-trust-constr` | SKIP (`N_tot > 450`) | — | — | — | — | — | — | — | — | — |

### `B6`: Box-Coupled Brockett with Exact Ky Fan QP Truth

- **Regime**: `z in [0.1,5]^10 x St(150,10) (Branch A, kappa(M)=10^4, certified f_star)` (`d = 10`, `R = 1`, `N_tot = 1510`, intrinsic dim `1455`, FD gradient max rel err `3.113e-10`)

| Solver | Solve Rate | Raw Evals Med (IQR) | Penalized Evals Med (IQR) | Wall ms Med (IQR) | Median KKT Ref | Median Box Viol | Median Lin Viol | Median Orth Viol | Median Objective | Wilcoxon Evals p |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `fortchuck` | 10/10 (100%) | 1922.0 (305.0) | 1922.0 (305.0) | 1295.01 (255.49) | `8.77e-07` | `0.00e+00` | `0.00e+00` | `6.41e-14` | `-10.445510` | `—` |
| `fortchuck-ABB` | 8/10 (80%) | 4688.5 (560.5) | 4688.5 (560.5) | 1435.12 (64.37) | `7.11e-07` | `0.00e+00` | `0.00e+00` | `2.98e-14` | `-10.445510` | `1.95e-03` |
| `Alternating-BCD` | 0/10 (0%) | 5000.0 (0.0) | 5000.0 (0.0) | 3190.03 (172.14) | `9.64e-03` | `0.00e+00` | `0.00e+00` | `1.06e-15` | `-10.443625` | `1.95e-03` |
| `pymanopt-Warped-CG` | 0/10 (0%) | 5000.0 (0.0) | 5000.0 (0.0) | 4846.70 (263.71) | `1.65e-03` | `0.00e+00` | `0.00e+00` | `1.16e-15` | `-10.445257` | `1.95e-03` |
| `pymanopt-TrustRegions` | 0/10 (0%) | 5000.0 (0.0) | 5000.0 (0.0) | 3153.64 (158.77) | `1.47e-03` | `0.00e+00` | `0.00e+00` | `6.34e-15` | `-10.445437` | `1.95e-03` |
| `SciPy-L-BFGS-B` | 0/10 (0%) | 5000.0 (0.0) | 5000.0 (0.0) | 4520.76 (127.35) | `3.25e-03` | `0.00e+00` | `0.00e+00` | `1.50e-05` | `-10.445593` | `1.95e-03` |
| `SciPy-SLSQP` | SKIP (`N_tot > 450`) | — | — | — | — | — | — | — | — | — |
| `SciPy-trust-constr` | SKIP (`N_tot > 450`) | — | — | — | — | — | — | — | — | — |

### `B7A`: Pure-Box Active-Bound Chained Rosenbrock + Quadratic

- **Regime**: `z in [l,u]^500, R=0 (incl. 5 fixed l_i==u_i, ~60% active bounds)` (`d = 500`, `R = 0`, `N_tot = 500`, intrinsic dim `500`, FD gradient max rel err `8.252e-10`)

| Solver | Solve Rate | Raw Evals Med (IQR) | Penalized Evals Med (IQR) | Wall ms Med (IQR) | Median KKT Ref | Median Box Viol | Median Lin Viol | Median Orth Viol | Median Objective | Wilcoxon Evals p |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `fortchuck` | 10/10 (100%) | 127.5 (6.5) | 127.5 (6.5) | 15.78 (1.31) | `8.51e-07` | `0.00e+00` | `0.00e+00` | `0.00e+00` | `87.046277` | `—` |
| `fortchuck-ABB` | 10/10 (100%) | 88.5 (2.8) | 88.5 (2.8) | 9.88 (1.12) | `7.14e-07` | `0.00e+00` | `0.00e+00` | `0.00e+00` | `87.046277` | `1.95e-03` |
| `Alternating-BCD` | 10/10 (100%) | 66.5 (2.0) | 66.5 (2.0) | 36.57 (3.66) | `8.99e-07` | `0.00e+00` | `0.00e+00` | `0.00e+00` | `87.046277` | `1.95e-03` |
| `pymanopt-Warped-CG` | 0/10 (0%) | 5000.0 (0.0) | 5000.0 (0.0) | 1714.06 (38.61) | `2.38e-02` | `0.00e+00` | `0.00e+00` | `0.00e+00` | `87.048625` | `1.95e-03` |
| `pymanopt-TrustRegions` | 0/10 (0%) | 5000.0 (0.0) | 5000.0 (0.0) | 1503.69 (32.02) | `1.88e-04` | `0.00e+00` | `0.00e+00` | `0.00e+00` | `87.046492` | `1.95e-03` |
| `SciPy-L-BFGS-B` | 10/10 (100%) | 66.5 (2.0) | 66.5 (2.0) | 37.95 (4.98) | `8.99e-07` | `0.00e+00` | `0.00e+00` | `0.00e+00` | `87.046277` | `1.95e-03` |
| `SciPy-SLSQP` | SKIP (`N_tot > 450`) | — | — | — | — | — | — | — | — | — |
| `SciPy-trust-constr` | SKIP (`N_tot > 450`) | — | — | — | — | — | — | — | — | — |

### `B7B`: Pure-Stiefel Tall Brockett & Orthogonal Procrustes

- **Regime**: `d=0, St(200,8) x St(24,24) (Pure Stiefel Branch A & Branch B)` (`d = 0`, `R = 2`, `N_tot = 2176`, intrinsic dim `1840`, FD gradient max rel err `4.872e-10`)

| Solver | Solve Rate | Raw Evals Med (IQR) | Penalized Evals Med (IQR) | Wall ms Med (IQR) | Median KKT Ref | Median Box Viol | Median Lin Viol | Median Orth Viol | Median Objective | Wilcoxon Evals p |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `fortchuck` | 10/10 (100%) | 501.5 (56.5) | 501.5 (56.5) | 412.94 (40.47) | `6.40e-07` | `0.00e+00` | `0.00e+00` | `6.60e-14` | `-66.949417` | `—` |
| `fortchuck-ABB` | 10/10 (100%) | 690.0 (75.2) | 690.0 (75.2) | 268.12 (40.54) | `6.79e-07` | `0.00e+00` | `0.00e+00` | `7.02e-14` | `-66.949417` | `1.95e-03` |
| `Alternating-BCD` | 9/10 (90%) | 1096.5 (126.2) | 1114.0 (191.2) | 1301.93 (105.04) | `9.00e-07` | `0.00e+00` | `0.00e+00` | `2.57e-15` | `-66.949417` | `1.95e-03` |
| `pymanopt-Warped-CG` | 9/10 (90%) | 1096.5 (126.2) | 1114.0 (191.2) | 1350.04 (184.21) | `9.00e-07` | `0.00e+00` | `0.00e+00` | `2.57e-15` | `-66.949417` | `1.95e-03` |
| `pymanopt-TrustRegions` | 10/10 (100%) | 1737.5 (210.0) | 1737.5 (210.0) | 1192.48 (87.22) | `2.65e-09` | `0.00e+00` | `0.00e+00` | `2.76e-15` | `-66.949417` | `1.95e-03` |
| `SciPy-L-BFGS-B` | 0/10 (0%) | 2210.0 (353.2) | 5000.0 (0.0) | 2024.72 (312.69) | `1.42e-03` | `0.00e+00` | `0.00e+00` | `5.46e-05` | `-66.950913` | `1.95e-03` |
| `SciPy-SLSQP` | SKIP (`N_tot > 450`) | — | — | — | — | — | — | — | — | — |
| `SciPy-trust-constr` | SKIP (`N_tot > 450`) | — | — | — | — | — | — | — | — | — |

### `B8A`: Trace-Constrained Ensemble Kohn-Sham DFT

- **Regime**: `z in [0,1]^12 (1^T z = 5) x St(120,12) (SLBQP + Branch A, hard trace)` (`d = 12`, `R = 1`, `N_tot = 1452`, intrinsic dim `1373`, linear constraint `a^T z = 5.0`, FD gradient max rel err `1.486e-10`)

| Solver | Solve Rate | Raw Evals Med (IQR) | Penalized Evals Med (IQR) | Wall ms Med (IQR) | Median KKT Ref | Median Box Viol | Median Lin Viol | Median Orth Viol | Median Objective | Wilcoxon Evals p |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `fortchuck` | 10/10 (100%) | 157.0 (30.2) | 157.0 (30.2) | 129.04 (29.37) | `8.58e-07` | `0.00e+00` | `0.00e+00` | `2.43e-14` | `-8.904018` | `—` |
| `fortchuck-ABB` | 10/10 (100%) | 216.5 (49.2) | 216.5 (49.2) | 89.91 (15.87) | `6.06e-07` | `0.00e+00` | `0.00e+00` | `2.89e-14` | `-8.904018` | `1.95e-03` |
| `Alternating-BCD` | 10/10 (100%) | 410.0 (110.5) | 410.0 (110.5) | 448.38 (95.32) | `8.01e-07` | `0.00e+00` | `0.00e+00` | `1.30e-15` | `-8.904018` | `1.95e-03` |
| `pymanopt-Warped-CG` | 0/10 (0%) | 5000.0 (0.0) | 5000.0 (0.0) | 7038.82 (609.81) | `7.87e-03` | `0.00e+00` | `5.61e-05` | `1.31e-15` | `-8.908933` | `1.95e-03` |
| `pymanopt-TrustRegions` | 0/10 (0%) | 5000.0 (0.0) | 5000.0 (0.0) | 5136.18 (273.24) | `8.47e-03` | `0.00e+00` | `4.48e-06` | `1.49e-15` | `-8.909349` | `1.95e-03` |
| `SciPy-L-BFGS-B` | 0/10 (0%) | 2717.5 (1812.2) | 5000.0 (0.0) | 3688.06 (2219.00) | `5.89e-05` | `0.00e+00` | `6.58e-08` | `7.34e-06` | `-8.904085` | `1.95e-03` |
| `SciPy-SLSQP` | SKIP (`N_tot > 450`) | — | — | — | — | — | — | — | — | — |
| `SciPy-trust-constr` | SKIP (`N_tot > 450`) | — | — | — | — | — | — | — | — | — |

### `B8B`: Long-Only Weighted Orthogonal Risk-Factor Parity

- **Regime**: `z in [0,0.15]^24 (a^T z = 1) x St(32,6) x St(6,6) (N_tot=252, Branch A & B)` (`d = 24`, `R = 2`, `N_tot = 252`, intrinsic dim `209`, linear constraint `a^T z = 1.0`, FD gradient max rel err `1.181e-11`)

| Solver | Solve Rate | Raw Evals Med (IQR) | Penalized Evals Med (IQR) | Wall ms Med (IQR) | Median KKT Ref | Median Box Viol | Median Lin Viol | Median Orth Viol | Median Objective | Wilcoxon Evals p |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `fortchuck` | 10/10 (100%) | 466.0 (252.0) | 466.0 (252.0) | 106.83 (68.54) | `5.49e-07` | `0.00e+00` | `2.78e-17` | `3.38e-14` | `-4.990125` | `—` |
| `fortchuck-ABB` | 10/10 (100%) | 64.5 (9.2) | 64.5 (9.2) | 11.76 (1.33) | `7.56e-07` | `0.00e+00` | `5.55e-17` | `8.80e-15` | `-4.990125` | `1.95e-03` |
| `Alternating-BCD` | 10/10 (100%) | 367.0 (65.8) | 367.0 (65.8) | 462.81 (62.18) | `8.88e-07` | `0.00e+00` | `1.67e-16` | `1.11e-15` | `-4.990125` | `3.75e-01` |
| `pymanopt-Warped-CG` | 0/10 (0%) | 5000.0 (0.0) | 5000.0 (0.0) | 8786.56 (2036.03) | `9.83e-04` | `0.00e+00` | `5.15e-05` | `9.63e-16` | `-4.990070` | `1.95e-03` |
| `pymanopt-TrustRegions` | 0/10 (0%) | 5000.0 (0.0) | 5000.0 (0.0) | 5930.38 (522.23) | `1.98e-04` | `0.00e+00` | `2.21e-05` | `9.29e-16` | `-4.990134` | `1.95e-03` |
| `SciPy-L-BFGS-B` | 0/10 (0%) | 1100.0 (169.2) | 5000.0 (0.0) | 1284.50 (191.38) | `1.79e-05` | `0.00e+00` | `1.26e-07` | `2.19e-06` | `-4.990130` | `1.95e-03` |
| `SciPy-SLSQP` | 10/10 (100%) | 37.5 (5.8) | 37.5 (5.8) | 479.05 (28.25) | `7.53e-07` | `0.00e+00` | `6.66e-16` | `5.00e-12` | `-4.990125` | `1.95e-03` |
| `SciPy-trust-constr` | 0/10 (0%) | 183.5 (21.5) | 5000.0 (0.0) | 3479.97 (807.23) | `2.88e-05` | `0.00e+00` | `2.78e-17` | `2.47e-14` | `-4.990124` | `1.95e-03` |

## 11. Per-Problem Breakdown — Untouched Third Blind Holdout Suite (`Seeds 301..310`)

### `B1`: Ensemble Kohn-Sham DFT Fractional Occupations

- **Regime**: `z in [0,1]^10 x St(120,10) (Branch A, metallic orbital-occupation coupling)` (`d = 10`, `R = 1`, `N_tot = 1210`, intrinsic dim `1155`, FD gradient max rel err `2.275e-10`)

| Solver | Solve Rate | Raw Evals Med (IQR) | Penalized Evals Med (IQR) | Wall ms Med (IQR) | Median KKT Ref | Median Box Viol | Median Lin Viol | Median Orth Viol | Median Objective | Wilcoxon Evals p |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `fortchuck` | 10/10 (100%) | 2023.5 (655.8) | 2023.5 (655.8) | 1289.15 (406.59) | `8.92e-07` | `0.00e+00` | `0.00e+00` | `5.39e-14` | `-4.377192` | `—` |
| `fortchuck-ABB` | 0/10 (0%) | 5000.0 (0.0) | 5000.0 (0.0) | 1622.27 (81.69) | `2.59e-05` | `0.00e+00` | `0.00e+00` | `5.66e-14` | `-4.377185` | `1.95e-03` |
| `Alternating-BCD` | 1/10 (10%) | 5000.0 (0.0) | 5000.0 (0.0) | 3341.34 (187.93) | `8.37e-04` | `0.00e+00` | `0.00e+00` | `1.33e-15` | `-4.349318` | `3.91e-03` |
| `pymanopt-Warped-CG` | 0/10 (0%) | 5000.0 (0.0) | 5000.0 (0.0) | 5263.93 (396.98) | `4.25e-04` | `0.00e+00` | `0.00e+00` | `1.19e-15` | `-4.386121` | `1.95e-03` |
| `pymanopt-TrustRegions` | 0/10 (0%) | 5000.0 (0.0) | 5000.0 (0.0) | 3396.10 (133.47) | `7.50e-03` | `0.00e+00` | `0.00e+00` | `1.49e-13` | `-4.386074` | `1.95e-03` |
| `SciPy-L-BFGS-B` | 0/10 (0%) | 5000.0 (0.0) | 5000.0 (0.0) | 4593.80 (218.36) | `9.15e-04` | `0.00e+00` | `0.00e+00` | `3.19e-06` | `-4.368334` | `1.95e-03` |
| `SciPy-SLSQP` | SKIP (`N_tot > 450`) | — | — | — | — | — | — | — | — | — |
| `SciPy-trust-constr` | SKIP (`N_tot > 450`) | — | — | — | — | — | — | — | — | — |

### `B2A`: Amplitude-Limited Blind Source Separation (ICA)

- **Regime**: `Z in [-2,2]^{5x150} (d=750) x St(20,5) (Branch A, nonconvex log-cosh)` (`d = 750`, `R = 1`, `N_tot = 850`, intrinsic dim `835`, FD gradient max rel err `1.702e-09`)

| Solver | Solve Rate | Raw Evals Med (IQR) | Penalized Evals Med (IQR) | Wall ms Med (IQR) | Median KKT Ref | Median Box Viol | Median Lin Viol | Median Orth Viol | Median Objective | Wilcoxon Evals p |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `fortchuck` | 4/10 (40%) | 1014.5 (455.0) | 5000.0 (4088.5) | 289.44 (106.31) | `2.19e-06` | `0.00e+00` | `0.00e+00` | `6.04e-14` | `-311.310760` | `—` |
| `fortchuck-ABB` | 0/10 (0%) | 3362.0 (1871.0) | 5000.0 (0.0) | 802.55 (528.22) | `2.88e-06` | `0.00e+00` | `0.00e+00` | `5.72e-14` | `-308.599616` | `1.25e-01` |
| `Alternating-BCD` | 3/10 (30%) | 5000.0 (345.0) | 5000.0 (345.0) | 4301.17 (761.73) | `1.07e-03` | `0.00e+00` | `0.00e+00` | `7.67e-16` | `-307.215217` | `2.19e-01` |
| `pymanopt-Warped-CG` | 0/10 (0%) | 5000.0 (0.0) | 5000.0 (0.0) | 5673.81 (1262.20) | `5.00e-01` | `0.00e+00` | `0.00e+00` | `7.00e-16` | `-306.743575` | `1.25e-01` |
| `pymanopt-TrustRegions` | 0/10 (0%) | 5000.0 (0.0) | 5000.0 (0.0) | 3518.02 (465.16) | `7.23e-01` | `0.00e+00` | `0.00e+00` | `3.03e-11` | `-307.111514` | `1.25e-01` |
| `SciPy-L-BFGS-B` | 0/10 (0%) | 5000.0 (0.0) | 5000.0 (0.0) | 3941.67 (602.58) | `3.95e-02` | `0.00e+00` | `0.00e+00` | `3.45e-05` | `-308.408565` | `1.25e-01` |
| `SciPy-SLSQP` | SKIP (`N_tot > 450`) | — | — | — | — | — | — | — | — | — |
| `SciPy-trust-constr` | SKIP (`N_tot > 450`) | — | — | — | — | — | — | — | — | — |

### `B2B`: Smooth Penalty-Split Non-Negative Sparse PCA

- **Regime**: `Z in [0,5]^{120x6} (d=720) x St(120,6) (Branch A, ~80% clamped Z=0, FM-2)` (`d = 720`, `R = 1`, `N_tot = 1440`, intrinsic dim `1419`, FD gradient max rel err `1.215e-08`)

| Solver | Solve Rate | Raw Evals Med (IQR) | Penalized Evals Med (IQR) | Wall ms Med (IQR) | Median KKT Ref | Median Box Viol | Median Lin Viol | Median Orth Viol | Median Objective | Wilcoxon Evals p |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `fortchuck` | 10/10 (100%) | 130.5 (28.5) | 130.5 (28.5) | 39.63 (15.56) | `6.54e-07` | `0.00e+00` | `0.00e+00` | `1.57e-14` | `8.400194` | `—` |
| `fortchuck-ABB` | 10/10 (100%) | 887.0 (926.5) | 887.0 (926.5) | 173.73 (188.53) | `9.16e-07` | `0.00e+00` | `0.00e+00` | `3.05e-14` | `8.485785` | `3.91e-03` |
| `Alternating-BCD` | 0/10 (0%) | 4325.5 (1129.5) | 5000.0 (0.0) | 4038.10 (751.32) | `8.46e-05` | `0.00e+00` | `0.00e+00` | `1.15e-15` | `8.264379` | `1.95e-03` |
| `pymanopt-Warped-CG` | 5/10 (50%) | 1796.0 (3645.8) | 4706.5 (2226.8) | 2032.46 (4026.01) | `8.22e-03` | `0.00e+00` | `0.00e+00` | `1.19e-15` | `14.206347` | `1.95e-03` |
| `pymanopt-TrustRegions` | 8/10 (80%) | 2960.5 (1695.2) | 2960.5 (1695.2) | 2133.59 (817.99) | `8.01e-07` | `0.00e+00` | `0.00e+00` | `1.08e-15` | `8.231148` | `1.95e-03` |
| `SciPy-L-BFGS-B` | 0/10 (0%) | 700.5 (478.8) | 5000.0 (0.0) | 592.63 (382.00) | `1.96e-05` | `0.00e+00` | `0.00e+00` | `5.00e-06` | `9.289392` | `1.95e-03` |
| `SciPy-SLSQP` | SKIP (`N_tot > 450`) | — | — | — | — | — | — | — | — | — |
| `SciPy-trust-constr` | SKIP (`N_tot > 450`) | — | — | — | — | — | — | — | — | — |

### `B3`: Multi-Camera Anisotropic Bounded-Scale Rig Wahba

- **Regime**: `z in B^15 x St(3,3)^4 (Branch B SO(3)^4, R=4, kappa(W)=10^3)` (`d = 15`, `R = 4`, `N_tot = 51`, intrinsic dim `27`, FD gradient max rel err `1.384e-10`)

| Solver | Solve Rate | Raw Evals Med (IQR) | Penalized Evals Med (IQR) | Wall ms Med (IQR) | Median KKT Ref | Median Box Viol | Median Lin Viol | Median Orth Viol | Median Objective | Wilcoxon Evals p |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `fortchuck` | 10/10 (100%) | 42.0 (6.2) | 42.0 (6.2) | 11.78 (2.67) | `6.23e-07` | `0.00e+00` | `0.00e+00` | `4.31e-15` | `2.176801` | `—` |
| `fortchuck-ABB` | 10/10 (100%) | 81.5 (13.2) | 81.5 (13.2) | 22.12 (4.00) | `6.63e-07` | `0.00e+00` | `0.00e+00` | `7.31e-15` | `2.176801` | `1.95e-03` |
| `Alternating-BCD` | 9/10 (90%) | 310.0 (14.0) | 310.0 (14.0) | 497.68 (50.36) | `8.23e-07` | `0.00e+00` | `0.00e+00` | `6.90e-16` | `2.176801` | `1.95e-03` |
| `pymanopt-Warped-CG` | 10/10 (100%) | 735.0 (206.8) | 735.0 (206.8) | 1598.59 (603.58) | `8.37e-07` | `0.00e+00` | `0.00e+00` | `7.52e-16` | `2.176801` | `1.95e-03` |
| `pymanopt-TrustRegions` | 10/10 (100%) | 639.0 (119.2) | 639.0 (119.2) | 789.29 (118.53) | `6.40e-07` | `0.00e+00` | `0.00e+00` | `7.01e-16` | `2.176801` | `1.95e-03` |
| `SciPy-L-BFGS-B` | 0/10 (0%) | 813.0 (65.0) | 5000.0 (0.0) | 846.44 (94.91) | `2.24e-05` | `0.00e+00` | `0.00e+00` | `4.83e-06` | `2.176772` | `1.95e-03` |
| `SciPy-SLSQP` | 10/10 (100%) | 80.0 (8.2) | 80.0 (8.2) | 85.41 (16.28) | `2.32e-07` | `0.00e+00` | `0.00e+00` | `1.93e-14` | `2.176801` | `1.95e-03` |
| `SciPy-trust-constr` | 10/10 (100%) | 83.5 (21.8) | 83.5 (21.8) | 494.87 (211.90) | `8.10e-07` | `0.00e+00` | `0.00e+00` | `5.66e-15` | `2.176802` | `1.95e-03` |

### `B4`: Bounded-Variance ML Common Principal Components

- **Regime**: `z in [0.05,20]^150 x St(30,30) (Branch B O(30), log barrier, 1.6e5 scale ratio)` (`d = 150`, `R = 1`, `N_tot = 1050`, intrinsic dim `585`, FD gradient max rel err `7.163e-10`)

| Solver | Solve Rate | Raw Evals Med (IQR) | Penalized Evals Med (IQR) | Wall ms Med (IQR) | Median KKT Ref | Median Box Viol | Median Lin Viol | Median Orth Viol | Median Objective | Wilcoxon Evals p |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `fortchuck` | 10/10 (100%) | 1881.0 (171.2) | 1881.0 (171.2) | 1573.12 (311.73) | `6.37e-07` | `0.00e+00` | `0.00e+00` | `6.52e-14` | `17.596323` | `—` |
| `fortchuck-ABB` | 8/10 (80%) | 4379.5 (813.0) | 4379.5 (813.0) | 2423.93 (748.21) | `7.02e-07` | `0.00e+00` | `0.00e+00` | `5.50e-14` | `17.596323` | `1.95e-03` |
| `Alternating-BCD` | 0/10 (0%) | 5000.0 (0.0) | 5000.0 (0.0) | 4646.57 (1094.44) | `7.83e-04` | `0.00e+00` | `0.00e+00` | `3.34e-15` | `17.596324` | `1.95e-03` |
| `pymanopt-Warped-CG` | 0/10 (0%) | 5000.0 (0.0) | 5000.0 (0.0) | 7146.86 (1284.70) | `6.23e-03` | `0.00e+00` | `0.00e+00` | `3.04e-15` | `17.599196` | `1.95e-03` |
| `pymanopt-TrustRegions` | 0/10 (0%) | 5000.0 (0.0) | 5000.0 (0.0) | 4667.13 (525.76) | `4.15e-03` | `0.00e+00` | `0.00e+00` | `1.40e-13` | `17.597122` | `1.95e-03` |
| `SciPy-L-BFGS-B` | 0/10 (0%) | 5000.0 (0.0) | 5000.0 (0.0) | 5359.46 (392.96) | `5.19e-01` | `0.00e+00` | `0.00e+00` | `5.74e-06` | `20.590865` | `1.95e-03` |
| `SciPy-SLSQP` | SKIP (`N_tot > 450`) | — | — | — | — | — | — | — | — | — |
| `SciPy-trust-constr` | SKIP (`N_tot > 450`) | — | — | — | — | — | — | — | — | — |

### `B5`: Bounded Factor Correlation Matrix Calibration

- **Regime**: `z in B^110 x St(100,10) (Branch A, quartic valley + clamped psi)` (`d = 110`, `R = 1`, `N_tot = 1110`, intrinsic dim `1055`, FD gradient max rel err `7.298e-10`)

| Solver | Solve Rate | Raw Evals Med (IQR) | Penalized Evals Med (IQR) | Wall ms Med (IQR) | Median KKT Ref | Median Box Viol | Median Lin Viol | Median Orth Viol | Median Objective | Wilcoxon Evals p |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `fortchuck` | 9/10 (90%) | 1466.5 (233.2) | 1466.5 (240.0) | 869.06 (155.02) | `9.25e-07` | `0.00e+00` | `0.00e+00` | `4.76e-14` | `34.668813` | `—` |
| `fortchuck-ABB` | 0/10 (0%) | 5000.0 (0.0) | 5000.0 (0.0) | 1765.63 (49.29) | `3.78e-04` | `0.00e+00` | `0.00e+00` | `4.61e-14` | `34.668819` | `3.91e-03` |
| `Alternating-BCD` | 0/10 (0%) | 5000.0 (0.0) | 5000.0 (0.0) | 3899.44 (120.82) | `6.51e-03` | `0.00e+00` | `0.00e+00` | `1.15e-15` | `34.669151` | `3.91e-03` |
| `pymanopt-Warped-CG` | 0/10 (0%) | 5000.0 (0.0) | 5000.0 (0.0) | 5968.92 (380.28) | `3.22e-02` | `0.00e+00` | `0.00e+00` | `1.37e-15` | `34.671898` | `3.91e-03` |
| `pymanopt-TrustRegions` | 0/10 (0%) | 5000.0 (0.0) | 5000.0 (0.0) | 3874.64 (169.40) | `2.78e-02` | `0.00e+00` | `0.00e+00` | `3.55e-12` | `34.669870` | `3.91e-03` |
| `SciPy-L-BFGS-B` | 0/10 (0%) | 3007.5 (1162.5) | 5000.0 (0.0) | 3012.95 (1009.36) | `4.58e-05` | `0.00e+00` | `0.00e+00` | `2.16e-10` | `34.668813` | `3.91e-03` |
| `SciPy-SLSQP` | SKIP (`N_tot > 450`) | — | — | — | — | — | — | — | — | — |
| `SciPy-trust-constr` | SKIP (`N_tot > 450`) | — | — | — | — | — | — | — | — | — |

### `B6`: Box-Coupled Brockett with Exact Ky Fan QP Truth

- **Regime**: `z in [0.1,5]^10 x St(150,10) (Branch A, kappa(M)=10^4, certified f_star)` (`d = 10`, `R = 1`, `N_tot = 1510`, intrinsic dim `1455`, FD gradient max rel err `2.405e-10`)

| Solver | Solve Rate | Raw Evals Med (IQR) | Penalized Evals Med (IQR) | Wall ms Med (IQR) | Median KKT Ref | Median Box Viol | Median Lin Viol | Median Orth Viol | Median Objective | Wilcoxon Evals p |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `fortchuck` | 10/10 (100%) | 1875.5 (275.5) | 1875.5 (275.5) | 1184.81 (392.05) | `9.08e-07` | `0.00e+00` | `0.00e+00` | `6.77e-14` | `-10.445510` | `—` |
| `fortchuck-ABB` | 9/10 (90%) | 4563.0 (855.0) | 4563.0 (855.0) | 1280.69 (245.34) | `7.22e-07` | `0.00e+00` | `0.00e+00` | `4.76e-14` | `-10.445510` | `1.95e-03` |
| `Alternating-BCD` | 0/10 (0%) | 5000.0 (0.0) | 5000.0 (0.0) | 3139.84 (231.89) | `1.26e-02` | `0.00e+00` | `0.00e+00` | `9.63e-16` | `-10.443719` | `1.95e-03` |
| `pymanopt-Warped-CG` | 0/10 (0%) | 5000.0 (0.0) | 5000.0 (0.0) | 4844.96 (815.23) | `1.60e-03` | `0.00e+00` | `0.00e+00` | `9.81e-16` | `-10.445302` | `1.95e-03` |
| `pymanopt-TrustRegions` | 0/10 (0%) | 5000.0 (0.0) | 5000.0 (0.0) | 3125.18 (792.84) | `2.53e-03` | `0.00e+00` | `0.00e+00` | `1.83e-14` | `-10.445428` | `1.95e-03` |
| `SciPy-L-BFGS-B` | 0/10 (0%) | 5000.0 (0.0) | 5000.0 (0.0) | 4538.45 (144.32) | `3.67e-03` | `0.00e+00` | `0.00e+00` | `1.50e-05` | `-10.445568` | `1.95e-03` |
| `SciPy-SLSQP` | SKIP (`N_tot > 450`) | — | — | — | — | — | — | — | — | — |
| `SciPy-trust-constr` | SKIP (`N_tot > 450`) | — | — | — | — | — | — | — | — | — |

### `B7A`: Pure-Box Active-Bound Chained Rosenbrock + Quadratic

- **Regime**: `z in [l,u]^500, R=0 (incl. 5 fixed l_i==u_i, ~60% active bounds)` (`d = 500`, `R = 0`, `N_tot = 500`, intrinsic dim `500`, FD gradient max rel err `7.747e-10`)

| Solver | Solve Rate | Raw Evals Med (IQR) | Penalized Evals Med (IQR) | Wall ms Med (IQR) | Median KKT Ref | Median Box Viol | Median Lin Viol | Median Orth Viol | Median Objective | Wilcoxon Evals p |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `fortchuck` | 10/10 (100%) | 128.0 (9.8) | 128.0 (9.8) | 16.24 (2.74) | `8.07e-07` | `0.00e+00` | `0.00e+00` | `0.00e+00` | `87.046277` | `—` |
| `fortchuck-ABB` | 10/10 (100%) | 90.0 (4.8) | 90.0 (4.8) | 10.33 (1.94) | `7.35e-07` | `0.00e+00` | `0.00e+00` | `0.00e+00` | `87.046277` | `1.95e-03` |
| `Alternating-BCD` | 10/10 (100%) | 67.0 (1.8) | 67.0 (1.8) | 36.65 (1.63) | `9.05e-07` | `0.00e+00` | `0.00e+00` | `0.00e+00` | `87.046277` | `1.95e-03` |
| `pymanopt-Warped-CG` | 0/10 (0%) | 5000.0 (0.0) | 5000.0 (0.0) | 1792.26 (70.20) | `3.07e-01` | `0.00e+00` | `0.00e+00` | `0.00e+00` | `87.064904` | `1.95e-03` |
| `pymanopt-TrustRegions` | 0/10 (0%) | 5000.0 (0.0) | 5000.0 (0.0) | 1608.25 (37.35) | `1.16e-04` | `0.00e+00` | `0.00e+00` | `0.00e+00` | `87.046429` | `1.95e-03` |
| `SciPy-L-BFGS-B` | 10/10 (100%) | 67.0 (1.8) | 67.0 (1.8) | 36.38 (1.90) | `9.05e-07` | `0.00e+00` | `0.00e+00` | `0.00e+00` | `87.046277` | `1.95e-03` |
| `SciPy-SLSQP` | SKIP (`N_tot > 450`) | — | — | — | — | — | — | — | — | — |
| `SciPy-trust-constr` | SKIP (`N_tot > 450`) | — | — | — | — | — | — | — | — | — |

### `B7B`: Pure-Stiefel Tall Brockett & Orthogonal Procrustes

- **Regime**: `d=0, St(200,8) x St(24,24) (Pure Stiefel Branch A & Branch B)` (`d = 0`, `R = 2`, `N_tot = 2176`, intrinsic dim `1840`, FD gradient max rel err `4.765e-10`)

| Solver | Solve Rate | Raw Evals Med (IQR) | Penalized Evals Med (IQR) | Wall ms Med (IQR) | Median KKT Ref | Median Box Viol | Median Lin Viol | Median Orth Viol | Median Objective | Wilcoxon Evals p |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `fortchuck` | 10/10 (100%) | 490.5 (59.8) | 490.5 (59.8) | 418.71 (62.46) | `6.92e-07` | `0.00e+00` | `0.00e+00` | `5.54e-14` | `-66.949417` | `—` |
| `fortchuck-ABB` | 10/10 (100%) | 637.5 (31.8) | 637.5 (31.8) | 246.29 (42.14) | `7.26e-07` | `0.00e+00` | `0.00e+00` | `6.12e-14` | `-66.949417` | `1.95e-03` |
| `Alternating-BCD` | 10/10 (100%) | 1064.0 (142.2) | 1064.0 (142.2) | 1281.05 (218.27) | `9.06e-07` | `0.00e+00` | `0.00e+00` | `2.65e-15` | `-66.949417` | `1.95e-03` |
| `pymanopt-Warped-CG` | 10/10 (100%) | 1064.0 (142.2) | 1064.0 (142.2) | 1295.12 (225.78) | `9.06e-07` | `0.00e+00` | `0.00e+00` | `2.65e-15` | `-66.949417` | `1.95e-03` |
| `pymanopt-TrustRegions` | 10/10 (100%) | 1504.0 (235.2) | 1504.0 (235.2) | 1044.80 (203.31) | `2.06e-08` | `0.00e+00` | `0.00e+00` | `2.60e-15` | `-66.949417` | `1.95e-03` |
| `SciPy-L-BFGS-B` | 0/10 (0%) | 2636.5 (954.8) | 5000.0 (0.0) | 2361.48 (906.76) | `1.42e-03` | `0.00e+00` | `0.00e+00` | `5.46e-05` | `-66.884604` | `1.95e-03` |
| `SciPy-SLSQP` | SKIP (`N_tot > 450`) | — | — | — | — | — | — | — | — | — |
| `SciPy-trust-constr` | SKIP (`N_tot > 450`) | — | — | — | — | — | — | — | — | — |

### `B8A`: Trace-Constrained Ensemble Kohn-Sham DFT

- **Regime**: `z in [0,1]^12 (1^T z = 5) x St(120,12) (SLBQP + Branch A, hard trace)` (`d = 12`, `R = 1`, `N_tot = 1452`, intrinsic dim `1373`, linear constraint `a^T z = 5.0`, FD gradient max rel err `1.275e-10`)

| Solver | Solve Rate | Raw Evals Med (IQR) | Penalized Evals Med (IQR) | Wall ms Med (IQR) | Median KKT Ref | Median Box Viol | Median Lin Viol | Median Orth Viol | Median Objective | Wilcoxon Evals p |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `fortchuck` | 10/10 (100%) | 149.5 (17.2) | 149.5 (17.2) | 115.62 (17.51) | `8.07e-07` | `0.00e+00` | `0.00e+00` | `2.64e-14` | `-8.904018` | `—` |
| `fortchuck-ABB` | 10/10 (100%) | 225.0 (95.0) | 225.0 (95.0) | 100.17 (41.85) | `8.44e-07` | `0.00e+00` | `0.00e+00` | `2.77e-14` | `-8.904018` | `1.37e-02` |
| `Alternating-BCD` | 10/10 (100%) | 400.5 (159.0) | 400.5 (159.0) | 397.47 (153.31) | `7.17e-07` | `0.00e+00` | `0.00e+00` | `1.28e-15` | `-8.904018` | `1.95e-03` |
| `pymanopt-Warped-CG` | 0/10 (0%) | 5000.0 (0.0) | 5000.0 (0.0) | 6913.63 (333.99) | `2.98e-03` | `0.00e+00` | `6.29e-05` | `1.64e-15` | `-8.908905` | `1.95e-03` |
| `pymanopt-TrustRegions` | 0/10 (0%) | 5000.0 (0.0) | 5000.0 (0.0) | 5053.51 (106.55) | `7.88e-03` | `0.00e+00` | `4.35e-06` | `1.55e-15` | `-8.909342` | `1.95e-03` |
| `SciPy-L-BFGS-B` | 0/10 (0%) | 4509.5 (2054.5) | 5000.0 (0.0) | 6701.92 (3690.48) | `1.76e-04` | `0.00e+00` | `9.72e-08` | `7.34e-06` | `-8.904045` | `1.95e-03` |
| `SciPy-SLSQP` | SKIP (`N_tot > 450`) | — | — | — | — | — | — | — | — | — |
| `SciPy-trust-constr` | SKIP (`N_tot > 450`) | — | — | — | — | — | — | — | — | — |

### `B8B`: Long-Only Weighted Orthogonal Risk-Factor Parity

- **Regime**: `z in [0,0.15]^24 (a^T z = 1) x St(32,6) x St(6,6) (N_tot=252, Branch A & B)` (`d = 24`, `R = 2`, `N_tot = 252`, intrinsic dim `209`, linear constraint `a^T z = 1.0`, FD gradient max rel err `1.301e-11`)

| Solver | Solve Rate | Raw Evals Med (IQR) | Penalized Evals Med (IQR) | Wall ms Med (IQR) | Median KKT Ref | Median Box Viol | Median Lin Viol | Median Orth Viol | Median Objective | Wilcoxon Evals p |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `fortchuck` | 10/10 (100%) | 225.0 (334.5) | 225.0 (334.5) | 53.03 (91.22) | `5.99e-07` | `0.00e+00` | `2.78e-17` | `2.04e-14` | `-4.990125` | `—` |
| `fortchuck-ABB` | 10/10 (100%) | 60.5 (3.5) | 60.5 (3.5) | 10.89 (3.18) | `6.68e-07` | `0.00e+00` | `5.55e-17` | `1.03e-14` | `-4.990125` | `2.73e-02` |
| `Alternating-BCD` | 10/10 (100%) | 320.5 (118.2) | 320.5 (118.2) | 449.88 (97.11) | `8.51e-07` | `0.00e+00` | `2.22e-16` | `1.07e-15` | `-4.990125` | `1.93e-01` |
| `pymanopt-Warped-CG` | 0/10 (0%) | 5000.0 (0.0) | 5000.0 (0.0) | 8429.40 (919.51) | `8.18e-04` | `0.00e+00` | `5.17e-05` | `9.81e-16` | `-4.990097` | `1.95e-03` |
| `pymanopt-TrustRegions` | 0/10 (0%) | 5000.0 (0.0) | 5000.0 (0.0) | 5964.37 (679.88) | `3.91e-04` | `0.00e+00` | `3.06e-05` | `9.57e-16` | `-4.990137` | `1.95e-03` |
| `SciPy-L-BFGS-B` | 0/10 (0%) | 994.0 (68.2) | 5000.0 (0.0) | 1208.13 (153.75) | `2.38e-05` | `0.00e+00` | `1.26e-07` | `2.19e-06` | `-4.990130` | `1.95e-03` |
| `SciPy-SLSQP` | 10/10 (100%) | 36.0 (5.8) | 36.0 (5.8) | 480.74 (46.18) | `6.70e-07` | `0.00e+00` | `7.77e-16` | `4.72e-12` | `-4.990125` | `1.95e-03` |
| `SciPy-trust-constr` | 0/10 (0%) | 205.5 (14.2) | 5000.0 (0.0) | 3509.94 (362.49) | `2.79e-05` | `0.00e+00` | `5.55e-17` | `1.06e-13` | `-4.990124` | `1.95e-03` |

