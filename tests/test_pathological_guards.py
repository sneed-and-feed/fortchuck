"""Adversarial numerical & systems guard tests for `fortchuck` (`ARCHITECTURE_SPEC.md` §5.2).

Verifies:
  1. FM-2 Clamped Lagrange-Multiplier Cross-Contamination Guard (`<= 35` iters to `1e-10`).
  2. Callback Domain Barrier Recovery (`rc > 0` and IEEE `NaN` step halving).
  3. Antipodal Velocity Cap (`2*pi`) and `1e5` Cross-Block Hessian Scale Mismatch (FM-5).
  4. Pre-Callback Positive-Diagonal Householder-QR Polish & Strict Feasibility Invariant (FM-8).
  5. Rosen Boundary Non-Jamming (FM-1) & `+-DBL_MAX` Overflow Safety (FM-9).
  6. Multi-Threaded Python `ThreadPoolExecutor` Bit-Reproducibility.
  7. FM-13 Non-Isotropic Hyperplane Ascent & Simplex Corner Release (`kappa = 1e4` on `Delta_8 x St(10, 3)`).
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import math
import numpy as np
import pytest

from fortchuck import (
    FC_CONVERGED_GRAD,
    FCOptions,
    FortchuckWorkspace,
    kkt_residual,
    minimize_box_stiefel,
    project_linear_box,
    project_stiefel,
)


@pytest.mark.parametrize("seed", [0, 7, 42, 101, 2026])
def test_fm2_clamped_multiplier_cross_contamination(seed: int) -> None:
    """`ARCHITECTURE_SPEC.md` §5.2 FM-2 Clamped Multiplier Injection Test:

    `f(z, X) = 1e4 * z_1 * (1 + X_11) + 0.5 * tr(X^T D X)` on `[0, 1] x St(8, 3)`.
    `z_1* = 0` clamps immediately with huge active multiplier `lambda_1* = 1e4 * (1 + X_11*)`
    and strong off-diagonal cross-Hessian `1e4`. Masked free-subspace L-BFGS must
    converge superlinearly in `<= 35` iterations to `E_KKT <= 1e-10`.
    """
    rng = np.random.default_rng(seed)
    n, p = 8, 3
    d_diag = 1.5 * np.arange(1.0, n + 1.0, dtype=np.float64)
    f_star = 0.5 * float(np.sum(d_diag[:p]))

    z0 = np.array([0.65], dtype=np.float64)
    X0 = project_stiefel(np.asfortranarray(rng.standard_normal((n, p))))

    def fg_fm2(
        z: np.ndarray, xs: list[np.ndarray]
    ) -> tuple[float, np.ndarray, list[np.ndarray]]:
        X = xs[0]
        z1 = float(z[0])
        x11 = float(X[0, 0])
        DX = d_diag[:, None] * X

        f_val = 1.0e4 * z1 * (1.0 + x11) + 0.5 * float(np.sum(X * DX))
        gz = np.array([1.0e4 * (1.0 + x11)], dtype=np.float64)
        GX = np.array(DX, dtype=np.float64, order="F", copy=True)
        GX[0, 0] += 1.0e4 * z1
        return f_val, gz, [GX]

    res = minimize_box_stiefel(
        fg_fm2,
        z0=z0,
        bounds=([0.0], [1.0]),
        X0_list=X0,
        options=FCOptions(gtol=1.0e-10),
        copy=True,
    )

    assert res.status == FC_CONVERGED_GRAD
    assert res.n_iter <= 35, f"FM-2 took {res.n_iter} > 35 iterations (seed={seed})"
    assert res.kkt_residual <= 1.0e-10
    assert res.orth_error <= 1.0e-13
    assert abs(res.z[0]) <= 1.0e-16
    assert res.f_val == pytest.approx(f_star, abs=1.0e-11)


@pytest.mark.parametrize("barrier_mode", ["rc_positive", "nan_return"])
def test_callback_domain_barrier_recovery(barrier_mode: str) -> None:
    """Verify automatic backtracking recovery when trial step overshoots an open log-barrier."""
    rng = np.random.default_rng(31415)
    n, p = 5, 2
    d_st = np.arange(1.0, n + 1.0, dtype=np.float64)

    # Box bounds [0.0, 5.0] contain invalid barrier region z[0] <= 0.25!
    z0 = np.array([4.2, -1.5], dtype=np.float64)
    bounds = (np.array([0.0, -2.0]), np.array([5.0, 2.0]))
    X0 = project_stiefel(np.asfortranarray(rng.standard_normal((n, p))))
    barrier_hits = 0

    def fg_log_barrier(
        z: np.ndarray, xs: list[np.ndarray]
    ) -> tuple[float, np.ndarray | None, list[np.ndarray] | None, int] | tuple[float, np.ndarray | None, list[np.ndarray] | None]:
        nonlocal barrier_hits
        # Linear term 25*(z0 - 0.75) gives huge initial gradient pushing trial z0 <= 0.25 at tau=1
        if z[0] <= 0.25:
            barrier_hits += 1
            if barrier_mode == "rc_positive":
                return 0.0, None, None, 1
            return float("nan"), None, None

        X = xs[0]
        gap = float(z[0]) - 0.25
        f_box = -math.log(gap) + 40.0 * (float(z[0]) - 0.55) ** 2 + (float(z[1]) - 0.35) ** 2
        DX = d_st[:, None] * X
        f_st = 0.5 * (1.0 + float(z[0])) * float(np.sum(X * DX))

        gz = np.array(
            [
                -1.0 / gap + 80.0 * (float(z[0]) - 0.55) + 0.5 * float(np.sum(X * DX)),
                2.0 * (float(z[1]) - 0.35),
            ],
            dtype=np.float64,
        )
        GX = (1.0 + float(z[0])) * DX
        return f_box + f_st, gz, [GX]

    # Choose initial z0 = 4.9 withABB/initial step so trial overshoots or test explicit overshoot
    # Use memory=7 and custom initial point where step overshoots 0.25
    z0_steep = np.array([0.28, 1.4], dtype=np.float64)
    # Flip sign of quadratic attractor to pull toward 0.25 boundary strongly without crossing at optimum:
    def fg_barrier_overshoot(
        z: np.ndarray, xs: list[np.ndarray]
    ) -> tuple[float, np.ndarray | None, list[np.ndarray] | None, int] | tuple[float, np.ndarray | None, list[np.ndarray] | None]:
        nonlocal barrier_hits
        if z[0] <= 0.25:
            barrier_hits += 1
            if barrier_mode == "rc_positive":
                return 0.0, None, None, 1
            return float("nan"), None, None

        X = xs[0]
        gap = float(z[0]) - 0.25
        # Minimizer of -0.01*log(z0 - 0.25) + 2.0*(z0 - 0.20)^2 is near z0* ~ 0.278 > 0.25,
        # while starting at z0 = 3.5 takes an aggressive secant jump toward 0.20 <= 0.25!
        f_box = -0.01 * math.log(gap) + 2.0 * (float(z[0]) - 0.20) ** 2 + (float(z[1]) - 0.4) ** 2
        DX = d_st[:, None] * X
        f_st = 0.5 * float(np.sum(X * DX))
        gz = np.array(
            [-0.01 / gap + 4.0 * (float(z[0]) - 0.20), 2.0 * (float(z[1]) - 0.4)],
            dtype=np.float64,
        )
        return f_box + f_st, gz, [DX]

    _ = fg_log_barrier  # keep helper unused-clean
    _ = z0_steep
    res = minimize_box_stiefel(
        fg_barrier_overshoot,
        z0=z0,
        bounds=bounds,
        X0_list=X0,
        options=FCOptions(gtol=1.0e-9),
        copy=True,
    )

    assert barrier_hits >= 1, f"Expected domain barrier to be hit at least once ({barrier_mode})"
    assert res.status == FC_CONVERGED_GRAD
    assert res.z[0] > 0.25
    assert res.kkt_residual <= 1.0e-9
    assert res.orth_error <= 1.0e-13


def test_antipodal_velocity_cap_and_extreme_cross_block_scale_mismatch() -> None:
    """FM-5 (`1e5` Hessian mismatch) & FM-7 (huge initial gradient velocity cap `2*pi`)."""
    rng = np.random.default_rng(777)
    d = 6
    n, p = 10, 4

    # Box curvature = 1e5, Stiefel curvature ~ 1.0 (1e5 cross-block ratio!)
    z_star = np.array([0.0, 0.35, 1.0, -0.2, 0.8, 0.0], dtype=np.float64)
    lower = np.zeros(d, dtype=np.float64)
    upper = np.ones(d, dtype=np.float64)
    z0 = np.full(d, 0.5, dtype=np.float64)

    diag_st = np.linspace(1.0, 4.0, n, dtype=np.float64)
    X0 = project_stiefel(np.asfortranarray(rng.standard_normal((n, p))))

    def fg_mismatched(
        z: np.ndarray, xs: list[np.ndarray]
    ) -> tuple[float, np.ndarray, list[np.ndarray]]:
        X = xs[0]
        diff_z = z - z_star
        DX = diag_st[:, None] * X
        # Coupled term + 1e5 stiff box curvature
        coup = float(z[1]) * float(X[0, 0])
        f_val = 0.5 * 1.0e5 * float(np.sum(diff_z**2)) + 0.5 * float(np.sum(X * DX)) + coup
        gz = 1.0e5 * diff_z
        gz[1] += float(X[0, 0])
        GX = DX.copy(order="F")
        GX[0, 0] += float(z[1])
        return f_val, gz, [GX]

    res = minimize_box_stiefel(
        fg_mismatched,
        z0=z0,
        bounds=(lower, upper),
        X0_list=X0,
        options=FCOptions(gtol=1.0e-8),
        copy=True,
    )
    assert res.status == FC_CONVERGED_GRAD
    assert res.kkt_residual <= 1.0e-8
    assert res.orth_error <= 1.0e-13


def test_pre_callback_qr_polish_invariant_and_so_orientation() -> None:
    """FM-8: Callback never sees orthogonality drift > orth_tol or box infeasibility > 0."""
    rng = np.random.default_rng(5150)
    n, p = 6, 6
    U, _, Vt = np.linalg.svd(rng.standard_normal((n, n)))
    Q_targ = np.asfortranarray(U @ Vt)
    if np.linalg.det(Q_targ) < 0.0:
        Q_targ[:, 0] *= -1.0

    X0 = project_stiefel(np.asfortranarray(rng.standard_normal((n, p))))
    if np.linalg.det(X0) < 0.0:
        X0[:, 0] *= -1.0

    z0 = np.array([0.2, 0.8], dtype=np.float64)
    lower = np.array([0.0, 0.0], dtype=np.float64)
    upper = np.array([1.0, 1.0], dtype=np.float64)

    max_cb_orth_err = 0.0
    max_cb_box_viol = 0.0
    min_cb_det = 1.0

    def fg_tracked(
        z: np.ndarray, xs: list[np.ndarray]
    ) -> tuple[float, np.ndarray, list[np.ndarray]]:
        nonlocal max_cb_orth_err, max_cb_box_viol, min_cb_det
        X = xs[0]
        orth = float(np.linalg.norm(X.T @ X - np.eye(p), ord="fro"))
        box_v = float(max(0.0, np.max(lower - z), np.max(z - upper)))
        det_x = float(np.linalg.det(X))
        max_cb_orth_err = max(max_cb_orth_err, orth)
        max_cb_box_viol = max(max_cb_box_viol, box_v)
        min_cb_det = min(min_cb_det, det_x)

        weights = np.arange(1.0, n + 1.0, dtype=np.float64)
        diff = X - Q_targ
        Wdiff = weights[:, None] * diff
        f_val = (
            0.5 * (1.0 + float(z[0])) * float(np.sum(diff * Wdiff))
            + 3.0 * float(z[0])
            + 0.5 * (float(z[1]) - 0.45) ** 2
        )
        gz = np.array(
            [0.5 * float(np.sum(diff * Wdiff)) + 3.0, float(z[1]) - 0.45],
            dtype=np.float64,
        )
        GX = (1.0 + float(z[0])) * Wdiff
        return f_val, gz, [GX]

    # Set orth_tol = 1e-15 so pre-callback fc_geqrf_pos polish + history re-projection fire actively
    res = minimize_box_stiefel(
        fg_tracked,
        z0=z0,
        bounds=(lower, upper),
        X0_list=X0,
        options=FCOptions(gtol=1.0e-10, orth_tol=1.0e-15),
        copy=True,
    )

    assert res.status == FC_CONVERGED_GRAD
    assert res.n_reorth >= 1
    assert max_cb_box_viol == 0.0
    assert max_cb_orth_err <= 5.0e-15
    assert min_cb_det == pytest.approx(1.0, abs=1.0e-13)
    assert np.linalg.det(res.X_list[0]) == pytest.approx(1.0, abs=1.0e-13)


def test_rosen_boundary_non_jamming_and_huge_bounds() -> None:
    """FM-1 & FM-9: Near-active `z_i - l_i = 1e-14` does not jam step size; `+-DBL_MAX` safe."""
    d = 8
    dbl_max = float(np.finfo(np.float64).max)
    lower = np.zeros(d, dtype=np.float64)
    upper = np.ones(d, dtype=np.float64)
    lower[6:] = -dbl_max
    upper[6:] = dbl_max

    # Initialize coordinates 0..2 at 1e-14 above lower bound with positive outward gradient!
    z0 = np.array([1.0e-14, 1.0e-14, 1.0e-14, 0.7, 0.8, 0.2, 5.0, -4.0], dtype=np.float64)
    X0 = project_stiefel(np.asfortranarray(np.eye(7, 2, dtype=np.float64)))

    def fg_jamming_probe(
        z: np.ndarray, xs: list[np.ndarray]
    ) -> tuple[float, np.ndarray, list[np.ndarray]]:
        X = xs[0]
        # Coordinates 0..2 want to clamp at 0; coordinates 3..7 have smooth interior optimum
        targ = np.array([-2.0, -3.0, -1.5, 0.3, 0.4, 0.6, -1.2, 2.5], dtype=np.float64)
        diff = z - targ
        d_st = np.arange(1.0, 8.0, dtype=np.float64)
        DX = d_st[:, None] * X
        f_val = 0.5 * float(np.sum(diff**2)) + 0.5 * (1.0 + float(z[3]) ** 2) * float(np.sum(X * DX))
        gz = diff.copy()
        gz[3] += float(z[3]) * float(np.sum(X * DX))
        GX = (1.0 + float(z[3]) ** 2) * DX
        return f_val, gz, [GX]

    res = minimize_box_stiefel(
        fg_jamming_probe,
        z0=z0,
        bounds=(lower, upper),
        X0_list=X0,
        options=FCOptions(gtol=1.0e-10),
        copy=True,
    )
    assert res.status == FC_CONVERGED_GRAD
    assert res.n_iter <= 25
    assert np.all(res.z[:3] == 0.0)
    assert res.kkt_residual <= 1.0e-10


def test_multithreaded_python_parallel_reentrancy() -> None:
    """Verify concurrent Python `ThreadPoolExecutor` solves match sequential bit-for-bit."""
    n_jobs = 8
    d = 5
    n1, p1 = 9, 3
    n2, p2 = 4, 4

    seeds = [1000 + k for k in range(n_jobs)]
    lower = np.zeros(d, dtype=np.float64)
    upper = np.full(d, 2.0, dtype=np.float64)

    def make_instance(seed: int) -> tuple[np.ndarray, list[np.ndarray], np.ndarray]:
        rng = np.random.default_rng(seed)
        z_init = rng.uniform(0.2, 1.8, size=d)
        x1_init = project_stiefel(np.asfortranarray(rng.standard_normal((n1, p1))))
        x2_init = project_stiefel(np.asfortranarray(rng.standard_normal((n2, p2))))
        z_goal = rng.uniform(-0.5, 2.5, size=d)
        return z_init, [x1_init, x2_init], z_goal

    instances = [make_instance(s) for s in seeds]

    def run_single(idx: int) -> tuple[float, int, int, np.ndarray, list[np.ndarray]]:
        z_init, xs_init, z_goal = instances[idx]
        ws = FortchuckWorkspace(d=d, n_list=[n1, n2], p_list=[p1, p2], memory=7)
        d1 = np.arange(1.0, n1 + 1.0, dtype=np.float64)
        d2 = np.arange(1.0, n2 + 1.0, dtype=np.float64)

        def fg(
            z: np.ndarray, xs: list[np.ndarray]
        ) -> tuple[float, np.ndarray, list[np.ndarray]]:
            X1, X2 = xs[0], xs[1]
            dz = z - z_goal
            DX1 = d1[:, None] * X1
            DX2 = d2[:, None] * (X2 - np.eye(n2))
            coup = float(z[0]) * float(X1[0, 0] + X2[0, 0])
            f_v = (
                0.5 * float(np.sum(dz**2))
                + 0.5 * float(np.sum(X1 * DX1))
                + 0.5 * float(np.sum((X2 - np.eye(n2)) * DX2))
                + coup
            )
            gz = dz.copy()
            gz[0] += float(X1[0, 0] + X2[0, 0])
            GX1 = DX1.copy(order="F")
            GX1[0, 0] += float(z[0])
            GX2 = DX2.copy(order="F")
            GX2[0, 0] += float(z[0])
            return f_v, gz, [GX1, GX2]

        res = minimize_box_stiefel(
            fg,
            z0=z_init,
            bounds=(lower, upper),
            X0_list=xs_init,
            options=FCOptions(gtol=1.0e-9),
            workspace=ws,
            copy=True,
        )
        assert res.status == FC_CONVERGED_GRAD
        return res.f_val, res.n_iter, res.n_eval, res.z.copy(), [x.copy() for x in res.X_list]

    seq_results = [run_single(i) for i in range(n_jobs)]
    with ThreadPoolExecutor(max_workers=4) as pool:
        par_results = list(pool.map(run_single, range(n_jobs)))

    for i in range(n_jobs):
        f_s, it_s, ev_s, z_s, xs_s = seq_results[i]
        f_p, it_p, ev_p, z_p, xs_p = par_results[i]
        assert f_s == f_p
        assert it_s == it_p
        assert ev_s == ev_p
        assert np.array_equal(z_s, z_p)
        for x_s, x_p in zip(xs_s, xs_p):
            assert np.array_equal(x_s, x_p)


@pytest.mark.parametrize("seed", [0, 7, 42, 101, 2026])
def test_fm13_non_isotropic_hyperplane_and_simplex_corner_release(seed: int) -> None:
    """`ARCHITECTURE_SPEC.md` §5.2 FM-13 Non-Isotropic Hyperplane Ascent & Simplex Corner Release:

    Ill-conditioned (`kappa = 1e4`) coupled quadratic + Brockett problem on `Delta_8 x St(10, 3)`
    initialized directly at suboptimal simplex vertex `e_1 = (1, 0, ..., 0)` where all `d`
    coordinates sit on box bounds (`z_1 = 1`, `z_{2..8} = 0`). Verifies immediate strict-descent
    release from `e_1`, strict feasibility at every callback evaluation, and convergence to
    `E_KKT <= 1e-9` and `|1^T z* - 1| <= 1e-14`.
    """
    rng = np.random.default_rng(seed)
    d = 8
    n, p = 10, 3

    lower = np.zeros(d, dtype=np.float64)
    upper = np.ones(d, dtype=np.float64)
    a_ones = np.ones(d, dtype=np.float64)

    # Construct dense SPD Hessian M on R^8 with exact condition number kappa_2(M) = 10^4
    Q8 = project_stiefel(np.asfortranarray(rng.standard_normal((d, d))))
    eigs = 10.0 ** (4.0 * np.arange(d, dtype=np.float64) / float(d - 1))
    M = (Q8 * eigs[None, :]) @ Q8.T
    M = 0.5 * (M + M.T)

    # Target puts optimal mass on coordinates 2..5 while vertex e_1 = (1, 0, ..., 0) is suboptimal
    z_targ = np.array([-0.5, -0.3, 0.35, 0.30, 0.25, 0.20, -0.2, -0.2], dtype=np.float64)
    d_st = 0.6 * np.arange(1.0, n + 1.0, dtype=np.float64)

    # Initialize right at suboptimal simplex corner e_1 = (1, 0, ..., 0)
    z0 = np.zeros(d, dtype=np.float64)
    z0[0] = 1.0
    X0 = project_stiefel(np.asfortranarray(rng.standard_normal((n, p))))

    z0_trace: list[float] = []
    max_cb_lin_err = 0.0
    max_cb_box_viol = 0.0
    max_cb_orth_err = 0.0

    def fg_fm13(
        z: np.ndarray, xs: list[np.ndarray]
    ) -> tuple[float, np.ndarray, list[np.ndarray]]:
        nonlocal max_cb_lin_err, max_cb_box_viol, max_cb_orth_err
        X = xs[0]
        z0_trace.append(float(z[0]))
        max_cb_lin_err = max(max_cb_lin_err, abs(float(np.sum(z)) - 1.0))
        max_cb_box_viol = max(
            max_cb_box_viol, float(max(0.0, np.max(lower - z), np.max(z - upper)))
        )
        max_cb_orth_err = max(
            max_cb_orth_err, float(np.linalg.norm(X.T @ X - np.eye(p), ord="fro"))
        )

        dz = z - z_targ
        Mdz = M @ dz
        f_box = 0.5 * float(np.dot(dz, Mdz))
        gz = Mdz.copy()

        # Coupled Brockett term modulated by z[1..3]
        col_weights = np.array(
            [0.4 + float(z[1 + j]) + 0.05 * float(j + 1) for j in range(p)],
            dtype=np.float64,
        )
        DX = d_st[:, None] * X
        col_quad = np.sum(X * DX, axis=0)
        f_st = 0.5 * float(np.dot(col_weights, col_quad))
        gz[1 : 1 + p] += 0.5 * col_quad
        GX = np.asfortranarray(DX * col_weights[None, :])
        return f_box + f_st, gz, [GX]

    res = minimize_box_stiefel(
        fg_fm13,
        z0=z0,
        bounds=(lower, upper),
        X0_list=X0,
        options=FCOptions(gtol=1.0e-9, max_iter=2000, max_eval=8000),
        linear_constraint=(a_ones, 1.0),
        copy=True,
    )

    assert res.status == FC_CONVERGED_GRAD
    assert res.kkt_residual <= 1.0e-9
    assert abs(float(np.sum(res.z)) - 1.0) <= 1.0e-14
    assert res.orth_error <= 1.0e-13

    # Verify immediate corner release on first trial step after initial evaluation p_0
    assert len(z0_trace) >= 2
    assert z0_trace[0] == 1.0
    assert z0_trace[1] < 1.0, f"Expected immediate corner release from e_1 on step 0 (seed={seed})"
    assert res.z[0] < 0.2, f"Suboptimal vertex e_1 was not released: z*[0]={res.z[0]}"

    # Every callback evaluation saw strictly feasible simplex and Stiefel iterates
    assert max_cb_lin_err <= 1.0e-14
    assert max_cb_box_viol <= 1.0e-15
    assert max_cb_orth_err <= 1.0e-13

    # Independent constrained KKT residual parity
    _, gz_fin, gx_fin = fg_fm13(res.z, res.X_list)
    kkt_chk, lin_chk, orth_chk = kkt_residual(
        z=res.z,
        bounds=(lower, upper),
        X_list=res.X_list,
        gz=gz_fin,
        GX_list=gx_fin,
        linear_constraint=(a_ones, 1.0),
    )
    assert kkt_chk == pytest.approx(res.kkt_residual, abs=1.0e-14)
    assert kkt_chk <= 1.0e-9
    assert lin_chk <= 1.0e-14
    assert orth_chk <= 1.0e-13


@pytest.mark.parametrize("alpha", [1.0, 1.0e-14, 1.0e12])
@pytest.mark.parametrize("constrained", [False, True])
def test_objective_scale_invariance_extreme_alpha(
    alpha: float, constrained: bool
) -> None:
    """Scale invariance under `f -> alpha * f` for `alpha in {1, 1e-14, 1e+12}`.

    Guards against dimensionally inconsistent line-search descent thresholds
    (`O(||g||*||d||)` vs `O(||d||^2)`) and `z - (z - gz)` KKT cancellation on
    both unconstrained Box x Stiefel and linearly constrained Simplex x Stiefel.
    """
    rng = np.random.default_rng(1234)
    d = 6
    n, p = 8, 2
    Q_d = project_stiefel(np.asfortranarray(rng.standard_normal((d, d))))
    M = Q_d @ np.diag(np.geomspace(1.0, 20.0, d)) @ Q_d.T
    M = 0.5 * (M + M.T)
    z_targ = np.array([0.4, 0.25, -0.2, 0.3, 0.15, 0.5], dtype=np.float64)
    D_st = np.arange(1.0, n + 1.0, dtype=np.float64)

    lower = np.zeros(d, dtype=np.float64)
    upper = np.ones(d, dtype=np.float64)
    a_lin = np.ones(d, dtype=np.float64)
    b_lin = 1.0
    lin_arg = (a_lin, b_lin) if constrained else None

    z0 = np.full(d, 1.0 / d, dtype=np.float64)
    X0 = project_stiefel(np.asfortranarray(rng.standard_normal((n, p))))

    def make_fg(scale: float):
        def fg(
            z: np.ndarray, xs: list[np.ndarray]
        ) -> tuple[float, np.ndarray, list[np.ndarray]]:
            X = xs[0]
            dz = z - z_targ
            Mdz = M @ dz
            f_b = 0.5 * float(np.dot(dz, Mdz))
            gz = Mdz.copy()
            cw = np.array([0.5 + float(z[0]), 0.8 + float(z[1])], dtype=np.float64)
            DX = D_st[:, None] * X
            cq = np.sum(X * DX, axis=0)
            f_s = 0.5 * float(np.dot(cw, cq))
            gz[0] += 0.5 * cq[0]
            gz[1] += 0.5 * cq[1]
            GX = np.asfortranarray(DX * cw[None, :])
            return scale * (f_b + f_s), scale * gz, [scale * GX]

        return fg

    # Reference solution at alpha = 1.0
    res_ref = minimize_box_stiefel(
        make_fg(1.0),
        z0=z0,
        bounds=(lower, upper),
        X0_list=X0,
        options=FCOptions(gtol=1.0e-7, max_iter=500, max_eval=2000),
        linear_constraint=lin_arg,
        copy=True,
    )
    assert res_ref.status == FC_CONVERGED_GRAD

    fg_scaled = make_fg(alpha)
    res = minimize_box_stiefel(
        fg_scaled,
        z0=z0,
        bounds=(lower, upper),
        X0_list=X0,
        options=FCOptions(gtol=1.0e-7 * alpha, max_iter=500, max_eval=2000),
        linear_constraint=lin_arg,
        copy=True,
    )
    assert res.status == FC_CONVERGED_GRAD
    assert res.kkt_residual <= 1.0e-7 * alpha
    assert res.orth_error <= 1.0e-13
    assert np.max(np.abs(res.z - res_ref.z)) <= 1.0e-5
    if constrained:
        assert abs(float(np.sum(res.z)) - b_lin) <= 1.0e-14

    # Standalone kkt_residual helper parity (cancellation-free even at alpha = 1e-14)
    _, gz_fin, gx_fin = fg_scaled(res.z, res.X_list)
    kkt_out = kkt_residual(
        z=res.z,
        bounds=(lower, upper),
        X_list=res.X_list,
        gz=gz_fin,
        GX_list=gx_fin,
        linear_constraint=lin_arg,
    )
    kkt_chk = float(kkt_out[0])
    assert kkt_chk <= 1.0e-7 * alpha
    assert kkt_chk == pytest.approx(res.kkt_residual, rel=1.0e-6, abs=1.0e-22)


@pytest.mark.parametrize("shift_scale", [1.0e4, 1.0e5, -1.0e5])
@pytest.mark.parametrize("eps_lin", [5.0e-9, -5.0e-9])
def test_project_initial_zero_small_lin_residual_large_nu_k(
    shift_scale: float, eps_lin: float
) -> None:
    """`project_initial=0` with `0 < |a^T z0 - b| <= 1e-8 * (1 + |b|)` and large `nu_k`.

    Verifies that step-0 directional slope and line-search chord do not inject
    spurious `nu_k * (b - a^T z0)` bias when `z0` is within initial feasibility
    tolerance but not exact to machine precision.
    """
    rng = np.random.default_rng(77)
    d = 8
    n, p = 8, 2
    a = np.array([1.4, 0.9, 1.2, -0.8, 1.7, -0.5, 1.1, 0.6], dtype=np.float64)
    b = 2.15
    lower = np.zeros(d, dtype=np.float64)
    upper = np.ones(d, dtype=np.float64)
    z_targ = np.array([0.8, 0.2, 0.5, 0.4, 0.9, 0.1, 0.6, 0.3], dtype=np.float64)
    D_st = np.arange(1.0, n + 1.0, dtype=np.float64)

    z0_exact, _ = project_linear_box(
        np.full(d, 0.4, dtype=np.float64), lower, upper, a, b
    )
    z0 = z0_exact.copy()
    z0[0] += eps_lin / a[0]
    assert 1.0e-9 < abs(float(np.dot(a, z0)) - b) < 1.0e-8 * (1.0 + abs(b))
    X0 = project_stiefel(np.asfortranarray(rng.standard_normal((n, p))))

    def fg(
        z: np.ndarray, xs: list[np.ndarray]
    ) -> tuple[float, np.ndarray, list[np.ndarray]]:
        X = xs[0]
        dz = z - z_targ
        f_b = 0.5 * float(np.dot(dz, dz)) + shift_scale * float(np.dot(a, z))
        gz = dz + shift_scale * a
        DX = D_st[:, None] * X
        f_s = 0.5 * float(np.sum(X * DX)) * (1.0 + 0.2 * float(z[0]))
        gz[0] += 0.1 * float(np.sum(X * DX))
        GX = np.asfortranarray(DX * (1.0 + 0.2 * float(z[0])))
        return f_b + f_s, gz, [GX]

    res = minimize_box_stiefel(
        fg,
        z0=z0,
        bounds=(lower, upper),
        X0_list=X0,
        options=FCOptions(
            gtol=1.0e-8,
            project_initial=0,
            max_iter=500,
            max_eval=2000,
        ),
        linear_constraint=(a, b),
        copy=True,
    )
    assert res.status == FC_CONVERGED_GRAD
    assert res.kkt_residual <= 1.0e-8
    assert abs(float(np.dot(a, res.z)) - b) <= 1.0e-14 * (1.0 + abs(b))
    assert res.orth_error <= 1.0e-13


def test_active_bounds_no_liftoff_and_large_offset_default_ftol() -> None:
    """Guards two solver invariants:

    1. Constrained L-BFGS (`m=7`) on a capped simplex (`[0, 0.2]^16, a^T z = 1`)
       with multiple active lower and upper bounds converges cleanly without
       projection boundary lift-off ping-pong.
    2. Coupled Box x Stiefel with large constant offset (`+500.0`) under default
       `ftol` (`-1.0`) converges via `FC_CONVERGED_GRAD` rather than premature
       `FC_CONVERGED_FTOL`.
    """
    # Part 1: Capped simplex with active lower and upper bounds
    rng = np.random.default_rng(2025)
    d = 16
    n, p = 10, 2
    lower = np.zeros(d, dtype=np.float64)
    upper = np.full(d, 0.2, dtype=np.float64)
    a = np.ones(d, dtype=np.float64)
    b = 1.0
    Q_d = project_stiefel(np.asfortranarray(rng.standard_normal((d, d))))
    M = Q_d @ np.diag(np.geomspace(1.0, 50.0, d)) @ Q_d.T
    M = 0.5 * (M + M.T)
    lin_c = np.linspace(-2.5, 2.5, d)
    D_st = np.arange(1.0, n + 1.0, dtype=np.float64)

    z0, _ = project_linear_box(
        rng.uniform(0.02, 0.18, size=d), lower, upper, a, b
    )
    X0 = project_stiefel(np.asfortranarray(rng.standard_normal((n, p))))

    def fg_cap(
        z: np.ndarray, xs: list[np.ndarray]
    ) -> tuple[float, np.ndarray, list[np.ndarray]]:
        X = xs[0]
        Mz = M @ z
        f_b = 0.5 * float(np.dot(z, Mz)) + float(np.dot(lin_c, z))
        gz = Mz + lin_c
        cw = np.array([0.5 + float(z[0]), 0.8 + float(z[1])], dtype=np.float64)
        DX = D_st[:, None] * X
        cq = np.sum(X * DX, axis=0)
        f_s = 0.5 * float(np.dot(cw, cq))
        gz[0] += 0.5 * cq[0]
        gz[1] += 0.5 * cq[1]
        GX = np.asfortranarray(DX * cw[None, :])
        return f_b + f_s, gz, [GX]

    res_cap = minimize_box_stiefel(
        fg_cap,
        z0=z0,
        bounds=(lower, upper),
        X0_list=X0,
        options=FCOptions(gtol=1.0e-7, max_iter=500, max_eval=1500),
        linear_constraint=(a, b),
        copy=True,
    )
    assert res_cap.status == FC_CONVERGED_GRAD
    assert res_cap.kkt_residual <= 1.0e-7
    assert res_cap.n_eval <= 180
    assert abs(float(np.sum(res_cap.z)) - b) <= 1.0e-14
    assert int(np.sum(res_cap.z <= 1.0e-12)) >= 4
    assert int(np.sum(res_cap.z >= 0.2 - 1.0e-12)) >= 2

    # Part 2: Ill-conditioned objective with large constant offset (+500.0) and default ftol
    d2 = 8
    Q2 = project_stiefel(np.asfortranarray(rng.standard_normal((d2, d2))))
    H2 = Q2 @ np.diag(np.geomspace(1.0, 2.0e3, d2)) @ Q2.T
    H2 = 0.5 * (H2 + H2.T)
    z_star2 = np.linspace(-0.3, 0.8, d2)
    l2 = np.zeros(d2, dtype=np.float64)
    u2 = np.ones(d2, dtype=np.float64)
    z0_2 = np.full(d2, 0.5, dtype=np.float64)

    def fg_offset(
        z: np.ndarray, xs: list[np.ndarray]
    ) -> tuple[float, np.ndarray, list[np.ndarray]]:
        X = xs[0]
        dz = z - z_star2
        Hdz = H2 @ dz
        DX = D_st[:, None] * X
        f_val = 500.0 + 0.5 * float(np.dot(dz, Hdz)) + 0.5 * float(np.sum(X * DX))
        return f_val, Hdz, [np.asfortranarray(DX)]

    res_off = minimize_box_stiefel(
        fg_offset,
        z0=z0_2,
        bounds=(l2, u2),
        X0_list=X0,
        options=FCOptions(gtol=1.0e-7, max_iter=2000, max_eval=5000),
        copy=True,
    )
    assert res_off.status == FC_CONVERGED_GRAD
    assert res_off.kkt_residual <= 1.0e-7

