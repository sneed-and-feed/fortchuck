"""Comprehensive C99 ABI, binary struct layout, zero-copy FFI, and status-code tests.

Covers `ARCHITECTURE_SPEC.md` §4.1, §4.2, and §5.2 (`test_c_abi_and_python.py`).
"""

from __future__ import annotations

import ctypes
import math
import sys
import numpy as np
import pytest

import fortchuck as fc
from fortchuck import (
    FC_CONVERGED_FTOL,
    FC_CONVERGED_GRAD,
    FC_CONVERGED_XTOL,
    FC_DAMP_CAUTIOUS,
    FC_DAMP_POWELL,
    FC_ERR_CALLBACK_ABORT,
    FC_ERR_CALLBACK_NAN,
    FC_ERR_INFEASIBLE_LIN,
    FC_ERR_INVALID_BOUNDS,
    FC_ERR_INVALID_DIM,
    FC_ERR_INVALID_OPTS,
    FC_ERR_INVALID_X0,
    FC_ERR_LS_FAILED,
    FC_ERR_NULL_PTR,
    FC_ERR_WORK_TOO_SMALL,
    FC_MAX_EVAL_REACHED,
    FC_MAX_ITER_REACHED,
    FC_METRIC_CANONICAL,
    FC_METRIC_EUCLIDEAN,
    FC_SUCCESS_QUERY,
    FCEvalCallback,
    FCOptions,
    FCResult,
    FortchuckWorkspace,
    default_options,
    kkt_residual,
    load_native_library,
    minimize_box_stiefel,
    project_linear_box,
    project_stiefel,
    workspace_query,
)

C_I32_P = ctypes.POINTER(ctypes.c_int32)
C_I64_P = ctypes.POINTER(ctypes.c_int64)
C_F64_P = ctypes.POINTER(ctypes.c_double)


def _exact_spec_workspace_sizes(
    d: int, n_list: list[int], p_list: list[int], m: int
) -> tuple[int, int]:
    """Reference unified v0.2.0 formula from `ARCHITECTURE_SPEC.md` §3.4."""
    R = len(n_list)
    d_pad = max(d, 1)
    n_st = sum(n * p for n, p in zip(n_list, p_list))
    n_st_pad = max(n_st, 1)
    n_tot_pad = max(d + n_st, 1)
    n_skew_pad = max(1, sum(p * p for p in p_list))
    m_pad = max(m, 1)
    k_max = 2 * m + 2

    c_sum = 0
    q_sum = 0
    scr_max = 64
    for n, p in zip(n_list, p_list):
        scr_base = 2 * n * p + 6 * p * p + 4 * p
        if 2 * p < n:
            c_sum += 2 * n * p + 3 * p * p
            q_sum += p
            scr_batch = (n + 2 * p) * p * k_max + p * p
        else:
            c_sum += 2 * n * n
            q_sum += n
            scr_batch = 2 * n * p * k_max
        scr_max = max(scr_max, scr_base, scr_batch)

    n_cay = max(1, c_sum)
    n_piv = max(1, q_sum)
    n_scr = scr_max

    lwork = (
        7 * d_pad
        + 6 * n_st_pad
        + 4 * n_tot_pad
        + n_skew_pad
        + 4 * (R + 1)
        + 2 * m * n_tot_pad
        + 2 * m * n_skew_pad
        + 5 * m_pad
        + n_cay
        + n_scr
    )
    liwork = d_pad + 2 * m_pad + 4 * (R + 1) + n_piv
    return lwork, liwork


def test_struct_binary_layout_and_default_options_parity() -> None:
    assert fc.__version__ == "0.2.0"
    assert FC_ERR_INFEASIBLE_LIN == -10
    assert fc.STATUS_NAMES[FC_ERR_INFEASIBLE_LIN] == "FC_ERR_INFEASIBLE_LIN"
    assert ctypes.sizeof(FCOptions) == 96
    assert ctypes.sizeof(FCResult) == 88
    assert fc.is_native_available() is True

    opts_py = FCOptions()
    opts_c = default_options()
    for field_name, _ in FCOptions._fields_:
        v_py = getattr(opts_py, field_name)
        v_c = getattr(opts_c, field_name)
        if isinstance(v_py, float):
            assert math.isclose(v_py, v_c, rel_tol=0.0, abs_tol=1e-16)
        else:
            assert v_py == v_c

    opts_override = FCOptions(gtol=1.0e-8, max_eval=5000)
    assert opts_override.gtol == pytest.approx(1.0e-8)
    assert opts_override.max_eval == 5000
    assert opts_override.memory == 7
    assert opts_override.c1_armijo == pytest.approx(1.0e-4)
    assert opts_override.max_step_vel == pytest.approx(2.0 * math.pi)
    assert opts_override.project_initial == 1

    with pytest.raises(TypeError, match="unexpected keyword argument"):
        FCOptions(invalid_option_name=123)


@pytest.mark.parametrize(
    ("d", "n_list", "p_list"),
    [
        (25, [], []),
        (0, [20], [3]),
        (0, [8], [6]),
        (12, [15, 6], [2, 6]),
    ],
)
@pytest.mark.parametrize("memory", [0, 1, 7, 15])
def test_workspace_query_and_lwork_minus_one_parity(
    d: int, n_list: list[int], p_list: list[int], memory: int
) -> None:
    lib = load_native_library()
    lw_q, liw_q = workspace_query(d=d, n_list=n_list, p_list=p_list, memory=memory)
    exp_lw, exp_liw = _exact_spec_workspace_sizes(d, n_list, p_list, memory)
    assert lw_q == exp_lw
    assert liw_q == exp_liw

    R = len(n_list)
    n_arr = np.array(n_list, dtype=np.int32)
    p_arr = np.array(p_list, dtype=np.int32)
    n_ptr = n_arr.ctypes.data_as(C_I32_P) if R > 0 else None
    p_ptr = p_arr.ctypes.data_as(C_I32_P) if R > 0 else None

    opts = FCOptions(memory=memory)
    work_probe = np.zeros(1, dtype=np.float64)
    iwork_probe = np.zeros(1, dtype=np.int32)
    res = FCResult()

    # Query via lwork = -1 with NULL iterate/callback pointers
    st = lib.fortchuck_solve(
        d,
        R,
        n_ptr,
        p_ptr,
        None,
        None,
        None,
        None,
        FCEvalCallback(0),
        None,
        ctypes.byref(opts),
        work_probe.ctypes.data_as(C_F64_P),
        -1,
        iwork_probe.ctypes.data_as(C_I32_P),
        -1,
        ctypes.byref(res),
    )
    assert st == FC_SUCCESS_QUERY
    assert res.status == FC_SUCCESS_QUERY
    assert int(work_probe[0]) == lw_q
    assert int(iwork_probe[0]) == liw_q
    assert res.lwork_min == lw_q
    assert res.liwork_min == liw_q

    # Query via liwork = -1 alone (with NULL work/iwork buffers allowed)
    res2 = FCResult()
    st2 = lib.fortchuck_solve(
        d,
        R,
        n_ptr,
        p_ptr,
        None,
        None,
        None,
        None,
        FCEvalCallback(0),
        None,
        ctypes.byref(opts),
        None,
        lw_q,
        None,
        -1,
        ctypes.byref(res2),
    )
    assert st2 == FC_SUCCESS_QUERY
    assert res2.lwork_min == lw_q
    assert res2.liwork_min == liw_q

    # Parity with fortchuck_solve_constrained (requires d >= 1; returns FC_ERR_INVALID_DIM on d == 0)
    work_probe_c = np.zeros(1, dtype=np.float64)
    iwork_probe_c = np.zeros(1, dtype=np.int32)
    res_c = FCResult()
    st_c = lib.fortchuck_solve_constrained(
        d,
        R,
        n_ptr,
        p_ptr,
        None,
        None,
        None,
        None,
        0.0,
        None,
        FCEvalCallback(0),
        None,
        ctypes.byref(opts),
        work_probe_c.ctypes.data_as(C_F64_P),
        -1,
        iwork_probe_c.ctypes.data_as(C_I32_P),
        -1,
        ctypes.byref(res_c),
    )
    if d >= 1:
        assert st_c == FC_SUCCESS_QUERY
        assert res_c.status == FC_SUCCESS_QUERY
        assert int(work_probe_c[0]) == lw_q
        assert int(iwork_probe_c[0]) == liw_q
        assert res_c.lwork_min == lw_q
        assert res_c.liwork_min == liw_q

        res_c2 = FCResult()
        st_c2 = lib.fortchuck_solve_constrained(
            d,
            R,
            n_ptr,
            p_ptr,
            None,
            None,
            None,
            None,
            1.0,
            None,
            FCEvalCallback(0),
            None,
            ctypes.byref(opts),
            None,
            lw_q,
            None,
            -1,
            ctypes.byref(res_c2),
        )
        assert st_c2 == FC_SUCCESS_QUERY
        assert res_c2.lwork_min == lw_q
        assert res_c2.liwork_min == liw_q
    else:
        assert st_c == FC_ERR_INVALID_DIM
        assert res_c.status == FC_ERR_INVALID_DIM


def test_zero_copy_fortran_order_views_and_reusable_workspace() -> None:
    rng = np.random.default_rng(12345)
    d = 4
    n, p = 10, 3
    diag_w = np.arange(1.0, n + 1.0, dtype=np.float64)
    z_target = np.array([0.0, 0.4, 1.0, -0.5], dtype=np.float64)
    lower = np.array([0.0, -1.0, -1.0, -0.5], dtype=np.float64)
    upper = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float64)

    z0 = np.array([0.75, -0.25, 0.95, 0.3], dtype=np.float64)
    X0 = project_stiefel(np.asfortranarray(rng.standard_normal((n, p))))
    assert X0.flags.f_contiguous

    z0_addr = z0.ctypes.data
    X0_addr = X0.ctypes.data
    view_checks: list[bool] = []

    def fg_inplace(
        z: np.ndarray,
        X_list: list[np.ndarray],
        gz_out: np.ndarray,
        GX_out_list: list[np.ndarray],
    ) -> float:
        X = X_list[0]
        GX_out = GX_out_list[0]
        view_checks.append(
            (not z.flags.writeable)
            and (not X.flags.writeable)
            and X.flags.f_contiguous
            and gz_out.flags.writeable
            and GX_out.flags.f_contiguous
            and GX_out.flags.writeable
        )
        diff = z - z_target
        f_box = 0.5 * float(np.dot(diff, diff)) + 2.0 * float(z[0])
        gz_out[:] = diff
        gz_out[0] += 2.0

        WX = diag_w[:, None] * X
        scale = 1.0 + 0.5 * float(z[1] ** 2)
        f_st = 0.5 * scale * float(np.sum(X * WX))
        gz_out[1] += 0.5 * float(z[1]) * float(np.sum(X * WX))
        GX_out[:] = scale * WX
        return f_box + f_st

    ws = FortchuckWorkspace(d=d, n_list=[n], p_list=[p], memory=7)
    work_addr = ws.work.ctypes.data
    iwork_addr = ws.iwork.ctypes.data

    res = minimize_box_stiefel(
        fg_inplace,
        z0=z0,
        bounds=(lower, upper),
        X0_list=X0,
        options={"gtol": 1.0e-10},
        workspace=ws,
        copy=False,
        in_place_grad=True,
    )

    assert res.status == FC_CONVERGED_GRAD
    assert res.converged is True
    assert res.kkt_residual <= 1.0e-10
    assert res.orth_error <= 1.0e-13
    assert len(view_checks) == res.n_eval and all(view_checks)

    # Exact zero-copy pointer identity on both z0 and Fortran-contiguous X0
    assert res.z is z0
    assert res.z.ctypes.data == z0_addr
    assert res.X_list[0] is X0
    assert res.X is X0
    assert res.X_list[0].ctypes.data == X0_addr
    assert ws.work.ctypes.data == work_addr
    assert ws.iwork.ctypes.data == iwork_addr

    # Verify copy=True leaves inputs untouched and workspace reuse is bit-identical
    z_clean = np.array([0.75, -0.25, 0.95, 0.3], dtype=np.float64)
    X_clean = project_stiefel(np.asfortranarray(rng.standard_normal((n, p))))
    z_snap = z_clean.copy()
    X_snap = X_clean.copy(order="F")

    res_a = minimize_box_stiefel(
        fg_inplace,
        z0=z_clean,
        bounds=(lower, upper),
        X0_list=X_clean,
        options={"gtol": 1.0e-10},
        workspace=ws,
        copy=True,
        in_place_grad=True,
    )
    assert np.array_equal(z_clean, z_snap)
    assert np.array_equal(X_clean, X_snap)

    res_b = minimize_box_stiefel(
        fg_inplace,
        z0=z_clean,
        bounds=(lower, upper),
        X0_list=X_clean,
        options={"gtol": 1.0e-10},
        workspace=ws,
        copy=True,
        in_place_grad=True,
    )
    assert res_a.f_val == res_b.f_val
    assert res_a.n_iter == res_b.n_iter
    assert res_a.n_eval == res_b.n_eval
    assert np.array_equal(res_a.z, res_b.z)
    assert np.array_equal(res_a.X_list[0], res_b.X_list[0])


def test_regimes_pure_box_pure_stiefel_so_det_and_ky_fan_truth() -> None:
    rng = np.random.default_rng(2026)

    # 1. Pure Box (R = 0) with +-DBL_MAX, +-inf, active bounds, and fixed l_i == u_i
    d_box = 6
    dbl_max = float(np.finfo(np.float64).max)
    l_box = np.array([-dbl_max, -np.inf, 0.5, -1.0, 2.25, -5.0], dtype=np.float64)
    u_box = np.array([dbl_max, np.inf, 4.0, 0.2, 2.25, 5.0], dtype=np.float64)
    targ_box = np.array([1.5, -2.0, -1.0, 2.0, 9.0, 0.7], dtype=np.float64)
    z0_box = np.array([0.0, 0.0, 2.0, 0.0, 2.25, -3.0], dtype=np.float64)

    def fg_pure_box(z: np.ndarray, _xs: list[np.ndarray]) -> tuple[float, np.ndarray, list[np.ndarray]]:
        w = np.arange(1.0, d_box + 1.0)
        diff = z - targ_box
        return 0.5 * float(np.sum(w * diff * diff)), w * diff, []

    res_box = minimize_box_stiefel(
        fg_pure_box, z0=z0_box, bounds=(l_box, u_box), options={"gtol": 1.0e-11}, copy=True
    )
    assert res_box.status == FC_CONVERGED_GRAD
    assert res_box.kkt_residual <= 1.0e-11
    exp_z = np.array([1.5, -2.0, 0.5, 0.2, 2.25, 0.7], dtype=np.float64)
    assert np.allclose(res_box.z, exp_z, atol=1.0e-9)

    # 2. Pure Sphere St(12, 1) (Branch A scalar fast-path) & Pure SO(6) (Branch B det=+1)
    q_sphere = rng.standard_normal(12)
    q_sphere /= np.linalg.norm(q_sphere)
    X0_sph = project_stiefel(rng.standard_normal((12, 1)))

    def fg_sphere(_z: np.ndarray, xs: list[np.ndarray]) -> tuple[float, None, list[np.ndarray]]:
        x = xs[0]
        return -float(np.dot(q_sphere, x[:, 0])), None, [-q_sphere[:, None]]

    res_sph = minimize_box_stiefel(
        fg_sphere, X0_list=[X0_sph], options={"gtol": 1.0e-10}, copy=True
    )
    assert res_sph.status == FC_CONVERGED_GRAD
    assert res_sph.f_val == pytest.approx(-1.0, abs=1.0e-11)

    # Pure SO(6) Procrustes preserving det(X) = +1
    U6, _, Vt6 = np.linalg.svd(rng.standard_normal((6, 6)))
    Q_target = U6 @ Vt6
    if np.linalg.det(Q_target) < 0.0:
        Q_target[:, 0] *= -1.0
    X0_so6 = project_stiefel(np.eye(6) + 0.35 * rng.standard_normal((6, 6)))
    if np.linalg.det(X0_so6) < 0.0:
        X0_so6[:, 0] *= -1.0
    assert np.linalg.det(X0_so6) == pytest.approx(1.0, abs=1.0e-14)

    def fg_so6(_z: np.ndarray, xs: list[np.ndarray]) -> tuple[float, None, list[np.ndarray]]:
        diff = xs[0] - Q_target
        return 0.5 * float(np.sum(diff * diff)), None, [diff]

    res_so6 = minimize_box_stiefel(
        fg_so6, X0_list=X0_so6, options={"gtol": 1.0e-10}, copy=True
    )
    assert res_so6.status == FC_CONVERGED_GRAD
    assert np.linalg.det(res_so6.X_list[0]) == pytest.approx(1.0, abs=1.0e-13)
    assert np.allclose(res_so6.X_list[0], Q_target, atol=1.0e-8)

    # 3. Box-Coupled Ky Fan Brockett across all metric/damping/memory modes
    n_kf, p_kf = 14, 3
    alphas = np.array([-8.0, -5.0, -2.5] + [1.25 * i for i in range(1, 12)], dtype=np.float64)
    Q14 = project_stiefel(rng.standard_normal((n_kf, n_kf)))
    A_kf = (Q14 * alphas[None, :]) @ Q14.T
    M_kf = np.array(
        [[10.0, -1.0, -0.5], [-1.0, 10.0, -1.0], [-0.5, -1.0, 10.0]],
        dtype=np.float64,
    )
    z_star = np.array([3.0, 1.6, 0.2], dtype=np.float64)
    m_dz_star = np.array([4.0, 5.0, 6.0], dtype=np.float64)
    z_ref = z_star - np.linalg.solve(M_kf, m_dz_star)
    f_star = float(np.dot(alphas[:3], z_star) + 0.5 * np.dot(z_star - z_ref, m_dz_star))

    z0_kf = np.array([0.7, 2.4, 1.9], dtype=np.float64)
    X0_kf = project_stiefel(rng.standard_normal((n_kf, p_kf)))
    l_kf = np.array([0.2, 0.2, 0.2], dtype=np.float64)
    u_kf = np.array([3.0, 3.0, 3.0], dtype=np.float64)

    def fg_ky_fan(
        z: np.ndarray, xs: list[np.ndarray]
    ) -> tuple[float, np.ndarray, list[np.ndarray]]:
        X = xs[0]
        AX = A_kf @ X
        ray = np.sum(X * AX, axis=0)
        dz = z - z_ref
        Mdz = M_kf @ dz
        f_val = float(np.dot(z, ray) + 0.5 * np.dot(dz, Mdz))
        gz = ray + Mdz
        GX = 2.0 * AX * z[None, :]
        return f_val, gz, [GX]

    for metric, damp, mem in [
        (FC_METRIC_CANONICAL, FC_DAMP_CAUTIOUS, 7),
        (FC_METRIC_CANONICAL, FC_DAMP_POWELL, 7),
        (FC_METRIC_EUCLIDEAN, FC_DAMP_CAUTIOUS, 7),
        (FC_METRIC_EUCLIDEAN, FC_DAMP_POWELL, 7),
        (FC_METRIC_CANONICAL, FC_DAMP_CAUTIOUS, 0),
    ]:
        res_kf = minimize_box_stiefel(
            fg_ky_fan,
            z0=z0_kf,
            bounds=(l_kf, u_kf),
            X0_list=X0_kf,
            options=FCOptions(
                gtol=1.0e-9,
                metric=metric,
                damping_mode=damp,
                memory=mem,
                max_iter=2000,
                max_eval=10000,
            ),
            copy=True,
        )
        assert res_kf.status == FC_CONVERGED_GRAD
        assert abs(res_kf.f_val - f_star) <= 1.0e-11
        assert res_kf.kkt_residual <= 1.0e-8
        assert res_kf.orth_error <= 1.0e-13
        assert np.allclose(res_kf.z, z_star, atol=1.0e-8)


def test_all_fc_err_and_budget_status_codes() -> None:
    lib = load_native_library()

    def simple_fg(
        z: np.ndarray, xs: list[np.ndarray]
    ) -> tuple[float, np.ndarray, list[np.ndarray]]:
        X = xs[0]
        d_diag = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
        DX = d_diag[:, None] * X
        f_val = 0.5 * float(np.sum((z - 0.3) ** 2) + 0.5 * np.sum(X * DX))
        return f_val, z - 0.3, [0.5 * DX]

    z0 = np.array([0.8, -0.4], dtype=np.float64)
    bounds = (np.array([-1.0, -1.0]), np.array([1.0, 1.0]))
    X0 = np.asfortranarray(np.eye(4, 2, dtype=np.float64))

    # Strict max_eval enforcement across tiny and moderate budgets
    for cap in (1, 2, 3, 5, 8):
        calls = 0

        def counted_fg(
            z: np.ndarray, xs: list[np.ndarray]
        ) -> tuple[float, np.ndarray, list[np.ndarray]]:
            nonlocal calls
            calls += 1
            # Non-stationary coupled problem taking >10 evals
            X = xs[0]
            B = np.array([[0.2, 0.9], [-0.7, 0.3], [0.5, -0.4], [0.1, 0.8]])
            diff_z = z - np.array([0.123, -0.456])
            diff_x = X - B
            return (
                0.5 * float(np.sum(diff_z**2) + np.sum(diff_x**2)),
                diff_z,
                [diff_x],
            )

        r_cap = minimize_box_stiefel(
            counted_fg,
            z0=z0,
            bounds=bounds,
            X0_list=X0,
            options={"max_eval": cap, "gtol": 1.0e-14},
            copy=True,
        )
        assert r_cap.status == FC_MAX_EVAL_REACHED
        assert r_cap.n_eval == cap
        assert calls == cap
        assert r_cap.n_eval <= cap

    # FC_MAX_ITER_REACHED (4), FC_CONVERGED_FTOL (2), FC_CONVERGED_XTOL (3)
    def curved_fg(
        z: np.ndarray, xs: list[np.ndarray]
    ) -> tuple[float, np.ndarray, list[np.ndarray]]:
        X = xs[0]
        B = np.array([[0.2, 0.9], [-0.7, 0.3], [0.5, -0.4], [0.1, 0.8]])
        diff_z = z - np.array([0.123, -0.456])
        diff_x = X - B
        return (
            0.5 * float(np.sum(diff_z**2) + np.sum(diff_x**2)),
            diff_z,
            [diff_x],
        )

    r_iter = minimize_box_stiefel(
        curved_fg, z0=z0, bounds=bounds, X0_list=X0, options={"max_iter": 2, "gtol": 1e-15}, copy=True
    )
    assert r_iter.status == FC_MAX_ITER_REACHED
    assert r_iter.n_iter == 2

    r_ftol = minimize_box_stiefel(
        curved_fg,
        z0=z0,
        bounds=bounds,
        X0_list=X0,
        options={"gtol": 0.0, "ftol": 1.0e-3, "xtol": 0.0},
        copy=True,
    )
    assert r_ftol.status == FC_CONVERGED_FTOL

    r_xtol = minimize_box_stiefel(
        curved_fg,
        z0=z0,
        bounds=bounds,
        X0_list=X0,
        options={"gtol": 0.0, "ftol": 0.0, "xtol": 1.0e-2},
        copy=True,
    )
    assert r_xtol.status == FC_CONVERGED_XTOL

    # FC_ERR_INVALID_DIM (-1)
    r_dim0 = minimize_box_stiefel(simple_fg)
    assert r_dim0.status == FC_ERR_INVALID_DIM

    r_dim_p = minimize_box_stiefel(simple_fg, X0_list=np.ones((2, 4)), copy=True)
    assert r_dim_p.status == FC_ERR_INVALID_DIM

    r_dim_mem = minimize_box_stiefel(
        simple_fg, z0=z0, bounds=bounds, X0_list=X0, options={"memory": -1}, copy=True
    )
    assert r_dim_mem.status == FC_ERR_INVALID_DIM

    # FC_ERR_NULL_PTR (-2) via direct C-ABI calls
    res_null = FCResult()
    opts_ok = FCOptions()
    w_ok = np.zeros(4096, dtype=np.float64)
    iw_ok = np.zeros(512, dtype=np.int32)
    n_c = np.array([4], dtype=np.int32)
    p_c = np.array([2], dtype=np.int32)

    assert (
        lib.fortchuck_solve(
            2, 0, None, None, None, None, None, None, FCEvalCallback(0), None, None, None, 10, None, 10, None
        )
        == FC_ERR_NULL_PTR
    )
    assert (
        lib.fortchuck_solve(
            0,
            1,
            None,
            p_c.ctypes.data_as(C_I32_P),
            None,
            None,
            None,
            X0.ctypes.data_as(C_F64_P),
            FCEvalCallback(0),
            None,
            ctypes.byref(opts_ok),
            w_ok.ctypes.data_as(C_F64_P),
            w_ok.size,
            iw_ok.ctypes.data_as(C_I32_P),
            iw_ok.size,
            ctypes.byref(res_null),
        )
        == FC_ERR_NULL_PTR
    )
    assert (
        lib.fortchuck_solve(
            2,
            0,
            None,
            None,
            None,
            bounds[0].ctypes.data_as(C_F64_P),
            bounds[1].ctypes.data_as(C_F64_P),
            None,
            FCEvalCallback(0),
            None,
            ctypes.byref(opts_ok),
            w_ok.ctypes.data_as(C_F64_P),
            w_ok.size,
            iw_ok.ctypes.data_as(C_I32_P),
            iw_ok.size,
            ctypes.byref(res_null),
        )
        == FC_ERR_NULL_PTR
    )
    assert (
        lib.fortchuck_solve(
            2,
            0,
            None,
            None,
            z0.ctypes.data_as(C_F64_P),
            bounds[0].ctypes.data_as(C_F64_P),
            bounds[1].ctypes.data_as(C_F64_P),
            None,
            FCEvalCallback(0),
            None,
            ctypes.byref(opts_ok),
            w_ok.ctypes.data_as(C_F64_P),
            w_ok.size,
            iw_ok.ctypes.data_as(C_I32_P),
            iw_ok.size,
            ctypes.byref(res_null),
        )
        == FC_ERR_NULL_PTR
    )
    assert n_c.size == 1

    # FC_ERR_INVALID_BOUNDS (-3)
    for bad_l, bad_u in [
        (np.array([1.0, 0.0]), np.array([0.0, 1.0])),
        (np.array([np.nan, 0.0]), np.array([1.0, 1.0])),
        (np.array([np.inf, 0.0]), np.array([np.inf, 1.0])),
        (np.array([-1.0, -np.inf]), np.array([1.0, -np.inf])),
    ]:
        r_b = minimize_box_stiefel(simple_fg, z0=z0, bounds=(bad_l, bad_u), X0_list=X0, copy=True)
        assert r_b.status == FC_ERR_INVALID_BOUNDS

    # FC_ERR_INVALID_X0 (-4): rank-deficient, NaN, and unprojected with project_initial=0
    X_rank_def = np.zeros((4, 2), dtype=np.float64, order="F")
    assert (
        minimize_box_stiefel(simple_fg, z0=z0, bounds=bounds, X0_list=X_rank_def, copy=True).status
        == FC_ERR_INVALID_X0
    )
    X_nan = X0.copy(order="F")
    X_nan[0, 0] = np.nan
    assert (
        minimize_box_stiefel(simple_fg, z0=z0, bounds=bounds, X0_list=X_nan, copy=True).status
        == FC_ERR_INVALID_X0
    )
    X_unorth = 2.0 * X0
    assert (
        minimize_box_stiefel(
            simple_fg,
            z0=z0,
            bounds=bounds,
            X0_list=X_unorth,
            options={"project_initial": 0},
            copy=True,
        ).status
        == FC_ERR_INVALID_X0
    )

    # FC_ERR_WORK_TOO_SMALL (-5)
    lw_min, liw_min = workspace_query(d=2, n_list=[4], p_list=[2], memory=7)
    r_small_w = minimize_box_stiefel(
        simple_fg,
        z0=z0,
        bounds=bounds,
        X0_list=X0,
        work=np.empty(lw_min - 1, dtype=np.float64),
        iwork=np.empty(liw_min, dtype=np.int32),
        copy=True,
    )
    assert r_small_w.status == FC_ERR_WORK_TOO_SMALL

    r_small_iw = minimize_box_stiefel(
        simple_fg,
        z0=z0,
        bounds=bounds,
        X0_list=X0,
        work=np.empty(lw_min, dtype=np.float64),
        iwork=np.empty(liw_min - 1, dtype=np.int32),
        copy=True,
    )
    assert r_small_iw.status == FC_ERR_WORK_TOO_SMALL

    # FC_ERR_CALLBACK_NAN (-6) at p_0
    r_nan0 = minimize_box_stiefel(
        lambda _z, _xs: (np.nan, None, None),
        z0=z0,
        bounds=bounds,
        X0_list=X0,
        copy=True,
    )
    assert r_nan0.status == FC_ERR_CALLBACK_NAN

    r_rcpos0 = minimize_box_stiefel(
        lambda _z, _xs: (0.0, None, None, 1),
        z0=z0,
        bounds=bounds,
        X0_list=X0,
        copy=True,
    )
    assert r_rcpos0.status == FC_ERR_CALLBACK_NAN

    # FC_ERR_CALLBACK_ABORT (-7) via explicit rc < 0
    eval_cnt = 0

    def abort_on_third(
        z: np.ndarray, xs: list[np.ndarray]
    ) -> tuple[float, np.ndarray | None, list[np.ndarray] | None, int]:
        nonlocal eval_cnt
        eval_cnt += 1
        if eval_cnt >= 3:
            return 0.0, None, None, -1
        f_v, gz_v, gx_v = curved_fg(z, xs)
        return f_v, gz_v, gx_v, 0

    r_abort = minimize_box_stiefel(
        abort_on_third, z0=z0, bounds=bounds, X0_list=X0, copy=True
    )
    assert r_abort.status == FC_ERR_CALLBACK_ABORT
    assert r_abort.n_eval == 3

    # FC_ERR_LS_FAILED (-8): let step 0 succeed (populating L-BFGS m_cur=1), then flip gradient sign
    # on step 1 so QN line search fails -> triggers ABB fallback -> fails with FC_ERR_LS_FAILED!
    ls_calls = 0

    def adversarial_wrong_grad(
        z: np.ndarray, xs: list[np.ndarray]
    ) -> tuple[float, np.ndarray, list[np.ndarray]]:
        nonlocal ls_calls
        ls_calls += 1
        f_v, gz_v, gx_v = curved_fg(z, xs)
        if ls_calls <= 2:
            return f_v, gz_v, gx_v
        return f_v, -gz_v, [-gx_v[0]]

    r_ls = minimize_box_stiefel(
        adversarial_wrong_grad,
        z0=z0,
        bounds=bounds,
        X0_list=X0,
        options={"max_ls_steps": 12, "gtol": 1.0e-14},
        copy=True,
    )
    assert r_ls.status == FC_ERR_LS_FAILED
    assert r_ls.n_fallbacks >= 1

    # FC_ERR_INVALID_OPTS (-9)
    for bad_kw in [
        {"c1_armijo": 0.0},
        {"c1_armijo": 1.0},
        {"gtol": -1.0},
        {"zh_eta": 1.0},
        {"orth_tol": 0.0},
        {"max_step_vel": 0.0},
        {"max_iter": -1},
        {"max_eval": 0},
        {"max_ls_steps": 0},
        {"metric": 77},
        {"damping_mode": -3},
        {"project_initial": 2},
    ]:
        r_opt = minimize_box_stiefel(
            simple_fg, z0=z0, bounds=bounds, X0_list=X0, options=bad_kw, copy=True
        )
        assert r_opt.status == FC_ERR_INVALID_OPTS


def test_python_exception_propagation_across_cfunctype(monkeypatch: pytest.MonkeyPatch) -> None:
    unraisable_events: list[object] = []
    monkeypatch.setattr(sys, "unraisablehook", lambda args: unraisable_events.append(args))

    class RefereeStopSentinel(RuntimeError):
        pass

    z0 = np.array([0.6, -0.3], dtype=np.float64)
    X0 = project_stiefel(np.array([[1.0, 0.2], [0.1, 1.0], [-0.3, 0.4]], order="F"))
    target_x = np.array([[0.2, 0.9], [-0.7, 0.4], [0.6, -0.3]], dtype=np.float64)

    for trigger_call in (1, 3):
        calls = 0

        def raising_cb(
            z: np.ndarray, xs: list[np.ndarray]
        ) -> tuple[float, np.ndarray, list[np.ndarray]]:
            nonlocal calls
            calls += 1
            if calls == trigger_call:
                raise RefereeStopSentinel(f"budget check triggered at call {trigger_call}")
            X = xs[0]
            dz = z - np.array([-0.25, 0.45])
            dx = X - target_x
            return 0.5 * float(np.sum(dz**2) + np.sum(dx**2)), dz, [dx]

        with pytest.raises(
            RefereeStopSentinel, match=f"budget check triggered at call {trigger_call}"
        ):
            minimize_box_stiefel(
                raising_cb,
                z0=z0,
                bounds=([-1.0, -1.0], [1.0, 1.0]),
                X0_list=X0,
                options={"gtol": 1.0e-14},
                copy=True,
            )

        assert calls == trigger_call
        assert unraisable_events == []


def test_project_stiefel_and_kkt_residual_c_abi_helpers() -> None:
    rng = np.random.default_rng(99)

    # 1. project_stiefel zero-copy, positive R_ii small-drift bound, and SO(n) det=+1
    U5, _, Vt5 = np.linalg.svd(rng.standard_normal((5, 5)))
    Q5 = np.asfortranarray(U5 @ Vt5)
    if np.linalg.det(Q5) < 0.0:
        Q5[:, 0] *= -1.0

    perturbed = np.asfortranarray(Q5 + 1.0e-12 * rng.standard_normal((5, 5)))
    ptr_before = perturbed.ctypes.data
    polished = project_stiefel(perturbed, copy=False)
    assert polished is perturbed
    assert polished.ctypes.data == ptr_before
    assert np.linalg.norm(polished.T @ polished - np.eye(5), ord="fro") <= 1.0e-15
    assert np.linalg.det(polished) == pytest.approx(1.0, abs=1.0e-14)
    assert np.linalg.norm(polished - Q5, ord="fro") <= 5.0e-12

    with pytest.raises(ValueError, match="FC_ERR_INVALID_X0"):
        project_stiefel(np.zeros((6, 3), dtype=np.float64))

    with pytest.raises(ValueError, match="FC_ERR_INVALID_DIM"):
        project_stiefel(np.ones((3, 5), dtype=np.float64))

    # 2. kkt_residual parity with solver result on canonical and Euclidean metrics
    d = 3
    l_arr = np.array([0.0, -1.0, 0.5], dtype=np.float64)
    u_arr = np.array([1.0, 1.0, 0.5], dtype=np.float64)
    z = np.array([0.0, 0.25, 0.5], dtype=np.float64)
    gz = np.array([4.5, -0.15, 99.0], dtype=np.float64)
    X1 = project_stiefel(rng.standard_normal((9, 3)))
    GX1 = rng.standard_normal((9, 3))

    for metric in (FC_METRIC_CANONICAL, FC_METRIC_EUCLIDEAN):
        kkt_val, orth_val = kkt_residual(
            z=z, bounds=(l_arr, u_arr), X_list=[X1], gz=gz, GX_list=[GX1], metric=metric
        )
        r_box = z - np.clip(z - gz, l_arr, u_arr)
        box_inf = float(np.max(np.abs(r_box)))
        if metric == FC_METRIC_CANONICAL:
            rg = GX1 - X1 @ (GX1.T @ X1)
            xt_rg = X1.T @ rg
            st_norm = math.sqrt(float(np.sum(rg * rg) - 0.5 * np.sum(xt_rg * xt_rg)))
        else:
            sym = 0.5 * (X1.T @ GX1 + GX1.T @ X1)
            rg = GX1 - X1 @ sym
            st_norm = float(np.linalg.norm(rg, ord="fro"))
        assert kkt_val == pytest.approx(max(box_inf, st_norm), rel=1e-14, abs=1e-15)
        assert orth_val == pytest.approx(
            float(np.linalg.norm(X1.T @ X1 - np.eye(3), ord="fro")), abs=1e-15
        )


def test_project_linear_box_and_kkt_residual_constrained() -> None:
    lib = load_native_library()
    rng = np.random.default_rng(202602)

    # 1. Unit simplex & capped simplex projection + variational optimality certificate
    d = 8
    lower = np.zeros(d, dtype=np.float64)
    upper = np.full(d, 0.35, dtype=np.float64)
    a_ones = np.ones(d, dtype=np.float64)
    b_cap = 1.25

    y = rng.standard_normal(d)
    y_snap = y.copy()
    z_proj, nu = project_linear_box(y, lower, upper, a_ones, b_cap)
    # Default copy=True leaves input untouched
    assert np.array_equal(y, y_snap)
    assert z_proj is not y
    assert np.all(z_proj >= lower - 1.0e-16) and np.all(z_proj <= upper + 1.0e-16)
    assert abs(float(np.dot(a_ones, z_proj)) - b_cap) <= 1.0e-14
    assert np.allclose(z_proj, np.clip(y - nu * a_ones, lower, upper), atol=1.0e-14, rtol=0.0)

    # Variational inequality <y - z_proj, w - z_proj> <= 0 across random feasible w in C
    for _ in range(25):
        w_raw = rng.uniform(-2.0, 2.0, size=d)
        w_feas, _ = project_linear_box(w_raw, lower, upper, a_ones, b_cap)
        assert float(np.dot(y - z_proj, w_feas - z_proj)) <= 1.0e-14

    # Zero-copy in-place fast path when copy=False on writeable float64 1D ndarray
    y_inplace = y.copy()
    ptr_before = y_inplace.ctypes.data
    z_in, nu_in = project_linear_box(y_inplace, lower, upper, a_ones, b_cap, copy=False)
    assert z_in is y_inplace
    assert z_in.ctypes.data == ptr_before
    assert np.array_equal(z_in, z_proj)
    assert nu_in == pytest.approx(nu, rel=0.0, abs=1.0e-16)

    # 2. General signed weights, +-inf / +-DBL_MAX bounds, singleton faces, and raw C-ABI NULL nu_out
    dbl_max = float(np.finfo(np.float64).max)
    l_gen = np.array([-np.inf, -dbl_max, 0.0, -1.5, 0.4, 0.0], dtype=np.float64)
    u_gen = np.array([1.0, dbl_max, np.inf, 2.0, 0.4, 3.0], dtype=np.float64)
    a_gen = np.array([1.2, -0.75, 0.5, -1.4, 2.0, 0.8], dtype=np.float64)
    b_gen = 1.65
    y_gen = np.array([2.5, -1.0, -0.8, 1.2, 9.0, 0.3], dtype=np.float64)
    zp_gen, nu_gen = project_linear_box(y_gen, l_gen, u_gen, a_gen, b_gen)
    assert zp_gen[4] == pytest.approx(0.4, abs=1.0e-16)
    assert abs(float(np.dot(a_gen, zp_gen)) - b_gen) <= 1.0e-14
    assert np.allclose(zp_gen, np.clip(y_gen - nu_gen * a_gen, l_gen, u_gen), atol=1.0e-13, rtol=0.0)

    # Raw C-ABI call with nu_out == NULL and singleton boundary faces (b == b_min, b == b_max)
    z_c = y.copy()
    st_null_nu = lib.fortchuck_project_linear_box(
        d,
        lower.ctypes.data_as(C_F64_P),
        upper.ctypes.data_as(C_F64_P),
        a_ones.ctypes.data_as(C_F64_P),
        b_cap,
        z_c.ctypes.data_as(C_F64_P),
        None,
    )
    assert st_null_nu == FC_SUCCESS_QUERY
    assert np.allclose(z_c, z_proj, atol=1.0e-15, rtol=0.0)

    z_bmin, _ = project_linear_box(y, lower, upper, a_ones, 0.0)
    assert np.allclose(z_bmin, lower, atol=1.0e-15, rtol=0.0)
    z_bmax, _ = project_linear_box(y, lower, upper, a_ones, float(np.sum(upper)))
    assert np.allclose(z_bmax, upper, atol=1.0e-15, rtol=0.0)

    # 3. fortchuck_kkt_residual_constrained lwork = -1 query & Python kkt_residual(..., linear_constraint=...)
    w_q = np.zeros(1, dtype=np.float64)
    st_q0 = lib.fortchuck_kkt_residual_constrained(
        d, 0, None, None, None, None, None, None, b_cap, None, None, None,
        FC_METRIC_CANONICAL, w_q.ctypes.data_as(C_F64_P), -1, None, None, None,
    )
    assert st_q0 == FC_SUCCESS_QUERY and int(w_q[0]) == d

    n_arr = np.array([10, 4], dtype=np.int32)
    p_arr = np.array([2, 4], dtype=np.int32)
    st_q2 = lib.fortchuck_kkt_residual_constrained(
        d, 2, n_arr.ctypes.data_as(C_I32_P), p_arr.ctypes.data_as(C_I32_P),
        None, None, None, None, b_cap, None, None, None,
        FC_METRIC_CANONICAL, w_q.ctypes.data_as(C_F64_P), -1, None, None, None,
    )
    exp_lw_kkt = max(1, d + max(10 * 2 + 3 * 4, 4 * 4 + 3 * 16))
    assert st_q2 == FC_SUCCESS_QUERY and int(w_q[0]) == exp_lw_kkt

    gz_test = rng.standard_normal(d)
    X1 = project_stiefel(rng.standard_normal((10, 2)))
    X2 = project_stiefel(rng.standard_normal((4, 4)))
    GX1 = rng.standard_normal((10, 2))
    GX2 = rng.standard_normal((4, 4))

    for metric in (FC_METRIC_CANONICAL, FC_METRIC_EUCLIDEAN):
        res_tuple = kkt_residual(
            z=z_proj,
            bounds=(lower, upper),
            X_list=[X1, X2],
            gz=gz_test,
            GX_list=[GX1, GX2],
            linear_constraint=(a_ones, b_cap),
            metric=metric,
        )
        assert len(res_tuple) == 3
        kkt_val, lin_val, orth_val = res_tuple

        z_step_proj, _ = project_linear_box(z_proj - gz_test, lower, upper, a_ones, b_cap)
        box_inf_ref = float(np.max(np.abs(z_proj - z_step_proj)))
        st_norms: list[float] = []
        for X_r, GX_r in ((X1, GX1), (X2, GX2)):
            if metric == FC_METRIC_CANONICAL:
                rg = GX_r - X_r @ (GX_r.T @ X_r)
                xt_rg = X_r.T @ rg
                st_norms.append(math.sqrt(float(np.sum(rg * rg) - 0.5 * np.sum(xt_rg * xt_rg))))
            else:
                sym = 0.5 * (X_r.T @ GX_r + GX_r.T @ X_r)
                rg = GX_r - X_r @ sym
                st_norms.append(float(np.linalg.norm(rg, ord="fro")))
        exp_orth = max(
            float(np.linalg.norm(X1.T @ X1 - np.eye(2), ord="fro")),
            float(np.linalg.norm(X2.T @ X2 - np.eye(4), ord="fro")),
        )
        assert kkt_val == pytest.approx(max(box_inf_ref, *st_norms), rel=1e-14, abs=1e-15)
        assert lin_val == pytest.approx(abs(float(np.dot(a_ones, z_proj)) - b_cap), abs=1e-15)
        assert orth_val == pytest.approx(exp_orth, abs=1e-15)


def test_infeasible_linear_constraint_and_validation_errors() -> None:
    lib = load_native_library()
    d = 4
    lower = np.zeros(d, dtype=np.float64)
    upper = np.ones(d, dtype=np.float64)
    a_ok = np.array([1.0, 2.0, 0.5, 1.5], dtype=np.float64)  # b_min = 0.0, b_max = 5.0
    z0 = np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float64)
    X0 = np.asfortranarray(np.eye(5, 2, dtype=np.float64))

    def simple_fg(
        z: np.ndarray, xs: list[np.ndarray]
    ) -> tuple[float, np.ndarray, list[np.ndarray]]:
        X = xs[0] if xs else None
        f_v = 0.5 * float(np.sum((z - 0.3) ** 2))
        gx_list: list[np.ndarray] = []
        if X is not None:
            f_v += 0.5 * float(np.sum(X * X))
            gx_list.append(X.copy(order="F"))
        return f_v, z - 0.3, gx_list

    # 1. Raw C-ABI FC_ERR_INFEASIBLE_LIN (-10) across project_linear_box, kkt_residual_constrained, solve_constrained
    z_tmp = z0.copy()
    gz_tmp = np.ones(d, dtype=np.float64)
    w_tmp = np.zeros(128, dtype=np.float64)
    iw_tmp = np.zeros(64, dtype=np.int32)
    kkt_o = ctypes.c_double(0.0)
    lin_o = ctypes.c_double(0.0)
    orth_o = ctypes.c_double(0.0)
    opts = FCOptions()
    res_c = FCResult()
    lw_min, liw_min = workspace_query(d=d, n_list=[], p_list=[], memory=opts.memory)
    work_s = np.zeros(lw_min, dtype=np.float64)
    iwork_s = np.zeros(liw_min, dtype=np.int32)

    @FCEvalCallback
    def dummy_cb(
        _d: int, _r: int, _np: object, _pp: object, _zp: object, _xp: object,
        f_p: object, _gzp: object, _gxp: object, _u: object
    ) -> int:
        f_p[0] = 0.0  # type: ignore[index]
        return 0

    a_zero = np.zeros(d, dtype=np.float64)
    a_nan = np.array([1.0, np.nan, 0.5, 1.5], dtype=np.float64)
    a_inf = np.array([1.0, np.inf, 0.5, 1.5], dtype=np.float64)

    for bad_a, bad_b in [
        (a_ok, 5.5),       # b > b_max
        (a_ok, -0.25),     # b < b_min
        (a_zero, 0.0),     # ||a||_inf == 0
        (a_nan, 1.0),      # NaN in a
        (a_inf, 1.0),      # Inf in a
        (a_ok, np.nan),    # NaN in b
        (a_ok, np.inf),    # Inf in b
    ]:
        assert (
            lib.fortchuck_project_linear_box(
                d,
                lower.ctypes.data_as(C_F64_P),
                upper.ctypes.data_as(C_F64_P),
                bad_a.ctypes.data_as(C_F64_P),
                float(bad_b),
                z_tmp.ctypes.data_as(C_F64_P),
                None,
            )
            == FC_ERR_INFEASIBLE_LIN
        )
        assert (
            lib.fortchuck_kkt_residual_constrained(
                d, 0, None, None,
                z_tmp.ctypes.data_as(C_F64_P),
                lower.ctypes.data_as(C_F64_P),
                upper.ctypes.data_as(C_F64_P),
                bad_a.ctypes.data_as(C_F64_P),
                float(bad_b),
                None,
                gz_tmp.ctypes.data_as(C_F64_P),
                None,
                FC_METRIC_CANONICAL,
                w_tmp.ctypes.data_as(C_F64_P),
                w_tmp.size,
                ctypes.byref(kkt_o),
                ctypes.byref(lin_o),
                ctypes.byref(orth_o),
            )
            == FC_ERR_INFEASIBLE_LIN
        )
        assert (
            lib.fortchuck_solve_constrained(
                d, 0, None, None,
                z_tmp.ctypes.data_as(C_F64_P),
                lower.ctypes.data_as(C_F64_P),
                upper.ctypes.data_as(C_F64_P),
                bad_a.ctypes.data_as(C_F64_P),
                float(bad_b),
                None,
                dummy_cb,
                None,
                ctypes.byref(opts),
                work_s.ctypes.data_as(C_F64_P),
                work_s.size,
                iwork_s.ctypes.data_as(C_I32_P),
                iwork_s.size,
                ctypes.byref(res_c),
            )
            == FC_ERR_INFEASIBLE_LIN
        )
        assert res_c.status == FC_ERR_INFEASIBLE_LIN

        # Python solver returns OptimizeResult with status == FC_ERR_INFEASIBLE_LIN
        r_py = minimize_box_stiefel(
            simple_fg,
            z0=z0,
            bounds=(lower, upper),
            X0_list=X0,
            linear_constraint=(bad_a, float(bad_b)),
            copy=True,
        )
        assert r_py.status == FC_ERR_INFEASIBLE_LIN
        assert r_py.status_name == "FC_ERR_INFEASIBLE_LIN"
        assert r_py.converged is False

        # Python helpers raise ValueError mentioning FC_ERR_INFEASIBLE_LIN
        with pytest.raises(ValueError, match="FC_ERR_INFEASIBLE_LIN"):
            project_linear_box(z0, lower, upper, bad_a, float(bad_b))
        with pytest.raises(ValueError, match="FC_ERR_INFEASIBLE_LIN"):
            kkt_residual(
                z=z0, bounds=(lower, upper), gz=gz_tmp, linear_constraint=(bad_a, float(bad_b))
            )

    # 2. Additional C-ABI & Python constrained guard codes (NULL, d=0, project_initial=0)
    assert (
        lib.fortchuck_project_linear_box(
            0, lower.ctypes.data_as(C_F64_P), upper.ctypes.data_as(C_F64_P),
            a_ok.ctypes.data_as(C_F64_P), 1.0, z_tmp.ctypes.data_as(C_F64_P), None,
        )
        == FC_ERR_INVALID_DIM
    )
    assert (
        lib.fortchuck_project_linear_box(
            d, None, upper.ctypes.data_as(C_F64_P),
            a_ok.ctypes.data_as(C_F64_P), 1.0, z_tmp.ctypes.data_as(C_F64_P), None,
        )
        == FC_ERR_NULL_PTR
    )
    assert (
        lib.fortchuck_solve_constrained(
            d, 0, None, None,
            z_tmp.ctypes.data_as(C_F64_P),
            lower.ctypes.data_as(C_F64_P),
            upper.ctypes.data_as(C_F64_P),
            None,
            1.0,
            None,
            dummy_cb,
            None,
            ctypes.byref(opts),
            work_s.ctypes.data_as(C_F64_P),
            work_s.size,
            iwork_s.ctypes.data_as(C_I32_P),
            iwork_s.size,
            ctypes.byref(res_c),
        )
        == FC_ERR_NULL_PTR
    )
    _ = iw_tmp

    # project_initial = 0 with a^T z0 != b -> FC_ERR_INVALID_X0
    r_unproj = minimize_box_stiefel(
        simple_fg,
        z0=z0,  # a_ok^T z0 = 1.25 != 2.5
        bounds=(lower, upper),
        X0_list=X0,
        options={"project_initial": 0},
        linear_constraint=(a_ok, 2.5),
        copy=True,
    )
    assert r_unproj.status == FC_ERR_INVALID_X0

    # Python structural argument validation
    with pytest.raises(ValueError, match="d >= 1"):
        minimize_box_stiefel(simple_fg, X0_list=X0, linear_constraint=(a_ok, 1.0), copy=True)
    with pytest.raises(ValueError, match="1D float64 array"):
        minimize_box_stiefel(
            simple_fg, z0=z0, bounds=(lower, upper), linear_constraint=(a_ok[:2], 1.0), copy=True
        )
    with pytest.raises(ValueError, match="1D float64 array"):
        minimize_box_stiefel(
            simple_fg,
            z0=z0,
            bounds=(lower, upper),
            linear_constraint=(a_ok.reshape(2, 2), 1.0),
            copy=True,
        )
    with pytest.raises(ValueError, match="2-tuple"):
        minimize_box_stiefel(
            simple_fg, z0=z0, bounds=(lower, upper), linear_constraint=(a_ok,), copy=True  # type: ignore[arg-type]
        )
    with pytest.raises(TypeError, match="real scalar"):
        minimize_box_stiefel(
            simple_fg,
            z0=z0,
            bounds=(lower, upper),
            linear_constraint=(a_ok, np.array([1.0, 2.0])),  # type: ignore[arg-type]
            copy=True,
        )


def test_minimize_box_stiefel_linear_constraint_regimes() -> None:
    rng = np.random.default_rng(202603)

    # 1. Pure Simplex / Capped Knapsack (R = 0) with reusable FortchuckWorkspace
    d0 = 12
    l0 = np.zeros(d0, dtype=np.float64)
    u0 = np.full(d0, 0.4, dtype=np.float64)
    a0 = np.linspace(0.6, 1.7, d0, dtype=np.float64)
    b0 = 2.0
    w0 = np.linspace(1.5, 8.0, d0, dtype=np.float64)
    targ0 = np.array([-0.5, 0.9, 0.25, -0.3, 0.8, 0.15, 0.6, -0.2, 0.35, 0.1, 0.7, -0.4])
    z0_init = np.full(d0, 0.1, dtype=np.float64)  # Infeasible sum -> projected at k=0

    def fg_knapsack(
        z: np.ndarray, _xs: list[np.ndarray]
    ) -> tuple[float, np.ndarray, list[np.ndarray]]:
        diff = z - targ0
        return 0.5 * float(np.sum(w0 * diff * diff)), w0 * diff, []

    ws0 = FortchuckWorkspace(d=d0, n_list=[], p_list=[], memory=7)
    res0 = minimize_box_stiefel(
        fg_knapsack,
        z0=z0_init,
        bounds=(l0, u0),
        options={"gtol": 1.0e-11},
        linear_constraint=(a0, b0),
        workspace=ws0,
        copy=True,
    )
    assert res0.status == FC_CONVERGED_GRAD
    assert res0.kkt_residual <= 1.0e-11
    assert abs(float(np.dot(a0, res0.z)) - b0) <= 1.0e-14
    assert np.any(res0.z == 0.0) and np.any(np.isclose(res0.z, 0.4, atol=1e-14))
    _, gz0_fin, _ = fg_knapsack(res0.z, [])
    kkt0, lin0, orth0 = kkt_residual(
        z=res0.z, bounds=(l0, u0), gz=gz0_fin, linear_constraint=(a0, b0)
    )
    assert kkt0 == pytest.approx(res0.kkt_residual, abs=1.0e-14)
    assert lin0 <= 1.0e-14 and orth0 == 0.0

    # 2. Coupled Trace Constraint a = 1 (R = 1) + Zero-Copy order='F' + Mode Coverage
    d1, n1, p1 = 6, 12, 3
    l1 = np.zeros(d1, dtype=np.float64)
    u1 = np.ones(d1, dtype=np.float64)
    a1 = np.ones(d1, dtype=np.float64)
    b1 = 2.5
    d_st1 = np.linspace(1.0, 6.0, n1, dtype=np.float64)
    targ1 = np.array([-0.2, 0.85, 0.55, 1.3, 0.35, -0.4], dtype=np.float64)

    z1_zc = np.array([0.5, 0.4, 0.4, 0.5, 0.4, 0.3], dtype=np.float64)
    X1_zc = project_stiefel(np.asfortranarray(rng.standard_normal((n1, p1))))
    z1_ptr = z1_zc.ctypes.data
    X1_ptr = X1_zc.ctypes.data

    def fg_trace_inplace(
        z: np.ndarray,
        xs: list[np.ndarray],
        gz_out: np.ndarray,
        gx_out_list: list[np.ndarray],
    ) -> float:
        X = xs[0]
        GX_out = gx_out_list[0]
        diff = z - targ1
        f_box = 0.5 * float(np.sum(3.0 * diff * diff))
        gz_out[:] = 3.0 * diff

        DX = d_st1[:, None] * X
        col_quad = np.sum(X * DX, axis=0)
        weights = 0.5 + z[:p1]
        f_st = 0.5 * float(np.dot(weights, col_quad))
        gz_out[:p1] += 0.5 * col_quad
        GX_out[:] = DX * weights[None, :]
        return f_box + f_st

    ws1 = FortchuckWorkspace(d=d1, n_list=[n1], p_list=[p1], memory=7)
    res1_zc = minimize_box_stiefel(
        fg_trace_inplace,
        z0=z1_zc,
        bounds=(l1, u1),
        X0_list=X1_zc,
        options={"gtol": 1.0e-10},
        linear_constraint=(a1, b1),
        workspace=ws1,
        copy=False,
        in_place_grad=True,
    )
    assert res1_zc.status == FC_CONVERGED_GRAD
    assert res1_zc.z is z1_zc and res1_zc.z.ctypes.data == z1_ptr
    assert res1_zc.X_list[0] is X1_zc and res1_zc.X_list[0].ctypes.data == X1_ptr
    assert res1_zc.kkt_residual <= 1.0e-10
    assert abs(float(np.sum(res1_zc.z)) - b1) <= 1.0e-14
    assert res1_zc.orth_error <= 1.0e-13

    for metric, damp, mem in [
        (FC_METRIC_CANONICAL, FC_DAMP_CAUTIOUS, 7),
        (FC_METRIC_CANONICAL, FC_DAMP_POWELL, 7),
        (FC_METRIC_EUCLIDEAN, FC_DAMP_CAUTIOUS, 7),
        (FC_METRIC_CANONICAL, FC_DAMP_CAUTIOUS, 0),
    ]:
        r_mode = minimize_box_stiefel(
            fg_trace_inplace,
            z0=np.full(d1, b1 / d1, dtype=np.float64),
            bounds=(l1, u1),
            X0_list=X1_zc,
            options=FCOptions(gtol=1.0e-9, metric=metric, damping_mode=damp, memory=mem),
            linear_constraint=(a1, b1),
            copy=True,
            in_place_grad=True,
        )
        assert r_mode.status == FC_CONVERGED_GRAD
        assert r_mode.kkt_residual <= 1.0e-9
        assert abs(float(np.sum(r_mode.z)) - b1) <= 1.0e-14

    # 3. Coupled General Signed Weights a != 1 (R = 2: St(10, 2) x SO(4)) + Shift Invariance
    d2 = 8
    n_list2 = [10, 4]
    p_list2 = [2, 4]
    l2 = np.array([0.0, 0.0, 0.35, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    u2 = np.array([1.0, 1.0, 0.35, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float64)
    a2 = np.array([1.4, 0.9, 1.2, -0.8, 1.7, -0.5, 1.1, 0.6], dtype=np.float64)
    b2 = 2.15
    targ2 = np.array([-0.4, 1.6, 0.35, 0.55, 0.75, -0.8, 0.45, 1.3], dtype=np.float64)
    w_box2 = np.arange(3.5, 3.5 + d2, dtype=np.float64)
    d_st2a = np.linspace(1.0, 4.5, 10, dtype=np.float64)
    d_st2b = np.linspace(0.5, 2.0, 4, dtype=np.float64)

    z2_0 = rng.uniform(0.1, 0.9, size=d2)
    X2a_0 = project_stiefel(np.asfortranarray(rng.standard_normal((10, 2))))
    X2b_0 = project_stiefel(np.asfortranarray(rng.standard_normal((4, 4))))
    if np.linalg.det(X2b_0) < 0.0:
        X2b_0[:, 0] *= -1.0

    def make_fg_multi(shift_mu: float):
        def _fg(
            z: np.ndarray, xs: list[np.ndarray]
        ) -> tuple[float, np.ndarray, list[np.ndarray]]:
            X_a, X_b = xs[0], xs[1]
            dz = z - targ2
            f_v = 0.5 * float(np.sum(w_box2 * dz * dz)) + shift_mu * float(np.dot(a2, z))
            gz = w_box2 * dz + shift_mu * a2

            DXa = d_st2a[:, None] * X_a
            qa = np.sum(X_a * DXa, axis=0)
            f_v += 0.5 * float(np.dot(0.5 + z[:2], qa))
            gz[:2] += 0.5 * qa
            GXa = DXa * (0.5 + z[:2])[None, :]

            Db = X_b - np.eye(4, dtype=np.float64)
            WDb = d_st2b[:, None] * Db
            qb = np.sum(Db * WDb, axis=0)
            f_v += 0.5 * float(np.dot(0.8 + 0.5 * z[4:8], qb))
            gz[4:8] += 0.25 * qb
            GXb = WDb * (0.8 + 0.5 * z[4:8])[None, :]

            # Cross-factor trilinear coupling
            tr_a = float(X_a[0, 0] + X_a[1, 1])
            tr_b = float(np.trace(X_b))
            f_v += 0.2 * float(z[5]) * tr_a * tr_b
            gz[5] += 0.2 * tr_a * tr_b
            GXa[0, 0] += 0.2 * float(z[5]) * tr_b
            GXa[1, 1] += 0.2 * float(z[5]) * tr_b
            for j in range(4):
                GXb[j, j] += 0.2 * float(z[5]) * tr_a
            return f_v, gz, [GXa, GXb]

        return _fg

    ws2 = FortchuckWorkspace(d=d2, n_list=n_list2, p_list=p_list2, memory=7)
    res2_base = minimize_box_stiefel(
        make_fg_multi(0.0),
        z0=z2_0,
        bounds=(l2, u2),
        X0_list=[X2a_0, X2b_0],
        options={"gtol": 1.0e-9},
        linear_constraint=(a2, b2),
        workspace=ws2,
        copy=True,
    )
    assert res2_base.status == FC_CONVERGED_GRAD
    assert res2_base.kkt_residual <= 1.0e-9
    assert abs(float(np.dot(a2, res2_base.z)) - b2) <= 1.0e-14
    assert res2_base.z[2] == pytest.approx(0.35, abs=1.0e-16)
    assert np.linalg.det(res2_base.X_list[1]) == pytest.approx(1.0, abs=1.0e-13)

    # Verify Rank-1 KKT Nullspace Two-Loop multiplier shift invariance: gz -> gz + 1e4 * a2
    res2_shift = minimize_box_stiefel(
        make_fg_multi(1.0e4),
        z0=z2_0,
        bounds=(l2, u2),
        X0_list=[X2a_0, X2b_0],
        options={"gtol": 1.0e-9},
        linear_constraint=(a2, b2),
        workspace=ws2,
        copy=True,
    )
    assert res2_shift.status == FC_CONVERGED_GRAD
    assert np.allclose(res2_shift.z, res2_base.z, atol=1.0e-8, rtol=0.0)
    assert abs((res2_shift.f_val - 1.0e4 * b2) - res2_base.f_val) <= 1.0e-9
