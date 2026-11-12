"""Zero-copy Python solver interface and reusable workspace for `fortchuck`.

Provides `minimize_box_stiefel()`, `FortchuckWorkspace`, `project_stiefel()`,
`kkt_residual()`, `workspace_query()`, and `default_options()` backed by the
stateless Modern Fortran 2018 C-ABI engine.
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
import math
from typing import Any, Callable, Mapping, Sequence
import numpy as np

from fortchuck._native import (
    FC_CONVERGED_FTOL,
    FC_CONVERGED_GRAD,
    FC_CONVERGED_XTOL,
    FC_ERR_CALLBACK_ABORT,
    FC_METRIC_CANONICAL,
    FC_SUCCESS_QUERY,
    FCEvalCallback,
    FCOptions,
    FCResult,
    STATUS_NAMES,
    load_native_library,
)

_C_I32_P = ctypes.POINTER(ctypes.c_int32)
_C_F64_P = ctypes.POINTER(ctypes.c_double)
_EMPTY_F64 = np.zeros(0, dtype=np.float64)
_EMPTY_F64.flags.writeable = False


@dataclass
class OptimizeResult:
    """Structured optimization result returned by `minimize_box_stiefel()`."""

    z: np.ndarray
    X_list: list[np.ndarray]
    f_val: float
    kkt_residual: float
    box_residual: float
    stiefel_residual: float
    orth_error: float
    last_step_size: float
    lwork_min: int
    liwork_min: int
    status: int
    n_iter: int
    n_eval: int
    n_reorth: int
    n_fallbacks: int
    n_active_box: int
    raw_result: FCResult

    @property
    def status_name(self) -> str:
        return STATUS_NAMES.get(self.status, f"UNKNOWN_STATUS({self.status})")

    @property
    def success(self) -> bool:
        return self.status in (FC_CONVERGED_GRAD, FC_CONVERGED_FTOL, FC_CONVERGED_XTOL)

    @property
    def converged(self) -> bool:
        return self.success

    @property
    def fun(self) -> float:
        return self.f_val

    @property
    def nit(self) -> int:
        return self.n_iter

    @property
    def nfev(self) -> int:
        return self.n_eval

    @property
    def x_list(self) -> list[np.ndarray]:
        return self.X_list

    @property
    def X(self) -> np.ndarray | list[np.ndarray] | None:
        if not self.X_list:
            return None
        if len(self.X_list) == 1:
            return self.X_list[0]
        return self.X_list


def default_options() -> FCOptions:
    """Query `fortchuck_default_options` from the C-ABI library."""
    lib = load_native_library()
    opts = FCOptions()
    lib.fortchuck_default_options(ctypes.byref(opts))
    return opts


def workspace_query(
    d: int = 0,
    n_list: Sequence[int] = (),
    p_list: Sequence[int] = (),
    memory: int = 7,
) -> tuple[int, int]:
    """Query exact `(lwork, liwork)` lengths for `(d, n_list, p_list, memory)`."""
    if len(n_list) != len(p_list):
        raise ValueError(f"Length mismatch: len(n_list)={len(n_list)} != len(p_list)={len(p_list)}")
    R = len(n_list)
    lib = load_native_library()

    if R > 0:
        n_arr = np.ascontiguousarray(n_list, dtype=np.int32)
        p_arr = np.ascontiguousarray(p_list, dtype=np.int32)
        n_ptr = n_arr.ctypes.data_as(_C_I32_P)
        p_ptr = p_arr.ctypes.data_as(_C_I32_P)
    else:
        n_ptr = None
        p_ptr = None

    lwork = ctypes.c_int64(0)
    liwork = ctypes.c_int64(0)
    st = int(
        lib.fortchuck_workspace_query(
            int(d),
            int(R),
            n_ptr,
            p_ptr,
            int(memory),
            ctypes.byref(lwork),
            ctypes.byref(liwork),
        )
    )
    if st != FC_SUCCESS_QUERY:
        name = STATUS_NAMES.get(st, str(st))
        raise ValueError(f"fortchuck_workspace_query failed with status {st} ({name})")
    return int(lwork.value), int(liwork.value)


class FortchuckWorkspace:
    """Reusable pre-allocated `(work, iwork)` workspace for zero-allocation Python loops."""

    def __init__(
        self,
        d: int = 0,
        n_list: Sequence[int] = (),
        p_list: Sequence[int] = (),
        memory: int = 7,
    ) -> None:
        lwork, liwork = workspace_query(d=d, n_list=n_list, p_list=p_list, memory=memory)
        self.d: int = int(d)
        self.R: int = len(n_list)
        self.n_arr: np.ndarray = np.ascontiguousarray(n_list, dtype=np.int32)
        self.p_arr: np.ndarray = np.ascontiguousarray(p_list, dtype=np.int32)
        self.memory: int = int(memory)
        self.lwork: int = lwork
        self.liwork: int = liwork
        self.work: np.ndarray = np.empty(lwork, dtype=np.float64)
        self.iwork: np.ndarray = np.empty(liwork, dtype=np.int32)


def _parse_bounds(
    d: int,
    bounds: Any,
) -> tuple[np.ndarray, np.ndarray]:
    if d == 0:
        return np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.float64)

    if bounds is None:
        return np.full(d, -np.inf, dtype=np.float64), np.full(d, np.inf, dtype=np.float64)

    # Support scipy.optimize.Bounds duck-typing (.lb, .ub)
    if hasattr(bounds, "lb") and hasattr(bounds, "ub"):
        lower = np.ascontiguousarray(bounds.lb, dtype=np.float64).ravel()
        upper = np.ascontiguousarray(bounds.ub, dtype=np.float64).ravel()
        if lower.size != d or upper.size != d:
            raise ValueError(f"Bounds shape mismatch: expected d={d}, got ({lower.size}, {upper.size})")
        return lower, upper

    # Distinguish `(lower_vec, upper_vec)` tuple from `[(l_0, u_0), ..., (l_{d-1}, u_{d-1})]`
    if isinstance(bounds, tuple) and len(bounds) == 2:
        b0, b1 = bounds
        is_pair_of_vectors = isinstance(b0, (np.ndarray, list)) and isinstance(b1, (np.ndarray, list))
        if is_pair_of_vectors or d != 2:
            lower = np.ascontiguousarray(
                [-np.inf if v is None else v for v in np.asarray(b0, dtype=object).ravel()],
                dtype=np.float64,
            )
            upper = np.ascontiguousarray(
                [np.inf if v is None else v for v in np.asarray(b1, dtype=object).ravel()],
                dtype=np.float64,
            )
            if lower.size != d or upper.size != d:
                raise ValueError(
                    f"Bounds length mismatch: expected d={d}, got ({lower.size}, {upper.size})"
                )
            return lower, upper

    bounds_seq = list(bounds)
    if len(bounds_seq) != d:
        raise ValueError(f"Bounds sequence length {len(bounds_seq)} does not match d={d}")
    lower = np.empty(d, dtype=np.float64)
    upper = np.empty(d, dtype=np.float64)
    for idx, pair in enumerate(bounds_seq):
        if len(pair) != 2:
            raise ValueError(f"Bound entry {idx} must be a (low, high) pair, got {pair!r}")
        lo, hi = pair
        lower[idx] = -np.inf if lo is None else float(lo)
        upper[idx] = np.inf if hi is None else float(hi)
    return lower, upper


def _normalize_x_list(
    X0_list: np.ndarray | Sequence[np.ndarray] | None,
) -> list[np.ndarray]:
    if X0_list is None:
        return []
    if isinstance(X0_list, np.ndarray):
        if X0_list.ndim == 2:
            return [X0_list]
        raise ValueError(f"Single X0 ndarray must be 2D (n, p), got shape {X0_list.shape}")
    out = list(X0_list)
    for r_idx, xr in enumerate(out):
        arr = np.asarray(xr)
        if arr.ndim != 2:
            raise ValueError(f"Stiefel factor X0_list[{r_idx}] must be 2D, got shape {arr.shape}")
    return out


def project_stiefel(
    X: np.ndarray,
    work: np.ndarray | None = None,
    *,
    copy: bool = False,
) -> np.ndarray:
    """Project `X` (`n x p`, `p <= n`) onto `St(n, p)` via positive-diagonal Householder QR.

    Uses zero-copy in-place projection when `copy=False` and `X` is already a
    writeable Fortran-contiguous (`order="F"`) `float64` NumPy array.
    """
    lib = load_native_library()
    arr = np.asarray(X)
    if arr.ndim != 2:
        raise ValueError(f"X must be a 2D array (n, p), got shape {arr.shape}")
    n, p = int(arr.shape[0]), int(arr.shape[1])

    if (
        not copy
        and isinstance(X, np.ndarray)
        and X.dtype == np.float64
        and X.flags.f_contiguous
        and X.flags.writeable
    ):
        out = X
    else:
        out = np.array(arr, dtype=np.float64, order="F", copy=True)

    if work is None:
        q_buf = np.empty(1, dtype=np.float64)
        st_q = int(
            lib.fortchuck_project_stiefel(n, p, None, q_buf.ctypes.data_as(_C_F64_P), -1)
        )
        if st_q != FC_SUCCESS_QUERY:
            name = STATUS_NAMES.get(st_q, str(st_q))
            raise ValueError(f"fortchuck_project_stiefel query failed with status {st_q} ({name})")
        work_buf = np.empty(max(1, int(q_buf[0])), dtype=np.float64)
    else:
        if work.dtype != np.float64 or not (work.flags.c_contiguous or work.flags.f_contiguous):
            raise ValueError("work must be a contiguous float64 NumPy array")
        work_buf = work

    st = int(
        lib.fortchuck_project_stiefel(
            n,
            p,
            out.ctypes.data_as(_C_F64_P),
            work_buf.ctypes.data_as(_C_F64_P),
            int(work_buf.size),
        )
    )
    if st != 0:
        name = STATUS_NAMES.get(st, str(st))
        raise ValueError(f"fortchuck_project_stiefel failed with status {st} ({name})")
    return out


def kkt_residual(
    z: np.ndarray | Sequence[float] | None = None,
    bounds: Any = None,
    X_list: np.ndarray | Sequence[np.ndarray] | None = None,
    gz: np.ndarray | Sequence[float] | None = None,
    GX_list: np.ndarray | Sequence[np.ndarray] | None = None,
    *,
    metric: int = FC_METRIC_CANONICAL,
    work: np.ndarray | None = None,
) -> tuple[float, float]:
    """Compute exact `(kkt_residual, orth_error)` via `fortchuck_kkt_residual`."""
    lib = load_native_library()

    if z is None:
        z_arr = np.zeros(0, dtype=np.float64)
    else:
        z_arr = np.ascontiguousarray(z, dtype=np.float64).ravel()
    d = int(z_arr.size)

    xs_raw = _normalize_x_list(X_list)
    R = len(xs_raw)
    if d == 0 and R == 0:
        raise ValueError("At least one of z (d > 0) or X_list (R > 0) must be provided")

    if d > 0:
        if gz is None:
            raise ValueError("gz must be provided when d > 0")
        gz_arr = np.ascontiguousarray(gz, dtype=np.float64).ravel()
        if gz_arr.size != d:
            raise ValueError(f"gz length {gz_arr.size} does not match d={d}")
        l_arr, u_arr = _parse_bounds(d, bounds)
        z_ptr = z_arr.ctypes.data_as(_C_F64_P)
        l_ptr = l_arr.ctypes.data_as(_C_F64_P)
        u_ptr = u_arr.ctypes.data_as(_C_F64_P)
        gz_ptr = gz_arr.ctypes.data_as(_C_F64_P)
    else:
        z_ptr = l_ptr = u_ptr = gz_ptr = None

    if R > 0:
        gxs_raw = _normalize_x_list(GX_list)
        if len(gxs_raw) != R:
            raise ValueError(f"GX_list length {len(gxs_raw)} does not match R={R}")
        n_arr = np.array([x.shape[0] for x in xs_raw], dtype=np.int32)
        p_arr = np.array([x.shape[1] for x in xs_raw], dtype=np.int32)
        for r_idx, (xr, gxr) in enumerate(zip(xs_raw, gxs_raw)):
            if xr.shape != gxr.shape:
                raise ValueError(
                    f"GX_list[{r_idx}] shape {gxr.shape} != X_list[{r_idx}] shape {xr.shape}"
                )
        if (
            R == 1
            and isinstance(xs_raw[0], np.ndarray)
            and xs_raw[0].dtype == np.float64
            and xs_raw[0].flags.f_contiguous
        ):
            X_packed = xs_raw[0]
        else:
            X_packed = np.concatenate([ np.asarray(x, dtype=np.float64).ravel(order="F") for x in xs_raw ])
        if (
            R == 1
            and isinstance(gxs_raw[0], np.ndarray)
            and gxs_raw[0].dtype == np.float64
            and gxs_raw[0].flags.f_contiguous
        ):
            GX_packed = gxs_raw[0]
        else:
            GX_packed = np.concatenate([ np.asarray(gx, dtype=np.float64).ravel(order="F") for gx in gxs_raw ])

        n_ptr = n_arr.ctypes.data_as(_C_I32_P)
        p_ptr = p_arr.ctypes.data_as(_C_I32_P)
        X_ptr = X_packed.ctypes.data_as(_C_F64_P)
        GX_ptr = GX_packed.ctypes.data_as(_C_F64_P)
    else:
        n_ptr = p_ptr = None
        X_ptr = GX_ptr = None

    if work is None:
        q_buf = np.empty(1, dtype=np.float64)
        st_q = int(
            lib.fortchuck_kkt_residual(
                d,
                R,
                n_ptr,
                p_ptr,
                None,
                None,
                None,
                None,
                None,
                None,
                int(metric),
                q_buf.ctypes.data_as(_C_F64_P),
                -1,
                None,
                None,
            )
        )
        if st_q != FC_SUCCESS_QUERY:
            name = STATUS_NAMES.get(st_q, str(st_q))
            raise ValueError(f"fortchuck_kkt_residual query failed with status {st_q} ({name})")
        work_buf = np.empty(max(1, int(q_buf[0])), dtype=np.float64)
    else:
        if work.dtype != np.float64 or not (work.flags.c_contiguous or work.flags.f_contiguous):
            raise ValueError("work must be a contiguous float64 NumPy array")
        work_buf = work

    kkt_inf = ctypes.c_double(0.0)
    orth_err = ctypes.c_double(0.0)
    st = int(
        lib.fortchuck_kkt_residual(
            d,
            R,
            n_ptr,
            p_ptr,
            z_ptr,
            l_ptr,
            u_ptr,
            X_ptr,
            gz_ptr,
            GX_ptr,
            int(metric),
            work_buf.ctypes.data_as(_C_F64_P),
            int(work_buf.size),
            ctypes.byref(kkt_inf),
            ctypes.byref(orth_err),
        )
    )
    if st != 0:
        name = STATUS_NAMES.get(st, str(st))
        raise ValueError(f"fortchuck_kkt_residual failed with status {st} ({name})")
    return float(kkt_inf.value), float(orth_err.value)


def minimize_box_stiefel(
    fun_and_grad: Callable[..., Any],
    z0: np.ndarray | Sequence[float] | None = None,
    bounds: Any = None,
    X0_list: np.ndarray | Sequence[np.ndarray] | None = None,
    options: FCOptions | Mapping[str, Any] | None = None,
    work: np.ndarray | None = None,
    iwork: np.ndarray | None = None,
    *,
    workspace: FortchuckWorkspace | None = None,
    copy: bool = False,
    in_place_grad: bool = False,
) -> OptimizeResult:
    """Minimize `f(z, X_1, ..., X_R)` over `[l, u]^d x St(n_1, p_1) x ... x St(n_R, p_R)`.

    Parameters
    ----------
    fun_and_grad : callable
        - When `in_place_grad=False` (default): `fun_and_grad(z, X_list)` returning
          `(f, gz, GX_list)` or `(f, gz, GX_list, rc)` where `rc == 0` on success,
          `rc > 0` on recoverable domain barrier (triggers step halving), or `rc < 0`
          to abort with `FC_ERR_CALLBACK_ABORT`.
        - When `in_place_grad=True`: `fun_and_grad(z, X_list, gz_out, GX_out_list)`
          populating zero-copy Fortran views `gz_out` and `GX_out_list` in-place and
          returning `f` or `(f, rc)`.
    z0 : array_like or None
        Initial box vector of length `d` (`None` or empty for pure-Stiefel `d = 0`).
        Mutated in-place with zero copies when `copy=False` and `z0` is a writeable
        1D contiguous `float64` NumPy array.
    bounds : tuple `(lower, upper)`, list of `(lo, hi)`, `scipy.optimize.Bounds`, or None
        Box bounds for `z` (`None` defaults to `(-inf, +inf)`).
    X0_list : 2D ndarray, sequence of 2D ndarrays, or None
        Initial Stiefel factor(s) `(n_r, p_r)`. When `R == 1`, `copy=False`, and `X0`
        is a writeable Fortran-contiguous (`order="F"`) `float64` array, `X0` is passed
        directly to Fortran with **zero copies** (`res.X_list[0] is X0`).
    options : `FCOptions`, mapping of overrides, or None
        Solver configuration (`FCOptions()` defaults used when `None`).
    work, iwork : ndarray or None
        Optional caller-allocated `float64` / `int32` scratch buffers.
    workspace : `FortchuckWorkspace` or None
        Optional reusable pre-allocated workspace object.
    copy : bool, default False
        If True, always copy `z0` and `X0_list` rather than mutating inputs in-place.
    in_place_grad : bool, default False
        If True, pass `(z, X_list, gz_out, GX_out_list)` zero-copy views to `fun_and_grad`.
    """
    lib = load_native_library()

    if options is None:
        opts_c = FCOptions()
    elif isinstance(options, FCOptions):
        opts_c = options
    elif isinstance(options, Mapping):
        opts_c = FCOptions(**options)
    else:
        raise TypeError(f"options must be FCOptions, dict, or None; got {type(options).__name__}")

    # Prepare box iterate z and bounds [l, u]
    if z0 is None:
        d = 0
        z_buf = np.zeros(0, dtype=np.float64)
        l_arr = np.zeros(0, dtype=np.float64)
        u_arr = np.zeros(0, dtype=np.float64)
    else:
        z_raw = np.asarray(z0)
        if z_raw.size == 0:
            d = 0
            z_buf = np.zeros(0, dtype=np.float64)
            l_arr = np.zeros(0, dtype=np.float64)
            u_arr = np.zeros(0, dtype=np.float64)
        else:
            if (
                not copy
                and isinstance(z0, np.ndarray)
                and z0.dtype == np.float64
                and z0.ndim == 1
                and (z0.flags.c_contiguous or z0.flags.f_contiguous)
                and z0.flags.writeable
            ):
                z_buf = z0
            else:
                z_buf = np.array(z_raw, dtype=np.float64, copy=True).ravel()
            d = int(z_buf.size)
            l_arr, u_arr = _parse_bounds(d, bounds)

    # Prepare Stiefel factors X_1 .. X_R
    xs_in = _normalize_x_list(X0_list)
    R = len(xs_in)

    factor_specs: list[tuple[int, int, int, int]] = []  # (offset, size, n_r, p_r)
    n_st_tot = 0
    if R > 0:
        n_arr = np.empty(R, dtype=np.int32)
        p_arr = np.empty(R, dtype=np.int32)
        for r_idx, xr in enumerate(xs_in):
            nr, pr = int(xr.shape[0]), int(xr.shape[1])
            n_arr[r_idx] = nr
            p_arr[r_idx] = pr
            sz = nr * pr
            factor_specs.append((n_st_tot, sz, nr, pr))
            n_st_tot += sz

        # True zero-copy fast path when R == 1 and X0 is writeable float64 order='F'
        if (
            R == 1
            and not copy
            and isinstance(xs_in[0], np.ndarray)
            and xs_in[0].dtype == np.float64
            and xs_in[0].flags.f_contiguous
            and xs_in[0].flags.writeable
        ):
            X_packed = xs_in[0]
            zero_copy_single_x = True
        else:
            zero_copy_single_x = False
            X_packed = np.empty(max(1, n_st_tot), dtype=np.float64)
            for (off, sz, nr, pr), xr in zip(factor_specs, xs_in):
                if sz > 0:
                    X_packed[off : off + sz] = np.asarray(xr, dtype=np.float64).ravel(order="F")
    else:
        zero_copy_single_x = False
        n_arr = np.zeros(0, dtype=np.int32)
        p_arr = np.zeros(0, dtype=np.int32)
        X_packed = np.zeros(0, dtype=np.float64)

    # Resolve work and iwork buffers
    if workspace is not None:
        if work is not None or iwork is not None:
            raise ValueError("Pass either `workspace` or `(work, iwork)`, not both")
        work_buf = workspace.work
        iwork_buf = workspace.iwork
    elif work is not None or iwork is not None:
        if work is None or iwork is None:
            raise ValueError("Both `work` and `iwork` must be provided together when explicit")
        if work.dtype != np.float64 or not (work.flags.c_contiguous or work.flags.f_contiguous):
            raise ValueError("work must be a contiguous float64 NumPy array")
        if iwork.dtype != np.int32 or not (iwork.flags.c_contiguous or iwork.flags.f_contiguous):
            raise ValueError("iwork must be a contiguous int32 NumPy array")
        work_buf = work
        iwork_buf = iwork
    else:
        n_ptr_q = n_arr.ctypes.data_as(_C_I32_P) if R > 0 else None
        p_ptr_q = p_arr.ctypes.data_as(_C_I32_P) if R > 0 else None
        lw_q = ctypes.c_int64(0)
        liw_q = ctypes.c_int64(0)
        st_q = int(
            lib.fortchuck_workspace_query(
                d,
                R,
                n_ptr_q,
                p_ptr_q,
                int(opts_c.memory),
                ctypes.byref(lw_q),
                ctypes.byref(liw_q),
            )
        )
        if st_q == FC_SUCCESS_QUERY:
            work_buf = np.empty(max(1, int(lw_q.value)), dtype=np.float64)
            iwork_buf = np.empty(max(1, int(liw_q.value)), dtype=np.int32)
        else:
            # Let fortchuck_solve produce the deterministic error code & FCResult
            work_buf = np.empty(1, dtype=np.float64)
            iwork_buf = np.empty(1, dtype=np.int32)

    captured_exc: list[BaseException] = []

    @FCEvalCallback
    def _c_callback(
        d_cb: int,
        R_cb: int,
        _n_ptr: Any,
        _p_ptr: Any,
        z_ptr: Any,
        X_ptr: Any,
        f_out_ptr: Any,
        gz_out_ptr: Any,
        GX_out_ptr: Any,
        _udata: Any,
    ) -> int:
        try:
            if d_cb > 0:
                z_view = np.ctypeslib.as_array(z_ptr, shape=(d_cb,))
                z_view.flags.writeable = False
                gz_view = np.ctypeslib.as_array(gz_out_ptr, shape=(d_cb,))
            else:
                z_view = _EMPTY_F64
                gz_view = _EMPTY_F64

            if R_cb > 0:
                X_flat = np.ctypeslib.as_array(X_ptr, shape=(n_st_tot,))
                GX_flat = np.ctypeslib.as_array(GX_out_ptr, shape=(n_st_tot,))
                X_views: list[np.ndarray] = []
                GX_views: list[np.ndarray] = []
                for off, sz, nr, pr in factor_specs:
                    xv = X_flat[off : off + sz].reshape((nr, pr), order="F")
                    xv.flags.writeable = False
                    X_views.append(xv)
                    GX_views.append(GX_flat[off : off + sz].reshape((nr, pr), order="F"))
            else:
                X_views = []
                GX_views = []

            if in_place_grad:
                ret = fun_and_grad(z_view, X_views, gz_view, GX_views)
                if isinstance(ret, tuple):
                    f_val, rc_val = float(ret[0]), int(ret[1])
                else:
                    f_val, rc_val = float(ret), 0
                f_out_ptr[0] = f_val
                return rc_val

            ret = fun_and_grad(z_view, X_views)
            if not isinstance(ret, tuple) or len(ret) not in (3, 4):
                raise TypeError(
                    "fun_and_grad(z, X_list) must return (f, gz, GX_list) or (f, gz, GX_list, rc)"
                )
            if len(ret) == 4:
                f_raw, gz_ret, gx_ret, rc_raw = ret
                rc_val = int(rc_raw)
            else:
                f_raw, gz_ret, gx_ret = ret
                rc_val = 0

            f_val = float(f_raw)
            f_out_ptr[0] = f_val

            if rc_val != 0:
                return rc_val

            if d_cb > 0:
                if gz_ret is None:
                    if not math.isfinite(f_val):
                        gz_view.fill(0.0)
                    else:
                        raise ValueError("fun_and_grad returned gz=None when d > 0 and f is finite")
                else:
                    gz_view[:] = np.asarray(gz_ret, dtype=np.float64).ravel()

            if R_cb > 0:
                if gx_ret is None:
                    if not math.isfinite(f_val):
                        for gxv in GX_views:
                            gxv.fill(0.0)
                    else:
                        raise ValueError(
                            "fun_and_grad returned GX_list=None when R > 0 and f is finite"
                        )
                else:
                    if R_cb == 1 and isinstance(gx_ret, np.ndarray) and gx_ret.ndim == 2:
                        GX_views[0][:] = gx_ret
                    else:
                        if len(gx_ret) != R_cb:
                            raise ValueError(
                                f"fun_and_grad returned {len(gx_ret)} Stiefel gradients, expected R={R_cb}"
                            )
                        for gxv, gx_item in zip(GX_views, gx_ret):
                            gxv[:] = gx_item

            return 0
        except BaseException as exc:
            captured_exc.append(exc)
            return -1

    res_c = FCResult()
    n_ptr = n_arr.ctypes.data_as(_C_I32_P) if R > 0 else None
    p_ptr = p_arr.ctypes.data_as(_C_I32_P) if R > 0 else None
    z_ptr = z_buf.ctypes.data_as(_C_F64_P) if d > 0 else None
    l_ptr = l_arr.ctypes.data_as(_C_F64_P) if d > 0 else None
    u_ptr = u_arr.ctypes.data_as(_C_F64_P) if d > 0 else None
    X_ptr = X_packed.ctypes.data_as(_C_F64_P) if R > 0 else None

    status = int(
        lib.fortchuck_solve(
            d,
            R,
            n_ptr,
            p_ptr,
            z_ptr,
            l_ptr,
            u_ptr,
            X_ptr,
            _c_callback,
            None,
            ctypes.byref(opts_c),
            work_buf.ctypes.data_as(_C_F64_P),
            int(work_buf.size),
            iwork_buf.ctypes.data_as(_C_I32_P),
            int(iwork_buf.size),
            ctypes.byref(res_c),
        )
    )

    if captured_exc:
        raise captured_exc[0]

    # Assemble output Stiefel views
    out_x_list: list[np.ndarray] = []
    if R > 0:
        if zero_copy_single_x:
            out_x_list.append(X_packed)
        else:
            for (off, sz, nr, pr), xr_orig in zip(factor_specs, xs_in):
                view_f = X_packed[off : off + sz].reshape((nr, pr), order="F")
                if (
                    not copy
                    and isinstance(xr_orig, np.ndarray)
                    and xr_orig.dtype == np.float64
                    and xr_orig.flags.writeable
                    and xr_orig.shape == (nr, pr)
                ):
                    xr_orig[:] = view_f
                    if xr_orig.flags.f_contiguous:
                        out_x_list.append(xr_orig)
                    else:
                        out_x_list.append(view_f)
                else:
                    out_x_list.append(view_f)

    return OptimizeResult(
        z=z_buf,
        X_list=out_x_list,
        f_val=float(res_c.f_val),
        kkt_residual=float(res_c.kkt_residual),
        box_residual=float(res_c.box_residual),
        stiefel_residual=float(res_c.stiefel_residual),
        orth_error=float(res_c.orth_error),
        last_step_size=float(res_c.last_step_size),
        lwork_min=int(res_c.lwork_min),
        liwork_min=int(res_c.liwork_min),
        status=status if status != int(res_c.status) else int(res_c.status),
        n_iter=int(res_c.n_iter),
        n_eval=int(res_c.n_eval),
        n_reorth=int(res_c.n_reorth),
        n_fallbacks=int(res_c.n_fallbacks),
        n_active_box=int(res_c.n_active_box),
        raw_result=res_c,
    )
