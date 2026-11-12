"""Fair external baseline solver wrappers for the fortchuck blind benchmark suite.

Implements:
  - `SciPy-L-BFGS-B`: Native Fortran L-BFGS-B on pure-box (`B7A`) and 3-stage
    quadratic Stiefel penalty continuation (`rho in {1e2, 1e4, 1e6}`) when `R >= 1`.
  - `SciPy-SLSQP`: Constrained SQP with `Bounds(l, u)` and analytical `vech(X^T X - I) = 0`
    equality constraints + analytical constraint Jacobian (`N_tot <= 450`, `20s` timeout).
  - `SciPy-trust-constr`: Interior-point trust-region NLP with `BFGS()` Hessian,
    analytical Stiefel equality Jacobian, and exact analytical constraint Hessian
    (`N_tot <= 450`, `20s` timeout).
  - `pymanopt-Warped-CG`: `pymanopt 2.2.1` Riemannian `ConjugateGradient` on
    `Product([Euclidean(d), Stiefel(...)])` via smooth `tanh`/`exp` box warps,
    and native `Stiefel` / `Product([Stiefel(...)])` when `d == 0` (`B7B`).
  - `pymanopt-TrustRegions`: `pymanopt 2.2.1` Riemannian `TrustRegions` with
    cached first-order directional finite-difference Euclidean HVP (`RTR-FD`).
  - `Alternating-BCD`: Strictly feasible alternating minimization (`10` inner
    `SciPy L-BFGS-B` steps on `z` <-> `10` inner `pymanopt CG` steps on `X_{1..R}`).
"""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Callable, Sequence
import warnings
import numpy as np
import pymanopt
from pymanopt.manifolds import Euclidean, Product, Stiefel
from pymanopt.optimizers import ConjugateGradient, TrustRegions
from pymanopt.optimizers.line_search import AdaptiveLineSearcher
from scipy.optimize import BFGS, Bounds, NonlinearConstraint, minimize

from benchmarks.problems import BenchmarkProblem
from benchmarks.referee import (
    BudgetExhaustedError,
    EvalOracle,
    RefereeAudit,
    TargetKKTReachedError,
    WallTimeoutError,
    evaluate_kkt_and_feasibility,
)

EXTERNAL_BASELINE_NAMES: tuple[str, ...] = (
    "Alternating-BCD",
    "pymanopt-Warped-CG",
    "pymanopt-TrustRegions",
    "SciPy-L-BFGS-B",
    "SciPy-SLSQP",
    "SciPy-trust-constr",
)


@dataclass(frozen=True)
class BaselineResult:
    """Standardized execution record for any solver on one `(problem, seed)` pair."""

    solver_name: str
    problem_id: str
    seed: int
    status: str
    skipped: bool
    n_eval: int
    wall_ms: float
    z: np.ndarray
    x_list: list[np.ndarray]
    audit: RefereeAudit


class SmoothBoxWarp:
    """Smooth C^infinity diffeomorphism mapping R^d onto open box interior (plus fixed coords).

    - Two-sided finite `[l_i, u_i]`: `z_i = m_i + r_i * tanh(theta_i)`
    - Lower-bounded `[l_i, +inf)`:   `z_i = l_i + exp(theta_i)`
    - Upper-bounded `(-inf, u_i]`:   `z_i = u_i - exp(theta_i)`
    - Unbounded `(-inf, +inf)`:      `z_i = theta_i`
    - Fixed `l_i == u_i`:            `z_i = l_i` (`dz_i / dtheta_i = 0`)
    """

    def __init__(self, lower: np.ndarray, upper: np.ndarray) -> None:
        self.d = int(lower.size)
        self.lower = np.asarray(lower, dtype=np.float64)
        self.upper = np.asarray(upper, dtype=np.float64)

        l_fin = np.isfinite(self.lower)
        u_fin = np.isfinite(self.upper)

        self.mask_fixed = self.lower == self.upper
        self.mask_both = (~self.mask_fixed) & l_fin & u_fin
        self.mask_lo = (~self.mask_fixed) & l_fin & (~u_fin)
        self.mask_hi = (~self.mask_fixed) & (~l_fin) & u_fin
        self.mask_free = (~self.mask_fixed) & (~l_fin) & (~u_fin)

        self.mid = np.zeros(self.d, dtype=np.float64)
        self.rad = np.ones(self.d, dtype=np.float64)
        if np.any(self.mask_both):
            lb = self.lower[self.mask_both]
            ub = self.upper[self.mask_both]
            self.mid[self.mask_both] = 0.5 * lb + 0.5 * ub
            self.rad[self.mask_both] = 0.5 * ub - 0.5 * lb

    def forward_and_jac(self, theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Compute `(z(theta), dz/dtheta)` without overflow for arbitrary `theta in R^d`."""
        z = np.empty(self.d, dtype=np.float64)
        jac = np.zeros(self.d, dtype=np.float64)

        if np.any(self.mask_fixed):
            z[self.mask_fixed] = self.lower[self.mask_fixed]

        if np.any(self.mask_both):
            th = theta[self.mask_both]
            t = np.tanh(th)
            r = self.rad[self.mask_both]
            z[self.mask_both] = np.clip(
                self.mid[self.mask_both] + r * t,
                self.lower[self.mask_both],
                self.upper[self.mask_both],
            )
            jac[self.mask_both] = r * (1.0 - t * t)

        if np.any(self.mask_lo):
            th = np.clip(theta[self.mask_lo], -500.0, 500.0)
            ex = np.exp(th)
            z[self.mask_lo] = self.lower[self.mask_lo] + ex
            jac[self.mask_lo] = ex

        if np.any(self.mask_hi):
            th = np.clip(theta[self.mask_hi], -500.0, 500.0)
            ex = np.exp(th)
            z[self.mask_hi] = self.upper[self.mask_hi] - ex
            jac[self.mask_hi] = -ex

        if np.any(self.mask_free):
            z[self.mask_free] = theta[self.mask_free]
            jac[self.mask_free] = 1.0

        return z, jac

    def inverse(self, z0: np.ndarray) -> np.ndarray:
        """Map feasible interior point `z0` to unconstrained `theta0`."""
        theta0 = np.zeros(self.d, dtype=np.float64)
        if np.any(self.mask_both):
            scaled = (z0[self.mask_both] - self.mid[self.mask_both]) / self.rad[self.mask_both]
            scaled = np.clip(scaled, -1.0 + 1.0e-12, 1.0 - 1.0e-12)
            theta0[self.mask_both] = np.arctanh(scaled)
        if np.any(self.mask_lo):
            diff = np.maximum(z0[self.mask_lo] - self.lower[self.mask_lo], 1.0e-12)
            theta0[self.mask_lo] = np.log(diff)
        if np.any(self.mask_hi):
            diff = np.maximum(self.upper[self.mask_hi] - z0[self.mask_hi], 1.0e-12)
            theta0[self.mask_hi] = np.log(diff)
        if np.any(self.mask_free):
            theta0[self.mask_free] = z0[self.mask_free]
        return theta0


def _pack_flat(z: np.ndarray, x_list: Sequence[np.ndarray]) -> np.ndarray:
    parts = [z.ravel()] + [x.ravel(order="F") for x in x_list]
    return np.concatenate(parts) if parts else np.zeros(0, dtype=np.float64)


def _unpack_flat(
    flat: np.ndarray, problem: BenchmarkProblem
) -> tuple[np.ndarray, list[np.ndarray]]:
    z = np.array(flat[: problem.d], dtype=np.float64, copy=True)
    offset = problem.d
    xs: list[np.ndarray] = []
    for n_r, p_r in zip(problem.n_list, problem.p_list):
        sz = n_r * p_r
        chunk = flat[offset : offset + sz]
        xs.append(np.array(chunk, dtype=np.float64, copy=True).reshape((n_r, p_r), order="F"))
        offset += sz
    return z, xs


def _finalize_baseline_result(
    solver_name: str,
    problem: BenchmarkProblem,
    seed: int,
    status: str,
    oracle: EvalOracle,
    wall_ms: float,
    cand_z: np.ndarray,
    cand_xs: list[np.ndarray],
    gtol: float,
    feas_tol: float,
) -> BaselineResult:
    if oracle.target_z is not None and oracle.target_xs is not None:
        z_fin = oracle.target_z
        xs_fin = oracle.target_xs
    else:
        z_fin = cand_z
        xs_fin = cand_xs
        aud_cand = evaluate_kkt_and_feasibility(
            problem, z_fin, xs_fin, kkt_tol=gtol, feas_tol=feas_tol
        )
        if (
            not aud_cand.solved
            and oracle.best_z is not None
            and oracle.best_xs is not None
            and (not np.isfinite(aud_cand.f_val) or max(aud_cand.v_box, aud_cand.v_orth) > 1.0e-4)
        ):
            z_fin = oracle.best_z
            xs_fin = oracle.best_xs

    audit = evaluate_kkt_and_feasibility(
        problem, z_fin, xs_fin, kkt_tol=gtol, feas_tol=feas_tol
    )
    return BaselineResult(
        solver_name=solver_name,
        problem_id=problem.problem_id,
        seed=seed,
        status=status,
        skipped=False,
        n_eval=oracle.n_eval,
        wall_ms=wall_ms,
        z=z_fin,
        x_list=xs_fin,
        audit=audit,
    )


def run_scipy_lbfgsb(
    problem: BenchmarkProblem,
    z0: np.ndarray,
    x0_list: Sequence[np.ndarray],
    seed: int = 101,
    gtol: float = 1.0e-6,
    feas_tol: float = 1.0e-8,
    max_eval: int = 5000,
    timeout_sec: float = 20.0,
) -> BaselineResult:
    """Run SciPy L-BFGS-B (native on R=0; 3-stage quadratic Stiefel penalty on R>=1)."""
    oracle = EvalOracle(
        problem,
        max_eval=max_eval,
        timeout_sec=timeout_sec,
        enforce_budget_exception=True,
        stop_on_kkt_tol=gtol,
        feas_tol=feas_tol,
    )
    x_cur = _pack_flat(z0, x0_list)
    lb = np.concatenate(
        [problem.lower, np.full(problem.n_stiefel_tot, -np.inf, dtype=np.float64)]
    )
    ub = np.concatenate(
        [problem.upper, np.full(problem.n_stiefel_tot, np.inf, dtype=np.float64)]
    )
    bounds = Bounds(lb, ub)

    t0 = time.perf_counter()
    status = "COMPLETED"

    try:
        if problem.R == 0:

            def fg_pure_box(x_vec: np.ndarray) -> tuple[float, np.ndarray]:
                f_val, gz, _ = oracle.eval_fg(x_vec, [])
                if not (np.isfinite(f_val) and np.all(np.isfinite(gz))):
                    return 1.0e100, np.zeros_like(x_vec)
                return f_val, gz

            res = minimize(
                fg_pure_box,
                x_cur,
                method="L-BFGS-B",
                jac=True,
                bounds=bounds,
                options={
                    "gtol": gtol * 0.1,
                    "ftol": 0.0,
                    "maxfun": max_eval,
                    "maxiter": max_eval,
                },
            )
            x_cur = np.asarray(res.x, dtype=np.float64)
            status = "CONVERGED" if res.success else f"STOP_{res.status}"
        else:
            rho_schedule = (1.0e2, 1.0e4, 1.0e6)
            stage_tols = (1.0e-4, 1.0e-5, gtol * 0.1)
            for s_idx, (rho, s_tol) in enumerate(zip(rho_schedule, stage_tols)):
                rem = max_eval - oracle.n_eval
                if rem <= 0:
                    status = "MAX_EVAL"
                    break
                stage_cap = rem if s_idx == len(rho_schedule) - 1 else max(25, rem // (3 - s_idx))

                def fg_pen(x_vec: np.ndarray, rho_val: float = rho) -> tuple[float, np.ndarray]:
                    z_v, xs_v = _unpack_flat(x_vec, problem)
                    f_val, gz, gxs = oracle.eval_fg(z_v, xs_v)
                    if not np.isfinite(f_val):
                        return 1.0e100, np.zeros_like(x_vec)
                    f_pen = f_val
                    gxs_pen: list[np.ndarray] = []
                    for r, X_r in enumerate(xs_v):
                        p_r = problem.p_list[r]
                        err_orth = X_r.T @ X_r - np.eye(p_r, dtype=np.float64)
                        f_pen += 0.25 * rho_val * float(np.sum(err_orth * err_orth))
                        gxs_pen.append(gxs[r] + rho_val * (X_r @ err_orth))
                    return float(f_pen), _pack_flat(gz, gxs_pen)

                res = minimize(
                    fg_pen,
                    x_cur,
                    method="L-BFGS-B",
                    jac=True,
                    bounds=bounds,
                    options={
                        "gtol": s_tol,
                        "ftol": 0.0,
                        "maxfun": stage_cap,
                        "maxiter": stage_cap,
                    },
                )
                x_cur = np.asarray(res.x, dtype=np.float64)
                status = "CONVERGED" if res.success else f"STAGE{s_idx + 1}_{res.status}"
    except TargetKKTReachedError:
        status = "TARGET_KKT_REACHED"
    except BudgetExhaustedError:
        status = "MAX_EVAL_REACHED"
    except WallTimeoutError:
        status = "TIMEOUT"

    wall_ms = 1000.0 * (time.perf_counter() - t0)
    z_fin, xs_fin = _unpack_flat(x_cur, problem)
    return _finalize_baseline_result(
        "SciPy-L-BFGS-B", problem, seed, status, oracle, wall_ms, z_fin, xs_fin, gtol, feas_tol
    )


def _build_stiefel_vech_constraints(
    problem: BenchmarkProblem, free_idx: np.ndarray, fixed_vals: np.ndarray
) -> tuple[
    int,
    Callable[[np.ndarray], np.ndarray],
    Callable[[np.ndarray], np.ndarray],
    Callable[[np.ndarray, np.ndarray], np.ndarray],
]:
    """Construct orthonormalized vech(X_r^T X_r - I) = 0 constraints + exact Jac & Hess."""
    n_full = problem.n_tot
    n_free = int(free_idx.size)
    m_eq = int(sum((p_r * (p_r + 1)) // 2 for p_r in problem.p_list))
    inv_sqrt2 = float(1.0 / np.sqrt(2.0))

    # Precompute Stiefel block starting offsets in full packed vector
    block_starts: list[int] = []
    cur = problem.d
    for n_r, p_r in zip(problem.n_list, problem.p_list):
        block_starts.append(cur)
        cur += n_r * p_r

    def expand_full(x_red: np.ndarray) -> np.ndarray:
        if n_free == n_full:
            return x_red
        full = fixed_vals.copy()
        full[free_idx] = x_red
        return full

    def eq_fun(x_red: np.ndarray) -> np.ndarray:
        x_full = expand_full(x_red)
        _, xs = _unpack_flat(x_full, problem)
        c_out = np.empty(m_eq, dtype=np.float64)
        pos = 0
        for r, X_r in enumerate(xs):
            p_r = problem.p_list[r]
            gram = X_r.T @ X_r
            for b in range(p_r):
                for a in range(b + 1):
                    if a == b:
                        c_out[pos] = 0.5 * (gram[a, a] - 1.0)
                    else:
                        c_out[pos] = inv_sqrt2 * gram[a, b]
                    pos += 1
        return c_out

    def eq_jac(x_red: np.ndarray) -> np.ndarray:
        x_full = expand_full(x_red)
        _, xs = _unpack_flat(x_full, problem)
        jac_full = np.zeros((m_eq, n_full), dtype=np.float64)
        pos = 0
        for r, X_r in enumerate(xs):
            n_r = problem.n_list[r]
            p_r = problem.p_list[r]
            base = block_starts[r]
            for b in range(p_r):
                col_b = slice(base + b * n_r, base + (b + 1) * n_r)
                for a in range(b + 1):
                    if a == b:
                        jac_full[pos, col_b] = X_r[:, a]
                    else:
                        col_a = slice(base + a * n_r, base + (a + 1) * n_r)
                        jac_full[pos, col_a] = inv_sqrt2 * X_r[:, b]
                        jac_full[pos, col_b] = inv_sqrt2 * X_r[:, a]
                    pos += 1
        return jac_full if n_free == n_full else jac_full[:, free_idx]

    def eq_hess(x_red: np.ndarray, v_mult: np.ndarray) -> np.ndarray:
        del x_red
        hess_full = np.zeros((n_full, n_full), dtype=np.float64)
        pos = 0
        for r, (n_r, p_r) in enumerate(zip(problem.n_list, problem.p_list)):
            base = block_starts[r]
            eye_n = np.eye(n_r, dtype=np.float64)
            for b in range(p_r):
                sb = slice(base + b * n_r, base + (b + 1) * n_r)
                for a in range(b + 1):
                    vm = float(v_mult[pos])
                    if vm != 0.0:
                        if a == b:
                            hess_full[sb, sb] += vm * eye_n
                        else:
                            sa = slice(base + a * n_r, base + (a + 1) * n_r)
                            w = inv_sqrt2 * vm * eye_n
                            hess_full[sa, sb] += w
                            hess_full[sb, sa] += w
                    pos += 1
        if n_free == n_full:
            return hess_full
        return hess_full[np.ix_(free_idx, free_idx)]

    return m_eq, eq_fun, eq_jac, eq_hess


def _run_scipy_constrained_nlp(
    method_name: str,
    scipy_method: str,
    problem: BenchmarkProblem,
    z0: np.ndarray,
    x0_list: Sequence[np.ndarray],
    seed: int = 101,
    gtol: float = 1.0e-6,
    feas_tol: float = 1.0e-8,
    max_eval: int = 5000,
    timeout_sec: float = 20.0,
    max_n_tot: int = 450,
) -> BaselineResult:
    if problem.n_tot > max_n_tot:
        audit_init = evaluate_kkt_and_feasibility(
            problem, z0, x0_list, kkt_tol=gtol, feas_tol=feas_tol
        )
        return BaselineResult(
            solver_name=method_name,
            problem_id=problem.problem_id,
            seed=seed,
            status=f"SKIPPED_NTOT_{problem.n_tot}_GT_{max_n_tot}",
            skipped=True,
            n_eval=0,
            wall_ms=0.0,
            z=z0.copy(),
            x_list=[x.copy(order="F") for x in x0_list],
            audit=audit_init,
        )

    oracle = EvalOracle(
        problem,
        max_eval=max_eval,
        timeout_sec=timeout_sec,
        enforce_budget_exception=True,
        stop_on_kkt_tol=gtol,
        feas_tol=feas_tol,
    )

    x_init_full = _pack_flat(z0, x0_list)
    lb_full = np.concatenate(
        [problem.lower, np.full(problem.n_stiefel_tot, -np.inf, dtype=np.float64)]
    )
    ub_full = np.concatenate(
        [problem.upper, np.full(problem.n_stiefel_tot, np.inf, dtype=np.float64)]
    )
    free_mask = lb_full < ub_full
    free_idx = np.flatnonzero(free_mask)
    fixed_vals = x_init_full.copy()

    x_cur_red = x_init_full[free_idx]
    bounds_red = Bounds(lb_full[free_idx], ub_full[free_idx])

    m_eq, eq_fun, eq_jac, eq_hess = _build_stiefel_vech_constraints(
        problem, free_idx, fixed_vals
    )

    def expand_full(x_red: np.ndarray) -> np.ndarray:
        if free_idx.size == x_init_full.size:
            return x_red
        out = fixed_vals.copy()
        out[free_idx] = x_red
        return out

    def fg_red(x_red: np.ndarray) -> tuple[float, np.ndarray]:
        x_full = expand_full(x_red)
        z_v, xs_v = _unpack_flat(x_full, problem)
        f_val, gz, gxs = oracle.eval_fg(z_v, xs_v)
        if not np.isfinite(f_val):
            return 1.0e100, np.zeros_like(x_red)
        g_full = _pack_flat(gz, gxs)
        g_red = g_full if free_idx.size == x_init_full.size else g_full[free_idx]
        if not np.all(np.isfinite(g_red)):
            return 1.0e100, np.zeros_like(x_red)
        return float(f_val), g_red

    t0 = time.perf_counter()
    status = "COMPLETED"
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if scipy_method == "SLSQP":
                cons = (
                    [{"type": "eq", "fun": eq_fun, "jac": eq_jac}] if m_eq > 0 else []
                )
                res = minimize(
                    fg_red,
                    x_cur_red,
                    method="SLSQP",
                    jac=True,
                    bounds=bounds_red,
                    constraints=cons,
                    options={"ftol": 1.0e-15, "maxiter": max_eval},
                )
            else:
                nlc = (
                    [NonlinearConstraint(eq_fun, 0.0, 0.0, jac=eq_jac, hess=eq_hess)]
                    if m_eq > 0
                    else []
                )
                res = minimize(
                    fg_red,
                    x_cur_red,
                    method="trust-constr",
                    jac=True,
                    hess=BFGS(),
                    bounds=bounds_red,
                    constraints=nlc,
                    options={"gtol": gtol * 0.1, "xtol": 1.0e-14, "maxiter": max_eval},
                )
            x_cur_red = np.asarray(res.x, dtype=np.float64)
            status = "CONVERGED" if res.success else f"STOP_{res.status}"
    except TargetKKTReachedError:
        status = "TARGET_KKT_REACHED"
    except BudgetExhaustedError:
        status = "MAX_EVAL_REACHED"
    except WallTimeoutError:
        status = "TIMEOUT"

    wall_ms = 1000.0 * (time.perf_counter() - t0)
    x_fin_full = expand_full(x_cur_red)
    z_fin, xs_fin = _unpack_flat(x_fin_full, problem)
    return _finalize_baseline_result(
        method_name, problem, seed, status, oracle, wall_ms, z_fin, xs_fin, gtol, feas_tol
    )


def run_scipy_slsqp(
    problem: BenchmarkProblem,
    z0: np.ndarray,
    x0_list: Sequence[np.ndarray],
    seed: int = 101,
    gtol: float = 1.0e-6,
    feas_tol: float = 1.0e-8,
    max_eval: int = 5000,
    timeout_sec: float = 20.0,
    max_n_tot: int = 450,
) -> BaselineResult:
    """Run SciPy SLSQP with analytical Stiefel equality Jacobians (N_tot <= 450)."""
    return _run_scipy_constrained_nlp(
        "SciPy-SLSQP",
        "SLSQP",
        problem,
        z0,
        x0_list,
        seed=seed,
        gtol=gtol,
        feas_tol=feas_tol,
        max_eval=max_eval,
        timeout_sec=timeout_sec,
        max_n_tot=max_n_tot,
    )


def run_scipy_trust_constr(
    problem: BenchmarkProblem,
    z0: np.ndarray,
    x0_list: Sequence[np.ndarray],
    seed: int = 101,
    gtol: float = 1.0e-6,
    feas_tol: float = 1.0e-8,
    max_eval: int = 5000,
    timeout_sec: float = 20.0,
    max_n_tot: int = 450,
) -> BaselineResult:
    """Run SciPy trust-constr with BFGS objective Hessian & exact constraint Hessians."""
    return _run_scipy_constrained_nlp(
        "SciPy-trust-constr",
        "trust-constr",
        problem,
        z0,
        x0_list,
        seed=seed,
        gtol=gtol,
        feas_tol=feas_tol,
        max_eval=max_eval,
        timeout_sec=timeout_sec,
        max_n_tot=max_n_tot,
    )


def _run_pymanopt_solver(
    solver_name: str,
    use_trust_regions: bool,
    problem: BenchmarkProblem,
    z0: np.ndarray,
    x0_list: Sequence[np.ndarray],
    seed: int = 101,
    gtol: float = 1.0e-6,
    feas_tol: float = 1.0e-8,
    max_eval: int = 5000,
    timeout_sec: float = 20.0,
) -> BaselineResult:
    oracle = EvalOracle(
        problem,
        max_eval=max_eval,
        timeout_sec=timeout_sec,
        enforce_budget_exception=True,
        stop_on_kkt_tol=gtol,
        feas_tol=feas_tol,
    )
    warp = SmoothBoxWarp(problem.lower, problem.upper) if problem.d > 0 else None

    # Build native Stiefel manifold when d == 0, Euclidean(d) when R == 0, else Product
    has_single_factor = (problem.d == 0 and problem.R == 1) or (
        problem.d > 0 and problem.R == 0
    )
    if problem.d == 0 and problem.R == 1:
        manifold = Stiefel(problem.n_list[0], problem.p_list[0])
        init_pt: object = x0_list[0].copy(order="F")
    elif problem.d == 0 and problem.R >= 2:
        manifold = Product(
            [Stiefel(n_r, p_r) for n_r, p_r in zip(problem.n_list, problem.p_list)]
        )
        init_pt = [x.copy(order="F") for x in x0_list]
    elif problem.d > 0 and problem.R == 0:
        assert warp is not None
        manifold = Euclidean(problem.d)
        init_pt = warp.inverse(z0)
    else:
        assert warp is not None
        factors = [Euclidean(problem.d)] + [
            Stiefel(n_r, p_r) for n_r, p_r in zip(problem.n_list, problem.p_list)
        ]
        manifold = Product(factors)
        init_pt = [warp.inverse(z0)] + [x.copy(order="F") for x in x0_list]

    def decode_point(args: Sequence[np.ndarray]) -> tuple[np.ndarray, list[np.ndarray], np.ndarray | None]:
        if problem.d == 0:
            z_v = np.zeros(0, dtype=np.float64)
            xs_v = [np.asfortranarray(a, dtype=np.float64) for a in args]
            return z_v, xs_v, None
        assert warp is not None
        z_v, jac_v = warp.forward_and_jac(np.asarray(args[0], dtype=np.float64))
        xs_v = [np.asfortranarray(a, dtype=np.float64) for a in args[1:]]
        return z_v, xs_v, jac_v

    def raw_cost_and_egrad(args: Sequence[np.ndarray]) -> tuple[float, list[np.ndarray]]:
        z_v, xs_v, jac_v = decode_point(args)
        f_val, gz, gxs = oracle.eval_fg(z_v, xs_v)
        if not np.isfinite(f_val):
            zero_grads = [np.zeros_like(a) for a in args]
            return 1.0e100, zero_grads
        if problem.d == 0:
            return float(f_val), gxs
        assert jac_v is not None
        g_theta = jac_v * gz
        return float(f_val), [g_theta] + gxs

    @pymanopt.function.numpy(manifold)
    def cost_fn(*args: np.ndarray) -> float:
        f_val, _ = raw_cost_and_egrad(args)
        return float(f_val)

    @pymanopt.function.numpy(manifold)
    def egrad_fn(*args: np.ndarray):  # type: ignore[no-untyped-def]
        _, grads = raw_cost_and_egrad(args)
        return grads[0] if has_single_factor else grads

    ehess_fn = None
    if use_trust_regions:
        n_factors = 1 if has_single_factor else (1 if problem.d > 0 else 0) + problem.R

        @pymanopt.function.numpy(manifold)
        def ehess_raw(*all_args: np.ndarray):  # type: ignore[no-untyped-def]
            pts = all_args[:n_factors]
            dirs = all_args[n_factors:]
            dir_norm_sq = sum(float(np.sum(d_arr * d_arr)) for d_arr in dirs)
            if dir_norm_sq <= 1.0e-60:
                zeros = [np.zeros_like(p_arr) for p_arr in pts]
                return zeros[0] if has_single_factor else zeros

            dir_norm = float(np.sqrt(dir_norm_sq))
            pt_norm = float(np.sqrt(sum(float(np.sum(p_arr * p_arr)) for p_arr in pts)))
            h_step = 1.0e-7 * max(1.0, pt_norm) / dir_norm

            _, g_base = raw_cost_and_egrad(pts)
            pts_pert = [p_arr + h_step * d_arr for p_arr, d_arr in zip(pts, dirs)]
            _, g_pert = raw_cost_and_egrad(pts_pert)
            hvps = [(gp - gb) / h_step for gp, gb in zip(g_pert, g_base)]
            return hvps[0] if has_single_factor else hvps

        ehess_fn = ehess_raw

    pm_problem = pymanopt.Problem(
        manifold,
        cost_fn,
        euclidean_gradient=egrad_fn,
        euclidean_hessian=ehess_fn,
    )

    if use_trust_regions:
        optimizer = TrustRegions(
            verbosity=0,
            max_time=timeout_sec,
            max_iterations=max_eval,
            min_gradient_norm=gtol * 0.1,
            max_cost_evaluations=max_eval,
        )
    else:
        als = AdaptiveLineSearcher(
            contraction_factor=0.5,
            sufficient_decrease=1.0e-4,
            max_iterations=40,
        )
        optimizer = ConjugateGradient(
            line_searcher=als,
            verbosity=0,
            max_time=timeout_sec,
            max_iterations=max_eval,
            min_gradient_norm=gtol * 0.1,
            min_step_size=1.0e-14,
            max_cost_evaluations=max_eval,
        )

    cand_z = z0.copy()
    cand_xs = [x.copy(order="F") for x in x0_list]
    t0 = time.perf_counter()
    status = "COMPLETED"

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            res = optimizer.run(pm_problem, initial_point=init_pt)
        pt_out = [res.point] if has_single_factor else list(res.point)
        cand_z, cand_xs, _ = decode_point(pt_out)
        status = str(res.stopping_criterion)
    except TargetKKTReachedError:
        status = "TARGET_KKT_REACHED"
    except BudgetExhaustedError:
        status = "MAX_EVAL_REACHED"
        if oracle.best_z is not None and oracle.best_xs is not None:
            cand_z = oracle.best_z
            cand_xs = oracle.best_xs
    except WallTimeoutError:
        status = "TIMEOUT"
        if oracle.best_z is not None and oracle.best_xs is not None:
            cand_z = oracle.best_z
            cand_xs = oracle.best_xs

    wall_ms = 1000.0 * (time.perf_counter() - t0)
    return _finalize_baseline_result(
        solver_name, problem, seed, status, oracle, wall_ms, cand_z, cand_xs, gtol, feas_tol
    )


def run_pymanopt_warped_cg(
    problem: BenchmarkProblem,
    z0: np.ndarray,
    x0_list: Sequence[np.ndarray],
    seed: int = 101,
    gtol: float = 1.0e-6,
    feas_tol: float = 1.0e-8,
    max_eval: int = 5000,
    timeout_sec: float = 20.0,
) -> BaselineResult:
    """Run pymanopt 2.2.1 ConjugateGradient (smooth box warps + native Stiefel on B7B)."""
    return _run_pymanopt_solver(
        "pymanopt-Warped-CG",
        False,
        problem,
        z0,
        x0_list,
        seed=seed,
        gtol=gtol,
        feas_tol=feas_tol,
        max_eval=max_eval,
        timeout_sec=timeout_sec,
    )


def run_pymanopt_trust_regions(
    problem: BenchmarkProblem,
    z0: np.ndarray,
    x0_list: Sequence[np.ndarray],
    seed: int = 101,
    gtol: float = 1.0e-6,
    feas_tol: float = 1.0e-8,
    max_eval: int = 5000,
    timeout_sec: float = 20.0,
) -> BaselineResult:
    """Run pymanopt 2.2.1 TrustRegions with cached first-order FD-HVP."""
    return _run_pymanopt_solver(
        "pymanopt-TrustRegions",
        True,
        problem,
        z0,
        x0_list,
        seed=seed,
        gtol=gtol,
        feas_tol=feas_tol,
        max_eval=max_eval,
        timeout_sec=timeout_sec,
    )


def run_alternating_bcd(
    problem: BenchmarkProblem,
    z0: np.ndarray,
    x0_list: Sequence[np.ndarray],
    seed: int = 101,
    gtol: float = 1.0e-6,
    feas_tol: float = 1.0e-8,
    max_eval: int = 5000,
    timeout_sec: float = 20.0,
    inner_box_iters: int = 10,
    inner_stiefel_iters: int = 10,
) -> BaselineResult:
    """Run strictly feasible Alternating BCD (SciPy L-BFGS-B <-> pymanopt CG)."""
    if problem.R == 0:
        res_box = run_scipy_lbfgsb(
            problem,
            z0,
            x0_list,
            seed=seed,
            gtol=gtol,
            feas_tol=feas_tol,
            max_eval=max_eval,
            timeout_sec=timeout_sec,
        )
        return BaselineResult(
            solver_name="Alternating-BCD",
            problem_id=problem.problem_id,
            seed=seed,
            status=res_box.status,
            skipped=False,
            n_eval=res_box.n_eval,
            wall_ms=res_box.wall_ms,
            z=res_box.z,
            x_list=res_box.x_list,
            audit=res_box.audit,
        )

    if problem.d == 0:
        res_st = run_pymanopt_warped_cg(
            problem,
            z0,
            x0_list,
            seed=seed,
            gtol=gtol,
            feas_tol=feas_tol,
            max_eval=max_eval,
            timeout_sec=timeout_sec,
        )
        return BaselineResult(
            solver_name="Alternating-BCD",
            problem_id=problem.problem_id,
            seed=seed,
            status=res_st.status,
            skipped=False,
            n_eval=res_st.n_eval,
            wall_ms=res_st.wall_ms,
            z=res_st.z,
            x_list=res_st.x_list,
            audit=res_st.audit,
        )

    oracle = EvalOracle(
        problem,
        max_eval=max_eval,
        timeout_sec=timeout_sec,
        enforce_budget_exception=True,
        stop_on_kkt_tol=gtol,
        feas_tol=feas_tol,
    )

    z_cur = z0.copy()
    xs_cur = [x.copy(order="F") for x in x0_list]
    box_bounds = Bounds(problem.lower, problem.upper)
    single_st = problem.R == 1
    st_manifold = (
        Stiefel(problem.n_list[0], problem.p_list[0])
        if single_st
        else Product([Stiefel(n_r, p_r) for n_r, p_r in zip(problem.n_list, problem.p_list)])
    )
    als = AdaptiveLineSearcher(
        contraction_factor=0.5,
        sufficient_decrease=1.0e-4,
        max_iterations=40,
    )
    cg_opt = ConjugateGradient(
        line_searcher=als,
        verbosity=0,
        max_time=timeout_sec,
        max_iterations=inner_stiefel_iters,
        min_gradient_norm=gtol * 0.1,
        min_step_size=1.0e-14,
        max_cost_evaluations=max_eval,
    )

    t0 = time.perf_counter()
    status = "MAX_CYCLES"

    try:
        for cycle in range(max_eval):
            z_prev = z_cur.copy()
            xs_prev = [x.copy(order="F") for x in xs_cur]

            rem_box = max_eval - oracle.n_eval
            if rem_box <= 0:
                status = "MAX_EVAL_REACHED"
                break

            # 1. Inner Euclidean Box Subproblem via SciPy L-BFGS-B
            def fg_box_sub(z_vec: np.ndarray) -> tuple[float, np.ndarray]:
                f_val, gz, _ = oracle.eval_fg(z_vec, xs_cur)
                if not (np.isfinite(f_val) and np.all(np.isfinite(gz))):
                    return 1.0e100, np.zeros_like(z_vec)
                return float(f_val), gz

            res_b = minimize(
                fg_box_sub,
                z_cur,
                method="L-BFGS-B",
                jac=True,
                bounds=box_bounds,
                options={
                    "maxiter": inner_box_iters,
                    "maxfun": rem_box,
                    "gtol": gtol * 0.1,
                    "ftol": 0.0,
                },
            )
            z_cur = np.clip(np.asarray(res_b.x, dtype=np.float64), problem.lower, problem.upper)

            rem_st = max_eval - oracle.n_eval
            if rem_st <= 0:
                status = "MAX_EVAL_REACHED"
                break

            # 2. Inner Stiefel Product Subproblem via pymanopt ConjugateGradient
            @pymanopt.function.numpy(st_manifold)
            def cost_st(*x_args: np.ndarray) -> float:
                xs_in = [np.asfortranarray(a, dtype=np.float64) for a in x_args]
                f_val, _, _ = oracle.eval_fg(z_cur, xs_in)
                return float(f_val) if np.isfinite(f_val) else 1.0e100

            @pymanopt.function.numpy(st_manifold)
            def egrad_st(*x_args: np.ndarray):  # type: ignore[no-untyped-def]
                xs_in = [np.asfortranarray(a, dtype=np.float64) for a in x_args]
                f_val, _, gxs = oracle.eval_fg(z_cur, xs_in)
                if not np.isfinite(f_val):
                    zeros = [np.zeros_like(a) for a in xs_in]
                    return zeros[0] if single_st else zeros
                return gxs[0] if single_st else gxs

            cg_opt._max_time = max(0.05, timeout_sec - (time.perf_counter() - t0))
            cg_opt._max_cost_evaluations = rem_st
            st_prob = pymanopt.Problem(st_manifold, cost_st, euclidean_gradient=egrad_st)
            init_st = xs_cur[0] if single_st else xs_cur
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                res_s = cg_opt.run(
                    st_prob, initial_point=init_st, reuse_line_searcher=(cycle > 0)
                )
            xs_out = [res_s.point] if single_st else list(res_s.point)
            xs_cur = [np.asfortranarray(a, dtype=np.float64) for a in xs_out]

            dz_max = float(np.max(np.abs(z_cur - z_prev)))
            dx_max = max(
                float(np.linalg.norm(xn - xo, ord="fro")) for xn, xo in zip(xs_cur, xs_prev)
            )
            if max(dz_max, dx_max) <= 1.0e-15:
                status = "STAGNATED"
                break
    except TargetKKTReachedError:
        status = "TARGET_KKT_REACHED"
    except BudgetExhaustedError:
        status = "MAX_EVAL_REACHED"
        if oracle.best_z is not None and oracle.best_xs is not None:
            z_cur = oracle.best_z
            xs_cur = oracle.best_xs
    except WallTimeoutError:
        status = "TIMEOUT"
        if oracle.best_z is not None and oracle.best_xs is not None:
            z_cur = oracle.best_z
            xs_cur = oracle.best_xs

    wall_ms = 1000.0 * (time.perf_counter() - t0)
    return _finalize_baseline_result(
        "Alternating-BCD", problem, seed, status, oracle, wall_ms, z_cur, xs_cur, gtol, feas_tol
    )


BASELINE_REGISTRY: dict[str, Callable[..., BaselineResult]] = {
    "Alternating-BCD": run_alternating_bcd,
    "pymanopt-Warped-CG": run_pymanopt_warped_cg,
    "pymanopt-TrustRegions": run_pymanopt_trust_regions,
    "SciPy-L-BFGS-B": run_scipy_lbfgsb,
    "SciPy-SLSQP": run_scipy_slsqp,
    "SciPy-trust-constr": run_scipy_trust_constr,
}


def run_all_external_baselines_single_seed(
    problem: BenchmarkProblem,
    seed: int = 101,
    gtol: float = 1.0e-6,
    feas_tol: float = 1.0e-8,
    max_eval: int = 5000,
    timeout_sec: float = 20.0,
) -> list[BaselineResult]:
    """Run all 6 external baselines on one `(problem, seed)` pair from identical `(z0, X0)`."""
    from benchmarks.referee import generate_initial_point

    z0, x0_list = generate_initial_point(problem, seed=seed)
    results: list[BaselineResult] = []
    for name in EXTERNAL_BASELINE_NAMES:
        runner = BASELINE_REGISTRY[name]
        results.append(
            runner(
                problem,
                z0.copy(),
                [x.copy(order="F") for x in x0_list],
                seed=seed,
                gtol=gtol,
                feas_tol=feas_tol,
                max_eval=max_eval,
                timeout_sec=timeout_sec,
            )
        )
    return results
