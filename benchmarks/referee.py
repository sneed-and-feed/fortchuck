"""Independent external referee for the fortchuck blind benchmark suite (v0.2.0).

Provides:
  1. Independent pure-NumPy linear-box / simplex projection (`project_linear_box_ref`)
     via safeguarded bisection + secant root-finding (`<= 1e-14`, zero `fortchuck` calls).
  2. Independent external KKT optimality residual (`E_KKT_ref`), box violation (`V_box`),
     linear equality violation (`V_lin`), and Stiefel orthogonality violation (`V_orth`).
  3. Opaque evaluation counter (`EvalOracle`) enforcing hard `max_eval` and
     wall-clock timeout budgets while avoiding double-counting when external
     optimizers invoke separate `cost(x)` and `grad(x)` callbacks at one point.
  4. Deterministic `PCG64` feasible `(z0, X0_list)` initialization generator
     supporting primary (`101..110`), holdout (`201..210`), and untouched third
     blind holdout (`301..310`) seed suites while preserving exact `v0.1.0` draws
     on `B1..B7B`.
  5. Full coordinatewise central finite-difference gradient verifier (`<= 1e-7`).
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
import time
from typing import Any, Sequence
import numpy as np

from benchmarks.problems import (
    BenchmarkProblem,
    get_all_benchmark_problems,
)

DEFAULT_SEEDS: tuple[int, ...] = tuple(range(101, 111))
PRIMARY_SEEDS: tuple[int, ...] = DEFAULT_SEEDS
HOLDOUT_SEEDS: tuple[int, ...] = tuple(range(201, 211))
BLIND3_SEEDS: tuple[int, ...] = tuple(range(301, 311))


class BudgetExhaustedError(RuntimeError):
    """Raised inside Python baseline wrappers when `max_eval` is reached."""


class WallTimeoutError(RuntimeError):
    """Raised inside Python baseline wrappers when `timeout_sec` is exceeded."""


class TargetKKTReachedError(RuntimeError):
    """Raised inside Python baseline wrappers once exact target KKT + feasibility holds."""


@dataclass(frozen=True)
class RefereeAudit:
    """Independent post-solve optimality and primal feasibility certificate."""

    f_val: float
    e_kkt_ref: float
    box_residual: float
    stiefel_residual: float
    v_box: float
    v_orth: float
    n_active_box: int
    solved: bool
    f_gap: float | None = None
    v_lin: float = 0.0

    @property
    def success(self) -> bool:
        return self.solved

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class GradientFDReport:
    """Result of full coordinatewise central finite-difference gradient verification."""

    problem_id: str
    seed: int
    n_tot: int
    fd_step: float
    max_abs_err_z: float
    max_rel_err_z: float
    max_abs_err_x: float
    max_rel_err_x: float
    max_rel_err_total: float
    passed: bool


def parse_seeds(seed_spec: str | Sequence[int] | None = None) -> list[int]:
    """Parse CLI seed specification (e.g. '101..110', '201..210', '101,102,105-107')."""
    if seed_spec is None:
        return list(DEFAULT_SEEDS)
    if not isinstance(seed_spec, str):
        seeds = [int(s) for s in seed_spec]
        if not seeds:
            raise ValueError("Seed sequence must not be empty.")
        return seeds

    text = seed_spec.strip()
    if not text:
        return list(DEFAULT_SEEDS)

    out: list[int] = []
    for token in text.split(","):
        tok = token.strip()
        if not tok:
            continue
        if ".." in tok:
            parts = tok.split("..", 1)
            start, end = int(parts[0].strip()), int(parts[1].strip())
            step = 1 if end >= start else -1
            out.extend(range(start, end + step, step))
        elif "-" in tok and not tok.startswith("-"):
            parts = tok.split("-", 1)
            start, end = int(parts[0].strip()), int(parts[1].strip())
            step = 1 if end >= start else -1
            out.extend(range(start, end + step, step))
        else:
            out.append(int(tok))

    if not out:
        raise ValueError(f"Could not parse any seeds from {seed_spec!r}.")
    return out


def project_linear_box_ref(
    y: np.ndarray,
    l: np.ndarray,
    u: np.ndarray,
    a: np.ndarray,
    b: float,
) -> np.ndarray:
    """Pure-NumPy Euclidean projection onto `{z in [l, u]^d : a^T z = b}`.

    Uses safeguarded monotone bisection + secant root-finding with `math.fsum`
    summation to `<= 1e-14` accuracy (never calling `fortchuck`).
    """
    y_arr = np.asarray(y, dtype=np.float64).ravel()
    l_arr = np.asarray(l, dtype=np.float64).ravel()
    u_arr = np.asarray(u, dtype=np.float64).ravel()
    a_arr = np.asarray(a, dtype=np.float64).ravel()
    b_val = float(b)

    d = y_arr.size
    if d == 0 or l_arr.size != d or u_arr.size != d or a_arr.size != d:
        raise ValueError("Dimension mismatch or empty input in project_linear_box_ref.")
    if not (np.all(np.isfinite(y_arr)) and np.all(np.isfinite(a_arr)) and np.isfinite(b_val)):
        raise ValueError("Non-finite y, a, or b in project_linear_box_ref.")
    if np.any(np.isnan(l_arr)) or np.any(np.isnan(u_arr)) or np.any(l_arr > u_arr):
        raise ValueError("Invalid box bounds in project_linear_box_ref.")

    a_max = float(np.max(np.abs(a_arr)))
    if a_max <= 0.0:
        raise ValueError("Zero normal vector a in project_linear_box_ref.")

    pos = a_arr > 0.0
    neg = a_arr < 0.0
    nz = pos | neg

    has_unbounded_lo = bool(np.any(~np.isfinite(l_arr[pos])) or np.any(~np.isfinite(u_arr[neg])))
    has_unbounded_hi = bool(np.any(~np.isfinite(u_arr[pos])) or np.any(~np.isfinite(l_arr[neg])))

    b_min = (
        -np.inf
        if has_unbounded_lo
        else math.fsum(
            [*(float(ai * li) for ai, li in zip(a_arr[pos], l_arr[pos])),
             *(float(ai * ui) for ai, ui in zip(a_arr[neg], u_arr[neg]))]
        )
    )
    b_max = (
        np.inf
        if has_unbounded_hi
        else math.fsum(
            [*(float(ai * ui) for ai, ui in zip(a_arr[pos], u_arr[pos])),
             *(float(ai * li) for ai, li in zip(a_arr[neg], l_arr[neg]))]
        )
    )

    feas_tol_lin = 1.0e-14 * (1.0 + abs(b_val))
    if (not has_unbounded_lo and b_val < b_min - feas_tol_lin) or (
        not has_unbounded_hi and b_val > b_max + feas_tol_lin
    ):
        raise ValueError(
            f"Infeasible linear-box constraint: b={b_val} outside [{b_min}, {b_max}]."
        )

    singleton_tol = 1.0e-15 * (1.0 + abs(b_val))
    if not has_unbounded_lo and b_val <= b_min + singleton_tol:
        z_out = np.clip(y_arr, l_arr, u_arr)
        z_out[pos] = l_arr[pos]
        z_out[neg] = u_arr[neg]
        return z_out
    if not has_unbounded_hi and b_val >= b_max - singleton_tol:
        z_out = np.clip(y_arr, l_arr, u_arr)
        z_out[pos] = u_arr[pos]
        z_out[neg] = l_arr[neg]
        return z_out

    a_hat = a_arr / a_max
    b_hat = b_val / a_max
    a_l1 = float(np.sum(np.abs(a_arr)))
    eps_mach = float(np.finfo(np.float64).eps)

    def _eval_cand(lam: float) -> tuple[np.ndarray, float, float]:
        z_c = np.clip(y_arr - lam * a_hat, l_arr, u_arr)
        raw_err = math.fsum(float(ai) * float(zi) for ai, zi in zip(a_arr, z_c)) - b_val
        psi_val = raw_err / a_max
        return z_c, psi_val, raw_err

    def _polish_free_face(z_init: np.ndarray) -> np.ndarray:
        """Eliminate O(eps_mach * ||y||) subtraction cancellation directly in z-space."""
        z_p = z_init.copy()
        tol_res = 0.5 * eps_mach * (1.0 + abs(b_val))
        for _ in range(4):
            res = math.fsum(float(ai) * float(zi) for ai, zi in zip(a_arr, z_p)) - b_val
            if abs(res) <= tol_res:
                break
            free = nz & (z_p > l_arr) & (z_p < u_arr)
            if not np.any(free):
                if res > 0.0:
                    free = (pos & (z_p > l_arr)) | (neg & (z_p < u_arr))
                else:
                    free = (pos & (z_p < u_arr)) | (neg & (z_p > l_arr))
            denom = float(np.dot(a_arr[free], a_arr[free]))
            if denom <= 0.0:
                break
            step = res / denom
            z_p[free] = np.clip(z_p[free] - step * a_arr[free], l_arr[free], u_arr[free])
        return z_p

    # Unconstrained hyperplane fast test
    z_clip0 = np.clip(y_arr, l_arr, u_arr)
    lam0 = (
        math.fsum(float(ah) * float(zi) for ah, zi in zip(a_hat, z_clip0)) - b_hat
    ) / float(np.dot(a_hat, a_hat))
    z0_try, psi0, err0 = _eval_cand(lam0)
    if abs(err0) <= 0.5 * eps_mach * (1.0 + abs(b_val)):
        return z0_try

    # Construct guaranteed monotone bracket [lam_lo, lam_hi] with psi(lam_lo) >= 0 >= psi(lam_hi)
    all_nz_finite = bool(np.all(np.isfinite(l_arr[nz])) and np.all(np.isfinite(u_arr[nz])))
    min_abs_ahat = float(np.min(np.abs(a_hat[nz])))
    if all_nz_finite and min_abs_ahat > 1.0e-12:
        r_u = (y_arr[nz] - u_arr[nz]) / a_hat[nz]
        r_l = (y_arr[nz] - l_arr[nz]) / a_hat[nz]
        lam_lo = float(np.min(np.minimum(r_u, r_l)))
        lam_hi = float(np.max(np.maximum(r_u, r_l)))
        z_lo, psi_lo, err_lo = _eval_cand(lam_lo)
        z_hi, psi_hi, err_hi = _eval_cand(lam_hi)
    else:
        lam_lo = lam0
        lam_hi = lam0
        z_lo, psi_lo, err_lo = z0_try, psi0, err0
        z_hi, psi_hi, err_hi = z0_try, psi0, err0

    step_expand = max(1.0, abs(lam0), float(np.max(np.abs(y_arr))))
    while psi_lo < 0.0:
        lam_hi, z_hi, psi_hi, err_hi = lam_lo, z_lo, psi_lo, err_lo
        lam_lo -= step_expand
        z_lo, psi_lo, err_lo = _eval_cand(lam_lo)
        step_expand *= 2.0

    step_expand = max(1.0, abs(lam0), float(np.max(np.abs(y_arr))))
    while psi_hi > 0.0:
        lam_lo, z_lo, psi_lo, err_lo = lam_hi, z_hi, psi_hi, err_hi
        lam_hi += step_expand
        z_hi, psi_hi, err_hi = _eval_cand(lam_hi)
        step_expand *= 2.0

    # Tighten with initial trial lam0 if inside [lam_lo, lam_hi]
    if lam_lo < lam0 < lam_hi:
        if psi0 >= 0.0:
            lam_lo, z_lo, psi_lo, err_lo = lam0, z0_try, psi0, err0
        else:
            lam_hi, z_hi, psi_hi, err_hi = lam0, z0_try, psi0, err0

    w_lo = psi_lo
    w_hi = psi_hi
    last_side = 0

    for it in range(120):
        if abs(err_lo) <= abs(err_hi):
            best_z, best_err = z_lo, err_lo
        else:
            best_z, best_err = z_hi, err_hi

        if abs(best_err) <= 0.5 * eps_mach * (1.0 + abs(b_val)):
            return best_z

        width = lam_hi - lam_lo
        if width <= 2.0 * eps_mach * (1.0 + abs(lam_lo) + abs(lam_hi)):
            return _polish_free_face(best_z)

        phase = it % 4
        if phase == 0 and (psi_lo - psi_hi) > 0.0:
            # Exact unscaled secant (converges in 1 step once inside optimal affine piece)
            lam_cand = lam_lo + (psi_lo / (psi_lo - psi_hi)) * width
        elif phase in (1, 2) and (w_lo - w_hi) > 0.0:
            # Illinois anti-stagnation secant across piecewise-linear kinks
            lam_cand = lam_lo + (w_lo / (w_lo - w_hi)) * width
        else:
            # Safeguarded midpoint bisection
            lam_cand = 0.5 * (lam_lo + lam_hi)

        lo_guard = np.nextafter(lam_lo, np.inf)
        hi_guard = np.nextafter(lam_hi, -np.inf)
        if lo_guard >= hi_guard:
            return _polish_free_face(best_z)
        lam_cand = float(np.clip(lam_cand, lo_guard, hi_guard))

        z_c, psi_c, err_c = _eval_cand(lam_cand)
        if psi_c == 0.0:
            return z_c
        if psi_c > 0.0:
            lam_lo, z_lo, psi_lo, err_lo = lam_cand, z_c, psi_c, err_c
            w_lo = psi_c
            if last_side == -1:
                w_hi *= 0.5
            last_side = -1
        else:
            lam_hi, z_hi, psi_hi, err_hi = lam_cand, z_c, psi_c, err_c
            w_hi = psi_c
            if last_side == 1:
                w_lo *= 0.5
            last_side = 1

    return _polish_free_face(z_lo if abs(err_lo) <= abs(err_hi) else z_hi)


def generate_initial_point(
    problem: BenchmarkProblem, seed: int
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Generate deterministic feasible starting point `(z0, X0_list)` via PCG64(seed).

    Preserves the exact `v0.1.0` RNG draw sequence for `B1..B7B` (`lin_a is None`).
    When `problem.lin_a is not None` (`B8A`, `B8B`), projects the initial `z0` draw
    onto the central 80% sub-box `[l + 0.1*(u - l), u - 0.1*(u - l)]` intersected
    with `a^T z = b` via `project_linear_box_ref` so every solver (including
    interior/tanh-warped baselines) starts from the exact same strictly interior
    feasible point. Square Stiefel factors (`n_r == p_r`) are oriented in `SO(n_r)`.
    """
    rng = np.random.Generator(np.random.PCG64(int(seed)))

    if problem.d > 0:
        z0 = np.empty(problem.d, dtype=np.float64)
        l_arr = problem.lower
        u_arr = problem.upper
        u_draws = rng.uniform(0.1, 0.9, size=problem.d)
        pos_draws = rng.uniform(0.5, 2.5, size=problem.d)
        norm_draws = rng.standard_normal(problem.d)

        for i in range(problem.d):
            li = float(l_arr[i])
            ui = float(u_arr[i])
            if li == ui:
                z0[i] = li
            elif np.isfinite(li) and np.isfinite(ui):
                # Overflow-safe convex combination (1 - t)*li + t*ui
                t = float(u_draws[i])
                z0[i] = (1.0 - t) * li + t * ui
            elif np.isfinite(li):
                z0[i] = li + float(pos_draws[i])
            elif np.isfinite(ui):
                z0[i] = ui - float(pos_draws[i])
            else:
                z0[i] = float(norm_draws[i])

        if problem.lin_a is not None:
            assert problem.lin_b is not None
            l_sub = np.empty(problem.d, dtype=np.float64)
            u_sub = np.empty(problem.d, dtype=np.float64)
            for i in range(problem.d):
                li = float(l_arr[i])
                ui = float(u_arr[i])
                if li == ui:
                    l_sub[i] = li
                    u_sub[i] = ui
                elif np.isfinite(li) and np.isfinite(ui):
                    l_sub[i] = 0.9 * li + 0.1 * ui
                    u_sub[i] = 0.1 * li + 0.9 * ui
                elif np.isfinite(li):
                    l_sub[i] = li + 0.1
                    u_sub[i] = np.inf
                elif np.isfinite(ui):
                    l_sub[i] = -np.inf
                    u_sub[i] = ui - 0.1
                else:
                    l_sub[i] = -np.inf
                    u_sub[i] = np.inf
            try:
                z0 = project_linear_box_ref(
                    z0, l_sub, u_sub, problem.lin_a, float(problem.lin_b)
                )
            except ValueError:
                # Fallback to full box only if central 80% sub-box cannot reach singleton face
                z0 = project_linear_box_ref(
                    z0, l_arr, u_arr, problem.lin_a, float(problem.lin_b)
                )
    else:
        z0 = np.zeros(0, dtype=np.float64)

    x0_list: list[np.ndarray] = []
    for n_r, p_r in zip(problem.n_list, problem.p_list):
        raw = rng.standard_normal((n_r, p_r))
        q_mat, r_mat = np.linalg.qr(raw, mode="reduced")
        signs = np.where(np.diag(r_mat) >= 0.0, 1.0, -1.0)
        q_mat = q_mat * signs[None, :]
        if n_r == p_r and np.linalg.det(q_mat) < 0.0:
            q_mat[:, -1] *= -1.0
        x0_list.append(np.asfortranarray(q_mat, dtype=np.float64))

    return z0, x0_list


def compute_metrics_from_fg(
    problem: BenchmarkProblem,
    z: np.ndarray,
    x_list: Sequence[np.ndarray],
    f_val: float,
    gz: np.ndarray,
    gx_list: Sequence[np.ndarray],
    kkt_tol: float = 1.0e-6,
    feas_tol: float = 1.0e-8,
) -> RefereeAudit:
    """Compute independent referee KKT and feasibility residuals from evaluated `(f, gz, gx)`."""
    if not np.isfinite(f_val):
        return RefereeAudit(
            f_val=float("nan"),
            e_kkt_ref=float("inf"),
            box_residual=float("inf"),
            stiefel_residual=float("inf"),
            v_box=float("inf"),
            v_orth=float("inf"),
            n_active_box=0,
            solved=False,
            f_gap=None,
            v_lin=float("inf") if problem.lin_a is not None else 0.0,
        )

    if problem.d > 0:
        if not (np.all(np.isfinite(z)) and np.all(np.isfinite(gz))):
            return RefereeAudit(
                f_val=float(f_val),
                e_kkt_ref=float("inf"),
                box_residual=float("inf"),
                stiefel_residual=float("inf"),
                v_box=float("inf"),
                v_orth=float("inf"),
                n_active_box=0,
                solved=False,
                f_gap=None,
                v_lin=float("inf") if problem.lin_a is not None else 0.0,
            )
        if problem.lin_a is not None:
            assert problem.lin_b is not None
            lin_b_val = float(problem.lin_b)
            proj = project_linear_box_ref(
                z - gz, problem.lower, problem.upper, problem.lin_a, lin_b_val
            )
            box_res = float(np.max(np.abs(z - proj)))
            v_lin = float(abs(float(np.dot(problem.lin_a, z)) - lin_b_val) / (1.0 + abs(lin_b_val)))
        else:
            proj = np.clip(z - gz, problem.lower, problem.upper)
            box_res = float(np.max(np.abs(z - proj)))
            v_lin = 0.0

        viol_lo = np.maximum(0.0, problem.lower - z)
        viol_hi = np.maximum(0.0, z - problem.upper)
        v_box = float(np.max(np.maximum(viol_lo, viol_hi)))
        active_lo = np.isfinite(problem.lower) & (np.abs(z - problem.lower) <= 1.0e-7)
        active_hi = np.isfinite(problem.upper) & (np.abs(problem.upper - z) <= 1.0e-7)
        n_active_box = int(np.count_nonzero(active_lo | active_hi))
    else:
        box_res = 0.0
        v_box = 0.0
        v_lin = 0.0
        n_active_box = 0

    st_res = 0.0
    v_orth = 0.0
    for r, (X_r, G_r) in enumerate(zip(x_list, gx_list)):
        p_r = problem.p_list[r]
        if not (np.all(np.isfinite(X_r)) and np.all(np.isfinite(G_r))):
            return RefereeAudit(
                f_val=float(f_val),
                e_kkt_ref=float("inf"),
                box_residual=box_res,
                stiefel_residual=float("inf"),
                v_box=v_box,
                v_orth=float("inf"),
                n_active_box=n_active_box,
                solved=False,
                f_gap=None,
                v_lin=v_lin,
            )
        xtg = X_r.T @ G_r
        sym_xtg = 0.5 * (xtg + xtg.T)
        rgrad_euc = G_r - X_r @ sym_xtg
        st_res = max(st_res, float(np.linalg.norm(rgrad_euc, ord="fro")))

        orth_mat = X_r.T @ X_r - np.eye(p_r, dtype=np.float64)
        v_orth = max(v_orth, float(np.linalg.norm(orth_mat, ord="fro")))

    e_kkt_ref = max(box_res, st_res)
    solved = bool(e_kkt_ref <= kkt_tol and max(v_box, v_lin, v_orth) <= feas_tol)
    f_gap = float(f_val - problem.f_star) if problem.f_star is not None else None

    return RefereeAudit(
        f_val=float(f_val),
        e_kkt_ref=e_kkt_ref,
        box_residual=box_res,
        stiefel_residual=st_res,
        v_box=v_box,
        v_orth=v_orth,
        n_active_box=n_active_box,
        solved=solved,
        f_gap=f_gap,
        v_lin=v_lin,
    )


def evaluate_kkt_and_feasibility(
    problem: BenchmarkProblem,
    z: np.ndarray,
    x_list: Sequence[np.ndarray],
    kkt_tol: float = 1.0e-6,
    feas_tol: float = 1.0e-8,
) -> RefereeAudit:
    """Independently evaluate objective, gradients, `E_KKT_ref`, `V_box`, `V_lin`, and `V_orth`."""
    z_arr = np.asarray(z, dtype=np.float64).reshape(problem.d)
    xs = [
        np.asfortranarray(X_r, dtype=np.float64).reshape((n_r, p_r), order="F")
        for X_r, n_r, p_r in zip(x_list, problem.n_list, problem.p_list)
    ]
    if (problem.d > 0 and not np.all(np.isfinite(z_arr))) or any(
        not np.all(np.isfinite(X_r)) for X_r in xs
    ):
        return RefereeAudit(
            f_val=float("nan"),
            e_kkt_ref=float("inf"),
            box_residual=float("inf"),
            stiefel_residual=float("inf"),
            v_box=float("inf"),
            v_orth=float("inf"),
            n_active_box=0,
            solved=False,
            f_gap=None,
            v_lin=float("inf") if problem.lin_a is not None else 0.0,
        )

    f_val, gz, gx_list = problem.eval_fg(z_arr, xs)
    return compute_metrics_from_fg(
        problem,
        z_arr,
        xs,
        f_val,
        gz,
        gx_list,
        kkt_tol=kkt_tol,
        feas_tol=feas_tol,
    )


evaluate_solution = evaluate_kkt_and_feasibility


class EvalOracle:
    """Opaque evaluation counter and budget/timeout guard for benchmark runs.

    Caches the single most recent iterate `(z, X_list)` so external libraries
    that invoke `cost(x)` and `grad(x)` separately at the exact same point
    consume 1 oracle evaluation instead of 2.
    """

    def __init__(
        self,
        problem: BenchmarkProblem,
        max_eval: int = 5000,
        timeout_sec: float | None = None,
        enforce_budget_exception: bool = True,
        stop_on_kkt_tol: float | None = None,
        feas_tol: float = 1.0e-8,
    ) -> None:
        self.problem = problem
        self.max_eval = int(max_eval)
        self.timeout_sec = timeout_sec
        self.enforce_budget_exception = bool(enforce_budget_exception)
        self.stop_on_kkt_tol = stop_on_kkt_tol
        self.feas_tol = float(feas_tol)

        self.n_eval: int = 0
        self.start_time: float = time.perf_counter()

        self._last_z: np.ndarray | None = None
        self._last_xs: list[np.ndarray] | None = None
        self._last_out: tuple[float, np.ndarray, list[np.ndarray]] | None = None

        self.best_z: np.ndarray | None = None
        self.best_xs: list[np.ndarray] | None = None
        self.best_f: float = float("inf")
        self.target_z: np.ndarray | None = None
        self.target_xs: list[np.ndarray] | None = None

    def _matches_cached(self, z: np.ndarray, x_list: Sequence[np.ndarray]) -> bool:
        if self._last_out is None or self._last_z is None or self._last_xs is None:
            return False
        if not np.array_equal(z, self._last_z):
            return False
        for a, b in zip(x_list, self._last_xs):
            if not np.array_equal(a, b):
                return False
        return True

    def eval_fg(
        self, z: np.ndarray, x_list: Sequence[np.ndarray]
    ) -> tuple[float, np.ndarray, list[np.ndarray]]:
        """Evaluate `(f, gz, gx_list)` at `(z, x_list)`, incrementing `n_eval` on distinct points."""
        if self.timeout_sec is not None and (time.perf_counter() - self.start_time) > self.timeout_sec:
            raise WallTimeoutError(f"Exceeded wall-clock timeout ({self.timeout_sec:.1f}s).")

        if self._matches_cached(z, x_list):
            assert self._last_out is not None
            f_c, gz_c, gx_c = self._last_out
            return f_c, gz_c.copy(), [g.copy(order="F") for g in gx_c]

        if self.n_eval >= self.max_eval and self.enforce_budget_exception:
            raise BudgetExhaustedError(f"Evaluation cap max_eval={self.max_eval} reached.")

        self.n_eval += 1
        f_val, gz, gx_list = self.problem.eval_fg(z, x_list)

        z_copy = np.array(z, dtype=np.float64, copy=True)
        xs_copy = [np.array(x, dtype=np.float64, copy=True, order="F") for x in x_list]
        gz_copy = np.array(gz, dtype=np.float64, copy=True)
        gxs_copy = [np.array(g, dtype=np.float64, copy=True, order="F") for g in gx_list]

        self._last_z = z_copy
        self._last_xs = xs_copy
        self._last_out = (float(f_val), gz_copy, gxs_copy)

        if np.isfinite(f_val):
            # Record best feasible (or near-feasible) iterate seen so far
            is_feas = True
            if self.problem.d > 0:
                v_b = float(
                    np.max(
                        np.maximum(
                            0.0,
                            np.maximum(self.problem.lower - z_copy, z_copy - self.problem.upper),
                        )
                    )
                )
                if v_b > 1.0e-5:
                    is_feas = False
                if self.problem.lin_a is not None:
                    assert self.problem.lin_b is not None
                    lin_b_val = float(self.problem.lin_b)
                    v_l = abs(float(np.dot(self.problem.lin_a, z_copy)) - lin_b_val) / (
                        1.0 + abs(lin_b_val)
                    )
                    if v_l > 1.0e-4:
                        is_feas = False
            for r, X_r in enumerate(xs_copy):
                p_r = self.problem.p_list[r]
                v_o = float(np.linalg.norm(X_r.T @ X_r - np.eye(p_r), ord="fro"))
                if v_o > 1.0e-4:
                    is_feas = False
                    break

            if is_feas and float(f_val) < self.best_f:
                self.best_f = float(f_val)
                self.best_z = z_copy
                self.best_xs = xs_copy

            if self.stop_on_kkt_tol is not None:
                audit = compute_metrics_from_fg(
                    self.problem,
                    z_copy,
                    xs_copy,
                    float(f_val),
                    gz_copy,
                    gxs_copy,
                    kkt_tol=self.stop_on_kkt_tol,
                    feas_tol=self.feas_tol,
                )
                if audit.solved:
                    self.target_z = z_copy
                    self.target_xs = xs_copy
                    raise TargetKKTReachedError("Exact referee KKT + feasibility target reached.")

        return float(f_val), gz_copy.copy(), [g.copy(order="F") for g in gxs_copy]


def check_grad(
    problem: BenchmarkProblem,
    z: np.ndarray | None = None,
    x_list: Sequence[np.ndarray] | None = None,
    *,
    seed: int = 101,
    fd_step: float = 1.0e-5,
    tol: float = 1.0e-7,
) -> GradientFDReport:
    """Verify every coordinate of `(grad_z f, grad_{X_1..X_R} f)` via central finite differences."""
    if z is None or x_list is None:
        z0, x0_list = generate_initial_point(problem, seed=seed)
    else:
        z0 = np.array(z, dtype=np.float64, copy=True).reshape(problem.d)
        x0_list = [
            np.asfortranarray(X_r, dtype=np.float64).reshape((n_r, p_r), order="F")
            for X_r, n_r, p_r in zip(x_list, problem.n_list, problem.p_list)
        ]

    f0, gz_ana, gx_ana_list = problem.eval_fg(z0, x0_list)
    if not np.isfinite(f0):
        raise ValueError(f"Non-finite f0={f0} on {problem.problem_id} at seed={seed}.")

    max_abs_z = 0.0
    max_rel_z = 0.0
    if problem.d > 0:
        gz_fd = np.empty(problem.d, dtype=np.float64)
        z_work = z0.copy()
        for i in range(problem.d):
            orig = z_work[i]
            z_work[i] = orig + fd_step
            f_plus, _, _ = problem.eval_fg(z_work, x0_list)
            z_work[i] = orig - fd_step
            f_minus, _, _ = problem.eval_fg(z_work, x0_list)
            z_work[i] = orig
            gz_fd[i] = (f_plus - f_minus) / (2.0 * fd_step)

        diff_z = np.abs(gz_ana - gz_fd)
        max_abs_z = float(np.max(diff_z))
        scale_z = max(1.0, float(np.max(np.abs(gz_ana))))
        max_rel_z = max_abs_z / scale_z

    max_abs_x = 0.0
    max_rel_x = 0.0
    if problem.R > 0:
        x_work = [x.copy(order="F") for x in x0_list]
        for r, (n_r, p_r) in enumerate(zip(problem.n_list, problem.p_list)):
            gx_fd = np.empty((n_r, p_r), dtype=np.float64, order="F")
            xr = x_work[r]
            for j in range(p_r):
                for a in range(n_r):
                    orig = xr[a, j]
                    xr[a, j] = orig + fd_step
                    f_plus, _, _ = problem.eval_fg(z0, x_work)
                    xr[a, j] = orig - fd_step
                    f_minus, _, _ = problem.eval_fg(z0, x_work)
                    xr[a, j] = orig
                    gx_fd[a, j] = (f_plus - f_minus) / (2.0 * fd_step)

            diff_xr = np.abs(gx_ana_list[r] - gx_fd)
            abs_r = float(np.max(diff_xr))
            scale_r = max(1.0, float(np.max(np.abs(gx_ana_list[r]))))
            rel_r = abs_r / scale_r
            max_abs_x = max(max_abs_x, abs_r)
            max_rel_x = max(max_rel_x, rel_r)

    max_rel_tot = max(max_rel_z, max_rel_x)
    passed = bool(max_rel_tot <= tol)
    return GradientFDReport(
        problem_id=problem.problem_id,
        seed=seed,
        n_tot=problem.n_tot,
        fd_step=fd_step,
        max_abs_err_z=max_abs_z,
        max_rel_err_z=max_rel_z,
        max_abs_err_x=max_abs_x,
        max_rel_err_x=max_rel_x,
        max_rel_err_total=max_rel_tot,
        passed=passed,
    )


def verify_problem_gradient(
    problem: BenchmarkProblem,
    seed: int = 101,
    fd_step: float = 1.0e-5,
    tol: float = 1.0e-7,
) -> GradientFDReport:
    """Verify every coordinate of `(grad_z f, grad_{X_1..X_R} f)` via central finite differences."""
    return check_grad(problem, seed=seed, fd_step=fd_step, tol=tol)


def verify_all_analytical_gradients(
    problems: Sequence[BenchmarkProblem] | None = None,
    seed: int = 101,
    fd_step: float = 1.0e-5,
    tol: float = 1.0e-7,
) -> list[GradientFDReport]:
    """Run central finite-difference gradient verification across all benchmark problems."""
    probs = list(problems) if problems is not None else get_all_benchmark_problems()
    reports = [
        verify_problem_gradient(prob, seed=seed, fd_step=fd_step, tol=tol) for prob in probs
    ]
    failed = [rep.problem_id for rep in reports if not rep.passed]
    if failed:
        raise AssertionError(
            f"Analytical gradient FD check failed (> {tol:.1e}) on problems: {failed}"
        )
    return reports


def _cli_main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify analytical gradients and deterministic initializations for fortchuck benchmarks."
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default="101..110",
        help="Seed range/list (e.g., '101..110', '201..210', or '301..310').",
    )
    parser.add_argument(
        "--all-seed-gradients",
        action="store_true",
        help="Verify central FD gradients across every seed in --seeds (not only seeds[0]).",
    )
    parser.add_argument(
        "--fd-step",
        type=float,
        default=1.0e-5,
        help="Central finite-difference step size (default: 1e-5).",
    )
    parser.add_argument(
        "--tol",
        type=float,
        default=1.0e-7,
        help="Maximum relative gradient error tolerance (default: 1e-7).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON summary.",
    )
    args = parser.parse_args()

    seeds = parse_seeds(args.seeds)
    problems = get_all_benchmark_problems()
    if args.all_seed_gradients:
        fd_reports: list[GradientFDReport] = []
        for s in seeds:
            fd_reports.extend(
                verify_all_analytical_gradients(
                    problems=problems, seed=s, fd_step=args.fd_step, tol=args.tol
                )
            )
    else:
        fd_reports = verify_all_analytical_gradients(
            problems=problems, seed=seeds[0], fd_step=args.fd_step, tol=args.tol
        )

    # Verify strict primal feasibility of generated initial points across all requested seeds
    max_init_vbox = 0.0
    max_init_vlin = 0.0
    max_init_vorth = 0.0
    for prob in problems:
        for s in seeds:
            z0, x0_list = generate_initial_point(prob, seed=s)
            aud = evaluate_kkt_and_feasibility(prob, z0, x0_list)
            max_init_vbox = max(max_init_vbox, aud.v_box)
            max_init_vlin = max(max_init_vlin, aud.v_lin)
            max_init_vorth = max(max_init_vorth, aud.v_orth)
            if aud.v_box > 0.0 or aud.v_lin > 1.0e-13 or aud.v_orth > 1.0e-13:
                raise AssertionError(
                    f"Infeasible initial point on {prob.problem_id} seed={s}: "
                    f"V_box={aud.v_box:.3e}, V_lin={aud.v_lin:.3e}, V_orth={aud.v_orth:.3e}"
                )

    if args.json:
        print(
            json.dumps(
                {
                    "seeds": seeds,
                    "max_init_v_box": max_init_vbox,
                    "max_init_v_lin": max_init_vlin,
                    "max_init_v_orth": max_init_vorth,
                    "gradient_fd_reports": [asdict(r) for r in fd_reports],
                },
                indent=2,
            )
        )
    else:
        grad_note = (
            f"all {len(seeds)} seeds ({seeds[0]}..{seeds[-1]})"
            if args.all_seed_gradients
            else f"seed={seeds[0]}"
        )
        print(
            f"Verified {len(problems)} benchmark problems on {grad_note} "
            f"and initial feasibility across {len(seeds)} seeds ({seeds[0]}..{seeds[-1]}):"
        )
        print(f"  max initial V_box  = {max_init_vbox:.3e}")
        print(f"  max initial V_lin  = {max_init_vlin:.3e}")
        print(f"  max initial V_orth = {max_init_vorth:.3e}")
        shown = fd_reports[: len(problems)]
        for r in shown:
            print(
                f"  [{r.problem_id:<3s}] N_tot={r.n_tot:4d} | "
                f"rel_err_z={r.max_rel_err_z:.3e} | rel_err_X={r.max_rel_err_x:.3e} | "
                f"max_rel={r.max_rel_err_total:.3e} <= {args.tol:.1e} PASS"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli_main())
