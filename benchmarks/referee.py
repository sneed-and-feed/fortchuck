"""Independent external referee for the fortchuck blind benchmark suite.

Provides:
  1. Independent external KKT optimality residual (`E_KKT_ref`), box violation
     (`V_box`), and Stiefel orthogonality violation (`V_orth`).
  2. Opaque evaluation counter (`EvalOracle`) enforcing hard `max_eval` and
     wall-clock timeout budgets while avoiding double-counting when external
     optimizers invoke separate `cost(x)` and `grad(x)` callbacks at one point.
  3. Deterministic `PCG64` feasible `(z0, X0_list)` initialization generator
     supporting arbitrary seed specifications (default `101..110`, holdout
     `201..210` via `--seeds`).
  4. Full coordinatewise central finite-difference gradient verifier (`<= 1e-7`).
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import time
from typing import Any, Sequence
import numpy as np

from benchmarks.problems import (
    BenchmarkProblem,
    get_all_benchmark_problems,
)

DEFAULT_SEEDS: tuple[int, ...] = tuple(range(101, 111))
HOLDOUT_SEEDS: tuple[int, ...] = tuple(range(201, 211))


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


def generate_initial_point(
    problem: BenchmarkProblem, seed: int
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Generate deterministic feasible starting point `(z0, X0_list)` via PCG64(seed).

    Box coordinates with two-sided finite bounds are sampled strictly inside the
    central 80% sub-box `[l + 0.1*(u - l), u - 0.1*(u - l)]` (and fixed coords
    at `l_i == u_i`) so smooth-warping (`arctanh`) and interior-point baselines
    can start from the exact bit-identical point as projection-based solvers.
    Square Stiefel factors (`n_r == p_r`) are oriented in `SO(n_r)` (`det = +1`).
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
            )
        proj = np.clip(z - gz, problem.lower, problem.upper)
        box_res = float(np.max(np.abs(z - proj)))
        viol_lo = np.maximum(0.0, problem.lower - z)
        viol_hi = np.maximum(0.0, z - problem.upper)
        v_box = float(np.max(np.maximum(viol_lo, viol_hi)))
        active_lo = np.isfinite(problem.lower) & (np.abs(z - problem.lower) <= 1.0e-7)
        active_hi = np.isfinite(problem.upper) & (np.abs(problem.upper - z) <= 1.0e-7)
        n_active_box = int(np.count_nonzero(active_lo | active_hi))
    else:
        box_res = 0.0
        v_box = 0.0
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
            )
        xtg = X_r.T @ G_r
        sym_xtg = 0.5 * (xtg + xtg.T)
        rgrad_euc = G_r - X_r @ sym_xtg
        st_res = max(st_res, float(np.linalg.norm(rgrad_euc, ord="fro")))

        orth_mat = X_r.T @ X_r - np.eye(p_r, dtype=np.float64)
        v_orth = max(v_orth, float(np.linalg.norm(orth_mat, ord="fro")))

    e_kkt_ref = max(box_res, st_res)
    solved = bool(e_kkt_ref <= kkt_tol and max(v_box, v_orth) <= feas_tol)
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
    )


def evaluate_kkt_and_feasibility(
    problem: BenchmarkProblem,
    z: np.ndarray,
    x_list: Sequence[np.ndarray],
    kkt_tol: float = 1.0e-6,
    feas_tol: float = 1.0e-8,
) -> RefereeAudit:
    """Independently evaluate objective, gradients, `E_KKT_ref`, `V_box`, and `V_orth`."""
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


def verify_problem_gradient(
    problem: BenchmarkProblem,
    seed: int = 101,
    fd_step: float = 1.0e-5,
    tol: float = 1.0e-7,
) -> GradientFDReport:
    """Verify every coordinate of `(grad_z f, grad_{X_1..X_R} f)` via central finite differences."""
    z0, x0_list = generate_initial_point(problem, seed=seed)
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
        help="Seed range/list (e.g., '101..110' or holdout '201..210').",
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
    fd_reports = verify_all_analytical_gradients(
        problems=problems, seed=seeds[0], fd_step=args.fd_step, tol=args.tol
    )

    # Verify strict primal feasibility of generated initial points across all requested seeds
    max_init_vbox = 0.0
    max_init_vorth = 0.0
    for prob in problems:
        for s in seeds:
            z0, x0_list = generate_initial_point(prob, seed=s)
            aud = evaluate_kkt_and_feasibility(prob, z0, x0_list)
            max_init_vbox = max(max_init_vbox, aud.v_box)
            max_init_vorth = max(max_init_vorth, aud.v_orth)
            if aud.v_box > 0.0 or aud.v_orth > 1.0e-13:
                raise AssertionError(
                    f"Infeasible initial point on {prob.problem_id} seed={s}: "
                    f"V_box={aud.v_box:.3e}, V_orth={aud.v_orth:.3e}"
                )

    if args.json:
        print(
            json.dumps(
                {
                    "seeds": seeds,
                    "max_init_v_box": max_init_vbox,
                    "max_init_v_orth": max_init_vorth,
                    "gradient_fd_reports": [asdict(r) for r in fd_reports],
                },
                indent=2,
            )
        )
    else:
        print(
            f"Verified {len(problems)} benchmark problems on seed={seeds[0]} "
            f"and initial feasibility across {len(seeds)} seeds ({seeds[0]}..{seeds[-1]}):"
        )
        print(f"  max initial V_box  = {max_init_vbox:.3e}")
        print(f"  max initial V_orth = {max_init_vorth:.3e}")
        for r in fd_reports:
            print(
                f"  [{r.problem_id:<3s}] N_tot={r.n_tot:4d} | "
                f"rel_err_z={r.max_rel_err_z:.3e} | rel_err_X={r.max_rel_err_x:.3e} | "
                f"max_rel={r.max_rel_err_total:.3e} <= {args.tol:.1e} PASS"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli_main())
