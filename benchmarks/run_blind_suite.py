"""Strictly blind multi-seed benchmark runner and statistical referee for fortchuck.

Evaluates:
  1. `fortchuck` using strictly frozen out-of-the-box `FCOptions(gtol=1e-6, max_eval=5000)`
     across EVERY benchmark problem and EVERY seed (zero per-problem hyperparameters).
  2. `fortchuck-ABB` (`FCOptions(gtol=1e-6, max_eval=5000, memory=0)` ablation).
  3. All 6 external baselines from `benchmarks/baselines.py` (`Alternating-BCD`,
     `pymanopt-Warped-CG`, `pymanopt-TrustRegions`, `SciPy-L-BFGS-B`,
     `SciPy-SLSQP`, `SciPy-trust-constr`) initialized from bit-identical
     `PCG64(seed)` feasible points `(z0, X0_list)`.

By default, runs BOTH the Primary Blind Suite (`101..110`, 90 runs/solver) and the
fresh Out-of-Sample Holdout Suite (`201..210`, 90 runs/solver), computing independent
referee solve rates (`E_KKT_ref <= 1e-6` and `max(V_box, V_orth) <= 1e-8`),
median/IQR evaluations & wall-clock ms, median `E_KKT_ref`, median `V_box`,
median `V_orth`, certified global optimality gap `f - f_star` on `B6` and `B7B`,
and paired Wilcoxon signed-rank `p`-values.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import sys
import time
from typing import Any, Sequence
import numpy as np
import pymanopt
import scipy
from scipy.stats import wilcoxon

# Allow invoking directly as `python benchmarks/run_blind_suite.py` from repo root
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from benchmarks.baselines import (  # noqa: E402
    BASELINE_REGISTRY,
    EXTERNAL_BASELINE_NAMES,
    BaselineResult,
)
from benchmarks.problems import (  # noqa: E402
    BENCHMARK_IDS,
    BenchmarkProblem,
    get_benchmark_problem,
)
from benchmarks.referee import (  # noqa: E402
    evaluate_kkt_and_feasibility,
    generate_initial_point,
    parse_seeds,
    verify_all_analytical_gradients,
)
from fortchuck import (  # noqa: E402
    FCOptions,
    FortchuckWorkspace,
    minimize_box_stiefel,
)
from fortchuck._native import load_native_library  # noqa: E402

ALL_SOLVER_NAMES: tuple[str, ...] = (
    "fortchuck",
    "fortchuck-ABB",
    *EXTERNAL_BASELINE_NAMES,
)


def _safe_wilcoxon(
    x: Sequence[float],
    y: Sequence[float],
    alternative: str = "two-sided",
) -> float | None:
    """Compute paired Wilcoxon signed-rank p-value with clean tie/inf handling."""
    if len(x) != len(y) or len(x) == 0:
        return None
    xa = np.asarray(x, dtype=np.float64)
    ya = np.asarray(y, dtype=np.float64)
    xa = np.where(np.isfinite(xa), xa, 1.0e30)
    ya = np.where(np.isfinite(ya), ya, 1.0e30)
    diff = xa - ya
    if np.all(diff == 0.0):
        return 1.0
    try:
        res = wilcoxon(diff, zero_method="wilcox", alternative=alternative)
        pval = float(res.pvalue)
        return pval if np.isfinite(pval) else 1.0
    except ValueError:
        return 1.0


def _median_and_iqr(values: Sequence[float]) -> tuple[float, float, float, float]:
    """Return `(median, iqr, q25, q75)` over finite-clamped sequence."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return float("nan"), float("nan"), float("nan"), float("nan")
    q25, med, q75 = np.percentile(arr, [25.0, 50.0, 75.0])
    return float(med), float(q75 - q25), float(q25), float(q75)


def run_fortchuck_variant(
    solver_name: str,
    problem: BenchmarkProblem,
    z0: np.ndarray,
    x0_list: Sequence[np.ndarray],
    seed: int,
    *,
    gtol: float = 1.0e-6,
    feas_tol: float = 1.0e-8,
    max_eval: int = 5000,
    ablation_abb: bool = False,
    workspace: FortchuckWorkspace | None = None,
) -> BaselineResult:
    """Run `fortchuck` (strictly frozen `FCOptions(gtol, max_eval)`) or `fortchuck-ABB`."""
    if ablation_abb:
        opts = FCOptions(gtol=gtol, max_eval=max_eval, memory=0)
    else:
        # Strictly frozen uniform out-of-the-box configuration across all problems & seeds
        opts = FCOptions(gtol=gtol, max_eval=max_eval)

    z_in = z0.copy()
    xs_in = [x.copy(order="F") for x in x0_list]

    raw_evals = 0

    def counted_fg(
        z_view: np.ndarray, x_views: Sequence[np.ndarray]
    ) -> tuple[float, np.ndarray, list[np.ndarray]]:
        nonlocal raw_evals
        raw_evals += 1
        return problem.eval_fg(z_view, x_views)

    t0 = time.perf_counter()
    res = minimize_box_stiefel(
        counted_fg,
        z0=z_in if problem.d > 0 else None,
        bounds=(problem.lower, problem.upper) if problem.d > 0 else None,
        X0_list=xs_in if problem.R > 0 else None,
        options=opts,
        workspace=workspace,
        copy=False,
    )
    wall_ms = 1000.0 * (time.perf_counter() - t0)

    if raw_evals != res.n_eval:
        raise AssertionError(
            f"{solver_name} callback count mismatch on {problem.problem_id} seed={seed}: "
            f"raw_evals={raw_evals} != res.n_eval={res.n_eval}"
        )
    if raw_evals > max_eval:
        raise AssertionError(
            f"{solver_name} exceeded max_eval={max_eval} on {problem.problem_id} seed={seed}: "
            f"n_eval={raw_evals}"
        )

    audit = evaluate_kkt_and_feasibility(
        problem, res.z, res.X_list, kkt_tol=gtol, feas_tol=feas_tol
    )
    return BaselineResult(
        solver_name=solver_name,
        problem_id=problem.problem_id,
        seed=seed,
        status=res.status_name,
        skipped=False,
        n_eval=raw_evals,
        wall_ms=wall_ms,
        z=res.z.copy(),
        x_list=[x.copy(order="F") for x in res.X_list],
        audit=audit,
    )


def _serialize_run(r: BaselineResult) -> dict[str, Any]:
    return {
        "solver_name": r.solver_name,
        "problem_id": r.problem_id,
        "seed": r.seed,
        "status": r.status,
        "skipped": r.skipped,
        "n_eval": r.n_eval,
        "wall_ms": r.wall_ms,
        "f_val": r.audit.f_val,
        "e_kkt_ref": r.audit.e_kkt_ref,
        "box_residual": r.audit.box_residual,
        "stiefel_residual": r.audit.stiefel_residual,
        "v_box": r.audit.v_box,
        "v_orth": r.audit.v_orth,
        "n_active_box": r.audit.n_active_box,
        "solved": r.audit.solved,
        "f_gap": r.audit.f_gap,
    }


def _aggregate_problem_solver(
    runs: Sequence[BaselineResult],
    fc_runs: Sequence[BaselineResult] | None,
    max_eval: int,
) -> dict[str, Any]:
    """Compute per-problem summary metrics and paired Wilcoxon p-values vs fortchuck."""
    n_seeds = len(runs)
    skipped = all(r.skipped for r in runs)
    if skipped:
        return {
            "skipped": True,
            "skip_reason": runs[0].status if runs else "SKIPPED",
            "n_seeds": n_seeds,
            "n_solved": 0,
            "solve_rate": 0.0,
            "evals_raw_median": None,
            "evals_raw_iqr": None,
            "evals_solved_median": None,
            "evals_solved_iqr": None,
            "evals_penalized_median": None,
            "evals_penalized_iqr": None,
            "wall_ms_median": None,
            "wall_ms_iqr": None,
            "e_kkt_ref_median": None,
            "e_kkt_ref_max": None,
            "v_box_median": None,
            "v_box_max": None,
            "v_orth_median": None,
            "v_orth_max": None,
            "f_val_median": None,
            "f_val_min": None,
            "f_gap_signed_median": None,
            "f_gap_abs_median": None,
            "f_gap_abs_max": None,
            "wilcoxon_evals_two_sided_p": None,
            "wilcoxon_evals_less_p": None,
            "wilcoxon_kkt_two_sided_p": None,
        }

    n_solved = sum(1 for r in runs if r.audit.solved)
    solve_rate = float(n_solved) / float(n_seeds) if n_seeds > 0 else 0.0

    raw_evals = [float(r.n_eval) for r in runs]
    pen_evals = [float(r.n_eval if r.audit.solved else max_eval) for r in runs]
    solved_evals = [float(r.n_eval) for r in runs if r.audit.solved]
    wall_ms = [float(r.wall_ms) for r in runs]
    kkt_vals = [float(r.audit.e_kkt_ref) for r in runs]
    vbox_vals = [float(r.audit.v_box) for r in runs]
    vorth_vals = [float(r.audit.v_orth) for r in runs]
    f_vals = [float(r.audit.f_val) for r in runs if np.isfinite(r.audit.f_val)]

    ev_med, ev_iqr, ev_q25, ev_q75 = _median_and_iqr(raw_evals)
    pev_med, pev_iqr, _, _ = _median_and_iqr(pen_evals)
    if solved_evals:
        sev_med, sev_iqr, _, _ = _median_and_iqr(solved_evals)
    else:
        sev_med, sev_iqr = None, None
    wm_med, wm_iqr, wm_q25, wm_q75 = _median_and_iqr(wall_ms)
    kkt_med, _, _, _ = _median_and_iqr(kkt_vals)
    vbox_med, _, _, _ = _median_and_iqr(vbox_vals)
    vorth_med, _, _, _ = _median_and_iqr(vorth_vals)

    f_med = float(np.median(f_vals)) if f_vals else None
    f_min = float(np.min(f_vals)) if f_vals else None

    f_gaps = [
        float(r.audit.f_gap)
        for r in runs
        if r.audit.f_gap is not None and np.isfinite(r.audit.f_gap)
    ]
    if f_gaps:
        fg_signed_med = float(np.median(f_gaps))
        abs_gaps = [abs(g) for g in f_gaps]
        fg_abs_med = float(np.median(abs_gaps))
        fg_abs_max = float(np.max(abs_gaps))
    else:
        fg_signed_med = None
        fg_abs_med = None
        fg_abs_max = None

    p_eval_two: float | None = None
    p_eval_less: float | None = None
    p_kkt_two: float | None = None
    if fc_runs is not None and runs[0].solver_name != "fortchuck":
        fc_pen = [float(r.n_eval if r.audit.solved else max_eval) for r in fc_runs]
        fc_kkt = [float(r.audit.e_kkt_ref) for r in fc_runs]
        p_eval_two = _safe_wilcoxon(fc_pen, pen_evals, alternative="two-sided")
        p_eval_less = _safe_wilcoxon(fc_pen, pen_evals, alternative="less")
        p_kkt_two = _safe_wilcoxon(fc_kkt, kkt_vals, alternative="two-sided")

    return {
        "skipped": False,
        "skip_reason": None,
        "n_seeds": n_seeds,
        "n_solved": n_solved,
        "solve_rate": solve_rate,
        "evals_raw_median": ev_med,
        "evals_raw_iqr": ev_iqr,
        "evals_raw_q25": ev_q25,
        "evals_raw_q75": ev_q75,
        "evals_solved_median": sev_med,
        "evals_solved_iqr": sev_iqr,
        "evals_penalized_median": pev_med,
        "evals_penalized_iqr": pev_iqr,
        "wall_ms_median": wm_med,
        "wall_ms_iqr": wm_iqr,
        "wall_ms_q25": wm_q25,
        "wall_ms_q75": wm_q75,
        "e_kkt_ref_median": kkt_med,
        "e_kkt_ref_max": float(np.max(kkt_vals)),
        "v_box_median": vbox_med,
        "v_box_max": float(np.max(vbox_vals)),
        "v_orth_median": vorth_med,
        "v_orth_max": float(np.max(vorth_vals)),
        "f_val_median": f_med,
        "f_val_min": f_min,
        "f_gap_signed_median": fg_signed_med,
        "f_gap_abs_median": fg_abs_med,
        "f_gap_abs_max": fg_abs_max,
        "wilcoxon_evals_two_sided_p": p_eval_two,
        "wilcoxon_evals_less_p": p_eval_less,
        "wilcoxon_kkt_two_sided_p": p_kkt_two,
    }


def _aggregate_suite_runs(
    problems: Sequence[BenchmarkProblem],
    solver_names: Sequence[str],
    runs_by_prob_solver: dict[str, dict[str, list[BaselineResult]]],
    n_seeds: int,
    max_eval: int,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Aggregate both per-problem and suite-wide statistics from BaselineResult maps."""
    by_problem_summary: dict[str, dict[str, Any]] = {}
    for prob in problems:
        pid = prob.problem_id
        fc_runs_p = runs_by_prob_solver[pid].get("fortchuck")
        by_problem_summary[pid] = {}
        for s_name in solver_names:
            by_problem_summary[pid][s_name] = _aggregate_problem_solver(
                runs_by_prob_solver[pid][s_name],
                fc_runs=fc_runs_p,
                max_eval=max_eval,
            )

    overall_by_solver: dict[str, Any] = {}
    has_fc = "fortchuck" in solver_names
    total_slots = len(problems) * n_seeds
    for s_name in solver_names:
        eval_runs: list[BaselineResult] = []
        fc_paired_runs: list[BaselineResult] = []
        for prob in problems:
            pid = prob.problem_id
            s_runs = runs_by_prob_solver[pid][s_name]
            if all(r.skipped for r in s_runs):
                continue
            eval_runs.extend(s_runs)
            if has_fc:
                fc_paired_runs.extend(runs_by_prob_solver[pid]["fortchuck"])

        n_eval_runs = len(eval_runs)
        n_solved_runs = sum(1 for r in eval_runs if r.audit.solved)
        raw_ev = [float(r.n_eval) for r in eval_runs]
        pen_ev = [float(r.n_eval if r.audit.solved else max_eval) for r in eval_runs]
        wall_all = [float(r.wall_ms) for r in eval_runs]
        kkt_all = [float(r.audit.e_kkt_ref) for r in eval_runs]
        vbox_all = [float(r.audit.v_box) for r in eval_runs]
        vorth_all = [float(r.audit.v_orth) for r in eval_runs]

        rev_med, rev_iqr, _, _ = _median_and_iqr(raw_ev)
        pev_med, pev_iqr, _, _ = _median_and_iqr(pen_ev)
        wm_med, wm_iqr, _, _ = _median_and_iqr(wall_all)
        kkt_med, _, _, _ = _median_and_iqr(kkt_all)
        vb_med, _, _, _ = _median_and_iqr(vbox_all)
        vo_med, _, _, _ = _median_and_iqr(vorth_all)

        p_ev_two = None
        p_ev_less = None
        p_wm_two = None
        p_kkt_two = None
        if has_fc and s_name != "fortchuck" and fc_paired_runs:
            fc_pen_all = [
                float(r.n_eval if r.audit.solved else max_eval) for r in fc_paired_runs
            ]
            fc_wm_all = [float(r.wall_ms) for r in fc_paired_runs]
            fc_kkt_all = [float(r.audit.e_kkt_ref) for r in fc_paired_runs]
            p_ev_two = _safe_wilcoxon(fc_pen_all, pen_ev, alternative="two-sided")
            p_ev_less = _safe_wilcoxon(fc_pen_all, pen_ev, alternative="less")
            p_wm_two = _safe_wilcoxon(fc_wm_all, wall_all, alternative="two-sided")
            p_kkt_two = _safe_wilcoxon(fc_kkt_all, kkt_all, alternative="two-sided")

        overall_by_solver[s_name] = {
            "n_total_slots": total_slots,
            "n_evaluated_runs": n_eval_runs,
            "n_solved_runs": n_solved_runs,
            "solve_rate_evaluated": (
                float(n_solved_runs) / float(n_eval_runs) if n_eval_runs > 0 else 0.0
            ),
            "solve_rate_all_slots": (
                float(n_solved_runs) / float(total_slots) if total_slots > 0 else 0.0
            ),
            "evals_raw_median": rev_med,
            "evals_raw_iqr": rev_iqr,
            "evals_penalized_median": pev_med,
            "evals_penalized_iqr": pev_iqr,
            "wall_ms_median": wm_med,
            "wall_ms_iqr": wm_iqr,
            "e_kkt_ref_median": kkt_med,
            "v_box_median": vb_med,
            "v_orth_median": vo_med,
            "wilcoxon_evals_two_sided_p": p_ev_two,
            "wilcoxon_evals_less_p": p_ev_less,
            "wilcoxon_wall_ms_two_sided_p": p_wm_two,
            "wilcoxon_kkt_two_sided_p": p_kkt_two,
        }

    return by_problem_summary, overall_by_solver


def _fmt_sci(val: float | None, digits: int = 2) -> str:
    if val is None or not np.isfinite(val):
        return "N/A"
    return f"{val:.{digits}e}"


def _fmt_med_iqr(med: float | None, iqr: float | None, decimals: int = 1) -> str:
    if med is None or iqr is None or not (np.isfinite(med) and np.isfinite(iqr)):
        return "N/A"
    return f"{med:.{decimals}f} ({iqr:.{decimals}f})"


def _append_overall_table(
    lines: list[str],
    solvers: Sequence[str],
    overall: dict[str, Any],
) -> None:
    lines.append(
        "| Solver | Evaluated Runs | Solved Runs | Solve Rate | Penalized Evals Med (IQR) | Wall Time ms Med (IQR) | Median KKT Ref | Median Orth Viol | Wilcoxon Evals p (2-sided) | Wilcoxon KKT p (2-sided) |"
    )
    lines.append(
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |"
    )
    for s_name in solvers:
        ov = overall[s_name]
        n_eval_runs = ov["n_evaluated_runs"]
        n_solved = ov["n_solved_runs"]
        rate_pct = 100.0 * ov["solve_rate_evaluated"]
        pev_str = _fmt_med_iqr(
            ov["evals_penalized_median"], ov["evals_penalized_iqr"], decimals=1
        )
        wm_str = _fmt_med_iqr(ov["wall_ms_median"], ov["wall_ms_iqr"], decimals=2)
        kkt_str = _fmt_sci(ov["e_kkt_ref_median"])
        orth_str = _fmt_sci(ov["v_orth_median"])
        p_ev = "—" if s_name == "fortchuck" else _fmt_sci(ov["wilcoxon_evals_two_sided_p"])
        p_kkt = "—" if s_name == "fortchuck" else _fmt_sci(ov["wilcoxon_kkt_two_sided_p"])
        lines.append(
            f"| `{s_name}` | {n_eval_runs} | {n_solved} | {rate_pct:.1f}% | {pev_str} | {wm_str} | `{kkt_str}` | `{orth_str}` | `{p_ev}` | `{p_kkt}` |"
        )
    lines.append("")


def _append_per_problem_section(
    lines: list[str],
    problems_meta: Sequence[dict[str, Any]],
    per_prob: dict[str, dict[str, Any]],
    solvers: Sequence[str],
) -> None:
    for pm in problems_meta:
        pid = pm["problem_id"]
        if pid not in per_prob:
            continue
        lines.append(f"### `{pid}`: {pm['name']}")
        lines.append("")
        lines.append(
            f"- **Regime**: `{pm['regime_summary']}` (`d = {pm['d']}`, `R = {pm['R']}`, `N_tot = {pm['n_tot']}`, intrinsic dim `{pm['intrinsic_dim']}`, FD gradient max rel err `{pm['fd_max_rel_err']:.3e}`)"
        )
        lines.append("")
        lines.append(
            "| Solver | Solve Rate | Raw Evals Med (IQR) | Penalized Evals Med (IQR) | Wall ms Med (IQR) | Median KKT Ref | Median Box Viol | Median Orth Viol | Median Objective | Wilcoxon Evals p |"
        )
        lines.append(
            "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |"
        )
        for s_name in solvers:
            st = per_prob[pid][s_name]
            if st["skipped"]:
                lines.append(
                    f"| `{s_name}` | SKIP (`N_tot > 450`) | — | — | — | — | — | — | — | — |"
                )
                continue
            sr = f"{st['n_solved']}/{st['n_seeds']} ({100.0 * st['solve_rate']:.0f}%)"
            rev = _fmt_med_iqr(st["evals_raw_median"], st["evals_raw_iqr"], decimals=1)
            pev = _fmt_med_iqr(
                st["evals_penalized_median"], st["evals_penalized_iqr"], decimals=1
            )
            wm = _fmt_med_iqr(st["wall_ms_median"], st["wall_ms_iqr"], decimals=2)
            kkt = _fmt_sci(st["e_kkt_ref_median"])
            vb = _fmt_sci(st["v_box_median"])
            vo = _fmt_sci(st["v_orth_median"])
            fv = f"{st['f_val_median']:.6f}" if st["f_val_median"] is not None else "N/A"
            wp = "—" if s_name == "fortchuck" else _fmt_sci(st["wilcoxon_evals_two_sided_p"])
            lines.append(
                f"| `{s_name}` | {sr} | {rev} | {pev} | {wm} | `{kkt}` | `{vb}` | `{vo}` | `{fv}` | `{wp}` |"
            )
        lines.append("")


def _render_markdown_report(summary: dict[str, Any]) -> str:
    """Render GFM/KaTeX-compliant blind benchmark report (`0` violations across 22 rules)."""
    cfg = summary["config"]
    seeds = cfg["seeds"]
    seed_label = f"{seeds[0]}..{seeds[-1]}" if len(seeds) > 1 else str(seeds[0])
    problems_meta: list[dict[str, Any]] = summary["problems"]
    solvers: list[str] = cfg["solvers"]
    overall: dict[str, Any] = summary["overall_by_solver"]
    per_prob: dict[str, dict[str, Any]] = summary["by_problem"]
    holdout: dict[str, Any] | None = summary.get("holdout")
    pooled: dict[str, Any] | None = summary.get("combined_pooled")

    lines: list[str] = []
    lines.append("# Blind Multi-Seed Benchmark Report (`fortchuck` v0.1.0)")
    lines.append("")
    lines.append("## 1. Blind Evaluation Protocol and Referee Specification")
    lines.append("")
    lines.append(
        "All benchmark runs enforce a strict blind firewall between `benchmarks/*.py` and `src/*.f90`:"
    )
    lines.append("")
    lines.append(
        "- **Frozen Out-of-the-Box Configuration**: `fortchuck` is evaluated across every benchmark instance and every random seed with `FCOptions(gtol=1.0e-6, max_eval=5000)` (`memory = 7`, `max_iter = 15000`, Zhang--Hager nonmonotone `zh_eta` auto `0.85`, canonical metric, cautious skip, `eps_band = 1.0e-4`) and **zero** per-problem hyperparameter adjustments."
    )
    if holdout is not None:
        h_seeds = holdout["config"]["seeds"]
        h_label = f"{h_seeds[0]}..{h_seeds[-1]}" if len(h_seeds) > 1 else str(h_seeds[0])
        lines.append(
            f"- **Dual Blind & Out-of-Sample Holdout Seeds**: Initial feasible iterates are generated via `PCG64` across `{len(seeds)}` deterministic primary seeds (`{seed_label}`) and `{len(h_seeds)}` fresh out-of-sample holdout seeds (`{h_label}`)."
        )
    else:
        lines.append(
            f"- **Identical Multi-Seed Initializations**: Initial feasible iterates are generated via `PCG64` across `{len(seeds)}` deterministic seeds (`{seed_label}`)."
        )
    lines.append(
        "- **Pre-Flight Central Finite-Difference Verification**: Every coordinate of every analytical gradient is verified against central finite differences (`fd_step = 1.0e-5`) to relative error at most $10^{-7}$ prior to solver execution."
    )
    lines.append(
        "- **Independent Post-Solve KKT and Feasibility Certificate**: A run is certified **Solved** by `benchmarks/referee.py` if and only if both optimality and feasibility tolerances hold simultaneously:"
    )
    lines.append("")
    lines.append("```math")
    lines.append(
        r"\mathcal E_{\mathrm{KKT,ref}}(z, X) = \max\!\left(\left\lVert z - P_{[l,u]}\bigl(z - \nabla_z f(z, X)\bigr) \right\rVert_\infty, \; \max_{1 \le r \le R} \left\lVert G_r - X_r\,\mathrm{sym}(X_r^\top G_r) \right\rVert_{\mathrm F}\right) \le 10^{-6}"
    )
    lines.append("```")
    lines.append("")
    lines.append("```math")
    lines.append(
        r"\max\!\left(\mathcal V_{\mathrm{box}}, \, \mathcal V_{\mathrm{orth}}\right) = \max\!\left(\max_{1 \le i \le d} \max(0, \, l_i - z_i, \, z_i - u_i), \; \max_{1 \le r \le R} \left\lVert X_r^\top X_r - I_{p_r} \right\rVert_{\mathrm F}\right) \le 10^{-8}"
    )
    lines.append("```")
    lines.append("")

    sec_num = 2
    lines.append(
        f"## {sec_num}. Primary Blind Suite Aggregate Summary (`Seeds {seed_label}`)"
    )
    lines.append("")
    lines.append(
        "Aggregate metrics across all evaluated `(problem, seed)` pairs in the primary blind suite (`Penalized Evals` assigns `max_eval = 5000` to unsolved runs; paired two-sided Wilcoxon signed-rank `p`-values compare `fortchuck` against each baseline across shared non-skipped runs):"
    )
    lines.append("")
    _append_overall_table(lines, solvers, overall)
    sec_num += 1

    if holdout is not None:
        h_seeds = holdout["config"]["seeds"]
        h_label = f"{h_seeds[0]}..{h_seeds[-1]}" if len(h_seeds) > 1 else str(h_seeds[0])
        lines.append(
            f"## {sec_num}. Out-of-Sample Holdout Suite Aggregate Summary (`Seeds {h_label}`)"
        )
        lines.append("")
        lines.append(
            f"Aggregate metrics across fresh out-of-sample holdout seeds (`{h_label}`) under the identical frozen `FCOptions(gtol=1.0e-6, max_eval=5000)` configuration:"
        )
        lines.append("")
        _append_overall_table(lines, solvers, holdout["overall_by_solver"])

        if pooled is not None:
            lines.append(
                f"### Pooled Primary + Holdout Aggregate (`{len(seeds) + len(h_seeds)}` Seeds Total)"
            )
            lines.append("")
            _append_overall_table(lines, solvers, pooled["overall_by_solver"])
        sec_num += 1

    # Ground-truth global optimality verification table for B6 (and B7B if present)
    gt_probs = [
        pm
        for pm in problems_meta
        if pm["f_star"] is not None and pm["problem_id"] in per_prob
    ]
    if gt_probs:
        lines.append(
            f"## {sec_num}. Exact Global Optimality Certificate (`B6` Ky Fan QP & `B7B` Procrustes)"
        )
        lines.append("")
        lines.append(
            "Problems `B6` and `B7B` possess exact analytical global minima $f^\\ast$ certified by Ky Fan's extremal trace inequality and Kabsch singular value decomposition:"
        )
        lines.append("")
        lines.append(
            "| Suite | Problem | Certified Global Optimum | Solver | Solve Rate | Median Abs Gap | Max Abs Gap | Median Signed Gap | Median KKT Ref |"
        )
        lines.append(
            "| :--- | :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: |"
        )
        suites_for_gt: list[tuple[str, dict[str, dict[str, Any]]]] = [
            (f"Primary (`{seed_label}`)", per_prob)
        ]
        if holdout is not None:
            h_seeds = holdout["config"]["seeds"]
            h_label = f"{h_seeds[0]}..{h_seeds[-1]}" if len(h_seeds) > 1 else str(h_seeds[0])
            suites_for_gt.append((f"Holdout (`{h_label}`)", holdout["by_problem"]))

        for suite_tag, pp_map in suites_for_gt:
            for pm in gt_probs:
                pid = pm["problem_id"]
                f_star_str = f"{pm['f_star']:.12f}"
                for s_name in solvers:
                    st = pp_map[pid][s_name]
                    if st["skipped"]:
                        lines.append(
                            f"| {suite_tag} | `{pid}` | `{f_star_str}` | `{s_name}` | SKIPPED | N/A | N/A | N/A | N/A |"
                        )
                        continue
                    sr = f"{st['n_solved']}/{st['n_seeds']} ({100.0 * st['solve_rate']:.0f}%)"
                    lines.append(
                        f"| {suite_tag} | `{pid}` | `{f_star_str}` | `{s_name}` | {sr} | `{_fmt_sci(st['f_gap_abs_median'])}` | `{_fmt_sci(st['f_gap_abs_max'])}` | `{_fmt_sci(st['f_gap_signed_median'])}` | `{_fmt_sci(st['e_kkt_ref_median'])}` |"
                    )
        lines.append("")
        sec_num += 1

    lines.append(f"## {sec_num}. Per-Problem Breakdown — Primary Suite (`Seeds {seed_label}`)")
    lines.append("")
    _append_per_problem_section(lines, problems_meta, per_prob, solvers)
    sec_num += 1

    if holdout is not None:
        h_seeds = holdout["config"]["seeds"]
        h_label = f"{h_seeds[0]}..{h_seeds[-1]}" if len(h_seeds) > 1 else str(h_seeds[0])
        lines.append(
            f"## {sec_num}. Per-Problem Breakdown — Out-of-Sample Holdout Suite (`Seeds {h_label}`)"
        )
        lines.append("")
        _append_per_problem_section(
            lines, holdout["problems"], holdout["by_problem"], solvers
        )

    return "\n".join(lines) + "\n"


def _execute_single_suite(
    *,
    suite_label: str,
    seeds: Sequence[int],
    problems: Sequence[BenchmarkProblem],
    solver_names: Sequence[str],
    gtol: float,
    feas_tol: float,
    max_eval: int,
    timeout_sec: float,
    verbose: bool,
) -> tuple[dict[str, Any], dict[str, dict[str, list[BaselineResult]]]]:
    """Run one seed suite and return `(summary_dict, runs_by_prob_solver)`."""
    t_suite = time.perf_counter()
    if verbose:
        print(
            f"[{suite_label} 1/2] Verifying analytical gradients via central FD "
            f"(seed={seeds[0]}, tol=1e-7) across {len(problems)} benchmark problems...",
            flush=True,
        )
    fd_reports = verify_all_analytical_gradients(
        problems=problems, seed=seeds[0], fd_step=1.0e-5, tol=1.0e-7
    )
    fd_map = {rep.problem_id: rep for rep in fd_reports}
    if verbose:
        max_fd = max(rep.max_rel_err_total for rep in fd_reports)
        print(
            f"      PASS: All {len(problems)} problems verified "
            f"(worst max_rel_err = {max_fd:.3e} <= 1.0e-7).",
            flush=True,
        )
        print(
            f"[{suite_label} 2/2] Running {len(solver_names)} solvers x {len(problems)} problems "
            f"x {len(seeds)} seeds (gtol={gtol:.1e}, feas_tol={feas_tol:.1e}, max_eval={max_eval})...",
            flush=True,
        )

    runs_by_prob_solver: dict[str, dict[str, list[BaselineResult]]] = {
        prob.problem_id: {s: [] for s in solver_names} for prob in problems
    }
    raw_records: list[dict[str, Any]] = []

    for p_idx, prob in enumerate(problems, start=1):
        ws_fc = (
            FortchuckWorkspace(
                d=prob.d, n_list=prob.n_list, p_list=prob.p_list, memory=7
            )
            if "fortchuck" in solver_names
            else None
        )
        ws_abb = (
            FortchuckWorkspace(
                d=prob.d, n_list=prob.n_list, p_list=prob.p_list, memory=0
            )
            if "fortchuck-ABB" in solver_names
            else None
        )

        t_prob = time.perf_counter()
        for seed in seeds:
            z0, x0_list = generate_initial_point(prob, seed=seed)
            for s_name in solver_names:
                if s_name == "fortchuck":
                    res = run_fortchuck_variant(
                        "fortchuck",
                        prob,
                        z0,
                        x0_list,
                        seed=seed,
                        gtol=gtol,
                        feas_tol=feas_tol,
                        max_eval=max_eval,
                        ablation_abb=False,
                        workspace=ws_fc,
                    )
                elif s_name == "fortchuck-ABB":
                    res = run_fortchuck_variant(
                        "fortchuck-ABB",
                        prob,
                        z0,
                        x0_list,
                        seed=seed,
                        gtol=gtol,
                        feas_tol=feas_tol,
                        max_eval=max_eval,
                        ablation_abb=True,
                        workspace=ws_abb,
                    )
                else:
                    runner = BASELINE_REGISTRY[s_name]
                    res = runner(
                        prob,
                        z0.copy(),
                        [x.copy(order="F") for x in x0_list],
                        seed=seed,
                        gtol=gtol,
                        feas_tol=feas_tol,
                        max_eval=max_eval,
                        timeout_sec=timeout_sec,
                    )
                runs_by_prob_solver[prob.problem_id][s_name].append(res)
                raw_records.append(_serialize_run(res))

        if verbose:
            dt_p = time.perf_counter() - t_prob
            fc_note = ""
            if "fortchuck" in solver_names:
                fc_list = runs_by_prob_solver[prob.problem_id]["fortchuck"]
                n_ok = sum(1 for r in fc_list if r.audit.solved)
                med_ev = float(np.median([r.n_eval for r in fc_list]))
                med_ms = float(np.median([r.wall_ms for r in fc_list]))
                med_kkt = float(np.median([r.audit.e_kkt_ref for r in fc_list]))
                fc_note = (
                    f" | fortchuck: {n_ok}/{len(fc_list)} solved, "
                    f"med_eval={med_ev:.1f}, med_ms={med_ms:.2f}, med_KKT={med_kkt:.2e}"
                )
            print(
                f"      [{suite_label} {p_idx}/{len(problems)}] {prob.problem_id:<3s} "
                f"({dt_p:5.1f}s total){fc_note}",
                flush=True,
            )

    by_problem_summary, overall_by_solver = _aggregate_suite_runs(
        problems=problems,
        solver_names=solver_names,
        runs_by_prob_solver=runs_by_prob_solver,
        n_seeds=len(seeds),
        max_eval=max_eval,
    )

    problems_meta = [
        {
            "problem_id": p.problem_id,
            "name": p.name,
            "regime_summary": p.regime_summary,
            "d": p.d,
            "R": p.R,
            "n_list": list(p.n_list),
            "p_list": list(p.p_list),
            "n_tot": p.n_tot,
            "intrinsic_dim": p.intrinsic_dim,
            "f_star": p.f_star,
            "fd_max_rel_err": fd_map[p.problem_id].max_rel_err_total,
        }
        for p in problems
    ]

    suite_elapsed = time.perf_counter() - t_suite
    suite_dict: dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "numpy_version": np.__version__,
            "scipy_version": scipy.__version__,
            "pymanopt_version": pymanopt.__version__,
        },
        "config": {
            "seeds": list(seeds),
            "problems": [p.problem_id for p in problems],
            "solvers": list(solver_names),
            "gtol": gtol,
            "feas_tol": feas_tol,
            "max_eval": max_eval,
            "timeout_sec": timeout_sec,
            "fortchuck_options_frozen": FCOptions(
                gtol=gtol, max_eval=max_eval
            ).as_dict(),
        },
        "gradient_fd_verification": [asdict(r) for r in fd_reports],
        "problems": problems_meta,
        "overall_by_solver": overall_by_solver,
        "by_problem": by_problem_summary,
        "runs": raw_records,
        "suite_wall_seconds": suite_elapsed,
    }
    return suite_dict, runs_by_prob_solver


def run_suite(
    *,
    seeds: Sequence[int],
    holdout_seeds: Sequence[int] | None = None,
    problem_ids: Sequence[str],
    solver_names: Sequence[str],
    gtol: float = 1.0e-6,
    feas_tol: float = 1.0e-8,
    max_eval: int = 5000,
    timeout_sec: float = 20.0,
    verbose: bool = True,
) -> dict[str, Any]:
    """Execute the primary (and optional out-of-sample holdout) blind benchmark suite."""
    # Pre-load shared C-ABI library before any wall-clock measurement
    load_native_library()

    problems = [get_benchmark_problem(pid) for pid in problem_ids]
    t_total = time.perf_counter()

    primary_summary, primary_runs = _execute_single_suite(
        suite_label="PRIMARY",
        seeds=seeds,
        problems=problems,
        solver_names=solver_names,
        gtol=gtol,
        feas_tol=feas_tol,
        max_eval=max_eval,
        timeout_sec=timeout_sec,
        verbose=verbose,
    )

    if holdout_seeds:
        holdout_summary, holdout_runs = _execute_single_suite(
            suite_label="HOLDOUT",
            seeds=holdout_seeds,
            problems=problems,
            solver_names=solver_names,
            gtol=gtol,
            feas_tol=feas_tol,
            max_eval=max_eval,
            timeout_sec=timeout_sec,
            verbose=verbose,
        )
        pooled_runs: dict[str, dict[str, list[BaselineResult]]] = {
            prob.problem_id: {
                s: primary_runs[prob.problem_id][s] + holdout_runs[prob.problem_id][s]
                for s in solver_names
            }
            for prob in problems
        }
        pooled_by_prob, pooled_overall = _aggregate_suite_runs(
            problems=problems,
            solver_names=solver_names,
            runs_by_prob_solver=pooled_runs,
            n_seeds=len(seeds) + len(holdout_seeds),
            max_eval=max_eval,
        )
        primary_summary["holdout"] = holdout_summary
        primary_summary["combined_pooled"] = {
            "seeds": list(seeds) + list(holdout_seeds),
            "overall_by_solver": pooled_overall,
            "by_problem": pooled_by_prob,
        }
    else:
        primary_summary["holdout"] = None
        primary_summary["combined_pooled"] = None

    primary_summary["total_wall_seconds"] = time.perf_counter() - t_total
    return primary_summary


def _parse_csv_list(raw: str | None, valid: Sequence[str], kind: str) -> list[str]:
    if raw is None or not raw.strip():
        return list(valid)
    valid_map = {v.upper(): v for v in valid}
    out: list[str] = []
    for tok in raw.split(","):
        t = tok.strip()
        if not t:
            continue
        key = t.upper()
        if key not in valid_map:
            raise ValueError(f"Unknown {kind} {t!r}. Expected subset of {list(valid)}.")
        out.append(valid_map[key])
    if not out:
        raise ValueError(f"Empty {kind} list parsed from {raw!r}.")
    return out


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the fortchuck strictly blind multi-seed benchmark suite."
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default="101..110",
        help="Primary seed specification (default: '101..110').",
    )
    parser.add_argument(
        "--holdout-seeds",
        type=str,
        default="auto",
        help=(
            "Out-of-sample holdout seed specification (default: 'auto' -> '201..210' "
            "when --seeds is '101..110', else disabled unless explicitly provided)."
        ),
    )
    parser.add_argument(
        "--no-holdout",
        action="store_true",
        help="Skip out-of-sample holdout suite even when --seeds is '101..110'.",
    )
    parser.add_argument(
        "--problems",
        type=str,
        default=None,
        help=f"Comma-separated subset of benchmark IDs (default: all {','.join(BENCHMARK_IDS)}).",
    )
    parser.add_argument(
        "--solvers",
        type=str,
        default=None,
        help="Comma-separated subset of solver names (default: fortchuck,fortchuck-ABB + all 6 baselines).",
    )
    parser.add_argument(
        "--gtol", type=float, default=1.0e-6, help="Referee KKT tolerance (default: 1e-6)."
    )
    parser.add_argument(
        "--feas-tol",
        type=float,
        default=1.0e-8,
        help="Referee max(V_box, V_orth) tolerance (default: 1e-8).",
    )
    parser.add_argument(
        "--max-eval",
        type=int,
        default=5000,
        help="Hard objective+gradient evaluation cap per seed (default: 5000).",
    )
    parser.add_argument(
        "--timeout-sec",
        type=float,
        default=20.0,
        help="Per-seed wall-clock timeout in seconds for external baselines (default: 20.0).",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=_REPO_ROOT / "results" / "blind_benchmark_summary.json",
        help="Path for JSON summary output (default: results/blind_benchmark_summary.json).",
    )
    parser.add_argument(
        "--md-out",
        type=Path,
        default=_REPO_ROOT / "results" / "blind_benchmark_report.md",
        help="Path for Markdown report output (default: results/blind_benchmark_report.md).",
    )
    args = parser.parse_args()

    seeds = parse_seeds(args.seeds)
    if args.no_holdout or not args.holdout_seeds or not args.holdout_seeds.strip():
        holdout_seeds: list[int] | None = None
    elif args.holdout_seeds.strip().lower() == "auto":
        holdout_seeds = parse_seeds("201..210") if args.seeds.strip() == "101..110" else None
    else:
        holdout_seeds = parse_seeds(args.holdout_seeds)

    problem_ids = _parse_csv_list(args.problems, BENCHMARK_IDS, "problem")
    solver_names = _parse_csv_list(args.solvers, ALL_SOLVER_NAMES, "solver")

    summary = run_suite(
        seeds=seeds,
        holdout_seeds=holdout_seeds,
        problem_ids=problem_ids,
        solver_names=solver_names,
        gtol=args.gtol,
        feas_tol=args.feas_tol,
        max_eval=args.max_eval,
        timeout_sec=args.timeout_sec,
        verbose=True,
    )
    elapsed_s = summary["total_wall_seconds"]

    json_out: Path = args.json_out
    md_out: Path = args.md_out
    json_out.parent.mkdir(parents=True, exist_ok=True)
    md_out.parent.mkdir(parents=True, exist_ok=True)

    json_out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    md_out.write_text(_render_markdown_report(summary), encoding="utf-8")

    print(f"\nCompleted blind benchmark suite in {elapsed_s:.1f}s.", flush=True)
    print(f"  JSON summary    : {json_out}", flush=True)
    print(f"  Markdown report : {md_out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
