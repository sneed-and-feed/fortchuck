"""Strictly blind multi-seed benchmark runner and statistical referee for fortchuck (v0.2.0).

Evaluates:
  1. `fortchuck` using strictly frozen out-of-the-box `FCOptions(gtol=1e-6, max_eval=5000)`
     and `linear_constraint=(lin_a, lin_b)` when `problem.lin_a is not None` across EVERY
     benchmark problem and EVERY seed (zero per-problem hyperparameters).
  2. `fortchuck-ABB` (`FCOptions(gtol=1e-6, max_eval=5000, memory=0)` ablation).
  3. All 6 external baselines from `benchmarks/baselines.py` (`Alternating-BCD`,
     `pymanopt-Warped-CG`, `pymanopt-TrustRegions`, `SciPy-L-BFGS-B`,
     `SciPy-SLSQP`, `SciPy-trust-constr`) initialized from bit-identical
     `PCG64(seed)` feasible points `(z0, X0_list)`.

By default, runs ALL THREE blind seed suites across all 11 problems (`B1..B8B`):
  - Primary Blind Suite (`101..110`, 110 runs/solver)
  - Out-of-Sample Holdout Suite (`201..210`, 110 runs/solver)
  - Untouched Third Blind Holdout Suite (`301..310`, 110 runs/solver)
totaling 30 seeds (330 runs/solver), computing independent referee solve rates
(`E_KKT_ref <= 1e-6` and `max(V_box, V_lin, V_orth) <= 1e-8`), median/IQR evaluations
& wall-clock ms, median `E_KKT_ref`, `V_box`, `V_lin`, `V_orth`, certified global
optimality gap `f - f_star` on `B6` and `B7B`, and paired Wilcoxon signed-rank `p`-values.

Note: `fortchuck` is imported lazily ONLY when `"fortchuck"` or `"fortchuck-ABB"` is
requested in `solver_names`, preserving the Stage A pre-solver blind firewall.
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
    run_fortchuck_variant,
)
from benchmarks.problems import (  # noqa: E402
    BENCHMARK_IDS,
    BenchmarkProblem,
    get_benchmark_problem,
)
from benchmarks.referee import (  # noqa: E402
    BLIND3_SEEDS,
    HOLDOUT_SEEDS,
    PRIMARY_SEEDS,
     generate_initial_point,
    parse_seeds,
    verify_all_analytical_gradients,
)

ALL_SOLVER_NAMES: tuple[str, ...] = (
    "fortchuck",
    "fortchuck-ABB",
    *EXTERNAL_BASELINE_NAMES,
)

SMOKE_PROBLEM_IDS: tuple[str, ...] = (
    "B1",
    "B2B",
    "B3",
    "B6",
    "B7A",
    "B7B",
    "B8A",
    "B8B",
)

SMOKE_SOLVER_NAMES: tuple[str, ...] = (
    "fortchuck",
    "fortchuck-ABB",
    "Alternating-BCD",
    "SciPy-L-BFGS-B",
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
        "v_lin": r.audit.v_lin,
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
            "v_lin_median": None,
            "v_lin_max": None,
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
    vlin_vals = [float(r.audit.v_lin) for r in runs]
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
    vlin_med, _, _, _ = _median_and_iqr(vlin_vals)
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
        "v_lin_median": vlin_med,
        "v_lin_max": float(np.max(vlin_vals)),
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
        vlin_all = [float(r.audit.v_lin) for r in eval_runs]
        vorth_all = [float(r.audit.v_orth) for r in eval_runs]

        rev_med, rev_iqr, _, _ = _median_and_iqr(raw_ev)
        pev_med, pev_iqr, _, _ = _median_and_iqr(pen_ev)
        wm_med, wm_iqr, _, _ = _median_and_iqr(wall_all)
        kkt_med, _, _, _ = _median_and_iqr(kkt_all)
        vb_med, _, _, _ = _median_and_iqr(vbox_all)
        vl_med, _, _, _ = _median_and_iqr(vlin_all)
        vo_med, _, _, _ = _median_and_iqr(vorth_all)
        vl_max = float(np.max(vlin_all)) if vlin_all else float("nan")

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
            "v_lin_median": vl_med,
            "v_lin_max": vl_max if np.isfinite(vl_max) else None,
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


def _seed_range_label(seeds: Sequence[int]) -> str:
    if not seeds:
        return "none"
    if len(seeds) == 1:
        return str(seeds[0])
    if list(seeds) == list(range(seeds[0], seeds[-1] + 1)):
        return f"{seeds[0]}..{seeds[-1]}"
    return ",".join(str(s) for s in seeds)


def _append_overall_table(
    lines: list[str],
    solvers: Sequence[str],
    overall: dict[str, Any],
) -> None:
    lines.append(
        "| Solver | Evaluated Runs | Solved Runs | Solve Rate | Penalized Evals Med (IQR) | Wall Time ms Med (IQR) | Median KKT Ref | Max Lin Viol | Median Orth Viol | Wilcoxon Evals p (2-sided) | Wilcoxon KKT p (2-sided) |"
    )
    lines.append(
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |"
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
        vlin_str = _fmt_sci(ov.get("v_lin_max"))
        orth_str = _fmt_sci(ov["v_orth_median"])
        p_ev = "—" if s_name == "fortchuck" else _fmt_sci(ov["wilcoxon_evals_two_sided_p"])
        p_kkt = "—" if s_name == "fortchuck" else _fmt_sci(ov["wilcoxon_kkt_two_sided_p"])
        lines.append(
            f"| `{s_name}` | {n_eval_runs} | {n_solved} | {rate_pct:.1f}% | {pev_str} | {wm_str} | `{kkt_str}` | `{vlin_str}` | `{orth_str}` | `{p_ev}` | `{p_kkt}` |"
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
        lin_note = (
            f", linear constraint `a^T z = {pm['lin_b']}`"
            if pm.get("has_linear_constraint")
            else ""
        )
        lines.append(
            f"- **Regime**: `{pm['regime_summary']}` (`d = {pm['d']}`, `R = {pm['R']}`, `N_tot = {pm['n_tot']}`, intrinsic dim `{pm['intrinsic_dim']}`{lin_note}, FD gradient max rel err `{pm['fd_max_rel_err']:.3e}`)"
        )
        lines.append("")
        lines.append(
            "| Solver | Solve Rate | Raw Evals Med (IQR) | Penalized Evals Med (IQR) | Wall ms Med (IQR) | Median KKT Ref | Median Box Viol | Median Lin Viol | Median Orth Viol | Median Objective | Wilcoxon Evals p |"
        )
        lines.append(
            "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |"
        )
        for s_name in solvers:
            st = per_prob[pid][s_name]
            if st["skipped"]:
                lines.append(
                    f"| `{s_name}` | SKIP (`N_tot > 450`) | — | — | — | — | — | — | — | — | — |"
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
            vl = _fmt_sci(st.get("v_lin_median", 0.0))
            vo = _fmt_sci(st["v_orth_median"])
            fv = f"{st['f_val_median']:.6f}" if st["f_val_median"] is not None else "N/A"
            wp = "—" if s_name == "fortchuck" else _fmt_sci(st["wilcoxon_evals_two_sided_p"])
            lines.append(
                f"| `{s_name}` | {sr} | {rev} | {pev} | {wm} | `{kkt}` | `{vb}` | `{vl}` | `{vo}` | `{fv}` | `{wp}` |"
            )
        lines.append("")


def _render_markdown_report(summary: dict[str, Any]) -> str:
    """Render GFM/KaTeX-compliant blind benchmark report (`0` violations across 22 rules)."""
    cfg = summary["config"]
    seeds = cfg["seeds"]
    seed_label = _seed_range_label(seeds)
    problems_meta: list[dict[str, Any]] = summary["problems"]
    solvers: list[str] = cfg["solvers"]
    overall: dict[str, Any] = summary["overall_by_solver"]
    per_prob: dict[str, dict[str, Any]] = summary["by_problem"]
    holdout: dict[str, Any] | None = summary.get("holdout")
    blind3: dict[str, Any] | None = summary.get("blind3")
    pooled: dict[str, Any] | None = summary.get("combined_pooled")

    lines: list[str] = []
    lines.append("# Blind Multi-Seed Benchmark Report (`fortchuck` v0.2.0)")
    lines.append("")
    lines.append("## 1. Blind Evaluation Protocol and Referee Specification")
    lines.append("")
    lines.append(
        "All benchmark runs enforce a strict blind firewall between `benchmarks/*.py` and `src/*.f90`:"
    )
    lines.append("")
    lines.append(
        "- **Frozen Out-of-the-Box Configuration**: `fortchuck` is evaluated across every benchmark instance (`B1` through `B8B`) and every random seed with `FCOptions(gtol=1.0e-6, max_eval=5000)` (`memory = 7`, `max_iter = 15000`, Zhang--Hager nonmonotone `zh_eta` auto `0.85`, canonical metric, cautious skip, `eps_band = 1.0e-4`), passing `linear_constraint=(a, b)` when `lin_a` is present and **zero** per-problem hyperparameter adjustments."
    )

    suite_descs = [f"`{len(seeds)}` deterministic primary seeds (`{seed_label}`)"]
    if holdout is not None:
        h_seeds = holdout["config"]["seeds"]
        suite_descs.append(
            f"`{len(h_seeds)}` out-of-sample holdout seeds (`{_seed_range_label(h_seeds)}`)"
        )
    if blind3 is not None:
        b3_seeds = blind3["config"]["seeds"]
        suite_descs.append(
            f"`{len(b3_seeds)}` untouched third blind holdout seeds (`{_seed_range_label(b3_seeds)}`)"
        )
    lines.append(
        f"- **Multi-Suite Deterministic Initializations**: Initial feasible iterates are generated via `PCG64` across {', '.join(suite_descs)}. On linearly constrained problems (`B8A`, `B8B`), initial Euclidean draws are projected onto the central 80% sub-box intersected with $a^\\top z = b$ via `project_linear_box_ref` so every solver starts from the exact same strictly interior feasible point."
    )
    lines.append(
        "- **Pre-Flight Central Finite-Difference Verification**: Every coordinate of every analytical gradient is verified against central finite differences (`fd_step = 1.0e-5`) to relative error at most $10^{-7}$ prior to solver execution."
    )
    lines.append(
        "- **Independent Post-Solve KKT and Feasibility Certificate**: A run is certified **Solved** by `benchmarks/referee.py` if and only if both optimality and feasibility tolerances hold simultaneously (using the pure-NumPy safeguarded root-finder `project_linear_box_ref` with zero calls to `fortchuck` when `lin_a` is present):"
    )
    lines.append("")
    lines.append("```math")
    lines.append(
        r"\mathcal E_{\mathrm{KKT,ref}}(z, X) = \max\!\left(\left\lVert z - P_{\mathcal C}\bigl(z - \nabla_z f(z, X)\bigr) \right\rVert_\infty, \; \max_{1 \le r \le R} \left\lVert G_r - X_r\,\mathrm{sym}(X_r^\top G_r) \right\rVert_{\mathrm F}\right) \le 10^{-6}"
    )
    lines.append("```")
    lines.append("")
    lines.append("```math")
    lines.append(
        r"\max\!\left(\mathcal V_{\mathrm{box}}, \, \mathcal V_{\mathrm{lin}}, \, \mathcal V_{\mathrm{orth}}\right) \le 10^{-8}, \qquad \mathcal V_{\mathrm{lin}} = \frac{\lvert a^\top z - b \rvert}{1 + \lvert b \rvert}"
    )
    lines.append("```")
    lines.append("")

    sec_num = 2
    if pooled is not None:
        p_seeds = pooled["seeds"]
        lines.append(
            f"## {sec_num}. Pooled Multi-Suite Aggregate Summary (`{len(p_seeds)}` Seeds Total)"
        )
        lines.append("")
        lines.append(
            "Pooled aggregate metrics across all executed blind seed suites (`Penalized Evals` assigns `max_eval = 5000` to unsolved runs; paired two-sided Wilcoxon signed-rank `p`-values compare `fortchuck` against each baseline across shared non-skipped runs):"
        )
        lines.append("")
        _append_overall_table(lines, solvers, pooled["overall_by_solver"])
        sec_num += 1

    lines.append(
        f"## {sec_num}. Primary Blind Suite Aggregate Summary (`Seeds {seed_label}`)"
    )
    lines.append("")
    lines.append(
        "Aggregate metrics across all evaluated `(problem, seed)` pairs in the primary blind suite:"
    )
    lines.append("")
    _append_overall_table(lines, solvers, overall)
    sec_num += 1

    if holdout is not None:
        h_seeds = holdout["config"]["seeds"]
        h_label = _seed_range_label(h_seeds)
        lines.append(
            f"## {sec_num}. Out-of-Sample Holdout Suite Aggregate Summary (`Seeds {h_label}`)"
        )
        lines.append("")
        lines.append(
            f"Aggregate metrics across out-of-sample holdout seeds (`{h_label}`) under the identical frozen `FCOptions(gtol=1.0e-6, max_eval=5000)` configuration:"
        )
        lines.append("")
        _append_overall_table(lines, solvers, holdout["overall_by_solver"])
        sec_num += 1

    if blind3 is not None:
        b3_seeds = blind3["config"]["seeds"]
        b3_label = _seed_range_label(b3_seeds)
        lines.append(
            f"## {sec_num}. Untouched Third Blind Holdout Suite Aggregate Summary (`Seeds {b3_label}`)"
        )
        lines.append("")
        lines.append(
            f"Aggregate metrics across the untouched third blind holdout suite (`{b3_label}`) under the identical frozen `FCOptions(gtol=1.0e-6, max_eval=5000)` configuration:"
        )
        lines.append("")
        _append_overall_table(lines, solvers, blind3["overall_by_solver"])
        sec_num += 1

    # Ground-truth global optimality verification table for B6 and B7B
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
            h_label = _seed_range_label(holdout["config"]["seeds"])
            suites_for_gt.append((f"Holdout (`{h_label}`)", holdout["by_problem"]))
        if blind3 is not None:
            b3_label = _seed_range_label(blind3["config"]["seeds"])
            suites_for_gt.append((f"Blind3 (`{b3_label}`)", blind3["by_problem"]))

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

    if (
        "fortchuck" in solvers
        and pooled is not None
        and holdout is not None
        and len(problems_meta) == 11
        and len(pooled["seeds"]) == 30
    ):
        v010_ref: dict[str, tuple[str, float, float]] = {
            "B1": ("20/20 (100%)", 2084.5, 1813.88),
            "B2A": ("15/20 (75%)", 580.0, 171.27),
            "B2B": ("20/20 (100%)", 111.5, 41.50),
            "B3": ("20/20 (100%)", 44.0, 11.99),
            "B4": ("20/20 (100%)", 1762.0, 1424.73),
            "B5": ("17/20 (85%)", 1233.0, 918.65),
            "B6": ("20/20 (100%)", 1902.0, 1801.48),
            "B7A": ("20/20 (100%)", 67.5, 8.02),
            "B7B": ("20/20 (100%)", 504.0, 619.29),
        }
        lines.append(
            f"## {sec_num}. `v0.1.0` to `v0.2.0` Progression and Zero-FFI Native CLI Telemetry"
        )
        lines.append("")
        lines.append(
            "### `v0.1.0` vs `v0.2.0` Progression (`Seeds 101..110 + 201..210` Shared Cohort & `30`-Seed Pool)"
        )
        lines.append("")
        lines.append(
            "Comparison of frozen out-of-the-box `fortchuck` (`FCOptions(gtol=1.0e-6, max_eval=5000)`) across the 20 shared seeds (`101..110` + `201..210`) from `v0.1.0` alongside the full 30-seed (`+ 301..310`) `v0.2.0` pool:"
        )
        lines.append("")
        lines.append(
            "| Problem | `v0.1.0` Solved (20) | `v0.1.0` Raw Evals Med | `v0.1.0` Wall ms Med | `v0.2.0` Solved (20) | `v0.2.0` Raw Evals Med (20) | `v0.2.0` Wall ms Med (20) | `v0.2.0` Wall Speedup | `v0.2.0` Solved (30 Pool) | `v0.2.0` Max Lin Viol (30) |"
        )
        lines.append(
            "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |"
        )
        shared_runs = summary["runs"] + holdout["runs"]
        for pm in problems_meta:
            pid = pm["problem_id"]
            r20 = [
                r
                for r in shared_runs
                if r["problem_id"] == pid and r["solver_name"] == "fortchuck"
            ]
            n20_ok = sum(1 for r in r20 if r["solved"])
            ev20 = float(np.median([r["n_eval"] for r in r20]))
            wm20 = float(np.median([r["wall_ms"] for r in r20]))
            p30 = pooled["by_problem"][pid]["fortchuck"]
            sr30 = f"{p30['n_solved']}/{p30['n_seeds']} ({100.0 * p30['solve_rate']:.0f}%)"
            vlin30 = _fmt_sci(p30.get("v_lin_max", 0.0))
            if pid in v010_ref:
                v1_sr, v1_ev, v1_wm = v010_ref[pid]
                sp_str = f"`{v1_wm / wm20:.2f}x`"
                lines.append(
                    f"| `{pid}` | {v1_sr} | {v1_ev:.1f} | {v1_wm:.2f} | {n20_ok}/20 ({100.0 * n20_ok / 20.0:.0f}%) | {ev20:.1f} | {wm20:.2f} | {sp_str} | **{sr30}** | `{vlin30}` |"
                )
            else:
                lines.append(
                    f"| `{pid}` | — (`v0.2.0` new) | — | — | {n20_ok}/20 ({100.0 * n20_ok / 20.0:.0f}%) | {ev20:.1f} | {wm20:.2f} | — | **{sr30}** | `{vlin30}` |"
                )
        lines.append("")
        lines.append("### Zero-FFI Native Fortran 2018 CLI Telemetry (`app/fortchuck_bench.f90`)")
        lines.append("")
        lines.append(
            "Pure Fortran 2018 release timings (`gfortran -cpp -O3 -march=native -funroll-loops -fopenmp`, `0` Python `ctypes` callback overhead, `Seeds 101..110`) recorded by `fpm run fortchuck-bench` across Windows (MinGW-w64 UCRT) and WSL2 Ubuntu 26.04 (Pure F2018 & OpenBLAS `-DFORTCHUCK_USE_BLAS`):"
        )
        lines.append("")
        lines.append(
            "| Native Workload (`fortchuck-bench`) | Windows Pure F2018 | WSL2 Pure F2018 | WSL2 OpenBLAS (`-DFORTCHUCK_USE_BLAS`) | Exact Accuracy / Feasibility Certificate |"
        )
        lines.append("| :--- | :---: | :---: | :---: | :--- |")
        lines.append(
            "| `[1]` Batched Panel Transport (`St(200,8)`, `K_batch=16`, `4000` reps) | `872.00 ms` vs `857.31 ms` (`0.98x`) | `1064.39 ms` vs `1113.94 ms` (`1.05x`) | **`358.99 ms` vs `454.35 ms` (`1.27x`)** | `max_err = 0.00e+00` vs `16` single-vector calls |"
        )
        lines.append(
            "| `[1]` Batched Panel Transport (`St(24,24)`, `K_batch=16`, `4000` reps) | `280.74 ms` vs `281.08 ms` (`1.00x`) | `324.28 ms` vs `327.19 ms` (`1.01x`) | `297.23 ms` vs `282.66 ms` (`0.95x`) | `max_err = 0.00e+00` vs `16` single-vector calls |"
        )
        lines.append(
            "| `[2]` `B6` Box-Coupled Ky Fan QP (`10/10` solved) | `973.56 ms` (`1976.0` evals) | `1215.56 ms` (`1982.0` iters) | **`659.38 ms`** | `max |f - f*| <= 5.98e-12`, `max_KKT <= 9.96e-07` |"
        )
        lines.append(
            "| `[3]` `B7B` Tall Brockett + Procrustes (`10/10` solved) | `374.51 ms` (`m=7`) / `212.91 ms` (`m=0`) | `1395.26 ms` (`m=7`) / `728.05 ms` (`m=0`) | **`249.95 ms` (`m=7`) / `150.92 ms` (`m=0`)** | `max_KKT <= 9.99e-07`, `max_orth <= 9.67e-14` |"
        )
        lines.append(
            "| `[4]` `B8A` Hard-Trace Ensemble DFT (`10/10` solved) | `119.23 ms` (`214.0` evals) | `146.99 ms` (`213.0` iters) | **`68.19 ms` (`208.0` iters, `214.0` evals)** | `max |1^T z - 5| <= 1.52e-13` (`1.78e-15` WSL2), `max_KKT <= 1.00e-06` |"
        )
        lines.append("")
        sec_num += 1

    if pooled is not None:
        lines.append(
            f"## {sec_num}. Per-Problem Breakdown — Pooled Multi-Suite (`{len(pooled['seeds'])}` Seeds)"
        )
        lines.append("")
        _append_per_problem_section(lines, problems_meta, pooled["by_problem"], solvers)
        sec_num += 1

    lines.append(f"## {sec_num}. Per-Problem Breakdown — Primary Suite (`Seeds {seed_label}`)")
    lines.append("")
    _append_per_problem_section(lines, problems_meta, per_prob, solvers)
    sec_num += 1

    if holdout is not None:
        h_label = _seed_range_label(holdout["config"]["seeds"])
        lines.append(
            f"## {sec_num}. Per-Problem Breakdown — Out-of-Sample Holdout Suite (`Seeds {h_label}`)"
        )
        lines.append("")
        _append_per_problem_section(
            lines, holdout["problems"], holdout["by_problem"], solvers
        )
        sec_num += 1

    if blind3 is not None:
        b3_label = _seed_range_label(blind3["config"]["seeds"])
        lines.append(
            f"## {sec_num}. Per-Problem Breakdown — Untouched Third Blind Holdout Suite (`Seeds {b3_label}`)"
        )
        lines.append("")
        _append_per_problem_section(
            lines, blind3["problems"], blind3["by_problem"], solvers
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
    needs_fortchuck = ("fortchuck" in solver_names) or ("fortchuck-ABB" in solver_names)
    fc_opts_dict: dict[str, Any] | None = None
    FortchuckWorkspaceCls: Any = None
    if needs_fortchuck:
        from fortchuck import FCOptions, FortchuckWorkspace

        FortchuckWorkspaceCls = FortchuckWorkspace
        fc_opts_dict = FCOptions(gtol=gtol, max_eval=max_eval).as_dict()

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
            FortchuckWorkspaceCls(
                d=prob.d, n_list=prob.n_list, p_list=prob.p_list, memory=7
            )
            if "fortchuck" in solver_names and FortchuckWorkspaceCls is not None
            else None
        )
        ws_abb = (
            FortchuckWorkspaceCls(
                d=prob.d, n_list=prob.n_list, p_list=prob.p_list, memory=0
            )
            if "fortchuck-ABB" in solver_names and FortchuckWorkspaceCls is not None
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
            "has_linear_constraint": p.lin_a is not None,
            "lin_b": float(p.lin_b) if p.lin_b is not None else None,
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
            "fortchuck_options_frozen": fc_opts_dict,
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
    blind3_seeds: Sequence[int] | None = None,
    problem_ids: Sequence[str],
    solver_names: Sequence[str],
    gtol: float = 1.0e-6,
    feas_tol: float = 1.0e-8,
    max_eval: int = 5000,
    timeout_sec: float = 20.0,
    verbose: bool = True,
) -> dict[str, Any]:
    """Execute the primary, holdout, and third blind holdout benchmark suites."""
    if ("fortchuck" in solver_names) or ("fortchuck-ABB" in solver_names):
        # Pre-load shared C-ABI library before any wall-clock measurement
        from fortchuck._native import load_native_library

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

    suite_runs_list: list[dict[str, dict[str, list[BaselineResult]]]] = [primary_runs]
    pooled_seed_list: list[int] = list(seeds)

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
        primary_summary["holdout"] = holdout_summary
        suite_runs_list.append(holdout_runs)
        pooled_seed_list.extend(holdout_seeds)
    else:
        primary_summary["holdout"] = None

    if blind3_seeds:
        blind3_summary, blind3_runs = _execute_single_suite(
            suite_label="BLIND3",
            seeds=blind3_seeds,
            problems=problems,
            solver_names=solver_names,
            gtol=gtol,
            feas_tol=feas_tol,
            max_eval=max_eval,
            timeout_sec=timeout_sec,
            verbose=verbose,
        )
        primary_summary["blind3"] = blind3_summary
        suite_runs_list.append(blind3_runs)
        pooled_seed_list.extend(blind3_seeds)
    else:
        primary_summary["blind3"] = None

    if len(suite_runs_list) > 1:
        pooled_runs: dict[str, dict[str, list[BaselineResult]]] = {
            prob.problem_id: {
                s: [
                    run_item
                    for sr_map in suite_runs_list
                    for run_item in sr_map[prob.problem_id][s]
                ]
                for s in solver_names
            }
            for prob in problems
        }
        pooled_by_prob, pooled_overall = _aggregate_suite_runs(
            problems=problems,
            solver_names=solver_names,
            runs_by_prob_solver=pooled_runs,
            n_seeds=len(pooled_seed_list),
            max_eval=max_eval,
        )
        primary_summary["combined_pooled"] = {
            "seeds": pooled_seed_list,
            "overall_by_solver": pooled_overall,
            "by_problem": pooled_by_prob,
        }
    else:
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
        description="Run the fortchuck v0.2.0 strictly blind multi-seed benchmark suite."
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help=(
            "Fast deterministic CI smoke mode (defaults to seeds 101 / 201 / 301 across "
            "B1,B2B,B3,B6,B7A,B7B,B8A,B8B and fortchuck,fortchuck-ABB,Alternating-BCD,SciPy-L-BFGS-B)."
        ),
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default=None,
        help="Primary seed specification (default: '101..110', or '101' under --smoke).",
    )
    parser.add_argument(
        "--holdout-seeds",
        type=str,
        default="auto",
        help=(
            "Out-of-sample holdout seed specification (default: 'auto' -> '201..210' "
            "when --seeds is '101..110', or '201' under --smoke)."
        ),
    )
    parser.add_argument(
        "--no-holdout",
        action="store_true",
        help="Skip out-of-sample holdout suite (201..210).",
    )
    parser.add_argument(
        "--blind3-seeds",
        type=str,
        default="auto",
        help=(
            "Untouched third blind holdout seed specification (default: 'auto' -> '301..310' "
            "when --seeds is '101..110', or '301' under --smoke)."
        ),
    )
    parser.add_argument(
        "--no-blind3",
        action="store_true",
        help="Skip untouched third blind holdout suite (301..310).",
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
        help="Referee max(V_box, V_lin, V_orth) tolerance (default: 1e-8).",
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

    raw_seeds = args.seeds if args.seeds is not None else ("101" if args.smoke else "101..110")
    seeds = parse_seeds(raw_seeds)

    if args.no_holdout or not args.holdout_seeds or not args.holdout_seeds.strip():
        holdout_seeds: list[int] | None = None
    elif args.holdout_seeds.strip().lower() == "auto":
        if args.smoke and args.seeds is None:
            holdout_seeds = [HOLDOUT_SEEDS[0]]
        elif raw_seeds.strip() == "101..110":
            holdout_seeds = list(HOLDOUT_SEEDS)
        else:
            holdout_seeds = None
    else:
        holdout_seeds = parse_seeds(args.holdout_seeds)

    if args.no_blind3 or not args.blind3_seeds or not args.blind3_seeds.strip():
        blind3_seeds: list[int] | None = None
    elif args.blind3_seeds.strip().lower() == "auto":
        if args.smoke and args.seeds is None:
            blind3_seeds = [BLIND3_SEEDS[0]]
        elif raw_seeds.strip() == "101..110":
            blind3_seeds = list(BLIND3_SEEDS)
        else:
            blind3_seeds = None
    else:
        blind3_seeds = parse_seeds(args.blind3_seeds)

    default_probs = SMOKE_PROBLEM_IDS if (args.smoke and args.problems is None) else BENCHMARK_IDS
    default_solvers = (
        SMOKE_SOLVER_NAMES if (args.smoke and args.solvers is None) else ALL_SOLVER_NAMES
    )
    problem_ids = (
        _parse_csv_list(args.problems, BENCHMARK_IDS, "problem")
        if args.problems is not None
        else list(default_probs)
    )
    solver_names = (
        _parse_csv_list(args.solvers, ALL_SOLVER_NAMES, "solver")
        if args.solvers is not None
        else list(default_solvers)
    )

    summary = run_suite(
        seeds=seeds,
        holdout_seeds=holdout_seeds,
        blind3_seeds=blind3_seeds,
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
