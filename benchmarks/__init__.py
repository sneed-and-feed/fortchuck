"""Blind benchmark suite, independent external referee, and baseline wrappers for fortchuck."""

from __future__ import annotations

from typing import Any

__all__ = [
    "BENCHMARK_IDS",
    "BenchmarkProblem",
    "DEFAULT_SEEDS",
    "HOLDOUT_SEEDS",
    "EvalOracle",
    "RefereeAudit",
    "evaluate_kkt_and_feasibility",
    "generate_initial_point",
    "get_all_benchmark_problems",
    "get_benchmark_problem",
    "parse_seeds",
    "verify_all_analytical_gradients",
    "verify_problem_gradient",
]


def __getattr__(name: str) -> Any:
    if name in {
        "BENCHMARK_IDS",
        "BenchmarkProblem",
        "get_all_benchmark_problems",
        "get_benchmark_problem",
    }:
        from benchmarks import problems

        return getattr(problems, name)
    if name in {
        "DEFAULT_SEEDS",
        "HOLDOUT_SEEDS",
        "EvalOracle",
        "RefereeAudit",
        "evaluate_kkt_and_feasibility",
        "generate_initial_point",
        "parse_seeds",
        "verify_all_analytical_gradients",
        "verify_problem_gradient",
    }:
        from benchmarks import referee

        return getattr(referee, name)
    raise AttributeError(f"module 'benchmarks' has no attribute {name!r}")
