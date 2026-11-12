"""`fortchuck`: Zero-Allocation Coupled Box x Stiefel Quasi-Newton Optimizer.

Exposes zero-copy C99/Fortran 2018 bindings (`minimize_box_stiefel`,
`FortchuckWorkspace`, `FCOptions`, `FCResult`, `OptimizeResult`,
`project_stiefel`, `kkt_residual`, `workspace_query`, `default_options`) and
all deterministic status, metric, and damping constants.
"""

from __future__ import annotations

from fortchuck._native import (
    FC_CONVERGED_FTOL,
    FC_CONVERGED_GRAD,
    FC_CONVERGED_XTOL,
    FC_DAMP_CAUTIOUS,
    FC_DAMP_POWELL,
    FC_ERR_CALLBACK_ABORT,
    FC_ERR_CALLBACK_NAN,
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
    STATUS_NAMES,
    is_native_available,
    load_native_library,
)
from fortchuck.solver import (
    FortchuckWorkspace,
    OptimizeResult,
    default_options,
    kkt_residual,
    minimize_box_stiefel,
    project_stiefel,
    workspace_query,
)

__version__ = "0.1.0"

__all__ = [
    "__version__",
    # High-level solver & utilities
    "minimize_box_stiefel",
    "FortchuckWorkspace",
    "OptimizeResult",
    "default_options",
    "workspace_query",
    "project_stiefel",
    "kkt_residual",
    # Native C-ABI structures & loader
    "FCOptions",
    "FCResult",
    "FCEvalCallback",
    "load_native_library",
    "is_native_available",
    "STATUS_NAMES",
    # Status codes
    "FC_SUCCESS_QUERY",
    "FC_CONVERGED_GRAD",
    "FC_CONVERGED_FTOL",
    "FC_CONVERGED_XTOL",
    "FC_MAX_ITER_REACHED",
    "FC_MAX_EVAL_REACHED",
    "FC_ERR_INVALID_DIM",
    "FC_ERR_NULL_PTR",
    "FC_ERR_INVALID_BOUNDS",
    "FC_ERR_INVALID_X0",
    "FC_ERR_WORK_TOO_SMALL",
    "FC_ERR_CALLBACK_NAN",
    "FC_ERR_CALLBACK_ABORT",
    "FC_ERR_LS_FAILED",
    "FC_ERR_INVALID_OPTS",
    # Metric & damping enums
    "FC_METRIC_CANONICAL",
    "FC_METRIC_EUCLIDEAN",
    "FC_DAMP_CAUTIOUS",
    "FC_DAMP_POWELL",
]
