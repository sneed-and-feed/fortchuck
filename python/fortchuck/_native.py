"""Low-level `ctypes` C99 ABI bindings and shared library loader for `fortchuck`.

Matches `include/fortchuck.h` and `src/fortchuck_kinds.f90` byte-for-byte,
verifying struct sizes (`sizeof(FCOptions) == 96`, `sizeof(FCResult) == 88`)
and field offsets at import time.
"""

from __future__ import annotations

import ctypes
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any

# =============================================================================
# Status, Metric, and Damping Enumerations (matching include/fortchuck.h)
# =============================================================================

FC_SUCCESS_QUERY: int = 0
FC_CONVERGED_GRAD: int = 1
FC_CONVERGED_FTOL: int = 2
FC_CONVERGED_XTOL: int = 3
FC_MAX_ITER_REACHED: int = 4
FC_MAX_EVAL_REACHED: int = 5
FC_ERR_INVALID_DIM: int = -1
FC_ERR_NULL_PTR: int = -2
FC_ERR_INVALID_BOUNDS: int = -3
FC_ERR_INVALID_X0: int = -4
FC_ERR_WORK_TOO_SMALL: int = -5
FC_ERR_CALLBACK_NAN: int = -6
FC_ERR_CALLBACK_ABORT: int = -7
FC_ERR_LS_FAILED: int = -8
FC_ERR_INVALID_OPTS: int = -9

FC_METRIC_CANONICAL: int = 0
FC_METRIC_EUCLIDEAN: int = 1

FC_DAMP_CAUTIOUS: int = 0
FC_DAMP_POWELL: int = 1

STATUS_NAMES: dict[int, str] = {
    FC_SUCCESS_QUERY: "FC_SUCCESS_QUERY",
    FC_CONVERGED_GRAD: "FC_CONVERGED_GRAD",
    FC_CONVERGED_FTOL: "FC_CONVERGED_FTOL",
    FC_CONVERGED_XTOL: "FC_CONVERGED_XTOL",
    FC_MAX_ITER_REACHED: "FC_MAX_ITER_REACHED",
    FC_MAX_EVAL_REACHED: "FC_MAX_EVAL_REACHED",
    FC_ERR_INVALID_DIM: "FC_ERR_INVALID_DIM",
    FC_ERR_NULL_PTR: "FC_ERR_NULL_PTR",
    FC_ERR_INVALID_BOUNDS: "FC_ERR_INVALID_BOUNDS",
    FC_ERR_INVALID_X0: "FC_ERR_INVALID_X0",
    FC_ERR_WORK_TOO_SMALL: "FC_ERR_WORK_TOO_SMALL",
    FC_ERR_CALLBACK_NAN: "FC_ERR_CALLBACK_NAN",
    FC_ERR_CALLBACK_ABORT: "FC_ERR_CALLBACK_ABORT",
    FC_ERR_LS_FAILED: "FC_ERR_LS_FAILED",
    FC_ERR_INVALID_OPTS: "FC_ERR_INVALID_OPTS",
}

# =============================================================================
# C-ABI Struct Definitions & Compile/Import-Time Binary Layout Verification
# =============================================================================

_DEFAULT_OPTIONS_VALUES: dict[str, float | int] = {
    "gtol": 1.0e-6,
    "ftol": 1.0e-14,
    "xtol": 1.0e-14,
    "c1_armijo": 1.0e-4,
    "zh_eta": -1.0,
    "eps_band": 1.0e-4,
    "orth_tol": 1.0e-13,
    "max_step_vel": 2.0 * math.pi,
    "max_iter": 15000,
    "max_eval": 15000,
    "memory": 7,
    "metric": FC_METRIC_CANONICAL,
    "damping_mode": FC_DAMP_CAUTIOUS,
    "project_initial": 1,
    "max_ls_steps": 40,
    "reserved_flags": 0,
}


class FCOptions(ctypes.Structure):
    """96-byte naturally aligned options struct (`fc_options_t` in `fortchuck.h`).

    Instantiating `FCOptions(**overrides)` populates all 16 fields with exact
    canonical defaults matching `fortchuck_default_options` before applying any
    explicit overrides.
    """

    _fields_ = [
        ("gtol", ctypes.c_double),
        ("ftol", ctypes.c_double),
        ("xtol", ctypes.c_double),
        ("c1_armijo", ctypes.c_double),
        ("zh_eta", ctypes.c_double),
        ("eps_band", ctypes.c_double),
        ("orth_tol", ctypes.c_double),
        ("max_step_vel", ctypes.c_double),
        ("max_iter", ctypes.c_int32),
        ("max_eval", ctypes.c_int32),
        ("memory", ctypes.c_int32),
        ("metric", ctypes.c_int32),
        ("damping_mode", ctypes.c_int32),
        ("project_initial", ctypes.c_int32),
        ("max_ls_steps", ctypes.c_int32),
        ("reserved_flags", ctypes.c_int32),
    ]

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__()
        for name, default_val in _DEFAULT_OPTIONS_VALUES.items():
            setattr(self, name, default_val)

        field_names = [f[0] for f in self._fields_]
        if len(args) > len(field_names):
            raise TypeError(
                f"FCOptions() takes at most {len(field_names)} positional arguments ({len(args)} given)"
            )
        for name, val in zip(field_names, args):
            if name in kwargs:
                raise TypeError(f"FCOptions() got multiple values for argument '{name}'")
            setattr(self, name, val)

        for name, val in kwargs.items():
            if name not in _DEFAULT_OPTIONS_VALUES:
                raise TypeError(f"FCOptions() got an unexpected keyword argument '{name}'")
            setattr(self, name, val)

    def as_dict(self) -> dict[str, float | int]:
        return {name: getattr(self, name) for name, _ in self._fields_}


class FCResult(ctypes.Structure):
    """88-byte naturally aligned result struct (`fc_result_t` in `fortchuck.h`)."""

    _fields_ = [
        ("f_val", ctypes.c_double),
        ("kkt_residual", ctypes.c_double),
        ("box_residual", ctypes.c_double),
        ("stiefel_residual", ctypes.c_double),
        ("orth_error", ctypes.c_double),
        ("last_step_size", ctypes.c_double),
        ("lwork_min", ctypes.c_int64),
        ("liwork_min", ctypes.c_int64),
        ("status", ctypes.c_int32),
        ("n_iter", ctypes.c_int32),
        ("n_eval", ctypes.c_int32),
        ("n_reorth", ctypes.c_int32),
        ("n_fallbacks", ctypes.c_int32),
        ("n_active_box", ctypes.c_int32),
    ]

    @property
    def status_name(self) -> str:
        return STATUS_NAMES.get(int(self.status), f"UNKNOWN_STATUS({int(self.status)})")

    @property
    def converged(self) -> bool:
        return int(self.status) in (FC_CONVERGED_GRAD, FC_CONVERGED_FTOL, FC_CONVERGED_XTOL)

    def as_dict(self) -> dict[str, float | int]:
        return {name: getattr(self, name) for name, _ in self._fields_}


def _verify_struct_abi_layout() -> None:
    if ctypes.sizeof(FCOptions) != 96:
        raise AssertionError(f"ABI mismatch: sizeof(FCOptions) == {ctypes.sizeof(FCOptions)} != 96")
    if ctypes.sizeof(FCResult) != 88:
        raise AssertionError(f"ABI mismatch: sizeof(FCResult) == {ctypes.sizeof(FCResult)} != 88")

    expected_opts_offsets = {
        "gtol": 0,
        "ftol": 8,
        "xtol": 16,
        "c1_armijo": 24,
        "zh_eta": 32,
        "eps_band": 40,
        "orth_tol": 48,
        "max_step_vel": 56,
        "max_iter": 64,
        "max_eval": 68,
        "memory": 72,
        "metric": 76,
        "damping_mode": 80,
        "project_initial": 84,
        "max_ls_steps": 88,
        "reserved_flags": 92,
    }
    for field_name, exp_off in expected_opts_offsets.items():
        actual_off = getattr(FCOptions, field_name).offset
        if actual_off != exp_off:
            raise AssertionError(
                f"ABI offset mismatch on FCOptions.{field_name}: {actual_off} != {exp_off}"
            )

    expected_res_offsets = {
        "f_val": 0,
        "kkt_residual": 8,
        "box_residual": 16,
        "stiefel_residual": 24,
        "orth_error": 32,
        "last_step_size": 40,
        "lwork_min": 48,
        "liwork_min": 56,
        "status": 64,
        "n_iter": 68,
        "n_eval": 72,
        "n_reorth": 76,
        "n_fallbacks": 80,
        "n_active_box": 84,
    }
    for field_name, exp_off in expected_res_offsets.items():
        actual_off = getattr(FCResult, field_name).offset
        if actual_off != exp_off:
            raise AssertionError(
                f"ABI offset mismatch on FCResult.{field_name}: {actual_off} != {exp_off}"
            )


_verify_struct_abi_layout()

# C callback type matching `fc_eval_fg_cb` in `include/fortchuck.h`
FCEvalCallback = ctypes.CFUNCTYPE(
    ctypes.c_int32,
    ctypes.c_int32,
    ctypes.c_int32,
    ctypes.POINTER(ctypes.c_int32),
    ctypes.POINTER(ctypes.c_int32),
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double),
    ctypes.c_void_p,
)

# =============================================================================
# Windows DLL Directory Registration & Shared Library Loader
# =============================================================================

_LIB: ctypes.CDLL | None = None
_DLL_DIR_HANDLES: list[Any] = []
_DLL_DIRS_REGISTERED: bool = False


def _register_windows_dll_directories() -> None:
    global _DLL_DIRS_REGISTERED
    if _DLL_DIRS_REGISTERED or os.name != "nt" or not hasattr(os, "add_dll_directory"):
        return

    candidate_dirs: list[str] = []
    candidate_dirs.extend(os.environ.get("PATH", "").split(os.pathsep))

    try:
        import winreg

        for hive, subkey in (
            (winreg.HKEY_CURRENT_USER, "Environment"),
            (
                winreg.HKEY_LOCAL_MACHINE,
                r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment",
            ),
        ):
            try:
                with winreg.OpenKey(hive, subkey) as key:
                    val, _ = winreg.QueryValueEx(key, "Path")
                    if isinstance(val, str):
                        candidate_dirs.extend(os.path.expandvars(val).split(os.pathsep))
            except OSError:
                pass
    except ImportError:
        pass

    local_appdata = os.environ.get("LOCALAPPDATA", "")
    if local_appdata:
        links_dir = Path(local_appdata) / "Microsoft" / "WinGet" / "Links"
        candidate_dirs.append(str(links_dir))
        pkgs_dir = Path(local_appdata) / "Microsoft" / "WinGet" / "Packages"
        if pkgs_dir.is_dir():
            for mingw_bin in sorted(pkgs_dir.glob("BrechtSanders.WinLibs*/mingw64/bin")):
                candidate_dirs.insert(0, str(mingw_bin))

    for fallback in (r"C:\msys64\ucrt64\bin", r"C:\msys64\mingw64\bin", r"C:\mingw64\bin"):
        candidate_dirs.append(fallback)

    seen: set[str] = set()
    for raw_dir in candidate_dirs:
        d_str = raw_dir.strip()
        if not d_str or not os.path.isdir(d_str):
            continue
        norm = os.path.normcase(os.path.normpath(d_str))
        if norm in seen:
            continue
        seen.add(norm)
        try:
            handle = os.add_dll_directory(d_str)
            _DLL_DIR_HANDLES.append(handle)
        except OSError:
            pass

    _DLL_DIRS_REGISTERED = True


def _shared_lib_name() -> str:
    sys_name = platform.system()
    if sys_name == "Windows":
        return "fortchuck.dll"
    if sys_name == "Darwin":
        return "libfortchuck.dylib"
    return "libfortchuck.so"


def _find_or_build_shared_library() -> Path:
    env_override = os.environ.get("FORTCHUCK_LIB_PATH", "").strip()
    if env_override:
        override_path = Path(env_override).expanduser().resolve()
        if override_path.is_file():
            return override_path
        raise FileNotFoundError(f"FORTCHUCK_LIB_PATH points to missing file: {override_path}")

    pkg_dir = Path(__file__).resolve().parent
    lib_name = _shared_lib_name()
    local_lib = pkg_dir / lib_name
    if local_lib.is_file():
        return local_lib

    # Auto-build fallback when running inside a source checkout with scripts/build_shared.py
    repo_root = pkg_dir.parent.parent
    build_script = repo_root / "scripts" / "build_shared.py"
    if build_script.is_file():
        proc = subprocess.run(
            [sys.executable, str(build_script), "--output-dir", str(pkg_dir)],
            capture_output=True,
            text=True,
        )
        if proc.returncode == 0 and local_lib.is_file():
            return local_lib
        raise RuntimeError(
            f"Shared library '{lib_name}' was not found and auto-build failed:\n"
            f"{proc.stdout}\n{proc.stderr}"
        )

    raise FileNotFoundError(
        f"Could not locate compiled shared library '{lib_name}' in '{pkg_dir}'. "
        "Run `python scripts/build_shared.py` first."
    )


def _configure_c_signatures(lib: ctypes.CDLL) -> None:
    c_i32 = ctypes.c_int32
    c_i64 = ctypes.c_int64
    c_i32_p = ctypes.POINTER(ctypes.c_int32)
    c_i64_p = ctypes.POINTER(ctypes.c_int64)
    c_f64_p = ctypes.POINTER(ctypes.c_double)

    lib.fortchuck_default_options.argtypes = [ctypes.POINTER(FCOptions)]
    lib.fortchuck_default_options.restype = None

    lib.fortchuck_workspace_query.argtypes = [
        c_i32,
        c_i32,
        c_i32_p,
        c_i32_p,
        c_i32,
        c_i64_p,
        c_i64_p,
    ]
    lib.fortchuck_workspace_query.restype = c_i32

    lib.fortchuck_solve.argtypes = [
        c_i32,
        c_i32,
        c_i32_p,
        c_i32_p,
        c_f64_p,
        c_f64_p,
        c_f64_p,
        c_f64_p,
        FCEvalCallback,
        ctypes.c_void_p,
        ctypes.POINTER(FCOptions),
        c_f64_p,
        c_i64,
        c_i32_p,
        c_i64,
        ctypes.POINTER(FCResult),
    ]
    lib.fortchuck_solve.restype = c_i32

    lib.fortchuck_project_stiefel.argtypes = [
        c_i32,
        c_i32,
        c_f64_p,
        c_f64_p,
        c_i64,
    ]
    lib.fortchuck_project_stiefel.restype = c_i32

    lib.fortchuck_kkt_residual.argtypes = [
        c_i32,
        c_i32,
        c_i32_p,
        c_i32_p,
        c_f64_p,
        c_f64_p,
        c_f64_p,
        c_f64_p,
        c_f64_p,
        c_f64_p,
        c_i32,
        c_f64_p,
        c_i64,
        c_f64_p,
        c_f64_p,
    ]
    lib.fortchuck_kkt_residual.restype = c_i32


def load_native_library() -> ctypes.CDLL:
    """Load and configure the `fortchuck` C-ABI shared library (cached singleton)."""
    global _LIB
    if _LIB is not None:
        return _LIB

    _register_windows_dll_directories()
    lib_path = _find_or_build_shared_library()
    if os.name == "nt" and hasattr(os, "add_dll_directory"):
        try:
            _DLL_DIR_HANDLES.append(os.add_dll_directory(str(lib_path.parent)))
        except OSError:
            pass

    lib = ctypes.CDLL(str(lib_path))
    _configure_c_signatures(lib)
    _LIB = lib
    return lib


def is_native_available() -> bool:
    """Return True if the compiled `fortchuck` native shared library loads cleanly."""
    try:
        load_native_library()
        return True
    except Exception:
        return False
