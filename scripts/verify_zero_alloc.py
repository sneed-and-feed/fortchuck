"""Zero-allocation binary symbol gate (`nm -u`) for fortchuck Fortran 2018 modules.

Compiles all `src/*.f90` modules in strict dependency order inside an isolated
temporary directory with `-Warray-temporaries -Werror`, builds `libfortchuck.a`,
and inspects `nm -u` across every `.o` object and the static archive to verify
zero references to heap allocators, array-packing temporaries, or Fortran I/O/stop
runtime symbols.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

ORDERED_SOURCES: tuple[str, ...] = (
    "src/fortchuck_kinds.f90",
    "src/fortchuck_linalg.f90",
    "src/fortchuck_box.f90",
    "src/fortchuck_stiefel.f90",
    "src/fortchuck_lbfgs.f90",
    "src/fortchuck_solver.f90",
    "src/fortchuck_c_abi.f90",
    "src/fortchuck.f90",
)

FORBIDDEN_EXACT_C_SYMBOLS: frozenset[str] = frozenset(
    {
        "malloc",
        "free",
        "realloc",
        "calloc",
        "aligned_alloc",
        "posix_memalign",
        "_aligned_malloc",
        "_aligned_free",
    }
)

FORBIDDEN_GFORTRAN_PREFIXES: tuple[str, ...] = (
    "_gfortran_allocate",
    "_gfortran_deallocate",
    "_gfortran_os_allocate",
    "_gfortran_internal_pack",
    "_gfortran_internal_unpack",
    "_gfortran_st_write",
    "_gfortran_st_read",
    "_gfortran_transfer_",
    "_gfortran_stop",
    "_gfortran_error_stop",
    "_gfortran_os_error",
)


def _augment_windows_toolchain_path() -> None:
    if os.name != "nt":
        return

    extra_dirs: list[str] = []
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
                        extra_dirs.extend(os.path.expandvars(val).split(os.pathsep))
            except OSError:
                pass
    except ImportError:
        pass

    local_appdata = os.environ.get("LOCALAPPDATA", "")
    if local_appdata:
        links_dir = Path(local_appdata) / "Microsoft" / "WinGet" / "Links"
        if links_dir.is_dir():
            extra_dirs.append(str(links_dir))
        pkgs_dir = Path(local_appdata) / "Microsoft" / "WinGet" / "Packages"
        if pkgs_dir.is_dir():
            for mingw_bin in sorted(pkgs_dir.glob("BrechtSanders.WinLibs*/mingw64/bin")):
                if mingw_bin.is_dir():
                    extra_dirs.insert(0, str(mingw_bin))

    for fallback in (r"C:\msys64\ucrt64\bin", r"C:\msys64\mingw64\bin", r"C:\mingw64\bin"):
        if os.path.isdir(fallback):
            extra_dirs.append(fallback)

    cur_paths = os.environ.get("PATH", "").split(os.pathsep)
    cur_norm = {os.path.normcase(os.path.normpath(p)) for p in cur_paths if p}
    to_prepend: list[str] = []
    for p in extra_dirs:
        p = p.strip()
        if not p or not os.path.isdir(p):
            continue
        norm = os.path.normcase(os.path.normpath(p))
        if norm not in cur_norm:
            cur_norm.add(norm)
            to_prepend.append(p)

    if to_prepend:
        os.environ["PATH"] = os.pathsep.join(to_prepend + cur_paths)


def _classify_forbidden_symbol(raw_sym: str) -> str | None:
    sym = raw_sym.strip()
    if not sym:
        return None
    # Strip dynamic symbol version suffix (e.g., malloc@GLIBC_2.2.5) and Windows __imp_ prefix
    sym_no_ver = sym.split("@", 1)[0]
    if sym_no_ver.startswith("__imp_"):
        sym_no_ver = sym_no_ver[len("__imp_") :]

    candidates = {sym_no_ver, sym_no_ver.lower()}
    if sym_no_ver.startswith("_") and not sym_no_ver.startswith("_gfortran_") and not sym_no_ver.startswith("_aligned_"):
        candidates.add(sym_no_ver[1:])
        candidates.add(sym_no_ver[1:].lower())

    for cand in candidates:
        if cand in FORBIDDEN_EXACT_C_SYMBOLS:
            return f"C heap allocator symbol '{raw_sym}'"
        for prefix in FORBIDDEN_GFORTRAN_PREFIXES:
            if cand.startswith(prefix) or cand.startswith(prefix[1:]):
                return f"Forbidden GFortran runtime symbol '{raw_sym}' (matches {prefix}*)"

    return None


def _extract_undefined_symbols(nm_exe: str, binary_path: Path) -> list[str]:
    proc = subprocess.run(
        [nm_exe, "-u", str(binary_path)],
        capture_output=True,
        text=True,
        check=True,
    )
    symbols: list[str] = []
    for raw_line in proc.stdout.splitlines():
        line = raw_line.strip()
        if not line or line.endswith(":"):
            continue
        # Match GNU nm undefined format: optional address/spaces + 'U'/'w'/'v' + symbol
        m = re.match(r"^(?:[0-9a-fA-F]+\s+)?[Uwv]\s+(\S+)$", line)
        if m:
            symbols.append(m.group(1))
    return symbols


def verify_zero_alloc(compiler: str = "gfortran", opt_level: str = "-O3") -> int:
    _augment_windows_toolchain_path()

    repo_root = Path(__file__).resolve().parent.parent
    gfortran_exe = shutil.which(compiler)
    nm_exe = shutil.which("nm")
    ar_exe = shutil.which("ar")

    if not gfortran_exe:
        print(f"ERROR: Fortran compiler '{compiler}' not found on PATH.", file=sys.stderr)
        return 1
    if not nm_exe:
        print("ERROR: GNU Binutils 'nm' not found on PATH.", file=sys.stderr)
        return 1
    if not ar_exe:
        print("ERROR: GNU Binutils 'ar' not found on PATH.", file=sys.stderr)
        return 1

    actual_src = {p.relative_to(repo_root).as_posix() for p in (repo_root / "src").glob("*.f90")}
    expected_src = set(ORDERED_SOURCES)
    if actual_src != expected_src:
        missing = sorted(expected_src - actual_src)
        unlisted = sorted(actual_src - expected_src)
        print(
            f"ERROR: Source file mismatch in src/ (missing={missing}, unlisted={unlisted}).",
            file=sys.stderr,
        )
        return 1

    print(f"Using compiler : {gfortran_exe}")
    print(f"Using nm       : {nm_exe}")
    print(f"Using ar       : {ar_exe}")

    with tempfile.TemporaryDirectory(prefix="fortchuck_zero_alloc_") as tmp_str:
        tmpdir = Path(tmp_str)
        obj_paths: list[Path] = []

        base_flags = [
            "-cpp",
            "-std=f2018",
            opt_level,
            "-Wall",
            "-Wextra",
            "-Werror",
            "-pedantic",
            "-Warray-temporaries",
            "-fimplicit-none",
            "-fPIC",
            "-fopenmp",
            f"-J{tmpdir}",
            f"-I{tmpdir}",
        ]

        for rel_src in ORDERED_SOURCES:
            src_path = repo_root / rel_src
            obj_path = tmpdir / (src_path.stem + ".o")
            cmd = [gfortran_exe, *base_flags, "-c", str(src_path), "-o", str(obj_path)]
            proc = subprocess.run(cmd, capture_output=True, text=True)
            if proc.returncode != 0:
                print(f"FAILED compiling {rel_src}:", file=sys.stderr)
                if proc.stdout:
                    print(proc.stdout, file=sys.stderr)
                if proc.stderr:
                    print(proc.stderr, file=sys.stderr)
                return 1
            obj_paths.append(obj_path)

        archive_path = tmpdir / "libfortchuck.a"
        ar_proc = subprocess.run(
            [ar_exe, "rcs", str(archive_path), *(str(p) for p in obj_paths)],
            capture_output=True,
            text=True,
        )
        if ar_proc.returncode != 0:
            print(f"FAILED creating static archive: {ar_proc.stderr}", file=sys.stderr)
            return 1

        violations: list[tuple[str, str]] = []
        for obj_path in obj_paths:
            undef_syms = _extract_undefined_symbols(nm_exe, obj_path)
            ext_non_mod = [
                s for s in undef_syms if "fortchuck_" not in s.lower()
            ]
            print(f"  [OBJ] {obj_path.name:<26} external_undef={ext_non_mod}")
            for sym in undef_syms:
                reason = _classify_forbidden_symbol(sym)
                if reason is not None:
                    violations.append((obj_path.name, reason))

        archive_syms = sorted(set(_extract_undefined_symbols(nm_exe, archive_path)))
        ext_archive_syms = [s for s in archive_syms if "fortchuck_" not in s.lower()]
        print(f"  [LIB] {'libfortchuck.a':<26} external_undef={ext_archive_syms}")
        for sym in archive_syms:
            reason = _classify_forbidden_symbol(sym)
            if reason is not None:
                violations.append(("libfortchuck.a", reason))

        if violations:
            print(f"\nZERO-ALLOCATION AUDIT FAILED ({len(violations)} forbidden symbol references):", file=sys.stderr)
            for target_name, reason in violations:
                print(f"  - {target_name}: {reason}", file=sys.stderr)
            return 1

    print(
        f"\nPASS: All {len(ORDERED_SOURCES)} Fortran 2018 objects and libfortchuck.a compiled cleanly "
        "with -Warray-temporaries -Werror and 0 heap/pack/IO/stop symbols!"
    )
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compiler", default="gfortran", help="Fortran compiler executable (default: gfortran)")
    parser.add_argument("--opt", default="-O3", help="Optimization flag (default: -O3)")
    args = parser.parse_args()
    sys.exit(verify_zero_alloc(compiler=args.compiler, opt_level=args.opt))


if __name__ == "__main__":
    main()
