"""Cross-platform shared library builder for fortchuck (`fortchuck.dll` / `libfortchuck.so`).

Compiles all Modern Fortran 2018 sources in strict dependency order inside an
isolated temporary directory and links the C-ABI shared library into
`python/fortchuck/`. Supports optional external BLAS dispatch (`--blas`).
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import platform
import shlex
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


def shared_library_filename(system: str | None = None) -> str:
    sys_name = system or platform.system()
    if sys_name == "Windows":
        return "fortchuck.dll"
    if sys_name == "Darwin":
        return "libfortchuck.dylib"
    return "libfortchuck.so"


def _resolve_blas_link_flags(gfortran_exe: str, blas_libs_arg: str | None) -> list[str]:
    if blas_libs_arg:
        return shlex.split(blas_libs_arg)
    env_flags = os.environ.get("FORTCHUCK_BLAS_LIBS", "").strip()
    if env_flags:
        return shlex.split(env_flags)

    flags: list[str] = []
    toolchain_lib = Path(gfortran_exe).resolve().parent.parent / "lib"
    if toolchain_lib.is_dir():
        has_openblas = any(toolchain_lib.glob("libopenblas*"))
        if has_openblas:
            flags.append(f"-L{toolchain_lib}")
            if platform.system() != "Windows":
                flags.append(f"-Wl,-rpath,{toolchain_lib}")
    flags.append("-lopenblas")
    return flags


def build_shared_library(
    *,
    compiler: str = "gfortran",
    use_blas: bool = False,
    blas_libs: str | None = None,
    march_native: bool = True,
    output_dir: Path | None = None,
    quiet: bool = False,
) -> Path:
    _augment_windows_toolchain_path()

    repo_root = Path(__file__).resolve().parent.parent
    out_dir = (output_dir or (repo_root / "python" / "fortchuck")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    gfortran_exe = shutil.which(compiler)
    if not gfortran_exe:
        raise RuntimeError(f"Fortran compiler '{compiler}' not found on PATH.")

    sys_name = platform.system()
    out_path = out_dir / shared_library_filename(sys_name)

    compile_flags = [
        "-cpp",
        "-std=f2018",
        "-O3",
    ]
    if march_native:
        compile_flags.append("-march=native")
    compile_flags.extend(
        [
            "-funroll-loops",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-pedantic",
            "-Warray-temporaries",
            "-fimplicit-none",
            "-fPIC",
            "-fopenmp",
        ]
    )
    if use_blas:
        compile_flags.append("-DFORTCHUCK_USE_BLAS")

    link_flags = ["-shared", "-fPIC", "-O3", "-funroll-loops"]
    if march_native:
        link_flags.append("-march=native")
    link_flags.append("-fopenmp")
    if sys_name == "Windows":
        link_flags.extend(["-static-libgcc", "-static-libgfortran"])

    extra_libs = _resolve_blas_link_flags(gfortran_exe, blas_libs) if use_blas else []

    if not quiet:
        print(f"Building {out_path.name} ({sys_name}) via {gfortran_exe}")
        print(f"  Compile flags : {' '.join(compile_flags)}")
        print(f"  BLAS enabled  : {use_blas} {extra_libs if use_blas else ''}")

    with tempfile.TemporaryDirectory(prefix="fortchuck_build_") as tmp_str:
        tmpdir = Path(tmp_str)
        obj_paths: list[Path] = []

        for rel_src in ORDERED_SOURCES:
            src_path = repo_root / rel_src
            if not src_path.is_file():
                raise FileNotFoundError(f"Missing Fortran source file: {src_path}")
            obj_path = tmpdir / (src_path.stem + ".o")
            cmd = [
                gfortran_exe,
                *compile_flags,
                f"-J{tmpdir}",
                f"-I{tmpdir}",
                "-c",
                str(src_path),
                "-o",
                str(obj_path),
            ]
            proc = subprocess.run(cmd, capture_output=True, text=True)
            if proc.returncode != 0:
                msg = f"Failed compiling {rel_src}:\n{proc.stdout}\n{proc.stderr}"
                raise RuntimeError(msg)
            obj_paths.append(obj_path)

        staged_lib = tmpdir / out_path.name
        link_cmd = [
            gfortran_exe,
            *link_flags,
            *(str(p) for p in obj_paths),
            *extra_libs,
            "-o",
            str(staged_lib),
        ]
        link_proc = subprocess.run(link_cmd, capture_output=True, text=True)
        if link_proc.returncode != 0:
            msg = f"Failed linking {out_path.name}:\n{link_proc.stdout}\n{link_proc.stderr}"
            raise RuntimeError(msg)

        shutil.copy2(staged_lib, out_path)

    if not quiet:
        size_kb = out_path.stat().st_size / 1024.0
        print(f"SUCCESS: Built {out_path} ({size_kb:.1f} KiB)")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compiler", default="gfortran", help="Fortran compiler executable (default: gfortran)")
    parser.add_argument(
        "--blas",
        action="store_true",
        help="Compile with -DFORTCHUCK_USE_BLAS and link external BLAS (OpenBLAS)",
    )
    parser.add_argument(
        "--blas-libs",
        default=None,
        help="Explicit BLAS linker flags overriding auto-detection (e.g. '-L/usr/lib -lopenblas')",
    )
    parser.add_argument(
        "--no-march-native",
        action="store_true",
        help="Omit -march=native for portable binary distribution",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for shared library (default: python/fortchuck)",
    )
    args = parser.parse_args()
    try:
        build_shared_library(
            compiler=args.compiler,
            use_blas=args.blas,
            blas_libs=args.blas_libs,
            march_native=not args.no_march_native,
            output_dir=args.output_dir,
        )
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
