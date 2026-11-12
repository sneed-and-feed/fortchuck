# Toolchain, Build Verification & Repository Engineering Standards (`fortchuck`)

> **Scope**: Verified build, FFI, benchmarking, formal-verification, and repository engineering standards for [`sneed-and-feed/fortchuck`](https://github.com/sneed-and-feed/fortchuck) (`c:\Users\x\Documents\antigravity\fortchuck`). All compilers, package managers, shared-library ABIs, OpenMP runtimes, BLAS/LAPACK backends, Python benchmarking environments, and Lean 4 toolchains documented below have been provisioned and end-to-end smoke-tested on both **Windows Native (`x86_64-w64-mingw32` UCRT)** and **WSL2 Linux (`Ubuntu 26.04 LTS` `x86_64-linux-gnu`)**.

---

## Table of Contents

- [1. Safe Git Repository Bootstrap Status](#1-safe-git-repository-bootstrap-status)
- [2. Dual-OS Provisioned Toolchain Matrix](#2-dual-os-provisioned-toolchain-matrix)
- [3. Verified Fortran 2018 Compiler, fpm & Shared-Library Recipes](#3-verified-fortran-2018-compiler-fpm--shared-library-recipes)
- [4. Linear Algebra Strategy: Zero-Dependency Core + Optional OpenBLAS](#4-linear-algebra-strategy-zero-dependency-core--optional-openblas)
- [5. Zero-Copy Python ctypes FFI & Blind Benchmark Environment](#5-zero-copy-python-ctypes-ffi--blind-benchmark-environment)
- [6. Ecosystem Engineering, Verification & CI Standards Audit](#6-ecosystem-engineering-verification--ci-standards-audit)

---

## 1. Safe Git Repository Bootstrap Status

- **Local Repository Path**: `c:\Users\x\Documents\antigravity\fortchuck`
- **Remote Origin**: `https://github.com/sneed-and-feed/fortchuck`
- **Tracked Branch**: `main` tracking `origin/main` at initial upstream commit `487f299d916e1457de7c1ed92012cd44e396b83a`
- **Upstream Baseline Files Checked Out**:
  - `LICENSE` (`CC0-1.0 Universal` Public Domain Dedication, 121 lines — matches `triad-core` and sibling repositories)
  - `README.md` (`# fortchuck` stub, 1 line)
- **Push Policy**: Remote pushes remain strictly disabled unless explicitly instructed by the user/coordinator.

---

## 2. Dual-OS Provisioned Toolchain Matrix

Both native Windows PowerShell and WSL2 Ubuntu 26.04 now provide **identical GCC/GFortran 16.2.0 + `fpm` 0.13.0 + Python 3.12.13** toolchains so implementation and verification workers can compile, test, and benchmark seamlessly in either environment (with native Windows ideal for fast PowerShell iteration and WSL2 providing exact Linux ELF `.so` parity with GitHub Actions `ubuntu-latest`).

| Component | Windows Native (`x86_64-ucrt-posix-seh`) | WSL2 Linux (`Ubuntu 26.04`, `x86_64-linux-gnu`) | Verification Status |
| :--- | :--- | :--- | :--- |
| **Fortran Compiler** | `gfortran.exe` `16.2.0` (WinLibs POSIX UCRT r3, on `PATH`) | `gfortran` `16.2.0` (`conda-forge` `fortchuck` env, on login `PATH`) | Verified (`-std=f2018`) |
| **C Compiler** | `gcc.exe` `16.2.0` (WinLibs POSIX UCRT r3, on `PATH`) | `gcc` `16.2.0` (`conda-forge`) & `/usr/bin/gcc` `15.2.0` | Verified |
| **Fortran Package Manager** | `C:\Users\x\.local\bin\fpm.exe` (`0.13.0`, SHA-256 verified) | `/home/sneed/.local/bin/fpm` (`0.13.0`, SHA-256 verified) | Verified (`fpm test`) |
| **OpenMP Runtime** | `libgomp-1.dll` (POSIX threads, OpenMP 4.5/5.x) | `libgomp.so` (`_openmp_mutex 4.5`, OpenMP 4.5/5.x) | Verified (`-fopenmp`) |
| **Binary Symbol Tools** | `nm.exe`, `objdump.exe`, `ar.exe` (`GNU Binutils 2.47`) + `pefile` | `nm`, `objdump`, `readelf`, `ar` (`GNU Binutils 2.46.1`) | Verified (`0` heap allocs) |
| **Optional BLAS / LAPACK** | SciPy OpenBLAS / optional `-lopenblas` | `openblas 0.3.34` (`pthreads`) + `lapack 3.11.0` + `/usr/lib/.../libopenblas.so` (`0.3.32`) | Verified (`dgemm`, `dgesv`) |
| **Python Environment Manager** | `C:\Users\x\.local\bin\uv.exe` (`0.13.4`) | `/home/sneed/.local/bin/uv` | Verified |
| **Benchmarking Python Venv** | `C:\Users\x\.venvs\fortchuck\Scripts\python.exe` (`CPython 3.12.13`) | `/home/sneed/.venvs/fortchuck/bin/python` (`CPython 3.12.13`) | Verified |
| **Python Packages Installed** | `numpy 2.5.4`, `scipy 1.18.1`, `pytest 9.2.0`, `pymanopt 2.2.1`, `pefile 2024.8.26` | `numpy 2.5.4`, `scipy 1.18.1`, `pytest 9.2.0`, `pymanopt 2.2.1` | Verified |
| **Lean 4 Prover & Lake** | `C:\Users\x\.elan\bin\lean.exe` (`v4.30.0` default; `v4.34.0-rc2` with cached Mathlib4) | Accessible via Windows `elan` or standalone Lean 4 | Verified (`0` warnings) |

> [!NOTE]
> **Shell `PATH` Refresh Note on Windows**: Because `WinLibs` was installed during this session, existing parent processes without refreshed User `PATH` can prepend `C:\Users\x\AppData\Local\Microsoft\WinGet\Links` (or the `mingw64\bin` package directory) if `Get-Command gfortran` is ever empty in a fresh subshell. In WSL2, `wsl -e bash -lc "..."` automatically sources `~/.profile`, which prepends `~/.local/share/mamba/envs/fortchuck/bin` and `~/.local/bin`.

---

## 3. Verified Fortran 2018 Compiler, `fpm` & Shared-Library Recipes

### 3.1 Strict Debug / Pedantic Sanitizer Profile (`fpm test`)

All Fortran source files must compile with zero warnings under strict Fortran 2018 conformance (`-Werror`) and zero hidden heap array temporaries (`-Warray-temporaries`):

```bash
# Identical command on Windows PowerShell and WSL2 Linux:
fpm test --compiler gfortran --flag "-std=f2018 -O2 -g -Wall -Wextra -Werror -pedantic -Warray-temporaries -fimplicit-none -fcheck=all -fbacktrace -ffpe-trap=invalid,zero,overflow -fopenmp"
```

### 3.2 High-Performance Release Profile (`fpm build`)

```bash
fpm build --compiler gfortran --flag "-std=f2018 -O3 -march=native -funroll-loops -Wall -Wextra -Werror -pedantic -Warray-temporaries -fopenmp"
```

### 3.3 Shared Library Compilation (`libfortchuck.so` / `fortchuck.dll`)

Because `fpm` builds static archives (`libfortchuck.a`) by default, building the C-ABI shared library for Python `ctypes` and external C/C++/Rust consumers uses the following verified two-step compilation (or linking `--whole-archive` from `fpm`'s object files):

#### WSL2 / Linux (`libfortchuck.so`)

```bash
# Compile position-independent F2018 objects in dependency order (or via fpm --flag "-fPIC ...")
gfortran -std=f2018 -O3 -march=native -Wall -Wextra -Werror -pedantic -Warray-temporaries \
  -fopenmp -fPIC -c src/fortchuck_kinds.f90 src/fortchuck_linalg.f90 ...

# Link shared library
gfortran -shared -fPIC -O3 -march=native -fopenmp *.o -o libfortchuck.so
```

#### Windows Native (`fortchuck.dll`)

```powershell
# Compile F2018 objects
gfortran -std=f2018 -O3 -march=native -Wall -Wextra -Werror -pedantic -Warray-temporaries `
  -fopenmp -c src/fortchuck_kinds.f90 src/fortchuck_linalg.f90 ...

# Link self-contained DLL (statically bundling libgcc/libgfortran; libgomp linked for OpenMP)
gfortran -shared -O3 -march=native -fopenmp -static-libgcc -static-libgfortran *.o -o fortchuck.dll
```

### 3.4 Fortran 2018 `-Werror` & Zero-Allocation Implementation Guardrails

Empirical testing with `gfortran 16.2.0` established four critical rules for Phase 2 authors:

1. **No Automatic Arrays or Expression Temporaries**: Never declare stack automatic arrays whose dimensions depend on runtime arguments (`real(c_double) :: tmp(n)`), and never pass non-contiguous slices to explicit-shape dummy arguments. Above `64 KB` (`-fmax-stack-var-size`), `gfortran` silently lowers automatic arrays to `malloc`/`free`, and non-contiguous slices trigger `_gfortran_internal_pack` (heap allocation). Slice all workspace buffers explicitly from caller-supplied `work(1:lwork)` and `iwork(1:liwork)`.
2. **No I/O or `stop` Inside Library Modules**: Never place `print`, `write`, `stop`, or `error stop` inside `src/*.f90` (keep those strictly in `test/*.f90`). Returning deterministic integer `status` / `info` codes eliminates all `_gfortran_st_write*` and `_gfortran_os_error*` runtime dependencies.
3. **Unused `type(c_ptr)` Dummy Arguments in Callbacks**: Under `-std=f2018 -Wall -Wextra -Werror`, `type(c_ptr)` has `PRIVATE` components (so `print *, udata` is a compile error). When a `bind(C)` test callback ignores `udata`, silence `-Wunused-dummy-argument` cleanly via:
   ```fortran
   if (.false. .and. c_associated(udata)) continue
   ```
4. **Object-Level Zero-Allocation Audit (`nm -u`)**: Even when user code performs zero heap allocations, the C runtime shared-library startup stub (`crtbeginS.o` / `dllcrt1.o`) references CRT heap initialization tables in `.so`/`.dll` headers. Therefore, our automated zero-allocation CI gate **must inspect every compiled library object file (`*.o` / `libfortchuck.a`) via `nm -u`**, asserting **zero** matches against:
   - `malloc`, `free`, `realloc`, `calloc`, `_aligned_malloc`, `_aligned_free`
   - `_gfortran_allocate*`, `_gfortran_deallocate*`, `_gfortran_os_allocate*`
   - `_gfortran_internal_pack*`, `_gfortran_internal_unpack*`

   In our smoke test (`fc_smoke.o` with `iso_c_binding`, `c_f_procpointer`, `ieee_arithmetic`, `pure` $2 \times 2$ solver, workspace query `lwork = -1`, and OpenMP parallel-do directives), `nm -u fc_smoke.o` on **both** Linux and Windows emitted **only** `GOMP_parallel`, `omp_get_num_threads`, and `omp_get_thread_num` (`0` allocator or `libgfortran` symbols).

---

## 4. Linear Algebra Strategy: Zero-Dependency Core + Optional OpenBLAS

To satisfy both **100% zero-system-dependency portability** (`fpm build` out-of-the-box anywhere with only `gfortran`) and **maximum throughput for large Stiefel dimensions**, `fortchuck` should adopt a clean compile-time / link-time dual path:

1. **Default Zero-Dependency Pure Fortran 2018 Linalg (`src/fortchuck_linalg.f90`)**:
   - `pure` unrolled / contiguous-loop `gemm`, `syrk`, Frobenius inner products, and `getrf` + `getrs` (LU with partial pivoting + reciprocal pivot ratio guard $\min_i \lvert U_{ii} \rvert < 10^{-12} \max_i \lvert U_{ii} \rvert$).
   - Sign-preserving thin Householder `geqrf` ($R_{ii} > 0$) + normal-space Newton-Schulz polish.
   - For typical small-to-medium block ranks $p_r \le 32$, pure `-O3 -march=native` Fortran SIMD-vectorized column-major loops operate entirely inside L1/L2 cache without external library call overhead or OpenBLAS internal thread-pool contention inside OpenMP parallel regions!
2. **Optional External BLAS/LAPACK Acceleration (`-DFORTCHUCK_USE_BLAS`)**:
   - When preprocessed with `-cpp -DFORTCHUCK_USE_BLAS` and linked against OpenBLAS (`-L/home/sneed/.local/share/mamba/envs/fortchuck/lib -lopenblas -llapack` in WSL2 or `-lopenblas` in CI), large matrix-matrix products (`dgemm`) and dense factorizations (`dgetrf`/`dgetrs`) above a dimension threshold ($n_r p_r > 4096$) dispatch to level-3 BLAS while keeping `OPENBLAS_NUM_THREADS=1` during multi-start OpenMP sweeps.

---

## 5. Zero-Copy Python `ctypes` FFI & Blind Benchmark Environment

### 5.1 Python Dependency & Virtual Environment Protocol

Per workspace Python dependency rules, **never run global `pip install`**. Pre-provisioned `CPython 3.12.13` virtual environments are ready at:
- **Windows**: `C:\Users\x\.venvs\fortchuck\Scripts\python.exe` (`pytest.exe` in `Scripts\`)
- **WSL2**: `/home/sneed/.venvs/fortchuck/bin/python` (`pytest` in `bin/`)

Once `pyproject.toml` and `.gitignore` are created in `fortchuck` during Phase 2, a project-local `.venv` can also be instantiated in `< 2s` from the warm `uv` cache:

```bash
uv venv .venv --python 3.12
uv pip install --python .venv -e ".[dev,bench]"
```

### 5.2 Verified Zero-Copy Fortran-Column-Major `ctypes` Pattern

- **Array Layout**: Stiefel factors $X_r \in \mathbb{R}^{n_r \times p_r}$ in Fortran are column-major. Python bindings should either accept/maintain `np.asfortranarray(X, dtype=np.float64)` (`order="F"`) for true zero-copy pointer passing (`arr.ctypes.data_as(ctypes.POINTER(ctypes.c_double))`) or transpose cleanly at the boundary while documenting zero-copy `order="F"` fast paths.
- **Windows DLL Search Path**: On Python $\ge 3.8$ for Windows, `ctypes.CDLL` ignores `PATH` when resolving transitive DLLs (`libgomp-1.dll`, `libwinpthread-1.dll`). Always register toolchain directories via `os.add_dll_directory(...)` before calling `ctypes.CDLL(lib_path)`:

```python
if os.name == "nt":
    for p in os.environ.get("PATH", "").split(os.pathsep):
        if p and os.path.isdir(p):
            try:
                os.add_dll_directory(p)
            except OSError:
                pass
```

### 5.3 Verified `pymanopt` (`2.2.1`) & `scipy.optimize` (`1.18.1`) Head-to-Head Harness Recipe

Both `scipy.optimize` (`L-BFGS-B`, `SLSQP`, `trust-constr`) and `pymanopt 2.2.1` (`SteepestDescent`, `ConjugateGradient`, `TrustRegions` on `Stiefel(n, p)` and `Product([...])`) are installed and verified **using pure NumPy cost and Euclidean gradient decorators (`@pymanopt.function.numpy(manifold)`)—requiring zero PyTorch/JAX/Autograd overhead**:

```python
import pymanopt
from pymanopt.manifolds import Stiefel
from pymanopt.optimizers import ConjugateGradient

manifold = Stiefel(n, p)

@pymanopt.function.numpy(manifold)
def cost(X):
    return float(eval_f(X))

@pymanopt.function.numpy(manifold)
def egrad(X):
    return eval_euclidean_grad(X)

problem = pymanopt.Problem(manifold, cost, euclidean_gradient=egrad)
result = ConjugateGradient(verbosity=0, max_iterations=500).run(problem)
```

Smoke-test verification on the Stiefel Rayleigh quotient problem:

```math
\min_{X \in \mathrm{St}(4, 2)} \frac{1}{2}\mathrm{Tr}\!\left(X^\top \mathrm{diag}(4, 3, 2, 1) X\right)
```

confirmed exact convergence to $f^\ast = 1.5000000000$ with orthogonality residual $\lVert X^\top X - I_2 \rVert_{\mathrm{F}} \le 7.02 \times 10^{-16}$ alongside `libfc_smoke` Cayley orthogonality $1.11 \times 10^{-16}$ on both OSes.

---

## 6. Ecosystem Engineering, Verification & CI Standards Audit

Read-only inspection of `c:\Users\x\Documents\antigravity\triad-core` (`README.md`, `.github/workflows/ci.yml`, `pyproject.toml`, `CITATION.cff`, `triad_core/_native.py`, `formalization/TriadCore.lean`, `scripts/verify_bcm2712_arena.py`) and `c:\Users\x\Documents\antigravity\darksnail\scripts\audit_gfm_math.py` establishes the exact repository polish and verification gates expected for `sneed-and-feed/fortchuck`:

### 6.1 Mandatory Repository Metadata & Polish
1. **License & Citation**: Keep `CC0-1.0 Universal` (`LICENSE`), add `CITATION.cff` (`cff-version: 1.2.0`, author `Noah Cohen`, `license: CC0-1.0`, `repository-code: https://github.com/sneed-and-feed/fortchuck`), and configure `pyproject.toml` (`setuptools>=61.0`, `numpy>=1.24`, optional `[dev]` and `[bench]` extras).
2. **README Architecture**: Top shield badges (`CI Matrix`, `Lean 4 0 sorry 0 warnings`, `Fortran 2018 0 heap alloc`, `License CC0-1.0`), linked Table of Contents (plain Unicode in anchor text—never inline math delimiters inside markdown links!), mathematical problem formulation, theorem verification table, architecture/complexity summary, C99/Fortran/Python quickstart snippets, repository tree, synchronous verification commands, and BibTeX citation block.
3. **ANSI C99/C11 Header (`include/fortchuck.h`)**: Clean `extern "C"` guards, `stddef.h`/`stdint.h` fixed-width types matching Fortran `iso_c_binding` (`c_int`, `c_double`, `c_funptr`, `c_ptr`), explicit status enum constants, and workspace query documentation.

### 6.2 Automated GFM / KaTeX Math Compliance Gate (`scripts/audit_gfm_math.py`)
Every Markdown file in `fortchuck` (`README.md`, `docs/*.md`, `docs/research/*.md`) **must pass `python scripts/audit_gfm_math.py <files>` with `0` violations across all 22 invariant rules** and stay well under the `< 800` math-expression rendering budget per document:

```text
Key GFM/KaTeX Linter Rules Enforced by scripts/audit_gfm_math.py:
- Display math: Column-0 ```math blocks surrounded by blank lines (no $$ in prose/tables, no indented math blocks).
- Cross-span subscripts (Rule 9): At most 1 inline math span containing '_' per line.
- Macro restrictions: Use \mathrm{...} (never \operatorname); use \left\lbrace ... \right\rbrace; use \ast (never *); use \lvert ... \rvert in tables.
- Delimiter collisions: Never wrap math in **bold**; never put math inside [link](url) anchors; never place $, $$, or shell $VAR inside inline backticks on prose lines.
- Flanking subscripts (Rule 16/16b): Never follow }, ), or ] with '_' on prose lines (write \mathcal F_k, \mathrm{St}(n, p)).
```

### 6.3 Optional Lean 4 Formalization Gate (`formalization/Fortchuck.lean`)
- **Installed Toolchains**: `lean` `v4.30.0` (default) and `v4.34.0-rc2` (`C:\Users\x\.elan\bin\lean.exe`, `lake.exe`), with a warm Mathlib4 build cache in sibling workspaces (`astrosnail/formalization`, Mathlib rev `de2ef68216c6074f338c8e61890ee0a379ddfb9b`).
- **Recommended Scope for `fortchuck`**: Machine-checked `0 sorry`, `0` warning proofs of the core algebraic invariants underlying `fortchuck`:
  - Skew-symmetry of $W_\xi = \tilde{\xi} X^\top - X \tilde{\xi}^\top \in \mathfrak{so}(n)$.
  - Exact Cayley orthogonality $R(\tau W)^\top R(\tau W) = I_n$ for skew-symmetric $W$.
  - Schur-complement / Woodbury identity for rank $2p$ Cayley retraction and strict directional descent $\phi'(0^+) < 0$ of projected-box plus Riemannian positive-definite directions.
- **Workflow Rule**: Any agent authoring or editing `.lean` files **must** use `lean-lsp` MCP tools (`lean_goal`, `lean_diagnostic_messages`, `lean_multi_attempt`, `lean_loogle`) interactively, reserving `lake build` strictly for final export validation.

### 6.4 Multi-Job GitHub Actions CI Matrix (`.github/workflows/ci.yml`)
Following `triad-core/.github/workflows/ci.yml`, `fortchuck`'s CI workflow on `ubuntu-latest` should enforce four synchronous gates:
1. **`gfm-math-audit`**: Runs `python scripts/audit_gfm_math.py` across `README.md` and all `docs/**/*.md`.
2. **`fortran-fpm-and-zero-alloc`**: Installs `gfortran` + `fpm`, runs strict sanitizer `fpm test` (`-std=f2018 -Wall -Wextra -Werror -fcheck=all -fopenmp`), builds `libfortchuck.so`, and runs `scripts/verify_zero_alloc.py` checking `nm -u` on all `*.o` objects (`0` heap allocator or array-packing symbols) plus OpenMP multi-threaded stress tests.
3. **`python-ffi-parity-and-pytest`**: Installs `numpy`, `scipy`, `pytest`, `pymanopt`, verifies zero-copy `ctypes` C-ABI parity, KKT optimality residuals, manifold orthogonality $\lVert X_r^\top X_r - I_{p_r} \rVert_{\mathrm{F}} \le 10^{-13}$, and deterministic status codes.
4. **`blind-benchmark-smoke`**: Executes a fast deterministic seed check of the blind benchmark suite, emitting verifiable JSON receipts under `results/`.
