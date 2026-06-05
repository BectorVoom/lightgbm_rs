# Technology Stack

> **Scope note:** This document maps the **LightGBM C++ REFERENCE implementation** located under `LightGBM/` (Microsoft's upstream gradient-boosting library, the system being ported). It does **NOT** describe the Rust crate under development in this repo's `src/` + `Cargo.toml`. All paths are relative to `LightGBM/`. Version per `LightGBM/VERSION.txt`: **4.6.0.99**.

**Analysis Date:** 2026-06-05

## Languages

**Primary:**
- C++11 - Core training/inference engine. Entire `src/` tree (boosting, treelearner, io, objective, metric, network, application). Standard fixed in `CMakeLists.txt`: `set(CMAKE_CXX_STANDARD 11)` (raised to 14 only when `BUILD_CPP_TEST=ON`, which pulls Google Test). `CMAKE_CXX_STANDARD_REQUIRED ON`.
- C - Project declares `LANGUAGES C CXX` in `CMakeLists.txt`; the public C API (`include/LightGBM/c_api.h`) is C-callable via `extern "C"` (see `include/LightGBM/export.h`).

**Secondary:**
- CUDA C++ - GPU backend, compiled only when `USE_CUDA=ON`. `.cu` files under `src/**/cuda/` and `src/cuda/`. `CMAKE_CUDA_STANDARD 11` (`CMakeLists.txt`).
- OpenCL C - GPU kernels for the (non-CUDA) `gpu` device type: `src/treelearner/ocl/histogram16.cl`, `histogram64.cl`, `histogram256.cl`.
- Python 3.7+ - Wrapper package `python-package/lightgbm/` (pure-Python ctypes binding, no compiled extension of its own).
- R - Wrapper package `R-package/` with a thin C++ shim `R-package/src/lightgbm_R.cpp`.
- SWIG interface (`.i`) - Java/JNI binding generators in `swig/` (e.g. `swig/lightgbmlib.i`).

## Runtime

**Environment:**
- Native compiled shared/static library (`lib_lightgbm.{so,dll,dylib}`) + standalone CLI executable (`lightgbm`). No managed runtime for the core.
- Python wrapper requires CPython 3.7-3.13 (`python-package/pyproject.toml` classifiers), loads the native lib via `ctypes.cdll.LoadLibrary` (`python-package/lightgbm/libpath.py`).
- R wrapper requires R; loads native `lightgbm.{so,dll,dylib}` (note: R build strips the `lib_` prefix, see `CMakeLists.txt` `__BUILD_FOR_R` block).

**Package Manager (build orchestration):**
- CMake >= 3.28 (`cmake_minimum_required(VERSION 3.28)` in `CMakeLists.txt`).
- Python build: `scikit-build-core>=0.10.1` backend + Ninja >= 1.11 (`python-package/pyproject.toml`, `[build-system]` / `[tool.scikit-build]`). Helper script `build-python.sh`.
- R build: autoconf-based (`R-package/configure`, `configure.ac`, `Makevars.in`) plus `build-cran-package.sh` / `build_r.R`.
- Lockfile: none for C++ (deps vendored via git submodules); Python deps pinned loosely in `pyproject.toml`.

## Frameworks

**Core (in-house, not third-party frameworks):**
- GBDT engine - `src/boosting/gbdt.cpp` (+ `dart`, `rf`, `goss` strategies via `src/boosting/boosting.cpp`, `sample_strategy.cpp`).
- Tree learners - `src/treelearner/serial_tree_learner.cpp` plus parallel variants (`data_parallel_tree_learner.cpp`, `feature_parallel_tree_learner.cpp`, `voting_parallel_tree_learner.cpp`), `linear_tree_learner.cpp`, `gpu_tree_learner.cpp`.

**Testing:**
- Google Test v1.14.0 - C++ unit tests, fetched via CMake `FetchContent` when not found in system (`CMakeLists.txt` `BUILD_CPP_TEST` block). Test sources in `tests/cpp_tests/` (e.g. `test_arrow.cpp`, `test_stream.cpp`, `test_single_row.cpp`).
- pytest - Python tests under `tests/python_package_test/` (not core).
- testthat - R tests under `R-package/tests/`.

**Build/Dev:**
- Ninja - default generator for Python builds.
- SWIG + JDK (`Java`, `JNI`, `UseJava`, `UseSWIG`) - only when `USE_SWIG=ON`, generates Java API (`CMakeLists.txt`).
- pre-commit (`.pre-commit-config.yaml`), biome (`biome.json`, JS/JSON lint), yamllint, typos (`.typos.toml`), editorconfig (`.editorconfig`).

## Key Dependencies

All C++ third-party deps are vendored as **git submodules** under `external_libs/` (declared in `.gitmodules`). In this checkout the submodule directories are present but **empty/uninitialized** — they must be fetched before a real build.

**Critical (header-only, always compiled in):**
- Eigen (`external_libs/eigen`, from `gitlab.com/libeigen/eigen`) - Linear algebra; used by linear-tree leaf fitting. Included globally (`include_directories(${EIGEN_DIR})`). Compiled with `-DEIGEN_MPL2_ONLY` and `-DEIGEN_DONT_PARALLELIZE` (`CMakeLists.txt`). MPL2 licensing constraint matters for a port.
- fmt (`external_libs/fmt`, from `github.com/fmtlib/fmt`) - String formatting. Header dir `external_libs/fmt/include`. On MSVC requires `/utf-8`.
- fast_double_parser (`external_libs/fast_double_parser`, from `github.com/lemire/fast_double_parser`) - Fast text→double parsing in the data parser (`src/io/parser.cpp`). Header dir `external_libs/fast_double_parser/include`.

**Vendored in-tree (not submodules):**
- json11 - JSON model serialization, vendored directly as `src/io/json11.cpp`.
- yamc - C++ shared-mutex/lock library, vendored under `include/LightGBM/utils/yamc/`.

**Infrastructure / optional:**
- Boost.Compute (`external_libs/compute`, from `boostorg/compute`) - OpenCL abstraction; only used when `USE_GPU=ON`. Also pulls system Boost `filesystem` + `system` >= 1.56.0 (`find_package(Boost 1.56.0 ...)`).
- OpenCL ICD loader - required for `USE_GPU=ON` (`find_package(OpenCL REQUIRED)`); can be vendored/built via `cmake/IntegratedOpenCL.cmake` when `__INTEGRATE_OPENCL=ON`.
- CUDA Toolkit >= 11.0 - required for `USE_CUDA=ON` (`find_package(CUDAToolkit 11.0 REQUIRED)`).
- OpenMP - parallelism, ON by default (`USE_OPENMP=ON`); wrapped in `src/utils/openmp_wrapper.cpp` and `include/LightGBM/utils/openmp_wrapper.h`. Forced ON when CUDA enabled.
- MPI - optional distributed backend (`USE_MPI=ON` → `find_package(MPI REQUIRED)`, `-DUSE_MPI`); otherwise socket backend (`-DUSE_SOCKET`).

**Python wrapper runtime deps** (`python-package/pyproject.toml`):
- Required: `numpy>=1.17.0`, `scipy`.
- Optional extras: `pyarrow>=6.0.1`+`cffi` (arrow), `dask`+`pandas` (dask), `pandas` (pandas), `graphviz`+`matplotlib` (plotting), `scikit-learn>=0.24.2` (sklearn API).

## Configuration

**Build options** (all CMake `option()` in `CMakeLists.txt`, default in parens):
- `USE_MPI` (OFF) - MPI distributed learning; else socket networking.
- `USE_OPENMP` (ON) - OpenMP multithreading.
- `USE_GPU` (OFF) - OpenCL/Boost.Compute GPU training.
- `USE_CUDA` (OFF) - CUDA GPU training (forces OpenMP ON).
- `USE_SWIG` (OFF) - generate Java API.
- `USE_CUDA` arch list auto-derived from toolkit version: SM 60-90/100/120 (`CUDA_ARCHS` block).
- `USE_TIMETAG` (OFF) `-DTIMETAG`, `USE_DEBUG` (OFF) `-DDEBUG`, `USE_SANITIZER` (OFF, `cmake/Sanitizer.cmake`, sanitizers: address/leak/undefined/thread).
- `BUILD_CLI` (ON) - build `lightgbm` CLI in addition to the lib.
- `BUILD_CPP_TEST` (OFF), `BUILD_STATIC_LIB` (OFF), `INSTALL_HEADERS` (ON).
- `__BUILD_FOR_PYTHON`, `__BUILD_FOR_R`, `__INTEGRATE_OPENCL` - internal flags set by the wrapper build scripts; disable CLI + header install.
- macOS: `USE_HOMEBREW_FALLBACK` (ON) to find Homebrew `libomp`.

**Compile-time feature macros injected via CMake:** `-DUSE_MPI`/`-DUSE_SOCKET`, `-DUSE_GPU`, `-DUSE_CUDA`, `-DEIGEN_MPL2_ONLY`, `-DEIGEN_DONT_PARALLELIZE`, `-DMM_PREFETCH`/`-DMM_MALLOC` (detected via `check_cxx_source_compiles`), `-DLGB_R_BUILD` (R), `-DWIN_HAS_INET_PTON` (Windows).

**Core numeric types** (`include/LightGBM/meta.h`) — important for a port to reproduce exactly:
- `data_size_t = int32_t` (row/index size; deliberately signed).
- `score_t = float` by default (gradients/scores); switchable to `double` via `SCORE_T_USE_DOUBLE`.
- `label_t = float` by default; switchable via `LABEL_T_USE_DOUBLE`.
- `comm_size_t = int32_t`. `kEpsilon = 1e-15f`, `kZeroThreshold = 1e-35f`, `kAlignedSize = 32`.

**Runtime model/training configuration:** `include/LightGBM/config.h` (~71 KB) + generated `src/io/config_auto.cpp`. CLI/config-file key parsing in `src/io/config.cpp`. Default `device_type = "cpu"`, `objective = "regression"`, `boosting = "gbdt"`, `tree_learner = "serial"`, `num_machines = 1` (`include/LightGBM/config.h`).

**Compiler flags (non-MSVC, release):** `-O3 -funroll-loops -fPIC -pthread -Wextra -Wall`. MSVC release: `/O2 /Ob2 /Oi /Ot /Oy /W4 /MP /utf-8`. (`CMakeLists.txt`.)

## Platform Requirements

**Development / minimum toolchains** (`CMakeLists.txt` version gates):
- GCC >= 4.8.2, Clang >= 3.8, AppleClang >= 8.1.0, or MSVC >= 1900 (VS2015).
- CMake >= 3.28.
- For CUDA: CUDA Toolkit >= 11.0 (CMake host compiler = the C++ compiler).
- For GPU(OpenCL): Boost >= 1.56.0 + an OpenCL ICD.
- For Java: SWIG + a JDK (`JAVA_HOME` set).
- For SSE prefetch/malloc intrinsics: `<xmmintrin.h>` / `<mm_malloc.h>` (auto-detected, optional).

**Production / deployment targets:**
- Linux / macOS / Windows (incl. MinGW, Cygwin). Windows networking links `ws2_32`, `iphlpapi`.
- Distribution channels: CLI binary, C/C++ shared lib, PyPI wheel (`build-python.sh`), CRAN package (`build-cran-package.sh`), NuGet (`.ci/create-nuget.py`), Docker images (`docker/dockerfile-cli`, `dockerfile-python`, `dockerfile-r`, `docker/gpu/`).

---

*Stack analysis: 2026-06-05*
