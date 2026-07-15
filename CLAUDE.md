## Project

**LightGBM-rs — Pure Rust LightGBM with CubeCL**

A pure-Rust rewrite of Microsoft's LightGBM gradient-boosting library, built as a Cargo workspace and using the `cubecl` crate for compute and GPU acceleration. It targets ML practitioners and LightGBM users who want a memory-safe Rust implementation that runs on both CPU and AMD ROCm GPUs while remaining numerically faithful to the original. The Microsoft C++ implementation under `LightGBM/` is the read-only reference being ported; the deliverable is the Rust crate(s).

**Core Value:** For identical inputs and configuration, the Rust implementation must reproduce the C++ LightGBM's outputs to within an absolute difference of **~1e-6 on every backend (CPU and ROCm)**, using `f32` (single-precision) data types end-to-end to match the C++ reference defaults (`score_t`/`label_t` = `float`). The CPU path — the `cubecl-cpu` f64-fold deterministic anchor — is the hard merge gate and, where the algorithm permits, achieves **bit-exact** parity with the C++ reference (e.g. binning, and the serial tree learner is bit-exact vs real `lib_lightgbm` 4.6 on both committed corpora); the ROCm path (`cubecl-hip`, f32) is held to ~1e-6 against that anchor, with residual f32-vs-f64 accumulation gaps documented per phase. Numerical fidelity at single precision is the non-negotiable contract; everything else serves it. _(Revised from an earlier 1e-12 framing — Decided 2026-06-05, Phase 1 discuss: 1e-12 is unachievable/meaningless against an f32 reference. See PROJECT.md Key Decisions.)_

### Constraints

- **Tech stack**: Pure Rust, Cargo workspace, `cubecl` for compute — no raw CUDA/OpenCL. Use latest available crate versions.
- **Compatibility**: 100% behavioral compatibility with C++ LightGBM for in-scope APIs, configs, and internal specifications (binning, split logic, model format).
- **Numerical**: `f32` (single-precision) data types end-to-end, matching the C++ `score_t`/`label_t` = `float` defaults; absolute output difference ≤ ~1e-6 vs C++ reference on **both** CPU and ROCm backends. The `cubecl-cpu` f64-fold path is the deterministic reference anchor and hard merge gate — bit-exact to C++ where the algorithm permits (e.g. the serial tree learner is bit-exact vs real `lib_lightgbm` 4.6 on both committed corpora); `cubecl-hip` (f32) is a separate ~1e-6 best-effort gate (residual f32-vs-f64 accumulation gaps documented per phase, e.g. `04-ROCM-GAPS.md`).
- **Hardware**: Tests validated on a **local ROCm GPU**; CubeCL `Plane` API used for warp-level ops.
- **Backends**: CPU and ROCm must be switchable (Cargo features and/or runtime configuration).
- **Error handling**: `thiserror` for structured domain errors at library boundaries; `anyhow` for ergonomic propagation in app/high-level layers.
- **Bindings**: Python interface must mirror the official `lightgbm` package API surface.


## Languages

- C++11 - Core training/inference engine. Entire `src/` tree (boosting, treelearner, io, objective, metric, network, application). Standard fixed in `CMakeLists.txt`: `set(CMAKE_CXX_STANDARD 11)` (raised to 14 only when `BUILD_CPP_TEST=ON`, which pulls Google Test). `CMAKE_CXX_STANDARD_REQUIRED ON`.
- C - Project declares `LANGUAGES C CXX` in `CMakeLists.txt`; the public C API (`include/LightGBM/c_api.h`) is C-callable via `extern "C"` (see `include/LightGBM/export.h`).
- CUDA C++ - GPU backend, compiled only when `USE_CUDA=ON`. `.cu` files under `src/**/cuda/` and `src/cuda/`. `CMAKE_CUDA_STANDARD 11` (`CMakeLists.txt`).
- OpenCL C - GPU kernels for the (non-CUDA) `gpu` device type: `src/treelearner/ocl/histogram16.cl`, `histogram64.cl`, `histogram256.cl`.
- Python 3.7+ - Wrapper package `python-package/lightgbm/` (pure-Python ctypes binding, no compiled extension of its own).
- R - Wrapper package `R-package/` with a thin C++ shim `R-package/src/lightgbm_R.cpp`.
- SWIG interface (`.i`) - Java/JNI binding generators in `swig/` (e.g. `swig/lightgbmlib.i`).

## Runtime

- Native compiled shared/static library (`lib_lightgbm.{so,dll,dylib}`) + standalone CLI executable (`lightgbm`). No managed runtime for the core.
- Python wrapper requires CPython 3.7-3.13 (`python-package/pyproject.toml` classifiers), loads the native lib via `ctypes.cdll.LoadLibrary` (`python-package/lightgbm/libpath.py`).
- R wrapper requires R; loads native `lightgbm.{so,dll,dylib}` (note: R build strips the `lib_` prefix, see `CMakeLists.txt` `__BUILD_FOR_R` block).
- CMake >= 3.28 (`cmake_minimum_required(VERSION 3.28)` in `CMakeLists.txt`).
- Python build: `scikit-build-core>=0.10.1` backend + Ninja >= 1.11 (`python-package/pyproject.toml`, `[build-system]` / `[tool.scikit-build]`). Helper script `build-python.sh`.
- R build: autoconf-based (`R-package/configure`, `configure.ac`, `Makevars.in`) plus `build-cran-package.sh` / `build_r.R`.
- Lockfile: none for C++ (deps vendored via git submodules); Python deps pinned loosely in `pyproject.toml`.

## Frameworks

- GBDT engine - `src/boosting/gbdt.cpp` (+ `dart`, `rf`, `goss` strategies via `src/boosting/boosting.cpp`, `sample_strategy.cpp`).
- Tree learners - `src/treelearner/serial_tree_learner.cpp` plus parallel variants (`data_parallel_tree_learner.cpp`, `feature_parallel_tree_learner.cpp`, `voting_parallel_tree_learner.cpp`), `linear_tree_learner.cpp`, `gpu_tree_learner.cpp`.
- Google Test v1.14.0 - C++ unit tests, fetched via CMake `FetchContent` when not found in system (`CMakeLists.txt` `BUILD_CPP_TEST` block). Test sources in `tests/cpp_tests/` (e.g. `test_arrow.cpp`, `test_stream.cpp`, `test_single_row.cpp`).
- pytest - Python tests under `tests/python_package_test/` (not core).
- testthat - R tests under `R-package/tests/`.
- Ninja - default generator for Python builds.
- SWIG + JDK (`Java`, `JNI`, `UseJava`, `UseSWIG`) - only when `USE_SWIG=ON`, generates Java API (`CMakeLists.txt`).
- pre-commit (`.pre-commit-config.yaml`), biome (`biome.json`, JS/JSON lint), yamllint, typos (`.typos.toml`), editorconfig (`.editorconfig`).

## Key Dependencies

- Eigen (`external_libs/eigen`, from `gitlab.com/libeigen/eigen`) - Linear algebra; used by linear-tree leaf fitting. Included globally (`include_directories(${EIGEN_DIR})`). Compiled with `-DEIGEN_MPL2_ONLY` and `-DEIGEN_DONT_PARALLELIZE` (`CMakeLists.txt`). MPL2 licensing constraint matters for a port.
- fmt (`external_libs/fmt`, from `github.com/fmtlib/fmt`) - String formatting. Header dir `external_libs/fmt/include`. On MSVC requires `/utf-8`.
- fast_double_parser (`external_libs/fast_double_parser`, from `github.com/lemire/fast_double_parser`) - Fast text→double parsing in the data parser (`src/io/parser.cpp`). Header dir `external_libs/fast_double_parser/include`.
- json11 - JSON model serialization, vendored directly as `src/io/json11.cpp`.
- yamc - C++ shared-mutex/lock library, vendored under `include/LightGBM/utils/yamc/`.
- Boost.Compute (`external_libs/compute`, from `boostorg/compute`) - OpenCL abstraction; only used when `USE_GPU=ON`. Also pulls system Boost `filesystem` + `system` >= 1.56.0 (`find_package(Boost 1.56.0 ...)`).
- OpenCL ICD loader - required for `USE_GPU=ON` (`find_package(OpenCL REQUIRED)`); can be vendored/built via `cmake/IntegratedOpenCL.cmake` when `__INTEGRATE_OPENCL=ON`.
- CUDA Toolkit >= 11.0 - required for `USE_CUDA=ON` (`find_package(CUDAToolkit 11.0 REQUIRED)`).
- OpenMP - parallelism, ON by default (`USE_OPENMP=ON`); wrapped in `src/utils/openmp_wrapper.cpp` and `include/LightGBM/utils/openmp_wrapper.h`. Forced ON when CUDA enabled.
- MPI - optional distributed backend (`USE_MPI=ON` → `find_package(MPI REQUIRED)`, `-DUSE_MPI`); otherwise socket backend (`-DUSE_SOCKET`).
- Required: `numpy>=1.17.0`, `scipy`.
- Optional extras: `pyarrow>=6.0.1`+`cffi` (arrow), `dask`+`pandas` (dask), `pandas` (pandas), `graphviz`+`matplotlib` (plotting), `scikit-learn>=0.24.2` (sklearn API).

## Configuration

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
- `data_size_t = int32_t` (row/index size; deliberately signed).
- `score_t = float` by default (gradients/scores); switchable to `double` via `SCORE_T_USE_DOUBLE`.
- `label_t = float` by default; switchable via `LABEL_T_USE_DOUBLE`.
- `comm_size_t = int32_t`. `kEpsilon = 1e-15f`, `kZeroThreshold = 1e-35f`, `kAlignedSize = 32`.

## Platform Requirements

- GCC >= 4.8.2, Clang >= 3.8, AppleClang >= 8.1.0, or MSVC >= 1900 (VS2015).
- CMake >= 3.28.
- For CUDA: CUDA Toolkit >= 11.0 (CMake host compiler = the C++ compiler).
- For GPU(OpenCL): Boost >= 1.56.0 + an OpenCL ICD.
- For Java: SWIG + a JDK (`JAVA_HOME` set).
- For SSE prefetch/malloc intrinsics: `<xmmintrin.h>` / `<mm_malloc.h>` (auto-detected, optional).
- Linux / macOS / Windows (incl. MinGW, Cygwin). Windows networking links `ws2_32`, `iphlpapi`.
- Distribution channels: CLI binary, C/C++ shared lib, PyPI wheel (`build-python.sh`), CRAN package (`build-cran-package.sh`), NuGet (`.ci/create-nuget.py`), Docker images (`docker/dockerfile-cli`, `dockerfile-python`, `dockerfile-r`, `docker/gpu/`).



## Linters & Formatters Present

| Tool | Config / Invocation | Scope |
|------|---------------------|-------|
| `cpplint` | `LightGBM/.ci/lint-cpp.sh` (filters: `-build/c++11,-build/include_subdir,-build/header_guard,-whitespace/line_length`) | `src/ include/ R-package/ swig/ tests/` |
| `cmakelint` | `LightGBM/.ci/lint-cpp.sh` (`--linelength=120`) | `CMakeLists.txt`, `cmake/*.cmake` |
| Custom OpenMP grep | `LightGBM/.ci/lint-cpp.sh` | Every `#pragma omp parallel` must specify `num_threads()` |
| EditorConfig | `LightGBM/.editorconfig` | All files; C++ = 2-space indent, Python/sh/js/json = 4-space, max line 120 |
| `ruff` (lint + format) | `LightGBM/python-package/pyproject.toml` `[tool.ruff]` | Python; line-length 120, double quotes, numpy docstrings |
| `mypy` | `LightGBM/python-package/pyproject.toml` `[tool.mypy]` (`disallow_untyped_defs = true`) | `python-package/` |
| `biome` | `LightGBM/biome.json` (formatter + linter `all`, organizeImports) | JavaScript/JSON |
| `lintr` | `LightGBM/.ci/lint-r-code.R` | R code |
| `typos` | `LightGBM/.typos.toml` | spelling, all files |
| `yamllint` | `LightGBM/.yamllint.yml` (`--strict`) | YAML |
| `shellcheck` | pre-commit hook | shell scripts |
| `pre-commit` | `LightGBM/.pre-commit-config.yaml` | end-of-file-fixer, trailing-whitespace, ruff, shellcheck, typos, yamllint, validate-pyproject |

## Naming Patterns

- Headers: `snake_case.h` in `include/LightGBM/` and `include/LightGBM/utils/` (e.g. `tree.h`,
- Implementation: `snake_case.cpp` under `src/<subsystem>/` (e.g. `src/boosting/gbdt.cpp`,
- A few `.cu` / `.cpp` pairs in `src/cuda/` and `src/treelearner/cuda/` for CUDA.
- `typedef int32_t data_size_t;` — row/sample index type (line 28)
- `typedef float score_t;` (or `double` if `SCORE_T_USE_DOUBLE`) — gradient/score type (line 38-40)
- `typedef float label_t;` (or `double` if `LABEL_T_USE_DOUBLE`) — label type (line 45-47)
- `typedef int32_t comm_size_t;` — network/communication sizes (line 59)
- Function-pointer typedefs `PredictFunction`, `ReduceFunction`, `AllgatherFunction`, etc.

## OpenMP Pragma Conventions

#pragma omp parallel for num_threads(OMP_NUM_THREADS()) schedule(static)

- `OMP_NUM_THREADS()` is the project's wrapper (declared in
- Reductions are written explicitly: `reduction(+:sum_gradient, sum_hessian)`.

#pragma omp parallel for num_threads(OMP_NUM_THREADS())

## Memory & Ownership Patterns

- **Ownership via `std::unique_ptr`.** Long-lived owned subsystems are `std::unique_ptr<T>`
- **Factory functions return raw owning pointers** that the caller immediately wraps:
- Non-owning references to shared data are passed as raw pointers / `const&` (e.g.
- Construction-from-string is a recurring pattern: `Tree(const char* str, size_t* used_len)`

## Logging

- Single static `Log` class in `include/LightGBM/utils/log.h`. Levels: `Fatal(-1)`,
- `printf`-style varargs API: `Log::Info("...%d...", x)`, `Log::Warning(...)`, `Log::Debug(...)`,
- Output is prefixed `[LightGBM] [<Level>] ...`. A user `Callback` can be registered to redirect
- R builds (`LGB_R_BUILD`) route output through `Rprintf`/`REprintf` + `R_FlushConsole` instead

## Config-Driven Design (Config struct)

- `struct Config` (`include/LightGBM/config.h`) is the single bag of all training/prediction
- **`config.h` is the single source of truth.** `LightGBM/.ci/parameter-generator.py` parses
- Special annotations recognized in `config.h` doc comments: `[no-automatically-extract]`
- **Port implication:** parameter names, aliases, defaults, and the auto-extraction rules must

## Python Binding Conventions

- Package at `LightGBM/python-package/lightgbm/`. Public modules: `basic.py` (low-level ctypes
- **Style:** ruff-formatted, double quotes, line length 120, numpy-style docstrings
- **Selected ruff rule groups:** `B, C4, D, E, F, I, NPY, PL, RET, SIM, T, W` (flake8-bugbear,
- **Optional dependencies** are guarded with `try/except ImportError` and `*_INSTALLED` flags in
- isort first-party = `lightgbm` (`[tool.ruff.lint.isort]`). `py.typed` marker present

## R Binding Conventions

- Package at `LightGBM/R-package/`. R sources in `R-package/R/`, C++ glue in `R-package/src/`,
- Linted by `lintr` via `LightGBM/.ci/lint-r-code.R`, which enforces an opinionated linter set,
- The core builds with `-DLGB_R_BUILD` so logging and error output route through R's I/O

## Function & Module Design

- C++ public methods are heavily Doxygen-documented; many small `inline` accessors return members
- Functions can take many parameters (`Tree::Split` takes ~14); pylint's "too many args" check is
- Polymorphism is via abstract base classes + string-keyed factories (see Memory & Ownership);



## Component Responsibilities

| Component | Responsibility | File |
|-----------|----------------|------|
| `Application` | CLI orchestration: parse params, load data, init train/predict | `src/application/application.cpp` |
| `Boosting` / `GBDT` | Ensemble gradient-boosting loop; owns models, score updaters, sample strategy | `include/LightGBM/boosting.h`, `src/boosting/gbdt.cpp` |
| `ObjectiveFunction` | Compute per-row gradients & hessians from current scores **[GPU-RELEVANT]** | `include/LightGBM/objective_function.h`, `src/objective/*.hpp` |
| `TreeLearner` | Grow one decision tree per call: histograms, split finding, partition **[GPU-RELEVANT]** | `include/LightGBM/tree_learner.h`, `src/treelearner/serial_tree_learner.cpp` |
| `Dataset` | Owns binned feature data (FeatureGroups), metadata; builds histograms | `include/LightGBM/dataset.h`, `src/io/dataset.cpp` |
| `Bin` / `BinMapper` | Feature binning (continuous→bin), per-bin histogram accumulation **[GPU-RELEVANT]** | `include/LightGBM/bin.h`, `src/io/bin.cpp`, `src/io/dense_bin.hpp` |
| `Metric` | Evaluate loss/metric on train & validation scores **[GPU-RELEVANT]** | `include/LightGBM/metric.h`, `src/metric/*.hpp` |
| `Tree` | Single decision tree model: nodes, splits, leaf outputs, prediction | `include/LightGBM/tree.h`, `src/io/tree.cpp` |
| `Predictor` | Batch prediction driver over a file/dataset | `src/application/predictor.hpp` |
| `Network` | Distributed allreduce/allgather over MPI or sockets | `include/LightGBM/network.h`, `src/network/network.cpp` |
| C API | Stable ABI exposing Dataset/Booster to bindings | `include/LightGBM/c_api.h`, `src/c_api.cpp` |

## Pattern Overview

- Every major subsystem is an **abstract base class with a static `Create*` factory** that switches on a string type + `device_type` (`cpu`/`gpu`/`cuda`). This is the primary porting seam — the Rust port replaces each factory + implementation.
- **Template-based parallel learners**: parallel tree learners (`FeatureParallelTreeLearner<T>`, `DataParallelTreeLearner<T>`, `VotingParallelTreeLearner<T>`, `LinearTreeLearner<T>`) are templated over a base serial/GPU learner — see `src/treelearner/tree_learner.cpp`.
- **Histogram-based split finding** is the core algorithm: data is pre-binned once into integer bins; per-node histograms of (sum_gradient, sum_hessian) are accumulated over bins, and splits are found by scanning histograms — not by sorting raw values.
- **Histogram subtraction trick**: the larger child's histogram is derived by subtracting the smaller child's histogram from the parent's (`use_subtract` in `ConstructHistograms`).
- **Device abstraction is compile-time + runtime**: CPU implementations are the default; `gpu` (OpenCL, `src/treelearner/ocl/*.cl`) and `cuda` (`.cu` files) are alternative backends guarded by `USE_GPU`/`USE_CUDA` macros.

## Layers

- Purpose: Translate external requests (CLI args, C ABI calls) into engine operations.
- Location: `src/application/`, `src/c_api.cpp`, `src/main.cpp`
- Contains: Argument parsing, `Application::Run()` dispatch, ~120 `LGBM_*` C functions.
- Depends on: Boosting, Dataset, Metric, Objective layers.
- Used by: CLI binary, language bindings (Python/R/etc.).
- Purpose: Run the outer gradient-boosting loop, manage the ensemble, bagging, scores, early stopping.
- Location: `src/boosting/`
- Contains: `GBDT`, `DART`, `RF`, `GOSS`/bagging sample strategies, score updaters, prediction.
- Depends on: ObjectiveFunction (grad/hess), TreeLearner (grow tree), Metric (eval).
- Used by: Application, C API.
- Purpose: Define the loss (gradients/hessians) and evaluation metrics.
- Location: `src/objective/`, `src/metric/` (header-only `.hpp` implementations + `.cpp` factories).
- Depends on: Dataset metadata (labels, weights, query boundaries).
- Used by: Boosting layer.
- Purpose: Grow one tree: build feature histograms, find best splits, partition data, set leaf outputs.
- Location: `src/treelearner/`
- Depends on: Dataset (binned data + histogram construction), Network (parallel learners).
- Used by: Boosting layer.
- Purpose: Load raw data, bin features, store binned representation, hold the tree model.
- Location: `src/io/`
- Depends on: utils, network (distributed loading).
- Used by: all upper layers.
- Purpose: Collective communication for distributed training.
- Location: `src/network/` (MPI `linkers_mpi.cpp`, socket `linkers_socket.cpp`).


### Model Serialization

- Text/JSON dump and load: `src/boosting/gbdt_model_text.cpp`, tree text in `src/io/tree.cpp`; if-else C++ codegen in the same files.
- Mutable training state lives in `GBDT` members (`models_`, `gradients_`, `hessians_`, score updaters) and in `TreeLearner` (`data_partition_`, histogram pool, `leaf_splits_`). `Dataset` is immutable after `FinishLoad`. Scores are accumulated in `ScoreUpdater` (`src/boosting/score_updater.hpp`).

## Key Abstractions

- Purpose: Ensemble strategy. `TrainOneIter`, `Predict*`, model I/O.
- Implementations: `GBDT` (`src/boosting/gbdt.cpp`), `DART` (`src/boosting/dart.hpp`), `RF` (`src/boosting/rf.hpp`); `GBDTBase` adds leaf get/set.
- Factory: `Boosting::CreateBoosting(type, filename)` (`src/boosting/boosting.cpp:34`).
- Purpose: `GetGradients(score, gradients, hessians)`; `ConvertOutput`, `BoostFromScore`.
- Implementations (header-only): regression (`src/objective/regression_objective.hpp`), binary (`binary_objective.hpp`), multiclass (`multiclass_objective.hpp`), ranking/lambdarank (`rank_objective.hpp`), cross-entropy (`xentropy_objective.hpp`); CUDA mirrors in `src/objective/cuda/`.
- Factory: `ObjectiveFunction::CreateObjectiveFunction(type, config)` (`src/objective/objective_function.cpp`).
- Purpose: `Train(gradients, hessians, is_first_tree) → Tree*`.
- Implementations: `SerialTreeLearner` (`src/treelearner/serial_tree_learner.cpp`), `GPUTreeLearner` (OpenCL, `src/treelearner/gpu_tree_learner.cpp`), `CUDASingleGPUTreeLearner` (`src/treelearner/cuda/cuda_single_gpu_tree_learner.cpp`), and templated parallel wrappers.
- Factory: `TreeLearner::CreateTreeLearner(learner_type, device_type, config, boosting_on_cuda)` (`src/treelearner/tree_learner.cpp:15`).
- Purpose: Binned columnar storage + histogram construction. `BinType` (numerical/categorical), `BinMapper::FindBin`, `Bin::ConstructHistogram*`, `MultiValBin` for grouped features.
- Implementations: `DenseBin` (`src/io/dense_bin.hpp`), `SparseBin` (`src/io/sparse_bin.hpp`), `MultiValDenseBin`/`MultiValSparseBin`.
- Purpose: `Eval(score, objective) → vector<double>`. Plus static `DCGCalculator` for NDCG.
- Implementations: regression/binary/multiclass/rank/xentropy metrics in `src/metric/*.hpp`; CUDA in `src/metric/cuda/`.

## Entry Points

- Location: `src/main.cpp` → `LightGBM::Application(argc, argv).Run()`.
- Triggers: command line invocation (`task=train|predict|convert_model|refit`).
- Responsibilities: top-level exception handling, MPI finalize.
- Location: ~120 `LGBM_Dataset*` and `LGBM_Booster*` functions (e.g. `LGBM_DatasetCreateFromMat:1299`, `LGBM_BoosterCreate:1939`, `LGBM_BoosterUpdateOneIter`, `LGBM_BoosterPredictForMat`).
- Triggers: language bindings.
- Responsibilities: stable ABI, handle lifetime, thread-safety wrappers; the primary surface the Rust crate must reproduce or FFI-bridge.

## Architectural Constraints

- **Threading:** Shared-memory parallelism via **OpenMP** (`#pragma omp parallel for num_threads(OMP_NUM_THREADS())`) pervasively in the boosting loop, histogram construction, and prediction. `src/utils/openmp_wrapper.cpp` and `include/LightGBM/utils/openmp_wrapper.h` provide the wrapper. The Rust port must map this onto rayon / cubecl kernels.
- **Device backends:** Selected at runtime by `config_->device_type` but gated at compile time by `USE_GPU` (OpenCL) and `USE_CUDA` macros. `LGBM_config_::current_device` / `current_learner` are process-global (`src/boosting/gbdt.cpp:24`).
- **Global state:** `Common::Timer global_timer` (`src/boosting/gbdt.cpp:22`); static `DCGCalculator` tables (`include/LightGBM/metric.h:133`); process-global device config. These are module-level singletons.
- **Histogram pool memory:** Tree learners use a fixed-size histogram pool with the subtraction trick; memory is sized by `num_leaves` × total bins.
- **Immutability:** `Dataset` is read-only after `FinishLoad()`; histograms and partitions are the only per-tree mutable structures.


## Error Handling

- `CHECK_*` macros (`CHECK_EQ`, `CHECK_NOTNULL`, `CHECK_GT`) assert invariants and fatal on failure.
- `OMP_INIT_EX()/OMP_LOOP_EX_BEGIN()/OMP_THROW_EX()` propagate exceptions out of OpenMP parallel regions.

## kaggle notebook
mkdir -p ~/.kaggle && echo KGAT_2966b842a0ca6e3c1029fbfea8657f97 > ~/.kaggle/access_token && chmod 600 ~/.kaggle/access_token


@AGENTS.md
