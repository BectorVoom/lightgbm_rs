# Coding Conventions

**Analysis Date:** 2026-06-05

> **Scope note:** This document maps the **LightGBM C++ REFERENCE implementation** (the
> system being ported), located under `LightGBM/`. It does NOT describe the Rust + cubecl
> crate under development. A porter should use these conventions to mirror the reference
> implementation's behavior faithfully.

LightGBM version under analysis: `4.6.0.99` (`LightGBM/VERSION.txt`).

## Linters & Formatters Present

There is **no `.clang-format` or `.clang-tidy`** file in the repo. C++ style is enforced
by external tools invoked from CI scripts, plus EditorConfig and a custom OpenMP grep check.

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

The umbrella lint task lives in `LightGBM/.ci/test.sh` under `TASK == "lint"`.

## Naming Patterns

**Files (C++):**
- Headers: `snake_case.h` in `include/LightGBM/` and `include/LightGBM/utils/` (e.g. `tree.h`,
  `dataset_loader.h`, `openmp_wrapper.h`). Template-heavy headers use `.hpp`
  (`chunked_array.hpp`, `multi_val_sparse_bin.hpp`); template implementations use `.tpp`
  (`arrow.tpp`).
- Implementation: `snake_case.cpp` under `src/<subsystem>/` (e.g. `src/boosting/gbdt.cpp`,
  `src/treelearner/serial_tree_learner.cpp`).
- A few `.cu` / `.cpp` pairs in `src/cuda/` and `src/treelearner/cuda/` for CUDA.

**Types / classes:** `PascalCase` — `Tree`, `Config`, `Boosting`, `ObjectiveFunction`,
`TreeLearner`, `Metric`, `ScoreUpdater`. Enums are `enum class` with `PascalCase` members
(`LogLevel::Fatal`, see `include/LightGBM/utils/log.h:77`).

**Functions / methods:** `PascalCase` — `Tree::Split(...)`, `Boosting::CreateBoosting(...)`,
`Config::GetMembersFromString(...)`. This is Google-C++-style PascalCase methods, not camelCase.

**Member variables:** `snake_case_` with a **trailing underscore** (e.g.
`num_tree_per_iteration_`, `models_`, `train_data_`, `config_`; see `src/boosting/gbdt.h`).

**Constants:** `k`-prefixed PascalCase — `kDefaultNumLeaves` (`config.h`), `kMinScore`,
`kMaxScore`, `kEpsilon` (`include/LightGBM/meta.h:50-54`), `kCategoricalMask` (`tree.h:20`).

**Typedefs (core scalar aliases — replicate these in the port):** defined in
`include/LightGBM/meta.h`:
- `typedef int32_t data_size_t;` — row/sample index type (line 28)
- `typedef float score_t;` (or `double` if `SCORE_T_USE_DOUBLE`) — gradient/score type (line 38-40)
- `typedef float label_t;` (or `double` if `LABEL_T_USE_DOUBLE`) — label type (line 45-47)
- `typedef int32_t comm_size_t;` — network/communication sizes (line 59)
- Function-pointer typedefs `PredictFunction`, `ReduceFunction`, `AllgatherFunction`, etc.

**C API symbols:** all exported functions are prefixed `LGBM_` and use `PascalCase` after the
prefix (`LGBM_BoosterCreate`, `LGBM_DatasetCreateFromFile`, `LGBM_GetLastError`); macros are
`C_API_`-prefixed `SCREAMING_SNAKE` (`C_API_DTYPE_FLOAT64`, `C_API_PREDICT_RAW_SCORE`). See
`include/LightGBM/c_api.h`.

## Header Organization

Every header follows this layout (see `include/LightGBM/tree.h`, `config.h`, `utils/log.h`):

1. Copyright banner: `/*! Copyright (c) <year> Microsoft Corporation ... MIT License ... */`
2. Include guard: `#ifndef LIGHTGBM_<PATH>_H_` / `#define ...` / `#endif` (full-path-based;
   note `cpplint` `-build/header_guard` is disabled, so guards may not match cpplint's expectation).
3. Includes ordered: project headers (`<LightGBM/...>`) first, then C++ stdlib (`<vector>`,
   `<string>`, `<memory>`). Order is not strictly alphabetized.
4. `namespace LightGBM { ... }` wraps all declarations.
5. Doxygen `/*! \brief ... \param ... \return ... */` comments on public classes and methods.

Public API headers carry `\file`/`\note` doc blocks (`c_api.h` documents float32/float64
dual-type rationale at the top).

## OpenMP Pragma Conventions

**Hard rule enforced by CI:** every `#pragma omp parallel` MUST specify an explicit
`num_threads(...)` clause (`LightGBM/.ci/lint-cpp.sh` greps for violations and fails the build).

**Canonical pattern** (see `src/application/predictor.hpp:236`,
`src/io/multi_val_sparse_bin.hpp:88`):
```cpp
#pragma omp parallel for num_threads(OMP_NUM_THREADS()) schedule(static)
```
- `OMP_NUM_THREADS()` is the project's wrapper (declared in
  `include/LightGBM/utils/openmp_wrapper.h`) returning the configured thread count; falls back
  to `1` when `_OPENMP` is undefined. Some sites pass an explicit `num_threads(num_threads)`
  local variable instead (e.g. `src/treelearner/gradient_discretizer.cpp:29`).
- Reductions are written explicitly: `reduction(+:sum_gradient, sum_hessian)`.

**Exception-safe parallel loops:** OpenMP cannot propagate C++ exceptions across the parallel
region boundary, so the codebase uses helper macros from
`include/LightGBM/utils/openmp_wrapper.h:76-87`:
```cpp
OMP_INIT_EX();                          // declares ThreadExceptionHelper omp_except_helper
#pragma omp parallel for num_threads(OMP_NUM_THREADS())
for (...) {
  OMP_LOOP_EX_BEGIN();                  // try {
  ...                                   // body; exceptions captured, not thrown out of loop
  OMP_LOOP_EX_END();                    // } catch(...) { omp_except_helper.CaptureException(); }
}
OMP_THROW_EX();                         // re-throws captured exception after the region
```
When `_OPENMP` is not defined these macros expand to nothing. **Port implication:** the Rust
port must reproduce this "capture exception inside the loop, re-throw after the join" semantics
(in Rust: collect a `Result`/panic flag across the parallel iterator, surface after the join).

Thread count is globally configurable via `LGBM_MAX_NUM_THREADS` / `LGBM_DEFAULT_NUM_THREADS`
externs and `OMP_SET_NUM_THREADS(int)` (`openmp_wrapper.h:10-46`), driven by the `num_threads`
parameter.

## Memory & Ownership Patterns

- **Ownership via `std::unique_ptr`.** Long-lived owned subsystems are `std::unique_ptr<T>`
  members: `config_`, `tree_learner_`, `train_score_updater_`, `data_sample_strategy_`,
  `models_[...]` (see `src/boosting/gbdt.cpp:67,101,107,123,293,389`).
- **Factory functions return raw owning pointers** that the caller immediately wraps:
  `Boosting::CreateBoosting(...)`, `ObjectiveFunction::CreateObjectiveFunction(...)`,
  `TreeLearner::CreateTreeLearner(...)`, `Metric::CreateMetric(...)`,
  `SampleStrategy::CreateSampleStrategy(...)` — declared in `include/LightGBM/boosting.h:314`,
  `metric.h:57`, `tree_learner.h:110`. Callers do `std::unique_ptr<T>(Factory::Create(...))`
  or `member_.reset(Factory::Create(...))`. **Port implication:** these are the polymorphic
  dispatch seams; in Rust they map to trait objects (`Box<dyn Trait>`) or enums selected by
  config string.
- Non-owning references to shared data are passed as raw pointers / `const&` (e.g.
  `train_data_` is a borrowed `const Dataset*`).
- Construction-from-string is a recurring pattern: `Tree(const char* str, size_t* used_len)`
  and `Boosting::CreateBoosting(type, filename)` reconstruct objects from the serialized model
  text — important for the conformance oracle (model files must round-trip identically).

## Error Handling

**Internal (C++ core):** failures call `Log::Fatal(...)` which **throws** `std::runtime_error`
(`include/LightGBM/utils/log.h:116-138`). There are no error-return codes internally — the code
assumes exceptions propagate. `CHECK*` macros (`CHECK`, `CHECK_EQ`, `CHECK_GE`, `CHECK_NOTNULL`,
etc., `log.h:39-74`) call `Log::Fatal` with file/line on failure.

**At the C API boundary** (`src/c_api.cpp:47-52`): exceptions are converted to integer return
codes. Every exported function body is wrapped:
```cpp
int LGBM_Something(...) {
  API_BEGIN();        // try {
  ... real work ...
  API_END();          // } catch(std::exception&){ return LGBM_APIHandleException(ex); }
                      //   catch(std::string&){...} catch(...){ "unknown exception" } return 0;
}
```
`LGBM_APIHandleException` stores the message via `LGBM_SetLastError` (thread-local, retrieved by
`LGBM_GetLastError()`, `c_api.h:1649`) and returns `-1`. Convention: **`0` on success, `-1` on
failure**, last error string fetched separately. Concurrency is guarded by
`UNIQUE_LOCK`/`SHARED_LOCK` macros over a `yamc::alternate::shared_mutex` (`c_api.cpp:54-58`).

**Port implication:** mirror the dual model — internal `Result`/panic for logic errors, plus a
C-ABI shim that catches and maps to `0`/`-1` with a thread-local last-error string, so existing
Python/R bindings can bind to the Rust core unchanged.

## Logging

- Single static `Log` class in `include/LightGBM/utils/log.h`. Levels: `Fatal(-1)`,
  `Warning(0)`, `Info(1)`, `Debug(2)` (`LogLevel` enum, `log.h:77`).
- `printf`-style varargs API: `Log::Info("...%d...", x)`, `Log::Warning(...)`, `Log::Debug(...)`,
  `Log::Fatal(...)` (the last throws).
- Output is prefixed `[LightGBM] [<Level>] ...`. A user `Callback` can be registered to redirect
  output (`Log::ResetCallBack`, exposed as `LGBM_RegisterLogCallback`). Minimum level via
  `Log::ResetLogLevel`.
- R builds (`LGB_R_BUILD`) route output through `Rprintf`/`REprintf` + `R_FlushConsole` instead
  of stdout/stderr. Log level state is `THREAD_LOCAL`.

## Config-Driven Design (Config struct)

- `struct Config` (`include/LightGBM/config.h`) is the single bag of all training/prediction
  parameters. It is constructed from a `std::unordered_map<std::string,std::string>` of
  param→value (`Config::Set` / `Config::GetMembersFromString`), supporting many parameters and
  alias names.
- **`config.h` is the single source of truth.** `LightGBM/.ci/parameter-generator.py` parses
  the `#pragma region Parameters` blocks and Doxygen `desc`/`descl2` annotations in `config.h`
  to **auto-generate**:
  - `src/io/config_auto.cpp` (parameter table, alias table, string parsing) — header comment:
    "This file is auto generated by ... parameter-generator.py from ... config.h".
  - `docs/Parameters.rst` documentation.
  CI re-runs the generator and `diff`s the output to ensure they stay in sync
  (`.ci/test.sh`, `TASK == "check-docs"`).
- Special annotations recognized in `config.h` doc comments: `[no-automatically-extract]`
  (custom parse logic) and `[no-save]` (excluded from saved model text). See `config.h:8-14`.
- **Port implication:** parameter names, aliases, defaults, and the auto-extraction rules must
  be replicated exactly from `config.h` for behavioral parity; consider porting
  `parameter-generator.py` logic or generating the Rust config from the same `config.h`.

## Python Binding Conventions

- Package at `LightGBM/python-package/lightgbm/`. Public modules: `basic.py` (low-level ctypes
  wrapper over the C API: `Dataset`, `Booster`, `LightGBMError`), `engine.py` (`train`, `cv`),
  `sklearn.py` (scikit-learn estimators), `callback.py`, `plotting.py`, `dask.py`,
  `compat.py`, `libpath.py`.
- **Style:** ruff-formatted, double quotes, line length 120, numpy-style docstrings
  (`[tool.ruff.lint.pydocstyle] convention = "numpy"`). `mypy` with `disallow_untyped_defs =
  true` — all defs are type-annotated. `# coding: utf-8` header on each module.
- **Selected ruff rule groups:** `B, C4, D, E, F, I, NPY, PL, RET, SIM, T, W` (flake8-bugbear,
  comprehensions, pydocstyle, pyflakes, isort, numpy, pylint, flake8-print, etc.). Notable
  ignores: `E501` (line length handled by formatter), `D105`, several `PLR*`. See
  `pyproject.toml:120-168`.
- **Optional dependencies** are guarded with `try/except ImportError` and `*_INSTALLED` flags in
  `compat.py` (`SKLEARN_INSTALLED`, `PANDAS_INSTALLED`, `MATPLOTLIB_INSTALLED`, `GRAPHVIZ_INSTALLED`,
  `CFFI_INSTALLED`, dask). Tests skip via these flags.
- isort first-party = `lightgbm` (`[tool.ruff.lint.isort]`). `py.typed` marker present
  (PEP 561 typed package). Build backend is scikit-build-core (`[tool.scikit-build]`).

## R Binding Conventions

- Package at `LightGBM/R-package/`. R sources in `R-package/R/`, C++ glue in `R-package/src/`,
  Roxygen-generated `man/*.Rd`.
- Linted by `lintr` via `LightGBM/.ci/lint-r-code.R`, which enforces an opinionated linter set,
  including: **no pipe operator** (`magrittr` `%>%` forbidden — "this project's code does not use
  the pipe operator"), assignment via `<-`, brace/comma/spacing rules, and bans on
  interactive-only functions (`help`, `install.packages`) in package code. Backport linter
  enforces compatibility with older R.
- The core builds with `-DLGB_R_BUILD` so logging and error output route through R's I/O
  (`R_ext/Print.h`, `REprintf`, `R_FlushConsole`) — see `include/LightGBM/utils/log.h:19-30`.

## Function & Module Design

- C++ public methods are heavily Doxygen-documented; many small `inline` accessors return members
  (`MaxFeatureIdx()`, `FeatureNames()`, see `gbdt.h`).
- Functions can take many parameters (`Tree::Split` takes ~14); pylint's "too many args" check is
  intentionally disabled in Python (`PLR0913`). The reference does not favor small functions —
  porters should not "refactor for taste," but mirror signatures to preserve behavior.
- Polymorphism is via abstract base classes + string-keyed factories (see Memory & Ownership);
  `boosting`, `objective`, `metric`, `tree_learner`, `device_type` config strings select impls.

---

*Convention analysis: 2026-06-05*
