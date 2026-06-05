# Testing Patterns

**Analysis Date:** 2026-06-05

> **Scope note:** This document maps the **LightGBM C++ REFERENCE implementation** (the
> system being ported), located under `LightGBM/`. It does NOT describe the Rust + cubecl
> crate under development. The most important purpose here is identifying which existing tests
> and fixtures can act as a **conformance oracle** for validating the Rust port.

## Test Frameworks

| Layer | Framework | Location | Config / Entry |
|-------|-----------|----------|----------------|
| C++ unit tests | **GoogleTest (gtest)** | `LightGBM/tests/cpp_tests/` | `CMakeLists.txt` (`BUILD_CPP_TEST` option) |
| Python tests | **pytest** + `numpy.testing` | `LightGBM/tests/python_package_test/` | `LightGBM/python-package/pyproject.toml` |
| C API smoke test | Python script | `LightGBM/tests/c_api_test/test_.py` | — |
| Distributed tests | pytest (drives the CLI binary) | `LightGBM/tests/distributed/` | `conftest.py` (`--execfile`) |
| R tests | **testthat** | `LightGBM/R-package/tests/testthat/` | `R-package/tests/testthat.R` |
| if-else / model-convert test | Python diff of predictions | `LightGBM/tests/cpp_tests/test.py` | driven by `.ci/test.sh` |

## C++ Unit Tests (GoogleTest)

**Location & files:** `LightGBM/tests/cpp_tests/`
- `test_main.cpp` — `main()` calling `testing::InitGoogleTest` + `RUN_ALL_TESTS()`; sets
  `gtest_death_test_style = "threadsafe"`.
- `test_array_args.cpp`, `test_arrow.cpp`, `test_byte_buffer.cpp`, `test_chunked_array.cpp`,
  `test_common.cpp`, `test_serialize.cpp`, `test_single_row.cpp`, `test_stream.cpp`.
- `testutils.cpp` / `testutils.h` — shared C++ test helpers (dataset/stream construction).
- `train.conf`, `predict.conf` — config fixtures used by the if-else and consistency tests.

**Build & run** (from `LightGBM/.ci/test.sh`, `TASK == "cpp-tests"`):
```bash
cmake -B build -S . -DBUILD_CPP_TEST=ON -DUSE_DEBUG=ON
cmake --build build --target testlightgbm -j4
./testlightgbm                         # runs all gtest cases
```
GoogleTest is located via `find_package(GTest CONFIG)` and, if absent, fetched with
`FetchContent` from `https://github.com/google/googletest.git` (`CMakeLists.txt:618-652`). The
test executable `testlightgbm` links `lightgbm_objs` + `lightgbm_capi_objs` + `GTest::GTest`.
Optional sanitizer build: `METHOD=with-sanitizers` adds `-DUSE_SANITIZER=ON` /
`-DENABLED_SANITIZERS=...`.

**Test structure** (from `tests/cpp_tests/test_array_args.cpp`):
```cpp
#include <gtest/gtest.h>
#include <LightGBM/meta.h>
#include <LightGBM/utils/array_args.h>

using LightGBM::data_size_t;
using LightGBM::score_t;

TEST(Partition, JustWorks) {
  std::vector<score_t> gradients({0.5f, 5.0f, 1.0f, 2.0f, 2.0f});
  ...
  EXPECT_EQ(gradients[middle_begin + 1], gradients[middle_end - 1]);
  EXPECT_GT(gradients[0], gradients[middle_begin + 1]);
}
```
- One `TEST(Suite, Case)` per behavior; suites group by component (`Partition`, etc.).
- Assertions: `EXPECT_EQ`, `EXPECT_GT`, `EXPECT_NEAR`, `ASSERT_*`. Edge cases enumerated
  explicitly (`Empty`, `PartitionOneElement`, `AllEqual`).
- These are narrow unit tests over utility/IO/serialization code — **not** full end-to-end
  training. Coverage of the learning algorithms themselves lives mostly in the Python suite.

## Python Tests (pytest)

**Location:** `LightGBM/tests/python_package_test/`
- `test_engine.py` — largest; trains/predicts via `lgb.train`/`lgb.cv` across objectives,
  metrics, callbacks, early stopping, feature importance, model save/load.
- `test_basic.py` — `Dataset`/`Booster` low-level (`basic.py`) behavior.
- `test_sklearn.py` — scikit-learn estimator wrappers.
- `test_consistency.py` — **cross-implementation consistency** (see Oracle section).
- `test_arrow.py`, `test_callback.py`, `test_dask.py`, `test_dual.py` (CPU vs GPU),
  `test_plotting.py`, `test_utilities.py`.
- `utils.py` — shared dataset loaders & data generators.
- `conftest.py` — fixtures.

**Fixtures** (`tests/python_package_test/conftest.py`):
```python
@pytest.fixture(scope="function")
def rng():
    return np.random.default_rng()

@pytest.fixture(scope="function")
def rng_fixed_seed():
    return np.random.default_rng(seed=42)

@pytest.fixture(scope="function")
def missing_module_cffi(monkeypatch):
    monkeypatch.setattr(lightgbm.compat, "CFFI_INSTALLED", False)
    ...
```

**Shared data helpers** (`tests/python_package_test/utils.py`): `@lru_cache`-wrapped sklearn
loaders (`load_breast_cancer`, `load_digits`, `load_iris`, `load_linnerud`) plus a custom
`make_ranking(...)` generator for learning-to-rank datasets. `SERIALIZERS = ["pickle",
"joblib", "cloudpickle"]` parametrizes serialization round-trips.

**Common patterns:**
- Parametrization is heavy: `@pytest.mark.parametrize("task", tasks)`,
  `@pytest.mark.parametrize("output", data_output)`, `@pytest.mark.parametrize("serializer", SERIALIZERS)`.
- Optional-dependency gating: `@pytest.mark.skipif(not GRAPHVIZ_INSTALLED, ...)`,
  `not MATPLOTLIB_INSTALLED`, etc. (flags from `lightgbm.compat`).
- Numerical assertions use `np.testing.assert_allclose(...)` (tolerance-based), never bare `==`,
  for predictions/scores.

**Run** (`LightGBM/.ci/test.sh`, `bdist`/`sdist` tasks):
```bash
pytest ./tests                      # full suite after wheel install (bdist)
pytest ./tests/python_package_test  # sdist task
```
There are also `.ci/test-python-latest.sh` and `.ci/test-python-oldest.sh` for dependency
version matrices. Conda env requirements: `.ci/conda-envs/ci-core.txt` (and `-py37.txt`,
`-py38.txt`).

## Distributed Tests

**Location:** `LightGBM/tests/distributed/`
- `_test_distributed.py` — spins up multiple LightGBM processes over localhost sockets to test
  distributed/parallel training. Uses `ThreadPoolExecutor`, finds random open ports
  (`_find_random_open_port`), writes config via `_write_dict`, generates data with
  `make_blobs`/`make_regression`.
- `conftest.py` — adds `--execfile` option (path to the compiled `lightgbm` CLI binary, default
  `<repo>/lightgbm`); the `executable` fixture exposes it. These tests exercise the **CLI
  binary**, not the Python API.

## R Tests (testthat)

**Location:** `LightGBM/R-package/tests/testthat/`
- Entry: `R-package/tests/testthat.R`; helpers in `helper.R`.
- Test files mirror R API surface: `test_basic.R`, `test_Predictor.R`, `test_dataset.R`,
  `test_lgb.Booster.R`, `test_custom_objective.R`, `test_learning_to_rank.R`, `test_metrics.R`,
  `test_multithreading.R`, `test_parameters.R`, `test_weighted_loss.R`,
  `test_lgb.importance.R`, `test_lgb.interprete.R`, `test_lgb.model.dt.tree.R`,
  `test_lgb.convert_with_rules.R`, `test_lgb.plot.*.R`, `test_utils.R`.
- Run via `LightGBM/.ci/test-r-package.sh` → `R CMD check` (`.ci/run-r-cmd-check.sh`). A valgrind
  variant exists (`.ci/test-r-package-valgrind.sh`, workflow `r_valgrind.yml`).

## CI Configuration

**GitHub Actions** (`LightGBM/.github/workflows/`):
- `python_package.yml` — builds wheels/sdist and runs pytest on linux (incl. aarch64 in
  manylinux docker), macOS, across Python versions. Triggers on push/PR to `master`.
- `r_package.yml`, `r_configure.yml`, `r_valgrind.yml` — R package check matrix + valgrind.
- `cuda.yml` — GPU/CUDA builds and tests.
- `static_analysis.yml` — runs the `lint` / `check-docs` tasks.
- `linkchecker.yml`, `optional_checks.yml`, plus housekeeping (`lock.yml`, `no_response.yml`,
  `release_drafter.yml`, `triggering_comments.yml`).

**Azure Pipelines:** `LightGBM/.vsts-ci.yml` (primary multi-platform matrix). **AppVeyor:**
`LightGBM/.appveyor.yml` (Windows).

**Central dispatcher:** `LightGBM/.ci/test.sh` is the single entry that all CI calls with a
`TASK` env var. Recognized tasks include: `cpp-tests`, `if-else`, `swig`, `lint`, `check-docs`,
`check-links`, `sdist`, `bdist`, `gpu`, `r-package`. Supporting scripts: `.ci/setup.sh`,
`.ci/test-python-latest.sh`, `.ci/test-python-oldest.sh`, `.ci/test-windows.ps1`.

**Docs/parameter consistency gate** (`.ci/test.sh`, `TASK == "check-docs"`): re-runs
`.ci/parameter-generator.py` and `diff`s the regenerated `docs/Parameters.rst` and
`src/io/config_auto.cpp` against committed versions — fails if `config.h` drifts from generated
artifacts.

## Conformance Oracles for the Rust Port

These are the highest-value validation targets when porting. A Rust impl should reproduce them
bit-for-bit (or within the existing `assert_allclose` tolerances).

1. **`tests/python_package_test/test_consistency.py` (PRIMARY ORACLE).** Cross-checks three
   prediction paths against each other for binary/regression/etc. example datasets:
   - `FileLoader` reads `examples/<task>/train.conf` to mirror exact training parameters.
   - `train_predict_check` asserts the Python in-memory prediction equals the **C++ CLI
     prediction** (`cpp_pred = gbm.predict(X_test_fn)`) AND a fresh sklearn-API prediction:
     ```python
     np.testing.assert_allclose(y_pred, cpp_pred)
     np.testing.assert_allclose(y_pred, sk_pred)
     ```
   - `load_cpp_result()` loads `LightGBM_predict_result.txt` produced by the CLI.
   A Rust port can be dropped in as the engine and must keep these `assert_allclose`s green.

2. **`tests/cpp_tests/test.py` + the `if-else` task** (`.ci/test.sh`). The CLI converts a trained
   model to generated C++ if-else code (`convert_model=...gbdt_prediction.cpp`), then compares
   `origin.pred` vs `ifelse.pred` with `np.testing.assert_allclose`. This validates that
   serialized model → prediction is path-independent — a strong invariant for a port.

3. **`tests/cpp_tests/test_serialize.cpp` + `test_single_row.cpp` + `test_stream.cpp`.**
   Model/dataset serialization round-trips and single-row prediction equivalence — directly
   reusable to validate the port's I/O and prediction kernels against the reference's
   byte/score outputs.

4. **Example datasets + `.conf` files** under `LightGBM/examples/` (`binary_classification/`,
   `regression/`, `lambdarank/`, `multiclass_classification/`, `xendcg/`) and
   `LightGBM/tests/data/categorical.data`. These provide fixed inputs + canonical configs; the
   reference CLI can generate golden `*.pred` outputs that the Rust port must match.

5. **Reference binary as golden generator.** The compiled reference `lightgbm` CLI (built from
   this tree) is the authoritative oracle: train on an example config, save the model + predicted
   scores, then diff the Rust port's model text and predictions against those golden files using
   the same `assert_allclose` tolerances the suite already uses.

## Coverage Notes

- No project-wide C++ coverage threshold is enforced; `cpp_tests` cover utilities, IO,
  serialization, single-row/stream prediction — **not** end-to-end training accuracy.
- End-to-end algorithmic behavior is validated indirectly through the Python (`test_engine.py`)
  and consistency suites and the R testthat suite.
- Numerical comparisons throughout use tolerance-based `assert_allclose` / `EXPECT_NEAR`, not
  exact equality — the port should target matching within those tolerances, accounting for
  float vs double (`score_t`/`label_t` typedefs in `include/LightGBM/meta.h`) and
  reduction-order differences in parallel sums.

---

*Testing analysis: 2026-06-05*
