---
phase: 08-python-bindings
plan: 05
subsystem: api
tags: [pyo3, params-coercion, config, d-06, d-07, d-08, ab-parity, error-taxonomy]

# Dependency graph
requires:
  - phase: 08-python-bindings
    plan: 02
    provides: "lgbm-python cdylib + train/Dataset #[pyfn] + LgbmErrorWrap error taxonomy (the basic params coercion this plan replaces)"
provides:
  - "crates/lgbm-python/src/params.rs — the single Python params dict -> Config seam (D-06/07/08)"
  - "coerce_value / coerce_params_dict (D-08): bool-before-int, float shortest round-trip, list/tuple comma-join, nested [a,b] form for interaction_constraints"
  - "reject_unimplemented (D-07): raises PyValueError for OUT_OF_SCOPE_PARAMS (referenced from lgbm_core::config::scope) + device_type=gpu/cuda; unknown typos pass through (D-06 warn-not-fatal)"
  - "build_config / build_config_with_overrides: coerce -> gate -> Config::from_params (full alias + CHECK validation)"
  - "train #[pyfn] now routes the WHOLE official params surface through the D-06/07/08 pipeline"
  - "python/tests/test_params.py — coercion / list-join / unimplemented-raises / unknown-warn / alias pytest (11 tests)"
affects: [08-06-callbacks, 08-07-sklearn, 08-08-persistence]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "extension-module moved to maturin-only (pyproject features); cargo test links libpython via pyo3/auto-initialize dev-dep + rlib crate-type, so Rust units can drive a live PyDict"
    - "D-07 gate references the lgbm_core scope const (single source of truth) instead of re-typing the out-of-scope set"
    - "num_boost_round injected as a canonical num_iterations OVERRIDE after coercion + the D-07 gate but before from_params (alias precedence preserved)"

key-files:
  created:
    - crates/lgbm-python/src/params.rs
    - crates/lgbm-python/python/tests/test_params.py
  modified:
    - crates/lgbm-python/src/lib.rs
    - crates/lgbm-python/src/booster.rs
    - crates/lgbm-python/Cargo.toml

key-decisions:
  - "extension-module is no longer a default Cargo feature: maturin enables it via [tool.maturin].features, and a pyo3/auto-initialize dev-dep + an added rlib crate-type let `cargo test` link libpython and run the Rust coercion units (the extension-module feature makes a standalone test binary fail to link the Py* symbols). The wheel build is unaffected (verified by maturin develop + import + 26-test pytest)."
  - "Dataset has NO params surface in the current architecture (binning happens inside the facade train_raw from the Config), so build_config wiring applies to train only; the plan's 'Dataset binning-params path' has no code to route. Documented as a deviation, not a stub."
  - "float coercion uses Rust `{}` (shortest round-trip), which the C++ Common::Atof re-parses to the identical f64 — matches the official package's f-string repr for finite floats."

requirements: [PYB-01]
metrics:
  duration: ~10 min
  completed: 2026-06-08
  tasks: 2
  files_created: 2
  files_modified: 3
status: complete
---

# Phase 8 Plan 5: Full Params-Dict Surface (D-08 coercion + D-07 gate) Summary

The full official `params` dict is now the primary, safe config surface: a Python
typed-value → string coercion layer (D-08) feeds `Config::from_params` (D-06, full
alias + CHECK validation), guarded by a D-07 gate that raises a clear `ValueError`
for recognized-but-unimplemented params while letting truly-unknown typos warn.

## What landed

- **Task 1 — `params.rs` coercion + D-07 gate (commit `fac87af`):**
  `coerce_value` dispatches by Python type — `bool → "true"/"false"` (checked
  BEFORE `int`, since Python bool is an int subclass), `int → decimal`,
  `float → shortest round-trip repr` (C++ `Atof`-faithful), `str → verbatim`,
  `list`/`tuple → comma-join` with nested lists rendered as `[a,b]` chunks (the
  C++ `interaction_constraints` form, e.g. `[[0,1],[2]] → "[0,1],[2]"`).
  `reject_unimplemented` raises `PyValueError` for any alias-resolved key in
  `lgbm_core::config::scope::OUT_OF_SCOPE_PARAMS` (distributed / GPU-OpenCL /
  linear-tree / quantized-grad — referenced, not re-typed) plus
  `device_type=gpu/cuda`; unknown typos pass through. `build_config` /
  `build_config_with_overrides` chain coerce → gate → `Config::from_params`. No
  panic; clippy clean. 5 Rust unit tests green (coercion, list-join, gate,
  full-pipeline, reject) via the `pyo3/auto-initialize` test harness.
- **Task 2 — wire into `train` + pytest (commit `7774914`):** the 08-02 basic
  inline coercion (`python_value_to_string`/`params_dict_to_map`) was removed and
  `train` now calls `params::build_config_with_overrides(params, [("num_iterations",
  num_boost_round)])`, so every train invocation runs the full D-06/07/08
  pipeline with `num_boost_round` keeping official-package precedence over any
  params iteration alias. `python/tests/test_params.py` (11 tests) covers
  int/float/bool coercion, `num_leaves` taking effect, list/tuple join, the four
  unimplemented-param rejections, unknown-typo-warns, and `n_estimators` alias
  resolution.

## Verification

- `cargo test -p lgbm-python` → **5 passed** (params coercion/gate/pipeline units).
- `cargo clippy -p lgbm-python` → clean; no `unwrap`/`expect`/`panic!` in `src/`
  production code (only `#[cfg(test)]` and doc comments).
- `maturin develop` → ✓ (wheel built, `import lightgbm_rs` ✓ — the
  extension-module feature move did NOT break the wheel build).
- `pytest python/tests/` → **26 passed** (15 existing + 11 new `test_params.py`):
  coercion, list-join (`monotone_constraints`/`eval_at`/`interaction_constraints`),
  unimplemented-raises (`device_type=gpu`, `linear_tree`, `num_machines`,
  `use_quantized_grad`), unknown-typo-warns, alias resolves.
- `cargo test --workspace` → 60 passed, 1 pre-existing unrelated failure
  (`oracle-harness::goss_parity_matrix`, see Deviations).
- `LightGBM/` not git-added; `.venv`/`target`/`_core*.so` not committed.

## Deviations from Plan

### Auto-fixed / structural adjustments

**1. [Rule 3 - Blocking] `extension-module` feature moved to maturin-only**
- **Found during:** Task 1 (`cargo test -p lgbm-python` link step).
- **Issue:** with `pyo3/extension-module` enabled, the `cargo test` binary fails
  to link (undefined `Py*` symbols — the feature deliberately leaves libpython
  to the host interpreter, which a standalone test binary lacks). The plan
  requires Rust coercion units.
- **Fix:** removed `extension-module` from the default Cargo deps (maturin still
  enables it via `[tool.maturin].features = ["pyo3/extension-module"]`), added a
  `pyo3` dev-dependency with `auto-initialize` (boots a CPython interpreter for
  tests) and an `rlib` crate-type. Wheel build re-verified via `maturin develop`
  + import + the full 26-test pytest.
- **Files modified:** `crates/lgbm-python/Cargo.toml`.
- **Commit:** `fac87af`.

**2. [Scope clarification] No Dataset params-binning path exists to wire**
- **Found during:** Task 2 (the plan asks to route "the Dataset binning-params
  path" through `build_config`).
- **Issue:** the current `Dataset` constructor takes no `params` dict — binning
  is performed internally by the facade `train_raw` from the `Config` produced at
  train time. There is no separate Dataset params surface to route.
- **Resolution:** `build_config` wiring applies to `train` (the only params entry
  point); `Dataset` is unchanged. Not a stub — there is no data source to wire
  because the architecture bins from the train-time Config.

### Out-of-scope (NOT fixed — SCOPE BOUNDARY)

**`oracle-harness::goss_parity_matrix` fails under `cargo test --workspace`.**
This is the PRE-EXISTING `DEF-08-OOS-01` GOSS bit-exactness divergence (identical
tree-11 leaf-value bits as already documented in this phase's `deferred-items.md`),
in an unrelated crate that contains zero references to `lgbm-python`. This plan
touched only `crates/lgbm-python/`. Left untouched per the scope boundary; already
tracked for a Phase-7-style learner-level FP-trace fix.

## Notes for downstream plans

- `params::build_config` / `build_config_with_overrides` are the canonical
  params→`Config` seam — 08-06 (callbacks), 08-07 (sklearn), 08-08 (persistence)
  should route any new params entry point through them, not re-coerce.
- The D-07 gate auto-tracks `lgbm_core::config::scope::OUT_OF_SCOPE_PARAMS`: when
  a later phase IMPLEMENTS one of those groups, move its names out of
  `OUT_OF_SCOPE_PARAMS` (in `lgbm-core`) and the gate stops rejecting it — no
  change needed in `params.rs`.
- `cargo test -p lgbm-python` now requires the `pyo3/auto-initialize` dev-dep
  (a CPython interpreter on PATH); the wheel build is unchanged.

## Threat Flags

None — no new network endpoints, auth paths, file access, or schema changes at a
trust boundary. The Python params dict → `Config` boundary (T-08-05-01/02/03) is
mitigated exactly as the threat register specifies: D-07 gate (silent-divergence),
`PyResult` coercion + `from_params` CHECK validation (malformed values), explicit
type dispatch with `ValueError`-not-panic for odd Python objects.

## Self-Check: PASSED

- FOUND: `crates/lgbm-python/src/params.rs`
- FOUND: `crates/lgbm-python/python/tests/test_params.py`
- FOUND commit: `fac87af` (Task 1)
- FOUND commit: `7774914` (Task 2)
