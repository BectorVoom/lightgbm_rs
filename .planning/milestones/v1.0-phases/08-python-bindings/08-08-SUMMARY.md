---
phase: 08-python-bindings
plan: 08
subsystem: api
tags: [pyo3, pickle, model-text, persistence, cross-format-parity, sklearn]

# Dependency graph
requires:
  - phase: 08-python-bindings
    provides: "08-01 facade Booster::model_to_string / save_model / model_from_string (C++-compatible Phase-3 model text)"
  - phase: 08-python-bindings
    provides: "08-02 #[pyclass] Booster + LgbmError->PyErr taxonomy; engine _TrainedBooster wrapper; sklearn LGBMModel estimators"
  - phase: 03-tree-model-and-predict
    provides: "lgbm-model model_text save/load (the validated, byte-stable v4 text loader)"
provides:
  - "PyO3 Booster persistence #[pymethods]: model_to_string / save_model / from_model_string / from_model_file"
  - "PyO3 Booster pickle dunders: __getstate__ / __setstate__ / __reduce__ over the model string"
  - "_TrainedBooster persistence delegation + pickle (model_to_string/save_model + __getstate__/__setstate__)"
  - "LGBMModel sklearn-pipeline pickle (__getstate__/__setstate__)"
  - "cross-format persistence parity test: a real-lightgbm-4.6 text model loads in lightgbm_rs within ~1e-6 (D-10)"
affects: []

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Pickle a no-#[new] #[pyclass] via __reduce__ -> (from_model_string, (model_text,)): reconstruction routes through the validated text loader so unpickle is a parse, never an unchecked field-set"
    - "Persistence is thin delegation: every PyO3 persistence method calls one 08-01 facade method down; no new serialization code in the binding"
    - "_TrainedBooster / LGBMModel pickle over the booster's model string (not raw object state), mirroring the official package's handle-as-text persistence"

key-files:
  created:
    - crates/lgbm-python/python/tests/test_persistence.py
  modified:
    - crates/lgbm-python/src/booster.rs
    - crates/lgbm-python/python/lightgbm_rs/engine.py
    - crates/lgbm-python/python/lightgbm_rs/sklearn.py

key-decisions:
  - "from_model_string / from_model_file are #[staticmethod] constructors (the #[pyclass] has no #[new]) rather than extending a constructor with model_str=/model_file= kwargs — keeps the train/predict construction seam untouched and mirrors a clear factory surface"
  - "__reduce__ returns (from_model_string, (state,)) so unpickle FULLY reconstructs via the validated loader; __getstate__/__setstate__ are also implemented for the explicit D-10 contract but pickle uses the reduce callable+args path (no double-parse)"
  - "_TrainedBooster.__getstate__ serializes {model_str, params, best_iteration, best_score}; the core booster is rebuilt from its model text on unpickle (the _core.Booster cannot carry Python attrs)"
  - "LGBMModel pickle uses the default __dict__ (its _booster is a _TrainedBooster that pickles via its own model-string state) — no estimator-level model-text plumbing needed"

patterns-established:
  - "PyO3 pickle for a loader-backed pyclass: __reduce__ -> (validated_factory, (serialized_text,)) makes unpickle a parse through the same untrusted-text loader as model_str= (Security V5)"
  - "Persistence test shape: text round-trip (byte-identical) + file save/load (byte-identical) + cross-format load vs real lightgbm 4.6 (atol 1e-6) + pickle (Booster + estimator) + malformed-raises"

requirements-completed: [PYB-01]

# Metrics
duration: 12min
completed: 2026-06-08
---

# Phase 8 Plan 08: Model Persistence + Pickle (D-10) Summary

**The Python surface now saves/loads the Phase-3 C++-compatible model text (`save_model`/`model_to_string`/`from_model_string`/`from_model_file`) and pickles a `Booster` and a fitted `LGBMClassifier` over the model string — and a model trained by real `lightgbm` 4.6, saved to text, loads in `lightgbm_rs` and predicts within ~1e-6 (the D-10 cross-format contract). This closes Phase 8.**

## Performance

- **Duration:** ~12 min
- **Tasks:** 2 completed (both `type=auto`)
- **Files modified:** 3 modified + 1 created

## Accomplishments

- **PyO3 Booster persistence #[pymethods] (Task 1):** `model_to_string()` and `save_model(filename)` delegate to the 08-01 facade text I/O; `from_model_string(model_str)` and `from_model_file(model_file)` are `#[staticmethod]` constructors that parse untrusted text through the validated `lgbm-model` loader (file-read + parse failures map to a typed `LightGBMError`, never a panic — T-08-08-01/02).
- **PyO3 pickle support (Task 1):** `__getstate__` returns the model string, `__setstate__` rebuilds `inner` from it, and `__reduce__ -> (from_model_string, (model_text,))` reconstructs the `#[pyclass]` (which has no `#[new]`) through the validated loader so unpickle is a parse — D-10 pickle satisfied for the core `Booster`.
- **Python-layer persistence + pickle (Task 2):** `_TrainedBooster` gained `model_to_string`/`save_model` delegation and `__getstate__`/`__setstate__` (over `{model_str, params, best_iteration, best_score}`); `LGBMModel` gained `__getstate__`/`__setstate__` so fitted estimators pickle in sklearn pipelines.
- **Cross-format persistence parity test (Task 2):** `test_persistence.py` (7 tests) proves text round-trip + file save/load are byte-identical, a **real-lightgbm-4.6** model cross-loads in `lightgbm_rs` within `atol=1e-6` (the D-10 C++-compatibility proof, test RUNS not skipped), a `Booster` and a fitted `LGBMClassifier` survive a `pickle` round-trip with identical predictions, and malformed model text / a missing model file raise typed exceptions.

## Task Commits

1. **Task 1: Persistence #[pymethods] + pickle dunders (D-10)** — `b1caa69` (feat)
2. **Task 2: Cross-format persistence parity + pickle pytest (D-10)** — `16a699e` (test)

**Plan metadata:** (final docs commit)

## Files Created/Modified

- `crates/lgbm-python/src/booster.rs` — added `model_to_string`/`save_model`/`from_model_string`/`from_model_file` `#[pymethods]` + `__getstate__`/`__setstate__`/`__reduce__` pickle dunders (all thin delegations to the 08-01 facade; no panic across FFI).
- `crates/lgbm-python/python/lightgbm_rs/engine.py` — `_TrainedBooster.model_to_string`/`save_model` delegation + `__getstate__`/`__setstate__` (pickle over the model string + metadata).
- `crates/lgbm-python/python/lightgbm_rs/sklearn.py` — `LGBMModel.__getstate__`/`__setstate__` for sklearn-pipeline pickling.
- `crates/lgbm-python/python/tests/test_persistence.py` — text round-trip, file save/load, cross-format load (vs real lightgbm 4.6, atol 1e-6), Booster + estimator pickle, malformed-text/file raises.

## Decisions Made

- **Static-method constructors over a kwargs constructor.** `from_model_string`/`from_model_file` are `#[staticmethod]` factories rather than extending a `#[new]` with `model_str=`/`model_file=` — the `#[pyclass]` has no `#[new]` (it is produced by `train`), and a clear factory surface keeps the train/predict seam untouched.
- **`__reduce__`-driven pickle.** `__reduce__` returns `(from_model_string, (model_text,))` so unpickle fully reconstructs through the validated loader (no double-parse). `__getstate__`/`__setstate__` are also implemented for the explicit D-10 getstate/setstate contract.
- **Estimator pickle via default `__dict__`.** `LGBMModel._booster` is a `_TrainedBooster` that pickles over its own model-string state, so the estimator needs no model-text plumbing of its own.

## Deviations from Plan

None — plan executed exactly as written. (The plan's `files_modified` listed `python/lightgbm_rs/__init__.py`, but no change was needed there: `Booster` is already re-exported and the new persistence methods live on it. `engine.py` was modified instead, to give the public `_TrainedBooster` wrapper the same persistence + pickle surface — within the plan's Task-2 scope of "delegate to the booster string".)

## Issues Encountered

None. The cross-format load against real `lightgbm` 4.6 matched within `atol=1e-6` on the first run, confirming the Phase-3 text format is read-compatible with the C++ reference writer.

## Threat Surface

All threat-register mitigations applied:
- **T-08-08-01 (untrusted model text parse)** → `from_model_string` / `__setstate__` / `__reduce__` all route through the validated `lgbm-model` loader inside the 08-01 facade; malformed text → typed `LightGBMError`, never raw-text indexing. Proven by `test_malformed_model_text_raises`.
- **T-08-08-02 (malformed model_file path)** → `from_model_file` maps a file-read failure to a typed `LightGBMError`; no panic. Proven by `test_malformed_model_file_raises`.
- **T-08-08-03 (pickle of model string)** → ACCEPTED per the register (pickle is not a security boundary); documented in the `__getstate__` docstring (never unpickle from an untrusted party).
- **T-08-08-SC** → no NEW crates added (uses already-pinned/audited deps).

No new security surface beyond the plan's threat model.

## Known Stubs

None — every persistence method is wired to a real lower-layer implementation; `plot_tree` remains the only `NotImplementedError` surface (08-07, explicitly out-of-scope for this plan).

## User Setup Required

None - no external service configuration required.

## Verification

- `cargo build -p lgbm-python` — compiles with persistence methods + pickle dunders.
- `cargo clippy -p lgbm-python` — clean (the only clippy warnings are pre-existing in the `lgbm-boosting` dependency, out of scope).
- No `unwrap`/`panic!`/`expect` in `crates/lgbm-python/src/booster.rs`.
- `maturin develop` — succeeds (editable install in the project `.venv`).
- `python -m pytest python/tests/test_persistence.py` — **7 passed** (text round-trip, file save/load, cross-format load vs real lightgbm 4.6 @atol=1e-6, Booster pickle, estimator pickle, malformed-text-raises, malformed-file-raises).
- `python -m pytest python/tests/` — **53 passed / 3 skipped** (the prior 46 passed / 3 skipped preserved + the 7 new persistence tests; no regression).
- `LightGBM/` never git-added; `.venv` / `target` / `_core*.so` not staged.

## Next Phase Readiness

- Phase 8 (python-bindings) is COMPLETE (08-01..08-08). The Python surface mirrors the official `lightgbm` low-level + sklearn API for the in-scope set, including D-10 persistence + pickle and cross-format interop with real lightgbm 4.6.
- Note for the phase verifier: 08-07's `plot_tree` `NotImplementedError` is now technically satisfiable (model text is exposed), but wiring it fully was out-of-scope here and remains deferred. The pre-existing `goss_parity_matrix` failure (DEF-08-OOS-01) is unrelated to the Python bindings.

## Self-Check: PASSED

All created/modified files exist on disk and both task commits (`b1caa69`, `16a699e`) are present in git history.
