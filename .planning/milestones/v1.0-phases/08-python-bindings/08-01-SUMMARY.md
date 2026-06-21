---
phase: 08-python-bindings
plan: 01
subsystem: api
tags: [rust-facade, binning, bin-mapper, custom-metric, feval, refit, model-text, oracle-parity]

# Dependency graph
requires:
  - phase: 02-dataset-binning
    provides: bit-exact BinMapper (find_bin_from_column / find_bin_categorical / value_to_bin)
  - phase: 03-tree-model-and-predict
    provides: GbdtModel + model_text save/load + feature_importance + refit_one_tree
  - phase: 06-gbdt-spine-core-objectives-metrics
    provides: lgbm facade (train / train_custom / train_with_valid / Booster / DenseCorpus)
provides:
  - "RawCorpus + build_feature_columns_from_raw + train_raw (D-02 raw→bin→train bridge)"
  - "Booster facade methods: predict (batch), predict_raw_batch, feature_importance_split, feature_importance_gain, refit, model_to_string, save_model, model_from_string"
  - "custom-metric (feval) eval-history hook: EvalMetric::Custom + train_custom_with_metric (the upstream symbol 08-06 consumes)"
  - "Rust oracle test: raw→bin→train bit-exact to the identity path (cpp-golden SKIP-marked)"
affects: [08-02-pyo3-scaffold, 08-04-booster-pyclass, 08-05-marshal, 08-06-callbacks-feval]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Route A binning: per-column BinMapper (find_bin_from_column / find_bin_categorical) + value_to_bin, mirroring the identity FeatureColumn construction shape"
    - "Boxed-closure eval metric (EvalMetric::Custom) feeding the existing eval-history loop with zero loop-structure changes"
    - "Column-based training driver (train_inner_columns_full) shared by identity and raw paths — binning happens BEFORE the driver"

key-files:
  created:
    - crates/oracle-harness/tests/raw_bin_train_parity.rs
    - .planning/phases/08-python-bindings/deferred-items.md
  modified:
    - crates/lgbm/src/booster.rs
    - crates/lgbm/src/lib.rs
    - crates/lgbm/src/error.rs

key-decisions:
  - "feval route: added EvalMetric::Custom(Box<dyn Fn(&[f64],&[f32]) -> (String,f64,bool)>) so the EXISTING eval-history loop records the custom metric with ZERO loop changes (the plan's preferred route, not the train_inner threading fallback)"
  - "custom-metric name resolved lazily on the first eval (placeholder 'custom' overwritten with the closure-supplied name) — the history-key setup runs before any scores exist"
  - "feature_infos formats integral bounds as integers (preserving the identity-path capture bytes) and reals with their value (raw path)"
  - "facade-owned LgbmError::Io (save_model) + LgbmError::CustomMetric (non-finite feval) — neither maps cleanly onto a faithful-mirror layer error"

patterns-established:
  - "raw→bin→train bridge: build_feature_columns_from_raw drives every FeatureColumn bin-layout field from a per-column BinMapper, reusing offset_for_most_freq_bin (the single offset source, D-09)"
  - "thin facade delegation: each new Booster method delegates one layer down (GbdtModel / model_text) with no new algorithm"

requirements-completed: [PYB-01, PYB-04]

# Metrics
duration: 7min
completed: 2026-06-07
---

# Phase 8 Plan 01: Rust Facade Enabling Slice (D-02 raw→bin→train + Booster methods + feval hook) Summary

**The `lgbm` facade now trains from RAW (arbitrary-value) features via the bit-exact BinMapper, exposes batch predict / feature importance / refit / C++-compatible model text I/O as Booster methods, and accepts a custom-metric (feval) closure that feeds the SAME eval-history loop — the Rust-side slice every Phase-8 Python entry depends on.**

## Performance

- **Duration:** ~7 min
- **Tasks:** 3 completed (all `type=auto`, all `tdd=true` for tasks 1–2)
- **Files modified:** 3 modified + 2 created

## Accomplishments

### Task 1 — RawCorpus + raw→bin→train bridge (D-02) — commit `7a2fa3a`
- `RawCorpus` struct: row-major raw f64 features + f32 labels + `categorical_features: Vec<usize>` + binning/training `Config` (defaults from `Config::default()`).
- `build_feature_columns_from_raw`: mirrors the EXACT `FeatureColumn { .. }` shape of `build_feature_columns` but drives every bin-layout field from a per-column `BinMapper` (Route A): `bins[row] = mapper.value_to_bin(raw[row])`, `num_bin = mapper.num_bin_`, `default_bin = mapper.default_bin_`, `most_freq_bin = mapper.most_freq_bin_`, `missing_type = mapper.missing_type_`, `bin_upper_bound = mapper.bin_upper_bound_.clone()`, `offset = offset_for_most_freq_bin(most_freq_bin)` (the SAME authoritative helper). Numeric columns → `find_bin_from_column`; categorical columns → `find_bin_categorical`.
- Shape validation (empty / labels≠num_data / ragged row / out-of-range categorical index) → `LgbmError::InvalidCorpus` BEFORE any binning (T-08-01-01, Security V5).
- `train_raw(config, corpus)` builds the columns then calls the SAME consumer as `train` via a new `train_inner_columns` entry.
- Refactor: `train_inner_full` → `train_inner_columns_full` (a column-based driver taking the raw feature rows + labels + pre-built columns). The identity path (`build_feature_columns` → driver) is preserved byte-for-byte; the parity harness depends on it.
- 4 unit tests: identity-bin equivalence (bins + offset + num_bin match), raw==identity train bit-exact leaf values, real-value train + monotone order, shape validation.

### Task 2 — Booster facade methods + custom-metric (feval) hook (PYB-04) — commit `896b4bc`
- New thin `Booster` methods (each one delegation layer down, no new algorithm): `predict` (batch), `predict_raw_batch`, `feature_importance_split`, `feature_importance_gain`, `refit`, `model_to_string`, `save_model`, `model_from_string`.
- `model_from_string` parses untrusted text via `lgbm_model::model_text::load` (`ModelError` → `LgbmError::Model`, T-08-01-03), carries the loaded objective so `predict` applies the right transform.
- custom-metric (feval) hook: `EvalMetric::Custom(CustomMetricClosure)` where `CustomMetricClosure = Box<dyn Fn(&[f64],&[f32]) -> (String,f64,bool)>` (mirroring C++ `_EvalFunctionWrapper`). It feeds the EXISTING eval-history loop with zero loop-structure changes; the name is resolved lazily from the closure; a non-finite value → `LgbmError::CustomMetric` (T-08-01-04).
- `train_custom_with_metric(config, corpus, obj, Option<feval>)` builds `metrics = [EvalMetric::Custom(feval)]` when supplied (else the existing `[l2]` default); `train_custom` delegates to it with `None` so the existing signature still works and nothing downstream breaks. This is the upstream symbol 08-06's Python `feval` marshalling consumes.
- 7 unit tests incl.: batch-predict == per-row (bit-exact), importance delegation, model-text round-trip predicts bit-exact, refit changes leaves, garbage model rejected (`LgbmError::Model`), **custom feval bit-matches the built-in `Metric::L2` value-for-value** (the checker-required acceptance), NaN feval → typed error.

### Task 3 — raw→bin→train parity oracle — commit `c2c5e47`
- `crates/oracle-harness/tests/raw_bin_train_parity.rs`:
  - `raw_bin_train_matches_identity_bin`: trains the integer-valued spine via both `lgbm::train` (DenseCorpus identity) and `lgbm::train_raw` (RawCorpus BinMapper); asserts all tree leaf values BIT-EXACT via `compare_exact_f64_bits`, plus a one-ULP teeth self-check on the comparator.
  - `raw_bin_train_matches_cpp_golden`: real-value RawCorpus under pinned deterministic settings (`force_row_wise=true`, `num_threads=1`, fixed seed, `min_data_in_bin=1`); SKIP-passes with a printed `golden absent` marker (no reusable real-value C++ golden yet) while still exercising train+predict end-to-end.
- `LightGBM/` never git-added.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] `min_data_in_bin` must be 1 for the trivial-case equivalence**
- **Found during:** Task 1 (the first identity-equivalence test failed: the BinMapper merged 6 distinct integer values into 4 bins).
- **Issue:** the default `min_data_in_bin = 3` makes the BinMapper merge adjacent integer bins, so raw≠identity on the "trivial" case unless each distinct value gets its own bin.
- **Fix:** the equivalence tests (and the oracle test) pin `min_data_in_bin = 1`. This is the correct identity-binning precondition, documented in-test; no production-path change.
- **Files modified:** crates/lgbm/src/booster.rs (test config), crates/oracle-harness/tests/raw_bin_train_parity.rs
- **Commit:** 7a2fa3a (test), c2c5e47 (oracle)

**2. [Rule 2 - Missing functionality] facade-owned error variants for the new boundaries**
- **Issue:** the plan's threat model requires typed errors for `save_model` I/O and a non-finite feval value, but no existing layer error fits (ModelError is model-text-only; MetricError has no InvalidValue variant — and adding one to a faithful-mirror crate would be inappropriate).
- **Fix:** added facade-owned `LgbmError::Io { detail }` (save_model) and `LgbmError::CustomMetric { detail }` (non-finite feval). Both are facade-boundary concerns, the correct home.
- **Files modified:** crates/lgbm/src/error.rs
- **Commit:** 896b4bc

## Deferred Issues

**DEF-08-OOS-01 — pre-existing GOSS parity failure (NOT introduced by 08-01).** `oracle-harness::boosting_parity::goss_parity_matrix` fails at tree 11 (a deep f64 split-gain knife-edge, same class as DEF-07-02). Verified PRE-EXISTING by reverting `crates/lgbm/src/{booster,error,lib}.rs` to commit `c13d380` (before any 08-01 work) and reproducing the IDENTICAL failure; it originates in prior-session commits to `learner.rs`/`boosting_parity.rs`, not this plan. Plan 08-01 touches none of the GOSS / learner / boosting-loop code. Logged to `.planning/phases/08-python-bindings/deferred-items.md` (SCOPE BOUNDARY — left untouched; belongs in a Phase-7-style learner FP-trace fix).

## Threat Surface

All threat-register mitigations applied:
- T-08-01-01 (shape) → `build_feature_columns_from_raw` validates before binning.
- T-08-01-02 (DoS) → every fallible facade method returns `Result<_, LgbmError>`; no `unwrap`/`panic!` on user input.
- T-08-01-03 (untrusted model text) → `model_from_string` via validated `model_text::load`.
- T-08-01-04 (feval NaN/invalid) → non-finite value → `LgbmError::CustomMetric`, never a panic.
- T-08-01-SC → no new external crates added (facade + oracle test only).

No NEW security surface beyond the plan's threat model (the new boundaries — raw corpus, feval closure, model text — are exactly the registered ones).

## Verification

- `cargo test -p lgbm` — 41 passed / 0 failed (raw-bridge + facade-method + custom-metric eval-history unit tests).
- `cargo test -p oracle-harness --test raw_bin_train_parity` — 2 passed (identity bit-exact; cpp-golden SKIP-marked).
- `cargo clippy -p lgbm --tests` / `cargo clippy -p oracle-harness --tests` — clean on edited code (only pre-existing `lgbm-dataset` clippy warnings remain, out of scope).
- `cargo test --workspace` — GREEN except the pre-existing `goss_parity_matrix` (DEF-08-OOS-01, not introduced here).

## Known Stubs

None — every new method is wired to a real lower-layer implementation; the cpp-golden oracle SKIP is an honest deferred-capture marker (prints a reason, still trains+predicts), not a stub.

## Self-Check: PASSED
