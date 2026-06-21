---
phase: 07-parity-completing-variants
plan: 10
subsystem: predict
tags: [treeshap, predict_contrib, shap, prediction-early-stop, lightgbm-4.6, predict-modes]

# Dependency graph
requires:
  - phase: 03-model-format
    provides: the Tree parallel-array structure + predict path (get_leaf/decision/find_in_bitset) + GbdtModel ensemble + model_text::load + the predict.rs driver
  - phase: 07-parity-completing-variants (07-08)
    provides: categorical decision-type handling in the Tree predict path (find_in_bitset / categorical_decision) that TreeSHAP must recurse over
provides:
  - "Tree::expected_value (cover-weighted leaf average) + Tree::predict_contrib (TreeSHAP recursion: ExtendPath/UnwindPath/UnwoundPathSum + PathElement)"
  - "predict.rs GBDT-level predict_contrib_{mat,csr,csc} driver (per-class block num_features+1, base at [num_features])"
  - "GbdtModel::predict_raw_early_stop (per-iteration accumulator + margin hook) + ObjectiveKind::need_accurate_prediction gate + predict_raw_early_stop_mat model-aware driver"
  - "builder predict_contrib / pred_early_stop / pred_early_stop_freq / pred_early_stop_margin setters"
  - "xtask predict-mode-oracle-capture + real lib_lightgbm 4.6 predict-mode goldens (contrib + early_stop, numeric/categorical/multiclass)"
  - "crates/oracle-harness/tests/predict_parity.rs (contrib_* + early_stop_* cells, capture-gated)"
affects: [07-11 (learner constraints), 07-12 (phase wrap-up)]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Per-tree SHAP method on Tree (expected_value + predict_contrib) with the GBDT-level striding/accumulation driver in predict.rs — mirrors the existing predict_raw / predict_mat split"
    - "Flat preallocated PathElement buffer recursed via split_at_mut frames (the C++ `parent_unique_path + unique_depth` pointer-arithmetic over one array, in safe Rust)"
    - "Objective-gated prediction early stop: the mechanism (predict_raw_early_stop) is faithful to PredictRaw; the GATE (need_accurate_prediction) lives in the model-aware driver, mirroring the C++ Predictor ctor"

key-files:
  created:
    - crates/oracle-harness/tests/predict_parity.rs
    - xtask/py/predict_mode_oracle_capture.py
    - crates/oracle-harness/tests/fixtures/predict_modes/{numeric,categorical,multiclass}/{model.txt,X.txt,contrib.txt} (+ numeric/multiclass early_stop.txt)
  modified:
    - crates/lgbm-model/src/tree.rs
    - crates/lgbm-model/src/predict.rs
    - crates/lgbm-model/src/ensemble.rs
    - crates/lgbm-model/src/objective.rs
    - crates/lgbm/src/builder.rs
    - crates/lgbm-core/tests/config_defaults.rs
    - xtask/src/main.rs

key-decisions:
  - "TreeSHAP recursion encoded over a flat preallocated PathElement Vec via split_at_mut frames (safe-Rust equivalent of the C++ pointer-arithmetic), keeping the gate order + arithmetic 1:1"
  - "Prediction early stop is GATED on ObjectiveKind::need_accurate_prediction (only binary/multiclass/ova return false) — the regression early_stop cell only passed once the gate was added (capture-revealed Rule 1)"
  - "predict_contrib output uses f64 end-to-end; parity asserted within ORACLE_TOL at the f32 boundary AND the exact sum+base==raw invariant in f64"

patterns-established:
  - "Capture-gated predict-mode parity cells that SKIP-pass absent the golden and assert both within-ORACLE_TOL parity AND the structural SHAP invariant (sum+base==raw)"

requirements-completed: [PRD-04, PRD-05]

# Metrics
duration: ~12min
completed: 2026-06-07
---

# Phase 7 Plan 10: TreeSHAP feature contributions + prediction early stopping Summary

**TreeSHAP `predict_contrib` (cover-weighted ExpectedValue base + the Lundberg path-weight recursion over numeric/categorical/multiclass trees, sum+base==raw) and objective-gated prediction early stopping (binary 2*|p| / multiclass top1-top2 margin every freq iters), both matching real lib_lightgbm 4.6.**

## Performance

- **Duration:** ~12 min
- **Started:** 2026-06-07T08:44:29Z
- **Completed:** 2026-06-07T08:56:52Z
- **Tasks:** 4 (3 code/test + 1 human-gated capture, satisfied by the provided 4.6.0 venv)
- **Files modified:** 7 modified + 2 source/test files created + 11 golden fixtures

## Accomplishments
- **PRD-04 TreeSHAP** — `Tree::expected_value` (cover-weighted leaf average via `leaf_count`/`internal_count`, tree.cpp:1031-1039) + `Tree::predict_contrib` with the full TreeSHAP recursion (`ExtendPath`/`UnwindPath`/`UnwoundPathSum` + `PathElement`, tree.cpp:868-977) faithfully ported; the GBDT-level `predict_contrib_{mat,csr,csc}` driver lays out the per-class `num_features+1` block with the expected-value base at `[num_features]`, accumulated across the ensemble (gbdt.cpp:640-651). Works over numeric, categorical (`find_in_bitset` via `Decision`), and multiclass (per-class stride) trees. The load-bearing INVARIANT — `sum(per-feature contrib) + base == raw margin` — is asserted in unit tests AND the parity gate.
- **PRD-05 prediction early stopping** — `GbdtModel::predict_raw_early_stop` ports the per-iteration accumulator + the `PredictionEarlyStopInstance` margin check every `freq` iters (gbdt_prediction.cpp:13-32 + prediction_early_stop.cpp): binary `2*|p0| > margin`, multiclass `(top1-top2) > margin`; returns `(score, iters_evaluated)`. The model-aware `predict_raw_early_stop_mat` driver applies the C++ `!NeedAccuratePrediction()` gate.
- **Builder + config** — config params already present (defaults false/10/10.0); added builder setters `predict_contrib`/`pred_early_stop`/`pred_early_stop_freq`/`pred_early_stop_margin` + resolution tests.
- **Capture + parity** — `predict-mode-oracle-capture` xtask + py script trained numeric/categorical/multiclass on real lib_lightgbm 4.6 and dumped contrib + early-stop goldens (byte-idempotent); the new `predict_parity.rs` `contrib_*` (3) + `early_stop_*` (2) cells flip skip→GREEN within ORACLE_TOL with sum+base==raw satisfied.

## Task Commits

1. **Task 1: TreeSHAP predict_contrib + ExpectedValue** - `e8ab949` (feat)
2. **Task 2: prediction early stopping accumulator hook** - `d8cbe7b` (feat)
3. **Task 3: builder setters + predict-mode capture + predict_parity.rs** - `40681ef` (feat)
4. **Task 4: capture goldens + NeedAccuratePrediction gate** - `e3773ae` (test)

## Files Created/Modified
- `crates/lgbm-model/src/tree.rs` — `expected_value`, `data_count`, `max_depth`/`recompute_leaf_depths`, `PathElement`, `extend_path`/`unwind_path`/`unwound_path_sum`, `tree_shap`, `predict_contrib` + unit tests (numeric/categorical SHAP invariant).
- `crates/lgbm-model/src/predict.rs` — `predict_contrib_{mat,csr,csc}` + `predict_raw_early_stop_mat` (objective-gated) drivers + unit tests.
- `crates/lgbm-model/src/ensemble.rs` — `predict_raw_early_stop` + `pred_early_stop_should_stop` (binary/multiclass margin) + unit tests.
- `crates/lgbm-model/src/objective.rs` — `ObjectiveKind::need_accurate_prediction` + test.
- `crates/lgbm/src/builder.rs` — predict-mode setters + test.
- `crates/lgbm-core/tests/config_defaults.rs` — pred_early_stop defaults + from_params resolution test.
- `xtask/src/main.rs`, `xtask/py/predict_mode_oracle_capture.py` — `predict-mode-oracle-capture` subcommand + script.
- `crates/oracle-harness/tests/predict_parity.rs` — new contrib + early_stop cells.
- `crates/oracle-harness/tests/fixtures/predict_modes/*` — real lib_lightgbm 4.6 goldens.

## Decisions Made
- TreeSHAP recursion encoded over a flat preallocated `PathElement` Vec via `split_at_mut` frames (safe-Rust analog of the C++ `parent_unique_path + unique_depth` pointer arithmetic + `std::copy`), preserving the gate order + arithmetic 1:1.
- `max_depth` recomputed from the node structure (`recompute_leaf_depths`) since a parsed model carries no serialized `leaf_depth_` — matches C++ `RecomputeMaxDepth`.
- `predict_contrib` is f64 end-to-end; parity asserted within ORACLE_TOL at the f32 boundary, and the exact `sum+base==raw` invariant asserted in f64.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug, capture-revealed] Prediction early stop must be gated on `NeedAccuratePrediction`**
- **Found during:** Task 4 (running the captured early_stop cells)
- **Issue:** The first implementation applied the binary margin hook for ANY objective. The real-binary `early_stop_numeric` (regression) golden returns the FULL ensemble score, but the Rust path froze at the binary `2*|p|` margin (rust 2.13 vs cpp 2.56). C++ `Predictor` installs the active early-stop instance only when `early_stop && !boosting->NeedAccuratePrediction()` (predictor.hpp:46); the base `ObjectiveFunction::NeedAccuratePrediction` returns `true` (regression/poisson/cross-entropy), so early stop is silently ignored there — only binary/multiclass/ova (and rank) return `false`.
- **Fix:** Added `ObjectiveKind::need_accurate_prediction` (false only for binary/multiclass/multiclassova) + the model-aware `predict_raw_early_stop_mat` driver that forces `freq=0` (disabled) when accurate prediction is needed, returning the full predict. The pure `predict_raw_early_stop` mechanism stays faithful to `PredictRaw` (the gate belongs in the driver, as in C++).
- **Files modified:** crates/lgbm-model/src/objective.rs, crates/lgbm-model/src/predict.rs, crates/oracle-harness/tests/predict_parity.rs
- **Verification:** `early_stop_numeric` (regression, full score) + `early_stop_multiclass` (margin) both GREEN within ORACLE_TOL; unit tests assert the gate (regression ignores, binary fires).
- **Committed in:** `e3773ae` (Task 4 commit)

---

**Total deviations:** 1 auto-fixed (1 bug, capture-revealed)
**Impact on plan:** The gate is required for faithful parity (regression early stop is a no-op in C++); no scope creep. No tolerance weakened, no horizon capped.

## Issues Encountered
- The contrib unit-test tolerance initially used 1e-9 against an f32 raw score, producing a spurious "2.1 != 2.1" failure; loosened to 1e-6 (the actual f32 contract — the f64 sum equals raw to f64 precision, only the f32 raw cast differs).

## User Setup Required
None - the capture used the pre-provisioned `/tmp/lgbm-capture-venv` (lightgbm 4.6.0). The pip lightgbm is a capture-time tool only; `cargo test` does not need it.

## Next Phase Readiness
- PRD-04 + PRD-05 land faithful vs real lib_lightgbm 4.6 (contrib sum+base==raw over numeric/categorical/multiclass; early stop matches binary/multiclass margins + regression no-op).
- `cargo test --workspace` GREEN (0 failed / 13 ignored DEF-07-02 unchanged); learner_parity 15/15, kernel_parity 4/4, predict_parity 5/5; spine + existing predict parity unregressed; `cargo build --workspace --tests` exit 0; clippy clean on the new code. `LightGBM/` never git-added.
- Unblocks 07-11 (learner constraints) / 07-12 (phase wrap-up). DEF-07-02 (objective-side learner knife-edges) untouched.

---
*Phase: 07-parity-completing-variants*
*Completed: 2026-06-07*

## Self-Check: PASSED
- All created files exist on disk (tree.rs, predict.rs, predict_parity.rs, predict_mode_oracle_capture.py, contrib/early_stop goldens, SUMMARY).
- All 4 task commits present in git history (e8ab949, d8cbe7b, 40681ef, e3773ae).
- `cargo test --workspace`: 0 failed / 13 ignored (DEF-07-02 unchanged); predict_parity 5/5, learner_parity 15/15, kernel_parity 4/4; capture byte-idempotent.
