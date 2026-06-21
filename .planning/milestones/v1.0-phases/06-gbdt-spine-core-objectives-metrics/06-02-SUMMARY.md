---
phase: 06-gbdt-spine-core-objectives-metrics
plan: 02
subsystem: boosting
tags: [gbdt, objective, metric, score-updater, boost-from-average, builder, booster, oracle, bit-exact]

# Dependency graph
requires:
  - phase: 06-01-wave-0-foundation
    provides: lgbm-objective/metric/boosting/lgbm scaffolds + error boundaries; Tree::shrinkage/add_bias; SerialTreeLearner::add_prediction_to_score + renew_tree_output seam; boosting_parity Nyquist scaffold + capture stub
  - phase: 05-tree-learner-split-finding
    provides: SerialTreeLearner::train (bit-exact serial tree growth), DataPartition::indices_in_leaf, offset_for_most_freq_bin
  - phase: 03-model-text-predict
    provides: lgbm-model GbdtModel + predict_raw + model_text load/save (%.17g), ObjectiveKind::convert (ConvertOutput)
  - phase: 01-oracle-contract-foundations
    provides: lgbm-core Config (from_params alias+CHECK), K_EPSILON, f32 ScoreT/LabelT contract; oracle-harness comparators
provides:
  - "Objective::Regression{sqrt} training-side enum: get_gradients (f64-subtract->single f32 cast, hess=1.0f32), boost_from_score (ordered f64 label-mean), transform_labels (sqrt Init)"
  - "Metric::{L2,Rmse,L1} enum: eval (ordered f64 LossOnPoint reduction + AverageLoss), factor_to_bigger_better=-1"
  - "lgbm-boosting Gbdt::train_one_iter (verbatim TrainOneIter order) + ScoreUpdater (f64 score_, add_constant + add_tree_train_path scatter)"
  - "SerialTreeLearner::train_returning_partition (exposes the grown DataPartition for the bit-exact score scatter)"
  - "Public lgbm facade: TrainingBuilder (->Config), Booster (best_iteration + eval_history + iter snapshots), train()/predict_row/predict_row_raw"
  - "boosting-oracle-capture xtask + boosting_oracle_capture.py (real lightgbm 4.6 spine cell) + 6 committed L1/L2/L3/L5 goldens"
  - "boosting_parity spine_end_to_end (L5 bit-exact) + score_accumulation (L2 bit-exact f64) + gradients (L1 ~1e-6) un-#[ignore]d/passing"
affects: [06-03-objectives, 06-04-multiclass-metrics, 06-05-bagging-early-stopping, 08-pyo3-bindings]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Training-side enum-dispatch objective/metric factories (CreateObjectiveFunction/CreateMetric mirror) — net-new math is small; the loop is wiring in C++ order"
    - "Identity-binning public corpus ingestion (bin == raw integer; bin_upper_bound = (b+0.5)+1ULP) reproducing the capture's binning so Rust-grown trees are bit-comparable to the real binary"
    - "Owned-builder -> Config::from_params (D-02): the builder records (key,value) and routes through the verbatim alias+CHECK table; never forks defaults"
    - "Learner returns its locally-built DataPartition (train_returning_partition) so the boosting score-scatter uses the C++ data_partition_ row sets verbatim"

key-files:
  created:
    - crates/lgbm-objective/src/regression.rs
    - crates/lgbm-metric/src/regression.rs
    - crates/lgbm-boosting/src/gbdt.rs
    - crates/lgbm-boosting/src/score_updater.rs
    - crates/lgbm/src/builder.rs
    - crates/lgbm/src/booster.rs
    - crates/lgbm/src/error.rs
    - xtask/py/boosting_oracle_capture.py
    - crates/oracle-harness/tests/fixtures/boosting/regression_spine_model.txt
    - crates/oracle-harness/tests/fixtures/boosting/regression_spine_pred.txt
    - crates/oracle-harness/tests/fixtures/boosting/regression_scores.txt
    - crates/oracle-harness/tests/fixtures/boosting/regression_gh_iter1.txt
    - crates/oracle-harness/tests/fixtures/boosting/regression_gh_iterN.txt
    - crates/oracle-harness/tests/fixtures/boosting/regression_metrics.txt
  modified:
    - crates/lgbm-objective/src/lib.rs
    - crates/lgbm-objective/src/error.rs
    - crates/lgbm-metric/src/lib.rs
    - crates/lgbm-metric/src/error.rs
    - crates/lgbm-boosting/src/lib.rs
    - crates/lgbm-boosting/Cargo.toml
    - crates/lgbm/src/lib.rs
    - crates/lgbm/Cargo.toml
    - crates/lgbm-treelearner/src/learner.rs
    - xtask/src/main.rs
    - crates/oracle-harness/tests/boosting_parity.rs
    - crates/oracle-harness/tests/fixtures/REFERENCE_MANIFEST.md

key-decisions:
  - "Open-Q1 RESOLVED: ConvertOutput stays in lgbm-model (lgbm-objective re-exports ObjectiveKind, does NOT re-port it); lgbm-objective owns the training side only."
  - "Open-Q2/A4 RESOLVED: the per-iter score L2 golden is BIT-EXACT (not ~1e-6) — the Rust internal score_ after k iters equals predict(raw_score=True,num_iteration=k) bit-for-bit on this cell. This phase-wide L2 precision contract is recorded in REFERENCE_MANIFEST.md for 06-05 to inherit without re-deciding."
  - "SerialTreeLearner gains train_returning_partition (not on the C++ API) because the port builds DataPartition locally inside train_inner and does not retain it on self; the boosting score-scatter needs it (the C++ data_partition_ member reproduced as a return value)."
  - "lgbm-boosting + lgbm depend on lgbm-compute for the Backend trait bound only (the learner is generic over B: Backend); CMP-01 holds (no cubecl runtime named — grep cubecl Cargo.toml empty)."
  - "Eval metrics default to [l2, rmse] in train() because Config has no in-scope `metric` field (it is eval-only); matches the capture's metric=[l2,rmse] L3 golden."

patterns-established:
  - "Bit-exact spine replay: the full builder->Config->train->score-accumulate->predict->metric loop reproduces the real lib_lightgbm 4.6 spine model-text leaf values BIT-EXACT (0 bit mismatches across all 10 trees, incl. the AddBias-folded tree 0) + per-iter scores bit-exact f64."
  - "Capture-mirror corpus: the Rust parity test's corpus + identity-binning + seed are kept identical to the python capture so the comparison falsifies only the loop/objective/metric math."

requirements-completed: [BST-01, BST-02, OBJ-01, OBJ-03, MET-01, API-01]

# Metrics
duration: ~40min
completed: 2026-06-07
---

# Phase 6 Plan 02: GBDT Spine Vertical Slice Summary

**The minimal end-to-end regression spine — `Objective::Regression`(L2) + `l2`/`rmse`/`l1` metrics + the `Gbdt::train_one_iter` loop (verbatim `TrainOneIter` order, f64 `ScoreUpdater`, `boost_from_average`) + the public `TrainingBuilder`/`Booster`/`train`/`predict` — runs end-to-end and replays the real `lib_lightgbm` 4.6 spine golden BIT-EXACT (model-text leaf values + per-iter f64 scores) with predictions within ~1e-6, on ONE objective before any axis widens.**

## Performance

- **Duration:** ~40 min
- **Started:** 2026-06-07T08:40Z (approx)
- **Completed:** 2026-06-07
- **Tasks:** 3
- **Files modified:** 26 (14 created + 12 modified; excludes Cargo.lock churn)

## Accomplishments

- **Task 1 — objective + metrics:** `Objective::Regression{sqrt}` (training-side `get_gradients` with the verbatim f64-subtract-then-single-f32-cast op order, `hess=1.0f32`; `boost_from_score` = ordered f64 label-mean; `transform_labels` for the sqrt Init). `Metric::{L2,Rmse,L1}` (ordered f64 `LossOnPoint` reduction + `AverageLoss`; `factor_to_bigger_better=-1`). Every module cites the exact `regression_objective.hpp` / `regression_metric.hpp` lines. `ConvertOutput` is re-exported from `lgbm-model`, not re-ported (Open-Q1).
- **Task 2 — loop + score updater:** `ScoreUpdater` (f64 class-major `score_`; `add_constant` = BoostFromAverage `AddScore`; `add_tree_train_path` delegating to the learner's bit-exact `add_prediction_to_score` per-leaf scatter). `Gbdt::train_one_iter` mirrors `TrainOneIter` exactly: BoostFromAverage (via `AddScore`, NOT `AddBias`) → GetGradients → per-class `learner.train` → RenewTreeOutput (L2 no-op) → Shrinkage **before** UpdateScore → AddBias **after** (model-text only, no double-add). Added `SerialTreeLearner::train_returning_partition` to expose the grown partition for the scatter.
- **Task 3 — public API + capture + replay:** `TrainingBuilder` (→`Config::from_params`, D-02) + `Booster` (best_iteration, eval_history, per-iter score + g/h snapshots) + `train`/`predict_row`/`predict_row_raw`. `boosting-oracle-capture` xtask + `boosting_oracle_capture.py` train the real `lightgbm==4.6.0` spine cell and emit the L1/L2/L3/L5 goldens (version-asserted, byte-idempotent). `boosting_parity::{spine_end_to_end, score_accumulation, gradients}` un-`#[ignore]`d and passing.

## Numerical-fidelity result

- **L5 (model text):** the Rust-grown ensemble's per-tree leaf values match the real `lib_lightgbm` 4.6 golden **BIT-EXACT** (0 mismatches across all 10 trees, including the AddBias-folded tree 0 whose leaves carry the `boost_from_average` init mean of 11.0). `predict()` within `ORACLE_TOL` (~1e-6).
- **L2 (per-iter scores):** the internal f64 `score_` after each iter matches the golden **BIT-EXACT** for every row × every k.
- **L1 (g/h):** per-row grad/hess at iter 1 and iter 5 within `ORACLE_TOL`.

## Task Commits

1. **Task 1: regression L2 objective + l2/rmse/l1 metrics** — `b528437` (feat)
2. **Task 2: GBDT loop + f64 ScoreUpdater + boost_from_average** — `b7426ef` (feat)
3. **Task 3: public builder/Booster/train/predict + spine L1-L5 capture + replay** — `ff9a769` (feat)

## Open Questions Resolved

- **Open-Q1 (where ConvertOutput lives):** RESOLVED — it stays in `lgbm-model::ObjectiveKind`; `lgbm-objective` re-exports it and owns only the training side (`get_gradients`/`boost_from_score`). No re-port, no Phase-3 breakage.
- **Open-Q2 / Assumption A4 (is `predict(raw_score,k)` == internal `score_`?):** RESOLVED — **YES, bit-for-bit** on this cell. Verified two ways: `lgbm::booster::predict_raw_equals_internal_score_open_q2` (asserts `raw.to_bits() == internal[i].to_bits()` for every row × k) and `boosting_parity::score_accumulation` (the internal score_ replays the golden bit-exact).
- **GATE — phase-wide L2 precision contract:** the per-iter score L2 golden is **BIT-EXACT** (rationale: the init enters `score_` once via `BoostFromAverage→AddScore` and the model folds it into tree 0 via `AddBias`, and the training-path scatter is the same f64 per-leaf accumulation the C++ uses; the two re-derivation paths agree to the bit). RECORDED in `crates/oracle-harness/tests/fixtures/REFERENCE_MANIFEST.md` so 06-05's ~40-cell matrix inherits it (06-05 Task 3 reads this contract rather than re-deciding).

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] `lgbm-boosting` (and `lgbm`) depend on `lgbm-compute` for the `Backend` trait bound**
- **Found during:** Task 2
- **Issue:** The 06-01 scaffold asserted `lgbm-boosting` does NOT depend on `lgbm-compute`. But the learner is generic over `B: Backend` (where `Backend` is `lgbm_compute::Backend`), and the `ScoreUpdater::add_tree_train_path` signature names `SerialTreeLearner<'_, B: Backend>` — that bound cannot be written without `lgbm-compute`. This mirrors `lgbm-treelearner`, which itself depends on `lgbm-compute` for the trait only.
- **Fix:** Added `lgbm-compute = { path = "../lgbm-compute" }` to both `lgbm-boosting` and `lgbm` for the trait/types only. The CMP-01 hard gate (`grep cubecl Cargo.toml` empty) still holds — no cubecl runtime is named.
- **Files modified:** crates/lgbm-boosting/Cargo.toml, crates/lgbm/Cargo.toml
- **Verification:** `grep cubecl crates/lgbm-boosting/Cargo.toml` empty; `cargo build` green.
- **Committed in:** b7426ef (Task 2), ff9a769 (Task 3)

**2. [Rule 3 - Blocking] `SerialTreeLearner::train_returning_partition` added (not on the C++ API)**
- **Found during:** Task 2
- **Issue:** The bit-exact training-path score scatter (`add_prediction_to_score`) needs the `DataPartition` the tree was grown over. The C++ `data_partition_` is a learner member, but this port builds the partition locally inside `train_inner` and does not retain it on `self` (a 06-01 decision). The loop therefore had no partition to pass.
- **Fix:** `train_inner` now returns the `DataPartition` as a 4th tuple element (threaded transparently through `train`/`train_with_snapshots`/`train_with_col_sampler_trace` — callers unchanged); new `train_returning_partition` hands it to the boosting layer. The scatter math is identical to the reference.
- **Files modified:** crates/lgbm-treelearner/src/learner.rs
- **Verification:** Phase-5 `learner_parity` unregressed (12 passed); `boosting_parity` L2 + L5 bit-exact.
- **Committed in:** b7426ef (Task 2)

**3. [Rule 3 - Blocking] Eval metrics default to `[l2, rmse]` in `train()` (no `Config.metric` field)**
- **Found during:** Task 3
- **Issue:** The builder's `.metric()` setter records a param, but `lgbm-core::Config` has no in-scope `metric` field (it is an eval-only param, not extracted by `from_params`), so `config.metric` does not compile.
- **Fix:** `train()` evaluates the regression default metrics `[l2, rmse]` (matching the capture's `metric=["l2","rmse"]` L3 golden). When the facade plumbs an explicit eval-metric list (later wave), route it through `default_regression_metrics`'s call site.
- **Files modified:** crates/lgbm/src/booster.rs
- **Verification:** eval_history has l2+rmse, 10 values each; `public_api_train_predict_round_trip` passes.
- **Committed in:** ff9a769 (Task 3)

**Total deviations:** 3 auto-fixed (all Rule 3 — blocking, dependency-graph / API-shape adaptations). No scope creep; all plan acceptance gates satisfied.

## Authentication / Capture Gates

- The `boosting-oracle-capture` real-binary capture requires a `lightgbm==4.6.0` pip wheel. A capture venv (`/tmp/lgbm-capture-venv`) with the recorded version was available, so the capture ran in-flow (NOT a blocking gate this time). The version is asserted before training (threat T-06-02-SC); `cargo test` reads only the committed goldens and needs no toolchain. `LightGBM/` was never `git add`ed.

## Verification

- `cargo test --workspace` GREEN (0 failures across all crates).
- `lgbm-objective` (gradients + metric eval), `lgbm-boosting` (12), `lgbm` (8), `boosting_parity` (3 passed / 3 deferred-#[ignore]d), `learner_parity` (12, unregressed).
- Capture byte-idempotent (md5 identical across two runs).
- Acceptance grep gates: `regression_objective.hpp` / `regression_metric.hpp` citations present; `add_prediction_to_score` + `boost_from_score` wired; `grep cubecl crates/lgbm-boosting/Cargo.toml` empty; `Config` in builder.rs; no forked alias table.

## Known Stubs

- `Booster.best_iteration` is the last trained iteration (no early stopping yet — 06-05) and `eval_history` is the per-round training-set l2/rmse (no valid-set / early-stop population yet). These are intentional spine-scope placeholders, documented for 06-05; they do not block the spine goal (train→predict→metric runs end-to-end and matches the golden).
- `Gbdt::has_init_score` is hard-`false` (the spine corpus carries no `init_score` metadata; the public API does not yet plumb it). The `BoostFromAverage` gate is otherwise complete.

## Next Phase Readiness

- 06-03 (remaining objectives) is UNBLOCKED: the `Objective` enum, the loop's `RenewTreeOutput` hook (already wired, no-op for L2), and the layered-golden capture pipeline are in place; regression_l1 adds a `RenewTreeOutput` median-residual closure + a `PercentileFun` `BoostFromScore`.
- 06-05 inherits the **bit-exact L2 precision contract** from REFERENCE_MANIFEST.md (no re-decision).
- No blockers. CMP-01 holds; `LightGBM/` never git-added.

## Self-Check: PASSED

All key created files exist on disk; all 3 task commit hashes (`b528437`, `b7426ef`, `ff9a769`) are present in git history (verified below).
