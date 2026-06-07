---
phase: 07-parity-completing-variants
plan: 07
subsystem: boosting-variant
tags: [random-forest, rf, boosting-variant, bst-06, averaged-trees, multiply-score, running-average, mandatory-bagging, average-output, renew-tree-output, lightgbm-4.6, oracle-parity, numerical-fidelity]

# Dependency graph
requires:
  - phase: 07-06
    provides: "the BoostingVariant {Gbdt, Dart, Rf} enum field on the single Gbdt driver (RESEARCH Pattern 1 — enum field, NOT trait objects) + the boosting-variant facade/builder/xtask/parity-cell idioms (read_*_golden skip-pass, compare_exact_f64_bits, the 4.6.0 capture posture); RF is the third BoostingVariant branch, mirroring DART's design"
  - phase: 07-01
    provides: "the BAGGING BIT-EXACT foundation (D-05 min_gain_shift faithful-fix) — RF has MANDATORY bagging and inherits the bit-exact BaggingSampleStrategy RNG golden; the bagged-subset leaf structure reaches faithful parity (not a bounded-cap)"
  - phase: 06-05
    provides: "the Gbdt::train_one_iter loop + the ScoreUpdater f64 accumulator + add_tree_predict_path (the full-corpus predict-side scatter RF reuses for the running-average fold)"
provides:
  - "BoostingVariant::Rf branch on the single Gbdt driver (rf.hpp:90-182): averaged (not accumulated) trees via the MultiplyScore(iter); AddScore; MultiplyScore(1/(iter+1)) running-average sandwich, shrinkage 1.0 (NO learning-rate accumulation)"
  - "RF::Boosting (rf.hpp:90-109) ported as Gbdt::rf_boosting — grad/hess derived ONCE from a CONSTANT init-score buffer (every row of class k = init_scores_[k] = BoostFromAverage(k, false)), reused every iteration; the trees differ only through the per-iteration bagged subset"
  - "the two RF CHECKs (rf.hpp:35-40,91-93) as a typed BoostingError::RfConfig at the top of the RF path (iter 0, before any tree grows): objective != null (custom rejected) AND (bagging active OR feature_fraction<1)"
  - "RF RenewTreeOutput GATED on obj->IsRenewTreeOutput() (serial_tree_learner.cpp:922) — a NO-OP for L2 (the leaf is the learner's gradient-fit -sum_grad/sum_hess Newton output over the bagged subset, then AddBias(init)), ACTIVE for regression_l1/quantile/mape (dispatched via obj.renew_leaf_output, the correct percentile — NOT a plain mean) with residual_getter = label - init_scores_[k] (a CONSTANT pred)"
  - "ScoreUpdater::multiply_score (C++ MultiplyScore, score_updater.hpp:63-69) for the running average"
  - "into_model sets average_output=true for RF; Booster::predict_row divides the per-tree sum by num_iteration for average_output models (gbdt_prediction.cpp:57-59)"
  - "boosting=rf facade selection (Gbdt::with_rf) in train_inner_full; RF reuses the proven BaggingSampleStrategy; builder feature_fraction setter (RF's alt randomization source)"
  - "rf-oracle-capture xtask + xtask/py/rf_oracle_capture.py; 2 real-lib_lightgbm-4.6 cells (single-output regression + multiclass, mandatory bagging) model+pred, byte-idempotent"
  - "BST-06 RF validated: real-binary parity (single-output BIT-EXACT leaf values across 12 averaged trees; multiclass class-major structure exact + predict within ORACLE_TOL for the documented exp-libm residual)"
affects: [07]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "RF as a BoostingVariant ENUM FIELD on the single Gbdt driver (RESEARCH Pattern 1) — C++ subclasses GBDT, the Rust port branches on the discriminant. RF is structurally distinct enough (constant grad/hess, running-average score, init-score residual, no shrinkage) that it gets a dedicated train_one_iter_rf early-return branch rather than weaving into the spine loop body, keeping the GBDT spine path byte-unchanged."
    - "RF averages trees: the score buffer is kept as a RUNNING AVERAGE via MultiplyScore(iter); AddScore; MultiplyScore(1/(iter+1)) (rf.hpp:157-159), the stored leaf values are the RAW per-tree outputs, and prediction divides the per-tree sum by num_iteration (average_output, gbdt_prediction.cpp:57-59). This is the opposite of the GBDT additive-shrinkage accumulation."
    - "RF grad/hess are derived ONCE from a CONSTANT init-score buffer (rf.hpp Boosting()), NOT re-derived from the accumulated score each iter — the 'random forest of trees on a fixed target' semantics. The trees differ only through the per-iteration bagged subset."
    - "RenewTreeOutput is GATED on obj->IsRenewTreeOutput() in RF exactly as in the GBDT spine (a no-op for L2). The first RF cut renewed unconditionally to a mean residual, which diverged from the real binary's L2 leaf (the learner's Newton gradient-fit) by f64 fold order — proving the gate is load-bearing, not cosmetic."

key-files:
  created:
    - crates/oracle-harness/tests/fixtures/rf/.gitkeep
    - crates/oracle-harness/tests/fixtures/rf/rf_single_bag_model.txt
    - crates/oracle-harness/tests/fixtures/rf/rf_single_bag_pred.txt
    - crates/oracle-harness/tests/fixtures/rf/rf_multi_bag_model.txt
    - crates/oracle-harness/tests/fixtures/rf/rf_multi_bag_pred.txt
    - xtask/py/rf_oracle_capture.py
    - .planning/phases/07-parity-completing-variants/07-07-SUMMARY.md
  modified:
    - crates/lgbm-boosting/src/gbdt.rs
    - crates/lgbm-boosting/src/lib.rs
    - crates/lgbm-boosting/src/error.rs
    - crates/lgbm-boosting/src/score_updater.rs
    - crates/lgbm/src/booster.rs
    - crates/lgbm/src/builder.rs
    - crates/oracle-harness/tests/boosting_parity.rs
    - xtask/src/main.rs
    - .planning/REFERENCE_MANIFEST.md

key-decisions:
  - "RF is a BoostingVariant::Rf enum field on Gbdt (RESEARCH Pattern 1), NOT a trait object. Because RF's iteration is structurally distinct from the GBDT spine (constant grad/hess derived once, running-average score via the MultiplyScore sandwich, init-score residual renew, no shrinkage), train_one_iter dispatches to a dedicated train_one_iter_rf early-return branch at the top — keeping the GBDT/DART/GOSS spine body byte-unchanged. RfState (Option<RfState>, Some iff variant==Rf) holds the once-computed grad/hess + init_scores, cloned out per iter to satisfy the borrow checker across the learner + score_updater mut-borrows."
  - "RF::Boosting derives grad/hess ONCE from a CONSTANT init-score buffer (rf.hpp:90-109), not from the accumulated score. Computed on iter 0 in rf_boosting and reused every iteration. The init score per class is obj.boost_from_score(k) (BoostFromAverage(k, update_scorer=false) — RF does NOT additively inject init into score_; score_ is a running average seeded by the first tree)."
  - "[Rule 1 - Bug, capture-revealed] RF RenewTreeOutput MUST be gated on obj->IsRenewTreeOutput() (serial_tree_learner.cpp:922) — a NO-OP for L2. The first cut renewed unconditionally to the mean residual; the real-binary capture revealed the L2 leaf is the learner's gradient-fit -sum_grad/sum_hess Newton output over the bagged subset (rust 2.5 vs cpp 2.5000000000000036, an f64 fold-order divergence). With the gate, the L2 leaf comes from the learner output then AddBias(init), and the renew helpers dispatch via obj.renew_leaf_output (correct percentile for L1/quantile/mape, not a plain mean) for renewing objectives."
  - "The single-output (regression) RF cell asserts BIT-EXACT leaf values vs real lib_lightgbm 4.6 (the bagged-subset leaf structure reaches faithful parity, inheriting the 07-01 D-05 min_gain_shift fix). The multiclass RF cell asserts the class-major STRUCTURE exactly (tree count == iters × num_class, the stride) and predict() within ORACLE_TOL — the redundant-form softmax exp is a transcendental whose Rust-libm vs C++-wheel-libm ~1-ULP gap is the documented exp-libm residual (06-04-SUMMARY); no fabricated bit-exactness."
  - "No DEF-07-02-style deferral was needed. RF (a boosting-loop variant) reached faithful parity — the only fix was the IsRenewTreeOutput gate (a faithful bug), not a learner-level knife-edge requiring an FP-trace deferral. BST-06 is marked complete."

patterns-established:
  - "A boosting VARIANT whose per-iteration math diverges structurally from the spine (RF: constant grad/hess, running-average score, no shrinkage) gets a dedicated train_one_iter_<variant> early-return branch off train_one_iter, not an in-line weave — its per-train state is an Option<VariantState> drained/cloned during the mut-self learner+score_updater interaction."
  - "average_output (RF) keeps the model text carrying the RAW per-tree leaf values + a bare `average_output` line; the running-average is reproduced in score_ at train time (MultiplyScore sandwich) and at predict time (/= num_iteration in Booster::predict_row). The stored leaves are NOT pre-averaged — matching the real binary's model.txt bit-exact."

requirements-completed: [BST-06]

# Metrics
duration: ~1 session (TDD RF core + rf_boosting/running-average + facade/builder/xtask wiring, capture via the ready 4.6.0 venv, the IsRenewTreeOutput-gate fix the capture revealed, byte-idempotent verify + teeth)
completed: 2026-06-07
---

# Phase 7 Plan 07: Random Forest Boosting Variant (BST-06) Summary

**Random Forest ships end-to-end as a `BoostingVariant::Rf` ENUM FIELD on the single `Gbdt` driver (RESEARCH Pattern 1 — not trait objects), faithful 1:1 to `rf.hpp`: grad/hess derived ONCE from a CONSTANT `BoostFromAverage` init-score buffer (`RF::Boosting`, rf.hpp:90-109), AVERAGED (not accumulated) trees via the `MultiplyScore(iter); AddScore; MultiplyScore(1/(iter+1))` running-average sandwich (rf.hpp:157-159) with NO learning-rate shrinkage, per-tree leaf renewal GATED on `obj->IsRenewTreeOutput()` (a no-op for L2 — the learner's gradient-fit Newton output) using `residual_getter = label - init_scores_[k]` (a CONSTANT pred), and `average_output` prediction (the per-tree sum / num_iteration). It is selected on `boosting=rf`, reuses the proven `BaggingSampleStrategy` (the 07-01 bit-exact bagging RNG golden carries over), and enforces the two RF CHECKs (objective != null; bagging-or-feature_fraction active) as a typed `BoostingError::RfConfig`. Validated against real `lib_lightgbm` 4.6: single-output regression BIT-EXACT leaf values (12 averaged trees, mandatory bagging), multiclass class-major structure exact (36 trees = 12 iters × 3 classes) + predict within ORACLE_TOL (the documented exp-libm residual).**

## Performance

- **Duration:** ~1 session.
- **Completed:** 2026-06-07
- **Tasks:** 3 — (1) `BoostingVariant::Rf` branch + `rf_boosting` + the running-average fold + the two CHECKs (TDD, 6 unit tests); (2) builder/facade selection + `feature_fraction` setter + xtask `rf-oracle-capture` emitter + capture-gated parity cells + 3 facade/builder tests; (3) the real-binary capture (the wheel gate was already satisfied by the ready `/tmp/lgbm-capture-venv` 4.6.0 venv, so the executor completed it in-session rather than halting — and it revealed the `IsRenewTreeOutput` gate fix).

## What shipped

1. **`BoostingVariant::Rf` + `RfConfig` + `RfState`** (`crates/lgbm-boosting/src/gbdt.rs`) — the third `{Gbdt, Dart, Rf}` enum-field branch, the resolved RF config (mandatory-randomization flags), and the per-train state (once-computed `gradients`/`hessians`/`init_scores`). `with_rf` chained ctor.
2. **`rf_boosting`** (`rf.hpp:90-109`): the two CHECKs (objective != null → `RfConfig`; bagging-or-feature_fraction active → `RfConfig`) + grad/hess derived ONCE from the constant init-score buffer (every row of class `k` = `init_scores_[k]`).
3. **`train_one_iter_rf`** (`rf.hpp:111-182`): mandatory bag draw (reusing `BaggingSampleStrategy`) → per-class tree on the in-bag subset against the once-derived grad/hess → `RenewTreeOutput` GATED on `is_renew_tree_output()` (residual `label - init`, the objective's percentile) → `AddBias(init)` → the running-average score fold. No learning-rate shrinkage.
4. **`rf_update_score`** (`rf.hpp:157-159`): `MultiplyScore(iter); AddScore(tree) over the full corpus; MultiplyScore(1/(iter+1))` — `score_` is the running mean of the per-tree raw outputs.
5. **`ScoreUpdater::multiply_score`** (`crates/lgbm-boosting/src/score_updater.rs`, C++ `MultiplyScore`).
6. **Facade + builder** (`crates/lgbm/src/booster.rs`, `builder.rs`): `boosting=rf` selects `with_rf` in `train_inner_full` (RF reuses the bagging strategy); `predict_row` averages for `average_output` models; `feature_fraction` setter; `RfConfig` export. `into_model` sets `average_output=true` for RF.
7. **Capture** (`xtask/src/main.rs` + `xtask/py/rf_oracle_capture.py`): `rf-oracle-capture` emits the single-output + multiclass real-binary model+pred cells (mandatory bagging); byte-idempotent; version-pinned 4.6.0.
8. **Parity cells** (`boosting_parity.rs`): `rf_single_parity` (per-tree leaf values bit-exact, `average_output`, predict within ORACLE_TOL) + `rf_multi_parity` (class-major structure exact, predict within ORACLE_TOL).

## Deviations from Plan

### [Rule 1 — Bug, capture-revealed] RF RenewTreeOutput must be gated on `IsRenewTreeOutput()`

- **Found during:** Task 3 — running `rf_single_parity` after the first RF cut captured its goldens. The Rust L2 leaf came out `2.5` vs the real binary's `2.5000000000000036` (8 ULPs).
- **Issue:** the first RF cut renewed leaf outputs UNCONDITIONALLY to the mean residual `label - init`. But C++ `SerialTreeLearner::RenewTreeOutput` is gated on `obj->IsRenewTreeOutput()` (serial_tree_learner.cpp:922) — **FALSE for L2**. For L2 the leaf value is the tree learner's gradient-fit Newton output `-sum_grad/sum_hess` over the bagged subset (then `AddBias(init)`), whose f64 fold order differs from a directly-computed mean residual.
- **Fix:** gated both RF renew sites on `self.objective.is_renew_tree_output()` (a no-op for L2 → the learner output is kept; active for regression_l1/quantile/mape → the renew helpers now dispatch via `obj.renew_leaf_output`, the correct percentile, not a plain mean). The single-output cell then asserts BIT-EXACT.
- **Files modified:** `crates/lgbm-boosting/src/gbdt.rs`.
- **Commit:** `f4969ab`.

### Builder `feature_fraction` setter added (Rule 2 — completeness)

- **Found during:** Task 2 — the plan asked to "confirm bagging_fraction/bagging_freq/feature_fraction setters exist". `bagging_*` existed; `feature_fraction` did not.
- **Fix:** added `TrainingBuilder::feature_fraction` (RF's alternative mandatory-randomization source, `rf.hpp:37`). The capture matrix uses bagging, but the setter completes the RF randomization surface.
- **Files modified:** `crates/lgbm/src/builder.rs`. **Commit:** `7743e1f`.

**Total deviations:** 2 (1 capture-revealed faithful bug fix; 1 missing setter). No architectural change, no user decision needed. No DEF-07-02-style deferral was required — RF reached faithful parity (the only fix was the `IsRenewTreeOutput` gate, not a learner-level knife-edge).

## Verification

- `cargo test -p lgbm-boosting` — **GREEN** (50 lib tests incl. 6 RF: variant selection, both CHECK rejects, running-average-no-shrinkage, predict-averages-tree-outputs, multiply_score running-average).
- `cargo test -p lgbm --lib` — **GREEN** (21 tests incl. `rf_setters_route_into_config`, `rf_train_predict_averages_tree_outputs`, `rf_without_randomization_is_typed_error`).
- `cargo test -p oracle-harness --test boosting_parity rf` — **GREEN** (`rf_single_parity` BIT-EXACT leaf values, `rf_multi_parity` class-major structure + predict within ORACLE_TOL — both with goldens present, NOT skip-passing).
- **Teeth verified:** corrupting `rf_single_bag_model.txt` leaf_value FAILS `rf_single_parity`; corrupting `rf_multi_bag_pred.txt` FAILS `rf_multi_parity`; both restored.
- **Byte-idempotent:** a second `rf-oracle-capture` left identical md5s over all 4 `fixtures/rf/` files.
- `cargo test --workspace` — **GREEN** (boosting_parity 60 passed / 13 ignored — the 13 ignored are the unrelated DEF-07-02 cells, untouched).
- **Spine NOT regressed:** `learner_parity` 12/12, `kernel_parity` 4/4; the GBDT spine, bagging, GOSS, and DART cells all GREEN.
- `cargo build --workspace --tests` — exit 0; clippy clean on every RF-edited region (the only `boosting_parity`/`booster`/`builder` clippy warnings are pre-existing and unrelated; the two `&feature_row` warnings in `gbdt.rs` predate this plan, noted out-of-scope in 07-06).
- `git status --porcelain LightGBM/` — `LightGBM/` untracked, never git-added.

## Out-of-scope (not fixed — deviation scope boundary)

- Pre-existing `clippy::needless_borrows_for_generic_args` `&feature_row` warnings in the bagging/DART path of `gbdt.rs` (documented out-of-scope in 07-06-SUMMARY) — left untouched. The RF code (`train_one_iter_rf`, `rf_boosting`, `rf_update_score`, `rf_renew_*`) is clippy-clean.

## Known Stubs

None. The `rf_renew_full` / `rf_renew_subset` helpers are currently exercised only when `is_renew_tree_output()` is true (regression_l1/quantile/mape RF) — the captured cells are L2 (no-op renew) and multiclass. They are kept faithful (dispatch via `obj.renew_leaf_output`, the correct percentile) for any future renewing-objective RF, NOT a stub.

## Task Commits

1. `ce45915` — `feat(07-07)`: BoostingVariant::Rf branch — averaged trees, MultiplyScore rescale, per-tree renew, 2 CHECKs.
2. `7743e1f` — `feat(07-07)`: RF builder/facade selection + capture emitter + capture-gated parity cells.
3. `f4969ab` — `test(07-07)`: capture RF real-lib_lightgbm-4.6 goldens + gate RenewTreeOutput on IsRenewTreeOutput.

## Self-Check: PASSED

- `07-07-SUMMARY.md` exists on disk; `BoostingVariant::Rf` + the 2 model cells + 2 pred files + `rf_oracle_capture.py` + the rf fixture dir all present.
- Commits `ce45915` / `7743e1f` / `f4969ab` present in history.
- `cargo test --workspace` GREEN; `LightGBM/` never git-added.
