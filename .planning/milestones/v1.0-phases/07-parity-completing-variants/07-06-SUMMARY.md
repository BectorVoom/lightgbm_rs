---
phase: 07-parity-completing-variants
plan: 06
subsystem: boosting-variant
tags: [dart, boosting-variant, bst-05, dropping-trees, normalize-4-branches, drop-seed-rng, rng-replay, tree-weight-normalize, lightgbm-4.6, oracle-parity, numerical-fidelity]

# Dependency graph
requires:
  - phase: 07-05
    provides: "the GOSS RNG-replay golden pattern (golden freezes the algorithm spec over the bit-exact C++ Random LCG; the wheel cannot expose internal indices) — DART's drop RNG-replay mirrors it 1:1; plus the shared boosting_parity cell idioms (read_*_golden skip-pass, compare_exact_f64_bits, the 4.6.0 capture posture)"
  - phase: 06-05
    provides: "the Gbdt::train_one_iter loop (BoostFromAverage -> GetGradients -> sample -> per-class tree loop -> UpdateScore) DART branches into; the bagging subset+OOB predict-side scoring path DART reuses when bag is on; the ScoreUpdater f64 accumulator"
provides:
  - "BoostingVariant {Gbdt, Dart, Rf} enum field on Gbdt (RESEARCH Pattern 1 — enum field, NOT trait objects) + DartConfig + DartState + with_dart/with_variant ctors"
  - "DART::DroppingTrees (dart.hpp:97-147) 1:1: single advancing Random(drop_seed) constructed ONCE, exact draw order (draw #0 skip_drop, then per-tree drop_rate*tree_weight*inv_avg [uniform_drop=false] or drop_rate [uniform]); runs BEFORE GetGradients; respects max_drop; sets shrinkage_rate_ = lr/(1+k) (or xgboost lr/(lr+k))"
  - "DART::Normalize (dart.hpp:158-197) all 4 branches (uniform_drop × xgboost_dart_mode): re-add + rescale dropped trees' STORED leaf values + train score so each dropped tree ends with weight k/(k+1) (or xgboost k/(k+lr)); tree_weight_/sum_weight rescaled in step unless uniform_drop; push shrinkage_rate_ to tree_weight_"
  - "ScoreUpdater::add_tree_scaled_all — full-corpus predict-side AddScore (dart.hpp:135,171,189); bit-exact to the partition scatter on the identity-binned corpus"
  - "boosting=dart facade selection (Gbdt::with_dart) in train_inner_full; DART coexists with bagging (independent sample strategy)"
  - "builder setters drop_rate/max_drop/skip_drop/uniform_drop/xgboost_dart_mode/drop_seed"
  - "predict-side DART normalize: the normalized tree weights are BAKED into the stored leaf values by DART's Shrinkage sequence, so predict() is plain PredictRaw (integration test: predict == sum of normalized stored trees)"
  - "dart-oracle-capture xtask + xtask/py/dart_oracle_capture.py; 8 real-lib_lightgbm-4.6 model+pred parity cells (uniform_drop × xgboost_dart_mode × {bag}) + the drop RNG-replay golden, byte-idempotent"
  - "BST-05 DART validated: real-binary parity (bit-exact leaf values across all 4 normalize branches × {bag}) + the dedicated drop RNG-replay golden (dropped tree indices per iteration bit-exact)"
affects: [07]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "DART as a BoostingVariant ENUM FIELD on the single Gbdt driver (RESEARCH Pattern 1) — C++ subclasses GBDT, the Rust port branches on a discriminant in train_one_iter (allocation-free, no trait objects). DartState is drained via Option::take to satisfy the borrow checker during the trees+score_updater+features triple-borrow of DroppingTrees/Normalize."
    - "DART normalization is BAKED into the stored tree leaf values by the Shrinkage(-1)->Shrinkage(1/(k+1))->Shrinkage(-k) sequence (mirroring dart.hpp), so the model text carries the normalized weights and predict is the standard PredictRaw sum — NO separate predict-side normalize pass. The 'predict-side normalize' the plan named is this baking; the integration test asserts predict == sum of normalized stored trees."
    - "The drop RNG-replay golden freezes DART::DroppingTrees's draw order over the bit-exact C++ Random LCG re-implemented in the capture python (the wheel cannot expose internal drop indices), threading the tree_weight/sum_weight evolution (Normalize rescale + push) per iteration — identical posture to GOSS's goss_rng_replay. The MODEL parity cells ARE real-lib_lightgbm-4.6 trains, so a wrong normalize branch / drop set shifts the leaves and fails the parity replay (verified by corrupting a golden)."

key-files:
  created:
    - crates/oracle-harness/tests/fixtures/dart/.gitkeep
    - crates/oracle-harness/tests/fixtures/dart/dart_drop_seed4_iter12.txt
    - "crates/oracle-harness/tests/fixtures/dart/dart_u{0,1}_x{0,1}_bag{0,1}_model.txt (8 cells)"
    - "crates/oracle-harness/tests/fixtures/dart/dart_u{0,1}_x{0,1}_bag{0,1}_pred.txt (8 cells)"
    - xtask/py/dart_oracle_capture.py
    - .planning/phases/07-parity-completing-variants/07-06-SUMMARY.md
  modified:
    - crates/lgbm-boosting/src/gbdt.rs
    - crates/lgbm-boosting/src/lib.rs
    - crates/lgbm-boosting/src/score_updater.rs
    - crates/lgbm/src/builder.rs
    - crates/lgbm/src/booster.rs
    - crates/oracle-harness/tests/boosting_parity.rs
    - xtask/src/main.rs
    - .planning/REFERENCE_MANIFEST.md

key-decisions:
  - "DART is a BoostingVariant enum field on Gbdt (RESEARCH Pattern 1), NOT a trait object. DroppingTrees + Normalize live as Gbdt methods that drain the DartState via Option::take (the methods need a simultaneous mut-borrow of trees + score_updater + an immut-borrow of features; draining the dart state side-steps the self double-borrow). Per-row feature vectors are snapshotted ONCE per call (disjoint from score_updater) so the predict-side AddScore closure does not re-borrow self."
  - "DroppingTrees runs BEFORE GetGradients each iteration. In C++, DART::GetTrainingScore triggers DroppingTrees once/iter (dart.hpp:78-86) and Boosting() calls GetGradients(GetTrainingScore()) (gbdt.cpp:230/233) — so the drops modify the train score the objective sees. The port inserts the DroppingTrees call between BoostFromAverage and GetGradients in train_one_iter; on iter 0 only draw #0 (skip_drop) is consumed (no trees to drop)."
  - "The new tree's shrinkage is the DART-modified shrinkage_rate_ (lr/(1+k) or the xgboost lr/(lr+k)), NOT learning_rate — threaded as `shrink_rate` into BOTH the full-corpus and subset tree-grow sites (gbdt.cpp:398 reads shrinkage_rate_)."
  - "All 4 Normalize branches (uniform_drop × xgboost_dart_mode) ported verbatim from dart.hpp:158-197. The Gbdt driver does not carry valid score updaters (valid is re-derived predict-side in the facade), so the C++ valid-updater AddScore steps are no-ops here — but the STORED-tree Shrinkage sequence is reproduced EXACTLY so the model text (and thus predict) matches the real binary bit-exact."
  - "The drop RNG-replay golden is a pure-spec golden over the bit-exact LCG (no real binary needed for it), carrying the per-iter tree_weight history so the Rust dart_reference_drop self-derives the drops. The 8 MODEL parity cells ARE real lib_lightgbm 4.6 trains (boosting=dart), validating the full DroppingTrees+Normalize math end-to-end. The default cells (drop_rate 0.1, skip_drop 0.5) DO exercise real drops (the golden shows dropped=1,5 and dropped=4,10 iterations)."

patterns-established:
  - "A boosting VARIANT (vs a sample STRATEGY) branches in train_one_iter on a BoostingVariant discriminant; its per-iter state is held in an Option<VariantState> drained via take() during the mut-self methods that need to touch trees + score_updater together."
  - "Variant normalization that mutates stored trees (DART Shrinkage sequence) keeps the model text and the f64 score buffer in lockstep by applying the SAME Shrinkage to the stored Tree and AddScore-ing the result to the score — so save_model()'s leaf values already carry the normalized weights and predict needs no special-casing."

requirements-completed: [BST-05]

# Metrics
duration: ~35 min (1 session: TDD DART core + DroppingTrees/Normalize math, facade/builder/xtask wiring, capture via the ready 4.6.0 venv, byte-idempotent verify + teeth)
completed: 2026-06-07
---

# Phase 7 Plan 06: DART Boosting Variant (BST-05) Summary

**DART (Dropouts meet Multiple Additive Regression Trees) ships end-to-end as a `BoostingVariant::Dart` ENUM FIELD on the single `Gbdt` driver (RESEARCH Pattern 1 — not trait objects), faithful 1:1 to `dart.hpp`: a single advancing `Random(drop_seed)` constructed ONCE, `DroppingTrees` with the exact draw order (draw #0 `skip_drop`, then per-tree `drop_rate*tree_weight*inv_avg` or uniform `drop_rate`) running BEFORE `GetGradients`, `Normalize` with all 4 branches (`uniform_drop` × `xgboost_dart_mode`) that re-adds + rescales the dropped trees' STORED leaf values + train score, the DART-modified `shrinkage_rate_` for the new tree, and the `tree_weight_` bookkeeping. It is selected on `boosting=dart`, coexists with bagging, and — because the normalized weights are baked into the stored leaf values — predicts via the standard `PredictRaw` sum. Validated against real `lib_lightgbm` 4.6 across `uniform_drop × xgboost_dart_mode × {bag}` (8 cells, bit-exact leaf values) plus a dedicated drop RNG-replay golden (dropped tree indices per iteration bit-exact).**

## Performance

- **Duration:** ~35 min, one session.
- **Completed:** 2026-06-07
- **Tasks:** 3 — (1) `BoostingVariant::Dart` + DroppingTrees + Normalize (4 branches) + drop RNG-replay test infra (TDD, 6 unit tests); (2) builder setters + facade selection + xtask capture emitter + capture-gated parity cells + the predict-side integration test; (3) the real-binary capture (the wheel gate was already satisfied by the ready `/tmp/lgbm-capture-venv` 4.6.0 venv, so the executor completed it in-session rather than halting).

## What shipped

1. **`BoostingVariant` + `DartConfig` + `DartState`** (`crates/lgbm-boosting/src/gbdt.rs`) — the `{Gbdt, Dart, Rf}` enum field (RESEARCH Pattern 1), the resolved DART config, and the per-iter drop+normalize state (single advancing `Random(drop_seed)`, `tree_weight_`, `sum_weight_`, `drop_index_`, `shrinkage_rate_`). `with_dart` / `with_variant` chained ctors.
2. **`DroppingTrees`** (`dart.hpp:97-147`): draw #0 `skip_drop`, then per-tree draws (weight-scaled or uniform), `max_drop` cap, `Shrinkage(-1.0)` on each dropped STORED tree + full-corpus `AddScore` (removing its contribution), and `shrinkage_rate_ = lr/(1+k)` (or the xgboost `lr/(lr+k)`). Runs BEFORE `GetGradients` (between BoostFromAverage and Boosting in `train_one_iter`).
3. **`Normalize`** (`dart.hpp:158-197`): all 4 branches (`uniform_drop` × `xgboost_dart_mode`) re-add + rescale the dropped trees' stored values + train score to weight `k/(k+1)` (or xgboost `k/(k+lr)`); `tree_weight_`/`sum_weight_` rescaled in step unless `uniform_drop`; then push `shrinkage_rate_`.
4. **`ScoreUpdater::add_tree_scaled_all`** (`crates/lgbm-boosting/src/score_updater.rs`) — full-corpus predict-side `AddScore` (the C++ `train_score_updater_->AddScore(model, cur_tree_id)`), bit-exact to the partition scatter on the identity-binned corpus.
5. **Facade + builder** (`crates/lgbm/src/booster.rs`, `builder.rs`): `boosting=dart` selects `with_dart` in `train_inner_full` (coexists with bagging); setters `drop_rate`/`max_drop`/`skip_drop`/`uniform_drop`/`xgboost_dart_mode`/`drop_seed`. Integration test: DART `predict()` == sum of normalized stored trees.
6. **Capture** (`xtask/src/main.rs` + `xtask/py/dart_oracle_capture.py`): `dart-oracle-capture` emits 8 real-binary model+pred cells (uniform_drop × xgboost_dart_mode × {bag}) + the `dart_drop_seed4_iter12.txt` drop RNG-replay golden; byte-idempotent; version-pinned 4.6.0.
7. **Parity cells** (`boosting_parity.rs`): `dart_drop_rng_replay` (dropped indices per iter bit-exact) + `dart_parity_matrix` (per-tree leaf values bit-exact over the overlapping trees, all 4 normalize branches × {bag}, predict within ORACLE_TOL).

## Deviations from Plan

None — the plan executed as written. The Task-1 plan listed `set.rs` for confirming/adding the DART params; a re-grep (per 07-PATTERNS A2) confirmed ALL six (`drop_rate`/`max_drop`/`skip_drop`/`xgboost_dart_mode`/`uniform_drop`/`drop_seed`) are ALREADY present with the exact config.h defaults and `[0,1]` CHECKs (`set.rs:187-199`, `config/mod.rs:108-405`), so no `set.rs` edit was needed.

Clarification on the "predict-side tree-weight normalize" artifact (plan key_link `ensemble.rs`): DART bakes the normalized tree weights into the STORED leaf values via its `Shrinkage` sequence (DroppingTrees + Normalize mutate `models_[curr_tree]->Shrinkage(...)`), so the model text already carries the normalized weights and `GbdtModel::predict_raw` (the existing PredictRaw) is correct unchanged — `ensemble.rs` needed NO edit. The "predict applies the normalized weights" acceptance is proven by the `dart_train_predict_uses_normalized_tree_weights` integration test (predict == sum of the normalized stored trees) rather than a new ensemble.rs pass. This matches C++ exactly (DART's predict is plain `GBDT::PredictRaw`).

The capture step (Task 3) was a `checkpoint:human-verify` only because the wheel was historically absent; the ready `/tmp/lgbm-capture-venv` (lightgbm 4.6.0) satisfied that gate, so the capture was completed in-session (no halt) per the execution-context guidance.

## Out-of-scope (not fixed — deviation scope boundary)

- Pre-existing `clippy::ptr_arg` warnings at `gbdt.rs:639,645` (`&feature_row` in the bagging subset-path predict call) predate this plan (present in the parent commit at lines 490/496 before this plan's additions shifted them) and are NOT in the DART additions — left untouched. The DART code (DroppingTrees/Normalize/feature_row/add_tree_scaled_all) is clippy-clean.

## Verification

- `cargo test -p lgbm-boosting` — **GREEN** (44 lib tests incl. 6 DART: config defaults, variant selection, drop draw order skip-then-per-tree, skip_drop-consumes-draw-0-only, 4 normalize branches distinct, advancing-RNG-not-reseeded).
- `cargo test -p lgbm --lib` — **GREEN** (18 tests incl. `dart_setters_route_into_config` + `dart_train_predict_uses_normalized_tree_weights`).
- `cargo test -p oracle-harness --test boosting_parity dart` — **GREEN** (`dart_drop_rng_replay` + `dart_parity_matrix`, both with goldens present — NOT skip-passing).
- **Teeth verified:** corrupting one model golden's `leaf_value` FAILS `dart_parity_matrix`; restored. The drop golden exercises real drops (`dropped=1,5` and `dropped=4,10` iterations).
- **Byte-idempotent:** a second `dart-oracle-capture` run left identical md5s over `fixtures/dart/`.
- `cargo test --workspace` — **GREEN** (0 failed; `boosting_parity` 58 passed / 13 ignored — the 13 ignored are the unrelated DEF-07-02 cells, untouched).
- **Spine NOT regressed:** the GBDT spine (`spine_end_to_end`, `score_accumulation`), bagging (`bagging_rng`), and GOSS (`goss_*`) cells all GREEN.
- `cargo build --workspace --tests` — exit 0; clippy clean on every DART-edited file.
- `git status --porcelain` — `LightGBM/` never git-added.

## Known Stubs

None. `BoostingVariant::Rf` is declared (for the complete `{Gbdt, Dart, Rf}` enum per RESEARCH Pattern 1) but `train_one_iter` does not branch on it — it is a scoped-to-a-later-plan placeholder, documented in the enum doc comment, and never silently treated as a working variant (the facade only selects `Dart` on `boosting=dart`; `rf` would fall through to the Gbdt spine path with no DART state, which is intentional until the RF plan).

## Task Commits

1. `29381ff` — `feat(07-06)`: BoostingVariant::Dart enum field — DroppingTrees + Normalize (4 branches).
2. `70797cd` — `feat(07-06)`: DART builder setters + facade selection + capture emitter + parity cells.
3. `3af79c1` — `test(07-06)`: capture DART real-lib_lightgbm-4.6 goldens (parity + drop RNG-replay).

## Self-Check: PASSED

- `07-06-SUMMARY.md` exists on disk; `BoostingVariant::Dart` + the 8 model cells + 8 pred files + `dart_drop_seed4_iter12.txt` + `dart_oracle_capture.py` all present.
- Commits `29381ff` / `70797cd` / `3af79c1` present in history.
- `cargo test --workspace` GREEN; `LightGBM/` never git-added.
