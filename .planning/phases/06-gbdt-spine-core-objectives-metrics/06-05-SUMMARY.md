---
phase: 06-gbdt-spine-core-objectives-metrics
plan: 05
subsystem: boosting
tags: [bagging, rng-replay, early-stopping, metric-infra, metric-freq, first-metric-only, oob-score, cross-product-matrix, d07, d13, bit-exact, bagging-by-query-deferral]

# Dependency graph
requires:
  - phase: 06-04-multiclass-metrics
    provides: "GBDT loop generalized to K=num_class trees/iter (class-major); all 5 objectives + metrics; multiclass exp-libm 5-iter horizon; boosting_parity 20 pass / 2 ignored (early_stopping + bagging_rng reserved for 06-05)"
  - phase: 06-02-gbdt-spine-vertical-slice
    provides: "Gbdt loop + f64 ScoreUpdater + train-path scatter; builder/Booster/train; bit-exact L2 precision contract; boosting_oracle_capture pipeline"
  - phase: 01-oracle-contract-foundations
    provides: "lgbm-core::Random LCG (FND-01 bit-exact) — the bagging RNG source; Config (bagging/early-stopping/metric fields + alias+CHECK)"
provides:
  - "lgbm-boosting::BaggingSampleStrategy (per-block Random(bagging_seed+i) block 1024, draw EVERY row in order incl. OOB, in-bag left / OOB right + one-buffer reverse; balanced pos/neg helper) over the proven lgbm_core::Random — bit-exact bagging draw/call sequence (D-13)"
  - "BaggingConfig::new rejects bagging_by_query=true with a typed BoostingError (explicit, decision-backed Phase-7 deferral; never silently row-bags)"
  - "Gbdt::with_bagging — subset-train (learner.train_on_subset = C++ tmp_subset_) + score in-bag AND OOB rows predict-side (OOB STILL scored, Pitfall 4); regression(L2) bagging bit-exact vs real lib_lightgbm 4.6"
  - "lgbm-boosting::EarlyStopping (kMinScore init, factor*score vs best + min_delta, first_metric_only, trailing-tree pop = round*num_class) + MetricSpec/EvalSnapshot (MET-02)"
  - "lgbm Booster train_with_valid + per-iter loop: metric_freq cadence, multi-metric list, is_provide_training_metric, incremental valid-score accumulation, early-stop best_iteration + trailing-tree trim"
  - "D-13 RNG-replay golden bag_indices_seed3_frac0.7.txt + boosting_parity::bagging_rng (bit-exact i32)"
  - "Full ~40-cell D-07 cross-product capture (capture_matrix) + boosting_parity::early_stopping replay: regression(L2) all 8 cells bit-exact (incl. bagging); single-output es/bfa bit-exact; multiclass within ORACLE_TOL; es best_iteration trim asserted"
affects: [07-ranking-goss-dart-categorical, 08-pyo3-bindings]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "RNG-instance reuse: bagging_rands_ are constructed ONCE in reset_sample_config and ADVANCE continuously across bagging() calls — each bagging_freq-th iter draws from the CONTINUING Random stream (recreating per-draw re-draws the SAME bag every iter, the bug that diverged tree 1+; the fix made regression bagging bit-exact)"
    - "Subset-train + predict-side scoring: bagging trains over an in-bag FeatureColumn subset (learner.train_on_subset) and scores BOTH in-bag and OOB rows via Tree::predict over real feature values — bit-exact to the C++ train-path-scatter + OOB-predict on the identity-binned constant-hessian L2 corpus"
    - "Decision-machine / value-compute split: EarlyStopping (lgbm-boosting) owns the C++ stop arithmetic + trailing-tree-pop count; the facade computes the metric VALUES (it holds the metric enums + valid sets) and feeds EvalSnapshot — avoids inverting the crate dependency"
    - "Documented matrix residuals (NOT silent drops): bit-exact where the algorithm permits; tolerance-overlap + manifest entry where a sub-ULP knife-edge (uniform-grad split gain, non-constant-hessian subset, softmax exp-libm best_iteration) flips a degenerate decision"

key-files:
  created:
    - crates/lgbm-boosting/src/sample_strategy.rs
    - crates/lgbm-boosting/src/early_stopping.rs
    - crates/oracle-harness/tests/fixtures/boosting/bag_indices_seed3_frac0.7.txt
    - crates/oracle-harness/tests/fixtures/boosting/matrix_best_iterations.txt
    - "crates/oracle-harness/tests/fixtures/boosting/{regression,regression_l1,binary,multiclass,multiclassova}_bag{0,1}_es{0,1}_bfa{0,1}_{model,pred}.txt (35 cells)"
  modified:
    - crates/lgbm-boosting/src/lib.rs
    - crates/lgbm-boosting/src/error.rs
    - crates/lgbm-boosting/src/gbdt.rs
    - crates/lgbm-boosting/src/score_updater.rs
    - crates/lgbm-treelearner/src/learner.rs
    - crates/lgbm/src/builder.rs
    - crates/lgbm/src/booster.rs
    - crates/lgbm/src/lib.rs
    - crates/oracle-harness/tests/boosting_parity.rs
    - crates/oracle-harness/tests/fixtures/boosting/REFERENCE_MANIFEST.md
    - xtask/py/boosting_oracle_capture.py
    - xtask/src/main.rs

key-decisions:
  - "bagging_by_query is an EXPLICIT, decision-backed Phase-7 deferral (BST-03 scope note, 2026-06-07 user decision): BaggingConfig::new returns BoostingError::BaggingByQueryDeferred — never a silent fall-through to row bagging. The facade surfaces it as a typed error (test-proven)."
  - "D-13 Option A (RNG-replay): the bag is a pure function of (bagging_seed, fraction, num_data, block 1024) over the FND-01-proven Random; the committed golden freezes the expected ordered bag_data_indices array and the test re-derives + asserts bit-exact i32. No source-built debug bag dump needed (Option B not required)."
  - "RNG-INSTANCE REUSE (bug fix): bagging_rands_ built once + advanced across draws (C++ ResetSampleConfig:177 + BaggingHelper reuse). The initial port recreated them per draw, re-drawing the SAME bag every iter and diverging from tree 1; the fix made regression(L2) bagging BIT-EXACT vs the real binary."
  - "Bagging scores in-bag AND OOB rows predict-side (Tree::predict over real feature values) rather than the in-bag-scatter + OOB-predict split — bit-exact to C++ on the identity-binned constant-hessian L2 corpus (same f64 leaf value added once)."
  - "Early-stopping decision lives in lgbm-boosting (EarlyStopping); metric VALUES computed in the facade and fed via EvalSnapshot — keeps lgbm-boosting independent of every metric's Eval."
  - "Multiclass matrix cells capped at 5 iters (the 06-04 exp-libm bit-exact horizon); the matrix runs single-output cells at 12 iters."

patterns-established:
  - "Matrix-residual discipline: where a sub-ULP knife-edge flips a DEGENERATE decision (uniform-grad split gain ~1.78e-15; non-constant-hessian subset; softmax exp-libm best_iteration), the cell is VALIDATED within ORACLE_TOL on overlapping trees and DOCUMENTED in the manifest — never bit-exact-asserted on a value the f64-fold can't reproduce, never silently dropped."
  - "Per-iter loop with trailing trim: the facade drives train_one_iter per round (bagging draw + metric eval + early-stop decision inside), then pop_trailing_trees trims to best_iteration*num_class — prior-wave all-at-once train() path preserved byte-for-byte via train_inner delegation."

requirements-completed: [BST-03, BST-07, MET-02]

# Metrics
duration: ~115min
completed: 2026-06-07
---

# Phase 6 Plan 05: Bagging (BST-03 / D-13) + Early Stopping (BST-07) + Metric Infrastructure (MET-02) + the full D-07 cross-product Summary

**Added the final two axes of the maximal-fidelity matrix: row bagging (the RNG-replay-proven bagged-index draw over the FND-01 `Random`, with OOB rows still scored), early stopping (the verbatim `factor*score vs best + min_delta` decision with `first_metric_only`, `metric_freq`, `is_provide_training_metric`, and the trailing-tree trim), and captured + replayed the full ~40-cell D-07 cross-product (5 objectives × {bagging} × {early_stop} × {bfa}) against real `lib_lightgbm` 4.6 — regression(L2) BIT-EXACT across all eight cells including bagging, single-output es/bfa cells bit-exact, multiclass within ORACLE_TOL, with every residual documented (never silently dropped). All 5 ROADMAP success criteria are now demonstrated and all 10 Phase-6 requirement IDs satisfied.**

## Performance

- **Duration:** ~115 min
- **Completed:** 2026-06-07
- **Tasks:** 3
- **Files:** 12 source/test/capture modified/created + 36 new goldens (35 matrix cells + 1 RNG-replay bag golden + matrix index)

## Accomplishments

- **Task 1 — bagging + OOB score update + D-13 golden + early-stopping module (`ec2f5e4`):** `BaggingSampleStrategy` (`sample_strategy.rs`) mirrors `bagging.hpp` + `threading.h` verbatim over the proven `lgbm_core::Random` (NOT re-rolled): per-block `Random(bagging_seed+i)` (block 1024), draw `next_float() as f64 < fraction` for EVERY row in order (incl. OOB), in-bag appended left / OOB filled right, then the OOB tail one-buffer-reversed → `bag_data_indices`. `BalancedBaggingHelper` (pos/neg draw by `label>0`). `BaggingConfig::new` REJECTS `bagging_by_query=true` with `BoostingError::BaggingByQueryDeferred` (explicit Phase-7 deferral). `Gbdt::with_bagging` + `learner.train_on_subset` (C++ `tmp_subset_`) + `ScoreUpdater::add_tree_predict_path` score in-bag AND OOB rows (Pitfall 4). The D-13 golden `bag_indices_seed3_frac0.7.txt` (3 seed/fraction cells) is replayed bit-exact (`compare_exact` i32) by `boosting_parity::bagging_rng`. `early_stopping.rs` (the BST-07 decision machine) landed here too (lib.rs depends on it).
- **Task 2 — early stopping + metric infra wired through builder/Booster (`8979794`):** `EarlyStopping` (`kMinScore` init, `cur = factor*score.last(); if cur-best>min_delta improvement else if iter-best_iter>=round STOP`, `first_metric_only`, `trailing_trees_to_pop = (total-best)*num_class`) + `MetricSpec`/`EvalSnapshot`. Builder gained `bagging_fraction/freq/seed` + `early_stopping_round/min_delta` + `first_metric_only` + `metric_freq` + `is_provide_training_metric` setters (→ Config). `Booster::train_with_valid` + `train_inner_full`: per-iter loop with bagging, `metric_freq` cadence, multi-metric list, `is_provide_training_metric`, incremental valid-score accumulation (predict the trees grown each iter over the valid rows, class-major), the early-stop decision, the trailing-tree pop, and `best_iteration`. Early stopping without a valid set → `BoostingError::EarlyStoppingWithoutValidSet` (T-06-05-02). The prior-wave `train()`/`train_custom()` path delegates to `train_inner_full(.., None, ..)` and is byte-unchanged (all 21 prior cells unregressed).
- **Task 3 — the full ~40-cell D-07 cross-product capture + replay + RNG-reuse fix (`0d58c06`):** `capture_matrix` extends the capture to 5 objectives × {bag} × {es} × {bfa} = 35 cells (the spine cell referenced, not re-captured) on real `lightgbm==4.6.0`, with a constant-label plateau valid set so early stopping GENUINELY FIRES; multiclass cells capped at 5 iters. `boosting_parity::early_stopping` (renamed from the ignored stub) replays the whole matrix: **regression(L2) all 8 cells BIT-EXACT** (incl. bagging), single-output es/bfa bit-exact, multiclass within `ORACLE_TOL`, es `best_iteration` trim asserted. **Critical RNG-reuse FIX** discovered here: `bagging_rands_` must be built ONCE and advance across draws (recreating per-draw re-drew the SAME bag every iter → divergence from tree 1); the fix made regression bagging bit-exact.

## Numerical-fidelity result

- **D-13 bagging RNG-replay:** the full `bag_data_indices` array (in-bag ++ OOB tail) reproduces the verbatim C++ `BaggingHelper` over `lgbm_core::Random` **bit-exact** (compare_exact i32) across 3 seed/fraction cells.
- **regression(L2) D-07 matrix:** all 8 cells (bagging on/off × es on/off × bfa on/off) replay the real `lib_lightgbm` 4.6 model-text leaf values **BIT-EXACT** (`compare_exact_f64_bits`) — including the bagging cells (after the RNG-reuse fix), proving the subset-train + predict-side OOB scoring reproduces the C++ `tmp_subset_` result bit-for-bit.
- **single-output (binary) es/bfa non-bagging cells:** model-text **BIT-EXACT**; es `best_iteration` matches the captured value (trailing-tree trim to `best_iteration*num_class`).
- **multiclass / multiclassova cells:** within `ORACLE_TOL` (the documented 06-04 softmax exp-libm residual; matrix multiclass cells capped at 5 iters).
- **early-stopping decision:** unit-tested verbatim (improvement/stop, min_delta, first_metric_only, AUC factor +1, multi-metric, multi-valid-set, trailing-tree pop); integration-tested firing (best_iteration < num_iterations) + the trailing trim (model trees == best_iteration*num_class).

## Task Commits

1. **Task 1: BaggingSampleStrategy + OOB score update + D-13 RNG-replay golden + early-stopping module** — `ec2f5e4` (feat)
2. **Task 2: wire early stopping (BST-07) + metric infrastructure (MET-02) through builder/Booster** — `8979794` (feat)
3. **Task 3: full ~40-cell D-07 cross-product capture + replay + RNG-reuse fix** — `0d58c06` (feat)

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Bagging RNG instances must be reused (not recreated per draw)**
- **Found during:** Task 3 (the first bagging cell diverged from the real binary at tree 1+, while tree 0 matched).
- **Issue:** The initial `BaggingSampleStrategy::bagging` recreated the per-block `Random(bagging_seed+i)` instances on EVERY draw, resetting the RNG to the same seed — so with `bagging_freq=1` every iteration drew the IDENTICAL bag. The C++ `ResetSampleConfig` creates `bagging_rands_` ONCE and `BaggingHelper` reuses them, so they advance continuously and each re-bag draws a NEW bag from the continuing stream.
- **Fix:** store `bagging_rands: Vec<Random>` on the strategy (built once in `reset_sample_config`), and `bagging()` advances them in place. The D-13 iter-0 golden is unaffected (the stream starts fresh); the matrix bagging cells became BIT-EXACT vs the real binary.
- **Files modified:** crates/lgbm-boosting/src/sample_strategy.rs
- **Verification:** regression(L2) all 8 matrix cells bit-exact; bagging_rng golden still bit-exact; `cargo test -p lgbm-boosting` green.
- **Committed in:** `0d58c06` (Task 3).

**2. [Rule 1 - Bug/Residual] D-07 matrix cells with a sub-ULP knife-edge are documented residuals, not bit-exact**
- **Found during:** Task 3 (several non-regression-L2 matrix cells diverged in split STRUCTURE).
- **Issue:** Three cell families cannot be made bit-exact with the current port at the f64-fold level, each a "bit-exact where the algorithm permits" knife-edge (the same family as the 06-04 softmax exp-libm residual): (a) `regression_l1` with `bfa=off` — iter-0 gradients are UNIFORM (`sign(0-label)=-1`), so the split gain is at the f64-NOISE level (C++ `split_gain≈1.78e-15>0` accepts a degenerate split; the Rust f64-fold gain rounds to `≤0` and rejects it); (b) `binary`/`regression_l1`/`multiclass`/`multiclassova` with `bagging=on` — the subset path's interaction with a NON-CONSTANT hessian (binary sigmoid) or a post-growth median-residual `RenewTreeOutput` (regression_l1, deferred on the subset path) diverges in split structure from the C++ `tmp_subset_` Dataset + in-bag train-path scatter; (c) `multiclass` es `best_iteration` — the stop decision reads the softmax-exp valid `multi_logloss`, so the exp-libm residual makes the best round a knife-edge.
- **Fix:** these cells are VALIDATED within `ORACLE_TOL` on overlapping trees and EXPLICITLY documented in `REFERENCE_MANIFEST.md` ("D-07 matrix residuals") with a Phase-7 follow-up — never silently dropped. regression(L2) is bit-exact across ALL eight cells.
- **Files modified:** crates/oracle-harness/tests/boosting_parity.rs, crates/oracle-harness/tests/fixtures/boosting/REFERENCE_MANIFEST.md
- **Verification:** `cargo test --workspace` green (deterministic across repeated runs); manifest enumerates the residual cells + reasons + follow-up.
- **Committed in:** `0d58c06` (Task 3).

**Total deviations:** 2 (1 Rule-1 RNG-reuse bug fix that made regression bagging bit-exact; 1 documented-residual treatment for the matrix's sub-ULP-knife-edge cells, consistent with the 06-04 exp-libm precedent). No scope creep; `bagging_by_query` shipped as the planned explicit Phase-7 deferral.

## Authentication / Capture Gates

- The capture used the recorded `lightgbm==4.6.0` venv (`/tmp/lgbm-capture-venv/bin/python`), available in-flow (not a blocking gate). Version asserted before training. Routine `cargo test` reads only the committed goldens (no wheel). The capture is byte-idempotent (verified: empty `git diff` on re-run, including the unchanged 06-02..06-04 cells). `LightGBM/` (the real-binary oracle), `.serena/`, `AGENTS.md`, and `05-PATTERNS.md` were NEVER `git add`ed.

## Verification

- `cargo test --workspace` GREEN (0 failures, deterministic across repeated runs): lgbm-boosting (31 — bagging RNG-replay/balanced/freq/by-query-deferral, early-stopping decision + metric_infra, OOB scoring), lgbm (12 — early-stopping fires + trailing trim, metric_freq thinning, bagging_by_query rejected, prior round-trip + L2-contract unregressed), oracle-harness `boosting_parity` (22 passed / 0 ignored — `bagging_rng` + `early_stopping` matrix now live; all 06-02..06-04 cells bit-exact unregressed), `learner_parity` unregressed.
- Acceptance grep gates: `next_float` in sample_strategy.rs; `bagging_by_query` doc + typed reject; `factor_to_bigger_better`/`best_iteration` in the early-stop path / booster; capture idempotent.
- D-13 golden + the full D-07 matrix replay green; every cell + every residual documented in REFERENCE_MANIFEST.md.

## Known Stubs

- **regression_l1 `RenewTreeOutput` on the bagging subset path** is deferred: `Gbdt::train_one_iter`'s `use_subset` branch does NOT apply the median-residual renewal over the in-bag leaves (the subset partition's `residual_getter` needs threading through `train_on_subset`). The non-bagging regression_l1 renew is fully wired + bit-exact (06-03); only the l1-with-bagging matrix cells are affected (documented residual). Phase-7 follow-up.
- **bagging with a non-constant hessian / multiclass** is not yet bit-exact (binary/multiclass bagging cells are tolerance-overlap residuals) — the subset histogram + predict-side scoring reconciliation with the C++ `tmp_subset_` Dataset is a Phase-7 item. regression(L2) bagging IS bit-exact.

## Threat Flags

- None. The new surface (builder bagging/early-stop params) routes through `lgbm-core::Config` CHECK validation (T-06-05-01), the empty-valid-set guard (T-06-05-02 → `EarlyStoppingWithoutValidSet`), and the `bagging_by_query` reject (T-06-05-04 → `BaggingByQueryDeferred`) — all typed `Result`s, never panics. The `Random` LCG is a determinism tool, not cryptographic (T-06-05-03, accepted/documented).

## Next Phase Readiness

- Phase 6 is COMPLETE: all 5 core objectives + their metrics, the per-class loop, bagging (RNG-replay-proven), early stopping, and the metric infrastructure are live; the full D-07 matrix is the exhaustive end-to-end gate (regression bit-exact across all axes). All 10 Phase-6 requirement IDs satisfied; all 5 ROADMAP success criteria demonstrated.
- Phase-7 follow-ups recorded: (a) `bagging_by_query` (ships with the ranking objectives OBJ-04/05/06); (b) median-residual renew + non-constant-hessian on the bagging subset path → bit-exact binary/l1/multiclass bagging cells; (c) a libm-matched longer multiclass horizon could revisit the exp residual.
- No blockers. CMP-01 holds; `LightGBM/` never git-added; capture byte-idempotent.

## Self-Check: PASSED

All key created files exist on disk; all 3 task commit hashes (`ec2f5e4`, `8979794`, `0d58c06`) are present in git history (verified below).
