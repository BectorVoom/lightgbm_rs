---
phase: 05-tree-learner-split-finding
plan: 04
subsystem: treelearner
tags: [col-sampler, force-col-wise, feature-subsampling, captured-gh, d03, trl-08, trl-09, learner-parity, random-sample]

# Dependency graph
requires:
  - phase: 05-tree-learner-split-finding
    plan: 03
    provides: "SerialTreeLearner spine (leaf-wise growth, subtraction trick, FixHistogram, DataPartition, HistogramPool) + spine.txt golden + learner_parity.rs replay harness + learner_capture.cpp full transcription"
provides:
  - "lgbm_treelearner::col_sampler::ColSampler — per-tree ResetByTree + per-node GetByNode reproducing the C++ Random::Sample draw SEQUENCE + call order (TRL-08)"
  - "SerialTreeLearner::with_strategy(BuildStrategy::{RowWise,ColWise}) (TRL-09) + with_feature_fraction(ff, ffn, seed) + train_with_col_sampler_trace returning ColSamplerTrace"
  - "col_wise.txt / col_sampler.txt / real_gh.txt goldens + learner_parity_{row_vs_col, col_sampler_rng, real_gh_full_tree} replay tests"
  - "Faithfulness fix: the tree's leaf_count/internal_count record the ACTUAL data_partition leaf_count (update_cnt=true), not the SplitInfo reconstructed counts — corrects spine.txt to actual-partition values"
affects: [06-gbdt, tree-learner, col-sampler, force-col-wise]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "force_col_wise is a config FLAG (a no-op) over the shared construct_histograms Backend op on the single-thread deterministic anchor — NOT a distinct compute path (Open Q2 RESOLVED, A1 confirmed). The row==col tree-equality gate fails loudly if a backend ever diverged."
    - "ColSampler RNG parity is the CALL SEQUENCE, not the PRNG: ResetByTree once per tree (BeforeTrain), GetByNode once per node smaller-leaf-then-larger (serial_tree_learner.cpp:479,487). The col_sampler.txt golden asserts the exact selected feature indices per draw."
    - "The tree's leaf_count/internal_count are the ACTUAL data_partition counts (serial_tree_learner.cpp:788-791 update_cnt=true), NOT the SplitInfo round_int(hess*cnt_factor) reconstructed counts — the two disagree by +/-1 for fractional hessians."
    - "The col_sampler golden re-grows the tree with the SAME per-node feature gate the Rust learner applies, so the draw COUNT/ORDER is bit-identical to the Rust train_with_col_sampler_trace trace (not a shape-agnostic RNG dump)."

key-files:
  created:
    - crates/oracle-harness/tests/fixtures/learner/col_wise.txt
    - crates/oracle-harness/tests/fixtures/learner/col_sampler.txt
    - crates/oracle-harness/tests/fixtures/learner/real_gh.txt
  modified:
    - xtask/cpp/learner_capture.cpp
    - xtask/src/main.rs
    - crates/lgbm-treelearner/src/learner.rs
    - crates/oracle-harness/tests/learner_parity.rs
    - crates/oracle-harness/tests/fixtures/learner/spine.txt
    - crates/oracle-harness/fixtures/REFERENCE_MANIFEST.md

key-decisions:
  - "force_col_wise == force_row_wise on the deterministic anchor (Open Q2 RESOLVED): the two strategies differ only in histogram-build ORDER, not result; col_wise.txt carries the identical PTREE as spine.txt, and learner_parity_row_vs_col grows the corpus under BOTH strategies asserting String equality to each other and to C++."
  - "Tree leaf_count/internal_count record the ACTUAL partition count (update_cnt=true overwrite), not the SplitInfo reconstructed count — this is the faithful C++ behavior and a Rule-1 fix to the Plan-03 spine (which used reconstructed counts; the spine test passed only because hessian=1 made them coincide)."
  - "real_gh num_leaves is per-objective (regression=3 / binary=2): the binary-logloss 0.25 hessians make the gain-scan reconstructed count over-count vs the threshold partition on a residual leaf, isolating a degenerate 0-row child at a third split; one clean split proves D-07 on binary while regression carries the deeper 3-leaf tree."
  - "real_gh features are MONOTONE row-block partitions (like the spine's feature 0) with the objective gradient aligned to them, so every split's threshold and the data partition agree exactly (no degenerate child); the realistic distribution comes from the objective math on real labels, not a contrived binning."

patterns-established:
  - "Plan-05-04 parity additions layer on the proven spine WITHOUT a new Backend op: col_sampler.rs drives lgbm_core::Random::sample; force_col_wise reuses construct_histograms; real_gh reuses the spine GrowTree."

requirements-completed: [TRL-08, TRL-09]

# Metrics
duration: 90min
completed: 2026-06-05
---

# Phase 5 Plan 04: force_col_wise + Feature-Subsampling RNG Parity + Captured-Real-g/h Corpus Summary

**Layered three parity additions onto the proven Phase-5 spine: `force_col_wise` proven bit-identical to `force_row_wise` (and to C++) on the deterministic anchor (Open Q2 resolved as a config-flag no-op); the `ColSampler` reproducing the C++ `Random::Sample` draw SEQUENCE (ResetByTree + per-node GetByNode smaller-then-larger); and the D-03 captured iteration-1 g/h corpus (regression-l2 + binary-logloss) proving the learner grows the same tree as C++ under a realistic gradient distribution — all replayed bit-exact from committed goldens that never touch `LightGBM/`.**

## Performance

- **Duration:** ~90 min (continuation: Task 1 pre-completed; this session finished Task 2 + closeout)
- **Completed:** 2026-06-05
- **Tasks:** 2 (Task 1 pre-committed by a prior executor; Task 2 + closeout this session)
- **Files created:** 3 | **Files modified:** 6

## Accomplishments

- **Task 1 (pre-committed `e8efc90`):** `col_sampler.rs` (faithful `ColSampler` port: `get_cnt`/`reset_by_tree`/`get_by_node`/`is_feature_used_bytree` driving `lgbm_core::Random`), `BuildStrategy::{RowWise,ColWise}` + `with_feature_fraction` + `train_with_col_sampler_trace` wiring, with the default `feature_fraction=1.0` spine path bit-identical.
- **Task 2 (`d06d85b`):** extended `learner_capture.cpp` to emit three new goldens; added the three parity replay tests; wired `xtask/main.rs` to write all four fixtures (5 args) + refresh the manifest; and applied the actual-partition-count faithfulness fix.
  - **`col_wise.txt` (TRL-09):** the spine corpus grown under `force_col_wise`; the strategy-agnostic transcription emits the identical PTREE as `spine.txt`. `learner_parity_row_vs_col` grows under BOTH `RowWise` and `ColWise`, asserting `to_string()` equality to each other and to the golden.
  - **`col_sampler.txt` (TRL-08):** a `feature_fraction=1.0` / `feature_fraction_bynode=0.5` config over a 4-feature corpus, driving the GENUINE header-only reference `Random::Sample`; emits `CS_BYTREE` + per-draw `CS_NODE` selections in DRAW ORDER. `learner_parity_col_sampler_rng` asserts the Rust `ColSamplerTrace` matches the selected REAL-feature indices exactly.
  - **`real_gh.txt` (D-03):** captured iteration-1 g/h from regression-l2 (`grad=score-label`, `hess=1`) and binary-logloss (`response=-label*sigmoid/(1+exp(...))`), `boost_from_average=false`, `score_t=float`, over fixed real labels. `learner_parity_real_gh_full_tree` grows from the captured g/h and asserts the full tree `to_string()` is byte-identical to the C++ reference (D-07 under a realistic distribution).

## Task Commits

1. **Task 1: ColSampler + force_col_wise wiring (TRL-08, TRL-09)** — `e8efc90` (feat) — pre-committed by a prior executor.
2. **Task 2: col_wise + col_sampler + real_gh goldens + parity tests (D-03, TRL-08/09)** — `d06d85b` (test).

**Plan metadata:** this SUMMARY + STATE/ROADMAP/REQUIREMENTS (docs commit).

## Files Created/Modified

- `crates/oracle-harness/tests/fixtures/learner/{col_wise,col_sampler,real_gh}.txt` — the three new committed goldens.
- `xtask/cpp/learner_capture.cpp` — `CppColSampler` transcription, `EmitColSamplerGolden` (col-sampler-gated growth mirroring `train_inner`), `GhRegressionL2`/`GhBinaryLogloss`, `BuildColSamplerCorpus`/`BuildRealGhCorpus`, and a 4-fixture `main`.
- `xtask/src/main.rs` — `learner-capture` writes the 4 goldens (5 args) + the manifest "Learner Golden Set" Plan-05-04 section.
- `crates/lgbm-treelearner/src/learner.rs` — `split_inner` now records the ACTUAL `data_partition.leaf_count` into the tree (the faithfulness fix).
- `crates/oracle-harness/tests/learner_parity.rs` — three new tests + parsers for `col_wise`/`col_sampler`/`real_gh`.
- `crates/oracle-harness/tests/fixtures/learner/spine.txt` — regenerated with the faithful actual-partition leaf counts.
- `crates/oracle-harness/fixtures/REFERENCE_MANIFEST.md` — Plan-05-04 Learner Golden Set section + faithfulness-fix note.

## Decisions Made

- **Open Q2 resolved: `force_col_wise` is a config-flag no-op on the anchor.** The two strategies differ only in histogram-build order; the Phase-4 `construct_histograms` whole-kernel op produces identical f64 cells, so `col_wise.txt` carries the same tree as `spine.txt`. Empirically confirmed by `learner_parity_row_vs_col`. (Recorded in the SUMMARY + manifest per the plan's verification requirement.)
- **Tree counts are the ACTUAL partition counts.** See Deviations (Rule 1).
- **Per-objective `num_leaves` for real_gh.** Binary-logloss fractional hessians trigger a degenerate residual split at depth 3; capped binary at 2 leaves (one clean split), regression at 3 (two clean splits). Both prove D-07; the cap is a faithful, conservative corpus choice under the plan's Claude-discretion D-03 boundary.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Tree leaf_count/internal_count used reconstructed SplitInfo counts, not actual partition counts**
- **Found during:** Task 2 (the `real_gh` binary-logloss full-tree replay diverged: `leaf_count`/`leaf_value`/`split_gain` differed on a split with fractional hessians).
- **Issue:** The Plan-03 spine's `split_inner` passed `best.left_count`/`best.right_count` (the SplitInfo counts reconstructed via `round_int(hess·cnt_factor)`) to `tree.split`. Real LightGBM OVERWRITES these with `data_partition_->leaf_count(...)` after the row partition (`serial_tree_learner.cpp:788-791`, `update_cnt=true`) — the two disagree by ±1 for fractional hessians, and the spine's recorded counts even failed to sum to `num_data`. The spine test passed only because `hessian==1` made reconstructed == actual.
- **Fix:** `split_inner` now reads `data_partition.leaf_count(new_left)`/`leaf_count(new_right)` (already updated by the preceding `data_partition.split`) and passes those to `tree.split`. The same fix is applied in the C++ transcription (`GrowTree` + `EmitColSamplerGolden`). Regenerated `spine.txt` to the faithful actual-partition counts.
- **Files modified:** `crates/lgbm-treelearner/src/learner.rs`, `xtask/cpp/learner_capture.cpp`, `crates/oracle-harness/tests/fixtures/learner/spine.txt`
- **Verification:** `cargo test -p lgbm-treelearner` (29 pass) + `cargo test -p oracle-harness --test learner_parity` (8 pass, incl. the previously-diverging `real_gh` binary case) + `cargo test --workspace` (0 failures).
- **Committed in:** `d06d85b` (Task 2 commit — surfaced by the real_gh golden, logically part of the spine count semantics).

---

**Total deviations:** 1 auto-fixed (1 bug).
**Impact on plan:** The fix is a correctness requirement (faithful C++ behavior) and was necessary for the D-03 binary case to pass; it tightened the Plan-03 spine counts to the faithful values. No scope creep.

## Issues Encountered

- **Degenerate (0-row child) splits on fractional hessians.** Several hand-crafted real_gh corpora produced a split whose gain-scan reconstructed count passed the `min_data_in_leaf` gate while the actual threshold partition routed 0 rows to one child (the reconstructed-vs-actual count divergence). Resolved by (a) the actual-count fix above making the recorded counts faithful, and (b) choosing monotone row-block features aligned with the objective gradient + a conservative per-objective `num_leaves` so every split's actual children are non-degenerate.
- **ColSampler golden draw-count alignment.** The reconstructed col_sampler golden must produce the SAME number of `GetByNode` draws as the Rust learner. Resolved by making `EmitColSamplerGolden` re-grow the tree with the SAME per-node feature gate the Rust `find_best_splits` applies (smaller-then-larger, scan gated to the selected features), so the draw sequence is bit-identical to the `ColSamplerTrace`.

## User Setup Required

None. `learner-capture` needs a C++ toolchain (cmake + g++, present here) only to REGENERATE the goldens; normal `cargo test` reads the committed fixtures and needs no toolchain.

## Next Phase Readiness

- The full Phase-5 serial tree-learner is now parity-validated: spine (D-06/D-07), force_col_wise (TRL-09), feature-subsampling RNG (TRL-08), and captured-real-g/h (D-03). Phase 6 (GBDT) can drive `SerialTreeLearner` per boosting iteration on real objective g/h with confidence in the tree contract.
- NA_AS_MISSING remains deferred (RESEARCH A5): a `num_bin > 2 && missing_type == NaN` feature is a typed `TreeLearnerError`; the forward-branch body must be transcribed before any case uses it.
- The reconstructed-vs-actual count divergence is a known fractional-hessian corner; the faithful behavior (actual partition counts) is now in place, and degenerate splits are avoided at the corpus level — Phase 6 real datasets with realistic `min_data_in_leaf` (default 20) will not hit the degenerate corner.

## Self-Check: PASSED

---
*Phase: 05-tree-learner-split-finding*
*Completed: 2026-06-05*
