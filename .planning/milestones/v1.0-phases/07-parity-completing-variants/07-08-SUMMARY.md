---
phase: 07-parity-completing-variants
plan: 08
subsystem: treelearner
tags: [categorical, tree-learner, split-finding, one-hot, many-vs-many, bit-exact, lightgbm-4.6]

# Dependency graph
requires:
  - phase: 05-tree-learner-split-finding
    provides: the bit-exact serial tree-learner spine (find_best_split, scan_leaf_histogram, data_partition, the offset/compaction conventions)
  - phase: 03-model-format (DAT-08)
    provides: Tree categorical predict path (categorical_decision / find_in_bitset) + cat_boundaries/cat_threshold model-text parse
  - phase: 02-binning
    provides: bit-exact categorical binning (bin_2_categorical_, categorical_2_bin_, most_freq_bin/default_bin rules)
provides:
  - "Categorical split finding (one-hot + many-vs-many) as a purely-additive bin_type branch in the serial learner"
  - "Tree::split_categorical + grown-tree categorical model-text emit/round-trip"
  - "DataPartition::split_categorical (category-value routing)"
  - "GainConfig + FeatureColumn categorical fields; builder cat_* setters"
  - "xtask categorical-oracle-capture + real lib_lightgbm 4.6 categorical goldens"
  - "learner_parity categorical_onehot / categorical_manyvsmany layered cells + the D-06 no-regression gate"
affects: [07-09 (ranking), 07-10 (predict-modes), 07-11 (learner constraints)]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Additive bin_type dispatch at the TOP of the per-feature scan loop — numeric spine byte-untouched (D-06)"
    - "Categorical winner bitset carried in a per-leaf RefCell side-structure (kept OUT of the Copy SplitInfo)"
    - "Category-value partition routing (equivalent to C++ inner-bin-bitset, consistent with the predict path)"
    - "JSON sidecar pinning per-row bins + bin_2_categorical so the bit-exact gate isolates the split logic, never binning"

key-files:
  created:
    - crates/lgbm-treelearner/src/feature_histogram_categorical.rs
    - xtask/py/categorical_oracle_capture.py
    - crates/oracle-harness/tests/fixtures/categorical/{cat_onehot,cat_manyvsmany}.{txt,bins.json}
  modified:
    - crates/lgbm-treelearner/src/learner.rs
    - crates/lgbm-treelearner/src/data_partition.rs
    - crates/lgbm-model/src/tree.rs
    - crates/lgbm-compute/src/gain.rs
    - crates/lgbm/src/builder.rs
    - xtask/src/main.rs
    - crates/oracle-harness/tests/learner_parity.rs

key-decisions:
  - "Carry the categorical winner bitset in a per-leaf RefCell<Vec<Option<Vec<u32>>>> rather than widening the Copy SplitInfo with a Vec"
  - "Partition categorical splits by CATEGORY VALUE through bin_to_category (provably equivalent to the C++ inner-bin-bitset routing, and identical to the predict path) — avoids the bin/offset off-by-one bookkeeping"
  - "Pin the categorical bin layout via a JSON sidecar (per-row bins + bin_2_categorical) so the bit-exact tree comparison can only falsify split/gain logic, never binning"

patterns-established:
  - "Additive variant branch under a strict bit-exact invariant: dispatch around the spine, never through it; an explicit no-regression gate proves the spine is untouched"

requirements-completed: [TRL-06]

# Metrics
duration: ~95min
completed: 2026-06-07
---

# Phase 7 Plan 08: Categorical Splits (TRL-06) Summary

**Categorical split finding (one-hot + many-vs-many gain math) added as a purely-additive bin_type branch in the bit-exact serial tree learner, validated bit-exact vs real lib_lightgbm 4.6 on a categorical corpus while the numeric spine stays byte-untouched and bit-exact (D-06).**

## Performance

- **Duration:** ~95 min
- **Started:** 2026-06-07
- **Completed:** 2026-06-07
- **Tasks:** 4 (3 code/test + 1 human-gated capture, satisfied by the available 4.6.0 venv)
- **Files modified:** 7 modified + 3 created (1 module, 1 py script, 4 fixtures)

## Accomplishments
- `find_best_threshold_categorical` (feature_histogram.cpp:143-382): one-hot one-vs-rest + many-vs-many sorted-by-ctr group scan, with the deliberate `l2 += cat_l2` asymmetry (cat_l2 only in per-category gain, original l2 in gain_shift).
- bin_type dispatch at the TOP of the per-feature scan loop: Numerical → the byte-untouched continuous spine; Categorical → the additive finder.
- `Tree::split_categorical` + grown-tree categorical model-text emit; byte-stable grow→to_string→parse round-trip.
- `DataPartition::split_categorical` (category-value routing) + the categorical `split_inner_categorical` node growth.
- builder cat_l2/cat_smooth/min_data_per_group/max_cat_threshold/max_cat_to_onehot setters; GainConfig + FeatureColumn categorical fields.
- xtask `categorical-oracle-capture` + py script; captured one-hot + many-vs-many real lib_lightgbm 4.6 goldens (byte-idempotent); the layered learner_parity cells flip skip→GREEN bit-exact.
- D-06 HELD: spine_real/mfb_pos/growth_path_subtract + kernel_parity bit-exact; an explicit no-regression gate GREEN.

## Task Commits

1. **Task 1: Config + bin_type dispatch + find_best_threshold_categorical** - `4beb4ca` (feat)
2. **Task 2: Tree::split_categorical + model-text round-trip** - `9b0b1fd` (feat)
3. **Task 3: Builder + capture + layered cells + no-regression gate** - `69f8fb1` (feat)
4. **Task 4: Capture the categorical corpus + partition fix** - `50bc6fb` (test)

_Task 1 also folded in the FeatureColumn/GainConfig `Default` impls + literal-default fixes across call sites (so the spine sites stay unchanged)._

## Files Created/Modified
- `crates/lgbm-treelearner/src/feature_histogram_categorical.rs` (new) — `find_best_threshold_categorical` (one-hot + many-vs-many) + `construct_bitset`.
- `crates/lgbm-treelearner/src/learner.rs` — bin_type dispatch, `split_inner_categorical`, per-leaf categorical bitset side-structure, FeatureColumn `bin_type`/`bin_to_category` + `Default`.
- `crates/lgbm-treelearner/src/data_partition.rs` — `split_categorical` (category-value routing).
- `crates/lgbm-model/src/tree.rs` — `Tree::split_categorical` + round-trip tests.
- `crates/lgbm-compute/src/gain.rs` — GainConfig cat fields + `Default`.
- `crates/lgbm/src/builder.rs` — cat_* setters + test.
- `xtask/src/main.rs`, `xtask/py/categorical_oracle_capture.py` — categorical capture subcommand + script.
- `crates/oracle-harness/tests/learner_parity.rs` — categorical cells + no-regression gate + sidecar parser.
- `crates/oracle-harness/tests/fixtures/categorical/*` — real lib_lightgbm 4.6 goldens + sidecars.

## Decisions Made
- Categorical winner bitset lives in a per-leaf `RefCell` side-structure, keeping `SplitInfo` `Copy`.
- Categorical partition routes by category value (via `bin_to_category`) — equivalent to the C++ inner-bin-bitset, consistent with the predict path.
- Categorical bin layout pinned via a JSON sidecar so the gate isolates the split/gain logic.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Categorical data-partition off-by-one (all rows on one side)**
- **Found during:** Task 4 (running the captured categorical cells)
- **Issue:** The first implementation of `DataPartition::split_categorical` transcribed the C++ `DenseBin::SplitCategoricalInner<USE_MIN_BIN=false>` inner-bin-bitset routing (`bin - 1 + offset`) but, combined with the finder's `cat_threshold = bin` and the most_freq_bin/offset this corpus realizes, produced a consistent off-by-one that routed every row to one side (`leaf_count [0,0,0,40]`), causing the learner to keep splitting empty leaves (num_cat 3 vs golden 1).
- **Fix:** Route the partition by CATEGORY VALUE through `bin_to_category` against the real category bitset — the row-partition analog of `Tree::CategoricalDecision`, provably equivalent to the C++ inner-bin routing and guaranteed consistent with the serialized bitset + predict path. Result: `leaf_count [10,30]`, bit-exact to the golden.
- **Files modified:** crates/lgbm-treelearner/src/data_partition.rs, crates/lgbm-treelearner/src/learner.rs
- **Verification:** `learner_parity_categorical_onehot` + `_manyvsmany` GREEN bit-exact; predict-side routing unchanged.
- **Committed in:** `50bc6fb` (Task 4 commit)

**2. [Rule 3 - Blocking] Sidecar JSON float-array parsing**
- **Found during:** Task 4
- **Issue:** The hand JSON parser parsed the `grad` array (floats) with the int parser, panicking.
- **Fix:** Added a `float_array` parser for `grad`.
- **Files modified:** crates/oracle-harness/tests/learner_parity.rs
- **Committed in:** `50bc6fb` (Task 4 commit)

---

**Total deviations:** 2 auto-fixed (1 bug, 1 blocking)
**Impact on plan:** Both essential for the categorical gate to pass; no scope creep. The numeric spine was never touched (D-06 held throughout).

## Issues Encountered
- The categorical bin↔offset bookkeeping (finder `cat_threshold = bin` vs the C++ partition `bin - min_bin + offset`) is subtle and corpus-dependent; resolved by routing the partition on category value (Deviation 1), which is both simpler and provably faithful.

## User Setup Required
None - the capture used the pre-provisioned `/tmp/lgbm-capture-venv` (lightgbm 4.6.0). The pip lightgbm is a capture-time tool only; `cargo test` does not need it.

## Next Phase Readiness
- Categorical splits land bit-exact with the spine intact — unblocks Wave 6 (07-09 ranking, 07-10 predict-modes, 07-11 learner constraints).
- DEF-07-02 (objective-side learner knife-edges) remains the only ignored set; untouched by this plan.

---
*Phase: 07-parity-completing-variants*
*Completed: 2026-06-07*

## Self-Check: PASSED
- All created files exist on disk (feature_histogram_categorical.rs, categorical_oracle_capture.py, cat_onehot/cat_manyvsmany goldens, SUMMARY).
- All 4 task commits present in git history (4beb4ca, 9b0b1fd, 69f8fb1, 50bc6fb).
- `cargo test --workspace`: 567 passed / 13 ignored / 0 failed; D-06 spine goldens + kernel_parity bit-exact; categorical cells GREEN.
