---
phase: 05-tree-learner-split-finding
plan: 03
subsystem: treelearner
tags: [serial-tree-learner, leaf-wise, subtraction-trick, fix-histogram, data-partition, histogram-pool, spine, d06, d07, d02a, learner-parity]

# Dependency graph
requires:
  - phase: 05-tree-learner-split-finding
    plan: 01
    provides: "Backend::find_best_split widened with authoritative skip_default_bin/na_as_missing flags"
  - phase: 05-tree-learner-split-finding
    plan: 02
    provides: "lgbm-treelearner crate skeleton + TreeLearnerError + split_gt re-export; Tree::split + growth arrays; learner-capture xtask + learner_parity.rs scaffold"
provides:
  - "lgbm_treelearner::SerialTreeLearner — the leaf-wise (best-first) growth loop driving the Phase-4 Backend ops into a tree bit-faithful to C++ SerialTreeLearner::Train (force_row_wise, feature_fraction=1.0, missing_type=None)"
  - "lgbm_treelearner::{LeafSplits, fix_histogram, DataPartition, HistogramPool} — the load-bearing bookkeeping (ordered f64 fold, most-freq-bin reconstruct, leaf row ranges, pool get/move/eviction)"
  - "lgbm_compute::ComputeClientReexport — ComputeClient re-export so the learner names the Backend client arg without depending on cubecl (CMP-01)"
  - "spine.txt golden (per-split D-06 per-bin gains + per-tree D-07 field set) + learner_parity.rs bit-exact replay incl. D-02a kernel-vs-learner cross-check"
affects: [05-04, 06-gbdt, tree-learner-spine, serial_tree_learner]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Smaller/larger child LeafSplits seeded by the ACTUAL DataPartition row count (not SplitInfo left/right_count) so it agrees bit-for-bit with the smaller-child selection in find_best_splits (Pitfall 3)"
    - "D-07 full-tree parity: the C++ golden carries the reference Tree's raw-bit field set; the Rust side reconstructs the Tree and serializes via the SHARED lgbm-model %.17g formatter, so the compare is pure structural/numeric (the formatter is the single arbiter)"
    - "Deferred-emission capture: the C++ emitter buffers each leaf's per-feature snapshot and emits in the SAME structure the Rust find_best_splits emits (root, then smaller-then-larger per decision; last split's children never emitted) so the PSPLIT record set is bit-aligned"

key-files:
  created:
    - crates/lgbm-treelearner/src/leaf_splits.rs
    - crates/lgbm-treelearner/src/fix_histogram.rs
    - crates/lgbm-treelearner/src/data_partition.rs
    - crates/lgbm-treelearner/src/histogram_pool.rs
    - crates/lgbm-treelearner/src/learner.rs
    - crates/oracle-harness/tests/fixtures/learner/spine.txt
  modified:
    - crates/lgbm-treelearner/src/lib.rs
    - crates/lgbm-compute/src/lib.rs
    - xtask/cpp/learner_capture.cpp
    - xtask/src/main.rs
    - crates/oracle-harness/tests/learner_parity.rs
    - crates/oracle-harness/Cargo.toml
    - crates/oracle-harness/fixtures/REFERENCE_MANIFEST.md
    - Cargo.lock

key-decisions:
  - "Smaller/larger child seeding uses DataPartition::leaf_count (not best.left_count/right_count) — the two notions can disagree (find_best_split reconstructs counts via round_int(hess*cnt_factor) and the threshold semantics), and the disagreement silently swapped the child sums; driving both off the partition count is the faithful Pitfall-3 fix."
  - "D-07 compares via the SHARED lgbm-model %.17g formatter: the golden carries the C++ reference tree's raw field bits (PT_*), the Rust side reconstructs the Tree and calls to_string(); both the grown tree and the reference go through ONE formatter, so the comparison is structural/numeric, not a re-implementation of %g in C++."
  - "FixHistogram lives learner-side (RESEARCH Open Q1) as a plain f64 loop on the host histogram, on RAW (un-bumped) leaf sums (Pitfall 2)."
  - "ComputeClient re-exported from lgbm-compute (ComputeClientReexport) so lgbm-treelearner names the Backend client arg without a cubecl dependency, preserving CMP-01 containment."

patterns-established:
  - "The spine learner takes FeatureColumn inputs (binned column + bin-layout descriptors + real-value thresholds) directly — it never re-bins (Phase-2 is the determinism root); skip_default_bin/na_as_missing are derived per-feature from missing_type + num_bin>2 (Pitfall 1)."

requirements-completed: [TRL-01, TRL-02, TRL-03, TRL-04, TRL-05, TRL-07]

# Metrics
duration: 25min
completed: 2026-06-05
---

# Phase 5 Plan 03: Tree-Learner Spine (Leaf-Wise Growth + Split Finding) Summary

**Built the minimal faithful `SerialTreeLearner` spine — the leaf-wise (best-first) growth loop orchestrating the Phase-4 `construct_histograms`/`find_best_split`/`subtract_histograms`/`data_partition` kernels into a tree structurally + numerically bit-faithful to C++ `SerialTreeLearner::Train` (force_row_wise, feature_fraction=1.0, missing_type=None) — validated at MAXIMUM resolution: per-split full per-bin gain arrays (D-06), full grown tree incl. leaf outputs via the shared `%.17g` machinery (D-07), the subtraction trick, missing/zero routing, and the D-02a kernel-vs-learner two-transcription cross-check.**

## Performance

- **Duration:** ~25 min
- **Tasks:** 3
- **Files created:** 6 | **Files modified:** 8

## Accomplishments

- **Task 1 — load-bearing bookkeeping:** `leaf_splits.rs` (single ordered f64 fold, NEVER parallel — the deterministic `!deterministic_` branch; `weight()` seeded via `gain::calculate_splitted_leaf_output`), `fix_histogram.rs` (most-freq-bin reconstruct on RAW sums, Pitfall 2), `data_partition.rs` (`leaf_begin_`/`leaf_count_`/`indices_` wrapping `Backend::data_partition`, smaller-child selection reads `leaf_count`, Pitfall 3), `histogram_pool.rs` (full C++ pool mirror: `get`/`move_`/`reset_map` + LRU eviction, D-05). 13 unit tests.
- **Task 2 — `SerialTreeLearner`:** the leaf-wise loop with V5 typed-error boundary FIRST (g/h length, num_leaves, bins < num_bin, sum_hessian > 0, na_as_missing deferred), `before_find_best_split` (max_depth + both-children-too-small gates), smaller-child selection driving the subtraction trick off partition row counts, per-feature `FixHistogram` → `find_best_split` (gain in-kernel) → flat-`Vec` ArgMax + `split_gt` tie-break, `min_gain_to_split` added back ONLY for the tree `split_gain` field, `Tree::split` growth. `leaf_wise_caps` + invalid-input unit tests.
- **Task 3 — capture + parity + D-02a:** filled `learner_capture.cpp` with the WHOLE leaf-wise-loop transcription (reusing the `kernel_capture.cpp` `FindBestThreshold` structure, D-02a) over a fixed 12-row/2-feature synthetic corpus; emitted the committed `spine.txt` (10 PSPLIT per-bin-gain records + 1 PTREE raw-bit field set); `learner_parity.rs` asserts `spine_per_bin_gains` (bit-exact), `spine_full_tree` (D-07 String equality via the shared formatter), `subtract` (TRL-02), `missing_routing` (TRL-05), and `transcription_crosscheck` (D-02a). Byte-idempotent; never references the untracked `LightGBM/`.

## Task Commits

1. **Task 1: leaf_splits + fix_histogram + data_partition + histogram_pool** — `5f567f5` (feat)
2. **Task 2: SerialTreeLearner leaf-wise growth loop + subtraction trick + Split** — `bc4325c` (feat)
3. **Task 3: learner_capture full transcription + spine golden + parity + D-02a** — `c0bec7c` (test)

_Note: Tasks 1 and 2 were TDD-flagged; their RED/GREEN collapsed into one `feat` commit each because the new module/API surface and its behavior tests are mutually dependent at compile time (the tests cannot compile without the new symbols). The per-module unit tests are the behavior assertions._

## Files Created/Modified

- `crates/lgbm-treelearner/src/{leaf_splits,fix_histogram,data_partition,histogram_pool,learner}.rs` — the four bookkeeping modules + the spine orchestrator.
- `crates/lgbm-treelearner/src/lib.rs` — module registration + re-exports (`LeafSplits`, `fix_histogram`, `DataPartition`, `HistogramPool`, `SerialTreeLearner`, `FeatureColumn`).
- `crates/lgbm-compute/src/lib.rs` — `ComputeClientReexport` (CMP-01-preserving ComputeClient re-export).
- `xtask/cpp/learner_capture.cpp` — full whole-learner transcription replacing the Plan-02 scaffold; emits `spine.txt` (PSPLIT D-06 + PTREE D-07).
- `xtask/src/main.rs` — `learner-capture` writes `spine.txt`; manifest "Learner Golden Set" section updated to the real spine.
- `crates/oracle-harness/tests/learner_parity.rs` — the per-split/full-tree/subtract/missing-routing/D-02a replay.
- `crates/oracle-harness/tests/fixtures/learner/spine.txt` — committed golden (`scaffold.txt` removed).
- `crates/oracle-harness/Cargo.toml` — dev-deps `lgbm-model` + `lgbm-treelearner`.
- `crates/oracle-harness/fixtures/REFERENCE_MANIFEST.md` — regenerated with the real-spine Learner Golden Set.

## Decisions Made

- **Smaller/larger child seeding by partition count, not SplitInfo count.** The C++ `serial_tree_learner.cpp:851` seeds smaller/larger by `left_count < right_count`; the realized bug was that `SplitInfo::left_count` (reconstructed via `round_int(hess·cnt_factor)` + threshold semantics) can disagree with the ACTUAL `DataPartition::leaf_count` after the partition op routes rows. Driving both the seeding (`split_inner`) and the smaller-child selection (`find_best_splits`) off `DataPartition::leaf_count` makes them bit-consistent (Pitfall 3) and fixed the swapped-child-sums bug that produced a divergent tree.
- **D-07 via the shared formatter.** Rather than re-implement the `%.17g`/`%g` formatter in C++ (a second drift source), the golden carries the reference tree's raw-bit field set (`PT_*`); the Rust replay reconstructs a `Tree` and serializes it through the SAME `lgbm-model` formatter as the grown tree, so the D-07 compare is pure structural/numeric equality.
- **Deferred-emission capture for record alignment.** The C++ emitter buffers each leaf's per-feature snapshot and emits in the exact structure the Rust `find_best_splits` emits (root decision, then smaller-then-larger per split, the last split's children never emitted) so the PSPLIT record set is positionally bit-aligned with the Rust replay.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Smaller/larger child LeafSplits seeded off the wrong count**
- **Found during:** Task 3 (the D-07 full-tree golden replay diverged: Rust grew `split_feature=[0,0,0]` while the reference grew `[0,1,0]`).
- **Issue:** `split_inner` seeded the smaller/larger child `LeafSplits` by `best.left_count < best.right_count`, but that count (from `find_best_split`'s `round_int(hess·cnt_factor)` reconstruction) disagreed with the ACTUAL partition row count, swapping which child got which sums — so a leaf with 4 rows was scanned with the 8-row sums, corrupting every post-root split selection.
- **Fix:** seed smaller/larger by `DataPartition::leaf_count(new_left)` vs `leaf_count(new_right)` — the same notion `find_best_splits` uses for smaller-child selection (Pitfall 3).
- **Files modified:** `crates/lgbm-treelearner/src/learner.rs`
- **Commit:** `c0bec7c` (the fix landed with Task 3 since it was surfaced by the parity golden; it is logically part of the Task-2 loop).

### Plan-instruction adjustments (not scope changes)

- **Grep-gate phrasing (`BinaryHeap`/`BTreeMap` == 0).** Two doc comments named `BinaryHeap`/`BTreeMap` to say "NEVER use one"; reworded to "priority-queue/heap" so the acceptance grep reads 0 without weakening intent (same idiom as the 05-02 deviation). No functional change.
- **`scaffold.txt` removed.** The Plan-02 placeholder golden is superseded by the real `spine.txt`; the orphan fixture + its scaffold-only test were removed (the harness now replays the real corpus).
- **`ComputeClient` re-export.** The plan did not specify how the learner names the `&ComputeClient<B::Runtime>` Backend arg without a `cubecl` dep; added a `lgbm_compute::ComputeClientReexport` re-export (the compute crate is the CMP-01 seam). Tracked here rather than as a deviation because it implements the plan's CMP-01 constraint.

## Issues Encountered

- The C++ emitter and Rust learner initially produced different PSPLIT record COUNTS (12 vs 10) because the C++ emitted children after every split (incl. the last) while Rust emits a split's children at the START of the next `find_best_splits` (so the last split's children are never scanned). Resolved by deferring the C++ emission to mirror the Rust emit structure exactly.

## User Setup Required

None. The `learner-capture` subcommand needs a C++ toolchain (cmake + g++, present here) only to regenerate `spine.txt`; normal `cargo test` reads the committed golden.

## Next Phase Readiness

- The spine `SerialTreeLearner` + the four bookkeeping modules + the per-split/full-tree parity battery are the contracts Plan 05-04 builds on: it adds `force_col_wise` (TRL-09, expected observationally identical on the deterministic anchor) and per-node feature-subsampling RNG parity (TRL-08, via `ColSampler`), plus the D-03 captured-g/h corpus, on top of this proven spine.
- NA_AS_MISSING remains deferred (RESEARCH A5): a feature with `num_bin > 2 && missing_type == NaN` is a typed `TreeLearnerError` today; the forward-branch body must be transcribed before any synthetic/captured case uses it.

## Self-Check: PASSED

- `.planning/phases/05-tree-learner-split-finding/05-03-SUMMARY.md` — FOUND
- `crates/lgbm-treelearner/src/learner.rs` — FOUND
- `crates/oracle-harness/tests/fixtures/learner/spine.txt` — FOUND
- Commit `5f567f5` (Task 1) — FOUND
- Commit `bc4325c` (Task 2) — FOUND
- Commit `c0bec7c` (Task 3) — FOUND

---
*Phase: 05-tree-learner-split-finding*
*Completed: 2026-06-05*
