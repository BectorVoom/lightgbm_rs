---
phase: 05-tree-learner-split-finding
plan: 05
subsystem: testing
tags: [tree-learner, histogram, offset, compaction, data-partition, split-finding, cr-01, d-09]

# Dependency graph
requires:
  - phase: 05-01
    provides: authoritative SKIP_DEFAULT_BIN/NA_AS_MISSING flags + find_best_split offset threading
  - phase: 05-02
    provides: lgbm-treelearner crate skeleton, error boundary, split-info tie-break
  - phase: 05-03
    provides: SerialTreeLearner growth loop, FixHistogram, DataPartition, D-06/D-07 goldens
  - phase: 05-04
    provides: force_col_wise/ColSampler/real-g/h corpora + spine.txt/col_wise.txt/real_gh.txt
provides:
  - "offset_for_most_freq_bin(most_freq_bin) -> i32 — THE single authoritative offset rule (most_freq_bin==0 -> 1, else 0)"
  - "compacted offset==1 histogram in the learner (compact_histogram) so the scan reads real bin t+offset"
  - "single-feature-group partition min_bin = min_bin + offset, collapsing the verbatim --th to th=threshold for mfb==0"
  - "learner_parity_routing_self_consistency — oracle-independent get_leaf-tally == data-partition leaf_count invariant (CR-01)"
affects: [05-06, 05-07, 06-gbdt]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Single authoritative convention helper (offset_for_most_freq_bin) — exactly one place computes most_freq_bin->offset"
    - "Compacted-histogram layout for offset==1 (drop bin-0/most-freq slot, cell c = real bin c+offset)"
    - "Oracle-INDEPENDENT routing invariant (get_leaf tally == leaf_count) as a loud self-consistency gate"

key-files:
  created: []
  modified:
    - crates/lgbm-treelearner/src/lib.rs
    - crates/lgbm-treelearner/src/learner.rs
    - crates/oracle-harness/tests/learner_parity.rs

key-decisions:
  - "D-09 adopted end-to-end: offset==1 + compacted histogram for most_freq_bin==0, superseding D-01's offset==0 non-compacted convention"
  - "The CR-01 off-by-one root cause is the single-feature-group min_bin: C++ FeatureGroup::Split(num_feature_==1) dispatches to DenseBin::Split(max_bin,...) which hard-codes min_bin=1; passing min_bin=0 left th=threshold-1 (partition [4,8]) while predict routed bin<=threshold left ([6,6]). Passing min_bin+offset collapses --th to th=threshold so all three boundaries agree."
  - "The pre-D-09 learner_capture.cpp full-tree/per-bin goldens (spine.txt/col_wise.txt/real_gh.txt) share the buggy convention (CR-02), so their assertions are superseded by 05-06's real lib_lightgbm oracle; the oracle-independent routing test is the live CR-01 gate in this plan."

patterns-established:
  - "Pattern 1: convention rules live in ONE helper + ONE compaction site; a grep gate proves no inlined offset literal at any construction site"
  - "Pattern 2: a golden-free invariant (predict routing reproduces the stored partition exactly) is the falsification gate that the self-transcription golden could not provide"

requirements-completed: [TRL-05, TRL-07, TRL-01]

# Metrics
duration: 13min
completed: 2026-06-06
---

# Phase 5 Plan 05: Real-offset==1 + Compacted-histogram Convention (CR-01 Closure) Summary

**Adopted the real-LightGBM offset==1 + compacted-histogram convention end-to-end via a single `offset_for_most_freq_bin` helper, fixed the single-feature-group `min_bin` so the stored threshold / partition `--th` / predict `fval<=threshold` all agree for most_freq_bin==0, and added an oracle-independent `get_leaf`-tally == `leaf_count` routing test that reproduces and now passes the CR-01 `[4,8]` vs `[6,6]` divergence.**

## Performance

- **Duration:** 13 min
- **Started:** 2026-06-06T02:34:00Z
- **Completed:** 2026-06-06T02:47:53Z
- **Tasks:** 3
- **Files modified:** 3

## Accomplishments
- **One authoritative offset rule:** `offset_for_most_freq_bin` (`lib.rs`) returns 1 for `most_freq_bin==0`, else 0; every corpus/feature offset derivation (spine f0/f1, col_sampler `make`, missing_routing, real_gh parser) routes through it. The three contradictory inlined rules are deleted; a grep gate proves 0 inlined offset literals at construction sites.
- **Compacted histogram for offset==1:** `compact_histogram` shifts the stride-2 histogram into the C++ `data_` compacted layout (cell `c` = real bin `c+offset`, bin-0/most-freq dropped, tail zeroed to keep `2*num_bin` length) before both `find_best_split` and the host per-bin re-scan, mirroring `feature_histogram.hpp:619/943/950`. No-op for `offset==0`.
- **CR-01 closed (the real root cause):** the single-feature-group `min_bin` — C++ `FeatureGroup::Split` (`num_feature_==1`) dispatches to `DenseBin::Split(max_bin,...)` which hard-codes `min_bin=1`. Passing `min_bin + offset` makes the verbatim `th = threshold + min_bin; --th` collapse to `th = threshold` for `most_freq_bin==0`, so partition `bin>threshold -> right` matches predict `bin<=threshold -> left`. The previous raw `min_bin==0` left `th=threshold-1` (the `[4,8]` partition vs `[6,6]` predict off-by-one).
- **Oracle-independent CR-01 gate:** `learner_parity_routing_self_consistency` routes every training row through the grown tree's `get_leaf`, tallies per leaf, and asserts `== tree.leaf_count` (the stored data-partition count) EXACTLY across spine / col_wise / col_sampler / real_gh. It FAILS on the pre-fix code (reproduced `tally [6,2,0,4]` vs stored `[4,2,0,6]`) and PASSES after the min_bin fix.

## Task Commits

Each task was committed atomically:

1. **Task 1: single shared offset helper + route all derivations** — `bc3dd95` (feat)
2. **Task 2: compacted offset==1 histogram, self-consistent threshold/partition/predict** — `2e996e0` (feat)
3. **Task 3: oracle-independent routing test + single-feature min_bin fix** — `ee0255f` (test)

**Plan metadata:** (this commit) `docs(05-05): complete plan`

_Note: Tasks 1/2 are TDD-style (helper unit test + compact_histogram unit tests precede/accompany the change); the partition min_bin fix that completes Task 2's self-consistency was surfaced by Task 3's test and committed with it._

## Files Created/Modified
- `crates/lgbm-treelearner/src/lib.rs` — added `pub fn offset_for_most_freq_bin` (THE offset rule, citing feature_histogram.hpp:1429-1433 + bin.h:180-258, supersedes D-01 per D-09) + its unit test.
- `crates/lgbm-treelearner/src/learner.rs` — `compact_histogram` + its application after FixHistogram/before the scan; single-feature-group `partition_min_bin = min_bin + offset`; FeatureColumn.offset doc points at the helper; compact_histogram unit tests.
- `crates/oracle-harness/tests/learner_parity.rs` — all corpus offsets via the helper; new `learner_parity_routing_self_consistency` (+`row_feature_values`/`assert_routing_self_consistent`); the pre-D-09 self-transcription full-tree/per-bin assertions superseded (skip with a 05-06 re-point note); `row==col` equality (convention-independent) kept live; golden parsers `#[allow(dead_code)]` for 05-06 reuse.

## Decisions Made
- **D-09 end-to-end:** offset==1 + compacted histogram for most_freq_bin==0 (supersedes D-01).
- **CR-01 root cause = single-feature-group min_bin (not the histogram alone):** compaction alone did NOT fix the partition; the `min_bin+offset` correction (mirroring `DenseBin::Split(max_bin,...)`'s hard-coded `min_bin=1`) is what collapses `--th` so partition and predict agree. This is the load-bearing fidelity point.
- **Stale self-transcription goldens superseded by 05-06:** the D-09 convention change intentionally grows a different (now self-consistent) tree than the pre-D-09 `learner_capture.cpp` goldens, which baked in the CR-01 bug (CR-02). Asserting against a known-wrong golden is worse than no assertion; 05-06 re-points the full-tree/per-bin tests at a real `lib_lightgbm` 4.6 oracle.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Single-feature-group partition `min_bin` (the actual CR-01 fix)**
- **Found during:** Task 3 (routing self-consistency test reproduced `[4,2,0,6]` partition vs `[6,2,0,4]` predict)
- **Issue:** The plan's Task 2 (offset==1 compaction + verbatim `--th`) was necessary but NOT sufficient — with the corpus `min_bin==0`, `th = threshold + 0 - 1 = threshold-1` still routed `bin==threshold` RIGHT while predict routed it LEFT (the CR-01 off-by-one persisted after compaction).
- **Fix:** Pass `partition_min_bin = f.min_bin + f.offset` to `data_partition.split`, mirroring the C++ single-feature `FeatureGroup::Split` -> `DenseBin::Split(max_bin,...)` overload (`dense_bin.hpp:423-433`) which hard-codes `min_bin=1`. For offset==1 this collapses `--th` to `th=threshold`; for offset==0 it is unchanged.
- **Files modified:** crates/lgbm-treelearner/src/learner.rs
- **Verification:** `learner_parity_routing_self_consistency` PASSES (tally == leaf_count for all corpora); FAILED before this fix.
- **Committed in:** ee0255f (Task 3 commit)

**2. [Rule 3 - Blocking] Superseded the pre-D-09 self-transcription parity assertions**
- **Found during:** Task 2/3 (the offset==1 + min_bin convention grows a different, self-consistent tree than the pre-D-09 spine.txt/col_wise.txt/real_gh.txt goldens)
- **Issue:** The plan says "no new fixture file in this plan — fixtures change in 05-06," but the convention change necessarily breaks the full-tree/per-bin comparisons against the old (buggy-convention) goldens, which would leave `cargo test --workspace` red.
- **Fix:** The 4 full-tree/per-bin assertions (`spine_full_tree`, `spine_per_bin_gains`, `transcription_crosscheck`, `real_gh_full_tree`, + the golden half of `row_vs_col`) skip with a `STALE_SELF_TRANSCRIPTION_NOTE` documenting 05-06's real-binary re-point. The convention-independent `row==col` equality (TRL-09) stays a live gate; the oracle-independent routing test is the live CR-01 gate. Golden parsers retained via `#[allow(dead_code)]` for 05-06 reuse.
- **Files modified:** crates/oracle-harness/tests/learner_parity.rs
- **Verification:** `cargo test -p oracle-harness learner_parity` 9/9 green; `cargo test --workspace` 0 failed.
- **Committed in:** ee0255f (Task 3 commit)

---

**Total deviations:** 2 auto-fixed (1 bug — the actual CR-01 min_bin fix; 1 blocking — stale-golden supersession)
**Impact on plan:** Both essential. Deviation 1 is the genuine CR-01 closure the plan's Task 2 framing under-specified (compaction was necessary but the min_bin convention is the off-by-one fix). Deviation 2 keeps the workspace green while honoring "fixtures change in 05-06." No scope creep.

## Issues Encountered
- **Abstract derivation showed a persistent off-by-one even after compaction.** Resolved by tracing the C++ `FeatureGroup::Split` dispatch: for a single-feature group (`num_feature_==1`) C++ uses the `DenseBin::Split(max_bin,...)` overload with hard-coded `min_bin=1`/`USE_MIN_BIN=false`, which is the missing piece that makes `--th` self-consistent for `most_freq_bin==0`. The synthetic spine corpus declares `mfb=0` while all bins tie (no genuine most-freq), which is exactly why its self-transcription golden is unreliable (CR-02) and 05-06 uses real data.
- **Pre-existing clippy warnings** in lgbm-dataset / other crates are out of scope (not introduced by this plan) — logged, not fixed.

## Next Phase Readiness
- CR-01 closed: stored threshold / partition `--th` / predict routing agree under the offset==1 compacted convention for `most_freq_bin==0`; the offset==0 (`most_freq_bin>0`) path is byte-unchanged (kernel_parity 4/4 bit-exact, lgbm-treelearner lib green).
- **05-06 (next, depends on 05-05):** re-point the now-skipped full-tree/per-bin parity tests at the REAL `lib_lightgbm` 4.6 oracle (`spine_real.txt` / `mfb_pos_real.txt`), giving the offset==1 scan+partition path its first real bit-exact coverage. The golden parsers are retained for reuse.
- `offset_for_most_freq_bin` + `compact_histogram` are the single convention sources future learner work (and the real-data corpora) build on.

## Self-Check: PASSED

- FOUND: crates/lgbm-treelearner/src/lib.rs, crates/lgbm-treelearner/src/learner.rs, crates/oracle-harness/tests/learner_parity.rs (all modified + compile)
- FOUND: .planning/phases/05-tree-learner-split-finding/05-05-SUMMARY.md
- FOUND commits: bc3dd95 (Task 1), 2e996e0 (Task 2), ee0255f (Task 3)
- Verification: grep single-offset-source 0; `cargo test -p oracle-harness learner_parity` 9/9; `cargo test --workspace` 0 failed; kernel_parity 4/4 bit-exact; routing self-consistency green.

---
*Phase: 05-tree-learner-split-finding*
*Completed: 2026-06-06*
