---
phase: 05-tree-learner-split-finding
plan: 01
subsystem: compute
tags: [find_best_split, skip_default_bin, na_as_missing, missing_type, kernel-parity, cubecl, treelearner]

# Dependency graph
requires:
  - phase: 04-compute-backend
    provides: "Backend::find_best_split + find_best_split_cpu/_raw_f32_on kernels, the split.txt golden + kernel_parity harness, the cfg_skip_default_bin Phase-4 heuristic this plan replaces"
provides:
  - "Backend::find_best_split signature widened with authoritative skip_default_bin/na_as_missing flags (replacing the cfg_skip_default_bin heuristic)"
  - "na_as_missing==true as a typed ComputeError::Runtime (deferred NA_AS_MISSING forward branch, never a silent wrong answer)"
  - "split.txt golden carrying na_as_missing on every case + a skip_default_bin==false divergence case that replays bit-exact"
affects: [05-02, 05-03, 05-04, tree-learner-spine, serial_tree_learner]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Authoritative C++ dispatch flags threaded from the learner (missing_type + num_bin>2) instead of re-derived from bin layout in the kernel"
    - "Deferred kernel branch surfaced as a typed error at the boundary (T-05-01-01), never a silent divergence"

key-files:
  created: []
  modified:
    - crates/lgbm-compute/src/lib.rs
    - crates/lgbm-compute/src/kernels/split.rs
    - xtask/cpp/kernel_capture.cpp
    - crates/oracle-harness/tests/fixtures/kernels/split.txt
    - crates/oracle-harness/tests/kernel_parity.rs

key-decisions:
  - "The skip_default_bin==false divergence is observable in default_left (not threshold/gain): for the {0,1,2}|{3} boundary, skip=false makes the FORWARD branch reach t=2 and record default_left=0, while the old heuristic skip=true skips bin 2 so only the REVERSE branch finds the boundary (default_left=1) — same threshold+gain, opposite default_left."
  - "na_as_missing==true returns ComputeError::Runtime (asserted before length/num_bin validation) so the deferred NA_AS_MISSING forward branch can never produce a wrong SplitInfo."
  - "The kernel-capture toolchain (header-only against random.h) is available here; the golden was regenerated via real capture (byte-idempotent), not hand-edited."

patterns-established:
  - "Pitfall-1 resolution: per-feature missing_type-derived flags cross the Backend boundary as explicit params; the kernel no longer approximates the C++ dispatch."

requirements-completed: [TRL-05]

# Metrics
duration: 7min
completed: 2026-06-05
---

# Phase 5 Plan 01: Thread authoritative skip_default_bin/na_as_missing through find_best_split Summary

**Replaced the Phase-4 `cfg_skip_default_bin(default_bin, num_bin)` heuristic with the authoritative C++ `missing_type`-derived `skip_default_bin`/`na_as_missing` flags threaded through `Backend::find_best_split`, proved by a new `default_left`-divergence golden case, with `na_as_missing==true` made an explicit typed error (deferred NA_AS_MISSING branch).**

## Performance

- **Duration:** ~7 min
- **Started:** 2026-06-05T22:17:19Z
- **Completed:** 2026-06-05T22:23:49Z
- **Tasks:** 2
- **Files modified:** 5

## Accomplishments
- Widened `Backend::find_best_split` (trait + `CpuBackend` impl), `find_best_split_cpu`, and `find_best_split_raw_f32_on` with explicit `skip_default_bin: bool` / `na_as_missing: bool` params (inserted after `most_freq_bin`); deleted the `cfg_skip_default_bin` heuristic and replaced both outer call sites with the threaded flag.
- Made `na_as_missing == true` a typed `ComputeError::Runtime` on both the f64 and f32 host paths, validated before any other check — the deferred NA_AS_MISSING forward branch can never silently mis-compute (T-05-01-01).
- Added a `skip_default_bin_false` divergence golden case (`missing_type==None`, `num_bin>2`, `default_bin=2 < num_bin=4`) whose winning `default_left` genuinely differs from what the old `default_bin < num_bin` heuristic would have produced; brute-force verified before committing the hand-picked g/h.
- Regenerated `split.txt` via the real C++ capture toolchain (6 cases, `na_as_missing` field on every SCASE), byte-idempotent; updated the parity replay to parse and pass the new flags through every kernel call and to assert the divergence case + `na_as_missing==false` on all committed cases.

## Task Commits

Each task was committed atomically:

1. **Task 1: Thread skip_default_bin/na_as_missing through find_best_split (kernel + trait)** - `0572324` (feat)
2. **Task 2: Add the skip_default_bin==false divergence golden case + update parity callers** - `6cc1d76` (test)

_Note: Task 1 was TDD-flagged; its RED/GREEN collapsed into one commit because the new behavior (the `na_as_missing` typed-error test + signature change) and its implementation are mutually dependent at the trait level — the test cannot compile without the widened signature. The `find_best_split_na_as_missing_is_typed_error` unit test is the behavior assertion._

## Files Created/Modified
- `crates/lgbm-compute/src/lib.rs` - `Backend::find_best_split` trait + `CpuBackend` impl widened with `skip_default_bin`/`na_as_missing`; doc updated.
- `crates/lgbm-compute/src/kernels/split.rs` - `find_best_split_cpu` + `find_best_split_raw_f32_on` accept the two flags; `na_as_missing` typed-error guard; `cfg_skip_default_bin` fn deleted (replaced by a history note); inline tests updated + new `na_as_missing` test.
- `xtask/cpp/kernel_capture.cpp` - `SplitCfg.na_as_missing` field; `na_as_missing` emitted on every SCASE header; new `skip_default_bin_false` divergence case.
- `crates/oracle-harness/tests/fixtures/kernels/split.txt` - regenerated (6 cases) with `na_as_missing` fields + the divergence case.
- `crates/oracle-harness/tests/kernel_parity.rs` - parse `na_as_missing`; pass both flags into every `find_best_split` / `find_best_split_raw_f32_on` (cpu + hip); divergence-case + `na_as_missing==false` coverage asserts.

## Decisions Made
- **Divergence is in `default_left`, not threshold/gain.** A first hand-picked histogram (low-neg/high-pos) produced an identical winner under skip and no-skip because the boundary was reachable by the non-skipped branch. A brute-force search over the 4-bin integer-gradient space found that the genuine observable divergence is the `default_left` flag: for the `{0,1,2}|{3}` boundary, `skip=false` lets the FORWARD branch reach `t=2` (recording `default_left=0`), while the old `skip=true` heuristic skips bin 2 so only the REVERSE branch finds the same threshold/gain boundary (`default_left=1`). The committed case (`g=[-10,-10,-10,1]`, hess 5/bin) exercises exactly this.
- **`na_as_missing` validated first.** The guard returns the typed error before length/`num_bin`/`sum_hessian` checks so the deferral is unambiguous even on otherwise-valid input.
- **Real capture, not hand-edit.** `random.h` is present, so `cargo run -p xtask -- kernel-capture` regenerated the golden byte-idempotently; the committed golden is the authoritative C++-transcription output.

## Deviations from Plan

None - plan executed exactly as written.

The plan anticipated a divergence in the *winning threshold*; the realized divergence is in *`default_left`* (same threshold + gain, opposite default branch). This is a faithful, stronger demonstration of the same Pitfall-1 root cause (the old heuristic skipping bin 2 changes the SplitInfo), not a scope change — the golden replays bit-exact including this field, satisfying every acceptance criterion. Documented above rather than tracked as a deviation because no plan instruction was contradicted.

## Issues Encountered
- The initial divergence histogram did not actually diverge (skipping bin 2 left the winner unchanged because the boundary was found by the forward branch without bin 2). Resolved by a bounded brute-force search over integer gradients to locate a histogram where skip vs no-skip yield different `default_left`, then transcribing those exact values into the C++ case.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- `Backend::find_best_split` now exposes the authoritative dispatch surface the Phase-5 spine (05-02+) will drive: the learner derives `skip_default_bin = (num_bin > 2 && missing_type == Zero)` and `na_as_missing = (num_bin > 2 && missing_type == NaN)` per feature from `bin_mapper.missing_type()` and passes them down.
- NA_AS_MISSING forward-branch body (`feature_histogram.hpp:945-961`) remains deferred (RESEARCH A5); it is a typed error today and must be implemented before any synthetic/captured case uses `missing_type == NaN` with `num_bin > 2`. The spine can pin synthetic cases to `missing_type == None` to defer it; the D-03 dataset composition decision still owns whether the branch must land.

## Self-Check: PASSED

- `.planning/phases/05-tree-learner-split-finding/05-01-SUMMARY.md` — FOUND
- Commit `0572324` (Task 1) — FOUND
- Commit `6cc1d76` (Task 2) — FOUND

---
*Phase: 05-tree-learner-split-finding*
*Completed: 2026-06-05*
