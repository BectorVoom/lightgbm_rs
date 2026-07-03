---
status: complete
phase: quick-260609-bfx
plan: 01
subsystem: treelearner
tags: [tree-learner, histogram, allocation, perf, fix_histogram, compact_histogram, parity]

# Dependency graph
requires:
  - phase: quick-260609-8go
    provides: per-iteration grad/hess/score move-not-clone (the prior allocation win this builds on)
provides:
  - "build_leaf_histogram_into runs fix_histogram + compact_histogram IN PLACE on the owned raw buffer (no per-feature to_vec clone)"
  - "one fewer heap allocation per feature per directly-built leaf build"
affects: [tree-learner perf, CPU f64-fold spine hot path]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "In-place fix+compact on a &mut sub-slice of the backend-owned raw f64 buffer (no intermediate Vec)"

key-files:
  created: []
  modified:
    - crates/lgbm-treelearner/src/learner.rs

key-decisions:
  - "Removed the per-feature to_vec() clone at learner.rs:1647; same f64 cells, same op order, same storage — parity-neutral by construction."
  - "Did NOT touch fix_histogram, compact_histogram, the Backend call, feature_bins/num_bins collects, or the buildable short-circuit — purely one clone eliminated."
  - "Reported bench delta honestly as flat/noise on the single-threaded synthetic corpus (consistent with 260609-8go); the deliverable is the allocation reduction, not a wall-clock speedup claim."

patterns-established:
  - "Reuse the backend's already-owned raw buffer for in-place host post-processing instead of cloning per feature."

requirements-completed: [QUICK-260609-bfx]

# Metrics
duration: ~10min
completed: 2026-06-09
---

# Phase quick-260609-bfx: Optimise memory bottleneck in tree construction Summary

**Eliminated the per-feature `to_vec()` clone in `build_leaf_histogram_into` by running `fix_histogram` + `compact_histogram` in place on a `&mut` sub-slice of the learner-owned `raw` buffer — one fewer heap allocation per feature per directly-built leaf, bit-exact CPU f64-fold spine preserved.**

## Performance

- **Duration:** ~10 min
- **Started:** 2026-06-09
- **Completed:** 2026-06-09
- **Tasks:** 3
- **Files modified:** 1

## Accomplishments

- **Confirmed the target before editing (Task 1):** verified all three facts that make the `learner.rs:1647` `to_vec()` the correct, lowest-risk, parity-neutral target:
  - (a) `raw` (learner.rs:1629) is an owned `Vec<f64>` returned by `backend.build_leaf_histograms_raw(...)`; within the function body its ONLY reader is the per-feature loop (1645-1655) — no reader after line 1655.
  - (b) `fix_histogram(&mut [f64], ...)` (fix_histogram.rs:50) and `compact_histogram(&mut [f64], i32)` (learner.rs:3150) both mutate their slice IN PLACE.
  - (c) the loop runs once per feature per directly-built leaf; call sites learner.rs:1127/1434/1528/1546 (smaller-child / root / audit / fallback paths, every tree growth).
- **Named target / why highest-impact-lowest-risk:** the clone fired `num_features` times per directly-built leaf × per leaf × per tree × per boosting iteration — pure churn with no semantic role, since the same two in-place ops can run on the owned buffer. Provably parity-neutral: same f64 cells, same op order, same storage type; only the intermediate `Vec` is removed.
- **Eliminated the clone (Task 2):** `let mut raw = ...`; the loop now borrows `&mut raw[range]` in an inner scope for `fix_histogram` + `compact_histogram`, ends the borrow, then `buf[range].copy_from_slice(&raw[range])`. Same two functions, same args (`f.most_freq_bin, sum_g, sum_h, f.offset`), same cells, same order.
- **Full parity gate GREEN (Task 3):** the CPU f64-fold spine stays BIT-EXACT vs real lib_lightgbm 4.6 (`learner_parity_spine_real_binary` passes). No new failure, no newly-required tolerance, no `#[ignore]` added.

## Task Commits

1. **Task 1: Measure baseline + confirm facts (a)-(c)** - no code change (inspection + baseline bench)
2. **Task 2: Eliminate the per-feature `to_vec()`** - `9ffe71f` (perf)
3. **Task 3: Full parity gate + before/after bench** - no further code change (verification only)

_Docs commit (SUMMARY/STATE) handled by the orchestrator._

## Files Created/Modified

- `crates/lgbm-treelearner/src/learner.rs` - `build_leaf_histogram_into`: `raw` made `mut`; per-feature `to_vec()` clone removed; `fix_histogram` + `compact_histogram` now run on a `&mut` sub-slice of `raw` in an inner scope, then the fixed+compacted sub-slice is copied into `buf`. Explanatory comments (FixHistogram / compaction / Pitfall 2) retained, attached to the in-place ops.

## Before/After Bench (cargo run --release --example bench_train; allocator: system, iters: 100, leaves: 31)

| size   | rows  | feat | bins | train_median BEFORE | train_median AFTER | delta        |
|--------|-------|------|------|---------------------|--------------------|--------------|
| small  | 2000  | 12   | 32   | 40.67ms             | 41.15ms            | +1.2% (noise) |
| medium | 8000  | 30   | 64   | 246.48ms            | 247.24ms           | +0.3% (noise) |
| large  | 20000 | 50   | 128  | 856.87ms            | 845.37ms           | -1.3% (noise) |

**Honest read:** flat / within run-to-run noise on this single-threaded synthetic corpus — exactly as 260609-8go found for allocator-level changes. **No speedup is claimed.** The deliverable is the *allocation reduction* (one fewer `Vec` per feature per directly-built leaf), which is correct regardless of wall-clock noise.

## Parity Gate Results

- `cargo test -p oracle-harness --test learner_parity` — **29 passed / 0 failed / 0 ignored** (incl. `spine_real_binary` bit-exact vs real lib_lightgbm 4.6).
- `cargo test -p oracle-harness --test kernel_parity` — **6 passed / 0 failed / 0 ignored** (CPU; hip cell feature-gated off on this run).
- Full gate `cargo test -p lgbm -p lgbm-treelearner -p lgbm-boosting -p oracle-harness` — **GREEN, 0 failed, 0 ignored** across all binaries:
  - lgbm 41, lgbm-treelearner 55, lgbm-boosting 64, advanced_parity 5, **boosting_parity 75**, comparator 5, config_drift 3, kernel_parity 6, learner_parity 29, metric_parity 15, predict_parity 5, rank_parity 4, raw_bin_train_parity 2, rng_parity 1, plus 3+3 misc.
- `cargo clippy -p lgbm-treelearner` — no warnings in the edited region (lines 1600-1659). The 16 pre-existing crate warnings are out of scope (Scope Boundary; not introduced by this change).
- `LightGBM/` never git-added. Only `crates/lgbm-treelearner/src/learner.rs` staged in the atomic commit.
- Post-commit deletion check: 0 deletions.

**parity_class = neutral.**

## Decisions Made

None beyond the plan — followed it exactly. Bench delta reported honestly as noise (no overclaim).

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None.

## Known Stubs

None.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- The allocation churn in the directly-built-leaf histogram path is reduced by one Vec per feature per leaf build, with the CPU spine bit-exact and the full parity suite GREEN.
- No blockers introduced. The pre-existing 16 clippy warnings in `lgbm-treelearner` and the out-of-scope DEF-07-02 parked goldens are unchanged.

## Self-Check: PASSED

- FOUND: crates/lgbm-treelearner/src/learner.rs (modified, clone removed)
- FOUND commit: 9ffe71f
- Full parity gate GREEN (0 failed, 0 ignored); spine bit-exact preserved.

---
*Phase: quick-260609-bfx*
*Completed: 2026-06-09*
