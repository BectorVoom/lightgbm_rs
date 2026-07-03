---
phase: 23-perf-validation-default-on-rollout-dod
reviewed: 2026-07-03T05:17:46Z
depth: standard
files_reviewed: 6
files_reviewed_list:
  - crates/lgbm-compute/src/kernels/grow_driver.rs
  - crates/lgbm-compute/src/lib.rs
  - crates/lgbm-compute/tests/cuda_on_device.rs
  - crates/lgbm-compute/tests/on_device_launch_count.rs
  - crates/lgbm-treelearner/src/learner.rs
  - crates/lgbm-treelearner/src/phase_prof.rs
findings:
  critical: 0
  warning: 3
  info: 2
  total: 5
status: issues_found
---

# Phase 23: Code Review Report

**Reviewed:** 2026-07-03T05:17:46Z
**Depth:** standard
**Files Reviewed:** 6
**Status:** issues_found

## Summary

The phase-23-specific delta (isolated from the six large files in scope via
`git log`) is small and self-contained:

- `grow_driver.rs`: the compute-owned `ON_DEVICE_LAUNCH_CNT` atomic +
  `launch_prof_enabled()` / `bump_launch()` / `on_device_launch_count_take()`,
  with `bump_launch()` calls at the build, subtract, and numeric-scan dispatch
  sites (commit `53adcca`).
- `phase_prof.rs`: 12 lines folding `on_device_launch_count_take()` into the
  `device_launches=` COUNTS line (commit `5ae7f85`).
- `cuda_on_device.rs`: unit tests for the tri-state resolver mapping (commit
  `f671230`).
- `on_device_launch_count.rs`: an instrumentation test that grows a tiny corpus
  on the cubecl-cpu runtime and asserts a non-zero, sub-baseline launch count
  (commit `ee3858f`).

Correctness of the default merge gate is sound: with `LGBM_PHASE_PROF` unset the
counter is truly inert (gated `OnceLock` bump; `dump` early-returns before the
`take`), and with `LGBM_CUDA_ON_DEVICE` unset the on-device driver is never
invoked — so SC-4 byte-unchanged holds. The tri-state mapping tests are exact
and correct.

The findings below are measurement-integrity and test-quality defects. They
matter because this phase's DoD is a perf-validation A/B whose decisive metric
(SC-2) is exactly the launch count these changes produce. None cause incorrect
tree output, so none are BLOCKERs — but the launch figure that feeds the A/B
verdict is not what it is documented to be.

## Warnings

### WR-01: `device_launches=` sums two incompatible launch granularities

**File:** `crates/lgbm-compute/src/kernels/grow_driver.rs:447` and `crates/lgbm-treelearner/src/phase_prof.rs:202-206`
**Issue:** The on-device driver bumps the counter **once per feature** — `bump_launch()` is inside the `for (fpos, f) in features.iter()` loop in `build_leaf_hist` (line 447) and inside the per-feature loop in `scan_leaf` (line 517). The host resident counters it is summed with (`BUILD_RESIDENT_CNT`, `SCAN_RESIDENT_CNT`, `FUSED_CNT`) are bumped **once per leaf** — a single batched launch over all features (see `build_resident_leaf` / `scan_resident_leaf` semantics documented in `lib.rs:948-1033`). `phase_prof::dump` then adds them into one `device_launches=` total (`launches = bld_cnt + sub_cnt + scn_cnt + fus_cnt + on_dev`). A per-feature count and a per-leaf-batched count are not the same unit, so the on-device figure is inflated by roughly a factor of `num_features` relative to how the host baseline was measured. For the SC-2 "launch collapse" comparison this understates the collapse (on-device looks like it launches far more than a batched implementation would), which biases the A/B verdict against the on-device path with a number that is not measuring what the metric name claims.
**Fix:** Make the two counters count the same unit. Either bump `ON_DEVICE_LAUNCH_CNT` **once per leaf** (hoist the bump out of the per-feature loops in `build_leaf_hist` / `scan_leaf` to the `build_leaf_hist` / `scan_leaf` call sites in `grow_tree_on_device_driver_with_cfg`), matching the per-leaf host counters, or split the reported figure into distinct per-feature and per-leaf fields so the A/B harness never sums across granularities. Document which unit `on_device=` is in, next to the host counters' unit.

### WR-02: the launch-count assertion is ~100× loose and does not validate collapse

**File:** `crates/lgbm-compute/tests/on_device_launch_count.rs:121-124`
**Issue:** The test asserts `launches < HOST_BASELINE_LAUNCHES` where `HOST_BASELINE_LAUNCHES = 8570` is documented (line 22-24) as the host denominator for **100 trees**, but the test grows a **single** tiny tree (3 features, 256 rows, 8 leaves). The actual on-device count for this corpus is on the order of ~70-90 bumps, so the bound is roughly 100× loose. The assertion would pass even if the on-device path launched dramatically **more** per tree than the host path does per tree — i.e. it cannot fail for the very regression (launch non-collapse) SC-2 exists to detect. As written it proves only "non-zero and finite," not "collapsed."
**Fix:** Compare like-for-like on a per-tree basis. Divide the host baseline by its tree count (`8570 / 100 ≈ 86`) and assert the single-tree on-device count is at or below a per-tree bound, or grow the same corpus through the host resident path in-test and assert `on_device_count <= host_count`. At minimum, tighten the constant to a per-tree figure and drop the "100 trees" denominator so the bound is meaningful.

### WR-03: right-child leaf id trusted via `debug_assert` only; release desync silently corrupts the partition

**File:** `crates/lgbm-compute/src/kernels/grow_driver.rs:912-945`
**Issue:** After each split the driver does `debug_assert_eq!(new_right as usize, leaves.len(), ...)` then unconditionally `leaves.push(...)`, and immediately afterward indexes `leaves[larger_leaf as usize]` / `leaves[smaller_leaf as usize]` where `larger_leaf`/`smaller_leaf` may be `new_right` (line 928-945). `new_right` comes from the device kernel's `right_leaf_index` (`tree.rs` assigns the new right leaf id as the tree's internal `num_leaves`). The driver's `leaves.len()` and the tree's `num_leaves` are kept in lockstep only by construction. In a release build the `debug_assert` is compiled out, so if the two ever desync, the code either panics on out-of-bounds indexing or — worse — silently writes the derived histogram into the wrong leaf slot, corrupting the partition with no typed error. This is inconsistent with the surrounding driver, which converts analogous invariants into typed `ComputeError`s (see IN-01/IN-02 handling at lines 750, 854).
**Fix:** Replace the `debug_assert_eq!` with a real check that returns `ComputeError::Runtime` when `new_right as usize != leaves.len()`, before the `push` and before any `leaves[new_right]` access, so a kernel/driver leaf-id desync fails loudly in release rather than corrupting state.

## Info

### IN-01: duplicated `LGBM_PHASE_PROF` gate across the crate boundary

**File:** `crates/lgbm-compute/src/kernels/grow_driver.rs:62-65` vs `crates/lgbm-treelearner/src/phase_prof.rs:112-115`
**Issue:** `grow_driver::launch_prof_enabled()` re-implements `phase_prof::enabled()` verbatim (`std::env::var("LGBM_PHASE_PROF").map(|v| v == "1")` behind a private `OnceLock`). Two independent process caches of the same env var. The crate-cycle rationale for not sharing is sound, but the duplication is a silent-divergence risk if one interpretation ever changes (e.g. accepting `"true"`).
**Fix:** Add a cross-reference comment on each function pointing at the other as the canonical twin, or expose a single tiny shared helper in a crate below both. Low priority — purely a maintainability note.

### IN-02: on-device count excludes tree-mutation/partition device dispatches (scope note)

**File:** `crates/lgbm-compute/src/kernels/grow_driver.rs:819`, `886`, `780`
**Issue:** `bump_launch()` deliberately covers only build/subtract/scan. The per-split `tree.split_on_device` / `split_categorical_on_device` (lines 819, 886), `add_bias` (line 683), and `partition_categorical_on_device` (line 780) are real device dispatches that are not counted. This is consistent with the host `*_RESIDENT_CNT` counters (which also omit tree mutation), so the A/B comparison is not skewed by it — but the field is labeled `device_launches`, which reads as a total. Note this is separate from WR-01: even at matched granularity, `device_launches` is a build+subtract+scan subtotal, not a true total.
**Fix:** Rename/annotate the field (e.g. `hist_launches=` or a doc note "build+subtract+scan only") so a future reader does not treat it as the full device-launch count.

---

_Reviewed: 2026-07-03T05:17:46Z_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
