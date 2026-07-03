---
phase: 23-perf-validation-default-on-rollout-dod
fixed_at: 2026-07-03T05:27:02Z
review_path: .planning/phases/23-perf-validation-default-on-rollout-dod/23-REVIEW.md
iteration: 1
findings_in_scope: 5
fixed: 5
skipped: 0
status: all_fixed
---

# Phase 23: Code Review Fix Report

**Fixed at:** 2026-07-03T05:27:02Z
**Source review:** .planning/phases/23-perf-validation-default-on-rollout-dod/23-REVIEW.md
**Iteration:** 1

**Summary:**
- Findings in scope: 5 (fix_scope = all — includes Info findings)
- Fixed: 5
- Skipped: 0

**Note on commit granularity:** the five findings interleave within two shared
source files (`grow_driver.rs` carries WR-01, WR-03, IN-01; `phase_prof.rs`
carries WR-01, IN-01, IN-02). Because the edits were applied together, commits
are atomic **per file** rather than strictly per finding — each commit compiles
independently and its message enumerates the findings it addresses. WR-01 and
IN-01 therefore span two commits (the grow-driver side and the phase-prof side).

## Fixed Issues

### WR-01: `device_launches=` sums two incompatible launch granularities

**Files modified:** `crates/lgbm-compute/src/kernels/grow_driver.rs`, `crates/lgbm-treelearner/src/phase_prof.rs`
**Commits:** `617ea76` (grow_driver), `1527b4e` (phase_prof)
**Applied fix:** Chose the reviewer's primary option — make the on-device counter
count the SAME unit as the host per-leaf resident counters. Hoisted `bump_launch()`
out of the per-feature loops: it now fires ONCE per `build_leaf_hist` invocation
(after the empty-rows early return) and ONCE per `scan_leaf` invocation (after the
`sum_h > 0 && num_data > 0` guard), instead of once per feature. The subtract bump
(once per split) already matched per-leaf granularity and was left as-is. Updated
the `ON_DEVICE_LAUNCH_CNT` / `bump_launch` doc comments with an explicit
per-leaf granularity contract, and documented the unit next to the host counters
in `phase_prof::dump` (both a block comment and a trailing `launch_unit=...` token
on the COUNTS line). The `device_launches=` emitted KEY was deliberately kept
verbatim so the 23-03 harness regex `device_launches=(?P<launches>\d+)` still
matches (renaming the key would have broken that capture — the unit is annotated
via a separate token instead).

### WR-02: the launch-count assertion is ~100× loose and does not validate collapse

**Files modified:** `crates/lgbm-compute/tests/on_device_launch_count.rs`
**Commit:** `6bb27cc`
**Applied fix:** Replaced the ~100×-loose `launches < 8570` (the 100-tree
denominator) with a like-for-like per-tree comparison. Added
`HOST_BASELINE_PER_TREE = 8570 / 100` (≈ 86) and, more importantly, an EXACT
per-leaf collapse bound `2 + 4*(num_leaves - 1)`. Because the counter is now
per-leaf (WR-01), a per-feature regression would scale the build+scan terms by
`num_features` and blow past this bound, so the test can actually fail for the
launch non-collapse SC-2 exists to detect. Verified: the test compiles and passes
green on the cubecl-cpu runtime (`cargo test -p lgbm-compute --test
on_device_launch_count` → `1 passed`), so the real per-leaf count for the tiny
3-feature / 8-leaf corpus sits at or below the bound of 30.

### WR-03: right-child leaf id trusted via `debug_assert` only; release desync silently corrupts the partition

**Files modified:** `crates/lgbm-compute/src/kernels/grow_driver.rs`
**Commit:** `617ea76`
**Applied fix:** Replaced `debug_assert_eq!(new_right as usize, leaves.len(), ...)`
with a real runtime check that returns `ComputeError::Runtime` when
`new_right as usize != leaves.len()`, placed BEFORE the `leaves.push(...)` and
before any `leaves[new_right]` access. A kernel/driver leaf-id desync now fails
loudly with a typed error in ALL build profiles (including release) instead of
either panicking on an out-of-bounds index or silently writing the derived
histogram into the wrong leaf slot. This matches the driver's existing typed
error boundaries at the split sites (IN-01/IN-02 handling).
**Human-verification note:** this changes control flow on an internal invariant.
The guard is a defensive equality check (compile-verified, and the existing
structure/parity tests still exercise the happy path), but a reviewer should
confirm the intended invariant is exactly `new_right == leaves.len()` at that
point (it is, by the tree's `num_leaves`-as-next-id construction).

### IN-01: duplicated `LGBM_PHASE_PROF` gate across the crate boundary

**Files modified:** `crates/lgbm-compute/src/kernels/grow_driver.rs`, `crates/lgbm-treelearner/src/phase_prof.rs`
**Commits:** `617ea76` (grow_driver), `1527b4e` (phase_prof)
**Applied fix:** Added reciprocal cross-reference comments. `phase_prof::enabled()`
is annotated as the CANONICAL gate; `grow_driver::launch_prof_enabled()` is
annotated as its deliberate verbatim TWIN that cannot share a helper without a
crate cycle (`phase_prof` lives above `lgbm-compute` in the DAG). Each comment
instructs the reader to keep the two env interpretations in lockstep. No behavior
change (documentation-only, as the reviewer scoped it).

### IN-02: on-device count excludes tree-mutation/partition device dispatches (scope note)

**Files modified:** `crates/lgbm-treelearner/src/phase_prof.rs`
**Commit:** `1527b4e`
**Applied fix:** Annotated the `device_launches=` field as a
build+subtract+scan SUBTOTAL at per-leaf granularity, both in a block comment
above the COUNTS emission and via a `launch_unit=build+subtract+scan,per-leaf`
token appended to the emitted line. The field KEY was left unchanged (rather than
renamed to `hist_launches=`) to preserve the 23-03 harness regex, so a future
reader is warned it is not a full device-launch total without breaking the
downstream capture.

---

_Fixed: 2026-07-03T05:27:02Z_
_Fixer: Claude (gsd-code-fixer)_
_Iteration: 1_
