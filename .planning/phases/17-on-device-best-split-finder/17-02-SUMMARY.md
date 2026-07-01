---
phase: 17-on-device-best-split-finder
plan: 02
subsystem: treelearner
tags: [cubecl, gain-math, path-smooth, smoothing, f32-mirror, device-kernel]

# Dependency graph
requires:
  - phase: 05-tree-learner (gain primitives)
    provides: "crate::gain scalar #[cube] primitives (threshold_l1, get_leaf_gain, get_split_gains, calculate_splitted_leaf_output) + f32 mirrors"
provides:
  - "Net-new USE_SMOOTHING (path_smooth) gain path in crate::gain as additive #[cube] fns: calculate_splitted_leaf_output_smoothed (form B blend) and get_leaf_gain_smoothed (form D given-output gain), each with an f32 mirror"
  - "get_leaf_gain_given_output promoted from plain host fn to #[cube] (+ get_leaf_gain_given_output_f32 mirror) so the smoothing gain path runs on device"
affects: [17-03, 17-04, 17-05, best-split-finder-stage-1, cuda-on-device-training]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Additive gain dispatch (D-09): net-new smoothing fns added below the reused non-smoothing fns; reused fn signatures byte-unchanged so the host path (feature_histogram_categorical.rs) is untouched"
    - "Dual host/device int→float cast: `num_data as f64` (real host impl + #[cube]-lowered device cast) instead of f64::cast_from (whose host stub panics 'Unexpanded Cube functions should not be called')"
    - "WR-05 literal pinning in every f32 mirror (bare 1.0f32 denominators) so cubecl cannot widen the blend to f64 on the hip path"

key-files:
  created: []
  modified:
    - "crates/lgbm-compute/src/gain.rs — additive smoothing gain path + #[cube]-promoted given-output gain + f32 mirrors + 2 unit tests"

key-decisions:
  - "Used `num_data as f64`/`as f32` rather than f64::cast_from: cast_from's host stub panics when the #[cube] fn is called unexpanded (host unit tests + CPU anchor), whereas `as` has a real host impl and the #[cube] macro lowers it to a device cast"
  - "Transcribed the blend with verbatim C++ precedence `ret * nps / (nps + 1)` == `(ret*nps)/(nps+1)`, NOT `ret * (nps/(nps+1))` (the two differ in the last bit)"
  - "get_leaf_gain_smoothed reuses calculate_splitted_leaf_output for the base output (bit-identical to inlining the branch) then applies the blend — mirrors cuda_leaf_splits.hpp structure exactly"

patterns-established:
  - "Additive-only gain evolution: new template-flag branches (USE_SMOOTHING) become new fns; existing default-path fns stay byte-unchanged and the launcher dispatches at runtime"

requirements-completed: [ODL-11]

# Metrics
duration: 4min
completed: 2026-07-01
status: complete
---

# Phase 17 Plan 02: On-Device Best-Split Finder — Smoothing Gain Path Summary

**Net-new `USE_SMOOTHING` (path_smooth) output-blend + given-output gain path added to `crate::gain` as additive `#[cube]` functions (f64 + f32 mirrors), with `get_leaf_gain_given_output` promoted to `#[cube]` so the smoothing gain runs on device — reused non-smoothing gain fns byte-unchanged.**

## Performance

- **Duration:** ~4 min
- **Started:** 2026-07-01T10:58:51+09:00
- **Completed:** 2026-07-01T11:02:28+09:00
- **Tasks:** 2 (both TDD: RED → GREEN)
- **Files modified:** 1

## Accomplishments
- Promoted `get_leaf_gain_given_output` from a plain host `fn` to `#[cube]` (still a plain Rust fn, so the monotone host caller in `monotone_constraints.rs` is byte-unchanged) and added the `get_leaf_gain_given_output_f32` mirror.
- Added `calculate_splitted_leaf_output_smoothed` (`#[cube]`, f64) — form (B) output-blend toward `parent_output`, verbatim from `cuda_leaf_splits.hpp:74-90`.
- Added `get_leaf_gain_smoothed` (`#[cube]`, f64) — form (D) given-output gain at the blended output (`cuda_leaf_splits.hpp:117-121`), NOT the `sg²/(h+l2)` closed form.
- Added f32 mirrors of both smoothing fns with all literals pinned f32 (WR-05).
- Proved bit-exact consistency: the given-output form equals the closed-form leaf gain at `o = -g/(h+l2)`, and the blend + form-(D) gain match a hand-computed reference.

## Task Commits

Each task was committed atomically (TDD RED → GREEN):

1. **Task 1: Promote get_leaf_gain_given_output to #[cube] + f32 mirror**
   - `4dad410` (test — RED: failing given-output closed-form consistency test)
   - `d119cbc` (feat — GREEN: #[cube] promotion + f32 mirror)
2. **Task 2: Net-new USE_SMOOTHING output-blend gain path**
   - `987560c` (test — RED: failing smoothing output-blend reference test)
   - `eea0962` (feat — GREEN: calculate_splitted_leaf_output_smoothed + get_leaf_gain_smoothed + f32 mirrors)

## Files Created/Modified
- `crates/lgbm-compute/src/gain.rs` — additive `#[cube]` smoothing gain path (`calculate_splitted_leaf_output_smoothed(+_f32)`, `get_leaf_gain_smoothed(+_f32)`), `#[cube]`-promoted `get_leaf_gain_given_output(+_f32)`, and two unit tests (`given_output_matches_closed_form`, `smoothing_blend_matches_reference`).

## Decisions Made
- **`num_data as f64` over `f64::cast_from`:** `f64::cast_from`'s host implementation is a panic stub ("Unexpanded Cube functions should not be called"), which fired because the smoothing fns are called on the host (unit tests + the CPU anchor), not only launched as kernels like the existing `split.rs` uses. The `as` cast has a real host impl and the `#[cube]` macro lowers it to a device cast — satisfying the dual host/device requirement. (Not in the plan text, which suggested `f64::cast_from`; see Deviations.)
- **Verbatim blend precedence:** transcribed `ret * nps / (nps + 1)` as `(ret*nps)/(nps+1)` (left-to-right), not `ret*(nps/(nps+1))`, matching the C++ associativity to the last bit.
- **Base-output reuse:** `calculate_splitted_leaf_output_smoothed` calls `calculate_splitted_leaf_output` for the base, bit-identical to inlining the branch, preserving the non-smoothing bit-identity.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Switched `num_data` int→float cast from `f64::cast_from` to `num_data as f64`**
- **Found during:** Task 2 (net-new smoothing path)
- **Issue:** The plan/reference comment suggested `f64::cast_from(num_data)` (the idiom used in `split.rs`, which is only ever launched as a kernel). But the smoothing fns must be host-callable (unit tests + the CPU anchor call them directly), and `f64::cast_from`'s unexpanded host body panics with "Unexpanded Cube functions should not be called" — the `smoothing_blend_matches_reference` test panicked.
- **Fix:** Used `num_data as f64` (f32 mirror: `num_data as f32`), which has a real host implementation and is lowered to a device cast by the `#[cube]` macro. Updated the module comment to document the rationale.
- **Files modified:** `crates/lgbm-compute/src/gain.rs`
- **Verification:** `smoothing_blend_matches_reference` passes on host; the full `gain` suite (7 tests) green; `cargo build -p lgbm-treelearner` green.
- **Committed in:** `eea0962` (Task 2 GREEN commit)

---

**Total deviations:** 1 auto-fixed (1 blocking)
**Impact on plan:** Necessary for the fns to be host-testable and CPU-anchor-callable per the plan's own "callable as a plain fn" requirement. No scope creep — same math, same `#[cube]` device lowering.

## Issues Encountered
- `f64::cast_from` host panic (see Deviation 1) — resolved by the `as` cast. No other issues.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Wave-2 stage-1 (17-03) can now dispatch on the runtime `use_smoothing` flag: the existing (reused, byte-unchanged) gain fns for `use_smoothing=false`, and the new `*_smoothed` fns for `use_smoothing=true` — all `#[cube]` so both branches run on device.
- The three reused non-smoothing gain functions preserve D-02 bit-identity; the default host path (`feature_histogram_categorical.rs`) is byte-unchanged (D-09).
- No blockers.

## Self-Check: PASSED

---
*Phase: 17-on-device-best-split-finder*
*Completed: 2026-07-01*
