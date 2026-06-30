---
phase: 16-on-device-histogram-constructor
plan: 02
subsystem: infra
tags: [cubecl, histogram, subtraction-trick, device-memory, gpu, cpu-anchor]

# Dependency graph
requires:
  - phase: 14-foundation-shared-device-primitives-device-structs-rng
    provides: DeviceSplitInfo allocate-exactly-once counted client.empty pattern + ComputeError taxonomy
  - phase: 16-on-device-histogram-constructor
    provides: subtract_hist_kernel (the verbatim FeatureHistogram::Subtract f64 fold the rotation contract round-trips through)
provides:
  - "HistArena: pre-allocated-once histogram slot pool (USED_HISTOGRAM_BUFFER_NUM analog)"
  - "The {parent, smaller, larger} hist_t** handle-rotation contract (D-02)"
  - "rotate(): larger derived in-place in the parent buffer, smaller into a fresh non-aliasing slot, zero realloc / zero bulk copy"
affects: [16-04, 18-data-partition-tree-mutation, histogram-subtraction-trick, on-device-tree-learner]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Counted single client.empty closure + assert(allocations == num_slots) (D-09 allocate-exactly-once)"
    - "hist_t** pointer rotation as host-side index bookkeeping (no device copy)"
    - "Anchor the index contract on the cpu f64 fold via the verbatim subtract kernel (never GPU-vs-GPU)"

key-files:
  created:
    - crates/lgbm-compute/src/kernels/histogram_arena.rs
  modified:
    - crates/lgbm-compute/src/kernels/mod.rs

key-decisions:
  - "Slot pool sized in f64 hist_t cells; slab byte size via checked_mul before any alloc (T-16-02-01)"
  - "rotate() rejects num_slots < 2 with a typed ComputeError (cannot supply a non-aliasing smaller slot from a 1-slot pool)"
  - "Fresh smaller slot picked as (parent_idx + 1) % num_slots — guaranteed != parent_idx for num_slots >= 2"
  - "Round-trip anchor test launches subtract_hist_kernel directly with the arena's larger slot as output to prove in-place landing"

patterns-established:
  - "Pattern 1: HistArena mirrors DeviceSplitInfo::new — one counted client.empty per slot, asserted, frozen counter"
  - "Pattern 2: rotation reassigns role indices only; allocation count is the structural proof of no-copy/no-realloc"

requirements-completed: [ODL-10]

# Metrics
duration: 4min
completed: 2026-06-30
status: complete
---

# Phase 16 Plan 02: On-Device Histogram Arena + hist_t** Rotation Contract Summary

**HistArena pre-allocates a fixed histogram slot pool exactly once (counted + asserted, D-09) and exposes the explicit {parent, smaller, larger} hist_t** rotation contract (D-02) where the larger child is derived in-place in the parent's buffer and the smaller takes a fresh non-aliasing slot — zero realloc, zero bulk copy, anchor-tested on the cpu f64 fold.**

## Performance

- **Duration:** 4 min
- **Started:** 2026-06-30T21:47:14Z
- **Completed:** 2026-06-30T21:51:18Z
- **Tasks:** 2
- **Files modified:** 2 (1 created, 1 modified)

## Accomplishments
- New `histogram_arena.rs` module (registered ungated in `mod.rs` so the cpu f64 anchor exercises it, D-08).
- `HistArena::new` allocates exactly `num_slots` device buffers via a single counted `client.empty` closure, asserted `== num_slots`; rejects zero counts and overflowing slabs with typed `ComputeError` (V5 / T-16-02-01).
- `HistArena::rotate()` implements the `hist_t**` pointer rotation: `larger_idx <- parent_idx` (in-place derivation), `smaller_idx <- fresh slot`, enforcing the no-alias invariant `smaller_idx != parent_idx` (T-16-02-02) and freezing the allocation counter (no realloc, no bulk copy).
- Subtraction-trick round-trip anchor-tested in isolation on the cpu f64 fold: `out = parent - smaller` via the verbatim `subtract_hist_kernel` lands the derived histogram in the larger (== old parent) slot.

## Task Commits

Each task was committed atomically:

1. **Task 1: HistArena — allocate-exactly-once slot pool (D-02/D-09)** - `d02dccb` (feat)
2. **Task 2: hist_t** pointer rotation — larger in-place, smaller fresh, no aliasing (D-02)** - `d83eca0` (feat)

_TDD-style: tests authored alongside each task; all 10 module tests green._

## Files Created/Modified
- `crates/lgbm-compute/src/kernels/histogram_arena.rs` - The `HistArena` slot pool, the `{parent, smaller, larger}` role indices + handle accessors, `new()` (allocate-once, counted, checked_mul, zero-reject) and `rotate()` (in-place larger, fresh smaller, no-alias invariant), plus 10 anchor tests.
- `crates/lgbm-compute/src/kernels/mod.rs` - `pub mod histogram_arena;` registration (ungated, like the Phase-14/15 modules).

## Decisions Made
- Slots sized in f64 `hist_t` cells; slab byte length overflow-checked via `checked_mul` before any allocation (mirrors `split_info.rs:276-284`).
- `rotate()` returns `Result` and rejects `num_slots < 2` with a typed error rather than aliasing the parent — a 1-slot pool cannot supply a non-aliasing smaller slot.
- Fresh smaller slot = `(parent_idx + 1) % num_slots`, which is provably `!= parent_idx` for `num_slots >= 2`.
- The round-trip test launches `subtract_hist_kernel` directly (the production verbatim kernel) with the arena's larger slot handle as the output, proving the result lands in the parent's buffer in-place — on the cpu f64 anchor, never GPU-vs-GPU.

## Deviations from Plan

None - plan executed exactly as written.

The only minor adjustment was a test-mechanics fix: `unwrap_err()` requires the `Ok` type to be `Debug`, which `HistArena` is not (it owns CubeCL `Handle`s); the rejection tests use `assert!(matches!(res, Err(ComputeError::Runtime { .. })))` instead. This is a test idiom choice, not a behavioral deviation.

## Issues Encountered
- Initial compile error: `Result::unwrap_err` needs `T: Debug`. Resolved by asserting on the `Result` with `matches!` rather than unwrapping the error — no `Debug` derive added to the handle-owning struct.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- The `{parent, smaller, larger}` handle contract is ready for plan 16-04 (the build->fix->subtract entry) to drive.
- The whole-tree pool SWAP (`SplitTreeStructureKernel`) remains correctly deferred to Phase 18 (16-CONTEXT §9) — this plan demonstrates the rotation for one triple in isolation only.
- Additive + behind the OFF-by-default `LGBM_CUDA_ON_DEVICE` seam (D-07): full workspace `cargo test` stays green.

## Self-Check: PASSED

- `crates/lgbm-compute/src/kernels/histogram_arena.rs` — FOUND
- `mod.rs` contains `pub mod histogram_arena;` — FOUND
- Commit `d02dccb` (Task 1) — FOUND
- Commit `d83eca0` (Task 2) — FOUND

---
*Phase: 16-on-device-histogram-constructor*
*Completed: 2026-06-30*
