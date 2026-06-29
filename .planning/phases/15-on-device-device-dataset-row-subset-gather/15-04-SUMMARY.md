---
phase: 15-on-device-device-dataset-row-subset-gather
plan: 04
subsystem: compute
tags: [cubecl, cuda-on-device, copy-subrow, row-subset-gather, bagging, rng, parity]

# Dependency graph
requires:
  - phase: 15-on-device-device-dataset-row-subset-gather
    plan: 01
    provides: copy_subrow_kernel<B: Int> D-07 skeleton + bagging-draw stub + the 3 ODL-04 parity tests + inline host_bag_data_indices reference
  - phase: 14-foundation-shared-device-primitives-device-structs-rng
    provides: draw_next_float_on launcher (f32-exact NextFloat stream), cpu f64 anchor (cpu_client), BinColumn narrow store + gather oracle
provides:
  - copy_subrow_on — V5-validated, per-column native-width row-subset gather launcher (same width in/out, D-07)
  - bagging_draw_on — on-device bagging draw reusing draw_next_float_on, host-route to [in-bag asc] ++ [OOB desc], anchored to host bag_data_indices
affects: [16-histogram-constructor]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "launch_native! per-width dispatch (partition.rs precedent) applied to the gather: u8/u16/u32 monomorph, native-width upload/readback, output BinColumn keeps the source width (D-07, no widen)"
    - "On-device subset-selection draw = device RNG stream (one task per BAGGING_RAND_BLOCK block) + HOST-side route (Open Question 2): simplest fully-anchored shape, f32 NextFloat promoted to f64 before < bagging_fraction (Pitfall 6)"
    - "Avoid i32::div_ceil under `use cubecl::prelude::*` — it collides with the cubecl DivCeil trait (unstable_name_collisions); use explicit (n + b - 1) / b"

key-files:
  created: []
  modified:
    - crates/lgbm-compute/src/kernels/copy_subrow.rs

key-decisions:
  - "Validate every used_indices[i] in [0, num_data) before the unsafe launch (T-15-IDX): negative -> ComputeError::Runtime, >= num_data -> ComputeError::BinIndexOutOfRange; the C++ raw-pointer subset table has no bounds check, the Rust port adds one"
  - "bagging_draw_on returns the Vec<i32> index layout only (the test asserts the full layout, which structurally encodes the in-bag count); the Wave-0 docstring's (indices, cnt) tuple framing was narrowed to match the actual single-Vec signature"
  - "Seeds[b] = bagging_seed + b is the iteration-0 anchor (Pitfall 5); multi-iteration per-block RNG continuity is an explicit Phase-21 host-state-supply seam, documented in the function doc"

requirements-completed: [ODL-04]

# Metrics
duration: 4min
completed: 2026-06-29
status: complete
---

# Phase 15 Plan 04: CopySubrow Row-Subset Gather + On-Device Bagging Draw Summary

**Filled `copy_subrow.rs`: a V5-validated, per-column native-width `CopySubrow` gather that compacts any host-supplied (GOSS-shaped) index set bit-identically to `BinColumn::gather`, plus an on-device bagging draw that reuses the Phase-14 `draw_next_float_on` launcher and routes host-side to a `[in-bag asc] ++ [OOB desc]` layout bit-for-bit equal to the inline host `bag_data_indices` reference over ≥2 blocks.**

## Performance

- **Duration:** ~4 min
- **Tasks:** 2
- **Files modified:** 1 (`crates/lgbm-compute/src/kernels/copy_subrow.rs`)

## Accomplishments
- `copy_subrow_on` implemented: validates every `used_indices[i] ∈ [0, num_data)` BEFORE the unsafe launch (T-15-IDX — negative → `Runtime`, `>= num_data` → `BinIndexOutOfRange`), `checked_mul` on output byte sizing (T-15-PART), per-width `launch_native!`-style dispatch driving `copy_subrow_kernel::launch::<B, R>`, reads back into a SAME-width `BinColumn` (D-07, no widen). All cubecl unsafe confined to the launcher with a SAFETY comment (CMP-01).
- `bagging_draw_on` implemented: one device RNG task per `BAGGING_RAND_BLOCK` (=1024) block seeded `bagging_seed + block`, calls `draw_next_float_on(&seeds, 1024)`, then the host route promotes each f32 `NextFloat` to f64 (Pitfall 6) and compares `< bagging_fraction`, filling in-bag left / OOB right and reversing the OOB tail → `[in-bag asc] ++ [OOB desc]`.
- All 4 `copy_subrow_parity` tests green on the cpu f64 anchor: `gather_matches_host_all_widths`, `gather_arbitrary_indices`, `out_of_range_index_rejected_before_launch`, `bagging_draw_matches_host` (spans 3 blocks at num_data=2500).
- Zero crate cycle: `BAGGING_RAND_BLOCK` is a local const; no `lgbm-boosting` `use` or Cargo dep (only explanatory doc comments mention the prohibition).

## Task Commits

Each task was committed atomically:

1. **Task 1: CopySubrow gather kernel launcher + V5 index validation** — `74181b6` (feat)
2. **Task 2: On-device bagging draw anchored to host bag_data_indices** — `84bbf4b` (feat)

## Files Modified
- `crates/lgbm-compute/src/kernels/copy_subrow.rs` — filled the two Wave-0 `todo!()` stubs (`copy_subrow_on`, `bagging_draw_on`); added a `use crate::kernels::random::draw_next_float_on;` import. The `copy_subrow_kernel<B: Int>` D-07 skeleton was unchanged (Wave-0 already shipped it real).

## Decisions Made
- Followed the plan as specified for both tasks (validation precedent, width dispatch, anchor discipline).
- Narrowed the `bagging_draw_on` doc from the Wave-0 `(indices, cnt)` tuple framing to the actual `Vec<i32>` return — the test asserts the full layout, which structurally encodes the in-bag split point.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] `i32::div_ceil` name-collision warning under the cubecl prelude**
- **Found during:** Task 2
- **Issue:** `nd.div_ceil(BAGGING_RAND_BLOCK)` on the `i32` `num_data` triggered an `unstable_name_collisions` warning — `use cubecl::prelude::*` brings the cubecl `DivCeil` trait into scope, which provides a colliding `div_ceil`. The plan requires no warnings / clippy clean.
- **Fix:** Replaced with explicit ceil-div `(nd + BAGGING_RAND_BLOCK - 1) / BAGGING_RAND_BLOCK`, which also exactly mirrors the host reference's arithmetic. (The Task-1 `(num_used as u32).div_ceil` does NOT collide — the warning is specific to the `i32` path.)
- **Files modified:** `crates/lgbm-compute/src/kernels/copy_subrow.rs`
- **Committed in:** `84bbf4b` (Task 2 commit)

**2. [Rule 1 - Lint] `needless_range_loop` clippy warning on the route loop**
- **Found during:** Task 2
- **Issue:** `for i in 0..nd_usize { ... draws[i] ... }` tripped `clippy::needless_range_loop`.
- **Fix:** Rewrote as `for (i, &draw_f32) in draws.iter().enumerate().take(nd_usize)` — `take(nd_usize)` because the device draws a full 1024 per block (the last block's tail is correctly ignored, matching the host which only draws `nd` times).
- **Files modified:** `crates/lgbm-compute/src/kernels/copy_subrow.rs`
- **Committed in:** `84bbf4b` (Task 2 commit)

---

**Total deviations:** 2 auto-fixed (1 blocking warning, 1 lint). Both confined to Task 2's own new code; no scope creep.
**Impact on plan:** None — both keep the implementation warning- and clippy-clean per the plan's verification block.

## Threat Mitigations Applied
- **T-15-IDX** (untrusted `used_indices` → unsafe device read): every index host-validated in `[0, num_data)` before launch, returning `ComputeError` (`out_of_range_index_rejected_before_launch` proves it).
- **T-15-PART** (draw/gather length sizing overflow): `checked_mul(num_used * elem_width)` in the gather; `draw_next_float_on` already `checked_mul`s `n_blocks * BLOCK`.
- **T-15-CMP** (`from_raw_parts` in the launcher): all cubecl unsafe confined to `copy_subrow_on` with a SAFETY comment proving handle sizing + lifetime.

## Known Stubs
None — both owned symbols are fully implemented. (Other Phase-15 stubs in `row_data.rs`/`column_data.rs` are out of this plan's scope, resolved by 15-02/15-03 which are already complete.)

## Verification
- `cargo test -p lgbm-compute --test copy_subrow_parity` — 4/4 green on the cpu f64 anchor.
- `cargo clippy -p lgbm-compute --tests` — 0 warnings on `copy_subrow.rs`.
- `cargo build -p lgbm-compute` — 0 warnings.
- No `lgbm-boosting` `use`/dep; route promoted f32→f64; same width in/out.

## Next Phase Readiness
- ODL-04 complete. The row-subset gather + bagging draw are anchor-pinned to the host (never GPU-vs-GPU). Phase-16 histogram construction can consume the compacted device subset.
- Documented seam: multi-iteration per-block RNG continuity needs host block-state supply (Phase-21).
- No blockers.

## Self-Check: PASSED
`copy_subrow.rs` modified on disk; both task commits (`74181b6`, `84bbf4b`) present in git history; all 4 parity tests pass.

---
*Phase: 15-on-device-device-dataset-row-subset-gather*
*Completed: 2026-06-29*
