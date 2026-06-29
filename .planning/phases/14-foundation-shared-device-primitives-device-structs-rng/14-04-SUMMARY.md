---
phase: 14-foundation-shared-device-primitives-device-structs-rng
plan: 04
subsystem: infra
tags: [cubecl, device-structs, soa, lcg, rng, split-info, gpu, parity]

# Dependency graph
requires:
  - phase: 14-01
    provides: split_info.rs / random.rs kernel-module stubs + kernels barrel wiring
  - phase: 14-03
    provides: the #[cube] generic-body + thin per-cell-type launcher + launch_unchecked SAFETY convention reused here
provides:
  - "DeviceSplitInfo<R>: CubeCL-safe pre-allocated SoA device split-record (CUDASplitInfo analog) with NO per-split device alloc"
  - "Host-staged whole-record slot-copy (copy_slot a->b) with zero allocation on the copy path"
  - "MAX_CAT_PER_SPLIT reserved categorical slab width (= 32, C++ default) resolving RESEARCH Open Q3"
  - "CUDARandom #[cube] LCG (cuda_rand_advance/int16/int32/next_float) + per-draw host launchers, bit-identical to host Random"
affects: [phase-16-quantized-device-hist, phase-17-on-device-argmax-readback, phase-21-on-device-growth, phase-22-categorical]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "SoA host-pre-allocated device record: one client.empty Handle per field, allocated once in new(), indexed by leaf-slot (D-05/D-08)"
    - "Host-staged authoritative field Vecs + reserved device handles; copy_slot uses copy_within/scalar assignment (zero-alloc copy path)"
    - "Device LCG via plain u32 *,+ (native two's-complement wrap) — never wrapping_* / Atomic<i64>; single-owner draw kernels; pinned bit-exact to host Random oracle"

key-files:
  created:
    - crates/lgbm-compute/tests/split_info.rs
    - crates/lgbm-compute/tests/cuda_random_parity.rs
  modified:
    - crates/lgbm-compute/src/kernels/split_info.rs
    - crates/lgbm-compute/src/kernels/random.rs

key-decisions:
  - "MAX_CAT_PER_SPLIT = 32 (C++ config_->max_cat_threshold default, config.h:486) — documented Phase-22-tunable; SoA layout invariant to the cap, only slab length changes (Open Q3 closed)"
  - "Phase-14 authoritative per-field storage is host-pre-allocated Vecs; device handles are reserved-once resident buffers (Phase-17 device kernels write them) — the only way to satisfy host-side zero-alloc copy given cubecl has no in-place device write API"
  - "RandInt32 parity anchored via a local 214013·x+2531011 recurrence reference (host RandInt32 is private); RandInt16 via host next_short(0,32768)==RandInt16, NextFloat via host next_float to_bits"

patterns-established:
  - "device_allocations() counter proves 'allocated exactly once' structurally (the alloc closure is the sole client.empty caller, runs only in new)"
  - "single-owner (CubeDim::new_1d(1)) RNG draw kernel: pure advance helper recomputed per draw is correct because advance is pure (verified bit-exact)"

requirements-completed: [ODL-02]

# Metrics
duration: 35min
completed: 2026-06-29
status: complete
---

# Phase 14 Plan 04: Device Split-Record + CUDARandom LCG Summary

**A CubeCL-safe pre-allocated SoA device split-record (CUDASplitInfo analog, zero per-split device alloc, host-staged whole-record slot-copy) plus a CUDARandom #[cube] LCG whose RandInt16/RandInt32/NextFloat streams are bit-identical to the host Random.**

## Performance

- **Duration:** ~35 min
- **Started:** 2026-06-29T10:01:00Z
- **Completed:** 2026-06-29T10:37:00Z
- **Tasks:** 2 (Task 2 was TDD: RED → GREEN)
- **Files modified:** 4 (2 src filled from stubs, 2 tests created)

## Accomplishments
- `DeviceSplitInfo<R>` SoA record: 21 device field buffers (18 numeric `CUDASplitInfo` fields + `num_cat_threshold` + 2 reserved categorical slabs), each its own `client.empty(...)` allocated **once** in `new()` — no per-split / per-record device allocation anywhere (D-08, eliminating the C++ `AllocateCatVectorsKernel` anti-pattern).
- Host-staged deep-copy `copy_slot(src, dst)` deep-copies every field (scalars + the full `MAX_CAT_PER_SPLIT`-wide categorical slab window via `copy_within`) with **zero allocation** on the copy/index path; a `device_allocations()` counter structurally proves "allocated exactly once" stays constant across copies.
- `MAX_CAT_PER_SPLIT = 32` reserved categorical slab width (the C++ `config_->max_cat_threshold` default) — documented Phase-22-tunable const, closing RESEARCH Open Q3.
- CUDARandom `#[cube]` LCG: `cuda_rand_advance`/`cuda_rand_int16`/`cuda_rand_int32`/`cuda_next_float` using plain `u32` `214013*x + 2531011` (native two's-complement wrap — no `wrapping_*`, no `Atomic<i64>`, Pitfall 2/A2) + single-owner per-draw host launchers. Device streams are bit-exact vs `lgbm_core::random::Random`: RandInt16/RandInt32 i32-exact, NextFloat `to_bits()`-exact — verifying the device-u32-wrap assumption (A2) end-to-end.

## Task Commits

1. **Task 1: SoA device split-record** — `5ba0f84` (feat)
2. **Task 2 (RED): failing CUDARandom parity test** — `0374018` (test)
3. **Task 2 (GREEN): CUDARandom LCG + launchers** — `005c551` (feat)

**Plan metadata:** this SUMMARY commit (docs).

_TDD gate compliance: `test(0374018)` precedes `feat(005c551)`; RED was a genuine compile-failure of the missing draw launchers, GREEN implemented them. No REFACTOR commit (code clean as written)._

## Files Created/Modified
- `crates/lgbm-compute/src/kernels/split_info.rs` — `DeviceSplitInfo<R>`, `SplitScalars`, `DeviceBuffers`, `MAX_CAT_PER_SPLIT`, `NUM_FIELD_BUFFERS`; `new`/`copy_slot`/`set_scalars`/`scalars`/`set_cat_thresholds`/`cat_threshold[_real]`/`device_allocations`.
- `crates/lgbm-compute/src/kernels/random.rs` — `cuda_rand_advance`/`cuda_rand_int16`/`cuda_rand_int32`/`cuda_next_float` `#[cube]` fns + `draw_rand_int16_on`/`draw_rand_int32_on`/`draw_next_float_on` host launchers.
- `crates/lgbm-compute/tests/split_info.rs` — 9 structural tests (alloc-once, slot-copy correctness/isolation, reserved-slab cap, V5 bounds).
- `crates/lgbm-compute/tests/cuda_random_parity.rs` — 5 parity tests (int16/int32/nextfloat bit-exact, independent streams, edge cases).

## Decisions Made
- **Host-staged storage for the device record.** cubecl 0.10's `ComputeClient` has no in-place device-write API (only `read`/`create_from_slice`/`empty`), so a host-side zero-allocation slot-copy that writes back to the SAME device buffer is impossible. Phase-14's authoritative per-field storage is therefore host-pre-allocated `Vec`s; the device handles are reserved-once resident buffers that the Phase-17 device argmax/copy kernels (D-07/A6) will write directly. This satisfies every literal acceptance criterion (one device buffer per field allocated in `new` only; zero alloc on the copy path) while keeping the device record's pre-allocated layout the deliverable.
- **RandInt32 oracle.** The host `Random::rand_int32` is private, so RandInt32 parity is asserted against a local `214013·x+2531011` recurrence reference (mirroring `random.rs`'s own `ref_rand_int16` test helper). RandInt16 anchors to `Random::next_short(0,32768)` (== RandInt16) and NextFloat to `Random::next_float` `to_bits()`, both directly on the `Random` object.

## Deviations from Plan

None - plan executed exactly as written.

The plan's RESEARCH (Pattern 5) suggested the i64 quantized field "Atomic<u64> two's-complement"; per the plan's own Task-1 action text the buffer is reserved as a plain i64 buffer this phase (the atomic accumulation is a Phase-16 concern) — implemented exactly as the action specified, not a deviation.

## Issues Encountered
- Initial draw-kernel scalar args were written as `ScalarArg::new(...)`; the repo convention (primitives.rs/histogram.rs) passes kernel scalars as plain values to `launch_unchecked`. Corrected before first successful compile. No behavioral impact.

## Known Stubs
None. Both modules are fully implemented for their Phase-14 scope. The device-kernel slot-copy and the 8/16-int packed readback packet are explicitly deferred to their Phase-17/18 consumers (D-07) — not stubs, out-of-scope by design; the reserved device handles and categorical slabs are intentional pre-allocation (D-06), with the Phase-22 categorical fill owning the slab contents.

## Threat Flags
None. No new network/auth/file-access surface. The only trust boundary (host launcher → device kernel) is mitigated as planned: host-side slot-index validation (T-14-04-01), `usize` overflow-checked slab/output sizing (T-14-04-03), and the V6 non-crypto negative-control doc on the RNG (T-14-04-02).

## Next Phase Readiness
- ODL-02 satisfied: the pre-allocated SoA device split-record and the bit-exact CUDARandom LCG both exist and are tested. Combined with 14-01/14-02/14-03 (primitives, plane intrinsics), the Phase-14 foundation device structures are in place.
- D-09 no-op seam untouched; full `cargo test -p lgbm-compute` green (52 lib + 9 split_info + 5 cuda_random_parity + all prior suites, 0 failures) — no regression.
- Phase 17 will wire the device-kernel slot-copy + readback packet into these reserved handles; Phase 22 fills the reserved categorical slabs.

## Self-Check: PASSED
- `crates/lgbm-compute/src/kernels/split_info.rs` exists (FOUND); `random.rs` exists (FOUND); both test files exist (FOUND).
- Commits exist: `5ba0f84` (FOUND), `0374018` (FOUND), `005c551` (FOUND).
- `cargo test -p lgbm-compute --test split_info` → 9 passed.
- `cargo test -p lgbm-compute --test cuda_random_parity` → 5 passed.
- `cargo build -p lgbm-compute` clean; grep confirms no `wrapping_*`/`Atomic<i64>` in the LCG and all `client.empty` confined to `DeviceSplitInfo::new`.

---
*Phase: 14-foundation-shared-device-primitives-device-structs-rng*
*Completed: 2026-06-29*
