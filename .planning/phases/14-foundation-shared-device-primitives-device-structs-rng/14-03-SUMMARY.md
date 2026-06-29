---
phase: 14-foundation-shared-device-primitives-device-structs-rng
plan: 03
subsystem: compute
tags: [cubecl, primitives, prefix-sum, reduction, bitonic-argsort, plane-intrinsics, lds, gpu, rocm]

# Dependency graph
requires:
  - phase: 14-foundation-shared-device-primitives-device-structs-rng
    plan: "01"
    provides: "plane-intrinsic lowering finding (hip lowers all four; cpu has no plane) + wired-but-empty primitives.rs stub"
  - phase: 04-compute-seam
    provides: "histogram.rs #[cube] generic-body + thin-wrapper + launch_unchecked SAFETY + SharedMemory/sync_cube LDS prior art; runtime capability probe"
provides:
  - "Full-depth block + multi-kernel global prefix-sum (inclusive/exclusive), f64 cpu anchor + f32 mirror, ONE reused client.empty scratch"
  - "Shuffle reductions sum/max/min + dot-product (f64 cpu anchor + f32 mirror); Open Q2 f64-order policy RESOLVED per reduction"
  - "Single-block index-only bitonic argsort with C++-matched comparator/tie order (permutation bit-exact vs serial reference)"
  - "GPU plane-intrinsic variants (rocm-gated): plane_inclusive_sum/exclusive_sum block scan + LDS staging, plane_sum/max/min + dot reductions — cross-validated in 14-06"
  - "Serial-f64-reference-anchored self-test suite (tests/primitives_self.rs), 13 tests green on the cpu anchor"
affects: [14-06, 15-minimal-on-device-growth, 16, 17, 18]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Single-source generic #[cube] body + thin per-cell-type (f64/f32) launch wrappers (histogram idiom) for every numeric primitive"
    - "3-launch global prefix-sum (block scan -> block-totals scan -> add-back) with ONE reused client.empty scratch (Pattern 3, D-05 pre-alloc)"
    - "Single-owner serial fold as the cpu anchor (CubeDim 1); plane-intrinsic + LDS-staging variant as the rocm GPU leg"
    - "Index-only bitonic argsort: comparator reads keys[indices[i]], swaps only indices; sentinel-padded to next pow2; serial single-owner walk == parallel form (disjoint compare-swap pairs per stage)"

key-files:
  created:
    - crates/lgbm-compute/tests/primitives_self.rs
  modified:
    - crates/lgbm-compute/src/kernels/primitives.rs

key-decisions:
  - "Open Q2 (f64 reduction-order) RESOLVED per reduction: sum + dot-product are BIT-EXACT vs a serial Rust f64 fold in ASCENDING index order (matched-order policy — the cpu anchor's single-owner fold IS that order); max/min are BIT-EXACT and order-INDEPENDENT (selection only, no rounding). f32 hip warp-tree reductions are held to ~1e-6 only (deferred to 14-06), never asserted GPU-vs-GPU (def-f8u-01)."
  - "Argsort tie convention: mirror the AMD-fork BitonicArgSort_1024 comparator EXACTLY — strict `>`, ascending parity `outer_segment_index & 1 == 0`, sentinel padding (+inf ascending / -inf descending) to the next power of two; the permutation is asserted bit-exact (integer, no float tolerance) vs a serial Rust reference implementing the identical comparator, including a tie-rich input."
  - "cpu anchor stays the single-owner serial fold (14-01: cubecl-cpu has NO plane support); the plane intrinsics are the rocm GPU leg only, gated #[cfg(feature = \"rocm\")]."
  - "Recursive >1024-block global prefix-sum DEFERRED to Phase 15 (its first real consumer); the Phase-14 kernel GUARDS num_blocks <= 1024 with a typed ComputeError rather than silently truncating."

requirements-completed: [ODL-01]

# Metrics
duration: ~50 min
completed: 2026-06-29
status: complete
---

# Phase 14 Plan 03: Full-Depth Grow-Loop Primitives Summary

**Authored the full-depth grow-loop primitive subset (D-01) in `primitives.rs` — block + multi-kernel global prefix-sum (inclusive/exclusive), shuffle reductions (sum/max/min + dot-product), and single-block index-only bitonic argsort — each as a single-owner serial f64 cpu anchor (bit-exact vs a serial Rust reference) plus an f32/plane-intrinsic GPU leg, with the Open-Q2 f64-order policy and the C++-matched argsort tie convention recorded.**

## Performance

- **Duration:** ~50 min
- **Completed:** 2026-06-29
- **Tasks:** 3 (each TDD: RED test commit -> GREEN feat commit)
- **Files modified:** 2 (1 created, 1 filled from stub)
- **Tests:** 13 serial-anchored self-tests, all green on the cpu anchor

## Accomplishments

### Task 1 — Block + multi-kernel global prefix-sum (inclusive/exclusive)
- Single-source generic `block_scan_body<N>` + per-cell-type f64/f32 launch wrappers (histogram idiom). The 3-launch global structure (block scan -> block-totals exclusive scan -> add-back) with **ONE reused `client.empty` scratch** for the block totals (Pattern 3, D-05 pre-alloc — no per-launch alloc).
- cpu anchor = single-owner serial fold per block (`CubeDim::new_1d(1)`, one cube per block via `CUBE_POS_X`); no cross-cube barrier assumed. Bit-exact vs a serial Rust scan on within-block, multi-block (64-block over 1000 elements), boundary (block_size=2 over 7), and empty/single inputs.
- V5 boundary: `block_size >= 1` and `num_blocks <= 1024` guard (the Phase-15 recursion deferral) -> typed `ComputeError` before any `launch_unchecked`.

### Task 2 — Shuffle reductions (sum/max/min, dot-product)
- Single-owner ordered folds (sum/dot ascending; max/min selection) + per-cell-type wrappers. **Open Q2 RESOLVED per reduction** (see Decisions). Empty handling: sum/dot -> 0.0; max/min -> `LengthMismatch`; dot `a.len() != b.len()` -> `LengthMismatch`.
- Anchored to a serial f64 reference; bit-exact on within-plane and cross-plane (1000-element) inputs.

### Task 3 — Single-block index-only bitonic argsort
- Comparator reads `keys[indices[i]]` (indirection) and swaps **only `indices`** — `keys` passed as `&Array<f32>` (provably unmutated; the test reads keys back and asserts equality). Mirrors the AMD-fork `cuda_algorithms.hpp` `BitonicArgSort_1024` comparator/tie order EXACTLY. Permutation bit-exact vs a serial Rust reference on distinct AND tie-rich inputs; single/empty handled.
- Single-owner serial walk over each network stage (disjoint compare-swap pairs per stage => identical to the parallel `sync_cube` form; the cpu anchor needs no barrier and no plane op).

### GPU leg (rocm-gated, cross-validated in 14-06)
- `plane_inclusive_sum`/`plane_exclusive_sum` block scan with `SharedMemory` + `sync_cube()` LDS staging (Pattern 2), and `plane_sum`/`plane_max`/`plane_min` + dot reductions (Pattern 1), as the `has_plane` f32 mirror. Verified compiling under `cargo check -p lgbm-compute --features rocm`; the C++-fixture parity gate (~1e-6, never GPU-vs-GPU) lands in 14-06.

## Open Q2 — f64 reduction-order policy (RESOLVED, per reduction)

| Reduction | Policy | Why |
|-----------|--------|-----|
| sum | BIT-EXACT vs serial f64 fold, **ascending matched order** | the cpu anchor's single-owner fold IS the ascending order |
| dot-product | BIT-EXACT vs `acc += a[i]*b[i]` ascending | same matched-order fold; per-product round then add |
| max | BIT-EXACT, order-INDEPENDENT | selection only, no rounding |
| min | BIT-EXACT, order-INDEPENDENT | selection only, no rounding |

f32 hip warp-tree reductions: ~1e-6 only (plane width changes the reduction tree, Pitfall 3) — deferred to the ROCm leg in 14-06.

## Argsort tie convention (recorded for 14-06 fixture cross-check)

Strict `>` comparison; ascending segment parity `outer_segment_index & 1 == 0` (ASCENDING); `aligned` = next power of two >= `num_items`, `depth` = log2(aligned)+1; keys sentinel-padded with `+inf` (ascending) / `-inf` (descending) so padding sorts to the tail; returned slices truncated to `keys.len()`. This is the exact `BitonicArgSort_1024` convention; 14-06 cross-checks the same convention against the committed C++ fixture (tie-rich input locks it here).

## Deferred (recorded so it is not lost)

- **Recursive >1024-block global prefix-sum** (arrays > ~1M elements, where the block-totals exceed the single tile and launch 2 must recurse): **OUT OF SCOPE this phase, OWNED by Phase 15** (on-device dataset, the first real consumer). The Phase-14 kernel guards `num_blocks <= 1024` (`MAX_GLOBAL_SCAN_BLOCKS`) rather than truncating.
- **Percentile / multi-block `…Global` argsort / `BitonicArgSortItems`**: anchor-pinned skeletons in 14-05 (D-02).
- **C++/HIP golden-fixture cross-validation** of every numeric primitive + the f32 ROCm ~1e-6 leg: 14-06.

## Task Commits

1. **Task 1 RED** — `e037e81` (test): failing prefix-sum self-tests
2. **Task 1 GREEN** — `f9a5fb3` (feat): block + multi-kernel global prefix-sum
3. **Task 2 RED** — `73c1188` (test): failing reduction self-tests
4. **Task 2 GREEN** — `d0d7b6c` (feat): sum/max/min + dot-product reductions
5. **Task 3 RED** — `2120ed4` (test): failing bitonic argsort self-tests
6. **Task 3 GREEN** — `228caef` (feat): index-only bitonic argsort + GPU plane variants

## Verification

- `cargo test -p lgbm-compute --test primitives_self` -> 13 passed (prefix_sum, reduce, argsort).
- `cargo build -p lgbm-compute` -> clean, no warnings.
- `cargo test -p lgbm-compute --lib` -> 52 passed, 0 failed (no regression; D-11 default-path unchanged for the compute crate).
- `cargo check -p lgbm-compute --features rocm` -> clean (the plane-intrinsic GPU variants compile under rocm).
- `cargo clippy -p lgbm-compute --features rocm` -> no primitives warnings.
- Plan checks: no `Atomic<i64>` and no `wrapping_*` in `#[cube]` (grep clean); 20 plane-intrinsic call sites present (key_links satisfied); the global scan uses exactly one reused `client.empty` scratch.

## Deviations from Plan

None - plan executed exactly as written.

The plan's per-task "within-plane / cross-plane / multi-block" coverage maps onto the cpu anchor as "within-block / multi-block" because cubecl-cpu has `plane_size == 1` (14-01) — the within-plane/cross-plane distinction is a GPU (`has_plane`) concern exercised by the rocm plane kernels and cross-validated in 14-06. This is the intended consequence of the 14-01 finding, not a deviation: the cpu anchor is the serial fold, the plane intrinsics are the rocm leg.

## Known Stubs

None. The percentile / multi-block-argsort / items-sort skeletons are explicitly 14-05's scope (D-02), not stubs in this plan's files. The rocm plane variants are complete kernels (not stubs); their C++-fixture parity gate is 14-06's scope by plan design.

## Issues Encountered

- cubecl indexing requires `usize` indices and `SharedMemory::new` a `usize` comptime size; resolved by converting builtins (`UNIT_POS`/`CUBE_DIM`/`CUBE_POS_X`) to `usize` up front (the histogram idiom) and doing all index math in `usize`. In-kernel f32 zero literals use the cube constructor `f32::new(0.0)`, and `SharedMemory` written via indexed assignment must be `let mut`.

## Next Phase Readiness

- **14-06 (fixture cross-validation, same phase):** unblocked — the numeric primitives + the rocm plane variants exist with the f64-order policy and argsort tie convention recorded; 14-06 builds the C++/HIP golden harness and asserts bit-exact (int/permutation/f64-anchor) + ~1e-6 (f32 hip).
- **Phase 15 (on-device growth):** the block + global prefix-sum and reductions are the reusable building blocks; Phase 15 owns the recursive >1024-block global scan when its first consumer needs it.
- No blockers.

## Self-Check: PASSED
- Files created/modified exist on disk: `crates/lgbm-compute/tests/primitives_self.rs`, `crates/lgbm-compute/src/kernels/primitives.rs` (both FOUND).
- Commits exist: `e037e81`, `f9a5fb3`, `73c1188`, `d0d7b6c`, `2120ed4`, `228caef` (all FOUND).
- `cargo test -p lgbm-compute --test primitives_self` -> 13 passed.

---
*Phase: 14-foundation-shared-device-primitives-device-structs-rng*
*Completed: 2026-06-29*
