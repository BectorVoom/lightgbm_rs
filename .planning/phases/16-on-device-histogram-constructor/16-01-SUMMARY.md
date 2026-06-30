---
phase: 16-on-device-histogram-constructor
plan: 01
subsystem: testing
tags: [cubecl, histogram, fixtures, cpu-anchor, parity, rocm, gsd-wave-0]

# Dependency graph
requires:
  - phase: 15-on-device-device-dataset-row-subset-gather
    provides: "CudaRowData / divide_cuda_feature_groups §13 partition layout + synthetic sparse/large-bin column shapes"
  - phase: 14-foundation-shared-device-primitives-device-structs-rng
    provides: "construct_histograms_f64_on cpu f64 fold anchor + subtract.rs subtract_histograms_cpu"
provides:
  - "Synthetic sparse fixture forcing CSR row_ptr_type {16,32,64} (oracle-harness)"
  - "Synthetic large-bin/global-spill fixture forcing num_large_bin_partition > 0"
  - "most_freq_bin != 0 fixture with f64-anchored golden repaired default-bin value (leaf_total − Σ ascending)"
  - "interleave_layout [2b]/[2b+1] assert helper (lgbm-compute rocm_cuda_mirror)"
  - "SubtractOrderingGuard build-fully-synced-before-subtract (8aed100) harness"
  - "Partition-aware cpu f64 anchor case entry points (sparse, large-bin spill) + assert_close envelope self-test"
affects: [16-02-histogram-arena, 16-03-build-kernel, 16-04-fix-subtract, on-device-histogram]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Wave-0 anchor-first: every numeric expectation pinned to the cubecl-cpu f64 fold (construct_histograms_f64_on, CubeDim::new_1d(1)); never GPU-vs-GPU (def-f8u-01)"
    - "Golden repaired default-bin value computed in plain Rust f64, recomputed independently in-test — never the fix kernel under test"
    - "cpu-anchor scaffold un-gated (runs everywhere = merge gate); GPU mirror tests gated under mod hip (--features rocm)"

key-files:
  created: []
  modified:
    - "crates/oracle-harness/tests/kernel_parity.rs"
    - "crates/lgbm-compute/tests/rocm_cuda_mirror.rs"

key-decisions:
  - "Synthetic in-code fixtures (not C++-captured golden files) for the partition-layout/Fix/ordering half of the D-05 anchor — no C++ toolchain needed to regenerate; the committed dense histogram.txt remains the C++-captured half"
  - "ODL-09/ODL-10 NOT marked complete: they are phase-spanning requirements satisfied only when the build/fix/subtract kernels land (16-03/16-04); this plan lands only the Wave-0 verification targets"
  - "Un-gated the previously fully rocm-gated rocm_cuda_mirror.rs so the new cpu-anchor harness runs everywhere; moved the 4 existing GPU mirror tests verbatim into a #[cfg(feature=rocm)] mod hip"

patterns-established:
  - "Ordering invariant modeled as a host-level publish guard: SubtractOrderingGuard.build_parent (sync) must precede subtract_smaller; premature subtract returns Err (8aed100)"
  - "Non-panicking Result core (close_within / check_interleave_layout) + panicking assert wrapper, so the envelope/layout checkers are unit-testable both ways without catch_unwind"

requirements-completed: []

# Metrics
duration: 35min
completed: 2026-07-01
status: complete
---

# Phase 16 Plan 01: On-Device Histogram Constructor — Wave 0 Fixtures & Anchor Harness Summary

**Synthetic sparse/large-bin/most_freq_bin≠0 build-fix fixtures + the build-before-subtract ordering guard and [2b]/[2b+1] interleave assert, all pinned to the cubecl-cpu f64 fold — closing the six VALIDATION.md Wave-0 gaps before any Phase-16 kernel is written.**

## Performance

- **Duration:** ~35 min
- **Completed:** 2026-07-01
- **Tasks:** 2
- **Files modified:** 2

## Accomplishments
- **Task 1 (oracle-harness):** synthetic sparse fixture forcing CSR `row_ptr_type` {16,32,64} via the §13 re-lay (asserts on the *resolved* `row_ptr_bit_type`, not just the nnz tag); large-bin fixture forcing `num_large_bin_partition > 0` (the `_GlobalMemory` spill); a purpose-built `most_freq_bin != 0` column whose golden repaired default-bin value (`leaf_total − Σ(other bins, ascending)`) is computed in plain Rust f64 and recomputed independently in-test — never the fix kernel. Build histogram anchored to `construct_histograms_f64_on`; full `2*num_bin` shape (no compaction, Pitfall 5).
- **Task 2 (lgbm-compute):** `interleave_layout` `[2b]/[2b+1]` assert helper (accepts correct, rejects swapped); `SubtractOrderingGuard` modeling build-fully-synced-before-subtract (ordered run reproduces `parent − smaller` bit-exact; mis-order detected — the 8aed100 guard); partition-aware cpu f64 anchor case entry points (sparse, large-bin spill) that 16-03/16-04 extend with the GPU kernel call + `assert_close`; `assert_close` envelope self-test.
- **Coverage shift:** `rocm_cuda_mirror` went from **0 cpu tests** (whole file was rocm-gated) to **5 cpu tests** that are the always-on merge gate; the 4 GPU mirror tests still run under `--features rocm` (`mod hip` compiles clean under the flag).

## Task Commits

1. **Task 1: Synthetic sparse + large-bin/global-spill + mfb!=0 fixtures (D-05)** — `efb355d` (test)
2. **Task 2: Ordering-invariant harness + [2b]/[2b+1] interleave assert helper (D-05)** — `13e6962` (test)

## Files Created/Modified
- `crates/oracle-harness/tests/kernel_parity.rs` — added `mod kernel16_fixtures` (sparse/large-bin/mfb builders) + 3 tests (`kernel16_sparse_fixture_forces_all_row_ptr_widths`, `kernel16_large_bin_fixture_forces_global_spill`, `fix_mfb_nonzero_repaired_default_bin_anchored`).
- `crates/lgbm-compute/tests/rocm_cuda_mirror.rs` — un-gated the cpu f64-anchor scaffold; added `close_within`/`check_interleave_layout`/`interleave_layout`, `SubtractOrderingGuard`, `cpu_anchor_columns`, and 5 cpu tests; moved the 4 GPU mirror tests into `#[cfg(feature="rocm")] mod hip`.

## Decisions Made
- **Synthetic in-code fixtures** for the partition/Fix/ordering anchors (vs C++-captured `.txt` goldens) — regenerable without a C++ toolchain, self-validating in plain f64. The committed `histogram.txt` dense corpora anchor is untouched.
- **Did NOT mark ODL-09/ODL-10 complete.** Both are phase-spanning requirements (on-device build + subtraction trick) that are only satisfied once the build/fix/subtract kernels land in 16-03/16-04. This plan delivers only the Wave-0 verification targets, so `requirements-completed: []`. Marking them now would be a false completion claim against the REQUIREMENTS traceability.
- **Un-gated `rocm_cuda_mirror.rs`** (was `#![cfg(feature = "rocm")]` over the whole file) so the cpu-anchor harness is the always-on merge gate; the GPU mirror tests moved verbatim into a gated `mod hip` (verified compiling under `--features rocm`).

## Deviations from Plan

None — plan executed exactly as written. The only minor adjustment was removing an unused `sparse_num_rows` helper to keep the test build warning-clean (no behavioral impact).

## Issues Encountered
None. Full-workspace `cargo test` stays green (additive test-only change); `mod hip` confirmed compiling under `--features rocm`.

## Self-Check: PASSED
- Files: FOUND `crates/oracle-harness/tests/kernel_parity.rs`, FOUND `crates/lgbm-compute/tests/rocm_cuda_mirror.rs`
- Commits: FOUND `efb355d`, FOUND `13e6962`
- `cargo test -p oracle-harness --test kernel_parity` → 10 passed (3 new); `cargo test -p lgbm-compute --test rocm_cuda_mirror` → 5 passed (was 0); `cargo test --workspace` → all green.

## Next Phase Readiness
- All six Wave-0 gaps closed and green on the cpu anchor: sparse {16,32,64}, large-bin/global-spill, mfb≠0 with anchored repaired value, build-before-subtract ordering, `[2b]/[2b+1]` interleave, and the extended `cpu_anchor`/`assert_close` partition-aware scaffold.
- 16-02 (HistArena slot pool) and 16-03/16-04 (build / fix-subtract kernels) now have a green anchor target on their first commit: the partition-aware case stubs need only the GPU kernel call + `assert_close(anchor, gpu)`; the `fix_mfb_nonzero` golden is ready for 16-04's FixHistogram parity.

---
*Phase: 16-on-device-histogram-constructor*
*Completed: 2026-07-01*
