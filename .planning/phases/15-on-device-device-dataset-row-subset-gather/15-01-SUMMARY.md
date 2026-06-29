---
phase: 15-on-device-device-dataset-row-subset-gather
plan: 01
subsystem: infra
tags: [cubecl, cuda-on-device, device-dataset, copy-subrow, bagging, scaffolding, parity-tests]

# Dependency graph
requires:
  - phase: 14-foundation-shared-device-primitives-device-structs-rng
    provides: ungated-kernel-module pattern (random/split), CUDARandom LCG, cpu f64 anchor (cpu_client), BinColumn narrow store
provides:
  - kernels/row_data.rs — CudaRowData + FeaturePartitionLayout + divide_cuda_feature_groups + §13 accessor stubs
  - kernels/column_data.rs — CudaColumnData + ColumnFeatureMeta numeric per-feature meta stubs
  - kernels/copy_subrow.rs — real D-07 copy_subrow_kernel<B: Int> skeleton + host launcher + bagging-draw wrapper stubs
  - tests/device_dataset_parity.rs — 4 ODL-03 parity tests + D-04 sparse synthesizer
  - tests/copy_subrow_parity.rs — 3 ODL-04 tests + inline host bagging reference + V5/T-15-IDX rejection
affects: [15-02-dense-relay-column-data, 15-03-sparse-relay, 15-04-copy-subrow-bagging, 16-histogram-constructor]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Wave-0 scaffold: register modules ungated + stub full public API with todo!() + author Nyquist parity tests that compile/link/run (fail at todo!) so each Wave-1 plan fills exactly one owned file with no mod.rs/test contention (Phase-14 pattern)"
    - "ABSOLUTE_POS is usize in cubecl 0.10 — widen u32 scalar guards (num_used as usize) and index arrays by usize (histogram resident_bins idiom)"
    - "Explicit width seam: pass row_ptr_bit_type to get_sparse_data_partitioned so the 3x3 (bit_type x row_ptr_type) matrix is exercisable without materializing a 2^32-nnz column"

key-files:
  created:
    - crates/lgbm-compute/src/kernels/row_data.rs
    - crates/lgbm-compute/src/kernels/column_data.rs
    - crates/lgbm-compute/src/kernels/copy_subrow.rs
    - crates/lgbm-compute/tests/device_dataset_parity.rs
    - crates/lgbm-compute/tests/copy_subrow_parity.rs
  modified:
    - crates/lgbm-compute/src/kernels/mod.rs

key-decisions:
  - "Registered the 3 new modules ungated (not #[cfg(feature=gpu)]) so the cpu f64 anchor runs them (D-08), mirroring random/split"
  - "copy_subrow_kernel body is the real D-07 skeleton (trivial, width-generic over B: Int); all other bodies are todo!()"
  - "Inline host bagging reference reproduces sample_strategy::bagging via the shared lgbm_core::random::Random (no boosting-crate dev-dep — crate-cycle prohibition)"
  - "Sparse 3x3 matrix exercised via an explicit row_ptr_bit_type param + an nnz-carried-as-number synthesizer (2^32 cell not materialized)"

patterns-established:
  - "Pattern 1: per-feature numeric meta (ColumnFeatureMeta) carries §3 fields; categorical bitset meta deferred to Phase 22 (TODO marker)"
  - "Pattern 2: partition-local re-lay invariant (stored bin = global bin - partition_hist_offsets[partition]) asserted via read_partition_local_bin (Pitfall 4)"

requirements-completed: [ODL-03, ODL-04]

# Metrics
duration: 8min
completed: 2026-06-30
status: complete
---

# Phase 15 Plan 01: Wave-0 Device-Dataset Scaffolding Summary

**Registered three ungated lgbm-compute kernel modules (row_data / column_data / copy_subrow) with stubbed §3/§13 public APIs plus a real D-07 gather kernel, and authored two anchor-pinned Nyquist parity test files (ODL-03 dataset re-lay + ODL-04 row-subset gather/bagging) that compile, link, and run against the cpu f64 anchor.**

## Performance

- **Duration:** 8 min
- **Started:** 2026-06-29T21:26:15Z
- **Completed:** 2026-06-29T21:34:39Z
- **Tasks:** 3
- **Files modified:** 6 (5 created, 1 modified)

## Accomplishments
- Three new kernel modules registered ungated in `kernels/mod.rs` (mirrors `random`/`split`, NOT `#[cfg(feature=gpu)]`) so the default cpu f64 anchor runs them (D-08); `cargo build -p lgbm-compute` green.
- `copy_subrow.rs` ships the real D-07 `copy_subrow_kernel<B: Int>` skeleton (one unit per selected row, native-width gather) — the only non-`todo!()` body, validated to compile through the cube macro.
- Two parity test files compile + link (`--no-run` exit 0) and RUN (panic at `todo!()`, not `#[ignore]`-skipped): `device_dataset_parity.rs` (4 ODL-03 tests + D-04 sparse synthesizer forcing all three `row_ptr_type` widths + the large-bin spill) and `copy_subrow_parity.rs` (3 ODL-04 tests + inline host bagging reference + V5/T-15-IDX rejection).
- Zero crate cycle: the bagging reference is reproduced inline via the shared `lgbm_core::random::Random`; no boosting-crate import (negative grep guard passes).

## Task Commits

Each task was committed atomically:

1. **Task 1: Register modules + stub public API surface** - `ad39478` (feat)
2. **Task 2: Author device_dataset_parity.rs (ODL-03) + D-04 synthesizer** - `23fe1f3` (test)
3. **Task 3: Author copy_subrow_parity.rs (ODL-04) + inline bagging ref** - `08117c2` (test)

## Files Created/Modified
- `crates/lgbm-compute/src/kernels/mod.rs` - registered `row_data`/`column_data`/`copy_subrow` ungated with a Phase-15 comment block
- `crates/lgbm-compute/src/kernels/row_data.rs` - `CudaRowData<R>` + `FeaturePartitionLayout` + `divide_cuda_feature_groups` + §13 accessors + dense/sparse re-lay + read-back stubs
- `crates/lgbm-compute/src/kernels/column_data.rs` - `CudaColumnData<R>` + `ColumnFeatureMeta` (numeric per-feature meta) + constructor + per-column read-back stubs
- `crates/lgbm-compute/src/kernels/copy_subrow.rs` - real `copy_subrow_kernel<B: Int>` + `copy_subrow_on` + `bagging_draw_on` stubs + local `BAGGING_RAND_BLOCK`/`COPY_SUBROW_BLOCK_SIZE`
- `crates/lgbm-compute/tests/device_dataset_parity.rs` - ODL-03 parity tests + `mod synth` D-04 sparse synthesizer
- `crates/lgbm-compute/tests/copy_subrow_parity.rs` - ODL-04 gather/bagging tests + inline `host_bag_data_indices`

## Decisions Made
- Followed the plan as specified for module registration, kernel skeleton, and test authorship.
- Resolved two cubecl-0.10 cube-macro specifics during the kernel skeleton: `ABSOLUTE_POS` is `usize` (widen the `num_used: u32` guard via `as usize`), and arrays index by `usize` (the histogram `resident_bins[... as usize]` idiom).

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Added read-back accessors + explicit row_ptr width seam to row_data.rs (a Task-1 file) during Task 2**
- **Found during:** Task 2 (authoring `device_dataset_parity.rs`)
- **Issue:** The four ODL-03 tests must NAME a device read-back to assert host-vs-device parity, and must drive the sparse 3×3 matrix without materializing a 2^32-nnz column. The Task-1 stub surface had no read-back method and no `row_ptr_type` selector, so the test file could not compile.
- **Fix:** Added `CudaRowData::read_bin` (logical global bin) and `read_partition_local_bin` (Pitfall-4 partition-local invariant) read-back stubs, and added an explicit `row_ptr_bit_type: u32` parameter to `get_sparse_data_partitioned` so the 3×3 cell is exercisable from the synthesizer's nnz-derived width. All remain `todo!()` bodies (Wave-1 scope).
- **Files modified:** `crates/lgbm-compute/src/kernels/row_data.rs`
- **Verification:** `cargo test -p lgbm-compute --test device_dataset_parity --no-run` exits 0; the test compiles and links.
- **Committed in:** `23fe1f3` (Task 2 commit)

---

**Total deviations:** 1 auto-fixed (1 blocking)
**Impact on plan:** The read-back/width-seam additions are the minimal API the ODL-03 tests need to be nameable; param lists were explicitly "Plan 15-02/refinement scope — keep them minimal but nameable from the tests". No scope creep — all bodies stay `todo!()`.

## Issues Encountered
- The plan's Task-3 verify grep guard (`! grep -q "lgbm.boosting\|lgbm_boosting"`) initially tripped because the test's doc comments mentioned the boosting crate by name (the `.` matches `-`). Reworded the comments to "the boosting crate" / "sample_strategy" so the negative guard passes while preserving the no-import invariant.

## Known Stubs (intentional — Wave-0 scaffold)
This plan is scaffolding by design: every body except `copy_subrow_kernel` is a `todo!("15-NN")` stub, and both parity test files RUN and fail at those stubs (not `#[ignore]`-skipped). This is the expected Wave-0 state — the stubs are resolved by the Wave-1 plans:
- `divide_cuda_feature_groups`, `CudaRowData::get_dense_data_partitioned` + read-backs, `CudaColumnData::{new,read_column}` → Plan 15-02
- `CudaRowData::get_sparse_data_partitioned` + sparse read-backs → Plan 15-03
- `copy_subrow_on`, `bagging_draw_on` → Plan 15-04
No stub flows to user-facing output (the whole subsystem is behind the OFF-by-default `LGBM_CUDA_ON_DEVICE` seam; `on_device_growth_supported()` stays false this phase).

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Skeleton compiles; the two Nyquist parity test files exist, compile, and run. The three Wave-1 plans (15-02 dense + column data, 15-03 sparse, 15-04 copy_subrow/bagging) can each fill exactly one owned source file with no shared-file contention.
- No blockers.

## Self-Check: PASSED
All 5 created files exist on disk; all 3 task commits (`ad39478`, `23fe1f3`, `08117c2`) present in git history.

---
*Phase: 15-on-device-device-dataset-row-subset-gather*
*Completed: 2026-06-30*
