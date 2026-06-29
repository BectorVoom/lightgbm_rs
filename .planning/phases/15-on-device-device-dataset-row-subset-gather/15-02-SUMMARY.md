---
phase: 15-on-device-device-dataset-row-subset-gather
plan: 02
subsystem: compute
tags: [cubecl, cuda-on-device, device-dataset, row-data, feature-partition, csr, parity-tests]

# Dependency graph
requires:
  - phase: 15-on-device-device-dataset-row-subset-gather
    plan: 01
    provides: row_data.rs stub API (CudaRowData + FeaturePartitionLayout + divide_cuda_feature_groups + §13 accessors) + device_dataset_parity.rs ODL-03 tests
provides:
  - "divide_cuda_feature_groups — §13 host layout math (budget = shared_hist_size/2, in-order packing, large-bin spill)"
  - "FeaturePartitionLayout — populated five §13 accessor tables (partition-local column_hist_offsets, global partition_hist_offsets, column-index offsets, counts)"
  - "CudaRowData::get_dense_data_partitioned — row-major partition-local dense store, upload-once"
  - "CudaRowData::get_sparse_data_partitioned — per-partition CSR re-lay (partition-local bins) over the 3×3 {bit_type}×{row_ptr_type} dispatch"
  - "CudaRowData read-back accessors (read_bin / read_partition_local_bin) + bit_type/row_ptr_bit_type + data_handle()/row_ptr_handle() Phase-16 seam"
affects: [15-05-phase-integration, 16-histogram-constructor]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "§13 row store: dense buffer stores RAW per-column bins (kernel applies column_hist_offsets at accumulation), sparse buffer stores PARTITION-LOCAL bins (column_hist_offsets folded in = global_bin − partition_hist_start); read_bin reconstructs the logical global bin uniformly by dispatching on is_sparse"
    - "InitSparseData<BIN_TYPE,PTR_TYPE> 3×3 ported as a 9-arm nested match over derived bit_type ∈ {8,16,32} and supplied row_ptr_bit_type ∈ {16,32,64}, each arm a generic init_sparse_data::<B,P> — unsupported width returns ComputeError before any build/upload (never a silent widen)"
    - "Pure store/relay needs no kernel: safe create_from_slice upload-once + read_one_unchecked read-back + per-width decode, so no unsafe is introduced (CMP-01 unsafe stays in kernel launchers)"

key-files:
  created:
    - .planning/phases/15-on-device-device-dataset-row-subset-gather/15-02-SUMMARY.md
  modified:
    - crates/lgbm-compute/src/kernels/row_data.rs
    - crates/lgbm-compute/tests/device_dataset_parity.rs

key-decisions:
  - "divide_cuda_feature_groups ports the design-doc/C++ budget literally (max_num_bin_per_partition = shared_hist_size/2 = 3072 for the DP value 6144); the feature_partition_layout test's hand-computed expectations were arithmetically wrong (treated cumulative 3000 as overflowing 3072) and were corrected to match the literal port — Rule 1"
  - "Both dense and sparse use ONE row-major partition-local value buffer (partition p occupies [off[p]·num_rows, off[p+1]·num_rows), cell at row·ncol_p+local_col); a NEW buffer distinct from the feature-major resident_bins (Pitfall 3)"
  - "row_ptr_bit_type is honored from the explicit param (15-01 width seam) and validated ∈ {16,32,64} before build; init_sparse_data also asserts every CSR offset fits the chosen PTR width before upload (T-15-PTR)"

requirements-completed: [ODL-03]

# Metrics
duration: 7min
completed: 2026-06-30
status: complete
---

# Phase 15 Plan 02: §13 CUDARowData Row + Feature-Partition Store Summary

**Filled in `row_data.rs` with the §13 `CUDARowData` row store: the host `DivideCUDAFeatureGroups` partition layout math, a row-major partition-local dense re-lay, a per-partition CSR sparse re-lay over the 3×3 `{bit_type}×{row_ptr_type}` dispatch, and width-dispatched read-back accessors — all uploaded once and anchored bit-exact to host `BinColumn` values on the cpu f64 anchor.**

## Performance

- **Duration:** ~7 min
- **Started:** 2026-06-29T21:48:50Z
- **Completed:** 2026-06-29T21:56:01Z
- **Tasks:** 3
- **Files modified:** 2 (1 source, 1 test)

## Accomplishments
- `divide_cuda_feature_groups` ports §13 exactly: budget `max_num_bin_per_partition = shared_hist_size/2`, in-order column packing, large-bin columns spill to their own partition (`num_large_bin_partition += 1`); builds all five accessor tables (partition-local `column_hist_offsets`, global `partition_hist_offsets`, `feature_partition_column_index_offsets`, `max_num_column_per_partition`, `num_feature_partitions`).
- `get_dense_data_partitioned` assembles a NEW row-major partition-local bin buffer (`data[row*ncol+col]`, distinct from the feature-major `resident_bins` — Pitfall 3), uploaded once at the widest column's `bit_type`.
- `get_sparse_data_partitioned` builds per-partition CSR with bins made partition-local (`column_hist_offsets[col]` folded in = `global_bin − partition_hist_start`, Pitfall 4), dispatched through the 9-arm `InitSparseData<BIN,PTR>` 3×3; unsupported `row_ptr` widths return `ComputeError` before any build/upload (Pitfall 2).
- Read-back accessors (`read_bin` global, `read_partition_local_bin` partition-local) decode the cached device handle per width; the sparse 2-partition invariant `local == global − partition_hist_offsets[p]` holds across all three forced `row_ptr_type` widths.
- All three ODL-03 row-store tests green on the cpu f64 anchor; `cargo clippy -p lgbm-compute --tests` clean on the touched files.

## Task Commits

Each task was committed atomically:

1. **Task 1: DivideCUDAFeatureGroups + offset tables + accessors** — `91115d0` (feat)
2. **Task 2: GetDenseDataPartitioned row-major partition-local store + upload-once** — `d9cd54a` (feat)
3. **Task 3: GetSparseDataPartitioned 3×3 CSR re-lay + partition-local bins** — `acfbd3d` (feat)

## Files Created/Modified
- `crates/lgbm-compute/src/kernels/row_data.rs` — implemented `divide_cuda_feature_groups`, `get_dense_data_partitioned`, `get_sparse_data_partitioned`, `read_bin`, `read_partition_local_bin`, `bit_type`/`row_ptr_bit_type`/`data_handle`/`row_ptr_handle` accessors + private helpers (`column_bit_width`, `validate_columns`, `upload_store_buffer`, `decode_store_cell`, `init_sparse_data`, `StoreInt`/`PtrInt` traits).
- `crates/lgbm-compute/tests/device_dataset_parity.rs` — corrected the `feature_partition_layout` test's hand-computed expectations (see Deviations).

## Decisions Made
- Ported the §13 budget literally (`shared_hist_size/2`) as the must-have mandates, and reconciled the wrong test expectations to it rather than bending the algorithm.
- Unified the dense/sparse read path: both store one row-major partition-local buffer; dense stores RAW bins (kernel applies offsets later) while sparse stores partition-local bins; `read_bin` reconstructs the global bin by dispatching on `is_sparse`. Both readbacks satisfy `local == global − partition_hist_offsets[p]`.
- Implemented the re-lay as a pure safe upload/readback round-trip (no kernel launch), so no `unsafe` is introduced — CMP-01's confined-unsafe rule is met vacuously.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Corrected the `feature_partition_layout` test's hand-computed expectations**
- **Found during:** Task 1
- **Issue:** The 15-01-authored `feature_partition_layout` test asserted, for `num_bin_per_column = [1000,1000,1000,4000]` at `shared_hist_size=6144` (budget `3072`), that column 2 opens a new partition — its comment literally reads "3000 <= 3072 would overflow at the 3rd". Cumulative `3000 ≤ 3072` fits, so under the plan's must-have budget formula (`max_num_bin_per_partition = shared_hist_size/2 = 3072`, confirmed in `docs/cuda-kernel-design.md §13` and `LightGBM-release-4.6.0.99/src/io/cuda/cuda_row_data.cpp`), columns 0–2 pack into one partition. The expected values were internally inconsistent with the mandated formula and would force a wrong (budget < 3000) algorithm.
- **Fix:** Implemented the literal `/2` budget and corrected the four expected values to the faithful result: `num_feature_partitions = 2`, `feature_partition_column_index_offsets = [0,3,4]`, `column_hist_offsets = [0,1000,2000,0]`, `partition_hist_offsets = [0,3000,7000]` (`num_large_bin_partition = 1` unchanged). Per RESEARCH §17/A1 the partition grouping has no float-parity impact, so the corrected (cleaner) grouping is valid; the load-bearing requirement is the literal budget port, which is honored.
- **Files modified:** `crates/lgbm-compute/tests/device_dataset_parity.rs`
- **Verification:** `cargo test -p lgbm-compute --test device_dataset_parity feature_partition_layout` passes.
- **Committed in:** `91115d0` (Task 1 commit)

---

**Total deviations:** 1 auto-fixed (1 bug). The test file is outside this plan's declared `files_modified` (`row_data.rs` only), but the correction was a prerequisite to a faithful implementation; it touches only the wrong assertions of the one test this plan owns, not 15-03's `column_store_parity`.

## Threat Mitigations Applied
- **T-15-PART** (length-product overflow): `checked_mul` on `num_columns * num_rows` (dense + sparse) and `num_partitions * (num_rows+1)` (sparse row_ptr) before any `create_from_slice`.
- **T-15-PTR** (row_ptr width too small): `row_ptr_bit_type` validated ∈ {16,32,64} before build; `init_sparse_data` asserts every CSR offset ≤ `PtrInt::MAX` before upload (no truncation → no OOB). Unsupported width → `ComputeError`, never a silent widen (Pitfall 2).
- **T-15-CMP** (read-back `from_raw_parts`): not applicable — the re-lay is a pure safe `create_from_slice`/`read_one_unchecked` round-trip with no kernel launch, so no `from_raw_parts` / `unsafe` is introduced.

## Known Stubs
None introduced by this plan. (`column_store_parity` in the shared test file still fails at the `column_data.rs` `todo!()` — that is plan 15-03's owned scope, untouched here.)

## Issues Encountered
- `gsd-tools query state.record-metric` / `state.add-decision` require named flags (`--phase/--plan/--duration/--tasks/--files`, `--summary`), not positional args; re-ran with the correct form.

## Next Phase Readiness
- The §13 row store is the direct Phase-16 histogram input (`blockIdx.x` = partition, `threadIdx.x` = column); `data_handle()` + `row_ptr_handle()` + `row_ptr_bit_type()` expose the resident buffers the histogram launcher will consume.
- Wave-1 siblings 15-03 (column_data) and 15-04 (copy_subrow) own distinct source files; this plan only added to `row_data.rs` (+ one shared-test fix), so no source contention.

## Self-Check: PASSED
`crates/lgbm-compute/src/kernels/row_data.rs` exists (710 lines, contains `divide_cuda_feature_groups` + `get_dense_data_partitioned` + `get_sparse_data_partitioned`). All three task commits (`91115d0`, `d9cd54a`, `acfbd3d`) present in git history. All three ODL-03 row-store tests pass; clippy clean on touched files.

---
*Phase: 15-on-device-device-dataset-row-subset-gather*
*Completed: 2026-06-30*
