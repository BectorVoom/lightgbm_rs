---
phase: 15-on-device-device-dataset-row-subset-gather
reviewed: 2026-06-30T00:00:00Z
depth: standard
files_reviewed: 6
files_reviewed_list:
  - crates/lgbm-compute/src/kernels/mod.rs
  - crates/lgbm-compute/src/kernels/row_data.rs
  - crates/lgbm-compute/src/kernels/column_data.rs
  - crates/lgbm-compute/src/kernels/copy_subrow.rs
  - crates/lgbm-compute/tests/device_dataset_parity.rs
  - crates/lgbm-compute/tests/copy_subrow_parity.rs
findings:
  critical: 2
  warning: 4
  info: 3
  total: 9
status: fixes_applied
fixes_applied:
  fixed_at: 2026-06-30
  resolved:
    - id: CR-01
      commit: 874fd07
    - id: CR-02
      commit: 8c62207
    - id: WR-01
      commit: 07d0427
    - id: WR-02
      commit: 07d0427
  remaining:
    - WR-03  # layout/columns consistency guard — not in fix scope
    - WR-04  # tautological bagging-count assertion — not in fix scope
    - IN-01  # duplicated width-dispatch helpers
    - IN-02  # row_ptr widths not exercised near capacity
    - IN-03  # per-cell whole-buffer read-back
---

# Phase 15: Code Review Report

**Reviewed:** 2026-06-30
**Depth:** standard
**Files Reviewed:** 6
**Status:** issues_found

## Summary

Phase 15 adds the resident on-device columnar/row binned dataset (`CudaColumnData` §3,
`CudaRowData` §13), the `DivideCUDAFeatureGroups` feature-partition layout, the sparse
3×3 (`bit_type` × `row_ptr_type`) CSR dispatch, the `CopySubrow` row-subset gather kernel,
and the on-device bagging draw — all ungated so the cpu f64 anchor exercises them.

The partition-packing math (`divide_cuda_feature_groups`) is correct and well unit-tested.
The bagging draw reproduces the host `[in-bag asc] ++ [OOB desc]` layout faithfully at
iteration 0 (the f32→f64 promotion before the `< fraction` compare is correct, Pitfall 6).
The overflow guards (`checked_mul`) and the `row_ptr` capacity assert (`P::MAX`) are present.

However, two correctness/safety defects are latent in the new device store and gather, and
the parity tests do **not** actually exercise the two most error-prone code paths they
claim to cover — meaning the BLOCKERs are masked by weak test corpora. Because Phase 16
consumes the §13 row store directly, the sparse-width truncation will corrupt histograms
when wired.

## Critical Issues

### CR-01: Sparse re-lay silently truncates partition-local bins when `bit_type` is narrower than the partition budget

**Status:** RESOLVED (commit `874fd07`) — `bit_type` is now selected via `bit_type_for_value(max(vals))` over the assembled partition-local values, so the store width always holds the partition-local bin range. Regression-guarded by `sparse_partition_local_nonzero_offset_fold` (WR-01).

**File:** `crates/lgbm-compute/src/kernels/row_data.rs:359, 397, 412-421, 600-614, 666`
**Issue:**
For the sparse store, `bit_type` is selected as the **maximum raw per-column width**:

```rust
let bit_type = columns.iter().map(column_bit_width).max().unwrap_or(8); // line 359
```

But the sparse store does NOT store raw bins — it stores **partition-local** bins
(`raw + column_hist_offsets[c]`, line 397), whose magnitude can range up to the partition
budget (`shared_hist_size / 2 = 3072` for the default), independent of any single column's
bin count. Those partition-local values are then narrowed to `bit_type` via
`StoreInt::from_u32_lossy`, which is a **truncating** `v as u8` / `v as u16` (lines 600-614).

Concrete failure: a store of only `u8` columns selects `bit_type = 8`. Up to 12 such columns
(`256 × 12 = 3072`) pack into one partition; the 12th column gets `column_hist_offsets = 2816`.
Storing its bin `2816 + raw` through `init_sparse_data::<R, u8, _>` truncates
`2816 as u8 → 0`, corrupting the partition-local bin (and hence the Phase-16 histogram). The
`StoreInt` doc comment claiming the value is "byte-faithful when narrowed" holds for raw bins
(`< num_bin ≤ width capacity`) but is **false** for sparse partition-local bins.

The dense path is unaffected (it stores raw bins, line 305, which always fit their own width;
`read_bin` adds the offset back at read time).

**Fix:** Size the sparse store width by the partition-local bin range, not the raw column width
— i.e. derive `bit_type` from `max_num_bin_per_partition` (the partition budget), matching the
C++ `CUDARowData` row-data width selection. Confirm the exact rule against
`include/LightGBM/cuda/cuda_row_data.hpp`. Minimally, replace the silent `v as u8` narrowing
with a checked conversion that returns `ComputeError` on overflow:

```rust
fn from_u32_checked(v: u32) -> Result<Self, ComputeError> {
    u8::try_from(v).map_err(|_| ComputeError::Runtime {
        detail: format!("partition-local bin {v} does not fit chosen bit_type=8"),
    })
}
```

### CR-02: `copy_subrow_on` validates indices against `num_data` but launches against the column buffer length — unchecked OOB device read

**Status:** RESOLVED (commit `8c62207`) — `copy_subrow_on` now returns `ComputeError::LengthMismatch` when `column.len() != num_data`, before any launch, so the per-element `< num_data` check and the kernel's `num_data == n_in` SAFETY invariant are both enforced.

**File:** `crates/lgbm-compute/src/kernels/copy_subrow.rs:81-102, 130-149`
**Issue:**
The V5/T-15-IDX boundary validation checks every `used_indices[i] ∈ [0, num_data)` (lines
89-102), but the kernel reads from a device buffer uploaded from `column` whose length is
`n_in = column.len()` (line 130). Nothing enforces `num_data == column.len()`. The
`num_data` and `column` arguments are independent public parameters.

If a caller passes `num_data > column.len()`, an index in `[column.len(), num_data)` passes
validation yet produces an out-of-bounds device read at `in_col[src]` — exactly the OOB the
control is meant to prevent. The `SAFETY` comment (lines 135-138) explicitly asserts
"and `num_data == n_in`", but that invariant is **never checked** in code; it is silently
assumed. The bound used for the security check is the wrong quantity (the logical row count,
not the uploaded buffer length).

Tests pass only because they always supply `num_data == column.len()`.

**Fix:** Validate the source-buffer relationship at the boundary, before the loop:

```rust
if column.len() != num_data {
    return Err(ComputeError::LengthMismatch {
        expected: num_data,
        actual: column.len(),
    });
}
```

(or validate each index against `column.len()` directly, which is the memory-safety-relevant
bound). Then the `SAFETY` comment's invariant becomes enforced rather than assumed.

## Warnings

### WR-01: Sparse partition-local invariant (Pitfall 4) is never exercised with a non-zero offset

**Status:** RESOLVED (commit `07d0427`) — added `sparse_partition_local_nonzero_offset_fold`: three u8 columns pack into one partition with non-zero offsets (partition-local bins up to 599), asserting `read_partition_local_bin == raw + column_hist_offset` across all three `row_ptr_type` widths plus a large-bin spill corpus. Verified to FAIL against pre-CR-01-fix code (143 vs 399) and pass after.

**File:** `crates/lgbm-compute/tests/device_dataset_parity.rs:160-225`
**Issue:**
`sparse_relay_3x3_and_partition_local` claims to validate the `partition_hist_start`
subtraction (Pitfall 4, "the single most error-prone line"). But the synthesized corpus
(`synth_columns`) has `num_bin = [200, 4000, 100000, 4096]`; three of four columns exceed the
3072 budget, so EVERY column becomes its own single-column partition. With one column per
partition, all `column_hist_offsets` are `0`, so the stored partition-local bin equals the raw
bin and the assertion `local == global - part_off` holds trivially — it never exercises the
`raw + column_hist_offsets[c]` fold-in (row_data.rs:397) with a non-zero offset. This is the
exact path that CR-01 corrupts; the test corpus masks the bug.

**Fix:** Add a corpus where ≥2 narrow columns pack into ONE partition (e.g. several
`num_bin ≤ 256` columns whose cumulative bins stay under 3072), then assert each stored
partition-local bin equals `raw + column_hist_offsets[c]` for non-zero offsets. This corpus
also reproduces CR-01.

### WR-02: Dense re-lay offset round-trip is untested (degenerate all-zero layout)

**Status:** RESOLVED (commit `07d0427`) — `dense_bin_parity_all_widths` now uses real per-column bin counts `[200, 2000, 100000]` so the U8+U16 columns share a partition with a non-zero offset and the U32 column spills; it asserts `read_bin` returns the true global bin (`raw + global_feature_offset`).

**File:** `crates/lgbm-compute/tests/device_dataset_parity.rs:103-118`
**Issue:**
`dense_bin_parity_all_widths` builds the layout from `num_bin_per_column =
columns.iter().map(|_| 0usize)` — all zeros. The real columns have `num_bin = 200/4000/100000`.
With a zero layout, `column_hist_offsets` and `partition_hist_offsets` are all `0`, so
`read_bin`'s offset-add (`stored + column_hist_offsets[column] + part_off`, row_data.rs:475)
is never validated with non-zero offsets — the test only proves raw passthrough. A transposed
or mis-added offset would not be caught.

**Fix:** Pass the real per-column bin counts into `divide_cuda_feature_groups` and assert
`read_bin` returns the true global bin (`raw + global_feature_offset`) for a multi-column
partition.

### WR-03: No consistency check between the passed-in `layout` and `columns` → internal panic instead of typed error

**File:** `crates/lgbm-compute/src/kernels/row_data.rs:300-308, 388-403`
**Issue:**
`get_dense_data_partitioned` / `get_sparse_data_partitioned` accept a `layout` argument and
index `columns[lo..hi]` and `column_hist_offsets[c]` with offsets taken from that layout. If
the caller passes a layout computed for a different column set (e.g. `num_bin_per_column.len()
!= columns.len()`, or `feature_partition_column_index_offsets.last() != num_columns`), these
slice/index operations panic with an out-of-bounds rather than returning the typed
`ComputeError` the rest of the module is careful to use (Pitfall 2 discipline). Public
constructors should fail closed with a structured error.

**Fix:** At entry, validate
`layout.feature_partition_column_index_offsets.last() == Some(&num_columns)` and
`layout.column_hist_offsets.len() == num_columns`, returning `ComputeError::LengthMismatch`
otherwise.

### WR-04: `bagging_draw_matches_host` contains a tautological no-op assertion claiming to verify the in-bag count

**File:** `crates/lgbm-compute/tests/copy_subrow_parity.rs:119-129`
**Issue:**
The block computing `device_cnt` uses `take_while(|&&v| { let _ = v; true })` — a predicate
that is always `true`, so it counts the whole vector unconditionally — then discards the
result (`let _ = device_cnt;`) and finishes with `assert_eq!(host_cnt, host_cnt, ...)`, which
compares a value to itself and can never fail. This asserts nothing about the realized in-bag
split point, despite the comment claiming it proves the count. It is dead/misleading test
code that inflates apparent coverage.

**Fix:** Either remove the block, or actually verify the split point, e.g. count the strictly
ascending in-bag prefix of `device` and `assert_eq!(in_bag_prefix_len as i32, host_cnt)`.

## Info

### IN-01: Duplicated width-dispatch helpers across the two store modules

**File:** `crates/lgbm-compute/src/kernels/row_data.rs:564-570, 679-710` and
`crates/lgbm-compute/src/kernels/column_data.rs:170-216`
**Issue:** `column_bit_width`, `decode_store_cell`, and `upload_*_buffer` are near-identical
copies in `row_data.rs` and `column_data.rs`. Divergence risk if one is fixed (e.g. for CR-01)
and the other is not.
**Fix:** Hoist the shared width-dispatch/decode helpers into a small private module
(e.g. `kernels/store_width.rs`) and reuse from both stores.

### IN-02: Sparse `row_ptr_type` widths are dispatch-selected but never exercised near their capacity

**File:** `crates/lgbm-compute/tests/device_dataset_parity.rs:70-81, 170-214`
**Issue:** The synthesizer tags columns with `nnz` to force the `{16,32,64}` dispatch arm, but
the actual CSR `row_ptr` values built in the re-lay are `r * ncol_p` (max `8` for the 8-row
corpus). So the `P::MAX` capacity guard (row_data.rs:655-664) and any near-`2^16`/`2^32`
offset behavior are never exercised — only the monomorph arm is. This is acknowledged by the
D-04 design ("carried as a number; not materialized"), but the test's `row_ptr` magnitudes give
no confidence in the offset-width sizing logic itself.
**Fix (optional):** Add a small synthetic `row_ptr` vector whose max offset crosses `u16::MAX`
and assert that the too-narrow width is rejected (`init_sparse_data::<_, _, u16>` errors), and
the next width succeeds.

### IN-03: Per-cell read-back reads the entire device buffer

**File:** `crates/lgbm-compute/src/kernels/row_data.rs:533-547`,
`crates/lgbm-compute/src/kernels/column_data.rs:159`
**Issue:** `read_stored_cell` calls `client.read_one_unchecked(self.data_handle.clone())`
(whole buffer) for every `(row, column)` decode; `read_bin`/`read_partition_local_bin` call it
per cell. This is correctness-neutral and confined to parity read-back (not a hot path), but it
is a footgun if reused outside tests. A doc note or a batched read-back would prevent misuse.
**Fix:** Document that these accessors are verification-only, or read the buffer once and decode
many cells.

---

_Reviewed: 2026-06-30_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
