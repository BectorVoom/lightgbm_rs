---
phase: 15-on-device-device-dataset-row-subset-gather
plan: 03
subsystem: compute
tags: [cubecl, cuda-on-device, device-dataset, column-store, parity-tests, ODL-03]

# Dependency graph
requires:
  - phase: 15-on-device-device-dataset-row-subset-gather
    plan: 01
    provides: CudaColumnData + ColumnFeatureMeta stub surface; column_store_parity Nyquist test
provides:
  - kernels/column_data.rs — §3 CudaColumnData filled (per-column native-width upload + numeric ColumnFeatureMeta + read-back); column_store_parity green
affects: [18-data-partition-tree-mutation-prediction]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Column-major device store: one device buffer PER feature column at its OWN native width (u8/u16/u32 per BinColumn variant) — NOT widened to the matrix's widest column (contrast the §13 row store's single uniform-width buffer); the two layouts coexist (Pitfall 3)"
    - "Pure safe create_from_slice (upload) + read_one_unchecked (read-back) round-trip — no kernel launch ⇒ no unsafe in the store/relay path (CMP-01: unsafe stays confined to kernel launchers)"

key-files:
  created: []
  modified:
    - crates/lgbm-compute/src/kernels/column_data.rs

key-decisions:
  - "Reused the proven 15-02 row_data.rs upload/decode shape (column_bit_width / upload_*_buffer / decode_store_cell) as module-local helpers — kept the change inside the single owned file (no cross-module coupling, no shared-file contention)"
  - "Validated feature_meta.len() == columns.len() at the boundary (ComputeError::LengthMismatch) since the per-feature numeric meta is parallel to the columns (Rule 2 — boundary correctness)"

patterns-established:
  - "Per-column native-width column store mirrors C++ in_cuda_data_by_column; categorical bitset meta carried only as a documented TODO(Phase 22) marker (Open Question 1 — numeric meta only this phase)"

requirements-completed: [ODL-03]

# Metrics
duration: 5min
completed: 2026-06-30
status: complete
---

# Phase 15 Plan 03: §3 CUDAColumnData column store + numeric meta Summary

**Filled the §3 `CudaColumnData` column-major device store (ODL-03): one native-width device buffer per `BinColumn` column uploaded once (D-09), the numeric per-feature `ColumnFeatureMeta`, and a per-column read-back — `column_store_parity` now reproduces the host `BinColumn::to_u32_vec()` and numeric meta bit-for-bit across all three widths on the cpu f64 anchor.**

## Performance

- **Duration:** ~5 min
- **Started:** 2026-06-29T21:59:34Z
- **Completed:** 2026-06-30
- **Tasks:** 1 (tdd)
- **Files modified:** 1

## Accomplishments
- `CudaColumnData::new` uploads each feature column ONCE at its OWN native width (`u8`/`u16`/`u32` per the `BinColumn` variant) via `create_from_slice(<w>::as_bytes(..))`, storing per-column handles + bit-types + lengths column-major (D-09 alloc-once — no per-call/per-tree re-upload).
- `read_column` reads one column back from the cached device `Handle` (`read_one_unchecked`) and decodes per-width to `u32` for the ODL-03 parity assert.
- Numeric `ColumnFeatureMeta` (`bit_type`, `feature_min/max_bin`, `offset`, `most_freq_bin`, `default_bin`, missing flags, `feature_to_column`) carried 1:1 with the columns; the categorical-bitset `TODO(Phase 22)` marker is retained (Open Question 1 — numeric meta only).
- `column_store_parity` is green; the full `device_dataset_parity` file is 4/4 green (the sibling 15-02 dense/layout + sparse tests were already filled in `row_data.rs`), confirming no cross-file regression.

## TDD Gate Compliance
- **RED:** the failing `column_store_parity` test was authored in plan 15-01 (commit `23fe1f3`), failing at the `todo!()` in `column_data.rs:74` — confirmed failing before implementation this plan.
- **GREEN:** this plan's single `feat(15-03)` commit (`1ffffb7`) makes it pass.
- **REFACTOR:** none needed (the implementation reused the established 15-02 helper shape as-is).

## Task Commits

1. **Task 1: CUDAColumnData per-column buffers + ColumnFeatureMeta numeric meta (upload once)** — `1ffffb7` (feat)

## Files Created/Modified
- `crates/lgbm-compute/src/kernels/column_data.rs` — filled `CudaColumnData::{new, read_column}` + module-local `column_bit_width` / `upload_column_buffer` / `decode_store_cell` helpers; added per-column `column_handles` / `column_bit_types` / `column_lens` storage fields. (216 lines.)

## Decisions Made
- Followed the plan as specified. Mirrored the 15-02 `row_data.rs` upload/decode shape as module-local helpers to keep the entire change inside the single owned file (scope boundary).

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 2 - Missing validation] feature_meta/columns length parity check**
- **Found during:** Task 1
- **Issue:** The per-feature numeric meta is parallel to the columns, but `new` accepted any-length `feature_meta` with no guard — a mismatch would silently index past the columns downstream.
- **Fix:** Added a boundary `feature_meta.len() == columns.len()` check returning `ComputeError::LengthMismatch` before any upload.
- **Files modified:** `crates/lgbm-compute/src/kernels/column_data.rs`
- **Commit:** `1ffffb7`

**Total deviations:** 1 auto-fixed (1 missing-validation). No architectural changes; no auth gates.

## Threat Model Compliance
- **T-15-PART (DoS / memory corruption — per-column buffer sizing):** mitigated — `checked_mul` on every `len * bytes_per` product before the device alloc.
- **T-15-CMP (memory corruption — unsafe in launcher):** N/A by construction — the column store/read-back is a pure safe `create_from_slice` / `read_one_unchecked` round-trip with NO kernel launch, so there is no `unsafe` to confine (CMP-01 preserved; documented in the doc-comments).

## Known Stubs
None — `CudaColumnData` is fully implemented. Per the plan prohibition, NO consumer is wired this phase (the Phase-18 tree-walk prediction kernel is the consumer); this is build + parity-test only, by design, not a stub.

## User Setup Required
None.

## Next Phase Readiness
- The §3 column store is built + parity-tested; Phase-18 prediction can consume `CudaColumnData` (per-column handles + numeric `ColumnFeatureMeta`). Categorical-bitset meta remains deferred to Phase 22.
- Sibling Wave-1 plan 15-04 (`copy_subrow.rs`) is untouched and remains its own owned file.
- No blockers.

## Self-Check: PASSED
`crates/lgbm-compute/src/kernels/column_data.rs` exists on disk; commit `1ffffb7` present in git history; `column_store_parity` green; `column_data.rs` clippy-clean.

---
*Phase: 15-on-device-device-dataset-row-subset-gather*
*Completed: 2026-06-30*
