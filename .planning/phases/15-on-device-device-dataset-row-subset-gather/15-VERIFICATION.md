---
phase: 15-on-device-device-dataset-row-subset-gather
verified: 2026-06-30T00:00:00Z
status: passed
score: 4/4 must-haves verified
behavior_unverified: 0
overrides_applied: 0
---

# Phase 15: Device Dataset + Row-Subset Gather Verification Report

**Phase Goal:** The binned feature matrix and the bagging/GOSS subset live resident on device in the feature-partition layout the histogram kernel is built around.
**Verified:** 2026-06-30
**Status:** passed
**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | On-device columnar binned dataset (u8/16/32 dispatch; dense + sparse CSR) resident on device, carrying the feature-partition layout; too-wide column → own large-bin partition (global-mem path) | ✓ VERIFIED | `column_data.rs` `CudaColumnData` (per-column native-width device buffers); `row_data.rs` `CudaRowData` dense + sparse CSR with `divide_cuda_feature_groups`→`FeaturePartitionLayout` (budget = `shared_hist_size/2`, `num_large_bin_partition`), full 3×3 (`bit_type`×`row_ptr_type`) sparse dispatch (lines 423-441). Tests `dense_bin_parity_all_widths`, `feature_partition_layout`, `sparse_relay_3x3_and_partition_local`, `column_store_parity` all PASS. |
| 2 | On-device row-subset gather (CopySubrow analog) builds the bagging/GOSS subset, anchor-pinned to the host subset-selection draw sequence | ✓ VERIFIED | `copy_subrow.rs` `copy_subrow_kernel` (D-07) + `copy_subrow_on` launcher + `bagging_draw_on`, anchored bit-for-bit to inline `host_bag_data_indices` via shared `lgbm_core::random::Random`. Tests `gather_matches_host_all_widths`, `bagging_draw_matches_host` (num_data=2500 → 3 blocks, D-05), `gather_arbitrary_indices` (GOSS-shaped), `out_of_range_index_rejected_before_launch` (V5) all PASS. |
| 3 | Resident dataset reproduces host binned values exactly (per-column bin parity); bin-width + partition dispatch validated across all three widths and the large-bin spill | ✓ VERIFIED | CR-01 fix present (`row_data.rs:415-416`: `bit_type` sized by max partition-local value via `bit_type_for_value`, not raw column width). New test `sparse_partition_local_nonzero_offset_fold` (line 269) asserts `read_partition_local_bin == raw + column_hist_offset` with NON-ZERO offsets (200/400 → partition-local bins up to 599, which overflow a u8 store) across all three `row_ptr_type` widths {16,32,64}, plus corpus B large-bin spill. `dense_bin_parity_all_widths` now uses real bin counts [200,2000,100000] with a non-zero packed offset + spill (WR-02). All three widths forced (`widths_seen` BTreeSet assertion). All PASS. |
| 4 | CPU / ROCm / existing-host-CUDA paths byte-unchanged; merge gate green | ✓ VERIFIED | Cumulative phase diff touches ONLY 6 files, all under `crates/lgbm-compute` (3 new modules + 10-line ungated `mod.rs` registration + 2 new test files); no file outside lgbm-compute changed. New modules are ungated but inert (no consumer this phase). `on_device_growth_supported()` untouched (stays trait-default false, lib.rs:1239/2212). Full `cargo test -p lgbm-compute` green: 52 unit + all integration tests pass, 0 failed. |

**Score:** 4/4 truths verified (0 present, behavior-unverified)

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `crates/lgbm-compute/src/kernels/mod.rs` | ungated registration of 3 modules | ✓ VERIFIED | `pub mod column_data/copy_subrow/row_data` (lines 35-37), NOT `#[cfg(feature="gpu")]`; comment confirms "Ungated like random/split" |
| `crates/lgbm-compute/src/kernels/row_data.rs` | §13 CUDARowData dense+sparse re-lay + layout | ✓ VERIFIED | 736 lines; `divide_cuda_feature_groups`, `get_dense_data_partitioned`, `get_sparse_data_partitioned` (3×3 dispatch), accessors, CR-01 width-sizing; no `todo!()` |
| `crates/lgbm-compute/src/kernels/column_data.rs` | §3 CUDAColumnData per-column store + meta | ✓ VERIFIED | 216 lines; `CudaColumnData::new`/`read_column`, `ColumnFeatureMeta`; no `todo!()` |
| `crates/lgbm-compute/src/kernels/copy_subrow.rs` | CopySubrow kernel + launcher + bagging draw | ✓ VERIFIED | 246 lines; real kernel body, `copy_subrow_on` (CR-02 + V5 guards), `bagging_draw_on`; no `todo!()` |
| `crates/lgbm-compute/tests/device_dataset_parity.rs` | ODL-03 parity tests + synthesizer | ✓ VERIFIED | 5 tests pass incl. `sparse_partition_local_nonzero_offset_fold` |
| `crates/lgbm-compute/tests/copy_subrow_parity.rs` | ODL-04 gather + bagging + inline ref | ✓ VERIFIED | 4 tests pass; inline `host_bag_data_indices`, no lgbm-boosting import |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|----|--------|---------|
| `kernels/mod.rs` | `kernels/row_data.rs` | `pub mod row_data;` (ungated) | ✓ WIRED | line 37, no cfg gate |
| `copy_subrow_parity.rs` | `lgbm_core::random::Random` | inline host bagging ref (NO lgbm-boosting dev-dep) | ✓ WIRED | `use lgbm_core::random::Random` (line 23); Cargo dev-deps = only `oracle-harness`; lgbm-boosting appears only in explanatory comments |
| `bagging_draw_on` | `kernels::random::draw_next_float_on` | per-block device RNG draw | ✓ WIRED | copy_subrow.rs:30,224 |

### Prohibition Checks

| Prohibition | Status | Evidence |
|-------------|--------|----------|
| NEVER add lgbm-boosting as dev-dep of lgbm-compute (crate cycle) | ✓ HELD | dev-deps = `oracle-harness` only; no `lgbm-boosting`/`lgbm_boosting` code reference (comments only) |
| NEVER gate the 3 new modules behind `#[cfg(feature=gpu)]` | ✓ HELD | registered ungated; cpu f64 anchor runs all 9 new tests |
| `on_device_growth_supported()` stays false (do not touch) | ✓ HELD | untouched; remains trait default |

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|------------|-------------|--------|----------|
| ODL-03 | 15-01..15-03,15-05 | On-device columnar binned dataset (u8/16/32; dense+sparse CSR) + feature-partition layout + large-bin spill | ✓ SATISFIED | Truths #1, #3; REQUIREMENTS.md marks ODL-03 = Complete |
| ODL-04 | 15-01,15-04,15-05 | On-device row-subset gather (CopySubrow) anchored to host draw | ✓ SATISFIED | Truth #2; REQUIREMENTS.md marks ODL-04 = Complete |

No orphaned requirements: REQUIREMENTS.md maps Phase 15 to exactly {ODL-03, ODL-04}, both claimed in PLAN frontmatter.

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| Two phase-15 parity test files | `cargo test -p lgbm-compute --test device_dataset_parity --test copy_subrow_parity` | 5 + 4 = 9 passed, 0 failed | ✓ PASS |
| Crate builds (modules registered) | `cargo build -p lgbm-compute` | Finished, exit 0 | ✓ PASS |
| Merge gate (full compute suite) | `cargo test -p lgbm-compute` | 52 unit + all integration green, 0 failed | ✓ PASS |

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| `tests/copy_subrow_parity.rs` | 119-129 | WR-04 tautological `assert_eq!(host_cnt, host_cnt)` + dead `take_while(\|\|true)` | ℹ️ Info | Dead/misleading test code, BUT the preceding `assert_eq!(device, host_indices)` (line 118) proves the full layout incl. the in-bag split point bit-for-bit — test is NOT degenerate. Acknowledged out-of-fix-scope in 15-REVIEW.md. |
| `src/kernels/column_data.rs` | 28 | `TODO(Phase 22)` categorical bitset meta | ℹ️ Info | References formal follow-up (ODL-22 / Phase 22) — passes debt-marker gate; numeric meta is in-scope and complete. |

Remaining 15-REVIEW.md items (WR-03 layout/columns consistency guard, IN-01 duplicated helpers, IN-02 row_ptr near-capacity, IN-03 whole-buffer read-back) are all Info/Warning, explicitly out of fix scope, and do not block the phase goal.

### Review Resolution Verification

| ID | Claimed | Verified in code |
|----|---------|------------------|
| CR-01 (sparse width truncation) | RESOLVED `874fd07` | ✓ `row_data.rs:415-416` sizes `bit_type` from `max(vals)` partition-local range; regression test passes (would truncate 399→143 pre-fix) |
| CR-02 (copy_subrow OOB) | RESOLVED `8c62207` | ✓ `copy_subrow.rs:92-97` returns `LengthMismatch` when `column.len() != num_data` before launch |
| WR-01 (nonzero-offset fold) | RESOLVED `07d0427` | ✓ `sparse_partition_local_nonzero_offset_fold` present, asserts `local == raw + offset` with offsets 200/400 across all 3 widths + spill |
| WR-02 (dense offset round-trip) | RESOLVED `07d0427` | ✓ `dense_bin_parity_all_widths` uses real bin counts [200,2000,100000], asserts global bin with non-zero packed offset + spill |

### Gaps Summary

No gaps. All four success criteria are observably true in the codebase, confirmed by 9 passing host-vs-device parity tests on the cpu f64 anchor (the merge gate), which is the deterministic reference contract for this project. The two Critical parity bugs found in review (CR-01, CR-02) are fixed and the previously-degenerate tests (WR-01, WR-02) now genuinely exercise non-zero partition offsets and the large-bin spill across all three bin widths and all three row-pointer widths. The phase is fully additive (6 lgbm-compute files), leaving CPU/ROCm/CUDA production paths byte-unchanged.

---

_Verified: 2026-06-30_
_Verifier: Claude (gsd-verifier)_
