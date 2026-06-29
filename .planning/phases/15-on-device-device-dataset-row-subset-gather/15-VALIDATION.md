---
phase: 15
slug: on-device-device-dataset-row-subset-gather
status: draft
nyquist_compliant: true
wave_0_complete: false
created: 2026-06-30
---

# Phase 15 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.
> Derived from 15-RESEARCH.md "Validation Architecture". Every numeric output is
> anchored host-vs-device (cubecl-cpu f64 fold / host Rust binned values / host
> `Random`) — **never GPU-vs-GPU** (D-08, def-f8u-01).

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | Rust built-in `#[test]` (cargo test) — project standard, no external framework |
| **Config file** | none — per-crate `tests/*.rs` integration tests + in-module `#[cfg(test)]` units |
| **Quick run command** | `cargo test -p lgbm-compute --test <touched_test_file>` |
| **Full suite command** | `cargo test --workspace` |
| **Estimated runtime** | ~30–120 seconds (workspace) |

---

## Sampling Rate

- **After every task commit:** Run `cargo test -p lgbm-compute --test <touched_test_file>` (the new `device_dataset_parity` / `copy_subrow_parity` files)
- **After every plan wave:** Run `cargo test -p lgbm-compute` + `cargo test -p lgbm-boosting` (bagging anchor) + `cargo clippy -p lgbm-compute --tests`
- **Before `/gsd-verify-work`:** `cargo test --workspace` must be green (merge gate D-10 unchanged)
- **Max feedback latency:** ~120 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| Wave-0 | 00 | 0 | ODL-03/04 | — | N/A (test scaffold) | unit | `cargo test -p lgbm-compute --test device_dataset_parity` | ❌ W0 | ⬜ pending |
| §13 row store | — | 1 | ODL-03 | — | binned values per-column match host (dense, u8/16/32) | parity | `cargo test -p lgbm-compute --test device_dataset_parity dense_bin_parity_all_widths` | ❌ W0 | ⬜ pending |
| DivideCUDAFeatureGroups | — | 1 | ODL-03 | T-15-PART | partition count/offsets match hand-computed; large-bin spill increments `NumLargeBinPartition` | unit | `cargo test -p lgbm-compute --test device_dataset_parity feature_partition_layout` | ❌ W0 | ⬜ pending |
| Sparse CSR 3×3 re-lay | — | 1 | ODL-03 | T-15-PTR | each `row_ptr_type{16,32,64}` cell exercised; `partition_hist_start` subtracted → partition-local bins | parity | `cargo test -p lgbm-compute --test device_dataset_parity sparse_relay_3x3_and_partition_local` | ❌ W0 | ⬜ pending |
| §3 column store | — | 1 | ODL-03 | — | binned values + numeric per-feature meta match host | parity | `cargo test -p lgbm-compute --test device_dataset_parity column_store_parity` | ❌ W0 | ⬜ pending |
| CopySubrow gather | — | 2 | ODL-04 | T-15-IDX | compacted subset == host `BinColumn::gather(used_indices)` all widths, dense + row-major | parity | `cargo test -p lgbm-compute --test copy_subrow_parity gather_matches_host_all_widths` | ❌ W0 | ⬜ pending |
| Bagging draw | — | 2 | ODL-04 | T-15-IDX | draw stream + route bit-for-bit vs host `bag_data_indices` (NextFloat `to_bits()`, route set+order); spans ≥2 blocks (>1024 rows) | parity | `cargo test -p lgbm-compute --test copy_subrow_parity bagging_draw_matches_host` | ❌ W0 | ⬜ pending |
| Arbitrary index set | — | 2 | ODL-04 | T-15-IDX | gather works for arbitrary host-supplied (GOSS-shaped) index set | unit | `cargo test -p lgbm-compute --test copy_subrow_parity gather_arbitrary_indices` | ❌ W0 | ⬜ pending |
| Merge gate | — | 3 | D-10 | — | default-path suites byte-identical with `LGBM_CUDA_ON_DEVICE` unset | regression | `cargo test --workspace` | ✅ exists | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `crates/lgbm-compute/tests/device_dataset_parity.rs` — ODL-03 (dense all widths, partition layout, sparse 3×3 + partition-local, §3 column store). Mirror the `cuda_random_parity.rs` host-vs-device anchor shape.
- [ ] `crates/lgbm-compute/tests/copy_subrow_parity.rs` — ODL-04 (`CopySubrow` vs `BinColumn::gather`, bagging draw vs host `bag_data_indices`, arbitrary index set).
- [ ] In-test sparse-column synthesizer (helper) — generates columns whose nnz forces each `row_ptr_type{16,32,64}` + a column over the shared-hist budget (D-04). Lives in the test file or a `tests/` helper module.
- [ ] No framework install needed — cargo test is in place.

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| A1 constant confirmation (`DP_SHARED_HIST_SIZE=6144`) | ODL-03 | Design-doc summary value; no parity impact per §17 (only grouping) | If `LightGBM/` reference tree is fetched, cross-check `include/LightGBM/cuda/cuda_row_data.hpp` / `cuda_histogram_constructor.hpp` constants. Otherwise treat the design doc as authoritative. |

*All other phase behaviors have automated verification.*

---

## Validation Sign-Off

- [x] All tasks have `<automated>` verify or Wave 0 dependencies
- [x] Sampling continuity: no 3 consecutive tasks without automated verify
- [x] Wave 0 covers all MISSING references
- [x] No watch-mode flags
- [x] Feedback latency < 120s
- [x] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
