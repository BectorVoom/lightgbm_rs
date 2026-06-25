---
phase: quick-260625-qn9
plan: 01
subsystem: lgbm-treelearner / data_partition
tags: [perf, partition, bit-exact, spike-032, cpu-anchor]
requires:
  - spike-032 V1 fold (spike032_partition_validation_fold_ab.rs::v1_fold)
provides:
  - Folded split_fused_host with single-gather inline range-check
affects:
  - DataPartition::split host-partition path
tech-stack:
  added: []
  patterns:
    - "fold the per-row validation into the route gather (one random gather + a branch)"
key-files:
  created: []
  modified:
    - crates/lgbm-treelearner/src/data_partition.rs
decisions:
  - "Range-check returns BEFORE any write to route/indices, preserving the no-mutation + lowest-index BinIndexOutOfRange guarantee"
metrics:
  duration: ~6 min
  completed: 2026-06-25
status: complete
---

# Phase quick-260625-qn9 Plan 01: Wire spike-032 V1 fold into production Summary

Eliminated the redundant validation random-gather in `DataPartition::split_fused_host` by folding the per-row bin range-check INTO pass-1's route gather (spike-032 V1) — one random gather over the leaf's scattered rows instead of two, bit-exact on both the success path and the lowest-index error path; the CPU bit-exact anchor gate stays green.

## What Was Built

- **Task 1 — code fold (`crates/lgbm-treelearner/src/data_partition.rs`):**
  - Deleted the standalone per-row validation `for i in 0..count` loop (and its stale ascending-leaf-position explanatory comment) that read `feature_bins.bin(row)` and checked `b >= num_bin`.
  - Merged the range-check into pass-1's route+count loop: each row gathers `b = feature_bins.bin(row as usize)` ONCE, range-checks `if b >= num_bin { return Err(BinIndexOutOfRange { row: i, bin: b, num_bin }) }` BEFORE writing the `route` scratch, then routes off the already-gathered `b` via `go_right(b)` (no re-gather).
  - Updated the `split_fused_host` doc comment to note the range-check is now folded into the single pass-1 gather (spike-032 V1), with the no-mutation early-return invariant spelled out.
  - KEPT UNCHANGED: the `num_bin == 0` and `threshold >= num_bin` pre-checks, the router setup block, and pass-2 scatter + `copy_from_slice` + count return.

- **Task 2 — bit-exact CPU anchor gate:** ran the full merge gate on the main checkout (untracked `LightGBM/` ref tree + built `lib_lightgbm` 4.6 present), both green.

## Correctness Invariants Preserved

- The range-check `return Err(..)` occurs BEFORE any `route` write and before pass-2, so an early-return on the first bad bin leaves `self.indices` UNMUTATED — pass-1 only writes the local `route`/`left_count`.
- ASCENDING leaf-position iteration (`for i in 0..count`) so the FIRST offender is the LOWEST-index one — same `BinIndexOutOfRange { row: i, .. }` as the deleted loop.
- `go_right` receives the already-gathered `b` (no redundant `feature_bins.bin(row)` re-call).
- Success-path `[left|right]` order is byte-identical (pass-2 scatter untouched).

## Gate Result (auditable)

`cargo build -p lgbm-treelearner` — clean, no warnings from the edit.

`cargo test -p lgbm-treelearner --lib`:
```
running 79 tests
test data_partition::tests::split_fused_equals_serial ... ok
test result: ok. 77 passed; 0 failed; 2 ignored; 0 measured; 0 filtered out; finished in 0.19s
```
(the 2 ignored are pre-existing `#[ignore]` cells, unrelated to this edit; `split_fused_equals_serial` + the three `data_partition::tests::*` partition tests all ran and passed.)

`cargo test -p oracle-harness --test raw_bin_train_parity` (end-to-end bit-exact vs real lib_lightgbm 4.6):
```
running 2 tests
test raw_bin_train_matches_cpp_golden ... ok
test raw_bin_train_matches_identity_bin ... ok
test result: ok. 2 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.13s
```

Both gates GREEN — the fold is byte-identical to the shipped path. The CPU bit-exact anchor is unregressed.

## Deviations from Plan

None - plan executed exactly as written.

## Commits

- `6b6fb09`: perf(quick-260625-qn9): fold spike-032 V1 validation gather into split_fused_host pass-1 (one random gather, bit-exact)

## Self-Check: PASSED

- File modified exists: `crates/lgbm-treelearner/src/data_partition.rs` — FOUND
- Commit exists: `6b6fb09` — FOUND
- No accidental file deletions in the commit.
