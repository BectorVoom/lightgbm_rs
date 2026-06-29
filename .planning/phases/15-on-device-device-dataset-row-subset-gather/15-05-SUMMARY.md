---
phase: 15-on-device-device-dataset-row-subset-gather
plan: 05
subsystem: testing
tags: [merge-gate, device-dataset, cuda-on-device, parity, regression, additive]

# Dependency graph
requires:
  - phase: 15-02
    provides: "§13 CUDARowData row store + DivideCUDAFeatureGroups + dense/sparse partition-local layout (ODL-03)"
  - phase: 15-03
    provides: "§3 CUDAColumnData per-column store + numeric per-feature meta (ODL-03)"
  - phase: 15-04
    provides: "CopySubrow row-subset gather + on-device bagging draw anchored to host bag_data_indices (ODL-04)"
provides:
  - "D-10 merge-gate evidence: cargo test --workspace green with LGBM_CUDA_ON_DEVICE unset"
  - "Confirmation that Phase-15 device-dataset + gather additions are fully additive (default cpu/ROCm/host-CUDA paths byte-unchanged)"
affects: [phase-16-histogram-constructor, gsd-verify-work]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Merge-gate verification plan: no source changes, only goal-backward proof that additive modules break nothing on the default path"

key-files:
  created:
    - .planning/phases/15-on-device-device-dataset-row-subset-gather/15-05-SUMMARY.md
  modified: []

key-decisions:
  - "Clippy warnings in lgbm-compute are all pre-existing (Phase-14 split/subtract/primitives/partition + earlier crates); the 3 new Phase-15 files (column_data/copy_subrow/row_data) are clippy-clean — out-of-scope pre-existing warnings left untouched per SCOPE BOUNDARY"

patterns-established:
  - "The hard merge gate (D-10) is the cpu f64 anchor: raw_bin_train_matches_cpp_golden + learner_parity byte-unchanged with the on-device seam OFF"

requirements-completed: [ODL-03, ODL-04]

# Metrics
duration: 10min
completed: 2026-06-29
status: complete
---

# Phase 15 Plan 05: Device-Dataset Merge Gate (D-10) Summary

**Hard merge gate green: the full workspace suite passes with `LGBM_CUDA_ON_DEVICE` unset, the cpu f64 anchor suites (`raw_bin_train_matches_cpp_golden`, `learner_parity`) are byte-unchanged, and the Phase-15 device-dataset + row-subset-gather additions are confirmed fully additive.**

## Performance

- **Duration:** 10 min
- **Started:** 2026-06-29T22:09:49Z
- **Completed:** 2026-06-29T22:19:30Z
- **Tasks:** 1
- **Files modified:** 1 (SUMMARY only — verification-only plan, zero source changes)

## Accomplishments

- Ran the full `cargo test --workspace` merge gate with `LGBM_CUDA_ON_DEVICE` UNSET — **all green, 0 failures** across every crate.
- Confirmed the cpu f64 bit-exact anchor suites pass byte-unchanged: `raw_bin_train_matches_cpp_golden` (1/1) and `learner_parity` (29/29).
- Confirmed `on_device_growth_supported()` still returns `false` (trait default at `crates/lgbm-compute/src/lib.rs:1239`; asserted false for both CpuBackend and GpuBackend in `learner_parity.rs:2481-2486`) — this phase mirrors the binning resident store, it does not grow.
- Confirmed the 3 new Phase-15 modules (`column_data`, `copy_subrow`, `row_data`) are registered **ungated** in `kernels/mod.rs` (reachable by the cpu f64 anchor per D-08) and are **clippy-clean**.

## Merge-Gate Results (LGBM_CUDA_ON_DEVICE unset)

| Suite | Command | Result |
|-------|---------|--------|
| Full workspace | `cargo test --workspace` | **PASS** — 0 failed across all crates (a few pre-existing `ignored` tests: 1 in lgbm-compute, 2 in lgbm-treelearner) |
| cpu f64 anchor (bit-exact) | `cargo test -p oracle-harness raw_bin_train_matches_cpp_golden` | **PASS** 1/1 — byte-unchanged |
| learner parity (anchor) | `cargo test -p oracle-harness --test learner_parity` | **PASS** 29/29 — byte-unchanged |
| lgbm-compute (all) | `cargo test -p lgbm-compute` | **PASS** 102 total / 0 failed / 1 ignored |
| └ device_dataset_parity (ODL-03) | `cargo test -p lgbm-compute --test device_dataset_parity` | **PASS** 4/4 |
| └ copy_subrow_parity (ODL-04) | `cargo test -p lgbm-compute --test copy_subrow_parity` | **PASS** 4/4 |
| lgbm-treelearner (lib) | `cargo test -p lgbm-treelearner --lib` | **PASS** 77/0/2-ignored |
| lgbm (facade) | `cargo test -p lgbm` | **PASS** 41/0 |
| clippy (new code) | `cargo clippy -p lgbm-compute --tests` | **EXIT 0** — new Phase-15 files clean; remaining warnings all pre-existing (see below) |
| `on_device_growth_supported()` | grep `crates/lgbm-compute/src/lib.rs:1239` | **`false`** (trait default; asserted in learner_parity) |

## D-10 Confirmation (default path byte-unchanged)

The Phase-15 changes are purely additive:
- The only `mod.rs` change registers `column_data` / `copy_subrow` / `row_data` (ungated stubs filled per-plan); no default-path source was edited.
- With `LGBM_CUDA_ON_DEVICE` unset, the learner's on-device eligibility gate (`backend.on_device_growth_supported() && cuda_on_device_env()`) is `false && false` → the byte-unchanged host/per-leaf path is always taken.
- The cpu f64 bit-exact anchor (`raw_bin_train_matches_cpp_golden`) and `learner_parity` pass unchanged, proving the default cpu/ROCm/host-CUDA paths are byte-identical. **No default-path edge was introduced.**

T-15-DEFAULT (Tampering, default training path) is mitigated: the full merge gate with the seam OFF proves no regression.

## Files Created/Modified

- `.planning/phases/15-on-device-device-dataset-row-subset-gather/15-05-SUMMARY.md` — this merge-gate evidence (only file touched; no source changes).

## Decisions Made

- **Pre-existing clippy warnings left untouched (SCOPE BOUNDARY).** `cargo clippy -p lgbm-compute --tests` emits 38 warnings, but a file-path breakdown shows they all originate in pre-existing code: `lib.rs` (22), `kernels/split.rs` (8), `kernels/subtract.rs` (6), `kernels/primitives.rs` (2), `kernels/partition.rs` (2), plus earlier crates (`dataset/bin_mapper.rs`, `model/format.rs`, etc.). **Zero** warnings reference the 3 new Phase-15 files. The `unused variable: shapes` warning surfaces only from a pre-existing spike example (`examples/spike042_vector_scan_pair_ab.rs:186`). All are out of scope for this merge-gate plan and not introduced by Phase-15.

## Deviations from Plan

None - plan executed exactly as written. Verification-only plan; no source changes, no deviation rules triggered.

## Issues Encountered

None. Every required suite was green on the first run.

## Next Phase Readiness

- Phase 15 (device dataset + row-subset gather, ODL-03/04) is merge-gate clean and ready for `/gsd-verify-work`.
- The additive, OFF-by-default `LGBM_CUDA_ON_DEVICE` seam is intact; Phase 16 (histogram constructor, ODL-09/10) can build on the device-dataset stores with the same additive discipline.
- No blockers.

## Self-Check: PASSED

---
*Phase: 15-on-device-device-dataset-row-subset-gather*
*Completed: 2026-06-29*
