---
phase: 14-scaffold-oracle-slice-0
plan: 01
subsystem: infra
tags: [cubecl, backend-seam, on-device-tree, crate-cycle, payload-pod, cuda]

# Dependency graph
requires:
  - phase: 04 (compute backend seam)
    provides: "trait Backend on lgbm-compute + GpuBackend<R> generic impl + CpuBackend anchor"
provides:
  - "lgbm_dataset::LeafPartitionLayout — lower-crate POD partition payload P (D-03 Option A)"
  - "lgbm-compute → lgbm-model path dep (acyclic) so the seam can name lgbm_model::Tree"
  - "Backend::on_device_growth_supported() -> bool, default false (CpuBackend + GpuBackend<R> false in Slice 0)"
  - "Backend::grow_tree_on_device(..) -> Result<Option<(Tree, LeafPartitionLayout)>, ComputeError>, default Ok(None)"
  - "GpuBackend<R> explicit no-op override of grow_tree_on_device (SC#2 proof)"
affects: [14-02 (learner fork + DataPartition::from_payload + env toggle), 14-03 (oracle assert_on_device_tree_matches_cpu_anchor), slice-1 (real on-device kernel)]

# Tech tracking
tech-stack:
  added: [lgbm-compute→lgbm-model internal path dep]
  patterns:
    - "Lower-crate POD payload type to break a would-be crate cycle (name P in a crate below both producer and consumer)"
    - "Additive Backend seam returns Ok(None) (not Err(NotSupported)) to keep the default path error-noise-free"
    - "Discriminator defaults false on one generic GpuBackend<R> impl shared by ROCm/CUDA/WGPU — never flip a shared impl to true for one backend"

key-files:
  created: []
  modified:
    - crates/lgbm-dataset/src/dataset.rs
    - crates/lgbm-dataset/src/lib.rs
    - crates/lgbm-compute/Cargo.toml
    - crates/lgbm-compute/src/lib.rs

key-decisions:
  - "D-03 resolved via Option A: name the partition payload P (LeafPartitionLayout) in lgbm-dataset (a lower crate), so the (Tree, P) seam lives on Backend without a treelearner→compute→treelearner cycle."
  - "Seam returns Ok(None) ('I did not grow it'), NOT a typed Err(NotSupported), keeping the default CPU/ROCm route error-noise-free."
  - "on_device_growth_supported() stays false on GpuBackend<R> in Slice 0 — it is ONE generic impl shared by ROCm/CUDA/WGPU; a true would claim all three and no kernel exists until Slice 1."

patterns-established:
  - "Lower-crate payload P pattern: when a trait seam in crate B must return a type owned conceptually by crate C (which depends on B), define a POD mirror in a crate below both."
  - "No-op-but-explicit override on the GPU backend proves (SC#2) the default route is provably untouched, not merely inherited."

requirements-completed: [ODL-01]

# Metrics
duration: 12min
completed: 2026-06-29
status: complete
---

# Phase 14 Plan 01: scaffold-oracle-slice-0 Summary

**ODL-01 scaffolding — `LeafPartitionLayout` POD payload in lgbm-dataset + the additive `Backend` on-device tree-growth seam (`on_device_growth_supported` default-false discriminator and `grow_tree_on_device` default-`Ok(None)`), with `GpuBackend<R>` a provable no-op, zero behavior change.**

## Performance

- **Duration:** ~12 min
- **Started:** 2026-06-29
- **Completed:** 2026-06-29
- **Tasks:** 2
- **Files modified:** 4

## Accomplishments
- `lgbm_dataset::LeafPartitionLayout` — the D-03 Option A lower-crate partition payload `P` (POD: `num_data`, `indices`, `leaf_begin`, `leaf_count`), mirroring the four fields `DataPartition` wraps, re-exported from `lgbm-dataset`, with no upward dep on treelearner/compute (acyclic).
- `lgbm-compute → lgbm-model` path dep added (verified acyclic — lgbm-model depends only on lgbm-core + lgbm-dataset + thiserror) so the seam can name `lgbm_model::Tree`.
- `Backend::on_device_growth_supported() -> bool` discriminator, default `false`; both `CpuBackend` (inherited) and `GpuBackend<R>` (kept default) report false in Slice 0.
- `Backend::grow_tree_on_device(gradients, hessians, num_leaves, max_depth) -> Result<Option<(lgbm_model::Tree, lgbm_dataset::LeafPartitionLayout)>, ComputeError>` default `Ok(None)`, plus an explicit `GpuBackend<R>` no-op override returning `Ok(None)` (SC#2 proof). The seam doc-comment carries the cubecl-0.10 Slice-1 kernel checklist (no global barrier; Atomic<i64> broken; wrapping_add not an intrinsic; plane-sum ≤ plane width; launch_unchecked unsafe).
- Full merge gate green AND byte-unchanged with `LGBM_CUDA_ON_DEVICE` unset.

## Task Commits

Each task was committed atomically:

1. **Task 1: Add LeafPartitionLayout payload struct to lgbm-dataset** - `1e9e086` (feat)
2. **Task 2: Add the Backend discriminator + grow_tree_on_device seam (no-op) and the lgbm-model dep** - `45edf6c` (feat)

## Files Created/Modified
- `crates/lgbm-dataset/src/dataset.rs` - Added `pub struct LeafPartitionLayout` (POD, `#[derive(Debug, Clone)]`) beside `EfbSamples`.
- `crates/lgbm-dataset/src/lib.rs` - Re-export `LeafPartitionLayout` alongside `Dataset`/`FinishedDataset`.
- `crates/lgbm-compute/Cargo.toml` - Added `lgbm-model = { path = "../lgbm-model" }` (acyclic).
- `crates/lgbm-compute/src/lib.rs` - Added `on_device_growth_supported()` discriminator (default false) and `grow_tree_on_device()` seam (default Ok(None)) on `trait Backend`, plus the `GpuBackend<R>` no-op override.

## Decisions Made
- **D-03 → Option A.** The partition payload `P` is named in `lgbm-dataset` (a crate below both lgbm-compute and lgbm-treelearner), so the `(Tree, P)` seam can live on `Backend` without the treelearner→compute→treelearner cycle. The learner reconstructs a real `DataPartition` from this POD in Plan 02.
- **`Ok(None)` not `Err(NotSupported)`** — the default route stays error-noise-free; "unsupported" is a quiet `None`, not an error the learner must filter.
- **Discriminator stays false on `GpuBackend<R>`** — one generic impl shared by ROCm/CUDA/WGPU; flipping it true would wrongly claim all three, and no on-device kernel exists until Slice 1.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Reworded a doc-comment that contained the literal string `use lgbm_treelearner`**
- **Found during:** Task 2 (Backend seam doc-comment)
- **Issue:** The seam's doc-comment explained the cycle hazard using the literal phrase `use lgbm_treelearner`, which made the acceptance grep `grep -c 'use lgbm_treelearner' ... == 0` report `1` (a doc mention, not a real import — there is no actual import).
- **Fix:** Reworded the doc-comment to "importing lgbm-treelearner here" so the literal `use lgbm_treelearner` no longer appears anywhere in the file; the no-cycle guarantee is unchanged.
- **Files modified:** crates/lgbm-compute/src/lib.rs
- **Verification:** `grep -c 'use lgbm_treelearner' crates/lgbm-compute/src/lib.rs` now returns `0`; build + tests green.
- **Committed in:** `45edf6c` (Task 2 commit)

---

**Total deviations:** 1 auto-fixed (1 blocking — cosmetic doc reword to satisfy the literal acceptance grep)
**Impact on plan:** No functional change; the seam, return type, and acyclic guarantee are exactly as specified. No scope creep.

## Issues Encountered
None.

## Verification Evidence
- `cargo build --workspace` exits 0 (no crate cycle from the lgbm-model edge).
- `cargo test -p lgbm-dataset` 75+ green; `cargo test -p lgbm-compute` 52 passed / 1 ignored.
- Merge gate: `cargo test -p oracle-harness --test raw_bin_train_parity` 2/2; `--test learner_parity` 29/29.
- `cargo test --workspace` — **775 passed, 0 failed** (byte-unchanged with `LGBM_CUDA_ON_DEVICE` unset, SC#1).
- `grep 'fn grow_tree_on_device'` → 2 sites (trait default + GpuBackend<R> override); `grep -c 'use lgbm_treelearner'` → 0.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- The concrete `P` type (`LeafPartitionLayout`) and the seam signature are now nameable, unblocking Plan 02 (learner fork: `cuda_on_device_env`, `on_device_eligible`, `DataPartition::from_payload`, the `train_inner` fork) and Plan 03 (oracle `assert_on_device_tree_matches_cpu_anchor`).
- The runtime input `LGBM_CUDA_ON_DEVICE` is read in Plan 02; this plan adds only the off-by-default seam it gates.
- Slice 1 will wire a real on-device kernel and flip the discriminator; the cubecl-0.10 checklist is baked into the seam doc-comment for that work.

## Self-Check: PASSED

All 4 modified files present; both task commits (`1e9e086`, `45edf6c`) exist in git history.

---
*Phase: 14-scaffold-oracle-slice-0*
*Completed: 2026-06-29*
