---
status: complete
quick_id: 260608-lsx
slug: fuse-histogram-construction-and-partition-search
type: execute
date: 2026-06-08
phase: quick-260608-lsx
plan: 01
subsystem: treelearner/compute
tags: [gpu-perf, backend-seam, split-finding, parity, lad-part2]
requires:
  - 260608-lad (Backend trait + build_leaf_histograms_raw batched precedent)
provides:
  - Backend::find_best_splits_batched (default per-feature-order loop + RocmBackend override)
  - lgbm_compute::BatchedSplitFeature plain-data struct
  - two-pass scan_leaf_histogram (gate pre-pass -> ONE batched call -> argmax)
affects:
  - crates/lgbm-compute/src/lib.rs
  - crates/lgbm-compute/src/kernels/split.rs
  - crates/lgbm-treelearner/src/learner.rs
  - crates/oracle-harness/tests/kernel_parity.rs
tech-stack:
  added: []
  patterns:
    - "batched per-leaf Backend seam (default impl = bit-exact CPU anchor, GPU overrides)"
    - "gate-only pre-pass + result lookup keeps the load-bearing argmax order byte-identical"
key-files:
  created: []
  modified:
    - crates/lgbm-compute/src/lib.rs
    - crates/lgbm-compute/src/kernels/split.rs
    - crates/lgbm-treelearner/src/learner.rs
    - crates/oracle-harness/tests/kernel_parity.rs
decisions:
  - "RocmBackend override delegates to a runtime-generic per-feature f64 GPU loop (find_best_splits_batched_f64_on); the single fused per-leaf kernel is the GPU-perf follow-up. The trait seam, default bit-exact composition, two-pass learner refactor, and CPU parity gate all land here (plan-sanctioned)."
  - "Per-feature region bounds-checked (ComputeError::LengthMismatch) before any indexing; input feature order preserved in the returned Vec (the cross-feature-argmax tie-break invariant)."
metrics:
  duration_min: 18
  tasks_completed: 3
  files_modified: 4
  completed: 2026-06-08
---

# Quick 260608-lsx: Fuse Histogram Construction and Partition Search Summary

A single `Backend::find_best_splits_batched` method now finds all of a leaf's spine-feature splits, generic over `<B: Backend>` — CpuBackend uses the DEFAULT per-feature-order loop (the f64 bit-exact merge-gate anchor) and RocmBackend OVERRIDES it for the GPU; `scan_leaf_histogram` issues ONE batched call per leaf and grows trees byte-identical to today on CPU. Implements the deferred 260608-lad Part 2.

## What was built

**Task 1 — `Backend::find_best_splits_batched` (commit a6a0d1d).**
- `BatchedSplitFeature` plain-data struct in `kernels/split.rs` (re-exported as `lgbm_compute::BatchedSplitFeature`) carrying exactly the per-feature args `find_best_split` takes today (`slot_off`, `num_bin`, `offset`, `default_bin`, `most_freq_bin`, `skip_default_bin`, `na_as_missing`, `run_forward`).
- `Backend::find_best_splits_batched` trait method with a DEFAULT impl that loops `self.find_best_split` over the feature list in order, reading each feature's `[slot_off, slot_off + 2*num_bin)` region of the concatenated stride-2 f64 leaf buffer (the same layout `build_leaf_histograms_raw` produces). The default IS the bit-exact CpuBackend anchor — no CpuBackend-specific override.
- RocmBackend OVERRIDE → `kernels::split::find_best_splits_batched_f64_on<R: Runtime>` (one f64 GPU launch per feature via the proven `find_best_split_f64_on`; single fused launch deferred to the GPU-perf follow-up).
- Per-feature region bounds-checked (`ComputeError::LengthMismatch`, T-lsx-02); input order preserved (T-lsx-01); empty batch ⇒ empty Vec, no launch (T-lsx-03).

**Task 2 — two-pass `scan_leaf_histogram` (commit 44a5741).**
- PASS 1 gate-only pre-pass walks features applying the SAME gates in the SAME order as the main loop (col-sampler mask, the GOSS-critical `parent_splittable` gate, the ADV-02 interaction gate) and the SAME branch-selection predicates (not categorical, `monotone_type == 0`, not extra-trees) to classify SPINE features, recording each into a `Vec<BatchedSplitFeature>` and its `fpos → batch slot` mapping (`spine_batch_index`).
- ONE `find_best_splits_batched` call per leaf for all spine features.
- The main loop's spine branch now LOOKS UP its `SplitInfo` from the batched results instead of calling `find_best_split` inline. Categorical / monotone / extra-trees branches, the cross-feature argmax order, `this_leaf_splittable`/CEGB/monotone post-processing, the `feature_splittable` persistence, the `best_cat_threshold` cleanup, and `data_partition` are all UNCHANGED.

**Task 3 — batched-vs-per-feature CPU parity gate (commit dc73090).**
- `kernel_parity_batched_equals_per_feature_on_cpu`: lays the committed split golden's feature cases into one concatenated leaf buffer + matching `BatchedSplitFeature` records, then asserts `find_best_splits_batched` == per-feature `find_best_split` cell-for-cell via `compare_exact_f64_bits` (no tolerance) on `gain`/`left_output`/`right_output`/the four sum fields + exact integer/flag equality on `threshold`/`left_count`/`right_count`/`default_left`/`is_splittable`. Includes a multi-feature batch run proving input-order preservation (T-lsx-01). CPU-only, no LightGBM/ dependency.

## Verification (real output)

CPU floor — all GREEN:
- `cargo build --workspace` (default cpu) → exit 0
- `cargo build --workspace --tests` → exit 0
- `cargo test -p oracle-harness --test kernel_parity` → **5 passed** / 0 failed (the new `kernel_parity_batched_equals_per_feature_on_cpu` + the 4 originals, all unchanged)
- `cargo test -p oracle-harness --test learner_parity` → **29 passed** / 0 failed (trees grown bit-exact to today — the core merge gate; the plan's "12/12" figure is stale, the suite has grown to 29)

GPU floor — GREEN:
- `cargo build --workspace --features rocm` → exit 0 (the RocmBackend override compiles)

GPU hardware run (gfx1100, optional this session — RUN, real numbers):
- `cargo test -p oracle-harness --test kernel_parity --features rocm` → **8 passed, 1 failed**. The single failure is `hip::kernel_parity_split_within_tol_on_hip`, the EXISTING f32 hip split path's documented f32-vs-f64 accumulation gap (e.g. `split/reverse_winner` hip=126.15001 vs cpu_anchor=126.15, abs_diff=7.6e-6 > ORACLE_TOL=1e-6; plus a `default_left` flip on the f32 knife-edge — 04-ROCM-GAPS.md / D-03a).

## Deviations from Plan

### Pre-existing failure, NOT a regression (verified)

**`hip::kernel_parity_split_within_tol_on_hip` fails on gfx1100 — PRE-EXISTING.**
- **Found during:** Task 3 optional GPU hardware run.
- **Root cause:** the existing **f32** hip split kernel (`find_best_split_kernel_f32` / `find_best_split_raw_f32_on`) accumulates in f32, so its winner gains diverge from the f64 cpu anchor by ~2–8e-6 (> ORACLE_TOL 1e-6) and a `default_left` flip on a knife-edge case. This is the documented residual f32-vs-f64 gap (04-ROCM-GAPS.md / D-03a).
- **NOT introduced by 260608-lsx:** confirmed by checking out the four edited files at the pre-task commit `9f185c5` and re-running the hip test — it fails IDENTICALLY (same abs_diff values, same `default_left` flip; only the source line number shifts, 1067 baseline vs 1266 after my additive test insertion). 260608-lsx's changes are purely additive and never touch the f32 hip split kernel or this test. The new `find_best_splits_batched` RocmBackend override uses the f64 `find_best_splits_batched_f64_on` path (bit-exact on gfx1100), not the f32 mirror.
- **Disposition:** out of scope for this task (an existing ROCm gap, not a fused-split-finding defect). No assertion weakened. The single fused per-leaf GPU kernel (the launch-count collapse) is the GPU-perf follow-up; this task lands the seam + the f64 numerics.

Otherwise: plan executed as written. CpuBackend default impl is the bit-exact anchor; no CpuBackend override added (per the plan).

## Numerical contract

The non-negotiable CLAUDE.md contract holds: the CpuBackend f64-fold path is BIT-EXACT to today (kernel_parity 5/5 incl. the new batched-vs-per-feature assertion, learner_parity 29/29 — trees grown byte-identical). No numerical assertion was weakened, tolerance-wrapped, or deleted. `LightGBM/` was never git-added; `data_partition` untouched.

## Self-Check: PASSED
- Commits a6a0d1d, 44a5741, dc73090 — all FOUND in git log.
- All four modified files — all FOUND on disk.
