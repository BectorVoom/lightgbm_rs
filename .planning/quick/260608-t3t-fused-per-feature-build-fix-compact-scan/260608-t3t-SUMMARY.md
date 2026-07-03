---
status: complete
phase: 260608-t3t
plan: 01
subsystem: lgbm-compute / lgbm-treelearner (GPU split-finding)
tags: [gpu, rocm, fused-kernel, perf, launch-bound, bit-exact, honest-negative]
requires:
  - "RocmBackend resident histogram pool + slot mirror (260608-p90)"
  - "split_scan_body shared #[cube] scan helper (260608-mc5)"
  - "fix_compact_kernel f64 fix+compact logic (260608-oib / s2b Lever A)"
  - "construct_leaf_hist_resident_kernel resident bin gather (260608-nn7)"
provides:
  - "build_fix_scan_fused_kernel (#[cube(launch)], GPU-only) — build+fix+compact+scan in 1 launch"
  - "build_fix_scan_resident_f64_on host launcher (build-all / scan-subset, returns hist Handle + SplitInfos)"
  - "Backend::build_fix_scan_resident seam (default error / RocmBackend impl)"
  - "fused_directly_built_eligible gate + LGBM_FUSED_FORCE bench knob (FUSED_MAX_NUM_DATA=-1, OFF by default)"
  - "kernel_parity_build_fix_scan_equals_host_on_hip (fused==host bit-exact oracle)"
  - "learner_parity_fused_equals_host_tree_on_hip (fused==host TREE equivalence)"
affects:
  - "directly-built-leaf routing (root + smaller children) when LGBM_FUSED_FORCE=1"
tech-stack:
  added: []
  patterns:
    - "one cube per feature, single-owner sequential f64 (bit-exact build) — NO atomics"
    - "build-ALL / scan-SUBSET: complete resident histogram for the subtract trick + gated scan"
key-files:
  created: []
  modified:
    - crates/lgbm-compute/src/kernels/histogram.rs
    - crates/lgbm-compute/src/kernels/split.rs
    - crates/lgbm-compute/src/lib.rs
    - crates/lgbm-treelearner/src/learner.rs
    - crates/lgbm-treelearner/src/resident_pool.rs
    - crates/oracle-harness/tests/kernel_parity.rs
    - crates/oracle-harness/tests/learner_parity.rs
decisions:
  - "Fused kernel gated OFF by default — bench is FLAT-to-NEGATIVE at every band (honesty mandate #5)"
  - "Build-ALL / scan-SUBSET split so the subtraction trick still derives larger children correctly"
metrics:
  duration: "~1 session"
  completed: 2026-06-08
---

# Phase 260608-t3t Plan 01: Fused per-feature build+fix+compact+scan Summary

A GPU-only fused per-feature kernel that collapses a directly-built leaf's
construct(1) + fix(1) + scan(1) = 3 launches into ONE — proven BIT-EXACT to the host
pipeline (histogram + every SplitInfo) and tree-equivalent — but **net FLAT-to-WORSE
on gfx1100, so gated OFF by default and reported honestly.**

## What was built

- **`build_fix_scan_fused_kernel`** (`histogram.rs`, `#[cfg(feature="rocm")] #[cube(launch)]`):
  one cube per feature (`CubeCount::Static(num_features,1,1)`, `CubeDim::new_1d(1)` —
  single-owner ⇒ bit-exact, NO atomics). Per cube: (1) SEQUENTIAL f64 gather→fold in
  ascending leaf-row order (the cpu-anchor order, `f64::cast_from` widen); (2) inline
  FixHistogram (RAW un-bumped sum_g/sum_h seed — Pitfall 2 — verbatim from
  `fix_compact_kernel`); (3) inline compact (offset shift + tail zero, verbatim);
  (4) the SHARED `split_scan_body` over the fixed+compacted region (2*kEpsilon-bumped
  sum_hessian + host `min_gain_shift`), writing the RAW 12-cell SplitInfo. Output BOTH
  the resident fixed+compacted f64 histogram (for the subtraction trick) AND the
  SplitInfos, in ONE launch.

- **`build_fix_scan_resident_f64_on`** host launcher: V5 validation + marshalling
  (mirrors `build_fix_compact_resident_f64_on`) + the host pre-step (2*kEpsilon bump +
  `min_gain_shift`) and decode/accept-gate (mirrors `find_best_splits_fused_inner`).
  Returns `(hist Handle, len, Vec<SplitInfo>)`.

- **Backend seam** `Backend::build_fix_scan_resident`: default typed-error body
  (CpuBackend never reaches it — the gate is off on cpu), RocmBackend impl stores the
  returned Handle into the pool mirror slot (so `subtract_resident` still finds the
  parent) and returns the SplitInfos.

- **Learner wiring** (`learner.rs`): directly-built leaves (root + smaller children) on
  the fused-eligible path skip the standalone `build_resident_leaf_into` and route
  through ONE `build_fix_scan_resident` call inside `scan_leaf_histogram`. Subtract-
  derived larger children + large (atomic resident) + cpu paths UNCHANGED.

- **Size gate** (`resident_pool.rs`): `fused_directly_built_eligible` (same fail-safe
  numeric spine as `resident_eligible`) + `FUSED_MAX_NUM_DATA` + `LGBM_FUSED_FORCE`
  bench knob.

## Key design correction (build-ALL / scan-SUBSET)

The first cut built+scanned only the spine SUBSET (`batched_feats`). That is WRONG for
subtree leaves: when `parent_splittable` gates out some features, the resident
histogram regions for those features would be left zeroed, and the subtract-derived
larger sibling (`larger = parent − smaller` over the WHOLE buffer) would be corrupted.

Fix: the fused kernel BUILDS+fixes+compacts EVERY feature (a COMPLETE resident
histogram, exactly like `build_resident_leaf_into`) but SCANS only the scan-active
subset via a `scan_active` mask; gated-out features leave their 12-cell out window
zeroed (host decodes `is_splittable == 0`). The launcher returns SplitInfos for active
features in scan-active order (== `batched_feats` order). This is what makes the
`learner_parity_fused_equals_host_tree_on_hip` test pass for the full 31-leaf tree
(including subtree nodes with parent-splittability gating).

## Bit-exactness (HARD GATE #2 — MET)

- **`kernel_parity_build_fix_scan_equals_host_on_hip`** (gfx1100): the fused kernel's
  RESIDENT fixed+compacted histogram is BIT-EXACT (`compare_exact_f64_bits`) to the
  host sequential f64 build → `fix_histogram` → host compact; the per-feature 12-cell
  SplitInfo is EXACTLY equal (every field) to `find_best_split_cpu_native` over the same
  region. Covers mfb>0 reconstruct, mfb==0/offset==1 compaction (DEF-07-02 path),
  offset==0 no-op, a REVERSE-winner, a FORWARD-winner, and a no-split feature. GREEN.
- **`learner_parity_fused_equals_host_tree_on_hip`** (gfx1100): the `LGBM_FUSED_FORCE=1`
  tree == the forced-host tree — structural fields BIT-EXACT (topology / split_feature /
  threshold / decision_type / counts), leaf values within ~1e-6 (31 leaves, max leaf
  diff 4.25e-7 — the f32-atomic RAW build is NOT used here; the residual is the same
  ~1e-6 ROCm contributor the resident path has). GREEN.

## BEFORE / AFTER bench — HONEST result (HARD GATE #5)

Measured on the local gfx1100, `bench_train` all three ways via `LGBM_RESIDENT_FORCE` /
`LGBM_FUSED_FORCE`, `train_median` of 5, 2 runs each, NO `memory_usage()` hook:

| rows  | HOST (before) | RESIDENT (before) | FUSED (after) | winner   |
|-------|---------------|-------------------|---------------|----------|
| 2000  | 1.42 / 1.49 s | 1.62 / 1.66 s     | 1.62 / 1.66 s | HOST     |
| 8000  | 4.56 / 4.25 s | 4.74 / 4.88 s     | 5.03 / 5.05 s | HOST     |
| 20000 | 11.82 / 11.97 s | 11.58 / 11.68 s | 12.13 / 12.19 s | RESIDENT |

**Verdict: the fused path is FLAT-to-NEGATIVE at every band. NO net win.**

Launch reduction (reasoned from code, learner.rs:1373-1453 directly-built vs
subtract-derived): per directly-built leaf the resident chain is
`construct_leaf_hist_resident_kernel`(1) + `fix_compact_kernel`(1) +
`find_best_splits_fused_kernel`(1) = 3 launches; fused = 1. Directly-built leaves ≈
root + smaller children ≈ ~half of a num_leaves=31 tree ≈ ~16 leaves/tree × 2 launches
saved ≈ **~32 of ~205 launches/tree ≈ ~15% fewer launches** — and the bench CONFIRMS the
launches drop, but the wall-clock does NOT improve.

Why: the single-owner SEQUENTIAL f64 build (mandatory for bit-exactness — non-negotiable
#2) replaces the resident chain's PARALLEL f32-atomic build. On gfx1100 that sequential
per-feature fold costs MORE than the ~2 launches/leaf it eliminates. The launch saving
does not pay for the lost build parallelism. This is the same "correct-but-flat" class
as kt8/lad/s2b; no win was manufactured.

Per non-negotiable #5: `FUSED_MAX_NUM_DATA = -1` ⇒ `num_data <= FUSED_MAX_NUM_DATA` is
false for every real workload (`num_data >= 1`), so the fused path NEVER auto-engages.
The proven bit-exact kernel + both oracles + the `LGBM_FUSED_FORCE` override stay landed.

## Scope honesty + follow-up (NOT implemented)

This fully fuses DIRECTLY-BUILT leaves only. Subtract-derived larger children still do
subtract(1) + scan(1). A possible T2-follow-up — fuse subtract+scan→1 for derived
larger children — was NOT implemented: it would save ~directly-built-leaf-count more
launches/tree (~the other ~half of 31 leaves), but given THIS plan's measured result
(the ~15% launch cut on directly-built leaves bought no wall-clock), a subtract+scan
fusion is unlikely to net a win on gfx1100 for the same reason (launch count is not the
binding constraint at small/medium once the build is sequential). It is scoped, not
recommended, unless a future profile shows launches are again the dominant cost.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Build-ALL / scan-SUBSET correctness fix**
- **Found during:** Task 2 (the first `learner_parity_fused_equals_host_tree_on_hip` run)
- **Issue:** the initial fused launcher built+scanned only the spine subset; for subtree
  leaves with `parent_splittable` gating that left gated-out feature regions zeroed,
  which would corrupt the subtract-derived larger sibling's histogram.
- **Fix:** the kernel now BUILDS+fixes+compacts ALL features (complete histogram) and
  scans only the `scan_active` subset; a `scan_active: &[bool]` mask was threaded through
  the launcher + Backend seam + learner.
- **Files modified:** histogram.rs, lib.rs, learner.rs
- **Commit:** 5335609 (T2)

**2. [Rule 3 - Blocking] Relaxed an over-strict debug_assert**
- The initial `slot_off`-vs-`feats` positional `debug_assert` tripped when `feats` was a
  subset; once `feats` became the FULL list (build-all), it is positional again. Fixed
  the assert to the correct invariant.

No CpuBackend / fix_histogram / host-compact / host-split-scan edits — the diff is
additive on the GPU path; the CPU bit-exact floor (learner_parity 29, boosting_parity 75)
is byte-unchanged.

## Parity floor (all GREEN modulo documented pre-existing)

- `cargo test -p oracle-harness --test kernel_parity --features rocm`: 14 passed
  (incl. the new fused==host oracle); only the PRE-EXISTING D-03a `split_within_tol`
  f32-gap fails (out of scope, verified on clean HEAD before any edit).
- `cargo test -p oracle-harness --test learner_parity --features rocm`: 30 passed (incl.
  the new fused==host tree test); only the PRE-EXISTING flaky
  `resident_equals_host_tree_on_hip` (~1.26e-6, just over 1e-6) fails — passes pinned
  with `LGBM_RESIDENT_FORCE=1` (noted, NOT chased, per Task 0).
- `cargo test -p oracle-harness --test boosting_parity`: 75 passed (incl.
  `mfb_zero_offset_histogram_contract`, `goss_parity_matrix`).
- `cargo build --workspace` and `cargo build --workspace --features rocm`: both exit 0.
- `git diff`: ZERO hunks in CpuBackend / fix_histogram.rs / host compact / host split-scan.

## Commits

- `99d1910` feat(260608-t3t p1): fused kernel + launcher + Backend seam + fused==host oracle
- `5335609` feat(260608-t3t p2): wire fused for directly-built leaves + build-all/scan-subset + tree-equivalence test
- `2d9338e` perf(260608-t3t p3): gate fused OFF (flat/negative bench) + provenance table

## Self-Check: PASSED
