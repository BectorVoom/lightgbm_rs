---
phase: 18
fixed_at: 2026-07-01T00:00:00Z
review_path: .planning/phases/18-on-device-data-partition-tree-mutation-prediction/18-REVIEW.md
iteration: 2
findings_in_scope: 8
fixed: 7
skipped: 0
no_change_needed: 1
status: all_fixed
---

# Phase 18: Code Review Fix Report

**Fixed at:** 2026-07-01
**Source review:** `.planning/phases/18-on-device-data-partition-tree-mutation-prediction/18-REVIEW.md`
**Iteration:** 2

**Summary (all 8 findings — WR-01..WR-04, IN-01..IN-04):**
- Findings in scope: 8 (fix_scope = `all`: 4 Warning + 4 Info)
- Fixed: 7 (WR-01, WR-02, WR-03[doc], WR-04, IN-01, IN-02, IN-03)
- No change needed: 1 (IN-04, optional refactor with no behavioral change)
- Skipped: 0

This is a two-pass record. A **prior fixer pass** (iteration 1) ran against an
earlier, smaller version of this review and already landed four commits on `master`:
`c9a7fd1` (WR-01), `5c068eb` (WR-03 doc note), `8f1d6d1` (IN-01 predict-walk
preconditions), and `1dd083b` (the u16-overflow guard now folded into IN-02's
context). The review was subsequently refreshed to 8 renumbered findings. **This
pass (iteration 2)** applied the four still-outstanding fixes — WR-02, WR-04,
IN-02, IN-03 — and re-verified the prior work matches the refreshed review.

All phase-18 on-device kernels remain OFF behind `LGBM_CUDA_ON_DEVICE` /
`on_device_growth_supported() == false`, so every finding is latent. No numeric
accumulation order or tie-breaking was changed; bit-exactness is preserved.

**Verification (iteration 2, final combined state):**
- `cargo test -p lgbm-compute --lib` → 98 passed, 0 failed, 1 ignored.
- `cargo test -p oracle-harness` predict_parity / partition_parity /
  tree_mutation_parity / kernel_parity → 7 + 3 + 3 + 10 = 23 passed, 0 failed.
  No parity regression.

## Fixed Issues

### WR-01: `HistArena::swap` picks a "fresh" slot that can alias a live leaf's histogram

**Files modified:** `crates/lgbm-compute/src/kernels/histogram_arena.rs`
**Commit:** `c9a7fd1` (landed in the prior pass)
**Outcome:** fixed — verified against the refreshed WR-01 sketch, no further change needed.
**Applied fix:** The naive `(parent_slot + 1) % num_slots` picker is replaced by a scan
over the `leaf_to_slot` occupancy set (`self.leaf_to_slot.values()`), choosing the first
slot that is neither `parent_slot` nor referenced by any live leaf, and returning a typed
`ComputeError::Runtime` when the pool is exhausted. The larger child inherits
`parent_slot`, the smaller child takes the fresh slot, and the consumed `parent_leaf`
entry is dropped from `leaf_to_slot` (it is now an internal node). This matches the
current review's fix sketch exactly (free-slot scan + drop stale parent key + insert
larger→parent_slot / smaller→fresh); no additional change was required this pass.

### WR-02: `validate_walk` does not validate the tree indices its SAFETY comment claims

**Files modified:** `crates/lgbm-compute/src/kernels/predict.rs`
**Commit:** `d0b97ac`
**Outcome:** fixed.
**Applied fix:** Added index-range validation to `validate_walk`: for every node,
`split_feature_inner ∈ [0, num_features)`; each `left_child`/`right_child` is either a
leaf (`~child == -child-1 < leaf_value.len()`) or an internal-node index (`< num_nodes`);
categorical nodes must have `cat_idx + 1 < cat_boundaries_inner.len()`. Separately,
`cat_boundaries_inner` must be monotone non-decreasing with its last word bound
`<= bitset_inner.len()` (the `n` length guard `find_in_bitset` relies on). A malformed
tree now returns a typed `ComputeError` at the SP-4 host boundary instead of an in-kernel
out-of-bounds. The launch SAFETY comment was corrected to enumerate exactly what
`validate_walk` checks.

### WR-03: Phase-18 parity gates compare three copies of the same hand transcription

**Files modified:** `.planning/phases/18-on-device-data-partition-tree-mutation-prediction/18-VALIDATION.md`
**Commit:** `5c068eb` (landed in the prior pass)
**Outcome:** fixed (documentation only — the finding's own guidance is "No code change now").
**Applied fix:** The golden-provenance limitation is documented in 18-VALIDATION.md: the
partition / categorical / predict goldens are hand-written host re-transcriptions of the
C++ source, so the parity cells prove three-way transcription consistency rather than
fidelity to compiled `lib_lightgbm`. The deferred remediation (capture a compiled-library
partition/predict/categorical cross-check on the Kaggle CUDA harness before Phase-21
depends on these kernels) is recorded, cross-referenced to the existing MEMORY note. No
kernel/capture code was changed.

### WR-04: `add_prediction_bagging_on_device` rejects legitimate subset leaf-maps

**Files modified:** `crates/lgbm-compute/src/kernels/predict.rs`
**Commit:** `65cc3dc`
**Outcome:** fixed.
**Applied fix:** In `USE_BAGGING` subset mode the kernel only reads
`data_index_to_leaf[used_indices[i]]`, so validation now checks **only the walked
entries** — un-sampled rows legitimately carry the `-1` sentinel that
`update_data_index_to_leaf_on` writes (`data_partition.rs:841`) and are no longer
rejected. The identity path still validates every entry. A shared `check_leaf` closure
keeps the error text identical across both paths. No numeric change.

## Info

### IN-01: Non-atomic `score[data_index] +=` races if `used_indices` contains duplicates

**Files modified:** `crates/lgbm-compute/src/kernels/predict.rs`
**Commit:** `8f1d6d1` (landed in the prior pass)
**Outcome:** fixed.
**Applied fix:** The uniqueness precondition is documented on both public drivers
(`add_prediction_to_score_on_device` via `validate_walk`, and
`add_prediction_bagging_on_device`), with a debug-only duplicate check (`debug_assert!`
over a `HashSet` insert) on each `used_indices` path. Verified present in this pass;
release builds are unchanged, `score[data_index] += ...` semantics untouched.

### IN-02: `scatter_marked` does real device work only to feed a `debug_assert`

**Files modified:** `crates/lgbm-compute/src/kernels/data_partition.rs`
**Commit:** `d5e936d`
**Outcome:** fixed.
**Applied fix:** The u16 inclusive prefix-sum block in `scatter_marked` feeds only a
`debug_assert_eq!` cross-check of the inclusive/exclusive scan relation, yet release
builds still launched three extra kernels per partition (`n <= 65535`) with the result
discarded. The whole block is now gated behind `#[cfg(debug_assertions)]`, so release
runs never allocate or launch it. The earlier u16-overflow guard (`1dd083b`) is preserved
— this pass addresses the separate release-build launch cost the refreshed review flagged.

### IN-03: `DeviceCudaTree::new` leaves six field buffers uninitialized

**Files modified:** `crates/lgbm-compute/src/kernels/tree.rs`
**Commit:** `c0b488e`
**Outcome:** fixed.
**Applied fix:** `init_tree_kernel` now zeroes all 18 field buffers — adding `split_gain`,
`internal_weight`, `internal_value`, `threshold`, `cat_boundaries`, and
`cat_boundaries_inner` to the 12 it already initialized — so the tree starts fully
defined rather than relying on the implicit write-before-read invariant for never-split
node slots. The two `cat_boundaries*` slabs are `max_leaves + 1` long, so they get their
own `i < cat_n` guard and `launch_init` sizes the grid to the larger slab.

### IN-04: Tree-walk kernel carries ~17 flat array params under a `too_many_arguments` allow

**Files modified:** none
**Commit:** n/a
**Outcome:** no_change_needed.
**Rationale:** The finding is explicitly Optional with "No behavioral change required."
Grouping the per-feature meta arrays into a `#[derive(CubeLaunch)]` struct is an
ergonomics/safety refactor of the launch contract, not a correctness fix, and touching the
17-arg CubeCL launch signature risks incidental churn on an OFF-by-default kernel with no
functional payoff this phase. Deferred as a Phase-21 cleanup candidate.

---

_Fixed: 2026-07-01_
_Fixer: Claude (gsd-code-fixer)_
_Iteration: 2_
