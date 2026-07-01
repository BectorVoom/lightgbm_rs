---
phase: 18-on-device-data-partition-tree-mutation-prediction
reviewed: 2026-07-01T13:04:27Z
depth: standard
files_reviewed: 11
files_reviewed_list:
  - crates/lgbm-compute/src/kernels/data_partition.rs
  - crates/lgbm-compute/src/kernels/tree.rs
  - crates/lgbm-compute/src/kernels/predict.rs
  - crates/lgbm-compute/src/kernels/primitives.rs
  - crates/lgbm-compute/src/kernels/histogram_arena.rs
  - crates/lgbm-compute/src/kernels/mod.rs
  - crates/oracle-harness/tests/partition_parity.rs
  - crates/oracle-harness/tests/tree_mutation_parity.rs
  - crates/oracle-harness/tests/predict_parity.rs
  - crates/oracle-harness/tests/kernel_parity.rs
  - xtask/cpp/kernel_capture.cpp
findings:
  critical: 0
  warning: 1
  info: 4
  total: 5
status: issues_found
---

# Phase 18: Code Review Report

**Reviewed:** 2026-07-01T13:04:27Z
**Depth:** standard
**Files Reviewed:** 11
**Status:** issues_found

## Summary

Reviewed the phase-18 on-device data-partition / tree-mutation / prediction kernels and
their parity scaffolds. The production kernels are unusually careful: the numeric route
fan-out (`route_to_left` / `route_left_host`) and its C++ mirror (`SplitRouteFanout`) were
cross-checked branch-by-branch and agree exactly; the `mark → prefix-sum → scatter` scatter
math (`excl` rank + `i - rank` rights-before) is index-correct with a proven `dest ∈ [0,n)`
permutation and no under/overflow; NaN→0 uses `f64::is_nan` (the cubecl-cpu `x!=x`→false
trap is correctly avoided); `SplitKernel`'s 15-thread disjoint-cell fan-out has no
read-after-write hazard; the prefix-sum block-scan/exclusive/inclusive derivations are
correct and `scan_block_size` provably keeps `num_blocks ≤ 1024`; all device buffers are
pre-allocated once (D-15) and every kernel guards `i < n`. Host-boundary validation
(`validate_partition`, `validate_walk`, `check_split`) rejects the obvious bad inputs.

No correctness or parity BLOCKER was found in any path that is wired live (all phase-18
kernels are OFF behind `LGBM_CUDA_ON_DEVICE` and `on_device_growth_supported()` stays
`false`). The one WARNING is a latent defect in `HistArena::swap` that will misbehave once
Phase-21 drives it with a real multi-leaf tree. The INFOs concern golden provenance and a
few defensive-validation gaps.

## Warnings

### WR-01: `HistArena::swap` fresh-slot picker can alias a live sibling leaf's histogram slot

**File:** `crates/lgbm-compute/src/kernels/histogram_arena.rs:365`
**Issue:** `swap()` selects the smaller child's buffer with
`let fresh = (parent_slot + 1) % self.num_slots;` and asserts only `fresh != parent_slot`.
It never consults the `leaf_to_slot` occupancy map, so in a real breadth-first grow loop
with more than two live leaves the "fresh" slot can already hold another live leaf's
histogram. `swap` then does `self.leaf_to_slot.insert(smaller_leaf, fresh)` and the smaller
child rebuilds directly into that slot (`swap_subtract_lands_in_larger_slot` shows the
build/subtract writing the slot), silently corrupting the aliased leaf's histogram. The
reference `SplitTreeStructureKernel` (`cuda_data_partition.cu:827-906`) draws the smaller
child's buffer from a tracked free-buffer pool precisely to avoid this. The module doc bills
this as "the `SplitTreeStructureKernel` whole-pool swap" and the leaf→slot table implies
multi-leaf support, so the naive `(parent+1)%num_slots` is a genuine correctness gap, not
just a demo simplification. It is latent today (no live consumer:
`on_device_growth_supported()` is `false`), but Phase-21 is the documented consumer and will
hit it.
**Fix:** Pick `fresh` from a set of slots not currently referenced by `leaf_to_slot`
(a free-slot list or a scan for an unused index), erroring when the pool is exhausted; and
drop the now-internal `parent_leaf` key from `leaf_to_slot` after the split. Minimal sketch:
```rust
let occupied: std::collections::HashSet<usize> = self.leaf_to_slot.values().copied().collect();
let fresh = (0..self.num_slots)
    .find(|s| *s != parent_slot && !occupied.contains(s))
    .ok_or_else(|| ComputeError::Runtime {
        detail: "HistArena::swap: no free non-aliasing slot for the smaller child".into(),
    })?;
// ... assign larger→parent_slot, smaller→fresh ...
self.leaf_to_slot.remove(&parent_leaf); // parent is now an internal node
```

## Info

### IN-01: Partition/tree/predict goldens are self-authored transcriptions, not compiled-reference captures

**File:** `xtask/cpp/kernel_capture.cpp:512` (also `:585`, `:598`, `:1437`)
**Issue:** The partition (`SplitRouteFanout`), categorical (`SplitCategoricalRoute` /
`FindInBitsetHost`) and predict (`PredWalk`) goldens are produced by hand-written host
re-implementations of `dense_bin.hpp` / `cuda_tree.cu`, not by invoking the compiled
reference (`lib_lightgbm` / the AMD-fork CUDA path). The Rust kernels
(`route_to_left`, `route_left_host`, the predict walk) are independent transcriptions of the
*same* C++ source lines. The parity cells therefore prove the two transcriptions agree —
not that either matches the compiled reference. A transcription error shared by both (e.g.
the remapped-`bin`-vs-raw-`default_bin` missing comparison at `predict.rs:130` /
`kernel_capture.cpp:1462`) would pass green. This differs from the project's binning /
serial-tree-learner goldens, which are captured from real `lib_lightgbm` 4.6. Worth noting
because the numerical-fidelity contract in CLAUDE.md is stated against the *compiled*
reference.
**Fix:** Document this provenance limitation in the phase artifacts (or 18-VALIDATION.md),
and where the AMD-fork CUDA path can be built on the Kaggle CUDA harness, capture at least
one PORDER/predict golden from the compiled kernel to pin the transcription.

### IN-02: `validate_walk` / bagging validation do not reject duplicate `used_indices`

**File:** `crates/lgbm-compute/src/kernels/predict.rs:472` (and `:363`)
**Issue:** `add_prediction_to_score_kernel` and `add_prediction_bagging_kernel` do
`score[data_index] += ...`. Uniqueness of `used_indices` is assumed but never checked. If a
caller passes duplicate indices, two units accumulate into the same `score` cell — a data
race on a real GPU backend and a double-count everywhere. Reference bagging indices are
unique so this is an unusual input, but the host boundary is billed as the SP-4/T-18-06
validation seam and silently accepts it.
**Fix:** Either document the uniqueness precondition explicitly on the public drivers, or add
a debug-mode duplicate check in `validate_walk` / the bagging validator.

### IN-03: Identity walk silently accepts `num_rows < num_data` and scores only a prefix

**File:** `crates/lgbm-compute/src/kernels/predict.rs:428`
**Issue:** With `used_indices == None`, `data_index = i` and only rows `[0, num_rows)` are
walked; `num_rows < num_data` is accepted (`num_rows > num_data` is the only rejection). The
reference `USE_INDICES=false` semantics are `num_rows == num_data`. A caller that
under-supplies `num_rows` gets a partially-scored accumulator with no error. Harmless for the
current parity tests (which pass `num_rows == num_data` on the identity path) but a latent
foot-gun.
**Fix:** On the `None` path, require `num_rows == num_data` (or document that the identity
path intentionally scores a prefix).

### IN-04: `prefix_sum_inclusive_u16_on` has no overflow guard for general callers

**File:** `crates/lgbm-compute/src/kernels/primitives.rs:482`
**Issue:** The u16 inclusive scan accumulates in `u16`; an inclusive total > 65535 wraps
silently. The phase-18 usage is safe — in `scatter_marked` it is a debug-only cross-check
gated by `if n <= u16::MAX` over 0/1 marks (`data_partition.rs:605`) — but the public
primitive itself neither guards nor documents the overflow ceiling, so a future non-partition
caller could get silently wrong results.
**Fix:** Document the `sum ≤ u16::MAX` precondition on the function, or add a debug assertion
on the scanned tail.

---

_Reviewed: 2026-07-01T13:04:27Z_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
