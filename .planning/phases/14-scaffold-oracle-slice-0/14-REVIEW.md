---
phase: 14-scaffold-oracle-slice-0
reviewed: 2026-06-29T00:00:00Z
depth: standard
files_reviewed: 7
files_reviewed_list:
  - crates/lgbm-compute/Cargo.toml
  - crates/lgbm-compute/src/lib.rs
  - crates/lgbm-dataset/src/dataset.rs
  - crates/lgbm-dataset/src/lib.rs
  - crates/lgbm-treelearner/src/data_partition.rs
  - crates/lgbm-treelearner/src/learner.rs
  - crates/oracle-harness/tests/learner_parity.rs
findings:
  critical: 0
  warning: 4
  info: 1
  total: 5
status: issues_found
---

# Phase 14: Code Review Report

**Reviewed:** 2026-06-29
**Depth:** standard
**Files Reviewed:** 7
**Status:** issues_found

## Summary

Phase 14 "scaffold + oracle (Slice 0)" adds an additive `Backend` on-device tree-growth
seam (`grow_tree_on_device` defaulting to `Ok(None)`, `on_device_growth_supported`
defaulting to `false`), a `LeafPartitionLayout` POD payload in `lgbm-dataset`,
`DataPartition::from_payload`, a decide-once `train_inner` routing fork gated on
`on_device_eligible`, and a rocm-gated tie-aware oracle comparator + tests.

**The core phase contract is met.** With `LGBM_CUDA_ON_DEVICE` unset, the default route is
byte-unchanged: `on_device_eligible = backend.on_device_growth_supported() && cuda_on_device_env()`
is computed once in `new`, and because `on_device_growth_supported()` is `false` on every
Slice-0 backend (CpuBackend default + GpuBackend explicit override), the `&&` short-circuits
so `cuda_on_device_env()` is never even evaluated (zero added syscalls), and the
`train_inner` fork block is statically dead — execution falls through unchanged into the
host/resident path. No BLOCKER was found against the byte-unchanged invariant.

The findings below are all latent-risk / test-strength defects that are **dormant in Slice 0**
but will bite when the seam is activated in Slice 1, plus one observation about the new
oracle comparator providing weaker protection than it advertises. They are WARNING-tier
because they degrade robustness/coverage rather than break the current default route.

## Warnings

### WR-01: On-device fork ignores `capture_snapshots`, will silently return empty snapshots/trace when activated

**File:** `crates/lgbm-treelearner/src/learner.rs:704-714`
**Issue:** The fork unconditionally returns `Ok((tree, Vec::new(), ColSamplerTrace::default(), part))`,
discarding the `capture_snapshots` request. The sibling resident gate is explicitly disabled
when snapshots are requested (`resident_eligible(..., capture_snapshots, ...)` ANDs it in),
but the on-device fork has no such guard. When Slice 1 sets
`on_device_growth_supported() == true` and `LGBM_CUDA_ON_DEVICE=1`, a caller via
`train_with_snapshots` / `train_with_col_sampler_trace` (the D-06 / TRL-08 golden-replay
paths, which pass `capture_snapshots = true`) would silently receive empty snapshots and a
default trace, making the golden replays pass against no data instead of failing loudly. This
is the same "never silently ship a divergent result" invariant the rest of the learner
guards. Dormant in Slice 0 (seam returns `None`), but the asymmetry with `resident_eligible`
is a real latent gap.
**Fix:** Gate the fork on the capture flag at the fork site (it cannot be folded into
`on_device_eligible`, which is computed in `new` before the per-call capture flag is known):
```rust
if self.on_device_eligible && !capture_snapshots {
    if let Some((tree, payload)) = self.backend.grow_tree_on_device(
        gradients, hessians, self.num_leaves, self.max_depth,
    )? {
        let part = DataPartition::from_payload(payload);
        return Ok((tree, Vec::new(), ColSamplerTrace::default(), part));
    }
}
```

### WR-02: On-device fork is placed BEFORE the V5 boundary validation block

**File:** `crates/lgbm-treelearner/src/learner.rs:704-714` (fork) vs `:780-834` (V5 validation)
**Issue:** The fork sits ahead of every V5 / threat-mitigation check: the
`hessians.len() == gradients.len()` length check, the `num_leaves >= 1` check, the
relocated once-per-train bin-range gate (`f.bins.first_ge(f.num_bin)`, the T-04-01
memory-safety mitigation for the branchless fold), and the `na_as_missing` deferral.
In Slice 0 the seam returns `Ok(None)` and these run on fall-through, so there is no current
exposure. But when Slice 1 wires a real kernel, the on-device path will receive UNVALIDATED
gradients/hessians/feature bins — re-introducing exactly the out-of-range-bin UB the
relocated gate exists to prevent. Note the seam signature doesn't even forward `features`
yet, so the on-device path cannot perform the bin-range gate itself today.
**Fix:** When Slice 1 activates the seam, either (a) move the validation block above the fork
so both routes share it, or (b) replicate the length/num_leaves/bin-range/na_as_missing gates
inside `grow_tree_on_device` before any device launch. Add a regression test that the
on-device path rejects an out-of-range bin with `BinIndexOutOfRange`.

### WR-03: Tie-aware comparator's `default_left` tie assertion is tautological — it can never fail

**File:** `crates/oracle-harness/tests/learner_parity.rs:2142-2188`
**Issue:** `assert_on_device_tree_matches_cpu_anchor` first calls
`assert_tree_structure_and_leaves`, which asserts the FULL vectors
`candidate.threshold == anchor.threshold`, `leaf_count`, and `internal_count` are bit-exact
(lines 2080, 2083, 2084). Consequently, by the time the `decision_type` tie loop runs,
`same_threshold` (`on_device.threshold[node] == anchor.threshold[node]`) and
`same_child_counts` are ALWAYS `true` for every node. The tie guard
`assert!(same_threshold && same_child_counts, ...)` therefore can never fire. The net effect
is that `assert_on_device_tree_matches_cpu_anchor` is equivalent to the strict comparator
EXCEPT that `default_left` (bit1) is effectively ignored unconditionally: any `default_left`
flip is accepted. For the `missing_type == None` spine corpora the doc treats default_left as
"predict-irrelevant," but a flipped default_left does change predict-time routing for
out-of-range / default-bin feature values, so a genuine wrong-direction kernel bug in Slice 1
would pass this oracle undetected. The "flip on a NON-tie node hard-fails" claim in the
doc-comment is not actually true as written. Also note the comparator is only ever exercised
by the host-fallback test, where the candidate IS the cpu anchor, so it has no real divergence
coverage yet.
**Fix:** Make the tie genuinely conditional rather than guaranteed-true. Either relax the
shared body so `threshold` is compared with the documented near-tie tolerance (so a real
threshold tie can occur and the per-node tie logic becomes meaningful), or assert the
default_left bit STRICTLY in the shared/structural compare and only fall to the tie path on a
proven near-tie input. As written, document explicitly that default_left is unverified in
Slice 0 so the dormant branch is not mistaken for active coverage.

### WR-04: `DataPartition::from_payload` performs no validation or shape-consistency checks

**File:** `crates/lgbm-treelearner/src/data_partition.rs:74-81`
**Issue:** `from_payload` is a raw field move from the fully-public `LeafPartitionLayout`
POD with no invariants checked: it does not verify `indices.len() == num_data`, that
`leaf_begin`/`leaf_count` have matching lengths, that they are sized to the learner's
`num_leaves`, or that `leaf_begin[l] + leaf_count[l] <= indices.len()`. Downstream accessors
index by leaf id and slice `indices[begin..begin+count]` (`indices_in_leaf`, `leaf_count`,
`split`), so a malformed payload yields silent row corruption or an out-of-bounds panic at
the V5 boundary the rest of the crate is careful to guard. The constructed `DataPartition`
also sizes `leaf_begin`/`leaf_count` to the payload's vector lengths rather than `num_leaves`
(unlike `DataPartition::new`), so a caller iterating leaves up to `num_leaves-1` could panic.
Reachable but unexercised in Slice 0 (seam returns `None`); becomes live in Slice 1.
**Fix:** Validate at the boundary and return `Result` rather than panicking, e.g.:
```rust
pub fn from_payload(p: LeafPartitionLayout, num_leaves: i32) -> Result<Self, ComputeError> {
    if p.indices.len() != p.num_data.max(0) as usize { /* LengthMismatch */ }
    if p.leaf_begin.len() != p.leaf_count.len() { /* LengthMismatch */ }
    // optional: verify per-leaf [begin, begin+count) ranges lie within indices
    Ok(Self { num_data: p.num_data, indices: p.indices,
              leaf_begin: p.leaf_begin, leaf_count: p.leaf_count })
}
```

## Info

### IN-01: `LeafPartitionLayout` exposes all fields `pub` with no constructor or invariants

**File:** `crates/lgbm-dataset/src/dataset.rs:87-97`
**Issue:** The payload is a bare `pub` struct (intentionally, per the D-03 Option-A
lower-crate-payload design). Combined with WR-04 there is no single place that establishes
its internal consistency (`indices.len() == num_data`, `leaf_begin.len() == leaf_count.len()`,
per-leaf ranges in bounds). For a struct that crosses a crate boundary specifically to be
reconstructed into a `DataPartition`, a checked constructor (or at least documented field
invariants the producer must uphold) would localize the contract instead of deferring all of
it to `from_payload`.
**Fix:** Add a doc-stated invariant block on the struct and/or a `LeafPartitionLayout::new(..)`
checked constructor that the Slice-1 kernel producer is required to use.

---

_Reviewed: 2026-06-29_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
