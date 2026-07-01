---
phase: 18-on-device-data-partition-tree-mutation-prediction
reviewed: 2026-07-01T13:04:27Z
depth: standard
files_reviewed: 14
files_reviewed_list:
  - crates/lgbm-compute/src/kernels/data_partition.rs
  - crates/lgbm-compute/src/kernels/histogram_arena.rs
  - crates/lgbm-compute/src/kernels/mod.rs
  - crates/lgbm-compute/src/kernels/predict.rs
  - crates/lgbm-compute/src/kernels/primitives.rs
  - crates/lgbm-compute/src/kernels/tree.rs
  - crates/oracle-harness/tests/fixtures/kernels/partition.txt
  - crates/oracle-harness/tests/fixtures/kernels/predict.txt
  - crates/oracle-harness/tests/kernel_parity.rs
  - crates/oracle-harness/tests/partition_parity.rs
  - crates/oracle-harness/tests/predict_parity.rs
  - crates/oracle-harness/tests/tree_mutation_parity.rs
  - xtask/cpp/kernel_capture.cpp
  - xtask/src/main.rs
findings:
  critical: 0
  warning: 4
  info: 4
  total: 8
status: issues_found
---

# Phase 18: Code Review Report

**Reviewed:** 2026-07-01
**Depth:** standard
**Files Reviewed:** 14
**Status:** issues_found

## Summary

Phase 18 delivers the on-device `mark → prefix-sum → scatter` data partition
(`data_partition.rs`), the flat `CUDATree` mutation kernels (`tree.rs`), the
tree-walk prediction kernel (`predict.rs`), and the cross-tree histogram-pool
SWAP on `HistArena` (`histogram_arena.rs`), all anchored to the cubecl-cpu f64
fold with golden parity replay.

The numerically load-bearing logic is, to the extent it can be verified against
the committed transcriptions, **correct**. I traced the full `route_to_left`
flag fan-out (numeric + categorical) against the C++ `SplitRouteFanout` /
`SplitCategoricalRoute` generator branch-by-branch — including the degenerate
`min_bin == max_bin` branch and every missing/NA/MFB-coincidence sub-branch — and
against the plain-Rust host mirror; the three agree. The `split_inner_scatter_kernel`
derives a correct stable partition (`rank`/`rights_before` yield a permutation of
`[0,n)` with left-then-right ordering, `dest < n`), the exclusive/inclusive scan
relation (Pitfall 2) holds, and the predict remap + numeric/categorical routing +
`~node` leaf decode line up across kernel, host, and golden. No correctness
BLOCKER was proven.

The findings concern (1) a latent slot-aliasing defect in `HistArena::swap` that
is not yet reachable but will corrupt histograms once the Phase-21 multi-leaf
grow loop wires it, (2) validation gaps where SAFETY comments claim inputs are
"validated" that in fact are not, (3) the structural weakness that every parity
gate in this phase compares three re-transcriptions of the same hand port rather
than compiled `lib_lightgbm`, and (4) an over-strict bagging validator.

## Warnings

### WR-01: `HistArena::swap` picks a "fresh" slot that can alias a live leaf's histogram

**File:** `crates/lgbm-compute/src/kernels/histogram_arena.rs:336-381` (same pattern in `rotate`, `:243-274`)
**Issue:** `swap` chooses the smaller child's slot as
`fresh = (parent_slot + 1) % num_slots` and only asserts `fresh != parent_slot`.
It never checks whether that slot is currently owned by *another* live leaf in
`leaf_to_slot`. In a whole-tree grow loop with more than two concurrently-live
leaves (the Phase-21 use case this function exists for), `(parent_slot + 1)` will
frequently land on a slot already holding a different leaf's histogram; the
smaller child then rebuilds into it and silently overwrites that leaf's data. The
doc comment's promise — "the smaller child takes a FRESH non-aliasing slot" — is
false in general (it is non-aliasing only relative to the parent/larger). No
production caller exercises `swap` yet (tests only), so no data loss ships today,
but this is a latent correctness defect that becomes a BLOCKER the moment the
multi-leaf grow loop lands. Separately, the consumed `parent_leaf → parent_slot`
entry is left stale in `leaf_to_slot` (both `parent_leaf` and `larger_leaf` map to
`parent_slot`), so any later `leaf_slot(parent_leaf)` returns a slot now owned by
`larger_leaf`.
**Fix:** Allocate the smaller child from a genuinely-unused slot and drop the
consumed parent entry:
```rust
let occupied: std::collections::HashSet<usize> = self.leaf_to_slot.values().copied().collect();
let fresh = (0..self.num_slots)
    .find(|s| *s != parent_slot && !occupied.contains(s))
    .ok_or_else(|| ComputeError::Runtime { detail: "HistArena::swap: no free slot".into() })?;
self.leaf_to_slot.remove(&parent_leaf);
self.leaf_to_slot.insert(larger_leaf, parent_slot);
self.leaf_to_slot.insert(smaller_leaf, fresh);
```

### WR-02: `validate_walk` does not validate the tree indices its SAFETY comment claims are "validated"

**File:** `crates/lgbm-compute/src/kernels/predict.rs:411-482` (validator), `:282-286` (SAFETY comment)
**Issue:** The launch SAFETY comment asserts "the walk only reads valid
node/feature/leaf indices from the fixture tree (validated)." But `validate_walk`
only checks array *lengths*, `num_rows <= num_data`, and `used_indices <
num_data`. It never checks that `split_feature_inner[node] < num_features`, that
`left_child`/`right_child` leaf decodes (`~node`) fall within `leaf_value.len()`,
or that a categorical `cat_idx = threshold_in_bin[node]` and its
`cat_boundaries_inner[cat_idx+1]` (`end`) are within `bitset_inner.len()`. A
malformed tree therefore indexes out of bounds. Because the kernel uses the
checked `::launch`, this is a runtime panic rather than UB, but (a) the SAFETY
justification is inaccurate, and (b) the SP-4 boundary the module advertises
(typed `ComputeError`, never a panic) is not enforced for tree structure.
`find_in_bitset(bitset_inner, end, …)` using the logical boundary `end` as the
array-length bound is only OOB-safe while `end <= bitset_inner.len()`, which is
unchecked.
**Fix:** Add index-range validation to `validate_walk`:
```rust
for (n, &sf) in tree.split_feature_inner.iter().enumerate() {
    if sf < 0 || sf as usize >= nf {
        return Err(ComputeError::Runtime { detail: format!("node {n}: split_feature_inner {sf} out of range") });
    }
}
// also: child leaf decodes within leaf_value.len(); cat_boundaries_inner monotone and last <= bitset_inner.len()
```
and correct the SAFETY comment to state what is actually checked.

### WR-03: Every Phase-18 parity gate compares three copies of the same hand transcription, not `lib_lightgbm`

**File:** `xtask/cpp/kernel_capture.cpp:490-621, 1437-1593`; consumed by `partition_parity.rs`, `predict_parity.rs`, `tree_mutation_parity.rs`
**Issue:** The golden generator (`SplitRouteFanout`, `SplitCategoricalRoute`,
`PredWalk`, the packet/tree builders) is an explicit hand transcription of the
reference — the file header states the real `dense_bin.hpp`/`cuda_tree.cu` cannot
be compiled here (`external_libs/` unvendored). The Rust device kernel
(`route_to_left`, `add_prediction_to_score_kernel`) and the Rust host anchor
(`route_left_host`, `host_split_mirror`) independently mirror the *same*
transcription. The parity tests therefore prove internal three-way consistency,
not fidelity to compiled LightGBM 4.6 — matching the project memory note
"on-device-kernel-goldens-are-re-transcriptions." Two transcription decisions are
plausible but unverified against the real source: the categorical default
direction tests membership of the **raw** `most_freq_bin`
(`kernel_capture.cpp:604`, mirrored at `data_partition.rs:163-165`) while in-range
rows test the **local** `bin-min+offset`; and the numeric predict missing check
runs on the **remapped** bin (`predict.rs:130-131`, `kernel_capture.cpp:1462`).
For a project whose core contract is ~1e-6 fidelity to C++, a green gate that
cannot detect a shared transcription error is a real weakness.
**Fix:** Track this as an explicit fidelity gap and schedule a
compiled-`lib_lightgbm` cross-check (Kaggle/nvcc or a source build) for at least
one partition, one predict, and one categorical case before Phase-21 depends on
these kernels. No code change now, but do not treat the current gates as fidelity
proof.

### WR-04: `add_prediction_bagging_on_device` rejects legitimate subset leaf-maps

**File:** `crates/lgbm-compute/src/kernels/predict.rs:352-362`
**Issue:** The validator requires *every* `data_index_to_leaf[di]` in `[0,
num_data)` to satisfy `0 <= leaf < num_leaves`, even in `USE_BAGGING` subset mode
where the kernel only reads `data_index_to_leaf[used_indices[i]]`. A realistic
bagged map that leaves un-sampled rows at the sentinel `-1` — the exact value
`update_data_index_to_leaf_on` initializes non-leaf rows to (`data_partition.rs:841`)
— is rejected as out-of-range even though those entries are never read. This makes
the bagging driver incompatible with the `-1`-initialized leaf map produced
elsewhere in the same module.
**Fix:** In subset mode validate only the walked indices:
```rust
match used_indices {
    Some(idx) => for &di in idx {
        let leaf = data_index_to_leaf[di as usize];
        if leaf < 0 || leaf as usize >= leaf_value.len() { /* typed error */ }
    },
    None => { /* validate all, as today */ }
}
```

## Info

### IN-01: Non-atomic `score[data_index] +=` races if `used_indices` contains duplicates

**File:** `crates/lgbm-compute/src/kernels/predict.rs:139, 159-161`
**Issue:** Both prediction kernels do a non-atomic `score[data_index] +=`. The
identity path has unique `data_index == i`, but the subset path never validates
that `used_indices` is duplicate-free, so a duplicate would produce two concurrent
read-modify-write ops to one cell on a real GPU. Bagging subsets are unique in
practice, so this is latent.
**Fix:** Document the uniqueness precondition and, if cheap, assert it in the
subset validators.

### IN-02: `scatter_marked` does real device work only to feed a `debug_assert`

**File:** `crates/lgbm-compute/src/kernels/data_partition.rs:604-613`
**Issue:** For every device partition with `n <= 65535`, the code runs a full
three-launch u16 inclusive prefix-sum whose only consumer is a `debug_assert_eq!`
cross-check. In release builds the result is discarded; in debug builds it is
three extra kernel launches per partition purely for a sanity check.
**Fix:** Gate the u16 block behind `#[cfg(debug_assertions)]` so release builds
never allocate/launch it.

### IN-03: `DeviceCudaTree::new` leaves six field buffers uninitialized

**File:** `crates/lgbm-compute/src/kernels/tree.rs:95-126, 566-585`
**Issue:** `init_tree_kernel` initializes 12 of the 18 field buffers;
`split_gain`, `internal_weight`, `internal_value`, `threshold`, `cat_boundaries`,
and `cat_boundaries_inner` retain uninitialized `client.empty` memory. This is
safe today only because every node slot is written by a `split` before
`to_host_tree` reads it (an implicit write-before-read invariant); a future read
of a never-split node index would surface garbage.
**Fix:** Extend `init_tree_kernel` to zero the remaining node buffers (cheap
`max_leaves`-sized arrays) so the tree has a fully-defined initial state.

### IN-04: Tree-walk kernel carries ~17 flat array params under a `too_many_arguments` allow

**File:** `crates/lgbm-compute/src/kernels/predict.rs:63-141`
**Issue:** The kernel and host driver thread ~17 parallel array arguments through
a macro with `#[allow(clippy::too_many_arguments, unused_assignments)]`.
Understandable given the CubeCL launch contract, but the flat list of same-typed
`&Array<i32>` meta arrays is an easy place to transpose two arguments at a future
call site with no compiler help.
**Fix:** Optional — group the per-feature meta arrays into a `#[derive(CubeLaunch)]`
struct if CubeCL supports it, to make the launch positionally safe. No behavioral
change required.

---

_Reviewed: 2026-07-01_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
