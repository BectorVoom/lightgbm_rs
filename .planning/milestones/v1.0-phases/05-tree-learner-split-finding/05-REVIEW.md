---
phase: 05-tree-learner-split-finding
reviewed: 2026-06-06T00:00:00Z
depth: standard
files_reviewed: 21
files_reviewed_list:
  - crates/lgbm-compute/src/kernels/split.rs
  - crates/lgbm-compute/src/lib.rs
  - crates/lgbm-model/src/ensemble.rs
  - crates/lgbm-model/src/model_text.rs
  - crates/lgbm-model/src/objective.rs
  - crates/lgbm-model/src/predict.rs
  - crates/lgbm-model/src/tree.rs
  - crates/lgbm-treelearner/src/col_sampler.rs
  - crates/lgbm-treelearner/src/data_partition.rs
  - crates/lgbm-treelearner/src/error.rs
  - crates/lgbm-treelearner/src/fix_histogram.rs
  - crates/lgbm-treelearner/src/histogram_pool.rs
  - crates/lgbm-treelearner/src/leaf_splits.rs
  - crates/lgbm-treelearner/src/learner.rs
  - crates/lgbm-treelearner/src/lib.rs
  - crates/lgbm-treelearner/src/split_info.rs
  - crates/oracle-harness/tests/kernel_parity.rs
  - crates/oracle-harness/tests/learner_parity.rs
  - xtask/cpp/kernel_capture.cpp
  - xtask/cpp/learner_capture.cpp
  - xtask/src/main.rs
findings:
  critical: 2
  warning: 6
  info: 4
  total: 12
status: issues_found
---

# Phase 5: Code Review Report

**Reviewed:** 2026-06-06
**Depth:** standard
**Files Reviewed:** 21
**Status:** issues_found

## Summary

Reviewed the Phase-5 serial tree learner (leaf-wise growth, histogram split
finding, data partition, feature subsampling, force_col_wise) plus its C++
capture/parity harness, and the supporting `lgbm-compute` split kernel and
`lgbm-model` re-used code. All committed unit tests and parity goldens pass
(`cargo test -p lgbm-treelearner`, `oracle-harness` kernel/learner parity), so
the *self-consistent* paths are bit-exact against the captured fixtures.

The adversarial focus is therefore on **what the goldens cannot catch**: the
C++ capture harness is a hand-transcription that uses the *same* conventions as
the Rust port, so any convention error shared by both sides passes the parity
gate while still diverging from real `lib_lightgbm` and/or from the model's own
predict path.

Two such defects are confirmed:

- **CR-01 (proven by a constructed test):** for a `most_freq_bin == 0` feature
  (which the port encodes with `offset == 0` and a *non-compacted* histogram),
  the data-partition's `--th` adjustment is off by one relative to the threshold
  the scan records and stores. The tree's `leaf_count`/leaf outputs are computed
  for a different partition than `predict` actually uses — a silent train/predict
  inconsistency and a numerical-fidelity violation. I reproduced it directly:
  data-partition gives `leaf_count=[4,8]` while predicting the same rows through
  the stored `threshold=2.5` gives `[6,6]`.
- **CR-02:** the `offset` field invariant is documented one way, used the
  opposite way in every corpus, and inverted again in the `real_gh` parser, and
  the only feature path actually validated bit-exact is `most_freq_bin == 0`
  (the broken one), so no faithful oracle exists for `offset == 1`.

The remaining findings are robustness/faithfulness gaps (dead histogram-pool +
subtraction-trick orchestration, panic-capable indexing in the model decoder,
softmax/empty-input edge cases) and documentation/test-coverage debt.

## Critical Issues

### CR-01: most_freq_bin==0 partition is off-by-one vs the stored threshold (train/predict divergence)

**File:** `crates/lgbm-compute/src/kernels/partition.rs:58-65`, coupled with
`crates/lgbm-compute/src/kernels/split.rs:299` and
`crates/lgbm-treelearner/src/learner.rs:961-1008`

**Issue:** The port stores a **non-compacted** histogram (index `i` == bin `i`,
all `2*num_bin` cells, bin 0 included; the most-freq bin is reconstructed by
`fix_histogram`). For a `most_freq_bin == 0` feature the spine sets
`offset == 0`, so the FORWARD scan records `threshold = t + offset = t` meaning
"left = bins `<= threshold`". But `data_partition_kernel` still executes the C++
`if (most_freq_bin == 0) th -= 1;` step (line 59-61), so it routes
`bin > threshold-1 → right`, i.e. left = bins `< threshold`. The two boundaries
disagree on the bin equal to `threshold`.

That `--th` is only correct in real LightGBM because there `offset == 1` when
`most_freq_bin == 0` (`feature_histogram.hpp:1430-1432`,
`dense_bin.hpp:31-34`) — the recorded threshold already accounts for the
reserved most-freq bin. With the port's `offset == 0` + non-compacted layout the
`--th` double-counts the adjustment.

Confirmed by a constructed test on the spine feature 0 (6 bins, 2 rows/bin,
mfb=0): the learner produced `threshold_in_bin = 2`, stored real
`threshold = 2.5`, and data-partition `leaf_count = [4, 8]`, but routing the same
12 rows through the grown tree's own `get_leaf` gave `left = 6, right = 6`. The
leaf the model serializes (count 4, output from 4 rows) is **not** the leaf the
model predicts into (6 rows), so leaf outputs, `leaf_count`, and
`internal_count` are all inconsistent with the tree's own decision boundary — a
silent ≥1e-12 fidelity break.

The parity goldens do not catch it because `learner_capture.cpp::PartitionLeaf`
(lines 406-408) hard-codes the identical `--th`-with-`offset==0` convention, so
both sides agree with each other while both diverge from a predict-consistent
partition; `learner_parity_*` only compares tree *text* (which carries the same
wrong counts on both sides), never train-vs-predict routing.

**Fix:** Make the partition boundary consistent with the stored threshold for the
port's non-compacted layout. The cleanest fix is to NOT apply `--th` when the
port uses `offset == 0` for a `most_freq_bin == 0` feature (since the scan did
not bake in the offset), OR adopt the real LightGBM convention end-to-end
(`offset == 1` + a compacted histogram). Either way, add a parity assertion that
the grown tree's `get_leaf` routing of every training row reproduces the
data-partition `leaf_count` exactly:

```rust
// in learner_parity.rs, after growing each spine/real_gh tree:
let (mut l, mut r) = (0i32, 0i32);
for row in 0..num_data {
    let fv = /* representative real value for f.bins[row] */;
    if tree.get_leaf(&[fv]) == 0 { l += 1 } else { r += 1 }
}
assert_eq!([l, r], [tree.leaf_count[0], tree.leaf_count[1]],
           "data-partition leaf counts must match the stored-threshold routing");
```

This assertion fails today and is the regression test the fix needs.

### CR-02: `FeatureColumn.offset` invariant is documented one way and used the opposite way everywhere (no faithful oracle for offset==1)

**File:** `crates/lgbm-treelearner/src/learner.rs:91-94` (doc) vs
`crates/oracle-harness/tests/learner_parity.rs:234-239, 768` (use), and
`xtask/cpp/learner_capture.cpp` (no offset derivation at all)

**Issue:** The doc comment for `FeatureColumn.offset` states the correct C++ rule
— "1 when `most_freq_bin == 0`, else 0" (matching
`feature_histogram.hpp:1430-1432`). But:

- The spine corpus (`learner_parity.rs:234-239`) and col-sampler corpus set
  `most_freq_bin: 0` with `offset: 0` — the opposite of the documented rule.
- The `real_gh` parser (`learner_parity.rs:768`) sets
  `offset: if most_freq_bin == 0 { 0 } else { 1 }` — inverted relative to the
  documented rule AND relative to LightGBM.
- `learner_capture.cpp` never derives `offset` from `most_freq_bin`; it
  transcribes whatever the corpus hard-codes.

Because both Rust and C++ use the same (non-LightGBM) `offset` value, the parity
gate is self-consistent but validates nothing about real-`lib_lightgbm`
fidelity. The only feature layout exercised bit-exact is `most_freq_bin == 0`,
which is exactly the layout CR-01 shows is mis-partitioned. There is **zero**
bit-exact coverage of the `offset == 1` / `most_freq_bin > 0` scan+partition
path: `learner_parity_missing_routing` uses `most_freq_bin: 1` but only asserts
`total == 8` row conservation, never a C++ golden tree.

**Fix:** Pick ONE offset convention, document it as the authoritative one, derive
`offset` from `most_freq_bin` in a single helper used by both the learner and the
harness (not three contradictory inlined rules), and add a `most_freq_bin > 0`
corpus with a committed C++ reference tree (PTREE) so the `offset==1` path is
validated bit-exact. Until then, treat `most_freq_bin > 0` as unsupported and
reject it with a typed error rather than silently growing an unvalidated tree.

## Warnings

### WR-01: HistogramPool (D-05) is entirely dead in the growth path

**File:** `crates/lgbm-treelearner/src/learner.rs:379-380, 546` (`_pool`)

**Issue:** `train_inner` constructs a `HistogramPool` and calls `reset_map()`,
then passes it to `find_best_splits` as `_pool: &mut HistogramPool` (underscored,
never read). `buffer`/`buffer_mut`/`get`/`move_` are never invoked from the
learner — the whole eviction/LRU/slot-reuse module (which D-05 says is observable
through the subtraction trick) is exercised only by its own unit tests. The
learner rebuilds every leaf's histogram from scratch, so the pool's leaf→slot
assignment never influences a result.

**Fix:** Either wire the pool into the actual smaller/larger histogram buffers
(so the subtraction-trick buffer-reuse decision is real), or remove the pool from
the learner and document D-05 as deferred. Do not keep a fully-mirrored
eviction policy that no production path reads.

### WR-02: Subtraction trick is never used in the gain scan; `subtract_from` is dead

**File:** `crates/lgbm-treelearner/src/learner.rs:727-741`

**Issue:** The comment claims the larger leaf is built "via subtraction when
`use_subtract`", but `find_best_split_for_leaf` always calls
`construct_histograms` directly (line 734) and then `let _ = subtract_from;`
(line 741) discards the smaller-sibling id. The subtraction trick
(`parent − smaller`) — a keystone Phase-5 deliverable (TRL-02, RESEARCH A3) — is
only run in the standalone `learner_parity_subtract` test, never in the real
growth loop. This is faithful *numerically* only because the direct build equals
the subtracted one, but the orchestration claimed in the module docs and
`find_best_splits` doc-comment does not happen.

**Fix:** Actually derive the larger child via `subtract_histograms(parent,
smaller)` in the growth path (using the pool's retained parent buffer per
WR-01), or correct the doc-comments to state the larger child is rebuilt
directly and the subtraction trick is validated separately.

### WR-03: `Tree::categorical_decision` indexes parsed arrays with raw `[]` (panic on malformed cat tree)

**File:** `crates/lgbm-model/src/tree.rs:190-193`

**Issue:** `categorical_decision` does
`cat_idx = self.threshold[node] as i32 as usize; lo = self.cat_boundaries[cat_idx]; hi = self.cat_boundaries[cat_idx + 1];` with no bounds check. The module docstring (lines 30-35) advertises that the port is *stricter* than C++ and "never a panic" on malformed files, but `Tree::parse` validates `cat_threshold` length against `cat_boundaries.back()` only — it does not validate that every internal node's `threshold` (used here as `cat_idx`) is `< cat_boundaries.len() - 1`. A categorical node whose `threshold` encodes an out-of-range `cat_idx` panics here (OOB index), contradicting the T-03-03 no-panic guarantee.

**Fix:** Validate, during `Tree::parse`, that for every categorical-decision node `0 <= cat_idx` and `cat_idx + 1 < cat_boundaries.len()`, returning `ModelError::MalformedModel` otherwise; or bounds-check in `categorical_decision` and route NaN-style to `right_child`.

### WR-04: `objective::convert` / `softmax` panic on empty input slices

**File:** `crates/lgbm-model/src/objective.rs:189-204, 231-237`

**Issue:** `convert` indexes `input[0]`/`output[0]` for Regression/Binary, and
`softmax` reads `input[0]` (line 234) before the loop. `num_output()` returns
`(num_class).max(0) as usize`, which is `0` for a `Multiclass { num_class: 0 }`.
`ObjectiveKind::parse` rejects `num_class < 1`, so the *parsed* path is safe, but
`convert`/`softmax` are `pub` and callable directly with a zero-length slice,
which panics (index OOB / `input[0]` on empty). For a library boundary that
elsewhere insists on typed errors over panics, these public helpers are an
unchecked panic surface.

**Fix:** Either make these `pub(crate)`, or guard `softmax`/`convert` with a
`debug_assert!`-plus-early-return on empty input and document the length
precondition explicitly.

### WR-05: `predict_raw` panics if `num_tree_per_iteration` mis-sized vs `trees.len()`

**File:** `crates/lgbm-model/src/ensemble.rs:90-102`

**Issue:** `predict_raw` computes `idx = i*ntpi + k` and indexes `self.trees[idx]`
directly. `num_iteration()` guards against `ntpi <= 0`, but if a loaded model has
`num_tree_per_iteration` inconsistent with `trees.len()` (e.g. `trees.len()` not
a multiple of `ntpi`, which `load()` never validates), `init_predict` can yield a
range whose `idx` exceeds `trees.len()` and panics. `model_text::load` validates
feature counts and `tree_sizes` but never checks
`trees.len() % num_tree_per_iteration == 0`.

**Fix:** In `load`, assert `num_tree_per_iteration >= 1` and
`trees.len() % num_tree_per_iteration == 0`, returning `MalformedModel`
otherwise; or use `self.trees.get(idx)` and treat a miss as a malformed model.

### WR-06: `find_best_split` kernel's REVERSE `done` flag can diverge from C++ `break` when an early continue precedes a later break

**File:** `crates/lgbm-compute/src/kernels/split.rs:194-220`

**Issue:** The REVERSE/FORWARD scans replace C++ `break` with a sticky `done`
flag and `continue` with the `cont` gate. The kernel comment argues the break
gates are *monotone* so `done` is equivalent to `break`. That holds for the
hessian/count gates on the spine (constant hessians). But the comment's own
monotonicity argument is asserted, not proven, for the general case: a `continue`
(`right_count < min_data_in_leaf`) followed later by accumulation that flips the
`break` gate is handled, but the equivalence relies on `sum_left_hessian` /
`left_count` being monotone in `t`, which is only guaranteed for non-negative
hessians. The committed goldens use hessian ≥ 0 everywhere, so no fixture
exercises a mixed-sign hessian where `done`-vs-`break` could diverge.

**Fix:** Add a fixture/case with a candidate ordering that interleaves
`continue` and `break` conditions non-monotonically (or document that hessians
are contractually non-negative — `score_t` hessians are, but state it and assert
it at the boundary) so the `done == break` equivalence is tested, not assumed.

## Info

### IN-01: `model_text::save` uses non-`%.17g` for `tree_sizes`/counts but relies on byte round-trip only

**File:** `crates/lgbm-model/src/model_text.rs:240-256`

**Issue:** `tree_sizes` is recomputed from the serialized block byte length. This
is correct only if `Tree::to_string()` is byte-identical to what produced the
loaded `tree_sizes`. The round-trip test covers the committed fixture, but any
formatter drift (e.g. a future `%g` change) silently changes `tree_sizes` and the
boundary slicing on the next load. Consider asserting `sum(tree_sizes) ==
region_consumed` on save as a self-check.

### IN-02: `arg_max` starts `best_idx = 0` even when leaf 0 is already split

**File:** `crates/lgbm-treelearner/src/learner.rs:1088-1101`

**Issue:** `arg_max` seeds `best_idx = 0`. Leaf 0's `SplitInfo` is reset to
`none()` (gain `-inf`) after it is split, so it never wins, and `split_gt`
correctly skips `-inf`. This is correct but fragile: it depends on every
non-candidate leaf carrying `gain == -inf`. A future change that leaves a stale
positive gain in slot 0 would silently re-pick it. A `debug_assert` that the
chosen leaf's gain is finite-or-`-inf`-by-design would harden it.

### IN-03: Magic `0.0` gain sentinel in the kernel relies on `min_gain_shift >= 0`

**File:** `crates/lgbm-compute/src/kernels/split.rs:144, 234-237`

**Issue:** The kernel uses `best_gain = 0.0` as the "no winner yet" sentinel
(instead of C++ `-inf`) and justifies it by "valid gains are strictly positive".
That holds because `min_gain_shift = gain_shift + min_gain_to_split >= 0`. This
is a load-bearing invariant buried in a comment; if `min_gain_to_split` were ever
allowed negative (it is validated `>= 0` only implicitly), the sentinel would
admit a 0-gain split. Add an explicit `min_gain_to_split >= 0` boundary check.

### IN-04: `learner_capture.cpp` max_depth handling differs structurally from the Rust gate

**File:** `xtask/cpp/learner_capture.cpp:585-595` vs
`crates/lgbm-treelearner/src/learner.rs:497-528`

**Issue:** The C++ `GrowTree` applies the max_depth cap *after* ArgMax on
`best_leaf` (with a `--split; continue;` re-pick), while the Rust
`before_find_best_split` applies it to `left_leaf`/`right_leaf` *before* the
scan. The two converge on the committed corpora (max_depth = -1, or a single
split) but the control structures are not 1:1, so a corpus that actually
exercises a mid-tree depth cap is not guaranteed to match. Add a max_depth-cap
corpus with a committed reference tree to validate the equivalence.

---

_Reviewed: 2026-06-06_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
