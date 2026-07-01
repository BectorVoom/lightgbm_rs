---
phase: 17-on-device-best-split-finder
reviewed: 2026-07-01T03:22:53Z
depth: standard
files_reviewed: 3
files_reviewed_list:
  - crates/lgbm-compute/src/kernels/best_split.rs
  - crates/lgbm-compute/src/gain.rs
  - crates/lgbm-compute/src/kernels/mod.rs
findings:
  critical: 0
  warning: 3
  info: 3
  total: 6
status: issues-found
---

# Phase 17: Code Review Report

**Reviewed:** 2026-07-01T03:22:53Z
**Depth:** standard
**Files Reviewed:** 3
**Status:** issues_found

## Summary

Reviewed the on-device best-split finder (`best_split.rs`, ~2600 lines: stage-1/2/3
pipeline, f64 anchor + f32 mirror + gpu-gated block/globalmem kernels), the net-new
`USE_SMOOTHING` gain path (`gain.rs`), and the module wiring (`mod.rs`). I cross-checked
the numerical core line-by-line against the C++ reference
(`LightGBM/src/treelearner/cuda/cuda_best_split_finder.cu:146-320` and
`cuda_leaf_splits.hpp:60-140`).

**Overall the transcription is high-quality and faithful for the paths the fixture
exercises.** The count-recovery round-ties-even divergence (Pitfall 1), the two-phase
kEpsilon placement (Pitfall 2), complement-from-parent (Pitfall 4), and the strict-`>`
argmax tie-break (Pitfall 5) all match the C++ exactly, and the `USE_SMOOTHING`
form-(B)/form-(D) blend in `gain.rs` is verbatim-correct against `cuda_leaf_splits.hpp`.
The USE_RAND `NextInt(0, num_bin-2)` draw (`draw[0] % (num_bin_i - 2)`) matches the C++
truncated-modulo semantics.

**No currently-executing (BLOCKER) defect was found** — every path in this file is behind
the OFF-by-default `LGBM_CUDA_ON_DEVICE` seam and `Backend::on_device_growth_supported`
stays `false`, so nothing here reaches a shipping code path. The findings below are (a) a
genuine but **dormant** correctness divergence in the `na_as_missing` (NaN-feature)
branch that is undocumented and silent, (b) a parity-harness gap that means the Pitfall-3
`default_left != reverse` behavior is not actually validated through the kernel, and (c)
minor quality/doc issues.

## Warnings

### WR-01: `na_as_missing` (NaN-feature) branch is unimplemented and silently divergent in the f64 anchor and the block kernel

**File:** `crates/lgbm-compute/src/kernels/best_split.rs:419-467` (`split_eval_body`), `:1410-1447` (`split_eval_block_kernel_f32`)
**Issue:** The C++ reference has two `na_as_missing`-specific branches that the Rust
transcription omits:
1. **Reverse lower-bound gate.** C++ gates both the read (`cu:187-188`) and the candidate
   (`cu:213`) on `threadIdx_x >= static_cast<unsigned int>(task->na_as_missing)`. For a
   NaN feature the reverse scan must therefore **exclude the top (NaN/missing) bin** from
   the accumulation. The Rust `split_eval_body` uses only `read_active = t < fnbmo && !skip`
   (line 425) and `in_range = (rev && t <= rev_end) ...` (line 460) — with no `t >= na_as_missing`
   lower bound. At `t == 0` the Rust reverse path reads `bin = fnbmo-1-0 = fnbmo-1` (the NaN
   bin) and folds it into the prefix sum, whereas C++ skips it. The scanned side then
   diverges for every reverse candidate of a NaN feature.
2. **Forward `mfb_offset == 1` special reduction.** C++ (`cu:173-204`, `:236`, `:254-256`)
   reads bins `threadIdx_x-1`, `ShuffleReduceSum`s the non-default bins, folds the default
   bin into thread 0, uses `end = num_bin-2`, and records `threshold_value = threadIdx_x`
   (not `+ mfb_offset`). None of this exists in the Rust body.

`build_split_find_tasks` DOES emit `na_as_missing=true` tasks for every `MissingType::NaN`
feature (lines 197-235), so a NaN feature would drive this divergent path. Unlike the
`split_eval_globalmem_kernel_f32` which explicitly documents the same gap as a "known
limitation, not a silent stub" (lines 1582-1587), the anchor `split_eval_body` and the
block kernel carry **no doc caveat, no `debug_assert`, and no error** — a caller passing
`na_as_missing=true` gets silently wrong splits. All 11 golden fixture cases use
`na_as_missing=0`, so this is entirely untested. Dormant today (gated off), but it breaks
the "bit-exact f64 anchor" contract (CLAUDE.md) the moment on-device growth is wired, and
NaN features are common in real data.
**Fix:** Either implement the two `na_as_missing` branches faithfully (mirror `cu:173-204`
+ the `>= na_as_missing` gate on the reverse read/candidate), or — at minimum for this
milestone — reject/guard the path so it cannot silently mis-evaluate:
```rust
// in find_best_splits_stage1_on, before launch:
if task.na_as_missing {
    return Err(ComputeError::Runtime {
        detail: "stage1: na_as_missing (NaN-feature) path not yet implemented".into(),
    });
}
```
and add a golden fixture case with `na_as_missing=1` (both fwd-mfb1 and reverse) before
un-gating on-device growth.

### WR-02: parity harness ties `reverse == assume_out_default_left`, so the Pitfall-3 decoupling is never validated through the kernel

**File:** `crates/oracle-harness/tests/best_split_parity.rs:221-236` (`task_of`)
**Issue:** `task_of` builds the stage-1 task with `reverse: g.win_default_left` **and**
`assume_out_default_left: g.win_default_left` — both derived from the same golden field.
Every fixture case therefore has `reverse == assume_out_default_left`. The stage-1 kernel
writes `default_left = assume_out_default_left` verbatim (the D-01 Pitfall-3 landmine,
`best_split.rs:515`), but because the two are always equal in the replay, a regression
that erroneously wrote `default_left = reverse` instead would still pass the golden test.
The very decoupling this phase exists to protect (`default_left != reverse`) is only
covered by the `build_split_find_tasks` unit test (`assume_out_default_left_table`), never
end-to-end through `split_eval_body`. It also means no golden drives a reverse task whose
`assume_out_default_left` is `false` (the `num_bin<=2 NaN` row).
**Fix:** Add a golden whose `reverse` and `assume_out_default_left` differ (e.g. a
`num_bin<=2` NaN reverse task, `reverse=true`, `assume=false`) and thread both fields
independently through `task_of` (carry a distinct `reverse` column in the SCASE record
rather than reusing `win_default_left`).

### WR-03: `find_best_splits_stage1_on` reads `scalars.is_larger` nowhere — the stage-1 launcher cannot honor the IS_LARGER record-base duality it documents

**File:** `crates/lgbm-compute/src/kernels/best_split.rs:279-280` (field), `:632-728` (launcher)
**Issue:** `Stage1Scalars.is_larger` is documented (line 279) as the "IS_LARGER task-index
base selector (smaller → `[t]`, larger → `[t+num_tasks]`)", but the stage-1 launcher
`find_best_splits_stage1_on` (and its f32/globalmem siblings) never reads it — the smaller/
larger record placement is entirely a stage-2 concern (`sync_best_split_for_leaf_on`'s
`is_smaller` param). The field is dead in the only paths that consume `Stage1Scalars`. This
is more than cosmetic: the parity harness populates it from the fixture's `is_larger`
column (`scalars_of`, `best_split_parity.rs:243`), and the `default_fwd_larger` /
`default_rev_larger` goldens exist specifically to exercise the larger-leaf case — but
since stage-1 ignores `is_larger`, those "larger" goldens are byte-identical runs of the
"smaller" ones, giving false confidence that IS_LARGER is covered. Confirm whether the C++
stage-1 kernel actually depends on IS_LARGER (it selects which `CUDALeafSplitsStruct` to
read); if so, the leaf totals in `Stage1Scalars` must be selected by `is_larger` upstream
and the field's presence here is a latent trap.
**Fix:** Either remove `is_larger` from `Stage1Scalars` (and drop the meaningless
"_larger" goldens or fold them into stage-2 where the duality lives), or, if the larger
leaf is meant to supply different `sum_gradient`/`sum_hessian`/`parent_*` totals, document
that the caller must populate those per-leaf and assert the "_larger" goldens differ from
their "_smaller" counterparts.

## Info

### IN-01: stale `gain.rs` module doc contradicts the smoothing path this phase added

**File:** `crates/lgbm-compute/src/gain.rs:27-32`
**Issue:** The "## Scope" doc still states "Only the `USE_L1=false/true`, `USE_MAX_OUTPUT=false`,
`USE_SMOOTHING=false`, `USE_MC=false` ... instantiation is transcribed" and that
"`path_smooth` (smoothing) ... are Phase-7+ scope and are validated to be at their no-op
defaults by the launcher." Phase 17 added the full `USE_SMOOTHING` form-(B)/form-(D) path
(`calculate_splitted_leaf_output_smoothed`, `get_leaf_gain_smoothed`, and the `*_f32`
mirrors) in this same file, so the scope note is now false and misleading for maintainers.
**Fix:** Update the "## Scope" paragraph to note that `USE_SMOOTHING` is now transcribed
(cuda_leaf_splits.hpp:74-122) and consumed by the stage-1 body via the runtime
`use_smoothing` select; drop the "validated no-op by the launcher" claim for `path_smooth`.

### IN-02: USE_RAND draw diverges from C++ for `num_bin <= 2` (defensive, but a silent parity gap)

**File:** `crates/lgbm-compute/src/kernels/best_split.rs:650-655`, `:962-967`, `:1827-1832`
**Issue:** The Rust guards the draw with `scalars.use_rand && num_bin_i - 2 > 0`, returning
`rand_threshold = -1` when `num_bin <= 2`. C++ (`cu:158-161`) guards the *inner* draw the
same way (`if (task->num_bin - 2 > 0)`) but leaves the shared `rand_threshold`
**uninitialized** when `num_bin == 2`, then compares against it. The Rust `-1` sentinel
(which matches no threshold, disabling the split) is safer and deterministic, but it is a
deliberate behavioral divergence from the reference that is worth an explicit comment so a
future reader does not "fix" it back toward the C++ UB.
**Fix:** Add a one-line comment noting the `num_bin<=2` sentinel is an intentional
safe-divergence from the C++ uninitialized-`rand_threshold` case (no behavior change
needed).

### IN-03: `round_ties_even_cube` is exercised only indirectly; the branch-free identity is unit-tested only for the host `round_ties_even_branchfree` twin

**File:** `crates/lgbm-compute/src/kernels/best_split.rs:324-334`, `:2255-2262`
**Issue:** The `count_recovery_ties_even` test proves `round_ties_even` (intrinsic) ==
`round_ties_even_branchfree` (host `i64`-parity), but the actual in-kernel function
`round_ties_even_cube` (float-parity via `f - 2.0*floor(f*0.5)`) is only validated
transitively through the golden replay's `count_ties_even` case (a single tie value). The
float-parity computation is a distinct implementation from the `i64`-parity fallback the
test checks; an error in the float-parity branch (e.g. for large `f` where
`f - 2*floor(f*0.5)` loses the low bit) would not be caught by the existing unit test.
**Fix:** Add a direct assertion loop over `round_ties_even_cube` for the same
`[0.5,1.5,2.5,3.5,...,10.5,11.5]` set used for the branch-free twin (a `#[cube]` fn is also
a plain Rust `fn`, so it is host-callable in the test).

---

_Reviewed: 2026-07-01T03:22:53Z_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
