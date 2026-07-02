---
phase: 21-end-to-end-on-device-driver-integration-parity-gate
reviewed: 2026-07-02T00:00:00Z
depth: standard
files_reviewed: 3
files_reviewed_list:
  - crates/lgbm-compute/src/kernels/grow_driver.rs
  - crates/lgbm-compute/src/kernels/histogram_arena.rs
  - crates/oracle-harness/tests/learner_parity.rs
findings:
  critical: 1
  warning: 5
  info: 5
  total: 11
status: issues_found
---

# Phase 21: Code Review Report

**Reviewed:** 2026-07-02
**Depth:** standard
**Status:** issues_found

## Summary

This is an adversarial, full-file review of the three Phase-21 sources (not a diff-scoped
refactor review). The prior committed REVIEW scoped only the extract-parameter diff and
found 0 blockers; reviewing the whole driver surfaces a correctness defect it missed.

- `grow_driver.rs` — the on-device best-first grow driver. Its histogram handling
  (FixHistogram + compaction + subtraction trick) is faithful; the `split_gt` tie rule
  matches C++ `SplitInfo::operator>`; and the per-leaf `Vec<f64>` design *sidesteps* the
  prior `HistArena::swap` aliasing hazard entirely (the driver never touches `HistArena`).
  But the `BeforeFindBestSplit` min-data gate is transcribed incorrectly (CR-01) and the
  parity suite was shaped to *avoid* exercising it rather than the code being fixed.
- `histogram_arena.rs` — correct and well-tested in isolation, but by its own doc-comment
  entirely **unused by the live driver**: unit-test-locked dead code relative to Phase 21.
- `learner_parity.rs` — strong live real-binary gates, but several named "gate" tests are
  now vacuous `eprintln!` stubs, and the on-device STRUCTURE gates validate the driver only
  when `LGBM_CUDA_ON_DEVICE=1` — the default merge-gate run does not exercise the driver's
  broad growth path.

## Critical Issues

### CR-01: `both_too_small` combined-AND gate diverges from C++ per-leaf `BeforeFindBestSplit` under `min_data_in_leaf`

**File:** `crates/lgbm-compute/src/kernels/grow_driver.rs:677-684`
**Issue:**
The child pre-scan gate combines both children with a logical AND:

```rust
let both_too_small = left_count < min_data * 2 && right_count < min_data * 2;
for &child in &[new_left, new_right] {
    let depth_capped = max_depth > 0 && leaves[child as usize].depth >= max_depth;
    if depth_capped || both_too_small { ... continue; }
    ...
}
```

C++ `SerialTreeLearner::BeforeFindBestSplit` gates each leaf **independently**
(`num_data(leaf) < min_data_in_leaf * 2` → that leaf alone is unsplittable, and it also
checks `min_sum_hessian_in_leaf`). The driver instead only skips scanning when **both**
children are below `min_data*2`, and it never checks `min_sum_hessian_in_leaf` in this gate
at all. When exactly one child is too small and its sibling is large, `both_too_small ==
false`, so the too-small child is scanned; correctness then depends entirely on downstream
enforcement inside `find_best_split_f64_on`, not on this gate.

This is not hypothetical — the phase's own suite documents the divergence and works around
it instead of fixing the code (`crates/oracle-harness/tests/learner_parity.rs:2574-2576`):
> "We bind via `min_sum_hessian_in_leaf` ... rather than `min_data_in_leaf`, whose
> `min_data*2` both-too-small pre-gate takes a divergent child-leaf path on this small
> corpus."

So the STRUCTURE gate for `min_data_in_leaf` was deliberately avoided because the driver
does **not** reproduce the cpu f64 anchor when `min_data_in_leaf` binds. `min_data_in_leaf`
is a core LightGBM parameter and the driver's tree must match the anchor for it (the
non-negotiable parity contract). The proving config uses `min_data_in_leaf = 1` (rarely
binds), which is why the default gates stay green.

**Fix:** Gate each child independently, mirroring the per-leaf C++ check:

```rust
for &child in &[new_left, new_right] {
    let c = &leaves[child as usize];
    let n = c.rows.len() as i32;
    let depth_capped = max_depth > 0 && c.depth >= max_depth;
    let too_small = n < min_data * 2 || c.sum_h < cfg.min_sum_hessian_in_leaf * 2.0;
    if depth_capped || too_small {
        leaves[child as usize].best = SplitInfo::none();
        leaves[child as usize].best_fpos = -1;
        continue;
    }
    ...
}
```

Then add an on-device STRUCTURE gate that binds `min_data_in_leaf` (removing the case-C
workaround) so the fix is anchored bit-exact.

## Warnings

### WR-01: Default merge-gate run does not exercise the on-device driver's growth path

**File:** `crates/oracle-harness/tests/learner_parity.rs:2316-2351, 2449-2483, 2533-2564`
**Issue:** The primary STRUCTURE parity gates (`learner_parity_on_device_structure_gate`,
`_deep_multileaf_gate`, `_nosplit_gate`, and the ROCm cells) assert the grown tree only when
`LGBM_CUDA_ON_DEVICE=1`. In the default `cargo test` run (env unset), each asserts only
`grown.is_none()`. The driver's multi-feature / deep (>2 live leaves) growth path therefore
has no parity coverage in the default hard merge gate; only `learner_parity_on_device_mindata_gate`
calls the driver env-independently (a single-feature, 8-row, constrained case). A regression
in `grow_driver.rs` on the broad path would pass the default gate.
**Fix:** Drive `grow_tree_on_device_driver` directly (env-independent, as the mindata gate
already does) so the broad structure gates run by default, and/or add a CI job that runs the
suite with `LGBM_CUDA_ON_DEVICE=1`.

### WR-02: Several named "parity gate" tests are vacuous `eprintln!` stubs

**File:** `crates/oracle-harness/tests/learner_parity.rs:311-315, 317-324, 423-432, 456-488, 774-783`
**Issue:** `learner_parity_spine_full_tree`, `learner_parity_spine_per_bin_gains`,
`learner_parity_transcription_crosscheck`, `learner_parity_real_gh_full_tree`, and the
col_wise-golden half of `learner_parity_row_vs_col` now only print
`STALE_SELF_TRANSCRIPTION_NOTE` and assert nothing — they pass unconditionally. Their
replacements (`learner_parity_spine_real_binary`, `_mfb_pos_real_binary`) SKIP gracefully
when the real-binary fixtures are absent (`load_real_tree` → `None` → early return). In a
clean checkout without the committed goldens, full-tree parity has zero live assertions while
CI shows green across all these test names — a masked-divergence risk.
**Fix:** Delete the vacuous stubs or mark them `#[ignore]` with a reason; and make the
real-binary gates fail (not skip) in CI where the goldens are expected present.

### WR-03: `HistArena` is dead code relative to the live driver

**File:** `crates/lgbm-compute/src/kernels/histogram_arena.rs:1-423` (see doc-comment `322-327`)
**Issue:** The module's own doc-comment states the live driver "does NOT consume this
arena ... so this arena is unit-test-locked." The entire ~420-line struct
(`rotate`/`swap`/`set_leaf_slot`/`leaf_handle` + pool + `client.empty` alloc counter) exists
only to satisfy its own unit tests. Presenting it as live production plumbing is misleading
and risks a future caller wiring it and re-introducing the WR-01 aliasing class this
milestone spent effort closing.
**Fix:** Either wire `HistArena` into the driver (replacing the per-leaf `Vec<f64>` clones)
or move it behind an explicitly "reserved, not wired" boundary and drop its public surface
until it is consumed.

### WR-04: `build_leaf_map_on` / `LeafMapBufferStrategy` A/B harness is unused by the driver

**File:** `crates/lgbm-compute/src/kernels/grow_driver.rs:96-178`
**Issue:** The `LeafMapBufferStrategy` enum and `pub fn build_leaf_map_on` A/B experiment
"locks the safe" row→leaf buffer strategy, but the actual driver never calls it and carries
no running leaf-map buffer — it partitions each leaf into fresh `Vec<u32>` via
`partition_leaf_stable`. The "LOCKED: DOUBLE-BUFFER" conclusion (asserted only at
`learner_parity.rs:2080-2082`) is never applied in the shipped driver. This is test-only
`pub` API that reads as load-bearing.
**Fix:** Move the harness into the test module (or label it as a decision-record experiment)
and drop the "locked strategy" language, since the driver uses neither strategy.

### WR-05: `grow_tree_on_device_driver_with_cfg` guard error strings name the wrong function

**File:** `crates/lgbm-compute/src/kernels/grow_driver.rs:467, 472`
**Issue:** The extract-parameter refactor moved the guard bodies into `_with_cfg`, but both
`ComputeError::Runtime` detail strings still hardcode the delegator name
`grow_tree_on_device_driver:`. `learner_parity_on_device_mindata_gate` is the first caller to
invoke `_with_cfg` **directly** — any caller that trips these guards (`num_leaves < 1`, empty
`features`) gets an error naming a function it did not call, degrading diagnosability at a
`thiserror` domain boundary (per CLAUDE.md).
**Fix:** Reference the actual function (or a neutral op label) in both strings, e.g.
`"grow_tree_on_device_driver_with_cfg: num_leaves must be >= 1, got {num_leaves}"`.

## Info

### IN-01: `real_threshold` fallback silently records a bin index as a real threshold

**File:** `crates/lgbm-compute/src/kernels/grow_driver.rs:592-596`
**Issue:** `f.bin_upper_bound.get(best.threshold as usize).copied().unwrap_or(best.threshold as f64)`
— if `best.threshold` is ever out of range, the driver records the raw bin index cast to
`f64` as the tree's real threshold: a plausible-looking wrong value that would corrupt
prediction routing and mask an off-by-one in the offset/compaction threshold space.
**Fix:** Return a typed `ComputeError` on out-of-range threshold instead of `unwrap_or`.

### IN-02: `default_left` tie tolerance is applied on the pure-f64 CPU structure gate

**File:** `crates/oracle-harness/tests/learner_parity.rs:2239-2284` (used at `2332`, `2466`, `2548`)
**Issue:** `assert_on_device_tree_matches_cpu_anchor` tolerates a `default_left` flip on a
`split_gain` near-tie. That allowance is for the f32-vs-f64 ROCm path, but the `CpuBackend`
driver runs f64 vs an f64 anchor — no f32 accumulation, so `default_left` should be
bit-exact. The tolerance is looser than needed for the CPU gate and could mask a real
direction bug.
**Fix:** Use the strict `assert_gpu_tree_matches_cpu_anchor` (full `decision_type` equality)
for the cubecl-cpu driver gates; reserve the tie-aware comparator for the `mod hip` cells.

### IN-03: Case C's `env_on` lane checks only leaf-count inequality, not structure parity

**File:** `crates/oracle-harness/tests/learner_parity.rs:2652-2666`
**Issue:** In `learner_parity_on_device_mindata_gate`, the `env_on` branch asserts only
`driver_tree.num_leaves < seam_tree.num_leaves`. The constrained STRUCTURE-bit-exact compare
against `cpu_anchor_tree` runs only in the env-unset branch. Neither single lane fully
validates the constrained driver; a driver defect producing the right leaf count but a wrong
structure would pass the env=1 lane.
**Fix:** In the env=1 lane, also build the constrained anchor and call
`assert_on_device_tree_matches_cpu_anchor(&driver_tree, &constrained_anchor, ...)` — the
direct `_with_cfg` call is already env-independent.

### IN-04: `HistArena::swap` leaves the `{parent,smaller,larger}_idx` role fields stale after multi-leaf use

**File:** `crates/lgbm-compute/src/kernels/histogram_arena.rs:416-420`
**Issue:** `swap` updates the single-triple role indices to reflect only the *last* swap, so
`parent_handle`/`smaller_handle`/`larger_handle` are meaningless in a multi-leaf loop (only
`leaf_handle(leaf)` is correct). Harmless today (arena unused, WR-03) but a trap for a future
caller mixing the two APIs.
**Fix:** If retained, drop the single-triple role fields from the multi-leaf `swap` path or
document that they are undefined after `swap`.

### IN-05: Redundant `num_leaves.max(1)` after the `num_leaves < 1` guard

**File:** `crates/lgbm-compute/src/kernels/grow_driver.rs:519`
**Issue:** `DeviceCudaTree::<R>::new(client, num_leaves.max(1) as usize, ...)` — `num_leaves
>= 1` is already guaranteed by the guard at lines 465-469, so `.max(1)` is dead defensiveness
that obscures the invariant.
**Fix:** Use `num_leaves as usize` directly.

---

_Reviewed: 2026-07-02_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
