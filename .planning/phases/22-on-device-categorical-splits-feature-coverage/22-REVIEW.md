---
phase: 22-on-device-categorical-splits-feature-coverage
reviewed: 2026-07-02T00:00:00Z
depth: standard
files_reviewed: 8
files_reviewed_list:
  - crates/lgbm-compute/src/kernels/best_split.rs
  - crates/lgbm-compute/src/kernels/categorical_split.rs
  - crates/lgbm-compute/src/kernels/grow_driver.rs
  - crates/lgbm-compute/src/kernels/mod.rs
  - crates/lgbm-compute/src/kernels/split_info.rs
  - crates/lgbm-compute/tests/split_info.rs
  - crates/lgbm-treelearner/src/learner.rs
  - crates/oracle-harness/tests/learner_parity.rs
findings:
  critical: 1
  warning: 4
  info: 2
  total: 7
status: issues_found
---

# Phase 22: Code Review Report

**Reviewed:** 2026-07-02
**Depth:** standard
**Files Reviewed:** 8
**Status:** issues_found

## Summary

Phase 22 wires on-device categorical splits: a byte-for-byte transcription of the
golden host categorical evaluator (`categorical_split.rs`), its integration into the
on-device grow driver (`grow_driver.rs`), the categorical dispatch seam in
`best_split.rs`, and the runtime-width categorical slab in `split_info.rs`.

The §8.1 evaluator transcription (`categorical_split.rs`) was checked line-by-line
against the authoritative golden host source
(`crates/lgbm-treelearner/src/feature_histogram_categorical.rs`) and is faithful
(one-hot vs many-vs-many dispatch, the deliberate `l2` asymmetry, the round-half-up
count recovery, the stable ctr sort, the finalize bitset reconstruction for both
directions). `split_info.rs`'s runtime-width slab and overflow-guarded sizing are
correct. The `learner.rs` D-06 host-fallback gate for the categorical+quantized combo
is correct and well-tested.

The dominant concern is **routing-convention divergence between the host categorical
partition (route by REAL category value) and the on-device categorical partition
(route by INNER bin)**. The two are only equivalent when `min_bin == offset`, which
holds for both committed fixtures (`most_freq_bin == 1 ⇒ offset 0`) but NOT for a
categorical feature whose `most_freq_bin == 0` (`⇒ offset 1`). No test covers the
`offset == 1` categorical case, and the entire on-device categorical parity suite is
gated behind `LGBM_CUDA_ON_DEVICE=1`, so this escapes the default merge gate.

## Critical Issues

### CR-01: Device categorical partition is off-by-`offset` for `most_freq_bin == 0` features

**File:** `crates/lgbm-compute/src/kernels/categorical_split.rs:124-148` (inner bitset
construction) + `crates/lgbm-compute/src/kernels/grow_driver.rs:717-732` (device
partition call) + `crates/lgbm-compute/src/kernels/data_partition.rs:153-176`
(`route_to_left_categorical`).

**Issue:** The on-device categorical grow branch partitions rows with the INNER-bin
bitset via `partition_categorical_on_device(...)`, whose per-row decision is
`FindInBitset(inner_bitset, bin - min_bin + offset)` with `offset = (most_freq_bin ==
0) ? 1 : 0`. But `set_real_threshold` builds `inner_bitset` with bits set at the RAW
winning bins (`cat_threshold_bins` as-is), i.e. `construct_bitset(&inner_keys)` where
`inner_keys[i] == winning_bin`. For a row belonging to a winning category, the lookup
key is `winning_bin - min_bin + offset`, but the bit is at `winning_bin`. These agree
only when `min_bin == offset`.

Both committed fixtures use `most_freq_bin == 1 ⇒ offset 0`, `min_bin == 0`, so
`min_bin == offset == 0` and the paths coincide. For a categorical feature with
`most_freq_bin == 0 ⇒ offset 1` (a real configuration — see `learner.rs:4842` fixture
with `offset: 1, most_freq_bin: 0`), the device routes rows using key `bin + 1` while
the bit is at `bin`, sending members to the wrong child. This silently diverges from
the golden host path, which routes by REAL category value
(`learner.rs:3701-3721`, `data_partition.split_categorical(..., &cat_bitset_real,
&f.bin_to_category)`) and is therefore correct for all offsets. The `set_real_threshold`
doc comment even encodes the wrong assumption explicitly: it annotates the lookup as
`FindInBitset(bitset, bin - 0 + 0)`, dropping the `+ offset` term that
`route_to_left_categorical` actually applies.

This violates the non-negotiable bit-exact-vs-C++ contract for any categorical feature
whose most-frequent bin is bin 0. It is latent this milestone
(`on_device_growth_supported()` stays `false`, so the path is only reachable via the
direct `grow_tree_on_device_driver_with_cfg` test entrypoints) but is a genuine
correctness defect in the reviewed code and is guarded by no test.

**Fix:** Make the device inner-bin routing key-consistent with the bitset. Either build
the inner bitset over the transformed keys the router looks up:
```rust
// set_real_threshold: build inner bitset over (bin - min_bin + offset) keys, not raw bins.
let offset = if most_freq_bin == 0 { 1 } else { 0 };
let inner_keys: Vec<u32> = cat_threshold_bins
    .iter()
    .map(|&b| (b - min_bin + offset) as u32)
    .collect();
let inner_bitset = construct_bitset(&inner_keys);
```
(threading `min_bin`/`most_freq_bin` in), OR — simpler and matching the golden host
exactly — route the device partition by the REAL category-value bitset +
`bin_to_category` (as `data_partition.split_categorical` does) instead of the inner-bin
bitset. Add an `offset == 1` (`most_freq_bin == 0`) categorical fixture to the parity
suite so the fix is proven and the regression cannot recur.

## Warnings

### WR-01: On-device categorical parity gates never run in the default merge gate

**File:** `crates/oracle-harness/tests/learner_parity.rs:1580-1636` (`run_categorical_cell_on_device`),
`:2524-2531` (structure gate categorical extension).

**Issue:** Both the STRUCTURE gate (`assert_categorical_device_structure_gate`) and the
real-4.6-golden gate (`run_categorical_cell_on_device`) are guarded by
`if env_on` / `LGBM_CUDA_ON_DEVICE == "1"`. With the env unset (the default merge gate)
neither runs, so the on-device categorical grow-driver wiring — the partition routing,
the slab staging, the bitset construction — is exercised by NO test in the default CI
lane. Only the pure evaluator unit tests in `categorical_split.rs` run unconditionally.
This is what allows CR-01 (and any future wiring regression) to ship green. Note the
structure gate is a pure f64-vs-f64 comparison (no GPU required), so it could run
unconditionally on the cpu anchor.

**Fix:** Run `assert_categorical_device_structure_gate` on the cubecl-cpu lane
unconditionally (it needs no GPU and no env), keeping only the real-golden fidelity
gate behind the env if desired. This restores default-merge-gate coverage of the
categorical driver wiring.

### WR-02: Stage-1 categorical seam uses default categorical config, not the runtime config

**File:** `crates/lgbm-compute/src/kernels/best_split.rs:625-635` (`categorical_gain_config`),
consumed at `:701-713`, `:1028-1044`, `:1899-1916`.

**Issue:** `categorical_gain_config(scalars)` fills only the numeric gain knobs from
`Stage1Scalars` and takes `cat_l2`/`cat_smooth`/`max_cat_threshold`/`max_cat_to_onehot`/
`min_data_per_group` from `GainConfig::default()` (i.e. `Config::default()` = 10.0 /
10.0 / 32 / 4 / 100). Any categorical task evaluated through
`find_best_splits_stage1_on` / `find_best_splits_stage1_f32_on` /
`find_best_splits_stage1_globalmem_f32_on` therefore silently uses the DEFAULT
categorical config regardless of the user's actual config, producing wrong splits. The
`min_data_per_group = 100` default would suppress many-vs-many entirely on small leaves.
This is documented as a "seam" and the live driver (`scan_leaf`) correctly bypasses it
(calling `find_best_threshold_categorical` directly with `categorical_feature_config`),
but the three `pub` stage-1 launchers remain a silent-wrong-answer foot-gun.

**Fix:** Either extend `Stage1Scalars` to carry the five per-feature categorical
scalars and thread them through `categorical_gain_config`, or make the stage-1
categorical branch return a typed `ComputeError` ("categorical config not supplied on
the stage-1 seam") instead of silently evaluating with defaults.

### WR-03: No coverage for `most_freq_bin == 0` (offset 1) categorical anywhere in Phase 22

**File:** `crates/lgbm-compute/src/kernels/categorical_split.rs` (tests, lines 519-814),
`crates/lgbm-compute/src/kernels/grow_driver.rs` (tests, lines 931-1015),
`crates/oracle-harness/tests/learner_parity.rs` (cat fixtures).

**Issue:** Every categorical fixture and test uses `offset == 0` (`most_freq_bin == 1`
for the on-device fixtures; the evaluator unit tests pass `offset` in `{0, 1}` to the
finder but never drive the finalize→`set_real_threshold`→partition chain with
`offset == 1`). The `offset == 1` branch of `finalize` (`bin_start = 0`, `cat_threshold
= compacted_idx + 1`), the compaction interaction, and the partition routing are thus
untested — which is precisely how CR-01 hides. `set_real_threshold_out_of_range_bin_does_not_panic`
tests bounds but not offset semantics.

**Fix:** Add an `offset == 1` (`most_freq_bin == 0`) categorical fixture that drives the
full evaluator → bitset → partition chain and pins it to a host/golden reference.

### WR-04: `both_too_small` combined gate diverges from C++ per-leaf `min_data*2` gate

**File:** `crates/lgbm-compute/src/kernels/grow_driver.rs:890-908`.

**Issue:** `let both_too_small = left_count < min_data * 2 && right_count < min_data * 2;`
applies a SINGLE combined predicate to BOTH children, whereas C++ `BeforeFindBestSplit`
applies `num_data < min_data_in_leaf * 2` per leaf. When one child is small and the
other large, `both_too_small` is `false` and the small child is still scanned. In
practice `scan_leaf`'s per-side `min_data_in_leaf` guards reject any split the small
child could produce, so the grown tree is unaffected — but the logic is not a faithful
mirror of the reference gate and is fragile if the downstream guards ever change. Also
note `min_data * 2` can overflow `i32` for a pathological `min_data > i32::MAX/2`
(not realistic, but unchecked).

**Fix:** Gate each child independently: `let too_small = leaves[child].rows.len() as i32
< min_data * 2;` inside the per-child loop, mirroring the C++ per-leaf check, and use
`i32::saturating_mul` (or validate `min_data`) for the `* 2`.

## Info

### IN-01: Categorical `cat_threshold_real` slab is staged but its content is dead on the grow path

**File:** `crates/lgbm-compute/src/kernels/grow_driver.rs:700-718`.

**Issue:** `win_real` is computed (`bin_to_category` mapping) and stored into the
`cat_threshold_real` slab via `set_cat_thresholds`, but the real bitset actually used
for the split (`real_bitset`) is recomputed independently by `set_real_threshold` from
`slab_bins` (which equals `win_bins`, not `win_real`). The `cat_threshold_real` slab
content is therefore never read on this path — `win_real` and the slab-real store are
redundant work. Harmless (both compute the same mapping) but confusing.

**Fix:** Either consume `dsi.cat_threshold_real(best_leaf)` to build `real_bitset`
(removing the recompute in `set_real_threshold`), or drop the `win_real` computation and
document that the real slab is reserved-only on the driver path.

### IN-02: Unreachable `.expect` in the categorical branch should be a typed error / `unreachable!`

**File:** `crates/lgbm-compute/src/kernels/grow_driver.rs:705-707`.

**Issue:** `split_info.as_mut().expect("DeviceSplitInfo is allocated whenever
categorical features exist")` panics if `None`. It is provably unreachable (the
categorical branch is entered only when `f.bin_type == Categorical`, which implies
`has_categorical`, which implies `split_info == Some`), but a raw panic in the grow loop
is inconsistent with the module's otherwise-uniform typed-`ComputeError` boundary.

**Fix:** Use `.ok_or_else(|| ComputeError::Runtime { .. })?` or `unreachable!` with the
invariant justification to keep the error surface uniform.

---

_Reviewed: 2026-07-02_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
