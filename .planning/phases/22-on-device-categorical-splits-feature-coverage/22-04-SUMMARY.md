---
phase: 22-on-device-categorical-splits-feature-coverage
plan: 04
subsystem: lgbm-compute / on-device categorical grow driver
tags: [categorical, bitset, grow-driver, split-finder, kepsilon, routing-convention, ODL-22]
status: complete
requires:
  - "22-01: DeviceSplitInfo runtime cat_width slab (D-03); set_cat_thresholds; D-06 learner gate"
  - "22-02: GrowFeature native categorical fields (bin_to_category, cat_smooth, cat_l2, max_cat_threshold, max_cat_to_onehot, min_data_per_group)"
  - "22-03: categorical_split::{find_best_threshold_categorical, set_real_threshold, construct_bitset} + the sum_hessian pre-bump caller contract"
provides:
  - "best_split.rs: the three categorical dispatch seams call the §8.1 evaluator and map into SplitScalars (with num_cat_threshold); PASS THROUGH the pre-bumped sum_hessian (never bump)"
  - "grow_driver.rs: a categorical grow branch (scan_leaf routes categorical; driver body stages winners into the pre-allocated DeviceSplitInfo cat slab, materializes real+inner bitsets, calls the EXISTING split_categorical_on_device + partition_categorical_on_device); the SINGLE +2*kEpsilon sum_h bump lives here"
  - "Unit test categorical_driver_bumps_sum_hessian_once (W-4 double/missed-bump guard)"
  - "Unit test categorical_partition_counts_match_host_stable (Open Q1 routing-convention isolation)"
affects:
  - "22-05: the full on-device categorical structure gate consumes this driver branch end-to-end"
tech-stack:
  added: []
  patterns:
    - "Branched driver body: categorical vs numeric differ ONLY in partition+mutation; child-seed/subtract/scan shared"
    - "Pre-allocated DeviceSplitInfo cat slab staged on the LIVE grow path (ODL-02 allocate-once), bitsets DERIVED from the slab"
    - "Single-owner f64 categorical anchor reused from best_split's f32 mirrors (def-f8u-01, never GPU-vs-GPU)"
key-files:
  created: []
  modified:
    - "crates/lgbm-compute/src/kernels/best_split.rs"
    - "crates/lgbm-compute/src/kernels/grow_driver.rs"
decisions:
  - "best_split.rs stage-1 seams build the §8.1 GainConfig from Stage1Scalars + GainConfig defaults for the five categorical knobs (the CUDA-mirror seam is not fed per-feature cat config; the LIVE driver threads real per-feature config via GrowFeature). Confined the change to best_split.rs (files_modified contract) — no Stage1Scalars widening / no test-file edits."
  - "DeviceSplitInfo is allocated ONLY when the feature set has a categorical feature (Option<DeviceSplitInfo>), so a pure-numeric grow allocates nothing new (SC #4 byte-for-byte)."
  - "Open Q1 resolved: partition routing uses the INNER-bin bitset (construct_bitset over the winning bins); the REAL category-value bitset is the model cat_threshold_. Both device and host derive offset=(most_freq_bin==0)?1:0 internally, so the same offset math is used on both sides."
metrics:
  duration: ~45m
  completed: 2026-07-02
  tasks: 3
  files: 2
  commits: 3
---

# Phase 22 Plan 04: Wire the On-Device Categorical Grow Path End-to-End Summary

Connected the 22-03 §8.1 categorical evaluator + §6.3 bitset construction to the on-device
grow driver, replacing the two bail-out seams (`best_split.rs` `is_valid=false` sentinel and
`grow_driver.rs` `scan_leaf` `continue`) so the driver grows a categorical tree on-device by
calling the EXISTING §9 partition and §10 `SplitCategorical` kernels. Everything additive; the
numeric spine is byte-for-byte unchanged.

## What was built

- **Task 1 — `best_split.rs` dispatch seam.** The three former categorical sentinels
  (`find_best_splits_stage1_on` f64, `find_best_splits_stage1_f32_on`,
  `find_best_splits_stage1_globalmem_f32_on`) now call
  `categorical_split::find_best_threshold_categorical` and map the returned `CategoricalSplit`
  into `SplitScalars` (via the new `categorical_split_scalars` helper), setting `num_cat_threshold`
  from the evaluator (numeric leaves it 0), preserving `is_valid`-when-no-gain, and PASSING
  THROUGH the pre-bumped `sum_hessian` (never bumps — W-4). The f32 mirrors widen their `&[f32]`
  histogram to f64 (categorical is always the f64 single-owner anchor, def-f8u-01). A
  `categorical_gain_config` helper builds the evaluator config from `Stage1Scalars` + `GainConfig`
  defaults for the five categorical knobs.
- **Task 2 — `grow_driver.rs` categorical grow branch.** `scan_leaf` now splits the bail-out:
  `na_as_missing` stays deferred, but `BinType::Categorical` routes into the §8.1 evaluator. The
  driver applies the SINGLE `+2*kEpsilon` `sum_h` bump here (`bump_sum_hessian_cat`, mirroring host
  `learner.rs:2760`) before the evaluator. The driver body, for a categorical best split: (1) stages
  the winning thresholds into the pre-allocated `DeviceSplitInfo` cat slab via `set_cat_thresholds`
  (W-3/SC #1, allocate-once, guarded to `cat_width`); (2) materializes the real + inner bitsets FROM
  the slab-staged thresholds via `set_real_threshold` (§6.3); (3) partitions parent rows by
  categorical membership via the EXISTING `partition_categorical_on_device` (§9); (4) grows the node
  via the EXISTING `split_categorical_on_device` (§10) with both bitsets. The numeric branch is
  byte-for-byte unchanged (extracted into an `else` arm).
- **Task 3 — Open Q1 isolation test.** `categorical_partition_counts_match_host_stable` asserts the
  on-device categorical partition (left/right counts AND full stable order) equals the host
  `partition_categorical_stable` for both fixtures' winning inner bitsets.

## Where the +2*kEpsilon bump lands (W-4)

**Exactly once, driver-owned.** `bump_sum_hessian_cat(sum_h) = sum_h + 2.0*f64::from(K_EPSILON)` is
called in `scan_leaf`'s categorical arm (grow_driver.rs), before the evaluator. `best_split.rs`
PASSES THROUGH the already-bumped value (documented at each seam; no bump expression exists in
best_split.rs on the categorical path — verified by grep), and the 22-03 evaluator does not bump
internally. `categorical_driver_bumps_sum_hessian_once` pins the driver-supplied value to
`raw + 2*kEpsilon` bit-exact for BOTH fixtures (40.0 and 60.0). Note: at those magnitudes `2*kEpsilon`
(2e-15) sits below the f64 ULP (~7e-15) so the bump is numerically absorbed — faithful to the host,
but not discriminating there; the double/missed-bump guard is therefore additionally exercised at a
representable magnitude (`tiny = 1e-13`), where one bump differs from raw (missed-bump) and from a
two-bump chain (double-bump).

## SC #1 — pre-allocated representation is end-to-end (W-3)

The chosen categorical thresholds land in the pre-allocated `DeviceSplitInfo` cat_threshold slab
(D-03 runtime width = max `max_cat_threshold` across categorical features, default
`MAX_CAT_PER_SPLIT`) via `set_cat_thresholds` on the LIVE grow path — allocate-once, no per-`SplitInfo`
device alloc. The host `Vec<u32>` real/inner bitsets that `split_categorical_on_device` consumes are
DERIVED from the slab-staged thresholds (read back via `dsi.cat_threshold(slot)` → `set_real_threshold`),
not a parallel per-split allocation. The `DeviceSplitInfo` is allocated only when the feature set
contains a categorical feature, so pure-numeric grows allocate nothing new.

## Resolved Open Q1 routing convention

The host router (`route_to_left_categorical` / `partition_categorical_stable`) keys membership on
`FindInBitset(bitset, bin - min_bin + offset)` with `offset = (most_freq_bin==0)?1:0`. The driver
therefore feeds `partition_categorical_on_device` the **INNER-bin bitset** (`construct_bitset` over the
winning bins) with `min_bin = f.min_bin` and `most_freq_bin = f.most_freq_bin`; both device and host
derive the same `offset` internally (== `offset_for_most_freq_bin`). The **REAL category-value bitset**
is passed to `split_categorical_on_device` as the model `cat_threshold_`. Task 3 confirms device
partition counts == host stable counts for both fixtures, isolating this convention BEFORE the full
structure gate in 22-05.

## Parity / test results

- `cargo test -p lgbm-compute --lib` → **131 passed, 0 failed, 1 ignored** (best_split, grow_driver
  including the two new tests, categorical_split, all existing kernels).
- `cargo test --workspace` (env unset) → **961 passed, 0 failed** — numeric spine byte-unchanged (SC #4).
- `cargo clippy -p lgbm-compute --lib --tests` → **0 warnings in the two changed files**.

## Deviations from Plan

**[Rule 3 — Blocking] `best_split.rs` categorical config source.** The plan's Task 1 said to pass
"the categorical config scalars (from the task/GrowFeature)", but `best_split.rs`'s `Stage1Scalars` /
`SplitFindTask` do not carry the five per-feature categorical knobs, and `GrowFeature` is not visible
in `best_split.rs`. Widening `Stage1Scalars` would force edits to two test files, violating the plan's
`files_modified: [best_split.rs, grow_driver.rs]` contract. Resolution: the `best_split.rs` seam builds
its `GainConfig` from the numeric `Stage1Scalars` fields + `GainConfig` defaults for the categorical
knobs, documented as "the CUDA-mirror stage-1 seam is not fed per-feature cat config; the LIVE grow
driver threads real config via GrowFeature (Task 2)". The best_split.rs stage-1 launchers are the
separate Phase-17 CUDA-mirror path (not used by the current driver's `scan_leaf`, which routes
categorical directly into the evaluator), so this is faithful for that seam and keeps the change
confined. No behavior change on the live merge-gate path.

**[Adjustment] W-4 bump-pin test at fixture magnitudes.** The plan's `categorical_driver_bumps_sum_hessian_once`
asserts the bumped value bit-exact; at the fixture leaf hessian sums (40/60) `2*kEpsilon` is below the
f64 ULP and absorbed, so the bit-exact fixture assertion is `bump(raw) == raw + 2*kEpsilon` (which holds
by rounding on both sides) and the discriminating double/missed-bump guard is placed at a representable
magnitude (`1e-13`). This is faithful to the host (the host bump is likewise absorbed at those sums) and
still catches an accidental `+4*kEpsilon` or omitted bump.

## Self-Check: PASSED

- FOUND: `crates/lgbm-compute/src/kernels/best_split.rs`
- FOUND: `crates/lgbm-compute/src/kernels/grow_driver.rs`
- FOUND commit `851cc9d` (Task 1)
- FOUND commit `47e94c4` (Task 2 + Task 3)
