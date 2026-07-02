---
phase: 22-on-device-categorical-splits-feature-coverage
plan: 05
subsystem: testing / on-device categorical parity gate
tags: [categorical, bitset, dual-anchor, real-golden, structure-gate, def-f8u-01, ODL-22]

# Dependency graph
requires:
  - phase: 22-04
    provides: "grow_driver categorical branch wired end-to-end (§8.1 evaluator, +2*kEpsilon bump, DeviceSplitInfo cat slab, real+inner bitsets, partition/split_categorical_on_device)"
  - phase: 22-03
    provides: "categorical_split::find_best_threshold_categorical (§8.1) + construct_bitset (§6.3)"
  - phase: 22-02
    provides: "GrowFeature native categorical fields carried by grow_features_of"
provides:
  - "learner_parity_on_device_structure_gate extended with cat_onehot + cat_manyvsmany cases driving the DEVICE driver against the cpu f64 tie-aware anchor (D-01 #2)"
  - "learner_parity_categorical_{onehot,manyvsmany}_on_device: DEVICE tree pinned bit-exact to the real lib_lightgbm 4.6 goldens (num_cat / kCategoricalMask / cat_boundaries / real cat_threshold bitset) + predict-through the bitset (D-01 #1, SC #3)"
  - "categorical is the FIRST on-device subsystem anchored to a REAL reference (not a host re-transcription)"
  - "fix: device categorical many-vs-many now uses the host's f64 std::stable_sort (crash-free + golden-faithful on NaN ctr child leaves)"
affects: [on-device-categorical-followups, cuda-on-device-training-backend]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Merge two existing harness twins (structure gate + real-golden host cell) rather than authoring a new harness"
    - "Dual-anchor discipline: DEVICE tree pinned to BOTH the cpu f64 fold (structure) AND the real 4.6 golden (fidelity); never GPU-vs-GPU (def-f8u-01)"
    - "New on-device fidelity cells gated behind LGBM_CUDA_ON_DEVICE so the default env-unset merge gate stays byte-green"

key-files:
  created: []
  modified:
    - "crates/oracle-harness/tests/learner_parity.rs"
    - "crates/lgbm-compute/src/kernels/categorical_split.rs"

key-decisions:
  - "The categorical device structure-gate cases use the tie-aware default_left comparator (assert_on_device_tree_matches_cpu_anchor) per the plan must_haves; on the CpuBackend f64 lane there are no ties so it behaves like the strict comparator but honors the contract"
  - "Predict-through (SC #3) asserts predict_leaf_index equality per training row across the DEVICE tree, its model-text reparse, AND the golden — exercising the real find_in_bitset routing rather than comparing shrinkage-divergent leaf values"
  - "The many-vs-many device sort was reverted from f32 bitonic_argsort_on to the host's f64 std::stable_sort (faithful transcription of the source; the evaluator IS the def-f8u-01 f64 anchor)"

patterns-established:
  - "On-device categorical fidelity gate: drive grow_tree_on_device_driver_with_cfg on cat_corpus fixtures, assert vs cpu f64 anchor + real golden, gated behind LGBM_CUDA_ON_DEVICE"

requirements-completed: [ODL-22]

# Metrics
duration: ~40min
completed: 2026-07-02
status: complete
---

# Phase 22 Plan 05: On-Device Categorical Dual-Anchor Parity Gate Summary

**The DEVICE-grown categorical tree is now pinned bit-exact to the real lib_lightgbm 4.6 goldens (bitset / kCategoricalMask / num_cat / cat_boundaries) AND structure-bit-exact to the cpu f64 anchor for both one-hot and many-vs-many — making categorical the first on-device subsystem anchored to a REAL reference, with the merge gate staying byte-green env-unset.**

## Performance

- **Duration:** ~40 min
- **Started:** 2026-07-02
- **Completed:** 2026-07-02
- **Tasks:** 2 (+1 auto-fixed deviation)
- **Files modified:** 2

## Accomplishments
- **D-01 #2 (Task 1):** `learner_parity_on_device_structure_gate` now drives `cat_onehot` and `cat_manyvsmany` through `grow_tree_on_device_driver_with_cfg` on the cubecl-cpu f64 lane and pins each DEVICE tree STRUCTURE bit-exact to the cpu f64 anchor via the tie-aware `default_left` comparator (never GPU-vs-GPU).
- **D-01 #1 + SC #3 (Task 2):** `learner_parity_categorical_onehot_on_device` and `learner_parity_categorical_manyvsmany_on_device` grow each fixture's tree through the DEVICE driver and assert bit-exact vs the committed real 4.6 goldens on all four properties (`num_cat`, `dt & 1` kCategoricalMask, `cat_boundaries`, real `cat_threshold` bitset) + the full learner-authoritative fields, then round-trip prediction through the categorical bitset (`predict_leaf_index`) on the device tree and its model-text reparse vs the golden.
- **Fidelity upgrade realized:** categorical is now the FIRST on-device subsystem pinned to a REAL lib_lightgbm reference rather than a host re-transcription.
- **Merge gate preserved (SC #4):** `cargo test --workspace` env-unset = **963 passed, 0 failed** (up from 961 in 22-04; the 2 new device cells skip-pass when the env is unset).

## Task Commits

1. **Deviation fix (prerequisite): device categorical f64 stable-sort** - `0566370` (fix)
2. **Task 1 + Task 2: device categorical dual-anchor cells** - `edcba63` (test)

## Files Created/Modified
- `crates/oracle-harness/tests/learner_parity.rs` - Added `assert_categorical_device_structure_gate` + two calls inside `learner_parity_on_device_structure_gate` (D-01 #2, env-gated); added `run_categorical_cell_on_device` + `learner_parity_categorical_{onehot,manyvsmany}_on_device` (D-01 #1, SC #3, env-gated).
- `crates/lgbm-compute/src/kernels/categorical_split.rs` - Many-vs-many ctr sort reverted from f32 `bitonic_argsort_on` to the host's f64 `std::stable_sort` (deviation fix, below).

## Tie-free confirmation for cat_manyvsmany

The top-level many-vs-many split is TIE-FREE (ctr = 0, -1, -2, -8, -9, -10 — strictly distinct), so the bitset compare against the golden is deterministic (T-22-13). The DEVICE tree's real `cat_threshold` bitset, `cat_boundaries`, and `kCategoricalMask` bit match the real 4.6 golden bit-exact.

## Merge-gate confirmation

The default cubecl-cpu f64 lane merge gate stays byte-green: `cargo test --workspace` with `LGBM_CUDA_ON_DEVICE` unset = 963 passed / 0 failed, and `learner_parity_categorical_no_regression_numeric_spine` is green. All new fidelity cells are gated behind `LGBM_CUDA_ON_DEVICE`, so the env-unset numeric spine is byte-unchanged (SC #4).

## Decisions Made
- Structure-gate cases use the tie-aware comparator (`assert_on_device_tree_matches_cpu_anchor`) per the plan must_haves; on the f64 CpuBackend lane there are no genuine ties, so it is as strict as the numeric gate's comparator while honoring the "tie-aware on default_left" contract.
- Predict-through (SC #3) compares `predict_leaf_index` per training row (device tree vs its model-text reparse vs golden) rather than raw predicted values, because device leaf values are the RAW Newton output while the golden carries the shrinkage'd leaf — leaf-index routing is the correct invariant for the categorical bitset and it exercises the real `find_in_bitset` path.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Device categorical many-vs-many crashed (OOB) and diverged from the golden on NaN ctr keys**
- **Found during:** Task 1/Task 2 (first end-to-end exercise of the many-vs-many device grow loop — the plan's declared "linchpin")
- **Issue:** `categorical_split::find_best_threshold_categorical` (22-03) substituted the host's f64 `std::stable_sort` of the ctr order with an f32 `bitonic_argsort_on`. The 22-03/22-04 unit tests only ever called the evaluator on the TOP-LEVEL histogram (finite, tie-free ctr). Driving the FULL device grow loop (num_leaves=4) reaches deeper child leaves where a zero-hessian categorical bin yields `ctr = grad/(hess+cat_smooth) = 0/0 = NaN`. On NaN keys the f32 bitonic sort (a) diverges from the f64-stable order (which defines the golden) and (b) leaks its power-of-two padding indices (6, 7 for a 6-element input padded to 8) into the truncated permutation, indexing `sorted_idx` (len 6) out of bounds → panic `index out of bounds: the len is 6 but the index is 6`. Because `LGBM_CUDA_ON_DEVICE=1` routes SerialTreeLearner through the device path, both the on-device cells AND the host `learner_parity_categorical_manyvsmany` cell (under env=1) hit the panic.
- **Fix:** Transcribed the host anchor's sort verbatim (`feature_histogram_categorical.rs:226-232`): `sorted_idx.sort_by(|&a,&b| ctr(a).partial_cmp(&ctr(b)).unwrap_or(Equal))`. The categorical evaluator IS the single-owner f64 anchor (def-f8u-01), so its ctr order MUST match the host (== the golden) bit-exact, including the NaN-Equal stable behavior on degenerate child leaves. This is crash-free by construction (no padding). `client` is now unused in the body (host-serial sort); kept in the signature for API symmetry (no caller edits) and discarded with `let _ = client;`. Doc comment updated.
- **Files modified:** `crates/lgbm-compute/src/kernels/categorical_split.rs`
- **Verification:** `cargo test -p lgbm-compute --lib categorical` = 18 passed (finite-hist regressions unchanged); `LGBM_CUDA_ON_DEVICE=1` many-vs-many device + host cells now green; `cargo test --workspace` env-unset = 963 passed. Clippy: no warnings cite the changed file.
- **Committed in:** `0566370` (fix commit, prerequisite to the Task 1/2 test commit)

---

**Total deviations:** 1 auto-fixed (1 bug, correctness/crash on the plan's linchpin path)
**Impact on plan:** The fix is a faithful transcription correction that makes the device many-vs-many path bit-exact to the golden — exactly the plan's goal. No scope creep; confined to the one evaluator function. The plan's test-only file contract expanded by one prerequisite source fix because the fidelity gate surfaced a real prior-wave defect it exists to catch.

## Issues Encountered
- Under `LGBM_CUDA_ON_DEVICE=1`, many unrelated NUMERIC parity cells (monotone, extra_trees, col_sampler, growth_path_subtract) fail because the on-device driver does not yet implement those numeric features (e.g. monotone → the driver splits to 4 leaves where the monotone golden is 1 leaf). Verified pre-existing by restoring both files to HEAD~2 and re-running `learner_parity_monotone_basic` under env=1 (identical failure). This is the documented D-04 posture: env=1 is best-effort on-device smoke, the env-unset cpu f64 lane is the hard merge gate. Not introduced by this plan.

## Next Phase Readiness
- Categorical on-device path is now dual-anchored (cpu f64 structure + real 4.6 golden fidelity) for both one-hot and many-vs-many, with predict-through verified.
- Follow-up (out of scope here): broaden on-device numeric-feature coverage (monotone/extra_trees/col_sampler) so more numeric cells can run under `LGBM_CUDA_ON_DEVICE=1`.

## Self-Check: PASSED
- FOUND: `crates/oracle-harness/tests/learner_parity.rs`
- FOUND: `crates/lgbm-compute/src/kernels/categorical_split.rs`
- FOUND commit `0566370` (fix)
- FOUND commit `edcba63` (test — Tasks 1 + 2)

---
*Phase: 22-on-device-categorical-splits-feature-coverage*
*Completed: 2026-07-02*
