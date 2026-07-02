---
phase: 22-on-device-categorical-splits-feature-coverage
plan: 01
subsystem: treelearner
tags: [cubecl, categorical-splits, on-device, quantized-grad, device-split-info, config-guardrails]

# Dependency graph
requires:
  - phase: 14-scaffold-oracle-slice-0
    provides: "DeviceSplitInfo SoA pre-allocated split-record with reserved categorical slabs (MAX_CAT_PER_SPLIT)"
  - phase: 20-on-device-driver-integration
    provides: "on_device_eligible gate + cuda_on_device_env() env AND-gate in the learner"
provides:
  - "Runtime categorical slab width (D-03): DeviceSplitInfo.cat_width read once from config.max_cat_threshold at new (default MAX_CAT_PER_SPLIT=32), no silent truncation, allocate-once"
  - "on_device_eligible_gate() D-06 host-fallback: categorical && use_quantized_grad routes to host, mirroring the CUDA reference asm(\"trap;\") non-support"
  - "with_quantized_grad builder + use_quantized_grad field on SerialTreeLearner"
affects: [22-02, 22-03, 22-04, 22-05, on-device-categorical-bitset, on-device-categorical-eval]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Pre-allocated slab, runtime width: a compile-time cap becomes a runtime field read once at construction; SoA layout invariant to width, only slab length changes (D-03)"
    - "Config-boundary host-fallback gate lives in the learner (where use_quantized_grad + per-feature bin_type are visible), not lgbm-compute (whose on_device_growth_supported takes no config args) — RESEARCH A4"

key-files:
  created: []
  modified:
    - crates/lgbm-compute/src/kernels/split_info.rs
    - crates/lgbm-treelearner/src/learner.rs
    - crates/lgbm-compute/tests/split_info.rs

key-decisions:
  - "cat_width is a new usize field + max_cat_threshold param on DeviceSplitInfo::new; MAX_CAT_PER_SPLIT retained as pub const usize = 32 (the DEFAULT, not a hard cap)"
  - "D-06 gate applied in the learner via refresh_on_device_eligibility() called from with_features/with_quantized_grad (setup-time, compute-once — preserves the D-05 'not per-train' property); the new-site base gate is unchanged"
  - "use_quantized_grad threaded into the learner as a new field + with_quantized_grad builder (GBDT does not yet call it — see Deferred)"

patterns-established:
  - "Pattern 2 (RESEARCH): pre-allocated slab, runtime width for config-tunable device caps"
  - "Pure gate helper (on_device_eligible_gate) tested deterministically across the truth table, no env manipulation"

requirements-completed: [ODL-22]

# Metrics
duration: ~18min
completed: 2026-07-02
status: complete
---

# Phase 22 Plan 01: Categorical Device-Path Config Guardrails Summary

**Runtime-width categorical slab sizing in `DeviceSplitInfo` (D-03) plus a D-06 categorical+quantized on-device host-fallback gate in the learner — both additive, both byte-unchanged with `LGBM_CUDA_ON_DEVICE` unset.**

## Performance

- **Duration:** ~18 min
- **Completed:** 2026-07-02
- **Tasks:** 2
- **Files modified:** 3 (2 source + 1 integration-test caller fix)

## Accomplishments
- **D-03 runtime slab width:** `MAX_CAT_PER_SPLIT` demoted to the DEFAULT constant (still `pub const usize = 32`); `DeviceSplitInfo` gained a `cat_width: usize` field read once from a new `max_cat_threshold` param on `new`. Every indexing/length site now uses the runtime width — slab sizing + `checked_mul` overflow guard (T-22-01), the `set_cat_thresholds` length guard (T-22-02), the base index, both accessors, and the `copy_slot` window. A `max_cat_threshold = 40` config is faithful with NO silent truncation and NO per-split alloc.
- **D-06 host-fallback gate:** `on_device_eligible_gate(base, has_categorical_feature, use_quantized_grad) = base && !(cat && quantized)` mirrors the CUDA reference `asm("trap;")` non-support of the combo. Wired into the learner via a new `use_quantized_grad` field + `with_quantized_grad` builder + `refresh_on_device_eligibility()` (called from `with_features`/`with_quantized_grad`), with a one-shot host-fallback log that never fires when the env is unset.
- Two non-vacuous unit tests (width>32 + the three-case eligibility truth table); full workspace green with the env unset (SC #4).

## Task Commits

1. **Task 1: Runtime cat_width slab sizing (D-03)** — `e165546` (feat)
2. **Task 1 follow-up: integration-test caller fix (Rule 3)** — `335bf11` (test)
3. **Task 2: D-06 categorical+quantized host-fallback gate** — `170e429` (feat)

_Plan metadata commit follows this summary._

## Files Created/Modified
- `crates/lgbm-compute/src/kernels/split_info.rs` — Added `cat_width` field + `max_cat_threshold` param on `new`; threaded the runtime width through slab sizing (`checked_mul` guard intact), `set_cat_thresholds` guard/base, both accessors, `copy_slot`; added `cat_width()` accessor + `cat_slab_width_gt_32_no_truncation` and `cat_slab_default_width_is_32_and_guards_at_32` tests.
- `crates/lgbm-treelearner/src/learner.rs` — Added `on_device_eligible_gate()` free fn; `use_quantized_grad` + `cat_quant_fallback_logged` fields; `with_quantized_grad` builder; `refresh_on_device_eligibility()` (called from `with_features`/`with_quantized_grad`) with a one-shot host-fallback log; `on_device_eligible_false_for_categorical_plus_quantized` test.
- `crates/lgbm-compute/tests/split_info.rs` — Updated the 9 `DeviceSplitInfo::new` call sites to pass the default `MAX_CAT_PER_SPLIT` (behavior-preserving).

## Decisions Made
- **`new` signature:** `DeviceSplitInfo::new(client, num_leaf_slots, max_cat_threshold)` — the third arg is the runtime slab width; callers pass `MAX_CAT_PER_SPLIT` for the historical default. There were no production callers of `new` (only structural integration tests), so no runtime path changed.
- **D-06 placement:** the categorical+quantized negation cannot live at the `new` site (features are empty there and `use_quantized_grad` is not yet known), so it is (re)applied in `refresh_on_device_eligibility()` from the builders. This preserves the "compute once, NOT per-train" property (D-05) while making `has_categorical_feature` and `use_quantized_grad` visible. The base gate at `new` is unchanged.
- **`has_categorical_feature` source:** `self.features.iter().any(|f| f.bin_type == BinType::Categorical)` — the spine's per-feature `FeatureColumn::bin_type`.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Updated `DeviceSplitInfo::new` integration-test callers**
- **Found during:** Task 2 workspace verification (`cargo test --workspace`)
- **Issue:** The D-03 signature change (added `max_cat_threshold` param) broke 9 call sites in `crates/lgbm-compute/tests/split_info.rs` — the plan's `read_first` had noted "update all existing callers" but the initial caller scan excluded the `split_info.rs` filename, hiding the integration-test file.
- **Fix:** Passed the default `MAX_CAT_PER_SPLIT` at every call site (behavior-preserving).
- **Files modified:** `crates/lgbm-compute/tests/split_info.rs`
- **Verification:** `cargo test --workspace` green (env unset).
- **Committed in:** `335bf11`

---

**Total deviations:** 1 auto-fixed (1 blocking).
**Impact on plan:** Necessary to keep the build green after the signature change; behavior-preserving, no scope creep.

## Issues Encountered
- None beyond the caller fix above.

## Deferred / Follow-up (out of this plan's file scope)
- **GBDT wiring of `with_quantized_grad`:** `crates/lgbm-boosting/src/gbdt.rs` builds the learner and knows `config.use_quantized_grad`, but does not yet call `.with_quantized_grad(...)`. The learner field therefore defaults `false` in production today. This is safe: the on-device path is env-gated OFF (`LGBM_CUDA_ON_DEVICE` unset) and `grow_tree_on_device` returns `Ok(None)` in Slice 0, so no categorical+quantized on-device run can occur regardless. Wiring belongs to the downstream categorical-path plans (22-02..05) that activate the on-device categorical route; it was left out here because this plan's `files_modified` is scoped to `split_info.rs` + `learner.rs`. The gate + test infrastructure this plan delivers is what those plans consume.

## Verification
- `cargo test -p lgbm-compute --lib split_info` — green incl. `cat_slab_width_gt_32_no_truncation` (width 40: 33/40 fit, 41 errors) and default-width-32 guard.
- `cargo test -p lgbm-compute --lib` — 115 passed.
- `cargo test -p lgbm-treelearner --lib` — 78 passed incl. `on_device_eligible_false_for_categorical_plus_quantized` (three-case truth table).
- `cargo test --workspace` (env unset) — all green; numerical spine byte-unchanged (SC #4).
- `MAX_CAT_PER_SPLIT` preserved as `pub const usize = 32`; `checked_mul` overflow guard intact in `new`.
- `cargo clippy` on both crates: no new warnings attributable to the changed code (pre-existing warnings only).

## Merge Gate
With `LGBM_CUDA_ON_DEVICE` unset: the D-03 default callers pass `MAX_CAT_PER_SPLIT` (identical slab sizing), and the D-06 base gate is `false` so `on_device_eligible_gate` returns `false`, the host-fallback log never fires, and the numerical spine / per-leaf host path is byte-identical to master. Confirmed green.

## Next Phase Readiness
- D-03 faithful slab sizing is ready for the §6.3 bitset writes into the slabs (SC #1).
- D-06 gate is ready for the categorical device path; downstream plans should wire `with_quantized_grad` from GBDT when they activate the on-device categorical route.

## Self-Check: PASSED

- `crates/lgbm-compute/src/kernels/split_info.rs` — FOUND (cat_width threaded; tests pass)
- `crates/lgbm-treelearner/src/learner.rs` — FOUND (on_device_eligible_gate + builder; test passes)
- `crates/lgbm-compute/tests/split_info.rs` — FOUND (callers updated)
- Commit `e165546` — FOUND
- Commit `335bf11` — FOUND
- Commit `170e429` — FOUND

---
*Phase: 22-on-device-categorical-splits-feature-coverage*
*Completed: 2026-07-02*
