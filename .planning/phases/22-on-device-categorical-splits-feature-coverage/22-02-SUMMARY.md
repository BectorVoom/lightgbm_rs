---
phase: 22-on-device-categorical-splits-feature-coverage
plan: 02
subsystem: treelearner
tags: [grow_driver, categorical, on-device, crate-cycle, oracle-harness]

# Dependency graph
requires:
  - phase: 22-01
    provides: runtime cat_width slab sizing + D-06 categorical+quantized host-fallback gate (no file overlap)
provides:
  - "GrowFeature extended with 6 native categorical fields (bin_to_category + cat_smooth/cat_l2/max_cat_threshold/max_cat_to_onehot/min_data_per_group)"
  - "grow_features_of (harness) populates the new fields from cfg + FeatureColumn.bin_to_category"
affects: [22-03, 22-04]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Additive struct extension across a crate-cycle seam using native primitives only (Vec<i32>/f64/i32) — never a lgbm-treelearner/lgbm-dataset type"
    - "Config-driven field population threaded via &GainConfig with GainConfig::default() fallback at call sites whose corpus builder returns no cfg"

key-files:
  created: []
  modified:
    - crates/lgbm-compute/src/kernels/grow_driver.rs
    - crates/oracle-harness/tests/learner_parity.rs
    - crates/lgbm-treelearner/src/learner.rs

key-decisions:
  - "Threaded cfg: &GainConfig into grow_features_of (vs inline defaults) to faithfully satisfy the must-have 'populate from cfg'; call sites without a cfg pass GainConfig::default()"
  - "Updated the lgbm-treelearner learner.rs GrowFeature construction site (Rule 3 blocking) since adding non-Default fields breaks the existing on-device fork construction"

patterns-established:
  - "GrowFeature categorical carriers are inert this milestone (empty bin_to_category, config-default scalars, unconsumed until 22-04) — numeric path byte-unchanged"

requirements-completed: [ODL-22]

# Metrics
duration: 8 min
completed: 2026-07-02
status: complete
---

# Phase 22 Plan 02: GrowFeature Categorical Metadata Extension Summary

**GrowFeature gains six native categorical fields (bin_to_category + five config scalars) as crate-cycle-safe carriers, plumbed through the oracle-harness helper and the learner fork, with the numeric grow spine byte-unchanged.**

## Performance

- **Duration:** 8 min
- **Started:** 2026-07-02
- **Completed:** 2026-07-02
- **Tasks:** 2
- **Files modified:** 3

## Accomplishments
- Extended `GrowFeature` (grow_driver.rs) with `bin_to_category: Vec<i32>`, `cat_smooth: f64`, `cat_l2: f64`, `max_cat_threshold: i32`, `max_cat_to_onehot: i32`, `min_data_per_group: i32` — all native primitives, no external crate type imported (crate-cycle-safe, RESEARCH A3). Updated the struct doc to describe the categorical carriers for the 22-04 grow branch (§6.3/§8.1).
- `grow_features_of` (harness) now populates the six new fields; threaded `cfg: &GainConfig` through the helper and all 8 call sites (in-scope cfg for the `corpus()` / `proving_slice_config()` sites, `GainConfig::default()` where the corpus builder returns no cfg).
- Numeric on-device structure gate and full workspace suite stay green — the new fields are inert until 22-04 consumes them.

## Task Commits

Each task was committed atomically:

1. **Task 1: Extend GrowFeature with additive categorical metadata** - `8e50a57` (feat) — includes the Rule 3 blocking fix to the learner.rs construction site
2. **Task 2: Populate the new fields in grow_features_of** - `c827921` (test)

## Files Created/Modified
- `crates/lgbm-compute/src/kernels/grow_driver.rs` - `GrowFeature` += six native categorical fields; doc comment updated (removed the "deliberately omits bin_to_category this milestone" note, replaced with the Phase-22 carrier note).
- `crates/oracle-harness/tests/learner_parity.rs` - `grow_features_of` extended to fill the six fields from `cfg` + `FeatureColumn.bin_to_category`; signature gains `cfg: &GainConfig`; all 8 call sites updated.
- `crates/lgbm-treelearner/src/learner.rs` - on-device-fork `GrowFeature` construction (learner.rs:794) sets the new fields from `self.cfg` + `FeatureColumn.bin_to_category` (Rule 3 blocking fix — see Deviations).

### The six new fields + numeric-path defaults
On the numeric path the fields are inert: `bin_to_category = Vec::new()` (empty), and the scalars take the config defaults — `cat_smooth = 10.0`, `cat_l2 = 10.0`, `max_cat_threshold = 32`, `max_cat_to_onehot = 4`, `min_data_per_group = 100`. In the harness these defaults arrive via `GainConfig::default()`; in the learner via `self.cfg` (whose defaults are the same config.h values). No `lgbm-treelearner`/`lgbm-dataset` type was imported into `lgbm-compute` — `cargo build -p lgbm-compute` green proves no crate cycle was introduced.

## Decisions Made
- **Thread `cfg` rather than inline defaults.** The plan allowed either, but the must-have specifies "populate from cfg". Threading `&GainConfig` keeps the harness faithful (categorical corpora will carry real sidecar scalars) and future-proofs 22-04. Call sites whose corpus builder returns no cfg (`on_device_proving_corpus`, `deep_multileaf_corpus`, `nosplit_corpus`, `mindata_corpus`) pass `GainConfig::default()`.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Updated the learner.rs GrowFeature construction site**
- **Found during:** Task 1 (GrowFeature extension)
- **Issue:** Adding six non-`Default` fields to `GrowFeature` breaks every construction site. Besides the harness (Task 2), `crates/lgbm-treelearner/src/learner.rs:794` constructs `GrowFeature` at the on-device fork. Without updating it, `cargo build -p lgbm-treelearner` and the plan-mandated `cargo test --workspace` (SC #4) would not compile. The plan's `files_modified` listed only the two files, but the workspace-green verification requires this site to be fixed.
- **Fix:** Populated the six new fields at learner.rs:794 — `bin_to_category: f.bin_to_category.clone()` (the spine `FeatureColumn` already carries it) and the five scalars from `self.cfg` (the learner's `GainConfig`). This is both the compile fix and the semantically correct population 22-04 will rely on. The construction sits inside the `on_device_eligible` block (DEAD with `LGBM_CUDA_ON_DEVICE` unset), so the host path stays byte-identical.
- **Files modified:** crates/lgbm-treelearner/src/learner.rs
- **Verification:** `cargo build -p lgbm-compute` + `cargo build -p oracle-harness --tests` + `cargo test --workspace` all green.
- **Committed in:** `8e50a57` (Task 1 commit)

---

**Total deviations:** 1 auto-fixed (1 blocking).
**Impact on plan:** Necessary for the crate to compile and for the SC #4 workspace-green gate. No scope creep — the fix is the natural companion of the struct extension and matches the field-population intent 22-04 needs.

## Issues Encountered
None.

## Verification Results
- `cargo build -p lgbm-compute` — success (no crate cycle; T-22-04 mitigated).
- `cargo test -p lgbm-compute --lib grow_driver` — `test result: ok` (0 grow_driver-named tests; ran clean, no errors).
- `cargo build -p oracle-harness --tests` — success against the extended `GrowFeature`.
- `LGBM_CUDA_ON_DEVICE=1 ... learner_parity_on_device_structure_gate` — `test result: ok. 1 passed` (numeric structure gate byte-unchanged; T-22-05 mitigated).
- `cargo test --workspace` (env unset) — all suites green, 0 failed (SC #4: numeric spine byte-unchanged).
- Acceptance greps: `bin_to_category` present in grow_driver.rs (2 hits) and learner_parity.rs (3 hits); all five scalar fields present in the struct.

## Next Phase Readiness
- `GrowFeature` is a settled struct carrying the categorical metadata; 22-03 (evaluator transcription reads the scalars) and 22-04 (wiring reads `bin_to_category` for SetRealThreshold) can build on it.
- Fields are inert until 22-04 wires the categorical branch — no consumer added this plan (by design).

## Self-Check: PASSED
- Modified files exist on disk: grow_driver.rs, learner_parity.rs, learner.rs — all present.
- Commits exist: `8e50a57` (Task 1), `c827921` (Task 2) — both in `git log`.
- All task acceptance criteria and plan-level verification re-run and green.

---
*Phase: 22-on-device-categorical-splits-feature-coverage*
*Completed: 2026-07-02*
