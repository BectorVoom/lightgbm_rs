---
phase: 07-parity-completing-variants
plan: 12
subsystem: boosting
tags: [refit, feature-importance, gbdt, model-text, oracle-parity, lightgbm]

# Dependency graph
requires:
  - phase: 07-07
    provides: RF/boosting variants on gbdt.rs (into_model, num_init_iteration accounting)
  - phase: 07-10
    provides: predict-mode parity + the GbdtModel predict path importance reuses
  - phase: 03-tree-model-model-text-i-o-predict-parity
    provides: model_text load/save (Phase-3 model I/O reused by refit + continue-training)
provides:
  - "Tree::refit_leaf + set_leaf_output (FitByExistingTree decay blend) + GbdtModel::refit_one_tree"
  - "Gbdt::refit (GBDT::RefitTree mirror) leaf-refit driver + Gbdt::with_loaded_model/num_init_iteration continue-training"
  - "GbdtModel::feature_importance_gain + feature_importance_split_count_guarded (split_gain>0 CR-02 guard)"
  - "model_text::save_with_importance(model, type) — saved_feature_importance_type split/gain emit"
  - "builder setters refit_decay_rate / input_model / saved_feature_importance_type"
  - "xtask advanced-oracle-capture + advanced_parity.rs (5 cells GREEN vs real lib_lightgbm 4.6)"
affects: [phase-07-verification, refit, feature-importance, end-of-phase-gate]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Leaf-refit decay blend inlined in the model layer (no kernel-layer dep edge)"
    - "RefitTree as a Gbdt driver: per-iter Boosting() -> per-tree leaf-refit -> AddScore"
    - "Capture-gated advanced_parity cells skip-pass until the wheel golden is committed"

key-files:
  created:
    - crates/oracle-harness/tests/advanced_parity.rs
    - xtask/py/advanced_oracle_capture.py
    - crates/oracle-harness/tests/fixtures/advanced/ (real lib_lightgbm 4.6 goldens)
  modified:
    - crates/lgbm-model/src/tree.rs
    - crates/lgbm-model/src/ensemble.rs
    - crates/lgbm-model/src/model_text.rs
    - crates/lgbm-boosting/src/gbdt.rs
    - crates/lgbm/src/builder.rs
    - xtask/src/main.rs
    - crates/lgbm-core/tests/config_validation.rs

key-decisions:
  - "Inlined the CalculateSplittedLeafOutput leaf-output formula in lgbm-model (no new lgbm-compute dependency edge), citing the kernel-layer canonical fn"
  - "save() defaults to importance_type=0 (split); save_with_importance(model, type) carries the saved_feature_importance_type selector"
  - "Split-count emit switched to the CR-02 split_gain>0-guarded variant (C++ applies the >0 guard for BOTH split and gain); legacy unguarded feature_importance_split_count retained"
  - "refit RESETS the score buffer (C++ RefitTree re-scores from the refit trees); continue-training (with_loaded_model) re-scores the loaded ensemble then appends"

patterns-established:
  - "Two-slice refit (RESEARCH Open Q3): leaf-refit reuses Phase-3 load; continue-training reuses num_init_iteration accounting"
  - "Gain importance accumulated in f64 from stored f32 split_gain, truncated to size_t for the model-text block (C++ static_cast<size_t>)"

requirements-completed: [ADV-06, ADV-07]

# Metrics
duration: 19min
completed: 2026-06-07
---

# Phase 7 Plan 12: Refit / Continue-Training + Feature Importance Summary

**Leaf-decay refit (`refit_decay_rate`) + `input_model` continue-training (ADV-06) and gain+split feature importance with the `split_gain>0` guard (ADV-07), all bit-exact / within-1e-4 vs real lib_lightgbm 4.6 — the final additive plan of the phase.**

## Performance

- **Duration:** ~19 min
- **Started:** 2026-06-07T09:42:26Z
- **Completed:** 2026-06-07T10:01:30Z
- **Tasks:** 4 (3 auto + 1 capture checkpoint, satisfied autonomously — wheel present)
- **Files modified:** 16 (7 source + 1 test + 1 capture script + 6 goldens + .gitkeep)

## Accomplishments
- **ADV-06 refit (two slices):** `Tree::refit_leaf` (FitByExistingTree decay blend `decay*old + (1-decay)*shrunk_newton`), `Gbdt::refit` (the `GBDT::RefitTree` per-iteration driver), and `Gbdt::with_loaded_model` + `num_init_iteration` continue-training that re-scores the loaded ensemble and appends.
- **ADV-07 importance:** `GbdtModel::feature_importance_gain` (sum split_gain per feature) + `feature_importance_split_count_guarded`, both applying the CR-02 `split_gain>0` guard; `model_text::save_with_importance(model, type)` selects split(0)/gain(1) for the `feature_importances:` block (size_t-truncated, descending stable-sort).
- **Builder + capture:** `refit_decay_rate` / `input_model` / `saved_feature_importance_type` setters; `xtask advanced-oracle-capture` (+ python script) dumping real lib_lightgbm 4.6 base/refit(0.9,0.0)/continue/importance goldens (byte-idempotent, version-pinned 4.6.0).
- **Parity:** `advanced_parity.rs` 5 cells GREEN — importance split + gain bit-exact vs C++ `feature_importance`; refit leaf values within 1e-4 (f32-vs-f64 refit accumulation); continue-training tree-count growth from the loaded base.

## Task Commits

1. **Task 1: Refit (leaf-decay + continue-training)** - `9e2a6f9` (feat) + `d7818e2` (feat, the `refit_one_tree` model primitive landed with the ADV-07 file)
2. **Task 2: Gain importance + saved_feature_importance_type emit** - `d7818e2` (feat)
3. **Task 3: Builder + advanced capture + advanced_parity.rs** - `499ae00` (feat)
4. **Task 4: Capture real-binary goldens** - `0fd10dd` (test)

_TDD tasks 1-2: behavior was driven test-first (Tree/ensemble unit tests RED→GREEN before the parity capture). The shared `ensemble.rs` file carries both the Task-1 refit primitive and the Task-2 gain importance, so its single commit (`d7818e2`) spans both._

## Files Created/Modified
- `crates/lgbm-model/src/tree.rs` - `refit_leaf` (decay blend, inlined leaf-output formula) + `set_leaf_output`
- `crates/lgbm-model/src/ensemble.rs` - `feature_importance_gain`, `feature_importance_split_count_guarded`, `refit_one_tree`
- `crates/lgbm-model/src/model_text.rs` - `save_with_importance(model, type)`; `save` now uses the guarded split count
- `crates/lgbm-boosting/src/gbdt.rs` - `Gbdt::refit` (RefitTree driver), `with_loaded_model`, `num_init_iteration`, `refit_one_tree_inplace`
- `crates/lgbm/src/builder.rs` - `refit_decay_rate` / `input_model` / `saved_feature_importance_type` setters + tests
- `crates/lgbm-core/tests/config_validation.rs` - refit_decay_rate [0,1] CHECK + resolution tests
- `xtask/src/main.rs` + `xtask/py/advanced_oracle_capture.py` - `advanced-oracle-capture` subcommand + capture script
- `crates/oracle-harness/tests/advanced_parity.rs` - 5 capture-gated refit/importance/continue cells
- `crates/oracle-harness/tests/fixtures/advanced/*` - real lib_lightgbm 4.6 goldens (base, refit 0.9/0.0, continue, importance, sidecar)

## Decisions Made
- **No new dependency edge for the leaf-output math:** `Tree::refit_leaf` inlines the `CalculateSplittedLeafOutput<USE_L1,false,false>` formula (a 4-line pure numeric fn) rather than adding `lgbm-compute` as a dependency of `lgbm-model`, citing the canonical kernel fn for fidelity. (Avoided an architectural layering change — would have been a Rule 4 checkpoint.)
- **Guarded split-count is now the model-text emit:** C++ `FeatureImportance` applies the `split_gain>0` guard for BOTH importance types; the model-text `feature_importances:` block switched to the guarded count to stay byte-faithful. The legacy unguarded `feature_importance_split_count` is retained for callers wanting the raw structural count.
- **refit resets the score; continue re-scores then appends:** mirrors the two distinct C++ paths (`RefitTree` re-accumulates from the refit trees; continue-training via `ResetTrainingData` re-scores the loaded ensemble, then `num_init_iteration_` indexing appends).

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Model-text split-count emit missing the CR-02 `split_gain>0` guard**
- **Found during:** Task 2 (gain importance + model-text emit)
- **Issue:** The existing `feature_importances:` block used the UNGUARDED `feature_importance_split_count`, but C++ `GBDT::FeatureImportance` filters splits on `split_gain > 0` for type 0 (split) as well as type 1 (gain). A model with a zero-gain split would emit a divergent split count.
- **Fix:** Added `feature_importance_split_count_guarded` and switched the model-text emit to it; kept the unguarded fn for non-parity callers.
- **Files modified:** crates/lgbm-model/src/ensemble.rs, crates/lgbm-model/src/model_text.rs
- **Verification:** `save_with_gain_truncates_and_guards` unit test + the `importance_split_matches_real_binary` parity cell (bit-exact vs C++); all prior oracle-harness round-trip cells still GREEN (0 failed).
- **Committed in:** `d7818e2`

---

**Total deviations:** 1 auto-fixed (1 bug — parity correctness).
**Impact on plan:** The guard fix was required for byte-faithful importance parity. No scope creep; the legacy unguarded path is preserved.

## Issues Encountered
- **Refit driver design:** C++ `RefitTree` recomputes gradients per refit-iteration against an accumulating score (not a single pass). Reproduced faithfully as a `Gbdt::refit` loop (per-iter `Boosting()` → per-tree leaf-refit → `AddScore`), with the score buffer reset first. The capture-gated parity cell confirmed the leaf values match real `Booster.refit` within 1e-4.

## User Setup Required
None - no external service configuration required. (The capture wheel `lightgbm==4.6.0` was already present at `/tmp/lgbm-capture-venv`; `cargo test` never needs it — the goldens are committed.)

## Next Phase Readiness
- **Phase 7 additive work is COMPLETE.** All 12 plans executed; this was the last additive plan before the single end-of-phase verification gate (D-01).
- Workspace GREEN: 680 passed / 0 failed / 17 ignored (13 DEF-07-02 + 4 DEF-07-11 — unchanged; no new deferrals).
- No new divergence/knife-edge surfaced (refit + importance reproduced cleanly), so no new DEF-07 entry was needed.
- LightGBM/ remains untracked (never git-added); goldens are byte-idempotent.

## Threat Flags
None — `input_model` deserialization (T-07-12-01) reuses the Phase-3-validated `model_text::load` bounds-checking (typed `ModelError`, never panic); `refit_decay_rate` (T-07-12-02) is surfaced as a typed `[0,1]` CHECK; the capture wheel (T-07-12-SC) is version-asserted and `LightGBM/` is never git-added. No new trust-boundary surface introduced.

## Self-Check: PASSED

All claimed files exist on disk; all 4 task commits (`9e2a6f9`, `d7818e2`, `499ae00`, `0fd10dd`) are present in git history.

---
*Phase: 07-parity-completing-variants*
*Completed: 2026-06-07*
