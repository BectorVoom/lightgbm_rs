---
phase: 06-gbdt-spine-core-objectives-metrics
plan: 03
subsystem: boosting
tags: [objective, metric, regression_l1, binary, custom, percentile, auc, renew-tree-output, oracle, bit-exact]

# Dependency graph
requires:
  - phase: 06-02-gbdt-spine-vertical-slice
    provides: "Gbdt loop (TrainOneIter order, f64 ScoreUpdater, boost_from_average); renew_tree_output seam (no-op for L2); public TrainingBuilder/Booster/train/predict; boosting_oracle_capture spine pipeline + L1-L5 replay; bit-exact L2 precision contract"
  - phase: 06-01-wave-0-foundation
    provides: "lgbm-objective/metric/boosting/lgbm scaffolds + error boundaries; Objective enum; SerialTreeLearner::renew_tree_output Wave-0 seam (Option<closure>)"
  - phase: 05-tree-learner-split-finding
    provides: "SerialTreeLearner::train_returning_partition (bit-exact growth + DataPartition::indices_in_leaf)"
  - phase: 03-model-text-predict
    provides: "lgbm-model ObjectiveKind::convert + convert_binary (sigmoid ConvertOutput) re-used by metrics + predict"
provides:
  - "Objective::RegressionL1 (Sign grad, hess=1, median BoostFromScore via PercentileFun, is_renew_tree_output=true, renew_leaf_output = median residual)"
  - "lgbm-objective::percentile (PercentileFun/WeightedPercentileFun verbatim port of regression_objective.hpp:18-88)"
  - "lgbm-objective::Binary (sigmoid grad/hess + logit BoostFromScore w/ kEps clamp + class_need_train, binary_objective.hpp:105-177)"
  - "lgbm-objective::CustomObjective (D-04/OBJ-02 preds:&[f64]->( grad,hess) closure; wrong-length -> LengthMismatch, V5)"
  - "lgbm-metric::BinaryMetric {BinaryLogloss, BinaryError, Auc} (binary_metric.hpp:119-251; ConvertOutput-first; auc tie-order-invariant; factor +1/-1)"
  - "SerialTreeLearner::renew_tree_output BODY wired in the GBDT loop (l1 median residual = label[row]-train_score[offset+row], BEFORE shrinkage)"
  - "lgbm-boosting::BoostObjective dispatch enum (regression/regression_l1/binary/custom); custom forces boost_from_average OFF (C++ obj==null)"
  - "lgbm::train per-objective dispatch + per-objective eval metrics; lgbm::train_custom (D-04)"
  - "regression_l1 / binary / custom L1-L5 goldens + custom cross-anchor (native regression L2 bfa-off) on real lightgbm 4.6"
affects: [06-04-multiclass-metrics, 06-05-bagging-early-stopping, 08-pyo3-bindings]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "PercentileFun simplification: the C++ ArgMaxAtK quickselect + ArgMax/ArgMin reads resolve to (sorted_desc[pos-1], sorted_desc[pos]); a descending sort reproduces the exact median (ties give identical values) — proven bit-exact on the l1 renew goldens"
    - "renew_tree_output as an Option<Fn(i32,&[u32])->f64> seam (06-01) keeps the learner decoupled from lgbm-objective; the boosting layer supplies the residual_getter+median closure, filling the body without inverting the crate dependency"
    - "BoostObjective enum unifies regression/binary/custom behind the small loop op-set (boost_from_score/get_gradients/is_renew_tree_output/renew_leaf_output/boost_from_average_enabled) — wiring, not new math"
    - "Custom-objective cross-anchor: an L2-equivalent fobj (grad=score-label, hess=1) is bit-anchorable to the native regression(L2 bfa-off) model text, giving OBJ-02 a real-binary anchor without a distinct C++ custom objective"

key-files:
  created:
    - crates/lgbm-objective/src/percentile.rs
    - crates/lgbm-objective/src/binary.rs
    - crates/lgbm-objective/src/custom.rs
    - crates/lgbm-metric/src/binary.rs
    - crates/lgbm-boosting/src/objective.rs
    - crates/oracle-harness/tests/fixtures/boosting/regression_l1_{spine_model,spine_pred,scores,gh_iter1,gh_iterN,metrics}.txt
    - crates/oracle-harness/tests/fixtures/boosting/binary_{spine_model,spine_pred,scores,gh_iter1,gh_iterN,metrics}.txt
    - crates/oracle-harness/tests/fixtures/boosting/custom_{spine_model,spine_pred,scores,gh_iter1,gh_iterN,metrics}.txt
    - crates/oracle-harness/tests/fixtures/boosting/custom_crossanchor_l2_model.txt
    - crates/oracle-harness/tests/fixtures/boosting/REFERENCE_MANIFEST.md
  modified:
    - crates/lgbm-objective/src/lib.rs
    - crates/lgbm-objective/src/regression.rs
    - crates/lgbm-metric/src/lib.rs
    - crates/lgbm-boosting/src/gbdt.rs
    - crates/lgbm-boosting/src/lib.rs
    - crates/lgbm-treelearner/src/learner.rs
    - crates/lgbm/src/booster.rs
    - crates/lgbm/src/lib.rs
    - crates/oracle-harness/tests/boosting_parity.rs
    - crates/oracle-harness/tests/fixtures/REFERENCE_MANIFEST.md
    - xtask/py/boosting_oracle_capture.py
    - xtask/src/main.rs

key-decisions:
  - "RegressionL1 is a distinct Objective enum variant (not a flag on Regression) — clean match arms; PercentileFun lives in its own percentile.rs module shared by BoostFromScore + renew."
  - "PercentileFun ported as a descending-sort + index pick (proven equivalent to the C++ ArgMaxAtK quickselect for the percentile value) — simpler and bit-exact, validated on odd/even/extreme + the real l1 leaf goldens."
  - "DEVIATION (Rule 1) — custom-objective preds are &[f64], NOT the &[f32] RESEARCH D-04 specified. LightGBM 4.6's Python wrapper passes f64 preds to a custom objective (verified empirically); the f32 cast broke the bit-exact cross-anchor. Real-binary behavior is authoritative."
  - "Custom is NOT an Objective enum variant (the enum derives Clone/PartialEq, incompatible with a boxed Fn). Instead a BoostObjective dispatch enum in lgbm-boosting wraps regression/binary/custom; mirrors C++ objective_function_==nullptr for custom (bfa skipped)."
  - "Per-objective eval metrics in lgbm::train match the capture's metric= list: [l2,rmse] regression, [l1,l2,rmse] l1, [binary_logloss,binary_error,auc] binary, [l2] custom."

patterns-established:
  - "Layered per-objective golden replay: each of regression_l1/binary/custom replays L1 g/h (~1e-6), L2 per-iter scores (bit-exact f64), L5 model text + predict (bit-exact leaves) against real lightgbm 4.6 — the spine discipline widened one axis at a time (D-17)."
  - "regression_l1 leaf values are the median RESIDUAL (RenewTreeOutput), proven bit-exact AND distinct from the L2 Newton leaves (negative control) — Pitfall 2/3 closed."

requirements-completed: [OBJ-01, OBJ-02, OBJ-03, MET-01]

# Metrics
duration: ~50min
completed: 2026-06-07
---

# Phase 6 Plan 03: Core Objectives/Metrics breadth (regression_l1 + binary + custom + binary metrics) Summary

**Widened the proven 06-02 regression spine along the objective + metric axis (D-17 step 1): `regression_l1` (Sign grad + median `BoostFromScore` + the `RenewTreeOutput` median-residual leaf overwrite + a verbatim `PercentileFun`), `binary` (sigmoid grad/hess + logit init + sigmoid predict), the `custom` closure (D-04/OBJ-02, bfa forced off), and the binary metrics (`binary_logloss`/`binary_error`/`auc`) — each a thin, individually bit-exact-validated addition on the unchanged loop, replaying real `lib_lightgbm` 4.6 L1–L5 goldens; custom is additionally cross-anchored bit-exact to the native `regression`(L2) model text.**

## Performance

- **Duration:** ~50 min
- **Completed:** 2026-06-07
- **Tasks:** 3
- **Files:** 37 changed (10 created + 12 modified source/test + new + idempotent goldens)

## Accomplishments

- **Task 1 — regression_l1 + PercentileFun (`6811bdf`):** `percentile.rs` ports `PercentileFun`/`WeightedPercentileFun` verbatim (Q3, the highest-risk new numerical code), validated on odd/even/extreme medians. `Objective::RegressionL1` adds Sign grad + unit hess, the median `BoostFromScore` (NOT the mean), `is_renew_tree_output()==true`, and `renew_leaf_output` (median residual). The 06-01 `renew_tree_output` seam BODY is wired in the GBDT loop: per-leaf median of `label[row]-train_score[offset+row]`, computed on the pre-update train score, BEFORE shrinkage (gbdt.cpp:409-411 / serial_tree_learner.cpp:920-940).
- **Task 2 — binary + custom + binary metrics (`02e3829`):** `Binary` (verbatim sigmoid response grad/hess, logit-of-base-rate `BoostFromScore` with the `kEpsilon` clamp, `class_need_train`). `CustomObjective` (D-04 closure; wrong-length return → `ObjectiveError::LengthMismatch`, never a panic — T-06-03-01). `BinaryMetric::{BinaryLogloss, BinaryError, Auc}` (ConvertOutput-first via the re-used `lgbm_model::convert_binary`; AUC grouped-accumulation with a tie-order-invariant unstable sort; `factor_to_bigger_better` -1/+1). `BoostObjective` dispatch enum wires all three (+ regression) into the loop; custom forces `boost_from_average` OFF (C++ `obj==null`).
- **Task 3 — capture + layered replay (`1cec0d3`):** extended `boosting_oracle_capture.py` to capture regression_l1/binary/custom cells + the custom cross-anchor (native regression L2, bfa-off) on real `lightgbm==4.6.0`; generalized `lgbm::train` to dispatch the objective + per-objective eval metrics and added `lgbm::train_custom`; parametrized `boosting_parity.rs` per objective + the custom cross-anchor + the l1 median-residual renew (with a negative control). **12 passing / 2 ignored (06-05).**

## Numerical-fidelity result

- **regression_l1:** L5 model-text leaf values (the median RESIDUALS) replay **BIT-EXACT** vs the real binary; L2 per-iter scores bit-exact f64; L1 Sign g/h within ORACLE_TOL. The `regression_l1_renew_leaf_is_median_residual` negative control confirms the l1 leaves DIFFER from the L2 Newton leaves on the same corpus (the renew is load-bearing, Pitfall 2/3).
- **binary:** L5 model + sigmoid predict bit-exact / within tol; L2 scores bit-exact f64; L1 sigmoid g/h within tol; binary_logloss/binary_error/auc match.
- **custom:** L5 model + scores + g/h replay the captured custom golden; AND the custom model text bit-matches `custom_crossanchor_l2_model.txt` (native regression L2 bfa-off) on every tree leaf — OBJ-02 anchored to the real binary (same g/h ⇒ same trees ⇒ same model).

## Task Commits

1. **Task 1: regression_l1 + PercentileFun + renew body** — `6811bdf` (feat)
2. **Task 2: binary objective + custom closure + binary metrics + auc** — `02e3829` (feat)
3. **Task 3: capture + replay l1/binary/custom L1-L5 goldens** — `1cec0d3` (feat)

## Open Questions Resolved

- **Open-Q3 (PercentileFun exact algorithm):** RESOLVED — ported verbatim into `percentile.rs`, validated on the l1 `BoostFromScore` median + the per-leaf renew goldens. The C++ `ArgMaxAtK` quickselect + `ArgMax`/`ArgMin` reads resolve to a descending-sort index pick, reproduced bit-exact.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Custom-objective `preds` are `&[f64]`, not `&[f32]` (RESEARCH D-04)**
- **Found during:** Task 3 (the custom cross-anchor failed at tree 2 leaf 3, Δ ≈ 1.2e-8).
- **Issue:** RESEARCH D-04 said to pass the score "cast to f32 (mirror Python's f32 preds)". But LightGBM 4.6's Python wrapper actually hands a custom objective **f64** preds (verified empirically: `preds.dtype == float64`). Casting to f32 first did the grad subtraction in f32, diverging from the native L2 (which subtracts in f64) and breaking the bit-exact cross-anchor.
- **Fix:** `CustomObjective` (and `train_custom`) take `Fn(&[f64]) -> (Vec<f32>, Vec<f32>)`; the closure does the f64-subtract then f32-cast (the native L2 op order). Real-binary behavior is authoritative over the inferred default.
- **Files modified:** crates/lgbm-objective/src/custom.rs, crates/lgbm/src/booster.rs, crates/oracle-harness/tests/boosting_parity.rs
- **Verification:** `custom_objective` + `custom_cross_anchored_to_native_regression_l2` both pass bit-exact.
- **Committed in:** `1cec0d3` (Task 3) — the custom.rs signature change rode with Task 3 since that is where the cross-anchor surfaced it.

**2. [Rule 3 - Blocking] `Custom` is a `BoostObjective` dispatch variant, not an `Objective` enum variant**
- **Found during:** Task 2.
- **Issue:** `Objective` derives `Clone`/`PartialEq` (used by the builder/tests); a boxed `Fn` cannot. Putting custom on `Objective` would force dropping those derives across the crate.
- **Fix:** added a `BoostObjective<'a>` dispatch enum in `lgbm-boosting` (Builtin/Binary/Custom) carrying the small loop op-set; `Gbdt` gained a lifetime + `with_objective`. Mirrors C++ `objective_function_` (nullptr for custom). No churn to the `Objective` derives.
- **Files modified:** crates/lgbm-boosting/src/objective.rs (new), gbdt.rs, lib.rs, crates/lgbm/src/booster.rs
- **Verification:** `cargo build`/`test` green; bfa forced off for custom verified by the cross-anchor (init=0 matches native bfa-off).
- **Committed in:** `02e3829` (Task 2).

**Total deviations:** 2 (1 Rule-1 bug fix, 1 Rule-3 blocking design adaptation). No scope creep; all plan acceptance gates satisfied.

## Authentication / Capture Gates

- The capture requires a `lightgbm==4.6.0` pip wheel. The recorded capture venv (`/tmp/lgbm-capture-venv`) was available, so the capture ran in-flow (not a blocking gate). Version asserted before training (T-06-03-SC); `cargo test` reads only committed goldens (no wheel). `LightGBM/` was never `git add`ed. Capture is byte-idempotent (verified across two runs).

## Verification

- `cargo test --workspace` GREEN (0 failures): lgbm-objective (percentile/l1/binary/custom), lgbm-metric (regression + binary/auc), lgbm-treelearner (renew_tree_output Some/None/single-leaf), lgbm-boosting, lgbm, oracle-harness `boosting_parity` (12 passed / 2 ignored for 06-05), `learner_parity` (12, unregressed).
- Acceptance grep gates: `common.h` in percentile.rs; `binary_objective.hpp` in binary.rs; `binary_metric.hpp` in metric binary.rs; CMP-01 (`grep cubecl crates/lgbm-boosting/Cargo.toml` empty).
- Capture idempotent; regression spine goldens unchanged (06-02 bytes preserved).

## Known Stubs

- The `WeightedPercentileFun` path is ported but not exercised by an in-scope corpus (the spine/l1 corpora are unweighted); it is validated by a unit test only. Weighted regression_l1 is a later-wave surface. Documented, not blocking.
- `BoostObjective::renew_leaf_output` returns `0.0` for Binary/Custom (unreachable — guarded by `is_renew_tree_output()==false`); a defensive no-op, not a stub that affects output.

## Next Phase Readiness

- 06-04 (multiclass) is UNBLOCKED: the objective/metric breadth (regression, regression_l1, binary + their metrics + custom) is complete on the single-output spine; the `BoostObjective` dispatch + the per-objective capture/replay pipeline generalize to per-class trees.
- 06-05 inherits the bit-exact L2 contract + the per-objective golden cells; `early_stopping`/`bagging_rng` stay `#[ignore]`d as the named seams.
- No blockers. CMP-01 holds; `LightGBM/` never git-added.

## Self-Check: PASSED

All key created files exist on disk; all 3 task commit hashes are present in git history (verified below).
