---
status: complete
phase: 07-parity-completing-variants
source:
  - 07-01-SUMMARY.md
  - 07-02-SUMMARY.md
  - 07-03-SUMMARY.md
  - 07-04-SUMMARY.md
  - 07-05-SUMMARY.md
  - 07-06-SUMMARY.md
  - 07-07-SUMMARY.md
  - 07-08-SUMMARY.md
  - 07-09-SUMMARY.md
  - 07-10-SUMMARY.md
  - 07-11-SUMMARY.md
  - 07-12-SUMMARY.md
mode: standard-library-uat
started: 2026-06-07T11:19:41Z
updated: 2026-06-07T11:26:00Z
---

## Current Test

[testing complete]

## Tests

### 1. Cold Start Smoke Test (full workspace build + test)
expected: From a clean tree, `cargo test --workspace` builds every crate with zero compile warnings; suite reports 680 passed / 0 failed / 17 ignored (13 DEF-07-02 + 4 DEF-07-11). Command: `LGBM_CAPTURE_PYTHON=/tmp/lgbm-capture-venv/bin/python cargo test --workspace`
result: pass
note: "First run FAILED (blocker) — boosting_parity::early_stopping errored Metric(MulticlassLabelOutOfRange { label: 10.0, num_class: 3 }), a regression from the WR-03 code-review fix. Root cause: WR-03 added a hard out-of-range multiclass-label rejection, but C++ multiclass_metric.hpp LossOnPoint indexes ref_score[static_cast<size_t>(label)] with NO bounds check, and the early_stopping cell deliberately feeds constant valid label 10.0 (num_class=3) to plateau the metric — mirroring boosting_oracle_capture.py. WR-03 was a false positive; reverted in commit ed52c5b (restored pre-fix floor/clamp). Re-run: 682 passed / 0 failed / 17 ignored, zero warnings."

### 2. Boosting Variants — GOSS / DART / Random Forest (BST-04/05/06)
expected: `cargo test -p oracle-harness --test boosting_parity` is GREEN. GOSS trains bit-exact leaf values across top_rate×other_rate×{es}×{bfa} plus the RNG-replay golden; DART matches across uniform_drop×xgboost_dart_mode×{bag} plus its drop RNG-replay; RF (mandatory bagging, averaged trees) matches model+pred goldens — all vs real lib_lightgbm 4.6.
result: pass
evidence: "boosting_parity GREEN within `cargo test --workspace`: goss_parity_matrix, dart_parity_matrix, rf_single_parity, rf_multi_parity all ok; sample_strategy RNG-replay goldens (goss/bagging) ok."

### 3. Categorical Splits (TRL-06)
expected: `cargo test -p oracle-harness --test learner_parity` categorical cells are GREEN — one-hot and many-vs-many SplitCategorical produce matching category bitsets, gains, and model-text round-trip (max_cat_threshold / cat_smooth / min_data_per_group / max_cat_to_onehot / cat_l2).
result: pass
evidence: "lgbm-treelearner 64/0 (learner categorical: learner_grows_a_categorical_split, feature_histogram_categorical onehot/many-vs-many) + lgbm-dataset categorical_folding golden ok within workspace run."

### 4. Regression Objectives (OBJ-04: huber/fair/poisson/quantile/mape/gamma/tweedie)
expected: `cargo test -p oracle-harness --test boosting_parity` regression-objective cells are GREEN. Gradients/hessians and supported learner-level cells match the reference; the documented DEF-07-02 deferrals (5 fair, 4 gamma, quantile-bagged, tweedie-bfa-off) remain `#[ignore]`'d with intact bit-exact assertions — 13 ignores total, exactly as contracted.
result: pass
evidence: "boosting_parity 60 passed / 13 ignored — supported huber/poisson/mape/quantile-spine cells GREEN; the 13 DEF-07-02 deferrals (fair, gamma, quantile bagged+iterated, tweedie bfa-off) #[ignore]'d exactly as contracted."

### 5. Cross-Entropy Objectives (OBJ-05)
expected: cross_entropy and cross_entropy_lambda objective parity cells are GREEN vs lib_lightgbm 4.6 (gradients/hessians + trained model).
result: pass
evidence: "boosting_parity cross_entropy + cross_entropy_lambda gradients/score/spine/loop cells all ok."

### 6. Ranking Objectives + Ranking Metrics (OBJ-06 / MET-04)
expected: `cargo test -p oracle-harness --test rank_parity` is GREEN — lambdarank and rank_xendcg (query boundaries + DCGCalculator + objective_seed) match; per-query ndcg/map match; bagging-by-query and rank_xendcg objective_seed RNG-replay goldens match.
result: pass
evidence: "lgbm-metric rank + lgbm-objective rank suites GREEN (lambdarank/rank_xendcg, ndcg/map, rank_xendcg objective_seed + bagging_by_query RNG-replay)."

### 7. Extended Metrics (MET-03: auc / average_precision / auc_mu / ...)
expected: `cargo test -p oracle-harness --test metric_parity` is GREEN — extended metrics match per-query/aggregate with correct tie-invariance (AUC/AP).
result: pass
evidence: "lgbm-metric 60/0 — auc tie-invariance, average_precision, auc_mu, multi_error/auc_mu all ok within workspace run."

### 8. SHAP Contributions + Prediction Early Stopping (PRD-04 / PRD-05)
expected: `cargo test -p oracle-harness --test predict_parity` is GREEN — predict_contrib reproduces full node/cover SHAP structure; pred_early_stop / _freq / _margin produce C++-matching outputs. (Includes the CR-01 RF average_output fix round-trip.)
result: pass
evidence: "lgbm-model 96/0 incl. predict::contrib_* SHAP (sum+base==raw), pred_early_stop binary/multiclass margin, and transformed_rf_average_output_divides_by_num_iteration (CR-01 fix)."

### 9. Constraints & Tree Controls (ADV-01..05: monotone / interaction / forced splits/bins / extra trees / CEGB)
expected: `cargo test -p oracle-harness --test advanced_parity` constraint cells are GREEN — monotone (basic/intermediate/advanced + monotone_penalty), interaction constraints, forced splits/bins, extra trees, and CEGB reproduce C++ behavior. The one DEF-07-11 monotone mixed-vector last-ULP cell stays `#[ignore]`'d (structure bit-exact, ~1.4e-17 leaf drift).
result: pass
evidence: "advanced_parity + lgbm-treelearner learner: monotone_constraint_alters_chosen_tree, interaction_constraint_restricts_features, forced_split_drives_root_feature, extra_trees_is_deterministic_per_seed, cegb_penalty_changes_split_selection ok; 4 DEF-07-11 cells #[ignore]'d."

### 10. Refit / Continue-Training + Feature Importance (ADV-06/07)
expected: `cargo test -p oracle-harness --test advanced_parity` refit/importance cells are GREEN — Booster.refit() and continue-training reproduce C++ outputs; split/gain feature importance reporting matches.
result: pass
evidence: "advanced_parity 5/0: importance_split/importance_gain_matches_real_binary, refit_decay00/decay09_matches_real_binary, continue_training_grows_from_base all ok."

## Summary

total: 10
passed: 10
issues: 0
pending: 0
skipped: 0
note: Test 1 failed on first run (WR-03 regression), fixed in-session (revert ed52c5b); all 10 GREEN on re-run.

## Gaps

# Resolved in-session (not carried forward):
- truth: "cargo test --workspace passes with 0 failures"
  status: resolved
  reason: "WR-03 code-review fix introduced a parity regression (multiclass OOB-label rejection broke boosting_parity::early_stopping). Reverted in commit ed52c5b after confirming C++ tolerates OOB labels (unchecked index) and the fixture relies on it."
  severity: blocker
  test: 1
  fix_commit: ed52c5b
