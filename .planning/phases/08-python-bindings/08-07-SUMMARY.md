---
phase: 08-python-bindings
plan: 07
subsystem: python-bindings
tags: [python, sklearn, callbacks, cv, plotting, pyo3, lightgbm-parity]

# Dependency graph
requires:
  - phase: 08-06
    provides: "compiled _core (Dataset/Booster/train with fobj/feval), low-level lightgbm_rs surface"
provides:
  - "sklearn estimators: LGBMModel/LGBMRegressor/LGBMClassifier/LGBMRanker (PYB-03)"
  - "training-callback list protocol: early_stopping/log_evaluation/record_evaluation/reset_parameter (D-09)"
  - "lgb.cv pure-Python k-fold over _core.train (D-09)"
  - "plotting: plot_importance/plot_metric (functional) + plot_tree (pending D-10) (D-09)"
affects: [08-08, future model-persistence plan, dask/distributed]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Pure-Python orchestration over batch _core.train via incrementing num_boost_round (per-iteration callback loop without a Rust-side incremental booster)"
    - "_TrainedBooster wrapper carries Python-side metadata (best_iteration/params) the compiled #[pyclass] Booster cannot hold"
    - "Optional-dep plotting: matplotlib/graphviz imported lazily inside each function"

key-files:
  created:
    - crates/lgbm-python/python/lightgbm_rs/callback.py
    - crates/lgbm-python/python/lightgbm_rs/engine.py
    - crates/lgbm-python/python/lightgbm_rs/sklearn.py
    - crates/lgbm-python/python/lightgbm_rs/plotting.py
    - crates/lgbm-python/python/tests/test_sklearn_parity.py
    - crates/lgbm-python/python/tests/test_callbacks_cv.py
  modified:
    - crates/lgbm-python/python/lightgbm_rs/__init__.py

key-decisions:
  - "Public train() is the callback-aware engine.train (mirrors official lightgbm); _core.train kept as _core_train"
  - "Per-iteration loop re-trains the booster with incrementing num_boost_round (no _core incremental-update API exists); metrics evaluated in Python on transformed predictions"
  - "Engine train() takes valid_sets as (X,y) numpy tuples (pure-Python world has no shared Dataset reference)"
  - "best_iteration exposed 1-based (tree COUNT) to match official Booster.best_iteration"

patterns-established:
  - "Pattern: _TrainedBooster wrapper delegates predict/feature_importance/num_iteration/refit while holding best_iteration/best_score/params"
  - "Pattern: _ObjectiveFunctionWrapper/_EvalFunctionWrapper bridge sklearn func(y_true,y_pred[,w[,g]]) to the _core (preds, dataset_labels) fobj/feval contract"

requirements-completed: [PYB-03]

# Metrics
duration: 19min
completed: 2026-06-07
---

# Phase 8 Plan 07: Pure-Python sklearn / callbacks / cv / plotting Summary

**sklearn-style LGBMRegressor/Classifier/Ranker, the official callback list protocol (early_stopping/log_evaluation/record_evaluation/reset_parameter), lgb.cv, and feature-importance plotting — all pure-Python orchestration over the validated _core engine, A/B-matching real lightgbm 4.6 within 1e-6 (regressor + classifier).**

## Performance

- **Duration:** 19 min
- **Started:** 2026-06-07T23:36:08Z
- **Completed:** 2026-06-07T23:55:40Z
- **Tasks:** 3 completed
- **Files modified:** 7 (6 created, 1 modified)

## Accomplishments
- sklearn estimator hierarchy (PYB-03) matching the official API: `fit`/`predict`/`predict_proba`/`feature_importances_`/`classes_`/`n_classes_`/`n_features_in_`/`booster_`/`get_params`/`set_params`. Regressor and classifier A/B-match real lightgbm 4.6 to `atol=1e-6` (predictions and `predict_proba`).
- Training-callback list protocol (D-09): the four official factories with `order`/`before_iteration`, sorted and dispatched around each boosting round. `record_evaluation` history and `early_stopping` `best_iteration` both A/B-match real lightgbm (best-iteration predictions agree to 1e-6).
- `lgb.cv` k-fold orchestration returning the official `{"valid <metric>-mean"/"-stdv": [...]}` dict shape, with stratified/explicit-fold support.
- Plotting helpers with matplotlib/graphviz as lazily-imported optional deps; `plot_importance`/`plot_metric` functional, `plot_tree` raises a documented `NotImplementedError` pending model persistence (D-10).
- No new Rust: all layered on the existing `_core` (`Dataset`/`Booster`/`train`/`feature_importance`/`refit`).

## Task Commits

1. **Task 1: callback list protocol + lgb.cv** - `130fffb` (feat)
2. **Task 2: sklearn estimators + plotting** - `0621b4a` (feat)
3. **Task 3: sklearn + callbacks/cv A/B parity tests** - `9e9931d` (test; also folds the engine best_iteration/valid_sets fix)

## Files Created/Modified
- `python/lightgbm_rs/callback.py` (created) - CallbackEnv/EarlyStopException + early_stopping/log_evaluation/record_evaluation/reset_parameter, mirroring the official protocol (order, before_iteration).
- `python/lightgbm_rs/engine.py` (created) - pure-Python `train` (callback list over `_core.train`) + `cv` (k-fold) + metric computation (l2/rmse/l1/binary_logloss/multi_logloss/auc) + `_TrainedBooster`/`CVBooster`.
- `python/lightgbm_rs/sklearn.py` (created) - LGBMModel base + LGBMRegressor/LGBMClassifier/LGBMRanker + `_ObjectiveFunctionWrapper`/`_EvalFunctionWrapper`.
- `python/lightgbm_rs/plotting.py` (created) - plot_importance/plot_metric/plot_tree (matplotlib/graphviz optional).
- `python/lightgbm_rs/__init__.py` (modified) - re-export the high-level `train` (engine), `cv`, the four callbacks, the estimators, and the plotting helpers.
- `python/tests/test_sklearn_parity.py` (created) - estimator A/B parity.
- `python/tests/test_callbacks_cv.py` (created) - callbacks/cv/plotting A/B + behavior tests.

## How It Works (design note)

The compiled `_core.train(params, dataset, num_boost_round, fobj, feval) -> Booster` is a BATCH train; it does not surface per-iteration eval history nor an incremental booster-update API to Python. So the Python `engine.train` drives the boosting loop itself: each round it re-trains the booster with `num_boost_round = i+1`, predicts on the validation `(X,y)` tuples, computes the requested LightGBM metric(s) in numpy on the transformed predictions, builds the `evaluation_result_list`, and dispatches the callbacks (sorted by `order`, `before_iteration` first). `EarlyStopException` truncates the loop; the engine then retrains to the best iteration so `predict()` uses it. This is O(rounds²) trees-trained but is pure-Python orchestration with zero new numerical code, exactly as scoped by D-09 / RESEARCH §"callbacks list / cv / plotting → no new Rust".

The compiled `#[pyclass] Booster` cannot hold arbitrary Python attributes, so the engine returns a thin `_TrainedBooster` wrapper that carries `best_iteration`/`params`/`best_score` and delegates `predict`/`feature_importance`/`num_iteration`/`refit`.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] engine.train ignored the public `valid_sets` argument**
- **Found during:** Task 3 (record_evaluation / early_stopping tests raised KeyError / "no evaluation results")
- **Issue:** the engine only read the internal `_valid_data` channel (added for the sklearn path), so `lr.train(..., valid_sets=[(Xv,yv)])` had no eval sets and callbacks saw an empty `evaluation_result_list`.
- **Fix:** prefer `_valid_data` when provided, else fall back to the public `valid_sets`.
- **Files modified:** `engine.py`
- **Commit:** `9e9931d`

**2. [Rule 1 - Bug] best_iteration convention off-by-one vs official**
- **Found during:** Task 3 (early_stopping reported 59 vs reference 60)
- **Issue:** the callback `EarlyStopException.best_iteration` is 0-based; official `Booster.best_iteration` is the 1-based tree COUNT.
- **Fix:** retrain to `best_iteration+1` trees and expose `best_iteration` 1-based on `_TrainedBooster`.
- **Files modified:** `engine.py`
- **Commit:** `9e9931d`

## Known Limitations (documented, not bugs)

- **LGBMRanker lambdarank training is a `_core` gap.** The ranking objective exists in the Rust facade (`crates/lgbm-objective/src/rank.rs`) but the model-side `ObjectiveKind::parse` (`crates/lgbm-model/src/objective.rs`) used by `_core.train` only wires regression/binary/multiclass, so `objective="lambdarank"` raises `unsupported objective`. The `LGBMRanker` class + API surface are delivered and tested on a supported objective; the lambdarank A/B-parity test is `skipif`-guarded with a clear reason. Fixing requires new Rust (wire the ranking objective into the model objective layer) — out of scope for this pure-Python wrapper plan.
- **`lgb.cv` per-fold binning scope differs from real lightgbm.** Real lightgbm builds one `Dataset` and shares bin boundaries across folds (valid sets reference the full-data binning); this engine builds a fresh `_core.Dataset` per fold (binning scoped to each fold's training rows). The `valid l2-mean` therefore agrees in magnitude/trend (within ~1.5% on the tested data) but not to 1e-6. Bit-exact cv parity needs a `_core` subset/reference-binning capability — new Rust, out of scope. The cv test asserts structural + magnitude (rtol=0.05) + monotone-decreasing parity instead.
- **`plot_tree` pending D-10.** Tree rendering needs the C++-compatible model text dump (`model_to_string`), which lands with model persistence (D-10, a separate plan). `plot_tree` honours the matplotlib/graphviz optional-dep imports then raises a documented `NotImplementedError`.

## Authentication Gates

None.

## Verification

- `python -m pytest python/tests/` → **46 passed, 3 skipped** (existing 37 + 9 new; skips: lambdarank `_core` gap, matplotlib absent, graphviz absent — all clean importorskip/skipif).
- A/B vs real lightgbm 4.6: LGBMRegressor predict + LGBMClassifier predict_proba within `atol=1e-6`; record_evaluation l2 history within 1e-6; early_stopping same best_iteration + predictions within 1e-6.
- No Rust files changed → `cargo test --workspace` unaffected. `LightGBM/` reference tree NOT git-added; no `.venv/`/`target/`/`_core*.so` staged.

## Self-Check: PASSED

All created files verified on disk (callback.py, engine.py, sklearn.py, plotting.py, test_sklearn_parity.py, test_callbacks_cv.py, 08-07-SUMMARY.md) and all three task commits (`130fffb`, `0621b4a`, `9e9931d`) found in git history.
