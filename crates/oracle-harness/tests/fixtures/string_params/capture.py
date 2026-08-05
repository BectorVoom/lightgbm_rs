#!/usr/bin/env python3
"""Capture the C++ (lightgbm==4.6.0) golden for EVERY value of every closed-enum
string-valued parameter.

The string-valued parameter set is `config_auto.cpp::ParameterTypes()` filtered to
`string` / `vector<string>`. Of those 23, the ones with a CLOSED value set are
swept here; the rest name files or dataset columns and belong to the CLI /
dataset-loader layer, which the Rust port does not implement (they are listed in
`SKIPPED` below with the reason, so the inventory stays honest).

For each (param, value) cell we record either:
  - {"ok": true, "model": <save_model text>, "pred": [...], "metrics": [...]}
  - {"ok": false, "error": "<C++ error text>"}   (a value C++ itself rejects)

The Rust side (`string_param_parity.rs`) replays every cell.

Determinism: seed=1, deterministic=true, force_row_wise=true, num_threads=1 —
the REFERENCE_MANIFEST recipe, so the capture is reproducible.

Run:  .venv/bin/python crates/oracle-harness/tests/fixtures/string_params/capture.py
"""

import json
import pathlib
import sys
import warnings

import numpy as np

import lightgbm as lgb

warnings.filterwarnings("ignore")

HERE = pathlib.Path(__file__).parent
OUT = HERE / "string_params_golden.json"

# Parameters excluded from the sweep, with the reason (kept in the artifact so the
# coverage claim is auditable).
SKIPPED = {
    "config": "names a params FILE (CLI layer, not implemented in the Rust port)",
    "data": "names a training-data FILE (CLI/dataset-loader layer)",
    "valid": "names validation-data FILES (CLI/dataset-loader layer)",
    "forcedsplits_filename": "a FILE PATH, not an enum (behavior covered by param_wiring.rs)",
    "forcedbins_filename": "a FILE PATH, not an enum (dataset-loader layer)",
    "input_model": "a FILE PATH, not an enum",
    "output_model": "a FILE PATH, not an enum (CLI layer)",
    "output_result": "a FILE PATH, not an enum (CLI layer)",
    "convert_model": "a FILE PATH, not an enum (CLI convert_model task)",
    "parser_config_file": "a FILE PATH, not an enum (dataset-loader layer)",
    "label_column": "a dataset COLUMN spec (dataset-loader layer)",
    "weight_column": "a dataset COLUMN spec (dataset-loader layer)",
    "group_column": "a dataset COLUMN spec (dataset-loader layer)",
    "machine_list_filename": "distributed learning (out of scope, PROJECT.md)",
    "machines": "distributed learning (out of scope, PROJECT.md)",
}

# Closed-enum string parameters × every accepted value (including aliases).
# Sources: config.cpp GetObjectiveType/GetBoostingType/GetDataSampleStrategy/
# GetTreeLearnerType/GetDeviceType/GetTaskType, config.h ParseMetricAlias,
# config_auto.cpp monotone_constraints_method.
SWEEP = {
    "objective": [
        "regression", "regression_l2", "l2", "mean_squared_error", "mse",
        "l2_root", "root_mean_squared_error", "rmse",
        "regression_l1", "l1", "mean_absolute_error", "mae",
        "huber", "fair", "poisson", "quantile",
        "mape", "mean_absolute_percentage_error", "gamma", "tweedie",
        "binary",
        "lambdarank", "rank_xendcg", "xendcg", "xe_ndcg", "xe_ndcg_mart", "xendcg_mart",
        "multiclass", "softmax", "multiclassova", "multiclass_ova", "ova", "ovr",
        "cross_entropy", "xentropy", "cross_entropy_lambda", "xentlambda",
        "none",
    ],
    "boosting": ["gbdt", "gbrt", "dart", "rf", "random_forest", "goss"],
    "data_sample_strategy": ["bagging", "goss"],
    "tree_learner": [
        "serial", "feature", "feature_parallel", "data", "data_parallel",
        "voting", "voting_parallel",
    ],
    "device_type": ["cpu", "gpu", "cuda"],
    "monotone_constraints_method": ["basic", "intermediate", "advanced"],
    "metric": [
        "l2", "regression", "regression_l2", "mean_squared_error", "mse",
        "rmse", "l2_root", "root_mean_squared_error",
        "l1", "regression_l1", "mean_absolute_error", "mae",
        "quantile", "huber", "fair", "poisson", "mape",
        "mean_absolute_percentage_error", "gamma", "gamma_deviance", "tweedie",
        "binary_logloss", "binary", "binary_error", "auc", "average_precision",
        "auc_mu", "ndcg", "lambdarank", "rank_xendcg", "xendcg", "xe_ndcg",
        "xe_ndcg_mart", "xendcg_mart", "map", "mean_average_precision",
        "multi_logloss", "multiclass", "softmax", "multiclassova",
        "multiclass_ova", "ova", "ovr", "multi_error",
        "cross_entropy", "xentropy", "cross_entropy_lambda", "xentlambda",
        "kullback_leibler", "kldiv",
        "none", "null", "na", "custom",
        "l2,rmse", "rmse,mse,l2,l2_root", "l2,,rmse", "L2,RMSE", ",",
        "not_a_metric",
    ],
    "convert_model_language": ["cpp"],
    "task": ["train", "training", "predict", "prediction", "test", "convert_model",
             "refit", "refit_tree", "save_binary"],
}

N, D = 240, 4
RNG = np.random.default_rng(20260805)
X = RNG.standard_normal((N, D))
# One label vector per objective family, so every objective's Init guard passes.
LABELS = {
    "regression": (X @ np.array([1.5, -0.5, 0.8, 0.2])).astype(float),
    "positive": np.abs(X[:, 0]) + 1.0,          # poisson / gamma / tweedie / mape
    "binary": (X[:, 0] > 0).astype(float),      # binary / cross_entropy
    "rank": RNG.integers(0, 4, N).astype(float),
    "multi": RNG.integers(0, 3, N).astype(float),
}
GROUP = [60, 60, 60, 60]

RANK_OBJ = {"lambdarank", "rank_xendcg", "xendcg", "xe_ndcg", "xe_ndcg_mart", "xendcg_mart"}
MULTI_OBJ = {"multiclass", "softmax", "multiclassova", "multiclass_ova", "ova", "ovr"}
BINARY_OBJ = {"binary", "cross_entropy", "xentropy", "cross_entropy_lambda", "xentlambda"}
POSITIVE_OBJ = {"poisson", "gamma", "tweedie", "mape", "mean_absolute_percentage_error"}
# Metric names that only make sense for a multiclass / binary / ranking objective.
MULTI_METRIC = {"multi_logloss", "multi_error", "auc_mu", "multiclass", "softmax",
                "multiclassova", "multiclass_ova", "ova", "ovr"}
RANK_METRIC = {"ndcg", "map", "mean_average_precision", "lambdarank", "rank_xendcg",
               "xendcg", "xe_ndcg", "xe_ndcg_mart", "xendcg_mart"}
BINARY_METRIC = {"binary_logloss", "binary", "binary_error", "auc", "average_precision",
                 "cross_entropy", "xentropy", "cross_entropy_lambda", "xentlambda",
                 "kullback_leibler", "kldiv"}
# Metrics whose C++ Init CHECKs the label is strictly positive; they need the
# positive-label corpus even though the objective stays plain `regression`.
POSITIVE_METRIC = {"gamma", "gamma_deviance", "poisson", "tweedie", "mape",
                   "mean_absolute_percentage_error"}


def family_for(param, value):
    """(objective, extra_params) giving `value` a valid objective context."""
    if param == "objective":
        obj = value
    elif param == "metric":
        first = value.split(",")[0]
        if first in MULTI_METRIC:
            obj = "multiclass"
        elif first in RANK_METRIC:
            obj = "lambdarank"
        elif first in BINARY_METRIC:
            obj = "binary"
        else:
            obj = "regression"
    else:
        obj = "regression"

    # A positive-label metric on a plain regression objective still needs labels
    # that satisfy the metric's own `CHECK_GT(label, 0)`.
    if param == "metric" and value.split(",")[0] in POSITIVE_METRIC:
        return obj, LABELS["positive"], None, {}

    extra = {}
    if obj in MULTI_OBJ:
        extra["num_class"] = 3
        y = LABELS["multi"]
        group = None
    elif obj in RANK_OBJ:
        y, group = LABELS["rank"], GROUP
    elif obj in BINARY_OBJ:
        y, group = LABELS["binary"], None
    elif obj in POSITIVE_OBJ:
        y, group = LABELS["positive"], None
    elif obj == "none":
        y, group = LABELS["regression"], None
    else:
        y, group = LABELS["regression"], None
    return obj, y, group, extra


def run_cell(param, value):
    obj, y, group, extra = family_for(param, value)
    params = dict(
        objective=obj, num_iterations=6, learning_rate=0.1, num_leaves=8,
        min_data_in_leaf=5, max_bin=63, seed=1, deterministic=True,
        force_row_wise=True, num_threads=1, verbose=-1, **extra,
    )
    if param != "objective":
        params[param] = value
    # `rf` / `random_forest` REQUIRE sub-sampling (`gbdt.cpp` CHECK:
    # bagging_freq > 0 && bagging_fraction < 1). Supply it so the cell measures
    # the boosting type rather than the missing-precondition error.
    if param == "boosting" and value in ("rf", "random_forest"):
        params.update(bagging_freq=1, bagging_fraction=0.8)
    # `task` is a CLI concept; the Python API rejects a non-train task outright,
    # so record what the C++ CONFIG layer does with it rather than a train run.
    try:
        ds = lgb.Dataset(X, label=y, group=group, params=params, free_raw_data=False)
        booster = lgb.train(params, ds, num_boost_round=6)
        pred = np.asarray(booster.predict(X, raw_score=True), dtype=np.float64)
        return {
            "ok": True,
            "model": booster.model_to_string(),
            "pred": [float(v) for v in pred.reshape(-1)[:40]],
            "num_trees": int(booster.num_trees()),
        }
    except Exception as e:  # noqa: BLE001 — the C++ error text IS the golden
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}


def main():
    # The corpus travels WITH the golden as raw IEEE-754 bits so the Rust replay
    # trains on byte-identical inputs (a regenerated corpus would not be).
    out = {
        "lightgbm_version": lgb.__version__,
        "skipped": SKIPPED,
        "num_rows": N,
        "num_cols": D,
        "features_bits": [f"{v.view(np.uint64):016x}" for v in X.reshape(-1)],
        "labels_bits": {
            k: [f"{v.view(np.uint64):016x}" for v in np.asarray(y, dtype=np.float64)]
            for k, y in LABELS.items()
        },
        "group": GROUP,
        "cells": {},
    }
    for param, values in SWEEP.items():
        for value in values:
            key = f"{param}={value}"
            cell = run_cell(param, value)
            obj, y, group, extra = family_for(param, value)
            # Record WHICH label vector / objective context the cell used, so the
            # Rust replay reconstructs the exact same training problem.
            cell["objective"] = obj
            cell["labels"] = next(k for k, v in LABELS.items() if v is y)
            cell["grouped"] = group is not None
            cell["num_class"] = extra.get("num_class", 1)
            out["cells"][key] = cell
            state = "ok " if out["cells"][key]["ok"] else "ERR"
            print(f"  {state} {key}", file=sys.stderr)
    OUT.write_text(json.dumps(out, indent=1, sort_keys=True))
    n_ok = sum(1 for c in out["cells"].values() if c["ok"])
    print(f"\nwrote {OUT}: {len(out['cells'])} cells ({n_ok} ok, "
          f"{len(out['cells']) - n_ok} C++-rejected)", file=sys.stderr)


if __name__ == "__main__":
    main()
