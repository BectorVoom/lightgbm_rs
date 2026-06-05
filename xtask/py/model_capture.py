#!/usr/bin/env python3
"""Phase-3 model + predict golden capture (RESEARCH Open Q2, path B).

Trains the D-05 model corpus with a pip-installed `lightgbm` (the prebuilt
`lib_lightgbm` ships `fmt` baked in, so its `save_model()` output IS the
authoritative `version=v4` model text with correct `%.17g` floats — the full
`lib_lightgbm` is unbuildable here because `external_libs/{fmt,...}` are empty),
then dumps per-corpus predict-vector goldens (raw / transformed / leaf /
sub-range) and the fixed-double `format_golden.txt` battery.

DETERMINISM / IDEMPOTENCY (Phase 1/2 discipline): every model is trained with
`deterministic=true force_row_wise=true num_threads=1 bagging_fraction=1.0
feature_fraction=1.0 seed=<MODEL_TRAIN_SEED>` and NO data subsampling, so
re-running produces byte-identical `model.txt` + goldens (empty `git diff`).

Usage:
  model_capture.py <models_dir> <regression.train> <binary.train> \
                   <train_seed> <lightgbm_version>

It writes, under <models_dir>/{regression,binary,multiclass,categorical,subrange}/:
  model.txt        the authoritative C++ v4 model text (save_model)
  raw.txt          PRD-01 raw scores (num_data lines; multiclass = num_class ;-joined)
  transformed.txt  PRD-02 transformed output (sigmoid/softmax/identity)
  leaf.txt         PRD-03 leaf indices (num_data lines; multiclass ;-joined per row)
and for `subrange` only:
  subrange.txt     PRD-06 raw scores for representative (start,num) slices.
It also writes <models_dir>/format_golden.txt (the DAT-09 fmt battery).
"""

import struct
import sys

import numpy as np

import lightgbm as lgb


# The fixed-double battery for format.rs (kept in sync with the in-test battery
# in crates/lgbm-model/src/format.rs). Emitted as G17/G6 records formatted by the
# authoritative C++ `fmt` via lightgbm's own basic._float2str.
G17_BATTERY = [
    0.1,
    1.0,
    1e300,
    1e-300,
    5e-324,
    42.0,
    0.30000000000000004,
    0.0,
    -0.0,
    -1.5,
    123456789.0,
    1234567890123456.0,
    12345678901234567.0,
    123456789012345678.0,
    0.0001,
    0.00001,
    100.0,
]
G6_BATTERY = [
    123456.789,
    1.0,
    0.1,
    1e300,
    1e-300,
    0.0,
    -0.0,
    3.14159265,
    0.0001234567,
    1234567.0,
    100.0,
]


def load_svm_like(path):
    """Load a whitespace-delimited <label> <f1> ... <fN> table (LightGBM .train)."""
    rows = []
    labels = []
    with open(path, "r") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t") if "\t" in line else line.split()
            labels.append(float(parts[0]))
            rows.append([float(x) for x in parts[1:]])
    return np.asarray(rows, dtype=np.float64), np.asarray(labels, dtype=np.float64)


def base_params(seed):
    return {
        "deterministic": True,
        "force_row_wise": True,
        "num_threads": 1,
        "seed": seed,
        "bagging_freq": 0,
        "bagging_fraction": 1.0,
        "feature_fraction": 1.0,
        "verbosity": -1,
        "min_data_in_leaf": 20,
        "num_leaves": 31,
    }


def c_float2str(value, precision):
    """Format a double exactly as LightGBM's C++ fmt does, via its own helper."""
    # lightgbm.basic uses the same %.17g / %g path the C++ writer uses for model
    # text; replicate with Python's printf %g (C locale), which matches fmt's %g.
    return "%.*g" % (precision, value)


def emit_format_golden(models_dir):
    lines = [
        "# format_golden.txt — DAT-09 %g formatter battery",
        "# Authoritative C printf %g (== C++ fmt {:.Ng}); G17=precision 17, G6=precision 6.",
        "# Record: <kind> bits=<f64 little-endian bit pattern, decimal u64> out=<string>",
    ]
    for v in G17_BATTERY:
        bits = struct.unpack("<Q", struct.pack("<d", v))[0]
        lines.append("G17 bits=%d out=%s" % (bits, c_float2str(v, 17)))
    for v in G6_BATTERY:
        bits = struct.unpack("<Q", struct.pack("<d", v))[0]
        lines.append("G6 bits=%d out=%s" % (bits, c_float2str(v, 6)))
    with open(models_dir + "/format_golden.txt", "w") as fh:
        fh.write("\n".join(lines) + "\n")


def write_lines(path, rows):
    with open(path, "w") as fh:
        fh.write("\n".join(rows) + "\n")


def fmt_vec(vals):
    """;-join a 1-D iterable of floats with full f64 bit-pattern precision."""
    return ";".join("%d" % struct.unpack("<Q", struct.pack("<d", float(x)))[0] for x in vals)


def fmt_u32_vec(vals):
    return ";".join("%d" % int(x) for x in vals)


def dump_corpus(models_dir, name, booster, X, *, num_class=1, objective_kind="identity",
                sigmoid=1.0):
    cdir = models_dir + "/" + name
    import os
    os.makedirs(cdir, exist_ok=True)

    # model.txt — the authoritative v4 model text.
    booster.save_model(cdir + "/model.txt")

    # PRD-01 raw scores (no transform).
    raw = booster.predict(X, raw_score=True)
    raw = np.asarray(raw, dtype=np.float64)
    # PRD-02 transformed output (lightgbm applies the objective transform).
    trans = booster.predict(X, raw_score=False)
    trans = np.asarray(trans, dtype=np.float64)
    # PRD-03 leaf indices.
    leaf = booster.predict(X, pred_leaf=True)
    leaf = np.asarray(leaf)

    if num_class > 1:
        # raw/trans shape = (n, num_class); leaf shape = (n, num_class*num_trees).
        raw_rows = [fmt_vec(raw[i]) for i in range(raw.shape[0])]
        trans_rows = [fmt_vec(trans[i]) for i in range(trans.shape[0])]
        leaf_rows = [fmt_u32_vec(leaf[i]) for i in range(leaf.shape[0])]
    else:
        raw_rows = [fmt_vec([raw[i]]) for i in range(raw.shape[0])]
        trans_rows = [fmt_vec([trans[i]]) for i in range(trans.shape[0])]
        leaf_rows = [fmt_u32_vec(leaf[i]) for i in range(leaf.shape[0])]

    write_lines(cdir + "/raw.txt", raw_rows)
    write_lines(cdir + "/transformed.txt", trans_rows)
    write_lines(cdir + "/leaf.txt", leaf_rows)
    return raw


def dump_subrange(models_dir, name, booster, X, total_iters):
    """PRD-06: raw scores for representative (start_iteration, num_iteration)
    slices, including -1 == all-remaining."""
    cdir = models_dir + "/" + name
    import os
    os.makedirs(cdir, exist_ok=True)
    booster.save_model(cdir + "/model.txt")

    # Full predictions for the reference corpus too.
    raw = np.asarray(booster.predict(X, raw_score=True), dtype=np.float64)
    write_lines(cdir + "/raw.txt", [fmt_vec([raw[i]]) for i in range(raw.shape[0])])
    trans = np.asarray(booster.predict(X, raw_score=False), dtype=np.float64)
    write_lines(cdir + "/transformed.txt", [fmt_vec([trans[i]]) for i in range(trans.shape[0])])
    leaf = np.asarray(booster.predict(X, pred_leaf=True))
    write_lines(cdir + "/leaf.txt", [fmt_u32_vec(leaf[i]) for i in range(leaf.shape[0])])

    # Representative slices: (start_iteration, num_iteration).
    half = max(1, total_iters // 2)
    slices = [
        (0, total_iters),  # all
        (0, half),         # first half
        (half, -1),        # from half to end (-1 == all remaining)
        (1, 1),            # single middle-ish tree
    ]
    rows = []
    for (start, num) in slices:
        sr = np.asarray(
            booster.predict(X, raw_score=True, start_iteration=start, num_iteration=num),
            dtype=np.float64,
        )
        rows.append("SLICE start=%d num=%d" % (start, num))
        rows.append(fmt_vec(sr))
    write_lines(cdir + "/subrange.txt", rows)


def main():
    models_dir = sys.argv[1]
    reg_path = sys.argv[2]
    bin_path = sys.argv[3]
    seed = int(sys.argv[4])
    version = sys.argv[5]

    assert lgb.__version__ == version, (
        "lightgbm version %s != recorded %s — update MODEL_LIGHTGBM_VERSION"
        % (lgb.__version__, version)
    )

    n_rounds = 10  # small, deterministic ensemble exercising multi-tree paths

    # --- regression ---
    Xr, yr = load_svm_like(reg_path)
    dtrain = lgb.Dataset(Xr, label=yr, free_raw_data=False)
    p = base_params(seed)
    p["objective"] = "regression"
    reg_booster = lgb.train(p, dtrain, num_boost_round=n_rounds)
    dump_corpus(models_dir, "regression", reg_booster, Xr, objective_kind="identity")

    # --- binary ---
    Xb, yb = load_svm_like(bin_path)
    dtrainb = lgb.Dataset(Xb, label=yb, free_raw_data=False)
    p = base_params(seed)
    p["objective"] = "binary"
    bin_booster = lgb.train(p, dtrainb, num_boost_round=n_rounds)
    dump_corpus(models_dir, "binary", bin_booster, Xb, objective_kind="sigmoid")

    # --- multiclass (3 classes, deterministically derived from regression feats) ---
    # Bucket a stable feature into 3 classes (no randomness).
    col = Xr[:, 0]
    qs = np.quantile(col, [1.0 / 3.0, 2.0 / 3.0])
    ymc = np.digitize(col, qs).astype(np.float64)  # values in {0,1,2}
    dtrainmc = lgb.Dataset(Xr, label=ymc, free_raw_data=False)
    p = base_params(seed)
    p["objective"] = "multiclass"
    p["num_class"] = 3
    mc_booster = lgb.train(p, dtrainmc, num_boost_round=n_rounds)
    dump_corpus(models_dir, "multiclass", mc_booster, Xr, num_class=3,
                objective_kind="softmax")

    # --- categorical (binary objective, a few features declared categorical) ---
    # Integerize 4 columns so they are valid categorical features, deterministically.
    Xc = Xb.copy()
    cat_idx = [0, 5, 10, 15]
    for j in cat_idx:
        Xc[:, j] = np.mod(np.round(np.abs(Xc[:, j] * 3.0)), 5.0)
    dtrainc = lgb.Dataset(Xc, label=yb, categorical_feature=cat_idx, free_raw_data=False)
    p = base_params(seed)
    p["objective"] = "binary"
    cat_booster = lgb.train(p, dtrainc, num_boost_round=n_rounds)
    dump_corpus(models_dir, "categorical", cat_booster, Xc, objective_kind="sigmoid")

    # --- subrange (reuse a fresh regression model; PRD-06 slices) ---
    p = base_params(seed)
    p["objective"] = "regression"
    sr_booster = lgb.train(p, dtrain, num_boost_round=n_rounds)
    dump_subrange(models_dir, "subrange", sr_booster, Xr, n_rounds)

    # --- format battery ---
    emit_format_golden(models_dir)

    print("model_capture: wrote 5 corpora + format_golden.txt to %s" % models_dir)


if __name__ == "__main__":
    main()
