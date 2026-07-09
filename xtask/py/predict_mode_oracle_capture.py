#!/usr/bin/env python3
"""Phase-7 (plan 07-10, PRD-04 + PRD-05) REAL lib_lightgbm predict-mode capture.

Models the Phase-3 `model_capture.py` posture (the real pip `lib_lightgbm` 4.6
`save_model()` IS the authoritative v4 model text) for the TWO new prediction
modes:

  - PRD-04 TreeSHAP feature contributions: `booster.predict(X, pred_contrib=True)`
    returns, per row, a `(num_features + 1)`-wide block PER class — the per-feature
    contributions followed by the expected-value base — and the block SUMS to the
    raw margin (the load-bearing invariant the Rust gate asserts). For multiclass
    the blocks are concatenated class-major: `[class0 (nf+1), class1 (nf+1), ...]`.

  - PRD-05 prediction early stopping: `booster.predict(X, raw_score=True,
    pred_early_stop=True, pred_early_stop_freq=F, pred_early_stop_margin=M)` returns
    the raw score frozen at the early-stop point. We dump the score under a small
    freq×margin axis so the Rust `predict_raw_early_stop` final score can be
    compared within ORACLE_TOL.

Per corpus we dump under <out_dir>/<name>/:
  model.txt       authoritative v4 model text (save_model)
  X.txt           the predict input matrix: one row per line, f64 bit patterns
                  (space-separated), preceded by a `# rows=<n> cols=<m>` header
  contrib.txt     pred_contrib output: one row per line, f64 bit patterns
                  (space-separated), preceded by `# rows=<n> width=<w>`
  early_stop.txt  (binary + multiclass only) the freq×margin axis: per cell a
                  `CELL freq=<f> margin=<m>` line then a row-per-line f64-bit dump
                  of the raw scores

DETERMINISM / IDEMPOTENCY: every model trains with `deterministic=true
force_row_wise=true num_threads=1 seed=<seed> bagging_fraction=1.0
feature_fraction=1.0` and no subsampling => byte-identical re-runs.

NEVER `git add` the LightGBM/ tree; goldens land only under the tracked
`crates/oracle-harness/tests/fixtures/predict_modes/`.

Usage:
  predict_mode_oracle_capture.py <out_dir> <seed> <lightgbm_version>
"""

import os
import struct
import sys

import numpy as np

import lightgbm as lgb


def f64bits(v):
    return struct.unpack("<Q", struct.pack("<d", float(v)))[0]


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
        "min_data_in_leaf": 5,
        "num_leaves": 7,
    }


def write_matrix(path, mat, header_key):
    mat = np.asarray(mat, dtype=np.float64)
    if mat.ndim == 1:
        mat = mat.reshape(-1, 1)
    rows, cols = mat.shape
    lines = ["# rows=%d %s=%d" % (rows, header_key, cols)]
    for r in range(rows):
        lines.append(" ".join("%d" % f64bits(mat[r, c]) for c in range(cols)))
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")


def dump_predict_modes(out_dir, name, booster, X, *, do_early_stop):
    cdir = os.path.join(out_dir, name)
    os.makedirs(cdir, exist_ok=True)

    booster.save_model(os.path.join(cdir, "model.txt"))
    write_matrix(os.path.join(cdir, "X.txt"), X, "cols")

    # PRD-04: TreeSHAP contributions (block width (nf+1) per class, sums to raw).
    contrib = np.asarray(booster.predict(X, pred_contrib=True), dtype=np.float64)
    write_matrix(os.path.join(cdir, "contrib.txt"), contrib, "width")

    if do_early_stop:
        # PRD-05: freq×margin axis. freq in {10,1}, margin in {10.0,2.0} (RESEARCH).
        cells = [(10, 10.0), (1, 10.0), (1, 2.0)]
        lines = []
        for (freq, margin) in cells:
            raw = np.asarray(
                booster.predict(
                    X,
                    raw_score=True,
                    pred_early_stop=True,
                    pred_early_stop_freq=freq,
                    pred_early_stop_margin=margin,
                ),
                dtype=np.float64,
            )
            if raw.ndim == 1:
                raw = raw.reshape(-1, 1)
            rows, cols = raw.shape
            lines.append("CELL freq=%d margin=%g rows=%d width=%d" % (freq, margin, rows, cols))
            for r in range(rows):
                lines.append(" ".join("%d" % f64bits(raw[r, c]) for c in range(cols)))
        with open(os.path.join(cdir, "early_stop.txt"), "w") as fh:
            fh.write("\n".join(lines) + "\n")


def main():
    out_dir = sys.argv[1]
    seed = int(sys.argv[2])
    version = sys.argv[3]

    assert lgb.__version__ == version, (
        "lightgbm version %s != recorded %s — update "
        "PREDICT_MODE_ORACLE_LIGHTGBM_VERSION" % (lgb.__version__, version)
    )
    os.makedirs(out_dir, exist_ok=True)

    rng = np.random.RandomState(seed & 0x7FFFFFFF)

    # --- numeric regression (binary-class predict shape; ntpi=1) ---
    Xn = rng.rand(80, 4)
    yn = Xn[:, 0] * 3.0 + Xn[:, 1] * 2.0 - Xn[:, 2] + rng.rand(80) * 0.1
    dn = lgb.Dataset(Xn, label=yn, free_raw_data=False)
    p = base_params(seed)
    p["objective"] = "regression"
    bn = lgb.train(p, dn, num_boost_round=15)
    dump_predict_modes(out_dir, "numeric", bn, Xn[:10], do_early_stop=True)

    # --- categorical regression (TreeSHAP over categorical decision nodes) ---
    # 2 numeric + 2 categorical columns (small cardinality, integerized).
    Xc = rng.rand(80, 4)
    Xc[:, 2] = np.mod(np.round(Xc[:, 2] * 5.0), 5.0)
    Xc[:, 3] = np.mod(np.round(Xc[:, 3] * 4.0), 4.0)
    yc = Xc[:, 0] * 2.0 + (Xc[:, 2] == 3.0).astype(float) * 4.0 + rng.rand(80) * 0.1
    dc = lgb.Dataset(Xc, label=yc, categorical_feature=[2, 3], free_raw_data=False)
    p = base_params(seed)
    p["objective"] = "regression"
    bc = lgb.train(p, dc, num_boost_round=15)
    # Re-predict on a slice; contrib over categorical nodes must still sum to raw.
    dump_predict_modes(out_dir, "categorical", bc, Xc[:10], do_early_stop=False)

    # --- multiclass (3 classes; per-class contrib block + multiclass early stop) ---
    Xm = rng.rand(90, 4)
    ym = np.digitize(Xm[:, 0], np.quantile(Xm[:, 0], [1.0 / 3.0, 2.0 / 3.0])).astype(float)
    dm = lgb.Dataset(Xm, label=ym, free_raw_data=False)
    p = base_params(seed)
    p["objective"] = "multiclass"
    p["num_class"] = 3
    bm = lgb.train(p, dm, num_boost_round=15)
    dump_predict_modes(out_dir, "multiclass", bm, Xm[:10], do_early_stop=True)

    print("predict_mode_oracle_capture: wrote numeric/categorical/multiclass to %s" % out_dir)


if __name__ == "__main__":
    main()
