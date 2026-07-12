#!/usr/bin/env python3
"""Generate C++ LightGBM (4.6.0) linear-tree goldens for Rust parity/speed checks.

For each case we emit, under OUT/<case>/:
  - params.json   : training params + shape metadata
  - X_train.csv   : training design matrix (row-major, %.17g)  [training cases]
  - y_train.csv   : training labels
  - model.txt     : C++ save_model() text (the authoritative linear model)
  - X_test.csv    : held-out design matrix for prediction parity
  - pred.csv      : C++ booster.predict(X_test)  (%.17g)
  - timing.json   : C++ training wall-clock (single-thread, deterministic)

The model side (load+predict) is validated with (model.txt, X_test, pred).
The training side is validated by re-training on (X_train, y_train, params) and
comparing to model.txt / pred.  Determinism: single thread, fixed seed.
"""
import json
import os
import sys
import time

import numpy as np
import lightgbm as lgb

OUT = os.environ.get(
    "LINEAR_GOLDEN_OUT",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "linear_goldens"),
)


def g17(a):
    a = np.asarray(a, dtype=np.float64).ravel()
    return "\n".join("%.17g" % v for v in a)


def mat17(M):
    M = np.asarray(M, dtype=np.float64)
    return "\n".join(" ".join("%.17g" % v for v in row) for row in M)


def emit(case, X, y, params, n_round, nan_test=False):
    d = os.path.join(OUT, case)
    os.makedirs(d, exist_ok=True)
    X = np.ascontiguousarray(X, dtype=np.float64)
    y = np.ascontiguousarray(y, dtype=np.float64)
    # KEY: pass params to Dataset so raw data is retained for linear leaves.
    ds = lgb.Dataset(X, label=y, params=params, free_raw_data=False)
    t0 = time.perf_counter()
    bst = lgb.train(params, ds, num_boost_round=n_round)
    dt = time.perf_counter() - t0
    bst.save_model(os.path.join(d, "model.txt"))

    # Held-out test matrix; include NaNs in a linear feature for the nan case.
    rng = np.random.RandomState(999)
    Xt = rng.rand(64, X.shape[1]).astype(np.float64)
    if nan_test:
        Xt[0, 0] = np.nan
        Xt[5, 1] = np.nan
        Xt[9, :] = np.nan
    pred = bst.predict(Xt)

    with open(os.path.join(d, "X_train.csv"), "w") as f:
        f.write(mat17(X))
    with open(os.path.join(d, "y_train.csv"), "w") as f:
        f.write(g17(y))
    with open(os.path.join(d, "X_test.csv"), "w") as f:
        f.write(mat17(Xt))
    with open(os.path.join(d, "pred.csv"), "w") as f:
        f.write(g17(pred))
    # Per-iteration raw score on the TRAIN set (score BEFORE tree ti), so a Rust
    # fit test can form the L2 gradient g = score_before_ti - y (h = 1) that C++
    # used to fit tree ti's linear leaves. score_before_ti = predict(num_iter=ti).
    sdir = os.path.join(d, "scores")
    os.makedirs(sdir, exist_ok=True)
    for ti in range(n_round):
        sb = bst.predict(X, start_iteration=0, num_iteration=ti)
        with open(os.path.join(sdir, f"score_before_{ti}.csv"), "w") as f:
            f.write(g17(sb))

    with open(os.path.join(d, "params.json"), "w") as f:
        json.dump({**params, "num_boost_round": n_round,
                   "n_train": int(X.shape[0]), "n_feat": int(X.shape[1])}, f, indent=2)
    with open(os.path.join(d, "timing.json"), "w") as f:
        json.dump({"cpp_train_seconds": dt, "n_round": n_round}, f, indent=2)

    nfeat_lines = [l for l in open(os.path.join(d, "model.txt")).read().splitlines()
                   if l.startswith("num_features")]
    populated = any(any(int(x) > 0 for x in l.split("=", 1)[1].split()) for l in nfeat_lines)
    print(f"[{case:22}] round={n_round} dt={dt:.3f}s "
          f"pred[:3]={np.round(pred[:3],5)} linear_populated={populated}")


def base(**over):
    p = dict(objective="regression", num_leaves=7, learning_rate=0.1,
             linear_tree=True, min_data_in_leaf=20, num_threads=1,
             verbose=-1, seed=0, deterministic=True, force_col_wise=True)
    p.update(over)
    return p


def main():
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.RandomState(1)

    # Compact corpora (committed fixtures): small enough for git, large enough to
    # populate per-leaf linear models (rows/leaf > min_data_in_leaf).

    # Case A: clean linear signal, several rounds -> populated coefficients.
    N, nf = 800, 4
    X = rng.rand(N, nf)
    y = X @ np.array([3.0, -2.0, 1.0, 0.5]) + 0.01 * rng.randn(N)
    emit("reg_clean", X, y, base(), n_round=5)

    # Case B: with L2 on the linear model (linear_lambda).
    emit("reg_lambda", X, y, base(linear_lambda=2.0), n_round=5)

    # Case C: NaN handling in prediction (train clean, predict with NaNs).
    emit("reg_nan_predict", X, y, base(), n_round=5, nan_test=True)

    # Case D: single round -> first tree is constant by design (num_features=0).
    emit("reg_one_round", X, y, base(), n_round=1)

    # Case E: more features + deeper trees.
    N2, nf2 = 1200, 8
    X2 = rng.rand(N2, nf2)
    y2 = X2 @ (rng.randn(nf2)) + 0.05 * rng.randn(N2)
    emit("reg_wide", X2, y2, base(num_leaves=15, learning_rate=0.05), n_round=8)

    print(f"\nGoldens written under {OUT}")


if __name__ == "__main__":
    main()
