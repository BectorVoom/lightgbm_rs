"""C++ LightGBM 4.6 (pip wheel) reference timing on the SAME synthetic data the
Rust bench_train.rs example generates. One reference point for the speed report
(quick task 260608-iwj). Single-threaded to match the Rust serial path.
"""

import time

import lightgbm as lgb
import numpy as np


def make_corpus(rows, features, bins):
    """Mirror bench_train.rs::make_corpus exactly (same integer hash + label)."""
    X = np.empty((rows, features), dtype=np.float64)
    y = np.empty(rows, dtype=np.float64)
    MASK = (1 << 64) - 1
    for row in range(rows):
        acc = 0.0
        for f in range(features):
            if row < bins:
                v = (row + f) % bins
            else:
                h = (row * 2_654_435_761 + ((f * 40_503 + 0x9E37_79B9) & MASK)) & MASK
                v = h % bins
            X[row, f] = float(v)
            acc += float(v) * (1.0 + (f % 5))
        y[row] = acc * 0.01
    return X, y


def time_train(X, y, reps=5):
    params = {
        "objective": "regression",
        "num_leaves": 31,
        "learning_rate": 0.1,
        "min_data_in_leaf": 20,
        "num_threads": 1,          # match the Rust serial learner
        "deterministic": True,
        "seed": 42,
        "verbosity": -1,
        "max_bin": 255,
    }
    ds = lgb.Dataset(X, label=y, free_raw_data=False)
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        lgb.train(params, ds, num_boost_round=100)
        ts.append(time.perf_counter() - t0)
    ts.sort()
    return ts[len(ts) // 2]


if __name__ == "__main__":
    sizes = [
        ("small ", 2_000, 12, 32),
        ("medium", 8_000, 30, 64),
        ("large ", 20_000, 50, 128),
    ]
    print("# lightgbm 4.6 (pip wheel) reference  (num_threads=1, iters=100, leaves=31)")
    print(f"{'size':<7} {'rows':>7} {'feat':>5} {'bins':>5}  {'train_median':>12}")
    for name, rows, feat, bins in sizes:
        X, y = make_corpus(rows, feat, bins)
        med = time_train(X, y)
        print(f"{name:<7} {rows:>7} {feat:>5} {bins:>5}  {med*1000:>10.1f}ms")
