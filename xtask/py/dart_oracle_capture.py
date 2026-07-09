#!/usr/bin/env python3
"""Phase-7 W5 REAL lib_lightgbm DART (BST-05) oracle capture (plan 07-06).

Captures TWO kinds of golden for the DART boosting variant:

  (1) MODEL PARITY cells — `dart_u{U}_x{X}_bag{B}_model.txt` + `_pred.txt`. Trains the
      `boosting=dart` axis (uniform_drop × xgboost_dart_mode × {bag} — the 4 normalize
      branches × bagging) on the real prebuilt `lib_lightgbm` 4.6 pip wheel and dumps
      the authoritative `%.17g` model text + per-row predictions. DART bakes the
      normalized tree weights into the stored leaf values (its Shrinkage sequence in
      DroppingTrees + Normalize), so a wrong normalize branch or wrong drop set shifts
      the leaves and FAILS the parity replay.

  (2) DROP RNG-REPLAY golden — `dart_drop_seed4_iter12.txt`. The dropped tree indices
      PER ITERATION that `DART::DroppingTrees` (dart.hpp:97-147) produces, derived over
      the EXACT C++ `Random` LCG (ported below bit-for-bit from
      `include/LightGBM/utils/random.h`). The wheel does not expose the internal drop
      indices, so — exactly like the GOSS `goss_rng_replay` / bagging `bag_indices_*`
      goldens — the golden FREEZES the algorithm spec over the bit-exact LCG; the Rust
      `dart_reference_drop` independently reproduces it (boosting_parity
      `dart_drop_rng_replay`). Each line carries the per-iter `tree_weight` history +
      `sum_weight` (as f64 bits) so the replay re-derives the drop draws itself.

      The tree_weight evolution mirrors dart.hpp exactly: each iter pushes the new
      tree's `shrinkage_rate_` (unless uniform_drop), and Normalize rescales each
      dropped tree's weight by `k/(k+1)` (or the xgboost `k/(k+lr)`). This is a pure
      function of (drop_seed, drop_rate, max_drop, skip_drop, uniform_drop,
      xgboost_dart_mode, learning_rate, num_iterations) — no real binary needed for the
      RNG-replay, identical posture to GOSS.

DETERMINISM / IDEMPOTENCY: trained with `deterministic=true force_row_wise=true
num_threads=1 seed=<seed>`; both kinds of golden are pure functions of the pinned
inputs, so re-running produces byte-identical files (empty `git diff`).

NEVER `git add` the LightGBM/ tree; goldens land only under the tracked
`crates/oracle-harness/tests/fixtures/dart/`.

Usage:
  dart_oracle_capture.py <out_dir> <seed> <drop_seed> <lightgbm_version>
"""

import os
import struct
import sys

import numpy as np

import lightgbm as lgb

# ---- DART capture control (mirrors boosting_parity.rs DART_* constants) ----
NUM_ITERATIONS = 12
NUM_LEAVES = 4
LEARNING_RATE = 0.1
# The drop RNG-replay golden uses the cell defaults (drop_rate 0.1, max_drop 50,
# skip_drop 0.5). The parity matrix varies uniform_drop × xgboost_dart_mode × {bag}.
DROP_RATE = 0.1
MAX_DROP = 50
SKIP_DROP = 0.5
# Bagging axis (matches the boosting matrix MATRIX_BAGGING_* constants).
BAGGING_FRACTION = 0.7
BAGGING_FREQ = 1
BAGGING_SEED = 3


def f64_bits(value):
    return struct.unpack("<Q", struct.pack("<d", float(value)))[0]


# =====================================================================
# Bit-for-bit port of the C++ `Random` LCG (include/LightGBM/utils/random.h)
# — the SAME recurrence the Rust `lgbm_core::random::Random` mirrors. Used ONLY
# to derive the drop RNG-replay golden (a deterministic, non-cryptographic PRNG).
# =====================================================================
class Random:
    def __init__(self, seed):
        self.x = seed & 0xFFFFFFFF

    def _rand_int16(self):
        self.x = (214013 * self.x + 2531011) & 0xFFFFFFFF
        return (self.x >> 16) & 0x7FFF

    def next_float(self):
        # static_cast<float>(RandInt16()) / 32768.0f — round to f32 exactly.
        return struct.unpack("<f", struct.pack("<f", self._rand_int16() / 32768.0))[0]


def dropping_trees(rng, drop_rate, max_drop, skip_drop, uniform_drop, it, tree_weight,
                   sum_weight):
    """Verbatim port of DART::DroppingTrees's DRAW order (dart.hpp:97-128). Returns the
    dropped tree indices. `rng` is the SINGLE advancing Random (not re-seeded)."""
    drop = []
    if float(rng.next_float()) < skip_drop:
        return drop
    rate = drop_rate
    if not uniform_drop:
        inv_avg = len(tree_weight) / sum_weight
        if max_drop > 0:
            rate = min(rate, max_drop * inv_avg / sum_weight)
        for i in range(it):
            if float(rng.next_float()) < rate * tree_weight[i] * inv_avg:
                drop.append(i)
                if len(drop) >= max_drop:
                    break
    else:
        if max_drop > 0:
            rate = min(rate, max_drop / float(it))
        for i in range(it):
            if float(rng.next_float()) < rate:
                drop.append(i)
                if len(drop) >= max_drop:
                    break
    return drop


def capture_drop_rng_replay(out_dir, drop_seed):
    """Write the DART drop RNG-replay golden: for the cell defaults (drop_rate 0.1,
    max_drop 50, skip_drop 0.5, uniform_drop false, xgboost_dart_mode false), the
    dropped tree indices PER ITERATION the dart.hpp DroppingTrees produces over the
    bit-exact C++ LCG, with the tree_weight/sum_weight evolution mirroring Normalize.
    The Rust dart_reference_drop reproduces this bit-exact."""
    uniform_drop = False
    xgboost_dart_mode = False
    lr = LEARNING_RATE
    rng = Random(drop_seed)
    tree_weight = []
    sum_weight = 0.0
    lines = [
        "# DART (BST-05) drop RNG-replay golden (plan 07-06).",
        "#",
        "# The dropped tree indices PER ITERATION that DART::DroppingTrees",
        "# (dart.hpp:97-147) produces over the EXACT C++ Random LCG (random.h), with the",
        "# tree_weight/sum_weight evolution mirroring Normalize (dart.hpp:158-197). The",
        "# wheel cannot expose the internal drop indices, so this freezes the algorithm",
        "# spec over the bit-exact LCG (identical posture to the GOSS goss_rng_replay).",
        "# The Rust dart_reference_drop reproduces it bit-exact",
        "# (boosting_parity::dart_drop_rng_replay). tree_weight/sum_weight are f64 bit",
        "# patterns; dropped indices i32 (or `-` for an empty set).",
        "#",
        "# drop_seed=<S> drop_rate=<R> max_drop=<M> skip_drop=<SD> uniform_drop=<0|1> "
        "iter=<I> sum_weight=<f64 bits> tree_weight=<csv f64 bits | -> dropped=<csv i32 | ->",
    ]
    for it in range(NUM_ITERATIONS):
        # DroppingTrees runs at the START of iter `it` over the `it` trees grown so far.
        tw_snapshot = list(tree_weight)
        sw_snapshot = sum_weight
        drop = dropping_trees(rng, DROP_RATE, MAX_DROP, SKIP_DROP, uniform_drop, it,
                              tw_snapshot, sw_snapshot)
        # shrinkage_rate_ for the new tree (dart.hpp:138-146).
        k = float(len(drop))
        if not xgboost_dart_mode:
            shrinkage_rate = lr / (1.0 + k)
        elif not drop:
            shrinkage_rate = lr
        else:
            shrinkage_rate = lr / (lr + k)
        # Normalize rescales each dropped tree's weight (dart.hpp:173-176, non-xgboost).
        for i in drop:
            if not uniform_drop:
                sum_weight -= tree_weight[i] * (1.0 / (k + 1.0))
                tree_weight[i] *= (k / (k + 1.0))
        # push the new tree's weight (dart.hpp:67) unless uniform_drop.
        if not uniform_drop:
            tree_weight.append(shrinkage_rate)
            sum_weight += shrinkage_rate
        tw_csv = (",".join(str(f64_bits(w)) for w in tw_snapshot)
                  if tw_snapshot else "-")
        drop_csv = ",".join(str(i) for i in drop) if drop else "-"
        lines.append(
            f"drop_seed={drop_seed} drop_rate={DROP_RATE} max_drop={MAX_DROP} "
            f"skip_drop={SKIP_DROP} uniform_drop={int(uniform_drop)} iter={it} "
            f"sum_weight={f64_bits(sw_snapshot)} tree_weight={tw_csv} dropped={drop_csv}"
        )
    with open(os.path.join(out_dir, "dart_drop_seed4_iter12.txt"), "w") as fh:
        fh.write("\n".join(lines) + "\n")


def base_params(seed, drop_seed, uniform_drop, xgboost_dart_mode, bag):
    p = {
        "objective": "regression",
        "metric": ["l2"],
        "boosting": "dart",
        "drop_rate": DROP_RATE,
        "max_drop": MAX_DROP,
        "skip_drop": SKIP_DROP,
        "uniform_drop": bool(uniform_drop),
        "xgboost_dart_mode": bool(xgboost_dart_mode),
        "drop_seed": drop_seed,
        "boost_from_average": True,
        "deterministic": True,
        "force_row_wise": True,
        "num_threads": 1,
        "seed": seed,
        "learning_rate": LEARNING_RATE,
        "num_leaves": NUM_LEAVES,
        "verbosity": -1,
        "max_bin": 255,
        "min_data_in_bin": 1,
        "bin_construct_sample_cnt": 1_000_000,
        "feature_pre_filter": False,
        "min_data_in_leaf": 1,
        "min_sum_hessian_in_leaf": 1e-3,
    }
    if bag:
        p["bagging_fraction"] = BAGGING_FRACTION
        p["bagging_freq"] = BAGGING_FREQ
        p["bagging_seed"] = BAGGING_SEED
    return p


def dart_corpus():
    """24-row, 2-feature identity-binned corpus (mirrors boosting_parity.rs dart_corpus
    == goss_corpus): feature 0 = i//4 (0..5), feature 1 = i%3 (0..2); labels a clean
    monotone spread."""
    n = 24
    f0 = [i // 4 for i in range(n)]
    f1 = [i % 3 for i in range(n)]
    X = np.array([f0, f1], dtype=np.float64).T
    labels = np.array([i * 1.5 + 1.0 for i in range(n)], dtype=np.float64)
    return X, labels


def cell_tag(uniform_drop, xgboost_dart_mode, bag):
    return "dart_u{}_x{}_bag{}".format(int(uniform_drop), int(xgboost_dart_mode),
                                       int(bag))


def f64_bits_line(values):
    return " ".join(str(f64_bits(v)) for v in values)


def capture_model_cell(out_dir, X, labels, seed, drop_seed, uniform_drop,
                       xgboost_dart_mode, bag):
    p = base_params(seed, drop_seed, uniform_drop, xgboost_dart_mode, bag)
    dtrain = lgb.Dataset(X, label=labels, params=p, free_raw_data=False)
    dtrain.construct()
    booster = lgb.train(p, dtrain, num_boost_round=NUM_ITERATIONS)
    tag = cell_tag(uniform_drop, xgboost_dart_mode, bag)
    booster.save_model(os.path.join(out_dir, f"{tag}_model.txt"))
    # Per-row transformed predictions (regression identity), as f64 bits (one line).
    preds = booster.predict(X)
    with open(os.path.join(out_dir, f"{tag}_pred.txt"), "w") as fh:
        fh.write("# DART per-row transformed predict() (f64 bit patterns).\n")
        fh.write(f64_bits_line(preds) + "\n")


def main():
    if len(sys.argv) != 5:
        sys.exit("usage: dart_oracle_capture.py <out_dir> <seed> <drop_seed> "
                 "<lightgbm_version>")
    out_dir = sys.argv[1]
    seed = int(sys.argv[2])
    drop_seed = int(sys.argv[3])
    expected_version = sys.argv[4]
    if lgb.__version__ != expected_version:
        sys.exit(f"ABORT: lightgbm {lgb.__version__} != recorded {expected_version}")
    os.makedirs(out_dir, exist_ok=True)

    X, labels = dart_corpus()
    # MODEL parity cells over uniform_drop × xgboost_dart_mode × {bag} (4 normalize
    # branches × bagging).
    for uniform_drop in (False, True):
        for xgboost_dart_mode in (False, True):
            for bag in (False, True):
                capture_model_cell(out_dir, X, labels, seed, drop_seed,
                                   uniform_drop, xgboost_dart_mode, bag)
    # Drop RNG-replay golden.
    capture_drop_rng_replay(out_dir, drop_seed)
    print("dart_oracle_capture: wrote DART model cells + drop RNG-replay golden to",
          out_dir)


if __name__ == "__main__":
    main()
