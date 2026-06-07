#!/usr/bin/env python3
"""Phase-7 W4 REAL lib_lightgbm GOSS (BST-04) oracle capture (plan 07-05).

Captures TWO kinds of golden for the GOSS sample strategy:

  (1) MODEL PARITY cells — `goss_t{T}_o{O}_es{E}_bfa{B}_model.txt`. Trains the
      `boosting=goss` axis (top_rate × other_rate × {early_stop} × {bfa}; GOSS forbids
      bagging, so there is NO bag axis) on the real prebuilt `lib_lightgbm` 4.6 pip
      wheel and dumps the authoritative `%.17g` model text. The GOSS sampling +
      grad/hess amplification is REFLECTED in the captured trees (the real binary
      runs the goss.hpp Helper internally), so a wrong Rust amplification factor or
      wrong kept-set shifts the leaves and FAILS the parity replay.

  (2) RNG-REPLAY golden — `goss_rng_replay.txt`. The kept/dropped row indices that
      the goss.hpp `Helper` produces for a FIXED grad/hess input, derived over the
      EXACT C++ `Random` LCG (ported below bit-for-bit from
      `include/LightGBM/utils/random.h`) and the EXACT `ArrayArgs::ArgMaxAtK`
      partition (`include/LightGBM/utils/array_args.h`). The wheel does not expose the
      internal bag indices, so — exactly like the bagging `bag_indices_*` golden — the
      golden FREEZES the algorithm spec over the bit-exact LCG; the Rust
      `GossSampleStrategy` then independently reproduces it (boosting_parity
      `goss_rng_replay`). The line also carries the input grad/hess (as f32 bits) so
      the replay re-derives the |g*h| magnitudes + ArgMaxAtK threshold itself.

DETERMINISM / IDEMPOTENCY: trained with `deterministic=true force_row_wise=true
num_threads=1 seed=<seed> bagging_seed=<bagging_seed>`; both kinds of golden are pure
functions of the pinned inputs, so re-running produces byte-identical files (empty
`git diff`).

NEVER `git add` the LightGBM/ tree; goldens land only under the tracked
`crates/oracle-harness/tests/fixtures/goss/`.

Usage:
  goss_oracle_capture.py <out_dir> <seed> <bagging_seed> <lightgbm_version>
"""

import os
import struct
import sys

import numpy as np

import lightgbm as lgb

# ---- GOSS capture control (mirrors boosting_parity.rs GOSS_* constants) ----
NUM_ITERATIONS = 12
NUM_LEAVES = 4
LEARNING_RATE = 0.1
TOP_RATES = [0.2, 0.1]
OTHER_RATES = [0.1, 0.05]
EARLY_STOPPING_ROUND = 2
BAGGING_RAND_BLOCK = 1024


def f64_bits_line(values):
    return " ".join(str(struct.unpack("<Q", struct.pack("<d", float(v))[:8])[0])
                    for v in values)


def f32_bits(value):
    return struct.unpack("<I", struct.pack("<f", float(value)))[0]


# =====================================================================
# Bit-for-bit port of the C++ `Random` LCG (include/LightGBM/utils/random.h)
# — the SAME recurrence the Rust `lgbm_core::random::Random` mirrors. Used ONLY
# to derive the RNG-replay golden (a deterministic, non-cryptographic PRNG).
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


# =====================================================================
# Bit-for-bit port of ArrayArgs<float>::Partition + ArgMaxAtK
# (include/LightGBM/utils/array_args.h:101-146) over a Python float list that
# carries f32-rounded values (we round every |g*h| to f32 first).
# =====================================================================
def _partition(arr, start, end):
    i = start - 1
    j = end - 1
    p = i
    q = j
    if start >= end - 1:
        return start - 1, end
    v = arr[end - 1]
    while True:
        while True:
            i += 1
            if not (arr[i] > v):
                break
        while True:
            j -= 1
            if not (v > arr[j]):
                break
            if j == start:
                break
        if i >= j:
            break
        arr[i], arr[j] = arr[j], arr[i]
        if arr[i] == v:
            p += 1
            arr[p], arr[i] = arr[i], arr[p]
        if v == arr[j]:
            q -= 1
            arr[j], arr[q] = arr[q], arr[j]
    arr[i], arr[end - 1] = arr[end - 1], arr[i]
    j = i - 1
    i = i + 1
    k = start
    while k <= p:
        arr[k], arr[j] = arr[j], arr[k]
        k += 1
        j -= 1
    k = end - 2
    while k >= q:
        arr[i], arr[k] = arr[k], arr[i]
        i += 1
        k -= 1
    return j, i


def _arg_max_at_k(arr, start, end, k):
    if start >= end - 1:
        return start
    l, r = _partition(arr, start, end)
    if (k > l and k < r) or (l == start - 1 and r == end - 1):
        return k
    elif k <= l:
        return _arg_max_at_k(arr, start, l + 1, k)
    else:
        return _arg_max_at_k(arr, r, end, k)


def _f32(x):
    return struct.unpack("<f", struct.pack("<f", float(x)))[0]


def goss_helper(seed, top_rate, other_rate, grad, hess):
    """Verbatim port of GOSSStrategy::Helper (goss.hpp:116-165), single-class. Returns
    (bag_data_indices, bag_data_cnt). grad/hess are NOT mutated (the golden records the
    kept/dropped layout; the Rust replay reproduces both layout and amplification)."""
    cnt = len(grad)
    n_blocks = (cnt + BAGGING_RAND_BLOCK - 1) // BAGGING_RAND_BLOCK
    rands = [Random(seed + i) for i in range(n_blocks)]
    # tmp_gradients[i] = |grad[i] * hess[i]|, rounded to f32 like score_t.
    tmp = [_f32(abs(_f32(grad[i]) * _f32(hess[i]))) for i in range(cnt)]
    top_k = int(cnt * top_rate)        # (data_size_t) truncation
    other_k = int(cnt * other_rate)
    top_k = max(1, top_k)
    _arg_max_at_k(tmp, 0, len(tmp), top_k - 1)
    threshold = tmp[top_k - 1]
    buf = [0] * cnt
    cur_left = 0
    cur_right = cnt
    big = 0
    for i in range(cnt):
        g = _f32(abs(_f32(grad[i]) * _f32(hess[i])))
        if g >= threshold:
            buf[cur_left] = i
            cur_left += 1
            big += 1
        else:
            sampled = cur_left - big
            rest_need = other_k - sampled
            rest_all = (cnt - i) - (top_k - big)
            prob = rest_need / float(rest_all)
            block = i // BAGGING_RAND_BLOCK
            if float(rands[block].next_float()) < prob:
                buf[cur_left] = i
                cur_left += 1
            else:
                cur_right -= 1
                buf[cur_right] = i
    return buf, cur_left


def base_params(seed, bagging_seed, top_rate, other_rate, bfa):
    return {
        "objective": "regression",
        "metric": ["l2"],
        "boosting": "goss",                 # alias-expands to gbdt + data_sample_strategy=goss
        "top_rate": top_rate,
        "other_rate": other_rate,
        "bagging_seed": bagging_seed,
        "boost_from_average": bool(bfa),
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


def goss_corpus():
    """24-row, 2-feature identity-binned corpus (mirrors boosting_parity.rs
    goss_corpus): feature 0 = i//4 (0..5), feature 1 = i%3 (0..2); labels a clean
    monotone spread so per-row |g*h| magnitudes are well separated."""
    n = 24
    f0 = [i // 4 for i in range(n)]
    f1 = [i % 3 for i in range(n)]
    X = np.array([f0, f1], dtype=np.float64).T
    labels = np.array([i * 1.5 + 1.0 for i in range(n)], dtype=np.float64)
    return X, labels


def matrix_valid_corpus(X):
    return X.copy(), np.full(X.shape[0], 10.0, dtype=np.float64)


def cell_tag(top, other, es, bfa):
    t = int(round(top * 1000))
    o = int(round(other * 1000))
    return "goss_t{}_o{}_es{}_bfa{}".format(t, o, int(es), int(bfa))


def capture_model_cell(out_dir, X, labels, seed, bagging_seed, top, other, es, bfa):
    p = base_params(seed, bagging_seed, top, other, bfa)
    dtrain = lgb.Dataset(X, label=labels, params=p, free_raw_data=False)
    dtrain.construct()
    valid_sets = [dtrain]
    valid_names = ["training"]
    callbacks = []
    if es:
        Xv, lv = matrix_valid_corpus(X)
        dvalid = lgb.Dataset(Xv, label=lv, reference=dtrain, free_raw_data=False)
        dvalid.construct()
        valid_sets = [dtrain, dvalid]
        valid_names = ["training", "valid_0"]
        callbacks.append(lgb.early_stopping(EARLY_STOPPING_ROUND, verbose=False))
    booster = lgb.train(
        p, dtrain, num_boost_round=NUM_ITERATIONS,
        valid_sets=valid_sets, valid_names=valid_names, callbacks=callbacks,
    )
    tag = cell_tag(top, other, es, bfa)
    booster.save_model(os.path.join(out_dir, f"{tag}_model.txt"))


def capture_rng_replay(out_dir, bagging_seed):
    """Write the GOSS RNG-replay golden: for several (seed, top, other) over a FIXED
    grad/hess input, the kept/dropped indices the goss.hpp Helper produces over the
    bit-exact C++ LCG. The Rust GossSampleStrategy reproduces this bit-exact."""
    n = 24
    # A fixed grad/hess with well-separated |g*h| magnitudes (hess=1 => |g*h|=|g|).
    grad = [((i * 7) % 13) - 6.0 for i in range(n)]
    hess = [1.0] * n
    lines = [
        "# GOSS (BST-04) RNG-replay golden (plan 07-05).",
        "#",
        "# The kept/dropped row indices ([kept rows] ++ [dropped rows]) the goss.hpp",
        "# Helper (goss.hpp:116-165) produces for the recorded grad/hess input, derived",
        "# over the EXACT C++ Random LCG (random.h) + ArgMaxAtK partition (array_args.h).",
        "# The wheel cannot expose the internal bag indices, so this freezes the",
        "# algorithm spec over the bit-exact LCG (identical posture to the bagging",
        "# bag_indices_* golden). The Rust GossSampleStrategy reproduces it bit-exact",
        "# (boosting_parity::goss_rng_replay). grad/hess are f32 bit patterns; indices i32.",
        "#",
        "# seed=<S> top=<T> other=<O> num_data=<N> bag_data_cnt=<C> "
        "grad=<csv f32 bits> hess=<csv f32 bits> indices=<csv i32>",
    ]
    grad_bits = ",".join(str(f32_bits(g)) for g in grad)
    hess_bits = ",".join(str(f32_bits(h)) for h in hess)
    for seed, top, other in [(bagging_seed, 0.2, 0.1), (bagging_seed, 0.1, 0.05),
                             (7, 0.2, 0.1)]:
        idx, cnt = goss_helper(seed, top, other, grad, hess)
        idx_csv = ",".join(str(i) for i in idx)
        lines.append(
            f"seed={seed} top={top} other={other} num_data={n} bag_data_cnt={cnt} "
            f"grad={grad_bits} hess={hess_bits} indices={idx_csv}"
        )
    with open(os.path.join(out_dir, "goss_rng_replay.txt"), "w") as fh:
        fh.write("\n".join(lines) + "\n")


def main():
    if len(sys.argv) != 5:
        sys.exit("usage: goss_oracle_capture.py <out_dir> <seed> <bagging_seed> "
                 "<lightgbm_version>")
    out_dir = sys.argv[1]
    seed = int(sys.argv[2])
    bagging_seed = int(sys.argv[3])
    expected_version = sys.argv[4]
    if lgb.__version__ != expected_version:
        sys.exit(f"ABORT: lightgbm {lgb.__version__} != recorded {expected_version}")
    os.makedirs(out_dir, exist_ok=True)

    X, labels = goss_corpus()
    # MODEL parity cells over top × other × {es} × {bfa} (NO bag axis — GOSS forbids it).
    for top in TOP_RATES:
        for other in OTHER_RATES:
            for es in (False, True):
                for bfa in (False, True):
                    capture_model_cell(out_dir, X, labels, seed, bagging_seed,
                                       top, other, es, bfa)
    # RNG-replay golden.
    capture_rng_replay(out_dir, bagging_seed)
    print("goss_oracle_capture: wrote GOSS model cells + RNG-replay golden to", out_dir)


if __name__ == "__main__":
    main()
