#!/usr/bin/env python3
"""Phase-7 W8 REAL lib_lightgbm ranking (OBJ-06 / MET-04) oracle capture (plan 07-09).

Captures FOUR kinds of golden for the ranking stack (lambdarank/rank_xendcg +
ndcg/map + bagging_by_query):

  (1) MODEL PARITY cells — `rank_{obj}_byq{Q}_es{E}_model.txt`. Trains a QUERY/group
      corpus with `objective in {lambdarank, rank_xendcg}` across
      `{bagging_by_query} × {early_stop}` on the real prebuilt `lib_lightgbm` 4.6
      wheel and dumps the authoritative `%.17g` model text.

  (2) PER-QUERY METRIC + SCORE goldens — `rank_{obj}_scores.txt` (per-row raw
      score + label + query_boundaries) and `rank_ndcg.txt` / `rank_map.txt`
      (the real-binary ndcg@k / map@k values). The Rust RankMetric evals over the
      captured scores and must match per-query within ORACLE_TOL (or bit-exact
      where the algorithm permits).

  (3) bagging_by_query RNG-REPLAY golden — `bagging_by_query_seed{S}.txt`. The
      sampled query indices + expanded row indices + sampled_query_boundaries the
      `bagging.hpp` query branch produces over the EXACT C++ `Random` LCG (ported
      below bit-for-bit). The wheel does not expose the internal bag indices, so —
      exactly like the bagging `bag_indices_*` golden — this freezes the algorithm
      spec over the bit-exact LCG; the Rust BaggingSampleStrategy::bagging_by_query
      reproduces it (rank_parity::bagging_by_query_rng_replay).

  (4) rank_xendcg objective_seed RNG-REPLAY golden — `rank_xendcg_objseed{S}.txt`.
      The per-query gamma draw ORDER (one NextFloat() per row, query rng
      `Random(objective_seed + q)`) that RankXENDCG::GetGradientsForOneQuery
      consumes. The Rust RankXendcg reproduces the draw order bit-exact
      (rank_parity::rank_xendcg_objseed_rng_replay).

DETERMINISM / IDEMPOTENCY: trained with `deterministic=true force_row_wise=true
num_threads=1 seed=<seed>`; every golden is a pure function of the pinned inputs,
so re-running produces byte-identical files (empty `git diff`).

NEVER `git add` the LightGBM/ tree; goldens land only under the tracked
`crates/oracle-harness/tests/fixtures/rank/`.

Usage:
  rank_oracle_capture.py <out_dir> <seed> <bagging_seed> <objective_seed> <lightgbm_version>
"""

import os
import struct
import sys

import numpy as np

import lightgbm as lgb

# ---- Ranking capture control (mirrors rank_parity.rs constants) ----
NUM_ITERATIONS = 10
NUM_LEAVES = 7
LEARNING_RATE = 0.1
EARLY_STOPPING_ROUND = 2
BAGGING_RAND_BLOCK = 1024
EVAL_AT = [1, 3, 5]


def f64_bits(value):
    return struct.unpack("<Q", struct.pack("<d", float(value)))[0]


def f32_bits(value):
    return struct.unpack("<I", struct.pack("<f", float(value)))[0]


# =====================================================================
# Bit-for-bit port of the C++ `Random` LCG (include/LightGBM/utils/random.h)
# — the SAME recurrence the Rust `lgbm_core::random::Random` mirrors. Used ONLY
# to derive the RNG-replay goldens (a deterministic, non-cryptographic PRNG).
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


def rank_corpus():
    """A small graded-relevance query corpus: 6 queries of varying size summing to
    30 rows, 3 features. Labels are graded 0..3. Identity-ish binning via integer
    features so the tree structure is stable."""
    # query sizes: 3,5,4,6,5,7 = 30 rows.
    group = [3, 5, 4, 6, 5, 7]
    n = sum(group)
    rng = np.random.RandomState(20240607)
    # 3 integer-ish features for stable bins.
    X = rng.randint(0, 8, size=(n, 3)).astype(np.float64)
    # graded labels 0..3, biased so each query has a mix.
    labels = rng.randint(0, 4, size=n).astype(np.float64)
    return X, labels, group


def query_boundaries_from_group(group):
    qb = [0]
    for g in group:
        qb.append(qb[-1] + g)
    return qb


def base_params(obj, seed, bagging_seed, objective_seed, by_query, bfa=False):
    p = {
        "objective": obj,
        "metric": ["ndcg", "map"],
        "eval_at": EVAL_AT,
        "ndcg_eval_at": EVAL_AT,
        "boosting": "gbdt",
        "deterministic": True,
        "force_row_wise": True,
        "num_threads": 1,
        "seed": seed,
        "bagging_seed": bagging_seed,
        "objective_seed": objective_seed,
        "learning_rate": LEARNING_RATE,
        "num_leaves": NUM_LEAVES,
        "verbosity": -1,
        "max_bin": 255,
        "min_data_in_bin": 1,
        "bin_construct_sample_cnt": 1_000_000,
        "feature_pre_filter": False,
        "min_data_in_leaf": 1,
        "min_sum_hessian_in_leaf": 1e-3,
        "lambdarank_truncation_level": 5,
        "lambdarank_norm": True,
    }
    if by_query:
        p["bagging_by_query"] = True
        p["bagging_fraction"] = 0.7
        p["bagging_freq"] = 1
    return p


def cell_tag(obj, by_query, es):
    return "rank_{}_byq{}_es{}".format(obj, int(by_query), int(es))


def capture_model_cell(out_dir, X, labels, group, obj, seed, bagging_seed,
                       objective_seed, by_query, es):
    p = base_params(obj, seed, bagging_seed, objective_seed, by_query)
    dtrain = lgb.Dataset(X, label=labels, group=group, params=p, free_raw_data=False)
    dtrain.construct()
    valid_sets = [dtrain]
    valid_names = ["training"]
    callbacks = []
    if es:
        callbacks.append(lgb.early_stopping(EARLY_STOPPING_ROUND, verbose=False))
    booster = lgb.train(
        p, dtrain, num_boost_round=NUM_ITERATIONS,
        valid_sets=valid_sets, valid_names=valid_names, callbacks=callbacks,
    )
    tag = cell_tag(obj, by_query, es)
    booster.save_model(os.path.join(out_dir, f"{tag}_model.txt"))
    return booster


def capture_scores_and_metrics(out_dir, X, labels, group, obj, seed, bagging_seed,
                               objective_seed):
    """Train the no-bag/no-es cell, dump the raw scores + labels + query_boundaries,
    and the real-binary per-query ndcg@k / map@k metric values."""
    p = base_params(obj, seed, bagging_seed, objective_seed, by_query=False)
    dtrain = lgb.Dataset(X, label=labels, group=group, params=p, free_raw_data=False)
    dtrain.construct()
    # record_evaluation captures the per-iteration ndcg@k / map@k on the training set
    # (the last entry is the final-model value, matching booster.predict below).
    rec = {}
    booster = lgb.train(
        p, dtrain, num_boost_round=NUM_ITERATIONS,
        valid_sets=[dtrain], valid_names=["training"],
        callbacks=[lgb.record_evaluation(rec)],
    )
    raw = booster.predict(X, raw_score=True)
    qb = query_boundaries_from_group(group)
    # raw scores (f64 bits), labels (f32 bits), query_boundaries (i32).
    score_bits = ",".join(str(f64_bits(s)) for s in raw)
    label_bits = ",".join(str(f32_bits(l)) for l in labels)
    qb_csv = ",".join(str(b) for b in qb)
    with open(os.path.join(out_dir, f"{obj}_scores.txt"), "w") as fh:
        fh.write(f"# {obj} raw scores + labels + query_boundaries (real lib_lightgbm 4.6).\n")
        fh.write(f"# scores: f64 bits; labels: f32 bits; query_boundaries: i32.\n")
        fh.write(f"scores={score_bits}\n")
        fh.write(f"labels={label_bits}\n")
        fh.write(f"query_boundaries={qb_csv}\n")
    # Per-query ndcg/map: the real binary's last-iteration eval over the training set
    # (== the final model the raw scores above come from). The Rust RankMetric over the
    # same scores must match these per @k.
    ndcg_vals = {}
    map_vals = {}
    train_metrics = rec.get("training", {})
    for mname, vals in train_metrics.items():
        if mname.startswith("ndcg@"):
            ndcg_vals[int(mname.split("@")[1])] = vals[-1]
        elif mname.startswith("map@"):
            map_vals[int(mname.split("@")[1])] = vals[-1]
    with open(os.path.join(out_dir, f"{obj}_ndcg.txt"), "w") as fh:
        fh.write(f"# {obj} ndcg@k (real lib_lightgbm 4.6), one per eval_at.\n")
        for k in EVAL_AT:
            if k in ndcg_vals:
                fh.write(f"ndcg@{k}={f64_bits(ndcg_vals[k])}\n")
    with open(os.path.join(out_dir, f"{obj}_map.txt"), "w") as fh:
        fh.write(f"# {obj} map@k (real lib_lightgbm 4.6), one per eval_at.\n")
        for k in EVAL_AT:
            if k in map_vals:
                fh.write(f"map@{k}={f64_bits(map_vals[k])}\n")


def bag_by_query_reference(seed, fraction, num_data, qb):
    """Verbatim re-impl of bagging.hpp:52-104 query branch over the bit-exact LCG:
    draw QUERIES, expand in-bag queries to row ranges, build sampled_query_boundaries.
    Returns (sampled_query_indices, sampled_query_boundaries, expanded_rows)."""
    num_queries = len(qb) - 1
    # bagging_rands sized by num_data (bagging.hpp:178-181).
    n_blocks = (num_data + BAGGING_RAND_BLOCK - 1) // BAGGING_RAND_BLOCK
    rands = [Random(seed + i) for i in range(n_blocks)]
    buf = [0] * num_data
    left = 0
    right = num_queries
    for q in range(num_queries):
        block = q // BAGGING_RAND_BLOCK
        if float(rands[block].next_float()) < fraction:
            buf[left] = q
            left += 1
        else:
            right -= 1
            buf[right] = q
    sampled = buf[:left]
    sqb = [0]
    for q in sampled:
        sqb.append(sqb[-1] + (qb[q + 1] - qb[q]))
    total = sqb[-1]
    expanded = [0] * total
    for s, q in enumerate(sampled):
        ds, de = qb[q], qb[q + 1]
        ss = sqb[s]
        for off, i in enumerate(range(ds, de)):
            expanded[ss + off] = i
    return sampled, sqb, expanded


def capture_bag_by_query_rng(out_dir, bagging_seed, group):
    qb = query_boundaries_from_group(group)
    num_data = qb[-1]
    lines = [
        "# bagging_by_query (BST-03 / 07-09) RNG-replay golden.",
        "#",
        "# The sampled query indices + sampled_query_boundaries + expanded row indices",
        "# the bagging.hpp query branch (bagging.hpp:52-104) produces over the EXACT C++",
        "# Random LCG (random.h), sized by num_data (bagging.hpp:178-181). The wheel",
        "# cannot expose internal bag indices, so this freezes the algorithm spec over",
        "# the bit-exact LCG. The Rust BaggingSampleStrategy::bagging_by_query reproduces",
        "# it bit-exact (rank_parity::bagging_by_query_rng_replay).",
        "#",
        "# seed=<S> fraction=<F> num_data=<N> query_boundaries=<csv i32> "
        "sampled_queries=<csv i32> sampled_query_boundaries=<csv i32> expanded_rows=<csv i32>",
    ]
    qb_csv = ",".join(str(b) for b in qb)
    for seed, frac in [(bagging_seed, 0.7), (bagging_seed, 0.5), (7, 0.6)]:
        sampled, sqb, expanded = bag_by_query_reference(seed, frac, num_data, qb)
        lines.append(
            f"seed={seed} fraction={frac} num_data={num_data} query_boundaries={qb_csv} "
            f"sampled_queries={','.join(str(s) for s in sampled)} "
            f"sampled_query_boundaries={','.join(str(s) for s in sqb)} "
            f"expanded_rows={','.join(str(e) for e in expanded)}"
        )
    with open(os.path.join(out_dir, f"bagging_by_query_seed{bagging_seed}.txt"), "w") as fh:
        fh.write("\n".join(lines) + "\n")


def capture_xendcg_objseed_rng(out_dir, objective_seed, group):
    """Freeze the rank_xendcg per-query gamma draw ORDER over the bit-exact LCG. For
    each query q, Random(objective_seed + q) yields one NextFloat() per row (in row
    order); we dump those draws so the Rust RankXendcg replay is bit-exact."""
    qb = query_boundaries_from_group(group)
    num_queries = len(qb) - 1
    lines = [
        "# rank_xendcg objective_seed RNG-replay golden (OBJ-06 / 07-09).",
        "#",
        "# Per query q, Random(objective_seed + q) yields ONE NextFloat() per row in",
        "# row order (rank_objective.hpp:389-391,417). The draws (as f32 bit patterns)",
        "# are the gamma g in Phi(label, g) = 2^label - g. The Rust RankXendcg",
        "# reproduces the draw order bit-exact (rank_parity::rank_xendcg_objseed_rng_replay).",
        "#",
        "# objective_seed=<S> query_boundaries=<csv i32> draws=<csv f32 bits, row-major over all rows>",
    ]
    for seed in [objective_seed, 7]:
        draws = []
        for q in range(num_queries):
            cnt = qb[q + 1] - qb[q]
            rng = Random(seed + q)
            for _ in range(cnt):
                draws.append(f32_bits(rng.next_float()))
        qb_csv = ",".join(str(b) for b in qb)
        lines.append(
            f"objective_seed={seed} query_boundaries={qb_csv} "
            f"draws={','.join(str(d) for d in draws)}"
        )
    with open(os.path.join(out_dir, f"rank_xendcg_objseed{objective_seed}.txt"), "w") as fh:
        fh.write("\n".join(lines) + "\n")


def main():
    if len(sys.argv) != 6:
        sys.exit("usage: rank_oracle_capture.py <out_dir> <seed> <bagging_seed> "
                 "<objective_seed> <lightgbm_version>")
    out_dir = sys.argv[1]
    seed = int(sys.argv[2])
    bagging_seed = int(sys.argv[3])
    objective_seed = int(sys.argv[4])
    expected_version = sys.argv[5]
    if lgb.__version__ != expected_version:
        sys.exit(f"ABORT: lightgbm {lgb.__version__} != recorded {expected_version}")
    os.makedirs(out_dir, exist_ok=True)

    X, labels, group = rank_corpus()
    # MODEL parity cells over {obj} × {bagging_by_query} × {es}.
    for obj in ("lambdarank", "rank_xendcg"):
        for by_query in (False, True):
            for es in (False, True):
                capture_model_cell(out_dir, X, labels, group, obj, seed,
                                   bagging_seed, objective_seed, by_query, es)
        # per-query scores + ndcg/map metrics (no-bag, no-es).
        capture_scores_and_metrics(out_dir, X, labels, group, obj, seed,
                                   bagging_seed, objective_seed)
    # RNG-replay goldens (no wheel-internal state needed).
    capture_bag_by_query_rng(out_dir, bagging_seed, group)
    capture_xendcg_objseed_rng(out_dir, objective_seed, group)
    print("rank_oracle_capture: wrote ranking model cells + per-query metrics + "
          "RNG-replay goldens to", out_dir)


if __name__ == "__main__":
    main()
