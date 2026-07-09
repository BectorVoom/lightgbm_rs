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

import math
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
# config.h `sigmoid_` default (base_params leaves `sigmoid` unset → 1.0). The
# score-derivation lambdarank grad/hess uses it verbatim.
LAMBDARANK_SIGMOID = 1.0


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


# =====================================================================
# Score-derivation lambdarank grad/hess (RESEARCH A1 route). Re-derives the per-row
# lambda/hessian from the captured raw scores + labels + query_boundaries, mirroring
# `lgbm_objective::rank::Lambdarank::get_gradients` (rank.rs) — itself a 1:1 port of
# the C++ `LambdarankNDCG::GetGradientsForOneQuery` (rank_objective.hpp:180-266) —
# BIT-FOR-BIT, including the f32 lambda/hessian accumulation. Emits the same
# `GRAD <u32 bits>` / `HESS <u32 bits>` format as boosting/*_gh_iter*.txt (D-01).
# =====================================================================
def _default_label_gain():
    """DCGCalculator::DefaultLabelGain (dcg_calculator.cpp:33-41): 2^i - 1 for
    i in 0..31 (entry 0 = 0), max_label = 31 to avoid overflow."""
    return [float((1 << i) - 1) for i in range(31)]


def _discount_table(n):
    """discount_[i] = 1 / log2(2 + i) (dcg_calculator.cpp Init, built once)."""
    return [1.0 / math.log2(2.0 + i) for i in range(n)]


def _sigmoid_table(sigmoid):
    """ConstructSigmoidTable (rank_objective.hpp:281-294) — the 2^20-bin lookup, with
    the ctor in-place `min = min / sigmoid / 2` mutation."""
    nbins = 1024 * 1024
    min_in = -50.0 / sigmoid / 2.0
    max_in = -min_in
    idx_factor = nbins / (max_in - min_in)
    i = np.arange(nbins, dtype=np.float64)
    score = i / idx_factor + min_in
    table = 1.0 / (1.0 + np.exp(score * sigmoid))
    return table, min_in, max_in, idx_factor, nbins


def _get_sigmoid(s, table, min_in, max_in, idx_factor, nbins):
    """LambdarankNDCG::GetSigmoid (rank_objective.hpp:268-279) — the table lookup."""
    if s <= min_in:
        return float(table[0])
    if s >= max_in:
        return float(table[nbins - 1])
    # `as usize` truncates toward zero; (s - min_in) >= 0 here.
    idx = int((s - min_in) * idx_factor)
    return float(table[idx])


def _cal_max_dcg_at_k(k, labels, label_gain, discount):
    """DCGCalculator::CalMaxDCGAtK (dcg_calculator.cpp:78-107 single-k form)."""
    num_data = len(labels)
    label_cnt = [0] * len(label_gain)
    for l in labels:
        label_cnt[int(l)] += 1
    top_label = len(label_gain) - 1
    kk = min(k, num_data)
    ret = 0.0
    for j in range(kk):
        while top_label > 0 and label_cnt[top_label] <= 0:
            top_label -= 1
        if top_label < 0:
            break
        ret += discount[j] * label_gain[top_label]
        label_cnt[top_label] -= 1
    return ret


def lambdarank_grad_hess(scores, labels, qb, sigmoid, norm, truncation_level):
    """Per-row grad/hess (f32) re-derived from raw `scores`, mirroring rank.rs
    `get_gradients` → `gradients_for_one_query` bit-for-bit (f32 accumulation)."""
    label_gain = _default_label_gain()
    discount = _discount_table(10000)
    sig_tbl, min_in, max_in, idx_factor, nbins = _sigmoid_table(sigmoid)
    n = len(scores)
    grad = np.zeros(n, dtype=np.float32)
    hess = np.zeros(n, dtype=np.float32)
    neg_inf = float("-inf")
    f32_001 = float(np.float32(0.01))  # C++ uses the 0.01f literal in f64 math.
    num_queries = len(qb) - 1
    for q in range(num_queries):
        start = qb[q]
        cnt = qb[q + 1] - start
        s = [float(scores[start + i]) for i in range(cnt)]
        ql = [int(labels[start + i]) for i in range(cnt)]
        m = _cal_max_dcg_at_k(
            truncation_level, [labels[start + i] for i in range(cnt)], label_gain, discount
        )
        inv_max_dcg = (1.0 / m) if m > 0.0 else m
        lam = [np.float32(0.0)] * cnt
        hes = [np.float32(0.0)] * cnt
        # sorted_idx by DESCENDING score, stable, ascending-index tie-break.
        sorted_idx = sorted(range(cnt), key=lambda a: (-s[a], a))
        best_score = s[sorted_idx[0]]
        worst_idx = cnt - 1
        if worst_idx > 0 and s[sorted_idx[worst_idx]] == neg_inf:
            worst_idx -= 1
        worst_score = s[sorted_idx[worst_idx]]
        sum_lambdas = 0.0
        i = 0
        while i + 1 < cnt and i < truncation_level:
            if s[sorted_idx[i]] == neg_inf:
                i += 1
                continue
            for j in range(i + 1, cnt):
                if s[sorted_idx[j]] == neg_inf:
                    continue
                if ql[sorted_idx[i]] == ql[sorted_idx[j]]:
                    continue
                if ql[sorted_idx[i]] > ql[sorted_idx[j]]:
                    high_rank, low_rank = i, j
                else:
                    high_rank, low_rank = j, i
                high = sorted_idx[high_rank]
                low = sorted_idx[low_rank]
                high_score = s[high]
                low_score = s[low]
                high_label_gain = label_gain[ql[high]]
                low_label_gain = label_gain[ql[low]]
                high_discount = discount[high_rank]
                low_discount = discount[low_rank]
                delta_score = high_score - low_score
                dcg_gap = high_label_gain - low_label_gain
                paired_discount = abs(high_discount - low_discount)
                delta_pair_ndcg = dcg_gap * paired_discount * inv_max_dcg
                if norm and best_score != worst_score:
                    delta_pair_ndcg /= f32_001 + abs(delta_score)
                p_lambda = _get_sigmoid(delta_score, sig_tbl, min_in, max_in, idx_factor, nbins)
                p_hessian = p_lambda * (1.0 - p_lambda)
                p_lambda *= -sigmoid * delta_pair_ndcg
                p_hessian *= sigmoid * sigmoid * delta_pair_ndcg
                lam[low] = np.float32(lam[low] - np.float32(p_lambda))
                hes[low] = np.float32(hes[low] + np.float32(p_hessian))
                lam[high] = np.float32(lam[high] + np.float32(p_lambda))
                hes[high] = np.float32(hes[high] + np.float32(p_hessian))
                sum_lambdas -= 2.0 * p_lambda
            i += 1
        if norm and sum_lambdas > 0.0:
            norm_factor = math.log2(1.0 + sum_lambdas) / sum_lambdas
            for i in range(cnt):
                lam[i] = np.float32(float(lam[i]) * norm_factor)
                hes[i] = np.float32(float(hes[i]) * norm_factor)
        for i in range(cnt):
            grad[start + i] = lam[i]
            hess[start + i] = hes[i]
    return grad, hess


def _write_gh(out_dir, fname, grad, hess, header):
    def bits_line(vals):
        return " ".join(str(f32_bits(v)) for v in vals)

    with open(os.path.join(out_dir, fname), "w") as fh:
        fh.write(header + "\n")
        fh.write("GRAD " + bits_line(grad) + "\n")
        fh.write("HESS " + bits_line(hess) + "\n")


def capture_lambdarank_gh(out_dir, X, labels, group, seed, bagging_seed, objective_seed):
    """The one Wave-0 capture gap (19-VALIDATION / D-01): train real lib_lightgbm
    lambdarank and re-derive the iter-1 per-row grad/hess into lambdarank_gh_iter1.txt
    in the boosting *_gh bit format (score-derivation route). The iterN golden was
    dropped (WR-01): the rank scores fixture stores only the final raw-score line, so
    the iterN intermediate score vector cannot be reconstructed by any test."""
    p = base_params("lambdarank", seed, bagging_seed, objective_seed, by_query=False)
    sigmoid = LAMBDARANK_SIGMOID
    norm = p["lambdarank_norm"]
    trunc = p["lambdarank_truncation_level"]
    qb = query_boundaries_from_group(group)
    dtrain = lgb.Dataset(X, label=labels, group=group, params=p, free_raw_data=False)
    dtrain.construct()
    # Train the real binary as a smoke check that the corpus/params fit; the iter-1
    # golden below is derived from a zero score vector (ranking has no boost-from-average
    # init) and does not depend on the trained model. The iterN golden was DROPPED
    # (WR-01): the rank scores fixture stores only the final raw-score line, so the iterN
    # intermediate score vector cannot be reconstructed and no test could consume it.
    lgb.train(
        p, dtrain, num_boost_round=NUM_ITERATIONS,
        valid_sets=[dtrain], valid_names=["training"],
    )
    # iter1: score_0 = 0 for every row (ranking has no boost-from-average init).
    score0 = np.zeros(len(labels), dtype=np.float64)
    grad1, hess1 = lambdarank_grad_hess(score0, labels, qb, sigmoid, norm, trunc)
    _write_gh(
        out_dir, "lambdarank_gh_iter1.txt", grad1, hess1,
        "# lambdarank_gh_iter1 — iter-1 per-row grad/hess; f32 bits; GRAD then HESS "
        "(real lib_lightgbm 4.6 score-derivation, D-01)",
    )


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
    # lambdarank grad/hess golden (the one Wave-0 capture gap, D-01).
    capture_lambdarank_gh(out_dir, X, labels, group, seed, bagging_seed, objective_seed)
    print("rank_oracle_capture: wrote ranking model cells + per-query metrics + "
          "RNG-replay goldens + lambdarank_gh goldens to", out_dir)


if __name__ == "__main__":
    main()
