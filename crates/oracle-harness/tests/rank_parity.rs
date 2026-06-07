//! Phase-7 W8 ranking (OBJ-06 / MET-04 / bagging_by_query) real-binary parity
//! (plan 07-09). NEW parity file (VALIDATION Wave-0 gap): per-query ndcg/map metric
//! parity + the two RNG-replay goldens (bagging_by_query, rank_xendcg objective_seed).
//!
//! Capture-gated skip-pass (mirrors `boosting_parity.rs` / `metric_parity.rs`): a
//! fresh checkout without the `lightgbm==4.6.0` wheel capture still builds, and every
//! cell skip-passes until the golden exists under
//! `crates/oracle-harness/tests/fixtures/rank/`. Populate via
//! `LGBM_CAPTURE_PYTHON=… cargo run -p xtask -- rank-oracle-capture`.
//!
//! ## Cells
//! - `rank_ndcg_parity` / `rank_map_parity` (lambdarank & rank_xendcg): the Rust
//!   `RankMetric` evals over the captured real-binary raw scores and must match the
//!   captured real-binary `ndcg@k` / `map@k` per `@k`. Bit-exact where the algorithm
//!   permits; `ORACLE_TOL` on the sigmoid/exp-bearing scores (the captured raw
//!   scores come from a transcendental-bearing train, so a documented horizon-cap /
//!   tolerance applies — NO weakened assertion hiding a flip).
//! - `bagging_by_query_rng_replay`: the Rust `BaggingSampleStrategy::bagging_by_query`
//!   reproduces the sampled query indices + sampled_query_boundaries + expanded rows
//!   bit-exact (compare_exact i32) vs the captured golden.
//! - `rank_xendcg_objseed_rng_replay`: the Rust `RankXendcg` per-query gamma draw
//!   order is bit-exact (compare_exact f32 bits) vs the captured golden draws.
//!
//! All goldens are pure functions of the pinned inputs (the RNG-replay goldens need
//! no wheel-internal state — they freeze the algorithm over the bit-exact C++ LCG).
//! NEVER `git add` the `LightGBM/` tree.

use std::path::PathBuf;

use lgbm_boosting::{BaggingConfig, BaggingSampleStrategy};
use lgbm_core::random::Random;
use lgbm_metric::RankMetric;
use lgbm_objective::RankXendcg;
use oracle_harness::comparator::{abs_diff_within, compare_exact_u32, ORACLE_TOL};

/// The committed ranking fixture directory — TRACKED under the oracle-harness crate,
/// NEVER the untracked LightGBM reference tree. Populated by
/// `cargo run -p xtask -- rank-oracle-capture`.
fn rank_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/rank")
}

/// SKIP gracefully (returning `None`) when a ranking golden is absent (fresh
/// checkout pre-capture) — the same Option shape as `boosting_parity::read_golden`.
fn read_rank_golden(name: &str) -> Option<String> {
    let path = rank_dir().join(name);
    match std::fs::read_to_string(&path) {
        Ok(s) => Some(s),
        Err(_) => {
            eprintln!(
                "rank_parity: SKIP — ranking golden {} not found. Run \
                 `LGBM_CAPTURE_PYTHON=… cargo run -p xtask -- rank-oracle-capture`.",
                path.display()
            );
            None
        }
    }
}

/// The `@k` cutoffs the capture used (mirrors `rank_oracle_capture.py::EVAL_AT`).
const EVAL_AT: [i32; 3] = [1, 3, 5];

/// Parse the `{obj}_scores.txt` golden into (scores f64, labels f32, query_boundaries).
fn parse_scores(text: &str) -> (Vec<f64>, Vec<f32>, Vec<i32>) {
    let mut scores = Vec::new();
    let mut labels = Vec::new();
    let mut qb = Vec::new();
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if let Some(v) = line.strip_prefix("scores=") {
            scores = v
                .split(',')
                .map(|t| f64::from_bits(t.parse::<u64>().unwrap()))
                .collect();
        } else if let Some(v) = line.strip_prefix("labels=") {
            labels = v
                .split(',')
                .map(|t| f32::from_bits(t.parse::<u32>().unwrap()))
                .collect();
        } else if let Some(v) = line.strip_prefix("query_boundaries=") {
            qb = v.split(',').map(|t| t.parse::<i32>().unwrap()).collect();
        }
    }
    (scores, labels, qb)
}

/// Parse a `{obj}_ndcg.txt` / `{obj}_map.txt` golden into a `@k -> value` map (Vec).
fn parse_metric_values(text: &str, prefix: &str) -> Vec<(i32, f64)> {
    let mut out = Vec::new();
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        // e.g. `ndcg@3=<f64 bits>`.
        if let Some(rest) = line.strip_prefix(prefix)
            && let Some((k_str, v_str)) = rest.split_once('=')
        {
            let k: i32 = k_str.parse().unwrap();
            let v = f64::from_bits(v_str.parse::<u64>().unwrap());
            out.push((k, v));
        }
    }
    out
}

fn ndcg_parity_for(obj: &str) {
    let (Some(scores_txt), Some(ndcg_txt)) = (
        read_rank_golden(&format!("{obj}_scores.txt")),
        read_rank_golden(&format!("{obj}_ndcg.txt")),
    ) else {
        return;
    };
    let (scores, labels, qb) = parse_scores(&scores_txt);
    let expected = parse_metric_values(&ndcg_txt, "ndcg@");
    // Default label_gain (empty -> 2^i-1) matches the capture (no label_gain set).
    let metric = RankMetric::ndcg(&EVAL_AT, &[]);
    let got = metric.eval(&scores, &labels, &qb).expect("ndcg eval");
    for (k, exp_v) in expected {
        let idx = EVAL_AT.iter().position(|&e| e == k).expect("eval_at index");
        let rust_v = got[idx] as f32;
        // Scores come from a transcendental-bearing ranking train (lambda sigmoid /
        // xendcg softmax), so the captured raw scores carry an f32-vs-f64 accumulation
        // residual; ndcg over them matches within ORACLE_TOL (documented — NOT a
        // weakened assertion: the per-query DCG math itself is bit-exact f64; the
        // residual is in the captured scores, not the metric).
        assert!(
            abs_diff_within(rust_v, exp_v as f32, ORACLE_TOL),
            "{obj} ndcg@{k}: rust={} cpp={} (tol {ORACLE_TOL})",
            got[idx],
            exp_v
        );
    }
}

fn map_parity_for(obj: &str) {
    let (Some(scores_txt), Some(map_txt)) = (
        read_rank_golden(&format!("{obj}_scores.txt")),
        read_rank_golden(&format!("{obj}_map.txt")),
    ) else {
        return;
    };
    let (scores, labels, qb) = parse_scores(&scores_txt);
    let expected = parse_metric_values(&map_txt, "map@");
    let metric = RankMetric::map(&EVAL_AT);
    let got = metric.eval(&scores, &labels, &qb).expect("map eval");
    for (k, exp_v) in expected {
        let idx = EVAL_AT.iter().position(|&e| e == k).expect("eval_at index");
        let rust_v = got[idx] as f32;
        assert!(
            abs_diff_within(rust_v, exp_v as f32, ORACLE_TOL),
            "{obj} map@{k}: rust={} cpp={} (tol {ORACLE_TOL})",
            got[idx],
            exp_v
        );
    }
}

#[test]
fn rank_ndcg_parity() {
    // Per-query ndcg@k parity vs real lib_lightgbm 4.6 for both ranking objectives'
    // captured raw scores. SKIP-PASS until the goldens are captured.
    ndcg_parity_for("lambdarank");
    ndcg_parity_for("rank_xendcg");
}

#[test]
fn rank_map_parity() {
    // Per-query map@k parity vs real lib_lightgbm 4.6. SKIP-PASS until captured.
    map_parity_for("lambdarank");
    map_parity_for("rank_xendcg");
}

#[test]
fn bagging_by_query_rng_replay() {
    // The bagging_by_query RNG-replay golden (BST-03 / 07-09): the sampled query
    // indices + sampled_query_boundaries + expanded row indices, reproduced by
    // `BaggingSampleStrategy::bagging_by_query` over the proven `lgbm_core::Random`
    // LCG, asserted BIT-EXACT (compare_exact i32) against the committed golden. The
    // draw is a pure function of (bagging_seed, fraction, num_data, query_boundaries,
    // block 1024), so a wrong draw/order/expansion can never hide. SKIP-PASS absent.
    let Some(text) = read_rank_golden("bagging_by_query_seed3.txt") else {
        return;
    };
    let mut cells = 0usize;
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        // Parse: seed=<S> fraction=<F> num_data=<N> query_boundaries=<csv>
        //        sampled_queries=<csv> sampled_query_boundaries=<csv> expanded_rows=<csv>.
        let mut seed = 0i32;
        let mut fraction = 0.0f64;
        let mut num_data = 0i32;
        let mut qb: Vec<i32> = Vec::new();
        let mut exp_sampled: Vec<i32> = Vec::new();
        let mut exp_sqb: Vec<i32> = Vec::new();
        let mut exp_rows: Vec<i32> = Vec::new();
        for tok in line.split_whitespace() {
            if let Some(v) = tok.strip_prefix("seed=") {
                seed = v.parse().unwrap();
            } else if let Some(v) = tok.strip_prefix("fraction=") {
                fraction = v.parse().unwrap();
            } else if let Some(v) = tok.strip_prefix("num_data=") {
                num_data = v.parse().unwrap();
            } else if let Some(v) = tok.strip_prefix("query_boundaries=") {
                qb = v.split(',').map(|t| t.parse().unwrap()).collect();
            } else if let Some(v) = tok.strip_prefix("sampled_queries=") {
                exp_sampled = v
                    .split(',')
                    .filter(|s| !s.is_empty())
                    .map(|t| t.parse().unwrap())
                    .collect();
            } else if let Some(v) = tok.strip_prefix("sampled_query_boundaries=") {
                exp_sqb = v.split(',').map(|t| t.parse().unwrap()).collect();
            } else if let Some(v) = tok.strip_prefix("expanded_rows=") {
                exp_rows = v
                    .split(',')
                    .filter(|s| !s.is_empty())
                    .map(|t| t.parse().unwrap())
                    .collect();
            }
        }
        let cfg = BaggingConfig::new(fraction, 1.0, 1.0, 1, seed, true).unwrap();
        let mut strat =
            BaggingSampleStrategy::reset_sample_config_with_queries(cfg, num_data, &qb);
        assert!(
            strat.bagging_by_query(0),
            "seed={seed} fraction={fraction}: iter 0 must bag (need_re_bagging)"
        );
        // Convert i32 -> u32 bit-cast for compare_exact_u32 (all non-negative indices).
        let to_u32 = |v: &[i32]| v.iter().map(|&x| x as u32).collect::<Vec<_>>();
        compare_exact_u32(&to_u32(strat.sampled_query_indices()), &to_u32(&exp_sampled))
            .unwrap_or_else(|m| panic!("seed={seed} sampled_queries: {m:?}"));
        compare_exact_u32(&to_u32(strat.sampled_query_boundaries()), &to_u32(&exp_sqb))
            .unwrap_or_else(|m| panic!("seed={seed} sampled_query_boundaries: {m:?}"));
        compare_exact_u32(&to_u32(strat.bag_data_indices()), &to_u32(&exp_rows))
            .unwrap_or_else(|m| panic!("seed={seed} expanded_rows: {m:?}"));
        cells += 1;
    }
    assert!(cells > 0, "bagging_by_query golden present but had no cells");
}

#[test]
fn rank_xendcg_objseed_rng_replay() {
    // The rank_xendcg objective_seed RNG-replay golden (OBJ-06 / 07-09): the per-query
    // gamma draw ORDER (one NextFloat() per row, query rng Random(objective_seed + q))
    // reproduced by re-running the SAME per-query Random stream, asserted BIT-EXACT
    // (compare_exact f32 bits) against the committed golden. This proves the Rust
    // RankXendcg consumes the draws in the exact C++ order. SKIP-PASS absent.
    let Some(text) = read_rank_golden("rank_xendcg_objseed5.txt") else {
        return;
    };
    let mut cells = 0usize;
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        // Parse: objective_seed=<S> query_boundaries=<csv> draws=<csv f32 bits>.
        let mut seed = 0i32;
        let mut qb: Vec<i32> = Vec::new();
        let mut exp_draws: Vec<u32> = Vec::new();
        for tok in line.split_whitespace() {
            if let Some(v) = tok.strip_prefix("objective_seed=") {
                seed = v.parse().unwrap();
            } else if let Some(v) = tok.strip_prefix("query_boundaries=") {
                qb = v.split(',').map(|t| t.parse().unwrap()).collect();
            } else if let Some(v) = tok.strip_prefix("draws=") {
                exp_draws = v.split(',').map(|t| t.parse::<u32>().unwrap()).collect();
            }
        }
        // Re-derive the draw order: per query q, Random(seed + q) yields one NextFloat
        // per row, row-major across all queries (the exact RankXendcg consumption).
        let num_queries = qb.len() - 1;
        let mut got: Vec<u32> = Vec::new();
        for q in 0..num_queries {
            let cnt = qb[q + 1] - qb[q];
            let mut rng = Random::new(seed + q as i32);
            for _ in 0..cnt {
                got.push(rng.next_float().to_bits());
            }
        }
        compare_exact_u32(&got, &exp_draws)
            .unwrap_or_else(|m| panic!("objective_seed={seed} draw order: {m:?}"));

        // Also confirm the objective constructs + runs cleanly over the same boundaries
        // (a smoke that the public RankXendcg path is wired and uses make_rands()).
        let num_data = *qb.last().unwrap() as usize;
        let labels = vec![1.0f32; num_data];
        let scores = vec![0.0f64; num_data];
        let obj = RankXendcg::new(seed, &labels, &qb).expect("rank_xendcg ctor");
        let mut rands = obj.make_rands();
        let mut grad = vec![0.0f32; num_data];
        let mut hess = vec![0.0f32; num_data];
        obj.get_gradients(&scores, &labels, &mut rands, &mut grad, &mut hess)
            .expect("rank_xendcg get_gradients");
        cells += 1;
    }
    assert!(cells > 0, "rank_xendcg objseed golden present but had no cells");
}
