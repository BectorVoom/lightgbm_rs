//! Ranking-family on-device objective parity (Phase 19, 19-04 / ODL-08).
//!
//! Cells:
//! - `lambdarank` — the block-per-query pairwise-λ grad/hess (shared + `>2048`
//!   `_Sorted` variants) compared `compare_within(ORACLE_TOL)` vs BOTH the
//!   `lgbm_objective::rank::Lambdarank` f64 anchor (D-05) AND the real-lib_lightgbm
//!   `lambdarank_gh_iter1.txt` grad/hess golden (score-derivation route, iter-1 =
//!   zero-init scores). NOT `compare_exact` — the CUDA `atomicAdd_block` accumulation
//!   is the documented order-residual (D-05). Plus the shared==`_Sorted` variant
//!   equality and a twice-run determinism check.
//! - `rank_xendcg` — the per-query softmax cross-entropy-NDCG grad/hess (shared +
//!   `_GlobalMemory` variants) `compare_within(ORACLE_TOL)` vs the
//!   `lgbm_objective::rank::RankXendcg` f64 anchor. The `_GlobalMemory` path
//!   reproduces the CUDA hessian-buffer aliasing; shared and global hessians must
//!   match (the Pitfall 4 guard). Plus an RNG-replay cell asserting the device
//!   `draw_next_float_on` gamma stream is BIT-EXACT (`compare_exact_u32` on `to_bits`)
//!   vs the host `Random(seed + q)` stream — NEVER GPU-vs-GPU (D-05).
//!
//! Anchor discipline: NEVER GPU-vs-GPU (D-05 / def-f8u-01). The cubecl-cpu f64 fold is
//! the reference. The per-query DESCENDING-score item ranking composes
//! `bitonic_argsort_items_on`; the per-item RNG composes `draw_next_float_on`.

mod objective_common;

use lgbm_compute::kernels::objective_rank as obj;
use lgbm_compute::runtime::cpu_client;
use lgbm_metric::dcg_calculator::DcgCalculator;
use lgbm_objective::rank::Lambdarank;
use objective_common::{parse_gh, read_rank_golden};
use oracle_harness::comparator::{compare_exact_u32, compare_within, ORACLE_TOL};

/// The cubecl-cpu f64 anchor client type (oracle-harness stays cubecl-free at the
/// crate level, CMP-01 — only the re-exported client type is named here).
type Client = lgbm_compute::ComputeClientReexport<lgbm_compute::runtime::ActiveRuntime>;

/// The §5.4 lambdarank capture params (`rank_oracle_capture.py`): `sigmoid = 1.0`,
/// `lambdarank_norm = True`, `lambdarank_truncation_level = 5`.
const SIGMOID: f64 = 1.0;
const NORM: bool = true;
const TRUNC: i32 = 5;

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
            scores = v.split(',').map(|t| f64::from_bits(t.parse::<u64>().unwrap())).collect();
        } else if let Some(v) = line.strip_prefix("labels=") {
            labels = v.split(',').map(|t| f32::from_bits(t.parse::<u32>().unwrap())).collect();
        } else if let Some(v) = line.strip_prefix("query_boundaries=") {
            qb = v.split(',').map(|t| t.parse::<i32>().unwrap()).collect();
        }
    }
    (scores, labels, qb)
}

/// Build the init-once DCG tables the device launcher needs (owned by
/// `DcgCalculator`, supplied by the caller so `lgbm-compute` stays metric-crate-free):
/// `(label_gain[label], discount[rank], inverse_max_dcgs[query])`. Mirrors
/// `Lambdarank::new` (rank.rs:152-183).
fn dcg_tables(labels: &[f32], qb: &[i32]) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let label_gain = DcgCalculator::default_label_gain(&[]);
    let dcg = DcgCalculator::new(label_gain.clone());
    let max_cnt = qb.windows(2).map(|w| (w[1] - w[0]) as usize).max().unwrap_or(0);
    let discounts: Vec<f64> = (0..max_cnt).map(|r| dcg.discount(r)).collect();
    let mut inverse_max_dcgs = Vec::with_capacity(qb.len() - 1);
    for q in 0..qb.len() - 1 {
        let (s, e) = (qb[q] as usize, qb[q + 1] as usize);
        let mut m = dcg.cal_max_dcg_at_k(TRUNC as usize, &labels[s..e]);
        if m > 0.0 {
            m = 1.0 / m;
        }
        inverse_max_dcgs.push(m);
    }
    (label_gain, discounts, inverse_max_dcgs)
}

/// The host f64 anchor grad/hess (rank.rs `Lambdarank::get_gradients`).
fn host_lambdarank_gh(scores: &[f64], labels: &[f32], qb: &[i32]) -> (Vec<f32>, Vec<f32>) {
    let obj = Lambdarank::new(SIGMOID, NORM, TRUNC, &[], labels, qb).unwrap();
    let mut g = vec![0.0f32; scores.len()];
    let mut h = vec![0.0f32; scores.len()];
    obj.get_gradients(scores, &mut g, &mut h).expect("host lambdarank anchor");
    (g, h)
}

/// Assert device (shared + `_Sorted`) == host f64 anchor (D-05) within ORACLE_TOL,
/// and that the two device variants agree bit-exactly.
fn check_lambdarank(client: &Client, scores: &[f64], labels: &[f32], qb: &[i32], tag: &str) -> (Vec<f32>, Vec<f32>) {
    let (label_gain, discounts, inv_max) = dcg_tables(labels, qb);
    let (dg, dh) = obj::lambdarank_get_gradients_on(
        client, scores, labels, qb, &inv_max, &discounts, &label_gain, SIGMOID, NORM, TRUNC,
    )
    .expect("device lambdarank (shared)");
    let (sg, sh) = obj::lambdarank_get_gradients_sorted_on(
        client, scores, labels, qb, &inv_max, &discounts, &label_gain, SIGMOID, NORM, TRUNC,
    )
    .expect("device lambdarank (_Sorted >2048)");

    // The shared and >2048 `_Sorted` variants share the accumulation body → identical.
    compare_exact_u32(&bits(&dg), &bits(&sg))
        .unwrap_or_else(|m| panic!("{tag} lambdarank shared vs _Sorted grad differ: {m:?}"));
    compare_exact_u32(&bits(&dh), &bits(&sh))
        .unwrap_or_else(|m| panic!("{tag} lambdarank shared vs _Sorted hess differ: {m:?}"));

    // D-05 anchor: device == host f64 fold within ORACLE_TOL (atomic-order residual).
    let (hg, hh) = host_lambdarank_gh(scores, labels, qb);
    compare_within(&dg, &hg, ORACLE_TOL)
        .unwrap_or_else(|m| panic!("{tag} lambdarank grad vs host anchor: {m:?}"));
    compare_within(&dh, &hh, ORACLE_TOL)
        .unwrap_or_else(|m| panic!("{tag} lambdarank hess vs host anchor: {m:?}"));
    (dg, dh)
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

#[test]
fn lambdarank() {
    let client = cpu_client();

    // The corpus labels + query_boundaries come from the scores golden (skip-pass if
    // the fixture is absent — a fresh checkout builds without the capture run).
    let Some(scores_txt) = read_rank_golden("lambdarank_scores.txt") else {
        return;
    };
    let (final_scores, labels, qb) = parse_scores(&scores_txt);
    let n = labels.len();

    // iter-1: score_0 = 0 for every row (ranking has no boost-from-average init) — the
    // exact input the `lambdarank_gh_iter1` golden was derived from.
    let score0 = vec![0.0f64; n];
    let (dg, dh) = check_lambdarank(&client, &score0, &labels, &qb, "iter1");

    // Real-lib_lightgbm golden (score-derivation route), compare_within (atomic residual).
    if let Some(text) = read_rank_golden("lambdarank_gh_iter1.txt") {
        let (g_golden, h_golden) = parse_gh(&text);
        assert_eq!(g_golden.len(), n, "lambdarank golden grad width");
        compare_within(&dg, &g_golden, ORACLE_TOL)
            .unwrap_or_else(|m| panic!("lambdarank grad vs gh_iter1 golden: {m:?}"));
        compare_within(&dh, &h_golden, ORACLE_TOL)
            .unwrap_or_else(|m| panic!("lambdarank hess vs gh_iter1 golden: {m:?}"));
    }

    // Twice-run determinism on the cpu anchor (single-owner ⇒ bit-stable).
    let (label_gain, discounts, inv_max) = dcg_tables(&labels, &qb);
    let (g1, h1) = obj::lambdarank_get_gradients_on(
        &client, &score0, &labels, &qb, &inv_max, &discounts, &label_gain, SIGMOID, NORM, TRUNC,
    )
    .unwrap();
    compare_exact_u32(&bits(&dg), &bits(&g1)).unwrap_or_else(|m| panic!("lambdarank grad determinism: {m:?}"));
    compare_exact_u32(&bits(&dh), &bits(&h1)).unwrap_or_else(|m| panic!("lambdarank hess determinism: {m:?}"));

    // Device == host anchor on the DISTINCT final trained scores (no tie ambiguity —
    // an unambiguous-sort cross-check that bitonic == the host stable sort).
    check_lambdarank(&client, &final_scores, &labels, &qb, "final");
}
