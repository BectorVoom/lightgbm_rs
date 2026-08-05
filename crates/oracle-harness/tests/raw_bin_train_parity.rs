//! Phase-8 (plan 08-01, D-02 raw→bin→train bridge) ORACLE parity for the
//! `lgbm` facade's RAW-corpus training path.
//!
//! Two tests:
//!
//! - `raw_bin_train_matches_identity_bin` — trains a corpus of integer-valued
//!   features via BOTH `lgbm::train` (the DenseCorpus identity-bin path) and
//!   `lgbm::train_raw` (the RawCorpus BinMapper path) and asserts the two models'
//!   tree leaf values are BIT-EXACT (`compare_exact_f64_bits`). This proves the
//!   raw bridge collapses to the identity path on the trivial case (the values are
//!   already `0..K-1`, so the BinMapper produces identity bins). TEETH: perturbing
//!   one leaf value would fail.
//!
//! - `raw_bin_train_matches_cpp_golden` — trains a RawCorpus of REAL continuous
//!   values under the pinned deterministic settings. If a committed real-value C++
//!   golden is reusable it asserts predictions within `ORACLE_TOL`; otherwise it
//!   SKIP-passes with a printed `golden absent` marker (the advanced_parity SKIP
//!   discipline — never a silent pass).
//!
//! Deterministic settings (REFERENCE_MANIFEST): `deterministic=true`,
//! `force_row_wise=true`, `num_threads=1`, fixed seed, `min_data_in_bin=1` (so
//! each distinct integer value gets its OWN bin — the identity-binning precondition
//! the trivial-case equivalence relies on).
//!
//! `LightGBM/` is NEVER git-added.

use std::path::PathBuf;

use lgbm::{train, train_raw, Booster, Config, DenseCorpus, RawCorpus, TrainingBuilder};
use oracle_harness::comparator::{compare_exact_f64_bits, compare_within, ORACLE_TOL};

/// A pinned deterministic regression config (REFERENCE_MANIFEST shape).
/// `min_data_in_bin = 1` so the BinMapper does NOT merge adjacent integer bins
/// (the identity-binning precondition the trivial-case equivalence relies on).
fn det_config() -> Config {
    let mut cfg = TrainingBuilder::new()
        .objective("regression")
        .num_iterations(10)
        .learning_rate(0.1)
        .num_leaves(4)
        .min_data_in_leaf(1)
        .boost_from_average(true)
        .metric("l2,rmse")
        .seed(1)
        .deterministic(true)
        .build()
        .expect("config builds");
    cfg.force_row_wise = true;
    cfg.num_threads = 1;
    cfg.min_data_in_bin = 1;
    cfg.data_random_seed = 1;
    cfg
}

/// The learner-parity spine corpus: integer-valued features `0..K-1` (so the
/// BinMapper produces identity bins) + monotone labels.
fn spine_corpus() -> DenseCorpus {
    let f0 = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5];
    let f1 = [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2];
    let features: Vec<Vec<f64>> = (0..12).map(|i| vec![f0[i] as f64, f1[i] as f64]).collect();
    let labels = vec![
        2.0f32, 3.0, 5.0, 6.0, 9.0, 10.0, 12.0, 13.0, 16.0, 17.0, 19.0, 20.0,
    ];
    DenseCorpus { features, labels, query_boundaries: Vec::new() }
}

/// All leaf values across the model's trees, flattened in tree-then-leaf order
/// (the comparison vector for the bit-exact gate).
fn flat_leaf_values(booster: &Booster) -> Vec<f64> {
    booster
        .model()
        .trees
        .iter()
        .flat_map(|t| t.leaf_value.clone())
        .collect()
}

#[test]
fn raw_bin_train_matches_identity_bin() {
    let cfg = det_config();
    let corpus = spine_corpus();

    // Identity-bin path (DenseCorpus).
    let id_booster = train(&cfg, &corpus).expect("identity train ok");

    // Raw→bin→train path (RawCorpus, binned internally by the BinMapper).
    let raw = {
        let mut r = RawCorpus::new(corpus.features.clone(), corpus.labels.clone());
        r.config = cfg.clone();
        r
    };
    let raw_booster = train_raw(&cfg, &raw).expect("raw train ok");

    // Same tree count.
    assert_eq!(
        id_booster.model().trees.len(),
        raw_booster.model().trees.len(),
        "identity and raw paths must grow the same number of trees"
    );

    // Leaf values BIT-EXACT (the raw bridge collapses to the identity path).
    let id_leaves = flat_leaf_values(&id_booster);
    let raw_leaves = flat_leaf_values(&raw_booster);
    compare_exact_f64_bits(&raw_leaves, &id_leaves)
        .expect("raw-path leaf values must be bit-exact to the identity path");

    // TEETH self-check: the comparator IS sensitive — a perturbed leaf fails.
    let mut perturbed = id_leaves.clone();
    if let Some(first) = perturbed.first_mut() {
        *first = f64::from_bits(first.to_bits() ^ 1); // flip one ULP
        assert!(
            compare_exact_f64_bits(&raw_leaves, &perturbed).is_err(),
            "bit-exact comparator must reject a one-ULP perturbation (teeth)"
        );
    }
}

#[test]
fn raw_bin_train_matches_cpp_golden() {
    // A real-value RawCorpus trained under the pinned deterministic settings. If a
    // committed real-value C++ golden becomes reusable, assert predictions within
    // ORACLE_TOL; until then, SKIP-pass with a printed marker (advanced_parity
    // SKIP discipline — never a silent pass).
    let golden_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/raw_bin/real_value_golden.txt");

    let cfg = det_config();
    let f0 = [
        0.1f64, 0.15, 1.2, 1.25, 2.7, 2.75, 3.9, 3.95, 5.1, 5.15, 6.6, 6.65,
    ];
    let f1 = [
        0.0f64, 1.1, 2.2, 0.05, 1.15, 2.25, 0.1, 1.2, 2.3, 0.0, 1.1, 2.2,
    ];
    let features: Vec<Vec<f64>> = (0..12).map(|i| vec![f0[i], f1[i]]).collect();
    let labels = vec![
        2.0f32, 3.0, 5.0, 6.0, 9.0, 10.0, 12.0, 13.0, 16.0, 17.0, 19.0, 20.0,
    ];
    let raw = {
        let mut r = RawCorpus::new(features.clone(), labels);
        r.config = cfg.clone();
        r
    };
    let booster = train_raw(&cfg, &raw).expect("raw real-value train ok");

    if !golden_path.exists() {
        eprintln!(
            "raw_bin_train_parity: SKIP — golden absent ({}) — capture deferred \
             (no reusable real-value C++ golden with raw values + model yet)",
            golden_path.display()
        );
        // Even when SKIP-marked, prove the model trains + predicts coherently (so
        // the test still exercises the raw path end-to-end).
        let preds: Vec<f32> = features.iter().map(|r| booster.predict_row(r)[0]).collect();
        assert_eq!(preds.len(), features.len());
        assert!(
            preds[0] < preds[preds.len() - 1],
            "monotone sanity: low-label row {} < high-label row {}",
            preds[0],
            preds[preds.len() - 1]
        );
        return;
    }

    // Golden present: parse one prediction per row and compare within ORACLE_TOL.
    let text = std::fs::read_to_string(&golden_path).expect("read golden");
    let golden: Vec<f32> = text
        .split_whitespace()
        .map(|t| t.parse::<f32>().expect("golden value parses"))
        .collect();
    let preds: Vec<f32> = features.iter().map(|r| booster.predict_row(r)[0]).collect();
    assert_eq!(preds.len(), golden.len(), "golden length must match rows");
    compare_within(&preds, &golden, ORACLE_TOL)
        .expect("raw-path predictions within the numerical contract vs the C++ golden");
}
