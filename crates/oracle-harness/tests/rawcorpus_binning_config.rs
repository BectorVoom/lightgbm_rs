//! Regression: `train_raw`/`train` must bin with the TRAINING config's `max_bin`, not the
//! `RawCorpus`'s own (default) config. The phase-10 RawCorpus-gap root cause: `RawCorpus::new`
//! defaults `config` (max_bin=255), and binning silently used THAT instead of the `config`
//! passed to `train_raw` — so a caller-supplied `max_bin` was ignored, producing a ~0.28
//! divergence from C++ on continuous data. Fixed by `build_feature_columns_from_raw_with_config`.
//!
//! Self-contained (no Python golden): with `max_bin = 2` a single continuous feature can yield
//! at most ONE split threshold (2 bins → 1 boundary). Under the bug (max_bin ignored → ~N bins)
//! a deep tree splits at many distinct thresholds. So "≤1 distinct threshold" proves max_bin is
//! honored.

use lgbm::{train_raw, Config, RawCorpus, TrainingBuilder};

fn distinct_thresholds(model_text: &str) -> usize {
    let mut set = std::collections::BTreeSet::new();
    for line in model_text.lines() {
        if let Some(rest) = line.strip_prefix("threshold=") {
            for t in rest.split_whitespace() {
                // bit-key the f64 so near-equal values still count distinctly (we want the raw count)
                set.insert(t.parse::<f64>().unwrap_or(0.0).to_bits());
            }
        }
    }
    set.len()
}

fn cfg_with_max_bin(max_bin: i32) -> Config {
    let mut c: Config = TrainingBuilder::new()
        .objective("regression").num_iterations(1).num_leaves(31).min_data_in_leaf(1)
        .seed(1).deterministic(true).build().unwrap();
    c.max_bin = max_bin;
    c.num_threads = 1;
    c.force_row_wise = true;
    c.feature_pre_filter = false;
    c
}

#[test]
fn train_raw_honors_training_config_max_bin() {
    // One continuous feature, 40 distinct values (seeded LCG — self-contained).
    let mut s = 0x1234_5678u64;
    let mut next = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        (s >> 40) as f64 / (1u64 << 24) as f64
    };
    let col: Vec<f64> = (0..40).map(|_| next()).collect();
    let rows: Vec<Vec<f64>> = col.iter().map(|&v| vec![v]).collect();
    let labels: Vec<f32> = col.iter().map(|&v| v as f32).collect();

    // RawCorpus::new defaults config to max_bin=255 — the footgun. The TRAIN config wins now.
    let m2 = train_raw(&cfg_with_max_bin(2), &RawCorpus::new(rows.clone(), labels.clone())).unwrap();
    let t2 = distinct_thresholds(&m2.model_to_string());
    assert!(t2 <= 1, "max_bin=2 must yield ≤1 split threshold, got {t2} (max_bin ignored?)");

    // Sanity: a generous max_bin produces MORE distinct thresholds (binning actually varies).
    let m_many = train_raw(&cfg_with_max_bin(255), &RawCorpus::new(rows, labels)).unwrap();
    let t_many = distinct_thresholds(&m_many.model_to_string());
    assert!(t_many > t2, "max_bin=255 should split at more thresholds than max_bin=2 ({t_many} vs {t2})");
}
