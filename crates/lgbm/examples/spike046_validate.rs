//! Spike-046 throwaway validator: confirm the dump("train") hook added to
//! booster.rs::train_inner_columns_full fires on the SHIPPED public `train()`
//! path (the one the Python wheel drives). Run with:
//!   LGBM_PHASE_PROF=1 cargo run --release --example spike046_validate
//! Expect a `[phase_prof:train] ... BUDGET: ...` block on stderr. Delete after.

use lgbm::{train, DenseCorpus, TrainingBuilder};

fn main() {
    // spike-049: rows/iters overridable so the SAME train() path can be profiled at
    // the 500k×50 repro shape (LGBM_VALIDATE_ROWS=500000 LGBM_VALIDATE_ITERS=100).
    let rows = std::env::var("LGBM_VALIDATE_ROWS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(20_000usize);
    let iters = std::env::var("LGBM_VALIDATE_ITERS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(20i32);
    let feats = 50usize;
    let bins = 32usize;
    let mut features: Vec<Vec<f64>> = Vec::with_capacity(rows);
    let mut labels: Vec<f32> = Vec::with_capacity(rows);
    for row in 0..rows {
        let mut frow = Vec::with_capacity(feats);
        let mut acc = 0.0f64;
        for f in 0..feats {
            let v = if row < bins {
                (row + f) % bins
            } else {
                row.wrapping_mul(2_654_435_761)
                    .wrapping_add(f.wrapping_mul(40_503).wrapping_add(0x9E37_79B9))
                    % bins
            };
            frow.push(v as f64);
            acc += (v as f64) * (1.0 + (f % 5) as f64);
        }
        features.push(frow);
        // Real, balanced signal: label = (feature-weighted sum above its own running
        // threshold) — gives the learner genuine splits to find (not constant trees).
        let thr = (bins as f64) * (feats as f64) * 1.5;
        labels.push((acc > thr) as i32 as f32);
    }
    let corpus = DenseCorpus { features, labels };
    let cfg = TrainingBuilder::new()
        .objective("binary")
        .num_iterations(iters)
        .num_leaves(31)
        .learning_rate(0.1)
        .seed(42)
        .build()
        .expect("config builds");
    let _b = train(&cfg, &corpus).expect("train ok");
    eprintln!("[spike046_validate] train complete");
}
