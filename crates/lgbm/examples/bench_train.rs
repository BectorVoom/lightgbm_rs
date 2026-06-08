//! Train/predict speed benchmark for the pure-Rust `lightgbm_rs` engine.
//!
//! Rust-only before/after instrument for quick task 260608-iwj. Deterministic
//! synthetic identity-binned corpora at a few sizes; reports median wall-clock for
//! `train` and batch `predict` over N repeats. No RNG, no I/O — the numbers are
//! directly comparable across builds.
//!
//! Run (system allocator):    cargo run --release --example bench_train
//! Run (mimalloc allocator):  cargo run --release --features mimalloc --example bench_train
//!
//! The `#[global_allocator]` below only takes effect when the `mimalloc` feature is
//! on; with the feature off this is the system allocator. This is intentional — it
//! lets the M0/M1 baseline runs measure the default allocator and M2/M3 measure
//! mimalloc, with the exact same harness binary logic.

#[cfg(feature = "mimalloc")]
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

use std::time::{Duration, Instant};

use lgbm::{train, DenseCorpus, TrainingBuilder};

/// A benchmark size: `rows` × `features`, each feature identity-binned into
/// `bins` distinct integer values (`0..bins-1`, all present).
struct Size {
    name: &'static str,
    rows: usize,
    features: usize,
    bins: usize,
}

/// Build a deterministic identity-binned corpus. Each feature `f` takes integer
/// values in `0..bins`; the first `bins` rows are forced to `0,1,..,bins-1` to
/// guarantee full bin coverage (the `DenseCorpus` identity-binning precondition),
/// and the remaining rows are filled by a fixed integer hash (no RNG → fully
/// reproducible). The label is a deterministic linear combination of the features
/// so the trees have real structure to learn.
fn make_corpus(s: &Size) -> DenseCorpus {
    let mut features: Vec<Vec<f64>> = Vec::with_capacity(s.rows);
    let mut labels: Vec<f32> = Vec::with_capacity(s.rows);
    for row in 0..s.rows {
        let mut frow: Vec<f64> = Vec::with_capacity(s.features);
        let mut acc: f64 = 0.0;
        for f in 0..s.features {
            let v = if row < s.bins {
                // calibration block: guarantees every bin 0..bins-1 appears.
                (row + f) % s.bins
            } else {
                // fixed integer hash → reproducible, well-spread bins.
                let h = row
                    .wrapping_mul(2_654_435_761)
                    .wrapping_add(f.wrapping_mul(40_503).wrapping_add(0x9E37_79B9));
                h % s.bins
            };
            frow.push(v as f64);
            // weight features unequally so splits are informative.
            acc += (v as f64) * (1.0 + (f % 5) as f64);
        }
        features.push(frow);
        labels.push((acc * 0.01) as f32);
    }
    DenseCorpus { features, labels }
}

/// Median of a slice of durations (sorts a copy).
fn median(mut ds: Vec<Duration>) -> Duration {
    ds.sort();
    ds[ds.len() / 2]
}

fn main() {
    // Allocator label for the report header.
    #[cfg(feature = "mimalloc")]
    let alloc = "mimalloc";
    #[cfg(not(feature = "mimalloc"))]
    let alloc = "system";

    // Fixed training config across all sizes — a standard regression GBDT.
    const ITERS: i32 = 100;
    const LEAVES: i32 = 31;
    // Repeats per measurement (median): trade wall-clock for stability.
    const TRAIN_REPS: usize = 5;
    const PREDICT_REPS: usize = 7;

    let sizes = [
        Size { name: "small ", rows: 2_000, features: 12, bins: 32 },
        Size { name: "medium", rows: 8_000, features: 30, bins: 64 },
        Size { name: "large ", rows: 20_000, features: 50, bins: 128 },
    ];

    println!("# lightgbm_rs bench  (allocator: {alloc}, iters: {ITERS}, leaves: {LEAVES})");
    println!(
        "{:<7} {:>7} {:>5} {:>5}  {:>12} {:>14}  {:>12}",
        "size", "rows", "feat", "bins", "train_median", "train_rows/s", "predict_med"
    );

    for s in &sizes {
        let corpus = make_corpus(s);
        let cfg = TrainingBuilder::new()
            .objective("regression")
            .num_iterations(ITERS)
            .num_leaves(LEAVES)
            .learning_rate(0.1)
            .min_data_in_leaf(20)
            .seed(42)
            .deterministic(true)
            .build()
            .expect("config builds");

        // ---- train timing (median of TRAIN_REPS) ----
        let mut train_times = Vec::with_capacity(TRAIN_REPS);
        let mut last_booster = None;
        for _ in 0..TRAIN_REPS {
            let t0 = Instant::now();
            let booster = train(&cfg, &corpus).expect("train ok");
            train_times.push(t0.elapsed());
            last_booster = Some(booster);
        }
        let booster = last_booster.unwrap();
        let train_med = median(train_times);

        // ---- predict timing (median of PREDICT_REPS) over the full corpus ----
        let mut predict_times = Vec::with_capacity(PREDICT_REPS);
        let mut sink = 0.0f64; // defeat dead-code elimination
        for _ in 0..PREDICT_REPS {
            let t0 = Instant::now();
            let preds = booster.predict(&corpus.features);
            predict_times.push(t0.elapsed());
            sink += preds[0][0] as f64;
        }
        let predict_med = median(predict_times);
        std::hint::black_box(sink);

        let rows_per_s = s.rows as f64 / train_med.as_secs_f64();
        println!(
            "{:<7} {:>7} {:>5} {:>5}  {:>10.2?} {:>14.0}  {:>10.2?}",
            s.name, s.rows, s.features, s.bins, train_med, rows_per_s, predict_med
        );
    }
}
