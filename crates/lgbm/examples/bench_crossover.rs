//! GPU-vs-CPU **crossover** sweep harness (spike 001).
//!
//! Same engine as `bench_train.rs`, but the size list / iters / reps are
//! configurable at **runtime** via env vars so a single (expensive) `--features
//! rocm` fat-LTO build can be swept across many shapes without recompiling.
//!
//! Backend is compile-time:
//!   CPU anchor:  cargo run --release --example bench_crossover
//!   ROCm GPU:    cargo run --release --features rocm --example bench_crossover
//!
//! Env knobs (all optional):
//!   BENCH_SIZES = "name:rows:feat:bins,name:rows:feat:bins,..."
//!   BENCH_ITERS = num_iterations (default 30)
//!   BENCH_REPS  = train repeats for the median (default 3)
//!
//! Train time scales ~linearly in iters, so a smaller iter count keeps the GPU
//! sweep tractable while scaling both backends identically — the crossover row
//! count is iter-count-invariant. Identity-binned, no RNG → reproducible.

#[cfg(feature = "mimalloc")]
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

use std::time::{Duration, Instant};

use lgbm::{train, DenseCorpus, TrainingBuilder};

struct Size {
    name: String,
    rows: usize,
    features: usize,
    bins: usize,
}

/// Deterministic identity-binned corpus (same recipe as `bench_train.rs`).
fn make_corpus(s: &Size) -> DenseCorpus {
    let mut features: Vec<Vec<f64>> = Vec::with_capacity(s.rows);
    let mut labels: Vec<f32> = Vec::with_capacity(s.rows);
    for row in 0..s.rows {
        let mut frow: Vec<f64> = Vec::with_capacity(s.features);
        let mut acc: f64 = 0.0;
        for f in 0..s.features {
            let v = if row < s.bins {
                (row + f) % s.bins
            } else {
                let h = row
                    .wrapping_mul(2_654_435_761)
                    .wrapping_add(f.wrapping_mul(40_503).wrapping_add(0x9E37_79B9));
                h % s.bins
            };
            frow.push(v as f64);
            acc += (v as f64) * (1.0 + (f % 5) as f64);
        }
        features.push(frow);
        labels.push((acc * 0.01) as f32);
    }
    DenseCorpus { features, labels }
}

fn median(mut ds: Vec<Duration>) -> Duration {
    ds.sort();
    ds[ds.len() / 2]
}

fn default_sizes() -> Vec<Size> {
    // name, rows, features, bins. Fixed feat=50/bins=255 so only rows vary —
    // isolates the row-count axis the crossover hypothesis is about.
    [
        ("20k", 20_000usize),
        ("50k", 50_000),
        ("100k", 100_000),
        ("200k", 200_000),
        ("500k", 500_000),
    ]
    .iter()
    .map(|(n, r)| Size { name: (*n).to_string(), rows: *r, features: 50, bins: 255 })
    .collect()
}

fn parse_sizes() -> Vec<Size> {
    match std::env::var("BENCH_SIZES") {
        Ok(spec) if !spec.trim().is_empty() => spec
            .split(',')
            .filter(|s| !s.trim().is_empty())
            .map(|tok| {
                let p: Vec<&str> = tok.split(':').collect();
                assert!(p.len() == 4, "BENCH_SIZES entry must be name:rows:feat:bins, got {tok:?}");
                Size {
                    name: p[0].to_string(),
                    rows: p[1].parse().expect("rows"),
                    features: p[2].parse().expect("feat"),
                    bins: p[3].parse().expect("bins"),
                }
            })
            .collect(),
        _ => default_sizes(),
    }
}

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}

fn main() {
    #[cfg(feature = "rocm")]
    let backend = "ROCm-GPU";
    #[cfg(not(feature = "rocm"))]
    let backend = "CPU-anchor";

    let iters = env_usize("BENCH_ITERS", 30) as i32;
    let reps = env_usize("BENCH_REPS", 3);
    const LEAVES: i32 = 31;

    let sizes = parse_sizes();

    println!("# crossover sweep  backend={backend}  iters={iters}  reps={reps}  leaves={LEAVES}");
    println!(
        "{:<8} {:>9} {:>5} {:>5}  {:>13} {:>14}",
        "size", "rows", "feat", "bins", "train_median", "train_rows/s"
    );

    for s in &sizes {
        let corpus = make_corpus(s);
        let cfg = TrainingBuilder::new()
            .objective("regression")
            .num_iterations(iters)
            .num_leaves(LEAVES)
            .learning_rate(0.1)
            .min_data_in_leaf(20)
            .seed(42)
            .deterministic(true)
            .build()
            .expect("config builds");

        let mut train_times = Vec::with_capacity(reps);
        for _ in 0..reps {
            let t0 = Instant::now();
            let booster = train(&cfg, &corpus).expect("train ok");
            train_times.push(t0.elapsed());
            std::hint::black_box(&booster);
        }
        // Spike 002: per-phase wall-clock breakdown (env LGBM_PHASE_PROF=1).
        lgbm_treelearner::phase_prof::dump(&s.name);
        let train_med = median(train_times);
        let rows_per_s = s.rows as f64 / train_med.as_secs_f64();
        println!(
            "{:<8} {:>9} {:>5} {:>5}  {:>11.2?} {:>14.0}",
            s.name, s.rows, s.features, s.bins, train_med, rows_per_s
        );
        // Flush each row so a slow GPU sweep streams progress instead of buffering.
        use std::io::Write;
        std::io::stdout().flush().ok();
    }
}
