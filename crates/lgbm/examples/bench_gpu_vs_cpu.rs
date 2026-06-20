//! GPU(rocm)-vs-CPU whole-tree-learner train benchmark for `lightgbm_rs`
//! (quick task 260619-j9t — the speed-comparison deliverable for the CubeCL CUDA
//! mirror port).
//!
//! The GPU vs CPU backend is selected at **COMPILE TIME** by `--features rocm`
//! (`booster.rs` cfg-gates `RocmBackend` vs the `CpuBackend` f64 anchor). So this
//! example reports which backend it was built with and the operator runs it TWICE
//! to get the side-by-side numbers:
//!
//!   CPU (multi-threaded f64 anchor):
//!     cargo run --release --example bench_gpu_vs_cpu
//!   GPU (gfx1100 via cubecl-hip):
//!     cargo run --release --features rocm --example bench_gpu_vs_cpu
//!
//! WARM-VS-COLD RULE (spike-findings SKILL.md, load-bearing): the cold isolated
//! ceiling overstates the warm win 3–7× (allocator/JIT amortization). So BOTH paths
//! run `WARMUP` train iterations (DISCARDED) before the timed median loop, and we
//! report the **median of `TRAIN_REPS`** warm runs.
//!
//! The CPU column is the MULTI-THREADED anchor (feature-parallel histogram build is
//! rayon-parallel at leaf_rows >= 16384, spike-005) — at large leaves the CPU path
//! uses all cores, so the GPU only competes well above small leaves (the GPU build
//! is atomic/latency-bound, spike-006). The >=200k-row size below is where the GPU
//! path stops being purely launch-bound.

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

/// Build a deterministic identity-binned corpus (no RNG → fully reproducible across
/// builds, so the CPU and GPU runs train on byte-identical data). Mirrors
/// `bench_train.rs::make_corpus`: the first `bins` rows force full bin coverage (the
/// `DenseCorpus` identity-binning precondition); the rest are a fixed integer hash.
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

/// Median of a slice of durations (sorts a copy).
fn median(mut ds: Vec<Duration>) -> Duration {
    ds.sort();
    ds[ds.len() / 2]
}

fn main() {
    // The backend is selected at COMPILE TIME by `--features rocm`.
    #[cfg(feature = "rocm")]
    let backend = "rocm(gfx1100)";
    #[cfg(not(feature = "rocm"))]
    let backend = "cpu-f64-anchor";

    // Standard regression GBDT, same config for both backends.
    const LEAVES: i32 = 31;
    // WARM-VS-COLD RULE: discard WARMUP iters before timing; report median of TRAIN_REPS.
    const WARMUP: usize = 2;
    const TRAIN_REPS: usize = 5;
    // Sweep mode trains larger corpora (up to 512k rows); fewer iters keep the multi-
    // restart A/B wall time bounded while still warm-amortizing. Default keeps 50.
    let iters: i32 = if std::env::var("LGBM_BENCH_SWEEP").as_deref() == Ok("medocc") {
        12
    } else {
        50
    };
    let iters = iters; // bind for the closure below

    // Sizes: small/medium are launch-bound on the GPU; the >=200k size is where the
    // GPU path stops being purely launch-bound (and where the CPU anchor goes
    // multi-threaded). Fewer iters at 200k to keep the run time reasonable.
    //
    // SWEEP MODE (quick-260620-sqf): when `LGBM_BENCH_SWEEP=medocc` is set, replace
    // the default sizes with the medium-width occupancy A/B grid — feature counts
    // {30,60,100} × rows {50k,120k,256k,512k} bracketing the 256k row-partition gate.
    // The A/B (P=1 vs row-partitioned) is driven externally via `LGBM_ROWPART_MIN`
    // (the existing override at histogram.rs:731) so NO source change to the gate is
    // needed for the measurement. Methodology (warm/median) is unchanged.
    let sweep = std::env::var("LGBM_BENCH_SWEEP").ok();
    let default_sizes = vec![
        Size { name: "small ", rows: 2_000, features: 12, bins: 32 },
        Size { name: "medium", rows: 20_000, features: 30, bins: 64 },
        Size { name: "large ", rows: 200_000, features: 40, bins: 128 },
    ];
    let medocc_sizes: Vec<Size> = {
        let mut v = Vec::new();
        for &feat in &[30usize, 60, 100] {
            for &rows in &[50_000usize, 120_000, 256_000, 512_000] {
                // Leak a small static-ish name; bench example, fine to leak.
                let name: &'static str =
                    Box::leak(format!("f{feat}r{}k", rows / 1000).into_boxed_str());
                v.push(Size { name, rows, features: feat, bins: 128 });
            }
        }
        v
    };
    let sizes: Vec<Size> = match sweep.as_deref() {
        Some("medocc") => medocc_sizes,
        _ => default_sizes,
    };

    println!(
        "# lightgbm_rs GPU-vs-CPU bench  (backend: {backend}, iters: {iters}, leaves: {LEAVES}, warmup: {WARMUP}, reps: {TRAIN_REPS}, rowpart_min: {})",
        std::env::var("LGBM_ROWPART_MIN").unwrap_or_else(|_| "default(256000)".into())
    );
    println!(
        "# RUN BOTH FEATURE CONFIGS FOR THE SIDE-BY-SIDE COMPARISON:\n\
         #   CPU: cargo run --release --example bench_gpu_vs_cpu\n\
         #   GPU: cargo run --release --features rocm --example bench_gpu_vs_cpu\n\
         # (CPU column is the MULTI-THREADED f64 anchor; GPU build is atomic/latency-bound.)"
    );
    println!(
        "{:<7} {:>8} {:>5} {:>5}  {:>14} {:>14}",
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

        // ---- WARM-UP (discarded): amortize allocator/JIT so the timed loop is warm.
        for _ in 0..WARMUP {
            let _ = train(&cfg, &corpus).expect("warm-up train ok");
        }

        // ---- timed train (median of TRAIN_REPS warm runs) ----
        let mut train_times = Vec::with_capacity(TRAIN_REPS);
        let mut sink = 0.0f64; // defeat dead-code elimination
        for _ in 0..TRAIN_REPS {
            let t0 = Instant::now();
            let booster = train(&cfg, &corpus).expect("train ok");
            train_times.push(t0.elapsed());
            // touch the model so the train isn't optimized away.
            sink += booster.predict(&corpus.features[0..1])[0][0] as f64;
        }
        std::hint::black_box(sink);
        let train_med = median(train_times);
        let rows_per_s = s.rows as f64 / train_med.as_secs_f64();

        println!(
            "{:<7} {:>8} {:>5} {:>5}  {:>12.2?} {:>14.0}",
            s.name, s.rows, s.features, s.bins, train_med, rows_per_s
        );
    }
}
