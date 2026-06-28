//! Spike-050: attribute the Python-side ~25% chunk (numpy marshalling + binning)
//! that spike-049 left unbroken-down. Both are HOST/CPU, backend-independent, so a
//! pure-Rust proxy at the 500k×50 repro shape measures them without a wheel/Kaggle.
//!
//! Run: LGBM_PHASE_PROF=1 cargo run --release --example spike050_marshalling
//! Reports: (1) numpy→Vec<Vec<f64>> conversion proxy (dense_any_to_rows does the same
//! per-row `.collect()` — 500k inner-Vec allocs), (2) train_raw's binning (now wrapped
//! in BINNING_NS, spike-050) via the phase_prof dump.

use std::time::Instant;

use lgbm::{train_raw, RawCorpus, TrainingBuilder};

fn main() {
    let rows_n = std::env::var("LGBM_VALIDATE_ROWS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(500_000usize);
    let feats = 50usize;
    let iters = std::env::var("LGBM_VALIDATE_ITERS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(10i32);

    // A contiguous row-major f64 buffer == what a numpy 2-D array hands the pyo3 layer.
    // Continuous values (not identity-binned) so binning does real find_bin work.
    let mut flat: Vec<f64> = Vec::with_capacity(rows_n * feats);
    for r in 0..rows_n {
        for f in 0..feats {
            let h = r
                .wrapping_mul(2_654_435_761)
                .wrapping_add(f.wrapping_mul(40_503).wrapping_add(0x9E37_79B9));
            flat.push((h % 100_003) as f64 / 100_003.0); // continuous in [0,1)
        }
    }
    let labels: Vec<f32> = (0..rows_n).map(|r| (r % 2) as f32).collect();

    // (1) MARSHALLING proxy: contiguous buffer → Vec<Vec<f64>> (dense_any_to_rows does
    // `rows.push(row.iter().copied().collect())` per row — 500k small allocations).
    let t0 = Instant::now();
    let rows: Vec<Vec<f64>> = flat.chunks(feats).map(|c| c.to_vec()).collect();
    let marshal_ms = t0.elapsed().as_secs_f64() * 1e3;
    eprintln!(
        "[spike050] numpy->Vec<Vec<f64>> marshalling proxy: {marshal_ms:.3} ms ({rows_n} rows x {feats})"
    );

    let corpus = RawCorpus::new(rows, labels);
    let cfg = TrainingBuilder::new()
        .objective("binary")
        .num_iterations(iters)
        .num_leaves(31)
        .learning_rate(0.1)
        .seed(42)
        .build()
        .expect("config builds");

    // (2) train_raw: binning (now BINNING_NS-wrapped, spike-050) + the train loop.
    let t1 = Instant::now();
    let _b = train_raw(&cfg, &corpus).expect("train_raw ok");
    eprintln!(
        "[spike050] train_raw total (bin + {iters}-iter loop): {:.3} ms",
        t1.elapsed().as_secs_f64() * 1e3
    );
}
