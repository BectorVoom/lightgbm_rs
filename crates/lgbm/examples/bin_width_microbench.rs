//! Spike 004 micro-benchmark: does narrowing the bin column (u32 → u16 → u8)
//! speed the histogram gather+fold at large rows? Isolates the cache effect of
//! the random `bins[leaf_rows[i]]` access WITHOUT plumbing a narrow type through
//! the whole dataset. Mirrors the fused fold in `build_leaf_histograms_raw`:
//!   for i in 0..r: bin = bins[leaf_rows[i]]; hist[bin*2]+=g[i]; hist[bin*2+1]+=h[i]
//! over F independent feature columns (the real per-feature working-set pressure).
//!
//! Run: cargo run --release --example bin_width_microbench

use std::time::{Duration, Instant};

const ROWS: usize = 200_000;
const FEATS: usize = 32;
const NUM_BIN: usize = 64;
const REPS: usize = 15;

fn median(mut v: Vec<Duration>) -> Duration {
    v.sort();
    v[v.len() / 2]
}

/// Deterministic "random" permutation of 0..ROWS (no RNG dep): a full-period LCG
/// walk gives a scattered, cache-unfriendly access order like real leaf_rows.
fn scattered_rows() -> Vec<u32> {
    // Fisher–Yates with a fixed integer hash as the swap source (reproducible).
    let mut idx: Vec<u32> = (0..ROWS as u32).collect();
    let mut state: u64 = 0x9E37_79B9_7F4A_7C15;
    for i in (1..ROWS).rev() {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let j = (state >> 33) as usize % (i + 1);
        idx.swap(i, j);
    }
    idx
}

fn fold_u32(cols: &[Vec<u32>], rows: &[u32], g: &[f32], h: &[f32], scratch: &mut [f64]) -> f64 {
    let mut sink = 0.0;
    for col in cols {
        for c in scratch.iter_mut() {
            *c = 0.0;
        }
        for i in 0..rows.len() {
            let bin = col[rows[i] as usize] as usize;
            scratch[bin * 2] += f64::from(g[i]);
            scratch[bin * 2 + 1] += f64::from(h[i]);
        }
        sink += scratch[0];
    }
    sink
}

fn fold_u16(cols: &[Vec<u16>], rows: &[u32], g: &[f32], h: &[f32], scratch: &mut [f64]) -> f64 {
    let mut sink = 0.0;
    for col in cols {
        for c in scratch.iter_mut() {
            *c = 0.0;
        }
        for i in 0..rows.len() {
            let bin = col[rows[i] as usize] as usize;
            scratch[bin * 2] += f64::from(g[i]);
            scratch[bin * 2 + 1] += f64::from(h[i]);
        }
        sink += scratch[0];
    }
    sink
}

fn fold_u8(cols: &[Vec<u8>], rows: &[u32], g: &[f32], h: &[f32], scratch: &mut [f64]) -> f64 {
    let mut sink = 0.0;
    for col in cols {
        for c in scratch.iter_mut() {
            *c = 0.0;
        }
        for i in 0..rows.len() {
            let bin = col[rows[i] as usize] as usize;
            scratch[bin * 2] += f64::from(g[i]);
            scratch[bin * 2 + 1] += f64::from(h[i]);
        }
        sink += scratch[0];
    }
    sink
}

fn main() {
    // Build F feature columns of bins in 0..NUM_BIN (deterministic hash fill).
    let cols_u32: Vec<Vec<u32>> = (0..FEATS)
        .map(|f| {
            (0..ROWS)
                .map(|r| {
                    let h = (r as u64)
                        .wrapping_mul(2_654_435_761)
                        .wrapping_add((f as u64).wrapping_mul(40_503).wrapping_add(0x9E37_79B9));
                    (h % NUM_BIN as u64) as u32
                })
                .collect()
        })
        .collect();
    let cols_u16: Vec<Vec<u16>> = cols_u32.iter().map(|c| c.iter().map(|&b| b as u16).collect()).collect();
    let cols_u8: Vec<Vec<u8>> = cols_u32.iter().map(|c| c.iter().map(|&b| b as u8).collect()).collect();

    let rows = scattered_rows();
    let g: Vec<f32> = (0..ROWS).map(|i| (i % 17) as f32 * 0.1).collect();
    let h: Vec<f32> = (0..ROWS).map(|i| 1.0 + (i % 5) as f32 * 0.01).collect();
    let mut scratch = vec![0.0f64; 2 * NUM_BIN];

    println!("# bin-width microbench  rows={ROWS} feats={FEATS} num_bin={NUM_BIN} reps={REPS}");
    println!("# per-feature column: u32={}KB u16={}KB u8={}KB",
        ROWS * 4 / 1024, ROWS * 2 / 1024, ROWS / 1024);

    let mut sink = 0.0;
    macro_rules! bench {
        ($name:expr, $f:expr, $cols:expr) => {{
            // warm
            sink += $f($cols, &rows, &g, &h, &mut scratch);
            let mut ts = Vec::with_capacity(REPS);
            for _ in 0..REPS {
                let t0 = Instant::now();
                sink += $f($cols, &rows, &g, &h, &mut scratch);
                ts.push(t0.elapsed());
            }
            let m = median(ts);
            println!("{:<6} median = {:>10.3?}  ({:.1} Mrows/s)", $name, m,
                (ROWS * FEATS) as f64 / m.as_secs_f64() / 1e6);
            m
        }};
    }

    // Interleave to average out thermal drift.
    let mut u32m = Vec::new();
    let mut u16m = Vec::new();
    let mut u8m = Vec::new();
    for _ in 0..3 {
        u32m.push(bench!("u32", fold_u32, &cols_u32));
        u16m.push(bench!("u16", fold_u16, &cols_u16));
        u8m.push(bench!("u8", fold_u8, &cols_u8));
    }
    let u32b = median(u32m);
    let u16b = median(u16m);
    let u8b = median(u8m);
    println!("\n# BEST-OF-3-INTERLEAVED");
    println!("u32 {:>10.3?}  (baseline)", u32b);
    println!("u16 {:>10.3?}  ({:+.1}%)", u16b, (u16b.as_secs_f64() / u32b.as_secs_f64() - 1.0) * 100.0);
    println!("u8  {:>10.3?}  ({:+.1}%)", u8b, (u8b.as_secs_f64() / u32b.as_secs_f64() - 1.0) * 100.0);
    std::hint::black_box(sink);
}
