//! Spike 042 — vectorize the feature-per-lane SCAN's sequential `[grad,hess]` pair read
//! as `Vector<F,2>` (one load per bin-pair vs two scalar loads).
//!
//! ## Premise (from 041 + the scan attributions)
//! The shipped split scan (spike-021) is feature-per-lane: `CubeDim=W` lanes, each lane
//! sequentially scans ONE feature's `num_bins` `[g,h]` pairs, running a prefix sum + gain
//! argmax. The histogram is laid out `[g0,h0,g1,h1,…]` per feature, so each bin is a
//! natural `Vector<F,2>` — reading the pair in ONE vectorized load halves the per-lane
//! load count. Spike-041 proved `Vector<P,N>` is bit-exact + faster at width>1; the open
//! question is whether the scan is READ-bound enough for that to move (spike-034 says the
//! genuine scan is largely launch/readback-SYNC bound, only 3–7% of train).
//!
//! ## Method (spike-022b precedent)
//! A PROXY feature-per-lane scan kernel: single forward prefix-sum + representative gain
//! (`sg²/(sh+ε)`) argmax — faithful to the real scan's READ + accumulate STRUCTURE without
//! the full reverse+forward/default-bin machinery (which is scalar control flow the load
//! change can't touch anyway). Scalar reads `hist[base+2b]`,`hist[base+2b+1]`; the vector
//! arm reads `hist_v[base_v+b]` as `Vector<F,2>` and extracts `[0]`/`[1]`. The prefix-sum
//! arithmetic is IDENTICAL ⇒ bit-exact by construction (only the load is vectorized).
//!
//! Bench at the WIDE shape (041's lesson: the win needs size). Discipline per CONVENTIONS:
//! overwrite-launch ×LAUNCHES into one reused `out` + single read; interleave; median;
//! judge SIGN; 2 restarts.
//!
//! Run (twice each):
//!   cargo run --release --example spike042_vector_scan_pair_ab
//!   cargo run --release --features rocm --example spike042_vector_scan_pair_ab

use std::time::Instant;

use cubecl::prelude::*;

const LAUNCHES: usize = 50;
const REPS: usize = 11;
const W: u32 = 64; // feature-per-lane cube width (spike-021 shipped default)

/// Scalar feature-per-lane proxy scan: lane `f` reads its `[g,h]` pairs one element at a time.
#[cube(launch_unchecked)]
fn scan_scalar<F: Float>(hist: &Array<F>, num_bins: usize, n_feat: usize, out: &mut Array<F>) {
    let f = ABSOLUTE_POS as usize;
    if f < n_feat {
        let base = f * num_bins * 2;
        let mut sg = F::new(0.0);
        let mut sh = F::new(0.0);
        let mut best = F::new(-1.0);
        for b in 0..num_bins {
            sg += hist[base + b * 2];
            sh += hist[base + b * 2 + 1];
            let gain = sg * sg / (sh + F::new(1e-3));
            if gain > best {
                best = gain;
            }
        }
        out[f] = best;
    }
}

/// Vectorized twin: lane `f` reads each `[g,h]` bin as ONE `Vector<F,N>` (N=2) load.
/// Identical prefix-sum arithmetic ⇒ bit-exact; only the load is vectorized.
#[cube(launch_unchecked)]
fn scan_vec<F: Float, N: Size>(
    hist: &Array<Vector<F, N>>,
    num_bins: usize,
    n_feat: usize,
    out: &mut Array<F>,
) {
    let f = ABSOLUTE_POS as usize;
    if f < n_feat {
        let base_v = f * num_bins; // in vector (pair) units
        let mut sg = F::new(0.0);
        let mut sh = F::new(0.0);
        let mut best = F::new(-1.0);
        for b in 0..num_bins {
            let pair = hist[base_v + b];
            sg += pair[0];
            sh += pair[1];
            let gain = sg * sg / (sh + F::new(1e-3));
            if gain > best {
                best = gain;
            }
        }
        out[f] = best;
    }
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

fn gen_hist_f32(n_feat: usize, num_bins: usize) -> Vec<f32> {
    (0..n_feat * num_bins * 2)
        .map(|i| (((i as u64).wrapping_mul(2654435761) % 1000) as f32 - 500.0) * 0.001)
        .collect()
}
fn gen_hist_f64(n_feat: usize, num_bins: usize) -> Vec<f64> {
    (0..n_feat * num_bins * 2)
        .map(|i| (((i as u64).wrapping_mul(2654435761) % 1000) as f64 - 500.0) * 0.001)
        .collect()
}

fn run_ab<R: Runtime, F: Float + CubeElement>(
    client: &ComputeClient<R>,
    elem: &str,
    hist: &[F],
    n_feat: usize,
    num_bins: usize,
) {
    let count = CubeCount::Static(n_feat.div_ceil(W as usize) as u32, 1, 1);
    let dim = CubeDim::new_1d(W);
    let hh = client.create_from_slice(F::as_bytes(hist));
    let obytes = vec![0u8; n_feat * std::mem::size_of::<F>()];
    let ho_s = client.create_from_slice(&obytes);
    let ho_v = client.create_from_slice(&obytes);

    let run_scalar = |reps: usize| -> f64 {
        let mut ts = Vec::with_capacity(reps);
        for _ in 0..reps {
            let t = Instant::now();
            for _ in 0..LAUNCHES {
                unsafe {
                    scan_scalar::launch_unchecked::<F, R>(
                        client,
                        count.clone(),
                        dim.clone(),
                        ArrayArg::from_raw_parts(hh.clone(), hist.len()),
                        num_bins,
                        n_feat,
                        ArrayArg::from_raw_parts(ho_s.clone(), n_feat),
                    );
                }
            }
            let _ = client.read_one_unchecked(ho_s.clone());
            ts.push(t.elapsed().as_secs_f64() * 1e3);
        }
        median(ts)
    };
    let run_vec = |reps: usize| -> f64 {
        let nv = hist.len() / 2;
        let mut ts = Vec::with_capacity(reps);
        for _ in 0..reps {
            let t = Instant::now();
            for _ in 0..LAUNCHES {
                unsafe {
                    scan_vec::launch_unchecked::<F, R>(
                        client,
                        count.clone(),
                        dim.clone(),
                        2usize, // N = 2 (the [g,h] pair)
                        ArrayArg::from_raw_parts(hh.clone(), nv),
                        num_bins,
                        n_feat,
                        ArrayArg::from_raw_parts(ho_v.clone(), n_feat),
                    );
                }
            }
            let _ = client.read_one_unchecked(ho_v.clone());
            ts.push(t.elapsed().as_secs_f64() * 1e3);
        }
        median(ts)
    };

    // Correctness: one launch each, compare best-gain outputs bit-for-bit.
    let _ = run_scalar(1);
    let _ = run_vec(1);
    let base = client.read_one_unchecked(ho_s.clone()).to_vec();
    let got = client.read_one_unchecked(ho_v.clone()).to_vec();
    let exact = base == got;

    // Interleaved timing.
    let mut st = Vec::with_capacity(REPS);
    let mut vt = Vec::with_capacity(REPS);
    for _ in 0..REPS {
        st.push(run_scalar(1));
        vt.push(run_vec(1));
    }
    let (ms, mv) = (median(st), median(vt));
    println!(
        "  [{elem}] feat={n_feat} bins={num_bins}: scalar {ms:.3}ms  vec2 {mv:.3}ms  speedup {:.3}x  bit_exact={exact}",
        ms / mv
    );
}

fn main() {
    // Wide + narrow feature counts; 256 bins (production all-256-bin shape).
    let shapes = [(50usize, 256usize), (500, 256)];

    println!("== cubecl-cpu (default runtime) ==");
    {
        let client = lgbm_compute::runtime::cpu_client();
        for &(nf, nb) in &shapes {
            run_ab(&client, "f64", &gen_hist_f64(nf, nb), nf, nb);
            run_ab(&client, "f32", &gen_hist_f32(nf, nb), nf, nb);
        }
    }

    #[cfg(feature = "rocm")]
    {
        println!("\n== cubecl-hip (rocm, gfx1100/gfx1152 APU) — f32 ==");
        let client = lgbm_compute::runtime::rocm_client();
        for &(nf, nb) in &shapes {
            run_ab(&client, "f32", &gen_hist_f32(nf, nb), nf, nb);
        }
    }
    #[cfg(not(feature = "rocm"))]
    println!("\n(rocm arm skipped — rebuild with --features rocm)");
}
