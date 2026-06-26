//! Spike 044 — vectorize the `fix_compact` DEQUANT sub-step as `Vector<_,N>`.
//!
//! ## The candidate (the one untested streaming map in the histogram pipeline)
//! `fix_compact_kernel`'s first sub-step (histogram.rs:2347-2351) dequantizes the u64
//! fixed-point RAW histogram into f64: `hist[i] = f64::cast_from(i64::cast_from(h_raw[i])) /
//! 2^30`, element-wise over the `[g,h]` cells. Minimal per-element compute (one int
//! reinterpret + one power-of-2 divide) ⇒ MEMORY-BOUND, the same shape as the subtract
//! that WON in spike-041. The fix (dependent reduction) and compact (shift) sub-steps are
//! NOT vectorizable (042 pattern) — this spike isolates ONLY the dequant.
//!
//! ## Honest ceiling (set before measuring)
//! In production the dequant is fused inside `build_fix_scan_resident` and is a MINORITY
//! fraction of a single-threaded-per-feature, fix-reduction-dominated cube. So an isolated
//! dequant win is an UPPER bound — e2e is bounded by the dequant's fraction (likely a
//! 042-style "non-bottleneck inside a bigger kernel" once fused). This spike measures the
//! isolated dequant to decide whether it's worth wiring into the fused kernel.
//!
//! Bit-exact by construction: per-cell `i64-reinterpret / 2^30` (2^30 exact in f64), no
//! reorder, no reduction ⇒ the vector arm is byte-identical to scalar.
//!
//! Runs f64 on BOTH backends — the gfx1100/gfx1152 APU runs f64 despite `has_f64==false`
//! (histogram.rs:2310), exactly as the live `fix_compact_kernel` does.
//!
//! Run (twice each):
//!   cargo run --release --example spike044_vector_dequant_ab
//!   cargo run --release --features rocm --example spike044_vector_dequant_ab

use std::time::Instant;

use cubecl::prelude::*;

const LAUNCHES: usize = 50;
const REPS: usize = 11;
const SCALE_F64: f64 = 1_073_741_824.0; // 2^30 (matches fix_compact_kernel SCALE_F64)

/// Scalar dequant — byte-identical math to fix_compact_kernel's dequant loop.
#[cube(launch_unchecked)]
fn dequant_scalar(h_raw: &Array<u64>, hist: &mut Array<f64>, n: usize) {
    let stride = CUBE_COUNT_X as usize * CUBE_DIM_X as usize;
    let mut i = ABSOLUTE_POS as usize;
    while i < n {
        hist[i] = f64::cast_from(i64::cast_from(h_raw[i])) / SCALE_F64;
        i += stride;
    }
}

/// Vectorized twin: read/cast/divide/write a whole `Vector<_,N>` per lane.
/// Same per-lane ops ⇒ bit-exact; the int-reinterpret + power-of-2 divide vectorize.
#[cube(launch_unchecked)]
fn dequant_vec<N: Size>(
    h_raw: &Array<Vector<u64, N>>,
    hist: &mut Array<Vector<f64, N>>,
    n_vec: usize,
) {
    let stride = CUBE_COUNT_X as usize * CUBE_DIM_X as usize;
    let mut i = ABSOLUTE_POS as usize;
    while i < n_vec {
        let raw = h_raw[i];
        let asi = Vector::<i64, N>::cast_from(raw);
        let asf = Vector::<f64, N>::cast_from(asi);
        let scale = Vector::<f64, N>::new(SCALE_F64); // broadcast to N lanes
        hist[i] = asf / scale;
        i += stride;
    }
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

/// Representative quantized RAW histogram: each cell = `round(value * 2^30)` as i64-bits.
fn gen_raw(n: usize) -> Vec<u64> {
    (0..n)
        .map(|i| {
            let v = (((i as u64).wrapping_mul(2654435761) % 1000) as f64 - 500.0) * 0.001;
            let q = (v * SCALE_F64).round() as i64;
            q as u64 // store i64 bits as u64 (build_u64_rp idiom)
        })
        .collect()
}

fn run_ab<R: Runtime>(client: &ComputeClient<R>, n: usize) {
    let count = CubeCount::Static(64, 1, 1);
    let dim = CubeDim::new_1d(256);
    let raw = gen_raw(n);
    let h_raw = client.create_from_slice(u64::as_bytes(&raw));
    let zbytes = vec![0u8; n * std::mem::size_of::<f64>()];
    let ho_s = client.create_from_slice(&zbytes);

    let run_scalar = |reps: usize| -> f64 {
        let mut ts = Vec::with_capacity(reps);
        for _ in 0..reps {
            let t = Instant::now();
            for _ in 0..LAUNCHES {
                unsafe {
                    dequant_scalar::launch_unchecked::<R>(
                        client,
                        count.clone(),
                        dim.clone(),
                        ArrayArg::from_raw_parts(h_raw.clone(), n),
                        ArrayArg::from_raw_parts(ho_s.clone(), n),
                        n,
                    );
                }
            }
            let _ = client.read_one_unchecked(ho_s.clone());
            ts.push(t.elapsed().as_secs_f64() * 1e3);
        }
        median(ts)
    };

    let base = {
        let _ = run_scalar(1);
        client.read_one_unchecked(ho_s.clone()).to_vec()
    };

    let sizes: Vec<usize> = client.io_optimized_vector_sizes(std::mem::size_of::<f64>()).collect();
    println!("  n={n} cells  io_optimized_vector_sizes(f64)={sizes:?}");

    for &vs in &sizes {
        if n % vs != 0 {
            println!("    vs={vs}: SKIP (n not divisible)");
            continue;
        }
        let nv = n / vs;
        let ho_v = client.create_from_slice(&zbytes);
        let run_vec = |reps: usize| -> f64 {
            let mut ts = Vec::with_capacity(reps);
            for _ in 0..reps {
                let t = Instant::now();
                for _ in 0..LAUNCHES {
                    unsafe {
                        dequant_vec::launch_unchecked::<R>(
                            client,
                            count.clone(),
                            dim.clone(),
                            vs,
                            ArrayArg::from_raw_parts(h_raw.clone(), nv),
                            ArrayArg::from_raw_parts(ho_v.clone(), nv),
                            nv,
                        );
                    }
                }
                let _ = client.read_one_unchecked(ho_v.clone());
                ts.push(t.elapsed().as_secs_f64() * 1e3);
            }
            median(ts)
        };
        let _ = run_vec(1);
        let got = client.read_one_unchecked(ho_v.clone()).to_vec();
        let exact = got == base;

        let mut st = Vec::with_capacity(REPS);
        let mut vt = Vec::with_capacity(REPS);
        for _ in 0..REPS {
            st.push(run_scalar(1));
            vt.push(run_vec(1));
        }
        let (ms, mv) = (median(st), median(vt));
        println!(
            "    vs={vs}: scalar {ms:.3}ms  vec {mv:.3}ms  speedup {:.3}x  bit_exact={exact}",
            ms / mv
        );
    }
}

fn main() {
    let sizes = [25_600usize, 256_000];

    println!("== cubecl-cpu (default runtime) ==");
    {
        let client = lgbm_compute::runtime::cpu_client();
        for &n in &sizes {
            run_ab(&client, n);
        }
    }

    #[cfg(feature = "rocm")]
    {
        println!("\n== cubecl-hip (rocm, gfx1100/gfx1152 APU) — f64 (runs despite has_f64==false) ==");
        let client = lgbm_compute::runtime::rocm_client();
        for &n in &sizes {
            run_ab(&client, n);
        }
    }
    #[cfg(not(feature = "rocm"))]
    println!("\n(rocm arm skipped — rebuild with --features rocm)");
}
