//! Spike 043 — vectorize the u64 resident BUILD's `ord_g[k]`/`ord_h[k]` pair-load as
//! `Vector<f32,2>` (one coalesced load/row vs two). The BOUNDED probe.
//!
//! ## The honest ceiling (set BEFORE measuring — spike-030)
//! The wide build is **uncoalesced-bin-gather-latency bound: 86–95%** of build device-time
//! is the permuted gather `resident_bins[col + leaf_rows[k]]`. grad/hess global reads are
//! only **8–14%**, and the u64 atomic is ~0%. So vectorizing the grad/hess load — even if
//! it perfectly halves that 8–14% — has a HARD ceiling of ≈1.08–1.15× on the build, and
//! only if grad/hess load latency isn't already hidden behind the gather stall (it is).
//!
//! ## Why the dominant cost (the gather) CANNOT be vectorized — structural, not measured
//! `resident_bins[col + leaf_rows[k]]` is a PERMUTATION: consecutive `k` map to
//! NON-consecutive addresses (`leaf_rows` is a scattered/monotone subset, not `k`). A
//! vectorized load (`Vector`) only helps CONTIGUOUS addresses — there is no `Vector` read
//! that gathers `bins[p0],bins[p1],…` for arbitrary `p`. So the 86–95% is immune to this
//! lever by construction; only the coalesced `ord_g`/`ord_h` (read at sequential `k`) can
//! vectorize. This spike measures that bounded slice and documents the immunity.
//!
//! Most-favorable test: leaf_rows = identity (root build, COALESCED gather) ⇒ the gather is
//! cheapest ⇒ grad/hess is the LARGEST possible fraction ⇒ the vectorization's BEST case.
//! If it's null even here, it's null in production (deeper leaves → costlier gather → even
//! smaller grad/hess fraction).
//!
//! Bit-exact by construction: identical quantization `round(v·2^30)` → i64-bits → u64
//! atomic add; integer adds are order-independent, so scalar and vector arms produce the
//! byte-identical u64 histogram.
//!
//! Run (twice each):
//!   cargo run --release --example spike043_vector_build_gradhess_ab           # cubecl-cpu
//!   cargo run --release --features rocm --example spike043_vector_build_gradhess_ab  # cubecl-hip

use std::time::Instant;

use cubecl::prelude::*;

const LAUNCHES: usize = 20;
const REPS: usize = 11;
const HIST_LDS_MAX: usize = 512;
const SCALE_F32: f32 = 1_073_741_824.0; // 2^30

/// Scalar grad/hess: mirror of `construct_leaf_hist_resident_lds_kernel_u64` (P=1).
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn build_scalar(
    resident_bins: &Array<u32>,
    leaf_rows: &Array<u32>,
    ord_g: &Array<f32>,
    ord_h: &Array<f32>,
    num_data: usize,
    feat_len: usize,
    out: &mut Array<Atomic<u64>>,
) {
    let f = CUBE_POS_X as usize;
    let base = f * feat_len;
    let r = ord_g.len();
    let cd = CUBE_DIM as usize;
    let sub = SharedMemory::<Atomic<u64>>::new(HIST_LDS_MAX);
    let mut c = UNIT_POS as usize;
    while c < feat_len {
        sub[c].store(0u64);
        c += cd;
    }
    sync_cube();
    let col = f * num_data;
    let mut k = UNIT_POS as usize;
    while k < r {
        let bin = resident_bins[col + leaf_rows[k] as usize] as usize;
        let ti = bin * 2;
        let qg = u64::cast_from(i64::cast_from(f32::round(ord_g[k] * SCALE_F32)));
        let qh = u64::cast_from(i64::cast_from(f32::round(ord_h[k] * SCALE_F32)));
        sub[ti].fetch_add(qg);
        sub[ti + 1].fetch_add(qh);
        k += cd;
    }
    sync_cube();
    let mut m = UNIT_POS as usize;
    while m < feat_len {
        out[base + m].fetch_add(sub[m].load());
        m += cd;
    }
}

/// Vectorized grad/hess: read `[g,h]` as one `Vector<f32,N>` (N=2). Everything else
/// (gather, quantize, atomics, merge) is byte-identical ⇒ bit-exact.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn build_vec<N: Size>(
    resident_bins: &Array<u32>,
    leaf_rows: &Array<u32>,
    ord_gh: &Array<Vector<f32, N>>,
    num_data: usize,
    feat_len: usize,
    r: usize,
    out: &mut Array<Atomic<u64>>,
) {
    let f = CUBE_POS_X as usize;
    let base = f * feat_len;
    let cd = CUBE_DIM as usize;
    let sub = SharedMemory::<Atomic<u64>>::new(HIST_LDS_MAX);
    let mut c = UNIT_POS as usize;
    while c < feat_len {
        sub[c].store(0u64);
        c += cd;
    }
    sync_cube();
    let col = f * num_data;
    let mut k = UNIT_POS as usize;
    while k < r {
        let bin = resident_bins[col + leaf_rows[k] as usize] as usize;
        let ti = bin * 2;
        let gh = ord_gh[k];
        let qg = u64::cast_from(i64::cast_from(f32::round(gh[0] * SCALE_F32)));
        let qh = u64::cast_from(i64::cast_from(f32::round(gh[1] * SCALE_F32)));
        sub[ti].fetch_add(qg);
        sub[ti + 1].fetch_add(qh);
        k += cd;
    }
    sync_cube();
    let mut m = UNIT_POS as usize;
    while m < feat_len {
        out[base + m].fetch_add(sub[m].load());
        m += cd;
    }
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

fn run_ab<R: Runtime>(client: &ComputeClient<R>, n_feat: usize, num_bins: usize, rows: usize) {
    let feat_len = num_bins * 2;
    // resident bins: [n_feat * rows], values in 0..num_bins (identity leaf_rows ⇒ coalesced).
    let bins: Vec<u32> = (0..n_feat * rows)
        .map(|i| ((i as u64).wrapping_mul(2654435761) % num_bins as u64) as u32)
        .collect();
    let leaf_rows: Vec<u32> = (0..rows as u32).collect();
    // interleaved grad/hess for the vector arm; split for the scalar arm.
    let gh: Vec<f32> = (0..rows * 2)
        .map(|i| (((i as u64).wrapping_mul(40503) % 1000) as f32 - 500.0) * 0.001)
        .collect();
    let g: Vec<f32> = (0..rows).map(|k| gh[k * 2]).collect();
    let h: Vec<f32> = (0..rows).map(|k| gh[k * 2 + 1]).collect();

    let h_bins = client.create_from_slice(u32::as_bytes(&bins));
    let h_lr = client.create_from_slice(u32::as_bytes(&leaf_rows));
    let h_g = client.create_from_slice(f32::as_bytes(&g));
    let h_h = client.create_from_slice(f32::as_bytes(&h));
    let h_gh = client.create_from_slice(f32::as_bytes(&gh));
    let out_len = n_feat * feat_len;
    let zbytes = vec![0u8; out_len * std::mem::size_of::<u64>()];

    let count = CubeCount::Static(n_feat as u32, 1, 1);
    let dim = CubeDim::new_1d(256);

    let run_scalar = |reps: usize| -> (f64, Vec<u8>) {
        let mut ts = Vec::with_capacity(reps);
        let mut last = Vec::new();
        for _ in 0..reps {
            let ho = client.create_from_slice(&zbytes);
            let t = Instant::now();
            for _ in 0..LAUNCHES {
                unsafe {
                    build_scalar::launch_unchecked::<R>(
                        client,
                        count.clone(),
                        dim.clone(),
                        ArrayArg::from_raw_parts(h_bins.clone(), bins.len()),
                        ArrayArg::from_raw_parts(h_lr.clone(), rows),
                        ArrayArg::from_raw_parts(h_g.clone(), rows),
                        ArrayArg::from_raw_parts(h_h.clone(), rows),
                        num_data_arg(rows),
                        feat_len,
                        ArrayArg::from_raw_parts(ho.clone(), out_len),
                    );
                }
            }
            last = client.read_one_unchecked(ho.clone()).to_vec();
            ts.push(t.elapsed().as_secs_f64() * 1e3);
        }
        (median(ts), last)
    };
    let run_vec = |reps: usize| -> (f64, Vec<u8>) {
        let mut ts = Vec::with_capacity(reps);
        let mut last = Vec::new();
        for _ in 0..reps {
            let ho = client.create_from_slice(&zbytes);
            let t = Instant::now();
            for _ in 0..LAUNCHES {
                unsafe {
                    build_vec::launch_unchecked::<R>(
                        client,
                        count.clone(),
                        dim.clone(),
                        2usize, // N = 2 ([g,h])
                        ArrayArg::from_raw_parts(h_bins.clone(), bins.len()),
                        ArrayArg::from_raw_parts(h_lr.clone(), rows),
                        ArrayArg::from_raw_parts(h_gh.clone(), rows), // length in vector units
                        num_data_arg(rows),
                        feat_len,
                        rows,
                        ArrayArg::from_raw_parts(ho.clone(), out_len),
                    );
                }
            }
            last = client.read_one_unchecked(ho.clone()).to_vec();
            ts.push(t.elapsed().as_secs_f64() * 1e3);
        }
        (median(ts), last)
    };

    let (_, base) = run_scalar(1);
    let (_, got) = run_vec(1);
    let exact = base == got;

    let mut st = Vec::with_capacity(REPS);
    let mut vt = Vec::with_capacity(REPS);
    for _ in 0..REPS {
        st.push(run_scalar(1).0);
        vt.push(run_vec(1).0);
    }
    let (ms, mv) = (median(st), median(vt));
    println!(
        "  feat={n_feat} bins={num_bins} rows={rows}: scalar {ms:.3}ms  vec2(g/h) {mv:.3}ms  speedup {:.3}x  bit_exact={exact}",
        ms / mv
    );
}

// A bare `usize` kernel arg is passed raw; this helper just documents intent.
fn num_data_arg(n: usize) -> usize {
    n
}

fn main() {
    // NOTE: the u64-atomic resident build is a HIP-ONLY kernel by nature — cubecl-cpu has
    // no `Atomic<u64>` (it panics: "not yet implemented: atomic<u64>"), which is exactly why
    // the production CPU anchor uses the f64 non-atomic fold instead (spike-018b). So this
    // build A/B runs ONLY on rocm; the cpu arm is N/A.
    let shapes = [(50usize, 256usize, 200_000usize), (500, 256, 200_000)];

    #[cfg(feature = "rocm")]
    {
        println!("== cubecl-hip (rocm, gfx1100/gfx1152 APU) — u64 resident build ==");
        let client = lgbm_compute::runtime::rocm_client();
        for &(nf, nb, rows) in &shapes {
            run_ab(&client, nf, nb, rows);
        }
    }
    #[cfg(not(feature = "rocm"))]
    println!("(u64-atomic build is hip-only — rebuild with --features rocm; cubecl-cpu has no Atomic<u64>)");
}
