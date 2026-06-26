//! Spike 041 — `Vector<P,N>` (the cubecl-0.10 "Line") feasibility on the element-wise
//! histogram SUBTRACT — the KILL QUESTION for the whole vectorization lever.
//!
//! ## The API finding (the interesting part — code read from the cubecl 0.10 SOURCE)
//! The user asked for `Line<T>` / `Array<Line<T>>`. **cubecl 0.10 has NO `Line` type.**
//! The vectorized container is `Vector<P: Scalar, N: Size>` (renamed to `Line` only on a
//! later `main`; the burn book uses the new name). The launch ABI (from
//! `cubecl-core-0.10.0/src/runtime_tests/vector.rs`):
//!   - kernel sig: `&mut Array<Vector<F, N>>`; the `N: Size` generic is supplied as a
//!     RUNTIME `usize` positional arg inserted right after `CubeDim` (BEFORE the kernel's
//!     own args) — `launch::<F,R>(client, count, dim, vector_size, parent, child, out, …)`.
//!   - `ArrayArg::from_raw_parts(handle, LENGTH_IN_VECTOR_UNITS)` — length is counted in
//!     vectors, i.e. `n_elements / vector_size`, over the SAME byte buffer.
//!   - `client.io_optimized_vector_sizes(size_of::<F>())` yields the backend's useful
//!     widths (e.g. f32 → 4,2,1) = the natural per-backend sweep set.
//!
//! ## Why SUBTRACT is the kill-question target
//! `subtract_hist_kernel` (`kernels/subtract.rs`) is the cleanest possible vectorization
//! fit in the whole codebase: a pure element-wise `out[i] = parent[i] - child[i]` 1D
//! grid-stride loop over the `[g0,h0,g1,h1,…]` cells — NO atomics, NO permutation, NO
//! reduction. `Vector::sub` is element-wise, so the result is **BIT-EXACT** to the scalar
//! kernel by construction (no reordering of any float op). If `Vector` cannot compile,
//! cannot run on hip, or breaks parity HERE, the lever is dead for the harder kernels.
//!
//! ## Bench discipline (CONVENTIONS, spikes 017–019/040)
//! Accumulate `LAUNCHES` overwrite-launches into ONE reused `out` + a single
//! `read_one_unchecked` to force sync (subtract OVERWRITES, so re-launch is idempotent —
//! autotune-safe class per spike-038); interleave scalar vs vector per rep to cancel
//! drift; report the MEDIAN over `REPS`; judge the SIGN only (the spoofed 8-CU APU
//! confounds absolute time). Re-run across ≥2 process restarts.
//!
//! Run (twice each, for restart stability):
//!   cargo run --release --example spike041_vector_subtract_ab            # cubecl-cpu (f64 + f32)
//!   cargo run --release --features rocm --example spike041_vector_subtract_ab   # + cubecl-hip (f32)

use std::time::Instant;

use cubecl::prelude::*;

const LAUNCHES: usize = 50;
const REPS: usize = 11;

/// Scalar baseline — byte-identical structure to production `subtract_hist_kernel`.
#[cube(launch_unchecked)]
fn subtract_scalar<F: Float>(parent: &Array<F>, child: &Array<F>, out: &mut Array<F>, n: usize) {
    let stride = CUBE_COUNT_X as usize * CUBE_DIM_X as usize;
    let mut i = ABSOLUTE_POS as usize;
    while i < n {
        out[i] = parent[i] - child[i];
        i += stride;
    }
}

/// Vectorized twin — identical grid-stride loop, but each lane subtracts a whole
/// `Vector<F,N>` at once (one SIMD load/sub/store per N cells). `n_vec = n / N`.
#[cube(launch_unchecked)]
fn subtract_vec<F: Float, N: Size>(
    parent: &Array<Vector<F, N>>,
    child: &Array<Vector<F, N>>,
    out: &mut Array<Vector<F, N>>,
    n_vec: usize,
) {
    let stride = CUBE_COUNT_X as usize * CUBE_DIM_X as usize;
    let mut i = ABSOLUTE_POS as usize;
    while i < n_vec {
        out[i] = parent[i] - child[i];
        i += stride;
    }
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

/// Deterministic small-magnitude pattern (no rand dep); distinct parent/child so the
/// subtract is non-trivial and the byte-compare is meaningful.
fn gen_f32(n: usize) -> (Vec<f32>, Vec<f32>) {
    let p = (0..n)
        .map(|i| (((i as u64).wrapping_mul(2654435761) % 1000) as f32 - 500.0) * 0.001)
        .collect();
    let c = (0..n)
        .map(|i| (((i as u64).wrapping_mul(40503) % 1000) as f32 - 500.0) * 0.001)
        .collect();
    (p, c)
}
fn gen_f64(n: usize) -> (Vec<f64>, Vec<f64>) {
    let p = (0..n)
        .map(|i| (((i as u64).wrapping_mul(2654435761) % 1000) as f64 - 500.0) * 0.001)
        .collect();
    let c = (0..n)
        .map(|i| (((i as u64).wrapping_mul(40503) % 1000) as f64 - 500.0) * 0.001)
        .collect();
    (p, c)
}

/// Time `LAUNCHES` scalar launches into one reused `out`, returning (millis, out_bytes).
#[allow(clippy::too_many_arguments)]
fn time_scalar<R: Runtime, F: Float + CubeElement>(
    client: &ComputeClient<R>,
    hp: &cubecl::server::Handle,
    hc: &cubecl::server::Handle,
    ho: &cubecl::server::Handle,
    n: usize,
    count: CubeCount,
    dim: CubeDim,
) -> f64 {
    let t = Instant::now();
    for _ in 0..LAUNCHES {
        unsafe {
            subtract_scalar::launch_unchecked::<F, R>(
                client,
                count.clone(),
                dim.clone(),
                ArrayArg::from_raw_parts(hp.clone(), n),
                ArrayArg::from_raw_parts(hc.clone(), n),
                ArrayArg::from_raw_parts(ho.clone(), n),
                n, // bound: a bare `usize` kernel param is passed raw (cf. production `num_data,`)
            );
        }
    }
    let _ = client.read_one_unchecked(ho.clone());
    t.elapsed().as_secs_f64() * 1e3
}

#[allow(clippy::too_many_arguments)]
fn time_vec<R: Runtime, F: Float + CubeElement>(
    client: &ComputeClient<R>,
    hp: &cubecl::server::Handle,
    hc: &cubecl::server::Handle,
    ho: &cubecl::server::Handle,
    n: usize,
    vs: usize,
    count: CubeCount,
    dim: CubeDim,
) -> f64 {
    let n_vec = n / vs;
    let t = Instant::now();
    for _ in 0..LAUNCHES {
        unsafe {
            subtract_vec::launch_unchecked::<F, R>(
                client,
                count.clone(),
                dim.clone(),
                vs, // the N: Size generic, supplied as a runtime usize
                ArrayArg::from_raw_parts(hp.clone(), n_vec),
                ArrayArg::from_raw_parts(hc.clone(), n_vec),
                ArrayArg::from_raw_parts(ho.clone(), n_vec),
                n_vec,
            );
        }
    }
    let _ = client.read_one_unchecked(ho.clone());
    t.elapsed().as_secs_f64() * 1e3
}

fn run_ab<R: Runtime, F: Float + CubeElement>(
    client: &ComputeClient<R>,
    elem: &str,
    parent: &[F],
    child: &[F],
) {
    let n = parent.len();
    let count = CubeCount::Static(64, 1, 1);
    let dim = CubeDim::new_1d(256);
    let zbytes = vec![0u8; n * std::mem::size_of::<F>()];

    let hp = client.create_from_slice(F::as_bytes(parent));
    let hc = client.create_from_slice(F::as_bytes(child));
    let ho_s = client.create_from_slice(&zbytes);

    // Baseline scalar output (correctness reference).
    let _ = time_scalar::<R, F>(client, &hp, &hc, &ho_s, n, count.clone(), dim.clone());
    let base = client.read_one_unchecked(ho_s.clone()).to_vec();

    let sizes: Vec<usize> = client.io_optimized_vector_sizes(std::mem::size_of::<F>()).collect();
    println!(
        "  [{elem}] n={n} cells  io_optimized_vector_sizes={sizes:?}"
    );

    for &vs in &sizes {
        if n % vs != 0 {
            println!("    vs={vs}: SKIP (n not divisible)");
            continue;
        }
        let ho_v = client.create_from_slice(&zbytes);
        // Correctness first.
        let _ = time_vec::<R, F>(client, &hp, &hc, &ho_v, n, vs, count.clone(), dim.clone());
        let got = client.read_one_unchecked(ho_v.clone()).to_vec();
        let exact = got == base;

        // Interleaved timing.
        let mut st = Vec::with_capacity(REPS);
        let mut vt = Vec::with_capacity(REPS);
        for _ in 0..REPS {
            st.push(time_scalar::<R, F>(client, &hp, &hc, &ho_s, n, count.clone(), dim.clone()));
            vt.push(time_vec::<R, F>(client, &hp, &hc, &ho_v, n, vs, count.clone(), dim.clone()));
        }
        let (ms, mv) = (median(st), median(vt));
        let speedup = ms / mv;
        println!(
            "    vs={vs}: scalar {ms:.3}ms  vec {mv:.3}ms  speedup {speedup:.3}x  bit_exact={exact}"
        );
    }
}

fn main() {
    let sizes = [25_600usize, 256_000]; // 50feat×256bin×2 and 500feat×256bin×2 histograms

    println!("== cubecl-cpu (default runtime) ==");
    {
        let client = lgbm_compute::runtime::cpu_client();
        for &n in &sizes {
            let (p, c) = gen_f64(n);
            run_ab(&client, "f64", &p, &c);
            let (p, c) = gen_f32(n);
            run_ab(&client, "f32", &p, &c);
        }
    }

    #[cfg(feature = "rocm")]
    {
        println!("\n== cubecl-hip (rocm, gfx1100/gfx1152 APU) — f32 (hip has no f64) ==");
        let client = lgbm_compute::runtime::rocm_client();
        for &n in &sizes {
            let (p, c) = gen_f32(n);
            run_ab(&client, "f32", &p, &c);
        }
    }
    #[cfg(not(feature = "rocm"))]
    println!("\n(rocm arm skipped — rebuild with --features rocm for the hip backend)");
}
