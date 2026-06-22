//! Spike 018b — GPU feasibility + speed for fixed-point i64 atomic histogram (research Q2).
//!
//! 018a (CPU probe) opened the parity gate: i64 fixed-point at S=2^30 is within ~1e-6 of
//! the f64 anchor (exact in the cancelling regime), ~1e3–1e4× MORE accurate than the f32
//! path, and DETERMINISTIC. The value is parity-quality + determinism, NOT speed (W5: int
//! atomics gave no speed win). This bench answers two things on REAL gfx1100:
//!   1. FEASIBILITY: does cubecl-hip 0.10 even support `Atomic<i64>` LDS fetch_add? (If it
//!      doesn't compile/run, the GPU determinism win is structurally blocked.)
//!   2. SPEED + DETERMINISM: i64-fixed-point vs f32-2atomic throughput, AND does the i64
//!      kernel produce BIT-IDENTICAL output across two runs (determinism) while f32 does not?
//!
//! Both kernels are single-cube-per-feature (the wide P=1 regime). Scale S = 2^30.
//!
//! Run: cargo run --release --features rocm --example gpu_fixedpoint_i64

#[cfg(feature = "rocm")]
use cubecl::prelude::*;

#[cfg(feature = "rocm")]
const LDS_F32: usize = 512; // 2 f32 cells/bin
#[cfg(feature = "rocm")]
const LDS_I64: usize = 512; // 2 i64 cells/bin (grad,hess) — separate, not packed (wide fixed-point)
#[cfg(feature = "rocm")]
const SCALE: f32 = 1_073_741_824.0; // 2^30

/// f32 baseline: two f32 atomics/row into an LDS sub-hist, then merge to global.
#[cfg(feature = "rocm")]
#[cube(launch_unchecked)]
fn build_f32(
    binned: &Array<u32>,
    grad: &Array<f32>,
    hess: &Array<f32>,
    out: &mut Array<Atomic<f32>>,
    lds_len: u32,
) {
    let sub = SharedMemory::<Atomic<f32>>::new(LDS_F32);
    let cd = CUBE_DIM as usize;
    let lds = lds_len as usize;
    let n = binned.len();
    let mut c = UNIT_POS as usize;
    while c < lds {
        sub[c].store(0.0f32);
        c += cd;
    }
    sync_cube();
    let mut i = UNIT_POS as usize;
    while i < n {
        let ti = binned[i] as usize * 2;
        sub[ti].fetch_add(grad[i]);
        sub[ti + 1].fetch_add(hess[i]);
        i += cd;
    }
    sync_cube();
    let mut m = UNIT_POS as usize;
    while m < lds {
        out[m].fetch_add(sub[m].load());
        m += cd;
    }
}

/// Fixed-point via UNSIGNED 64-bit atomics (cubecl-hip 0.10 has NO `atomicExch(long long*)`,
/// so `Atomic<i64>::store` fails to compile — use `Atomic<u64>` + an OFFSET BIAS so every
/// quantized value is non-negative). q = round(value * 2^30) + BIAS as u64; two u64 atomics
/// /row; merge to a global u64 buffer. Caller dequantizes: (sum - n*BIAS) / 2^30.
/// Integer add is order-independent ⇒ deterministic.
#[cfg(feature = "rocm")]
#[cube(launch_unchecked)]
fn build_u64(
    binned: &Array<u32>,
    grad: &Array<f32>,
    hess: &Array<f32>,
    out: &mut Array<Atomic<u64>>,
    lds_len: u32,
) {
    let sub = SharedMemory::<Atomic<u64>>::new(LDS_I64);
    let cd = CUBE_DIM as usize;
    let lds = lds_len as usize;
    let n = binned.len();
    let mut c = UNIT_POS as usize;
    while c < lds {
        sub[c].store(0u64);
        c += cd;
    }
    sync_cube();
    let mut i = UNIT_POS as usize;
    while i < n {
        let ti = binned[i] as usize * 2;
        // Store the i64 quantized value's BITS as u64; wrapping u64 atomicAdd == signed
        // two's-complement i64 add. Host reinterprets the final u64 bits back to i64.
        let qg = u64::cast_from(i64::cast_from(f32::round(grad[i] * SCALE)));
        let qh = u64::cast_from(i64::cast_from(f32::round(hess[i] * SCALE)));
        sub[ti].fetch_add(qg);
        sub[ti + 1].fetch_add(qh);
        i += cd;
    }
    sync_cube();
    let mut m = UNIT_POS as usize;
    while m < lds {
        out[m].fetch_add(sub[m].load());
        m += cd;
    }
}

#[cfg(not(feature = "rocm"))]
fn main() {
    eprintln!("requires --features rocm (gfx1100/APU).");
}

#[cfg(feature = "rocm")]
fn main() {
    use lgbm_compute::runtime::rocm_client;
    use std::time::Instant;

    const N: usize = 1_000_000;
    const NUM_BIN: usize = 256;
    const LAUNCHES: usize = 20;
    const CUBE_DIM: u32 = 256;
    let client = rocm_client();

    let bins: Vec<u32> = (0..N)
        .map(|i| ((i as u64).wrapping_mul(2_654_435_761) % NUM_BIN as u64) as u32)
        .collect();
    let grad: Vec<f32> = (0..N).map(|i| (i as f32 * 0.013).sin() * 0.5).collect();
    let hess: Vec<f32> = (0..N).map(|i| 1.0 + (i % 7) as f32 * 0.01).collect();
    let lds_len = (2 * NUM_BIN) as u32;
    let out_len = 2 * NUM_BIN;

    let d_bin = client.create_from_slice(u32::as_bytes(&bins));
    let d_g = client.create_from_slice(f32::as_bytes(&grad));
    let d_h = client.create_from_slice(f32::as_bytes(&hess));

    // f64 anchor.
    let mut anchor = vec![0.0f64; out_len];
    for i in 0..N {
        let b = bins[i] as usize;
        anchor[b * 2] += grad[i] as f64;
        anchor[b * 2 + 1] += hess[i] as f64;
    }

    let run_f32 = || -> Vec<f32> {
        let out = client.create_from_slice(f32::as_bytes(&vec![0.0f32; out_len]));
        unsafe {
            build_f32::launch_unchecked(
                &client,
                CubeCount::Static(1, 1, 1),
                CubeDim::new_1d(CUBE_DIM),
                ArrayArg::from_raw_parts(d_bin.clone(), N),
                ArrayArg::from_raw_parts(d_g.clone(), N),
                ArrayArg::from_raw_parts(d_h.clone(), N),
                ArrayArg::from_raw_parts(out.clone(), out_len),
                lds_len,
            );
        }
        f32::from_bytes(&client.read_one_unchecked(out)).to_vec()
    };
    let run_i64 = || -> Vec<i64> {
        let out = client.create_from_slice(u64::as_bytes(&vec![0u64; out_len]));
        unsafe {
            build_u64::launch_unchecked(
                &client,
                CubeCount::Static(1, 1, 1),
                CubeDim::new_1d(CUBE_DIM),
                ArrayArg::from_raw_parts(d_bin.clone(), N),
                ArrayArg::from_raw_parts(d_g.clone(), N),
                ArrayArg::from_raw_parts(d_h.clone(), N),
                ArrayArg::from_raw_parts(out.clone(), out_len),
                lds_len,
            );
        }
        // wrapping u64 sum == two's-complement i64 sum: reinterpret bits back to i64.
        u64::from_bytes(&client.read_one_unchecked(out)).iter().map(|&u| u as i64).collect()
    };

    println!("# Spike 018b — GPU fixed-point i64 vs f32-2atomic  N={N} bins={NUM_BIN} S=2^30");

    // Correctness vs anchor + determinism (two runs each, check bit-identical).
    let f32a = run_f32();
    let f32b = run_f32();
    let i64a = run_i64();
    let i64b = run_i64();
    let f32_det = f32a == f32b;
    let i64_det = i64a == i64b;
    let f32_rel = f32a
        .iter()
        .zip(anchor.iter())
        .map(|(v, a)| (*v as f64 - a).abs() / a.abs().max(1.0))
        .fold(0.0, f64::max);
    let i64_rel = i64a
        .iter()
        .zip(anchor.iter())
        .map(|(v, a)| (*v as f64 / SCALE as f64 - a).abs() / a.abs().max(1.0))
        .fold(0.0, f64::max);
    println!("# f32 : rel_vs_anchor={f32_rel:.2e}  deterministic(2 runs bit-eq)={f32_det}");
    println!("# i64 : rel_vs_anchor={i64_rel:.2e}  deterministic(2 runs bit-eq)={i64_det}");

    // Compute-throughput: accumulate LAUNCHES launches into ONE reused buffer, single
    // readback forces device sync (NOT a per-launch readback — that's launch-bound). f32
    // and i64 differ ONLY in atomic type, so this isolates u64-vs-f32 LDS-atomic cost.
    let bench_f32 = || -> f64 {
        let out = client.create_from_slice(f32::as_bytes(&vec![0.0f32; out_len]));
        let t = Instant::now();
        for _ in 0..LAUNCHES {
            unsafe {
                build_f32::launch_unchecked(
                    &client,
                    CubeCount::Static(1, 1, 1),
                    CubeDim::new_1d(CUBE_DIM),
                    ArrayArg::from_raw_parts(d_bin.clone(), N),
                    ArrayArg::from_raw_parts(d_g.clone(), N),
                    ArrayArg::from_raw_parts(d_h.clone(), N),
                    ArrayArg::from_raw_parts(out.clone(), out_len),
                    lds_len,
                );
            }
        }
        let _ = client.read_one_unchecked(out);
        t.elapsed().as_secs_f64() * 1e3
    };
    let bench_i64 = || -> f64 {
        let out = client.create_from_slice(u64::as_bytes(&vec![0u64; out_len]));
        let t = Instant::now();
        for _ in 0..LAUNCHES {
            unsafe {
                build_u64::launch_unchecked(
                    &client,
                    CubeCount::Static(1, 1, 1),
                    CubeDim::new_1d(CUBE_DIM),
                    ArrayArg::from_raw_parts(d_bin.clone(), N),
                    ArrayArg::from_raw_parts(d_g.clone(), N),
                    ArrayArg::from_raw_parts(d_h.clone(), N),
                    ArrayArg::from_raw_parts(out.clone(), out_len),
                    lds_len,
                );
            }
        }
        let _ = client.read_one_unchecked(out);
        t.elapsed().as_secs_f64() * 1e3
    };

    // warm, then 11 INTERLEAVED reps; median + p25/p75; run the whole process ≥2× to
    // check sign-stability (CONVENTIONS).
    for _ in 0..4 {
        let _ = bench_f32();
        let _ = bench_i64();
    }
    const REPS: usize = 11;
    let (mut sf, mut si): (Vec<f64>, Vec<f64>) = (vec![], vec![]);
    for _ in 0..REPS {
        sf.push(bench_f32());
        si.push(bench_i64());
    }
    let pct = |v: &mut Vec<f64>, q: f64| {
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        v[(((v.len() - 1) as f64) * q).round() as usize]
    };
    let (f25, f50, f75) = (pct(&mut sf, 0.25), pct(&mut sf, 0.5), pct(&mut sf, 0.75));
    let (i25, i50, i75) = (pct(&mut si, 0.25), pct(&mut si, 0.5), pct(&mut si, 0.75));
    println!("# compute-throughput median[p25..p75] over {REPS} interleaved reps ({LAUNCHES} launches each):");
    println!("  f32: {f50:.0}ms[{f25:.0}..{f75:.0}]");
    println!("  i64: {i50:.0}ms[{i25:.0}..{i75:.0}]");
    let sep = if i75 < f25 { " SEP-WIN" } else if f75 < i25 { " SEP-LOSS" } else { " overlap" };
    println!("  i64/f32 = {:.2}x{sep}  (>1 ⇒ u64 atomics faster)", f50 / i50);
    println!("# i64/f32 > 1.0 ⇒ i64 atomics faster; <= 1.0 ⇒ slower/equal (W5 predicted no speed win).");
    println!("# KEY: i64 deterministic=true + within gate ⇒ a DETERMINISTIC, high-accuracy GPU build.");
}
