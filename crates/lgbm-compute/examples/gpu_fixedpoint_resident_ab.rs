//! Phase 11 Plan 03 (SPEC item 3) — device-time A/B: the **LIVE production** resident u64
//! fixed-point histogram BUILD vs an f32 TWIN, in the realistic resident `build_rp` layout,
//! across the two decisive spike-019 regimes (heavy wide + light).
//!
//! WHAT THIS MEASURES
//! ------------------
//! After Plan 11-01, `resident_raw_build_into` dispatches the u64 kernel
//! (`construct_leaf_hist_resident_lds_kernel_u64`) on the live fix-compact resident chain,
//! so the **production f32 resident launcher is REMOVED from the live path**. This A/B's
//!   * u64 arm = the LIVE production kernel
//!     `lgbm_compute::kernels::histogram::construct_leaf_hist_resident_lds_kernel_u64`
//!     (the exact symbol the production `resident_raw_build_into` drives), and its
//!   * f32 arm = `build_f32_rp`, an **example-local f32 TWIN** defined below — NOT a
//!     production launcher (the production f32 resident path no longer exists on the live
//!     path). The twin is a byte-for-byte copy of the production u64 kernel's body with ONLY
//!     the accumulated cell/atomic type swapped (`Atomic<u64>` + quantize → `Atomic<f32>`).
//!
//! Both arms run the IDENTICAL resident `build_rp` layout — `CubeCount = (num_features, P)`,
//! `CubeDim = 256`, double indirection (`resident_bins[f*num_data + leaf_rows[k]]`),
//! sentinel `slot_off` of length `num_features + 1`, per-feature LDS sub-hist, LDS→global
//! merge — and differ ONLY in atomic/cell type. So the device-time RATIO `f32/u64`
//! isolates the atomic-type cost. This is exactly the spike-018b / spike-019 methodology.
//!
//! MEASUREMENT DISCIPLINE (spike-018/019 CONVENTIONS — copied verbatim)
//! -------------------------------------------------------------------
//!   * Compute-throughput, NOT launch-bound: accumulate `LAUNCHES` launches into ONE reused
//!     buffer per timed sample, then read back ONCE (the single readback forces the device
//!     sync). A per-launch readback would measure launch/PCIe overhead, not atomic cost.
//!   * Interleaved warm-up reps, then `REPS` interleaved reps; report `median[p25..p75]`.
//!   * `SEP-WIN` verdict iff `u64_p75 < f32_p25` (non-overlapping quartile boxes ⇒ the u64
//!     arm is decisively faster); else `overlap`. `ratio = f32_median / u64_median`.
//!   * **Run the whole PROCESS ≥ 2× and check sign-stability** — a single process run on the
//!     8-CU APU is noise-confounded; the verdict is the SIGN that holds across runs.
//!
//! EXPECTATION (SPEC item 3)
//! -------------------------
//!   * HEAVY wide regime (16 feats × 1M rows, big-root large-leaf): integer build is NOT
//!     slower — expect a SEP-WIN ≈ 1.3–1.7× (spike-019: the win lands in heavy atomic load).
//!   * LIGHT regime (16 × 200k): null / overlap is ACCEPTABLE (the contention the integer
//!     atomic relieves is not present at low rows/cube).
//!   * Optional P sweep: spike-019 showed the win composes with row-partition P up to ~16.
//!
//! NOTE — the GPU here is a HSA-spoofed 8-CU APU (Radeon 860M), not a discrete gfx1100.
//! Absolute throughput is APU-confounded; trust only the RELATIVE A/B ratio + SEP sign.
//! This example does NOT assert pass/fail (it is an operator-run device-time proxy,
//! wall-clock unvalidatable here); the verdict is the printed SEP-WIN/overlap + ratio.
//! Captured numbers are recorded in 11-03-SUMMARY.md.
//!
//! Run: cargo run --release --features rocm --example gpu_fixedpoint_resident_ab
//!      (run it ≥ 2× and confirm the per-regime SEP sign is stable.)

#[cfg(feature = "rocm")]
use cubecl::prelude::*;

#[cfg(feature = "rocm")]
const HIST_LDS_MAX: usize = 512; // 2 cells/bin × 256 bins — matches the production kernel

/// f32 TWIN of the production `construct_leaf_hist_resident_lds_kernel_u64` — IDENTICAL
/// resident `build_rp` layout (sentinel `slot_off` of length `num_features + 1`, resident
/// double-indirection gather, row-partition over `CUBE_POS_Y`/`CUBE_COUNT_Y`, per-feature
/// LDS sub-hist, LDS→global merge) with ONLY the accumulated cell/atomic type swapped from
/// `Atomic<u64>` (with in-kernel quantize) back to plain `Atomic<f32>` (raw fetch_add).
///
/// This is the A/B's f32 arm. It is NOT a production launcher — the production f32 resident
/// path was removed from the live path by Plan 11-01 (which switched `resident_raw_build_into`
/// to the u64 kernel). The twin exists ONLY so the device-time ratio isolates atomic-type cost.
#[cfg(feature = "rocm")]
#[cube(launch_unchecked)]
fn build_f32_rp<B: Int>(
    resident_bins: &Array<B>,
    leaf_rows: &Array<u32>,
    ord_g: &Array<f32>,
    ord_h: &Array<f32>,
    slot_off: &Array<u32>, // length num_features + 1 (sentinel = slot_len)
    num_data: usize,
    out: &mut Array<Atomic<f32>>,
) {
    let f = CUBE_POS_X as usize; // ONE cube per feature (× P row-partitions over CUBE_POS_Y)
    let base = slot_off[f] as usize;
    let feat_len = slot_off[f + 1] as usize - base; // = 2*num_bin[f]
    let r = ord_g.len();
    let cd = CUBE_DIM as usize;

    let sub = SharedMemory::<Atomic<f32>>::new(HIST_LDS_MAX);
    let mut c = UNIT_POS as usize;
    while c < feat_len {
        sub[c].store(0.0f32);
        c += cd;
    }
    sync_cube();
    let col = f * num_data;
    let stride = CUBE_COUNT_Y as usize * cd;
    let mut k = CUBE_POS_Y as usize * cd + UNIT_POS as usize;
    while k < r {
        let bin = u32::cast_from(resident_bins[col + leaf_rows[k] as usize]) as usize;
        let ti = bin * 2;
        sub[ti].fetch_add(ord_g[k]);
        sub[ti + 1].fetch_add(ord_h[k]);
        k += stride;
    }
    sync_cube();
    let mut m = UNIT_POS as usize;
    while m < feat_len {
        out[base + m].fetch_add(sub[m].load());
        m += cd;
    }
}

#[cfg(not(feature = "rocm"))]
fn main() {
    eprintln!("requires --features rocm (gfx1100/APU).");
}

#[cfg(feature = "rocm")]
fn main() {
    // The u64 arm drives the LIVE PRODUCTION kernel — the exact symbol the production
    // `resident_raw_build_into` dispatches on the live fix-compact resident chain.
    use lgbm_compute::kernels::histogram::construct_leaf_hist_resident_lds_kernel_u64;
    use lgbm_compute::runtime::rocm_client;
    use std::time::Instant;

    const NUM_BIN: usize = 256; // LightGBM default max_bin; keeps the LDS branch (feat_len ≤ 512)
    const LAUNCHES: usize = 20; // launches accumulated per timed sample (compute-throughput)
    const CUBE_DIM: u32 = 256; // matches the production launcher's CubeDim::new_1d(256)
    const REPS: usize = 9; // interleaved reps for median + p25/p75

    let client = rocm_client();

    // Decisive spike-019 regimes. The HEAVY wide regime is where the integer build's win
    // lands (large root leaf, high rows/cube ⇒ high f32 CAS-retry contention); the LIGHT
    // regime is the null/overlap control. Each is also swept over row-partition P (the win
    // composes with P up to ~16, spike-019).
    let configs: [(usize, usize, &str); 2] = [
        (16, 1_000_000, "HEAVY wide  16feat×1M  (big-root large-leaf: win expected ~1.3-1.7x)"),
        (16, 200_000, "LIGHT       16feat×200k (low rows/cube: null/overlap acceptable)"),
    ];
    let p_sweep: [u32; 4] = [1, 4, 8, 16];

    let pct = |v: &mut Vec<f64>, q: f64| {
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        v[(((v.len() - 1) as f64) * q).round() as usize]
    };

    println!("# Phase 11 Plan 03 — resident u64 (LIVE) vs f32-TWIN build, device-time A/B");
    println!("# f32 arm = build_f32_rp TWIN (NOT production; production f32 resident path removed by Plan 01).");
    println!("# u64 arm = LIVE production construct_leaf_hist_resident_lds_kernel_u64.");
    println!("# compute-throughput: {LAUNCHES} launches accumulated → ONE readback; median[p25..p75] over {REPS} interleaved reps.");
    println!("# SEP-WIN iff u64_p75 < f32_p25; ratio = f32_median / u64_median (>1 ⇒ u64 faster). RUN ≥2× — verdict is the stable SEP sign.");

    for (feats, num_data, label) in configs {
        // Synthetic resident inputs (feature-major resident bins f*num_data + row; double
        // indirection through leaf_rows). Same generator as gpu_int_vs_f32_psweep.
        let mut bins: Vec<u32> = Vec::with_capacity(feats * num_data);
        for f in 0..feats {
            for row in 0..num_data {
                let h = (row as u64).wrapping_mul(2_654_435_761).wrapping_add(f as u64 * 97);
                bins.push((h % NUM_BIN as u64) as u32);
            }
        }
        let leaf_rows: Vec<u32> = (0..num_data as u32)
            .map(|i| (i.wrapping_mul(2_654_435_761)) % num_data as u32)
            .collect();
        let ord_g: Vec<f32> = (0..num_data).map(|i| (i as f32 * 0.013).sin() * 0.5).collect();
        let ord_h: Vec<f32> = (0..num_data).map(|i| 1.0 + (i % 7) as f32 * 0.01).collect();

        let feat_len = (2 * NUM_BIN) as u32;
        // Sentinel slot_off of length num_features + 1 (the production layout).
        let slot_off: Vec<u32> = (0..=feats as u32).map(|f| f * feat_len).collect();
        let slot_len = feats * 2 * NUM_BIN;
        let r = leaf_rows.len();
        let reads_per_launch = (feats * r) as f64;

        let d_bins = client.create_from_slice(u32::as_bytes(&bins));
        let d_rows = client.create_from_slice(u32::as_bytes(&leaf_rows));
        let d_g = client.create_from_slice(f32::as_bytes(&ord_g));
        let d_h = client.create_from_slice(f32::as_bytes(&ord_h));
        let d_slot = client.create_from_slice(u32::as_bytes(&slot_off));

        // f32 TWIN launcher (example-local; NOT production).
        macro_rules! launch_f32 {
            ($p:expr, $out:expr) => {
                build_f32_rp::launch_unchecked::<u32, _>(
                    &client,
                    CubeCount::Static(feats as u32, $p, 1),
                    CubeDim::new_1d(CUBE_DIM),
                    ArrayArg::from_raw_parts(d_bins.clone(), feats * num_data),
                    ArrayArg::from_raw_parts(d_rows.clone(), r),
                    ArrayArg::from_raw_parts(d_g.clone(), r),
                    ArrayArg::from_raw_parts(d_h.clone(), r),
                    ArrayArg::from_raw_parts(d_slot.clone(), feats + 1),
                    num_data,
                    ArrayArg::from_raw_parts($out.clone(), slot_len),
                )
            };
        }
        // u64 arm = the LIVE PRODUCTION kernel (same args/layout, only the cell type differs).
        macro_rules! launch_u64 {
            ($p:expr, $out:expr) => {
                construct_leaf_hist_resident_lds_kernel_u64::launch_unchecked::<u32, _>(
                    &client,
                    CubeCount::Static(feats as u32, $p, 1),
                    CubeDim::new_1d(CUBE_DIM),
                    ArrayArg::from_raw_parts(d_bins.clone(), feats * num_data),
                    ArrayArg::from_raw_parts(d_rows.clone(), r),
                    ArrayArg::from_raw_parts(d_g.clone(), r),
                    ArrayArg::from_raw_parts(d_h.clone(), r),
                    ArrayArg::from_raw_parts(d_slot.clone(), feats + 1),
                    num_data,
                    ArrayArg::from_raw_parts($out.clone(), slot_len),
                )
            };
        }

        // accumulate-LAUNCHES-then-ONE-read (compute-throughput, NOT launch-bound).
        let bench_f32 = |p: u32| -> f64 {
            let out = client.create_from_slice(f32::as_bytes(&vec![0.0f32; slot_len]));
            let t = Instant::now();
            for _ in 0..LAUNCHES {
                unsafe {
                    launch_f32!(p, out);
                }
            }
            let _ = client.read_one_unchecked(out);
            t.elapsed().as_secs_f64() * 1e3
        };
        let bench_u64 = |p: u32| -> f64 {
            let out = client.create_from_slice(u64::as_bytes(&vec![0u64; slot_len]));
            let t = Instant::now();
            for _ in 0..LAUNCHES {
                unsafe {
                    launch_u64!(p, out);
                }
            }
            let _ = client.read_one_unchecked(out);
            t.elapsed().as_secs_f64() * 1e3
        };

        println!("\n## REGIME: {label}");
        // warm-up
        for &p in &p_sweep {
            let _ = bench_f32(p);
            let _ = bench_u64(p);
        }

        let mut regime_any_sep = false;
        for &p in &p_sweep {
            let (mut sf, mut su): (Vec<f64>, Vec<f64>) = (vec![], vec![]);
            for _ in 0..REPS {
                sf.push(bench_f32(p));
                su.push(bench_u64(p));
            }
            let (f25, f50, f75) = (pct(&mut sf, 0.25), pct(&mut sf, 0.5), pct(&mut sf, 0.75));
            let (u25, u50, u75) = (pct(&mut su, 0.25), pct(&mut su, 0.5), pct(&mut su, 0.75));
            let mrs_u = reads_per_launch * LAUNCHES as f64 / (u50 / 1e3) / 1e6;
            let sep = if u75 < f25 {
                regime_any_sep = true;
                "SEP-WIN"
            } else {
                "overlap"
            };
            println!(
                "  P={p:>2}: f32 {f50:.0}ms[{f25:.0}..{f75:.0}]  u64 {u50:.0}ms[{u25:.0}..{u75:.0}] ({mrs_u:.0}Mr/s)  ratio f32/u64={:.2}x {sep}",
                f50 / u50
            );
        }
        let verdict = if regime_any_sep {
            "NOT-SLOWER (SEP-WIN at ≥1 P) ✓"
        } else {
            "overlap (no separation; acceptable for the LIGHT regime, re-check sign across runs)"
        };
        println!("  → regime verdict: {verdict}");
    }

    println!("\n# Diagnosis:");
    println!("#  HEAVY wide SEP-WIN ⇒ the integer build is NOT slower — confirms SPEC item 3 (~1.3-1.7x on large leaves).");
    println!("#  LIGHT overlap     ⇒ null-but-not-regressed (no high-contention CAS-retry to relieve) — acceptable.");
    println!("#  The win composes with row-partition P (spike-019). Re-run the PROCESS ≥2× and confirm the SEP sign is stable.");
    println!("#  (8-CU APU: absolute Mr/s is APU-confounded; only the RELATIVE f32/u64 ratio + SEP sign are load-bearing.)");
}
