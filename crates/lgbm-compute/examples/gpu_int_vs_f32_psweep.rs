//! Spike 019 — does the fixed-point u64-atomic win (spike-018, ~1.9× at P=1) SHRINK as
//! row-partition P rises? Confirms the contention-regime hypothesis: f32 `atomicAdd` on
//! RDNA is a CAS retry loop whose cost explodes under HIGH per-cube contention (P=1, wide
//! shape); integer `ds_add_u64` is native single-instruction. Row-partitioning (spike-007)
//! LOWERS per-cube contention (fewer rows/cube), so the integer advantage should fade
//! toward ~1.0× as P grows — reconciling spike-018 (P=1, ~1.9×) with W5 (P=16, null).
//!
//! Both kernels are the 007 `build_rp` layout (CubeCount=(feats,P), cube (f,p) folds its
//! row partition into an LDS sub-hist, merges to feature f's global slot) — the f32 and
//! u64 twins differ ONLY in atomic type, so the per-P i64/f32 ratio isolates the
//! atomic-type cost at each contention level.
//!
//! Interpretation:
//!   ratio falls 1.9× → ~1.0× as P↑  ⇒ hypothesis CONFIRMED; integer atomics and
//!     row-partition are SUBSTITUTES (both relieve the same CAS-retry contention).
//!   ratio stays ~1.9× across P       ⇒ the win is contention-independent (atomic-type
//!     intrinsic); integer atomics COMPOSE with / dominate row-partition.
//!
//! Run: cargo run --release --features rocm --example gpu_int_vs_f32_psweep

#[cfg(feature = "rocm")]
use cubecl::prelude::*;

#[cfg(feature = "rocm")]
const HIST_LDS_F32: usize = 512; // 2 f32/bin
#[cfg(feature = "rocm")]
const HIST_LDS_U64: usize = 512; // 2 u64/bin
#[cfg(feature = "rocm")]
const SCALE: f32 = 1_073_741_824.0; // 2^30

/// f32 row-partitioned build (== spike-007 build_rp).
#[cfg(feature = "rocm")]
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn build_f32_rp(
    resident_bins: &Array<u32>,
    leaf_rows: &Array<u32>,
    ord_g: &Array<f32>,
    ord_h: &Array<f32>,
    slot_off: &Array<u32>,
    num_data: usize,
    feat_len: u32,
    out: &mut Array<Atomic<f32>>,
) {
    let f = CUBE_POS_X as usize;
    let p = CUBE_POS_Y as usize;
    let np = CUBE_COUNT_Y as usize;
    let cd = CUBE_DIM as usize;
    let fl = feat_len as usize;
    let r = ord_g.len();
    let base = slot_off[f] as usize;
    let col = f * num_data;
    let sub = SharedMemory::<Atomic<f32>>::new(HIST_LDS_F32);
    let mut c = UNIT_POS as usize;
    while c < fl {
        sub[c].store(0.0f32);
        c += cd;
    }
    sync_cube();
    let stride = np * cd;
    let mut k = p * cd + UNIT_POS as usize;
    while k < r {
        let bin = resident_bins[col + leaf_rows[k] as usize] as usize;
        let ti = bin * 2;
        sub[ti].fetch_add(ord_g[k]);
        sub[ti + 1].fetch_add(ord_h[k]);
        k += stride;
    }
    sync_cube();
    let mut m = UNIT_POS as usize;
    while m < fl {
        out[base + m].fetch_add(sub[m].load());
        m += cd;
    }
}

/// u64 fixed-point row-partitioned twin (two's-complement; same layout as build_f32_rp).
#[cfg(feature = "rocm")]
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn build_u64_rp(
    resident_bins: &Array<u32>,
    leaf_rows: &Array<u32>,
    ord_g: &Array<f32>,
    ord_h: &Array<f32>,
    slot_off: &Array<u32>,
    num_data: usize,
    feat_len: u32,
    out: &mut Array<Atomic<u64>>,
) {
    let f = CUBE_POS_X as usize;
    let p = CUBE_POS_Y as usize;
    let np = CUBE_COUNT_Y as usize;
    let cd = CUBE_DIM as usize;
    let fl = feat_len as usize;
    let r = ord_g.len();
    let base = slot_off[f] as usize;
    let col = f * num_data;
    let sub = SharedMemory::<Atomic<u64>>::new(HIST_LDS_U64);
    let mut c = UNIT_POS as usize;
    while c < fl {
        sub[c].store(0u64);
        c += cd;
    }
    sync_cube();
    let stride = np * cd;
    let mut k = p * cd + UNIT_POS as usize;
    while k < r {
        let bin = resident_bins[col + leaf_rows[k] as usize] as usize;
        let ti = bin * 2;
        let qg = u64::cast_from(i64::cast_from(f32::round(ord_g[k] * SCALE)));
        let qh = u64::cast_from(i64::cast_from(f32::round(ord_h[k] * SCALE)));
        sub[ti].fetch_add(qg);
        sub[ti + 1].fetch_add(qh);
        k += stride;
    }
    sync_cube();
    let mut m = UNIT_POS as usize;
    while m < fl {
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
    use lgbm_compute::runtime::rocm_client;
    use std::time::Instant;

    const NUM_BIN: usize = 256;
    const LAUNCHES: usize = 20;
    const CUBE_DIM: u32 = 256;
    let p_sweep: [u32; 5] = [1, 2, 4, 8, 16];
    let client = rocm_client();

    // Triangulate the 018b ~1.9× vs 019 ~1.0×: is the win about UNDER-OCCUPANCY (#cubes)
    // or ROWS-PER-CUBE (contention)? Configs:
    //   (1,  1M)  = reproduce 018b exactly (1 cube, under-occupied, 1M rows/cube)
    //   (16, 1M)  = occupied (16 cubes) BUT high rows/cube
    //   (16, 200k)= occupied + moderate rows/cube (the original 019)
    //   (64, 200k)= well-occupied (≈64 cubes ≈ fill 8-CU APU)
    let configs: [(usize, usize, &str); 4] = [
        (1, 1_000_000, "1cube×1M  (=018b: under-occupied)"),
        (16, 1_000_000, "16cube×1M (occupied, high rows/cube)"),
        (16, 200_000, "16cube×200k (occupied, mod rows/cube)"),
        (64, 200_000, "64cube×200k (well-occupied)"),
    ];

    for (feats_c, num_data_c, label) in configs {
    let feats: usize = feats_c;
    let num_data: usize = num_data_c;

    let mut bins: Vec<u32> = Vec::with_capacity(feats * num_data);
    for f in 0..feats {
        for r in 0..num_data {
            let h = (r as u64).wrapping_mul(2_654_435_761).wrapping_add(f as u64 * 97);
            bins.push((h % NUM_BIN as u64) as u32);
        }
    }
    let leaf_rows: Vec<u32> =
        (0..num_data as u32).map(|i| (i.wrapping_mul(2_654_435_761)) % num_data as u32).collect();
    let ord_g: Vec<f32> = (0..num_data).map(|i| (i as f32 * 0.013).sin() * 0.5).collect();
    let ord_h: Vec<f32> = (0..num_data).map(|i| 1.0 + (i % 7) as f32 * 0.01).collect();
    let feat_len = (2 * NUM_BIN) as u32;
    let slot_off: Vec<u32> = (0..feats as u32).map(|f| f * feat_len).collect();
    let slot_len = feats * 2 * NUM_BIN;
    let r = leaf_rows.len();
    let reads_per_launch = (feats * r) as f64;

    let d_bins = client.create_from_slice(u32::as_bytes(&bins));
    let d_rows = client.create_from_slice(u32::as_bytes(&leaf_rows));
    let d_g = client.create_from_slice(f32::as_bytes(&ord_g));
    let d_h = client.create_from_slice(f32::as_bytes(&ord_h));
    let d_slot = client.create_from_slice(u32::as_bytes(&slot_off));

    macro_rules! launch_f32 {
        ($p:expr, $out:expr) => {
            build_f32_rp::launch_unchecked(
                &client,
                CubeCount::Static(feats as u32, $p, 1),
                CubeDim::new_1d(CUBE_DIM),
                ArrayArg::from_raw_parts(d_bins.clone(), feats * num_data),
                ArrayArg::from_raw_parts(d_rows.clone(), r),
                ArrayArg::from_raw_parts(d_g.clone(), r),
                ArrayArg::from_raw_parts(d_h.clone(), r),
                ArrayArg::from_raw_parts(d_slot.clone(), feats),
                num_data,
                feat_len,
                ArrayArg::from_raw_parts($out.clone(), slot_len),
            )
        };
    }
    macro_rules! launch_u64 {
        ($p:expr, $out:expr) => {
            build_u64_rp::launch_unchecked(
                &client,
                CubeCount::Static(feats as u32, $p, 1),
                CubeDim::new_1d(CUBE_DIM),
                ArrayArg::from_raw_parts(d_bins.clone(), feats * num_data),
                ArrayArg::from_raw_parts(d_rows.clone(), r),
                ArrayArg::from_raw_parts(d_g.clone(), r),
                ArrayArg::from_raw_parts(d_h.clone(), r),
                ArrayArg::from_raw_parts(d_slot.clone(), feats),
                num_data,
                feat_len,
                ArrayArg::from_raw_parts($out.clone(), slot_len),
            )
        };
    }

    let bench_f32 = |p: u32| -> f64 {
        let out = client.create_from_slice(f32::as_bytes(&vec![0.0f32; slot_len]));
        let t = Instant::now();
        for _ in 0..LAUNCHES {
            unsafe { launch_f32!(p, out); }
        }
        let _ = client.read_one_unchecked(out);
        t.elapsed().as_secs_f64() * 1e3
    };
    let bench_u64 = |p: u32| -> f64 {
        let out = client.create_from_slice(u64::as_bytes(&vec![0u64; slot_len]));
        let t = Instant::now();
        for _ in 0..LAUNCHES {
            unsafe { launch_u64!(p, out); }
        }
        let _ = client.read_one_unchecked(out);
        t.elapsed().as_secs_f64() * 1e3
    };

    println!("\n## CONFIG {label}  (rows/cube@P1={num_data}, cubes@P1={feats})");

    // warm
    for &p in &p_sweep {
        let _ = bench_f32(p);
        let _ = bench_u64(p);
    }
    const REPS: usize = 9;
    let pct = |v: &mut Vec<f64>, q: f64| {
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        v[(((v.len() - 1) as f64) * q).round() as usize]
    };
    println!("# median[p25..p75] over {REPS} interleaved reps, ratio = f32_median/u64_median:");
    for &p in &p_sweep {
        let (mut sf, mut su): (Vec<f64>, Vec<f64>) = (vec![], vec![]);
        for _ in 0..REPS {
            sf.push(bench_f32(p));
            su.push(bench_u64(p));
        }
        let (f25, f50, f75) = (pct(&mut sf, 0.25), pct(&mut sf, 0.5), pct(&mut sf, 0.75));
        let (u25, u50, u75) = (pct(&mut su, 0.25), pct(&mut su, 0.5), pct(&mut su, 0.75));
        let mrs_f = reads_per_launch * LAUNCHES as f64 / (f50 / 1e3) / 1e6;
        let sep = if u75 < f25 { "SEP-WIN" } else { "overlap" };
        println!(
            "  P={p:>2}: f32 {f50:.0}ms[{f25:.0}..{f75:.0}] ({mrs_f:.0}Mr/s)  u64 {u50:.0}ms[{u25:.0}..{u75:.0}]  i64/f32={:.2}x {sep}",
            f50 / u50
        );
    }
    } // end config loop
    println!("\n# Diagnosis:");
    println!("#  i64 win ONLY in 1cube×1M ⇒ 018b's ~1.9x was UNDER-OCCUPANCY artifact (1 cube, 7 idle CUs).");
    println!("#  i64 win ALSO in 16cube×1M ⇒ real ROWS-PER-CUBE/contention effect (matters for big root leaves).");
    println!("#  null everywhere occupied ⇒ integer atomics give no realistic-regime speed win.");
}
