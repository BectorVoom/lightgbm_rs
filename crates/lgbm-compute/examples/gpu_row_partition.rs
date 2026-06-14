//! Spike 007 GPU micro-benchmark: does ROW-PARTITIONING the per-feature LDS
//! histogram build raise GPU occupancy enough to beat the current one-cube-per-feature
//! kernel on a large leaf?
//!
//! The wired `construct_leaf_hist_resident_lds_kernel` launches `CubeCount = (num_features, 1)`
//! — ONE workgroup per feature. On a 1M-row / 50-feature leaf that is only 50 workgroups
//! on a 96-CU gfx1100, which wants several resident workgroups per CU to hide the
//! scattered-read / atomic latency that spike 006 measured (234 Mreads/s, "latency-bound").
//!
//! This bench parameterizes a row-partition `P`: `CubeCount = (num_features, P)`. Cube
//! `(f, p)` folds the leaf's rows `p*256+u, +P*256, …` into its OWN LDS sub-histogram,
//! then atomic-merges that sub-hist into feature f's global slot. **P = 1 is byte-identical
//! to the production kernel**, so sweeping P cleanly isolates the occupancy effect.
//! Global-merge atomic traffic grows only to `P * num_features * 2*num_bin` (negligible vs
//! the `2*n` row atomics). This mirrors LightGBM's `grid_dim_y` row split.
//!
//! Run: cargo run --release --features rocm --example gpu_row_partition

#[cfg(feature = "rocm")]
use cubecl::prelude::*;

/// LDS sub-hist cap: 512 f32 cells = 256 bins (grad+hess interleaved) = 2 KiB.
#[cfg(feature = "rocm")]
const HIST_LDS_MAX: usize = 512;

/// Row-partitioned LDS resident build. `CubeCount = (num_features, P)`; cube `(f, p)`
/// owns a private LDS sub-hist and strides its partition's rows into it, then merges to
/// feature f's global slot. P=1 reduces exactly to `construct_leaf_hist_resident_lds_kernel`.
#[cfg(feature = "rocm")]
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
fn build_rp(
    resident_bins: &Array<u32>,
    leaf_rows: &Array<u32>,
    ord_g: &Array<f32>,
    ord_h: &Array<f32>,
    slot_off: &Array<u32>, // start cell per feature (length num_features)
    num_data: usize,       // resident column stride
    feat_len: u32,         // 2*num_bin (active LDS cells; same for all feats here)
    out: &mut Array<Atomic<f32>>,
) {
    let f = CUBE_POS_X as usize; // feature
    let p = CUBE_POS_Y as usize; // row partition
    let np = CUBE_COUNT_Y as usize; // P
    let cd = CUBE_DIM as usize; // 256
    let fl = feat_len as usize;
    let r = ord_g.len();
    let base = slot_off[f] as usize;
    let col = f * num_data;

    let sub = SharedMemory::<Atomic<f32>>::new(HIST_LDS_MAX);
    // 1. zero this cube's active LDS cells.
    let mut c = UNIT_POS as usize;
    while c < fl {
        sub[c].store(0.0f32);
        c += cd;
    }
    sync_cube();
    // 2. scatter THIS partition's strided rows into the LDS sub-hist (LDS atomics).
    let stride = np * cd;
    let mut k = p * cd + UNIT_POS as usize;
    while k < r {
        let row = leaf_rows[k] as usize;
        let bin = resident_bins[col + row] as usize;
        let ti = bin * 2;
        sub[ti].fetch_add(ord_g[k]);
        sub[ti + 1].fetch_add(ord_h[k]);
        k += stride;
    }
    sync_cube();
    // 3. merge LDS sub-hist into feature f's global slot (one global atomic per cell).
    let mut m = UNIT_POS as usize;
    while m < fl {
        out[base + m].fetch_add(sub[m].load());
        m += cd;
    }
}

/// K=4 register-batched variant of `build_rp` (Lever 2, LightGBM `NUM_DATA_PER_THREAD`):
/// each unit gathers 4 strided rows' bins + grad/hess, then issues the 4 LDS atomics — so
/// the hardware pipelines the 4 independent scattered reads before the dependent atomics.
/// Same row-partition layout as `build_rp` (P = `CUBE_COUNT_Y`).
#[cfg(feature = "rocm")]
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
fn build_rp_k4(
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

    let sub = SharedMemory::<Atomic<f32>>::new(HIST_LDS_MAX);
    let mut c = UNIT_POS as usize;
    while c < fl {
        sub[c].store(0.0f32);
        c += cd;
    }
    sync_cube();

    let stride = np * cd;
    let s2 = stride * 2;
    let s3 = stride * 3;
    let step = stride * 4;
    let mut k = p * cd + UNIT_POS as usize;
    // 4-row register-batched body: gather all 4 bins/grads/hesses first (independent
    // loads the compiler can pipeline), then the 4 LDS atomics.
    while k + s3 < r {
        let b0 = resident_bins[col + leaf_rows[k] as usize] as usize * 2;
        let b1 = resident_bins[col + leaf_rows[k + stride] as usize] as usize * 2;
        let b2 = resident_bins[col + leaf_rows[k + s2] as usize] as usize * 2;
        let b3 = resident_bins[col + leaf_rows[k + s3] as usize] as usize * 2;
        let g0 = ord_g[k];
        let g1 = ord_g[k + stride];
        let g2 = ord_g[k + s2];
        let g3 = ord_g[k + s3];
        let h0 = ord_h[k];
        let h1 = ord_h[k + stride];
        let h2 = ord_h[k + s2];
        let h3 = ord_h[k + s3];
        sub[b0].fetch_add(g0);
        sub[b0 + 1].fetch_add(h0);
        sub[b1].fetch_add(g1);
        sub[b1 + 1].fetch_add(h1);
        sub[b2].fetch_add(g2);
        sub[b2 + 1].fetch_add(h2);
        sub[b3].fetch_add(g3);
        sub[b3 + 1].fetch_add(h3);
        k += step;
    }
    // tail (< 4 rows left for this unit)
    while k < r {
        let ti = resident_bins[col + leaf_rows[k] as usize] as usize * 2;
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

#[cfg(not(feature = "rocm"))]
fn main() {
    eprintln!("this micro-bench requires --features rocm (gfx1100). Re-run with it.");
}

#[cfg(feature = "rocm")]
fn main() {
    use lgbm_compute::runtime::rocm_client;
    use std::time::Instant;

    const NUM_DATA: usize = 1_000_000;
    const FEATS: usize = 50;
    const NUM_BIN: usize = 256;
    const LAUNCHES: usize = 20;
    const CUBE_DIM: u32 = 256;
    let partitions: [u32; 6] = [1, 2, 4, 8, 16, 32];

    let client = rocm_client();

    // Resident feature-major bin column (50 × 1M u32 = 200 MB), scattered leaf rows.
    let mut bins: Vec<u32> = Vec::with_capacity(FEATS * NUM_DATA);
    for f in 0..FEATS {
        for r in 0..NUM_DATA {
            let h = (r as u64).wrapping_mul(2_654_435_761).wrapping_add(f as u64 * 97);
            bins.push((h % NUM_BIN as u64) as u32);
        }
    }
    let leaf_rows: Vec<u32> = (0..NUM_DATA as u32)
        .map(|i| (i.wrapping_mul(2_654_435_761)) % NUM_DATA as u32)
        .collect();
    let ord_g: Vec<f32> = (0..NUM_DATA).map(|i| (i % 13) as f32 * 0.1).collect();
    let ord_h: Vec<f32> = (0..NUM_DATA).map(|i| 1.0 + (i % 7) as f32 * 0.01).collect();
    let feat_len = (2 * NUM_BIN) as u32;
    let slot_off: Vec<u32> = (0..FEATS as u32).map(|f| f * feat_len).collect();
    let slot_len = FEATS * 2 * NUM_BIN;
    let r = leaf_rows.len();

    let d_bins = client.create_from_slice(u32::as_bytes(&bins));
    let d_rows = client.create_from_slice(u32::as_bytes(&leaf_rows));
    let d_g = client.create_from_slice(f32::as_bytes(&ord_g));
    let d_h = client.create_from_slice(f32::as_bytes(&ord_h));
    let d_slot = client.create_from_slice(u32::as_bytes(&slot_off));

    let reads_per_launch = (FEATS * r) as f64;

    // One clean launch into a freshly-zeroed `out`; returns the f32 histogram.
    let verify = |p: u32| -> Vec<f32> {
        let out = client.create_from_slice(f32::as_bytes(&vec![0.0f32; slot_len]));
        unsafe {
            build_rp::launch(
                &client,
                CubeCount::Static(FEATS as u32, p, 1),
                CubeDim::new_1d(CUBE_DIM),
                ArrayArg::from_raw_parts(d_bins.clone(), FEATS * NUM_DATA),
                ArrayArg::from_raw_parts(d_rows.clone(), r),
                ArrayArg::from_raw_parts(d_g.clone(), r),
                ArrayArg::from_raw_parts(d_h.clone(), r),
                ArrayArg::from_raw_parts(d_slot.clone(), FEATS),
                NUM_DATA,
                feat_len,
                ArrayArg::from_raw_parts(out.clone(), slot_len),
            );
        }
        f32::from_bytes(&client.read_one_unchecked(out)).to_vec()
    };

    // Throughput: LAUNCHES accumulating launches, final read forces device sync.
    let bench = |p: u32| -> std::time::Duration {
        let out = client.create_from_slice(f32::as_bytes(&vec![0.0f32; slot_len]));
        let t = Instant::now();
        for _ in 0..LAUNCHES {
            unsafe {
                build_rp::launch(
                    &client,
                    CubeCount::Static(FEATS as u32, p, 1),
                    CubeDim::new_1d(CUBE_DIM),
                    ArrayArg::from_raw_parts(d_bins.clone(), FEATS * NUM_DATA),
                    ArrayArg::from_raw_parts(d_rows.clone(), r),
                    ArrayArg::from_raw_parts(d_g.clone(), r),
                    ArrayArg::from_raw_parts(d_h.clone(), r),
                    ArrayArg::from_raw_parts(d_slot.clone(), FEATS),
                    NUM_DATA,
                    feat_len,
                    ArrayArg::from_raw_parts(out.clone(), slot_len),
                );
            }
        }
        let _ = client.read_one_unchecked(out);
        t.elapsed()
    };

    println!("# gpu row-partition microbench  num_data={NUM_DATA} feats={FEATS} num_bin={NUM_BIN} launches={LAUNCHES}");
    println!("# P=1 == production one-cube-per-feature kernel (50 workgroups). gfx1100 = 96 CUs.");
    println!("# resident column: {} MB", FEATS * NUM_DATA * 4 / 1_048_576);

    // Correctness: every P must match the P=1 build to f32-atomic noise.
    let ref1 = verify(1);
    println!("\n# correctness vs P=1 (f32-atomic reorder noise expected):");
    for &p in &partitions {
        let v = verify(p);
        let (mut max_abs, mut max_rel) = (0.0f64, 0.0f64);
        for (a, b) in ref1.iter().zip(v.iter()) {
            let d = (*a as f64 - *b as f64).abs();
            max_abs = max_abs.max(d);
            let denom = (*a as f64).abs().max(1.0);
            max_rel = max_rel.max(d / denom);
        }
        println!("  P={p:>2}: cubes={:>5}  max_abs_diff={max_abs:.4e}  max_rel_diff={max_rel:.2e}", FEATS as u32 * p);
    }

    // Warm up, then 3 timed rounds across the full P sweep.
    for &p in &partitions {
        let _ = bench(p);
    }
    println!("\n# wall-clock ({LAUNCHES} launches/round):");
    let mut base_ms = [0.0f64; 6];
    for round in 1..=3 {
        print!("round{round}:");
        for (i, &p) in partitions.iter().enumerate() {
            let e = bench(p);
            let ms = e.as_secs_f64() * 1e3;
            if round == 1 {
                base_ms[i] = ms;
            }
            let mrs = reads_per_launch * LAUNCHES as f64 / e.as_secs_f64() / 1e6;
            let spd = base_ms[0] / ms; // vs P=1 round-1 baseline
            print!("  P{p}={ms:.0}ms({mrs:.0}Mr/s,{spd:.2}x)");
        }
        println!();
    }
    println!("\n# speedup column is vs P=1 round-1. >1.0 = row-partitioning wins.");

    // --- Lever 2: register-batching (K=4) vs K=1 at the winning P=16 ---
    let p_best = 16u32;
    let bench_k4 = |p: u32| -> std::time::Duration {
        let out = client.create_from_slice(f32::as_bytes(&vec![0.0f32; slot_len]));
        let t = Instant::now();
        for _ in 0..LAUNCHES {
            unsafe {
                build_rp_k4::launch(
                    &client,
                    CubeCount::Static(FEATS as u32, p, 1),
                    CubeDim::new_1d(CUBE_DIM),
                    ArrayArg::from_raw_parts(d_bins.clone(), FEATS * NUM_DATA),
                    ArrayArg::from_raw_parts(d_rows.clone(), r),
                    ArrayArg::from_raw_parts(d_g.clone(), r),
                    ArrayArg::from_raw_parts(d_h.clone(), r),
                    ArrayArg::from_raw_parts(d_slot.clone(), FEATS),
                    NUM_DATA,
                    feat_len,
                    ArrayArg::from_raw_parts(out.clone(), slot_len),
                );
            }
        }
        let _ = client.read_one_unchecked(out);
        t.elapsed()
    };
    // correctness: K4 must match K1 within f32-atomic noise.
    let k1v = verify(p_best);
    let k4_out = {
        let out = client.create_from_slice(f32::as_bytes(&vec![0.0f32; slot_len]));
        unsafe {
            build_rp_k4::launch(
                &client,
                CubeCount::Static(FEATS as u32, p_best, 1),
                CubeDim::new_1d(CUBE_DIM),
                ArrayArg::from_raw_parts(d_bins.clone(), FEATS * NUM_DATA),
                ArrayArg::from_raw_parts(d_rows.clone(), r),
                ArrayArg::from_raw_parts(d_g.clone(), r),
                ArrayArg::from_raw_parts(d_h.clone(), r),
                ArrayArg::from_raw_parts(d_slot.clone(), FEATS),
                NUM_DATA,
                feat_len,
                ArrayArg::from_raw_parts(out.clone(), slot_len),
            );
        }
        f32::from_bytes(&client.read_one_unchecked(out)).to_vec()
    };
    let k_max_rel = k1v
        .iter()
        .zip(k4_out.iter())
        .map(|(a, b)| (*a as f64 - *b as f64).abs() / (*a as f64).abs().max(1.0))
        .fold(0.0f64, f64::max);
    let _ = bench(p_best);
    let _ = bench_k4(p_best);
    println!("\n# Lever 2: register-batching K=4 vs K=1 at P={p_best} (correctness max_rel={k_max_rel:.2e}):");
    for round in 1..=3 {
        let e1 = bench(p_best);
        let e4 = bench_k4(p_best);
        let ms1 = e1.as_secs_f64() * 1e3;
        let ms4 = e4.as_secs_f64() * 1e3;
        println!(
            "round{round}:  K1={ms1:.0}ms  K4={ms4:.0}ms  K4/K1={:.2}x",
            ms1 / ms4
        );
    }
    println!("# K4/K1 > 1.0 ⇒ register-batching helps; ship it. <= 1.0 ⇒ keep K=1 (null result).");
}