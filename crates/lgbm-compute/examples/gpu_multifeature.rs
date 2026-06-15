//! Spike 009 — multi-feature-per-cube packing vs one-cube-per-feature (Lever: "ours
//! wastes a CU on small-bin features"). Parity-SAFE (same f32 exact accumulation).
//!
//! LightGBM packs many feature-columns into one block's shared histogram (block_dim_x over
//! columns). Our build runs one cube per feature (× P row-partitions). For MANY SMALL-bin
//! features, packing G features into one cube amortizes the per-cube fixed overhead (zero +
//! 2 syncs + merge) across G features — BUT it also divides the cube count by G, which fights
//! the occupancy the row-partition lever (spike-007) just bought.
//!
//! So this benches a MATCHED-OCCUPANCY comparison: tune the per-feature P and the packed P so
//! BOTH launch ~the same total cubes (~the gfx1100 8-wkgrps/CU target). That isolates the
//! overhead-amortization effect from the occupancy effect. If packing still wins at matched
//! occupancy, it's worth shipping; if not, it's a null (occupancy already solved it).
//!
//! Run: cargo run --release --features rocm --example gpu_multifeature

#[cfg(feature = "rocm")]
use cubecl::prelude::*;

#[cfg(feature = "rocm")]
const HIST_LDS_MAX: usize = 512;

/// One cube per (feature, row-partition) — the shipped row-partitioned build (baseline).
#[cfg(feature = "rocm")]
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
fn build_per_feature(
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
    let stride = CUBE_COUNT_Y as usize * cd;
    let mut k = CUBE_POS_Y as usize * cd + UNIT_POS as usize;
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

/// One cube per (feature-GROUP of G, row-partition). The cube holds G concatenated LDS
/// sub-histograms; it zeroes/merges `G*feat_len` cells once, and scatters all G features'
/// rows. `gpf` = G (features per group, comptime via scalar). LDS layout: feature j of the
/// group occupies `[j*feat_len, (j+1)*feat_len)`.
#[cfg(feature = "rocm")]
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
fn build_packed(
    resident_bins: &Array<u32>,
    leaf_rows: &Array<u32>,
    ord_g: &Array<f32>,
    ord_h: &Array<f32>,
    slot_off: &Array<u32>, // per-feature global slot starts
    num_data: usize,
    feat_len: u32,
    gpf: u32, // features per group
    num_features: u32,
    out: &mut Array<Atomic<f32>>,
) {
    let g = CUBE_POS_X; // group index
    let cd = CUBE_DIM as usize;
    let fl = feat_len as usize;
    let gp = gpf as usize;
    let r = ord_g.len();
    let active = fl * gp; // active LDS cells for this group

    let sub = SharedMemory::<Atomic<f32>>::new(HIST_LDS_MAX);
    let mut c = UNIT_POS as usize;
    while c < active {
        sub[c].store(0.0f32);
        c += cd;
    }
    sync_cube();

    // Scatter all G features of this group. Feature index = g*gpf + j.
    let stride = CUBE_COUNT_Y as usize * cd;
    let mut j = 0u32;
    while j < gpf {
        let f = g * gpf + j;
        if f < num_features {
            let col = f as usize * num_data;
            let ldsb = j as usize * fl;
            let mut k = CUBE_POS_Y as usize * cd + UNIT_POS as usize;
            while k < r {
                let ti = ldsb + resident_bins[col + leaf_rows[k] as usize] as usize * 2;
                sub[ti].fetch_add(ord_g[k]);
                sub[ti + 1].fetch_add(ord_h[k]);
                k += stride;
            }
        }
        j += 1;
    }
    sync_cube();

    // Merge each feature's sub-hist into its global slot.
    let mut jj = 0u32;
    while jj < gpf {
        let f = g * gpf + jj;
        if f < num_features {
            let base = slot_off[f as usize] as usize;
            let ldsb = jj as usize * fl;
            let mut m = UNIT_POS as usize;
            while m < fl {
                out[base + m].fetch_add(sub[ldsb + m].load());
                m += cd;
            }
        }
        jj += 1;
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
    const FEATS: usize = 128; // MANY features
    const NUM_BIN: usize = 32; // SMALL bins → packing's best case (8 fit one LDS)
    const LAUNCHES: usize = 20;
    const CUBE_DIM: u32 = 256;
    const TARGET_CUBES: u32 = 768; // ~8 wkgrps × 96 CU

    let client = rocm_client();
    let mut bins: Vec<u32> = Vec::with_capacity(FEATS * NUM_DATA);
    for f in 0..FEATS {
        for r in 0..NUM_DATA {
            let h = (r as u64).wrapping_mul(2_654_435_761).wrapping_add(f as u64 * 97);
            bins.push((h % NUM_BIN as u64) as u32);
        }
    }
    let leaf_rows: Vec<u32> = (0..NUM_DATA as u32)
        .map(|i| i.wrapping_mul(2_654_435_761) % NUM_DATA as u32)
        .collect();
    let g: Vec<f32> = (0..NUM_DATA).map(|i| (i % 13) as f32 * 0.1 - 0.5).collect();
    let h: Vec<f32> = (0..NUM_DATA).map(|i| 1.0 + (i % 7) as f32 * 0.01).collect();
    let feat_len = (2 * NUM_BIN) as u32;
    let slot_off: Vec<u32> = (0..FEATS as u32).map(|f| f * feat_len).collect();
    let slot_len = FEATS * 2 * NUM_BIN;
    let r = leaf_rows.len();

    let d_bins = client.create_from_slice(u32::as_bytes(&bins));
    let d_rows = client.create_from_slice(u32::as_bytes(&leaf_rows));
    let d_g = client.create_from_slice(f32::as_bytes(&g));
    let d_h = client.create_from_slice(f32::as_bytes(&h));
    let d_slot = client.create_from_slice(u32::as_bytes(&slot_off));

    let gpf = (HIST_LDS_MAX / feat_len as usize) as u32; // features per cube (8 here)
    let num_groups = (FEATS as u32).div_ceil(gpf);
    // Matched occupancy: per-feature has FEATS base cubes → P_pf to reach ~TARGET_CUBES.
    // packed has num_groups base cubes → P_pk to reach ~the same total.
    let p_pf = (TARGET_CUBES / FEATS as u32).max(1); // 768/128 = 6
    let p_pk = (TARGET_CUBES / num_groups).max(1); // 768/16 = 48 → matched total cubes
    println!("# multifeature  data={NUM_DATA} feats={FEATS} bins={NUM_BIN}  gpf={gpf} groups={num_groups}");
    println!("# per-feature: {FEATS}×P_pf({p_pf}) = {} cubes", FEATS as u32 * p_pf);
    println!("# packed:      {num_groups}×P_pk({p_pk}) = {} cubes (matched occupancy)", num_groups * p_pk);

    let run_pf = |p: u32| -> std::time::Duration {
        let out = client.create_from_slice(f32::as_bytes(&vec![0.0f32; slot_len]));
        let t = Instant::now();
        for _ in 0..LAUNCHES {
            unsafe {
                build_per_feature::launch(
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
    let run_pk = |p: u32| -> std::time::Duration {
        let out = client.create_from_slice(f32::as_bytes(&vec![0.0f32; slot_len]));
        let t = Instant::now();
        for _ in 0..LAUNCHES {
            unsafe {
                build_packed::launch(
                    &client,
                    CubeCount::Static(num_groups, p, 1),
                    CubeDim::new_1d(CUBE_DIM),
                    ArrayArg::from_raw_parts(d_bins.clone(), FEATS * NUM_DATA),
                    ArrayArg::from_raw_parts(d_rows.clone(), r),
                    ArrayArg::from_raw_parts(d_g.clone(), r),
                    ArrayArg::from_raw_parts(d_h.clone(), r),
                    ArrayArg::from_raw_parts(d_slot.clone(), FEATS),
                    NUM_DATA,
                    feat_len,
                    gpf,
                    FEATS as u32,
                    ArrayArg::from_raw_parts(out.clone(), slot_len),
                );
            }
        }
        let _ = client.read_one_unchecked(out);
        t.elapsed()
    };

    // Correctness: packed must match per-feature within f32-atomic noise.
    let read_pf = || {
        let out = client.create_from_slice(f32::as_bytes(&vec![0.0f32; slot_len]));
        unsafe {
            build_per_feature::launch(
                &client, CubeCount::Static(FEATS as u32, p_pf, 1), CubeDim::new_1d(CUBE_DIM),
                ArrayArg::from_raw_parts(d_bins.clone(), FEATS * NUM_DATA),
                ArrayArg::from_raw_parts(d_rows.clone(), r), ArrayArg::from_raw_parts(d_g.clone(), r),
                ArrayArg::from_raw_parts(d_h.clone(), r), ArrayArg::from_raw_parts(d_slot.clone(), FEATS),
                NUM_DATA, feat_len, ArrayArg::from_raw_parts(out.clone(), slot_len),
            );
        }
        f32::from_bytes(&client.read_one_unchecked(out)).to_vec()
    };
    let read_pk = || {
        let out = client.create_from_slice(f32::as_bytes(&vec![0.0f32; slot_len]));
        unsafe {
            build_packed::launch(
                &client, CubeCount::Static(num_groups, p_pk, 1), CubeDim::new_1d(CUBE_DIM),
                ArrayArg::from_raw_parts(d_bins.clone(), FEATS * NUM_DATA),
                ArrayArg::from_raw_parts(d_rows.clone(), r), ArrayArg::from_raw_parts(d_g.clone(), r),
                ArrayArg::from_raw_parts(d_h.clone(), r), ArrayArg::from_raw_parts(d_slot.clone(), FEATS),
                NUM_DATA, feat_len, gpf, FEATS as u32, ArrayArg::from_raw_parts(out.clone(), slot_len),
            );
        }
        f32::from_bytes(&client.read_one_unchecked(out)).to_vec()
    };
    let a = read_pf();
    let b = read_pk();
    let mrel = a.iter().zip(b.iter())
        .map(|(x, y)| (*x as f64 - *y as f64).abs() / (*x as f64).abs().max(1.0))
        .fold(0.0f64, f64::max);
    println!("# correctness packed-vs-per-feature max_rel={mrel:.2e}\n");

    let _ = run_pf(p_pf);
    let _ = run_pk(p_pk);
    for round in 1..=3 {
        let e_pf = run_pf(p_pf);
        let e_pk = run_pk(p_pk);
        let ms_pf = e_pf.as_secs_f64() * 1e3;
        let ms_pk = e_pk.as_secs_f64() * 1e3;
        println!("round{round}:  per-feature={ms_pf:.0}ms  packed={ms_pk:.0}ms  packed/pf={:.2}x", ms_pf / ms_pk);
    }
    println!("\n# packed/pf > 1.0 ⇒ multi-feature packing wins at matched occupancy; ship it.");
    println!("# <= 1.0 ⇒ null (row-partition occupancy already captured it); keep one-cube-per-feature.");
}
