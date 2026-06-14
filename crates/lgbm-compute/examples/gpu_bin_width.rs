//! Spike 006 GPU micro-benchmark: does narrowing the device bin column (u32 → u8)
//! speed the resident histogram BUILD kernel on the gfx1100? Isolates the
//! device-bin-read width effect of the exact `construct_leaf_hist_resident_kernel`
//! access pattern (`resident_bins[f*num_data + leaf_rows[k]]`, scattered row order,
//! f32-atomic accumulate) WITHOUT plumbing a narrow type through the resident cache.
//!
//! Run: cargo run --release --features rocm --example gpu_bin_width
//!
//! Validates BOTH feasibility (does `Array<u8>` compile + run on HIP) and the perf
//! benefit (scattered GPU reads may benefit less from u8 density than CPU's L2 win).

#[cfg(feature = "rocm")]
use cubecl::prelude::*;

// One unit per (feature, leaf-row); the kernel gathers the bin from the resident
// feature-major column and f32-atomic-accumulates grad/hess. IDENTICAL to the
// production naive resident build — only the `resident_bins` element width differs.
#[cfg(feature = "rocm")]
#[cube(launch)]
fn build_u32(
    resident_bins: &Array<u32>,
    leaf_rows: &Array<u32>,
    ord_g: &Array<f32>,
    ord_h: &Array<f32>,
    slot_off: &Array<u32>,
    num_data: usize,
    total: usize,
    out: &mut Array<Atomic<f32>>,
) {
    let idx = ABSOLUTE_POS;
    if idx < total {
        let r = ord_g.len();
        let f = idx / r;
        let k = idx % r;
        let row = leaf_rows[k] as usize;
        let bin = resident_bins[f * num_data + row] as usize;
        let cell = slot_off[f] as usize + bin * 2;
        out[cell].fetch_add(ord_g[k]);
        out[cell + 1].fetch_add(ord_h[k]);
    }
}

#[cfg(feature = "rocm")]
#[cube(launch)]
fn build_u8(
    resident_bins: &Array<u8>,
    leaf_rows: &Array<u32>,
    ord_g: &Array<f32>,
    ord_h: &Array<f32>,
    slot_off: &Array<u32>,
    num_data: usize,
    total: usize,
    out: &mut Array<Atomic<f32>>,
) {
    let idx = ABSOLUTE_POS;
    if idx < total {
        let r = ord_g.len();
        let f = idx / r;
        let k = idx % r;
        let row = leaf_rows[k] as usize;
        let bin = resident_bins[f * num_data + row] as usize;
        let cell = slot_off[f] as usize + bin * 2;
        out[cell].fetch_add(ord_g[k]);
        out[cell + 1].fetch_add(ord_h[k]);
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

    const NUM_DATA: usize = 200_000;
    const FEATS: usize = 32;
    const NUM_BIN: usize = 64;
    const LAUNCHES: usize = 30;

    let client = rocm_client();

    let mut bins_u32: Vec<u32> = Vec::with_capacity(FEATS * NUM_DATA);
    for f in 0..FEATS {
        for r in 0..NUM_DATA {
            let h = (r as u64).wrapping_mul(2_654_435_761).wrapping_add(f as u64 * 97);
            bins_u32.push((h % NUM_BIN as u64) as u32);
        }
    }
    let bins_u8: Vec<u8> = bins_u32.iter().map(|&b| b as u8).collect();
    let leaf_rows: Vec<u32> = (0..NUM_DATA as u32)
        .map(|i| (i.wrapping_mul(2_654_435_761)) % NUM_DATA as u32)
        .collect();
    let ord_g: Vec<f32> = (0..NUM_DATA).map(|i| (i % 13) as f32 * 0.1).collect();
    let ord_h: Vec<f32> = (0..NUM_DATA).map(|i| 1.0 + (i % 7) as f32 * 0.01).collect();
    let slot_off: Vec<u32> = (0..FEATS as u32).map(|f| f * (2 * NUM_BIN as u32)).collect();
    let slot_len = FEATS * 2 * NUM_BIN;
    let total = FEATS * NUM_DATA;

    let d_u32 = client.create_from_slice(u32::as_bytes(&bins_u32));
    let d_u8 = client.create_from_slice(&bins_u8);
    let d_rows = client.create_from_slice(u32::as_bytes(&leaf_rows));
    let d_g = client.create_from_slice(f32::as_bytes(&ord_g));
    let d_h = client.create_from_slice(f32::as_bytes(&ord_h));
    let d_slot = client.create_from_slice(u32::as_bytes(&slot_off));

    let cube_dim = 256u32;
    let cube_count = total.div_ceil(cube_dim as usize) as u32;

    println!("# gpu bin-width microbench  num_data={NUM_DATA} feats={FEATS} num_bin={NUM_BIN} launches={LAUNCHES}");
    println!("# resident column: u32={}MB u8={}MB", FEATS * NUM_DATA * 4 / 1_048_576, FEATS * NUM_DATA / 1_048_576);

    let run_u32 = || {
        let out = client.create_from_slice(f32::as_bytes(&vec![0.0f32; slot_len]));
        for _ in 0..LAUNCHES {
            unsafe {
                build_u32::launch(
                    &client,
                    CubeCount::Static(cube_count, 1, 1),
                    CubeDim::new_1d(cube_dim),
                    ArrayArg::from_raw_parts(d_u32.clone(), FEATS * NUM_DATA),
                    ArrayArg::from_raw_parts(d_rows.clone(), NUM_DATA),
                    ArrayArg::from_raw_parts(d_g.clone(), NUM_DATA),
                    ArrayArg::from_raw_parts(d_h.clone(), NUM_DATA),
                    ArrayArg::from_raw_parts(d_slot.clone(), FEATS),
                    NUM_DATA,
                    total,
                    ArrayArg::from_raw_parts(out.clone(), slot_len),
                );
            }
        }
        let _ = client.read_one_unchecked(out);
    };
    let run_u8 = || {
        let out = client.create_from_slice(f32::as_bytes(&vec![0.0f32; slot_len]));
        for _ in 0..LAUNCHES {
            unsafe {
                build_u8::launch(
                    &client,
                    CubeCount::Static(cube_count, 1, 1),
                    CubeDim::new_1d(cube_dim),
                    ArrayArg::from_raw_parts(d_u8.clone(), FEATS * NUM_DATA),
                    ArrayArg::from_raw_parts(d_rows.clone(), NUM_DATA),
                    ArrayArg::from_raw_parts(d_g.clone(), NUM_DATA),
                    ArrayArg::from_raw_parts(d_h.clone(), NUM_DATA),
                    ArrayArg::from_raw_parts(d_slot.clone(), FEATS),
                    NUM_DATA,
                    total,
                    ArrayArg::from_raw_parts(out.clone(), slot_len),
                );
            }
        }
        let _ = client.read_one_unchecked(out);
    };

    run_u32();
    run_u8();
    for round in 1..=3 {
        let t = Instant::now();
        run_u32();
        let e32 = t.elapsed();
        let t = Instant::now();
        run_u8();
        let e8 = t.elapsed();
        println!(
            "round{round}: u32 {:>9.3?} ({:.0} Mreads/s)  u8 {:>9.3?} ({:.0} Mreads/s)  u8 {:+.1}%",
            e32,
            (total * LAUNCHES) as f64 / e32.as_secs_f64() / 1e6,
            e8,
            (total * LAUNCHES) as f64 / e8.as_secs_f64() / 1e6,
            (e8.as_secs_f64() / e32.as_secs_f64() - 1.0) * 100.0
        );
    }
}
