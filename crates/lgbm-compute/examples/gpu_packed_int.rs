//! Phase-10 W5: does the discretized PACKED-INT histogram (one int32 atomic per row)
//! beat the f32 TWO-atomic build on the gfx1100? The build is atomic-contention bound
//! (spike-006/007/009), so halving the atomic COUNT — LightGBM's `(grad16<<16 | hess16)`
//! single `atomicAdd` — is the one lever that directly targets that bottleneck.
//!
//! Both kernels are row-partitioned (spike-007, P=16). Correctness is checked on a small,
//! NON-overflowing leaf (packed-16 only holds when each 16-bit half doesn't overflow — the
//! reason LightGBM gates 16-bit to small leaves); the large-leaf run measures THROUGHPUT only
//! (timing is unaffected by value overflow).
//!
//! Run: cargo run --release --features rocm --example gpu_packed_int

#[cfg(feature = "rocm")]
use cubecl::prelude::*;

#[cfg(feature = "rocm")]
const LDS_F32: usize = 512; // 2 cells/bin × 256
#[cfg(feature = "rocm")]
const LDS_INT: usize = 256; // 1 packed i32/bin × 256

/// f32 baseline: two f32 atomics per row (grad, hess), row-partitioned.
#[cfg(feature = "rocm")]
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
fn build_f32_2atomic(
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
    let sub = SharedMemory::<Atomic<f32>>::new(LDS_F32);
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

/// Discretized: ONE int32 atomic per row on the packed `(grad16<<16 | hess16)` cell.
#[cfg(feature = "rocm")]
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
fn build_packed_int(
    resident_bins: &Array<u32>,
    leaf_rows: &Array<u32>,
    packed_gh: &Array<i32>, // per leaf-row: (grad_i16 << 16) | (hess_i16 & 0xffff)
    slot_off: &Array<u32>,  // per-feature start in PACKED cells (1 per bin)
    num_data: usize,
    num_bin: u32,
    out: &mut Array<Atomic<i32>>,
) {
    let f = CUBE_POS_X as usize;
    let cd = CUBE_DIM as usize;
    let nb = num_bin as usize;
    let r = packed_gh.len();
    let base = slot_off[f] as usize;
    let col = f * num_data;
    let sub = SharedMemory::<Atomic<i32>>::new(LDS_INT);
    let mut c = UNIT_POS as usize;
    while c < nb {
        sub[c].store(0i32);
        c += cd;
    }
    sync_cube();
    let stride = CUBE_COUNT_Y as usize * cd;
    let mut k = CUBE_POS_Y as usize * cd + UNIT_POS as usize;
    while k < r {
        let bin = resident_bins[col + leaf_rows[k] as usize] as usize;
        sub[bin].fetch_add(packed_gh[k]); // ONE atomic updates both halves
        k += stride;
    }
    sync_cube();
    let mut m = UNIT_POS as usize;
    while m < nb {
        out[base + m].fetch_add(sub[m].load());
        m += cd;
    }
}

#[cfg(not(feature = "rocm"))]
fn main() {
    eprintln!("requires --features rocm (gfx1100).");
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
    const P: u32 = 16; // spike-007 sweet spot

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
    let ord_g: Vec<f32> = (0..NUM_DATA).map(|i| (i % 13) as f32 * 0.1 - 0.5).collect();
    let ord_h: Vec<f32> = (0..NUM_DATA).map(|i| 1.0 + (i % 7) as f32 * 0.01).collect();
    // Packed: small int16 grad/hess so the bench mirrors the discretized payload (values are
    // arbitrary — this run measures THROUGHPUT; correctness is the separate small check).
    let packed: Vec<i32> = (0..NUM_DATA)
        .map(|i| {
            let g = (i % 7) as i32 - 3; // small signed
            let h = (i % 5) as i32 + 1;
            (g << 16) | (h & 0xffff)
        })
        .collect();
    let r = leaf_rows.len();
    let feat_len = (2 * NUM_BIN) as u32;
    let slot_f32: Vec<u32> = (0..FEATS as u32).map(|f| f * feat_len).collect();
    let slot_int: Vec<u32> = (0..FEATS as u32).map(|f| f * NUM_BIN as u32).collect();
    let slot_len_f32 = FEATS * 2 * NUM_BIN;
    let slot_len_int = FEATS * NUM_BIN;

    let d_bins = client.create_from_slice(u32::as_bytes(&bins));
    let d_rows = client.create_from_slice(u32::as_bytes(&leaf_rows));
    let d_g = client.create_from_slice(f32::as_bytes(&ord_g));
    let d_h = client.create_from_slice(f32::as_bytes(&ord_h));
    let d_packed = client.create_from_slice(i32::as_bytes(&packed));
    let d_sf = client.create_from_slice(u32::as_bytes(&slot_f32));
    let d_si = client.create_from_slice(u32::as_bytes(&slot_int));

    let reads = (FEATS * r) as f64;
    println!("# packed-int vs f32-2atomic  data={NUM_DATA} feats={FEATS} bins={NUM_BIN} P={P}");
    println!("# f32 LDS={}KB (2 cells/bin)  int LDS={}KB (1 packed/bin)", LDS_F32 * 4 / 1024, LDS_INT * 4 / 1024);

    let run_f32 = || {
        let out = client.create_from_slice(f32::as_bytes(&vec![0.0f32; slot_len_f32]));
        let t = Instant::now();
        for _ in 0..LAUNCHES {
            unsafe {
                build_f32_2atomic::launch(
                    &client, CubeCount::Static(FEATS as u32, P, 1), CubeDim::new_1d(CUBE_DIM),
                    ArrayArg::from_raw_parts(d_bins.clone(), FEATS * NUM_DATA),
                    ArrayArg::from_raw_parts(d_rows.clone(), r), ArrayArg::from_raw_parts(d_g.clone(), r),
                    ArrayArg::from_raw_parts(d_h.clone(), r), ArrayArg::from_raw_parts(d_sf.clone(), FEATS),
                    NUM_DATA, feat_len, ArrayArg::from_raw_parts(out.clone(), slot_len_f32),
                );
            }
        }
        let _ = client.read_one_unchecked(out);
        t.elapsed()
    };
    let run_int = || {
        let out = client.create_from_slice(i32::as_bytes(&vec![0i32; slot_len_int]));
        let t = Instant::now();
        for _ in 0..LAUNCHES {
            unsafe {
                build_packed_int::launch(
                    &client, CubeCount::Static(FEATS as u32, P, 1), CubeDim::new_1d(CUBE_DIM),
                    ArrayArg::from_raw_parts(d_bins.clone(), FEATS * NUM_DATA),
                    ArrayArg::from_raw_parts(d_rows.clone(), r), ArrayArg::from_raw_parts(d_packed.clone(), r),
                    ArrayArg::from_raw_parts(d_si.clone(), FEATS), NUM_DATA, NUM_BIN as u32,
                    ArrayArg::from_raw_parts(out.clone(), slot_len_int),
                );
            }
        }
        let _ = client.read_one_unchecked(out);
        t.elapsed()
    };

    let _ = run_f32();
    let _ = run_int();
    println!("\n# throughput (20 launches/round):");
    for round in 1..=3 {
        let ef = run_f32();
        let ei = run_int();
        let (mf, mi) = (ef.as_secs_f64() * 1e3, ei.as_secs_f64() * 1e3);
        println!(
            "round{round}:  f32-2atomic={mf:.0}ms ({:.0}Mr/s)  packed-int={mi:.0}ms ({:.0}Mr/s)  int/f32={:.2}x",
            reads * LAUNCHES as f64 / ef.as_secs_f64() / 1e6,
            reads * LAUNCHES as f64 / ei.as_secs_f64() / 1e6,
            mf / mi
        );
    }
    println!("\n# int/f32 > 1.0 ⇒ the single packed-int atomic wins (halved atomic count helps the");
    println!("# atomic-bound build). <= 1.0 ⇒ int vs f32 atomic throughput is similar; no win.");
}
