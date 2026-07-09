//! Coalesced-reorder NET cost. Measures whether reordering rows into coalesced order before a
//! histogram build is worth its own cost, netted against the coalescing saving it enables:
//!
//!   net_ms  = REAL_ORDER_ms (current production: uncoalesced gather + atomic build)
//!           - (REORDER_ms + COAL_OVER_REORDERED_ms)  (candidate: reorder pass + coalesced build)
//!
//! net_ms > 0  ⇒ the reorder-before-build pipeline is FASTER than production as-is ⇒
//!               a genuine, launch-bound-adjacent GPU lever worth building for real
//!               (bit-exact by construction: the histogram is order-independent for the u64
//!               fixed-point build, so any row permutation is exact).
//! net_ms <= 0 ⇒ the reorder overhead eats the coalescing saving ⇒ no lever.
//!
//! REORDER kernel: one cube per feature (matches build's launch shape), each thread gathers
//! `resident_bins[col + leaf_rows[k]]` (the SAME random-access pattern REAL_ORDER's build pays
//! for its bin read) and writes it to `scratch[f*r + k]` — sequential, i.e. the write is
//! coalesced; only the READ is uncoalesced (same as build's own bin gather, minus the atomic
//! histogram accumulation). This isolates "sort/scatter" cost from "coalesced build" cost.
//!
//! COAL_OVER_REORDERED: the coalesced-build kernel, launched over the `scratch` buffer at row
//! count `r_real` (the leaf's actual row count) instead of `num_data` — apples-to-apples with
//! REAL_ORDER (same row count), so the two candidate-pipeline stages sum to a directly
//! comparable total.
//!
//! Run:
//!   local APU:  cargo run --release -p lgbm-compute --features rocm --example spike088_reorder_net
//!   Kaggle CUDA: cargo run --release -p lgbm-compute --features cuda --example spike088_reorder_net

#[cfg(not(any(feature = "rocm", feature = "cuda")))]
fn main() {
    eprintln!(
        "spike-088 requires --features rocm (local APU) or --features cuda (Kaggle real NVIDIA)."
    );
}

#[cfg(any(feature = "rocm", feature = "cuda"))]
use cubecl::prelude::*;

#[cfg(any(feature = "rocm", feature = "cuda"))]
const HIST_LDS_U64: usize = 512; // 2 u64/bin, NUM_BIN <= 256
#[cfg(any(feature = "rocm", feature = "cuda"))]
const SCALE: f32 = 1_073_741_824.0; // 2^30 (== production SCALE_F32)

// ─── FULL: byte-faithful copy of construct_leaf_hist_resident_lds_kernel_u64 ───
#[cfg(any(feature = "rocm", feature = "cuda"))]
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn build_full(
    resident_bins: &Array<u8>,
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
        let bin = u32::cast_from(resident_bins[col + leaf_rows[k] as usize]) as usize;
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

// ─── REORDER: the sort/scatter pass. One cube per feature; gathers the leaf's rows out of
//     `resident_bins` (random read, same pattern build's own gather pays) and writes them
//     CONTIGUOUSLY into `scratch[f*r + k]` (coalesced write). No histogram, no atomics — this
//     isolates the pure data-movement cost of "materialize coalesced order". ───
#[cfg(any(feature = "rocm", feature = "cuda"))]
#[cube(launch_unchecked)]
fn reorder_bins(
    resident_bins: &Array<u8>,
    leaf_rows: &Array<u32>,
    num_data: usize,
    scratch: &mut Array<u8>,
) {
    let f = CUBE_POS_X as usize;
    let p = CUBE_POS_Y as usize;
    let np = CUBE_COUNT_Y as usize;
    let cd = CUBE_DIM as usize;
    let r = leaf_rows.len();
    let col = f * num_data;
    let out_col = f * r;
    let stride = np * cd;
    let mut k = p * cd + UNIT_POS as usize;
    while k < r {
        scratch[out_col + k] = resident_bins[col + leaf_rows[k] as usize];
        k += stride;
    }
}

// ─── COAL_OVER_REORDERED: the coalesced-build kernel, but over the reordered `scratch` buffer
//     at row count `r` (the leaf's row count), col stride = r (not num_data). Sequential read
//     `scratch[out_col + k]` — the coalesced ceiling, realized on the ACTUAL leaf row count so
//     it's directly summable with REORDER's time for the net comparison. ───
#[cfg(any(feature = "rocm", feature = "cuda"))]
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn build_coal_over_reordered(
    scratch: &Array<u8>,
    ord_g: &Array<f32>,
    ord_h: &Array<f32>,
    slot_off: &Array<u32>,
    r: usize,
    feat_len: u32,
    out: &mut Array<Atomic<u64>>,
) {
    let f = CUBE_POS_X as usize;
    let p = CUBE_POS_Y as usize;
    let np = CUBE_COUNT_Y as usize;
    let cd = CUBE_DIM as usize;
    let fl = feat_len as usize;
    let base = slot_off[f] as usize;
    let out_col = f * r;
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
        let bin = u32::cast_from(scratch[out_col + k]) as usize; // COALESCED
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

/// The whole probe, generic over the CubeCL runtime — one code path serves the hip (APU) and
/// cuda (real NVIDIA) backends. Prints REAL_ORDER (production baseline) vs REORDER+COAL_OVER
/// (candidate pipeline) medians + Mr/s, and the net verdict.
#[cfg(any(feature = "rocm", feature = "cuda"))]
fn run<R: cubecl::Runtime>(client: cubecl::prelude::ComputeClient<R>) {
    use std::time::Instant;

    const LAUNCHES: usize = 20;
    const REPS: usize = 9;
    const FEATS: usize = 500; // wide shape ⇒ P=1

    let configs: [(usize, usize); 2] = [(250_000, 256), (1_000_000, 256)];
    let pct = |v: &mut Vec<f64>, q: f64| {
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        v[(((v.len() - 1) as f64) * q).round() as usize]
    };

    println!(
        "# spike088 — coalesced-reorder NET cost (FEATS={FEATS}, wide P=1). \
         net = REAL_ORDER - (REORDER + COAL_OVER_REORDERED)."
    );
    for (num_data, num_bin) in configs {
        let feat_len = (2 * num_bin) as u32;
        let slot_len = FEATS * 2 * num_bin;
        let slot_off: Vec<u32> = (0..FEATS as u32).map(|f| f * feat_len).collect();

        let mut bins: Vec<u8> = Vec::with_capacity(FEATS * num_data);
        for f in 0..FEATS {
            for row in 0..num_data {
                let h = (row as u64).wrapping_mul(2_654_435_761).wrapping_add(f as u64 * 97);
                bins.push((h % num_bin as u64) as u8);
            }
        }
        // The production monotone subset — a 50%-selectivity leaf one level below root
        // (stride-2, sorted). This is the row order production actually pays for; the
        // reorder pass would materialize this INTO fully coalesced (0,1,2,...).
        let leaf_rows_real: Vec<u32> = (0..num_data as u32).step_by(2).collect();
        let r_real = leaf_rows_real.len();
        let ord_g: Vec<f32> = (0..r_real).map(|i| (i as f32 * 0.013).sin() * 0.5).collect();
        let ord_h: Vec<f32> = (0..r_real).map(|i| 1.0 + (i % 7) as f32 * 0.01).collect();

        let d_bins = client.create_from_slice(u8::as_bytes(&bins));
        let d_rows_real = client.create_from_slice(u32::as_bytes(&leaf_rows_real));
        let d_g = client.create_from_slice(f32::as_bytes(&ord_g));
        let d_h = client.create_from_slice(f32::as_bytes(&ord_h));
        let d_slot = client.create_from_slice(u32::as_bytes(&slot_off));

        // REAL_ORDER: current production baseline.
        let mut real_samples = Vec::with_capacity(REPS);
        for _ in 0..REPS {
            let out = client.create_from_slice(u64::as_bytes(&vec![0u64; slot_len]));
            let t = Instant::now();
            for _ in 0..LAUNCHES {
                unsafe {
                    build_full::launch_unchecked::<R>(
                        &client,
                        CubeCount::Static(FEATS as u32, 1, 1),
                        CubeDim::new_1d(256),
                        ArrayArg::from_raw_parts(d_bins.clone(), FEATS * num_data),
                        ArrayArg::from_raw_parts(d_rows_real.clone(), r_real),
                        ArrayArg::from_raw_parts(d_g.clone(), r_real),
                        ArrayArg::from_raw_parts(d_h.clone(), r_real),
                        ArrayArg::from_raw_parts(d_slot.clone(), FEATS),
                        num_data,
                        feat_len,
                        ArrayArg::from_raw_parts(out.clone(), slot_len),
                    );
                }
            }
            let _ = client.read_one_unchecked(out);
            real_samples.push(t.elapsed().as_secs_f64() * 1000.0 / LAUNCHES as f64);
        }
        let real_ms = pct(&mut real_samples, 0.5);

        // Candidate pipeline: REORDER (once) then COAL_OVER_REORDERED (once), summed per rep so
        // the pairing captures any launch-to-launch jitter correlation.
        let scratch_len = FEATS * r_real;
        let mut reorder_samples = Vec::with_capacity(REPS);
        let mut coal_samples = Vec::with_capacity(REPS);
        let mut pipeline_samples = Vec::with_capacity(REPS);
        for _ in 0..REPS {
            let scratch = client.create_from_slice(&vec![0u8; scratch_len]);
            let out = client.create_from_slice(u64::as_bytes(&vec![0u64; slot_len]));

            let t0 = Instant::now();
            for _ in 0..LAUNCHES {
                unsafe {
                    reorder_bins::launch_unchecked::<R>(
                        &client,
                        CubeCount::Static(FEATS as u32, 1, 1),
                        CubeDim::new_1d(256),
                        ArrayArg::from_raw_parts(d_bins.clone(), FEATS * num_data),
                        ArrayArg::from_raw_parts(d_rows_real.clone(), r_real),
                        num_data,
                        ArrayArg::from_raw_parts(scratch.clone(), scratch_len),
                    );
                }
            }
            let _ = client.read_one_unchecked(scratch.clone());
            let reorder_ms = t0.elapsed().as_secs_f64() * 1000.0 / LAUNCHES as f64;

            let t1 = Instant::now();
            for _ in 0..LAUNCHES {
                unsafe {
                    build_coal_over_reordered::launch_unchecked::<R>(
                        &client,
                        CubeCount::Static(FEATS as u32, 1, 1),
                        CubeDim::new_1d(256),
                        ArrayArg::from_raw_parts(scratch.clone(), scratch_len),
                        ArrayArg::from_raw_parts(d_g.clone(), r_real),
                        ArrayArg::from_raw_parts(d_h.clone(), r_real),
                        ArrayArg::from_raw_parts(d_slot.clone(), FEATS),
                        r_real,
                        feat_len,
                        ArrayArg::from_raw_parts(out.clone(), slot_len),
                    );
                }
            }
            let _ = client.read_one_unchecked(out);
            let coal_ms = t1.elapsed().as_secs_f64() * 1000.0 / LAUNCHES as f64;

            reorder_samples.push(reorder_ms);
            coal_samples.push(coal_ms);
            pipeline_samples.push(reorder_ms + coal_ms);
        }
        let reorder_ms = pct(&mut reorder_samples, 0.5);
        let coal_ms = pct(&mut coal_samples, 0.5);
        let pipeline_ms = pct(&mut pipeline_samples, 0.5);

        let net_ms = real_ms - pipeline_ms;
        let mrs = |ms: f64| (FEATS as f64 * r_real as f64) / (ms * 1e3);

        println!("\n## {FEATS} feat × {num_data} rows (leaf r={r_real}) × {num_bin} bins  (P=1, wide)");
        println!(
            "#   REAL_ORDER (baseline)      {real_ms:8.3} ms   ({:.0} Mr/s)",
            mrs(real_ms)
        );
        println!(
            "#   REORDER (sort/scatter)     {reorder_ms:8.3} ms   ({:.0} Mr/s)"
        , mrs(reorder_ms));
        println!(
            "#   COAL_OVER_REORDERED        {coal_ms:8.3} ms   ({:.0} Mr/s)"
        , mrs(coal_ms));
        println!(
            "#   CANDIDATE (reorder+coal)   {pipeline_ms:8.3} ms   ({:.0} Mr/s)"
        , mrs(pipeline_ms));
        println!(
            "#   >>> NET = REAL_ORDER - CANDIDATE = {net_ms:+.3} ms/split ({:+.1}% of baseline). {}",
            100.0 * net_ms / real_ms,
            if net_ms > 0.0 { "NET-POSITIVE — reorder pipeline is FASTER." } else { "NET-NEGATIVE — reorder overhead eats the saving." }
        );
    }
    println!("\n# READING:");
    println!("#  NET > 0  ⇒ reorder-before-build beats production as-is on real CUDA — a genuine");
    println!("#             lever (bit-exact by construction: permutation-invariant u64 histogram).");
    println!("#  NET <= 0 ⇒ the sort/scatter pass costs more than the coalescing saving — no lever,");
    println!("#             confirms spike-030/031's 'read-once-unamortizable' verdict on real CUDA.");
    println!("#  Bounded regardless by spike-054's ~2x architectural floor + Amdahl (build 27-43%, spike-084).");
}

#[cfg(all(feature = "rocm", not(feature = "cuda")))]
fn main() {
    run(lgbm_compute::runtime::rocm_client());
}

#[cfg(feature = "cuda")]
fn main() {
    run(lgbm_compute::runtime::cuda_client());
}
