//! Spike 017 GPU micro-benchmark: does PER-WARP LDS sub-histogram REPLICATION cut
//! the f32-atomic histogram BUILD device-time at the WIDE shape (P=1 regime)?
//!
//! The shipped resident LDS build (`construct_leaf_hist_resident_lds_kernel` /
//! `construct_hist_kernel_lds_f32`) gives each CUBE ONE sub-histogram in LDS, so all
//! 256 threads (8 wave32 warps) atomic-add into the SAME 2 KiB sub-hist — 256-way
//! intra-cube contention. At the wide shape (×500 feat) row-partition is inactive
//! (`target_cubes / 500 -> P=1`), so adding more cubes is not an option; the only
//! remaining contention lever is to REPLICATE the sub-histogram inside the cube.
//!
//! This bench parameterizes a replication factor `R`: each cube holds R sub-hists in
//! LDS (R * 2*num_bin f32 cells), warp `w` accumulates into replica `(w % R)`, then the
//! cube sums the R replicas and issues one global atomic per cell. **R = 1 is
//! byte-identical to the shipped single-sub-hist kernel**, so sweeping R cleanly
//! isolates the contention-replication effect. LDS scales with R (comptime-sized), so
//! occupancy honestly reflects the LDS-pressure tradeoff (research finding #1 / SC20).
//!
//! Mechanism under test: p93 showed warp-AGGREGATION (ballot+shuffle) is NULL because
//! at 256 bins a 32-lane wave hits ~30 distinct bins (nothing to amortize). Replication
//! is the COMPLEMENTARY lever — it does not pre-reduce collisions, it spreads the
//! threads-per-atomic from 256 to 256/R. If BUILD is contention-bound, R>1 wins; if it
//! is scattered-read-latency bound (spike-006: 234 Mreads/s), R>1 is null. That is the
//! open question this spike settles on the device-time proxy.
//!
//! Run: cargo run --release --features rocm --example gpu_lds_replication

#[cfg(feature = "rocm")]
use cubecl::prelude::*;

/// LDS sub-hist cap PER REPLICA: 512 f32 cells = 256 bins (grad+hess interleaved) = 2 KiB.
#[cfg(feature = "rocm")]
const HIST_LDS_MAX: usize = 512;

/// Per-warp-replicated LDS resident build. One cube per feature (`CubeCount=(feats,1)`).
/// `R` (comptime) sub-histograms live in LDS; warp `w` accumulates into replica `w % R`;
/// the cube then sums all R replicas per cell and issues ONE global atomic. R=1 reduces
/// exactly to the shipped single-sub-hist LDS kernel.
#[cfg(feature = "rocm")]
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn build_repl(
    resident_bins: &Array<u32>,
    leaf_rows: &Array<u32>,
    ord_g: &Array<f32>,
    ord_h: &Array<f32>,
    slot_off: &Array<u32>, // start cell per feature (length num_features)
    num_data: usize,       // resident column stride
    feat_len: u32,         // 2*num_bin (active LDS cells per replica)
    out: &mut Array<Atomic<f32>>,
    #[comptime] replicas: usize,
) {
    let f = CUBE_POS_X as usize;
    let cd = CUBE_DIM as usize; // 256
    let fl = feat_len as usize;
    let r = ord_g.len();
    let gbase = slot_off[f] as usize;
    let col = f * num_data;
    let nrep = replicas;

    // R replicas, each `fl` active cells, packed densely (replica rr at [rr*fl, rr*fl+fl)).
    // Comptime size = R * HIST_LDS_MAX (max 256-bin slot) ≥ R * fl.
    let sub = SharedMemory::<Atomic<f32>>::new(replicas * HIST_LDS_MAX);

    // 1. zero the R replicas' active cells (total = nrep*fl), strided across the cube.
    let total = nrep * fl;
    let mut c = UNIT_POS as usize;
    while c < total {
        sub[c].store(0.0f32);
        c += cd;
    }
    sync_cube();

    // 2. each warp folds into its OWN replica: replica = (warp_id) % R. base = replica*fl.
    let replica = (UNIT_POS as usize / PLANE_DIM as usize) % nrep;
    let rbase = replica * fl;
    let mut k = UNIT_POS as usize;
    while k < r {
        let row = leaf_rows[k] as usize;
        let bin = resident_bins[col + row] as usize;
        let ti = rbase + bin * 2;
        sub[ti].fetch_add(ord_g[k]);
        sub[ti + 1].fetch_add(ord_h[k]);
        k += cd;
    }
    sync_cube();

    // 3. merge: for each active cell m, sum across the R replicas (ascending, deterministic
    //    order within the cube) and issue ONE global atomic.
    let mut m = UNIT_POS as usize;
    while m < fl {
        let mut acc = 0.0f32;
        let mut rr = 0usize;
        while rr < nrep {
            acc += sub[rr * fl + m].load();
            rr += 1;
        }
        out[gbase + m].fetch_add(acc);
        m += cd;
    }
}

#[cfg(not(feature = "rocm"))]
fn main() {
    eprintln!("this micro-bench requires --features rocm (gfx1100/APU). Re-run with it.");
}

#[cfg(feature = "rocm")]
fn main() {
    use lgbm_compute::runtime::rocm_client;
    use std::time::Instant;

    // Wide-shape-representative leaf: many features (≈ fill the 8-CU APU at ~8 wkgrp/CU
    // ⇒ ~64 cubes) so P=1-per-feature is the real regime, each cube under full
    // intra-cube contention over a large leaf.
    const NUM_DATA: usize = 200_000; // leaf rows per feature
    const FEATS: usize = 64; // ≈ 64 cubes (one per feature, P=1)
    const LAUNCHES: usize = 20;
    const CUBE_DIM: u32 = 256;
    let bins_sweep: [usize; 3] = [16, 64, 256];
    let replicas_sweep: [u32; 4] = [1, 2, 4, 8];

    let client = rocm_client();

    // Scattered leaf rows + per-row grad/hess (shared across features).
    let leaf_rows: Vec<u32> = (0..NUM_DATA as u32)
        .map(|i| (i.wrapping_mul(2_654_435_761)) % NUM_DATA as u32)
        .collect();
    let ord_g: Vec<f32> = (0..NUM_DATA).map(|i| (i % 13) as f32 * 0.1).collect();
    let ord_h: Vec<f32> = (0..NUM_DATA).map(|i| 1.0 + (i % 7) as f32 * 0.01).collect();
    let r = leaf_rows.len();

    let d_rows = client.create_from_slice(u32::as_bytes(&leaf_rows));
    let d_g = client.create_from_slice(f32::as_bytes(&ord_g));
    let d_h = client.create_from_slice(f32::as_bytes(&ord_h));

    println!("# gpu per-warp LDS replication microbench  leaf_rows={NUM_DATA} feats={FEATS} launches={LAUNCHES}");
    println!("# R=1 == shipped single-sub-hist LDS kernel. one cube/feature (P=1, wide regime). APU=8 CU.");

    // CPU f64 anchor for one feature (deterministic, ascending) — the parity reference.
    let cpu_anchor = |bins: &[u32], num_bin: usize| -> Vec<f64> {
        let mut h = vec![0.0f64; 2 * num_bin];
        for k in 0..r {
            let b = bins[leaf_rows[k] as usize] as usize;
            h[b * 2] += ord_g[k] as f64;
            h[b * 2 + 1] += ord_h[k] as f64;
        }
        h
    };

    for &num_bin in &bins_sweep {
        let feat_len = (2 * num_bin) as u32;
        let slot_len = FEATS * 2 * num_bin;
        let slot_off: Vec<u32> = (0..FEATS as u32).map(|f| f * feat_len).collect();
        let d_slot = client.create_from_slice(u32::as_bytes(&slot_off));

        // feature-major resident bins (FEATS × NUM_DATA), hashed (uniform over num_bin).
        let mut bins: Vec<u32> = Vec::with_capacity(FEATS * NUM_DATA);
        for f in 0..FEATS {
            for rr in 0..NUM_DATA {
                let h = (rr as u64).wrapping_mul(2_654_435_761).wrapping_add(f as u64 * 97);
                bins.push((h % num_bin as u64) as u32);
            }
        }
        let d_bins = client.create_from_slice(u32::as_bytes(&bins));
        let reads_per_launch = (FEATS * r) as f64;

        // One clean launch into a freshly-zeroed out; returns the f32 histogram.
        let verify = |rep: u32| -> Vec<f32> {
            let out = client.create_from_slice(f32::as_bytes(&vec![0.0f32; slot_len]));
            unsafe {
                build_repl::launch_unchecked(
                    &client,
                    CubeCount::Static(FEATS as u32, 1, 1),
                    CubeDim::new_1d(CUBE_DIM),
                    ArrayArg::from_raw_parts(d_bins.clone(), FEATS * NUM_DATA),
                    ArrayArg::from_raw_parts(d_rows.clone(), r),
                    ArrayArg::from_raw_parts(d_g.clone(), r),
                    ArrayArg::from_raw_parts(d_h.clone(), r),
                    ArrayArg::from_raw_parts(d_slot.clone(), FEATS),
                    NUM_DATA,
                    feat_len,
                    ArrayArg::from_raw_parts(out.clone(), slot_len),
                    rep as usize,
                );
            }
            f32::from_bytes(&client.read_one_unchecked(out)).to_vec()
        };

        let bench = |rep: u32| -> std::time::Duration {
            let out = client.create_from_slice(f32::as_bytes(&vec![0.0f32; slot_len]));
            let t = Instant::now();
            for _ in 0..LAUNCHES {
                unsafe {
                    build_repl::launch_unchecked(
                        &client,
                        CubeCount::Static(FEATS as u32, 1, 1),
                        CubeDim::new_1d(CUBE_DIM),
                        ArrayArg::from_raw_parts(d_bins.clone(), FEATS * NUM_DATA),
                        ArrayArg::from_raw_parts(d_rows.clone(), r),
                        ArrayArg::from_raw_parts(d_g.clone(), r),
                        ArrayArg::from_raw_parts(d_h.clone(), r),
                        ArrayArg::from_raw_parts(d_slot.clone(), FEATS),
                        NUM_DATA,
                        feat_len,
                        ArrayArg::from_raw_parts(out.clone(), slot_len),
                        rep as usize,
                    );
                }
            }
            let _ = client.read_one_unchecked(out);
            t.elapsed()
        };

        // Correctness: feature 0's sub-histogram vs the CPU f64 anchor, for R=1 and R=8.
        let anchor = cpu_anchor(&bins[0..NUM_DATA], num_bin);
        let rel_to_anchor = |v: &[f32]| -> f64 {
            v[0..2 * num_bin]
                .iter()
                .zip(anchor.iter())
                .map(|(a, b)| (*a as f64 - *b).abs() / b.abs().max(1.0))
                .fold(0.0f64, f64::max)
        };
        let ref1 = verify(1);
        let r1_rel = rel_to_anchor(&ref1);
        println!("\n## num_bin={num_bin}  (feat_len={feat_len}, LDS/replica={} B)", feat_len as usize * 4);
        println!("# correctness vs CPU f64 anchor + vs R=1 (f32-atomic reorder noise expected):");
        for &rep in &replicas_sweep {
            let v = verify(rep);
            let mut max_rel_vs1 = 0.0f64;
            for (a, b) in ref1.iter().zip(v.iter()) {
                let denom = (*a as f64).abs().max(1.0);
                max_rel_vs1 = max_rel_vs1.max((*a as f64 - *b as f64).abs() / denom);
            }
            let lds = rep as usize * feat_len as usize * 4;
            println!(
                "  R={rep}: LDS={lds:>5}B  rel_vs_anchor={:.2e}  rel_vs_R1={max_rel_vs1:.2e}",
                rel_to_anchor(&v)
            );
        }
        let _ = r1_rel;

        // Warm up the full R sweep (discard cold-start), then INTERLEAVED sampling:
        // each rep times all R back-to-back so they share the same thermal/load window;
        // report median + p25/p75 and speedup as median(R1)/median(R) — NOT vs a cold
        // round-1 baseline (CONVENTIONS: cold round-1 overstates). Run the whole process
        // ≥2 times to check sign-stability vs noise.
        for _ in 0..2 {
            for &rep in &replicas_sweep {
                let _ = bench(rep);
            }
        }
        const REPS: usize = 11;
        let mut samples: [Vec<f64>; 4] = [vec![], vec![], vec![], vec![]];
        for _ in 0..REPS {
            for (i, &rep) in replicas_sweep.iter().enumerate() {
                let ms = bench(rep).as_secs_f64() * 1e3;
                samples[i].push(ms);
            }
        }
        let pct = |v: &mut Vec<f64>, q: f64| -> f64 {
            v.sort_by(|a, b| a.partial_cmp(b).unwrap());
            v[((v.len() as f64 - 1.0) * q).round() as usize]
        };
        let mut med = [0.0f64; 4];
        let mut p25 = [0.0f64; 4];
        let mut p75 = [0.0f64; 4];
        for i in 0..4 {
            p25[i] = pct(&mut samples[i], 0.25);
            p75[i] = pct(&mut samples[i], 0.75);
            med[i] = pct(&mut samples[i], 0.50);
        }
        println!("# wall-clock median[p25..p75] over {REPS} interleaved reps ({LAUNCHES} launches each):");
        for (i, &rep) in replicas_sweep.iter().enumerate() {
            let mrs = reads_per_launch * LAUNCHES as f64 / (med[i] / 1e3) / 1e6;
            let spd = med[0] / med[i];
            // spread-separated win requires R's p75 below R1's p25.
            let sep = if rep != 1 && p75[i] < p25[0] { " SEP-WIN" } else { "" };
            println!(
                "  R{rep}: {:.0}ms[{:.0}..{:.0}]  {mrs:.0}Mr/s  {spd:.2}x{sep}",
                med[i], p25[i], p75[i]
            );
        }
    }
    println!("\n# speedup = median(R1)/median(R). SEP-WIN = R's p75 < R1's p25 (spread-separated).");
    println!("# Contention-bound ⇒ R>1 SEP-WINs (largest at low bins); latency/occupancy-bound ⇒ null/regress.");
}
