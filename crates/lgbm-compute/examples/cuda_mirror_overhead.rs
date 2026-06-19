//! Warmed-up before/after micro-bench for the CUDA-mirror histogram kernel
//! (`construct_histograms_cuda_mirror_on` vs `construct_histograms_cuda_mirror_resident_on`
//! in `lgbm_compute::kernels::histogram`) — quick task 260619-mwr (MWR-03).
//!
//! Two overhead levers, measured separately so each is attributable:
//!
//!  (MWR-01) launch_unchecked — drops in-kernel bounds-check codegen. BOTH launchers
//!    now use `::launch_unchecked` (the V5 validation before upload discharges the
//!    contract), so the launch-codegen win is baked into both timed paths; the manual
//!    rationale (no per-access bounds branch in the hot scatter loop) plus the residual
//!    per-call vs resident delta attribute it. We can't A/B `launch` vs `launch_unchecked`
//!    in one binary (it's a comptime kernel attribute), so the launch win is reported
//!    qualitatively + via the overall before/after.
//!
//!  (MWR-02) upload-once — the dominant NON-COMPUTE overhead. The per-call launcher
//!    `create_from_slice`s the FULL feature-major bin buffer (num_features*num_data*4
//!    bytes) on EVERY call; the resident launcher uploads it ONCE before the timed loop
//!    and re-uploads only the small per-leaf data_indices + per-iter grad/hess. The
//!    (per-call) vs (resident) median is the TRANSFER-overhead win, with MB-per-call
//!    quantified for each path.
//!
//! Honors the load-bearing warm-vs-cold rule (spike-findings SKILL): >=2 discarded
//! warm-up launches per variant before timing, MEDIAN of >=5 timed launches, and a
//! device sync (read-back of the result) inside each timed call so the timer captures
//! real device work, not just queue submission.
//!
//! This is a MANUAL rocm bench — NOT wired into any test gate.
//!
//! Run: cargo run --release -p lgbm-compute --features rocm --example cuda_mirror_overhead

#[cfg(not(feature = "rocm"))]
fn main() {
    eprintln!("this micro-bench requires --features rocm (gfx1100). Re-run with it.");
}

#[cfg(feature = "rocm")]
fn main() {
    use cubecl::prelude::*;
    use lgbm_compute::kernels::histogram::{
        construct_histograms_cuda_mirror_on, construct_histograms_cuda_mirror_resident_on,
    };
    use lgbm_compute::runtime::rocm_client;
    use std::time::Instant;

    // Realistic corpus per the plan's <measurement_requirement>: 50 features, ~200k rows.
    const NUM_DATA: usize = 200_000;
    const FEATS: usize = 50;
    const BIN_SWEEP: [u32; 3] = [16, 64, 256];
    const WARMUP: usize = 3; // >= 2 discarded warm-up launches
    const TIMED: usize = 7; // median of >= 5 timed launches

    let client = rocm_client();

    // Median helper (sort, take middle).
    let median = |xs: &mut [f64]| -> f64 {
        xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        xs[xs.len() / 2]
    };

    println!("# cuda-mirror overhead micro-bench  num_data={NUM_DATA} feats={FEATS}");
    println!("# warmup={WARMUP} discarded, median of {TIMED} timed launches per variant.");
    println!("# per-call = re-uploads the full feature-major bin buffer EVERY call;");
    println!("# resident = uploads the bin buffer ONCE, re-uploads only per-leaf data.");
    println!();
    println!(
        "{:>5} | {:>14} | {:>14} | {:>8} | {:>13} | {:>13}",
        "bins", "per-call ms", "resident ms", "speedup", "per-call MB", "resident MB"
    );
    println!("{:-<6}+{:-<16}+{:-<16}+{:-<10}+{:-<15}+{:-<15}", "", "", "", "", "", "");

    for &num_bin in &BIN_SWEEP {
        // Deterministic, well-spread feature-major bins (no RNG; same hash style as the
        // parity test / gpu_row_partition example).
        let mut resident: Vec<u32> = Vec::with_capacity(FEATS * NUM_DATA);
        for f in 0..FEATS {
            for row in 0..NUM_DATA {
                let h = (row as u64)
                    .wrapping_mul(2_654_435_761)
                    .wrapping_add((f as u64).wrapping_mul(40_503).wrapping_add(0x9E37_79B9));
                resident.push((h % num_bin as u64) as u32);
            }
        }
        // Full-corpus grad/hess (gathered in-kernel by data_index).
        let grad: Vec<f32> = (0..NUM_DATA).map(|i| (i as f32) * 0.001 - 1.0).collect();
        let hess: Vec<f32> = (0..NUM_DATA).map(|i| 1.0 + ((i % 7) as f32) * 0.1).collect();

        // LARGE leaf subset: ~half the rows (every 2nd) so the row loop dominates.
        let data_indices: Vec<u32> = (0..NUM_DATA as u32).step_by(2).collect();
        let rows = data_indices.len();

        let slot_off: Vec<usize> = (0..FEATS).map(|f| f * 2 * num_bin as usize).collect();
        let slot_len = FEATS * 2 * num_bin as usize;

        // --- per-call path: full re-upload every call (the BEFORE / transfer-heavy). ---
        let per_call = || {
            let out = construct_histograms_cuda_mirror_on(
                &client,
                &resident,
                NUM_DATA,
                FEATS,
                &data_indices,
                &grad,
                &hess,
                &slot_off,
                slot_len,
                num_bin,
            )
            .unwrap();
            // read-back already forced the device sync inside the launcher.
            out[0]
        };
        for _ in 0..WARMUP {
            let _ = per_call();
        }
        let mut pc_ms: Vec<f64> = Vec::with_capacity(TIMED);
        for _ in 0..TIMED {
            let t = Instant::now();
            let _ = per_call();
            pc_ms.push(t.elapsed().as_secs_f64() * 1e3);
        }
        let pc_med = median(&mut pc_ms);

        // --- resident path: upload the bin buffer ONCE, then time per-leaf calls. ---
        // The big feature-major buffer moves to the device exactly once here (outside the
        // timed loop), mirroring CUDA's resident model.
        let resident_handle = client.create_from_slice(u32::as_bytes(&resident));
        let resident_call = || {
            let out = construct_histograms_cuda_mirror_resident_on(
                &client,
                resident_handle.clone(),
                NUM_DATA,
                FEATS,
                &data_indices,
                &grad,
                &hess,
                &slot_off,
                slot_len,
                num_bin,
            )
            .unwrap();
            out[0]
        };
        for _ in 0..WARMUP {
            let _ = resident_call();
        }
        let mut rs_ms: Vec<f64> = Vec::with_capacity(TIMED);
        for _ in 0..TIMED {
            let t = Instant::now();
            let _ = resident_call();
            rs_ms.push(t.elapsed().as_secs_f64() * 1e3);
        }
        let rs_med = median(&mut rs_ms);

        // MB transferred per call:
        //   per-call: full bins (FEATS*NUM_DATA*4) + idx (rows*4) + grad+hess (2*NUM_DATA*4)
        //   resident: idx (rows*4) + grad+hess (2*NUM_DATA*4) only (bins already resident)
        let bins_bytes = (FEATS * NUM_DATA * 4) as f64;
        let perleaf_bytes = (rows * 4 + 2 * NUM_DATA * 4) as f64;
        let pc_mb = (bins_bytes + perleaf_bytes) / 1_048_576.0;
        let rs_mb = perleaf_bytes / 1_048_576.0;

        println!(
            "{:>5} | {:>14.3} | {:>14.3} | {:>7.2}x | {:>13.1} | {:>13.1}",
            num_bin,
            pc_med,
            rs_med,
            pc_med / rs_med,
            pc_mb,
            rs_mb
        );
    }

    println!();
    println!("# speedup = per-call median / resident median (>1.0 = upload-once wins).");
    println!("# per-call MB includes the full {FEATS}x{NUM_DATA} bin re-upload each call;");
    println!("# resident MB is only the per-leaf data_indices + full-corpus grad/hess.");
    println!("# launch_unchecked (MWR-01) is baked into BOTH paths — the per-call vs");
    println!("# resident delta isolates the TRANSFER-overhead win (MWR-02); the launch-");
    println!("# codegen win is qualitative (no in-kernel bounds branch in the scatter loop).");
}
