//! A/B micro-bench: the CUDA-mirror resident histogram kernel
//! (`construct_histograms_cuda_mirror_resident_on`) vs the WIRED LDS resident build
//! kernel (`build_leaf_histograms_resident_f32_on`) on the local gfx1100 — quick task
//! 260619-ngo (NGO-01).
//!
//! WHAT THIS DECIDES: the follow-up wiring disposition for the CUDA-mirror resident
//! kernel (default / gated-opt-in / leave-as-primitive). The mirror was landed as a
//! faithful rocm-gated primitive (260619-j9t / 260619-mwr); the LDS path is the
//! production-wired build (260609-fw1). This bench produces the real A/B numbers that
//! choose between them. It is MEASUREMENT ONLY — it does NOT wire either kernel into
//! lib.rs / the learner / the wired path, and it does NOT touch the CPU f64 anchor.
//!
//! FAIRNESS NOTES (the A/B compares like with like):
//!  - Both launchers are fed the SAME resident feature-major bin `Handle` (uploaded
//!    ONCE, OUTSIDE both timed loops — the upload-once win is settled in 260619-mwr and
//!    is EXCLUDED from this comparison), the SAME leaf rows, and the SAME plain
//!    sentinel-free `slot_off` (each launcher builds its own sentinel internally). They
//!    differ ONLY in their native grad/hess input form.
//!  - The mirror gathers grad/hess IN-KERNEL from full-corpus arrays (`grad[data_index]`),
//!    so there is no host-side gather to attribute — its number is intrinsic.
//!  - The LDS launcher `build_leaf_histograms_resident_f32_on` ALSO takes FULL-CORPUS
//!    grad/hess and gathers the leaf-length `ord_g`/`ord_h` HOST-SIDE *INSIDE itself*
//!    (histogram.rs `resident_raw_build_into`, `ord_g[k]=grad[leaf_rows[k]]`) — exactly
//!    as the wired production path passes them (learner.rs:1767, full-corpus). So both
//!    launchers receive full-corpus arrays; the host gather production pays is INTERNAL
//!    to the LDS launcher, not a caller step.
//!    [DEVIATION — Rule 3, 260619-ngo] The plan's interface contract said the caller
//!    host-gathers ord_g/ord_h to leaf length BEFORE the LDS call; the real launcher does
//!    that gather internally and a leaf-length array would index out of bounds. We honor
//!    the plan's INTENT (report production cost + a gather-excluded kernel-only number)
//!    by: timing the full launcher as lds_incl (gather included — production's real cost),
//!    timing the host gather alone, and deriving lds_excl = lds_incl - gather (kernel +
//!    uploads, the host gather production could independently remove). No kernel/lib edit.
//!  - So we report TWO LDS numbers:
//!      * lds_incl_ms — the full launcher (internal host gather INCLUDED) = production cost,
//!      * lds_excl_ms — lds_incl minus the separately-timed host gather (kernel-only view).
//!
//! WARM-VS-COLD (load-bearing, spike-findings SKILL): WARMUP>=2 discarded launches per
//! variant, MEDIAN of TIMED>=5, a device sync (result read-back) inside every timed call,
//! and the bench is meant to be run from a few separate process invocations to check
//! warmup drift. Each variant warms up and is medianed SEPARATELY.
//!
//! SANITY ASSERT (same-input correctness, NOT the parity gate): once per (bin, leaf)
//! cell the mirror RAW histogram and the LDS RAW histogram are asserted to agree within
//! the f32-atomic envelope (ABS 5e-6 / REL 1e-5, the `rocm_cuda_mirror.rs::assert_close`
//! tolerance). This pins the two kernels ONLY to each other as a same-input check so the
//! timing comparison is between CORRECT, equivalent computations. The REAL parity gate
//! stays the CPU f64 anchor — already covered by the existing `rocm_cuda_mirror.rs` /
//! `rocm_parallel_histogram.rs` tests; this bench does NOT re-assert against it.
//!
//! This is a MANUAL rocm bench — NOT a test gate.
//!
//! Run: cargo run --release -p lgbm-compute --features rocm --example mirror_vs_lds

#[cfg(not(feature = "rocm"))]
fn main() {
    eprintln!("this micro-bench requires --features rocm (gfx1100). Re-run with it.");
}

#[cfg(feature = "rocm")]
fn main() {
    use cubecl::prelude::*;
    use lgbm_compute::kernels::histogram::{
        build_leaf_histograms_resident_f32_on, construct_histograms_cuda_mirror_resident_on,
    };
    use lgbm_compute::runtime::rocm_client;
    use std::time::Instant;

    // Realistic corpus per the measurement design: 50 features, ~200k rows, the bin sweep.
    const NUM_DATA: usize = 200_000;
    const FEATS: usize = 50;
    const BIN_SWEEP: [u32; 3] = [16, 64, 256];
    const WARMUP: usize = 3; // >= 2 discarded warm-up launches
    const TIMED: usize = 7; // median of >= 5 timed launches

    let client = rocm_client();

    // Median helper (sort, take middle) — same as cuda_mirror_overhead.rs.
    let median = |xs: &mut [f64]| -> f64 {
        xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        xs[xs.len() / 2]
    };

    // Same-input correctness check (f32-atomic envelope, NOT the parity gate). Panics on
    // the first diverging cell with its index + both values.
    let assert_same_input = |mirror: &[f64], lds: &[f64], cell: &str| {
        const ABS: f64 = 5e-6;
        const REL: f64 = 1e-5;
        assert_eq!(mirror.len(), lds.len(), "{cell}: RAW histogram length mismatch");
        for (i, (m, l)) in mirror.iter().zip(lds).enumerate() {
            let diff = (m - l).abs();
            let tol = ABS + REL * m.abs();
            assert!(
                diff <= tol,
                "{cell}: cell {i} diverged beyond the f32-atomic envelope \
                 — mirror {m} vs lds {l} (|diff| {diff} > tol {tol})"
            );
        }
    };

    println!("# mirror-vs-LDS resident-histogram A/B  num_data={NUM_DATA} feats={FEATS}");
    println!("# warmup={WARMUP} discarded, median of {TIMED} timed launches per variant.");
    println!("# resident bin buffer uploaded ONCE per cell, OUTSIDE both timed loops (excluded).");
    println!("# lds_incl = full LDS launcher (its INTERNAL host ord_g/ord_h gather included) = prod cost;");
    println!("# lds_excl = lds_incl minus the separately-timed host gather (kernel+uploads only).");
    println!("# speedup       = lds_incl / mirror  (>1.0 => mirror beats the production LDS path)");
    println!("# speedup_kernel= lds_excl / mirror  (>1.0 => mirror beats the LDS kernel alone)");
    println!();
    println!(
        "{:>9} | {:>5} | {:>11} | {:>11} | {:>11} | {:>9} | {:>9} | {:>13}",
        "leaf", "bins", "mirror_ms", "lds_incl_ms", "lds_excl_ms", "gather_ms", "speedup",
        "speedup_kernel"
    );
    println!(
        "{:-<10}+{:-<7}+{:-<13}+{:-<13}+{:-<13}+{:-<11}+{:-<11}+{:-<15}",
        "", "", "", "", "", "", "", ""
    );

    // Full-corpus grad/hess (the mirror's native form; gathered in-kernel by data_index).
    // Same deterministic generation as cuda_mirror_overhead.rs — no RNG.
    let grad: Vec<f32> = (0..NUM_DATA).map(|i| (i as f32) * 0.001 - 1.0).collect();
    let hess: Vec<f32> = (0..NUM_DATA).map(|i| 1.0 + ((i % 7) as f32) * 0.1).collect();

    // TWO leaf sizes: a LARGE leaf (all rows ~= full corpus) and a MID leaf (~50k rows).
    let large_leaf: Vec<u32> = (0..NUM_DATA as u32).collect();
    let mid_leaf: Vec<u32> = (0..NUM_DATA as u32).step_by(4).collect();
    let leaves: [(&str, &Vec<u32>); 2] =
        [("large", &large_leaf), ("mid~50k", &mid_leaf)];

    for &num_bin in &BIN_SWEEP {
        // Deterministic, well-spread feature-major bins (no RNG; same hash style as
        // cuda_mirror_overhead.rs / the parity test). resident[f*NUM_DATA + row].
        let mut resident: Vec<u32> = Vec::with_capacity(FEATS * NUM_DATA);
        for f in 0..FEATS {
            for row in 0..NUM_DATA {
                let h = (row as u64)
                    .wrapping_mul(2_654_435_761)
                    .wrapping_add((f as u64).wrapping_mul(40_503).wrapping_add(0x9E37_79B9));
                resident.push((h % num_bin as u64) as u32);
            }
        }

        // Plain sentinel-free slot_off (num_features entries) — fed to BOTH launchers.
        let slot_off: Vec<usize> = (0..FEATS).map(|f| f * 2 * num_bin as usize).collect();
        let slot_len = FEATS * 2 * num_bin as usize;

        // Upload the big feature-major bin buffer to the device ONCE per cell. Both timed
        // loops reuse this single Handle (cloned per call); the upload is EXCLUDED from the
        // comparison (settled in 260619-mwr).
        let resident_handle = client.create_from_slice(u32::as_bytes(&resident));

        for (leaf_name, leaf_rows) in leaves.iter() {
            let leaf_rows: &[u32] = leaf_rows.as_slice();

            // ---- (3) MIRROR: in-kernel gather is intrinsic — no gather-excluded variant.
            let mirror_call = || {
                let out = construct_histograms_cuda_mirror_resident_on(
                    &client,
                    resident_handle.clone(),
                    NUM_DATA,
                    FEATS,
                    leaf_rows,
                    &grad,
                    &hess,
                    &slot_off,
                    slot_len,
                    num_bin,
                )
                .unwrap();
                out // returned for the sanity check; read of out[0] forced sync inside.
            };
            for _ in 0..WARMUP {
                let _ = mirror_call();
            }
            let mut mirror_ms: Vec<f64> = Vec::with_capacity(TIMED);
            let mut mirror_hist = Vec::new();
            for _ in 0..TIMED {
                let t = Instant::now();
                let out = mirror_call();
                mirror_ms.push(t.elapsed().as_secs_f64() * 1e3);
                mirror_hist = out;
            }
            let mirror_med = median(&mut mirror_ms);

            // ---- (1) LDS gather-INCLUDED (production cost): the FULL launcher, fed
            //          FULL-CORPUS grad/hess exactly as the wired learner does
            //          (learner.rs:1767). Its internal host gather (resident_raw_build_into)
            //          is part of the timed cost — this is what production pays per leaf.
            let lds_incl_call = || {
                let out = build_leaf_histograms_resident_f32_on(
                    &client,
                    resident_handle.clone(),
                    lgbm_compute::ResidentBinWidth::U32, // quick-260621-qix: u32 buffer
                    FEATS,
                    NUM_DATA,
                    &slot_off,
                    slot_len,
                    leaf_rows,
                    &grad, // FULL-CORPUS — launcher gathers ord_g/ord_h to leaf length itself.
                    &hess,
                )
                .unwrap();
                out
            };
            for _ in 0..WARMUP {
                let _ = lds_incl_call();
            }
            let mut lds_incl_ms: Vec<f64> = Vec::with_capacity(TIMED);
            let mut lds_hist = Vec::new();
            for _ in 0..TIMED {
                let t = Instant::now();
                let out = lds_incl_call();
                lds_incl_ms.push(t.elapsed().as_secs_f64() * 1e3);
                lds_hist = out;
            }
            let lds_incl_med = median(&mut lds_incl_ms);

            // ---- (2) HOST GATHER alone: time the exact gather the launcher does internally
            //          (ord_g[k]=grad[leaf_rows[k]], ord_h likewise). lds_excl is then
            //          lds_incl - gather (kernel + uploads only; the host gather production
            //          could remove independently). black_box keeps the gather from being
            //          optimized away (the result feeds a sink read).
            let gather_call = || {
                let g: Vec<f32> = leaf_rows.iter().map(|&r| grad[r as usize]).collect();
                let h: Vec<f32> = leaf_rows.iter().map(|&r| hess[r as usize]).collect();
                std::hint::black_box(g[0]) + std::hint::black_box(h[0])
            };
            for _ in 0..WARMUP {
                let _ = gather_call();
            }
            let mut gather_ms: Vec<f64> = Vec::with_capacity(TIMED);
            for _ in 0..TIMED {
                let t = Instant::now();
                let _ = gather_call();
                gather_ms.push(t.elapsed().as_secs_f64() * 1e3);
            }
            let gather_med = median(&mut gather_ms);
            let lds_excl_med = (lds_incl_med - gather_med).max(0.0);

            // SANITY ASSERT (same-input correctness, NOT the parity gate): the two kernels
            // must compute the SAME RAW histogram within the f32-atomic envelope. The real
            // parity gate stays the CPU f64 anchor (existing rocm_cuda_mirror.rs tests).
            let cell = format!("leaf={leaf_name} bins={num_bin}");
            assert_same_input(&mirror_hist, &lds_hist, &cell);

            println!(
                "{:>9} | {:>5} | {:>11.3} | {:>11.3} | {:>11.3} | {:>9.3} | {:>8.2}x | {:>12.2}x",
                leaf_name,
                num_bin,
                mirror_med,
                lds_incl_med,
                lds_excl_med,
                gather_med,
                lds_incl_med / mirror_med,
                lds_excl_med / mirror_med,
            );
        }
    }

    println!();
    println!("# EXCLUDED from both timed loops: the resident feature-major bin upload (once/cell).");
    println!("# INCLUDED in lds_incl: the launcher's INTERNAL host ord_g/ord_h gather (production cost).");
    println!("# lds_excl = lds_incl - gather_ms (kernel + per-leaf uploads; gather removed). Mirror");
    println!("# gathers grad/hess in-kernel, so the gather tax is intrinsic to it (no separate column).");
    println!("# speedup>1.0 => mirror faster than the production LDS path; <1.0 => LDS faster.");
    println!("# Same-input sanity assert (ABS 5e-6 / REL 1e-5) passed for every cell printed above.");
    println!("# MEASUREMENT ONLY — does NOT wire into lib.rs / the learner. MANUAL bench, NOT a gate.");
}
