//! A/B micro-bench: immediate-sync vs deferred-sync (cubecl LAZY EXECUTION, manual
//! ch.05) for a leaf's per-feature f32-atomic histogram launches on the local
//! gfx1100 — quick task 260619-q2z (QUICK-260619-Q2Z).
//!
//! WHAT THIS QUANTIFIES: the one genuinely un-measured-in-isolation GPU histogram
//! OVERHEAD mechanism — cubecl LAZY EXECUTION. Today every per-feature histogram
//! launcher (`construct_histograms_parallel_f32_on`, histogram.rs:416) does a
//! `launch_unchecked` IMMEDIATELY followed by a BLOCKING `client.read_one_unchecked`.
//! So across a leaf's `feats` features the CPU submit serializes with the GPU
//! execute: submit → block → submit → block …. The manual's ch.05 lever decouples
//! submission from synchronization: submit all N per-feature launches back-to-back
//! (the CPU never blocks between them), then issue ONE deferred drain at the end.
//!
//!   - **Arm A (immediate-sync, current pattern):** for f in 0..feats { upload the
//!     f's fresh `out` handle (zeros) + launch the atomic kernel + IMMEDIATELY
//!     `read_one_unchecked(out_f)` }. N launches, N blocking drains.
//!   - **Arm B (deferred-sync, lazy):** allocate `feats` distinct `out` handles up
//!     front; for f in 0..feats launch into out-handle[f] WITHOUT reading; AFTER the
//!     launch loop, `read_one_unchecked` each handle (the deferred drain — the last
//!     read forces the whole queued batch to complete). N launches, 1 drain phase.
//!
//! The arms differ ONLY in WHEN the device sync happens (per-launch vs once at the
//! end). The device-resident binned matrix + a shared grad/hess pair are uploaded
//! ONCE per cell OUTSIDE both timed arms (transfer held equal), so the A/B isolates
//! the sync/dispatch pattern, not transfer.
//!
//! EXPECTED RESULT — likely NULL/modest (heavy corroborating in-repo prior art):
//! 260609-c2l proved the histogram path is LAUNCH-bound not transfer-bound;
//! nn7/oib found round-trip elimination a MIXED win; 260619-ol8 found
//! `launch_unchecked` itself NULL for the atomic-class kernels (they are contention/
//! scattered-read-latency / launch-FIXED-COST bound); 260619-nrw already SHIPPED
//! `launch_unchecked`. The lever's payoff scales with launches-per-leaf, so
//! FEATURE-COUNT is the PRIMARY axis (more features ⇒ more per-launch drains arm A
//! pays that arm B collapses into one). A credible benched NULL kept as a PRIMITIVE
//! pattern (NOT wired) is a fully valid outcome — same disposition as c2l/ol8/p93.
//! Do NOT manufacture a win.
//!
//! MEASUREMENT-ONLY: NO production kernel, launcher, wiring, or the CPU f64 anchor
//! is modified. The only artifacts of this task are this example + a FINDINGS.md.
//! Arm B is a HARNESS pattern (a launcher-call-ORDERING change), not a new kernel —
//! a WIRE verdict would mean refactoring the per-feature leaf loop to defer its
//! read_ones, which is a FOLLOW-UP plan, OUT OF SCOPE here. This spike only MEASURES.
//!
//! WARM-VS-COLD (load-bearing, spike-findings SKILL — cold-ceiling-overstates-warm):
//! WARMUP discarded launches per arm, MEDIAN of N timed launches, the device sync
//! (result read-back) FORCED INSIDE every timed call (arm A's reads are inline; arm
//! B's deferred reads are inside the timed block too — the whole point is arm B pays
//! ONE drain phase vs arm A's N drains), arms INTERLEAVED so thermal/clock drift hits
//! both equally, and the bench is meant to be RE-RUN across >=2 process restarts to
//! confirm the delta SIGN is stable vs noise. A delta within the p25/p75 spread, or
//! whose sign flips across runs, is SUB-NOISE / NULL (the ol8 disposition rule).
//!
//! This is a MANUAL rocm bench — NOT a test gate.
//!
//! Run: cargo run --release -p lgbm-compute --features rocm --example lazy_dispatch_ab

#[cfg(not(feature = "rocm"))]
fn main() {
    eprintln!("this micro-bench requires --features rocm (gfx1100). Re-run with it.");
}

// The `#[cube]` macro expansion + ArrayArg/CubeCount/CubeDim need the cubecl prelude
// in scope at the item level (same as histogram.rs's module-level prelude import).
#[cfg(feature = "rocm")]
use cubecl::prelude::*;

#[cfg(feature = "rocm")]
fn main() {
    // The SHIPPED per-feature atomic kernel — confirmed `pub` + `#[cube(launch_unchecked)]`
    // in histogram.rs (~388). This A/B inlines its `::launch_unchecked` into both arms;
    // the arms differ ONLY in when the sync (`read_one_unchecked`) happens. No twin is
    // needed — both arms launch the SAME shipped kernel, so there is zero risk of body
    // drift (the same-input assert below is purely a deferred-sync-changes-nothing guard).
    use lgbm_compute::kernels::histogram::construct_hist_kernel_atomic_f32;
    use lgbm_compute::runtime::{probe_capabilities, rocm_client};
    use std::time::Instant;

    let client = rocm_client();

    // This lever needs no plane collectives, but probe + print the capabilities for
    // the record (same as the sibling harnesses), and confirm the f32-atomic path the
    // arms exercise is actually available on this device.
    let caps = probe_capabilities(&client);
    assert!(
        caps.has_f32_atomic,
        "lazy_dispatch_ab exercises the f32-atomic kernel; probed {caps:?}"
    );
    println!("# probed capabilities: {caps:?}");

    // Median helper (sort, take middle) — same as the existing harnesses.
    let median = |xs: &mut [f64]| -> f64 {
        xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        xs[xs.len() / 2]
    };
    // p25 / p75 spread of an arm (the reader judges significance against this).
    let spread = |xs: &[f64]| -> (f64, f64) {
        let mut s = xs.to_vec();
        s.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let lo = s[s.len() / 4];
        let hi = s[(s.len() * 3) / 4];
        (lo, hi)
    };

    // Same-input drift guard — a GPU-f32-vs-GPU-f32 sanity check that DEFERRING the
    // sync does not change numerics, NOT the parity gate. The parity gate is the CPU
    // f64 anchor (per DEF-f8u-01 two nondeterministic GPU f32 paths must never be held
    // to each other at 1e-6, only to the f64 anchor). Arm A and arm B launch the SAME
    // kernel with the SAME inputs in the SAME per-feature order; only the read timing
    // differs — so they should agree to the f32-atomic envelope (ABS 5e-6 / REL 1e-5).
    // Copied verbatim from launch_unchecked_ab.rs.
    let assert_same_input_f32 = |a: &[f64], b: &[f64], cell: &str| {
        const ABS: f64 = 5e-6;
        const REL: f64 = 1e-5;
        assert_eq!(a.len(), b.len(), "{cell}: concatenated histogram length mismatch");
        for (i, (x, y)) in a.iter().zip(b).enumerate() {
            let diff = (x - y).abs();
            let tol = ABS + REL * x.abs();
            assert!(
                diff <= tol,
                "{cell}: cell {i} diverged beyond the f32-atomic envelope — \
                 immediate-sync {x} vs deferred-sync {y} (|diff| {diff} > tol {tol}); \
                 deferring the sync must NOT change numerics (same kernel, same inputs)"
            );
        }
    };

    // Deterministic feature-major bin generator (no RNG; same hash as launch_unchecked_ab.rs).
    // `resident[f*num_data + row]` is the device-resident binned matrix BOTH arms reuse.
    let gen_resident = |feats: usize, num_data: usize, num_bin: u32| -> Vec<u32> {
        let mut resident = Vec::with_capacity(feats * num_data);
        for f in 0..feats {
            for row in 0..num_data {
                let h = (row as u64)
                    .wrapping_mul(2_654_435_761)
                    .wrapping_add((f as u64).wrapping_mul(40_503).wrapping_add(0x9E37_79B9));
                resident.push((h % num_bin as u64) as u32);
            }
        }
        resident
    };
    let gen_grad = |n: usize| -> Vec<f32> { (0..n).map(|i| (i as f32) * 0.001 - 1.0).collect() };
    let gen_hess = |n: usize| -> Vec<f32> { (0..n).map(|i| 1.0 + ((i % 7) as f32) * 0.1).collect() };

    // PRIMARY axes: FEATURE-COUNT (the launches/leaf knob — the lever's payoff scales
    // with it) AND bin-count.
    const FEATS_SWEEP: [usize; 3] = [8, 32, 128];
    const BIN_SWEEP: [u32; 3] = [16, 64, 256];

    println!("# lazy execution (deferred sync) vs immediate sync A/B  —  gfx1100, 260619-q2z");
    println!("# arm A (immediate) = feats x {{ launch construct_hist_kernel_atomic_f32 into a fresh out + IMMEDIATE read_one }}");
    println!("# arm B (deferred)  = feats launches into feats distinct out handles (no read), THEN read each (one drain phase)");
    println!("# SAME shipped kernel both arms; differ ONLY in WHEN the device sync happens.");
    println!("# Resident bin matrix + shared grad/hess uploaded ONCE per cell, OUTSIDE both timed arms (transfer held equal).");
    println!("# INTERLEAVED arm A / arm B per timed iter; WARMUP discarded; median + p25/p75 spread.");
    println!("# device sync FORCED inside each timed call (arm A's N inline reads; arm B's deferred reads in-block).");
    println!("# delta% = (a_med - b_med) / a_med * 100   (>0 => arm B / deferred-sync faster)");
    println!("# PRIMARY axes: feats (launches/leaf) AND bins. RE-RUN across >=2 restarts to confirm delta SIGN stable vs noise.");
    println!("# EXPECTED: NULL/modest — c2l launch-bound-not-transfer; ol8 launch_unchecked NULL for atomics (launch-fixed-cost bound).");
    println!();

    println!("## per-feature f32-atomic histogram: deferred-sync vs immediate-sync");
    println!(
        "{:>13} | {:>5} | {:>5} | {:>11} | {:>11} | {:>8} | {:>17} | {:>17}",
        "regime", "feats", "bins", "a_ms", "b_ms", "delta%", "A p25/p75", "B p25/p75"
    );
    println!(
        "{:-<14}+{:-<7}+{:-<7}+{:-<13}+{:-<13}+{:-<10}+{:-<19}+{:-<19}",
        "", "", "", "", "", "", "", ""
    );

    // (regime label, num_data, leaf rows, warmup, timed). Launch-bound: small leaf so
    // per-launch submit/sync fixed cost dominates and the lever is most likely to show.
    // Compute-bound: large leaf so GPU execute dominates and the lever should wash out.
    let cells: [(&str, usize, usize, usize, usize); 2] = [
        ("launch-bound", 4_096, 1_024, 15, 21),
        ("compute-bound", 200_000, 200_000, 3, 7),
    ];
    for (regime, num_data, leaf_n, warmup, timed) in cells {
        for &feats in &FEATS_SWEEP {
            for &num_bin in &BIN_SWEEP {
                let resident = gen_resident(feats, num_data, num_bin);
                let grad = gen_grad(num_data);
                let hess = gen_hess(num_data);
                // One shared grad/hess gathered to leaf length (rows 0..leaf_n), like the
                // LDS cells in launch_unchecked_ab.rs — every feature shares it.
                let ord_g: Vec<f32> = (0..leaf_n).map(|r| grad[r]).collect();
                let ord_h: Vec<f32> = (0..leaf_n).map(|r| hess[r]).collect();
                let out_len = 2 * num_bin as usize;

                // Upload the device-resident pieces ONCE per cell, OUTSIDE both timed
                // arms (transfer held equal across A/B — the bench isolates sync timing,
                // not transfer). The per-feature binned column is sliced from the resident
                // matrix; we upload each feature's column once here too so neither arm
                // pays per-iter upload of the bin/grad/hess inputs — ONLY the per-feature
                // `out` handle is created fresh inside the arms (it must be zeroed per launch).
                let h_bins: Vec<_> = (0..feats)
                    .map(|f| {
                        let col = &resident[f * num_data..f * num_data + leaf_n];
                        client.create_from_slice(u32::as_bytes(col))
                    })
                    .collect();
                let h_g = client.create_from_slice(f32::as_bytes(&ord_g));
                let h_h = client.create_from_slice(f32::as_bytes(&ord_h));

                let cube_dim = 256u32;
                let cube_count = (leaf_n as u32).div_ceil(cube_dim);
                let zeros = vec![0.0f32; out_len];

                // ---- Arm A: immediate-sync (the current per-feature launcher pattern) ----
                // For each feature: fresh zeroed out handle, launch, IMMEDIATELY drain.
                // SAFETY: identical host-proven obligations to the shipped
                // construct_histograms_parallel_f32_on launcher — the bin column / grad /
                // hess are sized leaf_n, the kernel guards idx < leaf_n (tail units idle),
                // every resident bin < num_bin keeps out[bin*2+1] < out_len, and out is
                // sized out_len = 2*num_bin. cubecl unsafe is confined to the example (CMP-01).
                let arm_a = || -> Vec<f64> {
                    let mut concat = Vec::with_capacity(feats * out_len);
                    for f in 0..feats {
                        let h_out = client.create_from_slice(f32::as_bytes(&zeros));
                        unsafe {
                            construct_hist_kernel_atomic_f32::launch_unchecked(
                                &client,
                                CubeCount::Static(cube_count, 1, 1),
                                CubeDim::new_1d(cube_dim),
                                ArrayArg::from_raw_parts(h_bins[f].clone(), leaf_n),
                                ArrayArg::from_raw_parts(h_g.clone(), leaf_n),
                                ArrayArg::from_raw_parts(h_h.clone(), leaf_n),
                                ArrayArg::from_raw_parts(h_out.clone(), out_len),
                            );
                        }
                        // IMMEDIATE blocking drain — the current pattern (submit → block).
                        let bytes = client.read_one_unchecked(h_out);
                        concat.extend(f32::from_bytes(&bytes).iter().map(|&x| f64::from(x)));
                    }
                    concat
                };

                // ---- Arm B: deferred-sync (cubecl lazy execution, manual ch.05) ----
                // Allocate feats distinct out handles, submit all feats launches back-to-
                // back WITHOUT reading, THEN drain each handle (the deferred sync — the
                // reads force the whole queued batch to complete). Same SAFETY obligations
                // as arm A (identical launches, identical inputs); only the sync timing moves.
                let arm_b = || -> Vec<f64> {
                    // N distinct zeroed out handles up front.
                    let h_outs: Vec<_> = (0..feats)
                        .map(|_| client.create_from_slice(f32::as_bytes(&zeros)))
                        .collect();
                    // Submit all feats launches back-to-back; the CPU never blocks here.
                    for f in 0..feats {
                        unsafe {
                            construct_hist_kernel_atomic_f32::launch_unchecked(
                                &client,
                                CubeCount::Static(cube_count, 1, 1),
                                CubeDim::new_1d(cube_dim),
                                ArrayArg::from_raw_parts(h_bins[f].clone(), leaf_n),
                                ArrayArg::from_raw_parts(h_g.clone(), leaf_n),
                                ArrayArg::from_raw_parts(h_h.clone(), leaf_n),
                                ArrayArg::from_raw_parts(h_outs[f].clone(), out_len),
                            );
                        }
                    }
                    // Deferred drain: read each handle. The last read forces the full queue
                    // to drain — arm B pays ONE drain phase vs arm A's N interleaved drains.
                    let mut concat = Vec::with_capacity(feats * out_len);
                    for h_out in h_outs {
                        let bytes = client.read_one_unchecked(h_out);
                        concat.extend(f32::from_bytes(&bytes).iter().map(|&x| f64::from(x)));
                    }
                    concat
                };

                // Interleaved warmup (untimed).
                for _ in 0..warmup {
                    let _ = arm_a();
                    let _ = arm_b();
                }
                let mut a_ms = Vec::with_capacity(timed);
                let mut b_ms = Vec::with_capacity(timed);
                let mut a_hist = Vec::new();
                let mut b_hist = Vec::new();
                for _ in 0..timed {
                    // Time arm A (its N reads are inline, inside the timed region).
                    let t = Instant::now();
                    let ah = arm_a();
                    a_ms.push(t.elapsed().as_secs_f64() * 1e3);
                    // Time arm B (its deferred reads are inside the timed block too — the
                    // whole point is arm B pays ONE drain at the end vs arm A's N drains).
                    let t = Instant::now();
                    let bh = arm_b();
                    b_ms.push(t.elapsed().as_secs_f64() * 1e3);
                    a_hist = ah;
                    b_hist = bh;
                }
                let cell = format!("lazy {regime} feats={feats} bins={num_bin}");
                // Deferring the sync must not change numerics (same kernel, same inputs,
                // same per-feature order) — concatenated A vs B agree within the f32 envelope.
                assert_same_input_f32(&a_hist, &b_hist, &cell);
                let a_med = median(&mut a_ms);
                let b_med = median(&mut b_ms);
                let (al, ah_) = spread(&a_ms);
                let (bl, bh_) = spread(&b_ms);
                let delta = (a_med - b_med) / a_med * 100.0;
                println!(
                    "{:>13} | {:>5} | {:>5} | {:>11.4} | {:>11.4} | {:>7.2} | {:>7.4}/{:>7.4} | {:>7.4}/{:>7.4}",
                    regime, feats, num_bin, a_med, b_med, delta, al, ah_, bl, bh_
                );
            }
        }
    }

    println!();
    println!("# LEGEND: delta% > 0 => arm B (deferred-sync / lazy) faster; < 0 => arm A (immediate-sync) faster.");
    println!("# The same-input asserts above passed (f32 envelope) ⇒ deferring the sync is numerically IDENTICAL.");
    println!("# A delta within the p25/p75 spread, or whose SIGN flips across process restarts, is");
    println!("#   SUB-NOISE / NULL — re-run >=2 times and compare. Per c2l/ol8 the win is expected NULL/modest");
    println!("#   (launch-bound, but the per-launch FIXED submit cost may not be hideable for these short atomic");
    println!("#   kernels). The lever's payoff scales with feats (launches/leaf), so the high-feats launch-bound");
    println!("#   cells are where a real win would surface. MEASUREMENT ONLY — production untouched.");
}
