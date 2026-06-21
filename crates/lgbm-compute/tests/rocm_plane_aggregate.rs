//! Warp-aggregated (`use_plane=true`) f32-atomic histogram: correctness re-pin vs
//! the CPU f64 anchor (~1e-6 ROCm gate) on the real gfx1100 — quick task
//! 260619-p93. Feature-gated on `rocm`.
//!
//! These tests PIN the plane variant (`construct_histograms_parallel_f32_plane_on`
//! with `use_plane=true`) to the deterministic CPU f64 anchor
//! (`construct_histograms_cpu`), NEVER GPU-f32 to GPU-f32 (DEF-f8u-01: two
//! nondeterministic f32 paths must never be compared to each other at 1e-6). This
//! proves the plane same-bin TREE reduction is a CORRECT reordering INSIDE the
//! existing rocm contract (ABS 5e-6 / REL 1e-5) — a naive whole-plane `plane_sum`
//! would corrupt the histogram and fail these. No tolerance is widened.
#![cfg(feature = "rocm")]

use lgbm_compute::kernels::histogram::{
    construct_histograms_cpu, construct_histograms_parallel_f32_plane_on,
};
use lgbm_compute::runtime::{cpu_client, probe_capabilities, rocm_client};

/// Plane (warp-aggregated) correctness under HIGH contention: 50k rows into a few
/// bins with EXACT-integer grad/hess (exact in f32, so order-independent). The
/// warp-aggregated result must EXACTLY equal the known per-bin sums at every swept
/// bin count — proving the same-bin grouping loses/duplicates no update (a naive
/// `plane_sum` over divergent bins would mis-route adds and fail this).
#[test]
fn plane_no_lost_updates_under_contention() {
    let gc = rocm_client();
    assert!(probe_capabilities(&gc).has_plane, "gfx1100 must report has_plane");
    let n = 50_000usize;
    for &num_bin in &[16u32, 64, 256] {
        let binned: Vec<u32> = (0..n).map(|i| (i as u32) % num_bin).collect();
        // Small exact integers (representable + order-independent in f32).
        let grad: Vec<f32> = binned.iter().map(|&b| ((b % 5) as f32) - 2.0).collect();
        let hess = vec![1.0f32; n];

        let mut exp = vec![0.0f64; 2 * num_bin as usize];
        for i in 0..n {
            let ti = binned[i] as usize * 2;
            exp[ti] += grad[i] as f64;
            exp[ti + 1] += hess[i] as f64;
        }

        let got =
            construct_histograms_parallel_f32_plane_on(&gc, &binned, &grad, &hess, num_bin, true).unwrap();
        for (i, (g, e)) in got.iter().zip(&exp).enumerate() {
            assert_eq!(
                *g, *e,
                "bins={num_bin} cell {i}: warp-aggregated lost/extra/misrouted update — got {g}, expected {e}"
            );
        }
    }
}

/// Plane variant vs the CPU f64 anchor on REAL (non-integer) f32 data stays within
/// the ~1e-6 ROCm gate (relative) at every swept bin count, at the SAME n=8000-row
/// regime the existing `rocm_parallel_histogram` gate is calibrated for. The
/// warp-aggregation tree-reduction order differs from the f64 ordered fold; assert
/// the per-cell relative error is inside the SAME envelope the naive atomic path is
/// held to (ABS 5e-6 / REL 1e-5). Pinned to the CPU f64 ANCHOR (never GPU-vs-GPU —
/// DEF-f8u-01).
///
/// (The much-larger-leaf f32-cancellation regime — where BOTH the shipped baseline
/// AND the plane path drift ~3-4e-4 — is covered by
/// `plane_large_leaf_drift_not_worse_than_baseline`; that drift is a documented f32
/// effect shared by the shipped kernel, not a plane regression, and the 1e-5 gate
/// is not applied there for EITHER GPU path. No tolerance is widened.)
#[test]
fn plane_within_tolerance_of_cpu_f64_anchor() {
    let cc = cpu_client();
    let gc = rocm_client();
    assert!(probe_capabilities(&gc).has_plane, "gfx1100 must report has_plane");

    let n = 8_000usize;
    for &num_bin in &[16u32, 64, 256] {
        let binned: Vec<u32> = (0..n)
            .map(|i| (i as u32).wrapping_mul(2_654_435_761) % num_bin)
            .collect();
        // Small-magnitude pseudo-residual gradients (like real boosting gradients).
        let grad: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.000_123).sin() * 0.5).collect();
        let hess = vec![1.0f32; n];

        let cpu = construct_histograms_cpu(&cc, &binned, &grad, &hess, num_bin).unwrap();
        let gpu =
            construct_histograms_parallel_f32_plane_on(&gc, &binned, &grad, &hess, num_bin, true).unwrap();

        let mut max_rel = 0.0f64;
        for (c, g) in cpu.iter().zip(&gpu) {
            let denom = c.abs().max(1.0);
            max_rel = max_rel.max((c - g).abs() / denom);
        }
        println!("plane-f32 vs cpu-f64: n={n} bins={num_bin} max relative error = {max_rel:e}");
        assert!(
            max_rel < 1e-5,
            "n={n} bins={num_bin}: warp-aggregated f32 diverged too far from the f64 anchor: {max_rel:e}"
        );
    }
}

/// LARGE-LEAF f32 drift is SHARED by the shipped baseline (NOT a plane regression).
/// At n=50000 / bins=16 the small-magnitude near-cancelling `sin*0.5` gradients sum
/// ~3125 rows into ONE f32 cell, so BOTH the shipped per-row baseline AND the plane
/// variant drift ~3-4e-4 rel from the f64 anchor (catastrophic f32 cancellation —
/// the documented large-leaf effect, RESEARCH §parity-assessment / 04-ROCM-GAPS).
/// Measured on gfx1100 (260619-p93): baseline-vs-anchor 3.45e-4, plane-vs-anchor
/// 3.79e-4 — the SAME regime. This test asserts the plane variant is NOT materially
/// WORSE than the shipped baseline at this shape (within 2× of the baseline's own
/// drift), i.e. the tree reduction introduces no new error class; it does NOT
/// assert the 1e-5 gate here (neither GPU path meets it at this f32-cancelling
/// shape — that gate is calibrated for the ≤8000-row regime, see
/// `plane_within_tolerance_of_cpu_f64_anchor`). Pinned to the CPU f64 anchor.
#[test]
fn plane_large_leaf_drift_not_worse_than_baseline() {
    let cc = cpu_client();
    let gc = rocm_client();
    assert!(probe_capabilities(&gc).has_plane, "gfx1100 must report has_plane");
    let n = 50_000usize;
    let num_bin = 16u32;
    let binned: Vec<u32> = (0..n)
        .map(|i| (i as u32).wrapping_mul(2_654_435_761) % num_bin)
        .collect();
    let grad: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.000_123).sin() * 0.5).collect();
    let hess = vec![1.0f32; n];

    let cpu = construct_histograms_cpu(&cc, &binned, &grad, &hess, num_bin).unwrap();
    // Baseline = the plane launcher's `use_plane=false` arm (the byte-faithful twin of
    // the deleted per-row global-atomic kernel).
    let base =
        construct_histograms_parallel_f32_plane_on(&gc, &binned, &grad, &hess, num_bin, false)
            .unwrap();
    let plane =
        construct_histograms_parallel_f32_plane_on(&gc, &binned, &grad, &hess, num_bin, true).unwrap();

    let rel = |a: &[f64], b: &[f64]| -> f64 {
        let mut m = 0.0f64;
        for (x, y) in a.iter().zip(b) {
            let d = x.abs().max(1.0);
            m = m.max((x - y).abs() / d);
        }
        m
    };
    let base_rel = rel(&cpu, &base);
    let plane_rel = rel(&cpu, &plane);
    println!("large-leaf n={n} bins={num_bin}: baseline-vs-anchor={base_rel:e}  plane-vs-anchor={plane_rel:e}");
    // The plane tree reduction must not introduce a NEW error class — it stays within
    // 2× of the shipped baseline's own large-leaf f32 drift (a generous bound that
    // still fails loudly if the same-bin aggregation were silently corrupting cells).
    assert!(
        plane_rel <= 2.0 * base_rel.max(1e-6),
        "plane drift {plane_rel:e} materially exceeds the shipped baseline's own large-leaf drift {base_rel:e}"
    );
}

/// On EXACT-integer data the plane variant must EXACTLY equal the baseline non-plane
/// path (the plane launcher's `use_plane=false` arm) — both produce the same
/// per-bin integer sums, so this confirms the warp-aggregated path is a drop-in
/// correctness equivalent of the per-row scatter (the A/B bench's two arms compute
/// the same thing on integer inputs). NOT a 1e-6 GPU-vs-GPU float comparison: the
/// data is exact-integer so both are deterministic on these inputs.
#[test]
fn plane_equals_baseline_on_integer_data() {
    let gc = rocm_client();
    assert!(probe_capabilities(&gc).has_plane, "gfx1100 must report has_plane");
    let n = 30_000usize;
    let num_bin = 128u32;
    let binned: Vec<u32> = (0..n).map(|i| (i as u32) % num_bin).collect();
    let grad: Vec<f32> = binned.iter().map(|&b| (b % 7) as f32 - 3.0).collect();
    let hess = vec![1.0f32; n];

    // Baseline = the plane launcher's `use_plane=false` arm (byte-faithful twin of the
    // deleted per-row global-atomic kernel).
    let baseline =
        construct_histograms_parallel_f32_plane_on(&gc, &binned, &grad, &hess, num_bin, false)
            .unwrap();
    let plane =
        construct_histograms_parallel_f32_plane_on(&gc, &binned, &grad, &hess, num_bin, true).unwrap();
    for (i, (a, b)) in baseline.iter().zip(&plane).enumerate() {
        assert_eq!(*a, *b, "cell {i}: plane {b} != baseline {a} on integer data");
    }
}
