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
    construct_histograms_cpu, construct_histograms_parallel_f32_on,
    construct_histograms_parallel_f32_plane_on,
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
/// the ~1e-6 ROCm gate (relative) at every swept bin count and at a small + a mid
/// leaf. The warp-aggregation tree-reduction order differs from the f64 ordered
/// fold; assert the per-cell relative error is inside the SAME envelope the naive
/// atomic path is held to (ABS 5e-6 / REL 1e-5). Pinned to the CPU f64 ANCHOR
/// (never GPU-vs-GPU — DEF-f8u-01).
#[test]
fn plane_within_tolerance_of_cpu_f64_anchor() {
    let cc = cpu_client();
    let gc = rocm_client();
    assert!(probe_capabilities(&gc).has_plane, "gfx1100 must report has_plane");

    for &n in &[8_000usize, 50_000usize] {
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
}

/// On EXACT-integer data the plane variant must EXACTLY equal the shipped baseline
/// non-plane path (`construct_histograms_parallel_f32_on`) — both produce the same
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

    let baseline = construct_histograms_parallel_f32_on(&gc, &binned, &grad, &hess, num_bin).unwrap();
    let plane =
        construct_histograms_parallel_f32_plane_on(&gc, &binned, &grad, &hess, num_bin, true).unwrap();
    for (i, (a, b)) in baseline.iter().zip(&plane).enumerate() {
        assert_eq!(*a, *b, "cell {i}: plane {b} != baseline {a} on integer data");
    }
}

/// The `use_plane=false` arm of the plane launcher must EXACTLY equal the shipped
/// `construct_histograms_parallel_f32_on` on integer data — proving the launcher's
/// baseline arm is a byte-faithful twin of the production kernel (the A/B baseline
/// arm is trustworthy as the reference point).
#[test]
fn plane_launcher_false_arm_matches_shipped_baseline() {
    let gc = rocm_client();
    let n = 20_000usize;
    let num_bin = 64u32;
    let binned: Vec<u32> = (0..n).map(|i| (i as u32) % num_bin).collect();
    let grad: Vec<f32> = binned.iter().map(|&b| (b % 9) as f32 - 4.0).collect();
    let hess = vec![1.0f32; n];

    let shipped = construct_histograms_parallel_f32_on(&gc, &binned, &grad, &hess, num_bin).unwrap();
    let false_arm =
        construct_histograms_parallel_f32_plane_on(&gc, &binned, &grad, &hess, num_bin, false).unwrap();
    for (i, (a, b)) in shipped.iter().zip(&false_arm).enumerate() {
        assert_eq!(*a, *b, "cell {i}: use_plane=false arm {b} != shipped baseline {a}");
    }
}
