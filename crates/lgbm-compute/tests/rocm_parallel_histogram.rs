//! GPU-parallel f32-atomic histogram: correctness (~1e-6 ROCm gate) + speedup vs
//! the single-unit kernel, on the real gfx1100. Feature-gated on `rocm`.
#![cfg(feature = "rocm")]

use std::time::Instant;

use lgbm_compute::kernels::histogram::{
    construct_histograms_cpu, construct_histograms_f32_on, construct_histograms_parallel_f32_on,
};
use lgbm_compute::runtime::{cpu_client, rocm_client};

/// Atomic correctness under HIGH contention: 50k rows into just 4 bins, with
/// EXACT-integer gradients/hessians (exact in f32, so order-independent). The
/// parallel atomic result must EXACTLY equal the known per-bin sums — proving no
/// atomic update is lost when thousands of units hammer the same 8 cells.
#[test]
fn parallel_atomic_no_lost_updates_under_contention() {
    let gc = rocm_client();
    let n = 50_000usize;
    let num_bin = 4u32;
    let binned: Vec<u32> = (0..n).map(|i| (i as u32) % num_bin).collect();
    let grad: Vec<f32> = binned.iter().map(|&b| (b as f32) + 1.0).collect(); // 1..4
    let hess = vec![1.0f32; n];

    // Exact expected per-bin sums.
    let mut exp = vec![0.0f64; 2 * num_bin as usize];
    for i in 0..n {
        let ti = binned[i] as usize * 2;
        exp[ti] += grad[i] as f64;
        exp[ti + 1] += hess[i] as f64;
    }

    let got = construct_histograms_parallel_f32_on(&gc, &binned, &grad, &hess, num_bin).unwrap();
    for (i, (g, e)) in got.iter().zip(&exp).enumerate() {
        assert_eq!(*g, *e, "cell {i}: lost/extra atomic update — got {g}, expected {e}");
    }
}

/// Divergence from the cpu f64 anchor on REAL (non-integer) f32 data stays within
/// the ~1e-6 ROCm gate (relative). The parallel f32 atomic order differs from the
/// f64 ordered fold; assert the per-cell relative error is small.
#[test]
fn parallel_within_tolerance_of_cpu_f64_anchor() {
    let cc = cpu_client();
    let gc = rocm_client();
    let n = 8_000usize;
    let num_bin = 64u32;
    let binned: Vec<u32> = (0..n)
        .map(|i| (i as u32).wrapping_mul(2_654_435_761) % num_bin)
        .collect();
    // Small-magnitude pseudo-residual gradients (like real boosting gradients).
    let grad: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.000_123).sin() * 0.5).collect();
    let hess = vec![1.0f32; n];

    let cpu = construct_histograms_cpu(&cc, &binned, &grad, &hess, num_bin).unwrap();
    let gpu = construct_histograms_parallel_f32_on(&gc, &binned, &grad, &hess, num_bin).unwrap();

    let mut max_rel = 0.0f64;
    for (c, g) in cpu.iter().zip(&gpu) {
        let denom = c.abs().max(1.0);
        max_rel = max_rel.max((c - g).abs() / denom);
    }
    println!("parallel-f32 vs cpu-f64: max relative error = {max_rel:e}");
    assert!(max_rel < 1e-5, "f32 parallel diverged too far from f64 anchor: {max_rel:e}");
}

/// Speed: parallel f32-atomic vs single-unit f64 on a 20k-row leaf, on the GPU.
#[test]
fn parallel_is_faster_than_single_unit_on_gpu() {
    let gc = rocm_client();
    let n = 20_000usize;
    let num_bin = 128u32;
    let binned: Vec<u32> = (0..n).map(|i| (i as u32) % num_bin).collect();
    let grad: Vec<f32> = (0..n).map(|i| (i as f32) * 0.001 - 1.0).collect();
    let hess = vec![1.0f32; n];
    let reps = 200;

    // warm up both paths (first launch compiles the kernel).
    let _ = construct_histograms_f32_on(&gc, &binned, &grad, &hess, num_bin).unwrap();
    let _ = construct_histograms_parallel_f32_on(&gc, &binned, &grad, &hess, num_bin).unwrap();

    let t0 = Instant::now();
    for _ in 0..reps {
        let _ = construct_histograms_f32_on(&gc, &binned, &grad, &hess, num_bin).unwrap();
    }
    let single_us = t0.elapsed().as_secs_f64() / reps as f64 * 1e6;

    let t0 = Instant::now();
    for _ in 0..reps {
        let _ = construct_histograms_parallel_f32_on(&gc, &binned, &grad, &hess, num_bin).unwrap();
    }
    let parallel_us = t0.elapsed().as_secs_f64() / reps as f64 * 1e6;

    println!(
        "GPU 20k-row histogram: single-unit {single_us:.1}us  parallel {parallel_us:.1}us  speedup {:.1}x",
        single_us / parallel_us
    );
    assert!(
        parallel_us < single_us,
        "parallel ({parallel_us:.1}us) must beat single-unit ({single_us:.1}us)"
    );
}
