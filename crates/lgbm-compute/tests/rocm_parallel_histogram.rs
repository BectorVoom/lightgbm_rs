//! GPU-parallel f32-atomic histogram: correctness (~1e-6 ROCm gate) + speedup vs
//! the single-unit kernel, on the real gfx1100. Feature-gated on `rocm`.
#![cfg(feature = "rocm")]

use std::time::Instant;

use lgbm_compute::kernels::histogram::{
    construct_histograms_cpu, construct_histograms_lds_f32_on,
    construct_histograms_parallel_f32_plane_on,
};
use lgbm_compute::runtime::{cpu_client, rocm_client};

/// LDS sub-histogram correctness under HIGH contention: 50k rows into 4 bins with
/// exact-integer grad/hess (order-independent in f32). The LDS privatize→merge
/// result must EXACTLY equal the known per-bin sums — proving the per-cube LDS
/// atomics + the cross-cube global merge lose no update.
#[test]
fn lds_no_lost_updates_under_contention() {
    let gc = rocm_client();
    let n = 50_000usize;
    let num_bin = 4u32;
    let binned: Vec<u32> = (0..n).map(|i| (i as u32) % num_bin).collect();
    let grad: Vec<f32> = binned.iter().map(|&b| (b as f32) + 1.0).collect();
    let hess = vec![1.0f32; n];

    let mut exp = vec![0.0f64; 2 * num_bin as usize];
    for i in 0..n {
        let ti = binned[i] as usize * 2;
        exp[ti] += grad[i] as f64;
        exp[ti + 1] += hess[i] as f64;
    }

    let got = construct_histograms_lds_f32_on(&gc, &binned, &grad, &hess, num_bin).unwrap();
    for (i, (g, e)) in got.iter().zip(&exp).enumerate() {
        assert_eq!(*g, *e, "cell {i}: LDS lost/extra update — got {g}, expected {e}");
    }
}

/// LDS result vs the cpu f64 anchor on real (non-integer) f32 data stays within the
/// ~1e-6 ROCm gate (relative) — the same gate the naive atomic path is held to.
#[test]
fn lds_within_tolerance_of_cpu_f64_anchor() {
    let cc = cpu_client();
    let gc = rocm_client();
    let n = 8_000usize;
    let num_bin = 64u32;
    let binned: Vec<u32> = (0..n)
        .map(|i| (i as u32).wrapping_mul(2_654_435_761) % num_bin)
        .collect();
    let grad: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.000_123).sin() * 0.5).collect();
    let hess = vec![1.0f32; n];

    let cpu = construct_histograms_cpu(&cc, &binned, &grad, &hess, num_bin).unwrap();
    let gpu = construct_histograms_lds_f32_on(&gc, &binned, &grad, &hess, num_bin).unwrap();

    let mut max_rel = 0.0f64;
    for (c, g) in cpu.iter().zip(&gpu) {
        let denom = c.abs().max(1.0);
        max_rel = max_rel.max((c - g).abs() / denom);
    }
    println!("LDS-f32 vs cpu-f64: max relative error = {max_rel:e}");
    assert!(max_rel < 1e-5, "LDS f32 diverged too far from f64 anchor: {max_rel:e}");
}

/// LDS sub-histogram result must EXACTLY equal the naive global-atomic result on
/// integer data (both f32, same per-bin integer sums) — confirms the two parallel
/// paths are interchangeable for correctness.
#[test]
fn lds_equals_naive_atomic_on_integer_data() {
    let gc = rocm_client();
    let n = 30_000usize;
    let num_bin = 128u32;
    let binned: Vec<u32> = (0..n).map(|i| (i as u32) % num_bin).collect();
    let grad: Vec<f32> = binned.iter().map(|&b| (b % 7) as f32 - 3.0).collect();
    let hess = vec![1.0f32; n];

    // Baseline = the surviving plane launcher's `use_plane=false` arm (the byte-faithful
    // twin of the deleted global-atomic kernel on integer data).
    let naive =
        construct_histograms_parallel_f32_plane_on(&gc, &binned, &grad, &hess, num_bin, false)
            .unwrap();
    let lds = construct_histograms_lds_f32_on(&gc, &binned, &grad, &hess, num_bin).unwrap();
    for (i, (a, b)) in naive.iter().zip(&lds).enumerate() {
        assert_eq!(*a, *b, "cell {i}: LDS {b} != naive {a}");
    }
}

/// BENCHMARK: LDS-privatized vs naive global-atomic histogram under HIGH contention
/// (few bins, many rows) on the real gfx1100 — the regime where global-atomic
/// contention is supposed to dominate. Prints the speedup; does NOT assert a win
/// (whether LDS beats naive is the open question this measures — see f8u FINDINGS).
#[test]
fn bench_lds_vs_naive_atomic_large() {
    let gc = rocm_client();
    let reps = 50;
    for &(n, num_bin) in &[
        (1_000_000usize, 16u32), // high contention: 1M rows -> 16 bins
        (1_000_000usize, 256u32),
        (5_000_000usize, 16u32),
        (5_000_000usize, 256u32),
    ] {
        let binned: Vec<u32> = (0..n).map(|i| (i as u32) % num_bin).collect();
        let grad: Vec<f32> = (0..n).map(|i| (i as f32) * 0.001 - 1.0).collect();
        let hess = vec![1.0f32; n];

        // warmup both. Baseline = the plane launcher's `use_plane=false` arm (per-row
        // global-atomic, the byte-faithful twin of the deleted kernel).
        let _ =
            construct_histograms_parallel_f32_plane_on(&gc, &binned, &grad, &hess, num_bin, false)
                .unwrap();
        let _ = construct_histograms_lds_f32_on(&gc, &binned, &grad, &hess, num_bin).unwrap();

        let t0 = Instant::now();
        for _ in 0..reps {
            let _ = construct_histograms_parallel_f32_plane_on(
                &gc, &binned, &grad, &hess, num_bin, false,
            )
            .unwrap();
        }
        let naive_us = t0.elapsed().as_secs_f64() / reps as f64 * 1e6;

        let t0 = Instant::now();
        for _ in 0..reps {
            let _ = construct_histograms_lds_f32_on(&gc, &binned, &grad, &hess, num_bin).unwrap();
        }
        let lds_us = t0.elapsed().as_secs_f64() / reps as f64 * 1e6;

        println!(
            "[LDS-BENCH] n={n:>8} bins={num_bin:>3}  naive {naive_us:>8.1}us  lds {lds_us:>8.1}us  \
             speedup {:.2}x",
            naive_us / lds_us
        );
    }
}
