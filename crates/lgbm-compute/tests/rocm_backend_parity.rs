//! GPU (RocmBackend) vs CPU (CpuBackend) bit-exact parity for all four hot-path
//! ops, on the real gfx1100. Feature-gated on `rocm` — runs only under
//! `cargo test -p lgbm-compute --features rocm`, never in a CPU-only build (SC#1).
//!
//! Proves the switchable GPU dispatch (260608-kfu) is numerically faithful: the f64
//! kernels run on cubecl-hip bit-exactly to the native-f64 CPU anchor.
#![cfg(feature = "rocm")]

use lgbm_compute::gain::GainConfig;
use lgbm_compute::runtime::{cpu_client, rocm_client};
use lgbm_compute::{Backend, CpuBackend, RocmBackend};

fn assert_bit_exact(cpu: &[f64], gpu: &[f64], what: &str) {
    assert_eq!(cpu.len(), gpu.len(), "{what}: length mismatch");
    for (i, (a, b)) in cpu.iter().zip(gpu).enumerate() {
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "{what}: cell {i} diverged — cpu {a} ({:#x}) vs gpu {b} ({:#x})",
            a.to_bits(),
            b.to_bits()
        );
    }
}

#[test]
fn rocm_backend_construct_histograms_bit_exact() {
    let cpu = CpuBackend;
    let gpu = RocmBackend::default();
    let cc = cpu_client();
    let gc = rocm_client();

    let binned = vec![0u32, 1, 2, 3, 0, 1, 2, 3, 1, 2, 3, 0, 2, 3];
    let grad: Vec<f32> = binned.iter().map(|&b| (b as f32) * 1.5 - 2.0).collect();
    let hess = vec![1.0f32; binned.len()];
    let num_bin = 4u32;

    let h_cpu = cpu
        .construct_histograms(&cc, &binned, &grad, &hess, num_bin)
        .unwrap();
    let h_gpu = gpu
        .construct_histograms(&gc, &binned, &grad, &hess, num_bin)
        .unwrap();
    assert_bit_exact(&h_cpu, &h_gpu, "construct_histograms");
}

#[test]
fn rocm_backend_find_best_split_bit_exact() {
    let cpu = CpuBackend;
    let gpu = RocmBackend::default();
    let cc = cpu_client();
    let gc = rocm_client();
    let cfg = GainConfig::default();

    // Build a histogram on the CPU and find the best split on both backends.
    let binned = vec![0u32, 0, 1, 1, 2, 2, 3, 3, 0, 1, 2, 3];
    let grad: Vec<f32> = vec![-3.0, -2.5, -1.0, -0.5, 0.5, 1.0, 2.0, 3.0, -2.0, 0.0, 1.5, 2.5];
    let hess = vec![1.0f32; binned.len()];
    let num_bin = 4u32;
    let hist = cpu
        .construct_histograms(&cc, &binned, &grad, &hess, num_bin)
        .unwrap();
    let sum_g: f64 = grad.iter().map(|&g| g as f64).sum();
    let sum_h: f64 = hess.iter().map(|&h| h as f64).sum();
    let num_data = binned.len() as i32;

    let s_cpu = cpu
        .find_best_split(
            &cc, &hist, &cfg, num_bin, 0, 0, 0, false, false, false, sum_g, sum_h, num_data,
        )
        .unwrap();
    let s_gpu = gpu
        .find_best_split(
            &gc, &hist, &cfg, num_bin, 0, 0, 0, false, false, false, sum_g, sum_h, num_data,
        )
        .unwrap();

    // Compare the SplitInfo field-by-field as bits.
    let to_cells = |s: &lgbm_compute::SplitInfo| -> Vec<f64> {
        vec![
            s.threshold as f64,
            s.gain,
            s.left_count as f64,
            s.right_count as f64,
            s.left_sum_gradient,
            s.left_sum_hessian,
            s.right_sum_gradient,
            s.right_sum_hessian,
            s.left_output,
            s.right_output,
            if s.default_left { 1.0 } else { 0.0 },
        ]
    };
    assert_bit_exact(&to_cells(&s_cpu), &to_cells(&s_gpu), "find_best_split");
}

#[test]
fn rocm_backend_subtract_histograms_bit_exact() {
    let cpu = CpuBackend;
    let gpu = RocmBackend::default();
    let cc = cpu_client();
    let gc = rocm_client();

    let parent: Vec<f64> = (0..16).map(|i| (i as f64) * 0.25 - 1.0).collect();
    let child: Vec<f64> = (0..16).map(|i| (i as f64) * 0.1).collect();

    let d_cpu = cpu.subtract_histograms(&cc, &parent, &child).unwrap();
    let d_gpu = gpu.subtract_histograms(&gc, &parent, &child).unwrap();
    assert_bit_exact(&d_cpu, &d_gpu, "subtract_histograms");
}

#[test]
fn rocm_backend_data_partition_matches() {
    let cpu = CpuBackend;
    let gpu = RocmBackend::default();
    let cc = cpu_client();
    let gc = rocm_client();

    let bins = vec![0u32, 1, 2, 3, 0, 1, 2, 3, 2, 1, 3, 0];
    let (r_cpu, sp_cpu) = cpu.data_partition(&cc, &bins, 4, 0, 3, 1, 0).unwrap();
    let (r_gpu, sp_gpu) = gpu.data_partition(&gc, &bins, 4, 0, 3, 1, 0).unwrap();
    assert_eq!(sp_cpu, sp_gpu, "data_partition split_point mismatch");
    assert_eq!(r_cpu, r_gpu, "data_partition reordered indices mismatch");
}
