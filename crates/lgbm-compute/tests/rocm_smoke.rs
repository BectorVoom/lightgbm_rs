//! ROCm/HIP bring-up smoke + capability assertion (CMP-03/CMP-04, ORA-04 rocm half).
//!
//! ENTIRELY feature-gated: the whole file is `#![cfg(feature = "rocm")]`, so it
//! compiles and runs ONLY under `cargo test -p lgbm-compute --features rocm` and
//! NEVER in a CPU-only build (SC#1 — the default `cargo test --workspace` does
//! not touch any ROCm toolchain).
//!
//! It exercises the REAL gfx1100 capability matrix (RESEARCH Pitfall 2):
//! `has_plane == true`, `has_f64 == false`, `has_f32_atomic == true`,
//! `plane_size == 32` — the asymmetric counterpart of the cpu matrix asserted in
//! `capability.rs`. Then it drives the f32-accumulate histogram kernel on the hip
//! runtime and asserts a finite result of the expected length (a SMOKE, not
//! bit-exact — the ~1e-6 hip-vs-cpu parity gate lives in
//! `oracle-harness/tests/kernel_parity.rs`).
#![cfg(feature = "rocm")]

use lgbm_compute::kernels::histogram::construct_histograms_f32_on;
use lgbm_compute::runtime::{
    probe_capabilities, rocm_client, AccumulateType, ReducePath,
};

/// Assert the verified hip capability matrix on the local gfx1100 (Pitfall 2):
/// Plane YES / f64 NO / f32-atomic YES / plane_size 32, and that the derived
/// gates select the Plane reduce path + the F32 accumulate type (the no-f64
/// hip routing).
#[test]
fn rocm_capability_matrix_gfx1100() {
    let client = rocm_client();
    let caps = probe_capabilities(&client);

    assert!(
        caps.has_plane,
        "cubecl-hip (gfx1100) must advertise Plane::Ops (wave32 collectives)"
    );
    assert!(
        !caps.has_f64,
        "cubecl-hip must NOT support f64 (disabled upstream) — the no-f64 path"
    );
    assert!(
        caps.has_f32_atomic,
        "cubecl-hip must advertise f32 atomic-add (AtomicUsage::Add registered)"
    );
    assert_eq!(
        caps.plane_size, 32,
        "gfx1100 (RDNA3) plane/warp size must be 32 (wave32)"
    );
    assert_eq!(
        caps.reduce_path(),
        ReducePath::Plane,
        "Plane::Ops present must select the Plane reduce path"
    );
    assert_eq!(
        caps.accumulate_type(),
        AccumulateType::F32,
        "no-f64 hip device must select the F32 accumulate cell type"
    );
}

/// Smoke: drive the f32-accumulate histogram kernel on the hip runtime over a
/// tiny hand-built input and assert the result has the expected length and is
/// all-finite. NOT bit-exact (that is the kernel_parity ~1e-6 gate's job).
#[test]
fn rocm_construct_histograms_f32_smoke() {
    let client = rocm_client();

    // 3 bins, 6 rows. Expected (f32): bin0 g=-3,h=2; bin1 g=5,h=2; bin2 g=2,h=2.
    let binned = vec![0u32, 1, 2, 0, 1, 2];
    let grad = vec![-1.0f32, 2.0, 1.0, -2.0, 3.0, 1.0];
    let hess = vec![1.0f32; 6];
    let num_bin = 3u32;

    let hist = construct_histograms_f32_on(&client, &binned, &grad, &hess, num_bin)
        .expect("hip f32 histogram smoke should succeed");

    assert_eq!(
        hist.len(),
        2 * num_bin as usize,
        "histogram must have 2*num_bin f32 cells"
    );
    assert!(
        hist.iter().all(|x| x.is_finite()),
        "every histogram cell must be finite, got {hist:?}"
    );
    // Sanity on the stride-2 layout (g0,h0,g1,h1,g2,h2): hessian cells == 2.0.
    assert_eq!(hist[1], 2.0, "bin0 hessian");
    assert_eq!(hist[3], 2.0, "bin1 hessian");
    assert_eq!(hist[5], 2.0, "bin2 hessian");
}
