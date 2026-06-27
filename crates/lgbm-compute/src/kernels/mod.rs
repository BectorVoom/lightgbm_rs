//! `#[cube]` compute kernels (CMP-05).
//!
//! Plan 04-01 shipped the minimal `construct_histograms` kernel that settled the
//! D-04a bit-determinism bet; 04-02 finalized it; 04-03 (this plan) adds
//! `find_best_split` (gain math in-kernel, D-01a), `data_partition` (stable
//! row->{left,right} routing), and `subtract_histograms` (the kernel-layer
//! histogram-subtraction math, A3 resolved in-scope).

/// Shared CubeCL-autotune plumbing (phase 13) for the GPU launch-config knobs.
/// gpu-gated (quick-260627-qxl widened from rocm): the default cpu build pulls no
/// autotune codegen (the f64 anchor is never autotuned), but every GPU backend
/// (rocm/cuda/wgpu) reuses the runtime-generic `cubecl::tune` plumbing.
#[cfg(feature = "gpu")]
pub mod autotune;
pub mod histogram;
pub mod partition;
pub mod split;
pub mod subtract;
