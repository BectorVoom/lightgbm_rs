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
// Phase-14 foundation (additive, behind the OFF-by-default `LGBM_CUDA_ON_DEVICE`
// seam): shared device primitives (14-03/14-05), the SoA pre-allocated device
// split-record (14-04), and the `CUDARandom` LCG (14-04). Ungated like the other
// kernel modules (NOT `#[cfg(feature = "gpu")]` like `autotune`) so Wave-2 plans
// can each fill exactly one owned file with no `mod.rs` contention; they land as
// empty compiling stubs.
pub mod primitives;
pub mod random;
pub mod split;
pub mod split_info;
pub mod subtract;
