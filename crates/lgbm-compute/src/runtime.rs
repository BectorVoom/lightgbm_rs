//! Runtime selection + startup capability gate (CMP-03 / CMP-04).
//!
//! This module is the one place that names a concrete CubeCL runtime (CMP-01):
//! the `cpu` feature binds [`cubecl::cpu::CpuRuntime`] (the D-04 deterministic
//! anchor) and the opt-in `rocm` feature binds the HIP runtime. A CPU-only
//! build needs no ROCm toolchain (SC#1).
//!
//! The startup capability gate ([`probe_capabilities`]) queries the active
//! runtime's [`Features`]/[`DeviceProperties`] exactly once and selects a
//! [`ReducePath`]. Per RESEARCH Pitfall 2 the capability matrix is asymmetric —
//! `Plane::Ops`/f64/f32-atomic differ between cubecl-cpu and cubecl-hip — so
//! every divergent feature is gated explicitly. On cubecl-cpu the
//! `Plane::Ops` feature is *absent* (`plane_size = 1`), so the
//! [`ReducePath::Sequential`] single-owner ordered fold IS the CPU path, not a
//! fallback option.

use cubecl::ir::features::{AtomicUsage, Plane};
use cubecl::ir::{ElemType, FloatKind, StorageType, Type};
use cubecl::prelude::*;

/// The device capabilities probed once at backend startup (CMP-04).
///
/// Mirrors the verified asymmetric matrix (RESEARCH Pitfall 2):
/// - cubecl-cpu: `has_plane=false`, `has_f64=true`, `has_f32_atomic=false`, `plane_size=1`
/// - cubecl-hip (gfx1100): `has_plane=true`, `has_f64=false`, `has_f32_atomic=true`, `plane_size=32`
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Capabilities {
    /// Whether warp/plane collective ops (`Plane::Ops`) are available.
    pub has_plane: bool,
    /// Whether f64 storage/arithmetic is supported (true on cpu, false on hip).
    pub has_f64: bool,
    /// Whether f32 atomic-add is registered (false on cpu, true on hip).
    pub has_f32_atomic: bool,
    /// The maximum plane (warp) size; 1 on cubecl-cpu, 32 on gfx1100 (wave32).
    pub plane_size: u32,
}

/// The reduction strategy selected from the probed [`Capabilities`].
///
/// On cubecl-cpu the only available path is [`ReducePath::Sequential`] (the
/// single-owner ordered f64 fold that is the deterministic anchor). On a
/// plane-capable runtime [`ReducePath::Plane`] may be used.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReducePath {
    /// Single-owner ordered fold (cubecl-cpu anchor; the C++ `num_threads=1` order).
    Sequential,
    /// Warp/plane collective reduction (plane-capable runtimes only).
    Plane,
}

/// The histogram/gain accumulation cell type, selected from the probed
/// [`Capabilities`] (CMP-04, RESEARCH Pitfall 2/3).
///
/// Gated EXPLICITLY on `has_f64`: cubecl-cpu supports f64 → the bit-exact f64
/// anchor; cubecl-hip (gfx1100) does NOT support f64 → the kernels accumulate in
/// f32, accepting the ~1e-6-tolerated divergence from the cpu f64 anchor (the
/// divergence the oracle contract was designed to absorb, D-03a). Every divergent
/// feature is gated off [`Capabilities`] — nothing is assumed present on both
/// backends.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AccumulateType {
    /// f64 cells — the cpu bit-exact anchor (`has_f64 == true`).
    F64,
    /// f32 cells — the no-f64 hip path (`has_f64 == false`).
    F32,
}

impl Capabilities {
    /// Derive the reduce path: `Plane` iff plane collectives are available,
    /// else `Sequential`. On cubecl-cpu `has_plane == false` so this is
    /// `Sequential` (the single-owner ordered fold IS the cpu path); on hip
    /// `has_plane == true`, but the single-owner ordered fold is still used FIRST
    /// for parity simplicity (RESEARCH Open Q2 RESOLVED), with the Plane
    /// reduction left as an optional optimization gated by this flag.
    #[must_use]
    pub fn reduce_path(&self) -> ReducePath {
        if self.has_plane {
            ReducePath::Plane
        } else {
            ReducePath::Sequential
        }
    }

    /// Derive the accumulation cell type: f64 iff the device supports f64
    /// (cpu anchor), else f32 (the no-f64 hip path). This is THE capability gate
    /// that routes the histogram/split/subtract accumulation between the f64
    /// anchor kernel and the f32 hip mirror (CMP-04, Pitfall 3).
    #[must_use]
    pub fn accumulate_type(&self) -> AccumulateType {
        if self.has_f64 {
            AccumulateType::F64
        } else {
            AccumulateType::F32
        }
    }
}

/// The f64 scalar storage type used to query f64 support.
fn f64_type() -> Type {
    Type::new(StorageType::Scalar(ElemType::Float(FloatKind::F64)))
}

/// The scalar (vector-size 1) f32 atomic type used to query atomic-add support.
fn f32_atomic_type() -> Type {
    Type::new(StorageType::Atomic(ElemType::Float(FloatKind::F32))).with_vector_size(1)
}

/// Probe the active runtime's capabilities via the verified cubecl 0.10.0 API
/// (CMP-04). Queried once at backend startup.
///
/// - `has_plane`  ← `client.features().plane.contains(Plane::Ops)`
/// - `has_f64`    ← `client.features().supports_type(f64)`
/// - `has_f32_atomic` ← `client.properties().atomic_type_usage(f32_atomic).contains(AtomicUsage::Add)`
/// - `plane_size` ← `client.properties().hardware.plane_size_max`
#[must_use]
pub fn probe_capabilities<R: Runtime>(client: &ComputeClient<R>) -> Capabilities {
    let has_plane = client.features().plane.contains(Plane::Ops);
    let has_f64 = client.features().supports_type(f64_type());
    let has_f32_atomic = client
        .properties()
        .atomic_type_usage(f32_atomic_type())
        .contains(AtomicUsage::Add);
    let plane_size = client.properties().hardware.plane_size_max;

    Capabilities {
        has_plane,
        has_f64,
        has_f32_atomic,
        plane_size,
    }
}

/// The active CubeCL runtime for the `cpu` feature build (the D-04 anchor).
///
/// Re-exported behind this crate's seam so no upstream crate names a cubecl
/// runtime (CMP-01).
#[cfg(feature = "cpu")]
pub type ActiveRuntime = cubecl::cpu::CpuRuntime;

/// Construct a compute client for the cpu reference runtime.
///
/// This is the runtime-selection entry for the default (cpu) build. The opt-in
/// `rocm` path ([`rocm_client`]) is compiled only under `#[cfg(feature = "rocm")]`.
#[cfg(feature = "cpu")]
#[must_use]
pub fn cpu_client() -> ComputeClient<cubecl::cpu::CpuRuntime> {
    cubecl::cpu::CpuRuntime::client(&cubecl::cpu::CpuDevice)
}

/// Construct a compute client for the ROCm/HIP runtime (opt-in, gfx-class GPUs).
///
/// Compiled only when the `rocm` feature is enabled, so the default build never
/// references `cubecl::hip` and needs no ROCm toolchain (SC#1, CMP-03). Binds
/// [`cubecl::hip::HipRuntime`] + `AmdDevice { index: 0 }` (the local gfx1100).
///
/// On this device [`probe_capabilities`] reports `has_f64 == false` and
/// `has_plane == true` (RESEARCH Pitfall 2). The histogram/split/subtract
/// accumulation MUST therefore route through the f32-cell kernels
/// ([`crate::kernels::histogram::construct_histograms_f32_on`],
/// [`crate::kernels::subtract::subtract_histograms_f32_on`],
/// [`crate::kernels::split::find_best_split_raw_f32_on`]) selected by
/// [`Capabilities::accumulate_type`] == [`AccumulateType::F32`]; `data_partition`
/// is f64-free and runs the SAME kernel
/// ([`crate::kernels::partition::data_partition_on`]) on both backends. The f32
/// path accepts the ~1e-6-tolerated divergence from the f64 cpu anchor (Pitfall
/// 3, D-03a — documented in `04-ROCM-GAPS.md`, never silent-passed).
#[cfg(feature = "rocm")]
#[must_use]
pub fn rocm_client() -> ComputeClient<cubecl::hip::HipRuntime> {
    rocm_client_for(&rocm_device(0))
}

/// The ROCm/HIP device at `index` — the `gpu_device_id` hyperparameter's
/// mapping onto CubeCL device selection.
///
/// Index `0` reproduces the historical hardcoded `AmdDevice::new(0)`, so the
/// default config selects exactly the device every prior build used.
#[cfg(feature = "rocm")]
#[must_use]
pub fn rocm_device(index: u32) -> cubecl::hip::AmdDevice {
    // Both `AmdDevice::new` and `CudaDevice::new` take a `usize` index in cubecl
    // 0.10; the seam takes `u32` because that is what `Config::gpu_device_index`
    // yields (a non-negative `gpu_device_id`).
    cubecl::hip::AmdDevice::new(index as usize)
}

/// Construct a ROCm/HIP client bound to an EXPLICIT device (see [`rocm_device`]).
///
/// Split out from [`rocm_client`] so the facade can honor `gpu_device_id`
/// without re-spelling the cubecl device type outside this module (CMP-01).
#[cfg(feature = "rocm")]
#[must_use]
pub fn rocm_client_for(device: &cubecl::hip::AmdDevice) -> ComputeClient<cubecl::hip::HipRuntime> {
    cubecl::hip::HipRuntime::client(device)
}

/// The active CubeCL runtime type for the `rocm` feature build (gfx-class GPUs).
///
/// Re-exported behind this crate's seam so no upstream crate names a cubecl
/// runtime (CMP-01).
#[cfg(feature = "rocm")]
pub type RocmRuntime = cubecl::hip::HipRuntime;

/// The active CubeCL runtime type for the `cuda` feature build (NVIDIA GPUs).
///
/// Re-exported behind this crate's seam so no upstream crate names a cubecl
/// runtime (CMP-01). Mirrors [`RocmRuntime`].
#[cfg(feature = "cuda")]
pub type CudaRuntime = cubecl::cuda::CudaRuntime;

/// Construct a compute client for the CUDA runtime (opt-in, NVIDIA GPUs).
///
/// Compiled only when the `cuda` feature is enabled, so the default build never
/// references `cubecl::cuda` and needs no CUDA toolkit present. Binds
/// [`cubecl::cuda::CudaRuntime`] + `CudaDevice::new(0)` (the first NVIDIA device),
/// mirroring [`rocm_client`]'s `AmdDevice::new(0)`.
///
/// CudaBackend dispatches the SAME runtime-generic GPU kernels RocmBackend uses
/// (`construct_histograms_lds_f32_on`, `find_best_split_f64_on`, `data_partition_on`,
/// `subtract_histograms_f64_on`) — no forked kernels.
#[cfg(feature = "cuda")]
#[must_use]
pub fn cuda_client() -> ComputeClient<cubecl::cuda::CudaRuntime> {
    cuda_client_for(&cuda_device(0))
}

/// The CUDA device at `index` — the `gpu_device_id` hyperparameter's mapping
/// onto CubeCL device selection.
///
/// Index `0` reproduces the historical hardcoded `CudaDevice::new(0)`, so the
/// default config selects exactly the device every prior build used.
#[cfg(feature = "cuda")]
#[must_use]
pub fn cuda_device(index: u32) -> cubecl::cuda::CudaDevice {
    cubecl::cuda::CudaDevice::new(index as usize)
}

/// Construct a CUDA client bound to an EXPLICIT device (see [`cuda_device`]).
///
/// Split out from [`cuda_client`] so the facade can honor `gpu_device_id`
/// without re-spelling the cubecl device type outside this module (CMP-01).
#[cfg(feature = "cuda")]
#[must_use]
pub fn cuda_client_for(
    device: &cubecl::cuda::CudaDevice,
) -> ComputeClient<cubecl::cuda::CudaRuntime> {
    // Flip the cubecl device-handle default to INLINE for CUDA trains (P100-measured
    // 1.045× wall win, preds byte-identical — plan doc §12). The env var stays the
    // user override in both directions (`CUBECL_DEVICE_INLINE=0` restores the
    // upstream channel handle); the choice latches process-globally on the first
    // client, so this must run before it — which is exactly this call site.
    cubecl_common::device::set_device_inline_default(true);
    cubecl::cuda::CudaRuntime::client(device)
}

/// The active CubeCL runtime type for the `wgpu` feature build (WebGPU/WGSL).
///
/// Re-exported behind this crate's seam so no upstream crate names a cubecl
/// runtime (CMP-01). Mirrors [`RocmRuntime`].
#[cfg(feature = "wgpu")]
pub type WgpuRuntime = cubecl::wgpu::WgpuRuntime;

/// Construct a compute client for the WGPU runtime (opt-in, WebGPU/WGSL targets).
///
/// Compiled only when the `wgpu` feature is enabled, so the default build never
/// references `cubecl::wgpu`. Binds [`cubecl::wgpu::WgpuRuntime`] +
/// `WgpuDevice::default()` (the default adapter), mirroring [`rocm_client`].
///
/// KNOWN RISK (locked decision #3): the shared LDS histogram kernel accumulates in
/// f32 atomics, and the WGSL target has NO f32 atomics — so a `--features wgpu`
/// build MAY fail to compile inside the f32-atomic kernel monomorphization. That is
/// an accepted, documented outcome; the kernel is NOT swapped or given an
/// f64/portable fallback to work around it. WgpuBackend dispatches the SAME
/// runtime-generic kernels RocmBackend/CudaBackend use — no forked kernels.
#[cfg(feature = "wgpu")]
#[must_use]
pub fn wgpu_client() -> ComputeClient<cubecl::wgpu::WgpuRuntime> {
    wgpu_client_for(&wgpu_device(0))
}

/// The WGPU device for `index`.
///
/// WGPU addresses adapters through `WgpuDevice` variants rather than a flat
/// index, so any `index` other than `0` cannot be honored; this returns the
/// default adapter in every case and the facade reports the unhonored
/// `gpu_device_id` via `Config::device_warnings` rather than silently binding a
/// different device than the user asked for.
#[cfg(feature = "wgpu")]
#[must_use]
pub fn wgpu_device(_index: u32) -> cubecl::wgpu::WgpuDevice {
    cubecl::wgpu::WgpuDevice::default()
}

/// Construct a WGPU client bound to an EXPLICIT device (see [`wgpu_device`]).
#[cfg(feature = "wgpu")]
#[must_use]
pub fn wgpu_client_for(
    device: &cubecl::wgpu::WgpuDevice,
) -> ComputeClient<cubecl::wgpu::WgpuRuntime> {
    cubecl::wgpu::WgpuRuntime::client(device)
}
