---
quick_id: 260627-qxl
slug: give-cudabackend-an-on-device-resident-p
date: 2026-06-27
status: in-progress
---

# Quick Task: Give CudaBackend (and WgpuBackend) an on-device resident pool

## Goal

Make `device_type="cuda"` train at ROCm-parity speed by giving `CudaBackend` the same
on-device resident histogram pool `RocmBackend` has (build→fix→compact→subtract→scan
kept device-resident, eliminating the per-leaf host read-back + re-upload). User-chosen
approach: **shared generic** — hoist `RocmBackend` into a runtime-generic
`GpuBackend<R: Runtime>` so Rocm/Cuda/Wgpu run the SAME code (CUDA correctness verified
by proxy via the APU ROCm parity gate; no NVIDIA hardware here).

## Design

### A. Backend → generic (`lgbm-compute/src/lib.rs`)
- `struct RocmBackend { resident_bins, resident_pool, resident_enabled }`
  → `struct GpuBackend<R> { resident_bins, resident_pool, resident_enabled, _rt: PhantomData<fn() -> R> }`,
  gated `#[cfg(feature = "gpu")]` (the umbrella). Hand-written `Debug` (no `R: Debug` bound).
- `impl Backend for RocmBackend { type Runtime = RocmRuntime; … }`
  → `impl<R: cubecl::Runtime> Backend for GpuBackend<R> { type Runtime = R; … }`.
  Method bodies are VERBATIM (only `type Runtime` changes) — the risk scan found the
  RocmRuntime mention was the single hip-specific line in all 566 lines.
- `Default` + `with_resident` → generic inherent impls on `GpuBackend<R>`.
- Type aliases preserve every existing name/call site:
  `pub type RocmBackend = GpuBackend<RocmRuntime>` (#[cfg(rocm)]),
  `CudaBackend = GpuBackend<CudaRuntime>` (#[cfg(cuda)]),
  `WgpuBackend = GpuBackend<WgpuRuntime>` (#[cfg(wgpu)]).
- DELETE the `gpu_core_backend!` macro + its two invocations (superseded).
- Widen `ResidentBins` + `ResidentBinWidth` + `resident_bin_width` from `rocm` → `gpu`.

### B. Kernels → widen the resident kernels `rocm` → `gpu` (`kernels/*.rs`)
The resident/build/scan kernel fns the generic impl calls are `#[cfg(feature = "rocm")]`.
Widen each to `#[cfg(feature = "gpu")]` (behavior-preserving for ROCm). Compile-driven:
`cargo check --features cuda` enumerates exactly which fns need it. KEEP `rocm`-only:
- `query_num_cu()` + the hip-sys CU-count FFI block (histogram.rs ~634-743) — needs
  `cubecl_hip_sys` (a rocm-only dep). Provide a `#[cfg(all(feature="gpu", not(feature="rocm")))]`
  fallback returning `None` so `resolve_target_cubes`/`row_partition_count` use the
  CUBES_PER_CU/env heuristic under cuda/wgpu.
- the `autotune` module (perf-only). CUDA dispatch falls back to the `row_partition_count`
  heuristic (no resident-pool dependency); keep autotune `rocm`-gated.

### C. Booster (`lgbm/src/booster.rs`)
- `let backend = CudaBackend;` → `CudaBackend::default()` (no longer a unit struct);
  same for `WgpuBackend`. RocmBackend already uses `::default()`.

## Verification

- `cargo check --features cuda` and `--features wgpu`: clean (CudaBackend/WgpuBackend =
  GpuBackend<…> satisfy Backend with the full resident surface).
- Default CPU gate: `cargo test -p lgbm-compute -p lgbm-treelearner`, oracle-harness
  default suite — unaffected.
- **ROCm proxy gate (RUN HERE on the APU — this is CUDA's correctness proof):**
  `cargo build --features rocm`; `cargo test -p lgbm-compute --features rocm`
  (rocm_backend_parity); `cargo test -p oracle-harness --features rocm`
  (kernel_parity + learner_parity incl. resident/fused/`with_resident` tree-equivalence).
  Must be byte-for-byte as green as before — proves the generic refactor didn't regress
  the tested GPU path, and (same code) that CUDA's resident path is correct.
- **NVIDIA runtime (user, Colab/Kaggle):** `maturin --features cuda`, train with
  `device_type="cuda"` — now resident-pool fast, faithful within the f32 GPU envelope.

## Out of scope
- Porting the hip-sys CU-count autotune to CUDA (perf-only; heuristic fallback suffices).
- Any CpuBackend behavior change; any ROCm numeric change (cfg-widen is additive).
