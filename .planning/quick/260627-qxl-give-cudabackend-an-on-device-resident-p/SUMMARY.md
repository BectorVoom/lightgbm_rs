---
quick_id: 260627-qxl
slug: give-cudabackend-an-on-device-resident-p
date: 2026-06-27
status: complete
commits:
  - 6b1ea9d feat(quick-260627-qxl): give CudaBackend/WgpuBackend the on-device resident pool
---

# Summary: CudaBackend (+WgpuBackend) on-device resident pool — ROCm-parity speed

## Goal & approach

`device_type="cuda"` trained via the slow per-leaf host read-back/re-upload path because
`CudaBackend`/`WgpuBackend` were pool-less 4-method unit structs (`gpu_core_backend!`
macro). User chose the **shared-generic** approach: hoist `RocmBackend` into a
runtime-generic `GpuBackend<R: cubecl::Runtime>` carrying the FULL resident histogram pool
(build→fix→compact→subtract→scan kept device-resident). `RocmBackend`/`CudaBackend`/
`WgpuBackend` are now type aliases over it, so CUDA/WGPU run the EXACT code the ROCm parity
gate validates on hardware — CUDA correctness is provable by proxy despite no NVIDIA GPU here.

## Key finding that made it feasible

The 566-line `impl Backend for RocmBackend` had exactly **one** HIP-specific line
(`type Runtime = RocmRuntime`); all methods were already written against `Self::Runtime`
and dispatch to runtime-generic `kernels::*_on<R>`. The only genuinely HIP-locked code is
`query_num_cu()` (hip-sys CU-count FFI), which lives in a kernel launcher, not the backend.

## Changes (commit 6b1ea9d)

- **`lib.rs`**: `struct RocmBackend` → generic `GpuBackend<R>` (`PhantomData<fn()->R>`;
  hand-written `Debug` to avoid an `R: Debug` bound); `impl Backend for RocmBackend` →
  `impl<R: cubecl::Runtime> Backend for GpuBackend<R>` (`type Runtime = R`, bodies verbatim);
  generic `Default`/`with_resident`; type aliases preserve every call site; deleted the
  `gpu_core_backend!` macro. Widened `ResidentBins`/`ResidentBinWidth` `rocm`→`gpu`.
- **`kernels/{histogram,split,subtract}.rs` + `mod.rs`**: widened the resident/build/scan/
  subtract kernels + the `cubecl::tune` autotune module `rocm`→`gpu` (behavior-preserving
  for ROCm: `gpu` is active under `rocm`). `query_num_cu` (hip-sys) stays `rocm`-only with
  a `None`-returning `#[cfg(all(gpu, not(rocm)))]` twin → cuda/wgpu use the
  `ROWPART_TARGET_CUBES_FALLBACK` heuristic. Fixed the `not(rocm)` oracle twins
  (`scan_cube_dim`, `autotuned`) to `not(gpu)` so they don't collide under cuda.
- **`Cargo.toml`**: moved `dep:serde` from `rocm`→`gpu` (autotune disk cache derives
  Serialize/Deserialize on `LaunchKey`, now shared by every GPU backend).
- **`booster.rs`**: `CudaBackend`/`WgpuBackend` now `::default()` (no longer unit structs).

Composes with quick-260627-o6i: `CudaBackend` now reports `resident_pool_supported()==true`
→ takes the resident/fused path; `host_unified_fused_supported()` stays false (the o6i gate
remains a correct safety net).

## Verification

- **Compile (here):** `lgbm-compute`/`lgbm`/`lgbm-python --features cuda` clean;
  `--features wgpu` clean; default CPU clean; **no warnings** on cuda or rocm.
- **ROCm PROXY GATE on the APU** (RocmBackend = the refactored generic ⇒ validates CUDA's
  resident code by construction): `rocm_backend_parity` 4/0; oracle `--features rocm`
  `kernel_parity` 21/0 (every resident kernel — resident_gather, build_fix_scan,
  resident_build_fix_compact, resident_build_all_pset, fused_scan_all_wset, sibling_copack,
  partition_exact); `learner_parity` 31/0 (incl. `learner_parity_resident_equals_host_tree_on_hip`
  + `fused_equals_host_tree`).
- **Default CPU merge gate:** lgbm-compute 52/0, lgbm-treelearner 77/0, oracle-harness full
  suite green (boosting 75, learner 29, kernel 7, C++ golden 2, config_drift 3).

## NOT verified here — NVIDIA runtime sign-off pending

No NVIDIA GPU / CUDA toolkit on this host. On Colab/Kaggle: `maturin build --release
--features cuda`, train with `device_type="cuda"` — expect ROCm-parity speed (resident pool
active) and fidelity within the f32 GPU envelope.

## Out of scope / follow-ups
- CUDA CU-count autotune: cuda uses the fallback heuristic (P from
  ROWPART_TARGET_CUBES_FALLBACK=64). A CUDA build can later read cubecl's reported
  `num_streaming_multiprocessors` (populated on cuda) for a tuned target.
- WGPU runtime: compiles, but the f32-atomic LDS kernel may fail at WGSL lowering at first
  launch (locked decision #3) — unchanged, documented, not worked around.
