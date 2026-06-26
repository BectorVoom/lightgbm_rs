---
phase: quick-260626-igc
plan: 01
subsystem: lgbm-compute / lgbm
tags: [backend, cubecl, cuda, wgpu, gpu, compile-gated-wiring]
status: complete
requires:
  - cubecl 0.10.0 (cuda/wgpu sub-features)
  - existing runtime-generic GPU kernels (construct_histograms_lds_f32_on, find_best_split_f64_on, data_partition_on, subtract_histograms_f64_on)
provides:
  - lgbm-compute features gpu/cuda/wgpu; CudaBackend + WgpuBackend
  - lgbm features cuda/wgpu; booster backend-selection cascade (rocm > cuda > wgpu > cpu)
affects:
  - crates/lgbm-compute/Cargo.toml
  - crates/lgbm-compute/src/runtime.rs
  - crates/lgbm-compute/src/kernels/histogram.rs
  - crates/lgbm-compute/src/lib.rs
  - crates/lgbm/Cargo.toml
  - crates/lgbm/src/booster.rs
tech-stack:
  added: []
  patterns: [umbrella-feature, runtime-generic-kernel-reuse, mutually-exclusive-cfg-cascade, macro-deduped-backend-impl]
key-files:
  created: []
  modified:
    - crates/lgbm-compute/Cargo.toml
    - crates/lgbm-compute/src/runtime.rs
    - crates/lgbm-compute/src/kernels/histogram.rs
    - crates/lgbm-compute/src/lib.rs
    - crates/lgbm/Cargo.toml
    - crates/lgbm/src/booster.rs
decisions:
  - "wgpu compiles at the cargo-check gate: the f32-atomic/WGSL incompatibility (locked decision #3) is a RUNTIME WGSL-lowering concern, not a compile-time failure — the #[cube] macro only emits host code that builds kernel IR; WGSL codegen happens at launch (out of compile-only scope)."
  - "HIST_LDS_MAX const widened rocm->gpu alongside the two LDS-construct items it backs (plan step 3 contingency); it is a plain usize, no hip-sys/rocm_client involvement."
  - "No new [dependencies] added — cuda/wgpu are cubecl sub-features (T-igc-01 accept)."
metrics:
  duration: ~35m
  completed: 2026-06-26
---

# Quick 260626-igc: Support CUDA and WGPU by reusing the ROCm kernels — Summary

Added `cuda` (cubecl-cuda / `cubecl::cuda::CudaRuntime`) and `wgpu` (cubecl-wgpu /
`cubecl::wgpu::WgpuRuntime`) compute backends to `lgbm-compute`, wired exactly like the
existing ROCm backend and REUSING the runtime-generic `#[cube]` GPU kernels, then
surfaced both through the `lgbm` facade's feature-switched backend cascade. Scope was
compile-gated wiring only; the deliverable bar is the six-check `cargo check` matrix
below — all six PASS.

## What was built

- **lgbm-compute/Cargo.toml**: umbrella `gpu = []` feature; `cuda = ["cubecl/cuda", "gpu"]`;
  `wgpu = ["cubecl/wgpu", "gpu"]`; `gpu` folded into the `rocm` set
  (`rocm = ["cubecl/hip", "dep:cubecl-hip-sys", "gpu"]`). No new `[dependencies]`.
- **runtime.rs**: `CudaRuntime`/`WgpuRuntime` type aliases + `cuda_client()`/`wgpu_client()`
  ctors, mirroring `RocmRuntime`/`rocm_client()` (`CudaDevice::new(0)` ↔ `AmdDevice::new(0)`;
  `WgpuDevice::default()`). The wgpu doc-comment records the f32-atomic/WGSL risk.
- **kernels/histogram.rs**: widened ONLY the two LDS-construct items
  (`construct_hist_kernel_lds_f32` `#[cube]` + `construct_histograms_lds_f32_on` launcher)
  and the `HIST_LDS_MAX` cap they reference from `#[cfg(feature = "rocm")]` to
  `#[cfg(feature = "gpu")]`. The `cubecl_hip_sys` CU-count FFI (`query_num_cu` /
  `rowpart_target_cubes`) and every other rocm gate were left untouched.
- **lgbm-compute/src/lib.rs**: a `#[cfg(any(feature = "cuda", feature = "wgpu"))]`
  `macro_rules! gpu_core_backend!($name, $rt)` emitting a unit-struct backend +
  `impl Backend` overriding EXACTLY the four required methods, each dispatching the SAME
  runtime-generic kernels RocmBackend uses. Invoked as `CudaBackend`/`WgpuBackend`. All
  other `Backend` methods inherit the trait default (no resident-pool / CU-count FFI).
  `RocmBackend` left byte-untouched.
- **lgbm/Cargo.toml**: `cuda = ["lgbm-compute/cuda"]`, `wgpu = ["lgbm-compute/wgpu"]`.
- **lgbm/src/booster.rs**: both backend-selection cfg sites (imports + instantiation)
  extended from a two-arm rocm/cpu switch to a four-arm mutually-exclusive cascade
  (priority rocm > cuda > wgpu > cpu) using `not(...)` guards, guaranteeing exactly one
  backend per feature combination.

## cargo check matrix

All six checks were run from the repo root against the final committed tree (after both
task commits). Every check exited 0.

| # | Check | Result | Notes |
|---|-------|--------|-------|
| 1 | `cargo check` (default cpu) — **HARD GATE** | PASS (exit 0) | cpu anchor unbroken |
| 2 | `cargo check --features rocm` — **HARD GATE** | PASS (exit 0) | rocm path unbroken; RocmBackend byte-identical |
| 3 | `cargo check -p lgbm-compute --features cuda` | PASS (exit 0) | cubecl-cuda builds via `cudarc`/`libloading` (runtime loader) — no CUDA toolkit needed at check time |
| 4 | `cargo check -p lgbm --features cuda` | PASS (exit 0) | facade cuda arm selects CudaBackend |
| 5 | `cargo check -p lgbm-compute --features wgpu` | PASS (exit 0) | see discovered-outcome record below |
| 6 | `cargo check -p lgbm --features wgpu` | PASS (exit 0) | facade wgpu arm selects WgpuBackend |

## Discovered-outcome record (locked decision #3)

The plan anticipated that `--features wgpu` MIGHT fail to compile in the f32-atomic kernel
monomorphization (WGSL has no f32 atomics). **The actual discovered outcome: both wgpu
checks PASS at the `cargo check` gate.**

- **What:** No compile failure occurred for either wgpu check.
- **Where/Why:** cubecl's `#[cube(launch_unchecked)]` macro on `construct_hist_kernel_lds_f32`
  emits ordinary host Rust that constructs the kernel's IR; it does NOT statically lower
  the kernel to the target shading language. WGSL codegen (where the absent f32-atomic
  support would surface) happens at **runtime, on first kernel launch** — which is out of
  scope for this compile-gated-only task (this machine has no NVIDIA GPU and the AMD GPU is
  out of scope). So the f32-atomic/WGSL incompatibility from locked decision #3 is real but
  manifests at launch, not at `cargo check`. No kernel was swapped and no f64/portable
  fallback was added — the shared kernel is reused verbatim, exactly as required.
- **CUDA environment note:** No environment limitation was hit for cuda either — cubecl-cuda
  depends on `cudarc` + `libloading` (a runtime dynamic loader), so the absence of a CUDA
  toolkit on this machine does not block `cargo check`.

## Deviations from Plan

**1. [Rule 3 - blocking, plan-anticipated] Widened `HIST_LDS_MAX` const rocm -> gpu.**
- **Found during:** Task 1 (the two widened LDS items reference `HIST_LDS_MAX`, which was
  `#[cfg(feature = "rocm")]`). Plan step 3 explicitly authorized widening a genuinely-called
  helper to `gpu` UNLESS it touches `cubecl_hip_sys`/`rocm_client`. `HIST_LDS_MAX` is a
  plain `const usize = 512` — no FFI involvement — so it was widened to `#[cfg(feature = "gpu")]`.
- **Files modified:** crates/lgbm-compute/src/kernels/histogram.rs
- **Commit:** 3c158be

**Process note (not a plan deviation):** `cargo fmt -p <crate>` reformats the ENTIRE crate,
which pulled large pre-existing formatting drift into unrelated example/test/source files.
To keep each commit atomically scoped to its task files (scope boundary), the unrelated
fmt-only changes were reverted via `git checkout -- <file>` and the four/two task files were
re-applied with hand-written rustfmt-conformant edits. The committed diffs are
lgbm-compute (Cargo.toml 17, runtime.rs 48, histogram.rs 11, lib.rs 113) and
lgbm (Cargo.toml 9, booster.rs 48) — task files only, no drift.

No other deviations. No auth gates. No architectural changes.

## Known Stubs

None. This is compile-gated backend wiring; CudaBackend/WgpuBackend dispatch the same
already-reviewed runtime-generic kernels with no placeholder data paths.

## Self-Check: PASSED

- Commits exist: `3c158be` (Task 1), `7f00ba8` (Task 2) — both verified in `git log`.
- All six `cargo check` matrix entries returned exit 0 against the committed tree.
- `RocmBackend` impl + rocm runtime block unchanged; no new `[dependencies]`.
