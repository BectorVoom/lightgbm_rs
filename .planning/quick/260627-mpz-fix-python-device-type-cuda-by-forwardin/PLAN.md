---
quick_id: 260627-mpz
slug: fix-python-device-type-cuda-by-forwardin
date: 2026-06-27
status: in-progress
---

# Quick Task: Fix Python `device_type="cuda"` (CudaBackend path)

## Problem (reported)

> "fix cuda device bug. It is not work."

Clarified with user:
- **Failing path:** Python `device_type="cuda"`.
- **Goal:** make the `CudaBackend` path actually work from Python.

## Root cause (investigated)

`device_type="cuda"` from Python can never reach the compiled `CudaBackend`, for
two independent reasons — and the `CudaBackend` dispatch code itself is fine:

1. **`crates/lgbm-python/Cargo.toml` has no `[features]` section.** It depends on
   `lgbm` with default features only and never forwards `cuda`/`rocm`/`wgpu`. So the
   Python extension module is **permanently compiled CPU-only** — `maturin --features cuda`
   would fail because the crate exposes no `cuda` feature to forward to `lgbm/cuda`.
2. **`crates/lgbm-python/src/params.rs:158` unconditionally rejects** `device_type=gpu/cuda`
   with a `ValueError`, even when a GPU backend is compiled in.

The `CudaBackend` (lib.rs:2096 via `gpu_core_backend!`) and `cuda_client()`
(runtime.rs:197) correctly reuse the runtime-generic ROCm kernels. Backend
selection is **compile-time** (`booster.rs:1103`, cascade rocm > cuda > wgpu > cpu);
`device_type` is otherwise a no-op for dispatch.

## Fix

1. **`crates/lgbm-python/Cargo.toml`** — add a `[features]` section forwarding the
   three GPU backends to `lgbm` (symmetric; same one-line gap affects rocm/wgpu):
   ```toml
   [features]
   default = []
   rocm = ["lgbm/rocm"]
   cuda = ["lgbm/cuda"]
   wgpu = ["lgbm/wgpu"]
   ```
2. **`crates/lgbm-python/src/params.rs`** — feature-gate the GPU `device_type`
   rejection: a GPU `device_type` ("gpu"/"cuda") is accepted **only** when this wheel
   was built with a matching CubeCL backend (`cfg!(feature = "rocm"|"cuda"|"wgpu")`).
   The default CPU-only wheel rejects them exactly as before.
3. **`crates/lgbm-python/pyproject.toml`** — document the CUDA wheel build command
   (`maturin build --release --features cuda`) in a comment near `[tool.maturin]`.

## Hardware constraint / verification scope

This machine has **no NVIDIA GPU and no CUDA toolkit** (confirmed: `nvcc`, `nvidia-smi`
absent). The `cuda` feature therefore **cannot be compiled or runtime-tested here**.

- **Verified here:** default CPU build still compiles; `lgbm-python` Rust unit tests
  (`reject_gate`, `build_config_rejects_unimplemented`) still pass (GPU device_types
  still rejected when no GPU backend is compiled in); `cargo check -p lgbm-python` clean.
- **Must verify on Colab/Kaggle (NVIDIA):** `maturin build --release --features cuda`
  compiles, the resulting wheel accepts `device_type="cuda"`, and training runs on the
  CudaBackend. (User already has the Colab/Kaggle benchmark harness in the tree.)

## Out of scope

- Touching the `CudaBackend`/`cuda_client()` kernel code (already correct).
- Making `device_type` a runtime backend selector (dispatch stays compile-time).
- Any change to the default CPU wheel's behavior.
