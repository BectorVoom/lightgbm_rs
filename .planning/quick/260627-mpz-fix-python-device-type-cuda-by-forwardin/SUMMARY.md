---
quick_id: 260627-mpz
slug: fix-python-device-type-cuda-by-forwardin
date: 2026-06-27
status: complete
commits:
  - 77d2acb fix(quick-260627-mpz): forward GPU cargo features through lgbm-python
  - fa023cb fix(quick-260627-mpz): feature-gate the Python device_type GPU rejection
---

# Summary: Fix Python `device_type="cuda"` (CudaBackend path)

## What was wrong

Python `device_type="cuda"` could never reach the compiled `CudaBackend`. The
`CudaBackend`/`cuda_client()` code itself was correct — the breakage was upstream
wiring:

1. `crates/lgbm-python/Cargo.toml` had **no `[features]` section**, so it never
   forwarded `cuda`/`rocm`/`wgpu` to `lgbm`. The extension module was permanently
   CPU-only and `maturin --features cuda` had no feature to forward.
2. `crates/lgbm-python/src/params.rs` **unconditionally rejected** `device_type=gpu/cuda`.

## Fix (2 commits)

- **77d2acb** — added `[features]` to `lgbm-python/Cargo.toml` forwarding
  `rocm`/`cuda`/`wgpu` to the `lgbm` crate; documented the CUDA wheel build command
  in `pyproject.toml`.
- **fa023cb** — feature-gated the gate: a GPU `device_type` is rejected only when no
  GPU backend is compiled in (`cfg!(feature = "rocm"|"cuda"|"wgpu")`). A GPU wheel now
  accepts `device_type="cuda"`; the default CPU wheel is unchanged.

## Verified (on this machine — CPU only)

- `cargo check -p lgbm-python` (default CPU): clean.
- `cargo test -p lgbm-python --lib`: **5/5 pass** (gate regression intact — GPU
  device_types still rejected when no GPU backend is built).
- `cargo tree -p lgbm-python --features cuda`: resolves and pulls in `cubecl-cuda
  v0.10.0` — proves the forwarding chain is correctly wired (previously this command
  errored: feature did not exist).

## NOT verified here — requires an NVIDIA host (BLOCKER for full sign-off)

This machine has no NVIDIA GPU / no CUDA toolkit (`nvcc`, `nvidia-smi` absent), so
the `cuda` feature cannot be **compiled** or **runtime-tested** locally. On Colab/Kaggle
(NVIDIA), the user must confirm:

1. `maturin build --release --features cuda` compiles the wheel.
2. The wheel accepts `device_type="cuda"` (no `ValueError`).
3. Training runs on the CudaBackend and is numerically faithful to the CPU anchor.

## Out of scope (intentionally untouched)

- `CudaBackend`/`cuda_client()` kernel code (already correct).
- Making `device_type` a runtime backend selector (dispatch stays compile-time;
  `device_type` on a GPU wheel selects whichever backend was compiled in, and only
  toggles force_row_wise vs force_col_wise per C++ semantics).
- Default CPU wheel behavior.
