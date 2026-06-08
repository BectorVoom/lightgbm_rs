---
quick_id: 260608-kfu
slug: switchable-rocmbackend-dispatch-gpu-f64
date: 2026-06-08
mode: quick (parity-gated; GPU dispatch)
---

# Quick Task 260608-kfu — switchable RocmBackend (dispatch compute to GPU, f64)

## Decision (user) + finding

User: add a switchable RocmBackend; ROCm supports f64. Empirically CONFIRMED on
the real gfx1100 — the f64 `construct_hist_kernel` runs on cubecl-hip and is
**bit-exact** to the CPU anchor (`max_abs_diff=0`), despite `probe_capabilities`
reporting `has_f64=false` (the flag is conservative/stale, not a real limit). So
the GPU runs the SAME f64 kernels → bit-exact, not the f32 ~1e-6 path.

## Tasks

- **T1 — generic f64 host fns (lgbm-compute).** Make the f64 cubecl host wrappers
  generic over `R: Runtime`: `construct_histograms_f64_on<R>`,
  `find_best_split_f64_on<R>`, `subtract_histograms_f64_on<R>` (move the `*_cpu`
  bodies into them; `*_cpu` delegate via `ActiveRuntime`). `data_partition_on<R>`
  already generic.
- **T2 — RocmBackend (lgbm-compute, `#[cfg(feature="rocm")]`).** `Backend` impl with
  `Runtime = RocmRuntime`, each op calling the `_f64_on::<RocmRuntime>` fn on the
  rocm client. Export it.
- **T3 — facade dispatch (lgbm).** Add `rocm` feature forwarding to
  `lgbm-compute/rocm`. In `booster.rs` gate the single `backend`/`client`/imports
  site (929-930): CpuBackend+cpu_client by default, RocmBackend+rocm_client under
  `rocm`. CPU stays the default; GPU is opt-in (switchable).
- **T4 — validate on GPU + report.** Focused lgbm-compute parity test (feature rocm)
  asserting RocmBackend == CpuBackend bit-exact for all 4 ops on shared inputs.
  Run on the real gfx1100. Note perf honestly (single-unit kernels = correct but
  not yet fast on GPU; parallelization is a follow-up). SUMMARY + STATE.

## Gates

- Default build UNCHANGED: `cargo test -p oracle-harness` GREEN (bit-exact),
  `cargo build` (no rocm) clean — the rocm path is fully `#[cfg]`-gated (SC#1).
- `cargo test -p lgbm-compute --features rocm` GREEN on the real GPU: RocmBackend
  ops bit-exact to the CPU anchor.
- Containment: GPU wiring stays in lgbm-compute; facade only flips the backend.
