---
quick_id: 260608-kfu
slug: switchable-rocmbackend-dispatch-gpu-f64
status: complete
date: 2026-06-08
---

# Quick Task 260608-kfu — switchable RocmBackend (GPU dispatch) — SUMMARY

## What was delivered

A **switchable GPU backend**: with `--features rocm`, the facade `train()` dispatches
all compute to the ROCm/cubecl-hip runtime on the local gfx1100, running the **f64**
kernels — **bit-exact** to the CPU anchor. Default build is unchanged (native-f64
CpuBackend); the GPU path is fully `#[cfg]`-gated.

## Key finding — ROCm runs f64 (the user was right)

cubecl-hip's `probe_capabilities().has_f64` reports **false** on the gfx1100, but
that flag is conservative: the f64 kernels actually **run and are bit-exact**
(probe: `max_abs_diff=0`). So GPU dispatch keeps the SAME bit-exact numerical
contract as CPU — not the f32 ~1e-6 mirror that the `has_f64=false` framing assumed.

## Changes

- **lgbm-compute:** f64 cube host wrappers made generic over `R: Runtime`
  (`construct_histograms_f64_on` / `find_best_split_f64_on` / `subtract_histograms_
  f64_on`; the `*_cpu` versions delegate via `ActiveRuntime`). New `RocmBackend`
  (`#[cfg(feature="rocm")]`) `Backend` impl over `RocmRuntime`.
- **lgbm:** `rocm` feature → `lgbm-compute/rocm`; `booster.rs` feature-switches the
  single backend/client construction site. Learner + GBDT loop already generic over
  `B: Backend`, so nothing else changed.

## Validation (real gfx1100)

- `rocm_backend_parity` (feature rocm): all 4 ops (construct / find_best_split /
  subtract / data_partition) **bit-exact** CPU vs GPU.
- Full `lgbm` facade suite (**41 tests**) passes dispatched to the GPU — they assert
  bit-exact vs CPU goldens, so the whole train path is bit-exact on the GPU.
- Default build untouched: oracle-harness bit-exact gate GREEN, compute units 18.
  CPU-only build never references the HIP runtime (SC#1).

## Honest perf result — correct, but SLOW (not yet a speed win)

| size | CPU native (M5c) | GPU (this) | GPU vs CPU |
|------|------------------|------------|-----------|
| small  | 38.7ms | 8.28s  | **~214× SLOWER** |
| medium | 258ms  | 40.8s  | ~158× slower |

The GPU runs the SAME `CubeDim::new_1d(1)` **single-unit** kernels — one GPU lane
executing a sequential loop, with every per-(feature,leaf) op now a GPU dispatch +
round-trip. That is pathologically slow on a GPU (and reintroduces exactly the
per-launch overhead R2 removed on CPU). **The GPU dispatch is a correctness /
enablement milestone, NOT a speedup.**

## To make the GPU actually fast (follow-up, larger effort)

The kernels must be PARALLELIZED to use the GPU's lanes:
- histogram construction: multi-unit with f32-atomic adds (gfx1100 has `has_f32_
  atomic=true`) or Plane (wave32) reductions — but atomics/Plane reorder the f64
  fold, so the bit-exact gate would relax to the ~1e-6 ROCm gate (by design).
- split scan: parallel prefix-sum over bins instead of the sequential running sum.
- batch features per leaf into one launch; keep the binned dataset device-resident
  across iterations (avoid per-op host↔device transfer).

This is the real GPU-acceleration phase; it trades the bit-exact CPU gate for the
~1e-6 ROCm gate (CLAUDE.md's intended GPU contract).
