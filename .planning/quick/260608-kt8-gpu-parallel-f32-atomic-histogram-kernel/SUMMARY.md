---
quick_id: 260608-kt8
slug: gpu-parallel-f32-atomic-histogram-kernel
status: complete
date: 2026-06-08
---

# Quick Task 260608-kt8 — GPU-parallel f32-atomic histogram kernel — SUMMARY

## What was delivered

A **parallel f32-atomic histogram kernel** for the GPU (`RocmBackend`), replacing the
single-unit (`CubeDim::new_1d(1)`, one GPU lane) fold. One unit per row, each doing
`Atomic<f32>::fetch_add` into the shared global histogram (gfx1100 has f32 atomics),
launched over `ceil(n/256)` cubes of 256 units (manual: `ABSOLUTE_POS` + bounds
check + `CubeCount`/`CubeDim`). f64-widened on readback.

Atomics + nondeterministic add order move the GPU path to the **~1e-6 ROCm gate**
(f32) — the CPU f64 anchor is untouched and stays the bit-exact merge gate. The
kernel + wrapper are `#[cfg(feature="rocm")]`-gated, so the CPU build emits no atomic
codegen.

## Validation (real gfx1100)

- **Atomic correctness:** 50k rows → 4 bins (extreme contention), exact-integer data
  → result **bit-exact** to the known sums. No lost/double atomic updates.
- **Tolerance:** vs the cpu f64 anchor on real (non-integer) f32 data, **max relative
  error 3.8e-7** (< 1e-6).
- **Speed (isolated, 20k-row leaf):** single-unit **5940µs** → parallel **392µs** =
  **15.2× faster**.
- Default CPU merge gate GREEN (oracle-harness bit-exact); `lgbm` facade suite (41)
  still passes under `--features rocm` (small exact-integer corpora are exact in f32).

## End-to-end GPU training (honest)

| size | GPU single-unit (kfu) | GPU + parallel hist (this) | CPU native (M5c) |
|------|----------------------|----------------------------|------------------|
| small  | 8.28s | **6.16s** (−26%) | 38.7ms |
| medium | 40.8s | **22.9s** (−44%) | 258ms |

The histogram is now fast, but the GPU is still ~160× slower than CPU native because
`find_best_split`, `subtract`, and `data_partition` are **still single-unit** on the
GPU — each a per-(feature,leaf) GPU dispatch, and the launch overhead dominates. The
histogram kernel was the scoped target and it's done; the GPU won't be competitive
until the rest is parallelized + the launch count is cut.

## Next levers (follow-ups)

1. **Parallel `find_best_split`** — the running-sum scan → a parallel prefix-sum over
   bins (per-cube, shared memory). This is the current end-to-end bottleneck.
2. **Cut launch count** — batch all features per leaf into one launch; keep the binned
   dataset device-resident across iterations (avoid per-op host↔device round-trips).
   This is what actually makes a histogram GBDT fast on a GPU.
3. (Optional) privatized shared-memory sub-histograms to cut global-atomic contention
   for few-bin/high-row leaves.
