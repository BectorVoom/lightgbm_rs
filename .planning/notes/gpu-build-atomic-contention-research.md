---
title: GPU BUILD-kernel atomic-contention speedup research (device-time proxy)
date: 2026-06-22
context: explore session — chase a real GPU wall-clock win, validated via CubeCL device-time as proxy
---

# GPU histogram-BUILD atomic-contention: speedup research

## Framing decided this session

- **Goal:** chase a real GPU wall-clock win on the f32-atomic histogram **BUILD**
  kernel — the dominant device-time cost (**53%** at 1M×50, per the CubeCL built-in
  profiler; see `cubecl-profiling-gpu-kernel-decomposition.md` and memory
  `gpu-bottleneck-now-seq-f64-scan`).
- **Measurement = CubeCL device-time as proxy** (option ii). Wall-clock CANNOT be
  validated on this box — it is a **spoofed 8-CU APU**, not discrete gfx1100
  (`gpu-is-spoofed-8cu-apu-not-gfx1100.md`). The just-shipped subtract
  parallelization (commit 6167c75, ~14× device-time, bit-exact) produced **zero
  wall-clock change here** — exactly because 8 CUs can't expose the parallelism.
  So we optimize to shrink BUILD's **profiler share** (53% → ?), knowing wall-clock
  stays flat until real silicon. Device-time share IS validatable on the APU.
- **Caveat carried in:** crossover is ERASED — the 16-core CPU beats the GPU at every
  shape, no convergence to 5M rows. This is a parity-maintenance / future-silicon
  track, not a path to beat the CPU here.

## The kernel's failure mode

Atomic-contention-bound: each thread does an **f32 atomic `fetch_add`** into per-bin
(sum_gradient, sum_hessian) accumulators in LDS. Already LDS-privatized at the
workgroup/cube level (one shared sub-histogram per workgroup). Throughput ~820 M
reads/s — far below bandwidth — because atomic adds **serialize on bin collisions**.
Bins per feature ≤256 (typically 255).

## Web research findings (2026-06-22) — mapped onto our constraints

| # | Technique | Mechanism | Verdict for us |
|---|-----------|-----------|----------------|
| 1 | **Replicated sub-histograms** (R copies in LDS, hash thread→copy, merge) | cuts collision prob ~R× | **Capacity-limited.** 255 bins × (f32,f32) ≈ 4 KB/copy → only a handful fit LDS. Helps most at ≤64 bins. This is WHY LightGBM ships `histogram16/64/256.cl`. Weak lever for wide/255-bin. |
| 2 | **Warp-aggregated atomics** (`match_any`/ballot+shuffle: reduce same-bin lanes intra-warp, 1 atomic per distinct bin) | up to 32× fewer atomics, **bin-count-independent** | **STANDOUT.** Vehicle = CubeCL **`Plane` API** (already used in project). Sidesteps the capacity wall. Determinism preserved if reduction order fixed. |
| 3 | **Integer/fixed-point LDS atomics** | AMD native int-atomic path faster than f32; **order-independent (deterministic)** | **Double win in principle, but snags.** GPU int kernel was already **null (W5)** for quantized training; fixed-point adds quantization error vs f32 anchor. ROCm only owes ~1e-6, so NOT dead — but a parity-budget question (see research Q2). |
| 4 | Two-phase block-local → single batched global merge | removes global-atomic contention entirely | Likely already in shape (single workgroup merge). Verify no per-bin global atomics. |
| 5 | **AMD LDS bank-conflict layout** (32 banks; XOR-swizzle, interleave sum_g/sum_h) | avoids 32-way bank conflict on strided bin access | **Cheap, always-on, parity-neutral.** Layer underneath (2). |
| 6 | Replication factor R tunable from bin count + collision degree (SC20) | balances contention vs LDS vs merge cost | Sweepable knob IF we pursue (1) at low bin counts. |
| 7 | Contention (not bandwidth) is the modeled bottleneck | confirms 820 M reads/s is serialization-gated | Validates that (1)+(2) target the right lever. |

## Emerging plan

**Warp-aggregated atomics via `Plane` API + bank-conflict-free LDS layout**, measured
as BUILD profiler-share drop at 1M×50. Feasibility-gated on whether CubeCL's `Plane`
exposes `match_any`-style intra-warp bin matching. → spike
`warp-aggregated-histogram-atomics`.

## UPDATE 2026-06-22 — spike resolved (both warp levers now closed)

- **Finding #2 (warp-aggregation): already CLOSED before this note.** quick-260619-p93
  built + hardware-benched the `Plane` ballot+shuffle kernel → **NULL/NEGATIVE** (slower
  5/6 cells; at 256 bins ~30 distinct bins/wave = nothing to amortize). Kept unwired.
- **Finding #1 (per-warp replication): spike-017 → VALIDATED MODEST.** `gpu_lds_replication.rs`,
  comptime `replicas`. **R8 (=1 replica/warp) sign-stable ~1.1× device-time** (1.06–1.29×,
  2 process runs); R2/R4 null (win is all-or-nothing at the warp boundary). First positive
  GPU build-kernel lever. Parity within ~1e-5. **Kept as evidence, NOT wired** — modest +
  APU-only, CPU still beats GPU ~4× at wide. My earlier "capacity-limited at 255 bins"
  worry was about HIGH R; R8 at 256 bins = 16 KB ≤ 64 KB, fine. Full:
  `.planning/spikes/017-perwarp-lds-replication/README.md`.
- **Findings #3 (int atomics, research Q2) + #5 (bank-conflict layout): still open**, but
  lower priority given the routing reality (GPU is parity-maintenance, not a speed path).

## Sources

- NVIDIA shared-atomics histograms — https://developer.nvidia.com/blog/gpu-pro-tip-fast-histograms-using-shared-atomics-maxwell/
- CUDA warp-aggregated atomics — https://developer.nvidia.com/blog/cuda-pro-tip-optimized-filtering-warp-aggregated-atomics/
- Voting & shuffling — https://developer.nvidia.com/blog/voting-and-shuffling-optimize-atomic-operations/
- Henriksen et al., generalized histograms, SC20 — https://hjemmesider.diku.dk/~zgh600/Publications/gen-histo-sc20.pdf
- ROCm GPU atomics support — https://rocm.docs.amd.com/en/latest/reference/gpu-atomics-operation.html
- AMD LDS bank-conflict blog — https://rocm.blogs.amd.com/software-tools-optimization/lds-bank-conflict/README.html
- HIP histogram tutorial — https://rocm.docs.amd.com/projects/HIP/en/develop/tutorial/programming-patterns/atomic_operations_histogram.html
- Quantized GBDT training — https://arxiv.org/pdf/2207.09682
- Modeling shared-memory atomic bottlenecks — https://arxiv.org/html/2503.17893v1
