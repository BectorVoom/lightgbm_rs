---
title: Root cause of training speed across wide/tall regimes — both backends
date: 2026-06-22
context: investigation (post spike-015) — "why is learning slow in wide and tall data", CPU + GPU
---

# Wide/tall training speed — two-backend root cause

All prior perf work (R2/R3/R4, spikes 002–005) was validated at ≤20k×50. This profiles
the wide (×500) and tall (×1M) regimes the campaign never measured, on BOTH backends,
vs C++ LightGBM 4.6 (both multi-threaded, 16 cores, iters=8, leaves=31, bins=128).

## CPU backend — near-parity at wide+tall; the real gap is tall-narrow

| regime | Rust full train | C++ full (bin+train) | end-to-end | per-iter learn gap |
|--------|-----------------|----------------------|------------|--------------------|
| wide+tall 1M×500 | 7.97s | 8.28s (4.98 train + 3.30 bin) | **~parity** | 1.09× |
| tall-narrow 1M×50 | 1.34s | 0.95s (0.62 + 0.33 bin) | 1.41× | **1.73×** |

- **The 40–80× gap is GONE.** At scale Rust CPU is ~1.6–2.2× C++ raw, and **end-to-end
  at wide+tall it's at parity** — Rust **binning is FASTER than C++** (2.06s vs 3.30s at
  1M×500; 0.23s vs 0.33s at 1M×50). R2/R3/R4 transferred to scale.
- **The real gap is tall-narrow (~1.7× per-iter).** Root cause located:
  **`DataPartition::split` (data_partition.rs:108–148) is single-threaded** — a plain
  `for` over rows — while **C++ parallelizes it**: `#pragma omp parallel for
  num_threads(...) schedule(static, 512) if (num_data_ >= 1024)` (data_partition.hpp:55).
  At 1M×50 partition is **29% of train** and serial; with only 50 features the
  rayon-over-features build can't hide it. At wide+tall partition is ~5% (dwarfed by the
  500-feature build) → why wide+tall is near-parity but tall-narrow isn't.
- **CPU phase shares:** tall 1M×50 = build 69% / partition 29% / scan 2%; wide & wide+tall
  ≥100 feat = build+scan FUSED ~95% (the `build=0` is the `unified_bfs_threshold=100`
  fusion artifact) / partition ~5%.
- Secondary: `in_learner_other` ≈ 1.1s/train (~21% of tree-learning) at wide+tall = per-leaf
  scratch `Vec` alloc in the unified fused path (a reuse-the-scratch micro-lever).

**CPU lever:** parallelize `DataPartition::split` (rayon over row chunks, deterministic
static schedule → bit-exact, same pattern as the histogram parallel build). Targets the
29% serial partition at tall shapes. See [[parallelize-data-partition-split]].

## GPU backend — loses to the 16-core CPU everywhere; crossover erased

Clean train walls (this session) + build/scan attribution (`LGBM_SCAN_DRAIN=1`):

| shape | CPU 16-core | GPU gfx1100 | GPU vs CPU | GPU build share |
|-------|-------------|-------------|------------|-----------------|
| tall 1M×50      | 1.34s | 2.16s  | 1.6× slower | 84% |
| wide 100k×500   | 0.90s | 7.52s  | **8.4× slower** | 83% |
| wide+tall 1M×500| 7.97s | 14.98s | 1.9× slower | 92% |
| 2M×50           | 2.47s | 4.31s  | 1.74× slower | — |
| 5M×50           | 6.48s | 10.38s | 1.60× slower | — |

- **The f32-atomic BUILD dominates every GPU regime** (83–92%, grows with rows) — the
  spike-015 finding, now confirmed across tall/wide/wide+tall. Scan round-trip 8–16%,
  marshal ~0%, resident upload ~3%.
- **The GPU loses to the 16-core CPU at EVERY tested shape** — 1.6× (tall-narrow, best)
  to 8.4× (wide-short, worst: 500 atomic builds × too-few rows to amortize).
- **The crossover that justified the GPU track is GONE.** Spike-001 found GPU wins
  ≳700k rows — but vs **single-thread** CPU. R4 made the CPU 16-core, erasing it. **Even
  5M×50 is 1.6× slower with a FLAT ratio** (1.74×@2M → 1.60×@5M) — the GPU never catches
  up; both scale ~linearly and the GPU's atomic-build throughput sits ~1.6× under the
  16-core CPU.
- **Root cause:** GPU build is atomic-contention-bound (~820 Mr/s, spike-006/007); the
  CPU's columnar-u8 cache-resident feature-parallel build sustains more. gfx1100's
  row-partition occupancy lever can't fire at wide (`768/500=1`→P=1).

**Strategic implication:** the ROCm backend is currently **parity-maintenance, not a
speed path** — it holds the ~1e-6 contract but beats the CPU at no tested training shape.
Only revival lever: finer per-warp LDS sub-histogram privatization (spike-015) to lift
the atomic ceiling above the 16-core CPU — uncertain ROI; must clear a stable ~1.6×
deficit even in the best regime. (Caveat: training wall-clock only; the ~1e-6 parity
contract is a separate valid reason the backend exists; inference/very-high-cardinality
shapes untested.)

## Reproduce

```
# Rust CPU/GPU phase profile (per regime):
LGBM_PHASE_PROF=1 LGBM_SCAN_PROF=1 LGBM_SCAN_DRAIN=1 LGBM_BENCH_SWEEP=wide \
  LGBM_BENCH_ROWS=1000000 LGBM_BENCH_FEAT=50 cargo run --release [--features rocm] --example bench_gpu_vs_cpu
# C++ 4.6 ref (16 threads): /tmp/bench_cpp_wide_tall.py + /tmp/bench_cpp_binning.py (throwaway, .venv/bin/python)
```
