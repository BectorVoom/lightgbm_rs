---
name: spike-findings-lightgbm_rs
description: Implementation blueprint from the lightgbm_rs train-speed perf campaign (spikes 001-013). Proven CPU histogram-build wins (once-gather, u8 bins, feature-parallel, pool flatten+reuse), GPU kernel findings (row-partition lever; u8/packing/quant nulls), GPU-vs-CPU routing, and the bit-exact + measurement rules. Auto-loaded during training-path performance work.
---

<context>
## Project: lightgbm_rs

Pure-Rust LightGBM port (CubeCL, CPU + ROCm). The CPU f64 anchor must stay bit-exact to
C++ LightGBM 4.6. These findings come from the train-speed perf campaign: localizing the
histogram-build bottleneck and stacking bit-exact CPU wins, probing the GPU kernel's real
bottleneck, and bounding what the GPU and quantization tracks can claim.

Spike sessions wrapped: 2026-06-17 — spikes **001–013** (full campaign); 2026-06-21 —
spikes **014a/014b** (GPU wide-shape 1M×500 attribution + the p9v/qix/rdu/rsh host-setup
levers, cumulative −68%). The histogram BUILD dominates CPU train; on the GPU wide shape
the kernel is only ⅓ — redundant device upload + cache-hostile per-train host setup were
the real costs.
</context>

<requirements>
## Requirements

- The CPU f64 anchor stays **bit-exact to C++** — gate every change with
  `cargo test -p oracle-harness` (esp. `raw_bin_train_matches_cpp_golden`),
  `cargo test -p lgbm-treelearner --lib`, `cargo test -p lgbm`.
- `Vec<Vec<T>>` is **not categorically a pessimization** — decide per-instance by usage
  (the decision rule in the reference). Do not blanket-flatten.
- **Ship on the end-to-end `bench_train` number, not the isolated microbench** — the cold
  ceiling overstates the warm win 3–7×.
</requirements>

<findings_index>
## Feature Areas

| Area | Reference | Key Finding |
|------|-----------|-------------|
| CPU histogram build (the bottleneck) | references/cpu-histogram-build.md | Build is 63–90% of train + 5.2× slower than C++ at low rows (002); four stacked bit-exact wins — once-per-leaf gather (003, −33/−39%), fused-branchless (003b), narrow u8/u16 bins (004, large −49%), feature-parallel ≥16384 rows (005, large −26%) |
| GPU (ROCm) histogram kernel | references/gpu-histogram-kernel.md | Build is atomic/latency-bound, not bandwidth-bound — row-partition to ~8 wkgrps/CU is the ONE lever (007, ~1.35×); u8 device bins (006) and multi-feature packing (009) are evidenced NULLs |
| GPU routing & quantization | references/gpu-routing-and-quantization.md | GPU crosses CPU ≈700k rows vs single-thread (001) — but moves to millions vs the multi-threaded anchor; int16 quantized hist is irreducibly approximate (008, ~3e-4 floor), opt-in mode only |
| Histogram & learning-path memory layout | references/histogram-learning-memory-layout.md | Flatten + reuse the histogram pool (~7% large, bit-exact, shipped); keep the parallel-build per-thread accumulators (load-bearing) and the KB-scale bool matrix (sub-noise); per-leaf rows are already flat |
| GPU wide-shape attribution & host-setup levers | references/gpu-wide-shape-attribution.md | At 1M×500 the histogram kernel is ≤⅓ of train (folded into "scan"; `build=0` is an artifact) — profile with the whole-train BUDGET (`LGBM_PHASE_PROF=1`). Four shipped bit-exact levers: upload-once-per-train (p9v −32%), native-width upload (qix ~5×/~4× mem), cache-friendly feature_infos (rdu ~8×) + binning (rsh ~2.3×) via transpose; cumulative 29.55→~9.5s (−68%). Measure before "fixing" a hypothesis (the to_vec-clones lever was a mis-attribution) |

## Cross-cutting rules (read these first)

- **Bit-exact gate** every CPU-anchor change: `cargo test -p lgbm-compute --lib` +
  `-p lgbm-treelearner --lib` + `-p oracle-harness` (esp. `raw_bin_train_matches_cpp_golden`,
  `learner_parity`).
- **Ship on end-to-end `bench_train`/`bench_crossover`, not the isolated microbench** — the
  cold isolated ceiling overstates the warm win 3–7× (allocator amortization).
- **Probe in an isolated A/B before plumbing** a multi-kernel change; sweep the size;
  2–3 process restarts to kill warmup drift. This discipline killed 3 duds cheaply (006/008/009/011).
- **`Vec<Vec<T>>` is not categorically bad** — flatten MB-scale `vec![template;n]`, keep
  per-thread accumulators and KB-scale structures (see the memory-layout reference).

## Source Files

Original spike READMEs (001–013) + the `#[ignore]`d microbenches and the 002 C++ A/B
harness (`gen_data.py`/`train.conf`) are preserved in `sources/`.
</findings_index>

<metadata>
## Processed Spikes

- 001-gpu-cpu-crossover (VALIDATED — GPU wins ≳1M rows vs single-thread)
- 002-lowrow-phase-ab (VALIDATED — gap is histogram build, 5.2×)
- 003-columnar-hist-build (VALIDATED + SHIPPED — once-gather)
- 004-columnar-u8-bins (VALIDATED + SHIPPED — narrow bins, large −49%)
- 005-feature-parallel-build (VALIDATED + SHIPPED — rayon ≥16384, large −26%)
- 006-gpu-u8-bins (INVALIDATED — ~0%, GPU not bandwidth-bound)
- 007-row-partitioned-histogram-build (VALIDATED — ~1.35× at ~8 wkgrps/CU)
- 008-16bit-discretized-hist (INVALIDATED for exact parity — opt-in only)
- 009-multifeature-per-cube (INVALIDATED — null at matched occupancy)
- 010-histogram-pool-arena (VALIDATED + SHIPPED — flat arena)
- 011-parallel-build-scatter (INVALIDATED — load-bearing)
- 012-reuse-pool-across-trees (VALIDATED + SHIPPED — reuse across trees)
- 013-feature-splittable-arena (INVALIDATED — sub-noise)
- 014a-coarse-phase-attribution (PARTIAL — GPU kernel folded into "scan"; <½ of wall-clock instrumented; overturns "kernel is the bottleneck")
- 014b-gpu-launch-vs-compute-split (VALIDATED — whole-train BUDGET names the cost: redundant per-tree resident-bin upload, ≈ the kernel; led to 4 shipped levers)

Full campaign 001–013 wrapped (2026-06-17); 014a/014b + shipped p9v/qix/rdu/rsh levers wrapped (2026-06-21, GPU wide-shape 1M×500, cumulative −68%).
</metadata>
