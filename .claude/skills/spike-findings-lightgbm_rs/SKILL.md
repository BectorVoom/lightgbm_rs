---
name: spike-findings-lightgbm_rs
description: Implementation blueprint from the lightgbm_rs train-speed perf campaign (spikes 001-022). Proven CPU histogram-build wins (once-gather, u8 bins, feature-parallel, pool flatten+reuse), the GPU build/scan kernel campaign (u64 fixed-point atomics SHIPPED, feature-per-lane scan SHIPPED; per-warp-replication & within-feature-scan parity-resolved-but-ROI-gated), GPU-vs-CPU routing, and the bit-exact + measurement rules. Auto-loaded during training-path performance work.
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
the real costs. 2026-06-25 — spikes **015–022** (the GPU build/scan KERNEL campaign):
post-014 the wide bottleneck is the atomic-bound histogram BUILD → **u64 fixed-point
integer atomics SHIPPED** (~1.3–1.7× + 3600× accuracy + deterministic); the per-leaf split
SCAN was `CubeDim(1)` single-threaded-per-feature → **feature-per-lane (W=64) SHIPPED**
(bit-exact, ~3× isolated). Per-warp LDS replication (017/020) and within-feature parallel
scan (016/022) are parity-characterized but ROI-gated (the spoofed 8-CU APU loses to the
CPU anchor everywhere — this track is ROCm-parity maintenance, not overall-fastest).
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
| GPU build — fixed-point atomics & contention | references/gpu-build-fixedpoint-atomics.md | Wide build is LDS-atomic-contention bound (015, ~820 Mr/s, grows w/ rows). The ONE win: f32→**u64 fixed-point integer atomics** (018/019, SHIPPED) — ~1.3–1.7× in heavy-load regime (f32 atomicAdd = CAS-retry loop; integer ds_add_u64 native) + ~3600× accuracy + deterministic; composes with row-partition. Per-warp LDS replication (017 f32 ~1.1× / 020 u64) **regresses at production P=1 — NULL**. `Atomic<i64>` broken in cubecl-hip 0.10 (use u64 two's-complement) |
| GPU split-scan — occupancy & within-feature parallelism | references/gpu-split-scan-occupancy.md | Post-u64 the per-leaf scan is ~half the cost; it was `CubeDim(1)` single-thread/feature. **Feature-per-lane (W=64) SHIPPED** (021, bit-exact, isolated scan ~3×, e2e ~1.27× Amdahl-capped by the build). Within-feature parallel scan (016/022) is **parity-SAFE within ~1e-6** (default_left flips cosmetic; gain gap linear in default-bin mass) but **ROI-gated** — only helps narrow, the GPU's weakest regime; don't wire. Re-profile after every build change (the bottleneck moves) |

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
- 015-parallel-f32-resident-build (PARTIAL/located — wide bottleneck = atomic-bound BUILD 86→92%, growing w/ rows; scan round-trip ≤14% & shrinking; `LGBM_SCAN_PROF`/`DRAIN` tooling)
- 016-parallel-scan-reorder-parity (PARTIAL — threshold stable under reorder; default_left flips deferred → resolved by 022)
- 017-perwarp-lds-replication (VALIDATED modest ~1.1× on f32, NOT wired — superseded by u64)
- 018-fixedpoint-int-atomics (VALIDATED strong — u64 fixed-point atomics, the build win; SHIPPED)
- 019-int-atomic-contention-regime (VALIDATED + corrects 018 — ~1.3–1.7× heavy-load, composes with row-partition)
- 020-perwarp-replication-on-u64 (PARTIAL/null-leaning — wins only at P=16, REGRESSES at production P=1; DON'T WIRE)
- 021-scan-feature-per-lane-occupancy (VALIDATED + SHIPPED — CubeDim(1)→W=64, bit-exact, isolated scan ~3× / e2e ~1.27×)
- 022-within-feature-parallel-scan-parity (VALIDATED — parity GATE resolved PARITY-SAFE ~1e-6; all default_left flips cosmetic; ROI-gated, don't wire)

Full campaign 001–013 wrapped (2026-06-17); 014a/014b + shipped p9v/qix/rdu/rsh levers wrapped (2026-06-21, GPU wide-shape 1M×500, cumulative −68%); 015–022 GPU build/scan KERNEL campaign wrapped (2026-06-25 — u64 fixed-point build + feature-per-lane scan SHIPPED; replication & within-feature scan parity-characterized but ROI-gated).
</metadata>
