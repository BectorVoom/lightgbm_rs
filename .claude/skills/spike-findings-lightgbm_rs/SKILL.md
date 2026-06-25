---
name: spike-findings-lightgbm_rs
description: Implementation blueprint from the lightgbm_rs train-speed perf campaign (spikes 001-033). Proven CPU histogram-build wins (once-gather, u8 bins, feature-parallel, pool flatten+reuse), the GPU build/scan kernel campaign (u64 fixed-point atomics SHIPPED, feature-per-lane scan SHIPPED; per-warp-replication & within-feature-scan parity-resolved-but-ROI-gated), the wide-build uncoalesced-gather re-attribution, the PARTITION row-routing arc (fuse-gather + one-gather-fold SHIPPED on CPU, narrow-upload SHIPPED on ROCm; parallelize/double-buffer/prefetch NULL — partition is memory-bound, cut traffic not cores), GPU-vs-CPU routing, and the bit-exact + measurement rules. Auto-loaded during training-path performance work.
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
2026-06-25 (cont.) — spikes **023/024** (GPU scan round-trip: regime-split attribution +
**sibling-scan co-pack**, ~2× isolated bit-exact, WIRED phase 12) and **026–029** (the
**PARTITION row-routing arc**): partition is memory-bandwidth-bound on shared DDR5, so
parallelizing it (026 cubecl-cpu / the reverted ia0 rayon) is NULL — the lever is to CUT
TRAFFIC. **Fuse the per-leaf bin gather + a ¼-width u8 route scratch SHIPPED on CPU** (027,
1.3–2.7×, bit-exact, the #1 remaining CPU-vs-C++ gap), and **narrow the GPU per-split upload
u32→native-width SHIPPED on ROCm** (029, ~1.2–1.7×, bit-exact on GPU — disproving "APU
shared-memory transfer is free"). Dropping the copy-back via double-buffer is NULL (028, the
copy-back is only 2–7%).
2026-06-25 (cont.) — spikes **030/031** (BUILD re-attribution: post-u64 the wide build is
**uncoalesced-bin-gather-bound** 86–95%, NOT atomic [015 stale] or grad/hess; the stable-partition
monotone `leaf_rows` order already banks ~70% of coalescing ⇒ build effectively tuned on the APU,
031 closed) and **022b/032/033** (closing the partition + within-scan arcs): **032 folds the
redundant validation gather** in the shipped 027 `split_fused_host` (two random gathers → ONE,
~1.14–1.41× U8, bit-exact, SHIPPED quick-260625-qn9) — found by *auditing the shipped wiring vs the
spike that shipped it*; **033 prefetch** is ROI-gated DON'T-WIRE (helps only when the bin column ≫
LLC = U32/multi-M rows; null at the production U8 width + an x86-only intrinsic + a typed-slice
autovectorization regression); **022b** perf-disproves the within-feature cooperative scan (beats
021 only at narrow ≤256 feat, wash-to-regression at the wide production shape). **Net: the CPU host
partition is DONE — no remaining positive-ROI lever on this hardware.**
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
| **GPU build — bottleneck RE-ATTRIBUTION (current, read first)** | references/gpu-build-bottleneck-reattribution.md | **Post-u64 the wide build is UNCOALESCED-BIN-GATHER-latency-bound (030, 86–95%) — NOT atomic-bound (015 STALE: atomic ~0% after u64) nor grad/hess-bound (8–14%). `COAL_BIN` proved it: same array read SEQUENTIALLY = 8–20× faster. BUT the stable-partition MONOTONE `leaf_rows` already banks ~70% of coalescing (random probe overstates 5–10×) ⇒ only ~1.4× residual, read-once-unamortizable (cf 028) ⇒ build is effectively tuned on the APU; reopens only on discrete gfx110x (re-run the probe there).** Supersedes the atomic framing below |
| GPU build — fixed-point atomics & contention _(NOTE: the "atomic-bound" premise is pre-u64; see the re-attribution ref above)_ | references/gpu-build-fixedpoint-atomics.md | Wide build is LDS-atomic-contention bound (015, ~820 Mr/s, grows w/ rows). The ONE win: f32→**u64 fixed-point integer atomics** (018/019, SHIPPED) — ~1.3–1.7× in heavy-load regime (f32 atomicAdd = CAS-retry loop; integer ds_add_u64 native) + ~3600× accuracy + deterministic; composes with row-partition. Per-warp LDS replication (017 f32 ~1.1× / 020 u64) **regresses at production P=1 — NULL**. `Atomic<i64>` broken in cubecl-hip 0.10 (use u64 two's-complement) |
| GPU split-scan — occupancy & within-feature parallelism | references/gpu-split-scan-occupancy.md | Post-u64 the per-leaf scan is ~half the cost; it was `CubeDim(1)` single-thread/feature. **Feature-per-lane (W=64) SHIPPED** (021, bit-exact, isolated scan ~3×, e2e ~1.27× Amdahl-capped by the build). Within-feature parallel scan (016/022) is **parity-SAFE within ~1e-6** (default_left flips cosmetic; gain gap linear in default-bin mass) AND **perf-disproven (022b** — cooperation beats shipped 021 only at narrow ≤256 feat, wash-to-regression at the wide production shape once the occupancy confound is removed); **don't wire**. Re-profile after every build change (the bottleneck moves) |
| GPU scan round-trip — attribution & sibling co-pack | references/gpu-scan-roundtrip-copack.md | Post-021 the GPU per-tree round-trip is REGIME-SPLIT (023): ~59 scan-readback SYNCS/tree (one per leaf-node, both siblings scanned separately) — the SYNC floor is the reclaimable residual launch-bound (small/medium), build-compute dominates+grows wide (96.5%@1M×500). Subtract trick already on-device (closed). **Sibling-scan co-pack (024) = ~2× isolated, bit-exact, WIRED phase 12** behind `LGBM_SIBLING_COPACK` (59→~30 syncs/tree; honest e2e ~10–15% small/medium, ~1.5% wide). Use `LGBM_SCAN_DRAIN` to de-alias build vs scan-sync; ROCm-parity-track |
| Partition (row-routing) — memory-traffic & narrow-upload | references/partition-memory-traffic.md | Partition is MEMORY-BANDWIDTH-bound (shared DDR5) — **cut traffic, don't add cores**. SHIPPED: fuse the gather + ¼-width u8 route scratch (027, **1.3–2.7× CPU, bit-exact**, the #1 CPU-vs-C++ gap); narrow the GPU upload u32→native-width via generic-over-Int kernel + `data_partition_native` (029, **~1.2–1.7× rocm, bit-exact on GPU**). NULLs: parallelize partition (026 — rayon/cubecl-cpu both lose, bandwidth-bound; "APU transfer free" is FALSE); double-buffer to drop the copyback (028 — copyback only 2–7%, C++ copies back too). **032: fold the redundant validation gather** in shipped 027 (two random gathers → ONE, ~1.14–1.41× U8, bit-exact, SHIPPED qn9 — audit the shipped wiring vs the spike). **033 prefetch ROI-gated DON'T-WIRE** (helps only when column ≫ LLC = U32/multi-M rows; null at U8; + a typed-slice→AVX-gather autovectorization regression: keep `.bin()` per-row matches). Backend gate via default-false trait methods (`prefers_host_partition`/`data_partition_native`), never a global. **CPU partition DONE.** |

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
- 023-post-021-roundtrip-attribution (VALIDATED measurement — GPU per-tree round-trip is REGIME-SPLIT; ~59 scan-readback syncs/tree, SYNC floor reclaimable launch-bound, build dominates+grows wide 96.5%@1M×500; subtract trick already on-device; host partition grows to 23%)
- 024-batch-sibling-scans (VALIDATED ~2× isolated + bit-exact — co-pack both siblings into ONE launch+readback; honest e2e ~10–15% small/medium, ~1.5% wide; WIRED phase 12 behind LGBM_SIBLING_COPACK)
- 026-cubecl-cpu-partition-scan-scatter (PARTIAL/NULL — bit-exact cubecl-cpu scan+scatter loses to serial-native except cache-resident ~100k; partition is memory-bandwidth-bound on shared DDR5, parallelism null at scale; reframes the ia0 rayon null — wall is DRAM bandwidth not build contention)
- 027-fused-gather-partition (VALIDATED + SHIPPED — fuse the gather + ¼-width u8 route scratch = 1.3–2.7× CPU, bit-exact, biggest ~2.3× at U8; the #1 remaining CPU-vs-C++ gap; quick-260625-hw2)
- 028-doublebuffer-partition (INVALIDATED/NULL — copy-back is only 2.2–6.8% of the fused op; double-buffer to drop it is within noise + doubles partition memory + cross-leaf bookkeeping; C++ copies back too)
- 029-gpu-narrow-upload-fuse (VALIDATED + SHIPPED — narrow the per-split GPU upload u32→native-width via generic-over-Int kernel + additive data_partition_native = ~1.2–1.7× rocm, bit-exact on GPU; "APU shared-DDR5 transfer is free" disproven; quick-260625-j1l)
- 030-wide-build-roofline-reattribution (VALIDATED measurement — remove-the-suspect A/B re-attributes the post-u64 wide build: UNCOALESCED-BIN-GATHER-bound 86–95%, atomic ~0% [015 STALE], grad/hess 8–14%; COAL_BIN [same array, sequential] = 8–20× faster ⇒ it's the access PATTERN not bandwidth; REAL_ORDER caveat — stable-partition monotone `leaf_rows` already ~70% of coalesced ceiling, random probe overstates 5–10× ⇒ ~1.4× residual; reopens only on discrete gfx110x)
- 031-crossfeature-gradhess-reuse (CLOSED by 030, not built — original grad/hess-reuse premise INVALIDATED [grad/hess only 8–14%]; redirect [coalesce the bin read] = ~1.4× ceiling, read-once-unamortizable [cf 028] ⇒ ROI marginal on APU; re-run the 030 probe on discrete gfx110x before prototyping)
- 022b-within-feature-scan-perf-ab (VALIDATED experiment — confirms DON'T WIRE: cooperative within-feature scan beats shipped 021 only at NARROW ≤256 feat [6×→2.2×], WASH-to-regression at the WIDE F=512 production shape once the cd256 occupancy confound is removed [real baseline = cd64 K1]; argmax mism=0, gainrel ≤9e-15 confirms 022's parity on the real kernel)
- 032-partition-validation-fold (VALIDATED + SHIPPED quick-260625-qn9 — reading the shipped 027 `split_fused_host` found TWO random gathers [standalone validation + route], not the ONE 027 measured; fold the range-check INTO pass-1 [gather b once, check before any write, bit-exact on success AND error] = ~1.14–1.41× U8 / up to ~1.8× U32 at scale, 3 restarts parity OK; lesson: audit the shipped wiring vs the spike that shipped it)
- 033-partition-gather-prefetch (PARTIAL — DON'T WIRE, ROI-gated: software-prefetch hides gather miss-latency only when the bin column ≫ LLC [~2–3× whole-op at 4M×U32, bestD=128], null-to-marginal at the production U8 width [~1.1× at a root split only]; x86-only intrinsic; SURPRISE landmine — hoisting `.bin()` into a typed-`&[T]` loop auto-vectorizes to a slow AVX gather 1.5–2× SLOWER than scalar `.bin()`)

Full campaign 001–013 wrapped (2026-06-17); 014a/014b + shipped p9v/qix/rdu/rsh levers wrapped (2026-06-21, GPU wide-shape 1M×500, cumulative −68%); 015–022 GPU build/scan KERNEL campaign wrapped (2026-06-25 — u64 fixed-point build + feature-per-lane scan SHIPPED; replication & within-feature scan parity-characterized but ROI-gated); 023/024 + 026–029 wrapped (2026-06-25 — GPU scan round-trip attribution + sibling co-pack WIRED, and the PARTITION memory-traffic arc: fuse-gather SHIPPED CPU + narrow-upload SHIPPED ROCm, parallelize/double-buffer NULL). 030/031 wrapped (2026-06-25 — BUILD bottleneck RE-ATTRIBUTION: post-u64 the wide build is uncoalesced-bin-gather-bound, not atomic/grad-hess; the stable-partition monotone order already banks ~70% of coalescing so the build is effectively tuned on the APU — reopens only on discrete gfx110x). 022b/032/033 wrapped (2026-06-25 — closing the partition + within-scan arcs: 032 one-gather validation fold SHIPPED on CPU [~1.14–1.41× U8, bit-exact, quick-260625-qn9; found by auditing the shipped wiring vs the spike]; 033 gather-prefetch PARTIAL/DON'T-WIRE [helps only when column ≫ LLC = U32/multi-M rows; + a typed-slice autovectorization landmine]; 022b perf-disproves the within-feature cooperative scan [wins only narrow ≤256 feat]. **The CPU host partition is DONE — no positive-ROI lever remains on this hardware.**).
</metadata>
