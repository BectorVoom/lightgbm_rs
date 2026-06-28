---
name: spike-findings-lightgbm_rs
description: Implementation blueprint from the lightgbm_rs train-speed perf campaign (spikes 001-050). Proven CPU histogram-build wins (once-gather, u8 bins, feature-parallel, pool flatten+reuse), the GPU build/scan kernel campaign (u64 fixed-point atomics SHIPPED, feature-per-lane scan SHIPPED; per-warp-replication & within-feature-scan parity-resolved-but-ROI-gated), the wide-build uncoalesced-gather re-attribution, the PARTITION row-routing arc (fuse-gather + one-gather-fold SHIPPED on CPU, narrow-upload + route-partition-on-host-by-default SHIPPED on ROCm; parallelize/double-buffer/prefetch NULL — partition is memory-bound, cut traffic not cores), the GPU branch-divergence don't-chase gate (kernels already branchless; divergence off the dominant path BUT cleanly sign-measurable on the spoofed APU — the one effect that survives the spoof), GPU kernel AUTOTUNING (CubeCL `cubecl::tune` works on cubecl-hip 0.10; accumulating kernels need a fresh-output InputGenerator or corrupt 27×; key on log2(rows) not exact rows; autotune BEATS the shipped 8-CU row_partition_count ~10% — measure-don't-model), the Vector<P,N> SIMD frontier (subtract SHIPPED, scan/build NULL — vectorize only memory-bound kernels), the FIRST real-discrete-CUDA attribution on Kaggle (the metric-eval fix SHIPPED −26%; GPU hist phases are the 53% architectural long-pole, Python marshalling the 25% next-easy-win, in_learner_other a diffuse dead-end; sync-floor/route-to-CPU/resident_reset all REFUTED on real NVIDIA), the re-attribute-after-every-wire rule (the bottleneck moved 4×), GPU-vs-CPU routing, and the bit-exact + measurement rules. Auto-loaded during training-path performance work.
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
2026-06-26 — spikes **034/035** (re-attribute-then-act, the GPU partition routing win). With the
co-pack (024) + narrow-upload (029) wires SHIPPED, **034 re-profiled and the bottleneck MOVED a 4th
time** (014→015→023→034): in the launch-bound regime the scan-sync floor is closed (scan 3–7%) and
the **device partition round-trip is now the #1 reclaimable phase (38% medium / 30% large)**; wide
stays build-dominated. **035 SHIPPED the fix** (quick-260626-a6t): route the rocm partition on the
HOST by default (`prefers_host_partition()` default-ON, off-switch `LGBM_ROCM_HOST_PARTITION=0`) —
~1.18–1.23× launch-bound, wash wide, parity within ~1e-6 (the device round-trip is pure overhead;
both paths land in host `indices_` so there is no index re-upload). **The rare GPU lever that wins on
the APU itself.** Gating it required fixing a latent FUSED-path `subtract_resident` bug (the Phase-12
co-pack scan deferral ran subtract before the fused smaller histogram was built; debug `8aed100`), and
re-wiring the inert `LGBM_SCAN_DRAIN` onto the co-pack scan fn (quick-260625-tw1). Method lesson
reinforced: **re-profile after every wire — a shipped lever moves the bottleneck.**
2026-06-26 (cont.) — spike **036** (the "optimize conditional branching in GPU kernels" GATE).
**Two findings, opposite signs.** (1) **Measurability = a reusable carve-out from the spoofed-APU
caveat:** a controlled-divergence ladder (4 arms, IDENTICAL total work, only the intra-wave
trip-count distribution differs) scales **near-ideal 1 : 1.9 : 3.7 : 27×** (restart-stable) ⇒
**wavefront lockstep-masking is FAITHFUL on the 8-CU APU** — divergence is the ONE GPU micro-arch
effect that survives the spoof and is cleanly sign-measurable (it's a scheduler property, not
CU-count/memory-bound — the confounded axes). (2) **Critical-path = WEAK ⇒ DON'T-CHASE:** the
kernels are ALREADY fully branchless (`select`-everywhere, cubecl-cpu MLIR forced it), so the only
live cross-lane divergence is the split-scan **loop-trip-count** imbalance — on the **3–7% scan**
phase, **zero** at the production all-256-bin cardinality, real only on mixed-cardinality data
(honest e2e ceiling ≪1%). The dominant wide **build** is uniform/divergence-free by construction
(030); partition is branchless + host-routed (035); the only heavily-divergent kernel is the p93
NULL path. **Recommendation: do not chase branch divergence as a general lever** (037 = bounded
mixed-cardinality curiosity; 038 break-vs-select = likely don't-build — `done` is intra-lane
predication). Bounded-don't-chase, the 030/031/033 shape.
2026-06-26 (cont.) — spikes **037–040** (the GPU-kernel-AUTOTUNING arc — replace the hand-tuned
launch-config heuristics with CubeCL `cubecl::tune`; NOTE the "037/038" numbers here are the
AUTOTUNE track, NOT the deferred divergence curiosities 036 mentioned). **All four VALIDATED.**
(037) Autotune compiles+runs+caches END-TO-END on cubecl-hip 0.10: in-proc hit ~6µs, persistent
disk cache across processes (`target/autotune/0.10.0/rocm_0/*.json.log`), and it independently
re-derived spike-007's P=16. **The `cubecl_manual` autotuning doc is WRONG on its 3 load-bearing
points — code from the SOURCE:** key-gen returns the `AutotuneKey` (not a String); `execute`'s
1st arg is the cache-namespace ID (not the key); the key needs `serde` derive (`std_io` always
on). (038) Accumulating kernels (histogram BUILD `fetch_add`) corrupt **27×** under
`CloneInputGenerator` (`Handle::clone` = ref-count bump, all reps hit the real `out`) → fix = a
**fresh-output `InputGenerator`** (winner's final run uses the original inputs ⇒ `rel_err 0` by
grad-conservation). Classify: OVERWRITE=safe, ACCUMULATE=fresh-out, in-place-RMW=deep-copy (but
partition is host-routed 035). (039) Keying the AutotuneKey on exact `rows` = a tuning STORM
(25/25 nodes cold, 975ms/tree); key on **`log2(rows)`** (occupancy regime) → ~3×, 20/25 free,
keeps the per-regime variant crossover. (040) **Autotune BEATS the shipped 8-CU heuristic ~10%
(NOT the predicted wash):** `row_partition_count(50,n)` resolves `target_cubes=64`→**P=1 at the
production 50-feat width** (the SLOWEST sweep point), the 8-CU correction over-corrected from the
phantom-96-CU P≈16; autotune picks P∈{4,8,16} and wins 2–16% sign-stable (3 restarts). Surfaced a
**latent production mis-tune** (recalibrate `row_partition_count` OR adopt autotune — the robust +
portability answer). Honest bound: ~10% on the spoofed-APU GPU build, which CPU beats e2e ⇒ the
deliverable is the METHOD (measure-don't-model) + portability. Example-only + a dev-dep; anchor
untouched. Reference: `gpu-kernel-autotuning.md`.
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
| GPU scan round-trip — attribution & sibling co-pack | references/gpu-scan-roundtrip-copack.md | Post-021 the GPU per-tree round-trip is REGIME-SPLIT (023): ~59 scan-readback SYNCS/tree (one per leaf-node, both siblings scanned separately) — the SYNC floor is the reclaimable residual launch-bound (small/medium), build-compute dominates+grows wide (96.5%@1M×500). Subtract trick already on-device (closed). **Sibling-scan co-pack (024) = ~2× isolated, bit-exact, WIRED phase 12** behind `LGBM_SIBLING_COPACK` (59→~30 syncs/tree; honest e2e ~10–15% small/medium, ~1.5% wide). Use `LGBM_SCAN_DRAIN` (needs `LGBM_SCAN_PROF=1` too; re-wired onto the co-pack scan fn in quick-260625-tw1) to de-alias build vs scan-sync; ROCm-parity-track. **034: post-co-pack the launch-bound bottleneck MOVED off the sync floor (scan 3–7%) to the device partition round-trip → see partition ref (035 SHIPPED).** LANDMINE: the co-pack scan deferral broke the FUSED `subtract_resident` (subtract ran before the fused smaller histogram was built; debug `8aed100` un-defers the smaller fused build) |
| **GPU branch / warp divergence (the don't-chase gate + APU-measurability carve-out)** | references/gpu-branch-divergence.md | **DON'T chase branch divergence as a general lever — kernels are already branchless (`select`-everywhere, cubecl-cpu MLIR forced it); the dominant wide build is divergence-free by construction, the only live divergence (split-scan trip-count) is on the 3–7% scan phase + zero at uniform cardinality. BUT divergence IS cleanly sign-measurable on the spoofed APU (036 ladder: near-ideal 1:2:4:32) — the one GPU effect that survives the spoof; reuse the ladder before any divergence A/B** |
| **GPU kernel AUTOTUNING (`cubecl::tune`) — feasible + beats the hand-tuned heuristic** | references/gpu-kernel-autotuning.md | **CubeCL autotune works END-TO-END on cubecl-hip 0.10 (037: compile/run/cache, in-proc 6µs + persistent disk cache; re-derived spike-007 P=16). Code from the SOURCE — the manual is wrong on 3 points (key-gen returns the key not a String; `execute`'s id≠key; serde-derive mandatory). Accumulating kernels need a FRESH-OUTPUT InputGenerator or they corrupt 27× (038); key on `log2(rows)` not exact rows or it's a tuning STORM (039). 040: autotune BEATS the shipped 8-CU `row_partition_count` ~10% — it under-partitions to P=1 at the production 50-feat width (latent mis-tune). Measure-don't-model + portability; ~10% bounded on the spoofed-APU GPU build (CPU wins e2e).** |
| Partition (row-routing) — memory-traffic & narrow-upload | references/partition-memory-traffic.md | Partition is MEMORY-BANDWIDTH-bound (shared DDR5) — **cut traffic, don't add cores**. SHIPPED: fuse the gather + ¼-width u8 route scratch (027, **1.3–2.7× CPU, bit-exact**, the #1 CPU-vs-C++ gap); narrow the GPU upload u32→native-width via generic-over-Int kernel + `data_partition_native` (029, **~1.2–1.7× rocm, bit-exact on GPU**). NULLs: parallelize partition (026 — rayon/cubecl-cpu both lose, bandwidth-bound; "APU transfer free" is FALSE); double-buffer to drop the copyback (028 — copyback only 2–7%, C++ copies back too). **032: fold the redundant validation gather** in shipped 027 (two random gathers → ONE, ~1.14–1.41× U8, bit-exact, SHIPPED qn9 — audit the shipped wiring vs the spike). **033 prefetch ROI-gated DON'T-WIRE** (helps only when column ≫ LLC = U32/multi-M rows; null at U8; + a typed-slice→AVX-gather autovectorization regression: keep `.bin()` per-row matches). Backend gate via default-false trait methods (`prefers_host_partition`/`data_partition_native`), never a global. **CPU partition DONE.** **034→035: post-co-pack the device partition round-trip became the GPU's #1 launch-bound phase (30–38%) → route the rocm partition on the HOST by default (035, SHIPPED quick-260626-a6t, `prefers_host_partition()` default-ON / off-switch `LGBM_ROCM_HOST_PARTITION=0`): ~1.18–1.23× launch-bound, wash wide, parity within ~1e-6 (anchor-pinned, def-f8u-01; the device round-trip is pure overhead — build reads host indices either way). The rare GPU lever that wins on the APU.** |
| **Discrete-CUDA bottleneck attribution + the metric-eval fix (real NVIDIA, Kaggle)** | references/cuda-discrete-gpu-bottleneck.md | **FIRST real-discrete-GPU profile (046/048/049): lgb_rs CUDA ~5–6× official at 500k×50. Post-fix wall map: GPU hist phases 53% (architectural on-device-learner long-pole), Python marshalling 25% (UNATTRIBUTED, likely next easy win), in_learner_other 15% (DEAD END — diffuse, resident_reset REFUTED 0.3ms), metric 0% (FIXED). SHIPPED the metric fix (quick-260628-f57): `provide_train` made C++-faithful → −26% wall, parity-neutral, Kaggle-confirmed metric 4489ms→0. ENABLER: spike-046 wired `phase_prof::dump("train")` into the Python path (was examples-only). REFUTED: sync-floor (286ms/1.7%), route-narrow-to-CPU (CUDA beats CPU on few-vCPU Kaggle), resident_reset. Kaggle CLI harness reusable.** |
| **Vector<P,N> SIMD vectorization of the histogram pipeline (frontier CLOSED)** | references/vector-simd-histogram-kernels.md | **RULE: `Vector<P,N>` (cubecl 0.10 — NOT `Line<T>`) pays only where the kernel is memory-bound AND the vectorized op covers the bottleneck. 041 subtract WON+SHIPPED (cpu 3.68×/hip 1.29×, quick-260627-agx); 042 scan NULL (dependent chain); 043 build-input NULL+wide-regression (gather is a permutation); 044 dequant feasible-but-ROI-bounded DON'T-WIRE; 045 coalesced-rewrite INVALIDATED. 3 cube-macro gotchas logged.** |

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
- 034-post-copack-narrowupload-reattribution (VALIDATED measurement — re-profile after the 024 co-pack + 029 narrow-upload wires: the bottleneck MOVED a 4th time. Launch-bound (small/med/large): the scan-sync floor is CLOSED by co-pack (scan 3–7%; syncs 59→30/tree confirmed); the device partition round-trip is the NEW #1 reclaimable phase, 38% medium / 30% large. Wide/compute-bound: UNCHANGED, build-dominated ~91% (neither lever targets it, predicted). Motivates 035. Also corrected the LGBM_SCAN_DRAIN tooling (needs LGBM_SCAN_PROF=1; was missing on the co-pack scan fn → quick-260625-tw1))
- 035-rocm-host-partition (VALIDATED + SHIPPED quick-260626-a6t — route the rocm partition on the HOST by default via the shipped 027 fused path instead of the device round-trip: `prefers_host_partition()` default-ON, off-switch `LGBM_ROCM_HOST_PARTITION=0`. ~1.18–1.23× launch-bound, wash wide. KEY: both paths land in host `indices_` and the build reads host indices either way ⇒ NO index re-upload; the device round-trip is pure overhead on shared DDR5. Parity within ~1e-6 (def-f8u-01 3-arm: host-vs-device max 1.907e-6 = inherent GPU f32 noise; NOT a bit-exact swap; anchor-pinned hip gate). The rare GPU lever that wins on the APU. Gated by first fixing a latent FUSED-path subtract_resident bug — debug 8aed100)
- 036-branch-divergence-inventory-gate (PARTIAL — measurability VALIDATED, critical-path WEAK ⇒ DON'T-CHASE. The "optimize conditional branching" gate. (1) Controlled-divergence LADDER (4 arms, identical total work, only intra-wave trip-count distribution differs) = near-ideal 1.00 / 1.89–1.95× / 3.62–3.84× / 25.6–29.3× (2 restarts) ⇒ wavefront lockstep-masking FAITHFUL on the spoofed 8-CU APU; divergence is the ONE GPU effect cleanly sign-measurable here (scheduler property, not CU-count/memory-bound). (2) But kernels are ALREADY branchless (`select`-everywhere — cubecl-cpu MLIR), the dominant wide build is uniform/divergence-free by construction (030), partition is branchless+host-routed (035), the heavily-divergent plane-atomic kernel is the p93 NULL/dead path; the only live divergence — split-scan loop-trip-count — is on the 3–7% scan phase, ZERO at the production all-256-bin cardinality, real only on mixed-cardinality data (e2e ceiling ≪1%). Don't chase as a general lever; 037 = bounded mixed-cardinality curiosity, 038 break-vs-select = likely don't-build [`done` is intra-lane predication ⇒ no wave-max-trip reduction unless early-exit correlated; + forks the bit-exact anchor])
- 037-autotune-hip-feasibility (VALIDATED — the AUTOTUNE track [NOT 036's deferred divergence "037/038"]. CubeCL `cubecl::tune` works END-TO-END on cubecl-hip 0.10: compile/run-on-device/benchmark-both/pick-winner, in-proc cache hit ~6µs [~78,000× vs 490ms cold-tune], PERSISTENT disk cache across processes [`target/autotune/0.10.0/rocm_0/*.json.log`, ~828µs cold-with-cache]; independently re-derived spike-007's P=16. The `cubecl_manual` doc is WRONG on 3 load-bearing points [code from SOURCE]: key-gen closure returns the AutotuneKey not a String; `execute`'s 1st arg is the cache-namespace ID not the key; AutotuneKey needs `serde::{Serialize,DeserializeOwned}` under std_io [always on linux] ⇒ serde dep mandatory [added dev-only])
- 038-autotune-inplace-correctness (VALIDATED — accumulating kernels [histogram BUILD `fetch_add` into resident `out`] corrupt **27×** under `CloneInputGenerator` [`Handle::clone` = ref-count bump NOT a buffer copy ⇒ every benchmark rep hits the real `out`; the 27 = whole sample budget, NOT a +1 bias the manual implies]. FIX = a fresh-output `InputGenerator` [`generate` returns a new Vec<Handle> with `out` replaced by a fresh zeroed buffer]; winner's final run uses the ORIGINAL inputs [tuner.rs:183 vs local.rs:170] ⇒ real `out` touched once ⇒ `rel_err 0` by grad-conservation [order-independent]. Classify: OVERWRITE=safe-as-is, ACCUMULATE=fresh-out, in-place-RMW=deep-copy-gen [but partition host-routed on rocm 035]. GAT gotcha: spell `generate<'a>` return as `<Vec<Handle> as TuneInputs>::At<'a>` or E0195)
- 039-autotune-key-cache-thrash (VALIDATED — keying granularity is a real lever w/ a clear sweet spot. EXACT(`rows`)=STORM [25/25 nodes cold, 0 reuse, 975ms for ONE shallow tree ~39ms/node — per-leaf counts never repeat]; FIXED(`feats`)=cheap [1 tune 158ms, 24 free] but mis-applies the ROOT's variant [P16] to small leaves; BUCKET(`log2 rows`)=SWEET SPOT [5 keys, 20/25 free, 325ms ~3× EXACT, keeps the per-regime P16↔P1 crossover]. Insight: variant choice tracks the OCCUPANCY REGIME not exact rows ⇒ log2(rows) amortizes. Warm hits ~3µs. SURPRISE: EXACT's P16/P1 split is run-to-run NOISY [small leaves near the tie] — 2nd argument vs over-fine keying. Don't rely on `tuner.init` to vary per-call inputs [memoizes first set]; build fresh Arc<TunableSet>, cache lives in LocalTuner)
- 040-autotune-vs-heuristic (VALIDATED comparison — autotune BEATS the shipped heuristic ~10%, NOT the predicted wash. `row_partition_count(50,n)` resolves target_cubes=8CU×8=64, MIN_LEAF=256k, clamp(64/50)=1 ⇒ **P=1 for every leaf at the production 50-feat width** [the 8-CU correction OVER-corrected from the phantom-96-CU P≈16, now DISABLES row-partitioning]. Rigorous P-sweep {1,4,8,16,32} device-time median ACC=20/REPS=11, 3 restarts: P=1 is the SLOWEST point at every size; autotune picks P∈{4,8,16,32}, beats heuristic ~2–16% [typ ~10%], NEVER loses [12 cells×3]. Curve FLAT P4–16 [exact best wanders] but SIGN rock-stable. Surfaced a LATENT PRODUCTION MIS-TUNE → recalibrate `row_partition_count` OR adopt autotune [robust+portable]. Honest bound: ~10% on spoofed-APU GPU build, CPU beats e2e [perf-gap-vs-cpp-40-80x] ⇒ deliverable = measure-don't-model + portability)
- 041-line-feasibility-subtract (VALIDATED + SHIPPED quick-260627-agx — `Vector<P,N>` SIMD subtract, cpu f32-vec16 3.68×/hip f32-vec4 1.29×, bit-exact; cubecl 0.10 type is `Vector<P,N>` NOT `Line<T>`)
- 042-line-scan-pair-read (NULL — scan is a dependent prefix-sum chain, not load-bound; vectorizing the load attacks a non-bottleneck)
- 043-line-build-gradhess-input (NULL + wide REGRESSION 0.83× — the dominant build bin-gather is a permutation, structurally un-vectorizable; grad/hess latency already hidden)
- 044-line-fixcompact-dequant (VALIDATED feasibility, ROI-BOUNDED DON'T-WIRE — dequant vectorizes bit-exactly [cpu f64-vec8 2.52×] but is a sub-1% fused-minority fraction; hip f64 caps at vec2)
- 045-coalesced-build-vector (INVALIDATED both counts — the reorder pass IS the permuted gather [net loss]; once coalesced the build is LDS-atomic-scatter-bound so Vector regresses. Closes the Vector<P,N> histogram frontier; 3 cube-macro gotchas logged)
- 046-python-path-phase-prof (VALIDATED — ENABLER: wired env-gated `phase_prof::dump("train")` into booster.rs [was bench-examples-only], making the shipped Python/CUDA path observable; the prior Kaggle log had ZERO attribution because of this gap)
- 048-kaggle-cuda-confirm (VALIDATED — FIRST real-discrete-CUDA attribution [Kaggle, 500k×50]: ~5–6× vs official. 26% = host per-iter training-metric eval [booster.rs:1291 `|| valid.is_none()`, divergent from C++] → SHIPPED fix quick-260628-f57, −26%, Kaggle-confirmed metric 4489→0ms. REFUTED: sync-floor [286ms/1.7%], route-narrow-to-CPU [CUDA beats CPU on few-vCPU Kaggle])
- 049-in-learner-other-attribution (VALIDATED — `in_learner_other` is a DEAD END: resident_reset REFUTED [0.3ms/100 trees], ~92% diffuse residual on CPU+GPU. Post-fix wall map: GPU hist phases 53% [architectural long-pole], Python marshalling 25% [→ attributed by 050], in_learner_other 15%, metric 0%. Don't scope a phase around in_learner_other)
- 050-python-marshalling-binning (VALIDATED + SHIPPED — the "Python marshalling 25%" is actually single-threaded BINNING: numpy→Vec<Vec> marshalling = 43ms [non-issue], raw→bin serial = 624ms [was hidden as binning=0 — train_raw's bin step was never BINNING_NS-wrapped]. FIX shipped bit-exact: feature-parallel rayon binning `(0..num_features).into_par_iter().map(bin_feature)` [order-preserving + per-feature deterministic seed], 6.5× [624→96ms@16c], `LGBM_PAR_BIN=0` serial gate; verified vs `raw_bin_train_matches_cpp_golden`. Lesson: match C++'s OpenMP-over-features for any per-feature host loop)

Full campaign 001–013 wrapped (2026-06-17); 014a/014b + shipped p9v/qix/rdu/rsh levers wrapped (2026-06-21, GPU wide-shape 1M×500, cumulative −68%); 015–022 GPU build/scan KERNEL campaign wrapped (2026-06-25 — u64 fixed-point build + feature-per-lane scan SHIPPED; replication & within-feature scan parity-characterized but ROI-gated); 023/024 + 026–029 wrapped (2026-06-25 — GPU scan round-trip attribution + sibling co-pack WIRED, and the PARTITION memory-traffic arc: fuse-gather SHIPPED CPU + narrow-upload SHIPPED ROCm, parallelize/double-buffer NULL). 030/031 wrapped (2026-06-25 — BUILD bottleneck RE-ATTRIBUTION: post-u64 the wide build is uncoalesced-bin-gather-bound, not atomic/grad-hess; the stable-partition monotone order already banks ~70% of coalescing so the build is effectively tuned on the APU — reopens only on discrete gfx110x). 022b/032/033 wrapped (2026-06-25 — closing the partition + within-scan arcs: 032 one-gather validation fold SHIPPED on CPU [~1.14–1.41× U8, bit-exact, quick-260625-qn9; found by auditing the shipped wiring vs the spike]; 033 gather-prefetch PARTIAL/DON'T-WIRE [helps only when column ≫ LLC = U32/multi-M rows; + a typed-slice autovectorization landmine]; 022b perf-disproves the within-feature cooperative scan [wins only narrow ≤256 feat]. **The CPU host partition is DONE — no positive-ROI lever remains on this hardware.**). 034/035 wrapped (2026-06-26 — re-attribute-then-act: after the co-pack + narrow-upload wires, 034 re-profiled and the launch-bound bottleneck MOVED off the (now-closed) scan-sync floor to the device partition round-trip [30–38%]; 035 SHIPPED the fix — route the rocm partition on the HOST by default [quick-260626-a6t, ~1.18–1.23× launch-bound, wash wide, parity within ~1e-6], **the rare GPU lever that wins on the 8-CU APU itself**. Unblocked by fixing a latent FUSED-path subtract_resident bug from the Phase-12 co-pack scan deferral [debug 8aed100] + re-wiring the inert LGBM_SCAN_DRAIN [quick-260625-tw1]). 036 wrapped (2026-06-26 — the branch-divergence GATE: measurability VALIDATED [the controlled-divergence ladder is near-ideal on the spoofed APU — divergence is the one GPU effect that survives the spoof, a reusable carve-out from the sign-only caveat], critical-path WEAK [kernels already branchless, dominant build divergence-free, only live divergence on the 3–7% scan + zero at uniform cardinality] ⇒ DON'T-CHASE as a general lever; 037/038 deferred as bounded curiosities). 037–040 wrapped (2026-06-26 — the GPU-kernel-AUTOTUNING arc, all VALIDATED: CubeCL `cubecl::tune` works end-to-end on cubecl-hip 0.10 [037, manual wrong on 3 points — code from source], accumulating kernels need a fresh-output InputGenerator or corrupt 27× [038], key on log2(rows) not exact rows or it's a tuning storm [039], and autotune BEATS the shipped 8-CU `row_partition_count` ~10% [040 — it under-partitions to P=1 at the production width; a latent mis-tune]. Deliverable = measure-don't-model + portability; ~10% bounded on the spoofed-APU GPU build which CPU wins e2e. Reference: gpu-kernel-autotuning.md). 041–045 + 046/048/049 wrapped (2026-06-28 — TWO arcs: (a) the **Vector<P,N> SIMD frontier CLOSED** [041 subtract WON+SHIPPED, 042/043/045 NULL/regression, 044 ROI-bounded; the RULE = Vector pays only on memory-bound kernels where the vectorized op covers the bottleneck; reference: vector-simd-histogram-kernels.md]; (b) the **FIRST real-discrete-CUDA attribution** [Kaggle, the spoofed-APU campaign never saw real NVIDIA]: spike-046 made the Python path observable, 048 found+SHIPPED the C++-faithful metric-eval fix [−26% wall, quick-260628-f57, parity-neutral], 049 mapped the post-fix wall [GPU hist phases 53% architectural long-pole, Python marshalling 25% unattributed/next-easy-win, in_learner_other a diffuse DEAD END]. REFUTED on real hardware: sync-floor, route-narrow-to-CPU, resident_reset. Reference: cuda-discrete-gpu-bottleneck.md). 047 skipped (Kaggle gave real numbers directly). 050 wrapped (2026-06-28 — closed the loop on the 049 map: the "Python marshalling 25%" is single-threaded BINNING [marshalling itself only 43ms]; SHIPPED feature-parallel rayon binning 6.5× bit-exact. Two of the three non-kernel chunks now closed [metric 0, binning 6.5×]; only the GPU hist phases [53%, architectural] remain).
</metadata>
