# Spike Manifest

## Idea

Optimise lightgbm_rs **learning (train) speed**, both ends of the row-count curve:
- **High rows (GPU track, spike 001):** the ROCm path is launch-bound and ~100× slower
  on small data; prove where it crosses below the CPU anchor at scale. → ~700k rows.
- **Low rows (CPU track, spike 002):** the CPU anchor is ~2–3× off C++ even where it's
  closest; localize the fixed per-iteration gap via a per-phase A/B. → histogram build.
See `.planning/notes/gpu-track-goal-crossover-at-scale.md` and
`.planning/notes/low-row-gap-fixed-per-iteration-overhead.md`.

## Requirements

- Backend stays **compile-time switched** (`--features rocm`); the CPU f64 anchor
  remains the default and the bit-exact merge gate. Speed work must not touch parity.
- "GPU is faster" is only claimed in the regime the data supports — never route
  small/medium datasets to the GPU.
- Crossover claims are wall-clock facts; the ~1e-6 GPU-vs-CPU parity contract at large
  shapes is a separate, still-open gate.

## Spikes

| # | Name | Type | Validates | Verdict | Tags |
|---|------|------|-----------|---------|------|
| 001 | gpu-cpu-crossover | standard | GPU train wall-clock crosses below the CPU anchor as rows scale up | ✅ VALIDATED — crossover ≈ 700k rows (feat=50/bins=255/31 leaves); GPU wins ≳1M, widening to 1.45× at 2M | gpu, rocm, performance, crossover, benchmark |
| 002 | lowrow-phase-ab | standard | Per-phase A/B vs C++ localizes the low-row fixed-overhead gap | ✅ VALIDATED — gap is ~entirely histogram BUILD: Rust 232µs/iter vs C++ 44.5 (5.2×) = 187µs of the ~188µs/iter gap; scan & partition near parity. Fix = R3 columnar storage + scratch reuse | performance, cpu, histogram, profiling, low-rows, a-b |
| 003 | columnar-hist-build | standard | Hoisting the redundant per-feature grad/hess gather to once-per-leaf speeds the build at both scales, bit-exactly | ✅ VALIDATED + SHIPPED — build −33% small / −39% large; train −16–18% small / −32–33% large; bit-exact gate green. The real R3 lever (quick-260614-p0n chased alloc and failed; the cost was redundant gather memory traffic) | performance, cpu, histogram, gather, bit-exact |
| 003b/r4o | fused-branchless-build | follow-on | Fold direct from the bin column into reused hot scratch, branchless (validation relocated upstream) | ✅ VALIDATED + SHIPPED (quick 260614-r4o) — train −17% small / −6.6% large on top of spike-003; bit-exact; per-element safety checks serialize the fold (relocated to once-per-train) | performance, cpu, histogram, branchless |
| 004 | columnar-u8-bins | standard | Narrowing the bin column u32→u8 speeds the gather+fold via cache density | ✅ VALIDATED + SHIPPED (quick 260614-ruz) — micro-bench isolated −58% u8; full impl (BinColumn u8/u16/u32 in lgbm-compute, per-width match fold) gave large train −49% (2.74→1.40s) / build −53%, small no-regression, bit-exact. Cumulative w/ 003+r4o: large −67% | performance, cpu, histogram, cache, bin-width |
| 005 | feature-parallel-build | standard | Parallelize the per-feature build across cores above a leaf-size threshold | ✅ VALIDATED + SHIPPED (inline) — rayon over features, gated at leaf_rows≥16384 (LGBM_PAR_THRESHOLD). Large train −26–31% (16 cores), small/medium NO regression (serial below threshold; unconditional parallel regressed small 5×). Bit-exact incl forced-parallel oracle 29/0 + permanent parallel==serial test. R4. Makes CPU anchor multi-threaded (reframes 1-core-vs-C++ basis + pushes spike-001 GPU crossover up) | performance, cpu, parallelism, rayon, R4 |
| 006 | gpu-u8-bins | standard | Narrow device bins u32→u8 to speed the GPU build (analog of the CPU u8 win) | ❌ INVALIDATED (gfx1100 micro-bench) — u8 vs u32 device-bin-read ≈0% (−0.6 to +1.7%). The GPU build is atomic-contention/scattered-read-LATENCY bound (234 Mreads/s, slow), NOT bin-bandwidth bound — CPU's L2-density mechanism doesn't apply. Cheap probe saved a multi-kernel plumbing dud. Bonus: Array<u8> proven to work on HIP | performance, gpu, rocm, bin-width, negative-result |
| 009 | multifeature-per-cube | standard | Does packing many small-bin features into one cube beat one-cube-per-feature at MATCHED occupancy? | ❌ INVALIDATED (gfx1100, 1M×128×32) — null/slight regression: packed/per-feature = 0.91–1.00× at matched 768 cubes (correctness 7.2e-6). Per-cube overhead was never the bottleneck; row-partitioning (007) already supplies occupancy, so packing has nothing to amortize and its extra LDS/loops slightly regress. Keep one-cube-per-feature × P. Closes the "wastes a CU on small-bin features" gap with evidence. | performance, gpu, packing, negative-result |
| 008 | 16bit-discretized-hist | standard | Can LightGBM's int16 discretized histogram (Lever 3) meet the exact-parity contract, or is it approximate-only? | ❌ INVALIDATED for exact parity (CPU probe) — quantization drift is irreducible: even at FULL int16 (65536 levels) rel err ≈ 3.2e-4, ~30× over the 1e-5 f32 gate / ~300× over the ~1e-6 contract; at LightGBM's default (4 bins) it's 200%+. It is `use_quantized_grad` (default FALSE) — an opt-in APPROXIMATE mode by construction. Can only ever be a separate opt-in mode, never a drop-in for the exact build. Cheap CPU probe killed a large packed-int-atomic kernel before plumbing. | performance, gpu, quantization, parity, negative-result |
| 007 | row-partitioned-histogram-build | standard | Split a feature's rows across P cubes (LightGBM grid_dim_y analog) to raise occupancy on large leaves | ✅ VALIDATED (gfx1100 micro-bench, 1M×50×256) — production kernel launches only 50 cubes (one/feature) on a 96-CU GPU = starved. Row-partition to P=16 (~800 cubes, ~8 wkgrps/CU) gives a stable **1.30–1.39×** build speedup (6/6 rounds across 3 process runs). Confirms spike-006's "latency-bound" was partly starved OCCUPANCY. P=32 over-partitions → regresses to ~1.0× (tune to ~8 wkgrps/CU, don't maximize). Win is modest + still atomic-bound (~820 Mr/s ≪ bandwidth ceiling) → pair with register-row-batching for the residual intra-cube contention. PARITY: P≥2 widens GPU-vs-P=1 f32 divergence 4e-7→~2e-5 rel (independent partial-sum trees); CPU f64 anchor untouched (merge gate safe) | performance, gpu, rocm, histogram, occupancy, row-partition |
