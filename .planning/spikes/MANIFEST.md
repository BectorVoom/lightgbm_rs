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
