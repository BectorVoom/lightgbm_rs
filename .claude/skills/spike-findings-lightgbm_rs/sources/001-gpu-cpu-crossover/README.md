---
spike: 001
name: gpu-cpu-crossover
type: standard
validates: "Given the existing batched/LDS ROCm kernels, when dataset rows are swept upward, then the GPU path's train wall-clock crosses below the f64 CPU anchor at some dataset size"
verdict: VALIDATED
related: []
tags: [gpu, rocm, performance, crossover, benchmark, treelearner]
---

# Spike 001: GPU-vs-CPU Crossover

## What This Validates

**Given** the existing batched histogram + LDS resident ROCm kernels (no kernel
changes), **when** dataset row count is swept upward at a fixed feature/bin shape,
**then** the GPU (`cubecl-hip`, f64) train wall-clock crosses *below* the
deterministic CPU f64 anchor at some measurable dataset size — proving the GPU path
has a reason to exist at scale.

## Research

No external-library research needed — this is an empirical benchmark of code that
already exists. Grounding facts carried in from project memory:

- Backend is **compile-time switched**: default build = `CpuBackend` (native f64
  anchor, bit-exact merge gate); `--features rocm` = `RocmBackend` (gfx1100, f64
  kernels). Single flag, `crates/lgbm/src/booster.rs`.
- GPU path was diagnosed **launch-bound** (~50µs/dispatch ≫ ≤256-bin scan ~1µs);
  histogram construction already batched (lad-p3) + LDS resident pool (fw1).
- Prior bench (`bench_train.rs`) topped out at 20k×50 — far too small for a GPU.

## How to Run

Harness: `crates/lgbm/examples/bench_crossover.rs` (size list / iters / reps are
runtime env vars so one expensive rocm fat-LTO build sweeps many shapes).

```bash
# CPU anchor
BENCH_SIZES="100k:100000:50:255,1M:1000000:50:255" BENCH_ITERS=30 BENCH_REPS=1 \
  cargo run --release --example bench_crossover

# ROCm GPU (gfx1100)
BENCH_SIZES="100k:100000:50:255,1M:1000000:50:255" BENCH_ITERS=30 BENCH_REPS=1 \
  cargo run --release --features rocm --example bench_crossover
```

Switching the `rocm` feature rebuilds the example (~40s, fat LTO + HIP); the env-var
sweep then needs no further rebuilds for that backend.

## What to Expect

CPU train time linear in rows (~47k rows/s, flat); GPU has a high fixed floor (~5–6s)
+ shallow slope, so its rows/s climbs with scale. The two curves cross where CPU's
linear growth overtakes GPU's near-constant floor.

## Investigation Trail

1. **Validated the GPU path runs at all** (highest risk first): `--features rocm`
   builds against ROCm 7.1 / gfx1100 and trains. 20k×50, 30 iters = 6.12s.
2. **Swept GPU up the row axis** (binary built once, env-var sweep): train time
   *barely moved* 20k→100k (6.12 → 6.15 → 5.86s) — textbook launch-bound floor.
   Past 100k it grows (200k 7.22s, 500k 11.80s, 1M 17.41s, 1.5M 23.68s, 2M 30.24s)
   and rows/s climbs 3.3k → 66k.
3. **Swept CPU** at matching sizes: dead linear ~47k rows/s (100k 2.10s … 2M 43.74s),
   essentially zero fixed cost.
4. **Located the crossover**: curves meet at ~700k (CPU 14.94s vs GPU 15.04s, 0.7%
   apart — a tie inside the ~10% single-rep noise band).
5. **Confirmed divergence past crossover** and checked stability with reps=2 at 1M
   (conservative median GPU 19.28s still < CPU 21.63s). Gap widens monotonically:
   1M 1.1–1.2×, 1.5M 1.4×, 2M 1.45× in GPU's favour.

## Results

**VERDICT: VALIDATED.** A crossover exists with *today's* kernels — no batched
find_best_split / subtract / partition work was required to reach it.

Crossover ≈ **700k rows** at (features=50, bins=255, num_leaves=31, regression).
Below it CPU wins (overwhelmingly so under ~200k); above ~1M GPU wins by a margin
that grows with scale.

| rows | CPU train | GPU train | winner | GPU speedup |
|------|-----------|-----------|--------|-------------|
| 20k  | 0.35s     | 6.12s     | CPU    | 0.06× |
| 50k  | 0.98s     | 6.15s     | CPU    | 0.16× |
| 100k | 2.10s     | 5.86s     | CPU    | 0.36× |
| 200k | 4.20s     | 7.22s     | CPU    | 0.58× |
| 500k | 10.59s    | 11.80s    | CPU    | 0.90× |
| 700k | 14.94s    | 15.04s    | **tie**| 0.99× |
| 1M   | 21.63s    | 17.4–19.3s| **GPU**| 1.12–1.24× |
| 1.5M | 32.87s    | 23.68s    | **GPU**| 1.39× |
| 2M   | 43.74s    | 30.24s    | **GPU**| 1.45× |

(30 iters, 1 rep except 1M/2M reps=2; gfx1100; CPU = native f64 anchor.)

### Surprises / nuance

- GPU rows/s is *still climbing* at 2M (66k vs CPU's flat 47k) — the GPU is not yet
  saturated, so the advantage should keep growing beyond 2M.
- The launch-bound floor is real and large (~5–6s for 30 iters regardless of rows
  ≤100k). Lowering that floor is exactly what the parked seed
  (batch find_best_split / subtract / partition) would do → it would *move the
  crossover left* (GPU wins at smaller datasets), not enable a win that's otherwise
  impossible. So the seed is an **optimisation**, not a **prerequisite**.

### Caveats (not covered by this spike)

- **Speed only, not parity.** This measured wall-clock, not the ~1e-6 GPU-vs-CPU
  output contract at these large shapes. Crossover is a timing fact; confirming the
  GPU still matches the anchor at 1M+ rows is a separate gate.
- Single-rep variance ~10%; crossover stated as a band (~700k tie, robust GPU win
  past ~1M), not a single exact row count.
- One shape family (feat=50/bins=255/31 leaves). Crossover row count will shift with
  feature count, bin count, and num_leaves (more leaves/features = more launches =
  higher GPU floor = crossover moves right).

## Signal for the Build

- The GPU track's reason to exist is **proven**: ROCm beats the CPU anchor for
  datasets ≳ ~1M rows, widening with scale. Pin any "GPU mode" guidance to that
  regime; never route small/medium data to GPU.
- The launch-bound floor is the lever for *lowering* the crossover — activates seed
  `batch-find-best-split-subtract-partition` as an optimisation with a now-quantified
  payoff (shave the ~5–6s floor → crossover moves below 700k).
- Before shipping GPU-at-scale, run a parity check at 1M+ rows (the open caveat).
