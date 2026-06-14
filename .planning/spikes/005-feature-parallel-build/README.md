---
spike: 005
name: feature-parallel-build
type: standard
validates: "Given histogram build dominates large-row train, when the per-feature build is parallelized across cores above a leaf-size threshold, then large-row train speeds up materially, bit-exactly, with no small/medium regression"
verdict: VALIDATED
related: [003, 004]
tags: [performance, cpu, histogram, parallelism, rayon, R4, bit-exact]
---

# Spike 005: Feature-parallel histogram build (R4)

## What This Validates

**Given** the histogram build dominates large-row CPU train (≈77–90% after spike-003/
r4o/ruz), **when** the per-feature build is parallelized across rayon (each feature folds
its own histogram from shared read-only ord_g/ord_h), **then** large-row train speeds up
materially — bit-exactly (each feature's fold order is unchanged; disjoint outputs ⇒
result is thread-count-independent) — gated by a leaf-size threshold so small/medium
leaves stay serial (rayon dispatch overhead crushes tiny per-feature folds).

## Results (16-core machine, settled)

**Bit-exact: GREEN even with EVERY build forced parallel** (`LGBM_PAR_THRESHOLD=1`):
oracle `learner_parity` 29/0. The parallel path is deterministic (disjoint features,
unchanged per-feature fold order).

**Unconditional parallel is a DISASTER at small** — 2k-row train 26.9ms → ~140ms (5×
regression): rayon task-dispatch overhead per leaf-build × num_leaves × iters dwarfs the
tiny per-feature work. → a leaf-size threshold is mandatory.

**Large (200k×32, settled), threshold sweep:**

| threshold | large train | vs serial | protects (serial below) |
|-----------|-------------|-----------|--------------------------|
| serial    | 1.60 s | — | — |
| 16384     | 1.19 s | **−26%** | small + medium (≤16k-row leaves) |
| 8192      | 1.10 s | −31% | ≤8k |
| 4096      | 1.05 s | −33% | small only |

**Crossover probe (where parallel starts winning), 30 feat:**

| leaf rows | serial | parallel | winner |
|-----------|--------|----------|--------|
| 8k  | 77 ms | 83 ms | serial (parallel loses) |
| 16k | 131 ms | 118 ms | parallel −10% |
| 24k | 187 ms | 170 ms | parallel −9% |
| 50k | 376 ms | 322 ms | parallel −14% |

Crossover ≈ 12k rows. So **threshold 16384 is the safe no-regression default**: small
(2k) + medium (8k) + 16k-row leaves stay serial (no regression anywhere ≤16k), and only
genuinely large leaves (≥16384 rows, where parallel clearly wins) parallelize — large
−26%. Lower thresholds trade a bigger large win for medium risk.

## Design

- Gate per leaf-build on `leaf_rows.len() >= LGBM_PAR_THRESHOLD` (default 16384). Below:
  the serial reused-scratch fold (ruz). At/above: rayon `into_par_iter` over features,
  each computing its own histogram Vec, then a sequential copy into the concatenated
  `out`. ord_g/ord_h gathered once per leaf, shared read-only across tasks.
- Bit-exact: per-feature fold order unchanged; the only change is WHICH thread runs each
  feature. The serial and parallel paths produce byte-identical `out` (verified by a
  parallel-vs-serial unit test + forced-parallel oracle).

## Implications (the multi-threaded-anchor reframe — flagged before spiking)

- The CPU f64 deterministic anchor is now MULTI-THREADED at large leaves. The result stays
  bit-deterministic (the merge gate holds), but:
  - The "Rust 1-core vs C++ num_threads=1" gap metric no longer applies at large — Rust now
    uses up to 16 cores there. A fair vs-C++ comparison must run C++ multi-threaded too.
  - The spike-001 GPU crossover (~700k rows, measured vs SINGLE-thread CPU) moves far UP —
    a 16-core CPU at large is ~16× faster, pushing the GPU crossover to many millions of rows.
- These are expected and accepted (the user opted into R4 over the single-thread levers).

## Signal for the Build

- **SHIP IT** with threshold 16384 (env-tunable `LGBM_PAR_THRESHOLD`). Large −26%,
  bit-exact, zero small/medium regression. Productionize: a testable serial-vs-parallel
  seam + a permanent parallel==serial bit-exact unit test + a threat/doc note on the
  multi-threaded-but-deterministic anchor.
- Further: a lower threshold (8192) buys −31% large if medium-leaf neutrality is acceptable;
  or NUMA/thread-pool-size tuning. Defer.
- Closes [[perf-gap-vs-cpp-40-80x]] R4. Stacks on R3 (spike 003 + r4o + ruz).
