---
title: Spike — find the GPU-vs-CPU crossover (sweep dataset sizes upward)
date: 2026-06-14
priority: high
type: spike
context: /gsd-explore "investigate bottleneck and optimise in learning speed"
---

# Spike: find the GPU-vs-CPU crossover

## Uncertainty to reduce

Does a dataset shape exist where the ROCm (`cubecl-hip`, f32) path beats the f64 CPU
native anchor in wall-clock — and if so, where? If not, what specifically still
dominates (launch dispatch vs transfer vs compute) at the largest tested scale?

## Approach (cheapest-first, mostly free)

The GPU kernels already exist (batched histogram + LDS resident pool), so the first
experiment needs **no kernel changes** — just run the existing path bigger:

1. Extend `crates/lgbm/examples/bench_train.rs` to sweep upward beyond today's
   "large" = 20k×50: e.g. {50k, 200k, 500k, 1M} rows × {50, 100, 200} features.
2. Run **both** backends (CPU native f64 anchor + ROCm) per shape; record wall-clock.
   GPU runs on the local gfx1100 — see [[rocm-gfx1100-available]].
3. Plot / tabulate CPU vs GPU wall-clock by row count. Identify the crossover row
   count (if any) where GPU < CPU.
4. If GPU never wins: instrument launch count + per-launch time at the biggest shape
   to confirm whether launches still dominate (→ activates the launch-elimination
   seed) or whether it's transfer/compute-bound (→ different work).

## Possible outcomes

- **Crossover found** → that's the proof; record the shape. GPU track has its win.
- **Launches still dominate at scale** → activates seed
  [[batch-find-best-split-subtract-partition]] (batched find_best_split, then
  subtract / partition batching), then re-sweep.
- **Transfer/compute-bound** → unexpected; redirects the optimisation entirely
  (the round-trip was previously NOT the bottleneck — [[l3-on-gpu-fixhistogram-deferred]]).

## Guardrails

- Keep the f64 CPU path as the bit-exact merge gate untouched; GPU is the separate
  ~1e-6 contract. Don't trade parity for speed.
- Large synthetic data only for benching; don't commit multi-MB fixtures.

## Outcome — DONE 2026-06-14 (spike 001, VALIDATED)

Crossover **exists with today's kernels** ≈ **700k rows** (feat=50/bins=255/31 leaves,
gfx1100). CPU wins below it (17× at 20k, 1.1× at 500k); GPU wins above ~1M and the
gap widens (1.24× at 1M, 1.45× at 2M, still climbing). No batched
find_best_split/subtract/partition work was needed to reach it.

Full write-up + data table: `.planning/spikes/001-gpu-cpu-crossover/README.md`.
Reusable harness: `crates/lgbm/examples/bench_crossover.rs`.

Open follow-ups:
- **Parity gate at scale** (NEW): confirm the ~1e-6 GPU-vs-CPU contract still holds at
  1M+ rows — spike measured speed only.
- Seed [[batch-find-best-split-subtract-partition]] is now an *optimisation* (lower the
  ~5–6s launch floor → move crossover left), not a prerequisite.

## Kick off (historical)

Ran via `/gsd-spike` on this todo. See
`.planning/notes/gpu-track-goal-crossover-at-scale.md` for the original framing.
