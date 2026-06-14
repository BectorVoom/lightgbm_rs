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

## Kick off

Run `/gsd-spike` on this todo to start the measurement loop.

See `.planning/notes/gpu-track-goal-crossover-at-scale.md` for the full framing.
