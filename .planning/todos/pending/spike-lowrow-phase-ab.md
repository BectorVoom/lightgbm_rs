---
title: Spike — per-phase A/B vs C++ at low rows (localize the fixed overhead)
date: 2026-06-14
priority: high
type: spike
context: /gsd-explore "investigate bottleneck low rows and optimise in learning speed"
---

# Spike: per-phase A/B vs C++ at low rows

## Uncertainty to reduce

Where do the ~188µs/iter of fixed overhead go that keep the Rust CPU f64 path ~1.9×
slower than C++ LightGBM 4.6 at 2k rows? Which phase(s) — histogram build, split find,
partition, score update, grad/hess, boosting glue — does Rust overspend vs C++?

## Approach: diff matching phase timers

1. **C++ side (nearly free).** Build `lib_lightgbm` 4.6 with `-DTIMETAG`
   (`USE_TIMETAG` CMake flag) → its `Common::Timer` dumps accumulated time per named
   phase tag at the end of training. Run the 2k-row regression config (100 iters, 31
   leaves, lr 0.1, min_data_in_leaf 20, single-thread) and capture the tag dump.
   - FIRST: confirm TIMETAG tag granularity (grep `global_timer`/`TIMETAG` in
     `LightGBM/src`) and that the tags map onto distinct Rust phases. If C++ tags are
     too coarse, add matching scoped timers on the Rust side only and compare totals.
2. **Rust side.** Instrument the train loop (boosting → tree-learner → compute) with
   per-phase scoped timers accumulating across the 100 iters: grad/hess, histogram
   construct, find_best_split, partition, score update, and per-iteration glue/alloc.
   Gate behind a cfg/env so it's zero-cost in normal builds.
3. **Diff** the two breakdowns at 2k rows. Output: a per-phase µs table (Rust vs C++)
   that localizes the gap to specific phase(s).

## Possible outcomes

- **Alloc/setup churn dominates** (histogram pool / tree nodes / partition scratch
  reallocated per tree) → activates seed [[reuse-tree-learner-scratch-across-iters]].
- **A compute phase overspends** (e.g. Rust's ordered f64 fold vs C++'s tight loop)
  → different, phase-specific optimisation.
- **Overhead is diffuse** (no single dominant phase) → the gap is the sum of many
  small inefficiencies; reconsider whether the 1.9× is worth chasing.

## Guardrails

- Measurement only — do not change train semantics. Keep the bit-exact CPU merge gate
  intact; timers must be removable / cfg-gated.
- Use the same synthetic 2k corpus as `bench_crossover.rs` / `bench_train.rs` for an
  apples-to-apples A/B; single-thread both sides to match.
- Don't git-add the `LightGBM/` C++ tree — see [[lightgbm-ref-tree-untracked]].

## Kick off

Run `/gsd-spike` on this todo. Framing: `.planning/notes/low-row-gap-fixed-per-iteration-overhead.md`.
