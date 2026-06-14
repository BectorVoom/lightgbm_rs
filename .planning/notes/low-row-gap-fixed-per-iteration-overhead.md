---
title: Low-row CPU gap = fixed per-iteration overhead (not per-row work)
date: 2026-06-14
context: /gsd-explore "investigate bottleneck low rows and optimise in learning speed"
type: note
---

# Low-row gap is fixed per-iteration overhead

## Diagnosis

The CPU native f64 path's gap to C++ LightGBM 4.6 is **smallest at low rows and grows
with rows** (bench, 100 iters, num_threads=1):

| size | rows | Rust | C++ | ratio |
|------|------|------|-----|-------|
| small  | 2k  | 38.7ms | 19.9ms | **1.9×** |
| medium | 8k  | 258ms  | 76ms   | 3.4× |
| large  | 20k | 887ms  | 207ms  | 4.3× |

So "low rows is slow" is NOT a per-row-work story (that's the high-row R3/R4 lever —
columnar storage + rayon over features). At 2k rows the ~19ms gap over 100 iters is
**~188µs/iter of FIXED, per-iteration/per-tree overhead** — costs that don't shrink
with the dataset: histogram-pool alloc/zeroing, tree/leaf node allocation,
`data_partition` setup, score-updater passes, objective grad/hess, boosting glue.

## Why it's worth targeting

That fixed overhead **also exists at medium/large** — it's just dwarfed by per-row
work there. Killing it improves every size; small is simply where it's most visible
and cleanest to isolate. This is the closest Rust gets to C++, so closing it is the
highest-confidence parity win.

## Method (chosen): per-phase A/B vs C++

Break BOTH engines' train loop into the same phases and diff tag-by-tag at 2k rows:
- **C++ side is nearly free** — build `lib_lightgbm` 4.6 with `-DTIMETAG`
  (`USE_TIMETAG` CMake flag) and read its internal `Common::Timer` per-tag dump.
  Real C++ lib builds in this tree — see [[lightgbm-ref-tree-untracked]].
- **Rust side** — instrument the train loop with matching phase timers.
- Diff: turn "1.9× slower" into "Rust spends Nµs more in phase X."

Honors the recorded lesson "profile before the next perf assumption"
(see [[l3-on-gpu-fixhistogram-deferred]]).

## Next action

Spike: `.planning/todos/pending/spike-lowrow-phase-ab.md`. Likely-fix seed (if alloc
churn dominates): [[reuse-tree-learner-scratch-across-iters]].

Related: [[perf-gap-vs-cpp-40-80x]], [[optimisor-manual-applicability]]
("real lever = tree-learner scratch + rayon"). Contrast: the HIGH-row bottleneck is
the GPU crossover at ~700k rows (spike 001) — opposite end of the same curve.
