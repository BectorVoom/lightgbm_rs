---
title: GPU large-data bottleneck — investigation framing
date: 2026-06-21
context: explore session — scoping the "GPU kernel is slow on big data" investigation
---

# GPU large-data bottleneck — investigation framing

## Intent (diagnostic-only)

Produce a **trustworthy stage-attribution profile** of the GPU training loop on
genuinely large data. We are starting from *"GPU is slow on big data and we don't
yet know why."* The success bar is **locate the bottleneck**, not fix it — decide
on kernel fixes vs. GPU-vs-CPU routing **only after** the numbers are in.

"Large" here means **all axes at once** — rows × features × bins × trees — a
genuinely big real-world dataset, not a synthetic micro-bench. When every axis is
large simultaneously, the bottleneck is unlikely to be a single kernel inner loop;
it's more likely **how the whole training loop drives the GPU at scale**
(allocation churn, per-tree sync points, per-iteration fixed overhead × many trees,
occupancy under real feature/bin counts).

## Central risk: measurement validity

A profile that **misattributes** time will send the next decision the wrong way —
exactly how the host↔device round-trip assumption misled us before
(see [[l3-on-gpu-fixhistogram-deferred]]: the round-trip was NOT the bottleneck).
The profile must cleanly separate:

- **cold vs. warm** (the "cold-ceiling-overstates-warm" rule — see
  [[vec-vec-optimisation-spikes-010-011]])
- **launch overhead** vs. **kernel execution**
- **sync-idle counted as kernel** vs. real kernel busy time
- **host-side CPU glue / transfer** vs. device time

## Prior-art guardrails (don't re-litigate these)

- GPU **hist-build inner-loop levers are closed** — row-partition SHIPPED;
  register-batch + multi-feature packing NULL; 16-bit discretized INVALIDATED for
  exact parity. See [[gpu-hist-levers-closed]] and [[spike-007-row-partition-occupancy]].
- The faithful CubeCL port of LightGBM's CUDA hist kernel is **~5.4× slower than
  the multi-threaded CPU anchor** — see [[cuda-mirror-kernel-slower-than-cpu]].
- Overall Rust path is ~40–80× slower than C++ 4.6 — see [[perf-gap-vs-cpp-40-80x]].

**Implication:** the profile should deliberately look **above** the hist inner
loop. Re-tuning the histogram kernel is the path memory already closed; the open
question is loop-level / occupancy / sync overhead at real scale.

## Sequencing

1. **Step zero — fixture.** No representative large dataset exists yet. Profiling
   on small/synthetic data would be dominated by fixed overhead and would
   *misrepresent* the large-data regime — the one thing this investigation can't
   afford. See todo `establish-large-data-benchmark-fixture`.
2. **Profile.** Instrument host stage timers + per-kernel GPU attribution; produce
   the stage breakdown. See todo `profile-gpu-training-loop-large-data`.
3. **Decide.** Kernel fix vs. routing — only once the breakdown is trustworthy.

## Possible honest endings

This investigation can legitimately conclude with *either* a kernel/loop fix *or*
"GPU is the wrong tool for this shape on gfx1100 — route large data to CPU." Both
are valid outcomes; the profile decides which. (GPU-vs-CPU routing is already a
known lever — see Skill `spike-findings-lightgbm_rs`.)
