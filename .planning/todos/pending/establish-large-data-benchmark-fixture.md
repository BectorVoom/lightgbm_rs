---
title: Establish a large-data benchmark fixture (step zero for GPU profiling)
date: 2026-06-21
priority: high
blocks: profile-gpu-training-loop-large-data
context: .planning/notes/gpu-large-data-bottleneck-framing.md
---

# Establish a large-data benchmark fixture

**This gates the GPU bottleneck investigation.** No representative large dataset
exists yet. Profiling on small/synthetic data is dominated by fixed overhead and
misrepresents the large-data regime — defeating the purpose of the investigation.

## Definition of "large" (all axes at once)

The fixture must stress every axis simultaneously, because the bottleneck is
expected to be loop-level, not a single kernel inner loop:

- **rows** — large enough that occupancy / row-partition (P) actually saturates
  the GPU (gfx1100, ~per spike-007 partitioning math)
- **features** — enough that the per-feature dispatch / one-cube-per-feature
  occupancy story is exercised
- **bins** — realistic `max_bin` (the LDS-capacity / histogram-fits-in-shared
  story); include at least the default 255-bin case
- **trees / iterations** — enough that per-iteration fixed overhead × many trees
  dominates, not a single tree

## Acceptance

- [ ] A genuinely large real-world dataset acquired or generated (document which,
      and its shape: rows, features, bins, trees).
- [ ] Wired into the existing bench harness (`crates/lgbm/examples/bench_train.rs`
      and/or `bench_gpu_vs_cpu.rs`) behind a clear selector.
- [ ] **Warm-measurement methodology enforced** — discard cold runs; report warm
      steady-state (cold-ceiling-overstates-warm rule). Document the warmup/repeat
      protocol so later profiles are comparable.
- [ ] Sanity check: CPU-anchor and GPU paths both train to completion on the
      fixture (parity not required here — this is a perf fixture, not a parity gate).

## Notes

- Real datasets preferred over synthetic so feature/bin distributions are
  realistic (binning behavior affects histogram shape and LDS pressure).
- Keep it reproducible (fixed seed / pinned download) so profiles are comparable
  across runs and across the fixture→profile→decide sequence.
