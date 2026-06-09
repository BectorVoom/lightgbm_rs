---
quick_id: 260609-f3g
title: Benchmark the GPU partition change (260609-eu9) on large data
date: 2026-06-09
status: complete
type: benchmark
code_changes: none (throwaway instrumentation reverted)
---

# Quick Task 260609-f3g — Summary

**Benchmark of the 260609-eu9 partition-kernel parallelization on the real gfx1100.**
Throwaway A/B instrumentation, fully reverted — no code shipped.

## Headline

Parallelizing the partition kernel (single-unit → one-unit-per-row) gives a
**~21–60× speedup on the kernel itself** at 100k–5M rows, **growing with data size**:

| n | kernel (GPU bucket) | full `data_partition_on` |
|---|---|---|
| 100k | 6.3 ms → 0.30 ms (**~21×**) | 6.85 → 0.60 ms (11.5×) |
| 1M | 68 ms → 1.6 ms (**~42×**) | 73.76 → 4.48 ms (16.5×) |
| 5M | 333 ms → 5.5 ms (**~60×**) | 349.65 → 15.51 ms (22.5×) |

The single-unit version scaled **linearly at a fixed ~14 Mrows/s** — the signature of
a serial one-lane walk; the parallel version reached **322 Mrows/s** at 5M. Full-call
speedup is smaller because the identical host upload + two-pass gather (unchanged O(n)
CPU work) dilute the kernel win.

## Caveats (honest)

- Partition kernel measured **in isolation at large n**. In training it runs per-split
  over leaf subsets; the production bench corpus is 20k rows where the GPU is
  launch-bound, so end-to-end impact at current bench sizes is modest. The win is real
  and **scales with data size** → matters for large-data GPU training.
- This quantifies WHY the as-built single-unit GPU kernels were ~214× slow (kfu) — a
  one-lane O(n) walk at ~14 Mrows/s.
- CPU production path (`data_partition_cpu_native`) unaffected.

Full numbers + method: `260609-f3g-FINDINGS.md`. Validates the eu9 change (commit
b141a82); see also 260609-eo5 Finding #1.

## Files modified

None (throwaway timing buckets in `partition.rs` + a rocm-gated bench test were added,
measured, then reverted via `git checkout` + file delete).
