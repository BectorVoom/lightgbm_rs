---
title: Profile the GPU training loop on large data (stage attribution)
date: 2026-06-21
priority: high
depends_on: establish-large-data-benchmark-fixture
context: .planning/notes/gpu-large-data-bottleneck-framing.md
---

# Profile the GPU training loop on large data — stage attribution

Produce a **trustworthy breakdown** of where GPU large-data time goes. Diagnostic
only: the deliverable is the attribution, not a fix.

**Blocked by** `establish-large-data-benchmark-fixture` — do not profile until the
large fixture exists, or the numbers will be fixed-overhead noise.

## Required attribution (the whole point)

Separate, with explicit numbers, the time spent in:

- **kernel execution** (real device-busy time, per kernel)
- **launch overhead** (dispatch / submission cost, esp. per-feature × per-tree)
- **host↔device transfer** (uploads/downloads, reads)
- **CPU-side glue** (partitioning, split-finding, allocation, orchestration)
- **sync-idle** (GPU idle while host blocks on `client.read` / barriers between trees)

Misattributing any of these repeats the host↔device round-trip mistake
(round-trip was NOT the bottleneck — see notes framing). **Do not** collapse
launch + sync-idle into "kernel time."

## Method

- **Host-side stage timers** around each loop stage (warm runs only), summed across
  iterations and reported per-stage and per-tree.
- **Per-kernel GPU attribution** via ROCm tooling (`rocprof` / `omniperf`) on
  gfx1100 — see research question on whether cubecl-hip-generated kernel names
  survive and how to isolate launch/sync from kernel busy time.
- **Cross-check** the two: host timers and GPU profiler should reconcile; large
  discrepancies are themselves a finding (e.g. sync-idle).
- Report **occupancy** under the real feature/bin counts (not the micro-bench
  occupancy) — this is the axis the framing flags as most likely.

## Output

- A stage-attribution table (kernel / launch / transfer / glue / sync-idle) for the
  large fixture, warm steady-state.
- A one-paragraph "where the time actually goes" verdict that the follow-on
  decision (kernel fix vs. GPU→CPU routing for large data) can be made against.

## Explicitly out of scope

- Re-tuning the histogram inner loop (levers closed — `gpu-hist-levers-closed`).
- Any fix. This todo ends at the trustworthy profile + verdict.
