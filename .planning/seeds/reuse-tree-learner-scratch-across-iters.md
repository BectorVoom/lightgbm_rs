---
title: Reuse tree-learner scratch across iterations (low-row fixed-overhead fix)
trigger_condition: The low-row per-phase A/B spike shows alloc/setup churn dominating the ~188µs/iter gap
planted_date: 2026-06-14
type: seed
context: /gsd-explore "investigate bottleneck low rows and optimise in learning speed"
---

# Seed: reuse tree-learner scratch across iterations

## Idea

At low rows the Rust↔C++ gap is fixed per-iteration overhead, not per-row work
(see [[low-row-gap-fixed-per-iteration-overhead]]). A prime suspect is **allocation /
setup churn**: histogram pool, tree/leaf node buffers, and `data_partition` scratch
reallocated (and zeroed) once per tree × 100 iters. C++ reuses a fixed-size histogram
pool + partition across the whole boosting run. Mirroring that — allocate the
tree-learner's scratch once and reset (not realloc) per iteration — would remove the
churn that's invisible at large rows but ~half the cost at 2k.

Aligns with the recorded "real lever = tree-learner scratch + rayon"
([[optimisor-manual-applicability]]).

## Trigger

Activate ONLY if the spike (`.planning/todos/pending/spike-lowrow-phase-ab.md`)
attributes the gap to alloc/setup phases. If the A/B instead fingers a compute phase
or finds the overhead diffuse, this seed does not apply — don't do it speculatively.

## Notes / guardrails

- Reset-not-realloc must preserve bit-exactness: same zeroing semantics, same fold
  order. The CPU f64 path is the merge gate — re-run the bit-exact corpora after.
- Watch for state leaking across iterations (stale histogram/partition data) — the
  reset must be as complete as a fresh alloc, just cheaper.
- Phase-sized; needs its own plan + the bit-exact gate.

Related: [[perf-gap-vs-cpp-40-80x]] (R3/R4 roadmap), [[low-row-gap-fixed-per-iteration-overhead]].
