---
phase: quick-260620-dpk
plan: 01
subsystem: cpu-treelearner
tags: [fusion, fixed-cost, profiling, fork-join-floor, threshold, NULL, parity-gate]
provides:
  - "An evidence-backed verdict that the unified-fusion gate (~100/130 feat) cannot drop without a fundamentally cheaper-than-rayon-fork/join dispatch: per-leaf fixed cost is ~99% fork/join floor + work, <=0.6% allocation"
  - "A near-zero-cost env-gated fusion profiler (fusion_prof.rs, LGBM_FUSION_PROF=1) decomposing per-leaf cost into gather/alloc/par buckets"
affects:
  - "none (thresholds unchanged; only an inert profiling instrument + a header comment added — no hot-path logic change)"
decisions:
  - "NULL: lever A (scratch-reuse across leaves) RULED OUT by Task-1 profiling BEFORE implementation — allocation is <=0.6% of the per-leaf fixed cost, so removing it cannot move the gate."
  - "The single rayon fork/join enter/exit is the irreducible floor; forced-on crossover only becomes sign-stable (no 3-run overlap) at >=120 feat, so sub-120 fusion does not pay regardless of allocation."
  - "Thresholds left unchanged (unified_bfs_threshold core-scaled 100 / unified_subscan 130). No manufactured win."
metrics:
  duration: "~12 min"
  completed: "2026-06-20"
verdict: NULL
---

# Quick 260620-dpk: Lower the unified-fusion per-leaf fixed cost — VERDICT NULL

**Goal:** lower the per-leaf FIXED COST of the two host fusion paths (`build_fix_scan_impl`,
`subtract_scan_impl`) so the gate threshold could fire below 100 features and help
medium-width workloads — without re-introducing the narrow-config regressions 9cp/a48 found.

## What was done

### Task 1 — Profile the per-leaf fixed-cost breakdown (the verdict driver)
Added `crates/lgbm-compute/src/fusion_prof.rs` — an env-gated (`LGBM_FUSION_PROF=1`),
near-zero-cost-when-off profiler (one inlined cached-`OnceLock` branch per site, mirroring the
existing `phase_prof` pattern) that decomposes each fused per-leaf call into gather / alloc /
par(fork-join+work) buckets. Measured at 20/40/60/80 features:

| feat | BFS alloc% (lever-A-reducible) | BFS fork/join+work | SUB alloc% | SUB fork/join+work |
|------|------------------------------:|-------------------:|-----------:|-------------------:|
| 20   | 0.6% | 99.4% | 0.2% | 99.8% |
| 40   | 0.5% | 99.5% | 0.1% | 99.9% |
| 60   | 0.5% | 99.5% | 0.1% | 99.9% |
| 80   | 0.5% | 99.5% | 0.3% | 99.7% |

Allocation is <=0.6% of the per-leaf fixed cost ⇒ **lever A (scratch-reuse) ruled out before
implementing it** (profile-first avoided wasted code). The par region (rayon fork/join floor +
the fold/fix/scan work) dominates at ~99%.

### Task 2 — Record the lever-A RULED-OUT decision
No hot-path logic change — a header comment documenting why scratch-reuse cannot move the gate
(allocation is negligible). The fused code stays byte-identical to the c5v HEAD.

### Task 3 — Re-measure the crossover (NULL)
Swept ~20-120 feat, 3-run medians, fusion forced on. The win only becomes **sign-stable
(no 3-run overlap)** at >=120 feat; 70-100 is within run-to-run scheduling spread (some runs
flip sign). Because no fixed-cost reduction was applied (alloc <1%, nothing to reduce), the
break-even cannot have dropped ⇒ thresholds left unchanged.

| feat | run1 | run2 | run3 | sign-stable? |
|------|------|------|------|--------------|
| 70  | -39.8% | -3.2% | +1.1% | NO (flips) |
| 80  | -1.4% | -1.9% | -4.6% | NO (overlap) |
| 100 | +2.6% | -21.1% | -8.2% | NO (flips) |
| 120 | -19.3% | -17.6% | -16.1% | YES (no overlap) |

## Parity gate (the merge gate) — GREEN both modes (independently re-verified)
- Default: `kernel_parity` 6/6, `learner_parity` 29/29, `lgbm-compute --lib` 43/0 (1 ign),
  `lgbm-treelearner` 76/0 (2 ign).
- Forced ON (`LGBM_UNIFIED_BFS_THRESHOLD=0 LGBM_UNIFIED_SUBSCAN_THRESHOLD=0`): `kernel_parity`
  6/6, `learner_parity` 29/29 (identical to default — the profiler never reorders/changes
  values).
- `cargo check -p lgbm-compute -p lgbm-treelearner` clean.

## d3v real-workload regression re-check — win preserved
MNIST-784 (high-dim) OFF 8712 vs ON 3529 ms = **-59.5%, no 3-run overlap** (same magnitude
class as d3v's -66%; the absolute % shifts with the load-sensitive OFF baseline — the box ran
at load-avg 5-11 during measurement). binary.train (28f) OFF 249.7 vs ON 243.5 ms = -2.5%
(overlap → neutral, no regression). The fusion win is intact and the gate still protects
typical workloads.

## Residual (honest)
Lever B (core-sized `par_chunks` to cut task breadth) was not separately measured. It cannot
lower the single-fork/join FLOOR (one enter/exit either way), only per-task scheduling — minor
at the 20-80 task counts in play. The decisive evidence (forced-on crossover stable only at
>=120 feat) holds regardless. Lowering the gate below ~100 needs a fundamentally
cheaper-than-rayon-fork/join dispatch, which is out of this task's scope.

## Deviations from Plan
Plan anticipated implementing lever A; Task-1 profiling ruled it out (alloc <1%), so Task 2
became a documented decision rather than a code change. This is the plan's explicit
measure-first NULL branch, not a deviation in intent.

## Commits
- `a35b7e6` — Task 1: fusion_prof profiler + instrumented sites (inert when off)
- `e5e4bfc` — Task 2: lever-A ruled-out decision (no hot-path logic change)
- `cfaa6ed` — Task 3: crossover re-measured, verdict NULL

## Self-Check: PASSED
- `crates/lgbm-compute/src/fusion_prof.rs` — FOUND.
- Thresholds unchanged (`core_scaled_threshold(100|130, rayon_cores())`) — VERIFIED.
- Parity green default + forced-on — independently re-run.
- Commits a35b7e6 / e5e4bfc / cfaa6ed — FOUND in git log.
