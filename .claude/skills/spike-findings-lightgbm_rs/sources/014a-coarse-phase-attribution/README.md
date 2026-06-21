---
spike: 014a
name: coarse-phase-attribution
type: standard
validates: "Given a 1M×500 GPU train, when the wired phase_prof dump runs, then one phase (build/scan/partition) is shown to dominate wall-clock — settling whether the histogram kernel is even the bottleneck at the wide shape"
verdict: PARTIAL
related: [001, 007]
tags: [performance, gpu, rocm, profiling, attribution, wide-shape]
---

# Spike 014a: Coarse Phase Attribution at the Wide Shape (1M×500, GPU)

## What This Validates

Given a never-before-benched 1M×500 GPU train, when the existing `phase_prof`
per-phase timer dump runs, then we learn **which phase dominates** — and thereby
whether "the GPU histogram kernel is the bottleneck on large data" is even true at
the wide shape, before building any new fine-grained kernel instrumentation.

## How to Run

```bash
# Build the rocm bench:
cargo build --release --features rocm --example bench_gpu_vs_cpu

# Wide sweep (250k/500k/1M × 500) with per-phase attribution:
LGBM_BENCH_SWEEP=wide LGBM_PHASE_PROF=1 \
  ./target/release/examples/bench_gpu_vs_cpu

# Single custom point + iters override (used for the fixed-vs-per-iter A/B):
LGBM_BENCH_SWEEP=wide LGBM_PHASE_PROF=1 \
  LGBM_BENCH_ROWS=250000 LGBM_BENCH_FEAT=500 LGBM_BENCH_ITERS=8 \
  ./target/release/examples/bench_gpu_vs_cpu
```

Changes shipped to the production bench (`crates/lgbm/examples/bench_gpu_vs_cpu.rs`):
- New `LGBM_BENCH_SWEEP=wide` mode: feat=500, rows {250k,500k,1M} (env-overridable
  via `LGBM_BENCH_ROWS`/`LGBM_BENCH_FEAT`/`LGBM_BENCH_ITERS`); lighter warmup/reps
  (1/3) since the per-phase **ratio** is the deliverable, not a tight median.
- Wired the existing `lgbm_treelearner::phase_prof::dump` (inert unless
  `LGBM_PHASE_PROF=1`): `warmup-discard` reset after warmup, per-size dump after the
  timed reps — same pattern as `bench_real.rs`.

## What to Expect

A `[phase_prof:<size>]` line per size giving before / hist+split (build+scan) /
partition in ms and %. The hypothesis under test: build (the histogram kernel)
dominates.

## Investigation Trail

**1. First run (250k×500) — the surprise.** `build=0.000ms`, `scan=93.1%`,
partition=5.1%. The histogram-build timer reads *zero*. Two anomalies:
- `build=0` despite a histogram obviously being built.
- `phase_prof total = 3.88s/rep` but `train_median = 8.6s/rep` → **~55% of GPU
  train wall-clock is outside the three instrumented phases entirely.**

**2. `build=0` is a fusion-labeling artifact, not a finding.** `build_leaf_histogram_into`
*is* wrapped by `BUILD_NS` (`learner.rs:1792`), so `build=0` means that function is
**never called** at 500 features. Reading `learner.rs:1523-1612,1955-1986`: at
`features.len() >= unified_bfs_threshold()` (default **100**), the standalone build is
SKIPPED — on the resident/GPU path `fused_build=true` runs ONE
`build_fix_scan_resident` device launch that **builds + fixes + compacts + scans** the
leaf in a single launch, all timed under `SCAN_NS` (`learner.rs:1986`). So at the wide
shape **the GPU histogram kernel is folded into "scan."** `scan=93%` *contains* the
kernel — `build=0` is an artifact of the resident fusion (the L3 on-GPU FixHistogram +
p90 resident-pool work), not evidence the kernel is cheap.

**3. The ~55% uninstrumented gap grows with rows.** Per-size, train vs instrumented:

| shape | train/rep | instrumented | scan% (of instr.) | **uninstrumented** |
|-------|-----------|--------------|-------------------|--------------------|
| 250k×500 | 8.60s | 3.88s | 93.1 | **4.72s (55%)** |
| 500k×500 | 13.69s | 5.25s | 92.4 | **8.44s (62%)** |
| 1M×500 | 26.21s | 8.12s | 93.1 | **18.1s (69%)** |

The uninstrumented fraction **rises 55%→69%** as rows scale. So the kernel
(scan-fused) is **at most ~31% of true wall-clock at 1M×500**, and a *growing
majority* is work the growth-loop timers never see.

**4. What is the uninstrumented majority? — fixed setup vs per-iteration.**
`train()` → `train_inner_full` calls `build_feature_columns(corpus)` **once per
train** (`booster.rs:966`) — an O(rows×features) host re-binning pass BEFORE the
timed iteration loop (`booster.rs:1265`). Hypothesis: the gap is per-train binning
repeated every rep. Tested with an iters A/B at 250k×500:
- iters=1 → 3.63s; iters=8 → 18.15s.
- per-iteration cost = (18.15−3.63)/7 = **2.07s/iter**.
- fixed setup (binning + objective + learner/device setup) = 3.63−2.07 = **~1.56s**.

So fixed binning is only **~18%** at iters=4 — real, and a **bench artifact** (it
amortizes in bin-once-train-many usage), but **not** the dominant gap. Of the ~2.07s
**per iteration**, only ~0.97s lands in instrumented phases → **roughly half of every
iteration is uninstrumented** (boosting-loop orchestration, per-tree GPU
gradient/hessian upload + sync, partition/pool bookkeeping outside the guards).

## Results

**Verdict: ⚠ PARTIAL — coarse attribution obtained, and it overturns the framing.**

1. **The histogram kernel is NOT shown to be the bottleneck — and is likely
   overstated.** At 1M×500 the kernel is folded into "scan" and is **≤31% of true
   train wall-clock**. The hypothesis "GPU kernel is the large-data bottleneck" is
   **not established** by the data; the wall-clock is split three ways:
   kernel(scan-fused) + a *growing* uninstrumented per-iteration overhead + a fixed
   per-train binning setup.
2. **Existing tooling cannot answer build-vs-scan-vs-transfer at the wide shape.**
   `phase_prof` (a) folds build into scan via resident fusion (`build=0` is an
   artifact), and (b) covers only 31–45% of train wall-clock. It is structurally
   inadequate for GPU attribution. **New instrumentation is required** — confirming
   the precondition 014b was scoped for, now with a sharper target.
3. **Surprises:** (a) `build=0` despite heavy histogram work; (b) the uninstrumented
   fraction *grows* with rows (55→69%) rather than being a fixed tax; (c) ~half of
   every iteration is invisible to the current timers.

**Retargets 014b** (was "separate launch/compute/readback in the histogram kernel"):
1. Add a **whole-train / whole-iteration budget** that splits binning-setup vs
   boosting-loop vs growth-phases vs device-transfer — to attribute the uninstrumented
   ~half, the single biggest unknown.
2. Instrument **inside the resident `build_fix_scan_resident` launch** to separate true
   device-compute from host-readback within the scan-fused span.
3. Only then is "is it slow compute vs sync round-trip vs transfer" answerable.

**Gate note (CONVENTIONS):** measurement-only; `phase_prof` is inert unless
`LGBM_PHASE_PROF=1`; the CPU f64 anchor / bit-exact merge gate is untouched.
