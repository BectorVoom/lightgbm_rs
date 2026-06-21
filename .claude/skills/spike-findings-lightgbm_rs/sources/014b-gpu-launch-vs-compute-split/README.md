---
spike: 014b
name: gpu-launch-vs-compute-split
type: standard
validates: "Given the dominant phase from 014a, when launch / device-compute / readback are separated with new GPU timers, then we learn whether the cost is slow device-compute (occupancy/atomics) vs the sync round-trip — pointing to the next real lever"
verdict: VALIDATED
related: [014a, 007, 001]
tags: [performance, gpu, rocm, profiling, attribution, redundant-upload, next-lever]
---

# Spike 014b: Attributing the Uninstrumented Half (whole-train budget)

## What This Validates

014a proved the existing growth-loop timers see only 31–45% of GPU train wall-clock
at the wide shape and that "build" is fused into "scan". This spike adds a **whole-train
budget** to attribute the uninstrumented majority and find **the next real lever** —
retargeted from the original "split launch vs device-compute inside the kernel" once
014a showed the kernel is *not* where the missing time is.

## How to Run

```bash
cargo build --release --features rocm --example bench_gpu_vs_cpu
LGBM_BENCH_SWEEP=wide LGBM_PHASE_PROF=1 \
  LGBM_BENCH_ROWS=1000000 LGBM_BENCH_FEAT=500 LGBM_BENCH_ITERS=4 \
  ./target/release/examples/bench_gpu_vs_cpu        # add nothing for CPU A/B (drop --features rocm)
```

Instrumentation added (all measurement-only, inert unless `LGBM_PHASE_PROF=1`):
- `phase_prof` budget counters `BINNING / GRAD / LEARNER / SCORE` + drill-down
  `UPLOAD_NS`, with a `BUDGET:` dump line (`crates/lgbm-treelearner/src/phase_prof.rs`).
- Seams wrapped: binning (`booster.rs:966`), grad/hess (`gbdt.rs` get_gradients),
  the per-tree `learner.train_returning_partition` call, score-update (`gbdt.rs`), and
  the resident-bin upload block (`learner.rs:826`). `LEARNER ⊇ growth-phases`, so
  `in_learner_other = LEARNER − (before+hist+split+partition)`.

## Investigation Trail

**1. Whole-train budget (per rep, ÷3 reps), GPU @ 1M×500, train=29.7s:**

| bucket | time/rep | % train |
|--------|----------|---------|
| binning (fixed setup) | 5.2s | 17.5% |
| growth phases (scan-fused kernel + partition) | 10.0s | 34% |
| **in_learner_other** (LEARNER − phases) | **10.4s** | **35%** |
| grad + score | 0.04s | ~0% |
| loop overhead (score `to_vec` clones, metrics) | ~4.1s | 14% |

`in_learner_other` is the missing half — and it **grows with rows** (250k→1M: 2.3s→10.4s,
~4.4× for 4× rows) and at 1M is *larger than the growth phases themselves*.

**2. GPU-specific? — CPU-backend A/B (same 250k×500 shape).** `in_learner_other`/iter:
GPU ≈ 586ms vs **CPU ≈ 36ms — a ~16× gap.** Binning is identical across backends
(~1.2s/rep, host pass). ⇒ `in_learner_other` is **GPU device work**, not host
orchestration.

**3. Which device op? — `UPLOAD_NS` drill-down.** The resident-bin upload alone is
**93% of `in_learner_other`** (250k: 7.06/7.58s; 1M: 27.87/29.86s over 3 reps). At
1M×500 the upload = **9.3s/rep = 31% of train wall-clock — equal to the histogram
kernel itself.**

**4. Root cause (code).** `learner.rs:826-831` does, **every `train_inner` call**
(= every tree, every iteration): `features.iter().map(|f| f.bins.to_u32_vec())` (a
`[nf×nd]` u32 alloc = **2GB at 1M×500**), then `Backend::upload_resident_bins`
(`lib.rs:2039`) **concatenates into a second 2GB buffer** and `create_from_slice`
uploads 2GB host→device — with **no dedup** (it overwrites the prior handle). But the
binned columns are **immutable for the whole train** (`Dataset` is read-only after
load). So every tree re-allocates ~4GB host and re-uploads 2GB to the GPU for a buffer
that never changes. GPU-only because `wants_resident_bins()` is `true` on Rocm, `false`
on CpuBackend — exactly matching the 16× A/B.

## Results

**Verdict: ✅ VALIDATED — attribution complete; the original hypothesis is overturned
and a concrete next lever is identified.**

At 1M×500 GPU training, the wall-clock splits roughly into thirds:
- **~31% redundant resident-bin re-upload** — pure waste (immutable bins re-uploaded
  per tree + 2×2GB host allocs/tree). **THE lever.**
- **~31% growth phases** — the actual GPU compute (histogram kernel fused into "scan" +
  partition). This is the part the original "GPU kernel is the bottleneck" framing
  meant; it is real but only one-third, and it is *not* the cheapest win.
- **~18% binning** (a bench artifact — `train()` re-bins every call; amortizes in
  bin-once-train-many usage) + **~17% boosting-loop host overhead** (per-iter `to_vec`
  clones of the 1M score buffer; metric eval).

**Answer to the central question:** the GPU large-data cost is NOT dominated by slow
device-compute or a sync round-trip inside the histogram kernel. It is dominated, in
equal part to the kernel, by a **redundant per-tree host→device upload of the immutable
binned feature matrix.**

**Next lever (high-value, low-risk, for a separate build task — NOT done here):** hoist
`upload_resident_bins` out of the per-tree `train_inner` to a once-per-train step
(first-tree guard, or a `resident_bins_uploaded` flag keyed on the immutable feature
set). Expected ≈ **−31% train wall-clock at 1M×500** and elimination of the 2×2GB/tree
host allocations. Bins are immutable ⇒ the resident handle is reusable across all trees;
parity is unaffected (same bytes, uploaded once). Pair-check: the `to_u32_vec` widening
at `learner.rs:827` can also be cached/skipped (it duplicates the concat in
`upload_resident_bins`).

**Surprises:** (1) the missing half was *not* the kernel at all; (2) it is exactly equal
in magnitude to the kernel, so naive "optimize the kernel" effort would have left half
the achievable win on the table; (3) a comment that reads "one-time per-train upload"
(`learner.rs:812`) is actually per-**tree** because `learner.train` is the per-tree entry
point — a stale-intent bug hiding in plain sight.

**Gate:** instrumentation is behavior-neutral (closures around existing calls; atomics
gated by `LGBM_PHASE_PROF`). Bit-exact gate green — `lgbm-treelearner --lib` 76/76,
`lgbm-boosting --lib` 55/55, `oracle-harness` all parity suites pass (kernel histogram
bit-exact, boosting 75/75, learner parity). CPU f64 anchor untouched.
