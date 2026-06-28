---
spike: 046
name: python-path-phase-prof
type: standard
validates: "Given the shipped train() path (what the Python/CUDA wheel drives), when LGBM_PHASE_PROF=1, then it emits the full per-phase BUDGET/LOOP/COUNTS attribution — the enabler for diagnosing the ~6x Kaggle-CUDA slowdown at 500k×50"
verdict: VALIDATED
related: [014, 015, 021, 023, 024, 040]
tags: [gpu, cuda, kaggle, profiling, phase-prof, attribution, narrow-shape]
---

# Spike 046: Make the Python/CUDA train path observable

## What This Validates

**Given** the shipped `lgbm::train()` path (the one the Python wheel + Kaggle
benchmark drive), **when** `LGBM_PHASE_PROF=1` is set, **then** it emits the full
per-phase `BUDGET` / `LOOP` / `COUNTS` attribution to stderr — so the ~6× CUDA
slowdown (official 3.43s vs lightgbm_rs 20.88s at 500k×50, 100 trees) can be
attributed to a phase instead of guessed.

## Why this is the enabler

The prior Kaggle run (`kaggle_out/lgb-rs-cuda-bench.log`) confirmed the 6.1× but
produced **zero attribution** — only two wall-clock totals. Root cause found in
code: `phase_prof::dump()` was wired ONLY into the Rust bench examples
(`bench_gpu_vs_cpu`, `bench_real`, `bench_crossover`) — **never** into
`booster.rs`, which is the path the Python wheel uses. So the shipped path was a
profiling black box; setting `LGBM_PHASE_PROF=1` on the Python benchmark would
accumulate the counters and print nothing.

**The fix (parity-neutral, env-gated):** one line in
`booster.rs::train_inner_columns_full`, right before it returns the `Booster`:

```rust
lgbm_treelearner::phase_prof::dump("train");
```

`dump()` is inert unless `LGBM_PHASE_PROF=1` (an internal `OnceLock` env gate); it
only prints to stderr and swap-resets the accumulators. It never touches train
semantics, so it cannot affect the bit-exact parity gate.

## How to Run (local validation — already passed)

```
LGBM_PHASE_PROF=1 cargo run --release --example spike046_validate
```

(`crates/lgbm/examples/spike046_validate.rs` — a 20k×50 binary train that exercises
the public `train()` path the wheel uses.)

## Investigation Trail

1. Confirmed the headline from the existing log: official LightGBM CUDA = **3.43s**,
   lightgbm_rs CUDA = **20.88s** (6.1×) at 500k×50, 100 trees, num_leaves=31, on a
   real Kaggle NVIDIA GPU.
2. Found the log had no phase data; traced `dump()` to examples-only.
3. Ruled out the spike-014 redundant per-tree bin re-upload as the CUDA culprit:
   the recent `qxl` work made `CudaBackend = GpuBackend<CudaRuntime>` with the FULL
   resident surface (once-per-train `resident_bins_uploaded`, learner.rs:842), so
   bins upload once. (Confirmed `BUDGET ... resident_bin_upload=` will read ~0.)
4. Added the env-gated `dump("train")` hook to the shipped path.
5. First validation run printed all-zero growth phases (`learner=0`, `hist+split=0`)
   — **a degenerate-label artifact**: the test labels produced all-constant
   (no-split) trees, which skip the growth loop. NOT an instrumentation gap.
6. Fixed the validator to a real, balanced signal → full attribution fires:

   ```
   [phase_prof:train] hist+split=43.245ms (build=37.734 scan=4.026) partition=4.612ms
   [phase_prof:train] %: hist+split=90.3 (build=78.8 scan=8.4) partition=9.6
   [phase_prof:train] BUDGET: binning=3.570 grad=1.491 learner=54.721
                       (phases=47.866 in_learner_other=6.855 [resident_bin_upload=0.000]) score=0.196
   ```

   (This is the CPU backend at 20k×50 — build-dominated, as expected. The CUDA
   shape at 500k×50 is what spike 048 hunts; the `COUNTS` line with `scan_roundtrips`
   only populates on a GPU backend.)

## Results

**VERDICT: VALIDATED.** The shipped train path now emits complete per-phase
attribution under `LGBM_PHASE_PROF=1`, parity-neutral. The default build is
unaffected (the dump is a no-op). This unblocks spike 048 (real Kaggle CUDA
attribution).

### Bottleneck hypothesis carried into spike 048

At the **narrow** 500k×50 shape, the prime suspect is the **per-leaf scan
round-trip sync floor** (`SCAN_RESIDENT_CNT`, ~30 blocking `read_one_unchecked`
syncs/tree post-spike-024). Each histogram kernel over 50 features is tiny, so the
GPU is starved and cost ≈ launch+sync **latency** × ~3000 leaves. The whole
001–040 campaign was tuned on a **shared-DDR5 APU where a sync is nearly free**; on
Kaggle's real discrete-PCIe GPU each sync costs 10–20µs+. Official LightGBM's
`CUDASingleGPUTreeLearner` grows the entire tree on-device with no per-leaf host
round-trips — exactly the gap this would explain. Secondary: per-tree grad/hess
upload + Python→Rust binning. The `COUNTS` line + `learner − phases =
in_learner_other` split will confirm or refute on real hardware.

## Files

- `bench_runner.py` — one-backend-per-subprocess runner (rs/off × cpu/cuda).
- `kaggle_bench_instrumented.py` — Kaggle orchestrator: builds the CPU + CUDA
  wheels, runs the full matrix under `LGBM_PHASE_PROF=1`, prints the summary +
  phase breakdown. **Requires the spike-046 patch pushed to the cloned repo**
  (set `REPO_BRANCH` if it lives on a non-default branch).
