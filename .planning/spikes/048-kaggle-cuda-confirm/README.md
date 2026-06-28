---
spike: 048
name: kaggle-cuda-confirm
type: standard
validates: "Given the spike-046-instrumented wheels on Kaggle (real NVIDIA discrete GPU), when run as lgb_rs CPU-vs-CUDA + official CPU/CUDA at 500k×50 under LGBM_PHASE_PROF=1, then attribute the ~6× CUDA slowdown to phases and identify the fix"
verdict: VALIDATED
related: [046, 014, 015, 021, 024, 040]
tags: [gpu, cuda, kaggle, profiling, attribution, narrow-shape, metric-eval, diagnose, fix]
---

# Spike 048: Attribute the Kaggle-CUDA slowdown (500k×50)

## What This Validates

Real discrete-NVIDIA attribution of lightgbm_rs's ~6× CUDA slowdown, via the
spike-046 `phase_prof` hook on the shipped Python path. **The hypothesis going in
(per-leaf scan sync floor) was REFUTED; the real picture is a 3-way split with one
big easy win.**

## The numbers (Kaggle, real NVIDIA GPU, 500k×50, 100 trees, num_leaves=31)

| Backend | Wall | vs official CUDA |
|---|---|---|
| official LightGBM CUDA | **3.26 s** | 1.0× |
| official LightGBM CPU | 7.52 s | — |
| **lightgbm_rs CUDA** | **17.04 s** | **5.2× slower** |
| lightgbm_rs CPU | 22.80 s | 7.0× slower |

### Surprise #1 — lgb_rs CUDA (17.04s) BEATS lgb_rs CPU (22.80s) on Kaggle.
This **REFUTES the "route narrow shapes to CPU" fix.** On the 16-core dev box the
CPU anchor wins ([[perf-gap-vs-cpp-40-80x]]); on Kaggle's ~4-vCPU box the
multi-threaded CPU advantage evaporates and the GPU is the better lgb_rs path.
Routing is environment-dependent, not a universal win.

## Attribution of lgb_rs CUDA 17.04s (timed run, phase_prof BUDGET/LOOP/COUNTS)

```
LOOP:   train_one_iter=9590ms (grad=477 learner=8791 score=132 snapshot=128) metric=4489ms
phases: hist+split=4508ms (build=0 scan=286) partition=1710ms          (= within learner)
COUNTS: device_launches=8570 (build_resident=2890 subtract_resident=2790 scan_resident=2890) syncs=2890
BUDGET: grad=477 learner=8791 (phases=6218 in_learner_other=2573 [resident_bin_upload=33]) score=132
```

| Chunk | Time | % wall | Attackability |
|---|---|---|---|
| **GPU tree-learning** (`learner`) | **8.79 s** | 52% | HARD (architectural) |
| — hist+split (on-device build+subtract) | 4.51 s | 26% | medium |
| — partition | 1.71 s | 10% | medium |
| — in_learner_other (launch/orchestration) | 2.57 s | 15% | the per-leaf host loop |
| **Training-metric eval** (host) | **4.49 s** | 26% | **EASY — pure waste** |
| Binning + numpy marshalling + setup | ~3.0 s | ~18% | medium (partly bench artifact) |
| grad + score | 0.6 s | 4% | low |

### Surprise #2 — the scan sync floor is NOT the bottleneck.
2890 scan round-trips (~29/tree) cost only **scan=286ms (1.7% of wall)** — even on
real discrete PCIe. The spike-021/024 feature-per-lane + sibling-copack work already
mitigated it. My going-in hypothesis was wrong; **measure-don't-model**, again.

### Surprise #3 — 26% of the wall is host-side metric eval that official skips.
`metric=4.49s` is identical on the CPU run (4489ms) and CUDA run (4489ms) — it is a
backend-independent **host** cost: `binary_logloss` over 500k rows **every iteration**.
Root cause `booster.rs:1291`:
```rust
let provide_train = config.is_provide_training_metric || valid.is_none();
```
With no `eval_set`, `valid.is_none()` forces per-iter training-metric eval. Official
LightGBM defaults `is_provide_training_metric=false` and computes nothing here →
empty `evals_result_`. **This is a behavioral divergence AND ~26% pure waste.**

Confirmed NOT the cause: `resident_bin_upload=33ms` once (spike-014 redundant
upload already fixed by the `qxl` CudaBackend resident surface).

## The fix (diagnose + prototype-fix scope)

**Win #1 (easy, ~26%): stop the redundant per-iter training-metric eval.**
- Prototype (zero code change): `metric_freq=N_ESTIMATORS` → eval only the last
  iter → see `kaggle_metric_ab.py`. Proves the ~4.49s drop.
- Real fix: make `provide_train` C++-faithful (`= config.is_provide_training_metric`,
  drop `|| valid.is_none()`). **CAVEAT:** this changes the default `eval_history`
  contract — `public_api_train_predict_round_trip` (booster.rs:1604) trains with no
  valid set + `.metric("l2,rmse")` and asserts training history exists. So the real
  fix is a deliberate decision for a build phase (gate on a new flag / verbosity, or
  update the test to match C++), not a silent edit. Projected: 17.04 → ~12.6s, gap
  5.2× → 3.9×.

**Remainder (hard, architectural): GPU tree-learning 8.79s vs official's <3.26s.**
Official's `CUDASingleGPUTreeLearner` grows the whole tree on-device; lgb_rs drives
8570 host-orchestrated launches/train (build/subtract/scan/partition per leaf) with
~2.57s of pure launch/orchestration overhead (`in_learner_other`). This is the
subject of the whole 001–045 campaign and the real long-pole. The narrow 50-feature
shape starves each kernel, so launch overhead dominates compute.

## Investigation Trail
1. kaggle-run-v5 (full matrix) → the 4 wall-clocks + phase_prof for CPU & CUDA.
2. Sync-floor hypothesis refuted (scan=286ms).
3. Route-to-CPU fix refuted (CUDA beats CPU on Kaggle's few-core box).
4. metric=4.49s identical across backends → host waste → traced to booster.rs:1291.
5. metric_freq A/B (kaggle_metric_ab.py) to prove the win without a risky edit.

## Results
**VERDICT: VALIDATED.** The 5.2× decomposes into ~26% removable host metric-eval,
~52% architectural GPU tree-learning (the hard campaign remainder), ~18%
binning/marshalling. Clean prototype win identified (metric eval); the rest is the
known long-pole. See `kaggle-run-v5/lgb-rs-cuda-bench.log` and the A/B result below.

### A/B RESULT — metric-eval win PROVEN (kaggle_metric_ab.py, CUDA wheel only)

Same-session A/B (the trustworthy delta; absolute wall drifts between Kaggle
sessions — this session's baseline was 6.11× vs run-v5's 5.2×, different HW assign):

```
lgb_rs CUDA metric_freq=1   (per-iter eval): 19.903 s   (metric phase = 5359 ms)
lgb_rs CUDA metric_freq=200 (eval last only): 15.343 s  (metric phase =   70 ms)
>>> metric-eval cost removed: 4.560 s (22.9% of wall) <<<
gap vs official CUDA: 6.11× → 4.71×
```

The `metric` phase counter collapses 5359ms → 70ms (only the last-iter eval
survives), and the wall drops by 4.56s. **The ~23–26% host-side metric-eval waste
is confirmed on real NVIDIA hardware, parity-neutral to the trees.** `metric_freq`
is an immediate user-facing workaround; the durable fix is the `provide_train`
change (with the test-contract caveat above).

