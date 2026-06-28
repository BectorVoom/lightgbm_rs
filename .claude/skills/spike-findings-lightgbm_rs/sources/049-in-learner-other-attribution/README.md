---
spike: 049
name: in-learner-other-attribution
type: standard
validates: "Given the post-metric-fix CUDA gap, when in_learner_other is sub-attributed on real NVIDIA hardware, then the remaining levers are identified before planning"
verdict: VALIDATED
related: [046, 048]
tags: [gpu, cuda, kaggle, profiling, attribution, in-learner-other, pre-plan, measure-dont-model]
---

# Spike 049: Attribute `in_learner_other` before planning the GPU long-pole

## Why
`/gsd-plan-phase` for the spike-048 architectural gap surfaced that two chunks were
never separately attributed on real CUDA: `in_learner_other` (~2.57s in v5) and the
Python-side marshalling (~3s). Per the project's measure-before-plan discipline, this
spike attributes `in_learner_other` first (the chunk inside the tree learner, most
relevant to a GPU-learner rework).

## Method
Added env-gated, parity-neutral `phase_prof` sub-counters (`ROOT_FOLD`,
`PARTITION_NEW`, `SCRATCH`, `RESIDENT_RESET`) splitting `in_learner_other =
learner − before − hist+split − partition`. The CPU components are
backend-independent, so most of the breakdown is valid from a LOCAL run; only the
GPU-specific `resident_reset` needed Kaggle. Validator made size-configurable
(`LGBM_VALIDATE_ROWS/ITERS`).

## Results

### Local CPU (500k×50, 30 trees) — the backend-independent components
```
IN_LEARNER_OTHER=289ms = root_fold 10 + partition_new 10 + scratch 0.2 + residual 269
```

### Real CUDA, Kaggle (500k×50, 100 trees, this session wall=11.07s)
```
IN_LEARNER_OTHER=1637.9ms = root_fold 71.7 + partition_new 28.9 + scratch 0.8
                          + resident_reset 0.308 + resident_bin_upload 31.7 + residual 1504.6
BUDGET: learner=7512.9ms (phases=5875.0  in_learner_other=1637.9)
LOOP:   train_one_iter=8291.1ms  metric=0.000ms   (metric fix confirmed again)
COUNTS: device_launches=8570  scan_roundtrips=2890
```

### Two hypotheses REFUTED
1. **`resident_reset` is NOT the GPU lever** — 0.308ms over 100 trees (negligible).
   The ~16ms/tree CPU-vs-GPU `in_learner_other` delta is NOT the device pool reset.
2. **`in_learner_other` is NOT a clean single lever** — ~92% is diffuse `residual`
   on BOTH CPU (269/289) and GPU (1504/1638). It's spread across per-tree growth-loop
   scaffolding (child leaf-splits seeding, arg_max, histogram-pool management, tree
   finalization) — `CegbModel::new`, `root_fold`, `partition_new`, `scratch` are all
   negligible. No high-ROI single fix.

## The measured post-metric-fix wall map (≈11.07s, real CUDA, 500k×50, 100 trees)

| Chunk | Time | % wall | Lever |
|---|---|---|---|
| **GPU histogram phases** (build+scan+partition) | 5.88s | **53%** | architectural — on-device monolithic learner (the big swing; 09–13 already tuned the kernels) |
| **Python marshalling / binding** (wall − train_one_iter) | 2.78s | **25%** | UNATTRIBUTED — likely the next *easy* win (numpy→corpus conversion in pyo3); separate from GPU architecture |
| `in_learner_other` (diffuse per-tree scaffolding) | 1.64s | 15% | no single lever; incremental only (reuse child leaf-splits buffers) |
| grad + score + snapshot + other | 0.77s | 7% | low |

(metric = 0 — the quick-260628-f57 fix; was 26% pre-fix.)

## Signal for planning
- The dominant remaining cost (53%) is the **GPU histogram phases** — closing it means
  the architectural on-device tree-learner (matching official's
  `CUDASingleGPUTreeLearner`). Big, milestone-sized, high-uncertainty swing.
- The **Python marshalling ~25%** is the highest-ROI *un-attributed* chunk and is a
  different subsystem (the pyo3 binding, not the GPU learner). It plausibly hides an
  easy win like the metric fix did — **attribute it next** before committing a plan.
- `in_learner_other` is a dead end for a single lever (resident_reset refuted, residual
  diffuse). Do NOT scope a phase around it.

## Verdict
**VALIDATED.** `in_learner_other` attributed and de-prioritized (diffuse, no lever).
The plan should target the GPU phases (architectural) and/or first spike the Python
marshalling (~25%, likely the next easy win). Evidence: `kaggle-run/lgb-rs-cuda-bench.log`.
