---
quick_id: 260621-rdu
title: Attribute & reduce the boosting-loop overhead at 1M×500
status: complete
date: 2026-06-21
---

# Quick Task 260621-rdu — Summary

## The measurement-first payoff (hypothesis refuted twice)

The ~25% of 1M×500 train wall-clock unaccounted by the spike-014b BUDGET
(binning+learner) was hypothesized to be per-iter `to_vec` clones of the score buffer.
**Measurement killed that** — and the "boosting-loop" framing too:

| bucket (per rep, 1M×500 iters=4) | time | verdict |
|---|---|---|
| snapshot (`scores().to_vec()` ×4/iter) | **3.2 ms** | NOT it (8MB clone math was right a priori) |
| metric eval (`m.eval` over 1M rows) | 43 ms | small |
| in_iter_other (boost/bagging/alloc) | 2 ms | negligible |
| `train_one_iter` − learner | ~20 ms | learner IS train_one_iter |
| **`feature_infos_from_rows` (model-metadata setup)** | **~3929 ms** | **THE cost (~24% of train)** |

The overhead is **not in the boosting loop at all** — it's a per-train model-metadata
pass (`feature_infos_from_rows`, booster.rs) computing per-feature min/max, doing
`num_features` (500) **cache-hostile COLUMN passes** over a **row-major** `Vec<Vec<f64>>`
(`row[j]` strides `num_features*8` = 4000 bytes/read) ⇒ ~500M cache-missing reads.

## Fix

One **cache-friendly single pass** over the row-major matrix accumulating per-feature
min/max arrays (each `row` read contiguously), instead of 500 strided column passes. Uses
the **same `f64::min`/`f64::max`** calls, only loop order changed — min/max are
commutative + associative ⇒ **byte-identical** `feature_infos` (the model-text line is
parity-checked). Also adds env-gated (`LGBM_PHASE_PROF=1`) `phase_prof` LOOP buckets
(`train_one_iter`/`snapshot`/`metric`/`feature_infos_setup`) — the instrumentation that
found this; inert in production.

## Verification

**Parity (HARD gate) — all GREEN** (the model-text golden directly checks `feature_infos`):
- `lgbm-boosting --lib` 55/0, `lgbm-treelearner --lib` 76/0.
- `oracle-harness` (cpu, incl. model-text / per-iter golden) all pass; `oracle-harness
  --features rocm` all pass. CPU f64 anchor untouched; build clean ±rocm.

**Speed (gfx1100, bench_gpu_vs_cpu wide, 1M×500 iters=4):**
- `feature_infos_setup`: **~3929 → ~490 ms/rep (~8×)** (3-run stable ~470–540 ms).
- train: **~15.9 → ~12.8 s (~−20%)** (3-run stable 12.69/12.82/12.85), rows/s 61k → 78k
  (**+29%**).

**Cumulative GPU wide-shape campaign** (spike-014 → p9v → qix → rdu), 1M×500 iters=4:
**29.55 s → ~12.8 s (−57%)** — once-per-train upload hoist (p9v) + native-width upload
(qix) + cache-friendly feature_infos (rdu).

## Honest notes

- `feature_infos_from_rows` is a **per-train** O(rows×features) host pass — like binning,
  it amortizes if the dataset/model-metadata is reused; the bench rebuilds per train. The
  ~8× is a real cache-locality win regardless (single contiguous pass vs 500 strided).
- The RawCorpus path (`feature_infos_from_columns`) was already column-contiguous — only
  the DenseCorpus row-major path had the pathology.
- Remaining 1M×500 buckets after rdu: learner (~7s/rep, the GPU histogram — addressed by
  the closed kernel levers) + binning (~5s/rep, per-train host setup, bench artifact). No
  cheap parity-neutral lever left in the boosting loop itself (snapshot/metric are ms-scale).
