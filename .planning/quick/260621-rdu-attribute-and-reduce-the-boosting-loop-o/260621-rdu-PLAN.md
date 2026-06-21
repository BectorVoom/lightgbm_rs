---
quick_id: 260621-rdu
title: Attribute & reduce the boosting-loop overhead at 1M×500
status: ready
date: 2026-06-21
---

# Quick Task 260621-rdu: Boosting-loop overhead (measurement-first)

## Problem

At 1M×500 GPU train (iters=4), ~25% of wall-clock (~4.0s/rep: train ≈15.99s − binning
≈5.13s − learner ≈6.88s) is in neither the binning nor the learner bucket. spike-014b
*hypothesized* per-iter `to_vec` clones of the score buffer — but a 1M-row f64 score Vec
is only ~8MB (~sub-ms to copy), so a few clones/iter cannot be 4s. **Measure before fixing.**

## Task 1 — attribute the ~4s/rep (phase_prof BUDGET extension)

- **action:** Add env-gated (`LGBM_PHASE_PROF=1`) coarse timers to `phase_prof`:
  `TRAIN_ONE_ITER_NS` (wrap the whole `gbdt.train_one_iter` call in booster.rs loop),
  `SNAPSHOT_NS` (the `scores().to_vec()` clones in gbdt.rs — `train_score_pre`:719 +
  IterSnapshot.score:1044/1066/1303), `METRIC_NS` (the booster-loop `m.eval` block).
  Dump line gains: `train_one_iter`, `loop_other = train − binning − Σtrain_one_iter`,
  `in_iter_other = train_one_iter − grad − learner − score − snapshot`, `snapshot`, `metric`.
- **verify:** build clean; run `LGBM_BENCH_SWEEP=wide LGBM_PHASE_PROF=1
  LGBM_BENCH_ROWS=1000000 LGBM_BENCH_FEAT=500 LGBM_BENCH_ITERS=4`.
- **done:** the ~4s/rep is attributed to named buckets (snapshot / metric / grad-alloc /
  boost / loop-accumulation).

## Task 2 — reduce what dominates (retargeted by Task 1)

- **action:** apply the cheapest parity-neutral reductions the numbers justify. Baseline
  safe win regardless: guard the `train_score_pre` clone behind `is_renew_tree_output()`
  (wasted for L2). If snapshot/accumulation or metric eval dominates and is not needed for
  a metric-less production train, gate it (off-by-default, like the per_bin_gains opt-in,
  quick-260608-jpj) — never change numerics.
- **verify:** bit-exact gate green (`lgbm-treelearner --lib`, `lgbm-boosting --lib`,
  `oracle-harness` incl. L2 per-iter golden / golden-replay that consume IterSnapshot.score;
  `oracle-harness --features rocm` if shared code changed); build ±rocm; re-measure 1M×500.
- **done:** measured wall-clock reduction OR honest NULL (ship only the wasted clones).

## must_haves

- **truths:** the 8MB clone math rules out clones-as-4s a priori ⇒ measure; `train_score_pre`
  is wasted for L2; snapshot.score feeds golden-replay (parity-load-bearing — gate, don't drop).
- **artifacts:** phase_prof BUDGET timers; whatever reduction Task 1 justifies.
- **key_links:** gbdt.rs:719/665/1044/1066/1303 (clones/alloc), booster.rs:1267 loop +
  metric eval ~1327, phase_prof.rs.
