---
quick_id: 260621-rsh
title: Cache-friendly binning (transpose-scatter)
status: ready
date: 2026-06-21
---

# Quick Task 260621-rsh: Cache-friendly `build_feature_columns`

## Problem

`build_feature_columns` (booster.rs:160) is ~5s/rep (~⅓ of train) at 1M×500: it does
`num_features` (500) STRIDED column passes over the row-major `Vec<Vec<f64>>` corpus
(`row[j]` jumps 4000B/read, booster.rs:177-196) ⇒ ~500M DRAM-latency-bound reads. Same
pathology as feature_infos (quick-260621-rdu, fixed ~8×).

## Task 1 — transpose-scatter

- **action:** Replace the per-feature collection loop with ONE pass over rows scattering
  `row[j] as u32` into pre-sized per-feature `cols_bins[j]` (streaming contiguous reads +
  ~num_features*64B L2-resident write tails). The per-feature max/seen/counts/modal/
  `BinColumn::new` run unchanged over each now-contiguous `cols_bins[j]`. Happy path
  byte-identical (same values + per-feature order). Validation now row-major-discovery
  order — same `LgbmError::InvalidCorpus` variant + same checks (row.len, non-neg-int).
- **verify:** build ±rocm; run `LGBM_BENCH_SWEEP=wide LGBM_PHASE_PROF=1
  LGBM_BENCH_ROWS=1000000 LGBM_BENCH_FEAT=500 LGBM_BENCH_ITERS=4` — BINNING_NS bucket must
  drop materially (target like feature_infos' ~8×); else NULL+revert (spike-011: scatter
  can lose — measure).
- **done:** binning bucket down, train wall-clock down, OR honest NULL.

## Task 2 — parity gate

- **verify:** `lgbm-treelearner --lib`, `lgbm-boosting --lib`, `oracle-harness` (model-text
  / per-iter / raw_bin goldens), `lgbm` in-crate booster tests, `oracle-harness
  --features rocm`. Build clean ±rocm. Bit-exact (FeatureColumns byte-identical).
- **done:** all green.

## must_haves

- **truths:** transpose moves strided access read→write; 32KB tail working set is
  L2-resident single-threaded (spike-011's loss was PARALLEL false-sharing); happy-path
  bins byte-identical; only `is_err()` is tested (booster.rs:1929), not the error (j,row).
- **key_links:** booster.rs:160-242 (build_feature_columns), phase_prof BINNING_NS.
