---
quick_id: 260621-rsh
title: Cache-friendly binning (transpose-scatter)
status: complete
date: 2026-06-21
---

# Quick Task 260621-rsh — Summary

## What changed

`build_feature_columns` (booster.rs) collected each feature's bins with `num_features`
(500) **cache-hostile column passes** over the row-major `Vec<Vec<f64>>` corpus (`row[j]`
strided `num_features*8` = 4000 bytes/read) — ~500M DRAM-latency-bound reads, ~⅓ of
1M×500 train (the same pathology fixed in feature_infos, quick-260621-rdu).

**Fix (transpose-scatter):** ONE pass over rows — each contiguous `row` scatters
`row[j] as u32` into pre-sized per-feature bin vectors `cols_bins[j]`. Streaming reads
(bandwidth-bound, prefetch-friendly over the 4GB matrix) + the ~`num_features*64B` (~32KB)
hot write tails stay L2-resident. Single-threaded, so spike-011's parallel-scatter
false-sharing loss does not apply. The per-feature max / seen-check / counts / modal-bin /
`BinColumn::new` run unchanged over each now-contiguous `cols_bins[j]`.

**Byte-identical:** happy-path bins carry the same values in the same per-feature row
order ⇒ `FeatureColumn`s byte-identical ⇒ bit-exact trees. Validation moved to row-major
discovery order (same `LgbmError::InvalidCorpus` variant + `row.len`/non-negative-integer
checks); only `is_err()` is asserted (booster.rs test), not the specific `(feature,row)`.

## Verification

**Parity (HARD gate) — all GREEN:** `lgbm --lib` 41/0, `lgbm-boosting --lib` 55/0,
`lgbm-treelearner --lib` 76/0, `oracle-harness` cpu (model-text / per-iter / raw_bin
goldens that consume the binned FeatureColumns) + `oracle-harness --features rocm` all
pass. Build clean ±rocm. CPU f64 anchor untouched.

**Speed (gfx1100, bench_gpu_vs_cpu wide, 1M×500 iters=4):**
- binning bucket: **~4900 → ~2100 ms/rep (~2.3×, −57%)** (3-run stable 2092/2135/2108).
- train: **~12.8 → ~9.5 s (−26%)** (3-run stable 9.60/9.50/9.43), rows/s 78k → 106k
  (**+36%**).

It's ~2.3× (not feature_infos' ~8×) because binning's scatter-WRITE side (500 destinations
+ per-element validation/cast/push) costs more than feature_infos' read-only min/max into
2 small arrays — but it's a clear win, not the spike-011 wash the NULL-clause guarded against.

**Cumulative GPU wide-shape campaign** (spike-014 → p9v → qix → rdu → rsh), 1M×500 iters=4:
**29.55 s → ~9.5 s (−68%)** — once-per-train upload hoist + native-width upload +
cache-friendly feature_infos + cache-friendly binning.

## Honest notes

- Like feature_infos, binning is **per-train host setup** — it amortizes if the binned
  dataset is reused across trains (the bench rebuilds per `train()` call). The ~2.3×
  cache-locality win is real regardless.
- Peak host memory during the pass: ~2GB of `u32` bin columns held simultaneously (the
  transpose needs all features collected before narrowing) — same order as the prior
  retained FeatureColumns; fine at 32GB.
- The RawCorpus path bins via the bit-exact `BinMapper` (separate code) and is untouched.
- Remaining 1M×500 bucket: `learner` (~7s/rep — the GPU histogram, covered by the closed
  kernel levers). The host per-train setup (binning + feature_infos) is now ~2.6s/rep
  combined, down from ~9s.
