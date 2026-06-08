---
quick_id: 260608-jpj
slug: make-d-06-snapshot-opt-in-r1-to-speed-up
status: complete
date: 2026-06-08
---

# Quick Task 260608-jpj — R1: D-06 snapshot opt-in (SUMMARY)

## What was done

Made the per-split D-06 snapshot (`per_bin_gains` host re-scan) **opt-in** so the
production boosting path stops computing golden-replay data it discards.

- Added `capture_snapshots: bool` to `SerialTreeLearner` (default false).
- Threaded a `capture` arg into `train_inner`; **decoupled `train()` from
  `train_with_snapshots`** (it used to delegate, which would have forced capture on).
- Wrappers: `train` / `train_returning_partition` (the boosting path, gbdt.rs:
  797,1125) → capture **false**; `train_with_snapshots` /
  `train_with_col_sampler_trace` (golden replay) → capture **true**.
- Gated the `per_bin_gains` call in `scan_leaf_histogram` on the flag (empty cand
  arrays when off).

## Safety (proven from source trace, then gate-verified)

The live split decision is `backend.find_best_split` → `best_split_per_leaf`; the
splittability gate is `this_leaf_splittable`. `per_bin_gains` (a pure read) feeds
ONLY `FeatureSplitRecord.cand_*` → `SplitSnapshot`, which the production paths
discard. ⇒ gating it is bit-identical for the grown tree.

## Parity gate — GREEN

`cargo test -p oracle-harness`: **learner_parity 29** + **boosting_parity 75** (the
D-06 snapshot replays, capture ON) + kernel/predict/raw_bin/rng all pass, 0 failed.
Core units: lgbm 41, treelearner 64. **Zero numeric change.**

## Result — honest

**No measurable wall-clock win (~0%, within noise):** large train M3 8.12s vs M4
~8.13s (median of 3). `per_bin_gains` is just mul/div over `num_bin` cells —
dwarfed by the histogram-construction gather. **The earlier REPORT framing of R1 as
"the largest win" was wrong; R2 (batched histogram dispatch) is the dominant cost.**

## Value retained despite ~0% here

- Parity-correct + architecturally cleaner (production no longer computes discarded
  golden snapshots).
- Its relative weight grows once R2 shrinks the dominant cost.

## Next

**R2** is the real lever: the per-feature-per-leaf `construct_histograms` host
gather → CubeCL-CPU buffer create → launch → readback dominates. Batch the dispatch
(all features per leaf in one launch) and/or keep bin data device-resident. Bigger
effort, needs its own plan + the same bit-exact gate.
