# 260620-a48 — Task 1 Feasibility + Design (AUDIT-BEFORE-WIRE)

## Verdict: **PROCEED**

The directly-built (smaller/root) leaf's spine path CAN be fused into a single CpuBackend
`build_fix_scan` region WITHOUT duplicating the PASS-1 spine-gate logic and WITHOUT touching
the subtract-larger / non-spine / RocmBackend / GPU-fused paths. The clean wiring is the **host
f64 analog of the existing GPU `build_fix_scan_resident`** path, which already solves the exact
same problem (build+fix ALL features for the subtract trick, scan only the `scan_active` subset).

## (a) `fix_histogram` per-feature inputs — all available at the seam

`fix_histogram(hist, most_freq_bin, sum_g, sum_h)` (`fix_histogram.rs:50`) needs only
`most_freq_bin` + the leaf RAW sums; `compact_histogram(hist, offset)` (`learner.rs:3296`) needs
`offset`. Both `most_freq_bin` and `offset` are already carried in `BatchedSplitFeature`
(`kernels/split.rs:77`, fields `most_freq_bin`, `offset`) — the SAME struct Pass-1 already builds
for every spine feature, and which `build_fix_scan_resident` already consumes as `all_feats`. The
leaf RAW (un-bumped) `sum_g`/`sum_h` are the `leaf_splits` totals already threaded into both
`build_leaf_histogram_into` and `scan_leaf_histogram`. **No data computed only inside the two-step
build is required** — the per-feature build (`fold_one_feature`, `lib.rs:199`) reads only the bin
column + ordered grads/hess, and fix+compact read only `(most_freq_bin, offset)` + leaf sums.

## (b) PASS-1 spine-gate — reused verbatim, NOT duplicated

The spine-eligibility predicate (col-sampler `used_features`, parent-splittability, ADV-02
interaction, categorical, monotone, extra-trees — `scan_leaf_histogram` lines ~1956-2005) ALREADY
runs inside `scan_leaf_histogram` and produces `spine_batch_index[fpos]` (Some ⇒ spine, None ⇒
inline/gated). The GPU fused branch already converts this to `scan_active[fpos] =
spine_batch_index[fpos].is_some()` (lines ~2051-2052) and passes it to `build_fix_scan_resident`.

The unified CPU path is added as a **4th dispatch branch INSIDE `scan_leaf_histogram`**, right
beside `fused_build` / `resident_slot` / `find_best_splits_batched` (lines ~2026-2089), consuming
the SAME Pass-1 `scan_active` mask. Zero gate-logic duplication: the region builds+fixes EVERY
feature (so the directly-built leaf's histogram in `buf` stays COMPLETE for the subtract-derived
larger child) and scans only the `scan_active` subset, returning spine SplitInfos in `feats` order
exactly like the GPU path. The serial cross-feature argmax loop below is byte-unchanged and still
merges spine + inline (categorical/monotone/extra-trees) results in feature-index order.

## (c) Scope boundary

The directly-built leaf is ALWAYS the smaller/root child on the CpuBackend path
(`smaller_fused == false && resident_eligible == false`, the `else` arm at learner.rs:1546-1556).
The subtract-derived LARGER child (learner.rs:1578-1680) reads `pool.buffer(smaller_slot)` (the
parent's COMPLETE histogram) for `subtract_histograms` — so the unified region MUST build all
features (it does), and the larger child's subtract+scan path is UNTOUCHED. Region scope is exactly
**"spine numeric features (scan) + all features (build) of the smaller/root leaf, CpuBackend, above
threshold"**.

## Chosen wiring

- **lib.rs** `unified_bfs_threshold()` — env `LGBM_UNIFIED_BFS_THRESHOLD`, mirrors
  `par_scan_threshold()` (lib.rs:314), keyed on `feats.len()`. Default = tuned in Task 3.
- **lib.rs** `CpuBackend::build_fix_scan(client, buf: &mut [f64], slot_off, num_bins, leaf_rows,
  gradients, hessians, all_feats, scan_active, cfg, sum_g, sum_h, num_data) -> Vec<Option<SplitInfo>>`:
  ONE rayon `par_iter` over feature positions; each feature folds its OWN private histogram
  (cache-hot), runs fix (`most_freq_bin`) + compact (`offset`) inline (the tiny documented f64
  loops, identical order to the two-step), scans it IF `scan_active[fpos]`, returns
  `(fpos, private_hist, Option<SplitInfo>)`. After join, a SERIAL ordered loop copies each private
  hist into `buf[slot_off..]` (complete buffer for subtract) and assembles the `Vec<Option<SplitInfo>>`
  in fpos order. Build+fix+scan stay co-located on ONE thread per feature; buf assembly + argmax
  stay serial+ordered ⇒ bit-exact by the same per-feature-independence argument as Spike-005 build
  and 9cp scan.
- **learner.rs** routing flag `unified_bfs` at the directly-built seam (learner.rs:1546): when
  CpuBackend (`!smaller_fused && !resident_eligible`) AND directly-built smaller leaf AND above
  threshold, SKIP the standalone `build_leaf_histogram_into` and pass `unified_build=true` +
  `&mut buf` into `scan_leaf_histogram`, which runs the unified region. Else the byte-unchanged
  two-step. Sub-threshold + ineligible leaves keep the existing path verbatim.

## Bit-exactness argument

Each feature is independent: disjoint histogram region, own ascending-leaf_rows / grad-at-`bin<<1`
fold (`fold_one_feature`, identical to `build_leaf_histograms_raw`), own `fix_histogram` (RAW sums +
most_freq_bin) and `compact_histogram` (offset) in identical op order, own scan
(`find_best_split_cpu_native`). Co-locating build+fix+scan for one feature on one thread changes
NEITHER per-feature op order NOR the cross-feature argmax (still serial, feature order, after the
region). Bit-exact by the same proof that makes the parallel build (Spike-005,
`build_histograms_parallel_equals_serial`) and the 9cp parallel scan bit-exact — both green
forced-on. Proven here by the Task-4 `LGBM_UNIFIED_BFS_THRESHOLD=0` forced-on parity gate.
