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

---

# Task 3 — A/B Verdict: CONDITIONAL WIN (threshold = 100 features)

Unlike the two prior sibling NULLs (8v4 per-call 2-lane fork/join ~13× slower; 9cp
cross-feature scan par_iter contending with the parallel build, +11% train-wall), the unified
region IS a sign-stable warm train-wall win — but ONLY above ~100 features, so it ships gated.

Measured on cubecl-cpu (f64 anchor), warm (cold iteration discarded), 3-run medians,
20 000 rows × 128 bins. Decisive metric = warm end-to-end train-wall (`bench_train`);
phase counters (`bench_split_scan`, 255 bins) localize the mechanism.

| config | mode A (two-step, default) | mode B (unified, THRESHOLD=0) | delta (train-wall) |
|--------|---------------------------|-------------------------------|--------------------|
| narrow (10 feat)  | 169 ms (162.6/169.3/172.8) | 246 ms (244.5/246.3/246.7) | **+45% WORSE** (sign-stable) |
| wide (120 feat)   | 810 ms (800.6/810.5/821.6) | 755 ms (754.7/755.1/762.7) | **−6.8% WIN** (sign-stable, no overlap) |

Phase counters (`bench_split_scan`, 255 bins, BUILD+SCAN fused into SCAN_NS in mode B):
- narrow: A build+scan 61.4+34.5 = 95.9 ms → B fused 124.9 ms (+30%); warm_wall 166.7→217.9 ms.
- wide:   A build+scan 431.2+348.5 = 779.7 ms → B fused 569.7 ms (**−27%**); warm_wall 1049.9→910.7 ms (−13%).

Mechanism confirmed: at wide, fusing build+fix+compact+scan into ONE rayon fork/join (vs the
two-step's TWO) keeps each feature's histogram cache-hot in its building thread, removing the
9cp two-region contention → −27% on the hot region, −6.8% warm train-wall. At narrow, the single
fork/join overhead at 10 features dominates (the 8v4/9cp narrow-is-catastrophic lesson).

Crossover sweep (train-wall, median of 3, 128 bins):

| feat | A med | B med | delta |
|------|-------|-------|-------|
| 20 | 221.3 | 297.7 | +34.6% |
| 30 | 289.8 | 339.7 | +17.2% |
| 40 | 351.9 | 372.8 |  +5.9% |
| 50 | 394.1 | 429.9 |  +9.1% |
| 60 | 459.6 | 476.1 |  +3.6% |
| 80 | 573.8 | 577.5 |  +0.6% (within spread) |
| 90 | 624.8/629.8/639.6 | 611.7/641.3/642.2 | overlapping / sign-flips |
| 100 | 705.7/709.3/721.3 | 674.7/681.5/691.6 | **−5%, no overlap** |
| 120 | 810.8/814.5/823.2 | 760.9/773.8/773.9 | **−6%, no overlap** |

B is a regression or within-spread up to ~90 feat; it becomes sign-stable-below-A only at
≥100 feat. **Tuned default threshold = 100** (keyed on `feats.len()`, mirroring
`par_scan_threshold`): narrow/medium (≤90 feat) keep the byte-unchanged serial two-step (zero
regression); genuinely wide leaves (≥100 feat) take the unified region for the −6% gain. Both
adoption gates met — wide sign-stable gain AND no narrow regression at the tuned threshold.

Default-path verification (threshold=100 baked): narrow 167–179 ms (== mode A, serial, no
regression); wide 748–776 ms warm (== mode B, the unified win vs mode-A 810 ms baseline).
