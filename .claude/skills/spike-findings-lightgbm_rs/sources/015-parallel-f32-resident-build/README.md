---
spike: 015
name: scan-roundtrip-decomposition
type: standard
validates: "Given a wide-shape GPU train (≥250k×500) where scan_resident_leaf is ~96% of train wall (~34ms/leaf-scan), when the per-leaf scan round-trip is decomposed into marshal/upload/launch+readback, then the dominant slice is identified and attacked (hoist constant per-feature arrays per-tree + cut the readback round-trip; prototype batching leaves per launch)"
verdict: PENDING
related: [007, 014]
tags: [performance, gpu, rocm, scan, round-trip, launch-bound, wide-shape]
---

# Spike 015: per-leaf scan round-trip decomposition (the wide-shape GPU bottleneck)

## What This Validates

Given a wide-shape GPU train (≥250k × 500) where `scan_resident_leaf` is ~96% of train
wall (~34ms per leaf-scan), when the per-leaf scan round-trip is decomposed into
marshal / upload / launch+readback, then the dominant slice is identified and the lever
chosen from evidence (not inference).

## Premise correction (the investigation that got us here)

This spike began life as "replace the on-device **sequential-f64 build** with a
parallel-f32-atomic build." Two findings from reading the live code **invalidated that
premise** before any kernel was written:

1. **The sequential-f64 FUSED path is OFF by default.** `resident_pool.rs`:
   `FUSED_MAX_NUM_DATA = -1` ⇒ `fused_directly_built_eligible` is false for every real
   workload (only `LGBM_FUSED_FORCE=1` engages it). The profiled wide run never used it.
2. **The non-fused wide path already builds f32-atomic.** `build_resident_leaf`
   (lib.rs:873) docstring: *"build → f32→f64 widen → fix → compact"* — the parallel
   f32-atomic resident build (spike-007 lineage) is already the wide-shape build path.
   So "switch the build to f32" was already done; it is not the cost.

Where the ~8.4s/train (250k×500, 3 reps) actually lives: **`scan_resident_leaf`** →
`find_best_splits_batched_fused_f64_from_handle_on` (split.rs:1105) →
`find_best_splits_fused_inner`. That launcher, **on every leaf scan (~248×/train)**:
- rebuilds **7 per-feature arrays** (`slot_off`, `num_bin`, `offset`, `default_bin`,
  `skip_default_bin`, `rev_count`, `fwd_count`) — all derived from feature metadata that
  is **constant across every leaf and every tree**;
- does **8 `create_from_slice`** device uploads;
- launches `find_best_splits_fused_kernel` at `CubeCount(n,1,1) × CubeDim(1)` (one
  single-threaded cube per feature);
- `read_one_unchecked` reads back `n*12` SplitInfo cells (a sync round-trip).

~34ms/launch for 500×128 bin-ops is **not compute** — it smells like per-leaf
marshal+upload+launch+sync. Structurally the same class of waste p9v fixed for the bin
upload (per-leaf redoing of per-train-constant work). See
[[gpu-bottleneck-moved-to-seq-f64-scan]] (note records the corrected story).

## Plan (user: "Pivot + try batching leaves too")

1. **Decompose** — `LGBM_SCAN_PROF=1` env-gated timers split the per-leaf scan into
   marshal / upload / launch+readback. Run the wide sweep, attribute the 34ms.
2. **Attack the dominant slice** — if marshal+upload dominate: hoist the constant
   per-feature arrays to once-per-tree (upload once, reuse Handles). If launch+sync
   dominates: prototype **batching multiple leaves per launch** to amortize.
3. **Measure end-to-end** on `bench_gpu_vs_cpu` wide; characterize parity (should be
   parity-neutral — no precision change).

## How to Run

```
# decomposition (after instrumentation lands):
LGBM_SCAN_PROF=1 LGBM_BENCH_SWEEP=wide cargo run --release --features rocm --example bench_gpu_vs_cpu
```

## Investigation Trail

- **Premise reversal #1** — fused-f64 gated off (`FUSED_MAX_NUM_DATA=-1`); wide build
  already f32-atomic. The "switch build to f32" lever was already shipped.
- **Premise reversal #2** — instrumented the scan round-trip (`LGBM_SCAN_PROF=1`):
  marshal 0.0% + upload 0.1% + **launch+readback 99.8%**. So neither array-hoisting nor
  marshalling is the cost. But cubecl is ASYNC: the build is launched un-synced in
  `build_resident_leaf_into`, and the scan's `read_one_unchecked` is the first sync →
  the build's device-compute materializes inside the scan's readback.
- **Disambiguation** — forced a pre-scan build drain (`LGBM_SCAN_DRAIN=1`, reads the
  resident histogram handle before the scan launch). Re-attribution (wide sweep):

  | shape | build_drain | scan launch+readback | marshal+upload |
  |-------|-------------|----------------------|----------------|
  | 250k×500 | 85.8% | 14.0% | 0.1% |
  | 500k×500 | 88.3% | 11.5% | 0.1% |
  | 1M×500   | **91.9%** | 8.0% | 0.1% |

  The **f32-atomic histogram BUILD** is the bottleneck, GROWING with rows (build scales
  with rows; the per-bin scan is row-independent). ~4.9s/train build at 250k×500 ≈ 5G
  atomic adds ÷ ~820 Mr/s (spike-006/007's measured atomic ceiling).
- **Row-partition (007's only build win) is structurally inactive at wide shapes**
  (`row_partition_count`, histogram.rs:646): `ROWPART_MIN_LEAF=256_000` (the 250k root
  is below it) AND `768/500 features = 1` → P=1 even at 1M rows. No env lever forces
  P>1 at 500 features (the `768/nf` divisor kills it; `LGBM_ROWPART_MIN` only moves the
  leaf-size gate).

## Results

**VERDICT: PARTIAL — premise invalidated, bottleneck definitively located.**

- The wide-shape (×500) GPU train bottleneck is the **f32-atomic resident histogram
  BUILD device-compute** — **86%→92% of the scan-attributed wall, growing with rows**.
  It is atomic-contention-bound (~820 Mr/s), already LDS-privatized + parallel.
- **Dead levers (measured, do NOT pursue):** batching leaves / cutting the scan
  round-trip (≤14% and SHRINKING with rows); hoisting the constant per-feature scan
  arrays (~0%); switching the build to f32 (already done); row-partition at wide (P=1).
- **Routing reality:** at 250k×500 the multi-threaded CPU anchor trains in 1.80s vs the
  GPU's 7.14s (~4× faster). The GPU only wins narrow-tall shapes (spike-001 crossover
  ~700k rows × 50 feat). **Wide shapes should route to CPU**, consistent with the
  MANIFEST Requirement "never route small/medium to the GPU" — extended: wide-many-
  feature shapes too.
- **Only credible remaining build lever** (untried; NOT in the closed 006/007/008/009
  set): **finer LDS sub-histogram privatization** — multiple sub-histograms per cube
  (e.g. per-warp) merged at the end, to cut intra-cube atomic contention below the
  current one-LDS-per-cube design. LightGBM's OpenCL `histogram*.cl` uses this. LDS
  budget allows ~8-way (128 bins × 2 × 4B = 1KB/sub-hist; 64KB LDS). Candidate for a
  follow-on spike — but weigh against the routing reality (CPU already wins wide).

## Observability (kept in-tree)

`LGBM_SCAN_PROF=1` → per-leaf scan round-trip breakdown (marshal/upload/launch+readback)
via `lgbm_compute::fusion_prof::dump_scan`. `LGBM_SCAN_DRAIN=1` → forces a pre-scan build
drain to separate build-compute from scan. Both inert/behavior-neutral when unset (the
CPU f64 anchor never enters this fused-scan path). Reusable for any future GPU build work.
