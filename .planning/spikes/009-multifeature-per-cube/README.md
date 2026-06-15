---
spike: 009
name: multifeature-per-cube
type: standard
validates: "Given many small-bin features each getting their own cube, when several features are packed into one cube's LDS at MATCHED occupancy, then it is decided whether overhead-amortization beats one-cube-per-feature"
verdict: INVALIDATED
related: [007, 005]
tags: [performance, gpu, rocm, histogram, occupancy, packing, negative-result]
---

# Spike 009: multi-feature-per-cube packing (INVALIDATED — null at matched occupancy)

## What This Validates

The "Feature → work" gap from `cubecl_kernel_gaps.md`: *"ours wastes a CU on small-bin
features."* LightGBM packs many feature-columns into one block's shared histogram
(`block_dim_x` over columns). Our build runs one cube per feature (× P row-partitions, spike-007).
Packing G features into one cube amortizes the per-cube fixed overhead (zero + 2 syncs + merge)
across G features — **but divides the cube count by G**, fighting the occupancy the row-partition
lever just bought. Parity-SAFE (identical f32 exact accumulation).

## Method — matched-occupancy isolation

`crates/lgbm-compute/examples/gpu_multifeature.rs` (gfx1100). 1M rows × **128 features × 32 bins**
(packing's best case: 8 small features fit one 512-cell LDS). The key control: tune each
variant's row-partition `P` so BOTH launch ~the same total cubes (~768 ≈ 8 wkgrps/CU), so the
comparison isolates **overhead-amortization** from **occupancy**:
- per-feature: 128 features × P=6 = 768 cubes
- packed (G=8): 16 groups × P=48 = 768 cubes

## Results

**VERDICT: INVALIDATED — null at matched occupancy.** packed/per-feature = **0.91, 0.93, 1.00×**
across 3 rounds (correctness max_rel = 7.2e-6, fine).

| round | per-feature | packed | packed/pf |
|------:|------------:|-------:|----------:|
| 1 | 4689ms | 5162ms | 0.91× |
| 2 | 4797ms | 5178ms | 0.93× |
| 3 | 5116ms | 5138ms | 1.00× |

## Why (the finding)

The per-cube fixed overhead (zero/sync/merge of `feat_len` cells) was **never the bottleneck** —
the build is **atomic-contention/scattered-read bound** (spike-006), and **row-partitioning
already solved occupancy** (spike-007). At matched occupancy there is nothing left for packing to
amortize, and the packed kernel's extra machinery (sequential per-feature inner loops, G× the LDS
per cube → fewer resident workgroups per CU) slightly *regresses*. LightGBM needs column-packing
because its decomposition reaches occupancy a different way (2D `block_dim_x × grid_dim_y` tile);
our P lever reaches it directly, so the column-packing axis is redundant for us.

## Signal for the Build

- **DO NOT ship multi-feature packing.** Null-to-slight-regression at matched occupancy; pure added
  complexity. Keep **one-cube-per-feature × P** (the shipped spike-007 design) — it is the right
  decomposition for our architecture.
- Closes the "wastes a CU on small-bin features" gap with evidence: the waste is not a perf cost
  once row-partitioning supplies the cubes.

Reusable: `gpu_multifeature.rs` (the matched-occupancy packed-vs-per-feature harness).
