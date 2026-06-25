---
spike: 027
name: fused-gather-partition
type: standard
validates: "Given the host DataPartition::split is memory-bandwidth-bound (spike-026) and materializes 3 per-leaf u32 Vecs (leaf_rows + u32-widened leaf_feature_bins + reordered) every split, when the bin GATHER is fused into the routing — read the indices slice in place, route directly off feature_bins.bin(row), scatter ROW ids into one output buffer via a 1/4-width u8 route scratch — then the op runs faster by cutting traffic, byte-identically"
verdict: VALIDATED
related: [026, 004]
tags: [performance, cpu, partition, memory-traffic, gather-fusion, bit-exact, narrow-bins, isolated-ab, shippable]
---

# Spike 027: Fuse the leaf-bin gather into partition routing

## What This Validates

Spike-026 proved the host `DataPartition::split` compaction is **memory-bandwidth-bound** at
scale (shared DDR5) — parallelism can't reclaim it; the lever is to cut TRAFFIC. The current
numeric branch (`data_partition.rs:141-173`) materializes **three** count-sized `Vec`s per
split:

1. `leaf_rows = indices[begin..].to_vec()` — count × u32
2. `leaf_feature_bins = leaf_rows.map(|r| feature_bins.bin(r))` — count × u32, **WIDENED to
   u32 even when the bin column is U8** (the common narrow-bin case, spike-004)
3. `reordered` (local indices) from the op → a writeback then remaps local→row via leaf_rows

≈ four count-sized u32 buffers written + reread per split. This spike **fuses the gather into
routing**: read the `indices` slice in place, route directly off `feature_bins.bin(row)`, and
scatter the **row ids** straight into one output buffer using a **u8 route scratch** (¼ the
width of the u32 leaf bins) — no `leaf_rows`, no u32-widened `leaf_feature_bins`, no remap.

CPU is REAL hardware (only the GPU is the spoofed APU) ⇒ legitimate wall-clock.

## How to Run

```
cargo run -p lgbm-compute --example spike027_fused_gather_partition_ab --release
```

## Variants (all byte-identical output; parity OK every cell, 2 restarts)

- **V0** — faithful replica of the current path (leaf_rows + u32 leaf_feature_bins +
  `data_partition_cpu_native` + local→row remap writeback).
- **V1 — FUSED, u8 route scratch** — 1 random gather → u8 route + count, then scatter rows
  direct. The candidate.
- **V2 — FUSED, 2-gather** — no scratch, recompute the route in pass 2 (two random gathers).
  Tests whether eliding the scratch beats a second gather.

## Results (median, 2 process restarts; ratio = V0 / Vk, >1 ⇒ fused faster)

| width | rows | skew | V0 (ms) | V1_u8 | **V1/V0** | V2/V0 |
|-------|------|------|--------|-------|-----------|-------|
| **U8** | 16k | 0.0 | 0.13 | 0.05 | **2.68×** | 2.15× |
| U8 | 100k | 0.0 | 0.91 | 0.39 | **2.36×** | 1.86× |
| U8 | 500k | 0.0 | 4.9 | 2.1 | **2.33×** | 1.89× |
| U8 | 1M | 0.0 | 10.5 | 4.6 | **2.30×** | 1.87× |
| U8 | 4M | 0.0 | 45 | 20 | **2.27×** | 1.79× |
| U8 | 1M | 0.9 | 5.4 | 2.2 | **2.43×** | 2.27× |
| U32 | 1M | 0.0 | 8.5 | 4.9 | **1.74×** | 1.39× |
| U32 | 4M | 0.0 | 51 | 31 | **1.64×** | 1.00× |
| U32 | 4M | 0.9 | 31 | 22 | **1.41×** | **0.79×** |

## Verdict: VALIDATED — fused-u8-route partition is bit-exact and 1.3–2.7× faster. SHIPPABLE (CPU path).

- **V1 wins at EVERY size, skew, and bin width** — unlike spike-026's parallelization (which
  won only cache-resident-balanced and lost skewed/large). Reducing TRAFFIC works where adding
  CORES failed: this is the direct confirmation of 026's signal.
- **Biggest at U8 (~2.3×)** — the production narrow-bin case (spike-004). The u32-widening of
  `leaf_feature_bins` was pure waste on a U8 column; the u8 route scratch removes it.
- **V1 > V2 everywhere; V2 REGRESSES at U32/4M (0.79×)** — a second random u32 gather costs more
  than the sequential u8 route scratch it elides. ⇒ the **u8 route scratch is the right design**;
  do not drop it to save the allocation.
- **Mechanism (why ~2×):** the op is memory-bound (026), so the speedup tracks the traffic cut.
  V0 ≈ 4 count-u32 buffers (leaf_rows + u32 leaf_feature_bins + reordered + writeback); V1 ≈ 1
  random gather + ¼-width u8 route + 1 u32 out + copyback ≈ half the bytes.

## Signal for the build

**Wire V1 into `DataPartition::split`'s numeric branch (CpuBackend path).** This directly
shrinks the partition residual that is BOTH ~29% of tall-narrow CPU train AND the #1 remaining
CPU-vs-C++ gap ([[perf-gap-vs-cpp-40-80x]] — C++ also avoids the materialization). Bit-exact by
construction (identical stable [left|right] order); gate with `split_*` unit tests +
`raw_bin_train_matches_cpp_golden` (partition order feeds the histogram-subtraction trick, so
any drift surfaces there). Keep the u8 route scratch.

**Open follow-ons (not built):**
- **Double-buffer `indices`** to drop the final `copy_from_slice` back (ping-pong the partition
  buffer) — removes one more count-u32 of traffic; needs a second indices buffer in DataPartition.
- **GPU path:** V0's host `leaf_feature_bins` materialization + u32-widened upload is ALSO paid
  before `RocmBackend::data_partition`; fusing the gather to upload a narrow (u8) buffer — or
  routing straight off the resident bins — is a separate GPU-side spike.
- The route scratch could be a bitset (1 bit/row) instead of u8 — likely sub-noise (u8 already
  ¼ width and cache-friendly), measure before adding the bit-twiddling.
