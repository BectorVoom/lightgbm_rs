---
spike: 004
name: columnar-u8-bins
type: standard
validates: "Given the histogram gather+fold is large-row-dominated, when the bin column is narrowed u32→u16→u8, then the gather+fold speeds up materially (cache density), bit-exactly"
verdict: VALIDATED
related: [002, 003]
tags: [performance, cpu, histogram, cache, bin-width, columnar]
---

# Spike 004: Columnar u8/u16 bin storage

## What This Validates

**Given** the histogram gather+fold dominates large-row CPU train (build ≈ 90% of
train at 200k rows after spike-003/r4o), **when** the per-feature bin column is stored
in the narrowest type (u32 → u16 → u8), **then** the random `bins[leaf_rows[i]]` gather
speeds up materially from cache density — bit-exactly (the bin VALUE is unchanged, just
stored narrower and widened at fold time).

## Method

Isolated micro-benchmark (`crates/lgbm/examples/bin_width_microbench.rs`) — measures ONLY
the fused gather+fold loop (the exact `build_leaf_histograms_raw` inner loop) over 32
independent feature columns × 200k rows × 64 bins, with a scattered (Fisher–Yates) row
order to mimic real `leaf_rows`. No dataset plumbing — this isolates the bin-width cache
effect before committing to a storage refactor.

```
cargo run --release --example bin_width_microbench
```

## Results

**VERDICT: VALIDATED.** Narrowing the bin column nearly halves the gather+fold.

| bin type | per-feature column | median (32 feat × 200k) | vs u32 | throughput |
|----------|--------------------|--------------------------|--------|------------|
| u32 | 781 KB | 15.3 ms | — | 434 Mrows/s |
| u16 | 390 KB | 9.0 ms | **−41%** | 731 Mrows/s |
| u8  | 195 KB | 6.5 ms | **−58%** | 991 Mrows/s |

The u32 column (781 KB/feature) overflows L2 → the random gather misses to L3; the u8
column (195 KB) fits L2 → hits. Faithful to C++ `DenseBin<uint8_t>`/`<uint16_t>`/
`<uint32_t>`, which picks the narrowest bin type per feature for exactly this reason.

### Scope of the win
- **Large rows:** big — the gather+fold is build's dominant cost and build is ~90% of
  large train (2.74s). Real-train delta is < the isolated −58% (build also has the
  once-per-leaf ord_g/ord_h gather + scratch copy, which narrowing doesn't touch), but
  substantial. Default `max_bin=255` ⇒ u8 covers the common case.
- **Small rows:** neutral — a 2k-row column is 8 KB (u32), already L1-resident; narrowing
  can't help and its one-time cost is negligible. No small regression expected.

## Signal for the Build

- **Implement columnar narrow bins** — store each `FeatureColumn`'s bins in the narrowest
  unsigned type for its `num_bin` (u8 ≤256, u16 ≤65536, else u32). The hot
  `build_leaf_histograms_raw` fold reads the narrow type directly; cold readers
  (partition, bagging, GPU upload) read via a widening accessor. Bit-exact (value
  unchanged), single-thread, keeps the 1-core-vs-C++ comparison basis.
- Design choice for the implementation: an enum `BinColumn { U8/U16/U32 }` with a
  `bin(row)->u32` accessor (no memory doubling) vs an additive `bins_u8` alongside the
  u32 (simpler, doubles bin memory). Prefer the enum — faithful to C++, no doubling.
- Measure the REAL train delta with bench_crossover + phase_prof at small (no regression)
  AND large (the win), bit-exact gate green, before shipping (the standing lesson).

Closes more of [[perf-gap-vs-cpp-40-80x]] R3 at large rows. Stacks on the once-gather
([[perf-gap-vs-cpp-40-80x]] spike 003) + fused-branchless (quick r4o) wins.
