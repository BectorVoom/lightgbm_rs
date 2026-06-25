---
spike: 026
name: cubecl-cpu-partition-scan-scatter
type: standard
validates: "Given the serial single-threaded stable-compaction gather is the host DataPartition::split cost, when it is reformulated as a bit-exact cubecl-cpu scan+scatter kernel (per-chunk count → host prefix-sum → disjoint scatter) on the 16 real CPU cores, then the op runs faster than data_partition_cpu_native — byte-identically"
verdict: PARTIAL
related: [005, 007]
tags: [performance, cpu, cubecl-cpu, partition, scan-scatter, simd, bit-exact, memory-bandwidth, isolated-ab]
---

# Spike 026: cubecl-cpu scan+scatter DataPartition::split

## What This Validates

The production CpuBackend partition (`data_partition_cpu_native`) does a **serial,
single-threaded** stable two-pass gather. The per-row routing is already a parallel
`#[cube]` kernel (b141a82); the expensive part — the order-preserving compaction — stays
serial and is ~23% of GPU-wide train / ~29% of tall-narrow train (spike-023, ia0). Can a
**bit-exact cubecl-cpu scan+scatter kernel** do that compaction faster on the 16 real CPU
cores + SIMD? (CPU is real hardware; only the GPU is the spoofed APU — so these wall-clock
ratios are legitimate.)

**Why cubecl-cpu, not rayon:** a rayon version was already built, proven bit-exact, and
**reverted as an end-to-end NULL** (quick-260622-ia0) — but the stated cause was *contention
with the already-rayon-parallel histogram BUILD on the CPU train path*. On the GPU train
path the build runs on-device, so the host pool is idle: the contention is absent. This
spike tests the user's preferred **cubecl-cpu** variant (unified `#[cube]`; SIMD; "same
value" = bit-exact) in isolation (no competing build).

## How to Run

```
cargo run -p lgbm-compute --example spike026_partition_scan_scatter_ab --release
# knobs: LGBM_SPIKE_CUBEDIM (default 16), LGBM_SPIKE_CHUNK (default 512),
#        LGBM_SPIKE_PROF=1 (sub-phase breakdown), LGBM_SPIKE_RUN=N (restart label)
```

## Algorithm (bit-exact parallel stable partition)

Mirrors ia0 / C++ `ParallelPartitionRunner` `schedule(static, 512)`:
1. **COUNT kernel** — one logical task per CHUNK; each counts its LEFT rows. (parallel)
2. **host prefix-sum** of per-chunk left-counts → per-chunk disjoint left/right write bases. (tiny)
3. **SCATTER kernel** — each chunk walks its rows ASCENDING, scatters into `[all-left | all-right]`
   at its disjoint base. (parallel across chunks; sequential within → stable order preserved)

Byte-identical to the serial two-pass gather by construction. **Parity OK on every cell of
every run** (cubecl output == `data_partition_cpu_native`, `reordered[]` + `split_point`).

## Investigation Trail

1. **Naive (`CubeDim(1)`, chunk=512):** bit-exact but **0.5–0.6× (≈2× SLOWER)** than serial at
   1M+, far worse on skewed/small. The mc5 "cubecl-cpu lost to native" lesson reproducing.
2. **Chunk-size sweep (512…262144):** **flat** (0.55–0.61×). So it is NOT compute-granularity
   bound — the cost is fixed per-call (dispatch/marshal/readback), not the parallel work.
3. **Sub-phase decomposition (`LGBM_SPIKE_PROF`, 1M balanced):** marshal ~0.9ms, count ~1.5ms,
   host-scan ~0ms, **scatter ~4ms**, readback(4MB) ~0.6ms. The 4MB readback is cheap (as
   predicted); the **scatter kernel is the wall** and was running ~single-threaded.
4. **Launch-geometry sweep (`LGBM_SPIKE_CUBEDIM`):** the key finding — **cubecl-cpu threads on
   the UNIT (`CubeDim`) axis, not the cube axis.** `CubeDim(1)` = serial. `CubeDim=16` cut the
   COUNT pass 2.5×→1.0ms. But the **SCATTER stayed ~3.5ms for ALL geometries** — branchy
   data-dependent scattered writes don't parallelize/vectorize.
5. **Branchless scatter (select-based, bit-exact):** pushed **100k balanced 1.53×→2.26×** (cache-
   resident) but **did NOT move 1M** (0.9×) or skewed (0.2×). ⇒ at scale the scatter wall is
   **memory bandwidth, not branch misprediction.**

## Results (optimised: CubeDim=16 + branchless scatter; 2 process restarts, parity OK every cell)

| rows | skew | serial(ms) | cubecl(ms) | ratio (serial/cubecl) |
|------|------|-----------|-----------|------------------------|
| 1,000   | 0.0 | 0.001 | 0.08 | **0.01×** (launch overhead) |
| 16,384  | 0.0 | 0.07–0.11 | 0.10–0.12 | 0.66–0.88× |
| **100,000** | **0.0** | 0.66 | 0.29–0.31 | **2.15–2.26× (the one win — cache-resident)** |
| 500,000 | 0.0 | 3.3 | 3.3–3.4 | ~1.00× (parity) |
| 1,000,000 | 0.0 | 6.6 | 7.0–7.4 | 0.89–0.95× |
| 4,000,000 | 0.0 | 26 | 30 | 0.85–0.88× |
| **any** | **0.9 (skewed)** | — | — | **0.17–0.61× (loses 2–5×)** |

## Verdict: PARTIAL — bit-exact, but ROI-NEGATIVE for the production goal. Do NOT wire.

**The mechanism is bit-exact and the kernel is correct**, but cubecl-cpu scan+scatter **only
beats serial native in a narrow cache-resident balanced window (~100k rows, ~2×)**. At the
scales and skews that dominate real training it is parity-to-3–5×-slower:

- **At scale (≥500k) it is memory-bandwidth-bound on SHARED DDR5.** The 16 cores share one
  DRAM controller (this is an APU; see `gpu-is-spoofed-8cu-apu`), so the compaction's
  read+scatter traffic saturates bandwidth and parallelism buys nothing. **This is the same
  reason ia0's rayon was null at scale — and it reframes ia0: the wall was never *build
  contention*, it is *shared DRAM bandwidth*. NO host-parallelism approach (rayon OR
  cubecl-cpu) reclaims the partition residual at scale.**
- **On skewed data serial native wins 2–5×** (branch-predictable, cache-friendly two-pass push)
  — and real trees produce *increasingly skewed* leaf bins as they deepen, so skew is the
  COMMON case for the small/deep leaves where the partition residual actually accrues.
- **Small leaves lose to per-launch overhead** (~0.07ms floor regardless of n).

**Signal for the build:** the partition residual is **not reclaimable by parallelizing
partition**. To shrink it, reduce its MEMORY TRAFFIC, not add cores — e.g. fuse the per-leaf
`leaf_feature_bins` gather (`data_partition.rs:141-145`) into the routing pass to avoid
materializing it, use narrower bin types, or eliminate the host writeback round-trip. That is
a separate spike.

**Reusable cubecl-cpu facts (this codebase):** parallelism is on the `CubeDim`/UNIT axis
(`CubeDim(1)` runs serial — use ≥16); `ABSOLUTE_POS`/`.len()`/Array-indexing are all `usize`
(cast u32 scalars in-kernel via `usize::cast_from`); `u32::from_bytes` returns a borrowed
`&[u32]` (bind the bytes, `.to_vec()` to own); scalars pass raw, `ArrayArg::from_raw_parts(h, n)`.
