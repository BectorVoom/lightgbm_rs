---
spike: 028
name: doublebuffer-partition
type: standard
validates: "Given the wired spike-027 V1 fused partition ends with copy_from_slice(out -> indices[begin..]), when indices is double-buffered (scatter directly into a persistent alt[begin..], ping-pong, no copy-back), then the op runs faster — and is the win worth the cross-leaf bookkeeping?"
verdict: INVALIDATED
related: [027, 026]
tags: [performance, cpu, partition, double-buffer, copy-back, null, isolated-ab]
---

# Spike 028: Double-buffer the partition indices (drop the V1 copy-back)

## What This Validates

The wired spike-027 V1 fused partition scatters row ids into a scratch `out` Vec then does
`self.indices[begin..begin+count].copy_from_slice(&out)` — an in-place scatter would clobber
unread rows, so a copy-back (or a second buffer) is structurally required. This spike asks:
keep a PERSISTENT second `indices` buffer and scatter directly into `alt[begin..]`
(ping-pong), skipping the copy-back + the `out` alloc — does it measurably beat V1?

**Honest framing baked in:** C++ LightGBM's `DataPartition::Split` ALSO copies back (via
`temp_indices_` inside `ParallelPartitionRunner`), so the copy-back is not obviously
removable. The spike measures the copy-back's share of V1 and the op-level ceiling of
removing it.

## How to Run

```
cargo run -p lgbm-compute --example spike028_doublebuffer_partition_ab --release
```

## Results (median, 2 process restarts; ratio = V1_copyback / V1_doublebuffer, parity OK every cell)

| width | rows | skew | V1 (ms) | copy-back % of V1 | ratio (dbuf vs V1) |
|-------|------|------|---------|-------------------|--------------------|
| U8 | 1M | 0.0 | 4.5 | **2.3%** | 1.01× |
| U8 | 1M | 0.9 | 1.95 | 5.4% | **0.93× (slower)** |
| U8 | 4M | 0.0 | 20.6 | 3.4% | 1.04× |
| U8 | 4M | 0.9 | 10.2 | 6.8% | 1.01× |
| U32 | 1M | 0.9 | 1.74 | 6.5% | **0.86× (slower)** |
| U32 | 4M | 0.0 | 32 | 2.2% | 1.02× |

## Verdict: INVALIDATED (NULL) — do NOT wire.

- **The copy-back is only 2.2–6.8% of V1.** The fused op is dominated by the random bin
  GATHER + the SCATTER; the copy-back (a single sequential `count`-u32 memcpy) is a negligible
  residual. spike-027 already reclaimed the big traffic (the ~4-buffer materialization); the
  copy-back was always the small leftover.
- **Removing it is within noise (0.86–1.04×), sometimes SLOWER.** Ceiling is ~1.02–1.07× (the
  copy-back share); the measured delta is noise. At skewed shapes the double-buffer is
  slightly slower (scattering into an offset slice of a full-N `alt` vs a fresh `count`-sized
  `out` — same write traffic, no copy-back saved enough to show).
- **The cost side is real and large.** A persistent second size-N `indices` buffer DOUBLES the
  partition memory (8MB→16MB at 1M rows), and correct ping-pong needs **cross-leaf bookkeeping**
  — tracking which buffer is canonical for each leaf region as the tree grows *leaf-wise* — which
  also interacts with the histogram-subtraction trick (it reads `indices` in partition order).
  C++ copies back for exactly this reason.

⇒ negligible op win, real memory + complexity + parity-risk cost. **The spike-027 fused-gather
win stands as the partition lever; the copy-back is not worth chasing.** This closes the
spike-027 "double-buffer indices" follow-on.

## Signal for the build

Partition optimization is **DONE** for the CPU path: spike-027 (fused-gather, shipped) captured
the reclaimable memory traffic; spike-028 confirms the remaining copy-back residual is too small
to matter. Any further partition gain would need a different representation entirely (e.g. the
GPU-side narrow-upload fuse, a separate spike), not micro-optimizing the host copy.
