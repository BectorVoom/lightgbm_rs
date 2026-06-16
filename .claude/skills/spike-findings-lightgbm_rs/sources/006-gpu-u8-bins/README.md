---
spike: 006
name: gpu-u8-bins
type: standard
validates: "Given the device build kernel reads scattered bins, when the device bin column is narrowed u32→u8, then the build kernel speeds up (GPU analog of the CPU u8 win)"
verdict: INVALIDATED
related: [004, 005]
tags: [performance, gpu, rocm, histogram, bin-width, atomics, negative-result]
---

# Spike 006: GPU u8 device bins (INVALIDATED)

## What This Validates (hypothesis)

The shipped CPU u8-bins win (spike 004: −58% gather+fold via L2 density) might transfer
to the GPU: the device build kernel reads `resident_bins[f*num_data + leaf_rows[k]]`
(scattered) per (row, feature); narrowing u32→u8 = 4× less global-memory traffic →
faster compute-bound build (80% of GPU train).

## Method

Isolated GPU micro-bench (`crates/lgbm-compute/examples/gpu_bin_width.rs`, `--features
rocm`): the EXACT production naive resident-build access pattern (one unit per
(feature,row), gather from a feature-major resident column, f32-atomic accumulate),
with `resident_bins` as `Array<u32>` vs `Array<u8>`. 200k rows × 32 feat × 64 bins,
scattered leaf_rows, 30 launches × 3 interleaved rounds on the gfx1100. No resident-cache
plumbing — isolates the device-bin-read width effect before committing.

## Results

**VERDICT: INVALIDATED — u8 gives ~0% on GPU** (within noise across 3 rounds):

| | u32 (24 MB column) | u8 (6 MB column) | Δ |
|--|--------------------|------------------|---|
| round1 | 827 ms | 822 ms | −0.6% |
| round2 | 812 ms | 826 ms | +1.7% |
| round3 | 820 ms | 828 ms | +0.9% |

≈234 M reads/s — **slow for a GPU** (gfx1100 does ~TB/s), so the kernel is NOT
bin-read-bandwidth bound. It is **atomic-contention / scattered-read-latency bound**:
- The f32-atomic `out[cell].fetch_add` serializes on rows sharing a bin (64 bins × 32
  feat = heavy collision) — this is the dominant cost.
- The scattered `resident_bins[...]` read is latency-bound: reading 1 byte vs 4 from a
  scattered (uncoalesced) location is the SAME memory transaction. Narrowing the element
  doesn't reduce the number of (cache-line-sized) transactions when access is random.

So the CPU win's mechanism (more of the column resident in L2) does NOT apply on GPU
where the access is latency/atomic-bound, not working-set-bandwidth-bound.

### Secondary (useful) outcome
**`Array<u8>` compiles AND runs correctly on the HIP backend** (cubecl 0.10,
`UIntKind::U8`) — feasibility confirmed for any future u8 GPU need. It's just not a
lever for the build kernel.

## Signal for the Build

- **DO NOT ship u8 device bins.** ~0% benefit; pure plumbing cost. The cheap probe
  (vs a multi-kernel resident-cache rewrite) saved the effort — same discipline that
  killed the CPU alloc lever (quick p0n).
- The real GPU build bottleneck is **atomic contention + uncoalesced scattered reads**,
  which the LDS-privatized kernels (260609-fw1, per-cube LDS sub-histograms) already
  target for ≤256-bin features. Further GPU gains need the DEEPER work — occupancy /
  atomic-contention / coalescing analysis via rocprof — a larger, uncertain effort.
- ROI caveat: with the CPU now multi-threaded (spike 005), the GPU loses to it at every
  tested size (200k: GPU 3.24s vs CPU 1.1s), so GPU optimization is ROCm-parity
  maintenance, not a speed win. Weigh the deeper GPU work against that.

Reusable: `gpu_bin_width.rs` (a device-kernel A/B harness + the `Array<u8>`-on-HIP proof).
Contrast: the CPU u8 win shipped (spike 004 / quick ruz, large −49%); GPU is a different
bottleneck regime.
