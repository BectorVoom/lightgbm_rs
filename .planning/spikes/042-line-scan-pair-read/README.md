---
spike: 042
name: line-scan-pair-read
type: standard
validates: "Given the feature-per-lane split scan reads [grad,hess] pairs sequentially, when the read is vectorized as Vector<F,2>, then scan time falls while staying bit-exact"
verdict: INVALIDATED
related: [041, 021, 034, 036]
tags: [performance, gpu, rocm, cpu, vectorization, vector, scan, split, null, bit-exact, dependent-chain]
---

# Spike 042 — vectorize the scan's `[grad,hess]` pair read as `Vector<F,2>`

## What This Validates

Given the shipped feature-per-lane split scan (spike-021, `CubeDim=W`: each lane scans
ONE feature's `num_bins` `[g,h]` pairs sequentially), when each bin's pair is read as one
`Vector<F,2>` load instead of two scalar loads, then the scan should run faster while
staying bit-exact (only the load changes; the arithmetic is identical).

## Research / Method

Builds on spike-041's `Vector<P,N>` recipe. Uses the spike-022b PROXY-kernel method: a
single forward prefix-sum + representative gain (`sg²/(sh+ε)`) argmax — faithful to the
real scan's READ + dependent-accumulate STRUCTURE without the full reverse+forward /
default-bin machinery (which is scalar control flow the load change can't touch anyway).
The only difference between arms is the load: scalar `hist[base+2b]`,`hist[base+2b+1]`
vs vector `hist_v[base_v+b]` then `pair[0]`/`pair[1]`. Bench at narrow (50) AND wide (500)
feature counts × 256 bins. Discipline per CONVENTIONS: overwrite-launch ×50, interleaved
median of 11, judge sign, 2 restarts.

## How to Run

```
cargo run --release --example spike042_vector_scan_pair_ab
cargo run --release --features rocm --example spike042_vector_scan_pair_ab
```
Source: `crates/lgbm-compute/examples/spike042_vector_scan_pair_ab.rs`.

## Results

**VERDICT: INVALIDATED (null).** Vectorizing the scan pair-read is **not a lever** — no
speedup on either backend — though it is **bit-exact on every cell** (parity-safe, as 041
predicted; the negative is purely performance).

| backend | feat=50 | feat=500 | verdict |
|---|---|---|---|
| cubecl-cpu f64 | 1.08× | 1.00× | null |
| cubecl-cpu f32 | 0.96× | 0.88× | null-to-slight-loss |
| cubecl-hip f32 (2 restarts) | 1.01× / 0.93× | 1.04× / 1.07× | **straddles 1.0 = null** |

### Why (the mechanism — this is the durable finding)

The scan is **NOT load-bound**, so vectorizing the load attacks a non-bottleneck:

- The per-lane work is a **dependent prefix-sum chain** (`sg`/`sh` carried across every bin)
  plus a **divide** (`sg²/(sh+ε)`) and a **compare** per bin. That serial dependent chain +
  the divide DOMINATE the lane; the pair load is a small fraction.
- The `Vector<F,2>` extract (`pair[0]`/`pair[1]`) can add a shuffle/extract that slightly
  REGRESSES the streaming f32 case (cpu 0.88× at wide).
- Consistent with spike-034 (the genuine scan is launch/readback-SYNC bound, 3–7% of train)
  and spike-036 (the scan's only real residual is loop-trip-count divergence, not load
  throughput). Subtract (041) won 2.5–3.7× because it is a PURE streaming load-sub-store —
  the opposite of the scan's dependent-chain profile.

### Contrast with 041 (why subtract won and scan didn't)

| | 041 subtract | 042 scan |
|---|---|---|
| op shape | element-wise streaming (load–sub–store) | dependent prefix-sum + divide + argmax |
| what vectorizes | load **and** store **and** compute | only the load |
| bottleneck | memory throughput → vectorized | dependent chain + divide → scalar, unchanged |
| result | 2.5–3.7× (cpu), 1.06–1.29× (hip) | null |

**Takeaway (carry forward):** `Vector<P,N>` only pays where the kernel is
**memory/throughput-bound and the vectorized op covers the bottleneck**. A kernel whose
cost is a dependent arithmetic chain (scan) gets nothing from vectorizing its loads.

### Disposition

DON'T WIRE — null on both backends. Keep the example as rocm-gated evidence. The
bit-exactness confirmation is reusable (Vector pair-read is parity-safe if a future reason
to vectorize the scan emerges, e.g. a within-feature cooperative scan where the load IS the
cost). Mirrors the 030/031/033/036 bounded-don't-chase pattern.
