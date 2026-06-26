---
spike: 043
name: line-build-gradhess-input
type: standard
validates: "Given the u64 resident build reads ord_g[k]/ord_h[k] in lockstep (coalesced), when grad+hess interleave as Array<Vector<f32,2>> (one load/row vs two), then the grad/hess load falls — bounded by the un-vectorizable gather (030)"
verdict: INVALIDATED
related: [041, 042, 030, 018]
tags: [performance, gpu, rocm, vectorization, vector, build, null, regression, bit-exact, gather-bound, documented-negative]
---

# Spike 043 — vectorize the build's `ord_g`/`ord_h` pair-load as `Vector<f32,2>`

## What This Validates

Given the u64 resident histogram BUILD (`construct_leaf_hist_resident_lds_kernel_u64`)
reads `ord_g[k]`/`ord_h[k]` in lockstep at a SEQUENTIAL (coalesced) `k`, when grad+hess
are interleaved as `Array<Vector<f32,2>>` (one vectorized load/row vs two scalar loads),
then the grad/hess-load slice of the build should fall — BUT bounded hard by the
un-vectorizable bin gather.

## Research — the ceiling was set BEFORE measuring (spike-030)

The wide build is **uncoalesced-bin-gather-latency bound (86–95%)**; grad/hess global
reads are only **8–14%**, the u64 atomic ~0%. So even a perfect halving of the grad/hess
load has a ≈1.08–1.15× HARD ceiling on the build — and only if that load isn't already
latency-hidden behind the gather stall.

### Why the dominant cost CANNOT be vectorized (structural, not measured)

`resident_bins[col + leaf_rows[k]]` is a **permutation** — consecutive `k` map to
NON-consecutive addresses. `Vector` loads only accelerate CONTIGUOUS addresses; there is
no vectorized read that gathers `bins[p0],bins[p1],…` for arbitrary `p`. So the 86–95% is
immune to this lever **by construction**. Only the coalesced `ord_g`/`ord_h` (read at
sequential `k`) can vectorize — and that's the bounded slice this spike measures.

## How to Run

```
cargo run --release --features rocm --example spike043_vector_build_gradhess_ab
```
Source: `crates/lgbm-compute/examples/spike043_vector_build_gradhess_ab.rs`. **HIP-only:**
the u64-atomic build needs `Atomic<u64>`, which cubecl-cpu does NOT implement (panics
"not yet implemented: atomic<u64>") — exactly why the production CPU anchor uses the f64
non-atomic fold (spike-018b). The cpu arm is N/A. Most-favorable test: `leaf_rows` =
identity (root build, COALESCED gather) ⇒ grad/hess is the LARGEST possible fraction ⇒ the
vectorization's best case. Bit-exact by construction (identical `round(v·2^30)`→i64-bits→u64
atomic add; integer adds are order-independent).

## Results

**VERDICT: INVALIDATED.** Vectorizing the build's grad/hess input does **not help** — null
at narrow, a **REGRESSION at wide** — bit-exact every cell (2 restarts).

| shape | run 1 | run 2 | verdict |
|---|---|---|---|
| feat=50 × 200k rows | 1.036× | 0.998× | null (straddles 1.0) |
| **feat=500 × 200k rows (wide)** | **0.830×** | **0.874×** | **REGRESSION** |

### Why (mechanism — the durable finding)

1. **The grad/hess load is a non-bottleneck (8–14%, spike-030)** and its latency is already
   **hidden behind the dominant uncoalesced-gather stall** — so halving it banks nothing.
2. **The `Vector<f32,2>` extract actively REGRESSES the wide build.** `gh[0]`/`gh[1]` add
   extract/register pressure; at wide (500 cubes, high occupancy demand on the 8-CU APU) the
   extra register/LDS pressure costs occupancy → **0.83×**. Vectorizing a non-bottleneck isn't
   free — it competes with the real bottleneck for resources.
3. The dominant cost (the permuted gather) is **structurally un-vectorizable** (above), so
   there is no version of this lever that attacks the actual bottleneck.

This is the third confirmation of 042's rule: **`Vector<P,N>` pays only where the kernel is
memory/throughput-bound AND the vectorized op covers the bottleneck.** The build fails both
clauses — it's gather-LATENCY bound, and the vectorizable slice (grad/hess) is neither the
bottleneck nor contiguous-with it.

### Disposition

DON'T WIRE — null-to-regression. Keep the example as rocm-gated evidence. Definitively
closes the "vectorize the build" question: the wide build is immune to vectorization (the
86–95% gather is a permutation; the 8–14% grad/hess slice regresses when vectorized under
occupancy pressure). Mirrors 030/031/033/036/042 bounded-don't-chase. If discrete gfx110x
ever reopens the build (harsher uncoalesced penalty per 030), the lever there is a
COALESCED build (reorder bins as a partition side-effect), NOT input vectorization.
