---
spike: 039
name: autotune-key-cache-thrash
type: standard
validates: "Given per-leaf histogram shapes vary every node (rows differ per leaf), when AutotuneKey keys on problem dims, then the cache hits warm instead of re-benchmarking per-leaf (a tuning storm) — find the key granularity that amortizes"
verdict: VALIDATED
related: [037, 038, 040, 007]
tags: [gpu, rocm, autotune, autotunekey, cache, thrash, granularity, key-design]
---

# Spike 039: AutotuneKey Cache-Thrash / Granularity

## What This Validates

The histogram build runs once per tree NODE, and every node has a different row count.
If the `AutotuneKey` includes the exact `rows`, (almost) every node is a brand-new key ⇒ a
full cold benchmark at every node ⇒ a **tuning storm** that would dwarf training itself.
This spike finds the key granularity that amortizes — the cache must HIT warm across nodes,
without keying so coarse that it picks the wrong kernel variant for some size regime.

## Research / Setup

- Cache lives in the `LocalTuner`, keyed by `AutotuneKey` + a checksum of the TunableSet
  structure (stable across calls), so a freshly-built set with a seen key still hits.
  (We deliberately build a fresh set per node rather than `tuner.init` — `init` memoizes
  the first set per closure-type and would pin the first node's `rows`.)
- Cleared the persistent cache (`target/autotune`) at start so first-touch of each key is a
  true cold tune; cold/warm split tracked deterministically by first-seen.
- Workload: a simulated tree's per-node row counts — repeatedly split the largest open node
  (child fraction ∈ [0.35,0.65]) until all opens are below the 12k resident gate; collect
  every split node (25 build nodes, sizes 12,514 → 200,000).
- Three keying strategies, same two-variant set (`build_rp` P=1 vs P=16):
  EXACT (`rows`), BUCKET (`floor(log2 rows)`), FIXED (`feats` only).

## How to Run

```bash
cargo run --release --features rocm --example spike039_autotune_key_cache_thrash
```

## Results

| Strategy | Distinct keys | Cold-tune wall | Warm-hit wall | TOTAL | Variant selection |
|----------|--------------:|---------------:|--------------:|------:|-------------------|
| **EXACT** (`rows`) | **25/25** | 975ms | 0ns (none repeat) | **975ms — STORM** | P16×6 / P1×19 (finest, per-size) |
| **BUCKET** (`log2 rows`) | **5/25** | 325ms | 54µs (20 hits) | **325ms (~3.0×)** | P16×3 / P1×2 (per-regime) |
| **FIXED** (`feats`) | **1/25** | 158ms | 53µs (24 hits) | **158ms (~6.2×)** | P16-only (mis-selects small) |

(Warm hits are ~3µs each — effectively free; the cache works perfectly. Numbers
restart-stable to a few %; the one noisy axis is EXACT's P16/P1 split — see Surprises.)

**VERDICT: VALIDATED — keying granularity is a real design lever with a clear sweet spot.**

- **EXACT keying = a tuning storm.** 25/25 nodes cold, ZERO cache reuse, 975ms for ONE
  shallow tree's 25 nodes (~39ms/node). Real trees have hundreds of nodes across hundreds
  of boosting iterations ⇒ this scales to *minutes of pure tuning overhead*. Unusable.
  Per-leaf row counts essentially never repeat exactly, so the cache never amortizes.
- **FIXED keying = cheapest but coarsest.** One tune ever (158ms), 24/25 free — but it
  tunes once on the FIRST node (the 200k root → P16) and applies **P16 to every leaf**,
  including the 12k leaves where 19/25 of EXACT's per-size tunes preferred **P1**. Too
  coarse: it discards the occupancy regime the variant choice actually depends on.
- **BUCKET (`log2 rows`) = the sweet spot.** Only 5 distinct keys (one per size *decade*),
  20/25 nodes free, 3× faster than EXACT — AND it still captures the variant crossover
  (P16 for the big buckets, P1 for the small: 3×P16 / 2×P1). The key insight: **the variant
  choice is a function of the occupancy REGIME (is the leaf big enough to saturate the CUs?),
  not the exact row count.** `log2(rows)` is the natural regime coordinate, so it amortizes
  the cache while preserving selection quality.

## Investigation Trail

1. First pass used a `>50ms` cold-classifier — under-counted (benchmarking these small
   kernels is only ~5–40ms). Switched to a deterministic first-seen split + cold/warm wall
   buckets; the **total wall** and **distinct-key count** are the honest signals.
2. Added a persisted-winners dump (parse `fastest_index` per key file) to measure the
   *selection-quality* side of the tradeoff, not just the tuning cost — this is what
   distinguishes BUCKET (good) from FIXED (cheap but wrong for small leaves).

## Surprises

- **EXACT's P16/P1 split is itself noisy run-to-run** (10/15 one run, 6/19 the next). Many
  small-leaf keys sit near the P1↔P16 *selection tie* on this APU, so the winner is decided
  by f32-atomic timing noise. That's a *second* argument against over-fine keying: you pay a
  full cold tune at every node just to resolve a near-tie that doesn't affect runtime.
  BUCKET/FIXED spend no tuning budget chasing that noise.

## Carry-Forward Requirement

The real build's `AutotuneKey` for the histogram path must key on a **bucketed occupancy
coordinate (`log2(rows)` or a small set of size bands), plus `num_features`/`num_bins`** —
NOT exact `rows`. This amortizes the cache to ~one tune per size band per shape while
keeping the per-regime variant choice. (Combine with the spike-038 fresh-output
InputGenerator.) Feeds directly into spike-040's autotune-vs-heuristic comparison.
