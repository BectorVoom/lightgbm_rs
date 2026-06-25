---
spike: 033
name: partition-gather-prefetch
type: standard
validates: "Given the post-spike-032 one-random-gather host partition (split_fused_host pass-1), when the random feature_bins.bin(row) gather is software-prefetched D rows ahead, then the gather runs faster (latency hidden) — byte-identical (prefetch cannot change values)"
verdict: PARTIAL
related: [032, 027, 030, 026]
tags: [performance, cpu, partition, prefetch, memory-latency, gather, x86, autovectorization, roi-gated, isolated-ab, wide-shape]
---

# Spike 033: Software-Prefetch the Residual Random Bin-Gather

## What This Validates

Given the post-spike-032 one-random-gather host partition (`split_fused_host` pass-1),
when the random `feature_bins.bin(self.indices[begin+i])` gather is software-prefetched
`D` rows ahead (`_mm_prefetch(col[indices[begin+i+D]])`), then the gather runs faster by
hiding miss latency — byte-identical (a prefetch hint cannot change values).

## Research

No external deps — host logic against the real `BinColumn`. Method = the CONVENTIONS
"CPU/host isolated-A/B harness (026–029)": self-contained
`crates/lgbm-compute/examples/spike033_*.rs`, deterministic LCG, scattered leaf, sweep
size (100k→4M) × skew (0.0/0.9) × bin width (U8 production / U32). **pass-1-only timing
with PREALLOCATED buffers** (the sensitive isolation — prefetch acts only on the gather;
the allocator would swamp it), plus the honest whole-op (pass1 + shared scatter) dilution.
median of 25 interleaved reps, warmup discard, 3 process restarts.

Three variants, clean attribution (vary one thing):

| Variant | Gather | Prefetch | vs production |
|---------|--------|----------|---------------|
| **V0 prod** | `.bin()` enum-match per row (the shipped path) | no | baseline |
| **V1 hoist** | match variant once → typed `&[T]` slice | no | isolates the match-hoist |
| **V2 pf** | typed `&[T]` slice | `_mm_prefetch` at D∈{16,32,64,128}, T0/NTA | isolates prefetch over V1 |

Prefetch via `core::arch::x86_64::_mm_prefetch` (stable; no-op fallback off-x86).

## How to Run

```
cargo run -p lgbm-compute --example spike033_partition_gather_prefetch_ab --release
# >=2 restarts: LGBM_SPIKE_RUN=2 …, =3 …
```

## Investigation Trail

- Built V0/V1/V2, swept D, T0 vs NTA. Hypothesis up front: prefetch may be NULL because
  the gather loads are independent across `i` ⇒ a wide OoO core already extracts MLP.
- **Run 1 surfaced TWO findings:** (1) prefetch helps only at extreme scale; (2) a SURPRISE
  — V1 hoist (typed-slice, no prefetch) is *slower* than V0 prod (`p0/p1 = 0.5–0.9×`),
  badly so at 4M U32 (~36–50 ms vs ~24 ms).
- Iterated: auto-pick best-D over {16,32,64,128}, added the T0/NTA pollution check, 3
  restarts. Findings held sign-stable.

## Results

**VERDICT: PARTIAL — prefetch is a REAL latency-hiding win when the bin column vastly
exceeds LLC (wide U32 columns at multi-million rows: ~2–3× whole-op at 4M×U32), but at the
production-default U8 width the column is dense enough that even 4M rows only marginally
exceeds cache ⇒ ~1.05–1.16× whole-op at the root split ONLY, and null-to-SLOWER everywhere
else. DON'T WIRE on this codebase's defaults.**

Stable across 3 restarts; parity OK every cell. `p0/pfb` = prod vs best-D prefetch
(pass-1); `op0/opf` = honest whole-op (pass1+scatter); `p0/p1` = prod vs no-prefetch hoist:

| rows | width | p0/pfb (pass1) | op0/opf (whole) | bestD | p0/p1 (hoist) |
|------|-------|----------------|------------------|-------|----------------|
| 100k | 8 | 0.65× (slower) | 0.91× | 16 | 0.60× |
| 500k | 8 | 0.66–0.69× | 0.78–0.92× | 16 | 0.55× |
| 1M | 8 | 0.74–0.85× | 0.83–0.96× | 16–32 | 0.52–0.59× |
| 4M | 8 | **1.08–1.27×** | **1.05–1.16×** | 16 | 0.51–0.62× |
| 100k | 32 | 0.63–0.65× | 0.91× | 16–32 | 0.52–0.55× |
| 500k | 32 | 0.96–1.02× | 0.98–1.01× | 16 | 0.53–0.55× |
| 1M | 32 | **1.22–1.81×** | **1.14–1.23×** | 16 | 0.60–0.88× |
| 4M | 32 | **3.47–3.64×** | **1.89–2.97×** | **128** | 0.63–0.64× |

Reading:
- **Latency-bound crossover = column ≫ LLC.** U32 (4 B/row) crosses at ~1M rows (4 MB);
  U8 (1 B/row) only at ~4M (4 MB). Below the crossover prefetch is pure overhead (slower).
- **Optimal D grows with miss latency / column size:** D=16 for U8, up to D=128 at 4M×U32.
- **T0 ≈ NTA** (1.00–1.02×) — cache pollution of the route/out buffers is a non-issue here.
- **Production trains on U8 columns** (`max_bin` default 255 ⇒ `BinColumn::U8`). The U8 row
  is the real case: prefetch is null/negative below 4M and only ~1.1× whole-op at a 4M-row
  ROOT split; every deeper (smaller, cache-resident) leaf is null-or-slower. The big 2–3×
  is U32-only — a high-cardinality (`num_bin > 65536`) regime that the defaults never hit.

### The surprise — DON'T refactor `.bin()` into a typed-slice loop (autovectorization trap)

V1 hoist (match-once → typed `&[T]` gather) is **consistently 1.5–2× SLOWER** than V0 prod's
per-row `.bin()` match (`p0/p1 = 0.5–0.9×`, all sizes, 3 restarts). The tight typed loop
auto-vectorizes into an AVX **gather** (`vpgatherdd`), which serializes cache-missing lanes
worse than scalar independent loads do under OoO MLP; the per-row enum `match` in `.bin()`
defeats vectorization and stays scalar. So the *natural* way to obtain the column base
pointer for prefetch (match the variant, use the slice) introduces a regression that
prefetch must first claw back — at U8 the net is a wash-to-loss. This is a reusable wiring
landmine (added to CONVENTIONS).

### Disposition — DON'T WIRE (ROI-gated to a regime the defaults don't reach)

The spike-032 one-gather path is already near-optimal for the **U8 production case** on this
hardware — prefetch confirms there is little gather latency left to hide at U8 (the dense
column streams well; HW prefetcher + OoO already saturate MLP). Wiring would add:
- an **x86-only** intrinsic (cfg + non-x86 fallback) — a portability cost on a CPU anchor
  that must build everywhere;
- the typed-slice refactor with its autovectorization regression, or a contrived
  `.bin()`+separate-pointer hybrid;
for a payoff that is **null-to-~1.1× at production width** and only materializes (2–3×) for
**wide U16/U32 columns at multi-million rows** — high-cardinality features, not the default.

**Revisit only if** a high-cardinality (U16/U32) + multi-million-row workload becomes a
target; then wire prefetch behind a `width != U8 && leaf_rows ≥ ~2M` gate at the tuned D
(≈64–128 for U32). On the default U8 path, leave `split_fused_host` as spike-032 shipped it.

Bit-exact gate: N/A (probe-only example; live kernel untouched). Prefetch is value-neutral
by construction (parity OK every cell confirms it).
