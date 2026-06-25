---
spike: 030
name: wide-build-roofline-reattribution
type: measurement
validates: "Given the LIVE post-u64+LDS resident histogram BUILD at the wide shape (250k–1M×500, P=1), when build device-time is decomposed by a remove-the-suspect A/B (drop the LDS atomic / replace grad+hess reads with a constant / drop the random bin gather) and compared to the APU DDR5 ceiling, then we learn whether the build is now MEMORY-bandwidth-bound (cross-feature grad/hess read redundancy) or still atomic-bound"
verdict: VALIDATED
related: [015, 018, 019, 009, 007, 023]
tags: [performance, gpu, rocm, histogram, build, re-attribution, roofline, memory-bound, atomic-bound, remove-the-suspect, wide-shape, kill-check]
---

# Spike 030: Wide-Build Roofline Re-Attribution

## What This Validates

Given the LIVE post-u64+LDS resident histogram BUILD at the wide shape (250k–1M×500, P=1),
when build device-time is decomposed by a **remove-the-suspect** A/B and compared to the APU
DDR5 ceiling, then we learn whether the build is now **memory-bandwidth-bound** (the
cross-feature grad/hess read redundancy) or **still atomic-bound** — the kill-check that
gates spike-031.

## Research

**Why re-profile now.** The manifest's iron rule: *"re-profile after every build change — the
bottleneck has moved three times (014→015→023)."* The build attribution that everything since
relies on — **spike-015: "atomic-contention-bound ~820 Mr/s, grows with rows"** — was measured
**before** the u64 fixed-point atomics shipped (spike-018/019). u64 turned the per-row atomic
from an f32 `atomicAdd` CAS-retry loop into a native single-instruction `ds_add_u64`. So the
build's bottleneck has almost certainly **moved again and was never re-attributed**.

**The redundancy premise (confirmed by source read).** The live build
(`histogram.rs:1246` `construct_leaf_hist_resident_lds_kernel_u64`) is **one-cube-per-feature**.
Within a feature the row-partitions don't double-read grad/hess — but **across features, every
feature's cube re-reads the SAME leaf's `ord_g`/`ord_h` from global memory.** Per leaf-build the
device reads:

| Array | width | reads/leaf | bytes @ 1M×500 |
|-------|-------|-----------|----------------|
| `ord_g` + `ord_h` | 4 B each | feats × rows | **4.0 GB** |
| `resident_bins` (u8) | 1 B | feats × rows | 0.5 GB |
| `leaf_rows` (u32 index) | 4 B | feats × rows | 2.0 GB |

⇒ grad/hess traffic ≈ **8× the bin traffic**. If the build is memory-bound, this redundancy is
the bottleneck, and the lever is to reuse grad/hess across features (spike-031).

**Why 009's null doesn't pre-empt 031.** Spike-009 (multi-feature-per-cube) found packing null —
but that was measured when the build was **atomic-bound** (pre-u64), so packing had no traffic
win to capture. If 030 shows memory-bound, 009's null is **stale** and 031 re-tests it in the
new regime.

### Method — "remove the suspect"

Four variants of the live u64 kernel, each deleting ONE suspected cost. Whichever deletion
moves the clock IS the bottleneck (the campaign's cheap-probe discipline; cf. 015's
`LGBM_SCAN_DRAIN` build-drain):

| Variant | What's deleted | Isolates |
|---------|---------------|----------|
| FULL | — (baseline) | — |
| NOATOMIC | per-row LDS `fetch_add` (loads+quantize kept; `acc` defeats DCE) | the ATOMIC cost |
| CONST_GH | grad/hess global reads → constant (atomic+gather kept) | grad/hess READ bandwidth |
| SEQ_BIN | random bin gather → `k%num_bin` (grad/hess+atomic kept) | random-GATHER latency |

Plus the verdict cross-checks achieved **effective GB/s** vs the APU shared-DDR5 peak (~60–100
GB/s for LPDDR5x). Bins u8 (prod native width), P=1 (the wide regime: `target_cubes/500 → 1`).

**Chosen approach:** self-contained `--features rocm` example
(`crates/lgbm-compute/examples/spike030_build_roofline_ab.rs`), modeled on the spike-019
psweep harness (interleaved reps, percentile, SEP-WIN). Spoofed 8-CU APU ⇒ judge the
SIGN/mechanism, not the magnitude.

## How to Run

```
cargo run --release --features rocm --example spike030_build_roofline_ab
```

## What to Expect

Per (250k, 1M) × 500-feat config: FULL / NOATOMIC / CONST_GH / SEQ_BIN median ms, each as a
% of FULL, plus the implied atomic / grad-hess-read / gather shares and FULL's effective GB/s.

- **NOATOMIC ≈ FULL** ⇒ NOT atomic-bound (015 stale post-u64).
- **CONST_GH ≪ FULL** ⇒ grad/hess READ bandwidth dominates ⇒ **031 GREEN**.
- **SEQ_BIN ≪ FULL but CONST_GH ≈ FULL** ⇒ random-gather LATENCY-bound instead.
- effective GB/s near the DDR5 peak ⇒ bandwidth-bound corroborated.

## Investigation Trail

- Built the four-variant probe on the live u64 kernel. Two cube-codegen fixes: `wrapping_add`
  is not a cube intrinsic (use `+`); `build_seqbin`'s shared macro slot must be `usize`.
- **Run 1** (FULL/NOATOMIC/CONST_GH/SEQ_BIN): NOATOMIC ≈ FULL (atomic ~0%), CONST_GH ≈ FULL
  (grad/hess ~8%), **SEQ_BIN = 2–5% of FULL** ⇒ the bin gather is ~95%. But SEQ_BIN deleted
  TWO things (the random pattern AND the `leaf_rows` indirection) — ambiguous.
- **Run 2** added **COAL_BIN** (same 500 MB bin array, same bytes, but read `col+k` SEQUENTIAL
  instead of `col+leaf_rows[k]` RANDOM). COAL_BIN = 5–14% of FULL ⇒ the cost is the
  **uncoalesced ACCESS PATTERN (86–95%)**, not bin-array bandwidth (3–8%). Decisive.
- **Run 3** added **REAL_ORDER** — the honesty check. The probe's random `leaf_rows` is the
  WORST case; real training uses a STABLE partition ⇒ `leaf_rows` is a MONOTONE-INCREASING
  subset. Modeled it as `(0..N).step_by(2)` (50%-selectivity leaf). Result below.

## Results

**VERDICT: VALIDATED (measurement) — the wide GPU build is UNCOALESCED-BIN-GATHER-LATENCY-bound,
NOT atomic-bound (spike-015 is STALE post-u64) and NOT grad/hess-bandwidth-bound (the original
spike-031 premise is DEAD). The dominant cost (86–95%) is the random access pattern of
`resident_bins[col + leaf_rows[k]]`.**

Per-variant share of the live u64 build (P=1, median of 9 interleaved reps, gfx1152 8-CU APU):

| | 250k×500 | 1M×500 |
|---|---|---|
| FULL (baseline) | 3111 ms · 804 Mr/s · **10.4 GB/s eff** | 28898 ms · 346 Mr/s · **4.5 GB/s eff** |
| NOATOMIC (no LDS atomic) | 102% of FULL → **atomic ≈ 0%** | 102% → **atomic ≈ 0%** |
| CONST_GH (no grad/hess read) | 92% → grad/hess ≈ 8% | 86% → grad/hess ≈ 14% |
| COAL_BIN (same array, sequential) | 14% → **uncoalesced pattern ≈ 86%** | 7% → **uncoalesced ≈ 93%** |
| SEQ_BIN (no bin array at all) | 6% → bin-bandwidth ≈ 8% | 3% → bin-bandwidth ≈ 4% |

Effective bandwidth is **4.5–10 GB/s — far below the APU DDR5 peak (~60–100 GB/s)** — the
signature of a latency/divergence stall on uncoalesced reads, NOT bandwidth saturation. u64
made the atomic a native `ds_add_u64`, so it's now free (NOATOMIC is even *slightly slower* —
the final-write keeps the loads live without the LDS staging that overlaps latency).

### The decisive caveat — REAL_ORDER (this is what training actually pays)

Comparing **Mr/s** (normalizes the halved row count):

| order | 250k×500 | 1M×500 | vs random | vs coalesced ceiling |
|-------|----------|--------|-----------|----------------------|
| random `leaf_rows` (FULL) | 804 | 346 | 1.0× | 14% / 7% |
| **monotone subset (REAL_ORDER)** | **4093** | **3405** | **5.1× / 9.8× faster** | **73% / 69%** |
| sequential (COAL ceiling) | 5636 | 4914 | 7.0× / 14× | 100% |

**The fully-random probe overstated the penalty ~5–10×.** Because LightGBM's partition is
STABLE, every leaf's `leaf_rows` is monotone-increasing, and that order alone already reaches
**~70% of the coalesced ceiling** (the GPU L2 absorbs the small, regular strides of shallow
high-row leaves — which dominate build time). The residual coalescing headroom over the real
order is only **~1.4×** (5636/4093, 4914/3405) — BEFORE paying for any reordering.

### Impact on spike-031 (gated on this result)

- **Original 031 (cross-feature grad/hess reuse) — INVALIDATED.** grad/hess reads are 8–14%,
  not the bottleneck. Do not build it.
- **Redirected 031 (coalesce the bin read by pre-ordering bins like `ord_g`/`ord_h`) — real
  lever, but MARGINAL ROI, and 030's data largely closes it without a separate build:**
  the ceiling is ~1.4× over the already-monotone real order, and any scheme to capture it
  either (a) adds a reorder PASS that can't amortize — the build reads each bin ONCE per leaf,
  the stable order changes every split so no cross-level reuse (the same read-once wall that
  killed CPU double-buffering, spike-028), or (b) full-scans in natural order with a membership
  mask, reading skipped rows at COAL speed — which only breaks even below ~1/5 selectivity, i.e.
  exactly the deep, cheap leaves, while LOSING on the shallow high-row leaves that dominate.
  Net ≈ null-to-slight-loss on this read-once build.

### Signal for the build / discrete-GPU

The build is **all access-pattern**: locality of `leaf_rows` is worth 5–10×. The stable
partition already banks most of it. The ONE place this reopens: **discrete gfx110x**, where
the uncoalesced penalty is harsher (GDDR6, no shared-DDR5 cache) and the random→monotone gap
may widen — re-run this exact probe there before any coalescing investment. On the APU the
build is effectively tuned; the wide GPU still loses to the CPU anchor (ROCm-parity-track).

Bit-exact gate: N/A (probe-only example, no production change; live kernel untouched).
