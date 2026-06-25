# GPU Build — Bottleneck Re-Attribution (post-u64)

**The current, authoritative attribution of the wide GPU histogram BUILD.** Supersedes the
"atomic-contention-bound" framing in `gpu-build-fixedpoint-atomics.md` (that was measured by
spike-015 BEFORE the u64 ship; the u64 atomic made the per-row add free, so the bottleneck
moved and 015 went stale).

## Requirements

- CPU f64 anchor stays bit-exact to C++ — gate any build change with `cargo test -p
  oracle-harness` (esp. `raw_bin_train_matches_cpp_golden`), `-p lgbm-treelearner --lib`,
  `-p lgbm`.
- "GPU is faster" only in the regime the data supports — on the spoofed 8-CU APU the GPU loses
  to the 16-core CPU anchor everywhere; this is ROCm-parity-track maintenance, not
  overall-fastest. Real build-perf payoff is on **discrete gfx110x**.

## The Finding (spike-030)

The live u64 resident build (`histogram.rs:1246 construct_leaf_hist_resident_lds_kernel_u64`,
one-cube-per-feature, P=1 at wide) was decomposed by a **remove-the-suspect A/B** at 250k/1M×500.
Share of build device-time:

| Suspect (deleted in a variant) | Share of build |
|--------------------------------|----------------|
| LDS atomic (`fetch_add`) | **~0%** — u64 made it a native `ds_add_u64`; NOATOMIC ≈ FULL |
| grad/hess global reads | **8–14%** |
| bin-array bandwidth | **3–8%** |
| **uncoalesced bin-gather ACCESS PATTERN** | **86–95%** ← the bottleneck |

Proof it's the *pattern*, not bandwidth: `COAL_BIN` reads the **same 500 MB bin array, same
byte count**, but sequentially (`resident_bins[col + k]`) instead of via the permutation
(`resident_bins[col + leaf_rows[k]]`) — and runs **8–20× faster**. Effective bandwidth is
4.5–10 GB/s, far below the APU DDR5 peak (~60–100) = a latency/divergence stall, not saturation.

## What to Avoid (dead levers, with evidence)

- **Don't chase the atomic.** Post-u64 it's ~0% of the build. Spike-015's "atomic-bound
  ~820 Mr/s" is stale; per-warp LDS replication (017/020) is already shown null at production P=1.
- **Don't reuse grad/hess across features.** That was the original spike-031 hypothesis —
  INVALIDATED: grad/hess reads are 8–14%, not the bottleneck.
- **Don't pre-reorder bins to coalesce the build (on the APU).** The redirected spike-031.
  Ceiling is only **~1.4×** over the real order (see below) and it's **read-once-unamortizable**:
  the build reads each bin once per leaf, and the stable order changes every split, so a reorder
  pass can't amortize (same wall as spike-028's double-buffer null). A membership-mask full-scan
  only breaks even below ~1/5 selectivity — the deep, cheap leaves — while losing on the shallow
  high-row leaves that dominate.

## The Critical Caveat — model the REAL access order

A **random** `leaf_rows` permutation is the WORST case and **overstates the penalty 5–10×**.
LightGBM's partition is STABLE ⇒ every leaf's `leaf_rows` is a **monotone-increasing subset**.
Measured (Mr/s, normalizes row count):

| order | 250k×500 | 1M×500 | vs coalesced ceiling |
|-------|----------|--------|----------------------|
| random (worst case) | 804 | 346 | 14% / 7% |
| **monotone subset (REAL training)** | **4093** | **3405** | **73% / 69%** |
| sequential (ceiling) | 5636 | 4914 | 100% |

The stable order alone already banks ~70% of coalescing. So the build is **effectively tuned**
on the APU.

## When This Reopens

**Discrete gfx110x only** (GDDR6, no shared-DDR5 cache → harsher uncoalesced penalty; the
random→monotone gap may widen). Before any coalescing investment there, **re-run the exact probe
`examples/spike030_build_roofline_ab.rs`** on the real device. If REAL_ORDER sits well below the
coalesced ceiling, a coalesced build (bins pre-ordered as a side effect of partition, or
membership-mask full-scan, regime-gated by selectivity) becomes worth prototyping — bit-exact by
construction (reorder doesn't change the per-bin sum).

## Method (reusable)

"Remove-the-suspect": N variants of the LIVE kernel, each deletes ONE cost, register-accumulate
deleted loads to defeat DCE, pair complementary deletions to disambiguate (SEQ_BIN+COAL_BIN),
report Mr/s, model the REAL access order. Re-attribute after EVERY build change. See CONVENTIONS.

## Origin

Synthesized from spike: 030 (VALIDATED measurement), 031 (CLOSED by 030, not built).
Source files in: sources/030-wide-build-roofline-reattribution/, sources/031-crossfeature-gradhess-reuse/
