---
spike: 045
name: coalesced-build-vector
type: standard
validates: "Given the wide u64 build's permuted bin gather blocks vectorization (043), when each leaf's rows are REORDERED contiguous first and grad/hess/bin are read coalesced as Vector<P,N> + scattered into LDS (the user's OpenCL-style architecture), then (A) reorder+coalesced-build beats the permuted build NET, and (B) Vector pays on the now-contiguous layout"
verdict: INVALIDATED
related: [043, 042, 041, 030, 017, 020, 031]
tags: [performance, gpu, rocm, vectorization, vector, build, coalesce, reorder, lds, null, regression, bit-exact, documented-negative]
---

# Spike 045 — coalesced-build + `Vector<P,N>` (the user's "reorder → coalesced vector read → LDS scatter" architecture)

## What This Validates

The user's idea: read grad/hess/bin **continuously** as `Array<Vector<P,N>>` in the histogram
build, with **coalesced reads** + **LDS-buffered scattered atomic writes** (the LightGBM OpenCL
local-memory architecture). Spikes 041–044 already mapped `Vector<P,N>` across the histogram
pipeline; **043** tested vectorizing grad/hess in the build directly and found null→regression
because the dominant cost — the permuted bin gather `bins[col+leaf_rows[k]]` — is
**structurally un-vectorizable** (consecutive `k` → non-consecutive addresses; `Vector` only
loads contiguous addresses).

This spike tests the **one un-probed angle 043 explicitly named**: remove the blocker by
**reordering each leaf's rows contiguous FIRST** (a partition side-effect), so the build then
reads grad/hess/bin coalesced — and `Vector<P,N>` can finally apply. Two questions nobody had
measured:

- **(A) NET** — is `t(reorder) + t(coalesced build)` < `t(permuted build)`? (030 measured the
  coalesced build's *ceiling* but assumed the reorder pass free and never timed it.)
- **(B) Vector-on-coalesced** — does `Vector<P,N>` pay on the contiguous layout (the cell 043
  was structurally blocked from testing)?

## Research / Method

- Builds on the 030 roofline harness (same wide P=1 shape, same cache-hostile bin data) and the
  041/043 `Vector<P,N>` launch recipe (generic `N: Size`, runtime N positional arg after
  `CubeDim`, `ArrayArg::from_raw_parts` length in vector units, `#[unroll] for j in 0..N::value()`).
- **Honest baseline = the MONOTONE leaf order** (030's decisive caveat): `leaf_rows =
  (0..num_data).step_by(2)` (a 50%-selectivity leaf), NOT a random permutation — the stable
  partition makes every leaf's rows monotone-increasing, which already sits at ~70% of the
  coalesced ceiling. Using a random permutation would overstate FULL's cost 5–10×.
- **grad/hess need NO reorder.** In production they are already leaf-ordered
  (`ordered_gradients_`, read at sequential `k` by the build); only the global bin matrix is
  indexed by row id. So the reorder pass = the **bins gather only** (the honest cost).
- Arms (all build the byte-identical u64 histogram):
  - **FULL** — permuted scalar build (`bins[col+leaf_rows[k]]`, the production kernel).
  - **REORDER** — write-coalesced bins gather `bins_c[col+k] = bins[col+leaf_rows[k]]` (price of admission).
  - **COAL_S** — coalesced scalar build over `bins_c` (030's ceiling, re-confirmed).
  - **COAL_V2/V4** — coalesced **vector** build: `Vector<u32/f32,N>` loads + N scalar LDS scatters.
- Discipline (CONVENTIONS 017–019/030): accumulate LAUNCHES into one reused `out` + a single
  `read_one_unchecked`; median of REPS; bit-exact column (single-launch) every cell; judge the
  **sign** (spoofed 8-CU APU confounds magnitude); ≥2 process restarts.
- **3 implementation gotchas hit** (carry-forward): (1) the `#[cube]` macro REJECTS a literal
  `Vector<u32, 2>` ("Punctuated::push_value" panic) — `N` must be a generic `Size` param, not a
  literal. (2) `let nlanes = N::value() as usize; for j in 0..nlanes` makes `vb[j]` a RUNTIME
  Vector index ⇒ **segfault**; the idiom is `#[unroll] for j in 0..N::value()` (keeps `j`
  comptime; cubecl `runtime_tests/vector.rs:80`). (3) the reorder dest stride is `r` (leaf row
  count), NOT the source stride `num_data` — conflating them = OOB write ⇒ segfault.

## How to Run

```
cargo run --release --features rocm --example spike045_coalesced_build_vector_ab
S045_250K=1 cargo run --release --features rocm --example spike045_coalesced_build_vector_ab  # 250k only
```
Source: `crates/lgbm-compute/examples/spike045_coalesced_build_vector_ab.rs`. **HIP-only** (u64
atomics; cubecl-cpu has no `Atomic<u64>`). Bit-exact by construction (reorder is a permutation
of the same rows ⇒ same multiset of `round(v·2^30)` integer adds; order-independent).

## Results

**VERDICT: INVALIDATED — on BOTH counts.** The coalesced-build rewrite is a NET LOSS, and
`Vector<P,N>` REGRESSES even on the contiguous layout. Bit-exact every cell (2 restarts).

Sign-stable across **2 process restarts × 2 shapes** (the spoofed APU confounds magnitude — the
REORDER pass especially, 53–114% of FULL run-to-run — so the SIGN is the deliverable):

| shape | arm | vs FULL | vs COAL_S | bit-exact |
|---|---|---|---|---|
| **250k×500** | REORDER (bins gather) | 53% / 114% of FULL | — | — |
| | COAL_S (coalesced scalar) | 2.06× / 1.56× | 1.0× | ✓ |
| | COAL_V2 | — | **0.78× / 0.91×** | — |
| | COAL_V4 | — | **0.70× / 0.87×** | ✓ |
| | **NET = REORDER+COAL_S** | **0.98× / 0.56× (LOSS)** | — | — |
| **1M×500** | REORDER (bins gather) | 62% / 69% of FULL | — | — |
| | COAL_S (coalesced scalar) | 1.37× / 1.78× | 1.0× | ✓ |
| | COAL_V2 | — | **0.84× / 0.97×** | — |
| | COAL_V4 | — | **0.87× / 0.97×** | ✓ |
| | **NET = REORDER+COAL_S** | **0.74× / 0.80× (LOSS)** | — | — |

### (A) NET architecture = LOSS — the reorder IS the gather you were avoiding

NET is a **loss in all 4 cells** (0.56–0.98× vs FULL). The REORDER pass is the SAME permuted bin
gather (030's 86–93% bottleneck), merely write-coalesced + no atomic instead of
accumulate-into-LDS — so it's sometimes a bit cheaper than FULL (53–69%) but never free. The
algebra is fatal: COAL_S is still ~half of FULL (real work remains), so `NET = REORDER + ~0.55·FULL`
beats FULL only if REORDER < ~0.45·FULL — and it never gets that low (best observed 53%). You pay
the expensive permuted-read traffic ONCE in the copy and THEN run a non-trivial coalesced build =
strictly more work than paying the gather once inside the build. This is the concrete confirmation
of 030's prediction: the build reads each bin **once per leaf**, the stable order **changes every
split**, so a reorder pass has **nothing to amortize against** (the read-once wall that also killed
CPU double-buffering, spike-028). Even the single near-break-even cell (250k r1, 0.98×) is a wash,
not a win.

### (B) Vector-on-coalesced = REGRESSION — 043's reopened cell answers NEGATIVE

Even with reads now contiguous (043's structural blocker removed), COAL_V2/V4 are **slower than
COAL_S in all 8 cells** (0.70–0.97×, never > 1.0). Mechanism (consistent with the 041-rule): once the
build is coalesced, its bottleneck is the **scattered LDS atomic** (N `fetch_add`s per vector
load) + occupancy, NOT the load — so vectorizing the load attacks a non-bottleneck, and the
`vb[j]`/`vg[j]` extract adds register/occupancy pressure that regresses (the exact 043 wide
mechanism). `Vector` pays only where the kernel is load/throughput-bound AND the vectorized op
covers the bottleneck; the coalesced build fails the second clause.

### Disposition

DON'T WIRE — null-to-loss on both counts; the live build + CPU anchor untouched (bit-exact gate
N/A, probe-only). This **closes the user's "coalesced-build + Vector" architecture**: the one
lever 043 named is a net loss on the APU because the reorder can't amortize, and vectorization
doesn't survive the move to a coalesced (atomic-scatter-bound) build. Combined with 041–044, the
`Vector<P,N>` histogram frontier is now **fully closed including the coalesced-rewrite escape
hatch**. The ONE place this could reopen: **discrete gfx110x**, where 030 predicts the permuted
penalty is harsher (GDDR6, no shared-DDR5 L2 absorbing the monotone strides) — there FULL gets
costlier so the reorder *might* amortize and COAL_S's win widens; re-run this exact probe there
before any coalescing investment. On this APU the wide build already loses to the 16-core CPU
anchor ~4× (ROCm-parity-track), so the question is moot for production routing.

## Investigation Trail

1. Confirmed via 041/043 READMEs + CONVENTIONS that the direct grad/hess/bin vectorization is
   already closed (043 INVALIDATED; bin gather structurally un-vectorizable). Identified the
   reorder→coalesced rewrite as the single un-probed angle (043's own disposition names it).
2. Built the 4-arm harness off the 030 roofline shape. Hit 3 segfaults/compile-panics in
   sequence (literal-`Vector<_,2>` macro panic → generic `N:Size`; runtime-index unroll segfault
   → `#[unroll] for j in 0..N::value()`; reorder dest-stride OOB → separate `col_src`/`col_dst`).
3. First green run showed `bit_exact=false` — caught a TEST bug: `build_full` reads `ord_g[k]`
   (grad/hess already leaf-ordered) but the initial reorder re-gathered g/h by `leaf_rows[k]`.
   Corrected: only bins need reorder; g/h read directly. Bit-exact then held.
4. Ran both shapes × 2 restarts. NET LOSS in all 4 cells (0.56–0.98×); COAL_V < COAL_S in all 8
   cells (0.70–0.97×). REORDER magnitude noisy (53–114% of FULL) but the two signs never flip.
   Both counts INVALIDATED.
