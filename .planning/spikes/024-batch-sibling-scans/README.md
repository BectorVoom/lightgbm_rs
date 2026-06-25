---
spike: 024
name: batch-sibling-scans
type: standard
validates: "Given a leaf-split that creates two children, when both children's best-split scans are co-packed into ONE fused launch + ONE host readback instead of two, then the per-tree scan round-trips (blocking syncs) ~halve and the scan launch+readback time falls — bit-exactly (each feature's scan unchanged, just co-packed)"
verdict: VALIDATED
related: [021, 022, 022b, 023]
tags: [performance, gpu, rocm, scan, roundtrip, sibling-batch, sync-floor, bit-exact, isolated-ab]
---

# Spike 024: Batch Sibling Scans

## What This Validates

Spike-023 found the post-021 GPU per-tree cost includes **~59 scan-readback SYNCS/tree =
ONE per leaf-node** — the two siblings of every split are scanned in **two separate
launches + two separate readbacks**. This spike asks: co-pack both siblings into **one**
launch + **one** readback — does the scan launch+readback time fall, bit-exactly?

Per CONVENTIONS (*"probe in an isolated A/B before plumbing a multi-kernel change"* — it
killed 006/008/009/011 cheaply), this is an **isolated microbench**, NOT a growth-loop wiring.

## Method

`examples/spike024_sibling_scan_ab.rs` (self-contained, rocm-gated). Two `#[cube]` kernels
over synthetic interleaved (grad,hess) f64 histograms, feature-per-lane **W=64** (the
SHIPPED spike-021 packing), representative forward best-split scan (compute ∝ num_bin):
- **BASELINE A (= current production):** `scan_one(hist_a)` + `scan_one(hist_b)` — TWO
  launches, TWO `read_one_unchecked` syncs.
- **CANDIDATE B (co-packed):** `scan_two(hist_a, hist_b)` — ONE launch over 2×n_feats
  feature-slots (lane g<n ⇒ sibling-A feat g; g≥n ⇒ sibling-B feat g−n), ONE readback.

Two siblings carry DIFFERENT leaf totals (different seeds). Median + [p25,p75] over 9
interleaved reps × 30 launches, **2 process restarts**. Correctness gate: B's two halves ==
A's two results, **byte-for-byte**, every cell.

## Investigation Trail

- cubecl authoring friction (3 rebuilds): (1) scalars pass **raw**, not `ScalarArg`;
  (2) u32→f64 needs `f64::cast_from`, not `as`; (3) the cube macro supports **neither** a
  `#[cube]` helper here **nor** a `macro_rules!` body — inline directly (022b precedent);
  (4) the best-gain sentinel **must** init from a plain literal `0.0f64` — a
  `-1.0e30f64` scientific literal trips the MLIR lowering (`From<NativeExpand<f64>>`).
  Gains are ≥0 so `0.0` is a valid sentinel (split_scan_body uses the same).

## Results

**VERDICT: VALIDATED (isolated ~2×, bit-exact). The structural sync-floor win is real;
the e2e is bounded by 023's scan-sync fraction (~10–15% small/medium, ~0 wide).**

### Isolated launch+readback A/B (gfx1152 APU; ratio = baseline / candidate, >1 ⇒ co-pack faster)

| regime | RUN 1 | RUN 2 | parity |
|--------|-------|-------|--------|
| small  n=12 nb=32 | **1.99×** | **2.08×** | OK |
| medium n=30 nb=64 | **1.96×** | **2.02×** | OK |
| large  n=40 nb=128 | **2.01×** | **1.98×** | OK |
| n=128 nb=128 | 1.99× | 2.03× | OK |
| n=256 nb=128 | 2.81×(noisy) | 2.09× | OK |
| wide n=512 nb=128 | 1.51× | 1.67× | OK |

**Sign-stable ~2.0× across both runs** at small→n=256, ~1.5–1.7× even at wide. **Parity OK
every cell** (B's two halves byte-identical to A's two scans). The ~2× holds even where the
scan compute is largest (n=512) because on this APU the per-launch **fixed sync latency
dominates** the scan compute — so turning 2 syncs into 1 nearly halves the time.

### Honest e2e ceiling (the cold-isolated-overstates-warm rule, spike-021)

The isolated 2× is on the **scan launch+readback component ONLY**. Co-packing halves the
sync COUNT, not the scan compute, so it reclaims ~half the genuine scan-sync time. Per
spike-023's DRAIN attribution (genuine scan-sync as % of total train):

| regime | scan-sync % of train (023) | ⇒ co-pack e2e ceiling (~½×) |
|--------|------|------|
| small | ~27% | **~13%** |
| medium | ~29% | **~14%** |
| large | ~22% | **~11%** |
| 1M×500 (wide) | ~3.2% | **~1.5%** |

⇒ **e2e is ~10–15% at small/medium, ~0 at wide** — NOT 2×. Exactly the 023 prediction.
(The full e2e A/B needs the production wiring below; the bound is analytic + isolated-backed.)

### ROI gate (the disposition)

Small/medium is **exactly where the CPU anchor crushes the GPU** on this 8-CU APU
(spike-001: GPU 0.06–0.36× of CPU at 20k–100k rows). A 10–15% GPU speedup there does NOT
flip the regime — it nudges the crossover marginally left, no more. The win matters on a
**real discrete gfx110x** (more CUs ⇒ the launch floor is a relatively larger share, and
the GPU is genuinely competitive in the mid regime). On THIS box it is ROCm-parity-track
maintenance, like 021/022.

**Wiring cost (if pursued):** a production 2-slot scan kernel (two histogram Handles, 2×n
SplitInfos, one readback) + a growth-loop reorder (defer the smaller-child scan past the
`subtract_resident` so both slots scan together — both Handles are already simultaneously
resident at that point, per the 023 map) + an oracle parity re-pin. Contained, bit-exact
by construction (each feature's sequential scan is unchanged — no spike-016 reorder).

### Caveats

- Spoofed 8-CU APU ⇒ SIGN judged, not magnitude. The 2× isolated is robust; the e2e is the
  analytic bound, not measured (would need the production wiring).
- Representative gain (g²/(h+λ)), not the full `split_scan_body` — faithful for the
  LAUNCH/READBACK structure question (the split math is already parity-proven, 021).
- The 2-slot candidate branches the histogram read by sibling per bin (Array refs can't be
  selected into a binding) — a slight CONSERVATIVE inflation of B; the real win is ≥ shown.
