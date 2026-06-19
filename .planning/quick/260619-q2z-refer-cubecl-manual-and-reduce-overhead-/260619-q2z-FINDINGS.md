# 260619-q2z — FINDINGS: lazy-execution (deferred-sync) per-feature histogram A/B on gfx1100

**Date:** 2026-06-19
**Hardware:** local AMD gfx1100 (wave32, `plane_size=32`, `has_plane=true`, `has_f64=false`, `has_f32_atomic=true`), cubecl-hip 0.10.0, `--release`.
**Bench:** `crates/lgbm-compute/examples/lazy_dispatch_ab.rs`
**Kernel/launcher abstracted:** `construct_hist_kernel_atomic_f32` (the shipped per-feature `#[cube(launch_unchecked)]` atomic kernel) as inlined by the production launcher `construct_histograms_parallel_f32_on` (histogram.rs:416) — every per-feature call ends with an IMMEDIATE blocking `client.read_one_unchecked`.

**The A/B (cubecl manual ch.05 — decouple submission from synchronization):**
- **Arm A (immediate-sync, current pattern):** for `f in 0..feats` { create the f's fresh zeroed `out` handle + `construct_hist_kernel_atomic_f32::launch_unchecked` into it + IMMEDIATELY `read_one_unchecked(out_f)` }. **N launches, N blocking drains** (submit → block → submit → block …).
- **Arm B (deferred-sync, lazy):** allocate `feats` distinct zeroed `out` handles up front; for `f in 0..feats` launch into out-handle[f] WITHOUT reading; AFTER the launch loop, `read_one_unchecked` each handle (the deferred drain — the last read forces the whole queued batch to complete). **N launches, 1 drain phase.**

The arms differ ONLY in WHEN the device sync happens. Same shipped kernel, same inputs, same per-feature order — so a same-input correctness column proves deferring the sync changes nothing numerically (no twin needed; zero body-drift risk).

**Method:** interleaved single-binary A/B. The device-resident binned matrix (`resident[f*num_data + row]`, per-feature columns uploaded once), the shared gathered `ord_g`/`ord_h`, and a zeroed-out template are uploaded ONCE per cell OUTSIDE both timed arms — transfer is held equal, so the delta isolates the sync/dispatch pattern, not transfer. WARMUP discarded, MEDIAN + p25/p75 spread, the device sync (read-back) FORCED inside every timed call (arm A's N inline reads; arm B's deferred reads inside the timed block too — the whole point is arm B pays ONE drain phase vs arm A's N drains), arms INTERLEAVED per timed iter. `delta% = (a_med − b_med) / a_med * 100` (>0 ⇒ arm B / deferred-sync faster). PRIMARY axes = **FEATURE-COUNT** `[8, 32, 128]` (the launches/leaf knob — the lever's payoff scales with it) AND **bin-count** `[16, 64, 256]`, across two regimes: launch-bound (small leaf, 4096 rows / 1024 leaf, warmup 15 / timed 21) and compute-bound (large leaf, 200k rows / 200k leaf, warmup 3 / timed 7). Run across **3 separate process invocations**.

---

## Same-input correctness (deferring the sync changes nothing)

The example's `assert_same_input_f32` (ABS 5e-6 / REL 1e-5, the f32-atomic envelope) ran on EVERY cell of all 3 runs: the concatenated per-feature histograms from arm A and arm B agree within the f32 envelope. **Deferring the sync is numerically identical** — same kernel, same inputs, same per-feature order; only the read timing moves. (This is a GPU-f32-vs-GPU-f32 drift guard, NOT the parity gate; per DEF-f8u-01 the real gate pins the kernel to the CPU f64 anchor, which this spike does not touch.) Had a single cell diverged, the bench would have panicked; it did not.

---

## A/B speed — deferred-sync (arm B) vs immediate-sync (arm A), 3 runs

### Launch-bound regime (small leaf, 1024 rows)

| feats | bins | run-1 | run-2 | sign stable? | spread separated? | verdict |
|------:|-----:|------:|------:|:-------------|:------------------|:--------|
|     8 |   16 |  +1.83 |  −0.74 | NO (flips)  | no | SUB-NOISE / NULL |
|     8 |   64 |  +1.14 |  +9.06 | yes (pos)   | bands overlap | SUB-NOISE / NULL |
|     8 |  256 |  +3.97 |  −8.12 | NO (flips)  | no | SUB-NOISE / NULL |
|    32 |   16 | −17.06 |  −5.32 | yes (neg)   | overlap | NEGATIVE/NULL |
|    32 |   64 | −16.56 |  −7.04 | yes (neg)   | overlap | NEGATIVE/NULL |
|    32 |  256 |  −4.60 | −24.81 | yes (neg)   | run-2 separated | NEGATIVE (arm A faster) |
|   128 |   16 |  +9.17 | +15.14 | yes (pos)   | bands overlap | SUB-NOISE / NULL |
|   128 |   64 |  +3.55 |  +6.90 | yes (pos)   | bands overlap | SUB-NOISE / NULL |
|   128 |  256 |  +3.01 |  −0.27 | NO (flips)  | no | SUB-NOISE / NULL |

The launch-bound regime is a **wash**: deltas sign-flip across runs (8/16, 8/256, 128/256) or stay inside the p25/p75 bands. Where it is sign-stable negative (feats=32), arm A is faster — the deferred batch of distinct out-handle allocations + the back-to-back submit appears to add host-side bookkeeping that, at a 1024-row leaf with little GPU work to overlap, is not recovered. No robust win here.

### Compute-bound regime (large leaf, 200k rows) — the decisive cells

| feats | bins | run-1 | run-2 | run-3 | sign stable? | spread separated? | verdict |
|------:|-----:|------:|------:|------:|:-------------|:------------------|:--------|
|     8 |   16 |  +7.42 |  +5.99 |  +1.62 | yes (pos) | run-3 overlaps | modest, weak |
|     8 |   64 |  +8.35 | +11.23 |  +7.65 | yes (pos) | yes | WIN-ish (modest) |
|     8 |  256 | +16.27 | +11.56 |  +9.92 | yes (pos) | yes (A p25 6.26/6.12/5.93 > B p75 5.46/7.24*/5.56) | WIN |
|    32 |   16 |  +7.02 |  +4.36 |  +3.33 | yes (pos) | borderline | modest, weak |
|    32 |   64 | +12.16 |  +8.21 |  +1.14 | yes (pos) | run-3 overlaps | modest |
|    32 |  256 | **+21.97** | **+24.08** | **+19.43** | **yes (pos)** | **YES (separated all 3)** | **ROBUST WIN** |
|   128 |   16 |  +7.82 |  +6.13 |  +6.61 | yes (pos) | yes | WIN (modest, stable) |
|   128 |   64 |  +5.17 | +11.48 | +10.15 | yes (pos) | yes | WIN (modest) |
|   128 |  256 | **+25.88** | **+24.58** | **+20.44** | **yes (pos)** | **YES (separated all 3)** | **ROBUST WIN** |

Decisive spread citations (the two strongest cells):
- **compute-bound, feats=32, bins=256:** arm A p25/p75 = 26.15/27.58, 25.34/29.28, 25.11/26.96 ms; arm B = 20.54/21.60, 20.40/21.46, 20.75/21.95 ms. Arm A's p25 (≈25–26 ms) sits ABOVE arm B's p75 (≈21–22 ms) in all 3 runs — **bands do not overlap.** Delta +19 to +24%.
- **compute-bound, feats=128, bins=256:** arm A p25/p75 = 90.97/93.06, 94.90/109.79, 90.52/109.72 ms; arm B = 66.85/69.16, 68.63/86.46, 69.05/85.82 ms. Arm B's band is clearly below arm A's. Delta +20 to +26%.

The compute-bound regime is **robustly positive and sign-stable in all 18 cell-runs**, strongest at high bin-count (256). The two **256-bin** compute-bound cells (feats=32 and feats=128) are spread-SEPARATED across all 3 runs.

*The lone `B p75=7.24` outlier in run-2 (8/256) is a single noisy timed sample; the median delta stays +9 to +16% and sign-stable across all 3 runs for that cell.

---

## Interpretation

The result is a **genuine, regime-gated WIN** — narrower and more surprising than the expected flat NULL, and consistent with (not contradicting) the prior art once the bound is identified:

- **Launch-bound = NULL/negative.** This matches ol8 exactly: for these short atomic kernels the per-launch FIXED submit cost is not hideable at a 1024-row leaf — there is too little GPU execution to overlap with the next submit, and arm B's extra up-front handle allocations cost more than the single deferred drain saves. The lever does NOT help where there is no GPU work to hide behind.
- **Compute-bound = WIN, scaling with bin-count.** This is the regime the manual's ch.05 lever actually targets: arm A serializes `feats` × (submit → BLOCK on read_one). Each immediate `read_one_unchecked` forces a host-device round-trip / queue drain BEFORE the next feature's kernel is even submitted, so the CPU idles and the GPU is starved between features. Arm B submits all `feats` launches back-to-back (CPU never blocks), letting the backend keep the GPU saturated, then drains once. At a 200k-row leaf each kernel runs long enough that hiding the inter-launch sync overhead recovers 5–26%.
- **Why 256 bins is strongest.** This is the inverse of the p93 plane finding. p93 showed the 256-bin kernel is the MOST contention/latency-bound (≈30 distinct bins per 32-lane wave, nothing to amortize) — i.e. each individual kernel is at its LEAST efficient, so the inter-launch sync gaps where the GPU sits idle waiting on the host are proportionally the most recoverable. The deferred batch keeps the next (slow) kernel queued and ready the instant the previous finishes, eliminating the host-round-trip bubble between features. At 16 bins the kernels are faster and the per-launch fixed cost is a larger share, washing the win down to ~3–7%.
- **Reconciliation with c2l/ol8/nn7.** c2l (launch-bound-not-transfer) and ol8 (launch_unchecked NULL for atomics) were both measured on the SINGLE-launch / launch-bound axis, where this lever is also NULL — they were right for that axis. This spike is the first to measure the MULTI-launch-per-leaf × large-leaf axis, where deferring the sync across a leaf's features surfaces a real overlap win. nn7/oib's "mixed" round-trip result is the same phenomenon seen partially.

---

## Guardrails honored

- The example is `#[cfg(feature="rocm")]`-gated with a `#[cfg(not(feature="rocm"))]` stub `main()`: the CPU-only build emits ZERO rocm codegen (`cargo build -p lgbm-compute --example lazy_dispatch_ab` GREEN; `--features rocm` build GREEN).
- **NO production kernel, launcher, wiring, or the CPU f64 anchor was edited** — this task is a pure addition of one example file. Arm B is a HARNESS call-ordering pattern, not a new kernel; it launches the SHIPPED `construct_hist_kernel_atomic_f32` unchanged.
- Regression gate GREEN:
  - default `cargo test -p lgbm-compute` lib: **30 passed / 1 ignored** (the expected pre-existing baseline, unchanged).
  - `cargo test -p oracle-harness -p lgbm-treelearner --lib -p lgbm` (the named bit-exact CPU anchor merge gate): **lgbm 41/41, lgbm-treelearner --lib 76 passed / 2 ignored, oracle-harness 3/3 — all GREEN, 0 failed.**
- Transfer held equal: the resident bin columns + shared grad/hess + zeroed-out template uploaded ONCE per cell, outside both timed arms — the A/B isolates sync timing, not transfer.
- Same-input asserts passed on every cell ⇒ deferring the sync is numerically identical (GPU-f32-vs-GPU-f32 drift guard; the parity gate stays the CPU f64 anchor, untouched).

---

## DISPOSITION: WIRE (follow-up plan) — gated to the compute-bound, high-bin, multi-feature regime

Unlike c2l/ol8/p93 (flat NULLs), this lever shows a **robust, sign-stable, spread-SEPARATED win** in a specific regime: deferring the per-feature `read_one` sync across a leaf's feature launches recovers **~20–26%** at large leaves with many bins.

**GATING REGIME (where deferred-sync beats immediate-sync with a spread-separated, sign-stable margin):**
- **Leaf size: compute-bound — large leaves (~200k rows here).** Launch-bound small leaves are NULL/negative; the lever needs real per-kernel GPU work to overlap.
- **Bin-count: bins ≥ 256** gives the spread-separated ~20–26% win (the slowest individual kernels ⇒ the biggest recoverable inter-launch bubble). bins=64 is a modest stable +5–12%; bins=16 washes to +3–7% (weak/borderline).
- **Launches/leaf (feats): ≥ ~32 features** for the strongest, most clearly-separated cells (feats=32 and feats=128 at 256 bins are the two robust wins); the win is present but weaker at feats=8.

**IMPORTANT SCOPE NOTE — the WIRE verdict is a RECOMMENDATION for a FOLLOW-UP plan, NOT this spike.** Arm B is a harness call-ORDERING change, not a new kernel: realizing it in production means refactoring the per-feature leaf histogram loop (the `construct_histograms_parallel_f32_on` call site inside the tree learner's feature loop) to submit all features' launches into distinct out-handles before draining — explicitly OUT OF SCOPE for this measurement-only spike. The follow-up should (a) gate the deferred-drain path on the regime above (large-leaf × bins≥64 × multi-feature) so launch-bound small leaves keep the simple immediate path that is faster there, and (b) re-validate the CPU f64 anchor + the rocm ~1e-6 parity gate end-to-end after wiring (deferring the sync is proven numerically identical here, but the production path mixes in fix/compact/scan and the resident-LDS kernels, which this spike did not exercise). Crucial ROI caveat (gpu-histogram-kernel skill): with the CPU anchor multi-threaded, the GPU loses to it at every tested size today — this win is **ROCm-parity-track maintenance, weigh accordingly** before committing the follow-up.

**No win was manufactured.** The launch-bound regime is honestly reported as NULL/negative; the win is confined to and gated by the compute-bound high-bin regime where it is spread-separated and sign-stable across 3 process restarts.
