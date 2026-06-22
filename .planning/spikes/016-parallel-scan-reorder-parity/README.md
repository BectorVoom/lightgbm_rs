---
spike: 016
name: parallel-scan-reorder-parity
type: standard
validates: "Given the within-feature best-split scan, when its f64 prefix-sums are REORDERED (as a parallel scan would), then does the chosen split (threshold/default_left) stay within the hip ~1e-6 parity gate, or does the argmax flip → tree divergence?"
verdict: PARTIAL
related: [015]
tags: [performance, gpu, rocm, scan, parity, argmax, reorder, host-probe, cheap-probe]
---

# Spike 016: parallel-scan reorder parity probe

## What This Validates

Parallelizing the within-feature scan (`split_scan_body`, split.rs) requires a parallel
f64 prefix-sum, which REORDERS the additions vs the current sequential scan. Bit-exactness
is impossible. The open question: does the reorder FLIP the argmax (best_threshold /
default_left) → a different tree → divergence beyond the ~1e-6 hip gate?

## Method (cheap host probe — spike-008 precedent, no GPU)

Pure host f64. Replicated the reverse+forward scan (common `offset=0`/no-skip path) calling
the REAL `gain::get_split_gains`, computing the best split TWO ways over 200k representative
histograms (bins 16/64/128/256): **sequential** cumulative sum (current) vs **pairwise-tree**
sum (what a parallel prefix-scan produces). Measured best_threshold / default_left flip rate
+ gain divergence. `examples/spike016_scan_reorder_probe.rs` (`SPIKE016_TIESTRESS` env: 0.0 =
realistic, 0.3 = artificial near-tie worst case).

## Results

| config | thr_flips | dleft_flips | max gain reldiff |
|--------|-----------|-------------|------------------|
| realistic (TIESTRESS=0.0) | **0 / 200k (0.00%)** | 68190 (34%) | 1.0e-12 |
| near-tie worst (TIESTRESS=0.3, 1e-9 clustering) | 21158 (10.6%) | 56189 (28%) | 2.3e-12 |

**VERDICT: PARTIAL — the partition is parity-safe; the default-direction flag is the open residual.**

- **best_threshold is STABLE under reorder on realistic data: 0 flips in 200k.** Since the
  threshold never flips, the **partition of present data is always identical**. Gains diverge
  only ~1e-12 — three+ orders of magnitude inside the 1e-6 hip gate. The core fear (reorder
  picks a *different split* → tree divergence) is NOT realized on generic continuous
  histograms; flips need ~1e-9 near-ties that don't occur naturally (only when artificially
  injected). So a parallel scan would choose the **same splits**.
- **default_left flips ~34%** at equal-gain reverse-vs-forward ties (threshold identical, only
  the missing/default-direction flag differs — `strict >` resolves the 1e-12-apart gains the
  other way after reorder). This is the same near-tie phenomenon the hip split parity test was
  already made tie-aware for (def-hip-split, commit 1832206).

## Limitations (honest)

- The probe models the common `offset=0`/no-skip path; it does NOT faithfully reproduce the
  kernel's **default-bin (`most_freq_bin`) semantics**, so whether a default_left flip is
  COSMETIC (missing-only, predictions identical on present data) or a REAL default-bin
  partition difference is **not resolved here**. That requires the actual GPU kernel run
  against `kernel_parity_split_within_tol_on_hip`.
- "Pairwise-tree" is a representative proxy for a parallel-scan order, not the exact order a
  specific LDS/plane implementation would use; the divergence MAGNITUDE (~1e-12) is the
  transferable result, not the exact per-case flip.

## Signal for the build (wire / don't wire)

**Feasible and promising, NOT a clean green-light:**
1. The partition (threshold) is parity-safe under reorder — the main risk is retired.
2. The residual default_left tie-flips MUST be confirmed against the real hip split parity
   test (a GPU prototype) before wiring — the host probe cannot resolve the default-bin
   semantics. If that test (already tie-aware) passes → wire within the ~1e-6 best-effort gate.
3. The parallel argmax should still implement the lowest-t / reverse-first tie-break to
   minimise gratuitous flips.
4. **ROI is low:** scan is 11% of GPU device time, already cross-feature parallel (n cubes),
   and on the spoofed 8-CU APU the GPU loses to the CPU overall ([[wide-tall-two-backend-root-cause]]).
   Worth it only if GPU perf becomes a priority on real discrete gfx110x hardware.

**Recommendation:** DEFER wiring. The parity risk is acceptable-pending-GPU-confirmation, but
the ROI doesn't justify the parallel-scan + tie-aware-argmax implementation now. Revisit if/when
discrete-GPU training perf matters; the next step is a GPU prototype gated on the hip parity test.
