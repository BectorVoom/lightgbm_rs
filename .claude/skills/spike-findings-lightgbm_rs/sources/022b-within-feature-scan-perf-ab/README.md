---
spike: 022b
name: within-feature-scan-perf-ab
type: standard
validates: "Given the parity-safe within-feature cooperative scan (022), when a cooperative (K lanes/feature, segmented LDS scan) kernel is A/B'd against the SHIPPED spike-021 feature-per-lane scan (W=64) across feature counts, then does cooperation beat 021 — and in which regime?"
verdict: VALIDATED (experiment) — confirms DON'T WIRE. Cooperation beats 021 only at NARROW (≤256 feat, the GPU's weakest regime vs CPU); at the WIDE production shape (512 feat) it is a WASH-to-regression. Closes the deferred 022b perf question; the within-feature lever stays unwired.
related: [022, 021, 016, 015]
tags: [performance, gpu, rocm, scan, split, occupancy, within-feature, cooperative, plane-scan, device-time-proxy, wide-shape, comptime-ab]
---

# Spike 022b: within-feature parallel scan — the deferred PERF A/B

## What This Validates

spike-022 retired the PARITY risk (within-feature reordered scan is parity-safe ~1e-6) and
deferred the ROI question. This spike answers it: does a **cooperative within-feature scan**
(K lanes per feature, segmented LDS scan + argmax) beat the **SHIPPED spike-021 feature-per-
lane scan** (one lane per feature, `CubeDim=W=64`), and in which regime? Hypothesis (021's
occupancy reasoning): cooperation wins NARROW (021 under-fills the device with few features),
washes/regresses WIDE (021 already saturates).

## How to Run

```
cargo run --release -p lgbm-compute --features rocm --example spike022b_within_feature_scan_ab
```

ONE kernel `scan_coop` with `#[comptime] coop = K` (CONVENTIONS in-kernel-A/B-by-comptime,
p93/017): K=1 = feature-per-lane (no LDS/sync = the 021 path); K>1 = K lanes segment-scan the
bins, combine prefixes + argmax via LDS. `cube_dim` is a runtime param so the baseline can be
021's real W=64. Both arms compute the SAME per-bin gain + argmax ⇒ the ratio isolates the
parallelism structure; K>1 is verified to reproduce K=1's (best_gain, best_bin) per feature.
num_bin=256, device-time median[p25..p75] over 9 interleaved reps, 2 process runs.

## Investigation Trail

1. **First run had an OCCUPANCY CONFOUND.** Initial design fixed `CubeDim=256` for all K, so
   the K=1 baseline ran only `F/256` cubes — *under-occupied* (2 cubes at F=512 on an 8-CU
   device). Result: "cooperation wins everywhere (2–6×)" — but partly because K>1 launched MORE
   cubes, not because of cooperation. The shipped spike-021 is `CubeDim=64` (8 cubes at F=512 =
   fills the device), a much stronger baseline.
2. **Fixed:** made `cube_dim` a kernel parameter; re-benchmarked every candidate against the
   REAL shipped baseline (`cd64 K1`). Kept `cd256 K1` in the sweep to *expose* the confound —
   it runs **0.61–0.63×** the `cd64 K1` baseline at F≥256 (SEP-LOSS), proving the run-1 baseline
   was artificially weak.

## Results (vs the SHIPPED 021 = cd64 K1; ratio >1 ⇒ candidate faster; 2 process runs)

| F (features) | best cooperative vs 021 | verdict |
|--------------|-------------------------|---------|
| 8 | cd256 K32 **5.6×** | cooperation wins big |
| 32 | cd256 K32 **6.2×** | wins big |
| 128 | cd256 K16 **3.3×** | wins |
| 256 | cd64 K8 **2.2×** (SEP-WIN both runs) | wins, shrinking |
| **512** | cd256 K16 **~1.2–1.3×** (noisy `[4..8]`); cd64 K2/K8 **0.95–1.0× / 0.66–0.72×** | **WASH-to-REGRESSION** |

- **Monotonic decay:** the cooperative win shrinks 6× → ~1× as F grows; the device fills with
  feature-level parallelism (021), leaving no idle HW for cooperation to exploit — exactly the
  hypothesis, once the occupancy confound is removed.
- **At the WIDE production shape (F=512), cooperation does NOT beat 021:** most configs ≈tie
  (0.95–1.00×), `cd64 K8` *regresses* to 0.66–0.72× (LDS/sync overhead with no occupancy gain),
  and the only positive (`cd256 K16` ~1.2–1.3×) is noisy AND is an OCCUPANCY play (more cubes),
  not a cooperation win per se — and 021's W is already a tuned occupancy knob.
- **Correctness PERFECT** across all configs/F: `mism=0` argmax flips, `max gainrel ≤ 9e-15`
  (pure f64 reorder noise) — confirms spike-022's parity-safe finding on the *real* cooperative
  kernel, not just the host probe.

**VERDICT: VALIDATED (the experiment) — confirms DON'T WIRE.** Within-feature cooperation beats
the shipped feature-per-lane scan only at NARROW feature counts (≤256), which is exactly the
regime where the GPU is **least competitive vs the multi-threaded CPU anchor** (few features =
little work = CPU wins outright). At the WIDE shape where GPU work actually matters, cooperation
is a wash-to-regression. The spike-022 disposition ("parity-safe but ROI-gated") is now
confirmed on the PERF axis: **the within-feature parallel scan is not worth wiring.**

## Limitations (honest)

- **Proxy kernel:** a single forward scan with a representative `g²/(h+λ)` gain — not the full
  production reverse+forward + default-bin scan (~2× the work + branches). Both arms scale
  equally, so the A/B ratio transfers; the absolute Mr/s does not.
- **Spoofed 8-CU gfx1152 APU:** judge the SIGN/trend, not magnitudes. On real discrete gfx110x
  (more CUs) the crossover would shift — 021 (cd64) would under-fill until larger F, so the
  narrow-win band widens; but the *wide-shape wash* (the regime that matters) should persist,
  and the ROI verdict (GPU loses to CPU; this is parity-track maintenance) is unchanged.
- The marginal `cd256 K16` F=512 win is an occupancy lever (cube count), separable from — and
  simpler than — within-feature cooperation; if wide-GPU occupancy ever matters, tune 021's W,
  don't add cooperation.

## Origin

Closes the deferred 022b perf question from spike-022. Source: `spike022b_within_feature_scan_ab.rs`.
</content>
