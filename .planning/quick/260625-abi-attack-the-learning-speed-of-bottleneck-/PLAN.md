---
quick_id: 260625-abi
slug: attack-the-learning-speed-of-bottleneck
created: 2026-06-25
status: complete
mode: re-profile-then-attack
---

# Attack the GPU learning-speed bottleneck

## Framing (grounded in spikes 007, 014–020 + live code map)

The naive target — the atomic-contention histogram **BUILD** — is **already heavily
optimized**. The live wide-shape kernel is `construct_leaf_hist_resident_lds_kernel_u64`
(u64 fixed-point LDS atomics, spike-018/019: ~1.3–1.7× + ~3600× accuracy + deterministic),
on top of row-partition (007), once-per-train bin upload (p9v), native-width upload (qix),
and cache-friendly host passes (rdu/rsh). The build micro-lever space is **exhausted**:
u8 bins (006), feature-packing (009), int16 quant (008), warp-aggregation (p93), and
per-warp LDS replication (017/020 — regresses at production P=1) are all dead/null.

**Decision (user, 2026-06-25):** *Re-profile first, then attack the scan-launcher.*

## Why re-profile is mandatory before touching code

The last full attribution (spike-014/015) **predates the u64 build win**. spike-015
measured the per-leaf scan round-trip as `marshal 0.0% + upload 0.1% + launch+readback
99.8%` — but a build-drain (`LGBM_SCAN_DRAIN=1`) revealed the async f32-atomic BUILD was
*materializing inside the scan's readback*. So "scan = 96%" was partly the build hiding
there. Now that the build is faster (u64), the true split is unknown.

**Hard caveat:** the spike-015 candidate — hoisting the 7 per-tree-constant per-feature
arrays out of the per-leaf scan launcher (`find_best_splits_fused_inner`, split.rs:1180–1234)
— was *already measured at ~0.1%* (marshal+upload). It is **not assumed to be the win.**
The re-profile decides the real target among:
  1. **marshal+upload** (the constant-array hoist) — likely small per spike-015.
  2. **launch** — the scan kernel launches single-threaded cubes (`CubeDim(1)`, one per
     feature); massive GPU under-utilization (but spike-016 deferred parallel-scan: parity
     risk + ROI, scan ~11% of device time).
  3. **readback round-trip** — synchronous `read_one_unchecked` per leaf (~248×/train).
  4. **build still dominant** — if so, the levers are genuinely exhausted on this APU.

## ROI honesty (non-negotiable to state)

This GPU is a **spoofed 8-CU APU** (gfx1152, shared DDR5); the CPU f64 anchor **beats it
at every tested shape**. Any GPU speedup is **ROCm-parity-track maintenance**, not an
overall-fastest win. Wall-clock on this APU is APU-confounded — judge **relative
attribution shares and sign-stable A/B ratios**, not absolute Mr/s.

## Tasks

- [ ] **T1 — Re-profile (whole-train BUDGET).** `LGBM_PHASE_PROF=1 LGBM_BENCH_SWEEP=wide`
      on `bench_gpu_vs_cpu` (rocm). Capture macro attribution: binning / setup / per-iter
      grad+score / LEARNER (per-tree) / in_learner_other. Confirm whether the scan/split
      path is still the dominant in-learner cost post-u64.
- [ ] **T2 — Scan round-trip decomposition.** `LGBM_SCAN_PROF=1` (+ `LGBM_SCAN_DRAIN=1`
      A/B) to split per-leaf scan into marshal / upload / launch+readback / build-drain.
      This is the decisive measurement — it names the true target.
- [ ] **T3 — Attack the dominant slice** (contingent on T1/T2):
      - if marshal+upload non-trivial → hoist constant per-feature arrays to once-per-tree
        (cache Handles keyed on the leaf-invariant feats layout; guard feature_fraction_bynode).
      - if launch dominates → widen the scan cube (`CubeDim>1`) / batch leaves per launch.
      - if readback dominates → defer/coalesce the per-leaf sync.
      Gate: bit-exact CPU anchor untouched; GPU within ~1e-6 ROCm gate.
- [ ] **T4 — Measure end-to-end** on `bench_gpu_vs_cpu` wide (sign-stable, ≥2 restarts);
      record the win (or honest NULL) and parity. Atomic commit.

## Gate

`cargo test -p lgbm-treelearner --lib`, `-p lgbm-boosting --lib`, `-p oracle-harness`
(model-text / per-iter / raw_bin goldens). CPU f64 anchor stays bit-exact; the change is
GPU-path only.
</content>
</invoke>
