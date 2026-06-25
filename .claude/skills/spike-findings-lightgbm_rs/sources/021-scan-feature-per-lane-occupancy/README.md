---
spike: 021
name: scan-feature-per-lane-occupancy
type: standard
validates: "Given the post-u64-build wide GPU train where the per-leaf split SCAN (not the build) is now ~half the per-leaf round-trip, when the fused split-scan kernel is repacked from one SINGLE-THREADED cube per feature (CubeDim=1) to one feature per LANE (CubeDim=W), then the genuine scan launch+readback falls (lane under-utilization removed) while staying bit-exact per feature (each feature's scan is still sequential — no spike-016 reorder)"
verdict: VALIDATED (isolated scan ~3× at W=64) + SHIPPED — end-to-end modest (~1.2–1.3×, APU-noisy); first positive GPU SCAN-kernel lever; bit-exact per feature
related: [015, 016, 018, 019, 007]
tags: [performance, gpu, rocm, scan, split, occupancy, cube-dim, feature-per-lane, wide-shape, bit-exact, device-time-proxy]
---

# Spike 021: feature-per-lane scan occupancy (the post-u64 bottleneck)

## What This Validates

The whole GPU-build campaign (006/007/009/017/018/019/020) optimized the histogram
BUILD; the strong win (u64 fixed-point, spike-018/019) shipped. This spike RE-PROFILED
the wide GPU train **after** that win and found the bottleneck had **moved**: the per-leaf
split SCAN — launched as `CubeCount=(num_features,1,1) × CubeDim(1)`, i.e. ONE
**single-threaded** cube per feature (1 active lane of each wave32 ⇒ ~1/32 ALU
utilization) — is now ~half the dominant per-leaf round-trip. Repacking to **one feature
per LANE** (`CubeDim(W)`, `CubeCount=ceil(num_features/W)`) fills the wave while keeping
each feature's scan **sequential** (bit-exact; no spike-016 reorder).

## Re-profile (the evidence that redirected the attack) — 250k×500, gfx1152 8-CU APU

Decision was *re-profile first, then attack* — because the last full attribution
(spike-014/015) **predated the u64 build win**, and spike-015 had already measured the
obvious candidate (hoisting the 7 per-tree-constant per-feature arrays out of the
per-leaf scan launcher) at **0.2%** — a dead lever.

- **Whole-train BUDGET** (`LGBM_PHASE_PROF=1`): `hist+split` ≈ 85% of train, `build=0`
  (resident-fusion labeling artifact — build folds into "scan"), partition ≈ 13%.
- **Scan round-trip decomposition** (`LGBM_SCAN_PROF=1`): `marshal 0.0% + upload 0.2% +
  launch+readback 99.8%` — confirms the constant-array hoist is dead.
- **Build-drain A/B** (`LGBM_SCAN_DRAIN=1`, drains the async build out of the scan's
  readback sync): the per-leaf "scan" splits **~46% build / ~54% genuine scan**. So the
  genuine scan ≈ 44% of train — a real, large, attackable slice, and it's a
  `CubeDim(1)` single-threaded-per-feature kernel.

## Result — genuine scan launch+readback (build drained), W-sweep

| W (lanes/cube) | cubes @500 feat | launch+readback | speedup vs W=1 |
|----------------|-----------------|----------------:|---------------:|
| 1 (original)   | 500             | 11.80 s         | 1.00×          |
| 32             | 16              | 5.74 s          | **2.05×**      |
| **64**         | **8**           | **3.99 s**      | **2.96×**      |
| 128            | 4               | 3.33 s          | 3.54×          |

Monotonic, diminishing — the scan was genuinely lane-under-utilized. **Default W=64**
(1 cube/CU × 2 wave32 at 500 feat; the robust ~3× knee, not over-fit to W=128's
APU-specific latency-hiding peak). Env override `LGBM_SCAN_CUBEDIM`.

## End-to-end (no drain) — honest, APU-noisy

The isolated 3× does NOT carry to a 3× train: in production the per-leaf readback **sync
is also gated by the (unchanged) build**, so speeding the scan 3× just makes the build
the new bottleneck (Amdahl). Structured `phase_prof`: **learner cumulative −20%**
(23.2→18.4 s), scan share 84.2%→75.5%, total train −8%. Raw wall-clock A/B median
**~1.27×** (27115 vs 21376 rows/s) but with severe APU variance (paired reps ranged
1.4× → wash; one cold run showed a spurious 3.8× — discarded as warmup/thermal). The
**cold isolated ceiling overstates the warm end-to-end** (CONVENTIONS), as expected.

## Parity (bit-exact by construction)

Each lane runs the SAME sequential `split_scan_body` over a DISJOINT histogram region —
no shared state, no reorder — so the per-feature f64 result is **identical for every W**.
`W=1` is byte-identical to the original launch (`ABSOLUTE_POS == CUBE_POS_X`); non-rocm
(cubecl-cpu oracle gate) is pinned to W=1. Gated by `cargo test -p oracle-harness
--features rocm --test kernel_parity` (cubecl-cpu fused==per-feature==native at W=1, hip
split within ~1e-6 at W=64). CPU f64 anchor untouched.

## Disposition

SHIPPED (`LGBM_SCAN_CUBEDIM` default 64 on rocm). The first positive GPU **scan**-kernel
lever (build levers were exhausted). ROI honesty: the GPU is a spoofed 8-CU APU that
loses to the multi-threaded CPU anchor at every shape, so this is **ROCm-parity-track
maintenance** — but the under-utilization it removes is even more wasteful on a real
discrete gfx110x (more CUs, more lanes idle at CubeDim=1), where the end-to-end share
should be larger. spike-016's within-feature parallel scan (reorder-risky, deferred)
remains the *next* scan lever if the readback sync itself becomes the bottleneck.

## How to Run

```
# re-profile (build vs genuine scan):
LGBM_PHASE_PROF=1 LGBM_SCAN_PROF=1 LGBM_SCAN_DRAIN=1 LGBM_BENCH_SWEEP=wide \
  cargo run --release --features rocm --example bench_gpu_vs_cpu
# W-sweep:
for W in 1 32 64 128; do LGBM_SCAN_CUBEDIM=$W ... bench_gpu_vs_cpu; done
# parity gate:
cargo test -p oracle-harness --features rocm --test kernel_parity
```

Change: `crates/lgbm-compute/src/kernels/split.rs` — `find_best_splits_fused_kernel`
(ABSOLUTE_POS feature index + `n_feats` tail guard) and `find_best_splits_fused_inner`
(`scan_cube_dim()` → `CubeDim(W)`, `CubeCount=ceil(n/W)`).
</content>
