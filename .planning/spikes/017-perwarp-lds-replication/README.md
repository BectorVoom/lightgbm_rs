---
spike: 017
name: perwarp-lds-replication
type: standard
validates: "Given the wide-shape f32-atomic histogram BUILD (one cube/feature, P=1), when each WARP gets its own LDS sub-histogram replica (R = warps/cube) instead of all 256 threads sharing one, then per-cube atomic contention drops and BUILD device-time falls — measured on the CubeCL device-time proxy"
verdict: VALIDATED (modest, ~1.1×) — first positive GPU build-kernel lever; requires FULL per-warp privatization (R8); does NOT change CPU routing
related: [007, 006, 008, 009, 015, 016]
tags: [performance, gpu, rocm, histogram, lds, privatization, replication, contention, device-time-proxy, wide-shape]
---

# Spike 017: Per-warp LDS sub-histogram replication

## What This Validates

Given the wide-shape f32-atomic histogram BUILD (the shipped resident LDS kernel:
ONE sub-histogram per cube, all 256 threads / 8 wave32 warps contending), when each
**warp** gets its **own** LDS replica (R replicas, warp `w` → replica `w % R`, merge
R→1 at the end), then intra-cube atomic contention drops and BUILD device-time falls.
Measured on the CubeCL device-time proxy (option-ii: wall-clock is unvalidatable on
this 8-CU APU, so we read relative device-time A/B ratios).

This is the lever the MANIFEST repeatedly named "the only untried build lever — finer
per-warp LDS sub-histogram privatization (LightGBM's OpenCL `histogram*.cl` does this)"
after spike-015 and quick-260619-p93.

## Research / prior art (consulted before building)

- **quick-260619-p93** already built + hardware-benched the OTHER warp lever —
  **warp-AGGREGATION** (`Plane` ballot+shuffle `match_any` emulation, research
  finding #2). Verdict **NULL/NEGATIVE** (slower in 5/6 cells; at 256 bins a 32-lane
  wave hits ~30 distinct bins → nothing to amortize). Kernel kept as a rocm-gated
  primitive, never wired. **This spike is the COMPLEMENTARY finding #1** (replication),
  NOT a re-run of p93.
- **spike-006** — u8 device bins null; build is "atomic-contention / scattered-read
  latency bound, 234 Mr/s". **spike-007** — row-partition (more cubes) wins 1.3–1.4×
  but is INACTIVE at wide (`target_cubes/500 → P=1`). **spike-015** — wide BUILD is the
  dominant device-time cost, atomic-bound.
- Web research (`.planning/notes/gpu-build-atomic-contention-research.md`): finding #1
  replicated sub-histograms (SC20, NVIDIA shared-atomics), finding #5 AMD LDS
  bank-conflict layout.

Replication is the niche lever for the WIDE shape: row-partition can't add cubes
(already 500 features = enough workgroups), so the only remaining contention lever is
to split the per-cube atomic traffic across per-warp LDS replicas.

## How to Run

```
cargo run --release -p lgbm-compute --features rocm --example gpu_lds_replication
```

Bench: `examples/gpu_lds_replication.rs`. One `#[cube]` kernel `build_repl` with a
comptime `replicas` factor; **R=1 is byte-equivalent to the shipped single-sub-hist LDS
kernel**, so the R sweep {1,2,4,8} isolates exactly the replication codegen + LDS
footprint (comptime-sized, so occupancy honestly reflects the LDS-pressure tradeoff).
Layout mirrors the resident wide path: `CubeCount=(feats,1)`, one cube per feature.
Harness per CONVENTIONS (p93/007): interleaved arms, median + p25/p75 over 11 reps,
2 process restarts, correctness vs the CPU f64 anchor AND vs R=1.

## What to Expect

R8 (= one replica per warp) sign-stable fastest; R2/R4 null; parity within the ~1e-5
ROCm gate.

## Investigation Trail

1. **Prior-art collision (caught before building).** The spike was first scoped as
   warp-AGGREGATION (research finding #2). Reading the code found p93 had already built
   + benched that exact kernel NULL 3 days prior, and the seam was rewired to LDS
   (iaq). Pivoted to the genuinely-untried finding #1 (per-warp replication) via the
   checkpoint.
2. **First run (3-round, speedup-vs-cold-R1-round1): inconclusive + misleading.** R1
   itself swung 366–694ms; a 1.81× "win" was a cold-baseline artifact. Per CONVENTIONS,
   replaced with interleaved median + p25/p75 + 2 process runs.
3. **Rigorous run: R8 sign-stable win, R2/R4 null.** See Results. The non-monotonic
   bin-dependence (1.06–1.29×) is mostly spread noise; the robust core is R8 ≈ 1.1×.
4. **Mechanism isolated from the R-sweep shape.** CUBE_DIM=256 / PLANE_DIM=32 = 8
   warps. R2 = 4 warps/replica, R4 = 2 warps/replica → still cross-warp LDS-atomic
   contention → NULL. Only **R8 = 1 warp/replica** removes inter-warp same-address LDS
   atomic serialization entirely → the win. The win requires **R = warps-per-cube**;
   partial replication does nothing. This is exactly "per-warp privatization."

## Results

**VALIDATED (modest).** R8 (full per-warp privatization) is a **sign-stable ~1.1×
device-time win** (median(R1)/median(R8)), reproduced across both process runs and all
three bin counts; R2/R4 are null (~1.00, often slightly slower). The first POSITIVE
GPU build-kernel lever in the campaign (006/008/009 negative; 007 positive but
wide-inactive; p93 null).

| num_bin | R8 run1 | R8 run2 | R2 | R4 |
|--------:|--------:|--------:|----:|----:|
| 16  | 1.12× (SEP-WIN) | 1.09× | 0.95–0.99× | 0.98–1.04× |
| 64  | 1.06× (SEP-WIN) | 1.07× | 0.94–1.01× | 0.99–1.01× |
| 256 | 1.29× (SEP-WIN) | 1.11× | 0.98–1.04× | 0.98–1.05× |

Throughput R1 ~270–330 Mr/s → R8 ~280–378 Mr/s. R8 wins **despite** 8× the LDS
(worse occupancy on the 8-CU APU) ⇒ the gain is genuine contention relief, not
occupancy. SEP-WIN = R8's p75 below R1's p25 (cleared in the lower-noise run1; run2
medians agree, spreads wider).

**Parity (R8 vs CPU f64 anchor, the merge gate's reference):** 1.1e-5 (16 bins) /
1.1e-6 (64) / 5.9e-7 (256) — within the existing ~1e-5 ROCm gate. At 16 bins R8 is
*closer* to the anchor than R1 (1.1e-5 vs 6.8e-5) — per-warp partial sums are slightly
better-conditioned, the same f32-cancellation regime p93 flagged. R8-vs-R1 reorder
divergence ≤7.9e-5 at 16 bins, ≤2e-5 elsewhere (same class as spike-007's P≥2
divergence; the cpu f64 anchor is untouched — merge gate safe).

### Surprises

- The win **requires the maximal R = warps/cube (R8)**; R2/R4 give nothing. Contention
  relief is all-or-nothing at the warp boundary, because partial replication leaves
  cross-warp atomic serialization intact.
- The win does NOT shrink toward high bin counts as a pure-collision model predicts
  (256-bin is among the largest) — consistent with the bottleneck being per-address
  LDS-atomic *serialization latency* (relieved by privatization regardless of collision
  rate), not collision count. Not fully isolated; flagged.

## Disposition

**Keep `gpu_lds_replication.rs` as the rocm-gated evidence; do NOT wire into production
yet** (p93 precedent for a modest/uncertain GPU win):

- The win is real on the device-time proxy but **modest (~1.1×)** and **APU-only**;
  the LDS-pressure/occupancy tradeoff at R8 (8× LDS) could differ on discrete gfx1100
  (96 CU, more concurrent cubes) — unvalidatable here (option-ii caveat).
- It does **not change the routing reality**: the 16-core CPU still beats the GPU ~4×
  at wide (spike-015); a ~1.1× on BUILD (~53% of GPU device-time) → ≤~1.05× on total
  GPU train. Parity-maintenance / future-discrete-silicon track only.
- Wiring would change the GPU f32 accumulation order → requires an oracle-harness
  parity re-pin (kernel/learner/boosting on gfx1100) + the def-f8u-01 guardrail (pin to
  the CPU anchor, never GPU-f32-vs-GPU-f32).

**Recommended follow-up (only if discrete-GPU perf becomes a goal):** wire R = warps/cube
into the resident LDS build (`construct_hist_kernel_lds_f32` / the resident kernel),
re-pin parity, and re-measure on real gfx110x silicon where the occupancy tradeoff is
real. Closes the MANIFEST's "only untried build lever" with a measured modest-positive.
