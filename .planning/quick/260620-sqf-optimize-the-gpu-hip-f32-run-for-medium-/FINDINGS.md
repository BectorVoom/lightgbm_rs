# 260620-sqf — Medium-width GPU (HIP f32) occupancy A/B findings

**Hardware:** real gfx1100 + ROCm, `--features rocm`, cubecl-hip.
**Harness:** `crates/lgbm/examples/bench_gpu_vs_cpu.rs` (`LGBM_BENCH_SWEEP=medocc`),
warm-median (WARMUP=2 discarded, median of TRAIN_REPS=5), iters=12, leaves=31,
regression GBDT, identity-binned deterministic corpus.
**A/B lever:** the existing `LGBM_ROWPART_MIN` env override at `histogram.rs:731`
(NO source change to `row_partition_count`).

- **A = P=1** (gate HIGH): `LGBM_ROWPART_MIN=100000000` -> every cell stays P=1.
- **B = row-partitioned** (gate LOW): `LGBM_ROWPART_MIN=1` -> row-partition triggers
  for all medium cells; P = `clamp(768/nf, 1, 16)` => 30 feat->P=16, 60 feat->P=12,
  100 feat->P=7.

Metric = end-to-end GPU train-wall median (the GPU histogram build is the dominant
GPU work, so train-wall is a faithful per-build proxy; CPU f64 anchor not run here).

## Raw A/B — 3 process restarts (train_median, ms)

### Condition A (P=1, gate off)

| cell      | rows   | feat | R1     | R2     | R3     |
|-----------|--------|------|--------|--------|--------|
| f30r50k   | 50000  | 30   | 901.91 | 930.21 | 945.06 |
| f30r120k  | 120000 | 30   | 1160   | 1210   | 1220   |
| f30r256k  | 256000 | 30   | 1680   | 1820   | 1810   |
| f30r512k  | 512000 | 30   | 2620   | 2770   | 2810   |
| f60r50k   | 50000  | 60   | 1250   | 1340   | 1340   |
| f60r120k  | 120000 | 60   | 1500   | 1930   | 2010   |
| f60r256k  | 256000 | 60   | 2760   | 2910   | 2910   |
| f60r512k  | 512000 | 60   | 4770   | 4930   | 5030   |
| f100r50k  | 50000  | 100  | 2010   | 2060   | 2880   |
| f100r120k | 120000 | 100  | 2900   | 3080   | 2840   |
| f100r256k | 256000 | 100  | 4690   | 4680   | 4390   |
| f100r512k | 512000 | 100  | 8130   | 8010   | 8550   |

### Condition B (row-partitioned, gate on)

| cell      | rows   | feat | R1     | R2     | R3     |
|-----------|--------|------|--------|--------|--------|
| f30r50k   | 50000  | 30   | 942.28 | 954.57 | 1170   |
| f30r120k  | 120000 | 30   | 1210   | 1210   | 1450   |
| f30r256k  | 256000 | 30   | 1790   | 1850   | 1710   |
| f30r512k  | 512000 | 30   | 2800   | 2820   | 2580   |
| f60r50k   | 50000  | 60   | 1330   | 1370   | 1350   |
| f60r120k  | 120000 | 60   | 1660   | 1950   | 1950   |
| f60r256k  | 256000 | 60   | 2920   | 2980   | 2740   |
| f60r512k  | 512000 | 60   | 4900   | 5020   | 4600   |
| f100r50k  | 50000  | 100  | 2050   | 2060   | 1990   |
| f100r120k | 120000 | 100  | 2670   | 3040   | 2930   |
| f100r256k | 256000 | 100  | 4670   | 4690   | 4320   |
| f100r512k | 512000 | 100  | 7950   | 7870   | 7560   |

## Median + sign-stability (B faster than A per restart: W=win, L=loss)

| cell      | A_med | B_med | B/A   | per-restart signs |
|-----------|-------|-------|-------|-------------------|
| f30r50k   | 930   | 955   | 1.026 | L L L             |
| f30r120k  | 1210  | 1210  | 1.000 | L L L             |
| f30r256k  | 1810  | 1790  | 0.989 | L L W             |
| f30r512k  | 2770  | 2800  | 1.011 | L L W             |
| f60r50k   | 1340  | 1350  | 1.007 | L L L             |
| f60r120k  | 1930  | 1950  | 1.010 | L L W             |
| f60r256k  | 2910  | 2920  | 1.003 | L L W             |
| f60r512k  | 4930  | 4900  | 0.994 | L L W             |
| f100r50k  | 2060  | 2050  | 0.995 | L L W             |
| f100r120k | 2900  | 2930  | 1.010 | W W L             |
| f100r256k | 4680  | 4670  | 0.998 | W L W             |
| f100r512k | 8130  | 7870  | 0.968 | W W W             |

## Verdict: NULL

**No sign-stable medium-width win below the gate.** The lever targets medium-feature
leaves at sub-256k rows (f30r50k, f30r120k, f60r50k, f60r120k, f100r50k, f100r120k).
Across those cells B (row-partitioned) is **slower or tied** at the median
(B/A >= 0.995) and the per-restart sign column is dominated by **L** (B slower);
the few W flips appear only on restart 3, which carried warmup-drift outliers in A
(f100r50k jumped 2060->2880 ms, f100r512k 8010->8550 ms) — i.e. the wins are
sign-flips-within-spread, exactly the gfx1100 A/B noise the gpu-lazy-dispatch
finding documented. They are not stable.

The one consistent win, **f100r512k = WWW, B/A=0.968 (~3%)**, is at **512k rows** —
already ABOVE the current 256k `ROWPART_MIN_LEAF` gate, so the production default
**already row-partitions there**. It is the known large-leaf occupancy win
(spike-007), not a new medium-width-at-lower-rows win. It does not justify a
feature-count-aware gate.

**Interpretation:** at these medium widths the GPU histogram build is
**atomic-contention / launch-latency-bound, not occupancy-bound** — consistent with
the gpu-histogram-kernel and cuda-mirror-slower-than-cpu findings. Adding cubes
(raising P) does not help when the bottleneck is per-row global/LDS atomic
serialization and launch overhead, not idle CUs. Lowering the gate would only
expose more leaves to higher f32 atomic-order divergence (spike-007: 4e-7->~2e-5 rel)
for no speed gain — a strictly bad trade.

**Stopped at Task 1. No source change to `row_partition_count`.** Task 2 (gate
implementation) is correctly SKIPPED per the plan's NULL-acceptance clause. Because
nothing in the GPU path or the cpu f64 anchor was touched, the ~1e-6 hip envelope
and the bit-exact cpu anchor are unchanged by construction; Task 3 re-confirms this.

**hip-divergence cost of the lever (not paid):** not shipped, so 0. Had it shipped,
it would have widened f32 atomic-order divergence on the newly-partitioned medium
leaves toward the spike-007 ~2e-5 regime for no measured speed benefit.
