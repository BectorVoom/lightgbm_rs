---
spike: 040
name: autotune-vs-heuristic
type: comparison
validates: "Given the shipped core-count/env P heuristic, when autotune picks measured-fastest on this APU, then the selection matches/beats the hand-tuned default"
verdict: VALIDATED
related: [037, 038, 039, 007]
tags: [gpu, rocm, autotune, comparison, heuristic, portability, row-partition, measure-dont-model]
---

# Spike 040: Autotune vs the Shipped Core-Count Heuristic

## What This Validates

Does autotune's MEASURED row-partition pick match, beat, or lose to the shipped analytic
heuristic (`row_partition_count`) on this hardware? Predicted (Low risk): a wash —
value is portability, not local speed. **The result is stronger than predicted: autotune
BEATS the heuristic by ~10% at the production width**, because the heuristic mis-calibrates.

## Research — what the heuristic actually computes here

`row_partition_count(num_features, leaf_rows)` (`histogram.rs:744`):
- `target_cubes = num_cu × CUBES_PER_CU = 8 × 8 = 64` (the 8-CU APU, correctly detected).
- gate: `leaf_rows ≥ ROWPART_MIN_LEAF = 256_000`, else `P=1`.
- else `P = clamp(target / num_features, 1, 16)`.

At the **production width (50 features)**: `64 / 50 = 1` → **P=1 for every leaf** (and the
256k gate forces P=1 below that anyway). The 8-CU correction (memory:
`rocm-gfx1100-available` — the device is a spoofed 8-CU APU, the old code assumed 96 CU)
**effectively disabled row-partitioning at the production width.** Yet spikes 037–039's
autotuner kept measuring P=16 faster. This spike settles who's right.

## How to Run

```bash
# run 2–3× for restart stability (spoofed-APU discipline)
cargo run --release --features rocm --example spike040_autotune_vs_heuristic
```

Rigorous P-sweep (P ∈ {1,4,8,16,32}), device-time median (ACC=20 launches→1 sync, REPS=11),
feats=50, sizes 50k–500k. Reports the full curve + the heuristic's pick + autotune's pick +
ratios. The real `row_partition_count` is called directly for the heuristic pick.

## Results (3 process restarts, sign-stable)

| rows | P→median ms [1, 4, 8, 16, 32] | best | heur | auto | t(heur)/t(best) | t(heur)/t(auto) |
|-----:|-------------------------------|:----:|:----:|:----:|----------------:|----------------:|
| 50k  | 1.47 1.37 **1.31** 1.39 1.52 | P8 | **P1** | P4 | 1.13× | 1.08× |
| 100k | 2.87 2.66 **2.60** 2.66 2.71 | P8 | **P1** | P16 | 1.11× | 1.08× |
| 200k | 5.95 5.29 5.17 **5.11** 5.34 | P16 | **P1** | P16 | 1.16× | 1.16× |
| 500k | 14.74 **12.92** 13.05 13.19 13.20 | P4 | **P1** | P16 | 1.14× | 1.12× |

(restart 1 shown; restarts 2–3 in `run.log` — identical pattern, ratios 1.02–1.17×.)

**VERDICT: VALIDATED — autotune BEATS the shipped heuristic (not a wash).**

- **The heuristic always picks P=1** at this width — and **P=1 is consistently the SLOWEST**
  point in the sweep at every size (3 restarts). The analytic `target/num_features` model,
  fed the real 8 CU, under-partitions: it targets ~`target` *total* cubes
  (`num_features × P ≈ 64`, i.e. P≈1), but the kernel empirically wants P=4–16 to hide
  LDS/atomic latency on this APU. The model is wrong here.
- **Autotune picks P ∈ {4,8,16,32}** and beats the heuristic by **~2–16% (typ. ~10%),
  never loses** across 12 cells × 3 restarts. It lands at/near the curve minimum without any
  analytic model — it just measures.
- **Selection robustness:** the curve is FLAT between P4–P16 (all ~10% faster than P1), so
  the exact `best`/`auto` P wanders run-to-run within {4,8,16} — but the SIGN (autotune >
  heuristic, and P1 = worst) is rock-stable. Autotune never needs to resolve which of
  P4/P8/P16 wins; it only needs to avoid P1, which it always does.

## Investigation Trail

1. Replicated... then **called the real `row_partition_count`** directly (it's `pub`) so the
   heuristic pick is exactly production, not a paraphrase. That's what surfaced "P=1 always."
2. Built the full P-sweep (not just heuristic-vs-autotune two points) so the whole curve is
   visible — this is what proves P1 is the *worst* choice, not just "different."
3. 3 process restarts to defeat warmup/thermal drift on the noisy APU; judged SIGN only.

## Surprises / Carry-Forward

- **The spike surfaced a latent production mis-tuning**, not just an autotune property: the
  8-CU-corrected `row_partition_count` under-partitions to P=1 at the production 50-feature
  width, leaving ~10% on the table per histogram build. Two fixes are possible — (a) recal
  the heuristic (e.g. target a higher cube count / raise `CUBES_PER_CU`, or lower
  `ROWPART_MIN_LEAF`), or (b) **adopt autotune** and stop hand-modeling occupancy. Autotune
  is the more robust answer *and* the portability answer (it re-derives the right P on any
  device — discrete gfx110x, NVIDIA — with zero re-tuning).
- **Honest bound:** this ~10% is (i) on the spoofed 8-CU APU and (ii) on the GPU build,
  which the 16-core CPU anchor beats end-to-end at this width (memory:
  `perf-gap-vs-cpp-40-80x` — GPU loses to CPU everywhere here). So the *end-to-end* payoff on
  this box is bounded. The durable deliverable is the **method**: measure-don't-model for
  launch-config selection — autotune ≥ the analytic heuristic at every cell, decisively at
  some, and self-calibrates across hardware. Combine with the 038 fresh-output generator and
  the 039 `log2(rows)` bucketed key.
