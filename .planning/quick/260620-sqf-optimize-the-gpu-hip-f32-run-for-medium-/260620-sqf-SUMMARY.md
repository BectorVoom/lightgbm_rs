---
status: complete
phase: quick-260620-sqf
plan: 01
subsystem: gpu-treelearner-rocm
tags: [rocm, hip, f32, occupancy, row-partition, gfx1100, gate-tuning, NULL, atomic-bound, sign-flip-within-spread]
provides:
  - "Evidence that the GPU (HIP f32) histogram build is atomic-contention/launch-latency-bound (NOT occupancy-bound) for medium-width leaves at sub-256k rows on gfx1100, so a feature-count-aware lowering of the row-partition gate gives no sign-stable medium-width win."
  - "A reusable medium-width occupancy A/B sweep mode in bench_gpu_vs_cpu.rs (LGBM_BENCH_SWEEP=medocc), driven entirely through the existing LGBM_ROWPART_MIN override (no gate source change)."
affects:
  - "none (no hot-path logic change; only a sweep-mode addition to a bench example + NULL documentation; row_partition_count and the 256k gate are UNCHANGED)"
decisions:
  - "NULL: row-partitioning gives no sign-stable medium-width speedup below the 256k ROWPART_MIN_LEAF gate. Across the targeted sub-256k medium cells, B (row-partitioned) is slower or tied at the median (B/A >= 0.995); the few W flips are restart-3 warmup-drift in A (sign-flips-within-spread, the gpu-lazy-dispatch gfx1100 noise)."
  - "The one consistent win (f100r512k, B/A=0.968 ~3%, WWW) is at 512k rows -- ALREADY above the current 256k gate, so production already row-partitions there (existing spike-007 large-leaf win), not a new medium-width lever."
  - "row_partition_count and the 256k gate left UNCHANGED. Task 2 (feature-count-aware gate) correctly SKIPPED per the plan NULL-acceptance clause. Lowering the gate would only widen f32 atomic-order divergence (spike-007 4e-7->~2e-5) for no speed gain -- a strictly bad trade."
key-files:
  modified:
    - "crates/lgbm/examples/bench_gpu_vs_cpu.rs (added LGBM_BENCH_SWEEP=medocc occupancy A/B sweep grid; warm/median methodology unchanged)"
metrics:
  duration: "~25 min"
  completed: "2026-06-20"
verdict: NULL
---

# Quick 260620-sqf: Optimize the GPU (HIP f32) run for medium-width leaves — VERDICT NULL

**Goal:** test whether the row-partition occupancy trigger (`row_partition_count`,
`histogram.rs:730`), which gates on ROWS ONLY (`ROWPART_MIN_LEAF=256_000`), under-fills
the 96-CU gfx1100 for medium-width leaves — a 50-feature leaf launches only ~50 cubes
yet stays at P=1 until 256k rows. The proposed lever: a feature-count-aware gate that
row-partitions medium leaves at lower row counts. The catch: row-partitioning changes
f32 atomic order (spike-007 saw rel divergence rise to ~2e-5), trading occupancy vs the
~1e-6 hip envelope. MEASURE BOTH. Entirely the RocmBackend f32 path; the cpu f64
bit-exact anchor is UNTOUCHED.

## What was done

### Task 1 — RAN the gfx1100 medium-width occupancy A/B (pure measurement, no gate change)

On real gfx1100 + ROCm (`--features rocm`, cubecl-hip), swept medium feature counts
{30,60,100} x rows bracketing the 256k gate {50k,120k,256k,512k}, warm-median harness
(WARMUP=2 discarded, median of TRAIN_REPS=5), 3 process restarts. A/B driven through the
existing `LGBM_ROWPART_MIN` override — NO source change to `row_partition_count`:

- **A = P=1** (gate HIGH, `LGBM_ROWPART_MIN=100000000`)
- **B = row-partitioned** (gate LOW, `LGBM_ROWPART_MIN=1`; P=clamp(768/nf,1,16) => 30feat->16, 60feat->12, 100feat->7)

#### Verbatim A/B occupancy table (median of 3 restarts, train_median ms; full raw in FINDINGS.md)

| cell      | rows   | feat | A_med (P=1) | B_med (rowpart) | B/A   | per-restart signs (W=B faster) |
|-----------|--------|------|-------------|-----------------|-------|--------------------------------|
| f30r50k   | 50000  | 30   | 930         | 955             | 1.026 | L L L                          |
| f30r120k  | 120000 | 30   | 1210        | 1210            | 1.000 | L L L                          |
| f30r256k  | 256000 | 30   | 1810        | 1790            | 0.989 | L L W                          |
| f30r512k  | 512000 | 30   | 2770        | 2800            | 1.011 | L L W                          |
| f60r50k   | 50000  | 60   | 1340        | 1350            | 1.007 | L L L                          |
| f60r120k  | 120000 | 60   | 1930        | 1950            | 1.010 | L L W                          |
| f60r256k  | 256000 | 60   | 2910        | 2920            | 1.003 | L L W                          |
| f60r512k  | 512000 | 60   | 4930        | 4900            | 0.994 | L L W                          |
| f100r50k  | 50000  | 100  | 2060        | 2050            | 0.995 | L L W                          |
| f100r120k | 120000 | 100  | 2900        | 2930            | 1.010 | W W L                          |
| f100r256k | 256000 | 100  | 4680        | 4670            | 0.998 | W L W                          |
| f100r512k | 512000 | 100  | 8130        | 7870            | 0.968 | W W W                          |

**Crossover:** there is no medium-width crossover below the gate. Row-partitioning only
wins consistently (WWW) at **f100r512k = 512k rows**, which is already ABOVE the current
256k gate (production already partitions there — the known spike-007 large-leaf win).
Every targeted sub-256k medium cell is L-dominated; the scattered W flips are confined to
restart 3, where A carried warmup-drift outliers (f100r50k 2060->2880, f100r512k
8010->8550), i.e. sign-flips-within-spread — not a stable signal.

### Task 2 — SKIPPED (NULL-acceptance clause)

Task 1 produced no sign-stable medium-width win, so per the plan's GUARD
("execute ONLY if Task 1 produced a sign-stable medium-width win"), the feature-count-aware
gate was NOT implemented. `row_partition_count` and `ROWPART_MIN_LEAF=256_000` are
unchanged. The unit test `row_partition_count_heuristic` is unchanged and still green
(confirming the <=8k-row parity shapes still take the P=1 path).

### Speed-vs-hip-divergence

Not shipped => hip-divergence cost paid = **0**. Had the gate been lowered, it would have
exposed the newly-partitioned medium leaves to higher f32 atomic-order divergence
(toward the spike-007 ~2e-5 regime) for **no measured speed benefit** — a strictly bad
trade. The ~1e-6 hip envelope and cpu f64 anchor are unchanged by construction.

## Verdict: NULL-A (build is atomic/latency-bound, not occupancy-bound)

At these medium widths the GPU histogram build is atomic-contention / launch-latency-bound,
consistent with the gpu-histogram-kernel and cuda-mirror-slower-than-cpu findings. Adding
cubes (raising P) does not help when the bottleneck is per-row global/LDS atomic
serialization and launch overhead, not idle CUs. The row-count-only 256k gate is NOT
under-occupying medium-width leaves in a recoverable way. Sub-100-feature GPU gate line:
this lever is closed.

## HARD gate (Task 3) — verbatim parity counts

Because no source touched the GPU gate, the anchor and hip envelope are intact by
construction; the full suite confirms it.

### hip envelope (the f32 ship gate) — gfx1100, `--features rocm`, all within ~1e-6 vs cpu f64 anchor
- `rocm_row_partition`      : **ok. 2 passed; 0 failed**
- `rocm_backend_parity`     : **ok. 4 passed; 0 failed**
- `rocm_parallel_histogram` : **ok. 7 passed; 0 failed**
- `rocm_cuda_mirror`        : **ok. 4 passed; 0 failed**
- `lgbm-compute --lib --features rocm` (incl `row_partition_count_heuristic`): **ok. 44 passed; 0 failed; 1 ignored**

### cpu f64 anchor UNTOUCHED (bit-exact, separate gate) — default features
- `oracle-harness --test kernel_parity`  : **ok. 6 passed; 0 failed**
- `oracle-harness --test learner_parity` : **ok. 23 passed; 0 failed** (29 combined with kernel_parity in one run)
- `lgbm-compute --lib`                    : **ok. 43 passed; 0 failed; 1 ignored**

### Build clean both ways
- `cargo check -p lgbm-compute`                : **Finished (clean)**
- `cargo check -p lgbm-compute --features rocm`: **Finished (clean)**
- `cargo build --release --features rocm`      : **Finished (clean)**

## Deviations from Plan

None affecting logic. The bench harness gained an env-gated `LGBM_BENCH_SWEEP=medocc`
sweep mode and a sweep-only reduced iter count (12 vs 50) to bound multi-restart wall
time (Rule 3 — measurement infra, no methodology change; warm/median preserved). No
change to `row_partition_count`, the 256k gate, or any GPU/CPU hot path.

## Self-Check: PASSED
- FINDINGS.md present at `.planning/quick/260620-sqf-optimize-the-gpu-hip-f32-run-for-medium-/FINDINGS.md`
- SUMMARY.md present (this file)
- commits: bench harness (test), FINDINGS+PLAN (docs), STATE row (docs)
- row_partition_count source: UNCHANGED (verified — Task 2 skipped)
- hip + cpu parity: all green, counts recorded verbatim above
