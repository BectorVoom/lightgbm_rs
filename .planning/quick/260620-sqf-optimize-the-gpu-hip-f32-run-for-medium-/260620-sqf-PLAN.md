---
phase: quick-260620-sqf
plan: 01
type: execute
wave: 1
depends_on: []
files_modified:
  - crates/lgbm-compute/src/kernels/histogram.rs
autonomous: false
requirements: [SQF-GPU-MEDLEAF-OCC]
must_haves:
  truths:
    - "The gfx1100 A/B sweep produces per-build GPU times for P=1 vs row-partitioned across medium feature counts (30/60/100) and rows bracketing the 256k gate (50k/120k/256k/512k)."
    - "A WIN/NULL verdict is recorded: row-partitioning either gives a real, sign-stable medium-width speedup or it does not (atomic/latency-bound)."
    - "If shipped, the feature-count-aware gate keeps hip divergence inside the documented ~1e-6 envelope vs the cpu f64 anchor."
    - "The cpu f64 bit-exact anchor is provably UNTOUCHED (cpu kernel_parity + learner_parity green; build clean with and without --features rocm)."
  artifacts:
    - path: "crates/lgbm-compute/src/kernels/histogram.rs"
      provides: "row_partition_count gate (Task 2 only, on a real win)"
      contains: "fn row_partition_count"
  key_links:
    - from: "row_partition_count"
      to: "LGBM_ROWPART_MIN env override"
      via: "std::env::var gate threshold"
      pattern: "LGBM_ROWPART_MIN"
---

<objective>
Measurement-first spike: determine whether the GPU (HIP f32) per-leaf histogram
build is leaving **medium-width-leaf occupancy** on the table because the
row-partition trigger (`row_partition_count`, `histogram.rs:730`) is gated on a
**row-count-only** threshold (`ROWPART_MIN_LEAF = 256_000`, `histogram.rs:718`)
even though occupancy actually scales with `num_features × P` workgroups on the
gfx1100 (96 CUs).

The gap (verified by direct code reading): a MEDIUM/NARROW-width leaf (e.g. 50
features) launches only ~50 cubes — far below the GPU's appetite (96 CUs × ~8
wkgrps ≈ 768 target cubes, `ROWPART_TARGET_CUBES`) — yet stays at `P=1` until
256k rows. Unit test `row_partition_count_heuristic` (`histogram.rs:2841`)
confirms `row_partition_count(50, 8_000) == 1`. A **feature-count-aware gate**
could trigger row-partitioning for narrow/medium leaves at LOWER row counts,
restoring occupancy for exactly these shapes. The mechanism already exists
(`row_partition_count` + the LDS f32 kernels at `histogram.rs:967/1450/1599/1682`)
— this is a GATE-tuning change, not a new kernel.

The hard constraint: row-partitioning changes the f32 ATOMIC accumulation ORDER,
raising divergence vs the cpu f64 anchor (spike-007 saw rel divergence rise
4e-7 → ~2e-5 on large leaves). The 256k gate was set PARTLY to keep that
divergence away from the ≤8k-row parity tests (`histogram.rs:713-716` comment).
Lowering / feature-aware-gating EXPOSES more leaves to higher f32 divergence and
**could break the ~1e-6 hip parity envelope**. This is the GPU run only
(RocmBackend f32, ~1e-6 hip envelope) — the cpu f64 bit-exact anchor is a
separate gate and stays UNTOUCHED.

Purpose: close the unexplored GATE question (is the row-count-only 256k gate
under-occupying medium-width leaves?) WITHOUT manufacturing a win or shipping a
parity-envelope violation.
Output: a documented WIN or NULL verdict; on a real win, an env-gated
feature-count-aware gate that stays inside the ~1e-6 hip envelope.
</objective>

<execution_context>
@$HOME/.claude/gsd-core/workflows/execute-plan.md
@$HOME/.claude/gsd-core/templates/summary.md
</execution_context>

<context>
@.planning/PROJECT.md
@.planning/STATE.md
@crates/lgbm-compute/src/kernels/histogram.rs
@crates/lgbm/examples/bench_gpu_vs_cpu.rs
@.planning/phases/04-compute-backend-cpu-first-integer-histograms-rocm/04-ROCM-GAPS.md

# Prior spike (row-partition mechanism, already SHIPPED — do NOT re-explore the
# mechanism; only the GATE is open):
@.planning/spikes/007-row-partition-occupancy
</context>

<tasks>

<task type="auto">
  <name>Task 1: RUN the gfx1100 medium-width occupancy A/B sweep</name>
  <files>crates/lgbm/examples/bench_gpu_vs_cpu.rs</files>
  <action>
    On real gfx1100 hardware with `--features rocm`, measure whether the GPU
    histogram build is occupancy-bound (not atomic/latency-bound) for
    medium-width leaves. Sweep medium feature counts {30, 60, 100} × row counts
    that bracket the 256k gate {50_000, 120_000, 256_000, 512_000}. Use the
    existing `bench_gpu_vs_cpu.rs` warm-median harness (WARMUP discarded, median
    of TRAIN_REPS — the warm-vs-cold rule from the spike-findings skill is
    load-bearing; the cold ceiling overstates the warm win 3-7×). Add the medium
    sizes to the `Size` list if not already present; do NOT change the harness's
    warm/median methodology.

    Drive the A/B via the `LGBM_ROWPART_MIN` env override already wired at
    `histogram.rs:731`: run A = gate HIGH (`LGBM_ROWPART_MIN` = e.g. 100_000_000,
    forcing P=1 / current behaviour for medium-row leaves) vs B = gate LOW
    (`LGBM_ROWPART_MIN` = 1, forcing row-partitioning on). This isolates the
    occupancy lever WITHOUT any source change. Run multi-restart (≥3 process
    restarts) and report sign-stability across restarts within spread — a single
    run is not a verdict (the gpu-lazy-dispatch finding showed gfx1100 A/B can
    sign-flip within spread).

    Record per-build / per-leaf GPU time A (P=1) vs B (row-partitioned) for each
    (num_features, rows) cell, and identify the crossover: at what
    (num_features, rows) does row-partitioning start winning, if ever. Do NOT
    edit `row_partition_count` in this task — this is pure measurement through
    the env override.

    NULL acceptance: if row-partitioning gives no medium-width win at these
    sizes (e.g. the build is atomic-contention/latency-bound, consistent with the
    GPU-build-is-atomic/latency-bound finding and the cuda-mirror-slower-than-cpu
    memory), record NULL with the numbers and STOP — do not proceed to Task 2.
    Do NOT manufacture a win.
  </action>
  <verify>
    <automated>cargo build --release --features rocm --example bench_gpu_vs_cpu 2>&1 | tail -5</automated>
  </verify>
  <done>
    A/B GPU build times recorded for all {30,60,100}×{50k,120k,256k,512k} cells,
    A (LGBM_ROWPART_MIN high / P=1) vs B (LGBM_ROWPART_MIN=1 / row-partitioned),
    ≥3 restarts, sign-stability noted. A WIN (sign-stable medium-width speedup +
    crossover identified) or NULL (no occupancy win; build is contention/latency-
    bound) verdict is written into the SUMMARY. No source change to
    row_partition_count in this task.
  </done>
</task>

<task type="auto">
  <name>Task 2: (ON A REAL WIN ONLY) implement the feature-count-aware gate</name>
  <files>crates/lgbm-compute/src/kernels/histogram.rs</files>
  <action>
    GUARD: execute this task ONLY if Task 1 produced a sign-stable medium-width
    win. If Task 1 was NULL, SKIP to Task 3 (which then just re-confirms the
    untouched anchor) and document the NULL.

    Modify `row_partition_count` (`histogram.rs:730`) so the trigger is
    feature-count-aware instead of the flat row-count-only `ROWPART_MIN_LEAF`:
    trigger row-partitioning when a leaf under-fills the GPU — i.e. when
    `num_features` is well below the occupancy target (`ROWPART_TARGET_CUBES`) —
    at a row threshold that SCALES with how badly it under-fills, derived from
    Task 1's measured crossover (narrower leaves → partition at lower row counts;
    leaves at/above the occupancy target keep `P=1`). Keep the existing P
    computation `clamp(ROWPART_TARGET_CUBES / num_features, 1, ROWPART_P_MAX)` and
    the P_MAX=16 clamp (P=32 over-partitions/regresses — spike-007, do not exceed
    P_MAX). PRESERVE the `LGBM_ROWPART_MIN` env override path for future benching.

    Update the unit test `row_partition_count_heuristic` (`histogram.rs:2838`) so
    it still asserts: the ≤8k-row parity-test shapes stay `P=1` (critical — the
    parity tests must keep running the unchanged path so the hip envelope is not
    silently moved under them), degenerate `num_features==0` and
    already-saturated (`>= ROWPART_TARGET_CUBES`) stay `P=1`, and add a cell
    asserting the new medium-width-at-lower-rows behaviour matches Task 1's
    crossover.

    Default-OFF / env-gated posture: if Task 3 later finds the new gate breaks the
    ~1e-6 hip envelope, the gate must be revertible to the current 256k behaviour
    by default with the new behaviour reachable only via env override. Structure
    the change so that fallback is a one-line constant flip, not a rewrite.

    Re-measure end-to-end GPU train-wall on one medium-width workload (reuse the
    bench harness) to confirm the gate change reproduces the Task 1 win at the
    learner level, not just the isolated build.
  </action>
  <verify>
    <automated>cargo test -p lgbm-compute --features rocm row_partition_count_heuristic 2>&1 | tail -5</automated>
  </verify>
  <done>
    `row_partition_count` is feature-count-aware with the crossover derived from
    Task 1; ≤8k-row shapes still return P=1; P never exceeds ROWPART_P_MAX;
    LGBM_ROWPART_MIN override preserved; unit test green; end-to-end GPU
    train-wall reproduces the Task 1 medium-width win. (SKIPPED + documented if
    Task 1 was NULL.)
  </done>
</task>

<task type="checkpoint:human-verify" gate="blocking-human">
  <what-built>
    The feature-count-aware row-partition gate (Task 2), OR a documented NULL
    (Task 1). This checkpoint is the HARD hip-parity gate AND the
    anchor-untouched proof. The ~1e-6 hip envelope is the ship gate here (NOT
    bit-exact — this is the RocmBackend f32 path); the cpu f64 anchor is a
    separate, untouched gate.
  </what-built>
  <how-to-verify>
    1. HIP envelope (the f32 ship gate) — on gfx1100 with --features rocm, run
       the rocm parity cells and confirm divergence stays within the documented
       ~1e-6 hip envelope vs the cpu f64 anchor:
         cargo test -p lgbm-compute --features rocm --test rocm_row_partition
         cargo test -p lgbm-compute --features rocm --test rocm_backend_parity
         cargo test -p lgbm-compute --features rocm --test rocm_parallel_histogram
         cargo test -p lgbm-compute --features rocm --test rocm_cuda_mirror
       Expect: all green; any divergence reported within the ~1e-6 hip envelope
       (cross-check against 04-ROCM-GAPS.md per-phase residuals). Reminder
       (def-f8u-01): never compare two nondeterministic GPU f32 paths to each
       other at 1e-6 — compare to the cpu f64 anchor.

    2. cpu f64 anchor UNTOUCHED (bit-exact, separate gate) — confirm no
       regression on the deterministic anchor:
         cargo test -p oracle-harness --test kernel_parity
         cargo test -p oracle-harness --test learner_parity
       Expect: bit-exact green (these are the ≤8k-row shapes that must still run
       the P=1 path).

    3. Build clean both ways:
         cargo check
         cargo check --features rocm
         cargo build --release --features rocm
       Expect: clean with AND without --features rocm.

    4. Verdict:
       - WIN: Task 1 showed a sign-stable medium-width gfx1100 speedup AND the
         new gate keeps the ~1e-6 hip envelope AND the anchor is untouched → ship
         the feature-count-aware gate ON by default.
       - TRADEOFF/NULL-B: row-partitioning wins on speed but the new gate breaks
         the ~1e-6 hip envelope → REVERT the gate to the current 256k default,
         leave the new behaviour env-gated/OFF, document the
         speed-vs-divergence tradeoff (still a valid finding).
       - NULL-A: Task 1 showed no medium-width win (atomic/latency-bound) → no
         source change shipped; document the NULL with the numbers.
       Do NOT ship a parity-envelope violation as default. Do NOT manufacture a
       win.
  </how-to-verify>
  <resume-signal>
    Type "approved: WIN", "approved: tradeoff/env-gated", or "approved: NULL"
    (with which verdict and the measured hip divergence), or describe issues.
  </resume-signal>
</task>

</tasks>

<verification>
- gfx1100 A/B sweep complete with ≥3 restarts and sign-stability noted (Task 1).
- On a win: feature-count-aware gate implemented, ≤8k-row shapes still P=1,
  P ≤ ROWPART_P_MAX, LGBM_ROWPART_MIN preserved, unit test green (Task 2).
- rocm parity cells green within the ~1e-6 hip envelope; cpu kernel_parity +
  learner_parity bit-exact green; `cargo check` clean with and without
  `--features rocm` (Task 3 checkpoint).
</verification>

<success_criteria>
SHIP ONLY on a real, sign-stable medium-width gfx1100 win that stays within the
~1e-6 hip envelope AND leaves the cpu f64 bit-exact anchor untouched. Otherwise
revert (env-gate the change OFF) and document the tradeoff/NULL. A documented
NULL or a documented env-gated tradeoff is a fully successful outcome.
</success_criteria>

<output>
Create `.planning/quick/260620-sqf-optimize-the-gpu-hip-f32-run-for-medium-/260620-sqf-SUMMARY.md`
when done, recording the A/B numbers, the verdict (WIN / tradeoff-env-gated /
NULL), the measured hip divergence at the new gate, and any 04-ROCM-GAPS.md
residual update.
</output>
