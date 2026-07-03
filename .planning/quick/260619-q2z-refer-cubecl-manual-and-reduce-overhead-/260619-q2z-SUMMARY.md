---
status: complete
phase: quick-260619-q2z
plan: 01
subsystem: infra
tags: [cubecl, rocm, gfx1100, histogram, lazy-execution, deferred-sync, benchmark, gpu]

requires:
  - phase: quick-260619-ol8
    provides: launch_unchecked A/B harness pattern (immediate read_one as the single-launch sync)
  - phase: quick-260619-p93
    provides: plane_aggregate_ab harness pattern + the 256-bin contention-bound characterization
provides:
  - "rocm-gated A/B example isolating the cubecl lazy-execution lever (manual ch.05): N immediate-sync per-feature launches (arm A) vs N submitted launches + one deferred drain (arm B)"
  - "Measurement-disposition: WIRE (follow-up) gated to the compute-bound, bins>=256, multi-feature regime (~20-26% spread-separated, sign-stable); launch-bound = NULL"
affects: [gpu-histogram-routing, rocm-perf-followup]

tech-stack:
  added: []
  patterns:
    - "Deferred-sync (lazy execution) leaf-loop dispatch: submit all per-feature launches into distinct out-handles before draining, the manual ch.05 host-device-overlap pattern"

key-files:
  created:
    - crates/lgbm-compute/examples/lazy_dispatch_ab.rs
    - .planning/quick/260619-q2z-refer-cubecl-manual-and-reduce-overhead-/260619-q2z-FINDINGS.md
  modified: []

key-decisions:
  - "DISPOSITION = WIRE (follow-up plan), gated to compute-bound x bins>=256 x feats>=32 where deferred-sync is ~20-26% spread-separated and sign-stable across 3 restarts; launch-bound regime is NULL/negative"
  - "Arm B is a harness call-ORDERING change, not a new kernel — wiring is a follow-up that refactors the per-feature leaf histogram loop, OUT OF SCOPE for this measurement-only spike"
  - "No twin kernel needed — both arms launch the SHIPPED construct_hist_kernel_atomic_f32; the same-input assert is a deferred-sync-changes-nothing guard, not a body-drift guard"

patterns-established:
  - "Lazy-execution A/B: hold transfer equal (resident upload once outside timed arms), force the deferred drain inside the timed region, sweep launches-per-leaf (feats) as the primary axis"

requirements-completed: []

duration: ~35min
completed: 2026-06-19
---

# Phase quick-260619-q2z Plan 01: Lazy-execution (deferred-sync) per-feature histogram A/B Summary

**Benched the cubecl lazy-execution lever (manual ch.05) on gfx1100 — deferring the per-feature `read_one` sync across a leaf's launches is a robust ~20-26% spread-separated win in the compute-bound, bins>=256, multi-feature regime (NULL/negative in launch-bound); DISPOSITION = WIRE as a regime-gated follow-up.**

## Performance

- **Duration:** ~35 min
- **Tasks:** 2
- **Files created:** 2 (1 source example committed; 1 FINDINGS doc handled by orchestrator)

## Accomplishments

- New rocm-gated A/B example `lazy_dispatch_ab.rs` isolates the ONE un-measured-in-isolation GPU overhead mechanism: cubecl lazy execution. Arm A = N per-feature atomic-histogram launches each with an IMMEDIATE blocking `read_one_unchecked`; arm B = N launches submitted into N distinct out-handles back-to-back, then ONE deferred drain phase. Arms differ ONLY in sync timing; the device-resident bin matrix + shared grad/hess are uploaded once per cell (transfer held equal).
- Swept FEATURE-COUNT [8, 32, 128] (the launches/leaf knob) AND bin-count [16, 64, 256] across launch-bound (1024-row leaf) and compute-bound (200k-row leaf) regimes; interleaved arms, WARMUP discard, median + p25/p75, same-input A-vs-B correctness assert on every cell.
- Ran on the real gfx1100 across 3 process restarts. Finding: **launch-bound = NULL/negative (sign-flips or arm A faster); compute-bound = robustly positive and sign-stable, strongest at 256 bins (feats=32: +19-24%, feats=128: +20-26%, both spread-SEPARATED in all 3 runs).** The lever recovers the inter-launch host-round-trip bubble where the GPU has enough per-kernel work to overlap.
- CPU-only build compiles (stub main, zero rocm codegen); regression gates GREEN.

## Task Commits

1. **Task 1: Write the rocm-gated lazy-dispatch A/B example** - `9fd4bcd` (feat)
2. **Task 2: Run the A/B (3 restarts), regression-gate, write FINDINGS.md** - FINDINGS.md authored (docs commit handled by orchestrator)

## Files Created/Modified

- `crates/lgbm-compute/examples/lazy_dispatch_ab.rs` - rocm-gated interleaved A/B of the immediate-sync vs deferred-sync per-feature histogram dispatch pattern (committed `9fd4bcd`)
- `.planning/quick/260619-q2z-.../260619-q2z-FINDINGS.md` - honest 3-run measurement + DISPOSITION verdict (orchestrator commits docs)

## Decisions Made

- **DISPOSITION = WIRE (follow-up plan), regime-gated.** Unlike c2l/ol8/p93 (flat NULLs), this lever shows a sign-stable, spread-separated ~20-26% win in compute-bound x bins>=256 x feats>=32. The gating regime (large leaf, bins>=64, multi-feature) and the launch-bound NULL are documented in FINDINGS.md. Reconciliation: c2l/ol8 measured the single-launch / launch-bound axis (NULL there too); this is the first measurement of the multi-launch-per-leaf x large-leaf axis where deferring the sync surfaces a real overlap win.
- **No twin kernel.** Both arms launch the SHIPPED `construct_hist_kernel_atomic_f32`; the same-input assert is purely a "deferring the sync changes nothing numerically" guard (it passed on every cell).
- **Wiring is explicitly OUT OF SCOPE for this spike** — arm B is a call-ordering pattern; production realization requires refactoring the per-feature leaf loop + end-to-end re-validation of the CPU f64 anchor and the rocm ~1e-6 gate, a follow-up plan.

## Deviations from Plan

None - plan executed exactly as written. The plan front-loaded an "expected NULL" framing; the honest measurement instead found a regime-gated WIN, which is reported as such (no win manufactured — launch-bound is honestly NULL/negative, the win is confined to and gated by the compute-bound high-bin regime, spread-separated and sign-stable across 3 restarts).

## Issues Encountered

None. Both builds compiled first try; all 3 A/B runs and the same-input asserts ran clean on the real gfx1100.

## Regression Gate

- `cargo build -p lgbm-compute --example lazy_dispatch_ab` (CPU-only) GREEN; `--features rocm` GREEN.
- Default `cargo test -p lgbm-compute` lib: **30 passed / 1 ignored** (pre-existing baseline, unchanged).
- Bit-exact CPU anchor gate `cargo test -p oracle-harness -p lgbm-treelearner --lib -p lgbm`: **lgbm 41/41, lgbm-treelearner --lib 76 passed / 2 ignored, oracle-harness 3/3 — 0 failed, GREEN.**
- No production kernel/launcher/anchor modified (pure addition of one example).

## Next Phase Readiness

- The lever is measured and dispositioned. A follow-up plan (if pursued) should wire the deferred-drain into the per-feature leaf histogram loop, gate it on the regime (large-leaf x bins>=64 x multi-feature), keep the immediate path for launch-bound small leaves, and re-validate parity end-to-end. ROI caveat (gpu-histogram-kernel skill): with the CPU anchor multi-threaded the GPU loses at every tested size — this is ROCm-parity-track maintenance, weigh before committing the follow-up.

## Self-Check: PASSED

- `crates/lgbm-compute/examples/lazy_dispatch_ab.rs` — FOUND (committed `9fd4bcd`)
- `.planning/quick/260619-q2z-refer-cubecl-manual-and-reduce-overhead-/260619-q2z-FINDINGS.md` — FOUND (contains DISPOSITION)
- Commit `9fd4bcd` — FOUND in git log

---
*Phase: quick-260619-q2z*
*Completed: 2026-06-19*
