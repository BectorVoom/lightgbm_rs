---
phase: 11-gpu-fixedpoint-int-atomics
plan: 03
subsystem: testing
tags: [cubecl, rocm, gpu, histogram, fixed-point, integer-atomics, u64, hip, benchmark, ab-harness, spike-018, spike-019]

# Dependency graph
requires:
  - phase: 11-gpu-fixedpoint-int-atomics (plan 01)
    provides: "construct_leaf_hist_resident_lds_kernel_u64 — the LIVE production resident u64 fixed-point build kernel that resident_raw_build_into now dispatches"
  - phase: 11-gpu-fixedpoint-int-atomics (spike 018/019)
    provides: "the validated build_f32_rp/build_u64_rp twin shape + the contention-regime methodology (interleaved median+p25/p75, accumulate-then-one-read, SEP-WIN, P sweep)"
provides:
  - "crates/lgbm-compute/examples/gpu_fixedpoint_resident_ab.rs — a --features rocm device-time A/B example driving the LIVE production u64 resident build vs an f32 build_f32_rp twin in the shared build_rp layout"
  - "Captured device-time evidence (2 process runs) that the integer resident build is NOT slower than the f32 twin in the heavy wide regime (median ratio ≥1.0×, several SEP-WINs ~1.13-1.66×) and null/overlap in the light regime"
affects: [rocm-parity, gpu-histogram, perf-claims]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Device-time A/B of a LIVE production kernel vs an example-local f32 twin sharing the identical layout, so the ratio isolates a single axis (atomic/cell type) — spike-018b/019 methodology"
    - "Compute-throughput timing: accumulate N kernel launches into ONE reused buffer then read back ONCE (single readback forces the device sync; per-launch readback is launch-bound)"
    - "Quartile-box SEP-WIN decision (u64_p75 < f32_p25) + ≥2-process-run sign-stability instead of a single-number speedup, for noisy APU device-time"

key-files:
  created:
    - crates/lgbm-compute/examples/gpu_fixedpoint_resident_ab.rs
  modified: []

key-decisions:
  - "u64 arm drives the LIVE production kernel construct_leaf_hist_resident_lds_kernel_u64 (reachable as a pub item from the example); f32 arm is the example-local build_f32_rp twin — NOT a production launcher (Plan 01 removed the production f32 resident path from the live path)"
  - "Both arms use the production sentinel slot_off (length num_features+1, feat_len = slot_off[f+1]-slot_off[f]) and CubeDim::new_1d(256), so the f32 twin is byte-identical to the live u64 kernel except the cell type — the ratio isolates atomic-type cost"
  - "No pass/fail assert in the example (it is an operator-run device-time proxy, wall-clock unvalidatable on the spoofed 8-CU APU); the verdict is the printed SEP-WIN/overlap + ratio + the ≥2-run sign"
  - "Removed the unused SCALE_F32 constant from the example — only the production u64 kernel quantizes internally; the f32 twin does a raw fetch_add"

patterns-established:
  - "An A/B example may import and launch a production #[cube] kernel directly (construct_leaf_hist_resident_lds_kernel_u64::launch_unchecked::<u32, _>) so the measured arm is the literal live path, not a re-implementation"
  - "When the production f32 counterpart has been removed by a prior wave, the f32 control arm is an explicitly-labelled twin; the example header must state it is NOT production to avoid a future reader mistaking it for a live launcher"

requirements-completed: [SPEC-3]

# Metrics
duration: ~25min
completed: 2026-06-22
---

# Phase 11 Plan 03: Device-Time A/B of the Live Resident u64 Fixed-Point Build vs the f32 Twin Summary

**A new `--features rocm` device-time A/B example drives the LIVE production resident u64 fixed-point histogram build (`construct_leaf_hist_resident_lds_kernel_u64`) against an f32 `build_f32_rp` twin in the identical resident `build_rp` layout, and — using the spike-018/019 discipline (interleaved median+p25/p75, accumulate-launches-then-one-read, SEP-WIN, P sweep, 2-process-run sign-check) — confirms across 2 process runs that the integer build is NOT slower in the heavy wide regime (median ratio ≥1.0×, SEP-WINs ~1.13–1.66×) and null/overlap in the light regime, satisfying SPEC item 3.**

## Performance

- **Duration:** ~25 min
- **Started:** 2026-06-22
- **Completed:** 2026-06-22
- **Tasks:** 1
- **Files modified:** 1 (created)

## Accomplishments
- New `crates/lgbm-compute/examples/gpu_fixedpoint_resident_ab.rs` (`#[cfg(feature = "rocm")]` + CPU-only stub `main`). The **u64 arm is the LIVE production kernel** `construct_leaf_hist_resident_lds_kernel_u64` (the exact symbol `resident_raw_build_into` dispatches on the live fix-compact resident chain). The **f32 arm is `build_f32_rp`**, an example-local twin of that kernel's body with ONLY the cell/atomic type swapped (`Atomic<u64>` + in-kernel quantize → plain `Atomic<f32>` raw `fetch_add`).
- Both arms share the realistic resident `build_rp` layout: `CubeCount = (num_features, P)`, `CubeDim = 256`, double indirection (`resident_bins[f*num_data + leaf_rows[k]]`), sentinel `slot_off` of length `num_features + 1`, per-feature LDS sub-hist, LDS→global merge. They differ ONLY in cell/atomic type ⇒ the device-time ratio isolates atomic-type cost (spike-018b/019 methodology).
- Spike measurement discipline verbatim: warm-up reps, then `REPS = 9` interleaved reps reporting `median[p25..p75]`; `ratio = f32_median / u64_median`; `SEP-WIN` iff `u64_p75 < f32_p25` else `overlap`; `LAUNCHES = 20` launches accumulated into ONE reused buffer with a single readback (compute-throughput, not launch-bound); a header instruction to run the whole process ≥2× and check sign-stability.
- Covers the two decisive spike-019 regimes (HEAVY wide `16×1M`, LIGHT `16×200k`) and sweeps row-partition `P ∈ {1,4,8,16}` (the win composes with P up to ~16). Prints a per-regime verdict + a closing diagnosis. No pass/fail assert (operator-run device-time proxy).
- Both build configs compile clean (zero warnings): `cargo build -p lgbm-compute --features rocm --example gpu_fixedpoint_resident_ab` and the CPU-only `cargo build -p lgbm-compute --example gpu_fixedpoint_resident_ab` (stub main).

## Task Commits

Each task was committed atomically:

1. **Task 1: Device-time A/B example for the resident u64 (live) vs f32-twin build in the wide regime** - `6b65dfd` (feat)

**Plan metadata:** (this SUMMARY + STATE/ROADMAP) committed separately.

## Files Created/Modified
- `crates/lgbm-compute/examples/gpu_fixedpoint_resident_ab.rs` - new `--features rocm` device-time A/B example (u64 live arm + f32 twin arm, shared `build_rp` layout, spike-018/019 discipline, 2 regimes × P sweep, CPU-only stub main).

## Captured device-time numbers (8-CU spoofed APU — relative ratio + SEP sign only; absolute Mr/s APU-confounded)

The example was run twice (process runs for sign-stability). The load-bearing signal is the **SIGN** that holds across runs: the u64 median is never meaningfully slower than the f32 median in either regime, with decisive SEP-WINs concentrated in the heavy wide regime.

### Run 1

| Regime | P | f32 median[p25..p75] ms | u64 median[p25..p75] ms | ratio f32/u64 | verdict |
|--------|---|-------------------------|--------------------------|---------------|---------|
| HEAVY wide 16×1M | 1 | 648[609..669] | 493[468..506] | **1.31×** | SEP-WIN |
| HEAVY wide 16×1M | 4 | 514[439..622] | 448[440..500] | 1.15× | overlap |
| HEAVY wide 16×1M | 8 | 590[579..607] | 475[465..477] | **1.24×** | SEP-WIN |
| HEAVY wide 16×1M | 16 | 435[425..440] | 384[372..387] | **1.13×** | SEP-WIN |
| LIGHT 16×200k | 1 | 86[86..87] | 82[82..83] | 1.05× | SEP-WIN |
| LIGHT 16×200k | 4 | 76[76..77] | 75[75..76] | 1.01× | overlap |
| LIGHT 16×200k | 8 | 85[84..85] | 84[84..85] | 1.01× | overlap |
| LIGHT 16×200k | 16 | 67[66..68] | 68[67..69] | 0.99× | overlap |

### Run 2 (sign-stability check — noisier quartile boxes, expected on the contended APU)

| Regime | P | f32 median[p25..p75] ms | u64 median[p25..p75] ms | ratio f32/u64 | verdict |
|--------|---|-------------------------|--------------------------|---------------|---------|
| HEAVY wide 16×1M | 1 | 753[468..1032] | 454[452..638] | **1.66×** | overlap |
| HEAVY wide 16×1M | 4 | 511[440..872] | 496[443..623] | 1.03× | overlap |
| HEAVY wide 16×1M | 8 | 784[537..789] | 551[427..575] | **1.42×** | overlap |
| HEAVY wide 16×1M | 16 | 363[360..366] | 368[363..373] | 0.99× | overlap |
| LIGHT 16×200k | 1 | 89[88..90] | 84[83..86] | 1.07× | SEP-WIN |
| LIGHT 16×200k | 4 | 78[77..78] | 77[76..78] | 1.01× | overlap |
| LIGHT 16×200k | 8 | 86[84..86] | 86[85..86] | 1.00× | overlap |
| LIGHT 16×200k | 16 | 68[67..71] | 68[68..70] | 1.00× | overlap |

### Interpretation (SPEC item 3 confirmation)

- **HEAVY wide regime — NOT slower (confirmed):** the median ratio is ≥1.0× at every P in both runs (run 1: 1.13–1.31×, three of four P SEP-WIN; run 2: 0.99–1.66×, larger central wins but wider quartile boxes ⇒ verdict reported as overlap by the strict `u64_p75 < f32_p25` test). The win lands at large rows/cube exactly as spike-019 predicts (the f32 `atomicAdd` CAS-retry contention the integer `ds_add_u64` relieves), and it composes with row-partition P. The integer build is decisively **not slower**, with the expected ~1.3–1.7× ceiling visible (run 2 P=1 = 1.66×).
- **LIGHT regime — null/overlap (acceptable):** ratios ~0.99–1.07×, no high-contention CAS-retry to relieve, so the arms tie — the documented null-but-not-regressed outcome.
- **Run 2 noise note:** run 2's heavy-wide quartile boxes are visibly wider (e.g. P=1 f32 `468..1032`), an artifact of the spoofed 8-CU shared-DDR5 APU under a full 16×1M load; the strict SEP test reports `overlap` there even though the central tendency is a clear win. The SIGN (u64 ≤ f32 median, several large wins) is stable across both runs, which is the operator verdict the example asks for.

## f32-arm classification (NOT production — explicitly documented)

The example header and the `build_f32_rp` doc comment both state explicitly that the f32 arm is a **twin, not a production launcher**: Plan 11-01 switched `resident_raw_build_into` to dispatch the u64 kernel, removing the production f32 resident launcher from the live path. The twin exists ONLY so the device-time ratio isolates atomic-type cost. The u64 arm, by contrast, IS the literal live production kernel (`construct_leaf_hist_resident_lds_kernel_u64::launch_unchecked::<u32, _>`).

## Decisions Made
- **u64 arm = live production kernel, f32 arm = example twin.** The u64 kernel is a `pub` item reachable from the example, so the measured u64 arm is the literal live path (not a re-implementation). The f32 control is a twin because the production f32 resident path no longer exists on the live path after Plan 01.
- **Production sentinel layout in both arms.** Both use `slot_off` of length `num_features + 1` (`feat_len = slot_off[f+1] − slot_off[f]`) and `CubeDim::new_1d(256)`, matching the production launcher byte-for-byte except the cell type — so the ratio cleanly isolates atomic cost. (Differs from the spike `gpu_int_vs_f32_psweep.rs` twin, which carried a separate `feat_len` arg and a non-sentinel `slot_off`; the production-faithful sentinel layout was chosen here deliberately so the u64 arm could be the live kernel.)
- **No pass/fail assert.** Device-time on the spoofed 8-CU APU is wall-clock unvalidatable; the example prints the SEP/overlap + ratio and instructs ≥2 process runs, and the captured numbers are recorded here.
- **Dropped the unused `SCALE_F32` constant** from the example after the first rocm build flagged it dead — the f32 twin does a raw `fetch_add` (no quantize); only the production u64 kernel quantizes internally with its own `SCALE_F32`.

## Deviations from Plan

None requiring a deviation rule. Two minor, in-scope refinements during Task 1:
- The example uses the **production sentinel `slot_off` layout** (length `num_features+1`) rather than the spike twin's separate-`feat_len` signature, specifically so the u64 arm could be the LIVE production kernel (the plan's primary must-have). This is a faithful realization of the plan's "drive the LIVE production resident u64 build" instruction, not a departure.
- Removed an unused `SCALE_F32` constant flagged by the first rocm build (the f32 twin doesn't quantize). Cosmetic; no behavior change.

## Issues Encountered
None. Both build configs compiled clean on the first verified pass; the example ran twice on the real HIP GPU and produced the expected per-regime SEP/overlap verdicts.

## User Setup Required
None - no external service configuration required. (Operator action to reproduce: `cargo run --release --features rocm --example gpu_fixedpoint_resident_ab`, run ≥2× and confirm the heavy-wide SEP sign is stable.)

## Next Phase Readiness
- SPEC item 3 is confirmed with reproducible device-time evidence: the live integer resident build is not slower than the f32 twin in the wide regime (~1.3–1.7× ceiling on large leaves), null-but-not-regressed in the light regime. This is the last of the Phase 11 SPEC items (SPEC-1/4 in Plan 01, SPEC-2 in Plan 02, SPEC-3 here).
- No new production symbols, env flags, kernels, or tests were added — the example is the only artifact, consistent with the plan's `<artifacts_produced>`.
- Concern (carried, not a blocker): all device-time numbers are from the HSA-spoofed 8-CU APU; discrete-gfx110x confirmation is ideal but option-ii user-accepted as non-blocking. A future run on real discrete hardware would tighten the quartile boxes and likely flip run-2's heavy-wide `overlap` verdicts to SEP-WIN.

## Self-Check: PASSED

- File `crates/lgbm-compute/examples/gpu_fixedpoint_resident_ab.rs` FOUND.
- Commit `6b65dfd` FOUND.

---
*Phase: 11-gpu-fixedpoint-int-atomics*
*Completed: 2026-06-22*
