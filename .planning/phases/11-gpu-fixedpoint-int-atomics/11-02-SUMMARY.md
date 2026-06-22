---
phase: 11-gpu-fixedpoint-int-atomics
plan: 02
subsystem: testing
tags: [rocm, gpu, histogram, fixed-point, u64, parity, determinism, def-f8u-01, hip]

# Dependency graph
requires:
  - phase: 11-gpu-fixedpoint-int-atomics (plan 01)
    provides: "u64 two's-complement fixed-point resident LDS build (build_fix_compact_resident_readback_f64_on now dispatches the u64 kernel via resident_raw_build_into; fix_compact_kernel dequantizes (bits as i64)/2^30 -> f64)"
provides:
  - "kernel_parity_resident_build_fix_compact_equals_host_on_hip re-anchored to a CPU f64 anchor (construct_histograms_cpu + host fix/compact) instead of a second GPU launcher (def-f8u-01)"
  - "Tightened fixed-point parity gate FIXEDPOINT_REL_GATE=1e-7 (spike-018 ref ~5.9e-9; measured max_rel=0.0 on this leaf) replacing the generous f32 HIP_SANITY_REL=1e-3 envelope for the resident u64 path"
  - "2-runs-bit-equal determinism assert (to_bits compare) proving the u64 resident build is deterministic by construction"
  - "Confirmation the UNCHANGED non-resident f32 + bit-exact f64 subtract/construct pins stay green at their existing tolerances"
affects: [11-03, rocm-parity, gpu-histogram]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Re-anchor a GPU parity test to the CPU f64 fold (construct_histograms_cpu + host fix/compact) when both arms collapse to GPU launchers (def-f8u-01: never GPU-vs-GPU for nondeterministic paths)"
    - "Tightened fixed-point gate (FIXEDPOINT_REL_GATE) distinct from the f32 envelope (HIP_SANITY_REL): a u64 fixed-point path is held to quantize-rounding error, not f32-cancellation error"
    - "Determinism sub-assert via to_bits bit-equality across two runs for integer-atomic GPU paths"

key-files:
  created:
    - .planning/phases/11-gpu-fixedpoint-int-atomics/deferred-items.md
  modified:
    - crates/oracle-harness/tests/kernel_parity.rs
    - crates/lgbm-compute/tests/rocm_row_partition.rs

key-decisions:
  - "FIXEDPOINT_REL_GATE = 1e-7: above the spike-018 hardware worst case (~5.9e-9) with margin, far below the f32 envelope (1e-3); measured max_rel=0.0 on the test leaf (exact in the cancelling integer regime)"
  - "Do NOT reuse assert_within (HIP_SANITY_REL=1e-3) for the fixed-point path — that envelope is calibrated for the f32 launchers and would let a fixed-point regression slip"
  - "cuda_mirror full-corpus flake is a PRE-EXISTING f32-atomic nondeterminism in a DIFFERENT kernel untouched by phase 11 (DEF-11-OOS-01), out of scope for confirmation-only 11-02"

patterns-established:
  - "GPU fixed-point parity pin = (CPU f64 anchor) + (tightened rel gate) + (2-runs determinism); never a second GPU launcher as the reference"

requirements-completed: [SPEC-2]

# Metrics
duration: ~20min
completed: 2026-06-22
---

# Phase 11 Plan 02: Re-pin Resident u64 Fixed-Point Build Parity to the CPU f64 Anchor Summary

**The resident u64 fixed-point histogram build (the path Plan 11-01 changed) is now parity-pinned to a fresh CPU f64 anchor at a tightened 1e-7 gate (measured max_rel=0.0, spike-018 ref ~5.9e-9) with a 2-runs bit-equal determinism assert — replacing a GPU-vs-GPU comparison that violated def-f8u-01 — while the unchanged non-resident f32 and bit-exact f64 subtract/construct pins are confirmed green at their existing tolerances.**

## Performance

- **Duration:** ~20 min
- **Started:** 2026-06-22
- **Completed:** 2026-06-22
- **Tasks:** 2
- **Files modified:** 2 (+1 created)

## Accomplishments
- Re-anchored `kernel_parity_resident_build_fix_compact_equals_host_on_hip`: its reference is now the bit-exact CPU f64 fold `construct_histograms_cpu` (per feature, over the leaf's gathered `(bin, grad, hess)`) plus the SAME host `fix_histogram` + `host_compact_histogram` the GPU chain runs — NOT the second GPU launcher `build_leaf_histograms_resident_f32_on` (which, after Plan 01, also became u64-on-GPU ⇒ a def-f8u-01-violating GPU-vs-GPU comparison).
- Tightened the gate from the generous f32 envelope (`HIP_SANITY_REL = 1e-3`, via `assert_within`) to a dedicated `FIXEDPOINT_REL_GATE = 1e-7`. Measured `max_rel_vs_cpu_f64_anchor = 0.0` on the test leaf (exact in the cancelling integer regime, consistent with spike-018a).
- Added a determinism sub-assert: two `build_fix_compact_resident_readback_f64_on` runs over identical inputs are bit-equal (`to_bits`) on every read-back f64 cell.
- Confirmed the UNCHANGED paths green: `rocm_row_partition` 2/2 (non-resident batched f32, existing 1e-5/5e-5 tolerance, NOT changed) and `rocm_backend_parity` 4/4 (bit-exact subtract/f64 + non-resident construct pins intact).

## Task Commits

Each task was committed atomically:

1. **Task 1: Re-pin the resident u64 fixed-point build parity to a CPU f64 anchor (tightened) + determinism assert** - `9b43f0e` (test)
2. **Task 2: Confirm unchanged f32/bit-exact pins green; log pre-existing cuda_mirror flake** - `8eecf7b` (test)

## Files Created/Modified
- `crates/oracle-harness/tests/kernel_parity.rs` - `kernel_parity_resident_build_fix_compact_equals_host_on_hip` re-anchored to the CPU f64 anchor; `FIXEDPOINT_REL_GATE=1e-7` replaces the `assert_within` (f32-envelope) gate for this test; determinism sub-assert added; doc comment rewritten for the fixed-point rationale; dropped the now-unused `build_leaf_histograms_resident_f32_on` import.
- `crates/lgbm-compute/tests/rocm_row_partition.rs` - one-line phase-11 classification comment (non-resident f32 path, unchanged); no numeric change.
- `.planning/phases/11-gpu-fixedpoint-int-atomics/deferred-items.md` - new; logs DEF-11-OOS-01 (pre-existing cuda_mirror flake).

## Tightened tolerance constant chosen + measured rel-to-anchor

| Item | Value |
|------|-------|
| Re-pinned test | `kernel_parity_resident_build_fix_compact_equals_host_on_hip` (name unchanged) |
| Old gate | `assert_within` → `HIP_SANITY_REL = 1e-3` (generous f32 envelope) |
| New gate | `FIXEDPOINT_REL_GATE = 1e-7` (rel = `|gpu - anchor| / max(|anchor|, 1.0)`) |
| Spike-018 hardware reference | ~5.9e-9 rel (exact 0.0 in the cancelling regime) |
| **Measured on this leaf** | **max_rel = 0.0** (printed to stderr each run) |
| Determinism assert location | same test, after the gate loop — `gpu.to_bits() == gpu_again.to_bits()` over all cells |
| Reference arm | CPU f64 anchor (`construct_histograms_cpu` + host `fix_histogram`/`host_compact_histogram`), NOT a second GPU launcher |

The 1e-7 gate sits ~17× above the spike-measured hardware worst case (~5.9e-9) and 4 orders of magnitude below the f32 envelope (1e-3), so a real fixed-point regression cannot slip while the bounded quantize-rounding residual passes.

## Decisions Made
- **1e-7 gate, not exact-0:** although measured max_rel was 0.0 on this small integer-valued leaf, the gate is set at 1e-7 (above the spike-018 hardware worst case ~5.9e-9 with margin) so the test stays green on larger/non-cancelling leaves where bounded quantize-rounding error is non-zero, while still being 4 orders tighter than the f32 envelope.
- **Kept the existing test-fn name** (`kernel_parity_resident_build_fix_compact_equals_host_on_hip`) — only its reference arm + gate + determinism changed.
- **No edit to `rocm_backend_parity.rs`** — its bit-exact pins operate on post-dequant f64 cells (subtract/construct), confined away from the u64 RAW seam, so they stay bit-exact unchanged; confirmation-only.

## Deviations from Plan

None requiring auto-fix. The plan's Task 2 anticipated exactly the out-of-scope possibility encountered and instructed to log it: the `rocm_cuda_mirror` full-corpus test surfaced a flaky failure that is **pre-existing and out of scope** (logged, not fixed) — see DEF-11-OOS-01 below.

## Issues Encountered

**DEF-11-OOS-01 — `cuda_mirror_full_corpus_leaf_matches_anchor` flaky (pre-existing, out of scope):**
During Task 2's confirmation run, this test intermittently failed the full-corpus large-leaf assert (e.g. cell 50 `anchor 0.12500596 vs gpu 0.12501263`, |diff| 6.68e-6 > tol 6.25e-6). Across 4 consecutive runs: FAILED, FAILED, ok, ok — **flaky**.

- **Root cause:** it drives `construct_hist_cuda_mirror_kernel` (the CUDA-faithful f32 mirror), which accumulates via f32 atomicAdd CAS-retry — order-dependent / nondeterministic. On the largest leaf the f32-accumulation envelope occasionally overshoots its `ABS 5e-6 / REL 1e-5` gate by a few percent.
- **Pre-existing / out of scope:** this is a DIFFERENT kernel from the resident u64 build Phase 11 changed. Verified `git log 434efb3..HEAD -- crates/lgbm-compute/src/kernels/histogram.rs` — the phase-11 commits (`6ec996e`, `cc3b040`, `c95518d`) touch ONLY the resident LDS build + `fix_compact_kernel`, none touch the cuda_mirror kernel; the test file `rocm_cuda_mirror.rs` has no phase-11 commits. The other three cuda_mirror tests pass reliably.
- **Disposition:** logged to `deferred-items.md` (DEF-11-OOS-01), NOT fixed (SCOPE BOUNDARY — confirmation-only plan, pre-existing failure in an unrelated kernel). Fix direction = port the cuda_mirror kernel to the same u64 fixed-point atomics this phase shipped, or relax its full-corpus assert to the documented f32-atomic envelope.

## Unchanged-path confirmation

| Suite | Result | Classification |
|-------|--------|----------------|
| `rocm_row_partition` | 2/2 PASS | Non-resident batched f32 (`build_leaf_histograms_batched_f32_on`) — unchanged; existing 1e-5/5e-5 tolerance, NOT tightened/relaxed |
| `rocm_backend_parity` | 4/4 PASS | Bit-exact `assert_bit_exact` subtract/f64 + non-resident construct — bit-exact, NOT relaxed; no edit |
| `rocm_cuda_mirror` | 3/4 (1 flaky) | Different kernel (`construct_hist_cuda_mirror_kernel`), unchanged; flake = DEF-11-OOS-01 (pre-existing) |
| `kernel_parity_resident_build_fix_compact_equals_host_on_hip` | PASS | RE-PINNED (Task 1): CPU f64 anchor, 1e-7 gate, determinism |

No GPU-vs-GPU comparison remains for the changed fixed-point path; no bit-exact pin relaxed; no fixed-point pin added outside Task 1.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- The resident u64 fixed-point build is now anchor-pinned (CPU f64) + determinism-asserted within a tightened gate — SPEC item 2 satisfied. Plan 11-03 can build on the validated, deterministic resident path.
- Concern (deferred, not a blocker): DEF-11-OOS-01 — the `cuda_mirror` full-corpus test is flaky due to pre-existing f32-atomic nondeterminism in a separate kernel. A follow-up should either port that kernel to u64 fixed-point or relax its large-leaf tolerance.

## Self-Check: PASSED

- Files: `crates/oracle-harness/tests/kernel_parity.rs`, `crates/lgbm-compute/tests/rocm_row_partition.rs`, `deferred-items.md`, `11-02-SUMMARY.md` all FOUND.
- Commits: `9b43f0e`, `8eecf7b` all FOUND.

---
*Phase: 11-gpu-fixedpoint-int-atomics*
*Completed: 2026-06-22*
