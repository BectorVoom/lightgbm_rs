---
phase: 12-gpu-sibling-scan-copack
plan: 02
subsystem: testing
tags: [cubecl, rocm, gpu, histogram-scan, sibling-copack, kernel-parity, bit-exact, def-f8u-01]

# Dependency graph
requires:
  - phase: 12-01-PLAN
    provides: find_best_splits_fused_siblings_from_handles_on (2-slot launcher), Backend::scan_resident_siblings, LGBM_SIBLING_COPACK gate (spine-equality fall-back)
  - phase: 260608-p90 / spike-021
    provides: find_best_splits_batched_fused_f64_on (single-slot host-buf scan), upload_f64_buffer, find_best_split_cpu_native (CPU f64 anchor), feature-per-lane W=64 / W=1 cubecl-cpu
provides:
  - kernel_parity_sibling_copack_equals_two_scans_on_hip (--features rocm: byte-identical + ~1e-6 CPU f64 anchor pin)
  - kernel_parity_sibling_copack_equals_two_scans_on_cpu (cubecl-cpu W=1 byte-identity, no rocm)
  - copack_feats / copack_cfg / copack_two_histograms (shared top-level test fixtures)
affects: [12-03-PLAN (device-time + e2e A/B + SCAN_RESIDENT_CNT — the perf half of SC-3/SC-4)]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Co-pack parity gate: (a) co-pack (co_a,co_b) == two single-slot scans via assert_eq! per SplitInfo (EXACT, same f64 kernel/same bits, only the launch differs); (b) each co-pack SplitInfo within hip ~1e-6 of the CPU f64 anchor (find_best_split_cpu_native), per sibling — def-f8u-01: anchor is CPU f64, NEVER a 2nd GPU path"
    - "W=1 byte-identity provable WITHOUT rocm: the same co-pack launcher on cubecl-cpu (W=1) is byte-identical to two single-slot scans — the bit-exact-by-construction claim on the always-available CPU runtime"
    - "Two-seed siblings: DIFFERENT leaf totals (smaller-built A vs larger-subtract-derived B) over the SAME shared feature layout (the spike-024 asymmetry)"

key-files:
  created: []
  modified:
    - crates/oracle-harness/tests/kernel_parity.rs

key-decisions:
  - "~1e-6 envelope constant = the established assert_within helper (ORACLE_TOL=1e-6 surfaced for the 04-ROCM-GAPS ledger + HIP_SANITY_REL=1e-3 hard-fail bound); REUSED, not relaxed. Measured: NO HIP PARITY GAP surfaced for any co-pack SplitInfo field => every field is within ORACLE_TOL=1e-6 of the CPU f64 anchor (tighter than the 1e-3 sanity bound), on both siblings"
  - "byte-identical half uses assert_eq! on the full SplitInfo (PartialEq-derived) — EXACT, valid because co-pack and single-slot run the SAME f64 kernel over the SAME input bits (co-packing only changes WHICH launch a feature scans in, per 12-CONTEXT 'bit-exact by construction')"
  - "anchor = find_best_split_cpu_native (CPU f64), per feature per sibling over [slot_off, slot_off+2*num_bin) of that sibling's buf — NEVER the GPU single-slot scan (def-f8u-01); guards a silent per-sibling mis-decode that a GPU-vs-GPU check would miss (T-12-04)"
  - "Task 2 confirmation-only (no code change): the CPU merge gate stayed green at existing tolerances; no golden relaxed (T-12-05)"

requirements-completed: [SC-1, SC-2]

# Metrics
duration: 25min
completed: 2026-06-25
status: complete
---

# Phase 12 Plan 02: gpu-sibling-scan-copack Summary

**Re-pins the resident-scan parity gate for Plan 01's co-pack path: a new `--features rocm` cell asserts the co-packed 2-slot sibling scan is BYTE-IDENTICAL (`assert_eq!`) to two separate single-slot scans per sibling AND within ~1e-6 of the CPU f64 anchor (def-f8u-01), plus a cubecl-cpu W=1 cell proving byte-identity without rocm — and confirms the CPU merge gate stays green with no golden relaxed.**

## Performance
- **Duration:** ~25 min
- **Completed:** 2026-06-25
- **Tasks:** 2
- **Files modified:** 1

## Accomplishments
- **`kernel_parity_sibling_copack_equals_two_scans_on_hip`** (`--features rocm`): drives `find_best_splits_fused_siblings_from_handles_on` on the real hip GPU over TWO concatenated f64 histograms (`buf_a` smaller / `buf_b` larger) with DIFFERENT leaf totals over the SAME 3-feature layout, and asserts:
  - **(a) BYTE-IDENTICAL:** `(co_a, co_b)` == two single-slot `find_best_splits_batched_fused_f64_on` scans per sibling, via `assert_eq!` on the full `SplitInfo` (the productionized spike-024 byte-for-byte gate).
  - **(b) ~1e-6 ANCHOR (def-f8u-01):** each co-pack `SplitInfo` (gain/threshold/counts/sums/outputs/default_left) within the hip ~1e-6 envelope of a CPU f64 anchor scan (`find_best_split_cpu_native`), per feature per sibling. The reference is the CPU f64 anchor, NEVER a second GPU path.
- **`kernel_parity_sibling_copack_equals_two_scans_on_cpu`** (NOT rocm-gated): runs the SAME co-pack launcher on the cubecl-cpu runtime (W=1) and asserts byte-identity to two single-slot scans — the bit-exact-by-construction proof on the always-available CPU runtime (no GPU needed for the bit-exact gate).
- **Shared fixtures** `copack_feats` / `copack_cfg` / `copack_two_histograms` (3 features covering a REVERSE winner, a FORWARD winner, and a no-split feature; two seed histograms with different leaf totals) — reused by both cells so they exercise the identical fixture.
- **CPU merge gate confirmed green** (Task 2, confirmation-only): `lgbm-treelearner --lib` (76), `lgbm-boosting --lib` (55), `raw_bin_train_matches_cpp_golden` (bit-exact vs lib_lightgbm 4.6), `learner_parity` (29) — all pass at existing tolerances, no golden relaxed.

## Task Commits
1. **Task 1: co-pack parity cells (byte-identical + ~1e-6 anchor + W=1)** — `25f7fdd` (test)
2. **Task 2: CPU merge gate confirmation** — no commit (confirmation-only; no code change, gate green on the committed state)

## Files Created/Modified
- `crates/oracle-harness/tests/kernel_parity.rs` — added two parity cells + three shared fixture helpers (327 insertions). The CPU cell sits at top level (uses `cpu_client()`); the rocm cell sits inside `mod hip` (`#[cfg(feature = "rocm")]`, uses `rocm_client()` + `super::copack_*` fixtures + the existing `assert_within`). No existing test, tolerance, or anchor helper touched.

## Wiring details (per the plan's output spec)
- **New cell names:** `kernel_parity_sibling_copack_equals_two_scans_on_hip` (rocm), `kernel_parity_sibling_copack_equals_two_scans_on_cpu` (cpu).
- **~1e-6 envelope constant + measured rel-to-anchor:** the envelope is the established `assert_within` helper — `ORACLE_TOL = 1e-6` (surfaced for the 04-ROCM-GAPS ledger on a miss, no silent pass) backed by `HIP_SANITY_REL = 1e-3` (the hard-fail bound that distinguishes the f32-accumulation gap from a real kernel bug). REUSED unchanged. **Measured on hardware:** NO `HIP PARITY GAP` line was surfaced for ANY co-pack SplitInfo field on either sibling — i.e. every field is within `ORACLE_TOL = 1e-6` of the CPU f64 anchor (well inside the 1e-3 sanity bound). The byte-identical half found ZERO field mismatches (`assert_eq!` passed on all features, both siblings).
- **byte-identical half uses `assert_eq!` (EXACT):** confirmed — `assert_eq!(co, si)` on the full `SplitInfo` per feature, both siblings (co-pack and single-slot run the same f64 kernel on the same input bits).
- **~1e-6 half pins to the CPU f64 anchor (def-f8u-01):** confirmed — `find_best_split_cpu_native` per feature per sibling is the reference; the GPU single-slot scan is used ONLY in the byte-identical EXACTNESS half, never as the ~1e-6 reference.
- **cubecl-cpu W=1 cell passes without rocm:** confirmed — `cargo test -p oracle-harness kernel_parity_sibling_copack_equals_two_scans_on_cpu` passes on the default build.
- **CPU merge gate stayed green at existing tolerances, no golden relaxed:** confirmed.

## Deviations from Plan
None — plan executed exactly as written. Both cells were added as specified (rocm byte-identical + ~1e-6 anchor, cpu W=1 byte-identity); Task 2 was confirmation-only and required no edits.

## Issues Encountered
None. The Plan-01 launcher signature (`a_totals` / `b_totals` tuples, `(Vec<SplitInfo>, Vec<SplitInfo>)` return) matched the plan's read-first exactly; the helper imports (`upload_f64_buffer`, `find_best_splits_batched_fused_f64_on`, `find_best_split_cpu_native`) and `SplitInfo: PartialEq` all resolved on first compile.

## Verification Results
- `cargo test -p oracle-harness --features rocm kernel_parity_sibling_copack` — 2 passed (`..._on_hip` byte-identical + ~1e-6 anchor; `..._on_cpu` W=1), 0 failed.
- `cargo test -p oracle-harness kernel_parity_sibling_copack_equals_two_scans_on_cpu` — 1 passed (W=1 byte-identity, no rocm).
- `cargo test -p lgbm-treelearner --lib` — 76 passed, 0 failed.
- `cargo test -p lgbm-boosting --lib` — 55 passed, 0 failed.
- `cargo test -p oracle-harness raw_bin_train_matches_cpp_golden` — passed (bit-exact vs lib_lightgbm 4.6).
- `cargo test -p oracle-harness learner_parity` — 29 passed, 0 failed.
- `cargo clippy -p oracle-harness --features rocm --tests` — 0 NEW warnings in the added cells (the 3 kernel_parity warnings at lines 754/1786/2246 are pre-existing, none in the co-pack code).

## Threat Mitigations (from the plan's STRIDE register)
- **T-12-04 (Spoofing — GPU-vs-GPU masking a divergence):** mitigated. The ~1e-6 half pins to the CPU f64 anchor (`find_best_split_cpu_native`), never a second GPU path (def-f8u-01). The byte-identical half is an EXACT same-kernel-same-bits check, valid as an exactness gate. A silent per-sibling mis-decode (wrong half / wrong min_gain_shift) would fail the anchor pin.
- **T-12-05 (Tampering — relaxing a golden to force green):** mitigated. Task 2 was confirmation-only; the CPU merge gate passed at existing tolerances with zero golden edits.
- **T-12-SC (cargo installs):** accept — no new package-manager installs (test-only Rust changes).

## Next Phase Readiness
- Plan 03 can wire the device-time + e2e A/B (`LGBM_SIBLING_COPACK=0/1`) and assert `SCAN_RESIDENT_CNT` ~halves per tree (~59 → ~30) under `LGBM_PHASE_PROF=1` — the parity foundation (SC-1) is now gated on both hip and cubecl-cpu.

## Self-Check: PASSED
- `crates/oracle-harness/tests/kernel_parity.rs` exists on disk with both new cells + the three fixture helpers.
- Task 1 commit `25f7fdd` is present in git history.

---
*Phase: 12-gpu-sibling-scan-copack*
*Completed: 2026-06-25*
