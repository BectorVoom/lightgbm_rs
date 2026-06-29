---
phase: 14-foundation-shared-device-primitives-device-structs-rng
plan: 01
subsystem: infra
tags: [cubecl, plane-intrinsics, prefix-sum, gpu, rocm, hip, kernels, scaffolding]

# Dependency graph
requires:
  - phase: 04-compute-seam
    provides: cubecl kernel crate (histogram.rs plane/LDS/launch prior art), runtime capability probe
provides:
  - "Resolved RESEARCH Open Q1 / Pitfall 1: cubecl-0.10 plane scan/reduction intrinsic lowering, per backend"
  - "plane_intrinsic_smoke.rs Wave-0 de-risk gate (cpu capability assertion + hip per-intrinsic parity)"
  - "Finding: plane_inclusive_sum/exclusive_sum/max/min all LOWER on cubecl-hip (gfx1100); NO plane_shuffle_up fallback needed"
  - "Finding: plane intrinsics + UNIT_POS_PLANE are UNSUPPORTED on cubecl-cpu — the cpu anchor is the serial fold, not a plane kernel"
  - "Three wired-but-empty kernel module stubs: primitives.rs, split_info.rs, random.rs"
affects: [14-03, 14-04, 14-05, 15-minimal-on-device-growth]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Per-intrinsic-per-backend plane-op smoke test anchored to a serial Rust reference (never GPU-vs-GPU)"
    - "Wired-but-empty doc-only kernel module stub so Wave-2 plans each own one file with zero mod.rs contention"

key-files:
  created:
    - crates/lgbm-compute/tests/plane_intrinsic_smoke.rs
    - crates/lgbm-compute/src/kernels/primitives.rs
    - crates/lgbm-compute/src/kernels/split_info.rs
    - crates/lgbm-compute/src/kernels/random.rs
  modified:
    - crates/lgbm-compute/src/kernels/mod.rs

key-decisions:
  - "cubecl-hip lowers all four plane scan/reduction intrinsics correctly (~1e-6) — 14-03 builds directly on the intrinsics, no plane_shuffle_up manual-scan fallback"
  - "cubecl-cpu has NO plane support (has_plane=false, plane_size=1); the f64 cpu anchor stays the serial sequential fold — plane collectives are a strictly has_plane (GPU) concern"
  - "The plane_shuffle_up fallback is a within-backend GPU fallback only; it is NOT a cpu option (it is itself a plane op, equally unsupported on cpu)"
  - "Three new kernel modules are ungated (like histogram), NOT #[cfg(feature=gpu)] like autotune"

patterns-established:
  - "Plane smoke test gates the no-op case via exclusive_sum-of-first-lane==0 (catches a silent return-the-input lowering failure)"
  - "Plane width read from probe_capabilities().plane_size, never hard-coded 32/64"

requirements-completed: [ODL-01]

# Metrics
duration: ~35 min
completed: 2026-06-29
status: complete
---

# Phase 14 Plan 01: Foundation Wave-0 — Plane-Intrinsic De-Risk + Kernel Scaffolding Summary

**Resolved the phase's single MEDIUM unknown — cubecl-0.10 plane scan/reduction intrinsic lowering — proving all four lower on cubecl-hip (no fallback) while cubecl-cpu has no plane support (anchor stays the serial fold), and laid down three wired-but-empty kernel module stubs so Wave-2 plans each own exactly one file.**

## Performance

- **Duration:** ~35 min
- **Completed:** 2026-06-29T09:34:37Z
- **Tasks:** 2
- **Files modified:** 5 (4 created, 1 modified)

## Accomplishments
- **De-risk gate (Open Q1 / Pitfall 1) resolved on BOTH backends.** Authored a per-intrinsic-per-backend smoke test that exercises `plane_inclusive_sum`, `plane_exclusive_sum`, `plane_max`, `plane_min` against a plain serial Rust reference fold (anchor discipline D-10, never GPU-vs-GPU).
- **cubecl-hip (gfx1100 APU): all four intrinsics LOWER and match within ~1e-6** — the `plane_intrinsics_lower_on_hip` test passes. No `plane_shuffle_up` manual-scan fallback is required; 14-03 builds directly on the 0.10 intrinsics.
- **cubecl-cpu: plane intrinsics are NOT supported** — launching any of them aborts with `plane_inclusive_sum(..) is not supported on CPU.` (and `UNIT_POS_PLANE` aborts with `Unsupported builtin was used: UnitPosPlane`). This is consistent with `has_plane=false` / `plane_size=1`. The cpu f64 anchor is therefore the serial sequential fold (`ReducePath::Sequential`), not a plane kernel — asserted as the architectural fact so a future cubecl bump that changed cpu plane support would flip the gate.
- **Three kernel module stubs wired into the barrel** (`primitives.rs`, `split_info.rs`, `random.rs`), each documenting its owning Wave-2 plan, locked decisions, and analog file; `primitives.rs` records the plane-lowering finding inline for 14-03 to consume.

## Per-Intrinsic Lowering Result (the input 14-03 consumes)

| Intrinsic | cubecl-cpu (anchor) | cubecl-hip (gfx1100) | Path for 14-03 |
|-----------|---------------------|----------------------|----------------|
| `plane_inclusive_sum` | unsupported (use serial fold) | LOWERS, ~1e-6 ✅ | intrinsic (GPU); serial fold (cpu anchor) |
| `plane_exclusive_sum` | unsupported (use serial fold) | LOWERS, ~1e-6 ✅ | intrinsic (GPU); serial fold (cpu anchor) |
| `plane_max` | unsupported (use serial fold) | LOWERS, ~1e-6 ✅ | intrinsic (GPU); serial fold (cpu anchor) |
| `plane_min` | unsupported (use serial fold) | LOWERS, ~1e-6 ✅ | intrinsic (GPU); serial fold (cpu anchor) |

**No `plane_shuffle_up` manual-scan fallback is needed on any backend.** On hip the native intrinsics work; on cpu no plane op (incl. `plane_shuffle_up`) is available at all, so the cpu anchor uses the existing serial sequential fold (`runtime::Capabilities::accumulate_type()` f64 path), exactly as the shipped histogram path already does.

## Task Commits

1. **Task 1: Plane-intrinsic per-backend smoke test (Open Q1)** - `de50a96` (test)
2. **Task 2: Scaffold primitives/split_info/random module stubs + wire barrel** - `319406e` (feat)

## Files Created/Modified
- `crates/lgbm-compute/tests/plane_intrinsic_smoke.rs` (created) - Wave-0 de-risk gate: `cpu_anchor_has_no_plane_support_serial_fold_is_the_reference` (default, asserts the cpu capability fact + validates the serial reference folds) and `plane_intrinsics_lower_on_hip` (rocm-gated, `has_plane`-guarded, per-intrinsic ~1e-6 parity vs serial reference).
- `crates/lgbm-compute/src/kernels/primitives.rs` (created) - Stub for the shared device primitives (filled by 14-03/14-05; ODL-01, D-01/D-02); records the 14-01 plane-lowering finding.
- `crates/lgbm-compute/src/kernels/split_info.rs` (created) - Stub for the SoA pre-allocated device split-record (filled by 14-04; ODL-02, D-05/D-06/D-08), incl. the `MAX_CAT_PER_SPLIT` reserved-slab note (Open Q3 → 14-04).
- `crates/lgbm-compute/src/kernels/random.rs` (created) - Stub for the `CUDARandom` LCG (filled by 14-04; ODL-02, D-04), incl. the recurrence/draw methods and V6 negative control.
- `crates/lgbm-compute/src/kernels/mod.rs` (modified) - Wired `pub mod primitives; pub mod random; pub mod split_info;` ungated alongside the existing kernel modules.

## Verification
- `cargo test -p lgbm-compute --test plane_intrinsic_smoke` → 1 passed (cpu default), 0 warnings.
- `cargo test -p lgbm-compute --features rocm --test plane_intrinsic_smoke` → 2 passed (cpu + hip).
- `cargo build -p lgbm-compute` → clean, no warnings introduced by the stubs.
- `cargo test -p lgbm-compute --no-run` → all test targets compile.
- `cargo test -p lgbm-compute --lib` → 52 passed, 0 failed (existing compute tests unchanged).

## Decisions Made
- See `key-decisions` frontmatter. Headline: 14-03 uses the cubecl-0.10 plane intrinsics directly on the GPU path (no fallback) and the serial f64 fold on the cpu anchor; plane collectives are gated on `has_plane`.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Corrected Assumption] cpu test asserts "no plane support" instead of running plane ops on the cpu anchor**
- **Found during:** Task 1 (the de-risk smoke test — its explicit purpose)
- **Issue:** The plan (and RESEARCH Assumption A1 / acceptance criterion "All four plane intrinsics produce results matching the serial reference on the cubecl-cpu anchor, bit-exact") assumed the plane intrinsics lower on cubecl-cpu. The smoke test discovered they do NOT: `UNIT_POS_PLANE` aborts with `Unsupported builtin was used: UnitPosPlane` and `plane_inclusive_sum(..)` aborts with `... is not supported on CPU.`. This is consistent with the established `has_plane=false` / `plane_size=1` capability matrix (RESEARCH Pitfall 2).
- **Fix:** Reframed the default (cpu) test to assert the architectural fact — `has_plane==false`, `plane_size==1`, and that the serial reference folds (the actual cpu anchor) are correct — and gated the plane probe kernels + launcher behind `#[cfg(feature = "rocm")]`. The plane-intrinsic lowering proof lives in the rocm-gated `plane_intrinsics_lower_on_hip` test. The `<done>` criterion ("the plane-intrinsic lowering question is resolved on both backends and the chosen path recorded for 14-03") is fully met; only the literal "bit-exact on cpu" wording was based on the wrong assumption the gate existed to test.
- **Files modified:** crates/lgbm-compute/tests/plane_intrinsic_smoke.rs
- **Verification:** cpu test green (asserts the capability + serial reference); hip test green (per-intrinsic ~1e-6). No warnings.
- **Committed in:** `de50a96` (Task 1 commit)

---

**Total deviations:** 1 (corrected assumption surfaced by the de-risk gate doing its job — not a code bug).
**Impact on plan:** None on scope. The finding *strengthens* the downstream plans: 14-03 now knows to use the intrinsics on the `has_plane` GPU path and the serial fold on the cpu anchor, with no fallback work and no cpu plane kernel.

## Issues Encountered
None beyond the documented deviation. The `ArrayArg::from_raw_parts(handle, len)` 2-arg signature and the `UNIT_POS_PLANE as usize` index cast were resolved against the in-repo histogram/partition launcher idioms.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- **14-03 (primitives, Wave 2):** unblocked — owns `primitives.rs`; build the prefix-sum/reduction/argsort kernels directly on the cubecl-0.10 plane intrinsics for the GPU path (no `plane_shuffle_up` fallback), serial f64 fold on the cpu anchor. The block/global LDS staging (RESEARCH Pattern 2/3) is the real porting work, not the warp primitives.
- **14-04 (split-record + RNG, Wave 2):** unblocked — owns `split_info.rs` + `random.rs`; both stubs document the field layout, `MAX_CAT_PER_SPLIT` (Open Q3), the LCG recurrence, and the parity oracle.
- No `mod.rs` contention: each Wave-2 plan fills exactly one already-wired file.
- No blockers. ROCm parity confirmed on the local (spoofed gfx1100) APU.

## Self-Check: PASSED
- Files created exist on disk: `plane_intrinsic_smoke.rs`, `primitives.rs`, `split_info.rs`, `random.rs` (all FOUND).
- `mod.rs` declares all three modules (verified via `pub mod (primitives|split_info|random)` grep).
- Commits exist: `de50a96` (Task 1), `319406e` (Task 2).

---
*Phase: 14-foundation-shared-device-primitives-device-structs-rng*
*Completed: 2026-06-29*
