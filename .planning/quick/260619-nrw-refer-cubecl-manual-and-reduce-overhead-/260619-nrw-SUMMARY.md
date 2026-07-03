---
status: complete
phase: quick-260619-nrw
plan: 01
subsystem: infra
tags: [cubecl, rocm, gpu, histogram, launch_unchecked, parity]

requires:
  - phase: quick-260619-mwr
    provides: launch_unchecked lever + per-access SAFETY-contract template (applied to the CUDA-mirror only)
  - phase: quick-260619-ngo
    provides: A/B that confirmed the wired LDS resident path is the optimal production kernel
provides:
  - All 8 rocm-gated production histogram kernels launch via ::launch_unchecked (no in-kernel bounds-check codegen)
  - Per-kernel SAFETY enumerations on every swept launcher (host V5 validation discharges the contract)
  - GPU-vs-CPU-f64-anchor re-pin coverage for the one swept kernel (naive batched fallback) that lacked it
affects: [gpu-overhead-campaign, histogram-build-perf, future-comptime-specialization]

tech-stack:
  added: []
  patterns:
    - "launch_unchecked + host-V5-discharge: production rocm kernels drop bounds-check codegen, every device access enumerated in a SAFETY block proven by pre-upload validation"

key-files:
  created: []
  modified:
    - crates/lgbm-compute/src/kernels/histogram.rs
    - crates/lgbm-compute/tests/rocm_row_partition.rs

key-decisions:
  - "NRW-02 (#[comptime] specialization) DROPPED, not deferred — the only material candidate (bin-count lds_len/feat_len) re-introduces the multi-binary cost the repo deliberately avoids; the remaining run-constant scalar (num_data stride) gates no GPU-side branch"
  - "Stage-3 order-changing restructure NOT pursued — ngo's A/B already showed the wired LDS kernel is at/near optimal; no win regime"
  - "The two CPU f64 anchor kernels (construct_hist_kernel, construct_hist_kernel_f32) left #[cube(launch)] and byte-unchanged — they carry the bit-exact merge gate, not the rocm-overhead target"

patterns-established:
  - "Per-kernel SAFETY enumeration: copy the mwr mirror template, list every device array index reachable in the kernel, cite the host check that bounds it, end with the numerics-unchanged clause"

requirements-completed: [NRW-01, NRW-03]

duration: 38min
completed: 2026-06-19
---

# Quick 260619-nrw: launch_unchecked sweep of the production GPU histogram kernels Summary

**All 8 rocm-gated production histogram kernels now launch via `::launch_unchecked`, dropping the in-kernel per-access bounds-check codegen the `#[cube(launch)]` macro emitted in their scatter hot loops; the host-side V5 validation already present in every launcher discharges the unsafe contract, and every swept kernel is re-pinned GPU-vs-CPU-f64-anchor (f64 kernels bit-exact, f32 within ~1e-6).**

## Performance

- **Duration:** ~38 min
- **Started:** 2026-06-19
- **Completed:** 2026-06-19
- **Tasks:** 3 completed
- **Files modified:** 2

## Accomplishments

### Task 1 — f64-deterministic + wired LDS path (commit d4dde2f)
Switched the SAFEST/highest-value kernels first (RESEARCH §Recommended-Ordering 1a+1b):
- `fix_compact_kernel` (2 launch sites) and `build_fix_scan_fused_kernel` — f64, one-cube-per-feature, ascending fold ⇒ **zero numeric risk, bit-exact**.
- `construct_leaf_hist_resident_lds_kernel` — the **wired production training path** (`resident_raw_build_into` LDS branch).

### Task 2 — remaining 5 rocm f32 kernels (commit 609cc67)
`construct_hist_kernel_atomic_f32`, `construct_hist_kernel_lds_f32`, `construct_leaf_hist_batched_kernel`, `construct_leaf_hist_batched_lds_kernel`, `construct_leaf_hist_resident_kernel`. After this, **no `#[cfg(feature="rocm")]` production kernel uses `#[cube(launch)]`** — the only surviving `::launch(` sites are the two CPU f64 anchor launchers (:177, :362).

### Task 3 — re-pin GPU-vs-CPU-f64-anchor (commit 61aac21)
Audited existing coverage: 7 of 8 swept kernels were already pinned (rocm_parallel_histogram = atomic+lds; oracle-harness kernel_parity hip module = resident-LDS/fix_compact/resident-fix_compact/fused; rocm_row_partition = batched-LDS). Added the one missing pin — the **naive batched fallback** (`construct_leaf_hist_batched_kernel`, the `max_w > HIST_LDS_MAX` branch) — forcing it with one 300-bin feature and asserting against the CPU f64 anchor within ABS 5e-6 / REL 1e-5 using the bounded leaf subset `(7..n).step_by(3)`.

## Verification

| Suite | Result |
|-------|--------|
| `cargo build -p lgbm-compute` (CPU-only) | compiles |
| `cargo build -p lgbm-compute --features rocm` | compiles |
| `rocm_parallel_histogram` (atomic + lds vs anchor) | 7/7 green |
| `rocm_row_partition` (batched LDS P1/P>1 + new naive fallback) | 2/2 green |
| `oracle-harness kernel_parity --features rocm` (resident, fix_compact, fused — f64 **bit-exact** pins) | 15/15 green |
| `rocm_cuda_mirror` | 3/4 (1 pre-existing DEF-MWR-01 flake — see below) |
| CPU f64 anchor merge-gate (`lgbm-compute` lib + cpu `kernel_parity`) | green (anchor numerics untouched) |
| clippy on edited `histogram.rs` + new test | no new warnings from the edits |

The two f64-deterministic kernels (`fix_compact`, fused) stayed **bit-exact** to the host after the switch (`kernel_parity_fix_compact_equals_host_on_hip`, `kernel_parity_build_fix_scan_equals_host_on_hip`), empirically confirming launch_unchecked is numerics-preserving.

## Deviations from Plan

None — plan executed exactly as written. Tasks 1–3 ran in order; the comptime (NRW-02) and Stage-3 work were already scoped-out in the plan (not invented), and no architectural change was needed.

## Parity Residual — DEF-MWR-01 (pre-existing, FLAGGED, attributed — NOT a regression)

`cuda_mirror_full_corpus_leaf_matches_anchor` in `rocm_cuda_mirror.rs` fails **intermittently** with |diff| ~6.5e-6–7.2e-6 on cell 0 (anchor ≈ -0.125 — a full-corpus near-zero-grad cancellation cell, ABS-floor tol 6.25e-6).

Attribution (per CONTEXT/RESEARCH Pitfall 1, and verified this session):
- The CUDA-mirror kernel and both its launchers were **untouched by nrw** — they were already `#[cube(launch_unchecked)]` from mwr (commit 61b96d3). `git diff c3e5b05 HEAD` shows zero `cuda_mirror` lines changed.
- **Reproduced on pristine pre-nrw code** (reverted `histogram.rs` to c3e5b05): 2/6 failures, identical intermittent rate to the nrw branch.
- The |diff| **varies run-to-run** (6.5e-6 / 7.15e-6 …) ⇒ pure f32-atomic accumulation order. `launch_unchecked` **cannot change accumulation order** (it only removes bounds-check codegen), so it cannot be the cause.

This is the documented DEF-MWR-01 landmine (out-of-scope CUDA-mirror primitive, full-corpus near-zero-grad f32-atomic cancellation). **No tolerance gate was widened.** No swept production kernel — and no bounded-subset cell — moved past tolerance: every in-scope re-pin is green.

## Scope Notes (dropped, not deferred — no padding)

- **NRW-02 (`#[comptime]`):** the only material candidate (bin-count `lds_len`/`feat_len`) would re-introduce the per-bin-count multi-binary cost the repo deliberately avoids (histogram.rs:458-463); the remaining run-constant scalar (`num_data` stride) gates no GPU-side branch. Dropped.
- **Stage-3 restructure:** ngo's A/B showed the wired LDS kernel at/near optimal — no win regime. Not invented just because CONTEXT permits it.

## Known Stubs

None.

## Self-Check: PASSED
- crates/lgbm-compute/src/kernels/histogram.rs — FOUND (modified, both builds compile)
- crates/lgbm-compute/tests/rocm_row_partition.rs — FOUND (new test green)
- commit d4dde2f — FOUND
- commit 609cc67 — FOUND
- commit 61aac21 — FOUND
- Gate grep: 8 production launchers on `::launch_unchecked`; only the 2 CPU anchors remain `::launch(` / `#[cube(launch)]` (verified)
