---
phase: quick-260621-iaq
plan: 01
subsystem: lgbm-compute (GPU histogram kernels)
tags: [gpu, rocm, histogram, lds, parity, dead-code-removal]
status: complete
requires: []
provides:
  - "GPU parity seam (RocmBackend::construct_histograms) driven by the LDS kernel"
affects:
  - "oracle-harness kernel_parity / learner_parity / boosting_parity (route through the seam)"
  - "rocm_backend_parity (exercises the rewired seam)"
tech-stack:
  added: []
  patterns:
    - "single GPU f32 accumulation path (LDS privatize-then-merge) at the parity seam"
key-files:
  created: []
  modified:
    - crates/lgbm-compute/src/lib.rs
    - crates/lgbm-compute/src/kernels/histogram.rs
    - crates/lgbm-compute/tests/rocm_parallel_histogram.rs
    - crates/lgbm-compute/tests/rocm_plane_aggregate.rs
  deleted:
    - crates/lgbm-compute/examples/lazy_dispatch_ab.rs
    - crates/lgbm-compute/examples/batched_read_audit_ab.rs
    - crates/lgbm-compute/examples/plane_aggregate_ab.rs
    - crates/lgbm-compute/examples/launch_unchecked_ab.rs
decisions:
  - "Retarget surviving plane/LDS test baselines to construct_histograms_parallel_f32_plane_on(.., false) — the byte-faithful twin of the deleted kernel"
metrics:
  tasks_completed: 2
  tasks_total: 3
  completed: 2026-06-21
---

# Phase quick-260621-iaq Plan 01: Replace old GPU global-atomic histogram kernel with LDS Summary

Rewired the GPU parity seam `RocmBackend::construct_histograms` to drive the LDS-privatized sub-histogram kernel (`construct_histograms_lds_f32_on`) and deleted the now-dead global-atomic kernel + launcher and its A/B scaffolding; both builds compile and the full `lgbm-compute` rocm suite is green on gfx1100. Task 3 (the blocking oracle-harness parity re-run) is the next step and was NOT self-cleared.

## What Was Done

### Task 1 (commit `4a37ae8`) — Rewire seam + delete dead kernel
- `lib.rs`: `RocmBackend::construct_histograms` now calls `kernels::histogram::construct_histograms_lds_f32_on(...)` (identical arg list). Replaced the stale "NOT wired here" comment block with a concise note that the seam now drives the LDS path (per-cube LDS atomics then a single global merge), correct vs naive on integer data, within the ~1e-6 ROCm gate vs the CPU f64 anchor, and faster under contention — and that the swap CHANGES the seam's f32 accumulation ORDER (the GPU ~1e-6 best-effort contract; the CPU f64 anchor is unaffected and stays the bit-exact hard gate).
- `histogram.rs`: deleted `construct_hist_kernel_atomic_f32` (the `#[cube(launch_unchecked)]` kernel) and its host launcher `construct_histograms_parallel_f32_on`. Reworded the 4 doc/comment references to those symbols (2 rustdoc intra-doc links + 2 plain `//` comments) so no broken-link warnings and no name references to the deleted symbols remain in `src/`.
- Untouched (per HARD CONSTRAINTS): the `Backend::construct_histograms` trait signature, the CpuBackend f64 anchor, the resident/LDS production BUILD path, and the surviving `_plane` and `_lds` kernel/launcher bodies.

Verified: `cargo build -p lgbm-compute` and `--features rocm` both compile clean; `grep` reports SRC CLEAN (no refs to the deleted symbols); `cargo doc --features rocm` emits no broken-link warnings referencing the deleted symbols (the remaining doc warnings are all pre-existing and unrelated).

### Task 2 (commit `c8b76b4`) — Retarget dependents + run the rocm gate
- `rocm_parallel_histogram.rs`: removed the dead-kernel import; deleted the 3 tests exclusive to the removed kernel (`parallel_atomic_no_lost_updates_under_contention`, `parallel_within_tolerance_of_cpu_f64_anchor`, `parallel_is_faster_than_single_unit_on_gpu`); retargeted the LDS integer-equality baseline (`lds_equals_naive_atomic_on_integer_data`) and the bench (`bench_lds_vs_naive_atomic_large`) to `construct_histograms_parallel_f32_plane_on(.., false)`. Kept all LDS tests.
- `rocm_plane_aggregate.rs`: removed the dead-kernel import; retargeted both baseline uses (in `plane_large_leaf_drift_not_worse_than_baseline` and `plane_equals_baseline_on_integer_data`) to the plane `use_plane=false` arm; deleted the now-degenerate `plane_launcher_false_arm_matches_shipped_baseline`.
- Deleted the 4 obsolete A/B example files (measurement-only, zero production callers per STATE): `lazy_dispatch_ab.rs`, `batched_read_audit_ab.rs`, `plane_aggregate_ab.rs`, `launch_unchecked_ab.rs`.

Verified: DEP CLEAN (no test/example refs to deleted symbols); `cargo build -p lgbm-compute --features rocm --tests --examples` compiles clean; `cargo test --release -p lgbm-compute --features rocm` GREEN on gfx1100 — all binaries 0 failed:
- lib unittests 44 passed / 1 ignored
- rocm_backend_parity 4/4 (exercises the rewired seam)
- rocm_parallel_histogram 4/4, rocm_plane_aggregate 4/4, rocm_row_partition 2/2, rocm_smoke 2/2, rocm_cuda_mirror 4/4
- capability/cmp01/determinism all pass

No 1e-6 knife-edge failure surfaced in the lgbm-compute suite. LDS + plane coverage preserved.

## Deviations from Plan

None of substance. One extra-but-implied cleanup beyond the plan's enumerated edit points: two PLAIN `//` comment references to the deleted symbols (histogram.rs SAFETY comment + an LDS-kernel inline comment) and one test doc-comment reference were also reworded, to satisfy the done criterion "no reference to the deleted symbols remains anywhere in `crates/lgbm-compute/`" (the plan named only the rustdoc intra-doc links explicitly). No behavior change.

## Known Stubs

None.

## Threat Flags

None — no new network/auth/file/schema surface introduced.

## Checkpoint Status

Task 3 is a `checkpoint:human-verify gate="blocking"` and was deliberately NOT self-cleared. The oracle-harness end-to-end parity tests must be re-run on gfx1100 to confirm the seam swap (which changes the seam's f32 accumulation order) stays within the ~1e-6 gate:

```
cargo test --release -p oracle-harness --features rocm kernel_parity
cargo test --release -p oracle-harness --features rocm learner_parity
cargo test --release -p oracle-harness --features rocm boosting_parity
cargo test --release -p lgbm-compute --features rocm rocm_backend_parity   # already green this session: 4/4
```

If ANY cell lands on the 1e-6 knife-edge (cf. DEF-f8u-01), surface the failing cell + both values — do NOT loosen any tolerance and do NOT touch the CPU f64 anchor.

## Self-Check: PASSED
- Commits exist: `4a37ae8` (Task 1), `c8b76b4` (Task 2) — confirmed via `git log`.
- Deleted symbols absent across `crates/lgbm-compute/` (src + tests + examples) — confirmed via `grep` (NONE).
- Seam wired to `construct_histograms_lds_f32_on` in `lib.rs` — confirmed via `grep`.
- Modified files exist; 4 example files confirmed deleted.
