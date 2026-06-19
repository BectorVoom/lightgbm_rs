---
phase: quick-260619-mwr
plan: 01
subsystem: infra
tags: [cubecl, rocm, gpu, histogram, launch_unchecked, resident-handle, benchmark]

requires:
  - phase: quick-260619-j9t
    provides: the CUDA-mirror histogram kernel + per-call launcher + rocm_cuda_mirror parity test
provides:
  - launch_unchecked CUDA-mirror histogram kernel (drops redundant in-kernel bounds-check codegen)
  - construct_histograms_cuda_mirror_resident_on (upload-once / CUDA-faithful resident-Handle launcher)
  - parity coverage pinning the resident path to the CPU f64 anchor
  - warmed-up before/after micro-bench with real gfx1100 transfer-overhead figures
affects: [DEF-f8u-01 live-wiring, gpu-histogram-kernel, gpu-routing]

tech-stack:
  added: []
  patterns:
    - "launch_unchecked + host-side V5 validation discharging the in-kernel bounds-check contract"
    - "resident-Handle launcher variant (upload-once) mirroring build_leaf_histograms_resident_f32_on"

key-files:
  created:
    - crates/lgbm-compute/examples/cuda_mirror_overhead.rs
  modified:
    - crates/lgbm-compute/src/kernels/histogram.rs
    - crates/lgbm-compute/tests/rocm_cuda_mirror.rs

key-decisions:
  - "launch_unchecked is numerics-preserving (only removes bounds-check codegen) — proven by an isolation A/B against the prior `launch` kernel"
  - "The resident-Handle variant validates everything reachable host-side; the bin-range invariant is the caller's upload-time responsibility (same contract as build_leaf_histograms_resident_f32_on)"
  - "Pre-existing flaky full-corpus parity test logged as DEF-MWR-01, NOT fixed (tolerance-change is out of scope per the plan)"

patterns-established:
  - "Pattern: a resident-Handle launcher variant beside a per-call launcher, sharing the same kernel/CubeCount/CubeDim — the upload-once seam"

requirements-completed: [MWR-01, MWR-02, MWR-03]

duration: ~35min
completed: 2026-06-19
---

# Quick 260619-mwr: CUDA-mirror histogram overhead reduction Summary

**Cut the CUDA-mirror GPU histogram kernel's non-compute overhead ~9–11× by uploading the
feature-major bin buffer ONCE (resident-Handle launcher) instead of re-uploading the full
40 MB matrix every call, plus switching the kernel to `launch_unchecked` to drop redundant
in-kernel bounds-check codegen — measured on gfx1100, parity to the CPU f64 anchor held.**

## Performance

- **Duration:** ~35 min
- **Tasks:** 3 / 3
- **Files modified:** 2 (1 created, 1 modified) + 1 test modified

## Accomplishments

### Task 1 (MWR-01 + MWR-02) — `launch_unchecked` kernel + resident-Handle launcher

- Switched `construct_hist_cuda_mirror_kernel` from `#[cube(launch)]` to
  `#[cube(launch_unchecked)]`, and the per-call launcher
  `construct_histograms_cuda_mirror_on` from `::launch` to `::launch_unchecked`. Extended
  the SAFETY comment to state that the V5 boundary validation before upload (every
  `data_indices[k] < num_data`; `bin < num_bin <= 256`; length checks) discharges every
  device-access obligation the `launch_unchecked` contract requires (`data[col+data_index]`,
  `grad[data_index]`, `out[base+m]`, `sub[bin*2+1]`). The launch does NOT change numerics —
  only bounds-check codegen is removed.
- Added `construct_histograms_cuda_mirror_resident_on<R>` — the upload-once / CUDA-faithful
  variant. It accepts a pre-uploaded `cubecl::server::Handle` for the feature-major bin
  buffer (length `num_features * num_data`), does NOT re-upload it, and per call uploads only
  `data_indices`, `grad`, `hess`, the sentinel `slot_off`, and the zeroed `out`. It launches
  the SAME kernel with the SAME `CubeCount::Static(num_features, p, 1)` / `CubeDim::new_1d(256)`
  config (so numerics are identical), validates everything reachable host-side (grad/hess
  lengths, `slot_off.len()`, `num_bin<=256`, `data_index<num_data`, `num_features!=0`), early-
  returns zeros on an empty leaf, and widens f32→f64 on read-back. The bin-range invariant is
  documented as the caller's upload-time responsibility (mirrors `build_leaf_histograms_resident_f32_on`).
- Both `--features rocm` and CPU-only builds compile (the mirror code stays `#[cfg(feature="rocm")]`).

### Task 2 (MWR-02 parity) — resident path pinned to the CPU f64 anchor

- Added `cuda_mirror_resident_matches_cpu_anchor_within_tol`: uploads the corpus's
  feature-major `resident` buffer ONCE via the rocm client, calls the resident launcher with
  that `Handle`, and asserts against the SAME `cpu_anchor` used by the existing tests
  (GPU-vs-CPU-f64-anchor, NEVER GPU-vs-GPU, per memory DEF-f8u-01). Reuses `make_corpus`,
  `assert_close` (ABS 5e-6 / REL 1e-5), and the `(7..num_data).step_by(3)` leaf subset.
- The new test is STABLE (5/5 across repeated runs); the existing `dense` and `empty` tests
  (which now exercise the per-call `launch_unchecked` switch unchanged) are stable too. No
  existing test or tolerance was changed.

### Task 3 (MWR-03) — warmed-up before/after micro-bench with REAL figures

- Created `crates/lgbm-compute/examples/cuda_mirror_overhead.rs` (rocm-gated, with a CPU-only
  stub `main`). 50 features / 200k rows, ~half-rows leaf, sweep at 16/64/256 bins. Honors the
  warm-vs-cold rule: 3 discarded warm-ups + median of 7 timed launches per variant, with a
  device sync via result read-back inside each timed call. Quantifies MB-transferred-per-call.

## Measured results (gfx1100, release)

Two reproducible runs (medians, 50 features × 200k rows, ~100k-row leaf):

| bins | per-call median ms | resident median ms | speedup | per-call MB | resident MB |
|------|-------------------:|--------------------:|--------:|------------:|------------:|
| 16   | 38.7 / 37.9        | 3.63 / 3.43         | 10.7× / 11.1× | 40.1 | 1.9 |
| 64   | 35.8 / 32.4        | 3.29 / 3.39         | 10.9× / 9.6×  | 40.1 | 1.9 |
| 256  | 35.5 / 32.9        | 3.34 / 3.67         | 10.6× / 9.0×  | 40.1 | 1.9 |

**Transfer-overhead win (MWR-02, the dominant lever):** the resident upload-once path is
**~9–11× faster** than the per-call full-re-upload path. The per-call path moves **40.1 MB**
every call (the full 50×200k×4 = 38.1 MB feature-major bin buffer + 1.9 MB per-leaf
idx/grad/hess); the resident path moves only **1.9 MB** per call — a **~21× transfer
reduction**. The speedup is flat across bin counts because the cost is transfer-bound, not
compute-bound, exactly as hypothesized.

**Launch-overhead win (MWR-01):** `launch_unchecked` is baked into BOTH timed paths (it's a
comptime kernel attribute, so it can't be A/B-toggled in one binary). It removes the
in-kernel per-access bounds-check branch from the hot scatter loop per the cubecl manual
rationale; its effect is subsumed in the measured medians and reported qualitatively. The
dominant measured overhead is transfer, not launch, at these sizes.

## Deviations from Plan

### Out-of-scope discovery (logged, not fixed)

**1. [SCOPE BOUNDARY] `cuda_mirror_full_corpus_leaf_matches_anchor` is PRE-EXISTING flaky — DEF-MWR-01**
- **Found during:** Task 2 (running the full mirror suite).
- **Issue:** the pre-existing all-2000-rows full-corpus parity test fails intermittently
  (e.g. |diff| ~8.7e-6 > the test's ABS 5e-6 floor) on grad cells whose true sum is near
  zero — the f32-atomic cancellation residual occasionally exceeds the floor, and the
  accumulation order is nondeterministic run-to-run. The test's documented "~2.4e-6 max" was
  optimistic.
- **Proven pre-existing:** reverting BOTH `histogram.rs` and `rocm_cuda_mirror.rs` to `HEAD~1`
  (the `#[cube(launch)]` checked kernel + the original 3 tests) and running the full-corpus
  test 6× still produced a failure (1/6). `launch_unchecked` only removes bounds-check
  codegen — it cannot change f32-atomic accumulation order — so this task did not introduce
  or worsen it.
- **Why not fixed:** the plan explicitly forbids weakening the tolerance or changing the
  existing three tests. The fix (raise the full-corpus ABS floor to match the real f32-atomic
  cancellation envelope, or bound the leaf size) is a separate decision.
- **Logged to:** `deferred-items.md` in this quick-task directory (DEF-MWR-01).
- **Scope-clean evidence:** the new resident test + the per-call `dense`/`empty` tests are
  stable; both `launch_unchecked` paths agree with the CPU f64 anchor on the plan-specified
  `(7..num_data).step_by(3)` leaf subset.

Otherwise: plan executed as written.

## Constraints honored

- PARITY GATE: both launch paths pinned GPU-vs-CPU-f64-anchor (never GPU-vs-GPU); the new
  resident test stable; tolerance unchanged.
- The mirror stays a rocm-gated TESTED PRIMITIVE — NOT wired into production
  `construct_histograms` / build path; the CPU f64 anchor
  (`construct_hist_kernel` / `construct_histograms_cpu`) is UNTOUCHED (lib tests 30/0).
- Warm-vs-cold rule: 3 discarded warm-ups + median of 7 per variant; numbers reproduced
  across two runs.
- `LightGBM-release-4.6.0.99/` and `LightGBM/` reference trees never git-added.
- Clippy clean on all new code (the only `histogram.rs` warnings are pre-existing, at
  lines 1832/1833/2300, outside the edited region).

## Commits

- `61b96d3` perf(quick-260619-mwr-01): launch_unchecked mirror kernel + resident-Handle launcher
- `94ac054` test(quick-260619-mwr-02): pin resident-Handle mirror launcher to CPU f64 anchor
- `a313fbb` perf(quick-260619-mwr-03): warmed-up before/after micro-bench for the mirror kernel

## Self-Check: PASSED

- Files: `cuda_mirror_overhead.rs`, `histogram.rs`, `rocm_cuda_mirror.rs`, this SUMMARY — all FOUND.
- Commits `61b96d3` / `94ac054` / `a313fbb` — all FOUND.
- Contract greps: `launch_unchecked` in histogram.rs, `resident` in the test, `rocm` in the example — all present.
- `cargo build -p lgbm-compute --features rocm` + CPU-only build compile; lib tests 30/0 (CPU anchor untouched); new resident parity test stable.
