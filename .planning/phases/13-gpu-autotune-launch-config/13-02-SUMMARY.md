---
phase: 13-gpu-autotune-launch-config
plan: 02
subsystem: gpu-compute
tags: [cubecl, autotune, rocm, histogram, build-P, row-partition, launch-config]

# Dependency graph
requires:
  - phase: 13-01
    provides: "kernels::autotune module (LaunchKey, size_band, autotune_enabled, cache_namespace_id) + serde rocm dep + prefers_autotune_launch_config seam"
  - phase: 11-quantized-training
    provides: "u64 fixed-point resident-build kernel (parity-neutral across P)"
  - phase: spikes-037-040
    provides: "FreshOutGenerator pattern, log2 key, vs-heuristic win, manual-API corrections"
provides:
  - "FreshOutGenerator<R> — accumulating-kernel-safe InputGenerator (swaps out-slot for fresh zeroed u64/f32 buffer per benchmark rep)"
  - "BUILD_TUNER + BUILD_PSET + build_pset_tunable_set + launch_build_at — the build-P autotune machinery"
  - "force_row_partition() LGBM_AUTOTUNE_FORCE_P seam (consumed by 13-04 parity gate)"
  - "resident_raw_build_into wired to default-on autotune for the build row-partition P (BOTH live resident classes)"
affects: [13-04-parity-bench]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "fresh-output InputGenerator for ACCUMULATING autotune kernels (vs CloneInputGenerator for OVERWRITE kernels) — spike-038 classification"
    - "rebuild the TunableSet fresh per call (Arc::new), NOT LocalTuner::init — init memoizes by closure TYPE-id and would freeze the first call's dimensions; the persistent winner lives in the tuner's per-key state instead"
    - "macro_rules launch threaded with an explicit $p so FORCE_P/fallback DIRECT launches stay byte-identical while the tuner reuses the same kernel via a handle-slice launcher"

key-files:
  created: []
  modified:
    - crates/lgbm-compute/src/kernels/histogram.rs

key-decisions:
  - "Only resident_raw_build_into (the funnel for both live resident builds) is wired; sites 982/1543/1692 stay on the heuristic per the locked cold-start-fallback decision (non-silent inline comment)"
  - "BUILD_PSET = {1,4,8,16,32} clamped to ROWPART_P_MAX (=16) — deliberately coarse; spike-040 says the P4..P16 curve is flat, the only job is to AVOID P1"
  - "LaunchKey.bins = max_w/2 (widest feature's bin count) — the per-feature slot-width driver"

patterns-established:
  - "Pattern: an accumulating GPU kernel under cubecl autotune REQUIRES a fresh-output InputGenerator; a grad-conservation assert (Σgrad == feats·Σord_g) guards it"
  - "Pattern: per-call-varying launch dimensions ⇒ rebuild the TunableSet each call + let the LocalTuner key-state carry the persistent winner (do not use init memoization)"

requirements-completed: [AT-BUILD-P]

# Metrics
duration: 18min
completed: 2026-06-26
status: complete
---

# Phase 13 Plan 02: GPU Autotune Build-P Summary

**CubeCL autotune is now the default-on rocm selector for the histogram-build row-partition `P`, wired at the single shared `resident_raw_build_into` funnel (both the f32-resident and u64 fixed-point resident builds), with a fresh-output benchmark generator, an `LGBM_AUTOTUNE_FORCE_P` pin, and the `row_partition_count` heuristic kept as the `LGBM_AUTOTUNE=0` cold-start fallback.**

## Performance

- **Duration:** ~18 min
- **Tasks:** 2
- **Files modified:** 1 (`crates/lgbm-compute/src/kernels/histogram.rs`)

## Accomplishments

- **Task 1 — autotune machinery (commit `300bba0`):** Added `FreshOutGenerator<R>` (the spike-038 fix: an `InputGenerator<LaunchKey, Vec<Handle>>` that hands each cold-benchmark rep a fresh zeroed `out` handle at slot index 5 — `u64` zeros for the fixed-point build, `f32` for the f32 build — so the ACCUMULATING build never corrupts the real `out`), `BUILD_TUNER` (`local_tuner!("build")`), `BUILD_PSET = {1,4,8,16,32}` (clamped to `ROWPART_P_MAX`), `launch_build_at<R>` (a single handle-slice launcher mirroring the resident LDS macros — u64/f32 × u8/u16/u32 dispatch), and `build_pset_tunable_set<R>` (one `Tunable` per `P`, keyed on `LaunchKey{ size_band(rows), feats, bins }`). Plus `force_row_partition()` reading `LGBM_AUTOTUNE_FORCE_P` fresh (clamped `[1, ROWPART_P_MAX]`).
- **Task 2 — wiring (commit `f972f47`):** Replaced the lone `row_partition_count` pick in the LDS branch of `resident_raw_build_into` with a three-way selection: (a) `LGBM_AUTOTUNE_FORCE_P=k` → pinned single-`P` direct launch; (b) default → `BUILD_TUNER.execute` over `BUILD_PSET`; (c) `LGBM_AUTOTUNE=0` → the existing heuristic + direct launch, byte-for-byte unchanged. The `launch_lds_*` macros now thread `$p` explicitly so the direct paths stay byte-identical. Because `resident_raw_build_into` is the shared funnel for BOTH the f32-resident path (`build_leaf_histograms_resident_f32_on`, reached via `Backend::build_leaf_histograms_raw`) AND the u64 fixed-point device-resident pool (`build_fix_compact_resident_f64_on`), wiring this one site puts every steady-state GPU histogram build under autotune.
- **Tests (rocm, green):** `build_tuner_grad_conservation_fresh_vs_clone` (fresh arm `rel_err ~0`; `CloneInputGenerator` control inflated ≫1×, proving the generator is load-bearing) and `build_tuner_u64_bit_identical_across_p` (the u64 fixed-point build is bit-identical for P ∈ {1,4,8,16} — parity-neutral integer merge).

## Verification

- `cargo test -p lgbm-compute --features rocm build_tuner` → 2 passed (grad-conservation + u64 cross-P bit-identity).
- `cargo test -p lgbm-compute --features rocm` resident anchor parity — **both modes green**:
  - Default-on autotune: `rocm_cuda_mirror` (resident + dense vs CPU anchor within tol) 4 passed, `rocm_backend_parity` 4 passed, `rocm_row_partition` 2 passed.
  - `LGBM_AUTOTUNE=0`: `cuda_mirror_resident_matches_cpu_anchor_within_tol` passed (heuristic fallback reproduces prior path).
  - `LGBM_AUTOTUNE_FORCE_P=8`: resident anchor parity passed (the pin seam stays in-gate).
- CPU merge gate untouched: `cargo test -p lgbm-treelearner --lib` → 77 passed (rocm-gated wiring only; the f64 anchor is unaffected).
- Default (cpu, no-rocm) build of `lgbm-compute` still compiles (the new code is `#[cfg(feature = "rocm")]`).

## Scope boundary (non-silent — the four `row_partition_count` sites)

Per the locked CONTEXT decision ("autotune is the default; the heuristic is only the cold-start / cache-miss fallback bound"), an inline comment at the wired site names all four sites:

- **1791 `resident_raw_build_into` — WIRED** (the live steady-state build for both resident classes).
- **982 `build_leaf_histograms_batched_f32_on` — NOT wired.** Production-reachable only as the cache-empty defensive COLD fallback in `build_leaf_histograms_raw` (lib.rs), taken only when `upload_resident_bins` was never called. Uses a different host-gather batched LDS kernel; it IS the cold-start case the locked decision designates for the heuristic. Deliberate, deferred (a separate batched tuner is out of scope).
- **1543 / 1692 `construct_histograms_cuda_mirror_on` / `_resident_on` — NOT wired.** Test/example-only launchers; no production path.

All three non-wired sites remain textually unchanged.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] `read_one_unchecked` returns `Bytes`, not `Vec<u8>`**
- **Found during:** Task 1 (compiling the `build_tuner_u64_bit_identical_across_p` test)
- **Issue:** the `run_at` test closure annotated `-> Vec<u8>` but `client.read_one_unchecked(out)` returns `cubecl::bytes::Bytes` (E0308).
- **Fix:** append `.to_vec()` to materialize the bytes for `assert_eq!` comparison.
- **Files modified:** crates/lgbm-compute/src/kernels/histogram.rs (test only)
- **Committed in:** `300bba0` (Task 1)

**Total deviations:** 1 auto-fixed (1 bug, test-only). Production code matched the plan exactly.

## Decisions Made

- **Rebuild the TunableSet fresh each call (`Arc::new(build_pset_tunable_set(..))`), NOT `LocalTuner::init`.** `init` memoizes the set by the initializer-closure TYPE-id and runs once per process — it would freeze the FIRST call's `rows`/`num_features`/`slot_len` into the launch closures forever. Rebuilding is cheap (closures only); the persistent winner survives in `BUILD_TUNER`'s `LaunchKey`→`fastest_index` state, and the fixed `BUILD_PSET` registration order keeps `fastest_index`→`P` stable across rebuilds.
- **Keep the `launch_lds_*` macros for the direct paths** (FORCE_P / fallback) and add a parallel `launch_build_at` for the tuner closures (which only have a `Vec<Handle>`). The two intentionally launch byte-identical kernels; a sync note marks them.
- **`LaunchKey.bins = max_w/2`** — the widest feature's bin count, the per-feature slot-width driver, so the occupancy regime captures the per-feature LDS pressure.

## Threat mitigations applied

- **T-13-02-01 (tamper `LGBM_AUTOTUNE_FORCE_P`):** parsed + `clamp(1, ROWPART_P_MAX)`; non-numeric / `0` / unset → `None` (falls through), never a no-launch.
- **T-13-02-02 (benchmark reps corrupt the accumulating build):** `FreshOutGenerator` isolates the benchmark `out`; the `build_tuner_grad_conservation` test is the standing guard.
- **T-13-02-03 (poisoned cache picks a different P):** accepted — every PSET variant is parity-gated to the CPU f64 anchor (13-04); a wrong pick is at worst a perf regression.

## Known limitations / honest bound

- Absolute wall-clock is APU-confounded (spoofed 8-CU APU); the durable deliverable is the SELECTION method (measure-don't-model) + portability to real discrete GPUs, not a local e2e train-time win (the 16-core CPU still beats this GPU build end-to-end here).
- The grad-conservation/u64 tests use a unique per-run cache namespace id to force a cold tune (so the benchmark reps that expose the fresh-vs-clone difference always run); this writes small throwaway `target/autotune/...` log files per run.

## Next Phase Readiness

- 13-04 (all-variants parity bench) can now pin each `P` via `LGBM_AUTOTUNE_FORCE_P=k` and assert every `BUILD_PSET` variant against the CPU f64 anchor, and toggle `LGBM_AUTOTUNE=0` to exercise the heuristic fallback.

## Self-Check: PASSED

- `crates/lgbm-compute/src/kernels/histogram.rs` present; all six symbols (`FreshOutGenerator`, `BUILD_TUNER`, `BUILD_PSET`, `build_pset_tunable_set`, `launch_build_at`, `force_row_partition`) verified by grep.
- Task commits verified in git history: `300bba0` (Task 1), `f972f47` (Task 2).

---
*Phase: 13-gpu-autotune-launch-config*
*Completed: 2026-06-26*
