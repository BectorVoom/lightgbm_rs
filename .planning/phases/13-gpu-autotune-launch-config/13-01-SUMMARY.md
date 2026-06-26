---
phase: 13-gpu-autotune-launch-config
plan: 01
subsystem: infra
tags: [cubecl, autotune, rocm, serde, gpu, launch-config]

# Dependency graph
requires:
  - phase: 11-quantized-training
    provides: "u64 fixed-point resident-build kernel (parity-neutral across P) the build-knob autotunes"
  - phase: spikes-037-040
    provides: "validated cubecl::tune feasibility, FreshOutGenerator, log2 key, vs-heuristic win"
provides:
  - "kernels::autotune module: LaunchKey AutotuneKey (serde + Display), size_band log2 bucketer, autotune_enabled off-switch, cache_namespace_id"
  - "serde promoted to a real rocm-gated optional dependency"
  - "Backend::prefers_autotune_launch_config default-false trait seam + RocmBackend default-on override"
affects: [13-02-build-P, 13-03-scan-W, 13-04-parity-bench]

# Tech tracking
tech-stack:
  added: [serde-1-rocm-gated]
  patterns:
    - "rocm-gated cross-cutting module (#[cfg(feature = rocm)] pub mod) so the cpu hot path pulls no autotune codegen"
    - "default-false Backend trait method as the backend discriminator seam (mirrors prefers_host_partition)"
    - "env off-switch read fresh (not OnceLock) so parity tests can toggle within a process"

key-files:
  created:
    - crates/lgbm-compute/src/kernels/autotune.rs
  modified:
    - crates/lgbm-compute/Cargo.toml
    - crates/lgbm-compute/src/kernels/mod.rs
    - crates/lgbm-compute/src/lib.rs

key-decisions:
  - "LaunchKey is ONE shared shape both knobs reuse: build sets bucket=size_band(rows), scan sets bucket=0"
  - "autotune_enabled() is the single source of truth; the trait method delegates to it (no divergent flag)"
  - "cache_namespace_id() returns the constant rocm:0 (matches AmdDevice::new(0); no public ordinal accessor)"

patterns-established:
  - "Pattern: cross-cutting GPU plumbing lives in one rocm-gated kernels::autotune module so per-knob wirings only touch their own kernel file"
  - "Pattern: size_band(rows)=floor(log2) keys autotune on the occupancy regime not the exact count (avoids per-leaf tuning storm)"

requirements-completed: [AT-SERDE, AT-DEFAULT-ON]

# Metrics
duration: 3min
completed: 2026-06-26
status: complete
---

# Phase 13 Plan 01: GPU Autotune Foundation Summary

**Shared rocm-gated `kernels::autotune` module (LaunchKey serde AutotuneKey + log2 size_band bucketer + LGBM_AUTOTUNE off-switch + rocm:0 cache id), serde promoted to a real rocm-gated dep, and a default-false `Backend::prefers_autotune_launch_config` seam that RocmBackend turns default-on.**

## Performance

- **Duration:** 3 min
- **Started:** 2026-06-26T11:37:43Z
- **Completed:** 2026-06-26T11:40:42Z
- **Tasks:** 3
- **Files modified:** 4 (1 created, 3 modified)

## Accomplishments
- Promoted `serde` from a dev-only dependency to a real `optional = true` dependency wired into the `rocm` feature — the default cpu build compiles and links no serde-derive on the hot path, while `--features rocm` has it available for the persistent-cache AutotuneKey.
- Created `kernels::autotune` (rocm-gated): `LaunchKey` (Clone/Debug/Hash/Eq + serde + `cubecl::tune::AutotuneKey` + Display), `size_band(rows)` log2 occupancy bucketer (spike-039, never panics on 0), `autotune_enabled()` default-on off-switch read fresh per call, and `cache_namespace_id()` → `"rocm:0"`. 5 unit tests green.
- Added the `Backend::prefers_autotune_launch_config` default-false trait method (cpu anchor never autotuned) with the RocmBackend override delegating to `autotune_enabled()` (default-on; `LGBM_AUTOTUNE=0` falls back to the heuristic). CPU merge gate untouched.

## Task Commits

Each task was committed atomically:

1. **Task 1: Promote serde to a real rocm-gated dependency** - `b45b245` (build)
2. **Task 2: Create the kernels::autotune module** - `739553b` (feat)
3. **Task 3: Add the default-on rocm discriminator trait method** - `6408a7d` (feat)

## Files Created/Modified
- `crates/lgbm-compute/src/kernels/autotune.rs` - NEW: shared LaunchKey AutotuneKey, size_band bucketer, autotune_enabled off-switch, cache_namespace_id + 5 unit tests
- `crates/lgbm-compute/Cargo.toml` - serde moved [dev-dependencies] → [dependencies] optional, added to `rocm` feature list
- `crates/lgbm-compute/src/kernels/mod.rs` - `#[cfg(feature = "rocm")] pub mod autotune;`
- `crates/lgbm-compute/src/lib.rs` - `Backend::prefers_autotune_launch_config` default-false + RocmBackend override

## Decisions Made
- **One shared LaunchKey shape** for both launch knobs: the build tunable (13-02) sets `bucket = size_band(rows)`, the scan tunable (13-03) sets `bucket = 0` (feature-per-lane scan is bit-exact and W does not depend on row count).
- **`autotune_enabled()` is the single source of truth.** The trait method delegates to it rather than holding its own flag, so the rocm free-function launch sites (which hold only a `client`, like the existing `scan_cube_dim` / `row_partition_count` patterns) and the trait seam never diverge.
- **`cache_namespace_id()` returns the constant `"rocm:0"`** tied to `rocm_client`'s `AmdDevice::new(0)` — `runtime.rs` exposes no clean device-ordinal accessor and the client is hardcoded to ordinal 0.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Wrapped env mutation in `unsafe` for the Rust 2024 edition**
- **Found during:** Task 2 (autotune unit tests)
- **Issue:** `std::env::set_var` / `remove_var` are `unsafe` in the workspace's Rust 2024 edition; the off-switch test failed to compile (E0133).
- **Fix:** Wrapped the three env calls in `unsafe { … }` blocks with a `// SAFETY: single-threaded unit test` note (matches the established codebase idiom for env-toggling tests).
- **Files modified:** crates/lgbm-compute/src/kernels/autotune.rs
- **Verification:** `cargo test -p lgbm-compute --features rocm --lib autotune` → 5 passed.
- **Committed in:** 739553b (Task 2 commit)

---

**Total deviations:** 1 auto-fixed (1 blocking)
**Impact on plan:** The fix is a test-only edition-conformance adjustment. No scope creep; production code unaffected.

## Issues Encountered
None beyond the deviation above.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- The shared symbols (`LaunchKey`, `size_band`, `autotune_enabled`, `cache_namespace_id`, `prefers_autotune_launch_config`) are in place, so 13-02 (build-P, `histogram.rs`) and 13-03 (scan-W, `split.rs`) can run in parallel touching only their own kernel files.
- 13-02 must add a fresh-output `InputGenerator` (the build kernel ACCUMULATES — spike-038); 13-03's scan OVERWRITES so `CloneInputGenerator` is safe there.
- Verification note: the `--features rocm` builds link and run on this host but absolute wall-clock is APU-confounded (spoofed 8-CU APU); the autotune SELECTION axis is the spoof-robust deliverable.

## Self-Check: PASSED
- All 4 files verified present on disk.
- All 3 task commits verified in git history (b45b245, 739553b, 6408a7d).

---
*Phase: 13-gpu-autotune-launch-config*
*Completed: 2026-06-26*
