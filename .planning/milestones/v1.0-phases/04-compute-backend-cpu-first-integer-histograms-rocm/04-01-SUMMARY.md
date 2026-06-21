---
phase: 04-compute-backend-cpu-first-integer-histograms-rocm
plan: 01
subsystem: infra
tags: [cubecl, cubecl-cpu, gpu, histograms, thiserror, determinism, rocm]

# Dependency graph
requires:
  - phase: 01-oracle-contract-foundations
    provides: lgbm-compute kernel-free Backend skeleton (CMP-01 seam), thiserror boundary-error idiom, oracle-harness comparators
  - phase: 02-dataset-binning-determinism-root
    provides: binned columnar store (Bin trait) as the eventual histogram-kernel input
provides:
  - ComputeError thiserror boundary type (BinIndexOutOfRange/LengthMismatch/CapabilityUnavailable/Runtime)
  - runtime.rs cpu/rocm runtime selection + startup capability gate (Capabilities/ReducePath/probe_capabilities)
  - minimal construct_hist_kernel #[cube(launch)] single-owner ordered f64 fold + construct_histograms_cpu host launcher
  - D-04a bit-determinism spike (empirically PROVEN — cubecl-cpu fold is bit-exact)
  - CMP-01 containment guard test (no upper crate names cubecl)
  - Cargo feature wiring (cpu default, rocm opt-in via cubecl/hip)
affects: [04-02, 04-03, 04-04, phase-05-tree-learner]

# Tech tracking
tech-stack:
  added: [cubecl-cpu (via cubecl/cpu feature), cubecl-hip (via opt-in rocm feature)]
  patterns:
    - "single-owner ordered f64 fold (CubeDim::new_1d(1)) as the cubecl-cpu deterministic anchor"
    - "startup capability gate via client.features()/client.properties() with asymmetric cpu/hip matrix"
    - "V5 boundary validation before unsafe kernel launch (typed ComputeError, never panic/UB)"
    - "CMP-01 containment guard test greps upper-crate Cargo.toml + src for cubecl"

key-files:
  created:
    - crates/lgbm-compute/src/error.rs
    - crates/lgbm-compute/src/runtime.rs
    - crates/lgbm-compute/src/kernels/mod.rs
    - crates/lgbm-compute/src/kernels/histogram.rs
    - crates/lgbm-compute/tests/determinism_spike.rs
    - crates/lgbm-compute/tests/capability.rs
    - crates/lgbm-compute/tests/cmp01_containment.rs
  modified:
    - crates/lgbm-compute/Cargo.toml
    - crates/lgbm-compute/src/lib.rs
    - Cargo.lock

key-decisions:
  - "D-04a SETTLED: cubecl-cpu single-owner ordered f64 fold is bit-stable across 25 launches AND bit-exact vs the C++-order sequential fold — the bit-exact anchor assumption holds, no fallback to ~1e-6 needed"
  - "Backend trait kept in lib.rs (not moved to runtime.rs) — minimal change to the Phase-1 skeleton; runtime.rs owns runtime selection + capability gate only"
  - "construct_histograms_cpu allocates `out` from an explicit zero slice (not client.empty()) because empty() returns recycled uninitialized pool memory"

patterns-established:
  - "Pattern 1: kernels transcribe the C++ body verbatim (dense_bin.hpp:120-135) inside a #[cube] scaffold; f32 read / f64 accumulate (hist_t=double)"
  - "Pattern 2: every divergent device capability is gated explicitly; Sequential IS the cpu path, not a fallback"
  - "Pattern 3: all cubecl unsafe confined to lgbm-compute with a documented handle/length-correspondence safety comment"

requirements-completed: [CMP-01, CMP-02, CMP-04]

# Metrics
duration: 8min
completed: 2026-06-05
---

# Phase 4 Plan 01: Compute Foundation + D-04a Determinism Spike Summary

**Settled the load-bearing D-04a bet empirically (cubecl-cpu single-owner ordered f64 fold is bit-exact across 25 launches AND vs a C++-order sequential fold), and wired the ComputeError boundary, cpu/rocm runtime selection + capability gate, the minimal construct_histograms cube kernel, and the CMP-01 containment guard.**

## Performance

- **Duration:** 8 min
- **Started:** 2026-06-05T18:32:40Z
- **Completed:** 2026-06-05T18:40:48Z
- **Tasks:** 3
- **Files modified:** 10 (7 created, 3 modified)

## Accomplishments
- **D-04a empirically proven (the phase keystone):** the cubecl-cpu single-owner ordered f64 fold (`CubeDim::new_1d(1)`) produces byte-identical f64 histograms across 25 repeated launches and is bit-exact (`to_bits()`) versus a hand-computed sequential f64 fold in C++ `dense_bin.hpp` order. The bit-exact anchor assumption holds — 04-02/04-03 may build the full kernel suite on it without the ~1e-6 fallback.
- **ComputeError boundary type** with the four V5 validation variants; `lgbm-compute` now depends on `lgbm-core`/`lgbm-dataset`/`thiserror`, with `cpu` default and `rocm` opt-in.
- **Startup capability gate (CMP-04)** reporting the verified asymmetric cpu matrix (`has_plane=false`, `has_f64=true`, `has_f32_atomic=false`, `plane_size=1` → `ReducePath::Sequential`).
- **CMP-01 containment guard** that fails if any crate above `lgbm-compute` names `cubecl` in `Cargo.toml` or non-comment `src/`.
- **CPU-only build needs no ROCm toolchain** — the `rocm` runtime path is behind `#[cfg(feature = "rocm")]`.

## Task Commits

Each task was committed atomically:

1. **Task 1: Wire deps/features + ComputeError boundary type** - `0021114` (feat)
2. **Task 2: Runtime selection + startup capability gate (CMP-04)** - `ef69c91` (feat)
3. **Task 3: D-04a bit-determinism spike — histogram kernel + N-launch proof** - `c9669f6` (feat, tdd)

_TDD note: Task 3's first spike run failed loudly (red), which surfaced a real memory-recycling bug; the fix (zero-init `out`) turned it green._

## Files Created/Modified
- `crates/lgbm-compute/Cargo.toml` - cubecl `cpu` feature default, `rocm` opt-in; lgbm-core/lgbm-dataset/thiserror deps; oracle-harness dev-dep
- `crates/lgbm-compute/src/lib.rs` - module decls (error/runtime/kernels), `pub use ComputeError`, `Backend::Runtime: cubecl::Runtime`
- `crates/lgbm-compute/src/error.rs` - ComputeError thiserror enum (4 V5 variants)
- `crates/lgbm-compute/src/runtime.rs` - Capabilities/ReducePath, probe_capabilities, cpu_client / rocm_client(cfg)
- `crates/lgbm-compute/src/kernels/mod.rs` - kernels module (histogram)
- `crates/lgbm-compute/src/kernels/histogram.rs` - construct_hist_kernel #[cube] + construct_histograms_cpu host launcher with V5 validation
- `crates/lgbm-compute/tests/determinism_spike.rs` - D-04a spike (25-launch invariance + sequential-fold bit-exact + boundary validation)
- `crates/lgbm-compute/tests/capability.rs` - asserts the cpu capability matrix
- `crates/lgbm-compute/tests/cmp01_containment.rs` - CMP-01 guard

## Decisions Made
- **D-04a resolved to PASS (no fallback).** The cubecl-cpu fold is bit-stable; the ~1e-6 anchor relaxation is NOT needed and a separate scalar reference is NOT required.
- **Kept `Backend` in `lib.rs`** rather than moving it into `runtime.rs` (PATTERNS.md left the seam location to discretion) — minimal diff to the Phase-1 skeleton.
- **Zero-initialize `out` via `create_from_slice(&zeros)`** instead of `client.empty()` — see Deviations.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] `client.empty()` returns recycled, uninitialized device memory**
- **Found during:** Task 3 (determinism spike — RED run)
- **Issue:** The kernel accumulates (`out[ti] += ...`), so `out` must start zeroed. The plan's reference pattern used `client.empty()`, which returns UNINITIALIZED memory from the cubecl pool. The pool recycled the previous launch's `h_out` buffer, so the second launch folded on top of stale values (cell 0 = 109.0 on launch 0, 217.0 on launch 1). The spike caught this as an apparent "nondeterminism" — but it was a memory-init bug, not a fold-order failure. The fold itself was correct and deterministic.
- **Fix:** Allocate `out` from an explicit zero slice (`client.create_from_slice(f64::as_bytes(&vec![0.0; out_len]))`) so each launch starts from a zeroed histogram, matching the C++ histogram being zeroed before accumulation.
- **Files modified:** crates/lgbm-compute/src/kernels/histogram.rs
- **Verification:** After the fix, 25 launches are byte-identical and bit-exact vs the sequential fold; `cargo test -p lgbm-compute determinism_spike` green.
- **Committed in:** `c9669f6` (Task 3 commit)

---

**Total deviations:** 1 auto-fixed (1 bug)
**Impact on plan:** The fix is required for correctness (and is exactly the class of latent bug the spike was designed to surface). No scope creep. The D-04a conclusion is unaffected and strengthened — the fold is genuinely bit-deterministic.

## Issues Encountered
- **cubecl 0.10.0 API differs from the plan's pseudocode in minor ways** (verified against vendored source): kernel launch is `kernel::launch(&client, CubeCount::Static(1,1,1), CubeDim::new_1d(1), ArrayArg...)` (non-generic kernel takes no turbofish); buffers via `client.create_from_slice(T::as_bytes(&v))`; `Array::len()` returns `usize` so bin values must be widened (`binned[i] as usize * 2`) before indexing `out`. Capability queries use `client.features().plane.contains(Plane::Ops)`, `client.features().supports_type(f64_type)`, `client.properties().atomic_type_usage(f32_atomic).contains(AtomicUsage::Add)`, and `client.properties().hardware.plane_size_max`. All resolved against the vendored 0.10.0 source.

## User Setup Required
None - no external service configuration required. (ROCm bring-up is deferred to 04-04; the CPU gate needs no ROCm toolchain.)

## Next Phase Readiness
- **04-02 (full histogram kernel + golden capture) is unblocked:** the bit-exact cubecl-cpu anchor is proven, the `ComputeError`/runtime/capability foundation is in place, and the minimal histogram kernel + host-launch idiom are established to extend.
- **No blockers.** ROCm bring-up (04-04) remains best-effort per D-03; this plan did not touch the `rocm` feature beyond cfg-gating it (verified the default build needs no ROCm toolchain).

## Self-Check: PASSED

All 7 created files verified present on disk; all 3 task commits (`0021114`, `ef69c91`, `c9669f6`) verified in git history. `cargo test --workspace` green.

---
*Phase: 04-compute-backend-cpu-first-integer-histograms-rocm*
*Completed: 2026-06-05*
