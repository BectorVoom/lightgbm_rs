---
phase: 01-oracle-contract-foundations
plan: 01
subsystem: testing
tags: [rust, cargo-workspace, cubecl, rng, lcg, oracle-harness, thiserror, anyhow, edition-2024]

# Dependency graph
requires: []
provides:
  - "Cargo virtual workspace (edition 2024): lgbm-core, lgbm-compute skeleton, oracle-harness, xtask"
  - "lgbm-core::types — f32 ScoreT/LabelT + meta.h constant contract (K_EPSILON, K_ZERO_THRESHOLD, K_MIN_SCORE, K_MAX_SCORE, NO_SPECIFIC, K_ALIGNED_SIZE)"
  - "lgbm-core::random — bit-for-bit u32 port of C++ LightGBM::Random LCG (both Sample branches)"
  - "lgbm-core::error — thiserror domain enums CoreError + ConfigError (boundary errors)"
  - "lgbm-compute::Backend — CubeCL isolation seam trait skeleton (CMP-01, no kernels)"
  - "oracle-harness::comparator — abs-diff ~1e-6 comparator with first-offending-index reporting (ORA-01)"
  - "oracle-harness fixtures — committed randomized C++ RNG golden set (512 cases) + pinned reference manifest (ORA-02, D-14)"
  - "xtask regen — dev-only C++ capture/regen deriving the randomized case set from one recorded master seed"
affects: [config, binning, predict, compute, treelearner, gbdt, objective]

# Tech tracking
tech-stack:
  added: [thiserror-2.0, anyhow-1.0, cubecl-0.10, rust-1.95-edition-2024]
  patterns:
    - "Virtual workspace (no root package, resolver=3, workspace.dependencies pinning)"
    - "Oracle harness: committed C++ golden fixtures replayed by Rust at test time with no C++ toolchain (D-06)"
    - "Randomized-at-capture goldens derived deterministically from one recorded master seed (idempotent regen, D-14)"
    - "CubeCL confined to lgbm-compute behind a single Backend seam (CMP-01)"

key-files:
  created:
    - Cargo.toml
    - rust-toolchain.toml
    - crates/lgbm-core/src/types.rs
    - crates/lgbm-core/src/error.rs
    - crates/lgbm-core/src/random.rs
    - crates/lgbm-compute/src/lib.rs
    - crates/oracle-harness/src/comparator.rs
    - crates/oracle-harness/tests/rng_parity.rs
    - crates/oracle-harness/tests/comparator.rs
    - crates/oracle-harness/fixtures/rng_sequence.txt
    - crates/oracle-harness/fixtures/REFERENCE_MANIFEST.md
    - xtask/src/main.rs
    - xtask/cpp/rng_capture.cpp
    - xtask/cpp/CMakeLists.txt
  modified:
    - xtask/cpp/CMakeLists.txt
    - xtask/src/main.rs

key-decisions:
  - "Header-only C++ RNG capture: compile rng_capture directly against include/LightGBM/utils/random.h instead of linking lib_lightgbm (external_libs submodules not vendored). Numerically identical reference source."
  - "Master seed 1592594996 fixed as compile-time constant; 512 cases (256 RNG + 256 Sample) regenerated idempotently."

patterns-established:
  - "Pattern 1: Walking-skeleton parity spine — every later numeric layer validates against the same committed-golden oracle seam."
  - "Pattern 2: RNG draws compared for exact equality (integer + exact-bit f32); ~1e-6 tolerance applies to float comparisons only."

requirements-completed: [FND-01, FND-02, FND-03, FND-04, ORA-01, ORA-02]

# Metrics
duration: ~continuation finalize
completed: 2026-06-05
---

# Phase 1 Plan 01: Oracle Contract Foundations Summary

**Bit-for-bit u32 port of the C++ LightGBM Random LCG, proven against a committed 512-case randomized C++ golden set by an abs-diff ~1e-6 oracle harness, atop a Cargo virtual workspace (edition 2024) with the f32 type/constant contract and thiserror/anyhow error layering.**

## Performance

- **Tasks:** 3 (Tasks 1-2 autonomous, Task 3 author + human-action checkpoint resolved)
- **Files created:** 20 tracked workspace files (crates/ + xtask/ + root manifests)
- **Completed:** 2026-06-05

## Accomplishments

- Restructured the repo into a Cargo virtual workspace under edition 2024 (lgbm-core, lgbm-compute skeleton, oracle-harness, xtask); `src/main.rs` removed; `Cargo.lock` + `rust-toolchain.toml` committed.
- Ported `LightGBM::Random` bit-for-bit over `u32` with `wrapping_mul`/`wrapping_add` — RandInt16/RandInt32 recurrence, NextFloat as f32 / 32768.0, and both `Sample(N,K)` branches (streaming + BTreeSet) across the `K > N/log2(K)` boundary.
- Established the f32 numerical contract (ScoreT/LabelT = f32, meta.h constants) and thiserror domain errors (CoreError, ConfigError) at the lgbm-core boundary; anyhow in harness/xtask.
- Built the oracle comparator at the locked ~1e-6 tolerance with first-offending-index reporting (ORA-01).
- Captured a randomized C++ RNG golden set (512 cases: 256 RNG seed sequences + 256 randomized `(N,K)` Sample cases straddling the branch boundary) from one recorded master seed, with `rng_parity` proving the Rust port reproduces every committed case bit-for-bit — with no C++ toolchain needed at normal `cargo test` time.

## Task Commits

1. **Task 1: Virtual workspace + lgbm-core types/errors + lgbm-compute Backend skeleton** - `e5668e5` (feat)
2. **Task 2: Port Random LCG bit-for-bit + oracle abs-diff comparator** - `5c76032` (feat)
3. **Task 3 (authoring): xtask regen + C++ capture harness + manifest scaffold + rng_parity test** - `a330929` (feat)
4. **Task 3 (checkpoint pause record)** - `27be91a` (docs)
5. **Task 3 (deviation): header-only C++ RNG capture** - `8233302` (fix)
6. **Task 3 (capture): randomized C++ RNG golden set + manifest** - `24dc518` (feat)

_Task 3 was a `checkpoint:human-action` gate (building the C++ reference + capturing the golden set). The human step was performed and verified, then the deliverables committed in this continuation._

## Files Created/Modified

- `Cargo.toml` - virtual workspace manifest (no root package, resolver 3, workspace.dependencies)
- `rust-toolchain.toml` - pinned toolchain channel for edition 2024
- `crates/lgbm-core/src/types.rs` - f32 ScoreT/LabelT aliases + meta.h constants
- `crates/lgbm-core/src/error.rs` - thiserror CoreError + ConfigError enums
- `crates/lgbm-core/src/random.rs` - bit-exact u32 Random LCG (both Sample branches)
- `crates/lgbm-compute/src/lib.rs` - Backend trait skeleton (CMP-01 seam, no kernels)
- `crates/oracle-harness/src/comparator.rs` - abs-diff ~1e-6 comparator + ORACLE_TOL + Mismatch
- `crates/oracle-harness/tests/rng_parity.rs` - replays Rust Random against every committed case
- `crates/oracle-harness/tests/comparator.rs` - comparator unit tests
- `crates/oracle-harness/fixtures/rng_sequence.txt` - committed 512-case C++ golden set (master seed 1592594996)
- `crates/oracle-harness/fixtures/REFERENCE_MANIFEST.md` - pinned commit 195c26fc, VERSION 4.6.0.99, flags, master seed, case count
- `xtask/src/main.rs` - dev-only regen subcommand (toolchain check, case generation, manifest text)
- `xtask/cpp/rng_capture.cpp` - standalone C++ capture program over LightGBM::Random
- `xtask/cpp/CMakeLists.txt` - standalone CMake target (header-only against random.h)

## Decisions Made

- **Header-only C++ capture** (see Deviations) — keeps the parity contract intact without the unavailable lib_lightgbm build.
- Master seed `1592594996` fixed as a compile-time constant; 512 cases (256 RNG + 256 Sample) — single deterministic source of randomness so regen is idempotent.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Header-only C++ RNG capture instead of linking lib_lightgbm**
- **Found during:** Task 3 (human-action checkpoint resolution)
- **Issue:** The original plan/CMakeLists assumed building and linking the full `lib_lightgbm`. That is impossible in this environment — the in-repo `LightGBM/external_libs/` submodules (fast_double_parser, fmt, eigen, compute) are not vendored/checked out, so `lib_lightgbm` cannot build (`fast_double_parser.h` missing).
- **Fix:** Since `LightGBM::Random` (`include/LightGBM/utils/random.h`) is a self-contained header-only class (depends only on `<cstdint> <random> <set> <vector>`), the C++ capture was changed to compile `rng_capture` directly against the pinned header — no `add_subdirectory(LIGHTGBM_DIR)`, no lib link.
- **Why valid:** Numerically identical (same exact reference source), so the parity contract (FND-01, ORA-02, D-14) holds fully. The read-only submodule tree is not modified.
- **Files modified:** `xtask/cpp/CMakeLists.txt`, `xtask/src/main.rs` (comment + manifest template text)
- **Verification:** `cargo run -p xtask -- regen` captured 512 cases idempotently (byte-stable across re-runs, empty git diff); `cargo test -p oracle-harness rng_parity` passes — Rust Random reproduces every committed C++ case bit-for-bit.
- **Committed in:** `8233302`

---

**Total deviations:** 1 auto-fixed (1 blocking)
**Impact on plan:** Necessary to complete the capture in this environment; preserves numerical fidelity and the read-only-submodule invariant. No scope creep.

## Issues Encountered

None beyond the deviation above. Full verification ran green end-to-end:
- `cargo build --workspace` — success (edition 2024).
- `cargo test -p lgbm-core` — 14 tests pass (types, error, random/LCG + both Sample branches).
- `cargo test -p oracle-harness` — comparator 5 pass + rng_parity 1 pass (every committed case bit-for-bit).

## User Setup Required

None - no external service configuration required. (The C++ toolchain is needed only for the dev-only `xtask regen`; normal `cargo test` reads the committed fixtures and needs no C++ toolchain, per D-06.)

## Next Phase Readiness

- The parity spine is complete and green: every later numeric layer can now validate against the same falsifiable f32 / ~1e-6 oracle seam over varied distributions.
- Plan 01-02 (config slice) can extend `ConfigError` variants and build on the established workspace + error layering.
- No blockers.

## Self-Check: PASSED

Created files verified present on disk and commits verified in git log:
- `crates/oracle-harness/fixtures/rng_sequence.txt` — FOUND (517 lines, master seed 1592594996, COUNTS rng=256 sample=256)
- `crates/oracle-harness/fixtures/REFERENCE_MANIFEST.md` — FOUND (commit 195c26fc, master seed 1592594996, 512 cases, deterministic/force_row_wise/num_threads flags)
- Commits `e5668e5`, `5c76032`, `a330929`, `27be91a`, `8233302`, `24dc518` — all FOUND in git log.
- Test suites green: lgbm-core 14/14, oracle-harness comparator 5/5 + rng_parity 1/1.

---
*Phase: 01-oracle-contract-foundations*
*Completed: 2026-06-05*
