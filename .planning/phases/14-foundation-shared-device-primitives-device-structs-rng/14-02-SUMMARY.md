---
phase: 14-foundation-shared-device-primitives-device-structs-rng
plan: 02
subsystem: oracle-harness / device-primitive golden capture
tags: [hip, hipcc, cuda-algorithms, golden-fixtures, xtask, ODL-01, D-03]
requires:
  - "xtask C++ capture-harness pattern (kernel_capture.cpp / regen driver)"
  - "in-repo AMD fork LightGBM-release-4.6.0.99 (device-primitive signatures)"
  - "local ROCm hipcc 7.1 + spoofed gfx1100 APU"
provides:
  - "xtask/cpp/primitive_capture.cu — self-contained HIP golden-capture harness"
  - "primitive-capture xtask subcommand (wired into dispatch + regen)"
  - "crates/oracle-harness/fixtures/primitives/{prefix_sum,reduce,argsort,percentile}.txt"
  - "REFERENCE_MANIFEST.md device-primitive section + PRIMITIVE_MASTER_SEED"
affects:
  - "14-06 (Rust CubeCL device-primitive replay test consumes these goldens)"
tech-stack:
  added: ["hipcc (ROCm 7.1) HIP build target via CMake custom target"]
  patterns: ["verbatim device-primitive transcription + __global__ shim per __device__ helper", "MASTER_SEED-driven byte-idempotent capture", "off-cargo-build dev harness"]
key-files:
  created:
    - "xtask/cpp/primitive_capture.cu"
    - "crates/oracle-harness/fixtures/primitives/prefix_sum.txt"
    - "crates/oracle-harness/fixtures/primitives/reduce.txt"
    - "crates/oracle-harness/fixtures/primitives/argsort.txt"
    - "crates/oracle-harness/fixtures/primitives/percentile.txt"
  modified:
    - "xtask/cpp/CMakeLists.txt"
    - "xtask/src/main.rs"
    - "crates/oracle-harness/fixtures/REFERENCE_MANIFEST.md"
key-decisions:
  - "Self-contained verbatim transcription of the AMD-fork device primitives (no external_libs / lib_lightgbm link), mirroring the sibling C++ harnesses — external_libs are empty here."
  - "Weighted PercentileDevice is non-idempotent on the spoofed gfx1100 APU (reference block-cooperative UB); its cases are emitted as deterministic Kaggle/nvcc deferral markers (env-gated real capture) rather than dropped, per the plan's fallback guidance."
requirements-completed: [ODL-01]
duration: 18 min
completed: 2026-06-29
---

# Phase 14 Plan 02: Device-Primitive Golden-Fixture Capture Path Summary

Built a self-contained C++/HIP (`hipcc`) golden-capture harness that launches every numeric LightGBM device primitive — block inclusive/exclusive prefix-sum + the multi-kernel global prefix-sum, the shuffle reductions sum/max/min + dot-product, single- and multi-block bitonic argsort (index-only, incl. a tie-rich input), and `PercentileDevice` — verbatim-transcribed from the in-repo AMD fork, plus the `primitive-capture` xtask subcommand that drives it and the committed byte-idempotent goldens 14-06 replays against.

## What was built

- **`xtask/cpp/primitive_capture.cu`** (Task 1) — a HIP harness verbatim-transcribing the AMD-fork `cuda_algorithms.hpp` `__device__` block helpers (`ShufflePrefixSum`/`Exclusive`, `ShuffleReduceSum/Max/Min`, `BitonicArgSortDevice`, `ShuffleSortedPrefixSumDevice`, `PercentileDevice`) and the `cuda_algorithms.cu` multi-kernel global wrappers (`ShufflePrefixSumGlobal`, `BitonicArgSortGlobal` + its compare/merge kernels), wrapping each pure `__device__` block helper in a one-line `__global__` shim. No `external_libs` / `lib_lightgbm` link (they are empty here) — only the header-only `LightGBM::Random` for synthetic inputs. All inputs derive from one argv `MASTER_SEED`.
- **CMake target + xtask subcommand** (Task 2) — a `find_program(hipcc)`-gated custom CMake target (off the global `enable_language(HIP)` so the sibling C++ targets stay buildable without HIP; arch overridable via `-DLGBM_HIP_ARCH`, default `gfx1100`), and a `primitive-capture` subcommand mirroring `kernel_capture()` (cmake configure → build → run with `PRIMITIVE_MASTER_SEED` → write goldens → refresh manifest), wired into the dispatch match, usage strings, and `regen()` (best-effort: missing hipcc/GPU skips with a notice, never fails regen).
- **Committed goldens** under `crates/oracle-harness/fixtures/primitives/`: `prefix_sum.txt` (12 block + 4 global cases), `reduce.txt` (24 cases = sum/max/min/dot × f64+f32 × 3 lengths), `argsort.txt` (13 cases: single+multi-block, asc+desc, incl. 3 tie-rich), `percentile.txt` (9 unweighted goldens + 9 weighted deferral markers). `PRIMITIVE_MASTER_SEED = 0x0DE71CE5` (233250021).

## Verification results

- `cargo build -p xtask` — clean.
- `cargo run -p xtask -- primitive-capture` — builds the harness via hipcc on the gfx1100 APU and writes all 4 fixtures; second run → empty `git diff` on `crates/oracle-harness/fixtures/primitives/` (byte-idempotent, confirmed over 20 consecutive captures during development).
- Fixtures listed in `REFERENCE_MANIFEST.md` (new "Device-Primitive Golden Set" section).
- Argsort permutations independently validated: every case is a valid permutation AND monotonic under its order, for all lengths (7/64/128/200/300/1500/2500/3000).
- Harness is OUT of the cargo build graph (a `.cu` built only by the hipcc CMake target; `cargo build` of every crate is unaffected).

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Map `__shfl_*_sync` onto the AMD-fork's `__shfl_*`**
- **Found during:** Task 1 (first hipcc compile).
- **Issue:** ROCm 7.1's native `__shfl_up_sync`/`__shfl_down_sync` `static_assert` a 64-bit mask; the transcribed primitives pass the full `0xffffffff` (32-bit) mask → compile error.
- **Fix:** Mirror the AMD fork's `cuda_rocm_interop.h:45-48`, mapping the `_sync` variants onto plain `__shfl_up`/`__shfl_down` (mask discarded — always full). Faithful to the reference's own HIP shim.
- **Files modified:** `xtask/cpp/primitive_capture.cu`.
- **Verification:** clean hipcc compile.
- **Commit:** bf24bef.

**2. [Rule 1 - Bug] Weighted prefix-sum base + argsort padding-lane UB (idempotency)**
- **Found during:** Task 1 (idempotency testing).
- **Issue:** Two reference UB sites broke byte-idempotency on the gfx1100 APU: (a) `ShuffleSortedPrefixSumDevice` reads `shared_buffer[threadIdx.x]` (sized `WARPSIZE=32`) with `threadIdx.x` up to `blockDim-1=255` — an OOB LDS read; (b) `BitonicArgSortDevice` leaves padding-lane (`index >= len`) `shared_values` uninitialised.
- **Fix:** Use the exclusive-prefix RETURN value as the per-thread base (the algorithm's intent), and sentinel-init the argsort padding lanes. Both restore determinism without changing any real-element result — verified the committed argsort goldens are byte-identical with and without the sentinel (the `other_data_index < len` guard isolates real elements).
- **Files modified:** `xtask/cpp/primitive_capture.cu`.
- **Verification:** 20/20 consecutive captures byte-identical (post-fix).
- **Commit:** bf24bef.

**3. [Rule 3 - Blocker / plan-sanctioned fallback] Weighted PercentileDevice non-idempotent on the spoofed APU**
- **Found during:** Task 1 (idempotency testing).
- **Issue:** Even after fix #2, the weighted `PercentileDevice` path remains intermittently non-idempotent on the spoofed gfx1100 APU for the golden lengths — its block-cooperative bitonic sort (`BLOCK_DIM=256`, `MAX_DEPTH=9`) + sorted-prefix-sum carry `len`-vs-`BLOCK_DIM`/`MAX_DEPTH` preconditions the synthetic lengths violate (small `len` drives `depth` negative → OOB; residual drift even at valid `len`). Committing nondeterministic values would violate the byte-idempotency acceptance gate.
- **Fix (per plan guidance "mark its fixture for Kaggle/nvcc fallback rather than dropping the case", RESEARCH A3):** the full weighted path stays compiled + runnable behind `LGBM_PRIMITIVE_WEIGHTED_PERCENTILE=1` (for a future Kaggle/nvcc capture), but by default the weighted cases are emitted as deterministic `status=deferred_kaggle_nvcc` marker records (seed-derived inputs + a note, no flaky values). The UNWEIGHTED percentile is fully deterministic and committed.
- **Files modified:** `xtask/cpp/primitive_capture.cu`, `REFERENCE_MANIFEST.md`.
- **Verification:** percentile.txt byte-identical across 20 runs; 9 unweighted goldens + 9 deferral markers.
- **Commit:** bf24bef (harness), 50d867d (markers committed + manifest).

**Total deviations:** 3 auto-fixed (2 Rule-1 reference-UB faithfulness fixes, 1 Rule-3 plan-sanctioned Kaggle/nvcc fallback for the weighted percentile). **Impact:** numeric/index goldens for all primitives are committed and byte-idempotent; the only deferred surface is the weighted percentile (a "skeleton" primitive per 14-RESEARCH), captured later on a CUDA box.

## Known Stubs

- **Weighted `PercentileDevice` goldens** — emitted as `status=deferred_kaggle_nvcc` markers (no committed output values) because the reference path is non-idempotent on the local spoofed gfx1100 APU. Resolved by a future capture on a CUDA box via `LGBM_PRIMITIVE_WEIGHTED_PERCENTILE=1` (RESEARCH A3); the inputs are `PRIMITIVE_MASTER_SEED`-derived so the deferred case is reproducible there. Unweighted percentile + all other primitives are real committed goldens.

## Threat Flags

None — no new network/auth/file-access surface. The harness is an off-cargo-build dev tool reading only the in-repo read-only AMD fork; outputs are committed seed-derived text (T-14-02-01 byte-idempotency mitigated; re-run yields empty `git diff`).

## Next

Ready for the remaining Wave-1 plans / 14-06 (the Rust CubeCL device primitives replay these goldens). The committed fixtures need no toolchain at `cargo test` time; only `primitive-capture`/`regen` need hipcc + a HIP GPU.

## Self-Check: PASSED
- Created files exist on disk: `xtask/cpp/primitive_capture.cu`, `crates/oracle-harness/fixtures/primitives/{prefix_sum,reduce,argsort,percentile}.txt` — all present.
- Commits exist: `bf24bef` (harness), `50d867d` (subcommand + goldens) — both in `git log`.
- Plan verification re-run post-commit: `cargo build -p xtask` clean; `cargo run -p xtask -- primitive-capture` → empty `git diff` on the fixtures.
