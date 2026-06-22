---
phase: quick-260622-k4g
plan: 01
subsystem: infra
tags: [cubecl, rocm, gpu, histogram, subtract-trick, grid-stride, bit-exact, perf]

# Dependency graph
requires:
  - phase: notes/cubecl-profiling-gpu-kernel-decomposition
    provides: CubeCL built-in per-kernel device-time profiler recipe; identified the 1-lane subtract kernel as ~32% of GPU device time
provides:
  - Grid-stride parallel GPU histogram-subtract kernels (f64 live path + f32 mirror)
  - Real-grid launch dims at all three subtract.rs launch sites (no remaining CubeDim::new_1d(1))
  - to_bits() bit-exact parallel-equals-serial tests (f64 + f32, representative + stride-remainder lengths)
  - Re-measured subtract device-time share (~31.6% -> ~2.19% steady-state)
affects: [gpu-histogram-kernel, gpu-perf, subtract-trick]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "1D grid-stride loop for elementwise GPU kernels (stride = CUBE_COUNT_X * CUBE_DIM_X, start = ABSOLUTE_POS), bit-exact-by-construction"
    - "Fixed over-provisioned launch grid (256 x 64 lanes) + while-bound for arbitrary length (no ceil-that-could-be-0)"

key-files:
  created: []
  modified:
    - crates/lgbm-compute/src/kernels/subtract.rs

key-decisions:
  - "Kept a FIXED CubeCount::Static(64,1,1) x CubeDim::new_1d(256) instead of a length-derived ceil — n==0 returns before launch, so a fixed C is safe and avoids a possible 0-cube launch; the while-bound covers any remainder"
  - "Live GPU path is the f64 subtract_hist_kernel (scope correction), not the f32 mirror the note originally profiled; both parallelized identically for consistency"

patterns-established:
  - "Pattern: elementwise device kernels run a grid-stride loop across a real workgroup, never 1-lane UNIT_POS==0"

requirements-completed: [QUICK-260622-k4g]

# Metrics
duration: ~12min
completed: 2026-06-22
---

# Quick 260622-k4g: Grid-stride parallelize the GPU histogram-subtract kernel Summary

**The live GPU `subtract_hist_kernel` (f64) and its f32 mirror now run a 1D grid-stride loop across a real workgroup instead of on a single lane (`UNIT_POS == 0`); proven byte-identical to the serial fold via `to_bits()` tests, and its measured GPU device-time share dropped from ~31.6% to ~2.19% steady-state.**

## Performance

- **Duration:** ~12 min
- **Tasks:** 2
- **Files modified:** 1 (`crates/lgbm-compute/src/kernels/subtract.rs`)

## Accomplishments

- Rewrote both `#[cube(launch)]` kernels (`subtract_hist_kernel` f64 — the LIVE GPU resident/host path — and `subtract_hist_kernel_f32` mirror) from a 1-lane serial `for` loop to a 1D grid-stride loop: `stride = CUBE_COUNT_X * CUBE_DIM_X`, start `i = ABSOLUTE_POS`, `while i < n { out[i] = parent[i] - child[i]; i += stride }`. Each thread owns disjoint indices; the `while` bound guards every write so the grid may over-cover any length.
- Updated all three launch sites in `subtract.rs` from `CubeCount::Static(1,1,1) × CubeDim::new_1d(1)` to `CubeCount::Static(64,1,1) × CubeDim::new_1d(256)` (16384 lanes amply cover ~25k cells; stride loop handles the remainder):
  1. `subtract_histograms_f64_on` (f64 host path)
  2. `subtract_histograms_f64_from_handles_on` (LIVE rocm resident-handle path the GPU bench hits)
  3. `subtract_histograms_f32_on` (f32 host path)
- Added `subtract_parallel_equals_serial_f64` and `_f32` unit tests proving `to_bits()` equality, cell by cell, against a plain serial Rust `parent[i] - child[i]` reference on lengths **25600** (representative: 50 feat × 256 bins × 2) AND **12345** (odd — exercises the stride remainder).
- Bit-exact merge gate held; native CPU anchor (`subtract_histograms_cpu_native`) byte-unchanged.
- Re-measured the subtract device-time share via the CubeCL profiling recipe on the 1M×50 rocm bench and reported it honestly below.

## Task Commits

1. **Task 1: Grid-stride parallelize both subtract kernels + update all launch dims + bit-exact tests** — `6167c75` (perf)
2. **Task 2: Bit-exact merge gate + honest device-time-share re-measurement** — no code change (verification + measurement only; SUMMARY documents results)

## Bit-Exactness Results

`cargo test -p lgbm-compute --lib subtract` — **8 passed, 0 failed**:

| Test | Lengths | Result |
|------|---------|--------|
| `subtract_parallel_equals_serial_f64` | 25600, 12345 | `to_bits()`-equal — PASS |
| `subtract_parallel_equals_serial_f32` | 25600, 12345 | `to_bits()`-equal — PASS |
| `subtract_elementwise` (legacy) | small | PASS |
| `subtract_length_mismatch` (legacy) | — | PASS |
| `subtract_empty_ok` (legacy, n==0 early return) | 0 | PASS |

## Build & Gate Results

| Gate | Command | Result |
|------|---------|--------|
| CPU build | `cargo build --release -p lgbm-compute` | compiled |
| ROCm build | `cargo build --release -p lgbm-compute --features rocm` | compiled |
| Merge gate (treelearner) | `cargo test -p lgbm-treelearner --lib` | **76 passed, 0 failed, 2 ignored** |
| Subtract CPU parity (bit-exact) | `cargo test -p oracle-harness subtract` → `kernel_parity_subtract_bit_exact_on_cpu` | **PASS** |
| Subtract HIP parity (device, tol) | `cargo test -p oracle-harness --features rocm subtract` → `hip::kernel_parity_subtract_within_tol_on_hip` | **PASS** (ran on real gfx-spoofed APU, not skipped) |
| Temp-toml hygiene | `test ! -f cubecl.toml` | **PASS** (deleted; not staged/committed) |

## Device-Time-Share Re-Measurement (honest)

Recipe: temporary repo-root `cubecl.toml` (`[profiling] logger = full → /tmp/cubecl_profiling.log`), then
`LGBM_BENCH_SWEEP=wide LGBM_BENCH_ROWS=1000000 LGBM_BENCH_FEAT=50 LGBM_BENCH_ITERS=1 cargo run --release --features rocm --example bench_gpu_vs_cpu`.
Parsed `/tmp/cubecl_profiling.log` per-kernel, JIT first-call (the single max per kernel, ≈124–176 ms one-time) excluded from steady-state.

| kernel | calls | steady device-time | **share (after)** | share (before, from note) |
|--------|-------|--------------------|-------------------|---------------------------|
| `construct_leaf_hist_resident_lds_kernel` (BUILD) | 120 | 386.4 ms | 75.82% | 52.6% |
| `find_best_splits_fused_kernel` (SCAN) | 236 | 83.8 ms | 16.45% | 11.3% |
| `data_partition_kernel` | 120 | 15.2 ms | 2.99% | 2.7% |
| `fix_compact_kernel` | 120 | 13.0 ms | 2.56% | 1.9% |
| **`subtract_hist_kernel`** (this change) | 116 | **11.2 ms** | **2.19%** | **31.6%** |

**Total steady device-kernel time: 509.7 ms.**

**Result: the subtract share dropped from ~31.6% to ~2.19%** — a ~14× reduction in its share of GPU device time. Per-call steady-state subtract cost is now ~97 µs mean (min 20 µs), versus the single-lane kernel that previously serialized all ~25k cells on one of the device's lanes. As the subtract collapsed, the BUILD kernel's share rose correspondingly (52.6% → 75.82%) — expected, since removing a co-dominant kernel re-weights the remainder toward the now-dominant atomic/memory-bound BUILD; the absolute totals confirm the subtract itself shrank, not just its relative slice.

**No wall-clock win claimed.** Per the plan and the spoofed-8-CU-APU finding, the GPU loses to the multi-threaded CPU anchor overall on this iGPU (bench reported `r1000kf50` GPU median 550 ms); this change is a correctness-shaped GPU-internal cleanup (no kernel runs 1-lane) + a proven device-time-share reduction, valuable on real discrete hardware where the GPU path matters. The deliverable — device-time-share drop + no 1-lane kernel + proven bit-exactness — is met.

## Decisions Made

- **Fixed launch grid over a length-derived ceil:** chose `CubeCount::Static(64,1,1)` (256×64 = 16384 lanes) for all sites. `n == 0` returns before any launch, so a fixed `C` cannot produce a 0-cube launch, and the `while i < n` bound makes the grid robust to any non-multiple length (including the 12345 stride-remainder case). Simpler and over-provisioned vs the ~25k cells.
- **Live path is the f64 kernel (scope correction honored):** the original note profiled the f32 mirror, but the live GPU subtract dispatches the f64 `subtract_hist_kernel` (`subtract_resident` → `_from_handles_on`; `RocmBackend::subtract_histograms` → `_f64_on`). Both kernels were parallelized identically.

## Deviations from Plan

None - plan executed exactly as written. The native CPU anchor (`subtract_histograms_cpu_native`) was left byte-unchanged, the `read_one_unchecked` readback, V5 length checks, `n == 0` early returns, and handle allocation were all preserved as specified.

## Issues Encountered

- **cubecl type mismatch (resolved inline during Task 1):** the initial kernel body mixed cubecl's `u32` built-ins (`ABSOLUTE_POS`, `CUBE_COUNT_X`, `CUBE_DIM_X`) with `parent.len()`, producing an `E0277`/`E0308` build error. Fixed by casting the index/stride/bound to `usize` (`ABSOLUTE_POS as usize`, etc.) — matching the existing grid-stride idiom in `histogram.rs:787`. Not a behavioral deviation; build-blocking type plumbing only.

## Threat Model Disposition

- **T-k4g-01 (OOB write past `len`):** mitigated — `while i < n` bounds every write; `n == 0` returns before launch; SAFETY comments updated to note the grid over-covers but the bound guards.
- **T-k4g-02 (numeric divergence):** mitigated — byte-identical by construction; `to_bits()` tests on two lengths + CPU bit-exact + HIP tolerance parity gates all green.
- **T-k4g-03 (committing temp profiling toml):** mitigated — `cubecl.toml` deleted after measurement; `test ! -f cubecl.toml` passes; not staged/committed.

## Next Phase Readiness

- The GPU subtract kernel is no longer a 1-lane device bottleneck. The remaining GPU device-time is now ~76% BUILD (atomic/memory-bound `construct_leaf_hist_resident_lds_kernel`) and ~16% SCAN — the next candidate levers per the GPU wide-shape findings, though on this spoofed 8-CU APU the GPU still loses to the CPU anchor overall.

## Self-Check: PASSED

- `crates/lgbm-compute/src/kernels/subtract.rs` exists; commit `6167c75` found in git history.
- `260622-k4g-SUMMARY.md` written.
- `ABSOLUTE_POS` present in both kernels (4 refs); zero remaining `new_1d(1)` 1-lane launches.
- No `cubecl.toml` in repo root; not present in commit `6167c75`.

---
*Phase: quick-260622-k4g*
*Completed: 2026-06-22*
