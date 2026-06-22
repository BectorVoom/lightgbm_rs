---
phase: 11-gpu-fixedpoint-int-atomics
plan: 01
subsystem: infra
tags: [cubecl, rocm, gpu, histogram, fixed-point, integer-atomics, u64, hip]

# Dependency graph
requires:
  - phase: 11-gpu-fixedpoint-int-atomics (spike 018/019)
    provides: validated u64 two's-complement fixed-point build twin (build_u64_rp) + host dequant reference
provides:
  - "construct_leaf_hist_resident_lds_kernel_u64<B: Int> — u64 fixed-point resident LDS build kernel"
  - "fix_compact_kernel u64->f64 dequant ((bits as i64)/2^30) replacing the f32->f64 widen"
  - "resident_raw_build_into fixed_point flag selecting the u64 vs f32 LDS kernel"
  - "build_fix_compact_resident_f64_on u64 RAW merge buffer + i64@2^30 overflow guard"
  - "RocmBackend::build_resident_leaf seam doc describing the fixed-point accumulation"
affects: [11-02, 11-03, gpu-histogram, rocm-parity]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "u64 two's-complement fixed-point integer LDS atomics (store i64 bits as u64, wrapping fetch_add) for deterministic GPU histogram accumulation"
    - "quantize round(v*2^30)->i64-bits at build, dequant (bits as i64)/2^30->f64 confined to the fix-compact widen seam"
    - "shared launcher kernel-selection flag (fixed_point) so two callers share one resident-build path at different precisions"

key-files:
  created: []
  modified:
    - crates/lgbm-compute/src/kernels/histogram.rs
    - crates/lgbm-compute/src/lib.rs

key-decisions:
  - "Atomic<u64> with .store(0u64)/.fetch_add(bits) — NEVER Atomic<i64> (cubecl-hip 0.10 lowers Atomic<i64>::store to atomicExch(long long*) which HIP lacks)"
  - "Naive >256-bin fallback stays f32 (CONTEXT Claude's-discretion); a fixed_point caller must keep every feature <=256 bins, enforced by an assert"
  - "Overflow guard scans the leaf grad/hess for max|v| and errors (no silent clamp) when rows*max|v|*2^30 >= i64::MAX"
  - "fix_compact_f64_on (test-only) quantizes its f32 raw to u64 bits to feed the now-shared u64 kernel"

patterns-established:
  - "Fixed-point seam confinement: only the RAW-build merge target + the fix-compact widen change; FixHistogram fold, compact, subtract, scan, move, upload all stay f64 and byte-unchanged"
  - "Build-side SCALE_F32 / dequant-side SCALE_F64 = 2^30 paired constants"

requirements-completed: [SPEC-1, SPEC-4]

# Metrics
duration: ~35min
completed: 2026-06-22
---

# Phase 11 Plan 01: Fixed-Point Int-Atomic Resident Histogram Build Summary

**ROCm resident histogram BUILD now accumulates grad/hess as u64 two's-complement fixed-point (S=2^30) via integer LDS atomics — a deterministic, ~3600x-more-accurate replacement for the f32 atomicAdd CAS-retry path — dequantized to f64 at the fix-compact seam, everything downstream unchanged.**

## Performance

- **Duration:** ~35 min
- **Started:** 2026-06-22 (Phase 11 execution start)
- **Completed:** 2026-06-22
- **Tasks:** 3
- **Files modified:** 2

## Accomplishments
- New `construct_leaf_hist_resident_lds_kernel_u64<B: Int>` u64 fixed-point resident LDS build kernel, a byte-for-byte twin of the f32 original with only the cell type + quantize/store/merge idiom swapped.
- `fix_compact_kernel` consumes a `&Array<u64>` RAW buffer and dequantizes `(bits as i64)/2^30 -> f64` in its (formerly f32->f64 widen) pass; the FixHistogram fold + compact tail are byte-unchanged.
- Documented i64@2^30 overflow guard at the resident-build boundary (returns a typed error, never silently clamps).
- All 15 `kernel_parity` tests pass on real HIP GPU, including the resident-chain HIP tests — the fixed-point path matches the host f64 fold within the ~1e-6 ROCm gate.
- CPU-only build emits ZERO fixed-point codegen (the entire kernel + u64 path is behind `#[cfg(feature = "rocm")]`); the CPU f64 anchor kernels are byte-untouched.

## Task Commits

Each task was committed atomically:

1. **Task 1: New u64 two's-complement fixed-point resident LDS build kernel** - `6ec996e` (feat)
2. **Task 2: u64 RAW buffer + dispatch in resident_raw_build_into; u64->f64 dequant in fix_compact_kernel** - `cc3b040` (feat)
3. **Task 3: Overflow guard + build_resident_leaf seam doc update** - `c95518d` (feat)

## Files Created/Modified
- `crates/lgbm-compute/src/kernels/histogram.rs` - new u64 build kernel, `SCALE_F32` const, `fixed_point` dispatch flag on `resident_raw_build_into`, u64 RAW alloc + overflow guard in `build_fix_compact_resident_f64_on`, u64 dequant + `SCALE_F64` in `fix_compact_kernel`, f32->u64 quantize in the test-only `fix_compact_f64_on`.
- `crates/lgbm-compute/src/lib.rs` - `RocmBackend::build_resident_leaf` seam doc updated to describe the u64 fixed-point accumulation under the ~1e-6 contract.

## u64-vs-f32 RAW-site classification

Per Task 2's audit of the `f32::as_bytes(&zeros...)` RAW sites flagged in PATTERNS:

| Site | Disposition | Reason |
|------|-------------|--------|
| `build_fix_compact_resident_f64_on` RAW merge target (was 2114) | **u64** | The live resident-chain merge target the new dequant consumes. |
| `resident_raw_build_into` `h_g`/`h_h` grad/hess INPUTS | **f32 (unchanged)** | The kernel quantizes them in-kernel; widening them would be wrong. |
| `build_leaf_histograms_resident_f32_on` `h_out` (readback oracle) | **f32 (unchanged)** | Reads back as f32; passes `fixed_point=false`. |
| `fix_compact_f64_on` `raw` input (test-only) | **f32 input, quantized to u64 before upload** | Feeds the now-shared u64 kernel; dequant inverts it. |
| Naive >256-bin resident fallback kernel | **f32 (unchanged)** | LDS-only u64 port; out of scope, asserted unreachable when `fixed_point`. |
| `construct_histograms_lds_f32_on` (817) + batched non-resident launchers (932, 1098) | **f32 (unchanged)** | Separate non-resident seams, out of scope. |

## Overflow-guard bound chosen

Bound: `leaf_rows.len() * max|v| * 2^30 < i64::MAX`, where `max|v|` is the max absolute grad/hess over the leaf's actual rows (a one-pass scan of the rows about to be accumulated — the tightest in-codebase bound, not a passed-in estimate). On violation: `ComputeError::Runtime { detail: "fixed-point histogram accumulation may overflow i64 at S=2^30 (rows x |value| x 2^30 exceeds i64::MAX)" }`. Inline comment cites spike-018's "i64@2^30 safe to ~1e9 rows x |g| <= 8" (SPEC item 3/4). No silent clamp.

## Unchanged-seam confirmation

`subtract_resident`, `move_resident`, `scan_resident_leaf`, `build_fix_scan_resident`, and `upload_resident_bins` are UNCHANGED — `git diff` shows no `+`/`-` lines for them. They consume the post-dequant f64 Handle / u8-u16-u32 bin indices, not histogram cells. The Stage-1 fused `build_fix_scan_resident` (gated OFF, `FUSED_MAX_NUM_DATA = -1`) remains a sequential f64 fold producing a dimensionally-compatible f64 cell layout — not a fixed-point target — so it is unaffected. The f32 `construct_leaf_hist_resident_lds_kernel` original and the CPU anchor kernels are byte-untouched.

## Decisions Made
- **`Atomic<u64>` not `Atomic<i64>`** — the HARD CONSTRAINT from spike-018b/CONTEXT: cubecl-hip 0.10 lowers `Atomic<i64>::store` to `atomicExch(long long*)`, which HIP lacks (compiles, fails at runtime). i64 quantized values are stored BITS-as-u64; wrapping u64 `fetch_add` == two's-complement i64 add. No bias offset (each bin sums a variable row count).
- **Naive fallback stays f32** with a loud assert guarding the `fixed_point` case, per CONTEXT Claude's-discretion. For `max_bin <= 255` (LightGBM default) the LDS branch always fires, so the assert never trips on sane configs.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] `resident_raw_build_into` is shared by two callers at different precisions**
- **Found during:** Task 2 (dispatch swap)
- **Issue:** The plan directed dispatching the u64 kernel in `resident_raw_build_into`'s LDS arm, but that function is ALSO called by `build_leaf_histograms_resident_f32_on` (the f32 readback oracle, a live lib.rs:2161 seam) which allocates an f32 `h_out` and reads it back as f32. An unconditional u64 dispatch would write u64 bits into an f32 buffer and corrupt the readback.
- **Fix:** Added a `fixed_point: bool` parameter to `resident_raw_build_into`; the LDS arm selects the u64 kernel (`true`, live fix-compact chain) or the f32 kernel (`false`, readback oracle). The f32 readback caller passes `false`; `build_fix_compact_resident_f64_on` passes `true`. No architectural restructure — one flag, the LDS/naive decision still lives in one place.
- **Files modified:** crates/lgbm-compute/src/kernels/histogram.rs
- **Verification:** Both callers build; kernel_parity HIP tests (both the f32-readback `kernel_parity_resident_gather_equals_host_gather_on_hip` and the u64 `kernel_parity_resident_build_fix_compact_equals_host_on_hip`) pass.
- **Committed in:** `cc3b040` (Task 2 commit)

**2. [Rule 1 - Bug] Shared `fix_compact_kernel` signature change breaks the test-only `fix_compact_f64_on`**
- **Found during:** Task 2 (kernel signature change)
- **Issue:** Changing `fix_compact_kernel`'s `h_raw` from `&Array<f32>` to `&Array<u64>` broke `fix_compact_f64_on`, a second launcher that takes an externally-built f32 `raw: &[f32]` (kernel_parity test path) and would feed an f32 buffer to the now-u64 kernel — a type mismatch / mis-decode.
- **Fix:** `fix_compact_f64_on` now quantizes its f32 `raw` to u64 fixed-point bits (`round(v*2^30) as i64 as u64`) before upload, matching the live build kernel; the in-kernel dequant inverts it (round-trip exact for integer-valued cells, <= 1/2^30 abs error otherwise — well within the ~1e-6 gate). The empty-feats early-return path (direct f32->f64 widen, no kernel) is unchanged.
- **Files modified:** crates/lgbm-compute/src/kernels/histogram.rs
- **Verification:** `kernel_parity_fix_compact_equals_host_on_hip` passes on HIP.
- **Committed in:** `cc3b040` (Task 2 commit)

**3. [Rule 2 - Missing Critical] Assert guarding the naive-fallback + fixed_point combination**
- **Found during:** Task 2 (naive fallback disposition)
- **Issue:** The naive >256-bin fallback stays f32, but a `fixed_point` caller dequantizes `h_out` as u64 downstream — if a feature ever exceeded 256 bins the f32 naive write would be silently mis-decoded.
- **Fix:** Added an `assert!(!fixed_point, ...)` in the naive arm with a message pointing at the <=256-bin contract. Correctness guard, not a silent corruption.
- **Files modified:** crates/lgbm-compute/src/kernels/histogram.rs
- **Verification:** Build + all kernel_parity tests pass (max_bin <= 255 keeps the LDS branch).
- **Committed in:** `cc3b040` (Task 2 commit)

---

**Total deviations:** 3 auto-fixed (1 blocking shared-seam, 1 bug shared-kernel, 1 missing-critical guard)
**Impact on plan:** All three are forced by the same root cause — `resident_raw_build_into` and `fix_compact_kernel` are SHARED across the live u64 chain, the f32 readback oracle, and the test path. The plan specified the u64 chain but under-specified the shared seams. The `fixed_point` flag + the test-path quantize keep all three callers correct with no architectural change and no scope creep. The plan's must-have key-links (`resident_raw_build_into -> construct_leaf_hist_resident_lds_kernel_u64`, `build_fix_compact_resident_f64_on -> fix_compact_kernel dequant`) are honored exactly.

## Issues Encountered
None beyond the shared-seam deviations above. All builds (rocm + CPU-only) and all 15 kernel_parity tests (CPU + HIP) pass; zero clippy errors on the rocm build.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Plans 02 and 03 can consume the new symbols: `construct_leaf_hist_resident_lds_kernel_u64`, the u64 `fix_compact_kernel`, the `fixed_point` dispatch flag, and the overflow guard.
- No new env flags introduced (the fixed-point path is unconditional on the resident LDS branch).
- Concern: the naive >256-bin fixed-point fallback is asserted-unreachable rather than implemented; if a future plan needs `max_bin > 255` on the resident GPU chain, the naive kernel will need a u64 port. Out of scope for Plan 01.

## Self-Check: PASSED

- Files: `crates/lgbm-compute/src/kernels/histogram.rs`, `crates/lgbm-compute/src/lib.rs`, `11-01-SUMMARY.md` all FOUND.
- Commits: `6ec996e`, `cc3b040`, `c95518d` all FOUND.

---
*Phase: 11-gpu-fixedpoint-int-atomics*
*Completed: 2026-06-22*
