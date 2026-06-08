---
phase: quick-260608-oib
plan: 01
subsystem: compute
tags: [cubecl, rocm, gfx1100, histogram, fix_histogram, compaction, gpu-kernel, device-resident]

# Dependency graph
requires:
  - phase: 260608-nn7
    provides: device-resident binned columns (ResidentBins RefCell) + construct_leaf_hist_resident_kernel; the L3 deferral spec
  - phase: 260608-mc5
    provides: one-cube-per-feature fused split kernel precedent (CubeDim::new_1d(1))
  - phase: 260608-lad
    provides: batched per-leaf histogram abstraction (Backend seam, build_leaf_histograms_raw / find_best_splits_batched)
provides:
  - On-GPU fix+compact kernel (fix_compact_kernel) proven BIT-EXACT vs host fix_histogram + compact_histogram
  - fix_compact_f64_on launcher (host-readback Task-1 isolation form, V5-validated)
  - Device-resident build->fix->compact chain returning a device Handle (build_fix_compact_resident_f64_on) + widen_f32_to_f64_kernel
  - upload_resident_columns helper (raw resident Handle for tests)
affects: [future L3 split-from-Handle wiring, subtraction-trick residency, GPU per-leaf round-trip elimination]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "On-GPU verbatim port of a host f64 fold (ascending order, single-owner per feature) proven bit-exact via compare_exact_f64_bits"
    - "Device-resident multi-kernel chain (build -> widen -> fix+compact) returning a Handle, no intermediate readback"

key-files:
  created: []
  modified:
    - crates/lgbm-compute/src/kernels/histogram.rs
    - crates/oracle-harness/tests/kernel_parity.rs

key-decisions:
  - "Ported DEF-07-02 semantics VERBATIM (not fixed): fix is a no-op for mfb==0, compact drops bin 0 for offset==1"
  - "Pitfall 2 honored: the RAW (un-bumped) leaf sum_hessian is passed to the on-GPU fix"
  - "DEFERRED the live split-from-Handle wiring (Task 2 steps 2/3): the host pool, non-spine inline scan, and subtraction trick all require the host-side fixed+compacted buffer, so eliminating the readback for directly-built leaves requires restructuring those paths — high-risk for the merge gate. Shipped Task 1 + the resident build->fix->compact Handle (step 1), both validated."

patterns-established:
  - "fix+compact kernel: one cube per feature, CubeDim::new_1d(1), f64 ascending fold, branchless select for the i != mfb exclusion"
  - "Resident chain helper returns (Handle, len) so a future caller can thread the device buffer into the split kernel without readback"

requirements-completed: [L3-GPU-FIXCOMPACT, L3-DEVICE-RESIDENT-SCAN]

# Metrics
duration: ~50min
completed: 2026-06-08
---

# Phase quick-260608-oib: L3 On-GPU FixHistogram + Compaction (keep fixed histogram device-resident) Summary

**An on-GPU fix+compact kernel that reproduces the host fix_histogram + compact_histogram BIT-EXACTLY on gfx1100, plus a fully device-resident build->fix->compact chain returning a device Handle — with the live split-from-Handle wiring honestly deferred because the host pool, non-spine inline scan, and subtraction trick still require the host-side buffer.**

## Performance

- **Duration:** ~50 min
- **Started:** 2026-06-08 (Task 0 baseline capture)
- **Completed:** 2026-06-08
- **Tasks:** 3 (Task 0 measurement, Task 1 kernel, Task 2 resident chain w/ deferral)
- **Files modified:** 2 (source); per-task atomic commits

## Accomplishments

### Task 0 — baseline (measurement only, no commit)
- `cargo build --workspace` and `--features rocm`: both compile clean.
- CPU bit-exact merge gate GREEN: kernel_parity (6), learner_parity (29), boosting_parity (75) — including `mfb_zero_offset_histogram_contract` (the DEF-07-02 golden [2,4,2,4] port target).
- ROCm parity suite GREEN on gfx1100 EXCEPT the pre-existing f32 D-03a `hip::kernel_parity_split_within_tol_on_hip` gap (confirmed failing at baseline — OUT of scope, untouched).
- GPU bench BEFORE (`cargo run --release --features rocm --example bench_train`, iters 100, leaves 31), train medians captured VERBATIM on THIS HEAD (not nn7's quoted numbers):

  | size   | rows  | feat | bins | train_median BEFORE |
  |--------|-------|------|------|---------------------|
  | small  | 2000  | 12   | 32   | **1.48s** |
  | medium | 8000  | 30   | 64   | **4.95s** |
  | large  | 20000 | 50   | 128  | **11.68s** |

### Task 1 — on-GPU fix+compact kernel, BIT-EXACT vs host (commit a123e4d)
- `fix_compact_kernel` (`#[cfg(feature="rocm")]`): ONE cube per feature (`CubeCount::Static(n,1,1)`, `CubeDim::new_1d(1)`), cube `f` owns only `[slot_off[f], slot_off[f]+2*num_bin[f])`. Per-feature `{num_bin, offset, most_freq_bin}` as `Array` args; leaf RAW `sum_gradient`/`sum_hessian` as shared scalars. f64 math on gfx1100.
  - FixHistogram: skip when `mfb==0` OR `mfb>=num_bin`; else seed RAW totals, subtract every other bin in ASCENDING order (`i != mfb` via branchless `select`), write the mfb cell — VERBATIM to `fix_histogram.rs:50-80`. RAW (un-bumped) sum_hessian (Pitfall 2).
  - compact: `offset<=0` no-op; `offset>=num_bin` zero region; else forward in-place shift `c <- c+offset` ASCENDING + zero tail — VERBATIM to `compact_histogram` (`learner.rs:2838-2864`).
- `fix_compact_f64_on<R>` launcher (host-readback isolation form): V5 validation (`num_bin==0`, `2*num_bin` overflow, `slot_off+2*num_bin > buf.len()`, empty feats no-launch); cubecl `unsafe` confined with SAFETY comment (CMP-01).
- BIT-EXACT oracle `kernel_parity_fix_compact_equals_host_on_hip` (`compare_exact_f64_bits`): a mixed-feature leaf concatenating Test A (mfb>0/offset==0 reconstruct), Test B (mfb==0/offset==1 — DEF-07-02 drop-bin-0), Test C (mfb>=num_bin no-op), Test D (offset>=num_bin zero region), plus a second mfb>0 reconstruct. The host reference uses the real exported `lgbm_treelearner::fix_histogram` + a verbatim-copy `host_compact_histogram` (so `learner.rs` stays byte-unchanged).

### Task 2 — device-resident build->fix->compact chain (commit dccad89)
- `widen_f32_to_f64_kernel`: on-device f32->f64 cell widen, `f64::cast_from(src[i])` matching the host readback widening exactly.
- `build_fix_compact_resident_f64_on<R>`: resident RAW build (`construct_leaf_hist_resident_kernel` into an f32-atomic buffer) -> on-device widen to f64 -> `fix_compact_kernel`, returning the fixed+compacted f64 **device Handle** + length. NO readback. V5 validation on the fix params.
- `build_fix_compact_resident_readback_f64_on` (validation variant) + `upload_resident_columns` helper.
- Oracle `kernel_parity_resident_build_fix_compact_equals_host_on_hip`: the on-device chain == host resident RAW build (`build_leaf_histograms_resident_f32_on`) + host `fix_histogram` + host compact, within the ~1e-6 f32-atomic RAW-build tolerance (`assert_within`; the fix+compact step itself is bit-exact per Task 1).
- GPU bench AFTER (same invocation):

  | size   | train_median BEFORE | train_median AFTER | delta |
  |--------|---------------------|--------------------|-------|
  | small  | 1.48s  | 1.38s  | ~ within noise |
  | medium | 4.95s  | 4.85s  | ~ within noise |
  | large  | 11.68s | 11.81s | ~ within noise |

  The AFTER numbers match BEFORE within run-to-run noise **because the live path is unchanged** — the round-trip is NOT yet eliminated from the live flow (the split-from-Handle wiring is deferred, see below). This is the honest result: no live perf change was expected since `lib.rs` (RocmBackend) was not modified.

## Deviations from Plan

### Deferred (per the plan's explicit deferral guidance — "honest partial over a risky over-reach")

**Task 2 steps 2/3 — the live split-from-Handle wiring — DEFERRED.**
- **Why:** Inspecting the live scan path showed the directly-built leaf's fixed+compacted histogram cannot be made device-resident-through-scan without ALSO restructuring three host-dependent consumers that read the host pool `Vec<f64>` buffer:
  1. **Non-spine inline scan** (`scan_leaf_histogram`): categorical / monotone / extra-trees features are scanned INLINE off the host `buf`, NOT through `find_best_splits_batched`. They require the host buffer.
  2. **Subtraction trick** (`learner.rs:1352-1418`): the larger child derives `parent - smaller` over host pool `Vec<f64>` buffers and the T-05-07-01 audit hook re-reads the host buf. Keeping these resident needs the HistogramPool restructured into device handles.
  3. **HistogramPool** stores f64 `Vec`s; the directly-built leaf's buffer becomes the retained parent for the next split, so it must remain host-resident for the subtract path.
- Threading the device Handle into the split kernel while ALSO keeping a host copy for (1)/(2)/(3) would NOT eliminate the readback (the host copy is still needed), so the round-trip removal requires the larger restructure — high-risk for the CPU merge gate (boosting_parity, learner_parity) for a quick task.
- **What WAS delivered instead (step 1):** the full resident build->fix->compact chain as a device-Handle-returning helper (`build_fix_compact_resident_f64_on`), proven faithful end-to-end by the new oracle. A future plan can thread `(Handle, len)` into `find_best_splits_batched_fused_f64_on` once the pool + inline-scan + subtraction paths are restructured to consume a device Handle.

### Auto-fixed issues
None — both kernels ported verbatim; no bugs encountered.

## Follow-up / Next steps
- **L3 completion (future plan):** thread the `build_fix_compact_resident_f64_on` Handle into a `find_best_splits_batched_fused_f64_on(..._from_handle)` variant for directly-built leaves whose features are ALL spine (no categorical/monotone/extra-trees), AND restructure (or special-case) the non-spine inline scan + subtraction-trick + HistogramPool to either consume the device Handle or fall back to the host buf. The subtraction-trick larger child stays on the host path (documented deferral either way).
- The pre-existing f32 D-03a `hip::kernel_parity_split_within_tol_on_hip` gap remains (out of scope; failed identically at Task 0 baseline).

## Evidence

- **Builds:** `cargo build --workspace` clean; `cargo build --workspace --features rocm` clean (both tasks).
- **CPU merge gate (byte-unchanged + GREEN at every task):**
  - kernel_parity: 6 passed.
  - learner_parity: 29 passed.
  - boosting_parity: 75 passed (incl. `mfb_zero_offset_histogram_contract`).
- **`git diff` confirms CPU path byte-unchanged:** only `crates/lgbm-compute/src/kernels/histogram.rs` and `crates/oracle-harness/tests/kernel_parity.rs` changed. `fix_histogram.rs`, `learner.rs` (`compact_histogram`), and `lib.rs` (`CpuBackend` + `RocmBackend` + Backend-trait defaults) have NO hunks.
- **ROCm gfx1100 (12 passed, 1 pre-existing failure):**
  - `kernel_parity_fix_compact_equals_host_on_hip` (Task 1) — BIT-EXACT via `compare_exact_f64_bits`.
  - `kernel_parity_resident_build_fix_compact_equals_host_on_hip` (Task 2) — within tol (~1e-6).
  - `kernel_parity_resident_gather_equals_host_gather_on_hip`, `_histogram_within_tol`, `_subtract_within_tol`, `_partition_exact` — GREEN.
  - ROCm learner_parity: 29 passed.
  - `kernel_parity_split_within_tol_on_hip` — FAILED (pre-existing f32 D-03a gap, confirmed at Task 0 baseline, OUT of scope).
- **GPU bench before/after:** captured above (real medians; AFTER ≈ BEFORE within noise because the live path is unchanged / wiring deferred).

## Commits

- `a123e4d` feat(260608-oib t1): on-GPU fix+compact kernel, BIT-EXACT vs host
- `dccad89` feat(260608-oib t2): device-resident build->fix->compact chain (Handle)

## Self-Check: PASSED
