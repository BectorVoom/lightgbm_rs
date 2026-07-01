---
phase: 16-on-device-histogram-constructor
verified: 2026-07-01T00:00:00Z
status: passed
score: 9/9 must-haves verified
behavior_unverified: 0
overrides_applied: 0
requirements_verified: [ODL-09, ODL-10]
forward_risks:  # Behind the dormant LGBM_CUDA_ON_DEVICE seam — must close before Phase 18/21 wires it on
  - id: WR-01
    concern: "Sparse build launcher never validates CSR `data` length; SAFETY comment over-claims. OOB device read risk on the sparse launch_unchecked path."
    activates_in: "Phase 18/21 (when the growth driver enables the seam)"
  - id: WR-02
    concern: "Multi-partition sparse indexing (no partition-base add) is inconsistent with the dense arm and is exercised by no committed test (all sparse fixtures are single-partition)."
    activates_in: "Phase 18/21"
  - id: WR-03
    concern: "`by = (256/bx).max(1)` degenerates for wide partitions (>256 columns); no upper clamp on bx nor device max-threads validation before CubeDim::new_2d."
    activates_in: "Phase 18/21"
deferred:
  - truth: "ROCm f32 full-corpus mirror near-tie flake (hip::cuda_mirror_full_corpus_leaf_matches_anchor)"
    addressed_in: "Out-of-scope (DEF-16-OOS-02)"
    evidence: "Pre-existing, def-f8u-01 class; the underlying construct_leaf_hist_resident_lds_kernel_u64 is byte-unchanged vs phase base 8df8523; runs only under --features rocm; default merge gate (cargo test --workspace) is GREEN 845/0. Correctly scoped as out-of-scope, not a Phase-16 regression."
---

# Phase 16: On-Device Histogram Constructor — Verification Report

**Phase Goal:** The hot-path histogram — build, fix, and subtract — runs on-device, building only the smaller leaf and deriving the larger by subtraction via pointer rotation, anchor-pinned.
**Verified:** 2026-07-01
**Status:** passed
**Re-verification:** No — initial verification

## Goal Achievement

The build → fix → subtract on-device hot path is landed as real, wired code behind the OFF-by-default `LGBM_CUDA_ON_DEVICE` seam. The subtraction trick (build-smaller-only, derive-larger-in-place via `hist_t**` pointer rotation) is behaviorally proven on the cpu f64 anchor; the GPU-device numeric parity was human-verified and approved at the 16-05 checkpoint. The hard merge gate (`cargo test --workspace`, default features, seam unset) was independently re-run during this verification: **845 passed / 0 failed** — matching the SUMMARY claim exactly and confirming the phase is strictly additive.

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | On-device two-tier §13-geometry BUILD kernel exists (dense+sparse × shared-LDS + `_GlobalMemory` spill), u64 fixed-point, no f64 per-row hot loop, no `Atomic<i64>` (ODL-09) | ✓ VERIFIED | `construct_leaf_hist_partition_u64<B>` @histogram.rs:1372, `construct_leaf_hist_partition_global_u64<B>` @:1471; scatter uses `Atomic<u64>::fetch_add` on `round(v·2^30)` i64-bits (no f64 in loop); comment-stripped `Atomic<i64>` grep = 0; ROCm device parity human-approved (16-05 Task 2) |
| 2 | Raw u64 fixed-point histogram de-quanted ONCE to hist_t after merge (separate pass, no f64 per-row loop) | ✓ VERIFIED | `dequant_leaf_hist` @:1557 + `dequant_leaf_hist_f32` @:1567 — `(bits as i64)/2^30`, distinct post-merge pass (RESEARCH Pattern 3) |
| 3 | Shipped per-feature ROCm build kernel byte-unchanged and coexists (D-03/D-07) | ✓ VERIFIED | `git` body diff of `construct_leaf_hist_resident_lds_kernel_u64` 8df8523→HEAD: first 49 lines identical; the only delta is NEW code inserted *after* it (+480/0) |
| 4 | HistArena pre-allocated-once slot pool with explicit {parent,smaller,larger} handle contract; allocate-exactly-once counted + asserted (D-02/D-09) | ✓ VERIFIED | `struct HistArena` @histogram_arena.rs:54; single counted `client.empty` closure @:118-123, `checked_mul` slab @:108; `mod.rs:43 pub mod histogram_arena;`; `new_allocates_exactly_num_slots` + zero/overflow-reject tests green |
| 5 | Pointer rotation derives larger IN-PLACE in parent's buffer, smaller into a fresh slot, no bulk copy, no aliasing | ✓ VERIFIED (behavioral) | `rotate()` @:236 sets larger_idx←parent_idx; `rotate_bookkeeping_no_alias`, `rotate_does_not_allocate`, `rotate_subtract_lands_in_larger_parent_slot` all green (cpu f64 anchor) |
| 6 | FixHistogram repairs omitted most-freq bin in the hist_t FLOAT domain (`leaf_total−Σ` ascending), only `mfb>0 && mfb<num_bin`; compact step DROPPED (ODL-10) | ✓ VERIFIED | `fix_histogram_mfb` @:3080: `do_fix = mfb>0 && mfb<nb`, ascending branchless `select` fold, writes `hist[mfb·2]/[mfb·2+1]`; no `if off > 0` compact; no 2^30 re-quantize; ROCm mfb!=0 golden human-approved |
| 7 | SubtractHistogram derives larger = parent − smaller via arena rotation reusing `subtract_hist_kernel`, only after parent fully built+synced (8aed100 ordering) | ✓ VERIFIED (behavioral) | `subtract_histogram_on_device` @subtract.rs:332 calls `arena.rotate()` + verbatim `subtract_hist_kernel::launch`, `larger_leaf_index<0` guard; `subtract_ordering_parent_synced_before_subtract` guard test green |
| 8 | `construct_histogram_for_leaf` entry sequences build→dequant→fix→rotate→subtract behind `LGBM_CUDA_ON_DEVICE` (OFF by default); `on_device_growth_supported()` stays false | ✓ VERIFIED | Entry @histogram.rs:3310 wires `construct_leaf_hist_on_device`→`dequant_leaf_hist`→`fix_histogram_mfb_on`→`subtract_histogram_on_device`, §7.0 early-return `Ok(None)`; `on_device_growth_supported()` returns `false` @lib.rs:1240; `cuda_on_device_enabled()` @:1311 OnceLock-cached, default off; `on_device_growth_supported_stays_false` + `cuda_on_device_seam_off_by_default` green |
| 9 | Full workspace green with default features + `LGBM_CUDA_ON_DEVICE` unset; default CPU/ROCm/host-CUDA paths byte-unchanged (the hard merge gate) | ✓ VERIFIED | Re-ran `cargo test` (seam unset): **845 passed / 0 failed** across the workspace; no error/failed lines |

**Score:** 9/9 truths verified (0 present, behavior-unverified)

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|-------------|-------------|--------|----------|
| ODL-09 | 16-01, 16-03, 16-05 | On-device histogram BUILD (dense+sparse × shared+global spill), two-tier u64 fixed-point, no f64 per-row hot loop, anchor-pinned | ✓ SATISFIED | Truths 1–3 (REQUIREMENTS.md: Phase 16, Complete) |
| ODL-10 | 16-01, 16-02, 16-04, 16-05 | Subtraction trick on device — build-smaller-only, FixHistogram, SubtractHistogram via `hist_t**` rotation, no bulk copy | ✓ SATISFIED | Truths 4–8 (REQUIREMENTS.md: Phase 16, Complete) |

Phase 16 owns exactly ODL-09 and ODL-10 (REQUIREMENTS.md rollup: "Phase 16 — Histogram constructor | ODL-09, ODL-10 | 2"). No orphaned requirements. ODL-19 (the merge-gate non-negotiable referenced by 16-05) is owned by Phase 21 (Pending) — correctly NOT claimed as a Phase-16 requirement in any PLAN frontmatter.

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `crates/lgbm-compute/src/kernels/histogram_arena.rs` | HistArena slot pool + rotation | ✓ VERIFIED | `struct HistArena`, `new`/`rotate`/handle accessors; registered in mod.rs; 10 tests green |
| `crates/lgbm-compute/src/kernels/histogram.rs` | build (dense/sparse × shared/global) + de-quant + fix + entry | ✓ VERIFIED | 5 new symbols present (build×2, dequant×2, launcher, fix, fix-launcher, entry); shipped kernel byte-unchanged |
| `crates/lgbm-compute/src/kernels/subtract.rs` | subtract wiring reusing subtract_hist_kernel | ✓ VERIFIED | `subtract_histogram_on_device` @:332; no new subtract kernel |
| `crates/lgbm-compute/src/lib.rs` | seam gate; growth stays false | ✓ VERIFIED | `cuda_on_device_enabled()` @:1311, `on_device_growth_supported()` false @:1240 |
| `crates/lgbm-compute/tests/rocm_cuda_mirror.rs` | cpu-anchor scaffold + mirror cases | ✓ VERIFIED | 5 cpu-anchor tests green; GPU cases under `mod hip` (--features rocm) |
| `crates/oracle-harness/tests/kernel_parity.rs` | Wave-0 fixtures | ✓ VERIFIED | `mod kernel16_fixtures` + 3 tests (sparse {16,32,64}, large-bin spill, mfb!=0 golden) |

### Key Link Verification

| From | To | Via | Status |
|------|----|----|--------|
| `construct_histogram_for_leaf` | `construct_leaf_hist_on_device` / `dequant_leaf_hist` / `fix_histogram_mfb_on` / `subtract_histogram_on_device` | sequenced build→dequant→fix→rotate→subtract | ✓ WIRED (histogram.rs:3310+) |
| `subtract_histogram_on_device` | `HistArena` | `arena.rotate()` + larger_handle (in-place) | ✓ WIRED (subtract.rs:332+) |
| `subtract_histogram_on_device` | `subtract_hist_kernel` | verbatim reuse, no new kernel | ✓ WIRED |
| `construct_leaf_hist_on_device` | `FeaturePartitionLayout` (row_data.rs) | §13 geometry selection (dense/sparse, shared/global) | ✓ WIRED |
| production growth path | `construct_histogram_for_leaf` | `cuda_on_device_enabled()` gate (OFF default) | ⚠️ DORMANT BY DESIGN (IN-01: no production call site yet — Phase 18/21 consumes it; `on_device_growth_supported()` independently false) |

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| Pointer-rotation subtraction round-trip lands in parent slot (cpu f64 anchor) | `cargo test -p lgbm-compute --lib histogram_arena` | 10 passed | ✓ PASS |
| Build-synced-before-subtract ordering (8aed100 guard) | `cargo test -p lgbm-compute --test rocm_cuda_mirror` | `subtract_ordering_parent_synced_before_subtract` ok | ✓ PASS |
| [2b]/[2b+1] interleave helper accepts correct / rejects swapped | (same) | `interleave_layout_accepts_correct_and_rejects_swapped` ok | ✓ PASS |
| Seam OFF by default + growth stays false | `cargo test -p lgbm-compute --lib on_device` | 2 passed | ✓ PASS |
| Wave-0 fixtures (sparse widths, spill, mfb!=0 golden) | `cargo test -p oracle-harness --test kernel_parity` | mod kernel16_fixtures green (in-workspace) | ✓ PASS |
| Hard merge gate (workspace, default features, seam unset) | `cargo test` | **845 passed / 0 failed** | ✓ PASS |
| GPU-device build/fix numeric parity (dense/sparse/spill/mfb!=0 vs cpu f64 anchor, ~1e-6) | `LGBM_CUDA_ON_DEVICE=1 cargo test --features rocm --test rocm_cuda_mirror` | Not re-run in this env (spoofed 8-CU APU; no discrete GPU) — human-verified & APPROVED at 16-05 Task 2 checkpoint | ? SKIP (already human-approved) |

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| histogram.rs | 3930 | "na_as_missing not yet implemented" | ℹ️ Info | Pre-existing (present at phase base 8df8523); a typed `ComputeError` detail for an unsupported config, NOT a Phase-16 stub. No action. |

No `TBD`/`FIXME`/`XXX` markers in any phase-modified source file. No stub returns, no hardcoded-empty data, no placeholder implementations in the new code.

### Human Verification

The one manual-only check for this phase — ROCm f32 device parity within ~1e-6 of the cpu f64 anchor — was executed and **APPROVED** at the 16-05 Task 2 `checkpoint:human-verify` gate (recorded in 16-05-SUMMARY). It is not re-opened here. Note for transparency: this verifier could not independently re-run the `--features rocm` suite (the local GPU is a spoofed 8-CU APU, not a discrete backend), so the device-kernel numeric parity rests on that already-approved checkpoint plus the byte-unchanged/green default merge gate re-run here.

### Forward Risks (not this-phase gaps)

The code review (16-REVIEW.md) found **0 blockers, 3 warnings, 3 info**. All three warnings (WR-01 sparse CSR `data`-length validation gap + over-claiming SAFETY comment; WR-02 untested multi-partition sparse indexing; WR-03 wide-partition block-dim degeneracy) live entirely on the sparse/multi-partition BUILD path, which is unreachable in production this phase (behind the dormant seam, no production call site — IN-01). They are correctly classified WARNING not BLOCKER. **They MUST be closed before the Phase-18/21 growth driver wires `cuda_on_device_enabled()` on.** Captured in frontmatter `forward_risks`.

### Deferred Items

`DEF-16-OOS-02` (ROCm f32 full-corpus mirror near-tie flake) is correctly scoped out-of-scope: pre-existing def-f8u-01 class, the underlying `construct_leaf_hist_resident_lds_kernel_u64` byte-unchanged vs phase base, runs only under `--features rocm`, and the default merge gate is deterministically GREEN 845/0. `DEF-16-OOS-01` (autotune Display drift) is likewise pre-existing and untouched by Phase 16.

### Gaps Summary

None. All 9 observable truths verified, both requirements (ODL-09, ODL-10) satisfied, all artifacts present/substantive/wired, the subtraction-trick state transition and 8aed100 ordering invariant proven behaviorally on the cpu f64 anchor, the merge gate independently re-run green (845/0), the shipped kernel byte-unchanged, and the seam confirmed OFF by default with `on_device_growth_supported()` false. The GPU-device numeric parity was human-verified and approved at the phase's own 16-05 checkpoint. The three review warnings are forward risks on the dormant sparse path, tracked for Phase 18/21.

---

_Verified: 2026-07-01_
_Verifier: Claude (gsd-verifier)_
