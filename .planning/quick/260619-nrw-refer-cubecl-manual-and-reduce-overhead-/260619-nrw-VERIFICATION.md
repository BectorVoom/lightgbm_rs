---
phase: quick-260619-nrw
verified: 2026-06-19T00:00:00Z
status: passed
score: 5/5 must-haves verified
overrides_applied: 0
---

# Quick 260619-nrw: launch_unchecked sweep of production GPU histogram kernels — Verification Report

**Task Goal:** Refer the cubecl manual and reduce non-compute/launch/codegen overhead in the production GPU histogram kernels — sweep all 8 rocm-gated production histogram kernels from `::launch` to `::launch_unchecked` (dropping in-kernel bounds-check codegen) with per-kernel host-side SAFETY validation, while preserving the GPU-vs-CPU-f64-anchor parity contract.
**Verified:** 2026-06-19
**Status:** passed
**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
| --- | --- | --- | --- |
| 1 | Every rocm-gated production histogram kernel launches via `::launch_unchecked` (no in-kernel bounds-check codegen) | ✓ VERIFIED | 8 production kernels carry `#[cube(launch_unchecked)]` (atomic_f32@389, lds_f32@535, batched@677, resident@832, resident_lds@931, batched_lds@984, fix_compact@1574, fused@2041) + the out-of-scope mirror@1067. All 9 production launch sites use `::launch_unchecked(` (456, 644, 765, 803, 1483, 1524, 1771, 1936, 2405; fix_compact has 2). `git diff parent..61aac21` shows exactly 8 `-#[cube(launch)]` → `+#[cube(launch_unchecked)]` swaps. |
| 2 | The two f64-deterministic kernels (fix_compact, build_fix_scan_fused) stay bit-exact to the CPU f64 anchor after the switch | ✓ VERIFIED | `kernel_parity_fix_compact_equals_host_on_hip` + `kernel_parity_build_fix_scan_equals_host_on_hip` (compare_exact / bit pins) — both green in the 9/9 hip module run on gfx1100. |
| 3 | Every f32 production kernel still matches the CPU f64 anchor within ABS 5e-6 / REL 1e-5 (GPU-vs-CPU-f64-anchor) | ✓ VERIFIED | `rocm_parallel_histogram` 7/7 (atomic + lds anchor pins), `rocm_row_partition` (batched-LDS P1/P>1 + new naive-fallback pin) green, hip kernel_parity resident/subtract/split/histogram within-tol pins 9/9 green — all on real gfx1100, GPU-vs-CPU-f64-anchor (never GPU-vs-GPU). |
| 4 | The CPU f64 anchor kernels (construct_hist_kernel f64@84, construct_hist_kernel_f32@105) are byte-unchanged | ✓ VERIFIED | Both remain the only `#[cube(launch)]` attributes (lines 84, 105); their `::launch(` sites (177, 362) intact. `git diff parent..61aac21` contains zero lines matching `construct_hist_kernel(::|_f32|()` — anchors untouched. CPU lib tests 30/30 green. |
| 5 | CPU-only build and `--features rocm` build both compile | ✓ VERIFIED | `cargo build -p lgbm-compute` Finished; `cargo build -p lgbm-compute --features rocm` Finished; `--features rocm --tests` Finished. |

**Score:** 5/5 truths verified

### Required Artifacts

| Artifact | Expected | Status | Details |
| --- | --- | --- | --- |
| `crates/lgbm-compute/src/kernels/histogram.rs` | All 8 production launchers on launch_unchecked w/ per-access SAFETY enumerations | ✓ VERIFIED | 8 attribute swaps; SAFETY block above all 9 production launch sites (fused block at 2374-2406 is the per-access enumeration ending in the mwr numerics-unchanged clause). 184-line diff, +167/-17. |
| `crates/lgbm-compute/tests/rocm_row_partition.rs` | GPU-vs-CPU-f64-anchor re-pin for the one uncovered swept kernel | ✓ VERIFIED | New `naive_batched_fallback_matches_cpu_anchor_within_tol` (62 added lines, 0 deletions): bounded subset `(7..n).step_by(3)`, ABS 5e-6 / REL 1e-5 vs CPU f64 anchor, passes on gfx1100. |

### Key Link Verification

| From | To | Via | Status | Details |
| --- | --- | --- | --- | --- |
| construct_leaf_hist_resident_lds_kernel (wired production path) | resident_raw_build_into LDS launcher (:1483) | `::launch_unchecked` switch + extended SAFETY | ✓ WIRED | Kernel @931 is `#[cube(launch_unchecked)]`; launcher @1483 calls `::launch_unchecked`; SAFETY block present. |
| swept production kernels | CPU f64 anchor | rocm parity tests (ABS 5e-6 / REL 1e-5, bounded subset) | ✓ WIRED | rocm_parallel_histogram 7/7, rocm_row_partition pins, hip kernel_parity 9/9 — all green GPU-vs-CPU-f64-anchor on gfx1100. |

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
| --- | --- | --- | --- |
| CPU-only build compiles | `cargo build -p lgbm-compute` | Finished | ✓ PASS |
| rocm build compiles | `cargo build -p lgbm-compute --features rocm` | Finished | ✓ PASS |
| rocm tests compile | `cargo build -p lgbm-compute --features rocm --tests` | Finished | ✓ PASS |
| naive-batched fallback anchor pin | `cargo test ... rocm_row_partition naive_batched_fallback... --exact` | 1 passed | ✓ PASS |
| atomic + lds production anchor pins | `cargo test ... --test rocm_parallel_histogram` | 7 passed | ✓ PASS |
| f64 bit-exact + resident hip pins | `cargo test -p oracle-harness --features rocm --test kernel_parity hip` | 9 passed | ✓ PASS |
| CPU merge-gate lib sanity | `cargo test -p lgbm-compute --lib` | 30 passed | ✓ PASS |

### Parity / DEF-MWR-01 Attribution

| Item | Status | Evidence |
| --- | --- | --- |
| DEF-MWR-01 documented as pre-existing, not swallowed | ✓ VERIFIED | SUMMARY §"Parity Residual" attributes the intermittent `cuda_mirror_full_corpus_leaf_matches_anchor` flake (\|diff\| ~6.5–7.2e-6) to pre-existing full-corpus near-zero-grad f32-atomic cancellation. Verified independently: `git diff parent..61aac21` shows **zero** cuda_mirror lines changed by nrw, and the mirror was already `#[cube(launch_unchecked)]` at the pre-nrw parent (c3e5b05) — so the sweep cannot be the cause (launch_unchecked does not change accumulation order). |
| No tolerance gate widened | ✓ VERIFIED | Test-file diff across the 3 commits = +62 / -0 (only the new test added). No existing tolerance literal removed/loosened. The new test uses the established ABS 5e-6 / REL 1e-5 gate. |

### Commits

| Commit | Status | Subject |
| --- | --- | --- |
| d4dde2f | ✓ FOUND on master | sweep f64-deterministic + wired LDS kernels to launch_unchecked |
| 609cc67 | ✓ FOUND on master | sweep remaining 5 rocm f32 histogram kernels to launch_unchecked |
| 61aac21 | ✓ FOUND on master | re-pin naive batched fallback kernel GPU-vs-CPU-f64-anchor |

### Anti-Patterns Found

None. No new debt markers, no silently-widened gates, no GPU-vs-GPU comparisons introduced. The plan's scope-drops (NRW-02 comptime, Stage-3 restructure) are recorded with research rationale, not padded.

### Notes / Benign Deviations

- The PLAN `files_modified` frontmatter listed `tests/rocm_parallel_histogram.rs` as the test file; the actual new pin landed in `tests/rocm_row_partition.rs`. The SUMMARY documents this accurately (audited existing coverage → 7/8 kernels already pinned across rocm_parallel_histogram / hip kernel_parity / rocm_row_partition; added the one missing naive-batched-fallback pin to rocm_row_partition). This is a documentation-level deviation that does not reduce coverage and is honestly reported.

### Gaps Summary

No gaps. All 5 must-haves verified against the actual codebase and confirmed on real gfx1100 hardware. The 8 production launchers are on `::launch_unchecked` with complete per-access SAFETY enumerations; the 2 CPU f64 anchor kernels are byte-unchanged; both builds compile; parity is re-pinned green GPU-vs-CPU-f64-anchor (f64 bit-exact, f32 within ABS 5e-6 / REL 1e-5); the DEF-MWR-01 residual is independently confirmed pre-existing and correctly attributed with no gate weakened; all 3 commits exist on master.

---

_Verified: 2026-06-19_
_Verifier: Claude (gsd-verifier)_
