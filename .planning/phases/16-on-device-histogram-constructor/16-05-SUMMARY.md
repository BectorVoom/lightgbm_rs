---
phase: 16-on-device-histogram-constructor
plan: 05
subsystem: compute
tags: [merge-gate, cargo-test, rocm, cpu-anchor, byte-unchanged, atomic-i64, f64-hot-loop, ODL-09, ODL-10, ODL-19, D-07, D-08]

# Dependency graph
requires:
  - phase: 16-on-device-histogram-constructor
    plan: 04
    provides: "construct_histogram_for_leaf entry (build->de-quant->fix->rotate->subtract) behind LGBM_CUDA_ON_DEVICE; cuda_on_device_enabled() seam"
  - phase: 16-on-device-histogram-constructor
    plan: 03
    provides: "construct_leaf_hist_partition_u64 two-tier build + dequant_leaf_hist"
  - phase: 16-on-device-histogram-constructor
    plan: 02
    provides: "HistArena hist_t** rotation contract"
  - phase: 16-on-device-histogram-constructor
    plan: 01
    provides: "cpu f64-anchor scaffold (assert_close, cpu_anchor_columns)"
provides:
  - "Merge-gate sign-off record for Phase 16 (ODL-19 hard gate): workspace green 845/0 with default features + LGBM_CUDA_ON_DEVICE unset; no f64 per-row hot loop / no Atomic<i64> (D-08); shipped per-feature build kernel byte-unchanged (D-07); ROCm f32 parity attested within ~1e-6 of the cpu f64 anchor (human-verified, never GPU-vs-GPU)"
affects: [17-best-split-finder, 18-data-partition-tree-mutation, 21-end-to-end-driver]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "The hard merge gate is the cpu f64 fold: cargo test --workspace GREEN with default features and LGBM_CUDA_ON_DEVICE unset (ODL-19)"
    - "ROCm f32 parity is a SEPARATE ~1e-6 best-effort gate, human-verified on the physical backend, always hip-vs-cpu-anchor (never GPU-vs-GPU, def-f8u-01)"
    - "Phase 16 is strictly additive: the on-device entry is reachable only via the OFF-by-default LGBM_CUDA_ON_DEVICE seam; every default CPU/ROCm/host-CUDA path is byte-unchanged"

key-files:
  created:
    - ".planning/phases/16-on-device-histogram-constructor/16-05-SUMMARY.md"
  modified: []

key-decisions:
  - "Task 1 is verification-only — no source change, no code commit (the gate proves the phase is additive; a gate cannot also mutate the tree it guards)"
  - "The failing hip::cuda_mirror_full_corpus_leaf_matches_anchor case is accepted as pre-existing out-of-scope (DEF-16-OOS-02, def-f8u-01 class) — its test body AND the underlying construct_leaf_hist_resident_lds_kernel_u64 are byte-unchanged vs the phase base 8df8523, so Phase 16 did not cause it; it is a nondeterministic f32-atomic near-tie on the spoofed 8-CU APU, not a regression"
  - "The default-features merge gate (ODL-19) is the merge contract, and it is deterministically GREEN 845/0 with LGBM_CUDA_ON_DEVICE unset; the ROCm f32 flake lives only under --features rocm on the physical GPU and does not gate merge"

requirements-completed: [ODL-09, ODL-10]

# Metrics
duration: 5min
completed: 2026-07-01
status: complete
---

# Phase 16 Plan 05: Hard Merge Gate — Workspace Green + ROCm Parity Attestation Summary

**The Phase 16 merge-gate sign-off (D-07 / D-08 / ODL-19): the full workspace suite is GREEN 845 passed / 0 failed with default features and `LGBM_CUDA_ON_DEVICE` unset (the hard cpu f64 merge gate); the new build/fix kernels carry no f64 per-row hot loop and no `Atomic<i64>` (comment-stripped grep = 0); the shipped per-feature `construct_leaf_hist_resident_lds_kernel_u64` is byte-unchanged vs the phase base `8df8523`; and the on-device build/fix/subtract ROCm f32 parity is human-verified within ~1e-6 of the cpu f64 anchor (hip-vs-anchor, never GPU-vs-GPU). The one residual `--features rocm` flake is pre-existing and out-of-scope (DEF-16-OOS-02, def-f8u-01 class) — it does not gate the default-features merge.**

## Performance

- **Duration:** ~5 min (gate + attestation finalization; a fresh continuation agent after the human-verify checkpoint was approved)
- **Tasks:** 2 (Task 1 auto/verification-only; Task 2 checkpoint:human-verify — APPROVED)
- **Files modified:** 0 source (verification-only plan; produces the sign-off record only)

## Accomplishments

- **Task 1 — Hard merge gate (D-07 / D-08 / ODL-19), verification-only:**
  - **Workspace green:** `cargo test` (default features, `LGBM_CUDA_ON_DEVICE` unset) = **845 passed / 0 failed** across the workspace — the non-negotiable cpu f64 merge gate holds; Phase 16 did not weaken the bit-exact anchor.
  - **No Atomic<i64> (D-08):** the comment-stripped grep (`grep -vE '^\s*//' … | grep -c 'Atomic<i64>'`) returns **0** over `histogram.rs`.
  - **No f64 per-row hot loop (D-08):** neither the per-row scatter of `construct_leaf_hist_partition_u64` nor the ascending fold of `fix_histogram_mfb` accumulates in f64 in its hot loop; f64 appears only in the documented post-merge de-quant and scalar/gain math (u64 fixed-point in the build; hist_t float domain in the fix).
  - **Default path byte-unchanged (D-07):** the shipped per-feature `construct_leaf_hist_resident_lds_kernel_u64` body is byte-identical vs the phase base `8df8523` (whitespace-insensitive diff empty). The on-device entry is reachable only via the OFF-by-default `LGBM_CUDA_ON_DEVICE` seam, so CPU / ROCm / existing-host-CUDA default behavior is unchanged.
- **Task 2 — ROCm f32 parity human-verify (APPROVED):** on the physical ROCm backend, the on-device build / fix / subtract are attested within ~1e-6 of the cpu f64 anchor (the `assert_close` ABS=5e-6 / REL=1e-5 envelope), all hip-vs-cpu-anchor (never GPU-vs-GPU, def-f8u-01). The large-bin/global-spill case and the `most_freq_bin != 0` fix case are among those run (not skipped/ignored). The one failing case — `hip::cuda_mirror_full_corpus_leaf_matches_anchor` — was independently verified by the orchestrator as pre-existing and out-of-scope (see Deferred Issues) and the checkpoint was approved by the user.

## Task Commits

1. **Task 1: Hard merge gate — workspace green (845/0), no f64 hot loop, no Atomic<i64>, default byte-unchanged (D-07/D-08/ODL-19)** — verification-only, no code commit (a gate mutates nothing; results recorded here).
2. **Task 2: ROCm f32 parity human-verify** — APPROVED (attestation accepted; DEF-16-OOS-02 recorded).

The only commit for this plan is the docs/metadata commit carrying this SUMMARY + STATE.md + ROADMAP.md.

## Files Created/Modified

- `.planning/phases/16-on-device-histogram-constructor/16-05-SUMMARY.md` — this merge-gate sign-off record (created).
- No source files modified — this is a gate-only plan (`files_modified: []` in the PLAN frontmatter).

## Decisions Made

- **Task 1 is verification-only — no code commit.** The merge gate proves Phase 16 is additive and parity-clean; it does not (and must not) mutate the tree it guards. The gate results are the deliverable, recorded here.
- **The failing `hip::cuda_mirror_full_corpus_leaf_matches_anchor` is accepted as pre-existing out-of-scope (DEF-16-OOS-02).** Independently confirmed: both the test body and the underlying `construct_leaf_hist_resident_lds_kernel_u64` are byte-unchanged vs the phase base `8df8523` (whitespace-insensitive diff), so Phase 16 did not cause it. It is a nondeterministic f32-atomic near-tie on the spoofed 8-CU APU (magnitude 7–9e-6 vs a tol of ~5–6e-6; ~1/6 pass rate in the original session, 18/18 in the orchestrator's session) — the documented def-f8u-01 class, not a deterministic miss.
- **The default-features merge gate (ODL-19) is the merge contract, and it is deterministically GREEN 845/0** with `LGBM_CUDA_ON_DEVICE` unset. The ROCm f32 flake exists only under `--features rocm` on the physical GPU and does not gate merge.

## Deviations from Plan

None requiring approval. The plan's Task 2 checkpoint surfaced one failing ROCm case, which was correctly handled as a pre-existing out-of-scope flake (SCOPE BOUNDARY): logged to `deferred-items.md` as DEF-16-OOS-02, verified byte-unchanged vs the phase base, and approved by the user at the checkpoint rather than "fixed" (fixing the f32 add-order/tolerance flake is a separate concern — tighten the assert to a multi-run/seed-stable envelope, or pin the f32 mirror's add-order).

## Known Stubs

None. This plan ships no source symbols — it is the gate that certifies the Phase 16 on-device build/fix/subtract (shipped in 16-02..16-04) is additive, parity-clean, and off-by-default.

## Deferred Issues

- **DEF-16-OOS-02 (pre-existing, OUT OF SCOPE, def-f8u-01 class):** `lgbm-compute --features rocm --test rocm_cuda_mirror hip::cuda_mirror_full_corpus_leaf_matches_anchor` is a nondeterministic f32-atomic full-corpus mirror near-tie (|diff| 7.15e-6 – 9.18e-6 vs a ~5.0e-6 – 6.25e-6 tol; varying cell/magnitude per run). Root cause: the f32-atomic accumulation path (`construct_leaf_hist_resident_lds_kernel_u64` / the f32 resident mirror) has nondeterministic add-order on the spoofed 8-CU APU. **PRE-EXISTING** — the kernel is byte-unchanged this phase (verified vs `8df8523`), it is one of the shipped CUDA-mirror tests (16-03), NOT a Phase-16 on-device build/fix/subtract case, and it runs only under `--features rocm` on the physical GPU. The default merge gate (`cargo test --workspace`, ODL-19) is GREEN 845/0. Logged in `.planning/phases/16-on-device-histogram-constructor/deferred-items.md`. The tolerance/flakiness fix is a separate concern.
- **DEF-16-OOS-01 (pre-existing, OUT OF SCOPE):** `autotune.rs:148` `LaunchKey` Display format-string drift (`"LaunchKey(bucket=10,...)"` vs `"LaunchKey(b10,f50,b256)"`); `autotune.rs` is untouched by Phase 16. Logged in the same `deferred-items.md`.

## Threat Surface

Both STRIDE mitigations from the plan's `<threat_model>` are discharged: T-16-05-01 (regression of the byte-unchanged default path) is mitigated by the `cargo test --workspace` merge gate + the git-diff byte-unchanged check on `construct_leaf_hist_resident_lds_kernel_u64`; T-16-05-SC (package installs) is vacuously satisfied — this is a gate-only plan with zero new dependencies and no `npm`/`pip`/`cargo` installs. No new network/auth/untrusted-input surface.

## Next Phase Readiness

- **Phase 16 is complete and merge-clean.** All 5 plans (16-01 scaffold, 16-02 arena, 16-03 build, 16-04 fix+subtract+entry, 16-05 merge gate) are shipped; ODL-09 and ODL-10 are satisfied on the physical ROCm APU pinned to the cpu f64 anchor, off-by-default behind `LGBM_CUDA_ON_DEVICE`.
- **Phase 17 (best-split finder)** consumes `ConstructedLeafHists.{smaller, larger}` `hist_in_leaf` from the 16-04 entry.
- **Phase 18 (data partition / tree mutation)** consumes the HistArena rotation contract for the whole-tree pool SWAP.
- `on_device_growth_supported()` stays frozen false; `cuda_on_device_enabled()` is the seam the Phase-18/21 growth driver checks.

## Self-Check: PASSED

- Files: FOUND `.planning/phases/16-on-device-histogram-constructor/16-05-SUMMARY.md`
- DEF-16-OOS-02: FOUND in `.planning/phases/16-on-device-histogram-constructor/deferred-items.md` (not duplicated)
- Gate record: workspace 845/0 (default features, LGBM_CUDA_ON_DEVICE unset); Atomic<i64> grep = 0; `construct_leaf_hist_resident_lds_kernel_u64` byte-unchanged vs `8df8523`
- ROCm parity: human-verify checkpoint APPROVED (within ~1e-6 of the cpu f64 anchor, hip-vs-anchor)

---
*Phase: 16-on-device-histogram-constructor*
*Completed: 2026-07-01*
