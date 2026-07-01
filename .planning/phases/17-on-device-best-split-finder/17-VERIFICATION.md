---
phase: 17-on-device-best-split-finder
verified: 2026-07-01T03:15:16Z
status: passed
score: 8/8 must-haves verified
behavior_unverified: 0
overrides_applied: 0
---

# Phase 17: On-Device Best-Split Finder Verification Report

**Phase Goal:** Per-feature split evaluation and the cross-feature/cross-leaf argmax run on-device, returning the chosen split with a single small scalar readback and tie-aware `default_left` parity.
**Verified:** 2026-07-01T03:15:16Z
**Status:** passed
**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths (ROADMAP Success Criteria + merged PLAN must-haves)

| #   | Truth | Status | Evidence |
| --- | ----- | ------ | -------- |
| 1 | SC#1 — On-device per-(leaf,feature) stage-1 eval: block prefix-sum, count recovery via cnt_factor + round-ties-even, min-data/min-sum-hessian guards, gain math, fwd/rev default-bin scan, block argmax → per-task record | ✓ VERIFIED | `best_split_parity_stage1_bit_exact_on_cpu` PASSES (all 6 D-07 categories compare_exact_f64_bits on cpu f64 fold); `split_eval_body` present (2 defs); lib tests `count_recovery_ties_even`, `validate_stage1_inputs_rejects_bad_shapes` green |
| 2 | SC#2 — Stage-2 cross-feature reduce + stage-3 cross-leaf argmax → chosen (leaf,feature,threshold,default_left) with a single 8-int scalar readback | ✓ VERIFIED | `best_split_parity_stage2_cross_feature_reduce` + `best_split_parity_stage3_export` PASS; readback audit: stage-2 launcher (best_split.rs 1970–2135) has zero `read_one_unchecked`, stage-3 has exactly one (line 2216); `prepare_leaf_best_split_info_kernel` packs 8-int layout, `out[7]==0` continuous asserted |
| 3 | SC#3 — Tie-aware default_left parity on hip: flip accepted only on verified f32 tie, non-tie flip hard-fails, empty/sparse pass | ✓ VERIFIED | `rocm_backend_default_left_tie` **RAN and PASSED on the ROCm APU** (5/5 rocm_backend_parity tests pass); `assert_split_tie_aware` has all three branches (accepted-flip requires verified_tie, non-tie HARD-FAIL via assert!, is_valid=false → pass); near-tie (fwd/rev pair) + clean-margin (single fwd) + empty fixtures present |
| 4 | SC#4 — Chosen split anchor-pinned (structure bit-exact, values ~1e-5); CPU/ROCm/host-CUDA byte-unchanged; merge gate green | ✓ VERIFIED | `cargo test --workspace` GREEN (0 failures); `stage1_f32_within_tol_on_cpu` passes; `on_device_growth_supported()` returns `false` (lib.rs:1239); gpu + rocm feature builds compile; default-path parity suites (kernel_parity, learner_parity, raw_bin) unregressed |
| 5 | Round-ties-even count recovery diverges from split.rs round-half-up (D-01) | ✓ VERIFIED | `round_ties_even_cube` (floor-based branch-free) in core; grep for `round_int`/`x+0.5` in core finds only doc-comments + one test-contrast closure (line 2247, in `#[cfg(test)]`); `count_recovery_ties_even` proves 2.5→2 (even) |
| 6 | default_left = task.assume_out_default_left, NOT reverse (Pitfall 3) | ✓ VERIFIED | `assume_out_default_left_table` lib test PASSES incl. the reverse=true && assume=false divergence (num_bin≤2 NaN); `build_split_find_tasks` present |
| 7 | USE_SMOOTHING gain path additive #[cube] (f64+f32); reused non-smoothing fns byte-unchanged (D-02/D-09) | ✓ VERIFIED | gain.rs: `calculate_splitted_leaf_output_smoothed`, `get_leaf_gain_smoothed`, `#[cube]`-promoted `get_leaf_gain_given_output` (+f32 mirrors) present; `smoothing_blend_matches_reference` + full gain suite (7) green |
| 8 | Stage-3 self-invalidation: chosen leaf AND freshly-created slot marked is_valid=false (behavioral) | ✓ VERIFIED | `stage3_cross_leaf_argmax_export_and_self_invalidation` lib test PASSES; code sets `leaf_best[...].is_valid=false` + `cur_num_leaves` slot invalidation (best_split.rs:2063–2118) |

**Score:** 8/8 truths verified (0 present, behavior-unverified)

### Required Artifacts

| Artifact | Expected | Status | Details |
| -------- | -------- | ------ | ------- |
| `crates/lgbm-compute/src/kernels/best_split.rs` | SplitFindTask, task-gen, stage 1/2/3 kernels + globalmem + validation | ✓ VERIFIED | 2599 lines; all 21 key symbols present and wired; registered ungated in mod.rs (line 50) |
| `crates/lgbm-compute/src/gain.rs` | additive smoothing #[cube] path | ✓ VERIFIED | smoothing fns + #[cube] given-output + f32 mirrors present; reused fns byte-unchanged |
| `crates/oracle-harness/tests/best_split_parity.rs` | golden-anchor harness, 4 stage tests, coverage booleans | ✓ VERIFIED | 4 tests run + pass; no `#[ignore]` attribute remains (only doc-comment mentions); 6 coverage booleans asserted |
| `crates/oracle-harness/tests/fixtures/kernels/best_split.txt` | 6-category D-07 golden matrix | ✓ VERIFIED | 22KB fixture, 26 SCASE/SWIN/COUNTS markers; provenance guard present |
| `crates/lgbm-compute/tests/rocm_backend_parity.rs` | tie-aware default_left hip test | ✓ VERIFIED | `rocm_backend_default_left_tie` runs + passes on ROCm; CpuBackend anchor vs RocmBackend mirror (never GPU-vs-GPU) |

### Key Link Verification

| From | To | Via | Status |
| ---- | -- | --- | ------ |
| best_split.rs | gain.rs | stage-1 calls get_split_gains/get_leaf_gain or *_smoothed | ✓ WIRED |
| best_split.rs | split_info.rs | writes DeviceSplitInfo slot [t]/[t+num_tasks] | ✓ WIRED |
| best_split.rs | random.rs | USE_RAND draws via CUDARandom seeded extra_seed+task_index | ✓ WIRED |
| best_split.rs | Stage1GlobalMemScratch (alloc-once) | client.empty confined to constructor + device_allocations counter | ✓ WIRED (globalmem_scratch_allocated_exactly_once == 4) |
| best_split.rs | host (Phase 18) | 8-int buffer = single device→host readback | ✓ WIRED |
| rocm_backend_parity.rs | best_split.rs | hip mirror anchored to CpuBackend f64 fold | ✓ WIRED |

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
| -------- | ------- | ------ | ------ |
| Stage 1-3 cpu-anchor parity | `cargo test -p oracle-harness --test best_split_parity` | 4 passed; 0 failed; 0 ignored | ✓ PASS |
| best_split lib units | `cargo test -p lgbm-compute --lib best_split` | 13 passed; 0 failed | ✓ PASS |
| gain smoothing units | `cargo test -p lgbm-compute --lib gain` | 7 passed; 0 failed | ✓ PASS |
| On-device tie-aware default_left | `cargo test -p lgbm-compute --features rocm --test rocm_backend_parity` | 5 passed; 0 failed (incl. rocm_backend_default_left_tie) | ✓ PASS |
| gpu-gated block/globalmem kernels compile | `cargo build -p lgbm-compute --features gpu` | Finished, clean | ✓ PASS |
| Merge gate | `cargo test --workspace` | 0 failures across all crates | ✓ PASS |

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
| ----------- | ----------- | ----------- | ------ | -------- |
| ODL-11 | 17-01/02/03/04 | On-device per-feature split evaluation (stage 1) | ✓ SATISFIED | SC#1 truths verified; REQUIREMENTS.md line 34/98 marked Complete |
| ODL-12 | 17-05 | Cross-feature reduce (stage 2) + cross-leaf argmax (stage 3) + tie-aware default_left | ✓ SATISFIED | SC#2 + SC#3 truths verified; REQUIREMENTS.md line 35/99 marked Complete |

No orphaned requirements: REQUIREMENTS.md maps exactly [ODL-11, ODL-12] to Phase 17; both claimed by plans and verified.

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
| ---- | ---- | ------- | -------- | ------ |
| best_split.rs | 1065, 1133 | `C++ TODO / Phase-22 / v2 QGD-02` (categorical + discretized _GlobalMemory reserved buffers) | ℹ️ Info | Documented out-of-scope seams (D-04 categorical → Phase-22; discretized spill → v2), not phase-17 incomplete work; the continuous forward/reverse >256-bin path IS ported and bit-exact |

No `TBD`/`FIXME`/`XXX` debt markers in any phase-17 file. No stub/placeholder returns in the shipped path (stage stubs were filled in Waves 2-4).

### Known Limitation (documented, in-scope)

The `_GlobalMemory` `na_as_missing && mfb_offset == 1` special reduction subcase (cu:1095-1114) is not ported — not exercised by the D-07 globalmem fixture (forward, mfb=0, na=0). Documented in the kernel doc as a sub-branch to complete when a golden needs it. The continuous forward/reverse branches the fixture drives are ported and bit-exact. This does not block the phase goal.

### Gaps Summary

None. All four ROADMAP success criteria are verified with executed behavioral evidence — including SC#3's on-device tie-aware `default_left` parity, whose ROCm test actually ran and passed in this environment (not merely compile-checked). All eight merged must-have truths resolve to VERIFIED, all artifacts exist / are substantive / wired, all key links wired, the merge gate is green, and both requirement IDs (ODL-11, ODL-12) are satisfied and accounted for.

---

_Verified: 2026-07-01T03:15:16Z_
_Verifier: Claude (gsd-verifier)_
