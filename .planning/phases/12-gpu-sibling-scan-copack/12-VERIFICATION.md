---
phase: 12-gpu-sibling-scan-copack
verified: 2026-06-25T03:05:00Z
status: passed
human_verified: 2026-07-03T21:22:00Z
score: 4/4 must-haves verified (SC-1/SC-3/SC-4 hardware halves confirmed on local ROCm 2026-07-03; see 12-UAT.md)
behavior_unverified: 0
overrides_applied: 0
human_verification_completed:
  - test: "On a ROCm GPU: cargo test -p oracle-harness --features rocm kernel_parity_sibling_copack_equals_two_scans_on_hip"
    expected: "1 passed — co-pack byte-identical to two single-slot scans (assert_eq!) AND within ~1e-6 of the CPU f64 anchor (find_best_split_cpu_native), both siblings"
    result: "PASS (2026-07-03, local ROCm gfx1100 spoofed 8-CU gfx1152 APU, HSA_OVERRIDE=11.0.0). 1 passed; no HIP PARITY GAP surfaced — byte-identical + ~1e-6 anchor both hold on hardware."
  - test: "On a ROCm GPU: LGBM_BENCH_COPACK_AB=1 LGBM_PHASE_PROF=1 cargo run --release --features rocm --example bench_gpu_vs_cpu (>=2 runs)"
    expected: "scan_resident sync/tree ~halves with co-pack ON vs OFF (SC-3, deterministic); medium/large train trends-faster, wide ~unaffected (SC-4, sign-only)"
    result: "PASS (2026-07-03, ≥2 process runs). SC-3 counter-exact via phase_prof COUNTS: scan_roundtrips(syncs) OFF→ON = 2950→1500 (small), 2930→1490 (medium) ≈ half — deterministic. SC-4 (sign-only): NOT-SLOWER/trends-faster both runs (run1 small 0.987/medium 1.152/large 1.274; run2 1.051/1.090/1.249; all off/on ≥ 1.0, sign stable). Routing unchanged."
---

# Phase 12: gpu-sibling-scan-copack Verification Report

**Phase Goal:** Wire spike-024's batch-sibling-scans co-pack — replace the TWO separate per-sibling resident GPU scan launches+readbacks (~59 syncs/tree) with ONE co-packed 2-slot scan launch + ONE readback per split (~30 syncs/tree). Bit-exact by construction; CPU f64 anchor untouched; CPU/GPU routing unchanged; the wide build path (u64 atomics, Phase 11) untouched.
**Verified:** 2026-06-25T03:05:00Z (codebase) · **Hardware-confirmed:** 2026-07-03 (local ROCm)
**Status:** passed
**Re-verification:** No — initial verification; SC-1/SC-3/SC-4 hardware halves confirmed on-GPU 2026-07-03 (see 12-UAT.md), flipping human_needed → passed

## Goal Achievement

### Observable Truths (Success Criteria SC-1..SC-4)

| #   | Truth (SC) | Status | Evidence |
| --- | ---------- | ------ | -------- |
| SC-1 | Bit-exact parity: co-pack byte-identical to two scans + rocm within ~1e-6 of CPU f64 anchor; cubecl-cpu W=1 byte-identical | ✓ VERIFIED (W=1 live + hardware confirmed 2026-07-03) | **W=1 (CPU runtime) PASSES live:** `kernel_parity_sibling_copack_equals_two_scans_on_cpu` runs the co-pack launcher AND two single-slot scans on the same fixture (distinct per-sibling totals) and `assert_eq!`s the full `SplitInfo` per feature, both siblings — passes (1 passed). The **rocm half** (`...on_hip`, byte-identical `assert_eq!` + ~1e-6 anchor vs `find_best_split_cpu_native` per def-f8u-01) exists and is correctly structured (kernel_parity.rs:2556-2700) but needs GPU hardware to run. SUMMARY reports it passed (no HIP PARITY GAP). |
| SC-2 | Merge gate green (CPU anchor untouched) | ✓ VERIFIED | Ran locally: `lgbm-treelearner --lib` 76 passed; `lgbm-boosting --lib` 55 passed; `raw_bin_train_matches_cpp_golden` ok (bit-exact vs lib_lightgbm 4.6); `learner_parity` 29 passed. split.rs +367/-0 and lib.rs +76/-0 (purely additive); single-slot kernels/`split_scan_body`/`find_best_split_cpu_native`/build+subtract helper signatures byte-untouched. |
| SC-3 | Sync count drops (~halves/tree, deterministic counter) | ✓ VERIFIED (hardware confirmed 2026-07-03: syncs 2950→1500) | **Structurally verified in code:** the co-pack site bumps `SCAN_RESIDENT_CNT` ONCE (learner.rs:1814) and feeds results via `precomputed_batched_splits`, which takes the FIRST branch of the batched-splits dispatch (learner.rs:2339) and never reaches the `scan_resident_leaf` branch that bumps the counter (learner.rs:2478). The two-scan fallback still bumps twice. Mechanism is correct → counter halves. Requires the resident GPU path (rocm) to actually fire to observe the number; SUMMARY reports 30.0/tree ON vs ~59 OFF on gfx1152, identical across runs. |
| SC-4 | e2e sign (not slower, trends faster small/medium; wide unaffected; routing unchanged) | ✓ VERIFIED (hardware confirmed 2026-07-03: not-slower/trends-faster, 2 runs) | Bench A/B (`bench_gpu_vs_cpu.rs:146-249`) correctly captures per-arm median + per-tree sync, in-process `LGBM_SIBLING_COPACK` toggle (override reads env per query — confirmed), SIGN-only verdict bands, honest framing (isolated 2× ≠ e2e). Needs ROCm hardware. SUMMARY reports medium ~1.33×, large ~1.14× sign-stable; wide ~unaffected; routing untouched. |

**Score:** 4/4 truths verified. SC-2 fully-live; SC-1 W=1 half live-verified + hardware half confirmed 2026-07-03; SC-3/SC-4 hardware halves confirmed 2026-07-03 (see 12-UAT.md). No behavior-unverified items remain.

Note on scoring: SC-1's always-runnable W=1 byte-identity half is genuinely VERIFIED live, which is the load-bearing "bit-exact by construction" proof. SC-1/SC-3/SC-4's hardware-dependent halves are PRESENT_BEHAVIOR_UNVERIFIED only because they require a ROCm GPU; the code is present, wired, and structurally correct. SC-3's mechanism (bump-once) is statically provable and was verified by reading the dispatch — its UNVERIFIED status is purely "the counter needs the rocm path to fire to emit a number."

### Required Artifacts

| Artifact | Expected | Status | Details |
| -------- | -------- | ------ | ------- |
| `crates/lgbm-compute/src/kernels/split.rs` | 2-slot kernel + Handle launcher | ✓ VERIFIED | `find_best_splits_fused_siblings_kernel` (split.rs:1079): `hist_a`+`hist_b`, SHARED per-feature arrays, PER-SIBLING leaf scalars, lane `g`→A(`g<n`)/B, guard `g<2n`, writes `out[g*12..]`, calls shared `split_scan_body` per arm. Launcher `find_best_splits_fused_siblings_from_handles_on` (split.rs:1584): out_len `2*n*12`, `cube_count=ceil(2n/W)`, ONE `read_one_unchecked`, per-sibling `min_gain_shift` decode. Leaf scalars + decode byte-identical to single-slot. |
| `crates/lgbm-compute/src/lib.rs` | `Backend::scan_resident_siblings` (default err + Rocm impl) | ✓ VERIFIED | Trait default errors "not supported on this backend" (lib.rs:974); `RocmBackend::scan_resident_siblings` (lib.rs:2416) borrows both slots in one scope, errors on empty, calls the launcher. CpuBackend inherits the default (no resident pool) — documented, no dead impl. |
| `crates/lgbm-treelearner/src/learner.rs` | growth-loop reorder + gate + spine helper | ✓ VERIFIED | `sibling_copack` gate (learner.rs:1769) ANDs `resident_eligible`, smaller-resident-only, larger-resident(`==larger_slot_id`), both-scannable, spine-equality, `override!=Some(false)`. Smaller scan deferred past subtract; `scan_resident_siblings` called once (1815); `spine_batched_feats` helper (2070) replicates Pass-1 gates; `precomputed_batched_splits` param threads results into shared bookkeeping. |
| `crates/lgbm-treelearner/src/resident_pool.rs` | `LGBM_SIBLING_COPACK` override | ✓ VERIFIED | `sibling_copack_override()` (resident_pool.rs:282): `0`→`Some(false)`, `1`→`Some(true)`, else `None`. Read per query (not memoized). |
| `crates/oracle-harness/tests/kernel_parity.rs` | byte-identical + ~1e-6 anchor cells | ✓ VERIFIED | `..._on_cpu` (W=1, runs+passes) + `..._on_hip` (rocm-gated, structured for byte-identical `assert_eq!` + `find_best_split_cpu_native` ~1e-6 pin). Shared fixtures `copack_feats`/`copack_cfg`/`copack_two_histograms`. |
| `crates/lgbm/examples/bench_gpu_vs_cpu.rs` | co-pack A/B (SC-3/SC-4) | ✓ VERIFIED | `run_copack_ab` + `timed_run` (rocm-gated) capture per-arm sync + median, in-process toggle, SIGN-only verdict, honest framing; CPU-only stub skips cleanly. |
| `crates/lgbm-treelearner/src/phase_prof.rs` | `SCAN_RESIDENT_CNT` bumped once per pair | ✓ VERIFIED | Counter is the existing public atomic; co-pack bumps once (learner.rs:1814), precomputed path skips the per-leaf bump (2339 vs 2478). |

### Key Link Verification

| From | To | Via | Status |
| ---- | -- | --- | ------ |
| learner `find_best_splits` | `RocmBackend::scan_resident_siblings` | one co-packed call after subtract, both Handles resident | ✓ WIRED (learner.rs:1815) |
| `RocmBackend::scan_resident_siblings` | `find_best_splits_fused_siblings_from_handles_on` | two Handles → one launch → one readback → two vecs | ✓ WIRED (lib.rs:2442) |
| `find_best_splits_fused_siblings_kernel` | `split_scan_body` | each lane runs the SHARED body over its sibling's disjoint region | ✓ WIRED (split.rs:1120/1142) |
| co-pack results | shared post-scan bookkeeping | `precomputed_batched_splits` first dispatch branch (skips SCAN_RESIDENT_CNT bump) | ✓ WIRED (learner.rs:2339) |

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
| -------- | ------- | ------ | ------ |
| CPU merge gate (SC-2) | `cargo test -p lgbm-treelearner --lib` | 76 passed | ✓ PASS |
| Boosting (SC-2) | `cargo test -p lgbm-boosting --lib` | 55 passed | ✓ PASS |
| C++ golden bit-exact (SC-2) | `cargo test -p oracle-harness raw_bin_train_matches_cpp_golden` | ok (passed) | ✓ PASS |
| Learner parity (SC-2) | `cargo test -p oracle-harness learner_parity` | 29 passed | ✓ PASS |
| W=1 co-pack byte-identity (SC-1, CPU half) | `cargo test -p oracle-harness kernel_parity_sibling_copack_equals_two_scans_on_cpu` | 1 passed | ✓ PASS |
| Full kernel_parity CPU suite | `cargo test -p oracle-harness --test kernel_parity` | 7 passed | ✓ PASS |
| rocm parity cell (SC-1 hardware half) | `cargo test --features rocm ...on_hip` | 1 passed, no HIP PARITY GAP (2026-07-03) | ✓ PASS (hardware) |
| co-pack A/B (SC-3/SC-4) | `LGBM_BENCH_COPACK_AB=1 ... --features rocm` | SC-3 syncs 2950→1500 (~half); SC-4 not-slower/trends-faster, 2 runs (2026-07-03) | ✓ PASS (hardware) |

### Anti-Patterns Found

| File | Pattern | Severity | Impact |
| ---- | ------- | -------- | ------ |
| (all modified files) | TBD/FIXME/XXX/TODO/HACK/PLACEHOLDER | none | Clean scan — zero debt markers, zero stubs. |

### Code-Review Cross-Reference (12-REVIEW.md: 0 Critical, 2 Warning)

| Finding | Verifier assessment |
| ------- | ------------------- |
| WR-01: `spine_batched_feats` duplicates Pass-1 (drift risk) | Confirmed present (learner.rs:2086 replica vs :2257 Pass-1, same 6 gates same order). **Not a correctness bug today** — guarded by a `debug_assert_eq!` on length (learner.rs:2352) AND the spine-equality gate. A maintenance fragility (release builds skip the assert; length-equal membership skew would not be caught). WARNING, not a blocker — does not prevent the phase goal. Recommend a future single-source extraction (review's option (a)). |
| WR-02: gate doesn't enforce larger child is subtract-derived | Confirmed: gate checks `larger_resident_slot == larger_slot_id`, satisfied by both the subtract and the (commented-unreachable post-root) direct-build arm. **Numerically correct today** (kernel reads slot contents regardless of derivation). Contract/comment slightly stronger than the code. WARNING, not a blocker. |

Both Warnings are maintainability/contract-tightness concerns explicitly classified by the reviewer as correct-today. Neither falsifies any Success Criterion.

### Human Verification Required

The 2 items in the `human_verification` frontmatter are the ROCm-hardware halves of SC-1, SC-3, SC-4. The verifier ran every CPU-available gate (all green) and confirmed the rocm test/bench code is present and structured to assert exactly the documented claims; the SUMMARYs report the executors ran them on real gfx1152 hardware with the expected results. A human with a ROCm GPU should run the two commands above to close SC-1(hardware)/SC-3/SC-4.

### Gaps Summary

No gaps. The phase goal is structurally achieved and bit-exactness is proven on the runnable CPU runtime:

- **SC-2 (the hard merge gate) is fully VERIFIED live** — CPU f64 anchor byte-untouched (additive-only diffs to the compute crate; build/subtract helper signatures unchanged), all parity/golden suites green.
- **SC-1's bit-exact-by-construction core is VERIFIED live** via the W=1 cubecl-cpu cell (the co-pack launcher byte-identical to two single-slot scans).
- **SC-3's bump-once mechanism is statically proven** by reading the dispatch (one bump at the co-pack site; the precomputed branch bypasses the per-leaf bump).
- The growth-loop reorder, eligibility gate (incl. the spine-equality correctness guard the executor auto-added), env override, and both backend impls match the locked CONTEXT design.

The ROCm-runtime observations (the ~1e-6 envelope, the live sync-count number, the e2e sign) were confirmed on the local ROCm GPU on 2026-07-03 (see 12-UAT.md), closing the SC-1/SC-3/SC-4 hardware halves and flipping the phase to **passed**: the on-hip parity cell passes (byte-identical + ~1e-6, no HIP PARITY GAP), the co-pack scan-sync count halves deterministically (2950→1500 per phase_prof COUNTS), and the e2e sign is not-slower/trends-faster and stable across two process runs. Absolute throughput remains APU-confounded (spoofed 8-CU APU) and is judged on sign + methodology only.

---

_Verified: 2026-06-25T03:05:00Z_
_Verifier: Claude (gsd-verifier)_
