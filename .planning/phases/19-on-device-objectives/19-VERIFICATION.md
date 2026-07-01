---
phase: 19-on-device-objectives
verified: 2026-07-02T00:00:00Z
status: passed
score: 5/5 must-haves verified
behavior_unverified: 0
overrides_applied: 0
---

# Phase 19: On-Device Objectives Verification Report

**Phase Goal:** All CUDA-supported objectives compute grad/hess (plus ConvertOutput, BoostFromScore, RenewTreeOutput) on-device, anchor-pinned, so the boosting layer never round-trips gradients to host.
**Verified:** 2026-07-02
**Status:** passed
**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths

Truths are the five ROADMAP Success Criteria (the phase contract), merged with PLAN
frontmatter must_haves. Each is behavior-dependent (grad/hess computation, RNG stream,
state layout) and each is backed by a PASSING anchor-parity test against real
lib_lightgbm 4.6 goldens — so they are VERIFIED on behavioral evidence, not presence alone.

| # | Truth (ROADMAP SC) | Status | Evidence |
| --- | --- | --- | --- |
| 1 | On-device regression-family grad/hess (L2/L1/Quantile/Huber/Fair/Poisson) + ConvertOutput + BoostFromScore (mean/median) + RenewTreeOutput (per-leaf median), anchor-pinned | ✓ VERIFIED | `objective_regression.rs` (603 lines); `objective_parity_regression` 6/6 cells pass (regression, boost_from_score, renew_leaf, convert_regression, weight-branch/determinism, poisson-label-guard) bit-exact vs f64 anchor + `*_gh` goldens (Poisson exp within ORACLE_TOL) |
| 2 | On-device binary-logloss grad/hess + BoostFromScore logit init + sigmoid ConvertOutput + OVA label reset, anchor-pinned | ✓ VERIFIED | `objective_binary.rs` (410 lines); `objective_parity_binary` 4/4 cells pass; grad/hess bit-exact vs `binary_gh`, init within ORACLE_TOL, OVA reset bit-exact |
| 3 | On-device multiclass grad/hess (softmax + OVA, class-major `[k·num_data+i]` layout), anchor-pinned | ✓ VERIFIED | `objective_multiclass.rs` (416 lines); class-major stride confirmed (21 `num_data*k`/offset refs); `objective_parity_multiclass` 3/3 pass bit-exact vs class-major `multiclass_gh`/`multiclassova_gh`; Σ_k grad≈0 invariant holds |
| 4 | On-device ranking grad/hess (LambdaRank-NDCG + RankXENDCG, per-query block, bitonic item ranking, per-item RNG bit-identical), anchor-pinned | ✓ VERIFIED | `objective_rank.rs` (812 lines); `bitonic_argsort_items_on` (7×) + `draw_next_float_on` (5×); shared+_Sorted+_GlobalMemory variants built; `objective_parity_rank` 3/3 pass incl `rank_xendcg_rng_replay` bit-exact (compare_exact_u32) vs host `Random(seed+q)` stream |
| 5 | CUDA-unsupported objectives fall back to host; CPU/ROCm/host-CUDA byte-unchanged; merge gate green | ✓ VERIFIED | `device_objective_supported` rejects all 7 unsupported, accepts 11 supported (2 unit tests pass); `on_device_growth_supported()` returns `false` (unchanged, test asserts); full workspace `cargo test --workspace --lib --tests` = 909 passed / 0 failed |

**Score:** 5/5 truths verified (0 present, behavior-unverified)

### Required Artifacts

| Artifact | Expected | Status | Details |
| --- | --- | --- | --- |
| `crates/lgbm-compute/src/kernels/objective_regression.rs` | 6 grad/hess kernels + ConvertOutput + BoostFromScore + RenewTreeOutput | ✓ VERIFIED | 603 lines, contains `get_gradients`; composes reduce/percentile primitives (36 refs) |
| `crates/lgbm-compute/src/kernels/objective_binary.rs` | binary grad/hess + BoostFromScore + sigmoid ConvertOutput + OVA reset | ✓ VERIFIED | 410 lines; composes `reduce_sum_f64_on` (5×) |
| `crates/lgbm-compute/src/kernels/objective_multiclass.rs` | class-major softmax grad/hess + softmax ConvertOutput + OVA | ✓ VERIFIED | 416 lines; class-major stride load-bearing (21 refs) |
| `crates/lgbm-compute/src/kernels/objective_rank.rs` | LambdaRank {shared,>2048} + RankXENDCG {shared,global} + RNG | ✓ VERIFIED | 812 lines; 4 kernel variants + argsort/RNG composition |
| `crates/lgbm-compute/src/device_objective.rs` | DeviceObjectiveKind + device_objective_supported (SC #5) | ✓ VERIFIED | 173 lines; 11-true/7-false partition, re-exported from lib.rs |
| `crates/oracle-harness/tests/objective_common/mod.rs` | parse_gh + read_boosting_golden + read_rank_golden | ✓ VERIFIED | 89 lines; capture-gated skip-pass readers |
| `crates/oracle-harness/tests/fixtures/rank/lambdarank_gh_iter{1,N}.txt` | real lib_lightgbm grad/hess goldens (GRAD/HESS bit lines) | ✓ VERIFIED | Both present; GRAD/HESS u32-bits format confirmed |

### Key Link Verification

| From | To | Via | Status | Details |
| --- | --- | --- | --- | --- |
| `lib.rs` | `device_objective.rs` | `pub mod` + re-export | ✓ WIRED | `pub mod device_objective;` (L14) + `pub use ...device_objective_supported` (L21) |
| `kernels/mod.rs` | 4 objective modules | ungated `pub mod` | ✓ WIRED | All four `pub mod objective_*;` present, NOT `#[cfg(feature="gpu")]` (D-08) |
| `objective_regression.rs` | `primitives.rs` | reduce/percentile composition | ✓ WIRED | 36 composition refs; no `percentile_device` symbol referenced (4 hits are doc comments documenting Discrepancy 1) |
| `objective_rank.rs` | `random.rs` / `primitives.rs` | `draw_next_float_on` + `bitonic_argsort_items_on` | ✓ WIRED | 5 + 7 refs respectively |
| parity tests | `*_gh` goldens | read_*_golden + parse_gh + compare | ✓ WIRED | All boosting + rank goldens present on disk → assertions execute (not skip-passed) |

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
| --- | --- | --- | --- |
| SC #5 discriminator partition | `cargo test -p lgbm-compute --lib device_objective` | 2 passed | ✓ PASS |
| Regression family parity | `cargo test -p oracle-harness --test objective_parity_regression` | 6 passed | ✓ PASS |
| Binary parity | `cargo test -p oracle-harness --test objective_parity_binary` | 4 passed | ✓ PASS |
| Multiclass parity | `cargo test -p oracle-harness --test objective_parity_multiclass` | 3 passed | ✓ PASS |
| Ranking parity + RNG-replay | `cargo test -p oracle-harness --test objective_parity_rank` | 3 passed | ✓ PASS |
| No host-path regression | `cargo test --workspace --lib --tests -j 4` | 909 passed / 0 failed | ✓ PASS |

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
| --- | --- | --- | --- | --- |
| ODL-05 | 19-01 | On-device regression-family grad/hess + Convert/Boost/Renew | ✓ SATISFIED | 6/6 regression parity cells pass |
| ODL-06 | 19-02 | On-device binary-logloss + BoostFromScore + sigmoid + OVA reset | ✓ SATISFIED | 4/4 binary parity cells pass |
| ODL-07 | 19-03 | On-device multiclass softmax + OVA class-major | ✓ SATISFIED | 3/3 multiclass parity cells pass |
| ODL-08 | 19-04 | On-device ranking LambdaRank + RankXENDCG + RNG | ✓ SATISFIED | 3/3 rank parity cells pass |

All four PLAN-declared requirement IDs (ODL-05..08) are accounted for. REQUIREMENTS.md
maps exactly ODL-05, ODL-06, ODL-07, ODL-08 to Phase 19 (all marked Complete) — no
orphaned requirements. ODL-19 (f64-hot-loop / byte-unchanged merge gate) is mapped to
Phase 21, not this phase; the byte-unchanged portion is nonetheless corroborated here by
the 909/0 workspace pass with `LGBM_CUDA_ON_DEVICE` unset.

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
| --- | --- | --- | --- | --- |
| `crates/oracle-harness/tests/objective_parity_rank.rs` | — | 3 benign unused-import warnings | ℹ️ Info | Test-only; no behavior impact (noted in task context) |

No `TBD`/`FIXME`/`XXX`/`todo!`/`unimplemented!` debt markers in any phase source file.
No stubs, no hollow data paths — every kernel is wired to the f64 anchor and gated by a
passing parity test.

### Human Verification Required

None. Every behavior-dependent truth (grad/hess computation, RNG stream determinism,
class-major state layout, buffer aliasing) is exercised by a passing anchor-parity test
against real lib_lightgbm 4.6 goldens (bit-exact where arithmetic-only; within ORACLE_TOL
for documented transcendental/atomic residuals). No visual/real-time/external-service
surface exists.

### Gaps Summary

No gaps. All five ROADMAP success criteria are observably true in the codebase, all four
requirement IDs are satisfied and parity-tested, and the default host path is
byte-unchanged (909/0 workspace pass).

**Scope note (informational, not a gap):** The goal's trailing clause "so the boosting
layer never round-trips gradients to host" describes the eventual outcome. Per decision
D-02, the objectives are delivered STANDALONE this phase (no GBDT/boosting-seam wiring);
the actual driver integration is Phase 21 (ODL-18/ODL-19). This is by design and
consistent with SC #5 (`on_device_growth_supported()` stays `false`, off by default). The
phase contract — on-device, anchor-pinned kernels + host-fallback discriminator — is fully
met; the wiring that realizes the "never round-trips" purpose is correctly deferred.

---

_Verified: 2026-07-02_
_Verifier: Claude (gsd-verifier)_
