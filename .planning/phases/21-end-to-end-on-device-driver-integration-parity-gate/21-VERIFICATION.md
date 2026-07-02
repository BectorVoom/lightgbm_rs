---
phase: 21-end-to-end-on-device-driver-integration-parity-gate
verified: 2026-07-02T07:21:14Z
status: passed
score: 9/9 must-haves verified
behavior_unverified: 0
overrides_applied: 0
---

# Phase 21: Harden the On-Device Driver Verification Report

**Phase Goal:** Harden the just-landed on-device grow loop on the continuous-feature slice — confirm the WR-01 HistArena::swap free-slot fix, broaden STRUCTURE parity evidence with targeted risk cases (deep >2-live-leaf, no-split break, min-data/min-hessian-constrained) each pinned to the cpu f64 anchor, and reconcile the ODL-18/19 requirement/ROADMAP bookkeeping. No driver re-implementation.
**Verified:** 2026-07-02T07:21:14Z
**Status:** passed
**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths

| #   | Truth                                                                                                    | Status     | Evidence                                                                                                                                                    |
| --- | ------------------------------------------------------------------------------------------------------- | ---------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1   | WR-01 free-slot scan in HistArena::swap confirmed present; both/all repro tests pass                    | ✓ VERIFIED | `let occupied` + `let fresh = (0..` free-slot scan at histogram_arena.rs:388-390 inside `swap`; doc block :322-354 cites c9a7fd1. 3 repro tests + 15 total green env-unset. |
| 2   | grow_tree_on_device_driver_with_cfg accepts explicit GainConfig; base driver delegates via proving_slice_config() | ✓ VERIFIED | New pub fn at grow_driver.rs:456 with trailing `cfg: GainConfig`; delegator :424-441 calls `_with_cfg(..., proving_slice_config())`. Count == 1.            |
| 3   | Backend::grow_tree_on_device trait seam byte-unchanged (no GainConfig threaded)                          | ✓ VERIFIED | `git diff HEAD -- lib.rs` empty; trait sig at lib.rs:1294 takes only g/h/features/num_leaves/max_depth, returns `Ok(None)`. No cfg on seam.                 |
| 4   | On-device tree STRUCTURE bit-exact to cpu f64 anchor with >2 simultaneously-live leaves                  | ✓ VERIFIED | `learner_parity_on_device_deep_multileaf_gate` PASSES env=1 (non-vacuous): asserts vs `cpu_anchor_tree` + `num_leaves>=5` + row-conservation; PASSES env-unset (Ok(None)). |
| 5   | No-split break path (best_leaf==-1) grows a root-only tree matching the anchor                           | ✓ VERIFIED | `learner_parity_on_device_nosplit_gate` PASSES env=1: asserts vs anchor + `num_leaves==1` (A3 pin); PASSES env-unset. Behavioral test observed green.       |
| 6   | A min_data/min_sum_hessian constraint observably binds; constrained on-device tree matches constrained anchor | ✓ VERIFIED | `learner_parity_on_device_mindata_gate` PASSES both lanes: env-unset asserts constrained driver==constrained anchor + constrained<unconstrained; env=1 asserts constrained driver < live unconstrained seam. Binds via `min_sum_hessian_in_leaf=3.0` (plan permitted either). |
| 7   | ODL-18 and ODL-19 marked Complete + attributed to Phase 20 in REQUIREMENTS traceability + rollup        | ✓ VERIFIED | REQUIREMENTS.md:106-107 `ODL-18/19 \| Phase 20 \| Complete`; checklist :50-51 `[x]` with Phase-20 6/6 citation; rollup :123 Phase 20 = ODL-16,17,18,19.    |
| 8   | A new ODL-18H hardening requirement exists, mapped to Phase 21                                           | ✓ VERIFIED | Checklist :52 `**ODL-18H**`; traceability :108 `ODL-18H \| Phase 21`; rollup :124 Phase 21 = ODL-18H; coverage :130-133 = 23 total, 100% mapped, no orphans. |
| 9   | ROADMAP Phase 21 body reflects the hardening scope, not the stale driver-integration text               | ✓ VERIFIED | ROADMAP.md:289 heading "Harden the On-Device Driver"; Goal :291 matches phase goal; 5 Success Criteria :295-300; Notes defer categorical→P22, perf→P23; stale "reconstitutes" text absent from P21 section. |

**Score:** 9/9 truths verified (0 present, behavior-unverified)

### Required Artifacts

| Artifact                                             | Expected                                                          | Status     | Details                                                                            |
| ---------------------------------------------------- | ---------------------------------------------------------------- | ---------- | --------------------------------------------------------------------------------- |
| `crates/lgbm-compute/src/kernels/grow_driver.rs`     | `_with_cfg` extract-param variant + delegating base driver        | ✓ VERIFIED | `fn grow_tree_on_device_driver_with_cfg` (1), delegator wires proving_slice_config() |
| `crates/lgbm-compute/src/kernels/histogram_arena.rs` | WR-01 free-slot scan confirmed + documented                       | ✓ VERIFIED | `let occupied` + `let fresh = (0..` in `swap`; c9a7fd1 doc block; repro tests green |
| `crates/oracle-harness/tests/learner_parity.rs`      | 3 D-02 STRUCTURE gate tests + 3 corpus fns, cpu-f64-anchored, env-gated | ✓ VERIFIED | deep_multileaf/nosplit/mindata gates + corpus fns present; both lanes pass         |
| `.planning/REQUIREMENTS.md`                           | ODL-18/19 Complete (Phase 20); ODL-18H rows; coverage 23/100%     | ✓ VERIFIED | All rows present, counts consistent                                               |
| `.planning/ROADMAP.md`                               | Re-cut Phase 21 body for hardening scope                          | ✓ VERIFIED | Heading + Goal + 5 SC + Notes re-cut; other phase bodies untouched                 |

### Key Link Verification

| From                                   | To                                            | Via                                         | Status | Details                                            |
| -------------------------------------- | --------------------------------------------- | ------------------------------------------- | ------ | -------------------------------------------------- |
| grow_tree_on_device_driver             | grow_tree_on_device_driver_with_cfg           | delegation w/ proving_slice_config()        | WIRED  | grow_driver.rs:432-440                              |
| each new gate test                     | cpu_anchor_tree + assert_on_device...anchor   | tie-aware structure comparator vs cpu f64   | WIRED  | reused verbatim at :2465-2466, :2547-2548, :2670-2671 |
| mindata gate                           | grow_tree_on_device_driver_with_cfg           | direct call w/ constrained GainConfig       | WIRED  | learner_parity.rs:2634 direct call, bypasses seam  |

### Behavioral Spot-Checks

| Behavior                                          | Command                                                                        | Result             | Status |
| ------------------------------------------------- | ------------------------------------------------------------------------------ | ------------------ | ------ |
| WR-01 repro tests                                 | `cargo test -p lgbm-compute --lib histogram_arena`                             | 15 passed, 0 failed | ✓ PASS |
| 3 new gates (merge-gate lane)                     | env-unset `cargo test -p oracle-harness --test learner_parity -- <gates>`      | 3 passed            | ✓ PASS |
| 3 new gates (non-vacuous on-device lane)          | `LGBM_CUDA_ON_DEVICE=1 ... -- deep_multileaf/nosplit/mindata`                   | 3 passed            | ✓ PASS |
| Full learner_parity merge gate (env-unset)        | `cargo test -p oracle-harness --test learner_parity`                           | 35 passed, 0 failed | ✓ PASS |
| Trait seam byte-unchanged                         | `git diff HEAD -- crates/lgbm-compute/src/lib.rs`                              | empty               | ✓ PASS |

### Requirements Coverage

| Requirement | Source Plan       | Description                                | Status      | Evidence                                                              |
| ----------- | ----------------- | ------------------------------------------ | ----------- | -------------------------------------------------------------------- |
| ODL-18H     | 21-01/21-02/21-03 | On-device driver hardening + parity corpus | ✓ SATISFIED | All Phase-21 code + bookkeeping delivered; gates green both lanes     |
| ODL-18      | 21-03             | On-device driver (delivered Phase 20)      | ✓ SATISFIED | Re-attributed Complete → Phase 20 (20-VERIFICATION 6/6) per D-01      |
| ODL-19      | 21-03             | No-f64-per-row merge gate (delivered Phase 20) | ✓ SATISFIED | Re-attributed Complete → Phase 20 per D-01                            |

All three PLAN requirement IDs accounted for in REQUIREMENTS.md. No orphaned requirements.

### Anti-Patterns Found

| File                                            | Line | Pattern       | Severity | Impact                                                                            |
| ----------------------------------------------- | ---- | ------------- | -------- | --------------------------------------------------------------------------------- |
| crates/oracle-harness/tests/learner_parity.rs   | 2879 | "placeholder" | ℹ️ Info  | Pre-existing comment in an unrelated hip fused test ("placeholder OFF threshold"); not phase-modified code, not a stub. |

No blocking anti-patterns. No unreferenced TBD/FIXME/XXX in phase code.

### Notes

- **Case C constraint mechanism (documented deviation):** The must-have truth permits `min_data_in_leaf / min_sum_hessian_in_leaf` (either). The executor bound via `min_sum_hessian_in_leaf=3.0` (min_data's `min_data*2` both-too-small pre-gate diverges between driver and learner on tiny corpora) and moved the constrained STRUCTURE-parity assert to the env-unset lane (under LGBM_CUDA_ON_DEVICE=1 the host anchor forks to the cfg-less seam and drops the cfg). Both lanes remain non-vacuous — verified by running both. Truth satisfied.
- **Bookkeeping drift (info):** SUMMARY 21-03 claimed the ODL-18H traceability row and checklist were left "Pending"/unchecked; on-disk they are now `Complete`/`[x]` (a subsequent phase-completion marker). Direction is more-complete, not a gap — all three requirement IDs are consistently accounted for.

### Gaps Summary

No gaps. All 9 must-have truths are VERIFIED with behavioral evidence. The three STRUCTURE parity gates pass non-vacuously under `LGBM_CUDA_ON_DEVICE=1` and pass the byte-unchanged env-unset merge gate (35/35 learner_parity, 15/15 histogram_arena). The WR-01 fix is confirmed present and documented; the additive `_with_cfg` variant exists with the trait seam byte-unchanged (empty lib.rs diff); the REQUIREMENTS/ROADMAP bookkeeping is reconciled. The phase goal — harden the on-device grow loop via WR-01 confirmation, broadened targeted parity evidence, and reconciled bookkeeping, with no driver re-implementation — is achieved.

---

_Verified: 2026-07-02T07:21:14Z_
_Verifier: Claude (gsd-verifier)_
