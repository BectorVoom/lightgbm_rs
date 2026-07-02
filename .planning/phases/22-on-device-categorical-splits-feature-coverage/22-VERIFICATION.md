---
phase: 22-on-device-categorical-splits-feature-coverage
verified: 2026-07-02T00:00:00Z
status: passed
score: 4/4 roadmap success-criteria verified (+ all plan must-have truths; 1 via superseding implementation)
behavior_unverified: 0
overrides_applied: 0
re_verification:
  previous_status: none
  previous_score: n/a
notes:
  - "22-03 plan truth 'many-vs-many ctr sort uses primitives::bitonic_argsort_on' was intentionally superseded in 22-05 by the host's f64 std::stable_sort (Rule-1 correctness fix: f32 bitonic crashed OOB + diverged from golden on NaN ctr child leaves). The OBSERVABLE truth — many-vs-many golden bit-exact — is VERIFIED. Suggest an override to record the deviation (below)."
  - "GBDT does not yet call learner.with_quantized_grad(...) — the D-06 gate defaults use_quantized_grad=false in production. Benign & documented (22-01 SUMMARY): the on-device path is env-gated OFF in production, so categorical+quantized-on-device cannot occur regardless. Wiring belongs to Phase-23 default-on rollout."
  - "Real-ROCm (cubecl-hip f32) categorical smoke is explicitly D-04 best-effort / non-gating, deferred to Phase 23's Kaggle DoD (VALIDATION.md Manual-Only). The hard gate is the env-unset cubecl-cpu f64 lane, which is fully automated and green."
override_suggestions:
  - must_have: "The many-vs-many ctr sort uses primitives::bitonic_argsort_on (index-only)"
    reason: "Reverted to the host's f64 std::stable_sort in 22-05 (commit 0566370). The f32 bitonic path crashed (OOB on power-of-two padding indices) and diverged from the f64-stable golden order on NaN ctr keys reached at deeper child leaves. The evaluator IS the single-owner f64 anchor (def-f8u-01), so its ctr order MUST equal the host/golden bit-exact. Intent (golden-faithful many-vs-many split) is verified by the passing manyvsmany device + unit cells."
    accepted_by: "<pending human>"
    accepted_at: "<pending>"
---

# Phase 22: On-Device Categorical Splits (Feature Coverage) Verification Report

**Phase Goal:** Categorical splits work end-to-end on the proven numerical driver — bitset construction, categorical split evaluation, categorical partition membership, and SplitCategorical — via a pre-allocated bitset representation (no per-`SplitInfo` device alloc).
**Verified:** 2026-07-02
**Status:** passed (PASS-WITH-NOTES)
**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths (ROADMAP Success Criteria — the contract)

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | On-device categorical **bitset construction** (§6.3) via the **pre-allocated** representation, NOT per-`SplitInfo` alloc | ✓ VERIFIED | `categorical_split.rs::construct_bitset` (no `Atomic`, no alloc) + `set_real_threshold`; grow_driver stages winners into pre-allocated `DeviceSplitInfo` cat slab via `set_cat_thresholds`, derives host bitsets from the slab (`grow_driver.rs:696-720`). `DeviceSplitInfo` allocated only when a categorical feature exists (pure-numeric grows allocate nothing). 14 `categorical_split` unit tests green; slab pinned to real-4.6 golden bitsets (`cat_onehot=8`, `cat_manyvsmany root=56`). |
| 2 | On-device **categorical split eval** (one-hot + many-vs-many) + **partition membership** via `FindInBitset`, anchor-pinned | ✓ VERIFIED | `find_best_threshold_categorical` (both paths, l2 asymmetry, `max_num_cat` clamp); `partition_categorical_on_device` wired at `grow_driver.rs:723`; Open-Q1 routing isolated by `categorical_partition_counts_match_host_stable`. Device parity cells green under `LGBM_CUDA_ON_DEVICE=1`. |
| 3 | `SplitCategorical` **tree mutation** (`num_cat`, `cat_boundaries`) + **predicts correctly**, anchor-pinned | ✓ VERIFIED | `tree.split_categorical_on_device` wired at `grow_driver.rs:762`; device cells assert `num_cat`, `dt&1` kCategoricalMask, `cat_boundaries`, real `cat_threshold` bitset bit-exact vs golden + per-row predict-through (`predict_leaf_index` device vs model-text reparse vs golden) — `learner_parity.rs:1613-1665`. |
| 4 | Numeric spine byte-untouched; CPU/ROCm/host-CUDA byte-unchanged; **merge gate green** | ✓ VERIFIED | `cargo test --workspace` (env unset) = **963 passed / 0 failed** (verifier-run). `learner_parity_categorical_no_regression_numeric_spine` green. All new device cells skip-pass env-unset. |

**Score:** 4/4 roadmap success criteria verified (0 present/behavior-unverified).

All four are behavior-dependent (grow-loop state transitions / partition / tree mutation); each is upgraded to VERIFIED by a passing behavioral test that drives the full device grow loop — not symbol presence alone.

### Plan Must-Have Truths (per-plan detail)

| Plan | Truth | Status |
|------|-------|--------|
| 22-01 | Runtime `cat_width` slab from `config.max_cat_threshold`, no truncation > 32 (D-03) | ✓ VERIFIED (`cat_slab_width_gt_32_no_truncation`) |
| 22-01 | `on_device_eligible` false for categorical+quantized, host log (D-06) | ✓ VERIFIED (`on_device_eligible_false_for_categorical_plus_quantized`) |
| 22-02 | `GrowFeature` carries native categorical metadata, crate-cycle-safe | ✓ VERIFIED (`bin_to_category: Vec<i32>` + 5 scalars, native only; `cargo build -p lgbm-compute` green) |
| 22-03 | `construct_bitset` / `set_real_threshold` bit-identical to host | ✓ VERIFIED (pinned to real-4.6 golden bitsets) |
| 22-03 | Evaluator reproduces host find_best_threshold_categorical both paths (l2 asymmetry, clamp, pre-bump) | ✓ VERIFIED (unit + device cells) |
| 22-03 | many-vs-many ctr sort uses `bitonic_argsort_on` | ⚠️ SUPERSEDED — see note; reverted to f64 `std::stable_sort` in 22-05 (correctness fix). Observable intent (golden bit-exact) VERIFIED. Override suggested. |
| 22-04 | best_split dispatch calls §8.1 evaluator, passes through sum_hessian | ✓ VERIFIED |
| 22-04 | Driver routes categorical, single +2*kEpsilon bump, real+inner bitsets, §9/§10 calls | ✓ VERIFIED (`categorical_driver_bumps_sum_hessian_once`) |
| 22-04 | Thresholds staged into pre-allocated slab, bitsets materialized from slab | ✓ VERIFIED (SC #1) |
| 22-05 | DEVICE tree bit-exact vs real lib_lightgbm 4.6 goldens, both fixtures (D-01 #1) | ✓ VERIFIED |
| 22-05 | DEVICE tree passes cpu-f64 structure gate, tie-aware default_left (D-01 #2) | ✓ VERIFIED |
| 22-05 | Predict-through the categorical bitset round-trips (SC #3) | ✓ VERIFIED |

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `crates/lgbm-compute/src/kernels/categorical_split.rs` | §6.3 bitset + §8.1 evaluator | ✓ VERIFIED | 814 lines; `construct_bitset`, `set_real_threshold`, `find_best_threshold_categorical`; registered in `mod.rs` |
| `crates/lgbm-compute/src/kernels/split_info.rs` | runtime `cat_width` (D-03) | ✓ VERIFIED | `cat_width: usize` threaded through slab sizing/accessors; `MAX_CAT_PER_SPLIT` demoted to default |
| `crates/lgbm-compute/src/kernels/grow_driver.rs` | GrowFeature cat fields + cat grow branch | ✓ VERIFIED | 6 native cat fields; categorical branch calls eval + set_cat_thresholds + set_real_threshold + partition/split_categorical_on_device |
| `crates/lgbm-compute/src/kernels/best_split.rs` | cat dispatch seam filled | ✓ VERIFIED | `is_categorical`/`is_one_hot` seam calls `find_best_threshold_categorical`, maps to SplitScalars |
| `crates/lgbm-treelearner/src/learner.rs` | D-06 gate | ✓ VERIFIED | `on_device_eligible_gate` + `with_quantized_grad` + `refresh_on_device_eligibility` |
| `crates/oracle-harness/tests/learner_parity.rs` | dual-anchor device cells | ✓ VERIFIED | structure-gate cat cases + `learner_parity_categorical_{onehot,manyvsmany}_on_device` |
| `fixtures/categorical/{cat_onehot,cat_manyvsmany}.{txt,bins.json}` | real 4.6 goldens | ✓ PRESENT | committed, loaded (not SKIP) |

### Key Link Verification

| From | To | Via | Status |
|------|----|----|--------|
| grow_driver categorical branch | tree.rs `split_categorical_on_device` + data_partition `partition_categorical_on_device` | §9/§10 entrypoints w/ real+inner bitsets | ✓ WIRED (`grow_driver.rs:723,762`) |
| grow_driver | `DeviceSplitInfo` cat slab | `set_cat_thresholds` → `set_real_threshold` (allocate-once) | ✓ WIRED (`grow_driver.rs:708,718`) |
| best_split cat seam | categorical_split evaluator | `find_best_threshold_categorical` → SplitScalars | ✓ WIRED (`best_split.rs:67,240`) |
| learner D-06 | config.use_quantized_grad | `with_quantized_grad`/`refresh_on_device_eligibility` | ✓ WIRED at learner; ⚠️ NOT called from GBDT (benign deferral — see note) |

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| categorical_split unit math | `cargo test -p lgbm-compute --lib categorical_split` | 14 passed / 0 failed | ✓ PASS |
| Dual-anchor device parity + predict-through | `LGBM_CUDA_ON_DEVICE=1 cargo test -p oracle-harness --test learner_parity -- learner_parity_categorical learner_parity_on_device_structure_gate` | 6 passed / 0 failed | ✓ PASS |
| Merge gate (SC #4, hard gate) | `cargo test --workspace` (env unset) | 963 passed / 0 failed | ✓ PASS |

### Anti-Patterns Found

None. No `TBD`/`FIXME`/`XXX` debt markers, no `todo!()`/`unimplemented!()`, no stub returns on the categorical grow path across all 5 modified source files.

### Requirements Coverage

| Requirement | Description | Status | Evidence |
|-------------|-------------|--------|----------|
| ODL-22 | On-device categorical splits end-to-end (bitset §6.3, eval §8.1, partition §9, SplitCategorical §10) via pre-allocated bitset | ✓ SATISFIED | All 4 SCs verified; dual-anchor (real 4.6 golden + cpu f64 structure) green for one-hot + many-vs-many |

### Deferred / Follow-Up Items (informational — do NOT block this phase)

| Item | Disposition |
|------|-------------|
| GBDT `.with_quantized_grad(...)` wiring | Deferred to Phase-23 default-on rollout. Benign: on-device path env-gated OFF in production; D-06 gate + test already present. |
| Real-ROCm (cubecl-hip f32) categorical smoke | D-04 best-effort / non-gating; Phase-23 Kaggle DoD. Hard gate is the automated env-unset cpu-f64 lane (green). |
| Broaden on-device numeric feature coverage (monotone/extra_trees/col_sampler) so more cells run under env=1 | Documented D-04 posture; pre-existing (verified not a regression in 22-05). |

### Gaps Summary

No gaps. The phase goal — categorical splits end-to-end on the on-device driver via a pre-allocated bitset — is achieved and independently confirmed: all four ROADMAP success criteria verified by verifier-run tests, categorical is dual-anchored to a REAL lib_lightgbm 4.6 reference (first on-device subsystem to be, not a re-transcription), and the env-unset merge gate is byte-green (963 passed). Two documented, benign notes (an intentional bitonic→f64-stable-sort correctness deviation that IMPROVES fidelity, and the Phase-23 GBDT quantized-grad wiring seam) warrant recording but do not reduce success-criteria coverage.

---

_Verified: 2026-07-02_
_Verifier: Claude (gsd-verifier)_
