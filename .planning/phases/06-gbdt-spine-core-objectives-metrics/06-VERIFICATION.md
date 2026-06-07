---
phase: 06-gbdt-spine-core-objectives-metrics
verified: 2026-06-07T00:00:00Z
status: gaps_found
score: 3/5 must-haves verified
overrides_applied: 0
gaps:
  - truth: "SC#1 — same-tree structural match / bit-exact model text vs C++ (Rust-native train→predict within ~1e-6)"
    status: partial
    reason: >
      Predictions and leaf_value arrays are bit-exact/within-tol, but the
      serialized MODEL TEXT is NOT byte-exact vs C++ for any model containing a
      constant (num_leaves=1) tree. Tree::as_constant hardcodes leaf_count=vec![0];
      C++ AsConstantTree(val, num_data_) sets leaf_count_[0]=num_data and ToString
      ALWAYS emits leaf_count= (no single-leaf write-side early return,
      tree.cpp:363). The committed golden regression_l1_bag1_es0_bfa0_model.txt
      contains a constant Tree=0 with leaf_count=12 where Rust emits leaf_count=0.
      No test in the suite byte-compares emitted model text (assert_model_and_pred
      and the matrix replay both parse-then-compare leaf_value ONLY), so the
      bit-exact model-text claim is unverified at the serialization layer and
      provably wrong for the constant-tree case. (CR-01)
    artifacts:
      - path: "crates/lgbm-model/src/tree.rs:658"
        issue: "as_constant() takes no count arg; sets leaf_count: vec![0] (C++ uses num_data)"
      - path: "crates/lgbm-boosting/src/gbdt.rs:290,362,420"
        issue: "Three Tree::as_constant(const_val) call sites pass no count"
      - path: "crates/oracle-harness/tests/boosting_parity.rs:206-209,1026-1041"
        issue: "Model comparison is leaf_value-only; never byte-compares model text / leaf_count"
    missing:
      - "Give as_constant a count parameter and thread self.num_data at all 3 gbdt.rs call sites (match gbdt.cpp:430/433)"
      - "Add a byte-for-byte model-text assertion against at least one constant-tree golden so the bit-exact contract is actually enforced"
  - truth: "SC#3 / SC#5 — regression_l1 + bagging computes correct (median-residual) leaf values, AND the D-07 matrix cells are 'VALIDATED within ORACLE_TOL, never silently dropped'"
    status: partial
    reason: >
      Two compounding defects. (a) WR-03: in the use_subset (bagging) branch the
      `if self.objective.is_renew_tree_output()` block (gbdt.rs:315-322) is EMPTY —
      regression_l1 + bagging leaves carry the learner Newton output, not the median
      residual the objective requires, so those leaf values are numerically wrong vs
      C++. (b) WR-01: the matrix residual cells call compare_within(...).ok()
      (boosting_parity.rs:985,1010), DISCARDING the Result — 22 of ~40 matrix cells
      (every regression_l1, every bagged non-regression, every multiclass-es cell)
      assert NOTHING numerically; only cells_checked is incremented. The SUMMARY
      (06-05 lines 64,120) and in-code comments claim these are "VALIDATED within
      ORACLE_TOL ... never silently dropped" — contradicted by the code: a gross
      regression of any magnitude passes silently. The "full ~40-cell D-07 replays
      vs the real binary" headline overstates coverage: 18 cells assert, 22 do not.
    artifacts:
      - path: "crates/lgbm-boosting/src/gbdt.rs:314-322"
        issue: "Empty if-block: regression_l1 RenewTreeOutput silently skipped on bagging path"
      - path: "crates/oracle-harness/tests/boosting_parity.rs:980-986,1004-1011"
        issue: ".ok() discards compare_within Result — 22 matrix cells assert nothing"
    missing:
      - "Replace .ok() with .unwrap_or_else(panic) (or an explicit documented residual tolerance asserted against THAT) so residual cells enforce a real bound"
      - "Either apply the median-residual renewal on the subset path, or make regression_l1+bagging a typed BoostingError instead of a silent wrong-leaf no-op"
deferred:
  - truth: "bagging_by_query draw (query-grouped bagging)"
    addressed_in: "Phase 7"
    evidence: "ROADMAP BST-03 scope note + REQUIREMENTS.md BST-04/05/06 + Phase 7 ranking objectives; Phase 6 typed-rejects bagging_by_query=true (decision-backed deferral, not a silent drop)"
human_verification: []
---

# Phase 6: GBDT Spine + Core Objectives/Metrics Verification Report

**Phase Goal:** The first end-to-end ~1e-6 (f32) train→predict run — the simplest boosting variant proves the full spine before any variant is added.
**Verified:** 2026-06-07
**Status:** gaps_found
**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths (ROADMAP Success Criteria)

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | Rust-native API (Dataset, Booster, train, predict) trains GBDT + predicts within ~1e-6 (f32) of C++ with same-tree structural match | ✗ PARTIAL | Predictions/leaf_values bit-exact (spine_end_to_end, regression/binary/multiclass replays). BUT model-text NOT byte-exact for constant trees (CR-01): tree.rs:658 leaf_count=0 vs C++ num_data; golden `regression_l1_bag1_es0_bfa0_model.txt` has leaf_count=12. No test byte-compares model text (only leaf_value). "Bit-exact model text / structural match" unverified + provably wrong for constant trees |
| 2 | GBDT loop (TrainOneIter, UpdateScore, per-class trees, shrinkage, boost_from_average) + score updater deterministic reduction | ✓ VERIFIED | gbdt.rs:829 lines (train_one_iter mirrors TrainOneIter ordering); score_updater.rs f64 train-path scatter; boost_from_average default=true (config/mod.rs:388); score_accumulation tests bit-exact (boosting_parity.rs:353,742); per-class num_data*cur_tree_id offset present |
| 3 | Core objectives (regression, regression_l1, binary, multiclass, multiclassova, custom) grad/hess, ConvertOutput, BoostFromScore, reg_sqrt within ~1e-6 | ✗ PARTIAL | grad/hess validated for all 6 objectives (gradients tests, custom cross-anchor). BUT regression_l1+bagging produces WRONG leaves (WR-03 empty renew block, gbdt.rs:315-322) and the cells that should catch it assert nothing (WR-01). reg_sqrt code exists (regression.rs:42-63) but NO golden exercises reg_sqrt=1 (all goldens [reg_sqrt: 0]) — unverified |
| 4 | Core metrics (l1,l2,rmse,binary_logloss,binary_error,auc,multi_logloss) + multi-metric infra (metric_freq, training-metric eval) match; early stopping fires identically | ✗ PARTIAL | Metrics + metric_freq cadence + multi-metric verified (metric tests, metric_freq_thins_eval_history). BUT early-stop DECISION is gated by metric_freq (booster.rs:516-540) — C++ runs valid-eval + stop EVERY iter when ES on, independent of metric_freq (gbdt.cpp:574). With metric_freq>1 best_iteration + trailing-trim diverge. Latent: all goldens use metric_freq=1; no test crosses metric_freq>1 with ES (CR-02) |
| 5 | Bagging / row subsampling (fraction/freq/seed, pos/neg, bagging_by_query) selects same rows via RNG-matching sequence + call order | ✓ VERIFIED | bagging_rng test asserts bag_data_indices BIT-EXACT (compare_exact i32) vs RNG-replay golden bag_indices_seed3_frac0.7.txt (3 cells); per-block Random(seed+i) block 1024 over FND-01 Random; bagging_by_query=true typed-rejected (decision-backed Phase-7 deferral, not silent) |

**Score:** 3/5 truths verified (SC#2, SC#5 full; SC#1, SC#3, SC#4 partial). SC#4's metric_freq=1 path is correct; the defect is the latent metric_freq>1 + ES interaction.

### Deferred Items

| # | Item | Addressed In | Evidence |
|---|------|--------------|----------|
| 1 | bagging_by_query (query-grouped draw) | Phase 7 | ROADMAP BST-03 scope note; only affects ranking/query objectives (OBJ-04/05/06, all Phase 7); Phase 6 typed-rejects it |

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| crates/lgbm/src/builder.rs | Public training builder → Config | ✓ VERIFIED | 250 lines; metric_freq/bagging/es setters |
| crates/lgbm/src/booster.rs | Booster train/predict + es loop | ⚠️ WIRED w/ defect | 858 lines; es decision gated by metric_freq (CR-02) |
| crates/lgbm-boosting/src/gbdt.rs | GBDT loop | ⚠️ WIRED w/ defect | 829 lines; empty l1-bagging renew block (WR-03) |
| crates/lgbm-boosting/src/score_updater.rs | f64 ScoreUpdater | ✓ VERIFIED | 177 lines; train-path scatter |
| crates/lgbm-boosting/src/sample_strategy.rs | BaggingSampleStrategy | ✓ VERIFIED | 395 lines; RNG-replay bit-exact |
| crates/lgbm-boosting/src/early_stopping.rs | Early-stop decision | ✓ VERIFIED (module) | 293 lines; verbatim C++ arithmetic; defect is in the CALLER's cadence gating |
| crates/lgbm-model/src/tree.rs (as_constant) | Constant tree | ✗ DEFECT | leaf_count=vec![0]; C++ uses num_data (CR-01) |
| crates/lgbm-objective/src/{regression,binary,multiclass,custom}.rs | 6 objectives | ✓ VERIFIED | All present + substantive; grad/hess validated |
| crates/lgbm-metric/src/{binary,multiclass,regression}.rs | 7 metrics | ✓ VERIFIED | All present; per-round values validated |
| goldens (boosting/) | D-07 matrix + spine goldens | ✓ EXIST | ~110 golden files present and parsed |

### Key Link Verification

| From | To | Via | Status |
|------|----|----|--------|
| builder.rs | lgbm_core::Config | from_config / setters | ✓ WIRED |
| gbdt.rs | SerialTreeLearner::train / train_on_subset | per-class growth | ✓ WIRED |
| score_updater.rs | add_prediction_to_score | per-leaf f64 scatter | ✓ WIRED |
| sample_strategy.rs | lgbm_core::Random | per-block next_float draw | ✓ WIRED |
| booster.rs es loop | EarlyStopping::update | factor*score vs best | ⚠️ WIRED but metric_freq-gated (CR-02) |
| gbdt.rs (l1 bagging) | renew_tree_output | median-residual leaf overwrite | ✗ NOT WIRED (empty block, WR-03) |

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| Full workspace suite | `cargo test --workspace` | 426 passed / 0 failed / 0 ignored | ✓ PASS (but suite does not exercise the 3 defect triggers) |
| Constant-tree leaf_count in golden | grep leaf_count in regression_l1_bag1 golden | C++=12, Rust as_constant=0 | ✗ FAIL (divergence confirmed) |
| metric_freq+ES golden | grep metric_freq in capture/manifest | none (default 1 everywhere) | ✗ SKIP (CR-02 trigger never captured) |
| Matrix cell assertion strength | classify D-07 cells | 18 assert / 22 use .ok() (assert nothing) | ✗ FAIL (WR-01 confirmed) |

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|-------------|-------------|--------|----------|
| BST-01 | 06-01,02,04 | GBDT training loop | ✓ SATISFIED | train_one_iter; per-class trees; score_accumulation bit-exact |
| BST-02 | 06-01,02 | Score updater deterministic reduction | ✓ SATISFIED | f64 ScoreUpdater train-path scatter |
| BST-03 | 06-05 | Bagging / row subsampling | ✓ SATISFIED | bagging_rng bit-exact; bagging_by_query Phase-7 deferral |
| BST-07 | 06-05 | Early stopping | ⚠️ PARTIAL | Decision arithmetic correct; metric_freq-gated cadence diverges from C++ (CR-02, latent) |
| OBJ-01 | 06-02,03,04 | Core objectives | ⚠️ PARTIAL | All 6 present + grad/hess validated; l1+bagging leaves wrong (WR-03) |
| OBJ-02 | 06-03 | custom objective | ✓ SATISFIED | custom_objective + cross-anchor-to-L2 tests |
| OBJ-03 | 06-01,02,03,04 | GetGradients/ConvertOutput/BoostFromScore/reg_sqrt | ⚠️ PARTIAL | Core paths validated; reg_sqrt=1 never exercised by any golden |
| MET-01 | 06-02,03,04 | Core metrics | ✓ SATISFIED | l1/l2/rmse/logloss/error/auc/multi_logloss validated |
| MET-02 | 06-05 | Metric infra (metric_freq, training-metric eval) | ✓ SATISFIED | metric_freq cadence + multi-metric + training eval (the eval-history cadence is correct; the ES interaction is CR-02) |
| API-01 | 06-02 | Rust-native API | ✓ SATISFIED | Dataset/Booster/train/predict; spine_end_to_end |

All 10 declared requirement IDs are mapped to Phase 6 in REQUIREMENTS.md (lines 185-194); no ORPHANED requirements. 3 IDs are PARTIAL due to the latent/unverified defects above.

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| crates/lgbm-model/src/tree.rs | 673 | hardcoded leaf_count: vec![0] | 🛑 Blocker | Model-text divergence vs C++ for any constant tree (CR-01) |
| crates/lgbm-boosting/src/gbdt.rs | 314-322 | empty `if {}` (renew skipped) | 🛑 Blocker | Wrong l1+bagging leaf values; silent no-op in a true-branch (WR-03) |
| crates/oracle-harness/tests/boosting_parity.rs | 985,1010 | `compare_within(...).ok()` | 🛑 Blocker | 22/40 matrix cells assert nothing; "validated" claim is false (WR-01) |
| crates/lgbm/src/booster.rs | 516-540 | es decision under metric_freq gate | ⚠️ Warning | best_iteration diverges when metric_freq>1 (CR-02, latent) |
| crates/lgbm-objective/src/regression.rs | 42-63 | reg_sqrt path with no exercising golden | ⚠️ Warning | reg_sqrt fidelity claim unverified |
| crates/lgbm-boosting/src/early_stopping.rs | 15-16 | doc claims metric_freq gates eval | ℹ️ Info | Doc misdescribes C++ (gbdt.cpp:574 runs eval every iter under ES) |

No unreferenced TODO/FIXME/XXX debt markers were found in the modified files that would independently gate the phase; the blockers above are correctness/verification-integrity defects, not debt markers.

### Human Verification Required

None. All defects are programmatically observable against the C++ reference and the committed goldens.

### Gaps Summary

The phase delivers a substantial, well-structured, mostly-faithful GBDT spine: the loop, score updater, all six objectives, seven metrics, bagging RNG (bit-exact), and early-stopping arithmetic are real and validated on the default-config (metric_freq=1, bfa-on, L2) paths. SC#2 and SC#5 are fully verified. However, the phase GOAL — "first end-to-end ~1e-6 (f32) train→predict ... proves the full spine" — is undercut by three confirmed defects that the green 426-test suite does not catch because the goldens never exercise their triggers:

1. **CR-01 (BLOCKER) — constant-tree model text diverges.** `Tree::as_constant` emits `leaf_count=0`; C++ emits `leaf_count=num_data`. The committed golden `regression_l1_bag1_es0_bfa0_model.txt` proves the divergence (leaf_count=12). No test byte-compares emitted model text, so SC#1's "same-tree structural match / bit-exact model text" is both unverified and provably wrong for any model with a constant tree (absent multiclass class, single-class binary, or a no-split first iteration).

2. **WR-01 + WR-03 (BLOCKER) — 22/40 matrix cells assert nothing AND l1+bagging leaves are wrong.** The matrix residual cells discard their `compare_within` Result via `.ok()`, so more than half the D-07 matrix is unasserted despite SUMMARY/comment claims of "VALIDATED within ORACLE_TOL, never silently dropped." Compounding this, the regression_l1 bagging path has an empty renew block, so its leaf values are numerically wrong — and the cells that should catch it are exactly the swallowed ones.

3. **CR-02 (WARNING) — early-stop decision gated by metric_freq.** C++ runs valid-eval + stop check every iteration when ES is on (gbdt.cpp:574); the Rust port gates it behind the metric_freq cadence. Latent because every golden uses metric_freq=1, but it makes SC#4's "early stopping fires identically" false for metric_freq>1.

Recommended resolution order: fix WR-01 first (it is a verification-integrity gap that is currently masking WR-03 and could mask future regressions), then CR-01 and WR-03, then CR-02. The KNOWN/accepted deviations (custom f64 preds, multiclass 5-iter exp-libm cap, RNG-instance reuse) are decision-backed and were NOT reported as gaps.

---

_Verified: 2026-06-07_
_Verifier: Claude (gsd-verifier)_
