---
phase: 07-parity-completing-variants
plan: 13
subsystem: tree-learner
tags: [tree-learner, split-gain, partition-count, leaf-splits, is-splittable, goss, subtraction-trick, min-data-in-leaf, fp-execution-trace, lightgbm-4.6, oracle-parity, numerical-fidelity, def-07-02, def-07-03]

# Dependency graph
requires:
  - phase: 07-01
    provides: "the source-built lib_lightgbm 4.6 CPU-only single-thread FP-execution-trace technique (D-05) — reused here to localize the GOSS subtracted-child splittability divergence"
  - phase: 07-02
    provides: "the deferred DEF-07-02 fair + quantile Family-A cells (the un-defer target)"
  - phase: 07-03
    provides: "the deferred DEF-07-02 extension: gamma + tweedie Family-A cells"
provides:
  - "DEF-07-02/03 Family A CLOSED for 12 of 13 cells: all fair, all gamma, both tweedie, and quantile_alpha_axis un-#[ignore]d and asserting real-lib_lightgbm-4.6 parity"
  - "ROOT CAUSE (source-built FP trace), corrected from the plan's diagnosed histogram-compaction theory: TWO learner-side count-SOURCE bugs, not an f64 split-gain operand knife-edge and not the most_freq_bin/offset histogram representation"
  - "Fix 1 (15263df): split_inner seeds smaller/larger leaf-splits by DATA-PARTITION counts (part_left < part_right), not SplitInfo round_int(hess*cnt_factor) counts — C++ uses partition counts for both the tie-break (serial_tree_learner.cpp:790-791/851, update_cnt=true) and the histogram-pool dance (GetGlobalDataCountInLeaf)"
  - "Fix 2 (56c31c7): propagate the parent is_splittable_ gate to subtracted children (serial_tree_learner.cpp:395-399) — under GOSS amplification cnt_factor rounds per-bin counts to 0 so a feature failing min_data_in_leaf at the parent can look splittable on a subtracted child; this also closed fair_loop_matrix (downstream, not a 3rd defect)"
  - "DEF-07-13-01: the lone remaining bagged-renew sub-cell quantile_bag1_es0_bfa0 re-scoped as a distinct bagging-draw × RenewTreeOutput structural divergence (CLI cannot reproduce its golden)"
affects: [08]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Partition-count parity invariant: every learner decision that C++ keys off GetGlobalDataCountInLeaf / the update_cnt-overwritten best_split_info counts (smaller/larger leaf-splits tie-break, histogram-pool slot dance, tree-node counts) must use the data-partition counts, NOT the SplitInfo round_int(hess*cnt_factor) counts — the two diverge for non-constant/amplified hessians"
    - "Subtracted-child splittability gate: the histogram-subtraction-trick child inherits the parent is_splittable_ flag; a feature not splittable at the parent (per-bin counts round to 0 under min_data_in_leaf) must not be re-scanned on the subtracted child even when its larger cnt_factor no longer rounds to 0"
    - "Source-built FP trace as arbiter (07-01/D-05), extended to the GOSS bagging/amplification split-count path; CLI-non-reproducible goldens (quantile+bfa-off) fall outside this method and need a wheel-side oracle"

key-files:
  created:
    - .planning/phases/07-parity-completing-variants/07-13-PLAN.md
    - .planning/phases/07-parity-completing-variants/07-13-SUMMARY.md
  modified:
    - crates/lgbm-treelearner/src/learner.rs
    - crates/oracle-harness/tests/boosting_parity.rs
    - .planning/phases/07-parity-completing-variants/deferred-items.md

key-decisions:
  - "ROOT CAUSE CORRECTED via the source-built FP trace: the plan diagnosed a most_freq_bin==0/offset histogram-representation divergence; the trace proved the histograms are correct and the real defects are two count-SOURCE bugs in the learner. The plan explicitly left the fix mechanism to executor judgment (binding constraint = byte-equality with the C++ trace + Family-A goldens), so the corrected fix is in-scope, not a Rule-4 escalation."
  - "Fix 1 — partition-count seeding in split_inner (15263df): compare part_left < part_right. Non-constant hessians made SplitInfo round_int(hess*cnt_factor) counts disagree with partition counts on the tie/±1 case (gamma node{0,1}|{2,3}: SplitInfo (1,3) vs partition (2,2)), attaching the wrong child's sums to the histogram and flipping the gain (1.0417 vs 0.0333)."
  - "Fix 2 — parent-splittability gate for subtracted children (56c31c7): the correct partition-count tie-break un-masked a latent GOSS defect (it had been coincidentally compensated). Under amplification cnt_factor is small → per-bin counts round to 0 → a feature fails min_data_in_leaf at the parent but looks splittable on a subtracted child. C++ propagates is_splittable_ (serial_tree_learner.cpp:395-399); Rust scanned unconditionally. Fixing it restored goss_parity_matrix bit-exact AND closed fair_loop_matrix (tree-5 ~2.085, downstream of the same gate)."
  - "1 of 13 cells re-scoped, not weakened: quantile_bag1_es0_bfa0 (bagged-renew) is a distinct bagging-draw × quantile-RenewTreeOutput structural divergence (12-vs-10 trees). The deterministic source-built CLI does not reproduce its 10-tree golden (stops at 1 tree for quantile+bfa-off), so the D-05 FP-trace method cannot localize it. Left #[ignore]d with an honest reason, tracked as DEF-07-13-01."

patterns-established:
  - "Partition-count vs SplitInfo-count parity invariant for the serial learner"
  - "Subtracted-child is_splittable_ inheritance (min_data_in_leaf under amplification)"

requirements-completed: []

# Metrics
duration: ~3 sessions (debug FP-trace root-cause → plan → execute with a multi-agent pause/resume)
completed: 2026-06-08
---

# Phase 7 Plan 13: Family-A Learner Count-Source Parity Fix Summary

**The DEF-07-02/03 Family-A divergence (fair/gamma/tweedie/quantile) was NOT the f64 split-gain operand knife-edge nor the most_freq_bin==0/offset histogram representation the plan diagnosed — a source-built `lib_lightgbm` 4.6 FP execution trace proved the histograms are correct and localized TWO learner-side count-SOURCE bugs, both fixed C++-faithfully: (1) `split_inner` seeded the smaller/larger leaf-splits from SplitInfo `round_int(hess·cnt_factor)` counts instead of the data-partition counts C++ uses for both the tie-break and the histogram-pool dance; (2) the histogram-subtraction child did not inherit the parent `is_splittable_` gate, so under GOSS amplification a feature failing `min_data_in_leaf` at the parent was re-scanned on a subtracted child. Fixing both un-ignores 12 of 13 Family-A cells at real-binary parity with the bit-exact CPU merge gate fully GREEN; the lone remaining bagged-renew sub-cell is a distinct structural divergence re-scoped to DEF-07-13-01.**

## Performance

- **Duration:** ~3 sessions — `/gsd-debug` FP-trace root-cause, `/gsd-plan-phase` (07-13), then `/gsd-execute-phase`-style execution interrupted by a concurrent Phase-8 session (paused at the user's direction, resumed after Phase-8 settled).
- **Completed:** 2026-06-08

## Root cause (source-built FP execution trace, 07-01/D-05 method)

Reused the retained `/tmp/LightGBM` CPU-only single-thread build (the repo's read-only `LightGBM/` reference tree was never git-added or touched; C++ instrumentation env-gated and reverted after capture). The trace reproduced the gamma + GOSS goldens bit-exact, then exposed that the per-bin histograms are correct — the divergences are count-source bugs:

1. **Partition-count seeding (`15263df`).** C++ overwrites `best_split_info.left/right_count` with the data-partition counts (`serial_tree_learner.cpp:790-791`, `update_cnt=true`) before the `:851` tie-break, and the histogram-pool slot dance uses `GetGlobalDataCountInLeaf` (= `data_partition_->leaf_count`). Rust's `split_inner` instead keyed the smaller/larger tie-break off the SplitInfo `round_int(hess·cnt_factor)` counts. For non-constant hessians the two disagree on the tie/±1 case (gamma node{0,1}|{2,3}: SplitInfo (1,3) vs partition (2,2)) → wrong child's sums attached to the histogram → gain flips 1.0417 vs 0.0333 → tree-0 topology `[1,8,2,1]` vs golden `[2,4,2,4]`. Fix: compare `part_left < part_right`.

2. **Subtracted-child splittability gate (`56c31c7`).** The correct partition-count tie-break un-masked a latent GOSS defect it had been coincidentally compensating. Under GOSS amplification `cnt_factor = num_data / amplified_sum_hessian` is small, so per-bin `round_int(hess·cnt_factor)` rounds to 0 and a feature can fail `min_data_in_leaf` at the parent (not splittable) yet look splittable on a subtracted child whose larger `cnt_factor` no longer rounds to 0. C++ propagates the parent `is_splittable_` flag to the subtracted child (`serial_tree_learner.cpp:395-399`); Rust scanned all features unconditionally and selected a split C++ never considers (GOSS root f1 `best_gain=-inf` → C++ picks f0 gain 1.14; Rust picked f1 gain 4.36 → tree-10 topology flip → tree-11 score mismatch). Fixing the gate restored `goss_parity_matrix` bit-exact AND closed `fair_loop_matrix` (tree-5 ~2.085 — downstream of the same gate, NOT a third defect).

## What shipped

- **12 of 13 Family-A cells un-ignored** (`8a4a5af`) asserting real-lib_lightgbm-4.6 parity: `fair_*` (5), `gamma_*` (4), `tweedie_loop_matrix` + `tweedie_variance_power_axis`, `quantile_alpha_axis`.
- Both fixes are in `crates/lgbm-treelearner/src/learner.rs`; the `offset==0` path, the f64/f32 split.rs consumers, threshold recording, partition, and predict routing were left in sync (the histogram representation the plan suspected was correct as-is and was NOT changed).

## Deviations from Plan

### [Rule 1 — corrected diagnosis] Root cause was count-source, not histogram representation
**Found during:** Task 2 FP-trace investigation.
**Issue:** the plan (and the originating debug session) diagnosed a most_freq_bin==0/offset histogram-compaction bug. The trace proved the histograms are correct; the defect is the SplitInfo-vs-partition count source in `split_inner`.
**Fix:** one-line partition-count tie-break (`15263df`). In-scope — the plan named `split_inner`/`learner.rs` and left the mechanism to executor judgment.

### [Rule 2 — missing fix for an un-masked second defect] GOSS subtracted-child splittability gate
**Found during:** Task 4 no-regression run — `goss_parity_matrix` regressed once the count-source fix landed.
**Issue:** the old buggy tie-break had been coincidentally compensating a missing `is_splittable_` propagation to subtracted children under GOSS amplification.
**Fix:** propagate the parent gate (`56c31c7`); restored goss bit-exact and additionally closed `fair_loop_matrix`.

### [Re-scope — not a weakening] quantile bagged-renew left deferred
**Found during:** Task 3/4 — `quantile_bag1_es0_bfa0` is a 12-vs-10-tree STRUCTURAL divergence (bagging-draw × quantile-`RenewTreeOutput`), not the count/offset path. The deterministic source-built CLI does not reproduce its golden, so the D-05 method cannot localize it. Assertion left intact under `#[ignore]`; re-scoped to **DEF-07-13-01** (the plan's "clear all DEF-07-02/03" adjusted to "clear the 12 closed; re-scope the 1 distinct cell").

**Total deviations:** 2 in-scope fixes (1 corrected-diagnosis Rule-1, 1 un-masked-second-defect Rule-2) + 1 honest re-scope. No tolerance weakened, no horizon capped, no blanket skip.

## Verification

- **Full `LGBM_CAPTURE_PYTHON=… cargo test --workspace`** (independently re-run by the orchestrator at the Task 4 blocking-human gate): **0 failed**.
- `boosting_parity` — 73 passed, 0 failed, **1 ignored** (`quantile_loop_matrix` / DEF-07-13-01).
- `learner_parity` — 25 passed, 0 failed, 4 ignored (Family-B DEF-07-11, untouched).
- `kernel_parity` 4/4 bit-exact; `goss_parity_matrix` GREEN (regression cleared); `subset_determinism_diagnostic` + all `*_spine`/`*_gradients` GREEN; `mfb_zero_offset_histogram_contract` PASS; `lgbm-treelearner` 64/64; `lgbm-compute` GREEN.
- `git status --porcelain LightGBM/` — never git-added; `/tmp/LightGBM` instrumentation reverted.

## Task Commits

1. `194abb3` — `test(07-13)`: pin mfb==0/offset histogram+scan contract as failing diagnostic.
2. `15263df` — `fix(07-13)`: seed smaller/larger leaf-splits by partition counts, not SplitInfo counts.
3. `56c31c7` — `fix(07-13)`: parent-splittability gate for subtracted children (GOSS 2nd defect).
4. `8a4a5af` — `test(07-13)`: un-ignore 12 Family-A cells; quantile bagged-renew stays deferred.
5. `a53daa6` — `docs(07-13)`: clear DEF-07-02/03 (12 cells closed); re-scope quantile bagged-renew to DEF-07-13-01.

## Self-Check: PASSED

- `07-13-PLAN.md` + `07-13-SUMMARY.md` exist on disk.
- Fix commits `15263df` / `56c31c7` and test commits `194abb3` / `8a4a5af` / `a53daa6` present in history.
- `cargo test --workspace` GREEN (0 failed); 12 Family-A cells un-ignored; merge gate bit-exact; `LightGBM/` never git-added.
- 1 cell (DEF-07-13-01) honestly deferred with assertion intact; Family-B DEF-07-11 untouched.
