---
phase: 07-parity-completing-variants
plan: 14
subsystem: boosting
tags: [gbdt, boosting-loop, no-split-emission, bagging, quantile, tree-count, renew, fp-execution-trace, python-wheel-oracle, lightgbm-4.6, oracle-parity, def-07-13-01]

# Dependency graph
requires:
  - phase: 07-13
    provides: "DEF-07-02/03 Family-A closed (12 cells); the re-scoped DEF-07-13-01 quantile bagged sub-cell this plan closes"
  - phase: debug/remaining-ignored-cells
    provides: "the root cause: GBDT no-split bagged-tree emission policy (Python-wheel oracle), NOT a bagging-draw or renew bug"
provides:
  - "DEF-07-13-01 CLOSED — the LAST ignored parity cell (quantile_loop_matrix / quantile_bag1_es0_bfa0) un-#[ignore]d; cargo test --workspace = 0 failed, 0 IGNORED (full real-lib_lightgbm-4.6 parity)"
  - "GBDT no-split bagged-tree emission policy made C++-faithful: train_one_iter sets should_continue only on a real split, and on a NON-first no-split bagged round pops the round's constant trees + does NOT advance iter (mirrors gbdt.cpp:406-447); first-iter constant baseline preserved"
  - "booster.rs wheel-driver bookkeeping skips the popped round (no duplicate iter_scores push; best_iteration = emitted count)"
affects: [08]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "No-split tree-emission policy: a NON-first boosting round with no positive-gain split (any class) emits NO tree and does NOT advance the iteration counter — pop the round's would-be constants and re-bag next round (gbdt.cpp:406-447). The FIRST iteration keeps its no-split constant baseline (models_.size() < num_tree_per_iteration_ guard)."
    - "Driver-loop bookkeeping must key off EMITTED iterations, not loop rounds: a popped round must be skipped for iter_scores / best_iteration / grad-hess snapshots so they stay aligned to the surviving tree count"
    - "Python-wheel oracle for boosting-loop policy: when the deterministic CLI early-stops (is_finished) and can't reproduce a multi-tree golden, drive TrainOneIter via the lgb.train wheel to capture continue-and-re-bag semantics"

key-files:
  created:
    - .planning/phases/07-parity-completing-variants/07-14-PLAN.md
    - .planning/phases/07-parity-completing-variants/07-14-SUMMARY.md
  modified:
    - crates/lgbm-boosting/src/gbdt.rs
    - crates/lgbm/src/booster.rs
    - crates/oracle-harness/tests/boosting_parity.rs
    - .planning/phases/07-parity-completing-variants/deferred-items.md

key-decisions:
  - "Root cause (debug session, Python-wheel oracle): the quantile bagged divergence is NOT a bagging-draw or quantile-RenewTreeOutput bug (both bit-exact through tree 3). It is the GBDT no-split tree-EMISSION policy — C++ GBDT::TrainOneIter (gbdt.cpp:406-447) pops a no-split round's constant trees and does NOT advance iter_; the lgb.train driver re-bags next round → 10 trees. Rust appended a 1-leaf Tree::as_constant and advanced → 12 trees, shifting every later tree (wheel tree4 == Rust tree5)."
  - "Fix (824d30f): train_one_iter captures should_continue (set true only when a real split grows num_leaves>1, in both the bagging-subset and full-corpus branches) and, when !should_continue AND tree_count_before>0, truncates the round's pushed constants and returns WITHOUT advancing self.iter. IterSnapshot gained an `emitted` flag; the first iteration (tree_count_before==0) keeps the constant baseline and advances (emitted:true), so regression_l1_bag1 Tree=0 + gbdt unit tests are byte-unchanged. Re-bag cadence untouched (bag1 freq=1 re-draws every call; the popped round's draw already advanced bagging_rands, so the next round draws the next bag — C++-faithful)."
  - "booster.rs driver: the real parity driver loop (booster.rs ~1100, not gbdt.rs::train) skips a popped round via `if !snap.emitted { continue; }` BEFORE ran_iters / iter_scores.push / iter_grad_hess.push, preventing a duplicate score push; best_iteration = num_iteration() correctly returns the emitted (10) count. The prev_tree_count skip was already no-new-tree robust."
  - "Scope: bagging draw, RenewTreeOutput, learner, histogram, DART/GOSS/RF paths left byte-unchanged. Only the later-no-split bagged emission path + its driver bookkeeping changed."

patterns-established:
  - "GBDT no-split emission parity (pop + no-iter-advance + re-bag) with emitted-iteration-keyed driver bookkeeping"

requirements-completed: []

# Metrics
duration: ~1 session (root cause carried from the debug session; plan + execute + cross-variant gate)
completed: 2026-06-08
---

# Phase 7 Plan 14: GBDT No-Split Bagged-Tree Emission Parity Summary

**The last deferred parity cell (DEF-07-13-01, `quantile_bag1_es0_bfa0`) was the GBDT no-split bagged-tree emission policy — not a bagging-draw or renew bug. C++ `GBDT::TrainOneIter` pops a no-split round's constant trees and does NOT advance the iteration counter, re-bagging next round (10 trees); Rust appended a 1-leaf constant and advanced (12 trees, shifting every later tree). Replicating the C++ pop / no-iter-advance / re-bag semantics in `train_one_iter` (+ skipping the popped round in the `booster.rs` driver bookkeeping), while preserving the first-iteration constant baseline, closes the cell bit-exact and leaves the entire workspace at `cargo test --workspace` = 0 failed, 0 ignored.**

## Performance

- **Duration:** ~1 session (the root cause was produced by the `remaining-ignored-cells` debug session via a Python-wheel oracle; this plan implemented + gated it).
- **Completed:** 2026-06-08

## The fix

`crates/lgbm-boosting/src/gbdt.rs` — `train_one_iter` (mirrors `gbdt.cpp:406-447`):
- `IterSnapshot` gained `emitted: bool`.
- `should_continue` starts false; set true in BOTH real-split branches (bagging-subset and full-corpus) when the grown tree has `num_leaves > 1`.
- After the per-class loop, before DART: `if !should_continue && tree_count_before > 0 { self.trees.truncate(tree_count_before); return Ok(IterSnapshot{ …, emitted:false }); }` — pops the round's constants, does NOT advance `self.iter`. The first iteration (`tree_count_before == 0`) is never popped (keeps the constant baseline, advances, `emitted:true`).

`crates/lgbm/src/booster.rs` — the real driver loop (~1100): `if !snap.emitted { continue; }` before `ran_iters`/`iter_scores.push`/`iter_grad_hess.push`, so a popped round doesn't push a duplicate score; `best_iteration = num_iteration()` returns the emitted count.

`crates/oracle-harness/tests/boosting_parity.rs`: new `quantile_bagged_no_split_emission_contract` diagnostic (owns the exact 10-tree count guard) + un-`#[ignore]` `quantile_loop_matrix`.

## Deviations from Plan

None — plan executed as written. The Task-2 change is precisely the sanctioned architectural fix the plan exists to land (replicating `gbdt.cpp:406-447`); no Rule 1-4 deviation triggered, no tolerance weakened, no horizon capped, no quantile special-casing. `sample_strategy.rs` needed no change (confirmed the re-bag cadence is already C++-faithful), so it was not modified despite being a candidate.

## Verification

- **Failing-then-passing diagnostic:** `quantile_bagged_no_split_emission_contract` FAILED on the pre-fix model (12 trees `[1,2,3,3,1,3,3,3,1,3,2,3]`) and PASSES after (10 trees `[1,2,3,3,3,3,3,3,2,3]`; wheel tree4 ≡ Rust tree5 within MATRIX_RESIDUAL_TOL).
- **Cross-variant no-regression gate (independently re-run at the Task-4 blocking-human checkpoint):** full `LGBM_CAPTURE_PYTHON=… cargo test --workspace` → **0 failed, 0 ignored**. DART (`dart_parity_matrix`, `dart_drop_rng_replay`), GOSS (`goss_parity_matrix`, `goss_rng_replay`), RF (`rf_single_parity`, `rf_multi_parity`), bagging (`bagging_rng`, `*_bag1_*`), `subset_determinism_diagnostic`, first-iter baseline (`regression_l1_*`, `class_need_train_false_pushes_constant_tree`, lgbm-boosting 55/55), 07-13 Family-A + Family-B cells, kernel_parity 4/4, learner_parity 29/29, all `*_spine`/`*_gradients` — all GREEN.
- `boosting_parity` — **75 passed, 0 ignored** (was 74 + 1 ignored).
- `git status --porcelain LightGBM/` — never git-added.

## Task Commits

1. `2dfb4f9` — `test(07-14)`: add quantile_bagged_no_split_emission_contract diagnostic.
2. `824d30f` — `fix(07-14)`: C++-faithful no-split bagged pop/no-advance/re-bag in GBDT loop.
3. `b560d13` — `test(07-14)`: un-ignore quantile_loop_matrix (DEF-07-13-01 closed).
4. `<this commit>` — `docs(07-14)`: clear DEF-07-13-01 + SUMMARY + STATE/ROADMAP.

## Self-Check: PASSED

- `07-14-PLAN.md` + `07-14-SUMMARY.md` exist on disk.
- Fix commit `824d30f` + test commits `2dfb4f9` / `b560d13` present in history.
- `cargo test --workspace` GREEN — **0 failed, 0 ignored** (entire parity suite asserts real lib_lightgbm 4.6 parity); merge gate bit-exact; `LightGBM/` never git-added.
- DEF-07-13-01 cleared; DEF-07-02/03 + DEF-07-11 remain RESOLVED.
