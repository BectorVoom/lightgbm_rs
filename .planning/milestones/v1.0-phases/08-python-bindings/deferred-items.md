# Phase 08 — Deferred / Out-of-Scope Items

## DEF-08-OOS-01 — pre-existing GOSS parity failure (NOT introduced by 08-01)

**Discovered during:** Plan 08-01 Task 3 (`cargo test --workspace` gate).

**Symptom:** `oracle-harness::boosting_parity::goss_parity_matrix` FAILS:
```
goss_t200_o50_es0_bfa0 tree 11 leaf_value not bit-exact vs real GOSS golden:
  index 1, rust: 1.032918262481689 (0x3FF086D54CCCCCCB)
  cpp:  1.0179183006286616 (0x3FF04964B3333331)
```

**Status: PRE-EXISTING — out of scope for 08-01.**
- Verified by checking out `crates/lgbm/src/{booster,error,lib}.rs` at commit
  `c13d380` (the commit immediately BEFORE any 08-01 work) and re-running: the
  failure reproduces IDENTICALLY (same tree, same bits). It was committed by a
  prior session's changes to `crates/lgbm-treelearner/src/learner.rs` +
  `crates/oracle-harness/tests/boosting_parity.rs` (the `053dcea..c13d380` range),
  NOT by plan 08-01.
- Plan 08-01 touched ONLY `crates/lgbm/src/booster.rs`, `error.rs`, `lib.rs`, and
  added `crates/oracle-harness/tests/raw_bin_train_parity.rs`. None of these touch
  the GOSS sample strategy, the tree learner, or the boosting loop.
- The divergence is a deep learner-level f64 split-gain knife-edge at tree 11
  (the same class as DEF-07-02 / the fair tiny-hessian and quantile-bagged
  divergences): g/h into the tree is correct; the split flips on a borderline
  f64 gain. Needs a source-built `lib_lightgbm` 4.6 FP execution trace to settle
  (an 07-01-style learner-fix), NOT a Phase-8 (Python-bindings) concern.

**Action:** Left untouched (SCOPE BOUNDARY — only auto-fix issues DIRECTLY caused
by the current task). Should be folded into the existing Phase-7 DEF-07-02
learner-level split-gain FP-trace fix plan, or a dedicated GOSS-parity fix plan,
NOT plan 08-01.
