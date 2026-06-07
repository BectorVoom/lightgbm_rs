---
phase: 06-gbdt-spine-core-objectives-metrics
reviewed: 2026-06-07T00:00:00Z
depth: standard
files_reviewed: 28
files_reviewed_list:
  - crates/lgbm-boosting/src/early_stopping.rs
  - crates/lgbm-boosting/src/error.rs
  - crates/lgbm-boosting/src/gbdt.rs
  - crates/lgbm-boosting/src/lib.rs
  - crates/lgbm-boosting/src/objective.rs
  - crates/lgbm-boosting/src/sample_strategy.rs
  - crates/lgbm-boosting/src/score_updater.rs
  - crates/lgbm-metric/src/binary.rs
  - crates/lgbm-metric/src/error.rs
  - crates/lgbm-metric/src/lib.rs
  - crates/lgbm-metric/src/multiclass.rs
  - crates/lgbm-metric/src/regression.rs
  - crates/lgbm-model/src/tree.rs
  - crates/lgbm-objective/src/binary.rs
  - crates/lgbm-objective/src/custom.rs
  - crates/lgbm-objective/src/error.rs
  - crates/lgbm-objective/src/lib.rs
  - crates/lgbm-objective/src/multiclass.rs
  - crates/lgbm-objective/src/percentile.rs
  - crates/lgbm-objective/src/regression.rs
  - crates/lgbm-treelearner/src/learner.rs
  - crates/lgbm/src/booster.rs
  - crates/lgbm/src/builder.rs
  - crates/lgbm/src/error.rs
  - crates/lgbm/src/lib.rs
  - crates/oracle-harness/tests/boosting_parity.rs
  - xtask/py/boosting_oracle_capture.py
  - xtask/src/main.rs
findings:
  critical: 2
  warning: 5
  info: 3
  total: 10
status: issues_found
---

# Phase 6: Code Review Report

**Reviewed:** 2026-06-07
**Depth:** standard
**Files Reviewed:** 28
**Status:** issues_found

## Summary

Phase 6 wires the proven Phase-5 serial tree learner into the GBDT boosting loop plus the five core objectives, seven metrics, bagging RNG, and early stopping. The bulk of the net-new numerical code (objective grad/hess, metric reductions, percentile, softmax gather, bagging draw) is a faithful, well-cited 1:1 port and reads correct against the C++ reference — the documented deviations (custom-objective f64 preds, multiclass 5-iter exp-libm horizon, RNG-instance reuse) all match what the summaries claim, and were NOT re-reported here.

Two correctness defects vs the C++ reference survive and are not exercised by the committed goldens (so the green test suite does not catch them):

1. **Constant-tree `leaf_count` is hardcoded to 0** instead of `num_data`, diverging from C++ `AsConstantTree(init, num_data_)` model-text output for the `class_need_train==false` / no-split path.
2. **Early-stopping decision is gated by `metric_freq`** — C++ evaluates the valid metric + stop check EVERY iteration when early stopping is on, regardless of `metric_freq`. With `metric_freq > 1` the Rust port skips stop checks on off-cadence rounds, changing `best_iteration` and the trailing-tree trim.

Both are latent because the matrix/goldens run at `metric_freq=1` and never byte-compare a serialized constant tree's `leaf_count`. The remaining findings are robustness/quality issues, with one notable test-integrity gap (residual matrix cells swallow their comparison result via `.ok()`).

## Critical Issues

### CR-01: Constant tree serializes `leaf_count=0` instead of `num_data` (model-text divergence vs C++)

**File:** `crates/lgbm-model/src/tree.rs:658-687` (and call sites `crates/lgbm-boosting/src/gbdt.rs:290, 362, 420`)
**Issue:** C++ `GBDT::TrainOneIter` pushes a degenerate class via `new_tree->AsConstantTree(init_scores[cur_tree_id], num_data_)` (`gbdt.cpp:430`) — the second arg sets `leaf_count_[0] = num_data` (`tree.h:232-240`). The Rust `Tree::as_constant(constant_value)` always sets `leaf_count: vec![0]` and takes no count parameter. C++ `Tree::ToString` always emits `leaf_count=` (no single-leaf write-side early return — `src/io/tree.cpp:363`), so a constant tree serializes `leaf_count=<num_data>` in C++ but `leaf_count=0` in Rust. Any model that contains a constant tree (absent multiclass class, single-class binary, or a first iteration with no positive-gain split) will NOT be byte-exact against the C++ model text — violating the bit-exact model-text contract. This is currently latent: the D-07 matrix replay compares `leaf_value` arrays only (boosting_parity.rs:1026-1041), not the full serialized text, and the absent-class path is only Rust-unit-tested (gbdt.rs:801-828), never against the real binary.
**Fix:** Give `as_constant` the count argument and thread it from the loop:
```rust
// tree.rs
pub fn as_constant(constant_value: f64, count: i32) -> Tree {
    Tree { /* ... */ leaf_count: vec![count], /* ... */ }
}
// gbdt.rs — match gbdt.cpp:430 (init iter) / :433 (later iters, count=num_data, val=0)
self.trees.push(Tree::as_constant(const_val, self.num_data));
```
Note C++ passes `num_data_` in BOTH the init-iter (`:430`) and the extend-with-zeros (`:433`) branches, so use `self.num_data` for all three call sites.

### CR-02: Early-stopping stop check is gated by `metric_freq`, diverging from C++ (every-iteration check)

**File:** `crates/lgbm/src/booster.rs:516-540`
**Issue:** The Rust loop computes valid metrics and calls `early.update(...)` only inside `if do_eval`, where `do_eval = (it + 1) % metric_freq == 0 || it + 1 == total_iters` (line 516). In C++ `OutputMetric` (`gbdt.cpp:551-608`), `need_output = (iter % metric_freq) == 0` gates only the *training* metric output/logging; the validation-metric eval AND the early-stop decision run under `if (need_output || early_stopping_round_ > 0)` (`gbdt.cpp:574`) — i.e. **every iteration when early stopping is enabled**, independent of `metric_freq`. With `metric_freq > 1` and early stopping on, the Rust port skips the stop check on off-cadence rounds, so `iter - best_iter >= round` triggers on a different round (or not at all within the horizon), changing `best_iteration` and the count of trailing trees popped. Latent only because the committed matrix cells and unit tests use the default `metric_freq = 1`.
**Fix:** Decouple the early-stop eval from the `metric_freq` output cadence — when early stopping is enabled, evaluate the valid set and run `early.update` every iteration; keep `metric_freq` gating the training-metric history push only:
```rust
let do_train_eval = (it + 1) % metric_freq == 0 || it + 1 == total_iters;
let do_valid_eval = valid_nd > 0 && (do_train_eval || es_enabled);
if provide_train && do_train_eval { /* push training metrics */ }
if do_valid_eval {
    let mut row = Vec::with_capacity(metrics.len());
    for m in &metrics { row.push(m.eval(&valid_score, &valid_labels)?); }
    // record valid history only on the output cadence to match C++ logging;
    // but always run the stop decision when es is enabled:
    if es_enabled && early.update(it, &EvalSnapshot { values: vec![row] }) { break; }
}
```
Mirror C++ `gbdt.cpp:574` exactly: valid eval + stop runs on `need_output || early_stopping_round_ > 0`.

## Warnings

### WR-01: Matrix residual cells discard their comparison via `.ok()` (test asserts nothing)

**File:** `crates/oracle-harness/tests/boosting_parity.rs:980-986, 1004-1011`
**Issue:** The `uniform_grad_residual` and multiclass-es residual branches call `compare_within(...).ok()`, discarding the `Result`. The surrounding comments and the SUMMARY claim these cells are "VALIDATED within ORACLE_TOL on overlapping trees," but `.ok()` means a mismatch of any magnitude — including a gross regression far beyond ORACLE_TOL — passes silently. These cells are effectively unasserted; only `cells_checked += 1` records that they ran. The numerical-fidelity claim for the residual cells is therefore not enforced.
**Fix:** Assert the result (or collect into a max-diff and assert the max is within a documented residual bound). At minimum:
```rust
compare_within(&rl_f32, &gl_f32, ORACLE_TOL)
    .unwrap_or_else(|m| panic!("{cell} tree {i} residual exceeds ORACLE_TOL: {m:?}"));
```
If overlapping trees genuinely cannot meet ORACLE_TOL, use an explicit, documented residual tolerance and assert against THAT — never `.ok()`.

### WR-02: D-07 matrix replay uses one valid set; C++ capture monitors `[training, valid_0]`

**File:** `crates/lgbm/src/booster.rs:290-319, 461-468` vs `xtask/py/boosting_oracle_capture.py:454-466`
**Issue:** The capture passes `valid_sets=[dtrain, dvalid]` with `valid_names=["training", "valid_0"]` and the `lgb.early_stopping` callback (default monitors ALL valid sets). C++ thus tracks `best_score_`/`best_iter_` for BOTH the training set and valid_0 and stops on whichever plateaus first. The Rust `train_with_valid` constructs `EarlyStopping::new(..., num_valid_sets = 1, ...)` (booster.rs:466) — only valid_0. The two agree only because the training metric typically keeps improving (so valid_0 drives the stop). This is a fragile equivalence: a corpus/objective where the training metric plateaus first would diverge in `best_iteration`, and the test (boosting_parity.rs:1047) asserts the trimmed tree count against the C++ `best_iteration`.
**Fix:** Either (a) make the Rust early-stop path accept multiple valid sets and have the matrix replay register the training set as a monitored valid set too (matching the capture), or (b) change the capture to monitor only `valid_0` (`callbacks=[lgb.early_stopping(..., first_metric_only=...)]` with `valid_sets=[dvalid]` only) so the oracle and the implementation track the same set. Document the choice in REFERENCE_MANIFEST.md.

### WR-03: regression_l1 RenewTreeOutput silently skipped on the bagging-subset path

**File:** `crates/lgbm-boosting/src/gbdt.rs:314-322`
**Issue:** In the `use_subset` branch, when `is_renew_tree_output()` is true the code enters an empty `if` block containing only a comment — the median-residual leaf renewal is NOT applied. So `regression_l1 + bagging` leaves carry the learner's Newton output, not the median residual the objective requires (`RegressionL1loss::IsRenewTreeOutput()==true`). The SUMMARY documents this as a known deferral, but in code it is a silent no-op inside a `true` branch rather than an explicit guard/error, so a future reader can mistake it for "handled." It produces numerically wrong leaf values for any l1-with-bagging run.
**Fix:** Make the gap explicit and loud rather than a silent empty block — e.g. thread the subset partition's `residual_getter` through `train_on_subset` and apply the renewal (the real fix), or, until then, return a typed `BoostingError` for `regression_l1 + bagging` so a caller cannot silently get wrong leaves:
```rust
if self.objective.is_renew_tree_output() {
    return Err(BoostingError::Unsupported(/* l1 + bagging renewal pending */));
}
```

### WR-04: `feature_row` closure recomputes the feature-vector width per row inside the OOB scoring loop

**File:** `crates/lgbm-boosting/src/gbdt.rs:332-344`
**Issue:** The `feature_row` closure recomputes `width = features.iter().map(...).max()...` on every invocation, and it is called once per scored row (in_bag + OOB) per tree. Beyond the O(rows×features) waste (out of scope), the `max().map(|m| (m+1)).unwrap_or(0)` produces a zero-width vector when `features` is empty, and `v[f.real_feature_index]` would then panic on a non-empty `features` with an out-of-range index — though in practice `features` is non-empty on this path. The bigger correctness risk is that the row vector is indexed by `real_feature_index`, which must be dense `0..width`; a sparse/non-contiguous `real_feature_index` set would leave gaps as `0.0` that `Tree::predict` then traverses, silently mis-routing OOB rows.
**Fix:** Hoist the width computation out of the closure and assert the feature index space is dense, or build the row vector by gathering only the features the tree actually splits on. At minimum compute `width` once before the loop and document the dense-index precondition.

### WR-05: `MulticlassOva` does not range-check labels; relies on `f32 as i32`/`as usize` saturation

**File:** `crates/lgbm-objective/src/multiclass.rs:226-239` and `crates/lgbm-metric/src/multiclass.rs:96-101`
**Issue:** `MulticlassOva::new` intentionally skips the label range check (mirroring C++ `MulticlassOVA`, which only maps `is_pos = (int)label == i`). That is faithful for grad/hess. But the `multi_logloss` metric indexes `rec` by `labels[i] as usize` (multiclass.rs:96). A negative label casts to `0usize` under Rust saturating `f32 as usize` and a too-large label is caught by `rec.get(kk)`, so there is no panic — but a negative or out-of-range OVA label silently scores against class 0 / the floor instead of being rejected, which can diverge from C++ where `(size_t)label` of a negative is a huge index (different behavior). This is a latent fidelity gap for malformed multiclassova labels.
**Fix:** Decide and document the contract: either range-check OVA labels at `MulticlassOva::new` (typed `LabelOutOfRange`, as softmax does) or explicitly mirror the exact C++ `(size_t)label` behavior in the metric. Do not leave it to Rust cast-saturation, which matches neither cleanly.

## Info

### IN-01: `train_inner_full` carries dead `train_eval_history` plumbing

**File:** `crates/lgbm/src/booster.rs:473-476, 519-525, 561`
**Issue:** `train_eval_history` is populated (line 523) but then explicitly discarded with `let _ = train_eval_history;` (line 561) — the training metrics are routed through `legacy_eval_history` instead. The vector and its per-round pushes are dead work and confuse the data flow.
**Fix:** Remove `train_eval_history` and push training metrics directly to the legacy/keyed history, or actually surface `training <metric>` keys in the public `eval_history` if that was the intent.

### IN-02: `BoostObjective::renew_leaf_output` returns `0.0` for non-renew variants instead of being unreachable

**File:** `crates/lgbm-boosting/src/objective.rs:129-139`
**Issue:** For Binary/Custom/Multiclass/MulticlassOva the method returns `0.0` "defensively." It is only ever called when `is_renew_tree_output()` is true (regression_l1), so the `0.0` arms are dead; a stray future caller would get a silently-wrong leaf value of 0 rather than a loud failure.
**Fix:** `unreachable!("renew_leaf_output called for a non-renew objective")` (or `debug_assert!`) makes the invariant explicit and turns a future misuse into a loud panic in tests rather than a silent 0.

### IN-03: `Booster.best_iteration` doc/field comments still describe 06-02 stub behavior

**File:** `crates/lgbm/src/booster.rs:196-203`
**Issue:** The doc comments on `best_iteration` / `eval_history` still say "06-02: the last trained iteration (no early stopping yet)" even though 06-05 wired early stopping and the field is now populated from `EarlyStopping::best_iteration()`. Stale comments mislead readers about the field's current semantics.
**Fix:** Update the field docs to describe the 06-05 behavior (best_iteration from the early-stop decision, or `num_iteration()` when early stopping is off).

---

_Reviewed: 2026-06-07_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
