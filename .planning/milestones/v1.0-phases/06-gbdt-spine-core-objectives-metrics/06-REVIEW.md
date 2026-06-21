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
  critical: 0
  warning: 6
  info: 5
  total: 11
status: issues_found
---

# Phase 6: Code Review Report

**Reviewed:** 2026-06-07
**Depth:** standard
**Files Reviewed:** 28
**Status:** issues_found

## Summary

Reviewed the Phase-6 GBDT spine: the boosting loop (`gbdt.rs`), score accumulator,
bagging RNG, early stopping, six objectives, seven metrics, the public facade
(`booster.rs` / `builder.rs`), the `Tree` model, and the capture/replay harness.

The faithful-mirror discipline is strong and the C++ citations are precise. I
traced the BoostFromAverage → GetGradients → Bagging → per-class tree →
RenewTreeOutput → Shrinkage → UpdateScore → AddBias ordering against
`gbdt.cpp:383-452` and it matches; the early-stop `iter - best_iter >= round`
arithmetic and the `(total_iters - best_iteration) * num_class` trailing-tree pop
reconcile with the C++ `EvalAndCheckEarlyStopping` fixed pop without an off-by-one.
The percentile interpolation and `upper_bound`/`partition_point` mapping are
correct against `regression_objective.hpp:50-88`.

No blockers found. The findings below are correctness-edge and maintainability
concerns. The most material is **WR-01**: a real divergence from the C++
`OutputMetric` cadence (`booster.rs` adds an "always eval on the last iter" clause
the reference does not have), which silently changes the recorded eval-history
length whenever `num_iterations` is not a multiple of `metric_freq` — a case the
committed tests do not exercise.

## Warnings

### WR-01: Eval-history cadence diverges from C++ when `num_iterations % metric_freq != 0`

**File:** `crates/lgbm/src/booster.rs:527`
**Issue:** The recorded-history gate is
`let do_eval = (it + 1) % metric_freq == 0 || it + 1 == total_iters;`. The C++
reference (`gbdt.cpp:552`, `OutputMetric`) gates output purely on
`need_output = (iter % config_->metric_freq) == 0` with **no** last-iteration
special case. So for, e.g., `num_iterations = 10`, `metric_freq = 3`, C++ records
on iters 3, 6, 9 (3 values) while this port records on 3, 6, 9, **and 10** (4
values). The committed `metric_freq_thins_eval_history` test only uses 9 iters (a
multiple of 3), so the divergence is untested. This changes
`booster.eval_history` length and the per-round value vector vs the C++
`record_evaluation` golden for any non-multiple horizon.
**Fix:** Drop the last-iter clause to mirror C++ exactly:
```rust
let do_eval = (it + 1) % metric_freq == 0;
```
If a final-iteration value is desired for ergonomics, gate it behind an explicit,
documented facade choice that is NOT applied to the C++-parity eval-history keys.

### WR-02: `MulticlassSoftmax::new(1, ..)` produces `factor = +inf` (NaN hessians) instead of a typed reject

**File:** `crates/lgbm-objective/src/multiclass.rs:84`
**Issue:** `new` validates only `num_class >= 1`, then computes
`factor = num_class / (num_class - 1.0)`. For `num_class == 1` this is `1.0 / 0.0
= +inf`, so every hessian `factor * p * (1 - p)` becomes `inf`/`NaN` and silently
corrupts the whole tree-growth path rather than failing at the boundary (Security
V5 / typed-reject discipline this crate otherwise follows). C++ upstream config
validation forbids `num_class < 2` for multiclass, but this constructor is a
public boundary and does not.
**Fix:** Reject `num_class < 2` for softmax explicitly:
```rust
if num_class < 2 {
    return Err(ObjectiveError::Unsupported {
        name: format!("multiclass num_class {num_class} must be >= 2 (redundant-form factor divides by num_class-1)"),
    });
}
```

### WR-03: `MulticlassOva` does not validate the per-row integer label range (silent class drop)

**File:** `crates/lgbm-objective/src/multiclass.rs:226-239`
**Issue:** `MulticlassOva::new` truncates `label as i32` with no range check (the
doc-comment frames this as "mirror C++"). A label `>= num_class` or `< 0` then
maps to no positive class in any `class_labels(i)` (`class_id` never equals it),
so that row is treated as negative for **every** one-vs-all binary — it is
silently dropped from all positive sets with no error. Unlike `MulticlassSoftmax`
(which raises `LabelOutOfRange`), this can hide a caller mistake as a quietly
mis-trained model. The C++ `MulticlassOVA` indeed skips the range fatal, but the
project's stated stance (typed-reject over wrong-but-similar) argues for at least
a guard against negative labels, which are never valid.
**Fix:** Reject negative labels (always invalid) and consider warning/erroring on
`label >= num_class`:
```rust
for &l in labels {
    let li = l as i32;
    if li < 0 {
        return Err(ObjectiveError::LabelOutOfRange { label: li, num_class });
    }
}
```

### WR-04: `MultiLogloss::eval` clamps an out-of-range label to the floor, masking a caller error

**File:** `crates/lgbm-metric/src/multiclass.rs:96-101`
**Issue:** `let kk = labels[i] as usize;` then `rec.get(kk).copied().unwrap_or(0.0)`
clamps a label outside `[0, num_class)` to probability `0.0` → the `-log(eps)`
floor, silently inflating the loss instead of surfacing the malformed input as a
typed `MetricError`. The comment calls this "defensive (Security V5)", but
swallowing a programmer error as a plausible-looking metric value is exactly the
soft-failure the adversarial stance warns against — it cannot be distinguished
from a genuinely bad prediction. (Also note `labels[i] as usize` on a negative
f32 wraps to a huge `usize`, which `.get()` then misses — correct against panic,
but the value is meaningless.)
**Fix:** Return `MetricError` (or a new `LabelOutOfRange`-style variant) when
`kk >= k_classes`, rather than clamping:
```rust
if kk >= k_classes {
    return Err(MetricError::LengthMismatch { expected: k_classes, actual: kk });
}
```

### WR-05: Bagging-subset path scores in-bag rows via per-row `predict`, not the bit-exact partition scatter

**File:** `crates/lgbm-boosting/src/gbdt.rs:407-418`
**Issue:** On the `use_subset` branch BOTH in-bag and OOB rows are scored with
`add_tree_predict_path` (a per-row `Tree::predict` over the bin-index-as-real-value
feature vector). The C++ path scores **in-bag** rows via the data-partition leaf
scatter (`UpdateScore` → `AddPredictionToScore`) and only **OOB** rows via
predict (`gbdt.cpp:491-509`). The doc comment asserts these are bit-identical "on
the identity-binned corpus" because `bin index == raw value` and thresholds are
the bin upper bounds — which is true for the current Phase-6 corpora — but it is a
load-bearing assumption that will silently diverge the moment a non-identity
binning is introduced (real `bin_upper_bound` midpoints vs raw values), and the
predict path can route a row to a different leaf than the partition scatter when a
feature value sits on a threshold boundary. Correct for Phase-6 scope, but it is a
fidelity boundary, not a permanent equivalence.
**Fix:** When the in-bag partition is available (`subset_partition`), score in-bag
rows through the train-path scatter (`add_tree_train_path`) as C++ does, reserving
the predict path for OOB rows only. At minimum add a debug-assert that the
predict-path leaf equals the partition leaf for in-bag rows so a future
non-identity corpus fails loudly.

### WR-06: `as i32` label truncation can wrap large/NaN f32 labels into a valid-looking class index

**File:** `crates/lgbm-objective/src/multiclass.rs:90` (and `:233`)
**Issue:** Class labels are derived via `l as i32`. In Rust, `f32 as i32`
saturates (NaN → 0, +large → i32::MAX, -large → i32::MIN). For softmax the
subsequent range check catches `i32::MAX`/`i32::MIN`, but `NaN as i32 == 0` slips
through as a spurious class-0 label with no error. For OVA there is no range check
at all (see WR-03). This is an unlikely input on the deterministic anchor but is a
genuine silent mis-classification of NaN labels.
**Fix:** Reject non-finite labels before truncation:
```rust
if !l.is_finite() {
    return Err(ObjectiveError::LabelOutOfRange { label: i32::MIN, num_class });
}
let li = l as i32;
```

## Info

### IN-01: `feature_row` closure recomputes the feature width on every row

**File:** `crates/lgbm-boosting/src/gbdt.rs:394-406`
**Issue:** The `feature_row` closure recomputes `width` (a `max` scan over all
feature columns) on every invocation, i.e. once per scored row per tree. This is a
maintainability/readability smell (the width is loop-invariant). Performance is
out of v1 scope, but hoisting it also makes the intent clearer.
**Fix:** Compute `width` once before the closure and capture it.

### IN-02: `train`/`train_with_valid` duplicate the objective-dispatch `match` block verbatim

**File:** `crates/lgbm/src/booster.rs:252-275` and `:296-316`
**Issue:** The five-arm objective-resolution match (binary / multiclass /
multiclassova / regression) is copy-pasted between `train` and `train_with_valid`.
Divergence risk if one is edited and the other is not (e.g. a future objective
added to one path only).
**Fix:** Extract a `resolve_boost_objective(config, corpus) -> (BoostObjective, Vec<f32>)`
helper and call it from both entry points.

### IN-03: `weighted_percentile_fun` is dead code in Phase 6

**File:** `crates/lgbm-objective/src/percentile.rs:90-125`
**Issue:** The weighted percentile path is only reachable via weighted
`regression_l1`, and `regression_l1 + bagging` is typed-rejected while the
unweighted full-corpus path uses `percentile_fun`. The function is unexercised in
Phase 6 (only its uniform-weight unit test runs). It is faithful and documented as
kept-for-completeness; flagging so it is tracked as intentionally-fallow rather
than silently rotting.
**Fix:** No change required for scope; ensure a real weighted-L1 parity test lands
when the weighted path is activated.

### IN-04: `BoostObjective::renew_leaf_output` returns `0.0` for non-renew variants

**File:** `crates/lgbm-boosting/src/objective.rs:129-139`
**Issue:** For Binary/Custom/Multiclass/MulticlassOva the renewal returns a silent
`0.0`. It is guarded by `is_renew_tree_output()` at every call site, so it is
unreachable, but a future caller that forgets the guard would get a silent zero
leaf rather than a loud failure.
**Fix:** Consider `unreachable!()` (or a debug_assert) in the non-renew arms to
convert a future mis-wire into a loud panic in debug builds rather than a silent
wrong leaf.

### IN-05: `has_init_score()` is hard-wired to `false` with no plumbing seam

**File:** `crates/lgbm-boosting/src/gbdt.rs:164-169`
**Issue:** `has_init_score` always returns `false`; init-score `Dataset` metadata
is never threaded from the facade (`ScoreUpdater::new` accepts it, but `train_*`
always pass `None`). This is documented as a later-wave seam, but means a corpus
carrying `init_score` would silently get a re-run BoostFromAverage instead of
honoring the supplied init (C++ `gbdt.cpp:422` gates on
`!train_score_updater_->has_init_score()`). No defect in current scope (no init
metadata path exists), but the gap is a latent correctness divergence once
init_score is plumbed.
**Fix:** When init_score plumbing lands, wire the flag through `with_objective` so
`boost_from_average` is correctly suppressed; until then keep the doc note.

---

_Reviewed: 2026-06-07_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
