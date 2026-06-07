---
phase: 07-parity-completing-variants
reviewed: 2026-06-07T10:09:55Z
depth: standard
files_reviewed: 31
files_reviewed_list:
  - crates/lgbm-boosting/src/gbdt.rs
  - crates/lgbm-boosting/src/sample_strategy.rs
  - crates/lgbm-boosting/src/objective.rs
  - crates/lgbm-compute/src/gain.rs
  - crates/lgbm-compute/src/kernels/split.rs
  - crates/lgbm-metric/src/dcg_calculator.rs
  - crates/lgbm-metric/src/rank.rs
  - crates/lgbm-metric/src/regression.rs
  - crates/lgbm-metric/src/xentropy.rs
  - crates/lgbm-metric/src/multiclass.rs
  - crates/lgbm-metric/src/binary.rs
  - crates/lgbm-model/src/tree.rs
  - crates/lgbm-model/src/predict.rs
  - crates/lgbm-model/src/ensemble.rs
  - crates/lgbm-model/src/model_text.rs
  - crates/lgbm-model/src/objective.rs
  - crates/lgbm-objective/src/regression.rs
  - crates/lgbm-objective/src/xentropy.rs
  - crates/lgbm-objective/src/rank.rs
  - crates/lgbm-treelearner/src/learner.rs
  - crates/lgbm-treelearner/src/feature_histogram_categorical.rs
  - crates/lgbm-treelearner/src/monotone_constraints.rs
  - crates/lgbm-treelearner/src/cost_effective_gradient_boosting.rs
  - crates/lgbm-treelearner/src/forced_splits.rs
  - crates/lgbm-treelearner/src/data_partition.rs
  - crates/lgbm-treelearner/src/leaf_splits.rs
  - crates/lgbm-core/src/config/set.rs
  - crates/lgbm-core/src/config/mod.rs
  - crates/lgbm/src/builder.rs
  - crates/lgbm/src/booster.rs
  - xtask/src/main.rs
findings:
  critical: 0
  warning: 4
  info: 4
  total: 8
status: issues_found
---

# Phase 7: Code Review Report

**Reviewed:** 2026-06-07T10:09:55Z
**Depth:** standard
**Files Reviewed:** 31
**Status:** issues_found

## Summary

Reviewed the Phase-7 "parity-completing variants" surface: DART/RF/GOSS boosting
variants, the extended objective/metric families (huber/fair/quantile/mape/
poisson/gamma/tweedie, ranking, cross-entropy, multiclass AUC/error), categorical
splits, monotone constraints, CEGB, forced-splits JSON parsing, TreeSHAP, leaf
refit, and the config/facade wiring.

This is a numerically-faithful 1:1 port of C++ LightGBM 4.6. I verified the
load-bearing math against the in-tree C++ reference under `LightGBM/src/` directly
(rank_objective.hpp, xentropy_objective.hpp, dart.hpp, dcg_calculator.cpp,
binary_metric.hpp, feature_histogram.hpp). The core gain math, RNG draw order
(bagging/GOSS/rank_xendcg verbatim quickselect + per-block LCG), histogram
subtraction trick, smaller-child selection, kEpsilon provenance, leaf-renew
ordering, lambdarank pairwise accumulation, cross-entropy stable form,
average_precision/AUC accumulation, softmax max-subtraction, and TreeSHAP
recursion all match the reference faithfully. Input-boundary handling
(model-text parse, forced_splits JSON, predict shape checks, config CHECK_*) is
defensive and typed — no panicking paths found on hostile input.

No BLOCKER-tier correctness or security defects were found. The findings below are
narrow algorithmic-edge divergences (config corners outside the validated matrix),
a UTF-8 handling bug in the forced-splits string parser, and a few maintainability
items. Per the review charter, the accepted deferrals DEF-07-02 and DEF-07-11
(objective/constraint knife-edges) are NOT reported.

## Warnings

### WR-01: DART `max_drop < 0` caps drops at 1 instead of "no cap" (C++ divergence)

**File:** `crates/lgbm-boosting/src/gbdt.rs:1280` and `:1293`
**Issue:** The drop cap is written `dart.drop_index.len() >= cfg.max_drop.max(0) as usize`.
C++ writes `drop_index_.size() >= static_cast<size_t>(config_->max_drop)`
(dart.hpp:109-110, 124-125). For a NEGATIVE `max_drop` (e.g. `-1`), C++
`static_cast<size_t>(-1) == SIZE_MAX`, so the break NEVER fires → unlimited drops.
The Rust `.max(0)` collapses any negative `max_drop` to `0`, so `len() >= 0` is
always true → the loop breaks after the FIRST pushed drop index, capping at 1
drop. `max_drop == 0` matches C++ (both cap at 1), but `max_drop < 0` diverges.
`config/set.rs:191` applies NO `>= 0` range check on `max_drop`, so a user can
reach this. The `DartConfig.max_drop` doc comment ("`<= 0` ⇒ no cap") is also
inconsistent with the actual `== 0` behavior.
**Fix:** Mirror the C++ unsigned-wrap semantics explicitly:
```rust
// C++ static_cast<size_t>(config_->max_drop): a negative max_drop wraps to a
// huge bound (no effective cap); 0 caps at 1; positive caps at that value.
let cap = if cfg.max_drop < 0 { usize::MAX } else { cfg.max_drop as usize };
// ... inside the loop:
if dart.drop_index.len() >= cap { break; }
```
Apply to both the `!uniform_drop` (`:1280`) and `uniform_drop` (`:1293`) branches.

### WR-02: Forced-splits JSON string parser mangles non-ASCII bytes (`c as char`)

**File:** `crates/lgbm-treelearner/src/forced_splits.rs:223-224`
**Issue:** `parse_string` builds the result with `s.push(c as char)` where `c: u8`.
For any byte `>= 0x80` (the continuation/lead bytes of a multi-byte UTF-8
sequence) this reinterprets each raw byte as a Unicode scalar value, corrupting
non-ASCII content (e.g. a UTF-8 key or value is silently mojibake'd rather than
decoded). `forced_splits_filename` is explicitly called out as the untrusted
input boundary in this module's threat header (T-07-11-01); a hostile or merely
non-ASCII document is silently mis-parsed instead of rejected or correctly
decoded. It does not crash (the schema only reads numeric `feature`/`threshold`),
but it is a latent correctness/robustness defect at a security boundary and the
"hand-rolled JSON reader" comment implies RFC-ish fidelity it does not have.
**Fix:** Accumulate raw bytes into a `Vec<u8>` and decode once with
`String::from_utf8(...)`, returning `ForcedSplitError::Syntax` on invalid UTF-8;
or restrict the parser to ASCII and reject bytes `>= 0x80` explicitly. Example:
```rust
Some(c) => { buf.push(c); self.pos += 1; }
// on close quote:
String::from_utf8(buf).map_err(|_| ForcedSplitError::Syntax("invalid utf-8 in string".into()))
```

### WR-03: DART drop loop uses `self.iter` instead of `num_init_iteration_ + iter` (continue-training divergence)

**File:** `crates/lgbm-boosting/src/gbdt.rs:1276`, `:1289`, `:1305-1315`
**Issue:** `dropping_trees` iterates `for i in 0..iter` (`iter = self.iter`) and
indexes `self.trees[(i * k + cur_tree_id)]`. C++ `DroppingTrees` iterates
`for (int i = 0; i < iter_; ++i)` and pushes `num_init_iteration_ + i`
(dart.hpp:107-108, 116-117), then indexes `models_[(num_init_iteration_ + i) * ntpi + cur_tree_id]`.
The Rust port hard-codes the `num_init_iteration_ = 0` assumption (acknowledged in
the inline comment). For a FRESH train this is correct, but if DART is combined
with `with_loaded_model` (continue-training, ADV-06), `self.iter` after seeding
equals the loaded iteration count and the drop indices/tree lookups would no
longer match the C++ `(num_init_iteration_ + i)` indexing — DART would consider
dropping the pre-loaded trees and the `tree_weight_`/`drop_index_` bookkeeping
(which only covers freshly-grown iterations) would index incorrectly.
**Fix:** If DART + continue-training is out of scope, add a guard in
`with_loaded_model`/`with_dart` that rejects the combination with a typed error
(so it can never silently mis-index). If in scope, thread `num_init_iteration`
through `dropping_trees`/`normalize` and offset the tree index + `tree_weight_`
access by it, mirroring `(num_init_iteration_ + i)`.

### WR-04: `model_text::load` rejects a zero-feature model (`"".split(' ').count() == 1`)

**File:** `crates/lgbm-model/src/model_text.rs:112-123`
**Issue:** The feature-count validation does
`feature_names.split(' ').count() != expected_cols`. For a degenerate model with
`max_feature_idx = -1` (zero features), `expected_cols = (−1 + 1) = 0`, but an
empty `feature_names=` value yields `"".split(' ').count() == 1`, so a valid
zero-feature model would be rejected as malformed. Likewise `feature_infos`.
Real LightGBM models always have `>= 1` feature so this is unlikely to surface,
but it is an off-by-one in the boundary check that would mis-classify a valid
edge-case model as malformed.
**Fix:** Special-case the empty string before counting:
```rust
let fn_count = if feature_names.is_empty() { 0 } else { feature_names.split(' ').count() };
```
(and the same for `feature_infos`).

## Info

### IN-01: `average_precision` declares `accum_prec` without initializer (dead C++ init dropped)

**File:** `crates/lgbm-metric/src/binary.rs:132`
**Issue:** `let mut accum_prec;` is declared uninitialized; C++ initializes it to
`1.0f` (binary_metric.hpp:319). The C++ `1.0f` init is dead (it is overwritten on
every threshold change before being read), so the Rust behavior is equivalent —
but a future edit that reads `accum_prec` before the first threshold transition
would fail to compile (Rust) where C++ would silently read `1.0`. Cosmetic
fidelity gap; no behavior change.
**Fix:** Initialize `let mut accum_prec = 1.0f64;` to mirror the C++ source 1:1.

### IN-02: `feature_importance_split_count` (unguarded) is a near-duplicate of the guarded variant

**File:** `crates/lgbm-model/src/ensemble.rs:157-168` vs `:200-212`
**Issue:** `feature_importance_split_count` (no `split_gain > 0` guard) and
`feature_importance_split_count_guarded` differ only by the gain filter. The
unguarded one is retained "for callers that want the raw structural count" but
the model-text emit + parity use the guarded one. Two methods with subtly
different semantics invite a wrong-method bug at a call site.
**Fix:** If no live caller needs the unguarded count, remove it; otherwise rename
to make the distinction unmissable (e.g. `..._raw` / `..._cpp_faithful`) and
document which one matches C++ `FeatureImportance`.

### IN-03: `renew_tree_output` recomputes every leaf incl. empty leaves (works, but relies on `percentile_fun([])==0.0`)

**File:** `crates/lgbm-treelearner/src/learner.rs:2486-2489`
**Issue:** The loop calls `renew(leaf, rows)` for every leaf unconditionally. C++
`RenewTreeOutput` (serial_tree_learner.cpp:935-942) special-cases `cnt_leaf_data == 0`
by setting `SetLeafOutput(i, 0.0)` (single-machine). The Rust closures
(gbdt.rs:699-717, :787-802) feed an empty `residuals`/`labels` slice into
`obj.renew_leaf_output`, which routes to `percentile_fun(&[], 0.5) == 0.0` —
coincidentally matching the C++ `0.0`. The behavior is correct ONLY because the
empty-input percentile returns `0.0`; it is not guarded explicitly. Leaf-wise
growth with `min_data_in_leaf >= 1` never produces empty leaves on the in-scope
corpora, so this is latent. The MAPE weighted path (`weighted_percentile_fun(&[],&[],0.5)`)
should be confirmed to also return `0.0`/not panic on empty input.
**Fix:** Add an explicit `if rows.is_empty() { tree.leaf_value[leaf]=0.0; continue; }`
guard in `renew_tree_output` to make the C++ correspondence explicit and robust
(the RF path `rf_renew_full` already does this; the GBDT-spine closures do not).

### IN-04: `find_in_bitset` / `categorical_decision` slice indexing assumes valid `cat_boundaries` on a parsed model

**File:** `crates/lgbm-model/src/tree.rs:201-204`
**Issue:** `categorical_decision` indexes `self.cat_boundaries[cat_idx]` and
`[cat_idx + 1]` then slices `self.cat_threshold[lo..hi]` with `cat_idx` derived
from `self.threshold[node] as i32`. `Tree::parse` validates `cat_boundaries`
length (`num_cat + 1`) and `cat_threshold` length (`== cat_boundaries.back()`),
but does NOT validate that each numeric `threshold` value used as a categorical
`cat_idx` is `< num_cat`, nor that `cat_boundaries` is monotone non-decreasing
(so `lo <= hi`). A crafted model with an out-of-range categorical threshold or a
non-monotone `cat_boundaries` could panic on slice indexing at predict time
(`lo > hi` or `cat_idx + 1` OOB). This is a defense-in-depth gap at the model
parse boundary (the module's own header promises "never a panic" on malformed
files); numeric splits are fully bounds-checked but the categorical-threshold ↔
`cat_boundaries` linkage is not.
**Fix:** In `Tree::parse`, for each categorical-decision node validate
`(threshold as usize) + 1 < cat_boundaries.len()` and that `cat_boundaries` is
monotone non-decreasing, returning `ModelError::MalformedModel` otherwise — so
predict-time slicing can never panic on a hostile model.

---

_Reviewed: 2026-06-07T10:09:55Z_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
