---
phase: 03-tree-model-model-text-i-o-predict-parity
reviewed: 2026-06-05T00:00:00Z
depth: standard
files_reviewed: 15
files_reviewed_list:
  - crates/lgbm-model/src/format.rs
  - crates/lgbm-model/src/tree.rs
  - crates/lgbm-model/src/ensemble.rs
  - crates/lgbm-model/src/model_text.rs
  - crates/lgbm-model/src/objective.rs
  - crates/lgbm-model/src/predict.rs
  - crates/lgbm-model/src/error.rs
  - crates/lgbm-model/src/lib.rs
  - xtask/src/main.rs
  - crates/lgbm-model/tests/golden/mod.rs
  - crates/lgbm-model/tests/model_text_roundtrip.rs
  - crates/lgbm-model/tests/predict_raw_parity.rs
  - crates/lgbm-model/tests/predict_transform.rs
  - crates/lgbm-model/tests/predict_leaf_parity.rs
  - crates/lgbm-model/tests/predict_subrange.rs
findings:
  critical: 3
  warning: 6
  info: 4
  total: 13
status: issues_found
---

# Phase 3: Code Review Report

**Reviewed:** 2026-06-05
**Depth:** standard
**Files Reviewed:** 15
**Status:** issues_found

## Summary

Phase 3 delivers the `lgbm-model` crate: a faithful parallel-array `Tree`, GBDT
ensemble, model-text load/save, the four core `ConvertOutput` transforms, and a
dense/CSR/CSC predict driver. The transcription quality is generally high — the
decision-type bit decode, NaN/zero routing, softmax max-subtraction, f64
accumulation, sub-range clamping, and the parser's strict length validation all
match the C++ reference and the RESEARCH pitfalls are largely heeded.

However, adversarial cross-checking against the pinned C++ source surfaced
**three correctness divergences that break the phase's non-negotiable contracts**
(byte-exact serialization and C++ predict parity), plus several robustness gaps:

1. **`feature_importance_split_count` ignores the C++ `split_gain > 0` guard** —
   over-counts on any zero-gain split, breaking the DAT-09 byte-exact round-trip.
2. **Leaf-index prediction ignores the sub-range** (`start_iteration_for_pred_`),
   diverging from C++ `PredictLeafIndex` whenever a non-default range is set.
3. **`average_output` (RF mean) is never applied** in raw/transformed predict,
   silently mis-predicting any `average_output` model.

The committed goldens were not present in the working tree at review time (the
parity tests SKIP without fixtures), so these divergences are **not caught by the
current suite** — they were found by direct source comparison, not by a red test.
That is precisely the failure mode the adversarial review exists to catch.

## Critical Issues

### CR-01: `feature_importance_split_count` omits the C++ `split_gain > 0` guard — breaks byte-exact round-trip

**File:** `crates/lgbm-model/src/ensemble.rs:108-119`
**Issue:** C++ `GBDT::FeatureImportance` (split-count, `gbdt_model_text.cpp:636-643`)
counts a split **only when `models_[iter]->split_gain(split_idx) > 0`**:

```cpp
for (int split_idx = 0; split_idx < models_[iter]->num_leaves() - 1; ++split_idx) {
  if (models_[iter]->split_gain(split_idx) > 0) {
    feature_importances[models_[iter]->split_feature(split_idx)] += 1.0;
  }
}
```

The Rust port counts **every** split unconditionally:

```rust
for tree in &self.trees {
    for &sf in &tree.split_feature {
        if sf >= 0 && (sf as usize) < n {
            counts[sf as usize] += 1;
        }
    }
}
```

Any tree containing a split with `split_gain <= 0` (legal in real models — e.g.
forced splits, monotone-constrained splits, or splits emitted with gain exactly
0) will be over-counted. Since `model_text::save` (`model_text.rs:265-280`) feeds
this count directly into the `feature_importances:` block, the count value and/or
the descending-sort order will differ from the committed C++ `.txt`, breaking the
DAT-09 byte-exact round-trip — the linchpin contract of the phase. This will not
surface until a fixture with a zero-gain split is committed; the
`model_text_roundtrip` test currently SKIPs without fixtures.
**Fix:** Gate the count on `split_gain`, mirroring C++:
```rust
pub fn feature_importance_split_count(&self) -> Vec<u64> {
    let n = (self.max_feature_idx + 1).max(0) as usize;
    let mut counts = vec![0u64; n];
    for tree in &self.trees {
        for (i, &sf) in tree.split_feature.iter().enumerate() {
            if tree.split_gain[i] > 0.0 && sf >= 0 && (sf as usize) < n {
                counts[sf as usize] += 1;
            }
        }
    }
    counts
}
```
(Note `split_gain` is parsed via `parse_f32_list` and validated to length
`num_leaves-1`, so `tree.split_gain[i]` is in-bounds for every `i` in
`split_feature`.)

### CR-02: Leaf-index prediction ignores the sub-range (`start_iteration` / `num_iteration`)

**File:** `crates/lgbm-model/src/predict.rs:350-360`, `449-515`, `517-524`
**Issue:** C++ `GBDT::PredictLeafIndex` (`gbdt_prediction.cpp:79-86`) honors the
predict sub-range:
```cpp
int start_tree = start_iteration_for_pred_ * num_tree_per_iteration_;
int num_trees  = num_iteration_for_pred_  * num_tree_per_iteration_;
const auto* models_ptr = models_.data() + start_tree;
for (int i = 0; i < num_trees; ++i) { output[i] = models_ptr[i]->PredictLeafIndex(features); }
```
The Rust `predict_row_leaf` hard-codes the full range:
```rust
let num_iter = model.num_iteration().max(0) as usize;
for i in 0..num_iter {
    for k in 0..ntpi {
        let idx = i * ntpi + k;
        let leaf = model.trees[idx].predict_leaf_index(row);
        ...
```
and `leaf_width` (`predict.rs:520-524`) likewise uses the full `num_iteration()`.
The public leaf-index entry points expose no `start_iteration`/`num_iteration`
parameters at all, unlike the raw driver (`predict_raw_mat_range`). For PRD-03 +
PRD-06 combined (leaf index over a sub-range), the Rust output length and content
diverge from C++. The `predict_leaf_parity` test only exercises the full range,
so the gap is untested.
**Fix:** Thread `start_iteration`/`num_iteration` into the leaf-index entry points
and resolve them via `init_predict`, exactly as the raw path does:
```rust
fn predict_row_leaf(model: &GbdtModel, row: &[f64], start: i32, num: i32, out: &mut Vec<u32>) {
    let ntpi = model.num_tree_per_iteration.max(0) as usize;
    let (s, n) = model.init_predict(start, num);
    for i in s..s + n {
        for k in 0..ntpi {
            out.push(model.trees[i as usize * ntpi + k].predict_leaf_index(row) as u32);
        }
    }
}
```
and size `leaf_width` from `init_predict(start, num).1` rather than
`num_iteration()`. Add `_range` variants mirroring the raw driver.

### CR-03: `average_output` (RF mean) is never applied in raw / transformed predict

**File:** `crates/lgbm-model/src/ensemble.rs:90-102`, `crates/lgbm-model/src/predict.rs:336-342`
**Issue:** C++ `GBDT::Predict` (`gbdt_prediction.cpp:55-66`) divides the raw
accumulator by `num_iteration_for_pred_` when `average_output_` is set, BEFORE
`ConvertOutput`:
```cpp
PredictRaw(features, output, early_stop);
if (average_output_) {
  for (int k = 0; k < num_tree_per_iteration_; ++k) { output[k] /= num_iteration_for_pred_; }
}
if (objective_function_ != nullptr) { objective_function_->ConvertOutput(output, output); }
```
`GbdtModel` parses and stores `average_output` (`ensemble.rs:46-47`,
`model_text.rs:81-82`), but neither `predict_raw` nor `predict_row_transformed`
ever divides by the iteration count. Any model with `average_output` (the RF /
`boosting=rf` envelope, which emits the bare `average_output` line) will produce
raw and transformed scores that are a factor of `num_iteration_for_pred_` too
large. The field is loaded, round-tripped, and then silently ignored at predict
time — a latent data-correctness bug.
**Fix:** Apply the mean in the raw accumulator path after the loop, mirroring C++.
Note the divisor is `num_iteration_for_pred_` (the resolved `num`, not the total),
and division happens before `ConvertOutput`:
```rust
// in predict_raw, after the accumulation loop:
if self.average_output && num > 0 {
    for o in output.iter_mut() { *o /= num as f64; }
}
```
If RF/`average_output` is genuinely out of Phase-3 scope, that must be enforced at
load time (reject `average_output` with a typed `ModelError`) rather than
accepted-and-ignored, so a caller cannot get wrong numbers.

## Warnings

### WR-01: `feature_names` / `feature_infos` count check splits on a single space and breaks on empty / multi-space metadata

**File:** `crates/lgbm-model/src/model_text.rs:112-123`
**Issue:** The count validation uses `feature_names.split(' ').count()`. C++ joins
names with a single space, so on well-formed input this matches — but `split(' ')`
on an empty string returns 1 (not 0), and any double space yields a spurious empty
token. A model with `max_feature_idx = -1` (zero features, `expected_cols = 0`) or
with feature names that legitimately contain runs the validation incorrectly
rejects a model the C++ loader accepts. More importantly the same split is reused
at `model_text.rs:266` (`feature_names.split(' ').collect()`) to index the
importances block by feature index, so a miscount here can shift importance names.
**Fix:** Match the C++ tokenization. C++ uses `Common::Split(..., ' ')` and
compares against `max_feature_idx + 1`; replicate its empty-handling (or use
`split_whitespace()` consistently in BOTH the count check and the importances
name lookup so they cannot disagree), and special-case the zero-feature model.

### WR-02: `find_in_bitset` uses signed `pos / 32` / `pos % 32` — wrong block for `pos < 0` (C++ guards earlier, Rust relies on it)

**File:** `crates/lgbm-model/src/tree.rs:122-129`
**Issue:** `find_in_bitset` computes `let i1 = (pos / 32) as usize;`. For a
negative `pos`, Rust integer division truncates toward zero, so `-1 / 32 == 0`
and `(0) as usize == 0` — a valid in-bounds index — and `-1 % 32 == -1`, making
`bits[0] >> -1` **panic in debug / wrap in release** (shift by a negative/huge
amount). The only reason this is safe today is that `categorical_decision`
(`tree.rs:159-177`) returns early for `int_fval < 0` before calling
`find_in_bitset`, so `pos` is always `>= 0`. That is a fragile invariant: the
helper is `pub`-adjacent (module-internal) and its contract isn't enforced.
**Fix:** Make the helper total over its input type by mirroring C++ semantics
explicitly — take `pos: i32`, and `if pos < 0 { return false; }` at the top, or
debug-assert `pos >= 0`. Defense-in-depth against a future caller.

### WR-03: `objective.rs::convert` indexes `input[0]` / `output[0]` without a length guard

**File:** `crates/lgbm-model/src/objective.rs:189-204`, `231-246`, `250-255`
**Issue:** `convert` does `output[0] = convert_regression(input[0], ...)` for the
scalar objectives and `softmax`/`convert_multiclassova` index `input[0]` /
`input[..]` directly. These are only reachable via `predict.rs::resolve_objective`
which checks `kind.num_output() == ntpi` and sizes `raw_buf`/`conv_buf` to `m`, so
on the integrated path the slices are correctly sized. But `ObjectiveKind` and its
`convert` are `pub` (re-exported at the crate root, `lib.rs:32`); a direct caller
passing an empty `input` panics with an index-out-of-bounds rather than a typed
error — violating the "never panic on caller input" boundary contract the crate
states for itself (`error.rs:6-7`). `softmax` additionally reads `input[0]` and
`input[1..]` (`objective.rs:234-236`) with no empty-slice guard.
**Fix:** Either make `convert`/`softmax`/`convert_multiclassova` non-`pub` (keep
them crate-internal so the only entry is the validated `predict` path), or add an
explicit length check returning `ModelError` / documenting the debug-assert
precondition.

### WR-04: `split_kv` first-`=` split silently mis-handles `feature_names` / `monotone_constraints` values containing `=`

**File:** `crates/lgbm-model/src/model_text.rs:44-46`, `83-95`
**Issue:** The doc comment claims fidelity to the C++ `feature_names` /
`monotone_constraints` substr special-cases (`gbdt_model_text.cpp:439-442`), but
the implementation just does `line.split_once('=')`. C++ special-cases those two
keys precisely because a feature NAME can contain `=`. With the first-`=` split, a
feature named `a=b` is captured as value `b` for key `a` and the real
`feature_names=` content is lost / misattributed. On the committed fixtures
feature names are `Column_N` (no `=`), so this is latent, but the comment asserts a
behavior the code does not implement.
**Fix:** Special-case `feature_names` / `monotone_constraints` by matching the key
prefix (`line.strip_prefix("feature_names=")`) and taking the entire remainder as
the value, exactly as C++ does — or update the comment to state the limitation and
add a parser test with an `=`-bearing feature name.

### WR-05: `tree_sizes_overflow_is_err` test does not test what it documents

**File:** `crates/lgbm-model/src/model_text.rs:390-398`
**Issue:** The test replaces `"tree_sizes="` with `"tree_sizes=999999"`, producing
the line `tree_sizes=999999<original_size>` (the prefix is glued to the real
size). The comment says this makes the value "huge ... exceeds region", but the
actual token is a concatenation that may parse as one large number by luck of the
fixture; the test asserts only that *some* error occurs, so it would pass even if
the error were an unrelated parse failure. It does not exercise the
`total > region.len()` branch (`model_text.rs:146-153`) deterministically, nor the
`checked_add` overflow branch (`142-145`).
**Fix:** Construct a model whose `tree_sizes` sum deterministically exceeds the
tree region (e.g. a single tree with `tree_sizes=100000`), and assert the error
message contains "exceeds available tree region". Add a separate case for the
`usize` overflow path.

### WR-06: CSC/CSR `if c < width` / `if col < num_cols` silently drops in-range-but-trailing columns instead of erroring

**File:** `crates/lgbm-model/src/predict.rs:199-201`, `276-278`, `640-642`, `668-670`
**Issue:** When a sparse `col` is `>= width` (i.e. `>= max_feature_idx + 1`) but
`< num_cols`, the value is silently ignored (`if c < width { ... }`). This is
benign for prediction (the model never reads those columns), but it diverges from
the validation philosophy stated in the module header ("validate caller input at
the boundary FIRST"). A caller supplying `num_cols > max_feature_idx + 1` plus
nonzero trailing-column values gets a silently-different result with no signal.
Also note `check_cols` only enforces `num_cols >= need`, so `num_cols` strictly
greater than the model width is accepted and the extra columns are dropped without
comment.
**Fix:** This is acceptable behavior if intentional (it mirrors C++ ignoring
out-of-model features), but document it explicitly at each drop site, or decide to
reject `num_cols > max_feature_idx + 1`. At minimum the silent-drop semantics
should be a documented contract, not an implicit one.

## Info

### IN-01: `format_g` relies on `format!("{:.*e}", ...)` rounding == C `%g` rounding — unproven for ties

**File:** `crates/lgbm-model/src/format.rs:83-105`
**Issue:** The formatter delegates significant-digit rounding to Rust's
`{:.*e}` and assumes it produces byte-identical digits to C `fmt`'s `{:.17g}` /
`{:g}` for every double, including round-half-to-even tie cases. Rust's float
formatter is correctly-rounded (round-to-nearest-even), as is C99 `printf`, so
they should agree — but this is the byte-exact linchpin and the only guard is the
hand-written `g17_battery`/`g6_battery` plus a golden test that **SKIPs when the
fixture is absent** (`format.rs:336-345`). At review time `format_golden.txt` is
not committed, so the authoritative-vs-formatter assertion never runs.
**Fix:** Land `format_golden.txt` and confirm `golden_matches_formatter` actually
executes (fails loudly, not SKIPs) in CI. Consider adding a few explicit
round-half-to-even tie values (e.g. a double whose 18th digit forces a tie) to the
battery.

### IN-02: Unused byte-view bindings retained "for clarity / future use"

**File:** `crates/lgbm-model/src/model_text.rs:70`, `155`, `170`
**Issue:** `bytes` and `region_bytes` are computed and then explicitly discarded
with `let _ = (bytes, region_bytes);`. Dead bindings that only add noise and a
false impression they are load-bearing.
**Fix:** Remove `let bytes = text.as_bytes();`, `let region_bytes = region.as_bytes();`,
and the discard line.

### IN-03: `objective.rs` accepts `sqrt` token on any regression alias though C++ only honors it for `regression`

**File:** `crates/lgbm-model/src/objective.rs:121-128`
**Issue:** The code (and its own comment) note that only `regression` honors
`sqrt` in C++, yet it sets `sqrt` for all l1/l2 aliases. The comment argues this is
"harmless" because the token only appears for `regression sqrt`. True for emitted
models, but it means `regression_l1 sqrt` (which C++ would treat as plain l1)
would be transformed as sqrt here — a latent divergence on hand-crafted input.
**Fix:** Restrict `sqrt` honoring to the `regression` name only, matching C++
exactly; ignore it for the aliases.

### IN-04: `MODEL_LIGHTGBM_VERSION` (4.6.0) differs from pinned reference `LIGHTGBM_VERSION` (4.6.0.99)

**File:** `xtask/src/main.rs:49`, `63`
**Issue:** The capture pins pip `lightgbm` 4.6.0 while the in-repo C++ reference is
4.6.0.99. The model-text `version=v4` envelope is stable across these, but any
formatter/serialization drift between the pip wheel's `fmt` and the pinned source
would silently propagate into the goldens (the goldens become authoritative over
the pinned source). This is documented as human-approved in REFERENCE_MANIFEST, so
it is a recorded risk rather than a defect — flagged for traceability.
**Fix:** None required; ensure the manual-verification note in 03-VALIDATION.md
remains the audit trail for the version skew.

---

_Reviewed: 2026-06-05_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
