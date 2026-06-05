---
phase: 02-dataset-binning-determinism-root
reviewed: 2026-06-05T00:00:00Z
depth: standard
files_reviewed: 16
files_reviewed_list:
  - crates/lgbm-dataset/src/bin_mapper.rs
  - crates/lgbm-dataset/src/dataset.rs
  - crates/lgbm-dataset/src/efb.rs
  - crates/lgbm-dataset/src/error.rs
  - crates/lgbm-dataset/src/feature_group.rs
  - crates/lgbm-dataset/src/ingest.rs
  - crates/lgbm-dataset/src/lib.rs
  - crates/lgbm-dataset/src/metadata.rs
  - crates/lgbm-dataset/src/multi_val_bin.rs
  - crates/lgbm-dataset/src/bin/mod.rs
  - crates/lgbm-dataset/src/bin/dense_bin.rs
  - crates/lgbm-dataset/src/bin/sparse_bin.rs
  - crates/oracle-harness/src/comparator.rs
  - xtask/cpp/bin_capture.cpp
  - xtask/cpp/CMakeLists.txt
  - xtask/src/main.rs
findings:
  critical: 2
  warning: 5
  info: 4
  total: 11
status: issues_found
---

# Phase 02: Code Review Report

**Reviewed:** 2026-06-05
**Depth:** standard
**Files Reviewed:** 16
**Status:** issues_found

## Summary

The phase-02 binning layer is a careful, heavily-documented transcription of the
C++ `src/io/` subsystem, and most of the numeric kernel (`value_to_bin`,
`greedy_find_bin`, `find_bin_with_zero_as_one_bin`, the dedup `next_up` math, the
4-bit/sparse storage byte layouts, EFB `find_groups`/`fast_feature_bundling`, the
multi-val `+1` push convention, and the f32 query-weight arithmetic) was verified
line-for-line against both the in-tree C++ reference (`LightGBM/src/io/bin.cpp`,
`feature_group.h`, `sparse_bin.hpp`, `metadata.cpp`) and the transcription
harness and found faithful.

Two correctness/fidelity defects were found that violate the project's
non-negotiable parity contract under the **default** configuration:

1. **The ingestion pre-filter argument is wrong (CR-01).** `ingest.rs` passes the
   raw `min_data_in_leaf` as `min_split_data` to `find_bin_numeric`, but the C-API
   in-memory path it claims to mirror passes the *scaled* `filter_cnt =
   (min_data_in_leaf * total_sample_size) / num_dist_data`. Because
   `feature_pre_filter` defaults to `true` and `min_data_in_leaf` defaults to 20,
   every default-config ingest can mark different features trivial than C++ —
   a determinism-root divergence.

2. **The reviewed-path tests disable the very feature that hides CR-01 (CR-02).**
   Every `ingest.rs` test sets `feature_pre_filter = false`, so the corpus never
   exercises the default pre-filter path; the divergence ships untested.

The remaining findings are robustness/maintainability concerns (panics reachable
on internally-but-not-statically-guaranteed inputs, a documented CSR/CSC
densification semantics gap, and dead/duplicated code).

## Critical Issues

### CR-01: Ingestion passes raw `min_data_in_leaf` instead of the scaled `filter_cnt` to `FindBin`

**File:** `crates/lgbm-dataset/src/ingest.rs:90-100` (and `:94`)
**Issue:** `build_mapper` calls `BinMapper::find_bin_numeric(..., cfg.min_data_in_leaf, ...)`,
passing `min_data_in_leaf` directly as the `min_split_data` (pre-filter) argument.
The module doc-comment states this path mirrors the C-API in-memory path
(`LGBM_DatasetCreateFromMat` → `CreateSampleIndices` → `FindBin`). That path
routes through `DatasetLoader::ConstructFromSampleData`
(`LightGBM/src/io/dataset_loader.cpp:623-646`), which computes:

```cpp
const data_size_t filter_cnt = static_cast<data_size_t>(
    static_cast<double>(config_.min_data_in_leaf * total_sample_size) / num_dist_data);
bin_mappers[i]->FindBin(..., config_.min_data_in_bin, filter_cnt, config_.feature_pre_filter, ...);
```

i.e. C++ passes the **scaled** `filter_cnt`, not raw `min_data_in_leaf`. Since
`feature_pre_filter` defaults to `true` (`config.h:723`,
`lgbm-core/src/config/mod.rs:356`) and `min_data_in_leaf` defaults to 20
(`mod.rs:305`), `need_filter` will receive a different threshold than C++ for
every feature on the default path, so `is_trivial_` (and therefore which features
get a bin store, the EFB `used_features` set, and every downstream split) can
diverge. This breaks the ≤1e-12 parity contract — and it is the binning
determinism root, so the divergence cascades. The bug is masked today only
because all tests set `pre_filter = false`.

**Fix:** Compute `filter_cnt` exactly as C++ and pass it. `total_sample_size` is
the sampled-row count (`total_sample_cnt = k`); `num_dist_data` is the number of
distinct sampled rows (`std::set` size in `c_api.cpp`'s sampler). Thread both
through `finish_from_columns`/`build_mapper`:

```rust
// num_dist_data = number of DISTINCT sampled rows across all columns
// (matches c_api.cpp CreateSampleIndices dedup); total_sample_size = k.
let filter_cnt = ((cfg.min_data_in_leaf as i64 * total_sample_size as i64)
    / num_dist_data as i64) as i32;
BinMapper::find_bin_numeric(
    sampled, cfg.max_bin, cfg.min_data_in_bin,
    filter_cnt,                 // <-- was cfg.min_data_in_leaf
    cfg.feature_pre_filter, cfg.use_missing, cfg.zero_as_missing,
    total_sample_cnt, &[],
);
```

Verify the precise `total_sample_size` / `num_dist_data` definitions against
`c_api.cpp` (the in-memory sampler) before locking, since the integer-division
truncation is itself parity-load-bearing.

### CR-02: Ingestion parity tests never exercise the default `feature_pre_filter=true` path

**File:** `crates/lgbm-dataset/src/ingest.rs:341-349` (`cfg()` helper) and all tests using it
**Issue:** The shared test `cfg()` sets `c.feature_pre_filter = false`, and every
`from_mat`/`from_csr`/`from_csc` test reuses it. The default LightGBM behavior is
`feature_pre_filter = true`. As a result the entire pre-filter → `is_trivial_`
path (the one that surfaces CR-01) is untested at the ingestion boundary, so a
determinism-root divergence ships green. For a project whose sole contract is
bit-faithfulness, the highest-risk default path having zero coverage is itself a
defect.

**Fix:** Add an ingestion parity case with `feature_pre_filter = true` and
`min_data_in_leaf` at its default (20), built against a C++ golden captured
through the same scaled-`filter_cnt` path, asserting identical `is_trivial_` /
`num_bin_` / per-row `ASSIGN` vectors. This case must fail before CR-01 is fixed
and pass after.

## Warnings

### WR-01: `Dataset::construct` accepts an empty mapper set despite the doc claiming it is rejected

**File:** `crates/lgbm-dataset/src/dataset.rs:78-104`
**Issue:** The doc-comment says construct "Validates `num_data >= 0` **and a
non-empty mapper set** at the boundary, returning [`DatasetError`] rather than
panicking (Security V5)." The body only validates `num_data < 0`; an empty
`bin_mappers` vector is silently accepted (producing a 0-feature dataset). The
stated invariant is not enforced, so a caller relying on it gets no error.
**Fix:** Either drop the "non-empty mapper set" claim from the doc, or enforce it:
```rust
if bin_mappers.is_empty() {
    return Err(DatasetError::ShapeMismatch {
        detail: "bin_mappers must be non-empty".into(),
    });
}
```
Match whichever the C++ reference actually does (C++ allows 0 features in some
paths — confirm before adding the check, but the doc and code must agree).

### WR-02: `push_value` / `feature_to_group` / `feature_to_subfeature` panic on out-of-range real-feature index

**File:** `crates/lgbm-dataset/src/dataset.rs:219-248` (and `FinishedDataset` 337-361)
**Issue:** `self.used_feature_map_[real_feature]` indexes with a caller-supplied
`real_feature: usize` with no bounds check; an out-of-range index panics
(index-out-of-bounds) rather than being ignored or surfaced. `push_row` validates
the row *width* but `push_value` is also public and is the documented per-value
entry. The boundary contract for this crate is "never panic on caller input"
(error.rs module doc, Security V5). `real_feature_idx()` on `FinishedDataset`
(`:359-361`) has the same unchecked `self.real_feature_idx_[packed]`.
**Fix:** Bounds-check and return `DatasetError` (or document these as
internal-only and make them non-`pub`). At minimum:
```rust
pub fn push_value(&mut self, real_feature: usize, row: i32, value: f64) {
    let Some(&packed) = self.used_feature_map_.get(real_feature) else { return; };
    ...
}
```

### WR-03: `MultiValBin::push_one_row` sparse branch can panic / silently misbehave on a non-monotone `idx`

**File:** `crates/lgbm-dataset/src/multi_val_bin.rs:181-191`
**Issue:** The sparse branch writes `self.row_ptr_[(idx + 1) as usize] = values.len()`.
This assumes rows are pushed in ascending `idx` with no gaps (the per-row count is
later prefix-summed in `finish_load`). If `idx + 1 > num_data` it panics; if rows
are pushed out of order or skipped, the CSR `row_ptr_` is silently wrong (a later
row's `data_` slice will be misattributed) because `data_` is a single appended
stream while `row_ptr_` is positional. There is no assertion tying the append
position to `idx`. C++ `MultiValSparseBin::PushOneRow` has the same positional
assumption but is only ever driven by the ordered `PushDataToMultiValBin` loop;
the Rust API is `pub` with no such guard.
**Fix:** Assert the ordering invariant (`debug_assert_eq!` the running append
offset against the expected position) or document the precondition and keep the
method crate-private. At minimum bounds-check `idx`.

### WR-04: `arg_max` returns 0 for an empty slice — masks an invalid-state read

**File:** `crates/lgbm-dataset/src/bin_mapper.rs:666-674`
**Issue:** `arg_max(&[])` returns `0`, and callers then index `cnt_in_bin[0]`
(`bin_mapper.rs:373`, `:612`). In the current control flow `arg_max` is only
reached when `!is_trivial_` (so `num_bin_ > 1` and `cnt_in_bin` is non-empty), so
it is not currently triggerable — but the function silently returns a valid-looking
index for an empty input, which would turn a future logic error into a wrong
result (reading bin 0 of an empty histogram) instead of a panic. C++ `ArgMax`
returns 0 for empty too, so this is faithful, but the silent-zero is a latent trap.
**Fix:** Add `debug_assert!(!v.is_empty())` to localize any future misuse without
changing release behavior.

### WR-05: `next_nonzero` advances `cur_pos` using an out-of-range delta before the bound check (relies on the terminator invariant)

**File:** `crates/lgbm-dataset/src/bin/sparse_bin.rs:96-105`
**Issue:** `next_nonzero` does `*i_delta += 1; *cur_pos += self.deltas_[*i_delta]`
*before* checking `*i_delta < self.num_vals_`. This is a verbatim port of C++
`NextNonzero` and is only safe because `load_from_pair` always pushes a trailing
`deltas_.push(0)` terminator so `deltas_[num_vals_]` is in range. However, if a
`SparseBin` is read before `finish_load` (e.g. `data()` called on a freshly
`new`'d bin, `deltas_` empty), the index `deltas_[0]` panics. `data()` is `pub`
and has no "finish_load first" guard.
**Fix:** Guard `data()`/`next_nonzero` against an unloaded (`deltas_.is_empty()`)
state, or document that read methods require `finish_load` and `debug_assert!` it.

## Info

### IN-01: CSR/CSC ingest densifies by column, diverging from the C++ row-iterator sample/zero accounting

**File:** `crates/lgbm-dataset/src/ingest.rs:259-269` (CSR), `:322-332` (CSC)
**Issue:** Both sparse ingest paths build a fully dense per-column buffer with
absent entries defaulting to `0.0`, then bin every column as dense. The real
C-API CSR/CSC path samples non-zero entries and tracks `num_per_col` / zero counts
differently; densifying to explicit zeros changes `total_sample_cnt`, `zero_cnt`,
and the sampled set feeding `find_bin`. The code comments label this "Open Q2",
so it is a known/accepted MVP simplification — flagged for traceability because it
is a potential parity divergence for sparse inputs once goldens cover them.
**Fix:** None required for this phase; ensure a sparse-input parity golden is added
before the sparse path is declared faithful.

### IN-02: Dead helper `find_bin_with_zero_as_one_bin` parameter / unused `cur_cnt_inbin` tail

**File:** `crates/lgbm-dataset/src/bin_mapper.rs:727` and `find_bin_from_column` (`:635-662`)
**Issue:** (a) `greedy_find_bin` has a commented-out `cur_cnt_inbin += counts[...]`
tail (`:727`) faithfully noting the C++ dead store — fine, but leave a one-line
note that it is intentionally elided. (b) `BinMapper::find_bin_from_column`
duplicates `build_mapper` in `ingest.rs` almost exactly (same sample→gather→
find_bin_numeric), but passes `min_split_data` differently and is not used by the
ingest path; it is a second, divergent copy of the same logic (and would inherit a
*different* fix than CR-01). Duplicated determinism-root logic is a maintenance
hazard.
**Fix:** Either remove `find_bin_from_column` or have `build_mapper` delegate to it
(single source of truth), so the CR-01 fix cannot be applied to only one copy.

### IN-03: `min_val_`/`max_val_` use `unwrap_or(0.0)` masking the empty-distinct-values case

**File:** `crates/lgbm-dataset/src/bin_mapper.rs:282-283`, `:490-491`
**Issue:** `*distinct_values.first().unwrap_or(&0.0)` silently substitutes 0.0 when
`distinct_values` is empty. C++ reads `distinct_values.front()/back()` directly
(UB if empty), so in practice `distinct_values` is never empty here (the
zero-pseudo-value push guarantees ≥1 element when relevant). The `unwrap_or` is
safer than C++ but hides the assumption; if the invariant ever breaks, the value
silently becomes 0.0 rather than failing.
**Fix:** `debug_assert!(!distinct_values.is_empty())` to document/verify the
invariant.

### IN-04: `comparator.rs` `ValueMismatch` / `abs_diff_within` are unused by the binning goldens

**File:** `crates/oracle-harness/src/comparator.rs:86-112`
**Issue:** Phase-02 binning goldens are all bit-exact (`compare_exact_*`); the
`~1e-6` tolerance comparators (`abs_diff_within`, `compare_within`,
`Mismatch::ValueMismatch`) are not exercised by any phase-02 source path (only by
their own unit tests). Not a defect — they are forward-looking API for training
goldens — but worth noting so a future reviewer does not assume the tolerance path
is in use for binning (using it for binning would be a parity bug).
**Fix:** None; optionally add `#[allow(dead_code)]`-free justification doc, already
largely present.

---

_Reviewed: 2026-06-05_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
