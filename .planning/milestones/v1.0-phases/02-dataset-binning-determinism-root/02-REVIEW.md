---
phase: 02-dataset-binning-determinism-root
reviewed: 2026-06-05T00:00:00Z
depth: standard
scope: 02-07 gap-closure delta (1854704^..HEAD)
files_reviewed: 4
files_reviewed_list:
  - crates/lgbm-dataset/src/dataset.rs
  - crates/lgbm-dataset/src/ingest.rs
  - crates/lgbm-dataset/tests/default_config_ingest_parity.rs
  - xtask/cpp/bin_capture.cpp
findings:
  critical: 0
  warning: 4
  info: 3
  total: 7
status: issues_found
---

# Phase 02 (plan 02-07 gap-closure): Code Review Report

**Reviewed:** 2026-06-05
**Depth:** standard
**Files Reviewed:** 4
**Status:** issues_found

> Scope note: this report reviews ONLY the 02-07 gap-closure delta
> (`1854704^..HEAD`) that closes CR-01. The broader phase-02 review (16 files) was
> recorded in `02-06-REVIEW.md`. This file was overwritten per the workflow `review_path`.

## Summary

Adversarial review of the 02-07 delta closing CR-01 on the dataset-binning determinism root.
The delta (a) filters trivial features in `Dataset::construct`/`construct_bundled`
(`used_feature_map_[real] = -1`), (b) routes the ingest tail (`finish_from_columns`) through
the `enable_bundle` dispatch with an `EfbSamples` built to the sampled-set convention, (c) adds
the `default_config_ingest_parity` test asserting trivial-exclusion + per-non-trivial
group/subfeature parity, and (d) extends `bin_capture.cpp` with the default-config emitter.

I traced the live ingest path end-to-end against the C++ reference transcription and the
supporting `efb.rs` / `bin_mapper.rs` / `feature_group.rs` code. The core CR-01 fix and the
parity test are **correct and genuinely discriminating**: the test asserts post-shuffle group
IDs (`feature_to_group`) against the golden, reads STORED bins (not recomputed), panics on a
missing golden, and requires both a trivial and a non-trivial feature to be present. The Rust
live path (`construct_bundled` → `FeatureGroup::new` with `force_dense=true`) and the emitter
(`FeatureGroupMV` with `CreateDenseBin`) agree, and the EFB sample-set conventions match
between the two sides bit-for-bit. The test compiles and runs green (verified).

No BLOCKER-class divergence was proven on the **live** determinism path. I did find WARNING-class
issues: a real C++-fidelity divergence in the now-test-only no-bundle `Dataset::construct` path
(uses `new_single` → `force_dense=false`, contradicting both C++ and this file's own module
doc); an internal-invariant `expect()` that panics on an EFB-partition bug instead of erroring;
an unverifiable-fidelity gap for the EfbSamples convention (the fixture is
convention-independent so cannot detect a convention error); and a structural weakness in how
the test discovers the trivial feature. Info items cover a misplaced doc comment, a
bool-vs-true assertion, and a dead-store edge in the test parser.

## Structural Findings (fallow)

No `<structural_findings>` block was supplied with this review; none to report.

## Narrative Findings (AI reviewer)

## Warnings

### WR-01: No-bundle `Dataset::construct` diverges from C++ `force_dense=true` (and from its own doc)

**File:** `crates/lgbm-dataset/src/dataset.rs:130` (module doc `:19-21`)
**Issue:** `Dataset::construct` builds each group via `FeatureGroup::new_single(m, num_data)`,
which calls `create_bin_data(num_data, false, /*force_dense=*/false, false)`
(`feature_group.rs:161`). With `force_dense=false`, a single-value feature whose
`sparse_rate_ >= 0.7` is stored as a **SparseBin**. But C++ `Dataset::Construct` builds **every**
group — including the `OneFeaturePerGroup` (no-bundle) branch — through the primary
`FeatureGroup` constructor with `CreateBinData(..., force_dense=true, ...)` (confirmed in
`02-RESEARCH.md` §8 lines 413-419 and `02-PATTERNS.md:264`), so C++ would store the SAME
feature as a **DenseBin**. The live `construct_bundled` path correctly uses `FeatureGroup::new`
(`force_dense=true`, `:230`), so the two construct paths produce different storage backends for a
sparse single-value feature — a real C++-fidelity divergence affecting which `Bin`
implementation (and byte-layout golden) backs the feature.

The module doc-comment at `dataset.rs:19-21` explicitly claims `construct` "uses
`CreateBinData(..., force_dense=true, force_sparse=false)`" — the code contradicts the doc.

Mitigating fact: `Dataset::construct` is now called ONLY from unit tests
(`dataset.rs:430,438,445,462,541`); production ingest routes through `construct_bundled`. So
this is not a live determinism BLOCKER today, but it is a public API that silently diverges from
C++ and will mislead a future non-EFB caller.
**Fix:** Route `construct` through the same group-build loop as `construct_bundled` (i.e. build
each group via `FeatureGroup::new(1, false, vec![m], num_data, group_id)` with `force_dense=true`),
eliminating the second storage path; or, if `new_single`'s sparse-selection is intentional for a
different (binary-load) constructor, correct the `construct` module doc to stop claiming
`force_dense=true`.

### WR-02: `.expect()` on EFB partition can panic instead of returning a typed error

**File:** `crates/lgbm-dataset/src/dataset.rs:225` (also `:127-129`)
**Issue:** `construct_bundled` does
`mappers_opt[real_fidx as usize].take().expect("each used feature assigned to exactly one group")`.
If `fast_feature_bundling` ever returns a partition where a feature appears in two groups (or a
real index out of range), this **panics** rather than returning a `DatasetError`. The module's
stated discipline (`ingest.rs:15-18`, "VALIDATES all caller input at the boundary FIRST … never
a panic") is undermined because the panic is reachable through the grouping algorithm rather than
a checked invariant. The clean-partition property is neither asserted nor validated; a future EFB
change could turn a grouping bug into a crash on the ingest path.
**Fix:** Convert to a typed error and/or assert the partition invariant:
```rust
let m = mappers_opt[real_fidx as usize].take().ok_or_else(|| {
    DatasetError::ShapeMismatch { detail: format!(
        "EFB produced a non-partition: feature {real_fidx} assigned to multiple groups") }
})?;
```

### WR-03: EfbSamples sampled-set convention is self-consistent but its fidelity to real lib_lightgbm is unverifiable here

**File:** `crates/lgbm-dataset/src/ingest.rs:138-162`; `xtask/cpp/bin_capture.cpp:2258-2301`
**Issue:** `build_efb_samples` (Rust) and the EFB block of `EmitDefaultConfigIngest` (C++) both
build `sample_indices`/`num_per_col`/`total_sample_cnt` to the "sampled-set position
`0..sample_cnt`, non-zero/NaN filter (`|v| > 1e-35 || isnan`), `total_sample_cnt = sample_cnt`"
convention attributed to `c_api.cpp:1352-1374`. I verified the two SIDES agree bit-for-bit (same
threshold, same NaN clause, same position semantics, same `total_sample_cnt`). The golden
therefore proves Rust == emitter, but because `external_libs` is unvendored and `lib_lightgbm`
cannot be built here (`bin_capture.cpp:10-32`), the golden does NOT independently prove
emitter == real `lib_lightgbm` C-API. The emitter itself concedes the grouping is
"convention-independent on this dense fixture" (`single_val_max_conflict_cnt = sample_cnt/10000
= 0`, every feature dense → strict one-feature-per-group, `:2253-2257`) — so this fixture cannot
detect a convention error; the test passes regardless of whether the sampled-set convention is
the truly-correct c_api convention.
**Fix:** Add (in a later plan) an EFB fixture where the convention is **observable** — sparse
mutually-exclusive features with `sample_cnt < num_rows` so `single_val_max_conflict_cnt > 0` and
the index/total convention actually changes grouping. Until then, annotate the test that it gates
only Rust-vs-emitter agreement, not c_api-convention fidelity.

### WR-04: Test "discovers" the trivial feature only via the golden it is validating (weak independent check)

**File:** `crates/lgbm-dataset/tests/default_config_ingest_parity.rs:290,302-305,322-336`
**Issue:** `used_count`, the "at least one trivial + one non-trivial" guard, and the per-feature
branch are all derived from `g.features[*].is_trivial`, read out of the **same golden file** whose
correctness is under test. The genuine Rust-vs-golden cross-check of `is_trivial`
(`mapper.is_trivial_ == fg.is_trivial`, `:366-370`) runs ONLY inside the `!fg.is_trivial` branch.
A feature the golden wrongly marks trivial (with Rust agreeing it is dropped) is never
re-derived and compared — the test just asserts `feature_to_group(f) == -1` and `continue`s. So an
emitter mistake that over-marks a feature trivial moves `used_count`, `num_groups`, and the branch
in lockstep and still passes.
**Fix:** For EVERY feature (trivial or not), independently rebuild the Rust mapper from the raw
column + header config and assert `rust_mapper.is_trivial_ == fg.is_trivial` BEFORE branching on
the golden flag, so the discriminating signal does not depend on the artifact being validated.

## Info

### IN-01: Misplaced doc comment in dataset.rs tests

**File:** `crates/lgbm-dataset/src/dataset.rs:474-479`
**Issue:** The `///` block describing the "compile-fail documentation" / immutability proof is
attached to `empty_samples(num_cols)`, which is unrelated. It belongs above
`finished_dataset_has_no_mutating_api` (`:536`) or `finish_load_yields_immutable_read_only_view`
(`:451`).
**Fix:** Move the block to the relevant test, or demote it to `//` so it stops misdocumenting
`empty_samples`.

### IN-02: `assert_eq!(cond, true, ...)` instead of `assert!`

**File:** `crates/lgbm-dataset/tests/default_config_ingest_parity.rs:242-248`
**Issue:** `assert_eq!(g.bin_construct_sample_cnt < g.num_rows, true, ...)` compares a bool to
`true` (clippy `bool_assert_comparison`); less readable than `assert!`.
**Fix:** `assert!(g.bin_construct_sample_cnt < g.num_rows, "golden must have sample_cnt ({}) < num_rows ({})", g.bin_construct_sample_cnt, g.num_rows);`

### IN-03: ASSIGN parser silently zeroes `stored` when the cell list is absent

**File:** `crates/lgbm-dataset/tests/default_config_ingest_parity.rs:178-188`
**Issue:** When an `ASSIGN` line has no value list, `tokens.last()` is `"f=<i>"`, the
`starts_with("f=")` guard yields `stored = Vec::new()` while `has_assign` is still `true`. A
non-trivial feature would then pass the representation gate (`:209-214`) yet later compare an
empty `stored` against 200 Rust bins, surfacing as an opaque length mismatch rather than a clear
"ASSIGN had no per-row data" diagnostic. Latent today (the emitter always writes all rows).
**Fix:** In the `f=`-only branch for a non-trivial feature, panic with an explicit
"non-trivial ASSIGN line missing per-row data" message instead of storing an empty vector.

---

_Reviewed: 2026-06-05_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
