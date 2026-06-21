---
phase: 02-dataset-binning-determinism-root
plan: 06
reviewed: 2026-06-05T00:00:00Z
depth: standard
files_reviewed: 5
files_reviewed_list:
  - crates/lgbm-dataset/src/bin_mapper.rs
  - crates/lgbm-dataset/src/ingest.rs
  - crates/lgbm-dataset/tests/default_config_ingest_parity.rs
  - xtask/cpp/bin_capture.cpp
  - xtask/src/main.rs
findings:
  critical: 1
  warning: 4
  info: 3
  total: 8
status: issues_found
---

# Phase 2 Plan 06 (gap closure): Code Review Report

**Reviewed:** 2026-06-05
**Depth:** standard
**Files Reviewed:** 5
**Status:** issues_found

## Summary

Plan 02-06 fixes the right root cause: the in-memory ingest path now feeds the
SCALED `filter_cnt = (min_data_in_leaf * sample_cnt) / num_rows` (C++ integer
truncation) into `find_bin_numeric` through a single source-of-truth helper
`scaled_filter_cnt`. The helper's arithmetic, the argv-slot shift in the C++
capture harness (kFirstExampleArgv 9->10 with the seed anchor preserved), the
seed wiring, and the f32-representable matrix round-trip are correct, and the new
parity test passes. I verified: the example-golden seed base
`master_seed + 8 + (a - kFirstExampleArgv)` is genuinely unchanged by the slot
shift; `scaled_filter_cnt(20,50,200)==5` and `(20,200,200)==20`; the argc
validation correctly rejects the empty-example argc=10 case while accepting
argc>=11; the dense/CSR/CSC paths all funnel through `finish_from_columns ->
build_mapper` so they inherit the fix.

The one BLOCKER: the gap-closure golden is built against the WRONG reference for
its per-row STORED-bin assertion. The C++ emitter `StoredBinSingleGroup` mirrors
the Rust non-bundled `Dataset::construct` (which keeps trivial features in the
store) rather than true C++ `Dataset::Construct` (dataset.cpp:337-341), which
EXCLUDES trivial features entirely. The committed golden's trivial control
feature f2 carries non-zero stored bins for all 200 rows, but a real C++ dataset
would store nothing for f2. The test passes only because the emitter was written
to match the divergent Rust path, so the STORED-bin parity assertion does not
compare against C++ Construct for trivial features.

## Critical Issues

### CR-01: Trivial features are stored (not excluded), diverging from C++ `Dataset::Construct`; the new golden masks it

**File:** `crates/lgbm-dataset/src/dataset.rs:84-94` (non-bundled `construct`),
`crates/lgbm-dataset/src/ingest.rs:122` (uses it),
`xtask/cpp/bin_capture.cpp:2140-2152` (`StoredBinSingleGroup`),
`crates/lgbm-dataset/tests/fixtures/default_config_ingest.txt` (f2 ASSIGN)

**Issue:**
C++ `Dataset::Construct` (LightGBM/src/io/dataset.cpp:337-341) builds
`used_features` from only the NON-trivial mappers
(`!ref_bin_mappers[i]->is_trivial()`), groups over just those, and sets
`used_feature_map_[real] = -1` for every trivial feature — so a trivial feature
is dropped: no FeatureGroup, no bin offset, no stored bins.

The Rust non-bundled `Dataset::construct` used by `from_mat`
(ingest.rs:122 -> finish_from_columns:121) instead builds a FeatureGroup for
EVERY feature and sets `used_feature_map_ = (0..num_features_)` (dataset.rs:94),
so `push_value` (dataset.rs:219-227) never early-returns for a trivial feature and
stores its bins. (The bundled path dataset.rs:131-133 DOES filter trivial
features — but `from_mat` calls the non-bundled `construct`, not
`construct_bundled`.)

Verified against the committed golden: trivial control feature **f2**
(`is_trivial=1`, num_bin=3, trivial via pre-filter not via num_bin<=1) has an
ASSIGN line of all `1`/`2` non-zero stored bins for 200 rows. A real C++ dataset
would store NOTHING for f2 and would not even allocate it as a group. The
divergence also shifts `num_features_`, `feature2group_`, `feature2subfeature_`,
and every downstream `num_total_bin_` / bin-offset once a trivial feature is
present — the exact determinism-root invariants this phase protects.

The new parity test does not catch this because `StoredBinSingleGroup`
(bin_capture.cpp:2140-2152) computes `ValueToBin` for trivial features too — it
never consults `is_trivial_` — so the golden was produced to match the Rust
behavior, not C++ Construct. The plan's acceptance criterion ("STORED per-row bins
vs a C++ golden") is thus met against an emitter that does not model C++ Construct
for the trivial case, defeating the intent.

**Fix:**
Make the non-bundled `Dataset::construct` exclude trivial features as C++ does,
and regenerate the golden through a path that models C++ `Dataset::Construct`
(trivial feature -> dropped / no store), so the STORED-bin assertion actually
compares against C++:

```rust
// dataset.rs::construct — mirror dataset.cpp:337-341
let used_features: Vec<i32> = (0..num_features_)
    .filter(|&i| !bin_mappers[i as usize].is_trivial_)
    .collect();
// build groups / feature2group_ / feature2subfeature_ over `used_features` only;
// used_feature_map_[real] = -1 for trivial features so push_value drops them.
```

Then emit the trivial-feature representation that C++ Construct yields (skip the
ASSIGN/store for trivial features) and have the Rust test assert the trivial
feature is excluded from `used_feature_map_`. The is_trivial_ flip on f1 (which is
non-trivial under the correct scaled threshold) must remain stored normally so the
test still discriminates the fix.

## Warnings

### WR-01: Parity test silently PASSES when the golden fixture is missing

**File:** `crates/lgbm-dataset/tests/default_config_ingest_parity.rs:190-198`

**Issue:** When `default_config_ingest.txt` cannot be read, the test prints a SKIP
message and `return`s — reporting success. A lost, truncated, or un-regenerated
committed golden then yields a green test that asserts nothing, breaking the
fails-before/passes-after contract and giving false confidence that default-config
parity holds. The fixture is committed, so a missing file is an error, not a valid
skip.

**Fix:**
```rust
let text = std::fs::read_to_string(&gpath).unwrap_or_else(|e| {
    panic!("committed golden {} unreadable: {e}", gpath.display())
});
```
If a toolchain-less opt-out is truly needed, gate it on an explicit env var or
`#[ignore]`, not a silent pass.

### WR-02: `build_mapper` duplicates the sampling/gather/filter path instead of delegating to `find_bin_from_column`

**File:** `crates/lgbm-dataset/src/ingest.rs:81-107` vs
`crates/lgbm-dataset/src/bin_mapper.rs:642-672`

**Issue:** The plan preferred collapsing both sites onto ONE sampling+find_bin
path (build_mapper delegating to find_bin_from_column). Instead `build_mapper`
reimplements `create_sample_indices` -> gather -> `scaled_filter_cnt` ->
`find_bin_numeric`, duplicating logic `find_bin_from_column` already contains; only
`scaled_filter_cnt` is shared. The two copies can drift (gather indexing,
total_sample_cnt source, column-index cast) and silently diverge — the precise
failure mode that produced this gap.

**Fix:** Delegate so there is exactly one sampling+find_bin path:
```rust
fn build_mapper(column: &[f64], cfg: &Config) -> BinMapper {
    BinMapper::find_bin_from_column(
        column, cfg.max_bin, cfg.min_data_in_bin, cfg.min_data_in_leaf,
        cfg.feature_pre_filter, cfg.use_missing, cfg.zero_as_missing,
        cfg.bin_construct_sample_cnt, cfg.data_random_seed, &[],
    )
}
```

### WR-03: `from_mat` ingest cannot apply forced bin bounds (forcedbins_filename) — silent C++ divergence

**File:** `crates/lgbm-dataset/src/ingest.rs:96-106` (passes `&[]`),
`bin_mapper.rs:642-672` (forced param exists but unused by ingest)

**Issue:** C++ `ConstructFromSampleData` feeds per-feature `forced_bin_bounds`
(dataset_loader.cpp:621,645-648) into `FindBin`. The Rust ingest path hard-codes
`&[]`, so any config with `forcedbins_filename` set bins differently from C++ with
no error. Pre-existing (not introduced by 02-06) but on the determinism contract
and adjacent to the touched code.

**Fix:** Thread per-feature forced bounds through `finish_from_columns ->
build_mapper -> find_bin_numeric`, or reject `forcedbins_filename` at the boundary
until supported.

### WR-04: `need_filter` numeric branch indexes via `len()-1` guarded only by `>= 1`; asymmetric with the categorical `saturating_sub`

**File:** `crates/lgbm-dataset/src/bin_mapper.rs:724-746`

**Issue:** The numeric branch is safe TODAY (callers only reach it with
`num_bin > 1`, so `cnt_in_bin.len() >= 2`, and the `>= 1` guard plus `0..len-1`
yields an empty range for len==1). But the categorical branch uses
`saturating_sub(1)` while the numeric branch relies on the caller invariant — a
future call with an empty `cnt_in_bin` would panic on the numeric `len()-1`. The
asymmetry is a latent footgun.

**Fix:** Use `0..cnt_in_bin.len().saturating_sub(1)` in the numeric branch too for
symmetry and to remove the empty-slice panic edge.

## Info

### IN-01: `scaled_filter_cnt` doc overstates the i64-vs-double equivalence

**File:** `crates/lgbm-dataset/src/bin_mapper.rs:699-721`

**Issue:** The doc claims the i64 integer divide is "bit-identical" to C++'s double
divide-then-truncate. This holds only for non-negative operands whose product fits
in 53 mantissa bits (true for the magnitudes reachable here). For very large
products the double becomes inexact and the two can differ by 1. The i64 form is
actually safer than C++'s int-multiply (overflow). Worth bounding the claim.

**Fix:** Tighten the doc to "identical for the non-negative, <=2^53 operands
reachable here"; optionally add a large-but-in-range unit case.

### IN-02: Golden parser reads MATRIX/ASSIGN payload via `tokens.last()` — position-fragile

**File:** `crates/lgbm-dataset/tests/default_config_ingest_parity.rs:142,159`

**Issue:** The payload is parsed as the LAST whitespace token; appending any
trailing field to the emitter (e.g. a checksum) would silently break parsing. The
ASSIGN `starts_with("f=")` special-case (line 160) hints the author already hit
ordering fragility.

**Fix:** Select the payload by an explicit key prefix (e.g. `cells=`/`data=`)
emitted by `bin_capture.cpp`, rather than by token position.

### IN-03: Narrowing casts in the kernel follow C++ `static_cast` but lack debug guards

**File:** `crates/lgbm-dataset/src/bin_mapper.rs:676-684` and the `as i32` /
`as usize` / `as u32` casts in `find_bin_numeric`

**Issue:** The casts mirror C++ semantics faithfully (within the verbatim-
transcription discipline) but would wrap silently on out-of-range inputs. This is
informational; a few `debug_assert!`s on load-bearing invariants
(`num_bin_ >= 1`, indices `< len`) would surface contract violations in tests
without affecting release parity.

**Fix:** Optional: add `debug_assert!` guards on the binning invariants.

---

_Reviewed: 2026-06-05_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
