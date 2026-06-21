---
phase: 08-python-bindings
plan: 04
subsystem: api
tags: [pyo3-polars, arrow, polars, categorical, dtype-routing, ab-parity]

# Dependency graph
requires:
  - phase: 08-python-bindings
    plan: 03
    provides: "dense/sparse Dataset constructors densifying into f64 RawCorpus rows"
  - phase: 08-python-bindings
    plan: 01
    provides: "RawCorpus.categorical_features -> build_feature_columns_from_raw find_bin_categorical routing"
provides:
  - "marshal::polars_df_to_corpus — pyo3-polars PyDataFrame -> (f64 rows, names, categorical indices), Arrow-side (no numpy round-trip)"
  - "Dataset::from_polars staticmethod + dataset_from_polars Python wrapper (auto / names+indices override)"
  - "dtype routing (D-04): Categorical/Enum/String -> categorical feature; numeric -> numeric; explicit override has precedence"
  - "polars 0.53 dep with dtype-u8/u16 enabled (narrow categorical code widths)"
affects: [08-06-callbacks, 08-07-sklearn, 08-08-persistence]

# Tech tracking
tech-stack:
  added:
    - "polars 0.53 (direct dep, default-features=false, dtype-categorical/u8/u16)"
  patterns:
    - "Categorical code extraction via Series::to_physical_repr() (handles u8/u16/u32 widths) then cast f64"
    - "Override precedence: Some(names) fully decides routing; None = dtype 'auto'"
    - "Column-major extract -> transpose to the row-major f64 rows the facade bins"

key-files:
  created:
    - crates/lgbm-python/python/tests/test_polars_input.py
  modified:
    - crates/lgbm-python/src/marshal.rs
    - crates/lgbm-python/src/dataset.rs
    - crates/lgbm-python/Cargo.toml
    - Cargo.lock
    - crates/lgbm-python/python/lightgbm_rs/__init__.py

key-decisions:
  - "polars Categorical codes are read via to_physical_repr() (not the new 0.53 Categories/RevMapping API directly), which is width-agnostic; dtype-u8/u16 features are REQUIRED or to_physical_repr panics with unimplemented!() for narrow code widths."
  - "A/B categorical parity uses the integer-code override path on BOTH sides (rs polars override + real lightgbm categorical_feature=[idx]) to guarantee identical category-id alignment, sidestepping polars' internal code-assignment order. The Enum dtype-routing test then proves auto-routing extracts those same codes."
  - "categorical_feature override accepts names AND integer indices (resolved to names in the Python wrapper); 'auto' (default) = dtype routing."

requirements: [PYB-02]
status: complete
---

## What landed

polars DataFrames now ingest **zero-copy via Arrow** (pyo3-polars), with
dtype-driven categorical/numeric routing and an explicit override (D-03/D-04).
Columns are consumed Arrow-side in Rust — NO numpy round-trip, which would erase
the `Categorical`/`Enum` dtype.

- **Task 1 — ingest + routing:** `marshal::polars_df_to_corpus` classifies each
  column: `Categorical`/`Enum`/`String` → categorical (physical codes via
  `to_physical_repr`, any width), numeric → f64. `Dataset::from_polars` sets
  `RawCorpus.categorical_features` so 08-01's `build_feature_columns_from_raw`
  routes those columns through `find_bin_categorical`. An explicit
  `categorical_feature` list takes precedence over dtype routing.
- **Task 2 — parity tests + wrapper:** `dataset_from_polars` (auto / names+indices)
  + `test_polars_input.py` (4 tests, green).

## Verification

- `cargo build -p lgbm-python` ✓; `cargo clippy -p lgbm-python` ✓ clean (type alias
  for the corpus-parts tuple; no unwrap/panic in production src).
- `maturin develop` ✓; `pytest python/tests/` → **30 passed** (4 new in
  `test_polars_input.py`):
  - `test_polars_numeric_matches_numpy` — polars numeric == numpy (bit-identical).
  - `test_enum_dtype_routes_like_integer_codes` — Enum auto-routing == integer-code
    override (proves auto-routing extracts the correct physical codes).
  - `test_categorical_ab_matches_real_lightgbm` — **categorical splits match real
    lightgbm 4.6 at atol=1e-6** (categorical_feature A/B).
  - `test_categorical_override_changes_routing` — override takes effect.

## Deviations

- **Recovery deviation (process):** the assigned executor subagent died on a
  transient API socket error mid-plan (only the Cargo.toml polars-dep edit +
  unused imports in `marshal.rs` were on disk, uncommitted, no commits). The
  orchestrator completed the plan inline: wrote the polars marshalling +
  `Dataset::from_polars` + wrapper + tests, resolved the polars-0.53 categorical
  API (new `Categories`/`CategoricalMapping` system) via the width-agnostic
  `to_physical_repr`, fixed the `dtype-u8/u16` panic, built + tested green, and
  committed in 2 atomic commits + SUMMARY.
- **Rule 2 (deps):** added `dtype-u8`/`dtype-u16` to the polars features beyond
  the planned `dtype-categorical` — required because polars 0.53 stores small
  categoricals with a u8/u16 physical code width whose Series impl is otherwise
  feature-gated off (`to_physical_repr` → `unimplemented!()` panic).

## Notes for downstream plans

- The `lightgbm_rs` Python surface now accepts numpy (f32/f64), scipy CSR/CSC, and
  polars DataFrames (with categorical routing).
- 08-06 (callbacks/feval) and 08-07 (sklearn) build on this Dataset/train surface.

## Self-Check: PASSED
