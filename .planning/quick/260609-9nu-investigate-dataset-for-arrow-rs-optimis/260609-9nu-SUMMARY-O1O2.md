---
quick_id: 260609-9nu
title: Implement O1+O2 — column-major RawCorpus
status: complete
date: 2026-06-08
type: implementation
---

# Summary — O1+O2 column-major `RawCorpus`

Implemented the column-major flat `RawCorpus` from the investigation (O1 eliminate the
double transpose, O2 contiguous buffer). **Bit-exact gate GREEN; Python A/B GREEN.**

## What changed
- **`RawCorpus`** (`crates/lgbm/src/booster.rs`): `features: Vec<Vec<f64>>` (row-major)
  → private flat **column-major** store (`col_major: Vec<f64>` + `num_rows`/`num_cols`).
  New API: `new(rows)` (row-major in, transposes once — numpy/CSR/CSC), `from_columns`
  (column-major in, NO transpose — polars/Arrow, **O1**), `num_data`/`num_features`,
  `column(j) -> &[f64]` (contiguous, **O2**), `value`, `to_rows`. Ragged input is caught
  by the `col_major.len() == rows*cols` invariant → typed `InvalidCorpus` (no panic).
- **`build_feature_columns_from_raw`**: reads `corpus.column(j)` directly — no per-column
  `iter().map(row[j])` gather; ragged validation via the invariant.
- **Shared driver** `train_inner_columns[_full]`: dropped the `feature_rows: &[Vec<f64>]`
  param (was used only for `num_data` + `feature_infos`); now takes precomputed `num_data`
  + `feature_infos`. DenseCorpus caller computes them row-major (`feature_infos_from_rows`);
  RawCorpus caller column-major (new `feature_infos_from_columns`, byte-identical output).
  `train_raw` label-view now carries empty features (resolve_objective reads labels only).
- **Python** (`marshal.rs` + `dataset.rs`): `polars_df_to_corpus` returns COLUMN-major
  data (deleted the transpose block); `from_polars` → `RawCorpus::from_columns` via new
  `from_columns_with_categorical`. Numpy/CSR/CSC keep `RawCorpus::new`. Removed dead
  `from_rows_with_categorical`.
- **Tests**: migrated `RawCorpus { .. }` literals (booster.rs ×4, raw_bin_train_parity ×2)
  to constructors. `DenseCorpus` untouched everywhere (bench_train, boosting_parity, predict).

## Net effect
- **Polars/Arrow path:** col→row→col double transpose **eliminated** (0 transposes).
- **All paths:** binning reads each feature as a contiguous slice; `RawCorpus` is one
  flat allocation instead of `num_rows` nested `Vec`s.
- **Parity:** value-preserving reorder only → bit-exact gate unchanged.

## Verification
lgbm 41/41 · raw_bin_train_parity 2/2 (C++ golden + identity bit-exact) · boosting_parity
75/75 · rng_parity 1/1 · learner_parity 29/29 · kernel_parity ✓ · Python 18/18
(test_polars_input, test_numpy_sparse_parity, test_smoke). All four crates compile;
maturin release build OK.

## Not done (out of scope, future): O3 (skip redundant f64 cast), O4 (bulk null memcpy),
O5 (skip materialize for single-chunk). Wall-clock perf not benchmarked — change is a
parity-neutral allocation/transpose reduction, not a measured speedup claim.
