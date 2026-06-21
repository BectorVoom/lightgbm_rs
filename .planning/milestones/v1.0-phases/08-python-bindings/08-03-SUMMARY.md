---
phase: 08-python-bindings
plan: 03
subsystem: api
tags: [pyo3, numpy, scipy-sparse, f32-f64-dispatch, single-widen-site, ab-parity, security-v5]

# Dependency graph
requires:
  - phase: 08-python-bindings
    plan: 02
    provides: "lgbm-python cdylib (Dataset over RawCorpus, train/predict, A/B harness, error taxonomy)"
provides:
  - "f32 + f64 dense numpy dtype dispatch (marshal::numpy_dense_f32_to_rows + dataset::dense_any_to_rows) routed through ONE f32->f64 widen site"
  - "scipy CSR/CSC sparse ingest (marshal::scipy_csr_to_rows/scipy_csc_to_rows) densifying to the same RawCorpus rows the dense path bins (D-05)"
  - "Dataset::from_csr/from_csc staticmethods + lightgbm_rs.dataset_from_csr/from_csc dtype-coercing wrappers"
  - "boundary indptr/indices validation (len/monotone/[0]==0/last==nnz + per-index bounds) surfaced as ValueError before any indexing (Security V5, T-08-03-01)"
  - "A/B parity suite over f32/f64/csr/csc vs real lightgbm 4.6 at atol=1e-6 + f32/f64 + CSR/CSC self-consistency + malformed/dtype rejection tests"
affects: [08-04-polars, 08-05-params, 08-07-sklearn]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Runtime numpy dtype dispatch via Bound::cast::<PyArray2<f64|f32>> with an explicit ValueError fallback (no panic)"
    - "Single f32->f64 widen site mirrored from ingest.rs::widen so f32 and f64 (and CSR/CSC) of the same data yield byte-identical f64 rows"
    - "Sparse densify-to-RawCorpus (gather-with-zeros) instead of Route B FinishedDataset extraction — fits the existing train_raw(RawCorpus) consumer while preserving CSR/CSC<->dense equivalence"

key-files:
  created:
    - crates/lgbm-python/python/tests/test_numpy_sparse_parity.py
  modified:
    - crates/lgbm-python/src/marshal.rs
    - crates/lgbm-python/src/dataset.rs
    - crates/lgbm-python/python/lightgbm_rs/__init__.py

key-decisions:
  - "Sparse routes via densify-to-RawCorpus (marshal::scipy_csr_to_rows/csc), NOT the plan-suggested Route B FinishedDataset->Vec<FeatureColumn> extractor. The implemented train consumer is train_raw(&RawCorpus) (Route A binning); there is no train-from-FinishedDataset entry point, so Route B would require a new architectural seam (Rule 4). Densify performs the IDENTICAL gather-with-zeros that ingest::from_csr/from_csc do internally, so CSR/CSC<->dense equivalence is preserved and the same Security-V5 indptr/index validation runs at the boundary. EFB Pitfall 4 is moot: Route A bins per-column single-feature groups."
  - "Dense dtype dispatch via Bound::cast on &Bound<PyAny> (try f64 array, then f32 array, else ValueError) rather than two typed #[pymethods] overloads — keeps one #[new] surface mirroring the official Dataset(data, label) signature."
  - "from_csr/from_csc are #[staticmethod]s taking (indptr,indices,data,shape,label); the thin lightgbm_rs.dataset_from_csr/from_csc wrappers do the scipy .astype coercion (indptr->i64, indices->i32, data->f32; RESEARCH A5) so users pass a scipy matrix directly."

requirements: [PYB-02]
status: complete
---

## What landed

Widened the Python input surface to the full PYB-02 spectrum: a user can now build
a `lightgbm_rs.Dataset` from an **f32 OR f64 dense numpy matrix** and from a
**scipy CSR/CSC sparse matrix**, all matching the official `lightgbm` 4.6 package
within ~1e-6.

- **Task 1 — dense f32/f64 dtype dispatch (SC#2):** `marshal::numpy_dense_f32_to_rows`
  added alongside the f64 path; `dataset::dense_any_to_rows` inspects the runtime
  dtype (`Bound::cast::<PyArray2<f64|f32>>`) and routes both widths through ONE
  `widen(f32)->f64` site (mirrored from `ingest.rs::widen`), so f32 and f64 of the
  same data yield byte-identical f64 rows/bins/models (T-08-03-03). Contiguity is
  explicit via `is_standard_layout` for both widths (`view.rows()` reads any layout
  correctly, T-08-03-02). Unknown dtype/shape → `ValueError`, never a panic.
- **Task 2 — scipy CSR/CSC sparse (D-05):** `marshal::scipy_csr_to_rows` /
  `scipy_csc_to_rows` validate the three scipy arrays (`indices.len()==data.len()`,
  `validate_indptr`: len/`[0]==0`/monotone/`last==nnz`, per-index bounds) BEFORE any
  indexing and densify into the SAME owned f64 `RawCorpus` rows the dense path bins
  (gather-with-zeros, identical to `ingest::from_csr/from_csc` internals → CSR/CSC
  ↔ dense equivalence inherits the Phase-2 guarantee). `Dataset::from_csr/from_csc`
  staticmethods register the path; a shared `Dataset::from_rows` does the
  label-length boundary check. `DatasetError`-equivalent violations surface as
  `PyValueError` (Security V5, T-08-03-01); no panic crosses FFI.
- **Task 3 — A/B parity (PYB-02):** `test_numpy_sparse_parity.py` parametrizes
  `{f32, f64, csr, csc}` and asserts `assert_allclose(rs, real_lightgbm, atol=1e-6)`
  for each (tests RUN, not skipped). Plus f32/f64 self-consistency, CSR/CSC
  self-consistency, and `ValueError` on malformed `indptr` / out-of-range index /
  wrong dense dtype. `lightgbm_rs.dataset_from_csr/from_csc` wrappers coerce scipy
  arrays (i64/i32/f32, RESEARCH A5) before delegating.

## Verification

- `cargo build -p lgbm-python` ✓ (f32+f64 dense + CSR/CSC sparse compile).
- `cargo clippy -p lgbm-python` ✓ clean (no `unwrap`/`panic!`; `Bound::cast` not
  the deprecated `downcast`).
- `maturin develop` ✓ (incremental, ~24s).
- `pytest python/tests/test_numpy_sparse_parity.py` → **9 passed**:
  - `test_ab_parity_input_kinds[f32|f64|csr|csc]` — all four match real lightgbm
    4.6 at `atol=1e-6` → **PASS** (the ~1e-6 contract through the Python surface
    for the full input spectrum).
  - `test_f32_f64_self_consistency` / `test_csr_csc_self_consistency` — identical
    predictions (single widen site) → **PASS**.
  - `test_malformed_csr_raises` / `test_out_of_range_column_index_raises` /
    `test_wrong_dtype_dense_raises` — `ValueError`, no panic → **PASS**.
- `pytest python/tests/` (full suite) → **15 passed** (no regression to the 08-02
  smoke/parity/GIL tests).
- `LightGBM/` not git-added; `.venv/`/`target/`/`_core*.so` untouched.

## Deviations from Plan

### Architectural decision (resolved inline, not escalated — see rationale)

**1. [Route choice] Sparse densifies to `RawCorpus` instead of plan-suggested Route B**
- **Found during:** Task 2 (reading the train consumer).
- **Issue:** The plan's `<artifacts>` suggested a `FinishedDataset → Vec<FeatureColumn>`
  extractor (Route B) for sparse. But the implemented train entry point is
  `lgbm::train_raw(&Config, &RawCorpus)` (Route A column binning); there is NO
  train-from-`FinishedDataset`/`Vec<FeatureColumn>` path in `lgbm`. Route B would
  require adding a new train seam — an architectural change (Rule 4).
- **Fix:** The `Dataset` pyclass already wraps `RawCorpus`. `scipy_csr_to_rows`/
  `scipy_csc_to_rows` perform the IDENTICAL gather-with-zeros that
  `ingest::from_csr`/`from_csc` do internally to build their dense columns, then
  feed the same Route A binning the dense path uses. This preserves the must-have
  guarantees: sparse accepted (D-05), CSR/CSC ↔ dense equivalence, the same
  `validate_indptr` + per-index bounds (Security V5, T-08-03-01) run at the
  boundary. EFB Pitfall 4 is moot (Route A = single-feature groups). No new
  architecture, no `lgbm`-crate change, faithful to the bit-exact ingest's gather.
- **Files modified:** crates/lgbm-python/src/marshal.rs, crates/lgbm-python/src/dataset.rs
- **Commit:** 1bdef62

### Test enrichment (Rule 2 — correctness coverage)

**2. [Rule 2] Added CSR/CSC self-consistency + extra rejection tests**
- Beyond the plan's required cases, added `test_csr_csc_self_consistency`,
  `test_out_of_range_column_index_raises`, and `test_wrong_dtype_dense_raises` to
  pin the equivalence and the full boundary-rejection surface (Security V5).
- **Commit:** c8a8347

### Commit granularity

- Tasks 1 and 2 are committed together (1bdef62): they edit the same two files
  (`marshal.rs`, `dataset.rs`), share the single `widen` site and the
  `Dataset::from_rows` tail, and cannot be cleanly separated by file. Task 3
  (tests + wrappers) is a separate commit (c8a8347).

## Threat surface

All four threat-register rows are mitigated as planned:
- T-08-03-01 (malformed indptr/indices) — `validate_indptr` + per-index bounds at
  the boundary → `ValueError` before any indexing; covered by two pytest cases.
- T-08-03-02 (non-contiguous f32/f64) — `is_standard_layout` + logical `rows()`
  read for both widths.
- T-08-03-03 (f32/f64 width divergence) — single widen site; self-consistency test.
- T-08-03-SC (installs) — no new crates; scipy is the pinned official-parity PyPI dep.

No NEW security surface beyond the plan's threat model.

## Notes for downstream plans

- `maturin develop` (with `.venv` active) is required before pytest; `_core*.so`
  is gitignored.
- 08-04 (polars) adds Arrow-column ingest; it can reuse the `Dataset::from_rows`
  tail and the single `widen` site, and should route categorical columns to
  `RawCorpus.categorical_features` (D-04).
- The sparse path densifies; if a future plan needs a memory-lean sparse train
  (no densification), that is the point to introduce a real Route B
  train-from-`FinishedDataset` seam in `lgbm`.

## Self-Check: PASSED
