# Phase 8: Python Bindings - Research

**Researched:** 2026-06-07
**Domain:** PyO3 Python extension (Rust→Python FFI), maturin abi3 packaging, rust-numpy / pyo3-polars / scipy-sparse interop, sklearn-API mirroring, custom-callback grad/hess marshalling, raw→bin→train facade wiring
**Confidence:** HIGH (crate versions verified on crates.io; PyO3 0.27 API verified vs official migration guide; Rust facade + binning gap verified by reading source; official Python package read directly)

## Summary

Phase 8 wraps the validated Rust `lgbm` facade in a Python package (`lightgbm_rs` import / `lightgbm-rs` dist) that mirrors the official `lightgbm` API. It is mostly **integration + marshalling**, not new algorithms: the binning pipeline (`BinMapper`/`Dataset::construct`/CSR-CSC ingest), the GBDT spine, custom objectives, refit, feature importance, and C++-compatible text I/O all already exist. The two genuinely new Rust surfaces are (1) the **raw→bin→train wiring gap** in `crates/lgbm/src/booster.rs` (D-02) and (2) the **PyO3/maturin extension crate** `crates/lgbm-python` (D-12).

The single most important technical finding is a **crate-version alignment constraint**: `pyo3-polars 0.26.0` (the polars zero-copy path, D-03) requires `pyo3 ^0.27`, but the newest `numpy` (rust-numpy) is `0.28.0` requiring `pyo3 ^0.28`. There is **no pyo3-polars release built against pyo3 0.28** as of this research. The compatible, mutually-consistent set is **`pyo3 = 0.27`, `numpy = 0.27.1`, `pyo3-polars = 0.26.0`**. The plan must pin this triple; mixing 0.28 numpy with 0.26 pyo3-polars will not compile.

The second key finding: the facade `Booster` exposes only `predict_row` / `predict_row_raw` and `model()`. Batch predict, `feature_importance`, `refit`, and `model_to_string`/`model_from_string` exist at the **`lgbm-model` `GbdtModel`/`ensemble` level** but are NOT surfaced as `Booster` methods. The plan must add thin facade methods (Rust-side, oracle-testable) before the Python layer can call them — this is the same "wiring, not new algorithm" pattern as D-02.

**Primary recommendation:** Build vertically (MVP slices). Slice 1 = the Rust-side raw→bin→train bridge (`DenseCorpus`/raw-corpus → `BinMapper` → `FeatureColumn`) landed and oracle-tested in the `lgbm` facade — this unblocks everything Python. Slice 2 = minimal PyO3 extension (`Dataset` from numpy f32/f64 dense, `train`, `Booster.predict`) with GIL released via `Python::detach`, A/B parity-tested against side-by-side real `lightgbm`. Then widen: polars/sparse input, params-dict coercion, sklearn wrappers, callbacks/cv, custom obj/metric + refit, persistence. Pin `pyo3 0.27 / numpy 0.27.1 / pyo3-polars 0.26.0`.

## User Constraints (from CONTEXT.md)

### Locked Decisions
- **D-01** Bin internally from raw data; Python `Dataset`/`train` accept raw continuous/categorical values (numpy + polars), binned internally via the already-bit-exact `lgbm-dataset` `BinMapper` + `Dataset::construct`. Pre-binned integer input is NOT the Python contract.
- **D-02** The raw→bin→train wiring lives in the Rust `lgbm` facade (`crates/lgbm/src/booster.rs`), NOT in Python. Today `train()` consumes only an identity-binned `DenseCorpus`/`FeatureColumn` (bins `0..K-1`); the `BinMapper`/`Dataset::construct` pipeline exists in `lgbm-dataset` but is not wired into that path. Phase 8 closes this gap (raw → `BinMapper` → `FeatureColumn`s → `train`). Integration of validated binning, NOT new binning.
- **D-03** Polars ingested zero-copy via Arrow (pyo3-polars / Arrow FFI) — consume polars' Arrow-backed columns directly in Rust, no numpy round-trip.
- **D-04** dtype auto-routing: polars `Categorical`/`Enum`/string → LightGBM categorical features (Phase-7 TRL-06 categorical splits); numeric → numeric bins. Mirrors pandas behavior in official lightgbm. Explicit `categorical_feature` override likely a companion (confirm against official API).
- **D-05** scipy CSR/CSC sparse input is in v1, routed through existing bit-exact ingest (`crates/lgbm-dataset/src/ingest.rs`). Satisfies PYB-02 "dense AND sparse".
- **D-06** params dict is the primary config surface: Python dict → `HashMap<String,String>` → `Config::from_params` (`crates/lgbm-core/src/config/set.rs`), which already ports C++ `Config::Set` (full alias table, seed derivation, CHECK validation, unknown-param→warn-never-fatal). Typed `TrainingBuilder` stays Rust-only convenience. kwargs sugar folding into the same dict acceptable where sklearn wrappers need it.
- **D-07** Error on recognized-but-unimplemented params: maintain an explicit "recognized by official lightgbm but NOT ported" set (`device_type=gpu` via the Python knob, `linear_tree`, distributed params) and raise a clear Python exception when set — prevent silent divergence. Truly-unknown keys (typos) still just warn (C++ fidelity).
- **D-08** Full Python→string value-coercion layer matching C++ parsing: bool → `true`/`false`, int/float (repr matching C++ parse), list/tuple params joined per C++ convention (`monotone_constraints`, `eval_at`, `label_gain`, `interaction_constraints`, `cegb_*`).
- **D-09** In scope beyond locked core: training-callback list protocol (`early_stopping()`, `log_evaluation()`, `record_evaluation()`, `reset_parameter()`); `lgb.cv` (pure-Python over `train()`); feature importance + plotting (`plot_importance`/`plot_tree`/`plot_metric`, matplotlib/graphviz optional).
- **D-10** Persistence = full C++-compatible text I/O (`save_model`/`model_to_string`/load via `model_str=`/`model_file=`, Phase-3 C++ text format) PLUS Python pickle (`__getstate__`/`__setstate__` over model string).
- **D-11** Import name `lightgbm_rs`, PyPI dist `lightgbm-rs`; side-by-side install with real `lightgbm` for A/B parity. Class/function names still mirror official.
- **D-12** New workspace crate `crates/lgbm-python` (PyO3 `cdylib`) + maturin `pyproject.toml`. Python-side wrapper code (sklearn, plotting, cv, callbacks) ships as a thin `python/` package.
- **D-13** Single abi3 (stable-ABI) wheel per platform, broad CPython range (3.8+/3.9+ — floor TBD). GIL released via `allow_threads` (now `Python::detach`) around training/prediction; return owned arrays.

### Claude's Discretion
- Exact `categorical_feature` override API shape (vs official package).
- Precise CPython version floor + wheel/CI matrix (within D-13 abi3 broad-range).
- Error/exception taxonomy: Rust `LgbmError` → Python exception types (mirror official `LightGBMError`?).
- Custom-callback Python↔Rust grad/hess marshalling (zero-copy numpy in/out).
- sklearn wrapper semantic depth (which official behaviors to replicate).

### Deferred Ideas (OUT OF SCOPE)
- **Dask / distributed wrapper (`lightgbm.dask`)** — blocked on v1-deferred distributed/network engine. Do NOT plan.
- **File / Arrow-file / binary-cache *file* ingestion (`ING-01/02/03`)** — already v2-deferred at Rust level; Python `Dataset(data='file.csv')` inherits the deferral. (In-memory numpy/polars/sparse IS in scope — distinct from file loaders.)

## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| PYB-01 | Python bindings (PyO3 + maturin) mirroring official `Booster`/`Dataset` API; GIL released; owned arrays | `crates/lgbm-python` cdylib (Standard Stack); `Python::detach` GIL pattern (Pattern 2); facade `train`/`Booster` exist; **gap:** facade `Booster` lacks batch `predict`/`feature_importance`/`refit`/text I/O methods (Runtime State Inventory). |
| PYB-02 | NumPy interop (rust-numpy) dense f32/f64 + sparse input + array outputs | `numpy` 0.27.1 `PyReadonlyArray2<f32/f64>` with contiguity coercion (Pattern 3); scipy CSR/CSC → existing `ingest::from_csr/from_csc` (Pattern 4); `IntoPyArray` for owned outputs. |
| PYB-03 | sklearn-style wrappers `LGBMClassifier`/`LGBMRegressor`/`LGBMRanker` | Pure-Python `python/` package over the compiled core (D-12); mirror official `sklearn.py` class hierarchy (Architecture). |
| PYB-04 | Python `custom` objective/metric callbacks + `Booster.refit()` | facade `train_custom` exists (`Fn(&[f64])->(Vec<f32>,Vec<f32>)`); official uses **f32 grad/hess** (verified); marshalling Pattern 5; `GbdtModel::refit_one_tree` exists, needs facade `Booster::refit`. |

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| raw→bin→train wiring | Rust facade (`lgbm`) | — | D-02 explicit; oracle-testable in Rust; Rust users benefit; Python stays thin. |
| numpy dense ingest (f32/f64) | PyO3 boundary (`lgbm-python`) | `lgbm-dataset` ingest | dtype/contiguity is a Python-array concern; binning belongs to `lgbm-dataset`. |
| polars ingest (Arrow FFI) | PyO3 boundary (`lgbm-python`) | `lgbm-dataset` | zero-copy Arrow column extraction is FFI-level; routing to feature-kind per D-04. |
| scipy CSR/CSC ingest | PyO3 boundary | `lgbm-dataset` `ingest.rs` | sparse buffers cross FFI as raw slices into existing bit-exact ingest. |
| params dict → Config | PyO3 boundary (coercion) | `lgbm-core` `from_params` | Python value→string coercion (D-08) is Python-typed; parsing is `lgbm-core`. |
| custom obj/metric callbacks | PyO3 boundary (GIL marshalling) | `lgbm` `train_custom` | grad/hess cross the GIL each iter; the boost math is Rust. |
| sklearn wrappers | Python `python/` package | compiled `lightgbm_rs` core | pure-Python class semantics (sklearn protocol); no Rust. |
| callbacks list / cv / plotting | Python `python/` package | core (eval history) | orchestration + matplotlib/graphviz; no new Rust. |
| persistence (text + pickle) | Rust (text I/O) + Python (pickle) | `lgbm-model` `model_text` | text format is `lgbm-model`; pickle wraps the model string in Python. |
| GIL release | PyO3 boundary | — | `Python::detach` around `train`/`predict` (D-13, SC#1). |

## Standard Stack

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| `pyo3` | **0.27** | Rust↔Python FFI, `#[pyclass]`/`#[pymethods]`/`#[pymodule]`, GIL (`Python::detach`), abi3 features | The de-facto Rust→CPython binding; `numpy`+`pyo3-polars` both build on it. **Pinned to 0.27 for pyo3-polars compatibility** (see Common Pitfalls). [VERIFIED: crates.io — pyo3 0.28.3 newest 2026-04-02; 0.27 is the required floor for pyo3-polars 0.26] |
| `numpy` (rust-numpy) | **0.27.1** | numpy ndarray ↔ Rust: `PyReadonlyArray1/2<f32\|f64>`, `IntoPyArray`/`PyArray::from_*` for owned outputs, dtype/contiguity | The canonical rust-numpy crate. Crate name is **`numpy`**, not `rust-numpy` (the `rust-numpy` name on crates.io is a stale 0.1.0 placeholder). 0.27.1 requires pyo3 0.27 — aligns with the pin. [VERIFIED: crates.io — numpy 0.28.0 newest, but 0.27.1 is the pyo3-0.27-compatible release; updated 2026-02-08] |
| `pyo3-polars` | **0.26.0** | Zero-copy polars DataFrame/Series ↔ Rust via Arrow FFI (`PyDataFrame`, `PySeries`) | D-03's named zero-copy path; bundles `polars-arrow`/`polars-ffi` 0.53. Requires **pyo3 ^0.27** — the binding constraint that pins the whole stack. [VERIFIED: crates.io — pyo3-polars 0.26.0 updated 2026-02-08, deps pyo3 ^0.27, polars ^0.53] |
| `maturin` | **1.13.3** | Build backend: abi3 wheels, mixed Rust/Python (`python-source`), per-platform single wheel | Standard PyO3 packaging tool; native abi3 + mixed-layout support (D-12/D-13). Build-time only (not a runtime dep). [VERIFIED: crates.io / pypi — maturin 1.13.3 updated 2026-05-11] |

### Supporting (Python-side `python/` package — runtime/optional deps, mirror official `lightgbm`)
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| `numpy` (PyPI) | `>=1.17.0` | array I/O on the Python side; required dep | Always (mirrors official). [CITED: LightGBM/python-package/pyproject.toml] |
| `scipy` | any | CSR/CSC sparse input detection (D-05) | Always (mirrors official required dep). [CITED: pyproject.toml] |
| `polars` (PyPI) | recent | DataFrame input (D-03) | Optional extra `[polars]`; the named modern input. [ASSUMED — version floor to confirm against pyo3-polars 0.26 / polars 0.53 ABI] |
| `scikit-learn` | `>=0.24.2` | sklearn wrapper base classes / tags (PYB-03) | Optional extra `[scikit-learn]` (mirrors official). [CITED: pyproject.toml] |
| `matplotlib` | any | `plot_importance`/`plot_metric` (D-09) | Optional extra `[plotting]`. [CITED: pyproject.toml] |
| `graphviz` | any | `plot_tree` (D-09) | Optional extra `[plotting]`. [CITED: pyproject.toml] |
| `pandas` | `>=0.24.0` | pandas DataFrame input (official supports; polars is the named target but pandas parity is cheap) | Optional extra `[pandas]`. [CITED: pyproject.toml] |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| pyo3 0.27 (pinned) | pyo3 0.28.3 (newest) | 0.28 gains free-threaded refinements but **no pyo3-polars 0.28 release exists** → D-03 polars zero-copy breaks. Pin 0.27 until pyo3-polars catches up. |
| pyo3-polars (Arrow FFI) | Convert polars→numpy in Python | Loses D-03 zero-copy + dtype routing; a numpy round-trip copies and erases Categorical/Enum dtype. Rejected by D-03. |
| numpy 0.27.1 | numpy 0.28.0 | 0.28 requires pyo3 0.28 → conflicts with pyo3-polars 0.26 pin. Use 0.27.1. |
| maturin | setuptools-rust | maturin is the PyO3-native standard with first-class abi3 + mixed layout; setuptools-rust is heavier and less idiomatic. |

**Installation (Cargo — `crates/lgbm-python/Cargo.toml`):**
```toml
[dependencies]
pyo3 = { version = "0.27", features = ["extension-module", "abi3-py39"] }
numpy = "0.27"
pyo3-polars = "0.26"
lgbm = { path = "../lgbm" }
lgbm-core = { path = "../lgbm-core" }
lgbm-dataset = { path = "../lgbm-dataset" }   # if direct binning needed

[lib]
crate-type = ["cdylib"]
```

**Version verification (run before locking the plan):**
```bash
cargo search pyo3            # confirm 0.27.x line still present
cargo search pyo3-polars     # confirm latest still requires pyo3 ^0.27 (re-pin if a 0.28 release lands)
cargo search numpy           # confirm 0.27.1 pyo3-0.27-compatible release
```

## Package Legitimacy Audit

External Rust crates added by this phase, verified against **crates.io** (the correct registry):

| Package | Registry | Newest / Pinned | Updated | Source Repo | slopcheck | Disposition |
|---------|----------|-----------------|---------|-------------|-----------|-------------|
| `pyo3` | crates.io | 0.28.3 / **pin 0.27** | 2026-04-02 | github.com/PyO3/pyo3 | not run (correct-ecosystem API verify used) | Approved |
| `numpy` (rust-numpy) | crates.io | 0.28.0 / **pin 0.27.1** | 2026-02-08 | github.com/PyO3/rust-numpy | API-verified | Approved |
| `pyo3-polars` | crates.io | **0.26.0** | 2026-02-08 | github.com/pola-rs/pyo3-polars | API-verified | Approved |
| `maturin` | crates.io/pypi | 1.13.3 | 2026-05-11 | github.com/PyO3/maturin | API-verified | Approved (build-only) |

**Python-side optional deps** (`numpy`, `scipy`, `polars`, `scikit-learn`, `matplotlib`, `graphviz`, `pandas`) are PyPI packages identical to the official `lightgbm` package's own declared deps — verified by reading `LightGBM/python-package/pyproject.toml`. They are NOT crates.

**slopcheck note:** `slopcheck install` defaults to the **crates.io** registry and produced false `[SLOP]` verdicts for the PyPI names (`scipy`/`scikit-learn`/`graphviz` "do not exist on crates.io") — this is an ecosystem mismatch, not a real signal; those names are authoritative PyPI packages cited from the official pyproject.toml. The Rust crates were instead verified directly via the crates.io API (existence, newest version, recent publish date, source repo). **No package removed; none flagged as genuinely suspicious.**

**Packages removed due to slopcheck [SLOP] verdict:** none (verdicts were ecosystem false-positives).
**Packages flagged as suspicious [SUS]:** none genuine. (`matplotlib` "only 38 downloads" was a crates.io reading of a PyPI package name — ignore.)

## Architecture Patterns

### System Architecture Diagram

```text
                         Python user code
                              │
        ┌─────────────────────┼──────────────────────────────┐
        │  python/ package (pure Python, thin)                │
        │   lightgbm_rs.sklearn  : LGBMClassifier/Regressor/  │
        │                          Ranker  (PYB-03)           │
        │   lightgbm_rs.callback : early_stopping/log_eval/   │
        │                          record_eval/reset_param    │
        │   lightgbm_rs.engine   : cv()  (orchestrates train) │
        │   lightgbm_rs.plotting : plot_importance/tree/metric│
        └─────────────────────┬───────────────────────────────┘
                              │  (imports compiled core)
        ┌─────────────────────▼───────────────────────────────┐
        │  lightgbm_rs._core  (compiled cdylib, crates/lgbm-python) │
        │   #[pyclass] Dataset    #[pyclass] Booster           │
        │   #[pyfn] train / cv-helpers                         │
        │                                                      │
        │  INPUT marshalling (Python → Rust):                  │
        │   numpy f32/f64 dense ─PyReadonlyArray2─┐            │
        │   polars DataFrame ──pyo3-polars/Arrow─┤ dtype-route │
        │   scipy CSR/CSC ────raw slices─────────┘ (D-04)      │
        │   params dict ──coerce(D-08)──► HashMap<String,String>│
        │   custom obj/metric ──Py callable (GIL marshal)──┐   │
        │                                                  │   │
        │  ── Python::detach (GIL released, D-13) ────────┐│   │
        └───────────────────────────────────────────────┐││───┘
                                                         ▼▼▼
        ┌──────────────────────────────────────────────────────┐
        │  lgbm facade (crates/lgbm)  ◄── D-02 WIRING GAP HERE   │
        │   raw data → BinMapper → FeatureColumn → train()      │
        │   train / train_with_valid / train_custom            │
        │   Booster: predict(batch)*, feature_importance*,     │
        │            refit*, model_to_string*  (*= TO ADD)     │
        └──────────────────────────────┬───────────────────────┘
            │                │                  │            │
    lgbm-dataset      lgbm-boosting     lgbm-treelearner  lgbm-model
   (BinMapper,         (GBDT loop)      (SerialTree,      (GbdtModel,
    Dataset::construct,                  FeatureColumn)    text I/O,
    ingest CSR/CSC)                                        refit, importance)
            │
        OUTPUT (Rust → Python): owned numpy arrays via IntoPyArray
```

### Recommended Project Structure
```text
crates/lgbm-python/
├── Cargo.toml            # cdylib, pyo3 0.27 abi3-py39, numpy 0.27, pyo3-polars 0.26
├── pyproject.toml        # [build-system] maturin; [tool.maturin] python-source="python"
├── src/
│   ├── lib.rs            # #[pymodule] _core; register Dataset, Booster, train
│   ├── dataset.rs        # #[pyclass] Dataset: numpy/polars/scipy ingest → binning
│   ├── booster.rs        # #[pyclass] Booster: predict/feature_importance/refit/save
│   ├── params.rs         # D-08 Python value → String coercion; D-07 unimplemented set
│   ├── marshal.rs        # dense/sparse/polars → Rust; owned-array outputs
│   ├── callbacks.rs      # custom obj/metric GIL marshalling (Pattern 5)
│   └── error.rs          # LgbmError → PyErr taxonomy (Pattern 6)
└── python/
    └── lightgbm_rs/
        ├── __init__.py   # re-export from _core + sklearn/callback/plotting/cv
        ├── sklearn.py    # LGBMClassifier/Regressor/Ranker  (PYB-03)
        ├── callback.py   # early_stopping/log_evaluation/record_evaluation/reset_parameter
        ├── engine.py     # cv()
        └── plotting.py   # plot_importance/plot_tree/plot_metric
```

### Pattern 1: Raw→bin→train bridge in the Rust facade (D-02, the critical Rust slice)
**What:** Extend the facade so a raw-data corpus is binned via the existing `BinMapper` and converted to the `FeatureColumn`s the treelearner consumes — closing the gap between the identity-binned `DenseCorpus` path and the real binning pipeline. **All `FeatureColumn` fields map 1:1 to `BinMapper` fields** (verified by reading both structs), so the bridge is mechanical, not algorithmic.
**When to use:** Before any Python ingest works on real (non-identity) data.
**Mapping (verified field-by-field):**
```text
FeatureColumn.bins[row]    = BinMapper::value_to_bin(raw_value[row])   // per-row bin
FeatureColumn.num_bin      = BinMapper.num_bin_ as u32
FeatureColumn.offset       = lgbm_treelearner::offset_for_most_freq_bin(most_freq_bin)  // authoritative helper
FeatureColumn.min_bin/max_bin = derived from per-feature bin range (single-feature group)
FeatureColumn.default_bin  = BinMapper.default_bin_
FeatureColumn.most_freq_bin= BinMapper.most_freq_bin_
FeatureColumn.missing_type = BinMapper.missing_type_
FeatureColumn.bin_upper_bound = BinMapper.bin_upper_bound_.clone()
FeatureColumn.real_feature_index = column index
```
**Two viable routes (planner to choose, prefer A for least new surface):**
- **Route A — column-direct:** new `BinMapper::find_bin_from_column` (exists) per raw column → build `FeatureColumn` directly. Skips `FeatureGroup`/EFB bundling (single-feature groups), which is fine for the v1 spine and keeps the bins flat exactly as the treelearner wants.
- **Route B — via FinishedDataset:** call `ingest::from_mat` → `FinishedDataset` (groups of `Bin` trait objects), then **add an extractor** `FinishedDataset → Vec<FeatureColumn>` (read `feature_group(g).bin_data().data(row)` per row, `bin_mapper(sub)` for metadata). Reuses the full validated ingest (incl. CSR/CSC for D-05) but requires a new group→column unbundling step (EFB bundling must be off or unbundled).
**Note:** keep the existing identity-binned `DenseCorpus`/`build_feature_columns` path intact — the parity harness relies on it (CONTEXT code_context).

### Pattern 2: GIL release around training/prediction (D-13, SC#1)
**What:** Wrap the long-running Rust call so the GIL is released; copy Python data into owned Rust buffers *before* releasing, and produce owned numpy outputs after re-acquiring.
**Idiom (PyO3 0.27):** `Python::allow_threads` was **renamed to `Python::detach`** in 0.27 (and `with_gil`→`attach`); old names remain as deprecated aliases. Use the new names.
```rust
// Source: pyo3.rs/v0.27.0/migration  [CITED]
#[pyfn(m)]
fn train(py: Python<'_>, /* PyReadonlyArray2, params dict, ... */) -> PyResult<Booster> {
    // 1. Marshal Python → owned Rust data WHILE holding the GIL
    let x: Vec<f32> = x_arr.as_slice()?.to_vec();   // copy out of numpy
    let cfg = Config::from_params(&coerced_params)?;
    // 2. Release the GIL for the CPU-bound boosting loop
    let booster = py.detach(|| lgbm::train(&cfg, &corpus))   // SC#1 mandate
        .map_err(to_pyerr)?;
    Ok(Booster { inner: booster })
}
```

### Pattern 3: numpy dense f32/f64 with explicit contiguity/dtype (PYB-02, SC#2)
**What:** Accept BOTH widths; handle non-C-contiguous arrays explicitly so results match the official package for either width.
```rust
// numpy 0.27.x  [VERIFIED: crates.io numpy 0.27.1]
fn dataset_from_numpy_f64(arr: PyReadonlyArray2<'_, f64>) -> PyResult<...> {
    let arr = arr.as_array();                 // ndarray view
    // contiguity: if !is_standard_layout, .to_owned() / iterate by (row,col)
    // dtype: a separate #[pymethods] overload (or runtime dtype dispatch) for f32
    // widen f32→f64 at the single widen site to match ingest::from_mat
}
```
- Official package widens custom-callback grad/hess to **f32** and asserts `c_contiguous` (verified in `basic.py.__boost`). Mirror: enforce/repair contiguity, dispatch on dtype, and route through `ingest::from_mat` which already does the single f32→f64 widen.
- **Owned outputs:** return predictions via `arr.into_pyarray(py)` / `PyArray1::from_vec` — never lend Rust-owned slices across the boundary (SC#1 "returning owned arrays").

### Pattern 4: scipy CSR/CSC sparse (D-05, PYB-02)
**What:** scipy sparse matrices expose `.data`, `.indices`, `.indptr` numpy arrays — extract as raw slices and feed the **existing** `ingest::from_csr` / `ingest::from_csc` (signature: `indptr:&[i64], indices:&[i32], values:&[f32], num_rows, num_cols, cfg, metadata`). No new sparse algorithm.
```python
# Python side: detect scipy and pass the three arrays down to _core
import scipy.sparse as sp
if sp.issparse(data):
    csr = data.tocsr()
    _core.dataset_from_csr(csr.indptr.astype('int64'), csr.indices.astype('int32'),
                           csr.data.astype('float32'), *csr.shape, params)
```
Note dtype coercion: `ingest::from_csr` wants `indptr: i64`, `indices: i32`, `values: f32` — coerce in Python or at the boundary.

### Pattern 5: Custom objective / metric callback marshalling (PYB-04, SC#4)
**What:** Each iteration, hand the current raw scores to a Python callable and get grad/hess back across the GIL. The facade already has `train_custom(config, corpus, closure: Fn(&[f64]) -> (Vec<f32>, Vec<f32>))`.
**Key parity fact (verified):** official `__boost` coerces grad/hess to **`np.float32`**, asserts `c_contiguous`, and for multiclass ravels `order="F"` (class-major) with length `num_data * num_class`. The Rust closure already returns `Vec<f32>` — widths match.
```rust
// The Rust closure re-acquires the GIL to call back into Python:
let py_obj = obj_callable.clone_ref(py);
let closure = move |scores: &[f64]| -> (Vec<f32>, Vec<f32>) {
    Python::attach(|py| {                          // re-attach GIL inside the boost loop
        let scores_np = scores.to_vec().into_pyarray(py);   // current raw scores (margin)
        let (grad, hess) = py_obj.call1(py, (preds_like, scores_np))?.extract(py)?;
        // grad/hess come back as numpy f32 → as_slice → to_vec
        (grad_f32, hess_f32)
    })
};
lgbm::train_custom(&cfg, &corpus, closure)
```
**Caveat:** `train_custom` is called under `py.detach` (GIL released) but the closure re-attaches per iter via `Python::attach` — this is the correct nested pattern; document the perf note (per-iter GIL round-trip). `boost_from_average` is forced OFF for custom (facade already does this, mirroring C++ `obj==null`).
**custom metric (`feval`)** mirrors `_EvalFunctionWrapper` in official `sklearn.py`: returns `(eval_name, eval_result, is_higher_better)` — wire into the eval-history loop.

### Pattern 6: Error taxonomy LgbmError → PyErr (Claude's Discretion)
**What:** Map the single facade `LgbmError` to Python exceptions. Official package raises `lightgbm.basic.LightGBMError` for C-API failures + `ValueError`/`TypeError` for input problems.
**Recommendation:** define a `lightgbm_rs.LightGBMError` (mirror the official name for drop-in parity), and map:
- `LgbmError::Config` / `InvalidConstraintLength` / D-07 unimplemented-param → `ValueError`
- `LgbmError::InvalidCorpus` / shape/dtype mismatches → `ValueError`
- `LgbmError::Objective` / `Metric` / `Model` / `Boosting` → `lightgbm_rs.LightGBMError`
**Hard rule (CLAUDE.md):** never panic across FFI. Every `#[pymethods]` returns `PyResult<_>`; convert `LgbmError` via a `From<LgbmError> for PyErr`. A Rust panic across the boundary is a hard violation of "no panics across FFI".

### Anti-Patterns to Avoid
- **Lending Rust-owned slices to Python** (dangling after Rust frees) — always `into_pyarray`/copy (SC#1 "owned arrays").
- **Binning in the Python layer** — D-01/D-02 mandate binning in the Rust facade; Python only marshals.
- **numpy round-trip for polars** — defeats D-03 zero-copy and erases Categorical/Enum dtype (breaks D-04 routing).
- **Mixing pyo3 0.28 numpy with pyo3-polars 0.26** — will not compile (see Pitfall 1).
- **Holding the GIL during the boost loop** — violates SC#1; release via `Python::detach`.
- **Silently accepting `device_type=gpu`/`linear_tree`/distributed params** — D-07 says raise (silent divergence).

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Feature binning | A Python-side binner | `lgbm-dataset` `BinMapper`/`Dataset::construct` (D-02 bridge) | Already bit-exact vs C++; re-implementing breaks the numerical contract. |
| CSR/CSC ingest | New sparse parser | `ingest::from_csr`/`from_csc` | Validated Phase-2 path with Security V5 indptr checks. |
| params parsing + aliases | Python dict→config logic | `Config::from_params` + `alias.rs` | Ports C++ `Config::Set` verbatim (alias table, seed derivation, CHECK, warn-not-fatal). |
| Arrow zero-copy from polars | Manual FFI buffer code | `pyo3-polars` `PyDataFrame`/`PySeries` | Handles the Arrow C-data-interface contract + polars version ABI. |
| numpy ↔ Rust ndarray | ctypes pointer juggling | `numpy` (rust-numpy) `PyReadonlyArray`/`IntoPyArray` | Dtype/contiguity/ownership handled; ctypes is what the official package endures, not a model. |
| abi3 wheel build | Custom build script | `maturin` `abi3-pyXX` features | Single stable-ABI wheel per platform (D-13) for free. |
| model text format | New serializer | `lgbm-model` `model_text::save`/`load` | C++-compatible round-trip already validated (Phase 3, D-10). |
| feature importance / refit | New Rust math | `GbdtModel::feature_importance_*`/`refit_one_tree` (surface as `Booster` methods) | Already validated vs real `lib_lightgbm` (`advanced_parity.rs` ADV-06/07). |

**Key insight:** Phase 8 is ~90% marshalling + ~10% the D-02/facade-method wiring. The numerical engine is done and oracle-locked; the risk is in the FFI boundary (ownership, contiguity, GIL, dtype) and crate-version alignment, not in any algorithm.

## Runtime State Inventory

This phase adds a new crate + new facade methods; it is not a rename/migration. The relevant "what isn't already wired" inventory:

| Category | Items Found | Action Required |
|----------|-------------|------------------|
| Facade `Booster` API gaps | `Booster` exposes only `model()`, `predict_row`, `predict_row_raw`. **Missing as `Booster` methods:** batch `predict` (numpy in/out), `feature_importance` (gain/split), `refit`, `model_to_string`/`model_from_string`/`save_model`. The underlying impls exist at `GbdtModel`/`ensemble.rs` (`predict_raw`, `feature_importance_split_count`/`_gain`, `refit_one_tree`) and `model_text.rs` (`save`/`load`). | Add thin facade `Booster` methods delegating to `GbdtModel` — Rust-side, oracle-testable, BEFORE the Python layer. |
| Identity-bin train path | `train()` consumes identity-binned `DenseCorpus` only; no raw→bin route exists in the facade. | The D-02 bridge (Pattern 1). Keep identity path for the parity harness. |
| Workspace membership | `crates/lgbm-python` not in root `Cargo.toml` `members`; no `pyproject.toml` anywhere. | Add crate + `members` entry + maturin `pyproject.toml`. |
| Custom-metric (`feval`) path | facade `train_custom` covers custom **objective**; custom **metric** eval-history wiring not obviously present at the facade. | Confirm/extend: a custom-metric eval hook in the train loop (mirror `_EvalFunctionWrapper`). |
| Re-exports for Python | `lgbm/src/lib.rs` re-exports `Config`, `Dataset`, `GbdtModel`, etc., but the Python crate needs `Booster` predict/importance/refit/save surfaced. | Ensure new facade methods are `pub`. |

**Nothing found in:** stored databases, OS-registered state, secrets/env vars (no runtime secrets in this phase). Build artifacts: the new wheel + `.egg-info` are created fresh; no stale artifacts to migrate.

## Common Pitfalls

### Pitfall 1: pyo3 / numpy / pyo3-polars version triangle (BLOCKING)
**What goes wrong:** Cargo fails to resolve, or links two incompatible pyo3 ABIs, if you take the newest of each crate independently.
**Why it happens:** `numpy` newest is 0.28 (needs pyo3 0.28); `pyo3-polars` newest is 0.26 (needs pyo3 **^0.27**, no 0.28 build). They are mutually exclusive at the newest tips.
**How to avoid:** Pin the aligned set **`pyo3 = "0.27"`, `numpy = "0.27"`, `pyo3-polars = "0.26"`** in `crates/lgbm-python/Cargo.toml` and add to `Cargo.lock`. Re-check at plan time whether a pyo3-0.28-compatible pyo3-polars has shipped (if so, bump all three together).
**Warning signs:** `error: failed to select a version for pyo3` / two `pyo3` versions in the dep tree / abi3 linker errors.

### Pitfall 2: GIL held during the CPU-bound boost loop
**What goes wrong:** SC#1 fails (no `allow_threads`/`detach`); Python threads can't run during training; CI parity test for GIL-release fails.
**How to avoid:** Marshal all Python data into owned Rust buffers first, then `py.detach(|| lgbm::train(...))`. Custom callbacks re-attach with `Python::attach` per iteration.
**Warning signs:** a deadlock if the in-loop callback tries `with_gil`/`attach` while the outer code still holds it.

### Pitfall 3: Non-contiguous / wrong-dtype numpy input silently mis-binned
**What goes wrong:** A Fortran-ordered or sliced numpy array read as if C-contiguous scrambles rows/cols → wrong bins → parity break.
**How to avoid:** Explicitly check `is_standard_layout` / use `PyReadonlyArray::as_array()` and `.to_owned()` on non-standard layout; dispatch f32 vs f64 explicitly; widen at the single site. (SC#2 mandates "contiguity/dtype handled explicitly".)
**Warning signs:** predictions match for C-contiguous arrays but diverge for `.T` / sliced inputs.

### Pitfall 4: EFB bundling breaks the FeatureColumn extraction (Route B)
**What goes wrong:** If the D-02 bridge goes through `FinishedDataset` and EFB has bundled multiple features into one group, naively reading `bin_data().data(row)` yields bundled (offset) bins, not per-feature bins.
**How to avoid:** Either use Route A (column-direct, single-feature groups), or disable/unbundle EFB for the extraction, mapping group-relative bins back per sub-feature via `feature_to_subfeature` + the group's offsets. (Verify against `feature_group.rs` `new_single` vs bundled `new`.)
**Warning signs:** off-by-offset bins for features that share a group; first feature correct, later features shifted.

### Pitfall 5: Custom obj/metric output shape (multiclass) mismatch
**What goes wrong:** Multiclass grad/hess must be class-major (`order="F"`, length `num_data*num_class`); a row-major or wrong-length return corrupts the boost.
**How to avoid:** Mirror official `__boost`: ravel `order="F"`, validate `len == num_data * num_class`, coerce f32, assert contiguous. The facade `train_custom` already surfaces a wrong-length return as a typed error (T-06-03-01) — propagate that as a Python `ValueError`.
**Warning signs:** binary/regression custom works, multiclass diverges or errors on length.

### Pitfall 6: panic across the FFI boundary
**What goes wrong:** A Rust `unwrap()`/index panic unwinds into CPython → abort/UB. Violates CLAUDE.md "no panics across FFI".
**How to avoid:** Every `#[pymethods]`/`#[pyfn]` returns `PyResult`; convert all `LgbmError` via `From<LgbmError> for PyErr`; no `unwrap`/`expect`/`panic!` in the binding crate. Consider `std::panic::catch_unwind` as a last-resort guard at the boundary.

## Code Examples

### Minimal pymodule + pyclass (slice 1 Python surface)
```rust
// Source: pyo3.rs/v0.27.0 (pymodule/pyclass)  [CITED]
use pyo3::prelude::*;

#[pyclass]
struct Booster { inner: lgbm::Booster }

#[pymethods]
impl Booster {
    fn predict<'py>(&self, py: Python<'py>, x: PyReadonlyArray2<'py, f64>)
        -> PyResult<Bound<'py, PyArray2<f32>>> {
        let preds = py.detach(|| { /* batch predict over rows */ });
        Ok(preds.into_pyarray(py))   // owned array
    }
}

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Booster>()?;
    m.add_class::<Dataset>()?;
    Ok(())
}
```

### maturin pyproject.toml (D-12 mixed layout, D-13 abi3)
```toml
# Source: maturin.rs/config  [CITED]
[build-system]
requires = ["maturin>=1.13,<2.0"]
build-backend = "maturin"

[project]
name = "lightgbm-rs"          # D-11 dist name
requires-python = ">=3.9"     # D-13 floor (confirm 3.8 vs 3.9)
dependencies = ["numpy>=1.17.0", "scipy"]

[project.optional-dependencies]
polars = ["polars"]
scikit-learn = ["scikit-learn>=0.24.2"]
plotting = ["matplotlib", "graphviz"]
pandas = ["pandas>=0.24.0"]

[tool.maturin]
python-source = "python"      # the thin python/ package
module-name = "lightgbm_rs._core"   # D-11 import name; compiled core nested
bindings = "pyo3"
features = ["pyo3/extension-module"]
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| `Python::allow_threads` | `Python::detach` (alias kept) | pyo3 0.27 | Use `detach` for GIL release (D-13); `allow_threads` is deprecated-but-works. |
| `Python::with_gil` | `Python::attach` | pyo3 0.27 | Use `attach` to re-acquire GIL inside callbacks. |
| `prepare_freethreaded_python` | `Python::initialize` | pyo3 0.27 | Relevant only for embedded/test init. |
| `*_bound` constructors (`PyArray::new_bound`) | plain names (`PyArray::new`) | pyo3 0.27 (Bound consolidation) | Drop `_bound` suffixes; `Bound<'py, T>` is the core pointer. |
| `rust-numpy` crate name | `numpy` crate (rust-numpy is a 0.1.0 placeholder) | longstanding | Depend on `numpy = "0.27"`, not `rust-numpy`. |
| GIL-centric mental model | thread-state attach/detach (Python 3.13 free-threading) | pyo3 0.27 | Naming reflects free-threaded builds; abi3 still single-wheel. |

**Deprecated/outdated:** `allow_threads`/`with_gil` names (still functional). The `rust-numpy` crates.io name (stale).

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | `polars` PyPI version floor compatible with pyo3-polars 0.26 / polars 0.53 ABI | Standard Stack | Wrong floor → Arrow ABI mismatch at runtime; confirm against pyo3-polars 0.26 release notes. |
| A2 | Route A (column-direct `find_bin_from_column` → FeatureColumn) is sufficient for the v1 spine without FeatureGroup/EFB | Pattern 1 | If EFB or multi-val groups are required for parity on some corpora, Route B's extractor is needed. Verify against Phase-2 dataset behavior. |
| A3 | Custom-metric (`feval`) eval hook can be added at the facade without new Rust math | Runtime State Inventory | If the eval-history loop is hard-coded to built-in metrics, a small facade extension (not just marshalling) is needed. |
| A4 | CPython floor 3.9 (abi3-py39) | maturin pyproject | If 3.8 support is required (official supports 3.7+), use `abi3-py38`; confirm against D-13 "3.8+/3.9+". |
| A5 | scipy sparse `.indptr`/`.indices` coerce cleanly to i64/i32 expected by `ingest::from_csr` | Pattern 4 | scipy default indptr is int32 on 32-bit builds / int64 on 64-bit; explicit `.astype` needed (noted in pattern). |
| A6 | Mirroring the name `LightGBMError` for the exception is desirable for drop-in parity | Pattern 6 | If users distinguish packages by exception type, a distinct name may be preferred; low risk (Claude's Discretion item). |

## Open Questions (RESOLVED)

1. **`categorical_feature` override API shape (Claude's Discretion / D-04)** — RESOLVED
   - Resolution: mirror the official surface. The binding accepts `categorical_feature='auto' | list[int|str]`; default `'auto'` = dtype auto-routing (D-04), an explicit list (column indices or names) overrides dtype detection for exactly those columns. Index-and-name forms both supported, mirroring official `Dataset.set_categorical_feature` / the `categorical_feature` train kwarg. Precedence: explicit list wins over dtype auto-routing for the listed columns.

2. **Route A vs Route B for the D-02 bridge** — RESOLVED
   - Resolution: Route A (dense, column-direct) for the dense path — build per-column `BinMapper`s and construct `FeatureColumn`s directly (least new surface), planned in 08-01. Route B (sparse via the existing `ingest::from_csr`/`from_csc` → `FinishedDataset` → a shared `FinishedDataset → Vec<FeatureColumn>` extractor) for the sparse path (D-05), planned in 08-03. The sparse extractor is the shared seam that unifies dense+sparse onto the same `Vec<FeatureColumn>` consumer.

3. **CPython floor + CI wheel matrix (Claude's Discretion / D-13)** — RESOLVED
   - Resolution: CPython floor = `abi3-py39` (one abi3 wheel per platform; drops EOL 3.8, supports 3.9–3.13). CI wheel matrix targets linux + macos at minimum (the local-ROCm CI host), built via maturin; windows is additive/deferred. Bindings target the CPU facade, so ROCm is orthogonal to the wheel matrix.

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| Rust / cargo | building the cdylib | ✓ | cargo 1.95.0 (edition 2024, rust 1.95) | — |
| Python 3.x | maturin build + tests | ✓ (pip/pip3 present) | system | — |
| maturin | wheel build | ✗ (not installed) | — | `pip install maturin` (build-time only) |
| real `lightgbm` 4.6 | A/B parity oracle | ✓ (used by oracle-harness vs `lib_lightgbm` 4.6) | 4.6.0 | side-by-side install (D-11) |
| polars (Rust + Py) | D-03 polars path | ✗ (new dep) | 0.53 (Rust) / TBD (Py) | numpy-only path still satisfies PYB-02 if polars deferred within a slice |
| scipy | D-05 sparse | likely (test env) | — | required for sparse tests |
| ROCm GPU (gfx1100) | not needed for Python bindings (CPU path) | ✓ | — | bindings target CPU facade; ROCm orthogonal |

**Missing dependencies with no fallback:** none blocking — `maturin` is a one-line install; it's build-tooling.
**Missing dependencies with fallback:** `polars` is a new dep but its slice can be sequenced after the numpy MVP if needed (PYB-02 dense+sparse is satisfiable without polars; D-03 polars is additive).

## Validation Architecture

Nyquist validation is ENABLED. The Phase-8 discipline reuses the existing `oracle-harness` + `xtask` capture pattern (Python-vs-Rust-vs-C++), extended with **A/B parity against the side-by-side real `lightgbm`** (D-11 is specifically to enable this).

### Test Framework
| Property | Value |
|----------|-------|
| Rust framework | `cargo test` + `crates/oracle-harness/tests/*` (existing comparator `compare_within(ORACLE_TOL)`) |
| Python framework | `pytest` (mirror official `tests/python_package_test/`); add `tests/python/` under `crates/lgbm-python` or repo `tests/` |
| Config file | none yet for Python — **Wave 0** adds `crates/lgbm-python/python/tests/` + a pytest invocation in CI |
| Quick run command | `cargo test -p lgbm` (facade bridge) ; `pytest -x tests/python/test_smoke.py` |
| Full suite command | `cargo test --workspace` + `cargo test -p oracle-harness` + `pytest tests/python/` |

### Phase Requirements → Test Map
| Req | Behavior | Test Type | Automated Command | File Exists? |
|-----|----------|-----------|-------------------|-------------|
| D-02 bridge | raw data → BinMapper → FeatureColumn → train matches identity-bin/C++ goldens | unit + oracle | `cargo test -p lgbm raw_bin_train` ; `cargo test -p oracle-harness` | ❌ Wave 0 (Rust) |
| PYB-01 | train+predict through PyO3 mirrors official; GIL released; owned arrays | integration (A/B) | `pytest tests/python/test_booster_parity.py` | ❌ Wave 0 |
| PYB-01 GIL | `Python::detach` releases GIL during train | unit | `pytest tests/python/test_gil_release.py` (background-thread progresses during train) | ❌ Wave 0 |
| PYB-02 | f32 AND f64 dense + CSR/CSC sparse match official for either width | integration (A/B) | `pytest tests/python/test_numpy_sparse_parity.py -k 'f32 or f64 or csr or csc'` | ❌ Wave 0 |
| PYB-02 polars | polars DataFrame (numeric + Categorical) routes per D-04, matches | integration | `pytest tests/python/test_polars_input.py` | ❌ Wave 0 |
| PYB-03 | LGBMClassifier/Regressor/Ranker semantics match official | integration (A/B) | `pytest tests/python/test_sklearn_parity.py` | ❌ Wave 0 |
| PYB-04 | custom obj/metric reproduce reference; `Booster.refit()` matches | integration (A/B) | `pytest tests/python/test_custom_refit_parity.py` | ❌ Wave 0 |
| D-10 | save_model/model_to_string round-trips with C++ format; pickle works | unit + parity | `pytest tests/python/test_persistence.py` (load C++-trained model, predict-equal) | ❌ Wave 0 |
| D-06/07/08 | params dict coercion; unimplemented-param raises; alias resolution | unit | `pytest tests/python/test_params.py` | ❌ Wave 0 |

### Sampling Rate
- **Per task commit:** `cargo test -p lgbm` (for Rust bridge tasks) or `pytest -x <relevant test file>` (for Python tasks).
- **Per wave merge:** `cargo test --workspace` + `pytest tests/python/`.
- **Phase gate:** full Rust workspace + oracle-harness green AND the Python A/B parity suite green vs side-by-side real `lightgbm` 4.6, within the numerical contract (~1e-6 absolute, CPU f64-fold anchor; bit-exact where the algorithm permits) before `/gsd-verify-work`.

### A/B parity harness shape (D-11 side-by-side)
```python
import lightgbm as ref            # real 4.6
import lightgbm_rs as rs          # this phase
# same data, same params dict, same seed/deterministic settings
m_ref = ref.train(params, ref.Dataset(X, y), num_boost_round=N)
m_rs  = rs.train(params, rs.Dataset(X, y),  num_boost_round=N)
np.testing.assert_allclose(m_rs.predict(Xt), m_ref.predict(Xt), atol=1e-6)
```
Reuse the pinned reference settings from `crates/oracle-harness/tests/fixtures/REFERENCE_MANIFEST.md` (`deterministic=true`, `force_row_wise=true`, `num_threads=1`, fixed seed) so the A/B comparison is on the bit-exact CPU anchor regime.

### Wave 0 Gaps
- [ ] `crates/lgbm-python/` crate + `pyproject.toml` + workspace `members` entry.
- [ ] Rust bridge tests for the D-02 raw→bin→train path (reuse oracle goldens).
- [ ] `tests/python/` pytest tree + CI invocation (pytest, maturin develop install).
- [ ] facade `Booster` method coverage (batch predict / feature_importance / refit / save) — Rust tests before Python.
- [ ] A/B parity fixtures: shared (X, y, params) feeding both `lightgbm` and `lightgbm_rs`.

## Security Domain

`security_enforcement: true`, `security_asvs_level: 1`, `security_block_on: high`. This phase is an FFI boundary consuming untrusted Python buffers — input validation is the dominant concern.

### Applicable ASVS Categories
| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | no | no auth surface (library). |
| V3 Session Management | no | n/a. |
| V4 Access Control | no | n/a. |
| V5 Input Validation | **yes** | Validate ALL Python inputs at the FFI boundary BEFORE indexing: array shapes/dtype/contiguity (Pattern 3); CSR/CSC `indptr`/`indices` bounds (existing `validate_indptr` + `indices[k] < num_cols` in `ingest.rs`, Security V5 / T-02-10); params dict coercion (D-08) rejects malformed values; D-07 rejects unimplemented params with a typed exception. Never index unchecked Python-supplied lengths. |
| V6 Cryptography | no | no crypto in this phase. |

### Known Threat Patterns for {Rust PyO3 FFI + numpy/scipy/polars ingest}
| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| Out-of-bounds via malformed CSR/CSC `indptr`/`indices` | Tampering / DoS | Existing `validate_indptr` + per-index `< num_cols` checks in `ingest.rs`; return `DatasetError`→`ValueError`, never index first. |
| Shape/length mismatch (data.len ≠ rows*cols, labels ≠ num_data) | Tampering | Explicit shape checks (already in `ingest::from_mat`); surface as `ValueError`. |
| Panic-across-FFI from `unwrap`/index → CPython abort/UB | DoS | No panics in the binding crate; all `#[pymethods]` return `PyResult`; `From<LgbmError> for PyErr`; optional `catch_unwind` boundary guard (Pitfall 6 / CLAUDE.md). |
| Non-contiguous/aliased numpy array misread | Tampering (silent corruption) | Explicit contiguity check + copy (Pattern 3); SC#2 mandate. |
| Untrusted model text in `model_from_string`/pickle | Tampering | Parse via the validated `lgbm-model` text loader (typed `ModelError`→exception); document that pickle of a model string is only as trusted as its source (standard pickle caveat). |
| Custom-callback returning wrong-length grad/hess | Tampering / DoS | Length-validate against `num_data*num_class` (Pattern 5); typed error, no buffer over-read. |

## Sources

### Primary (HIGH confidence)
- crates.io API — pyo3 (0.28.3 newest, 0.27 line), numpy/rust-numpy (0.28.0 newest, 0.27.1 pyo3-0.27-compatible), pyo3-polars (0.26.0, deps pyo3 ^0.27 + polars ^0.53), maturin (1.13.3). Versions + publish dates + dep requirements.
- pyo3.rs/v0.27.0/migration — `allow_threads`→`detach`, `with_gil`→`attach`, `prepare_freethreaded_python`→`initialize`, Bound consolidation (`_bound` dropped).
- maturin.rs/config — abi3 conditional features, `python-source` mixed layout, `module-name`.
- Codebase (read directly): `crates/lgbm/src/booster.rs` (train path, DenseCorpus, build_feature_columns, train_custom), `crates/lgbm/src/lib.rs`, `crates/lgbm/src/error.rs`, `crates/lgbm-dataset/src/{dataset,bin_mapper,ingest,feature_group}.rs`, `crates/lgbm-treelearner/src/learner.rs` (FeatureColumn fields), `crates/lgbm-model/src/{ensemble,model_text}.rs` (predict/importance/refit/save), `crates/oracle-harness/tests/*`.
- Official package (read directly): `LightGBM/python-package/{pyproject.toml, lightgbm/basic.py (__boost f32 grad/hess), engine.py (train signature), sklearn.py (estimator hierarchy, _ObjectiveFunctionWrapper/_EvalFunctionWrapper), callback.py (callback protocol + order/before_iteration)}`.

### Secondary (MEDIUM confidence)
- polars 0.53 ABI floor for the PyPI `polars` runtime dep (inferred from pyo3-polars 0.26 deps; confirm exact PyPI floor at plan time).

### Tertiary (LOW confidence)
- None material — all load-bearing claims verified against source or registry.

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — every crate version + dep constraint verified via crates.io API; the version-triangle constraint is the key risk and is concrete.
- Architecture / D-02 bridge: HIGH — FeatureColumn↔BinMapper field mapping verified by reading both structs; the gap is real and mechanical.
- Pitfalls: HIGH — version triangle, GIL idiom (verified vs 0.27 migration), contiguity, panic-across-FFI all grounded.
- Validation: HIGH — reuses the existing committed oracle-harness discipline; A/B side-by-side enabled by D-11.

**Research date:** 2026-06-07
**Valid until:** ~2026-07-07 (re-verify crate versions; specifically watch for a pyo3-0.28-compatible pyo3-polars release that would let the whole stack move to 0.28).
