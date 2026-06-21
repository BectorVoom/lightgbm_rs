# Phase 8: Python Bindings - Context

**Gathered:** 2026-06-07
**Status:** Ready for planning

<domain>
## Phase Boundary

Phase 8 delivers a **Python package mirroring the official `lightgbm` interface**, built with PyO3 + maturin + rust-numpy, layered over the validated Rust `lgbm` facade (Phases 1–7). It is the final v1 phase.

In scope (PYB-01..04 + the four success criteria):
- **PYB-01** — `Booster`/`Dataset`/`train` API mirroring the official package; GIL released (`allow_threads`) around training; owned arrays returned.
- **PYB-02** — NumPy interop (rust-numpy) for dense f32/f64 input + **polars** DataFrame input + scipy CSR/CSC sparse; explicit contiguity/dtype handling.
- **PYB-03** — sklearn-style wrappers `LGBMClassifier`/`LGBMRegressor`/`LGBMRanker` matching official semantics.
- **PYB-04** — Python `custom` objective/metric callbacks + `Booster.refit()` reproducing reference outputs.
- Plus (this discussion): training-callback list protocol, `lgb.cv`, feature importance + plotting, C++-compatible model persistence + pickle.

Out of scope (deferred — see `<deferred>`):
- **Dask / distributed wrapper** — blocked on the v1-deferred distributed/network engine.
- File/Arrow-file/binary-cache *file* ingestion (`ING-01/02/03`) — already v2-deferred at the Rust level. (In-memory numpy/polars/sparse ingest IS in scope and is distinct from file loaders.)
- The numerical contract is unchanged: `f32` end-to-end, ~1e-6 vs C++, CPU f64-fold anchor bit-exact. Python results must match the official Python package for either input width.

</domain>

<decisions>
## Implementation Decisions

### Data input & binning (Area 1)
- **D-01: Bin internally from raw data.** The Python `Dataset`/`train` accept **raw** continuous/categorical feature values (numpy + polars), and the binding bins them internally via the **already-bit-exact** `lgbm-dataset` `BinMapper` + `Dataset::construct`. Pre-binned integer input is NOT the Python contract. This mirrors official lightgbm UX.
- **D-02: The raw→bin→train wiring lives in the Rust `lgbm` facade, not in the Python layer.** Today the facade `train()` (`crates/lgbm/src/booster.rs`) only consumes an **identity-binned** `DenseCorpus`/`FeatureColumn` (bins `0..K-1`); the full `BinMapper`/`Dataset::construct` pipeline exists in `lgbm-dataset` but is **not wired into that path**. Phase 8 closes this "wiring gap" in the Rust facade (raw data → `BinMapper` → `FeatureColumn`s → `train`), so Rust users benefit too and it is oracle-testable in Rust. Python is then a thin wrapper. **This is integration of already-validated binning, not new binning — not scope creep.**
- **D-03: Polars is ingested zero-copy via Arrow** (pyo3-polars / Arrow FFI), consuming polars' Arrow-backed columns directly in Rust — no Python-side numpy round-trip. (polars core is Rust; numpy is the baseline array path.)
- **D-04: dtype auto-routing.** Columns are routed to feature kinds by dtype: polars `Categorical`/`Enum`/string columns → LightGBM **categorical** features (the Phase-7 `TRL-06` categorical splits, end-to-end from a DataFrame); numeric → numeric bins. (Mirrors pandas behavior in official lightgbm. An explicit `categorical_feature` override is a likely companion — planner/researcher to confirm against the official API.)
- **D-05: scipy CSR/CSC sparse input is in v1**, routed through the existing bit-exact CSR/CSC ingest (`crates/lgbm-dataset/src/ingest.rs`, Phase 2). Satisfies PYB-02 "dense AND sparse" fully.

### Parameter interface (Area 2)
- **D-06: params dict is the primary config surface.** Python passes `params={'objective':'binary','num_leaves':31, ...}` exactly like official lightgbm; routed Python dict → `HashMap<String,String>` → `Config::from_params` (`crates/lgbm-core/src/config/set.rs`), which already ports C++ `Config::Set` with the full alias table (`config/alias.rs`), seed derivation, CHECK validation, and the C++ "unknown param → warn, never fatal" semantics. The Rust typed `TrainingBuilder` stays a Rust-only convenience. (kwargs sugar that folds into the same dict is acceptable where sklearn wrappers need it.)
- **D-07: Error on recognized-but-unimplemented params.** Maintain an explicit "recognized by official lightgbm but NOT ported here" set (e.g. `device_type=gpu` via the Python device knob, `linear_tree`, distributed params) and raise a clear Python exception when set — preventing **silent divergence** (a user asking for behavior they won't get). Truly-unknown keys (typos) still just warn, preserving C++ fidelity.
- **D-08: Full Python→string value-coercion layer.** Robustly coerce Python typed values to the strings `from_params` expects, matching C++ parsing: bool → `true`/`false`, int/float (repr matching the C++ parse), and **list/tuple params joined per C++ convention** (`monotone_constraints`, `eval_at`, `label_gain`, `interaction_constraints`, `cegb_*` vectors, etc.).

### API surface & persistence (Area 3)
- **D-09: In scope beyond the locked core** (core = `Booster`/`Dataset`/`train` + sklearn wrappers + custom obj/metric + `refit`):
  - **Training-callback list protocol** — the official `callbacks=[...]` API: `early_stopping()`, `log_evaluation()`, `record_evaluation()`, `reset_parameter()`. (Early stopping + eval history already exist in the Rust facade as params/return data; this is the pluggable callback layer on top. Distinct from custom obj/metric, already locked.)
  - **`lgb.cv`** k-fold cross-validation — pure-Python orchestration over `train()`; no new Rust.
  - **Feature importance + plotting** — `Booster.feature_importance()` (Rust `ADV-07` exists) exposed to Python, plus `plot_importance`/`plot_tree`/`plot_metric` (matplotlib/graphviz as optional deps).
- **D-10: Persistence = full C++-compatible text I/O + pickle.** `save_model(path)` / `model_to_string()` / load via `model_str=` / `model_file=`, all on the **Phase-3 C++-compatible text format** (a C++-trained model loads and predicts identically, and vice-versa). PLUS Python `pickle` support (`__getstate__`/`__setstate__` over the model string) for sklearn-pipeline use.

### Packaging & naming (Area 4)
- **D-11: Distinct import name `lightgbm_rs`** (PyPI distribution `lightgbm-rs`). No collision with the real `lightgbm` package, so both can be installed side-by-side for A/B oracle/parity testing. Class/function **names** still mirror official lightgbm.
- **D-12: New workspace crate `crates/lgbm-python`** (PyO3 `cdylib`) added to the existing Cargo workspace, with a maturin `pyproject.toml`. Any Python-side wrapper code (sklearn API, plotting, `cv`, callbacks) ships as a thin `python/` package alongside the compiled extension. Consistent with the crate-per-responsibility workspace.
- **D-13: Single abi3 (stable-ABI) wheel per platform, broad CPython range** (3.8+/3.9+ — planner/CI to set the floor), matching official lightgbm's broad support. GIL released via `allow_threads` around training/prediction (success-criterion mandate), returning owned arrays.

### Claude's Discretion
- Exact `categorical_feature` override API shape, the precise CPython version floor, and the wheel/CI matrix details (within D-13's abi3 broad-range decision) are left to research/planning.
- Error/exception taxonomy mapping (Rust `LgbmError` → Python exception types; whether to mirror official `LightGBMError`), custom-callback Python↔Rust grad/hess marshalling, and sklearn wrapper semantic depth were not deep-dived — planner to scope against the official package + oracle matrix. (User chose to stop here; these are bounded implementation details, not open vision questions.)

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Phase definition & requirements
- `.planning/ROADMAP.md` — Phase 8 section: goal + the 4 success criteria (PyO3+maturin, numpy interop f32/f64 dense/sparse, sklearn wrappers, custom callbacks + refit).
- `.planning/REQUIREMENTS.md` — `PYB-01`/`PYB-02`/`PYB-03`/`PYB-04` (lines ~111–114, ~214–217).
- `.planning/PROJECT.md` — "Out of Scope" (distributed/MPI deferred beyond v1 — the boundary that defers Dask) + "Bindings: Python must mirror the official `lightgbm` package API surface".
- `.planning/phases/07-parity-completing-variants/07-CONTEXT.md` — records that the Phase-7 Rust API surface was deliberately "shaped so [Python bindings] map 1:1".

### Rust facade & the binning-wiring gap (D-01/D-02)
- `crates/lgbm/src/booster.rs` — `train`/`train_with_valid`/`train_custom`, `Booster` (`predict_row`/`predict_row_raw`/`model`), `DenseCorpus`, `build_feature_columns` (the **identity-binned** path that must be extended to raw→bin).
- `crates/lgbm/src/builder.rs` — `TrainingBuilder` (~70 typed setters; the Rust-side convenience surface).
- `crates/lgbm/src/lib.rs` — re-exports `Dataset`/`FinishedDataset`/`Metadata` from `lgbm-dataset`.
- `crates/lgbm/src/error.rs` — `LgbmError` (source for the Python exception mapping).

### Binning pipeline to wire in (D-01/D-03/D-04/D-05)
- `crates/lgbm-dataset/src/dataset.rs` — `Dataset::construct` / `construct_bundled` / `push_row` / `push_value` / `finish_load` (`FinishedDataset`).
- `crates/lgbm-dataset/src/bin_mapper.rs` — bit-exact `BinMapper` (numeric + categorical), `MissingType`.
- `crates/lgbm-dataset/src/ingest.rs` — dense/CSR/CSC ingest (the sparse path for D-05).
- `crates/lgbm-treelearner` — `FeatureColumn` (the binned input type the boosting/treelearner spine consumes).

### Parameter dict path (D-06/D-07/D-08)
- `crates/lgbm-core/src/config/set.rs` — `Config::from_params(&HashMap<String,String>)` (port of C++ `Config::Set`; unknown-warn semantics).
- `crates/lgbm-core/src/config/alias.rs` — `ALIAS_TABLE` + `resolve_alias` (verbatim C++ `alias_table()`).
- `crates/lgbm-core/src/config/mod.rs` — `Config` struct (canonical param names/defaults) + `scope` module (in-v1 param set — basis for the D-07 "recognized-but-unimplemented" set).

### Persistence (D-10)
- `crates/lgbm-model` — Tree + GBDT model-text I/O (`%g`/`%.17g` formatter), C++-text-format compatible round-trip (Phase 3).

### Official Python package to mirror (READ-ONLY reference — never git-add `LightGBM/`)
- `LightGBM/python-package/lightgbm/basic.py` — low-level `Dataset`/`Booster` ctypes API to mirror.
- `LightGBM/python-package/lightgbm/engine.py` — `train()` + `cv()` semantics.
- `LightGBM/python-package/lightgbm/sklearn.py` — `LGBMClassifier`/`LGBMRegressor`/`LGBMRanker` (PYB-03).
- `LightGBM/python-package/lightgbm/callback.py` — `early_stopping`/`log_evaluation`/`record_evaluation`/`reset_parameter` (D-09 callback protocol).
- `LightGBM/python-package/lightgbm/plotting.py` — `plot_importance`/`plot_tree`/`plot_metric` (D-09).
- `LightGBM/python-package/lightgbm/dask.py` — Dask wrapper (DEFERRED — see `<deferred>`).
- `LightGBM/python-package/pyproject.toml` — official packaging/version-support reference for D-13.

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- **`Config::from_params` + `alias.rs`** — a near-complete params-dict backend already exists; the Python dict surface (D-06) is mostly marshalling Python→`HashMap<String,String>`.
- **Bit-exact `BinMapper` / `Dataset::construct` / CSR-CSC ingest** (`lgbm-dataset`) — the entire raw→binned pipeline is built and validated; D-01/D-02/D-05 are *wiring*, not new algorithms.
- **`lgbm` facade** — `train`/`train_with_valid`/`train_custom`, `Booster.predict_row[_raw]`, `feature_importance` (ADV-07), `refit`/continue (ADV-06), TreeSHAP/`predict_contrib` (PRD-04), pred-early-stop (PRD-05) all exist behind the facade for the Python surface to call.
- **`lgbm-model` text I/O** — C++-compatible save/load for D-10.

### Established Patterns
- **Identity-binned facade contract** — the current `DenseCorpus`/`FeatureColumn` train path assumes bins == raw values. D-02 explicitly extends this to a raw-data path; keep the identity path available for the parity harness.
- **Typed errors, never panic** — facade returns `LgbmError`; the Python layer maps these to exceptions (no panics across the FFI boundary).
- **Oracle-harness discipline** — every prior phase validated against committed C++ goldens via `oracle-harness` + `xtask`; Python parity should reuse this (Python-vs-Rust-vs-C++).

### Integration Points
- New crate `crates/lgbm-python` (PyO3 cdylib) → depends on the `lgbm` facade (and `lgbm-dataset` for direct binning if needed).
- Raw→bin→train extension lands in `crates/lgbm/src/booster.rs` (+ possibly a new constructor on `DenseCorpus`/a raw corpus type) before the Python layer can call it.
- polars/Arrow FFI is the one genuinely new external dependency surface (pyo3-polars / arrow) — researcher to confirm the crate + zero-copy contract.

</code_context>

<specifics>
## Specific Ideas

- "**numpy + polars**" — the user explicitly wants polars DataFrames as a first-class Python input alongside numpy (zero-copy via Arrow), with dtype-driven categorical routing. This is a deliberate modernization over the official package's pandas-centric ingest (pandas may still be supported, but polars is the named target).
- Distinct package name `lightgbm-rs` / import `lightgbm_rs` chosen specifically to enable **side-by-side install with the real `lightgbm`** for parity testing.

</specifics>

<deferred>
## Deferred Ideas

- **Dask / distributed wrapper (`lightgbm.dask`)** — deferred; blocked on the v1-deferred distributed/network (allreduce) engine. A faithful Dask wrapper is not buildable without cross-worker gradient sync. Track for the milestone that brings distributed training into scope. (User explicitly chose "Defer Dask".)
- **File/Arrow-file/binary-cache ingestion (`ING-01/02/03`)** — already v2-deferred at the Rust level; the Python `Dataset(data='file.csv')` path inherits that deferral. In-memory numpy/polars/sparse ingest is the v1 Python surface.
- Error/exception taxonomy, custom-callback marshalling depth, and exact `categorical_feature` override API — not deferred to a future *phase*, but left for research/planning within Phase 8 (see Claude's Discretion).

</deferred>

---

*Phase: 8-Python Bindings*
*Context gathered: 2026-06-07*
