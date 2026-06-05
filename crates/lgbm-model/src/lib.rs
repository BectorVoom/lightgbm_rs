//! `lgbm-model` — faithful tree-model in-memory representation, model-text I/O,
//! and prediction (Phase 3: tree-model / model-text I/O / predict parity).
//!
//! Faithful 1:1 port of LightGBM's tree-model + serialization + prediction
//! subsystem. The byte-exact-write and predict-parity targets are:
//! - `include/LightGBM/tree.h` + `src/io/tree.cpp` — single decision tree
//!   (`Tree::ToString` section order, `Tree(const char*, size_t*)` parser).
//! - `src/boosting/gbdt_model_text.cpp` — `SaveModelToString` /
//!   `LoadModelFromString` ensemble envelope (`version=v4`, `tree_sizes=`,
//!   `feature_importances:`, `parameters:`).
//! - `src/boosting/gbdt_prediction.cpp` — raw / transformed / leaf-index /
//!   sub-range prediction (D-03 / D-04, PRD-01..PRD-06).
//!
//! This plan (Phase 3 Plan 01) delivers the Wave-0 enabling slice only:
//! - [`format`] — the `%.17g` / `{:g}` printf-faithful float formatters (the
//!   DAT-09 serialization linchpin, built and proven FIRST per RESEARCH).
//! - [`error`] — the [`ModelError`] boundary enum.
//!
//! Later plans add `tree`, `ensemble`, `model_text`, and `predict`; their
//! `pub mod` declarations are added here as they land.

pub mod error;
pub mod format;

pub use error::ModelError;
