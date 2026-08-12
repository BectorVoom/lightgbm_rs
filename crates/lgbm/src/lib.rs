//! `lgbm` — the umbrella facade crate: the single public import surface for the
//! whole LightGBM-rs engine (the Phase-8 PyO3 target).
//!
//! Faithful-mirror discipline lives in the layer crates; this crate is the
//! greenfield user-facing seam (CONTEXT D-01). It re-exports the curated public
//! types from each layer so a downstream consumer (and the eventual PyO3 binding)
//! imports from one place. The public builder + `Booster` (bounded by
//! [`lgbm_core::Config`], D-02/D-03) land in 06-02+; this Wave-0 scaffold
//! re-exports only what already compiles.
//!
//! Mirrors the `lgbm-model/src/lib.rs` facade shape: lean `pub use` re-exports of
//! the stable types from each crate.

pub mod booster;
pub mod builder;
pub mod error;
pub mod eval_metric;

// --- public training API (Phase 6) ---
pub use booster::{
    train, train_custom, train_custom_raw_with_metric, train_custom_with_metric, train_raw,
    train_with_valid, Booster, CustomMetricClosure, DenseCorpus, PredictOutput, RawCorpus,
};
pub use booster::{build_feature_columns_from_raw, build_feature_columns_from_raw_with_config};
pub use builder::TrainingBuilder;
pub use eval_metric::{create_metrics, EvalMetric};
pub use error::LgbmError;

// --- core config (the single source of truth, D-02) ---
// `DeviceKind` rides along: it is the typed reading of `Config::device_type` that
// the runtime backend dispatch matches on, so callers selecting a device need it.
pub use lgbm_core::config::DeviceKind;
pub use lgbm_core::Config;

// --- dataset surface ---
pub use lgbm_dataset::{Dataset, DatasetError, FinishedDataset, Metadata};

// --- model / predict surface (Phase 3) ---
pub use lgbm_model::{GbdtModel, ModelError, ObjectiveKind, Tree};

// --- engine error boundaries (Phase 6 Wave-0) ---
pub use lgbm_boosting::BoostingError;
pub use lgbm_metric::MetricError;
pub use lgbm_objective::ObjectiveError;
pub use lgbm_treelearner::TreeLearnerError;
