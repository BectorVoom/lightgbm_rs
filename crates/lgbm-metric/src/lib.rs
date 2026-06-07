//! `lgbm-metric` — the evaluation-metric layer (train/valid loss + metric
//! reduction over scores).
//!
//! Faithful 1:1 port target: `LightGBM/src/metric/` (the header-only
//! `*_metric.hpp` implementations + `metric.cpp` factory). A metric reduces the
//! current scores + dataset metadata to one (or more) `f64` values via
//! `Eval(score, objective) -> vector<double>`, plus a
//! `factor_to_bigger_better()` sign (-1 for losses, +1 for AUC).
//!
//! Prob-space metrics (`binary_logloss`/`binary_error`/`multi_logloss`) call
//! [`lgbm_model::ObjectiveKind::convert_output`] first (RESEARCH
//! binary_metric.hpp:80-83) — this crate does NOT duplicate the transform. The
//! logloss floor uses [`lgbm_core::types::K_EPSILON`] (1e-15f), never a fresh
//! literal.
//!
//! Wave-0 scaffold (06-01): this plan creates the compiling skeleton + the
//! [`error`] boundary only. The enum-dispatch metric factory (mirroring the C++
//! string-keyed `CreateMetric`) and the per-metric formulas (L1/L2/RMSE/
//! binary_logloss/binary_error/AUC/multi_logloss) land in 06-04+.

pub mod binary;
pub mod error;
pub mod multiclass;
pub mod regression;
pub mod xentropy;

pub use binary::BinaryMetric;
pub use error::MetricError;
pub use multiclass::{AucMu, MultiError, MultiLogloss};
pub use regression::{Metric, RegressionMetricParams};
pub use xentropy::XentropyMetric;
