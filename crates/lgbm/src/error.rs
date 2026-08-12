//! The facade error boundary ([`LgbmError`]) — the single public error type for
//! the `train` / `predict` / builder entry points (Security V5, FND-04).
//!
//! Uses `thiserror` derive (CLAUDE.md mandate). Wraps each layer's typed error via
//! `#[from]` so a downstream consumer (and the eventual PyO3 binding) matches one
//! error type for the whole pipeline, plus a facade-owned [`InvalidCorpus`] for
//! the public corpus-ingestion boundary (T-06-02-02).
//!
//! [`InvalidCorpus`]: LgbmError::InvalidCorpus

use thiserror::Error;

use lgbm_boosting::BoostingError;
use lgbm_core::error::ConfigError;
use lgbm_metric::MetricError;
use lgbm_model::ModelError;
use lgbm_objective::ObjectiveError;

/// Errors raised at the public `lgbm` facade boundary.
#[derive(Debug, Error)]
pub enum LgbmError {
    /// A config param value / combination failed C++ CHECK validation
    /// (T-06-02-01/03). Wrapped from [`lgbm_core::ConfigError`].
    #[error("config error: {0}")]
    Config(#[from] ConfigError),

    /// The configured objective is unsupported in the current scope. Wrapped from
    /// [`lgbm_objective::ObjectiveError`].
    #[error("objective error: {0}")]
    Objective(#[from] ObjectiveError),

    /// A configured metric is unsupported. Wrapped from
    /// [`lgbm_metric::MetricError`].
    #[error("metric error: {0}")]
    Metric(#[from] MetricError),

    /// The predict-side objective transform could not be parsed. Wrapped from
    /// [`lgbm_model::ModelError`].
    #[error("model error: {0}")]
    Model(#[from] ModelError),

    /// The boosting loop / tree learner failed. Wrapped from
    /// [`lgbm_boosting::BoostingError`].
    #[error("boosting error: {0}")]
    Boosting(#[from] BoostingError),

    /// The training corpus is malformed at the public ingestion boundary
    /// (mismatched dimensions, non-identity-binnable feature values, etc.) —
    /// validated before any FP work (T-06-02-02). Facade-owned (no upstream layer
    /// covers the public-corpus contract).
    #[error("invalid corpus: {detail}")]
    InvalidCorpus {
        /// Human-readable description of the corpus defect.
        detail: String,
    },

    /// An I/O failure at the facade boundary (e.g. writing a model file via
    /// [`Booster::save_model`](crate::Booster::save_model)). Facade-owned so the
    /// caller never panics on a filesystem error.
    #[error("io error: {detail}")]
    Io {
        /// Human-readable description of the I/O failure.
        detail: String,
    },

    /// A user-supplied custom-metric (feval) closure returned an invalid value
    /// (non-finite) on the custom-train path (T-08-01-04). Facade-owned — the
    /// closure boundary is a facade concern (the closure is supplied by the Rust
    /// caller / the eventual Python `feval` marshalling in 08-06). Surfaced as a
    /// typed error so a NaN/inf metric value never silently corrupts the
    /// eval-history / early-stopping decision.
    #[error("custom metric error: {detail}")]
    CustomMetric {
        /// Human-readable description of the custom-metric defect.
        detail: String,
    },

    /// A configured capability cannot be honored by the corpus/facade in use —
    /// e.g. a query-grouped ranking metric (`ndcg`/`map`) evaluated over a corpus
    /// that carries no query/group boundaries. Facade-owned: the metric object
    /// itself is valid, it just cannot run against this input, so the caller gets
    /// a precise typed error instead of a silent wrong value.
    #[error("unsupported in this context: {name}")]
    Unsupported {
        /// Human-readable description of what is unsupported and why.
        name: String,
    },

    /// `device_type` names a device whose compute backend was not compiled into
    /// this build (e.g. `device_type=cuda` on a default CPU-only build).
    ///
    /// Backend selection is a RUNTIME choice keyed on `device_type`, but each GPU
    /// backend is still compiled in behind a cargo feature (`cuda` / `rocm` /
    /// `wgpu`) so a CPU-only build needs no GPU toolchain. Asking for a device
    /// this binary cannot reach is therefore a typed error naming the feature to
    /// rebuild with — NEVER a silent fallback to a different device, which would
    /// train a model the caller did not ask for on hardware they did not choose.
    #[error("unsupported device_type `{device}`: {detail}")]
    UnsupportedDevice {
        /// The requested canonical `device_type` (`cpu` / `gpu` / `cuda`).
        device: String,
        /// What is missing and how to enable it.
        detail: String,
    },

    /// A per-feature constraint/penalty vector (`monotone_constraints`,
    /// `cegb_penalty_feature_coupled`, or `cegb_penalty_feature_lazy`) was
    /// non-empty but its length did not equal `num_features` (T-07-11-02).
    /// Mirrors the C++ length CHECKs — `GBDT::Init` (`gbdt.cpp:58`,
    /// `CHECK_EQ(num_total_features, monotone_constraints.size())`) and
    /// `CostEfficientGradientBoosting::Init`
    /// (`cost_effective_gradient_boosting.hpp:47-60`, `Log::Fatal` "should be the
    /// same size as feature number") — surfaced here as a typed error before any
    /// tree grows rather than silently applying constraints to the wrong features.
    #[error(
        "invalid {param} length: {actual} != num_features {num_features}"
    )]
    InvalidConstraintLength {
        /// The offending config parameter name.
        param: &'static str,
        /// The vector's actual length.
        actual: usize,
        /// The expected length (number of features).
        num_features: usize,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn invalid_corpus_displays_detail() {
        let e = LgbmError::InvalidCorpus {
            detail: "labels length 3 != num_data 4".into(),
        };
        assert!(e.to_string().contains("labels length 3"));
    }

    #[test]
    fn wraps_config_error_via_from() {
        // A from-conversion compiles + routes to the Config variant.
        let _ = |e: ConfigError| -> LgbmError { e.into() };
    }
}
