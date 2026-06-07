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
