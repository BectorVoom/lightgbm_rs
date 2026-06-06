//! Structured domain error types at the `lgbm-metric` boundary (Security V5,
//! FND-04, threat T-06-01-01/02).
//!
//! Uses `thiserror` derive (CLAUDE.md mandate) — never hand-roll
//! `impl std::error::Error`. The C++ metric layer's `CHECK`/`Log::Fatal`
//! input-validation sites (e.g. score/label length agreement) are surfaced here as
//! typed `Result` variants — never a panic on caller input.
//!
//! Mirrors the `lgbm-core::ConfigError` / `lgbm-treelearner::TreeLearnerError`
//! idiom (struct-style variants, `#[error("…{field}…")]`). Later waves fill in the
//! per-metric `eval` bodies that raise these at the boundary before any reduction.

use thiserror::Error;

/// Errors raised at the `lgbm-metric` boundary.
///
/// Each variant corresponds to a C++ metric-layer validation site surfaced as a
/// typed error, so a malformed caller request can never reach an abort or an
/// out-of-bounds reduction.
#[derive(Debug, Error, Clone, PartialEq)]
pub enum MetricError {
    /// Two metric input arrays had inconsistent lengths.
    ///
    /// Threat T-06-01-02: e.g. `scores.len()` disagrees with the dataset's
    /// `num_data` (or `labels.len()`). Validated before any per-row reduction.
    #[error("array length mismatch: expected `{expected}`, got `{actual}`")]
    LengthMismatch {
        /// The required length.
        expected: usize,
        /// The actual length supplied.
        actual: usize,
    },

    /// The requested metric name is not supported in the current scope.
    ///
    /// Mirrors the C++ `Metric::CreateMetric` "Unknown metric type" `Log::Fatal`
    /// site — surfaced as a typed error. 06-02 supports `l2`/`rmse`/`l1`; binary /
    /// multiclass / AUC metrics land in 06-04+.
    #[error("unsupported metric: `{name}`")]
    Unsupported {
        /// The unrecognized / out-of-scope metric name.
        name: String,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn length_mismatch_displays_values() {
        let e = MetricError::LengthMismatch {
            expected: 700,
            actual: 699,
        };
        let msg = e.to_string();
        assert!(msg.contains("700") && msg.contains("699"));
    }
}
