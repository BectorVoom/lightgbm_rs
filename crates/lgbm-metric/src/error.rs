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

    /// A ranking label was negative or exceeded the `label_gain` table size
    /// (threat T-07-09-02). Mirrors `DCGCalculator::CheckLabel`
    /// (`dcg_calculator.cpp:147-162`): rank labels must be non-negative ints
    /// `< label_gain_.size()`. Surfaced as a typed error before any DCG indexing,
    /// never an out-of-bounds gain-table access.
    #[error("ranking label out of range: label `{label}` not in [0, {num_gains})")]
    LabelOutOfRange {
        /// The offending (truncated-to-int) label value.
        label: i64,
        /// The size of the `label_gain` table (the exclusive upper bound).
        num_gains: usize,
    },

    /// A multiclass label was not a non-negative integer `< num_class` (WR-03).
    /// The C++ multiclass metrics (`multiclass_metric.hpp` `LossOnPoint`) index
    /// `ref_score[static_cast<size_t>(label)]` with NO bounds check, relying on the
    /// objective `Init` having validated the label. Surfaced here as a typed error
    /// so both `multi_logloss` and `multi_error` reject a bad label CONSISTENTLY
    /// (rather than one clamping and the other silently flooring).
    #[error("multiclass label out of range: label `{label}` not a non-negative integer in [0, {num_class})")]
    MulticlassLabelOutOfRange {
        /// The offending label value.
        label: f64,
        /// The number of classes (the exclusive upper bound).
        num_class: usize,
    },

    /// A ranking label was not an integer (threat T-07-09-02, WR-02). Mirrors the
    /// FIRST check in `DCGCalculator::CheckLabel` (`dcg_calculator.cpp:148-153`):
    /// `fabs(label - static_cast<int>(label)) > kEpsilon` fatals because a
    /// fractional rank label would silently truncate to a wrong gain-table index.
    /// Surfaced as a typed error before any DCG indexing.
    #[error("ranking label is not an integer: label `{label}` (gain table is indexed by truncation)")]
    NonIntegerLabel {
        /// The offending fractional label value.
        label: f64,
    },

    /// The supplied `query_boundaries` were malformed (threat T-07-09-01):
    /// non-monotonic, not starting at 0, or not ending at `num_data`. Validated
    /// before per-query iteration so a hostile boundary array can never index
    /// out of bounds or feed a negative-length query to DCG.
    #[error("invalid query_boundaries: {what}")]
    InvalidQueryBoundaries {
        /// Human-readable description of the violated boundary invariant.
        what: String,
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
