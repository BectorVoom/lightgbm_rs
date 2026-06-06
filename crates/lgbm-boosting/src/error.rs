//! Structured domain error types at the `lgbm-boosting` boundary (Security V5,
//! FND-04, threat T-06-01-01/02).
//!
//! Uses `thiserror` derive (CLAUDE.md mandate) — never hand-roll
//! `impl std::error::Error`. The GBDT loop validates its own inputs (e.g.
//! grad/hess/score slice lengths) to typed variants, and WRAPS the three upstream
//! layer errors via `#[from]` — exactly the `lgbm-treelearner::TreeLearnerError`
//! discipline that `#[from]`-wraps `lgbm_compute::ComputeError`. The loop never
//! re-validates what an upstream layer already owns; it only surfaces it.
//!
//! Maps C++ `CHECK`/`Log::Fatal` sites in `src/boosting/gbdt.cpp` to typed
//! `Result` errors — never a panic on caller input.

use thiserror::Error;

use lgbm_metric::MetricError;
use lgbm_objective::ObjectiveError;
use lgbm_treelearner::TreeLearnerError;

/// Errors raised at the `lgbm-boosting` boundary.
///
/// Own boundary checks become struct-style variants; upstream layer failures are
/// surfaced via `#[from]` so a caller can match a single error type for the whole
/// training pipeline.
#[derive(Debug, Error, Clone, PartialEq)]
pub enum BoostingError {
    /// Two boosting input arrays had inconsistent lengths.
    ///
    /// Threat T-06-01-02: e.g. `gradients.len() != hessians.len()`, or a score
    /// buffer that disagrees with `num_data * num_tree_per_iteration`. Validated
    /// before any per-iteration FP work.
    #[error("array length mismatch: expected `{expected}`, got `{actual}`")]
    LengthMismatch {
        /// The required length.
        expected: usize,
        /// The actual length supplied.
        actual: usize,
    },

    /// A tree-learner error propagated up from `lgbm-treelearner`.
    ///
    /// Wrapped via `#[from]` — the loop NEVER re-implements the learner-boundary
    /// checks; it only surfaces them.
    #[error("tree learner error: {0}")]
    TreeLearner(#[from] TreeLearnerError),

    /// An objective error propagated up from `lgbm-objective`.
    #[error("objective error: {0}")]
    Objective(#[from] ObjectiveError),

    /// A metric error propagated up from `lgbm-metric`.
    #[error("metric error: {0}")]
    Metric(#[from] MetricError),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn length_mismatch_displays_values() {
        let e = BoostingError::LengthMismatch {
            expected: 7000,
            actual: 6999,
        };
        let msg = e.to_string();
        assert!(msg.contains("7000") && msg.contains("6999"));
    }

    #[test]
    fn wraps_tree_learner_error_via_from() {
        let tle = TreeLearnerError::LengthMismatch {
            expected: 4,
            actual: 3,
        };
        let e: BoostingError = tle.into();
        assert!(matches!(e, BoostingError::TreeLearner(_)));
        assert!(e.to_string().contains("tree learner error"));
    }

    #[test]
    fn wraps_objective_error_via_from() {
        let oe = ObjectiveError::LengthMismatch {
            expected: 4,
            actual: 3,
        };
        let e: BoostingError = oe.into();
        assert!(matches!(e, BoostingError::Objective(_)));
        assert!(e.to_string().contains("objective error"));
    }

    #[test]
    fn wraps_metric_error_via_from() {
        let me = MetricError::LengthMismatch {
            expected: 4,
            actual: 3,
        };
        let e: BoostingError = me.into();
        assert!(matches!(e, BoostingError::Metric(_)));
        assert!(e.to_string().contains("metric error"));
    }
}
