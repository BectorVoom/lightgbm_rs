//! Structured domain error types at the `lgbm-objective` boundary (Security V5,
//! FND-04, threat T-06-01-01/02).
//!
//! Uses `thiserror` derive (CLAUDE.md mandate) — never hand-roll
//! `impl std::error::Error`. The C++ objective layer signals fatal input errors
//! via `Log::Fatal` (e.g. a multiclass label outside `[0, num_class)` at
//! `LightGBM/src/objective/multiclass_objective.hpp:62`); this port maps every
//! such site to a typed `Result` variant — it NEVER panics on caller input.
//!
//! Mirrors the `lgbm-treelearner::TreeLearnerError` idiom (struct-style variants,
//! `#[error("…{field}…")]`). Later waves (06-03..06-05) fill in the per-objective
//! `get_gradients` / `boost_from_score` bodies that raise these at the boundary
//! before any FP work.

use thiserror::Error;

/// Errors raised at the `lgbm-objective` boundary.
///
/// Every variant corresponds to a C++ `CHECK`/`Log::Fatal` site surfaced as a
/// typed error, so a malformed caller request can never reach undefined behaviour
/// or an abort.
#[derive(Debug, Error, Clone, PartialEq)]
pub enum ObjectiveError {
    /// Two objective input arrays had inconsistent lengths.
    ///
    /// Threat T-06-01-02: e.g. `score.len() != gradients.len()` or
    /// `gradients.len() != num_data`. Validated before any per-row gradient write.
    #[error("array length mismatch: expected `{expected}`, got `{actual}`")]
    LengthMismatch {
        /// The required length.
        expected: usize,
        /// The actual length supplied.
        actual: usize,
    },

    /// A multiclass label fell outside the valid `[0, num_class)` range.
    ///
    /// Security V5: C++ `multiclass_objective.hpp:62` does `Log::Fatal` here; this
    /// port returns a typed error instead of aborting/panicking.
    #[error("multiclass label `{label}` out of range (num_class = {num_class})")]
    LabelOutOfRange {
        /// The offending label value.
        label: i32,
        /// The exclusive upper bound (number of classes).
        num_class: i32,
    },

    /// The requested objective name is not supported in the current scope.
    ///
    /// Mirrors the C++ `ObjectiveFunction::CreateObjectiveFunction` `Log::Fatal`
    /// "Unknown objective type" site — surfaced as a typed error rather than an
    /// abort. 06-02 supports only the regression L2/sqrt family; later waves add
    /// the remaining core objectives.
    #[error("unsupported objective: `{name}`")]
    Unsupported {
        /// The unrecognized / out-of-scope objective name.
        name: String,
    },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn length_mismatch_displays_values() {
        let e = ObjectiveError::LengthMismatch {
            expected: 10,
            actual: 9,
        };
        let msg = e.to_string();
        assert!(msg.contains("10") && msg.contains('9'));
    }

    #[test]
    fn label_out_of_range_displays_values() {
        let e = ObjectiveError::LabelOutOfRange {
            label: 5,
            num_class: 3,
        };
        let msg = e.to_string();
        assert!(msg.contains('5') && msg.contains('3'));
    }
}
