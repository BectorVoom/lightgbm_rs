//! Structured domain error types at the `lgbm-treelearner` boundary (Security V5,
//! threat T-05-02-01).
//!
//! Uses `thiserror` derive (CLAUDE.md mandate) — never hand-roll
//! `impl std::error::Error`, and never pull the app-layer error crate into this
//! library crate. These variants surface the learner's input-validation contract:
//! g/h slice lengths, bin indices, `num_leaves`, and the `sum_hessian` guard for
//! the `cnt_factor = num_data / sum_hessian` division cross from the (eventual)
//! caller into the learner and become typed `Result` errors instead of panics.
//!
//! Upstream compute-backend failures are WRAPPED via `#[from]
//! lgbm_compute::ComputeError`, never re-validated here — the kernel boundary
//! owns its own checks (CMP-01 / RESEARCH §Security V5).

use thiserror::Error;

/// Errors raised at the `lgbm-treelearner` boundary.
///
/// Mirrors the `lgbm-compute::ComputeError` discipline: every learner input is
/// validated to a typed error before any kernel launch / division, so a
/// malformed caller request can never reach undefined behaviour or a NaN-poisoned
/// `cnt_factor`.
#[derive(Debug, Error, Clone, PartialEq)]
pub enum TreeLearnerError {
    /// Two learner input arrays had inconsistent lengths.
    ///
    /// Threat T-05-02-01: e.g. `gradients.len() != hessians.len()` or
    /// `g.len() != num_data`. Validated before any per-leaf reduction.
    #[error("array length mismatch: expected `{expected}`, got `{actual}`")]
    LengthMismatch {
        /// The required length.
        expected: usize,
        /// The actual length supplied.
        actual: usize,
    },

    /// A leaf's `sum_hessian` was non-positive or NaN.
    ///
    /// Guards the `cnt_factor = num_data / sum_hessian` division (the
    /// `find_best_split_cpu` precedent): the `!(sum_hessian > 0.0)` check catches
    /// both `<= 0.0` AND `NaN` (since any comparison with NaN is false), so the
    /// division can never produce a NaN/inf `cnt_factor`.
    #[error("non-positive or NaN sum_hessian: {value}")]
    InvalidLeafHessian {
        /// The offending hessian sum.
        value: f64,
    },

    /// A per-row bin index fell outside the valid `[0, num_bin)` range before it
    /// could be threaded into a histogram/split op.
    #[error("bin index `{index}` is out of range (num_bin = {num_bin})")]
    BinIndexOutOfRange {
        /// The out-of-range bin value.
        index: u32,
        /// The exclusive upper bound (number of bins).
        num_bin: u32,
    },

    /// `num_leaves` was non-positive (a tree must have at least one leaf).
    #[error("invalid num_leaves: {value} (must be >= 1)")]
    InvalidNumLeaves {
        /// The offending `num_leaves` value.
        value: i32,
    },

    /// A compute-backend error propagated up from `lgbm-compute`.
    ///
    /// Wrapped via `#[from]` — the learner NEVER re-implements the kernel-boundary
    /// checks (`BinIndexOutOfRange`/`LengthMismatch`/etc. on the kernel side stay
    /// owned by `ComputeError`); it only surfaces them.
    #[error("compute backend error: {0}")]
    Compute(#[from] lgbm_compute::ComputeError),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn length_mismatch_displays_values() {
        let e = TreeLearnerError::LengthMismatch {
            expected: 10,
            actual: 9,
        };
        let msg = e.to_string();
        assert!(msg.contains("10") && msg.contains('9'));
    }

    #[test]
    fn invalid_leaf_hessian_displays_value() {
        let e = TreeLearnerError::InvalidLeafHessian { value: -1.5 };
        assert!(e.to_string().contains("-1.5"));
    }

    #[test]
    fn compute_error_wraps_via_from() {
        let ce = lgbm_compute::ComputeError::LengthMismatch {
            expected: 4,
            actual: 3,
        };
        // `#[from]` gives a free `From<ComputeError>` conversion.
        let e: TreeLearnerError = ce.into();
        assert!(matches!(e, TreeLearnerError::Compute(_)));
        assert!(e.to_string().contains("compute backend error"));
    }
}
