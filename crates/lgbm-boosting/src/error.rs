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

    /// `bagging_by_query = true` was requested in Phase 6 (BST-03, threat
    /// T-06-05-04). The query-grouped draw is an EXPLICIT, decision-backed Phase-7
    /// deferral (06-CONTEXT.md BST-03 scope note) — it ships with the Phase-7
    /// ranking objectives (OBJ-04/05/06), the only objectives it affects. Phase 6
    /// rejects it with this typed error rather than silently falling through to
    /// pos/neg row bagging (which would produce a wrong-but-similar bag).
    #[error(
        "bagging_by_query is deferred to Phase 7 (ships with the ranking objectives); \
         Phase 6 supports pos/neg row bagging only (BST-03 scope note)"
    )]
    BaggingByQueryDeferred,

    /// Early stopping was requested (`early_stopping_round > 0`) but no validation
    /// set was supplied (threat T-06-05-02). The stop decision reads
    /// `score.last()` of each valid set's metric history; with no valid set there is
    /// nothing to evaluate, so the loop returns this typed error rather than
    /// panicking on an empty vector.
    #[error("early_stopping_round > 0 requires at least one validation set")]
    EarlyStoppingWithoutValidSet,

    /// A GOSS config check failed (BST-04, threat T-07-05-01). Mirrors the C++
    /// `GOSSStrategy::ResetSampleConfig` invariants (`goss.hpp:85-86`):
    /// `CHECK_LE(top_rate + other_rate, 1.0)` and
    /// `CHECK(top_rate > 0 && other_rate > 0)`. Surfaced as a typed `Result` so a
    /// caller never silently samples against an out-of-range rate.
    #[error("invalid GOSS config: {what}")]
    GossConfig {
        /// Human-readable description of the violated GOSS invariant.
        what: String,
    },

    /// A config combination that is in scope for the C++ reference but DEFERRED in
    /// this phase is rejected here with an honest, decision-backed typed error
    /// rather than silently producing wrong-but-similar output.
    ///
    /// Phase-6 instance (06-06, Task 2b — typed-reject decision): `regression_l1 +
    /// bagging`. The faithful subset-path median-residual renewal IS implemented and
    /// retained (8330cee), and full-corpus `regression_l1` stays bit-exact; but
    /// `regression_l1` over a bagged SUBSET diverges from C++ in leaf STRUCTURE (the
    /// L1 sign-gradient split-gain is a knife-edge over the bagged subset — e.g. a
    /// 2-vs-3-leaf split-count divergence, `rust:0.0` vs `cpp:11.0` at
    /// `regression_l1_bag1_es0_bfa0` tree 0). No leaf-VALUE renewal can fix a
    /// divergent leaf STRUCTURE, so the combination is typed-rejected in Phase 6 and
    /// deferred to a later phase (see ROADMAP / REQUIREMENTS deferral notes).
    #[error("unsupported config (deferred): {what}")]
    UnsupportedConfig {
        /// Human-readable description of the deferred combination.
        what: String,
    },

    /// A Random Forest (`boosting=rf`) config check failed (BST-06, threat
    /// T-07-07-01). Mirrors the C++ `RF::Init` / `RF::Boosting` invariants
    /// (`rf.hpp:35-40,91-93`), enforced at the TOP of the RF train path BEFORE any
    /// tree grows:
    /// - `RF::Boosting` requires `objective_function_ != nullptr` — RF does NOT
    ///   support a custom objective (it re-derives grad/hess from the built-in
    ///   `BoostFromAverage` init score). The Rust port maps the `custom` objective
    ///   (`BoostObjective::Custom`) to this error.
    /// - `RF::Init` requires mandatory randomization: either active row bagging
    ///   (`bagging_freq > 0 && 0 < bagging_fraction < 1`) OR active feature
    ///   sub-sampling (`0 < feature_fraction < 1`). Without either, every tree
    ///   would be identical and the "forest" would collapse to one tree — C++
    ///   `CHECK`s this; the Rust port surfaces it as this typed error.
    #[error("invalid RF config: {what}")]
    RfConfig {
        /// Human-readable description of the violated RF invariant.
        what: String,
    },
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
