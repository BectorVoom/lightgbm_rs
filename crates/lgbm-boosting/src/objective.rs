//! The boosting-layer objective dispatch — the training-side objective the GBDT
//! loop drives, mirroring the C++ `objective_function_` pointer (which is
//! `nullptr` for the `custom` pass-through).
//!
//! This thin enum unifies the `lgbm-objective` training-side types
//! ([`Objective`] = regression/regression_l1, [`Binary`], and the
//! [`CustomObjective`] closure) behind the small set of operations the loop calls:
//! `boost_from_score`, `get_gradients`, `is_renew_tree_output`,
//! `renew_leaf_output`, and `boost_from_average_enabled` (forced OFF for custom,
//! mirroring the C++ `obj == null` BoostFromAverage skip — gbdt.cpp:355-372).
//!
//! Faithful-mirror note: this is loop WIRING, not new numerical code — every
//! formula lives in `lgbm-objective` with its own C++ citation. The enum just
//! routes to the right one.

use lgbm_objective::{Binary, CustomObjective, Objective, ObjectiveError};

/// The training-side objective the boosting loop drives.
pub enum BoostObjective<'a> {
    /// A built-in regression objective (`regression` L2/sqrt or `regression_l1`).
    Builtin(Objective),
    /// The `binary` (sigmoid logloss) objective.
    Binary(Binary),
    /// A user-supplied `custom` closure (OBJ-02 / D-04). `boost_from_average` is
    /// forced OFF for this variant (C++ `obj == null`).
    Custom(CustomObjective<'a>),
}

impl<'a> BoostObjective<'a> {
    /// C++ `obj->BoostFromScore(class)`. Returns `0.0` for custom (no built-in
    /// init score — BoostFromAverage is skipped when `obj == null`).
    pub fn boost_from_score(&self, label: &[f32]) -> f64 {
        match self {
            BoostObjective::Builtin(o) => o.boost_from_score(label),
            BoostObjective::Binary(b) => b.boost_from_score(label),
            BoostObjective::Custom(_) => 0.0,
        }
    }

    /// Whether `BoostFromAverage` may run at all for this objective. The C++ loop
    /// skips BoostFromAverage entirely when `objective_function_ == nullptr` (the
    /// custom path) — so this is `false` for [`BoostObjective::Custom`].
    pub fn boost_from_average_enabled(&self) -> bool {
        !matches!(self, BoostObjective::Custom(_))
    }

    /// C++ `obj->GetGradients(score, grad, hess)` (or the custom closure in its
    /// place). `label` is ignored for the custom variant (the closure closes over
    /// whatever dataset metadata it needs, like Python's `fobj`).
    ///
    /// # Errors
    /// [`ObjectiveError::LengthMismatch`] from the underlying objective / a
    /// wrong-length custom return (T-06-03-01).
    pub fn get_gradients(
        &self,
        score: &[f64],
        label: &[f32],
        gradients: &mut [f32],
        hessians: &mut [f32],
    ) -> Result<(), ObjectiveError> {
        match self {
            BoostObjective::Builtin(o) => o.get_gradients(score, label, gradients, hessians),
            BoostObjective::Binary(b) => b.get_gradients(score, label, gradients, hessians),
            BoostObjective::Custom(c) => c.get_gradients(score, gradients, hessians),
        }
    }

    /// C++ `obj->IsRenewTreeOutput()` — true only for `regression_l1`.
    pub fn is_renew_tree_output(&self) -> bool {
        match self {
            BoostObjective::Builtin(o) => o.is_renew_tree_output(),
            BoostObjective::Binary(_) | BoostObjective::Custom(_) => false,
        }
    }

    /// The per-leaf renewal (`regression_l1` median residual). Only called when
    /// [`Self::is_renew_tree_output`] is true.
    pub fn renew_leaf_output(&self, residuals: &[f64]) -> f64 {
        match self {
            BoostObjective::Builtin(o) => o.renew_leaf_output(residuals),
            // Unreachable in practice (guarded by is_renew_tree_output); return the
            // input-derived no-op rather than panic.
            BoostObjective::Binary(_) | BoostObjective::Custom(_) => 0.0,
        }
    }
}
