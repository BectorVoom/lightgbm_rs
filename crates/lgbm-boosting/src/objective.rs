//! The boosting-layer objective dispatch — the training-side objective the GBDT
//! loop drives, mirroring the C++ `objective_function_` pointer (which is
//! `nullptr` for the `custom` pass-through).
//!
//! This thin enum unifies the `lgbm-objective` training-side types
//! ([`Objective`] = regression/regression_l1, [`Binary`], the [`CustomObjective`]
//! closure, and the multiclass [`MulticlassSoftmax`] / [`MulticlassOva`]) behind
//! the small set of operations the loop calls: `boost_from_score`,
//! `get_gradients`, `is_renew_tree_output`, `renew_leaf_output`,
//! `boost_from_average_enabled`, `num_model_per_iteration`, and `class_need_train`.
//! `boost_from_average` is forced OFF for custom (mirroring the C++ `obj == null`
//! BoostFromAverage skip — gbdt.cpp:355-372).
//!
//! Faithful-mirror note: this is loop WIRING, not new numerical code — every
//! formula lives in `lgbm-objective` with its own C++ citation. The enum just
//! routes to the right one. The multiclass variants are the ONLY ones with
//! `num_model_per_iteration > 1`; the loop grows that many trees per iteration over
//! the class-major score/grad/hess layout (multiclass_objective.hpp:149/255).

use lgbm_objective::{
    Binary, CustomObjective, MulticlassOva, MulticlassSoftmax, Objective, ObjectiveError,
};

/// The training-side objective the boosting loop drives.
pub enum BoostObjective<'a> {
    /// A built-in regression objective (`regression` L2/sqrt or `regression_l1`).
    Builtin(Objective),
    /// The `binary` (sigmoid logloss) objective.
    Binary(Binary),
    /// A user-supplied `custom` closure (OBJ-02 / D-04). `boost_from_average` is
    /// forced OFF for this variant (C++ `obj == null`).
    Custom(CustomObjective<'a>),
    /// `multiclass` (softmax) — grows `num_class` trees/iter; `get_gradients`
    /// gathers strided across the class-major score buffer.
    Multiclass(MulticlassSoftmax),
    /// `multiclassova` — `num_class` one-vs-all binary objectives, one tree/class.
    MulticlassOva(MulticlassOva),
}

impl<'a> BoostObjective<'a> {
    /// C++ `objective->NumModelPerIteration()` — the number of trees grown per
    /// iteration (and the class-major stride). `num_class` for the multiclass
    /// variants, `1` for the single-output objectives (regression/binary/custom).
    pub fn num_model_per_iteration(&self) -> i32 {
        match self {
            BoostObjective::Builtin(_)
            | BoostObjective::Binary(_)
            | BoostObjective::Custom(_) => 1,
            BoostObjective::Multiclass(m) => m.num_model_per_iteration(),
            BoostObjective::MulticlassOva(o) => o.num_model_per_iteration(),
        }
    }

    /// C++ `obj->BoostFromScore(class_id)`. For single-output objectives the
    /// `class_id` is always 0 and the result is over the whole `label` slice; for
    /// the multiclass variants it is the per-class init (softmax: log-class-prob;
    /// ova: per-class binary logit). Returns `0.0` for custom (no built-in init —
    /// BoostFromAverage is skipped when `obj == null`).
    pub fn boost_from_score(&self, class_id: i32, label: &[f32]) -> f64 {
        match self {
            BoostObjective::Builtin(o) => o.boost_from_score(label),
            BoostObjective::Binary(b) => b.boost_from_score(label),
            BoostObjective::Custom(_) => 0.0,
            BoostObjective::Multiclass(m) => m.boost_from_score(class_id),
            BoostObjective::MulticlassOva(o) => o.boost_from_score(class_id),
        }
    }

    /// C++ `obj->ClassNeedTrain(class_id)`. True for the single-output objectives
    /// when both classes are present (binary) / always (regression/custom); for the
    /// multiclass variants it gates the per-class constant-tree path (Pitfall 6).
    pub fn class_need_train(&self, class_id: i32, label: &[f32]) -> bool {
        match self {
            // Regression always trains; custom always trains (no class concept).
            BoostObjective::Builtin(_) | BoostObjective::Custom(_) => true,
            BoostObjective::Binary(b) => b.class_need_train(label),
            BoostObjective::Multiclass(m) => m.class_need_train(class_id),
            BoostObjective::MulticlassOva(o) => o.class_need_train(class_id),
        }
    }

    /// Whether `BoostFromAverage` may run at all for this objective. The C++ loop
    /// skips BoostFromAverage entirely when `objective_function_ == nullptr` (the
    /// custom path) — so this is `false` for [`BoostObjective::Custom`].
    pub fn boost_from_average_enabled(&self) -> bool {
        !matches!(self, BoostObjective::Custom(_))
    }

    /// C++ `obj->GetGradients(score, grad, hess)` (or the custom closure in its
    /// place). `score`/`gradients`/`hessians` are the WHOLE class-major buffers
    /// (length `num_data * num_model_per_iteration`); the multiclass variants gather
    /// strided across classes internally, while the single-output objectives treat
    /// the buffer as their one class. `label` is ignored for the custom variant (the
    /// closure closes over whatever dataset metadata it needs, like Python's
    /// `fobj`) and for the multiclass variants (labels are captured at construction).
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
            BoostObjective::Multiclass(m) => m.get_gradients(score, gradients, hessians),
            BoostObjective::MulticlassOva(o) => o.get_gradients(score, gradients, hessians),
        }
    }

    /// C++ `obj->IsRenewTreeOutput()` — true only for `regression_l1`.
    pub fn is_renew_tree_output(&self) -> bool {
        match self {
            BoostObjective::Builtin(o) => o.is_renew_tree_output(),
            BoostObjective::Binary(_)
            | BoostObjective::Custom(_)
            | BoostObjective::Multiclass(_)
            | BoostObjective::MulticlassOva(_) => false,
        }
    }

    /// The per-leaf renewal (regression_l1 / quantile / mape). `residuals` are the
    /// leaf rows' `label - score`; `labels` are the SAME rows' (untransformed)
    /// labels in the SAME order (needed for MAPE's `label_weight`). Only called when
    /// [`Self::is_renew_tree_output`] is true.
    pub fn renew_leaf_output(&self, residuals: &[f64], labels: &[f32]) -> f64 {
        match self {
            BoostObjective::Builtin(o) => o.renew_leaf_output(residuals, labels),
            // Unreachable in practice (guarded by is_renew_tree_output); return the
            // input-derived no-op rather than panic.
            BoostObjective::Binary(_)
            | BoostObjective::Custom(_)
            | BoostObjective::Multiclass(_)
            | BoostObjective::MulticlassOva(_) => 0.0,
        }
    }
}
