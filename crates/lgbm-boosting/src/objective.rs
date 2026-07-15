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

use std::cell::RefCell;

use lgbm_core::random::Random;
use lgbm_objective::{
    Binary, CustomObjective, Lambdarank, MulticlassOva, MulticlassSoftmax, Objective,
    ObjectiveError, RankXendcg, Xentropy,
};
// Phase-31 (31-04, ODS-02x) on-device grad/hess routing. These name lgbm-compute
// TYPES/FUNCTIONS only (the resident-Handle grad/hess launchers + the device buffer
// `Handle`), never a GPU compute runtime — the methods stay generic over `B::Runtime`
// via the re-exported `ComputeClient`, so the CMP-01 no-runtime gate still holds
// (exactly the `score_updater.rs` discipline).
use lgbm_compute::kernels::objective_binary::{
    get_gradients_resident_cached_on as binary_grad_hess_resident_cached_on,
    get_gradients_resident_on as binary_grad_hess_resident_on,
};
use lgbm_compute::kernels::objective_regression::{
    get_gradients_resident_cached_on as regression_grad_hess_resident_cached_on,
    get_gradients_resident_on as regression_grad_hess_resident_on, TAG_L1, TAG_L2,
};
use lgbm_compute::ComputeClientReexport as ComputeClient;
use lgbm_compute::{Backend, ComputeError, Handle};

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
    /// `cross_entropy` / `cross_entropy_lambda` (OBJ-05) — single-output
    /// cross-entropy point loss over labels in `[0, 1]`.
    Xentropy(Xentropy),
    /// `lambdarank` (OBJ-06) — per-query pairwise LambdaRank with NDCG. Single
    /// output (one tree/iter) over the whole corpus; captures `query_boundaries` +
    /// labels at construction. `boost_from_average` is OFF (ranking has no mean
    /// init — C++ `BoostFromScore` is not defined for ranking; NeedAccuratePrediction
    /// is false).
    Lambdarank(Lambdarank),
    /// `rank_xendcg` (OBJ-06) — per-query cross-entropy NDCG. Single output; draws a
    /// per-query gamma via `Random(objective_seed + q)`, advanced across iterations
    /// through the [`RefCell`]-held `rands` (mirroring the C++ mutable `rands_`).
    RankXendcg {
        /// The objective (captures `objective_seed` + `query_boundaries`).
        obj: RankXendcg,
        /// The per-query RNG, built once at construction and advanced on every
        /// `get_gradients` (C++ `mutable std::vector<Random> rands_`).
        rands: RefCell<Vec<Random>>,
    },
}

impl<'a> BoostObjective<'a> {
    /// Construct the [`BoostObjective::RankXendcg`] variant, building its per-query
    /// RNG once (the C++ `Init` `rands_` build). The RNG advances across iterations.
    pub fn rank_xendcg(obj: RankXendcg) -> Self {
        let rands = RefCell::new(obj.make_rands());
        BoostObjective::RankXendcg { obj, rands }
    }

    /// C++ `objective->NumModelPerIteration()` — the number of trees grown per
    /// iteration (and the class-major stride). `num_class` for the multiclass
    /// variants, `1` for the single-output objectives (regression/binary/custom/rank).
    pub fn num_model_per_iteration(&self) -> i32 {
        match self {
            BoostObjective::Builtin(_)
            | BoostObjective::Binary(_)
            | BoostObjective::Custom(_)
            | BoostObjective::Xentropy(_)
            | BoostObjective::Lambdarank(_)
            | BoostObjective::RankXendcg { .. } => 1,
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
            BoostObjective::Xentropy(x) => x.boost_from_score(label),
            BoostObjective::Custom(_) => 0.0,
            // Ranking objectives have no mean/median init score (C++ ranking
            // BoostFromScore is undefined; training starts from 0). boost_from_average
            // is also disabled for them (see boost_from_average_enabled).
            BoostObjective::Lambdarank(_) | BoostObjective::RankXendcg { .. } => 0.0,
            BoostObjective::Multiclass(m) => m.boost_from_score(class_id),
            BoostObjective::MulticlassOva(o) => o.boost_from_score(class_id),
        }
    }

    /// C++ `obj->ClassNeedTrain(class_id)`. True for the single-output objectives
    /// when both classes are present (binary) / always (regression/custom); for the
    /// multiclass variants it gates the per-class constant-tree path (Pitfall 6).
    pub fn class_need_train(&self, class_id: i32, label: &[f32]) -> bool {
        match self {
            // Regression always trains; custom + xentropy always train (no class
            // concept / single output).
            BoostObjective::Builtin(_)
            | BoostObjective::Custom(_)
            | BoostObjective::Xentropy(_)
            | BoostObjective::Lambdarank(_)
            | BoostObjective::RankXendcg { .. } => true,
            BoostObjective::Binary(b) => b.class_need_train(label),
            BoostObjective::Multiclass(m) => m.class_need_train(class_id),
            BoostObjective::MulticlassOva(o) => o.class_need_train(class_id),
        }
    }

    /// Whether `BoostFromAverage` may run at all for this objective. The C++ loop
    /// skips BoostFromAverage entirely when `objective_function_ == nullptr` (the
    /// custom path), and ranking objectives have no mean init (NeedAccuratePrediction
    /// is false; the C++ `BoostFromScore` returns the default 0 for ranking) — so this
    /// is `false` for custom and the two ranking variants.
    pub fn boost_from_average_enabled(&self) -> bool {
        !matches!(
            self,
            BoostObjective::Custom(_)
                | BoostObjective::Lambdarank(_)
                | BoostObjective::RankXendcg { .. }
        )
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
            BoostObjective::Xentropy(x) => x.get_gradients(score, label, gradients, hessians),
            BoostObjective::Custom(c) => c.get_gradients(score, gradients, hessians),
            BoostObjective::Multiclass(m) => m.get_gradients(score, gradients, hessians),
            BoostObjective::MulticlassOva(o) => o.get_gradients(score, gradients, hessians),
            // Lambdarank captures labels + query_boundaries at construction.
            BoostObjective::Lambdarank(l) => l.get_gradients(score, gradients, hessians),
            // rank_xendcg advances its per-query RNG across iterations (the C++
            // mutable rands_); the borrow is released before return.
            BoostObjective::RankXendcg { obj, rands } => {
                let mut rands = rands.borrow_mut();
                obj.get_gradients(score, label, &mut rands, gradients, hessians)
            }
        }
    }

    /// The canonical objective name (the C++ `ObjectiveFunction::GetName()` roster),
    /// used by the Phase-31 (31-04) on-device grad/hess seam to classify the objective
    /// through [`lgbm_compute::device_objective_supported`]. This is a pure name
    /// accessor — it never forks routing (D-02).
    #[must_use]
    pub fn name(&self) -> &'static str {
        match self {
            BoostObjective::Builtin(o) => o.name(),
            BoostObjective::Binary(_) => "binary",
            BoostObjective::Custom(_) => "custom",
            BoostObjective::Multiclass(_) => "multiclass",
            BoostObjective::MulticlassOva(_) => "multiclassova",
            BoostObjective::Xentropy(x) => x.name(),
            BoostObjective::Lambdarank(_) => "lambdarank",
            BoostObjective::RankXendcg { .. } => "rank_xendcg",
        }
    }

    /// Phase-31 (31-04, ODS-02x) — the on-device grad/hess router: compute this
    /// iteration's f32 grad/hess ON DEVICE directly from an already-resident `[num_data]`
    /// f64 score `Handle` (the GBDT-owned train-lifetime `ResidentScore`), returning the
    /// two device grad/hess Handles WITHOUT any host score snapshot — the closing half of
    /// 31-RESEARCH.md's Pitfall-2 host round-trip. Delegates to Plan-02's
    /// `get_gradients_resident_on` launchers (bit-exact vs the host-slice `get_gradients_on`,
    /// itself bit-exact vs THIS host objective's `get_gradients` on the cpu-f64 anchor).
    ///
    /// Returns:
    /// - `Some(Ok((grad, hess)))` for the objectives whose device grad/hess is BIT-EXACT
    ///   to the host path — `regression` (L2, incl. sqrt), `regression_l1` (L1), and
    ///   `binary` (BinaryLogloss). These are exactly the roster whose per-row grad/hess
    ///   carries no accumulation (`objective_parity_{regression,binary}` prove
    ///   `compare_exact_u32`), so the resulting model is bit-identical to the host arm.
    /// - `None` for every other objective (multiclass/rank are excluded upstream by the
    ///   `num_class==1` guard; huber/fair/quantile/poisson have an on-device kernel but
    ///   are only within-`ORACLE_TOL` of the host path, NOT bit-exact — routing them
    ///   would break the bit-exact contract, so they fall back to the host grad/hess).
    ///
    /// The GBDT loop passes UNWEIGHTED labels here (the boosting loop's `get_gradients`
    /// takes no per-row weights — regression/binary run the balanced/unweighted default),
    /// matching the host objective it must reproduce bit-for-bit.
    ///
    /// # Errors
    /// Propagates [`ComputeError`] from the launcher (a `num_data`/`labels` length
    /// disagreement) inside the `Some`.
    pub fn get_gradients_resident_on<B: Backend>(
        &self,
        client: &ComputeClient<B::Runtime>,
        resident_score: &lgbm_compute::ResidentScore<B::Runtime>,
        num_data: usize,
        labels: &[f32],
    ) -> Option<Result<(Handle, Handle), ComputeError>> {
        // Resolve the per-train GradResidency FIRST (labels uploaded once, grad/hess
        // outputs reused across iterations — see `ResidentScore::grad_residency`), but
        // only for the bit-exact roster below; every other objective must keep
        // returning `None` WITHOUT touching the residency.
        let supported = matches!(
            self,
            BoostObjective::Builtin(Objective::Regression { .. })
                | BoostObjective::Builtin(Objective::RegressionL1)
                | BoostObjective::Binary(_)
        );
        if !supported {
            // Every other objective (incl. the within-tol-only huber/fair/quantile/
            // poisson and the num_class>1 multiclass/rank, which the caller's guard
            // already excludes) falls back to the host grad/hess path.
            return None;
        }
        let scores_handle = resident_score.score_handle();
        // A/B escape hatch (`LGBM_GRAD_RESIDENCY=0`): the pre-residency launchers —
        // per-iteration label convert/upload + fresh grad/hess device allocs.
        // Bit-exact either way (same kernel + launch geometry).
        if !lgbm_compute::grad_residency_enabled() {
            return match self {
                BoostObjective::Builtin(Objective::Regression { .. }) => Some(
                    regression_grad_hess_resident_on::<B::Runtime>(
                        client, TAG_L2, scores_handle, num_data, labels, None, 0.0,
                    ),
                ),
                BoostObjective::Builtin(Objective::RegressionL1) => Some(
                    regression_grad_hess_resident_on::<B::Runtime>(
                        client, TAG_L1, scores_handle, num_data, labels, None, 0.0,
                    ),
                ),
                BoostObjective::Binary(b) => Some(binary_grad_hess_resident_on::<B::Runtime>(
                    client,
                    scores_handle,
                    num_data,
                    labels,
                    None,
                    b.sigmoid,
                    None,
                )),
                _ => unreachable!("gated by `supported` above"),
            };
        }
        let residency = match resident_score.grad_residency(client, labels) {
            Ok(r) => r,
            Err(e) => return Some(Err(e)),
        };
        match self {
            BoostObjective::Builtin(Objective::Regression { .. }) => Some(
                regression_grad_hess_resident_cached_on::<B::Runtime>(
                    client, TAG_L2, scores_handle, num_data, residency, 0.0,
                ),
            ),
            BoostObjective::Builtin(Objective::RegressionL1) => Some(
                regression_grad_hess_resident_cached_on::<B::Runtime>(
                    client, TAG_L1, scores_handle, num_data, residency, 0.0,
                ),
            ),
            BoostObjective::Binary(b) => Some(binary_grad_hess_resident_cached_on::<B::Runtime>(
                client,
                scores_handle,
                num_data,
                residency,
                b.sigmoid,
                None,
            )),
            _ => unreachable!("gated by `supported` above"),
        }
    }

    /// C++ `obj->IsRenewTreeOutput()` — true only for `regression_l1`.
    pub fn is_renew_tree_output(&self) -> bool {
        match self {
            BoostObjective::Builtin(o) => o.is_renew_tree_output(),
            BoostObjective::Binary(_)
            | BoostObjective::Custom(_)
            | BoostObjective::Multiclass(_)
            | BoostObjective::MulticlassOva(_)
            | BoostObjective::Xentropy(_)
            | BoostObjective::Lambdarank(_)
            | BoostObjective::RankXendcg { .. } => false,
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
            | BoostObjective::MulticlassOva(_)
            | BoostObjective::Xentropy(_)
            | BoostObjective::Lambdarank(_)
            | BoostObjective::RankXendcg { .. } => 0.0,
        }
    }
}
