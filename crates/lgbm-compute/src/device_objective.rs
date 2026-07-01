//! On-device objective support-set + the SC #5 host-fallback discriminator (ODL-05..08).
//!
//! Owning phase: **19** (19-00, Wave-1). Scope locked by **D-02** / **D-06**.
//!
//! ## Why this exists (SC #5)
//! Only eleven of LightGBM's objectives have a CUDA (on-device) grad/hess kernel in
//! the AMD-fork reference (`src/objective/cuda/cuda_{regression,binary,multiclass,
//! rank}_objective.cu`, §5 of `docs/cuda-kernel-design.md`). The remaining objectives
//! — `mape`, `gamma`, `gamma_deviance`, `tweedie`, `cross_entropy`,
//! `cross_entropy_lambda`, and `map`/`rank_map` — have **no device kernel** and MUST
//! fall back to the host path. [`device_objective_supported`] is the single
//! discriminator that answers "can this objective run on-device?" so the boosting
//! seam never silently routes an unsupported objective onto a nonexistent kernel
//! (T-19-00-01). It is a PURE classifier: it does not itself flip any routing — the
//! whole phase stays OFF by default behind `LGBM_CUDA_ON_DEVICE`
//! ([`crate::Backend::on_device_growth_supported`] stays `false`, D-02/D-06).
//!
//! ## The eleven supported (the CUDA §5 kernel roster)
//! - **Regression (§5.1):** L2, L1, Quantile, Huber, Fair, Poisson.
//! - **Binary (§5.2):** BinaryLogloss.
//! - **Multiclass (§5.3):** Softmax; OVA (reuses the §5.2 per-class binary kernel).
//! - **Rank (§5.4):** LambdarankNDCG, RankXENDCG.
//!
//! The objective-name aliases mirror `lgbm_objective`'s `parse` (`regression.rs`,
//! `binary.rs`, `multiclass.rs`, `rank.rs`) and the C++ `objective_function.cpp`
//! factory alias table — the SAME strings the host builder routes on (D-02: the
//! discriminator never forks the objective name, it classifies the parsed string).

/// The eleven CUDA (on-device) objective kinds — the §5 kernel roster
/// (`docs/cuda-kernel-design.md`). Everything NOT expressible as one of these has no
/// device kernel and is host-only (see [`device_objective_supported`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DeviceObjectiveKind {
    /// `regression` / `regression_l2` / `l2` / `mse` … — `RegressionL2loss` (§5.1).
    L2,
    /// `regression_l1` / `l1` / `mae` … — `RegressionL1loss` (§5.1).
    L1,
    /// `quantile` — `RegressionQuantileloss` (§5.1).
    Quantile,
    /// `huber` — `RegressionHuberLoss` (§5.1).
    Huber,
    /// `fair` — `RegressionFairLoss` (§5.1).
    Fair,
    /// `poisson` — `RegressionPoissonLoss` (§5.1).
    Poisson,
    /// `binary` — `BinaryLogloss` (§5.2).
    Binary,
    /// `multiclass` / `softmax` — `MulticlassSoftmax` (§5.3).
    MulticlassSoftmax,
    /// `multiclassova` / `ova` / `ovr` — `MulticlassOVA` (§5.3, reuses §5.2 binary).
    MulticlassOva,
    /// `lambdarank` — `LambdarankNDCG` (§5.4).
    Lambdarank,
    /// `rank_xendcg` / `xendcg` … — `RankXENDCG` (§5.4).
    RankXENDCG,
}

impl DeviceObjectiveKind {
    /// Classify an objective-name alias into a [`DeviceObjectiveKind`], or `None`
    /// when the objective has no on-device kernel (the seven host-only objectives).
    ///
    /// The alias sets mirror `lgbm_objective`'s `parse` and the C++
    /// `objective_function.cpp` factory table. Unknown / host-only names (`mape`,
    /// `gamma`, `gamma_deviance`, `tweedie`, `cross_entropy`, `cross_entropy_lambda`,
    /// `map`/`rank_map`) fall through to `None`.
    #[must_use]
    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            // §5.1 Regression L2 (+ the sqrt/rmse variants — same device kernel).
            "regression" | "regression_l2" | "l2" | "mean_squared_error" | "mse"
            | "l2_root" | "root_mean_squared_error" | "rmse" | "regression_sqrt" => Some(Self::L2),
            // §5.1 Regression L1.
            "regression_l1" | "l1" | "mean_absolute_error" | "mae" => Some(Self::L1),
            // §5.1 quantile / huber / fair / poisson.
            "quantile" => Some(Self::Quantile),
            "huber" => Some(Self::Huber),
            "fair" => Some(Self::Fair),
            "poisson" => Some(Self::Poisson),
            // §5.2 binary.
            "binary" => Some(Self::Binary),
            // §5.3 multiclass softmax / OVA.
            "multiclass" | "softmax" => Some(Self::MulticlassSoftmax),
            "multiclassova" | "multiclass_ova" | "ova" | "ovr" => Some(Self::MulticlassOva),
            // §5.4 rank.
            "lambdarank" => Some(Self::Lambdarank),
            "rank_xendcg" | "xendcg" | "xe_ndcg" | "xe_ndcg_mart" | "xendcg_mart" => {
                Some(Self::RankXENDCG)
            }
            // Host-only: mape, gamma, gamma_deviance, tweedie, cross_entropy,
            // cross_entropy_lambda, map/rank_map, and anything unrecognized.
            _ => None,
        }
    }
}

/// The SC #5 host-fallback discriminator: `true` iff the objective named by `name`
/// has an on-device (CUDA §5) grad/hess kernel and may therefore run on-device.
///
/// Returns `false` for all SEVEN CUDA-unsupported objectives (`mape`,
/// `gamma`/`regression_gamma`, `gamma_deviance`, `tweedie`, `cross_entropy`/`xentropy`,
/// `cross_entropy_lambda`/`xentlambda`, `map`/`rank_map`) and any unknown name, so an
/// unsupported objective can never be silently routed on-device (T-19-00-01). This is
/// a classifier only — it does NOT enable on-device growth (that stays gated behind
/// `LGBM_CUDA_ON_DEVICE` + [`crate::Backend::on_device_growth_supported`], D-02/D-06).
#[must_use]
pub fn device_objective_supported(name: &str) -> bool {
    DeviceObjectiveKind::from_name(name).is_some()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// SC #5: every one of the seven CUDA-unsupported objectives is host-only.
    #[test]
    fn unsupported_objectives_are_rejected() {
        for name in [
            "mape",
            "mean_absolute_percentage_error",
            "gamma",
            "regression_gamma",
            "gamma_deviance",
            "tweedie",
            "cross_entropy",
            "xentropy",
            "cross_entropy_lambda",
            "xentlambda",
            "map",
            "rank_map",
            "mean_average_precision",
        ] {
            assert!(
                !device_objective_supported(name),
                "{name} has no on-device kernel and must fall back to the host path (SC #5)"
            );
            assert!(DeviceObjectiveKind::from_name(name).is_none(), "{name} must classify to None");
        }
    }

    /// SC #5: the eleven CUDA-supported objectives route on-device.
    #[test]
    fn supported_objectives_are_accepted() {
        // The five acceptance-criteria anchors …
        for name in ["regression", "binary", "multiclass", "lambdarank", "rank_xendcg"] {
            assert!(
                device_objective_supported(name),
                "{name} has an on-device §5 kernel and must be supported (SC #5)"
            );
        }
        // … and the full eleven-kind alias roster.
        let expected = [
            ("regression_l1", DeviceObjectiveKind::L1),
            ("l1", DeviceObjectiveKind::L1),
            ("quantile", DeviceObjectiveKind::Quantile),
            ("huber", DeviceObjectiveKind::Huber),
            ("fair", DeviceObjectiveKind::Fair),
            ("poisson", DeviceObjectiveKind::Poisson),
            ("binary", DeviceObjectiveKind::Binary),
            ("softmax", DeviceObjectiveKind::MulticlassSoftmax),
            ("multiclassova", DeviceObjectiveKind::MulticlassOva),
            ("lambdarank", DeviceObjectiveKind::Lambdarank),
            ("rank_xendcg", DeviceObjectiveKind::RankXENDCG),
            ("l2", DeviceObjectiveKind::L2),
        ];
        for (name, kind) in expected {
            assert_eq!(
                DeviceObjectiveKind::from_name(name),
                Some(kind),
                "{name} must classify to {kind:?}"
            );
        }
    }
}
