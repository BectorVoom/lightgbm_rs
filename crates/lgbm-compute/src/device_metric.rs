//! On-device metric support-set + the host-fallback discriminator (ODL-17, D-05).
//!
//! Owning phase: **20** (20-00, Wave-0). Scope locked by **D-05**.
//!
//! ## Why this exists
//! Only the **twelve pointwise (regression/binary) losses** have an on-device
//! `EvalKernel` in the AMD-fork reference (§12 of `docs/cuda-kernel-design.md`). Every
//! other metric — the ranking/AUC/multiclass/cross-entropy families — has **no device
//! kernel** and MUST fall back to the host path. [`metric_supported`] is the single
//! discriminator that answers "can this metric run on-device?" so the boosting seam
//! never silently routes an unsupported metric onto a nonexistent kernel
//! (T-20-00-02). It is a PURE classifier: it does not itself flip any routing — the
//! whole phase stays OFF by default behind `LGBM_CUDA_ON_DEVICE`.
//!
//! ## The twelve supported (the §12 EvalKernel roster)
//! RMSE, L2, L1, Quantile, Huber, Fair, Poisson, MAPE, Gamma, Gamma-deviance,
//! Tweedie, Binary-logloss.
//!
//! ## CRITICAL asymmetry vs [`crate::device_objective`] (D-05)
//! This is a SEPARATE list from [`device_objective_supported`](crate::device_objective::device_objective_supported)
//! and MUST NOT delegate to it. **MAPE, Gamma, Gamma-deviance, and Tweedie are
//! metric-supported here even though their OBJECTIVES have no on-device grad/hess
//! kernel** (`device_objective.rs` classifies those four as host-only). The metric is a
//! pure evaluation function (score + label → number); its on-device availability is
//! independent of whether the matching objective can train on-device. Keying the
//! metric discriminator off the objective-support list would wrongly force those four
//! metrics onto the host even when the score they evaluate was produced on-device.
//!
//! The metric-name aliases mirror the C++ `Config` metric alias table
//! (`ParseMetrics`) — the SAME strings the host builder routes on.

/// The twelve on-device (CUDA §12 `EvalKernel`) pointwise metric kinds. Everything NOT
/// expressible as one of these has no device kernel and is host-only (see
/// [`metric_supported`]).
///
/// This is a support/eval-kernel classifier ONLY. Several distinct metric-name aliases
/// collapse to one kind because they share an eval kernel (e.g. `l2` / `mse` /
/// `regression` all map to [`Self::L2`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DeviceMetricKind {
    /// `l2` / `mean_squared_error` / `mse` / `regression_l2` / `regression` — the
    /// mean-squared-error `EvalKernel` (§12.1).
    L2,
    /// `rmse` / `root_mean_squared_error` / `l2_root` — the √MSE `EvalKernel` (§12.1).
    Rmse,
    /// `l1` / `mean_absolute_error` / `mae` / `regression_l1` — the L1 `EvalKernel` (§12.1).
    L1,
    /// `quantile` — the pinball-loss `EvalKernel` (§12.1).
    Quantile,
    /// `huber` — the Huber `EvalKernel` (§12.1).
    Huber,
    /// `fair` — the Fair-loss `EvalKernel` (§12.1).
    Fair,
    /// `poisson` — the Poisson-deviance `EvalKernel` (§12.1).
    Poisson,
    /// `mape` / `mean_absolute_percentage_error` — the MAPE `EvalKernel` (§12.1).
    /// **Metric-supported even though the `mape` OBJECTIVE is host-only (D-05).**
    Mape,
    /// `gamma` — the Gamma-deviance `EvalKernel` (§12.1).
    /// **Metric-supported even though the `gamma` OBJECTIVE is host-only (D-05).**
    Gamma,
    /// `gamma_deviance` — the Gamma-deviance-residual `EvalKernel` (§12.1).
    /// **Metric-supported even though the OBJECTIVE is host-only (D-05).**
    GammaDeviance,
    /// `tweedie` — the Tweedie-deviance `EvalKernel` (§12.1).
    /// **Metric-supported even though the `tweedie` OBJECTIVE is host-only (D-05).**
    Tweedie,
    /// `binary_logloss` / `binary` — the binary-logloss `EvalKernel` (§12.2).
    BinaryLogloss,
}

impl DeviceMetricKind {
    /// Classify a metric-name alias into a [`DeviceMetricKind`], or `None` when the
    /// metric has no on-device `EvalKernel` (the ranking / AUC / multiclass /
    /// cross-entropy families).
    ///
    /// The alias sets mirror the C++ `Config` metric alias table (`ParseMetrics`).
    /// Unknown / host-only names (`auc`, `auc_mu`, `ndcg`, `map`, `average_precision`,
    /// `multi_error`, `multi_logloss`, `cross_entropy`, `cross_entropy_lambda`,
    /// `kullback_leibler`, and anything unrecognized) fall through to `None`.
    #[must_use]
    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            // §12.1 mean-squared-error (+ the `regression` metric alias).
            "l2" | "mean_squared_error" | "mse" | "regression_l2" | "regression" => Some(Self::L2),
            // §12.1 root-mean-squared-error.
            "rmse" | "root_mean_squared_error" | "l2_root" => Some(Self::Rmse),
            // §12.1 mean-absolute-error.
            "l1" | "mean_absolute_error" | "mae" | "regression_l1" => Some(Self::L1),
            // §12.1 quantile / huber / fair / poisson.
            "quantile" => Some(Self::Quantile),
            "huber" => Some(Self::Huber),
            "fair" => Some(Self::Fair),
            "poisson" => Some(Self::Poisson),
            // §12.1 mape / gamma / gamma_deviance / tweedie — METRIC-supported even
            // though the matching OBJECTIVES are host-only (D-05 asymmetry).
            "mape" | "mean_absolute_percentage_error" => Some(Self::Mape),
            "gamma" => Some(Self::Gamma),
            "gamma_deviance" => Some(Self::GammaDeviance),
            "tweedie" => Some(Self::Tweedie),
            // §12.2 binary logloss.
            "binary_logloss" | "binary" => Some(Self::BinaryLogloss),
            // Host-only: auc, auc_mu, ndcg, map/average_precision, multi_error,
            // multi_logloss, cross_entropy/xentropy, cross_entropy_lambda/xentlambda,
            // kullback_leibler, and anything unrecognized.
            _ => None,
        }
    }
}

/// The ODL-17 host-fallback discriminator: `true` iff the metric named by `name` has
/// an on-device (CUDA §12) `EvalKernel` and may therefore be evaluated on-device.
///
/// Returns `false` for every non-pointwise metric (`auc`, `auc_mu`, `ndcg`,
/// `map`/`average_precision`, `multi_error`, `multi_logloss`,
/// `cross_entropy`/`xentropy`, `cross_entropy_lambda`/`xentlambda`,
/// `kullback_leibler`) and any unknown name, so an unsupported metric can never be
/// silently routed on-device (T-20-00-02).
///
/// **D-05 — this is an INDEPENDENT list; it does NOT call
/// [`device_objective_supported`](crate::device_objective::device_objective_supported).**
/// MAPE / Gamma / Gamma-deviance / Tweedie are metric-supported here even though those
/// objectives are host-only. This is a classifier only — it does NOT enable any
/// on-device path (that stays gated behind `LGBM_CUDA_ON_DEVICE`).
#[must_use]
pub fn metric_supported(name: &str) -> bool {
    DeviceMetricKind::from_name(name).is_some()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device_objective::device_objective_supported;

    /// ODL-17: every non-pointwise metric is host-only.
    #[test]
    fn metric_supported_rejects_unsupported() {
        for name in [
            "auc",
            "auc_mu",
            "ndcg",
            "map",
            "mean_average_precision",
            "average_precision",
            "multi_error",
            "multi_logloss",
            "cross_entropy",
            "xentropy",
            "cross_entropy_lambda",
            "xentlambda",
            "kullback_leibler",
            "this_is_not_a_metric",
        ] {
            assert!(
                !metric_supported(name),
                "{name} has no on-device EvalKernel and must fall back to the host path (ODL-17)"
            );
            assert!(
                DeviceMetricKind::from_name(name).is_none(),
                "{name} must classify to None"
            );
        }
    }

    /// ODL-17: all twelve pointwise losses are on-device supported, keyed on the
    /// METRIC name list — NOT the objective-support list.
    #[test]
    fn metric_supported_accepts_all_twelve() {
        let expected = [
            ("l2", DeviceMetricKind::L2),
            ("mean_squared_error", DeviceMetricKind::L2),
            ("mse", DeviceMetricKind::L2),
            ("regression", DeviceMetricKind::L2),
            ("rmse", DeviceMetricKind::Rmse),
            ("root_mean_squared_error", DeviceMetricKind::Rmse),
            ("l1", DeviceMetricKind::L1),
            ("mean_absolute_error", DeviceMetricKind::L1),
            ("quantile", DeviceMetricKind::Quantile),
            ("huber", DeviceMetricKind::Huber),
            ("fair", DeviceMetricKind::Fair),
            ("poisson", DeviceMetricKind::Poisson),
            ("mape", DeviceMetricKind::Mape),
            ("gamma", DeviceMetricKind::Gamma),
            ("gamma_deviance", DeviceMetricKind::GammaDeviance),
            ("tweedie", DeviceMetricKind::Tweedie),
            ("binary_logloss", DeviceMetricKind::BinaryLogloss),
            ("binary", DeviceMetricKind::BinaryLogloss),
        ];
        for (name, kind) in expected {
            assert!(
                metric_supported(name),
                "{name} has an on-device §12 EvalKernel and must be supported (ODL-17)"
            );
            assert_eq!(
                DeviceMetricKind::from_name(name),
                Some(kind),
                "{name} must classify to {kind:?}"
            );
        }
    }

    /// D-05 CRITICAL asymmetry: MAPE / Gamma / Gamma-deviance / Tweedie are
    /// metric-supported on-device even though their OBJECTIVES have no on-device
    /// grad/hess kernel. The metric discriminator MUST NOT delegate to the
    /// objective-support classifier.
    #[test]
    fn metric_supported_objective_asymmetry_holds() {
        for name in ["mape", "gamma", "gamma_deviance", "tweedie"] {
            assert!(
                metric_supported(name) && !device_objective_supported(name),
                "{name} must be metric-supported yet objective-unsupported (D-05 asymmetry)"
            );
        }
        // Sanity: the shared-name families that ARE supported on both sides.
        for name in ["l2", "l1", "quantile", "huber", "fair", "poisson", "binary"] {
            assert!(
                metric_supported(name) && device_objective_supported(name),
                "{name} should be supported by both discriminators"
            );
        }
    }
}
