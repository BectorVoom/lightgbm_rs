//! Phase-20 on-device pointwise metric evaluator (§12 / §12.1, ODL-17).
//!
//! Owning phase: **20**. Filled by Plan **20-02** (Wave-2).
//!
//! ## What lives here
//! The `EvalKernel` + two-stage f64 reduction over the twelve pointwise
//! (regression/binary) losses the [`crate::device_metric::metric_supported`]
//! discriminator gates on-device (RMSE, L2, L1, Quantile, Huber, Fair, Poisson,
//! MAPE, Gamma, Gamma-deviance, Tweedie, Binary-logloss):
//! - [`metric_on_point`] — a comptime-generic `#[cube]` per-row loss transcribing
//!   the §12.1 table VERBATIM from the host `lgbm-metric::{regression,binary}`
//!   `loss_on_point` arms (D-03 secondary anchor). One `#[comptime] metric: u32`
//!   branch per row folds to straight-line code at expansion (Pattern 1 twin of
//!   `objective_regression::grad_hess_body`).
//! - `EvalKernel<#[comptime] metric, #[comptime] use_weight>` — one thread per row
//!   emitting `(loss_out[i], weight_out[i])`, bounds-guarded `i < num_data`.
//! - the host launcher [`eval_pointwise_on`] folds the per-row partials to scalar
//!   `sum_loss` / `sum_weight` through the REUSED ordered f64 fold
//!   [`crate::kernels::primitives::reduce_sum_f64_on`] (D-10 — NOT rebuilt), then
//!   applies the per-metric `AverageLoss` finalizer (RMSE `sqrt`, Gamma-deviance
//!   `×2`, else `Σloss/Σweight`).
//! - [`eval_metric_on`] composes `ConvertOutput` (D-04) into the flow — the
//!   inverse-link runs into a pre-allocated `score_convert_buffer` BEFORE the
//!   `EvalKernel`, keyed off the ORIGINAL metric name (via [`DeviceMetricKind`]),
//!   NEVER `DeviceObjectiveKind`.
//!
//! ## Anchor discipline (D-05 / D-07)
//! The host `lgbm_metric` `loss_on_point` f64 fold is the parity oracle. On the
//! default cubecl-cpu f64 anchor the device value is **bit-exact** to it; the ROCm
//! f32-storage mirror is held to the ~1e-6 envelope — NEVER GPU-vs-GPU
//! (def-f8u-01). The per-row point/param f64 is the reference-blessed metric loss
//! (D-08) — NOT a grow/build hot loop, so the "no gratuitous f64 per-row hot loop"
//! prohibition does not apply. Additive and OFF by default behind the
//! `LGBM_CUDA_ON_DEVICE` seam.

use cubecl::prelude::*;

use crate::device_metric::DeviceMetricKind;
use crate::error::ComputeError;
use crate::kernels::primitives::reduce_sum_f64_on;

// =========================================================================
// §12.1 metric tags (comptime dispatch selector) — the parallel to the
// `objective_regression.rs` `TAG_*` convention, one per §12.1 row.
// =========================================================================

/// `l2` / `mse` / `regression`: `(score - label)^2`, `AverageLoss = Σ/Σw`.
pub const METRIC_L2: u32 = 0;
/// `rmse`: `(score - label)^2`, `AverageLoss = sqrt(Σ/Σw)`.
pub const METRIC_RMSE: u32 = 1;
/// `l1` / `mae`: `|score - label|`.
pub const METRIC_L1: u32 = 2;
/// `quantile`: pinball loss at `param = alpha`.
pub const METRIC_QUANTILE: u32 = 3;
/// `huber`: Huber loss with knee at `param = alpha`.
pub const METRIC_HUBER: u32 = 4;
/// `fair`: `c*x - c^2*log1p(x/c)`, `param = fair_c`.
pub const METRIC_FAIR: u32 = 5;
/// `poisson`: `s - label*ln(s)` (score is post-`exp` ConvertOutput).
pub const METRIC_POISSON: u32 = 6;
/// `mape`: `|label - score| / max(1, |label|)`.
pub const METRIC_MAPE: u32 = 7;
/// `gamma`: `psi = 1` deviance via `SafeLog` (score is post-`exp`).
pub const METRIC_GAMMA: u32 = 8;
/// `gamma_deviance`: `tmp - SafeLog(tmp) - 1`, `AverageLoss = Σ*2`, `param = 1e-9`.
pub const METRIC_GAMMA_DEVIANCE: u32 = 9;
/// `tweedie`: Tweedie deviance at `param = tweedie_variance_power` (post-`exp`).
pub const METRIC_TWEEDIE: u32 = 10;
/// `binary_logloss`: `-log(prob)` / `-log(1-prob)` with the `kEpsilon` floor
/// (score is post-`sigmoid` ConvertOutput).
pub const METRIC_BINARY_LOGLOSS: u32 = 11;

/// Map a [`DeviceMetricKind`] (the ORIGINAL metric-name classifier — NOT
/// `DeviceObjectiveKind`, D-04) to its §12.1 [`metric_on_point`] tag.
#[must_use]
pub fn metric_tag(kind: DeviceMetricKind) -> u32 {
    match kind {
        DeviceMetricKind::L2 => METRIC_L2,
        DeviceMetricKind::Rmse => METRIC_RMSE,
        DeviceMetricKind::L1 => METRIC_L1,
        DeviceMetricKind::Quantile => METRIC_QUANTILE,
        DeviceMetricKind::Huber => METRIC_HUBER,
        DeviceMetricKind::Fair => METRIC_FAIR,
        DeviceMetricKind::Poisson => METRIC_POISSON,
        DeviceMetricKind::Mape => METRIC_MAPE,
        DeviceMetricKind::Gamma => METRIC_GAMMA,
        DeviceMetricKind::GammaDeviance => METRIC_GAMMA_DEVIANCE,
        DeviceMetricKind::Tweedie => METRIC_TWEEDIE,
        DeviceMetricKind::BinaryLogloss => METRIC_BINARY_LOGLOSS,
    }
}

// =========================================================================
// §12.1 per-point loss — the comptime-generic map. One branch per §12.1 row,
// transcribed VERBATIM from `lgbm-metric::{regression,binary}::loss_on_point`.
// =========================================================================

/// `Common::SafeLog` (`common.h`): `x > kZeroThreshold ? ln(x) : ln(kZeroThreshold)`
/// with `kZeroThreshold = 1e-35`. Mirrors `lgbm-metric::regression::safe_log`. The
/// else-branch threshold is derived through f32 (`F::new` takes f32); it is never
/// reached on valid gamma inputs (positive labels / exp'd scores), where the
/// `x.ln()` branch is bit-exact vs the host f64 `safe_log`.
#[cube]
fn safe_log_metric<F: Float>(x: F) -> F {
    let thr = F::new(1e-35);
    select(x > thr, x.ln(), thr.ln())
}

/// The SINGLE-SOURCE-OF-TRUTH per-row §12.1 loss — one generic `#[cube]` fn over
/// the cell type `F`, dispatching the twelve pointwise losses via the
/// `#[comptime] metric` tag (folds to straight-line code at expansion). `score` is
/// POST-`ConvertOutput` (already `exp`'d for poisson/gamma/gamma_deviance/tweedie,
/// already `sigmoid`'d for binary — the compose happens in [`eval_metric_on`]).
/// `param` carries the single per-metric scalar: quantile/huber `alpha`, fair `c`,
/// tweedie `rho`, gamma_deviance `epsilon = 1e-9` (the one epsilon that is a host
/// f64 literal, so it must arrive through the kernel arg for bit-exactness); the
/// other epsilons are f32-derived (`1e-10f` / `kEpsilon = 1e-15f`) and reproduced
/// bit-exactly via `F::new`.
#[cube]
#[allow(unused_assignments)]
pub fn metric_on_point<F: Float>(label: F, score: F, param: F, #[comptime] metric: u32) -> F {
    let mut loss = F::new(0.0);

    if metric == METRIC_L2 || metric == METRIC_RMSE {
        // (score - label)^2. RMSE applies sqrt at AverageLoss, not here.
        let d = score - label;
        loss = d * d;
    } else if metric == METRIC_L1 {
        loss = (score - label).abs();
    } else if metric == METRIC_QUANTILE {
        // delta = label - score; delta < 0 ? (alpha-1)*delta : alpha*delta.
        let delta = label - score;
        loss = select(
            delta < F::new(0.0),
            (param - F::new(1.0)) * delta,
            param * delta,
        );
    } else if metric == METRIC_HUBER {
        // |diff| <= alpha ? 0.5*diff*diff : alpha*(|diff| - 0.5*alpha).
        let diff = score - label;
        let ad = diff.abs();
        loss = select(
            ad <= param,
            F::new(0.5) * diff * diff,
            param * (ad - F::new(0.5) * param),
        );
    } else if metric == METRIC_FAIR {
        // c*x - c^2*log1p(x/c), x = |score - label|.
        let x = (score - label).abs();
        let c = param;
        loss = c * x - c * c * (x / c).log1p();
    } else if metric == METRIC_POISSON {
        // eps = 1e-10f (f32-derived); s = max(score, eps); s - label*ln(s).
        let eps = F::new(1e-10);
        let s = select(score < eps, eps, score);
        loss = s - label * s.ln();
    } else if metric == METRIC_MAPE {
        // |label - score| / max(1, |label|).
        let al = label.abs();
        let denom = select(F::new(1.0) >= al, F::new(1.0), al);
        loss = (label - score).abs() / denom;
    } else if metric == METRIC_GAMMA {
        // psi = 1 deviance: theta = -1/score; b = -SafeLog(-theta);
        // c = SafeLog(label) - SafeLog(label) = 0; loss = -((label*theta - b) + c).
        let theta = F::new(-1.0) / score;
        let b = -safe_log_metric::<F>(-theta);
        let cc = safe_log_metric::<F>(label) - safe_log_metric::<F>(label);
        loss = -((label * theta - b) + cc);
    } else if metric == METRIC_GAMMA_DEVIANCE {
        // epsilon = 1e-9 (host f64 literal, arrives via param); tmp - SafeLog(tmp) - 1.
        let tmp = label / (score + param);
        loss = tmp - safe_log_metric::<F>(tmp) - F::new(1.0);
    } else if metric == METRIC_TWEEDIE {
        // rho = param; eps = 1e-10f (f32-derived); s = max(score, eps).
        let rho = param;
        let eps = F::new(1e-10);
        let s = select(score < eps, eps, score);
        let a = label * ((F::new(1.0) - rho) * s.ln()).exp() / (F::new(1.0) - rho);
        let bb = ((F::new(2.0) - rho) * s.ln()).exp() / (F::new(2.0) - rho);
        loss = -a + bb;
    } else {
        // METRIC_BINARY_LOGLOSS. prob is the sigmoid-converted score. eps = kEpsilon
        // = 1e-15f (f32-derived). label <= 0 -> -log(1-prob) guarded 1-prob > eps;
        // label > 0 -> -log(prob) guarded prob > eps; else -log(eps).
        let prob = score;
        let eps = F::new(1e-15);
        let neg_eps_ln = -(eps.ln());
        let one_minus = F::new(1.0) - prob;
        let cand_neg = select(one_minus > eps, -(one_minus.ln()), neg_eps_ln);
        let cand_pos = select(prob > eps, -(prob.ln()), neg_eps_ln);
        loss = select(label <= F::new(0.0), cand_neg, cand_pos);
    }

    loss
}

// =========================================================================
// EvalKernel<metric, use_weight> — one thread per row emitting the per-row
// (loss, weight) partials the two-stage fold reduces to scalars.
// =========================================================================

/// The `EvalKernel` body: one thread per row writes `loss_out[i]` (the per-row
/// §12.1 loss, `× weight[i]` on the weighted path) and `weight_out[i]` (the row's
/// contribution to `Σweight` — the weight, or `1.0` unweighted). Bounds-guarded
/// `i < num_data`.
#[cube]
#[allow(clippy::too_many_arguments)]
fn eval_body<F: Float>(
    scores: &Array<F>,
    labels: &Array<F>,
    weights: &Array<F>,
    loss_out: &mut Array<F>,
    weight_out: &mut Array<F>,
    param: F,
    #[comptime] metric: u32,
    #[comptime] use_weight: bool,
) {
    let i = ABSOLUTE_POS;
    if i < scores.len() {
        let raw = metric_on_point::<F>(labels[i], scores[i], param, metric);
        if use_weight {
            let w = weights[i];
            loss_out[i] = raw * w;
            weight_out[i] = w;
        } else {
            loss_out[i] = raw;
            weight_out[i] = F::new(1.0);
        }
    }
}

/// f64 cpu-anchor `EvalKernel` wrapper (the deterministic f64 reference, D-07). The
/// ROCm f32 mirror would instantiate `eval_body::<f32>` identically.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn eval_kernel_f64(
    scores: &Array<f64>,
    labels: &Array<f64>,
    weights: &Array<f64>,
    loss_out: &mut Array<f64>,
    weight_out: &mut Array<f64>,
    param: f64,
    #[comptime] metric: u32,
    #[comptime] use_weight: bool,
) {
    eval_body::<f64>(
        scores, labels, weights, loss_out, weight_out, param, metric, use_weight,
    );
}

/// The §12.1 `AverageLoss` finalizer per metric: RMSE `sqrt(Σloss/Σweight)`,
/// Gamma-deviance `Σloss*2`, everything else `Σloss/Σweight`
/// (`regression_metric.hpp` / `lgbm-metric::regression::eval`).
fn average_loss(metric: u32, sum_loss: f64, sum_weight: f64) -> f64 {
    if metric == METRIC_RMSE {
        (sum_loss / sum_weight).sqrt()
    } else if metric == METRIC_GAMMA_DEVIANCE {
        sum_loss * 2.0
    } else {
        sum_loss / sum_weight
    }
}

/// Evaluate one pointwise metric over ALREADY-CONVERTED `scores` (the caller runs
/// `ConvertOutput` first — see [`eval_metric_on`]). Runs the `EvalKernel` (one
/// thread per row) then folds the per-row `(loss, weight)` partials to scalars
/// through the REUSED ordered f64 fold [`reduce_sum_f64_on`] (D-10) and applies the
/// per-metric [`average_loss`] finalizer.
///
/// `scores` are the f64 post-`ConvertOutput` scores; `labels` the f32 labels;
/// `weights` `Some(&[f32])` for the weighted path (`AverageLoss` then divides by
/// `Σweight` instead of `num_data`). `param` is the single per-metric scalar (see
/// [`metric_on_point`]).
///
/// # Errors
/// [`ComputeError::LengthMismatch`] if `labels` (or `weights`) disagree with
/// `scores` in length (V5 boundary, validated before any device alloc / launch);
/// propagates the fold primitive errors.
pub fn eval_pointwise_on<R: cubecl::Runtime>(
    client: &ComputeClient<R>,
    metric: u32,
    scores: &[f64],
    labels: &[f32],
    weights: Option<&[f32]>,
    param: f64,
) -> Result<f64, ComputeError> {
    let n = scores.len();
    if labels.len() != n {
        return Err(ComputeError::LengthMismatch { expected: n, actual: labels.len() });
    }
    if let Some(w) = weights {
        if w.len() != n {
            return Err(ComputeError::LengthMismatch { expected: n, actual: w.len() });
        }
    }
    if n == 0 {
        return Ok(0.0);
    }

    let labels_f64: Vec<f64> = labels.iter().map(|&l| f64::from(l)).collect();
    let use_weight = weights.is_some();
    // A weights handle is always required even when the comptime branch compiles the
    // read out; a 1-cell dummy is never accessed on the unweighted path.
    let weights_f64: Vec<f64> = weights
        .map(|w| w.iter().map(|&x| f64::from(x)).collect())
        .unwrap_or_else(|| vec![0.0f64]);
    let w_len = weights_f64.len();

    let h_scores = client.create_from_slice(f64::as_bytes(scores));
    let h_labels = client.create_from_slice(f64::as_bytes(&labels_f64));
    let h_weights = client.create_from_slice(f64::as_bytes(&weights_f64));
    let h_loss = client.empty(n * core::mem::size_of::<f64>());
    let h_weight_out = client.empty(n * core::mem::size_of::<f64>());

    let cube_dim = 256u32;
    let cube_count = (n as u32).div_ceil(cube_dim);

    // SAFETY: `h_scores`/`h_labels`/`h_loss`/`h_weight_out` are each sized exactly
    // `n` f64 cells and `h_weights` `w_len`; all outlive the launch. The kernel
    // bounds-guards `i < scores.len()` so every `scores/labels/loss_out/weight_out[i]`
    // access is in `[0, n)`; `weights[i]` is only read on the (comptime) use_weight
    // path where `w_len == n`. cubecl unsafe is confined here (CMP-01 / T-20-02-01).
    unsafe {
        eval_kernel_f64::launch_unchecked(
            client,
            CubeCount::Static(cube_count, 1, 1),
            CubeDim::new_1d(cube_dim),
            ArrayArg::from_raw_parts(h_scores, n),
            ArrayArg::from_raw_parts(h_labels, n),
            ArrayArg::from_raw_parts(h_weights, w_len),
            ArrayArg::from_raw_parts(h_loss.clone(), n),
            ArrayArg::from_raw_parts(h_weight_out.clone(), n),
            param,
            metric,
            use_weight,
        );
    }

    let loss_bytes = client.read_one_unchecked(h_loss);
    let weight_bytes = client.read_one_unchecked(h_weight_out);
    let loss_partials = f64::from_bytes(&loss_bytes);
    let weight_partials = f64::from_bytes(&weight_bytes);

    // Two-stage fold: the per-row partials reduce to scalar Σloss / Σweight through
    // the single-owner ordered f64 fold (D-10, bit-exact to the host ascending fold).
    let sum_loss = reduce_sum_f64_on(client, loss_partials)?;
    let sum_weight = reduce_sum_f64_on(client, weight_partials)?;
    Ok(average_loss(metric, sum_loss, sum_weight))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::cpu_client;

    // --- host reference (the §12.1 loss_on_point math, transcribed for the test
    // only — lgbm-compute stays free of an lgbm-metric dep) ---

    fn host_safe_log(x: f64) -> f64 {
        const K_ZERO_THRESHOLD: f64 = 1e-35;
        if x > K_ZERO_THRESHOLD { x.ln() } else { K_ZERO_THRESHOLD.ln() }
    }

    fn host_loss_on_point(metric: u32, label: f64, score: f64, param: f64) -> f64 {
        match metric {
            METRIC_L2 | METRIC_RMSE => {
                let d = score - label;
                d * d
            }
            METRIC_L1 => (score - label).abs(),
            METRIC_QUANTILE => {
                let delta = label - score;
                if delta < 0.0 { (param - 1.0) * delta } else { param * delta }
            }
            METRIC_HUBER => {
                let diff = score - label;
                if diff.abs() <= param {
                    0.5 * diff * diff
                } else {
                    param * (diff.abs() - 0.5 * param)
                }
            }
            METRIC_FAIR => {
                let x = (score - label).abs();
                let c = param;
                c * x - c * c * (x / c).ln_1p()
            }
            METRIC_POISSON => {
                let eps = 1e-10_f32 as f64;
                let s = if score < eps { eps } else { score };
                s - label * s.ln()
            }
            METRIC_MAPE => (label - score).abs() / (1.0_f32 as f64).max(label.abs()),
            METRIC_GAMMA => {
                let theta = -1.0 / score;
                let b = -host_safe_log(-theta);
                let cc = host_safe_log(label) - host_safe_log(label);
                -((label * theta - b) + cc)
            }
            METRIC_GAMMA_DEVIANCE => {
                let tmp = label / (score + param);
                tmp - host_safe_log(tmp) - 1.0
            }
            METRIC_TWEEDIE => {
                let rho = param;
                let eps = 1e-10_f32 as f64;
                let s = if score < eps { eps } else { score };
                let a = label * ((1.0 - rho) * s.ln()).exp() / (1.0 - rho);
                let bb = ((2.0 - rho) * s.ln()).exp() / (2.0 - rho);
                -a + bb
            }
            _ => {
                // METRIC_BINARY_LOGLOSS. score is the sigmoid-converted prob.
                let prob = score;
                let eps = 1e-15_f32 as f64;
                if label <= 0.0 {
                    if 1.0 - prob > eps {
                        return -(1.0 - prob).ln();
                    }
                } else if prob > eps {
                    return -prob.ln();
                }
                -eps.ln()
            }
        }
    }

    /// The host two-stage fold: Σ loss_on_point (ascending) then the AverageLoss
    /// finalizer — the exact `lgbm-metric` reference shape.
    fn host_eval(metric: u32, scores: &[f64], labels: &[f32], param: f64) -> f64 {
        let n = scores.len();
        let mut sum_loss = 0.0f64;
        for i in 0..n {
            sum_loss += host_loss_on_point(metric, f64::from(labels[i]), scores[i], param);
        }
        average_loss(metric, sum_loss, n as f64)
    }

    /// The twelve §12.1 metrics with a `param` and a converted-score fixture that
    /// keeps every loss finite (positive scores/labels for the log-domain arms).
    fn cases() -> Vec<(u32, f64, Vec<f64>, Vec<f32>)> {
        let s = vec![2.0f64, 1.5, 3.25, 0.75, 2.5];
        let l = vec![1.0f32, 3.0, 3.0, 0.5, 2.0];
        vec![
            (METRIC_L2, 0.0, s.clone(), l.clone()),
            (METRIC_RMSE, 0.0, s.clone(), l.clone()),
            (METRIC_L1, 0.0, s.clone(), l.clone()),
            (METRIC_QUANTILE, 0.9, s.clone(), l.clone()),
            (METRIC_HUBER, 1.0, s.clone(), l.clone()),
            (METRIC_FAIR, 1.0, s.clone(), l.clone()),
            (METRIC_POISSON, 0.0, s.clone(), l.clone()),
            (METRIC_MAPE, 0.0, s.clone(), l.clone()),
            (METRIC_GAMMA, 0.0, s.clone(), l.clone()),
            (METRIC_GAMMA_DEVIANCE, 1e-9, s.clone(), l.clone()),
            (METRIC_TWEEDIE, 1.5, s.clone(), l.clone()),
            // binary: converted probs in (0,1), {0,1} labels.
            (
                METRIC_BINARY_LOGLOSS,
                0.0,
                vec![0.8f64, 0.3, 0.6, 0.1, 0.95],
                vec![1.0f32, 0.0, 1.0, 0.0, 1.0],
            ),
        ]
    }

    /// Each of the twelve device metric values equals the host `loss_on_point`
    /// fold bit-exact on the cpu f64 anchor (SC: per-metric bit-exact parity).
    #[test]
    fn metric_pointwise_all_twelve_bit_exact_vs_host() {
        let client = cpu_client();
        for (metric, param, scores, labels) in cases() {
            let device = eval_pointwise_on(&client, metric, &scores, &labels, None, param)
                .expect("device eval");
            let host = host_eval(metric, &scores, &labels, param);
            assert_eq!(
                device.to_bits(),
                host.to_bits(),
                "metric {metric}: device {device} != host {host} (not bit-exact)"
            );
        }
    }

    /// RMSE finalizes with `sqrt` and Gamma-deviance multiplies the summed loss by
    /// 2 (the two non-mean AverageLoss finalizers).
    #[test]
    fn metric_pointwise_rmse_sqrt_and_gamma_deviance_double() {
        let client = cpu_client();
        let scores = vec![2.0f64, 1.5, 3.25, 0.75, 2.5];
        let labels = vec![1.0f32, 3.0, 3.0, 0.5, 2.0];

        // RMSE == sqrt(L2).
        let l2 = eval_pointwise_on(&client, METRIC_L2, &scores, &labels, None, 0.0).unwrap();
        let rmse = eval_pointwise_on(&client, METRIC_RMSE, &scores, &labels, None, 0.0).unwrap();
        assert_eq!(rmse.to_bits(), l2.sqrt().to_bits(), "rmse {rmse} != sqrt(l2) {}", l2.sqrt());

        // Gamma-deviance == 2 * (Σ loss / … but finalizer is Σ*2, so == 2 * mean*n).
        let gd = eval_pointwise_on(&client, METRIC_GAMMA_DEVIANCE, &scores, &labels, None, 1e-9)
            .unwrap();
        let host = host_eval(METRIC_GAMMA_DEVIANCE, &scores, &labels, 1e-9);
        assert_eq!(gd.to_bits(), host.to_bits(), "gamma_deviance {gd} != host {host}");
    }

    /// The weighted path with uniform 1.0 weights is bit-identical to the
    /// unweighted path (weight-equivalence), exercising the `Σweight` fold.
    #[test]
    fn metric_pointwise_uniform_weights_match_unweighted() {
        let client = cpu_client();
        let scores = vec![2.0f64, 1.5, 3.25, 0.75, 2.5];
        let labels = vec![1.0f32, 3.0, 3.0, 0.5, 2.0];
        let ones = vec![1.0f32; scores.len()];
        for metric in [METRIC_L2, METRIC_L1, METRIC_POISSON] {
            let unweighted =
                eval_pointwise_on(&client, metric, &scores, &labels, None, 0.0).unwrap();
            let weighted =
                eval_pointwise_on(&client, metric, &scores, &labels, Some(&ones), 0.0).unwrap();
            assert_eq!(
                weighted.to_bits(),
                unweighted.to_bits(),
                "metric {metric}: weighted(1.0) {weighted} != unweighted {unweighted}"
            );
        }
    }

    /// A length mismatch is a typed V5-boundary error before any launch.
    #[test]
    fn metric_pointwise_length_mismatch_is_typed_error() {
        let client = cpu_client();
        let e = eval_pointwise_on(&client, METRIC_L2, &[1.0, 2.0], &[1.0f32], None, 0.0);
        assert!(matches!(e, Err(ComputeError::LengthMismatch { .. })));
    }

    /// Empty input folds to 0.0 (no launch), matching the host empty guard.
    #[test]
    fn metric_pointwise_empty_is_zero() {
        let client = cpu_client();
        let v = eval_pointwise_on(&client, METRIC_L2, &[], &[], None, 0.0).unwrap();
        assert_eq!(v, 0.0);
    }
}
