//! The public [`Booster`] (D-05) — the trained ensemble + eval history, plus the
//! `train` / `predict` entry points.
//!
//! `train` drives the full spine: builder→Config → identity-binned feature
//! columns → [`lgbm_boosting::Gbdt`] loop (BoostFromAverage → GetGradients →
//! per-class tree → Shrinkage → UpdateScore → AddBias) → [`GbdtModel`]. `predict`
//! delegates to [`GbdtModel::predict_raw`] then the predict-side
//! [`lgbm_model::ObjectiveKind::convert`] transform (identity for regression).
//!
//! 06-02 scope: the regression L2 single-output spine on an identity-binned dense
//! corpus (the capture's exact binning). Per-objective / bagging / early-stopping
//! widen in 06-03..06-05.

use lgbm_boosting::objective::BoostObjective;
use lgbm_boosting::{Gbdt, IterSnapshot};
use lgbm_compute::gain::GainConfig;
use lgbm_compute::runtime::cpu_client;
use lgbm_compute::CpuBackend;
use lgbm_core::Config;
use lgbm_dataset::bin_mapper::MissingType;
use lgbm_metric::{BinaryMetric, Metric, MultiLogloss};
use lgbm_model::{GbdtModel, ObjectiveKind};
use lgbm_objective::{Binary, CustomObjective, MulticlassOva, MulticlassSoftmax, Objective};
use lgbm_treelearner::learner::FeatureColumn;
use lgbm_treelearner::{offset_for_most_freq_bin, SerialTreeLearner};

use crate::error::LgbmError;

/// One configured eval metric — either a regression metric (raw-score) or a
/// binary metric (prob-space / AUC). Unifies the per-round eval-history loop
/// across objectives.
enum EvalMetric {
    /// A regression metric (`l1`/`l2`/`rmse`) over the raw score.
    Reg(Metric),
    /// A binary metric (`binary_logloss`/`binary_error`/`auc`).
    Bin(BinaryMetric),
    /// The `multi_logloss` metric over the class-major score buffer.
    Multi(MultiLogloss),
}

impl EvalMetric {
    fn name(&self) -> String {
        match self {
            EvalMetric::Reg(m) => m.name().to_string(),
            EvalMetric::Bin(m) => m.name().to_string(),
            EvalMetric::Multi(m) => m.name().to_string(),
        }
    }

    fn eval(&self, scores: &[f64], labels: &[f32]) -> Result<f64, LgbmError> {
        match self {
            EvalMetric::Reg(m) => m.eval(scores, labels).map_err(LgbmError::Metric),
            EvalMetric::Bin(m) => m.eval(scores, labels).map_err(LgbmError::Metric),
            EvalMetric::Multi(m) => m.eval(scores, labels).map_err(LgbmError::Metric),
        }
    }
}

/// A dense, identity-binned training corpus: raw integer-valued features (one
/// column per feature, bin index == raw value) + f32 labels.
///
/// This is the 06-02 spine training input — it mirrors the capture's identity
/// binning (distinct consecutive integers `0..K-1` per feature) so the Rust-grown
/// trees are bit-comparable to the real-binary goldens. (Full arbitrary-value
/// binning via the Phase-2 `Dataset` path widens later.)
#[derive(Debug, Clone)]
pub struct DenseCorpus {
    /// Row-major raw feature values, `num_data` rows × `num_features` columns.
    pub features: Vec<Vec<f64>>,
    /// Per-row labels (length `num_data`).
    pub labels: Vec<f32>,
}

/// LightGBM's real per-bin upper bound for an identity-binned integer feature:
/// the `(b + 0.5)` midpoint nudged up by exactly one ULP — LightGBM's
/// `GetDoubleUpperBound`, matching the `2.5000000000000004` /
/// `1.5000000000000002` values the capture emits (verified against the Phase-5
/// `learner_parity::real_upper_bounds`).
fn real_upper_bounds(num_bin: u32) -> Vec<f64> {
    (0..num_bin)
        .map(|b| {
            let mid = b as f64 + 0.5;
            f64::from_bits(mid.to_bits() + 1) // mid + 1 ULP (mid > 0)
        })
        .collect()
}

/// Build identity-binned [`FeatureColumn`]s from a dense corpus. Each feature's
/// distinct raw values must be the consecutive integers `0..K-1` (the
/// identity-binning precondition); `num_bin = K`, `bin == raw value`,
/// `most_freq_bin = the modal bin`, and `bin_upper_bound = real_upper_bounds(K)`.
///
/// # Errors
/// [`LgbmError::InvalidCorpus`] when a column is not identity-binnable (a raw
/// value is negative or non-integer, or the distinct values are not `0..K-1`).
fn build_feature_columns(corpus: &DenseCorpus) -> Result<Vec<FeatureColumn>, LgbmError> {
    let num_data = corpus.features.len();
    if num_data == 0 {
        return Err(LgbmError::InvalidCorpus {
            detail: "empty corpus (no rows)".into(),
        });
    }
    let num_features = corpus.features[0].len();
    if corpus.labels.len() != num_data {
        return Err(LgbmError::InvalidCorpus {
            detail: format!(
                "labels length {} != num_data {num_data}",
                corpus.labels.len()
            ),
        });
    }
    let mut columns = Vec::with_capacity(num_features);
    for j in 0..num_features {
        // Collect the column's raw values as bins (must be non-negative integers).
        let mut bins = Vec::with_capacity(num_data);
        for (i, row) in corpus.features.iter().enumerate() {
            if row.len() != num_features {
                return Err(LgbmError::InvalidCorpus {
                    detail: format!("row {i} has {} cols, expected {num_features}", row.len()),
                });
            }
            let v = row[j];
            if v < 0.0 || v.fract() != 0.0 {
                return Err(LgbmError::InvalidCorpus {
                    detail: format!(
                        "feature {j} row {i} value {v} is not a non-negative integer \
                         (identity binning requires consecutive integers 0..K-1)"
                    ),
                });
            }
            bins.push(v as u32);
        }
        // num_bin = max bin + 1; verify distinct values are exactly 0..num_bin-1.
        let max_bin = *bins.iter().max().unwrap();
        let num_bin = max_bin + 1;
        let mut seen = vec![false; num_bin as usize];
        for &b in &bins {
            seen[b as usize] = true;
        }
        if !seen.iter().all(|&s| s) {
            return Err(LgbmError::InvalidCorpus {
                detail: format!(
                    "feature {j} distinct bins are not the consecutive integers \
                     0..{num_bin}-1 (identity binning precondition)"
                ),
            });
        }
        // most_freq_bin = the modal bin (ties broken to the lowest, matching the
        // capture's expectation).
        let mut counts = vec![0i32; num_bin as usize];
        for &b in &bins {
            counts[b as usize] += 1;
        }
        let most_freq_bin = counts
            .iter()
            .enumerate()
            .max_by_key(|&(idx, &c)| (c, std::cmp::Reverse(idx)))
            .map(|(idx, _)| idx as u32)
            .unwrap_or(0);
        columns.push(FeatureColumn {
            bins,
            num_bin,
            offset: offset_for_most_freq_bin(most_freq_bin),
            min_bin: 0,
            max_bin,
            default_bin: num_bin,
            most_freq_bin,
            missing_type: MissingType::None,
            bin_upper_bound: real_upper_bounds(num_bin),
            real_feature_index: j as i32,
        });
    }
    Ok(columns)
}

/// The trained ensemble + eval history (D-05). Mirrors the Python `Booster`'s
/// `best_iteration_` / `best_score_` / `record_evaluation` surface; the full
/// early-stopping population lands in 06-05 (here `best_iteration` is the last
/// trained iteration and the eval history is the per-round metric values).
#[derive(Debug, Clone)]
pub struct Booster {
    /// The serializable ensemble (Phase-3 container).
    model: GbdtModel,
    /// The parsed predict-side objective transform.
    objective_kind: ObjectiveKind,
    /// C++ `best_iteration_` (1-based round). 06-02: the last trained iteration
    /// (no early stopping yet).
    pub best_iteration: i32,
    /// Per-metric eval history (metric name → per-round value), mirroring Python
    /// `record_evaluation` / `evals_result_`. 06-02: populated with the training
    /// l2/rmse per round when a valid metric is configured.
    pub eval_history: Vec<(String, Vec<f64>)>,
    /// The per-iteration L2 raw-score snapshots (the internal `score_` after each
    /// iter) — exposed for the L2 golden replay + Open-Q2/A4 verification.
    pub iter_scores: Vec<Vec<f64>>,
    /// The per-iteration L1 grad/hess snapshots — exposed for the L1 golden replay.
    pub iter_grad_hess: Vec<(Vec<f32>, Vec<f32>)>,
}

impl Booster {
    /// The trained ensemble (for model-text serialization / inspection).
    pub fn model(&self) -> &GbdtModel {
        &self.model
    }

    /// Transformed prediction for one raw feature row (width `max_feature_idx +
    /// 1`): `GbdtModel::predict_raw` then the objective's `ConvertOutput`
    /// (identity for regression). Returns a `num_tree_per_iteration`-wide f32
    /// vector.
    pub fn predict_row(&self, features: &[f64]) -> Vec<f32> {
        let raw = self.model.predict_raw(features, 0, -1);
        let mut transformed = vec![0.0f64; raw.len()];
        // For multiclass the convert reads the whole class vector; for the spine
        // (single output) it is the identity. ObjectiveKind::convert handles both.
        self.objective_kind.convert(&raw, &mut transformed);
        transformed.into_iter().map(|v| v as f32).collect()
    }

    /// Raw (untransformed) accumulated score for one feature row, over the first
    /// `num_iteration` trees (`<= 0` = all). This is the public mirror of
    /// `predict(raw_score=True, num_iteration=k)` (Open-Q2/A4).
    pub fn predict_row_raw(&self, features: &[f64], num_iteration: i32) -> Vec<f64> {
        self.model.predict_raw(features, 0, num_iteration)
    }
}

/// Train a regression spine [`Booster`] from a [`Config`] + a dense identity-binned
/// corpus (the 06-02 public entry point).
///
/// Drives the full vertical spine: objective/metric resolution → identity-binned
/// feature columns → the [`Gbdt`] loop → [`GbdtModel`]. The eval history is
/// populated when the config names an l2/rmse metric.
///
/// # Errors
/// [`LgbmError`] for an unsupported objective/metric, an invalid corpus, or a
/// loop/learner failure — never a panic (Security V5).
pub fn train(config: &Config, corpus: &DenseCorpus) -> Result<Booster, LgbmError> {
    // Resolve the training-side objective from config.objective (regression /
    // regression_l1 / binary). The custom-closure path is `train_custom`.
    let first = config.objective.split_whitespace().next().unwrap_or("");
    let (boost_obj, transformed_labels): (BoostObjective<'static>, Vec<f32>) = match first {
        "binary" => {
            let b = Binary::new(config.sigmoid).map_err(LgbmError::Objective)?;
            (BoostObjective::Binary(b), corpus.labels.clone())
        }
        "multiclass" | "softmax" => {
            // The integer class labels feed Init (label range check) + the per-row
            // strided softmax gather; they pass through unchanged (no transform).
            let m = MulticlassSoftmax::new(config.num_class, &corpus.labels)
                .map_err(LgbmError::Objective)?;
            (BoostObjective::Multiclass(m), corpus.labels.clone())
        }
        "multiclassova" | "multiclass_ova" | "ova" | "ovr" => {
            let o = MulticlassOva::new(config.num_class, config.sigmoid, &corpus.labels)
                .map_err(LgbmError::Objective)?;
            (BoostObjective::MulticlassOva(o), corpus.labels.clone())
        }
        _ => {
            // regression / regression_l1 (+ aliases) route through the enum factory.
            let o = Objective::from_config(config).map_err(LgbmError::Objective)?;
            let lbl = o.transform_labels(&corpus.labels);
            (BoostObjective::Builtin(o), lbl)
        }
    };
    let metrics = eval_metrics_for(first, config);
    train_inner(config, corpus, boost_obj, transformed_labels, metrics)
}

/// Train with a user-supplied `custom` objective closure (OBJ-02 / D-04). The
/// closure maps the current raw scores (f32) to `(grad, hess)`; `boost_from_average`
/// is forced OFF for custom (mirroring the C++ `obj == null` path).
///
/// # Errors
/// [`LgbmError`] for an invalid corpus or a loop/learner failure; a wrong-length
/// closure return surfaces as `LgbmError::Objective` (T-06-03-01), never a panic.
pub fn train_custom<'a, F>(
    config: &Config,
    corpus: &DenseCorpus,
    closure: F,
) -> Result<Booster, LgbmError>
where
    F: Fn(&[f64]) -> (Vec<f32>, Vec<f32>) + 'a,
{
    let custom = CustomObjective::new(closure);
    // The custom run's eval metric mirrors the capture (l2 over the raw score).
    let metrics = vec![EvalMetric::Reg(Metric::L2)];
    train_inner(
        config,
        corpus,
        BoostObjective::Custom(custom),
        corpus.labels.clone(),
        metrics,
    )
}

/// The shared training driver: identity-binned columns → the [`Gbdt`] loop →
/// per-round eval history → [`Booster`]. Generic over the [`BoostObjective`].
fn train_inner(
    config: &Config,
    corpus: &DenseCorpus,
    boost_obj: BoostObjective<'_>,
    labels: Vec<f32>,
    metrics: Vec<EvalMetric>,
) -> Result<Booster, LgbmError> {
    // Build the canonical `objective=` model line. LightGBM appends `num_class:` /
    // `sigmoid:` tokens for the multiclass objectives (e.g.
    // `multiclass num_class:3`, `multiclassova num_class:3 sigmoid:1`); the
    // single-output objectives use the bare name. This is BOTH the predict-side
    // ConvertOutput source AND the serialized model objective line (round-trip).
    let objective_string = canonical_objective_string(config);
    let objective_kind = ObjectiveKind::parse(&objective_string)
        // Custom objective (`objective = "custom"`/"none") has no predict-side
        // transform: fall back to identity (regression) for predict_row_raw.
        .unwrap_or(ObjectiveKind::Regression { sqrt: false });

    // ---- identity-binned feature columns ----
    let features = build_feature_columns(corpus)?;
    let num_data = corpus.features.len() as i32;
    let num_features = features.len();
    let max_feature_idx = features
        .iter()
        .map(|f| f.real_feature_index)
        .max()
        .unwrap_or(-1);

    // ---- the learner (single-output spine; K = num_class = 1) ----
    let backend = CpuBackend;
    let client = cpu_client();
    let gain = GainConfig::from_config(config);
    let mut learner = SerialTreeLearner::new(
        &backend,
        &client,
        gain,
        config.num_leaves,
        config.max_depth,
    )
    .with_features(features);

    // ---- the GBDT loop ----
    let mut gbdt = Gbdt::with_objective(
        boost_obj,
        config.learning_rate,
        config.num_class.max(1),
        num_data,
        config.boost_from_average,
        None,
    );
    let snaps: Vec<IterSnapshot> = gbdt
        .train(&mut learner, &labels, num_features, config.num_iterations)
        .map_err(LgbmError::Boosting)?;

    // ---- eval history (per-round metrics on the training scores) ----
    let mut eval_history: Vec<(String, Vec<f64>)> = metrics
        .iter()
        .map(|m| (m.name(), Vec::with_capacity(snaps.len())))
        .collect();
    for snap in &snaps {
        for (mi, m) in metrics.iter().enumerate() {
            let v = m.eval(&snap.score, &corpus.labels)?;
            eval_history[mi].1.push(v);
        }
    }

    let iter_scores: Vec<Vec<f64>> = snaps.iter().map(|s| s.score.clone()).collect();
    let iter_grad_hess: Vec<(Vec<f32>, Vec<f32>)> = snaps
        .iter()
        .map(|s| (s.gradients.clone(), s.hessians.clone()))
        .collect();

    let best_iteration = gbdt.num_iteration();
    let model = gbdt.into_model(
        objective_string,
        max_feature_idx,
        feature_names(num_features),
        feature_infos(corpus, num_features),
    );

    Ok(Booster {
        model,
        objective_kind,
        best_iteration,
        eval_history,
        iter_scores,
        iter_grad_hess,
    })
}

/// Build the canonical LightGBM `objective=` model line from the config. The
/// multiclass objectives append the `num_class:`/`sigmoid:` tokens exactly as the
/// C++ `MulticlassSoftmax::ToString` / `MulticlassOVA::ToString` do (so the model
/// text round-trips and `ObjectiveKind::parse` recovers num_class); the
/// single-output objectives use the bare `config.objective` name.
fn canonical_objective_string(config: &Config) -> String {
    let first = config.objective.split_whitespace().next().unwrap_or("");
    match first {
        "multiclass" | "softmax" => format!("multiclass num_class:{}", config.num_class),
        "multiclassova" | "multiclass_ova" | "ova" | "ovr" => format!(
            "multiclassova num_class:{} sigmoid:{}",
            config.num_class,
            // C++ ToString emits the sigmoid as the default-formatted double; the
            // spine uses 1 → "1" matches the golden `sigmoid:1`.
            format_sigmoid(config.sigmoid),
        ),
        _ => config.objective.clone(),
    }
}

/// Format `sigmoid` for the objective line: an integral value prints without a
/// decimal point (`1` not `1.0`), matching the C++ ostream default the golden uses.
fn format_sigmoid(sigmoid: f64) -> String {
    if sigmoid.fract() == 0.0 {
        format!("{}", sigmoid as i64)
    } else {
        format!("{sigmoid}")
    }
}

/// The per-objective default eval metrics (matching the capture's `metric=` list):
/// `[l2, rmse]` for regression, `[l1, l2, rmse]` for regression_l1,
/// `[binary_logloss, binary_error, auc]` for binary, `[multi_logloss]` for
/// multiclass/multiclassova.
fn eval_metrics_for(objective_first_token: &str, config: &Config) -> Vec<EvalMetric> {
    let sigmoid = config.sigmoid;
    match objective_first_token {
        "binary" => vec![
            EvalMetric::Bin(BinaryMetric::BinaryLogloss { sigmoid }),
            EvalMetric::Bin(BinaryMetric::BinaryError { sigmoid }),
            EvalMetric::Bin(BinaryMetric::Auc),
        ],
        "multiclass" | "softmax" => vec![EvalMetric::Multi(MultiLogloss::new(
            ObjectiveKind::Multiclass { num_class: config.num_class },
            config.num_class,
        ))],
        "multiclassova" | "multiclass_ova" | "ova" | "ovr" => {
            vec![EvalMetric::Multi(MultiLogloss::new(
                ObjectiveKind::MulticlassOva { num_class: config.num_class, sigmoid },
                config.num_class,
            ))]
        }
        "regression_l1" | "l1" | "mean_absolute_error" | "mae" => {
            vec![EvalMetric::Reg(Metric::L1), EvalMetric::Reg(Metric::L2), EvalMetric::Reg(Metric::Rmse)]
        }
        _ => vec![EvalMetric::Reg(Metric::L2), EvalMetric::Reg(Metric::Rmse)],
    }
}

/// `feature_names=` for the model text: `Column_0 Column_1 ...` (the LightGBM
/// default when no names are supplied — matching the capture).
fn feature_names(num_features: usize) -> String {
    (0..num_features)
        .map(|j| format!("Column_{j}"))
        .collect::<Vec<_>>()
        .join(" ")
}

/// `feature_infos=` for the model text: `[min:max]` per feature (identity bins
/// `0..num_bin-1`), matching the capture's `[0:5] [0:2]`.
fn feature_infos(corpus: &DenseCorpus, num_features: usize) -> String {
    (0..num_features)
        .map(|j| {
            let mut min = f64::INFINITY;
            let mut max = f64::NEG_INFINITY;
            for row in &corpus.features {
                min = min.min(row[j]);
                max = max.max(row[j]);
            }
            format!("[{}:{}]", min as i64, max as i64)
        })
        .collect::<Vec<_>>()
        .join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builder::TrainingBuilder;

    fn spine_corpus() -> DenseCorpus {
        let f0 = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5];
        let f1 = [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2];
        let features: Vec<Vec<f64>> = (0..12)
            .map(|i| vec![f0[i] as f64, f1[i] as f64])
            .collect();
        let labels = vec![
            2.0f32, 3.0, 5.0, 6.0, 9.0, 10.0, 12.0, 13.0, 16.0, 17.0, 19.0, 20.0,
        ];
        DenseCorpus { features, labels }
    }

    #[test]
    fn public_api_train_predict_round_trip() {
        let cfg = TrainingBuilder::new()
            .objective("regression")
            .num_iterations(10)
            .learning_rate(0.1)
            .num_leaves(4)
            .min_data_in_leaf(1)
            .boost_from_average(true)
            .metric("l2,rmse")
            .seed(1)
            .deterministic(true)
            .build()
            .unwrap();
        let corpus = spine_corpus();
        let booster = train(&cfg, &corpus).expect("train ok");
        // 10 iterations stored.
        assert_eq!(booster.model().num_iteration(), 10);
        assert_eq!(booster.best_iteration, 10);
        // Eval history has l2 + rmse, one value per round.
        assert_eq!(booster.eval_history.len(), 2);
        assert_eq!(booster.eval_history[0].0, "l2");
        assert_eq!(booster.eval_history[0].1.len(), 10);
        // predict round-trips: low-label rows predict lower than high-label rows.
        let p_low = booster.predict_row(&[0.0, 0.0]);
        let p_high = booster.predict_row(&[5.0, 2.0]);
        assert!(p_low[0] < p_high[0], "{} < {}", p_low[0], p_high[0]);
    }

    #[test]
    fn predict_raw_equals_internal_score_open_q2() {
        // Open-Q2/A4: predict(raw_score=True, num_iteration=k) == the internal
        // score_ after k iters, for every training row.
        let cfg = TrainingBuilder::new()
            .objective("regression")
            .num_iterations(10)
            .learning_rate(0.1)
            .num_leaves(4)
            .min_data_in_leaf(1)
            .boost_from_average(true)
            .seed(1)
            .deterministic(true)
            .build()
            .unwrap();
        let corpus = spine_corpus();
        let booster = train(&cfg, &corpus).unwrap();
        for k in 1..=10 {
            let internal = &booster.iter_scores[(k - 1) as usize];
            for (i, row) in corpus.features.iter().enumerate() {
                let raw = booster.predict_row_raw(row, k)[0];
                // The internal score_ folds the init via BoostFromAverage→AddScore;
                // predict_row_raw folds it via AddBias into tree 0. They must agree
                // bit-for-bit on this cell (the GATE for the phase-wide L2 contract).
                assert_eq!(
                    raw.to_bits(),
                    internal[i].to_bits(),
                    "row {i} iter {k}: predict_raw {raw} != internal score_ {} (bit-exact)",
                    internal[i]
                );
            }
        }
    }

    #[test]
    fn build_feature_columns_rejects_non_identity() {
        // A non-consecutive column (missing bin 1) must be rejected.
        let corpus = DenseCorpus {
            features: vec![vec![0.0], vec![2.0]],
            labels: vec![1.0, 2.0],
        };
        assert!(build_feature_columns(&corpus).is_err());
    }
}
