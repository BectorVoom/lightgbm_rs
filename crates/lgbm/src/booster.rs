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

    /// C++ `factor_to_bigger_better` — `+1` for AUC, `-1` for the losses. Drives
    /// the early-stopping comparison direction (BST-07).
    fn factor_to_bigger_better(&self) -> f64 {
        match self {
            EvalMetric::Reg(m) => m.factor_to_bigger_better(),
            EvalMetric::Bin(m) => m.factor_to_bigger_better(),
            // multi_logloss is a loss: lower is better.
            EvalMetric::Multi(_) => -1.0,
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

/// Train a [`Booster`] with an optional validation set (BST-07 early stopping +
/// MET-02 valid-metric eval history). When `config.early_stopping_round > 0` a
/// validation set is REQUIRED (else a typed error); when bagging is configured
/// (`bagging_freq > 0 && bagging_fraction < 1`) the per-iter loop draws the bag
/// over the RNG (D-13) and scores in-bag + OOB rows.
///
/// # Errors
/// [`LgbmError`] for an unsupported objective/metric, an invalid corpus,
/// early stopping without a valid set, `bagging_by_query = true` (Phase-7
/// deferral), or a loop/learner failure — never a panic (Security V5).
pub fn train_with_valid(
    config: &Config,
    corpus: &DenseCorpus,
    valid: &DenseCorpus,
) -> Result<Booster, LgbmError> {
    let first = config.objective.split_whitespace().next().unwrap_or("");
    let (boost_obj, transformed_labels): (BoostObjective<'static>, Vec<f32>) = match first {
        "binary" => {
            let b = Binary::new(config.sigmoid).map_err(LgbmError::Objective)?;
            (BoostObjective::Binary(b), corpus.labels.clone())
        }
        "multiclass" | "softmax" => {
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
            let o = Objective::from_config(config).map_err(LgbmError::Objective)?;
            let lbl = o.transform_labels(&corpus.labels);
            (BoostObjective::Builtin(o), lbl)
        }
    };
    let metrics = eval_metrics_for(first, config);
    train_inner_full(config, corpus, Some(valid), boost_obj, transformed_labels, metrics)
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
/// Delegates to [`train_inner_full`] with no validation set / no early stopping
/// (the 06-02..06-04 behavior, preserved byte-for-byte).
fn train_inner(
    config: &Config,
    corpus: &DenseCorpus,
    boost_obj: BoostObjective<'_>,
    labels: Vec<f32>,
    metrics: Vec<EvalMetric>,
) -> Result<Booster, LgbmError> {
    train_inner_full(config, corpus, None, boost_obj, labels, metrics)
}

/// The full training driver (06-05): per-iteration loop with bagging (BST-03),
/// metric-eval cadence (`metric_freq`, multi-metric, `is_provide_training_metric`;
/// MET-02), early stopping over an optional validation set (BST-07), and the
/// trailing-tree pop. When `valid` is `None`, `early_stopping_round == 0`,
/// `bagging_freq == 0` / `bagging_fraction == 1`, this reproduces the prior-wave
/// loop exactly.
fn train_inner_full(
    config: &Config,
    corpus: &DenseCorpus,
    valid: Option<&DenseCorpus>,
    boost_obj: BoostObjective<'_>,
    labels: Vec<f32>,
    metrics: Vec<EvalMetric>,
) -> Result<Booster, LgbmError> {
    use lgbm_boosting::{BaggingConfig, BaggingSampleStrategy, EarlyStopping, EvalSnapshot, MetricSpec};

    let objective_string = canonical_objective_string(config);
    let objective_kind = ObjectiveKind::parse(&objective_string)
        .unwrap_or(ObjectiveKind::Regression { sqrt: false });

    // ---- identity-binned feature columns ----
    let features = build_feature_columns(corpus)?;
    let num_data = corpus.features.len() as i32;
    let num_class = config.num_class.max(1);
    let num_features = features.len();
    let max_feature_idx = features
        .iter()
        .map(|f| f.real_feature_index)
        .max()
        .unwrap_or(-1);

    // ---- the learner ----
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
    .with_features(features.clone());

    // ---- the GBDT loop (optionally with bagging BST-03 / GOSS BST-04) ----
    // GOSS (data_sample_strategy=goss) and bagging are mutually exclusive — GOSS
    // forbids bagging (goss.hpp:87-89). The `boosting=goss` alias-expansion sets
    // data_sample_strategy=goss + boosting=gbdt (set.rs:472-476).
    let goss_on = config.data_sample_strategy == "goss";
    let bagging_on =
        !goss_on && config.bagging_freq > 0 && config.bagging_fraction < 1.0;
    let mut gbdt = Gbdt::with_objective(
        boost_obj,
        config.learning_rate,
        num_class,
        num_data,
        config.boost_from_average,
        None,
    );
    if goss_on {
        // GOSSStrategy::ResetSampleConfig CHECKs (top+other<=1, both>0) surface as a
        // typed Result. The per-block RNG seed base is bagging_seed (goss.hpp:97).
        let goss = lgbm_boosting::GossSampleStrategy::reset_sample_config(
            config.top_rate,
            config.other_rate,
            config.learning_rate,
            num_data,
            num_class,
            config.bagging_seed,
        )
        .map_err(LgbmError::Boosting)?;
        gbdt = gbdt.with_goss(goss, features.clone());
    } else if bagging_on {
        // BaggingConfig::new rejects bagging_by_query=true (explicit Phase-7 deferral).
        let bag_cfg = BaggingConfig::new(
            config.bagging_fraction,
            config.pos_bagging_fraction,
            config.neg_bagging_fraction,
            config.bagging_freq,
            config.bagging_seed,
            config.bagging_by_query,
        )
        .map_err(LgbmError::Boosting)?;
        let strat = BaggingSampleStrategy::reset_sample_config(bag_cfg, num_data, &labels);
        gbdt = gbdt.with_bagging(strat, features.clone());
    }

    // ---- early stopping setup (BST-07) ----
    let es_enabled = config.early_stopping_round > 0;
    if es_enabled && valid.is_none() {
        return Err(LgbmError::Boosting(
            lgbm_boosting::BoostingError::EarlyStoppingWithoutValidSet,
        ));
    }
    // The validation set's metric values drive the stop decision; we maintain an
    // incremental f64 valid score accumulator (class-major) updated each iter.
    let (valid_feat_rows, valid_labels): (Vec<Vec<f64>>, Vec<f32>) = match valid {
        Some(v) => (v.features.clone(), v.labels.clone()),
        None => (Vec::new(), Vec::new()),
    };
    let valid_nd = valid_feat_rows.len() as i32;
    // class-major valid score buffer (num_class blocks of valid_nd).
    let mut valid_score = vec![0.0f64; (valid_nd.max(0) as usize) * num_class as usize];
    let mut prev_tree_count = 0usize;
    // Inject the valid set's boost_from_average init (C++ BoostFromAverage adds the
    // init to EVERY valid score updater). The training init is folded into tree 0's
    // leaves via AddBias, so re-predicting trees over the valid rows already includes
    // it — we therefore re-derive valid scores by predicting the grown trees per iter
    // (which carry the AddBias-folded init), needing no separate init injection.

    let metric_specs: Vec<MetricSpec> = metrics
        .iter()
        .map(|m| MetricSpec {
            name: m.name(),
            factor_to_bigger_better: m.factor_to_bigger_better(),
        })
        .collect();
    let mut early = EarlyStopping::new(
        config.early_stopping_round,
        config.early_stopping_min_delta,
        config.first_metric_only,
        num_class,
        if valid.is_some() { 1 } else { 0 },
        metric_specs,
    );

    // eval history: training metrics (when is_provide_training_metric) + valid metrics.
    let provide_train = config.is_provide_training_metric || valid.is_none();
    let metric_freq = config.metric_freq.max(1);
    let mut train_eval_history: Vec<(String, Vec<f64>)> = metrics
        .iter()
        .map(|m| (format!("training {}", m.name()), Vec::new()))
        .collect();
    let mut valid_eval_history: Vec<(String, Vec<f64>)> = metrics
        .iter()
        .map(|m| (format!("valid_0 {}", m.name()), Vec::new()))
        .collect();
    // The legacy training-only eval history (06-02..06-04 keyed by bare metric name).
    let mut legacy_eval_history: Vec<(String, Vec<f64>)> = metrics
        .iter()
        .map(|m| (m.name(), Vec::new()))
        .collect();

    let mut iter_scores: Vec<Vec<f64>> = Vec::new();
    let mut iter_grad_hess: Vec<(Vec<f32>, Vec<f32>)> = Vec::new();

    let total_iters = config.num_iterations.max(0);
    let mut ran_iters = 0i32;
    for it in 0..total_iters {
        let snap: IterSnapshot = gbdt
            .train_one_iter(&mut learner, &labels, num_features)
            .map_err(LgbmError::Boosting)?;
        ran_iters = it + 1;
        iter_scores.push(snap.score.clone());
        iter_grad_hess.push((snap.gradients.clone(), snap.hessians.clone()));

        // Incremental valid-score update: predict the trees grown THIS iter over the
        // valid rows (class-major), adding to the running valid_score.
        if valid_nd > 0 {
            let trees = gbdt.trees();
            for (t_idx, tree) in trees.iter().enumerate().skip(prev_tree_count) {
                let cur_tree_id = (t_idx % num_class as usize) as i32;
                let off = (cur_tree_id as usize) * valid_nd as usize;
                for (r, row) in valid_feat_rows.iter().enumerate() {
                    valid_score[off + r] += tree.predict(row);
                }
            }
            prev_tree_count = trees.len();
        }

        // Metric eval cadence (metric_freq gate; MET-02). Always eval on the LAST
        // iter and on every freq multiple, matching the C++ OutputMetric cadence.
        //
        // CR-02 (06-06): `metric_freq` gates ONLY the recorded eval-HISTORY (the
        // pushes into train/valid/legacy_eval_history and any future logging). The
        // valid-score eval that FEEDS the early-stop DECISION + the `early.update`
        // call run EVERY iteration when ES is on, INDEPENDENT of metric_freq —
        // mirroring gbdt.cpp:574 where the valid-metric+ES block is
        // `if (need_output || early_stopping_round_ > 0)` and `need_output`
        // (= `iter % metric_freq == 0`) only guards the `Log::Info`. The pre-06-06
        // port gated `early.update` behind `do_eval`, so a `metric_freq > 1` run
        // skipped ES evaluation on the off-cadence iters and diverged from C++ on
        // best_iteration / trailing-trim.
        let do_eval = (it + 1) % metric_freq == 0 || it + 1 == total_iters;

        // Training metrics: history is metric_freq-gated (MET-02 unchanged).
        if do_eval && provide_train {
            for (mi, m) in metrics.iter().enumerate() {
                let v = m.eval(&snap.score, &corpus.labels)?;
                train_eval_history[mi].1.push(v);
                legacy_eval_history[mi].1.push(v);
            }
        }

        // Valid metrics: eval whenever we either record history (`do_eval`) OR need
        // the ES decision this iter (`es_enabled`). Push to valid_eval_history ONLY
        // on `do_eval` (so metric_freq still thins the RECORDED history per MET-02
        // and the metric_freq_thins_eval_history test stays green); feed `row` to
        // `early.update` whenever ES is on (EVERY iter — the CR-02 fix).
        if valid_nd > 0 && (do_eval || es_enabled) {
            let mut row = Vec::with_capacity(metrics.len());
            for (mi, m) in metrics.iter().enumerate() {
                let v = m.eval(&valid_score, &valid_labels)?;
                if do_eval {
                    valid_eval_history[mi].1.push(v);
                }
                row.push(v);
            }
            if es_enabled {
                let stop = early.update(it, &EvalSnapshot { values: vec![row] });
                if stop {
                    break;
                }
            }
        }
    }

    // best_iteration + trailing-tree pop (BST-07).
    let best_iteration = if es_enabled {
        let pop = early.trailing_trees_to_pop(ran_iters);
        if pop > 0 {
            gbdt.pop_trailing_trees(pop as usize);
        }
        early.best_iteration()
    } else {
        gbdt.num_iteration()
    };

    // Assemble the public eval_history: legacy bare-name training metrics (so the
    // 06-02..06-04 replay keys still resolve) PLUS the valid_0 metrics when present.
    let mut eval_history: Vec<(String, Vec<f64>)> = legacy_eval_history;
    if valid_nd > 0 {
        eval_history.extend(valid_eval_history);
    }
    let _ = train_eval_history; // training metrics are captured in legacy keys above

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
    fn early_stopping_fires_and_pops_trailing_trees() {
        // Train with a valid set that plateaus quickly so early stopping FIRES before
        // num_iterations; best_iteration < num_iterations and the trailing trees are
        // popped (model tree count == best_iteration).
        let train_corpus = spine_corpus();
        // A valid set whose labels are CONSTANT so the metric plateaus fast (the
        // model can't keep improving valid l2 after the first couple of trees).
        let valid_corpus = DenseCorpus {
            features: spine_corpus().features,
            labels: vec![10.0f32; 12], // constant => valid metric plateaus
        };
        let cfg = TrainingBuilder::new()
            .objective("regression")
            .num_iterations(30)
            .learning_rate(0.3)
            .num_leaves(4)
            .min_data_in_leaf(1)
            .boost_from_average(true)
            .early_stopping_round(2)
            .seed(1)
            .deterministic(true)
            .build()
            .unwrap();
        let booster = train_with_valid(&cfg, &train_corpus, &valid_corpus).unwrap();
        assert!(
            booster.best_iteration < 30,
            "early stopping must fire before num_iterations (best_iteration={})",
            booster.best_iteration
        );
        assert!(booster.best_iteration >= 1);
        // The model's tree count == best_iteration (trailing trees popped).
        assert_eq!(
            booster.model().trees.len() as i32,
            booster.best_iteration,
            "trailing trees must be popped to best_iteration"
        );
        // valid_0 metrics are in the eval history (MET-02).
        assert!(
            booster.eval_history.iter().any(|(n, _)| n.starts_with("valid_0")),
            "valid metrics must be recorded: {:?}",
            booster.eval_history.iter().map(|(n, _)| n.clone()).collect::<Vec<_>>()
        );
    }

    #[test]
    fn early_stopping_without_valid_set_is_typed_error() {
        let cfg = TrainingBuilder::new()
            .objective("regression")
            .num_iterations(10)
            .early_stopping_round(3)
            .num_leaves(4)
            .min_data_in_leaf(1)
            .seed(1)
            .deterministic(true)
            .build()
            .unwrap();
        let err = train(&cfg, &spine_corpus()).unwrap_err();
        assert!(matches!(err, LgbmError::Boosting(_)), "got {err:?}");
    }

    #[test]
    fn bagging_by_query_rejected_via_facade() {
        // bagging_by_query=true must surface as a typed error (Phase-7 deferral),
        // not silently row-bag.
        let mut base = Config::default();
        base.objective = "regression".into();
        base.num_iterations = 5;
        base.num_leaves = 4;
        base.min_data_in_leaf = 1;
        base.bagging_freq = 1;
        base.bagging_fraction = 0.7;
        base.bagging_by_query = true;
        let cfg = TrainingBuilder::new().from_config(base).build().unwrap();
        let err = train(&cfg, &spine_corpus()).unwrap_err();
        assert!(matches!(err, LgbmError::Boosting(_)), "got {err:?}");
    }

    #[test]
    fn metric_freq_thins_eval_history() {
        // metric_freq=3 over 9 iters => training metrics recorded on iters 3,6,9
        // (the cadence gate), i.e. 3 values, not 9.
        let cfg = TrainingBuilder::new()
            .objective("regression")
            .num_iterations(9)
            .learning_rate(0.1)
            .num_leaves(4)
            .min_data_in_leaf(1)
            .boost_from_average(true)
            .metric_freq(3)
            .seed(1)
            .deterministic(true)
            .build()
            .unwrap();
        let booster = train(&cfg, &spine_corpus()).unwrap();
        // legacy l2 history thinned to the freq cadence.
        let (_n, vals) = booster
            .eval_history
            .iter()
            .find(|(n, _)| n == "l2")
            .unwrap();
        assert_eq!(vals.len(), 3, "metric_freq=3 over 9 iters => 3 recorded rounds");
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
