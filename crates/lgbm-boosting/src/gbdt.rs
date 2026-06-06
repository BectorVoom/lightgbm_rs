//! The GBDT boosting loop (`GBDT::TrainOneIter` mirror) + `BoostFromAverage`,
//! ported 1:1 from the C++ reference.
//!
//! Faithful-mirror citations (read directly from the in-tree C++ source):
//! - `LightGBM/src/boosting/gbdt.cpp`:
//!   - `TrainOneIter` (the exact per-iteration order — RESEARCH Pattern 1):
//!     (1) `BoostFromAverage(class)` on iter 0; (2) `Boosting()` →
//!     `obj->GetGradients(train_score, grad, hess)`; (3) bagging (06-05);
//!     (4) per `cur_tree_id`: `learner->Train` → `RenewTreeOutput` →
//!     `tree.Shrinkage(learning_rate)` → `UpdateScore` →
//!     `if |init|>kEpsilon tree.AddBias(init)`; push to `models_`.
//!   - `BoostFromAverage` (gbdt.cpp:319-342): on iter 0 with empty `models_`,
//!     `!has_init_score`, `obj != null`, and `(boost_from_average ||
//!     num_features == 0)`: `init = obj->BoostFromScore(class)`; if
//!     `|init| > kEpsilon` `AddScore(init, class)` to the TRAIN score updater
//!     (and every valid updater — 06-05); return `init` (else 0).
//!   - the per-class loop offset `= cur_tree_id * num_data` (single class K=1
//!     here; per-class is 06-04).
//!
//! **Critical ordering note** (RESEARCH Pattern 1 + Pitfall 5): `Shrinkage` is
//! applied to the tree's leaf/internal values BEFORE `UpdateScore`; `AddBias` is
//! applied AFTER `UpdateScore` and rewrites only the STORED tree values (for model
//! text) — it does NOT touch `score_`. The init score enters `score_` exactly ONCE
//! via `BoostFromAverage → AddScore`, never via `AddBias` (no double-add).

use lgbm_compute::Backend;
use lgbm_model::{GbdtModel, Tree};
use lgbm_objective::Objective;
use lgbm_treelearner::SerialTreeLearner;

use crate::error::BoostingError;
use crate::score_updater::ScoreUpdater;

/// The GBDT ensemble driver — the f64 score accumulator + the grown model.
///
/// 06-02 scope: single-output regression spine (K = 1), single-threaded
/// deterministic core, bagging OFF, early-stopping OFF, `boost_from_average`
/// honored (the C++ regression default). Per-class generalization (multiclass),
/// bagging, and early stopping land in 06-04/06-05.
pub struct Gbdt {
    /// The f64 score accumulator (`score_`).
    score_updater: ScoreUpdater,
    /// The training-side objective (grad/hess + `BoostFromScore`).
    objective: Objective,
    /// `learning_rate` (`config_->learning_rate`) — the per-tree `Shrinkage`.
    learning_rate: f64,
    /// `num_class` (1 for the regression/binary spine).
    num_class: i32,
    /// `num_data_` (rows).
    num_data: i32,
    /// `boost_from_average` (the iter-0 init-score gate).
    boost_from_average: bool,
    /// The accumulated trees (`models_`), flat `[i * ntpi + k]`.
    trees: Vec<Tree>,
    /// `iter_` — the completed-iteration count.
    iter: i32,
}

/// One per-iteration snapshot for the layered goldens: the per-row g/h written
/// this iteration (L1) and the per-class accumulated raw score AFTER the iter (L2).
#[derive(Debug, Clone)]
pub struct IterSnapshot {
    /// Per-row gradients written by the objective this iteration (length
    /// `num_data * num_class`, class-major).
    pub gradients: Vec<f32>,
    /// Per-row hessians written this iteration (same layout).
    pub hessians: Vec<f32>,
    /// The full f64 score buffer AFTER this iteration's `UpdateScore`
    /// (class-major). This is the L2 per-iter accumulated-score golden.
    pub score: Vec<f64>,
}

impl Gbdt {
    /// Construct the boosting driver. `init_score` is the optional `Dataset`
    /// `init_score` metadata (class-major `num_data * num_class`); `None` zeroes
    /// the score buffer.
    pub fn new(
        objective: Objective,
        learning_rate: f64,
        num_class: i32,
        num_data: i32,
        boost_from_average: bool,
        init_score: Option<&[f64]>,
    ) -> Self {
        Self {
            score_updater: ScoreUpdater::new(num_data, num_class, init_score),
            objective,
            learning_rate,
            num_class,
            num_data,
            boost_from_average,
            trees: Vec::new(),
            iter: 0,
        }
    }

    /// Whether this driver already carried `init_score` metadata (C++
    /// `train_score_updater_->has_init_score()`). 06-02 spine has no init-score
    /// metadata; the gate is wired for fidelity.
    fn has_init_score(&self) -> bool {
        // 06-02: init_score metadata is not yet plumbed from the public API; the
        // spine corpus has none, so BoostFromAverage always runs. When the
        // facade plumbs init_score (later wave), thread the flag here.
        false
    }

    /// C++ `GBDT::BoostFromAverage(cur_tree_id, update_scorer=true)`
    /// (gbdt.cpp:319-342): on iter 0 (empty `models_`, no init-score metadata,
    /// `boost_from_average || num_features == 0`), compute `init =
    /// obj.boost_from_score(class)`; if `|init| > kEpsilon` add it to the TRAIN
    /// score updater. Returns `init` (0 if the gate is closed).
    fn boost_from_average(&mut self, cur_tree_id: i32, labels: &[f32], num_features: usize) -> f64 {
        if !self.trees.is_empty() || self.has_init_score() {
            return 0.0;
        }
        if !(self.boost_from_average || num_features == 0) {
            return 0.0;
        }
        let init = self.objective.boost_from_score(labels);
        if Objective::init_score_is_significant(init) {
            self.score_updater.add_constant(init, cur_tree_id);
        }
        init
    }

    /// C++ `GBDT::TrainOneIter` (gbdt.cpp:344-452) for the single-output spine.
    ///
    /// Grows one tree per class (K = 1 here), accumulating it into `score_` via the
    /// bit-exact training-path scatter, and returns an [`IterSnapshot`] (the L1
    /// per-row g/h + the L2 post-iter accumulated score). The caller owns the
    /// `learner` + the per-class `labels`; the learner is re-driven each iteration
    /// with the freshly-computed grad/hess.
    ///
    /// # Errors
    /// [`BoostingError::LengthMismatch`] (V5 boundary) when `labels` disagrees with
    /// `num_data`; objective/learner errors propagate via `#[from]`.
    pub fn train_one_iter<B: Backend>(
        &mut self,
        learner: &mut SerialTreeLearner<'_, B>,
        labels: &[f32],
        num_features: usize,
    ) -> Result<IterSnapshot, BoostingError> {
        // V5: validate lengths up front, before any FP work (T-06-02-02).
        let nd = self.num_data as usize;
        if labels.len() != nd {
            return Err(BoostingError::LengthMismatch {
                expected: nd,
                actual: labels.len(),
            });
        }
        let total = nd * self.num_class.max(1) as usize;

        // ---- (1) BoostFromAverage FIRST, per class, only on iter 0 ----
        // K = 1 here (single-output spine); the loop is class-generalization-ready.
        let mut init_scores = vec![0.0f64; self.num_class.max(1) as usize];
        for cur_tree_id in 0..self.num_class.max(1) {
            init_scores[cur_tree_id as usize] =
                self.boost_from_average(cur_tree_id, labels, num_features);
        }

        // ---- (2) Boosting(): obj.GetGradients on the CURRENT train score ----
        let mut gradients = vec![0.0f32; total];
        let mut hessians = vec![0.0f32; total];
        // Single class (K=1): the whole score buffer is class 0's. (Multiclass in
        // 06-04 strides per class and the objective gathers across classes.)
        self.objective.get_gradients(
            self.score_updater.scores(),
            labels,
            &mut gradients,
            &mut hessians,
        )?;

        // ---- (3) bagging: DEFERRED to 06-05 (no subsetting on the spine) ----

        // ---- (4) per-class tree loop ----
        for cur_tree_id in 0..self.num_class.max(1) {
            let offset = (cur_tree_id as usize) * nd;
            let grad = &gradients[offset..offset + nd];
            let hess = &hessians[offset..offset + nd];

            let is_first_tree = self.trees.len() < self.num_class.max(1) as usize;
            // learner.train() builds + returns the partition for the bit-exact
            // training-path score scatter (C++ data_partition_).
            let (mut tree, partition) =
                learner.train_returning_partition(grad, hess, is_first_tree)?;

            if tree.num_leaves > 1 {
                // RenewTreeOutput: no-op for L2 (IsRenewTreeOutput()==false). The
                // l1 median-residual closure lands with regression_l1 in 06-03.
                if self.objective.is_renew_tree_output() {
                    // The renewal closure is objective-specific (06-03); for the
                    // 06-02 spine objective this branch is never taken.
                    learner.renew_tree_output(
                        &mut tree,
                        &partition,
                        None::<fn(i32, &[u32]) -> f64>,
                    );
                }
                // Shrinkage BEFORE UpdateScore.
                tree.shrinkage(self.learning_rate);
                // UpdateScore: bit-exact training-path per-leaf scatter into score_.
                self.score_updater
                    .add_tree_train_path(learner, &tree, &partition, cur_tree_id);
                // AddBias AFTER UpdateScore — rewrites STORED tree values only
                // (model text), NEVER score_ (Pitfall 5: no double-add).
                let init = init_scores[cur_tree_id as usize];
                if Objective::init_score_is_significant(init) {
                    tree.add_bias(init);
                }
            } else {
                // Degenerate 1-leaf tree (no positive-gain split): no score change
                // beyond the constant already injected by BoostFromAverage. The
                // tree is still pushed so models_.len() == iter * K (Pitfall 6).
            }
            self.trees.push(tree);
        }

        self.iter += 1;
        Ok(IterSnapshot {
            gradients,
            hessians,
            score: self.score_updater.scores().to_vec(),
        })
    }

    /// Run the full boosting loop for `num_iterations` (the spine driver).
    /// Returns the per-iteration [`IterSnapshot`]s (for the layered goldens).
    ///
    /// # Errors
    /// Propagates any [`train_one_iter`](Self::train_one_iter) error.
    pub fn train<B: Backend>(
        &mut self,
        learner: &mut SerialTreeLearner<'_, B>,
        labels: &[f32],
        num_features: usize,
        num_iterations: i32,
    ) -> Result<Vec<IterSnapshot>, BoostingError> {
        let mut snaps = Vec::with_capacity(num_iterations.max(0) as usize);
        for _ in 0..num_iterations.max(0) {
            snaps.push(self.train_one_iter(learner, labels, num_features)?);
        }
        Ok(snaps)
    }

    /// Read-only view of the f64 score buffer (`score_`).
    pub fn scores(&self) -> &[f64] {
        self.score_updater.scores()
    }

    /// The number of completed iterations.
    pub fn num_iteration(&self) -> i32 {
        self.iter
    }

    /// Assemble the grown ensemble into a [`GbdtModel`] for serialization /
    /// predict (the Phase-3 container). `objective_string` is the verbatim
    /// `objective=` line; `max_feature_idx` is `max real_feature_index` across
    /// all features.
    pub fn into_model(
        self,
        objective_string: String,
        max_feature_idx: i32,
        feature_names: String,
        feature_infos: String,
    ) -> GbdtModel {
        GbdtModel {
            trees: self.trees,
            num_class: self.num_class,
            num_tree_per_iteration: self.num_class.max(1),
            label_index: 0,
            max_feature_idx,
            average_output: false,
            objective_string: Some(objective_string),
            feature_names,
            feature_infos,
            monotone_constraints: None,
            trailer: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lgbm_compute::gain::GainConfig;
    use lgbm_compute::runtime::cpu_client;
    use lgbm_compute::CpuBackend;
    use lgbm_treelearner::learner::FeatureColumn;
    use lgbm_dataset::bin_mapper::MissingType;

    /// A tiny 2-feature corpus that produces a real (>1-leaf) split, mirroring the
    /// learner_parity spine shape. 8 rows, 2 binary-ish features.
    fn corpus() -> (Vec<FeatureColumn>, Vec<f32>, GainConfig) {
        // offset MUST come from the authoritative rule (most_freq_bin==0 -> 1),
        // matching the learner_parity spine corpus convention (D-09). Hand-setting
        // offset=0 for a most_freq_bin==0 feature mis-routes the partition.
        let off0 = lgbm_treelearner::offset_for_most_freq_bin(0);
        let f0 = FeatureColumn {
            bins: vec![0, 0, 0, 0, 1, 1, 1, 1],
            num_bin: 2,
            offset: off0,
            min_bin: 0,
            max_bin: 1,
            default_bin: 2,
            most_freq_bin: 0,
            missing_type: MissingType::None,
            bin_upper_bound: vec![0.5, 1.5],
            real_feature_index: 0,
        };
        let f1 = FeatureColumn {
            bins: vec![0, 0, 1, 1, 0, 0, 1, 1],
            num_bin: 2,
            offset: off0,
            min_bin: 0,
            max_bin: 1,
            default_bin: 2,
            most_freq_bin: 0,
            missing_type: MissingType::None,
            bin_upper_bound: vec![0.5, 1.5],
            real_feature_index: 1,
        };
        // Labels separable by feature 0.
        let labels = vec![1.0f32, 1.0, 1.0, 1.0, 5.0, 5.0, 5.0, 5.0];
        let cfg = GainConfig {
            min_data_in_leaf: 1,
            min_sum_hessian_in_leaf: 1e-3,
            max_delta_step: 0.0,
            lambda_l1: 0.0,
            lambda_l2: 0.0,
            min_gain_to_split: 0.0,
            path_smooth: 0.0,
        };
        (vec![f0, f1], labels, cfg)
    }

    #[test]
    fn boost_from_average_enters_via_add_score_not_add_bias() {
        // With boost_from_average=true, after iter 0's BoostFromAverage (before any
        // tree adds) score_ holds init_score (the label mean) for all rows.
        let (features, labels, _cfg) = corpus();
        let mut gbdt = Gbdt::new(
            Objective::Regression { sqrt: false },
            0.1,
            1,
            labels.len() as i32,
            true,
            None,
        );
        // Drive BoostFromAverage directly (before any tree) and check score_.
        let init = gbdt.boost_from_average(0, &labels, features.len());
        // label mean = (4*1 + 4*5)/8 = 24/8 = 3.0.
        assert!((init - 3.0).abs() < 1e-12);
        for &s in gbdt.scores() {
            assert!((s - 3.0).abs() < 1e-12, "score_ must hold init before trees");
        }
        // No trees yet -> AddBias has not run.
        assert_eq!(gbdt.num_iteration(), 0);
    }

    #[test]
    fn score_accumulation_is_f64_via_train_path() {
        let backend = CpuBackend;
        let client = cpu_client();
        let (features, labels, cfg) = corpus();
        let num_data = labels.len() as i32;
        let mut learner =
            SerialTreeLearner::new(&backend, &client, cfg, 2, 1).with_features(features.clone());
        let mut gbdt = Gbdt::new(
            Objective::Regression { sqrt: false },
            1.0, // learning_rate = 1.0 so the first tree fully fits the residual
            1,
            num_data,
            true,
            None,
        );
        let snap = gbdt
            .train_one_iter(&mut learner, &labels, features.len())
            .expect("train one iter");
        // score_ is f64.
        let _s: &[f64] = gbdt.scores();
        // After iter 0: init (3.0) was added via AddScore; then the first tree's
        // shrunk (lr=1.0) leaf outputs were scattered. With a perfect split on
        // feature 0, the left leaf rows (label 1) and right leaf rows (label 5)
        // should move the score toward their labels.
        assert_eq!(snap.score.len(), num_data as usize);
        // The grad at iter 0 is score-label = 3 - label; the tree fits -grad-ish.
        // The post-iter score for the label-1 rows should be < the label-5 rows.
        let s = &snap.score;
        assert!(s[0] < s[7], "label-1 row score {} < label-5 row score {}", s[0], s[7]);
    }

    #[test]
    fn train_one_iter_length_mismatch_is_typed_error() {
        let backend = CpuBackend;
        let client = cpu_client();
        let (features, _labels, cfg) = corpus();
        let mut learner =
            SerialTreeLearner::new(&backend, &client, cfg, 2, 1).with_features(features.clone());
        let mut gbdt = Gbdt::new(
            Objective::Regression { sqrt: false },
            0.1,
            1,
            8,
            true,
            None,
        );
        let bad_labels = vec![1.0f32; 7]; // wrong length
        let err = gbdt
            .train_one_iter(&mut learner, &bad_labels, features.len())
            .unwrap_err();
        assert!(matches!(err, BoostingError::LengthMismatch { .. }));
    }

    #[test]
    fn shrinkage_before_update_score() {
        // learning_rate scales the leaf output that is scattered into score_.
        // With lr=0.5 the score delta from the tree is half of lr=1.0's delta
        // (plus the same init). Verify the shrinkage is applied to the SCATTERED
        // value (i.e. before UpdateScore), not after.
        let backend = CpuBackend;
        let client = cpu_client();
        let (features, labels, cfg) = corpus();
        let num_data = labels.len() as i32;

        let run = |lr: f64| -> Vec<f64> {
            let mut learner = SerialTreeLearner::new(&backend, &client, cfg, 2, 1)
                .with_features(features.clone());
            let mut gbdt = Gbdt::new(
                Objective::Regression { sqrt: false },
                lr,
                1,
                num_data,
                true,
                None,
            );
            let snap = gbdt
                .train_one_iter(&mut learner, &labels, features.len())
                .unwrap();
            snap.score
        };
        let s1 = run(1.0);
        let s_half = run(0.5);
        // Both start from init=3.0. The tree delta for row 0 at lr=0.5 must be
        // exactly half the delta at lr=1.0.
        let delta_full = s1[0] - 3.0;
        let delta_half = s_half[0] - 3.0;
        assert!((delta_half - delta_full * 0.5).abs() < 1e-9,
            "lr=0.5 delta {delta_half} must be half of lr=1.0 delta {delta_full}");
    }
}
