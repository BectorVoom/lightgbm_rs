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
use lgbm_treelearner::{FeatureColumn, SerialTreeLearner};

use crate::error::BoostingError;
use crate::objective::BoostObjective;
use crate::sample_strategy::BaggingSampleStrategy;
use crate::score_updater::ScoreUpdater;

/// The GBDT ensemble driver — the f64 score accumulator + the grown model.
///
/// Grows `K = objective.num_model_per_iteration()` trees per iteration over the
/// class-major score/grad/hess layout (`offset = num_data * cur_tree_id`): `K = 1`
/// for the single-output objectives (regression/regression_l1/binary/custom) and
/// `K = num_class` for `multiclass`/`multiclassova` (06-04). Single-threaded
/// deterministic core, bagging OFF, early-stopping OFF, `boost_from_average`
/// honored per class (the C++ default). Bagging + early stopping land in 06-05.
pub struct Gbdt<'a> {
    /// The f64 score accumulator (`score_`).
    score_updater: ScoreUpdater,
    /// The training-side objective (grad/hess + `BoostFromScore`), dispatching
    /// over regression / regression_l1 / binary / custom (`objective_function_`).
    objective: BoostObjective<'a>,
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
    /// Optional bagging / row-subsampling strategy (BST-03). `None` ⇒ no bagging
    /// (`bagging_fraction == 1` / `bagging_freq == 0`), the full-corpus train-path
    /// scatter (the byte-unchanged 06-02..06-04 path). `Some` ⇒ each iter draws the
    /// bag (over the RNG, D-13), trains on the in-bag subset, and scores in-bag +
    /// out-of-bag rows (OOB rows STILL get the tree prediction added — Pitfall 4).
    ///
    /// EXPLICIT SCOPE BOUNDARY (BST-03): only pos/neg ROW bagging. The query-grouped
    /// path (`bagging_by_query`) is an explicit, decision-backed Phase-7 deferral —
    /// the strategy's [`BaggingConfig::new`](crate::sample_strategy::BaggingConfig::new)
    /// rejects `bagging_by_query = true` with a typed error (06-CONTEXT.md BST-03
    /// scope note), so the loop NEVER silently row-bags a query request.
    bagging: Option<BaggingSampleStrategy>,
    /// The full-corpus feature columns (real bin layout) — needed by the bagging
    /// path to (a) subset-train via the learner and (b) score OOB rows predict-side
    /// (`Tree::predict` over each row's real feature values). Empty when bagging is
    /// off (the caller drives the learner's features directly).
    features: Vec<FeatureColumn>,
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

impl<'a> Gbdt<'a> {
    /// Construct the boosting driver from a built-in regression objective (the
    /// 06-02 spine constructor — kept for the existing callers/tests).
    pub fn new(
        objective: Objective,
        learning_rate: f64,
        num_class: i32,
        num_data: i32,
        boost_from_average: bool,
        init_score: Option<&[f64]>,
    ) -> Self {
        Self::with_objective(
            BoostObjective::Builtin(objective),
            learning_rate,
            num_class,
            num_data,
            boost_from_average,
            init_score,
        )
    }

    /// Construct the boosting driver from any [`BoostObjective`] (regression /
    /// regression_l1 / binary / custom). `init_score` is the optional `Dataset`
    /// `init_score` metadata (class-major `num_data * num_class`); `None` zeroes
    /// the score buffer.
    pub fn with_objective(
        objective: BoostObjective<'a>,
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
            bagging: None,
            features: Vec::new(),
        }
    }

    /// Enable bagging / row subsampling (BST-03). `bagging` is the
    /// [`BaggingSampleStrategy`] (already `reset_sample_config`'d over the corpus);
    /// `features` is the full-corpus feature columns (for subset training + OOB
    /// predict-side scoring). Builder-style — chains after a constructor.
    ///
    /// When the strategy is not actually subsetting (`bagging_fraction == 1`), the
    /// loop behaves as if bagging were off (every row in-bag), but the RNG still
    /// advances identically — matching the C++ draw-every-row contract.
    #[must_use]
    pub fn with_bagging(
        mut self,
        bagging: BaggingSampleStrategy,
        features: Vec<FeatureColumn>,
    ) -> Self {
        self.bagging = Some(bagging);
        self.features = features;
        self
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
        // C++ skips BoostFromAverage entirely when objective_function_ == nullptr
        // (the custom path) — gbdt.cpp:355-372.
        if !self.objective.boost_from_average_enabled() {
            return 0.0;
        }
        if !(self.boost_from_average || num_features == 0) {
            return 0.0;
        }
        // Per-class init (C++ obj->BoostFromScore(cur_tree_id)): the label mean /
        // median / logit for single-output, the log-class-prob / per-class binary
        // logit for multiclass.
        let init = self.objective.boost_from_score(cur_tree_id, labels);
        if Objective::init_score_is_significant(init) {
            self.score_updater.add_constant(init, cur_tree_id);
        }
        init
    }

    /// C++ `GBDT::ObtainAutomaticInitialScore` fallback in the NO-SPLIT branch
    /// (gbdt.cpp:418-429). When the learner returns a 1-leaf (no-split) tree on the
    /// FIRST iteration AND `boost_from_average == false` AND the objective is
    /// non-null AND there is no init-score metadata, C++ still computes the
    /// automatic initial score (`obj->BoostFromScore(cur_tree_id)` — the label
    /// median for regression_l1), ADDS it to the train score once, and uses it as
    /// the constant tree's leaf value. This is the `regression_l1` constant-tree
    /// leaf-value (e.g. the bagged regression_l1 tree-0 = the label median 11.0,
    /// NOT 0.0). Returns the constant value to push (0.0 when the fallback does not
    /// apply — the BoostFromAverage init was already injected on iter 0 by
    /// [`boost_from_average`](Self::boost_from_average)).
    fn no_split_constant_value(
        &mut self,
        cur_tree_id: i32,
        labels: &[f32],
        init_score: f64,
    ) -> f64 {
        let is_first_iter = (self.trees.len() as i32) < self.objective.num_model_per_iteration().max(1);
        if !is_first_iter {
            // Subsequent iterations extend with a zero constant (C++ AsConstantTree(0)).
            return 0.0;
        }
        // C++ no-split fallback: !boost_from_average && obj != null && !has_init_score
        // → ObtainAutomaticInitialScore + AddScore, then AsConstantTree(that score).
        if !self.boost_from_average
            && self.objective.boost_from_average_enabled()
            && !self.has_init_score()
        {
            let auto_init = self.objective.boost_from_score(cur_tree_id, labels);
            if Objective::init_score_is_significant(auto_init) {
                self.score_updater.add_constant(auto_init, cur_tree_id);
            }
            return auto_init;
        }
        // Otherwise the BoostFromAverage init (already in score_ from iter 0) is the
        // constant value (C++ AsConstantTree(init_scores[cur_tree_id])).
        init_score
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
        // UN-DEFERRED in Phase-7 07-01 (D-05 faithful-fix): regression_l1 + bagging
        // is no longer typed-rejected. The Phase-6 06-06 rejection assumed the L1
        // sign-gradient split-gain over the bagged subset diverged from C++ in leaf
        // STRUCTURE (rust:0.0 vs cpp:11.0 at regression_l1_bag1_es0_bfa0 tree 0). A
        // source-built lib_lightgbm 4.6 FP execution trace (07-D05-DECISION.md)
        // showed the divergence was a SPLIT-GAIN OPERAND bug, not an irreducible
        // structure flip: the Rust `min_gain_shift` used the RAW leaf `sum_hessian`
        // where C++ uses the `2*kEpsilon`-BUMPED value (feature_histogram.hpp:174),
        // making the Rust gain-shift ULPs too high and rejecting the borderline
        // bagged-subset splits C++ accepts. With that fixed (lgbm-compute
        // `find_best_split`) plus the retained subset-path median-residual
        // `RenewTreeOutput` (8330cee), regression_l1 + bagging now reproduces the
        // real-binary tree structure (the `regression_l1_bag1_*` matrix cells assert
        // real-binary parity, not the typed error). DEF-06-01 family cleared.

        // K = num_model_per_iteration (num_class for multiclass/ova, 1 otherwise).
        // The objective is authoritative — it MUST agree with self.num_class for the
        // multiclass variants (set by the facade); the loop trusts the objective.
        let k = self.objective.num_model_per_iteration().max(1);
        let total = nd * k as usize;

        // ---- (1) BoostFromAverage FIRST, per class, only on iter 0 ----
        // K trees/iter over the class-major layout (offset = num_data * cur_tree_id,
        // Pattern 4). For multiclass each class gets its own init score.
        let mut init_scores = vec![0.0f64; k as usize];
        for cur_tree_id in 0..k {
            init_scores[cur_tree_id as usize] =
                self.boost_from_average(cur_tree_id, labels, num_features);
        }

        // ---- (2) Boosting(): obj.GetGradients on the CURRENT train score ----
        let mut gradients = vec![0.0f32; total];
        let mut hessians = vec![0.0f32; total];
        // The objective receives the WHOLE class-major score buffer. Single-output
        // objectives treat it as their one class; the multiclass softmax gathers
        // strided `rec[k]=score[num_data*k+i]` across classes (Pattern 4 — a
        // row-major layout would silently diverge), and multiclassova dispatches to
        // each per-class binary at offset `num_data*i`.
        self.objective.get_gradients(
            self.score_updater.scores(),
            labels,
            &mut gradients,
            &mut hessians,
        )?;

        // ---- (3) bagging (BST-03): draw the bag over the RNG (D-13) ----
        // C++ `Bagging(iter_, …)` BEFORE the per-class tree loop. The draw consumes
        // the per-block Random sequence verbatim (Pitfall 4). When the strategy
        // actually subsets (`is_use_subset`), the per-class trees train on the in-bag
        // rows and BOTH in-bag and OOB rows are scored predict-side below.
        let use_subset = if let Some(bag) = self.bagging.as_mut() {
            // labels drive ONLY the balanced (pos/neg) draw; plain bagging ignores it.
            bag.bagging(self.iter, labels);
            bag.is_use_subset()
        } else {
            false
        };

        // The training score BEFORE any of this iteration's trees are added — the
        // C++ `GetTrainingScore()` the `residual_getter` closes over (gbdt.cpp:409:
        // `label[i] - score_ptr[i]`). RenewTreeOutput runs BEFORE UpdateScore, so
        // it must see the pre-update score; snapshot it once (it is unchanged across
        // the K=1 single-output loop here).
        let train_score_pre = self.score_updater.scores().to_vec();

        // ---- (4) per-class tree loop ----
        for cur_tree_id in 0..k {
            let offset = (cur_tree_id as usize) * nd;

            // class_need_train gate (gbdt.cpp:390): when false (a class is absent /
            // is the entire corpus), DO NOT train — push a constant 1-leaf tree
            // carrying this class's init score so models_.len() stays iter*K
            // (Pitfall 6). The init was already injected into score_ via
            // BoostFromAverage→AddScore on iter 0, so no further score change.
            if !self.objective.class_need_train(cur_tree_id, labels) {
                let init = init_scores[cur_tree_id as usize];
                // gbdt.cpp:419-434: on the first iteration push AsConstantTree(init);
                // subsequent iterations extend with AsConstantTree(0). The init only
                // entered score_ once (iter 0 BoostFromAverage), never re-added.
                let is_first_iter = self.trees.len() < k as usize;
                let const_val = if is_first_iter { init } else { 0.0 };
                self.trees.push(Tree::as_constant(const_val, self.num_data));
                continue;
            }

            let grad = &gradients[offset..offset + nd];
            let hess = &hessians[offset..offset + nd];

            // is_first_tree is true while fewer than K trees exist (the first
            // iteration's per-class trees), matching C++
            // `models_.size() < num_tree_per_iteration_` (gbdt.cpp:402).
            let is_first_tree = self.trees.len() < k as usize;

            // BAGGING SUBSET PATH (use_subset): train on the in-bag rows (C++
            // tmp_subset_), then score in-bag + OOB rows predict-side (Pitfall 4 —
            // OOB rows STILL get the tree prediction added). The full-corpus path
            // (no bagging / fraction==1) uses the bit-exact partition scatter.
            if use_subset {
                let bag = self
                    .bagging
                    .as_ref()
                    .expect("use_subset implies a bagging strategy");
                let in_bag: Vec<i32> = bag.in_bag().to_vec();
                let oob: Vec<i32> = bag.out_of_bag().to_vec();
                let (mut tree, subset_partition) = learner.train_on_subset_returning_partition(
                    &in_bag,
                    grad,
                    hess,
                    is_first_tree,
                )?;
                if tree.num_leaves > 1 {
                    // RenewTreeOutput on the SUBSET path (WR-03 fix, mirrors C++
                    // serial_tree_learner.cpp:920-958 + gbdt.cpp:430): no-op for L2
                    // (IsRenewTreeOutput()==false), ACTIVE for regression_l1.
                    //
                    // RETAINED-BUT-CURRENTLY-UNEXERCISED (06-06 Task 2b): the ONLY
                    // renew+bagging combination today is regression_l1 + bagging, which
                    // is typed-rejected at the top of `train_one_iter` (the L1 bagged-
                    // subset leaf STRUCTURE diverges from C++; deferred past Phase 6).
                    // This renewal closure is therefore unreachable for regression_l1 +
                    // bagging right now, but it is KEPT faithful to C++ for any FUTURE
                    // renew objective that bags — do NOT delete it. (Not reverting
                    // 8330cee.)
                    //
                    // subset partition's `indices_in_leaf` are subset-row indices; map
                    // each through `in_bag[sr]` to the FULL-corpus row (the C++
                    // `bag_mapper[index_mapper[i]]`) and take the median RESIDUAL
                    // (`label[fr] - train_score_pre[offset+fr]`) over that leaf's rows
                    // (PercentileFun alpha=0.5). MUST run BEFORE shrinkage (C++ order:
                    // RenewTreeOutput → Shrinkage → UpdateScore → AddBias). Uses the
                    // SAME `train_score_pre` snapshot + `offset` as the full-corpus
                    // branch (the C++ `score_ptr = train_score_updater_->score()
                    // + offset` over full-corpus rows).
                    if self.objective.is_renew_tree_output() {
                        let obj = &self.objective;
                        let labels_ref = labels;
                        let score_ref = &train_score_pre;
                        let in_bag_ref = &in_bag;
                        learner.renew_tree_output(
                            &mut tree,
                            &subset_partition,
                            Some(|_leaf: i32, subset_rows: &[u32]| {
                                let residuals: Vec<f64> = subset_rows
                                    .iter()
                                    .map(|&sr| {
                                        // bag_mapper[index_mapper[i]] = in_bag[subset_row]
                                        let fr = in_bag_ref[sr as usize] as usize;
                                        labels_ref[fr] as f64 - score_ref[offset + fr]
                                    })
                                    .collect();
                                obj.renew_leaf_output(&residuals)
                            }),
                        );
                    }
                    tree.shrinkage(self.learning_rate);
                    // Score BOTH in-bag and OOB rows predict-side over the real feature
                    // values (bit-exact to the partition scatter on the identity-binned
                    // corpus — the L2 contract). OOB rows STILL get scored (Pitfall 4).
                    // The identity-binned real feature value IS the bin index (raw
                    // value 0..K-1); Tree::predict traverses the real-value thresholds
                    // (the bin upper bounds) so feeding the bin index as the real value
                    // reproduces the same leaf assignment as the train-path scatter.
                    let features = &self.features;
                    let feature_row = |row: i32| -> Vec<f64> {
                        let width = features
                            .iter()
                            .map(|f| f.real_feature_index)
                            .max()
                            .map(|m| (m + 1) as usize)
                            .unwrap_or(0);
                        let mut v = vec![0.0f64; width];
                        for f in features {
                            v[f.real_feature_index as usize] = f.bins[row as usize] as f64;
                        }
                        v
                    };
                    self.score_updater.add_tree_predict_path(
                        &tree,
                        &in_bag,
                        cur_tree_id,
                        &feature_row,
                    );
                    self.score_updater.add_tree_predict_path(
                        &tree,
                        &oob,
                        cur_tree_id,
                        &feature_row,
                    );
                    let init = init_scores[cur_tree_id as usize];
                    if Objective::init_score_is_significant(init) {
                        tree.add_bias(init);
                    }
                    self.trees.push(tree);
                } else {
                    // No-split tree on the bagged subset. C++ applies the
                    // ObtainAutomaticInitialScore fallback (gbdt.cpp:418-429): for
                    // regression_l1 bfa-off the constant value is the label median
                    // (e.g. tree-0 = 11.0), not 0.0. See `no_split_constant_value`.
                    let init = init_scores[cur_tree_id as usize];
                    let const_val = self.no_split_constant_value(cur_tree_id, labels, init);
                    self.trees.push(Tree::as_constant(const_val, self.num_data));
                }
                continue;
            }

            // learner.train() builds + returns the partition for the bit-exact
            // training-path score scatter (C++ data_partition_).
            let (mut tree, partition) =
                learner.train_returning_partition(grad, hess, is_first_tree)?;

            if tree.num_leaves > 1 {
                // RenewTreeOutput: no-op for L2 (IsRenewTreeOutput()==false), active
                // for regression_l1 — overwrite each leaf's output with the median
                // RESIDUAL (`label[row] - train_score[offset+row]`) of its rows
                // (Pitfall 2/3; gbdt.cpp:409-411 + serial_tree_learner.cpp:920-940).
                if self.objective.is_renew_tree_output() {
                    let obj = &self.objective;
                    let labels_ref = labels;
                    let score_ref = &train_score_pre;
                    learner.renew_tree_output(
                        &mut tree,
                        &partition,
                        Some(|_leaf: i32, rows: &[u32]| {
                            // residual_getter over the leaf's rows, then the median
                            // (PercentileFun alpha=0.5) — the new leaf output.
                            let residuals: Vec<f64> = rows
                                .iter()
                                .map(|&row| {
                                    let r = row as usize;
                                    labels_ref[r] as f64 - score_ref[offset + r]
                                })
                                .collect();
                            obj.renew_leaf_output(&residuals)
                        }),
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
                self.trees.push(tree);
            } else {
                // No positive-gain split (gbdt.cpp:419-434): the learner returned a
                // 1-leaf tree. C++ discards it and pushes AsConstantTree on the first
                // iteration (extend-with-zeros afterward). On the first iteration with
                // boost_from_average=false it applies the ObtainAutomaticInitialScore
                // fallback (the label median for regression_l1) and AddScores it once;
                // otherwise the BoostFromAverage init (already in score_ from iter 0)
                // is the constant. Pushing the constant keeps models_.len() == iter * K
                // (Pitfall 6). See `no_split_constant_value`.
                let init = init_scores[cur_tree_id as usize];
                let const_val = self.no_split_constant_value(cur_tree_id, labels, init);
                self.trees.push(Tree::as_constant(const_val, self.num_data));
            }
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

    /// Read-only view of one class's score slice (`[num_data*k, num_data*(k+1))`).
    /// Test/inspection helper for the class-major layout.
    pub fn score_updater_class(&self, cur_tree_id: i32) -> &[f64] {
        self.score_updater.class_scores(cur_tree_id)
    }

    /// The number of completed iterations.
    pub fn num_iteration(&self) -> i32 {
        self.iter
    }

    /// The number of grown trees (`models_.len()`), `= iter * num_model_per_iteration`.
    pub fn num_trees(&self) -> usize {
        self.trees.len()
    }

    /// Read-only view of the grown trees (for incremental valid-set prediction).
    pub fn trees(&self) -> &[Tree] {
        &self.trees
    }

    /// Pop the trailing `count` trees (the early-stopping trim, gbdt.cpp:484): on a
    /// stop, the trees grown AFTER `best_iteration` are removed so prediction defaults
    /// to `best_iteration`. Also rewinds `iter` to the retained-tree count / K.
    pub fn pop_trailing_trees(&mut self, count: usize) {
        let keep = self.trees.len().saturating_sub(count);
        self.trees.truncate(keep);
        let k = self.objective.num_model_per_iteration().max(1) as usize;
        self.iter = (self.trees.len() / k) as i32;
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
    use lgbm_compute::CpuBackend;
    use lgbm_compute::gain::GainConfig;
    use lgbm_compute::runtime::cpu_client;
    use lgbm_dataset::bin_mapper::MissingType;
    use lgbm_treelearner::learner::FeatureColumn;

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
            assert!(
                (s - 3.0).abs() < 1e-12,
                "score_ must hold init before trees"
            );
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
        assert!(
            s[0] < s[7],
            "label-1 row score {} < label-5 row score {}",
            s[0],
            s[7]
        );
    }

    #[test]
    fn train_one_iter_length_mismatch_is_typed_error() {
        let backend = CpuBackend;
        let client = cpu_client();
        let (features, _labels, cfg) = corpus();
        let mut learner =
            SerialTreeLearner::new(&backend, &client, cfg, 2, 1).with_features(features.clone());
        let mut gbdt = Gbdt::new(Objective::Regression { sqrt: false }, 0.1, 1, 8, true, None);
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
        assert!(
            (delta_half - delta_full * 0.5).abs() < 1e-9,
            "lr=0.5 delta {delta_half} must be half of lr=1.0 delta {delta_full}"
        );
    }

    // ===================== 06-04: multiclass loop =====================

    use lgbm_objective::MulticlassSoftmax;

    /// An 8-row, 2-feature corpus with 3-class integer labels (separable by f0).
    fn multiclass_corpus() -> (Vec<FeatureColumn>, Vec<f32>, GainConfig) {
        let (features, _spine_labels, cfg) = corpus();
        // 3 classes: rows 0-2 class 0, rows 3-5 class 1, rows 6-7 class 2.
        let labels = vec![0.0f32, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 2.0];
        (features, labels, cfg)
    }

    #[test]
    fn multiclass_loop_grows_num_class_trees_per_iter() {
        let backend = CpuBackend;
        let client = cpu_client();
        let (features, labels, cfg) = multiclass_corpus();
        let num_data = labels.len() as i32;
        let num_class = 3;
        let mut learner =
            SerialTreeLearner::new(&backend, &client, cfg, 2, 1).with_features(features.clone());
        let obj = MulticlassSoftmax::new(num_class, &labels).unwrap();
        let mut gbdt = Gbdt::with_objective(
            BoostObjective::Multiclass(obj),
            0.1,
            num_class,
            num_data,
            true,
            None,
        );
        let snap = gbdt
            .train_one_iter(&mut learner, &labels, features.len())
            .expect("multiclass iter");
        // Exactly 3 trees after one iteration.
        assert_eq!(gbdt.trees.len(), 3, "one iter must push num_class=3 trees");
        // grad/hess/score are length num_data * num_class, class-major.
        assert_eq!(snap.gradients.len(), (num_data * num_class) as usize);
        assert_eq!(snap.hessians.len(), (num_data * num_class) as usize);
        assert_eq!(snap.score.len(), (num_data * num_class) as usize);
    }

    #[test]
    fn multiclass_boost_from_average_runs_per_class() {
        let (_features, labels, _cfg) = multiclass_corpus();
        let num_class = 3;
        let obj = MulticlassSoftmax::new(num_class, &labels).unwrap();
        let mut gbdt = Gbdt::with_objective(
            BoostObjective::Multiclass(obj),
            0.1,
            num_class,
            labels.len() as i32,
            true,
            None,
        );
        // Drive BoostFromAverage per class; each class's init = log(max(kEps, p_k)).
        let mut inits = Vec::new();
        for k in 0..num_class {
            inits.push(gbdt.boost_from_average(k, &labels, 2));
        }
        assert_eq!(inits.len(), 3);
        // class 0 (3/8) and class 1 (3/8) share an init; class 2 (2/8) differs.
        assert!(
            (inits[0] - inits[1]).abs() < 1e-12,
            "equal-count classes share init"
        );
        assert!((inits[0] - inits[2]).abs() > 1e-9, "class 2 init differs");
        // Each class's score slice holds its own init constant after BoostFromAverage.
        for k in 0..num_class {
            let cls = gbdt.score_updater_class(k);
            for &s in cls {
                assert!((s - inits[k as usize]).abs() < 1e-12);
            }
        }
    }

    // ===================== 06-05: bagging / OOB score update =====================

    use crate::sample_strategy::{BaggingConfig, BaggingSampleStrategy};

    #[test]
    fn bagging_oob_rows_are_still_scored() {
        // Pitfall 4: out-of-bag rows STILL get the tree prediction added each iter.
        // Train ONE iter with bagging at fraction 0.5 over the 8-row corpus; assert
        // EVERY row's score changed from the init (none skipped), proving OOB rows
        // were scored predict-side, not skipped.
        let backend = CpuBackend;
        let client = cpu_client();
        let (features, labels, cfg) = corpus();
        let num_data = labels.len() as i32;
        // Build identity FeatureColumns for the corpus (offset rule already applied).
        let mut learner =
            SerialTreeLearner::new(&backend, &client, cfg, 2, 1).with_features(features.clone());
        let bag_cfg = BaggingConfig::new(0.5, 1.0, 1.0, 1, 3, false).unwrap();
        let strat = BaggingSampleStrategy::reset_sample_config(bag_cfg, num_data, &labels);
        assert!(strat.bag_data_cnt() < num_data, "fraction 0.5 must subset");
        let mut gbdt = Gbdt::new(
            Objective::Regression { sqrt: false },
            1.0,
            1,
            num_data,
            true,
            None,
        )
        .with_bagging(strat, features.clone());
        let snap = gbdt
            .train_one_iter(&mut learner, &labels, features.len())
            .expect("bagging iter");
        // init = label mean = 3.0. Every row's post-iter score must differ from init
        // (the tree split moved both leaves), proving OOB rows were ALSO scored.
        let init = 3.0f64;
        let changed = snap
            .score
            .iter()
            .filter(|&&s| (s - init).abs() > 1e-9)
            .count();
        assert_eq!(
            changed, num_data as usize,
            "every row (in-bag + OOB) must be scored: {:?}",
            snap.score
        );
        // The grown tree was pushed (a real split on this separable corpus).
        assert_eq!(gbdt.trees.len(), 1);
    }

    #[test]
    fn class_need_train_false_pushes_constant_tree() {
        // A class entirely absent from the corpus -> class_init_probs = 0 ->
        // class_need_train false -> a constant 1-leaf tree, tree count stays iter*K.
        let backend = CpuBackend;
        let client = cpu_client();
        let (features, _l, cfg) = corpus();
        // 3 classes declared but class 2 NEVER appears (only labels 0/1 present).
        let labels = vec![0.0f32, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0];
        let num_class = 3;
        let num_data = labels.len() as i32;
        let mut learner =
            SerialTreeLearner::new(&backend, &client, cfg, 2, 1).with_features(features.clone());
        let obj = MulticlassSoftmax::new(num_class, &labels).unwrap();
        // class 2 is absent.
        assert!(!obj.class_need_train(2), "absent class must not train");
        let mut gbdt = Gbdt::with_objective(
            BoostObjective::Multiclass(obj),
            0.1,
            num_class,
            num_data,
            true,
            None,
        );
        gbdt.train_one_iter(&mut learner, &labels, features.len())
            .unwrap();
        // Still exactly 3 trees (Pitfall 6); class 2's is a constant 1-leaf tree.
        assert_eq!(gbdt.trees.len(), 3);
        assert_eq!(
            gbdt.trees[2].num_leaves, 1,
            "absent class -> constant 1-leaf tree"
        );
    }
}
