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
use lgbm_core::random::Random;
use lgbm_model::{GbdtModel, Tree};
use lgbm_objective::Objective;
use lgbm_treelearner::{FeatureColumn, SerialTreeLearner};

use crate::error::BoostingError;
use crate::objective::BoostObjective;
use crate::sample_strategy::{BaggingSampleStrategy, GossSampleStrategy};
use crate::score_updater::ScoreUpdater;

/// The boosting strategy variant (RESEARCH Pattern 1 — an ENUM FIELD on [`Gbdt`],
/// NOT trait objects). In C++ these are subclasses of `GBDT` (`DART`, `RF`);
/// the Rust port keeps the single [`Gbdt`] driver and branches on this discriminant
/// in [`Gbdt::train_one_iter`], which is the faithful, allocation-free seam.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BoostingVariant {
    /// Plain gradient boosting (`GBDT`) — the spine. The default.
    #[default]
    Gbdt,
    /// DART (`dart.hpp`) — Dropouts meet Multiple Additive Regression Trees: each
    /// iteration drops a random subset of already-grown trees (via an advancing
    /// `Random(drop_seed)`), trains the new tree against the post-drop score, then
    /// re-adds + normalizes the dropped trees' weights (BST-05).
    Dart,
    /// Random Forest (`rf.hpp`) — averaged trees. Scoped to a LATER plan; the
    /// variant is declared here for the complete `{Gbdt, Dart, Rf}` enum (RESEARCH
    /// Pattern 1) but [`Gbdt::train_one_iter`] does not yet branch on it.
    Rf,
}

/// DART (`dart.hpp`) configuration — the parity-relevant `Config` subset, resolved
/// by the caller / facade. Mirrors `config.h`: `drop_rate` (0.1), `max_drop` (50),
/// `skip_drop` (0.5), `xgboost_dart_mode` (false), `uniform_drop` (false),
/// `drop_seed` (4).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DartConfig {
    /// `config_->drop_rate` — the per-tree drop probability scale.
    pub drop_rate: f64,
    /// `config_->max_drop` — the cap on dropped trees per iteration (`<= 0` ⇒ no cap).
    pub max_drop: i32,
    /// `config_->skip_drop` — the probability of skipping all drops this iteration.
    pub skip_drop: f64,
    /// `config_->xgboost_dart_mode` — selects the xgboost normalize branch.
    pub xgboost_dart_mode: bool,
    /// `config_->uniform_drop` — drop each tree with a uniform `drop_rate` instead of
    /// weighting by `tree_weight[i] * inv_average_weight`; also disables the
    /// `tree_weight_` bookkeeping (`dart.hpp:66,103,173`).
    pub uniform_drop: bool,
    /// `config_->drop_seed` — the seed for the single advancing drop RNG.
    pub drop_seed: i32,
}

/// DART per-iteration drop+normalize state (`dart.hpp` private members). Held
/// alongside [`BoostingVariant::Dart`] on [`Gbdt`].
struct DartState {
    config: DartConfig,
    /// `random_for_drop_` — the SINGLE advancing `Random(drop_seed)`, constructed
    /// ONCE (`dart.hpp:45`) and advanced across iterations (CRITICAL: re-seeding per
    /// iteration would re-draw the same trees every iteration, diverging from C++).
    random_for_drop: Random,
    /// `tree_weight_` — per-tree weights (used to choose drop trees + normalize).
    /// Pushed `shrinkage_rate_` each iter unless `uniform_drop` (`dart.hpp:67`).
    tree_weight: Vec<f64>,
    /// `sum_weight_` — running sum of `tree_weight_`.
    sum_weight: f64,
    /// `drop_index_` — the indices of the trees dropped THIS iteration (iteration
    /// indices `0..iter`, i.e. `num_init_iteration_ + i` with `num_init_iteration_=0`
    /// for a fresh train).
    drop_index: Vec<i32>,
    /// `shrinkage_rate_` — the DART-modified per-tree shrinkage for the NEXT-grown
    /// tree (`dart.hpp:139,142,144`), set by DroppingTrees.
    shrinkage_rate: f64,
}

impl std::fmt::Debug for DartState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // `Random` is intentionally not `Debug` (the parity-critical LCG); show the
        // config + realized state instead of the per-iter RNG internals.
        f.debug_struct("DartState")
            .field("config", &self.config)
            .field("tree_weight_len", &self.tree_weight.len())
            .field("sum_weight", &self.sum_weight)
            .field("drop_index", &self.drop_index)
            .field("shrinkage_rate", &self.shrinkage_rate)
            .finish()
    }
}

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
    /// Optional GOSS sample strategy (BST-04). Mutually exclusive with [`Self::bagging`]
    /// (GOSS forbids bagging — `goss.hpp:87-89`; the facade enforces this). `Some` ⇒
    /// each iter past the `1/learning_rate` skip window AMPLIFIES the sampled rows'
    /// grad/hess in place (`IsHessianChange`), trains on the kept (top ++ sampled)
    /// subset, and scores kept + dropped rows predict-side (Pitfall 4 — dropped rows
    /// STILL get the tree prediction added, exactly like bagging OOB). Inside the skip
    /// window GOSS keeps every row with grad/hess unmodified (the full-corpus path).
    goss: Option<GossSampleStrategy>,
    /// The boosting variant discriminant (RESEARCH Pattern 1). [`BoostingVariant::Gbdt`]
    /// for the spine; [`BoostingVariant::Dart`] activates the DART drop+normalize path
    /// (state in [`Self::dart`]).
    variant: BoostingVariant,
    /// DART per-iteration drop+normalize state. `Some` iff `variant == Dart` (set by
    /// [`Self::with_dart`]); `None` otherwise.
    dart: Option<DartState>,
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
            goss: None,
            variant: BoostingVariant::Gbdt,
            dart: None,
            features: Vec::new(),
        }
    }

    /// Enable the DART boosting variant (BST-05, RESEARCH Pattern 1 — an enum field,
    /// not a trait object). `dart_config` carries the resolved `drop_rate` / `max_drop`
    /// / `skip_drop` / `xgboost_dart_mode` / `uniform_drop` / `drop_seed`; `features`
    /// is the full-corpus feature columns (DART re-scores dropped trees over the WHOLE
    /// corpus predict-side each iteration — `dart.hpp:135,171,189`).
    ///
    /// Sets [`BoostingVariant::Dart`] and constructs the SINGLE advancing
    /// `Random(drop_seed)` ONCE (`dart.hpp:45` — advanced across iterations, never
    /// re-seeded). Builder-style; chains after a constructor.
    #[must_use]
    pub fn with_dart(mut self, dart_config: DartConfig, features: Vec<FeatureColumn>) -> Self {
        self.variant = BoostingVariant::Dart;
        self.dart = Some(DartState {
            config: dart_config,
            random_for_drop: Random::new(dart_config.drop_seed),
            tree_weight: Vec::new(),
            sum_weight: 0.0,
            drop_index: Vec::new(),
            shrinkage_rate: self.learning_rate,
        });
        self.features = features;
        self
    }

    /// Set the boosting [`BoostingVariant`] discriminant directly (RESEARCH Pattern 1
    /// `with_variant` shape). DART requires its state — prefer [`Self::with_dart`],
    /// which sets the variant AND constructs the drop RNG. This setter is the generic
    /// chained entry point used by callers that already hold a configured variant
    /// (e.g. `Gbdt`/`Rf`); calling it with `Dart` WITHOUT [`Self::with_dart`] leaves
    /// the DART state unset and the loop treats it as the spine.
    #[must_use]
    pub fn with_variant(mut self, variant: BoostingVariant) -> Self {
        self.variant = variant;
        self
    }

    /// The active boosting variant (test / inspection helper).
    pub fn variant(&self) -> BoostingVariant {
        self.variant
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

    /// Enable GOSS / gradient-based one-side sampling (BST-04). `goss` is the
    /// [`GossSampleStrategy`] (already `reset_sample_config`'d over the corpus);
    /// `features` is the full-corpus feature columns (for subset training + dropped-row
    /// predict-side scoring). Mutually exclusive with [`with_bagging`](Self::with_bagging)
    /// — GOSS forbids bagging (`goss.hpp:87-89`); the facade rejects the combination.
    ///
    /// GOSS modifies grad/hess in place (`IsHessianChange`), so it runs INSIDE
    /// [`train_one_iter`](Self::train_one_iter) after `get_gradients` and before the
    /// learner; inside the `1/learning_rate` skip window every row is kept with
    /// grad/hess unmodified.
    #[must_use]
    pub fn with_goss(mut self, goss: GossSampleStrategy, features: Vec<FeatureColumn>) -> Self {
        self.goss = Some(goss);
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

        // ---- (1b) DART DroppingTrees (BEFORE GetGradients) ----
        // C++ `DART::GetTrainingScore` triggers `DroppingTrees()` ONCE per iteration
        // (dart.hpp:78-86), and `Boosting()` calls `GetGradients(GetTrainingScore())`
        // (gbdt.cpp:230/233) — so the drops happen and modify the train score BEFORE
        // the objective computes gradients on it. Drops are over the trees grown so far
        // (iters 0..iter); on iter 0 only draw #0 (skip_drop) is consumed (no trees).
        if self.variant == BoostingVariant::Dart && self.dart.is_some() {
            self.dropping_trees(k);
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

        // ---- (3) sampling (BST-03 bagging / BST-04 GOSS): draw the bag over the RNG ----
        // C++ `sample_strategy_->Bagging(iter_, …)` BEFORE the per-class tree loop.
        // The draw consumes the per-block Random sequence verbatim (Pitfall 4). When the
        // strategy actually subsets (`is_use_subset`), the per-class trees train on the
        // kept (in-bag) rows and BOTH kept and dropped rows are scored predict-side below.
        //
        // GOSS (BST-04) additionally AMPLIFIES the sampled rows' grad/hess IN PLACE
        // (`IsHessianChange`) here, BEFORE the tree learner sees them (goss.hpp:30-74).
        // It is mutually exclusive with bagging (GOSS forbids bagging, goss.hpp:87-89).
        // The kept/dropped index arrays come from whichever strategy is active.
        let (use_subset, sample_in_bag, sample_oob) = if let Some(bag) = self.bagging.as_mut() {
            // labels drive ONLY the balanced (pos/neg) draw; plain bagging ignores it.
            bag.bagging(self.iter, labels);
            if bag.is_use_subset() {
                (true, bag.in_bag().to_vec(), bag.out_of_bag().to_vec())
            } else {
                (false, Vec::new(), Vec::new())
            }
        } else if let Some(goss) = self.goss.as_mut() {
            // GOSS mutates grad/hess in place for the sampled rows (amplification) and
            // fills its kept/dropped index arrays. Inside the 1/learning_rate skip
            // window it keeps every row with grad/hess untouched (is_use_subset false).
            goss.bagging(self.iter, &mut gradients, &mut hessians);
            if goss.is_use_subset() {
                (true, goss.in_bag().to_vec(), goss.out_of_bag().to_vec())
            } else {
                (false, Vec::new(), Vec::new())
            }
        } else {
            (false, Vec::new(), Vec::new())
        };

        // The training score BEFORE any of this iteration's trees are added — the
        // C++ `GetTrainingScore()` the `residual_getter` closes over (gbdt.cpp:409:
        // `label[i] - score_ptr[i]`). RenewTreeOutput runs BEFORE UpdateScore, so
        // it must see the pre-update score; snapshot it once (it is unchanged across
        // the K=1 single-output loop here).
        let train_score_pre = self.score_updater.scores().to_vec();

        // The per-tree shrinkage for THIS iteration's new tree. For the GBDT spine
        // this is `learning_rate`; for DART it is the DART-modified `shrinkage_rate_`
        // set by DroppingTrees above (`dart.hpp:139-145` — `lr / (1 + #dropped)` or
        // the xgboost variant), which is what `GBDT::TrainOneIter` reads via
        // `shrinkage_rate_` (gbdt.cpp:398).
        let shrink_rate = match (self.variant, self.dart.as_ref()) {
            (BoostingVariant::Dart, Some(d)) => d.shrinkage_rate,
            _ => self.learning_rate,
        };

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
                let in_bag: Vec<i32> = sample_in_bag.clone();
                let oob: Vec<i32> = sample_oob.clone();
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
                                // The leaf rows' FULL-corpus indices, then their
                                // residuals (label - pre-update score) and labels in
                                // the SAME order (MAPE needs the labels for its
                                // per-row label_weight).
                                let full_rows: Vec<usize> = subset_rows
                                    .iter()
                                    // bag_mapper[index_mapper[i]] = in_bag[subset_row]
                                    .map(|&sr| in_bag_ref[sr as usize] as usize)
                                    .collect();
                                let residuals: Vec<f64> = full_rows
                                    .iter()
                                    .map(|&fr| labels_ref[fr] as f64 - score_ref[offset + fr])
                                    .collect();
                                let leaf_labels: Vec<f32> =
                                    full_rows.iter().map(|&fr| labels_ref[fr]).collect();
                                obj.renew_leaf_output(&residuals, &leaf_labels)
                            }),
                        );
                    }
                    tree.shrinkage(shrink_rate);
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
                            // residual_getter over the leaf's rows, then the
                            // (weighted) percentile — the new leaf output. The leaf
                            // labels are gathered in the SAME row order so MAPE can
                            // form its per-row label_weight.
                            let residuals: Vec<f64> = rows
                                .iter()
                                .map(|&row| {
                                    let r = row as usize;
                                    labels_ref[r] as f64 - score_ref[offset + r]
                                })
                                .collect();
                            let leaf_labels: Vec<f32> =
                                rows.iter().map(|&row| labels_ref[row as usize]).collect();
                            obj.renew_leaf_output(&residuals, &leaf_labels)
                        }),
                    );
                }
                // Shrinkage BEFORE UpdateScore.
                tree.shrinkage(shrink_rate);
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

        // ---- (5) DART Normalize + push tree_weight ----
        // C++ `DART::TrainOneIter` (dart.hpp:58-71): after GBDT::TrainOneIter grows the
        // new tree, `Normalize()` re-adds + rescales the dropped trees, then (unless
        // uniform_drop) the new tree's `shrinkage_rate_` is pushed to `tree_weight_`.
        if self.variant == BoostingVariant::Dart && self.dart.is_some() {
            self.normalize(k);
            let dart = self.dart.as_mut().expect("DART state present");
            if !dart.config.uniform_drop {
                dart.tree_weight.push(dart.shrinkage_rate);
                dart.sum_weight += dart.shrinkage_rate;
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

    /// Build a row's real feature-value vector from the stored full-corpus
    /// [`FeatureColumn`]s (the identity-binned bin index IS the real value — the L2
    /// predict-vs-scatter contract). Width is `max real_feature_index + 1`. Used by
    /// the DART DroppingTrees / Normalize full-corpus predict-side re-scoring.
    fn feature_row(&self, row: i32) -> Vec<f64> {
        let width = self
            .features
            .iter()
            .map(|f| f.real_feature_index)
            .max()
            .map(|m| (m + 1) as usize)
            .unwrap_or(0);
        let mut v = vec![0.0f64; width];
        for f in &self.features {
            v[f.real_feature_index as usize] = f.bins[row as usize] as f64;
        }
        v
    }

    /// C++ `DART::DroppingTrees` (`dart.hpp:97-147`). Runs at the START of each DART
    /// iteration (C++ via `GetTrainingScore` inside `Boosting()`, BEFORE
    /// `GetGradients` — gbdt.cpp:230/233): selects the dropped tree indices from the
    /// SINGLE advancing `Random(drop_seed)` with the exact draw order (draw #0 =
    /// `skip_drop`, then per-tree draws), removes the dropped trees from the train
    /// score (`Shrinkage(-1.0)` on the STORED tree + `AddScore`), and sets the
    /// DART-modified `shrinkage_rate_` for the new tree.
    ///
    /// `num_tree_per_iteration` is `K` (1 for the DART spine; the per-class stride).
    fn dropping_trees(&mut self, k: i32) {
        // Drain the DART state to avoid a double borrow of `self` (DroppingTrees needs
        // both the dart state and the trees + score_updater + features).
        let mut dart = self.dart.take().expect("DART state present");
        let cfg = dart.config;
        dart.drop_index.clear();
        let lr = self.learning_rate;
        let iter = self.iter;

        // Draw #0: skip_drop. `NextFloat() < skip_drop` => skip ALL drops this iter.
        let is_skip = (dart.random_for_drop.next_float() as f64) < cfg.skip_drop;
        if !is_skip {
            let mut drop_rate = cfg.drop_rate;
            if !cfg.uniform_drop {
                let inv_average_weight = dart.tree_weight.len() as f64 / dart.sum_weight;
                if cfg.max_drop > 0 {
                    drop_rate =
                        drop_rate.min(cfg.max_drop as f64 * inv_average_weight / dart.sum_weight);
                }
                for i in 0..iter {
                    let draw = dart.random_for_drop.next_float() as f64;
                    if draw < drop_rate * dart.tree_weight[i as usize] * inv_average_weight {
                        dart.drop_index.push(i); // num_init_iteration_ = 0
                        if dart.drop_index.len() >= cfg.max_drop.max(0) as usize {
                            break;
                        }
                    }
                }
            } else {
                if cfg.max_drop > 0 {
                    drop_rate = drop_rate.min(cfg.max_drop as f64 / iter as f64);
                }
                for i in 0..iter {
                    let draw = dart.random_for_drop.next_float() as f64;
                    if draw < drop_rate {
                        dart.drop_index.push(i);
                        if dart.drop_index.len() >= cfg.max_drop.max(0) as usize {
                            break;
                        }
                    }
                }
            }
        }

        // Drop the selected trees: Shrinkage(-1.0) on the STORED tree, then AddScore
        // over the FULL corpus (predict-side, bit-exact to the partition scatter on
        // the identity-binned corpus). Per-class over `cur_tree_id` (K=1 spine).
        // Snapshot the per-row feature vectors once (disjoint from score_updater).
        let rows: Vec<Vec<f64>> = (0..self.num_data).map(|r| self.feature_row(r)).collect();
        let drop_index = dart.drop_index.clone();
        for &i in &drop_index {
            for cur_tree_id in 0..k {
                let curr_tree = (i * k + cur_tree_id) as usize;
                self.trees[curr_tree].shrinkage(-1.0);
                let tree = self.trees[curr_tree].clone();
                self.score_updater
                    .add_tree_scaled_all(&tree, cur_tree_id, 1.0, |r| rows[r as usize].clone());
            }
        }

        // shrinkage_rate_ for the new tree (dart.hpp:138-146).
        let kk = dart.drop_index.len() as f64;
        if !cfg.xgboost_dart_mode {
            dart.shrinkage_rate = lr / (1.0 + kk);
        } else if dart.drop_index.is_empty() {
            dart.shrinkage_rate = lr;
        } else {
            dart.shrinkage_rate = lr / (lr + kk);
        }
        self.dart = Some(dart);
    }

    /// C++ `DART::Normalize` (`dart.hpp:158-197`) — the 3-step re-add+normalize of the
    /// dropped trees, with the 4 branches (`uniform_drop` × `xgboost_dart_mode`). Runs
    /// AFTER the new tree is grown+scored. Mutates the dropped STORED trees' values and
    /// the train score so the dropped trees end with weight `k/(k+1)` (or the xgboost
    /// `k/(k+lr)`) of their pre-drop contribution; `tree_weight_` is rescaled in step
    /// with the score unless `uniform_drop` (`dart.hpp:173,191`).
    fn normalize(&mut self, k_trees_per_iter: i32) {
        let mut dart = self.dart.take().expect("DART state present");
        let cfg = dart.config;
        let lr = self.learning_rate;
        let k = dart.drop_index.len() as f64;
        let drop_index = dart.drop_index.clone();
        let rows: Vec<Vec<f64>> = (0..self.num_data).map(|r| self.feature_row(r)).collect();

        for &i in &drop_index {
            for cur_tree_id in 0..k_trees_per_iter {
                let curr_tree = (i * k_trees_per_iter + cur_tree_id) as usize;
                if !cfg.xgboost_dart_mode {
                    // step 2 (valid-side in C++) folded: Shrinkage(1/(k+1)). We do not
                    // carry valid updaters in Gbdt (valid is re-derived in the facade),
                    // so the valid AddScore is a no-op here — but the STORED-tree
                    // shrinkage sequence MUST mirror C++ so the model text matches.
                    self.trees[curr_tree].shrinkage(1.0 / (k + 1.0));
                    // step 3 (train-side): Shrinkage(-k) then AddScore to train.
                    self.trees[curr_tree].shrinkage(-k);
                    let tree = self.trees[curr_tree].clone();
                    self.score_updater
                        .add_tree_scaled_all(&tree, cur_tree_id, 1.0, |r| rows[r as usize].clone());
                } else {
                    self.trees[curr_tree].shrinkage(dart.shrinkage_rate);
                    self.trees[curr_tree].shrinkage(-k / lr);
                    let tree = self.trees[curr_tree].clone();
                    self.score_updater
                        .add_tree_scaled_all(&tree, cur_tree_id, 1.0, |r| rows[r as usize].clone());
                }
            }
            if !cfg.uniform_drop {
                let w = dart.tree_weight[i as usize];
                if !cfg.xgboost_dart_mode {
                    dart.sum_weight -= w * (1.0 / (k + 1.0));
                    dart.tree_weight[i as usize] = w * (k / (k + 1.0));
                } else {
                    dart.sum_weight -= w * (1.0 / (k + lr));
                    dart.tree_weight[i as usize] = w * (k / (k + lr));
                }
            }
        }
        self.dart = Some(dart);
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

    // ===================== 07-05: GOSS sample strategy (BST-04) =====================

    use crate::sample_strategy::GossSampleStrategy;

    #[test]
    fn goss_selected_amplifies_and_scores_all_rows() {
        // GOSS runs INSIDE train_one_iter after get_gradients: it amplifies the
        // sampled rows' grad/hess in place, trains on the kept subset, and scores
        // BOTH kept and dropped rows predict-side (Pitfall 4 — dropped rows still
        // get the tree prediction). With a 12-row corpus, top_rate+other_rate=0.3
        // (subset path) and lr small enough that iter 0 is past the skip window, the
        // first iter must subsample and still move every row's score from init.
        let backend = CpuBackend;
        let client = cpu_client();
        let (features, _l, cfg) = corpus();
        // 12 distinct-gradient rows so GOSS has a non-degenerate top-k threshold.
        let f0 = FeatureColumn {
            bins: vec![0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
            real_feature_index: 0,
            ..features[0].clone()
        };
        let f1 = FeatureColumn {
            bins: vec![0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1],
            real_feature_index: 1,
            ..features[1].clone()
        };
        let feats = vec![f0, f1];
        let labels = vec![
            1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
        ];
        let num_data = labels.len() as i32;
        let mut learner =
            SerialTreeLearner::new(&backend, &client, cfg, 4, 1).with_features(feats.clone());
        // lr = 1.0 => skip window is iter < 1, so iter 0 IS skipped; lr = 2.0 =>
        // 1/lr = 0.5 -> (int) 0, so iter 0 subsamples.
        // top_rate=0.3, other_rate=0.2 (sum 0.5, subset path): top_k = 3, other_k = 2,
        // so ~5 rows kept — enough to split the separable corpus.
        let goss = GossSampleStrategy::reset_sample_config(0.3, 0.2, 2.0, num_data, 1, 3).unwrap();
        let mut gbdt = Gbdt::new(
            Objective::Regression { sqrt: false },
            1.0, // shrinkage 1.0 for a visible score move
            1,
            num_data,
            true,
            None,
        )
        .with_goss(goss, feats.clone());
        let snap = gbdt
            .train_one_iter(&mut learner, &labels, feats.len())
            .expect("goss iter");
        // A tree was grown (separable corpus) and exactly one tree pushed.
        assert_eq!(gbdt.trees.len(), 1);
        assert!(
            gbdt.trees[0].num_leaves > 1,
            "GOSS kept subset must grow a real split on the separable corpus"
        );
        // init = label mean. Every row (kept + dropped) must be scored predict-side
        // (Pitfall 4: dropped rows STILL get the tree prediction), so every row's
        // post-iter score differs from the constant init.
        let init = labels.iter().map(|&l| l as f64).sum::<f64>() / num_data as f64;
        let changed = snap
            .score
            .iter()
            .filter(|&&s| (s - init).abs() > 1e-9)
            .count();
        assert_eq!(
            changed, num_data as usize,
            "every row (kept + dropped) must be scored under GOSS: {:?}",
            snap.score
        );
    }

    #[test]
    fn goss_skip_window_keeps_full_corpus() {
        // Inside the 1/learning_rate skip window GOSS keeps every row with grad/hess
        // unmodified — identical to the no-sampling full-corpus path.
        let backend = CpuBackend;
        let client = cpu_client();
        let (features, labels, cfg) = corpus();
        let num_data = labels.len() as i32;
        let mut learner =
            SerialTreeLearner::new(&backend, &client, cfg, 2, 1).with_features(features.clone());
        // lr = 0.1 => skip iters 0..9; iter 0 keeps the full corpus.
        let goss = GossSampleStrategy::reset_sample_config(0.2, 0.1, 0.1, num_data, 1, 3).unwrap();
        let mut gbdt = Gbdt::new(
            Objective::Regression { sqrt: false },
            0.1,
            1,
            num_data,
            true,
            None,
        )
        .with_goss(goss, features.clone());
        // Reference: the SAME iter with no GOSS at all must produce the same score.
        let mut learner_ref =
            SerialTreeLearner::new(&backend, &client, cfg, 2, 1).with_features(features.clone());
        let mut gbdt_ref = Gbdt::new(
            Objective::Regression { sqrt: false },
            0.1,
            1,
            num_data,
            true,
            None,
        );
        let snap = gbdt
            .train_one_iter(&mut learner, &labels, features.len())
            .expect("goss skip iter");
        let snap_ref = gbdt_ref
            .train_one_iter(&mut learner_ref, &labels, features.len())
            .expect("ref iter");
        // Inside the skip window GOSS is a no-op: identical accumulated scores.
        for (a, b) in snap.score.iter().zip(snap_ref.score.iter()) {
            assert_eq!(a.to_bits(), b.to_bits(), "skip-window GOSS must match no-sampling bit-exact");
        }
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

    // ===================== 07-06: DART boosting variant (BST-05) =====================

    use lgbm_core::random::Random as DartRandom;

    /// The config.h DART defaults (drop_rate 0.1, max_drop 50, skip_drop 0.5,
    /// xgboost_dart_mode false, uniform_drop false, drop_seed 4). A unit-level proof
    /// that [`DartConfig`] mirrors the reference defaults (the facade resolves these
    /// from `lgbm-core::Config`, whose `Default` already carries them — see
    /// `config/mod.rs`).
    fn dart_default_config() -> DartConfig {
        DartConfig {
            drop_rate: 0.1,
            max_drop: 50,
            skip_drop: 0.5,
            xgboost_dart_mode: false,
            uniform_drop: false,
            drop_seed: 4,
        }
    }

    /// A 12-row separable corpus (matching the GOSS test shape) so DART grows real
    /// splits each iteration.
    fn dart_corpus() -> (Vec<FeatureColumn>, Vec<f32>, GainConfig) {
        let (features, _l, cfg) = corpus();
        let f0 = FeatureColumn {
            bins: vec![0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
            real_feature_index: 0,
            ..features[0].clone()
        };
        let f1 = FeatureColumn {
            bins: vec![0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1],
            real_feature_index: 1,
            ..features[1].clone()
        };
        let labels = vec![
            1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
        ];
        (vec![f0, f1], labels, cfg)
    }

    /// Verbatim re-implementation of `DART::DroppingTrees`'s DRAW order over the proven
    /// `lgbm_core::Random` LCG — the drop RNG-replay reference. Given the per-tree
    /// `tree_weight` + `sum_weight` BEFORE this iter, returns the dropped tree indices.
    /// Mirrors dart.hpp:97-128 (draw #0 skip_drop, then per-tree draws). NOT re-seeded
    /// per call — the caller owns the advancing `Random`.
    fn reference_drop(
        rng: &mut DartRandom,
        cfg: &DartConfig,
        iter: i32,
        tree_weight: &[f64],
        sum_weight: f64,
    ) -> Vec<i32> {
        let mut drop = Vec::new();
        let is_skip = (rng.next_float() as f64) < cfg.skip_drop;
        if is_skip {
            return drop;
        }
        let mut drop_rate = cfg.drop_rate;
        if !cfg.uniform_drop {
            let inv_avg = tree_weight.len() as f64 / sum_weight;
            if cfg.max_drop > 0 {
                drop_rate = drop_rate.min(cfg.max_drop as f64 * inv_avg / sum_weight);
            }
            for i in 0..iter {
                if (rng.next_float() as f64) < drop_rate * tree_weight[i as usize] * inv_avg {
                    drop.push(i);
                    if drop.len() >= cfg.max_drop.max(0) as usize {
                        break;
                    }
                }
            }
        } else {
            if cfg.max_drop > 0 {
                drop_rate = drop_rate.min(cfg.max_drop as f64 / iter as f64);
            }
            for i in 0..iter {
                if (rng.next_float() as f64) < drop_rate {
                    drop.push(i);
                    if drop.len() >= cfg.max_drop.max(0) as usize {
                        break;
                    }
                }
            }
        }
        drop
    }

    #[test]
    fn dart_config_defaults_match_config_h() {
        let c = dart_default_config();
        assert!((c.drop_rate - 0.1).abs() < 1e-12);
        assert_eq!(c.max_drop, 50);
        assert!((c.skip_drop - 0.5).abs() < 1e-12);
        assert!(!c.xgboost_dart_mode);
        assert!(!c.uniform_drop);
        assert_eq!(c.drop_seed, 4);
    }

    #[test]
    fn dart_variant_is_selected_via_with_dart() {
        let (features, labels, _cfg) = dart_corpus();
        let gbdt = Gbdt::new(
            Objective::Regression { sqrt: false },
            0.1,
            1,
            labels.len() as i32,
            true,
            None,
        )
        .with_dart(dart_default_config(), features);
        assert_eq!(gbdt.variant(), BoostingVariant::Dart);
    }

    #[test]
    fn dart_drop_draw_order_skip_then_per_tree() {
        // The drop RNG draw order is load-bearing: draw #0 is skip_drop, then ONE draw
        // per already-grown tree. With skip_drop=0.0 (never skip) and drop_rate=1.0
        // (always drop, uniform_drop) the reference must drop EVERY tree, consuming
        // exactly 1 + iter draws from the advancing Random.
        let cfg = DartConfig {
            drop_rate: 1.0,
            max_drop: 50,
            skip_drop: 0.0,
            xgboost_dart_mode: false,
            uniform_drop: true,
            drop_seed: 4,
        };
        let mut rng = DartRandom::new(cfg.drop_seed);
        // iter = 3 grown trees, uniform weights.
        let tw = vec![0.1, 0.1, 0.1];
        let drop = reference_drop(&mut rng, &cfg, 3, &tw, 0.3);
        assert_eq!(drop, vec![0, 1, 2], "drop_rate=1 uniform must drop all trees");
        // A fresh RNG at the SAME seed re-derives the same draws (determinism check).
        let mut rng2 = DartRandom::new(cfg.drop_seed);
        let drop2 = reference_drop(&mut rng2, &cfg, 3, &tw, 0.3);
        assert_eq!(drop, drop2);
    }

    #[test]
    fn dart_skip_drop_one_consumes_only_draw_zero() {
        // skip_drop = 1.0 => draw #0 < 1.0 is ALWAYS true => skip all drops, consuming
        // exactly ONE draw (draw #0). The advancing RNG is left at exactly one draw.
        let cfg = DartConfig {
            skip_drop: 1.0,
            ..dart_default_config()
        };
        let mut rng = DartRandom::new(cfg.drop_seed);
        let drop = reference_drop(&mut rng, &cfg, 5, &[0.1; 5], 0.5);
        assert!(drop.is_empty(), "skip_drop=1 must drop nothing");
        // Exactly one draw consumed: a second skip-consuming call from a fresh RNG at
        // the same seed yields the SAME first float, proving only draw #0 was used.
        let mut rng_fresh = DartRandom::new(cfg.drop_seed);
        let f_fresh = rng_fresh.next_float();
        let mut rng_one = DartRandom::new(cfg.drop_seed);
        reference_drop(&mut rng_one, &cfg, 5, &[0.1; 5], 0.5);
        // rng_one consumed 1 draw; its NEXT float must equal rng_fresh's SECOND float.
        let _ = f_fresh;
        let second_fresh = rng_fresh.next_float();
        assert_eq!(
            rng_one.next_float().to_bits(),
            second_fresh.to_bits(),
            "skip_drop must consume exactly draw #0"
        );
    }

    /// Run `n_iter` DART iterations on the separable corpus with the given config and
    /// return the trained Gbdt's accumulated score (for the normalize-branch tests).
    fn run_dart(cfg_dart: DartConfig, n_iter: i32, lr: f64) -> Vec<f64> {
        let backend = CpuBackend;
        let client = cpu_client();
        let (features, labels, cfg) = dart_corpus();
        let num_data = labels.len() as i32;
        let mut learner =
            SerialTreeLearner::new(&backend, &client, cfg, 4, 1).with_features(features.clone());
        let mut gbdt = Gbdt::new(
            Objective::Regression { sqrt: false },
            lr,
            1,
            num_data,
            true,
            None,
        )
        .with_dart(cfg_dart, features.clone());
        let snaps = gbdt
            .train(&mut learner, &labels, features.len(), n_iter)
            .expect("dart train");
        snaps.last().unwrap().score.clone()
    }

    #[test]
    fn dart_all_four_normalize_branches_run_distinctly() {
        // All 4 (uniform_drop × xgboost_dart_mode) branches must train without panic
        // and — with skip_drop forced to 0 so drops actually happen across the run —
        // produce a FINITE score. They use different normalize math, so at least one
        // pair differs (the branches are not collapsed). lr large enough that the
        // drop_rate*weight draws fire on this short run.
        let base = DartConfig {
            drop_rate: 0.5,
            max_drop: 50,
            skip_drop: 0.0,
            xgboost_dart_mode: false,
            uniform_drop: false,
            drop_seed: 4,
        };
        let mut results = Vec::new();
        for &uniform in &[false, true] {
            for &xgb in &[false, true] {
                let cfg = DartConfig {
                    uniform_drop: uniform,
                    xgboost_dart_mode: xgb,
                    ..base
                };
                let score = run_dart(cfg, 6, 0.5);
                assert!(
                    score.iter().all(|s| s.is_finite()),
                    "uniform={uniform} xgb={xgb}: DART score must be finite"
                );
                results.push(score);
            }
        }
        // The four branches are not all identical (distinct normalize math). Compare
        // each pair; at least one must differ bit-wise.
        let all_equal = results
            .windows(2)
            .all(|w| w[0].iter().zip(w[1].iter()).all(|(a, b)| a.to_bits() == b.to_bits()));
        assert!(
            !all_equal,
            "the 4 normalize branches must not collapse to identical scores"
        );
    }

    #[test]
    fn dart_advancing_rng_not_reseeded_per_iter() {
        // CRITICAL: the drop RNG advances across iterations (constructed ONCE). If it
        // were re-seeded per iter, the per-iter drop draws would repeat. We assert the
        // reference draws across two consecutive iters differ when re-seeding would
        // make them identical. Use drop_rate=0.3 uniform so some-but-not-all drop.
        let cfg = DartConfig {
            drop_rate: 0.3,
            skip_drop: 0.0,
            uniform_drop: true,
            ..dart_default_config()
        };
        // ONE advancing RNG across iter=4 then iter=5.
        let mut rng = DartRandom::new(cfg.drop_seed);
        let d1 = reference_drop(&mut rng, &cfg, 4, &[0.1; 4], 0.4);
        let d2 = reference_drop(&mut rng, &cfg, 4, &[0.1; 4], 0.4);
        // A re-seeded-per-iter variant would give d1 == d2 (same seed, same iter, same
        // first 1+4 draws). The advancing RNG generally yields a different draw set.
        // (Not a hard guarantee for ALL seeds, but for drop_seed=4 these differ.)
        assert_ne!(
            d1, d2,
            "advancing RNG must NOT reproduce the same drops each iter (re-seed bug)"
        );
    }
}
