//! The GBDT boosting loop (`GBDT::TrainOneIter` mirror) + `BoostFromAverage`,
//! ported 1:1 from the C++ reference.
//!
//! Faithful-mirror citations (read directly from the in-tree C++ source):
//! - `LightGBM/src/boosting/gbdt.cpp`:
//!   - `TrainOneIter` (the exact per-iteration order):
//!     (1) `BoostFromAverage(class)` on iter 0; (2) `Boosting()` →
//!     `obj->GetGradients(train_score, grad, hess)`; (3) bagging;
//!     (4) per `cur_tree_id`: `learner->Train` → `RenewTreeOutput` →
//!     `tree.Shrinkage(learning_rate)` → `UpdateScore` →
//!     `if |init|>kEpsilon tree.AddBias(init)`; push to `models_`.
//!   - `BoostFromAverage` (gbdt.cpp:319-342): on iter 0 with empty `models_`,
//!     `!has_init_score`, `obj != null`, and `(boost_from_average ||
//!     num_features == 0)`: `init = obj->BoostFromScore(class)`; if
//!     `|init| > kEpsilon` `AddScore(init, class)` to the TRAIN score updater
//!     (and every valid updater); return `init` (else 0).
//!   - the per-class loop offset `= cur_tree_id * num_data` (single class K=1
//!     here).
//!
//! **Critical ordering note**: `Shrinkage` is
//! applied to the tree's leaf/internal values BEFORE `UpdateScore`; `AddBias` is
//! applied AFTER `UpdateScore` and rewrites only the STORED tree values (for model
//! text) — it does NOT touch `score_`. The init score enters `score_` exactly ONCE
//! via `BoostFromAverage → AddScore`, never via `AddBias` (no double-add).

use lgbm_compute::Backend;
// GBDT-owned train-lifetime resident score. `ResidentScore`
// is generic over the runtime `R`; the GBDT struct is NOT generic over a `Backend`, so
// the instance is stored type-erased (`Box<dyn Any>`) and downcast to
// `ResidentScore<B::Runtime>` inside the already-`B`-generic methods (one `B` per train).
// This names lgbm-compute TYPES only — never a GPU compute runtime.
use lgbm_compute::ResidentScore;
use lgbm_dataset::LeafPartitionLayout;
use std::any::Any;
use lgbm_core::random::Random;
use lgbm_model::{GbdtModel, Tree};
use lgbm_objective::Objective;
use lgbm_treelearner::{FeatureColumn, GradientDiscretizer, SerialTreeLearner};

/// Quantize one class's `grad`/`hess` IN PLACE to their de-quantized int8 values.
/// `is_constant_hessian` is inferred from the buffer (L2 → constant, binary/etc → varies)
/// so no objective surgery is needed. Each row's grad/hess is replaced by `int8 × scale` (f32),
/// the quantized value the learner then builds histograms from. Deterministic rounding only.
fn quantize_grad_hess_in_place(
    grad: &mut [f32],
    hess: &mut [f32],
    num_grad_quant_bins: i32,
    stochastic: bool,
    seed: u64,
) {
    if grad.is_empty() {
        return;
    }
    let is_constant_hessian = hess.iter().all(|&h| h == hess[0]);
    let mut d = if stochastic {
        GradientDiscretizer::new_stochastic(num_grad_quant_bins, is_constant_hessian, seed)
    } else {
        GradientDiscretizer::new(num_grad_quant_bins, is_constant_hessian)
    };
    let q = d.discretize(grad, hess); // i8 pairs [hess, grad]
    let (gs, hs) = (d.grad_scale(), d.hess_scale());
    for i in 0..grad.len() {
        grad[i] = (f64::from(q[2 * i + 1]) * gs) as f32;
        hess[i] = (f64::from(q[2 * i]) * hs) as f32;
    }
}

/// Soft-threshold for the L1-regularized leaf output (C++ `Common::ThresholdL1`): shrink the
/// gradient sum toward 0 by `l1`. For `l1 == 0` it is the identity (the common case).
fn threshold_l1(sum_grad: f64, l1: f64) -> f64 {
    if sum_grad > l1 {
        sum_grad - l1
    } else if sum_grad < -l1 {
        sum_grad + l1
    } else {
        0.0
    }
}

use crate::error::BoostingError;
use crate::objective::BoostObjective;
use crate::sample_strategy::{BaggingSampleStrategy, GossSampleStrategy};
use crate::score_updater::ScoreUpdater;
// Resident cross-iteration score loop: the per-leaf `AddScore` device delegate walks a
// `PredictTree` reconstructed from the grown `Tree` + the spine's `FeatureColumn`s.
// Symbol only — the walk stays generic over `B::Runtime` via the score-updater's
// `*_on` methods (no GPU runtime pulled in).
use lgbm_compute::kernels::predict::PredictTree;

/// The resolved Random Forest `Boosting()` output: the per-class init scores plus
/// the ONCE-derived class-major gradients and hessians (`rf.hpp:90-109`
/// `init_scores_`/`gradients_`/`hessians_`). Factored into a type alias to keep the
/// [`Gbdt::rf_boosting`] signature readable.
type RfBoosting = (Vec<f64>, Vec<f32>, Vec<f32>);

/// The boosting strategy variant (an ENUM FIELD on [`Gbdt`], NOT trait objects).
/// In C++ these are subclasses of `GBDT` (`DART`, `RF`);
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
    /// re-adds + normalizes the dropped trees' weights.
    Dart,
    /// Random Forest (`rf.hpp`) — AVERAGED (not accumulated) trees with MANDATORY
    /// randomization (bagging or feature sub-sampling) and NO learning-rate
    /// shrinkage. Each iteration trains one tree per class on a fresh bagged subset
    /// against grad/hess derived ONCE from a constant `BoostFromAverage` init score
    /// (`rf.hpp:90-109`), renews the leaf outputs to the mean residual `label -
    /// init_score` (`rf.hpp:149-152`), and folds the tree into `score_` as a running
    /// average via `MultiplyScore(iter); AddScore; MultiplyScore(1/(iter+1))`
    /// (`rf.hpp:157-159`). Prediction divides the tree sum by `num_iteration`
    /// (`average_output_`, gbdt_prediction.cpp:57-59). State in [`Gbdt::rf`].
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

/// Random Forest (`rf.hpp`) configuration — the parity-relevant `Config` subset
/// for the two mandatory-randomization checks (`rf.hpp:35-40`). RF requires EITHER
/// active row bagging OR active feature sub-sampling; the facade resolves these
/// from `Config` and the [`Gbdt`] driver enforces them at the top of the RF path.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RfConfig {
    /// Whether row bagging is active for this run (`bagging_freq > 0 && 0 <
    /// bagging_fraction < 1`). One of `bagging_active` / `feature_subsampling_active`
    /// MUST be true (`rf.hpp:36-37`).
    pub bagging_active: bool,
    /// Whether feature sub-sampling is active (`0 < feature_fraction < 1`). The
    /// alternative randomization source to bagging (`rf.hpp:37`).
    pub feature_subsampling_active: bool,
}

/// Random Forest per-train state (`rf.hpp` private members). Held alongside
/// [`BoostingVariant::Rf`] on [`Gbdt`]. RF computes grad/hess ONCE from a constant
/// init-score buffer (`rf.hpp:90-109 Boosting()`) and reuses them every iteration —
/// the trees differ only through the per-iteration bagged subset, not the gradient
/// distribution (that is the "random forest of regression trees on a fixed target"
/// semantics). The per-tree leaf outputs are RENEWED to the mean residual
/// `label - init_score` and the score buffer is kept as a running AVERAGE.
struct RfState {
    config: RfConfig,
    /// `init_scores_` (`rf.hpp:94-97,230`) — the per-class constant init score
    /// (`BoostFromAverage(class, update_scorer=false)`), the target the per-tree
    /// residual getter (`label - init_scores_[k]`) closes over (`rf.hpp:149-150`).
    init_scores: Vec<f64>,
    /// `gradients_` — computed ONCE in [`Gbdt::rf_boosting`] from the constant
    /// init-score buffer (`rf.hpp:90-109`) and reused every iteration (NOT
    /// re-derived from the accumulated score). Class-major (`num_data * K`).
    gradients: Vec<f32>,
    /// `hessians_` — the once-computed hessians (same layout as `gradients`).
    hessians: Vec<f32>,
}

impl std::fmt::Debug for RfState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RfState")
            .field("config", &self.config)
            .field("init_scores", &self.init_scores)
            .field("gradients_len", &self.gradients.len())
            .field("hessians_len", &self.hessians.len())
            .finish()
    }
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
/// `K = num_class` for `multiclass`/`multiclassova`. Single-threaded
/// deterministic core, `boost_from_average` honored per class (the C++ default).
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
    /// Optional bagging / row-subsampling strategy. `None` ⇒ no bagging
    /// (`bagging_fraction == 1` / `bagging_freq == 0`), the full-corpus train-path
    /// scatter. `Some` ⇒ each iter draws the
    /// bag (over the RNG), trains on the in-bag subset, and scores in-bag +
    /// out-of-bag rows (OOB rows STILL get the tree prediction added).
    ///
    /// SCOPE BOUNDARY: only pos/neg ROW bagging. The query-grouped
    /// path (`bagging_by_query`) is out of scope —
    /// the strategy's [`BaggingConfig::new`](crate::sample_strategy::BaggingConfig::new)
    /// rejects `bagging_by_query = true` with a typed error, so the loop NEVER
    /// silently row-bags a query request.
    bagging: Option<BaggingSampleStrategy>,
    /// Optional GOSS sample strategy. Mutually exclusive with [`Self::bagging`]
    /// (GOSS forbids bagging — `goss.hpp:87-89`; the facade enforces this). `Some` ⇒
    /// each iter past the `1/learning_rate` skip window AMPLIFIES the sampled rows'
    /// grad/hess in place (`IsHessianChange`), trains on the kept (top ++ sampled)
    /// subset, and scores kept + dropped rows predict-side (dropped rows
    /// STILL get the tree prediction added, exactly like bagging OOB). Inside the skip
    /// window GOSS keeps every row with grad/hess unmodified (the full-corpus path).
    goss: Option<GossSampleStrategy>,
    /// The boosting variant discriminant. [`BoostingVariant::Gbdt`]
    /// for the spine; [`BoostingVariant::Dart`] activates the DART drop+normalize path
    /// (state in [`Self::dart`]).
    variant: BoostingVariant,
    /// DART per-iteration drop+normalize state. `Some` iff `variant == Dart` (set by
    /// [`Self::with_dart`]); `None` otherwise.
    dart: Option<DartState>,
    /// Random Forest per-train state. `Some` iff `variant == Rf` (set by
    /// [`Self::with_rf`]); `None` otherwise. Holds the once-computed grad/hess + the
    /// per-class init scores (`rf.hpp` `gradients_`/`hessians_`/`init_scores_`).
    rf: Option<RfState>,
    /// The full-corpus feature columns (real bin layout) — needed by the bagging
    /// path to (a) subset-train via the learner and (b) score OOB rows predict-side
    /// (`Tree::predict` over each row's real feature values). Empty when bagging is
    /// off (the caller drives the learner's features directly).
    features: Vec<FeatureColumn>,
    /// `config_->use_quantized_grad`. When true, each iteration's gradients
    /// and hessians are quantized in-place (deterministic int8) before the learner — the
    /// opt-in APPROXIMATE mode. Default false → the exact path is untouched.
    use_quantized_grad: bool,
    /// `config_->num_grad_quant_bins` (MUST be ≤254 for the i8 discretized storage).
    num_grad_quant_bins: i32,
    /// `config_->stochastic_rounding` (C++ default true). Functional (seeded, reproducible);
    /// the deterministic path is the C++ parity gate. Used only if `use_quantized_grad`.
    quant_stochastic: bool,
    /// `config_->quant_train_renew_leaf`. After a quantized tree's structure is built, recompute
    /// its leaf outputs from the ORIGINAL (non-quantized) gradients — `RenewIntGradTreeOutput`.
    /// Used only if `use_quantized_grad`.
    quant_renew_leaf: bool,
    /// `lambda_l1` / `lambda_l2` for the renewed-leaf formula `-ThresholdL1(ΣG,l1)/(ΣH+l2)`.
    quant_l1: f64,
    quant_l2: f64,
    /// `config_->linear_tree`. When true (and raw features are supplied via
    /// [`Self::with_linear_tree`]), each non-first tree gets per-leaf linear models
    /// fitted after growth (`LinearTreeLearner::CalculateLinear`). The first tree of
    /// the ensemble stays constant.
    linear_tree: bool,
    /// `config_->linear_lambda` — L2 penalty on the per-leaf linear coefficients.
    linear_lambda: f64,
    /// Row-major raw feature matrix (`num_data * num_features`, `f64`), indexed by
    /// ORIGINAL feature index — required by the linear-tree leaf fit. `None` ⇒ the
    /// linear-tree path is inactive even if `linear_tree` is set.
    raw_features: Option<Vec<f64>>,
    /// The GBDT-owned, TRAIN-LIFETIME resident f64 score
    /// buffer ([`lgbm_compute::ResidentScore`]<`B::Runtime`>), type-erased because
    /// `Gbdt` is not `Backend`-generic (downcast inside the `B`-generic methods). Built
    /// ONCE — lazily, on the first `train_one_iter` grad/hess call, uploading the host
    /// score AFTER `BoostFromAverage` — and accumulated in place across trees
    /// ([`ResidentScore::add_tree_on_device`], mirrored alongside the host scatter). The
    /// on-device grad/hess branch reads its Handle directly, so the host
    /// score is never round-tripped to compute grad/hess. `None`
    /// whenever [`Self::resident_active`] is `Some(false)` (the whole out-of-envelope
    /// class) — never constructed there, so those trains are byte-unchanged.
    ///
    /// The public `scores()` accessor and the `train_score_pre` snapshots used by
    /// `RenewTreeOutput` (spine and multiclass loops) and by the RF path always read
    /// the host-mirrored `score_` buffer directly; only the on-device grad/hess branch
    /// reads the resident Handle.
    resident_score: Option<Box<dyn Any>>,
    /// The ONCE-evaluated activation decision for [`Self::resident_score`]
    /// (`None` = not yet decided; `Some(true/false)` = decided on the first
    /// `train_one_iter`). The guard is `num_class == 1 && bagging.is_none() &&
    /// goss.is_none() && device_objective_supported(objective.name()) && boosting_on_cuda()`
    /// — bagging/GOSS/multiclass are excluded entirely rather than
    /// tracking a variable-length resident buffer. Evaluated once and frozen for the whole
    /// train (bagging/goss/num_class are construction constants; `boosting_on_cuda` is set
    /// before train), so a mid-train change never re-derives a stale length assumption.
    resident_active: Option<bool>,
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
    /// Whether this boost round EMITTED a tree set and advanced the iteration
    /// counter. `false` on a NON-FIRST no-split bagged round that C++
    /// `GBDT::TrainOneIter` POPS (gbdt.cpp:440-446, the `!should_continue` path):
    /// the would-be 1-leaf constant tree(s) are discarded, `self.iter` is NOT
    /// advanced, and the next boost round re-bags (fresh RNG draw) and retries.
    /// On such a round no tree was added and `score` is the UNCHANGED pre-round
    /// score; the driver must NOT count it as an emitted iteration (no duplicate
    /// `iter_scores` push, no `iter_grad_hess` push). On the FIRST iteration C++
    /// KEEPS the no-split constant baseline (e.g. `regression_l1_bag1` Tree=0) and
    /// advances, so `emitted` is `true` there.
    pub emitted: bool,
}

impl<'a> Gbdt<'a> {
    /// Construct the boosting driver from a built-in regression objective
    /// (kept for the existing callers/tests).
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
            rf: None,
            features: Vec::new(),
            use_quantized_grad: false,
            num_grad_quant_bins: 4,
            quant_stochastic: false,
            quant_renew_leaf: false,
            quant_l1: 0.0,
            quant_l2: 0.0,
            linear_tree: false,
            linear_lambda: 0.0,
            raw_features: None,
            resident_score: None,
            resident_active: None,
        }
    }

    /// Enable linear-tree training: after each non-first tree's structure is grown,
    /// fit per-leaf linear models over the RAW features (C++ `LinearTreeLearner`).
    /// `raw_features` is row-major `num_data * num_features` (`f64`), indexed by
    /// ORIGINAL feature index. No-op unless `enabled`.
    #[must_use]
    pub fn with_linear_tree(
        mut self,
        enabled: bool,
        linear_lambda: f64,
        raw_features: Vec<f64>,
    ) -> Self {
        self.linear_tree = enabled;
        self.linear_lambda = linear_lambda;
        self.raw_features = if enabled { Some(raw_features) } else { None };
        self
    }

    /// Enable `quant_train_renew_leaf`: after each quantized tree is grown, recompute its leaf
    /// outputs from the ORIGINAL gradients via `-ThresholdL1(ΣG,l1)/(ΣH+l2)`. No-op unless
    /// `use_quantized_grad` is also on. `l1`/`l2` are `lambda_l1`/`lambda_l2`.
    #[must_use]
    pub fn with_quant_renew_leaf(mut self, enabled: bool, l1: f64, l2: f64) -> Self {
        self.quant_renew_leaf = enabled;
        self.quant_l1 = l1;
        self.quant_l2 = l2;
        self
    }

    /// Enable the opt-in `use_quantized_grad` APPROXIMATE training mode: each
    /// iteration's gradients/hessians are quantized to int8 before the learner. `num_grad_quant_bins`
    /// MUST be ≤254 (i8 storage). `stochastic` selects stochastic rounding (C++ default) vs
    /// deterministic (the C++ parity gate). The default exact path is unaffected unless `enabled`.
    #[must_use]
    pub fn with_quantized_grad(
        mut self,
        enabled: bool,
        num_grad_quant_bins: i32,
        stochastic: bool,
    ) -> Self {
        self.use_quantized_grad = enabled;
        self.num_grad_quant_bins = num_grad_quant_bins;
        self.quant_stochastic = stochastic;
        self
    }

    /// Enable the DART boosting variant (an enum field,
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

    /// Enable the Random Forest boosting variant (an
    /// enum field, not a trait object). `rf_config` carries the resolved
    /// mandatory-randomization flags (`bagging_active` / `feature_subsampling_active`);
    /// `features` is the full-corpus feature columns (RF re-scores each grown tree
    /// over the WHOLE corpus predict-side to keep the running-average score buffer,
    /// `rf.hpp:157-159`, exactly like the bagging OOB path).
    ///
    /// Sets [`BoostingVariant::Rf`], `average_output` (no shrinkage — RF averages),
    /// and the per-train [`RfState`]. The two RF CHECKs (`rf.hpp:35-40,91-93`) are
    /// enforced at the top of the RF train path in [`Self::train_one_iter`] (iter 0)
    /// so they surface as a typed [`BoostingError::RfConfig`] rather than a panic.
    /// Builder-style; chains after a constructor. The caller MUST also enable bagging
    /// via [`Self::with_bagging`] when `bagging_active` is true (RF reuses the
    /// bit-exact [`BaggingSampleStrategy`] RNG).
    #[must_use]
    pub fn with_rf(mut self, rf_config: RfConfig, features: Vec<FeatureColumn>) -> Self {
        self.variant = BoostingVariant::Rf;
        self.rf = Some(RfState {
            config: rf_config,
            init_scores: Vec::new(),
            gradients: Vec::new(),
            hessians: Vec::new(),
        });
        // RF re-scores grown trees over the whole corpus (running average); keep the
        // feature columns even when bagging also sets them (idempotent).
        if !features.is_empty() {
            self.features = features;
        }
        self
    }

    /// Set the boosting [`BoostingVariant`] discriminant directly.
    /// DART requires its state — prefer [`Self::with_dart`],
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

    /// Enable bagging / row subsampling. `bagging` is the
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

    /// Enable GOSS / gradient-based one-side sampling. `goss` is the
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
    /// `train_score_updater_->has_init_score()`). The spine has no init-score
    /// metadata; the gate is wired for fidelity.
    fn has_init_score(&self) -> bool {
        // init_score metadata is not yet plumbed from the public API; the
        // spine corpus has none, so BoostFromAverage always runs. When the
        // facade plumbs init_score, thread the flag here.
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
        // Validate lengths up front, before any FP work.
        let nd = self.num_data as usize;
        if labels.len() != nd {
            return Err(BoostingError::LengthMismatch {
                expected: nd,
                actual: labels.len(),
            });
        }
        // regression_l1 + bagging is not typed-rejected: `find_best_split`'s
        // `min_gain_shift` uses the `2*kEpsilon`-BUMPED `sum_hessian`
        // (feature_histogram.hpp:174), not the RAW leaf `sum_hessian` — using the raw
        // value made the Rust gain-shift ULPs too high and rejected borderline
        // bagged-subset splits that C++ accepts. With that fixed, regression_l1 +
        // bagging reproduces the real-binary tree structure.

        // K = num_model_per_iteration (num_class for multiclass/ova, 1 otherwise).
        // The objective is authoritative — it MUST agree with self.num_class for the
        // multiclass variants (set by the facade); the loop trusts the objective.
        let k = self.objective.num_model_per_iteration().max(1);
        let total = nd * k as usize;

        // ---- RANDOM FOREST branch (rf.hpp) ----
        // RF is structurally distinct from the GBDT spine: grad/hess are derived ONCE
        // from a constant init-score buffer (not the accumulated score), trees are
        // AVERAGED (running-average score buffer) instead of accumulated with a
        // learning rate, and leaf outputs are renewed to the mean residual `label -
        // init_score`. Handle the whole RF iteration here and return — keeping the
        // spine path below byte-unchanged.
        if self.variant == BoostingVariant::Rf && self.rf.is_some() {
            return self.train_one_iter_rf(learner, labels, num_features, k, nd, total);
        }

        // ---- (1) BoostFromAverage FIRST, per class, only on iter 0 ----
        // K trees/iter over the class-major layout (offset = num_data * cur_tree_id).
        // For multiclass each class gets its own init score.
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

        // ---- resident-score activation decision ----
        // Evaluate the envelope guard ONCE and freeze it for the whole train.
        // Only the num_class==1 GBDT spine with no bagging/GOSS and an on-device-supported
        // objective, under the env-gated `boosting_on_cuda_` toggle, keeps the score
        // resident and computes grad/hess on device; every other case (multiclass, bagging,
        // GOSS, DART/RF, unsupported objective, or the env unset) takes the byte-unchanged
        // host path below — never a silent row-count mismatch.
        if self.resident_active.is_none() {
            let active = self.num_class == 1
                && self.bagging.is_none()
                && self.goss.is_none()
                && self.variant == BoostingVariant::Gbdt
                // Linear trees fit + score on the HOST (`add_linear_tree_train_path`
                // writes `self.score` directly); the resident device score buffer is
                // NOT synced by that path, so its gradients would go stale after the
                // first linear tree. Force the host path when linear_tree is on.
                && !self.linear_tree
                && lgbm_compute::device_objective_supported(self.objective.name())
                && self.boosting_on_cuda();
            self.resident_active = Some(active);
        }

        // ---- (2) Boosting(): obj.GetGradients on the CURRENT train score ----
        // Allocated lazily per arm: the resident device arm REPLACES the Vecs with the
        // readback wholesale (`gradients = g`), so pre-zeroing 2×`total` f32 there was
        // pure waste; the host arms need the pre-sized zero buffers for `get_gradients`.
        let mut gradients: Vec<f32>;
        let mut hessians: Vec<f32>;
        // The objective receives the WHOLE class-major score buffer. Single-output
        // objectives treat it as their one class; the multiclass softmax gathers
        // strided `rec[k]=score[num_data*k+i]` across classes (a
        // row-major layout would silently diverge), and multiclassova dispatches to
        // each per-class binary at offset `num_data*i`.
        // Time per-iter grad/hess compute into the whole-train budget.
        if self.resident_active == Some(true) {
            // ON-DEVICE grad/hess from the resident score Handle — no host score snapshot.
            // Build the GBDT-owned
            // ResidentScore ONCE (lazily, uploading the host score AFTER BoostFromAverage so
            // the iter-0 init constant is captured); update_score_resident then accumulates
            // it in place across trees, so the score is uploaded exactly once per train.
            let client = learner.client();
            if self.resident_score.is_none() {
                let rs = ResidentScore::<B::Runtime>::from_host_scores(
                    client,
                    self.score_updater.scores(),
                );
                self.resident_score = Some(Box::new(rs));
            }
            let rs = self
                .resident_score
                .as_ref()
                .expect("resident_score constructed above")
                .downcast_ref::<ResidentScore<B::Runtime>>()
                .expect("resident_score is ResidentScore<B::Runtime> for this train's backend");
            match self
                .objective
                .get_gradients_resident_on::<B>(client, rs, nd, labels)
            {
                // The bit-exact envelope (L2/L1/binary): device-in/device-out grad/hess,
                // read back to host to feed the (host) tree learner. Bit-exact to the host
                // objective on the cpu-f64 anchor (`objective_parity_{regression,binary}`).
                Some(res) => {
                    let (g_handle, h_handle) =
                        res.map_err(|e| BoostingError::TreeLearner(e.into()))?;
                    let (g, h) = lgbm_treelearner::phase_prof::time(
                        &lgbm_treelearner::phase_prof::GRAD_NS,
                        || {
                            (
                                lgbm_compute::read_handle_f32(client, &g_handle),
                                lgbm_compute::read_handle_f32(client, &h_handle),
                            )
                        },
                    );
                    gradients = g;
                    hessians = h;
                }
                // Supported-but-not-bit-exact (huber/fair/quantile/poisson) — the resident
                // score stays maintained, but grad/hess falls back to the host path so the
                // bit-exact contract is never traded for a within-tol device kernel.
                None => {
                    gradients = vec![0.0f32; total];
                    hessians = vec![0.0f32; total];
                    lgbm_treelearner::phase_prof::time(
                        &lgbm_treelearner::phase_prof::GRAD_NS,
                        || {
                            self.objective.get_gradients(
                                self.score_updater.scores(),
                                labels,
                                &mut gradients,
                                &mut hessians,
                            )
                        },
                    )?;
                }
            }
        } else {
            gradients = vec![0.0f32; total];
            hessians = vec![0.0f32; total];
            lgbm_treelearner::phase_prof::time(&lgbm_treelearner::phase_prof::GRAD_NS, || {
                self.objective.get_gradients(
                    self.score_updater.scores(),
                    labels,
                    &mut gradients,
                    &mut hessians,
                )
            })?;
        }

        // ---- (3) sampling (bagging / GOSS): draw the bag over the RNG ----
        // C++ `sample_strategy_->Bagging(iter_, …)` BEFORE the per-class tree loop.
        // The draw consumes the per-block Random sequence verbatim. When the
        // strategy actually subsets (`is_use_subset`), the per-class trees train on the
        // kept (in-bag) rows and BOTH kept and dropped rows are scored predict-side below.
        //
        // GOSS additionally AMPLIFIES the sampled rows' grad/hess IN PLACE
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
        let train_score_pre = lgbm_treelearner::phase_prof::time(
            &lgbm_treelearner::phase_prof::SNAPSHOT_NS,
            || self.score_updater.scores().to_vec(),
        );

        // The per-tree shrinkage for THIS iteration's new tree. For the GBDT spine
        // this is `learning_rate`; for DART it is the DART-modified `shrinkage_rate_`
        // set by DroppingTrees above (`dart.hpp:139-145` — `lr / (1 + #dropped)` or
        // the xgboost variant), which is what `GBDT::TrainOneIter` reads via
        // `shrinkage_rate_` (gbdt.cpp:398).
        let shrink_rate = match (self.variant, self.dart.as_ref()) {
            (BoostingVariant::Dart, Some(d)) => d.shrinkage_rate,
            _ => self.learning_rate,
        };

        // ---- (3b) quantize gradients (use_quantized_grad) ----
        // APPROXIMATE mode: replace each class's grad/hess in place with their
        // de-quantized int8 quantization, so the UNCHANGED learner builds the quantized model.
        // Runs AFTER GetGradients + bagging/GOSS amplification (mirroring C++ DiscretizeGradients
        // at the start of the learner's Train), and only here — the exact path never enters this
        // branch. Deterministic rounding only.
        // Preserve the ORIGINAL (non-quantized) grad/hess for leaf renewal (quant_train_renew_leaf)
        // — the in-place quantization below would otherwise overwrite them.
        let orig_grad_hess: Option<(Vec<f32>, Vec<f32>)> =
            if self.use_quantized_grad && self.quant_renew_leaf {
                Some((gradients.clone(), hessians.clone()))
            } else {
                None
            };
        if self.use_quantized_grad {
            for cur_tree_id in 0..k as usize {
                let off = cur_tree_id * nd;
                // Per-(iter, class) seed: stochastic rounding varies each tree yet repeats for a
                // fixed run (deterministic=true reproducibility).
                let seed = ((self.iter as u64) << 20) ^ (cur_tree_id as u64) ^ 0x5170_2010;
                quantize_grad_hess_in_place(
                    &mut gradients[off..off + nd],
                    &mut hessians[off..off + nd],
                    self.num_grad_quant_bins,
                    self.quant_stochastic,
                    seed,
                );
            }
        }

        // ---- (4) per-class tree loop ----
        // C++ `GBDT::TrainOneIter` `should_continue` (gbdt.cpp:386/407): set true the
        // moment ANY class's grown tree has `num_leaves > 1`. If NO class splits this
        // round, C++ POPS this round's pushed trees (when there were already trees —
        // i.e. NOT the first iteration) and does NOT advance `iter_` (gbdt.cpp:440-446).
        let mut should_continue = false;
        // `models_.size()` BEFORE this round's pushes (gbdt.cpp:402/421/442). The
        // first iteration has `tree_count_before == 0`; the C++ pop guard
        // `models_.size() > num_tree_per_iteration_` (= `tree_count_before > 0` after
        // pushing K) keeps the first-iteration no-split constant baseline.
        let tree_count_before = self.trees.len();
        for cur_tree_id in 0..k {
            let offset = (cur_tree_id as usize) * nd;

            // class_need_train gate (gbdt.cpp:390): when false (a class is absent /
            // is the entire corpus), DO NOT train — push a constant 1-leaf tree
            // carrying this class's init score so models_.len() stays iter*K.
            // The init was already injected into score_ via
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
            // tmp_subset_), then score in-bag + OOB rows predict-side —
            // OOB rows STILL get the tree prediction added. The full-corpus path
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
                    // A real split this class → C++ `should_continue = true`
                    // (gbdt.cpp:407): this round emits trees and advances `iter_`.
                    should_continue = true;
                    // RenewTreeOutput on the SUBSET path, mirrors C++
                    // serial_tree_learner.cpp:920-958 + gbdt.cpp:430: no-op for L2
                    // (IsRenewTreeOutput()==false), ACTIVE for regression_l1.
                    //
                    // This renewal closure is kept faithful to C++ for any FUTURE
                    // renew objective that bags — do NOT delete it.
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
                    // LINEAR TREE on the bagging subset (C++ `LinearTreeLearner` +
                    // bagging). When linear_tree is on the corpus is continuous, so the
                    // whole bagging score update must run over the RAW feature values
                    // (the bin-index `feature_row` below is only valid for the
                    // identity-binned corpus). We remap the subset partition's
                    // subset-row indices to full-corpus rows (`in_bag[sr]`) so both the
                    // fit and the in-bag score index the full-corpus raw/grad/hess
                    // directly. The linear leaves are fit on non-first trees only; the
                    // first (constant) tree still scores through the SAME raw path
                    // (`add_linear_tree_train_path` falls back to `leaf_value` when a
                    // tree carries no linear model), keeping the internal score exact so
                    // the next tree's gradients — and hence its linear fit — match C++.
                    let linear_mode = self.linear_tree && self.raw_features.is_some();
                    let full_partition = if linear_mode {
                        let fp = lgbm_treelearner::linear::remap_partition_to_full(
                            &subset_partition,
                            &in_bag,
                            tree.num_leaves,
                        );
                        if !is_first_tree {
                            let raw = self.raw_features.as_ref().unwrap();
                            let nfeat = raw.len() / nd.max(1);
                            lgbm_treelearner::linear::fit_linear_leaves(
                                &mut tree,
                                raw,
                                nfeat,
                                grad,
                                hess,
                                self.linear_lambda,
                                &fp,
                            );
                        }
                        Some(fp)
                    } else {
                        None
                    };
                    tree.shrinkage(shrink_rate);
                    if let Some(fp) = full_partition.as_ref() {
                        // In-bag rows score via the (bin) partition membership + the
                        // leaf's linear model (or constant leaf_value for a non-linear
                        // tree); OOB rows score predict-side over the RAW features
                        // (`Tree::predict` is linear-aware and routes on real thresholds).
                        let raw = self.raw_features.as_ref().unwrap();
                        let nfeat = raw.len() / nd.max(1);
                        self.score_updater
                            .add_linear_tree_train_path(&tree, fp, cur_tree_id, raw, nfeat);
                        let raw_row = |row: i32| -> Vec<f64> {
                            let b = row as usize * nfeat;
                            raw[b..b + nfeat].to_vec()
                        };
                        self.score_updater
                            .add_tree_predict_path(&tree, &oob, cur_tree_id, &raw_row);
                    } else {
                        // NON-LINEAR: score BOTH in-bag and OOB rows predict-side over the
                        // real feature values (bit-exact to the partition scatter on the
                        // identity-binned corpus). OOB rows STILL get scored. The
                        // identity-binned real feature value IS the bin index (raw value
                        // 0..K-1); Tree::predict traverses the real-value thresholds (the
                        // bin upper bounds) so feeding the bin index as the real value
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
                                v[f.real_feature_index as usize] = f.bins.bin(row as usize) as f64;
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
                    }
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
            // Times the per-iter tree-train call (a superset of the growth-loop
            // phases) into the budget — (learner − phases) is the in-learner GPU
            // upload / resident-pool / partition setup the phase guards never saw.
            let (mut tree, partition) = lgbm_treelearner::phase_prof::time(
                &lgbm_treelearner::phase_prof::LEARNER_NS,
                || learner.train_returning_partition(grad, hess, is_first_tree),
            )?;

            if tree.num_leaves > 1 {
                // A real split this class → C++ `should_continue = true`
                // (gbdt.cpp:407): this round emits trees and advances `iter_`.
                should_continue = true;
                // RenewTreeOutput: no-op for L2 (IsRenewTreeOutput()==false), active
                // for regression_l1 — overwrite each leaf's output with the median
                // RESIDUAL (`label[row] - train_score[offset+row]`) of its rows
                // (gbdt.cpp:409-411 + serial_tree_learner.cpp:920-940).
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
                // Quantized leaf renewal (quant_train_renew_leaf): recompute each
                // leaf output from the ORIGINAL gradients (RenewIntGradTreeOutput). Mutually
                // exclusive-in-practice with the objective renew above (that is regression_l1;
                // quantized renew is the gradient-sum formula). Full-corpus path only.
                if self.quant_renew_leaf {
                    if let Some((og, oh)) = orig_grad_hess.as_ref() {
                        let (l1, l2) = (self.quant_l1, self.quant_l2);
                        learner.renew_tree_output(
                            &mut tree,
                            &partition,
                            Some(|_leaf: i32, rows: &[u32]| {
                                let (mut sg, mut sh) = (0.0f64, 0.0f64);
                                for &row in rows {
                                    sg += f64::from(og[offset + row as usize]);
                                    sh += f64::from(oh[offset + row as usize]);
                                }
                                -threshold_l1(sg, l1) / (sh + l2)
                            }),
                        );
                    }
                }
                // ORDER: Shrinkage → UpdateScore →
                // (optional RenewTreeOutput). RenewTreeOutput
                // already ran ABOVE this point (before shrinkage), a no-op for the L2
                // slice (`is_renew_tree_output()==false`); Metric.Eval runs downstream
                // in the facade over `score_updater.scores()` — the resident buffer's
                // host mirror — composed with ConvertOutput. The
                // ORDERING CONTRACT for the L1/quantile follow-up: RenewTreeOutput MUST
                // stay BEFORE shrinkage+UpdateScore (it reads the pre-update score), so
                // any future device RenewTreeOutput refit slots in at the existing
                // renew site above, NOT here — do not reorder.
                //
                // LINEAR TREE (C++ `LinearTreeLearner::CalculateLinear`): fit
                // per-leaf linear models over the RAW features BEFORE shrinkage
                // (which then folds the learning rate into leaf_const/leaf_coeff,
                // exactly as `Tree::shrinkage` also scales leaf_value). Non-first
                // trees only — the ensemble's first tree stays constant.
                let linear_active =
                    self.linear_tree && !is_first_tree && self.raw_features.is_some();
                if linear_active {
                    let raw = self.raw_features.as_ref().unwrap();
                    let nfeat = raw.len() / nd.max(1);
                    lgbm_treelearner::linear::fit_linear_leaves(
                        &mut tree,
                        raw,
                        nfeat,
                        grad,
                        hess,
                        self.linear_lambda,
                        &partition,
                    );
                }
                //
                // Shrinkage BEFORE UpdateScore.
                tree.shrinkage(shrink_rate);
                // UpdateScore. A LINEAR tree's contribution is its per-row linear
                // output (const + Σ coeff·x), NOT the constant leaf_value the
                // train-path scatter would add — so linear trees score through the
                // predict-path over the raw feature rows (`Tree::predict` is
                // linear-aware). The non-linear path keeps the bit-exact
                // training-path per-leaf scatter into score_.
                if linear_active {
                    let raw = self.raw_features.as_ref().unwrap();
                    let nfeat = raw.len() / nd.max(1);
                    self.score_updater.add_linear_tree_train_path(
                        &tree,
                        &partition,
                        cur_tree_id,
                        raw,
                        nfeat,
                    );
                    let init = init_scores[cur_tree_id as usize];
                    if Objective::init_score_is_significant(init) {
                        tree.add_bias(init);
                    }
                    self.trees.push(tree);
                    continue;
                }
                // Times into the whole-train budget.
                //
                // RESIDENT slice: when the
                // `boosting_on_cuda_` toggle is active (`LGBM_CUDA_ON_DEVICE=1`, or the
                // driver seam forced it on) keep `cuda_score_` RESIDENT across the whole
                // train — route the per-leaf `AddScore` through the
                // `add_prediction_to_score_on_device` tree-walk delegate (no new
                // kernel). On the identity-binned L2 continuous corpus the device
                // tree-walk is bit-exact to the host partition scatter (proven in
                // `score_updater_parity`/`resident_score_ab`). With
                // the env unset the toggle is OFF and the host scatter below is
                // byte-unchanged. OUT-OF-SLICE: the DART/RF per-row-predict
                // paths (`add_tree_predict_path`/`add_tree_scaled_all`) stay on the host
                // path.
                if self.score_updater.boosting_on_cuda() {
                    self.update_score_resident::<B>(learner, &tree, &partition, cur_tree_id)?;
                } else {
                    lgbm_treelearner::phase_prof::time(
                        &lgbm_treelearner::phase_prof::SCORE_NS,
                        || {
                            self.score_updater
                                .add_tree_train_path(learner, &tree, &partition, cur_tree_id)
                        },
                    );
                }
                // AddBias AFTER UpdateScore — rewrites STORED tree values only
                // (model text), NEVER score_ (no double-add).
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
                // is the constant. Pushing the constant keeps models_.len() == iter * K.
                // See `no_split_constant_value`.
                let init = init_scores[cur_tree_id as usize];
                let const_val = self.no_split_constant_value(cur_tree_id, labels, init);
                self.trees.push(Tree::as_constant(const_val, self.num_data));
            }
        }

        // ---- (4b) C++ `!should_continue` POP (gbdt.cpp:440-447) ----
        // When NO class split this round (`should_continue == false`, every grown tree
        // was a 1-leaf no-split), C++ `GBDT::TrainOneIter`:
        //   - warns "Stopped training because there are no more leaves",
        //   - if `models_.size() > num_tree_per_iteration_` (i.e. there were ALREADY
        //     trees before this round — a NON-FIRST iteration) POPS this round's K
        //     pushed constants, and
        //   - `return true` WITHOUT `++iter_`.
        // The Python-wheel / `lgb.train` driver then re-bags (fresh RNG draw) on the
        // next boost round (`bagging_freq=1` re-bags every call regardless of `iter_`;
        // the discarded round's bag draw already advanced the RNG so the next round
        // draws the NEXT bag — the C++-faithful cadence) and grows a real tree. This
        // pop is required to match C++'s tree count WITHOUT touching the bagging draw,
        // the learner, the histogram, or RenewTreeOutput.
        //
        // The FIRST iteration (`tree_count_before == 0`, mirroring C++
        // `models_.size() <= num_tree_per_iteration_`) is NOT popped: its no-split
        // constant baseline is KEPT and the counter ADVANCES, leaving
        // `regression_l1_bag1` Tree=0 and the first-iter-constant unit tests
        // byte-unchanged. The pop is isolated to LATER no-split rounds.
        if !should_continue && tree_count_before > 0 {
            // Discard this round's pushed constant(s); do NOT advance `self.iter`.
            self.trees.truncate(tree_count_before);
            return Ok(IterSnapshot {
                gradients,
                hessians,
                // No tree was added → score_ is unchanged from before this round; the
                // driver must NOT count this as an emitted iteration (`emitted=false`):
                // no duplicate `iter_scores` / `iter_grad_hess` push, no advance.
                score: lgbm_treelearner::phase_prof::time(
                &lgbm_treelearner::phase_prof::SNAPSHOT_NS,
                || self.score_updater.scores().to_vec(),
            ),
                emitted: false,
            });
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
            score: lgbm_treelearner::phase_prof::time(
                &lgbm_treelearner::phase_prof::SNAPSHOT_NS,
                || self.score_updater.scores().to_vec(),
            ),
            emitted: true,
        })
    }

    /// C++ `RF::Boosting` (`rf.hpp:90-109`) — compute the per-class init score
    /// (`BoostFromAverage(class, update_scorer=false)`) and derive grad/hess ONCE
    /// from a CONSTANT init-score buffer (every row of class `k` set to
    /// `init_scores_[k]`), NOT from the accumulated `score_`. RF reuses these
    /// gradients every iteration — the trees differ only through the per-iteration
    /// bagged subset. Also enforces the two RF CHECKs (`rf.hpp:35-40,91-93`): the
    /// objective must be non-custom and either bagging OR feature sub-sampling must
    /// be active.
    ///
    /// Returns the resolved `(init_scores, gradients, hessians)`.
    ///
    /// # Errors
    /// [`BoostingError::RfConfig`] when the objective is custom (`obj == nullptr`)
    /// or neither randomization source is active; objective errors propagate.
    fn rf_boosting(
        &mut self,
        labels: &[f32],
        k: i32,
        nd: usize,
        total: usize,
    ) -> Result<RfBoosting, BoostingError> {
        let cfg = self.rf.as_ref().expect("RF state present").config;

        // CHECK 1 (rf.hpp:91-93): RF requires a built-in objective (objective_function_
        // != nullptr). The custom objective maps to the C++ nullptr path.
        if !self.objective.boost_from_average_enabled() {
            return Err(BoostingError::RfConfig {
                what: "RF (boosting=rf) requires a built-in objective; the custom \
                       objective is not supported (rf.hpp:91-93)"
                    .to_string(),
            });
        }
        // CHECK 2 (rf.hpp:35-40): RF requires mandatory randomization — either active
        // row bagging OR active feature sub-sampling. Without either, every tree is
        // identical and the forest collapses.
        if !cfg.bagging_active && !cfg.feature_subsampling_active {
            return Err(BoostingError::RfConfig {
                what: "RF (boosting=rf) requires either active bagging \
                       (bagging_freq > 0 && 0 < bagging_fraction < 1) or active \
                       feature sub-sampling (0 < feature_fraction < 1) (rf.hpp:35-40)"
                    .to_string(),
            });
        }

        // Per-class init score. C++ `RF::Boosting` calls `BoostFromAverage(class,
        // false)` — update_scorer=false, so it does NOT touch score_ (RF maintains
        // score_ as a running average, seeded by the first tree, not by an additive
        // init). We therefore compute the init via the objective directly here rather
        // than through `boost_from_average` (which adds to score_).
        let mut init_scores = vec![0.0f64; k as usize];
        for cur_tree_id in 0..k {
            init_scores[cur_tree_id as usize] = self.objective.boost_from_score(cur_tree_id, labels);
        }

        // Derive grad/hess ONCE from the constant init-score buffer (rf.hpp:98-108).
        let mut const_scores = vec![0.0f64; total];
        for (j, &init) in init_scores.iter().enumerate() {
            let off = j * nd;
            for s in &mut const_scores[off..off + nd] {
                *s = init;
            }
        }
        let mut gradients = vec![0.0f32; total];
        let mut hessians = vec![0.0f32; total];
        self.objective
            .get_gradients(&const_scores, labels, &mut gradients, &mut hessians)?;

        Ok((init_scores, gradients, hessians))
    }

    /// C++ `RF::TrainOneIter` (`rf.hpp:111-182`). One Random-Forest iteration:
    /// draw the mandatory bag, grow one tree per class on the in-bag subset against
    /// the ONCE-derived constant grad/hess, renew each leaf to the mean residual
    /// `label - init_score`, `AddBias(init_score)`, and fold the tree into `score_`
    /// as a RUNNING AVERAGE (`MultiplyScore(iter); AddScore; MultiplyScore(1/(iter+1))`,
    /// rf.hpp:157-159) with NO learning-rate shrinkage. Returns the per-iter snapshot.
    ///
    /// # Errors
    /// [`BoostingError::RfConfig`] from the iter-0 CHECKs; learner/objective errors
    /// propagate.
    fn train_one_iter_rf<B: Backend>(
        &mut self,
        learner: &mut SerialTreeLearner<'_, B>,
        labels: &[f32],
        _num_features: usize,
        k: i32,
        nd: usize,
        total: usize,
    ) -> Result<IterSnapshot, BoostingError> {
        // On the first iteration, run RF::Boosting once: enforce the CHECKs + derive
        // the constant grad/hess + init scores. Subsequent iterations reuse them.
        if self.iter == 0 {
            let (init_scores, gradients, hessians) = self.rf_boosting(labels, k, nd, total)?;
            let rf = self.rf.as_mut().expect("RF state present");
            rf.init_scores = init_scores;
            rf.gradients = gradients;
            rf.hessians = hessians;
        }

        // Snapshot the once-derived grad/hess + init scores (cloned out so the
        // borrow checker is happy across the learner + score_updater mut-borrows).
        let (init_scores, gradients, hessians) = {
            let rf = self.rf.as_ref().expect("RF state present");
            (
                rf.init_scores.clone(),
                rf.gradients.clone(),
                rf.hessians.clone(),
            )
        };

        // ---- mandatory bagging (rf.hpp:113-117) ----
        // RF MUST bag (or feature-subsample). The bag is drawn over the
        // bit-exact BaggingSampleStrategy RNG. When feature
        // sub-sampling is the randomization source (no row bagging), there is no
        // subset and trees train on the full corpus (the col-sampler differs per
        // tree — that wiring lives in the learner; here use_subset is false).
        // RF folds EVERY row into the running-average score (rf_update_score scores
        // the full corpus predict-side), so the OOB index set is not needed
        // separately — only the in-bag subset for TRAINING the tree.
        let (use_subset, sample_in_bag) = if let Some(bag) = self.bagging.as_mut() {
            bag.bagging(self.iter, labels);
            if bag.is_use_subset() {
                (true, bag.in_bag().to_vec())
            } else {
                (false, Vec::new())
            }
        } else {
            (false, Vec::new())
        };

        // The running-average rescale factor base (rf.hpp:157,159): num_init_iteration_
        // is 0 for a fresh train, so MultiplyScore(iter); ...; MultiplyScore(1/(iter+1)).
        let iter = self.iter as f64;

        // ---- per-class tree loop (rf.hpp:129-179) ----
        for cur_tree_id in 0..k {
            let offset = (cur_tree_id as usize) * nd;
            let init = init_scores[cur_tree_id as usize];

            // class_need_train gate (rf.hpp:132): a class that is not trained pushes a
            // constant tree (BoostFromScore / init) once, still folded into the
            // running average so models_.len() stays iter*K.
            if !self.objective.class_need_train(cur_tree_id, labels) {
                let is_first_iter = self.trees.len() < k as usize;
                if is_first_iter {
                    let const_val = init; // rf.hpp:163-170 (objective non-null branch)
                    let tree = Tree::as_constant(const_val, self.num_data);
                    self.rf_update_score(&tree, cur_tree_id, iter);
                    self.trees.push(tree);
                } else {
                    // Extend with a 1-leaf zero tree (no additional score move).
                    self.trees.push(Tree::as_constant(0.0, self.num_data));
                }
                continue;
            }

            let grad = &gradients[offset..offset + nd];
            let hess = &hessians[offset..offset + nd];
            let is_first_tree = self.trees.len() < k as usize;

            // Grow the tree on the bagged subset (or full corpus when feature-only
            // randomization). RF residual_getter = `label - init_scores_[k]` (a
            // CONSTANT pred, rf.hpp:149-150) — NOT the accumulated score (the GBDT
            // spine uses `label - score`). RF ALWAYS renews (rf.hpp:150-152 calls
            // RenewTreeOutput unconditionally for a grown tree).
            let mut tree;
            if use_subset {
                let in_bag = sample_in_bag.clone();
                let (grown, subset_partition) = learner
                    .train_on_subset_returning_partition(&in_bag, grad, hess, is_first_tree)?;
                tree = grown;
                if tree.num_leaves > 1 {
                    // RenewTreeOutput is GATED on obj->IsRenewTreeOutput()
                    // (serial_tree_learner.cpp:922) — a NO-OP for L2 (the leaf value is
                    // the learner's gradient-fit Newton output -sum_grad/sum_hess over
                    // the bagged subset, derived from the constant init buffer), ACTIVE
                    // for regression_l1 etc. (mean RESIDUAL `label - init`). RF renews
                    // with `pred = init_scores_[k]` (a CONSTANT, rf.hpp:149-150).
                    if self.objective.is_renew_tree_output() {
                        self.rf_renew_subset(
                            learner,
                            &mut tree,
                            &subset_partition,
                            &in_bag,
                            labels,
                            init,
                        );
                    }
                    // NO shrinkage (rf.hpp shrinkage_rate_ = 1.0).
                    if Objective::init_score_is_significant(init) {
                        tree.add_bias(init);
                    }
                    // Fold into the running-average score over BOTH in-bag and OOB
                    // rows (rf.hpp UpdateScore via MultiplyScore sandwich). RF scores
                    // EVERY row (the predict-side full-corpus add), so do it as the
                    // running-average rescale below.
                    self.rf_update_score(&tree, cur_tree_id, iter);
                    self.trees.push(tree);
                } else {
                    // No-split bagged tree: constant = init (rf.hpp falls back to the
                    // init score for the constant), folded into the running average.
                    let const_tree = Tree::as_constant(init, self.num_data);
                    self.rf_update_score(&const_tree, cur_tree_id, iter);
                    self.trees.push(const_tree);
                }
                continue;
            }

            // Feature-only randomization (no row bagging): grow on the full corpus.
            let (grown, partition) = learner.train_returning_partition(grad, hess, is_first_tree)?;
            tree = grown;
            if tree.num_leaves > 1 {
                // RenewTreeOutput gated on obj->IsRenewTreeOutput() (no-op for L2).
                if self.objective.is_renew_tree_output() {
                    self.rf_renew_full(&mut tree, &partition, learner, labels, init);
                }
                if Objective::init_score_is_significant(init) {
                    tree.add_bias(init);
                }
                self.rf_update_score(&tree, cur_tree_id, iter);
                self.trees.push(tree);
            } else {
                let const_tree = Tree::as_constant(init, self.num_data);
                self.rf_update_score(&const_tree, cur_tree_id, iter);
                self.trees.push(const_tree);
            }
        }

        self.iter += 1;
        Ok(IterSnapshot {
            gradients,
            hessians,
            score: lgbm_treelearner::phase_prof::time(
                &lgbm_treelearner::phase_prof::SNAPSHOT_NS,
                || self.score_updater.scores().to_vec(),
            ),
            // RF averages a tree every round (no no-split pop) → always emitted.
            emitted: true,
        })
    }

    /// RF running-average score fold (`rf.hpp:157-159`): `MultiplyScore(iter);
    /// AddScore(tree); MultiplyScore(1/(iter+1))` — keep `score_` as the mean of the
    /// `iter+1` per-tree raw outputs. The tree is added over the FULL corpus
    /// predict-side (every row scored, like the bagging OOB path) — bit-exact to the
    /// partition scatter on the identity-binned corpus (the L2 contract). `num_init_
    /// iteration_` is 0 for a fresh train, so the factors are exactly `iter` and
    /// `1/(iter+1)`.
    fn rf_update_score(&mut self, tree: &Tree, cur_tree_id: i32, iter: f64) {
        // MultiplyScore(cur_tree_id, iter) — un-average back to a sum of `iter` trees.
        self.score_updater.multiply_score(iter, cur_tree_id);
        // AddScore(tree) over the full corpus (predict-side, every row). Use the
        // running-average helper with multiply=1.0 (raw add). The identity-binned bin
        // index IS the real feature value (the L2 predict-vs-scatter contract).
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
                v[f.real_feature_index as usize] = f.bins.bin(row as usize) as f64;
            }
            v
        };
        // A constant (num_leaves<=1) tree still contributes its leaf value to every
        // row — add_tree_scaled_all handles both grown and constant trees.
        self.score_updater
            .add_tree_scaled_all(tree, cur_tree_id, 1.0, feature_row);
        // MultiplyScore(cur_tree_id, 1/(iter+1)) — re-average over `iter+1` trees.
        self.score_updater.multiply_score(1.0 / (iter + 1.0), cur_tree_id);
    }

    /// RF leaf renewal on the FULL-corpus path (`rf.hpp:150-152`,
    /// serial_tree_learner.cpp:920-958 RenewTreeOutput, GATED on
    /// `obj->IsRenewTreeOutput()`): overwrite each leaf's output with the objective's
    /// renewal of the residuals `label - init_score` (the C++ residual_getter
    /// `label[i] - pred` with `pred = init_scores_[cur_tree_id]`, a CONSTANT). The
    /// renewal is the objective's percentile (median for regression_l1, weighted for
    /// mape) — NOT a plain mean — dispatched via `renew_leaf_output`. A NO-OP for L2
    /// (the caller only invokes this when `is_renew_tree_output()` is true).
    fn rf_renew_full<B: Backend>(
        &self,
        tree: &mut Tree,
        partition: &lgbm_treelearner::DataPartition,
        learner: &SerialTreeLearner<'_, B>,
        labels: &[f32],
        init: f64,
    ) {
        let obj = &self.objective;
        learner.renew_tree_output(
            tree,
            partition,
            Some(|_leaf: i32, rows: &[u32]| {
                if rows.is_empty() {
                    return 0.0;
                }
                let residuals: Vec<f64> = rows
                    .iter()
                    .map(|&r| labels[r as usize] as f64 - init)
                    .collect();
                let leaf_labels: Vec<f32> =
                    rows.iter().map(|&r| labels[r as usize]).collect();
                obj.renew_leaf_output(&residuals, &leaf_labels)
            }),
        );
    }

    /// RF leaf renewal on the BAGGED-SUBSET path (`rf.hpp:150-152` with the subset
    /// index mapping, GATED on `obj->IsRenewTreeOutput()`). Mirrors
    /// [`Self::rf_renew_full`] but maps each leaf's subset-row index through
    /// `in_bag[sr]` to the full-corpus row before forming the residual `label[fr] -
    /// init`. The renewal is the objective's percentile (`renew_leaf_output`).
    fn rf_renew_subset<B: Backend>(
        &self,
        learner: &SerialTreeLearner<'_, B>,
        tree: &mut Tree,
        subset_partition: &lgbm_treelearner::DataPartition,
        in_bag: &[i32],
        labels: &[f32],
        init: f64,
    ) {
        let obj = &self.objective;
        learner.renew_tree_output(
            tree,
            subset_partition,
            Some(|_leaf: i32, subset_rows: &[u32]| {
                if subset_rows.is_empty() {
                    return 0.0;
                }
                let full_rows: Vec<usize> = subset_rows
                    .iter()
                    .map(|&sr| in_bag[sr as usize] as usize)
                    .collect();
                let residuals: Vec<f64> =
                    full_rows.iter().map(|&fr| labels[fr] as f64 - init).collect();
                let leaf_labels: Vec<f32> =
                    full_rows.iter().map(|&fr| labels[fr]).collect();
                obj.renew_leaf_output(&residuals, &leaf_labels)
            }),
        );
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

    /// Driver/test seam: force the resident
    /// `boosting_on_cuda_` toggle on the internal score updater. The env-gated default
    /// from [`ScoreUpdater::new`] already reflects `LGBM_CUDA_ON_DEVICE` (OFF unless
    /// `=1`); this override lets the resident-score A/B run BOTH arms — the
    /// resident (`true`) and the host reference (`false`) — inside one process (the
    /// `cuda_on_device_enabled()` env is a process-global `OnceLock`, so per-arm env
    /// toggling is not possible). Additive; unset-env default stays host.
    pub fn set_boosting_on_cuda(&mut self, on: bool) {
        self.score_updater.set_boosting_on_cuda(on);
    }

    /// Whether the internal score updater's resident `boosting_on_cuda_` path is active.
    #[must_use]
    pub fn boosting_on_cuda(&self) -> bool {
        self.score_updater.boosting_on_cuda()
    }

    /// Test/inspection seam: whether this train activated the GBDT-owned
    /// resident score + on-device grad/hess path (the num_class==1 / no-bagging / no-GOSS /
    /// GBDT-spine / on-device-supported-objective / `boosting_on_cuda()` envelope). `None`
    /// until the first [`Self::train_one_iter`] freezes the decision; `Some(true)` only
    /// inside the envelope, `Some(false)` for every out-of-envelope train.
    #[must_use]
    pub fn resident_score_active(&self) -> Option<bool> {
        self.resident_active
    }

    /// Resident UpdateScore: keep
    /// `cuda_score_` RESIDENT by routing the grown tree's per-leaf `AddScore` through
    /// the `add_prediction_to_score_on_device` tree-walk delegate (no
    /// new tree-walk kernel), mirroring the resident buffer back into `score_`.
    ///
    /// The `PredictTree` is reconstructed from the grown [`Tree`]'s INNER-bin arrays
    /// (`split_feature_inner`/`threshold_in_bin`/`decision_type`/children/`leaf_value`)
    /// plus the spine's [`FeatureColumn`] per-feature bin metadata; `rows` is the
    /// row-major `[num_data × num_features]` bin matrix. On the identity-binned L2
    /// continuous corpus this device tree-walk is bit-exact to the host partition
    /// scatter (the L2 contract). SCOPE: the single-feature-per-group L2 spine, where
    /// each split's inner feature index equals the feature's position in `self.features`
    /// (so `feat_column[f] = f`, identity). Categorical bitsets are empty this slice.
    ///
    /// # Errors
    /// Propagates the device delegate's [`lgbm_compute::ComputeError`] (length /
    /// bit-type / index validation) wrapped through the tree-learner error boundary.
    fn update_score_resident<B: Backend>(
        &mut self,
        learner: &SerialTreeLearner<'_, B>,
        tree: &Tree,
        partition: &lgbm_treelearner::DataPartition,
        cur_tree_id: i32,
    ) -> Result<(), BoostingError> {
        // If the tree was grown ON-DEVICE, the grow
        // already produced the resident row→leaf `partition`, returned as a HOST
        // `DataPartition`. Apply the per-leaf `AddScore` via the HOST partition scatter
        // (`add_tree_train_path`) — the SAME call the non-on-device arm uses.
        //
        // WHY: the score buffer is NOT actually device-resident —
        // `self.score` IS the buffer (`score_updater.rs:270-272`), and every score op
        // already mirrors to it on the host. So an eager per-tree device round-trip
        // (fresh `ResidentScore` alloc + `num_data` zero-upload + on-device scatter of the
        // leaf values + a BLOCKING full-buffer f64 readback via
        // `add_prediction_to_score_on_device_resident`) preserves NO residency benefit — it
        // is pure overhead relative to the host scatter, which is cheap because the host
        // already holds the tree's post-shrinkage `leaf_value` and the resident partition.
        // Bit-exact by construction: an integer-indexed
        // f64 `+=` of the same leaf values, each row written exactly once (a row is in
        // exactly one leaf) — no reduction, no accumulation-order dependence.
        //
        // `add_prediction_to_score_on_device_resident` / `ResidentScore` /
        // `add_tree_train_path_delta_on_device` remain defined and tested (resident_score_ab.rs)
        // but are no longer called from this seam. Reached only under `boosting_on_cuda_`
        // (env-gated), so with `LGBM_CUDA_ON_DEVICE` unset this is dead and the host score
        // path is unchanged. The fallback device-tree-walk arm below is untouched.
        // Mirror this tree's per-leaf AddScore onto the
        // GBDT-owned resident score buffer (ON DEVICE, no readback), keeping it in lockstep
        // with the host `score_updater.scores()` so the NEXT iteration's on-device grad/hess
        // branch reads a resident score bit-identical to the host accumulation. Only the
        // num_class==1 GBDT-spine envelope ever constructs `resident_score`, so this is
        // inert everywhere else. The host scatter below is UNCHANGED (it keeps
        // `score_updater.scores()` authoritative for Metric::Eval / the model text); this
        // ADDS the resident mirror ALONGSIDE it — never a silent partial-resident state
        // (residency only pays off WITH the matching resident grad/hess path, which is
        // exactly why they are wired together here). Bit-exact by construction: an
        // integer-indexed f64 scatter of the SAME post-shrinkage `tree.leaf_value` over the
        // SAME partition ranges the host scatter uses (`add_tree_on_device` ↔
        // `add_prediction_to_score`), each row written exactly once.
        let num_data_i32 = self.num_data;
        if let Some(boxed) = self.resident_score.as_mut() {
            let num_leaves = tree.num_leaves.max(1) as usize;
            let leaf_begin: Vec<i32> = (0..num_leaves)
                .map(|l| partition.leaf_begin(l as i32))
                .collect();
            let leaf_count: Vec<i32> = (0..num_leaves)
                .map(|l| partition.leaf_count(l as i32))
                .collect();
            let layout = LeafPartitionLayout {
                num_data: num_data_i32,
                indices: partition.indices().to_vec(),
                leaf_begin,
                leaf_count,
            };
            let client = learner.client();
            let rs = boxed
                .downcast_mut::<ResidentScore<B::Runtime>>()
                .expect("resident_score is ResidentScore<B::Runtime> for this train's backend");
            rs.add_tree_on_device(client, &layout, &tree.leaf_value)
                .map_err(|e| BoostingError::TreeLearner(e.into()))?;
        }

        if learner.take_resident_score_pending() {
            lgbm_treelearner::phase_prof::time(
                &lgbm_treelearner::phase_prof::SCORE_NS,
                || {
                    self.score_updater
                        .add_tree_train_path(learner, tree, partition, cur_tree_id)
                },
            );
            return Ok(());
        }

        // The learner is the authoritative holder of the columns the tree was grown
        // over (so each split's inner feature index equals the column position). The
        // GBDT-spine `self.features` is only populated on the bagging/DART/RF builders
        // (out-of-slice), so source the columns from the learner here.
        let features = learner.features();
        let nf = features.len();
        let nd = self.num_data as usize;

        // Per-(inner)feature bin metadata, indexed by inner feature index = the
        // feature's position in the learner's columns (single-feature-per-group spine).
        let feat_min: Vec<i32> = features.iter().map(|f| f.min_bin as i32).collect();
        let feat_max: Vec<i32> = features.iter().map(|f| f.max_bin as i32).collect();
        let feat_default: Vec<i32> = features.iter().map(|f| f.default_bin as i32).collect();
        let feat_mfb: Vec<i32> = features.iter().map(|f| f.most_freq_bin as i32).collect();
        // Predict-remap offset — the CUDA-tree walk remaps `bin = raw − min_bin + offset`
        // and compares against the tree's `threshold_in_bin`. A grown tree's
        // `threshold_in_bin` is recorded in the `min_bin`-relative PREDICT space (the
        // compaction/most-freq-bin shift is already baked in), so this predict offset is
        // 0 — matching the canonical `lib_lightgbm` predict golden (`predict.rs`
        // `numeric_tree` uses `feat_offset = [0, 0]` even for non-zero `most_freq_bin`).
        // This is DISTINCT from `FeatureColumn::offset` (the histogram-scan compaction
        // offset); reusing that here double-counts the shift and mis-routes every split.
        let feat_offset: Vec<i32> = vec![0i32; nf];
        // Identity inner→row-matrix-column mapping (column `f` of `rows` holds
        // feature `f`'s bins).
        let feat_column: Vec<i32> = (0..nf as i32).collect();

        // Row-major bin matrix [num_data × num_features]; the walk's `data_index`
        // indexes `rows[i*nf + column]`.
        let mut rows: Vec<u32> = Vec::with_capacity(nd * nf);
        for i in 0..nd {
            for f in features {
                rows.push(f.bins.bin(i));
            }
        }
        // Native column width for the walk: 8/16/32 by the widest feature.
        let bit_type: u32 = features
            .iter()
            .map(|f| match &f.bins {
                lgbm_compute::BinColumn::U8(_) => 8u32,
                lgbm_compute::BinColumn::U16(_) => 16u32,
                lgbm_compute::BinColumn::U32(_) => 32u32,
            })
            .max()
            .unwrap_or(8);

        // Widen the grown tree's `decision_type` (i8) to the walk's i32 array.
        let decision_type: Vec<i32> = tree.decision_type.iter().map(|&d| i32::from(d)).collect();

        let predict_tree = PredictTree {
            feat_min: &feat_min,
            feat_max: &feat_max,
            feat_default: &feat_default,
            feat_most_freq_bin: &feat_mfb,
            feat_offset: &feat_offset,
            feat_column: &feat_column,
            split_feature_inner: &tree.split_feature_inner,
            threshold_in_bin: &tree.threshold_in_bin,
            decision_type: &decision_type,
            left_child: &tree.left_child,
            right_child: &tree.right_child,
            leaf_value: &tree.leaf_value,
            bitset_inner: &[],
            cat_boundaries_inner: &[],
        };

        // Time into the whole-train SCORE budget, same as the host scatter branch.
        let res = lgbm_treelearner::phase_prof::time(
            &lgbm_treelearner::phase_prof::SCORE_NS,
            || {
                self.score_updater.add_tree_train_path_on_device::<B>(
                    learner.client(),
                    &predict_tree,
                    &rows,
                    nf,
                    bit_type,
                    cur_tree_id,
                )
            },
        );
        res.map_err(|e| BoostingError::TreeLearner(e.into()))?;
        Ok(())
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
            v[f.real_feature_index as usize] = f.bins.bin(row as usize) as f64;
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
                // DELIBERATE 0/0 = NaN, byte-for-byte with C++ dart.hpp:104
                // (`static_cast<double>(tree_weight_.size()) / sum_weight_`). On the
                // first drop-eligible iteration `sum_weight == 0`, so both this and
                // the `max_drop` cap term below are NaN. This is SAFE only because on
                // iter 0 (the sole iteration with `sum_weight == 0` under fresh
                // training, `num_init_iteration_ == 0`) the `for i in 0..iter` drop
                // loop is EMPTY, so the NaN is never reached by a `draw < ...`
                // comparison. WARNING: if DART continue-training is ever wired
                // (`with_loaded_model` + DART, `num_init_iteration_ > 0`), a non-empty
                // drop loop could reach this NaN, where `NaN < x` is always false
                // (no trees dropped) — guard `sum_weight == 0` there before relying on
                // it. Do NOT "fix" the 0/0 here: it must stay bit-exact to C++.
                let inv_average_weight = dart.tree_weight.len() as f64 / dart.sum_weight;
                if cfg.max_drop > 0 {
                    drop_rate =
                        drop_rate.min(cfg.max_drop as f64 * inv_average_weight / dart.sum_weight);
                }
                for i in 0..iter {
                    let draw = dart.random_for_drop.next_float() as f64;
                    if draw < drop_rate * dart.tree_weight[i as usize] * inv_average_weight {
                        dart.drop_index.push(i); // num_init_iteration_ = 0
                        // C++ `static_cast<size_t>(config_->max_drop)` (dart.hpp:111):
                        // a negative `max_drop` casts to a huge size_t so the cap
                        // never fires (unbounded drops); `max_drop == 0` casts to 0
                        // so it breaks after the first push. `i32 as usize` sign-
                        // extends to 64-bit identically — do NOT clamp with `.max(0)`,
                        // which would cap a negative `max_drop` at 1 drop.
                        if dart.drop_index.len() >= cfg.max_drop as usize {
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
                        // See the non-uniform branch above: `i32 as usize`
                        // reproduces C++ `static_cast<size_t>(config_->max_drop)`
                        // exactly — negative => unbounded, 0 => break after first.
                        if dart.drop_index.len() >= cfg.max_drop as usize {
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

    /// Leaf-refit (`GBDT::RefitTree`, gbdt.cpp:258-294 + the Python
    /// `Booster.refit`). Re-fits the LEAF OUTPUTS of the already-loaded ensemble on
    /// new `(features, labels)` WITHOUT changing tree STRUCTURE, blending each leaf by
    /// `refit_decay_rate` (`new = decay*old + (1-decay)*shrunk_newton`).
    ///
    /// Mirrors the C++ loop exactly: for each stored iteration, `Boosting()` recomputes
    /// per-row grad/hess on the CURRENT refit score (which starts at the constructor's
    /// init — 0 with `boost_from_average=false`), then each tree in that iteration is
    /// leaf-refit (rows routed through the tree's existing structure, per-leaf grad/hess
    /// accumulated, decay-blended) and its new predictions are added to the refit score
    /// (the C++ `train_score_updater_->AddScore(new_tree)`). The refit is in-place:
    /// `self.trees` ends with the re-fit leaf values.
    ///
    /// `use_l1` / `l1` / `l2` mirror the leaf-output config (`lambda_l1`/`lambda_l2`).
    /// The caller pre-loads the trees + the loaded score via [`Self::with_loaded_model`]
    /// — but refit RESETS the score to a clean accumulation (the C++ `RefitTree`
    /// re-scores from the refit trees themselves), so this method zeroes the score
    /// buffer first.
    ///
    /// # Errors
    /// Objective errors propagate via `#[from]`.
    #[allow(clippy::too_many_arguments)]
    pub fn refit(
        &mut self,
        labels: &[f32],
        refit_decay_rate: f64,
        use_l1: bool,
        l1: f64,
        l2: f64,
    ) -> Result<(), BoostingError> {
        let nd = self.num_data as usize;
        if labels.len() != nd {
            return Err(BoostingError::LengthMismatch {
                expected: nd,
                actual: labels.len(),
            });
        }
        let k = self.objective.num_model_per_iteration().max(1);
        let total = nd * k as usize;
        let num_iterations = self.trees.len() / k.max(1) as usize;

        // Snapshot the per-row feature vectors once (the C++ data_partition_ row→leaf
        // map, here re-derived by routing each row through each tree's structure).
        let rows: Vec<Vec<f64>> = (0..self.num_data).map(|r| self.feature_row(r)).collect();
        let decay = refit_decay_rate;

        // C++ RefitTree re-scores from the refit trees; reset the accumulation.
        for cur_tree_id in 0..k {
            self.score_updater.multiply_score(0.0, cur_tree_id);
        }

        let mut gradients = vec![0.0f32; total];
        let mut hessians = vec![0.0f32; total];
        for iter in 0..num_iterations {
            // Boosting(): grad/hess on the CURRENT refit score.
            self.objective.get_gradients(
                self.score_updater.scores(),
                labels,
                &mut gradients,
                &mut hessians,
            )?;
            for cur_tree_id in 0..k {
                let model_index = iter * k as usize + cur_tree_id as usize;
                let offset = (cur_tree_id as usize) * nd;
                let grad = &gradients[offset..offset + nd];
                let hess = &hessians[offset..offset + nd];
                // FitByExistingTree on the existing structure (in-place leaf refit).
                // We refit a single tree (`model_index`) — feed only this tree's
                // class-slice grad/hess (the C++ gradients_pointer_ + offset).
                self.refit_one_tree_inplace(model_index, &rows, grad, hess, decay, use_l1, l1, l2);
                // AddScore(new_tree) to the refit score (predict-side, full corpus).
                let tree = self.trees[model_index].clone();
                self.score_updater
                    .add_tree_scaled_all(&tree, cur_tree_id, 1.0, |r| rows[r as usize].clone());
            }
        }
        Ok(())
    }

    /// Leaf-refit one stored tree in place (the [`GbdtModel::refit_one_tree`] mirror
    /// at the boosting layer, operating on `self.trees`). Routes every row to its leaf
    /// in `self.trees[tree_index]`, accumulates per-leaf grad/hess, decay-blends.
    #[allow(clippy::too_many_arguments)]
    fn refit_one_tree_inplace(
        &mut self,
        tree_index: usize,
        rows: &[Vec<f64>],
        gradients: &[f32],
        hessians: &[f32],
        decay: f64,
        use_l1: bool,
        l1: f64,
        l2: f64,
    ) {
        let num_leaves = self.trees[tree_index].num_leaves.max(1) as usize;
        let mut sum_grad = vec![0.0f64; num_leaves];
        let mut sum_hess = vec![0.0f64; num_leaves];
        for (r, row) in rows.iter().enumerate() {
            let leaf = if self.trees[tree_index].num_leaves > 1 {
                self.trees[tree_index].get_leaf(row) as usize
            } else {
                0
            };
            sum_grad[leaf] += gradients[r] as f64;
            sum_hess[leaf] += hessians[r] as f64;
        }
        for leaf in 0..num_leaves {
            self.trees[tree_index].refit_leaf(
                leaf, sum_grad[leaf], sum_hess[leaf], decay, use_l1, l1, l2,
            );
        }
    }

    /// Continue-training (`input_model`): seed this driver with the trees of
    /// a previously-trained ensemble so subsequent [`Self::train_one_iter`] calls
    /// APPEND new trees on top (C++ `GBDT::InitTrain` with `num_init_iteration_ =
    /// models_.size() / num_tree_per_iteration_`, gbdt.cpp:38-40 + the
    /// `(i + num_init_iteration_)` curr-tree indexing at gbdt.cpp:200/773).
    ///
    /// `loaded_trees` are the loaded model's trees (flat, `[i*K + k]`); `features`
    /// are the full-corpus feature columns used to re-derive the loaded ensemble's
    /// per-row contribution to `score_` (the C++ `train_score_updater_` is
    /// re-initialized by predicting the loaded ensemble over the training data on a
    /// `ResetTrainingData`, gbdt.cpp:726-790). After seeding, `iter` equals the
    /// loaded iteration count, so `BoostFromAverage` is skipped (models_ is
    /// non-empty) and the next trained tree is iteration `num_init_iteration`.
    ///
    /// Builder-style; chains after a constructor. The caller MUST pass the same
    /// objective / num_class the loaded model was trained with.
    #[must_use]
    pub fn with_loaded_model(mut self, loaded_trees: Vec<Tree>, features: Vec<FeatureColumn>) -> Self {
        self.features = features;
        let k = self.objective.num_model_per_iteration().max(1);
        // Re-initialize score_ by adding each loaded tree's prediction over the full
        // corpus (the C++ ResetTrainingData re-score). The identity-binned bin index
        // IS the real feature value (the L2 predict-vs-scatter contract). Snapshot the
        // per-row feature vectors once (like the DART/RF re-score paths) so the
        // per-tree closure is a cheap index into the shared buffer.
        let rows: Vec<Vec<f64>> = (0..self.num_data).map(|r| self.feature_row(r)).collect();
        for (i, tree) in loaded_trees.iter().enumerate() {
            let cur_tree_id = (i as i32) % k;
            self.score_updater
                .add_tree_scaled_all(tree, cur_tree_id, 1.0, |r| rows[r as usize].clone());
        }
        let num_loaded = loaded_trees.len();
        self.trees = loaded_trees;
        // num_init_iteration_ = models_.size() / num_tree_per_iteration_.
        self.iter = (num_loaded / k as usize) as i32;
        self
    }

    /// The number of init iterations carried from a loaded `input_model` (C++
    /// `num_init_iteration_`). For a fresh train this is 0; after
    /// [`Self::with_loaded_model`] it is the loaded iteration count. Subsequently it
    /// equals [`Self::num_iteration`] until the first appended tree advances `iter`.
    pub fn num_init_iteration(&self) -> i32 {
        let k = self.objective.num_model_per_iteration().max(1) as usize;
        (self.trees.len() / k) as i32
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
    /// predict. `objective_string` is the verbatim
    /// `objective=` line; `max_feature_idx` is `max real_feature_index` across
    /// all features.
    pub fn into_model(
        self,
        objective_string: String,
        max_feature_idx: i32,
        feature_names: String,
        feature_infos: String,
    ) -> GbdtModel {
        // RF emits `average_output` (rf.hpp ctor sets `average_output_ = true`): the
        // stored tree leaf values are the RAW per-tree outputs, and prediction
        // divides the tree sum by num_iteration (gbdt_prediction.cpp:57-59).
        let average_output = self.variant == BoostingVariant::Rf;
        GbdtModel {
            trees: self.trees,
            num_class: self.num_class,
            num_tree_per_iteration: self.num_class.max(1),
            label_index: 0,
            max_feature_idx,
            average_output,
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
    use lgbm_treelearner::BinColumn;

    /// A tiny 2-feature corpus that produces a real (>1-leaf) split, mirroring the
    /// learner_parity spine shape. 8 rows, 2 binary-ish features.
    fn corpus() -> (Vec<FeatureColumn>, Vec<f32>, GainConfig) {
        // offset MUST come from the authoritative rule (most_freq_bin==0 -> 1),
        // matching the learner_parity spine corpus convention. Hand-setting
        // offset=0 for a most_freq_bin==0 feature mis-routes the partition.
        let off0 = lgbm_treelearner::offset_for_most_freq_bin(0);
        let f0 = FeatureColumn {
            bins: BinColumn::new(vec![0, 0, 0, 0, 1, 1, 1, 1], 2),
            num_bin: 2,
            offset: off0,
            min_bin: 0,
            max_bin: 1,
            default_bin: 2,
            most_freq_bin: 0,
            missing_type: MissingType::None,
            bin_upper_bound: vec![0.5, 1.5],
            real_feature_index: 0,
            ..Default::default()
        };
        let f1 = FeatureColumn {
            bins: BinColumn::new(vec![0, 0, 1, 1, 0, 0, 1, 1], 2),
            num_bin: 2,
            offset: off0,
            min_bin: 0,
            max_bin: 1,
            default_bin: 2,
            most_freq_bin: 0,
            missing_type: MissingType::None,
            bin_upper_bound: vec![0.5, 1.5],
            real_feature_index: 1,
            ..Default::default()
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
            ..Default::default()
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

    // ===================== multiclass loop =====================

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

    // ===================== bagging / OOB score update =====================

    use crate::sample_strategy::{BaggingConfig, BaggingSampleStrategy};

    #[test]
    fn bagging_oob_rows_are_still_scored() {
        // Out-of-bag rows STILL get the tree prediction added each iter.
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

    // ===================== GOSS sample strategy =====================

    use crate::sample_strategy::GossSampleStrategy;

    #[test]
    fn goss_selected_amplifies_and_scores_all_rows() {
        // GOSS runs INSIDE train_one_iter after get_gradients: it amplifies the
        // sampled rows' grad/hess in place, trains on the kept subset, and scores
        // BOTH kept and dropped rows predict-side — dropped rows still
        // get the tree prediction. With a 12-row corpus, top_rate+other_rate=0.3
        // (subset path) and lr small enough that iter 0 is past the skip window, the
        // first iter must subsample and still move every row's score from init.
        let backend = CpuBackend;
        let client = cpu_client();
        let (features, _l, cfg) = corpus();
        // 12 distinct-gradient rows so GOSS has a non-degenerate top-k threshold.
        let f0 = FeatureColumn {
            bins: BinColumn::new(vec![0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1], features[0].num_bin),
            real_feature_index: 0,
            ..features[0].clone()
        };
        let f1 = FeatureColumn {
            bins: BinColumn::new(vec![0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1], features[1].num_bin),
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
        // init = label mean. Every row (kept + dropped) must be scored predict-side —
        // dropped rows STILL get the tree prediction — so every row's
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
        // Still exactly 3 trees; class 2's is a constant 1-leaf tree.
        assert_eq!(gbdt.trees.len(), 3);
        assert_eq!(
            gbdt.trees[2].num_leaves, 1,
            "absent class -> constant 1-leaf tree"
        );
    }

    // ===================== DART boosting variant =====================

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
            bins: BinColumn::new(vec![0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1], features[0].num_bin),
            real_feature_index: 0,
            ..features[0].clone()
        };
        let f1 = FeatureColumn {
            bins: BinColumn::new(vec![0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1], features[1].num_bin),
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
                    // Mirror C++ `static_cast<size_t>(config_->max_drop)` —
                    // `i32 as usize` (no `.max(0)` clamp).
                    if drop.len() >= cfg.max_drop as usize {
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
                    // Mirror C++ `static_cast<size_t>(config_->max_drop)`.
                    if drop.len() >= cfg.max_drop as usize {
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

    // ===================== Random Forest boosting variant =====================

    fn rf_config_bagging() -> RfConfig {
        RfConfig {
            bagging_active: true,
            feature_subsampling_active: false,
        }
    }

    /// A bagging strategy over the test corpus (fraction 0.7, freq 1) — RF's
    /// mandatory randomization, reusing the bit-exact BaggingSampleStrategy.
    fn rf_bagging_strategy(num_data: i32, labels: &[f32]) -> BaggingSampleStrategy {
        let cfg = BaggingConfig::new(0.7, 1.0, 1.0, 1, 3, false).expect("bagging config");
        BaggingSampleStrategy::reset_sample_config(cfg, num_data, labels)
    }

    #[test]
    fn rf_variant_is_selected_via_with_rf() {
        let (features, labels, _cfg) = corpus();
        let gbdt = Gbdt::new(
            Objective::Regression { sqrt: false },
            0.1,
            1,
            labels.len() as i32,
            true,
            None,
        )
        .with_rf(rf_config_bagging(), features.clone());
        assert_eq!(gbdt.variant(), BoostingVariant::Rf);
    }

    #[test]
    fn rf_no_objective_is_typed_error() {
        // CHECK 1 (rf.hpp:91-93): RF requires a built-in objective; custom rejected.
        let backend = CpuBackend;
        let client = cpu_client();
        let (features, labels, cfg) = corpus();
        let num_data = labels.len() as i32;
        let mut learner =
            SerialTreeLearner::new(&backend, &client, cfg, 2, 1).with_features(features.clone());
        // A trivial custom objective (preds -> zero grad/hess).
        let custom = lgbm_objective::CustomObjective::new(|preds: &[f64]| {
            (vec![0.0f32; preds.len()], vec![1.0f32; preds.len()])
        });
        let mut gbdt = Gbdt::with_objective(
            BoostObjective::Custom(custom),
            1.0,
            1,
            num_data,
            true,
            None,
        )
        .with_rf(rf_config_bagging(), features.clone())
        .with_bagging(rf_bagging_strategy(num_data, &labels), features.clone());
        let err = gbdt
            .train_one_iter(&mut learner, &labels, features.len())
            .expect_err("RF with custom objective must be a typed error");
        assert!(
            matches!(err, BoostingError::RfConfig { .. }),
            "expected RfConfig, got {err:?}"
        );
    }

    #[test]
    fn rf_no_randomization_is_typed_error() {
        // CHECK 2 (rf.hpp:35-40): neither bagging nor feature sub-sampling active.
        let backend = CpuBackend;
        let client = cpu_client();
        let (features, labels, cfg) = corpus();
        let num_data = labels.len() as i32;
        let mut learner =
            SerialTreeLearner::new(&backend, &client, cfg, 2, 1).with_features(features.clone());
        let rf_cfg = RfConfig {
            bagging_active: false,
            feature_subsampling_active: false,
        };
        let mut gbdt = Gbdt::new(
            Objective::Regression { sqrt: false },
            1.0,
            1,
            num_data,
            true,
            None,
        )
        .with_rf(rf_cfg, features.clone());
        let err = gbdt
            .train_one_iter(&mut learner, &labels, features.len())
            .expect_err("RF with no randomization must be a typed error");
        assert!(
            matches!(err, BoostingError::RfConfig { .. }),
            "expected RfConfig, got {err:?}"
        );
    }

    #[test]
    fn rf_score_is_running_average_no_shrinkage() {
        // RF folds each tree into score_ as a running AVERAGE (MultiplyScore
        // sandwich), with NO learning-rate shrinkage. After N iters, every row's
        // score must equal the mean of the per-tree (renewed) raw outputs over that
        // row — bounded by the label range [1, 5] regardless of N (a learning-rate
        // accumulation would drift past the labels). We assert the score stays in
        // the label hull and that adding more trees does NOT push it out (the
        // averaging invariant), distinguishing RF from the additive GBDT spine.
        let backend = CpuBackend;
        let client = cpu_client();
        let (features, labels, cfg) = corpus();
        let num_data = labels.len() as i32;
        let mut learner =
            SerialTreeLearner::new(&backend, &client, cfg, 2, 1).with_features(features.clone());
        let mut gbdt = Gbdt::new(
            Objective::Regression { sqrt: false },
            0.1, // learning rate is IGNORED by RF (shrinkage 1.0) — averaging only
            1,
            num_data,
            true,
            None,
        )
        .with_rf(rf_config_bagging(), features.clone())
        .with_bagging(rf_bagging_strategy(num_data, &labels), features.clone());

        let snaps = gbdt
            .train(&mut learner, &labels, features.len(), 8)
            .expect("rf train");
        assert_eq!(snaps.len(), 8);
        let final_score = &snaps[7].score;
        // Averaging invariant: every row's averaged score is within the label hull
        // [1, 5] (an additive lr-shrinkage path would NOT respect this bound after
        // 8 fully-fit trees). Allow a small epsilon for renew/residual rounding.
        for &s in final_score {
            assert!(
                (0.5..=5.5).contains(&s),
                "RF averaged score {s} must stay within the label hull (no lr accumulation)"
            );
        }
        // The model emits average_output.
        let model = gbdt.into_model(
            "regression".to_string(),
            1,
            "Column_0 Column_1".to_string(),
            "[0:1] [0:1]".to_string(),
        );
        assert!(model.average_output, "RF model must set average_output");
    }

    #[test]
    fn rf_predict_averages_tree_outputs() {
        // RF predict = (sum of stored RAW tree outputs) / num_iteration. The stored
        // leaf values are the RAW per-tree outputs; the GbdtModel carries
        // average_output. Verify the running-average score buffer equals the
        // predict-side average for a representative row (the L2 predict-vs-scatter
        // contract on the identity-binned corpus).
        let backend = CpuBackend;
        let client = cpu_client();
        let (features, labels, cfg) = corpus();
        let num_data = labels.len() as i32;
        let mut learner =
            SerialTreeLearner::new(&backend, &client, cfg, 2, 1).with_features(features.clone());
        let mut gbdt = Gbdt::new(
            Objective::Regression { sqrt: false },
            1.0,
            1,
            num_data,
            true,
            None,
        )
        .with_rf(rf_config_bagging(), features.clone())
        .with_bagging(rf_bagging_strategy(num_data, &labels), features.clone());
        let snaps = gbdt
            .train(&mut learner, &labels, features.len(), 6)
            .expect("rf train");
        let n_iter = gbdt.num_iteration();
        let model = gbdt.into_model(
            "regression".to_string(),
            1,
            "Column_0 Column_1".to_string(),
            "[0:1] [0:1]".to_string(),
        );
        // Row 0 has features (0,0). predict_raw is the SUM of the trees; the averaged
        // prediction is that / num_iteration.
        let row0 = vec![0.0f64, 0.0];
        let raw_sum = model.predict_raw(&row0, 0, -1)[0];
        let avg_pred = raw_sum / n_iter as f64;
        // The running-average score buffer for row 0 must match the predict-side
        // average (both are the mean of the per-tree raw outputs over row 0).
        let score_row0 = snaps[5].score[0];
        assert!(
            (avg_pred - score_row0).abs() < 1e-9,
            "RF predict average {avg_pred} must match the running-average score {score_row0}"
        );
    }

    #[test]
    fn refit_decay_one_leaves_model_unchanged() {
        // Refit with decay=1.0 must leave every leaf value unchanged
        // (all-old), regardless of the new data's gradients.
        let backend = CpuBackend;
        let client = cpu_client();
        let (features, labels, cfg) = corpus();
        let num_data = labels.len() as i32;
        let mut learner =
            SerialTreeLearner::new(&backend, &client, cfg, 2, 1).with_features(features.clone());
        let mut base = Gbdt::new(
            Objective::Regression { sqrt: false },
            0.1,
            1,
            num_data,
            true,
            None,
        );
        base.train(&mut learner, &labels, features.len(), 3).expect("base train");
        let base_trees = base.trees().to_vec();
        let base_leaves: Vec<Vec<f64>> = base_trees.iter().map(|t| t.leaf_value.clone()).collect();

        let mut refit = Gbdt::new(
            Objective::Regression { sqrt: false },
            0.1,
            1,
            num_data,
            true,
            None,
        )
        .with_loaded_model(base_trees, features.clone());
        refit.refit(&labels, 1.0, false, 0.0, 0.0).expect("refit decay=1");
        for (t, old) in refit.trees().iter().zip(&base_leaves) {
            assert_eq!(&t.leaf_value, old, "decay=1 must leave leaves unchanged");
        }
    }

    #[test]
    fn refit_decay_zero_changes_leaves() {
        // Refit with decay=0.0 replaces leaves with the fresh shrunk Newton
        // output on the (same) data — the values must change from the base.
        let backend = CpuBackend;
        let client = cpu_client();
        let (features, labels, cfg) = corpus();
        let num_data = labels.len() as i32;
        let mut learner =
            SerialTreeLearner::new(&backend, &client, cfg, 2, 1).with_features(features.clone());
        let mut base = Gbdt::new(
            Objective::Regression { sqrt: false },
            0.1,
            1,
            num_data,
            true,
            None,
        );
        base.train(&mut learner, &labels, features.len(), 2).expect("base train");
        let base_trees = base.trees().to_vec();
        let base_first_leaves = base_trees[0].leaf_value.clone();

        let mut refit = Gbdt::new(
            Objective::Regression { sqrt: false },
            0.1,
            1,
            num_data,
            true,
            None,
        )
        .with_loaded_model(base_trees, features.clone());
        refit.refit(&labels, 0.0, false, 0.0, 0.0).expect("refit decay=0");
        // The structure (num_leaves) is preserved; the values differ.
        assert_eq!(refit.trees()[0].num_leaves, 2);
        assert_ne!(
            refit.trees()[0].leaf_value, base_first_leaves,
            "decay=0 must change the leaf outputs"
        );
        // Tree count unchanged (refit never adds/removes trees).
        assert_eq!(refit.num_trees(), 2);
    }

    #[test]
    fn continue_training_seeds_num_init_iteration_and_appends() {
        // Continue-training: a baseline model of N trees, loaded via
        // with_loaded_model, must set num_init_iteration to N and APPEND on train.
        let backend = CpuBackend;
        let client = cpu_client();
        let (features, labels, cfg) = corpus();
        let num_data = labels.len() as i32;

        // --- baseline: grow 3 trees and capture them ---
        let mut base_learner =
            SerialTreeLearner::new(&backend, &client, cfg.clone(), 2, 1).with_features(features.clone());
        let mut base = Gbdt::new(
            Objective::Regression { sqrt: false },
            0.1,
            1,
            num_data,
            true,
            None,
        );
        base.train(&mut base_learner, &labels, features.len(), 3)
            .expect("base train");
        let base_trees = base.trees().to_vec();
        let base_count = base_trees.len();
        assert_eq!(base_count, 3, "K=1, 3 iters => 3 trees");

        // --- continue: load the baseline and append 2 more iterations ---
        let mut cont_learner =
            SerialTreeLearner::new(&backend, &client, cfg, 2, 1).with_features(features.clone());
        let mut cont = Gbdt::new(
            Objective::Regression { sqrt: false },
            0.1,
            1,
            num_data,
            true,
            None,
        )
        .with_loaded_model(base_trees, features.clone());

        // num_init_iteration == the loaded count; iter starts there.
        assert_eq!(cont.num_init_iteration(), 3);
        assert_eq!(cont.num_iteration(), 3);

        cont.train(&mut cont_learner, &labels, features.len(), 2)
            .expect("continue train");

        // The tree count grows from the loaded base (3 -> 5), not from zero.
        assert_eq!(cont.num_trees(), base_count + 2, "must append, not restart");
        assert_eq!(cont.num_iteration(), 5);
    }

    // ========= resident score + on-device grad/hess =========

    use lgbm_objective::Binary;

    /// The spine 2-feature corpus with BINARY labels ({0,1}) separable by feature 0 —
    /// every tree splits, so the per-leaf AddScore (and its resident mirror) is exercised
    /// each iteration.
    fn binary_corpus() -> (Vec<FeatureColumn>, Vec<f32>, GainConfig) {
        let (features, _labels, cfg) = corpus();
        let labels = vec![0.0f32, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0];
        (features, labels, cfg)
    }

    /// Train `iters` binary trees on a FRESH learner/GBDT and return `(trees, scores,
    /// resident_active)`. `on` selects the arm via the `set_boosting_on_cuda` seam: `true`
    /// is the resident-score + on-device grad/hess path, `false` the host path.
    fn train_binary_arm(
        features: &[FeatureColumn],
        labels: &[f32],
        cfg: GainConfig,
        on: bool,
        iters: i32,
    ) -> (Vec<Tree>, Vec<f64>, Option<bool>) {
        let backend = CpuBackend;
        let client = cpu_client();
        let num_data = labels.len() as i32;
        let mut learner =
            SerialTreeLearner::new(&backend, &client, cfg, 4, -1).with_features(features.to_vec());
        let mut gbdt = Gbdt::with_objective(
            BoostObjective::Binary(Binary::new(1.0).unwrap()),
            0.3,
            1,
            num_data,
            true,
            None,
        );
        gbdt.set_boosting_on_cuda(on);
        gbdt.train(&mut learner, labels, features.len(), iters)
            .expect("binary train");
        let active = gbdt.resident_score_active();
        (gbdt.trees().to_vec(), gbdt.scores().to_vec(), active)
    }

    /// The num_class==1 / no-bagging / no-GOSS / binary / `boosting_on_cuda`
    /// envelope ACTIVATES the GBDT-owned resident score + on-device grad/hess branch and
    /// produces a model + score BIT-IDENTICAL to the host path over a multi-iteration train.
    #[test]
    fn resident_score_activates_and_is_bit_exact_binary_envelope() {
        let (features, labels, cfg) = binary_corpus();
        let (host_trees, host_scores, host_active) =
            train_binary_arm(&features, &labels, cfg.clone(), false, 4);
        let (dev_trees, dev_scores, dev_active) =
            train_binary_arm(&features, &labels, cfg, true, 4);

        assert_eq!(
            dev_active,
            Some(true),
            "the binary envelope must activate the resident + on-device grad/hess path"
        );
        assert_eq!(
            host_active,
            Some(false),
            "the host arm (boosting_on_cuda=false) must NOT activate the resident path"
        );

        // A real multi-iter train moved the score (non-vacuous parity proof).
        assert!(
            host_scores.iter().any(|&s| s.abs() > 1e-6),
            "a real multi-iter binary train must move the score"
        );
        // BIT-EXACT model (leaf-for-leaf, split-for-split; Tree: PartialEq) + score.
        assert_eq!(
            dev_trees, host_trees,
            "resident on-device grad/hess model must be bit-identical to the host path"
        );
        oracle_harness::comparator::compare_exact_f64_bits(&dev_scores, &host_scores)
            .expect("resident-arm score must equal the host score bit-for-bit (cpu f64 anchor)");
    }

    /// Every case OUTSIDE the envelope (multiclass, bagging, GOSS, an
    /// on-device-unsupported objective) leaves `resident_active` `Some(false)` — the
    /// resident path is NEVER activated (complete, not partial, host fallback), even with
    /// `boosting_on_cuda=true`.
    #[test]
    fn resident_score_stays_inactive_out_of_envelope() {
        let backend = CpuBackend;
        let client = cpu_client();

        // (a) MULTICLASS (num_class=3) — excluded from the envelope.
        {
            let (features, labels, cfg) = multiclass_corpus();
            let num_data = labels.len() as i32;
            let obj = MulticlassSoftmax::new(3, &labels).unwrap();
            let mut learner = SerialTreeLearner::new(&backend, &client, cfg, 2, 1)
                .with_features(features.clone());
            let mut gbdt =
                Gbdt::with_objective(BoostObjective::Multiclass(obj), 0.1, 3, num_data, true, None);
            gbdt.set_boosting_on_cuda(true);
            gbdt.train_one_iter(&mut learner, &labels, features.len())
                .expect("multiclass iter");
            assert_eq!(
                gbdt.resident_score_active(),
                Some(false),
                "multiclass (num_class=3) must never activate the resident path"
            );
        }

        // (b) BAGGING — excluded from the envelope. Also prove byte-identical: the true/false arms
        // (both host subset path, no resident branch) grow the SAME model.
        {
            let (features, labels, cfg) = corpus();
            let num_data = labels.len() as i32;
            let make = |on: bool| {
                let mut learner = SerialTreeLearner::new(&backend, &client, cfg.clone(), 4, -1)
                    .with_features(features.clone());
                let bag_cfg = BaggingConfig::new(0.5, 1.0, 1.0, 1, 3, false).unwrap();
                let strat = BaggingSampleStrategy::reset_sample_config(bag_cfg, num_data, &labels);
                let mut gbdt = Gbdt::new(
                    Objective::Regression { sqrt: false },
                    0.3,
                    1,
                    num_data,
                    true,
                    None,
                )
                .with_bagging(strat, features.clone());
                gbdt.set_boosting_on_cuda(on);
                gbdt.train(&mut learner, &labels, features.len(), 3)
                    .expect("bagging train");
                (gbdt.trees().to_vec(), gbdt.resident_score_active())
            };
            let (trees_on, active_on) = make(true);
            let (trees_off, active_off) = make(false);
            assert_eq!(active_on, Some(false), "bagging must never activate resident");
            assert_eq!(active_off, Some(false), "bagging host arm inactive too");
            assert_eq!(
                trees_on, trees_off,
                "bagging: boosting_on_cuda toggle must be byte-identical (no resident path)"
            );
        }

        // (c) GOSS — excluded from the envelope (goss.is_some()).
        {
            let (features, labels, cfg) = corpus();
            let num_data = labels.len() as i32;
            let goss =
                GossSampleStrategy::reset_sample_config(0.3, 0.2, 2.0, num_data, 1, 3).unwrap();
            let mut learner = SerialTreeLearner::new(&backend, &client, cfg, 4, -1)
                .with_features(features.clone());
            let mut gbdt = Gbdt::new(
                Objective::Regression { sqrt: false },
                2.0,
                1,
                num_data,
                true,
                None,
            )
            .with_goss(goss, features.clone());
            gbdt.set_boosting_on_cuda(true);
            gbdt.train_one_iter(&mut learner, &labels, features.len())
                .expect("goss iter");
            assert_eq!(
                gbdt.resident_score_active(),
                Some(false),
                "GOSS must never activate the resident path"
            );
        }

        // (d) ON-DEVICE-UNSUPPORTED OBJECTIVE (tweedie) — device_objective_supported=false.
        {
            let (features, labels, cfg) = corpus();
            let num_data = labels.len() as i32;
            let mut learner = SerialTreeLearner::new(&backend, &client, cfg, 4, -1)
                .with_features(features.clone());
            let mut gbdt = Gbdt::with_objective(
                BoostObjective::Builtin(Objective::Tweedie { rho: 1.5 }),
                0.3,
                1,
                num_data,
                true,
                None,
            );
            gbdt.set_boosting_on_cuda(true);
            gbdt.train_one_iter(&mut learner, &labels, features.len())
                .expect("tweedie iter");
            assert_eq!(
                gbdt.resident_score_active(),
                Some(false),
                "an on-device-unsupported objective (tweedie) must never activate the resident path"
            );
        }
    }
}
