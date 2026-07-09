//! `lgbm-boosting` — the GBDT spine: the outer gradient-boosting loop, score
//! accumulation, and bagging sample strategy.
//!
//! Faithful 1:1 port target: `LightGBM/src/boosting/` (`gbdt.cpp` `TrainOneIter`
//! ensemble loop, `score_updater.hpp` f64 accumulator, `sample_strategy.cpp`
//! bagging RNG). This crate drives the Phase-5 [`lgbm_treelearner`]
//! `SerialTreeLearner` via its existing public API — it owns the orchestration,
//! NOT the per-tree growth.
//!
//! ## CMP-01 containment
//! Like `lgbm-treelearner`, this crate has NO `cubecl` / `lgbm-compute`
//! dependency: the compute seam stays isolated below the learner. The boosting
//! loop names only `lgbm-objective` (grad/hess), `lgbm-metric` (eval),
//! `lgbm-treelearner` (grow tree), and `lgbm-model` (Tree / ensemble).
//!
//! Wave-0 scaffold (06-01): this plan creates the compiling skeleton + the
//! [`error`] boundary only. The `gbdt` loop, `score_updater`, and
//! `sample_strategy` modules (with the verbatim C++ `TrainOneIter` ordering —
//! BoostFromAverage → GetGradients → Bagging → per-class tree → RenewTreeOutput →
//! Shrinkage → UpdateScore → AddBias) land in 06-02 (L2 spine) and 06-05
//! (bagging/early-stopping).

pub mod early_stopping;
pub mod error;
pub mod gbdt;
pub mod objective;
pub mod sample_strategy;
pub mod score_updater;

pub use early_stopping::{EarlyStopping, EvalSnapshot, MetricSpec};
pub use error::BoostingError;
pub use gbdt::{BoostingVariant, DartConfig, Gbdt, IterSnapshot, RfConfig};
pub use objective::BoostObjective;
pub use sample_strategy::{
    BaggingConfig, BaggingSampleStrategy, GossSampleStrategy, BAGGING_RAND_BLOCK,
};
pub use score_updater::ScoreUpdater;
