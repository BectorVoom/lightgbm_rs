//! `lgbm-treelearner` — the histogram-based serial tree learner (D-04 spine).
//!
//! Faithful 1:1 port of LightGBM's `SerialTreeLearner`
//! (`src/treelearner/serial_tree_learner.cpp` + `serial_tree_learner.h`): the
//! leaf-wise growth loop, histogram construction + subtraction trick, cross-
//! feature split finding, and data partitioning that grows ONE decision tree per
//! `Train` call.
//!
//! ## Crate boundaries
//! - The learner takes a `&impl lgbm_compute::Backend` and NEVER names a cubecl
//!   runtime — all compute is dispatched through the [`Backend`](lgbm_compute::Backend)
//!   seam (CMP-01 containment; this crate has NO `cubecl` dependency).
//! - Caller input is validated to a typed [`TreeLearnerError`] (Security V5,
//!   threat T-05-02-01) before any reduction/division; backend failures are
//!   wrapped via `#[from]`.
//! - The split-result struct is REUSED from [`lgbm_compute::gain::SplitInfo`]
//!   (re-exported here as [`SplitInfo`]) — there is exactly one in the workspace.
//!
//! ## Plan status
//! This plan (Phase 5 Plan 02) delivers the Wave-1 enabling slice only: the
//! crate skeleton, the [`error`] boundary, and the [`split_info`] tie-break
//! helper. The `SerialTreeLearner` orchestrator (the `learner` module) lands in
//! Plan 03 against these contracts.

pub mod col_sampler;
pub mod data_partition;
pub mod error;
pub mod fix_histogram;
pub mod histogram_pool;
pub mod leaf_splits;
pub mod learner;
pub mod split_info;

pub use col_sampler::ColSampler;
pub use data_partition::DataPartition;
pub use error::TreeLearnerError;
pub use fix_histogram::fix_histogram;
pub use histogram_pool::HistogramPool;
pub use leaf_splits::LeafSplits;
pub use learner::{BuildStrategy, FeatureColumn, SerialTreeLearner};
pub use split_info::{split_gt, SplitInfo};
