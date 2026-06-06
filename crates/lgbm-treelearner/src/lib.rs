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

/// THE single authoritative threshold-offset rule (`meta_->offset`), shared by
/// the runtime learner AND every harness corpus builder — there is exactly ONE
/// place in the workspace that computes `most_freq_bin -> offset`.
///
/// Returns `1` when `most_freq_bin == 0`, else `0`, EXACTLY mirroring the C++
/// `FeatureHistogram` meta initialization
/// (`LightGBM/src/treelearner/feature_histogram.hpp:1429-1433`:
/// `if (GetMostFreqBin() == 0) offset = 1; else offset = 0;`) and the
/// `Dataset::CreateCUDAColumnData` mirror
/// (`src/io/dataset.cpp:1804`: `feature_offsets[i] = (most_freq_bin == 0)`).
///
/// ## Why offset==1 for most_freq_bin==0 (the compacted-histogram convention)
/// When the most-frequent bin is bin 0 it is the implicit default and is NEVER
/// directly folded into the histogram, so the per-feature histogram is COMPACTED:
/// the bin-0 slot is dropped and cell `c` holds REAL bin `c + offset`. The scan
/// ranges (`num_bin - offset` cells, `feature_histogram.hpp:943/950`), the
/// `GET_GRAD(data_, threshold - meta_->offset)` cell index
/// (`feature_histogram.hpp:619`), and the bin-layout semantics
/// (`include/LightGBM/bin.h:180-258`, `most_freq_bin_`/`default_bin_`) all assume
/// this compaction. For `most_freq_bin > 0` the histogram is non-compacted
/// (`offset == 0`, cell == real bin).
///
/// ## Supersedes D-01 (per D-09)
/// The earlier port stored `offset == 0` for `most_freq_bin == 0` with a
/// non-compacted histogram, which disagreed with the verbatim `--th` partition
/// adjustment (`dense_bin.hpp:324-327`) and produced a partition the serialized
/// tree's own `get_leaf` did not predict into (CR-01). D-09 adopts the real
/// `offset == 1` + compacted convention end-to-end; THIS helper is the one place
/// that rule lives.
#[must_use]
pub fn offset_for_most_freq_bin(most_freq_bin: u32) -> i32 {
    if most_freq_bin == 0 {
        1
    } else {
        0
    }
}

#[cfg(test)]
mod offset_tests {
    use super::offset_for_most_freq_bin;

    #[test]
    fn offset_is_one_only_for_most_freq_bin_zero() {
        assert_eq!(offset_for_most_freq_bin(0), 1, "most_freq_bin==0 -> offset 1");
        for n in 1u32..=8 {
            assert_eq!(
                offset_for_most_freq_bin(n),
                0,
                "most_freq_bin {n} (>0) -> offset 0"
            );
        }
    }
}
