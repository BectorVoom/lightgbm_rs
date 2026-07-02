//! On-device tree-growth driver seam — the ADDITIVE feature/bin metadata the
//! per-leaf grow loop consumes, expressed in ONLY lgbm-compute-reachable types
//! (Option A, D-01, ODL-18/ODL-19).
//!
//! ## Why this struct lives HERE (not in lgbm-treelearner)
//! The learner's per-feature spine column carries the same bin layout, but it lives
//! in `lgbm-treelearner`, which depends on `lgbm-compute`. Naming that learner type
//! from `lgbm-compute` (so [`crate::Backend::grow_tree_on_device`] could take it)
//! would form the crate cycle `treelearner → compute → treelearner` this replan
//! exists to avoid. Instead [`GrowFeature`] is a faithful, lgbm-compute-local MIRROR
//! of exactly the fields the Phase-16/17/18 kernels read — built from types that are
//! ALREADY reachable below `lgbm-compute`:
//! - [`BinColumn`] — lgbm-compute-local (defined in `lib.rs`).
//! - [`BinType`] / [`MissingType`] — `lgbm-dataset` (a dependency of `lgbm-compute`).
//! - primitive slices (`u32`/`i32`/`f64`).
//!
//! The learner builds a `Vec<GrowFeature>` from its `Vec` of spine columns
//! field-by-field at the on-device fork and passes `&grow_features` across the seam.
//! The
//! driver derives the split kernel's [`crate::kernels::best_split::FeatureMeta`]
//! from `GrowFeature` internally in 20-03b; this plan lands ONLY the metadata
//! carrier (the driver body still returns `Ok(None)`).
//!
//! Additive and OFF by default behind `LGBM_CUDA_ON_DEVICE` (D-09); ungated like
//! the other Phase-14..19 kernel modules (NOT `#[cfg(feature = "gpu")]`) so the
//! default cpu f64 anchor exercises the plumbing (D-08).

use lgbm_dataset::{BinType, MissingType};

use crate::error::ComputeError;
use crate::kernels::data_partition::update_data_index_to_leaf_on;
use crate::BinColumn;

/// One feature column's ADDITIVE grow-loop input — the faithful lgbm-compute-local
/// mirror of the fields the learner's spine feature column exposes to the
/// Phase-16/17/18 device kernels, using ONLY lgbm-compute-reachable types so the
/// seam never names a treelearner type (no crate cycle, D-01/Option A).
///
/// Field-for-field parity with the learner's spine column (minus the categorical
/// `bin_to_category` table, which the on-device numeric grow loop does not consume
/// this milestone). Every field is a plain value / `lgbm-dataset` enum / narrow
/// [`BinColumn`] — nothing here reaches up into `lgbm-treelearner`.
#[derive(Debug, Clone)]
pub struct GrowFeature {
    /// Per-GLOBAL-ROW bin index, length `num_data`, in the narrowest unsigned type
    /// for `num_bin` (mirrors the spine column's `bins`). lgbm-compute-local.
    pub bins: BinColumn,
    /// C++ `num_bin` — this feature's bin count (histogram has `2*num_bin` cells).
    pub num_bin: u32,
    /// C++ threshold-offset descriptor (`meta_->offset`), from
    /// `offset_for_most_freq_bin` at the boundary.
    pub offset: i32,
    /// C++ `min_bin` — the feature's first bin (partition lower bound).
    pub min_bin: u32,
    /// C++ `max_bin` — the feature's last bin (partition upper bound).
    pub max_bin: u32,
    /// C++ `default_bin_` (`ValueToBin(0)`) — drives the SKIP_DEFAULT_BIN continue.
    pub default_bin: u32,
    /// C++ `most_freq_bin_` — drives `FixHistogram` + the partition default dir.
    pub most_freq_bin: u32,
    /// C++ `missing_type_` — derives `skip_default_bin` / `na_as_missing` dispatch.
    pub missing_type: MissingType,
    /// Real-value per-bin upper bounds (`bin_upper_bound_`) — the split threshold
    /// the tree records (`threshold[bin] == bin_upper_bound_[bin]`).
    pub bin_upper_bound: Vec<f64>,
    /// The ORIGINAL feature index (`real_feature_idx_`) the tree records + predict
    /// traverses.
    pub real_feature_index: i32,
    /// C++ `BinMapper::bin_type()` — numeric vs categorical dispatch flag.
    pub bin_type: BinType,
}

// =========================================================================
// data->leaf map buffer-strategy A/B harness (Pitfall 3, ODL-19).
//
// The per-split `UpdateDataIndexToLeafIndex` rewrite reads the row->leaf map for
// the split leaf's rows and writes the two child leaf ids. When the 20-03b driver
// grows num_leaves>2 it applies this rewrite REPEATEDLY over one running map. The
// open aliasing question (RESEARCH Pitfall 3 / phase18-wr01 HistArena::swap): does
// the driver read+write ONE map buffer in place (ALIAS), or read a source buffer
// and write a distinct destination then swap (DOUBLE-BUFFER)? A wrong alias choice
// silently corrupts the partition at num_leaves>2. This helper exposes BOTH so the
// oracle A/B can anchor each to the cpu f64 partition and LOCK the safe strategy
// (double-buffer unless alias is proven bit-identical) BEFORE 20-03b writes the
// driver body. Each step drives the REAL Phase-18 device kernel
// (`update_data_index_to_leaf_on`); the strategies differ ONLY in how the running
// map buffer is carried across steps.
// =========================================================================

/// The data->leaf map buffer strategy for the per-split rewrite (Pitfall 3).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LeafMapBufferStrategy {
    /// (A) In-place alias — one running map buffer read AND written each step.
    Alias,
    /// (B) Ping-pong double-buffer — read the source map, write a distinct
    /// destination, swap. The conservative default (no read/write aliasing).
    DoubleBuffer,
}

/// One split's row->leaf rewrite input for [`build_leaf_map_on`]: the global row
/// ids currently in the leaf being split, their `to_left` marks (1 = routes to
/// `left_leaf`, 0 = routes to `right_leaf`, aligned to `data_indices`), and the two
/// child leaf ids.
#[derive(Clone, Copy, Debug)]
pub struct LeafMapStep<'a> {
    /// Global row ids in the leaf being split.
    pub data_indices: &'a [u32],
    /// Per-`data_indices` route marks (`1` = left child, `0` = right child).
    pub to_left: &'a [u32],
    /// The left child leaf id.
    pub left_leaf: i32,
    /// The right child leaf id.
    pub right_leaf: i32,
}

/// Apply `steps` in order to build the final `num_data`-length row->leaf map,
/// starting from `init_leaf` for every row, using the chosen buffer `strategy`
/// (Pitfall 3). Each step drives the real Phase-18
/// [`update_data_index_to_leaf_on`] device kernel (which writes the two child leaf
/// ids for the leaf's rows into a fresh `-1` map); the running map is then carried
/// forward either in place ([`LeafMapBufferStrategy::Alias`]) or via a swapped
/// destination copy ([`LeafMapBufferStrategy::DoubleBuffer`]). Rows a step does not
/// touch keep their prior leaf id. Both strategies MUST equal the cpu f64 partition
/// anchor — the A/B proves it and locks the safe one.
///
/// # Errors
/// [`ComputeError`] from [`update_data_index_to_leaf_on`] (length mismatch, or a
/// `data_index >= num_data`).
pub fn build_leaf_map_on<R: cubecl::Runtime>(
    client: &cubecl::prelude::ComputeClient<R>,
    num_data: usize,
    init_leaf: i32,
    steps: &[LeafMapStep<'_>],
    strategy: LeafMapBufferStrategy,
) -> Result<Vec<i32>, ComputeError> {
    let mut running = vec![init_leaf; num_data];
    for step in steps {
        // Real Phase-18 device kernel: a fresh -1 map with ONLY this leaf's rows set
        // to their child leaf id (rest stay -1). This is the read-side result the
        // buffer strategy then folds into the running map.
        let per_split = update_data_index_to_leaf_on(
            client,
            step.data_indices,
            step.to_left,
            num_data,
            step.left_leaf,
            step.right_leaf,
        )?;
        match strategy {
            LeafMapBufferStrategy::Alias => {
                // (A) in-place: read AND write the SAME running buffer.
                for (row, &v) in per_split.iter().enumerate() {
                    if v != -1 {
                        running[row] = v;
                    }
                }
            }
            LeafMapBufferStrategy::DoubleBuffer => {
                // (B) ping-pong: read the source `running`, write a distinct `next`,
                // then swap. No read/write aliasing of a single buffer.
                let mut next = running.clone();
                for (row, &v) in per_split.iter().enumerate() {
                    if v != -1 {
                        next[row] = v;
                    }
                }
                running = next;
            }
        }
    }
    Ok(running)
}
