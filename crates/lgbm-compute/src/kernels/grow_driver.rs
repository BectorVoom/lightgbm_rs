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
