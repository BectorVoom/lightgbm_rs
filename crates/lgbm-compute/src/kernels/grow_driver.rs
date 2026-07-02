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

use lgbm_dataset::{BinType, LeafPartitionLayout, MissingType};

use crate::error::ComputeError;
use crate::gain::{calculate_splitted_leaf_output, GainConfig, SplitInfo};
use crate::kernels::data_partition::{partition_leaf_stable, update_data_index_to_leaf_on};
use crate::kernels::histogram::construct_histograms_f64_on;
use crate::kernels::split::find_best_split_f64_on;
use crate::kernels::split_info::SplitScalars;
use crate::kernels::subtract::subtract_histograms_f64_on;
use crate::kernels::tree::DeviceCudaTree;
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

// =========================================================================
// 20-03b: the per-leaf best-first on-device grow DRIVER (D-01, ODL-18/ODL-19).
//
// This is the load-bearing STRUCTURE-bit-exact gate D-01 pulled forward from
// Phase 21: it grows an ENTIRE continuous-feature + L2 tree by SEQUENCING the
// already-golden Phase-16 (histogram build + subtract), Phase-4/17 (best-split
// finder), and Phase-18 (`DeviceCudaTree` mutation) kernels into the C++
// `SerialTreeLearner` §6/§16 best-first order — WITHOUT reusing lgbm-treelearner's
// `LeafSplits` / `HistogramPool` / `DataPartition` (those cannot be named from
// lgbm-compute: the crate wall this replan exists to keep). It reproduces the host
// order with its OWN lightweight [`DriverLeaf`] bookkeeping.
//
// ## Faithfulness scope (proving slice — Pitfall 4)
// Continuous features + L2 only: NO categorical, NO L1/smoothing/max_delta_step,
// NO RenewTreeOutput refit, NO col-sampler / monotone / interaction / extra-trees /
// CEGB / forced-splits. The proving corpus is `MissingType::None` (single reverse
// scan). The L1/quantile/categorical follow-up reuses this exact ordering contract
// but adds the missing-value forward preamble + the categorical split kernel.
//
// ## ODL-19 (no f64 per-row grow/build hot loop)
// The only per-ROW device work is the Phase-16 histogram BUILD
// ([`construct_histograms_f64_on`]) and the row PARTITION
// ([`partition_leaf_stable`]) — both operate on integer bins / f32 grad-hess and
// keep the u64/f32 build contract inside the kernel. The f64 that appears here is
// confined to O(num_bin) per-feature histogram post-processing (FixHistogram +
// compaction), the O(num_bin) subtraction, and the reference-blessed scalar
// gain/leaf-value math — NONE of it is a per-row loop.
// =========================================================================

/// The fixed L2 proving-slice gain config the driver grows under (the trait seam
/// [`crate::Backend::grow_tree_on_device`] carries no `GainConfig`, so the driver
/// pins the proving-slice config here and the STRUCTURE gate builds the cpu f64
/// anchor with the IDENTICAL config). Continuous + L2, permissive `min_data`.
#[must_use]
pub fn proving_slice_config() -> GainConfig {
    GainConfig {
        min_data_in_leaf: 1,
        min_sum_hessian_in_leaf: 0.0,
        max_delta_step: 0.0,
        lambda_l1: 0.0,
        lambda_l2: 0.0,
        min_gain_to_split: 0.0,
        path_smooth: 0.0,
        ..Default::default()
    }
}

/// The driver's OWN minimal per-leaf state — the lgbm-compute-local stand-in for
/// the learner's `LeafSplits` + `HistogramPool` slot + `best_split_per_leaf`, using
/// nothing above lgbm-compute (no crate cycle).
struct DriverLeaf {
    /// Global row ids currently in this leaf (partition order).
    rows: Vec<u32>,
    /// The leaf's seeded gradient sum (root = ordered f64 fold; child = the parent
    /// split's `left/right_sum_gradient`, kEpsilon-carrying — NOT a re-fold).
    sum_g: f64,
    /// The leaf's seeded hessian sum.
    sum_h: f64,
    /// This leaf's per-feature CONCATENATED fixed+compacted histogram (the parent
    /// buffer the subtraction trick derives the larger child from).
    hist: Vec<f64>,
    /// The leaf's best split (`gain == -inf` ⇒ no admissible split).
    best: SplitInfo,
    /// The winning feature POSITION (`-1` when no split); its real index is
    /// `features[best_fpos].real_feature_index`.
    best_fpos: i32,
    /// The leaf's depth (root = 0), for the `max_depth` gate.
    depth: i32,
}

/// `SerialTreeLearner`'s cross-feature / cross-leaf argmax tie rule
/// (`split_info.rs::split_gt`): strictly-greater gain wins; on an exact gain tie the
/// LOWER real feature index wins (`-1` ⇒ `i32::MAX`).
fn split_gt(a: &SplitInfo, a_feat: i32, b: &SplitInfo, b_feat: i32) -> bool {
    if a.gain != b.gain {
        return a.gain > b.gain;
    }
    let af = if a_feat == -1 { i32::MAX } else { a_feat };
    let bf = if b_feat == -1 { i32::MAX } else { b_feat };
    af < bf
}

/// C++ `FixHistogram` on the RAW leaf sums (`feature_histogram` — Pitfall 2). A
/// no-op for `most_freq_bin == 0` (the proving corpus). O(num_bin) scalar f64 fold
/// (ascending bin order is load-bearing — never reorder). NOT a per-row loop.
fn fix_histogram(hist: &mut [f64], most_freq_bin: u32, sum_gradient: f64, sum_hessian: f64) {
    if most_freq_bin == 0 {
        return;
    }
    let num_bin = hist.len() / 2;
    let mfb = most_freq_bin as usize;
    if mfb >= num_bin {
        return;
    }
    let g_idx = mfb << 1;
    let h_idx = g_idx + 1;
    let mut g = sum_gradient;
    let mut h = sum_hessian;
    for i in 0..num_bin {
        if i != mfb {
            g -= hist[i << 1];
            h -= hist[(i << 1) + 1];
        }
    }
    hist[g_idx] = g;
    hist[h_idx] = h;
}

/// C++ compacted-histogram shift (`offset` drops the leading `offset` bins). A
/// no-op for `offset == 0`. O(num_bin) scalar f64 copy. NOT a per-row loop.
fn compact_histogram(hist: &mut [f64], offset: i32) {
    if offset <= 0 {
        return;
    }
    let off = offset as usize;
    let num_bin = hist.len() / 2;
    if off >= num_bin {
        for cell in hist.iter_mut() {
            *cell = 0.0;
        }
        return;
    }
    for c in 0..(num_bin - off) {
        let dst = c << 1;
        let src = (c + off) << 1;
        hist[dst] = hist[src];
        hist[dst + 1] = hist[src + 1];
    }
    for cell in hist.iter_mut().skip((num_bin - off) << 1) {
        *cell = 0.0;
    }
}

/// Build one leaf's per-feature CONCATENATED fixed+compacted histogram by
/// DIRECTLY constructing each feature's raw histogram over the leaf's rows
/// (Phase-16 [`construct_histograms_f64_on`]), then FixHistogram + compacting each
/// feature region in place. `slot_off[fpos]` is the feature's start cell.
fn build_leaf_hist<R: cubecl::Runtime>(
    client: &cubecl::prelude::ComputeClient<R>,
    features: &[GrowFeature],
    gradients: &[f32],
    hessians: &[f32],
    rows: &[u32],
    sum_g: f64,
    sum_h: f64,
    slot_off: &[usize],
    hist_len: usize,
) -> Result<Vec<f64>, ComputeError> {
    let mut concat = vec![0.0f64; hist_len];
    if rows.is_empty() {
        return Ok(concat);
    }
    // Gather the leaf's ordered grad/hess ONCE (shared across features).
    let g: Vec<f32> = rows.iter().map(|&r| gradients[r as usize]).collect();
    let h: Vec<f32> = rows.iter().map(|&r| hessians[r as usize]).collect();
    for (fpos, f) in features.iter().enumerate() {
        let binned: Vec<u32> = rows.iter().map(|&r| f.bins.bin(r as usize)).collect();
        // Phase-16 device build: RAW f64 histogram (2*num_bin cells).
        let mut region = construct_histograms_f64_on(client, &binned, &g, &h, f.num_bin)?;
        // FixHistogram (RAW leaf sums) then compact — O(num_bin) f64, bit-exact to
        // the host reference fold (mfb==0 ⇒ fix is a no-op; offset==1 ⇒ drop bin 0).
        fix_histogram(&mut region, f.most_freq_bin, sum_g, sum_h);
        compact_histogram(&mut region, f.offset);
        let cells = 2 * f.num_bin as usize;
        concat[slot_off[fpos]..slot_off[fpos] + cells].copy_from_slice(&region);
    }
    Ok(concat)
}

/// Scan one leaf's concatenated compacted histogram: per-feature Phase-4/17
/// [`find_best_split_f64_on`] + the cross-feature `split_gt` argmax. Returns the
/// winning `(SplitInfo, feature-position)` (`(-inf, -1)` when nothing is admissible).
fn scan_leaf<R: cubecl::Runtime>(
    client: &cubecl::prelude::ComputeClient<R>,
    features: &[GrowFeature],
    hist: &[f64],
    sum_g: f64,
    sum_h: f64,
    num_data_in_leaf: i32,
    slot_off: &[usize],
    cfg: &GainConfig,
) -> Result<(SplitInfo, i32), ComputeError> {
    let mut best = SplitInfo::none();
    let mut best_fpos: i32 = -1;
    let mut best_real: i32 = -1;
    #[allow(clippy::neg_cmp_op_on_partial_ord)]
    if !(sum_h > 0.0) || num_data_in_leaf <= 0 {
        return Ok((best, best_fpos));
    }
    for (fpos, f) in features.iter().enumerate() {
        // Proving slice: continuous MissingType::None ⇒ single REVERSE scan, no
        // default-bin skip, no NA forward preamble.
        let skip_default_bin = f.num_bin > 2 && f.missing_type == MissingType::Zero;
        let na_as_missing = f.num_bin > 2 && f.missing_type == MissingType::NaN;
        let run_forward = f.num_bin > 2 && f.missing_type == MissingType::Zero;
        if na_as_missing || f.bin_type == BinType::Categorical {
            // Deferred (proving slice is numeric + non-NA); the finder rejects NA.
            continue;
        }
        let cells = 2 * f.num_bin as usize;
        let region = &hist[slot_off[fpos]..slot_off[fpos] + cells];
        let si = find_best_split_f64_on(
            client,
            region,
            cfg,
            f.num_bin,
            f.offset,
            f.default_bin,
            f.most_freq_bin,
            skip_default_bin,
            na_as_missing,
            run_forward,
            sum_g,
            sum_h,
            num_data_in_leaf,
        )?;
        if split_gt(&si, f.real_feature_index, &best, best_real) {
            best = si;
            best_fpos = fpos as i32;
            best_real = f.real_feature_index;
        }
    }
    Ok((best, best_fpos))
}

/// Grow an ENTIRE continuous-feature + L2 tree ON-DEVICE by sequencing the
/// Phase-16/17/18 kernels in the `SerialTreeLearner` best-first order, using the
/// driver's OWN [`DriverLeaf`] bookkeeping (D-01/ODL-18). Returns the grown
/// [`lgbm_model::Tree`] and the final row→leaf [`LeafPartitionLayout`].
///
/// Thin delegator to [`grow_tree_on_device_driver_with_cfg`] pinned to the fixed
/// [`proving_slice_config`] the [`crate::Backend::grow_tree_on_device`] trait seam
/// grows under (the seam carries no `GainConfig`, so the default merge-gate anchor
/// is unchanged). Tests that need a constrained gain config (e.g. binding
/// `min_data_in_leaf`) call the `_with_cfg` variant directly.
///
/// Runs on ANY `R` (the STRUCTURE gate drives it on the cubecl-cpu runtime, anchored
/// to the cpu f64 fold — never GPU-vs-GPU, def-f8u-01). `max_depth <= 0` ⇒ no depth
/// cap.
///
/// # Errors
/// [`ComputeError`] from any sequenced kernel (bad num_bin / histogram length,
/// out-of-range bin, device launch), or an empty feature set / non-positive
/// `num_leaves`.
pub fn grow_tree_on_device_driver<R: cubecl::Runtime>(
    client: &cubecl::prelude::ComputeClient<R>,
    gradients: &[f32],
    hessians: &[f32],
    features: &[GrowFeature],
    num_leaves: i32,
    max_depth: i32,
) -> Result<(lgbm_model::Tree, LeafPartitionLayout), ComputeError> {
    grow_tree_on_device_driver_with_cfg(
        client,
        gradients,
        hessians,
        features,
        num_leaves,
        max_depth,
        proving_slice_config(),
    )
}

/// Extract-parameter variant of [`grow_tree_on_device_driver`] that grows the tree
/// under an EXPLICIT `cfg: GainConfig` instead of the pinned [`proving_slice_config`].
/// The body is otherwise identical to the delegator (same length/num_leaves guards,
/// ordered-f64 root fold, build/subtract/scan/partition sequence, break path, and
/// typed `ComputeError` boundaries). Additive: the `Backend::grow_tree_on_device`
/// trait seam is untouched, so the default `LGBM_CUDA_ON_DEVICE`-unset merge gate is
/// byte-unchanged. Threading `cfg` here lets a test make a constraint (e.g.
/// `min_data_in_leaf`) observably bind through the driver without widening the seam.
///
/// # Errors
/// [`ComputeError`] from any sequenced kernel (bad num_bin / histogram length,
/// out-of-range bin, device launch), or an empty feature set / non-positive
/// `num_leaves`.
pub fn grow_tree_on_device_driver_with_cfg<R: cubecl::Runtime>(
    client: &cubecl::prelude::ComputeClient<R>,
    gradients: &[f32],
    hessians: &[f32],
    features: &[GrowFeature],
    num_leaves: i32,
    max_depth: i32,
    cfg: GainConfig,
) -> Result<(lgbm_model::Tree, LeafPartitionLayout), ComputeError> {
    if num_leaves < 1 {
        return Err(ComputeError::Runtime {
            detail: format!("grow_tree_on_device_driver: num_leaves must be >= 1, got {num_leaves}"),
        });
    }
    if features.is_empty() {
        return Err(ComputeError::Runtime {
            detail: "grow_tree_on_device_driver: at least one feature is required".to_string(),
        });
    }
    let num_data = gradients.len();
    if hessians.len() != num_data {
        return Err(ComputeError::LengthMismatch {
            expected: num_data,
            actual: hessians.len(),
        });
    }
    let min_data = cfg.min_data_in_leaf;

    // Per-feature concatenated-histogram offsets (2*num_bin cells each).
    let mut slot_off = Vec::with_capacity(features.len());
    let mut hist_len = 0usize;
    for f in features {
        slot_off.push(hist_len);
        hist_len += 2 * f.num_bin as usize;
    }

    // ---- Root init (§6.1): whole-dataset ordered f64 fold (== LeafSplits::init). ----
    let root_rows: Vec<u32> = (0..num_data as u32).collect();
    let mut root_sum_g = 0.0f64;
    let mut root_sum_h = 0.0f64;
    for &r in &root_rows {
        root_sum_g += f64::from(gradients[r as usize]);
        root_sum_h += f64::from(hessians[r as usize]);
    }
    let root_hist = build_leaf_hist(
        client, features, gradients, hessians, &root_rows, root_sum_g, root_sum_h, &slot_off,
        hist_len,
    )?;
    let (root_best, root_fpos) = scan_leaf(
        client, features, &root_hist, root_sum_g, root_sum_h, num_data as i32, &slot_off, &cfg,
    )?;

    let mut leaves: Vec<DriverLeaf> = vec![DriverLeaf {
        rows: root_rows,
        sum_g: root_sum_g,
        sum_h: root_sum_h,
        hist: root_hist,
        best: root_best,
        best_fpos: root_fpos,
        depth: 0,
    }];

    // ---- The device flat tree (Phase-18), pre-allocated once (D-15). ----
    let mut tree = DeviceCudaTree::<R>::new(client, num_leaves.max(1) as usize, num_data as i32)?;
    // Seed the root leaf value so a never-split root still matches the anchor.
    let root_output =
        calculate_splitted_leaf_output(cfg.use_l1(), root_sum_g, root_sum_h, cfg.lambda_l1, cfg.lambda_l2);
    tree.add_bias(client, root_output);

    // ---- The best-first leaf-wise loop (serial_tree_learner.cpp:218-236). ----
    for _split in 0..(num_leaves - 1) {
        // best_leaf = argmax(best_split_per_leaf) via split_gt (first-max).
        let mut best_leaf = 0i32;
        for i in 1..leaves.len() {
            let a = &leaves[i];
            let a_feat = if a.best_fpos < 0 {
                -1
            } else {
                features[a.best_fpos as usize].real_feature_index
            };
            let b = &leaves[best_leaf as usize];
            let b_feat = if b.best_fpos < 0 {
                -1
            } else {
                features[b.best_fpos as usize].real_feature_index
            };
            if split_gt(&a.best, a_feat, &b.best, b_feat) {
                best_leaf = i as i32;
            }
        }
        let best = leaves[best_leaf as usize].best;
        let best_fpos = leaves[best_leaf as usize].best_fpos;
        // No positive-gain split anywhere ⇒ stop (best_leaf == -1 sentinel analog).
        if best_fpos < 0 || !(best.gain > 0.0) {
            break;
        }

        let f = &features[best_fpos as usize];
        let parent_depth = leaves[best_leaf as usize].depth;
        let parent_hist = leaves[best_leaf as usize].hist.clone();

        // ---- Partition the parent leaf's rows (Phase-18 route), BEFORE the tree
        // mutation reads the child ids. Single-feature-group min_bin convention
        // (CR-01): partition_min_bin = min_bin + offset. ----
        let parent_rows = leaves[best_leaf as usize].rows.clone();
        let bins_sub = BinColumn::new(
            parent_rows.iter().map(|&r| f.bins.bin(r as usize)).collect(),
            f.num_bin,
        );
        let missing_type_u8 = match f.missing_type {
            MissingType::None => 0u8,
            MissingType::Zero => 1,
            MissingType::NaN => 2,
        };
        let partition_min_bin = f.min_bin + f.offset.max(0) as u32;
        let (reordered, split_point) = partition_leaf_stable(
            &bins_sub,
            &parent_rows,
            f.num_bin,
            partition_min_bin,
            f.max_bin,
            f.default_bin,
            f.most_freq_bin,
            missing_type_u8,
            best.default_left,
            best.threshold,
        )?;
        let left_rows: Vec<u32> = reordered[..split_point].to_vec();
        let right_rows: Vec<u32> = reordered[split_point..].to_vec();
        let left_count = left_rows.len() as i32;
        let right_count = right_rows.len() as i32;

        // ---- Grow the node ON DEVICE (Phase-18 DeviceCudaTree::split_on_device),
        // consuming its right_leaf_index. The tree's leaf/internal counts take the
        // ACTUAL partition counts (serial_tree_learner.cpp:788-791). ----
        let new_left = best_leaf;
        let real_threshold = f
            .bin_upper_bound
            .get(best.threshold as usize)
            .copied()
            .unwrap_or(best.threshold as f64);
        let missing_type_code = i32::from(missing_type_u8);
        let scalars = SplitScalars {
            is_valid: true,
            leaf_index: best_leaf,
            gain: best.gain + cfg.min_gain_to_split,
            inner_feature_index: f.real_feature_index,
            threshold: best.threshold,
            default_left: best.default_left,
            left_sum_gradients: best.left_sum_gradient,
            left_sum_hessians: best.left_sum_hessian,
            left_sum_gh_quant: 0,
            left_count,
            left_gain: 0.0,
            left_value: best.left_output,
            right_sum_gradients: best.right_sum_gradient,
            right_sum_hessians: best.right_sum_hessian,
            right_sum_gh_quant: 0,
            right_count,
            right_gain: 0.0,
            right_value: best.right_output,
            num_cat_threshold: 0,
        };
        let result = tree.split_on_device(
            client,
            best_leaf,
            f.real_feature_index,
            real_threshold,
            missing_type_code,
            &scalars,
        )?;
        let new_right = result.right_leaf_index;

        // ---- Seed the two child leaves from the SplitInfo (NOT a re-fold): the
        // kEpsilon-carrying sums are load-bearing for the next split (Pitfall 2). ----
        let child_depth = parent_depth + 1;
        // Update the reused left child in place.
        {
            let l = &mut leaves[best_leaf as usize];
            l.rows = left_rows;
            l.sum_g = best.left_sum_gradient;
            l.sum_h = best.left_sum_hessian;
            l.depth = child_depth;
            l.best = SplitInfo::none();
            l.best_fpos = -1;
        }
        // Append the new right child (leaf id == new_right).
        debug_assert_eq!(new_right as usize, leaves.len(), "right child takes the next leaf id");
        leaves.push(DriverLeaf {
            rows: right_rows,
            sum_g: best.right_sum_gradient,
            sum_h: best.right_sum_hessian,
            hist: vec![0.0; hist_len],
            best: SplitInfo::none(),
            best_fpos: -1,
            depth: child_depth,
        });

        // ---- Build the children histograms: SMALLER directly, LARGER by
        // subtraction from the PARENT (Phase-16 subtract, parent-built-before-child).
        // Smaller = the fewer-row child (num_left < num_right ⇒ left, else right). ----
        let smaller_is_left = left_count < right_count;
        let (smaller_leaf, larger_leaf) = if smaller_is_left {
            (new_left, new_right)
        } else {
            (new_right, new_left)
        };
        let (s_rows, s_g, s_h) = {
            let s = &leaves[smaller_leaf as usize];
            (s.rows.clone(), s.sum_g, s.sum_h)
        };
        let smaller_hist = build_leaf_hist(
            client, features, gradients, hessians, &s_rows, s_g, s_h, &slot_off, hist_len,
        )?;
        // LARGER = parent − smaller (Phase-16 subtract kernel), over the whole
        // concatenated compacted buffer (zeroed tails subtract to zero).
        let larger_hist = subtract_histograms_f64_on(client, &parent_hist, &smaller_hist)?;
        leaves[smaller_leaf as usize].hist = smaller_hist;
        leaves[larger_leaf as usize].hist = larger_hist;

        // ---- BeforeFindBestSplit gates + scan each child (compute its best). ----
        let both_too_small = left_count < min_data * 2 && right_count < min_data * 2;
        for &child in &[new_left, new_right] {
            let depth_capped = max_depth > 0 && leaves[child as usize].depth >= max_depth;
            if depth_capped || both_too_small {
                leaves[child as usize].best = SplitInfo::none();
                leaves[child as usize].best_fpos = -1;
                continue;
            }
            let (cg, ch, cn, chist) = {
                let c = &leaves[child as usize];
                (c.sum_g, c.sum_h, c.rows.len() as i32, c.hist.clone())
            };
            let (cbest, cfpos) =
                scan_leaf(client, features, &chist, cg, ch, cn, &slot_off, &cfg)?;
            leaves[child as usize].best = cbest;
            leaves[child as usize].best_fpos = cfpos;
        }
    }

    // ---- Reconstruct the host tree (Phase-18 to_host_tree) + the row→leaf layout. ----
    let host_tree = tree.to_host_tree(client);
    let final_leaves = host_tree.num_leaves as usize;
    let mut indices = Vec::with_capacity(num_data);
    let mut leaf_begin = Vec::with_capacity(final_leaves);
    let mut leaf_count = Vec::with_capacity(final_leaves);
    for leaf in leaves.iter().take(final_leaves) {
        leaf_begin.push(indices.len() as i32);
        leaf_count.push(leaf.rows.len() as i32);
        indices.extend_from_slice(&leaf.rows);
    }
    let layout = LeafPartitionLayout {
        num_data: num_data as i32,
        indices,
        leaf_begin,
        leaf_count,
    };
    Ok((host_tree, layout))
}
