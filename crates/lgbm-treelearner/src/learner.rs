//! `SerialTreeLearner` — the leaf-wise (best-first) growth loop (the D-04 spine).
//!
//! Faithful 1:1 port of `SerialTreeLearner::Train` + `BeforeTrain` +
//! `BeforeFindBestSplit` + `FindBestSplits` + `SplitInner`
//! (`LightGBM/src/treelearner/serial_tree_learner.cpp`, commit 195c26fc,
//! VERSION 4.6.0.99) for the pinned spine config: `force_row_wise=true`,
//! `feature_fraction=1.0` (no subsampling → every feature used), numeric splits,
//! `missing_type == None` (NA_AS_MISSING deferred, RESEARCH A5). It grows ONE
//! tree from FIXED g/h — there is no boosting loop (Phase 6) and no objective.
//!
//! ## What it drives (Phase-4 Backend ops — no new numerics)
//! - `construct_histograms` (smaller leaf) → host-side `Vec<f64>` histogram.
//! - `subtract_histograms` (larger leaf = parent − smaller) when `use_subtract`.
//! - `fix_histogram` (learner-side, RAW leaf sums — Pitfall 2) before each scan.
//! - `find_best_split` (per feature; gain math IN-kernel — D-01a, Anti-Pattern:
//!   never re-derive per-bin gains here).
//! - `data_partition` (row→leaf reorder, TRL-07) via [`DataPartition::split`].
//! - `Tree::split` (grow the node + two child leaves, D-07).
//!
//! ## Keystone fidelity points
//! - Smaller-child SELECTION drives the subtraction trick off
//!   `DataPartition::leaf_count` (Pitfall 3): `num_data_in_left < num_data_in_right`
//!   ⇒ smaller = left (the parent buffer is `Move`d to the larger child).
//! - Cross-feature argmax is a FLAT `Vec<SplitInfo>` + first-max scan using
//!   [`split_gt`] (gain, then smaller feature) — NEVER an ordered/priority-queue
//!   container (RESEARCH Standard Stack: a heap would change tie-break order).
//! - `min_gain_to_split` is added back ONLY for the tree-model `split_gain` field
//!   (`serial_tree_learner.cpp:804`), NOT for selection (the kernel already
//!   returns gain net of `min_gain_shift`).
//!
//! ## Dropped branches (Phase-7+ / project-dropped)
//! Every `use_quantized_grad` / monotone / cegb / linear / categorical branch is
//! ignored — only the no-constraint, non-linear, numeric default path is ported.

use lgbm_compute::error::ComputeError;
use lgbm_compute::gain::{calculate_splitted_leaf_output, GainConfig};
use lgbm_compute::Backend;
use lgbm_compute::ComputeClientReexport as ComputeClient;
use lgbm_dataset::bin_mapper::MissingType;
use lgbm_model::Tree;

use crate::col_sampler::ColSampler;
use crate::data_partition::DataPartition;
use crate::error::TreeLearnerError;
use crate::HistogramPool;
use crate::leaf_splits::LeafSplits;
use crate::split_info::{split_gt, SplitInfo};

/// The histogram-build strategy (`force_row_wise` / `force_col_wise`, TRL-09).
///
/// In C++ this selects between row-major and column-major histogram accumulation
/// in `GetShareStates` (`serial_tree_learner.cpp:81-112`). The two differ ONLY in
/// the ORDER bins are accumulated, NOT the result (Pitfall 5): on the
/// single-thread deterministic anchor the Phase-4 `construct_histograms`
/// whole-kernel op produces the SAME f64 cells for either strategy, so this is a
/// config FLAG over the shared Backend path (RESEARCH A1 / Open Q2 — verified
/// empirically by the `learner_parity_row_vs_col` golden, which asserts both
/// strategies grow a tree bit-identical to each other and to C++).
///
/// If a future backend's column-major accumulation ever diverged from the
/// row-major cells at the `construct_histograms` layer, that would be a Phase-4
/// boundary re-open (threat T-05-04-02) — the learner does not silently ship a
/// divergent tree; the row==col equality gate would fail loudly.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BuildStrategy {
    /// `force_row_wise=true` — the Plan-03 spine default.
    #[default]
    RowWise,
    /// `force_col_wise=true` — observationally identical on the anchor (A1).
    ColWise,
}

/// C++ `kMinScore = -inf` (`meta.h:50`) — the "never chosen" gain sentinel used
/// for the max_depth / both-children-too-small gates.
const K_MIN_SCORE: f64 = f64::NEG_INFINITY;

/// One feature column's spine input: the binned column over ALL rows plus the
/// bin-layout descriptors the Backend ops + `Tree::split` need.
///
/// This is the spine's faithful slice of the Phase-2 `Dataset` / `FeatureGroup` /
/// `BinMapper` surface — the learner does NOT re-bin (RESEARCH Don't-Hand-Roll);
/// the caller (or the capture harness) supplies these directly.
#[derive(Debug, Clone)]
pub struct FeatureColumn {
    /// Per-GLOBAL-ROW bin index (the `u32`-widened `Bin::data(row)`), length
    /// `num_data`. The histogram/partition ops index this by row id.
    pub bins: Vec<u32>,
    /// C++ `num_bin` — this feature's bin count (the histogram has `2*num_bin`
    /// cells).
    pub num_bin: u32,
    /// The threshold-offset arithmetic descriptor (`meta_->offset`). This MUST be
    /// derived from [`crate::offset_for_most_freq_bin`] — the single authoritative
    /// rule (`most_freq_bin == 0 -> 1`, else `0`) that supersedes D-01 per D-09. Do
    /// NOT inline the rule here; the helper is the sole source. It drives the
    /// FORWARD/REVERSE compacted-scan range (`num_bin - offset` cells) and the
    /// threshold recording (`t + offset` / `t - 1 + offset`).
    pub offset: i32,
    /// C++ `min_bin` — the feature's first bin in its FeatureGroup (the
    /// `data_partition` `USE_MIN_BIN` lower bound). For a single-feature group
    /// this is `most_freq_bin == 0 ? 1 : 0`'s group-relative analog; the spine
    /// passes the feature's own `[min_bin, max_bin]`.
    pub min_bin: u32,
    /// C++ `max_bin` — the feature's last bin (the partition upper bound).
    pub max_bin: u32,
    /// C++ `default_bin_` (`ValueToBin(0)`) — drives the SKIP_DEFAULT_BIN continue.
    pub default_bin: u32,
    /// C++ `most_freq_bin_` — drives `FixHistogram` + the partition default
    /// direction.
    pub most_freq_bin: u32,
    /// C++ `missing_type_` — derives the authoritative `skip_default_bin` /
    /// `na_as_missing` dispatch flags (`num_bin > 2 && missing_type == Zero/NaN`).
    pub missing_type: MissingType,
    /// Real-value bin upper bounds (`bin_upper_bound_`) — the per-bin threshold
    /// the tree stores as `threshold` (real f64). `threshold[bin]` is the value
    /// for a split AT `bin` (i.e. `bin_upper_bound_[bin]`).
    pub bin_upper_bound: Vec<f64>,
    /// The ORIGINAL feature index (`real_feature_idx_`) the tree's `split_feature`
    /// records and predict traverses.
    pub real_feature_index: i32,
}

impl FeatureColumn {
    /// The authoritative C++ `SKIP_DEFAULT_BIN` flag
    /// (`feature_histogram.hpp:284`): `num_bin > 2 && missing_type == Zero`.
    fn skip_default_bin(&self) -> bool {
        self.num_bin > 2 && self.missing_type == MissingType::Zero
    }

    /// The authoritative C++ `NA_AS_MISSING` flag
    /// (`feature_histogram.hpp:285`): `num_bin > 2 && missing_type == NaN`.
    fn na_as_missing(&self) -> bool {
        self.num_bin > 2 && self.missing_type == MissingType::NaN
    }

    /// The authoritative C++ FORWARD-branch dispatch flag
    /// (`feature_histogram.hpp:420-429`). LightGBM dispatches BOTH the REVERSE and
    /// FORWARD `FindBestThresholdSequentially` ONLY for
    /// `num_bin > 2 && missing_type == Zero`; for `missing_type == None` (and
    /// `num_bin <= 2`) it runs the REVERSE branch ONLY, so `FindBestThreshold:170`'s
    /// pre-set `default_left = true` survives → `decision_type == 2`. This is a
    /// verbatim transcription of that truth table (it equals `skip_default_bin()`
    /// here; the deferred NaN case is a typed error before this is reached).
    fn run_forward(&self) -> bool {
        self.num_bin > 2 && self.missing_type == MissingType::Zero
    }

    /// The real-value threshold a split at `threshold_bin` records on the tree
    /// (`bin_upper_bound_[threshold_bin]`). Falls back to the bin index as f64 if
    /// the upper-bound table is short (spine cases always supply it).
    fn real_threshold(&self, threshold_bin: u32) -> f64 {
        self.bin_upper_bound
            .get(threshold_bin as usize)
            .copied()
            .unwrap_or(threshold_bin as f64)
    }
}

/// The serial tree learner (`SerialTreeLearner`). Holds the per-tree growth state:
/// the data partition, the histogram pool, the per-leaf best split, and the
/// smaller/larger leaf-split sums.
pub struct SerialTreeLearner<'b, B: Backend> {
    backend: &'b B,
    client: &'b ComputeClient<B::Runtime>,
    cfg: GainConfig,
    num_leaves: i32,
    max_depth: i32,
    /// The spine's per-feature columns (set via [`with_features`](Self::with_features)).
    features: Vec<FeatureColumn>,
    /// `force_row_wise` / `force_col_wise` (TRL-09). Default `RowWise` (spine).
    strategy: BuildStrategy,
    /// Feature-subsampling config (TRL-08): `(feature_fraction,
    /// feature_fraction_bynode, feature_fraction_seed)`. `None` ⇒ the spine path
    /// (`feature_fraction == feature_fraction_bynode == 1.0`, no subsampling, the
    /// `ColSampler` is never built so the Plan-03 behavior is bit-identical).
    col_sampling: Option<(f64, f64, i32)>,
    /// TEST-ONLY audit hook (T-05-07-01). When `Some`, every `use_subtract` larger
    /// child in the live growth path records `(derived, direct)`: the histogram the
    /// wired `subtract_histograms(parent, smaller)` produced vs an independent
    /// direct build of the same leaf's rows. A parity test drains this to assert the
    /// subtracted larger child equals the direct build cell-for-cell (TRL-02), in
    /// the ACTUAL growth path (not just in isolation). `None` in production (no-op,
    /// zero overhead beyond the `Option` check).
    subtract_audit: Option<std::cell::RefCell<Vec<(Vec<f64>, Vec<f64>)>>>,
}

/// The per-split snapshot the spine emits for the D-06 golden: for each candidate
/// feature at one split decision, the full per-bin gain arrays + the chosen
/// winner. (Returned by [`SerialTreeLearner::train_with_snapshots`].)
#[derive(Debug, Clone)]
pub struct SplitSnapshot {
    /// The leaf being split at this decision.
    pub leaf: i32,
    /// Per-feature records: `(real_feature_index, reverse_gains, forward_gains,
    /// winning SplitInfo, winning_feature)`.
    pub per_feature: Vec<FeatureSplitRecord>,
    /// The cross-feature winning feature's ORIGINAL index (or -1 if no split).
    pub winner_feature: i32,
}

/// One feature's per-bin gain scan at a split decision (D-06).
#[derive(Debug, Clone)]
pub struct FeatureSplitRecord {
    /// ORIGINAL feature index.
    pub feature: i32,
    /// Per-candidate REVERSE-branch gains (NaN where gated).
    pub cand_rev: Vec<f64>,
    /// Per-candidate FORWARD-branch gains (NaN where gated).
    pub cand_fwd: Vec<f64>,
    /// The feature's best `SplitInfo`.
    pub split: SplitInfo,
}

/// The `ColSampler` selection trace for one tree (TRL-08 RNG call-sequence
/// golden). Records the per-tree `ResetByTree` selection and every per-node
/// `GetByNode` selection IN DRAW ORDER (smaller-leaf then larger-leaf per split),
/// so the parity golden can assert the exact selected feature indices.
#[derive(Debug, Clone, Default)]
pub struct ColSamplerTrace {
    /// `is_feature_used_bytree()` after `ResetByTree` — the selected REAL feature
    /// indices for this tree (ascending).
    pub bytree_selected: Vec<i32>,
    /// Per-node `GetByNode` selections in DRAW ORDER. Each entry is the selected
    /// REAL feature indices (ascending) for one `get_by_node` call — the order is
    /// smaller-leaf then larger-leaf within each split decision
    /// (serial_tree_learner.cpp:479,487).
    pub bynode_selected: Vec<Vec<i32>>,
}

impl<'b, B: Backend> SerialTreeLearner<'b, B> {
    /// Construct the learner over a `Backend` + its client and the gain/cap config.
    ///
    /// `num_leaves` / `max_depth` are the leaf-wise caps (`config_->num_leaves`,
    /// `config_->max_depth`; `max_depth <= 0` means "no depth cap").
    pub fn new(
        backend: &'b B,
        client: &'b ComputeClient<B::Runtime>,
        cfg: GainConfig,
        num_leaves: i32,
        max_depth: i32,
    ) -> Self {
        Self {
            backend,
            client,
            cfg,
            num_leaves,
            max_depth,
            features: Vec::new(),
            strategy: BuildStrategy::RowWise,
            col_sampling: None,
            subtract_audit: None,
        }
    }

    /// Grow one tree from fixed `gradients`/`hessians` (`SerialTreeLearner::Train`).
    ///
    /// `features` are the spine's per-feature columns (all used, `feature_fraction
    /// = 1.0`); `is_first_tree` is carried for API parity (unused on the spine —
    /// there is no histogram-cache reuse across trees here).
    ///
    /// # Errors
    /// V5 boundary (`TreeLearnerError`, never a panic): `gradients.len() ==
    /// hessians.len() == num_data`, `num_leaves >= 1`, every feature's bins
    /// `< num_bin`, root `sum_hessian > 0` (the `cnt_factor` guard), and any
    /// `na_as_missing` feature (deferred). Backend failures wrap via `#[from]`.
    pub fn train(
        &mut self,
        gradients: &[f32],
        hessians: &[f32],
        is_first_tree: bool,
    ) -> Result<Tree, TreeLearnerError> {
        Ok(self.train_with_snapshots(gradients, hessians, is_first_tree)?.0)
    }

    /// Like [`train`](Self::train) but also returns the per-split D-06 snapshots
    /// (full per-bin gain arrays per candidate feature at every split) for the
    /// golden replay.
    #[allow(clippy::type_complexity)]
    pub fn train_with_snapshots(
        &mut self,
        gradients: &[f32],
        hessians: &[f32],
        is_first_tree: bool,
    ) -> Result<(Tree, Vec<SplitSnapshot>), TreeLearnerError> {
        let (tree, snaps, _trace, _part) = self.train_inner(gradients, hessians, is_first_tree)?;
        Ok((tree, snaps))
    }

    /// Like [`train`](Self::train) but ALSO returns the final [`DataPartition`]
    /// the tree was grown over (the row→leaf mapping after the last split).
    ///
    /// The GBDT loop (06-02) needs this for the bit-exact training-path score
    /// scatter [`add_prediction_to_score`](Self::add_prediction_to_score): the
    /// C++ `data_partition_` is a learner member, but this port builds the
    /// partition locally inside `train_inner` and does not retain it on `self`, so
    /// the boosting caller takes ownership of it here and passes it back to the
    /// scatter. The returned partition's per-leaf row sets are exactly the C++
    /// `data_partition_->indices/leaf_begin/leaf_count` used by
    /// `AddPredictionToScore`.
    pub fn train_returning_partition(
        &mut self,
        gradients: &[f32],
        hessians: &[f32],
        is_first_tree: bool,
    ) -> Result<(Tree, DataPartition), TreeLearnerError> {
        let (tree, _snaps, _trace, part) =
            self.train_inner(gradients, hessians, is_first_tree)?;
        Ok((tree, part))
    }

    /// Like [`train_with_snapshots`](Self::train_with_snapshots) but also returns
    /// the [`ColSamplerTrace`] — the per-tree + per-node feature-subsampling
    /// selections in DRAW ORDER (TRL-08 RNG call-sequence golden). On the spine
    /// (`feature_fraction == feature_fraction_bynode == 1.0`) the trace records
    /// every feature as selected and no per-node draws (the sampler is inactive).
    #[allow(clippy::type_complexity)]
    pub fn train_with_col_sampler_trace(
        &mut self,
        gradients: &[f32],
        hessians: &[f32],
        is_first_tree: bool,
    ) -> Result<(Tree, Vec<SplitSnapshot>, ColSamplerTrace), TreeLearnerError> {
        let (tree, snaps, trace, _part) =
            self.train_inner(gradients, hessians, is_first_tree)?;
        Ok((tree, snaps, trace))
    }

    /// The shared growth driver behind [`train`](Self::train),
    /// [`train_with_snapshots`](Self::train_with_snapshots), and
    /// [`train_with_col_sampler_trace`](Self::train_with_col_sampler_trace).
    #[allow(clippy::type_complexity)]
    fn train_inner(
        &mut self,
        gradients: &[f32],
        hessians: &[f32],
        _is_first_tree: bool,
    ) -> Result<(Tree, Vec<SplitSnapshot>, ColSamplerTrace, DataPartition), TreeLearnerError> {
        let num_data = gradients.len() as i32;
        let features = self.features.clone();

        // force_row_wise / force_col_wise (TRL-09, A1 / Open Q2): the two strategies
        // differ ONLY in the histogram-build ORDER, not the result. On the
        // single-thread deterministic anchor the Phase-4 `construct_histograms`
        // whole-kernel op produces the SAME f64 cells for either, so `self.strategy`
        // does NOT route a distinct compute path here — both `RowWise` and `ColWise`
        // drive the identical `backend.construct_histograms` call below and therefore
        // grow a bit-identical tree (asserted by `learner_parity_row_vs_col`). If a
        // backend's column-major accumulation ever diverged at that layer, the
        // row==col equality gate would fail loudly (threat T-05-04-02) — we never
        // silently ship a divergent tree. `_strategy` is read here to make the
        // (verified) no-op explicit rather than dead.
        let _strategy: BuildStrategy = self.strategy;

        // ---- V5 boundary validation (T-05-03-01/02/03) ----
        if hessians.len() != gradients.len() {
            return Err(TreeLearnerError::LengthMismatch {
                expected: gradients.len(),
                actual: hessians.len(),
            });
        }
        if self.num_leaves < 1 {
            return Err(TreeLearnerError::InvalidNumLeaves {
                value: self.num_leaves,
            });
        }
        for f in &features {
            if f.bins.len() != num_data as usize {
                return Err(TreeLearnerError::LengthMismatch {
                    expected: num_data as usize,
                    actual: f.bins.len(),
                });
            }
            for &b in &f.bins {
                if b >= f.num_bin {
                    return Err(TreeLearnerError::BinIndexOutOfRange {
                        index: b,
                        num_bin: f.num_bin,
                    });
                }
            }
            if f.na_as_missing() {
                // NA_AS_MISSING forward branch deferred (RESEARCH A5) — surface the
                // compute layer's typed error rather than silently mis-routing.
                return Err(TreeLearnerError::Compute(ComputeError::Runtime {
                    detail: "train: na_as_missing feature (num_bin>2 && missing_type==NaN) \
                             is deferred (NA_AS_MISSING forward branch not implemented)"
                        .to_string(),
                }));
            }
        }

        // ---- BeforeTrain (serial_tree_learner.cpp:205-208, 288-...) ----
        // ColSampler.ResetByTree (col_sampler.hpp:74-89) happens HERE, once per
        // tree (BeforeTrain). The spine path (`col_sampling == None`) builds no
        // sampler and gates nothing — Plan-03 behavior is bit-identical.
        let mut trace = ColSamplerTrace::default();
        let num_features = features.len();
        // valid_feature_indices_ = 0..num_features on the spine (every feature is
        // valid — no trivial-feature dropping at this layer); InnerFeatureIndex is
        // the identity.
        let valid_feature_indices: Vec<i32> = features.iter().map(|f| f.real_feature_index).collect();
        let mut col_sampler = self.col_sampling.map(|(ff, ffn, seed)| {
            // num_features here is the COUNT of feature columns; is_feature_used_ is
            // indexed by real_feature_index, so size it to cover the max index + 1.
            let max_real = valid_feature_indices.iter().copied().max().unwrap_or(-1);
            let nf = (max_real + 1).max(num_features as i32) as usize;
            ColSampler::new(ff, ffn, seed, valid_feature_indices.clone(), nf)
        });
        // Record the per-tree selection (ResetByTree result) for the TRL-08 golden.
        if let Some(cs) = col_sampler.as_ref() {
            trace.bytree_selected = valid_feature_indices
                .iter()
                .copied()
                .filter(|&f| cs.is_feature_used_bytree(f))
                .collect();
        } else {
            trace.bytree_selected = valid_feature_indices.clone();
        }

        let mut data_partition = DataPartition::new(num_data, self.num_leaves);
        // The pool slot holds EVERY feature's compacted histogram CONCATENATED
        // (mirroring C++ `histogram_array_[feature_index]`, a contiguous per-leaf
        // buffer). `slot_off[fpos]` is feature `fpos`'s start cell in the slot; the
        // total slot length is `Σ 2*num_bin`. The whole concatenated buffer is the
        // subtraction-trick unit: `larger = parent − smaller` is a single
        // element-wise subtract over it (each feature's compacted region subtracts
        // independently; the zeroed compaction tails subtract to zero — D-05/A3).
        let (slot_off, slot_len) = feature_slot_layout(&features);
        let mut pool = HistogramPool::new(self.num_leaves, slot_len);
        pool.reset_map();

        // Root leaf sums via the ordered f64 fold over ALL rows.
        let root_indices: Vec<u32> = (0..num_data as u32).collect();
        let mut smaller_leaf_splits = LeafSplits::new();
        smaller_leaf_splits.init(gradients, hessians, &root_indices, &self.cfg);
        let mut larger_leaf_splits = LeafSplits::new();

        // Guard the root cnt_factor = num_data / sum_hessian division.
        #[allow(clippy::neg_cmp_op_on_partial_ord)]
        if !(smaller_leaf_splits.sum_hessians > 0.0) {
            return Err(TreeLearnerError::InvalidLeafHessian {
                value: smaller_leaf_splits.sum_hessians,
            });
        }

        // Seed the root leaf output (serial_tree_learner.cpp:205-208).
        let root_output = calculate_splitted_leaf_output(
            self.cfg.use_l1(),
            smaller_leaf_splits.sum_gradients,
            smaller_leaf_splits.sum_hessians,
            self.cfg.lambda_l1,
            self.cfg.lambda_l2,
        );
        let mut tree = root_tree(root_output, num_data);

        // best_split_per_leaf_ (flat Vec + ArgMax; NEVER a priority-queue/heap).
        let mut best_split_per_leaf: Vec<SplitInfo> =
            vec![SplitInfo::none(); self.num_leaves as usize];
        let mut best_split_feature: Vec<i32> = vec![-1; self.num_leaves as usize];

        let mut snapshots: Vec<SplitSnapshot> = Vec::new();

        // left_leaf / right_leaf track the children produced by the last Split.
        // The first iteration's "left_leaf" is the root (leaf 0); right_leaf = -1.
        let mut left_leaf: i32 = 0;
        let mut right_leaf: i32 = -1;

        // ---- the leaf-wise loop (serial_tree_learner.cpp:218-236) ----
        for _split in 0..(self.num_leaves - 1) {
            // BeforeFindBestSplit (gates + smaller-child selection).
            let did = self.before_find_best_split(
                &tree,
                &data_partition,
                left_leaf,
                right_leaf,
                &mut best_split_per_leaf,
            );
            if did {
                // FindBestSplits: build smaller histogram (and larger via subtract),
                // then per-feature find_best_split + cross-feature argmax. The
                // per-node ColSampler draws (smaller-then-larger) happen INSIDE
                // find_best_splits in the exact C++ order (T-05-04-01).
                let snap = self.find_best_splits(
                    &features,
                    gradients,
                    hessians,
                    &data_partition,
                    &mut pool,
                    &slot_off,
                    &smaller_leaf_splits,
                    &larger_leaf_splits,
                    left_leaf,
                    right_leaf,
                    &mut best_split_per_leaf,
                    &mut best_split_feature,
                    col_sampler.as_mut(),
                    &mut trace,
                )?;
                snapshots.push(snap);
            }

            // best_leaf = ArgMax(best_split_per_leaf_) (first-max via split_gt).
            let best_leaf = arg_max(&best_split_per_leaf, &best_split_feature);
            let best = best_split_per_leaf[best_leaf as usize];
            if best.gain <= 0.0 {
                break; // no positive-gain split (serial_tree_learner.cpp:225-227)
            }

            // SplitInner: partition + grow the node + seed child leaf-splits.
            let feat_idx = best_split_feature[best_leaf as usize];
            let (new_left, new_right) = self.split_inner(
                &mut tree,
                &mut data_partition,
                &features,
                best_leaf,
                feat_idx,
                &best,
                &mut smaller_leaf_splits,
                &mut larger_leaf_splits,
            )?;

            // Reset the just-split leaf's best so it is not re-selected, and the
            // new leaves to "no split" until BeforeFindBestSplit recomputes them.
            best_split_per_leaf[best_leaf as usize] = SplitInfo::none();
            best_split_feature[best_leaf as usize] = -1;
            best_split_per_leaf[new_left as usize] = SplitInfo::none();
            best_split_feature[new_left as usize] = -1;
            best_split_per_leaf[new_right as usize] = SplitInfo::none();
            best_split_feature[new_right as usize] = -1;

            left_leaf = new_left;
            right_leaf = new_right;
        }

        Ok((tree, snapshots, trace, data_partition))
    }

    /// `BeforeFindBestSplit` gates (`serial_tree_learner.cpp:343-378`): apply the
    /// `max_depth` cap and the both-children-too-small cap, then select the smaller
    /// child (drives the subtraction trick). Returns `false` when a gate forces the
    /// leaf's gain to `kMinScore` (skip the scan), `true` otherwise.
    ///
    /// (Smaller-child selection state is computed here but the actual pool Get/Move
    /// + histogram build happens in `find_best_splits`, keeping the spine's data
    /// flow linear.)
    fn before_find_best_split(
        &self,
        tree: &Tree,
        data_partition: &DataPartition,
        left_leaf: i32,
        right_leaf: i32,
        best_split_per_leaf: &mut [SplitInfo],
    ) -> bool {
        let num_left = data_partition.leaf_count(left_leaf);
        // max_depth gate (:343-352): the splitting leaf is too deep.
        if self.max_depth > 0 && tree.leaf_depth[left_leaf as usize] >= self.max_depth {
            best_split_per_leaf[left_leaf as usize].gain = K_MIN_SCORE;
            if right_leaf >= 0 {
                best_split_per_leaf[right_leaf as usize].gain = K_MIN_SCORE;
            }
            return false;
        }
        // both-children-too-small gate (:353-363).
        let min2 = self.cfg.min_data_in_leaf * 2;
        if right_leaf >= 0 {
            let num_right = data_partition.leaf_count(right_leaf);
            if num_right < min2 && num_left < min2 {
                best_split_per_leaf[left_leaf as usize].gain = K_MIN_SCORE;
                best_split_per_leaf[right_leaf as usize].gain = K_MIN_SCORE;
                return false;
            }
        } else if num_left < min2 {
            best_split_per_leaf[left_leaf as usize].gain = K_MIN_SCORE;
            return false;
        }
        true
    }

    /// `FindBestSplits` + `FindBestSplitsFromHistograms`
    /// (`serial_tree_learner.cpp:404-618`): build the smaller leaf's histogram
    /// (and the larger leaf's via subtraction when `use_subtract`), then for each
    /// feature `FixHistogram` (raw sums) → `find_best_split` → cross-feature argmax.
    ///
    /// SMALLER-CHILD SELECTION (Pitfall 3): when `right_leaf >= 0`,
    /// `num_data_in_left < num_data_in_right` ⇒ smaller = left, else smaller = right.
    /// The larger child's histogram is `parent − smaller`. On the root
    /// (`right_leaf < 0`) only the single leaf is built (`use_subtract = false`).
    #[allow(clippy::too_many_arguments)]
    fn find_best_splits(
        &self,
        features: &[FeatureColumn],
        gradients: &[f32],
        hessians: &[f32],
        data_partition: &DataPartition,
        pool: &mut HistogramPool,
        slot_off: &[usize],
        smaller_leaf_splits: &LeafSplits,
        larger_leaf_splits: &LeafSplits,
        left_leaf: i32,
        right_leaf: i32,
        best_split_per_leaf: &mut [SplitInfo],
        best_split_feature: &mut [i32],
        col_sampler: Option<&mut ColSampler>,
        trace: &mut ColSamplerTrace,
    ) -> Result<SplitSnapshot, TreeLearnerError> {
        // ---- HistogramPool slot dance (serial_tree_learner.cpp:364-378) ----
        // Decide smaller/larger by the ACTUAL data-partition row counts, and assign
        // pool slots EXACTLY as C++ `BeforeFindBestSplit` does:
        //   - root (right_leaf < 0): Get(left) → smaller slot; no larger; no parent.
        //   - num_left < num_right (smaller == left): the PARENT histogram lives in
        //     `left_leaf`'s slot (a cache HIT → retain as `parent`); Move(left→right)
        //     hands that buffer to the LARGER child (right), then Get(left) gives the
        //     SMALLER child (left) a fresh slot.
        //   - else (smaller == right): retain `left_leaf`'s slot as the LARGER child's
        //     (parent), and Get(right) gives the SMALLER child a fresh slot.
        // `use_subtract` is true iff a parent histogram was retained (cache hit) —
        // i.e. the larger child can be derived by `parent − smaller` instead of a
        // second direct build. On the root the parent slot is empty so use_subtract
        // is false and the single leaf is built directly.
        let (smaller_leaf, larger_leaf) = if right_leaf < 0 {
            (left_leaf, -1)
        } else {
            let num_left = data_partition.leaf_count(left_leaf);
            let num_right = data_partition.leaf_count(right_leaf);
            if num_left < num_right {
                (left_leaf, right_leaf) // smaller = left
            } else {
                (right_leaf, left_leaf) // smaller = right
            }
        };

        // Run the C++ pool Get/Move sequence and learn which slots back the smaller
        // child, the larger child, and (when a hit) the retained parent histogram.
        let (smaller_slot, larger_slot, parent_slot) = if right_leaf < 0 {
            // Only the root leaf.
            let (s, _existed) = pool.get(left_leaf);
            (s, None, None)
        } else if smaller_leaf == left_leaf {
            // smaller == left: parent lives in left's slot.
            let (parent, existed) = pool.get(left_leaf);
            let parent_opt = existed.then_some(parent);
            pool.move_(left_leaf, right_leaf); // hand parent's buffer to the larger child
            let (s, _e) = pool.get(left_leaf); // fresh slot for the smaller child
            (s, Some(parent), parent_opt)
        } else {
            // smaller == right: parent stays in left's slot (the larger child).
            let (parent, existed) = pool.get(left_leaf);
            let parent_opt = existed.then_some(parent);
            let (s, _e) = pool.get(right_leaf); // fresh slot for the smaller child
            (s, Some(parent), parent_opt)
        };
        // use_subtract == (a parent histogram was retained), C++
        // `parent_leaf_histogram_array_ != nullptr` (serial_tree_learner.cpp:398).
        let use_subtract = parent_slot.is_some();

        // ColSampler.GetByNode draw ORDER (serial_tree_learner.cpp:479,487): the
        // SMALLER leaf is drawn FIRST (always), the LARGER leaf SECOND (only when a
        // larger child exists, i.e. not the root). Each call advances the shared
        // PRNG once. The raw per-node selection is recorded in the trace (the
        // TRL-08 golden asserts these), then combined with the per-tree
        // is_feature_used_bytree flag to form the effective scan mask. On the spine
        // (`col_sampler == None`) no draw happens and every feature is used.
        let has_larger = use_subtract && larger_leaf >= 0;
        let (smaller_node_mask, larger_node_mask) = if let Some(cs) = col_sampler {
            let smaller_raw = cs.get_by_node();
            trace
                .bynode_selected
                .push(mask_to_indices(&smaller_raw, features));
            let larger_raw = if has_larger {
                let l = cs.get_by_node();
                trace.bynode_selected.push(mask_to_indices(&l, features));
                Some(l)
            } else {
                None
            };
            // Combine each per-node mask with the per-tree bytree flag:
            // is_feature_used[f] = is_feature_used_bytree(f) && node_selected(f).
            let combine = |node: &[i8]| -> Vec<i8> {
                features
                    .iter()
                    .map(|f| {
                        let fi = f.real_feature_index;
                        let used = cs.is_feature_used_bytree(fi)
                            && node.get(fi as usize).copied().unwrap_or(0) != 0;
                        i8::from(used)
                    })
                    .collect()
            };
            let smaller_eff = combine(&smaller_raw);
            let larger_eff = larger_raw.as_deref().map(combine);
            (Some(smaller_eff), larger_eff)
        } else {
            (None, None)
        };

        // The smaller leaf's seeded sums vs the larger leaf's. `split_inner` seeds
        // `smaller_leaf_splits` with the SMALLER-by-partition-count child and
        // `larger_leaf_splits` with the larger (tie ⇒ right is "smaller"), using the
        // SAME `data_partition.leaf_count` comparison that picks `smaller_leaf` /
        // `larger_leaf` above. Therefore `smaller_leaf_splits` ALWAYS holds
        // `smaller_leaf`'s sums and `larger_leaf_splits` holds `larger_leaf`'s — the
        // mapping is a DIRECT pass-through, mirroring C++ `smaller_leaf_splits_` /
        // `larger_leaf_splits_` carrying their own `leaf_index_`
        // (serial_tree_learner.cpp:851). The previous `smaller_leaf == left_leaf`
        // branch SWAPPED the slots whenever the smaller child was the right leaf
        // (incl. the equal-count tie), feeding `smaller_leaf` the larger sibling's
        // sums (CR-03 root cause: leaf 1 got leaf 0's −24 sum → wrong child splits /
        // `leaf_value` like −17.99).
        let (smaller_splits, larger_splits) = if right_leaf < 0 {
            (smaller_leaf_splits, smaller_leaf_splits)
        } else {
            (smaller_leaf_splits, larger_leaf_splits)
        };

        // ---- SMALLER child: build its concatenated histogram directly into its
        // pool slot (construct → FixHistogram raw sums → compact, per feature),
        // then scan it (serial_tree_learner.cpp:530-543). ----
        self.build_leaf_histogram_into(
            features,
            gradients,
            hessians,
            data_partition,
            slot_off,
            smaller_leaf,
            smaller_splits,
            pool.buffer_mut(smaller_slot),
        );
        let smaller_records = self.scan_leaf_histogram(
            features,
            slot_off,
            smaller_leaf,
            smaller_splits,
            pool.buffer(smaller_slot),
            best_split_per_leaf,
            best_split_feature,
            smaller_node_mask.as_deref(),
        )?;

        // ---- LARGER child: derive by subtraction (parent − smaller) in the pool,
        // OR build directly when no parent was retained. ----
        let mut larger_records: Vec<FeatureSplitRecord> = Vec::new();
        if larger_leaf >= 0 {
            let larger_slot = larger_slot.expect("non-root larger child must hold a pool slot");
            if let Some(parent_slot) = parent_slot {
                // use_subtract: larger = parent − smaller over the WHOLE concatenated
                // compacted buffer (FeatureHistogram::Subtract, feature_histogram.hpp:
                // 140-144 — `larger_data_[i] -= smaller_data_[i]` per compacted cell).
                // The smaller buffer here is already FixHistogram'd+compacted, and the
                // retained parent buffer is the parent leaf's FixHistogram'd+compacted
                // histogram; the derived larger is NOT re-FixHistogram'd (C++ runs no
                // FixHistogram on the use_subtract larger child, only ComputeBestSplit).
                // This is the kEpsilon-faithful derivation the direct rebuild cannot
                // reproduce bit-for-bit.
                debug_assert_eq!(larger_slot, parent_slot, "the larger child reuses the moved parent slot");
                let parent_buf = pool.buffer(parent_slot).to_vec();
                let smaller_buf = pool.buffer(smaller_slot).to_vec();
                let derived = self
                    .backend
                    .subtract_histograms(self.client, &parent_buf, &smaller_buf)?;
                // TEST audit hook (T-05-07-01): record (derived, direct) so a parity
                // test can assert the subtracted larger child == a direct build of its
                // own rows, cell-for-cell, in the LIVE growth path.
                if let Some(audit) = self.subtract_audit.as_ref() {
                    let mut direct = vec![0.0f64; derived.len()];
                    self.build_leaf_histogram_into(
                        features,
                        gradients,
                        hessians,
                        data_partition,
                        slot_off,
                        larger_leaf,
                        larger_splits,
                        &mut direct,
                    );
                    audit.borrow_mut().push((derived.clone(), direct));
                }
                pool.buffer_mut(larger_slot).copy_from_slice(&derived);
            } else {
                // No parent retained (cannot happen post-root in the current spine,
                // but kept faithful to the C++ `else` direct-build branch): build the
                // larger child directly into its slot.
                self.build_leaf_histogram_into(
                    features,
                    gradients,
                    hessians,
                    data_partition,
                    slot_off,
                    larger_leaf,
                    larger_splits,
                    pool.buffer_mut(larger_slot),
                );
            }
            larger_records = self.scan_leaf_histogram(
                features,
                slot_off,
                larger_leaf,
                larger_splits,
                pool.buffer(larger_slot),
                best_split_per_leaf,
                best_split_feature,
                larger_node_mask.as_deref(),
            )?;
        }
        let _ = use_subtract; // recorded for clarity; the parent_slot Option drives it

        // The snapshot records the SMALLER leaf's per-feature scan (the directly-
        // built one — the D-06 localizer); the winner is the smaller leaf's best.
        let winner_feature = best_split_feature[smaller_leaf as usize];
        let mut per_feature = smaller_records;
        per_feature.extend(larger_records);
        Ok(SplitSnapshot {
            leaf: smaller_leaf,
            per_feature,
            winner_feature,
        })
    }

    /// Build one leaf's per-feature CONCATENATED histogram DIRECTLY (construct over
    /// the leaf's rows → `FixHistogram` on the RAW leaf sums → compact) into the
    /// caller's pool-slot buffer `buf`. Each feature `fpos`'s compacted histogram
    /// occupies `buf[slot_off[fpos] .. slot_off[fpos] + 2*num_bin]`; unused trailing
    /// cells (from compaction) are zeroed. This is the C++ `ConstructHistograms`
    /// directly-built smaller leaf (serial_tree_learner.cpp:452-459), with
    /// `FixHistogram` (`:531-534`) folded in so the stored slot is the
    /// FixHistogram'd+compacted form the subtraction trick consumes.
    #[allow(clippy::too_many_arguments)]
    fn build_leaf_histogram_into(
        &self,
        features: &[FeatureColumn],
        gradients: &[f32],
        hessians: &[f32],
        data_partition: &DataPartition,
        slot_off: &[usize],
        leaf: i32,
        leaf_splits: &LeafSplits,
        buf: &mut [f64],
    ) {
        let sum_g = leaf_splits.sum_gradients;
        let sum_h = leaf_splits.sum_hessians;
        // Empty / no-hessian leaf: leave the slot zeroed (it will not be scanned).
        #[allow(clippy::neg_cmp_op_on_partial_ord)]
        let buildable = sum_h > 0.0 && leaf_splits.num_data_in_leaf > 0;
        for c in buf.iter_mut() {
            *c = 0.0;
        }
        if !buildable {
            return;
        }
        let leaf_rows = data_partition.indices_in_leaf(leaf);
        for (fpos, f) in features.iter().enumerate() {
            let cells = 2 * f.num_bin as usize;
            let region = &mut buf[slot_off[fpos]..slot_off[fpos] + cells];
            // ORDERED per-feature gradient/hessian for this leaf's rows (the C++
            // ordered fold — never reordered/parallelized).
            let mut ord_bins: Vec<u32> = Vec::with_capacity(leaf_rows.len());
            let mut ord_g: Vec<f32> = Vec::with_capacity(leaf_rows.len());
            let mut ord_h: Vec<f32> = Vec::with_capacity(leaf_rows.len());
            for &row in leaf_rows {
                ord_bins.push(f.bins[row as usize]);
                ord_g.push(gradients[row as usize]);
                ord_h.push(hessians[row as usize]);
            }
            let mut hist = self
                .backend
                .construct_histograms(self.client, &ord_bins, &ord_g, &ord_h, f.num_bin)
                .expect("construct_histograms on a validated leaf cannot fail");
            // FixHistogram on the RAW leaf sums (Pitfall 2). No-op for offset==1
            // (most_freq_bin==0), exactly as C++ `if (most_freq_bin > 0)`.
            crate::fix_histogram::fix_histogram(&mut hist, f.most_freq_bin, sum_g, sum_h);
            // COMPACTED layout (D-09): shift real-bin `c+offset` into cell `c`,
            // zero the dropped tail. No-op for offset==0.
            compact_histogram(&mut hist, f.offset);
            region.copy_from_slice(&hist);
        }
    }

    /// Scan one leaf's per-feature CONCATENATED compacted+fixed histogram (already
    /// in the pool slot `buf`, built directly OR derived via subtraction), running
    /// `find_best_split` per feature and recording the cross-feature argmax into
    /// `best_split_per_leaf[leaf]` (C++ `FindBestSplitsFromHistograms` per-feature
    /// `ComputeBestSplitForFeature`). Returns the per-feature D-06 records.
    #[allow(clippy::too_many_arguments)]
    fn scan_leaf_histogram(
        &self,
        features: &[FeatureColumn],
        slot_off: &[usize],
        leaf: i32,
        leaf_splits: &LeafSplits,
        buf: &[f64],
        best_split_per_leaf: &mut [SplitInfo],
        best_split_feature: &mut [i32],
        used_features: Option<&[i8]>,
    ) -> Result<Vec<FeatureSplitRecord>, TreeLearnerError> {
        let sum_g = leaf_splits.sum_gradients;
        let sum_h = leaf_splits.sum_hessians;
        let num_data_in_leaf = leaf_splits.num_data_in_leaf;

        let mut records: Vec<FeatureSplitRecord> = Vec::with_capacity(features.len());
        let mut leaf_best = SplitInfo::none();
        let mut leaf_best_feature: i32 = -1;

        // A leaf with no admissible hessian mass cannot be split (the `cnt_factor =
        // num_data / sum_hessian` division would divide by ~0). Leave its best as
        // `none()` rather than calling `find_best_split` (which would reject
        // sum_hessian<=0 as a typed error).
        #[allow(clippy::neg_cmp_op_on_partial_ord)]
        if !(sum_h > 0.0) || num_data_in_leaf <= 0 {
            best_split_per_leaf[leaf as usize] = SplitInfo::none();
            best_split_feature[leaf as usize] = -1;
            return Ok(records);
        }

        for (fpos, f) in features.iter().enumerate() {
            // ColSampler gate (serial_tree_learner.cpp: `if (!is_feature_used[fi])
            // continue;`). On the spine (`used_features == None`) every feature is
            // scanned.
            if let Some(mask) = used_features {
                if mask.get(fpos).copied().unwrap_or(1) == 0 {
                    continue;
                }
            }
            let cells = 2 * f.num_bin as usize;
            let hist = &buf[slot_off[fpos]..slot_off[fpos] + cells];

            // Authoritative dispatch flags (Pitfall 1). `run_forward` transcribes
            // the C++ per-missing_type branch dispatch (feature_histogram.hpp:
            // 420-429): the FORWARD scan runs ONLY for `num_bin>2 &&
            // missing_type==Zero`; for `missing_type==None` only REVERSE runs.
            let skip_default_bin = f.skip_default_bin();
            let na_as_missing = f.na_as_missing();
            let run_forward = f.run_forward();

            let split = self.backend.find_best_split(
                self.client,
                hist,
                &self.cfg,
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

            // Per-bin gain arrays for the D-06 snapshot (host re-scan of the SAME
            // fixed histogram via the gain primitive — localizes a divergence).
            let (cand_rev, cand_fwd) = self.per_bin_gains(hist, f, sum_g, sum_h, num_data_in_leaf);

            records.push(FeatureSplitRecord {
                feature: f.real_feature_index,
                cand_rev,
                cand_fwd,
                split,
            });

            // Cross-feature argmax via split_gt (gain, then smaller feature).
            if split.gain > K_MIN_SCORE
                && split_gt(&split, f.real_feature_index, &leaf_best, leaf_best_feature)
            {
                leaf_best = split;
                leaf_best_feature = f.real_feature_index;
            }
        }

        best_split_per_leaf[leaf as usize] = leaf_best;
        best_split_feature[leaf as usize] = leaf_best_feature;
        Ok(records)
    }

    /// Host re-scan of the per-bin gain arrays (REVERSE + FORWARD) on a FIXED
    /// histogram, for the D-06 snapshot. Reuses `gain::get_split_gains` so the
    /// learner emitter and kernel emitter agree (D-02a). NaN marks a gated bin.
    fn per_bin_gains(
        &self,
        hist: &[f64],
        f: &FeatureColumn,
        sum_gradient: f64,
        sum_hessian: f64,
        num_data: i32,
    ) -> (Vec<f64>, Vec<f64>) {
        use lgbm_compute::gain::get_split_gains;
        let cfg = &self.cfg;
        let use_l1 = cfg.use_l1();
        let l1 = cfg.lambda_l1;
        let l2 = cfg.lambda_l2;
        let offset = f.offset;
        let num_bin = f.num_bin as i32;
        let skip = f.skip_default_bin();
        let default_bin = f.default_bin as i32;
        let qnan = f64::NAN;

        // BeforeNumerical min_gain_shift (host, same as find_best_split).
        let gain_shift = lgbm_compute::gain::get_leaf_gain(use_l1, sum_gradient, sum_hessian, l1, l2);
        let min_gain_shift = gain_shift + cfg.min_gain_to_split;
        let eps = f64::from(lgbm_core::types::K_EPSILON);
        let sum_hessian_bumped = sum_hessian + 2.0 * eps;
        let cnt_factor = f64::from(num_data) / sum_hessian_bumped;
        let round_int = |x: f64| -> i32 { (x + f64::from(0.5f32)) as i32 };
        let get_grad = |t: i32| hist[(t as usize) << 1];
        let get_hess = |t: i32| hist[((t as usize) << 1) + 1];

        // REVERSE (:854-936).
        let mut cand_rev: Vec<f64> = Vec::new();
        {
            let mut sum_right_gradient = 0.0f64;
            let mut sum_right_hessian = eps;
            let mut right_count = 0i32;
            let t_start = num_bin - 1 - offset;
            let t_end = 1 - offset;
            let mut t = t_start;
            while t >= t_end {
                if skip && (t + offset) == default_bin {
                    cand_rev.push(qnan);
                    t -= 1;
                    continue;
                }
                sum_right_gradient += get_grad(t);
                sum_right_hessian += get_hess(t);
                right_count += round_int(get_hess(t) * cnt_factor);
                if right_count < cfg.min_data_in_leaf
                    || sum_right_hessian < cfg.min_sum_hessian_in_leaf
                {
                    cand_rev.push(qnan);
                    t -= 1;
                    continue;
                }
                let left_count = num_data - right_count;
                if left_count < cfg.min_data_in_leaf {
                    cand_rev.push(qnan);
                    break;
                }
                let sum_left_hessian = sum_hessian_bumped - sum_right_hessian;
                if sum_left_hessian < cfg.min_sum_hessian_in_leaf {
                    cand_rev.push(qnan);
                    break;
                }
                let sum_left_gradient = sum_gradient - sum_right_gradient;
                let g = get_split_gains(
                    use_l1,
                    sum_left_gradient,
                    sum_left_hessian,
                    sum_right_gradient,
                    sum_right_hessian,
                    l1,
                    l2,
                );
                if g <= min_gain_shift {
                    cand_rev.push(qnan);
                } else {
                    cand_rev.push(g);
                }
                t -= 1;
            }
        }

        // FORWARD (:937-1029).
        let mut cand_fwd: Vec<f64> = Vec::new();
        {
            let mut sum_left_gradient = 0.0f64;
            let mut sum_left_hessian = eps;
            let mut left_count = 0i32;
            let t_end = num_bin - 2 - offset;
            let mut t = 0i32;
            while t <= t_end {
                if skip && (t + offset) == default_bin {
                    cand_fwd.push(qnan);
                    t += 1;
                    continue;
                }
                sum_left_gradient += get_grad(t);
                sum_left_hessian += get_hess(t);
                left_count += round_int(get_hess(t) * cnt_factor);
                if left_count < cfg.min_data_in_leaf
                    || sum_left_hessian < cfg.min_sum_hessian_in_leaf
                {
                    cand_fwd.push(qnan);
                    t += 1;
                    continue;
                }
                let right_count = num_data - left_count;
                if right_count < cfg.min_data_in_leaf {
                    cand_fwd.push(qnan);
                    break;
                }
                let sum_right_hessian = sum_hessian_bumped - sum_left_hessian;
                if sum_right_hessian < cfg.min_sum_hessian_in_leaf {
                    cand_fwd.push(qnan);
                    break;
                }
                let sum_right_gradient = sum_gradient - sum_left_gradient;
                let g = get_split_gains(
                    use_l1,
                    sum_left_gradient,
                    sum_left_hessian,
                    sum_right_gradient,
                    sum_right_hessian,
                    l1,
                    l2,
                );
                if g <= min_gain_shift {
                    cand_fwd.push(qnan);
                } else {
                    cand_fwd.push(g);
                }
                t += 1;
            }
        }

        (cand_rev, cand_fwd)
    }

    /// `SplitInner` (`serial_tree_learner.cpp:779-806`): partition the leaf, grow
    /// the tree node, and seed the two child `LeafSplits` for the next iteration.
    /// Returns `(left_leaf, right_leaf)` of the grown node.
    #[allow(clippy::too_many_arguments)]
    fn split_inner(
        &self,
        tree: &mut Tree,
        data_partition: &mut DataPartition,
        features: &[FeatureColumn],
        best_leaf: i32,
        feat_idx: i32,
        best: &SplitInfo,
        smaller_leaf_splits: &mut LeafSplits,
        larger_leaf_splits: &mut LeafSplits,
    ) -> Result<(i32, i32), TreeLearnerError> {
        // The feature column for the winning feature (by ORIGINAL index).
        let f = features
            .iter()
            .find(|c| c.real_feature_index == feat_idx)
            .expect("winning feature index must be a known feature column");

        // The new right child takes leaf id `num_leaves` (Tree::split convention).
        let new_left = best_leaf;
        let new_right = tree.num_leaves;

        // Partition this leaf's rows (TRL-07) via the Backend op.
        //
        // SINGLE-FEATURE-GROUP min_bin convention (D-09, the CR-01 fix): the C++
        // single-feature `FeatureGroup::Split` (`feature_group.h`, `num_feature_
        // == 1`) dispatches to `DenseBin::Split(max_bin, …)` which HARD-CODES
        // `min_bin = 1` and `USE_MIN_BIN = false` (`dense_bin.hpp:423-433`). For a
        // `most_freq_bin == 0` (offset==1) feature this `min_bin = 1` makes the
        // verbatim `th = threshold + min_bin; --th` collapse to `th = threshold`,
        // so `bin > threshold → right` / `bin <= threshold → left` — EXACTLY the
        // predict-time `fval <= bin_upper_bound[threshold]` routing. Passing the
        // raw `min_bin == 0` instead (as before) left `th = threshold - 1`, routing
        // `bin == threshold` RIGHT while predict routed it LEFT — the `[4,8]` vs
        // `[6,6]` CR-01 divergence. We mirror the C++ overload by passing
        // `min_bin + offset` (== 1 for the offset==1 single-feature spine, == the
        // raw min_bin for offset==0). max_bin / most_freq_bin are unchanged; the
        // partition `--th` body stays verbatim.
        let partition_min_bin = f.min_bin + f.offset.max(0) as u32;
        data_partition.split(
            self.backend,
            self.client,
            best_leaf,
            new_right,
            &f.bins,
            f.num_bin,
            partition_min_bin,
            f.max_bin,
            best.threshold,
            f.most_freq_bin,
        )?;

        // Grow the node. split_gain stores best.gain + min_gain_to_split (added
        // back ONLY for the tree field, :804 — NOT for selection).
        let missing_type_code = match f.missing_type {
            MissingType::None => 0i8,
            MissingType::Zero => 1,
            MissingType::NaN => 2,
        };
        let threshold_real = f.real_threshold(best.threshold);
        let split_gain_field = (best.gain + self.cfg.min_gain_to_split) as f32;
        // ACTUAL partition counts for the tree's leaf_count/internal_count
        // (serial_tree_learner.cpp:788-791, `update_cnt=true`): after the
        // data-partition routes the rows, the SplitInfo's reconstructed
        // `left_count`/`right_count` (from `round_int(hess·cnt_factor)`) are
        // OVERWRITTEN with `data_partition_->leaf_count(...)`. For fractional
        // hessians the two can disagree by ±1; the faithful tree records the real
        // partition counts (Pitfall 3). The seeded leaf-split sums + outputs come
        // from the SplitInfo (left_sum_hessian etc.) and are NOT overwritten.
        let actual_left_count = data_partition.leaf_count(new_left);
        let actual_right_count = data_partition.leaf_count(new_right);
        tree.split(
            best_leaf,
            feat_idx, // inner feature index (== real on the single-group spine)
            feat_idx, // real feature index
            best.threshold,
            threshold_real,
            best.left_output,
            best.right_output,
            actual_left_count,
            actual_right_count,
            best.left_sum_hessian,
            best.right_sum_hessian,
            split_gain_field,
            missing_type_code,
            best.default_left,
        );

        // Seed the two child LeafSplits for the next iteration
        // (serial_tree_learner.cpp:851-871). C++ seeds each child DIRECTLY from the
        // parent's `best_split_info` — `Init(leaf, dp, best_split_info.left_sum_
        // gradient, best_split_info.left_sum_hessian, best_split_info.left_output)`
        // — NOT a re-fold over the child's rows. This is load-bearing for
        // bit-exactness: `best_split_info.left_sum_hessian` is `best_sum_left_
        // hessian - kEpsilon` (feature_histogram.hpp:1042), carrying the accumulated
        // `kEpsilon` provenance from the parent's REVERSE scan. The prior re-fold
        // produced a fresh `sum_hessian` (e.g. exactly `4.0` for the mfb>0 node-2)
        // that lost that provenance and shifted the grandchild leaf-output
        // denominator by 2 ULPs (the 05-09 mfb>0 node-2 leaf-0 residual). The seed
        // provenance was confirmed against a real `lib_lightgbm` 4.6 FP execution
        // trace: node-2's scan `sum_hessian` is the parent stored
        // `left_sum_hessian` (`0x4010000000000001` = `4.000000000000001`), bumped by
        // `+2·kEpsilon` in `FindBestThreshold` (feature_histogram.hpp:172), yielding
        // `best_sum_left_hessian = 0x4000000000000004` and the golden leaf value.
        //
        // The smaller/larger selection uses the SplitInfo counts
        // (`best_split_info.left_count < best_split_info.right_count`,
        // serial_tree_learner.cpp:851), NOT the partition counts.
        //
        // `num_data_in_leaf` is the PARTITION leaf-count (C++ `LeafSplits::Init`
        // sets `num_data_in_leaf_` via `GetIndexOnLeaf` — the routed row count,
        // possibly ±1 from the SplitInfo `left_count`/`right_count` for fractional
        // hessians, Pitfall 3); only the `sum_gradients`/`sum_hessians`/`weight`
        // come from the SplitInfo.
        let part_left = data_partition.leaf_count(new_left);
        let part_right = data_partition.leaf_count(new_right);
        if best.left_count < best.right_count {
            // smaller = left
            smaller_leaf_splits.init_from_split(
                part_left,
                best.left_sum_gradient,
                best.left_sum_hessian,
                best.left_output,
            );
            larger_leaf_splits.init_from_split(
                part_right,
                best.right_sum_gradient,
                best.right_sum_hessian,
                best.right_output,
            );
        } else {
            // smaller = right
            smaller_leaf_splits.init_from_split(
                part_right,
                best.right_sum_gradient,
                best.right_sum_hessian,
                best.right_output,
            );
            larger_leaf_splits.init_from_split(
                part_left,
                best.left_sum_gradient,
                best.left_sum_hessian,
                best.left_output,
            );
        }

        Ok((new_left, new_right))
    }
}

// ---------------------------------------------------------------------------
// `features` storage: the learner is constructed with `new`, then the caller sets
// the feature columns via `with_features` (kept separate so `new`'s signature
// mirrors the C++ ctor that takes config, not data).
// ---------------------------------------------------------------------------
impl<'b, B: Backend> SerialTreeLearner<'b, B> {
    /// Attach the spine's per-feature columns (consumed/cloned in `train`).
    pub fn with_features(mut self, features: Vec<FeatureColumn>) -> Self {
        self.features = features;
        self
    }

    /// Select the histogram-build strategy (`force_row_wise` / `force_col_wise`,
    /// TRL-09). Default [`BuildStrategy::RowWise`]. On the single-thread anchor both
    /// strategies route through the SAME `construct_histograms` op and produce
    /// bit-identical trees (A1) — this flag exists to drive + assert that equality.
    pub fn with_strategy(mut self, strategy: BuildStrategy) -> Self {
        self.strategy = strategy;
        self
    }

    /// Enable per-tree / per-node feature subsampling (TRL-08) with
    /// `(feature_fraction, feature_fraction_bynode, feature_fraction_seed)`. A
    /// `feature_fraction == feature_fraction_bynode == 1.0` is equivalent to NOT
    /// calling this (the spine path — no RNG advance, all features used).
    pub fn with_feature_fraction(
        mut self,
        feature_fraction: f64,
        feature_fraction_bynode: f64,
        feature_fraction_seed: i32,
    ) -> Self {
        self.col_sampling = Some((feature_fraction, feature_fraction_bynode, feature_fraction_seed));
        self
    }

    /// Enable the GROWTH-PATH subtraction audit (T-05-07-01, TEST hook). After a
    /// `train*` call, [`take_subtract_audit`](Self::take_subtract_audit) returns one
    /// `(derived, direct)` pair per `use_subtract` larger child grown: the histogram
    /// the wired `subtract_histograms(parent, smaller)` produced vs an independent
    /// direct build of that leaf's rows. A parity test asserts they are equal
    /// cell-for-cell, proving the subtraction trick fires (and is faithful) in the
    /// ACTUAL growth path — not just in `learner_parity_subtract`'s isolation.
    #[must_use]
    pub fn with_subtract_audit(mut self) -> Self {
        self.subtract_audit = Some(std::cell::RefCell::new(Vec::new()));
        self
    }

    /// Drain the growth-path subtraction audit recorded since the last `train*`
    /// (see [`with_subtract_audit`](Self::with_subtract_audit)). Each entry is
    /// `(derived_larger_hist, direct_larger_hist)` for one `use_subtract` larger
    /// child. Empty when the audit was not enabled or no subtraction fired.
    pub fn take_subtract_audit(&mut self) -> Vec<(Vec<f64>, Vec<f64>)> {
        self.subtract_audit
            .as_ref()
            .map(|a| std::mem::take(&mut *a.borrow_mut()))
            .unwrap_or_default()
    }

    /// C++ `SerialTreeLearner::AddPredictionToScore` (`serial_tree_learner.h:100-118`).
    ///
    /// The training-path score scatter: for every leaf of the just-grown `tree`,
    /// add that leaf's f64 output to each of its rows' scores, reading the rows
    /// directly from `data_partition` (the same partition the tree was grown over).
    /// Accumulation is in f64 (the score buffer is a f64 accumulator, RESEARCH
    /// score_updater.hpp:123). A single-leaf tree (`num_leaves <= 1`) contributes
    /// nothing and early-returns, mirroring the C++ guard.
    ///
    /// `out_score` is the class-major score slice for the current tree's class; the
    /// boosting layer (06-02) calls this rather than re-walking the tree per row.
    ///
    /// NOTE on partitioning: the learner builds its `DataPartition` locally inside
    /// `train_inner` (it is not retained on `self`), so the boosting caller passes
    /// the partition explicitly — the C++ `data_partition_` member is reproduced as
    /// an argument here, keeping the scatter math identical to the reference.
    pub fn add_prediction_to_score(
        &self,
        tree: &Tree,
        data_partition: &DataPartition,
        out_score: &mut [f64],
    ) {
        if tree.num_leaves <= 1 {
            return;
        }
        for leaf in 0..tree.num_leaves {
            let out = tree.leaf_value[leaf as usize];
            for &row in data_partition.indices_in_leaf(leaf) {
                out_score[row as usize] += out;
            }
        }
    }

    /// `SerialTreeLearner::RenewTreeOutput` seam (`serial_tree_learner.cpp`,
    /// `RenewTreeOutput`).
    ///
    /// The leaf-output renewal hook the GBDT loop calls after growth and BEFORE
    /// shrinkage. For objectives whose `IsRenewTreeOutput() == false` (the spine
    /// L2 objective, 06-02) this is a NO-OP. For `regression_l1` (06-03) the real
    /// body replaces each leaf's output with the weighted median of that leaf's
    /// residuals — that math lands with the objective in 06-03.
    ///
    /// This Wave-0 seam takes an optional per-leaf renewal closure so the loop
    /// contract is stable now without coupling the learner to `lgbm-objective`
    /// (which would invert the crate dependency direction). When `renew` is `None`
    /// the tree is unchanged (the `IsRenewTreeOutput()==false` path); when `Some`,
    /// each leaf's output is replaced by `renew(leaf, rows)` over that leaf's rows.
    pub fn renew_tree_output<F>(
        &self,
        tree: &mut Tree,
        data_partition: &DataPartition,
        renew: Option<F>,
    ) where
        F: Fn(i32, &[u32]) -> f64,
    {
        let Some(renew) = renew else {
            return; // IsRenewTreeOutput() == false — no-op (spine L2).
        };
        if tree.num_leaves <= 1 {
            return;
        }
        for leaf in 0..tree.num_leaves {
            let rows = data_partition.indices_in_leaf(leaf);
            tree.leaf_value[leaf as usize] = renew(leaf, rows);
        }
    }
}

/// Per-feature byte layout of a pool slot: `(slot_off, slot_len)` where
/// `slot_off[fpos]` is feature `fpos`'s first cell in the concatenated slot buffer
/// and `slot_len` is the total cell count (`Σ 2*num_bin`). Each feature's
/// compacted histogram occupies `[slot_off[fpos], slot_off[fpos] + 2*num_bin)`.
/// Mirrors the C++ contiguous per-leaf `histogram_array_` (one feature region after
/// another); the subtraction trick operates over the whole concatenation at once.
fn feature_slot_layout(features: &[FeatureColumn]) -> (Vec<usize>, usize) {
    let mut offs = Vec::with_capacity(features.len());
    let mut acc = 0usize;
    for f in features {
        offs.push(acc);
        acc += 2 * f.num_bin as usize;
    }
    (offs, acc)
}

/// Shift a stride-2 `[g0,h0,g1,h1,…]` histogram into the C++ COMPACTED layout for
/// `offset > 0` (D-09): cell `c` ends holding the pair from REAL bin `c + offset`,
/// dropping the first `offset` bins (the most-freq / bin-0 slot that is never
/// directly folded). The now-unused tail cells are zeroed so the buffer keeps its
/// original `2 * num_bin` length — mirroring the C++ `data_` buffer whose tail is
/// unread once `offset` bins are dropped. For `offset == 0` this is a no-op (the
/// non-compacted layout where cell == real bin is already correct).
///
/// This is the SINGLE place compaction happens; it pairs with
/// [`crate::offset_for_most_freq_bin`] (the offset rule) so the stored threshold
/// (`t + offset`, recorded against the compacted scan), the partition `--th`
/// boundary, and predict routing all agree on the `most_freq_bin == 0` layout.
fn compact_histogram(hist: &mut [f64], offset: i32) {
    if offset <= 0 {
        return;
    }
    let off = offset as usize;
    let num_bin = hist.len() / 2;
    if off >= num_bin {
        // Degenerate: nothing to keep — zero the whole buffer.
        for cell in hist.iter_mut() {
            *cell = 0.0;
        }
        return;
    }
    // Shift pair `c + off` down to `c` for c in 0..(num_bin - off), in ascending
    // order (source index always >= destination, so an in-place forward copy is
    // safe and does not clobber unread sources).
    for c in 0..(num_bin - off) {
        let dst = c << 1;
        let src = (c + off) << 1;
        hist[dst] = hist[src];
        hist[dst + 1] = hist[src + 1];
    }
    // Zero the unused tail (the dropped-bin slots) so a stray read is inert.
    for cell in hist.iter_mut().skip((num_bin - off) << 1) {
        *cell = 0.0;
    }
}

/// Convert a `ColSampler::get_by_node` mask (indexed by REAL feature index) into
/// the ascending list of SELECTED real feature indices, restricted to the feature
/// columns the learner actually holds. Used for the TRL-08 per-node golden trace.
fn mask_to_indices(mask: &[i8], features: &[FeatureColumn]) -> Vec<i32> {
    let mut out: Vec<i32> = features
        .iter()
        .map(|f| f.real_feature_index)
        .filter(|&fi| mask.get(fi as usize).copied().unwrap_or(0) != 0)
        .collect();
    out.sort_unstable();
    out
}

/// `ArrayArgs<SplitInfo>::ArgMax(best_split_per_leaf_)`
/// (`serial_tree_learner.cpp:225`): the FIRST leaf whose split strictly beats all
/// earlier ones under `split_gt` (gain, then smaller feature). A flat-Vec scan,
/// NOT a heap — keeping the first-max-on-tie semantics.
fn arg_max(best_split_per_leaf: &[SplitInfo], best_feature: &[i32]) -> i32 {
    let mut best_idx = 0i32;
    for i in 1..best_split_per_leaf.len() {
        if split_gt(
            &best_split_per_leaf[i],
            best_feature[i],
            &best_split_per_leaf[best_idx as usize],
            best_feature[best_idx as usize],
        ) {
            best_idx = i as i32;
        }
    }
    best_idx
}

#[cfg(test)]
mod tests {
    use super::*;
    use lgbm_compute::{runtime::cpu_client, CpuBackend};

    fn relaxed_cfg() -> GainConfig {
        GainConfig {
            min_data_in_leaf: 1,
            min_sum_hessian_in_leaf: 0.0,
            max_delta_step: 0.0,
            lambda_l1: 0.0,
            lambda_l2: 0.0,
            min_gain_to_split: 0.0,
            path_smooth: 0.0,
        }
    }

    /// A clean splittable feature: 8 rows, 4 bins, gradient sign separates low
    /// bins (neg) from high bins (pos) so a positive-gain split exists.
    fn splittable_feature() -> (FeatureColumn, Vec<f32>, Vec<f32>) {
        // rows 0..7 -> bins 0,0,1,1,2,2,3,3 (2 rows per bin).
        let bins = vec![0u32, 0, 1, 1, 2, 2, 3, 3];
        let gradients = vec![-5.0f32, -5.0, -4.0, -4.0, 4.0, 4.0, 5.0, 5.0];
        let hessians = vec![1.0f32; 8];
        let f = FeatureColumn {
            bins,
            num_bin: 4,
            offset: 0,
            min_bin: 0,
            max_bin: 3,
            default_bin: 4, // out of range -> never the skip target
            most_freq_bin: 0,
            missing_type: MissingType::None,
            bin_upper_bound: vec![0.5, 1.5, 2.5, 3.5],
            real_feature_index: 0,
        };
        (f, gradients, hessians)
    }

    #[test]
    fn compact_histogram_offset_zero_is_noop() {
        let mut hist = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let before = hist.clone();
        super::compact_histogram(&mut hist, 0);
        assert_eq!(hist, before, "offset==0 leaves the non-compacted histogram untouched");
    }

    #[test]
    fn compact_histogram_offset_one_drops_bin0_and_shifts() {
        // 3 bins, stride-2 [g,h]: bin0=(1,2), bin1=(3,4), bin2=(5,6).
        let mut hist = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        super::compact_histogram(&mut hist, 1);
        // cell 0 <- bin1, cell 1 <- bin2, tail (cell 2) zeroed.
        assert_eq!(hist, vec![3.0, 4.0, 5.0, 6.0, 0.0, 0.0]);
    }

    #[test]
    fn train_grows_a_split_from_fixed_gh() {
        let backend = CpuBackend;
        let client = cpu_client();
        let (f, g, h) = splittable_feature();
        let mut learner = SerialTreeLearner::new(&backend, &client, relaxed_cfg(), 8, -1)
            .with_features(vec![f]);
        let tree = learner.train(&g, &h, true).expect("train ok");
        assert!(tree.num_leaves >= 2, "a splittable input grows at least 2 leaves");
        // The root split routes low bins left, high bins right.
        assert_eq!(tree.split_feature[0], 0);
    }

    /// A minimal 2-leaf tree for the score-scatter test (leaf 0 = -3.0, leaf 1 =
    /// 7.0). Predict/topology fields are irrelevant to `add_prediction_to_score`,
    /// which reads `leaf_value` + the partition only.
    fn two_leaf_tree() -> Tree {
        Tree {
            num_leaves: 2,
            num_cat: 0,
            left_child: vec![-1],
            right_child: vec![-2],
            split_feature: vec![0],
            threshold: vec![1.5],
            decision_type: vec![2],
            split_gain: vec![0.0],
            leaf_value: vec![-3.0, 7.0],
            leaf_weight: vec![0.0, 0.0],
            leaf_count: vec![4, 4],
            internal_value: vec![0.0],
            internal_weight: vec![0.0],
            internal_count: vec![8],
            cat_boundaries: vec![],
            cat_threshold: vec![],
            shrinkage: 1.0,
            is_linear: false,
            leaf_depth: vec![0, 1],
            leaf_parent: vec![-1, 0],
            split_feature_inner: vec![0],
            threshold_in_bin: vec![2],
        }
    }

    #[test]
    fn add_prediction_to_score_scatters_leaf_values_in_f64() {
        let backend = CpuBackend;
        let client = cpu_client();
        // Partition 8 rows into two leaves on the splittable feature: bins
        // 0,0,1,1 (rows 0..3) <= threshold bin 1 -> stay leaf 0; bins 2,2,3,3
        // (rows 4..7) > threshold -> right leaf 1.
        let (f, _g, _h) = splittable_feature();
        let mut part = DataPartition::new(8, 2);
        let (lc, rc) = part
            .split(
                &backend, &client, 0, 1, &f.bins, f.num_bin, f.min_bin, f.max_bin,
                /*threshold*/ 1, f.most_freq_bin,
            )
            .expect("partition split ok");
        // The exact split counts depend on the partition's threshold convention;
        // what matters for the scatter is that the two leaves cover all 8 rows.
        assert_eq!(lc + rc, 8, "both children together cover every row");
        assert!(lc > 0 && rc > 0, "both leaves are non-empty");

        let tree = two_leaf_tree();
        let learner = SerialTreeLearner::new(&backend, &client, relaxed_cfg(), 8, -1);
        let mut score = vec![0.0f64; 8];
        learner.add_prediction_to_score(&tree, &part, &mut score);

        // Each row's score == its leaf's value. Leaf 0 -> -3.0; leaf 1 -> 7.0.
        for leaf in 0..2i32 {
            let want = tree.leaf_value[leaf as usize];
            for &row in part.indices_in_leaf(leaf) {
                assert_eq!(score[row as usize], want, "row {row} (leaf {leaf})");
            }
        }
        // Total == leaf0_value * leaf0_count + leaf1_value * leaf1_count (f64).
        let total: f64 = score.iter().sum();
        let expect = tree.leaf_value[0] * lc as f64 + tree.leaf_value[1] * rc as f64;
        assert_eq!(total, expect);
    }

    #[test]
    fn add_prediction_to_score_single_leaf_is_noop() {
        let backend = CpuBackend;
        let client = cpu_client();
        let learner = SerialTreeLearner::new(&backend, &client, relaxed_cfg(), 1, -1);
        let part = DataPartition::new(4, 1);
        let mut tree = two_leaf_tree();
        tree.num_leaves = 1; // single-leaf guard path
        let mut score = vec![1.0f64; 4];
        learner.add_prediction_to_score(&tree, &part, &mut score);
        assert_eq!(score, vec![1.0; 4], "single-leaf tree contributes nothing");
    }

    #[test]
    fn renew_tree_output_none_is_noop() {
        let backend = CpuBackend;
        let client = cpu_client();
        let learner = SerialTreeLearner::new(&backend, &client, relaxed_cfg(), 2, -1);
        let part = DataPartition::new(8, 2);
        let mut tree = two_leaf_tree();
        let before = tree.leaf_value.clone();
        learner.renew_tree_output(&mut tree, &part, None::<fn(i32, &[u32]) -> f64>);
        assert_eq!(tree.leaf_value, before, "IsRenewTreeOutput()==false leaves the tree unchanged");
    }

    /// `leaf_wise_caps`: a depth-1 cap yields exactly 2 leaves on a splittable
    /// input; a non-positive-gain synthetic yields a single-leaf tree; and the
    /// `num_leaves` cap bounds the loop.
    #[test]
    fn leaf_wise_caps() {
        let backend = CpuBackend;
        let client = cpu_client();

        // (a) max_depth=1 cap -> exactly 2 leaves (one split only).
        let (f, g, h) = splittable_feature();
        let mut learner = SerialTreeLearner::new(&backend, &client, relaxed_cfg(), 8, 1)
            .with_features(vec![f]);
        let tree = learner.train(&g, &h, true).expect("train ok");
        assert_eq!(tree.num_leaves, 2, "max_depth=1 caps growth at 2 leaves");

        // (b) num_leaves cap -> at most num_leaves leaves.
        let (f2, g2, h2) = splittable_feature();
        let mut learner2 = SerialTreeLearner::new(&backend, &client, relaxed_cfg(), 3, -1)
            .with_features(vec![f2]);
        let tree2 = learner2.train(&g2, &h2, true).expect("train ok");
        assert!(tree2.num_leaves <= 3, "num_leaves caps growth");

        // (c) a no-positive-gain synthetic (uniform gradient -> flat histogram)
        //     yields a single-leaf tree (best.gain <= 0 break).
        let flat = FeatureColumn {
            bins: vec![0u32, 1, 2, 3],
            num_bin: 4,
            offset: 0,
            min_bin: 0,
            max_bin: 3,
            default_bin: 4,
            most_freq_bin: 0,
            missing_type: MissingType::None,
            bin_upper_bound: vec![0.5, 1.5, 2.5, 3.5],
            real_feature_index: 0,
        };
        // identical gradient/hessian per row -> no separating split has gain > 0.
        let gf = vec![1.0f32; 4];
        let hf = vec![1.0f32; 4];
        let mut learner3 = SerialTreeLearner::new(&backend, &client, relaxed_cfg(), 8, -1)
            .with_features(vec![flat]);
        let tree3 = learner3.train(&gf, &hf, true).expect("train ok");
        assert_eq!(tree3.num_leaves, 1, "no positive-gain split -> single leaf");
    }

    #[test]
    fn invalid_input_returns_typed_error_never_panics() {
        let backend = CpuBackend;
        let client = cpu_client();
        let (f, g, _h) = splittable_feature();

        // (a) g/h length mismatch.
        let mut learner = SerialTreeLearner::new(&backend, &client, relaxed_cfg(), 8, -1)
            .with_features(vec![f.clone()]);
        let err = learner
            .train(&g, &[1.0f32; 3], true)
            .expect_err("length mismatch must be a typed error");
        assert!(matches!(err, TreeLearnerError::LengthMismatch { .. }));

        // (b) non-positive sum_hessian (all-zero hessian -> cnt_factor guard).
        let mut learner2 = SerialTreeLearner::new(&backend, &client, relaxed_cfg(), 8, -1)
            .with_features(vec![f.clone()]);
        let zero_h = vec![0.0f32; 8];
        let err2 = learner2
            .train(&g, &zero_h, true)
            .expect_err("non-positive sum_hessian must be a typed error");
        assert!(matches!(err2, TreeLearnerError::InvalidLeafHessian { .. }));

        // (c) num_leaves < 1.
        let mut learner3 = SerialTreeLearner::new(&backend, &client, relaxed_cfg(), 0, -1)
            .with_features(vec![f.clone()]);
        let err3 = learner3
            .train(&g, &vec![1.0f32; 8], true)
            .expect_err("num_leaves < 1 must be a typed error");
        assert!(matches!(err3, TreeLearnerError::InvalidNumLeaves { .. }));

        // (d) na_as_missing feature is the deferred typed error.
        let na_feat = FeatureColumn {
            num_bin: 4,
            missing_type: MissingType::NaN,
            ..f
        };
        let mut learner4 = SerialTreeLearner::new(&backend, &client, relaxed_cfg(), 8, -1)
            .with_features(vec![na_feat]);
        let err4 = learner4
            .train(&g, &vec![1.0f32; 8], true)
            .expect_err("na_as_missing is deferred -> typed error");
        assert!(matches!(err4, TreeLearnerError::Compute(_)));
    }

    /// Conservation: left_count + right_count of the root split == num_data, and
    /// the grown tree round-trips through to_string()/parse byte-stably.
    #[test]
    fn root_split_conserves_rows_and_serializes() {
        let backend = CpuBackend;
        let client = cpu_client();
        let (f, g, h) = splittable_feature();
        let mut learner = SerialTreeLearner::new(&backend, &client, relaxed_cfg(), 8, 1)
            .with_features(vec![f]);
        let tree = learner.train(&g, &h, true).expect("train ok");
        // 2-leaf tree: leaf_count sums to 8.
        let total: i32 = tree.leaf_count.iter().sum();
        assert_eq!(total, 8, "rows conserved across leaves");
        // Serialized form is byte-stable + round-trips.
        let s = tree.to_string();
        let parsed = Tree::parse(&s).expect("grown tree round-trips");
        assert_eq!(parsed.to_string(), s);
    }
}

/// A single-leaf root `Tree` at depth 0 (the C++ freshly-initialized growth
/// state): one leaf with the seeded `root_output`, `internal_count = num_data`.
fn root_tree(root_output: f64, num_data: i32) -> Tree {
    Tree {
        num_leaves: 1,
        num_cat: 0,
        left_child: Vec::new(),
        right_child: Vec::new(),
        split_feature: Vec::new(),
        threshold: Vec::new(),
        decision_type: Vec::new(),
        split_gain: Vec::new(),
        leaf_value: vec![root_output],
        leaf_weight: vec![0.0],
        leaf_count: vec![num_data],
        internal_value: Vec::new(),
        internal_weight: Vec::new(),
        internal_count: Vec::new(),
        cat_boundaries: Vec::new(),
        cat_threshold: Vec::new(),
        shrinkage: 1.0,
        is_linear: false,
        leaf_depth: vec![0],
        leaf_parent: vec![-1],
        split_feature_inner: Vec::new(),
        threshold_in_bin: Vec::new(),
    }
}
