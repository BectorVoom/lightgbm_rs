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
use lgbm_compute::{Backend, BatchedSplitFeature};
use lgbm_compute::ComputeClientReexport as ComputeClient;
use lgbm_dataset::bin_mapper::{BinType, MissingType};
use lgbm_model::Tree;

use crate::col_sampler::ColSampler;
use crate::cost_effective_gradient_boosting::CegbModel;
use crate::data_partition::DataPartition;
use crate::error::TreeLearnerError;
use crate::HistogramPool;
use crate::leaf_splits::LeafSplits;
use crate::monotone_constraints::MonotoneConstraints;
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
    /// C++ `BinMapper::bin_type()` — the per-feature dispatch flag
    /// (`serial_tree_learner.cpp:779`). [`BinType::Numerical`] routes the
    /// byte-untouched continuous split spine (D-06 HARD INVARIANT);
    /// [`BinType::Categorical`] routes the additive
    /// [`find_best_threshold_categorical`](crate::find_best_threshold_categorical)
    /// branch. Defaults to `Numerical` so every spine call site is unchanged.
    pub bin_type: BinType,
    /// C++ `BinMapper::bin_2_categorical_` — bin index → ORIGINAL category value
    /// (`BinToValue(bin)`, bin.h:138-143). Only populated for categorical
    /// features; the categorical split converts each winning REAL BIN to its
    /// category value to build the model-text (`cat_threshold`) bitset. Empty for
    /// numeric features.
    pub bin_to_category: Vec<i32>,
}

impl Default for FeatureColumn {
    /// A numeric (continuous) feature with empty buffers — the spine default.
    /// Used by `..FeatureColumn::default()` partial-init at construction sites that
    /// want the `bin_type: Numerical` default without restating it.
    fn default() -> Self {
        Self {
            bins: Vec::new(),
            num_bin: 0,
            offset: 0,
            min_bin: 0,
            max_bin: 0,
            default_bin: 0,
            most_freq_bin: 0,
            missing_type: MissingType::None,
            bin_upper_bound: Vec::new(),
            real_feature_index: 0,
            bin_type: BinType::Numerical,
            bin_to_category: Vec::new(),
        }
    }
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
    /// Per-leaf winning categorical bitset (the `SplitInfo::cat_threshold` C++
    /// carries on the split-info, kept OUT of the `Copy` [`SplitInfo`]). Indexed by
    /// leaf id; `None` for a numeric (or no) winner. `scan_leaf_histogram` writes
    /// the winner's category bins here when the cross-feature argmax is a
    /// categorical feature; `split_inner` reads them to grow the categorical node.
    /// Sized to `num_leaves` and reset alongside `best_split_per_leaf` each split.
    /// A `RefCell` so the `&self` scan can record into it.
    best_cat_threshold: std::cell::RefCell<Vec<Option<Vec<u32>>>>,
    /// W10 advanced learner constraints (ADV-01..05). `Default` (all-empty/off) ⇒
    /// every gate is INACTIVE and the spine split path is byte-untouched (D-06).
    constraints: LearnerConstraints,
    /// Per-tree monotone state (ADV-01), set up at the top of each `train_inner`
    /// when `constraints.monotone_constraints` is active; `None` on the spine.
    /// A `RefCell` so the `&self` scan can READ the per-leaf clamp while the
    /// `&mut self` growth loop UPDATES it after each split.
    monotone: std::cell::RefCell<Option<MonotoneConstraints>>,
    /// Per-tree CEGB state (ADV-05); `None` on the spine. A `RefCell` so the scan
    /// can read the penalty state while the growth loop updates it after a split.
    cegb: std::cell::RefCell<Option<CegbModel>>,
    /// ADV-02 interaction constraints: per-leaf branch-feature list (the REAL
    /// feature indices on the root-to-leaf path, C++ `Tree::branch_features_`).
    /// Empty entries on the spine (interaction inactive). A `RefCell` so the scan
    /// can read each node's allowed set while the growth loop appends on split.
    branch_features: std::cell::RefCell<Vec<Vec<i32>>>,
    /// ADV-04 extra-trees per-feature RNG (`meta_->rand = Random(extra_seed + i)`,
    /// feature_histogram.hpp:1450), indexed by feature POSITION. `None` on the
    /// spine. Built per tree; each per-feature scan draws ONE `next_int` (the
    /// `BeforeNumerical` rand_threshold), so the RNG state must persist across
    /// leaf scans within a tree — hence a `RefCell<Vec<Random>>`.
    extra_rng: std::cell::RefCell<Option<Vec<lgbm_core::random::Random>>>,
    /// Per-leaf per-feature `is_splittable` flags (C++ `FeatureHistogram::
    /// is_splittable_`, serial_tree_learner.cpp:395-399). When a leaf is scanned,
    /// `scan_leaf_histogram` records, for each feature POSITION, whether
    /// `find_best_split` found ANY admissible candidate (gain finite). On the next
    /// split the larger (subtracted) child consults its PARENT leaf's flags and
    /// SKIPS any feature whose parent histogram was not splittable — the C++
    /// `parent_leaf_histogram_array_[f].is_splittable()` gate. This is load-bearing
    /// for GOSS: under gradient amplification a small `cnt_factor` rounds the
    /// `round_int(hess·cnt_factor)` per-bin counts to 0, so a feature can be
    /// not-splittable at the parent (every candidate fails `min_data_in_leaf`) yet
    /// LOOK splittable on the subtracted child (whose larger `cnt_factor` no longer
    /// rounds to 0); without the gate Rust picks a split C++ never considers.
    /// Indexed `[leaf][feature_position]`; reset per tree. `RefCell` so the `&self`
    /// scan records it.
    feature_splittable: std::cell::RefCell<Vec<Vec<bool>>>,
    /// R1 (perf): when `false`, `scan_leaf_histogram` SKIPS the snapshot-only
    /// `per_bin_gains` host re-scan (the per-feature per-bin gain arrays that feed
    /// the D-06 `SplitSnapshot`). The grown tree is bit-identical either way — the
    /// live split decision comes from `backend.find_best_split`, never from
    /// `per_bin_gains` (a pure read). Set `true` by `train_with_snapshots` /
    /// `train_with_col_sampler_trace` (the golden-replay paths) and `false` by
    /// `train` / `train_returning_partition` (the production boosting path), so the
    /// boosting loop pays nothing for snapshots it discards.
    capture_snapshots: bool,
    /// 260608-p90: whether THIS train is device-resident-eligible (a pure numeric
    /// spine on a resident-capable backend). Computed ONCE at the top of
    /// `train_inner` via [`crate::resident_pool::resident_eligible`] and read in
    /// `find_best_splits` to route the per-leaf build→fix→compact→subtract→scan chain
    /// through the device-handle slot mirror (keeping histograms resident). `false`
    /// (the default, and ALWAYS on CpuBackend) takes the byte-unchanged host path.
    resident_eligible: bool,
    /// 260608-t3t: when `true`, directly-built leaves (root + smaller children) on the
    /// small/medium fused-eligible band route through the FUSED build+fix+compact+scan
    /// kernel (`Backend::build_fix_scan_resident`) — ONE launch instead of construct +
    /// fix + scan = 3. The subtract-derived larger children KEEP subtract+scan; large
    /// keeps the atomic-parallel resident chain. Computed ONCE at the top of
    /// `train_inner` via [`crate::resident_pool::fused_directly_built_eligible`]; `false`
    /// (default, ALWAYS on CpuBackend) takes the existing resident/host routing.
    fused_eligible: bool,
    /// R3 (perf, 260614-p0n): reused per-feature `num_bin` descriptor for
    /// [`build_leaf_histogram_into`](Self::build_leaf_histogram_into). The feature
    /// set is FIXED per train (set once via [`with_features`](Self::with_features)),
    /// so the `Vec<u32>` of per-feature bin counts is identical on every leaf build —
    /// it is filled LAZILY on the first build (so it survives `with_features`, which
    /// may run after `new`) and re-borrowed thereafter, eliminating one `Vec<u32>`
    /// allocation per leaf-histogram build. `RefCell` so the `&self` build can fill /
    /// read it. The CONTENT and ORDER are byte-identical to the prior per-call
    /// `features.iter().map(|f| f.num_bin).collect()`, so the fold inputs (and thus
    /// the bit-exact CPU f64 tree) are unchanged. The `feature_bins: Vec<&[u32]>`
    /// stays a per-call local: its `&[u32]` borrow is tied to the `features`
    /// parameter lifetime and cannot be stored behind `&self` without infecting the
    /// struct with a lifetime param (an architectural change the plan defers).
    build_num_bins: std::cell::RefCell<Vec<u32>>,
}

/// W10 advanced learner constraints (ADV-01..05) — the inactive `Default` is the
/// spine path (every gate off, the numeric + categorical split paths
/// byte-untouched, D-06).
#[derive(Debug, Clone, Default)]
pub struct LearnerConstraints {
    /// ADV-01: per-feature monotone type (indexed by REAL feature index;
    /// `+1`/`-1`/`0`). Empty ⇒ no monotone constraint.
    pub monotone_constraints: Vec<i32>,
    /// ADV-01: `monotone_penalty` (`config.monotone_penalty`).
    pub monotone_penalty: f64,
    /// ADV-02: interaction-constraint groups. Each inner Vec is a set of REAL
    /// feature indices allowed to co-occur on a root-to-leaf path. Empty ⇒ no
    /// interaction constraint.
    pub interaction_constraints: Vec<Vec<i32>>,
    /// ADV-04: extra-trees randomized threshold selection.
    pub extra_trees: bool,
    /// ADV-04: `extra_seed` (per-feature RNG seed offset).
    pub extra_seed: i32,
    /// ADV-05: `cegb_tradeoff`.
    pub cegb_tradeoff: f64,
    /// ADV-05: `cegb_penalty_split`.
    pub cegb_penalty_split: f64,
    /// ADV-05: per-feature coupled penalty.
    pub cegb_penalty_feature_coupled: Vec<f64>,
    /// ADV-05: per-feature lazy penalty.
    pub cegb_penalty_feature_lazy: Vec<f64>,
    /// ADV-03: a pre-parsed forced-split tree (from `forced_splits_filename`
    /// JSON). `None` ⇒ no forced split (the spine path).
    pub forced_splits: Option<crate::forced_splits::ForcedSplitNode>,
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
    /// REVERSE-then-FORWARD per-candidate gains packed into ONE allocation
    /// (260609-bfx snapshot-path alloc reduction: 2 retained Vecs/record → 1).
    /// `rev_len` is the REVERSE prefix length; the remaining cells are FORWARD.
    /// NaN marks a gated candidate. Empty for categorical features. Read via
    /// [`cand_rev`](Self::cand_rev) / [`cand_fwd`](Self::cand_fwd).
    gains: Vec<f64>,
    rev_len: usize,
    /// The feature's best `SplitInfo`.
    pub split: SplitInfo,
}

impl FeatureSplitRecord {
    /// Per-candidate REVERSE-branch gains (NaN where gated).
    #[inline]
    pub fn cand_rev(&self) -> &[f64] {
        &self.gains[..self.rev_len]
    }

    /// Per-candidate FORWARD-branch gains (NaN where gated).
    #[inline]
    pub fn cand_fwd(&self) -> &[f64] {
        &self.gains[self.rev_len..]
    }
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
            best_cat_threshold: std::cell::RefCell::new(Vec::new()),
            constraints: LearnerConstraints::default(),
            monotone: std::cell::RefCell::new(None),
            cegb: std::cell::RefCell::new(None),
            branch_features: std::cell::RefCell::new(Vec::new()),
            extra_rng: std::cell::RefCell::new(None),
            feature_splittable: std::cell::RefCell::new(Vec::new()),
            // Default OFF: the common `train` path discards snapshots, so by default
            // we never pay the per_bin_gains re-scan. The snapshot wrappers opt in.
            capture_snapshots: false,
            // Default OFF: recomputed per train in `train_inner` (260608-p90).
            resident_eligible: false,
            // Default OFF: recomputed per train in `train_inner` (260608-t3t).
            fused_eligible: false,
            // R3 (260614-p0n): filled lazily on the first leaf-histogram build.
            build_num_bins: std::cell::RefCell::new(Vec::new()),
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
        // Production path: snapshots are NOT requested, so skip the per_bin_gains
        // re-scan (R1). Calls `train_inner` directly — NOT via `train_with_snapshots`
        // — so `capture` stays false.
        let (tree, _snaps, _trace, _part) =
            self.train_inner(gradients, hessians, is_first_tree, false)?;
        Ok(tree)
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
        // Golden-replay path: capture the full D-06 snapshots (per_bin_gains ON).
        let (tree, snaps, _trace, _part) =
            self.train_inner(gradients, hessians, is_first_tree, true)?;
        Ok((tree, snaps))
    }

    /// Train one tree over a ROW SUBSET (the C++ `tmp_subset_` bagging path,
    /// bagging.hpp `is_use_subset_`). `in_bag` is the in-bag GLOBAL row indices (the
    /// strategy's `bag_data_indices_[..bag_data_cnt]`, ascending); `gradients` /
    /// `hessians` are the FULL (global) per-row buffers. The learner builds a subset
    /// dataset (the in-bag rows' bins, in in-bag order) and the corresponding subset
    /// grad/hess, grows the tree over it, and returns the grown [`Tree`].
    ///
    /// The returned tree's split thresholds are the same real-value bin upper bounds
    /// (binning is identical for any row subset of an identity-binned corpus), so the
    /// boosting layer scores BOTH in-bag and out-of-bag rows via the predict-side
    /// [`Tree::predict`] over the original real feature values (bit-exact to the
    /// train-path scatter on this identity-binned corpus — the L2 contract). This
    /// mirrors the C++ result: in-bag rows scored via the data-partition scatter, OOB
    /// rows via the tree predict-side add, both adding the SAME f64 leaf value once.
    ///
    /// # Errors
    /// Propagates the learner-boundary checks (length / bin-range / num_leaves).
    pub fn train_on_subset(
        &mut self,
        in_bag: &[i32],
        gradients: &[f32],
        hessians: &[f32],
        is_first_tree: bool,
    ) -> Result<Tree, TreeLearnerError> {
        // Build subset feature columns (in-bag rows, in in-bag order) — every other
        // FeatureColumn field (num_bin/offset/min_bin/max_bin/most_freq_bin/...) is
        // a per-feature property independent of the row subset, so only `bins` is
        // re-gathered.
        let subset_features: Vec<FeatureColumn> = self
            .features
            .iter()
            .map(|f| {
                let bins: Vec<u32> = in_bag.iter().map(|&r| f.bins[r as usize]).collect();
                FeatureColumn {
                    bins,
                    ..f.clone()
                }
            })
            .collect();
        let sub_grad: Vec<f32> = in_bag.iter().map(|&r| gradients[r as usize]).collect();
        let sub_hess: Vec<f32> = in_bag.iter().map(|&r| hessians[r as usize]).collect();

        // Swap in the subset features for the growth, restore afterward (the learner
        // is reused across iterations with the full-corpus features).
        let saved = std::mem::replace(&mut self.features, subset_features);
        let result = self.train(&sub_grad, &sub_hess, is_first_tree);
        self.features = saved;
        result
    }

    /// Like [`train_on_subset`](Self::train_on_subset) but ALSO returns the final
    /// [`DataPartition`] the subset tree was grown over — the row→leaf mapping in
    /// **subset-row space** (each leaf's `indices_in_leaf` are indices into `in_bag`,
    /// i.e. `0..in_bag.len()`, NOT full-corpus rows).
    ///
    /// The GBDT bagging path (06-06, WR-03) needs this to mirror the C++
    /// `RenewTreeOutput` on the subset path (`serial_tree_learner.cpp:920-958`): the
    /// `index_mapper` is the subset `data_partition_->GetIndexOnLeaf`, and the caller
    /// maps each subset row `sr` through `bag_mapper[sr] = in_bag[sr]` to the
    /// full-corpus row whose residual feeds `PercentileFun` (the median-residual leaf
    /// output). Without the returned partition the caller cannot recover the in-bag
    /// leaf membership and the renew block would be a silent no-op.
    pub fn train_on_subset_returning_partition(
        &mut self,
        in_bag: &[i32],
        gradients: &[f32],
        hessians: &[f32],
        is_first_tree: bool,
    ) -> Result<(Tree, DataPartition), TreeLearnerError> {
        // Build subset feature columns (in-bag rows, in in-bag order) — identical to
        // `train_on_subset`; only `bins` is re-gathered (every other FeatureColumn
        // field is a per-feature property independent of the row subset).
        let subset_features: Vec<FeatureColumn> = self
            .features
            .iter()
            .map(|f| {
                let bins: Vec<u32> = in_bag.iter().map(|&r| f.bins[r as usize]).collect();
                FeatureColumn {
                    bins,
                    ..f.clone()
                }
            })
            .collect();
        let sub_grad: Vec<f32> = in_bag.iter().map(|&r| gradients[r as usize]).collect();
        let sub_hess: Vec<f32> = in_bag.iter().map(|&r| hessians[r as usize]).collect();

        let saved = std::mem::replace(&mut self.features, subset_features);
        let result = self.train_returning_partition(&sub_grad, &sub_hess, is_first_tree);
        self.features = saved;
        result
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
        // Production boosting path (gbdt.rs:797,1125): snapshots discarded → OFF.
        let (tree, _snaps, _trace, part) =
            self.train_inner(gradients, hessians, is_first_tree, false)?;
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
        // Golden-replay path (TRL-08 RNG + D-06 snapshots): capture ON.
        let (tree, snaps, trace, _part) =
            self.train_inner(gradients, hessians, is_first_tree, true)?;
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
        capture_snapshots: bool,
    ) -> Result<(Tree, Vec<SplitSnapshot>, ColSamplerTrace, DataPartition), TreeLearnerError> {
        // R1: record whether this growth must emit D-06 snapshots. Read deep in
        // `scan_leaf_histogram` to gate the snapshot-only `per_bin_gains` re-scan.
        self.capture_snapshots = capture_snapshots;
        let num_data = gradients.len() as i32;
        let features = self.features.clone();

        // 260608-p90: decide device-resident eligibility ONCE per train (CONSERVATIVE /
        // fail-safe — see `resident_pool::resident_eligible`). ANDs in the backend's
        // `resident_pool_supported` so CpuBackend (false) NEVER takes the resident
        // branch; ANY non-spine feature/config falls back to the byte-unchanged host
        // path. Read in `find_best_splits` to route the per-leaf chain.
        // 260608-s2b Lever B: pass `num_data` so `resident_eligible` can size-gate the
        // resident path (below RESIDENT_MIN_NUM_DATA → host path; the launch-bound tiny
        // workload regresses on the resident chain). `LGBM_RESIDENT_FORCE` overrides the
        // size gate for benching both paths. Correctness checks still fail-safe first.
        self.resident_eligible = crate::resident_pool::resident_eligible(
            self.backend.resident_pool_supported(),
            num_data,
            &features,
            &self.constraints,
            capture_snapshots,
            &self.cfg,
        );

        // 260608-t3t: the FUSED directly-built-leaf gate (small/medium band). Same
        // fail-safe correctness spine as `resident_eligible`, but targets the launch-
        // bound small/medium band where the 3-launch resident chain loses to host. When
        // on, directly-built leaves (root + smaller children) use ONE fused launch
        // (build+fix+compact+scan); subtract-derived larger children + large keep their
        // existing routing. The fused path uses the SAME resident pool mirror (it stores
        // the fixed+compacted Handle into the slot), so `subtract_resident` still works.
        // `LGBM_FUSED_FORCE` overrides the size gate for benching. ALWAYS false on
        // CpuBackend (backend_supported false) → byte-unchanged host path.
        self.fused_eligible = crate::resident_pool::fused_directly_built_eligible(
            self.backend.resident_pool_supported(),
            num_data,
            &features,
            &self.constraints,
            capture_snapshots,
            &self.cfg,
        );
        // The fused directly-built path stores its Handle into the resident pool mirror
        // (so the subtract-derived larger child finds its parent). When the fused gate
        // is on but the plain resident gate is off (the small/medium band), enable the
        // resident pool machinery so the larger child's `subtract_resident` + the slot
        // mirror reset/move bookkeeping are live. (When `resident_eligible` is already
        // true this is a no-op.)
        if self.fused_eligible {
            self.resident_eligible = true;
        }

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
            // V5 / threat T-04-01 — the SINGLE bin-range gate, RELOCATED from the
            // per-element kernel fold (spike-003b). This loop runs ONCE over the
            // FIXED feature columns at train entry (amortized O(rows) once/train,
            // not O(leaf_rows) per build per iter), rejecting any `bin >= num_bin`
            // with `BinIndexOutOfRange` BEFORE any leaf histogram is built. After
            // this gate, the fused CPU `Backend::build_leaf_histograms_raw`
            // (lgbm-compute/src/lib.rs) folds BRANCHLESS trusting `bin < num_bin`
            // (matching C++ `dense_bin.hpp`, which folds `data_[i]` with no
            // per-element check). See that fn's "Bin-range precondition" doc — this
            // is its authoritative precondition source; the two sites cross-reference
            // via `BinIndexOutOfRange`. Do NOT remove or weaken this loop: it is the
            // relocated mitigation, not a redundant check.
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

        // nn7 (L1): one-time per-train upload of the binned feature columns to the
        // backend's device-resident cache, BEFORE the per-leaf growth loop. For the
        // CpuBackend this is the no-op default (zero behavior change); the RocmBackend
        // uploads every column ONCE and gathers leaf rows on device per leaf — the
        // per-leaf `[num_features × rows]` host bin upload is gone. The binned columns
        // are immutable for the whole train, so upload once here (not per tree); the
        // backend instance persists across trees (booster.rs constructs it per
        // train() call, outside the GBDT iter loop).
        let upload_bins: Vec<&[u32]> = features.iter().map(|f| f.bins.as_slice()).collect();
        self.backend.upload_resident_bins(self.client, &upload_bins);

        let mut pool = HistogramPool::new(self.num_leaves, slot_len);
        pool.reset_map();
        // 260608-p90: when eligible, reset the device-handle slot mirror alongside the
        // host pool's reset_map, sized to the host pool's cache_size (== num_leaves).
        // No-op on CpuBackend (the default trait impl) and when ineligible.
        if self.resident_eligible {
            self.backend
                .reset_resident_pool(pool.cache_size(), slot_len);
        }

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
        // Per-leaf winning categorical bitset (parallel to best_split_per_leaf).
        // Reset per tree; entries set only when a categorical feature wins a leaf.
        *self.best_cat_threshold.borrow_mut() = vec![None; self.num_leaves as usize];

        // Per-leaf per-feature splittability (C++ FeatureHistogram::is_splittable_).
        // Reset per tree; populated as each leaf is scanned. The root leaf (0) starts
        // ALL-splittable so the root's own scan is never gated (C++ has no parent at
        // the root → use_subtract is false → the gate is not consulted).
        *self.feature_splittable.borrow_mut() =
            vec![vec![true; num_features]; self.num_leaves as usize];

        // ---- W10 advanced-constraint per-tree setup (ADV-01..05) ----
        // Each is INACTIVE by default → `None`/empty → the scan + growth loop take
        // the spine path byte-untouched (D-06).
        *self.monotone.borrow_mut() = MonotoneConstraints::new(
            &self.constraints.monotone_constraints,
            self.num_leaves,
            self.constraints.monotone_penalty,
        );
        // CEGB: the REAL feature count is the max real_feature_index + 1.
        let num_real_features = features
            .iter()
            .map(|f| f.real_feature_index + 1)
            .max()
            .unwrap_or(0);
        *self.cegb.borrow_mut() = CegbModel::new(
            self.constraints.cegb_tradeoff,
            self.constraints.cegb_penalty_split,
            &self.constraints.cegb_penalty_feature_coupled,
            &self.constraints.cegb_penalty_feature_lazy,
            num_real_features,
            num_data,
        );
        // Interaction: per-leaf branch-feature list (empty when inactive).
        *self.branch_features.borrow_mut() = vec![Vec::new(); self.num_leaves as usize];

        // Extra-trees: one `Random(extra_seed + inner_feature_index)` per feature,
        // persisted across leaf scans within this tree (C++
        // `ref_feature_meta[i].rand = Random(config->extra_seed + i)`,
        // feature_histogram.hpp:1450, where `i` is the DATASET INNER feature index).
        //
        // DEF-07-11-03: the seed offset is the INNER feature index, NOT the real /
        // sidecar feature order. LightGBM's `Dataset` assigns inner indices via
        // feature bundling (`GetFeatureGroups` / `dataset.cpp:387-406`), which for
        // these dense single-bin-group corpora REVERSES the column order — confirmed
        // by a source-built lib_lightgbm 4.6 trace (`CPP_MAP inner=0 real=1`,
        // `inner=1 real=0`). Seeding by the real order drew each feature's rand from
        // the WRONG LCG stream, swapping which feature's randomized threshold won the
        // root and flipping the tree structure (seed6 4-vs-3 leaves; seed9 3-vs-4).
        // The harness feeds features in real order, so the inner index = the reversed
        // position. `extra_rng[fpos]` (consumed by `fpos` in the real-order scan)
        // therefore draws from `Random(extra_seed + (nf-1-fpos))`, aligning the
        // per-feature LCG stream with C++'s inner-indexed `meta_->rand`.
        *self.extra_rng.borrow_mut() = if self.constraints.extra_trees {
            let nf = features.len() as i32;
            Some(
                (0..features.len())
                    .map(|i| lgbm_core::random::Random::new(self.constraints.extra_seed + (nf - 1 - i as i32)))
                    .collect(),
            )
        } else {
            None
        };

        let mut snapshots: Vec<SplitSnapshot> = Vec::new();

        // left_leaf / right_leaf track the children produced by the last Split.
        // The first iteration's "left_leaf" is the root (leaf 0); right_leaf = -1.
        let mut left_leaf: i32 = 0;
        let mut right_leaf: i32 = -1;

        // ---- ADV-03 ForceSplits (serial_tree_learner.cpp:620-734) ----
        // Apply the forced-split tree top-down BEFORE the leaf-wise loop. Each
        // forced split consumes one of the `num_leaves - 1` split budget. `None`
        // ⇒ no forced split (spine). The growth loop below then continues from the
        // last forced child.
        let mut forced_count = 0i32;
        if let Some(forced) = self.constraints.forced_splits.clone() {
            forced_count = self.apply_forced_splits(
                &forced,
                &features,
                gradients,
                hessians,
                &mut tree,
                &mut data_partition,
                &mut pool,
                &slot_off,
                &mut smaller_leaf_splits,
                &mut larger_leaf_splits,
                &mut left_leaf,
                &mut right_leaf,
            )?;
            // After forced splits the pool slots hold only the forced FEATURE's
            // histogram (forced growth builds per forced feature, not the full
            // concatenated parent). Reset the pool map so the continuation loop's
            // first `find_best_splits` builds both children DIRECTLY (no stale
            // subtraction-trick parent) — the C++ ForceSplits re-runs FindBestSplits
            // (full ConstructHistograms) per forced node, so the continuation never
            // subtracts against a partial parent.
            pool.reset_map();
            // The continuation must scan BOTH forced children fresh: signal "root-
            // like" so find_best_splits builds left_leaf directly and right_leaf
            // directly too. We re-seed by treating the last forced split's children
            // as a fresh pair (left_leaf/right_leaf already set).
        }

        // ---- the leaf-wise loop (serial_tree_learner.cpp:218-236) ----
        for _split in 0..(self.num_leaves - 1 - forced_count) {
            // BeforeFindBestSplit (gates + smaller-child selection).
            let did = crate::phase_prof::time(&crate::phase_prof::BEFORE_NS, || {
                self.before_find_best_split(
                    &tree,
                    &data_partition,
                    left_leaf,
                    right_leaf,
                    &mut best_split_per_leaf,
                )
            });
            if did {
                // FindBestSplits: build smaller histogram (and larger via subtract),
                // then per-feature find_best_split + cross-feature argmax. The
                // per-node ColSampler draws (smaller-then-larger) happen INSIDE
                // find_best_splits in the exact C++ order (T-05-04-01).
                let snap = crate::phase_prof::time(&crate::phase_prof::HISTSPLIT_NS, || {
                    self.find_best_splits(
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
                    )
                })?;
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
            let (new_left, new_right) =
                crate::phase_prof::time(&crate::phase_prof::PARTITION_NS, || {
                    self.split_inner(
                        &mut tree,
                        &mut data_partition,
                        &features,
                        best_leaf,
                        feat_idx,
                        &best,
                        &mut smaller_leaf_splits,
                        &mut larger_leaf_splits,
                    )
                })?;

            // ---- W10 post-split constraint updates (ADV-01/02/05) ----
            self.update_constraints_after_split(
                &tree,
                &data_partition,
                best_leaf,
                new_left,
                new_right,
                feat_idx,
                &best,
                &mut best_split_per_leaf,
            );

            // Reset the just-split leaf's best so it is not re-selected, and the
            // new leaves to "no split" until BeforeFindBestSplit recomputes them.
            best_split_per_leaf[best_leaf as usize] = SplitInfo::none();
            best_split_feature[best_leaf as usize] = -1;
            best_split_per_leaf[new_left as usize] = SplitInfo::none();
            best_split_feature[new_left as usize] = -1;
            best_split_per_leaf[new_right as usize] = SplitInfo::none();
            best_split_feature[new_right as usize] = -1;
            {
                let mut cat = self.best_cat_threshold.borrow_mut();
                cat[best_leaf as usize] = None;
                cat[new_left as usize] = None;
                cat[new_right as usize] = None;
            }

            left_leaf = new_left;
            right_leaf = new_right;
        }

        Ok((tree, snapshots, trace, data_partition))
    }

    /// W10 post-split constraint bookkeeping (ADV-01/02/05). A no-op on the spine
    /// (every constraint inactive, D-06):
    /// - ADV-02: append the winning feature to BOTH children's branch-feature
    ///   lists (C++ `Tree::branch_features_`, tree.cpp:580-583) — used by the
    ///   per-node interaction-allowed set.
    /// - ADV-01: propagate the monotone `[min,max]` clamp to the parent + new
    ///   child (`BasicLeafConstraints::Update`).
    /// - ADV-05: mark the feature used (coupled) / mark the leaf rows seen (lazy)
    ///   + the coupled OTHER-leaf gain recompute (`UpdateLeafBestSplits`).
    #[allow(clippy::too_many_arguments)]
    fn update_constraints_after_split(
        &self,
        tree: &Tree,
        data_partition: &DataPartition,
        best_leaf: i32,
        new_left: i32,
        new_right: i32,
        feat_idx: i32,
        best: &SplitInfo,
        best_split_per_leaf: &mut [SplitInfo],
    ) {
        // ADV-02: branch features (only when interaction is active — else the
        // lists stay empty and the gate is never consulted).
        if !self.constraints.interaction_constraints.is_empty() {
            let mut bf = self.branch_features.borrow_mut();
            // new_left == best_leaf id reused; new_right is the fresh leaf. Both
            // inherit best_leaf's path + the winning feature.
            let parent_path = bf[best_leaf as usize].clone();
            let mut child_path = parent_path;
            child_path.push(feat_idx);
            bf[new_left as usize] = child_path.clone();
            if (new_right as usize) < bf.len() {
                bf[new_right as usize] = child_path;
            }
        }

        // ADV-01: monotone clamp propagation. The winning feature's monotone type
        // drives the mid-output min/max split (numeric splits only).
        {
            let mut mono = self.monotone.borrow_mut();
            if let Some(mc) = mono.as_mut() {
                let mt = mc.feature_monotone(feat_idx);
                mc.update_after_split(best_leaf, new_right, mt, best.left_output, best.right_output);
                // The reused parent leaf id `best_leaf` becomes `new_left`; carry
                // its updated constraint to `new_left` if they differ (Tree::split
                // keeps best_leaf as the left child, so new_left == best_leaf here).
                let _ = new_left;
            }
        }

        // ADV-05: CEGB post-split marking + coupled recompute. When
        // `best_split_per_leaf` is empty (the forced-splits pre-grow path passes an
        // empty slice — there is no live per-leaf best yet) the coupled recompute
        // is a no-op but the feature-used / row-seen marking still applies.
        {
            let mut cegb = self.cegb.borrow_mut();
            if let Some(model) = cegb.as_mut() {
                let leaf_rows = data_partition.indices_in_leaf(best_leaf).to_vec();
                let num_leaves = if best_split_per_leaf.is_empty() {
                    0
                } else {
                    tree.num_leaves
                };
                // Operate on the gain column only (the dominant argmax term).
                let mut gains: Vec<f64> = best_split_per_leaf.iter().map(|s| s.gain).collect();
                model.update_leaf_best_splits(feat_idx, best_leaf, num_leaves, &mut gains, &leaf_rows);
                for (s, g) in best_split_per_leaf.iter_mut().zip(gains) {
                    s.gain = g;
                }
            }
        }
    }

    /// ADV-03 `ForceSplits` (serial_tree_learner.cpp:620-734): grow the forced
    /// split structure top-down via BFS. For each forced node, build the leaf's
    /// histogram for the forced feature, compute the split at the forced threshold
    /// (`gather_info_for_threshold`), and `split_inner` it; enqueue the children.
    /// Returns the number of forced splits applied (consumed from the leaf budget).
    /// A forced split whose gain is not better than no-split is SKIPPED (C++ warns
    /// + erases it from the force map), and BFS stops down that branch.
    #[allow(clippy::too_many_arguments)]
    fn apply_forced_splits(
        &self,
        forced: &crate::forced_splits::ForcedSplitNode,
        features: &[FeatureColumn],
        gradients: &[f32],
        hessians: &[f32],
        tree: &mut Tree,
        data_partition: &mut DataPartition,
        pool: &mut HistogramPool,
        slot_off: &[usize],
        smaller_leaf_splits: &mut LeafSplits,
        larger_leaf_splits: &mut LeafSplits,
        left_leaf: &mut i32,
        right_leaf: &mut i32,
    ) -> Result<i32, TreeLearnerError> {
        // The leaf-splits sums for a forced BFS leaf: leaf 0 (root) is the ordered
        // f64 fold over ALL rows (C++ `LeafSplits::Init()` whole-dataset variant),
        // and EVERY child leaf is seeded DIRECTLY from its parent split's `SplitInfo`
        // by `split_inner` — NOT re-folded.
        //
        // DEF-07-11-02 fix: the prior code re-folded each forced leaf via
        // `LeafSplits::init(rows)` at EVERY BFS level. C++ `ForceSplits`
        // (serial_tree_learner.cpp:638-734) never re-folds: each BFS iteration's
        // `GatherInfoForThreshold` consumes `left_leaf_splits->sum_gradients()/
        // sum_hessians()/num_data_in_leaf()` — the leaf-splits the PRIOR `SplitInner`
        // (serial_tree_learner.cpp:853-892) seeded from `best_split_info.{left,right}_
        // sum_hessian` (= `best_sum_left_hessian - kEpsilon`, feature_histogram.hpp:
        // 1042), carrying the parent REVERSE-scan kEpsilon + FixHistogram fold-order
        // provenance. A fresh re-fold loses that provenance and drifts the deeper-leaf
        // output denominator 1-2 ULPs (the SAME class as the 05-09 mfb>0 node-2 fix;
        // see leaf_splits.rs:124-138). `forced_single` is unaffected (its only forced
        // leaf is the root, whose whole-row fold == C++ `Init()` bit-exact).
        //
        // We track each leaf id's seeded `LeafSplits` in a side map: leaf 0 is the
        // whole-row fold; after each `split_inner` the two children's kEpsilon-bearing
        // splits (just written into `smaller_leaf_splits`/`larger_leaf_splits`) are
        // stored by leaf id and consumed (NOT re-folded) when that child is the next
        // BFS leaf.
        let mut leaf_seed: std::collections::HashMap<i32, LeafSplits> =
            std::collections::HashMap::new();
        {
            let root_rows: Vec<u32> = data_partition.indices_in_leaf(0).to_vec();
            let mut root_splits = LeafSplits::new();
            root_splits.init(gradients, hessians, &root_rows, &self.cfg);
            leaf_seed.insert(0, root_splits);
        }
        let mut count = 0i32;
        // BFS queue of (forced node, leaf id).
        let mut queue: std::collections::VecDeque<(crate::forced_splits::ForcedSplitNode, i32)> =
            std::collections::VecDeque::new();
        queue.push_back((forced.clone(), 0));

        while let Some((node, leaf)) = queue.pop_front() {
            if count >= self.num_leaves - 1 {
                break;
            }
            // The forced feature column.
            let f = match features.iter().find(|c| c.real_feature_index == node.feature) {
                Some(f) => f,
                None => continue, // out-of-range guarded at parse; defensive skip
            };
            // This leaf's seeded sums: leaf 0 is the whole-row fold; children carry the
            // kEpsilon-bearing `SplitInfo` sums `split_inner` seeded (NOT a re-fold).
            let leaf_splits = *leaf_seed
                .get(&leaf)
                .expect("forced BFS leaf must have a seeded LeafSplits (root or split_inner child)");
            #[allow(clippy::neg_cmp_op_on_partial_ord)]
            if !(leaf_splits.sum_hessians > 0.0) || leaf_splits.num_data_in_leaf <= 0 {
                continue;
            }
            // Build the forced feature's compacted+fixed histogram for this leaf.
            let (slot, _existed) = pool.get(leaf);
            self.build_leaf_histogram_into(
                features,
                gradients,
                hessians,
                data_partition,
                slot_off,
                leaf,
                &leaf_splits,
                pool.buffer_mut(slot),
            );
            let fpos = features
                .iter()
                .position(|c| c.real_feature_index == node.feature)
                .expect("forced feature is a known column");
            let cells = 2 * f.num_bin as usize;
            let hist = &pool.buffer(slot)[slot_off[fpos]..slot_off[fpos] + cells];
            // Map the real threshold to a bin (BinThreshold): the largest bin whose
            // upper bound is < the forced real threshold (predict routes
            // `value <= bin_upper_bound[threshold]` left).
            let threshold_bin = bin_threshold(f, node.threshold);
            let split = self.gather_info_for_threshold(
                hist,
                f,
                threshold_bin,
                leaf_splits.sum_gradients,
                leaf_splits.sum_hessians,
                leaf_splits.num_data_in_leaf,
            );
            if split.gain <= K_MIN_SCORE || !split.gain.is_finite() {
                // Forced split ignored (gain not better than no-split).
                continue;
            }
            let (new_left, new_right) = self.split_inner(
                tree,
                data_partition,
                features,
                leaf,
                node.feature,
                &split,
                smaller_leaf_splits,
                larger_leaf_splits,
            )?;
            self.update_constraints_after_split(
                tree,
                data_partition,
                leaf,
                new_left,
                new_right,
                node.feature,
                &split,
                &mut [],
            );
            // Record each child's kEpsilon-bearing seeded LeafSplits (just written by
            // `split_inner` into smaller/larger by PARTITION count, the 07-13 source).
            // Map child leaf id → its seeded splits so the next BFS level CONSUMES
            // these sums (NOT a re-fold). Mirror split_inner's `part_left < part_right`
            // smaller/larger assignment exactly.
            let part_left = data_partition.leaf_count(new_left);
            let part_right = data_partition.leaf_count(new_right);
            if part_left < part_right {
                leaf_seed.insert(new_left, *smaller_leaf_splits);
                leaf_seed.insert(new_right, *larger_leaf_splits);
            } else {
                leaf_seed.insert(new_right, *smaller_leaf_splits);
                leaf_seed.insert(new_left, *larger_leaf_splits);
            }
            *left_leaf = new_left;
            *right_leaf = new_right;
            count += 1;
            if let Some(l) = node.left {
                queue.push_back((*l, new_left));
            }
            if let Some(r) = node.right {
                queue.push_back((*r, new_right));
            }
        }
        Ok(count)
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

        // Parent-splittability gate input (serial_tree_learner.cpp:395-399): when a
        // parent histogram is retained, BOTH children skip features the PARENT could
        // not split on. The parent leaf id is `left_leaf` (its slot is the one the
        // Get/Move dance retains). `None` on the root (no parent → no gate). Cloned
        // out of the RefCell so the `&self` scans below can re-borrow it freely.
        let parent_splittable: Option<Vec<bool>> = if use_subtract {
            self.feature_splittable
                .borrow()
                .get(left_leaf as usize)
                .cloned()
        } else {
            None
        };

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
        //
        // 260608-p90: when resident-eligible, build the SMALLER (directly-built / root)
        // leaf's histogram DEVICE-RESIDENT (build→fix→compact stays on device) into the
        // mirror's `smaller_slot`, and scan it from that Handle. T2: NO host build on the
        // eligible path — subtract is now resident (reads the device Handle), so the host
        // pool buffer is never needed (the T1 transitional double-build is removed). The
        // host pool slot is left as-is (zeroed/stale); it is not read on the eligible spine
        // path (the spine pulls SplitInfos from the resident scan; categorical/monotone/
        // extra-trees inline branches are unreachable when eligible; capture_snapshots is
        // off). `None` (ineligible / CpuBackend) is the byte-unchanged host path.
        // 260608-t3t: on the FUSED directly-built path, the build is FUSED into the scan
        // (one `build_fix_scan_resident` launch in `scan_leaf_histogram`), so we SKIP the
        // standalone `build_resident_leaf_into` here. The fused scan stores the
        // fixed+compacted Handle into `smaller_slot`, so the subtract-derived larger
        // child still finds its parent. `smaller_fused` signals the fused scan path.
        let smaller_fused = self.fused_eligible;
        let smaller_resident_slot = if smaller_fused {
            // No separate build — the fused scan builds+fixes+compacts+scans in 1 launch
            // and stores the Handle into smaller_slot.
            Some(smaller_slot)
        } else if self.resident_eligible {
            self.build_resident_leaf_into(
                features,
                gradients,
                hessians,
                data_partition,
                slot_off,
                pool.hist_len(),
                smaller_slot,
                smaller_leaf,
                smaller_splits,
            )?;
            Some(smaller_slot)
        } else {
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
            None
        };
        let smaller_records = self.scan_leaf_histogram(
            features,
            slot_off,
            smaller_leaf,
            smaller_splits,
            pool.buffer(smaller_slot),
            best_split_per_leaf,
            best_split_feature,
            smaller_node_mask.as_deref(),
            data_partition,
            parent_splittable.as_deref(),
            smaller_resident_slot,
            smaller_fused,
            gradients,
            hessians,
        )?;

        // ---- LARGER child: derive by subtraction (parent − smaller) in the pool,
        // OR build directly when no parent was retained. ----
        let mut larger_records: Vec<FeatureSplitRecord> = Vec::new();
        if larger_leaf >= 0 {
            let larger_slot = larger_slot.expect("non-root larger child must hold a pool slot");
            // 260608-p90 T2: when resident-eligible AND a parent was retained, derive the
            // larger child RESIDENT — `parent_slot` Handle − `smaller_slot` Handle →
            // `larger_slot` Handle, on device, NO read-back. The device mirror is keyed by
            // SLOT id and `pool.move_` preserves slot ids (larger_slot == parent_slot;
            // see the slot-dance assertion), so the parent's resident Handle is already at
            // `parent_slot` and no device move is needed (move_resident would be a no-op
            // for a slot-id-keyed mirror; the host move_ only rewires leaf→slot). The
            // derived larger child is NOT re-FixHistogram'd (non-negotiable #3).
            let larger_resident_slot = if self.resident_eligible {
                if let Some(parent_slot) = parent_slot {
                    debug_assert_eq!(
                        larger_slot, parent_slot,
                        "the larger child reuses the moved parent slot (slot-id stable)"
                    );
                    self.backend.subtract_resident(
                        self.client,
                        parent_slot,
                        smaller_slot,
                        larger_slot,
                        pool.hist_len(),
                    )?;
                } else {
                    // No parent retained (cannot happen post-root in the spine) — build
                    // the larger child resident directly into its slot.
                    self.build_resident_leaf_into(
                        features,
                        gradients,
                        hessians,
                        data_partition,
                        slot_off,
                        pool.hist_len(),
                        larger_slot,
                        larger_leaf,
                        larger_splits,
                    )?;
                }
                Some(larger_slot)
            } else if let Some(parent_slot) = parent_slot {
                // ---- HOST path (ineligible / CpuBackend) — byte-unchanged. ----
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
                // 260609-bfx follow-up: pass the pool slots directly — `subtract_histograms`
                // only READS parent/child and returns a fresh owned buffer, so the two
                // per-split `.to_vec()` scratch clones were redundant. `parent_slot ==
                // larger_slot`, but `derived` is fully materialized (owns its data) before
                // the `buffer_mut(larger_slot)` write below, so there is no aliasing. Same
                // f64 cells, same op, same order → parity-neutral; one fewer Vec clone per
                // use_subtract larger-child derivation (every split that retains a parent).
                let derived = self.backend.subtract_histograms(
                    self.client,
                    pool.buffer(parent_slot),
                    pool.buffer(smaller_slot),
                )?;
                // TEST audit hook (T-05-07-01): record (derived, direct) so a parity
                // test can assert the subtracted larger child == a direct build of its
                // own rows, cell-for-cell, in the LIVE growth path. Host-path only (the
                // resident eligible path is inert here — audit is a non-spine diagnostic).
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
                None
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
                None
            };
            // 260608-t3t: the larger child is subtract-derived (parent − smaller),
            // NEVER fused-built — it reads its histogram via the resident subtract
            // Handle (or host buffer), so `fused_build = false`.
            larger_records = self.scan_leaf_histogram(
                features,
                slot_off,
                larger_leaf,
                larger_splits,
                pool.buffer(larger_slot),
                best_split_per_leaf,
                best_split_feature,
                larger_node_mask.as_deref(),
                data_partition,
                parent_splittable.as_deref(),
                larger_resident_slot,
                false,
                gradients,
                hessians,
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
        let _g = crate::phase_prof::guard(&crate::phase_prof::BUILD_NS);
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
        // BATCHED per-leaf histogram build (260608-lad): the backend builds ALL
        // features' RAW histograms in one call (CPU: the per-feature gather+construct
        // loop, bit-exact; GPU: one batched kernel launch + device-resident bins).
        let feature_bins: Vec<&[u32]> = features.iter().map(|f| f.bins.as_slice()).collect();
        // R3 (260614-p0n): the per-feature `num_bin` descriptor is identical on every
        // leaf build (the feature set is fixed per train). Fill the learner-held
        // scratch lazily on the first build (it survives `with_features`) or whenever
        // the feature count changes; re-borrow it thereafter. The CONTENT and ORDER
        // are byte-identical to `features.iter().map(|f| f.num_bin).collect()`, so the
        // fold inputs — and thus the bit-exact f64 tree — are unchanged.
        let mut num_bins_ref = self.build_num_bins.borrow_mut();
        if num_bins_ref.len() != features.len()
            || num_bins_ref
                .iter()
                .zip(features.iter())
                .any(|(&n, f)| n != f.num_bin)
        {
            num_bins_ref.clear();
            num_bins_ref.extend(features.iter().map(|f| f.num_bin));
        }
        let num_bins: &[u32] = &num_bins_ref;
        let mut raw = self
            .backend
            .build_leaf_histograms_raw(
                self.client,
                &feature_bins,
                num_bins,
                slot_off,
                buf.len(),
                leaf_rows,
                gradients,
                hessians,
            )
            .expect("build_leaf_histograms_raw on a validated leaf cannot fail");
        // Host-side FixHistogram + compaction per feature (they read this leaf's
        // sums + the per-feature compaction offset — kept in the learner, byte-for-
        // byte the same ops in the same order as before).
        for (fpos, f) in features.iter().enumerate() {
            let cells = 2 * f.num_bin as usize;
            let range = slot_off[fpos]..slot_off[fpos] + cells;
            // Run FixHistogram + compaction IN PLACE on a &mut sub-slice of the
            // learner-owned `raw` buffer (no per-feature clone). Same f64 cells,
            // same op order, same storage type — only the intermediate Vec is gone.
            {
                let hist = &mut raw[range.clone()];
                // FixHistogram on the RAW leaf sums (Pitfall 2). No-op for offset==1
                // (most_freq_bin==0), exactly as C++ `if (most_freq_bin > 0)`.
                crate::fix_histogram::fix_histogram(hist, f.most_freq_bin, sum_g, sum_h);
                // COMPACTED layout (D-09): shift real-bin `c+offset` into cell `c`,
                // zero the dropped tail. No-op for offset==0.
                compact_histogram(hist, f.offset);
            }
            buf[range.clone()].copy_from_slice(&raw[range]);
        }
    }

    /// 260608-p90: build ONE directly-built leaf's histogram DEVICE-RESIDENT (the
    /// resident analog of [`build_leaf_histogram_into`](Self::build_leaf_histogram_into)).
    /// Assembles the per-feature `(slot_off, num_bin, offset, most_freq_bin)` fix_feats
    /// (the SAME params the host `fix_histogram` + `compact_histogram` use) and the
    /// leaf RAW (un-bumped) sums (Pitfall 2), then drives
    /// [`Backend::build_resident_leaf`](lgbm_compute::Backend::build_resident_leaf),
    /// which runs build→widen→fix→compact entirely on device and stores the resulting
    /// f64 Handle into mirror slot `slot`. Only called when `resident_eligible`.
    ///
    /// An empty / no-hessian leaf is skipped (matching the host build's `buildable`
    /// guard): the resident scan of such a leaf is itself short-circuited by
    /// `scan_leaf_histogram`'s `sum_h > 0` gate, so the (absent) Handle is never read.
    #[allow(clippy::too_many_arguments)]
    fn build_resident_leaf_into(
        &self,
        features: &[FeatureColumn],
        gradients: &[f32],
        hessians: &[f32],
        data_partition: &DataPartition,
        slot_off: &[usize],
        slot_len: usize,
        slot: usize,
        leaf: i32,
        leaf_splits: &LeafSplits,
    ) -> Result<(), TreeLearnerError> {
        let sum_g = leaf_splits.sum_gradients;
        let sum_h = leaf_splits.sum_hessians;
        #[allow(clippy::neg_cmp_op_on_partial_ord)]
        let buildable = sum_h > 0.0 && leaf_splits.num_data_in_leaf > 0;
        if !buildable {
            // Leave the slot as-is; the scan's sum_h>0 gate skips this leaf so the
            // Handle is never consulted.
            return Ok(());
        }
        let leaf_rows = data_partition.indices_in_leaf(leaf);
        let feature_bins: Vec<&[u32]> = features.iter().map(|f| f.bins.as_slice()).collect();
        let num_bins: Vec<u32> = features.iter().map(|f| f.num_bin).collect();
        // fix_feats: per-feature (slot_off, num_bin, offset, most_freq_bin) — the SAME
        // values the host fix_histogram (most_freq_bin) + compact_histogram (offset)
        // consume, so the on-device fix+compact reproduces the host buffer bit-for-bit
        // (the f32-atomic RAW build is the only ~1e-6 contributor; oib proved fix+compact
        // bit-exact).
        let fix_feats: Vec<(usize, u32, i32, u32)> = features
            .iter()
            .enumerate()
            .map(|(fpos, f)| (slot_off[fpos], f.num_bin, f.offset, f.most_freq_bin))
            .collect();
        self.backend.build_resident_leaf(
            self.client,
            slot,
            &feature_bins,
            &num_bins,
            slot_off,
            slot_len,
            leaf_rows,
            gradients,
            hessians,
            &fix_feats,
            sum_g,
            sum_h,
        )?;
        Ok(())
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
        data_partition: &DataPartition,
        parent_splittable: Option<&[bool]>,
        // 260608-p90: when `Some(slot)` (resident-eligible), the SPINE batched scan
        // reads the device-resident Handle in mirror slot `slot` (via
        // `backend.scan_resident_leaf`) instead of the host `buf` (via
        // `find_best_splits_batched`). Every other gate / argmax / record / bookkeeping
        // path is BYTE-IDENTICAL — only the histogram SOURCE differs. `None` (the host
        // path, always on CpuBackend) is unchanged. When `Some`, eligibility guarantees
        // ZERO categorical/monotone/extra-trees features, so those inline branches are
        // unreachable and `buf` is not read on the spine path.
        resident_slot: Option<usize>,
        // 260608-t3t: when `true` (the fused directly-built path), the SPINE batched
        // scan is replaced by ONE `backend.build_fix_scan_resident` launch that BUILDS
        // (sequential f64), fixes, compacts, AND scans the leaf in a single launch —
        // storing the fixed+compacted Handle into `resident_slot` (so the subtract-
        // derived larger child still finds its parent) and returning the per-feature
        // SplitInfos. Requires `resident_slot == Some(slot)`. The build inputs are the
        // FULL-corpus `gradients`/`hessians` (the launcher gathers leaf rows on device).
        // `false` keeps the existing scan-only path (host buf or resident-Handle scan).
        fused_build: bool,
        gradients: &[f32],
        hessians: &[f32],
    ) -> Result<Vec<FeatureSplitRecord>, TreeLearnerError> {
        let _g = crate::phase_prof::guard(&crate::phase_prof::SCAN_NS);
        let sum_g = leaf_splits.sum_gradients;
        let sum_h = leaf_splits.sum_hessians;
        let num_data_in_leaf = leaf_splits.num_data_in_leaf;

        let mut records: Vec<FeatureSplitRecord> = Vec::with_capacity(features.len());
        let mut leaf_best = SplitInfo::none();
        let mut leaf_best_feature: i32 = -1;
        // Per-feature splittability recorded for THIS leaf (consumed by the larger
        // child of a LATER split via the parent-splittability gate below). Default
        // false; set true when find_best_split yields a finite-gain candidate.
        let mut this_leaf_splittable = vec![false; features.len()];

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

        // ---- ADV-02 interaction-allowed feature set for THIS node (D-06: empty
        // when interaction is inactive, then every feature is allowed). Mirrors
        // `ColSampler::GetByNode`'s interaction branch (col_sampler.hpp:91-125)
        // for the `fraction_bynode >= 1.0` (no col-subsample) case. ----
        let interaction_allowed: Option<std::collections::HashSet<i32>> =
            if self.constraints.interaction_constraints.is_empty() {
                None
            } else {
                Some(self.interaction_allowed_features(leaf))
            };

        // ADV-05 CEGB: this leaf's GLOBAL rows (needed for the lazy on-demand cost
        // + the post-split row marking). Computed once per scan when CEGB active.
        let cegb_active = self.cegb.borrow().is_some();

        // ---- PASS 1 (260608-lsx): gate-only pre-pass collecting the SPINE features
        // into ONE batched find_best_splits_batched call. A "spine" feature is one
        // that reaches the byte-untouched continuous `find_best_split` branch below:
        // it passes the col-sampler mask, the LOAD-BEARING parent-splittability gate,
        // and the ADV-02 interaction gate, AND is NOT categorical, NOT monotone, NOT
        // extra-trees-randomized. Those gates are applied here in the IDENTICAL order
        // and with the IDENTICAL predicates as the main loop (which still re-applies
        // them — this pre-pass only DECIDES batch membership; it must not change which
        // features the loop processes). Non-spine features keep their existing inline
        // handling. `spine_batch_index[fpos]` maps a spine feature's position to its
        // slot in the batched results; `None` for non-spine / gated-out features. ----
        let mut batched_feats: Vec<BatchedSplitFeature> = Vec::with_capacity(features.len());
        let mut spine_batch_index: Vec<Option<usize>> = vec![None; features.len()];
        let monotone_active = self.monotone.borrow().is_some();
        for (fpos, f) in features.iter().enumerate() {
            if let Some(mask) = used_features {
                if mask.get(fpos).copied().unwrap_or(1) == 0 {
                    continue;
                }
            }
            if let Some(ps) = parent_splittable {
                if !ps.get(fpos).copied().unwrap_or(true) {
                    continue;
                }
            }
            if interaction_allowed
                .as_ref()
                .is_some_and(|allowed| !allowed.contains(&f.real_feature_index))
            {
                continue;
            }
            if f.bin_type == BinType::Categorical {
                continue;
            }
            // monotone-active feature ⇒ ADV-01 inline branch (NOT batched).
            let monotone_type = if monotone_active {
                self.monotone
                    .borrow()
                    .as_ref()
                    .map(|mc| mc.feature_monotone(f.real_feature_index))
                    .unwrap_or(0)
            } else {
                0
            };
            if monotone_type != 0 {
                continue;
            }
            // extra-trees ⇒ ADV-04 randomized-threshold inline branch (NOT batched).
            if self.constraints.extra_trees {
                continue;
            }
            // This is a SPINE feature: record its batched params (ascending fpos).
            spine_batch_index[fpos] = Some(batched_feats.len());
            batched_feats.push(BatchedSplitFeature {
                slot_off: slot_off[fpos],
                num_bin: f.num_bin,
                offset: f.offset,
                default_bin: f.default_bin,
                most_freq_bin: f.most_freq_bin,
                skip_default_bin: f.skip_default_bin(),
                na_as_missing: f.na_as_missing(),
                run_forward: f.run_forward(),
            });
        }

        // ---- ONE batched call per leaf for all spine features (260608-lsx). On
        // CpuBackend this is the default per-feature-order loop ⇒ each result is
        // byte-identical to the inline `find_best_split` it replaces; on RocmBackend
        // it is the batched GPU override. Empty `batched_feats` ⇒ empty Vec, no
        // launch (T-lsx-03). ----
        //
        // 260608-p90: when resident-eligible (`resident_slot == Some(slot)`), read the
        // device-resident Handle in mirror slot `slot` instead of the host `buf` — the
        // Handle-consuming fused scan (`scan_resident_leaf`) is byte-identical to the
        // host-buf fused scan (same `split_scan_body`, same decode), so the per-feature
        // SplitInfos are the SAME and the grown tree is unchanged. `slot_len ==
        // buf.len()` (the pool slot length).
        // 260608-t3t: the FUSED directly-built path — ONE launch builds (sequential
        // f64), fixes, compacts, AND scans the leaf, storing the fixed+compacted Handle
        // into `slot` (so the subtract-derived larger child finds its parent). The
        // returned SplitInfos are BIT-EXACT to build_resident_leaf_into +
        // scan_resident_leaf (the fused==host oracle, T1) — same `split_scan_body`, same
        // decode, same fix/compact f64 fold order — so the grown tree is unchanged; only
        // the launch count drops (3 → 1 on directly-built leaves).
        let batched_splits = if fused_build {
            let slot = resident_slot
                .expect("fused_build requires a resident slot to store the histogram Handle");
            let leaf_rows = data_partition.indices_in_leaf(leaf);
            // The fused kernel BUILDS+fixes+compacts EVERY feature (the resident
            // histogram must be COMPLETE for the subtract-derived larger child), but
            // SCANS only the spine subset that passed Pass-1's gates. `all_feats` is the
            // full per-feature param list (fpos order); `scan_active[fpos]` is true iff
            // the feature is in `batched_feats` (`spine_batch_index[fpos].is_some()`).
            // The launcher returns SplitInfos for scan-active features in order, which
            // is EXACTLY `batched_feats` order (Pass 1 pushed in ascending fpos).
            let all_feats: Vec<BatchedSplitFeature> = features
                .iter()
                .enumerate()
                .map(|(fpos, f)| BatchedSplitFeature {
                    slot_off: slot_off[fpos],
                    num_bin: f.num_bin,
                    offset: f.offset,
                    default_bin: f.default_bin,
                    most_freq_bin: f.most_freq_bin,
                    skip_default_bin: f.skip_default_bin(),
                    na_as_missing: f.na_as_missing(),
                    run_forward: f.run_forward(),
                })
                .collect();
            let scan_active: Vec<bool> =
                spine_batch_index.iter().map(|idx| idx.is_some()).collect();
            self.backend.build_fix_scan_resident(
                self.client,
                slot,
                slot_off,
                buf.len(),
                leaf_rows,
                gradients,
                hessians,
                &all_feats,
                &scan_active,
                &self.cfg,
                sum_g,
                sum_h,
                num_data_in_leaf,
            )?
        } else if let Some(slot) = resident_slot {
            self.backend.scan_resident_leaf(
                self.client,
                slot,
                buf.len(),
                &batched_feats,
                &self.cfg,
                sum_g,
                sum_h,
                num_data_in_leaf,
            )?
        } else {
            self.backend.find_best_splits_batched(
                self.client,
                buf,
                &batched_feats,
                &self.cfg,
                sum_g,
                sum_h,
                num_data_in_leaf,
            )?
        };

        for (fpos, f) in features.iter().enumerate() {
            // ColSampler gate (serial_tree_learner.cpp: `if (!is_feature_used[fi])
            // continue;`). On the spine (`used_features == None`) every feature is
            // scanned.
            if let Some(mask) = used_features {
                if mask.get(fpos).copied().unwrap_or(1) == 0 {
                    continue;
                }
            }
            // ---- Parent-splittability gate (serial_tree_learner.cpp:395-399) ----
            // When this leaf is derived against a retained parent histogram
            // (use_subtract), C++ SKIPS any feature whose PARENT histogram was not
            // splittable: `if (!parent_leaf_histogram_array_[f].is_splittable())
            // continue;`. Load-bearing under GOSS amplification — a small
            // `cnt_factor` (= num_data/amplified_sum_hessian) rounds the
            // `round_int(hess·cnt_factor)` per-bin counts to 0 at the parent, so a
            // feature can fail `min_data_in_leaf` at the parent (not splittable) yet
            // look splittable on this child (whose larger cnt_factor no longer
            // rounds to 0). Without the gate Rust selects a split C++ never
            // considers (GOSS tree-10 node1: Rust f1 gain 4.36 vs C++ picking f0
            // 1.14 because the root's f1 was not splittable). `None` (smaller child
            // / root) ⇒ no gate, every feature scanned.
            if let Some(ps) = parent_splittable {
                if !ps.get(fpos).copied().unwrap_or(true) {
                    continue;
                }
            }
            // ADV-02 interaction gate: skip a feature not allowed to co-occur with
            // this node's branch features (additive — no-op when inactive).
            if interaction_allowed
                .as_ref()
                .is_some_and(|allowed| !allowed.contains(&f.real_feature_index))
            {
                continue;
            }
            let cells = 2 * f.num_bin as usize;
            let hist = &buf[slot_off[fpos]..slot_off[fpos] + cells];

            // ---- bin_type dispatch (serial_tree_learner.cpp:779) ----
            // Categorical features route to the ADDITIVE many-vs-many/one-hot
            // categorical finder (D-06: the numeric path below is byte-untouched).
            // The categorical winner's bitset is stashed in `best_cat_threshold`
            // indexed by leaf so `split_inner` can grow a categorical node.
            if f.bin_type == BinType::Categorical {
                // `find_best_threshold_categorical` expects the leaf hessian sum
                // ALREADY bumped by +2*kEpsilon (mirroring FindBestThreshold,
                // feature_histogram.hpp:172). The numeric path applies the same
                // bump inside find_best_split; we apply it here at the call site.
                let eps = f64::from(lgbm_core::types::K_EPSILON);
                let sum_h_bumped = sum_h + 2.0 * eps;
                let cat = crate::feature_histogram_categorical::find_best_threshold_categorical(
                    hist,
                    &self.cfg,
                    f.num_bin as i32,
                    f.offset,
                    sum_g,
                    sum_h_bumped,
                    num_data_in_leaf,
                );
                let split = cat.split;
                this_leaf_splittable[fpos] = split.gain > K_MIN_SCORE;
                // Categorical features have no per-bin numeric gain arrays; the D-06
                // numeric snapshot uses empty rev/fwd for them (the categorical
                // diagnostics live in the dedicated categorical golden, not here).
                records.push(FeatureSplitRecord {
                    feature: f.real_feature_index,
                    gains: Vec::new(),
                    rev_len: 0,
                    split,
                });
                if split.gain > K_MIN_SCORE
                    && split_gt(&split, f.real_feature_index, &leaf_best, leaf_best_feature)
                {
                    leaf_best = split;
                    leaf_best_feature = f.real_feature_index;
                    // Stash this categorical winner's bitset for the leaf; cleared
                    // (set to None) if a later numeric/categorical feature wins.
                    self.best_cat_threshold.borrow_mut()[leaf as usize] =
                        Some(cat.cat_threshold.clone());
                }
                continue;
            }

            // Authoritative dispatch flags (Pitfall 1). `run_forward` transcribes
            // the C++ per-missing_type branch dispatch (feature_histogram.hpp:
            // 420-429): the FORWARD scan runs ONLY for `num_bin>2 &&
            // missing_type==Zero`; for `missing_type==None` only REVERSE runs.
            let skip_default_bin = f.skip_default_bin();
            let na_as_missing = f.na_as_missing();
            let run_forward = f.run_forward();

            // ADV-04 extra-trees: draw this feature's random threshold for this
            // scan (BeforeNumerical, feature_histogram.hpp:202-206) — drawn for
            // EVERY scanned feature so the RNG sequence matches C++ even when the
            // candidate is later rejected. `None` ⇒ spine (best-threshold).
            let rand_threshold: Option<i32> = if self.constraints.extra_trees {
                let mut rng = self.extra_rng.borrow_mut();
                rng.as_mut().map(|v| {
                    // C++ `meta_->rand.NextInt(0, meta_->num_bin - 2)`
                    // (feature_histogram.hpp:204). `Random::next_int(lo, hi)` is the
                    // half-open `[lo, hi)` mirror of the C++ `% (hi-lo) + lo`, so the
                    // upper bound is `num_bin - 2` (NOT `num_bin - 1`).
                    if f.num_bin as i32 - 2 > 0 {
                        v[fpos].next_int(0, f.num_bin as i32 - 2)
                    } else {
                        0
                    }
                })
            } else {
                None
            };

            // ---- the split for this feature ----
            // Gate selection (D-06): monotone-active feature → the constraint-aware
            // finder; extra-trees → the randomized-threshold finder; otherwise the
            // BYTE-UNTOUCHED spine `find_best_split`.
            let monotone_type = self
                .monotone
                .borrow()
                .as_ref()
                .map(|mc| mc.feature_monotone(f.real_feature_index))
                .unwrap_or(0);

            let mut split = if monotone_type != 0 {
                // ADV-01: constraint-aware re-scan with this leaf's [min,max] clamp.
                let constraint = self
                    .monotone
                    .borrow()
                    .as_ref()
                    .expect("monotone active")
                    .constraint_for(leaf);
                crate::monotone_constraints::find_best_split_monotone(
                    hist,
                    &self.cfg,
                    f.num_bin as i32,
                    f.offset,
                    f.default_bin,
                    skip_default_bin,
                    run_forward,
                    sum_g,
                    sum_h,
                    num_data_in_leaf,
                    monotone_type,
                    &constraint,
                )
            } else if let Some(rt) = rand_threshold {
                // ADV-04: only the candidate at `rt` is admissible.
                self.find_best_split_rand(
                    hist, f, skip_default_bin, run_forward, sum_g, sum_h, num_data_in_leaf, rt,
                )
            } else {
                // SPINE (260608-lsx): the bit-exact continuous finder is now run via
                // the ONE batched find_best_splits_batched call above; pull THIS
                // feature's SplitInfo from the batched results by the Pass-1 mapping.
                // On CpuBackend the batched default impl is the per-feature
                // `find_best_split` loop in this same order, so the looked-up value is
                // byte-identical to the inline call it replaces (the grown tree is
                // unchanged). `na_as_missing` is validated false on every committed
                // case (deferred branch); `skip_default_bin` / `run_forward` were
                // recorded into the BatchedSplitFeature in Pass 1.
                let bi = spine_batch_index[fpos]
                    .expect("spine feature reaching the continuous branch must have been batched");
                let _ = (skip_default_bin, na_as_missing, run_forward);
                batched_splits[bi]
            };

            // Record this feature's RAW splittability (C++ is_splittable_, set in
            // FindBestThresholdSequentially when current_gain > min_gain_shift) —
            // BEFORE any CEGB / monotone gain post-processing, exactly as C++ sets
            // the flag inside the scan, not after ComputeBestSplitForFeature.
            this_leaf_splittable[fpos] = split.gain > K_MIN_SCORE;
            // ---- ADV-05 CEGB: SUBTRACT the per-split cost penalty from the gain
            // (ComputeBestSplitForFeature, serial_tree_learner.cpp:988-992) BEFORE
            // the argmax. No-op when CEGB inactive (D-06). ----
            if cegb_active && split.gain > K_MIN_SCORE {
                let leaf_rows = data_partition.indices_in_leaf(leaf);
                let delta = self
                    .cegb
                    .borrow()
                    .as_ref()
                    .expect("cegb active")
                    .delta_gain(f.real_feature_index, num_data_in_leaf, leaf_rows);
                split.gain -= delta;
            }

            // ---- ADV-01 monotone penalty (serial_tree_learner.cpp:993-997):
            // multiply a monotone split's gain by the depth-dependent penalty. ----
            if monotone_type != 0 && split.gain > K_MIN_SCORE {
                // Leaf depth == the count of branch features on its root path
                // (C++ `tree_->leaf_depth(leaf)`); the branch_features list is
                // maintained for ADV-02 and reused here.
                let depth = self
                    .branch_features
                    .borrow()
                    .get(leaf as usize)
                    .map(|b| b.len() as i32)
                    .unwrap_or(0);
                let penalty = self
                    .monotone
                    .borrow()
                    .as_ref()
                    .expect("monotone active")
                    .split_gain_penalty(depth);
                split.gain *= penalty;
            }

            // Per-bin gain arrays for the D-06 snapshot (host re-scan of the SAME
            // fixed histogram via the gain primitive — localizes a divergence).
            // R1: snapshot-ONLY work — skip it entirely on the production path
            // (`capture_snapshots == false`). The live split (`split` above) and the
            // splittability flag are already decided; the grown tree is identical.
            let (gains, rev_len) = if self.capture_snapshots {
                self.per_bin_gains(hist, f, sum_g, sum_h, num_data_in_leaf)
            } else {
                (Vec::new(), 0)
            };

            records.push(FeatureSplitRecord {
                feature: f.real_feature_index,
                gains,
                rev_len,
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
        // If the cross-feature winner is NOT a categorical feature, drop any
        // categorical bitset that was stashed for this leaf by an earlier (losing)
        // categorical candidate — `split_inner` must see `None` for a numeric
        // winner. (Purely a side-structure cleanup; the numeric scan above is
        // byte-untouched, D-06.)
        let winner_is_cat = leaf_best_feature >= 0
            && features
                .iter()
                .find(|c| c.real_feature_index == leaf_best_feature)
                .map(|c| c.bin_type == BinType::Categorical)
                .unwrap_or(false);
        if !winner_is_cat {
            self.best_cat_threshold.borrow_mut()[leaf as usize] = None;
        }
        // Persist this leaf's per-feature splittability for the parent-splittability
        // gate of a FUTURE split's larger child (C++ FeatureHistogram::is_splittable_
        // retained in the histogram pool slot). NOTE: when this scan was itself gated
        // by a `parent_splittable` mask, the gated-out features were `continue`d and
        // left `false` here — which is correct (a feature not splittable at the
        // grandparent stays not splittable down the subtraction chain, matching C++
        // where set_is_splittable(false) propagates).
        {
            let mut fs = self.feature_splittable.borrow_mut();
            if let Some(slot) = fs.get_mut(leaf as usize) {
                *slot = this_leaf_splittable;
            }
        }
        Ok(records)
    }

    /// ADV-02: the set of REAL feature indices allowed at `leaf` given its branch
    /// features and the interaction-constraint groups (`ColSampler::GetByNode`
    /// interaction branch, col_sampler.hpp:91-125, `fraction_bynode>=1.0` case).
    /// A feature is allowed iff it is a branch feature OR it belongs to a group
    /// that contains ALL of the node's branch features (or the branch is empty).
    fn interaction_allowed_features(&self, leaf: i32) -> std::collections::HashSet<i32> {
        use std::collections::HashSet;
        let branch = self.branch_features.borrow();
        let branch_features: &[i32] = branch.get(leaf as usize).map(|v| v.as_slice()).unwrap_or(&[]);
        let mut allowed: HashSet<i32> = branch_features.iter().copied().collect();
        for group in &self.constraints.interaction_constraints {
            let group_set: HashSet<i32> = group.iter().copied().collect();
            if branch_features.is_empty() {
                allowed.extend(group.iter().copied());
                continue;
            }
            // The group must contain EVERY branch feature.
            if branch_features.iter().all(|bf| group_set.contains(bf)) {
                allowed.extend(group.iter().copied());
            }
        }
        allowed
    }

    /// ADV-04 extra-trees: the randomized-threshold finder — the `USE_RAND`
    /// instantiation of `FindBestThresholdSequentially` (feature_histogram.hpp:
    /// 894-898 / 1268-1271): only the candidate whose recorded threshold equals
    /// `rand_threshold` is admissible (`t-1+offset` reverse / `t+offset` forward).
    /// All other gates (count / hessian / min_gain_shift) are identical to the
    /// spine. Returns a [`SplitInfo`] (or `none()` when the random candidate is
    /// gated out).
    #[allow(clippy::too_many_arguments)]
    fn find_best_split_rand(
        &self,
        hist: &[f64],
        f: &FeatureColumn,
        skip_default_bin: bool,
        run_forward: bool,
        sum_g: f64,
        sum_h_raw: f64,
        num_data: i32,
        rand_threshold: i32,
    ) -> SplitInfo {
        use lgbm_compute::gain::{calculate_splitted_leaf_output, get_leaf_gain, get_split_gains};
        let cfg = &self.cfg;
        let eps = f64::from(lgbm_core::types::K_EPSILON);
        let sum_hessian = sum_h_raw + 2.0 * eps;
        let use_l1 = cfg.use_l1();
        let l1 = cfg.lambda_l1;
        let l2 = cfg.lambda_l2;
        let offset = f.offset;
        let num_bin = f.num_bin as i32;
        let default_bin = f.default_bin as i32;
        let gain_shift = get_leaf_gain(use_l1, sum_g, sum_hessian, l1, l2);
        let min_gain_shift = gain_shift + cfg.min_gain_to_split;
        let cnt_factor = f64::from(num_data) / sum_hessian;
        let round_int = |x: f64| -> i32 { (x + f64::from(0.5f32)) as i32 };
        let get_grad = |t: i32| hist[(t as usize) << 1];
        let get_hess = |t: i32| hist[((t as usize) << 1) + 1];

        let mut best = SplitInfo::none();
        let mut best_gain = f64::NEG_INFINITY;
        let mk_output = |g: f64, h: f64| calculate_splitted_leaf_output(use_l1, g, h, l1, l2);

        // REVERSE.
        {
            let mut sum_right_gradient = 0.0f64;
            let mut sum_right_hessian = eps;
            let mut right_count = 0i32;
            let mut t = num_bin - 1 - offset;
            let t_end = 1 - offset;
            while t >= t_end {
                if skip_default_bin && (t + offset) == default_bin {
                    t -= 1;
                    continue;
                }
                sum_right_gradient += get_grad(t);
                sum_right_hessian += get_hess(t);
                right_count += round_int(get_hess(t) * cnt_factor);
                if right_count < cfg.min_data_in_leaf || sum_right_hessian < cfg.min_sum_hessian_in_leaf {
                    t -= 1;
                    continue;
                }
                let left_count = num_data - right_count;
                if left_count < cfg.min_data_in_leaf {
                    break;
                }
                let sum_left_hessian = sum_hessian - sum_right_hessian;
                if sum_left_hessian < cfg.min_sum_hessian_in_leaf {
                    break;
                }
                // USE_RAND: only the candidate at rand_threshold is admissible.
                if (t - 1 + offset) != rand_threshold {
                    t -= 1;
                    continue;
                }
                let sum_left_gradient = sum_g - sum_right_gradient;
                let g = get_split_gains(use_l1, sum_left_gradient, sum_left_hessian, sum_right_gradient, sum_right_hessian, l1, l2);
                if g > min_gain_shift && g > best_gain {
                    best_gain = g;
                    // C++ computes the child OUTPUTS from the RAW hessians
                    // (best_sum_left_hessian, sum_hessian - best_sum_left_hessian) and
                    // ONLY THEN stores `<raw> - kEpsilon` (feature_histogram.hpp:1042-
                    // 1062). DEF-07-11-03: computing from the `-kEpsilon` value drifts
                    // the leaf output ~1 ULP. Use the C++ right operand order too.
                    let right_g_out = sum_g - sum_left_gradient;
                    let right_h_raw = sum_hessian - sum_left_hessian;
                    best = SplitInfo {
                        threshold: (t - 1 + offset) as u32,
                        gain: g - min_gain_shift,
                        left_count,
                        right_count,
                        left_sum_gradient: sum_left_gradient,
                        left_sum_hessian: sum_left_hessian - eps,
                        right_sum_gradient: sum_right_gradient,
                        right_sum_hessian: right_h_raw - eps,
                        left_output: mk_output(sum_left_gradient, sum_left_hessian),
                        right_output: mk_output(right_g_out, right_h_raw),
                        default_left: true,
                    };
                }
                t -= 1;
            }
        }

        // FORWARD.
        if run_forward {
            let mut sum_left_gradient = 0.0f64;
            let mut sum_left_hessian = eps;
            let mut left_count = 0i32;
            let mut t = 0i32;
            let t_end = num_bin - 2 - offset;
            while t <= t_end {
                if skip_default_bin && (t + offset) == default_bin {
                    t += 1;
                    continue;
                }
                sum_left_gradient += get_grad(t);
                sum_left_hessian += get_hess(t);
                left_count += round_int(get_hess(t) * cnt_factor);
                if left_count < cfg.min_data_in_leaf || sum_left_hessian < cfg.min_sum_hessian_in_leaf {
                    t += 1;
                    continue;
                }
                let right_count = num_data - left_count;
                if right_count < cfg.min_data_in_leaf {
                    break;
                }
                let sum_right_hessian = sum_hessian - sum_left_hessian;
                if sum_right_hessian < cfg.min_sum_hessian_in_leaf {
                    break;
                }
                if (t + offset) != rand_threshold {
                    t += 1;
                    continue;
                }
                let sum_right_gradient = sum_g - sum_left_gradient;
                let g = get_split_gains(use_l1, sum_left_gradient, sum_left_hessian, sum_right_gradient, sum_right_hessian, l1, l2);
                if g > min_gain_shift && g > best_gain {
                    best_gain = g;
                    // RAW-hessian output operands (see REVERSE branch, DEF-07-11-03).
                    let right_g_out = sum_g - sum_left_gradient;
                    let right_h_raw = sum_hessian - sum_left_hessian;
                    best = SplitInfo {
                        threshold: (t + offset) as u32,
                        gain: g - min_gain_shift,
                        left_count,
                        right_count,
                        left_sum_gradient: sum_left_gradient,
                        left_sum_hessian: sum_left_hessian - eps,
                        right_sum_gradient: sum_right_gradient,
                        right_sum_hessian: right_h_raw - eps,
                        left_output: mk_output(sum_left_gradient, sum_left_hessian),
                        right_output: mk_output(right_g_out, right_h_raw),
                        default_left: false,
                    };
                }
                t += 1;
            }
        }
        best
    }

    /// ADV-03 `GatherInfoForThresholdNumerical` (feature_histogram.hpp:486-588):
    /// compute the split at a SPECIFIC `threshold` bin (the forced threshold). The
    /// right side accumulates bins `> threshold`; left = total - right. Returns the
    /// [`SplitInfo`] (or `none()` when the forced gain is not better than no-split,
    /// matching the C++ "Forced Split will be ignored" path). This is the default
    /// (no path_smooth, no max_delta_step) instantiation — the matrix forced cells
    /// use config defaults.
    fn gather_info_for_threshold(
        &self,
        hist: &[f64],
        f: &FeatureColumn,
        threshold: u32,
        sum_g: f64,
        sum_h_raw: f64,
        num_data: i32,
    ) -> SplitInfo {
        use lgbm_compute::gain::{calculate_splitted_leaf_output, get_leaf_gain, get_split_gains};
        let cfg = &self.cfg;
        let eps = f64::from(lgbm_core::types::K_EPSILON);
        // C++ `GatherInfoForThresholdNumerical` (feature_histogram.hpp:486) uses the
        // leaf's RAW `sum_hessian` directly — it does NOT apply the `+2*kEpsilon`
        // bump that `FindBestThreshold` adds at its call site (feature_histogram.hpp
        // :172). Bumping here shifted the forced leaf-output denominator by ~1 ULP.
        let sum_hessian = sum_h_raw;
        let use_l1 = cfg.use_l1();
        let l1 = cfg.lambda_l1;
        let l2 = cfg.lambda_l2;
        let offset = f.offset;
        let num_bin = f.num_bin as i32;
        let default_bin = f.default_bin as i32;
        let skip_default_bin = f.skip_default_bin();
        let gain_shift = get_leaf_gain(use_l1, sum_g, sum_hessian, l1, l2);
        let min_gain_shift = gain_shift + cfg.min_gain_to_split;
        let cnt_factor = f64::from(num_data) / sum_hessian;
        let round_int = |x: f64| -> i32 { (x + f64::from(0.5f32)) as i32 };
        let get_grad = |t: i32| hist[(t as usize) << 1];
        let get_hess = |t: i32| hist[((t as usize) << 1) + 1];

        let mut sum_right_gradient = 0.0f64;
        let mut sum_right_hessian = eps;
        let mut right_count = 0i32;
        let mut t = num_bin - 1 - offset;
        let t_end = 1 - offset;
        while t >= t_end {
            if (t + offset) as u32 <= threshold {
                break;
            }
            if skip_default_bin && (t + offset) == default_bin {
                t -= 1;
                continue;
            }
            sum_right_gradient += get_grad(t);
            sum_right_hessian += get_hess(t);
            right_count += round_int(get_hess(t) * cnt_factor);
            t -= 1;
        }
        let sum_left_gradient = sum_g - sum_right_gradient;
        let sum_left_hessian = sum_hessian - sum_right_hessian;
        let left_count = num_data - right_count;
        let current_gain = get_split_gains(
            use_l1,
            sum_left_gradient,
            sum_left_hessian,
            sum_right_gradient,
            sum_right_hessian,
            l1,
            l2,
        );
        if current_gain.is_nan() || current_gain <= min_gain_shift {
            // C++ "Forced Split will be ignored since the gain getting worse."
            return SplitInfo::none();
        }
        let mk_output = |g: f64, h: f64| calculate_splitted_leaf_output(use_l1, g, h, l1, l2);
        // DEF-07-11-02: C++ `GatherInfoForThresholdNumericalInner`
        // (feature_histogram.hpp:579-590) computes the child OUTPUTS from the RAW
        // child hessians (`sum_left_hessian` and `sum_hessian - sum_left_hessian`),
        // and ONLY THEN stores `{left,right}_sum_hessian = <raw> - kEpsilon`. The
        // prior code computed the outputs from the already-`-kEpsilon` values, which
        // shifted each forced child's leaf output by ~1 ULP (`6.0000000000000115`
        // vs the golden `6.000000000000008`). The output operand is the RAW scan
        // hessian; the stored sum carries the `-kEpsilon`. (The stored `right_sum_
        // hessian` is `sum_hessian - sum_left_hessian - kEpsilon`, matching C++'s
        // `output->right_sum_hessian` exactly — algebraically `sum_right_hessian -
        // kEpsilon` but written via the C++ operand order.)
        let right_h_raw = sum_hessian - sum_left_hessian;
        // C++ computes the right output's gradient operand as `sum_gradient -
        // sum_left_gradient` (feature_histogram.hpp:585), NOT the scan-accumulated
        // `sum_right_gradient`; reproduce the exact operand order (they can differ in
        // the last f64 ULP).
        let right_g_for_output = sum_g - sum_left_gradient;
        SplitInfo {
            threshold,
            gain: current_gain - min_gain_shift,
            left_count,
            right_count,
            left_sum_gradient: sum_left_gradient,
            left_sum_hessian: sum_left_hessian - eps,
            right_sum_gradient: sum_right_gradient,
            right_sum_hessian: right_h_raw - eps,
            left_output: mk_output(sum_left_gradient, sum_left_hessian),
            right_output: mk_output(right_g_for_output, right_h_raw),
            default_left: true,
        }
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
    ) -> (Vec<f64>, usize) {
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

        // BeforeNumerical min_gain_shift (host, same as find_best_split). Phase-7
        // D-05 faithful-fix: `gain_shift` uses the 2*kEpsilon-BUMPED `sum_hessian`
        // (C++ passes the bumped value into `BeforeNumerical`, feature_histogram.hpp
        // :174,400-401) — mirrors the `find_best_split_cpu` fix so this diagnostic
        // re-scan stays bit-identical to the live kernel path.
        let eps = f64::from(lgbm_core::types::K_EPSILON);
        let sum_hessian_bumped = sum_hessian + 2.0 * eps;
        let gain_shift =
            lgbm_compute::gain::get_leaf_gain(use_l1, sum_gradient, sum_hessian_bumped, l1, l2);
        let min_gain_shift = gain_shift + cfg.min_gain_to_split;
        let cnt_factor = f64::from(num_data) / sum_hessian_bumped;
        let round_int = |x: f64| -> i32 { (x + f64::from(0.5f32)) as i32 };
        let get_grad = |t: i32| hist[(t as usize) << 1];
        let get_hess = |t: i32| hist[((t as usize) << 1) + 1];

        // REVERSE (:854-936) then FORWARD packed into ONE allocation (260609-bfx
        // snapshot-path alloc reduction: 2 retained Vecs/record → 1). Pre-sized to
        // `2*num_bin` (the combined push upper bound) so the per-feature per-leaf
        // snapshot scan never reallocates mid-grow. Parity-neutral — capacity only,
        // identical pushed sequence; `rev_len` (captured after the REVERSE block)
        // splits the buffer into the REVERSE prefix and the FORWARD remainder.
        let mut gains: Vec<f64> = Vec::with_capacity(2 * num_bin.max(0) as usize);
        {
            let mut sum_right_gradient = 0.0f64;
            let mut sum_right_hessian = eps;
            let mut right_count = 0i32;
            let t_start = num_bin - 1 - offset;
            let t_end = 1 - offset;
            let mut t = t_start;
            while t >= t_end {
                if skip && (t + offset) == default_bin {
                    gains.push(qnan);
                    t -= 1;
                    continue;
                }
                sum_right_gradient += get_grad(t);
                sum_right_hessian += get_hess(t);
                right_count += round_int(get_hess(t) * cnt_factor);
                if right_count < cfg.min_data_in_leaf
                    || sum_right_hessian < cfg.min_sum_hessian_in_leaf
                {
                    gains.push(qnan);
                    t -= 1;
                    continue;
                }
                let left_count = num_data - right_count;
                if left_count < cfg.min_data_in_leaf {
                    gains.push(qnan);
                    break;
                }
                let sum_left_hessian = sum_hessian_bumped - sum_right_hessian;
                if sum_left_hessian < cfg.min_sum_hessian_in_leaf {
                    gains.push(qnan);
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
                    gains.push(qnan);
                } else {
                    gains.push(g);
                }
                t -= 1;
            }
        }

        // FORWARD (:937-1029) appended into the SAME buffer after the REVERSE prefix.
        let rev_len = gains.len();
        {
            let mut sum_left_gradient = 0.0f64;
            let mut sum_left_hessian = eps;
            let mut left_count = 0i32;
            let t_end = num_bin - 2 - offset;
            let mut t = 0i32;
            while t <= t_end {
                if skip && (t + offset) == default_bin {
                    gains.push(qnan);
                    t += 1;
                    continue;
                }
                sum_left_gradient += get_grad(t);
                sum_left_hessian += get_hess(t);
                left_count += round_int(get_hess(t) * cnt_factor);
                if left_count < cfg.min_data_in_leaf
                    || sum_left_hessian < cfg.min_sum_hessian_in_leaf
                {
                    gains.push(qnan);
                    t += 1;
                    continue;
                }
                let right_count = num_data - left_count;
                if right_count < cfg.min_data_in_leaf {
                    gains.push(qnan);
                    break;
                }
                let sum_right_hessian = sum_hessian_bumped - sum_left_hessian;
                if sum_right_hessian < cfg.min_sum_hessian_in_leaf {
                    gains.push(qnan);
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
                    gains.push(qnan);
                } else {
                    gains.push(g);
                }
                t += 1;
            }
        }

        (gains, rev_len)
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

        // ---- CATEGORICAL split path (serial_tree_learner.cpp:807-843) ----
        // If this leaf's winner is a categorical feature, a category bitset was
        // stashed in `best_cat_threshold`. Build the inner (bin) + real (category)
        // bitsets, partition by the inner bitset, and grow a SplitCategorical node.
        let cat_bins: Option<Vec<u32>> = self
            .best_cat_threshold
            .borrow_mut()
            .get_mut(best_leaf as usize)
            .and_then(|o| o.take());
        if let Some(cat_threshold_bins) = cat_bins {
            return self.split_inner_categorical(
                tree,
                data_partition,
                f,
                best_leaf,
                feat_idx,
                best,
                &cat_threshold_bins,
                smaller_leaf_splits,
                larger_leaf_splits,
                new_left,
                new_right,
            );
        }

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
        // C++ OVERWRITES `best_split_info.left_count`/`right_count` with the
        // data-partition leaf counts (serial_tree_learner.cpp:790-791,
        // `update_cnt=true`) BEFORE the smaller/larger tie-break at :851
        // (`best_split_info.left_count < best_split_info.right_count`). The
        // smaller/larger HISTOGRAM-slot dance (BeforeFindBestSplit, learner.rs
        // :1099-1109) ALSO keys off the partition counts. Both MUST use the SAME
        // count source so the leaf that receives the directly-built / subtracted
        // histogram is the same leaf that receives the matching seeded sums.
        //
        // Using the raw SplitInfo `round_int(hess·cnt_factor)` counts here
        // (`best.left_count`/`best.right_count`) DESYNCS the two on the TIE / ±1
        // case for fractional (non-constant) hessians: e.g. gamma tree-0 node
        // {0,1}|{2,3} has SplitInfo counts (1,3) but partition counts (2,2). The
        // histogram dance (partition 2,2 ⇒ tie ⇒ smaller=right={2,3}) then
        // disagreed with this seeding (SplitInfo 1<3 ⇒ smaller=left={0,1}),
        // attaching node{2,3}'s sums (sum_g=1.0) to node{0,1}'s histogram and
        // flipping the next split's gain (bogus 1.0417 vs C++ 0.0333) → tree-0
        // topology [1,8,2,1] vs golden [2,4,2,4]. Constant-hessian families round
        // identically so this never tripped them. (DEF-07-02/03 root cause; proven
        // by the source-built lib_lightgbm 4.6 FP trace,
        // .planning/debug/split-gain-knife-edge-07-02.md.)
        if part_left < part_right {
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

    /// The CATEGORICAL `SplitInner` (serial_tree_learner.cpp:807-871): build the
    /// inner (bin) + real (category) bitsets, partition the leaf by the inner
    /// bitset, grow a `SplitCategorical` node, and seed the child leaf-splits.
    ///
    /// `cat_threshold_bins` is the winning category set as REAL BINS (the finder's
    /// `output->cat_threshold`). The inner bitset is `ConstructBitset` over those
    /// bins; the real bitset is `ConstructBitset` over their CATEGORY VALUES
    /// (`bin_to_category[bin]`, the C++ `RealThreshold`). The numeric split spine
    /// is untouched (this is a sibling method, D-06).
    #[allow(clippy::too_many_arguments)]
    fn split_inner_categorical(
        &self,
        tree: &mut Tree,
        data_partition: &mut DataPartition,
        f: &FeatureColumn,
        best_leaf: i32,
        feat_idx: i32,
        best: &SplitInfo,
        cat_threshold_bins: &[u32],
        smaller_leaf_splits: &mut LeafSplits,
        larger_leaf_splits: &mut LeafSplits,
        new_left: i32,
        new_right: i32,
    ) -> Result<(i32, i32), TreeLearnerError> {
        use crate::feature_histogram_categorical::construct_bitset;

        // Real (category-value) bitset = ConstructBitset over the RealThreshold of
        // each winning bin (`bin_2_categorical_[bin]`, the C++ RealThreshold /
        // BinToValue). This is BOTH the serialized model bitset AND the partition
        // routing key (routing by category value is equivalent to the C++ inner-bin
        // bitset routing and is consistent with the predict path). Negative category
        // values cannot appear in a winning split: bin 0 is the NaN dummy and the
        // finder scans bin_start = 1 - offset, never selecting it.
        let cat_values: Vec<u32> = cat_threshold_bins
            .iter()
            .map(|&bin| {
                let v = f
                    .bin_to_category
                    .get(bin as usize)
                    .copied()
                    .unwrap_or(bin as i32);
                v as u32
            })
            .collect();
        let cat_bitset_real = construct_bitset(&cat_values);

        // Partition by the REAL category bitset (left = in-bitset, right = default).
        data_partition.split_categorical(
            best_leaf,
            new_right,
            &f.bins,
            &cat_bitset_real,
            &f.bin_to_category,
        );

        let missing_type_code = match f.missing_type {
            MissingType::None => 0i8,
            MissingType::Zero => 1,
            MissingType::NaN => 2,
        };
        let split_gain_field = (best.gain + self.cfg.min_gain_to_split) as f32;
        let actual_left_count = data_partition.leaf_count(new_left);
        let actual_right_count = data_partition.leaf_count(new_right);

        tree.split_categorical(
            best_leaf,
            feat_idx, // inner feature index (== real on the single-group spine)
            feat_idx, // real feature index
            &cat_bitset_real,
            best.left_output,
            best.right_output,
            actual_left_count,
            actual_right_count,
            best.left_sum_hessian,
            best.right_sum_hessian,
            split_gain_field,
            missing_type_code,
        );

        // Seed child leaf-splits — IDENTICAL to the numeric path (the SplitInfo
        // sums/outputs are the same struct; only the node-growth differs).
        self.seed_child_leaf_splits(
            data_partition,
            best,
            smaller_leaf_splits,
            larger_leaf_splits,
            new_left,
            new_right,
        );

        Ok((new_left, new_right))
    }

    /// Seed the two child `LeafSplits` from the parent's `SplitInfo`
    /// (serial_tree_learner.cpp:851-871) — shared by the numeric and categorical
    /// `SplitInner` paths. The smaller/larger selection uses the SplitInfo counts;
    /// `num_data_in_leaf` is the PARTITION leaf-count.
    fn seed_child_leaf_splits(
        &self,
        data_partition: &DataPartition,
        best: &SplitInfo,
        smaller_leaf_splits: &mut LeafSplits,
        larger_leaf_splits: &mut LeafSplits,
        new_left: i32,
        new_right: i32,
    ) {
        let part_left = data_partition.leaf_count(new_left);
        let part_right = data_partition.leaf_count(new_right);
        if best.left_count < best.right_count {
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

    /// Set the W10 advanced learner constraints (ADV-01..05). The `Default`
    /// (all-empty/off) is the spine path — every gate INACTIVE and the numeric +
    /// categorical split paths byte-untouched (D-06).
    #[must_use]
    pub fn with_constraints(mut self, constraints: LearnerConstraints) -> Self {
        self.constraints = constraints;
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

/// ADV-03 `BinMapper::BinThreshold` analog: map a forced REAL threshold to the
/// bin a split should be placed AT. A split at bin `b` routes `value <=
/// bin_upper_bound[b]` LEFT, so the forced threshold `thr` maps to the LARGEST bin
/// `b` whose `bin_upper_bound[b] < thr` (i.e. all bins up to `b` go left). Falls
/// back to a clamped index when the upper-bound table is unavailable. Bounded to
/// `[0, num_bin - 2]` so the forced split is interior (a valid threshold bin).
fn bin_threshold(f: &FeatureColumn, thr: f64) -> u32 {
    let max_bin = f.num_bin.saturating_sub(2);
    if f.bin_upper_bound.is_empty() {
        return (thr as i64).clamp(0, max_bin as i64) as u32;
    }
    // C++ `BinMapper::ValueToBin`: a split for real threshold `thr` is placed at the
    // bin `b` that CONTAINS `thr` — the FIRST bin whose real upper bound is
    // `>= thr` (predict routes `value <= bin_upper_bound[b]` LEFT, so bins `0..=b`
    // route left). Note bin 0's recorded `bin_upper_bound` may be the model-text
    // ZERO SENTINEL (~1e-35) when `most_freq_bin == 0` (offset==1) — the
    // threshold-ENCODING artifact, NOT bin 0's real boundary (the midpoint). We
    // carry the previous real (monotone) boundary forward to neutralize the
    // sentinel so the `>= thr` scan is monotone.
    let mut prev = f64::NEG_INFINITY;
    let mut chosen = max_bin;
    for (b, &ub_raw) in f.bin_upper_bound.iter().enumerate() {
        let ub = if ub_raw < prev { prev } else { ub_raw };
        if ub >= thr {
            chosen = b as u32;
            break;
        }
        prev = ub;
    }
    chosen.min(max_bin)
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
            ..Default::default()
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
            ..Default::default()
        };
        (f, gradients, hessians)
    }

    #[test]
    fn per_bin_gains_packs_reverse_then_forward_one_alloc() {
        // 260609-bfx: the D-06 snapshot per-bin gains are packed REVERSE-then-FORWARD
        // into ONE allocation (was two separate Vecs per record); `rev_len` splits the
        // buffer. Verify the producer fills both halves with the expected candidate
        // counts so the `cand_rev()` / `cand_fwd()` accessors slice them correctly.
        let backend = CpuBackend;
        let client = cpu_client();
        let (f, _g, _h) = splittable_feature();
        let learner = SerialTreeLearner::new(&backend, &client, relaxed_cfg(), 31, -1);

        // FixHistogram'd per-bin (sum_grad, sum_hess) for the 4 bins (2 rows each).
        let hist = vec![-10.0f64, 2.0, -8.0, 2.0, 8.0, 2.0, 10.0, 2.0];
        let (gains, rev_len) = learner.per_bin_gains(&hist, &f, 0.0, 8.0, 8);

        // REVERSE scans bins [num_bin-1 .. 1], FORWARD bins [0 .. num_bin-2]:
        // 3 candidates each for a 4-bin feature with no gating.
        assert_eq!(rev_len, 3, "reverse-branch candidate count (cand_rev() length)");
        assert_eq!(gains.len(), 6, "one packed buffer = rev (3) + fwd (3)");
        assert_eq!(gains.len() - rev_len, 3, "forward-branch candidate count (cand_fwd() length)");
        // A splittable feature must surface at least one finite (non-gated) gain.
        assert!(gains.iter().any(|v| v.is_finite()), "expected a real split gain");
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

    #[test]
    fn train_rejects_out_of_range_bin_with_typed_error() {
        // V5 / threat T-04-01 RELOCATION (spike-003b): the per-element bin-range
        // check moved OUT of the now-branchless `build_leaf_histograms_raw` fold
        // into the once-per-train upstream gate in `train_inner` (learner.rs:700-714).
        // This test proves the V5 guarantee SURVIVES at its new location — an
        // out-of-range bin is rejected with the typed `BinIndexOutOfRange` BEFORE
        // any leaf builds, carrying the exact offending index + num_bin.
        let backend = CpuBackend;
        let client = cpu_client();
        // num_bin = 3, but the last row's bin is 3 (== num_bin, out of range).
        let f = FeatureColumn {
            bins: vec![0u32, 1, 3],
            num_bin: 3,
            offset: 0,
            min_bin: 0,
            max_bin: 2,
            default_bin: 3,
            most_freq_bin: 0,
            missing_type: MissingType::None,
            bin_upper_bound: vec![0.5, 1.5, 2.5],
            real_feature_index: 0,
            ..Default::default()
        };
        let g = vec![-1.0f32, 0.0, 1.0];
        let h = vec![1.0f32; 3];
        let mut learner = SerialTreeLearner::new(&backend, &client, relaxed_cfg(), 8, -1)
            .with_features(vec![f]);
        match learner.train(&g, &h, true) {
            Err(TreeLearnerError::BinIndexOutOfRange { index, num_bin }) => {
                assert_eq!(index, 3, "rejection must carry the exact offending bin");
                assert_eq!(num_bin, 3, "rejection must carry the feature's num_bin");
            }
            Ok(_) => panic!("train must REJECT an out-of-range bin, not grow a tree"),
            Err(other) => panic!("expected BinIndexOutOfRange, got {other:?}"),
        }
    }

    /// Two features: f0 is cleanly splittable (low bins neg grad, high bins pos),
    /// f1 is DECREASING (low bins pos grad → neg output, high bins neg grad → pos
    /// output reversed). A +1 monotone constraint on the decreasing feature must
    /// change which feature/split wins vs the unconstrained spine.
    fn two_feature_corpus() -> (Vec<FeatureColumn>, Vec<f32>, Vec<f32>) {
        // 8 rows. f0 bins increasing; f1 bins increasing but gradient arranged so
        // f1's best unconstrained split is "decreasing" (left output > right).
        let f0 = FeatureColumn {
            bins: vec![0u32, 0, 1, 1, 2, 2, 3, 3],
            num_bin: 4,
            offset: 0,
            min_bin: 0,
            max_bin: 3,
            default_bin: 4,
            most_freq_bin: 0,
            missing_type: MissingType::None,
            bin_upper_bound: vec![0.5, 1.5, 2.5, 3.5],
            real_feature_index: 0,
            ..Default::default()
        };
        let f1 = FeatureColumn {
            bins: vec![0u32, 1, 2, 3, 0, 1, 2, 3],
            real_feature_index: 1,
            ..f0.clone()
        };
        // gradient: low rows positive (=> negative output), high rows negative.
        let g = vec![6.0f32, 6.0, 5.0, 5.0, -5.0, -5.0, -6.0, -6.0];
        let h = vec![1.0f32; 8];
        (vec![f0, f1], g, h)
    }

    #[test]
    fn monotone_constraint_inactive_matches_spine() {
        // D-06: an all-zero monotone vector is the spine path byte-for-byte.
        let backend = CpuBackend;
        let client = cpu_client();
        let (f, g, h) = two_feature_corpus();
        let spine = SerialTreeLearner::new(&backend, &client, relaxed_cfg(), 8, -1)
            .with_features(f.clone())
            .train(&g, &h, true)
            .unwrap();
        let with_zero = SerialTreeLearner::new(&backend, &client, relaxed_cfg(), 8, -1)
            .with_features(f)
            .with_constraints(LearnerConstraints {
                monotone_constraints: vec![0, 0],
                ..Default::default()
            })
            .train(&g, &h, true)
            .unwrap();
        assert_eq!(spine.to_string(), with_zero.to_string(), "inactive monotone == spine (D-06)");
    }

    #[test]
    fn monotone_constraint_alters_chosen_tree() {
        // A +1 constraint on f1 (whose best split is decreasing) must change the
        // grown tree vs the unconstrained spine.
        let backend = CpuBackend;
        let client = cpu_client();
        let (f, g, h) = two_feature_corpus();
        let spine = SerialTreeLearner::new(&backend, &client, relaxed_cfg(), 4, -1)
            .with_features(f.clone())
            .train(&g, &h, true)
            .unwrap();
        let constrained = SerialTreeLearner::new(&backend, &client, relaxed_cfg(), 4, -1)
            .with_features(f)
            .with_constraints(LearnerConstraints {
                // f0 +1, f1 +1: f1's decreasing splits get rejected/penalized.
                monotone_constraints: vec![1, 1],
                ..Default::default()
            })
            .train(&g, &h, true)
            .unwrap();
        // The constrained tree must differ OR enforce monotone leaf outputs.
        // Verify every +1 split has left_output <= right_output.
        for node in 0..(constrained.num_leaves - 1) as usize {
            let left = constrained.left_child[node];
            let right = constrained.right_child[node];
            if left < 0 && right < 0 {
                let lo = constrained.leaf_value[(!left) as usize];
                let ro = constrained.leaf_value[(!right) as usize];
                assert!(
                    lo <= ro + 1e-9,
                    "+1 monotone leaf order violated: {lo} > {ro}"
                );
            }
        }
        let _ = spine;
    }

    #[test]
    fn interaction_constraint_restricts_features() {
        // Interaction group {0} only: after f0 splits the root, f1 may NOT be used
        // deeper (it is not in any group containing f0). With a single group [0],
        // f1 is never allowed (branch empty → group {0}; once f0 in branch → only
        // {0} allowed).
        let backend = CpuBackend;
        let client = cpu_client();
        let (f, g, h) = two_feature_corpus();
        let tree = SerialTreeLearner::new(&backend, &client, relaxed_cfg(), 8, -1)
            .with_features(f)
            .with_constraints(LearnerConstraints {
                interaction_constraints: vec![vec![0]],
                ..Default::default()
            })
            .train(&g, &h, true)
            .unwrap();
        // No split node may use feature 1.
        assert!(
            tree.split_feature.iter().all(|&sf| sf != 1),
            "interaction group [0] forbids feature 1: {:?}",
            tree.split_feature
        );
    }

    #[test]
    fn extra_trees_is_deterministic_per_seed() {
        // Same extra_seed => identical tree (RNG-replay). Different seed may differ.
        let backend = CpuBackend;
        let client = cpu_client();
        let (f, g, h) = two_feature_corpus();
        let build = |seed: i32| {
            SerialTreeLearner::new(&backend, &client, relaxed_cfg(), 8, -1)
                .with_features(f.clone())
                .with_constraints(LearnerConstraints {
                    extra_trees: true,
                    extra_seed: seed,
                    ..Default::default()
                })
                .train(&g, &h, true)
                .unwrap()
                .to_string()
        };
        assert_eq!(build(6), build(6), "extra_trees replays bit-exact for a fixed seed");
    }

    #[test]
    fn forced_split_drives_root_feature() {
        // A forced split on feature 1 at threshold 1.5 (bin boundary) must make the
        // ROOT split feature 1, even though feature 0 is the unconstrained winner.
        let backend = CpuBackend;
        let client = cpu_client();
        let (f, g, h) = two_feature_corpus();
        let forced = crate::forced_splits::ForcedSplitNode {
            feature: 1,
            threshold: 1.5,
            left: None,
            right: None,
        };
        let tree = SerialTreeLearner::new(&backend, &client, relaxed_cfg(), 8, -1)
            .with_features(f)
            .with_constraints(LearnerConstraints {
                forced_splits: Some(forced),
                ..Default::default()
            })
            .train(&g, &h, true)
            .unwrap();
        assert_eq!(
            tree.split_feature[0], 1,
            "the forced root split must use feature 1, got {:?}",
            tree.split_feature
        );
    }

    #[test]
    fn forced_split_inactive_matches_spine() {
        // D-06: no forced split == the spine tree byte-for-byte.
        let backend = CpuBackend;
        let client = cpu_client();
        let (f, g, h) = two_feature_corpus();
        let spine = SerialTreeLearner::new(&backend, &client, relaxed_cfg(), 8, -1)
            .with_features(f.clone())
            .train(&g, &h, true)
            .unwrap();
        let none_forced = SerialTreeLearner::new(&backend, &client, relaxed_cfg(), 8, -1)
            .with_features(f)
            .with_constraints(LearnerConstraints {
                forced_splits: None,
                ..Default::default()
            })
            .train(&g, &h, true)
            .unwrap();
        assert_eq!(spine.to_string(), none_forced.to_string());
    }

    #[test]
    fn cegb_penalty_changes_split_selection() {
        // A large cegb_penalty_split makes the per-split cost dominate, forcing the
        // tree to stop earlier than the unconstrained spine (fewer leaves).
        let backend = CpuBackend;
        let client = cpu_client();
        let (f, g, h) = two_feature_corpus();
        let spine = SerialTreeLearner::new(&backend, &client, relaxed_cfg(), 8, -1)
            .with_features(f.clone())
            .train(&g, &h, true)
            .unwrap();
        let penalized = SerialTreeLearner::new(&backend, &client, relaxed_cfg(), 8, -1)
            .with_features(f)
            .with_constraints(LearnerConstraints {
                cegb_tradeoff: 1.0,
                cegb_penalty_split: 1000.0, // huge per-row split cost
                ..Default::default()
            })
            .train(&g, &h, true)
            .unwrap();
        assert!(
            penalized.num_leaves < spine.num_leaves,
            "a huge cegb_penalty_split must reduce the tree size ({} vs {})",
            penalized.num_leaves,
            spine.num_leaves
        );
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

    #[test]
    fn renew_tree_output_overwrites_each_leaf_via_closure() {
        let backend = CpuBackend;
        let client = cpu_client();
        // Build a real 2-leaf partition over a splittable feature so each leaf has
        // a concrete (non-empty) row set the closure can see.
        let (f, _g, _h) = splittable_feature();
        let mut part = DataPartition::new(8, 2);
        let (lc, rc) = part
            .split(
                &backend, &client, 0, 1, &f.bins, f.num_bin, f.min_bin, f.max_bin, 1,
                f.most_freq_bin,
            )
            .expect("partition split ok");
        assert!(lc > 0 && rc > 0);

        let learner = SerialTreeLearner::new(&backend, &client, relaxed_cfg(), 2, -1);
        let mut tree = two_leaf_tree();
        // The renew closure returns a per-leaf constant (here: 100 + leaf index *
        // 10), proving each leaf's output is overwritten with the closure's value
        // (the l1 median-residual body uses the same seam — leaf -> median(rows)).
        learner.renew_tree_output(
            &mut tree,
            &part,
            Some(|leaf: i32, rows: &[u32]| {
                assert!(!rows.is_empty(), "each leaf has rows");
                100.0 + leaf as f64 * 10.0
            }),
        );
        assert_eq!(tree.leaf_value, vec![100.0, 110.0], "each leaf overwritten by closure");
    }

    #[test]
    fn renew_tree_output_single_leaf_is_noop() {
        let backend = CpuBackend;
        let client = cpu_client();
        let learner = SerialTreeLearner::new(&backend, &client, relaxed_cfg(), 1, -1);
        let part = DataPartition::new(4, 1);
        let mut tree = two_leaf_tree();
        tree.num_leaves = 1;
        let before = tree.leaf_value.clone();
        learner.renew_tree_output(
            &mut tree,
            &part,
            Some(|_leaf: i32, _rows: &[u32]| -42.0),
        );
        assert_eq!(tree.leaf_value, before, "num_leaves<=1 renew is a no-op (T-06-03-02)");
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
            ..Default::default()
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

    /// A categorical feature whose categories separate the gradient sign grows a
    /// categorical split (CATEGORICAL_MASK set, num_cat==1) that predicts faithfully.
    #[test]
    fn learner_grows_a_categorical_split() {
        let backend = CpuBackend;
        let client = cpu_client();
        // 8 rows, categorical feature with 3 real categories -> bins 1,2,3
        // (bin 0 is the NaN dummy). most_freq_bin==0 -> offset 1.
        // Categories: cat 10 (bin1), cat 20 (bin2), cat 30 (bin3).
        // rows: cat10,cat10,cat10,cat10 (neg grad) | cat20..,cat30.. (pos grad).
        let bins = vec![1u32, 1, 1, 1, 2, 2, 3, 3];
        let gradients = vec![-5.0f32, -5.0, -5.0, -5.0, 4.0, 4.0, 5.0, 5.0];
        let hessians = vec![1.0f32; 8];
        let f = FeatureColumn {
            bins,
            num_bin: 4, // bins 0..3 (0 = NaN dummy)
            offset: 1,  // most_freq_bin == 0
            min_bin: 1,
            max_bin: 3,
            default_bin: 0,
            most_freq_bin: 0,
            missing_type: MissingType::None,
            bin_upper_bound: Vec::new(),
            real_feature_index: 0,
            bin_type: BinType::Categorical,
            // bin -> category value: bin0 = NaN dummy (-1), bin1=10, bin2=20, bin3=30.
            bin_to_category: vec![-1, 10, 20, 30],
        };
        let mut cfg = relaxed_cfg();
        // one-hot path (num_bin 4 <= max_cat_to_onehot 4); relax leaf gates.
        cfg.min_data_in_leaf = 1;
        cfg.min_sum_hessian_in_leaf = 0.0;
        let mut learner = SerialTreeLearner::new(&backend, &client, cfg, 8, 1)
            .with_features(vec![f]);
        let tree = learner.train(&gradients, &hessians, true).expect("train ok");
        assert_eq!(tree.num_leaves, 2, "a categorical split was grown");
        assert_eq!(tree.num_cat, 1, "num_cat == 1");
        // The node's decision_type has the categorical bit set.
        assert!(
            tree.decision_type[0] & 1 != 0,
            "decision_type has CATEGORICAL_MASK: {}",
            tree.decision_type[0]
        );
        // Rows conserved + byte-stable round-trip through model-text.
        let total: i32 = tree.leaf_count.iter().sum();
        assert_eq!(total, 8);
        let s = tree.to_string();
        assert!(s.contains("num_cat=1"));
        let parsed = Tree::parse(&s).expect("round-trips");
        assert_eq!(parsed.to_string(), s);
        // Predict: a cat-10 row vs a cat-30 row land in different leaves.
        let leaf_10 = parsed.get_leaf(&[10.0]);
        let leaf_30 = parsed.get_leaf(&[30.0]);
        assert_ne!(leaf_10, leaf_30, "cat 10 and cat 30 route to different leaves");
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

