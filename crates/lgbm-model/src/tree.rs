//! Faithful 1:1 port of LightGBM's single decision `Tree` (D-04).
//!
//! Transcribed line-for-line from:
//! - `LightGBM/include/LightGBM/tree.h` — member layout, `Predict`/`GetLeaf`
//!   (587-615, 701-713), `NumericalDecision`/`CategoricalDecision`/`Decision`
//!   (337-415), `GetDecisionType`/`GetMissingType`/`IsZero` (254-281), masks
//!   (`kCategoricalMask`/`kDefaultLeftMask`, 20-21).
//! - `LightGBM/src/io/tree.cpp` — `Tree::ToString` (339-409, the byte-exact
//!   section order + per-field formatter mode) and the keyed parse ctor
//!   `Tree(const char* str, size_t* used_len)` (685-866, order-independent
//!   key=value map + per-field fallbacks + the single-leaf early return at 747).
//! - `LightGBM/include/LightGBM/utils/common.h` — `FindInBitset` (836).
//!
//! The in-memory representation is the C++ **parallel-array** layout, NOT an
//! idiomatic Rust node enum (D-04 mandate): predict walks node indices, leaves
//! are encoded as `~node` (bitwise-NOT) — a non-negative leaf id.
//!
//! # Arithmetic / fidelity notes
//! - Feature values are traversed as `f64` against the stored `f64` thresholds
//!   (`threshold_`) and `f64` leaf values — the predict path is the *real-value*
//!   path (`split_feature_`, ORIGINAL feature index), NEVER the `_inner_` bin
//!   path (RESEARCH anti-pattern). The ~1e-6 contract is applied at the f32
//!   comparison boundary only.
//! - `decision_type_` is an `i8` packing three facts: bit0 = categorical
//!   (`kCategoricalMask`), bit1 = default-left (`kDefaultLeftMask`), bits2-3 =
//!   `missing_type` (0=None, 1=Zero, 2=NaN).
//! - `is_zero` reuses `lgbm_core::types::K_ZERO_THRESHOLD` (1e-35, never
//!   redefined).
//!
//! # Strictness vs C++
//! The C++ parser indexes parsed arrays with raw `[]` (UB on a malformed file).
//! This port is STRICTER (Security V5 / T-03-03): every parsed array length is
//! validated against `num_leaves`/`num_cat` BEFORE any indexing, and a malformed
//! file returns [`ModelError::MalformedModel`] — never a panic. On a *valid*
//! model the observable behavior is identical to C++.

use std::collections::HashMap;
use std::fmt::Write as _;

use lgbm_core::types::K_ZERO_THRESHOLD;

use crate::error::ModelError;
use crate::format::{format_g17, format_g6};

/// C++ `#define kCategoricalMask (1)` (`tree.h:20`).
const CATEGORICAL_MASK: i8 = 1;
/// C++ `#define kDefaultLeftMask (2)` (`tree.h:21`).
const DEFAULT_LEFT_MASK: i8 = 2;

/// C++ `MissingType::Zero` numeric value (the `(decision_type >> 2) & 3` field).
const MISSING_ZERO: i8 = 1;
/// C++ `MissingType::NaN` numeric value.
const MISSING_NAN: i8 = 2;

/// C++ `Tree(const char*, size_t*)` reads at most 22 `key=value` header lines.
const MAX_HEADER_LINES: usize = 22;

/// C++ `Tree::PathElement` (`tree.h:443-454`) — one element of the unique feature
/// path used by the TreeSHAP recursion. `pweight` is the permutation weight (paths
/// with `i-1` ones) and is tracked separately from the fractions.
#[derive(Debug, Clone, Copy, Default)]
struct PathElement {
    feature_index: i32,
    zero_fraction: f64,
    one_fraction: f64,
    pweight: f64,
}

/// A single decision tree — faithful parallel-array mirror of C++ `Tree`
/// (`tree.h`). Field names keep the C++ correspondence explicit.
#[derive(Debug, Clone, PartialEq)]
pub struct Tree {
    /// C++ `int num_leaves_`.
    pub num_leaves: i32,
    /// C++ `int num_cat_`.
    pub num_cat: i32,
    /// C++ `std::vector<int> left_child_` (len `num_leaves-1`); negative = `~leaf`.
    pub left_child: Vec<i32>,
    /// C++ `std::vector<int> right_child_` (len `num_leaves-1`).
    pub right_child: Vec<i32>,
    /// C++ `std::vector<int> split_feature_` — ORIGINAL feature index (len `num_leaves-1`).
    pub split_feature: Vec<i32>,
    /// C++ `std::vector<double> threshold_` — real-value threshold (len `num_leaves-1`).
    pub threshold: Vec<f64>,
    /// C++ `std::vector<int8_t> decision_type_` (len `num_leaves-1`).
    pub decision_type: Vec<i8>,
    /// C++ `std::vector<float> split_gain_` (len `num_leaves-1`) — predict-irrelevant metadata.
    pub split_gain: Vec<f32>,
    /// C++ `std::vector<double> leaf_value_` (len `num_leaves`).
    pub leaf_value: Vec<f64>,
    /// C++ `std::vector<double> leaf_weight_` (len `num_leaves`).
    pub leaf_weight: Vec<f64>,
    /// C++ `std::vector<int> leaf_count_` (len `num_leaves`).
    pub leaf_count: Vec<i32>,
    /// C++ `std::vector<double> internal_value_` (len `num_leaves-1`) — metadata.
    pub internal_value: Vec<f64>,
    /// C++ `std::vector<double> internal_weight_` (len `num_leaves-1`) — metadata.
    pub internal_weight: Vec<f64>,
    /// C++ `std::vector<int> internal_count_` (len `num_leaves-1`) — metadata.
    pub internal_count: Vec<i32>,
    /// C++ `std::vector<int> cat_boundaries_` (len `num_cat+1`, only when `num_cat>0`).
    pub cat_boundaries: Vec<i32>,
    /// C++ `std::vector<uint32_t> cat_threshold_` — bitset blocks (only when `num_cat>0`).
    pub cat_threshold: Vec<u32>,
    /// C++ `double shrinkage_`.
    pub shrinkage: f64,
    /// C++ `bool is_linear_` — Phase 3 in-scope models are always non-linear.
    pub is_linear: bool,

    // --- growth-time-only bookkeeping (Phase 5, D-07) ---
    // These four parallel arrays mirror the C++ growth-time members used by
    // `Tree::Split` (`tree.h:543-585`). They track depth/parent/inner-feature/
    // bin-threshold during leaf-wise growth and are NOT serialized: `to_string()`
    // already matches C++ byte-for-byte WITHOUT them (the C++ `ToString` does not
    // emit `leaf_depth_`/`leaf_parent_`/`split_feature_inner_`/`threshold_in_bin_`).
    /// C++ `std::vector<int> leaf_depth_` (len `num_leaves`) — depth of each leaf.
    /// Growth-time only; never serialized.
    pub leaf_depth: Vec<i32>,
    /// C++ `std::vector<int> leaf_parent_` (len `num_leaves`) — the node index whose
    /// child is this leaf (root leaf 0 has parent `-1`). Growth-time only.
    pub leaf_parent: Vec<i32>,
    /// C++ `std::vector<int> split_feature_inner_` (len `num_leaves-1`) — the INNER
    /// (bin-mapper) feature index of each split, distinct from the ORIGINAL
    /// `split_feature_`. Growth-time only; never serialized.
    pub split_feature_inner: Vec<i32>,
    /// C++ `std::vector<uint32_t> threshold_in_bin_` (len `num_leaves-1`) — the
    /// integer bin threshold of each split (distinct from the real-value
    /// `threshold_`). Growth-time only; never serialized.
    pub threshold_in_bin: Vec<u32>,
}

/// C++ `Tree::GetDecisionType` (`tree.h:262`): `(decision_type & mask) > 0`.
#[inline]
fn get_decision_type(decision_type: i8, mask: i8) -> bool {
    (decision_type & mask) > 0
}

/// C++ `Tree::GetMissingType` (`tree.h:274`): `(decision_type >> 2) & 3`.
#[inline]
fn get_missing_type(decision_type: i8) -> i8 {
    (decision_type >> 2) & 3
}

/// C++ `Tree::IsZero` (`tree.h:254`):
/// `fval >= -kZeroThreshold && fval <= kZeroThreshold`.
#[inline]
fn is_zero(fval: f64) -> bool {
    fval >= -K_ZERO_THRESHOLD && fval <= K_ZERO_THRESHOLD
}

/// C++ `Common::FindInBitset` (`common.h:836`):
/// `i1 = pos/32; if (i1 >= n) return false; return (bits[i1] >> (pos%32)) & 1;`.
#[inline]
fn find_in_bitset(bits: &[u32], pos: i32) -> bool {
    let i1 = (pos / 32) as usize;
    if i1 >= bits.len() {
        return false;
    }
    let i2 = pos % 32;
    (bits[i1] >> i2) & 1 == 1
}

impl Tree {
    /// C++ `Tree::NumericalDecision` (`tree.h:337-355`). `node` is a non-negative
    /// internal node index; returns the next node (negative = `~leaf`).
    #[inline]
    fn numerical_decision(&self, mut fval: f64, node: usize) -> i32 {
        let missing_type = get_missing_type(self.decision_type[node]);
        if fval.is_nan() && missing_type != MISSING_NAN {
            fval = 0.0;
        }
        if (missing_type == MISSING_ZERO && is_zero(fval))
            || (missing_type == MISSING_NAN && fval.is_nan())
        {
            return if get_decision_type(self.decision_type[node], DEFAULT_LEFT_MASK) {
                self.left_child[node]
            } else {
                self.right_child[node]
            };
        }
        if fval <= self.threshold[node] {
            self.left_child[node]
        } else {
            self.right_child[node]
        }
    }

    /// C++ `Tree::CategoricalDecision` (`tree.h:374-390`). Built now (the array
    /// path exists); categorical-split PARITY is asserted in 03-03.
    #[inline]
    fn categorical_decision(&self, fval: f64, node: usize) -> i32 {
        let int_fval: i32;
        if fval.is_nan() {
            return self.right_child[node];
        } else {
            int_fval = fval as i32;
            if int_fval < 0 {
                return self.right_child[node];
            }
        }
        let cat_idx = self.threshold[node] as i32 as usize;
        let lo = self.cat_boundaries[cat_idx] as usize;
        let hi = self.cat_boundaries[cat_idx + 1] as usize;
        if find_in_bitset(&self.cat_threshold[lo..hi], int_fval) {
            self.left_child[node]
        } else {
            self.right_child[node]
        }
    }

    /// C++ `Tree::Decision` (`tree.h:401-407`): dispatch on the categorical mask.
    #[inline]
    fn decision(&self, fval: f64, node: usize) -> i32 {
        if get_decision_type(self.decision_type[node], CATEGORICAL_MASK) {
            self.categorical_decision(fval, node)
        } else {
            self.numerical_decision(fval, node)
        }
    }

    /// C++ `Tree::GetLeaf` (`tree.h:701-713`): descend from node 0, return `~node`.
    /// `feature_values` is the RAW per-row feature buffer (width `max_feature_idx+1`).
    pub fn get_leaf(&self, feature_values: &[f64]) -> i32 {
        let mut node = 0i32;
        if self.num_cat > 0 {
            while node >= 0 {
                let fv = feature_values[self.split_feature[node as usize] as usize];
                node = self.decision(fv, node as usize);
            }
        } else {
            while node >= 0 {
                let fv = feature_values[self.split_feature[node as usize] as usize];
                node = self.numerical_decision(fv, node as usize);
            }
        }
        !node
    }

    /// C++ `Tree::Predict` (`tree.h:587-615`, non-linear path): the leaf value of
    /// the leaf this row falls into. Single-leaf trees return `leaf_value[0]`.
    pub fn predict(&self, feature_values: &[f64]) -> f64 {
        if self.num_leaves > 1 {
            let leaf = self.get_leaf(feature_values);
            self.leaf_value[leaf as usize]
        } else {
            self.leaf_value[0]
        }
    }

    /// C++ `Tree::PredictLeafIndex` (`tree.h:650-657`): the leaf id (non-negative).
    pub fn predict_leaf_index(&self, feature_values: &[f64]) -> i32 {
        if self.num_leaves > 1 {
            self.get_leaf(feature_values)
        } else {
            0
        }
    }

    /// C++ `Tree::LeafOutput` (`tree.h:91`): `leaf_value_[leaf]`.
    #[inline]
    fn leaf_output(&self, leaf: usize) -> f64 {
        self.leaf_value[leaf]
    }

    /// C++ `Tree::data_count` (`tree.h:181`): the cover (row count) at `node`.
    /// `node >= 0` is an internal node (`internal_count_[node]`); `node < 0` is the
    /// leaf `~node` (`leaf_count_[~node]`).
    #[inline]
    fn data_count(&self, node: i32) -> f64 {
        if node >= 0 {
            self.internal_count[node as usize] as f64
        } else {
            self.leaf_count[(!node) as usize] as f64
        }
    }

    /// C++ `Tree::ExpectedValue` (`tree.cpp:1031-1039`): the cover-weighted average
    /// of the leaf outputs — the SHAP base / bias term. For a single-leaf tree this
    /// is the leaf output. Otherwise `total_count = internal_count_[0]` (the root
    /// cover) and `exp_value = Σ (leaf_count_[i] / total_count) * leaf_value_[i]`.
    pub fn expected_value(&self) -> f64 {
        if self.num_leaves == 1 {
            return self.leaf_output(0);
        }
        let total_count = self.internal_count[0] as f64;
        let mut exp_value = 0.0f64;
        for i in 0..self.num_leaves as usize {
            exp_value += (self.leaf_count[i] as f64 / total_count) * self.leaf_output(i);
        }
        exp_value
    }

    /// The maximum leaf depth of the grown tree (C++ `Tree::RecomputeMaxDepth`,
    /// `tree.cpp:1041-1053`). A single-leaf tree has depth 0. A parsed model carries
    /// no serialized `leaf_depth_`, so the depth is recomputed from the node
    /// structure by descending from the root.
    fn max_depth(&self) -> i32 {
        if self.num_leaves <= 1 {
            return 0;
        }
        let mut depths = vec![0i32; self.num_leaves as usize];
        self.recompute_leaf_depths(0, 0, &mut depths);
        depths.iter().copied().max().unwrap_or(0)
    }

    /// C++ `Tree::RecomputeLeafDepths` (`tree.h:691-699`): fill `depths[~node]` for
    /// every leaf by descending from `node` at `depth`.
    fn recompute_leaf_depths(&self, node: i32, depth: i32, depths: &mut [i32]) {
        if node < 0 {
            depths[(!node) as usize] = depth;
        } else {
            let n = node as usize;
            self.recompute_leaf_depths(self.left_child[n], depth + 1, depths);
            self.recompute_leaf_depths(self.right_child[n], depth + 1, depths);
        }
    }

    /// C++ `Tree::PredictContrib` (`tree.h:668-677`): accumulate this tree's
    /// per-feature SHAP contributions into `output[0..num_features]` and the
    /// expected-value base into `output[num_features]` (TreeSHAP, arXiv:1706.06060).
    ///
    /// `output` MUST have length `num_features + 1`; the caller owns zeroing /
    /// striding it across classes + accumulating across the ensemble. The
    /// INVARIANT (asserted in tests + the parity gate): for one tree,
    /// `Σ output[0..num_features] + output[num_features] == tree.predict(row)`.
    ///
    /// Works over numeric AND categorical decision nodes (the recursion calls
    /// [`Self::decision`], which dispatches on the categorical mask exactly as
    /// `GetLeaf` does, so `find_in_bitset` routing is honored).
    pub fn predict_contrib(&self, feature_values: &[f64], num_features: usize, output: &mut [f64]) {
        output[num_features] += self.expected_value();
        if self.num_leaves > 1 {
            let max_path_len = (self.max_depth() + 1) as usize;
            // C++ allocates max_path_len*(max_path_len+1)/2 PathElements.
            let mut unique_path_data =
                vec![PathElement::default(); max_path_len * (max_path_len + 1) / 2];
            self.tree_shap(feature_values, output, 0, 0, &mut unique_path_data, 1.0, 1.0, -1);
        }
    }

    /// C++ `Tree::ExtendPath` (`tree.cpp:868-880`). `unique_path` is the slice
    /// starting at the current frame's base (the caller passes `&mut path[base..]`).
    fn extend_path(
        unique_path: &mut [PathElement],
        unique_depth: usize,
        zero_fraction: f64,
        one_fraction: f64,
        feature_index: i32,
    ) {
        unique_path[unique_depth].feature_index = feature_index;
        unique_path[unique_depth].zero_fraction = zero_fraction;
        unique_path[unique_depth].one_fraction = one_fraction;
        unique_path[unique_depth].pweight = if unique_depth == 0 { 1.0 } else { 0.0 };
        let ud = unique_depth as f64;
        for i in (0..unique_depth).rev() {
            let prev = unique_path[i].pweight;
            unique_path[i + 1].pweight +=
                one_fraction * prev * (i as f64 + 1.0) / (ud + 1.0);
            unique_path[i].pweight =
                zero_fraction * prev * (ud - i as f64) / (ud + 1.0);
        }
    }

    /// C++ `Tree::UnwindPath` (`tree.cpp:882-905`).
    fn unwind_path(unique_path: &mut [PathElement], unique_depth: usize, path_index: usize) {
        let one_fraction = unique_path[path_index].one_fraction;
        let zero_fraction = unique_path[path_index].zero_fraction;
        let mut next_one_portion = unique_path[unique_depth].pweight;
        let ud = unique_depth as f64;

        for i in (0..unique_depth).rev() {
            if one_fraction != 0.0 {
                let tmp = unique_path[i].pweight;
                unique_path[i].pweight =
                    next_one_portion * (ud + 1.0) / ((i as f64 + 1.0) * one_fraction);
                next_one_portion =
                    tmp - unique_path[i].pweight * zero_fraction * (ud - i as f64) / (ud + 1.0);
            } else {
                unique_path[i].pweight =
                    (unique_path[i].pweight * (ud + 1.0)) / (zero_fraction * (ud - i as f64));
            }
        }

        for i in path_index..unique_depth {
            unique_path[i].feature_index = unique_path[i + 1].feature_index;
            unique_path[i].zero_fraction = unique_path[i + 1].zero_fraction;
            unique_path[i].one_fraction = unique_path[i + 1].one_fraction;
        }
    }

    /// C++ `Tree::UnwoundPathSum` (`tree.cpp:907-925`).
    fn unwound_path_sum(unique_path: &[PathElement], unique_depth: usize, path_index: usize) -> f64 {
        let one_fraction = unique_path[path_index].one_fraction;
        let zero_fraction = unique_path[path_index].zero_fraction;
        let mut next_one_portion = unique_path[unique_depth].pweight;
        let mut total = 0.0f64;
        let ud = unique_depth as f64;
        for i in (0..unique_depth).rev() {
            if one_fraction != 0.0 {
                let tmp = next_one_portion * (ud + 1.0) / ((i as f64 + 1.0) * one_fraction);
                total += tmp;
                next_one_portion =
                    unique_path[i].pweight - tmp * zero_fraction * ((ud - i as f64) / (ud + 1.0));
            } else {
                total +=
                    (unique_path[i].pweight / zero_fraction) / ((ud - i as f64) / (ud + 1.0));
            }
        }
        total
    }

    /// C++ `Tree::TreeSHAP` (`tree.cpp:927-977`) — the recursive polynomial-time
    /// SHAP value computation. `path` is the full preallocated buffer; the current
    /// frame's unique-path slice starts at offset `unique_depth` (mirroring the C++
    /// `parent_unique_path + unique_depth` pointer arithmetic over one flat array).
    #[allow(clippy::too_many_arguments)]
    fn tree_shap(
        &self,
        feature_values: &[f64],
        phi: &mut [f64],
        node: i32,
        unique_depth: usize,
        path: &mut [PathElement],
        parent_zero_fraction: f64,
        parent_one_fraction: f64,
        parent_feature_index: i32,
    ) {
        // Copy the parent's [0, unique_depth) frame forward to this frame's base,
        // then operate on the slice starting at `unique_depth` (the C++
        // `unique_path = parent_unique_path + unique_depth` + std::copy).
        if unique_depth > 0 {
            path.copy_within(0..unique_depth, unique_depth);
        }
        let (_, frame) = path.split_at_mut(unique_depth);

        Self::extend_path(
            frame,
            unique_depth,
            parent_zero_fraction,
            parent_one_fraction,
            parent_feature_index,
        );

        if node < 0 {
            // Leaf node: distribute the leaf output across the path features.
            let leaf_value = self.leaf_value[(!node) as usize];
            for i in 1..=unique_depth {
                let w = Self::unwound_path_sum(frame, unique_depth, i);
                let el = &frame[i];
                phi[el.feature_index as usize] +=
                    w * (el.one_fraction - el.zero_fraction) * leaf_value;
            }
        } else {
            let n = node as usize;
            let split_feat = self.split_feature[n];
            let hot_index = self.decision(feature_values[split_feat as usize], n);
            let cold_index = if hot_index == self.left_child[n] {
                self.right_child[n]
            } else {
                self.left_child[n]
            };
            let w = self.data_count(node);
            let hot_zero_fraction = self.data_count(hot_index) / w;
            let cold_zero_fraction = self.data_count(cold_index) / w;
            let mut incoming_zero_fraction = 1.0f64;
            let mut incoming_one_fraction = 1.0f64;

            // If we have already split on this feature, undo that split so we can
            // redo it for this node.
            let mut path_index = 0usize;
            while path_index <= unique_depth {
                if frame[path_index].feature_index == split_feat {
                    break;
                }
                path_index += 1;
            }
            let mut next_unique_depth = unique_depth;
            if path_index != unique_depth + 1 {
                incoming_zero_fraction = frame[path_index].zero_fraction;
                incoming_one_fraction = frame[path_index].one_fraction;
                Self::unwind_path(frame, unique_depth, path_index);
                next_unique_depth -= 1;
            }

            // Recurse on the hot then cold child. The child operates on `path`
            // starting at the SAME base as this frame (the C++ recursion passes
            // `unique_path`, which is `path + unique_depth`). We pass the same
            // `path` slice but the child's base offset is `next_unique_depth + 1`
            // RELATIVE to the parent's base — which here is `unique_depth`. So the
            // child's absolute base into the parent's `path` is
            // `unique_depth + next_unique_depth + 1`; replicate by passing the
            // sub-slice `&mut path[unique_depth..]` with depth `next_unique_depth + 1`.
            self.tree_shap(
                feature_values,
                phi,
                hot_index,
                next_unique_depth + 1,
                frame,
                hot_zero_fraction * incoming_zero_fraction,
                incoming_one_fraction,
                split_feat,
            );
            self.tree_shap(
                feature_values,
                phi,
                cold_index,
                next_unique_depth + 1,
                frame,
                cold_zero_fraction * incoming_zero_fraction,
                0.0,
                split_feat,
            );
        }
    }

    /// C++ `Tree::Split` (`tree.h:543-585` structural body + the public
    /// `tree.cpp:61-75` numerical wrapper) — grow `leaf` into an internal node
    /// with two child leaves.
    ///
    /// Transcribes the NUMERICAL (non-categorical) growth path. `feature` is the
    /// INNER (bin-mapper) feature index, `real_feature` the ORIGINAL feature index
    /// (the one `predict` traverses against `threshold`). `threshold_bin` is the
    /// integer bin threshold (`threshold_in_bin_`), `threshold` the real f64
    /// value. Leaf outputs/weights/counts are computed by the LEARNER (via
    /// `gain::calculate_splitted_leaf_output`) and passed in — this method only
    /// stores them and performs the C++ array rewiring.
    ///
    /// Mechanics (mirroring `tree.h:543-585`):
    /// - `new_node_idx = num_leaves - 1`; the old `leaf` keeps its leaf id, the new
    ///   right child takes leaf id `num_leaves`.
    /// - rewire the parent's child pointer (`leaf_parent_[leaf]`) to the new node;
    /// - `left_child_[node] = ~leaf`, `right_child_[node] = ~num_leaves`
    ///   (the C++ `~` leaf encoding);
    /// - `internal_value_[node] = leaf_value_[leaf]` (pre-split leaf output);
    /// - `leaf_value_[leaf] = left_value`, `leaf_value_[num_leaves] = right_value`
    ///   (same for weight/count);
    /// - `leaf_depth_[num_leaves] = leaf_depth_[leaf] + 1`, then `leaf_depth_[leaf] += 1`;
    /// - `decision_type_[node]` packs `default_left` (`kDefaultLeftMask`) +
    ///   `missing_type` (bits 2-3, via the C++ `SetMissingType` encoding).
    ///
    /// `missing_type` is the numeric C++ `MissingType` (0=None, 1=Zero, 2=NaN).
    /// Categorical splits (`SplitCategorical`) are TRL-06, deferred to Phase 7.
    #[allow(clippy::too_many_arguments)]
    pub fn split(
        &mut self,
        leaf: i32,
        feature: i32,
        real_feature: i32,
        threshold_bin: u32,
        threshold: f64,
        left_value: f64,
        right_value: f64,
        left_count: i32,
        right_count: i32,
        left_weight: f64,
        right_weight: f64,
        gain: f32,
        missing_type: i8,
        default_left: bool,
    ) {
        let new_node_idx = (self.num_leaves - 1) as usize;
        let leaf_u = leaf as usize;
        let new_leaf = self.num_leaves; // the right child's leaf id

        // Rewire the parent's child pointer to the new internal node. The root
        // leaf has parent -1 (no rewiring needed).
        let parent = self.leaf_parent[leaf_u];
        if parent >= 0 {
            let p = parent as usize;
            if self.right_child[p] == !leaf {
                self.right_child[p] = new_node_idx as i32;
            } else {
                self.left_child[p] = new_node_idx as i32;
            }
        }

        // Append the new internal node's split metadata (parallel-array growth).
        self.split_feature_inner.push(feature);
        self.split_feature.push(real_feature);
        self.split_gain.push(gain);
        self.threshold_in_bin.push(threshold_bin);
        self.threshold.push(threshold);
        // `internal_value_[node] = leaf_value_[leaf]` — pre-split leaf output.
        self.internal_value.push(self.leaf_value[leaf_u]);
        self.internal_weight.push(0.0);
        self.internal_count.push(left_count + right_count);
        // Child pointers: left = old leaf, right = new leaf (C++ `~leaf` encoding).
        self.left_child.push(!leaf);
        self.right_child.push(!new_leaf);

        // Pack decision_type: default_left bit + missing_type (bits 2-3); the
        // categorical bit stays 0 (numerical split).
        let mut dt: i8 = 0;
        if default_left {
            dt |= DEFAULT_LEFT_MASK;
        }
        // C++ `Tree::SetMissingType` writes `(missing_type & 3) << 2`.
        dt |= (missing_type & 3) << 2;
        self.decision_type.push(dt);

        // Reassign the split leaf's output to the LEFT child; append the RIGHT
        // child leaf as a new leaf.
        self.leaf_value[leaf_u] = left_value;
        self.leaf_weight[leaf_u] = left_weight;
        self.leaf_count[leaf_u] = left_count;
        self.leaf_value.push(right_value);
        self.leaf_weight.push(right_weight);
        self.leaf_count.push(right_count);

        // Depth + parent bookkeeping: new leaf is one deeper than the split leaf.
        self.leaf_depth.push(self.leaf_depth[leaf_u] + 1);
        self.leaf_depth[leaf_u] += 1;
        // Both children's parent is the new node.
        self.leaf_parent[leaf_u] = new_node_idx as i32;
        self.leaf_parent.push(new_node_idx as i32);

        self.num_leaves += 1;
    }

    /// C++ `Tree::SplitCategorical` (`tree.cpp:77-98`) — grow `leaf` into an
    /// internal CATEGORICAL node with two child leaves.
    ///
    /// Mirrors [`split`](Self::split) structurally (the same `Tree::Split` core
    /// rewiring/leaf bookkeeping) but:
    /// - sets the `kCategoricalMask` bit in `decision_type` (numerical splits clear
    ///   it); `default_left` is NOT encoded (the categorical decision routes NaN /
    ///   negative categories RIGHT, the in-bitset categories LEFT);
    /// - stores `threshold = num_cat` (as f64) and `threshold_in_bin = num_cat`
    ///   (the index into `cat_boundaries`), then `++num_cat`;
    /// - appends the REAL category bitset (`cat_threshold_real`) into
    ///   `cat_threshold` and the running offset into `cat_boundaries`
    ///   (`cat_boundaries.back() + num_threshold`); the INNER (bin) bitset is used
    ///   only for the (already-completed) data-partition routing and is NOT
    ///   serialized.
    ///
    /// `cat_boundaries` is lazily seeded to `[0]` on the FIRST categorical split
    /// (C++ initializes `cat_boundaries_ = {0}` at construction). `missing_type` is
    /// the numeric C++ `MissingType` (0=None, 1=Zero, 2=NaN).
    #[allow(clippy::too_many_arguments)]
    pub fn split_categorical(
        &mut self,
        leaf: i32,
        feature: i32,
        real_feature: i32,
        cat_threshold_real: &[u32],
        left_value: f64,
        right_value: f64,
        left_count: i32,
        right_count: i32,
        left_weight: f64,
        right_weight: f64,
        gain: f32,
        missing_type: i8,
    ) {
        let new_node_idx = (self.num_leaves - 1) as usize;
        let leaf_u = leaf as usize;
        let new_leaf = self.num_leaves;

        // Rewire the parent's child pointer to the new internal node.
        let parent = self.leaf_parent[leaf_u];
        if parent >= 0 {
            let p = parent as usize;
            if self.right_child[p] == !leaf {
                self.right_child[p] = new_node_idx as i32;
            } else {
                self.left_child[p] = new_node_idx as i32;
            }
        }

        // Append the new internal node's split metadata. `threshold` /
        // `threshold_in_bin` both store `num_cat` (the categorical-node index into
        // `cat_boundaries`), per tree.cpp:85-86.
        self.split_feature_inner.push(feature);
        self.split_feature.push(real_feature);
        self.split_gain.push(gain);
        self.threshold_in_bin.push(self.num_cat as u32);
        self.threshold.push(self.num_cat as f64);
        self.internal_value.push(self.leaf_value[leaf_u]);
        self.internal_weight.push(0.0);
        self.internal_count.push(left_count + right_count);
        self.left_child.push(!leaf);
        self.right_child.push(!new_leaf);

        // Pack decision_type: CATEGORICAL bit set, default_left NOT set, plus the
        // missing_type bits (tree.cpp:82-84).
        let mut dt: i8 = 0;
        dt |= CATEGORICAL_MASK;
        dt |= (missing_type & 3) << 2;
        self.decision_type.push(dt);

        // Seed cat_boundaries = {0} on the first categorical split (C++ ctor
        // initializes cat_boundaries_ with a single 0).
        if self.cat_boundaries.is_empty() {
            self.cat_boundaries.push(0);
        }
        let new_boundary = self.cat_boundaries.last().copied().unwrap_or(0)
            + cat_threshold_real.len() as i32;
        self.cat_boundaries.push(new_boundary);
        self.cat_threshold.extend_from_slice(cat_threshold_real);
        self.num_cat += 1;

        // Reassign the split leaf's output to the LEFT child; append the RIGHT child.
        self.leaf_value[leaf_u] = left_value;
        self.leaf_weight[leaf_u] = left_weight;
        self.leaf_count[leaf_u] = left_count;
        self.leaf_value.push(right_value);
        self.leaf_weight.push(right_weight);
        self.leaf_count.push(right_count);

        // Depth + parent bookkeeping.
        self.leaf_depth.push(self.leaf_depth[leaf_u] + 1);
        self.leaf_depth[leaf_u] += 1;
        self.leaf_parent[leaf_u] = new_node_idx as i32;
        self.leaf_parent.push(new_node_idx as i32);

        self.num_leaves += 1;
    }

    /// C++ `Tree::ToString` (`tree.cpp:339-409`) — the byte-exact per-tree block.
    ///
    /// Section order + per-field formatter mode are load-bearing for DAT-09:
    /// - plain ints: `split_feature`, `left_child`, `right_child`, `leaf_count`,
    ///   `internal_count`, and `decision_type` (the C++ `int8_t -> int` cast,
    ///   Pitfall 3 — small positive values 0..15);
    /// - `format_g17` (`{:.17g}`): `threshold`, `leaf_value`, `leaf_weight`;
    /// - `format_g6` (`{:g}`): `split_gain`, `internal_value`, `internal_weight`,
    ///   and `shrinkage` (the ostream-default path; the golden is the arbiter).
    ///
    /// Ends with a trailing blank line (`tree.cpp:406`).
    pub fn to_string(&self) -> String {
        let mut s = String::new();
        let _ = writeln!(s, "num_leaves={}", self.num_leaves);
        let _ = writeln!(s, "num_cat={}", self.num_cat);
        let _ = writeln!(s, "split_feature={}", join_ints(&self.split_feature));
        let _ = writeln!(s, "split_gain={}", join_f32_g6(&self.split_gain));
        let _ = writeln!(s, "threshold={}", join_f64_g17(&self.threshold));
        let _ = writeln!(s, "decision_type={}", join_decision_type(&self.decision_type));
        let _ = writeln!(s, "left_child={}", join_ints(&self.left_child));
        let _ = writeln!(s, "right_child={}", join_ints(&self.right_child));
        let _ = writeln!(s, "leaf_value={}", join_f64_g17(&self.leaf_value));
        let _ = writeln!(s, "leaf_weight={}", join_f64_g17(&self.leaf_weight));
        let _ = writeln!(s, "leaf_count={}", join_ints(&self.leaf_count));
        let _ = writeln!(s, "internal_value={}", join_f64_g6(&self.internal_value));
        let _ = writeln!(s, "internal_weight={}", join_f64_g6(&self.internal_weight));
        let _ = writeln!(s, "internal_count={}", join_ints(&self.internal_count));
        if self.num_cat > 0 {
            let _ = writeln!(s, "cat_boundaries={}", join_ints(&self.cat_boundaries));
            let _ = writeln!(s, "cat_threshold={}", join_u32(&self.cat_threshold));
        }
        let _ = writeln!(s, "is_linear={}", if self.is_linear { 1 } else { 0 });
        let _ = writeln!(s, "shrinkage={}", format_g6(self.shrinkage));
        s.push('\n');
        s
    }

    /// Parse one tree block (the keyed, order-independent C++
    /// `Tree(const char*, size_t*)` ctor, `tree.cpp:685-866`).
    ///
    /// Reads up to [`MAX_HEADER_LINES`] `key=value` lines (stop at a blank line)
    /// into a map, then pulls fields by key with per-field fallbacks, honoring the
    /// single-leaf early return (`num_leaves<=1 && !is_linear`). All array lengths
    /// are validated against `num_leaves`/`num_cat` before any indexing
    /// (T-03-03). Linear-tree models are out of scope → [`ModelError`].
    pub fn parse(block: &str) -> Result<Tree, ModelError> {
        let mut kv: HashMap<&str, &str> = HashMap::new();
        for (i, line) in block.lines().enumerate() {
            if i >= MAX_HEADER_LINES {
                break;
            }
            if line.is_empty() || line.starts_with('\r') {
                break;
            }
            // Split on the FIRST '=' (values never contain '=' in a tree block).
            let Some((key, val)) = line.split_once('=') else {
                return Err(ModelError::MalformedModel {
                    detail: format!("tree block line without '=': {line:?}"),
                });
            };
            kv.insert(key, val.trim_end_matches('\r'));
        }

        let num_leaves: i32 = parse_required_int(&kv, "num_leaves")?;
        let num_cat: i32 = parse_required_int(&kv, "num_cat")?;

        let leaf_value = match kv.get("leaf_value") {
            Some(v) => parse_f64_list(v, "leaf_value")?,
            None => {
                return Err(ModelError::MalformedModel {
                    detail: "tree model should contain leaf_value field".to_string(),
                });
            }
        };

        let shrinkage: f64 = match kv.get("shrinkage") {
            Some(v) => parse_f64_scalar(v, "shrinkage")?,
            None => 1.0,
        };

        let is_linear = match kv.get("is_linear") {
            Some(v) => parse_int_scalar(v, "is_linear")? != 0,
            None => false,
        };

        if is_linear {
            return Err(ModelError::MalformedModel {
                detail: "linear-tree models (is_linear=1) are out of scope for Phase 3".to_string(),
            });
        }

        let leaf_count = match kv.get("leaf_count") {
            Some(v) => parse_i32_list(v, "leaf_count")?,
            None => vec![0i32; num_leaves.max(0) as usize],
        };

        // Validate the `num_leaves`-length arrays.
        check_len(&leaf_value, num_leaves, "leaf_value")?;
        check_len(&leaf_count, num_leaves, "leaf_count")?;

        // Single-leaf early return (`tree.cpp:747`).
        if num_leaves <= 1 {
            return Ok(Tree {
                num_leaves,
                num_cat,
                left_child: Vec::new(),
                right_child: Vec::new(),
                split_feature: Vec::new(),
                threshold: Vec::new(),
                decision_type: Vec::new(),
                split_gain: Vec::new(),
                leaf_value,
                leaf_weight: vec![0.0; num_leaves.max(0) as usize],
                leaf_count,
                internal_value: Vec::new(),
                internal_weight: Vec::new(),
                internal_count: Vec::new(),
                cat_boundaries: Vec::new(),
                cat_threshold: Vec::new(),
                shrinkage,
                is_linear,
                // Growth-time arrays: a freshly-loaded single-leaf tree has one
                // leaf at depth 0 with parent -1 and no splits.
                leaf_depth: vec![0i32; num_leaves.max(0) as usize],
                leaf_parent: vec![-1i32; num_leaves.max(0) as usize],
                split_feature_inner: Vec::new(),
                threshold_in_bin: Vec::new(),
            });
        }

        let n_internal = num_leaves - 1;

        let left_child = parse_required_i32_list(&kv, "left_child")?;
        let right_child = parse_required_i32_list(&kv, "right_child")?;
        let split_feature = parse_required_i32_list(&kv, "split_feature")?;
        let threshold = match kv.get("threshold") {
            Some(v) => parse_f64_list(v, "threshold")?,
            None => {
                return Err(ModelError::MalformedModel {
                    detail: "tree model should contain threshold field".to_string(),
                });
            }
        };

        let split_gain = match kv.get("split_gain") {
            Some(v) => parse_f32_list(v, "split_gain")?,
            None => vec![0.0f32; n_internal as usize],
        };
        let internal_count = match kv.get("internal_count") {
            Some(v) => parse_i32_list(v, "internal_count")?,
            None => vec![0i32; n_internal as usize],
        };
        let internal_value = match kv.get("internal_value") {
            Some(v) => parse_f64_list(v, "internal_value")?,
            None => vec![0.0f64; n_internal as usize],
        };
        let internal_weight = match kv.get("internal_weight") {
            Some(v) => parse_f64_list(v, "internal_weight")?,
            None => vec![0.0f64; n_internal as usize],
        };
        let leaf_weight = match kv.get("leaf_weight") {
            Some(v) => parse_f64_list(v, "leaf_weight")?,
            None => vec![0.0f64; num_leaves as usize],
        };
        let decision_type = match kv.get("decision_type") {
            Some(v) => parse_i8_list(v, "decision_type")?,
            None => vec![0i8; n_internal as usize],
        };

        // T-03-03: validate every `num_leaves-1`-length array.
        check_len(&left_child, n_internal, "left_child")?;
        check_len(&right_child, n_internal, "right_child")?;
        check_len(&split_feature, n_internal, "split_feature")?;
        check_len(&threshold, n_internal, "threshold")?;
        check_len(&split_gain, n_internal, "split_gain")?;
        check_len(&internal_count, n_internal, "internal_count")?;
        check_len(&internal_value, n_internal, "internal_value")?;
        check_len(&internal_weight, n_internal, "internal_weight")?;
        check_len(&decision_type, n_internal, "decision_type")?;
        check_len(&leaf_weight, num_leaves, "leaf_weight")?;

        let (cat_boundaries, cat_threshold) = if num_cat > 0 {
            let cb = match kv.get("cat_boundaries") {
                Some(v) => parse_i32_list(v, "cat_boundaries")?,
                None => {
                    return Err(ModelError::MalformedModel {
                        detail: "tree model should contain cat_boundaries field".to_string(),
                    });
                }
            };
            check_len(&cb, num_cat + 1, "cat_boundaries")?;
            let ct = match kv.get("cat_threshold") {
                Some(v) => parse_u32_list(v, "cat_threshold")?,
                None => {
                    return Err(ModelError::MalformedModel {
                        detail: "tree model should contain cat_threshold field".to_string(),
                    });
                }
            };
            // cat_threshold length is cat_boundaries.back() (C++ tree.cpp:860).
            let expected = *cb.last().unwrap_or(&0);
            check_len(&ct, expected, "cat_threshold")?;
            (cb, ct)
        } else {
            (Vec::new(), Vec::new())
        };

        // T-03-03: child/split indices must be in range before predict ever runs.
        for (i, &sf) in split_feature.iter().enumerate() {
            if sf < 0 {
                return Err(ModelError::MalformedModel {
                    detail: format!("split_feature[{i}]={sf} is negative"),
                });
            }
        }
        for (name, arr) in [("left_child", &left_child), ("right_child", &right_child)] {
            for (i, &c) in arr.iter().enumerate() {
                // child >= 0 is an internal node index; child < 0 is ~leaf.
                if c >= 0 && c >= n_internal {
                    return Err(ModelError::MalformedModel {
                        detail: format!("{name}[{i}]={c} is an out-of-range node index"),
                    });
                }
                let leaf = !c;
                if c < 0 && (leaf < 0 || leaf >= num_leaves) {
                    return Err(ModelError::MalformedModel {
                        detail: format!("{name}[{i}]={c} (~={leaf}) is an out-of-range leaf index"),
                    });
                }
            }
        }

        Ok(Tree {
            num_leaves,
            num_cat,
            left_child,
            right_child,
            split_feature,
            threshold,
            decision_type,
            split_gain,
            leaf_value,
            leaf_weight,
            leaf_count,
            internal_value,
            internal_weight,
            internal_count,
            cat_boundaries,
            cat_threshold,
            shrinkage,
            is_linear,
            // Growth-time arrays are NOT serialized, so a parsed model carries
            // default bookkeeping (depth 0 / parent -1 / inner-feature -1 /
            // bin-threshold 0). They are only populated by `split()` during
            // leaf-wise growth; predict/serialize never read them.
            leaf_depth: vec![0i32; num_leaves as usize],
            leaf_parent: vec![-1i32; num_leaves as usize],
            split_feature_inner: vec![-1i32; n_internal as usize],
            threshold_in_bin: vec![0u32; n_internal as usize],
        })
    }

    /// C++ `Tree::Shrinkage(double rate)` (`tree.h:188`).
    ///
    /// Multiplies every `leaf_value_` and `internal_value_` entry by `rate`,
    /// snapping each product to `+0.0` via [`maybe_round_to_zero`] (the C++
    /// `MaybeRoundToZero`, `tree.h:258`), and records `shrinkage_ = rate`. This is
    /// the per-tree learning-rate scaling applied in the GBDT loop BEFORE the score
    /// update (RESEARCH Pitfall 5 critical-ordering note).
    pub fn shrinkage(&mut self, rate: f64) {
        for v in &mut self.leaf_value {
            *v = maybe_round_to_zero(*v * rate);
        }
        for v in &mut self.internal_value {
            *v = maybe_round_to_zero(*v * rate);
        }
        self.shrinkage = rate;
    }

    /// C++ `Tree::SetLeafOutput(int leaf, double output)` (`tree.h:294`): overwrite
    /// a single leaf's stored output. Used by the leaf-refit path (ADV-06) to write
    /// the decay-blended output back into the existing tree structure. Unlike
    /// [`Self::shrinkage`] / [`Self::add_bias`] this does NOT snap to zero — C++
    /// `SetLeafOutput` is a raw assignment.
    pub fn set_leaf_output(&mut self, leaf: usize, output: f64) {
        self.leaf_value[leaf] = output;
    }

    /// C++ `SerialTreeLearner::FitByExistingTree` leaf-refit blend
    /// (`serial_tree_learner.cpp:268-276`), per leaf:
    /// `new_output = CalculateSplittedLeafOutput(sum_grad, kEpsilon + sum_hess, l1, l2)
    /// * shrinkage()`; then
    /// `SetLeafOutput(leaf, decay*old_output + (1-decay)*new_output)`.
    ///
    /// `decay` is `config_->refit_decay_rate` (config.h default 0.9, CHECK `[0,1]`).
    /// `sum_grad` / `sum_hess` are the per-leaf gradient / hessian sums over the rows
    /// that route to `leaf` (the C++ `data_partition_->GetIndexOnLeaf` accumulation,
    /// with `sum_hess` seeded by the `kEpsilon` start). `use_l1` / `l1` / `l2` mirror
    /// the leaf-output config (`lambda_l1`/`lambda_l2`). The tree's stored
    /// [`Self::shrinkage`] scales the fresh Newton output BEFORE the blend (so a
    /// shrunk model keeps its learning-rate scale on the refit slice).
    ///
    /// `decay = 1.0` leaves the leaf unchanged (all-old); `decay = 0.0` replaces it
    /// with the fresh shrunk output (all-new) — the two parity corners (07-RESEARCH
    /// §Oracle Axis Matrix → Refit).
    #[allow(clippy::too_many_arguments)]
    pub fn refit_leaf(
        &mut self,
        leaf: usize,
        sum_grad: f64,
        sum_hess: f64,
        decay: f64,
        use_l1: bool,
        l1: f64,
        l2: f64,
    ) {
        // C++ `double sum_hess = kEpsilon;` then accumulates — the caller passes the
        // raw row-sum, and we add kEpsilon here to match the C++ seed exactly.
        let hess = sum_hess + lgbm_core::types::K_EPSILON as f64;
        // `CalculateSplittedLeafOutput<USE_L1,false,false>` (feature_histogram.hpp,
        // mirrored bit-for-bit from `lgbm_compute::gain::calculate_splitted_leaf_output`
        // — inlined here so the model layer does not take a kernel-layer dependency
        // edge): `use_l1 ? -ThresholdL1(g,l1)/(h+l2) : -g/(h+l2)`.
        let output = if use_l1 {
            let reg_s = (sum_grad.abs() - l1).max(0.0);
            let sign = (sum_grad > 0.0) as i32 as f64 - (sum_grad < 0.0) as i32 as f64;
            -(sign * reg_s) / (hess + l2)
        } else {
            -sum_grad / (hess + l2)
        };
        let new_leaf_output = output * self.shrinkage;
        let old_leaf_output = self.leaf_value[leaf];
        self.leaf_value[leaf] = decay * old_leaf_output + (1.0 - decay) * new_leaf_output;
    }

    /// C++ `Tree::AddBias(double val)` (`tree.h:213`).
    ///
    /// Adds `val` to every `leaf_value_` and `internal_value_` entry (each snapped
    /// via [`maybe_round_to_zero`] / `MaybeRoundToZero`, `tree.h:258`). This is a
    /// MODEL-TEXT-only adjustment (the init-score baseline folded into the first
    /// tree) — it does NOT touch any score buffer (RESEARCH Pitfall 5). The C++
    /// `AddBias` does not change `shrinkage_`.
    pub fn add_bias(&mut self, val: f64) {
        for v in &mut self.leaf_value {
            *v = maybe_round_to_zero(*v + val);
        }
        for v in &mut self.internal_value {
            *v = maybe_round_to_zero(*v + val);
        }
    }

    /// C++ `Tree::AsConstantTree(double val, int count = 0)` (`tree.h:232-240`):
    /// build a degenerate 1-leaf tree whose single leaf output is the constant
    /// `val`, with `shrinkage_ = 1.0f` (forced) and `leaf_count_[0] = count`.
    ///
    /// This backs the GBDT loop's `class_need_train_[k] == false` (and the
    /// no-split) path (gbdt.cpp:419-434): instead of training, a constant tree is
    /// pushed so `models_.len()` stays `iter * num_tree_per_iteration` (Pitfall 6).
    /// The constant is the per-class init score (or 0 for the extend-with-zeros
    /// case). `num_cat = 0`, all split/internal arrays empty, one leaf at depth 0
    /// with parent -1 — exactly the C++ `Tree(2,…)` default reduced to a single
    /// leaf.
    ///
    /// `count` is `leaf_count_[0]` — the row count of the single leaf. C++ passes
    /// `num_data_` at BOTH constant-tree call sites (gbdt.cpp:430 for the init case
    /// and gbdt.cpp:433 for the extend-with-zeros case), so the caller MUST thread
    /// `num_data` here (the CR-01 fix; was hardcoded `0`). `Tree::to_string` ALWAYS
    /// emits `leaf_count=` (tree.cpp:363 — no single-leaf write-side early return),
    /// so a constant tree's serialized model text reads `leaf_count=<num_data>`,
    /// byte-matching the C++ golden (e.g. `leaf_count=12`).
    pub fn as_constant(constant_value: f64, count: i32) -> Tree {
        Tree {
            num_leaves: 1,
            num_cat: 0,
            left_child: Vec::new(),
            right_child: Vec::new(),
            split_feature: Vec::new(),
            threshold: Vec::new(),
            decision_type: Vec::new(),
            split_gain: Vec::new(),
            // The C++ AsConstantTree does NOT round-to-zero the value (it assigns
            // leaf_value_[0] = val directly); a near-zero init thus stays as-is in
            // the constant tree's single leaf.
            leaf_value: vec![constant_value],
            leaf_weight: vec![0.0],
            leaf_count: vec![count],
            internal_value: Vec::new(),
            internal_weight: Vec::new(),
            internal_count: Vec::new(),
            cat_boundaries: Vec::new(),
            cat_threshold: Vec::new(),
            // C++ forces shrinkage_ = 1.0f for a constant tree.
            shrinkage: 1.0,
            is_linear: false,
            leaf_depth: vec![0],
            leaf_parent: vec![-1],
            split_feature_inner: Vec::new(),
            threshold_in_bin: Vec::new(),
        }
    }
}

/// C++ `Tree::MaybeRoundToZero` (`tree.h:258`): `IsZero(fval) ? 0 : fval`.
///
/// Snaps a magnitude below `kZeroThreshold` (1e-35) to a NORMALIZED `+0.0`
/// (so a `-0.0` product can never leak into the model text — matches the C++
/// signed-zero behavior relied on by the learner-parity goldens).
#[inline]
fn maybe_round_to_zero(fval: f64) -> f64 {
    if is_zero(fval) {
        0.0 // normalized +0.0 (never -0.0)
    } else {
        fval
    }
}

// --- formatting helpers (ToString) ---

fn join_ints(v: &[i32]) -> String {
    let mut s = String::new();
    for (i, x) in v.iter().enumerate() {
        if i > 0 {
            s.push(' ');
        }
        let _ = write!(s, "{x}");
    }
    s
}

fn join_u32(v: &[u32]) -> String {
    let mut s = String::new();
    for (i, x) in v.iter().enumerate() {
        if i > 0 {
            s.push(' ');
        }
        let _ = write!(s, "{x}");
    }
    s
}

/// `decision_type` is emitted as the C++ `int8_t -> int` cast (Pitfall 3).
fn join_decision_type(v: &[i8]) -> String {
    let mut s = String::new();
    for (i, x) in v.iter().enumerate() {
        if i > 0 {
            s.push(' ');
        }
        let _ = write!(s, "{}", *x as i32);
    }
    s
}

fn join_f64_g17(v: &[f64]) -> String {
    let mut s = String::new();
    for (i, x) in v.iter().enumerate() {
        if i > 0 {
            s.push(' ');
        }
        s.push_str(&format_g17(*x));
    }
    s
}

fn join_f64_g6(v: &[f64]) -> String {
    let mut s = String::new();
    for (i, x) in v.iter().enumerate() {
        if i > 0 {
            s.push(' ');
        }
        s.push_str(&format_g6(*x));
    }
    s
}

fn join_f32_g6(v: &[f32]) -> String {
    let mut s = String::new();
    for (i, x) in v.iter().enumerate() {
        if i > 0 {
            s.push(' ');
        }
        s.push_str(&format_g6(*x as f64));
    }
    s
}

// --- parsing helpers ---

fn parse_required_int(kv: &HashMap<&str, &str>, key: &str) -> Result<i32, ModelError> {
    match kv.get(key) {
        Some(v) => parse_int_scalar(v, key),
        None => Err(ModelError::MalformedModel {
            detail: format!("tree model should contain {key} field"),
        }),
    }
}

fn parse_required_i32_list(kv: &HashMap<&str, &str>, key: &str) -> Result<Vec<i32>, ModelError> {
    match kv.get(key) {
        Some(v) => parse_i32_list(v, key),
        None => Err(ModelError::MalformedModel {
            detail: format!("tree model string format error, should contain {key} field"),
        }),
    }
}

fn parse_int_scalar(s: &str, key: &str) -> Result<i32, ModelError> {
    s.trim().parse::<i32>().map_err(|_| ModelError::MalformedModel {
        detail: format!("{key} is not a valid integer: {s:?}"),
    })
}

fn parse_f64_scalar(s: &str, key: &str) -> Result<f64, ModelError> {
    s.trim().parse::<f64>().map_err(|_| ModelError::MalformedModel {
        detail: format!("{key} is not a valid float: {s:?}"),
    })
}

fn parse_f64_list(s: &str, key: &str) -> Result<Vec<f64>, ModelError> {
    s.split_whitespace()
        .map(|t| {
            t.parse::<f64>().map_err(|_| ModelError::MalformedModel {
                detail: format!("{key} contains a non-float token: {t:?}"),
            })
        })
        .collect()
}

fn parse_f32_list(s: &str, key: &str) -> Result<Vec<f32>, ModelError> {
    s.split_whitespace()
        .map(|t| {
            t.parse::<f32>().map_err(|_| ModelError::MalformedModel {
                detail: format!("{key} contains a non-float token: {t:?}"),
            })
        })
        .collect()
}

fn parse_i32_list(s: &str, key: &str) -> Result<Vec<i32>, ModelError> {
    s.split_whitespace()
        .map(|t| {
            t.parse::<i32>().map_err(|_| ModelError::MalformedModel {
                detail: format!("{key} contains a non-integer token: {t:?}"),
            })
        })
        .collect()
}

fn parse_i8_list(s: &str, key: &str) -> Result<Vec<i8>, ModelError> {
    // C++ parses decision_type as int8_t from the int-cast text (range 0..15).
    s.split_whitespace()
        .map(|t| {
            t.parse::<i32>()
                .ok()
                .and_then(|v| i8::try_from(v).ok())
                .ok_or_else(|| ModelError::MalformedModel {
                    detail: format!("{key} contains a non-int8 token: {t:?}"),
                })
        })
        .collect()
}

fn parse_u32_list(s: &str, key: &str) -> Result<Vec<u32>, ModelError> {
    s.split_whitespace()
        .map(|t| {
            t.parse::<u32>().map_err(|_| ModelError::MalformedModel {
                detail: format!("{key} contains a non-u32 token: {t:?}"),
            })
        })
        .collect()
}

fn check_len<T>(arr: &[T], expected: i32, key: &str) -> Result<(), ModelError> {
    if expected < 0 || arr.len() != expected as usize {
        return Err(ModelError::MalformedModel {
            detail: format!(
                "{key} has {} entries, expected {expected} (from num_leaves/num_cat)",
                arr.len()
            ),
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decision_type_bit_decode() {
        // default-left bit (mask 2) set, categorical bit (mask 1) clear.
        assert!(get_decision_type(2, DEFAULT_LEFT_MASK));
        assert!(!get_decision_type(2, CATEGORICAL_MASK));
        assert!(get_decision_type(3, CATEGORICAL_MASK));
        // missing_type = (dt >> 2) & 3.
        assert_eq!(get_missing_type(0b0000_0010), 0); // None
        assert_eq!(get_missing_type(0b0000_0110), 1); // Zero
        assert_eq!(get_missing_type(0b0000_1010), 2); // NaN
    }

    /// A single-leaf root tree (mirrors the learner's `root_tree`) for growth tests.
    fn root_tree_for_test(root_output: f64, num_data: i32) -> Tree {
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

    #[test]
    fn split_categorical_sets_mask_and_threshold_num_cat() {
        let mut t = root_tree_for_test(0.0, 10);
        // Grow a categorical node on leaf 0, feature 2 with the REAL bitset for
        // categories {1, 3} (bits 1 and 3 set -> 0b1010). `Tree::split_categorical`
        // stores the bitset verbatim (the learner constructs it from category vals).
        let cat_bitset = vec![0b1010u32];
        t.split_categorical(0, 2, 2, &cat_bitset, -1.0, 2.0, 6, 4, 3.0, 2.0, 5.0, 0);
        assert_eq!(t.num_leaves, 2);
        assert_eq!(t.num_cat, 1);
        // CATEGORICAL_MASK bit set, threshold == num_cat (0 for the first cat node).
        assert!(get_decision_type(t.decision_type[0], CATEGORICAL_MASK));
        assert_eq!(t.threshold[0], 0.0, "threshold stores the cat index (0)");
        // cat_boundaries seeded [0] then pushed 0 + bitset length (1 block).
        assert_eq!(t.cat_boundaries, vec![0, 1]);
        assert_eq!(t.cat_threshold, vec![0b1010]);
    }

    #[test]
    fn grown_categorical_tree_round_trips() {
        let mut t = root_tree_for_test(0.0, 20);
        let cat_bitset = vec![0b100100u32]; // bits 2 and 5
        t.split_categorical(0, 1, 1, &cat_bitset, 1.5, -0.5, 12, 8, 4.0, 3.0, 7.0, 0);
        let s = t.to_string();
        // The model-text emits the num_cat>0 cat lines.
        assert!(s.contains("num_cat=1"), "model-text has num_cat=1:\n{s}");
        assert!(s.contains("cat_boundaries="), "has cat_boundaries:\n{s}");
        assert!(s.contains("cat_threshold="), "has cat_threshold:\n{s}");
        let parsed = Tree::parse(&s).expect("grown categorical tree round-trips");
        assert_eq!(parsed.to_string(), s, "byte-stable round-trip");
        assert_eq!(parsed.num_cat, 1);
        assert!(get_decision_type(parsed.decision_type[0], CATEGORICAL_MASK));
    }

    #[test]
    fn is_zero_uses_zero_threshold() {
        assert!(is_zero(0.0));
        assert!(is_zero(1e-36));
        assert!(is_zero(-1e-36));
        assert!(!is_zero(1e-30));
        assert!(!is_zero(0.5));
    }

    /// A tiny 2-leaf numerical tree: node 0 splits feature 0 at threshold 1.5,
    /// `<=` goes left (leaf 0, value 10.0), `>` goes right (leaf 1, value 20.0).
    fn tiny_tree() -> Tree {
        Tree {
            num_leaves: 2,
            num_cat: 0,
            left_child: vec![-1],  // ~(-1) = 0 -> leaf 0
            right_child: vec![-2], // ~(-2) = 1 -> leaf 1
            split_feature: vec![0],
            threshold: vec![1.5],
            decision_type: vec![2], // default-left, missing_type None
            split_gain: vec![0.0],
            leaf_value: vec![10.0, 20.0],
            leaf_weight: vec![1.0, 1.0],
            leaf_count: vec![1, 1],
            internal_value: vec![0.0],
            internal_weight: vec![0.0],
            internal_count: vec![2],
            cat_boundaries: vec![],
            cat_threshold: vec![],
            shrinkage: 1.0,
            is_linear: false,
            // Growth-time arrays are NOT serialized, so they carry the same
            // defaults the parser reconstructs (depth 0 / parent -1 / inner -1 /
            // bin 0). This keeps the parse round-trip struct-equality contract.
            leaf_depth: vec![0, 0],
            leaf_parent: vec![-1, -1],
            split_feature_inner: vec![-1],
            threshold_in_bin: vec![0],
        }
    }

    /// A single-leaf root tree at depth 0 — the growth starting point for the
    /// `Tree::split` unit test (mirrors a freshly-initialized C++ `Tree`).
    fn root_leaf_tree(root_value: f64) -> Tree {
        Tree {
            num_leaves: 1,
            num_cat: 0,
            left_child: vec![],
            right_child: vec![],
            split_feature: vec![],
            threshold: vec![],
            decision_type: vec![],
            split_gain: vec![],
            leaf_value: vec![root_value],
            leaf_weight: vec![10.0],
            leaf_count: vec![100],
            internal_value: vec![],
            internal_weight: vec![],
            internal_count: vec![],
            cat_boundaries: vec![],
            cat_threshold: vec![],
            shrinkage: 1.0,
            is_linear: false,
            leaf_depth: vec![0],
            leaf_parent: vec![-1],
            split_feature_inner: vec![],
            threshold_in_bin: vec![],
        }
    }

    #[test]
    fn numerical_decision_le_threshold_routes_left() {
        let t = tiny_tree();
        // fval <= threshold -> left_child (-1 -> leaf 0).
        assert_eq!(t.numerical_decision(1.0, 0), -1);
        assert_eq!(t.numerical_decision(1.5, 0), -1); // boundary is <=
        assert_eq!(t.numerical_decision(2.0, 0), -2);
    }

    #[test]
    fn numerical_decision_nan_coerced_to_zero_when_not_nan_type() {
        let t = tiny_tree(); // missing_type None
        // NaN coerced to 0.0, 0.0 <= 1.5 -> left.
        assert_eq!(t.numerical_decision(f64::NAN, 0), -1);
    }

    #[test]
    fn numerical_decision_zero_missing_routes_default_left() {
        let mut t = tiny_tree();
        // missing_type Zero (1) + default-left bit: dt = (1<<2)|2 = 6.
        t.decision_type = vec![6];
        // is_zero(0.0) && Zero-type -> default-left (left_child = -1).
        assert_eq!(t.numerical_decision(0.0, 0), -1);
        // a non-zero value falls through to the <= test.
        assert_eq!(t.numerical_decision(5.0, 0), -2);
    }

    #[test]
    fn get_leaf_and_predict() {
        let t = tiny_tree();
        assert_eq!(t.get_leaf(&[1.0]), 0);
        assert_eq!(t.get_leaf(&[9.0]), 1);
        assert_eq!(t.predict(&[1.0]), 10.0);
        assert_eq!(t.predict(&[9.0]), 20.0);
        assert_eq!(t.predict_leaf_index(&[9.0]), 1);
    }

    #[test]
    fn as_constant_builds_one_leaf_tree() {
        // C++ AsConstantTree: num_leaves=1, shrinkage=1.0, leaf_value[0]=val,
        // leaf_count_[0]=count (gbdt.cpp passes num_data_).
        let t = Tree::as_constant(0.7, 12);
        assert_eq!(t.num_leaves, 1);
        assert_eq!(t.shrinkage, 1.0);
        assert_eq!(t.leaf_value, vec![0.7]);
        assert_eq!(t.leaf_count, vec![12]);
        assert_eq!(t.num_cat, 0);
        assert!(t.split_feature.is_empty());
        // A 1-leaf tree predicts its constant for any input.
        assert_eq!(t.predict(&[1.0, 2.0, 3.0]), 0.7);
        assert_eq!(t.predict(&[]), 0.7);
        // Zero constant (the extend-with-zeros case).
        assert_eq!(Tree::as_constant(0.0, 12).predict(&[0.0]), 0.0);
    }

    #[test]
    fn round_trip_parse_to_string_byte_identical() {
        let t = tiny_tree();
        let block = t.to_string();
        let parsed = Tree::parse(&block).expect("parse tiny tree");
        let reemit = parsed.to_string();
        assert_eq!(block, reemit, "ToString round-trip must be byte-identical");
        assert_eq!(parsed, t, "parsed tree must equal the original");
    }

    #[test]
    fn round_trip_real_regression_block() {
        // A real regression Tree=0 block from the committed fixture (header only,
        // trailing blank line). Re-emit must be byte-identical.
        let block = "num_leaves=3\n\
            num_cat=0\n\
            split_feature=25 26\n\
            split_gain=82.9831 56.3449\n\
            threshold=1.0675000000000001 0.6695000000000001\n\
            decision_type=2 2\n\
            left_child=1 -2\n\
            right_child=-1 -3\n\
            leaf_value=0.52777142827851431 0.50789565188099872 0.50520825138320746\n\
            leaf_weight=306 322 277\n\
            leaf_count=306 322 277\n\
            internal_value=0.530857 0.537792\n\
            internal_weight=7000 4980\n\
            internal_count=7000 4980\n\
            is_linear=0\n\
            shrinkage=1\n\
            \n";
        let parsed = Tree::parse(block).expect("parse real block");
        assert_eq!(parsed.num_leaves, 3);
        assert_eq!(parsed.threshold[0], 1.0675000000000001);
        assert_eq!(parsed.to_string(), block, "real block round-trip byte-exact");
    }

    #[test]
    fn tree_split_grows_root_into_two_leaves() {
        // Grow a single-leaf root (value 5.0) into a 2-leaf tree splitting
        // feature 3 (inner) / real feature 7 at bin threshold 2 / value 1.5.
        let mut t = root_leaf_tree(5.0);
        t.split(
            0,    // leaf
            3,    // inner feature
            7,    // real feature
            2,    // threshold_in_bin
            1.5,  // threshold (real)
            10.0, // left_value
            20.0, // right_value
            40,   // left_count
            60,   // right_count
            4.0,  // left_weight
            6.0,  // right_weight
            42.0, // gain
            0,    // missing_type None
            true, // default_left
        );

        assert_eq!(t.num_leaves, 2);
        // C++ `~leaf` child encoding: left = ~0 = -1, right = ~1 = -2.
        assert_eq!(t.left_child[0], !0);
        assert_eq!(t.right_child[0], !1);
        // internal_value preserves the pre-split root leaf output.
        assert_eq!(t.internal_value[0], 5.0);
        // Leaf outputs reassigned: leaf 0 -> left, leaf 1 -> right.
        assert_eq!(t.leaf_value, vec![10.0, 20.0]);
        assert_eq!(t.leaf_weight, vec![4.0, 6.0]);
        assert_eq!(t.leaf_count, vec![40, 60]);
        // Both children are at depth 1.
        assert_eq!(t.leaf_depth, vec![1, 1]);
        // Growth-time split metadata.
        assert_eq!(t.split_feature, vec![7]);
        assert_eq!(t.split_feature_inner, vec![3]);
        assert_eq!(t.threshold, vec![1.5]);
        assert_eq!(t.threshold_in_bin, vec![2]);
        // decision_type: default_left (mask 2), missing_type None (0).
        assert_eq!(t.decision_type, vec![DEFAULT_LEFT_MASK]);
        // Both leaves now parented by node 0.
        assert_eq!(t.leaf_parent, vec![0, 0]);

        // Predict routes <= threshold left (10.0), > threshold right (20.0).
        assert_eq!(t.predict(&[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]), 10.0);
        assert_eq!(t.predict(&[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0]), 20.0);

        // Serialized form is byte-stable across two calls (growth arrays excluded).
        let s1 = t.to_string();
        let s2 = t.to_string();
        assert_eq!(s1, s2, "to_string must be byte-stable");
        // And the serialized block round-trips through the parser unchanged.
        let parsed = Tree::parse(&s1).expect("parse grown tree");
        assert_eq!(parsed.to_string(), s1, "grown-tree round-trip byte-exact");
        // The growth arrays must NOT appear in the serialized text.
        assert!(!s1.contains("leaf_depth"));
        assert!(!s1.contains("leaf_parent"));
        assert!(!s1.contains("split_feature_inner"));
        assert!(!s1.contains("threshold_in_bin"));
    }

    #[test]
    fn tree_split_missing_type_packs_into_decision_type() {
        // missing_type Zero (1) without default_left: dt = (1 << 2) = 4.
        let mut t = root_leaf_tree(0.0);
        t.split(0, 0, 0, 1, 0.5, -1.0, 1.0, 1, 1, 1.0, 1.0, 0.0, 1, false);
        assert_eq!(t.decision_type, vec![1 << 2]);
        assert_eq!(get_missing_type(t.decision_type[0]), 1);
        assert!(!get_decision_type(t.decision_type[0], DEFAULT_LEFT_MASK));
    }

    #[test]
    fn single_leaf_early_return() {
        let block = "num_leaves=1\n\
            num_cat=0\n\
            leaf_value=0.5\n\
            leaf_count=7000\n\
            is_linear=0\n\
            shrinkage=1\n\
            \n";
        let t = Tree::parse(block).expect("parse single-leaf");
        assert_eq!(t.num_leaves, 1);
        assert_eq!(t.leaf_value, vec![0.5]);
        assert!(t.left_child.is_empty());
        assert_eq!(t.predict(&[1.0, 2.0]), 0.5);
    }

    #[test]
    fn malformed_missing_num_leaves_is_err() {
        let block = "num_cat=0\nleaf_value=0.5\nis_linear=0\n\n";
        let err = Tree::parse(block).unwrap_err();
        assert!(matches!(err, ModelError::MalformedModel { .. }));
        assert!(err.to_string().contains("num_leaves"));
    }

    #[test]
    fn malformed_inconsistent_array_length_is_err() {
        // num_leaves=3 -> split_feature must have 2 entries; give 1.
        let block = "num_leaves=3\n\
            num_cat=0\n\
            split_feature=25\n\
            threshold=1.0 2.0\n\
            left_child=1 -2\n\
            right_child=-1 -3\n\
            leaf_value=0.1 0.2 0.3\n\
            leaf_count=1 1 1\n\
            is_linear=0\n\
            shrinkage=1\n\
            \n";
        let err = Tree::parse(block).unwrap_err();
        assert!(matches!(err, ModelError::MalformedModel { .. }));
        assert!(err.to_string().contains("split_feature"));
    }

    #[test]
    fn malformed_linear_tree_is_err() {
        let block = "num_leaves=2\n\
            num_cat=0\n\
            leaf_value=0.1 0.2\n\
            leaf_count=1 1\n\
            is_linear=1\n\
            shrinkage=1\n\
            \n";
        let err = Tree::parse(block).unwrap_err();
        assert!(matches!(err, ModelError::MalformedModel { .. }));
        assert!(err.to_string().contains("linear"));
    }

    #[test]
    fn does_not_panic_on_nan_feature() {
        let t = tiny_tree();
        let _ = t.predict(&[f64::NAN]);
        let _ = t.predict(&[f64::INFINITY]);
    }

    #[test]
    fn tree_shrinkage_scales_all_values_and_sets_field() {
        // C++ Tree::Shrinkage (tree.h:188): multiply every leaf + internal value
        // by `rate` and record shrinkage_ = rate.
        let mut t = tiny_tree();
        t.internal_value = vec![15.0];
        t.shrinkage(0.1);
        assert_eq!(t.leaf_value, vec![1.0, 2.0]);
        assert_eq!(t.internal_value, vec![1.5]);
        assert_eq!(t.shrinkage, 0.1);
    }

    #[test]
    fn tree_add_bias_adds_to_all_values_and_keeps_shrinkage() {
        // C++ Tree::AddBias (tree.h:213): add `val` to every leaf + internal value;
        // does NOT touch shrinkage_.
        let mut t = tiny_tree();
        t.internal_value = vec![5.0];
        let before = t.shrinkage;
        t.add_bias(3.0);
        assert_eq!(t.leaf_value, vec![13.0, 23.0]);
        assert_eq!(t.internal_value, vec![8.0]);
        assert_eq!(t.shrinkage, before);
    }

    #[test]
    fn tree_shrinkage_snaps_tiny_to_positive_zero() {
        // A product with |v| < kZeroThreshold (1e-35) snaps to +0.0 (signed-zero
        // normalized) — C++ MaybeRoundToZero (tree.h:258).
        let mut t = tiny_tree();
        t.leaf_value = vec![1e-30, -1e-30];
        t.internal_value = vec![-1e-30];
        t.shrinkage(1e-10); // products ~1e-40 < kZeroThreshold -> +0.0
        for &v in &t.leaf_value {
            assert_eq!(v, 0.0);
            assert!(v.is_sign_positive(), "must be +0.0, not -0.0");
        }
        assert_eq!(t.internal_value[0], 0.0);
        assert!(t.internal_value[0].is_sign_positive());
    }

    // --- TreeSHAP / predict_contrib (PRD-04) ---

    /// A 3-leaf numeric tree over 2 features:
    /// node 0 splits feature 0 at 1.5 (cover 100): left -> node 1, right -> leaf 2 (val 30).
    /// node 1 splits feature 1 at 0.5 (cover 60): left -> leaf 0 (val 10), right -> leaf 1 (val 20).
    fn two_feature_tree() -> Tree {
        Tree {
            num_leaves: 3,
            num_cat: 0,
            left_child: vec![1, -1],   // node0 left=node1; node1 left=~0=leaf0
            right_child: vec![-3, -2], // node0 right=~2=leaf2; node1 right=~1=leaf1
            split_feature: vec![0, 1],
            threshold: vec![1.5, 0.5],
            decision_type: vec![2, 2],
            split_gain: vec![0.0, 0.0],
            leaf_value: vec![10.0, 20.0, 30.0],
            leaf_weight: vec![1.0, 1.0, 1.0],
            leaf_count: vec![25, 35, 40],
            internal_value: vec![0.0, 0.0],
            internal_weight: vec![0.0, 0.0],
            internal_count: vec![100, 60], // root cover 100, node1 cover 60
            cat_boundaries: vec![],
            cat_threshold: vec![],
            shrinkage: 1.0,
            is_linear: false,
            leaf_depth: vec![0, 0, 0],
            leaf_parent: vec![-1, -1, -1],
            split_feature_inner: vec![-1, -1],
            threshold_in_bin: vec![0, 0],
        }
    }

    #[test]
    fn expected_value_is_cover_weighted_leaf_average() {
        let t = two_feature_tree();
        // (25/100)*10 + (35/100)*20 + (40/100)*30 = 2.5 + 7.0 + 12.0 = 21.5
        assert!((t.expected_value() - 21.5).abs() < 1e-12);
    }

    #[test]
    fn expected_value_single_leaf_is_leaf_output() {
        let t = Tree::as_constant(0.7, 12);
        assert_eq!(t.expected_value(), 0.7);
    }

    #[test]
    fn shap_invariant_sum_plus_base_equals_raw_numeric() {
        let t = two_feature_tree();
        let nf = 2usize;
        for row in [
            vec![0.0, 0.0], // f0<=1.5, f1<=0.5 -> leaf0 = 10
            vec![1.0, 1.0], // f0<=1.5, f1>0.5  -> leaf1 = 20
            vec![5.0, 0.0], // f0>1.5           -> leaf2 = 30
            vec![5.0, 9.0], // f0>1.5           -> leaf2 = 30
        ] {
            let mut out = vec![0.0f64; nf + 1];
            t.predict_contrib(&row, nf, &mut out);
            let sum: f64 = out.iter().sum();
            let raw = t.predict(&row);
            assert!(
                (sum - raw).abs() < 1e-9,
                "SHAP invariant: sum(contrib)+base {sum} != raw {raw} for row {row:?}, out={out:?}"
            );
            // base term is the expected value.
            assert!((out[nf] - t.expected_value()).abs() < 1e-12);
        }
    }

    /// A categorical tree: node 0 splits categorical feature 0 (cat_idx 0); the
    /// bitset {1,3} (0b1010) routes those categories LEFT (leaf 0 = -5), the rest
    /// RIGHT (leaf 1 = 7). The SHAP invariant must hold through `find_in_bitset`.
    fn categorical_tree() -> Tree {
        let mut t = Tree {
            num_leaves: 1,
            num_cat: 0,
            left_child: vec![],
            right_child: vec![],
            split_feature: vec![],
            threshold: vec![],
            decision_type: vec![],
            split_gain: vec![],
            leaf_value: vec![0.0],
            leaf_weight: vec![0.0],
            leaf_count: vec![100],
            internal_value: vec![],
            internal_weight: vec![],
            internal_count: vec![],
            cat_boundaries: vec![],
            cat_threshold: vec![],
            shrinkage: 1.0,
            is_linear: false,
            leaf_depth: vec![0],
            leaf_parent: vec![-1],
            split_feature_inner: vec![],
            threshold_in_bin: vec![],
        };
        // Grow a categorical split on feature 0; categories {1,3} go LEFT.
        t.split_categorical(0, 0, 0, &[0b1010u32], -5.0, 7.0, 40, 60, 1.0, 1.0, 0.0, 0);
        // internal_count[0] is set to left+right by split_categorical (100).
        t
    }

    #[test]
    fn shap_invariant_holds_on_categorical_tree() {
        let t = categorical_tree();
        let nf = 1usize;
        for cat in [0.0, 1.0, 2.0, 3.0, 4.0] {
            let row = vec![cat];
            let mut out = vec![0.0f64; nf + 1];
            t.predict_contrib(&row, nf, &mut out);
            let sum: f64 = out.iter().sum();
            let raw = t.predict(&row);
            assert!(
                (sum - raw).abs() < 1e-9,
                "cat SHAP invariant: sum {sum} != raw {raw} for cat {cat}, out={out:?}"
            );
        }
    }

    #[test]
    fn predict_contrib_accumulates_into_existing_output() {
        // The driver accumulates across trees: calling predict_contrib twice doubles
        // both the contributions and the base (each tree adds its own).
        let t = two_feature_tree();
        let nf = 2usize;
        let row = vec![1.0, 1.0];
        let mut out = vec![0.0f64; nf + 1];
        t.predict_contrib(&row, nf, &mut out);
        t.predict_contrib(&row, nf, &mut out);
        let sum: f64 = out.iter().sum();
        assert!((sum - 2.0 * t.predict(&row)).abs() < 1e-9);
    }

    #[test]
    fn tree_add_bias_snaps_tiny_to_positive_zero() {
        let mut t = tiny_tree();
        t.leaf_value = vec![0.0, 0.0];
        t.internal_value = vec![1e-40];
        // adding a tiny negative to a tiny positive keeps |v| < kZeroThreshold
        // (1e-35), so MaybeRoundToZero snaps to +0.0.
        t.add_bias(-1e-40);
        for &v in &t.leaf_value {
            assert_eq!(v, 0.0);
            assert!(v.is_sign_positive());
        }
        assert_eq!(t.internal_value[0], 0.0);
        assert!(t.internal_value[0].is_sign_positive());
    }

    #[test]
    fn set_leaf_output_raw_assign() {
        // SetLeafOutput is a raw assignment — no MaybeRoundToZero snap.
        let mut t = tiny_tree();
        t.set_leaf_output(1, 1e-40);
        assert_eq!(t.leaf_value[1], 1e-40);
        t.set_leaf_output(0, -3.5);
        assert_eq!(t.leaf_value[0], -3.5);
    }

    #[test]
    fn refit_leaf_decay_corners() {
        // ADV-06 leaf-refit blend: new_output = -sum_grad/(sum_hess+kEpsilon) *
        // shrinkage; leaf = decay*old + (1-decay)*new.
        // decay = 1.0 => unchanged (all-old).
        let mut t = tiny_tree(); // leaf_value = [10, 20], shrinkage = 1.0
        t.refit_leaf(0, /*g*/ 4.0, /*h*/ 2.0, /*decay*/ 1.0, false, 0.0, 0.0);
        assert_eq!(t.leaf_value[0], 10.0, "decay=1 must leave the leaf unchanged");

        // decay = 0.0 => all-new: -4/(2+kEps)*1.0 ≈ -2.0.
        let mut t = tiny_tree();
        t.refit_leaf(0, 4.0, 2.0, 0.0, false, 0.0, 0.0);
        assert!(
            (t.leaf_value[0] - (-4.0 / (2.0 + lgbm_core::types::K_EPSILON as f64))).abs() < 1e-9,
            "decay=0 must replace the leaf with the fresh shrunk output, got {}",
            t.leaf_value[0]
        );

        // Mid blend (decay = 0.5): 0.5*10 + 0.5*(-2.0) = 4.0 (within kEpsilon).
        let mut t = tiny_tree();
        t.refit_leaf(0, 4.0, 2.0, 0.5, false, 0.0, 0.0);
        let new = -4.0 / (2.0 + lgbm_core::types::K_EPSILON as f64);
        assert!((t.leaf_value[0] - (0.5 * 10.0 + 0.5 * new)).abs() < 1e-9);
    }

    #[test]
    fn refit_leaf_respects_shrinkage() {
        // A shrunk tree keeps its learning-rate scale on the refit slice.
        let mut t = tiny_tree();
        t.shrinkage = 0.1;
        t.refit_leaf(1, 4.0, 2.0, 0.0, false, 0.0, 0.0);
        let new = (-4.0 / (2.0 + lgbm_core::types::K_EPSILON as f64)) * 0.1;
        assert!((t.leaf_value[1] - new).abs() < 1e-9);
    }
}
