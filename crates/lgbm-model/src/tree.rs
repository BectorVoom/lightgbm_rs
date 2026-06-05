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
}
