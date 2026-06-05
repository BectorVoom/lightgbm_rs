//! GBDT ensemble — faithful mirror of C++ `GBDT::models_` + the per-iteration /
//! per-class indexing and `InitPredict` sub-range math.
//!
//! Transcribed from:
//! - `LightGBM/src/boosting/gbdt.h` — `InitPredict` (426-435, the sub-range
//!   `start`/`num` clamp), `SubModelName="tree"` (466).
//! - `LightGBM/src/boosting/gbdt_prediction.cpp` — `PredictRaw` (13-32, the f64
//!   accumulation over `models_[i*ntpi + k]`).
//!
//! # The `num_tree_per_iteration` stride
//! `models_` is a FLAT `Vec<Tree>`; tree for iteration `i`, class `k` lives at
//! `trees[i * num_tree_per_iteration + k]` (`gbdt_prediction.cpp:21`).
//! `num_tree_per_iteration` is 1 for regression/binary and `num_class` for
//! multiclass — it drives BOTH the per-class output width and the stride.
//!
//! # Verbatim metadata (round-trip de-risk)
//! On a load→write round-trip the ensemble-level metadata strings
//! (`feature_names`, `feature_infos`, the `objective` line, `monotone_constraints`,
//! and the entire `parameters:`..EOF tail) are stored verbatim and re-emitted
//! UNCHANGED — never reformatted (RESEARCH Don't-Hand-Roll line 277). Only the
//! per-tree float arrays round-trip through parse→format.

use crate::tree::Tree;

/// C++ `GBDT::SubModelName()` (`gbdt.h:466`).
pub const SUB_MODEL_NAME: &str = "tree";
/// C++ `kModelVersion` (`gbdt_model_text.cpp:19`).
pub const MODEL_VERSION: &str = "v4";

/// An in-memory GBDT ensemble — the loaded `model.txt` (D-03).
///
/// Mirrors C++ `GBDT::models_` plus the envelope metadata captured verbatim for a
/// byte-exact round-trip.
#[derive(Debug, Clone, PartialEq)]
pub struct GbdtModel {
    /// C++ `models_` — flat tree list, indexed `[i*num_tree_per_iteration + k]`.
    pub trees: Vec<Tree>,
    /// C++ `num_class_`.
    pub num_class: i32,
    /// C++ `num_tree_per_iteration_` (stride).
    pub num_tree_per_iteration: i32,
    /// C++ `label_idx_`.
    pub label_index: i32,
    /// C++ `max_feature_idx_` — predict-row width is `max_feature_idx + 1`.
    pub max_feature_idx: i32,
    /// C++ `average_output_` (RF flag; emitted as a bare `average_output` line).
    pub average_output: bool,
    /// Verbatim `objective=` line value (e.g. `regression`, `binary sigmoid:1`).
    /// `None` when the model had no `objective=` line.
    pub objective_string: Option<String>,
    /// Verbatim `feature_names=` value (space-joined).
    pub feature_names: String,
    /// Verbatim `feature_infos=` value (space-joined `[min:max]`/`cat:cat`).
    pub feature_infos: String,
    /// Verbatim `monotone_constraints=` value, when present.
    pub monotone_constraints: Option<String>,
    /// Everything from the `parameters:` header to end of file, captured VERBATIM
    /// (includes `end of parameters` and any trailing `pandas_categorical:null` /
    /// `parser:` block). Re-emitted unchanged. `None` when the model had no
    /// `parameters:` block.
    pub trailer: Option<String>,
}

impl GbdtModel {
    /// The number of boosting iterations stored (`models_.size() / ntpi`).
    pub fn num_iteration(&self) -> i32 {
        if self.num_tree_per_iteration <= 0 {
            return 0;
        }
        self.trees.len() as i32 / self.num_tree_per_iteration
    }

    /// C++ `GBDT::InitPredict` (`gbdt.h:426-435`) — resolve the predict sub-range.
    /// Returns `(start_iteration_for_pred, num_iteration_for_pred)`.
    pub fn init_predict(&self, start_iteration: i32, num_iteration: i32) -> (i32, i32) {
        let total = self.num_iteration();
        let start = start_iteration.max(0).min(total);
        let num = if num_iteration > 0 {
            num_iteration.min(total - start)
        } else {
            total - start
        };
        (start, num)
    }

    /// C++ `GBDT::PredictRaw` (`gbdt_prediction.cpp:13-32`): accumulate each
    /// tree's `Predict` into the per-class `output[k]` in **f64**, over the
    /// resolved sub-range. `features` is the RAW row buffer of width
    /// `max_feature_idx + 1`. Returns a `num_tree_per_iteration`-wide vector.
    pub fn predict_raw(&self, features: &[f64], start_iteration: i32, num_iteration: i32) -> Vec<f64> {
        let ntpi = self.num_tree_per_iteration.max(0) as usize;
        let mut output = vec![0.0f64; ntpi];
        let (start, num) = self.init_predict(start_iteration, num_iteration);
        let end = start + num;
        for i in start..end {
            for k in 0..ntpi {
                let idx = i as usize * ntpi + k;
                output[k] += self.trees[idx].predict(features);
            }
        }
        output
    }

    /// C++ `GBDT::FeatureImportance` (split-count, `gbdt_model_text.cpp:627`):
    /// the number of splits on each ORIGINAL feature index, across all stored
    /// trees. Index `f` of the returned vector is feature `f`'s split count; the
    /// vector length is `max_feature_idx + 1`.
    pub fn feature_importance_split_count(&self) -> Vec<u64> {
        let n = (self.max_feature_idx + 1).max(0) as usize;
        let mut counts = vec![0u64; n];
        for tree in &self.trees {
            for &sf in &tree.split_feature {
                if sf >= 0 && (sf as usize) < n {
                    counts[sf as usize] += 1;
                }
            }
        }
        counts
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stump(value0: f64, value1: f64, feat: i32, thr: f64) -> Tree {
        Tree {
            num_leaves: 2,
            num_cat: 0,
            left_child: vec![-1],
            right_child: vec![-2],
            split_feature: vec![feat],
            threshold: vec![thr],
            decision_type: vec![2],
            split_gain: vec![0.0],
            leaf_value: vec![value0, value1],
            leaf_weight: vec![1.0, 1.0],
            leaf_count: vec![1, 1],
            internal_value: vec![0.0],
            internal_weight: vec![0.0],
            internal_count: vec![2],
            cat_boundaries: vec![],
            cat_threshold: vec![],
            shrinkage: 1.0,
            is_linear: false,
        }
    }

    fn two_tree_regression() -> GbdtModel {
        GbdtModel {
            trees: vec![stump(1.0, 2.0, 0, 0.5), stump(0.1, 0.2, 1, 0.5)],
            num_class: 1,
            num_tree_per_iteration: 1,
            label_index: 0,
            max_feature_idx: 1,
            average_output: false,
            objective_string: Some("regression".to_string()),
            feature_names: "Column_0 Column_1".to_string(),
            feature_infos: "[0:1] [0:1]".to_string(),
            monotone_constraints: None,
            trailer: None,
        }
    }

    #[test]
    fn num_iteration_uses_stride() {
        let m = two_tree_regression();
        assert_eq!(m.num_iteration(), 2);
    }

    #[test]
    fn init_predict_clamps_and_all() {
        let m = two_tree_regression();
        // num_iteration = -1 -> all remaining.
        assert_eq!(m.init_predict(0, -1), (0, 2));
        // start clamps into [0, total].
        assert_eq!(m.init_predict(5, -1), (2, 0));
        // num_iteration > 0 capped by remaining.
        assert_eq!(m.init_predict(1, 5), (1, 1));
        assert_eq!(m.init_predict(0, 1), (0, 1));
    }

    /// A 5-iteration single-class model (`total = 5`) for the full `<behavior>`
    /// `init_predict` clamp/slice battery (Task 1 Tests 1-4).
    fn five_iter_regression() -> GbdtModel {
        GbdtModel {
            trees: vec![
                stump(0.0, 1.0, 0, 0.5),
                stump(0.0, 2.0, 0, 0.5),
                stump(0.0, 4.0, 0, 0.5),
                stump(0.0, 8.0, 0, 0.5),
                stump(0.0, 16.0, 0, 0.5),
            ],
            num_class: 1,
            num_tree_per_iteration: 1,
            label_index: 0,
            max_feature_idx: 1,
            average_output: false,
            objective_string: Some("regression".to_string()),
            feature_names: "Column_0 Column_1".to_string(),
            feature_infos: "[0:1] [0:1]".to_string(),
            monotone_constraints: None,
            trailer: None,
        }
    }

    /// Task 1 Test 1 (`-1 == all`): `init_predict(0, -1)` -> `(0, total)`.
    #[test]
    fn init_predict_minus_one_is_all() {
        let m = five_iter_regression();
        let total = m.num_iteration(); // 5
        assert_eq!(m.init_predict(0, -1), (0, total));
        // `0` is also treated as "all remaining" per the C++ `num_iteration > 0` test.
        assert_eq!(m.init_predict(0, 0), (0, total));
    }

    /// Task 1 Test 2 (bounded count): `init_predict(0, 3)` -> `(0, min(3, total))`.
    #[test]
    fn init_predict_bounded_count() {
        let m = five_iter_regression();
        assert_eq!(m.init_predict(0, 3), (0, 3));
        // A count larger than total is capped to total.
        assert_eq!(m.init_predict(0, 99), (0, 5));
    }

    /// Task 1 Test 3 (non-zero start): `init_predict(2, -1)` -> `(2, total-2)`;
    /// `init_predict(2, 3)` -> `(2, min(3, total-2))`.
    #[test]
    fn init_predict_non_zero_start() {
        let m = five_iter_regression(); // total = 5
        assert_eq!(m.init_predict(2, -1), (2, 3));
        assert_eq!(m.init_predict(2, 3), (2, 3)); // min(3, 5-2) = 3
        assert_eq!(m.init_predict(2, 99), (2, 3)); // min(99, 3) = 3
        assert_eq!(m.init_predict(4, 99), (4, 1)); // min(99, 1) = 1
    }

    /// Task 1 Test 4 (clamp): `init_predict(total+5, -1)` -> `(total, 0)` (start
    /// clamped to total; empty slice; never indexes OOB, never panics — T-03-12).
    #[test]
    fn init_predict_over_range_clamps_to_empty() {
        let m = five_iter_regression(); // total = 5
        assert_eq!(m.init_predict(10, -1), (5, 0));
        assert_eq!(m.init_predict(5, 3), (5, 0));
        // Extreme / negative values clamp, never panic.
        assert_eq!(m.init_predict(i32::MAX, -1), (5, 0));
        assert_eq!(m.init_predict(-7, -1), (0, 5)); // negative start -> 0
        assert_eq!(m.init_predict(i32::MIN, i32::MIN), (0, 5));
        // Predict over the empty slice yields the zero accumulator (no panic).
        assert_eq!(m.predict_raw(&[1.0, 0.0], 10, -1), vec![0.0]);
    }

    #[test]
    fn predict_raw_accumulates_f64() {
        let m = two_tree_regression();
        // feature 0 = 1.0 (>0.5 -> tree0 leaf1 = 2.0); feature 1 = 0.0 (<=0.5 -> tree1 leaf0 = 0.1).
        let out = m.predict_raw(&[1.0, 0.0], 0, -1);
        assert_eq!(out.len(), 1);
        assert!((out[0] - 2.1).abs() < 1e-12);
        // sub-range: only tree 0.
        let out1 = m.predict_raw(&[1.0, 0.0], 0, 1);
        assert!((out1[0] - 2.0).abs() < 1e-12);
    }

    /// Task 1 Test 5 (slice accumulation): `predict_raw(row, 1, 2)` accumulates
    /// ONLY `trees[i*ntpi+k]` for `i in 1..3` — assert it equals the manual sum of
    /// exactly those trees, and differs from the full-range result.
    #[test]
    fn predict_raw_slice_accumulates_only_selected_iterations() {
        let m = five_iter_regression();
        // All leaves route feature-0=1.0 (>0.5) to leaf1 = {1,2,4,8,16}.
        let row = [1.0f64, 0.0];
        let ntpi = m.num_tree_per_iteration as usize;

        // Sub-range (start=1, num=2) -> iterations 1 and 2 only: 2.0 + 4.0 = 6.0.
        let sub = m.predict_raw(&row, 1, 2);
        // Manual sum of exactly trees[1*ntpi+0] and trees[2*ntpi+0].
        let manual: f64 = (1..3)
            .map(|i| m.trees[i * ntpi].predict(&row))
            .sum();
        assert!((sub[0] - manual).abs() < 1e-12);
        assert!((sub[0] - 6.0).abs() < 1e-12);

        // Full range = 1+2+4+8+16 = 31, which must differ from the slice.
        let full = m.predict_raw(&row, 0, -1);
        assert!((full[0] - 31.0).abs() < 1e-12);
        assert!((full[0] - sub[0]).abs() > 1e-9, "slice must differ from full range");
    }

    #[test]
    fn feature_importance_split_count() {
        let m = two_tree_regression();
        // tree0 splits feature 0, tree1 splits feature 1.
        assert_eq!(m.feature_importance_split_count(), vec![1, 1]);
    }
}
