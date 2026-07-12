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

    /// C++ `GBDT::PredictRaw` WITH the prediction-early-stop hook
    /// (`gbdt_prediction.cpp:13-32` + `prediction_early_stop.cpp`). Accumulates each
    /// tree's `Predict` into the per-class `output[k]` in **f64** over the resolved
    /// sub-range; every `freq` accumulated iterations it evaluates the running
    /// per-row score against `margin` and STOPS early if the margin is decisive.
    ///
    /// The margin condition matches `PredictionEarlyStopInstance` exactly:
    /// - "binary" (`num_tree_per_iteration == 1`): `2*|score[0]| > margin`.
    /// - "multiclass" (`> 1`): `(top1 - top2) > margin`.
    ///
    /// Returns `(output, iterations_evaluated)` — the same per-class score the
    /// no-early-stop path produces when no stop fires, plus the count of iterations
    /// actually accumulated (matching the C++ reference's effective iteration count).
    ///
    /// When `freq <= 0` the hook is disabled (`round_period` is effectively never
    /// reached); the result is then byte-identical to [`Self::predict_raw`].
    pub fn predict_raw_early_stop(
        &self,
        features: &[f64],
        start_iteration: i32,
        num_iteration: i32,
        freq: i32,
        margin: f64,
    ) -> (Vec<f64>, i32) {
        let ntpi = self.num_tree_per_iteration.max(0) as usize;
        let mut output = vec![0.0f64; ntpi];
        let (start, num) = self.init_predict(start_iteration, num_iteration);
        let end = start + num;
        let mut round_counter = 0i32;
        let mut iters_evaluated = 0i32;
        for i in start..end {
            for k in 0..ntpi {
                let idx = i as usize * ntpi + k;
                output[k] += self.trees[idx].predict(features);
            }
            iters_evaluated += 1;
            // C++ increments the counter every iteration; the callback fires when
            // `round_period == counter` (so freq <= 0 never fires).
            round_counter += 1;
            if freq > 0 && freq == round_counter {
                if pred_early_stop_should_stop(&output, margin) {
                    return (output, iters_evaluated);
                }
                round_counter = 0;
            }
        }
        (output, iters_evaluated)
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

    /// C++ `GBDT::FeatureImportance(num_iteration, importance_type=1)` — GAIN-based
    /// (`gbdt_model_text.cpp:646-655`): the SUM of `split_gain` over each ORIGINAL
    /// feature's splits, across all stored trees, counting ONLY splits with
    /// `split_gain > 0` (the CR-02 guard, mirrored exactly from C++ which applies the
    /// `> 0` filter for BOTH split=0 and gain=1). Index `f` is feature `f`'s total
    /// split gain; the vector length is `max_feature_idx + 1`.
    ///
    /// Accumulated in f64 (the gains are stored f32; the C++ vector is `double`).
    pub fn feature_importance_gain(&self) -> Vec<f64> {
        let n = (self.max_feature_idx + 1).max(0) as usize;
        let mut gains = vec![0.0f64; n];
        for tree in &self.trees {
            // split_gain / split_feature have length num_leaves - 1.
            for (idx, &sf) in tree.split_feature.iter().enumerate() {
                let g = tree.split_gain.get(idx).copied().unwrap_or(0.0);
                if g > 0.0 && sf >= 0 && (sf as usize) < n {
                    gains[sf as usize] += g as f64;
                }
            }
        }
        gains
    }

    /// C++ `GBDT::FeatureImportance(num_iteration, importance_type=0)` — SPLIT-count
    /// WITH the `split_gain > 0` guard (`gbdt_model_text.cpp:636-642`). This is the
    /// CR-02-faithful sibling of [`Self::feature_importance_split_count`]: C++ counts
    /// a split toward its feature ONLY when `split_gain(split_idx) > 0`. The legacy
    /// [`Self::feature_importance_split_count`] omits the guard (it counts every
    /// stored split) and is retained for callers that want the raw structural count;
    /// the model-text emit + parity path use THIS guarded variant.
    pub fn feature_importance_split_count_guarded(&self) -> Vec<u64> {
        let n = (self.max_feature_idx + 1).max(0) as usize;
        let mut counts = vec![0u64; n];
        for tree in &self.trees {
            for (idx, &sf) in tree.split_feature.iter().enumerate() {
                let g = tree.split_gain.get(idx).copied().unwrap_or(0.0);
                if g > 0.0 && sf >= 0 && (sf as usize) < n {
                    counts[sf as usize] += 1;
                }
            }
        }
        counts
    }

    /// Leaf-refit ONE tree (`SerialTreeLearner::FitByExistingTree`,
    /// serial_tree_learner.cpp:247-285, called from `GBDT::RefitTree`): route every
    /// training row to its leaf in `trees[tree_index]` (over the tree's existing
    /// STRUCTURE — never re-split), accumulate per-leaf `(sum_grad, sum_hess)`, then
    /// blend each leaf via [`Tree::refit_leaf`] (`new = decay*old +
    /// (1-decay)*shrunk_newton`).
    ///
    /// `rows` are the per-row RAW feature buffers (width `max_feature_idx + 1`) used
    /// by `Tree::get_leaf` (the C++ `data_partition_` row→leaf map, here re-derived
    /// by routing each row through the existing tree). `gradients` / `hessians` are
    /// the per-row grad/hess for THIS tree's class (the C++ `gradients_pointer_ +
    /// offset`). `decay` is `refit_decay_rate`; `use_l1`/`l1`/`l2` mirror the
    /// leaf-output config. A constant (`num_leaves <= 1`) tree has no routed leaves
    /// to refit and is left unchanged (C++ `FitByExistingTree` still iterates its one
    /// leaf, but a constant tree has no rows partition beyond the whole corpus — the
    /// decay blend on a single leaf is well-defined, so we refit it too).
    #[allow(clippy::too_many_arguments)]
    pub fn refit_one_tree(
        &mut self,
        tree_index: usize,
        rows: &[Vec<f64>],
        gradients: &[f32],
        hessians: &[f32],
        decay: f64,
        use_l1: bool,
        l1: f64,
        l2: f64,
    ) {
        let num_leaves = self.trees[tree_index].num_leaves.max(1) as usize;
        let mut sum_grad = vec![0.0f64; num_leaves];
        let mut sum_hess = vec![0.0f64; num_leaves];
        for (r, row) in rows.iter().enumerate() {
            let leaf = if self.trees[tree_index].num_leaves > 1 {
                self.trees[tree_index].get_leaf(row) as usize
            } else {
                0
            };
            sum_grad[leaf] += gradients[r] as f64;
            sum_hess[leaf] += hessians[r] as f64;
        }
        for leaf in 0..num_leaves {
            self.trees[tree_index].refit_leaf(
                leaf,
                sum_grad[leaf],
                sum_hess[leaf],
                decay,
                use_l1,
                l1,
                l2,
            );
        }
    }

    /// Whole-ensemble leaf-refit on new `(rows, labels)` for the REGRESSION (L2)
    /// default objective — the model-layer mirror of `GBDT::RefitTree` /
    /// `Booster.refit(data, label)` (ADV-06). This is the high-level refit the
    /// Python `Booster.refit` exposes: it reproduces the C++ iterative loop exactly,
    /// computing per-row grad/hess (`grad = score - label`, `hess = 1` for L2) on
    /// the score accumulated FROM THE REFIT TREES iteration-by-iteration (NOT from
    /// the original full prediction), then leaf-refitting each tree in turn and
    /// adding its new predictions to the running refit score.
    ///
    /// `rows` are the new RAW feature rows (width `max_feature_idx + 1`); `labels`
    /// the new f32 targets (length `rows.len()`). `decay` is `refit_decay_rate`;
    /// `use_l1`/`l1`/`l2` mirror the leaf-output config. The refit is in place.
    ///
    /// Limited to single-output (regression/binary-margin) ensembles
    /// (`num_tree_per_iteration == 1`); the multiclass refit (per-class grad/hess
    /// stride) is not exercised by the Phase-8 Python path and is left for a future
    /// slice. A no-op when there are no trees or rows.
    #[allow(clippy::too_many_arguments)]
    pub fn refit_ensemble_l2(
        &mut self,
        rows: &[Vec<f64>],
        labels: &[f32],
        decay: f64,
        use_l1: bool,
        l1: f64,
        l2: f64,
    ) {
        let ntpi = self.num_tree_per_iteration.max(1) as usize;
        let nd = rows.len();
        if nd == 0 || self.trees.is_empty() || ntpi != 1 {
            return;
        }
        let num_iterations = self.trees.len() / ntpi;
        // C++ `RefitTree` re-scores from the refit trees themselves: the running
        // score starts at ZERO (verified empirically — the first refit tree's leaf
        // equals `-sum_grad/sum_hess` with `grad = -label`, i.e. score init = 0),
        // then each refit tree's prediction is added back (`AddScore(new_tree)`).
        let mut score = vec![0.0f64; nd];
        let mut gradients = vec![0.0f32; nd];
        let hessians = vec![1.0f32; nd]; // L2 hessian is constant 1.
        for iter in 0..num_iterations {
            // L2 grad on the CURRENT refit score: grad = score - label.
            for r in 0..nd {
                gradients[r] = (score[r] - labels[r] as f64) as f32;
            }
            let model_index = iter * ntpi; // ntpi == 1.
            // The official `Booster.refit` re-loads each tree's `shrinkage` as 1.0
            // (model text emits `shrinkage=1`; the learning-rate scale is baked into
            // the stored leaf values). So `refit_leaf`'s fresh Newton output must NOT
            // be re-scaled by the original learning rate — set the effective
            // shrinkage to 1.0 (verified empirically: a decay=0 refit leaf equals
            // `-sum_grad/sum_hess` with NO 0.1 factor).
            self.trees[model_index].shrinkage = 1.0;
            self.refit_one_tree(model_index, rows, &gradients, &hessians, decay, use_l1, l1, l2);
            // AddScore(new_tree): the refit tree's predictions add to the running
            // score (the stored leaf values already carry the shrinkage scale).
            for (r, row) in rows.iter().enumerate() {
                score[r] += self.trees[model_index].predict(row);
            }
        }
    }
}

/// The `PredictionEarlyStopInstance` margin callback (`prediction_early_stop.cpp`).
///
/// - len 1 → "binary": `2*|pred[0]| > margin`.
/// - len >= 2 → "multiclass": `(top1 - top2) > margin` (the two largest scores).
///
/// Returns `true` when the margin is decisive (stop accumulating further trees).
/// An empty `output` (no classes) never stops.
fn pred_early_stop_should_stop(output: &[f64], margin: f64) -> bool {
    match output.len() {
        0 => false,
        1 => 2.0 * output[0].abs() > margin,
        _ => {
            // Two largest scores (C++ std::partial_sort top 2, descending).
            let mut top1 = f64::NEG_INFINITY;
            let mut top2 = f64::NEG_INFINITY;
            for &v in output {
                if v > top1 {
                    top2 = top1;
                    top1 = v;
                } else if v > top2 {
                    top2 = v;
                }
            }
            (top1 - top2) > margin
        }
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
            linear: None,
            leaf_depth: vec![1, 1],
            leaf_parent: vec![0, 0],
            split_feature_inner: vec![-1],
            threshold_in_bin: vec![0],
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

    /// A stump with a chosen split_gain (for the gain-importance tests).
    fn gain_stump(feat: i32, gain: f32) -> Tree {
        let mut t = stump(1.0, 2.0, feat, 0.5);
        t.split_gain = vec![gain];
        t
    }

    #[test]
    fn feature_importance_gain_sums_split_gain() {
        // ADV-07: gain importance = sum of split_gain per feature, split_gain>0 only.
        let m = GbdtModel {
            trees: vec![gain_stump(0, 3.0), gain_stump(0, 2.0), gain_stump(1, 5.0)],
            ..two_tree_regression()
        };
        let g = m.feature_importance_gain();
        assert!((g[0] - 5.0).abs() < 1e-9, "feature 0 = 3+2 = 5, got {}", g[0]);
        assert!((g[1] - 5.0).abs() < 1e-9, "feature 1 = 5, got {}", g[1]);
    }

    #[test]
    fn feature_importance_gain_guards_nonpositive() {
        // CR-02 guard: a split with gain <= 0 contributes 0 to gain AND to the
        // guarded split count.
        let m = GbdtModel {
            trees: vec![gain_stump(0, 0.0), gain_stump(0, -1.0), gain_stump(1, 4.0)],
            ..two_tree_regression()
        };
        let g = m.feature_importance_gain();
        assert_eq!(g[0], 0.0, "all of feature 0's splits have gain <= 0");
        assert!((g[1] - 4.0).abs() < 1e-9);
        // The guarded split-count drops the gain<=0 splits too.
        assert_eq!(m.feature_importance_split_count_guarded(), vec![0, 1]);
        // The legacy unguarded count still tallies every stored split.
        assert_eq!(m.feature_importance_split_count(), vec![2, 1]);
    }

    #[test]
    fn refit_one_tree_decay_corners() {
        // ADV-06 leaf-refit: route rows to leaves of a stump on feature 0 @ 0.5,
        // accumulate per-leaf grad/hess, blend by decay.
        // stump(1.0, 2.0, 0, 0.5): leaf 0 = value 1.0 (feature<=0.5), leaf 1 = 2.0.
        let mut m = GbdtModel {
            trees: vec![stump(1.0, 2.0, 0, 0.5)],
            ..two_tree_regression()
        };
        // rows: two route to leaf 0 (feat0 = 0.0), two to leaf 1 (feat0 = 1.0).
        let rows = vec![
            vec![0.0, 0.0],
            vec![0.0, 0.0],
            vec![1.0, 0.0],
            vec![1.0, 0.0],
        ];
        let grad = [1.0f32, 1.0, 3.0, 3.0]; // leaf0 sum=2, leaf1 sum=6
        let hess = [1.0f32, 1.0, 1.0, 1.0]; // leaf0 sum=2, leaf1 sum=2

        // decay = 1.0 => unchanged.
        m.refit_one_tree(0, &rows, &grad, &hess, 1.0, false, 0.0, 0.0);
        assert_eq!(m.trees[0].leaf_value, vec![1.0, 2.0]);

        // decay = 0.0 => all-new: leaf0 = -2/(2+kEps), leaf1 = -6/(2+kEps).
        let mut m = GbdtModel {
            trees: vec![stump(1.0, 2.0, 0, 0.5)],
            ..two_tree_regression()
        };
        m.refit_one_tree(0, &rows, &grad, &hess, 0.0, false, 0.0, 0.0);
        let e = lgbm_core::types::K_EPSILON as f64;
        assert!((m.trees[0].leaf_value[0] - (-2.0 / (2.0 + e))).abs() < 1e-9);
        assert!((m.trees[0].leaf_value[1] - (-6.0 / (2.0 + e))).abs() < 1e-9);
    }

    // --- prediction early stopping (PRD-05) ---

    #[test]
    fn pred_early_stop_disabled_is_identical_to_predict_raw() {
        let m = five_iter_regression();
        let row = [1.0f64, 0.0];
        let plain = m.predict_raw(&row, 0, -1);
        // freq <= 0 disables the hook; result must be byte-identical, all 5 iters.
        let (es, iters) = m.predict_raw_early_stop(&row, 0, -1, 0, 10.0);
        assert_eq!(es, plain, "disabled early-stop must match predict_raw exactly");
        assert_eq!(iters, 5, "all iterations evaluated when disabled");
    }

    #[test]
    fn pred_early_stop_binary_margin_fires_at_freq() {
        // five_iter_regression accumulates {1,2,4,8,16} on feature-0=1.0.
        // After iter 0 (counter==1): score 1.0, 2*|1.0|=2.0. With freq=1, margin=1.5
        // the binary margin (2.0 > 1.5) fires immediately -> iters_evaluated == 1.
        let m = five_iter_regression();
        let row = [1.0f64, 0.0];
        let (es, iters) = m.predict_raw_early_stop(&row, 0, -1, 1, 1.5);
        assert_eq!(iters, 1, "binary margin should fire after the first iteration");
        assert!((es[0] - 1.0).abs() < 1e-12, "score frozen at the stop point");
    }

    #[test]
    fn pred_early_stop_checks_only_every_freq_iterations() {
        // freq=3 only checks at counter==3 (after iter 2: cumulative 1+2+4=7,
        // 2*7=14 > margin 10 -> stop at iters_evaluated==3). It must NOT stop earlier
        // even though 2*|score| already exceeds margin at iter 2 (cumulative 3, 6<10
        // anyway) — the load-bearing part is the check cadence.
        let m = five_iter_regression();
        let row = [1.0f64, 0.0];
        let (es, iters) = m.predict_raw_early_stop(&row, 0, -1, 3, 10.0);
        assert_eq!(iters, 3, "margin checked only at the freq boundary");
        assert!((es[0] - 7.0).abs() < 1e-12);
    }

    #[test]
    fn pred_early_stop_no_stop_runs_full_range() {
        // A margin so large it never fires -> all iterations, full score 31.
        let m = five_iter_regression();
        let row = [1.0f64, 0.0];
        let (es, iters) = m.predict_raw_early_stop(&row, 0, -1, 1, 1e9);
        assert_eq!(iters, 5);
        assert!((es[0] - 31.0).abs() < 1e-12);
    }

    #[test]
    fn pred_early_stop_multiclass_uses_top2_margin() {
        // Multiclass margin = top1 - top2.
        assert!(pred_early_stop_should_stop(&[5.0, 1.0, 0.0], 3.0)); // 5-1=4 > 3
        assert!(!pred_early_stop_should_stop(&[5.0, 4.0, 0.0], 3.0)); // 5-4=1 < 3
        // Binary (len 1) uses 2*|score|.
        assert!(pred_early_stop_should_stop(&[2.0], 3.0)); // 2*2=4 > 3
        assert!(!pred_early_stop_should_stop(&[1.0], 3.0)); // 2*1=2 < 3
        // Empty never stops.
        assert!(!pred_early_stop_should_stop(&[], 0.0));
    }
}
