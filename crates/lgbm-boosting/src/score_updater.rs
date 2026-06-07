//! The f64 score accumulator (`ScoreUpdater`), ported 1:1 from the C++ reference.
//!
//! Faithful-mirror citations (read directly from the in-tree C++ source):
//! - `LightGBM/src/boosting/score_updater.hpp`:
//!   - `score_` is `std::vector<double>` of length `num_data * num_tree_per_iter`,
//!     class-major (offset `= num_data * cur_tree_id`). Accumulation is in f64
//!     throughout — only the gradients/hessians/leaf-values are f32 (RESEARCH
//!     Anti-Pattern: accumulating scores in f32 diverges).
//!   - init (iter 0): zero, unless the `Dataset` carries `init_score` metadata
//!     (then copy it).
//!   - `AddScore(double val, cur_tree_id)`: `score_[offset + i] += val` for all
//!     rows — used by `BoostFromAverage`.
//!   - training-path `AddScore(tree_learner, tree, cur_tree_id)`: delegates to
//!     `tree_learner->AddPredictionToScore(tree, score_ + offset)`, the per-leaf
//!     data-partition scatter (NOT a per-row tree walk) — the bit-exact hot path
//!     (serial_tree_learner.h:100-118).

use lgbm_treelearner::{DataPartition, SerialTreeLearner};
use lgbm_compute::Backend;
use lgbm_model::Tree;

/// C++ `ScoreUpdater::score_` — the f64 accumulated raw score buffer.
///
/// Length is `num_data * num_class` (class-major: class `k`'s rows occupy
/// `[k*num_data, (k+1)*num_data)`). For the 06-02 single-output spine `num_class
/// = 1`, but the layout/offset arithmetic is the multiclass-ready form so 06-04
/// reuses it unchanged.
#[derive(Debug, Clone, PartialEq)]
pub struct ScoreUpdater {
    /// `std::vector<double> score_` — the f64 accumulator.
    score: Vec<f64>,
    /// `num_data_` — rows per class (the per-class stride / offset multiplier).
    num_data: i32,
    /// `num_class` — number of class-major score blocks.
    num_class: i32,
}

impl ScoreUpdater {
    /// Construct the score buffer for `num_data` rows × `num_class` classes,
    /// initialized from optional `init_score` metadata (copied verbatim when
    /// present) else all-zero — mirroring `ScoreUpdater::ScoreUpdater`
    /// (score_updater.hpp:30-46).
    ///
    /// `init_score`, when `Some`, must have length `num_data * num_class`
    /// (class-major). When `None` the buffer is zeroed.
    pub fn new(num_data: i32, num_class: i32, init_score: Option<&[f64]>) -> Self {
        let total = (num_data.max(0) as usize) * (num_class.max(0) as usize);
        let score = match init_score {
            Some(s) if s.len() == total => s.to_vec(),
            // C++ falls back to zero when init_score is absent or mismatched
            // (it only copies when num_data*num_class entries are present).
            _ => vec![0.0f64; total],
        };
        Self {
            score,
            num_data,
            num_class,
        }
    }

    /// The class-major byte offset for `cur_tree_id` (C++ `offset = num_data *
    /// cur_tree_id`).
    #[inline]
    fn offset(&self, cur_tree_id: i32) -> usize {
        (self.num_data as usize) * (cur_tree_id.max(0) as usize)
    }

    /// Read-only view of the whole f64 score buffer (all classes, class-major).
    pub fn scores(&self) -> &[f64] {
        &self.score
    }

    /// Read-only view of one class's score slice (`[offset, offset + num_data)`).
    pub fn class_scores(&self, cur_tree_id: i32) -> &[f64] {
        let off = self.offset(cur_tree_id);
        &self.score[off..off + self.num_data as usize]
    }

    /// C++ `ScoreUpdater::AddScore(double val, int cur_tree_id)`
    /// (score_updater.hpp:54-61): add the constant `val` to every row of class
    /// `cur_tree_id`. Used by `BoostFromAverage` to inject the init score.
    pub fn add_constant(&mut self, val: f64, cur_tree_id: i32) {
        let off = self.offset(cur_tree_id);
        let end = off + self.num_data as usize;
        for s in &mut self.score[off..end] {
            *s += val;
        }
    }

    /// C++ training-path `ScoreUpdater::AddScore(tree_learner, tree, cur_tree_id)`
    /// (score_updater.hpp:88-92): delegate to
    /// `SerialTreeLearner::add_prediction_to_score` over class `cur_tree_id`'s
    /// score slice — the per-leaf data-partition scatter (the bit-exact hot path,
    /// serial_tree_learner.h:100-118), NOT a per-row tree walk.
    ///
    /// `data_partition` is the partition the `tree` was grown over (returned by
    /// `SerialTreeLearner::train_returning_partition`).
    pub fn add_tree_train_path<B: Backend>(
        &mut self,
        learner: &SerialTreeLearner<'_, B>,
        tree: &Tree,
        data_partition: &DataPartition,
        cur_tree_id: i32,
    ) {
        let off = self.offset(cur_tree_id);
        let end = off + self.num_data as usize;
        learner.add_prediction_to_score(tree, data_partition, &mut self.score[off..end]);
    }

    /// C++ predict-side `ScoreUpdater::AddScore(tree, data_indices, data_cnt,
    /// cur_tree_id)` (score_updater.hpp / gbdt.cpp:499-509 OOB path): add the tree's
    /// per-row prediction to `score_` for a SELECTED set of rows, traversing the
    /// tree by their real feature values (`tree.predict`). Used for the out-of-bag
    /// rows when bagging (and, on the identity-binned corpus, equivalently for the
    /// whole row set — the per-row predict is bit-exact to the partition scatter,
    /// the L2 contract). `rows` are the GLOBAL row indices; `feature_row(row)`
    /// yields that row's real feature value vector.
    ///
    /// A single-leaf (constant `as_constant`) tree still contributes its leaf value
    /// (the `AsConstantTree` path adds its constant to every selected row) — but the
    /// boosting loop only calls this for grown trees (`num_leaves > 1`) on the
    /// bagging path; the constant-tree init is injected via `add_constant`.
    pub fn add_tree_predict_path<FRow>(
        &mut self,
        tree: &Tree,
        rows: &[i32],
        cur_tree_id: i32,
        feature_row: FRow,
    ) where
        FRow: Fn(i32) -> Vec<f64>,
    {
        if tree.num_leaves <= 1 {
            return;
        }
        let off = self.offset(cur_tree_id);
        for &row in rows {
            let out = tree.predict(&feature_row(row));
            self.score[off + row as usize] += out;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_zeroes_without_init_score() {
        let su = ScoreUpdater::new(4, 1, None);
        assert_eq!(su.scores(), &[0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn new_copies_init_score() {
        let init = [1.0, 2.0, 3.0, 4.0];
        let su = ScoreUpdater::new(4, 1, Some(&init));
        assert_eq!(su.scores(), &init);
    }

    #[test]
    fn add_constant_is_per_class() {
        // 2 classes, 3 rows each.
        let mut su = ScoreUpdater::new(3, 2, None);
        su.add_constant(0.5, 0); // class 0 only
        assert_eq!(su.class_scores(0), &[0.5, 0.5, 0.5]);
        assert_eq!(su.class_scores(1), &[0.0, 0.0, 0.0]);
        su.add_constant(-1.0, 1); // class 1 only
        assert_eq!(su.class_scores(1), &[-1.0, -1.0, -1.0]);
    }

    #[test]
    fn score_buffer_is_f64() {
        // Type-level proof: the accumulator is f64, not f32.
        let su = ScoreUpdater::new(2, 1, None);
        let _s: &[f64] = su.scores();
    }
}
