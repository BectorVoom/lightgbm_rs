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
// Resident device score path. These are the kernel/type
// SYMBOLS the boosting toggle routes to — NOT a GPU compute runtime (the methods
// stay generic over `B::Runtime` via the re-exported `ComputeClient`), so the
// CMP-01 no-runtime gate still holds.
use lgbm_compute::kernels::predict::{add_prediction_to_score_on_device, PredictTree};
use lgbm_compute::kernels::score_updater::{add_score_constant_on, multiply_score_constant_on};
use lgbm_compute::ComputeClientReexport as ComputeClient;
use lgbm_compute::ComputeError;
use lgbm_model::Tree;

/// C++ `ScoreUpdater::score_` — the f64 accumulated raw score buffer.
///
/// Length is `num_data * num_class` (class-major: class `k`'s rows occupy
/// `[k*num_data, (k+1)*num_data)`). The current single-output spine uses `num_class
/// = 1`, but the layout/offset arithmetic is the multiclass-ready form so a future
/// multiclass path reuses it unchanged.
#[derive(Debug, Clone, PartialEq)]
pub struct ScoreUpdater {
    /// `std::vector<double> score_` — the f64 accumulator.
    score: Vec<f64>,
    /// `num_data_` — rows per class (the per-class stride / offset multiplier).
    num_data: i32,
    /// `num_class` — number of class-major score blocks.
    num_class: i32,
    /// The `boosting_on_cuda_` seam: when `true`, the resident
    /// device score path (`*_on` methods below) runs the constant/per-leaf ops on
    /// the device buffer and mirrors back to `self.score`; when `false` those
    /// methods delegate to the byte-unchanged host path. Keyed on
    /// `lgbm_compute::cuda_on_device_enabled()` at construction (D-09: OFF by
    /// default with `LGBM_CUDA_ON_DEVICE` unset), overridable by the GBDT device
    /// driver via [`ScoreUpdater::set_boosting_on_cuda`].
    boosting_on_cuda: bool,
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
            // D-09: OFF by default (env unset) so the host path is byte-unchanged.
            boosting_on_cuda: lgbm_compute::cuda_on_device_enabled(),
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

    /// C++ `ScoreUpdater::MultiplyScore(double val, int cur_tree_id)`
    /// (score_updater.hpp:63-69): multiply every row of class `cur_tree_id` by
    /// `val`. Used by the Random Forest variant (`rf.hpp:157-159,205-210`) to keep
    /// `score_` as a RUNNING AVERAGE of the per-tree raw outputs: each iter does
    /// `MultiplyScore(iter)` (un-average back to a sum), `AddScore(new_tree)` (add
    /// the raw tree), then `MultiplyScore(1/(iter+1))` (re-average over the now
    /// `iter+1` trees). NO learning-rate shrinkage is applied — RF averages, it does
    /// not accumulate shrunk increments.
    pub fn multiply_score(&mut self, val: f64, cur_tree_id: i32) {
        let off = self.offset(cur_tree_id);
        let end = off + self.num_data as usize;
        for s in &mut self.score[off..end] {
            *s *= val;
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

    // =====================================================================
    // Resident device score path + host-mirror toggle.
    //
    // ADDITIVE and GATED (D-09): with `LGBM_CUDA_ON_DEVICE` unset, `boosting_on_cuda`
    // is `false` and every `*_on` method delegates to the byte-unchanged host path
    // above. When the toggle is on, the constant ops route to the §11
    // `add_score_constant_on` / `multiply_score_constant_on` kernels and the per-leaf
    // training-path AddScore delegates to the
    // `add_prediction_to_score_on_device` tree-walk kernel (D-02 — NO new tree-walk
    // kernel), keeping the buffer resident and mirroring it back into `self.score`
    // (`CopyFromCUDADeviceToHost`) so non-resident consumers (`scores()` /
    // `class_scores()`) read the correct host vector.
    //
    // OUT-OF-SLICE (stay host): the DART/RF per-row-predict paths
    // [`Self::add_tree_predict_path`] / [`Self::add_tree_scaled_all`] are NOT part
    // of the continuous proving slice (they walk real feature values per row, not
    // the partition scatter) and remain on the host path.
    // =====================================================================

    /// Whether the resident device score path is active (the `boosting_on_cuda_`
    /// seam, keyed on `lgbm_compute::cuda_on_device_enabled()` at construction).
    #[inline]
    #[must_use]
    pub fn boosting_on_cuda(&self) -> bool {
        self.boosting_on_cuda
    }

    /// Driver seam: force the resident/host toggle. The GBDT device driver
    /// sets this from `on_device_eligible`; the env-gated default from
    /// [`ScoreUpdater::new`] already reflects `LGBM_CUDA_ON_DEVICE`.
    #[inline]
    pub fn set_boosting_on_cuda(&mut self, on: bool) {
        self.boosting_on_cuda = on;
    }

    /// §11 `AddScoreConstant` (resident): the device-kernel analog of
    /// [`Self::add_constant`]. Off ⇒ host method (byte-unchanged); on ⇒ run
    /// `add_score_constant_on` over the resident buffer and mirror the result back
    /// into `self.score`.
    ///
    /// # Errors
    /// Propagates [`ComputeError`] from the kernel launcher (invalid `cur_tree_id`
    /// / buffer-slice bounds).
    pub fn add_constant_on<B: Backend>(
        &mut self,
        client: &ComputeClient<B::Runtime>,
        val: f64,
        cur_tree_id: i32,
    ) -> Result<(), ComputeError> {
        if !self.boosting_on_cuda {
            self.add_constant(val, cur_tree_id);
            return Ok(());
        }
        // Whole-buffer resident op; mirror back (CopyFromCUDADeviceToHost analog).
        self.score = add_score_constant_on(client, &self.score, self.num_data, cur_tree_id, val)?;
        Ok(())
    }

    /// §11 `MultiplyScoreConstant` (resident): the device-kernel analog of
    /// [`Self::multiply_score`]. Off ⇒ host method (byte-unchanged); on ⇒ run
    /// `multiply_score_constant_on` over the resident buffer and mirror back.
    ///
    /// # Errors
    /// Propagates [`ComputeError`] from the kernel launcher.
    pub fn multiply_score_on<B: Backend>(
        &mut self,
        client: &ComputeClient<B::Runtime>,
        val: f64,
        cur_tree_id: i32,
    ) -> Result<(), ComputeError> {
        if !self.boosting_on_cuda {
            self.multiply_score(val, cur_tree_id);
            return Ok(());
        }
        self.score =
            multiply_score_constant_on(client, &self.score, self.num_data, cur_tree_id, val)?;
        Ok(())
    }

    /// §11 training-path `AddScore` (resident, D-02): the per-leaf score scatter
    /// delegates to the [`add_prediction_to_score_on_device`] tree-walk
    /// kernel — NO new per-row tree-walk kernel is introduced. The kernel returns
    /// the `num_data`-length per-row raw-margin delta; this adds it into class
    /// `cur_tree_id`'s slice and mirrors the resident buffer back to `self.score`.
    ///
    /// `predict_tree` / `rows` / `num_features` / `bit_type` are the walk inputs the
    /// device driver reconstructs from the grown `Tree` + `Dataset`
    /// bins; on the identity-binned corpus this is bit-exact to the host
    /// partition scatter [`Self::add_tree_train_path`] (the L2 contract).
    ///
    /// # Errors
    /// Propagates [`ComputeError`] from the tree-walk kernel (length / bit-type /
    /// index validation).
    #[allow(clippy::too_many_arguments)]
    pub fn add_tree_train_path_on_device<B: Backend>(
        &mut self,
        client: &ComputeClient<B::Runtime>,
        predict_tree: &PredictTree<'_>,
        rows: &[u32],
        num_features: usize,
        bit_type: u32,
        cur_tree_id: i32,
    ) -> Result<(), ComputeError> {
        let off = self.offset(cur_tree_id);
        let nd = self.num_data as usize;
        let delta = add_prediction_to_score_on_device(
            client,
            predict_tree,
            rows,
            nd,
            num_features,
            bit_type,
            nd,
            None,
        )?;
        for (i, d) in delta.iter().enumerate() {
            self.score[off + i] += *d;
        }
        Ok(())
    }

    /// Resident `AddScore` for an ON-DEVICE-grown tree:
    /// add a PRE-COMPUTED per-row leaf-value `delta` (produced on device by the grow
    /// driver's resident partition scatter
    /// [`lgbm_compute::add_prediction_to_score_on_device_resident`] over the resident
    /// row→leaf partition) in place into class `cur_tree_id`'s slice of `self.score`
    /// (the score buffer itself — there is no separate device-resident score copy;
    /// `self.score` IS the buffer, matching the sibling
    /// [`Self::add_tree_train_path_on_device`]).
    ///
    /// This is the SCATTER analog of [`Self::add_tree_train_path_on_device`] (which
    /// re-walks the tree over a reconstructed `[num_data × num_features]` bin matrix each
    /// iteration, a significant host-scoring cost): the grow already computed the
    /// row→leaf partition ON DEVICE, so the score move only scatters exact f64 leaf
    /// values by integer row index. Bit-exact to the host partition scatter
    /// [`Self::add_tree_train_path`] on the cpu-f64 anchor (`delta[i]` IS that leaf
    /// value); ~1e-6 vs host-CUDA. `delta` is `num_data` long (one raw-margin per row).
    pub fn add_tree_train_path_delta_on_device(&mut self, delta: &[f64], cur_tree_id: i32) {
        debug_assert_eq!(
            delta.len(),
            self.num_data as usize,
            "delta must be num_data long"
        );
        let off = self.offset(cur_tree_id);
        // IN-02 (25-REVIEW): bound the scatter to `num_data` so a mis-sized (long) delta can
        // never index `self.score[off + i]` past this class's slice in release (where the
        // `debug_assert_eq!` above is compiled out). A short delta still updates only a prefix,
        // but it can no longer corrupt neighbouring class scores or panic out of bounds.
        for (i, d) in delta.iter().take(self.num_data as usize).enumerate() {
            self.score[off + i] += *d;
        }
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

    /// C++ DART `train_score_updater_->AddScore(model, cur_tree_id)`
    /// (`dart.hpp:135,171,189`): add `multiply * tree.predict(row)` to EVERY row of
    /// class `cur_tree_id`, traversing the tree by each row's real feature values.
    /// DART's DroppingTrees / Normalize re-score dropped trees over the FULL corpus
    /// (the C++ `AddScore` over the whole data partition); on the identity-binned
    /// corpus the per-row predict is bit-exact to the partition scatter (the L2
    /// contract — see [`Self::add_tree_predict_path`]). `multiply` scales the tree's
    /// contribution (e.g. `-1.0` to drop, `1/(k+1)` to normalize a step) — this
    /// mirrors the C++ pattern of `Shrinkage(factor)` then `AddScore`, but applied to
    /// the SCORE only (the stored tree mutation is done separately by the caller so
    /// the model text and score stay in lockstep). Constant (`num_leaves<=1`) trees
    /// contribute their single leaf value × `multiply` to every row.
    pub fn add_tree_scaled_all<FRow>(
        &mut self,
        tree: &Tree,
        cur_tree_id: i32,
        multiply: f64,
        feature_row: FRow,
    ) where
        FRow: Fn(i32) -> Vec<f64>,
    {
        let off = self.offset(cur_tree_id);
        let nd = self.num_data as usize;
        for row in 0..nd {
            let out = tree.predict(&feature_row(row as i32)) * multiply;
            self.score[off + row] += out;
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
    fn multiply_score_is_per_class_and_running_average() {
        // 2 classes, 3 rows each.
        let mut su = ScoreUpdater::new(3, 2, None);
        su.add_constant(2.0, 0); // class 0 = [2,2,2]
        su.add_constant(4.0, 1); // class 1 = [4,4,4]
        su.multiply_score(0.5, 0); // class 0 only -> [1,1,1]
        assert_eq!(su.class_scores(0), &[1.0, 1.0, 1.0]);
        assert_eq!(su.class_scores(1), &[4.0, 4.0, 4.0]);

        // RF running-average pattern: start with the iter-0 average a0, add raw t1.
        // After iter 0 score holds a0 (the average of 1 tree). At iter_=1:
        //   MultiplyScore(1) -> a0 (un-average over 1 tree, no-op for iter 1)
        //   AddScore(t1)     -> a0 + t1
        //   MultiplyScore(1/2) -> (a0 + t1)/2  == the mean of {a0, t1}
        let mut rf = ScoreUpdater::new(1, 1, None);
        rf.add_constant(10.0, 0); // a0 = 10 (the first tree's raw output, "averaged" over 1)
        rf.multiply_score(1.0, 0); // iter_ = 1
        rf.add_constant(20.0, 0); // + raw tree t1
        rf.multiply_score(0.5, 0); // 1/(iter_+1) = 1/2
        assert_eq!(rf.class_scores(0), &[15.0]); // mean(10, 20)
    }

    #[test]
    fn score_buffer_is_f64() {
        // Type-level proof: the accumulator is f64, not f32.
        let su = ScoreUpdater::new(2, 1, None);
        let _s: &[f64] = su.scores();
    }
}
