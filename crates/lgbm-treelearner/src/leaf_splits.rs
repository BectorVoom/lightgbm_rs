//! `LeafSplits` — ordered f64 sum_gradient/sum_hessian seeding + leaf output.
//!
//! Faithful transcription of `LeafSplits` (`leaf_splits.hpp:98-180`, commit
//! 195c26fc, VERSION 4.6.0.99): the per-leaf running totals the learner seeds
//! before scanning a leaf, plus the `weight()` accessor used as `parent_output`.
//!
//! ## The ordered-fold gate (load-bearing — RESEARCH anti-pattern)
//! C++ `LeafSplits::Init` computes `sum_gradients`/`sum_hessians` over the leaf's
//! rows with an OpenMP reduction GUARDED by `if (... && !deterministic_)`:
//!
//! ```cpp
//! // leaf_splits.hpp (Init over a data-partition range)
//! double tmp_sum_gradients = 0.0f, tmp_sum_hessians = 0.0f;
//! #pragma omp parallel for ... reduction(+:...) if (num_data > 1024 && !deterministic_)
//! for (data_size_t i = 0; i < num_data; ++i) {
//!   tmp_sum_gradients += gradients[idx];
//!   tmp_sum_hessians  += hessians[idx];
//! }
//! ```
//!
//! In the pinned deterministic anchor (`deterministic=true`), the `!deterministic_`
//! clause DISABLES the parallel reduction, so the fold is a single LEFT-TO-RIGHT
//! ordered f64 accumulation. We reproduce exactly that — a plain sequential
//! `for` over the leaf's row indices — and NEVER a rayon / parallel reduction
//! (which would change the f64 summation order and break bit-exactness).
//!
//! ## Numerical type boundary
//! Gradients/hessians cross in as `f32` (`score_t = float`) and are accumulated
//! into `f64` (`hist_t = double`) — the `+=` widens each `f32` term to `f64`
//! before adding, exactly as the C++ `double tmp_sum += gradients[i]` does.

use lgbm_compute::gain::{calculate_splitted_leaf_output_full, GainConfig};

/// Per-leaf split state — the C++ `LeafSplits` running totals (`leaf_splits.hpp`).
///
/// Field names mirror the C++ members:
/// - `num_data_in_leaf` ↔ `num_data_in_leaf_`
/// - `sum_gradients`    ↔ `sum_gradients_` (f64)
/// - `sum_hessians`     ↔ `sum_hessians_` (f64)
/// - `weight`           ↔ `weight_` (the leaf output, `parent_output` source)
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LeafSplits {
    /// C++ `data_size_t num_data_in_leaf_`.
    pub num_data_in_leaf: i32,
    /// C++ `double sum_gradients_` — ordered f64 fold over the leaf's rows.
    pub sum_gradients: f64,
    /// C++ `double sum_hessians_` — ordered f64 fold over the leaf's rows.
    pub sum_hessians: f64,
    /// C++ `double weight_` — the leaf output (`CalculateSplittedLeafOutput`),
    /// used as `parent_output` for path-smoothing (no-op on the spine).
    pub weight: f64,
}

impl LeafSplits {
    /// A zeroed `LeafSplits` (the C++ default-constructed state before `Init`).
    pub fn new() -> Self {
        Self {
            num_data_in_leaf: 0,
            sum_gradients: 0.0,
            sum_hessians: 0.0,
            weight: 0.0,
        }
    }

    /// `LeafSplits::Init` over an explicit row-index slice (`leaf_splits.hpp`):
    /// the SINGLE ordered f64 fold of `gradients`/`hessians` over the leaf's rows.
    ///
    /// `indices` are the data-partition row indices belonging to this leaf
    /// (`data_partition.indices()[leaf_begin..leaf_begin+leaf_count]`). For the
    /// ROOT leaf the caller passes all rows `0..num_data` (the C++ `Init()` whole-
    /// dataset variant). The fold is strictly sequential in `indices` order.
    ///
    /// `cfg` supplies `lambda_l1`/`lambda_l2` for the `weight()` seed via the
    /// EXISTING `gain::calculate_splitted_leaf_output` (do NOT re-derive).
    pub fn init(&mut self, gradients: &[f32], hessians: &[f32], indices: &[u32], cfg: &GainConfig) {
        let mut sum_g = 0.0f64;
        let mut sum_h = 0.0f64;
        // Ordered LEFT-TO-RIGHT fold (deterministic anchor — NEVER parallel).
        for &idx in indices {
            let i = idx as usize;
            // f32 -> f64 widening at the `+=` (C++ `double tmp_sum += gradients[i]`).
            sum_g += f64::from(gradients[i]);
            sum_h += f64::from(hessians[i]);
        }
        self.num_data_in_leaf = indices.len() as i32;
        self.sum_gradients = sum_g;
        self.sum_hessians = sum_h;
        self.weight = Self::self_output(sum_g, sum_h, cfg);
    }

    /// The leaf's OWN output, as C++ `SerialTreeLearner::GetParentOutput` computes it
    /// for the root (`serial_tree_learner.cpp:1007-1013`):
    ///
    /// ```cpp
    /// CalculateSplittedLeafOutput<USE_MC=true, USE_L1=true, USE_MAX_OUTPUT=true,
    ///                             USE_SMOOTHING=false>(
    ///     sum_gradients, sum_hessians, lambda_l1, lambda_l2, max_delta_step,
    ///     BasicConstraint(), path_smooth, num_data_in_leaf, 0);
    /// ```
    ///
    /// Note the template axes: `max_delta_step` IS applied, smoothing is NOT ("we
    /// don't apply any smoothing to the root"), and the default-constructed
    /// `BasicConstraint()` is unbounded, so `USE_MC` clamps nothing.
    ///
    /// Seeding `weight` with this — rather than leaving it at C++'s literal `0` —
    /// is what lets the learner read `parent_output` uniformly off `LeafSplits`:
    /// a root leaf carries its own output here, and a CHILD leaf carries the
    /// output its parent split assigned it (see
    /// [`init_from_split`](Self::init_from_split)). That is exactly the two
    /// branches of `GetParentOutput`, decided by which seeding path ran instead of
    /// by an explicit `tree->num_leaves() == 1` test.
    fn self_output(sum_g: f64, sum_h: f64, cfg: &GainConfig) -> f64 {
        calculate_splitted_leaf_output_full(
            cfg.use_l1(),
            sum_g,
            sum_h,
            cfg.lambda_l1,
            cfg.lambda_l2,
            cfg.max_delta_step,
            // USE_SMOOTHING = false — the root is never smoothed.
            false,
            cfg.path_smooth,
            0,
            0.0,
        )
    }

    /// `LeafSplits::Init(sum_gradient, sum_hessian)` (`leaf_splits.hpp`): seed the
    /// totals DIRECTLY from already-computed child sums (the subtraction-trick
    /// case, where the larger child's sums are derived without re-folding rows).
    ///
    /// `num_data_in_leaf` is the partition row count for this leaf. The `weight()`
    /// is recomputed from the supplied sums.
    pub fn init_from_sums(
        &mut self,
        num_data_in_leaf: i32,
        sum_gradient: f64,
        sum_hessian: f64,
        cfg: &GainConfig,
    ) {
        self.num_data_in_leaf = num_data_in_leaf;
        self.sum_gradients = sum_gradient;
        self.sum_hessians = sum_hessian;
        self.weight = Self::self_output(sum_gradient, sum_hessian, cfg);
    }

    /// `LeafSplits::Init(leaf, data_partition, sum_gradients, sum_hessians,
    /// weight)` (`leaf_splits.hpp:47-54`): seed a CHILD leaf's totals DIRECTLY from
    /// the parent split's `SplitInfo` (`serial_tree_learner.cpp:851-871`), carrying
    /// the EXACT `sum_gradients`/`sum_hessians` the split produced AND the split's
    /// already-computed leaf `output` as `weight_` — NOT a re-fold over the child's
    /// rows and NOT a re-derivation of the output.
    ///
    /// This is load-bearing for bit-exactness: the C++ child seed is
    /// `best_split_info.{left,right}_sum_hessian` (= `best_sum_left_hessian -
    /// kEpsilon`, `feature_histogram.hpp:1042`), which carries the accumulated
    /// `kEpsilon` provenance from the parent's REVERSE scan. A fresh re-fold of the
    /// child's rows loses that provenance (e.g. yields exactly `4.0` where C++ has
    /// `4.000000000000001`), shifting the grandchild leaf-output denominator by 2
    /// ULPs (the 05-09 mfb>0 node-2 leaf-0 residual). Verified against a real
    /// `lib_lightgbm` 4.6 FP execution trace.
    pub fn init_from_split(
        &mut self,
        num_data_in_leaf: i32,
        sum_gradient: f64,
        sum_hessian: f64,
        weight: f64,
    ) {
        self.num_data_in_leaf = num_data_in_leaf;
        self.sum_gradients = sum_gradient;
        self.sum_hessians = sum_hessian;
        self.weight = weight;
    }

    /// C++ `LeafSplits::weight()` — the leaf output used as `parent_output`.
    pub fn weight(&self) -> f64 {
        self.weight
    }
}

impl Default for LeafSplits {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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

    /// The ordered f64 fold must equal a hand-computed sequential sum (no parallel
    /// reduction). Uses indices in a deliberately non-sorted order to prove the
    /// fold walks `indices` exactly.
    #[test]
    fn init_ordered_fold_matches_sequential_sum() {
        let g = vec![1.0f32, 2.0, 3.0, 4.0];
        let h = vec![0.5f32, 0.5, 0.5, 0.5];
        let indices = vec![2u32, 0, 3]; // rows 2,0,3 (skip row 1)
        let cfg = relaxed_cfg();
        let mut ls = LeafSplits::new();
        ls.init(&g, &h, &indices, &cfg);
        // Hand fold: 3.0 + 1.0 + 4.0 = 8.0; 0.5*3 = 1.5.
        let mut want_g = 0.0f64;
        let mut want_h = 0.0f64;
        for &i in &indices {
            want_g += f64::from(g[i as usize]);
            want_h += f64::from(h[i as usize]);
        }
        assert_eq!(ls.sum_gradients, want_g);
        assert_eq!(ls.sum_hessians, want_h);
        assert_eq!(ls.sum_gradients, 8.0);
        assert_eq!(ls.sum_hessians, 1.5);
        assert_eq!(ls.num_data_in_leaf, 3);
    }

    #[test]
    fn init_seeds_weight_via_calculate_leaf_output() {
        let g = vec![4.0f32, 0.0];
        let h = vec![1.0f32, 1.0];
        let indices = vec![0u32, 1];
        let cfg = relaxed_cfg();
        let mut ls = LeafSplits::new();
        ls.init(&g, &h, &indices, &cfg);
        // No-L1 output = -sum_g / (sum_h + l2) = -4 / 2 = -2.0.
        assert_eq!(ls.sum_gradients, 4.0);
        assert_eq!(ls.sum_hessians, 2.0);
        assert_eq!(ls.weight(), -2.0);
    }

    #[test]
    fn init_from_sums_seeds_directly() {
        let cfg = relaxed_cfg();
        let mut ls = LeafSplits::new();
        ls.init_from_sums(50, 6.0, 3.0, &cfg);
        assert_eq!(ls.num_data_in_leaf, 50);
        assert_eq!(ls.sum_gradients, 6.0);
        assert_eq!(ls.sum_hessians, 3.0);
        assert_eq!(ls.weight(), -2.0); // -6/3
    }
}
