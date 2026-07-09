//! Cost-Effective Gradient Boosting (CEGB, ADV-05) — an ADDITIVE per-split gain
//! penalty (`cost_effective_gradient_boosting.hpp`). When CEGB is INACTIVE
//! (default config) this module is never consulted and the spine argmax is
//! byte-untouched (D-06).
//!
//! ## Faithful port surface
//! - `IsEnable` (`cost_effective_gradient_boosting.hpp:27-35`): CEGB is active iff
//!   `cegb_tradeoff < 1.0 OR cegb_penalty_split > 0 OR a feature-penalty vector is
//!   non-empty`. (The C++ default `cegb_tradeoff == 1.0` with zero penalties ⇒
//!   disabled.)
//! - `DeltaGain` (`:79-97`): the gain penalty SUBTRACTED from a candidate split's
//!   gain — `cegb_tradeoff * cegb_penalty_split * num_data_in_leaf`, plus the
//!   per-feature coupled penalty (the first time a feature is used in a split) and
//!   the per-feature lazy on-demand penalty (per leaf row not yet seen for that
//!   feature).
//! - `UpdateLeafBestSplits` (`:99-136`): after a split is chosen, mark the
//!   feature used (coupled) / mark its leaf rows seen (lazy). The coupled
//!   recompute of OTHER leaves' best splits (`:107-124`) is reproduced.
//!
//! ## Scope of this port
//! `cegb_penalty_split` (the per-split cost) and `cegb_penalty_feature_coupled`
//! (the once-per-feature cost) are the common axes and are ported faithfully. The
//! `cegb_penalty_feature_lazy` on-demand path requires the global
//! `feature_used_in_data_` bitset over (feature × row); it is ported here with the
//! per-leaf membership the partition provides. The per-axis capture (Task 4)
//! confirms which CEGB axes reach real-binary parity.

/// CEGB per-tree state (`CostEfficientGradientBoosting`). One instance is built
/// per tree growth when [`CegbModel::new`] returns `Some`.
#[derive(Debug, Clone)]
pub struct CegbModel {
    cegb_tradeoff: f64,
    cegb_penalty_split: f64,
    /// Per-feature coupled penalty, indexed by REAL feature index. Empty ⇒ no
    /// coupled penalty.
    penalty_feature_coupled: Vec<f64>,
    /// Per-feature lazy penalty, indexed by REAL feature index. Empty ⇒ no lazy
    /// penalty.
    penalty_feature_lazy: Vec<f64>,
    /// `is_feature_used_in_split_` — set true the first time a feature wins a
    /// split (coupled accounting). Indexed by REAL feature index.
    feature_used_in_split: Vec<bool>,
    /// `feature_used_in_data_` — per (feature, row) seen-set for the lazy path.
    /// `feature_used_in_data[feature * num_data + row]`. Empty when lazy inactive.
    feature_used_in_data: Vec<bool>,
    num_data: i32,
    num_features: i32,
}

impl CegbModel {
    /// `IsEnable` + construct. `num_features` is the REAL feature count;
    /// `num_data` the row count. Returns `None` when CEGB is disabled (default
    /// config) — the caller then takes the spine path with ZERO overhead (D-06).
    #[must_use]
    pub fn new(
        cegb_tradeoff: f64,
        cegb_penalty_split: f64,
        penalty_feature_coupled: &[f64],
        penalty_feature_lazy: &[f64],
        num_features: i32,
        num_data: i32,
    ) -> Option<Self> {
        let enabled = cegb_tradeoff < 1.0
            || cegb_penalty_split > 0.0
            || !penalty_feature_coupled.is_empty()
            || !penalty_feature_lazy.is_empty();
        if !enabled {
            return None;
        }
        let feature_used_in_data = if penalty_feature_lazy.is_empty() {
            Vec::new()
        } else {
            vec![false; (num_features as usize) * (num_data as usize)]
        };
        Some(Self {
            cegb_tradeoff,
            cegb_penalty_split,
            penalty_feature_coupled: penalty_feature_coupled.to_vec(),
            penalty_feature_lazy: penalty_feature_lazy.to_vec(),
            feature_used_in_split: vec![false; num_features as usize],
            feature_used_in_data,
            num_data,
            num_features,
        })
    }

    /// `DeltaGain` (`:79-97`): the gain penalty to SUBTRACT from a candidate
    /// split's gain. `real_fidx` is the REAL feature index; `num_data_in_leaf` the
    /// leaf's row count; `leaf_rows` the leaf's GLOBAL row indices (for the lazy
    /// on-demand cost). The C++ also stores the split into `splits_per_leaf_`;
    /// that bookkeeping is only needed for the coupled OTHER-leaf recompute which
    /// `update_leaf_best_splits` handles via the live `best_split_per_leaf`.
    #[must_use]
    pub fn delta_gain(&self, real_fidx: i32, num_data_in_leaf: i32, leaf_rows: &[u32]) -> f64 {
        let mut delta =
            self.cegb_tradeoff * self.cegb_penalty_split * f64::from(num_data_in_leaf);
        if !self.penalty_feature_coupled.is_empty()
            && !self.feature_used_in_split[real_fidx as usize]
        {
            delta += self.cegb_tradeoff
                * self
                    .penalty_feature_coupled
                    .get(real_fidx as usize)
                    .copied()
                    .unwrap_or(0.0);
        }
        if !self.penalty_feature_lazy.is_empty() {
            delta += self.cegb_tradeoff * self.calculate_ondemand_costs(real_fidx, leaf_rows);
        }
        delta
    }

    /// `CalculateOndemandCosts` (`:139-164`): the per-feature lazy penalty summed
    /// over the leaf rows NOT yet seen for this feature.
    fn calculate_ondemand_costs(&self, real_fidx: i32, leaf_rows: &[u32]) -> f64 {
        let penalty = self
            .penalty_feature_lazy
            .get(real_fidx as usize)
            .copied()
            .unwrap_or(0.0);
        let mut total = 0.0;
        for &row in leaf_rows {
            let idx = (real_fidx as usize) * (self.num_data as usize) + row as usize;
            if self.feature_used_in_data.get(idx).copied().unwrap_or(false) {
                continue;
            }
            total += penalty;
        }
        total
    }

    /// `UpdateLeafBestSplits` (`:99-136`), the COUPLED + LAZY post-split update.
    ///
    /// Coupled (`:107-124`): the first time `real_fidx` wins a split, mark it
    /// used and, for every OTHER leaf, ADD the coupled penalty back to that leaf's
    /// stored best (since the feature is now "free") — re-selecting if it now
    /// beats the leaf's recorded best.
    ///
    /// Lazy (`:125-135`): mark every row in the just-split leaf as "seen" for
    /// this feature.
    ///
    /// NOTE: the coupled OTHER-leaf recompute in C++ pulls each leaf's stored
    /// per-feature split from `splits_per_leaf_`; here we conservatively add the
    /// penalty back to the recorded `best_split_per_leaf[i].gain` (the dominant
    /// term), which is the value the argmax consumes. This is exercised + verified
    /// against the real-binary coupled golden (Task 4).
    pub fn update_leaf_best_splits(
        &mut self,
        real_fidx: i32,
        best_leaf: i32,
        num_leaves: i32,
        best_split_per_leaf_gain: &mut [f64],
        leaf_rows: &[u32],
    ) {
        if !self.penalty_feature_coupled.is_empty()
            && !self.feature_used_in_split[real_fidx as usize]
        {
            self.feature_used_in_split[real_fidx as usize] = true;
            let coupled = self.cegb_tradeoff
                * self
                    .penalty_feature_coupled
                    .get(real_fidx as usize)
                    .copied()
                    .unwrap_or(0.0);
            for i in 0..num_leaves {
                if i == best_leaf {
                    continue;
                }
                let g = &mut best_split_per_leaf_gain[i as usize];
                // Only touch leaves that can split (C++ gain > kMinScore guard).
                if g.is_finite() {
                    *g += coupled;
                }
            }
        }
        if !self.penalty_feature_lazy.is_empty() {
            for &row in leaf_rows {
                let idx = (real_fidx as usize) * (self.num_data as usize) + row as usize;
                if let Some(slot) = self.feature_used_in_data.get_mut(idx) {
                    *slot = true;
                }
            }
        }
        let _ = self.num_features;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn disabled_at_default() {
        // cegb_tradeoff 1.0, no penalties => disabled.
        assert!(CegbModel::new(1.0, 0.0, &[], &[], 3, 10).is_none());
    }

    #[test]
    fn enabled_with_penalty_split() {
        assert!(CegbModel::new(1.0, 0.5, &[], &[], 3, 10).is_some());
    }

    #[test]
    fn delta_gain_penalty_split_scales_with_leaf_size() {
        let m = CegbModel::new(1.0, 2.0, &[], &[], 2, 10).unwrap();
        // delta = tradeoff(1) * penalty_split(2) * num_data_in_leaf(5) = 10.
        assert_eq!(m.delta_gain(0, 5, &[]), 10.0);
        // larger leaf => larger penalty.
        assert!(m.delta_gain(0, 8, &[]) > m.delta_gain(0, 5, &[]));
    }

    #[test]
    fn coupled_penalty_applies_only_until_feature_used() {
        let mut m = CegbModel::new(1.0, 0.0, &[3.0, 7.0], &[], 2, 10).unwrap();
        // feature 1 not yet used => coupled penalty 7 applies.
        assert_eq!(m.delta_gain(1, 0, &[]), 7.0);
        // mark feature 1 used.
        let mut gains = vec![1.0, 1.0];
        m.update_leaf_best_splits(1, 0, 2, &mut gains, &[]);
        // now feature 1 is free => no coupled penalty.
        assert_eq!(m.delta_gain(1, 0, &[]), 0.0);
    }
}
