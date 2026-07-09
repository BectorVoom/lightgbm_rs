//! `ColSampler` — per-tree / per-node feature subsampling (TRL-08).
//!
//! Faithful 1:1 port of LightGBM's `ColSampler`
//! (`LightGBM/src/treelearner/col_sampler.hpp`, commit 195c26fc, VERSION
//! 4.6.0.99) for the pinned spine config: numeric features, NO interaction
//! constraints (`interaction_constraints_vector` empty — the Phase-7+ branch is
//! dropped), single-group features (`InnerFeatureIndex(real) == real`).
//!
//! ## Why this exists (the parity risk)
//! The PRNG itself ([`lgbm_core::Random`]) is already bit-exact (FND-01). The
//! risk this module carries is the **CALL SEQUENCE** of `Random::Sample`:
//! - `reset_by_tree()` draws ONCE per tree (`BeforeTrain`), ONLY when
//!   `feature_fraction < 1.0` (`col_sampler.hpp:74-89`).
//! - `get_by_node()` draws ONCE per node, when `feature_fraction_bynode < 1.0`
//!   (`col_sampler.hpp:91-179`) — and the learner calls it for the SMALLER leaf
//!   THEN the larger leaf within one split decision
//!   (`serial_tree_learner.cpp:479,487`).
//!
//! A wrong draw order silently selects different features (threat T-05-04-01) —
//! so this module reproduces the exact order and the goldens assert the selected
//! indices per tree/node.
//!
//! ## Dropped branches (Phase-7+)
//! `interaction_constraints_` is always empty here, so the interaction-filtered
//! paths (`col_sampler.hpp:93-111,131-141,157-167`) collapse to the simple
//! `allowed_*_feature_indices = &valid/used_feature_indices_` case. The OpenMP
//! scatter loop (`:82-87,144-151,170-176`) is a plain sequential set-to-1 here
//! (single thread, deterministic).

use lgbm_core::random::Random;

/// C++ `Common::RoundInt(double x) { return (int)(x + 0.5f); }` (common.h:904).
/// The `0.5f` is an `f32` literal promoted to `f64` in the add — transcribed
/// exactly (Pitfall 4 discipline).
fn round_int(x: f64) -> i32 {
    (x + f64::from(0.5f32)) as i32
}

/// `ColSampler` — drives `feature_fraction` (per-tree) and
/// `feature_fraction_bynode` (per-node) feature selection via the bit-exact
/// [`Random`] LCG seeded by `feature_fraction_seed`.
///
/// On the spine, `valid_feature_indices_` is simply `0..num_features` (every
/// feature is valid — no trivial features dropped at this layer) and
/// `InnerFeatureIndex(real) == real` (single-group), so `is_feature_used_` is
/// indexed by the same id space as `valid_feature_indices_`.
pub struct ColSampler {
    /// C++ `fraction_bytree_` (`config->feature_fraction`).
    fraction_bytree: f64,
    /// C++ `fraction_bynode_` (`config->feature_fraction_bynode`).
    fraction_bynode: f64,
    /// C++ `random_` — seeded ONCE by `feature_fraction_seed`, advanced by every
    /// `Sample` call (the call sequence is the parity contract).
    random: Random,
    /// C++ `need_reset_bytree_` (`fraction_bytree_ < 1.0`).
    need_reset_bytree: bool,
    /// C++ `used_cnt_bytree_`.
    used_cnt_bytree: i32,
    /// C++ `valid_feature_indices_` — the candidate features. On the spine this
    /// is `0..num_features`.
    valid_feature_indices: Vec<i32>,
    /// C++ `used_feature_indices_` — the per-tree selected POSITIONS into
    /// `valid_feature_indices_` (set by [`reset_by_tree`](Self::reset_by_tree)).
    used_feature_indices: Vec<i32>,
    /// C++ `is_feature_used_` — per-(inner)-feature used flag for the current
    /// tree, length `num_features`.
    is_feature_used: Vec<i8>,
}

impl ColSampler {
    /// `static int GetCnt(size_t total_cnt, double fraction)` (col_sampler.hpp:33-37):
    /// `max( RoundInt(total_cnt * fraction), min(1, total_cnt) )`.
    ///
    /// The `min(1, total)` floor guarantees at least one feature is drawn whenever
    /// any feature exists; `RoundInt` uses the f32-`0.5` rounding.
    pub fn get_cnt(total_cnt: usize, fraction: f64) -> i32 {
        let min = std::cmp::min(1, total_cnt as i32);
        let used_feature_cnt = round_int(total_cnt as f64 * fraction);
        std::cmp::max(used_feature_cnt, min)
    }

    /// Construct + `SetTrainingData` + the initial `ResetByTree`
    /// (col_sampler.hpp:22-31,39-52). `num_features` is `train_data->num_features()`;
    /// `valid_feature_indices` is `train_data->ValidFeatureIndices()` — on the spine
    /// every feature is valid, so the caller passes `0..num_features`.
    ///
    /// This draws the per-tree sample IMMEDIATELY (mirroring the C++ ctor →
    /// `SetTrainingData` → `ResetByTree` order), advancing `random_` once when
    /// `feature_fraction < 1.0`.
    pub fn new(
        feature_fraction: f64,
        feature_fraction_bynode: f64,
        feature_fraction_seed: i32,
        valid_feature_indices: Vec<i32>,
        num_features: usize,
    ) -> Self {
        let need_reset_bytree = !(feature_fraction >= 1.0);
        let used_cnt_bytree = if need_reset_bytree {
            Self::get_cnt(valid_feature_indices.len(), feature_fraction)
        } else {
            valid_feature_indices.len() as i32
        };
        let mut s = Self {
            fraction_bytree: feature_fraction,
            fraction_bynode: feature_fraction_bynode,
            random: Random::new(feature_fraction_seed),
            need_reset_bytree,
            used_cnt_bytree,
            valid_feature_indices,
            used_feature_indices: Vec::new(),
            is_feature_used: vec![1i8; num_features],
        };
        s.reset_by_tree();
        s
    }

    /// `void ResetByTree()` (col_sampler.hpp:74-89): when `need_reset_bytree_`,
    /// zero `is_feature_used_`, draw `used_feature_indices_ = random_.Sample(...)`
    /// (advancing the PRNG ONCE), then set the selected inner-feature flags to 1.
    ///
    /// When `feature_fraction >= 1.0` this is a NO-OP (no draw, every flag stays 1
    /// — the spine path). The caller must call this once per tree, in `BeforeTrain`.
    pub fn reset_by_tree(&mut self) {
        if !self.need_reset_bytree {
            return;
        }
        for v in &mut self.is_feature_used {
            *v = 0;
        }
        self.used_feature_indices = self
            .random
            .sample(self.valid_feature_indices.len() as i32, self.used_cnt_bytree);
        for &pos in &self.used_feature_indices {
            // valid_feature_indices_[pos] is the REAL feature; InnerFeatureIndex(real)
            // == real on the single-group spine.
            let used_feature = self.valid_feature_indices[pos as usize];
            self.is_feature_used[used_feature as usize] = 1;
        }
    }

    /// `std::vector<int8_t> GetByNode(const Tree*, int leaf)`
    /// (col_sampler.hpp:91-179) for the no-interaction-constraint path: returns the
    /// per-(inner)-feature used flags for ONE node, drawing the PRNG ONCE when
    /// `feature_fraction_bynode < 1.0`.
    ///
    /// - `fraction_bynode_ >= 1.0` → all-ones (no draw — the spine path).
    /// - else if `need_reset_bytree_` → sample `GetCnt(used_feature_indices_.size,
    ///   bynode)` positions from `used_feature_indices_` (the per-tree subset).
    /// - else → sample `GetCnt(valid_feature_indices_.size, bynode)` positions from
    ///   `valid_feature_indices_`.
    ///
    /// The learner calls this for the SMALLER leaf then the LARGER leaf within one
    /// split decision (serial_tree_learner.cpp:479,487) — the draw order is the
    /// parity contract (T-05-04-01).
    pub fn get_by_node(&mut self) -> Vec<i8> {
        let num_features = self.is_feature_used.len();
        if self.fraction_bynode >= 1.0 {
            // No interaction constraints → every feature allowed.
            return vec![1i8; num_features];
        }
        let mut ret = vec![0i8; num_features];
        if self.need_reset_bytree {
            let used_feature_cnt =
                Self::get_cnt(self.used_feature_indices.len(), self.fraction_bynode);
            let sampled_indices = self
                .random
                .sample(self.used_feature_indices.len() as i32, used_feature_cnt);
            for &i in &sampled_indices {
                // valid_feature_indices_[used_feature_indices_[i]] is the real feature.
                let used_feature =
                    self.valid_feature_indices[self.used_feature_indices[i as usize] as usize];
                ret[used_feature as usize] = 1;
            }
        } else {
            let used_feature_cnt =
                Self::get_cnt(self.valid_feature_indices.len(), self.fraction_bynode);
            let sampled_indices = self
                .random
                .sample(self.valid_feature_indices.len() as i32, used_feature_cnt);
            for &i in &sampled_indices {
                let used_feature = self.valid_feature_indices[i as usize];
                ret[used_feature as usize] = 1;
            }
        }
        ret
    }

    /// `const std::vector<int8_t>& is_feature_used_bytree() const`
    /// (col_sampler.hpp:181-183): the per-tree used flag for `inner_feature`.
    pub fn is_feature_used_bytree(&self, inner_feature: i32) -> bool {
        self.is_feature_used
            .get(inner_feature as usize)
            .copied()
            .unwrap_or(0)
            != 0
    }

    /// Whether this sampler ever advances the PRNG / disables any feature
    /// (`feature_fraction < 1.0 || feature_fraction_bynode < 1.0`). The spine's
    /// `feature_fraction == feature_fraction_bynode == 1.0` makes this `false`, so
    /// the learner can skip the per-node draw entirely (no RNG advance) and keep
    /// the Plan-03 behavior bit-identical.
    pub fn is_active(&self) -> bool {
        !(self.fraction_bytree >= 1.0) || !(self.fraction_bynode >= 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `get_cnt` == `max(round_int(total*fraction), min(1,total))`, including the
    /// `min(1,total)` floor and the f32-`0.5` rounding.
    #[test]
    fn get_cnt_matches_cpp_formula() {
        // round_int(10 * 0.5) = round_int(5.0) = 5; max(5, 1) = 5.
        assert_eq!(ColSampler::get_cnt(10, 0.5), 5);
        // round_int(10 * 0.34) = round_int(3.4) = 3.
        assert_eq!(ColSampler::get_cnt(10, 0.34), 3);
        // round_int(10 * 0.349) = round_int(3.49) = 3.
        assert_eq!(ColSampler::get_cnt(10, 0.349), 3);
        // round_int(10 * 0.35) = round_int(3.5) = 4 (the 0.5 rounds up).
        assert_eq!(ColSampler::get_cnt(10, 0.35), 4);
        // min(1,total) floor: a tiny fraction still draws at least 1.
        assert_eq!(ColSampler::get_cnt(10, 0.0), 1);
        assert_eq!(ColSampler::get_cnt(10, 0.001), 1);
        // total == 0 -> min(1,0) = 0 floor, round_int(0) = 0 -> 0.
        assert_eq!(ColSampler::get_cnt(0, 0.5), 0);
        // round_int(3 * 1.0) = 3 == total.
        assert_eq!(ColSampler::get_cnt(3, 1.0), 3);
    }

    /// `feature_fraction == 1.0` draws NOTHING in `reset_by_tree` (no RNG advance,
    /// every feature used — the spine path is unchanged).
    #[test]
    fn fraction_one_resets_without_rng_advance() {
        let valid: Vec<i32> = (0..5).collect();
        // Build the sampler (which calls reset_by_tree once) at fraction 1.0.
        let sampler = ColSampler::new(1.0, 1.0, 42, valid.clone(), 5);
        // Every feature is used; the sampler is inactive (no draws happen).
        assert!(!sampler.is_active(), "fraction 1.0 sampler must be inactive");
        for f in 0..5 {
            assert!(
                sampler.is_feature_used_bytree(f),
                "fraction 1.0 keeps feature {f} used"
            );
        }

        // Compare RNG state: a fresh Random(42) that drew NOTHING must equal the
        // sampler's untouched stream. We assert by drawing the SAME first sample
        // from both and comparing — the sampler must NOT have advanced its PRNG.
        let mut sampler2 = ColSampler::new(1.0, 1.0, 42, valid, 5);
        let mut ref_rng = Random::new(42);
        // After construction at fraction 1.0, the sampler's PRNG is pristine: a
        // get_by_node at bynode 1.0 also draws nothing, so a subsequent manual draw
        // off a reference Random(42) must match what the sampler would draw next.
        let _ = sampler2.get_by_node(); // bynode 1.0 -> no draw
        let want = ref_rng.sample(5, 3);
        let got = sampler2.random.sample(5, 3);
        assert_eq!(got, want, "fraction/bynode 1.0 must not advance the PRNG");
    }

    /// `feature_fraction < 1.0` advances the RNG EXACTLY ONCE per `reset_by_tree`
    /// (one `Sample` draw), and the selected features match an independent
    /// `Random::sample` of the same `(n, k)`.
    #[test]
    fn fraction_lt_one_draws_once_per_reset() {
        let valid: Vec<i32> = (0..8).collect();
        let frac = 0.5; // used_cnt_bytree = get_cnt(8, 0.5) = round_int(4.0) = 4.
        let sampler = ColSampler::new(frac, 1.0, 7, valid, 8);
        assert!(sampler.is_active());

        // Independently reproduce the one Sample draw the ctor's reset_by_tree made.
        let mut ref_rng = Random::new(7);
        let cnt = ColSampler::get_cnt(8, frac);
        assert_eq!(cnt, 4);
        let selected_positions = ref_rng.sample(8, cnt);
        // is_feature_used_ must be exactly the selected REAL features (== positions
        // here since valid_feature_indices_ = 0..8 identity).
        for f in 0..8 {
            let want = selected_positions.contains(&f);
            assert_eq!(
                sampler.is_feature_used_bytree(f),
                want,
                "feature {f} used flag must match the single Sample draw"
            );
        }
    }

    /// `get_by_node` draw ORDER parity: within one split decision the learner draws
    /// for the SMALLER leaf, THEN the larger leaf. Each call advances the shared
    /// PRNG once; the two draws must equal an independent reference advancing the
    /// SAME `Random` in the SAME order (smaller-then-larger).
    #[test]
    fn get_by_node_draw_order_is_smaller_then_larger() {
        let valid: Vec<i32> = (0..6).collect();
        // bytree 1.0 (no per-tree reset draw), bynode 0.5 (per-node draws).
        let mut sampler = ColSampler::new(1.0, 0.5, 99, valid.clone(), 6);
        assert!(sampler.is_active(), "bynode < 1.0 makes the sampler active");

        // SMALLER-leaf draw, then LARGER-leaf draw (the call order).
        let smaller = sampler.get_by_node();
        let larger = sampler.get_by_node();

        // Reference: a fresh Random(99) drawing the SAME two Samples in order. Since
        // bytree == 1.0, need_reset_bytree is false -> get_by_node samples from
        // valid_feature_indices_ (size 6) with get_cnt(6, 0.5) = round_int(3.0) = 3.
        let mut ref_rng = Random::new(99);
        let cnt = ColSampler::get_cnt(6, 0.5);
        assert_eq!(cnt, 3);
        let s_ref = ref_rng.sample(6, cnt); // first (smaller)
        let l_ref = ref_rng.sample(6, cnt); // second (larger)

        let to_set = |flags: &[i8]| -> Vec<i32> {
            flags
                .iter()
                .enumerate()
                .filter(|&(_, &v)| v != 0)
                .map(|(i, _)| i as i32)
                .collect()
        };
        assert_eq!(
            to_set(&smaller),
            s_ref,
            "smaller-leaf node features must equal the FIRST Sample draw"
        );
        assert_eq!(
            to_set(&larger),
            l_ref,
            "larger-leaf node features must equal the SECOND Sample draw (order parity)"
        );
        // The two draws differ (the PRNG advanced between them) — proves order matters.
        assert_ne!(
            smaller, larger,
            "consecutive per-node draws advance the shared PRNG"
        );
    }
}
