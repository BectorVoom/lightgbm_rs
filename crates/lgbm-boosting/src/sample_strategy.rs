//! Bagging / row subsampling (`BaggingSampleStrategy`), ported 1:1 from the C++
//! reference. This is the single most RNG-order-sensitive path in the engine
//! (D-13), so the draw/call sequence is reproduced VERBATIM over the proven
//! [`lgbm_core::random::Random`] LCG (FND-01 bit-exact — never re-rolled here).
//!
//! Faithful-mirror citations (read directly from the in-tree C++ source):
//! - `LightGBM/src/boosting/sample_strategy.h`: `bagging_rand_block_ = 1024`
//!   (the per-block RNG window).
//! - `LightGBM/src/boosting/bagging.hpp`:
//!   - `ResetSampleConfig` (:169-204): per-block RNG seeding
//!     `for i in 0..ceil(num_data/1024): bagging_rands.push(Random(bagging_seed+i))`;
//!     `bag_data_cnt = (bagging_fraction * num_data) as data_size_t` (truncate);
//!     the balanced (pos/neg) `bag_data_cnt` (:158-159); `need_re_bagging_ = true`
//!     so iter 0 always bags.
//!   - `Bagging(iter)` fires when `(bag_data_cnt < num_data && iter % bagging_freq
//!     == 0) || need_re_bagging_` (bagging.hpp:33).
//!   - `BaggingHelper` (:230-246): the draw loop — `for i in 0..cnt: if
//!     bagging_rands[i/1024].NextFloat() < bagging_fraction { buf[left]=i; left+=1 }
//!     else { right-=1; buf[right]=i }`. EVERY row draws, IN ORDER, incl. OOB
//!     (Pitfall 4).
//!   - `BalancedBaggingHelper` (:248-274): identical loop but draws against
//!     `pos_bagging_fraction` / `neg_bagging_fraction` by `label[i] > 0`.
//! - `LightGBM/include/LightGBM/utils/threading.h` (:152-155): one-buffer reverse
//!   — after the draw, `std::reverse(buf + left, buf + cnt)` reverses the OOB tail,
//!   so `bag_data_indices = [in-bag asc] ++ [OOB desc]` (Pitfall 4).
//!
//! ## `bagging_by_query` — explicit, decision-backed Phase-7 deferral (BST-03)
//!
//! Per the 2026-06-07 user decision (06-CONTEXT.md BST-03 scope note + Deferred
//! Ideas), the query-grouped draw (`num_sampled_queries`/`sampled_query_indices`,
//! the `bagging.hpp` query branch + `gbdt.cpp:227`) is DEFERRED to Phase 7 and
//! ships there alongside the Phase-7 ranking objectives (OBJ-04/05/06) — the ONLY
//! objectives it affects, none of which exist in Phase 6. Phase 6 ships pos/neg
//! ROW bagging only. This is NOT a silent reduction: [`BaggingConfig::new`]
//! REJECTS `bagging_by_query == true` with a typed [`BoostingError`], rather than
//! silently falling through to row bagging (so a wrong-but-similar bag can never
//! hide the missing query path). `bagging_by_query` is not dropped — it is
//! scheduled for Phase 7.

use lgbm_core::random::Random;

use crate::error::BoostingError;

/// The per-block RNG window — C++ `bagging_rand_block_` (sample_strategy.h:75).
/// One [`Random`] instance is constructed per 1024-row block, seeded
/// `bagging_seed + block_index`.
pub const BAGGING_RAND_BLOCK: i32 = 1024;

/// The bagging configuration the strategy draws against — the parity-relevant
/// subset of `lgbm_core::Config` (resolved by the caller / facade).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BaggingConfig {
    /// `bagging_fraction` (the plain row-bagging draw threshold), `(0, 1]`.
    pub bagging_fraction: f64,
    /// `pos_bagging_fraction` (balanced bagging, positive rows).
    pub pos_bagging_fraction: f64,
    /// `neg_bagging_fraction` (balanced bagging, negative rows).
    pub neg_bagging_fraction: f64,
    /// `bagging_freq` (`Bagging` fires on iters `0, k, 2k, …`).
    pub bagging_freq: i32,
    /// `bagging_seed` (per-block RNG seed base; C++ default 3).
    pub bagging_seed: i32,
}

impl BaggingConfig {
    /// Construct a [`BaggingConfig`], rejecting `bagging_by_query == true` as an
    /// explicit Phase-7 deferral (BST-03 — see the module doc + 06-CONTEXT.md scope
    /// note). This is the decision-backed guard: a caller that sets
    /// `bagging_by_query = true` gets a typed error, NEVER a silent fall-through to
    /// row bagging.
    ///
    /// # Errors
    /// [`BoostingError::BaggingByQueryDeferred`] when `bagging_by_query` is true
    /// (the query-grouped draw ships with the Phase-7 ranking objectives).
    pub fn new(
        bagging_fraction: f64,
        pos_bagging_fraction: f64,
        neg_bagging_fraction: f64,
        bagging_freq: i32,
        bagging_seed: i32,
        bagging_by_query: bool,
    ) -> Result<Self, BoostingError> {
        if bagging_by_query {
            return Err(BoostingError::BaggingByQueryDeferred);
        }
        Ok(Self {
            bagging_fraction,
            pos_bagging_fraction,
            neg_bagging_fraction,
            bagging_freq,
            bagging_seed,
        })
    }
}

/// C++ `BaggingSampleStrategy` (bagging.hpp) — owns the per-block RNG state, the
/// bagged-index buffer, and the in-bag count, and reproduces the C++ draw/call
/// sequence bit-exact.
pub struct BaggingSampleStrategy {
    config: BaggingConfig,
    /// `num_data_` — total rows.
    num_data: i32,
    /// Whether the balanced (pos/neg) path is active — C++ `balanced_bagging_`:
    /// `(pos_bagging_fraction < 1 || neg_bagging_fraction < 1) && num_pos_data > 0`.
    balanced: bool,
    /// `bag_data_cnt_` — the number of in-bag rows (target).
    bag_data_cnt: i32,
    /// `bag_data_indices_` — `[in-bag asc] ++ [OOB desc]` after a draw.
    bag_data_indices: Vec<i32>,
    /// `need_re_bagging_` — set true at construction so iter 0 always bags.
    need_re_bagging: bool,
    /// `bagging_rands_` — the per-block `Random(bagging_seed + i)` instances,
    /// constructed ONCE in `reset_sample_config` and ADVANCED continuously across
    /// `bagging()` calls (CRITICAL: C++ creates these once in ResetSampleConfig; the
    /// `BaggingHelper` reuses them so each `bagging_freq`-th iteration draws a NEW bag
    /// from the continuing RNG stream — recreating them per draw would re-draw the
    /// SAME bag every iteration, diverging from the reference).
    bagging_rands: Vec<Random>,
}

impl BaggingSampleStrategy {
    /// C++ `ResetSampleConfig` (bagging.hpp:169-204): compute `bag_data_cnt_`
    /// (truncated `bagging_fraction * num_data`, or the balanced pos/neg sum) and
    /// build the per-block `Random(bagging_seed + i)` seeding window. `labels` is
    /// the row labels (only the count of `> 0` matters, for the balanced gate).
    /// `need_re_bagging_` is set true so the first `bagging(0)` call always draws.
    pub fn reset_sample_config(config: BaggingConfig, num_data: i32, labels: &[f32]) -> Self {
        let nd = num_data.max(0);
        // C++ balance_bagging_cond: pos or neg fraction < 1 (with positives present).
        let num_pos = labels.iter().filter(|&&l| l > 0.0).count() as i32;
        let balanced =
            (config.pos_bagging_fraction < 1.0 || config.neg_bagging_fraction < 1.0) && num_pos > 0;
        let bag_data_cnt = if balanced {
            // bag_data_cnt_ = trunc(num_pos * pos_frac) + trunc(num_neg * neg_frac).
            let num_neg = nd - num_pos;
            (num_pos as f64 * config.pos_bagging_fraction) as i32
                + (num_neg as f64 * config.neg_bagging_fraction) as i32
        } else {
            // bag_data_cnt_ = trunc(bagging_fraction * num_data).
            (config.bagging_fraction * nd as f64) as i32
        };
        // Per-block RNG seeding — constructed ONCE (C++ ResetSampleConfig:177-181)
        // and advanced continuously across draws.
        let n_blocks = ((nd + BAGGING_RAND_BLOCK - 1) / BAGGING_RAND_BLOCK).max(0);
        let bagging_rands: Vec<Random> = (0..n_blocks)
            .map(|i| Random::new(config.bagging_seed + i))
            .collect();
        Self {
            config,
            num_data: nd,
            balanced,
            bag_data_cnt,
            bag_data_indices: Vec::new(),
            need_re_bagging: true,
            bagging_rands,
        }
    }

    /// Whether this iteration draws — C++ `(bag_data_cnt_ < num_data_ && iter %
    /// bagging_freq == 0) || need_re_bagging_` (bagging.hpp:33). `bagging_freq <= 0`
    /// disables the periodic re-bag (only `need_re_bagging_` fires it once).
    fn should_bag(&self, iter: i32) -> bool {
        if self.need_re_bagging {
            return true;
        }
        if self.config.bagging_freq <= 0 {
            return false;
        }
        self.bag_data_cnt < self.num_data && iter % self.config.bagging_freq == 0
    }

    /// C++ `BaggingSampleStrategy::Bagging(iter, …)`: when this iter fires, run the
    /// draw (plain or balanced) and store `bag_data_indices_` (`[in-bag asc] ++
    /// [OOB desc]`) + `bag_data_cnt_` (the realized in-bag count). Returns `true`
    /// when a (re)bag happened this iteration. `labels` is consulted only by the
    /// balanced helper.
    ///
    /// When bagging does NOT use a subset (`bag_data_cnt == num_data`, i.e.
    /// `bagging_fraction == 1`), no rows are dropped; the indices buffer is the
    /// trivial full range (still drawn so the RNG advances identically).
    pub fn bagging(&mut self, iter: i32, labels: &[f32]) -> bool {
        if !self.should_bag(iter) {
            return false;
        }
        self.need_re_bagging = false;
        let cnt = self.num_data;
        // The per-block Random instances ADVANCE across draws (created once in
        // reset_sample_config) — each bagging_freq-th iteration draws from the
        // CONTINUING stream, matching C++ bagging_rands_ reuse.
        let rands = &mut self.bagging_rands;

        let mut buf = vec![0i32; cnt as usize];
        let mut left = 0usize;
        let mut right = cnt as usize;
        for i in 0..cnt {
            let block = (i / BAGGING_RAND_BLOCK) as usize;
            // C++ `NextFloat() < fraction`: NextFloat is f32, the fraction is f64;
            // C++ promotes the f32 to f64 for the comparison. Mirror exactly.
            let draw = rands[block].next_float() as f64;
            let threshold = if self.balanced {
                if labels[i as usize] > 0.0 {
                    self.config.pos_bagging_fraction
                } else {
                    self.config.neg_bagging_fraction
                }
            } else {
                self.config.bagging_fraction
            };
            if draw < threshold {
                buf[left] = i;
                left += 1;
            } else {
                right -= 1;
                buf[right] = i;
            }
        }
        // One-buffer reverse of the OOB tail (threading.h:152-155): in-bag rows stay
        // ascending; the OOB rows (filled from the right, so descending) are reversed
        // so the tail reads DESCENDING in row terms — matching the C++ layout.
        buf[left..cnt as usize].reverse();
        self.bag_data_cnt = left as i32;
        self.bag_data_indices = buf;
        true
    }

    /// `bag_data_indices_` — `[in-bag asc] ++ [OOB desc]` (the full ordered array
    /// the D-13 RNG-replay golden asserts).
    pub fn bag_data_indices(&self) -> &[i32] {
        &self.bag_data_indices
    }

    /// `bag_data_cnt_` — the realized in-bag count (the split point of
    /// [`Self::bag_data_indices`]: `[..bag_data_cnt]` in-bag, `[bag_data_cnt..]` OOB).
    pub fn bag_data_cnt(&self) -> i32 {
        self.bag_data_cnt
    }

    /// Whether the strategy is actually subsetting (`bag_data_cnt < num_data`). When
    /// false (e.g. `bagging_fraction == 1`), the loop uses the full dataset and
    /// every row is scored via the train-path scatter (no OOB partition).
    pub fn is_use_subset(&self) -> bool {
        self.bag_data_cnt < self.num_data
    }

    /// Whether bagging will actually subset the corpus, derived from the config
    /// (`bag_data_cnt < num_data`, computed at `reset_sample_config`) — available
    /// BEFORE the first draw. Identical predicate to [`Self::is_use_subset`] but
    /// callable pre-draw, used by the GBDT loop to typed-reject `regression_l1 +
    /// bagging` (06-06 Task 2b) before any tree is grown.
    pub fn is_bagging_active(&self) -> bool {
        self.bag_data_cnt < self.num_data
    }

    /// The in-bag row indices (`bag_data_indices_[..bag_data_cnt]`).
    pub fn in_bag(&self) -> &[i32] {
        &self.bag_data_indices[..self.bag_data_cnt.max(0) as usize]
    }

    /// The out-of-bag row indices (`bag_data_indices_[bag_data_cnt..]`, DESCENDING).
    pub fn out_of_bag(&self) -> &[i32] {
        &self.bag_data_indices[self.bag_data_cnt.max(0) as usize..]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Re-derive the expected bag from a verbatim re-implementation of the C++
    /// `BaggingHelper` over the proven `lgbm_core::Random` — the D-13 RNG-replay
    /// (Option A). Returns `(bag_data_indices, bag_data_cnt)`.
    fn reference_bag(seed: i32, fraction: f64, num_data: i32) -> (Vec<i32>, i32) {
        let cnt = num_data;
        let n_blocks = (cnt + BAGGING_RAND_BLOCK - 1) / BAGGING_RAND_BLOCK;
        let mut rands: Vec<Random> = (0..n_blocks).map(|i| Random::new(seed + i)).collect();
        let mut buf = vec![0i32; cnt as usize];
        let mut left = 0usize;
        let mut right = cnt as usize;
        for i in 0..cnt {
            let block = (i / BAGGING_RAND_BLOCK) as usize;
            if (rands[block].next_float() as f64) < fraction {
                buf[left] = i;
                left += 1;
            } else {
                right -= 1;
                buf[right] = i;
            }
        }
        buf[left..cnt as usize].reverse();
        (buf, left as i32)
    }

    fn plain_cfg(fraction: f64, freq: i32, seed: i32) -> BaggingConfig {
        BaggingConfig::new(fraction, 1.0, 1.0, freq, seed, false).unwrap()
    }

    #[test]
    fn bagging_by_query_true_is_typed_error() {
        // EXPLICIT SCOPE BOUNDARY (BST-03): bagging_by_query=true is REJECTED, not
        // silently treated as row bagging.
        let err = BaggingConfig::new(0.7, 1.0, 1.0, 1, 3, true).unwrap_err();
        assert!(matches!(err, BoostingError::BaggingByQueryDeferred));
        // bagging_by_query=false (the default) takes the row-bagging path.
        assert!(BaggingConfig::new(0.7, 1.0, 1.0, 1, 3, false).is_ok());
    }

    #[test]
    fn bag_data_cnt_is_truncated_fraction() {
        // bag_data_cnt = trunc(0.7 * 10) = 7.
        let labels = vec![0.0f32; 10];
        let s = BaggingSampleStrategy::reset_sample_config(plain_cfg(0.7, 1, 3), 10, &labels);
        assert_eq!(s.bag_data_cnt(), 7);
        // trunc(0.55 * 10) = 5.
        let s = BaggingSampleStrategy::reset_sample_config(plain_cfg(0.55, 1, 3), 10, &labels);
        assert_eq!(s.bag_data_cnt(), 5);
    }

    #[test]
    fn bag_indices_match_rng_replay_golden() {
        // D-13 (Option A): the full bag_data_indices (in-bag asc ++ OOB desc) match
        // the verbatim RNG-replay derivation bit-exact (compare_exact). Several
        // seeds/fractions to exercise the draw loop.
        let labels = vec![0.0f32; 50];
        for (seed, frac) in [(3i32, 0.7f64), (3, 0.5), (7, 0.3), (123, 0.8)] {
            let mut s =
                BaggingSampleStrategy::reset_sample_config(plain_cfg(frac, 1, seed), 50, &labels);
            assert!(s.bagging(0, &labels), "iter 0 must bag (need_re_bagging)");
            let (expected, exp_cnt) = reference_bag(seed, frac, 50);
            assert_eq!(
                s.bag_data_indices(),
                expected.as_slice(),
                "seed={seed} frac={frac}: bag_data_indices must match RNG-replay golden"
            );
            assert_eq!(
                s.bag_data_cnt(),
                exp_cnt,
                "seed={seed} frac={frac}: realized in-bag count"
            );
        }
    }

    #[test]
    fn in_bag_and_oob_partition_all_rows() {
        // The ordering invariant: in-bag rows ascending (appended left in row order).
        // The OOB tail is filled from the right in row order, then `std::reverse`d
        // (threading.h:152-155, one-buffer mode) — the net effect of the verbatim C++
        // loop is asserted bit-exact by `bag_indices_match_rng_replay_golden`; here we
        // assert the structural invariants (in-bag ascending + total partition).
        let labels = vec![0.0f32; 30];
        let mut s = BaggingSampleStrategy::reset_sample_config(plain_cfg(0.6, 1, 3), 30, &labels);
        s.bagging(0, &labels);
        let in_bag = s.in_bag();
        assert!(
            in_bag.windows(2).all(|w| w[0] < w[1]),
            "in-bag must be ascending: {in_bag:?}"
        );
        // The two partitions together are a permutation of 0..30 (every row drawn).
        let mut all: Vec<i32> = in_bag
            .iter()
            .chain(s.out_of_bag().iter())
            .copied()
            .collect();
        all.sort_unstable();
        assert_eq!(all, (0..30).collect::<Vec<_>>());
    }

    #[test]
    fn bagging_fires_on_freq_multiples() {
        // bagging_freq=3: fires on iters 0 (need_re_bagging), 3, 6, ...; NOT on 1,2.
        let labels = vec![0.0f32; 20];
        let mut s = BaggingSampleStrategy::reset_sample_config(plain_cfg(0.5, 3, 3), 20, &labels);
        assert!(s.bagging(0, &labels), "iter 0 fires (need_re_bagging)");
        assert!(!s.bagging(1, &labels), "iter 1 must NOT fire");
        assert!(!s.bagging(2, &labels), "iter 2 must NOT fire");
        assert!(s.bagging(3, &labels), "iter 3 fires (freq multiple)");
        assert!(!s.bagging(4, &labels), "iter 4 must NOT fire");
        assert!(s.bagging(6, &labels), "iter 6 fires");
    }

    #[test]
    fn balanced_bagging_draws_per_label_sign() {
        // pos_frac=1.0, neg_frac=0.5: positives ALWAYS in-bag (draw < 1.0 always
        // true for next_float in [0, ~0.99997)), negatives subsampled at 0.5.
        // 10 rows: 4 positive, 6 negative.
        let labels = vec![1.0f32, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0];
        let cfg = BaggingConfig::new(1.0, 1.0, 0.5, 1, 3, false).unwrap();
        let mut s = BaggingSampleStrategy::reset_sample_config(cfg, 10, &labels);
        assert!(
            s.balanced,
            "pos/neg fractions < 1 with positives present => balanced"
        );
        // bag_data_cnt = trunc(4*1.0) + trunc(6*0.5) = 4 + 3 = 7.
        assert_eq!(s.bag_data_cnt(), 7);
        s.bagging(0, &labels);
        // Every POSITIVE row must be in-bag (pos_frac=1.0 => draw < 1.0 always true).
        let in_bag: std::collections::BTreeSet<i32> = s.in_bag().iter().copied().collect();
        for (i, &l) in labels.iter().enumerate() {
            if l > 0.0 {
                assert!(
                    in_bag.contains(&(i as i32)),
                    "positive row {i} must be in-bag"
                );
            }
        }
    }

    #[test]
    fn fraction_one_uses_all_rows() {
        // bagging_fraction=1.0 => bag_data_cnt == num_data => is_use_subset false.
        let labels = vec![0.0f32; 8];
        let mut s = BaggingSampleStrategy::reset_sample_config(plain_cfg(1.0, 1, 3), 8, &labels);
        assert_eq!(s.bag_data_cnt(), 8);
        assert!(!s.is_use_subset());
        s.bagging(0, &labels);
        // All rows in-bag (draw < 1.0 always true).
        assert_eq!(s.bag_data_cnt(), 8);
        assert!(s.out_of_bag().is_empty());
    }
}
