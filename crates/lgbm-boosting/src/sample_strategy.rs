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
//! ## `bagging_by_query` — query-grouped bagging (BST-03, un-deferred in 07-09)
//!
//! When `bagging_by_query == true` the draw's minimal unit is a whole QUERY, not a
//! row (`bagging.hpp:52-104`): the per-block RNG draws over `num_queries_` against
//! `bagging_fraction`, the in-bag queries are expanded to their `[query_boundaries[q],
//! query_boundaries[q+1])` row ranges, and `sampled_query_boundaries_` is the
//! prefix-sum of in-bag query sizes. This shipped DEFERRED in Phase 6 (the now-
//! removed `BoostingError::BaggingByQueryDeferred`) pending the Phase-7 ranking
//! objectives; 07-09 un-defers it alongside lambdarank/rank_xendcg (D-02 step 5,
//! D-03). The query path reuses the SAME build-once `bagging_rands_` block-1024 RNG
//! discipline as row bagging (the 07-01 bit-exact bagging carries over).

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
    /// `bagging_by_query` — when true, the draw's minimal unit is a whole QUERY
    /// (un-deferred in 07-09; see the module doc). Requires the caller to provide
    /// `query_boundaries` to [`BaggingSampleStrategy::reset_sample_config`].
    pub bagging_by_query: bool,
}

impl BaggingConfig {
    /// Construct a [`BaggingConfig`]. `bagging_by_query` is now a supported field
    /// (07-09 un-deferred the query-grouped draw alongside the ranking objectives);
    /// the `Result` return is retained for API stability with the Phase-6 callers,
    /// and currently never errs.
    ///
    /// # Errors
    /// None at present — the signature returns `Result` so callers established in
    /// Phase 6 (which `?`-propagate this) keep compiling unchanged.
    #[allow(clippy::unnecessary_wraps)]
    pub fn new(
        bagging_fraction: f64,
        pos_bagging_fraction: f64,
        neg_bagging_fraction: f64,
        bagging_freq: i32,
        bagging_seed: i32,
        bagging_by_query: bool,
    ) -> Result<Self, BoostingError> {
        Ok(Self {
            bagging_fraction,
            pos_bagging_fraction,
            neg_bagging_fraction,
            bagging_freq,
            bagging_seed,
            bagging_by_query,
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
    /// `query_boundaries_` (prefix-sum, len num_queries+1) — present only when
    /// `bagging_by_query` is active; empty otherwise. Captured at
    /// `reset_sample_config` from the dataset metadata.
    query_boundaries: Vec<i32>,
    /// `bag_query_indices_` — `[in-bag query asc] ++ [OOB query desc]` after a query
    /// draw (the query analog of `bag_data_indices_`). Empty in row mode.
    bag_query_indices: Vec<i32>,
    /// `num_sampled_queries_` — the in-bag query count (split point of
    /// `bag_query_indices`).
    num_sampled_queries: i32,
    /// `sampled_query_boundaries_` — the prefix-sum of in-bag query sizes
    /// (`len num_sampled_queries + 1`), so the kept rows of sampled query `s` are
    /// `bag_data_indices[sampled_query_boundaries[s]..sampled_query_boundaries[s+1]]`.
    sampled_query_boundaries: Vec<i32>,
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
            query_boundaries: Vec::new(),
            bag_query_indices: Vec::new(),
            num_sampled_queries: 0,
            sampled_query_boundaries: Vec::new(),
        }
    }

    /// C++ `ResetSampleConfig` with `bagging_by_query == true` (bagging.hpp:171-176):
    /// the query-grouped variant. The draw's minimal unit is a whole QUERY; the
    /// per-block `bagging_rands_` are STILL sized by `num_data` (bagging.hpp:178-181,
    /// the seeding loop uses `num_data_`, NOT `num_queries_`), so the RNG stream is
    /// identical to row bagging up to which positions are consumed.
    ///
    /// `query_boundaries` is the prefix-sum group array (`[0, …, num_data]`,
    /// `num_queries + 1` entries). The balanced (pos/neg) path is NOT used in the
    /// query branch (C++ always calls the plain `BaggingHelper` there, bagging.hpp:58).
    pub fn reset_sample_config_with_queries(
        config: BaggingConfig,
        num_data: i32,
        query_boundaries: &[i32],
    ) -> Self {
        let nd = num_data.max(0);
        let num_queries = (query_boundaries.len() as i32 - 1).max(0);
        // bag_data_cnt_ = trunc(bagging_fraction * num_data) initially (bagging.hpp:161);
        // overwritten to the realized sampled row count after the first draw.
        let bag_data_cnt = (config.bagging_fraction * nd as f64) as i32;
        // Per-block RNG sized by num_data (bagging.hpp:178-181), built ONCE.
        let n_blocks = ((nd + BAGGING_RAND_BLOCK - 1) / BAGGING_RAND_BLOCK).max(0);
        let bagging_rands: Vec<Random> = (0..n_blocks)
            .map(|i| Random::new(config.bagging_seed + i))
            .collect();
        Self {
            config,
            num_data: nd,
            balanced: false,
            bag_data_cnt,
            bag_data_indices: Vec::new(),
            need_re_bagging: true,
            bagging_rands,
            query_boundaries: query_boundaries.to_vec(),
            bag_query_indices: vec![0i32; nd as usize],
            num_sampled_queries: 0,
            sampled_query_boundaries: vec![0i32; (num_queries + 1) as usize],
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

    /// C++ `BaggingSampleStrategy::Bagging` with `bagging_by_query == true`
    /// (bagging.hpp:52-104), ported 1:1. Draws QUERIES (not rows) via the per-block
    /// `bagging_rands_` against `bagging_fraction` over `num_queries_`, then expands
    /// each in-bag query to its `[query_boundaries[q], query_boundaries[q+1])` row
    /// range, building `sampled_query_boundaries_` (the prefix-sum of in-bag query
    /// sizes) and `bag_data_indices_` (the expanded kept rows in query order).
    /// Returns `true` when a (re)bag happened.
    ///
    /// The query draw does NOT do the one-buffer reverse (the C++ query branch's
    /// single bagging-runner block has no OOB tail to reverse); `bag_query_indices`
    /// holds `[in-bag query asc] ++ [OOB query desc-filled]` and only the in-bag
    /// prefix `[0, num_sampled_queries)` is expanded.
    pub fn bagging_by_query(&mut self, iter: i32) -> bool {
        if !self.should_bag(iter) {
            return false;
        }
        self.need_re_bagging = false;
        let num_queries = (self.query_boundaries.len() as i32 - 1).max(0);
        // BaggingHelper over num_queries: draw each query index IN ORDER against
        // bagging_fraction (bagging.hpp:230-246), the per-block RNG keyed by the
        // QUERY index / 1024 (cur_idx / bagging_rand_block_). cur_left filled left,
        // OOB filled from the right.
        let rands = &mut self.bagging_rands;
        let mut cur_left = 0usize;
        let mut cur_right = num_queries as usize;
        for q in 0..num_queries {
            let block = (q / BAGGING_RAND_BLOCK) as usize;
            let draw = rands[block].next_float() as f64;
            if draw < self.config.bagging_fraction {
                self.bag_query_indices[cur_left] = q;
                cur_left += 1;
            } else {
                cur_right -= 1;
                self.bag_query_indices[cur_right] = q;
            }
        }
        self.num_sampled_queries = cur_left as i32;

        // Build sampled_query_boundaries_ as the prefix-sum of in-bag query sizes
        // (bagging.hpp:62-91). sampled_query_boundaries_[0] = 0.
        self.sampled_query_boundaries.clear();
        self.sampled_query_boundaries.push(0);
        for s in 0..cur_left {
            let q = self.bag_query_indices[s] as usize;
            let q_size = self.query_boundaries[q + 1] - self.query_boundaries[q];
            let prev = *self.sampled_query_boundaries.last().unwrap();
            self.sampled_query_boundaries.push(prev + q_size);
        }
        let total_rows = *self.sampled_query_boundaries.last().unwrap();
        self.bag_data_cnt = total_rows;

        // Expand the in-bag queries to their row ranges (bagging.hpp:93-103).
        let mut bag_data_indices = vec![0i32; total_rows as usize];
        for s in 0..cur_left {
            let q = self.bag_query_indices[s] as usize;
            let data_start = self.query_boundaries[q];
            let data_end = self.query_boundaries[q + 1];
            let sampled_start = self.sampled_query_boundaries[s];
            for (off, i) in (data_start..data_end).enumerate() {
                bag_data_indices[(sampled_start + off as i32) as usize] = i;
            }
        }
        self.bag_data_indices = bag_data_indices;
        true
    }

    /// `bag_query_indices_` — `[in-bag query asc] ++ [OOB query desc]` (the query
    /// analog of [`Self::bag_data_indices`]; the RNG-replay golden asserts the in-bag
    /// prefix bit-exact).
    pub fn bag_query_indices(&self) -> &[i32] {
        &self.bag_query_indices
    }

    /// `num_sampled_queries_` — the realized in-bag query count.
    pub fn num_sampled_queries(&self) -> i32 {
        self.num_sampled_queries
    }

    /// The in-bag (sampled) query indices (`bag_query_indices_[..num_sampled_queries]`).
    pub fn sampled_query_indices(&self) -> &[i32] {
        &self.bag_query_indices[..self.num_sampled_queries.max(0) as usize]
    }

    /// `sampled_query_boundaries_` — the prefix-sum of in-bag query sizes
    /// (`len num_sampled_queries + 1`).
    pub fn sampled_query_boundaries(&self) -> &[i32] {
        &self.sampled_query_boundaries
    }

    /// `bag_data_indices_` — `[in-bag asc] ++ [OOB desc]` (the full ordered array
    /// the D-13 RNG-replay golden asserts). In query mode this is the expanded kept
    /// rows in query order (length `bag_data_cnt`).
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

/// C++ `ArrayArgs<score_t>::ArgMaxAtK` (`include/LightGBM/utils/array_args.h:131`)
/// with its `Partition` helper (:101), ported VERBATIM. GOSS uses this (NOT a full
/// sort) to find the top-k grad-magnitude threshold; the tie behavior of this
/// quickselect-style partition differs from a stable sort, so it MUST be mirrored
/// bit-for-bit (RESEARCH §"Re-sorting where C++ uses nth_element/ArgMaxAtK").
///
/// `k` is a 0-based index: `k = 0` selects the maximum. After the call, the element
/// at position `k` is the one that would be there in fully-descending order, and
/// `arr[k]` is the threshold GOSS reads (`tmp_gradients[top_k - 1]`).
///
/// Operates in place over `arr[start..end]`. `next_float`-equivalent comparisons use
/// `>` exactly as C++ (`while (ref[++i] > v)`), so f32 NaN/tie handling matches.
fn arg_max_at_k(arr: &mut [f32], start: i32, end: i32, k: i32) -> i32 {
    if start >= end - 1 {
        return start;
    }
    let (l, r) = partition(arr, start, end);
    if (k > l && k < r) || (l == start - 1 && r == end - 1) {
        k
    } else if k <= l {
        arg_max_at_k(arr, start, l + 1, k)
    } else {
        arg_max_at_k(arr, r, end, k)
    }
}

/// C++ `ArrayArgs<score_t>::Partition` (`array_args.h:101-128`), ported verbatim —
/// a descending three-way (Bentley–McIlroy) partition around the pivot `arr[end-1]`.
/// Returns `(l, r)`: elements `> pivot` end up in `[start, l]`, elements `== pivot`
/// in `(l, r)`, elements `< pivot` in `[r, end)`. Every index/swap detail is
/// load-bearing for the ArgMaxAtK tie behavior.
// The `!(a > v)` / `!(v > b)` forms are a VERBATIM port of the C++ `while (ref[++i] >
// v)` / `while (v > ref[--j])` loop conditions (array_args.h:114-115). Rewriting them
// as `<=`/`>=` would change the f32 NaN/tie semantics the partition relies on, so the
// faithful negated form is retained.
#[allow(clippy::neg_cmp_op_on_partial_ord)]
fn partition(arr: &mut [f32], start: i32, end: i32) -> (i32, i32) {
    let mut i = start - 1;
    let mut j = end - 1;
    let mut p = i;
    let mut q = j;
    if start >= end - 1 {
        return (start - 1, end);
    }
    let v = arr[(end - 1) as usize];
    loop {
        // while (ref[++i] > v) {}
        loop {
            i += 1;
            if !(arr[i as usize] > v) {
                break;
            }
        }
        // while (v > ref[--j]) { if (j == start) break; }
        loop {
            j -= 1;
            if !(v > arr[j as usize]) {
                break;
            }
            if j == start {
                break;
            }
        }
        if i >= j {
            break;
        }
        arr.swap(i as usize, j as usize);
        if arr[i as usize] == v {
            p += 1;
            arr.swap(p as usize, i as usize);
        }
        if v == arr[j as usize] {
            q -= 1;
            arr.swap(j as usize, q as usize);
        }
    }
    arr.swap(i as usize, (end - 1) as usize);
    j = i - 1;
    i += 1;
    let mut kk = start;
    while kk <= p {
        arr.swap(kk as usize, j as usize);
        kk += 1;
        j -= 1;
    }
    let mut kk = end - 2;
    while kk >= q {
        arr.swap(i as usize, kk as usize);
        i += 1;
        kk -= 1;
    }
    (j, i)
}

/// C++ `GOSSStrategy` (`src/boosting/goss.hpp`) — Gradient-based One-Side Sampling.
///
/// GOSS is a [`SampleStrategy`] sibling of [`BaggingSampleStrategy`]: instead of a
/// uniform row draw, it keeps the rows with the largest gradient magnitude (the
/// "top" rows) plus a random sample of the rest, and **amplifies the grad/hess of
/// the sampled rows in place** (`IsHessianChange() == true`, `goss.hpp:111-113`) so
/// the subsample stays an unbiased estimator. It therefore runs INSIDE
/// `train_one_iter` AFTER `get_gradients` and BEFORE the learner.
///
/// Faithful-mirror citations (read directly from the in-tree C++ source):
/// - `goss.hpp:33`: skip subsampling for `iter < (int)(1.0 / learning_rate)` —
///   returns the full set with grad/hess UNMODIFIED.
/// - `goss.hpp:95-98`: the per-block `Random(bagging_seed + i)` block-1024 array,
///   built ONCE in `ResetSampleConfig` and advanced across draws (the SAME RNG
///   discipline as bagging — STATE.md 06-05 CRITICAL fix; re-seeding per draw would
///   re-draw the same bag).
/// - `goss.hpp:116-165` (`Helper`): `top_k = max(1, (data_size_t)(cnt * top_rate))`,
///   `other_k = (data_size_t)(cnt * other_rate)`, `ArgMaxAtK(top_k - 1)` over
///   `|grad * hess|` for the threshold (NOT a full sort), then per row IN ORDER: if
///   `|g*h| >= threshold` keep (top); else if
///   `bagging_rands[idx/1024].NextFloat() < prob` keep AND amplify grad AND hess by
///   `multiply = (cnt - top_k) / other_k`; otherwise drop. `prob` is the running
///   `rest_need / rest_all` (`goss.hpp:148-151`) — draw order is load-bearing.
/// - `goss.hpp:85-86`: `CHECK_LE(top_rate + other_rate, 1)` and
///   `CHECK(top_rate > 0 && other_rate > 0)` (surfaced as a typed `Result`).
///
/// Single-class only here (`num_tree_per_iteration == 1`); the C++ multi-class
/// `|grad*hess|` is summed across the per-class stride (`goss.hpp:121-125,140-143`),
/// which is wired via `num_tree_per_iteration` for forward-compatibility but the
/// Phase-7 W4 spine validates `K = 1`.
pub struct GossSampleStrategy {
    /// `config_->top_rate`.
    top_rate: f64,
    /// `config_->other_rate`.
    other_rate: f64,
    /// `config_->learning_rate` — drives the `iter < 1/lr` skip window.
    learning_rate: f64,
    /// `num_data_`.
    num_data: i32,
    /// `num_tree_per_iteration_` (the per-class stride for `|grad*hess|`; 1 here).
    num_tree_per_iteration: i32,
    /// `bag_data_cnt_` — the realized kept-row count after a draw (== `num_data`
    /// before the first real draw / inside the skip window).
    bag_data_cnt: i32,
    /// `bag_data_indices_` — `[kept rows] ++ [dropped rows]` (kept = top ++ sampled,
    /// in row order; dropped filled from the right). Length `num_data`.
    bag_data_indices: Vec<i32>,
    /// `bagging_rands_` — per-block `Random(bagging_seed + i)`, built ONCE and
    /// advanced across draws (the bagging RNG discipline).
    bagging_rands: Vec<Random>,
}

impl std::fmt::Debug for GossSampleStrategy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // `Random` is intentionally not `Debug` (it is the parity-critical LCG); show
        // the config + realized counts instead of the per-block RNG state.
        f.debug_struct("GossSampleStrategy")
            .field("top_rate", &self.top_rate)
            .field("other_rate", &self.other_rate)
            .field("learning_rate", &self.learning_rate)
            .field("num_data", &self.num_data)
            .field("num_tree_per_iteration", &self.num_tree_per_iteration)
            .field("bag_data_cnt", &self.bag_data_cnt)
            .field("num_blocks", &self.bagging_rands.len())
            .finish()
    }
}

impl GossSampleStrategy {
    /// C++ `GOSSStrategy::ResetSampleConfig` (`goss.hpp:76-109`): validate the rate
    /// invariants, build the per-block RNG window ONCE, and initialize
    /// `bag_data_cnt_ = num_data_` (the "do not bag the first iterations" flag).
    ///
    /// # Errors
    /// [`BoostingError::GossConfig`] when `top_rate + other_rate > 1` or either rate
    /// is `<= 0` (the `goss.hpp:85-86` CHECKs).
    pub fn reset_sample_config(
        top_rate: f64,
        other_rate: f64,
        learning_rate: f64,
        num_data: i32,
        num_tree_per_iteration: i32,
        bagging_seed: i32,
    ) -> Result<Self, BoostingError> {
        // C++ CHECK_LE(top_rate + other_rate, 1.0f) (goss.hpp:85).
        if top_rate + other_rate > 1.0 {
            return Err(BoostingError::GossConfig {
                what: format!(
                    "top_rate + other_rate must be <= 1 (got {top_rate} + {other_rate} = {})",
                    top_rate + other_rate
                ),
            });
        }
        // C++ CHECK(top_rate > 0 && other_rate > 0) (goss.hpp:86).
        if !(top_rate > 0.0 && other_rate > 0.0) {
            return Err(BoostingError::GossConfig {
                what: format!(
                    "top_rate and other_rate must both be > 0 (got top_rate={top_rate}, other_rate={other_rate})"
                ),
            });
        }
        let nd = num_data.max(0);
        // Per-block RNG seeding — built ONCE (goss.hpp:94-98), advanced across draws.
        let n_blocks = ((nd + BAGGING_RAND_BLOCK - 1) / BAGGING_RAND_BLOCK).max(0);
        let bagging_rands: Vec<Random> = (0..n_blocks)
            .map(|i| Random::new(bagging_seed + i))
            .collect();
        Ok(Self {
            top_rate,
            other_rate,
            learning_rate,
            num_data: nd,
            num_tree_per_iteration: num_tree_per_iteration.max(1),
            bag_data_cnt: nd,
            bag_data_indices: vec![0i32; nd as usize],
            bagging_rands,
        })
    }

    /// The first iteration at which GOSS actually subsamples — C++
    /// `iter < static_cast<int>(1.0f / config_->learning_rate)` (`goss.hpp:33`).
    /// The cast is to `int` (truncation toward zero) over an `f32` reciprocal, which
    /// the port mirrors via `(1.0f32 / lr as f32) as i32`.
    fn skip_iters(&self) -> i32 {
        (1.0f32 / self.learning_rate as f32) as i32
    }

    /// C++ `GOSSStrategy::Bagging(iter, …)` (`goss.hpp:30-74`): when `iter` is past
    /// the skip window, run [`Self::helper`] over the WHOLE corpus (the C++
    /// `bagging_runner_.Run<true>` here is single-threaded with one block), which
    /// fills `bag_data_indices_` and AMPLIFIES the sampled rows' grad/hess in place.
    /// Returns whether subsampling happened this iteration.
    ///
    /// `gradients` / `hessians` are the class-major buffers (`num_data *
    /// num_tree_per_iteration`); they are MUTATED in place for the sampled rows.
    pub fn bagging(&mut self, iter: i32, gradients: &mut [f32], hessians: &mut [f32]) -> bool {
        // C++ goss.hpp:31 — bag_data_cnt_ = num_data_ before deciding.
        self.bag_data_cnt = self.num_data;
        // C++ goss.hpp:33 — not subsample for first iterations.
        if iter < self.skip_iters() {
            return false;
        }
        let cnt = self.num_data;
        let left_cnt = self.helper(0, cnt, gradients, hessians);
        self.bag_data_cnt = left_cnt;
        true
    }

    /// C++ `GOSSStrategy::Helper` (`goss.hpp:116-165`), ported 1:1.
    ///
    /// Computes per-row `|grad*hess|` (summed across the per-class stride), finds the
    /// top-k threshold via [`arg_max_at_k`], then walks the rows IN ORDER filling
    /// `bag_data_indices_` (`buffer`): top rows and randomly-sampled rest rows go to
    /// the left (kept), dropped rows to the right; sampled rows have their grad AND
    /// hess multiplied by `multiply = (cnt - top_k) / other_k`. Returns the kept-row
    /// count (`cur_left_cnt`).
    fn helper(
        &mut self,
        start: i32,
        cnt: i32,
        gradients: &mut [f32],
        hessians: &mut [f32],
    ) -> i32 {
        if cnt <= 0 {
            return 0;
        }
        let ntpi = self.num_tree_per_iteration as usize;
        let nd = self.num_data as usize;
        // tmp_gradients[i] = Σ_k |grad[k*nd + start + i] * hess[k*nd + start + i]|.
        let mut tmp_gradients = vec![0.0f32; cnt as usize];
        for (i, tmp) in tmp_gradients.iter_mut().enumerate() {
            for cur_tree_id in 0..ntpi {
                let idx = cur_tree_id * nd + start as usize + i;
                *tmp += (gradients[idx] * hessians[idx]).abs();
            }
        }
        // top_k = max(1, (data_size_t)(cnt * top_rate)); other_k = (data_size_t)(cnt * other_rate).
        let mut top_k = (cnt as f64 * self.top_rate) as i32;
        let other_k = (cnt as f64 * self.other_rate) as i32;
        top_k = top_k.max(1);
        // ArgMaxAtK(&tmp_gradients, 0, size, top_k - 1); threshold = tmp_gradients[top_k - 1].
        let n = tmp_gradients.len() as i32;
        arg_max_at_k(&mut tmp_gradients, 0, n, top_k - 1);
        let threshold = tmp_gradients[(top_k - 1) as usize];

        // multiply = (score_t)(cnt - top_k) / other_k  (f32 division — goss.hpp:133).
        let multiply = (cnt - top_k) as f32 / other_k as f32;
        let mut cur_left_cnt = 0i32;
        let mut cur_right_pos = cnt;
        let mut big_weight_cnt = 0i32;
        let buffer = &mut self.bag_data_indices;
        for i in 0..cnt {
            let cur_idx = start + i;
            // grad = Σ_k |grad[k*nd + cur_idx] * hess[k*nd + cur_idx]|.
            let mut grad = 0.0f32;
            for cur_tree_id in 0..ntpi {
                let idx = cur_tree_id * nd + cur_idx as usize;
                grad += (gradients[idx] * hessians[idx]).abs();
            }
            if grad >= threshold {
                buffer[cur_left_cnt as usize] = cur_idx;
                cur_left_cnt += 1;
                big_weight_cnt += 1;
            } else {
                let sampled = cur_left_cnt - big_weight_cnt;
                let rest_need = other_k - sampled;
                let rest_all = (cnt - i) - (top_k - big_weight_cnt);
                // prob = (rest_need) / (double)(rest_all)  (goss.hpp:151).
                let prob = rest_need as f64 / rest_all as f64;
                // C++ NextFloat() < prob: NextFloat is f32, prob is f64; C++ promotes
                // the f32 to f64 for the comparison. Mirror exactly.
                let block = (cur_idx / BAGGING_RAND_BLOCK) as usize;
                if (self.bagging_rands[block].next_float() as f64) < prob {
                    buffer[cur_left_cnt as usize] = cur_idx;
                    cur_left_cnt += 1;
                    for cur_tree_id in 0..ntpi {
                        let idx = cur_tree_id * nd + cur_idx as usize;
                        gradients[idx] *= multiply;
                        hessians[idx] *= multiply;
                    }
                } else {
                    cur_right_pos -= 1;
                    buffer[cur_right_pos as usize] = cur_idx;
                }
            }
        }
        cur_left_cnt
    }

    /// `bag_data_cnt_` — the realized kept-row count (the split point of
    /// [`Self::bag_data_indices`]).
    pub fn bag_data_cnt(&self) -> i32 {
        self.bag_data_cnt
    }

    /// `bag_data_indices_` — `[kept rows] ++ [dropped rows]` (the full ordered array
    /// the GOSS RNG-replay golden asserts).
    pub fn bag_data_indices(&self) -> &[i32] {
        &self.bag_data_indices
    }

    /// The kept (in-bag) row indices (`bag_data_indices_[..bag_data_cnt]`).
    pub fn in_bag(&self) -> &[i32] {
        &self.bag_data_indices[..self.bag_data_cnt.max(0) as usize]
    }

    /// The dropped (out-of-bag) row indices (`bag_data_indices_[bag_data_cnt..]`).
    pub fn out_of_bag(&self) -> &[i32] {
        &self.bag_data_indices[self.bag_data_cnt.max(0) as usize..]
    }

    /// Whether this iteration actually subsampled (`bag_data_cnt < num_data`). False
    /// inside the skip window (every row kept, grad/hess unmodified).
    pub fn is_use_subset(&self) -> bool {
        self.bag_data_cnt < self.num_data
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
    fn bagging_by_query_config_is_accepted() {
        // 07-09 un-deferred bagging_by_query: it is now a supported config field, not
        // a typed error. Both true and false construct cleanly.
        let cfg = BaggingConfig::new(0.7, 1.0, 1.0, 1, 3, true).unwrap();
        assert!(cfg.bagging_by_query);
        let cfg = BaggingConfig::new(0.7, 1.0, 1.0, 1, 3, false).unwrap();
        assert!(!cfg.bagging_by_query);
    }

    /// Verbatim re-implementation of the C++ query-grouped draw + expansion
    /// (bagging.hpp:52-104) over the proven `lgbm_core::Random` — the bagging_by_query
    /// RNG-replay reference. Returns (sampled_query_indices, sampled_query_boundaries,
    /// expanded_bag_data_indices).
    fn reference_bag_by_query(
        seed: i32,
        fraction: f64,
        num_data: i32,
        query_boundaries: &[i32],
    ) -> (Vec<i32>, Vec<i32>, Vec<i32>) {
        let num_queries = query_boundaries.len() as i32 - 1;
        // bagging_rands sized by num_data (bagging.hpp:178-181).
        let n_blocks = (num_data + BAGGING_RAND_BLOCK - 1) / BAGGING_RAND_BLOCK;
        let mut rands: Vec<Random> = (0..n_blocks).map(|i| Random::new(seed + i)).collect();
        let mut buf = vec![0i32; num_data as usize];
        let mut left = 0usize;
        let mut right = num_queries as usize;
        for q in 0..num_queries {
            let block = (q / BAGGING_RAND_BLOCK) as usize;
            if (rands[block].next_float() as f64) < fraction {
                buf[left] = q;
                left += 1;
            } else {
                right -= 1;
                buf[right] = q;
            }
        }
        let sampled: Vec<i32> = buf[..left].to_vec();
        let mut sqb = vec![0i32];
        for &q in &sampled {
            let sz = query_boundaries[q as usize + 1] - query_boundaries[q as usize];
            let prev = *sqb.last().unwrap();
            sqb.push(prev + sz);
        }
        let total = *sqb.last().unwrap();
        let mut expanded = vec![0i32; total as usize];
        for (s, &q) in sampled.iter().enumerate() {
            let ds = query_boundaries[q as usize];
            let de = query_boundaries[q as usize + 1];
            let ss = sqb[s];
            for (off, i) in (ds..de).enumerate() {
                expanded[(ss + off as i32) as usize] = i;
            }
        }
        (sampled, sqb, expanded)
    }

    #[test]
    fn bagging_by_query_matches_rng_replay_golden() {
        // The sampled query indices + sampled_query_boundaries + expanded row indices
        // match the verbatim query-grouped RNG-replay reference bit-exact.
        // 6 queries of varying sizes summing to 20 rows.
        let qb = vec![0i32, 3, 5, 9, 12, 18, 20];
        let num_data = 20;
        for (seed, frac) in [(3i32, 0.7f64), (3, 0.5), (7, 0.6), (3, 0.4)] {
            let cfg = BaggingConfig::new(frac, 1.0, 1.0, 1, seed, true).unwrap();
            let mut s =
                BaggingSampleStrategy::reset_sample_config_with_queries(cfg, num_data, &qb);
            assert!(s.bagging_by_query(0), "iter 0 must bag (need_re_bagging)");
            let (exp_sampled, exp_sqb, exp_rows) =
                reference_bag_by_query(seed, frac, num_data, &qb);
            assert_eq!(
                s.sampled_query_indices(),
                exp_sampled.as_slice(),
                "seed={seed} frac={frac}: sampled query indices must match RNG-replay"
            );
            assert_eq!(
                s.sampled_query_boundaries(),
                exp_sqb.as_slice(),
                "seed={seed} frac={frac}: sampled_query_boundaries must match"
            );
            assert_eq!(
                s.bag_data_indices(),
                exp_rows.as_slice(),
                "seed={seed} frac={frac}: expanded row indices must match"
            );
            assert_eq!(s.num_sampled_queries(), exp_sampled.len() as i32);
            assert_eq!(s.bag_data_cnt(), exp_rows.len() as i32);
        }
    }

    #[test]
    fn bagging_by_query_expands_whole_queries() {
        // Every kept row must belong to an in-bag query (whole-query granularity:
        // no partial queries).
        let qb = vec![0i32, 4, 7, 10];
        let cfg = BaggingConfig::new(0.6, 1.0, 1.0, 1, 3, true).unwrap();
        let mut s = BaggingSampleStrategy::reset_sample_config_with_queries(cfg, 10, &qb);
        s.bagging_by_query(0);
        let sampled: std::collections::BTreeSet<i32> =
            s.sampled_query_indices().iter().copied().collect();
        // Each kept row's query must be in the sampled set.
        for &row in s.bag_data_indices() {
            let q = qb.windows(2).position(|w| row >= w[0] && row < w[1]).unwrap() as i32;
            assert!(sampled.contains(&q), "row {row} (query {q}) must be in-bag");
        }
        // And all rows of an in-bag query are present.
        for &q in s.sampled_query_indices() {
            for r in qb[q as usize]..qb[q as usize + 1] {
                assert!(s.bag_data_indices().contains(&r), "row {r} of in-bag query {q} missing");
            }
        }
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

    // ===================== GOSS (BST-04) =====================

    /// A verbatim re-implementation of the C++ `GOSSStrategy::Helper`
    /// (`goss.hpp:116-165`) over the proven `lgbm_core::Random` — the GOSS RNG-replay
    /// reference (mirrors `reference_bag`). Single-class (`num_tree_per_iteration=1`).
    /// Returns `(bag_data_indices, bag_data_cnt, amplified_mask, multiply)`.
    ///
    /// `grad`/`hess` are the per-row buffers (NOT mutated — the reference reads the
    /// pre-draw values, exactly as `helper` does internally for the keep decision and
    /// applies `multiply` to the kept-sampled rows).
    fn reference_goss(
        seed: i32,
        top_rate: f64,
        other_rate: f64,
        grad: &[f32],
        hess: &[f32],
    ) -> (Vec<i32>, i32, Vec<bool>, f32) {
        let cnt = grad.len() as i32;
        let n_blocks = (cnt + BAGGING_RAND_BLOCK - 1) / BAGGING_RAND_BLOCK;
        let mut rands: Vec<Random> = (0..n_blocks).map(|i| Random::new(seed + i)).collect();
        let mut tmp: Vec<f32> = (0..cnt as usize)
            .map(|i| (grad[i] * hess[i]).abs())
            .collect();
        let mut top_k = (cnt as f64 * top_rate) as i32;
        let other_k = (cnt as f64 * other_rate) as i32;
        top_k = top_k.max(1);
        let n = tmp.len() as i32;
        arg_max_at_k(&mut tmp, 0, n, top_k - 1);
        let threshold = tmp[(top_k - 1) as usize];
        let multiply = (cnt - top_k) as f32 / other_k as f32;
        let mut buf = vec![0i32; cnt as usize];
        let mut amplified = vec![false; cnt as usize];
        let mut cur_left = 0i32;
        let mut cur_right = cnt;
        let mut big = 0i32;
        for i in 0..cnt {
            let g = (grad[i as usize] * hess[i as usize]).abs();
            if g >= threshold {
                buf[cur_left as usize] = i;
                cur_left += 1;
                big += 1;
            } else {
                let sampled = cur_left - big;
                let rest_need = other_k - sampled;
                let rest_all = (cnt - i) - (top_k - big);
                let prob = rest_need as f64 / rest_all as f64;
                let block = (i / BAGGING_RAND_BLOCK) as usize;
                if (rands[block].next_float() as f64) < prob {
                    buf[cur_left as usize] = i;
                    amplified[i as usize] = true;
                    cur_left += 1;
                } else {
                    cur_right -= 1;
                    buf[cur_right as usize] = i;
                }
            }
        }
        (buf, cur_left, amplified, multiply)
    }

    #[test]
    fn goss_config_check_top_plus_other_le_one() {
        // CHECK_LE(top_rate + other_rate, 1.0) (goss.hpp:85).
        let err =
            GossSampleStrategy::reset_sample_config(0.7, 0.4, 0.1, 10, 1, 3).unwrap_err();
        assert!(matches!(err, BoostingError::GossConfig { .. }));
        // both > 0 (goss.hpp:86).
        let err =
            GossSampleStrategy::reset_sample_config(0.0, 0.1, 0.1, 10, 1, 3).unwrap_err();
        assert!(matches!(err, BoostingError::GossConfig { .. }));
        // valid: 0.2 + 0.1 <= 1.
        assert!(GossSampleStrategy::reset_sample_config(0.2, 0.1, 0.1, 10, 1, 3).is_ok());
    }

    #[test]
    fn goss_skips_subsampling_for_early_iters() {
        // iter < (int)(1/lr): full set, grad/hess UNMODIFIED (goss.hpp:33).
        // lr = 0.1 => skip iters 0..9, subsample from iter 10.
        let n = 20;
        let mut grad: Vec<f32> = (0..n).map(|i| (i as f32) - 10.0).collect();
        let mut hess = vec![1.0f32; n as usize];
        let grad_before = grad.clone();
        let hess_before = hess.clone();
        let mut s = GossSampleStrategy::reset_sample_config(0.2, 0.1, 0.1, n, 1, 3).unwrap();
        // iter 5 (< 10): skipped.
        let did = s.bagging(5, &mut grad, &mut hess);
        assert!(!did, "iter 5 < 1/lr=10 must skip subsampling");
        assert!(!s.is_use_subset(), "skip window => not a subset");
        assert_eq!(s.bag_data_cnt(), n);
        assert_eq!(grad, grad_before, "skip window must NOT modify gradients");
        assert_eq!(hess, hess_before, "skip window must NOT modify hessians");
        // iter 10 (>= 10): subsamples.
        let did = s.bagging(10, &mut grad, &mut hess);
        assert!(did, "iter 10 >= 1/lr=10 must subsample");
        assert!(s.bag_data_cnt() < n, "GOSS keeps fewer rows: top_k + sampled");
    }

    #[test]
    fn goss_amplifies_grad_and_hess_by_multiply() {
        // Sampled (non-top, randomly-kept) rows have BOTH grad and hess multiplied by
        // multiply = (cnt - top_k) / other_k.
        let n = 20;
        // Distinct |g*h| so the top-k threshold is well defined; hess=1 so |g*h|=|g|.
        let mut grad: Vec<f32> = (0..n).map(|i| (i as f32 + 1.0) * 0.5).collect();
        let mut hess = vec![1.0f32; n as usize];
        let grad_before = grad.clone();
        let (exp_idx, exp_cnt, exp_amp, multiply) =
            reference_goss(3, 0.2, 0.1, &grad_before, &hess);
        let mut s = GossSampleStrategy::reset_sample_config(0.2, 0.1, 0.1, n, 1, 3).unwrap();
        s.bagging(10, &mut grad, &mut hess);
        assert_eq!(s.bag_data_cnt(), exp_cnt, "kept count must match reference");
        assert_eq!(
            s.bag_data_indices(),
            exp_idx.as_slice(),
            "bag indices must match GOSS RNG-replay reference"
        );
        // Every amplified row: grad *= multiply, hess (was 1) == multiply.
        for i in 0..n as usize {
            if exp_amp[i] {
                let expect_g = grad_before[i] * multiply;
                assert!(
                    (grad[i] - expect_g).abs() < 1e-6,
                    "row {i} grad must be amplified by multiply={multiply}"
                );
                assert!(
                    (hess[i] - multiply).abs() < 1e-6,
                    "row {i} hess must be amplified by multiply={multiply}"
                );
            } else {
                // Top rows and dropped rows keep their original grad/hess.
                assert!(
                    (grad[i] - grad_before[i]).abs() < 1e-9,
                    "non-amplified row {i} grad must be unchanged"
                );
                assert!((hess[i] - 1.0).abs() < 1e-9, "non-amplified row {i} hess unchanged");
            }
        }
    }

    #[test]
    fn goss_uses_arg_max_at_k_not_full_sort() {
        // ArgMaxAtK(k=0) returns the maximum element at index 0 (a full descending
        // sort would too, but a TIE block is only partially ordered — assert the
        // selected k-th element equals the descending-sort k-th, NOT a stable order).
        let mut a = vec![3.0f32, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0];
        let n = a.len() as i32;
        // k = 2 (the 3rd largest). Descending sorted: 9,6,5,4,3,2,1,1 => index 2 == 5.
        arg_max_at_k(&mut a, 0, n, 2);
        assert_eq!(a[2], 5.0, "ArgMaxAtK(k=2) selects the 3rd-largest value");
        // The partition guarantees a[0..2] are all >= a[2] and a[3..] are all <= a[2].
        for i in 0..2 {
            assert!(a[i] >= a[2], "left of k must be >= a[k]");
        }
        for i in 3..a.len() {
            assert!(a[i] <= a[2], "right of k must be <= a[k]");
        }
    }

    #[test]
    fn goss_bag_indices_match_rng_replay_golden() {
        // The full bag_data_indices (kept ++ dropped) + amplified mask match the
        // verbatim GOSS RNG-replay reference bit-exact (compare_exact). Several
        // seed/rate combos to exercise the draw loop.
        let n = 50;
        let grad: Vec<f32> = (0..n).map(|i| ((i * 7 % 13) as f32) - 6.0).collect();
        let hess = vec![1.0f32; n as usize];
        for (seed, top, other) in [(3i32, 0.2f64, 0.1f64), (3, 0.1, 0.05), (7, 0.2, 0.1)] {
            let mut g = grad.clone();
            let mut h = hess.clone();
            let (exp_idx, exp_cnt, _amp, _m) = reference_goss(seed, top, other, &grad, &hess);
            let mut s =
                GossSampleStrategy::reset_sample_config(top, other, 0.1, n, 1, seed).unwrap();
            s.bagging(10, &mut g, &mut h);
            assert_eq!(
                s.bag_data_cnt(),
                exp_cnt,
                "seed={seed} top={top} other={other}: kept count != reference"
            );
            assert_eq!(
                s.bag_data_indices(),
                exp_idx.as_slice(),
                "seed={seed} top={top} other={other}: GOSS indices not bit-exact vs RNG-replay"
            );
        }
    }
}
