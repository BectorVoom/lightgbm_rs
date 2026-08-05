//! `HistogramPool` — fixed-size leaf→histogram buffer pool with eviction (D-05).
//!
//! Faithful mirror of the C++ `HistogramPool` (`feature_histogram.hpp:1367-1486`,
//! commit 195c26fc, VERSION 4.6.0.99) — the sizing / eviction / `Get` / `Move` /
//! `ResetMap` machinery that decides WHICH child leaf reuses the parent's
//! histogram buffer. Per D-05 this is mirrored in FULL (not just the
//! FP-load-bearing parts) because the buffer-reuse decision is observable in the
//! subtraction-trick path (the parent histogram is the larger child's buffer).
//!
//! ## What the pool tracks (parallel-array mirror discipline — `tree.rs` style)
//! The C++ pool owns a fixed number of physical histogram SLOTS (`cache_size`,
//! capped at `num_leaves`) and two index maps:
//! - `mapper_[leaf]`   → the slot id currently holding `leaf`'s histogram, or `-1`.
//! - `inverse_mapper_[slot]` → the leaf currently owning `slot`, or `-1`.
//! plus an LRU `last_used_time_[slot]` for eviction. `Get(leaf, &out)` returns the
//! leaf's slot (allocating/evicting on a miss and signalling "fresh buffer", or
//! `true` = "this leaf already had a buffer" — the C++ return that drives
//! `parent_leaf_histogram_array_` retention at `serial_tree_learner.cpp:370-377`).
//! `Move(src, dst)` reassigns a slot from one leaf id to another WITHOUT copying
//! the buffer (the C++ `Move` just swaps the mapper entries). `ResetMap()` clears
//! both maps (called per tree at `BeforeTrain`).
//!
//! ## Buffers
//! Each slot holds a stride-2 `[g0,h0,g1,h1,…]` f64 histogram for ONE feature's
//! bins (the spine processes a single feature column at a time via the Backend
//! ops). The pool stores the host-side `Vec<f64>` so the learner can `FixHistogram`
//! / subtract on it. Buffer CONTENTS are written by the learner (via
//! `construct_histograms` / `subtract_histograms`); the pool owns only the
//! leaf→slot ASSIGNMENT and eviction policy.

/// The leaf→slot histogram pool (`HistogramPool`, `feature_histogram.hpp:1367`).
#[derive(Debug, Clone)]
pub struct HistogramPool {
    /// C++ `int cache_size_` — number of physical slots (== `num_leaves` here).
    cache_size: usize,
    /// All slot buffers live in ONE flat arena: slot `s` occupies
    /// `[s*hist_len, (s+1)*hist_len)`, each a stride-2 `[g,h]` f64 histogram for
    /// one feature column's bins. A single flat `vec![0.0; cache_size*hist_len]`
    /// allocation avoids the per-tree cost of many small allocations plus a
    /// zeroed-template clone-copy that a jagged `Vec<Vec<f64>>` would require,
    /// and it keeps the subtraction trick's parent/child slots contiguous. The
    /// public slot API (`buffer`/`buffer_mut`) is unaffected.
    buffers: Vec<f64>,
    /// C++ `std::vector<int> mapper_` — `mapper_[leaf]` = slot id, or `-1`.
    mapper: Vec<i32>,
    /// C++ `std::vector<int> inverse_mapper_` — `inverse_mapper_[slot]` = leaf, or `-1`.
    inverse_mapper: Vec<i32>,
    /// C++ `std::vector<int> last_used_time_` — LRU clock per slot.
    last_used_time: Vec<i32>,
    /// C++ `int cur_time_` — monotonically increasing LRU clock.
    cur_time: i32,
    /// The fixed per-slot histogram length (`2 * num_bin` for the spine's single
    /// feature column).
    hist_len: usize,
}

impl HistogramPool {
    /// `HistogramPool` construction + `DynamicChangeSize` sizing
    /// (`feature_histogram.hpp:1380-1416`): allocate `cache_size` slots, each a
    /// zeroed `hist_len`-cell buffer, with empty mapper / inverse-mapper.
    ///
    /// `num_leaves` caps the slot count (`cache_size = num_leaves`); `hist_len` is
    /// the per-slot histogram length (`2 * num_bin`).
    pub fn new(num_leaves: i32, hist_len: usize) -> Self {
        Self::with_cache_size(num_leaves.max(1) as usize, hist_len)
    }

    /// `SerialTreeLearner::Init`'s pool sizing (`serial_tree_learner.cpp:34-47`):
    /// derive the slot count from `config.histogram_pool_size` (the pool budget in
    /// MEGABYTES) instead of defaulting to one slot per leaf.
    ///
    /// ```text
    /// pool_size_mb <= 0  → num_leaves                       (unbounded: the default)
    /// otherwise          → pool_size_mb * 1024 * 1024 / total_histogram_size
    /// then clamped to    → max(2, ·) then min(·, num_leaves)
    /// ```
    ///
    /// `total_histogram_size` is `kHistEntrySize * Σ FeatureNumBin(i)` where
    /// `kHistEntrySize = 2 * sizeof(hist_t) = 16` bytes (bin.h:39).
    ///
    /// A smaller pool is NOT a numerical change — it only forces more LRU evictions,
    /// which cost recomputation, never a different histogram. Guarded against a
    /// zero `total_histogram_size` (a feature-less dataset) so the division and the
    /// `as` cast can never trap or wrap.
    pub fn with_pool_size(
        num_leaves: i32,
        hist_len: usize,
        histogram_pool_size_mb: f64,
        total_num_bins: usize,
    ) -> Self {
        /// `kHistEntrySize = 2 * sizeof(hist_t)` (bin.h:39); `hist_t` is `double`.
        const K_HIST_ENTRY_SIZE: usize = 2 * std::mem::size_of::<f64>();

        let num_leaves = num_leaves.max(1);
        let total_histogram_size = K_HIST_ENTRY_SIZE.saturating_mul(total_num_bins);
        let mut max_cache_size = if histogram_pool_size_mb <= 0.0 || total_histogram_size == 0 {
            num_leaves
        } else {
            let budget = histogram_pool_size_mb * 1024.0 * 1024.0 / total_histogram_size as f64;
            // C++ `static_cast<int>` truncates toward zero; clamp first so a huge
            // budget cannot overflow i32 (UB in C++, a saturating cast here).
            budget.clamp(0.0, i32::MAX as f64) as i32
        };
        max_cache_size = max_cache_size.max(2);
        max_cache_size = max_cache_size.min(num_leaves);
        Self::with_cache_size(max_cache_size.max(1) as usize, hist_len)
    }

    fn with_cache_size(cache_size: usize, hist_len: usize) -> Self {
        Self {
            cache_size,
            buffers: vec![0.0f64; cache_size * hist_len],
            mapper: vec![-1; cache_size],
            inverse_mapper: vec![-1; cache_size],
            last_used_time: vec![0; cache_size],
            cur_time: 0,
            hist_len,
        }
    }

    /// C++ `HistogramPool::ResetMap` (`feature_histogram.hpp:1418-1424`): clear
    /// both index maps + the LRU clock (called once per tree in `BeforeTrain`).
    /// Buffer CONTENTS are left as-is (the C++ leaves the memory; it is overwritten
    /// on the next `construct`).
    pub fn reset_map(&mut self) {
        for m in &mut self.mapper {
            *m = -1;
        }
        for im in &mut self.inverse_mapper {
            *im = -1;
        }
        for t in &mut self.last_used_time {
            *t = 0;
        }
        self.cur_time = 0;
    }

    /// C++ `HistogramPool::Get(idx, &out)` (`feature_histogram.hpp:1426-1456`):
    /// return the slot id holding leaf `idx`'s histogram.
    ///
    /// Returns `(slot, existed)`:
    /// - `existed == true`  → leaf `idx` ALREADY had a slot (a cache HIT). This is
    ///   the C++ `return true` that makes the caller retain the parent histogram
    ///   (`parent_leaf_histogram_array_ = larger_leaf_histogram_array_`,
    ///   `serial_tree_learner.cpp:370-377`).
    /// - `existed == false` → a fresh slot was allocated (evicting the LRU slot on
    ///   a full pool). The buffer is NOT zeroed (the learner overwrites it).
    ///
    /// Eviction picks the slot with the smallest `last_used_time_` whose current
    /// owner is not `idx` (LRU), matching the C++ `DynamicChangeSize`/`Get`
    /// fallback. `last_used_time_[slot]` is bumped to `cur_time_++` on every
    /// access so repeated `Get`s keep the hot slot resident.
    pub fn get(&mut self, idx: i32) -> (usize, bool) {
        let leaf = idx as usize;
        // Defensive: the mapper is sized to cache_size == num_leaves, so any
        // valid leaf id is in range. Grow if a caller exceeds it (should not
        // happen on the spine; keeps the pool panic-free, V5 spirit).
        if leaf >= self.mapper.len() {
            self.mapper.resize(leaf + 1, -1);
        }

        self.cur_time += 1;
        let existing = self.mapper[leaf];
        if existing >= 0 {
            // Cache hit: leaf already owns a slot.
            let slot = existing as usize;
            self.last_used_time[slot] = self.cur_time;
            return (slot, true);
        }

        // Miss: find a free slot, else evict the LRU slot (smallest last_used).
        let slot = self.allocate_slot();
        // Detach the slot's previous owner, if any.
        let prev_owner = self.inverse_mapper[slot];
        if prev_owner >= 0 {
            self.mapper[prev_owner as usize] = -1;
        }
        self.mapper[leaf] = slot as i32;
        self.inverse_mapper[slot] = leaf as i32;
        self.last_used_time[slot] = self.cur_time;
        (slot, false)
    }

    /// Pick a free slot (inverse_mapper == -1) or evict the least-recently-used.
    fn allocate_slot(&mut self) -> usize {
        // Prefer a genuinely free slot (preserves the C++ allocate-then-evict
        // order: fresh slots are used before any eviction happens).
        for (slot, &owner) in self.inverse_mapper.iter().enumerate() {
            if owner < 0 {
                return slot;
            }
        }
        // Full: evict the slot with the smallest last_used_time_ (LRU).
        let mut lru_slot = 0usize;
        let mut lru_time = self.last_used_time[0];
        for slot in 1..self.cache_size {
            if self.last_used_time[slot] < lru_time {
                lru_time = self.last_used_time[slot];
                lru_slot = slot;
            }
        }
        lru_slot
    }

    /// C++ `HistogramPool::Move(src_idx, dst_idx)`
    /// (`feature_histogram.hpp:1458-1470`): reassign `src`'s slot to `dst` WITHOUT
    /// copying the buffer (just rewires the mapper entries). `src` is left with no
    /// slot. This is the `histogram_pool_.Move(left_leaf, right_leaf)` at
    /// `serial_tree_learner.cpp:372` that hands the parent's buffer to the larger
    /// child before the smaller child gets a fresh one.
    pub fn move_(&mut self, src_idx: i32, dst_idx: i32) {
        let src = src_idx as usize;
        let dst = dst_idx as usize;
        if dst >= self.mapper.len() {
            self.mapper.resize(dst + 1, -1);
        }
        let slot = self.mapper[src];
        // Detach src.
        self.mapper[src] = -1;
        if slot >= 0 {
            self.mapper[dst] = slot;
            self.inverse_mapper[slot as usize] = dst as i32;
        } else {
            self.mapper[dst] = -1;
        }
    }

    /// Immutable view of a slot's histogram buffer (the slot's `hist_len`-cell
    /// region of the flat arena).
    pub fn buffer(&self, slot: usize) -> &[f64] {
        let base = slot * self.hist_len;
        &self.buffers[base..base + self.hist_len]
    }

    /// Mutable view of a slot's histogram buffer (the learner writes
    /// `construct_histograms` / `subtract_histograms` output here, then
    /// `FixHistogram`s it in place).
    pub fn buffer_mut(&mut self, slot: usize) -> &mut [f64] {
        let base = slot * self.hist_len;
        &mut self.buffers[base..base + self.hist_len]
    }

    /// The fixed per-slot histogram length (`2 * num_bin`).
    pub fn hist_len(&self) -> usize {
        self.hist_len
    }

    /// Number of physical slots (`cache_size_`).
    pub fn cache_size(&self) -> usize {
        self.cache_size
    }
}

#[cfg(test)]
mod spike010 {
    //! What is the per-tree CEILING for replacing the pool's
    //! `Vec<Vec<f64>>` buffers? `HistogramPool::new` runs once per tree
    //! (`learner.rs:816`): `num_leaves` slot allocations, each `slot_len` f64,
    //! zeroed then overwritten. This isolates that cost three ways at the bench
    //! shapes, in-process, so we measure the lever's ceiling (not a noisy
    //! end-to-end train). Run:
    //!   cargo test -p lgbm-treelearner --release --lib spike010_pool_alloc_ceiling -- --ignored --nocapture
    use super::*;
    use std::time::Instant;

    #[test]
    #[ignore]
    fn spike010_pool_alloc_ceiling() {
        // (name, num_leaves, slot_len = Σ 2*num_bin, per-tree wall from bench_train)
        // small : 12 feat × 32 bins → 12*64=768 ; ~266 µs/tree
        // medium: 30 feat × 64 bins → 30*128=3840; ~1.32 ms/tree
        // large : 50 feat ×128 bins → 50*256=12800; ~4.40 ms/tree
        let shapes = [
            ("small ", 31i32, 768usize, 266_000.0f64),
            ("medium", 31, 3_840, 1_320_000.0),
            ("large ", 31, 12_800, 4_400_000.0),
        ];
        let trees = 200usize;
        let warm = 20usize;
        let med = |v: &mut Vec<std::time::Duration>| {
            v.sort();
            v[v.len() / 2]
        };
        eprintln!("spike010 pool-alloc ceiling ({trees}'trees', median; sink defeats DCE):");
        for (name, nl, sl, per_tree_ns) in shapes {
            let cells = nl as usize * sl;
            let mut sink = 0.0f64;

            // (1) CURRENT: Vec<Vec<f64>> — nl separate allocs + zero, every tree.
            let mut t_cur = Vec::with_capacity(trees);
            for i in 0..(trees + warm) {
                let s = Instant::now();
                let pool = HistogramPool::new(nl, sl);
                let d = s.elapsed();
                sink += pool.buffer(0)[0] + pool.buffer((nl - 1) as usize)[sl - 1];
                if i >= warm {
                    t_cur.push(d);
                }
            }

            // (2) FLAT ARENA: one `vec![0.0; nl*sl]` alloc + zero, every tree.
            let mut t_flat = Vec::with_capacity(trees);
            for i in 0..(trees + warm) {
                let s = Instant::now();
                let buf = vec![0.0f64; cells];
                let d = s.elapsed();
                sink += buf[0] + buf[cells - 1];
                if i >= warm {
                    t_flat.push(d);
                }
            }

            // (3) REUSED ARENA: allocate ONCE, per tree only the index-map reset
            // (buffers are overwritten by construct before read, so no zeroing).
            // This is the hoist-across-trees ideal: per-tree buffer cost → 0.
            let mut arena = vec![0.0f64; cells];
            let mut mapper = vec![-1i32; nl as usize];
            let mut t_reuse = Vec::with_capacity(trees);
            for i in 0..(trees + warm) {
                let s = Instant::now();
                for m in mapper.iter_mut() {
                    *m = -1;
                }
                let d = s.elapsed();
                // touch the arena like a real per-tree use would (1 cell) to keep it live
                arena[i % cells] = i as f64;
                sink += arena[0];
                if i >= warm {
                    t_reuse.push(d);
                }
            }

            std::hint::black_box(sink);
            let cur = med(&mut t_cur);
            let flat = med(&mut t_flat);
            let reuse = med(&mut t_reuse);
            let pct = |d: std::time::Duration| 100.0 * d.as_secs_f64() * 1e9 / per_tree_ns;
            eprintln!(
                "  {name}: cur(Vec<Vec>)={:>9.3?} ({:>4.1}% of tree)  flat={:>9.3?} ({:>4.1}%)  reuse={:>9.3?} ({:>4.1}%)  | ceiling(cur−reuse)={:>4.1}% of per-tree",
                cur, pct(cur), flat, pct(flat), reuse, pct(reuse),
                pct(cur) - pct(reuse),
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn get_miss_then_hit() {
        let mut pool = HistogramPool::new(4, 8);
        let (slot0, existed0) = pool.get(0);
        assert!(!existed0, "first Get(0) is a miss (fresh buffer)");
        let (slot0b, existed0b) = pool.get(0);
        assert!(existed0b, "second Get(0) is a hit (parent retained)");
        assert_eq!(slot0, slot0b, "same leaf maps to the same slot");
    }

    #[test]
    fn move_reassigns_slot_without_copy() {
        let mut pool = HistogramPool::new(4, 4);
        let (slot_left, _) = pool.get(0); // left_leaf gets a slot
        // Write a sentinel into the buffer so we can prove Move does not copy.
        pool.buffer_mut(slot_left).copy_from_slice(&[1.0, 2.0, 3.0, 4.0]);
        // Move left_leaf -> right_leaf: right now owns the SAME slot/buffer.
        pool.move_(0, 1);
        let (slot_right, existed_right) = pool.get(1);
        assert!(existed_right, "right_leaf now owns the moved slot (a hit)");
        assert_eq!(slot_right, slot_left, "Move reassigns the same physical slot");
        assert_eq!(
            pool.buffer(slot_right),
            &[1.0, 2.0, 3.0, 4.0],
            "Move does not copy/clear the buffer"
        );
        // left_leaf no longer owns a slot -> next Get(0) is a miss.
        let (_, existed_left) = pool.get(0);
        assert!(!existed_left, "src leaf lost its slot after Move");
    }

    #[test]
    fn reset_map_clears_assignments() {
        let mut pool = HistogramPool::new(4, 4);
        pool.get(0);
        pool.get(1);
        pool.reset_map();
        // After reset every leaf is a miss again.
        assert!(!pool.get(0).1);
        assert!(!pool.get(2).1);
    }

    /// On a FULL pool a fresh leaf evicts the least-recently-used slot.
    #[test]
    fn full_pool_evicts_lru() {
        let mut pool = HistogramPool::new(2, 2); // cache_size = 2
        let (s0, _) = pool.get(0); // slot for leaf 0 (time 1)
        let (s1, _) = pool.get(1); // slot for leaf 1 (time 2)
        assert_ne!(s0, s1);
        // Touch leaf 1 so leaf 0's slot becomes the LRU.
        pool.get(1); // time 3 -> slot s1 hottest
        // Leaf 2 (miss) must evict leaf 0's slot (the LRU).
        let (s2, existed2) = pool.get(2);
        assert!(!existed2);
        assert_eq!(s2, s0, "evicted the LRU slot (leaf 0's)");
        // Leaf 0 was evicted -> it is now a miss.
        assert!(!pool.get(0).1);
    }
}
