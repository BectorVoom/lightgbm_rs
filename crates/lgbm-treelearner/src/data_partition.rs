//! `DataPartition` — leaf row-range bookkeeping wrapping `Backend::data_partition`.
//!
//! Faithful transcription of `DataPartition` (`data_partition.hpp`, commit
//! 195c26fc, VERSION 4.6.0.99): the `indices_` permutation of all row ids grouped
//! by leaf, plus the per-leaf `leaf_begin_` / `leaf_count_` ranges. The actual
//! per-row left/right routing math lives in the Phase-4 `Backend::data_partition`
//! op (the `DenseBin::SplitInner` `MissingType::None` transcription); this module
//! owns ONLY the leaf-range bookkeeping (`lib.rs:115-117`: "the Phase-5 learner
//! owns `leaf_begin_`/`leaf_count_` bookkeeping; this op returns only the
//! partition"). No new partition numerics.
//!
//! ## C++ correspondence
//! - `indices_`     ↔ `std::vector<data_size_t> indices_` — all row ids, grouped
//!   contiguously by leaf.
//! - `leaf_begin_`  ↔ `std::vector<data_size_t> leaf_begin_` — start offset of
//!   each leaf's slice in `indices_`.
//! - `leaf_count_`  ↔ `std::vector<data_size_t> leaf_count_` — row count per leaf
//!   (the GLOBAL count; no bagging this phase, so global == local, Pitfall 3).
//!
//! `Init` puts ALL rows into leaf 0 in identity order (`data_partition.hpp` Init).
//! `Split` calls the Backend op on the leaf's rows, then rewrites the leaf's slice
//! of `indices_` into [left rows | right rows] and updates the two child ranges.

use lgbm_compute::error::ComputeError;
use lgbm_compute::Backend;
use lgbm_compute::BinColumn;
// `ComputeClient` is re-exported by the compute seam (CMP-01) so this crate names
// the Backend ops' client argument without ever depending on `cubecl` directly.
use lgbm_compute::ComputeClientReexport as ComputeClient;

/// The leaf-partition bookkeeping (`DataPartition`, `data_partition.hpp`).
#[derive(Debug, Clone)]
pub struct DataPartition {
    /// C++ `data_size_t num_data_` — total rows.
    num_data: i32,
    /// C++ `std::vector<data_size_t> indices_` — row ids grouped by leaf.
    indices: Vec<u32>,
    /// C++ `std::vector<data_size_t> leaf_begin_` — per-leaf start offset.
    leaf_begin: Vec<i32>,
    /// C++ `std::vector<data_size_t> leaf_count_` — per-leaf row count.
    leaf_count: Vec<i32>,
}

impl DataPartition {
    /// `DataPartition(num_data, num_leaves)` + `Init()` (`data_partition.hpp`):
    /// all rows go into leaf 0 in identity order; every other leaf is empty.
    ///
    /// `num_leaves` is the maximum leaf capacity (the config `num_leaves`); the
    /// per-leaf range arrays are sized to it so later `Split`s never reallocate.
    pub fn new(num_data: i32, num_leaves: i32) -> Self {
        let n = num_data.max(0) as usize;
        let l = num_leaves.max(1) as usize;
        let indices: Vec<u32> = (0..n as u32).collect();
        let mut leaf_begin = vec![0i32; l];
        let mut leaf_count = vec![0i32; l];
        // Init: leaf 0 owns all rows starting at offset 0.
        leaf_begin[0] = 0;
        leaf_count[0] = num_data.max(0);
        Self {
            num_data,
            indices,
            leaf_begin,
            leaf_count,
        }
    }

    /// C++ `DataPartition::leaf_count(leaf)` — the GLOBAL row count in `leaf`.
    /// Drives the smaller-child selection (Pitfall 3).
    pub fn leaf_count(&self, leaf: i32) -> i32 {
        self.leaf_count[leaf as usize]
    }

    /// C++ `DataPartition::leaf_begin(leaf)` — `leaf`'s start offset in `indices_`.
    pub fn leaf_begin(&self, leaf: i32) -> i32 {
        self.leaf_begin[leaf as usize]
    }

    /// The full row-permutation array (`indices_`).
    pub fn indices(&self) -> &[u32] {
        &self.indices
    }

    /// The global row ids belonging to `leaf` (its slice of `indices_`).
    pub fn indices_in_leaf(&self, leaf: i32) -> &[u32] {
        let begin = self.leaf_begin[leaf as usize] as usize;
        let count = self.leaf_count[leaf as usize] as usize;
        &self.indices[begin..begin + count]
    }

    /// `DataPartition::Split` (`data_partition.hpp`): partition `leaf` by the
    /// feature threshold into `left_leaf` (kept as `leaf`) and `right_leaf`.
    ///
    /// `feature_bins` is the per-GLOBAL-ROW bin index for the split feature (the
    /// whole-column `Bin::data(row)` widened to `u32`, length `num_data`); the op
    /// reads only the rows in `leaf`. The Backend op returns a STABLE reorder of
    /// the leaf's rows (`[left | right]`, left first in original relative order)
    /// plus `split_point` (= left-row count). This method maps those positions
    /// back to GLOBAL row ids, rewrites the leaf's slice of `indices_`, and sets
    /// the two child ranges: `left_leaf` keeps `[leaf_begin, leaf_begin+left)`,
    /// `right_leaf` takes `[leaf_begin+left, leaf_begin+count)`.
    ///
    /// Returns `(left_count, right_count)`.
    ///
    /// # Errors
    /// Propagates [`ComputeError`] from the Backend op (V5 boundary; e.g. a bin
    /// index `>= num_bin`).
    ///
    /// NOTE (quick-260622-ia0, NULL — do not re-attempt naively): this reorder is
    /// single-threaded while C++ OMP-parallelizes its `Split` (data_partition.hpp).
    /// At tall-narrow shapes (1M×50) partition is ~29% of train, so it looks like an
    /// obvious rayon target. A bit-exact rayon version (static 512-row chunks →
    /// exclusive prefix-sum → disjoint scatter, byte-identical, proven) was built and
    /// A/B-benched: it is an **end-to-end NULL** — train-wall stayed within noise and
    /// the partition phase went BIMODAL (−15–24% pool-idle, +27–40% under contention)
    /// because the histogram BUILD (69% of train) is ALREADY rayon-parallel and
    /// saturates the same pool. Same lesson as the split-scan campaign: don't
    /// parallelize a phase harder against an already-parallel neighbour — fuse, don't
    /// contend. Reverted. Evidence: the quick-260622-ia0 SUMMARY.
    #[allow(clippy::too_many_arguments)]
    pub fn split<B: Backend>(
        &mut self,
        backend: &B,
        client: &ComputeClient<B::Runtime>,
        leaf: i32,
        right_leaf: i32,
        feature_bins: &BinColumn,
        num_bin: u32,
        min_bin: u32,
        max_bin: u32,
        threshold: u32,
        most_freq_bin: u32,
    ) -> Result<(i32, i32), ComputeError> {
        let leaf_u = leaf as usize;
        let begin = self.leaf_begin[leaf_u] as usize;
        let count = self.leaf_count[leaf_u] as usize;

        // Gather the leaf's per-row bins in the current `indices_` order so the
        // Backend op's stable reorder is RELATIVE to this leaf's row order. The
        // per-row bin READ widens via the accessor; the partition LOGIC is
        // unchanged.
        let leaf_rows: Vec<u32> = self.indices[begin..begin + count].to_vec();
        let leaf_feature_bins: Vec<u32> = leaf_rows
            .iter()
            .map(|&row| feature_bins.bin(row as usize))
            .collect();

        // The Backend op owns the SplitInner routing + stable two-pass gather.
        let (reordered_local, split_point) = backend.data_partition(
            client,
            &leaf_feature_bins,
            num_bin,
            min_bin,
            max_bin,
            threshold,
            most_freq_bin,
        )?;

        // Map the op's local positions back to GLOBAL row ids and write them back
        // into this leaf's slice of `indices_` (left rows first, then right).
        for (slot, &local_pos) in reordered_local.iter().enumerate() {
            self.indices[begin + slot] = leaf_rows[local_pos as usize];
        }

        let left_count = split_point as i32;
        let right_count = count as i32 - left_count;

        // left_leaf keeps `leaf`'s begin; right_leaf starts after the left rows.
        self.leaf_count[leaf_u] = left_count;
        let right_u = right_leaf as usize;
        self.leaf_begin[right_u] = begin as i32 + left_count;
        self.leaf_count[right_u] = right_count;

        Ok((left_count, right_count))
    }

    /// `DataPartition::Split` for a CATEGORICAL split — routes `leaf`'s rows by the
    /// REAL category bitset into `left_leaf` (kept as `leaf`, the in-bitset side)
    /// and `right_leaf` (the default/out-of-bitset side).
    ///
    /// This is the row-partition analog of the predict-side `Tree::CategoricalDecision`
    /// (`tree.h:374-390`): a row whose feature BIN maps to a CATEGORY VALUE that is
    /// in the real bitset routes LEFT; everything else (incl. the NaN dummy bin 0,
    /// whose category is `-1` / negative) routes RIGHT. Routing on the category
    /// value through `bin_to_category` is provably equivalent to the C++
    /// `DenseBin::SplitCategoricalInner` inner-bin-bitset routing (which is just a
    /// performance encoding of the same decision) and is GUARANTEED consistent with
    /// the model's serialized real bitset + the predict path — sidestepping the
    /// bin↔offset bookkeeping entirely.
    ///
    /// `cat_bitset_real` is `Common::ConstructBitset` over the winning CATEGORY
    /// VALUES (the same bitset the tree serializes as `cat_threshold`).
    /// `bin_to_category` maps a feature bin to its category value (`bin_2_categorical_`).
    ///
    /// Returns `(left_count, right_count)`.
    pub fn split_categorical(
        &mut self,
        leaf: i32,
        right_leaf: i32,
        feature_bins: &BinColumn,
        cat_bitset_real: &[u32],
        bin_to_category: &[i32],
    ) -> (i32, i32) {
        let leaf_u = leaf as usize;
        let begin = self.leaf_begin[leaf_u] as usize;
        let count = self.leaf_count[leaf_u] as usize;
        let leaf_rows: Vec<u32> = self.indices[begin..begin + count].to_vec();

        let mut lte: Vec<u32> = Vec::with_capacity(count); // left / in-bitset
        let mut gt: Vec<u32> = Vec::with_capacity(count); // right / default
        for &row in &leaf_rows {
            let bin = feature_bins.bin(row as usize);
            let cat = bin_to_category
                .get(bin as usize)
                .copied()
                .unwrap_or(bin as i32);
            // Negative category (the NaN dummy at bin 0) always routes RIGHT, exactly
            // as CategoricalDecision routes `int_fval < 0` / NaN to the right child.
            if cat >= 0 && find_in_bitset(cat_bitset_real, cat as u32) {
                lte.push(row);
            } else {
                gt.push(row);
            }
        }

        let left_count = lte.len() as i32;
        let right_count = gt.len() as i32;

        // Write [left | right] back into the leaf's slice (stable order).
        let mut w = begin;
        for &row in &lte {
            self.indices[w] = row;
            w += 1;
        }
        for &row in &gt {
            self.indices[w] = row;
            w += 1;
        }

        self.leaf_count[leaf_u] = left_count;
        let right_u = right_leaf as usize;
        self.leaf_begin[right_u] = begin as i32 + left_count;
        self.leaf_count[right_u] = right_count;

        (left_count, right_count)
    }

    /// Total row count (`num_data_`).
    pub fn num_data(&self) -> i32 {
        self.num_data
    }
}

/// `Common::FindInBitset(bits, n, pos)` (`common.h:836`): `i1 = pos/32; if i1 >= n
/// return false; (bits[i1] >> (pos%32)) & 1`.
#[inline]
fn find_in_bitset(bits: &[u32], pos: u32) -> bool {
    let i1 = (pos / 32) as usize;
    if i1 >= bits.len() {
        return false;
    }
    (bits[i1] >> (pos % 32)) & 1 == 1
}

#[cfg(test)]
mod tests {
    use super::*;
    use lgbm_compute::{runtime::cpu_client, CpuBackend};

    #[test]
    fn init_puts_all_rows_in_leaf_zero() {
        let dp = DataPartition::new(5, 3);
        assert_eq!(dp.leaf_count(0), 5);
        assert_eq!(dp.leaf_begin(0), 0);
        assert_eq!(dp.indices_in_leaf(0), &[0, 1, 2, 3, 4]);
    }

    /// `Split` must conserve rows (`left + right == parent`) and write the
    /// Backend-returned stable reorder into the leaf slice.
    #[test]
    fn split_conserves_count_and_uses_backend_reorder() {
        let backend = CpuBackend;
        let client = cpu_client();
        let mut dp = DataPartition::new(8, 4);
        // bins per row; threshold=3 -> bin<=3 left, bin>3 right (num_bin=8).
        let feature_bins = BinColumn::new(vec![1u32, 5, 3, 7, 0, 4, 2, 6], 8);
        let (left, right) = dp
            .split(&backend, &client, 0, 1, &feature_bins, 8, 0, 7, 3, 8)
            .expect("split ok");
        // left rows (bin<=3) in original order: 0(b1),2(b3),4(b0),6(b2)
        // right rows (bin>3): 1(b5),3(b7),5(b4),7(b6)
        assert_eq!(left, 4);
        assert_eq!(right, 4);
        assert_eq!(left + right, 8, "rows conserved");
        assert_eq!(dp.leaf_count(0), 4);
        assert_eq!(dp.leaf_count(1), 4);
        assert_eq!(dp.indices_in_leaf(0), &[0, 2, 4, 6]);
        assert_eq!(dp.indices_in_leaf(1), &[1, 3, 5, 7]);
    }

    /// A second split on a non-root leaf must operate on that leaf's slice only,
    /// using the global row ids it currently holds.
    #[test]
    fn split_on_child_leaf_uses_its_own_rows() {
        let backend = CpuBackend;
        let client = cpu_client();
        let mut dp = DataPartition::new(8, 4);
        let feature_bins = BinColumn::new(vec![1u32, 5, 3, 7, 0, 4, 2, 6], 8);
        dp.split(&backend, &client, 0, 1, &feature_bins, 8, 0, 7, 3, 8)
            .unwrap();
        // Now split leaf 1 (rows 1,3,5,7 with bins 5,7,4,6) at threshold 5:
        // bin<=5 left (rows 1(b5),5(b4)), bin>5 right (rows 3(b7),7(b6)).
        let (l, r) = dp
            .split(&backend, &client, 1, 2, &feature_bins, 8, 0, 7, 5, 8)
            .unwrap();
        assert_eq!((l, r), (2, 2));
        assert_eq!(dp.indices_in_leaf(1), &[1, 5]);
        assert_eq!(dp.indices_in_leaf(2), &[3, 7]);
        // Leaf 0 untouched.
        assert_eq!(dp.indices_in_leaf(0), &[0, 2, 4, 6]);
    }
}
