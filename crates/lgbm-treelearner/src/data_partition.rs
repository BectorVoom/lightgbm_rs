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
    #[allow(clippy::too_many_arguments)]
    pub fn split<B: Backend>(
        &mut self,
        backend: &B,
        client: &ComputeClient<B::Runtime>,
        leaf: i32,
        right_leaf: i32,
        feature_bins: &[u32],
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
        // Backend op's stable reorder is RELATIVE to this leaf's row order.
        let leaf_rows: Vec<u32> = self.indices[begin..begin + count].to_vec();
        let leaf_feature_bins: Vec<u32> = leaf_rows
            .iter()
            .map(|&row| feature_bins[row as usize])
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

    /// Total row count (`num_data_`).
    pub fn num_data(&self) -> i32 {
        self.num_data
    }
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
        let feature_bins = vec![1u32, 5, 3, 7, 0, 4, 2, 6];
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
        let feature_bins = vec![1u32, 5, 3, 7, 0, 4, 2, 6];
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
