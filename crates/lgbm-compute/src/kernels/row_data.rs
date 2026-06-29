//! `CUDARowData` device row-wise binned matrix + feature-partition layout — **15-02/15-03**.
//!
//! Owning plans: **15-02** (dense re-lay + layout), **15-03** (sparse CSR re-lay).
//! Scope locked by **ODL-03**, **D-04** (sparse synthesizer), **D-08** (cpu f64 anchor).
//!
//! ## What lives here (design doc §13, `cuda_row_data.{hpp,cpp}`)
//! The on-device row-wise binned feature matrix **plus the feature-partition
//! layout the histogram kernel (§7) is built around**. Pure host-side
//! infrastructure: it owns device buffers and uploads them, but defines no compute
//! kernel of its own (the bagging `CopySubrow` path lives in [`super::copy_subrow`]).
//!
//! - **Storage & width selectors.** Dense or sparse. [`CudaRowData::bit_type`] ∈
//!   {8,16,32} picks the live bin buffer; [`CudaRowData::row_ptr_bit_type`] ∈
//!   {16,32,64} picks the sparse CSR row-pointer width. The 3×3 (bit_type ×
//!   row_ptr_type) combinations are dispatched explicitly (Pitfall 2: an
//!   unsupported width is a typed [`ComputeError`], NEVER a silent widen).
//! - **Feature partitions** ([`divide_cuda_feature_groups`]). Features are grouped
//!   so one partition's histogram fits shared memory: budget
//!   `max_num_bin_per_partition = shared_hist_size / 2` (each entry is a grad+hess
//!   pair). A column whose own bin count exceeds the budget becomes its OWN
//!   large-bin partition (Pitfall 1: [`FeaturePartitionLayout::num_large_bin_partition`]
//!   > 0 ⇒ the kernel uses the `_GlobalMemory` path).
//!
//! ## Layout-difference warning (Pitfall 3)
//! The §13 row store is **row-major, partition-local**: a bin is read
//! `data[idx * ncol + tx]` — this is NOT the feature-major `resident_bins` buffer
//! the histogram build uses elsewhere. The sparse re-lay subtracts
//! `partition_hist_offsets[partition]` so each stored bin is partition-local
//! (Pitfall 4).

use crate::error::ComputeError;
use crate::BinColumn;

/// The feature-partition layout consumed by the histogram launcher (§13 accessor
/// table). Produced by [`divide_cuda_feature_groups`]; every field mirrors a
/// `CUDARowData` accessor.
#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub struct FeaturePartitionLayout {
    /// `cuda_feature_partition_column_index_offsets()` — partition `i` owns columns
    /// `[off[i], off[i + 1])` (length `num_feature_partitions + 1`).
    pub feature_partition_column_index_offsets: Vec<usize>,
    /// `cuda_column_hist_offsets()` — per-column bin offset WITHIN its partition
    /// (partition-local; Pitfall 4).
    pub column_hist_offsets: Vec<usize>,
    /// `cuda_partition_hist_offsets()` — global bin offset where each partition
    /// begins (length `num_feature_partitions + 1`).
    pub partition_hist_offsets: Vec<usize>,
    /// `max_num_column_per_partition()` — sizes the dense block_dim_x / sparse
    /// per-row nnz.
    pub max_num_column_per_partition: usize,
    /// `num_feature_partitions()` — = the histogram grid_dim_x.
    pub num_feature_partitions: usize,
    /// `NumLargeBinPartition()` — number of single-column partitions whose bin
    /// count exceeds the SMEM budget (Pitfall 1).
    pub num_large_bin_partition: usize,
    /// `shared_hist_size()` — the SMEM capacity the budget is derived from
    /// (`max_num_bin_per_partition = shared_hist_size / 2`).
    pub shared_hist_size: usize,
}

/// Walk the per-column bin counts and pack columns into partitions whose
/// histogram fits shared memory (`max_num_bin_per_partition = shared_hist_size /
/// 2`). A column whose own bin count exceeds the budget becomes its own large-bin
/// partition (Pitfall 1). Mirrors `CUDARowData::DivideCUDAFeatureGroups` (§13).
///
/// `num_bin_per_column[c]` is feature `c`'s bin count; `shared_hist_size` is the
/// grad+hess-pair SMEM capacity (e.g. `6144` ⇒ budget `3072`).
#[must_use]
pub fn divide_cuda_feature_groups(
    num_bin_per_column: &[usize],
    shared_hist_size: usize,
) -> FeaturePartitionLayout {
    // D-02 / §13: `max_num_bin_per_partition = shared_hist_size / 2`. Each histogram
    // entry is a `(grad, hess)` pair, so the SMEM budget in *bins* is half the entry
    // capacity — the `/2` is the parity-load-bearing detail (Pitfall 1). `6144` is the
    // design-doc DP value (A1); per §17 the grouping has NO float-parity impact (it only
    // decides shared-vs-global spill), so this is plain host `usize` math.
    let budget = shared_hist_size / 2;
    let n = num_bin_per_column.len();

    // `prefix[c]` = global bin offset where column `c` begins (running prefix sum).
    // Bin counts are bounded (a column has at most `max_bin` bins), so a `saturating_add`
    // is the conservative guard here; the device-buffer length products that can actually
    // overflow are `checked_mul`-guarded at the re-lay boundary (T-15-PART).
    let mut prefix = Vec::with_capacity(n + 1);
    prefix.push(0usize);
    for &b in num_bin_per_column {
        let last = *prefix.last().expect("prefix is seeded with 0");
        prefix.push(last.saturating_add(b));
    }

    // Walk columns in order, packing into partitions. `(lo, hi, is_large)` per partition.
    let mut partitions: Vec<(usize, usize, bool)> = Vec::new();
    let mut partition_start: Option<usize> = None;
    for c in 0..n {
        let b = num_bin_per_column[c];
        if b > budget {
            // A column whose own bin count exceeds the budget becomes its OWN large-bin
            // partition (Pitfall 1). Close the current accumulating partition first so the
            // small run preceding a large column is not silently merged into it.
            if let Some(start) = partition_start.take() {
                partitions.push((start, c, false));
            }
            partitions.push((c, c + 1, true));
        } else {
            match partition_start {
                None => partition_start = Some(c),
                Some(start) => {
                    // Close + reopen iff including column `c` would exceed the budget.
                    if prefix[c + 1] - prefix[start] > budget {
                        partitions.push((start, c, false));
                        partition_start = Some(c);
                    }
                }
            }
        }
    }
    if let Some(start) = partition_start.take() {
        partitions.push((start, n, false));
    }

    // Build the §13 accessor tables from the partition runs.
    let mut feature_partition_column_index_offsets = Vec::with_capacity(partitions.len() + 1);
    let mut partition_hist_offsets = Vec::with_capacity(partitions.len() + 1);
    let mut column_hist_offsets = vec![0usize; n];
    feature_partition_column_index_offsets.push(0);
    partition_hist_offsets.push(0);
    let mut num_large_bin_partition = 0usize;
    let mut max_num_column_per_partition = 0usize;
    for &(lo, hi, is_large) in &partitions {
        feature_partition_column_index_offsets.push(hi);
        partition_hist_offsets.push(prefix[hi]);
        if is_large {
            num_large_bin_partition += 1;
        }
        let ncol = hi - lo;
        if ncol > max_num_column_per_partition {
            max_num_column_per_partition = ncol;
        }
        // `column_hist_offsets[c]` is PARTITION-LOCAL: it resets to 0 at each partition
        // start (`prefix[c] - prefix[lo]`), Pitfall 4.
        for c in lo..hi {
            column_hist_offsets[c] = prefix[c] - prefix[lo];
        }
    }

    FeaturePartitionLayout {
        feature_partition_column_index_offsets,
        column_hist_offsets,
        partition_hist_offsets,
        max_num_column_per_partition,
        num_feature_partitions: partitions.len(),
        num_large_bin_partition,
        shared_hist_size,
    }
}

/// `CUDARowData` — the on-device row-wise binned matrix + its
/// [`FeaturePartitionLayout`]. Generic over the cubecl runtime `R` so the SAME
/// struct serves the cpu f64 anchor (D-08) and the GPU backends.
///
/// `R` is currently carried only as a marker; the device-buffer handle fields are
/// filled by Plan 15-02/15-03 (the `bit_type` × `row_ptr_type` 3×3 buffers).
pub struct CudaRowData<R: cubecl::Runtime> {
    /// The feature-partition layout the histogram launcher consumes.
    pub layout: FeaturePartitionLayout,
    /// Whether the store is sparse (CSR) vs dense.
    pub is_sparse: bool,
    _runtime: core::marker::PhantomData<R>,
}

impl<R: cubecl::Runtime> CudaRowData<R> {
    /// The live bin buffer width selector ∈ {8, 16, 32} (`bit_type()`).
    #[must_use]
    pub fn bit_type(&self) -> u32 {
        todo!("15-02: report the live dense/sparse bin buffer width (8/16/32)")
    }

    /// The sparse CSR row-pointer width selector ∈ {16, 32, 64}
    /// (`row_ptr_bit_type()`). Only meaningful when [`Self::is_sparse`].
    #[must_use]
    pub fn row_ptr_bit_type(&self) -> u32 {
        todo!("15-03: report the sparse CSR row-pointer width (16/32/64)")
    }

    /// `NumLargeBinPartition()` — number of single-column partitions too big for
    /// SMEM (Pitfall 1).
    #[must_use]
    pub fn num_large_bin_partition(&self) -> usize {
        self.layout.num_large_bin_partition
    }

    /// `cuda_feature_partition_column_index_offsets()` accessor.
    #[must_use]
    pub fn feature_partition_column_index_offsets(&self) -> &[usize] {
        &self.layout.feature_partition_column_index_offsets
    }

    /// `cuda_column_hist_offsets()` accessor (partition-local; Pitfall 4).
    #[must_use]
    pub fn column_hist_offsets(&self) -> &[usize] {
        &self.layout.column_hist_offsets
    }

    /// `cuda_partition_hist_offsets()` accessor.
    #[must_use]
    pub fn partition_hist_offsets(&self) -> &[usize] {
        &self.layout.partition_hist_offsets
    }

    /// `max_num_column_per_partition()` accessor.
    #[must_use]
    pub fn max_num_column_per_partition(&self) -> usize {
        self.layout.max_num_column_per_partition
    }

    /// `num_feature_partitions()` accessor (= histogram grid_dim_x).
    #[must_use]
    pub fn num_feature_partitions(&self) -> usize {
        self.layout.num_feature_partitions
    }

    /// `shared_hist_size()` accessor.
    #[must_use]
    pub fn shared_hist_size(&self) -> usize {
        self.layout.shared_hist_size
    }

    /// Re-lay DENSE row-wise binned data per-partition (row-major, partition-local
    /// `data[idx * ncol + tx]`; Pitfall 3) and read it back per (row, column) for
    /// the ODL-03 parity assert. Mirrors `GetDenseDataPartitioned` (§13).
    ///
    /// # Errors
    /// [`ComputeError`] on an unsupported `bit_type` width (Pitfall 2: never a
    /// silent widen).
    pub fn get_dense_data_partitioned(
        client: &cubecl::prelude::ComputeClient<R>,
        columns: &[BinColumn],
        layout: &FeaturePartitionLayout,
    ) -> Result<Self, ComputeError> {
        let _ = (client, columns, layout);
        todo!("15-02: GetDenseDataPartitioned — per-partition row-major re-lay + read-back")
    }

    /// Build per-partition CSR for SPARSE data, subtracting
    /// `partition_hist_offsets[partition]` to make each bin partition-local
    /// (Pitfall 4). Drives the `bit_type` × `row_ptr_type` 3×3 dispatch; an
    /// unsupported width is a typed error (Pitfall 2). Mirrors
    /// `GetSparseDataPartitioned` (§13).
    ///
    /// `row_ptr_bit_type` ∈ {16, 32, 64} selects the CSR row-pointer width (in the
    /// real path this is derived from the total nnz; it is passed explicitly here so
    /// the 3×3 matrix is exercisable without materializing a 2^32-nnz column —
    /// D-04). Any other value is an unsupported width (Pitfall 2).
    ///
    /// # Errors
    /// [`ComputeError`] on an unsupported `bit_type`/`row_ptr_type` width.
    pub fn get_sparse_data_partitioned(
        client: &cubecl::prelude::ComputeClient<R>,
        columns: &[BinColumn],
        layout: &FeaturePartitionLayout,
        row_ptr_bit_type: u32,
    ) -> Result<Self, ComputeError> {
        let _ = (client, columns, layout, row_ptr_bit_type);
        todo!("15-03: GetSparseDataPartitioned — per-partition CSR re-lay (partition-local bins) over the 3×3 width matrix")
    }

    /// Read the re-laid store's LOGICAL global bin at `(row, column)` back from
    /// device (dense or sparse, dispatched on [`Self::is_sparse`]) for the ODL-03
    /// parity assert against host [`BinColumn::bin`] truth.
    ///
    /// # Errors
    /// [`ComputeError`] if `(row, column)` is out of range.
    pub fn read_bin(
        &self,
        client: &cubecl::prelude::ComputeClient<R>,
        row: usize,
        column: usize,
    ) -> Result<u32, ComputeError> {
        let _ = (client, row, column);
        todo!("15-02/15-03: read back the logical global bin at (row, column) for parity")
    }

    /// Read the PARTITION-LOCAL stored bin for `partition`'s `local_row` and
    /// `local_column` (the column's index WITHIN the partition). Equals the logical
    /// global bin minus `partition_hist_offsets[partition]` (Pitfall 4) — the
    /// re-lay's partition-local invariant the sparse test asserts.
    ///
    /// # Errors
    /// [`ComputeError`] if any index is out of range.
    pub fn read_partition_local_bin(
        &self,
        client: &cubecl::prelude::ComputeClient<R>,
        partition: usize,
        local_row: usize,
        local_column: usize,
    ) -> Result<u32, ComputeError> {
        let _ = (client, partition, local_row, local_column);
        todo!("15-02/15-03: read back the partition-local stored bin (Pitfall 4)")
    }
}
