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
    let _ = (num_bin_per_column, shared_hist_size);
    todo!("15-02: DivideCUDAFeatureGroups — pack columns under shared_hist_size/2, spill large-bin columns to own partition")
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
    /// # Errors
    /// [`ComputeError`] on an unsupported `bit_type`/`row_ptr_type` width.
    pub fn get_sparse_data_partitioned(
        client: &cubecl::prelude::ComputeClient<R>,
        columns: &[BinColumn],
        layout: &FeaturePartitionLayout,
    ) -> Result<Self, ComputeError> {
        let _ = (client, columns, layout);
        todo!("15-03: GetSparseDataPartitioned — per-partition CSR re-lay (partition-local bins) over the 3×3 width matrix")
    }
}
