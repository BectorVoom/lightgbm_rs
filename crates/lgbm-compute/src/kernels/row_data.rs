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

use core::marker::PhantomData;

use cubecl::prelude::*;
use cubecl::server::Handle;

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
    /// The live bin buffer width ∈ {8, 16, 32} (= the widest column in the store).
    bit_type: u32,
    /// The sparse CSR row-pointer width ∈ {16, 32, 64}; `0` for a dense store.
    row_ptr_bit_type: u32,
    /// Rows in the store (every column shares this length).
    num_rows: usize,
    /// Total columns across all partitions.
    num_columns: usize,
    /// The ONCE-uploaded row-major partition-local bin buffer (Pitfall 3): a NEW
    /// buffer, distinct from the feature-major `resident_bins` (lib.rs:2360). Layout:
    /// partition `p`'s region is `[off[p]·num_rows, off[p+1]·num_rows)`, and within it
    /// cell `(row, local_col)` sits at `row·ncol_p + local_col`. `Handle` is cheaply
    /// clonable (ref-counted) — `read_bin` clones it to read back.
    data_handle: Handle,
    /// The ONCE-uploaded per-partition CSR row pointers (sparse only); `None` for dense.
    row_ptr_handle: Option<Handle>,
    _runtime: PhantomData<R>,
}

impl<R: cubecl::Runtime> CudaRowData<R> {
    /// The live bin buffer width selector ∈ {8, 16, 32} (`bit_type()`).
    #[must_use]
    pub fn bit_type(&self) -> u32 {
        self.bit_type
    }

    /// The sparse CSR row-pointer width selector ∈ {16, 32, 64}
    /// (`row_ptr_bit_type()`). Only meaningful when [`Self::is_sparse`] (`0` for dense).
    #[must_use]
    pub fn row_ptr_bit_type(&self) -> u32 {
        self.row_ptr_bit_type
    }

    /// The ONCE-uploaded row-major partition-local bin buffer handle — the direct
    /// Phase-16 histogram input (`blockIdx.x` = partition, `threadIdx.x` = column).
    #[must_use]
    pub fn data_handle(&self) -> &Handle {
        &self.data_handle
    }

    /// The ONCE-uploaded per-partition CSR row-pointer handle (sparse only; `None` for
    /// dense) — consumed by the Phase-16 sparse histogram launcher alongside
    /// [`Self::row_ptr_bit_type`].
    #[must_use]
    pub fn row_ptr_handle(&self) -> Option<&Handle> {
        self.row_ptr_handle.as_ref()
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
        let (num_rows, num_columns) = validate_columns(columns)?;
        // `bit_type` is the WIDEST column width (the store is one uniform-width buffer;
        // narrower columns widen losslessly into it — the `resident_bins` qix shape, but
        // row-major). C++ uses a single `bit_type()` for the whole row matrix.
        let bit_type = columns
            .iter()
            .map(column_bit_width)
            .max()
            .unwrap_or(8);

        // T-15-PART: overflow-check the buffer length product BEFORE any device alloc.
        let total = num_columns.checked_mul(num_rows).ok_or_else(|| ComputeError::Runtime {
            detail: format!(
                "GetDenseDataPartitioned: num_columns {num_columns} * num_rows {num_rows} \
                 overflows usize"
            ),
        })?;

        // Assemble the NEW row-major partition-local buffer (Pitfall 3 — NOT the
        // feature-major resident buffer). Dense stores the RAW per-column bin (the kernel
        // applies `column_hist_offsets` at accumulation time, §7.2); `read_bin` adds the
        // global offset back so it returns the logical global bin.
        let mut vals: Vec<u32> = Vec::with_capacity(total);
        for p in 0..layout.num_feature_partitions {
            let lo = layout.feature_partition_column_index_offsets[p];
            let hi = layout.feature_partition_column_index_offsets[p + 1];
            for row in 0..num_rows {
                for column in &columns[lo..hi] {
                    vals.push(column.bin(row));
                }
            }
        }
        debug_assert_eq!(vals.len(), total, "row-major re-lay fills exactly num_columns*num_rows");

        // Upload ONCE at native width (D-09 alloc-once; no per-call `create_from_slice`
        // in a loop). Pure store/relay: a safe `create_from_slice` round-trip, no kernel
        // launch and therefore no `unsafe` (CMP-01: the only unsafe in the crate stays in
        // kernel launchers; this re-lay has none).
        let data_handle = upload_store_buffer(client, &vals, bit_type)?;

        Ok(Self {
            layout: layout.clone(),
            is_sparse: false,
            bit_type,
            row_ptr_bit_type: 0,
            num_rows,
            num_columns,
            data_handle,
            row_ptr_handle: None,
            _runtime: PhantomData,
        })
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
        if row >= self.num_rows || column >= self.num_columns {
            return Err(ComputeError::Runtime {
                detail: format!(
                    "read_bin: (row={row}, column={column}) out of range \
                     (num_rows={}, num_columns={})",
                    self.num_rows, self.num_columns
                ),
            });
        }
        let partition = self.partition_of(column)?;
        let stored = self.read_stored_cell(client, partition, row, column)?;
        // The stored cell is partition-local for sparse (the re-lay already subtracted
        // `partition_hist_start`) but RAW for dense — add back the partition-local column
        // offset in the dense case so both return the same logical GLOBAL bin.
        let part_off = self.layout.partition_hist_offsets[partition] as u32;
        let global = if self.is_sparse {
            stored + part_off
        } else {
            stored + self.layout.column_hist_offsets[column] as u32 + part_off
        };
        Ok(global)
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
        if partition >= self.layout.num_feature_partitions {
            return Err(ComputeError::Runtime {
                detail: format!(
                    "read_partition_local_bin: partition {partition} out of range \
                     (num_feature_partitions={})",
                    self.layout.num_feature_partitions
                ),
            });
        }
        let lo = self.layout.feature_partition_column_index_offsets[partition];
        let hi = self.layout.feature_partition_column_index_offsets[partition + 1];
        let ncol_p = hi - lo;
        if local_column >= ncol_p || local_row >= self.num_rows {
            return Err(ComputeError::Runtime {
                detail: format!(
                    "read_partition_local_bin: (local_row={local_row}, \
                     local_column={local_column}) out of range (ncol_p={ncol_p}, \
                     num_rows={})",
                    self.num_rows
                ),
            });
        }
        let column = lo + local_column;
        let stored = self.read_stored_cell(client, partition, local_row, column)?;
        // Partition-local bin = global - partition_hist_offsets[partition] (Pitfall 4).
        // Sparse already stores it partition-local; dense stores RAW, so fold the
        // partition-local column offset in.
        let local = if self.is_sparse {
            stored
        } else {
            stored + self.layout.column_hist_offsets[column] as u32
        };
        Ok(local)
    }

    /// Decode the ONCE-uploaded store buffer's cell `(row, column)` (read back from the
    /// cached device [`Handle`]). `partition` is `column`'s owning partition. Returns the
    /// raw stored value (dense: raw bin; sparse: partition-local bin). The read is a safe
    /// `read_one_unchecked` round-trip — no kernel launch, no `unsafe` (CMP-01).
    fn read_stored_cell(
        &self,
        client: &cubecl::prelude::ComputeClient<R>,
        partition: usize,
        row: usize,
        column: usize,
    ) -> Result<u32, ComputeError> {
        let lo = self.layout.feature_partition_column_index_offsets[partition];
        let hi = self.layout.feature_partition_column_index_offsets[partition + 1];
        let ncol_p = hi - lo;
        // partition base = lo·num_rows, then row-major `row·ncol_p + local_col`.
        let idx = lo * self.num_rows + row * ncol_p + (column - lo);
        let bytes = client.read_one_unchecked(self.data_handle.clone());
        Ok(decode_store_cell(&bytes, self.bit_type, idx))
    }

    /// The partition index owning `column` (`off[p] <= column < off[p+1]`).
    fn partition_of(&self, column: usize) -> Result<usize, ComputeError> {
        let offs = &self.layout.feature_partition_column_index_offsets;
        for p in 0..self.layout.num_feature_partitions {
            if column >= offs[p] && column < offs[p + 1] {
                return Ok(p);
            }
        }
        Err(ComputeError::Runtime {
            detail: format!("column {column} belongs to no partition"),
        })
    }
}

/// Bin-store element width of a [`BinColumn`] (8/16/32) — selects `bit_type`.
fn column_bit_width(col: &BinColumn) -> u32 {
    match col {
        BinColumn::U8(_) => 8,
        BinColumn::U16(_) => 16,
        BinColumn::U32(_) => 32,
    }
}

/// Validate that every column shares one row count; return `(num_rows, num_columns)`.
///
/// # Errors
/// [`ComputeError`] if `columns` is empty or any column length differs.
fn validate_columns(columns: &[BinColumn]) -> Result<(usize, usize), ComputeError> {
    let num_columns = columns.len();
    if num_columns == 0 {
        return Err(ComputeError::Runtime {
            detail: "row data re-lay requires >= 1 column".to_string(),
        });
    }
    let num_rows = columns[0].len();
    for (i, c) in columns.iter().enumerate() {
        if c.len() != num_rows {
            return Err(ComputeError::LengthMismatch {
                expected: num_rows,
                actual: c.len(),
            });
        }
        let _ = i;
    }
    Ok((num_rows, num_columns))
}

/// Upload a `u32`-valued buffer to the device at the selected `bit_type` width, narrowing
/// losslessly (the value is a bin INDEX, byte-faithful across widths). ONE
/// `create_from_slice` — the alloc-once upload (D-09).
///
/// # Errors
/// [`ComputeError`] on an unsupported `bit_type` (Pitfall 2 — never a silent widen).
fn upload_store_buffer<R: cubecl::Runtime>(
    client: &cubecl::prelude::ComputeClient<R>,
    vals_u32: &[u32],
    bit_type: u32,
) -> Result<Handle, ComputeError> {
    let handle = match bit_type {
        8 => {
            let v: Vec<u8> = vals_u32.iter().map(|&x| x as u8).collect();
            client.create_from_slice(u8::as_bytes(&v))
        }
        16 => {
            let v: Vec<u16> = vals_u32.iter().map(|&x| x as u16).collect();
            client.create_from_slice(u16::as_bytes(&v))
        }
        32 => client.create_from_slice(u32::as_bytes(vals_u32)),
        other => {
            return Err(ComputeError::Runtime {
                detail: format!("unsupported bin bit_type {other} (expected 8/16/32)"),
            })
        }
    };
    Ok(handle)
}

/// Decode cell `idx` of a store buffer's raw bytes at the given `bit_type` to `u32`.
fn decode_store_cell(bytes: &[u8], bit_type: u32, idx: usize) -> u32 {
    match bit_type {
        8 => u32::from(u8::from_bytes(bytes)[idx]),
        16 => u32::from(u16::from_bytes(bytes)[idx]),
        _ => u32::from_bytes(bytes)[idx],
    }
}
