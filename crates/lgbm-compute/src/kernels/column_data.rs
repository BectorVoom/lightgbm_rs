//! `CUDAColumnData` device columnar binned store + per-feature numeric meta — **15-02**.
//!
//! Owning plan: **15-02**. Scope locked by **ODL-03**, **D-08** (cpu f64 anchor).
//!
//! ## What lives here (design doc §3, `cuda_column_data.{hpp,cpp}`)
//! `CUDAColumnData` holds the binned feature matrix COLUMN-wise on device. Each
//! column buffer is `uint8/16/32` depending on its bin count; a per-column
//! `bit_type` table lets kernels dispatch on width at runtime. It owns the
//! per-feature numeric meta ([`ColumnFeatureMeta`]) the prediction kernel (§10) and
//! `CopySubrow` (§3, [`super::copy_subrow`]) read.
//!
//! ## Layout-difference warning (Pitfall 3)
//! This is the COLUMN-major store (`in_cuda_data_by_column`); it is NOT the
//! row-major partition-local `data[idx * ncol + tx]` buffer of [`super::row_data`]
//! (§13). `CopySubrow` gathers across columns into the compacted subset.

use core::marker::PhantomData;

use cubecl::prelude::*;
use cubecl::server::Handle;

use crate::error::ComputeError;
use crate::BinColumn;

/// Per-feature NUMERIC binning meta (design doc §3 / §14 `cuda_feature_*`). The
/// categorical bitset meta is deferred.
///
// TODO(Phase 22): categorical bitset meta (Open Question 1 recommendation —
// categorical splits are ODL-22 / Phase 22 scope; this struct carries the numeric
// per-feature meta only).
#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub struct ColumnFeatureMeta {
    /// `cuda_column_bit_type_` — the column buffer width ∈ {8, 16, 32}.
    pub bit_type: u32,
    /// `cuda_feature_min_bin_` — the feature's minimum (in-range) bin.
    pub feature_min_bin: u32,
    /// `cuda_feature_max_bin_` — the feature's maximum (in-range) bin.
    pub feature_max_bin: u32,
    /// `cuda_feature_offset_` — the feature's global bin offset.
    pub offset: u32,
    /// `cuda_feature_most_freq_bin_` — the most-frequent bin (the compaction /
    /// default-direction pivot).
    pub most_freq_bin: u32,
    /// `cuda_feature_default_bin_` — the bin a missing/default value maps to.
    pub default_bin: u32,
    /// `cuda_feature_missing_is_zero_` — whether a missing value bins as zero.
    pub missing_is_zero: bool,
    /// `cuda_feature_missing_is_na_` — whether a missing value bins as NA.
    pub missing_is_na: bool,
    /// `cuda_feature_to_column_` — the column index this feature maps to.
    pub feature_to_column: usize,
}

/// `CUDAColumnData` — the device COLUMN-major binned store (§3). Generic over the
/// cubecl runtime `R` so the SAME struct serves the cpu f64 anchor (D-08) and the
/// GPU backends.
///
/// Unlike the §13 row store ([`super::row_data::CudaRowData`], one uniform-width
/// row-major buffer), this is COLUMN-major: each feature column gets its OWN device
/// buffer at its OWN native width (`u8`/`u16`/`u32` per its [`BinColumn`] variant),
/// so a narrow column is NOT widened up to the matrix's widest column (Pitfall 3 —
/// the two layouts coexist; this is `in_cuda_data_by_column`). All buffers are
/// uploaded ONCE at construction (D-09) and have NO consumer this phase — the
/// Phase-18 tree-walk prediction kernel is the consumer.
pub struct CudaColumnData<R: cubecl::Runtime> {
    /// Per-feature numeric meta, one entry per feature (parallel to the columns).
    pub feature_meta: Vec<ColumnFeatureMeta>,
    /// Per-column device buffer handles (column-major), uploaded ONCE at native
    /// width. `Handle` is cheaply clonable (ref-counted) — `read_column` clones to
    /// read back.
    column_handles: Vec<Handle>,
    /// Per-column bin-store width ∈ {8, 16, 32} (selects the read-back decode width).
    column_bit_types: Vec<u32>,
    /// Per-column row count (every column may differ in length only if the host does;
    /// each is read back at its own length).
    column_lens: Vec<usize>,
    _runtime: PhantomData<R>,
}

impl<R: cubecl::Runtime> CudaColumnData<R> {
    /// Upload the column-major binned store: one device buffer per [`BinColumn`] at
    /// its native width, plus the per-feature [`ColumnFeatureMeta`]. Mirrors
    /// `CUDAColumnData::Init` (§3 / §14).
    ///
    /// Each column is uploaded ONCE via `create_from_slice` at its native width
    /// (D-09 alloc-once — NO per-tree/per-call re-upload). Pure safe
    /// `create_from_slice` round-trip — no kernel launch, therefore no `unsafe`
    /// (CMP-01: the only `unsafe` in the crate stays in kernel launchers; this pure
    /// column store/relay has none).
    ///
    /// # Errors
    /// [`ComputeError::LengthMismatch`] if `feature_meta` does not have exactly one
    /// entry per column, or [`ComputeError::Runtime`] on a per-column buffer length
    /// product that overflows `usize` (T-15-PART) or an unsupported column width.
    pub fn new(
        client: &cubecl::prelude::ComputeClient<R>,
        columns: &[BinColumn],
        feature_meta: Vec<ColumnFeatureMeta>,
    ) -> Result<Self, ComputeError> {
        // The per-feature numeric meta is parallel to the columns (one entry each).
        if feature_meta.len() != columns.len() {
            return Err(ComputeError::LengthMismatch {
                expected: columns.len(),
                actual: feature_meta.len(),
            });
        }

        let mut column_handles = Vec::with_capacity(columns.len());
        let mut column_bit_types = Vec::with_capacity(columns.len());
        let mut column_lens = Vec::with_capacity(columns.len());
        for col in columns {
            let bit_type = column_bit_width(col);
            let len = col.len();
            // T-15-PART: overflow-check the per-column byte length product BEFORE the
            // device alloc (`random.rs:146` precedent). `bytes_per = bit_type / 8`.
            let bytes_per = (bit_type / 8) as usize;
            let _byte_len = len.checked_mul(bytes_per).ok_or_else(|| ComputeError::Runtime {
                detail: format!(
                    "CudaColumnData::new: column len {len} * bytes_per {bytes_per} overflows usize"
                ),
            })?;
            let vals = col.to_u32_vec();
            let handle = upload_column_buffer(client, &vals, bit_type)?;
            column_handles.push(handle);
            column_bit_types.push(bit_type);
            column_lens.push(len);
        }

        Ok(Self {
            feature_meta,
            column_handles,
            column_bit_types,
            column_lens,
            _runtime: PhantomData,
        })
    }

    /// Read column `column` back from device as widened `u32` bins for the ODL-03
    /// parity assert against the host [`BinColumn::to_u32_vec`] truth. Safe
    /// `read_one_unchecked` round-trip — no kernel launch, no `unsafe` (CMP-01).
    ///
    /// # Errors
    /// [`ComputeError::Runtime`] if `column` is out of range.
    pub fn read_column(
        &self,
        client: &cubecl::prelude::ComputeClient<R>,
        column: usize,
    ) -> Result<Vec<u32>, ComputeError> {
        if column >= self.column_handles.len() {
            return Err(ComputeError::Runtime {
                detail: format!(
                    "read_column: column {column} out of range (num_columns={})",
                    self.column_handles.len()
                ),
            });
        }
        let bit_type = self.column_bit_types[column];
        let len = self.column_lens[column];
        let bytes = client.read_one_unchecked(self.column_handles[column].clone());
        let mut out = Vec::with_capacity(len);
        for idx in 0..len {
            out.push(decode_store_cell(&bytes, bit_type, idx));
        }
        Ok(out)
    }
}

/// Bin-store element width of a [`BinColumn`] (8/16/32) — selects the per-column
/// `bit_type`. (Local to this module; mirrors the §13 row-store selector.)
fn column_bit_width(col: &BinColumn) -> u32 {
    match col {
        BinColumn::U8(_) => 8,
        BinColumn::U16(_) => 16,
        BinColumn::U32(_) => 32,
    }
}

/// Upload one `u32`-valued column to the device at its native `bit_type` width,
/// narrowing losslessly (a bin is an INDEX, byte-faithful across widths). ONE
/// `create_from_slice` — the alloc-once upload (D-09).
///
/// # Errors
/// [`ComputeError::Runtime`] on an unsupported `bit_type` (Pitfall 2 — never a
/// silent widen).
fn upload_column_buffer<R: cubecl::Runtime>(
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
                detail: format!("unsupported column bit_type {other} (expected 8/16/32)"),
            })
        }
    };
    Ok(handle)
}

/// Decode cell `idx` of a column buffer's raw bytes at the given `bit_type` to `u32`.
fn decode_store_cell(bytes: &[u8], bit_type: u32, idx: usize) -> u32 {
    match bit_type {
        8 => u32::from(u8::from_bytes(bytes)[idx]),
        16 => u32::from(u16::from_bytes(bytes)[idx]),
        _ => u32::from_bytes(bytes)[idx],
    }
}
