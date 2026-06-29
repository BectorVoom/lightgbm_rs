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
/// `R` is currently carried only as a marker; the per-column device-buffer handles
/// are filled by Plan 15-02.
pub struct CudaColumnData<R: cubecl::Runtime> {
    /// Per-feature numeric meta, one entry per feature (parallel to the columns).
    pub feature_meta: Vec<ColumnFeatureMeta>,
    _runtime: core::marker::PhantomData<R>,
}

impl<R: cubecl::Runtime> CudaColumnData<R> {
    /// Upload the column-major binned store: one device buffer per [`BinColumn`] at
    /// its native width, plus the per-feature [`ColumnFeatureMeta`]. Mirrors
    /// `CUDAColumnData::Init` (§3 / §14).
    ///
    /// # Errors
    /// [`ComputeError`] on an unsupported column width.
    pub fn new(
        client: &cubecl::prelude::ComputeClient<R>,
        columns: &[BinColumn],
        feature_meta: Vec<ColumnFeatureMeta>,
    ) -> Result<Self, ComputeError> {
        let _ = (client, columns, feature_meta);
        todo!("15-02: CUDAColumnData::Init — per-column native-width upload + per-feature meta")
    }

    /// Read column `column` back from device as widened `u32` bins for the ODL-03
    /// parity assert against the host [`BinColumn`] truth.
    ///
    /// # Errors
    /// [`ComputeError`] if `column` is out of range or its width is unsupported.
    pub fn read_column(
        &self,
        client: &cubecl::prelude::ComputeClient<R>,
        column: usize,
    ) -> Result<Vec<u32>, ComputeError> {
        let _ = (client, column);
        todo!("15-02: read a single column back from device (widened to u32) for parity")
    }
}
