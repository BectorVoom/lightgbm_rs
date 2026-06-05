//! Bit-for-bit port of `MultiValBin` dense/sparse storage for bundled (EFB)
//! groups, from `LightGBM/src/io/multi_val_dense_bin.hpp`,
//! `LightGBM/src/io/multi_val_sparse_bin.hpp`, and the
//! `MultiValBin::CreateMultiValBin` / `CreateMultiValDenseBin` /
//! `CreateMultiValSparseBin` factories (`bin.cpp:635-706`).
//!
//! Follows the `lgbm-core/src/random.rs` bit-exact transcription discipline.
//!
//! # Layout notes (load-bearing)
//!
//! A `MultiValBin` stores, per row, the bundled bin values of every sub-feature
//! in one EFB group. Two storage forms exist, selected by `sum_sparse_rate`:
//!
//! - **Dense** (`CreateMultiValDenseBin`, used when
//!   `sparse_rate < multi_val_bin_sparse_threshold (0.25)`): a flat row-major
//!   `data_` of `num_data * num_feature` cells. `PushOneRow(idx, values)` writes
//!   `data_[idx * num_feature + j] = values[j]` for every sub-feature `j`
//!   (`multi_val_dense_bin.hpp:44-49`). Every row stores ALL sub-features,
//!   including the implicit most-freq (bin 0). The element width (`u8`/`u16`/`u32`)
//!   is chosen from the MAX per-feature bin span over `offsets` (`bin.cpp:650-665`).
//!
//! - **Sparse** (`CreateMultiValSparseBin`, used when `sparse_rate >= 0.25`): a
//!   CSR layout — `row_ptr_` (`num_data + 1`) plus a packed `data_` of the
//!   non-most-freq entries (each already offset-added, see `feature_group.rs`
//!   multi-val push). `PushOneRow(idx, values)` appends `values` and records
//!   `row_ptr_[idx + 1] = values.len()`; `FinishLoad` prefix-sums `row_ptr_`
//!   (`multi_val_sparse_bin.hpp:52-108`). The element width is chosen from
//!   `num_bin` and the index width from the estimated total entries; for the
//!   single-threaded core we keep `data_` as `u32` and expose the chosen widths
//!   for the golden layout (the byte image is dumped width-aware).
//!
//! # The `+1` push convention (consumed by `feature_group.rs`)
//!
//! The bundled-group `FeatureGroup::PushData` multi-val branch pushes
//! `bin + 1` into the per-sub-feature storage so bin `0` stays reserved for the
//! implicit most-freq value (`feature_group.h:253-267`, RESEARCH §10). The
//! `MultiValBin` itself stores whatever value it is handed; the `+1`/offset
//! adjustment is the caller's responsibility (mirrors C++ exactly — see
//! `PushDataToMultiValBin`, `dataset.cpp:465-516`).

/// C++ `MultiValBin::multi_val_bin_sparse_threshold` (`bin.h:599`).
pub const MULTI_VAL_BIN_SPARSE_THRESHOLD: f64 = 0.25;

/// Element-width tag for a `MultiValBin`, mirroring the C++ `VAL_T` template
/// parameter selected by `CreateMultiValDenseBin` / `CreateMultiValSparseBin`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MultiValWidth {
    /// C++ `uint8_t`.
    U8,
    /// C++ `uint16_t`.
    U16,
    /// C++ `uint32_t`.
    U32,
}

impl MultiValWidth {
    /// C++ `CreateMultiValDenseBin` width selection on the MAX per-feature bin
    /// span (`bin.cpp:650-664`): `<=256 -> u8`, `<=65536 -> u16`, else `u32`.
    fn from_max_bin(max_bin: i32) -> Self {
        if max_bin <= 256 {
            MultiValWidth::U8
        } else if max_bin <= 65536 {
            MultiValWidth::U16
        } else {
            MultiValWidth::U32
        }
    }

    /// C++ `CreateMultiValSparseBin` VAL_T selection on `num_bin`
    /// (`bin.cpp:672-705`): `<=256 -> u8`, `<=65536 -> u16`, else `u32`.
    fn from_num_bin(num_bin: i32) -> Self {
        Self::from_max_bin(num_bin)
    }

    /// Width in bytes (for the byte-image golden dump).
    fn size_of(self) -> usize {
        match self {
            MultiValWidth::U8 => 1,
            MultiValWidth::U16 => 2,
            MultiValWidth::U32 => 4,
        }
    }

    /// Little-endian byte image of `value` narrowed to this width
    /// (C++ `static_cast<VAL_T>(value)` byte layout).
    fn to_le_bytes(self, value: u32) -> Vec<u8> {
        match self {
            MultiValWidth::U8 => vec![value as u8],
            MultiValWidth::U16 => (value as u16).to_le_bytes().to_vec(),
            MultiValWidth::U32 => value.to_le_bytes().to_vec(),
        }
    }
}

/// C++ `class MultiValBin` (`bin.h`), the bundled-group store. Built by
/// [`MultiValBin::create`], which dispatches dense vs sparse exactly like
/// `CreateMultiValBin` (`bin.cpp:635-643`).
#[derive(Debug)]
pub struct MultiValBin {
    /// C++ `data_size_t num_data_`.
    num_data_: i32,
    /// C++ `int num_bin_` (== `offsets.back()`).
    num_bin_: i32,
    /// C++ `int num_feature_` (dense only; sparse has no fixed per-row width).
    num_feature_: i32,
    /// C++ `std::vector<uint32_t> offsets_` — per-sub-feature bin offsets.
    offsets_: Vec<u32>,
    /// `true` => CSR sparse layout; `false` => row-major dense layout.
    is_sparse_: bool,
    /// The element width C++ selected for `VAL_T`.
    width_: MultiValWidth,
    /// Dense: `num_data * num_feature` row-major values (`u32` cells, dumped
    /// width-aware). Sparse: the packed non-most-freq entries (CSR `data_`).
    data_: Vec<u32>,
    /// Sparse only: CSR `row_ptr_` (`num_data + 1`); accumulated at `finish_load`.
    row_ptr_: Vec<u32>,
}

impl MultiValBin {
    /// C++ `MultiValBin* MultiValBin::CreateMultiValBin(num_data, num_bin,
    /// num_feature, sparse_rate, offsets)` (`bin.cpp:635-643`):
    ///
    /// ```text
    /// if sparse_rate >= multi_val_bin_sparse_threshold (0.25):
    ///   CreateMultiValSparseBin(num_data, num_bin, (1 - sparse_rate) * num_feature)
    /// else:
    ///   CreateMultiValDenseBin(num_data, num_bin, num_feature, offsets)
    /// ```
    pub fn create(
        num_data: i32,
        num_bin: i32,
        num_feature: i32,
        sparse_rate: f64,
        offsets: Vec<u32>,
    ) -> Self {
        if sparse_rate >= MULTI_VAL_BIN_SPARSE_THRESHOLD {
            // Sparse: VAL_T from num_bin (the index width is derived from the
            // estimated entry count in C++; the single-threaded core keeps the
            // CSR row_ptr_ in u32, sufficient for the golden layout).
            MultiValBin {
                num_data_: num_data,
                num_bin_: num_bin,
                num_feature_: num_feature,
                offsets_: offsets,
                is_sparse_: true,
                width_: MultiValWidth::from_num_bin(num_bin),
                data_: Vec::new(),
                row_ptr_: vec![0u32; (num_data + 1) as usize],
            }
        } else {
            // Dense: VAL_T from the MAX per-feature bin span over offsets.
            let mut max_bin = 0i32;
            for i in 0..offsets.len().saturating_sub(1) {
                let feature_bin = (offsets[i + 1] - offsets[i]) as i32;
                if feature_bin > max_bin {
                    max_bin = feature_bin;
                }
            }
            let len = (num_data as usize) * (num_feature as usize);
            MultiValBin {
                num_data_: num_data,
                num_bin_: num_bin,
                num_feature_: num_feature,
                offsets_: offsets,
                is_sparse_: false,
                width_: MultiValWidth::from_max_bin(max_bin),
                data_: vec![0u32; len],
                row_ptr_: Vec::new(),
            }
        }
    }

    /// C++ `void PushOneRow(int tid, data_size_t idx, const std::vector<uint32_t>&
    /// values)` (dense `multi_val_dense_bin.hpp:44-49`, sparse
    /// `multi_val_sparse_bin.hpp:52-74`).
    ///
    /// - Dense: `data_[idx * num_feature + j] = values[j]` for every sub-feature
    ///   (`values.len() == num_feature`, including most-freq cells == 0).
    /// - Sparse: append `values` (the offset-added non-most-freq entries) and
    ///   record `row_ptr_[idx + 1] = values.len()` (prefix-summed at finish_load).
    pub fn push_one_row(&mut self, idx: i32, values: &[u32]) {
        if self.is_sparse_ {
            self.row_ptr_[(idx + 1) as usize] = values.len() as u32;
            self.data_.extend_from_slice(values);
        } else {
            let start = (idx as usize) * (self.num_feature_ as usize);
            for (j, &v) in values.iter().enumerate() {
                self.data_[start + j] = v;
            }
        }
    }

    /// C++ `void FinishLoad()` — sparse prefix-sums `row_ptr_`
    /// (`multi_val_sparse_bin.hpp:76-108`, `row_ptr_[i+1] += row_ptr_[i]`); dense
    /// is a no-op (`multi_val_dense_bin.hpp:51-52`).
    pub fn finish_load(&mut self) {
        if self.is_sparse_ {
            for i in 0..self.num_data_ as usize {
                self.row_ptr_[i + 1] += self.row_ptr_[i];
            }
        }
    }

    /// Whether this is the CSR sparse layout (`MultiValBin::IsSparse`).
    pub fn is_sparse(&self) -> bool {
        self.is_sparse_
    }

    /// The selected element width (the C++ `VAL_T`).
    pub fn width(&self) -> MultiValWidth {
        self.width_
    }

    /// C++ `int num_bin()`.
    pub fn num_bin(&self) -> i32 {
        self.num_bin_
    }

    /// C++ `const std::vector<uint32_t>& offsets()`.
    pub fn offsets(&self) -> &[u32] {
        &self.offsets_
    }

    /// Sparse CSR `row_ptr_` (post-finish prefix sums); empty for dense.
    pub fn row_ptr(&self) -> &[u32] {
        &self.row_ptr_
    }

    /// Read the bundled value at `(row, sub_feature)` for golden round-trips.
    /// Dense reads `data_[row * num_feature + sub]`; sparse scans the CSR row
    /// (returns the offset-added stored value, or `None` if that sub-feature was
    /// the implicit most-freq and thus omitted from the sparse row).
    pub fn data_dense(&self, row: i32, sub: i32) -> u32 {
        debug_assert!(!self.is_sparse_, "data_dense on a sparse MultiValBin");
        self.data_[(row as usize) * (self.num_feature_ as usize) + sub as usize]
    }

    /// The packed CSR `data_` slice (sparse) or the row-major `data_` (dense).
    pub fn raw_values(&self) -> &[u32] {
        &self.data_
    }

    /// Width-aware little-endian byte image of `data_` (the exact C++ `data_`
    /// byte layout, for byte-exact golden comparison).
    pub fn raw_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(self.data_.len() * self.width_.size_of());
        for &v in &self.data_ {
            out.extend_from_slice(&self.width_.to_le_bytes(v));
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn create_selects_dense_when_sparse_rate_below_threshold() {
        // sum_sparse_rate 0.1 < 0.25 -> dense (row-major num_data*num_feature).
        let mvb = MultiValBin::create(4, 9, 2, 0.1, vec![0, 5, 9]);
        assert!(!mvb.is_sparse(), "sparse_rate < 0.25 -> dense");
        // 4 rows * 2 features = 8 cells.
        assert_eq!(mvb.raw_values().len(), 8);
    }

    #[test]
    fn create_selects_sparse_when_sparse_rate_at_or_above_threshold() {
        // sum_sparse_rate 0.25 >= 0.25 -> sparse (CSR).
        let mvb = MultiValBin::create(4, 9, 2, 0.25, vec![0, 5, 9]);
        assert!(mvb.is_sparse(), "sparse_rate >= 0.25 -> sparse");
        // CSR row_ptr_ has num_data + 1 entries.
        assert_eq!(mvb.row_ptr().len(), 5);
    }

    #[test]
    fn dense_push_one_row_is_row_major_including_most_freq_cells() {
        // Dense stores ALL sub-features per row (most-freq cells == 0).
        let mut mvb = MultiValBin::create(3, 9, 2, 0.0, vec![0, 5, 9]);
        mvb.push_one_row(0, &[3, 0]); // feature 0 -> bin 3, feature 1 -> most-freq (0)
        mvb.push_one_row(1, &[0, 7]); // feature 0 -> most-freq, feature 1 -> 7
        mvb.push_one_row(2, &[2, 4]);
        mvb.finish_load();
        assert_eq!(mvb.data_dense(0, 0), 3);
        assert_eq!(mvb.data_dense(0, 1), 0);
        assert_eq!(mvb.data_dense(1, 0), 0);
        assert_eq!(mvb.data_dense(1, 1), 7);
        assert_eq!(mvb.data_dense(2, 0), 2);
        assert_eq!(mvb.data_dense(2, 1), 4);
        // row-major: [3,0, 0,7, 2,4]
        assert_eq!(mvb.raw_values(), &[3, 0, 0, 7, 2, 4]);
    }

    #[test]
    fn sparse_push_one_row_builds_csr_and_prefix_sums_row_ptr() {
        // Sparse stores only the offset-added non-most-freq entries per row.
        let mut mvb = MultiValBin::create(3, 9, 2, 0.5, vec![0, 5, 9]);
        mvb.push_one_row(0, &[3]); // one non-most-freq entry
        mvb.push_one_row(1, &[8, 1]); // two entries
        mvb.push_one_row(2, &[]); // empty row (all most-freq)
        mvb.finish_load();
        // CSR offsets: [0, 1, 3, 3]
        assert_eq!(mvb.row_ptr(), &[0, 1, 3, 3]);
        assert_eq!(mvb.raw_values(), &[3, 8, 1]);
    }

    #[test]
    fn dense_width_selected_from_max_per_feature_span() {
        // offsets [0, 300, 305]: feature 0 span = 300 (>256) -> u16.
        let mvb = MultiValBin::create(2, 305, 2, 0.0, vec![0, 300, 305]);
        assert_eq!(mvb.width(), MultiValWidth::U16, "max span 300 -> u16");
        // offsets [0, 5, 9]: max span 5 (<=256) -> u8.
        let mvb2 = MultiValBin::create(2, 9, 2, 0.0, vec![0, 5, 9]);
        assert_eq!(mvb2.width(), MultiValWidth::U8, "max span 5 -> u8");
    }

    #[test]
    fn dense_raw_bytes_are_width_aware_little_endian() {
        // u16 width: each cell is 2 LE bytes.
        let mut mvb = MultiValBin::create(1, 305, 2, 0.0, vec![0, 300, 305]);
        mvb.push_one_row(0, &[0x0102, 0x03]);
        mvb.finish_load();
        // 0x0102 -> [0x02, 0x01]; 0x0003 -> [0x03, 0x00].
        assert_eq!(mvb.raw_bytes(), vec![0x02, 0x01, 0x03, 0x00]);
    }
}
