//! The columnar bin-storage layer: the `Bin` trait (C++ `Bin` virtual
//! interface), the `BinValue` width abstraction, and the `create_dense_bin` /
//! `create_sparse_bin` width-selection factories (C++ `Bin::CreateDenseBin` /
//! `Bin::CreateSparseBin`, `bin.cpp:613-633`).
//!
//! D-01 locks `Box<dyn Bin>` trait-object dispatch (a faithful mirror of the C++
//! virtual `Bin*` factory) — NOT an enum-of-typed-vecs. D-02 includes the 4-bit
//! packed `DenseBin<u8, true>` (`num_bin <= 16`) now.
//!
//! Module organization (the `pub mod` + `pub use` re-export pattern) follows the
//! Phase-1 `lgbm-core/src/config/mod.rs` analog.

pub mod dense_bin;
pub mod sparse_bin;

pub use dense_bin::DenseBin;
pub use sparse_bin::SparseBin;

/// Bin element-width abstraction (C++ template parameter `VAL_T`). The width is
/// `u8` / `u16` / `u32`, selected from `num_bin` by the factories. The bound set
/// is the minimum the storage hot path + golden byte-layout tests need:
/// zero-init, `u32` round-trip (push/read), and a little-endian byte image.
///
/// `Copy` is required so storage cells move without allocation; the `u32`
/// conversions mirror the C++ `static_cast<VAL_T>(value)` / widening on read.
pub trait BinValue: Copy + std::fmt::Debug + 'static {
    /// The zero element (C++ `static_cast<VAL_T>(0)`).
    const ZERO: Self;
    /// C++ `static_cast<VAL_T>(value)` — narrow a `u32` bin index to the width.
    fn from_u32(v: u32) -> Self;
    /// Widen the stored element back to `u32` (C++ implicit widening on read).
    fn to_u32(self) -> u32;
    /// Little-endian byte image of the element (the exact byte layout C++ stores
    /// in `data_`/`vals_`), for byte-exact golden comparison.
    fn to_le_bytes(self) -> Vec<u8>;
}

impl BinValue for u8 {
    const ZERO: Self = 0;
    #[inline]
    fn from_u32(v: u32) -> Self {
        v as u8
    }
    #[inline]
    fn to_u32(self) -> u32 {
        self as u32
    }
    #[inline]
    fn to_le_bytes(self) -> Vec<u8> {
        vec![self]
    }
}

impl BinValue for u16 {
    const ZERO: Self = 0;
    #[inline]
    fn from_u32(v: u32) -> Self {
        v as u16
    }
    #[inline]
    fn to_u32(self) -> u32 {
        self as u32
    }
    #[inline]
    fn to_le_bytes(self) -> Vec<u8> {
        u16::to_le_bytes(self).to_vec()
    }
}

impl BinValue for u32 {
    const ZERO: Self = 0;
    #[inline]
    fn from_u32(v: u32) -> Self {
        v
    }
    #[inline]
    fn to_u32(self) -> u32 {
        self
    }
    #[inline]
    fn to_le_bytes(self) -> Vec<u8> {
        u32::to_le_bytes(self).to_vec()
    }
}

/// C++ `class Bin` virtual interface (`include/LightGBM/bin.h`), restricted to
/// the methods this phase needs. Dispatched as `Box<dyn Bin>` per D-01.
pub trait Bin: std::fmt::Debug {
    /// C++ `void Push(int tid, data_size_t idx, uint32_t value)` — store a bin
    /// index at a row. (Single-threaded core, so no `tid`.)
    fn push(&mut self, idx: i32, value: u32);

    /// C++ `void FinishLoad()` — the immutability boundary's per-bin step
    /// (4-bit OR-merge / sparse delta-encode). After this the layout is frozen.
    fn finish_load(&mut self);

    /// C++ `VAL_T data(data_size_t idx)` widened to `u32` — read a stored bin
    /// index (used by golden round-trip checks; Phase 4 uses iterators).
    fn data(&self, idx: i32) -> u32;

    /// C++ `data_size_t num_data()` — number of rows.
    fn num_data(&self) -> i32;

    /// Raw `data_` (dense) / `deltas_` (sparse) byte image for the storage-layout
    /// golden tests (`tests/bin_storage_layout.rs`).
    fn raw_bytes(&self) -> Vec<u8>;

    /// Sparse `deltas_` if this is a sparse bin (`None` for dense).
    fn sparse_deltas(&self) -> Option<&[u8]>;

    /// Sparse `vals_` little-endian byte image if this is a sparse bin.
    fn sparse_vals_bytes(&self) -> Option<Vec<u8>>;
}

/// C++ `Bin* Bin::CreateDenseBin(data_size_t num_data, int num_bin)`
/// (`bin.cpp:613-622`) — width selection from `num_bin`, transcribed EXACTLY:
///
/// ```text
/// num_bin <= 16     -> DenseBin<u8,  true>   (IS_4BIT, D-02)
/// num_bin <= 256    -> DenseBin<u8,  false>
/// num_bin <= 65536  -> DenseBin<u16, false>
/// else              -> DenseBin<u32, false>
/// ```
pub fn create_dense_bin(num_data: i32, num_bin: i32) -> Box<dyn Bin> {
    if num_bin <= 16 {
        Box::new(DenseBin::<u8, true>::new(num_data))
    } else if num_bin <= 256 {
        Box::new(DenseBin::<u8, false>::new(num_data))
    } else if num_bin <= 65536 {
        Box::new(DenseBin::<u16, false>::new(num_data))
    } else {
        Box::new(DenseBin::<u32, false>::new(num_data))
    }
}

/// C++ `Bin* Bin::CreateSparseBin(data_size_t num_data, int num_bin)`
/// (`bin.cpp:624-632`) — width selection from `num_bin`, transcribed EXACTLY
/// (note: no 4-bit sparse variant in C++):
///
/// ```text
/// num_bin <= 256    -> SparseBin<u8>
/// num_bin <= 65536  -> SparseBin<u16>
/// else              -> SparseBin<u32>
/// ```
pub fn create_sparse_bin(num_data: i32, num_bin: i32) -> Box<dyn Bin> {
    if num_bin <= 256 {
        Box::new(SparseBin::<u8>::new(num_data))
    } else if num_bin <= 65536 {
        Box::new(SparseBin::<u16>::new(num_data))
    } else {
        Box::new(SparseBin::<u32>::new(num_data))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn create_dense_bin_selects_width_and_4bit_thresholds() {
        // num_bin <= 16 -> 4-bit u8 (1 byte per 2 rows): 4 rows -> 2 bytes.
        let b16 = create_dense_bin(4, 16);
        assert_eq!(b16.raw_bytes().len(), 2, "num_bin<=16 -> 4-bit packed");

        // num_bin == 17..=256 -> full u8 (1 byte per row).
        let b256 = create_dense_bin(4, 256);
        assert_eq!(b256.raw_bytes().len(), 4, "num_bin<=256 -> u8 dense");

        // num_bin == 257..=65536 -> u16 (2 bytes per row).
        let b65536 = create_dense_bin(4, 65536);
        assert_eq!(b65536.raw_bytes().len(), 8, "num_bin<=65536 -> u16 dense");

        // > 65536 -> u32 (4 bytes per row).
        let bbig = create_dense_bin(4, 65537);
        assert_eq!(bbig.raw_bytes().len(), 16, ">65536 -> u32 dense");
    }

    #[test]
    fn create_dense_bin_4bit_roundtrips_through_trait_object() {
        let mut b = create_dense_bin(2, 16);
        b.push(0, 0x7);
        b.push(1, 0xB);
        b.finish_load();
        assert_eq!(b.data(0), 0x7);
        assert_eq!(b.data(1), 0xB);
        // packed byte: high=0xB, low=0x7 -> 0xB7.
        assert_eq!(b.raw_bytes(), vec![0xB7]);
    }

    #[test]
    fn create_sparse_bin_selects_width() {
        let mut s = create_sparse_bin(10, 256);
        s.push(3, 5);
        s.finish_load();
        assert_eq!(s.sparse_deltas(), Some(&[3u8, 0u8][..]));
        // u8 width: vals are 1 byte each.
        assert_eq!(s.sparse_vals_bytes(), Some(vec![5u8]));
    }
}
