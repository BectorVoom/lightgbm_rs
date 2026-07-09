//! Bit-for-bit port of `DenseBin<VAL_T, IS_4BIT>` storage from
//! `LightGBM/src/io/dense_bin.hpp` (ctor 56-82, 4-bit `Push`/`FinishLoad`/`data`
//! 69-82 / 510-565).
//!
//! # Byte-layout notes (load-bearing)
//!
//! The raw `data_` byte vector this struct holds is the EXACT memory the Phase 4
//! histogram kernels read against. Any deviation in packing breaks parity, so the
//! 4-bit even/odd `buf_` split + OR-merge is transcribed verbatim:
//!
//! - **Non-4-bit:** `data_` has `num_data` elements of width `VAL_T`
//!   (`u8`/`u16`/`u32`); `push(idx, value)` writes `data_[idx] = value as VAL_T`.
//! - **4-bit (`IS_4BIT == true`, `VAL_T == u8` only):** two bins per byte.
//!   `data_`/`buf_` are each `(n + 1) / 2` bytes. `push` splits even indices
//!   (low nibble, written to `data_`) from odd indices (high nibble, written to
//!   `buf_`); `finish_load` OR-merges `data_[i] |= buf_[i]` then drops `buf_`.
//!   This even/odd split exists so the C++ OpenMP push can write the two nibbles
//!   of one byte without a data race — we mirror it for byte-identical output.
//!
//! `raw_bytes()` exposes `data_` as `&[u8]` for the storage-layout golden tests
//! (`tests/bin_storage_layout.rs`).

use super::{Bin, BinValue};

/// C++ `DenseBin<VAL_T, IS_4BIT>` (`dense_bin.hpp:52`). Fixed-width columnar bin
/// storage, optionally 4-bit-packed (two bins per byte) when `IS_4BIT`.
///
/// `VAL_T` is the element width (`u8`/`u16`/`u32`); for `IS_4BIT == true`, C++
/// asserts `sizeof(VAL_T) == 1`, so only `DenseBin<u8, true>` is valid.
#[derive(Debug)]
pub struct DenseBin<T: BinValue, const IS_4BIT: bool> {
    /// C++ `data_size_t num_data_` — number of rows.
    num_data_: i32,
    /// C++ `std::vector<VAL_T> data_` — the raw columnar store (the byte layout
    /// Phase 4 reads). For 4-bit this is `(n + 1) / 2` `u8` cells, each holding
    /// two nibbles.
    data_: Vec<T>,
    /// C++ `std::vector<uint8_t> buf_` — the odd-index nibble scratch buffer used
    /// only during load for the 4-bit path; merged into `data_` at `finish_load`
    /// and then cleared. Empty for the non-4-bit path.
    buf_: Vec<T>,
}

impl<T: BinValue, const IS_4BIT: bool> DenseBin<T, IS_4BIT> {
    /// C++ `DenseBin(data_size_t num_data)` (`dense_bin.hpp:56-65`).
    ///
    /// 4-bit resizes `data_` AND `buf_` to `(num_data + 1) / 2` zeroed cells;
    /// otherwise `data_` to `num_data` zeroed cells.
    pub fn new(num_data: i32) -> Self {
        let mut data_ = Vec::new();
        let mut buf_ = Vec::new();
        if IS_4BIT {
            // C++ CHECK_EQ(sizeof(VAL_T), 1): the 4-bit path is only valid for u8.
            debug_assert_eq!(
                std::mem::size_of::<T>(),
                1,
                "IS_4BIT requires a 1-byte VAL_T (u8) — mirrors C++ CHECK_EQ(sizeof(VAL_T), 1)"
            );
            let len = ((num_data + 1) / 2) as usize;
            data_.resize(len, T::ZERO);
            buf_.resize(len, T::ZERO);
        } else {
            data_.resize(num_data as usize, T::ZERO);
        }
        DenseBin {
            num_data_: num_data,
            data_,
            buf_,
        }
    }

    /// C++ `inline VAL_T data(data_size_t idx)` (`dense_bin.hpp:559-565`).
    ///
    /// 4-bit unpacks the nibble: `(data_[idx>>1] >> ((idx&1)<<2)) & 0xf`.
    pub fn data(&self, idx: i32) -> u32 {
        if IS_4BIT {
            let byte = self.data_[(idx >> 1) as usize].to_u32();
            (byte >> ((idx & 1) << 2)) & 0xf
        } else {
            self.data_[idx as usize].to_u32()
        }
    }

    /// Raw `data_` bytes for the storage-layout golden tests. For non-`u8`
    /// `VAL_T` this re-views the element vector as its little-endian byte image,
    /// which is exactly the C++ `data_.data()` byte layout the histogram path
    /// reads (and the C++ harness dumps).
    pub fn raw_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(self.data_.len() * std::mem::size_of::<T>());
        for v in &self.data_ {
            out.extend_from_slice(&v.to_le_bytes());
        }
        out
    }
}

impl<T: BinValue, const IS_4BIT: bool> Bin for DenseBin<T, IS_4BIT> {
    /// C++ `void Push(int, data_size_t idx, uint32_t value)`
    /// (`dense_bin.hpp:69-82`).
    fn push(&mut self, idx: i32, value: u32) {
        if IS_4BIT {
            let i1 = (idx >> 1) as usize;
            let i2 = (idx & 1) << 2;
            let v = T::from_u32((value & 0xff) << i2);
            if i2 == 0 {
                self.data_[i1] = v;
            } else {
                self.buf_[i1] = v;
            }
        } else {
            self.data_[idx as usize] = T::from_u32(value);
        }
    }

    /// C++ `void FinishLoad()` (`dense_bin.hpp:510-521`).
    ///
    /// 4-bit OR-merges `data_[i] |= buf_[i]` over `(num_data + 1) / 2` cells then
    /// clears `buf_`; non-4-bit is a no-op.
    fn finish_load(&mut self) {
        if IS_4BIT {
            if self.buf_.is_empty() {
                return;
            }
            let len = ((self.num_data_ + 1) / 2) as usize;
            for i in 0..len {
                let merged = self.data_[i].to_u32() | self.buf_[i].to_u32();
                self.data_[i] = T::from_u32(merged);
            }
            self.buf_.clear();
        }
    }

    fn data(&self, idx: i32) -> u32 {
        DenseBin::data(self, idx)
    }

    fn num_data(&self) -> i32 {
        self.num_data_
    }

    fn raw_bytes(&self) -> Vec<u8> {
        DenseBin::raw_bytes(self)
    }

    fn sparse_deltas(&self) -> Option<&[u8]> {
        None
    }

    fn sparse_vals_bytes(&self) -> Option<Vec<u8>> {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dense_u8_push_and_read_roundtrip() {
        let mut b: DenseBin<u8, false> = DenseBin::new(5);
        for (idx, v) in [(0u32, 1u32), (1, 2), (2, 3), (3, 4), (4, 5)] {
            b.push(idx as i32, v);
        }
        Bin::finish_load(&mut b);
        for idx in 0..5 {
            assert_eq!(b.data(idx), (idx + 1) as u32);
        }
        // raw bytes are exactly the pushed values for the u8 path.
        assert_eq!(Bin::raw_bytes(&b), vec![1, 2, 3, 4, 5]);
    }

    #[test]
    fn four_bit_packs_two_bins_per_byte_even_and_odd() {
        // Prove the (idx&1)<<2 packing: even idx -> low nibble, odd idx -> high
        // nibble of the same byte. data(idx) must round-trip both.
        let mut b: DenseBin<u8, true> = DenseBin::new(4);
        b.push(0, 0x3); // even -> low nibble of byte 0
        b.push(1, 0x5); // odd  -> high nibble of byte 0
        b.push(2, 0xA); // even -> low nibble of byte 1
        b.push(3, 0xC); // odd  -> high nibble of byte 1
        // Before finish_load the odd nibbles live in buf_, so the merged read is
        // only valid after finish_load (mirrors C++ FinishLoad OR-merge).
        Bin::finish_load(&mut b);
        assert_eq!(b.data(0), 0x3, "even idx 0 -> low nibble");
        assert_eq!(b.data(1), 0x5, "odd idx 1 -> high nibble");
        assert_eq!(b.data(2), 0xA, "even idx 2 -> low nibble");
        assert_eq!(b.data(3), 0xC, "odd idx 3 -> high nibble");
        // raw byte layout: byte 0 = 0x53 (high=5, low=3), byte 1 = 0xCA.
        assert_eq!(Bin::raw_bytes(&b), vec![0x53, 0xCA]);
    }

    #[test]
    fn four_bit_odd_count_sizes_to_half_rounded_up() {
        // 3 rows -> (3+1)/2 = 2 bytes; last byte holds only a low nibble.
        let mut b: DenseBin<u8, true> = DenseBin::new(3);
        b.push(0, 0x1);
        b.push(1, 0x2);
        b.push(2, 0x7);
        Bin::finish_load(&mut b);
        assert_eq!(Bin::raw_bytes(&b).len(), 2);
        assert_eq!(b.data(0), 0x1);
        assert_eq!(b.data(1), 0x2);
        assert_eq!(b.data(2), 0x7);
        assert_eq!(Bin::raw_bytes(&b), vec![0x21, 0x07]);
    }

    #[test]
    fn dense_u16_raw_bytes_are_little_endian() {
        let mut b: DenseBin<u16, false> = DenseBin::new(2);
        b.push(0, 0x0102);
        b.push(1, 0x0304);
        Bin::finish_load(&mut b);
        // little-endian: 0x0102 -> [0x02, 0x01]; 0x0304 -> [0x04, 0x03].
        assert_eq!(Bin::raw_bytes(&b), vec![0x02, 0x01, 0x04, 0x03]);
    }
}
