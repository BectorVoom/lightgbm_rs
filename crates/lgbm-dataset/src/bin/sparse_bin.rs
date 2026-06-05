//! Bit-for-bit port of `SparseBin<VAL_T>` delta-encoded storage from
//! `LightGBM/src/io/sparse_bin.hpp` (`Push` 92-97, `FinishLoad`/`LoadFromPair`
//! 598-659, `GetFastIndex` 661-687, `NextNonzero` 392-400).
//!
//! # Layout notes (load-bearing)
//!
//! A sparse column stores only nonzero bins as a run-length-delta stream:
//!
//! - `push(idx, value)` buffers `(idx, value)` only when `value != 0`.
//! - `finish_load` concatenates the push buffer (single-threaded here, so one
//!   buffer), `std::sort`s by data index (non-stable is fine because indices are
//!   unique after the `cur_delta == 0` skip below), then delta-encodes:
//!   - `cur_delta = idx - last_idx`; if `i > 0 && cur_delta == 0` skip (disallow
//!     two values in one row);
//!   - while `cur_delta >= 256` emit a `(255, 0)` run and subtract 255;
//!   - emit `(cur_delta as u8, bin)`; advance `last_idx`.
//!   - a trailing `deltas_.push(0)` terminates the stream ("avoid out of range").
//! - `num_vals_ = vals_.len()`.
//! - `GetFastIndex` builds a power-of-two-strided `(i_delta, cur_pos)` lookup so
//!   random access is O(stride). It is a derived acceleration table but its
//!   layout is golden-tested for byte parity.
//!
//! `kNumFastIndex = 64` (`sparse_bin.hpp:25`).

use super::{Bin, BinValue};

/// C++ `const size_t kNumFastIndex = 64;` (`sparse_bin.hpp:25`).
const K_NUM_FAST_INDEX: i32 = 64;

/// C++ `SparseBin<VAL_T>` (`sparse_bin.hpp:73`). Delta-encoded sparse columnar
/// store: only nonzero bins are kept, addressed by a run-length delta stream
/// plus a strided fast-index.
#[derive(Debug)]
pub struct SparseBin<T: BinValue> {
    /// C++ `data_size_t num_data_`.
    num_data_: i32,
    /// Single push buffer (single-threaded core, D-03). C++ keeps one buffer per
    /// OMP thread; with `num_threads = 1` that is a single buffer.
    push_buffer_: Vec<(i32, T)>,
    /// C++ `std::vector<uint8_t> deltas_`.
    deltas_: Vec<u8>,
    /// C++ `std::vector<VAL_T> vals_`.
    vals_: Vec<T>,
    /// C++ `data_size_t num_vals_`.
    num_vals_: i32,
    /// C++ `std::vector<std::pair<data_size_t, data_size_t>> fast_index_`.
    fast_index_: Vec<(i32, i32)>,
    /// C++ `data_size_t fast_index_shift_`.
    fast_index_shift_: i32,
}

impl<T: BinValue> SparseBin<T> {
    /// C++ `SparseBin(data_size_t num_data)` (`sparse_bin.hpp:77-80`).
    pub fn new(num_data: i32) -> Self {
        SparseBin {
            num_data_: num_data,
            push_buffer_: Vec::new(),
            deltas_: Vec::new(),
            vals_: Vec::new(),
            num_vals_: 0,
            fast_index_: Vec::new(),
            fast_index_shift_: 0,
        }
    }

    /// C++ `void LoadFromPair(...)` (`sparse_bin.hpp:624-659`) — delta-encode the
    /// sorted `(idx, bin)` pairs into `deltas_`/`vals_`.
    fn load_from_pair(&mut self, idx_val_pairs: &[(i32, T)]) {
        self.deltas_.clear();
        self.vals_.clear();
        let mut last_idx: i32 = 0;
        for (i, &(cur_idx, bin)) in idx_val_pairs.iter().enumerate() {
            let mut cur_delta = cur_idx - last_idx;
            // disallow the multi-val in one row
            if i > 0 && cur_delta == 0 {
                continue;
            }
            while cur_delta >= 256 {
                self.deltas_.push(255);
                self.vals_.push(T::ZERO);
                cur_delta -= 255;
            }
            self.deltas_.push(cur_delta as u8);
            self.vals_.push(bin);
            last_idx = cur_idx;
        }
        // avoid out of range
        self.deltas_.push(0);
        self.num_vals_ = self.vals_.len() as i32;
        // generate fast index
        self.get_fast_index();
    }

    /// C++ `inline bool NextNonzero(data_size_t* i_delta, data_size_t* cur_pos)`
    /// (`sparse_bin.hpp:392-400`).
    fn next_nonzero(&self, i_delta: &mut i32, cur_pos: &mut i32) -> bool {
        *i_delta += 1;
        *cur_pos += self.deltas_[*i_delta as usize] as i32;
        if *i_delta < self.num_vals_ {
            true
        } else {
            *cur_pos = self.num_data_;
            false
        }
    }

    /// C++ `void GetFastIndex()` (`sparse_bin.hpp:661-687`).
    fn get_fast_index(&mut self) {
        self.fast_index_.clear();
        // get shift cnt
        let mod_size = (self.num_data_ + K_NUM_FAST_INDEX - 1) / K_NUM_FAST_INDEX;
        let mut pow2_mod_size: i32 = 1;
        self.fast_index_shift_ = 0;
        while pow2_mod_size < mod_size {
            pow2_mod_size <<= 1;
            self.fast_index_shift_ += 1;
        }
        // build fast index
        let mut i_delta: i32 = -1;
        let mut cur_pos: i32 = 0;
        let mut next_threshold: i32 = 0;
        while self.next_nonzero(&mut i_delta, &mut cur_pos) {
            while next_threshold <= cur_pos {
                self.fast_index_.push((i_delta, cur_pos));
                next_threshold += pow2_mod_size;
            }
        }
        // avoid out of range
        while next_threshold < self.num_data_ {
            self.fast_index_.push((self.num_vals_ - 1, cur_pos));
            next_threshold += pow2_mod_size;
        }
    }

    /// `deltas_` accessor for the storage-layout golden tests.
    pub fn deltas(&self) -> &[u8] {
        &self.deltas_
    }

    /// `vals_` accessor for the storage-layout golden tests.
    pub fn vals(&self) -> &[T] {
        &self.vals_
    }

    /// `num_vals_` accessor.
    pub fn num_vals(&self) -> i32 {
        self.num_vals_
    }

    /// `fast_index_` accessor for the golden tests.
    pub fn fast_index(&self) -> &[(i32, i32)] {
        &self.fast_index_
    }

    /// `fast_index_shift_` accessor.
    pub fn fast_index_shift(&self) -> i32 {
        self.fast_index_shift_
    }

    /// `vals_` rendered as the little-endian byte image C++ would write — for
    /// byte-exact golden comparison across `VAL_T` widths.
    pub fn vals_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(self.vals_.len() * std::mem::size_of::<T>());
        for v in &self.vals_ {
            out.extend_from_slice(&v.to_le_bytes());
        }
        out
    }
}

impl<T: BinValue> Bin for SparseBin<T> {
    /// C++ `void Push(int tid, data_size_t idx, uint32_t value)`
    /// (`sparse_bin.hpp:92-97`): buffer only nonzero values.
    fn push(&mut self, idx: i32, value: u32) {
        let cur_bin = T::from_u32(value);
        if cur_bin.to_u32() != 0 {
            self.push_buffer_.push((idx, cur_bin));
        }
    }

    /// C++ `void FinishLoad()` (`sparse_bin.hpp:598-622`): sort the push buffer
    /// by data index then delta-encode.
    fn finish_load(&mut self) {
        let mut pairs = std::mem::take(&mut self.push_buffer_);
        // sort by data index (non-stable OK — unique post cur_delta==0 skip).
        pairs.sort_by_key(|&(idx, _)| idx);
        self.load_from_pair(&pairs);
    }

    /// `data(idx)` for a sparse store walks the delta stream. Not on the hot path
    /// here (Phase 4 iterates), but provided for completeness / testing.
    fn data(&self, idx: i32) -> u32 {
        let mut i_delta: i32 = -1;
        let mut cur_pos: i32 = 0;
        loop {
            if !self.next_nonzero(&mut i_delta, &mut cur_pos) {
                return 0;
            }
            if cur_pos == idx {
                return self.vals_[i_delta as usize].to_u32();
            }
            if cur_pos > idx {
                return 0;
            }
        }
    }

    fn num_data(&self) -> i32 {
        self.num_data_
    }

    /// For a sparse bin the "raw bytes" are the delta stream — but golden tests
    /// compare `deltas_`/`vals_`/`fast_index` explicitly, so this returns the
    /// delta bytes for a uniform byte-comparison entry point.
    fn raw_bytes(&self) -> Vec<u8> {
        self.deltas_.clone()
    }

    fn sparse_deltas(&self) -> Option<&[u8]> {
        Some(&self.deltas_)
    }

    fn sparse_vals_bytes(&self) -> Option<Vec<u8>> {
        Some(self.vals_bytes())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sparse_skips_zeros_and_delta_encodes() {
        // rows: 0:0, 1:5, 2:0, 3:7  -> nonzeros at idx 1 (delta 1) and idx 3
        // (delta 2). Trailing 0 terminator. vals = [5, 7].
        let mut b: SparseBin<u8> = SparseBin::new(4);
        b.push(0, 0);
        b.push(1, 5);
        b.push(2, 0);
        b.push(3, 7);
        Bin::finish_load(&mut b);
        assert_eq!(b.vals(), &[5u8, 7u8]);
        assert_eq!(b.deltas(), &[1u8, 2u8, 0u8], "deltas with trailing 0");
        assert_eq!(b.num_vals(), 2);
        assert_eq!(b.data(1), 5);
        assert_eq!(b.data(3), 7);
        assert_eq!(b.data(0), 0);
    }

    #[test]
    fn sparse_large_gap_emits_255_run() {
        // A gap >= 256 must emit a (255, 0) run-length pair then the remainder.
        // idx 300 with last_idx 0: cur_delta=300 -> emit (255,0), cur_delta=45,
        // then (45, bin). So deltas = [255, 45, 0], vals = [0, 9].
        let mut b: SparseBin<u16> = SparseBin::new(400);
        b.push(300, 9);
        Bin::finish_load(&mut b);
        assert_eq!(b.deltas(), &[255u8, 45u8, 0u8], "255 run then remainder + trailing 0");
        assert_eq!(b.vals(), &[0u16, 9u16]);
        assert_eq!(b.num_vals(), 2);
    }

    #[test]
    fn sparse_unsorted_pushes_are_sorted_by_index() {
        let mut b: SparseBin<u8> = SparseBin::new(10);
        b.push(7, 3);
        b.push(2, 1);
        b.push(5, 2);
        Bin::finish_load(&mut b);
        // sorted indices: 2,5,7 -> deltas 2,3,2 + trailing 0; vals 1,2,3.
        assert_eq!(b.deltas(), &[2u8, 3u8, 2u8, 0u8]);
        assert_eq!(b.vals(), &[1u8, 2u8, 3u8]);
    }

    #[test]
    fn sparse_fast_index_is_built() {
        let mut b: SparseBin<u8> = SparseBin::new(200);
        b.push(10, 1);
        b.push(100, 2);
        b.push(150, 3);
        Bin::finish_load(&mut b);
        // fast_index must cover the strided positions without panicking and be
        // non-empty for a 200-row column.
        assert!(!b.fast_index().is_empty());
    }
}
