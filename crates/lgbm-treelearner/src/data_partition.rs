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
use lgbm_compute::BinColumn;
// `ComputeClient` is re-exported by the compute seam (CMP-01) so this crate names
// the Backend ops' client argument without ever depending on `cubecl` directly.
use lgbm_compute::ComputeClientReexport as ComputeClient;
use rayon::prelude::*;

/// Static chunk size for the parallel reorder, mirroring C++ `ParallelPartitionRunner`'s
/// `schedule(static, 512)` (`data_partition.hpp`). A contiguous ascending partition of the
/// leaf's row slice so chunk order — and therefore the final `[all left | all right]`
/// concatenation — is deterministic and byte-identical to the serial stable two-pass gather.
const PAR_SPLIT_CHUNK: usize = 512;

/// The leaf-row count at/above which `DataPartition::split` reorders the leaf's rows in
/// parallel (rayon static-chunk + exclusive prefix-sum scatter). Below this, the EXISTING
/// serial Backend path runs verbatim so small/medium leaves never pay rayon fork/join cost
/// (the spike-005 regression class; C++ guards the analogous parallel runner with
/// `if num_data_ >= 1024`). Override via `LGBM_PAR_SPLIT_THRESHOLD`; default 16384 — the same
/// default + idiom as `par_build_threshold()`/`LGBM_PAR_THRESHOLD` in `lgbm-compute`.
fn par_split_threshold() -> usize {
    std::env::var("LGBM_PAR_SPLIT_THRESHOLD")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(16384)
}

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
        feature_bins: &BinColumn,
        num_bin: u32,
        min_bin: u32,
        max_bin: u32,
        threshold: u32,
        most_freq_bin: u32,
    ) -> Result<(i32, i32), ComputeError> {
        let leaf_u = leaf as usize;
        let begin = self.leaf_begin[leaf_u] as usize;
        let count = self.leaf_count[leaf_u] as usize;

        // Leaf-row gate: large leaves take the deterministic rayon reorder; small/medium
        // leaves run the EXISTING serial Backend path verbatim (the spike-005 fork/join
        // regression class is avoided by construction). The two paths are bit-identical.
        let split_point = if count >= par_split_threshold() {
            self.split_numeric_parallel(
                begin,
                count,
                feature_bins,
                num_bin,
                min_bin,
                max_bin,
                threshold,
                most_freq_bin,
            )?
        } else {
            // --- Serial path (unchanged behavior below the threshold) ---
            // Gather the leaf's per-row bins in the current `indices_` order so the
            // Backend op's stable reorder is RELATIVE to this leaf's row order. The
            // per-row bin READ widens via the accessor; the partition LOGIC is
            // unchanged.
            let leaf_rows: Vec<u32> = self.indices[begin..begin + count].to_vec();
            let leaf_feature_bins: Vec<u32> = leaf_rows
                .iter()
                .map(|&row| feature_bins.bin(row as usize))
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
            split_point
        };

        let left_count = split_point as i32;
        let right_count = count as i32 - left_count;

        // left_leaf keeps `leaf`'s begin; right_leaf starts after the left rows.
        self.leaf_count[leaf_u] = left_count;
        let right_u = right_leaf as usize;
        self.leaf_begin[right_u] = begin as i32 + left_count;
        self.leaf_count[right_u] = right_count;

        Ok((left_count, right_count))
    }

    /// Deterministic rayon static-chunk + exclusive prefix-sum reorder of the leaf's
    /// row slice `indices_[begin..begin+count]`, byte-identical to the serial stable
    /// two-pass gather. Mirrors C++ `ParallelPartitionRunner` (`data_partition.hpp`):
    /// static-chunk the rows, gather each chunk's left/right rows in ascending within-chunk
    /// order, then prefix-sum-concatenate (`[all left | all right]`).
    ///
    /// Returns `split_point` (= total left-row count = the serial `split_point`).
    ///
    /// # Errors
    /// Surfaces the SAME [`ComputeError`] for the SAME lowest-index offending row as the
    /// serial Backend op (V5): the per-row bin bounds are validated BEFORE the parallel
    /// region, walking rows in ascending leaf-relative order, so the error is independent
    /// of thread scheduling (threat T-ia0-02).
    #[allow(clippy::too_many_arguments)]
    fn split_numeric_parallel(
        &mut self,
        begin: usize,
        count: usize,
        feature_bins: &BinColumn,
        num_bin: u32,
        min_bin: u32,
        max_bin: u32,
        threshold: u32,
        most_freq_bin: u32,
    ) -> Result<usize, ComputeError> {
        // --- V5 boundary validation, identical to `data_partition_cpu_native` ---
        // num_bin > 0 and threshold < num_bin first (the op's order), then the per-row
        // bin check walking rows ASCENDING so the lowest-index offending row surfaces
        // regardless of thread scheduling (threat T-ia0-02).
        if num_bin == 0 {
            return Err(ComputeError::Runtime {
                detail: "data_partition: num_bin must be > 0".to_string(),
            });
        }
        if threshold >= num_bin {
            return Err(ComputeError::Runtime {
                detail: format!("data_partition: threshold {threshold} >= num_bin {num_bin}"),
            });
        }
        // The leaf-relative row order; the per-row bin is read for the GLOBAL row id it
        // currently holds. Validation walks this slice ascending (leaf-relative index =
        // the `row` reported by the serial op, which sees the gathered leaf bins).
        let leaf_rows = &self.indices[begin..begin + count];
        for (row, &global) in leaf_rows.iter().enumerate() {
            let b = feature_bins.bin(global as usize);
            if b >= num_bin {
                return Err(ComputeError::BinIndexOutOfRange {
                    row,
                    bin: b,
                    num_bin,
                });
            }
        }

        // Routing decision (dense_bin.hpp:322-365), integer-only — identical to
        // `data_partition_cpu_native`: th = threshold + min_bin (−1 if most_freq_bin == 0);
        // out-of-[min,max] rows take the default direction (`most_freq_bin > threshold` ⇒ gt).
        let min_b = min_bin as i32;
        let max_b = max_bin as i32;
        let thr = threshold as i32;
        let mut th = thr + min_b;
        if most_freq_bin == 0 {
            th -= 1;
        }
        let default_to_right = most_freq_bin as i32 > thr;
        let go_right = move |global: u32| -> bool {
            let bin = feature_bins.bin(global as usize) as i32;
            if bin < min_b || bin > max_b {
                default_to_right
            } else {
                bin > th
            }
        };

        // Phase 1: per static-chunk, gather this chunk's left (route==0) and right
        // (route==1) GLOBAL row ids, each in ascending within-chunk order. Chunks are a
        // contiguous ascending partition so their order is deterministic.
        let per_chunk: Vec<(Vec<u32>, Vec<u32>)> = leaf_rows
            .par_chunks(PAR_SPLIT_CHUNK)
            .map(|chunk| {
                let mut left: Vec<u32> = Vec::new();
                let mut right: Vec<u32> = Vec::new();
                for &global in chunk {
                    if go_right(global) {
                        right.push(global);
                    } else {
                        left.push(global);
                    }
                }
                (left, right)
            })
            .collect();

        // Phase 2: exclusive prefix-sum the per-chunk left/right counts to disjoint write
        // offsets. total_left = sum of chunk left-counts = the serial split_point. Left
        // rows occupy `[begin, begin+total_left)`, right rows `[begin+total_left, begin+count)`.
        let total_left: usize = per_chunk.iter().map(|(l, _)| l.len()).sum();

        let mut left_off = Vec::with_capacity(per_chunk.len());
        let mut right_off = Vec::with_capacity(per_chunk.len());
        let mut lacc = begin;
        let mut racc = begin + total_left;
        for (l, r) in &per_chunk {
            left_off.push(lacc);
            right_off.push(racc);
            lacc += l.len();
            racc += r.len();
        }

        // Phase 3: scatter each chunk's left/right ids into its prefix-summed disjoint
        // region of `indices_` in parallel (no atomics — chunks write non-overlapping
        // ranges). Because chunk order is ascending and within-chunk order is preserved,
        // the final `[all left | all right]` is byte-identical to the serial stable gather.
        //
        // SAFETY: each chunk writes exactly `[left_off[c], left_off[c]+l.len())` and
        // `[right_off[c], right_off[c]+r.len())`; the prefix sums make these regions
        // pairwise disjoint and all within `[begin, begin+count)`, so the parallel raw
        // writes never alias. The slice base pointer + length bound the writes.
        let base: *mut u32 = self.indices.as_mut_ptr();
        let indices_len = self.indices.len();
        debug_assert!(begin + count <= indices_len);
        struct SendPtr(*mut u32);
        // SAFETY: the disjoint, in-bounds offsets above guarantee no aliasing across threads.
        unsafe impl Sync for SendPtr {}
        let base = SendPtr(base);
        let base_ref = &base;
        per_chunk
            .par_iter()
            .zip(left_off.par_iter())
            .zip(right_off.par_iter())
            .for_each(|(((l, r), &lo), &ro)| {
                let p = base_ref.0;
                for (k, &id) in l.iter().enumerate() {
                    // SAFETY: lo+k < begin+total_left <= begin+count <= indices_len, disjoint.
                    unsafe {
                        *p.add(lo + k) = id;
                    }
                }
                for (k, &id) in r.iter().enumerate() {
                    // SAFETY: ro+k < begin+count <= indices_len, disjoint from all left + other chunks.
                    unsafe {
                        *p.add(ro + k) = id;
                    }
                }
            });

        Ok(total_left)
    }

    /// `DataPartition::Split` for a CATEGORICAL split — routes `leaf`'s rows by the
    /// REAL category bitset into `left_leaf` (kept as `leaf`, the in-bitset side)
    /// and `right_leaf` (the default/out-of-bitset side).
    ///
    /// This is the row-partition analog of the predict-side `Tree::CategoricalDecision`
    /// (`tree.h:374-390`): a row whose feature BIN maps to a CATEGORY VALUE that is
    /// in the real bitset routes LEFT; everything else (incl. the NaN dummy bin 0,
    /// whose category is `-1` / negative) routes RIGHT. Routing on the category
    /// value through `bin_to_category` is provably equivalent to the C++
    /// `DenseBin::SplitCategoricalInner` inner-bin-bitset routing (which is just a
    /// performance encoding of the same decision) and is GUARANTEED consistent with
    /// the model's serialized real bitset + the predict path — sidestepping the
    /// bin↔offset bookkeeping entirely.
    ///
    /// `cat_bitset_real` is `Common::ConstructBitset` over the winning CATEGORY
    /// VALUES (the same bitset the tree serializes as `cat_threshold`).
    /// `bin_to_category` maps a feature bin to its category value (`bin_2_categorical_`).
    ///
    /// Returns `(left_count, right_count)`.
    pub fn split_categorical(
        &mut self,
        leaf: i32,
        right_leaf: i32,
        feature_bins: &BinColumn,
        cat_bitset_real: &[u32],
        bin_to_category: &[i32],
    ) -> (i32, i32) {
        let leaf_u = leaf as usize;
        let begin = self.leaf_begin[leaf_u] as usize;
        let count = self.leaf_count[leaf_u] as usize;
        let leaf_rows: Vec<u32> = self.indices[begin..begin + count].to_vec();

        let mut lte: Vec<u32> = Vec::with_capacity(count); // left / in-bitset
        let mut gt: Vec<u32> = Vec::with_capacity(count); // right / default
        for &row in &leaf_rows {
            let bin = feature_bins.bin(row as usize);
            let cat = bin_to_category
                .get(bin as usize)
                .copied()
                .unwrap_or(bin as i32);
            // Negative category (the NaN dummy at bin 0) always routes RIGHT, exactly
            // as CategoricalDecision routes `int_fval < 0` / NaN to the right child.
            if cat >= 0 && find_in_bitset(cat_bitset_real, cat as u32) {
                lte.push(row);
            } else {
                gt.push(row);
            }
        }

        let left_count = lte.len() as i32;
        let right_count = gt.len() as i32;

        // Write [left | right] back into the leaf's slice (stable order).
        let mut w = begin;
        for &row in &lte {
            self.indices[w] = row;
            w += 1;
        }
        for &row in &gt {
            self.indices[w] = row;
            w += 1;
        }

        self.leaf_count[leaf_u] = left_count;
        let right_u = right_leaf as usize;
        self.leaf_begin[right_u] = begin as i32 + left_count;
        self.leaf_count[right_u] = right_count;

        (left_count, right_count)
    }

    /// Total row count (`num_data_`).
    pub fn num_data(&self) -> i32 {
        self.num_data
    }
}

/// `Common::FindInBitset(bits, n, pos)` (`common.h:836`): `i1 = pos/32; if i1 >= n
/// return false; (bits[i1] >> (pos%32)) & 1`.
#[inline]
fn find_in_bitset(bits: &[u32], pos: u32) -> bool {
    let i1 = (pos / 32) as usize;
    if i1 >= bits.len() {
        return false;
    }
    (bits[i1] >> (pos % 32)) & 1 == 1
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
        let feature_bins = BinColumn::new(vec![1u32, 5, 3, 7, 0, 4, 2, 6], 8);
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
        let feature_bins = BinColumn::new(vec![1u32, 5, 3, 7, 0, 4, 2, 6], 8);
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

    /// Bit-exact merge gate (quick 260622-ia0): the rayon static-chunk + exclusive
    /// prefix-sum parallel reorder MUST write a byte-identical leaf slice to the serial
    /// Backend path. Force BOTH paths on the SAME synthetic large leaf via the
    /// `LGBM_PAR_SPLIT_THRESHOLD` env knob (0 = force-parallel, usize::MAX = force-serial)
    /// and assert the `indices_in_leaf` slices are equal. Mirrors the shape of
    /// `build_histograms_parallel_equals_serial` (scattered rows, deterministic
    /// hash-generated bins). The partition order is load-bearing for the histogram
    /// subtraction trick, so any byte drift here would break the f64 anchor.
    #[test]
    fn split_parallel_equals_serial() {
        // Run a leaf large enough to cross the parallel chunk count, with scattered
        // global row ids, scattered bins, and randomized-but-fixed split params.
        let rows: u32 = 5000;
        let num_bin: u32 = 257; // forces BinColumn::U16 — exercises a non-u8 width
        let min_bin: u32 = 3;
        let max_bin: u32 = 250;
        let threshold: u32 = 97;
        let most_freq_bin: u32 = 0; // exercises the `th -= 1` branch

        // Deterministic hash-generated bins in [0, num_bin), per global row.
        let bins: Vec<u32> = (0..rows)
            .map(|r| {
                let h = (r as u64).wrapping_mul(2_654_435_761).wrapping_add(11);
                (h % num_bin as u64) as u32
            })
            .collect();
        let feature_bins = BinColumn::new(bins, num_bin);

        // Scattered (non-identity) leaf row order: seed leaf 0 to a permutation of all
        // rows so the per-row order the reorder must preserve is non-trivial. We build a
        // DataPartition then overwrite leaf 0's indices with a scattered permutation.
        let scattered: Vec<u32> = {
            // Fisher-Yates-ish deterministic shuffle via index hashing into a Vec.
            let mut v: Vec<u32> = (0..rows).collect();
            for i in (1..rows as usize).rev() {
                let j = ((i as u64).wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1)
                    % (i as u64 + 1)) as usize;
                v.swap(i, j);
            }
            v
        };

        // SERIAL path: the existing Backend `data_partition` gather, driven exactly as the
        // serial branch of `split` would (count below the default threshold). We invoke it
        // directly (not via the env gate) to avoid mutating the process-global env, which
        // would race with the other lib tests running in parallel.
        let serial: Vec<u32> = {
            let backend = CpuBackend;
            let client = cpu_client();
            let mut dp = DataPartition::new(rows as i32, 4);
            for (slot, &row) in scattered.iter().enumerate() {
                dp.indices[slot] = row;
            }
            let begin = 0usize;
            let count = rows as usize;
            let leaf_rows: Vec<u32> = dp.indices[begin..begin + count].to_vec();
            let leaf_feature_bins: Vec<u32> = leaf_rows
                .iter()
                .map(|&row| feature_bins.bin(row as usize))
                .collect();
            let (reordered_local, _split_point) = backend
                .data_partition(
                    &client,
                    &leaf_feature_bins,
                    num_bin,
                    min_bin,
                    max_bin,
                    threshold,
                    most_freq_bin,
                )
                .expect("serial partition ok");
            for (slot, &local_pos) in reordered_local.iter().enumerate() {
                dp.indices[begin + slot] = leaf_rows[local_pos as usize];
            }
            dp.indices[0..rows as usize].to_vec()
        };

        // PARALLEL path: the rayon static-chunk + exclusive prefix-sum reorder, invoked
        // directly on the same scattered leaf.
        let parallel: Vec<u32> = {
            let mut dp = DataPartition::new(rows as i32, 4);
            for (slot, &row) in scattered.iter().enumerate() {
                dp.indices[slot] = row;
            }
            dp.split_numeric_parallel(
                0,
                rows as usize,
                &feature_bins,
                num_bin,
                min_bin,
                max_bin,
                threshold,
                most_freq_bin,
            )
            .expect("parallel partition ok");
            dp.indices[0..rows as usize].to_vec()
        };

        assert_eq!(
            serial, parallel,
            "parallel reorder is NOT byte-identical to the serial stable gather"
        );
    }
}
