//! `DataPartition` — leaf row-range bookkeeping wrapping `Backend::data_partition`.
//!
//! Faithful transcription of `DataPartition` (`data_partition.hpp`, commit
//! 195c26fc, VERSION 4.6.0.99): the `indices_` permutation of all row ids grouped
//! by leaf, plus the per-leaf `leaf_begin_` / `leaf_count_` ranges. The actual
//! per-row left/right routing math lives in the `Backend::data_partition`
//! op (the `DenseBin::SplitInner` `MissingType::None` transcription); this module
//! owns ONLY the leaf-range bookkeeping (see `lib.rs:115-117`: the learner
//! owns `leaf_begin_`/`leaf_count_` bookkeeping; this op returns only the
//! partition. No new partition numerics.
//!
//! ## C++ correspondence
//! - `indices_`     ↔ `std::vector<data_size_t> indices_` — all row ids, grouped
//!   contiguously by leaf.
//! - `leaf_begin_`  ↔ `std::vector<data_size_t> leaf_begin_` — start offset of
//!   each leaf's slice in `indices_`.
//! - `leaf_count_`  ↔ `std::vector<data_size_t> leaf_count_` — row count per leaf
//!   (the GLOBAL count; no bagging currently, so global == local).
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

/// Minimum leaf-row count to engage the block-parallel
/// two-buffer partition in `split_fused_host`. Below this the serial two-buffer
/// path runs (no fork/join) — the win lives in the first few LARGE splits, so a
/// high default captures ~all of it with ~no small-leaf fork/join risk. Read ONCE
/// via `OnceLock` (env `LGBM_PAR_PARTITION_MIN`, default 65536). Set to
/// `18446744073709551615` (= `usize::MAX`) to FORCE the serial path — a bit-exact
/// serial-vs-parallel toggle (no parallel split ever engages).
///
/// Default = **65536**: a lower threshold lets a mid-size root leaf (e.g. 50k rows)
/// fork/join and its scheduling overhead leaks into the already-parallel
/// histogram build (shared rayon pool), causing a small wall-clock regression.
/// 65536 keeps such leaves serial while still capturing the parallel win on large
/// (e.g. ~1M-row) leaves, where the first several splits clear the threshold and
/// engage the block-parallel path.
fn par_partition_min() -> usize {
    use std::sync::OnceLock;
    static M: OnceLock<usize> = OnceLock::new();
    *M.get_or_init(|| {
        std::env::var("LGBM_PAR_PARTITION_MIN")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(65536)
    })
}

/// Per-block pass-1 result in the parallel partition: `(left_count, right_count,
/// lowest_bad_index_within_block)`. The optional carries the block's first (lowest
/// local) out-of-range `(leaf_position, bin)`; the global minimum is taken across blocks.
type BlockResult = (usize, usize, Option<(usize, u32)>);

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
    /// Two-buffer scratch ↔ C++ `ParallelPartitionRunner::left_`
    /// (`threading.h:91`): persistent left-side row-id scratch, sized `num_data`,
    /// allocated ONCE and reused across every `split_fused_host` call (no per-split
    /// `Vec` alloc). Pass-1 scatters left rows ascending into `[0..left_count)`.
    part_left: Vec<u32>,
    /// Two-buffer scratch ↔ C++ `ParallelPartitionRunner::right_`:
    /// persistent right-side row-id scratch, sized `num_data`, reused across splits.
    /// Pass-1 scatters right rows ascending into `[0..right_count)`.
    part_right: Vec<u32>,
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
            // Two-buffer scratch sized to hold every row (a single leaf can own all
            // `num_data` rows — the root — so each side must be able to hold `n`).
            part_left: vec![0u32; n],
            part_right: vec![0u32; n],
        }
    }

    /// Reconstruct a `DataPartition` from the
    /// lower-crate [`lgbm_dataset::LeafPartitionLayout`] payload `P` returned by
    /// `Backend::grow_tree_on_device`. A thin field-move — the payload mirrors the
    /// four fields `DataPartition` wraps (identical shapes), so this never depends
    /// on `lgbm-compute` (acyclic: treelearner already names both DataPartition and
    /// lgbm_dataset). The seam currently returns `Ok(None)`, so this is reachable
    /// but not exercised at runtime yet — kept minimal until a kernel is wired up.
    pub fn from_payload(p: lgbm_dataset::LeafPartitionLayout) -> Self {
        let n = p.num_data.max(0) as usize;
        Self {
            num_data: p.num_data,
            indices: p.indices,
            leaf_begin: p.leaf_begin,
            leaf_count: p.leaf_count,
            // Persistent two-buffer scratch, sized to the payload's row count.
            part_left: vec![0u32; n],
            part_right: vec![0u32; n],
        }
    }

    /// C++ `DataPartition::leaf_count(leaf)` — the GLOBAL row count in `leaf`.
    /// Drives the smaller-child selection.
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
    ///
    /// PARALLELISM: the host partition is a **two-buffer, all-passes
    /// block-parallel** reorder (`split_fused_host` → `partition_parallel`), gated
    /// on `LGBM_PAR_PARTITION_MIN`.
    ///
    /// BUILD and PARTITION are SEQUENTIAL phases per split, so a properly-gated
    /// parallel partition does not truly contend with the already-parallel
    /// histogram build. The win decomposes as: two-buffer (drop the third
    /// copyback pass) reduces single-thread TRAFFIC, and all-passes
    /// block-parallel adds PARALLELISM on top — gated on
    /// `LGBM_PAR_PARTITION_MIN` (default 65536) so small leaves stay serial.
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
        na_as_missing: bool,
        default_left: bool,
    ) -> Result<(i32, i32), ComputeError> {
        let leaf_u = leaf as usize;
        let begin = self.leaf_begin[leaf_u] as usize;
        let count = self.leaf_count[leaf_u] as usize;

        let (left_count, right_count) = if backend.prefers_host_partition() {
            // CpuBackend host anchor: fused u8-route path, IN PLACE on
            // `self.indices[begin..begin+count]`. ONE random gather + a ¼-width u8
            // route scratch + ONE u32 scatter — no leaf_rows clone, no u32-widened
            // leaf_feature_bins, no local→row remap. Byte-identical [left | right]
            // order to the materialize-then-op path below. `na_as_missing`
            // (T-G4-2, SPEC-G4-2) additionally routes the NA sentinel bin
            // (`num_bin-1`) per `default_left` — see `split_fused_host_impl`.
            self.split_fused_host(
                begin,
                count,
                feature_bins,
                num_bin,
                min_bin,
                max_bin,
                threshold,
                most_freq_bin,
                na_as_missing,
                default_left,
            )?
        } else {
            // RocmBackend / any device backend: route the leaf's rows ON-DEVICE.
            // Re-gather the leaf's bins at NATIVE width
            // (`BinColumn::gather` preserves u8/u16/u32) and route via the additive
            // `data_partition_native`. The CpuBackend default widens + delegates (so the
            // non-fused cpu path stays byte-unchanged); RocmBackend overrides it to
            // upload count×native-width bytes (4× fewer on all-u8 data) — bit-exact to
            // the prior u32 widen (value-identical routing). The fused host path above
            // (`prefers_host_partition`) is untouched.
            //
            // `na_as_missing` (T-G4-2) is NOT yet threaded into `data_partition_native`
            // — only the CPU host `split_fused_host` arm above routes the NA sentinel
            // bin by `default_left`. Silently falling through here would route those
            // rows by the ordinary numeric threshold instead, producing a WRONG model
            // on any backend that does not prefer host partitioning (ROCm/CUDA/WGPU)
            // with no error at all. Reject explicitly rather than mis-compute (V5).
            if na_as_missing {
                return Err(ComputeError::Runtime {
                    detail: "data_partition: na_as_missing NaN-sentinel-bin routing is only \
                             implemented on the CpuBackend host partition path \
                             (prefers_host_partition); device backends (ROCm/CUDA/WGPU) do \
                             not yet thread default_left into data_partition_native"
                        .to_string(),
                });
            }
            let leaf_rows: Vec<u32> = self.indices[begin..begin + count].to_vec();
            let leaf_bins = feature_bins.gather(&leaf_rows);

            // The Backend op owns the SplitInner routing + stable two-pass gather.
            let (reordered_local, split_point) = backend.data_partition_native(
                client,
                &leaf_bins,
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

            let left_count = split_point as i32;
            let right_count = count as i32 - left_count;
            (left_count, right_count)
        };

        // left_leaf keeps `leaf`'s begin; right_leaf starts after the left rows.
        self.leaf_count[leaf_u] = left_count;
        let right_u = right_leaf as usize;
        self.leaf_begin[right_u] = begin as i32 + left_count;
        self.leaf_count[right_u] = right_count;

        Ok((left_count, right_count))
    }

    /// TWO-BUFFER host split (C++ `ParallelPartitionRunner<INDEX_T, true>`,
    /// `threading.h:91`), run IN PLACE on the leaf's slice `self.indices[begin..+count]`.
    /// Returns `(left_count, right_count)`.
    ///
    /// ## Two passes, not three
    /// A naive implementation needs THREE O(count) passes: (1) gather→route into a
    /// `route: Vec<u8>` scratch, (2) scatter row ids into a fresh `out: Vec<u32>`, (3)
    /// `copy_from_slice(out)` back into `self.indices`. This two-buffer design collapses
    /// that to TWO passes matching the C++ runner:
    /// - **pass 1** (`func`): gather each leaf row's bin ONCE, range-check, route via the
    ///   SAME `SplitInner` `MissingType::None` `go_right` decision as before (byte-for-byte
    ///   router math), and scatter the ROW ID directly into the persistent `part_left`
    ///   scratch (left rows, ascending) or `part_right` scratch (right rows, ascending).
    /// - **pass 2** (`copy_n`): copy the `part_left[..left_count]` run then the
    ///   `part_right[..right_count]` run into `self.indices[begin..begin+count]`.
    ///
    /// Eliminating the third pass (the fresh per-split `out: Vec<u32>` + its copyback) is
    /// the traffic win, BEFORE any parallelism. `part_left`/`part_right` are persistent
    /// members (sized `num_data`, reused across splits) — no per-split allocation.
    ///
    /// ## Stability (byte-identical to the serial 3-pass path)
    /// Pass-1 walks leaf positions in ascending order, appending each left row to
    /// `part_left` and each right row to `part_right`, so both scratch runs are in
    /// ascending original order; pass-2 lays left-run then right-run, giving
    /// `[left ascending | right ascending]` — the exact order the C++ f64 anchor and the
    /// dump-model sha256 gate encode.
    ///
    /// ## Error semantics (lowest-index, `self.indices` unmutated)
    /// Pass-1 writes ONLY into `part_left`/`part_right` scratch, never `self.indices`. The
    /// first out-of-range bin (in ascending leaf position) early-returns
    /// `BinIndexOutOfRange` before pass-2 runs, so `self.indices` is UNMUTATED on error —
    /// bit-identical to the pre-rewrite lowest-index semantics.
    #[allow(clippy::too_many_arguments)]
    fn split_fused_host(
        &mut self,
        begin: usize,
        count: usize,
        feature_bins: &BinColumn,
        num_bin: u32,
        min_bin: u32,
        max_bin: u32,
        threshold: u32,
        most_freq_bin: u32,
        na_as_missing: bool,
        default_left: bool,
    ) -> Result<(i32, i32), ComputeError> {
        // Production dispatch: read the leaf-row threshold ONCE (env
        // `LGBM_PAR_PARTITION_MIN`). Tests drive `split_fused_host_impl` directly to
        // exercise both arms in one process.
        self.split_fused_host_impl(
            begin,
            count,
            feature_bins,
            num_bin,
            min_bin,
            max_bin,
            threshold,
            most_freq_bin,
            na_as_missing,
            default_left,
            par_partition_min(),
        )
    }

    /// Two-buffer partition core, parameterized on the parallel threshold `par_min`
    /// so tests can force the serial arm (`par_min = usize::MAX`) or the parallel arm
    /// (`par_min = 0`) and prove them byte-identical in one process. When
    /// `count >= par_min` BOTH passes run block-parallel; otherwise the serial
    /// two-buffer path runs. Both arms are bit-exact `[left ascending | right ascending]`.
    #[allow(clippy::too_many_arguments)]
    fn split_fused_host_impl(
        &mut self,
        begin: usize,
        count: usize,
        feature_bins: &BinColumn,
        num_bin: u32,
        min_bin: u32,
        max_bin: u32,
        threshold: u32,
        most_freq_bin: u32,
        na_as_missing: bool,
        default_left: bool,
        par_min: usize,
    ) -> Result<(i32, i32), ComputeError> {
        // V5 boundary validation FIRST (matches the variants/fields the
        // `backend.data_partition` path returns so callers see no behavior change),
        // surfacing the LOWEST-index offending bin.
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
        // Router (dense_bin.hpp:322-365) — identical to `make_router`. Byte-for-byte
        // the SAME `go_right` decision in both the serial and parallel arms.
        let min_b = min_bin as i32;
        let max_b = max_bin as i32;
        let thr = threshold as i32;
        let mut th = thr + min_b;
        if most_freq_bin == 0 {
            th -= 1;
        }
        let default_to_right = most_freq_bin as i32 > thr;
        // NA_AS_MISSING (T-G4-2, SPEC-G4-2): the NaN sentinel bin is ALWAYS
        // `num_bin - 1` (`BinMapper::ValueToBin`'s NaN routing,
        // `bin_mapper.rs:1091-1106`; the `na_as_missing()` gate already requires
        // `num_bin > 2`, so `MFB_IS_NA` — the rare case where the most-frequent
        // bin ALSO coincides with the NaN sentinel bin — is not covered here,
        // matching T-G4-1's histogram-scan scope). C++ `SplitInner`
        // (`dense_bin.hpp:340-345,350-352`) checks `MISS_IS_NA && !MFB_IS_NA &&
        // bin == maxb` FIRST, before the out-of-range/threshold routing, and
        // sends it to `lte_indices` iff `default_left`.
        let na_sentinel_bin = num_bin as i32 - 1;
        let go_right = |b: u32| -> bool {
            let bin = b as i32;
            if na_as_missing && bin == na_sentinel_bin {
                return !default_left;
            }
            if bin < min_b || bin > max_b {
                default_to_right
            } else {
                bin > th
            }
        };

        if count >= par_min && count >= 2 {
            return self.partition_parallel(begin, count, feature_bins, num_bin, &go_right);
        }

        // ---- SERIAL two-buffer path (below the threshold) ----
        // pass 1: gather + RANGE-CHECK + route + scatter row ids DIRECTLY into the
        // two-buffer scratch (the ONE random gather). Disjoint field borrows: read from
        // `self.indices`, write into `self.part_left`/`self.part_right`. The range-check
        // returns BEFORE any scratch write for the offending row, and pass-1 NEVER writes
        // `self.indices`, so an early-return on the first (lowest-index) bad bin leaves
        // `self.indices` UNMUTATED.
        let mut left_count = 0usize;
        let mut right_count = 0usize;
        for i in 0..count {
            let row = self.indices[begin + i];
            let b = feature_bins.bin(row as usize);
            if b >= num_bin {
                return Err(ComputeError::BinIndexOutOfRange {
                    row: i,
                    bin: b,
                    num_bin,
                });
            }
            if go_right(b) {
                self.part_right[right_count] = row;
                right_count += 1;
            } else {
                self.part_left[left_count] = row;
                left_count += 1;
            }
        }

        // pass 2: copy the left run then the right run into the leaf slice in place —
        // one O(count) pass total (left_count + right_count == count). Disjoint field
        // borrows again (`self.indices` written, `self.part_left`/`self.part_right` read).
        self.indices[begin..begin + left_count].copy_from_slice(&self.part_left[..left_count]);
        self.indices[begin + left_count..begin + count]
            .copy_from_slice(&self.part_right[..right_count]);

        Ok((left_count as i32, right_count as i32))
    }

    /// Block-parallel two-buffer partition (C++ `ParallelPartitionRunner::Run`,
    /// `threading.h`). BOTH passes run parallel over `B = min(num_threads,
    /// ceil(count/min_block))` aligned blocks of the leaf's rows:
    ///
    /// - **pass 1** (parallel): each block owns a disjoint contiguous row range and
    ///   scatters its left rows (ascending) into `part_left[block_start..]` and its
    ///   right rows (ascending) into `part_right[block_start..]` — disjoint by block
    ///   offset (`par_chunks_mut`). It returns `(block_left_count, block_right_count,
    ///   lowest_bad_index)`. Pass-1 writes ONLY scratch, never `self.indices`.
    /// - **error reduce**: the per-block lowest bad indices reduce to the GLOBAL
    ///   minimum; if any, `BinIndexOutOfRange` returns BEFORE pass-2 mutates
    ///   `self.indices` (unmutated on error), preserving lowest-index semantics.
    /// - **exclusive prefix-sum** of the per-block left/right counts →
    ///   `left_write_pos` / `right_write_pos`; `total_left = sum(left_counts)`.
    /// - **pass 2** (parallel): the leaf slice is `split_at_mut(total_left)` into a
    ///   left region and a right region; each is carved by successive `split_at_mut`
    ///   into per-block disjoint sub-slices in ascending block order (the prefix-sum
    ///   guarantees they tile each region exactly). Each block then `copy_from_slice`s
    ///   its scratch run into its sub-slice — FULLY SAFE disjoint writes (no unsafe;
    ///   `split_at_mut` proves non-aliasing).
    ///
    /// Stability: pass-1 appends ascending within each block; pass-2 lays block 0's
    /// left run, then block 1's, … then all right runs — so the result is
    /// `[left ascending | right ascending]`, byte-identical to the serial arm.
    fn partition_parallel(
        &mut self,
        begin: usize,
        count: usize,
        feature_bins: &BinColumn,
        num_bin: u32,
        go_right: &(impl Fn(u32) -> bool + Sync),
    ) -> Result<(i32, i32), ComputeError> {
        use rayon::prelude::*;

        // Block the leaf's rows: B = min(num_threads, ceil(count/min_block)) blocks.
        let min_block = 4096usize;
        let max_blocks = count.div_ceil(min_block).max(1);
        let n_blocks_target = rayon::current_num_threads().max(1).min(max_blocks).max(1);
        let block = count.div_ceil(n_blocks_target).max(1);

        // ---- pass 1 (parallel): disjoint field borrows — read `self.indices[begin..]`,
        // write per-block disjoint regions of `self.part_left` / `self.part_right`.
        let indices_slice = &self.indices[begin..begin + count];
        let left_scratch = &mut self.part_left[..count];
        let right_scratch = &mut self.part_right[..count];

        // Per block: (left_count, right_count, lowest_bad_index_within_block).
        let results: Vec<BlockResult> = left_scratch
            .par_chunks_mut(block)
            .zip(right_scratch.par_chunks_mut(block))
            .zip(indices_slice.par_chunks(block))
            .enumerate()
            .map(|(bi, ((lchunk, rchunk), ichunk))| {
                let base = bi * block;
                let mut lc = 0usize;
                let mut rc = 0usize;
                let mut bad: Option<(usize, u32)> = None;
                for (j, &row) in ichunk.iter().enumerate() {
                    let b = feature_bins.bin(row as usize);
                    if b >= num_bin {
                        // First (lowest local) bad in this block; global min taken below.
                        bad = Some((base + j, b));
                        break;
                    }
                    if go_right(b) {
                        rchunk[rc] = row;
                        rc += 1;
                    } else {
                        lchunk[lc] = row;
                        lc += 1;
                    }
                }
                (lc, rc, bad)
            })
            .collect();

        // Reduce per-block lowest-bad to the GLOBAL lowest leaf position. `self.indices`
        // is still UNMUTATED (pass-1 wrote only scratch), so return before pass-2.
        if let Some((row, bin)) = results
            .iter()
            .filter_map(|r| r.2)
            .min_by_key(|(idx, _)| *idx)
        {
            return Err(ComputeError::BinIndexOutOfRange { row, bin, num_bin });
        }

        // `total_left` = exclusive prefix-sum total of per-block left counts. The
        // per-block WRITE POSITIONS are the exclusive prefix sums of the left/right
        // counts; here they are realized IMPLICITLY by carving each region with
        // successive `split_at_mut` in ascending block order (pass 2 below) — the
        // running split offset IS the prefix sum, so no explicit position array is kept.
        let n_blocks = results.len();
        let total_left: usize = results.iter().map(|r| r.0).sum();

        // ---- pass 2 (parallel): carve the leaf slice into per-block disjoint dest
        // sub-slices via `split_at_mut` (SAFE, no aliasing), then copy each scratch run.
        let part_left = &self.part_left;
        let part_right = &self.part_right;
        let leaf = &mut self.indices[begin..begin + count];
        let (left_region, right_region) = leaf.split_at_mut(total_left);

        // Tile the left region by ascending block order (prefix-sum guarantees exact tiling).
        let mut left_parts: Vec<&mut [u32]> = Vec::with_capacity(n_blocks);
        let mut lrem: &mut [u32] = left_region;
        for r in results.iter() {
            let (head, tail) = lrem.split_at_mut(r.0);
            left_parts.push(head);
            lrem = tail;
        }
        let mut right_parts: Vec<&mut [u32]> = Vec::with_capacity(n_blocks);
        let mut rrem: &mut [u32] = right_region;
        for r in results.iter() {
            let (head, tail) = rrem.split_at_mut(r.1);
            right_parts.push(head);
            rrem = tail;
        }

        left_parts
            .into_par_iter()
            .zip(right_parts.into_par_iter())
            .enumerate()
            .for_each(|(bi, (ldst, rdst))| {
                let base = bi * block;
                let lc = ldst.len();
                let rc = rdst.len();
                ldst.copy_from_slice(&part_left[base..base + lc]);
                rdst.copy_from_slice(&part_right[base..base + rc]);
            });

        Ok((total_left as i32, (count - total_left) as i32))
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
            .split(&backend, &client, 0, 1, &feature_bins, 8, 0, 7, 3, 8, false, false)
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

    /// T-G4-2 (SPEC-G4-2) Red: on an `na_as_missing` feature, rows carrying the
    /// NaN sentinel bin (`num_bin-1`, per `BinMapper::ValueToBin`'s NaN routing)
    /// route down the winning split's `default_left` direction — REGARDLESS of
    /// where `num_bin-1` sits relative to the numeric `threshold` (here it is
    /// `> threshold`, so a non-NA row with that bin would normally go RIGHT;
    /// `default_left=true` must override that to LEFT for the NaN rows only).
    /// `most_freq_bin=5` (out of `[min_bin,max_bin]`) keeps the ordinary-bin `th`
    /// unadjusted so the non-NA routing is easy to hand-verify:
    /// bins = [0, 1, 2, 3(NaN), 3(NaN), 0], threshold=1 ⇒ non-NA rows with
    /// bin<=1 go LEFT (rows 0,1,5), bin>1-and-not-NaN go RIGHT (row 2 only,
    /// bin=2); rows 3,4 (bin=3=NaN) go LEFT because `default_left=true`.
    #[test]
    fn split_na_as_missing_routes_nan_bin_by_default_left() {
        let backend = CpuBackend;
        let client = cpu_client();
        let mut dp = DataPartition::new(6, 4);
        let feature_bins = BinColumn::new(vec![0u32, 1, 2, 3, 3, 0], 4);
        let (l, r) = dp
            .split(&backend, &client, 0, 1, &feature_bins, 4, 0, 3, 1, 5, true, true)
            .expect("split ok");
        assert_eq!(l, 5, "left = non-NA bin<=1 rows {{0,1,5}} + NA rows {{3,4}}");
        assert_eq!(r, 1, "right = the single non-NA bin=2 row {{2}}");
        assert_eq!(dp.indices_in_leaf(0), &[0, 1, 3, 4, 5]);
        assert_eq!(dp.indices_in_leaf(1), &[2]);
    }

    /// Companion: flipping `default_left=false` on the SAME histogram/threshold
    /// sends the NaN rows to the OTHER child — proving the routing genuinely
    /// reads `default_left` (not merely "NA always left").
    #[test]
    fn split_na_as_missing_default_left_false_routes_nan_right() {
        let backend = CpuBackend;
        let client = cpu_client();
        let mut dp = DataPartition::new(6, 4);
        let feature_bins = BinColumn::new(vec![0u32, 1, 2, 3, 3, 0], 4);
        let (l, r) = dp
            .split(&backend, &client, 0, 1, &feature_bins, 4, 0, 3, 1, 5, true, false)
            .expect("split ok");
        assert_eq!(l, 3, "left = non-NA bin<=1 rows only {{0,1,5}}");
        assert_eq!(r, 3, "right = non-NA bin=2 row {{2}} + NA rows {{3,4}}");
        assert_eq!(dp.indices_in_leaf(0), &[0, 1, 5]);
        assert_eq!(dp.indices_in_leaf(1), &[2, 3, 4]);
    }

    /// A second split on a non-root leaf must operate on that leaf's slice only,
    /// using the global row ids it currently holds.
    #[test]
    fn split_on_child_leaf_uses_its_own_rows() {
        let backend = CpuBackend;
        let client = cpu_client();
        let mut dp = DataPartition::new(8, 4);
        let feature_bins = BinColumn::new(vec![1u32, 5, 3, 7, 0, 4, 2, 6], 8);
        dp.split(&backend, &client, 0, 1, &feature_bins, 8, 0, 7, 3, 8, false, false)
            .unwrap();
        // Now split leaf 1 (rows 1,3,5,7 with bins 5,7,4,6) at threshold 5:
        // bin<=5 left (rows 1(b5),5(b4)), bin>5 right (rows 3(b7),7(b6)).
        let (l, r) = dp
            .split(&backend, &client, 1, 2, &feature_bins, 8, 0, 7, 5, 8, false, false)
            .unwrap();
        assert_eq!((l, r), (2, 2));
        assert_eq!(dp.indices_in_leaf(1), &[1, 5]);
        assert_eq!(dp.indices_in_leaf(2), &[3, 7]);
        // Leaf 0 untouched.
        assert_eq!(dp.indices_in_leaf(0), &[0, 2, 4, 6]);
    }

    /// Serial reference: gather the
    /// leaf's bins, run `data_partition_cpu_native`, remap local→row, write back.
    /// Returns the rewritten leaf slice + `(left_count, right_count)`.
    #[allow(clippy::too_many_arguments)]
    fn serial_reference_slice(
        leaf_rows: &[u32],
        feature_bins: &BinColumn,
        num_bin: u32,
        min_bin: u32,
        max_bin: u32,
        threshold: u32,
        most_freq_bin: u32,
    ) -> (Vec<u32>, i32, i32) {
        use lgbm_compute::kernels::partition::data_partition_cpu_native;
        let leaf_feature_bins: Vec<u32> = leaf_rows
            .iter()
            .map(|&r| feature_bins.bin(r as usize))
            .collect();
        let (reordered, split_point) = data_partition_cpu_native(
            &leaf_feature_bins,
            num_bin,
            min_bin,
            max_bin,
            threshold,
            most_freq_bin,
        )
        .expect("serial reference ok");
        let out: Vec<u32> = reordered
            .iter()
            .map(|&local| leaf_rows[local as usize])
            .collect();
        let left = split_point as i32;
        let right = leaf_rows.len() as i32 - left;
        (out, left, right)
    }

    /// Two-buffer error semantics: an out-of-range bin must surface
    /// `BinIndexOutOfRange` for the LOWEST leaf-position offending row, and
    /// `self.indices` must be UNMUTATED (pass-1 writes only scratch; the early-return
    /// precedes pass-2). Two bad bins are planted; the lower leaf position must win.
    #[test]
    fn split_fused_reports_lowest_bad_bin_unmutated() {
        let backend = CpuBackend;
        let client = cpu_client();
        let mut dp = DataPartition::new(8, 4);
        let before = dp.indices().to_vec();
        // num_bin=8; rows 3 and 5 carry out-of-range bins (9, 11). Row 3 is the lower
        // leaf position ⇒ its bin (9) must be reported.
        let feature_bins = BinColumn::new(vec![1u32, 5, 3, 9, 0, 11, 2, 6], 16);
        let err = dp
            .split(&backend, &client, 0, 1, &feature_bins, 8, 0, 7, 3, 8, false, false)
            .expect_err("out-of-range bin must error");
        match err {
            ComputeError::BinIndexOutOfRange { row, bin, num_bin } => {
                assert_eq!(row, 3, "lowest offending leaf position");
                assert_eq!(bin, 9);
                assert_eq!(num_bin, 8);
            }
            other => panic!("expected BinIndexOutOfRange, got {other:?}"),
        }
        assert_eq!(dp.indices(), &before[..], "self.indices must be unmutated on error");
    }

    /// The fused host path (CpuBackend, `prefers_host_partition()==true`) must
    /// produce a BYTE-IDENTICAL `[left | right]` indices slice — and the same
    /// `(left_count, right_count)` — as the serial `data_partition_cpu_native`
    /// reference, over a SCATTERED leaf (random gather), for both the
    /// `most_freq_bin == 0` branch and a U8 `BinColumn` (the production narrow case).
    #[test]
    fn split_fused_equals_serial() {
        let backend = CpuBackend;
        let client = cpu_client();
        assert!(
            backend.prefers_host_partition(),
            "CpuBackend must select the fused host path"
        );

        // Deterministic LCG for a scattered leaf + column.
        let lcg = |seed: u64| {
            let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1);
            move || {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                (s >> 33) as u32
            }
        };

        let n: usize = 2_000;
        let num_bin = 64u32;
        let min_bin = 0u32;
        let max_bin = num_bin - 1;
        let threshold = 31u32;

        // A shuffled permutation of [0,n) ⇒ the leaf's rows are NOT in identity order
        // (so the bin gather is random), exercising the scatter path.
        let mut scattered: Vec<u32> = (0..n as u32).collect();
        let mut next = lcg(0xBEEF);
        for i in (1..n).rev() {
            let j = (next() as usize) % (i + 1);
            scattered.swap(i, j);
        }

        // Full-column bin values in [0,num_bin), skewed into the low half.
        let mut nextv = lcg(0xC0FFEE);
        let raw: Vec<u32> = (0..n)
            .map(|_| {
                let r = nextv();
                if (r as f64 / u32::MAX as f64) < 0.6 {
                    r % (num_bin / 2).max(1)
                } else {
                    r % num_bin
                }
            })
            .collect();

        // (a) most_freq_bin == 0 branch + (b) U8 column (num_bin<=256 ⇒ BinColumn::U8).
        for &most_freq_bin in &[0u32, 5u32] {
            let col = BinColumn::new(raw.clone(), num_bin);
            assert!(matches!(col, BinColumn::U8(_)), "narrow U8 column");

            // Build a DataPartition whose leaf 0 holds the scattered rows directly.
            let mut dp = DataPartition::new(n as i32, 4);
            dp.indices.copy_from_slice(&scattered);

            let (l, r) = dp
                .split(
                    &backend,
                    &client,
                    0,
                    1,
                    &col,
                    num_bin,
                    min_bin,
                    max_bin,
                    threshold,
                    most_freq_bin,
                    false,
                    false,
                )
                .expect("fused split ok");

            let (ref_slice, ref_l, ref_r) = serial_reference_slice(
                &scattered,
                &col,
                num_bin,
                min_bin,
                max_bin,
                threshold,
                most_freq_bin,
            );

            assert_eq!(
                (l, r),
                (ref_l, ref_r),
                "(left,right) mismatch most_freq_bin={most_freq_bin}"
            );
            assert_eq!(
                dp.indices_in_leaf(0),
                &ref_slice[..ref_l as usize],
                "left slice not byte-identical most_freq_bin={most_freq_bin}"
            );
            assert_eq!(
                dp.indices_in_leaf(1),
                &ref_slice[ref_l as usize..],
                "right slice not byte-identical most_freq_bin={most_freq_bin}"
            );
            // The whole rewritten leaf region must match byte-for-byte.
            assert_eq!(
                &dp.indices()[0..n],
                &ref_slice[..],
                "full [left|right] slice not byte-identical most_freq_bin={most_freq_bin}"
            );
        }
    }

    /// Deterministic LCG (mirrors `split_fused_equals_serial`) for scattered corpora.
    fn test_lcg(seed: u64) -> impl FnMut() -> u32 {
        let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1);
        move || {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (s >> 33) as u32
        }
    }

    /// The BLOCK-PARALLEL arm (`par_min = 0`) must be BYTE-IDENTICAL to
    /// the SERIAL arm (`par_min = usize::MAX`) and to the `data_partition_cpu_native`
    /// reference on a LARGE scattered multi-block leaf, for both the `most_freq_bin==0`
    /// branch and a U8 column. Byte-identity to the serial arm element-for-element IS
    /// the robust stability proof (order-preserving subsequence per side).
    #[test]
    fn split_parallel_equals_serial_multiblock() {
        let n: usize = 50_000;
        let num_bin = 64u32;
        let min_bin = 0u32;
        let max_bin = num_bin - 1;
        let threshold = 31u32;

        // Shuffled leaf permutation (random bin gather + exercises the scatter).
        let mut scattered: Vec<u32> = (0..n as u32).collect();
        let mut next = test_lcg(0xBEEF);
        for i in (1..n).rev() {
            let j = (next() as usize) % (i + 1);
            scattered.swap(i, j);
        }
        // Full-column bins skewed into the low half (uneven left/right split).
        let mut nextv = test_lcg(0xC0FFEE);
        let raw: Vec<u32> = (0..n)
            .map(|_| {
                let r = nextv();
                if (r as f64 / u32::MAX as f64) < 0.6 {
                    r % (num_bin / 2).max(1)
                } else {
                    r % num_bin
                }
            })
            .collect();

        for &most_freq_bin in &[0u32, 5u32] {
            let col = BinColumn::new(raw.clone(), num_bin);
            assert!(matches!(col, BinColumn::U8(_)), "narrow U8 column");

            let mut dp_ser = DataPartition::new(n as i32, 4);
            dp_ser.indices.copy_from_slice(&scattered);
            let (ls, rs) = dp_ser
                .split_fused_host_impl(
                    0, n, &col, num_bin, min_bin, max_bin, threshold, most_freq_bin, false, false, usize::MAX,
                )
                .expect("serial arm ok");

            let mut dp_par = DataPartition::new(n as i32, 4);
            dp_par.indices.copy_from_slice(&scattered);
            let (lp, rp) = dp_par
                .split_fused_host_impl(
                    0, n, &col, num_bin, min_bin, max_bin, threshold, most_freq_bin, false, false, 0,
                )
                .expect("parallel arm ok");

            assert_eq!((ls, rs), (lp, rp), "counts differ mfb={most_freq_bin}");
            assert_eq!(
                dp_ser.indices(),
                dp_par.indices(),
                "parallel arm not byte-identical to serial arm mfb={most_freq_bin}"
            );

            let (ref_slice, ref_l, ref_r) = serial_reference_slice(
                &scattered, &col, num_bin, min_bin, max_bin, threshold, most_freq_bin,
            );
            assert_eq!((lp, rp), (ref_l, ref_r), "vs reference mfb={most_freq_bin}");
            assert_eq!(
                &dp_par.indices()[0..n],
                &ref_slice[..],
                "parallel not byte-identical to cpu_native reference mfb={most_freq_bin}"
            );
        }
    }

    /// DIRECT stability (defense-in-depth beyond sha/parity): on an
    /// IDENTITY-ordered multi-block leaf, each side must be STRICTLY INCREASING —
    /// left rows in ascending original order, then right rows in ascending original
    /// order (here original order == row id) — and equal the serial arm element-for-element.
    #[test]
    fn split_parallel_stable_identity_ordered() {
        let n: usize = 50_000;
        let num_bin = 64u32;
        let (min_bin, max_bin, threshold, most_freq_bin) = (0u32, num_bin - 1, 31u32, 5u32);

        let mut nextv = test_lcg(0x5EED);
        let raw: Vec<u32> = (0..n).map(|_| nextv() % num_bin).collect();
        let col = BinColumn::new(raw, num_bin);

        // Identity leaf (indices already 0..n from `new`).
        let mut dp_par = DataPartition::new(n as i32, 4);
        let (lp, _rp) = dp_par
            .split_fused_host_impl(
                0, n, &col, num_bin, min_bin, max_bin, threshold, most_freq_bin, false, false, 0,
            )
            .expect("parallel ok");

        let left = &dp_par.indices()[0..lp as usize];
        let right = &dp_par.indices()[lp as usize..n];
        assert!(
            left.windows(2).all(|w| w[0] < w[1]),
            "left side not strictly ascending (unstable reorder)"
        );
        assert!(
            right.windows(2).all(|w| w[0] < w[1]),
            "right side not strictly ascending (unstable reorder)"
        );

        // And byte-identical to the serial arm on the same identity leaf.
        let mut dp_ser = DataPartition::new(n as i32, 4);
        let col2 = {
            let mut nv = test_lcg(0x5EED);
            BinColumn::new((0..n).map(|_| nv() % num_bin).collect(), num_bin)
        };
        dp_ser
            .split_fused_host_impl(
                0, n, &col2, num_bin, min_bin, max_bin, threshold, most_freq_bin, false, false, usize::MAX,
            )
            .expect("serial ok");
        assert_eq!(dp_ser.indices(), dp_par.indices(), "parallel != serial (identity leaf)");
    }

    /// T-G4-2: the block-parallel arm's `go_right` closure is the SAME closure
    /// the serial arm uses (captured, not re-derived per block), so `na_as_missing`
    /// routing must stay byte-identical between the two arms on a large
    /// (multi-block) leaf too — the same parity discipline as
    /// `split_parallel_stable_identity_ordered`.
    #[test]
    fn split_parallel_equals_serial_na_as_missing() {
        let n: usize = 50_000;
        let num_bin = 64u32;
        let (min_bin, max_bin, threshold, most_freq_bin) = (0u32, num_bin - 1, 31u32, 5u32);
        let mut nextv = test_lcg(0x5EED);
        let raw: Vec<u32> = (0..n).map(|_| nextv() % num_bin).collect();
        let col = BinColumn::new(raw, num_bin);

        for &default_left in &[true, false] {
            let mut dp_ser = DataPartition::new(n as i32, 4);
            let (ls, rs) = dp_ser
                .split_fused_host_impl(
                    0, n, &col, num_bin, min_bin, max_bin, threshold, most_freq_bin, true,
                    default_left, usize::MAX,
                )
                .expect("serial ok");

            let mut dp_par = DataPartition::new(n as i32, 4);
            let (lp, rp) = dp_par
                .split_fused_host_impl(
                    0, n, &col, num_bin, min_bin, max_bin, threshold, most_freq_bin, true,
                    default_left, 0,
                )
                .expect("parallel ok");

            assert_eq!((ls, rs), (lp, rp), "counts differ, default_left={default_left}");
            assert_eq!(
                dp_ser.indices(),
                dp_par.indices(),
                "parallel != serial for na_as_missing, default_left={default_left}"
            );
        }
    }

    /// Error semantics on the PARALLEL path: two out-of-range bins are planted
    /// in DIFFERENT blocks of a large leaf; the LOWEST leaf position must be reported
    /// as `BinIndexOutOfRange` and `self.indices` must be UNMUTATED (pass-1 writes only
    /// scratch, error returns before pass-2).
    #[test]
    fn split_parallel_reports_lowest_bad_bin_unmutated() {
        let n: usize = 50_000;
        let num_bin = 64u32;
        let mut nextv = test_lcg(0xBAD5EED);
        let mut raw: Vec<u32> = (0..n).map(|_| nextv() % num_bin).collect();
        // Plant out-of-range bins at two identity leaf positions in (likely) distinct
        // blocks; the lower (10_000) must win.
        raw[10_000] = num_bin; // out of range
        raw[40_000] = num_bin + 3; // out of range, higher position
        let col = BinColumn::new(raw, num_bin);

        let mut dp = DataPartition::new(n as i32, 4);
        let before = dp.indices().to_vec();
        let err = dp
            .split_fused_host_impl(0, n, &col, num_bin, 0, num_bin - 1, 31, 5, false, false, 0)
            .expect_err("out-of-range bin must error on the parallel path");
        match err {
            ComputeError::BinIndexOutOfRange { row, bin, num_bin: nb } => {
                assert_eq!(row, 10_000, "lowest offending leaf position across blocks");
                assert_eq!(bin, num_bin);
                assert_eq!(nb, num_bin);
            }
            other => panic!("expected BinIndexOutOfRange, got {other:?}"),
        }
        assert_eq!(
            dp.indices(),
            &before[..],
            "self.indices must be unmutated on parallel-path error"
        );
    }
}
