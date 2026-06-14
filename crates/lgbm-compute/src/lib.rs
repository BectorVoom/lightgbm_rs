//! `lgbm-compute` — the single CubeCL isolation seam (CMP-01).
//!
//! This crate exists to confine all `cubecl` type names and API churn to one
//! place so the alpha-stage CubeCL surface can evolve without leaking into
//! `lgbm-core` or any other crate. Phase 4 fills the Phase-1 kernel-free
//! skeleton with the compute foundation: a typed [`ComputeError`] boundary, a
//! cpu/rocm runtime selection + startup capability gate ([`runtime`]), and the
//! first `#[cube]` histogram kernel ([`kernels`]).
//!
//! Downstream crates should depend only on the [`Backend`] abstraction (and the
//! re-exported [`ComputeError`]), never on `cubecl` directly — that is the whole
//! point of the seam, and the `cmp01_containment` guard test enforces it.

pub mod error;
pub mod gain;
pub mod kernels;
pub mod runtime;

pub use error::ComputeError;
pub use gain::{GainConfig, SplitInfo};
pub use kernels::split::BatchedSplitFeature;

use cubecl::prelude::ComputeClient;

/// Re-export of the cubecl [`ComputeClient`](cubecl::prelude::ComputeClient) so
/// downstream crates (e.g. `lgbm-treelearner`) can name the
/// `&ComputeClient<B::Runtime>` argument the [`Backend`] ops require WITHOUT
/// depending on `cubecl` directly — preserving the CMP-01 containment boundary
/// (the compute crate is the single CubeCL seam; everyone above it sees only
/// `lgbm_compute::ComputeClient`).
pub use cubecl::prelude::ComputeClient as ComputeClientReexport;

/// The compute backend seam (CMP-01).
///
/// Binds a concrete CubeCL [`Runtime`](cubecl::Runtime) (CPU or ROCm/HIP) that
/// kernels are dispatched to. The coarse whole-kernel ops (D-01) live on this
/// trait: [`construct_histograms`](Backend::construct_histograms) is finalized
/// in 04-02 (this plan); `find_best_split` / `data_partition` follow in 04-03.
///
/// This trait is the ONLY place where CubeCL runtime types should appear; that
/// is the whole point of the seam.
pub trait Backend {
    /// The concrete CubeCL runtime this backend dispatches kernels to.
    type Runtime: cubecl::Runtime;

    /// Construct a single feature column's gradient/hessian histogram (D-01
    /// whole-kernel op, faithful to `dense_bin.hpp:99-141`).
    ///
    /// Inputs (sourced from the Phase-2 binned store — do NOT re-bin):
    /// - `client`  — the compute client for [`Self::Runtime`].
    /// - `binned`  — the per-row bin indices for this feature column, i.e. the
    ///   `u32`-widened `Bin::data(idx)` for `idx in 0..num_data()`.
    /// - `ordered_gradients` / `ordered_hessians` — the `f32`
    ///   (`score_t = float`) gradient/hessian slice, one per row, in the SAME
    ///   row order as `binned`.
    /// - `num_bin` — the feature's bin count; the output has `2 * num_bin` cells.
    ///
    /// Output: the stride-2 interleaved `[g0,h0,g1,h1,…]` histogram of length
    /// `2 * num_bin`, indexed `ti = bin << 1` (`out[ti] += grad`,
    /// `out[ti + 1] += hess`). Gradients/hessians are read as `f32` but
    /// accumulated into `f64` cells (`hist_t = double`, RESEARCH Pitfall 3) on
    /// the single-owner ordered fold proven bit-exact in 04-01.
    ///
    /// # Errors
    /// Returns [`ComputeError::LengthMismatch`] if `ordered_gradients`/
    /// `ordered_hessians`/`binned` lengths differ, or
    /// [`ComputeError::BinIndexOutOfRange`] if any `binned[i] >= num_bin` (V5
    /// boundary validation, threat T-04-01) — never a panic / UB.
    fn construct_histograms(
        &self,
        client: &ComputeClient<Self::Runtime>,
        binned: &[u32],
        ordered_gradients: &[f32],
        ordered_hessians: &[f32],
        num_bin: u32,
    ) -> Result<Vec<f64>, ComputeError>;

    /// Find the best split threshold for a feature column (D-01 whole-kernel op,
    /// gain math in-kernel per D-01a), faithful to
    /// `feature_histogram.hpp:165-1057` (the default CPU template
    /// `<USE_RAND=false, USE_MC=false, USE_MAX_OUTPUT=false,
    /// USE_SMOOTHING=false>`; `USE_L1` keyed on `cfg.lambda_l1 > 0`).
    ///
    /// Inputs:
    /// - `hist` — the stride-2 `[g0,h0,g1,h1,…]` f64 histogram from
    ///   [`construct_histograms`](Backend::construct_histograms), length
    ///   `2 * num_bin`.
    /// - `cfg` — the [`GainConfig`] (the seven gain-relevant `Config` fields).
    /// - `num_bin` — the feature's bin count.
    /// - `offset` / `default_bin` / `most_freq_bin` — the Phase-2
    ///   `FeatureGroup`/`Bin` bin-layout descriptors driving the
    ///   `SKIP_DEFAULT_BIN` continue and the threshold offset arithmetic.
    /// - `skip_default_bin` / `na_as_missing` — the AUTHORITATIVE C++ dispatch
    ///   flags (`feature_histogram.hpp:284-285`), derived by the caller from the
    ///   feature's `missing_type` + `num_bin > 2`
    ///   (`skip == (num_bin > 2 && missing_type == Zero)`,
    ///   `na_as_missing == (num_bin > 2 && missing_type == NaN)`, both false for
    ///   `missing_type == None`). These REPLACE the Phase-4
    ///   `cfg_skip_default_bin(default_bin, num_bin)` heuristic (RESEARCH
    ///   Pitfall 1).
    /// - `run_forward` — the AUTHORITATIVE C++ FORWARD-branch dispatch flag
    ///   (`feature_histogram.hpp:420-429`): the FORWARD scan runs ONLY when
    ///   `num_bin > 2 && missing_type == Zero` (the sole dispatch invoking both the
    ///   REVERSE and FORWARD `FindBestThresholdSequentially`). For
    ///   `missing_type == None` (and `num_bin <= 2`) only the REVERSE branch runs,
    ///   so `FindBestThreshold`'s pre-set `default_left = true` survives and
    ///   `decision_type == 2`. Equal to `skip_default_bin` here (the deferred NaN
    ///   case is a typed error), but threaded explicitly as a verbatim transcription
    ///   of the C++ dispatch truth table, NOT a bin-layout heuristic.
    ///   `na_as_missing == true` is currently a typed
    ///   [`ComputeError::Runtime`] (the NA_AS_MISSING forward branch is deferred,
    ///   RESEARCH A5 — never a silent wrong answer).
    /// - `sum_gradient` / `sum_hessian` / `num_data` — the leaf totals.
    ///
    /// Returns a [`SplitInfo`]; `gain == f64::NEG_INFINITY` (C++ `kMinScore`)
    /// signals "no valid split found".
    ///
    /// # Errors
    /// [`ComputeError::LengthMismatch`] if `hist.len() != 2 * num_bin`, or
    /// [`ComputeError::Runtime`] for `num_bin == 0`, non-positive `sum_hessian`,
    /// `na_as_missing == true` (deferred branch), or unsupported non-default gain
    /// params (V5, T-04-01).
    #[allow(clippy::too_many_arguments)]
    fn find_best_split(
        &self,
        client: &ComputeClient<Self::Runtime>,
        hist: &[f64],
        cfg: &GainConfig,
        num_bin: u32,
        offset: i32,
        default_bin: u32,
        most_freq_bin: u32,
        skip_default_bin: bool,
        na_as_missing: bool,
        run_forward: bool,
        sum_gradient: f64,
        sum_hessian: f64,
        num_data: i32,
    ) -> Result<SplitInfo, ComputeError>;

    /// Partition a leaf's rows left/right by a feature threshold, mirroring the
    /// C++ `DataPartition::Split` stable reorder (`data_partition.hpp:101`, the
    /// `MissingType::None` numeric routing of `DenseBin::SplitInner`).
    ///
    /// Returns `(reordered, split_point)`: a STABLE reordered index array — the
    /// left rows in their original relative order followed by the right rows in
    /// their original relative order — and `split_point` = the left-row count
    /// (left indices occupy `[0, split_point)`, right `[split_point, len)`). The
    /// Phase-5 learner owns `leaf_begin_`/`leaf_count_` bookkeeping; this op
    /// returns only the partition.
    ///
    /// # Errors
    /// [`ComputeError::Runtime`] if `num_bin == 0` or `threshold >= num_bin`, or
    /// [`ComputeError::BinIndexOutOfRange`] for any `bins[i] >= num_bin` (V5).
    #[allow(clippy::too_many_arguments)]
    fn data_partition(
        &self,
        client: &ComputeClient<Self::Runtime>,
        bins: &[u32],
        num_bin: u32,
        min_bin: u32,
        max_bin: u32,
        threshold: u32,
        most_freq_bin: u32,
    ) -> Result<(Vec<u32>, usize), ComputeError>;

    /// Derive the larger child's histogram via the subtraction trick
    /// (`parent - child`), the kernel-layer MATH of `FeatureHistogram::Subtract`
    /// (`feature_histogram.hpp:99`). WHICH child is subtracted (the smaller
    /// sibling) is Phase-5 orchestration (RESEARCH A3: the subtract OP is
    /// in-scope at the kernel layer).
    ///
    /// `parent` / `child` are the stride-2 `[g0,h0,g1,h1,…]` f64 histograms of
    /// equal length `2 * num_bin`.
    ///
    /// # Errors
    /// [`ComputeError::LengthMismatch`] if `parent.len() != child.len()` (V5).
    fn subtract_histograms(
        &self,
        client: &ComputeClient<Self::Runtime>,
        parent: &[f64],
        child: &[f64],
    ) -> Result<Vec<f64>, ComputeError>;

    /// Build the RAW (pre-FixHistogram, pre-compact) per-feature histograms for ONE
    /// leaf's rows, concatenated into a single `slot_len`-cell f64 buffer (feature
    /// `fpos` occupies `[slot_off[fpos], slot_off[fpos] + 2*num_bins[fpos])`). This
    /// is the batched per-leaf abstraction seam (260608-lad): the learner calls it
    /// ONCE per leaf instead of looping `construct_histograms` per feature.
    ///
    /// The DEFAULT implementation here is exactly the per-feature host gather + per-
    /// feature `construct_histograms` loop (the bit-exact CPU anchor path). A GPU
    /// backend OVERRIDES this to gather + dispatch all features in ONE kernel launch
    /// (and to keep the binned dataset device-resident), collapsing the per-feature
    /// launch count to one per leaf.
    ///
    /// `feature_bins[fpos]` is feature `fpos`'s GLOBAL-row bin column; `leaf_rows`
    /// are the leaf's global row indices (the ordered fold order). FixHistogram +
    /// compaction stay in the caller (they read per-leaf sums + the compaction
    /// offset), applied to each feature's region of the returned RAW buffer.
    ///
    /// # Errors
    /// Propagates [`construct_histograms`](Backend::construct_histograms) errors.
    #[allow(clippy::too_many_arguments)]
    fn build_leaf_histograms_raw(
        &self,
        client: &ComputeClient<Self::Runtime>,
        feature_bins: &[&[u32]],
        num_bins: &[u32],
        slot_off: &[usize],
        slot_len: usize,
        leaf_rows: &[u32],
        gradients: &[f32],
        hessians: &[f32],
    ) -> Result<Vec<f64>, ComputeError> {
        let mut out = vec![0.0f64; slot_len];
        // Scratch gather buffers, reused across features (the R2 buffer-reuse).
        let mut ord_bins: Vec<u32> = Vec::with_capacity(leaf_rows.len());
        let mut ord_g: Vec<f32> = Vec::with_capacity(leaf_rows.len());
        let mut ord_h: Vec<f32> = Vec::with_capacity(leaf_rows.len());
        for (fpos, &bins) in feature_bins.iter().enumerate() {
            ord_bins.clear();
            ord_g.clear();
            ord_h.clear();
            for &row in leaf_rows {
                ord_bins.push(bins[row as usize]);
                ord_g.push(gradients[row as usize]);
                ord_h.push(hessians[row as usize]);
            }
            // R3 fold-in-place: fold this feature DIRECTLY into its pre-zeroed `out`
            // sub-slice — no per-feature Vec, no memset, no memcpy. `out` was zeroed
            // once above (`vec![0.0f64; slot_len]`), so the destination region is
            // already 0.0; the native accumulator only adds. The fold ORDER (same
            // ascending rows, same bin<<1 cells, f32→f64) is byte-identical to the
            // prior `construct_histograms + copy_from_slice` path, preserving the
            // bit-exact CPU f64 merge gate. `_client` is unused on the native fold.
            let _ = client;
            let cells = 2 * num_bins[fpos] as usize;
            let slot = slot_off[fpos];
            kernels::histogram::accumulate_histogram_into(
                &ord_bins,
                &ord_g,
                &ord_h,
                num_bins[fpos],
                &mut out[slot..slot + cells],
            )?;
        }
        Ok(out)
    }

    /// Find the best split for EVERY spine feature of ONE leaf in a single batched
    /// op (260608-lad Part 2): the fused per-leaf SPLIT SCAN over the concatenated
    /// stride-2 f64 histogram `buf` (the same layout
    /// [`build_leaf_histograms_raw`](Backend::build_leaf_histograms_raw) produces —
    /// feature `f` occupies `[f.slot_off, f.slot_off + 2*f.num_bin)`). The learner
    /// calls this ONCE per leaf instead of looping
    /// [`find_best_split`](Backend::find_best_split) per feature.
    ///
    /// `feats` carries the per-feature dispatch parameters (one entry per scanned
    /// spine feature, in ascending feature position); `cfg` + `sum_gradient` /
    /// `sum_hessian` / `num_data` are the leaf totals shared across the batch.
    /// Returns one [`SplitInfo`] per input feature, **in the SAME order as `feats`**
    /// — order-preservation keeps the caller's cross-feature argmax (gain, then
    /// smaller feature) tie-break identical, which is what keeps the CPU-grown tree
    /// bit-exact (threat T-lsx-01).
    ///
    /// The DEFAULT impl (used by [`CpuBackend`] unchanged) loops
    /// [`find_best_split`](Backend::find_best_split) over `feats` in order, so each
    /// feature's [`SplitInfo`] is byte-identical to today's per-feature call — the
    /// default IS the bit-exact f64 anchor. A GPU backend OVERRIDES this to find all
    /// features' splits in one launch per leaf.
    ///
    /// An empty `feats` (every feature gated out / categorical-only leaf) returns an
    /// empty Vec with no launch (threat T-lsx-03).
    ///
    /// # Errors
    /// Propagates [`find_best_split`](Backend::find_best_split) errors; returns
    /// [`ComputeError::LengthMismatch`] if any feature's
    /// `[slot_off, slot_off + 2*num_bin)` region exceeds `buf` (V5, threat T-lsx-02
    /// — no panic / UB).
    fn find_best_splits_batched(
        &self,
        client: &ComputeClient<Self::Runtime>,
        buf: &[f64],
        feats: &[BatchedSplitFeature],
        cfg: &GainConfig,
        sum_gradient: f64,
        sum_hessian: f64,
        num_data: i32,
    ) -> Result<Vec<SplitInfo>, ComputeError> {
        let mut out = Vec::with_capacity(feats.len());
        for f in feats {
            let cells = 2usize
                .checked_mul(f.num_bin as usize)
                .ok_or_else(|| ComputeError::Runtime {
                    detail: format!("num_bin {} overflows the histogram length", f.num_bin),
                })?;
            let end = f
                .slot_off
                .checked_add(cells)
                .ok_or_else(|| ComputeError::Runtime {
                    detail: "find_best_splits_batched: slot_off + region overflows".to_string(),
                })?;
            if end > buf.len() {
                return Err(ComputeError::LengthMismatch {
                    expected: end,
                    actual: buf.len(),
                });
            }
            let hist = &buf[f.slot_off..end];
            let si = self.find_best_split(
                client,
                hist,
                cfg,
                f.num_bin,
                f.offset,
                f.default_bin,
                f.most_freq_bin,
                f.skip_default_bin,
                f.na_as_missing,
                f.run_forward,
                sum_gradient,
                sum_hessian,
                num_data,
            )?;
            out.push(si);
        }
        Ok(out)
    }

    /// One-time per-train upload of the binned feature columns to the device
    /// (260608-nn7 L1). The learner calls this ONCE per `train_inner` (before the
    /// per-leaf growth loop) with every feature's GLOBAL-row bin column; a GPU
    /// backend uploads them ONCE and caches the device `Handle` (interior
    /// mutability), so per-leaf histogram builds gather rows ON DEVICE from the
    /// resident buffer instead of re-uploading a host-gathered
    /// `[num_features × rows]` bin matrix every leaf.
    ///
    /// The DEFAULT impl is a NO-OP: [`CpuBackend`] is the bit-exact host anchor and
    /// keeps its per-feature host gather + native f64 fold
    /// ([`build_leaf_histograms_raw`](Backend::build_leaf_histograms_raw) default),
    /// so this seam adds ZERO behavior change to the CPU path. `feature_bins[fpos]`
    /// is feature `fpos`'s full-column bin slice (length `num_data`); all columns
    /// share the same `num_data`.
    fn upload_resident_bins(
        &self,
        _client: &ComputeClient<Self::Runtime>,
        _feature_bins: &[&[u32]],
    ) {
    }

    // ===================================================================
    // 260608-p90: DEVICE-RESIDENT histogram-pool seam.
    //
    // A device-Handle slot mirror that follows the host `HistogramPool` slot
    // bookkeeping, so a pure-numeric-spine tree keeps its per-leaf histograms
    // DEVICE-RESIDENT from build through fix/compact/subtract/scan (eliminating the
    // per-leaf host read-back + re-upload). Every method has a DEFAULT impl that
    // makes the CPU path byte-unchanged: `resident_pool_supported() == false` means
    // the learner's eligibility gate never takes the resident branch on CpuBackend,
    // and the no-op / typed-error defaults are never reached on cpu. RocmBackend
    // OVERRIDES all of them.
    // ===================================================================

    /// Whether this backend supports the device-resident histogram pool (260608-p90).
    /// `false` (the default, CpuBackend) means the learner's `resident_eligible` gate
    /// ANDs this in and ALWAYS takes the byte-unchanged host path. RocmBackend returns
    /// `true`.
    fn resident_pool_supported(&self) -> bool {
        false
    }

    /// Clear/resize the device-handle slot mirror for a new tree (260608-p90), called
    /// alongside the host `HistogramPool::reset_map`. Default: no-op (CpuBackend never
    /// takes the resident branch).
    fn reset_resident_pool(&self, _num_slots: usize, _slot_len: usize) {}

    /// Build ONE leaf's per-feature histogram DEVICE-RESIDENT (build → f32→f64 widen →
    /// fix → compact) and store the resulting f64 `Handle` into mirror slot `slot`
    /// (260608-p90). Mirrors `build_leaf_histogram_into` but keeps the histogram on
    /// device. `fix_feats[fpos]` is `(slot_off, num_bin, offset, most_freq_bin)` for
    /// feature `fpos`; `sum_gradient` / `sum_hessian` are the leaf RAW (un-bumped)
    /// totals (Pitfall 2). Default: typed error (never called on cpu — the gate is off).
    ///
    /// # Errors
    /// [`ComputeError::Runtime`] (unsupported) on the default; propagates the resident
    /// build/fix/compact kernel errors on RocmBackend.
    #[allow(clippy::too_many_arguments)]
    fn build_resident_leaf(
        &self,
        _client: &ComputeClient<Self::Runtime>,
        _slot: usize,
        _feature_bins: &[&[u32]],
        _num_bins: &[u32],
        _slot_off: &[usize],
        _slot_len: usize,
        _leaf_rows: &[u32],
        _gradients: &[f32],
        _hessians: &[f32],
        _fix_feats: &[(usize, u32, i32, u32)],
        _sum_gradient: f64,
        _sum_hessian: f64,
    ) -> Result<(), ComputeError> {
        Err(ComputeError::Runtime {
            detail: "build_resident_leaf: device-resident pool not supported on this backend"
                .to_string(),
        })
    }

    /// Move the resident Handle from `src_slot` to `dst_slot` in the device mirror
    /// (260608-p90), mirroring the host `HistogramPool::move_` slot reassignment so the
    /// device mirror's slot→Handle map tracks the host pool's slot→leaf map. Default:
    /// no-op.
    fn move_resident(&self, _src_slot: usize, _dst_slot: usize) {}

    /// Derive the larger child's resident histogram by the subtraction trick on device
    /// (`parent_slot` Handle − `smaller_slot` Handle → `larger_slot` Handle, no
    /// read-back; 260608-p90 Task 2). The derived larger child is NOT re-FixHistogram'd
    /// (matches host/C++, non-negotiable #3). Default: typed error.
    ///
    /// # Errors
    /// [`ComputeError::Runtime`] (unsupported) on the default; propagates the resident
    /// subtract kernel errors on RocmBackend.
    fn subtract_resident(
        &self,
        _client: &ComputeClient<Self::Runtime>,
        _parent_slot: usize,
        _smaller_slot: usize,
        _larger_slot: usize,
        _slot_len: usize,
    ) -> Result<(), ComputeError> {
        Err(ComputeError::Runtime {
            detail: "subtract_resident: device-resident pool not supported on this backend"
                .to_string(),
        })
    }

    /// Scan slot `slot`'s resident histogram Handle for every spine feature's best
    /// split in ONE fused launch (260608-p90), reading back only the `n*12` SplitInfo
    /// cells (the histogram Handle never leaves the device). Returns one [`SplitInfo`]
    /// per `feats` entry, in input order (the cross-feature-argmax tie-break invariant).
    /// Default: typed error.
    ///
    /// # Errors
    /// [`ComputeError::Runtime`] (unsupported / empty slot) on the default; propagates
    /// the fused split-scan errors on RocmBackend.
    #[allow(clippy::too_many_arguments)]
    fn scan_resident_leaf(
        &self,
        _client: &ComputeClient<Self::Runtime>,
        _slot: usize,
        _slot_len: usize,
        _feats: &[BatchedSplitFeature],
        _cfg: &GainConfig,
        _sum_gradient: f64,
        _sum_hessian: f64,
        _num_data: i32,
    ) -> Result<Vec<SplitInfo>, ComputeError> {
        Err(ComputeError::Runtime {
            detail: "scan_resident_leaf: device-resident pool not supported on this backend"
                .to_string(),
        })
    }

    /// 260608-t3t: FUSED directly-built-leaf path — build + fix + compact + scan a
    /// leaf's per-feature histogram in ONE launch. Builds the leaf's histogram
    /// DEVICE-RESIDENT (sequential f64 fold ⇒ bit-exact), fixes+compacts it, and
    /// scans it for every SCAN-ACTIVE feature's best split — STORING the
    /// fixed+compacted f64 Handle into mirror slot `slot` (so `subtract_resident` can
    /// still derive the larger child from it) AND returning one [`SplitInfo`] per
    /// SCAN-ACTIVE feature in order. `feats` is the FULL per-feature list (fpos order)
    /// — build+fix+compact run for EVERY feature so the resident histogram is COMPLETE
    /// for the subtraction trick — and `scan_active[fpos]` selects which features are
    /// scanned (the spine subset that passed the learner's gates). Collapses
    /// `build_resident_leaf` + `scan_resident_leaf` (3 launches) into 1. The leaf RAW
    /// (un-bumped) `sum_gradient_raw` / `sum_hessian_raw` feed the FIX (Pitfall 2), the
    /// launcher derives the 2*kEpsilon-bumped scan operand internally. Default: typed
    /// error (never called on cpu — the fused gate is off there).
    ///
    /// # Errors
    /// [`ComputeError::Runtime`] (unsupported) on the default; propagates the fused
    /// kernel errors on RocmBackend.
    #[allow(clippy::too_many_arguments)]
    fn build_fix_scan_resident(
        &self,
        _client: &ComputeClient<Self::Runtime>,
        _slot: usize,
        _slot_off: &[usize],
        _slot_len: usize,
        _leaf_rows: &[u32],
        _gradients: &[f32],
        _hessians: &[f32],
        _feats: &[BatchedSplitFeature],
        _scan_active: &[bool],
        _cfg: &GainConfig,
        _sum_gradient_raw: f64,
        _sum_hessian_raw: f64,
        _num_data: i32,
    ) -> Result<Vec<SplitInfo>, ComputeError> {
        Err(ComputeError::Runtime {
            detail: "build_fix_scan_resident: device-resident fused path not supported on this \
                     backend"
                .to_string(),
        })
    }
}

/// The default cpu-runtime backend (the D-04 deterministic anchor, CMP-02).
///
/// Binds [`runtime::ActiveRuntime`] (cubecl-cpu under the default `cpu` feature)
/// and dispatches [`construct_histograms`](Backend::construct_histograms) to the
/// single-owner ordered f64 fold in [`kernels::histogram`].
#[cfg(feature = "cpu")]
#[derive(Debug, Clone, Copy, Default)]
pub struct CpuBackend;

#[cfg(feature = "cpu")]
impl Backend for CpuBackend {
    type Runtime = runtime::ActiveRuntime;

    fn construct_histograms(
        &self,
        _client: &ComputeClient<Self::Runtime>,
        binned: &[u32],
        ordered_gradients: &[f32],
        ordered_hessians: &[f32],
        num_bin: u32,
    ) -> Result<Vec<f64>, ComputeError> {
        // R2: native f64 fold — bit-identical to the single-unit `construct_hist_
        // kernel` but without the ~20–50µs cubecl-cpu launch per call (the dominant
        // train-time cost). The cubecl path stays in `construct_histograms_cpu` for
        // the kernel-parity / ROCm-mirror tests. `_client` is unused on the native
        // path (kept for the `Backend` trait signature + the hip/f32 backends).
        kernels::histogram::construct_histograms_cpu_native(
            binned,
            ordered_gradients,
            ordered_hessians,
            num_bin,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn find_best_split(
        &self,
        _client: &ComputeClient<Self::Runtime>,
        hist: &[f64],
        cfg: &GainConfig,
        num_bin: u32,
        offset: i32,
        default_bin: u32,
        most_freq_bin: u32,
        skip_default_bin: bool,
        na_as_missing: bool,
        run_forward: bool,
        sum_gradient: f64,
        sum_hessian: f64,
        num_data: i32,
    ) -> Result<SplitInfo, ComputeError> {
        // R2: native f64 scan — bit-identical to the single-unit find_best_split_
        // kernel, without the per-(feature,leaf) cubecl launch. The cubecl path
        // stays in find_best_split_cpu for kernel-parity / ROCm-mirror tests.
        kernels::split::find_best_split_cpu_native(
            hist,
            cfg,
            num_bin,
            offset,
            default_bin,
            most_freq_bin,
            skip_default_bin,
            na_as_missing,
            run_forward,
            sum_gradient,
            sum_hessian,
            num_data,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn data_partition(
        &self,
        _client: &ComputeClient<Self::Runtime>,
        bins: &[u32],
        num_bin: u32,
        min_bin: u32,
        max_bin: u32,
        threshold: u32,
        most_freq_bin: u32,
    ) -> Result<(Vec<u32>, usize), ComputeError> {
        // R2: native u32 routing + stable gather — bit-identical to the kernel path.
        kernels::partition::data_partition_cpu_native(
            bins,
            num_bin,
            min_bin,
            max_bin,
            threshold,
            most_freq_bin,
        )
    }

    fn subtract_histograms(
        &self,
        _client: &ComputeClient<Self::Runtime>,
        parent: &[f64],
        child: &[f64],
    ) -> Result<Vec<f64>, ComputeError> {
        // R2: native element-wise parent − child — bit-identical to the kernel path.
        kernels::subtract::subtract_histograms_cpu_native(parent, child)
    }

    // CPU batched split: 260608-mc5 Task-3 DECISION = keep the NATIVE per-feature
    // path (the `Backend::find_best_splits_batched` trait default, which calls
    // `self.find_best_split` == `find_best_split_cpu_native` per feature). The merge
    // initially routed CpuBackend through `find_best_splits_batched_fused_f64_on`
    // (the same fused cubecl kernel the GPU uses), but a measured bench_train run on
    // this HEAD showed a MATERIAL CPU regression — the cubecl-cpu per-leaf launch
    // dispatch dominates even when batched into ONE launch per leaf:
    //   fused cubecl-cpu  vs  native (same HEAD, R2-equivalent):
    //     small  223.92ms vs  42.86ms  (~5.2x slower)
    //     medium 618.49ms vs 256.17ms  (~2.4x slower)
    //     large    1.76s  vs 828.95ms  (~2.1x slower)
    // (same root cause R2/260608-jyl found: the cubecl-cpu launch fixed cost, not
    // the arithmetic). CLAUDE.md non-negotiable #2 forbids shipping a silent CPU
    // slowdown, so the CpuBackend override is intentionally NOT defined here — the
    // native trait default applies. The GPU `RocmBackend` KEEPS the fused override
    // (one launch per leaf on gfx1100, f64 bit-exact), and the shared
    // `split_scan_body` helper (THE MERGE: one source of the split math) stays for
    // BOTH paths regardless. See the 260608-mc5 SUMMARY for the full measurement.
    //
    // The fused launcher `find_best_splits_batched_fused_f64_on` is generic over R,
    // so it remains available for the cubecl-cpu runtime via the oracle three-way
    // bit-exact gate (`kernel_parity_fused_equals_per_feature_and_native`) — the
    // merge is PROVEN bit-exact on cpu even though it is not the production path.
}

/// The device-resident binned dataset cached inside [`RocmBackend`] (260608-nn7
/// L1). Holds the ONE concatenated feature-column bin buffer's device `Handle`
/// (feature-major, length `num_features * num_data`) plus the dimensions needed to
/// index it (`f * num_data + row`). `Handle` is cheaply clonable (ref-counted), so
/// "residency" = hold this Handle across all per-leaf launches within one train.
#[cfg(feature = "rocm")]
#[derive(Debug, Clone)]
struct ResidentBins {
    /// Concatenated feature-major bin columns: feature `f`'s row `r` is at
    /// `f * num_data + r`. Uploaded ONCE per train.
    handle: cubecl::server::Handle,
    num_features: usize,
    num_data: usize,
}

/// The ROCm/HIP GPU backend (opt-in `rocm` feature) — dispatches every hot-path op
/// to the cubecl-hip runtime running the **f64** kernels on the local gfx1100.
///
/// 260608-nn7 (L1): the backend now carries interior-mutable device state — a
/// `RefCell<Option<ResidentBins>>` cache of the binned feature columns uploaded ONCE
/// per train. The learner holds `&B` (shared ref) and the trait methods take
/// `&self`, so the cache MUST be behind interior mutability (RefCell), NOT a
/// `&mut self` signature change. The single-threaded train loop makes the RefCell
/// borrow safe. Because a `RefCell` is not `Copy`, this type no longer derives
/// `Copy` (it did before nn7); `CpuBackend` stays the stateless unit struct.
#[cfg(feature = "rocm")]
#[derive(Debug)]
pub struct RocmBackend {
    /// The device-resident binned dataset, populated ONCE per train by
    /// [`upload_resident_bins`](Backend::upload_resident_bins) and read by the
    /// per-leaf [`build_leaf_histograms_raw`](Backend::build_leaf_histograms_raw)
    /// override. `None` until the first upload (defensive fallback to the per-leaf
    /// host-gather path).
    resident_bins: std::cell::RefCell<Option<ResidentBins>>,
    /// 260608-p90: the device-handle slot mirror, indexed by host `HistogramPool`
    /// slot id. `resident_pool[slot]` holds the fixed+compacted f64 histogram `Handle`
    /// for whichever leaf currently owns that slot, or `None` when the slot is empty.
    /// The learner issues build/subtract/move/scan ops here at the SAME call sites
    /// (with the SAME slot ids) it drives the host pool, so this mirror tracks the
    /// host pool's slot→leaf map exactly (T-p90-02). Like `resident_bins`, the
    /// single-threaded train loop makes the RefCell borrow safe (the nn7 rationale).
    resident_pool: std::cell::RefCell<Vec<Option<cubecl::server::Handle>>>,
    /// 260608-p90: test-only toggle to FORCE the host path on RocmBackend (so the
    /// resident==host tree-equivalence test can grow the SAME f32-atomic-built tree
    /// through the host read-back/subtract/scan chain). `true` (the default) reports
    /// `resident_pool_supported() == true`; `false` forces the host path. Set only by
    /// the test-only [`with_resident`](RocmBackend::with_resident) constructor.
    resident_enabled: bool,
}

#[cfg(feature = "rocm")]
impl Default for RocmBackend {
    fn default() -> Self {
        Self {
            resident_bins: std::cell::RefCell::new(None),
            resident_pool: std::cell::RefCell::new(Vec::new()),
            // Production default: the device-resident pool is enabled.
            resident_enabled: true,
        }
    }
}

#[cfg(feature = "rocm")]
impl RocmBackend {
    /// TEST-ONLY constructor (260608-p90): build a RocmBackend that REPORTS
    /// `resident_pool_supported() == enabled`. The resident==host tree-equivalence
    /// test grows the SAME corpus twice on a RocmBackend — once with `with_resident(true)`
    /// (the resident chain) and once with `with_resident(false)` (forcing the host
    /// read-back/subtract/scan path) — and asserts the two trees match within ~1e-6.
    /// The same f32-atomic RAW build runs in both cases; only the build→fix→compact→
    /// subtract→scan ROUTING differs. Not used on the production path
    /// ([`Default`] enables residency).
    pub fn with_resident(enabled: bool) -> Self {
        Self {
            resident_bins: std::cell::RefCell::new(None),
            resident_pool: std::cell::RefCell::new(Vec::new()),
            resident_enabled: enabled,
        }
    }
}

#[cfg(feature = "rocm")]
impl Backend for RocmBackend {
    type Runtime = runtime::RocmRuntime;

    fn construct_histograms(
        &self,
        client: &ComputeClient<Self::Runtime>,
        binned: &[u32],
        ordered_gradients: &[f32],
        ordered_hessians: &[f32],
        num_bin: u32,
    ) -> Result<Vec<f64>, ComputeError> {
        // GPU-fast path (kt8): parallel f32-atomic accumulation (one unit per row,
        // all lanes busy) instead of the single-unit f64 fold. This moves the GPU
        // path to the ~1e-6 ROCm gate (f32 atomics, nondeterministic add order) —
        // by design the GPU's contract; the cpu anchor stays bit-exact.
        //
        // NOTE (f8u, eo5 Finding #2): the LDS-privatized sub-histogram kernel
        // (`construct_histograms_lds_f32_on`) is proven correct (exact vs naive on
        // integer data, <1e-5 rel vs the f64 anchor) and ~4–4.6× faster under high
        // contention on gfx1100, but is NOT wired here. Two reasons: (1) it would
        // give this path yet another f32 accumulation order, and (2) the would-be
        // gating test `learner_parity_resident_equals_host_tree_on_hip` is itself
        // PRE-EXISTING FLAKY (~4/6 runs fail on the unchanged naive path — the naive
        // atomic's nondeterministic f32 order puts leaf 11's output on the 1e-6
        // knife-edge vs the resident chain; DEF-f8u-01), so non-regression cannot be
        // cleanly verified. Wiring it live wants the resident/batched BUILD path
        // LDS-ified too (one shared accumulation order) — the larger Finding #2
        // follow-up. Until then the kernel ships as an available, tested primitive.
        kernels::histogram::construct_histograms_parallel_f32_on(
            client,
            binned,
            ordered_gradients,
            ordered_hessians,
            num_bin,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn find_best_split(
        &self,
        client: &ComputeClient<Self::Runtime>,
        hist: &[f64],
        cfg: &GainConfig,
        num_bin: u32,
        offset: i32,
        default_bin: u32,
        most_freq_bin: u32,
        skip_default_bin: bool,
        na_as_missing: bool,
        run_forward: bool,
        sum_gradient: f64,
        sum_hessian: f64,
        num_data: i32,
    ) -> Result<SplitInfo, ComputeError> {
        kernels::split::find_best_split_f64_on(
            client,
            hist,
            cfg,
            num_bin,
            offset,
            default_bin,
            most_freq_bin,
            skip_default_bin,
            na_as_missing,
            run_forward,
            sum_gradient,
            sum_hessian,
            num_data,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn data_partition(
        &self,
        client: &ComputeClient<Self::Runtime>,
        bins: &[u32],
        num_bin: u32,
        min_bin: u32,
        max_bin: u32,
        threshold: u32,
        most_freq_bin: u32,
    ) -> Result<(Vec<u32>, usize), ComputeError> {
        kernels::partition::data_partition_on(
            client,
            bins,
            num_bin,
            min_bin,
            max_bin,
            threshold,
            most_freq_bin,
        )
    }

    fn subtract_histograms(
        &self,
        client: &ComputeClient<Self::Runtime>,
        parent: &[f64],
        child: &[f64],
    ) -> Result<Vec<f64>, ComputeError> {
        kernels::subtract::subtract_histograms_f64_on(client, parent, child)
    }

    /// GPU override (260608-nn7 L1): upload the binned feature columns to the device
    /// ONCE per train and cache the device `Handle` in `self.resident_bins` (interior
    /// mutability). The columns are concatenated feature-major into ONE buffer
    /// (`f * num_data + row`) so a single resident `Handle` covers every feature.
    /// Per-leaf [`build_leaf_histograms_raw`](Backend::build_leaf_histograms_raw)
    /// then gathers leaf rows ON DEVICE from this buffer, eliminating the per-leaf
    /// `[num_features × rows]` host bin re-upload.
    fn upload_resident_bins(
        &self,
        client: &ComputeClient<Self::Runtime>,
        feature_bins: &[&[u32]],
    ) {
        let num_features = feature_bins.len();
        if num_features == 0 {
            *self.resident_bins.borrow_mut() = None;
            return;
        }
        let num_data = feature_bins[0].len();
        // Concatenate every feature column feature-major into one host buffer, then
        // upload ONCE. (All columns share num_data — the learner validates this.)
        let mut concat: Vec<u32> = Vec::with_capacity(num_features * num_data);
        for &col in feature_bins {
            concat.extend_from_slice(col);
        }
        // `as_bytes` is `CubeElement::as_bytes` (the same call the histogram launchers
        // use); the trait must be in scope to name it.
        use cubecl::prelude::CubeElement;
        let handle = client.create_from_slice(u32::as_bytes(&concat));
        *self.resident_bins.borrow_mut() = Some(ResidentBins {
            handle,
            num_features,
            num_data,
        });
    }

    /// GPU override (part 3 + 260608-nn7 L1): build ALL features' RAW histograms for
    /// this leaf in ONE batched kernel launch. When the device-resident bin cache is
    /// populated (the L1 path), the kernel gathers leaf rows ON DEVICE from the
    /// resident column buffer + a small per-leaf `leaf_rows` upload — the per-leaf
    /// `[num_features × rows]` host bin upload is gone. Falls back to the host-gather
    /// batched launcher if the cache is empty (defensive). Collapses the per-feature
    /// construct launches to one per leaf. f32 atomics ⇒ the ~1e-6 ROCm gate.
    #[allow(clippy::too_many_arguments)]
    fn build_leaf_histograms_raw(
        &self,
        client: &ComputeClient<Self::Runtime>,
        feature_bins: &[&[u32]],
        _num_bins: &[u32],
        slot_off: &[usize],
        slot_len: usize,
        leaf_rows: &[u32],
        gradients: &[f32],
        hessians: &[f32],
    ) -> Result<Vec<f64>, ComputeError> {
        // L1 device-resident path: gather on device from the cached column buffer.
        if let Some(resident) = self.resident_bins.borrow().as_ref() {
            return kernels::histogram::build_leaf_histograms_resident_f32_on(
                client,
                resident.handle.clone(),
                resident.num_features,
                resident.num_data,
                slot_off,
                slot_len,
                leaf_rows,
                gradients,
                hessians,
            );
        }
        // Defensive fallback: no resident cache (upload_resident_bins not called) —
        // the original per-leaf host-gather batched launcher.
        kernels::histogram::build_leaf_histograms_batched_f32_on(
            client,
            feature_bins,
            slot_off,
            slot_len,
            leaf_rows,
            gradients,
            hessians,
        )
    }

    /// GPU override (260608-mc5 THE COLLAPSE): find ALL spine features' best splits
    /// for this leaf in ONE fused launch
    /// ([`kernels::split::find_best_splits_batched_fused_f64_on`]) instead of the
    /// per-feature loop-of-launches — `CubeCount::Static(num_feats,1,1)`, cube `f`
    /// scans only its `[slot_off[f], slot_off[f]+2*num_bin[f])` region. Input order
    /// is preserved (T-lsx-01); the f64 result is bit-identical to the per-feature
    /// loop AND bit-exact to the CPU anchor on gfx1100 (`max_abs_diff=0`).
    fn find_best_splits_batched(
        &self,
        client: &ComputeClient<Self::Runtime>,
        buf: &[f64],
        feats: &[BatchedSplitFeature],
        cfg: &GainConfig,
        sum_gradient: f64,
        sum_hessian: f64,
        num_data: i32,
    ) -> Result<Vec<SplitInfo>, ComputeError> {
        kernels::split::find_best_splits_batched_fused_f64_on(
            client,
            buf,
            feats,
            cfg,
            sum_gradient,
            sum_hessian,
            num_data,
        )
    }

    // ---- 260608-p90: device-resident histogram-pool overrides ----

    fn resident_pool_supported(&self) -> bool {
        self.resident_enabled
    }

    /// Clear + resize the device-handle slot mirror for a new tree. `num_slots` is the
    /// host pool's `cache_size`; `slot_len` is informational (the Handles carry their
    /// own length). Drops every prior Handle (releasing device memory).
    fn reset_resident_pool(&self, num_slots: usize, _slot_len: usize) {
        let mut mirror = self.resident_pool.borrow_mut();
        mirror.clear();
        mirror.resize_with(num_slots, || None);
    }

    /// Build ONE leaf's histogram device-resident (build → widen → fix → compact) via
    /// the oib `build_fix_compact_resident_f64_on` chain and STORE the returned
    /// `(Handle, len)` into mirror slot `slot` (dropping any prior Handle). Falls back
    /// to a typed error if the resident bin cache is empty (defensive — the learner
    /// always uploads before the growth loop when eligible).
    #[allow(clippy::too_many_arguments)]
    fn build_resident_leaf(
        &self,
        client: &ComputeClient<Self::Runtime>,
        slot: usize,
        _feature_bins: &[&[u32]],
        _num_bins: &[u32],
        slot_off: &[usize],
        slot_len: usize,
        leaf_rows: &[u32],
        gradients: &[f32],
        hessians: &[f32],
        fix_feats: &[(usize, u32, i32, u32)],
        sum_gradient: f64,
        sum_hessian: f64,
    ) -> Result<(), ComputeError> {
        let resident = self.resident_bins.borrow();
        let Some(resident) = resident.as_ref() else {
            return Err(ComputeError::Runtime {
                detail: "build_resident_leaf: resident bin cache empty (upload_resident_bins not \
                         called)"
                    .to_string(),
            });
        };
        let (handle, len) = kernels::histogram::build_fix_compact_resident_f64_on(
            client,
            resident.handle.clone(),
            resident.num_features,
            resident.num_data,
            slot_off,
            slot_len,
            leaf_rows,
            gradients,
            hessians,
            fix_feats,
            sum_gradient,
            sum_hessian,
        )?;
        debug_assert_eq!(len, slot_len, "resident leaf handle length");
        let mut mirror = self.resident_pool.borrow_mut();
        if slot >= mirror.len() {
            mirror.resize_with(slot + 1, || None);
        }
        mirror[slot] = Some(handle);
        Ok(())
    }

    /// Move the resident Handle from `src_slot` to `dst_slot`, mirroring the host
    /// `HistogramPool::move_` (the slot reassignment that hands the parent's buffer to
    /// the larger child). `src_slot` is left empty.
    fn move_resident(&self, src_slot: usize, dst_slot: usize) {
        let mut mirror = self.resident_pool.borrow_mut();
        let max = src_slot.max(dst_slot);
        if max >= mirror.len() {
            mirror.resize_with(max + 1, || None);
        }
        let moved = mirror[src_slot].take();
        mirror[dst_slot] = moved;
    }

    /// Derive the larger child resident: `parent_slot` Handle − `smaller_slot` Handle
    /// → `larger_slot` Handle, on device, no read-back. The derived larger child is
    /// NOT re-FixHistogram'd (non-negotiable #3).
    fn subtract_resident(
        &self,
        client: &ComputeClient<Self::Runtime>,
        parent_slot: usize,
        smaller_slot: usize,
        larger_slot: usize,
        slot_len: usize,
    ) -> Result<(), ComputeError> {
        let (parent_h, smaller_h) = {
            let mirror = self.resident_pool.borrow();
            let parent_h = mirror.get(parent_slot).and_then(|h| h.clone()).ok_or_else(|| {
                ComputeError::Runtime {
                    detail: "subtract_resident: parent slot is empty".to_string(),
                }
            })?;
            let smaller_h = mirror.get(smaller_slot).and_then(|h| h.clone()).ok_or_else(|| {
                ComputeError::Runtime {
                    detail: "subtract_resident: smaller slot is empty".to_string(),
                }
            })?;
            (parent_h, smaller_h)
        };
        let derived = kernels::subtract::subtract_histograms_f64_from_handles_on(
            client, parent_h, smaller_h, slot_len,
        )?;
        let mut mirror = self.resident_pool.borrow_mut();
        if larger_slot >= mirror.len() {
            mirror.resize_with(larger_slot + 1, || None);
        }
        mirror[larger_slot] = Some(derived);
        Ok(())
    }

    /// Scan slot `slot`'s resident Handle for every spine feature in ONE fused launch
    /// (the Handle-consuming `find_best_splits_batched_fused_f64_from_handle_on`),
    /// reading back only the SplitInfo cells. Errors if the slot is empty (defensive).
    #[allow(clippy::too_many_arguments)]
    fn scan_resident_leaf(
        &self,
        client: &ComputeClient<Self::Runtime>,
        slot: usize,
        slot_len: usize,
        feats: &[BatchedSplitFeature],
        cfg: &GainConfig,
        sum_gradient: f64,
        sum_hessian: f64,
        num_data: i32,
    ) -> Result<Vec<SplitInfo>, ComputeError> {
        let handle = {
            let mirror = self.resident_pool.borrow();
            mirror.get(slot).and_then(|h| h.clone()).ok_or_else(|| ComputeError::Runtime {
                detail: "scan_resident_leaf: slot is empty".to_string(),
            })?
        };
        kernels::split::find_best_splits_batched_fused_f64_from_handle_on(
            client, handle, slot_len, feats, cfg, sum_gradient, sum_hessian, num_data,
        )
    }

    /// 260608-t3t: FUSED build+fix+compact+scan for a directly-built leaf. Reads the
    /// resident bin cache, runs the SINGLE fused-kernel launch (build → fix →
    /// compact → scan), STORES the returned fixed+compacted f64 Handle into mirror
    /// slot `slot` (so `subtract_resident` finds it as the parent), and returns the
    /// per-feature SplitInfos. Errors if the resident bin cache is empty (defensive).
    #[allow(clippy::too_many_arguments)]
    fn build_fix_scan_resident(
        &self,
        client: &ComputeClient<Self::Runtime>,
        slot: usize,
        slot_off: &[usize],
        slot_len: usize,
        leaf_rows: &[u32],
        gradients: &[f32],
        hessians: &[f32],
        feats: &[BatchedSplitFeature],
        scan_active: &[bool],
        cfg: &GainConfig,
        sum_gradient_raw: f64,
        sum_hessian_raw: f64,
        num_data: i32,
    ) -> Result<Vec<SplitInfo>, ComputeError> {
        let resident = self.resident_bins.borrow();
        let Some(resident) = resident.as_ref() else {
            return Err(ComputeError::Runtime {
                detail: "build_fix_scan_resident: resident bin cache empty (upload_resident_bins \
                         not called)"
                    .to_string(),
            });
        };
        let (handle, len, splits) = kernels::histogram::build_fix_scan_resident_f64_on(
            client,
            resident.handle.clone(),
            resident.num_features,
            resident.num_data,
            slot_off,
            slot_len,
            leaf_rows,
            gradients,
            hessians,
            feats,
            scan_active,
            cfg,
            sum_gradient_raw,
            sum_hessian_raw,
            num_data,
        )?;
        debug_assert_eq!(len, slot_len, "fused resident leaf handle length");
        let mut mirror = self.resident_pool.borrow_mut();
        if slot >= mirror.len() {
            mirror.resize_with(slot + 1, || None);
        }
        mirror[slot] = Some(handle);
        Ok(splits)
    }
}
