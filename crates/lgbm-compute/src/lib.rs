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
            let hist =
                self.construct_histograms(client, &ord_bins, &ord_g, &ord_h, num_bins[fpos])?;
            let cells = 2 * num_bins[fpos] as usize;
            out[slot_off[fpos]..slot_off[fpos] + cells].copy_from_slice(&hist);
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
#[derive(Debug, Default)]
pub struct RocmBackend {
    /// The device-resident binned dataset, populated ONCE per train by
    /// [`upload_resident_bins`](Backend::upload_resident_bins) and read by the
    /// per-leaf [`build_leaf_histograms_raw`](Backend::build_leaf_histograms_raw)
    /// override. `None` until the first upload (defensive fallback to the per-leaf
    /// host-gather path).
    resident_bins: std::cell::RefCell<Option<ResidentBins>>,
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
}
