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

/// A feature column's per-row bin indices, stored in the NARROWEST unsigned type
/// for its `num_bin` (spike 004 — columnar narrow bins). Faithful to C++
/// `DenseBin<uint8_t>` / `<uint16_t>` / `<uint32_t>`, which picks the narrowest
/// bin type per feature so the hot histogram gather+fold is cache-dense.
///
/// Defined HERE (the lowest crate, which owns the [`Backend`] trait + the hot
/// fold) and re-exported from `lgbm-treelearner` (`lgbm-treelearner` depends on
/// `lgbm-compute`, NOT vice versa — putting this in `lgbm-treelearner` and
/// importing it here would be a dependency CYCLE).
///
/// The bin VALUE is unchanged — only stored narrower and widened at read time —
/// so the f64 histogram fold order + values are byte-identical and the tree stays
/// bit-exact. The HOT CPU fold reads the narrow type DIRECTLY per-width
/// (monomorphic match, no per-element width branch in the row loop); COLD readers
/// (partition, bagging, validation, scatter, GPU upload) go through the widening
/// [`bin`](BinColumn::bin) / [`iter_u32`](BinColumn::iter_u32) /
/// [`to_u32_vec`](BinColumn::to_u32_vec) accessors.
#[derive(Clone, Debug, PartialEq)]
pub enum BinColumn {
    /// `num_bin <= 256` — the default `max_bin=255` common case (carries the win).
    U8(Vec<u8>),
    /// `256 < num_bin <= 65536`.
    U16(Vec<u16>),
    /// `num_bin > 65536`.
    U32(Vec<u32>),
}

impl BinColumn {
    /// Build the narrowest-typed column for `num_bin`: `u8` if `num_bin <= 256`,
    /// `u16` if `num_bin <= 65536`, else `u32`. Width is selected by `num_bin`
    /// (the type's capacity), NOT by the observed max value — so
    /// `new(vec![0,1], 256)` is `U8` even though the max is 1, mirroring C++
    /// `DenseBin<VAL_T>` (the bin TYPE is fixed by the feature's bin count).
    ///
    /// The once-per-train bin-range gate (the authoritative `bin < num_bin` VALUE
    /// check, `lgbm-treelearner` learner.rs) runs upstream of any tree growth, so
    /// width selection only needs the cast to be loss-free: a `debug_assert!`
    /// guards that each bin FITS the chosen narrow type (the truncation /
    /// memory-safety concern, T-ruz-01), which always holds because the type is
    /// sized to `num_bin`'s capacity. We do NOT assert `bin < num_bin` here — that
    /// is the gate's job, and a deliberately-edge value equal to `num_bin` is a
    /// valid input to construct (it is rejected later by the gate, not by `new`).
    #[must_use]
    pub fn new(bins: Vec<u32>, num_bin: u32) -> Self {
        if num_bin <= 256 {
            BinColumn::U8(
                bins.into_iter()
                    .map(|b| {
                        debug_assert!(b <= u32::from(u8::MAX), "bin {b} does not fit u8 width");
                        b as u8
                    })
                    .collect(),
            )
        } else if num_bin <= 65536 {
            BinColumn::U16(
                bins.into_iter()
                    .map(|b| {
                        debug_assert!(b <= u32::from(u16::MAX), "bin {b} does not fit u16 width");
                        b as u16
                    })
                    .collect(),
            )
        } else {
            BinColumn::U32(bins)
        }
    }

    /// Read row `row`'s bin index, WIDENED to `u32` (the cold-reader accessor).
    /// Identical to the prior `Vec<u32>` index read for every variant.
    #[inline]
    #[must_use]
    pub fn bin(&self, row: usize) -> u32 {
        match self {
            BinColumn::U8(v) => u32::from(v[row]),
            BinColumn::U16(v) => u32::from(v[row]),
            BinColumn::U32(v) => v[row],
        }
    }

    /// The number of rows in the column.
    #[inline]
    #[must_use]
    pub fn len(&self) -> usize {
        match self {
            BinColumn::U8(v) => v.len(),
            BinColumn::U16(v) => v.len(),
            BinColumn::U32(v) => v.len(),
        }
    }

    /// Whether the column has no rows.
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Re-gather the column at `rows` (each a global row id), PRESERVING the same
    /// width as `self` — the bagging-subset gather keeps the narrow storage.
    #[must_use]
    pub fn gather(&self, rows: &[u32]) -> BinColumn {
        match self {
            BinColumn::U8(v) => BinColumn::U8(rows.iter().map(|&r| v[r as usize]).collect()),
            BinColumn::U16(v) => BinColumn::U16(rows.iter().map(|&r| v[r as usize]).collect()),
            BinColumn::U32(v) => BinColumn::U32(rows.iter().map(|&r| v[r as usize]).collect()),
        }
    }

    /// Widen the WHOLE column to a `Vec<u32>` (cold; used only by the GPU upload
    /// and parity-test asserts). Round-trips: `new(v, nb).to_u32_vec() == v`.
    #[must_use]
    pub fn to_u32_vec(&self) -> Vec<u32> {
        match self {
            BinColumn::U8(v) => v.iter().map(|&b| u32::from(b)).collect(),
            BinColumn::U16(v) => v.iter().map(|&b| u32::from(b)).collect(),
            BinColumn::U32(v) => v.clone(),
        }
    }

    /// Iterate the column widened to `u32` (cold scans that don't need the tight
    /// monomorphic loop). Boxes the per-variant iterator so all three arms share one
    /// return type — do NOT use this on a hot per-row path (the boxed dynamic
    /// dispatch is a measured small-row regression); use a direct `match self` over
    /// the narrow slice, or [`first_ge`](BinColumn::first_ge), instead.
    pub fn iter_u32(&self) -> impl Iterator<Item = u32> + '_ {
        let it: Box<dyn Iterator<Item = u32> + '_> = match self {
            BinColumn::U8(v) => Box::new(v.iter().map(|&b| u32::from(b))),
            BinColumn::U16(v) => Box::new(v.iter().map(|&b| u32::from(b))),
            BinColumn::U32(v) => Box::new(v.iter().copied()),
        };
        it
    }

    /// Return the FIRST element `>= bound` (widened to `u32`), or `None` if every
    /// element is `< bound`. This is the allocation-free, MONOMORPHIC per-width scan
    /// the once-per-train bin-range gate uses (the `bin < num_bin` VALUE check) — it
    /// dispatches on the width ONCE then runs a tight slice loop per arm, avoiding
    /// the boxed [`iter_u32`](BinColumn::iter_u32) dynamic dispatch on the hot
    /// per-row path (spike 004 small-row regression fix).
    #[inline]
    #[must_use]
    pub fn first_ge(&self, bound: u32) -> Option<u32> {
        match self {
            BinColumn::U8(v) => v
                .iter()
                .map(|&b| u32::from(b))
                .find(|&b| b >= bound),
            BinColumn::U16(v) => v
                .iter()
                .map(|&b| u32::from(b))
                .find(|&b| b >= bound),
            BinColumn::U32(v) => v.iter().copied().find(|&b| b >= bound),
        }
    }
}

/// Fold ONE feature's narrow bin column into the pre-zeroed histogram `h`
/// (`len == 2 * num_bin`): ascending `leaf_rows`, grad at `bin<<1` / hess at `+1`,
/// f32-read → f64-accumulate. The [`BinColumn`] width `match` is OUTSIDE the row
/// loop (monomorphic arms — spike 004), and the fold ORDER is byte-identical to
/// `construct_histograms_cpu_native`, so this is the single bit-exact fold body used
/// by BOTH the serial and the parallel build paths.
///
/// Precondition (caller-established once per train, T-04-01 relocation): every
/// `bins[row] < num_bin`. Debug-asserted; release trusts the upstream gate.
#[inline]
fn fold_one_feature(bins: &BinColumn, leaf_rows: &[u32], ord_g: &[f32], ord_h: &[f32], h: &mut [f64]) {
    macro_rules! fold {
        ($v:expr) => {
            for (k, &row) in leaf_rows.iter().enumerate() {
                let bin = $v[row as usize] as usize;
                debug_assert!(bin * 2 + 1 < h.len(), "bin out of range — caller must establish bin < num_bin once per train (T-04-01)");
                h[bin * 2] += f64::from(ord_g[k]);
                h[bin * 2 + 1] += f64::from(ord_h[k]);
            }
        };
    }
    match bins {
        BinColumn::U8(v) => fold!(v),
        BinColumn::U16(v) => fold!(v),
        BinColumn::U32(v) => fold!(v),
    }
}

/// Build the concatenated stride-2 per-feature histogram buffer (feature `f` occupies
/// `[slot_off[f], slot_off[f] + 2*num_bins[f])`). `parallel` (spike 005 / R4) selects
/// rayon-over-features — each feature folds its OWN histogram Vec from the shared
/// read-only `ord_g`/`ord_h`, then a sequential copy assembles `out` — versus the
/// serial reused-scratch path for small leaves (rayon dispatch overhead crushes tiny
/// per-feature folds). Both paths call [`fold_one_feature`] with the SAME fold order
/// and write disjoint `out` regions, so the result is BYTE-IDENTICAL regardless of
/// `parallel` or thread scheduling (proven by `build_histograms_parallel_equals_serial`)
/// — the bit-exact merge gate holds for the multi-threaded anchor.
fn build_histograms_into(
    feature_bins: &[&BinColumn],
    num_bins: &[u32],
    slot_off: &[usize],
    slot_len: usize,
    leaf_rows: &[u32],
    ord_g: &[f32],
    ord_h: &[f32],
    parallel: bool,
) -> Vec<f64> {
    let mut out = vec![0.0f64; slot_len];
    if parallel {
        use rayon::prelude::*;
        // NOTE (spike 011, INVALIDATED): this per-feature `Vec<f64>` intermediate
        // looks like a flattenable `Vec<Vec<f64>>` wart, but it is LOAD-BEARING.
        // Each rayon task folds into its OWN cache-hot private buffer, then a
        // single sequential `copy_from_slice` assembles `out`. Replacing it with
        // a scatter (16 threads folding scattered writes directly into disjoint
        // sub-slices of one shared `out`) regressed the threshold-sized leaves
        // 13–21% on a 16-core box (cache-coherence/false-sharing traffic > the
        // alloc+copy it removed) and was null on huge leaves. Keep the private
        // accumulators. See `.planning/spikes/011-parallel-build-scatter/`.
        let hists: Vec<Vec<f64>> = (0..feature_bins.len())
            .into_par_iter()
            .map(|fpos| {
                let mut h = vec![0.0f64; 2 * num_bins[fpos] as usize];
                fold_one_feature(feature_bins[fpos], leaf_rows, ord_g, ord_h, &mut h);
                h
            })
            .collect();
        for (fpos, h) in hists.into_iter().enumerate() {
            out[slot_off[fpos]..slot_off[fpos] + h.len()].copy_from_slice(&h);
        }
    } else {
        let max_cells = num_bins.iter().copied().max().map_or(0, |m| 2 * m as usize);
        let mut scratch = vec![0.0f64; max_cells];
        for (fpos, &bins) in feature_bins.iter().enumerate() {
            let cells = 2 * num_bins[fpos] as usize;
            for c in scratch[..cells].iter_mut() {
                *c = 0.0;
            }
            fold_one_feature(bins, leaf_rows, ord_g, ord_h, &mut scratch[..cells]);
            out[slot_off[fpos]..slot_off[fpos] + cells].copy_from_slice(&scratch[..cells]);
        }
    }
    out
}

/// The leaf-row count at/above which a per-leaf build parallelizes over features
/// (spike 005: parallel wins ≈≥12k rows; 16384 keeps small+medium serial with zero
/// regression while parallelizing the genuinely large leaves — large train −26%).
/// Override via `LGBM_PAR_THRESHOLD`.
fn par_build_threshold() -> usize {
    std::env::var("LGBM_PAR_THRESHOLD")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(16384)
}

/// The feature-count at/above which a per-leaf SPLIT SCAN parallelizes across
/// features (ONE rayon fork/join per leaf, amortized over all features).
///
/// Unlike the BUILD gate ([`par_build_threshold`], keyed on leaf ROWS) the scan
/// work scales with `num_features × num_bins`, NOT leaf rows — each per-feature
/// `find_best_split` reads a DISJOINT `buf` region and is row-count-independent.
/// So this gate is keyed on `feats.len()` (the simplest defensible scan-work
/// proxy; bin counts are near-uniform across a leaf's spine features, so feature
/// count alone separates the narrow/wide regimes — confirmed by the quick
/// 260620-9cp A/B). Below the threshold the serial loop runs verbatim (the same
/// per-feature dispatch overhead that regressed the unconditional BUILD path —
/// Spike 005 — would regress narrow leaves here). Override via
/// `LGBM_PAR_SCAN_THRESHOLD`.
///
/// DEFAULT (quick 260620-9cp): HONEST NULL — the parallel scan is kept available
/// (env-reachable, bit-exact-proven FORCED-ON) but is NOT the effective default,
/// so the threshold is set effectively-unreachable (`usize::MAX`). The A/B measured
/// on cubecl-cpu (warm, 3-run medians):
///   - narrow (10 feat): SCAN serial 31.7ms vs parallel 104.4ms (3.3× WORSE),
///     train-wall 156ms vs 287ms — fork/join overhead at 10 features dominates.
///   - wide (120 feat):  SCAN serial 353ms vs parallel 309ms (−13%, sign-stable),
///     BUT train-wall 1066ms vs 1184ms (+11%, sign-stable WORSE) — the per-leaf
///     scan fork/join CONTENDS with the already-rayon-parallel BUILD path
///     (BUILD_NS rose 438→520ms), so the isolated SCAN_NS win does NOT translate
///     to a warm train-wall win; it regresses overall.
/// Adoption criterion (wide train-wall gain AND no narrow regression) is NOT met,
/// so per the project's audit-before-wire value the serial loop stays the effective
/// default. Set `LGBM_PAR_SCAN_THRESHOLD=0` (or any small value) to force the
/// parallel path on for the bit-exact parity proof / future re-measurement.
fn par_scan_threshold() -> usize {
    std::env::var("LGBM_PAR_SCAN_THRESHOLD")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(usize::MAX)
}

/// The feature-count at/above which the directly-built (smaller/root) leaf's
/// per-feature `{build histogram → fix_histogram → compact → scan}` runs inside ONE
/// rayon region (the host f64 analog of the GPU `build_fix_scan_resident` fusion,
/// quick 260620-a48) instead of the two-step `build_leaf_histogram_into` +
/// `find_best_splits_batched`.
///
/// Keyed on `feats.len()` — the SAME scan-work proxy as [`par_scan_threshold`] —
/// because the contention this lever removes (quick 260620-9cp: a parallel scan
/// `par_iter` fighting the already-parallel build `par_iter`) scales with the
/// feature count, and narrow leaves were catastrophic in BOTH 8v4 and 9cp. Below the
/// threshold the leaf takes the byte-unchanged serial two-step path. Override via
/// `LGBM_UNIFIED_BFS_THRESHOLD`.
///
/// DEFAULT (quick 260620-a48): CONDITIONAL WIN at `feats.len() >= 100`. The unified
/// region removes the quick-260620-9cp cross-region contention (a parallel scan
/// `par_iter` fighting the already-parallel build `par_iter`) by fusing
/// build+fix+compact+scan for each spine feature into ONE rayon fork/join, keeping
/// each feature's histogram cache-hot in its building thread through fix and scan.
/// The A/B measured on cubecl-cpu (warm, 3-run medians, train-wall = decisive metric):
///   - WIDE (120 feat, 20k rows): fused build+scan 779.7ms→569.7ms (−27%), warm
///     train-wall A 810ms vs B 755ms (−6.8%, sign-stable, no run overlap across all 3
///     runs at 100 AND 120 feat). The phase counters confirm the mechanism: the
///     two-step path forks/joins TWICE (build then scan), the unified path ONCE.
///   - NARROW (10 feat, 20k rows): fused build+scan 95.9ms→124.9ms (+30%), warm
///     train-wall A 169ms vs B 246ms (+45%, sign-stable WORSE) — the single rayon
///     fork/join overhead at 10 features dominates (the 8v4/9cp narrow-is-catastrophic
///     lesson). So narrow MUST stay serial two-step.
///   - SWEEP crossover: B regresses or is within-spread up to ~90 feat (feat=80 +0.6%,
///     feat=90 overlapping/sign-flips); B becomes sign-stable-below-A only at ≥100 feat
///     (feat=100 −5%, feat=120 −6%, neither overlapping A's spread).
/// Tuned default = 100: narrow/medium (≤90 feat) keep the byte-unchanged serial
/// two-step path (zero regression); genuinely wide leaves (≥100 feat) take the unified
/// region for the sign-stable −6% train-wall gain. Both adoption gates met (wide
/// sign-stable gain AND no narrow regression at the tuned threshold). Set
/// `LGBM_UNIFIED_BFS_THRESHOLD=0` to force the unified path on for the bit-exact parity
/// proof; set a large value (or `usize::MAX`) to force the serial two-step path.
/// See the 260620-a48 SUMMARY for the full A/B.
pub fn unified_bfs_threshold() -> usize {
    std::env::var("LGBM_UNIFIED_BFS_THRESHOLD")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(100)
}

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
    /// # Bin-range precondition (V5 / threat T-04-01, RELOCATED — spike-003b)
    /// The hot fold below is **branchless**: it reads `bins[row]` and folds it into
    /// `scratch[bin*2 (+1)]` with NO per-element `bin < num_bin` check. This is a
    /// CALLER-GUARANTEED PRECONDITION: every `feature_bins[fpos][row] <
    /// num_bins[fpos]` MUST hold. That invariant is established ONCE per train by the
    /// upstream bin-range gate in `lgbm-treelearner` `SerialTreeLearner::train`
    /// (`train_inner`, learner.rs:700-714), which iterates every feature column and
    /// every bin and rejects any `bin >= num_bin` with
    /// `TreeLearnerError::BinIndexOutOfRange` BEFORE any leaf is built. The feature
    /// columns are fixed for the whole train, so the amortized cost is O(rows) ONCE
    /// per train instead of O(leaf_rows) per build per iteration. This mirrors C++
    /// `dense_bin.hpp` (`ConstructHistogramInner`), which folds `data_[i]` directly
    /// with no per-element validation, trusting the binning invariant. Spike-003b
    /// proved ANY per-element check (early-return OR branchless clamp+OOB-flag)
    /// serializes the fold and regresses the 200k-row build ~3-8%; the branchless
    /// form wins both scales (-17% small / -4.5% large).
    ///
    /// # Errors
    /// The fused fold no longer returns `BinIndexOutOfRange` per element — there is
    /// no per-element check (see the precondition above; the production guarantee is
    /// the upstream once-per-train gate, and a `debug_assert!(bin < num_bin)` is the
    /// debug/test defense-in-depth that catches a violated precondition). The body
    /// has no fallible per-feature call; the `Result` is retained for the trait
    /// signature and a GPU override's fallible launch, and this default impl returns
    /// `Ok(out)`.
    #[allow(clippy::too_many_arguments)]
    fn build_leaf_histograms_raw(
        &self,
        _client: &ComputeClient<Self::Runtime>,
        feature_bins: &[&BinColumn],
        num_bins: &[u32],
        slot_off: &[usize],
        slot_len: usize,
        leaf_rows: &[u32],
        gradients: &[f32],
        hessians: &[f32],
    ) -> Result<Vec<f64>, ComputeError> {
        // SPIKE 003: gather the ordered gradients/hessians ONCE per leaf — they are
        // identical across every feature (only the bin column differs), so the prior
        // per-feature re-gather repeated this work `num_features` times. Mirrors C++
        // `ordered_gradients_`/`ordered_hessians_` reuse. Values + order unchanged ⇒
        // bit-exact.
        let r = leaf_rows.len();
        let mut ord_g: Vec<f32> = Vec::with_capacity(r);
        let mut ord_h: Vec<f32> = Vec::with_capacity(r);
        for &row in leaf_rows {
            ord_g.push(gradients[row as usize]);
            ord_h.push(hessians[row as usize]);
        }
        // SPIKE 003b: FUSE the per-feature bin gather into the fold. Read `bins[row]`
        // inline and fold directly into a REUSED per-feature hot scratch (sized to the
        // widest feature, <= 2*max_num_bin) — NOT `ord_bins` materialization, NOT a
        // per-feature alloc, and NOT a fold into the big multi-feature `out` buffer
        // (p0n proved folding into `out` cache-scatters and regresses large ~9%). The
        // fold is BRANCHLESS: no per-element bin check (see the precondition doc above),
        // only a `debug_assert!`. The f64 fold ORDER is byte-identical to
        // `construct_histograms_cpu_native` — ascending `leaf_rows`, grad at `bin<<1`,
        // hess at `+1`, f32-read -> f64-accumulate — so the bit-exact gate holds.
        //
        // SPIKE 004: the bin column is NARROW ([`BinColumn`], u8/u16/u32). Dispatch
        // on the width ONCE per feature (OUTSIDE the row loop) so each arm is a
        // MONOMORPHIC tight loop reading the narrow element directly — the
        // cache-density win lives here (no per-element width branch / accessor in the
        // hot loop). The fold ORDER and the `bin as usize * 2` index arithmetic are
        // IDENTICAL across arms and identical to the prior u32 path ⇒ bit-exact.
        // SPIKE 005 (R4): build each feature's histogram in parallel across rayon WHEN the
        // leaf is big enough to amortize task dispatch (>= par_build_threshold), else the
        // serial reused-scratch path. Both call the SAME `fold_one_feature` body with the
        // SAME ascending order into disjoint `out` regions ⇒ byte-identical result; the
        // per-feature independence makes it thread-count-deterministic (bit-exact gate
        // holds). The threshold protects small/medium leaves — unconditional parallel
        // regressed 2k-row train ~5× on rayon dispatch overhead.
        let parallel = r >= par_build_threshold();
        let out = build_histograms_into(
            feature_bins, num_bins, slot_off, slot_len, leaf_rows, &ord_g, &ord_h, parallel,
        );
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

    /// Whether [`upload_resident_bins`](Backend::upload_resident_bins) actually
    /// consumes its `&[&[u32]]` argument (spike 004). With narrow [`BinColumn`]
    /// storage, the learner must WIDEN each column to `u32` to call
    /// `upload_resident_bins`; that widening allocates `num_features` u32 Vecs ONCE
    /// per `train_inner`. On [`CpuBackend`] the upload is a no-op, so the learner
    /// SKIPS the widening entirely (gated on this returning `false`) — avoiding a
    /// per-tree allocation that has no effect. RocmBackend returns `true` so its
    /// resident u32 upload still receives the byte-identical column data.
    fn wants_resident_bins(&self) -> bool {
        false
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
        _feature_bins: &[&BinColumn],
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

    /// quick 260620-a48: UNIFIED host per-feature `{build → fix → compact → scan}` for
    /// the directly-built (smaller/root) leaf, run inside ONE rayon region — the host f64
    /// analog of [`build_fix_scan_resident`](Backend::build_fix_scan_resident).
    ///
    /// Each feature folds its OWN private histogram (cache-hot in the building thread),
    /// runs `fix_histogram` (RAW sums + `most_freq_bin` reconstruct) + `compact`
    /// (`offset` shift) IN PLACE on it, and — IF `scan_active[fpos]` — scans it, all
    /// WITHOUT a cross-region fork/join hand-off (the contention quick 260620-9cp found:
    /// a parallel scan `par_iter` fighting the parallel build `par_iter`). After the
    /// region a SERIAL ordered loop copies each private histogram into its disjoint
    /// `buf[slot_off..]` region (the COMPLETE leaf histogram for the subtract-derived
    /// larger child) and assembles the per-feature `Option<SplitInfo>` results in
    /// feature-index order.
    ///
    /// `all_feats[fpos]` carries every feature's `(slot_off, num_bin, offset,
    /// most_freq_bin, …)` (the SAME `BatchedSplitFeature` Pass-1 builds);
    /// build+fix+compact run for EVERY feature (the histogram must be COMPLETE for the
    /// subtract trick) while `scan_active[fpos]` selects which features are scanned (the
    /// spine subset that passed the caller's PASS-1 gates). `None` for non-scan-active
    /// features; the caller's serial cross-feature argmax merges these with the inline
    /// (categorical / monotone / extra-trees) branches in feature-index order.
    ///
    /// BIT-EXACT: each feature is independent — disjoint region, own ascending-`leaf_rows`
    /// fold, own ascending-bin `fix`/`compact`, own scan. Co-locating the three phases on
    /// one thread changes NEITHER per-feature op order NOR the (serial, feature-order)
    /// argmax ⇒ byte-identical to the two-step path (proven FORCED-ON via
    /// `LGBM_UNIFIED_BFS_THRESHOLD=0`). `sum_gradient`/`sum_hessian` are the RAW
    /// (un-bumped) leaf totals (Pitfall 2 — the `fix` operand).
    ///
    /// Default: typed error — only [`CpuBackend`] overrides this (the unified path is the
    /// CPU-only host analog; RocmBackend keeps `build_fix_scan_resident`). The learner's
    /// `unified_bfs` gate ANDs in `!resident_pool_supported()` so this is never reached on
    /// a GPU backend.
    ///
    /// # Errors
    /// [`ComputeError::Runtime`] (unsupported) on the default; on CpuBackend,
    /// [`ComputeError::LengthMismatch`] for an out-of-range feature region (ascending ⇒
    /// deterministic lowest-index error) or a propagated
    /// [`find_best_split`](Backend::find_best_split) error.
    #[allow(clippy::too_many_arguments)]
    fn build_fix_scan(
        &self,
        _client: &ComputeClient<Self::Runtime>,
        _buf: &mut [f64],
        _feature_bins: &[&BinColumn],
        _leaf_rows: &[u32],
        _gradients: &[f32],
        _hessians: &[f32],
        _all_feats: &[BatchedSplitFeature],
        _scan_active: &[bool],
        _cfg: &GainConfig,
        _sum_gradient: f64,
        _sum_hessian: f64,
        _num_data: i32,
    ) -> Result<Vec<Option<SplitInfo>>, ComputeError> {
        Err(ComputeError::Runtime {
            detail: "build_fix_scan: unified host build+fix+scan not supported on this backend"
                .to_string(),
        })
    }
}

/// The default cpu-runtime backend (the D-04 deterministic anchor, CMP-02).
///
/// Binds [`runtime::ActiveRuntime`] (cubecl-cpu under the default `cpu` feature)
/// and dispatches [`construct_histograms`](Backend::construct_histograms) to the
/// single-owner ordered f64 fold in [`kernels::histogram`].
/// quick-260620-8v4 opt-in gate for the additive 2-lane native split scan. Read
/// ONCE from `LGBM_SPLIT_2LANE` (`"1"` => on). OFF by default: the serial
/// [`kernels::split::find_best_split_cpu_native`] is the bit-exact source of truth
/// and the production path; the 2-lane variant is selected only for the explicit
/// A/B latency measurement (it produces byte-identical SplitInfos either way).
#[cfg(feature = "cpu")]
fn split_2lane_enabled() -> bool {
    use std::sync::OnceLock;
    static E: OnceLock<bool> = OnceLock::new();
    *E.get_or_init(|| std::env::var("LGBM_SPLIT_2LANE").map(|v| v == "1").unwrap_or(false))
}

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
        //
        // quick-260620-8v4: an OPT-IN additive 2-lane variant runs the REVERSE and
        // FORWARD passes on two rayon lanes (bit-identical winner — see
        // `find_best_split_cpu_native_2lane`). Gated behind `LGBM_SPLIT_2LANE=1` so
        // the DEFAULT path is the unchanged serial source of truth; the 2-lane path
        // is only selected for the explicit A/B measurement. Both paths produce
        // byte-identical SplitInfos (asserted by `split_2lane_equals_serial_*`).
        if split_2lane_enabled() {
            kernels::split::find_best_split_cpu_native_2lane(
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
        } else {
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

    // quick-260620-9cp (R4 split-scan lever): override the per-leaf SPLIT SCAN to
    // parallelize the per-feature loop with ONE rayon fork/join per leaf, amortized
    // across all features — replacing the trait-default serial `for f in feats`.
    //
    // BIT-EXACT SAFETY (the load-bearing invariant):
    //  - Each `find_best_split` reads a DISJOINT `buf[f.slot_off..end]` region and
    //    returns an INDEPENDENT `SplitInfo`; per-feature f64 accumulation order is
    //    UNTOUCHED ⇒ thread-count-deterministic.
    //  - `par_iter().map(...).collect::<Vec<_>>()` PRESERVES input order, so out[i]
    //    still corresponds to feats[i]. The caller's cross-feature argmax
    //    (`scan_leaf_histogram`, gain-then-smaller-feature tie-break) consumes the Vec
    //    in feature order ⇒ byte-identical to the serial loop ⇒ the bit-exact CPU
    //    f64 anchor gate holds (proven FORCED-ON via LGBM_PAR_SCAN_THRESHOLD=0).
    //
    // DETERMINISTIC ERROR ORDER: the `[slot_off, slot_off+2*num_bin)` range
    // validation is hoisted BEFORE the parallel map and walks features in ascending
    // index, returning the LOWEST-index offending feature's error — so a parallel
    // error race can NEVER change WHICH error surfaces (matches the serial loop,
    // which returns on the first offending feature). The map itself is then
    // infallible on the validated hist slices.
    //
    // GATE: `par_scan_threshold()` keyed on `feats.len()` (scan work ∝ features ×
    // bins, NOT leaf rows — a rows-based gate would be WRONG). Below threshold the
    // serial path runs verbatim, protecting narrow leaves from the per-feature
    // dispatch overhead (the same overhead that regressed the unconditional BUILD
    // path, Spike 005). Empty feats ⇒ empty Vec, no work (threat T-lsx-03).
    //
    // Scope: CpuBackend ONLY. RocmBackend keeps its fused-launch override (lib.rs).
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
        // Sub-threshold (incl. empty): the trait-default serial loop verbatim —
        // zero behavior change, and the per-feature dispatch overhead that would
        // regress narrow leaves is avoided.
        if feats.len() < par_scan_threshold() {
            let mut out = Vec::with_capacity(feats.len());
            for f in feats {
                let cells = 2usize
                    .checked_mul(f.num_bin as usize)
                    .ok_or_else(|| ComputeError::Runtime {
                        detail: format!("num_bin {} overflows the histogram length", f.num_bin),
                    })?;
                let end =
                    f.slot_off
                        .checked_add(cells)
                        .ok_or_else(|| ComputeError::Runtime {
                            detail: "find_best_splits_batched: slot_off + region overflows"
                                .to_string(),
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
            return Ok(out);
        }

        use rayon::prelude::*;

        // 1) Hoisted validation in ascending feature order ⇒ deterministic error:
        //    return the LOWEST-index offending feature's error, identical to the
        //    serial loop's first-failure behavior. Records each feature's validated
        //    `[start, end)` so the parallel map is infallible on the hist slice.
        let mut ranges: Vec<(usize, usize)> = Vec::with_capacity(feats.len());
        for f in feats {
            let cells =
                2usize
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
            ranges.push((f.slot_off, end));
        }

        // 2) ONE fork/join per leaf, amortized across all features. `par_iter()`
        //    preserves order ⇒ out[i] corresponds to feats[i]. find_best_split takes
        //    `&self` + `client: &ComputeClient` (shared refs) and `buf` is `&[f64]`
        //    (shared) ⇒ Send+Sync-safe to fan out; the validated ranges make each
        //    closure infallible.
        let out: Vec<SplitInfo> = feats
            .par_iter()
            .zip(ranges.par_iter())
            .map(|(f, &(start, end))| {
                let hist = &buf[start..end];
                // SAFETY of unwrap-free path: find_best_split itself only errors on
                // region-length issues already validated above; treat any residual
                // error as a hard fault by propagating through a Result collect.
                self.find_best_split(
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
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(out)
    }

    // quick 260620-a48: UNIFIED host build+fix+scan — delegate to the inherent impl
    // (the host f64 analog of the GPU build_fix_scan_resident). CpuBackend ONLY;
    // RocmBackend keeps the trait-default typed error (its fusion is the resident path).
    #[allow(clippy::too_many_arguments)]
    fn build_fix_scan(
        &self,
        client: &ComputeClient<Self::Runtime>,
        buf: &mut [f64],
        feature_bins: &[&BinColumn],
        leaf_rows: &[u32],
        gradients: &[f32],
        hessians: &[f32],
        all_feats: &[BatchedSplitFeature],
        scan_active: &[bool],
        cfg: &GainConfig,
        sum_gradient: f64,
        sum_hessian: f64,
        num_data: i32,
    ) -> Result<Vec<Option<SplitInfo>>, ComputeError> {
        self.build_fix_scan_impl(
            client,
            buf,
            feature_bins,
            leaf_rows,
            gradients,
            hessians,
            all_feats,
            scan_active,
            cfg,
            sum_gradient,
            sum_hessian,
            num_data,
        )
    }
}

/// quick 260620-a48: the host f64 analog of the GPU `build_fix_scan_resident` fusion.
#[cfg(feature = "cpu")]
impl CpuBackend {
    #[allow(clippy::too_many_arguments)]
    fn build_fix_scan_impl(
        &self,
        client: &ComputeClient<<Self as Backend>::Runtime>,
        buf: &mut [f64],
        feature_bins: &[&BinColumn],
        leaf_rows: &[u32],
        gradients: &[f32],
        hessians: &[f32],
        all_feats: &[BatchedSplitFeature],
        scan_active: &[bool],
        cfg: &GainConfig,
        sum_gradient: f64,
        sum_hessian: f64,
        num_data: i32,
    ) -> Result<Vec<Option<SplitInfo>>, ComputeError> {
        use rayon::prelude::*;

        debug_assert_eq!(feature_bins.len(), all_feats.len());
        debug_assert_eq!(scan_active.len(), all_feats.len());

        // 1) Gather the ordered gradients/hessians ONCE per leaf (identical across every
        //    feature — only the bin column differs; mirrors `build_leaf_histograms_raw`
        //    SPIKE 003). Values + order unchanged ⇒ bit-exact fold inputs.
        let r = leaf_rows.len();
        let mut ord_g: Vec<f32> = Vec::with_capacity(r);
        let mut ord_h: Vec<f32> = Vec::with_capacity(r);
        for &row in leaf_rows {
            ord_g.push(gradients[row as usize]);
            ord_h.push(hessians[row as usize]);
        }

        // 2) Hoisted ascending-order validation ⇒ deterministic lowest-index error
        //    (matches the two-step / serial path's first-failure behavior). Records each
        //    feature's validated `[start, end)` so the parallel map is infallible.
        let mut ranges: Vec<(usize, usize)> = Vec::with_capacity(all_feats.len());
        for f in all_feats {
            let cells =
                2usize
                    .checked_mul(f.num_bin as usize)
                    .ok_or_else(|| ComputeError::Runtime {
                        detail: format!("num_bin {} overflows the histogram length", f.num_bin),
                    })?;
            let end = f
                .slot_off
                .checked_add(cells)
                .ok_or_else(|| ComputeError::Runtime {
                    detail: "build_fix_scan: slot_off + region overflows".to_string(),
                })?;
            if end > buf.len() {
                return Err(ComputeError::LengthMismatch {
                    expected: end,
                    actual: buf.len(),
                });
            }
            ranges.push((f.slot_off, end));
        }

        // 3) ONE rayon fork/join over features. Each task: build → fix → compact → scan
        //    into its OWN private buffer (cache-hot, no cross-region hand-off). Returns
        //    `(private_hist, Option<SplitInfo>)`; `par_iter` preserves input order so the
        //    serial assembly below stays feature-index-ordered (bit-exact argmax).
        let results: Vec<Result<(Vec<f64>, Option<SplitInfo>), ComputeError>> = all_feats
            .par_iter()
            .zip(scan_active.par_iter())
            .zip(feature_bins.par_iter())
            .map(|((f, &active), bins)| {
                let cells = 2 * f.num_bin as usize;
                // (a) BUILD: own private histogram, ascending leaf_rows, grad at bin<<1.
                let mut hist = vec![0.0f64; cells];
                fold_one_feature(bins, leaf_rows, &ord_g, &ord_h, &mut hist);
                // (b) FIX: most_freq_bin reconstruct on RAW leaf sums (Pitfall 2). No-op
                //     for most_freq_bin==0 (the C++ `if (most_freq_bin > 0)` guard).
                fix_histogram_inline(&mut hist, f.most_freq_bin, sum_gradient, sum_hessian);
                // (c) COMPACT: shift real-bin `c+offset` into `c`, zero the tail (D-09).
                //     No-op for offset==0.
                compact_histogram_inline(&mut hist, f.offset);
                // (d) SCAN (only if spine-active): own SplitInfo from the disjoint hist.
                let split = if active {
                    Some(self.find_best_split(
                        client,
                        &hist,
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
                    )?)
                } else {
                    None
                };
                Ok((hist, split))
            })
            .collect();

        // 4) SERIAL ordered assembly: copy each private hist into its disjoint `buf`
        //    region (COMPLETE histogram for the subtract-larger child) and collect the
        //    per-feature SplitInfos in feature-index order. Propagate the lowest-index
        //    scan error (the validated ranges already guarantee region-length safety).
        let mut splits: Vec<Option<SplitInfo>> = Vec::with_capacity(all_feats.len());
        for (fpos, res) in results.into_iter().enumerate() {
            let (hist, split) = res?;
            let (start, end) = ranges[fpos];
            buf[start..end].copy_from_slice(&hist);
            splits.push(split);
        }
        Ok(splits)
    }
}

/// `Dataset::FixHistogram` reconstruct (the compute-crate inline twin of the
/// learner-side `fix_histogram::fix_histogram`, kept here to avoid a treelearner→
/// compute circular dep). Byte-identical op order: seed the most-freq cell with the
/// RAW leaf totals, subtract every other bin in ASCENDING order. No-op for
/// `most_freq_bin == 0` (C++ `if (most_freq_bin > 0)`).
#[cfg(feature = "cpu")]
#[inline]
fn fix_histogram_inline(hist: &mut [f64], most_freq_bin: u32, sum_gradient: f64, sum_hessian: f64) {
    if most_freq_bin == 0 {
        return;
    }
    let num_bin = hist.len() / 2;
    let mfb = most_freq_bin as usize;
    if mfb >= num_bin {
        return;
    }
    let g_idx = mfb << 1;
    let h_idx = g_idx + 1;
    let mut g = sum_gradient;
    let mut h = sum_hessian;
    for i in 0..num_bin {
        if i != mfb {
            g -= hist[i << 1];
            h -= hist[(i << 1) + 1];
        }
    }
    hist[g_idx] = g;
    hist[h_idx] = h;
}

/// COMPACTED-layout shift (the compute-crate inline twin of the learner-side
/// `compact_histogram`). Shift pair `c + off` down to `c` in ASCENDING order, zero the
/// dropped tail. No-op for `offset <= 0`. Byte-identical to the two-step compaction.
#[cfg(feature = "cpu")]
#[inline]
fn compact_histogram_inline(hist: &mut [f64], offset: i32) {
    if offset <= 0 {
        return;
    }
    let off = offset as usize;
    let num_bin = hist.len() / 2;
    if off >= num_bin {
        for cell in hist.iter_mut() {
            *cell = 0.0;
        }
        return;
    }
    for c in 0..(num_bin - off) {
        let dst = c << 1;
        let src = (c + off) << 1;
        hist[dst] = hist[src];
        hist[dst + 1] = hist[src + 1];
    }
    for cell in hist.iter_mut().skip((num_bin - off) << 1) {
        *cell = 0.0;
    }
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

    /// The GPU path's resident u32 upload DOES consume the widened columns (spike
    /// 004): the learner widens each [`BinColumn`] to u32 once and calls
    /// [`upload_resident_bins`](Backend::upload_resident_bins) so the resident buffer
    /// is byte-identical to HEAD.
    fn wants_resident_bins(&self) -> bool {
        true
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
        feature_bins: &[&BinColumn],
        _num_bins: &[u32],
        slot_off: &[usize],
        slot_len: usize,
        leaf_rows: &[u32],
        gradients: &[f32],
        hessians: &[f32],
    ) -> Result<Vec<f64>, ComputeError> {
        // L1 device-resident path: gather on device from the cached column buffer.
        // The kernel input is byte-IDENTICAL to HEAD — residency uses the u32 buffer
        // uploaded once by `upload_resident_bins`, so the narrow `feature_bins` is
        // not even consulted here.
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
        // widen each narrow column to u32 ONCE (cold) so the host-gather batched
        // launcher sees the SAME byte-identical u32 input as HEAD.
        let widened: Vec<Vec<u32>> = feature_bins.iter().map(|c| c.to_u32_vec()).collect();
        let widened_refs: Vec<&[u32]> = widened.iter().map(Vec::as_slice).collect();
        kernels::histogram::build_leaf_histograms_batched_f32_on(
            client,
            &widened_refs,
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
        _feature_bins: &[&BinColumn],
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

#[cfg(test)]
mod par_build_tests {
    use super::{build_histograms_into, BinColumn};

    /// Spike 005 / R4 bit-exact guard: the rayon-parallel per-feature build MUST
    /// produce a byte-identical (f64::to_bits) `out` to the serial path — the
    /// per-feature folds are independent + same-order, so thread scheduling can never
    /// change the result. Guards the multi-threaded anchor's determinism.
    #[test]
    fn build_histograms_parallel_equals_serial() {
        // Synthetic 4-feature leaf, mixed widths, scattered leaf_rows.
        let rows: u32 = 5000;
        let cols: Vec<BinColumn> = (0..4u32)
            .map(|f| {
                let nb = [32u32, 200, 257, 70000][f as usize];
                let v: Vec<u32> = (0..rows)
                    .map(|r| {
                        let h = (r as u64).wrapping_mul(2_654_435_761).wrapping_add(f as u64 * 97);
                        (h % nb as u64) as u32
                    })
                    .collect();
                BinColumn::new(v, nb)
            })
            .collect();
        let num_bins: Vec<u32> = vec![32, 200, 257, 70000];
        let mut slot_off = Vec::new();
        let mut off = 0usize;
        for &nb in &num_bins {
            slot_off.push(off);
            off += 2 * nb as usize;
        }
        let slot_len = off;
        // Scattered row order (mimic real leaf_rows).
        let leaf_rows: Vec<u32> = (0..rows).map(|i| (i.wrapping_mul(2_654_435_761)) % rows).collect();
        let ord_g: Vec<f32> = (0..rows).map(|i| (i % 13) as f32 * 0.1).collect();
        let ord_h: Vec<f32> = (0..rows).map(|i| 1.0 + (i % 7) as f32 * 0.01).collect();
        let refs: Vec<&BinColumn> = cols.iter().collect();

        let serial = build_histograms_into(&refs, &num_bins, &slot_off, slot_len, &leaf_rows, &ord_g, &ord_h, false);
        let parallel = build_histograms_into(&refs, &num_bins, &slot_off, slot_len, &leaf_rows, &ord_g, &ord_h, true);
        assert_eq!(serial.len(), parallel.len());
        for (i, (s, p)) in serial.iter().zip(&parallel).enumerate() {
            assert_eq!(s.to_bits(), p.to_bits(), "cell {i}: serial {s} != parallel {p} (not bit-identical)");
        }
    }

    /// Spike 011 microbench — isolates the parallel build's `Vec<Vec<f64>>`+copy
    /// strategy (BEFORE) against the disjoint-slot scatter (AFTER, the live
    /// `build_histograms_into` parallel branch) on ONE representative large leaf,
    /// many launches, same process. Ignored by default; run with:
    ///   cargo test -p lgbm-compute --release --lib spike011_microbench -- --ignored --nocapture
    #[test]
    #[ignore]
    fn spike011_microbench() {
        use super::fold_one_feature;
        use std::time::Instant;

        // Representative large leaf: a big leaf is where the parallel path fires.
        // Sweep via env (default 1M); 16384 is the live parallel threshold.
        let rows: u32 = std::env::var("SPIKE011_ROWS").ok().and_then(|s| s.parse().ok()).unwrap_or(1_000_000);
        let nfeat = 50usize;
        let bins = 256u32;
        let cols: Vec<BinColumn> = (0..nfeat as u32)
            .map(|f| {
                let v: Vec<u32> = (0..rows)
                    .map(|r| {
                        let h = (r as u64).wrapping_mul(2_654_435_761).wrapping_add(f as u64 * 97);
                        (h % bins as u64) as u32
                    })
                    .collect();
                BinColumn::new(v, bins)
            })
            .collect();
        let num_bins: Vec<u32> = vec![bins; nfeat];
        let mut slot_off = Vec::new();
        let mut off = 0usize;
        for &nb in &num_bins {
            slot_off.push(off);
            off += 2 * nb as usize;
        }
        let slot_len = off;
        // Whole leaf, scattered order (worst-case gather, like a real leaf).
        let leaf_rows: Vec<u32> = (0..rows).map(|i| i.wrapping_mul(2_654_435_761) % rows).collect();
        let ord_g: Vec<f32> = (0..rows).map(|i| (i % 13) as f32 * 0.1).collect();
        let ord_h: Vec<f32> = (0..rows).map(|i| 1.0 + (i % 7) as f32 * 0.01).collect();
        let refs: Vec<&BinColumn> = cols.iter().collect();

        // BEFORE (the LIVE production strategy): per-feature private `Vec<f64>`
        // accumulators + a sequential reassembly copy.
        let before = |out: &mut [f64]| {
            use rayon::prelude::*;
            let hists: Vec<Vec<f64>> = (0..refs.len())
                .into_par_iter()
                .map(|fpos| {
                    let mut h = vec![0.0f64; 2 * num_bins[fpos] as usize];
                    fold_one_feature(refs[fpos], &leaf_rows, &ord_g, &ord_h, &mut h);
                    h
                })
                .collect();
            for (fpos, h) in hists.into_iter().enumerate() {
                out[slot_off[fpos]..slot_off[fpos] + h.len()].copy_from_slice(&h);
            }
        };
        // AFTER (the REJECTED scatter): carve `out` into disjoint per-feature slots
        // and fold each feature straight into its slot (no intermediate, no copy).
        let after = |out: &mut [f64]| {
            use rayon::prelude::*;
            let mut slots: Vec<&mut [f64]> = Vec::with_capacity(refs.len());
            let mut rest = &mut out[..];
            let mut base = 0usize;
            for (fpos, &nb) in num_bins.iter().enumerate() {
                let (_gap, after_gap) = rest.split_at_mut(slot_off[fpos] - base);
                let (slot, tail) = after_gap.split_at_mut(2 * nb as usize);
                slots.push(slot);
                rest = tail;
                base = slot_off[fpos] + 2 * nb as usize;
            }
            slots.into_par_iter().enumerate().for_each(|(fpos, slot)| {
                fold_one_feature(refs[fpos], &leaf_rows, &ord_g, &ord_h, slot);
            });
        };

        let launches = 30usize;
        let warm = 5usize;
        // Interleave the two strategies per launch to cancel thermal/scheduler drift.
        let mut t_before = Vec::with_capacity(launches);
        let mut t_after = Vec::with_capacity(launches);
        let mut sink = 0.0f64;
        for i in 0..(launches + warm) {
            let mut out_b = vec![0.0f64; slot_len];
            let s0 = Instant::now();
            before(&mut out_b);
            let d_b = s0.elapsed();
            sink += out_b[0] + out_b[slot_len - 1];

            let mut out_a = vec![0.0f64; slot_len];
            let s1 = Instant::now();
            after(&mut out_a);
            let d_a = s1.elapsed();
            sink += out_a[0] + out_a[slot_len - 1];

            // Byte-equality spot check (same fold order ⇒ identical).
            assert_eq!(out_b[0].to_bits(), out_a[0].to_bits());
            assert_eq!(out_b[slot_len - 1].to_bits(), out_a[slot_len - 1].to_bits());

            if i >= warm {
                t_before.push(d_b);
                t_after.push(d_a);
            }
        }
        std::hint::black_box(sink);
        t_before.sort();
        t_after.sort();
        let med = |v: &[std::time::Duration]| v[v.len() / 2];
        let mb = med(&t_before);
        let ma = med(&t_after);
        let ratio = mb.as_secs_f64() / ma.as_secs_f64();
        eprintln!(
            "spike011 microbench ({rows} rows x {nfeat} feat x {bins} bins, {launches} launches, interleaved):\n  \
             LIVE   Vec<Vec>+copy (private accumulators): {mb:?}/build\n  \
             SCATTER into shared `out` (rejected)       : {ma:?}/build\n  \
             ratio live/scatter: {ratio:.3}x  (<1 ⇒ scatter SLOWER ⇒ keep Vec<Vec>)"
        );
    }
}

#[cfg(all(test, feature = "cpu"))]
mod build_fix_scan_tests {
    use super::{
        compact_histogram_inline, fix_histogram_inline, fold_one_feature, Backend, BatchedSplitFeature,
        BinColumn, CpuBackend,
    };
    use crate::gain::GainConfig;
    use crate::runtime::cpu_client;

    /// Build a small synthetic multi-feature leaf (mixed widths, scattered rows) and the
    /// per-feature `BatchedSplitFeature` params + slot layout.
    fn fixture() -> (
        Vec<BinColumn>,
        Vec<u32>,
        Vec<usize>,
        usize,
        Vec<u32>,
        Vec<f32>,
        Vec<f32>,
        Vec<BatchedSplitFeature>,
        f64,
        f64,
        i32,
    ) {
        let rows: u32 = 400;
        let num_bins: Vec<u32> = vec![8, 5, 16, 4];
        // most_freq_bin per feature: exercise both the offset==1 (mfb==0) and the
        // mfb>0 FixHistogram-reconstruct paths.
        let most_freq: Vec<u32> = vec![0, 2, 0, 1];
        let cols: Vec<BinColumn> = num_bins
            .iter()
            .enumerate()
            .map(|(f, &nb)| {
                let v: Vec<u32> = (0..rows)
                    .map(|r| {
                        let h = (r as u64).wrapping_mul(2_654_435_761).wrapping_add(f as u64 * 131);
                        (h % nb as u64) as u32
                    })
                    .collect();
                BinColumn::new(v, nb)
            })
            .collect();
        let mut slot_off = Vec::new();
        let mut off = 0usize;
        for &nb in &num_bins {
            slot_off.push(off);
            off += 2 * nb as usize;
        }
        let slot_len = off;
        let leaf_rows: Vec<u32> =
            (0..rows).map(|i| (i.wrapping_mul(2_654_435_761)) % rows).collect();
        let gradients: Vec<f32> = (0..rows).map(|i| ((i % 13) as f32 - 6.0) * 0.1).collect();
        let hessians: Vec<f32> = (0..rows).map(|i| 1.0 + (i % 7) as f32 * 0.01).collect();
        let feats: Vec<BatchedSplitFeature> = num_bins
            .iter()
            .enumerate()
            .map(|(f, &nb)| BatchedSplitFeature {
                slot_off: slot_off[f],
                num_bin: nb,
                offset: if most_freq[f] == 0 { 1 } else { 0 },
                default_bin: nb, // out of range -> SKIP_DEFAULT_BIN never fires
                most_freq_bin: most_freq[f],
                skip_default_bin: false,
                na_as_missing: false,
                run_forward: false,
            })
            .collect();
        // RAW leaf sums over the leaf rows (the fix operand, Pitfall 2).
        let mut sum_g = 0.0f64;
        let mut sum_h = 0.0f64;
        for &row in &leaf_rows {
            sum_g += f64::from(gradients[row as usize]);
            sum_h += f64::from(hessians[row as usize]);
        }
        let num_data = leaf_rows.len() as i32;
        (
            cols, num_bins, slot_off, slot_len, leaf_rows, gradients, hessians, feats, sum_g, sum_h,
            num_data,
        )
    }

    fn relaxed_cfg() -> GainConfig {
        GainConfig {
            min_data_in_leaf: 1,
            min_sum_hessian_in_leaf: 0.0,
            max_delta_step: 0.0,
            lambda_l1: 0.0,
            lambda_l2: 0.0,
            min_gain_to_split: 0.0,
            path_smooth: 0.0,
            ..Default::default()
        }
    }

    /// BEHAVIOR 1+2: the unified region returns SplitInfos byte-identical to the
    /// two-step path for every scan-active feature, in EXACTLY feature-index order.
    #[test]
    fn unified_build_fix_scan_bit_exact_and_ordered_vs_two_step() {
        let backend = CpuBackend;
        let client = cpu_client();
        let cfg = relaxed_cfg();
        let (cols, _num_bins, _slot_off, slot_len, leaf_rows, grads, hess, feats, sum_g, sum_h, num_data) =
            fixture();
        let refs: Vec<&BinColumn> = cols.iter().collect();
        // Ordered grads/hess for the two-step reference fold.
        let ord_g: Vec<f32> = leaf_rows.iter().map(|&r| grads[r as usize]).collect();
        let ord_h: Vec<f32> = leaf_rows.iter().map(|&r| hess[r as usize]).collect();

        // All features scan-active (every feature is spine).
        let scan_active = vec![true; feats.len()];
        let mut buf = vec![0.0f64; slot_len];
        let unified = backend
            .build_fix_scan(
                &client, &mut buf, &refs, &leaf_rows, &grads, &hess, &feats, &scan_active, &cfg,
                sum_g, sum_h, num_data,
            )
            .expect("unified region");

        assert_eq!(unified.len(), feats.len(), "one result slot per feature");
        for (fpos, f) in feats.iter().enumerate() {
            // Two-step reference for this feature: build raw -> fix -> compact -> scan.
            let cells = 2 * f.num_bin as usize;
            let mut hist = vec![0.0f64; cells];
            fold_one_feature(&cols[fpos], &leaf_rows, &ord_g, &ord_h, &mut hist);
            fix_histogram_inline(&mut hist, f.most_freq_bin, sum_g, sum_h);
            compact_histogram_inline(&mut hist, f.offset);
            let want = backend
                .find_best_split(
                    &client, &hist, &cfg, f.num_bin, f.offset, f.default_bin, f.most_freq_bin,
                    f.skip_default_bin, f.na_as_missing, f.run_forward, sum_g, sum_h, num_data,
                )
                .expect("two-step scan");
            let got = unified[fpos].expect("scan-active feature has a SplitInfo");
            assert_eq!(
                got.gain.to_bits(),
                want.gain.to_bits(),
                "feature {fpos}: unified gain != two-step gain (not bit-identical)"
            );
            assert_eq!(got.threshold, want.threshold, "feature {fpos}: threshold");
            assert_eq!(got.default_left, want.default_left, "feature {fpos}: default_left");
            assert_eq!(got.left_count, want.left_count, "feature {fpos}: left_count");
            assert_eq!(
                got.left_sum_gradient.to_bits(),
                want.left_sum_gradient.to_bits(),
                "feature {fpos}: left_sum_gradient"
            );
        }
    }

    /// BEHAVIOR: the unified region writes the COMPLETE fixed+compacted histogram into
    /// `buf` for EVERY feature (needed by the subtract-derived larger child) —
    /// byte-identical to the two-step `build_leaf_histograms_raw` + per-feature fix+compact.
    #[test]
    fn unified_buf_is_complete_and_bit_exact() {
        let backend = CpuBackend;
        let client = cpu_client();
        let cfg = relaxed_cfg();
        let (cols, num_bins, slot_off, slot_len, leaf_rows, grads, hess, feats, sum_g, sum_h, num_data) =
            fixture();
        let refs: Vec<&BinColumn> = cols.iter().collect();

        // Two-step reference buffer: batched raw build, then per-feature fix+compact.
        let mut want_buf = backend
            .build_leaf_histograms_raw(
                &client, &refs, &num_bins, &slot_off, slot_len, &leaf_rows, &grads, &hess,
            )
            .expect("raw build");
        for (fpos, f) in feats.iter().enumerate() {
            let cells = 2 * f.num_bin as usize;
            let range = slot_off[fpos]..slot_off[fpos] + cells;
            let h = &mut want_buf[range.clone()];
            fix_histogram_inline(h, f.most_freq_bin, sum_g, sum_h);
            compact_histogram_inline(h, f.offset);
        }

        // Unified region (mark only feature 0 scan-active to prove build runs for ALL).
        let mut scan_active = vec![false; feats.len()];
        scan_active[0] = true;
        let mut got_buf = vec![0.0f64; slot_len];
        let _ = backend
            .build_fix_scan(
                &client, &mut got_buf, &refs, &leaf_rows, &grads, &hess, &feats, &scan_active, &cfg,
                sum_g, sum_h, num_data,
            )
            .expect("unified region");

        for (i, (w, g)) in want_buf.iter().zip(&got_buf).enumerate() {
            assert_eq!(w.to_bits(), g.to_bits(), "buf cell {i}: two-step {w} != unified {g}");
        }
    }

    /// BEHAVIOR: an ineligible (non-scan-active) feature returns `None` (it is NOT
    /// scanned) while still being built into `buf`. Scan-active features return `Some`.
    #[test]
    fn unified_ineligible_feature_returns_none() {
        let backend = CpuBackend;
        let client = cpu_client();
        let cfg = relaxed_cfg();
        let (cols, _num_bins, _slot_off, slot_len, leaf_rows, grads, hess, feats, sum_g, sum_h, num_data) =
            fixture();
        let refs: Vec<&BinColumn> = cols.iter().collect();
        // Features 0 and 2 active, 1 and 3 ineligible.
        let scan_active = vec![true, false, true, false];
        let mut buf = vec![0.0f64; slot_len];
        let res = backend
            .build_fix_scan(
                &client, &mut buf, &refs, &leaf_rows, &grads, &hess, &feats, &scan_active, &cfg,
                sum_g, sum_h, num_data,
            )
            .expect("unified region");
        assert!(res[0].is_some(), "feature 0 active -> Some");
        assert!(res[1].is_none(), "feature 1 ineligible -> None");
        assert!(res[2].is_some(), "feature 2 active -> Some");
        assert!(res[3].is_none(), "feature 3 ineligible -> None");
    }
}

#[cfg(test)]
mod bin_column_tests {
    use super::BinColumn;

    #[test]
    fn width_selected_by_num_bin_boundaries() {
        // num_bin 256 -> U8 (the inclusive upper edge of u8 capacity).
        assert!(matches!(BinColumn::new(vec![0, 1, 255], 256), BinColumn::U8(_)));
        // num_bin 257 -> U16 (one past u8 capacity).
        assert!(matches!(BinColumn::new(vec![0, 256], 257), BinColumn::U16(_)));
        // num_bin 65536 -> U16 (the inclusive upper edge of u16 capacity).
        assert!(matches!(BinColumn::new(vec![0, 65535], 65536), BinColumn::U16(_)));
        // num_bin 65537 -> U32 (one past u16 capacity).
        assert!(matches!(BinColumn::new(vec![0, 65536], 65537), BinColumn::U32(_)));
    }

    #[test]
    fn width_selected_by_num_bin_not_observed_max() {
        // max == 1 but num_bin == 256 still selects U8 (type fixed by bin count).
        assert!(matches!(BinColumn::new(vec![0, 1], 256), BinColumn::U8(_)));
        // max == 1 but num_bin == 300 selects U16.
        assert!(matches!(BinColumn::new(vec![0, 1], 300), BinColumn::U16(_)));
    }

    #[test]
    fn len_and_bin_widen_per_variant() {
        let u8c = BinColumn::new(vec![0, 1, 255], 256);
        assert_eq!(u8c.len(), 3);
        assert_eq!(u8c.bin(2), 255u32);

        let u16c = BinColumn::new(vec![0, 300], 300);
        assert_eq!(u16c.bin(1), 300u32);

        let u32c = BinColumn::new(vec![0, 70_000], 70_000);
        assert_eq!(u32c.bin(1), 70_000u32);
    }

    #[test]
    fn is_empty_reports_empty_column() {
        assert!(BinColumn::U32(Vec::new()).is_empty());
        assert!(!BinColumn::new(vec![0], 256).is_empty());
    }

    #[test]
    fn gather_preserves_width() {
        let u8c = BinColumn::new(vec![5, 6, 7, 8], 256);
        let g = u8c.gather(&[3, 1]);
        assert!(matches!(g, BinColumn::U8(_)));
        assert_eq!(g.to_u32_vec(), vec![8, 6]);

        let u16c = BinColumn::new(vec![5, 6, 7, 8], 300);
        let g16 = u16c.gather(&[0, 2]);
        assert!(matches!(g16, BinColumn::U16(_)));
        assert_eq!(g16.to_u32_vec(), vec![5, 7]);

        let u32c = BinColumn::new(vec![5, 6, 7, 8], 70_000);
        let g32 = u32c.gather(&[2, 0]);
        assert!(matches!(g32, BinColumn::U32(_)));
        assert_eq!(g32.to_u32_vec(), vec![7, 5]);
    }

    #[test]
    fn to_u32_vec_round_trips_all_widths() {
        for (v, nb) in [
            (vec![0u32, 1, 255], 256u32),
            (vec![0u32, 1, 300], 301),
            (vec![0u32, 1, 70_000], 70_001),
        ] {
            assert_eq!(BinColumn::new(v.clone(), nb).to_u32_vec(), v);
        }
    }

    #[test]
    fn first_ge_finds_first_out_of_range_per_width() {
        // U8: first value >= 4 is the 5 at index 2.
        assert_eq!(BinColumn::new(vec![0, 1, 5, 2], 256).first_ge(4), Some(5));
        // U16: first value >= 300 is 350.
        assert_eq!(BinColumn::new(vec![0, 350, 1], 400).first_ge(300), Some(350));
        // U32: first value >= 70_000 is 70_000.
        assert_eq!(
            BinColumn::new(vec![0, 70_000, 1], 80_000).first_ge(70_000),
            Some(70_000)
        );
        // None when every element is below the bound.
        assert_eq!(BinColumn::new(vec![0, 1, 2, 3], 256).first_ge(4), None);
    }

    #[test]
    fn iter_u32_matches_to_u32_vec() {
        for (v, nb) in [
            (vec![0u32, 5, 255], 256u32),
            (vec![0u32, 5, 300], 301),
            (vec![0u32, 5, 70_000], 70_001),
        ] {
            let c = BinColumn::new(v.clone(), nb);
            assert_eq!(c.iter_u32().collect::<Vec<u32>>(), v);
        }
    }
}
