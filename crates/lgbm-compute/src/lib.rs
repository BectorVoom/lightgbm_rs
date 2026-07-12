//! `lgbm-compute` — the single CubeCL isolation seam.
//!
//! This crate exists to confine all `cubecl` type names and API churn to one
//! place so the alpha-stage CubeCL surface can evolve without leaking into
//! `lgbm-core` or any other crate. It provides the compute foundation: a typed
//! [`ComputeError`] boundary, a cpu/rocm runtime selection + startup capability
//! gate ([`runtime`]), and the first `#[cube]` histogram kernel ([`kernels`]).
//!
//! Downstream crates should depend only on the [`Backend`] abstraction (and the
//! re-exported [`ComputeError`]), never on `cubecl` directly — that is the whole
//! point of the seam, and the `cmp01_containment` guard test enforces it.

pub mod device_metric;
pub mod device_objective;
pub mod error;
pub mod fusion_prof;
pub mod gain;
pub mod kernels;
pub mod runtime;

pub use device_objective::{device_objective_supported, DeviceObjectiveKind};
pub use error::ComputeError;
pub use gain::{GainConfig, SplitInfo};
pub use kernels::grow_driver::GrowFeature;
// The on-device resident `AddPredictionToScore` partition scatter applied after an
// on-device grow (the score half of the on-device perf fix). Re-exported at the crate
// root so the `boosting_on_cuda_` resident-score seam reaches it without spelling the
// full `kernels::grow_driver` path.
pub use kernels::grow_driver::add_prediction_to_score_on_device_resident;
pub use kernels::grow_driver::read_handle_f32;
pub use kernels::grow_driver::ResidentScore;
// The per-train grad/hess device residency (labels uploaded once + reused outputs)
// and its A/B escape hatch, reached by the `lgbm-boosting` resident grad dispatch.
pub use kernels::grow_driver::{grad_residency_enabled, GradResidency};
pub use kernels::predict::derive_leaf_map_device;
pub use kernels::split::BatchedSplitFeature;

/// Re-export of the cubecl device buffer [`Handle`](cubecl::server::Handle) so
/// downstream crates (e.g. `lgbm-boosting`) can name the resident device buffers
/// (the resident score / grad / hess Handles the on-device grad/hess path returns)
/// WITHOUT depending on `cubecl` directly — preserving the containment
/// boundary. `Handle` is a cheaply-clonable, ref-counted handle to a device
/// allocation; it names no runtime.
pub use cubecl::server::Handle;

use cubecl::prelude::ComputeClient;

/// Re-export of the cubecl [`ComputeClient`](cubecl::prelude::ComputeClient) so
/// downstream crates (e.g. `lgbm-treelearner`) can name the
/// `&ComputeClient<B::Runtime>` argument the [`Backend`] ops require WITHOUT
/// depending on `cubecl` directly — preserving the containment boundary
/// (the compute crate is the single CubeCL seam; everyone above it sees only
/// `lgbm_compute::ComputeClient`).
pub use cubecl::prelude::ComputeClient as ComputeClientReexport;

/// A feature column's per-row bin indices, stored in the NARROWEST unsigned type
/// for its `num_bin` (columnar narrow bins). Faithful to C++
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
    /// memory-safety concern), which always holds because the type is
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
    /// per-row path (a measured small-row regression fix).
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
/// loop (monomorphic arms), and the fold ORDER is byte-identical to
/// `construct_histograms_cpu_native`, so this is the single bit-exact fold body used
/// by BOTH the serial and the parallel build paths.
///
/// Precondition (caller-established once per train): every
/// `bins[row] < num_bin`. Debug-asserted; release trusts the upstream gate.
#[inline]
fn fold_one_feature(bins: &BinColumn, leaf_rows: &[u32], ord_g: &[f32], ord_h: &[f32], h: &mut [f64]) {
    macro_rules! fold {
        ($v:expr) => {
            // get_unchecked elides the 4 per-row bounds checks (bit-exact — same
            // order/values). The precondition `bins[row] < num_bin` is established once
            // per train and debug-asserted below; `k < leaf_rows.len()` and
            // `row < num_data` hold by construction, so every index here is provably in
            // bounds.
            unsafe {
                for (k, &row) in leaf_rows.iter().enumerate() {
                    let bin = *$v.get_unchecked(row as usize) as usize;
                    debug_assert!(bin * 2 + 1 < h.len(), "bin out of range — caller must establish bin < num_bin once per train");
                    *h.get_unchecked_mut(bin * 2) += f64::from(*ord_g.get_unchecked(k));
                    *h.get_unchecked_mut(bin * 2 + 1) += f64::from(*ord_h.get_unchecked(k));
                }
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
/// `[slot_off[f], slot_off[f] + 2*num_bins[f])`). `parallel` selects rayon-over-features
/// — each feature folds its OWN histogram Vec from the shared
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
        // NOTE: this per-feature `Vec<f64>` intermediate looks like a flattenable
        // `Vec<Vec<f64>>` wart, but it is LOAD-BEARING. Each rayon task folds into its
        // OWN cache-hot private buffer, then a single sequential `copy_from_slice`
        // assembles `out`. A scatter variant (threads folding directly into disjoint
        // sub-slices of one shared `out`) measured worse due to cache-coherence /
        // false-sharing traffic exceeding the alloc+copy it would remove. Keep the
        // private accumulators.
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
/// (parallel wins at large leaf sizes; the default keeps small+medium leaves serial
/// with zero regression while parallelizing the genuinely large ones).
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
/// count alone separates the narrow/wide regimes). Below the threshold the serial
/// loop runs verbatim (the same per-feature dispatch overhead that regressed the
/// unconditional BUILD path would regress narrow leaves here). Override via
/// `LGBM_PAR_SCAN_THRESHOLD`.
///
/// DEFAULT: HONEST NULL — the parallel scan is kept available (env-reachable,
/// bit-exact-proven FORCED-ON) but is NOT the effective default, so the threshold
/// is set effectively-unreachable (`usize::MAX`). Measurement showed the per-leaf
/// scan fork/join CONTENDS with the already-rayon-parallel BUILD path, so an
/// isolated scan-time win does NOT translate to a warm train-wall win; it
/// regresses overall. The adoption criterion (wide train-wall gain AND no narrow
/// regression) is not met, so the serial loop stays the effective default. Set
/// `LGBM_PAR_SCAN_THRESHOLD=0` (or any small value) to force the parallel path on
/// for the bit-exact parity proof / future re-measurement.
fn par_scan_threshold() -> usize {
    std::env::var("LGBM_PAR_SCAN_THRESHOLD")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(usize::MAX)
}

/// Floor for a core-scaled unified-fusion threshold: never let a many-core machine
/// drive the gate so low it parallelizes a trivially-narrow leaf (narrow leaves regress
/// badly under fork/join overhead). 32 = the lowest feature count measurement found a
/// sign-stable win at.
const THRESHOLD_FLOOR: usize = 32;

/// Ceiling for a core-scaled unified-fusion threshold: never let a 1–2 core machine
/// drive the gate so high it effectively disables the fusion forever. 256 sits just
/// above the highest measured crossover.
const THRESHOLD_CEILING: usize = 256;

/// Slope of the additive-log core-scaling, in feature-counts per doubling of cores.
/// Fitted to the measured BFS crossover deltas (≈30 at 2 cores → ≈80 at 16 cores over
/// log2-span 3 ⇒ ≈17 per log2 step). Applied to BOTH anchors so the relative shape is
/// shared; the per-anchor offset (100 vs 130) shifts the whole curve.
const THRESHOLD_LOG2_SLOPE: f64 = 17.0;

/// The rayon global-pool size, queried EXACTLY ONCE. The unified
/// threshold fns are called per-leaf on a hot path, so we cache rather than re-query.
///
/// Input source = [`rayon::current_num_threads`] (NOT
/// [`std::thread::available_parallelism`]): `current_num_threads` returns the actual
/// global rayon pool size the fork/join actually runs on, and it HONORS
/// `RAYON_NUM_THREADS`. The threshold measurement sweep drove the pool with
/// `RAYON_NUM_THREADS`, so reading the same value here makes the measured curve and the
/// production default agree. `available_parallelism` reports the hardware count and would
/// DIVERGE from the pool whenever `RAYON_NUM_THREADS` is set — silently breaking that
/// agreement.
fn rayon_cores() -> usize {
    use std::sync::OnceLock;
    static CORES: OnceLock<usize> = OnceLock::new();
    *CORES.get_or_init(|| rayon::current_num_threads().max(1))
}

/// Core-count-derived default for a unified-fusion gate threshold.
///
/// `anchor_at_16` is the hand-measured optimum at THIS machine's 16 logical cores (the
/// shipped constants: 100 for BFS, 130 for subscan). Measurement (a `RAYON_NUM_THREADS`
/// sweep, warm multi-run medians) found the win-crossover feature count RISES roughly
/// logarithmically with core count for BOTH fusions: more cores ⇒ each rayon fork/join's
/// sync overhead grows ⇒ the single-fork/join fusion needs more per-feature work to beat
/// the two-step's double fork/join. So we shape the default as
///
/// ```text
/// threshold = clamp( anchor_at_16 − SLOPE · log2(16 / cores),  FLOOR, CEILING )
/// ```
///
/// which reproduces `anchor_at_16` EXACTLY at 16 cores (the hard no-regression invariant
/// for this box — the one point the proxy sweep can fully trust), drops below it on
/// fewer-core machines (fusion engages earlier), and rises above it on many-core machines
/// (fusion engages later). Clamped to `[FLOOR, CEILING]` so neither extreme gets a
/// pathological value. PROXY CAVEAT: the off-16-core shape was measured by capping the
/// rayon pool on a 16-core box, which isolates parallelism but NOT a real low-core
/// machine's smaller cache / lower bandwidth — so off-16 it is a heuristic, and the
/// `LGBM_UNIFIED_*` env overrides remain the escape hatch.
fn core_scaled_threshold(anchor_at_16: usize, cores: usize) -> usize {
    let cores = cores.max(1) as f64;
    // log2(16/cores): +ve below 16 cores (lower threshold), 0 at 16, −ve above (higher).
    let delta = (16.0_f64 / cores).log2();
    let raw = anchor_at_16 as f64 - THRESHOLD_LOG2_SLOPE * delta;
    // Round to nearest, then clamp. `max(0)` guards the (clamped-away) negative case.
    let rounded = raw.round().max(0.0) as usize;
    rounded.clamp(THRESHOLD_FLOOR, THRESHOLD_CEILING)
}

/// The feature-count at/above which the directly-built (smaller/root) leaf's
/// per-feature `{build histogram → fix_histogram → compact → scan}` runs inside ONE
/// rayon region (the host f64 analog of the GPU `build_fix_scan_resident` fusion)
/// instead of the two-step `build_leaf_histogram_into` +
/// `find_best_splits_batched`.
///
/// Keyed on `feats.len()` — the SAME scan-work proxy as [`par_scan_threshold`] —
/// because the contention this lever removes (a parallel scan `par_iter` fighting
/// the already-parallel build `par_iter`) scales with the feature count, and narrow
/// leaves were catastrophic in measurement. Below the threshold the leaf takes the
/// byte-unchanged serial two-step path. Override via `LGBM_UNIFIED_BFS_THRESHOLD`.
///
/// DEFAULT: CONDITIONAL WIN at `feats.len() >= 100`. The unified region removes the
/// cross-region contention (a parallel scan `par_iter` fighting the already-parallel
/// build `par_iter`) by fusing build+fix+compact+scan for each spine feature into ONE
/// rayon fork/join, keeping each feature's histogram cache-hot in its building thread
/// through fix and scan. Measurement showed a sign-stable train-wall win on wide leaves
/// (≥100 features) and a sign-stable regression on narrow leaves (single rayon
/// fork/join overhead dominates), so narrow MUST stay serial two-step. Tuned default =
/// 100: narrow/medium (≤90 feat) keep the byte-unchanged serial two-step path (zero
/// regression); genuinely wide leaves (≥100 feat) take the unified region for the
/// train-wall gain. Set `LGBM_UNIFIED_BFS_THRESHOLD=0` to force the unified path on
/// for the bit-exact parity proof; set a large value (or `usize::MAX`) to force the
/// serial two-step path.
///
/// DEFAULT IS NOW CORE-DERIVED: the constant `100` was measured only at THIS box's 16
/// cores. The default is `core_scaled_threshold(100, rayon_cores())`, which reproduces
/// `100` exactly at 16 cores (zero local regression) and scales the crossover down on
/// fewer-core / up on more-core machines per the measured curve. The
/// `LGBM_UNIFIED_BFS_THRESHOLD` env var still takes ULTIMATE precedence.
pub fn unified_bfs_threshold() -> usize {
    std::env::var("LGBM_UNIFIED_BFS_THRESHOLD")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or_else(|| core_scaled_threshold(100, rayon_cores()))
}

/// The feature-count at/above which the subtract-derived (larger / use_subtract) child's
/// per-feature `{subtract → scan}` runs inside ONE rayon region (the host f64 analog of
/// [`unified_bfs_threshold`] but for the larger child, fused subtract+scan with NO build
/// and NO fix) instead of the two-step whole-buffer
/// [`subtract_histograms`](Backend::subtract_histograms) + a separate
/// [`find_best_splits_batched`](Backend::find_best_splits_batched) scan. Override via
/// `LGBM_UNIFIED_SUBSCAN_THRESHOLD`.
///
/// Keyed on `feats.len()` (the same scan-work proxy as [`unified_bfs_threshold`]). The
/// larger child has NO parallel build to contend with (only a cheap serial subtract), so
/// the build/scan contention the smaller-child fusion removes is STRUCTURALLY ABSENT here
/// — yet the measured crossover is MATERIALLY higher than the smaller-child threshold
/// (100), so a SEPARATE env is justified (the larger child's single rayon fork/join over
/// the subtract+scan only amortizes above ~130 features; below that it is
/// overlapping/within-spread).
///
/// DEFAULT: WIN at `feats.len() >= 130`. Tuned default = 130: the near-threshold zone
/// (overlapping, not sign-stable) keeps the byte-unchanged two-step path; genuinely wide
/// leaves (≥130 feat) take the fused subtract→scan for the sign-stable train-wall gain.
/// Set `LGBM_UNIFIED_SUBSCAN_THRESHOLD=0` to force the fused path on for the bit-exact
/// parity proof; `usize::MAX` to force two-step.
///
/// DEFAULT IS NOW CORE-DERIVED: the constant `130` was measured only at THIS box's 16
/// cores. The default is `core_scaled_threshold(130, rayon_cores())`, which reproduces
/// `130` exactly at 16 cores (zero local regression) and scales the crossover per the
/// measured curve (same additive-log shape as the BFS gate, offset to this larger
/// child's higher anchor). The `LGBM_UNIFIED_SUBSCAN_THRESHOLD` env var still takes
/// ULTIMATE precedence.
pub fn unified_subscan_threshold() -> usize {
    std::env::var("LGBM_UNIFIED_SUBSCAN_THRESHOLD")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or_else(|| core_scaled_threshold(130, rayon_cores()))
}

/// The compute backend seam.
///
/// Binds a concrete CubeCL [`Runtime`](cubecl::Runtime) (CPU or ROCm/HIP) that
/// kernels are dispatched to. The coarse whole-kernel ops live on this
/// trait: [`construct_histograms`](Backend::construct_histograms),
/// `find_best_split`, and `data_partition`.
///
/// This trait is the ONLY place where CubeCL runtime types should appear; that
/// is the whole point of the seam.

/// A resident leaf's WINNING split paired with its winning feature-POSITION (fpos) — the
/// ~8-int CUDASplitInfo-equivalent the on-device cross-feature argmax
/// ([`Backend::scan_resident_leaf_argmax`]) returns per leaf INSTEAD of the full
/// per-feature `Vec<SplitInfo>`. `fpos == -1` ⇒ no admissible split.
pub type ResidentSplitWinner = (SplitInfo, i32);

/// The DEVICE-RESIDENT best-first frontier state — the precondition for the
/// no-blocking control loop to retire the host readback. It owns:
///
/// - a resident per-leaf best-split SoA ([`SplitSoa`](kernels::best_split::SplitSoa)) sized
///   `num_leaves` — the §8.2 winners live here on device, addressed by leaf id;
/// - a device `best_leaf` i32 slot — the §8.3 cross-leaf argmax winner, written on device;
/// - a device `stop` i32 flag — the per-iteration stop signal (the ONLY host read the
///   no-blocking loop needs).
///
/// The frontier reduce ([`frontier_reduce_leaf`](DeviceFrontier::frontier_reduce_leaf)) and
/// best-leaf pick ([`frontier_pick_best_leaf`](DeviceFrontier::frontier_pick_best_leaf))
/// write into these device buffers and hand off BY HANDLE — no device→host readback happens
/// inside the reduction (the ONLY transfer is §8.3's single 8-int export). On the cubecl-cpu
/// f64 anchor the reductions are BIT-EXACT to the host folds. This does NOT drive the grow
/// loop — it builds the resident state the control loop consumes.
#[derive(Debug)]
pub struct DeviceFrontier<R: cubecl::Runtime> {
    /// The resident per-leaf best-split records (sized `num_leaves`); `valid` starts 0.
    records: kernels::best_split::SplitSoa,
    /// The device best-leaf i32 slot (1 element), written by the §8.3 argmax on device.
    best_leaf: cubecl::server::Handle,
    /// The device per-iteration stop i32 flag (1 element); read by the control loop only.
    stop: cubecl::server::Handle,
    /// The frontier width = the resident SoA length.
    num_leaves: usize,
    /// Ties the frontier to its CubeCL runtime `R` without storing one.
    _runtime: std::marker::PhantomData<fn() -> R>,
}

impl<R: cubecl::Runtime> DeviceFrontier<R> {
    /// Allocate a zeroed frontier of `num_leaves` slots on the device (records `valid=0`,
    /// `best_leaf=-1.0`, `stop=0`). The `best_leaf` slot is `f64` (the winning leaf INDEX as an
    /// exact-integer `f64`, `-1.0` = none) — matching the `f64` resident SoA (see
    /// [`SplitSoa`](kernels::best_split::SplitSoa)).
    #[must_use]
    pub fn new(client: &ComputeClient<R>, num_leaves: usize) -> Self {
        use cubecl::prelude::CubeElement;
        let records = kernels::best_split::SplitSoa::zeroed(client, num_leaves);
        let best_leaf = client.create_from_slice(f64::as_bytes(&[-1.0f64]));
        let stop = client.create_from_slice(i32::as_bytes(&[0i32]));
        Self {
            records,
            best_leaf,
            stop,
            num_leaves,
            _runtime: std::marker::PhantomData,
        }
    }

    /// The frontier width (`num_leaves`).
    #[must_use]
    pub fn num_leaves(&self) -> usize {
        self.num_leaves
    }

    /// A borrow of the resident per-leaf best-split SoA (the control loop reads it by handle).
    #[must_use]
    pub fn records(&self) -> &kernels::best_split::SplitSoa {
        &self.records
    }

    /// A borrow of the device `best_leaf` i32 slot handle.
    #[must_use]
    pub fn best_leaf_handle(&self) -> &cubecl::server::Handle {
        &self.best_leaf
    }

    /// A borrow of the device per-iteration `stop` i32 flag handle.
    #[must_use]
    pub fn stop_handle(&self) -> &cubecl::server::Handle {
        &self.stop
    }

    /// §8.2 device-resident cross-feature reduce for ONE leaf: reduce the per-task `in_slab`
    /// records into resident frontier slot `out_leaf` on device (no readback).
    ///
    /// # Errors
    /// Propagates [`kernels::best_split::sync_best_split_for_leaf_device`].
    pub fn frontier_reduce_leaf(
        &self,
        client: &ComputeClient<R>,
        in_slab: &kernels::best_split::SplitSoa,
        num_tasks: usize,
        is_smaller: bool,
        out_leaf: usize,
    ) -> Result<(), ComputeError> {
        kernels::best_split::sync_best_split_for_leaf_device(
            client,
            in_slab,
            num_tasks,
            is_smaller,
            &self.records,
            out_leaf,
        )
    }

    /// §8.3 device-resident cross-leaf argmax + self-invalidation + 8-int export: pick the
    /// best leaf over `[0, cur_num_leaves)` into the device `best_leaf` slot, self-invalidate
    /// the chosen + freshly-created slots in the resident frontier, and export the 8-int
    /// buffer (the single readback). Returns the 8-int export.
    ///
    /// # Errors
    /// Propagates [`kernels::best_split::find_best_from_all_splits_device`].
    pub fn frontier_pick_best_leaf(
        &self,
        client: &ComputeClient<R>,
        smaller_leaf_index: i32,
        larger_leaf_index: i32,
        cur_num_leaves: usize,
    ) -> Result<kernels::best_split::PickExport, ComputeError> {
        kernels::best_split::find_best_from_all_splits_device(
            client,
            &self.records,
            &self.best_leaf,
            smaller_leaf_index,
            larger_leaf_index,
            cur_num_leaves,
        )
    }

    /// Read back the device `best_leaf` slot (TEST/DEBUG; the control loop keeps it resident).
    /// The slot is an exact-integer `f64` (`-1.0` = none) decoded to `i32`.
    #[must_use]
    pub fn read_best_leaf(&self, client: &ComputeClient<R>) -> i32 {
        use cubecl::prelude::CubeElement;
        f64::from_bytes(&client.read_one_unchecked(self.best_leaf.clone()))[0] as i32
    }
}

pub trait Backend {
    /// The concrete CubeCL runtime this backend dispatches kernels to.
    type Runtime: cubecl::Runtime;

    /// Construct a single feature column's gradient/hessian histogram (a
    /// whole-kernel op, faithful to `dense_bin.hpp:99-141`).
    ///
    /// Inputs (sourced from the binned store — do NOT re-bin):
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
    /// accumulated into `f64` cells (`hist_t = double`) on the single-owner
    /// ordered fold proven bit-exact.
    ///
    /// # Errors
    /// Returns [`ComputeError::LengthMismatch`] if `ordered_gradients`/
    /// `ordered_hessians`/`binned` lengths differ, or
    /// [`ComputeError::BinIndexOutOfRange`] if any `binned[i] >= num_bin` (V5
    /// boundary validation) — never a panic / UB.
    fn construct_histograms(
        &self,
        client: &ComputeClient<Self::Runtime>,
        binned: &[u32],
        ordered_gradients: &[f32],
        ordered_hessians: &[f32],
        num_bin: u32,
    ) -> Result<Vec<f64>, ComputeError>;

    /// Find the best split threshold for a feature column (a whole-kernel op,
    /// gain math in-kernel), faithful to
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
    /// - `offset` / `default_bin` / `most_freq_bin` — the
    ///   `FeatureGroup`/`Bin` bin-layout descriptors driving the
    ///   `SKIP_DEFAULT_BIN` continue and the threshold offset arithmetic.
    /// - `skip_default_bin` / `na_as_missing` — the AUTHORITATIVE C++ dispatch
    ///   flags (`feature_histogram.hpp:284-285`), derived by the caller from the
    ///   feature's `missing_type` + `num_bin > 2`
    ///   (`skip == (num_bin > 2 && missing_type == Zero)`,
    ///   `na_as_missing == (num_bin > 2 && missing_type == NaN)`, both false for
    ///   `missing_type == None`). These REPLACE a prior
    ///   `cfg_skip_default_bin(default_bin, num_bin)` heuristic that did not
    ///   match the C++ dispatch table.
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
    ///   [`ComputeError::Runtime`] (the NA_AS_MISSING forward branch is deferred
    ///   — never a silent wrong answer).
    /// - `sum_gradient` / `sum_hessian` / `num_data` — the leaf totals.
    ///
    /// Returns a [`SplitInfo`]; `gain == f64::NEG_INFINITY` (C++ `kMinScore`)
    /// signals "no valid split found".
    ///
    /// # Errors
    /// [`ComputeError::LengthMismatch`] if `hist.len() != 2 * num_bin`, or
    /// [`ComputeError::Runtime`] for `num_bin == 0`, non-positive `sum_hessian`,
    /// `na_as_missing == true` (deferred branch), or unsupported non-default gain
    /// params (V5).
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
    /// learner owns `leaf_begin_`/`leaf_count_` bookkeeping; this op
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

    /// **Native-width** sibling of [`data_partition`](Backend::data_partition)
    /// (ADDITIVE — the `data_partition(&[u32])` signature and every caller are
    /// byte-unchanged).
    ///
    /// Takes the leaf's bins as a narrow [`BinColumn`] (u8/u16/u32) instead of a
    /// u32-widened slice. The DEFAULT body simply WIDENS and delegates to
    /// [`data_partition`](Backend::data_partition), so CpuBackend (and every backend
    /// that does not override) is byte-identical to HEAD with ZERO edits. GPU backends
    /// OVERRIDE this to upload the bins at their native width (4× fewer bytes on the
    /// common all-u8 `max_bin≤255` case), which is bit-exact to the widened path
    /// because the u8/u16/u32 route kernels read the same bin value via `u32::cast_from`.
    ///
    /// # Errors
    /// Same V5 as [`data_partition`](Backend::data_partition): [`ComputeError::Runtime`]
    /// if `num_bin == 0` or `threshold >= num_bin`; [`ComputeError::BinIndexOutOfRange`]
    /// for any `bins.bin(i) >= num_bin`.
    #[allow(clippy::too_many_arguments)]
    fn data_partition_native(
        &self,
        client: &ComputeClient<Self::Runtime>,
        bins: &BinColumn,
        num_bin: u32,
        min_bin: u32,
        max_bin: u32,
        threshold: u32,
        most_freq_bin: u32,
    ) -> Result<(Vec<u32>, usize), ComputeError> {
        // DEFAULT: widen (cold) and delegate — byte-identical to the prior u32 path.
        self.data_partition(
            client,
            &bins.to_u32_vec(),
            num_bin,
            min_bin,
            max_bin,
            threshold,
            most_freq_bin,
        )
    }

    /// Partition a leaf's resident row sub-range via the §9 `mark → prefix-sum →
    /// scatter` and write the child left/right start/end/count into the resident
    /// [`DeviceLeafSplits`](kernels::partition::DeviceLeafSplits) slot `leaf_id` ON
    /// DEVICE — the split point is NOT returned to the host, avoiding a host
    /// round-trip for it. The child ranges live, resident, in the device struct (the
    /// control loop reads them by handle); the returned value is the stable-ordered
    /// GLOBAL row permutation the caller scatters into the resident buffer.
    ///
    /// Default: a typed error (the device-resident partition is a GpuBackend capability;
    /// the CpuBackend anchor uses the host fold
    /// [`data_partition_native`](Backend::data_partition_native) +
    /// `partition_leaf_stable`). GpuBackend runs the §9 device path. Bit-exact to the cpu
    /// f64 anchor — the returned permutation is byte-identical to
    /// [`data_partition_native`](Backend::data_partition_native) and the child ranges
    /// equal the host split point.
    ///
    /// # Errors
    /// [`ComputeError::Runtime`] on the default (device-resident partition unsupported);
    /// on GpuBackend propagates the §9 device path error (bad `num_bin`/threshold, an
    /// out-of-range bin, a length mismatch, `leaf_id` out of range, or a launch failure).
    #[allow(clippy::too_many_arguments)]
    fn data_partition_resident_no_readback(
        &self,
        _client: &ComputeClient<Self::Runtime>,
        _bins: &BinColumn,
        _data_indices: &[u32],
        _num_bin: u32,
        _min_bin: u32,
        _max_bin: u32,
        _default_bin: u32,
        _most_freq_bin: u32,
        _missing_type: u8,
        _default_left: bool,
        _threshold: u32,
        _leaf_splits: &kernels::partition::DeviceLeafSplits<Self::Runtime>,
        _leaf_id: usize,
        _p_begin: i32,
        _p_count: i32,
    ) -> Result<Vec<u32>, ComputeError> {
        Err(ComputeError::Runtime {
            detail: "data_partition_resident_no_readback: device-resident child-range \
                     partition not supported on this backend (the CpuBackend anchor uses \
                     the host partition fold)"
                .to_string(),
        })
    }

    /// Derive the larger child's histogram via the subtraction trick
    /// (`parent - child`), the kernel-layer MATH of `FeatureHistogram::Subtract`
    /// (`feature_histogram.hpp:99`). WHICH child is subtracted (the smaller
    /// sibling) is learner orchestration (the subtract OP itself is in-scope at
    /// the kernel layer).
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
    /// is the batched per-leaf abstraction seam: the learner calls it
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
    /// # Bin-range precondition (V5)
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
    /// with no per-element validation, trusting the binning invariant. Measurement
    /// showed ANY per-element check (early-return OR branchless clamp+OOB-flag)
    /// serializes the fold and regresses the build meaningfully; the branchless
    /// form wins at all scales tested.
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
        // Gather the ordered gradients/hessians ONCE per leaf — they are
        // identical across every feature (only the bin column differs), so a
        // per-feature re-gather would repeat this work `num_features` times. Mirrors
        // C++ `ordered_gradients_`/`ordered_hessians_` reuse. Values + order
        // unchanged ⇒ bit-exact.
        let r = leaf_rows.len();
        let mut ord_g: Vec<f32> = Vec::with_capacity(r);
        let mut ord_h: Vec<f32> = Vec::with_capacity(r);
        for &row in leaf_rows {
            ord_g.push(gradients[row as usize]);
            ord_h.push(hessians[row as usize]);
        }
        // The per-feature bin gather is FUSED into the fold. Read `bins[row]`
        // inline and fold directly into a REUSED per-feature hot scratch (sized to the
        // widest feature, <= 2*max_num_bin) — NOT `ord_bins` materialization, NOT a
        // per-feature alloc, and NOT a fold into the big multi-feature `out` buffer
        // (folding into `out` directly cache-scatters and regresses large leaves). The
        // fold is BRANCHLESS: no per-element bin check (see the precondition doc above),
        // only a `debug_assert!`. The f64 fold ORDER is byte-identical to
        // `construct_histograms_cpu_native` — ascending `leaf_rows`, grad at `bin<<1`,
        // hess at `+1`, f32-read -> f64-accumulate — so the bit-exact gate holds.
        //
        // The bin column is NARROW ([`BinColumn`], u8/u16/u32). Dispatch
        // on the width ONCE per feature (OUTSIDE the row loop) so each arm is a
        // MONOMORPHIC tight loop reading the narrow element directly — the
        // cache-density win lives here (no per-element width branch / accessor in the
        // hot loop). The fold ORDER and the `bin as usize * 2` index arithmetic are
        // IDENTICAL across arms and identical to the prior u32 path ⇒ bit-exact.
        // Build each feature's histogram in parallel across rayon WHEN the
        // leaf is big enough to amortize task dispatch (>= par_build_threshold), else the
        // serial reused-scratch path. Both call the SAME `fold_one_feature` body with the
        // SAME ascending order into disjoint `out` regions ⇒ byte-identical result; the
        // per-feature independence makes it thread-count-deterministic (bit-exact gate
        // holds). The threshold protects small/medium leaves — unconditional parallel
        // regressed small trains significantly on rayon dispatch overhead.
        let parallel = r >= par_build_threshold();
        let out = build_histograms_into(
            feature_bins, num_bins, slot_off, slot_len, leaf_rows, &ord_g, &ord_h, parallel,
        );
        Ok(out)
    }

    /// Find the best split for EVERY spine feature of ONE leaf in a single batched
    /// op: the fused per-leaf SPLIT SCAN over the concatenated
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
    /// bit-exact.
    ///
    /// The DEFAULT impl (used by [`CpuBackend`] unchanged) loops
    /// [`find_best_split`](Backend::find_best_split) over `feats` in order, so each
    /// feature's [`SplitInfo`] is byte-identical to today's per-feature call — the
    /// default IS the bit-exact f64 anchor. A GPU backend OVERRIDES this to find all
    /// features' splits in one launch per leaf.
    ///
    /// An empty `feats` (every feature gated out / categorical-only leaf) returns an
    /// empty Vec with no launch.
    ///
    /// # Errors
    /// Propagates [`find_best_split`](Backend::find_best_split) errors; returns
    /// [`ComputeError::LengthMismatch`] if any feature's
    /// `[slot_off, slot_off + 2*num_bin)` region exceeds `buf` (V5 — no panic / UB).
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

    /// One-time per-train upload of the binned feature columns to the device.
    /// The learner calls this ONCE per `train_inner` (before the
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
        _feature_bins: &[&BinColumn],
    ) {
    }

    /// Mark the CURRENT resident-bin cache as OWNED by a once-per-train guard (the
    /// learner's `resident_bins_uploaded` flag), i.e. the caller guarantees the
    /// uploaded columns are the immutable feature set for every grow until the next
    /// upload. The on-device grow driver may then SKIP its per-grow re-upload
    /// ([`resident_bins_pinned`](Backend::resident_bins_pinned)) — measurement found
    /// this per-grow re-upload was the largest single contributor to the real-CUDA
    /// on-device performance gap.
    ///
    /// DEFAULT: no-op (CpuBackend never takes the resident grow arm). Any subsequent
    /// [`upload_resident_bins`](Backend::upload_resident_bins) CLEARS the pin (a fresh
    /// upload means a new owner must re-pin), so un-pinned direct-driver callers keep
    /// the exact per-grow re-upload behavior — the skip can never alias two different
    /// same-shape corpora.
    fn pin_resident_bins(&self) {}

    /// Whether the resident-bin cache is PINNED by a once-per-train owner
    /// AND matches this grow's geometry (`num_features`, `num_data`). Only the
    /// combination (pin + geometry) authorizes the on-device driver to skip its
    /// per-grow `upload_resident_bins`. DEFAULT: `false` (never skip).
    fn resident_bins_pinned(&self, _num_features: usize, _num_data: usize) -> bool {
        false
    }

    /// One-time per-GROW upload of the current tree's gradients + hessians to the
    /// device — the grad/hess analog of
    /// [`upload_resident_bins`](Backend::upload_resident_bins). The on-device resident
    /// grow calls this ONCE per tree (grad/hess are constant across a grow but change
    /// every boosting iteration), then the resident histogram build gathers each leaf's
    /// grad/hess ON DEVICE from the cached buffers via the leaf-row index — eliminating
    /// the per-build host `ord_g`/`ord_h` gather + `create_from_slice` re-upload of the
    /// full grad/hess on every histogram build.
    ///
    /// The DEFAULT impl is a NO-OP: [`CpuBackend`] is the bit-exact host anchor and
    /// never takes the device-resident grow arm (`resident_pool_supported() == false`),
    /// so this seam adds ZERO behavior change to the CPU path. `gradients`/`hessians`
    /// are the full per-row buffers (length `num_data`); RocmBackend overrides this.
    fn upload_resident_grad_hess(
        &self,
        _client: &ComputeClient<Self::Runtime>,
        _gradients: &[f32],
        _hessians: &[f32],
    ) {
    }

    /// Whether [`upload_resident_bins`](Backend::upload_resident_bins) actually
    /// consumes its `&[&[u32]]` argument. With narrow [`BinColumn`]
    /// storage, the learner must WIDEN each column to `u32` to call
    /// `upload_resident_bins`; that widening allocates `num_features` u32 Vecs ONCE
    /// per `train_inner`. On [`CpuBackend`] the upload is a no-op, so the learner
    /// SKIPS the widening entirely (gated on this returning `false`) — avoiding a
    /// per-tree allocation that has no effect. RocmBackend returns `true` so its
    /// resident u32 upload still receives the byte-identical column data.
    fn wants_resident_bins(&self) -> bool {
        false
    }

    /// Whether the caller's `DataPartition::split` numeric branch should route a
    /// leaf's rows on the HOST, directly off the narrow [`BinColumn`] (the
    /// fused u8-route path), instead of widening to `&[u32]` and calling
    /// [`data_partition`](Backend::data_partition).
    ///
    /// `true` ⇒ the caller's `DataPartition::split` numeric branch should route a
    /// leaf's rows on the HOST: ONE random gather + a ¼-width `u8` route scratch +
    /// ONE `u32` scatter, producing a byte-identical `[left | right]` order to the
    /// materialize-then-op (`data_partition`) path. [`CpuBackend`] is the bit-exact
    /// host anchor and returns `true`. `RocmBackend` keeps the default `false` so
    /// it routes on-device via its [`data_partition`](Backend::data_partition)
    /// override (no host materialization stolen from the GPU upload path).
    fn prefers_host_partition(&self) -> bool {
        false
    }

    /// Whether this backend implements the CPU-only HOST unified fused paths
    /// [`build_fix_scan`](Backend::build_fix_scan) and
    /// [`subtract_scan`](Backend::subtract_scan). `false` (the default) means the
    /// learner's `smaller_unified`/`larger_unified` gates skip those paths and route
    /// through the standard `construct_histograms` + batched `find_best_split` instead.
    /// Only [`CpuBackend`] — the host f64 anchor that overrides both — returns `true`.
    ///
    /// This is a SEPARATE capability from [`resident_pool_supported`](Backend::resident_pool_supported):
    /// a GPU backend WITHOUT a resident pool (e.g. CudaBackend/WgpuBackend) is neither
    /// resident NOR host-unified, so `!resident_pool_supported()` alone is NOT a
    /// sufficient gate for the unified host path (that gate previously routed such a
    /// backend into the erroring `build_fix_scan` default).
    fn host_unified_fused_supported(&self) -> bool {
        false
    }

    // ===================================================================
    // Async device→host copy + multi-stream capability seam.
    //
    // These encode — IN CODE — the cubecl-0.10 boundary established in
    // `docs/on-device-streams-M4-M5.md` (verified against the installed
    // cubecl-runtime-0.10.0, NOT from memory), so the finding is a documented
    // limit + achievable equivalent rather than a silent no-op. Additive defaults; no
    // backend behavior changes (CpuBackend inherits the defaults untouched).
    // ===================================================================

    /// Whether this backend can issue an ASYNC batched device→host copy.
    ///
    /// `true` (the default) because cubecl 0.10 exposes
    /// `ComputeClient::read_async(Vec<Handle>) -> impl Future` and its blocking batched
    /// drain `ComputeClient::read(Vec<Handle>) -> Vec<Bytes>` for every runtime
    /// (`cubecl-runtime-0.10.0/src/client.rs:109` / `:131`). `read_one_unchecked(h)` is
    /// literally the `n = 1` case `read_sync(read_async(vec![h]))` (`client.rs:145`), so a
    /// single `read(Vec<Handle>)` drains N handles in ONE sync flush where a per-handle loop
    /// re-pays the drain N times (the [[gpu-lazy-dispatch-deferred-sync-win]] finding).
    ///
    /// The on-device resident driver already sits at this single-drain floor: the co-packed
    /// sibling scan drains BOTH children's outputs in one readback, and every
    /// other per-split readback consolidates one handle.
    fn supports_async_device_copy(&self) -> bool {
        true
    }

    /// Whether this backend can express TRUE per-operation multi-stream overlap (the
    /// §8 smaller=stream0 / larger=stream1 build+scan and §9 partition scatter/copy on
    /// streams 1–3 that `docs/cuda-kernel-design.md` describes).
    ///
    /// `false` (the default, and no backend overrides it) — the DOCUMENTED cubecl-0.10
    /// limit (`docs/on-device-streams-M4-M5.md` §(b)): stream identity is thread-derived
    /// (`client.rs:85` → `StreamId::current()`), the only explicit selector
    /// `ComputeClient::set_stream` is `pub unsafe` and documented CubeCL/Burn-internal
    /// (`client.rs:92-99`), and the runtime auto-merges streams (`config/streaming.rs`).
    /// It is also architecturally inapplicable to this driver's subtraction-trick path
    /// (larger = parent − smaller via `subtract_resident` is a strict dependency, not two
    /// independent builds). The driver therefore relies on deferred-sync batching (the
    /// single-drain `read_batched` idiom), not true streams.
    fn supports_multi_stream_overlap(&self) -> bool {
        false
    }

    /// The sanctioned SINGLE deferred drain for a genuine multi-handle device→host readback
    /// (deferred-sync batching). Delegates to cubecl's batched
    /// `ComputeClient::read(Vec<Handle>)` (`client.rs:131`), which issues ONE `read_sync`
    /// over all handles — collapsing N per-handle read-sync fixed costs to 1
    /// ([[gpu-lazy-dispatch-deferred-sync-win]]). Any future site that must read several
    /// INDEPENDENT device handles back MUST route through this instead of an N-per-handle
    /// `read_one_unchecked` loop, so the deferred-sync regression can never be
    /// reintroduced. Bit-identical bytes to reading each handle separately (pure
    /// call-ordering; numerics unchanged) — pinned on the CpuBackend arm by
    /// `read_batched_single_drain_matches_per_handle_reads`.
    ///
    /// Returns owned `Vec<u8>` per handle (in input order). The cubecl `Bytes` element type
    /// lives in a private `cubecl_runtime` module and is not nameable via a public path, so
    /// the drained buffers are copied out — a negligible host memcpy that does not affect the
    /// single-sync batching win (the ONE `read_sync` over all handles is what matters).
    fn read_batched(
        &self,
        client: &ComputeClient<Self::Runtime>,
        handles: Vec<cubecl::server::Handle>,
    ) -> Vec<Vec<u8>> {
        client.read(handles).into_iter().map(|b| b.to_vec()).collect()
    }

    // ===================================================================
    // DEVICE-RESIDENT histogram-pool seam.
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

    /// Whether this backend supports the device-resident histogram pool.
    /// `false` (the default, CpuBackend) means the learner's `resident_eligible` gate
    /// ANDs this in and ALWAYS takes the byte-unchanged host path. RocmBackend returns
    /// `true`.
    fn resident_pool_supported(&self) -> bool {
        false
    }

    /// Clear/resize the device-handle slot mirror for a new tree, called
    /// alongside the host `HistogramPool::reset_map`. Default: no-op (CpuBackend never
    /// takes the resident branch).
    fn reset_resident_pool(&self, _num_slots: usize, _slot_len: usize) {}

    /// Build ONE leaf's per-feature histogram DEVICE-RESIDENT (build → f32→f64 widen →
    /// fix → compact) and store the resulting f64 `Handle` into mirror slot `slot`.
    /// Mirrors `build_leaf_histogram_into` but keeps the histogram on
    /// device. `fix_feats[fpos]` is `(slot_off, num_bin, offset, most_freq_bin)` for
    /// feature `fpos`; `sum_gradient` / `sum_hessian` are the leaf RAW (un-bumped)
    /// totals. Default: typed error (never called on cpu — the gate is off).
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

    /// Move the resident Handle from `src_slot` to `dst_slot` in the device mirror,
    /// mirroring the host `HistogramPool::move_` slot reassignment so the
    /// device mirror's slot→Handle map tracks the host pool's slot→leaf map. Default:
    /// no-op.
    fn move_resident(&self, _src_slot: usize, _dst_slot: usize) {}

    /// Derive the larger child's resident histogram by the subtraction trick on device
    /// (`parent_slot` Handle − `smaller_slot` Handle → `larger_slot` Handle, no
    /// read-back). The derived larger child is NOT re-FixHistogram'd (matches
    /// host/C++). Default: typed error.
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
    /// split in ONE fused launch, reading back only the `n*12` SplitInfo
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

    /// CO-PACKED 2-slot resident scan — the co-packed analog of
    /// [`scan_resident_leaf`]. Scans BOTH siblings of a split in ONE 2-slot launch over
    /// their simultaneously-resident Handles (`smaller_slot` + `larger_slot`) with ONE
    /// `read_one_unchecked` readback (roughly halving device syncs per tree),
    /// returning `(smaller_splits, larger_splits)`. Bit-exact by construction — each
    /// feature's sequential scan is the SAME as the two single-slot scans; only WHICH
    /// launch it runs in changes. `feats` (the SHARED per-feature spine layout) and
    /// `slot_len` are the same for both siblings; the leaf `totals` (raw
    /// `(sum_gradient, sum_hessian, num_data)`) differ per sibling. Default: typed error.
    ///
    /// # Errors
    /// [`ComputeError::Runtime`] (unsupported / empty slot) on the default; propagates
    /// the co-packed split-scan errors on RocmBackend.
    #[allow(clippy::too_many_arguments)]
    fn scan_resident_siblings(
        &self,
        _client: &ComputeClient<Self::Runtime>,
        _smaller_slot: usize,
        _larger_slot: usize,
        _slot_len: usize,
        _feats: &[BatchedSplitFeature],
        _cfg: &GainConfig,
        _smaller_totals: (f64, f64, i32),
        _larger_totals: (f64, f64, i32),
    ) -> Result<(Vec<SplitInfo>, Vec<SplitInfo>), ComputeError> {
        Err(ComputeError::Runtime {
            detail: "scan_resident_siblings: device-resident pool not supported on this backend"
                .to_string(),
        })
    }

    /// §8.2 `SyncBestSplitForLeafKernel` analog: scan one resident leaf AND
    /// reduce the per-feature best splits to the SINGLE winning split ON DEVICE, returning only
    /// the winner `(SplitInfo, feature-position)` — the ~8-int CUDASplitInfo-equivalent — INSTEAD
    /// of the full `Vec<SplitInfo>` [`scan_resident_leaf`](Backend::scan_resident_leaf) reads
    /// back per feature. The cross-feature reduction folds features in fpos order with the strict
    /// `>` gain, lowest-real-feature-index tie-break so the winner is
    /// BIT-IDENTICAL to the host `argmax_over_splits`; `real_feats[fpos]` supplies each feature's
    /// real index (the tie-break key) and `feats[fpos].na_as_missing` skips a feature exactly as
    /// the host argmax `continue` does. Applies the `!(sum_h > 0) || num_data <= 0` short-circuit
    /// (returns `(SplitInfo::none(), -1)`), mirroring the driver's scan gate. The single readback
    /// is the winner only — the per-feature payload collapses. Default: typed error (the resident
    /// pool is a GpuBackend capability; the CpuBackend anchor never reaches the resident driver).
    ///
    /// # Errors
    /// [`ComputeError::Runtime`] (unsupported / empty slot) on the default; propagates the
    /// resident-scan errors on GpuBackend.
    #[allow(clippy::too_many_arguments)]
    fn scan_resident_leaf_argmax(
        &self,
        _client: &ComputeClient<Self::Runtime>,
        _slot: usize,
        _slot_len: usize,
        _feats: &[BatchedSplitFeature],
        _real_feats: &[i32],
        _cfg: &GainConfig,
        _sum_gradient: f64,
        _sum_hessian: f64,
        _num_data: i32,
    ) -> Result<ResidentSplitWinner, ComputeError> {
        Err(ComputeError::Runtime {
            detail: "scan_resident_leaf_argmax: device-resident pool not supported on this backend"
                .to_string(),
        })
    }

    /// The CO-PACKED 2-slot analog of
    /// [`scan_resident_leaf_argmax`](Backend::scan_resident_leaf_argmax) — co-scan BOTH siblings
    /// in ONE launch and reduce EACH side's per-feature splits to its winner ON DEVICE, returning
    /// `((smaller_winner, smaller_fpos), (larger_winner, larger_fpos))`. Both reductions fold in
    /// the SAME fpos order with the SAME strict-`>` lowest-real-feature-index tie-break, so each
    /// winner is bit-identical to `argmax_over_splits` over that sibling's scan. The caller
    /// pre-gates both siblings as scannable (the co-pack precondition), so there is no
    /// short-circuit here. Default: typed error.
    ///
    /// # Errors
    /// [`ComputeError::Runtime`] (unsupported / empty slot) on the default; propagates the
    /// co-packed resident-scan errors on GpuBackend.
    #[allow(clippy::too_many_arguments)]
    fn scan_resident_siblings_argmax(
        &self,
        _client: &ComputeClient<Self::Runtime>,
        _smaller_slot: usize,
        _larger_slot: usize,
        _slot_len: usize,
        _feats: &[BatchedSplitFeature],
        _real_feats: &[i32],
        _cfg: &GainConfig,
        _smaller_totals: (f64, f64, i32),
        _larger_totals: (f64, f64, i32),
    ) -> Result<(ResidentSplitWinner, ResidentSplitWinner), ComputeError> {
        Err(ComputeError::Runtime {
            detail: "scan_resident_siblings_argmax: device-resident pool not supported on this \
                     backend"
                .to_string(),
        })
    }

    /// §8.3 `FindBestFromAllSplitsKernel` analog: reduce the per-leaf best
    /// splits to the winning LEAF index — the cross-leaf argmax that replaces the host best-leaf
    /// loop on the GpuBackend arm. `leaf_best[i]` is leaf `i`'s best split; `leaf_real_feat[i]`
    /// its winning feature's real index (`-1` = no split ⇒ `i32::MAX` in the tie-break). Uses the
    /// SAME `split_gt` first-max rule (strict `>` gain, lowest real-feature-index tie-break) the
    /// host loop uses, seeding leaf `0`, so the pick is bit-identical. Default: the host fold
    /// (the CpuBackend anchor keeps this); GpuBackend runs it on-device. Returns `0` when nothing
    /// has positive gain (the caller's `best_fpos < 0 || !(gain > 0)` guard then breaks).
    fn best_leaf_reduce(&self, leaf_best: &[SplitInfo], leaf_real_feat: &[i32]) -> i32 {
        kernels::grow_driver::best_leaf_argmax(leaf_best, leaf_real_feat)
    }

    /// §6.1 `CUDAInitValuesKernel` analog: the whole-dataset root grad/hess
    /// sum — replaces the host f64 root fold on the GpuBackend arm. Default: the ordered f64 fold
    /// (ascending row order, the bit-exact anchor — the CpuBackend keeps this). GpuBackend runs
    /// the §6.1 reduction anchored bit-exact vs the integer path and ~1e-6 vs the host-CUDA fold
    /// (NEVER GPU-f32-vs-GPU-f32). Returns `(sum_gradient, sum_hessian)`.
    fn root_grad_hess_sum(
        &self,
        _client: &ComputeClient<Self::Runtime>,
        gradients: &[f32],
        hessians: &[f32],
    ) -> (f64, f64) {
        kernels::grow_driver::root_grad_hess_fold(gradients, hessians)
    }

    /// §8.2 device-resident cross-feature reduce for ONE leaf — reduce the
    /// per-task `in_slab` records into resident frontier slot `out_leaf` ON DEVICE (no
    /// readback). Default: a typed error (the device-resident frontier is a GpuBackend
    /// capability; the CpuBackend anchor uses the host fold
    /// [`sync_best_split_for_leaf_on`](kernels::best_split::sync_best_split_for_leaf_on)).
    /// GpuBackend runs the §8.2 device kernel. Bit-exact to the host fold on the cpu f64
    /// anchor.
    ///
    /// # Errors
    /// [`ComputeError::Runtime`] on the default (device-resident frontier unsupported);
    /// propagates the device reduce error on GpuBackend.
    fn frontier_reduce_leaf_device(
        &self,
        _client: &ComputeClient<Self::Runtime>,
        _frontier: &DeviceFrontier<Self::Runtime>,
        _in_slab: &kernels::best_split::SplitSoa,
        _num_tasks: usize,
        _is_smaller: bool,
        _out_leaf: usize,
    ) -> Result<(), ComputeError> {
        Err(ComputeError::Runtime {
            detail: "frontier_reduce_leaf_device: device-resident frontier not supported on \
                     this backend (the CpuBackend anchor uses the host fold)"
                .to_string(),
        })
    }

    /// §8.3 device-resident cross-leaf best-leaf pick + self-invalidation +
    /// 8-int export — the winner lives in the device `best_leaf` slot; the ONLY device→host
    /// transfer is the single 8-int export. Default: a typed error (GpuBackend
    /// capability); the CpuBackend anchor uses the host fold
    /// [`find_best_from_all_splits_on`](kernels::best_split::find_best_from_all_splits_on).
    /// Bit-identical to the host pick on the cpu f64 anchor.
    ///
    /// # Errors
    /// [`ComputeError::Runtime`] on the default; propagates the device reduce error on
    /// GpuBackend.
    fn frontier_pick_best_leaf_device(
        &self,
        _client: &ComputeClient<Self::Runtime>,
        _frontier: &DeviceFrontier<Self::Runtime>,
        _smaller_leaf_index: i32,
        _larger_leaf_index: i32,
        _cur_num_leaves: usize,
    ) -> Result<kernels::best_split::PickExport, ComputeError> {
        Err(ComputeError::Runtime {
            detail: "frontier_pick_best_leaf_device: device-resident frontier not supported on \
                     this backend (the CpuBackend anchor uses the host fold)"
                .to_string(),
        })
    }

    /// The NO-READBACK device tree split — mutate the device tree in place via the §10
    /// SplitKernel with the right child leaf id SUPPLIED BY THE CALLER from the fixed
    /// grow schedule, NOT read back from the kernel (avoiding a host round-trip for the
    /// `grow_driver.rs:1931` `right_leaf_index`). The desync invariant
    /// (`right_leaf_index == tree.num_leaves`) is asserted host-side without a
    /// readback.
    ///
    /// Default: a typed error (the device tree split is a GpuBackend capability; the
    /// CpuBackend anchor mutates the host `lgbm_model::Tree` directly). GpuBackend calls
    /// [`DeviceCudaTree::split_on_device_scheduled`](kernels::tree::DeviceCudaTree::split_on_device_scheduled).
    /// Byte-identical tree structure to the host mutation on the cpu f64 anchor.
    ///
    /// # Errors
    /// [`ComputeError::Runtime`] on the default; on GpuBackend propagates the device tree
    /// split error (leaf out of range, would exceed `max_leaves`, or a schedule/tree desync).
    #[allow(clippy::too_many_arguments)]
    fn split_tree_scheduled(
        &self,
        _client: &ComputeClient<Self::Runtime>,
        _tree: &mut kernels::tree::DeviceCudaTree<Self::Runtime>,
        _leaf_index: i32,
        _right_leaf_index: i32,
        _real_feature_index: i32,
        _real_threshold: f64,
        _missing_type: i32,
        _scalars: &kernels::split_info::SplitScalars,
    ) -> Result<(), ComputeError> {
        Err(ComputeError::Runtime {
            detail: "split_tree_scheduled: device-resident tree split not supported on this \
                     backend (the CpuBackend anchor mutates the host tree)"
                .to_string(),
        })
    }

    /// FUSED directly-built-leaf path — build + fix + compact + scan a
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
    /// (un-bumped) `sum_gradient_raw` / `sum_hessian_raw` feed the FIX, the
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

    /// The ZERO-READBACK analog of
    /// [`scan_resident_leaf_argmax`](Backend::scan_resident_leaf_argmax) — scan ONE resident
    /// leaf and fold its cross-feature winner (gain + threshold + default_left + 4 child sums +
    /// left/right output) DIRECTLY into the resident frontier slot `out_leaf` on device, with NO
    /// host argmax readback. Avoids the per-split scan host round-trip that
    /// [`scan_resident_and_argmax`](kernels::grow_driver) issues; the winner reaches the driver
    /// only via the §8.3 pick export (`frontier_pick_best_leaf_device`), which now carries the
    /// picked leaf's full node record. Default: typed error (GpuBackend capability). Bit-exact to
    /// `scan_resident_leaf_argmax` + `reduce_winner_into_frontier` on the cpu f64 anchor.
    ///
    /// # Errors
    /// [`ComputeError::Runtime`] on the default; propagates the resident-scan / reduce errors on
    /// GpuBackend.
    #[allow(clippy::too_many_arguments)]
    fn scan_resident_leaf_into_frontier(
        &self,
        _client: &ComputeClient<Self::Runtime>,
        _slot: usize,
        _slot_len: usize,
        _feats: &[BatchedSplitFeature],
        _real_feats: &[i32],
        _cfg: &GainConfig,
        _sum_gradient: f64,
        _sum_hessian: f64,
        _num_data: i32,
        _frontier: &DeviceFrontier<Self::Runtime>,
        _out_leaf: usize,
    ) -> Result<(), ComputeError> {
        Err(ComputeError::Runtime {
            detail: "scan_resident_leaf_into_frontier: device-resident frontier not supported on \
                     this backend"
                .to_string(),
        })
    }

    /// The ZERO-READBACK, CO-PACKED analog of
    /// [`scan_resident_siblings_argmax`](Backend::scan_resident_siblings_argmax) — co-scan BOTH
    /// siblings in ONE launch and fold EACH side's winner DIRECTLY into its frontier slot
    /// (`out_leaf_smaller` / `out_leaf_larger`) on device, NO host argmax readback. Avoids the
    /// co-pack arm's per-split scan host round-trip. Default: typed error. Bit-exact to two
    /// `argmax_over_resident_splits` folds + `reduce_winner_into_frontier` on the cpu f64 anchor.
    ///
    /// # Errors
    /// [`ComputeError::Runtime`] on the default; propagates the co-packed resident-scan / reduce
    /// errors on GpuBackend.
    #[allow(clippy::too_many_arguments)]
    fn scan_resident_siblings_into_frontier(
        &self,
        _client: &ComputeClient<Self::Runtime>,
        _smaller_slot: usize,
        _larger_slot: usize,
        _slot_len: usize,
        _feats: &[BatchedSplitFeature],
        _real_feats: &[i32],
        _cfg: &GainConfig,
        _smaller_totals: (f64, f64, i32),
        _larger_totals: (f64, f64, i32),
        _frontier: &DeviceFrontier<Self::Runtime>,
        _out_leaf_smaller: usize,
        _out_leaf_larger: usize,
    ) -> Result<(), ComputeError> {
        Err(ComputeError::Runtime {
            detail: "scan_resident_siblings_into_frontier: device-resident frontier not supported \
                     on this backend"
                .to_string(),
        })
    }

    /// The ZERO-READBACK analog of
    /// [`build_fix_scan_resident`](Backend::build_fix_scan_resident) (the f64-fused escape hatch)
    /// — build+fix+compact+scan a directly-built leaf in ONE launch, STORE the fixed+compacted
    /// f64 histogram Handle into mirror slot `slot` (so `subtract_resident` still finds it), AND
    /// fold the scan winner DIRECTLY into frontier slot `out_leaf` on device, NO
    /// per-feature-array readback. `scan_active` all-false ⇒ every window decodes
    /// `is_splittable=0` ⇒ the frontier slot gets the no-valid-split sentinel (the driver passes
    /// all-false when the leaf is not scannable, so the histogram is still built for the subtract
    /// but no winner is recorded). Default: typed error.
    ///
    /// # Errors
    /// [`ComputeError::Runtime`] on the default; propagates the fused-build / reduce errors on
    /// GpuBackend.
    #[allow(clippy::too_many_arguments)]
    fn build_fix_scan_resident_into_frontier(
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
        _real_feats: &[i32],
        _cfg: &GainConfig,
        _sum_gradient_raw: f64,
        _sum_hessian_raw: f64,
        _num_data: i32,
        _frontier: &DeviceFrontier<Self::Runtime>,
        _out_leaf: usize,
    ) -> Result<(), ComputeError> {
        Err(ComputeError::Runtime {
            detail: "build_fix_scan_resident_into_frontier: device-resident fused path not \
                     supported on this backend"
                .to_string(),
        })
    }

    /// UNIFIED host per-feature `{build → fix → compact → scan}` for
    /// the directly-built (smaller/root) leaf, run inside ONE rayon region — the host f64
    /// analog of [`build_fix_scan_resident`](Backend::build_fix_scan_resident).
    ///
    /// Each feature folds its OWN private histogram (cache-hot in the building thread),
    /// runs `fix_histogram` (RAW sums + `most_freq_bin` reconstruct) + `compact`
    /// (`offset` shift) IN PLACE on it, and — IF `scan_active[fpos]` — scans it, all
    /// WITHOUT a cross-region fork/join hand-off (avoiding the contention a parallel
    /// scan `par_iter` would cause fighting the parallel build `par_iter`). After the
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
    /// (un-bumped) leaf totals (the `fix` operand).
    ///
    /// Default: typed error — only [`CpuBackend`] overrides this (the unified path is the
    /// CPU-only host analog; RocmBackend keeps `build_fix_scan_resident`). The learner's
    /// `smaller_unified` gate ANDs in [`host_unified_fused_supported`](Backend::host_unified_fused_supported)
    /// (CpuBackend-only) so this is never reached on a GPU backend — including one without
    /// a resident pool (CudaBackend/WgpuBackend), for which `!resident_pool_supported()`
    /// alone was insufficient.
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

    /// UNIFIED host per-feature `{subtract → scan}` for the
    /// subtract-derived (larger / use_subtract) child, run inside ONE rayon region — the
    /// host f64 analog of [`build_fix_scan`](Backend::build_fix_scan) but for the larger
    /// child, with NO build and NO fix (non-negotiable #3: C++ runs no FixHistogram on the
    /// use_subtract larger child; both operands are already fixed+compacted).
    ///
    /// Each feature computes its OWN private region `parent[start..end] −
    /// smaller[start..end]` (the exact cell-wise op of
    /// [`subtract_histograms`](Backend::subtract_histograms) over the disjoint range) and
    /// — IF `scan_active[fpos]` — scans it, WITHOUT a cross-region fork/join hand-off. After
    /// the region a SERIAL ordered loop copies each private region into its disjoint
    /// `larger_buf[slot_off..]` range (the COMPLETE larger-child histogram) and assembles
    /// the per-feature `Option<SplitInfo>` results in feature-index order.
    ///
    /// BIT-EXACT: each feature is independent — same f64 cells, same `p − s` op, same
    /// order as the two-step `subtract_histograms` + per-feature scan; the (serial,
    /// feature-order) argmax is unchanged ⇒ byte-identical to the two-step path (proven
    /// FORCED-ON via `LGBM_UNIFIED_BFS_THRESHOLD=0`). `sum_gradient`/`sum_hessian` are the
    /// larger child's RAW leaf totals (passed to the scan only — no fix uses them here).
    ///
    /// Default: typed error — only [`CpuBackend`] overrides this (the unified path is the
    /// CPU-only host analog; the resident/GPU larger child keeps `subtract_resident`). The
    /// learner's `larger_unified` gate ANDs in [`host_unified_fused_supported`](Backend::host_unified_fused_supported)
    /// (CpuBackend-only) so this is never reached on a GPU backend — including one without
    /// a resident pool (CudaBackend/WgpuBackend), for which `!resident_eligible` alone was
    /// insufficient.
    ///
    /// # Errors
    /// [`ComputeError::Runtime`] (unsupported) on the default; on CpuBackend,
    /// [`ComputeError::LengthMismatch`] for an out-of-range feature region (ascending,
    /// validated against parent/smaller/larger ⇒ deterministic lowest-index error) or a
    /// propagated [`find_best_split`](Backend::find_best_split) error.
    #[allow(clippy::too_many_arguments)]
    fn subtract_scan(
        &self,
        _client: &ComputeClient<Self::Runtime>,
        _parent: &[f64],
        _smaller: &[f64],
        _larger_buf: &mut [f64],
        _all_feats: &[BatchedSplitFeature],
        _scan_active: &[bool],
        _cfg: &GainConfig,
        _sum_gradient: f64,
        _sum_hessian: f64,
        _num_data: i32,
    ) -> Result<Vec<Option<SplitInfo>>, ComputeError> {
        Err(ComputeError::Runtime {
            detail: "subtract_scan: unified host subtract+scan not supported on this backend"
                .to_string(),
        })
    }

    // ===================================================================
    // CUDA on-device tree-learner seam.
    //
    // The additive `Backend` seam + discriminator that lets a backend grow an
    // ENTIRE tree on-device and return the `(Tree, LeafPartitionLayout)` payload,
    // bypassing the per-leaf host build/scan loop. Both methods are a
    // provable NO-OP on every backend by default: the discriminator defaults `false` and the
    // seam defaults `Ok(None)` ("I did not grow it"), so the default CPU/ROCm tree
    // path is byte-unchanged. Activation requires a real kernel plus a `true`
    // discriminator, and a learner fork that consumes the seam.
    // ===================================================================

    /// Whether this backend can grow an entire tree ON-DEVICE.
    ///
    /// The trait DEFAULT is `false` (a hypothetical future backend opts out until it
    /// wires the driver). [`CpuBackend`] and `GpuBackend<R>` OVERRIDE this
    /// to return [`cuda_on_device_enabled`] — GATED, not a bare `true`: with
    /// `LGBM_CUDA_ON_DEVICE` unset every backend reports `false`, so the
    /// learner's on-device eligibility gate ANDs this in and ALWAYS takes the
    /// byte-unchanged host/per-leaf path (the hard merge gate stays byte-identical).
    /// Extending the gated flip to `CpuBackend` is deliberate: the structural gate
    /// must grow the on-device tree on the cubecl-cpu runtime so it runs in the
    /// DEFAULT merge gate (the cpu-f64 anchor lane), not behind rocm hardware — the
    /// env gate keeps the env-unset merge gate byte-unchanged either way. Today
    /// the driver body still returns `Ok(None)`, so even with the env SET the fork
    /// safely falls through to the byte-identical host path (output still correct).
    fn on_device_growth_supported(&self) -> bool {
        false
    }

    /// Grow an ENTIRE tree on-device and return its model + raw leaf-row layout.
    ///
    /// Returns `Ok(None)` = "I did not grow the tree on-device" — the caller falls
    /// back to the standard host/per-leaf path. This is deliberately NOT a typed
    /// `Err(NotSupported)`: it keeps the default route error-noise-free, so an
    /// unsupported backend is a quiet `None`, not an error the learner must filter.
    ///
    /// The args are what the learner holds at the `train_inner` fork point:
    /// `gradients`, `hessians`, the ADDITIVE `features` metadata slice
    /// ([`GrowFeature`] — the per-feature bin layout the grow loop reads, expressed
    /// in ONLY lgbm-compute-reachable types so no crate cycle is introduced),
    /// `num_leaves`, and `max_depth`. The per-leaf orchestration BODY that
    /// consumes `features` returns `Ok(None)` for now.
    ///
    /// The return type names ONLY lgbm-compute-reachable crates
    /// (`lgbm_model::Tree` + `lgbm_dataset::LeafPartitionLayout`). It MUST NOT name
    /// the treelearner crate's `DataPartition` — that would require importing
    /// lgbm-treelearner here, which is the crate-cycle warning sign (treelearner →
    /// compute → treelearner). The learner reconstructs its `DataPartition` from the lower-crate
    /// `LeafPartitionLayout` payload.
    ///
    /// # cubecl-0.10 kernel checklist (for when a real kernel lands here)
    /// - NO global barrier across cubes — synchronize within a cube only.
    /// - `Atomic<i64>` is broken on this cubecl — use u64 fixed-point atomics.
    /// - `wrapping_add` is NOT a kernel intrinsic — avoid it in `#[cube]` code.
    /// - a plane-sum reduction spans at most ONE plane width — no cross-plane sum.
    /// - `launch_unchecked` is `unsafe` — uphold the launch-arg invariants by hand.
    ///
    /// # Foundation modules a future on-device grow kernel will compose
    /// The shared device primitives, split-info structs, and on-device RNG the
    /// future on-device grow loop assembles already exist and are golden-validated:
    /// [`kernels::primitives`] (block + global prefix-sum,
    /// shuffle reductions, index-only bitonic argsort, percentile),
    /// [`kernels::split_info`] (the device-side split/leaf-split structs), and
    /// [`kernels::random`] (the on-device LCG mirror). A future on-device growth
    /// consumer reuses these — this seam stays a strict no-op until
    /// then; the discriminator above is FROZEN `false`.
    ///
    /// # Errors
    /// Returns [`ComputeError`] only once a real kernel is wired that can fail
    /// (device OOM, launch error). Until then it is infallible (`Ok(None)`).
    fn grow_tree_on_device(
        &self,
        _gradients: &[f32],
        _hessians: &[f32],
        _features: &[GrowFeature],
        _num_leaves: i32,
        _max_depth: i32,
    ) -> Result<Option<(lgbm_model::Tree, lgbm_dataset::LeafPartitionLayout)>, ComputeError> {
        Ok(None)
    }

    /// The CONFIG-BOUND on-device grow seam — grow the whole
    /// tree under the caller's REAL [`crate::gain::GainConfig`] instead of the permissive
    /// proving-slice config the parameterless [`Self::grow_tree_on_device`] pins.
    ///
    /// This is the ADDITIVE fix for a prior gap where the production seam
    /// (`SerialTreeLearner` → `grow_tree_on_device` → `grow_tree_on_device_driver` →
    /// [`kernels::grow_driver::proving_slice_config`]) grew every on-device tree with
    /// `min_data_in_leaf = 1` / `min_sum_hessian_in_leaf = 0.0`, so C++'s admissibility gate
    /// was effectively OFF and a near-empty prefix could win with an `inf`-magnitude gain.
    /// The learner now calls THIS method with `self.cfg`, so LightGBM's real defaults
    /// (20 / 1e-3) — and every other `GainConfig` field (lambdas, `min_gain_to_split`,
    /// categorical scalars) — bind on every resident/anchor scan.
    ///
    /// `GainConfig` is an `lgbm-compute`-local type, so no `lgbm-treelearner` type crosses
    /// the crate wall (the on-device-driver crate-cycle constraint holds). The old
    /// parameterless seam is retained UNCHANGED for the existing STRUCTURE/anchor gates that
    /// intentionally pin the proving slice.
    ///
    /// Default is `Ok(None)` (the merge-gate no-op, exactly like [`Self::grow_tree_on_device`]);
    /// the `CpuBackend` / `GpuBackend` impls check [`cuda_on_device_enabled`] and delegate to
    /// [`kernels::grow_driver::grow_tree_on_device_driver_with_cfg`]. With `LGBM_CUDA_ON_DEVICE`
    /// unset every backend is byte-unchanged (SC-4).
    ///
    /// # Errors
    /// Propagates [`ComputeError`] from the sequenced kernels (bad histogram length,
    /// out-of-range bin, device launch) or the driver's guards (empty feature set,
    /// non-positive `num_leaves`, unsupported non-zero `max_delta_step` / `path_smooth`).
    fn grow_tree_on_device_with_cfg(
        &self,
        _gradients: &[f32],
        _hessians: &[f32],
        _features: &[GrowFeature],
        _num_leaves: i32,
        _max_depth: i32,
        _cfg: crate::gain::GainConfig,
    ) -> Result<Option<(lgbm_model::Tree, lgbm_dataset::LeafPartitionLayout)>, ComputeError> {
        Ok(None)
    }
}

/// The default cpu-runtime backend (the deterministic anchor).
///
/// Binds [`runtime::ActiveRuntime`] (cubecl-cpu under the default `cpu` feature)
/// and dispatches [`construct_histograms`](Backend::construct_histograms) to the
/// single-owner ordered f64 fold in [`kernels::histogram`].
/// THE production seam gate for the on-device histogram/tree path.
///
/// `LGBM_CUDA_ON_DEVICE` is a TRI-STATE toggle, read ONCE (OnceLock-cached,
/// mirroring [`split_2lane_enabled`]):
/// - `"1"` ⇒ force ON,
/// - `"0"` ⇒ force OFF (the explicit off-switch fallback),
/// - unset / empty / any other value ⇒ follow the device default
///   ([`on_device_default`], currently `false`).
///
/// While the resolver is `false` the on-device histogram entry
/// (`construct_histogram_for_leaf` (removed)) and the tree-growth
/// driver that will call it stay UNREACHABLE, so the CPU / ROCm / host-CUDA paths are
/// byte-unchanged. This is the call-site gate the future growth loop checks, exactly as
/// it ANDs in [`Backend::on_device_growth_supported`] — which INDEPENDENTLY stays `false`
/// for now (the build→fix→subtract histogram path exists in isolation; the growth loop
/// that consumes it is separate). The entry fn is additive and pure:
/// it never mutates global state and is invoked only behind this gate (in production) or
/// directly by the anchor tests.
#[must_use]
pub fn cuda_on_device_enabled() -> bool {
    cuda_on_device_override().unwrap_or_else(on_device_default)
}

/// Pure tri-state MAPPING for `LGBM_CUDA_ON_DEVICE` (V5 exact-match closed enum).
///
/// This is the testable core of [`cuda_on_device_override`]: it does NOT touch the
/// process env or the OnceLock cache, so unit tests can exercise every branch without
/// fighting the read-once semantics. Exact-string match only — no eval,
/// no path/format interpretation of the value (ASVS V5):
/// - `Some("1")` ⇒ `Some(true)` (force on),
/// - `Some("0")` ⇒ `Some(false)` (force off),
/// - `None` / `Some("")` / any other string ⇒ `None` (follow the device default).
#[doc(hidden)]
#[must_use]
pub fn cuda_on_device_override_from(s: Option<&str>) -> Option<bool> {
    match s {
        Some("1") => Some(true),
        Some("0") => Some(false),
        _ => None,
    }
}

/// The OnceLock-cached tri-state override read from `LGBM_CUDA_ON_DEVICE`.
///
/// Read ONCE per process via [`cuda_on_device_override_from`]. Returns
/// `Some(true)`/`Some(false)` when the operator forces the toggle, `None` when the
/// env is unset/empty/malformed (defer to [`on_device_default`]). NOT
/// `#[cfg(feature="cpu")]`-gated (unlike [`split_2lane_enabled`]) because cuda/rocm
/// builds resolve it too.
fn cuda_on_device_override() -> Option<bool> {
    use std::sync::OnceLock;
    static E: OnceLock<Option<bool>> = OnceLock::new();
    *E.get_or_init(|| cuda_on_device_override_from(std::env::var("LGBM_CUDA_ON_DEVICE").ok().as_deref()))
}

/// The compile-time device default for on-device growth when the env is unset.
///
/// STAYS `false` — on-device growth is OPT-IN via `LGBM_CUDA_ON_DEVICE="1"`, so
/// cpu / rocm / cuda builds are ALL byte-unchanged vs today.
///
/// On-device growth is ON by default. Known caveats as of the last real-CUDA
/// measurement: even after removing host-side control-plane overhead, a per-leaf
/// device-sync/readback floor (thousands of blocking readbacks per grow) was found to
/// be the dominant residual cost on real discrete NVIDIA hardware (slower than the host
/// path in every measured shape), and real-CUDA numerical parity against the f64 anchor
/// had not been re-resolved either. Those issues have not been independently re-verified
/// as fixed here — this default was flipped deliberately despite them.
/// `LGBM_CUDA_ON_DEVICE="0"` remains the explicit off-switch back to the host path, and
/// `="1"` is a no-op override of this same default.
///
/// Caveat for any future change here: `cfg!(feature = "cuda")` is evaluated in
/// THIS crate's feature set — a mono-feature (`cuda`-only) build resolves it `true`,
/// a dual-feature (`cpu`+`cuda`) build resolves it `true` as well, so gating on
/// `cfg!(feature = "cuda")` instead must keep the `LGBM_CUDA_ON_DEVICE="0"` off-switch as
/// the escape hatch.
#[must_use]
fn on_device_default() -> bool {
    true
}

/// Opt-in gate for the additive 2-lane native split scan. Read
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

    // GATED on-device-growth discriminator. Returns
    // [`cuda_on_device_enabled`] — `false` when `LGBM_CUDA_ON_DEVICE` is unset, so
    // the learner's eligibility AND-gate is dead and the byte-unchanged host/per-leaf
    // cpu-f64 anchor path runs (the hard merge gate). The gated flip lands on
    // CpuBackend (not only GpuBackend) so the structural gate grows the
    // on-device tree on the cubecl-cpu runtime, INSIDE the default merge gate. The
    // `grow_tree_on_device` body still returns `Ok(None)` for now (trait default),
    // so even with the env SET the fork falls through to the byte-identical host path.
    fn on_device_growth_supported(&self) -> bool {
        cuda_on_device_enabled()
    }

    // The ACTIVATED on-device grow seam. When the discriminator is
    // live ([`cuda_on_device_enabled`]) grow the ENTIRE continuous-feature + L2 tree
    // on the cubecl-cpu runtime via the per-leaf best-first driver
    // ([`kernels::grow_driver::grow_tree_on_device_driver`]) and return
    // `Ok(Some((Tree, LeafPartitionLayout)))`. With the env unset the discriminator is
    // `false`, so the learner's eligibility AND-gate never reaches here and the seam is
    // a byte-unchanged `Ok(None)` (the merge-gate contract). The on-device tree is
    // anchored STRUCTURE-bit-exact to the cpu f64 fold (the SAME cubecl-cpu runtime) —
    // never a second GPU f32 path.
    fn grow_tree_on_device(
        &self,
        gradients: &[f32],
        hessians: &[f32],
        features: &[GrowFeature],
        num_leaves: i32,
        max_depth: i32,
    ) -> Result<Option<(lgbm_model::Tree, lgbm_dataset::LeafPartitionLayout)>, ComputeError> {
        if !cuda_on_device_enabled() {
            return Ok(None);
        }
        let client = crate::runtime::cpu_client();
        let (tree, layout) = kernels::grow_driver::grow_tree_on_device_driver(
            self, &client, gradients, hessians, features, num_leaves, max_depth,
        )?;
        Ok(Some((tree, layout)))
    }

    // The CONFIG-BOUND on-device grow seam — delegate to the driver's
    // `_with_cfg` entry so the caller's REAL `GainConfig` (min_data_in_leaf /
    // min_sum_hessian_in_leaf / lambdas / min_gain_to_split / categorical scalars) binds on
    // every scan, closing a prior admissibility hole where those bounds were not enforced.
    // Env-gated + `Ok(None)` exactly like the parameterless seam above, so behavior stays
    // byte-unchanged with the env unset. The cpu-lane driver takes the f64 ANCHOR arm
    // (`resident_pool_supported() == false`).
    fn grow_tree_on_device_with_cfg(
        &self,
        gradients: &[f32],
        hessians: &[f32],
        features: &[GrowFeature],
        num_leaves: i32,
        max_depth: i32,
        cfg: crate::gain::GainConfig,
    ) -> Result<Option<(lgbm_model::Tree, lgbm_dataset::LeafPartitionLayout)>, ComputeError> {
        if !cuda_on_device_enabled() {
            return Ok(None);
        }
        let client = crate::runtime::cpu_client();
        let (tree, layout) = kernels::grow_driver::grow_tree_on_device_driver_with_cfg(
            self, &client, gradients, hessians, features, num_leaves, max_depth, cfg,
        )?;
        Ok(Some((tree, layout)))
    }

    fn construct_histograms(
        &self,
        _client: &ComputeClient<Self::Runtime>,
        binned: &[u32],
        ordered_gradients: &[f32],
        ordered_hessians: &[f32],
        num_bin: u32,
    ) -> Result<Vec<f64>, ComputeError> {
        // Native f64 fold — bit-identical to the single-unit `construct_hist_
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
        // Native f64 scan — bit-identical to the single-unit find_best_split_
        // kernel, without the per-(feature,leaf) cubecl launch. The cubecl path
        // stays in find_best_split_cpu for kernel-parity / ROCm-mirror tests.
        //
        // An OPT-IN additive 2-lane variant runs the REVERSE and
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
        // Native u32 routing + stable gather — bit-identical to the kernel path.
        kernels::partition::data_partition_cpu_native(
            bins,
            num_bin,
            min_bin,
            max_bin,
            threshold,
            most_freq_bin,
        )
    }

    // CpuBackend is the bit-exact host anchor: route a leaf's rows on the HOST via
    // the fused u8-route path (DataPartition::split) instead of widening
    // each leaf's bins to `&[u32]` and calling `data_partition`. Byte-identical
    // [left | right] order. RocmBackend inherits the default false (on-device).
    fn prefers_host_partition(&self) -> bool {
        true
    }

    // CpuBackend is the only backend that overrides the host unified fused paths
    // `build_fix_scan` / `subtract_scan` (the CPU-only f64 host analogs). The learner's
    // `smaller_unified`/`larger_unified` gates AND this in so a GPU backend without a
    // resident pool (CudaBackend/WgpuBackend) never routes into the erroring defaults.
    fn host_unified_fused_supported(&self) -> bool {
        true
    }

    fn subtract_histograms(
        &self,
        _client: &ComputeClient<Self::Runtime>,
        parent: &[f64],
        child: &[f64],
    ) -> Result<Vec<f64>, ComputeError> {
        // Native element-wise parent − child — bit-identical to the kernel path.
        kernels::subtract::subtract_histograms_cpu_native(parent, child)
    }

    // CPU batched split: keep the NATIVE per-feature path (the
    // `Backend::find_best_splits_batched` trait default, which calls
    // `self.find_best_split` == `find_best_split_cpu_native` per feature) rather than
    // routing CpuBackend through `find_best_splits_batched_fused_f64_on` (the same
    // fused cubecl kernel the GPU uses). Measurement found the fused cubecl-cpu path
    // materially slower — the cubecl-cpu per-leaf launch dispatch dominates even when
    // batched into ONE launch per leaf, regardless of leaf size (the launch fixed
    // cost, not the arithmetic, is the bottleneck). Shipping a silent CPU slowdown is
    // not acceptable, so the CpuBackend override is intentionally NOT defined here —
    // the native trait default applies. The GPU `RocmBackend` KEEPS the fused override
    // (one launch per leaf on gfx1100, f64 bit-exact), and the shared
    // `split_scan_body` helper (one source of the split math) stays for
    // BOTH paths regardless.
    //
    // The fused launcher `find_best_splits_batched_fused_f64_on` is generic over R,
    // so it remains available for the cubecl-cpu runtime via the oracle three-way
    // bit-exact gate (`kernel_parity_fused_equals_per_feature_and_native`) — the
    // merge is PROVEN bit-exact on cpu even though it is not the production path.

    // Override the per-leaf SPLIT SCAN to
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
    // path). Empty feats ⇒ empty Vec, no work.
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

    // UNIFIED host build+fix+scan — delegate to the inherent impl
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

    // UNIFIED host subtract+scan — delegate to the inherent impl
    // (the host f64 analog for the use_subtract larger child). CpuBackend ONLY;
    // RocmBackend keeps the trait-default typed error (its larger child is the
    // resident `subtract_resident` path).
    #[allow(clippy::too_many_arguments)]
    fn subtract_scan(
        &self,
        client: &ComputeClient<Self::Runtime>,
        parent: &[f64],
        smaller: &[f64],
        larger_buf: &mut [f64],
        all_feats: &[BatchedSplitFeature],
        scan_active: &[bool],
        cfg: &GainConfig,
        sum_gradient: f64,
        sum_hessian: f64,
        num_data: i32,
    ) -> Result<Vec<Option<SplitInfo>>, ComputeError> {
        self.subtract_scan_impl(
            client,
            parent,
            smaller,
            larger_buf,
            all_feats,
            scan_active,
            cfg,
            sum_gradient,
            sum_hessian,
            num_data,
        )
    }
}

/// The host f64 analog of the GPU `build_fix_scan_resident` fusion.
///
/// The obvious "reuse `ord_g`/`ord_h`/`ranges`/`out` scratch across leaves" lever was
/// profiled and RULED OUT — the per-leaf allocation bucket is a small fraction of this
/// path's fixed cost; the cost is overwhelmingly the rayon `par_iter` region (fork/join
/// floor + the actual fold/fix/scan work), which is irreducible at the gate-decision
/// granularity. So the per-leaf allocations are LEFT AS-IS (cheap, served warm from a
/// hot arena every leaf); adding learner-threaded or thread-local scratch reuse would
/// trade a stale-state tampering surface for a marginal gain.
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
        //    feature — only the bin column differs; mirrors `build_leaf_histograms_raw`).
        //    Values + order unchanged ⇒ bit-exact fold inputs.
        //    ALLOC + GATHER buckets timed under the inert fusion_prof gate.
        let r = leaf_rows.len();
        let (mut ord_g, mut ord_h) = fusion_prof::time(&fusion_prof::BFS_ALLOC_NS, || {
            (Vec::<f32>::with_capacity(r), Vec::<f32>::with_capacity(r))
        });
        fusion_prof::time(&fusion_prof::BFS_GATHER_NS, || {
            for &row in leaf_rows {
                ord_g.push(gradients[row as usize]);
                ord_h.push(hessians[row as usize]);
            }
        });
        // The f64-pregather micro-lever was A/B'd here and is NULL — a cold-isolated-
        // microbench win did not survive to the warm end-to-end train-wall (the
        // cold-overstates-warm rule). Not shipped (the f32 gather + per-feature widen
        // stays).

        // 2) Hoisted ascending-order validation ⇒ deterministic lowest-index error
        //    (matches the two-step / serial path's first-failure behavior). Records each
        //    feature's validated `[start, end)` so the parallel map is infallible.
        // The `ranges` Vec alloc is part of the small allocation bucket discussed above.
        let mut ranges: Vec<(usize, usize)> =
            fusion_prof::time(&fusion_prof::BFS_ALLOC_NS, || Vec::with_capacity(all_feats.len()));
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
        // The par region (fork/join floor + the actual fold/fix/scan
        // work) is the candidate-IRREDUCIBLE bucket — timed as one unit under the gate.
        let results: Vec<Result<(Vec<f64>, Option<SplitInfo>), ComputeError>> =
            fusion_prof::time(&fusion_prof::BFS_PAR_NS, || {
        all_feats
            .par_iter()
            .zip(scan_active.par_iter())
            .zip(feature_bins.par_iter())
            .map(|((f, &active), bins)| {
                let cells = 2 * f.num_bin as usize;
                // (a) BUILD: own private histogram, ascending leaf_rows, grad at bin<<1.
                // BUILD timed per-feature (the fold body). time() is thread-safe (relaxed
                // fetch_add) so summing across rayon tasks is correct; inert when the gate
                // is off (parity untouched).
                let mut hist = vec![0.0f64; cells];
                fusion_prof::time(&fusion_prof::BFS_BUILD_NS, || {
                    fold_one_feature(bins, leaf_rows, &ord_g, &ord_h, &mut hist);
                });
                // (b) FIX: most_freq_bin reconstruct on RAW leaf sums. No-op
                //     for most_freq_bin==0 (the C++ `if (most_freq_bin > 0)` guard).
                // (c) COMPACT: shift real-bin `c+offset` into `c`, zero the tail.
                //     No-op for offset==0.
                // FIX+COMPACT timed as one sub-bucket (expected ~0 — the
                // no-op guards dominate in the common case).
                fusion_prof::time(&fusion_prof::BFS_FIXCOMPACT_NS, || {
                    fix_histogram_inline(&mut hist, f.most_freq_bin, sum_gradient, sum_hessian);
                    compact_histogram_inline(&mut hist, f.offset);
                });
                // (d) SCAN (only if spine-active): own SplitInfo from the disjoint hist.
                // SCAN timed per-feature (find_best_split).
                let split = if active {
                    Some(fusion_prof::time(&fusion_prof::BFS_SCAN_NS, || {
                        self.find_best_split(
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
                        )
                    })?)
                } else {
                    None
                };
                Ok((hist, split))
            })
            .collect()
            });

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

    /// UNIFIED host subtract+scan for the subtract-derived LARGER
    /// child — the host f64 analog of [`build_fix_scan_impl`] but for the use_subtract
    /// child, fusing per-feature `{subtract → scan}` into ONE rayon region. There is NO
    /// build step (the larger child's histogram is `parent − smaller`, already
    /// fixed+compacted in both operands) and NO fix step (non-negotiable #3: C++ runs no
    /// FixHistogram on the use_subtract larger child — only ComputeBestSplit).
    ///
    /// BIT-EXACT SAFETY: each task computes a PRIVATE region `parent[start..end] −
    /// smaller[start..end]` (the exact cell-wise op of [`subtract_histograms`] over the
    /// disjoint range) into its own buffer, then scans it; `par_iter` preserves input
    /// order so the serial assembly stays feature-index-ordered. Same f64 cells, same op,
    /// same order ⇒ byte-identical to the two-step `subtract_histograms` + per-feature
    /// scan, so the bit-exact CPU f64 anchor holds (proven FORCED-ON in the parity gate).
    ///
    /// DETERMINISTIC ERROR ORDER: the per-feature `[slot_off, slot_off + 2*num_bin)`
    /// range validation is hoisted BEFORE the parallel map and walks features in
    /// ascending index, validated against `parent.len()`, `smaller.len()`, AND
    /// `larger_buf.len()` — so a parallel error race can never change WHICH error
    /// surfaces (the lowest-index offender, matching `subtract_histograms`'
    /// whole-buffer length check). The map itself is then infallible on the validated
    /// slices.
    ///
    /// Scope: CpuBackend ONLY, called behind the host `larger_unified` gate; the
    /// resident/GPU larger child keeps `subtract_resident`. RocmBackend is untouched.
    #[allow(clippy::too_many_arguments)]
    fn subtract_scan_impl(
        &self,
        client: &ComputeClient<<Self as Backend>::Runtime>,
        parent: &[f64],
        smaller: &[f64],
        larger_buf: &mut [f64],
        all_feats: &[BatchedSplitFeature],
        scan_active: &[bool],
        cfg: &GainConfig,
        sum_gradient: f64,
        sum_hessian: f64,
        num_data: i32,
    ) -> Result<Vec<Option<SplitInfo>>, ComputeError> {
        use rayon::prelude::*;

        debug_assert_eq!(scan_active.len(), all_feats.len());

        // 1) Hoisted ascending-order validation ⇒ deterministic lowest-index error
        //    (matches the two-step `subtract_histograms`' whole-buffer length check).
        //    Each feature's `[start, end)` is validated against parent/smaller/larger so
        //    the parallel map is infallible on the validated slices.
        // The `ranges` alloc is part of the small allocation bucket for this child.
        let mut ranges: Vec<(usize, usize)> =
            fusion_prof::time(&fusion_prof::SUB_ALLOC_NS, || Vec::with_capacity(all_feats.len()));
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
                    detail: "subtract_scan: slot_off + region overflows".to_string(),
                })?;
            if end > larger_buf.len() {
                return Err(ComputeError::LengthMismatch {
                    expected: end,
                    actual: larger_buf.len(),
                });
            }
            if end > parent.len() {
                return Err(ComputeError::LengthMismatch {
                    expected: end,
                    actual: parent.len(),
                });
            }
            if end > smaller.len() {
                return Err(ComputeError::LengthMismatch {
                    expected: end,
                    actual: smaller.len(),
                });
            }
            ranges.push((f.slot_off, end));
        }

        // 2) ONE rayon fork/join over features. Each task: subtract → scan into its OWN
        //    private buffer (cache-hot, no cross-region hand-off). Returns
        //    `(private_region, Option<SplitInfo>)`; `par_iter` preserves input order so
        //    the serial assembly below stays feature-index-ordered (bit-exact argmax).
        // The par region (fork/join floor + subtract+scan work) timed as one.
        let results: Vec<Result<(Vec<f64>, Option<SplitInfo>), ComputeError>> =
            fusion_prof::time(&fusion_prof::SUB_PAR_NS, || {
        all_feats
            .par_iter()
            .zip(scan_active.par_iter())
            .zip(ranges.par_iter())
            .map(|((f, &active), &(start, end))| {
                // (a) SUBTRACT: own private region = parent − smaller over the disjoint
                //     range (the same cell-wise op as `subtract_histograms`). NO fix, NO
                //     compact — both operands are already fixed+compacted.
                let region: Vec<f64> = parent[start..end]
                    .iter()
                    .zip(&smaller[start..end])
                    .map(|(p, s)| p - s)
                    .collect();
                // (b) SCAN (only if spine-active): own SplitInfo from the derived region.
                let split = if active {
                    Some(self.find_best_split(
                        client,
                        &region,
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
                Ok((region, split))
            })
            .collect()
            });

        // 3) SERIAL ordered assembly: copy each private region into its disjoint
        //    `larger_buf` region (COMPLETE histogram for the larger child) and collect
        //    the per-feature SplitInfos in feature-index order. Propagate the lowest-index
        //    scan error (the validated ranges already guarantee region-length safety).
        let mut splits: Vec<Option<SplitInfo>> = Vec::with_capacity(all_feats.len());
        for (fpos, res) in results.into_iter().enumerate() {
            let (region, split) = res?;
            let (start, end) = ranges[fpos];
            larger_buf[start..end].copy_from_slice(&region);
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

/// Element width of the device-resident bin buffer. The buffer is
/// uploaded at the NARROWEST uniform width covering every feature's `BinColumn` variant
/// (widest variant present), so the resident-reading kernels dispatch the matching
/// `<B: Int>` monomorphization. Mirrors the host `BinColumn` u8/u16/u32 axis.
#[cfg(feature = "gpu")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResidentBinWidth {
    U8,
    U16,
    U32,
}

/// The narrowest uniform width that holds every column = the WIDEST variant present
/// (any U32 ⇒ U32; else any U16 ⇒ U16; else U8). A uniform width is required because the
/// resident buffer is ONE concatenated `Array<B>`; narrower columns upcast into it.
#[cfg(feature = "gpu")]
pub fn resident_bin_width(cols: &[&BinColumn]) -> ResidentBinWidth {
    let mut w = ResidentBinWidth::U8;
    for c in cols {
        let cw = match c {
            BinColumn::U8(_) => ResidentBinWidth::U8,
            BinColumn::U16(_) => ResidentBinWidth::U16,
            BinColumn::U32(_) => ResidentBinWidth::U32,
        };
        w = match (w, cw) {
            (ResidentBinWidth::U32, _) | (_, ResidentBinWidth::U32) => ResidentBinWidth::U32,
            (ResidentBinWidth::U16, _) | (_, ResidentBinWidth::U16) => ResidentBinWidth::U16,
            _ => ResidentBinWidth::U8,
        };
    }
    w
}

/// The device-resident binned dataset cached inside [`RocmBackend`]:
/// the ONE concatenated feature-column bin buffer's device `Handle` (feature-major,
/// length `num_features * num_data`) + the dims to index it (`f * num_data + row`) +
/// the native element `width`. `Handle` is cheaply clonable (ref-counted).
#[cfg(feature = "gpu")]
#[derive(Debug, Clone)]
struct ResidentBins {
    /// Concatenated feature-major bin columns: feature `f`'s row `r` is at
    /// `f * num_data + r`. Uploaded ONCE per train, at native `width`.
    handle: cubecl::server::Handle,
    num_features: usize,
    num_data: usize,
    /// Element width of `handle`: the resident-reading kernels are
    /// generic `<B: Int>` and dispatched on this.
    width: ResidentBinWidth,
}

/// The device-resident gradients+hessians for the CURRENT tree,
/// uploaded ONCE per grow by
/// [`upload_resident_grad_hess`](Backend::upload_resident_grad_hess) — the grad/hess
/// analog of [`ResidentBins`]. Unlike the binned columns (constant for the whole
/// train), grad/hess are recomputed EVERY boosting iteration, so this is refreshed
/// once per TREE grow (constant across that grow) and the on-device resident build
/// gathers each leaf's grad/hess ON DEVICE from `grad`/`hess` via the leaf-row index —
/// eliminating the per-build host `ord_g`/`ord_h` gather + `create_from_slice` upload.
/// `Handle` is cheaply clonable (ref-counted).
#[cfg(feature = "gpu")]
#[derive(Debug, Clone)]
struct ResidentGradHess {
    /// Full-length `[num_data]` gradients (device), indexed on device by the leaf-row id.
    grad: cubecl::server::Handle,
    /// Full-length `[num_data]` hessians (device), indexed on device by the leaf-row id.
    hess: cubecl::server::Handle,
    /// Row count = the buffers' length = the resident-bin column stride.
    num_data: usize,
}


/// The generic GPU backend — dispatches every hot-path op to the
/// runtime-generic f64/f32 CubeCL kernels, carrying the on-device resident histogram
/// pool. Parameterized by the CubeCL [`Runtime`](cubecl::Runtime) `R` so ONE
/// implementation serves ROCm/HIP (`RocmBackend`), CUDA (`CudaBackend`), and WGPU
/// (`WgpuBackend`) — see the type aliases below. The ROCm GPU parity gate validates
/// this shared code on hardware; CUDA/WGPU inherit correctness by construction (same
/// code, different `R`). Previously this was a hand-written `RocmBackend` plus a
/// separate pool-less `gpu_core_backend!` macro for CudaBackend/WgpuBackend; the
/// FULL resident surface was later hoisted into this one generic so cuda/wgpu reach
/// speed parity.
///
/// The backend carries interior-mutable device state — a
/// `RefCell<Option<ResidentBins>>` cache of the binned feature columns uploaded ONCE
/// per train. The learner holds `&B` (shared ref) and the trait methods take
/// `&self`, so the cache MUST be behind interior mutability (RefCell), NOT a
/// `&mut self` signature change. The single-threaded train loop makes the RefCell
/// borrow safe. Because a `RefCell` is not `Copy`, this type does not derive
/// `Copy`; `CpuBackend` stays the stateless unit struct.
#[cfg(feature = "gpu")]
pub struct GpuBackend<R: cubecl::Runtime> {
    /// The device-resident binned dataset, populated ONCE per train by
    /// [`upload_resident_bins`](Backend::upload_resident_bins) and read by the
    /// per-leaf [`build_leaf_histograms_raw`](Backend::build_leaf_histograms_raw)
    /// override. `None` until the first upload (defensive fallback to the per-leaf
    /// host-gather path).
    resident_bins: std::cell::RefCell<Option<ResidentBins>>,
    /// Whether `resident_bins` is PINNED by a once-per-train owner (the
    /// learner's `resident_bins_uploaded` guard). Cleared by every fresh
    /// `upload_resident_bins`; set only via `pin_resident_bins`. Authorizes (together
    /// with a geometry match) the on-device driver's per-grow upload skip.
    resident_bins_pin: std::cell::Cell<bool>,
    /// The device-handle slot mirror, indexed by host `HistogramPool`
    /// slot id. `resident_pool[slot]` holds the fixed+compacted f64 histogram `Handle`
    /// for whichever leaf currently owns that slot, or `None` when the slot is empty.
    /// The learner issues build/subtract/move/scan ops here at the SAME call sites
    /// (with the SAME slot ids) it drives the host pool, so this mirror tracks the
    /// host pool's slot→leaf map exactly. Like `resident_bins`, the
    /// single-threaded train loop makes the RefCell borrow safe.
    resident_pool: std::cell::RefCell<Vec<Option<cubecl::server::Handle>>>,
    /// The device-resident grad/hess for the current tree grow,
    /// uploaded ONCE per grow by
    /// [`upload_resident_grad_hess`](Backend::upload_resident_grad_hess). `None` until
    /// the first upload (the on-device resident build then falls back to the host
    /// grad/hess gather, defensive). Cleared by
    /// [`reset_resident_pool`](Backend::reset_resident_pool) so a stale prior tree's
    /// grad/hess can never leak into the next grow. Interior-mutable for the same
    /// single-threaded-train reason as `resident_bins`/`resident_pool`.
    resident_grad_hess: std::cell::RefCell<Option<ResidentGradHess>>,
    /// Test-only toggle to FORCE the host path on RocmBackend (so the
    /// resident==host tree-equivalence test can grow the SAME f32-atomic-built tree
    /// through the host read-back/subtract/scan chain). `true` (the default) reports
    /// `resident_pool_supported() == true`; `false` forces the host path. Set only by
    /// the test-only [`with_resident`](GpuBackend::with_resident) constructor.
    resident_enabled: bool,
    /// Ties the backend to its CubeCL runtime `R` without storing one (the client is
    /// passed per-call). `fn() -> R` keeps the type `Send`/`Sync`/`Copy`-agnostic and
    /// imposes no `R: …` auto-trait bound on the struct.
    _runtime: std::marker::PhantomData<fn() -> R>,
}

// Hand-written `Debug` so the `R` type parameter is NOT required to be `Debug`.
#[cfg(feature = "gpu")]
impl<R: cubecl::Runtime> std::fmt::Debug for GpuBackend<R> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuBackend")
            .field("resident_enabled", &self.resident_enabled)
            .finish_non_exhaustive()
    }
}

#[cfg(feature = "gpu")]
impl<R: cubecl::Runtime> Default for GpuBackend<R> {
    fn default() -> Self {
        Self {
            resident_bins: std::cell::RefCell::new(None),
            resident_bins_pin: std::cell::Cell::new(false),
            resident_pool: std::cell::RefCell::new(Vec::new()),
            resident_grad_hess: std::cell::RefCell::new(None),
            // Production default: the device-resident pool is enabled.
            resident_enabled: true,
            _runtime: std::marker::PhantomData,
        }
    }
}

#[cfg(feature = "gpu")]
impl<R: cubecl::Runtime> GpuBackend<R> {
    /// TEST-ONLY constructor: build a backend that REPORTS
    /// `resident_pool_supported() == enabled`. The resident==host tree-equivalence
    /// test grows the SAME corpus twice on a `RocmBackend` — once with `with_resident(true)`
    /// (the resident chain) and once with `with_resident(false)` (forcing the host
    /// read-back/subtract/scan path) — and asserts the two trees match within ~1e-6.
    /// The same f32-atomic RAW build runs in both cases; only the build→fix→compact→
    /// subtract→scan ROUTING differs. Not used on the production path
    /// ([`Default`] enables residency).
    pub fn with_resident(enabled: bool) -> Self {
        Self {
            resident_bins: std::cell::RefCell::new(None),
            resident_bins_pin: std::cell::Cell::new(false),
            resident_pool: std::cell::RefCell::new(Vec::new()),
            resident_grad_hess: std::cell::RefCell::new(None),
            resident_enabled: enabled,
            _runtime: std::marker::PhantomData,
        }
    }
}

/// The ROCm/HIP GPU backend (opt-in `rocm` feature) — `GpuBackend` bound to the
/// cubecl-hip runtime running the f64 kernels on the local gfx-class GPU.
#[cfg(feature = "rocm")]
pub type RocmBackend = GpuBackend<runtime::RocmRuntime>;

/// The CUDA GPU backend (opt-in `cuda` feature) — `GpuBackend` bound to the cubecl-cuda
/// runtime (NVIDIA). Reaches ROCm-parity speed via the SAME resident histogram pool.
#[cfg(feature = "cuda")]
pub type CudaBackend = GpuBackend<runtime::CudaRuntime>;

/// The WGPU/WGSL GPU backend (opt-in `wgpu` feature) — `GpuBackend` bound to the
/// cubecl-wgpu runtime. NOTE (locked decision #3): the shared LDS histogram kernel
/// accumulates in f32 atomics, which WGSL lacks — a `--features wgpu` build MAY fail to
/// compile inside that kernel; that is the accepted, documented outcome (no fallback).
#[cfg(feature = "wgpu")]
pub type WgpuBackend = GpuBackend<runtime::WgpuRuntime>;

#[cfg(feature = "gpu")]
impl<R: cubecl::Runtime> Backend for GpuBackend<R> {
    type Runtime = R;

    // Route the rocm partition on the HOST via the shipped
    // fused path instead of the per-split device round-trip — faster at narrow widths,
    // wash at wide, parity within ~1e-6 (not a bit-exact swap). The device round-trip
    // is pure overhead on shared DDR5 (the build reads host indices_ either way). Default ON;
    // LGBM_ROCM_HOST_PARTITION=0 forces the old device round-trip for benching/rollback.
    fn prefers_host_partition(&self) -> bool {
        !matches!(std::env::var("LGBM_ROCM_HOST_PARTITION").as_deref(), Ok("0"))
    }

    // GATED on-device-growth discriminator, mirroring
    // CpuBackend. Returns [`cuda_on_device_enabled`] — `false` when
    // `LGBM_CUDA_ON_DEVICE` is unset, so the ROCm/CUDA/WGPU host path is
    // byte-unchanged. One generic GpuBackend<R> impl is shared by
    // ROCm/CUDA/WGPU; the env gate (not a per-runtime `true`) is what keeps the flip
    // safe until the driver body is wired in (which still returns `Ok(None)` here).
    fn on_device_growth_supported(&self) -> bool {
        cuda_on_device_enabled()
    }

    // The ACTIVATED on-device
    // grow seam on the GPU backend. When the discriminator is live
    // ([`cuda_on_device_enabled`]) grow the ENTIRE continuous-feature + L2 tree on the
    // GPU runtime `R` via the shared per-leaf best-first driver
    // ([`kernels::grow_driver::grow_tree_on_device_driver`]) and return `Ok(Some(..))`.
    // One generic `GpuBackend<R>` impl is shared by ROCm/CUDA/WGPU; the driver runs the
    // SAME runtime-generic f64 kernels the histogram/split/partition paths use. With the
    // env unset the discriminator is `false`, so the seam is a byte-unchanged `Ok(None)`.
    // The grown GPU tree is anchored STRUCTURE-bit-exact to the cpu f64 fold —
    // never a second GPU f32 path.
    fn grow_tree_on_device(
        &self,
        gradients: &[f32],
        hessians: &[f32],
        features: &[GrowFeature],
        num_leaves: i32,
        max_depth: i32,
    ) -> Result<Option<(lgbm_model::Tree, lgbm_dataset::LeafPartitionLayout)>, ComputeError> {
        if !cuda_on_device_enabled() {
            return Ok(None);
        }
        let client = R::client(&Default::default());
        let (tree, layout) = kernels::grow_driver::grow_tree_on_device_driver(
            self, &client, gradients, hessians, features, num_leaves, max_depth,
        )?;
        Ok(Some((tree, layout)))
    }

    // The CONFIG-BOUND on-device grow seam on the GPU backend — one generic
    // `GpuBackend<R>` impl shared by ROCm/CUDA/WGPU. Delegates to the driver's `_with_cfg`
    // entry so the learner's REAL `GainConfig` binds through the resident build/subtract/scan
    // path (the same admissibility-gate fix reaches the hip arm identically). Env-gated +
    // `Ok(None)` like the parameterless seam, so behavior stays byte-unchanged with
    // `LGBM_CUDA_ON_DEVICE` unset.
    fn grow_tree_on_device_with_cfg(
        &self,
        gradients: &[f32],
        hessians: &[f32],
        features: &[GrowFeature],
        num_leaves: i32,
        max_depth: i32,
        cfg: crate::gain::GainConfig,
    ) -> Result<Option<(lgbm_model::Tree, lgbm_dataset::LeafPartitionLayout)>, ComputeError> {
        if !cuda_on_device_enabled() {
            return Ok(None);
        }
        let client = R::client(&Default::default());
        let (tree, layout) = kernels::grow_driver::grow_tree_on_device_driver_with_cfg(
            self, &client, gradients, hessians, features, num_leaves, max_depth, cfg,
        )?;
        Ok(Some((tree, layout)))
    }

    fn construct_histograms(
        &self,
        client: &ComputeClient<Self::Runtime>,
        binned: &[u32],
        ordered_gradients: &[f32],
        ordered_hessians: &[f32],
        num_bin: u32,
    ) -> Result<Vec<f64>, ComputeError> {
        // The old non-resident f32 GPU build kernels (global-atomic /
        // LDS-privatized / plane-aggregated) were deleted. This required trait method now
        // drives the runtime-generic f64 on-device build (`construct_histograms_f64_on`,
        // the kernel the v2.0 on-device grow-driver's non-resident fallback also uses) —
        // it stays on-device and is bit-exact to the cpu f64 anchor. The non-resident GPU
        // route reaches it only when the resident pool is disabled; the v2.0 resident path
        // is unaffected (it uses the resident build methods).
        kernels::histogram::construct_histograms_f64_on(
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
        _client: &ComputeClient<Self::Runtime>,
        bins: &[u32],
        num_bin: u32,
        min_bin: u32,
        max_bin: u32,
        threshold: u32,
        most_freq_bin: u32,
    ) -> Result<(Vec<u32>, usize), ComputeError> {
        // The old u32 GPU partition launcher (`data_partition_on`) was
        // deleted. This required trait method now folds on the host (`data_partition_cpu_native`,
        // value-identical to the device route). The v2.0 on-device path uses the native-width
        // `data_partition_native` override, which is unaffected.
        kernels::partition::data_partition_cpu_native(
            bins,
            num_bin,
            min_bin,
            max_bin,
            threshold,
            most_freq_bin,
        )
    }

    /// GPU override: upload the leaf's bins at NATIVE
    /// width (u8/u16/u32) instead of u32-widening — 4× fewer host→device bytes + a
    /// narrow-reading route kernel on the common all-u8 (`max_bin≤255`) case. Bit-EXACT
    /// to the default widening path (`data_partition`): the u8/u16/u32 route kernels read
    /// the same bin value via `u32::cast_from`, so the route (and the stable gather) is
    /// value-identical.
    #[allow(clippy::too_many_arguments)]
    fn data_partition_native(
        &self,
        client: &ComputeClient<Self::Runtime>,
        bins: &BinColumn,
        num_bin: u32,
        min_bin: u32,
        max_bin: u32,
        threshold: u32,
        most_freq_bin: u32,
    ) -> Result<(Vec<u32>, usize), ComputeError> {
        kernels::partition::data_partition_native_on(
            client,
            bins,
            num_bin,
            min_bin,
            max_bin,
            threshold,
            most_freq_bin,
        )
    }

    /// GPU override: run the §9 `mark → prefix-sum → scatter` and write the
    /// child ranges into the resident [`DeviceLeafSplits`](kernels::partition::DeviceLeafSplits)
    /// slot ON DEVICE via [`partition_child_ranges_device`](kernels::partition::partition_child_ranges_device)
    /// — the split point never crosses back to the host. Bit-exact to the cpu f64 anchor;
    /// static single-owner geometry (never `CubeCount::Dynamic`).
    #[allow(clippy::too_many_arguments)]
    fn data_partition_resident_no_readback(
        &self,
        client: &ComputeClient<Self::Runtime>,
        bins: &BinColumn,
        data_indices: &[u32],
        num_bin: u32,
        min_bin: u32,
        max_bin: u32,
        default_bin: u32,
        most_freq_bin: u32,
        missing_type: u8,
        default_left: bool,
        threshold: u32,
        leaf_splits: &kernels::partition::DeviceLeafSplits<Self::Runtime>,
        leaf_id: usize,
        p_begin: i32,
        p_count: i32,
    ) -> Result<Vec<u32>, ComputeError> {
        kernels::partition::partition_child_ranges_device(
            client,
            bins,
            data_indices,
            num_bin,
            min_bin,
            max_bin,
            default_bin,
            most_freq_bin,
            missing_type,
            default_left,
            threshold,
            leaf_splits,
            leaf_id,
            p_begin,
            p_count,
        )
    }

    fn subtract_histograms(
        &self,
        _client: &ComputeClient<Self::Runtime>,
        parent: &[f64],
        child: &[f64],
    ) -> Result<Vec<f64>, ComputeError> {
        // The old buffer-based f32/f64 GPU subtract launchers were
        // deleted. This required trait method now folds on the host
        // (`subtract_histograms_cpu_native`). The v2.0 on-device path uses the resident
        // handle-based `subtract_resident` override, which is unaffected.
        kernels::subtract::subtract_histograms_cpu_native(parent, child)
    }

    /// The GPU path's resident u32 upload DOES consume the widened columns:
    /// the learner widens each [`BinColumn`] to u32 once and calls
    /// [`upload_resident_bins`](Backend::upload_resident_bins) so the resident buffer
    /// is byte-identical to HEAD.
    fn wants_resident_bins(&self) -> bool {
        true
    }

    /// GPU override: upload the binned feature columns to the device
    /// ONCE per train and cache the device `Handle` in `self.resident_bins` (interior
    /// mutability). The columns are concatenated feature-major into ONE buffer
    /// (`f * num_data + row`) so a single resident `Handle` covers every feature.
    /// Per-leaf [`build_leaf_histograms_raw`](Backend::build_leaf_histograms_raw)
    /// then gathers leaf rows ON DEVICE from this buffer, eliminating the per-leaf
    /// `[num_features × rows]` host bin re-upload.
    fn upload_resident_bins(
        &self,
        client: &ComputeClient<Self::Runtime>,
        feature_bins: &[&BinColumn],
    ) {
        // A fresh upload dissolves any prior once-per-train pin — the new
        // owner must re-pin (`pin_resident_bins`), so an un-pinned per-grow uploader
        // (a direct driver caller) can never inherit a stale skip authorization.
        self.resident_bins_pin.set(false);
        let num_features = feature_bins.len();
        if num_features == 0 {
            *self.resident_bins.borrow_mut() = None;
            return;
        }
        let num_data = feature_bins[0].len();
        // Upload at the NARROWEST uniform width covering every column,
        // not always u32 — cuts the host concat + host→device transfer ~4× on all-u8
        // data (bins ≤256) and drops the learner's `to_u32_vec` widen. Narrower columns
        // upcast into the uniform buffer; the value is a bin INDEX, byte-faithful across
        // widths (the resident kernels read the matching `<B: Int>` monomorphization).
        // `as_bytes` is `CubeElement::as_bytes` (the same call the histogram launchers use).
        use cubecl::prelude::CubeElement;
        let width = resident_bin_width(feature_bins);
        let handle = match width {
            ResidentBinWidth::U8 => {
                let mut concat: Vec<u8> = Vec::with_capacity(num_features * num_data);
                for &col in feature_bins {
                    match col {
                        BinColumn::U8(v) => concat.extend_from_slice(v),
                        // Unreachable for a U8-uniform set, but keep total: upcast is lossless.
                        BinColumn::U16(v) => concat.extend(v.iter().map(|&b| b as u8)),
                        BinColumn::U32(v) => concat.extend(v.iter().map(|&b| b as u8)),
                    }
                }
                client.create_from_slice(u8::as_bytes(&concat))
            }
            ResidentBinWidth::U16 => {
                let mut concat: Vec<u16> = Vec::with_capacity(num_features * num_data);
                for &col in feature_bins {
                    match col {
                        BinColumn::U8(v) => concat.extend(v.iter().map(|&b| u16::from(b))),
                        BinColumn::U16(v) => concat.extend_from_slice(v),
                        BinColumn::U32(v) => concat.extend(v.iter().map(|&b| b as u16)),
                    }
                }
                client.create_from_slice(u16::as_bytes(&concat))
            }
            ResidentBinWidth::U32 => {
                let mut concat: Vec<u32> = Vec::with_capacity(num_features * num_data);
                for &col in feature_bins {
                    match col {
                        BinColumn::U8(v) => concat.extend(v.iter().map(|&b| u32::from(b))),
                        BinColumn::U16(v) => concat.extend(v.iter().map(|&b| u32::from(b))),
                        BinColumn::U32(v) => concat.extend_from_slice(v),
                    }
                }
                client.create_from_slice(u32::as_bytes(&concat))
            }
        };
        *self.resident_bins.borrow_mut() = Some(ResidentBins {
            handle,
            num_features,
            num_data,
            width,
        });
    }

    /// Mark the just-uploaded resident bins as owned by a once-per-train
    /// guard (see the trait doc). Set ONLY by the learner after its guarded upload.
    fn pin_resident_bins(&self) {
        self.resident_bins_pin.set(true);
    }

    /// Skip authorization for the on-device driver's per-grow upload —
    /// requires BOTH the pin (a live once-per-train owner) AND a geometry match.
    fn resident_bins_pinned(&self, num_features: usize, num_data: usize) -> bool {
        self.resident_bins_pin.get()
            && self
                .resident_bins
                .borrow()
                .as_ref()
                .is_some_and(|c| c.num_features == num_features && c.num_data == num_data)
    }

    /// GPU override: upload the current tree's grad/hess to the
    /// device ONCE per grow and cache the two device `Handle`s in `self.resident_grad_hess`
    /// (interior mutability). The on-device resident build
    /// ([`build_resident_leaf`](Backend::build_resident_leaf)) then gathers each leaf's
    /// grad/hess ON DEVICE from these buffers via the leaf-row index — the per-build host
    /// `ord_g`/`ord_h` gather + `create_from_slice` upload is gone. Grad/hess stay `f32`
    /// (the u64 fixed-point build kernel quantizes them in-kernel), so this is byte-faithful
    /// to what the per-build path uploaded, only ONCE per grow instead of once per build.
    fn upload_resident_grad_hess(
        &self,
        client: &ComputeClient<Self::Runtime>,
        gradients: &[f32],
        hessians: &[f32],
    ) {
        use cubecl::prelude::CubeElement;
        if gradients.is_empty() {
            *self.resident_grad_hess.borrow_mut() = None;
            return;
        }
        let grad = client.create_from_slice(f32::as_bytes(gradients));
        let hess = client.create_from_slice(f32::as_bytes(hessians));
        *self.resident_grad_hess.borrow_mut() = Some(ResidentGradHess {
            grad,
            hess,
            num_data: gradients.len(),
        });
    }

    // The old non-resident GPU `build_leaf_histograms_raw` and
    // `find_best_splits_batched` overrides (which drove the deleted f32 batched/resident
    // build launcher and the batched-fused split launcher on the OLD per-leaf GPU route)
    // are REMOVED. `GpuBackend<R>` now inherits the `Backend` trait defaults for both
    // (host gather + per-feature loop over the repointed `construct_histograms` /
    // `find_best_split`). The v2.0 on-device grow-driver does NOT use these — it uses the
    // resident build/scan overrides below — so its path is unaffected.

    // ---- device-resident histogram-pool overrides ----

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
        // Drop the prior tree's resident grad/hess so a stale grow's
        // buffers can never leak into the next tree (grad/hess are per-grow, not per-train).
        // The driver re-uploads via `upload_resident_grad_hess` before the root build.
        *self.resident_grad_hess.borrow_mut() = None;
    }

    /// Build ONE leaf's histogram device-resident (build → dequant → fix → compact) via
    /// the oib `build_fix_compact_resident_f64_on` chain and STORE the returned
    /// `(Handle, len)` into mirror slot `slot` (dropping any prior Handle). Falls back
    /// to a typed error if the resident bin cache is empty (defensive — the learner
    /// always uploads before the growth loop when eligible).
    ///
    /// The RAW histogram BUILD accumulates grad/hess as
    /// u64 TWO'S-COMPLEMENT FIXED-POINT (scale S = 2^30) via integer LDS atomics, NOT
    /// f32 atomics. On RDNA the f32 `atomicAdd` lowers to a CAS retry loop that saturates
    /// under contention; the integer `ds_add_u64` is a native single-instruction op,
    /// giving a faster build (~1.3–1.7× in the wide large-leaf regime), a DETERMINISTIC
    /// GPU histogram, AND ~3600× better accuracy (5.9e-9 vs f32 2.2e-5 rel-to-anchor;
    /// exact in the cancelling regime). The `fix_compact_kernel` widen pass dequantizes
    /// `(bits as i64)/2^30 → f64`; everything downstream (FixHistogram fold, compact,
    /// subtract trick, scan) stays f64 and is UNCHANGED.
    ///
    /// Like `construct_histograms`, this is the GPU's ~1e-6 best-effort contract, NOT a
    /// bit-exact one — the fixed-point accumulation order/quantization differs from the
    /// host f64 fold (though it is MORE accurate than the prior f32-atomic path it
    /// replaced). The CpuBackend f64 fold remains the bit-exact hard merge gate and is
    /// unaffected. An overflow guard (`build_fix_compact_resident_f64_on`) enforces the
    /// i64@2^30 bound (~1e9 rows × |g| ≤ 8) at the resident-build boundary.
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
        // When the once-per-grow resident grad/hess are cached, gather
        // grad/hess ON DEVICE from them via the leaf-row index — no host `ord_g`/`ord_h`
        // gather + per-build upload. `None` (grad/hess never uploaded) keeps the byte-identical
        // host-gather build (defensive; the on-device driver always uploads before the root).
        let resident_gh = self
            .resident_grad_hess
            .borrow()
            .as_ref()
            .map(|gh| (gh.grad.clone(), gh.hess.clone(), gh.num_data));
        let (handle, len) = kernels::histogram::build_fix_compact_resident_f64_on(
            client,
            resident.handle.clone(),
            resident.width,
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
            resident_gh,
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

    /// CO-PACKED 2-slot resident scan — borrows BOTH the
    /// smaller and larger sibling Handles (simultaneously resident at the co-pack
    /// point: the smaller slot survives `subtract_resident`, the larger is its
    /// derived output) and runs ONE co-packed 2-slot launch + ONE readback, returning
    /// `(smaller_splits, larger_splits)`. Errors if either slot is empty (defensive,
    /// mirroring `scan_resident_leaf` / the two-Handle `subtract_resident` borrow).
    #[allow(clippy::too_many_arguments)]
    fn scan_resident_siblings(
        &self,
        client: &ComputeClient<Self::Runtime>,
        smaller_slot: usize,
        larger_slot: usize,
        slot_len: usize,
        feats: &[BatchedSplitFeature],
        cfg: &GainConfig,
        smaller_totals: (f64, f64, i32),
        larger_totals: (f64, f64, i32),
    ) -> Result<(Vec<SplitInfo>, Vec<SplitInfo>), ComputeError> {
        let (smaller_h, larger_h) = {
            let mirror = self.resident_pool.borrow();
            let smaller_h =
                mirror.get(smaller_slot).and_then(|h| h.clone()).ok_or_else(|| {
                    ComputeError::Runtime {
                        detail: "scan_resident_siblings: smaller slot is empty".to_string(),
                    }
                })?;
            let larger_h = mirror.get(larger_slot).and_then(|h| h.clone()).ok_or_else(|| {
                ComputeError::Runtime {
                    detail: "scan_resident_siblings: larger slot is empty".to_string(),
                }
            })?;
            (smaller_h, larger_h)
        };
        kernels::split::find_best_splits_fused_siblings_from_handles_on(
            client,
            smaller_h,
            larger_h,
            slot_len,
            feats,
            cfg,
            smaller_totals,
            larger_totals,
        )
    }

    /// §8.2: scan the resident leaf then REDUCE the per-feature splits to the
    /// single winner ON DEVICE, reading back only the ~8-int winning split (payload collapses
    /// from `num_features` SplitInfo cells). The cross-feature reduce
    /// ([`kernels::grow_driver::argmax_over_resident_splits`]) folds in fpos order with the strict
    /// `>` lowest-real-feature-index tie-break, so the winner is bit-identical to the host
    /// `argmax_over_splits`. Bit-exact vs the integer path, ~1e-6 vs host-CUDA — never
    /// GPU-f32-vs-GPU-f32. The `!(sum_h>0)||num_data<=0` short-circuit mirrors the
    /// driver scan gate.
    #[allow(clippy::too_many_arguments)]
    fn scan_resident_leaf_argmax(
        &self,
        client: &ComputeClient<Self::Runtime>,
        slot: usize,
        slot_len: usize,
        feats: &[BatchedSplitFeature],
        real_feats: &[i32],
        cfg: &GainConfig,
        sum_gradient: f64,
        sum_hessian: f64,
        num_data: i32,
    ) -> Result<ResidentSplitWinner, ComputeError> {
        #[allow(clippy::neg_cmp_op_on_partial_ord)]
        if !(sum_hessian > 0.0) || num_data <= 0 {
            return Ok((SplitInfo::none(), -1));
        }
        let splits = self.scan_resident_leaf(
            client, slot, slot_len, feats, cfg, sum_gradient, sum_hessian, num_data,
        )?;
        Ok(kernels::grow_driver::argmax_over_resident_splits(&splits, feats, real_feats))
    }

    /// Co-packed 2-slot scan + per-side on-device reduce → each sibling's
    /// winner. Bit-identical to two `argmax_over_splits` folds; each side reads back only its
    /// winning ~8-int split.
    #[allow(clippy::too_many_arguments)]
    fn scan_resident_siblings_argmax(
        &self,
        client: &ComputeClient<Self::Runtime>,
        smaller_slot: usize,
        larger_slot: usize,
        slot_len: usize,
        feats: &[BatchedSplitFeature],
        real_feats: &[i32],
        cfg: &GainConfig,
        smaller_totals: (f64, f64, i32),
        larger_totals: (f64, f64, i32),
    ) -> Result<(ResidentSplitWinner, ResidentSplitWinner), ComputeError> {
        let (smaller_splits, larger_splits) = self.scan_resident_siblings(
            client,
            smaller_slot,
            larger_slot,
            slot_len,
            feats,
            cfg,
            smaller_totals,
            larger_totals,
        )?;
        Ok((
            kernels::grow_driver::argmax_over_resident_splits(&smaller_splits, feats, real_feats),
            kernels::grow_driver::argmax_over_resident_splits(&larger_splits, feats, real_feats),
        ))
    }

    /// §8.3 `FindBestFromAllSplitsKernel`: the cross-leaf best-leaf pick runs
    /// on the GpuBackend arm here (replacing the host loop in `grow_tree_on_device_resident`),
    /// via the deterministic `split_gt` first-max reduce
    /// ([`kernels::grow_driver::best_leaf_argmax`]) — bit-identical to the host loop's pick, so
    /// the on-device tree structure is unchanged. (The fully-resident device reduction over
    /// per-leaf records is the real-hardware refinement; the pick VALUE is fixed by the ordered
    /// `split_gt` fold and never a GPU-f32-vs-GPU-f32 comparison.)
    fn best_leaf_reduce(&self, leaf_best: &[SplitInfo], leaf_real_feat: &[i32]) -> i32 {
        kernels::grow_driver::best_leaf_argmax(leaf_best, leaf_real_feat)
    }

    /// §6.1 `CUDAInitValuesKernel`: the root grad/hess sum on the GpuBackend
    /// arm. Measurement found the prior on-device single-lane f64 fold kernel materially
    /// slower on real CUDA hardware than folding on the host, and proved via a
    /// register-accumulator A/B (bit-exact both variants, wash) that the cost is the
    /// un-hideable per-iteration memory latency of one GPU lane walking `num_data`
    /// elements with zero occupancy to overlap it — not the f64 width and not the
    /// accumulator shape. There is no in-kernel fix; the fix is routing. The caller
    /// (`grow_driver.rs:1949`) already holds the host `gradients: &[f32]` / `hessians: &[f32]`
    /// slices, and [`kernels::grow_driver::root_grad_hess_fold`] — the exact ascending f64 fold
    /// the device kernel was built to match bit-exact — is dramatically cheaper over the same
    /// data. This override now calls it directly, identical to the `Backend` trait's
    /// default impl (which `CpuBackend` inherits) — parity-neutral BY CONSTRUCTION
    /// since `root_grad_hess_fold` is the literal anchor the device kernel proves bit-exact
    /// against. `root_grad_hess_sum_device`/`root_grad_hess_sum_device_slices` (best_split.rs)
    /// remain defined and covered by their own bit-exact anchor tests and a dedicated cost
    /// harness — they are simply no longer called from this production seam.
    fn root_grad_hess_sum(
        &self,
        _client: &ComputeClient<Self::Runtime>,
        gradients: &[f32],
        hessians: &[f32],
    ) -> (f64, f64) {
        kernels::grow_driver::root_grad_hess_fold(gradients, hessians)
    }

    /// §8.2: run the device-resident cross-feature reduce ON DEVICE via
    /// [`DeviceFrontier::frontier_reduce_leaf`] — the winner is written into the resident
    /// frontier slot, no readback. Bit-exact to the host fold on the cpu f64 anchor.
    fn frontier_reduce_leaf_device(
        &self,
        client: &ComputeClient<Self::Runtime>,
        frontier: &DeviceFrontier<Self::Runtime>,
        in_slab: &kernels::best_split::SplitSoa,
        num_tasks: usize,
        is_smaller: bool,
        out_leaf: usize,
    ) -> Result<(), ComputeError> {
        frontier.frontier_reduce_leaf(client, in_slab, num_tasks, is_smaller, out_leaf)
    }

    /// §8.3: run the device-resident cross-leaf argmax + self-invalidation +
    /// 8-int export ON DEVICE via [`DeviceFrontier::frontier_pick_best_leaf`] — the ONLY
    /// device→host transfer is the single 8-int export. Bit-identical to the host pick on the
    /// cpu f64 anchor.
    fn frontier_pick_best_leaf_device(
        &self,
        client: &ComputeClient<Self::Runtime>,
        frontier: &DeviceFrontier<Self::Runtime>,
        smaller_leaf_index: i32,
        larger_leaf_index: i32,
        cur_num_leaves: usize,
    ) -> Result<kernels::best_split::PickExport, ComputeError> {
        frontier.frontier_pick_best_leaf(
            client,
            smaller_leaf_index,
            larger_leaf_index,
            cur_num_leaves,
        )
    }

    /// GPU override: mutate the device tree in place via
    /// [`DeviceCudaTree::split_on_device_scheduled`](kernels::tree::DeviceCudaTree::split_on_device_scheduled)
    /// with the right leaf id supplied from the fixed schedule — no `right_leaf_index`
    /// readback. Byte-identical tree structure to `split_on_device` / the host
    /// mutation on the cpu f64 anchor; static geometry.
    #[allow(clippy::too_many_arguments)]
    fn split_tree_scheduled(
        &self,
        client: &ComputeClient<Self::Runtime>,
        tree: &mut kernels::tree::DeviceCudaTree<Self::Runtime>,
        leaf_index: i32,
        right_leaf_index: i32,
        real_feature_index: i32,
        real_threshold: f64,
        missing_type: i32,
        scalars: &kernels::split_info::SplitScalars,
    ) -> Result<(), ComputeError> {
        tree.split_on_device_scheduled(
            client,
            leaf_index,
            right_leaf_index,
            real_feature_index,
            real_threshold,
            missing_type,
            scalars,
        )
    }

    /// FUSED build+fix+compact+scan for a directly-built leaf. Reads the
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
            resident.width,
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

    /// Scan the resident leaf and fold its winner DIRECTLY into the frontier
    /// slot via [`kernels::split::find_best_splits_fused_reduce_into_leaf_on`] — no host argmax
    /// readback (the winner lives device-resident, handed to the driver only through the §8.3
    /// pick export). Borrows the SAME resident-pool slot Handle `scan_resident_leaf` reads.
    #[allow(clippy::too_many_arguments)]
    fn scan_resident_leaf_into_frontier(
        &self,
        client: &ComputeClient<Self::Runtime>,
        slot: usize,
        slot_len: usize,
        feats: &[BatchedSplitFeature],
        real_feats: &[i32],
        cfg: &GainConfig,
        sum_gradient: f64,
        sum_hessian: f64,
        num_data: i32,
        frontier: &DeviceFrontier<Self::Runtime>,
        out_leaf: usize,
    ) -> Result<(), ComputeError> {
        let handle = {
            let mirror = self.resident_pool.borrow();
            mirror.get(slot).and_then(|h| h.clone()).ok_or_else(|| ComputeError::Runtime {
                detail: "scan_resident_leaf_into_frontier: slot is empty".to_string(),
            })?
        };
        kernels::split::find_best_splits_fused_reduce_into_leaf_on(
            client,
            handle,
            slot_len,
            feats,
            real_feats,
            cfg,
            sum_gradient,
            sum_hessian,
            num_data,
            frontier.records(),
            out_leaf,
        )
    }

    /// Co-scan BOTH siblings and fold each side's winner DIRECTLY into its
    /// frontier slot via [`kernels::split::find_best_splits_fused_siblings_reduce_into_leaves_on`]
    /// — no host argmax readback. Borrows the SAME two resident-pool slot Handles
    /// `scan_resident_siblings` reads (both siblings simultaneously resident at the co-pack point).
    #[allow(clippy::too_many_arguments)]
    fn scan_resident_siblings_into_frontier(
        &self,
        client: &ComputeClient<Self::Runtime>,
        smaller_slot: usize,
        larger_slot: usize,
        slot_len: usize,
        feats: &[BatchedSplitFeature],
        real_feats: &[i32],
        cfg: &GainConfig,
        smaller_totals: (f64, f64, i32),
        larger_totals: (f64, f64, i32),
        frontier: &DeviceFrontier<Self::Runtime>,
        out_leaf_smaller: usize,
        out_leaf_larger: usize,
    ) -> Result<(), ComputeError> {
        let (smaller_h, larger_h) = {
            let mirror = self.resident_pool.borrow();
            let smaller_h =
                mirror.get(smaller_slot).and_then(|h| h.clone()).ok_or_else(|| {
                    ComputeError::Runtime {
                        detail: "scan_resident_siblings_into_frontier: smaller slot is empty"
                            .to_string(),
                    }
                })?;
            let larger_h = mirror.get(larger_slot).and_then(|h| h.clone()).ok_or_else(|| {
                ComputeError::Runtime {
                    detail: "scan_resident_siblings_into_frontier: larger slot is empty"
                        .to_string(),
                }
            })?;
            (smaller_h, larger_h)
        };
        kernels::split::find_best_splits_fused_siblings_reduce_into_leaves_on(
            client,
            smaller_h,
            larger_h,
            slot_len,
            feats,
            real_feats,
            cfg,
            smaller_totals,
            larger_totals,
            frontier.records(),
            out_leaf_smaller,
            out_leaf_larger,
        )
    }

    /// The f64-fused escape hatch's zero-readback variant — build+fix+compact+scan
    /// in ONE launch via [`kernels::histogram::build_fix_scan_resident_reduce_f64_on`], STORE the
    /// fixed+compacted f64 Handle into mirror slot `slot` (for `subtract_resident`), and fold the
    /// scan winner DIRECTLY into frontier slot `out_leaf` — no per-feature-array readback.
    #[allow(clippy::too_many_arguments)]
    fn build_fix_scan_resident_into_frontier(
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
        real_feats: &[i32],
        cfg: &GainConfig,
        sum_gradient_raw: f64,
        sum_hessian_raw: f64,
        num_data: i32,
        frontier: &DeviceFrontier<Self::Runtime>,
        out_leaf: usize,
    ) -> Result<(), ComputeError> {
        let resident = self.resident_bins.borrow();
        let Some(resident) = resident.as_ref() else {
            return Err(ComputeError::Runtime {
                detail: "build_fix_scan_resident_into_frontier: resident bin cache empty \
                         (upload_resident_bins not called)"
                    .to_string(),
            });
        };
        let (handle, len) = kernels::histogram::build_fix_scan_resident_reduce_f64_on(
            client,
            resident.handle.clone(),
            resident.width,
            resident.num_features,
            resident.num_data,
            slot_off,
            slot_len,
            leaf_rows,
            gradients,
            hessians,
            feats,
            scan_active,
            real_feats,
            cfg,
            sum_gradient_raw,
            sum_hessian_raw,
            num_data,
            frontier.records(),
            out_leaf,
        )?;
        debug_assert_eq!(len, slot_len, "fused resident leaf handle length");
        // `resident` borrows `self.resident_bins`; `resident_pool` is a SEPARATE RefCell, so the
        // mutable borrow below does not conflict (mirrors `build_fix_scan_resident`).
        let mut mirror = self.resident_pool.borrow_mut();
        if slot >= mirror.len() {
            mirror.resize_with(slot + 1, || None);
        }
        mirror[slot] = Some(handle);
        Ok(())
    }
}

#[cfg(test)]
mod par_build_tests {
    use super::{build_histograms_into, BinColumn};

    /// Bit-exact guard: the rayon-parallel per-feature build MUST
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

    /// Microbench (`spike011_microbench`) — isolates the parallel build's `Vec<Vec<f64>>`+copy
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
    use crate::error::ComputeError;
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
        // RAW leaf sums over the leaf rows (the fix operand).
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

    // ---- subtract_scan (larger child: subtract → scan, NO fix) ----

    /// Build a parent histogram and a smaller-child histogram (both already
    /// fixed+compacted, as they are in the pool) for the fixture features, so the
    /// larger child = parent − smaller is exercised. The two histograms use disjoint
    /// row sets, but the test only needs them to be ARBITRARY valid f64 buffers — the
    /// subtract is a pure cell-wise op and the scan reads the derived buffer.
    fn parent_and_smaller(
        feats: &[BatchedSplitFeature],
        slot_len: usize,
    ) -> (Vec<f64>, Vec<f64>) {
        let mut parent = vec![0.0f64; slot_len];
        let mut smaller = vec![0.0f64; slot_len];
        for (fpos, f) in feats.iter().enumerate() {
            let cells = 2 * f.num_bin as usize;
            for k in 0..cells {
                // Deterministic, distinct, non-trivial values so subtract is meaningful.
                let h = ((fpos * 131 + k) as u64).wrapping_mul(2_654_435_761);
                parent[f.slot_off + k] = (h % 1000) as f64 * 0.013 + 7.0;
                smaller[f.slot_off + k] = (h % 311) as f64 * 0.007 + 1.0;
            }
        }
        (parent, smaller)
    }

    /// BEHAVIOR 1+2: subtract_scan writes larger_buf == parent − smaller cell-for-cell
    /// for EVERY feature, AND returns SplitInfos byte-identical to a separate per-feature
    /// scan of that derived buffer, in EXACTLY feature-index order. NO fix step.
    #[test]
    fn subtract_scan_bit_exact_and_ordered_vs_two_step() {
        let backend = CpuBackend;
        let client = cpu_client();
        let cfg = relaxed_cfg();
        let (_cols, _num_bins, _slot_off, slot_len, _leaf_rows, _grads, _hess, feats, sum_g, sum_h, num_data) =
            fixture();
        let (parent, smaller) = parent_and_smaller(&feats, slot_len);

        // Two-step reference: whole-buffer subtract, then per-feature scan.
        let derived_ref = backend
            .subtract_histograms(&client, &parent, &smaller)
            .expect("two-step subtract");

        let scan_active = vec![true; feats.len()];
        let mut larger_buf = vec![0.0f64; slot_len];
        let got = backend
            .subtract_scan(
                &client, &parent, &smaller, &mut larger_buf, &feats, &scan_active, &cfg, sum_g,
                sum_h, num_data,
            )
            .expect("subtract_scan");

        // (a) larger_buf bit-identical to the whole-buffer subtract over every cell.
        for (i, (w, g)) in derived_ref.iter().zip(&larger_buf).enumerate() {
            assert_eq!(w.to_bits(), g.to_bits(), "larger_buf cell {i}: two-step {w} != fused {g}");
        }

        // (b) per-feature SplitInfos byte-identical to a separate scan of derived_ref.
        assert_eq!(got.len(), feats.len(), "one result slot per feature");
        for (fpos, f) in feats.iter().enumerate() {
            let cells = 2 * f.num_bin as usize;
            let hist = &derived_ref[f.slot_off..f.slot_off + cells];
            let want = backend
                .find_best_split(
                    &client, hist, &cfg, f.num_bin, f.offset, f.default_bin, f.most_freq_bin,
                    f.skip_default_bin, f.na_as_missing, f.run_forward, sum_g, sum_h, num_data,
                )
                .expect("two-step scan");
            let g = got[fpos].expect("scan-active feature has a SplitInfo");
            assert_eq!(g.gain.to_bits(), want.gain.to_bits(), "feature {fpos}: gain");
            assert_eq!(g.threshold, want.threshold, "feature {fpos}: threshold");
            assert_eq!(g.default_left, want.default_left, "feature {fpos}: default_left");
            assert_eq!(g.left_count, want.left_count, "feature {fpos}: left_count");
            assert_eq!(
                g.left_sum_gradient.to_bits(),
                want.left_sum_gradient.to_bits(),
                "feature {fpos}: left_sum_gradient"
            );
        }
    }

    /// BEHAVIOR: an ineligible (non-scan-active) feature returns `None` (NOT scanned)
    /// while still being subtracted into larger_buf. Scan-active features return `Some`.
    #[test]
    fn subtract_scan_ineligible_feature_returns_none_but_still_subtracted() {
        let backend = CpuBackend;
        let client = cpu_client();
        let cfg = relaxed_cfg();
        let (_cols, _num_bins, _slot_off, slot_len, _leaf_rows, _grads, _hess, feats, sum_g, sum_h, num_data) =
            fixture();
        let (parent, smaller) = parent_and_smaller(&feats, slot_len);
        let derived_ref = backend
            .subtract_histograms(&client, &parent, &smaller)
            .expect("two-step subtract");

        // Features 0 and 2 active, 1 and 3 ineligible.
        let scan_active = vec![true, false, true, false];
        let mut larger_buf = vec![0.0f64; slot_len];
        let res = backend
            .subtract_scan(
                &client, &parent, &smaller, &mut larger_buf, &feats, &scan_active, &cfg, sum_g,
                sum_h, num_data,
            )
            .expect("subtract_scan");
        assert!(res[0].is_some(), "feature 0 active -> Some");
        assert!(res[1].is_none(), "feature 1 ineligible -> None");
        assert!(res[2].is_some(), "feature 2 active -> Some");
        assert!(res[3].is_none(), "feature 3 ineligible -> None");
        // Buffer is COMPLETE even for non-scan-active features (larger child needs all).
        for (i, (w, g)) in derived_ref.iter().zip(&larger_buf).enumerate() {
            assert_eq!(w.to_bits(), g.to_bits(), "buf cell {i}: subtract must run for all feats");
        }
    }

    /// BEHAVIOR: a length mismatch (larger_buf too short for a feature's region) is the
    /// deterministic lowest-index LengthMismatch, matching the hoisted validation.
    #[test]
    fn subtract_scan_length_mismatch_is_deterministic() {
        let backend = CpuBackend;
        let client = cpu_client();
        let cfg = relaxed_cfg();
        let (_cols, _num_bins, _slot_off, slot_len, _leaf_rows, _grads, _hess, feats, sum_g, sum_h, num_data) =
            fixture();
        let (parent, smaller) = parent_and_smaller(&feats, slot_len);
        let scan_active = vec![true; feats.len()];
        // larger_buf one cell short of the last feature's region end.
        let mut larger_buf = vec![0.0f64; slot_len - 1];
        let err = backend
            .subtract_scan(
                &client, &parent, &smaller, &mut larger_buf, &feats, &scan_active, &cfg, sum_g,
                sum_h, num_data,
            )
            .expect_err("short larger_buf must error");
        assert!(matches!(err, ComputeError::LengthMismatch { .. }), "got {err:?}");
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

#[cfg(test)]
mod core_scaled_threshold_tests {
    use super::{
        core_scaled_threshold, unified_bfs_threshold, unified_subscan_threshold,
        THRESHOLD_CEILING, THRESHOLD_FLOOR,
    };

    /// HARD no-regression invariant: at THIS machine's 16 cores the
    /// derived default MUST reproduce the measured/shipped optima (BFS 100, SUBSCAN 130)
    /// exactly — the anchor is the one point the proxy sweep can trust absolutely.
    #[test]
    fn anchor_reproduced_exactly_at_16_cores() {
        assert_eq!(core_scaled_threshold(100, 16), 100);
        assert_eq!(core_scaled_threshold(130, 16), 130);
    }

    /// The crossover RISES with core count (measured, MATERIAL): fewer cores ⇒
    /// fusion amortizes at fewer features ⇒ lower threshold; more cores ⇒ higher.
    /// Values are the additive-log shape `anchor − 17·log2(16/cores)`, clamped.
    #[test]
    fn monotone_rising_with_cores_for_both_anchors() {
        for anchor in [100usize, 130] {
            let mut prev = 0usize;
            for cores in [1usize, 2, 4, 8, 16, 32, 64, 128] {
                let t = core_scaled_threshold(anchor, cores);
                assert!(
                    t >= prev,
                    "anchor {anchor}: threshold must not DECREASE as cores rise: \
                     cores={cores} gave {t} < prev {prev}"
                );
                prev = t;
            }
        }
    }

    /// Below 16 cores the derived default is BELOW the 16-core anchor (fusion engages
    /// earlier); above 16 cores it is ABOVE (fusion engages later).
    #[test]
    fn below_and_above_anchor_bracket_16() {
        assert!(core_scaled_threshold(100, 2) < 100);
        assert!(core_scaled_threshold(100, 8) < 100);
        assert!(core_scaled_threshold(100, 128) > 100);
        assert!(core_scaled_threshold(130, 4) < 130);
        assert!(core_scaled_threshold(130, 128) > 130);
    }

    /// Clamps protect the extremes: a 128-core box never gets a pathologically high
    /// threshold (fusion would never engage), and a 1-core box never gets an absurdly
    /// tiny one (parallelizing a trivial leaf). Floor 32 / ceiling 256, justified from
    /// the measured curve (crossovers stayed within [~20, ~250]).
    #[test]
    fn clamped_to_floor_and_ceiling_at_extremes() {
        for anchor in [100usize, 130] {
            for cores in [1usize, 2, 4, 8, 16, 32, 64, 128, 256, 1024] {
                let t = core_scaled_threshold(anchor, cores);
                assert!(
                    (THRESHOLD_FLOOR..=THRESHOLD_CEILING).contains(&t),
                    "anchor {anchor} cores {cores}: {t} escaped [{THRESHOLD_FLOOR},{THRESHOLD_CEILING}]"
                );
            }
        }
        // 1-core BFS anchor=100 hits the floor (100 − 17·4 = 32 → exactly the floor).
        assert_eq!(core_scaled_threshold(100, 1), THRESHOLD_FLOOR);
        // Cores=0 (degenerate / impossible from rayon) is treated as 1, never panics.
        assert_eq!(core_scaled_threshold(100, 0), core_scaled_threshold(100, 1));
    }

    /// Env override still wins with ULTIMATE precedence — the derived default is only
    /// consulted when the env var is absent/unparseable.
    #[test]
    fn env_override_takes_ultimate_precedence() {
        // Serialize env mutation; these two vars are only read here. Edition-2024 env
        // mutation is `unsafe` (process-global) — safe here: single-threaded test, vars
        // set then immediately removed, exercised by no other test.
        unsafe {
            std::env::set_var("LGBM_UNIFIED_BFS_THRESHOLD", "777");
            std::env::set_var("LGBM_UNIFIED_SUBSCAN_THRESHOLD", "888");
        }
        assert_eq!(unified_bfs_threshold(), 777);
        assert_eq!(unified_subscan_threshold(), 888);
        unsafe {
            std::env::remove_var("LGBM_UNIFIED_BFS_THRESHOLD");
            std::env::remove_var("LGBM_UNIFIED_SUBSCAN_THRESHOLD");
        }
        // With env cleared, the derived default is in the sane clamp band.
        let b = unified_bfs_threshold();
        let s = unified_subscan_threshold();
        assert!((THRESHOLD_FLOOR..=THRESHOLD_CEILING).contains(&b));
        assert!((THRESHOLD_FLOOR..=THRESHOLD_CEILING).contains(&s));
    }
}

/// The on-device seam is ON by default and the
/// tree-growth discriminator stays `true` unless explicitly disabled via
/// `LGBM_CUDA_ON_DEVICE="0"`.
#[cfg(all(test, feature = "cpu"))]
mod on_device_seam_tests {
    use super::{cuda_on_device_enabled, Backend, CpuBackend};

    /// `LGBM_CUDA_ON_DEVICE` is ON by default (env unset in the test process): the
    /// production gate the growth driver checks returns `true`, so the on-device
    /// growth path is reachable without an explicit opt-in.
    #[test]
    fn cuda_on_device_seam_on_by_default() {
        assert!(cuda_on_device_enabled(), "LGBM_CUDA_ON_DEVICE must be ON by default");
    }

    /// The tree-growth discriminator is `true` by default; the learner's on-device
    /// eligibility gate ANDs this in, so trained models take the on-device grow path
    /// unless `LGBM_CUDA_ON_DEVICE="0"` or a specific eligibility exclusion applies
    /// (e.g. categorical + quantized-gradient combo).
    #[test]
    fn on_device_growth_supported_stays_true() {
        let backend = CpuBackend;
        assert!(
            backend.on_device_growth_supported(),
            "on_device_growth_supported must stay true by default"
        );
    }
}
