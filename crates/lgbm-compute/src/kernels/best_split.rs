//! On-device best-split finder — the 3-stage split-finding pipeline (§8) — **17-01+**.
//!
//! Owning phase: **17** (ODL-11 / ODL-12). Scope locked by **D-01…D-11**
//! (`17-CONTEXT.md`) and the six research flags resolved in `17-RESEARCH.md`.
//!
//! ## What lives here
//! This module ports `cuda_best_split_finder.cu` (§8) — the block-prefix-sum →
//! complement-from-parent → count-recovery → guards → gain → argmax numerical core
//! and its 3 faithful separate stages (stage-1 per-task eval / stage-2 cross-feature
//! reduce / stage-3 cross-leaf argmax + the single 8-int readback, D-06). It is
//! **additive and OFF by default behind `LGBM_CUDA_ON_DEVICE`** (D-09): the CPU /
//! ROCm / existing-host-CUDA paths stay byte-unchanged and
//! [`crate::Backend::on_device_growth_supported`] STAYS `false` this milestone. The
//! `best_split` module is declared **ungated** in `mod.rs` (NOT `#[cfg(feature =
//! "gpu")]`) so the default cubecl-cpu f64 anchor exercises this code (D-08).
//!
//! ## Wave-0 scaffolding (this plan, 17-01)
//! This file starts as the pure-host scaffolding every downstream kernel plan
//! (Waves 2-4) consumes BEFORE the kernel is written (17-VALIDATION Wave-0-first):
//! - [`SplitFindTask`] — the Rust mirror of `cuda_best_split_finder.hpp:28-41`.
//! - [`build_split_find_tasks`] — the host task-gen builder reproducing the C++
//!   `assume_out_default_left` table (`cuda_best_split_finder.cpp:137-227`) EXACTLY.
//! - [`round_ties_even`] — the count-recovery rounding helper (round-to-nearest,
//!   ties-to-EVEN) faithful to CUDA `__double2int_rn`. This DIVERGES from
//!   [`super::split::round_int`]'s round-half-up `(int)(x + 0.5f)` — the D-01
//!   landmine (17-RESEARCH Pitfall 1): counts recovered with round-half-up flip a
//!   `left_count >= min_data_in_leaf` guard at exact half-values and silently
//!   change the winning threshold.
//! - The stage-1/2/3 launcher STUB signatures ([`find_best_splits_stage1_on`],
//!   [`sync_best_split_for_leaf_on`], [`find_best_from_all_splits_on`]) — they
//!   COMPILE and return sentinels (an all-`is_valid=false` record / `best_leaf=-1`
//!   export) so the Wave-0 golden-anchor harness RED-fails via ASSERTION, not a
//!   build error. The numerical core lands in Waves 2-4.
//!
//! ## Per-task RNG seed formula (Open Q1 — LOCKED, `17-RESEARCH` Open Question 1)
//! The USE_RAND (extra-trees) path draws `rand_threshold =
//! CUDARandom.NextInt(0, num_bin - 2)` from a per-task `CUDARandom` seeded, per
//! `InitCUDARandomKernel` (`cuda_best_split_finder.cu:2220-2228`), as
//! `cuda_randoms[task_index].SetSeed(extra_seed + task_index)` — i.e. the per-task
//! seed is `extra_seed + task_index`. This is documentation prose locking Open Q1
//! so the Wave-2 USE_RAND kernel (17-03) has NO open flag; it is consumed there.
//!
//! ## Analog files
//! - `crates/lgbm-compute/src/kernels/split.rs` — the HOST serial scan (the WRONG
//!   `round_int` rounding for this path, and `default_left = REVERSE`); this module
//!   is the SEPARATE CUDA-core fold that D-01 mandates precisely because those two
//!   diverge.
//! - `crates/lgbm-compute/src/kernels/split_info.rs` — the pre-allocated
//!   `DeviceSplitInfo` / [`SplitScalars`] `CUDASplitInfo` analog the stage records
//!   and 8-int export write into (D-11).

use cubecl::prelude::*;

use lgbm_dataset::MissingType;

use crate::error::ComputeError;
use crate::kernels::split_info::SplitScalars;

/// The host-constructed per-(feature, direction) split-find task — a field-for-field
/// Rust mirror of the C++ `SplitFindTask` (`cuda_best_split_finder.hpp:28-41`).
///
/// One inner feature emits ONE or TWO tasks (forward+reverse) per
/// [`build_split_find_tasks`]. The stage-1 kernel reads a task's scalars to index
/// the resident histogram (`hist_offset`), scan its bins (`num_bin`, `mfb_offset`,
/// `default_bin`, `reverse`, `skip_default_bin`, `na_as_missing`), and — the D-01
/// landmine — writes `default_left = assume_out_default_left` VERBATIM (NOT
/// `reverse`; 17-RESEARCH Pitfall 3). `is_categorical`/`is_one_hot` are the Phase-22
/// dispatch seam (D-04). `rand_threshold` carries the USE_RAND drawn threshold
/// (`-1` when extra-trees is off).
///
/// Widths are faithful to the C++ struct: `inner_feature_index`/`rand_threshold`
/// are `data_size_t` (`i32`), `hist_offset`/`num_bin`/`default_bin` are `u32`,
/// `mfb_offset` is `u8`, and the six dispatch flags are `bool`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SplitFindTask {
    /// The inner feature index this task evaluates (`inner_feature_index`, i32).
    pub inner_feature_index: i32,
    /// Whether this is the REVERSE-direction scan (`reverse`).
    pub reverse: bool,
    /// Whether the default bin is skipped during the scan (`skip_default_bin`).
    pub skip_default_bin: bool,
    /// Whether NaN occupies the top bin as the missing marker (`na_as_missing`).
    pub na_as_missing: bool,
    /// The `default_left` value written VERBATIM into the split record
    /// (`assume_out_default_left`) — decoupled from `reverse` (Pitfall 3).
    pub assume_out_default_left: bool,
    /// Phase-22 dispatch seam: this task's feature is categorical (`is_categorical`).
    pub is_categorical: bool,
    /// Phase-22 dispatch seam: categorical one-hot (`is_one_hot`, `num_bin <=
    /// max_cat_to_onehot`).
    pub is_one_hot: bool,
    /// This feature's start offset into the resident histogram (`hist_offset`, u32).
    pub hist_offset: u32,
    /// The most-frequent-bin layout offset (`mfb_offset`, u8; 1 iff most_freq_bin==0).
    pub mfb_offset: u8,
    /// The feature's bin count (`num_bin`, u32).
    pub num_bin: u32,
    /// The feature's default bin (`default_bin`, u32).
    pub default_bin: u32,
    /// The USE_RAND drawn threshold (`rand_threshold`, i32; `-1` when extra-trees off).
    pub rand_threshold: i32,
}

/// The host feature-metadata input to [`build_split_find_tasks`] — the per-inner-
/// feature fields the C++ task-gen loop (`cuda_best_split_finder.cpp:137-227`)
/// reads to decide the emitted task(s) and their `assume_out_default_left`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FeatureMeta {
    /// The inner feature index (`inner_feature_index`).
    pub inner_feature_index: i32,
    /// The feature's bin count (`feature_num_bins_[i]`).
    pub num_bin: u32,
    /// The feature's missing type (`feature_missing_type_[i]`).
    pub missing_type: MissingType,
    /// Whether the feature is categorical (`is_categorical_[i]`).
    pub is_categorical: bool,
    /// The `max_cat_to_onehot_` threshold used to set `is_one_hot`.
    pub max_cat_to_onehot: i32,
    /// This feature's histogram start offset (`feature_hist_offsets_[i]`).
    pub hist_offset: u32,
    /// The most-frequent-bin layout offset (`feature_mfb_offsets_[i]`, u8).
    pub mfb_offset: u8,
    /// The feature's default bin (`feature_default_bins_[i]`).
    pub default_bin: u32,
    /// The USE_RAND drawn threshold to stamp (`-1` when extra-trees off). The actual
    /// draw (`CUDARandom.NextInt(0, num_bin-2)` seeded `extra_seed + task_index`) is
    /// a Wave-2 concern; the host builder only carries the value through.
    pub rand_threshold: i32,
}

/// `__double2int_rn` — round to nearest, ties to EVEN (IEEE round-half-to-even),
/// the CUDA-core count-recovery rounding (`cnt = __double2int_rn(scanned_hess *
/// cnt_factor)`, `cuda_best_split_finder.cu`).
///
/// This DIVERGES from [`super::split::round_int`] (`(int)(x + 0.5f)` =
/// round-half-up-then-truncate, the host `Common::RoundInt`). D-01 mandates a
/// SEPARATE fold precisely because these two round differently at exact half-values
/// (17-RESEARCH Pitfall 1). Hessian·cnt_factor ≥ 0 here, so the `x >= 0` domain is
/// sufficient.
///
/// Primary body uses the stable [`f64::round_ties_even`] intrinsic. Because
/// research Assumption A1 (does cubecl-cpu `#[cube]` lowering support
/// `round_ties_even`?) is UNVERIFIED, [`round_ties_even_branchfree`] provides the
/// branch-free even-round identity as the fallback; Wave 2 selects whichever
/// cubecl-cpu lowers inside `#[cube]` (both are proven equivalent on `x >= 0` by
/// the unit test below).
#[inline]
pub fn round_ties_even(x: f64) -> i32 {
    x.round_ties_even() as i32
}

/// Branch-free round-half-to-even for `x >= 0` (the `#[cube]`-lowering fallback for
/// [`round_ties_even`], 17-RESEARCH §"Count Recovery"). Kept byte-equivalent to the
/// intrinsic on the non-negative domain (hessian·cnt_factor ≥ 0).
#[inline]
pub fn round_ties_even_branchfree(x: f64) -> i32 {
    let f = x.floor();
    let diff = x - f; // in [0, 1)
    // tie (diff == 0.5) rounds to the EVEN neighbour; otherwise round-half-up.
    let up = diff > 0.5 || (diff == 0.5 && ((f as i64) & 1 == 1));
    (if up { f + 1.0 } else { f }) as i32
}

/// Reproduce the C++ `SplitFindTask` task-gen table
/// (`cuda_best_split_finder.cpp:137-227`) EXACTLY.
///
/// Emits, per inner feature, in C++ order (so task indices line up with the
/// smaller/larger stream split the Wave-3 stage-2 reader expects — smaller task `t`
/// ↔ record `[t]`, larger task `t` ↔ `[t + num_tasks]`):
///
/// | Feature condition | Tasks emitted | `assume_out_default_left` |
/// |---|---|---|
/// | `num_bin>2 && missing==Zero && !cat` | forward (skip_default_bin) THEN reverse (skip_default_bin) | fwd=**false**, rev=**true** |
/// | `num_bin>2 && missing==NaN && !cat` | forward (na_as_missing) THEN reverse (na_as_missing) | fwd=**false**, rev=**true** |
/// | `num_bin<=2 or missing==None`, non-cat | single reverse task | `(missing != NaN) ? **true** : **false**` |
/// | categorical | single forward task (`is_one_hot = num_bin <= max_cat_to_onehot`) | **false** (Phase-22 seam, D-04) |
///
/// The D-01 landmine (Pitfall 3): `default_left` is precomputed here at task-gen
/// time from the missing type, NOT from `reverse` — a `MissingType::None` feature
/// yields a single `reverse=true` task with `assume_out_default_left=false`
/// (`default_left != reverse`). No categorical eval math lives here (D-04 wires the
/// `is_categorical`/`is_one_hot` dispatch seam only; Phase 22 fills the eval).
pub fn build_split_find_tasks(features: &[FeatureMeta]) -> Vec<SplitFindTask> {
    let mut tasks = Vec::new();
    for f in features {
        if f.num_bin > 2 && f.missing_type != MissingType::None && !f.is_categorical {
            // Forward-then-reverse PAIR. `skip_default_bin`/`na_as_missing` differ by
            // missing type; `assume_out_default_left` is false on forward, true on
            // reverse (cpp:141-171 Zero, :172-200 NaN).
            let (skip_default_bin, na_as_missing) = match f.missing_type {
                MissingType::Zero => (true, false),
                MissingType::NaN => (false, true),
                MissingType::None => unreachable!("guarded by missing_type != None above"),
            };
            // Forward task (assume_out_default_left = false).
            tasks.push(SplitFindTask {
                inner_feature_index: f.inner_feature_index,
                reverse: false,
                skip_default_bin,
                na_as_missing,
                assume_out_default_left: false,
                is_categorical: false,
                is_one_hot: false,
                hist_offset: f.hist_offset,
                mfb_offset: f.mfb_offset,
                num_bin: f.num_bin,
                default_bin: f.default_bin,
                rand_threshold: f.rand_threshold,
            });
            // Reverse task (assume_out_default_left = true).
            tasks.push(SplitFindTask {
                inner_feature_index: f.inner_feature_index,
                reverse: true,
                skip_default_bin,
                na_as_missing,
                assume_out_default_left: true,
                is_categorical: false,
                is_one_hot: false,
                hist_offset: f.hist_offset,
                mfb_offset: f.mfb_offset,
                num_bin: f.num_bin,
                default_bin: f.default_bin,
                rand_threshold: f.rand_threshold,
            });
        } else {
            // Single task (cpp:202-227). Categorical → forward one-hot seam; else
            // reverse. `default_left = (missing != NaN && !categorical)`.
            let (reverse, is_categorical, is_one_hot) = if f.is_categorical {
                (false, true, (f.num_bin as i32) <= f.max_cat_to_onehot)
            } else {
                (true, false, false)
            };
            let assume_out_default_left = f.missing_type != MissingType::NaN && !f.is_categorical;
            tasks.push(SplitFindTask {
                inner_feature_index: f.inner_feature_index,
                reverse,
                skip_default_bin: false,
                na_as_missing: false,
                assume_out_default_left,
                is_categorical,
                is_one_hot,
                hist_offset: f.hist_offset,
                mfb_offset: f.mfb_offset,
                num_bin: f.num_bin,
                default_bin: f.default_bin,
                rand_threshold: f.rand_threshold,
            });
        }
    }
    tasks
}

/// The shared per-task stage-1 scalars — the leaf totals + gain/guard config the
/// stage-1 kernel reads alongside a [`SplitFindTask`]. Mirrors the
/// `CUDALeafSplitsStruct` leaf totals (`sum_gradient`/`sum_hessian`/`num_data`/
/// `parent_output`/`parent_gain`) + the `Config` guard/gain scalars. Carried as the
/// flattened literal-friendly widths (u32 flags, i32 counts, f64 scalars) so the
/// Wave-2 `#[cube]` body does not have to reshape them (`split.rs:180-202` MLIR
/// lowering constraints).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Stage1Scalars {
    /// `USE_L1` dispatch flag (`lambda_l1 > 0`).
    pub use_l1: bool,
    /// `USE_SMOOTHING` dispatch flag (`path_smooth > 0`).
    pub use_smoothing: bool,
    /// `USE_RAND` dispatch flag (extra-trees).
    pub use_rand: bool,
    /// `IS_LARGER` task-index base selector (smaller → `[t]`, larger → `[t+num_tasks]`).
    pub is_larger: bool,
    /// `min_data_in_leaf` guard.
    pub min_data_in_leaf: i32,
    /// `min_sum_hessian_in_leaf` guard.
    pub min_sum_hessian_in_leaf: f64,
    /// `lambda_l1`.
    pub lambda_l1: f64,
    /// `lambda_l2`.
    pub lambda_l2: f64,
    /// `min_gain_to_split` (added to `parent_gain` for `min_gain_shift`).
    pub min_gain_to_split: f64,
    /// `path_smooth` (USE_SMOOTHING blend).
    pub path_smooth: f64,
    /// The leaf's own output (`parent_output`, from `CUDALeafSplitsStruct`).
    pub parent_output: f64,
    /// The leaf total gradient sum (`sum_gradients`).
    pub sum_gradient: f64,
    /// The leaf total hessian sum (`sum_hessians`).
    pub sum_hessian: f64,
    /// The leaf row count (`num_data`).
    pub num_data: i32,
    /// The leaf gain (`parent_gain`) — `min_gain_shift = parent_gain + min_gain_to_split`.
    pub parent_gain: f64,
    /// The per-task RNG seed `extra_seed + task_index` (Open Q1) — consumed by the
    /// Wave-2 USE_RAND path; ignored when `use_rand == false`.
    pub rng_seed: i32,
}

/// STAGE 1 (STUB — Wave 2 fills the numerical core, 17-03).
///
/// `FindBestSplitsForLeafKernel<USE_RAND, USE_L1, USE_SMOOTHING, IS_LARGER>`: block
/// prefix-sum → complement-from-parent → count recovery ([`round_ties_even`]) →
/// guards → gain (`crate::gain`, + net-new smoothing branch) → block argmax, writing
/// ONE per-task [`SplitScalars`] `CUDASplitInfo` record. `hist` is the interleaved
/// `[g0,h0,g1,h1,…]` f64 histogram for the task's feature (already offset).
///
/// Wave-0 sentinel: returns [`SplitScalars::default`] (`is_valid=false`) so the
/// golden-anchor harness RED-fails via ASSERTION (the golden expects a valid record),
/// NOT a compile/panic error. The `client` is threaded now so the Wave-2 signature
/// does not change.
///
/// # Errors
/// [`ComputeError`] once the Wave-2 launch-boundary validation lands; the stub is
/// infallible.
pub fn find_best_splits_stage1_on<R: cubecl::Runtime>(
    client: &ComputeClient<R>,
    hist: &[f64],
    task: &SplitFindTask,
    scalars: &Stage1Scalars,
) -> Result<SplitScalars, ComputeError> {
    // Wave-0 stub: numerical core deferred to 17-03 (Wave 2). Thread the inputs so
    // the signature is stable; return the `is_valid=false` sentinel.
    let _ = (client, hist, task, scalars);
    Ok(SplitScalars::default())
}

/// STAGE 2 (STUB — Wave 3 fills the cross-feature reduce, 17-04).
///
/// `SyncBestSplitForLeafKernel`: block-reduce the per-task `(is_valid, gain)` records
/// for one leaf via `ReduceBestGain` (strict `>` ⇒ lowest task index survives a tie,
/// 17-RESEARCH Pitfall 5) into the leaf's best split. `read_index = is_smaller ?
/// task_index : task_index + num_tasks` (the IS_LARGER duality).
///
/// Wave-0 sentinel: returns [`SplitScalars::default`] (`is_valid=false`).
///
/// # Errors
/// [`ComputeError`] once Wave-3 launch-boundary validation lands; the stub is
/// infallible.
pub fn sync_best_split_for_leaf_on<R: cubecl::Runtime>(
    client: &ComputeClient<R>,
    per_task: &[SplitScalars],
    num_tasks: usize,
    is_smaller: bool,
) -> Result<SplitScalars, ComputeError> {
    let _ = (client, per_task, num_tasks, is_smaller);
    Ok(SplitScalars::default())
}

/// STAGE 3 (STUB — Wave 4 fills the cross-leaf argmax + 8-int export, 17-05).
///
/// `FindBestFromAllSplitsKernel` + `PrepareLeafBestSplitInfo`: cross-leaf argmax over
/// `(gain, leaf_index)` → `best_leaf_index`, SELF-INVALIDATE the chosen leaf AND the
/// freshly-created leaf slot (`cur_num_leaves`), then pack the 8-int buffer (the ONLY
/// device→host transfer per iteration, SC#2). Field layout (17-RESEARCH §"3-Stage
/// Reduction & Export"): `[0]` smaller.inner_feature_index, `[1]` smaller.threshold,
/// `[2]` smaller.default_left, `[3..6]` larger triple, `[6]` best_leaf_index (`-1` if
/// none), `[7]` the best leaf's categorical-threshold count (0 for continuous; Phase
/// 22 fills it for categorical).
///
/// Wave-0 sentinel: returns the `best_leaf = -1` (no split) export.
///
/// # Errors
/// [`ComputeError`] once Wave-4 launch-boundary validation lands; the stub is
/// infallible.
pub fn find_best_from_all_splits_on<R: cubecl::Runtime>(
    client: &ComputeClient<R>,
    per_leaf: &[SplitScalars],
    cur_num_leaves: usize,
) -> Result<[i64; 8], ComputeError> {
    let _ = (client, per_leaf, cur_num_leaves);
    // [6] = best_leaf_index = -1 (no split); all other cells sentinel-zero.
    Ok([0, 0, 0, 0, 0, 0, -1, 0])
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The D-01 count-recovery landmine (17-RESEARCH Pitfall 1): `round_ties_even`
    /// rounds `k.5` to the nearest EVEN integer, DIVERGING from `split.rs::round_int`
    /// (round-half-up). Proven for both the intrinsic and the branch-free fallback.
    #[test]
    fn count_recovery_ties_even() {
        // Ties (k.5): round to the EVEN neighbour.
        assert_eq!(round_ties_even(0.5), 0, "0.5 → 0 (even)");
        assert_eq!(round_ties_even(1.5), 2, "1.5 → 2 (even)");
        assert_eq!(round_ties_even(2.5), 2, "2.5 → 2 (even, NOT 3)");
        assert_eq!(round_ties_even(3.5), 4, "3.5 → 4 (even)");
        // Non-ties: ordinary round-to-nearest.
        assert_eq!(round_ties_even(2.4), 2, "2.4 → 2");
        assert_eq!(round_ties_even(2.6), 3, "2.6 → 3");

        // The load-bearing divergence: round-half-up (`(int)(x + 0.5f)`) would give
        // 3 for 2.5; ties-to-even gives 2. This is exactly why D-01 mandates a
        // separate fold from `split.rs::round_int`.
        let round_half_up = |x: f64| (x + 0.5_f32 as f64) as i32;
        assert_eq!(round_half_up(2.5), 3, "round-half-up gives 3 for 2.5");
        assert_ne!(
            round_ties_even(2.5),
            round_half_up(2.5),
            "ties-to-even MUST diverge from round-half-up at 2.5"
        );

        // The branch-free `#[cube]`-lowering fallback is byte-equivalent on x >= 0.
        for &x in &[0.5, 1.5, 2.5, 3.5, 2.4, 2.6, 0.0, 10.5, 11.5] {
            assert_eq!(
                round_ties_even(x),
                round_ties_even_branchfree(x),
                "branch-free fallback must match the intrinsic at {x}"
            );
        }
    }

    /// Helper: a non-categorical `FeatureMeta` with the given num_bin/missing_type.
    fn feat(idx: i32, num_bin: u32, missing: MissingType) -> FeatureMeta {
        FeatureMeta {
            inner_feature_index: idx,
            num_bin,
            missing_type: missing,
            is_categorical: false,
            max_cat_to_onehot: 4,
            hist_offset: 0,
            mfb_offset: 0,
            default_bin: 0,
            rand_threshold: -1,
        }
    }

    /// The `assume_out_default_left` task-gen table
    /// (`cuda_best_split_finder.cpp:137-227`) — all four rows, including the D-01
    /// load-bearing divergence `default_left != reverse` (17-RESEARCH Pitfall 3).
    #[test]
    fn assume_out_default_left_table() {
        // Row 1: num_bin>2 && Zero → forward(assume=false) THEN reverse(assume=true),
        // both skip_default_bin, both !na_as_missing. Emission order fwd-then-rev.
        let zero = build_split_find_tasks(&[feat(0, 5, MissingType::Zero)]);
        assert_eq!(zero.len(), 2, "Zero num_bin>2 emits a forward+reverse PAIR");
        assert!(!zero[0].reverse, "task[0] is forward");
        assert!(zero[1].reverse, "task[1] is reverse");
        assert!(!zero[0].assume_out_default_left, "forward assume=false");
        assert!(zero[1].assume_out_default_left, "reverse assume=true");
        assert!(zero[0].skip_default_bin && zero[1].skip_default_bin, "Zero → skip_default_bin");
        assert!(!zero[0].na_as_missing && !zero[1].na_as_missing, "Zero → !na_as_missing");
        // The Zero reverse task is the `reverse == true && assume == true` case.
        assert!(
            zero[1].reverse && zero[1].assume_out_default_left,
            "Zero num_bin>2 reverse: reverse==true AND assume_out_default_left==true"
        );

        // Row 2: num_bin>2 && NaN → forward(assume=false) THEN reverse(assume=true),
        // both na_as_missing, both !skip_default_bin.
        let nan = build_split_find_tasks(&[feat(0, 5, MissingType::NaN)]);
        assert_eq!(nan.len(), 2, "NaN num_bin>2 emits a forward+reverse PAIR");
        assert!(!nan[0].reverse && nan[1].reverse);
        assert!(!nan[0].assume_out_default_left && nan[1].assume_out_default_left);
        assert!(nan[0].na_as_missing && nan[1].na_as_missing, "NaN → na_as_missing");
        assert!(!nan[0].skip_default_bin && !nan[1].skip_default_bin, "NaN → !skip_default_bin");

        // Row 3a: MissingType::None (num_bin>2) → single reverse task. Per the C++
        // else-branch (cpp:216-220) `assume = (missing != NaN && !categorical)`, so
        // None → assume=**true**. default_left is still DECOUPLED from reverse (it is
        // set from the missing-type formula, NOT `= reverse` like host split.rs); for
        // None the two happen to coincide (both true).
        let none = build_split_find_tasks(&[feat(0, 5, MissingType::None)]);
        assert_eq!(none.len(), 1, "None emits a single reverse task");
        assert!(none[0].reverse, "None single task is reverse");
        assert!(
            none[0].assume_out_default_left,
            "MissingType::None non-cat: assume == (missing != NaN) == true (cpp:216-220)"
        );

        // Row 3b: num_bin<=2 Zero-missing → single reverse task, assume = (missing != NaN) = true.
        let small = build_split_find_tasks(&[feat(0, 2, MissingType::Zero)]);
        assert_eq!(small.len(), 1, "num_bin<=2 emits a single reverse task");
        assert!(small[0].reverse);
        assert!(
            small[0].reverse && small[0].assume_out_default_left,
            "num_bin<=2 Zero-missing reverse: reverse==true AND assume_out_default_left==true"
        );

        // Row 3c: THE load-bearing divergence `default_left != reverse` (Pitfall 3):
        // num_bin<=2 NaN, non-categorical → single reverse task, assume = (NaN != NaN)
        // = **false**. reverse==true WHILE assume_out_default_left==false — proving
        // default_left is decoupled from reverse (host split.rs would wrongly emit
        // default_left=reverse=true here).
        let small_nan = build_split_find_tasks(&[feat(0, 2, MissingType::NaN)]);
        assert_eq!(small_nan.len(), 1, "num_bin<=2 NaN emits a single reverse task");
        assert!(
            small_nan[0].reverse && !small_nan[0].assume_out_default_left,
            "num_bin<=2 NaN reverse: reverse==true AND assume_out_default_left==false \
             (default_left != reverse — Pitfall 3)"
        );

        // Row 4: categorical → single forward task, is_categorical=true,
        // is_one_hot = (num_bin <= max_cat_to_onehot), assume=false. No eval math.
        let mut cat = feat(0, 3, MissingType::None); // num_bin 3 <= max_cat_to_onehot 4 → one-hot
        cat.is_categorical = true;
        let cat_tasks = build_split_find_tasks(&[cat]);
        assert_eq!(cat_tasks.len(), 1, "categorical emits a single forward task");
        assert!(!cat_tasks[0].reverse, "categorical task is forward");
        assert!(cat_tasks[0].is_categorical, "is_categorical=true");
        assert!(cat_tasks[0].is_one_hot, "num_bin(3) <= max_cat_to_onehot(4) → one-hot");
        assert!(!cat_tasks[0].assume_out_default_left, "categorical assume=false (Phase-22 seam)");
        // Above the one-hot cap → is_one_hot=false.
        let mut cat_many = feat(0, 10, MissingType::None);
        cat_many.is_categorical = true;
        let cat_many_tasks = build_split_find_tasks(&[cat_many]);
        assert!(
            cat_many_tasks[0].is_categorical && !cat_many_tasks[0].is_one_hot,
            "num_bin(10) > max_cat_to_onehot(4) → categorical, NOT one-hot"
        );
    }
}
