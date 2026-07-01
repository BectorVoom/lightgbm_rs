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

use core::marker::PhantomData;

use cubecl::prelude::*;
use cubecl::server::Handle;

use lgbm_core::types::K_EPSILON;
use lgbm_dataset::MissingType;

use crate::error::ComputeError;
use crate::gain::{
    calculate_splitted_leaf_output, calculate_splitted_leaf_output_f32,
    calculate_splitted_leaf_output_smoothed, calculate_splitted_leaf_output_smoothed_f32,
    get_leaf_gain_given_output, get_leaf_gain_given_output_f32, get_leaf_gain_smoothed,
    get_leaf_gain_smoothed_f32, get_split_gains, get_split_gains_f32,
};
use crate::kernels::random::draw_rand_int32_on;
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

/// The number of f64 cells the stage-1 kernel writes per task (the flattened
/// [`SplitScalars`] the launcher decodes). Layout:
/// `[0]=is_valid [1]=threshold [2]=default_left [3]=gain [4]=left_sum_grad
///  [5]=left_sum_hess [6]=left_count [7]=right_sum_grad [8]=right_sum_hess
///  [9]=right_count [10]=left_value [11]=right_value [12]=left_gain [13]=right_gain]`.
const STAGE1_OUT_LEN: usize = 14;

/// `__double2int_rn` inside `#[cube]` — round to nearest, ties to EVEN, using only
/// `f64::floor` (a cubecl `Float` intrinsic) so it lowers on cubecl-cpu AND hip
/// (research Assumption A1: `f64::round_ties_even` is NOT relied on inside `#[cube]`).
///
/// Branch-free even-round identity for `x >= 0` (hessian·cnt_factor ≥ 0 here):
/// `f = floor(x)`, tie when `x - f == 0.5` rounds toward the EVEN neighbour; `f`'s
/// parity is `f - 2·floor(f/2)` (0 even, 1 odd) — pure float, no `i64` bit-ops. This
/// DIVERGES from [`super::split::round_int`]'s round-half-up (the D-01 landmine,
/// 17-RESEARCH Pitfall 1) — it is byte-equivalent to the host [`round_ties_even`].
#[cube]
fn round_ties_even_cube(x: f64) -> i32 {
    let f = f64::floor(x);
    let diff = x - f; // in [0, 1)
    // f's parity as a float: 0.0 if even, 1.0 if odd.
    let parity = f - 2.0 * f64::floor(f * 0.5);
    let f_is_odd = parity > 0.5;
    let up = diff > 0.5 || (diff == 0.5 && f_is_odd);
    let r = select(up, f + 1.0, f);
    i32::cast_from(r)
}

/// The stage-1 numerical core — a VERBATIM `#[cube]` transcription of
/// `FindBestSplitsForLeafKernelInner` (`cuda_best_split_finder.cu:146-320`) driven
/// SINGLE-OWNER (`CubeDim(1)`) as the deterministic cpu f64 fold (D-01/D-08). One
/// call evaluates one `(leaf,feature)` task: serial inclusive prefix-sum → cumulative
/// scanned side → complement-from-parent → two-phase count recovery
/// ([`round_ties_even_cube`]) → guards → gain (smoothing dispatch) → strict-`>`
/// argmax → the winning `CUDASplitInfo` record.
///
/// Landmines reproduced 1:1 (17-RESEARCH §"Common Pitfalls"):
/// - **Count recovery** is round-ties-EVEN, NOT `split.rs`'s round-half-up (Pitfall 1).
/// - **kEpsilon two-phase** (Pitfall 2): thread-0 adds `kEpsilon` ONCE at the scan
///   origin (the CUDA single-kEpsilon placement, NOT `split.rs`'s `2·kEpsilon`); the
///   guard recovers the count from the kEpsilon-INCLUDED hessian, the written record
///   subtracts kEpsilon first then re-recovers (an off-by-one between them is intended).
/// - **Complement-from-parent** (Pitfall 4): the non-scanned side is
///   `parent_total − scanned`, never a second scan; `reverse` flips only the default-bin
///   scan direction (`fnbmo-1-t` read, `num_bin-2-t` threshold) and the scanned/complement
///   left↔right assignment.
/// - **`default_left = assume_out_default_left`** (Pitfall 3), written verbatim, NOT
///   `reverse`.
/// - **strict `>` argmax** (Pitfall 5): the lowest bin index survives a tie.
///
/// `reverse`/`use_l1`/`use_smoothing`/`use_rand` are runtime `u32` flags (0|1) inside
/// the one shared body (research Pattern 2 — avoid a 16-way cubecl monomorphization).
/// Honors the `split.rs:180-202` MLIR constraints: loop-carried mutables init from
/// LITERALS, every conditional store is a branchless `select`, the scan is a bounded
/// RANGE loop. `min_gain_shift = parent_gain + min_gain_to_split` is host-computed.
/// The categorical eval is a Phase-22 seam (D-04) handled by the launcher, not here.
#[cube]
#[allow(clippy::too_many_arguments)]
pub fn split_eval_body(
    hist: &Array<f64>,
    out: &mut Array<f64>,
    num_bin: i32,
    mfb_offset: i32,
    default_bin: i32,
    skip_default_bin: u32,        // 0|1
    reverse: u32,                 // 0|1
    assume_out_default_left: u32, // 0|1 — written verbatim (Pitfall 3)
    use_l1: u32,                  // 0|1
    use_smoothing: u32,           // 0|1
    use_rand: u32,                // 0|1
    rand_threshold: i32,          // NextInt(0,num_bin-2) draw; -1 when use_rand off
    min_data_in_leaf: i32,
    min_sum_hessian_in_leaf: f64,
    lambda_l1: f64,
    lambda_l2: f64,
    path_smooth: f64,
    parent_output: f64,
    min_gain_shift: f64,
    sum_gradient: f64,
    sum_hessian: f64,
    num_data: i32,
    scan_count: i32, // host-computed = num_bin (covers every candidate threshold)
) {
    let eps = f64::cast_from(K_EPSILON);
    let l1 = lambda_l1;
    let l2 = lambda_l2;
    let rev = reverse != 0;
    let use_l1_b = use_l1 != 0;
    let sm = use_smoothing != 0;
    let use_rand_b = use_rand != 0;
    let skip_def = skip_default_bin != 0;

    // cnt_factor = num_data / sum_hessians (CUDA leaf-total division, cu:146). Note:
    // sum_hessian is the RAW leaf total (NOT kEpsilon-bumped, unlike host split.rs).
    let cnt_factor = f64::cast_from(num_data) / sum_hessian;
    let fnbmo = num_bin - mfb_offset; // feature_num_bin_minus_offset
    let fwd_end = fnbmo - 2; // forward candidate upper bound
    let rev_end = num_bin - 2; // reverse candidate upper bound

    // Serial inclusive prefix accumulators (LITERAL init — MLIR constraint #1).
    let mut acc_g = 0.0f64;
    let mut acc_h = 0.0f64;
    // Winner state (LITERAL init). best_gain sentinel 0.0: every VALID local_gain is
    // strictly > 0 (`current_gain > min_gain_shift`), so 0.0 rejects no valid candidate
    // and "no split" is signalled by is_valid == 0.0 (as the launcher decodes).
    let mut best_gain = 0.0f64;
    let mut best_threshold = 0i32;
    let mut best_scanned_g = 0.0f64;
    let mut best_scanned_h = 0.0f64;
    let mut is_valid = 0.0f64;

    for t in 0..scan_count {
        // ---- read this thread's bin (forward: bin t; reverse: bin fnbmo-1-t) ----
        // skip_sum excludes the default bin from the accumulation (a `continue`).
        let skip = skip_def
            && ((rev && (num_bin - 1 - t) == default_bin) || (!rev && (t + mfb_offset) == default_bin));
        // Only threads whose bin exists (t < fnbmo) and are not skipped load a value.
        let read_active = t < fnbmo && !skip;
        let bin_fwd = t;
        let bin_rev = fnbmo - 1 - t;
        let bin = select(rev, bin_rev, bin_fwd);
        // Branchless clamp: an inactive thread reads bin 0 inertly (contributes 0).
        let bin_safe = select(read_active, bin, 0i32);
        let base = (bin_safe as usize) * 2;
        let g = select(read_active, hist[base], 0.0);
        let h_raw = select(read_active, hist[base + 1], 0.0);
        // thread 0 seeds kEpsilon ONCE at the scan origin (cu:206, Pitfall 2).
        let h = h_raw + select(t == 0, eps, 0.0);

        // ---- serial inclusive prefix (the single-owner ShufflePrefixSum analog) ----
        acc_g += g;
        acc_h += h;

        // ---- guard phase: scanned side = acc, complement = parent - scanned ----
        let scanned_g = acc_g;
        let scanned_h = acc_h; // kEpsilon-INCLUDED (guard-phase recovery, Pitfall 2)
        let comp_g = sum_gradient - scanned_g;
        let comp_h = sum_hessian - scanned_h;
        let scanned_cnt = round_ties_even_cube(scanned_h * cnt_factor);
        let comp_cnt = num_data - scanned_cnt;

        // forward: left = scanned, right = complement; reverse: swapped.
        let l_g = select(rev, comp_g, scanned_g);
        let l_h = select(rev, comp_h, scanned_h);
        let lc = select(rev, comp_cnt, scanned_cnt);
        let r_g = select(rev, scanned_g, comp_g);
        let r_h = select(rev, scanned_h, comp_h);
        let rc = select(rev, scanned_cnt, comp_cnt);
        // reverse records `num_bin-2-t`; forward records `t + mfb_offset` (cu:230/318).
        let threshold = select(rev, rev_end - t, t + mfb_offset);

        // candidate range: forward t<=fnbmo-2, reverse t<=num_bin-2; not skipped.
        let in_range = (rev && t <= rev_end) || (!rev && t <= fwd_end);
        let cand = in_range && !skip;

        // guards (fixed order, cu:216-219 / 240-243) + USE_RAND single-threshold gate.
        let guard = l_h >= min_sum_hessian_in_leaf
            && lc >= min_data_in_leaf
            && r_h >= min_sum_hessian_in_leaf
            && rc >= min_data_in_leaf;
        let rand_ok = !use_rand_b || threshold == rand_threshold;

        // gain: dispatch on the runtime use_smoothing flag (both computed, selected).
        let gain_ns = get_split_gains(use_l1_b, l_g, l_h, r_g, r_h, l1, l2);
        let gain_sm = get_leaf_gain_smoothed(use_l1_b, l_g, l_h, l1, l2, path_smooth, lc, parent_output)
            + get_leaf_gain_smoothed(use_l1_b, r_g, r_h, l1, l2, path_smooth, rc, parent_output);
        let current_gain = select(sm, gain_sm, gain_ns);

        let valid = cand && guard && rand_ok && current_gain > min_gain_shift;
        let local_gain = current_gain - min_gain_shift;
        // strict `>` keeps the FIRST (lowest index) winner on a tie (Pitfall 5).
        let take = valid && local_gain > best_gain;
        best_gain = select(take, local_gain, best_gain);
        best_threshold = select(take, threshold, best_threshold);
        best_scanned_g = select(take, scanned_g, best_scanned_g);
        best_scanned_h = select(take, scanned_h, best_scanned_h);
        is_valid = select(take, 1.0, is_valid);
    }

    // ---- write phase (winning bin): kEpsilon SUBTRACTED, count RE-recovered ----
    // Reconstruct the scanned side from the winner's prefix, then complement.
    let w_scanned_g = best_scanned_g;
    let w_scanned_h = best_scanned_h - eps; // kEpsilon removed for the RECORD (cu:275/298)
    let w_scanned_cnt = round_ties_even_cube(w_scanned_h * cnt_factor);
    let w_comp_g = sum_gradient - w_scanned_g;
    let w_comp_h = sum_hessian - w_scanned_h - eps; // the second kEpsilon subtraction
    let w_comp_cnt = num_data - w_scanned_cnt;

    let wl_g = select(rev, w_comp_g, w_scanned_g);
    let wl_h = select(rev, w_comp_h, w_scanned_h);
    let wl_c = select(rev, w_comp_cnt, w_scanned_cnt);
    let wr_g = select(rev, w_scanned_g, w_comp_g);
    let wr_h = select(rev, w_scanned_h, w_comp_h);
    let wr_c = select(rev, w_scanned_cnt, w_comp_cnt);

    // per-side output (value) with smoothing dispatch; per-side gain via given-output.
    let l_out_ns = calculate_splitted_leaf_output(use_l1_b, wl_g, wl_h, l1, l2);
    let l_out_sm =
        calculate_splitted_leaf_output_smoothed(use_l1_b, wl_g, wl_h, l1, l2, path_smooth, wl_c, parent_output);
    let left_output = select(sm, l_out_sm, l_out_ns);
    let r_out_ns = calculate_splitted_leaf_output(use_l1_b, wr_g, wr_h, l1, l2);
    let r_out_sm =
        calculate_splitted_leaf_output_smoothed(use_l1_b, wr_g, wr_h, l1, l2, path_smooth, wr_c, parent_output);
    let right_output = select(sm, r_out_sm, r_out_ns);
    let left_gain = get_leaf_gain_given_output(use_l1_b, wl_g, wl_h, l1, l2, left_output);
    let right_gain = get_leaf_gain_given_output(use_l1_b, wr_g, wr_h, l1, l2, right_output);

    let default_left_f = select(assume_out_default_left != 0, 1.0, 0.0);

    out[0] = is_valid;
    out[1] = f64::cast_from(best_threshold);
    out[2] = default_left_f;
    out[3] = best_gain;
    out[4] = wl_g;
    out[5] = wl_h;
    out[6] = f64::cast_from(wl_c);
    out[7] = wr_g;
    out[8] = wr_h;
    out[9] = f64::cast_from(wr_c);
    out[10] = left_output;
    out[11] = right_output;
    out[12] = left_gain;
    out[13] = right_gain;
}

/// The f64 stage-1 launch wrapper (single-owner) — a thin `#[cube(launch)]` shell
/// delegating to [`split_eval_body`] (mirrors `split.rs:393` `find_best_split_kernel`).
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
pub fn split_eval_kernel_f64(
    hist: &Array<f64>,
    out: &mut Array<f64>,
    num_bin: i32,
    mfb_offset: i32,
    default_bin: i32,
    skip_default_bin: u32,
    reverse: u32,
    assume_out_default_left: u32,
    use_l1: u32,
    use_smoothing: u32,
    use_rand: u32,
    rand_threshold: i32,
    min_data_in_leaf: i32,
    min_sum_hessian_in_leaf: f64,
    lambda_l1: f64,
    lambda_l2: f64,
    path_smooth: f64,
    parent_output: f64,
    min_gain_shift: f64,
    sum_gradient: f64,
    sum_hessian: f64,
    num_data: i32,
    scan_count: i32,
) {
    split_eval_body(
        hist,
        out,
        num_bin,
        mfb_offset,
        default_bin,
        skip_default_bin,
        reverse,
        assume_out_default_left,
        use_l1,
        use_smoothing,
        use_rand,
        rand_threshold,
        min_data_in_leaf,
        min_sum_hessian_in_leaf,
        lambda_l1,
        lambda_l2,
        path_smooth,
        parent_output,
        min_gain_shift,
        sum_gradient,
        sum_hessian,
        num_data,
        scan_count,
    );
}

/// V5 launch-boundary validation (threat T-17-01/T-17-02) — reject the host scalars
/// that would drive an out-of-bounds `launch_unchecked` BEFORE the launch (mirrors
/// `split.rs::find_best_split_f64_on`'s pre-launch checks / `primitives.rs`
/// `validate_scan_inputs`). Returns the required `2*num_bin` histogram length.
///
/// # Errors
/// [`ComputeError::Runtime`] if `num_bin == 0` or `2*num_bin` overflows `usize`;
/// [`ComputeError::LengthMismatch`] if `hist.len() != 2*num_bin`.
pub fn validate_stage1_inputs(num_bin: u32, hist_len: usize) -> Result<usize, ComputeError> {
    if num_bin == 0 {
        return Err(ComputeError::Runtime {
            detail: "stage1: num_bin must be > 0".to_string(),
        });
    }
    let expected = 2usize
        .checked_mul(num_bin as usize)
        .ok_or_else(|| ComputeError::Runtime {
            detail: format!("stage1: num_bin {num_bin} overflows the histogram length"),
        })?;
    if hist_len != expected {
        return Err(ComputeError::LengthMismatch {
            expected,
            actual: hist_len,
        });
    }
    Ok(expected)
}

/// STAGE 1 — per-`(leaf,feature)` split evaluation (ODL-11, §8.1). Drives
/// [`split_eval_body`] single-owner (`CubeDim(1)`) as the cpu f64 fold anchor and
/// decodes the winning [`SplitScalars`] `CUDASplitInfo` record. `hist` is the
/// interleaved `[g0,h0,g1,h1,…]` f64 histogram for the task's feature (already
/// offset). Generic over `R` so the SAME body runs on cubecl-cpu (the anchor) and
/// cubecl-hip (the f32 mirror is the separate [`split_eval_kernel_f32`] path, 17-03
/// Task 2, anchored to THIS fold — never GPU-vs-GPU, def-f8u-01).
///
/// The categorical task is the Phase-22 dispatch seam (D-04): a categorical task
/// returns the `is_valid=false` sentinel WITHOUT running the numerical core (no
/// `BitonicArgSort`/`cat_threshold` eval).
///
/// # Errors
/// [`ComputeError`] from [`validate_stage1_inputs`] (bad `num_bin` / histogram length)
/// or the USE_RAND draw.
pub fn find_best_splits_stage1_on<R: cubecl::Runtime>(
    client: &ComputeClient<R>,
    hist: &[f64],
    task: &SplitFindTask,
    scalars: &Stage1Scalars,
) -> Result<SplitScalars, ComputeError> {
    validate_stage1_inputs(task.num_bin, hist.len())?;

    // Phase-22 categorical dispatch seam (D-04): the numerical core is continuous-only.
    if task.is_categorical {
        return Ok(SplitScalars::default());
    }

    let num_bin_i = task.num_bin as i32;
    // USE_RAND: draw rand_threshold = CUDARandom.NextInt(0, num_bin-2) seeded
    // `extra_seed + task_index` (Open Q1, carried in scalars.rng_seed). NextInt uses
    // RandInt32 (cuda_random.hpp:42-44); route through the Phase-14 `random.rs` LCG so
    // the draw is bit-identical to the verified device stream (key_link → random.rs).
    let rand_threshold: i32 = if scalars.use_rand && num_bin_i - 2 > 0 {
        let draw = draw_rand_int32_on(client, &[scalars.rng_seed as u32], 1)?;
        draw[0] % (num_bin_i - 2)
    } else {
        -1
    };

    let min_gain_shift = scalars.parent_gain + scalars.min_gain_to_split;

    let h_hist = client.create_from_slice(f64::as_bytes(hist));
    // The kernel WRITES all STAGE1_OUT_LEN cells unconditionally (single owner), so
    // `empty` needs no zero-init (the `split.rs` O1 idiom).
    let h_out = client.empty(STAGE1_OUT_LEN * core::mem::size_of::<f64>());

    // SAFETY: `h_hist` is sized exactly `hist.len() == 2*num_bin` (host-validated by
    // `validate_stage1_inputs`) and `h_out` is `STAGE1_OUT_LEN` cells; both outlive the
    // launch. The single-owner scan reads `hist[bin*2 (+1)]` with `bin` clamped to
    // `[0, fnbmo)` (`bin_safe`), so every index stays in `[0, 2*num_bin)`. cubecl unsafe
    // confined here (CMP-01, T-17-01).
    unsafe {
        split_eval_kernel_f64::launch(
            client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new_1d(1),
            ArrayArg::from_raw_parts(h_hist, hist.len()),
            ArrayArg::from_raw_parts(h_out.clone(), STAGE1_OUT_LEN),
            num_bin_i,
            task.mfb_offset as i32,
            task.default_bin as i32,
            if task.skip_default_bin { 1u32 } else { 0u32 },
            if task.reverse { 1u32 } else { 0u32 },
            if task.assume_out_default_left { 1u32 } else { 0u32 },
            if scalars.use_l1 { 1u32 } else { 0u32 },
            if scalars.use_smoothing { 1u32 } else { 0u32 },
            if scalars.use_rand { 1u32 } else { 0u32 },
            rand_threshold,
            scalars.min_data_in_leaf,
            scalars.min_sum_hessian_in_leaf,
            scalars.lambda_l1,
            scalars.lambda_l2,
            scalars.path_smooth,
            scalars.parent_output,
            min_gain_shift,
            scalars.sum_gradient,
            scalars.sum_hessian,
            scalars.num_data,
            num_bin_i,
        );
    }

    let bytes = client.read_one_unchecked(h_out);
    let cells = f64::from_bytes(&bytes);

    let is_valid = cells[0] != 0.0;
    if !is_valid {
        return Ok(SplitScalars::default());
    }
    Ok(SplitScalars {
        is_valid: true,
        leaf_index: -1,
        gain: cells[3],
        inner_feature_index: task.inner_feature_index,
        threshold: cells[1] as u32,
        default_left: cells[2] != 0.0,
        left_sum_gradients: cells[4],
        left_sum_hessians: cells[5],
        left_sum_gh_quant: 0,
        left_count: cells[6] as i32,
        left_gain: cells[12],
        left_value: cells[10],
        right_sum_gradients: cells[7],
        right_sum_hessians: cells[8],
        right_sum_gh_quant: 0,
        right_count: cells[9] as i32,
        right_gain: cells[13],
        right_value: cells[11],
        num_cat_threshold: 0,
    })
}

// ===========================================================================
// hip f32 mirror (17-03 Task 2) — the f32 numerical core.
//
// TWO paths, mirroring the codebase convention (Phases 14-16): the cpu-testable
// SINGLE-OWNER f32 fold ([`split_eval_kernel_f32`]) that drives through cubecl-cpu
// (which has NO plane support — primitives.rs:1182) so the f32 numerics are anchored
// to the Task-1 f64 fold (structure bit-exact, values within the ~1e-5 f32 envelope,
// def-f8u-01 — NEVER GPU-vs-GPU); AND the block-parallel hip path
// ([`split_eval_block_kernel_f32`]) built on the NET-NEW two-level LDS scan
// ([`stage1_block_scan`], D-03) + [`reduce_best_gain`] block argmax, `#[cfg(feature =
// "gpu")]` like every rocm kernel in `histogram.rs`/`primitives.rs`. The block path's
// on-device rocm parity assertion is added in 17-05 (this task ships + compiles it).
// NO f64 anywhere in the f32 path (D-10, WR-05 literal pinning throughout).
// ===========================================================================

/// f32 mirror of [`round_ties_even_cube`] (`__double2int_rn`, round-ties-EVEN) — the
/// no-f64 hip path. Uses `f32::floor` only; every literal pinned f32 (WR-05).
#[cube]
fn round_ties_even_f32_cube(x: f32) -> i32 {
    let f = f32::floor(x);
    let diff = x - f;
    let parity = f - 2.0f32 * f32::floor(f * 0.5f32);
    let f_is_odd = parity > 0.5f32;
    let up = diff > 0.5f32 || (diff == 0.5f32 && f_is_odd);
    let r = select(up, f + 1.0f32, f);
    i32::cast_from(r)
}

/// The f32 single-owner mirror of [`split_eval_body`] — the SAME §8.1 accumulation in
/// f32 (all literals pinned f32, WR-05; the `*_f32` gain mirrors), driven single-owner
/// so it runs on the cubecl-cpu anchor. Anchored to the Task-1 f64 fold within ~1e-5
/// (def-f8u-01). Structure (threshold/counts/default_left/is_valid) is bit-exact; the
/// per-side sums/value/gain absorb the f32-vs-f64 accumulation gap (~1e-5).
#[cube]
#[allow(clippy::too_many_arguments)]
pub fn split_eval_body_f32(
    hist: &Array<f32>,
    out: &mut Array<f32>,
    num_bin: i32,
    mfb_offset: i32,
    default_bin: i32,
    skip_default_bin: u32,
    reverse: u32,
    assume_out_default_left: u32,
    use_l1: u32,
    use_smoothing: u32,
    use_rand: u32,
    rand_threshold: i32,
    min_data_in_leaf: i32,
    min_sum_hessian_in_leaf: f32,
    lambda_l1: f32,
    lambda_l2: f32,
    path_smooth: f32,
    parent_output: f32,
    min_gain_shift: f32,
    sum_gradient: f32,
    sum_hessian: f32,
    num_data: i32,
    scan_count: i32,
) {
    let eps = f32::cast_from(K_EPSILON);
    let l1 = lambda_l1;
    let l2 = lambda_l2;
    let rev = reverse != 0;
    let use_l1_b = use_l1 != 0;
    let sm = use_smoothing != 0;
    let use_rand_b = use_rand != 0;
    let skip_def = skip_default_bin != 0;

    let cnt_factor = f32::cast_from(num_data) / sum_hessian;
    let fnbmo = num_bin - mfb_offset;
    let fwd_end = fnbmo - 2;
    let rev_end = num_bin - 2;

    let mut acc_g = 0.0f32;
    let mut acc_h = 0.0f32;
    let mut best_gain = 0.0f32;
    let mut best_threshold = 0i32;
    let mut best_scanned_g = 0.0f32;
    let mut best_scanned_h = 0.0f32;
    let mut is_valid = 0.0f32;

    for t in 0..scan_count {
        let skip = skip_def
            && ((rev && (num_bin - 1 - t) == default_bin) || (!rev && (t + mfb_offset) == default_bin));
        let read_active = t < fnbmo && !skip;
        let bin = select(rev, fnbmo - 1 - t, t);
        let bin_safe = select(read_active, bin, 0i32);
        let base = (bin_safe as usize) * 2;
        let g = select(read_active, hist[base], 0.0f32);
        let h_raw = select(read_active, hist[base + 1], 0.0f32);
        let h = h_raw + select(t == 0, eps, 0.0f32);

        acc_g += g;
        acc_h += h;

        let scanned_g = acc_g;
        let scanned_h = acc_h;
        let comp_g = sum_gradient - scanned_g;
        let comp_h = sum_hessian - scanned_h;
        let scanned_cnt = round_ties_even_f32_cube(scanned_h * cnt_factor);
        let comp_cnt = num_data - scanned_cnt;

        let l_g = select(rev, comp_g, scanned_g);
        let l_h = select(rev, comp_h, scanned_h);
        let lc = select(rev, comp_cnt, scanned_cnt);
        let r_g = select(rev, scanned_g, comp_g);
        let r_h = select(rev, scanned_h, comp_h);
        let rc = select(rev, scanned_cnt, comp_cnt);
        let threshold = select(rev, rev_end - t, t + mfb_offset);

        let in_range = (rev && t <= rev_end) || (!rev && t <= fwd_end);
        let cand = in_range && !skip;
        let guard = l_h >= min_sum_hessian_in_leaf
            && lc >= min_data_in_leaf
            && r_h >= min_sum_hessian_in_leaf
            && rc >= min_data_in_leaf;
        let rand_ok = !use_rand_b || threshold == rand_threshold;

        let gain_ns = get_split_gains_f32(use_l1_b, l_g, l_h, r_g, r_h, l1, l2);
        let gain_sm = get_leaf_gain_smoothed_f32(use_l1_b, l_g, l_h, l1, l2, path_smooth, lc, parent_output)
            + get_leaf_gain_smoothed_f32(use_l1_b, r_g, r_h, l1, l2, path_smooth, rc, parent_output);
        let current_gain = select(sm, gain_sm, gain_ns);

        let valid = cand && guard && rand_ok && current_gain > min_gain_shift;
        let local_gain = current_gain - min_gain_shift;
        let take = valid && local_gain > best_gain;
        best_gain = select(take, local_gain, best_gain);
        best_threshold = select(take, threshold, best_threshold);
        best_scanned_g = select(take, scanned_g, best_scanned_g);
        best_scanned_h = select(take, scanned_h, best_scanned_h);
        is_valid = select(take, 1.0f32, is_valid);
    }

    let w_scanned_g = best_scanned_g;
    let w_scanned_h = best_scanned_h - eps;
    let w_scanned_cnt = round_ties_even_f32_cube(w_scanned_h * cnt_factor);
    let w_comp_g = sum_gradient - w_scanned_g;
    let w_comp_h = sum_hessian - w_scanned_h - eps;
    let w_comp_cnt = num_data - w_scanned_cnt;

    let wl_g = select(rev, w_comp_g, w_scanned_g);
    let wl_h = select(rev, w_comp_h, w_scanned_h);
    let wl_c = select(rev, w_comp_cnt, w_scanned_cnt);
    let wr_g = select(rev, w_scanned_g, w_comp_g);
    let wr_h = select(rev, w_scanned_h, w_comp_h);
    let wr_c = select(rev, w_scanned_cnt, w_comp_cnt);

    let l_out_ns = calculate_splitted_leaf_output_f32(use_l1_b, wl_g, wl_h, l1, l2);
    let l_out_sm =
        calculate_splitted_leaf_output_smoothed_f32(use_l1_b, wl_g, wl_h, l1, l2, path_smooth, wl_c, parent_output);
    let left_output = select(sm, l_out_sm, l_out_ns);
    let r_out_ns = calculate_splitted_leaf_output_f32(use_l1_b, wr_g, wr_h, l1, l2);
    let r_out_sm =
        calculate_splitted_leaf_output_smoothed_f32(use_l1_b, wr_g, wr_h, l1, l2, path_smooth, wr_c, parent_output);
    let right_output = select(sm, r_out_sm, r_out_ns);
    let left_gain = get_leaf_gain_given_output_f32(use_l1_b, wl_g, wl_h, l1, l2, left_output);
    let right_gain = get_leaf_gain_given_output_f32(use_l1_b, wr_g, wr_h, l1, l2, right_output);

    let default_left_f = select(assume_out_default_left != 0, 1.0f32, 0.0f32);

    out[0] = is_valid;
    out[1] = f32::cast_from(best_threshold);
    out[2] = default_left_f;
    out[3] = best_gain;
    out[4] = wl_g;
    out[5] = wl_h;
    out[6] = f32::cast_from(wl_c);
    out[7] = wr_g;
    out[8] = wr_h;
    out[9] = f32::cast_from(wr_c);
    out[10] = left_output;
    out[11] = right_output;
    out[12] = left_gain;
    out[13] = right_gain;
}

/// The f32 single-owner launch wrapper (mirrors `split.rs:444` `find_best_split_kernel_f32`,
/// f32 cell type). Delegates to [`split_eval_body_f32`].
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
pub fn split_eval_kernel_f32(
    hist: &Array<f32>,
    out: &mut Array<f32>,
    num_bin: i32,
    mfb_offset: i32,
    default_bin: i32,
    skip_default_bin: u32,
    reverse: u32,
    assume_out_default_left: u32,
    use_l1: u32,
    use_smoothing: u32,
    use_rand: u32,
    rand_threshold: i32,
    min_data_in_leaf: i32,
    min_sum_hessian_in_leaf: f32,
    lambda_l1: f32,
    lambda_l2: f32,
    path_smooth: f32,
    parent_output: f32,
    min_gain_shift: f32,
    sum_gradient: f32,
    sum_hessian: f32,
    num_data: i32,
    scan_count: i32,
) {
    split_eval_body_f32(
        hist, out, num_bin, mfb_offset, default_bin, skip_default_bin, reverse,
        assume_out_default_left, use_l1, use_smoothing, use_rand, rand_threshold,
        min_data_in_leaf, min_sum_hessian_in_leaf, lambda_l1, lambda_l2, path_smooth,
        parent_output, min_gain_shift, sum_gradient, sum_hessian, num_data, scan_count,
    );
}

/// STAGE 1 f32 mirror launcher (single-owner) — drives [`split_eval_kernel_f32`] on
/// the cubecl-cpu anchor and decodes the [`SplitScalars`] record (f32 widened to the
/// f64 storage fields). Anchored to [`find_best_splits_stage1_on`] within ~1e-5
/// (def-f8u-01); the on-device rocm f32 path is [`split_eval_block_kernel_f32`] (17-05).
///
/// # Errors
/// [`ComputeError`] from [`validate_stage1_inputs`] or the USE_RAND draw.
pub fn find_best_splits_stage1_f32_on<R: cubecl::Runtime>(
    client: &ComputeClient<R>,
    hist: &[f32],
    task: &SplitFindTask,
    scalars: &Stage1Scalars,
) -> Result<SplitScalars, ComputeError> {
    validate_stage1_inputs(task.num_bin, hist.len())?;
    if task.is_categorical {
        return Ok(SplitScalars::default());
    }
    let num_bin_i = task.num_bin as i32;
    let rand_threshold: i32 = if scalars.use_rand && num_bin_i - 2 > 0 {
        let draw = draw_rand_int32_on(client, &[scalars.rng_seed as u32], 1)?;
        draw[0] % (num_bin_i - 2)
    } else {
        -1
    };
    let min_gain_shift = (scalars.parent_gain + scalars.min_gain_to_split) as f32;

    let h_hist = client.create_from_slice(f32::as_bytes(hist));
    let h_out = client.empty(STAGE1_OUT_LEN * core::mem::size_of::<f32>());

    // SAFETY: identical sizing/index contract as `find_best_splits_stage1_on` (T-17-01),
    // f32 cells. cubecl unsafe confined here (CMP-01).
    unsafe {
        split_eval_kernel_f32::launch(
            client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new_1d(1),
            ArrayArg::from_raw_parts(h_hist, hist.len()),
            ArrayArg::from_raw_parts(h_out.clone(), STAGE1_OUT_LEN),
            num_bin_i,
            task.mfb_offset as i32,
            task.default_bin as i32,
            if task.skip_default_bin { 1u32 } else { 0u32 },
            if task.reverse { 1u32 } else { 0u32 },
            if task.assume_out_default_left { 1u32 } else { 0u32 },
            if scalars.use_l1 { 1u32 } else { 0u32 },
            if scalars.use_smoothing { 1u32 } else { 0u32 },
            if scalars.use_rand { 1u32 } else { 0u32 },
            rand_threshold,
            scalars.min_data_in_leaf,
            scalars.min_sum_hessian_in_leaf as f32,
            scalars.lambda_l1 as f32,
            scalars.lambda_l2 as f32,
            scalars.path_smooth as f32,
            scalars.parent_output as f32,
            min_gain_shift,
            scalars.sum_gradient as f32,
            scalars.sum_hessian as f32,
            scalars.num_data,
            num_bin_i,
        );
    }
    let bytes = client.read_one_unchecked(h_out);
    let cells = f32::from_bytes(&bytes);
    if cells[0] == 0.0 {
        return Ok(SplitScalars::default());
    }
    Ok(SplitScalars {
        is_valid: true,
        leaf_index: -1,
        gain: cells[3] as f64,
        inner_feature_index: task.inner_feature_index,
        threshold: cells[1] as u32,
        default_left: cells[2] != 0.0,
        left_sum_gradients: cells[4] as f64,
        left_sum_hessians: cells[5] as f64,
        left_sum_gh_quant: 0,
        left_count: cells[6] as i32,
        left_gain: cells[12] as f64,
        left_value: cells[10] as f64,
        right_sum_gradients: cells[7] as f64,
        right_sum_hessians: cells[8] as f64,
        right_sum_gh_quant: 0,
        right_count: cells[9] as i32,
        right_gain: cells[13] as f64,
        right_value: cells[11] as f64,
        num_cat_threshold: 0,
    })
}

// ===========================================================================
// _GlobalMemory stage-1 spill variant (17-04, D-05/D-11) — the >256-bin path.
//
// For features whose bin count exceeds the block's thread count, the C++
// `FindBestSplitsForLeafKernel_GlobalMemory` (`cuda_best_split_finder.cu:1051-1273`)
// spills the per-bin scanned sums into a PRE-ALLOCATED global-memory scratch slab
// and scans it with `GlobalMemoryPrefixSum` (a chunked two-level in-place scan,
// `cuda_algorithms.hpp:169-185`) over STRIDED thread loops (each thread owns bins
// `t, t+blockDim, t+2·blockDim, …`). The gain / count-recovery / guard / argmax
// semantics are IDENTICAL to the in-block two-level path (17-03) — only the scan
// carrier (global scratch vs LDS) and the strided iteration differ (A4).
//
// The cpu single-owner f64 fold ([`split_eval_body`]) needs NO separate >256
// implementation: its serial body has no register/LDS cap, so a `num_bin=300`
// feature is handled by a larger loop bound — that is why the `globalmem_spill`
// golden (num_bin=300) already passes bit-exact on the cpu fold. This section adds
// the NET-NEW hip strided f32 kernel (gpu-gated, compile-verified; its on-device
// rocm parity assertion lands in 17-05, the same deferral as the block kernel) plus
// the alloc-once scratch (D-11) and the launch-boundary size validation (V5).
// ===========================================================================

/// The stage-1 block thread width (256 = the C++
/// `NUM_THREADS_PER_BLOCK_BEST_SPLIT_FINDER`). The dispatch boundary: a feature with
/// `num_bin` bins beyond this width spills to the [`split_eval_globalmem_kernel_f32`]
/// strided path; at or below it uses the 17-03 in-block two-level scan. Ungated so the
/// cpu-side dispatch decision ([`stage1_needs_globalmem`]) and validation are testable
/// without the `gpu` feature.
pub const STAGE1_BLOCK_THREADS: usize = 256;

/// The number of pre-allocated global-memory scratch buffers the `_GlobalMemory`
/// spill path reserves ONCE in [`Stage1GlobalMemScratch::new`] (D-11): the grad / hess
/// scan carriers plus the `stat` / `index` buffers reserved for the discretized &
/// categorical `_GlobalMemory` variants (C++ TODO / Phase-22 seam — reserved, not used
/// by the continuous kernel, mirroring how `DeviceSplitInfo` reserves its categorical
/// slabs). Used by the "allocated exactly once" counter assertion.
pub const NUM_STAGE1_SCRATCH_BUFFERS: usize = 4;

/// Stage-1 spill dispatch (D-05): does this feature's bin count exceed the block
/// thread width, requiring the `_GlobalMemory` strided path? `num_bin > block_threads`
/// spills; at or below stays on the in-block two-level scan (17-03). The cpu
/// single-owner fold ([`find_best_splits_stage1_on`]) handles BOTH via the same serial
/// body (A4), so this only routes the gpu block/globalmem kernels (wired on-device in
/// 17-05).
#[must_use]
pub fn stage1_needs_globalmem(num_bin: u32, block_threads: usize) -> bool {
    (num_bin as usize) > block_threads
}

/// V5 launch-boundary validation for the `_GlobalMemory` scratch slab (threat
/// T-17-01/T-17-02): reject a `largest_feature_bin_count × num_concurrent_blocks`
/// product that would overflow `usize` BEFORE any `client.empty` / strided
/// `launch_unchecked`. Returns the validated per-buffer slab length (in elements).
///
/// Mirrors `split_info.rs`'s `checked_mul` categorical-slab guard and `random.rs`'s
/// `validate_draw_inputs` overflow guard.
///
/// # Errors
/// [`ComputeError::Runtime`] if `largest_feature_bin_count == 0`,
/// `num_concurrent_blocks == 0`, or the slab length
/// `largest_feature_bin_count * num_concurrent_blocks` overflows `usize`.
pub fn validate_globalmem_scratch(
    largest_feature_bin_count: usize,
    num_concurrent_blocks: usize,
) -> Result<usize, ComputeError> {
    if largest_feature_bin_count == 0 {
        return Err(ComputeError::Runtime {
            detail: "globalmem scratch: largest_feature_bin_count must be > 0".to_string(),
        });
    }
    if num_concurrent_blocks == 0 {
        return Err(ComputeError::Runtime {
            detail: "globalmem scratch: num_concurrent_blocks must be > 0".to_string(),
        });
    }
    largest_feature_bin_count
        .checked_mul(num_concurrent_blocks)
        .ok_or_else(|| ComputeError::Runtime {
            detail: format!(
                "globalmem scratch: slab length {largest_feature_bin_count} * \
                 {num_concurrent_blocks} overflows usize"
            ),
        })
}

/// The pre-allocated `_GlobalMemory` stage-1 scratch (D-11) — the grad / hess scan
/// carriers plus the reserved `stat` / `index` buffers, each a CubeCL [`Handle`]
/// allocated **once** in [`Self::new`] via the counted `alloc` closure (the
/// `DeviceSplitInfo::new` idiom, `split_info.rs:289-293`). There is NO per-split /
/// in-kernel device allocation anywhere: the strided kernel indexes these reserved
/// handles, it never calls `client.empty`.
///
/// Each buffer is sized `largest_feature_bin_count * num_concurrent_blocks` so every
/// concurrently-launched spill block owns a disjoint `largest_feature_bin_count`-wide
/// scan region.
pub struct Stage1GlobalMemScratch<R: cubecl::Runtime> {
    /// `feature_hist_grad_buffer` — the strided gradient scan carrier (f32).
    pub feature_hist_grad_buffer: Handle,
    /// `feature_hist_hess_buffer` — the strided hessian scan carrier (f32).
    pub feature_hist_hess_buffer: Handle,
    /// `feature_hist_stat_buffer` — reserved for the discretized `_GlobalMemory`
    /// variant (C++ TODO / v2 QGD-02); allocated but unused by the continuous kernel.
    pub feature_hist_stat_buffer: Handle,
    /// `feature_hist_index_buffer` — reserved for the categorical `_GlobalMemory`
    /// variant (Phase-22 seam); allocated but unused by the continuous kernel.
    pub feature_hist_index_buffer: Handle,
    /// The largest per-feature bin count the scratch is sized for.
    largest_feature_bin_count: usize,
    /// The number of concurrently-launched spill blocks the scratch is sized for.
    num_concurrent_blocks: usize,
    /// The per-buffer slab length in elements
    /// (`largest_feature_bin_count * num_concurrent_blocks`).
    slab_len: usize,
    /// Count of `client.empty` allocations — equals [`NUM_STAGE1_SCRATCH_BUFFERS`]
    /// after [`Self::new`] and never changes (proves "allocated exactly once", D-11).
    device_allocations: usize,
    _runtime: PhantomData<R>,
}

impl<R: cubecl::Runtime> Stage1GlobalMemScratch<R> {
    /// Pre-allocate the four `_GlobalMemory` scratch buffers — **one `client.empty`
    /// per buffer, exactly once** (D-11). No allocation happens anywhere else (no
    /// per-split / in-kernel device alloc). The slab length is V5-validated
    /// ([`validate_globalmem_scratch`]) for overflow before any allocation.
    ///
    /// # Errors
    /// [`ComputeError::Runtime`] from [`validate_globalmem_scratch`] (zero or
    /// overflowing `largest_feature_bin_count * num_concurrent_blocks`).
    pub fn new(
        client: &ComputeClient<R>,
        largest_feature_bin_count: usize,
        num_concurrent_blocks: usize,
    ) -> Result<Self, ComputeError> {
        let slab_len = validate_globalmem_scratch(largest_feature_bin_count, num_concurrent_blocks)?;

        // The counted alloc closure is the ONLY caller of `client.empty` in the spill
        // path, and it runs only here in `new` — so `device_allocations` structurally
        // proves the alloc-once invariant (D-11), exactly like `DeviceSplitInfo::new`.
        let mut device_allocations = 0usize;
        let mut alloc = |elem_size: usize| -> Handle {
            device_allocations += 1;
            client.empty(slab_len * elem_size)
        };
        let feature_hist_grad_buffer = alloc(core::mem::size_of::<f32>());
        let feature_hist_hess_buffer = alloc(core::mem::size_of::<f32>());
        let feature_hist_stat_buffer = alloc(core::mem::size_of::<f32>());
        let feature_hist_index_buffer = alloc(core::mem::size_of::<i32>());

        Ok(Stage1GlobalMemScratch {
            feature_hist_grad_buffer,
            feature_hist_hess_buffer,
            feature_hist_stat_buffer,
            feature_hist_index_buffer,
            largest_feature_bin_count,
            num_concurrent_blocks,
            slab_len,
            device_allocations,
            _runtime: PhantomData,
        })
    }

    /// The largest per-feature bin count this scratch is sized for.
    #[must_use]
    pub fn largest_feature_bin_count(&self) -> usize {
        self.largest_feature_bin_count
    }

    /// The number of concurrent spill blocks this scratch is sized for.
    #[must_use]
    pub fn num_concurrent_blocks(&self) -> usize {
        self.num_concurrent_blocks
    }

    /// The per-buffer slab length in elements.
    #[must_use]
    pub fn slab_len(&self) -> usize {
        self.slab_len
    }

    /// The number of device buffers allocated — equals [`NUM_STAGE1_SCRATCH_BUFFERS`]
    /// after [`Self::new`] and never changes (proves the D-11 alloc-once invariant: no
    /// per-split / in-kernel alloc).
    #[must_use]
    pub fn device_allocations(&self) -> usize {
        self.device_allocations
    }
}

/// Max block width for the hip block-parallel stage-1 path (256 threads, the C++
/// `NUM_THREADS_PER_BLOCK_BEST_SPLIT_FINDER`). Also the `SharedMemory` staging cap.
#[cfg(feature = "gpu")]
const STAGE1_BLOCK_MAX: usize = STAGE1_BLOCK_THREADS;

/// Max planes per cube for the cross-plane carry stage (1024/32; 32 is the safe cap,
/// mirroring `primitives::N_PLANES_MAX`).
#[cfg(feature = "gpu")]
const STAGE1_N_PLANES_MAX: usize = 32;

/// NET-NEW two-level within-block inclusive scan (D-03) — the hip stage-1 scan, built
/// on the cubecl-0.10 plane intrinsics + a `SharedMemory` cross-plane carry under
/// `sync_cube()` (the idiom borrowed from `primitives.rs`, NOT the generic `block_scan`
/// segment contract). Returns unit `UNIT_POS`'s block-wide inclusive prefix of `v`.
/// Two levels: `plane_inclusive_sum` intra-plane, then plane 0 exclusive-scans the
/// per-plane totals and each unit adds its plane base back (mirrors
/// `primitives::plane_block_scan_kernel_f32`, forward=cumulative-LEFT / reverse=
/// cumulative-RIGHT via the caller's read pattern).
#[cfg(feature = "gpu")]
#[cube]
#[allow(clippy::manual_div_ceil)] // matches primitives::plane_block_scan_kernel_f32 (cubecl lowering)
fn stage1_block_scan(v: f32, plane_dim: u32) -> f32 {
    let mut stage = SharedMemory::<f32>::new(STAGE1_N_PLANES_MAX);
    let pd = plane_dim as usize;
    let i = UNIT_POS as usize;
    let lane = UNIT_POS_PLANE as usize;
    let plane_id = i / pd;
    // 1. inclusive scan WITHIN each plane.
    let local = plane_inclusive_sum(v);
    // 2. last lane of each plane stages its plane total.
    if lane == pd - 1 {
        stage[plane_id] = local;
    }
    sync_cube();
    // 3. plane 0 exclusive-scans the per-plane totals.
    let cd = CUBE_DIM as usize;
    let n_planes = (cd + pd - 1) / pd;
    if plane_id == 0 {
        let t = if lane < n_planes { stage[lane] } else { f32::new(0.0) };
        stage[lane] = plane_exclusive_sum(t);
    }
    sync_cube();
    // 4. add the plane base back → block-wide inclusive prefix.
    let base = stage[plane_id];
    base + local
}

/// `ReduceBestGainForLeaves` block argmax over `(local_gain, leaf_index)` — the winning
/// LEAF index (stage-3 cross-leaf reduce, `cu:67-123`). Mirror of [`reduce_best_gain`]: leaf
/// indices are non-negative, so a valid leaf encodes as `leaf_index as u32` and "no valid
/// leaf" as the `block_size` sentinel (u32 throughout — cubecl lowers the u32 shared-read
/// return, matching `reduce_best_gain`; the caller maps `block_size` → `-1`). Strict `>`
/// keeps the LOWEST leaf index on a tie, matching the Task-2 cpu fold. gpu-gated f32 (D-10 —
/// the leaf gains are widened to f32 on the hip path; the cpu anchor argmax is the host f64
/// fold in [`find_best_from_all_splits_on`]); compile-verified, on-device asserted in 17-05.
#[cfg(feature = "gpu")]
#[cube]
fn reduce_best_gain_for_leaves(local_gain: f32, leaf_index: i32, block_size: u32) -> u32 {
    let mut sh_gain = SharedMemory::<f32>::new(STAGE1_BLOCK_MAX);
    let mut sh_leaf = SharedMemory::<u32>::new(STAGE1_BLOCK_MAX);
    let mut sh_win = SharedMemory::<u32>::new(1usize);
    let i = UNIT_POS as usize;
    let valid = leaf_index != -1i32;
    sh_gain[i] = local_gain;
    // Encode: valid leaf → its (non-negative) index as u32; invalid → block_size sentinel.
    sh_leaf[i] = select(valid, u32::cast_from(leaf_index), block_size);
    sync_cube();
    if UNIT_POS == 0 {
        let mut best = block_size; // sentinel: no valid leaf
        let mut best_g = 0.0f32;
        let n = block_size as usize;
        for k in 0..n {
            let fnd = sh_leaf[k] != block_size;
            // strict `>` keeps the FIRST (lowest leaf index) on a tie; `best == sentinel`
            // admits the first valid candidate.
            let take = fnd && (best == block_size || sh_gain[k] > best_g);
            best = select(take, sh_leaf[k], best);
            best_g = select(take, sh_gain[k], best_g);
        }
        sh_win[0] = best;
    }
    sync_cube();
    sh_win[0]
}

/// `PrepareLeafBestSplitInfo` + the `FindBestFromAllSplitsKernel` `[6]`/`[7]` writes
/// (`cu:2113-2159`) — the single-owner 8-int export packer (SC#2, the ONLY device→host
/// transfer per iteration). `inp` carries the pre-read scalars
/// `[0]=smaller.inner_feature_index [1]=smaller.threshold [2]=smaller.default_left
///  [3]=larger.inner_feature_index [4]=larger.threshold [5]=larger.default_left
///  [6]=best_leaf_index [7]=best_leaf.num_cat_threshold [8]=has_larger`; `out` is the
/// 8-int buffer with the larger triple gated on `has_larger` and `[7]` gated on
/// `best_leaf_index != -1` — the C++ conditional-write layout.
#[cube(launch_unchecked)]
fn prepare_leaf_best_split_info_kernel(inp: &Array<i32>, out: &mut Array<i32>) {
    let has_larger = inp[8] != 0i32;
    let best = inp[6];
    out[0] = inp[0];
    out[1] = inp[1];
    out[2] = inp[2];
    out[3] = select(has_larger, inp[3], 0i32);
    out[4] = select(has_larger, inp[4], 0i32);
    out[5] = select(has_larger, inp[5], 0i32);
    out[6] = best;
    out[7] = select(best != -1i32, inp[7], 0i32);
}

/// `ReduceBestGain` block argmax over `(local_gain, found, thread_index)` — the winning
/// thread index, tie-break by the LOWEST index via strict `>` (Pitfall 5, matching the
/// Task-1 cpu fold's first-max-wins). Stages each unit's `(gain, found)` into
/// `SharedMemory`, `sync_cube()`, then unit 0 folds with strict `>`; a `block_size`
/// sentinel means "no thread found a split". D-03 borrows the LDS idiom, not the
/// generic reduction.
#[cfg(feature = "gpu")]
#[cube]
fn reduce_best_gain(local_gain: f32, found: bool, block_size: u32) -> u32 {
    let mut sh_gain = SharedMemory::<f32>::new(STAGE1_BLOCK_MAX);
    let mut sh_found = SharedMemory::<u32>::new(STAGE1_BLOCK_MAX);
    let mut sh_win = SharedMemory::<u32>::new(1usize);
    let i = UNIT_POS as usize;
    sh_gain[i] = local_gain;
    sh_found[i] = select(found, 1u32, 0u32);
    sync_cube();
    if UNIT_POS == 0 {
        let mut best = block_size; // sentinel: no winner
        let mut best_g = 0.0f32;
        let n = block_size as usize;
        for k in 0..n {
            let fnd = sh_found[k] == 1u32;
            // strict `>` keeps the FIRST (lowest index) on a tie; `best == sentinel`
            // admits the first found candidate.
            let take = fnd && (best == block_size || sh_gain[k] > best_g);
            best = select(take, k as u32, best);
            best_g = select(take, sh_gain[k], best_g);
        }
        sh_win[0] = best;
    }
    sync_cube();
    sh_win[0]
}

/// The hip block-parallel stage-1 kernel (D-03/D-06) — one block per `(leaf,feature)`
/// task, 256 threads: each unit loads its bin, the two-level [`stage1_block_scan`]
/// produces the cumulative scanned side, complement-from-parent + two-phase count
/// recovery + guards + gain (smoothing dispatch) per unit, [`reduce_best_gain`] picks
/// the winner (strict `>`), and the winning unit writes the record (kEpsilon-subtracted).
/// Byte-for-byte the §8.1 math of [`split_eval_body_f32`], parallelized. Anchored to the
/// Task-1 cpu f64 fold; the on-device rocm parity assertion lands in 17-05 (def-f8u-01,
/// NEVER GPU-vs-GPU). NO f64 (D-10, WR-05).
#[cfg(feature = "gpu")]
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub fn split_eval_block_kernel_f32(
    hist: &Array<f32>,
    out: &mut Array<f32>,
    num_bin: i32,
    mfb_offset: i32,
    default_bin: i32,
    skip_default_bin: u32,
    reverse: u32,
    assume_out_default_left: u32,
    use_l1: u32,
    use_smoothing: u32,
    use_rand: u32,
    rand_threshold: i32,
    min_data_in_leaf: i32,
    min_sum_hessian_in_leaf: f32,
    lambda_l1: f32,
    lambda_l2: f32,
    path_smooth: f32,
    parent_output: f32,
    min_gain_shift: f32,
    sum_gradient: f32,
    sum_hessian: f32,
    num_data: i32,
    plane_dim: u32,
    block_size: u32,
) {
    let eps = f32::cast_from(K_EPSILON);
    let rev = reverse != 0;
    let use_l1_b = use_l1 != 0;
    let sm = use_smoothing != 0;
    let use_rand_b = use_rand != 0;
    let skip_def = skip_default_bin != 0;

    let cnt_factor = f32::cast_from(num_data) / sum_hessian;
    let fnbmo = num_bin - mfb_offset;
    let fwd_end = fnbmo - 2;
    let rev_end = num_bin - 2;

    let t = UNIT_POS as i32;
    // ---- per-unit bin read (forward: bin t; reverse: bin fnbmo-1-t) ----
    let skip = skip_def
        && ((rev && (num_bin - 1 - t) == default_bin) || (!rev && (t + mfb_offset) == default_bin));
    let read_active = t < fnbmo && !skip;
    let bin = select(rev, fnbmo - 1 - t, t);
    let bin_safe = select(read_active, bin, 0i32);
    let base = (bin_safe as usize) * 2;
    let g = select(read_active, hist[base], 0.0f32);
    let h_raw = select(read_active, hist[base + 1], 0.0f32);
    // thread 0 seeds kEpsilon ONCE at the scan origin.
    let h = h_raw + select(t == 0, eps, 0.0f32);

    // ---- two-level block-inclusive prefix (grad and hess, two scans) ----
    let scanned_g = stage1_block_scan(g, plane_dim);
    let scanned_h = stage1_block_scan(h, plane_dim);

    // ---- guard phase at this unit ----
    let comp_g = sum_gradient - scanned_g;
    let comp_h = sum_hessian - scanned_h;
    let scanned_cnt = round_ties_even_f32_cube(scanned_h * cnt_factor);
    let comp_cnt = num_data - scanned_cnt;

    let l_g = select(rev, comp_g, scanned_g);
    let l_h = select(rev, comp_h, scanned_h);
    let lc = select(rev, comp_cnt, scanned_cnt);
    let r_g = select(rev, scanned_g, comp_g);
    let r_h = select(rev, scanned_h, comp_h);
    let rc = select(rev, scanned_cnt, comp_cnt);
    let threshold = select(rev, rev_end - t, t + mfb_offset);

    let in_range = (rev && t <= rev_end) || (!rev && t <= fwd_end);
    let cand = in_range && !skip;
    let guard = l_h >= min_sum_hessian_in_leaf
        && lc >= min_data_in_leaf
        && r_h >= min_sum_hessian_in_leaf
        && rc >= min_data_in_leaf;
    let rand_ok = !use_rand_b || threshold == rand_threshold;

    let gain_ns = get_split_gains_f32(use_l1_b, l_g, l_h, r_g, r_h, lambda_l1, lambda_l2);
    let gain_sm = get_leaf_gain_smoothed_f32(use_l1_b, l_g, l_h, lambda_l1, lambda_l2, path_smooth, lc, parent_output)
        + get_leaf_gain_smoothed_f32(use_l1_b, r_g, r_h, lambda_l1, lambda_l2, path_smooth, rc, parent_output);
    let current_gain = select(sm, gain_sm, gain_ns);

    let found = cand && guard && rand_ok && current_gain > min_gain_shift;
    let local_gain = select(found, current_gain - min_gain_shift, f32::new(0.0));

    // ---- block argmax (strict >, lowest index wins ties) ----
    let win = reduce_best_gain(local_gain, found, block_size);
    let no_winner = win >= block_size;

    // unit 0 initialises the "no valid split" sentinel; the winning unit overwrites.
    if UNIT_POS == 0 && no_winner {
        out[0] = f32::new(0.0);
        out[1] = f32::new(0.0);
        out[2] = f32::new(0.0);
        out[3] = f32::new(0.0);
        out[4] = f32::new(0.0);
        out[5] = f32::new(0.0);
        out[6] = f32::new(0.0);
        out[7] = f32::new(0.0);
        out[8] = f32::new(0.0);
        out[9] = f32::new(0.0);
        out[10] = f32::new(0.0);
        out[11] = f32::new(0.0);
        out[12] = f32::new(0.0);
        out[13] = f32::new(0.0);
    }
    if UNIT_POS == win {
        // write phase: kEpsilon subtracted, count RE-recovered (this unit's prefix).
        let w_scanned_g = scanned_g;
        let w_scanned_h = scanned_h - eps;
        let w_scanned_cnt = round_ties_even_f32_cube(w_scanned_h * cnt_factor);
        let w_comp_g = sum_gradient - w_scanned_g;
        let w_comp_h = sum_hessian - w_scanned_h - eps;
        let w_comp_cnt = num_data - w_scanned_cnt;

        let wl_g = select(rev, w_comp_g, w_scanned_g);
        let wl_h = select(rev, w_comp_h, w_scanned_h);
        let wl_c = select(rev, w_comp_cnt, w_scanned_cnt);
        let wr_g = select(rev, w_scanned_g, w_comp_g);
        let wr_h = select(rev, w_scanned_h, w_comp_h);
        let wr_c = select(rev, w_scanned_cnt, w_comp_cnt);

        let l_out_ns = calculate_splitted_leaf_output_f32(use_l1_b, wl_g, wl_h, lambda_l1, lambda_l2);
        let l_out_sm = calculate_splitted_leaf_output_smoothed_f32(
            use_l1_b, wl_g, wl_h, lambda_l1, lambda_l2, path_smooth, wl_c, parent_output,
        );
        let left_output = select(sm, l_out_sm, l_out_ns);
        let r_out_ns = calculate_splitted_leaf_output_f32(use_l1_b, wr_g, wr_h, lambda_l1, lambda_l2);
        let r_out_sm = calculate_splitted_leaf_output_smoothed_f32(
            use_l1_b, wr_g, wr_h, lambda_l1, lambda_l2, path_smooth, wr_c, parent_output,
        );
        let right_output = select(sm, r_out_sm, r_out_ns);
        let left_gain = get_leaf_gain_given_output_f32(use_l1_b, wl_g, wl_h, lambda_l1, lambda_l2, left_output);
        let right_gain = get_leaf_gain_given_output_f32(use_l1_b, wr_g, wr_h, lambda_l1, lambda_l2, right_output);
        let default_left_f = select(assume_out_default_left != 0, 1.0f32, 0.0f32);

        out[0] = f32::new(1.0);
        out[1] = f32::cast_from(threshold);
        out[2] = default_left_f;
        out[3] = local_gain;
        out[4] = wl_g;
        out[5] = wl_h;
        out[6] = f32::cast_from(wl_c);
        out[7] = wr_g;
        out[8] = wr_h;
        out[9] = f32::cast_from(wr_c);
        out[10] = left_output;
        out[11] = right_output;
        out[12] = left_gain;
        out[13] = right_gain;
    }
}

/// `GlobalMemoryPrefixSum` (`cuda_algorithms.hpp:169-185`) — a chunked two-level
/// IN-PLACE inclusive scan over a global-memory scratch `array[0..len]` (D-05). Each
/// unit owns a contiguous chunk of `ceil(len / blockDim)` elements: it sums its chunk,
/// the block exclusive-scans the per-chunk sums (the `ShufflePrefixSumExclusive`
/// analog, here `stage1_block_scan(sum) - sum`), each unit adds that base to its first
/// element, then serially propagates within its chunk. NO f64 (D-10, WR-05). Every
/// unit must reach both `sync_cube()`s (the C++ `__syncthreads()` after each scan is
/// issued by the caller).
#[cfg(feature = "gpu")]
#[cube]
#[allow(clippy::manual_div_ceil)] // `.div_ceil()` does not lower in cubecl `#[cube]`
fn global_memory_prefix_sum(array: &mut Array<f32>, len: u32, plane_dim: u32) {
    let bd = CUBE_DIM;
    let num_per_thread = (len + bd - 1) / bd;
    let start = UNIT_POS * num_per_thread;
    let cand_end = start + num_per_thread;
    let end = select(cand_end < len, cand_end, len);

    // 1. this unit's chunk sum (strided contiguous chunk `[start, end)`).
    let mut thread_sum = f32::new(0.0);
    for k in 0..num_per_thread {
        let index = start + k;
        let active = index < end;
        thread_sum += select(active, array[(select(active, index, 0u32)) as usize], f32::new(0.0));
    }

    // 2. exclusive block-scan of the per-chunk sums (base = inclusive − own).
    let incl = stage1_block_scan(thread_sum, plane_dim);
    let thread_base = incl - thread_sum;
    sync_cube();

    // 3. add the base to the chunk's first element, then serially propagate.
    if start < end {
        array[start as usize] += thread_base;
    }
    for k in 0..num_per_thread {
        let index = start + k;
        if index > start && index < end {
            array[index as usize] += array[(index - 1) as usize];
        }
    }
}

/// The hip `_GlobalMemory` stage-1 strided kernel (D-05/D-06) — one block per
/// `(leaf,feature)` task, 256 threads, for features whose `num_bin` exceeds the block
/// width. A VERBATIM strided port of `FindBestSplitsForLeafKernelInner_GlobalMemory`
/// (`cuda_best_split_finder.cu:1051-1273`, the continuous non-`na_as_missing`-mfb1
/// branches): each unit STRIDES over bins `t, t+blockDim, …` linearising the
/// scan-position sums into the pre-allocated `grad_buf`/`hess_buf` global scratch,
/// [`global_memory_prefix_sum`] scans them in place, then a second strided pass
/// evaluates guards + gain (smoothing dispatch) exactly as [`split_eval_body_f32`],
/// [`reduce_best_gain`] picks the winner (strict `>`, lowest index), and the winning
/// unit writes the record (kEpsilon-subtracted). SAME gain / count / guard / argmax
/// math as 17-03 — only the scan carrier (global scratch) and the strided iteration
/// differ (A4). Anchored to the cpu f64 fold; the on-device rocm assertion lands in
/// 17-05 (def-f8u-01, NEVER GPU-vs-GPU). NO f64 (D-10, WR-05).
///
/// Faithful-scope note: the `na_as_missing && mfb_offset == 1` special reduction
/// subcase (`cu:1095-1114`, a `ShuffleReduceSum` of the non-default bins) is NOT
/// exercised by the D-07 fixture (the `globalmem_spill` golden is forward, `mfb=0`,
/// `na=0`); it is a `_GlobalMemory` sub-branch to complete when a golden needs it,
/// tracked as a known limitation, not a silent stub (the continuous forward/reverse
/// branches this kernel ports are the ones the fixture drives).
#[cfg(feature = "gpu")]
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
#[allow(clippy::manual_div_ceil)] // `.div_ceil()` does not lower in cubecl `#[cube]`
pub fn split_eval_globalmem_kernel_f32(
    hist: &Array<f32>,
    grad_buf: &mut Array<f32>,
    hess_buf: &mut Array<f32>,
    out: &mut Array<f32>,
    num_bin: i32,
    mfb_offset: i32,
    default_bin: i32,
    skip_default_bin: u32,
    reverse: u32,
    assume_out_default_left: u32,
    use_l1: u32,
    use_smoothing: u32,
    use_rand: u32,
    rand_threshold: i32,
    min_data_in_leaf: i32,
    min_sum_hessian_in_leaf: f32,
    lambda_l1: f32,
    lambda_l2: f32,
    path_smooth: f32,
    parent_output: f32,
    min_gain_shift: f32,
    sum_gradient: f32,
    sum_hessian: f32,
    num_data: i32,
    plane_dim: u32,
    block_size: u32,
) {
    let eps = f32::cast_from(K_EPSILON);
    let rev = reverse != 0;
    let use_l1_b = use_l1 != 0;
    let sm = use_smoothing != 0;
    let use_rand_b = use_rand != 0;
    let skip_def = skip_default_bin != 0;

    let cnt_factor = f32::cast_from(num_data) / sum_hessian;
    let fnbmo = num_bin - mfb_offset; // feature_num_bin_minus_offset
    let fwd_end = fnbmo - 2;
    let rev_end = num_bin - 2;

    let bd = CUBE_DIM;
    let fnbmo_u = fnbmo as u32;
    // ceil(fnbmo / blockDim) strided iterations cover every scan position.
    let iters = (fnbmo_u + bd - 1) / bd;

    // ---- phase A: linearise the (skip/reverse-adjusted) per-bin sums into scratch ----
    for k in 0..iters {
        let bin = UNIT_POS + k * bd; // strided scan position in [0, fnbmo)
        let bin_i = bin as i32;
        if bin < fnbmo_u {
            // forward skip: (bin + mfb_offset) == default_bin; reverse skip: (num_bin-1-bin) == default_bin.
            let skip_sum = skip_def
                && ((rev && (num_bin - 1 - bin_i) == default_bin)
                    || (!rev && (bin_i + mfb_offset) == default_bin));
            // reverse read-index = fnbmo-1-bin; forward read-index = bin.
            let read_index = select(rev, fnbmo - 1 - bin_i, bin_i);
            let ro = (read_index as usize) * 2;
            let g = select(skip_sum, f32::new(0.0), hist[ro]);
            let h = select(skip_sum, f32::new(0.0), hist[ro + 1]);
            grad_buf[bin as usize] = g;
            hess_buf[bin as usize] = h;
        }
    }
    sync_cube();
    // thread 0 seeds kEpsilon ONCE at the scan origin (cu:1146).
    if UNIT_POS == 0 {
        hess_buf[0] += eps;
    }
    sync_cube();

    // ---- phase B: in-place global-memory inclusive prefix sums ----
    global_memory_prefix_sum(grad_buf, fnbmo_u, plane_dim);
    sync_cube();
    global_memory_prefix_sum(hess_buf, fnbmo_u, plane_dim);
    sync_cube();

    // ---- phase C: strided evaluate — per-unit best (last-beating within the stride,
    // then cross-thread ReduceBestGain), faithfully reproducing cu:1152-1208. ----
    let mut local_gain = f32::new(0.0);
    let mut threshold_value = 0i32;
    let mut found = false;
    for k in 0..iters {
        let bin = UNIT_POS + k * bd;
        let bin_i = bin as i32;
        // forward candidate upper bound (mfb1/na special not covered here — see note).
        let in_range = select(rev, bin_i <= rev_end, bin_i <= fwd_end) && bin < fnbmo_u;
        let skip_sum = skip_def
            && ((rev && (num_bin - 1 - bin_i) == default_bin)
                || (!rev && (bin_i + mfb_offset) == default_bin));
        if in_range && !skip_sum {
            let scanned_g = grad_buf[bin as usize];
            let scanned_h = hess_buf[bin as usize];
            let scanned_cnt = round_ties_even_f32_cube(scanned_h * cnt_factor);
            let comp_g = sum_gradient - scanned_g;
            let comp_h = sum_hessian - scanned_h;
            let comp_cnt = num_data - scanned_cnt;

            // reverse: scanned side = RIGHT; forward: scanned side = LEFT.
            let l_g = select(rev, comp_g, scanned_g);
            let l_h = select(rev, comp_h, scanned_h);
            let lc = select(rev, comp_cnt, scanned_cnt);
            let r_g = select(rev, scanned_g, comp_g);
            let r_h = select(rev, scanned_h, comp_h);
            let rc = select(rev, scanned_cnt, comp_cnt);
            let threshold = select(rev, rev_end - bin_i, bin_i + mfb_offset);

            let guard = l_h >= min_sum_hessian_in_leaf
                && lc >= min_data_in_leaf
                && r_h >= min_sum_hessian_in_leaf
                && rc >= min_data_in_leaf;
            let rand_ok = !use_rand_b || threshold == rand_threshold;

            let gain_ns = get_split_gains_f32(use_l1_b, l_g, l_h, r_g, r_h, lambda_l1, lambda_l2);
            let gain_sm =
                get_leaf_gain_smoothed_f32(use_l1_b, l_g, l_h, lambda_l1, lambda_l2, path_smooth, lc, parent_output)
                    + get_leaf_gain_smoothed_f32(use_l1_b, r_g, r_h, lambda_l1, lambda_l2, path_smooth, rc, parent_output);
            let current_gain = select(sm, gain_sm, gain_ns);

            // C++ overwrites on every beating candidate (last-in-stride wins within a
            // thread), then cross-thread ReduceBestGain picks the max (cu:1199-1204).
            let beat = guard && rand_ok && current_gain > min_gain_shift;
            local_gain = select(beat, current_gain - min_gain_shift, local_gain);
            threshold_value = select(beat, threshold, threshold_value);
            found = found || beat;
        }
    }

    // ---- block argmax (strict >, lowest thread index wins ties) ----
    let win = reduce_best_gain(local_gain, found, block_size);
    let no_winner = win >= block_size;

    if UNIT_POS == 0 && no_winner {
        out[0] = f32::new(0.0);
        out[1] = f32::new(0.0);
        out[2] = f32::new(0.0);
        out[3] = f32::new(0.0);
        out[4] = f32::new(0.0);
        out[5] = f32::new(0.0);
        out[6] = f32::new(0.0);
        out[7] = f32::new(0.0);
        out[8] = f32::new(0.0);
        out[9] = f32::new(0.0);
        out[10] = f32::new(0.0);
        out[11] = f32::new(0.0);
        out[12] = f32::new(0.0);
        out[13] = f32::new(0.0);
    }
    if UNIT_POS == win {
        // write phase: recover the winning bin's scanned side from scratch, kEpsilon
        // subtracted, count RE-recovered (cu:1215-1272).
        let best_bin = select(rev, num_bin - 2 - threshold_value, threshold_value - mfb_offset);
        let w_scanned_g = grad_buf[best_bin as usize];
        let w_scanned_h = hess_buf[best_bin as usize] - eps;
        let w_scanned_cnt = round_ties_even_f32_cube(w_scanned_h * cnt_factor);
        let w_comp_g = sum_gradient - w_scanned_g;
        let w_comp_h = sum_hessian - w_scanned_h - eps;
        let w_comp_cnt = num_data - w_scanned_cnt;

        let wl_g = select(rev, w_comp_g, w_scanned_g);
        let wl_h = select(rev, w_comp_h, w_scanned_h);
        let wl_c = select(rev, w_comp_cnt, w_scanned_cnt);
        let wr_g = select(rev, w_scanned_g, w_comp_g);
        let wr_h = select(rev, w_scanned_h, w_comp_h);
        let wr_c = select(rev, w_scanned_cnt, w_comp_cnt);

        let l_out_ns = calculate_splitted_leaf_output_f32(use_l1_b, wl_g, wl_h, lambda_l1, lambda_l2);
        let l_out_sm = calculate_splitted_leaf_output_smoothed_f32(
            use_l1_b, wl_g, wl_h, lambda_l1, lambda_l2, path_smooth, wl_c, parent_output,
        );
        let left_output = select(sm, l_out_sm, l_out_ns);
        let r_out_ns = calculate_splitted_leaf_output_f32(use_l1_b, wr_g, wr_h, lambda_l1, lambda_l2);
        let r_out_sm = calculate_splitted_leaf_output_smoothed_f32(
            use_l1_b, wr_g, wr_h, lambda_l1, lambda_l2, path_smooth, wr_c, parent_output,
        );
        let right_output = select(sm, r_out_sm, r_out_ns);
        let left_gain = get_leaf_gain_given_output_f32(use_l1_b, wl_g, wl_h, lambda_l1, lambda_l2, left_output);
        let right_gain = get_leaf_gain_given_output_f32(use_l1_b, wr_g, wr_h, lambda_l1, lambda_l2, right_output);
        let default_left_f = select(assume_out_default_left != 0, 1.0f32, 0.0f32);

        out[0] = f32::new(1.0);
        out[1] = f32::cast_from(threshold_value);
        out[2] = default_left_f;
        out[3] = local_gain;
        out[4] = wl_g;
        out[5] = wl_h;
        out[6] = f32::cast_from(wl_c);
        out[7] = wr_g;
        out[8] = wr_h;
        out[9] = f32::cast_from(wr_c);
        out[10] = left_output;
        out[11] = right_output;
        out[12] = left_gain;
        out[13] = right_gain;
    }
}

/// STAGE 1 `_GlobalMemory` spill launcher (hip f32, gpu-gated) — drives
/// [`split_eval_globalmem_kernel_f32`] over one `(leaf,feature)` task whose bin count
/// exceeds the block width, using the PRE-ALLOCATED [`Stage1GlobalMemScratch`] handles
/// (D-11 — never allocates the scan scratch here). The output packet is the standard
/// 14-cell [`SplitScalars`] readback (SC#2 single transfer). Anchored to
/// [`find_best_splits_stage1_on`] within the ~1e-5 f32 envelope; the on-device rocm
/// parity assertion lands in 17-05 (def-f8u-01).
///
/// `block_threads` is the launch width (≤ [`STAGE1_BLOCK_THREADS`]); `plane_dim` is the
/// device plane/warp width for the two-level per-chunk scan.
///
/// # Errors
/// [`ComputeError`] from [`validate_stage1_inputs`] (bad `num_bin` / histogram length)
/// or if the scratch slab is too small for this feature's `num_bin - mfb_offset`.
#[cfg(feature = "gpu")]
#[allow(clippy::too_many_arguments)]
pub fn find_best_splits_stage1_globalmem_f32_on<R: cubecl::Runtime>(
    client: &ComputeClient<R>,
    hist: &[f32],
    task: &SplitFindTask,
    scalars: &Stage1Scalars,
    scratch: &Stage1GlobalMemScratch<R>,
    block_threads: u32,
    plane_dim: u32,
) -> Result<SplitScalars, ComputeError> {
    validate_stage1_inputs(task.num_bin, hist.len())?;
    if task.is_categorical {
        return Ok(SplitScalars::default());
    }
    let num_bin_i = task.num_bin as i32;
    let fnbmo = (task.num_bin as usize).saturating_sub(task.mfb_offset as usize);
    if fnbmo > scratch.slab_len() {
        return Err(ComputeError::Runtime {
            detail: format!(
                "globalmem stage1: feature scan length {fnbmo} exceeds scratch slab {}",
                scratch.slab_len()
            ),
        });
    }
    let rand_threshold: i32 = if scalars.use_rand && num_bin_i - 2 > 0 {
        let draw = draw_rand_int32_on(client, &[scalars.rng_seed as u32], 1)?;
        draw[0] % (num_bin_i - 2)
    } else {
        -1
    };
    let min_gain_shift = (scalars.parent_gain + scalars.min_gain_to_split) as f32;

    let h_hist = client.create_from_slice(f32::as_bytes(hist));
    let h_out = client.empty(STAGE1_OUT_LEN * core::mem::size_of::<f32>());

    // SAFETY: `h_hist` is host-validated to `2*num_bin` cells; the scan scratch slab is
    // `>= fnbmo` (checked above) and pre-allocated once in `Stage1GlobalMemScratch::new`;
    // `h_out` is `STAGE1_OUT_LEN` cells. All indices derive from the validated bin count.
    // cubecl unsafe confined here (CMP-01, T-17-01/T-17-02).
    unsafe {
        split_eval_globalmem_kernel_f32::launch_unchecked(
            client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new_1d(block_threads),
            ArrayArg::from_raw_parts(h_hist, hist.len()),
            ArrayArg::from_raw_parts(scratch.feature_hist_grad_buffer.clone(), scratch.slab_len()),
            ArrayArg::from_raw_parts(scratch.feature_hist_hess_buffer.clone(), scratch.slab_len()),
            ArrayArg::from_raw_parts(h_out.clone(), STAGE1_OUT_LEN),
            num_bin_i,
            task.mfb_offset as i32,
            task.default_bin as i32,
            if task.skip_default_bin { 1u32 } else { 0u32 },
            if task.reverse { 1u32 } else { 0u32 },
            if task.assume_out_default_left { 1u32 } else { 0u32 },
            if scalars.use_l1 { 1u32 } else { 0u32 },
            if scalars.use_smoothing { 1u32 } else { 0u32 },
            if scalars.use_rand { 1u32 } else { 0u32 },
            rand_threshold,
            scalars.min_data_in_leaf,
            scalars.min_sum_hessian_in_leaf as f32,
            scalars.lambda_l1 as f32,
            scalars.lambda_l2 as f32,
            scalars.path_smooth as f32,
            scalars.parent_output as f32,
            min_gain_shift,
            scalars.sum_gradient as f32,
            scalars.sum_hessian as f32,
            scalars.num_data,
            plane_dim,
            block_threads,
        );
    }
    let bytes = client.read_one_unchecked(h_out);
    let cells = f32::from_bytes(&bytes);
    if cells[0] == 0.0 {
        return Ok(SplitScalars::default());
    }
    Ok(SplitScalars {
        is_valid: true,
        leaf_index: -1,
        gain: cells[3] as f64,
        inner_feature_index: task.inner_feature_index,
        threshold: cells[1] as u32,
        default_left: cells[2] != 0.0,
        left_sum_gradients: cells[4] as f64,
        left_sum_hessians: cells[5] as f64,
        left_sum_gh_quant: 0,
        left_count: cells[6] as i32,
        left_gain: cells[12] as f64,
        left_value: cells[10] as f64,
        right_sum_gradients: cells[7] as f64,
        right_sum_hessians: cells[8] as f64,
        right_sum_gh_quant: 0,
        right_count: cells[9] as i32,
        right_gain: cells[13] as f64,
        right_value: cells[11] as f64,
        num_cat_threshold: 0,
    })
}

/// The C++ `NUM_TASKS_PER_SYNC_BLOCK` (`cuda_best_split_finder.hpp:24`) — the stage-2
/// sync-block width. `num_blocks_per_leaf = ceil(num_tasks / NUM_TASKS_PER_SYNC_BLOCK)`;
/// for the anchor fixtures `num_tasks << 1024` so `num_blocks_per_leaf == 1` (the common
/// case the `…AllBlocks` fold collapses to — Claude's Discretion per 17-CONTEXT).
pub const NUM_TASKS_PER_SYNC_BLOCK: usize = 1024;

/// The number of stage-2 sync blocks per leaf for `num_tasks` tasks
/// (`(num_tasks + NUM_TASKS_PER_SYNC_BLOCK - 1) / NUM_TASKS_PER_SYNC_BLOCK`,
/// `cuda_best_split_finder.cu:2045`). Always ≥ 1.
#[must_use]
pub fn stage2_num_blocks_per_leaf(num_tasks: usize) -> usize {
    num_tasks.div_ceil(NUM_TASKS_PER_SYNC_BLOCK).max(1)
}

/// V5 launch-boundary validation for stage-2/3 (threat T-17-01/T-17-02): the
/// leaf-best-split slab is indexed at `leaf_index + block·num_leaves`, so
/// `num_leaves × num_blocks_per_leaf` must be non-zero and must not overflow BEFORE
/// the reduce writes the per-leaf winner. Returns the required slab length.
///
/// # Errors
/// [`ComputeError::Runtime`] if either operand is zero or the product overflows `usize`.
pub fn validate_stage2_inputs(
    num_leaves: usize,
    num_blocks_per_leaf: usize,
) -> Result<usize, ComputeError> {
    if num_leaves == 0 || num_blocks_per_leaf == 0 {
        return Err(ComputeError::Runtime {
            detail: format!(
                "stage2: num_leaves ({num_leaves}) and num_blocks_per_leaf \
                 ({num_blocks_per_leaf}) must both be > 0"
            ),
        });
    }
    num_leaves
        .checked_mul(num_blocks_per_leaf)
        .ok_or_else(|| ComputeError::Runtime {
            detail: format!(
                "stage2: num_leaves {num_leaves} × num_blocks_per_leaf {num_blocks_per_leaf} \
                 overflows the leaf-best-split slab"
            ),
        })
}

/// STAGE 2 — `SyncBestSplitForLeafKernel` cross-feature reduce per leaf (ODL-12, §8.2).
///
/// Reduces the per-task `(is_valid, gain)` records for ONE leaf via the `ReduceBestGain`
/// family (strict `>` ⇒ the FIRST / lowest task index survives a tie, 17-RESEARCH
/// Pitfall 5) into that leaf's best split. `per_task` is the full `2·num_tasks` record
/// slab stage-1 produced (smaller-leaf records `[0, num_tasks)`, larger-leaf records
/// `[num_tasks, 2·num_tasks)`); the reader indexes `read_index = is_smaller ? task_index
/// : task_index + num_tasks` (the IS_LARGER duality, `cu:1943`). The winner is copied
/// verbatim (its `inner_feature_index` was already stamped by stage-1 from
/// `task.inner_feature_index`, identical to the C++ re-stamp from `tasks[best_read_index]`);
/// a no-valid-split leaf yields `is_valid=false, gain=kMinScore`.
///
/// The reduction is a RESIDENT fold over records that already live from stage-1: the
/// deterministic strict-`>` order is the parity contract, and it performs **no device→host
/// readback** — the single readback is stage-3's 8-int export ONLY (SC#2). `client` is
/// unused here (reserved for the Phase-18 device-resident path).
///
/// For `num_blocks_per_leaf > 1` (num_tasks > `NUM_TASKS_PER_SYNC_BLOCK`) the per-block
/// winners are reduced by [`sync_best_split_all_blocks`]; the common
/// `num_blocks_per_leaf == 1` case is folded in here (Claude's Discretion, parity-neutral).
///
/// # Errors
/// [`ComputeError::LengthMismatch`] if `per_task` is shorter than the read window
/// (`base + num_tasks`); [`ComputeError::Runtime`] on `base + num_tasks` overflow.
pub fn sync_best_split_for_leaf_on<R: cubecl::Runtime>(
    client: &ComputeClient<R>,
    per_task: &[SplitScalars],
    num_tasks: usize,
    is_smaller: bool,
) -> Result<SplitScalars, ComputeError> {
    let _ = client; // reserved for the Phase-18 device-resident path (no readback here, SC#2).
    let base = if is_smaller { 0usize } else { num_tasks };
    let needed = base
        .checked_add(num_tasks)
        .ok_or_else(|| ComputeError::Runtime {
            detail: "stage2: base + num_tasks overflows the per-task record window".to_string(),
        })?;
    if per_task.len() < needed {
        return Err(ComputeError::LengthMismatch {
            expected: needed,
            actual: per_task.len(),
        });
    }

    // ReduceBestGain: strict `>` keeps the FIRST (lowest task index) on a tie (Pitfall 5).
    let mut best: Option<usize> = None;
    let mut best_gain = f64::NEG_INFINITY;
    for t in 0..num_tasks {
        let rec = &per_task[base + t];
        if rec.is_valid && (best.is_none() || rec.gain > best_gain) {
            best = Some(base + t);
            best_gain = rec.gain;
        }
    }

    Ok(match best {
        // The winner already carries its `inner_feature_index` (stage-1); the C++ re-stamps
        // it from `tasks[best_read_index]` (identical value). `is_valid` forced true.
        Some(i) => {
            let mut win = per_task[i];
            win.is_valid = true;
            win
        }
        // No valid split for this leaf: `gain = kMinScore` (cu:1966), `is_valid=false`.
        None => SplitScalars {
            is_valid: false,
            gain: f64::NEG_INFINITY,
            ..SplitScalars::default()
        },
    })
}

/// `SyncBestSplitForLeafKernelAllBlocks` (`cu:1972-2008`) — fold the per-block winners of
/// one leaf when `num_blocks_per_leaf > 1`. The block-0 winner is the accumulator; each
/// later block replaces it on `(other.is_valid && acc.is_valid && other.gain > acc.gain) ||
/// (!acc.is_valid && other.is_valid)` — i.e. strict `>`, block-0 survives a tie (ascending
/// block order, parity-neutral). For the common `num_blocks_per_leaf == 1` case this is the
/// identity, so [`sync_best_split_for_leaf_on`] handles it inline.
#[must_use]
pub fn sync_best_split_all_blocks(block_winners: &[SplitScalars]) -> SplitScalars {
    let mut best: Option<usize> = None;
    let mut best_gain = f64::NEG_INFINITY;
    for (b, rec) in block_winners.iter().enumerate() {
        if rec.is_valid && (best.is_none() || rec.gain > best_gain) {
            best = Some(b);
            best_gain = rec.gain;
        }
    }
    match best {
        Some(i) => {
            let mut w = block_winners[i];
            w.is_valid = true;
            w
        }
        None => SplitScalars {
            is_valid: false,
            gain: f64::NEG_INFINITY,
            ..SplitScalars::default()
        },
    }
}

/// `SetInvalidLeafSplitInfoKernel` (`cu:2010-2023`) — mark the smaller/larger leaf
/// best-split slots `is_valid=false` when a leaf produced no valid candidate (the C++
/// pre-pass that runs before the reduce for no-valid-split leaves). Bounds-checked
/// (`leaf_index >= 0 && < len`); `larger_leaf_index < 0` means "no larger leaf".
pub fn set_invalid_leaf_split_info(
    leaf_best: &mut [SplitScalars],
    is_smaller_leaf_valid: bool,
    is_larger_leaf_valid: bool,
    smaller_leaf_index: i32,
    larger_leaf_index: i32,
) {
    if !is_smaller_leaf_valid
        && smaller_leaf_index >= 0
        && (smaller_leaf_index as usize) < leaf_best.len()
    {
        leaf_best[smaller_leaf_index as usize].is_valid = false;
    }
    if !is_larger_leaf_valid
        && larger_leaf_index >= 0
        && (larger_leaf_index as usize) < leaf_best.len()
    {
        leaf_best[larger_leaf_index as usize].is_valid = false;
    }
}

/// V5 launch-boundary validation for stage-3 (threat T-17-01/T-17-02): the export reads
/// `per_leaf[smaller_leaf_index]` / `[larger_leaf_index]` and the argmax + self-invalidation
/// index `[0, cur_num_leaves]` plus the freshly-created leaf slot `[cur_num_leaves]`, so
/// every index must be in range BEFORE the export launch.
///
/// # Errors
/// [`ComputeError::Runtime`] if any leaf index is out of `[0, len)` (larger `< 0` = "no
/// larger leaf", allowed), or if `cur_num_leaves` exceeds the argmax window.
fn validate_stage3_inputs(
    len: usize,
    smaller_leaf_index: i32,
    larger_leaf_index: i32,
    cur_num_leaves: usize,
) -> Result<(), ComputeError> {
    let in_range = |idx: i32| idx >= 0 && (idx as usize) < len;
    if !in_range(smaller_leaf_index) {
        return Err(ComputeError::Runtime {
            detail: format!(
                "stage3: smaller_leaf_index {smaller_leaf_index} out of range [0, {len})"
            ),
        });
    }
    if larger_leaf_index >= 0 && !in_range(larger_leaf_index) {
        return Err(ComputeError::Runtime {
            detail: format!(
                "stage3: larger_leaf_index {larger_leaf_index} out of range [0, {len})"
            ),
        });
    }
    if cur_num_leaves > len {
        return Err(ComputeError::Runtime {
            detail: format!(
                "stage3: cur_num_leaves {cur_num_leaves} exceeds the per-leaf slab {len}"
            ),
        });
    }
    Ok(())
}

/// STAGE 3 — `FindBestFromAllSplitsKernel` + `PrepareLeafBestSplitInfo` (ODL-12, §8.3).
///
/// Cross-leaf argmax over `(gain, leaf_index)` for the `[0, cur_num_leaves)` per-leaf best
/// splits (strict `>` ⇒ the LOWEST leaf index survives a tie, matching
/// [`reduce_best_gain_for_leaves`]) → `best_leaf_index` (`-1` if none valid). Then the
/// behavioral SELF-INVALIDATION (`cu:2131-2135`): the chosen leaf's slot AND the
/// freshly-created leaf slot (`cur_num_leaves`) are marked `is_valid=false` so neither is
/// re-picked next iteration. Finally the 8-int `cuda_best_split_info_buffer` is packed via
/// [`prepare_leaf_best_split_info_kernel`] — the ONLY device→host transfer per iteration
/// (SC#2, a single `read_one_unchecked`); the full per-side records stay RESIDENT for
/// Phase 18.
///
/// Field layout (17-RESEARCH §"3-Stage Reduction & Export"):
/// `[0]=smaller.inner_feature_index [1]=smaller.threshold [2]=smaller.default_left
///  [3]=larger.inner_feature_index  [4]=larger.threshold  [5]=larger.default_left
///  [6]=best_leaf_index [7]=best_leaf.num_cat_threshold` (0 for continuous — Phase-22 fills
/// categorical). The larger triple `[3..6]` is written only when `larger_leaf_index >= 0`;
/// `[7]` only when `best_leaf_index != -1` (else 0). `per_leaf` is mutated in place by the
/// self-invalidation (observable to the caller / a two-iteration golden).
///
/// # Errors
/// [`ComputeError`] from [`validate_stage3_inputs`] (out-of-range leaf indices) or the
/// export launch.
pub fn find_best_from_all_splits_on<R: cubecl::Runtime>(
    client: &ComputeClient<R>,
    per_leaf: &mut [SplitScalars],
    smaller_leaf_index: i32,
    larger_leaf_index: i32,
    cur_num_leaves: usize,
) -> Result<[i64; 8], ComputeError> {
    validate_stage3_inputs(
        per_leaf.len(),
        smaller_leaf_index,
        larger_leaf_index,
        cur_num_leaves,
    )?;

    // FindBestFromAllSplitsKernel: cross-leaf argmax, strict `>` ⇒ lowest leaf index on a
    // tie (the deterministic cpu f64 anchor; the gpu-gated reduce_best_gain_for_leaves is
    // the hip mirror, 17-05 Task 3).
    let mut best_leaf: i32 = -1;
    let mut best_gain = f64::NEG_INFINITY;
    for (leaf, r) in per_leaf.iter().enumerate().take(cur_num_leaves) {
        if r.is_valid && r.gain > best_gain {
            best_gain = r.gain;
            best_leaf = leaf as i32;
        }
    }
    // [7] reads the chosen leaf's num_cat_threshold (before self-invalidation flips is_valid;
    // num_cat_threshold is unaffected — read it now).
    let best_num_cat = if best_leaf >= 0 {
        per_leaf[best_leaf as usize].num_cat_threshold
    } else {
        0
    };

    // Self-invalidation (behavioral, cu:2131-2135): the chosen leaf + the freshly-created
    // leaf slot so neither is re-picked next iteration.
    if best_leaf != -1 {
        per_leaf[best_leaf as usize].is_valid = false;
        if cur_num_leaves < per_leaf.len() {
            per_leaf[cur_num_leaves].is_valid = false;
        }
    }

    // PrepareLeafBestSplitInfo reads the smaller/larger leaves' feature/threshold/default_left
    // (NOT is_valid — unaffected by the self-invalidation above). SplitScalars is Copy.
    let smaller = per_leaf[smaller_leaf_index as usize];
    let has_larger = larger_leaf_index >= 0;
    let larger = if has_larger {
        per_leaf[larger_leaf_index as usize]
    } else {
        SplitScalars::default()
    };

    let inp: [i32; 9] = [
        smaller.inner_feature_index,
        smaller.threshold as i32,
        i32::from(smaller.default_left),
        larger.inner_feature_index,
        larger.threshold as i32,
        i32::from(larger.default_left),
        best_leaf,
        best_num_cat,
        i32::from(has_larger),
    ];

    let h_in = client.create_from_slice(i32::as_bytes(&inp));
    // The export kernel writes ALL 8 cells (single owner), so `empty` needs no zero-init.
    let h_out = client.empty(8 * core::mem::size_of::<i32>());

    // SAFETY: `h_in` is exactly 9 i32 cells, `h_out` 8 i32 cells; both outlive the launch.
    // The single-owner kernel indexes only `inp[0..9]` / `out[0..8]` (constant indices).
    // This is the ONLY device→host readback per iteration (SC#2). cubecl unsafe confined
    // here (CMP-01, T-17-01/T-17-03).
    unsafe {
        prepare_leaf_best_split_info_kernel::launch_unchecked(
            client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new_1d(1),
            ArrayArg::from_raw_parts(h_in, 9),
            ArrayArg::from_raw_parts(h_out.clone(), 8),
        );
    }
    let bytes = client.read_one_unchecked(h_out);
    let cells = i32::from_bytes(&bytes);

    let mut out = [0i64; 8];
    for (k, slot) in out.iter_mut().enumerate() {
        *slot = cells[k] as i64;
    }
    Ok(out)
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

    /// V5 launch-boundary validation (threat T-17-01): a zero `num_bin`, an
    /// overflowing `num_bin`, and a histogram-length mismatch are all rejected with a
    /// typed error BEFORE any launch.
    #[test]
    fn validate_stage1_inputs_rejects_bad_shapes() {
        // num_bin == 0 -> Runtime error.
        assert!(matches!(
            validate_stage1_inputs(0, 0),
            Err(ComputeError::Runtime { .. })
        ));
        // hist length mismatch (num_bin=4 wants 8 cells, given 6) -> LengthMismatch.
        assert!(matches!(
            validate_stage1_inputs(4, 6),
            Err(ComputeError::LengthMismatch { expected: 8, actual: 6 })
        ));
        // Correct length is accepted (returns 2*num_bin).
        assert_eq!(validate_stage1_inputs(4, 8).unwrap(), 8);
    }

    /// D-05 dispatch boundary: `num_bin > block_threads` spills to `_GlobalMemory`.
    #[test]
    fn stage1_dispatch_globalmem_boundary() {
        // At or below the block width → in-block two-level path.
        assert!(!stage1_needs_globalmem(1, STAGE1_BLOCK_THREADS));
        assert!(!stage1_needs_globalmem(256, STAGE1_BLOCK_THREADS));
        // Beyond the block width → strided global-memory spill (the golden's num_bin=300).
        assert!(stage1_needs_globalmem(257, STAGE1_BLOCK_THREADS));
        assert!(stage1_needs_globalmem(300, STAGE1_BLOCK_THREADS));
    }

    /// V5 (T-17-01/T-17-02): the `_GlobalMemory` scratch-slab sizing rejects a
    /// zero/overflowing `largest_feature_bin_count × num_concurrent_blocks` product
    /// with a typed error BEFORE any allocation.
    #[test]
    fn validate_globalmem_scratch_rejects_overflow() {
        // Zero bin count / zero blocks → Runtime error.
        assert!(matches!(
            validate_globalmem_scratch(0, 1),
            Err(ComputeError::Runtime { .. })
        ));
        assert!(matches!(
            validate_globalmem_scratch(300, 0),
            Err(ComputeError::Runtime { .. })
        ));
        // Overflowing product → Runtime error (usize::MAX bins × 2 blocks).
        assert!(matches!(
            validate_globalmem_scratch(usize::MAX, 2),
            Err(ComputeError::Runtime { .. })
        ));
        // A valid product returns the slab length.
        assert_eq!(validate_globalmem_scratch(300, 4).unwrap(), 1200);
    }

    /// D-11 alloc-once: the scratch constructor allocates EXACTLY
    /// [`NUM_STAGE1_SCRATCH_BUFFERS`] (+4) device buffers, once, and never more — the
    /// structural "no per-split device alloc" invariant (the `DeviceSplitInfo` counter
    /// idiom). Runs on the cubecl-cpu client (no `gpu` feature needed).
    #[test]
    fn globalmem_scratch_allocated_exactly_once() {
        use crate::runtime::cpu_client;
        let client = cpu_client();
        let scratch = Stage1GlobalMemScratch::new(&client, 300, 4)
            .expect("scratch alloc for 300 bins × 4 blocks");
        assert_eq!(
            scratch.device_allocations(),
            NUM_STAGE1_SCRATCH_BUFFERS,
            "the 4 scan/reserved buffers are allocated exactly once (D-11)"
        );
        assert_eq!(scratch.device_allocations(), 4);
        assert_eq!(scratch.slab_len(), 1200);
        assert_eq!(scratch.largest_feature_bin_count(), 300);
        assert_eq!(scratch.num_concurrent_blocks(), 4);
        // Overflowing construction is rejected (V5) before any alloc.
        assert!(Stage1GlobalMemScratch::new(&client, usize::MAX, 2).is_err());
    }

    /// A minimal valid [`SplitScalars`] with the given gain / inner_feature_index — a
    /// stage-1 output stand-in for the stage-2/3 reduction unit tests.
    fn rec(gain: f64, feat: i32, valid: bool) -> SplitScalars {
        SplitScalars {
            is_valid: valid,
            inner_feature_index: feat,
            gain,
            threshold: 3,
            default_left: false,
            ..SplitScalars::default()
        }
    }

    /// V5 stage-2 launch-boundary validation (T-17-01/T-17-02): zero operands and
    /// overflowing `num_leaves × num_blocks_per_leaf` are rejected before any reduce;
    /// `num_blocks_per_leaf` derivation is `ceil(num_tasks / 1024)` (≥ 1).
    #[test]
    fn validate_stage2_inputs_and_block_count() {
        assert!(matches!(
            validate_stage2_inputs(0, 1),
            Err(ComputeError::Runtime { .. })
        ));
        assert!(matches!(
            validate_stage2_inputs(1, 0),
            Err(ComputeError::Runtime { .. })
        ));
        assert!(matches!(
            validate_stage2_inputs(usize::MAX, 2),
            Err(ComputeError::Runtime { .. })
        ));
        assert_eq!(validate_stage2_inputs(7, 3).unwrap(), 21);
        // ceil(num_tasks / NUM_TASKS_PER_SYNC_BLOCK), floored at 1.
        assert_eq!(stage2_num_blocks_per_leaf(0), 1);
        assert_eq!(stage2_num_blocks_per_leaf(1), 1);
        assert_eq!(stage2_num_blocks_per_leaf(1024), 1);
        assert_eq!(stage2_num_blocks_per_leaf(1025), 2);
    }

    /// Stage-2 cross-feature reduce: strict `>` argmax (lowest task index on a tie), the
    /// smaller/larger read-window duality, the `…AllBlocks` fold, and the invalid-leaf
    /// marker. Runs on the cubecl-cpu client (no `gpu` feature).
    #[test]
    fn stage2_cross_feature_reduce_fold() {
        use crate::runtime::cpu_client;
        let client = cpu_client();
        // 3 smaller-leaf tasks (feat 0/1/2) + 3 larger-leaf tasks (feat 3/4/5), the
        // full 2·num_tasks slab. Smaller winner = feat 1 (gain 5); larger winner = feat 4.
        let per_task = vec![
            rec(2.0, 0, true),
            rec(5.0, 1, true),
            rec(5.0, 2, true), // tie with feat 1's gain — lowest index (feat 1) must win
            rec(1.0, 3, true),
            rec(9.0, 4, true),
            rec(4.0, 5, true),
        ];
        let smaller = sync_best_split_for_leaf_on(&client, &per_task, 3, true).unwrap();
        assert!(smaller.is_valid);
        assert_eq!(smaller.inner_feature_index, 1, "strict-> keeps the lowest-index tie (feat 1)");
        assert_eq!(smaller.gain, 5.0);
        let larger = sync_best_split_for_leaf_on(&client, &per_task, 3, false).unwrap();
        assert!(larger.is_valid);
        assert_eq!(larger.inner_feature_index, 4, "larger reads [num_tasks, 2·num_tasks)");
        assert_eq!(larger.gain, 9.0);

        // A short slab (missing the larger half) is rejected (V5).
        assert!(matches!(
            sync_best_split_for_leaf_on(&client, &per_task[..3], 3, false),
            Err(ComputeError::LengthMismatch { .. })
        ));

        // All-invalid tasks → no-valid-split sentinel (is_valid=false, gain=kMinScore).
        let none = vec![rec(2.0, 0, false), rec(5.0, 1, false)];
        let w = sync_best_split_for_leaf_on(&client, &none, 2, true).unwrap();
        assert!(!w.is_valid);
        assert_eq!(w.gain, f64::NEG_INFINITY);

        // …AllBlocks fold: block winners reduce with strict `>` (block-0 on a tie).
        let blocks = vec![rec(3.0, 7, true), rec(3.0, 8, true), rec(6.0, 9, true)];
        assert_eq!(sync_best_split_all_blocks(&blocks).inner_feature_index, 9);
        let blocks_tie = vec![rec(6.0, 7, true), rec(6.0, 8, true)];
        assert_eq!(
            sync_best_split_all_blocks(&blocks_tie).inner_feature_index,
            7,
            "…AllBlocks keeps the lowest block index on a tie"
        );

        // SetInvalidLeafSplitInfo marks no-valid-split leaves is_valid=false.
        let mut leaf_best = vec![rec(1.0, 0, true), rec(2.0, 1, true), rec(3.0, 2, true)];
        set_invalid_leaf_split_info(&mut leaf_best, false, true, 0, 2);
        assert!(!leaf_best[0].is_valid, "smaller leaf 0 invalidated");
        assert!(leaf_best[2].is_valid, "larger leaf 2 stays valid");
        // larger_leaf_index < 0 (no larger leaf) is a no-op on the larger slot.
        set_invalid_leaf_split_info(&mut leaf_best, true, false, 1, -1);
        assert!(leaf_best[1].is_valid, "no larger leaf → no invalidation");
    }

    /// Stage-3 cross-leaf argmax + 8-int export + self-invalidation (ODL-12, §8.3):
    /// the field layout, the strict-`>` lowest-leaf tie-break, the behavioral
    /// self-invalidation (chosen leaf + cur_num_leaves slot), and the no-split path.
    /// Runs on the cubecl-cpu client with the single 8-int readback (SC#2).
    #[test]
    fn stage3_cross_leaf_argmax_export_and_self_invalidation() {
        use crate::runtime::cpu_client;
        let client = cpu_client();

        // V5: out-of-range leaf indices rejected before the export.
        assert!(validate_stage3_inputs(2, 5, -1, 2).is_err());
        assert!(validate_stage3_inputs(2, 0, 9, 2).is_err());
        assert!(validate_stage3_inputs(2, 0, -1, 3).is_err());
        assert!(validate_stage3_inputs(4, 0, 1, 2).is_ok());

        // 3 valid leaves + a reserved freshly-created slot (index 3). smaller=leaf 0,
        // larger=leaf 1. Leaf 2 has the max gain → best_leaf_index=2.
        let mut per_leaf = vec![
            rec(2.0, 10, true), // leaf 0 (smaller)
            rec(3.0, 11, true), // leaf 1 (larger)
            rec(9.0, 12, true), // leaf 2 (cross-leaf winner)
            rec(0.0, -1, false),
        ];
        per_leaf[0].threshold = 5;
        per_leaf[0].default_left = true;
        per_leaf[1].threshold = 7;
        per_leaf[1].default_left = false;

        let out = find_best_from_all_splits_on(&client, &mut per_leaf, 0, 1, 3).unwrap();
        // Field layout idx 0-7.
        assert_eq!(out[0], 10, "[0] smaller.inner_feature_index");
        assert_eq!(out[1], 5, "[1] smaller.threshold");
        assert_eq!(out[2], 1, "[2] smaller.default_left");
        assert_eq!(out[3], 11, "[3] larger.inner_feature_index");
        assert_eq!(out[4], 7, "[4] larger.threshold");
        assert_eq!(out[5], 0, "[5] larger.default_left");
        assert_eq!(out[6], 2, "[6] best_leaf_index = the max-gain leaf");
        assert_eq!(out[7], 0, "[7] num_cat_threshold = 0 for continuous");

        // Self-invalidation: chosen leaf (2) AND the freshly-created slot (3) are now invalid.
        assert!(!per_leaf[2].is_valid, "chosen leaf 2 self-invalidated");
        assert!(!per_leaf[3].is_valid, "freshly-created slot 3 self-invalidated");
        assert!(per_leaf[0].is_valid && per_leaf[1].is_valid, "other leaves untouched");

        // Two-iteration proof: re-running the argmax no longer re-picks leaf 2 — the next
        // best is leaf 1 (gain 3 > leaf 0's 2).
        let out2 = find_best_from_all_splits_on(&client, &mut per_leaf, 0, 1, 3).unwrap();
        assert_eq!(out2[6], 1, "iteration 2 picks leaf 1 (leaf 2 was invalidated)");

        // No-larger-leaf: larger triple stays 0.
        let mut two = vec![rec(4.0, 20, true), rec(0.0, -1, false)];
        two[0].threshold = 8;
        let o = find_best_from_all_splits_on(&client, &mut two, 0, -1, 1).unwrap();
        assert_eq!([o[3], o[4], o[5]], [0, 0, 0], "no larger leaf → larger triple zero");
        assert_eq!(o[6], 0, "best_leaf_index = leaf 0");

        // No-valid-split path: all leaves invalid → best_leaf_index = -1, [7] = 0.
        let mut none = vec![rec(1.0, 0, false), rec(2.0, 1, false)];
        let o = find_best_from_all_splits_on(&client, &mut none, 0, 1, 2).unwrap();
        assert_eq!(o[6], -1, "no valid leaf → best_leaf_index = -1");
        assert_eq!(o[7], 0, "no valid leaf → num_cat_threshold = 0");
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
