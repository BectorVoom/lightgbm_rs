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

use lgbm_core::types::K_EPSILON;
use lgbm_dataset::MissingType;

use crate::error::ComputeError;
use crate::gain::{
    calculate_splitted_leaf_output, calculate_splitted_leaf_output_smoothed,
    get_leaf_gain_given_output, get_leaf_gain_smoothed, get_split_gains,
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
        let skip = (rev && skip_def && (num_bin - 1 - t) == default_bin)
            || (!rev && skip_def && (t + mfb_offset) == default_bin);
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
