//! On-device tree-growth driver seam — the ADDITIVE feature/bin metadata the
//! per-leaf grow loop consumes, expressed in ONLY lgbm-compute-reachable types.
//!
//! ## Why this struct lives HERE (not in lgbm-treelearner)
//! The learner's per-feature spine column carries the same bin layout, but it lives
//! in `lgbm-treelearner`, which depends on `lgbm-compute`. Naming that learner type
//! from `lgbm-compute` (so [`crate::Backend::grow_tree_on_device`] could take it)
//! would form the crate cycle `treelearner → compute → treelearner`. Instead
//! [`GrowFeature`] is a faithful, lgbm-compute-local MIRROR of exactly the fields
//! the device kernels read — built from types that are ALREADY reachable below
//! `lgbm-compute`:
//! - [`BinColumn`] — lgbm-compute-local (defined in `lib.rs`).
//! - [`BinType`] / [`MissingType`] — `lgbm-dataset` (a dependency of `lgbm-compute`).
//! - primitive slices (`u32`/`i32`/`f64`).
//!
//! The learner builds a `Vec<GrowFeature>` from its `Vec` of spine columns
//! field-by-field at the on-device fork and passes `&grow_features` across the seam.
//!
//! Additive and OFF by default behind `LGBM_CUDA_ON_DEVICE`; ungated like the other
//! on-device kernel modules (NOT `#[cfg(feature = "gpu")]`) so the default cpu f64
//! anchor exercises the plumbing.

use lgbm_dataset::{BinType, LeafPartitionLayout, MissingType};

use crate::error::ComputeError;
use crate::gain::{calculate_splitted_leaf_output, GainConfig, SplitInfo};
use crate::kernels::categorical_split::{
    construct_bitset, construct_inner_bitset, find_best_threshold_categorical,
};
use crate::kernels::data_partition::{
    partition_categorical_on_device, partition_leaf_stable, partition_leaf_stable_fused,
    update_data_index_to_leaf_on,
};
use crate::kernels::histogram::construct_histograms_f64_on;
use crate::kernels::split::{find_best_split_f64_on, BatchedSplitFeature};
use crate::kernels::split_info::{DeviceSplitInfo, SplitScalars, MAX_CAT_PER_SPLIT};
use crate::kernels::subtract::subtract_histograms_f64_on;
use crate::kernels::tree::DeviceCudaTree;
use crate::BinColumn;
use lgbm_core::types::K_EPSILON;

use std::sync::atomic::{AtomicU64, AtomicU8, Ordering};
use std::sync::OnceLock;

/// Compute-owned ON-DEVICE launch counter. The on-device grow driver bumps
/// this once per REAL device dispatch so an on-device train can report an HONEST,
/// NON-ZERO `device_launches` figure through the `phase_prof` COUNTS line — even though
/// `phase_prof` lives in `lgbm-treelearner` (ABOVE `lgbm-compute` in the crate DAG) and
/// cannot be imported here without a crate cycle. `phase_prof::dump` reads/swaps it via
/// [`on_device_launch_count_take`] (the learner depends on compute, so it may reference
/// this symbol).
///
/// **Real-dispatch contract:** on the RESIDENT fast arm this counts REAL device dispatches
/// — exactly one bump per launcher actually issued: the per-train `upload_resident_bins`,
/// each `build_fix_scan_resident` / `build_resident_leaf`, each `subtract_resident`, each
/// `scan_resident_leaf` / `scan_resident_siblings`, and each on-device
/// `data_partition_native`. A CO-PACKED sibling scan (`scan_resident_siblings`) scans BOTH
/// children in ONE launch and therefore counts as exactly ONE dispatch; the HOST partition
/// route issues NO device dispatch (0 bumps). This makes `device_launches=` an honest
/// per-dispatch total, and it lets the launch-count test detect a per-FEATURE non-collapse
/// regression — a per-feature layout would scale the build+scan terms with `num_features`
/// and blow past the analytic real-dispatch bound.
///
/// The NON-resident ANCHOR arm keeps its coarser per-leaf-phase bumps (`build_leaf_hist` /
/// subtract / `scan_leaf`, once per leaf regardless of feature count): it is the parity
/// anchor, not the perf A/B path, and its count stays `num_features`-independent by the
/// same launch-collapse property the test asserts.
pub static ON_DEVICE_LAUNCH_CNT: AtomicU64 = AtomicU64::new(0);

/// Read-once `LGBM_PHASE_PROF=="1"` gate (mirrors `phase_prof::enabled()`), so the
/// launch counter is INERT and zero-overhead in the default merge gate — the bump
/// never touches tree structure or values, keeping the on-device path parity-neutral
/// and byte-unchanged.
///
/// This is a deliberate verbatim TWIN of `lgbm_treelearner::phase_prof::enabled()`
/// — the two crates cannot share a helper without a crate cycle (`phase_prof` lives
/// ABOVE `lgbm-compute` in the DAG). If the env interpretation ever changes here (e.g.
/// accepting `"true"`), update the canonical twin `phase_prof::enabled()` in lockstep,
/// and vice-versa, so the two independent process caches never diverge.
fn launch_prof_enabled() -> bool {
    static E: OnceLock<bool> = OnceLock::new();
    *E.get_or_init(|| std::env::var("LGBM_PHASE_PROF").map(|v| v == "1").unwrap_or(false))
}

/// Bump the on-device launch counter by ONE real device dispatch: on the resident
/// arm one `upload_resident_bins` / `build_fix_scan_resident` / `build_resident_leaf` /
/// `subtract_resident` / `scan_resident_leaf` / `scan_resident_siblings` /
/// `data_partition_native` call actually issued (a co-packed sibling scan is ONE dispatch);
/// on the anchor arm one per-leaf `build_leaf_hist` / subtract / `scan_leaf`. No-op unless
/// `LGBM_PHASE_PROF=="1"` (parity-neutral in the default build/tests).
#[inline]
fn bump_launch() {
    if launch_prof_enabled() {
        ON_DEVICE_LAUNCH_CNT.fetch_add(1, Ordering::Relaxed);
    }
}

/// Swap the accumulated on-device launch count to zero and return the prior value.
/// Called by `phase_prof::dump` (in `lgbm-treelearner`) to fold the on-device launch
/// total into the `device_launches=` COUNTS field without a crate cycle.
pub fn on_device_launch_count_take() -> u64 {
    ON_DEVICE_LAUNCH_CNT.swap(0, Ordering::Relaxed)
}

/// The BLOCKING-READBACK sync counter, DISTINCT from
/// [`ON_DEVICE_LAUNCH_CNT`]. Where the launch counter counts every real device DISPATCH
/// (upload / build / subtract / scan / partition), this counts ONLY the real blocking
/// device→host READBACKS (syncs) — the point where the host waits for a device result:
/// each per-leaf scan readback, each on-device `data_partition_native` readback, and each
/// `tree.split_on_device` `right_leaf_index` readback. Builds / subtracts / uploads issue
/// NO readback (they stay resident) and do NOT bump this.
///
/// Moving the argmax and row-permutation work onto the device removes these host syncs,
/// and this counter is what proves the drop. It counts REAL syncs — a co-packed sibling
/// scan reads BOTH children back in ONE readback and bumps EXACTLY ONCE — so it can never
/// fall into a per-leaf counter trap. Drained by [`on_device_sync_count_take`] (folded
/// into the `phase_prof` COUNTS line without a crate cycle). Inert unless
/// `LGBM_PHASE_PROF=="1"` (parity-neutral).
pub static ON_DEVICE_SYNC_CNT: AtomicU64 = AtomicU64::new(0);

/// Bump the blocking-readback sync counter by ONE real device→host sync (see
/// [`ON_DEVICE_SYNC_CNT`]): one per scan readback / on-device partition readback /
/// `tree.split_on_device` readback. A co-packed sibling scan is ONE sync (both children in
/// one readback). No-op unless `LGBM_PHASE_PROF=="1"` (parity-neutral).
#[inline]
fn bump_sync() {
    if launch_prof_enabled() {
        ON_DEVICE_SYNC_CNT.fetch_add(1, Ordering::Relaxed);
    }
}

/// Swap the accumulated blocking-readback sync count to zero and return the prior value.
/// Called by `phase_prof::dump` (folds the total into the `blocking_readbacks(syncs)=`
/// COUNTS field) and by the `on_device_sync_count` wiring test — no crate cycle (the
/// learner depends on compute, so it may reference this symbol).
pub fn on_device_sync_count_take() -> u64 {
    ON_DEVICE_SYNC_CNT.swap(0, Ordering::Relaxed)
}

/// ROOT + directly-built-child PARALLEL-u64 build tripwire. Bumped ONCE at each of the
/// two build sites that use the parallel u64 fixed-point resident build
/// (`build_resident_leaf` → `build_fix_compact_resident_f64_on` →
/// `resident_raw_build_into(fixed_point=true)` → `construct_leaf_hist_resident_lds_kernel_u64`)
/// instead of the slower f64 single-owner fused build (`build_fix_scan_resident_f64_on`):
/// the ROOT ([`grow_tree_on_device_resident`]'s slot-0 build) and each directly-built
/// (co-pack-OFF) smaller child.
///
/// This counter is DISTINCT from the subtract-path smaller child (the co-pack-ON
/// `build_resident_leaf` at the sibling co-pack site), which was already parallel-u64.
/// Scoping the bump to the root + directly-built arm makes a NONZERO count here sufficient
/// proof that those two sites are actually running the parallel-u64 build rather than the
/// f64 single-owner kernel. Read/reset by the launch-count test AND folded into the
/// `phase_prof` COUNTS line. Inert unless `LGBM_PHASE_PROF=="1"` (parity-neutral).
pub static ON_DEVICE_ROOTBUILD_U64_CNT: AtomicU64 = AtomicU64::new(0);

/// NEGATIVE guard — bumped iff the on-device driver dispatches the f64 single-owner fused
/// build (`Backend::build_fix_scan_resident` → `build_fix_scan_resident_f64_on`), which is
/// reachable ONLY via the `LGBM_ONDEVICE_F64_FUSED=1` A/B escape hatch
/// ([`on_device_f64_fused_build`]). It MUST stay 0 on the DEFAULT (parallel-u64) path; the
/// launch-count test asserts `== 0` so the slower f64 kernel can never silently become the
/// default on-device build again. Only the DRIVER bumps this, so it can never be polluted
/// by the host-learner fused path that legitimately keeps using `build_fix_scan_resident`.
/// Inert unless `LGBM_PHASE_PROF=="1"`.
pub static ON_DEVICE_F64_FUSED_CNT: AtomicU64 = AtomicU64::new(0);

/// Bump the converted-site parallel-u64 build tripwire (see [`ON_DEVICE_ROOTBUILD_U64_CNT`]).
#[inline]
fn bump_rootbuild_u64() {
    if launch_prof_enabled() {
        ON_DEVICE_ROOTBUILD_U64_CNT.fetch_add(1, Ordering::Relaxed);
    }
}

/// Bump the f64-single-owner-fused negative guard (see [`ON_DEVICE_F64_FUSED_CNT`]).
#[inline]
fn bump_f64_fused() {
    if launch_prof_enabled() {
        ON_DEVICE_F64_FUSED_CNT.fetch_add(1, Ordering::Relaxed);
    }
}

/// Swap the converted-site u64-build tripwire to zero and return the prior value. Called by
/// `phase_prof::dump` (folds into the COUNTS line) and by the launch-count test (POSITIVE
/// assertion, `> 0`).
pub fn on_device_rootbuild_u64_count_take() -> u64 {
    ON_DEVICE_ROOTBUILD_U64_CNT.swap(0, Ordering::Relaxed)
}

/// Swap the f64-fused negative-guard counter to zero and return the prior value. Called by
/// the launch-count test (NEGATIVE assertion, `== 0` after the swap).
pub fn on_device_f64_fused_count_take() -> u64 {
    ON_DEVICE_F64_FUSED_CNT.swap(0, Ordering::Relaxed)
}

// ---- The on-device growth-loop PHASE LEDGER (env-gated, parity-neutral). ----
//
// The on-device-vs-host-cuda residual is largely UNATTRIBUTED inside
// [`grow_tree_on_device_resident`] — `phase_prof`'s growth-phase guards all read 0 on this
// path (`in_learner_other=100%` BY DESIGN), so the loop is a black box. These counters give
// the loop its own wall-clock ledger: every ns of the grow wall lands in exactly ONE bucket,
// and `phase_prof::dump` prints the breakdown + the `host_other` complement (wall − Σ).
//
// TIMING SEMANTICS: build / subtract / tree-split / frontier-reduce are ASYNC submissions —
// in free-run mode their bucket holds host SUBMISSION time only, and their device time
// drains inside the next BLOCKING bucket (scan / pick / device-route partition). The
// free-run ledger is therefore a TRUE wall decomposition but not a device-time attribution.
// `LGBM_GROW_DRAIN=1` (diagnostic, with LGBM_PHASE_PROF=1) blocks the queue empty inside
// each async phase's own timer (`grow_drain`) so device time lands in its own bucket — at
// the cost of changing the schedule (drain numbers rank phases; free-run numbers price the
// wall).
//
// Inert unless `LGBM_PHASE_PROF=="1"` (parity-neutral — same contract as every counter
// above).
pub static GROW_WALL_NS: AtomicU64 = AtomicU64::new(0);
pub static GROW_SETUP_NS: AtomicU64 = AtomicU64::new(0);
pub static GROW_UPLOAD_NS: AtomicU64 = AtomicU64::new(0);
pub static GROW_ROOTFOLD_NS: AtomicU64 = AtomicU64::new(0);
pub static GROW_BUILD_NS: AtomicU64 = AtomicU64::new(0);
pub static GROW_SUBTRACT_NS: AtomicU64 = AtomicU64::new(0);
pub static GROW_SCAN_NS: AtomicU64 = AtomicU64::new(0);
pub static GROW_PICK_NS: AtomicU64 = AtomicU64::new(0);
pub static GROW_PARTITION_NS: AtomicU64 = AtomicU64::new(0);
pub static GROW_TREESPLIT_NS: AtomicU64 = AtomicU64::new(0);
pub static GROW_REDUCE_NS: AtomicU64 = AtomicU64::new(0);
pub static GROW_TAIL_NS: AtomicU64 = AtomicU64::new(0);

/// One drained snapshot of the growth-loop phase ledger (all fields ns).
/// `host_other` is NOT stored — the consumer computes `wall − Σ(components)` at dump time.
#[derive(Debug, Clone, Copy, Default)]
pub struct GrowPhaseNs {
    pub wall: u64,
    pub setup: u64,
    pub upload: u64,
    pub rootfold: u64,
    pub build: u64,
    pub subtract: u64,
    pub scan: u64,
    pub pick: u64,
    pub partition: u64,
    pub treesplit: u64,
    pub reduce: u64,
    pub tail: u64,
}

/// Swap the whole growth-loop phase ledger to zero and return the prior values.
/// Called by `phase_prof::dump` (no crate cycle — the learner depends on compute).
pub fn on_device_grow_phase_take() -> GrowPhaseNs {
    GrowPhaseNs {
        wall: GROW_WALL_NS.swap(0, Ordering::Relaxed),
        setup: GROW_SETUP_NS.swap(0, Ordering::Relaxed),
        upload: GROW_UPLOAD_NS.swap(0, Ordering::Relaxed),
        rootfold: GROW_ROOTFOLD_NS.swap(0, Ordering::Relaxed),
        build: GROW_BUILD_NS.swap(0, Ordering::Relaxed),
        subtract: GROW_SUBTRACT_NS.swap(0, Ordering::Relaxed),
        scan: GROW_SCAN_NS.swap(0, Ordering::Relaxed),
        pick: GROW_PICK_NS.swap(0, Ordering::Relaxed),
        partition: GROW_PARTITION_NS.swap(0, Ordering::Relaxed),
        treesplit: GROW_TREESPLIT_NS.swap(0, Ordering::Relaxed),
        reduce: GROW_REDUCE_NS.swap(0, Ordering::Relaxed),
        tail: GROW_TAIL_NS.swap(0, Ordering::Relaxed),
    }
}

/// Time `f` into `c` (ns). Zero-overhead passthrough when the gate is off.
#[inline]
fn time_phase<T>(c: &'static AtomicU64, f: impl FnOnce() -> T) -> T {
    if launch_prof_enabled() {
        let t = std::time::Instant::now();
        let r = f();
        c.fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
        r
    } else {
        f()
    }
}

/// RAII phase timer for spans with `?` early-exits (accumulates on drop, so a
/// propagated error still books the partial span). Inert when the gate is off.
struct PhaseGuard {
    c: &'static AtomicU64,
    t: std::time::Instant,
    on: bool,
}
impl PhaseGuard {
    fn new(c: &'static AtomicU64) -> Self {
        Self { c, t: std::time::Instant::now(), on: launch_prof_enabled() }
    }
}
impl Drop for PhaseGuard {
    fn drop(&mut self) {
        if self.on {
            self.c.fetch_add(self.t.elapsed().as_nanos() as u64, Ordering::Relaxed);
        }
    }
}

/// Read-once `LGBM_GROW_DRAIN=="1"` — the de-alias diagnostic mode (see the
/// ledger header comment). Meaningful only together with `LGBM_PHASE_PROF=1`.
fn grow_drain_enabled() -> bool {
    static E: OnceLock<bool> = OnceLock::new();
    *E.get_or_init(|| std::env::var("LGBM_GROW_DRAIN").map(|v| v == "1").unwrap_or(false))
}

/// In drain mode, block until the device queue is EMPTY so the just-submitted async work's
/// device time is booked into the CURRENT phase timer instead of aliasing into the next
/// blocking readback. No-op unless BOTH gates are on — never active in the default
/// build/tests (parity-neutral; it only waits, never mutates).
#[inline]
fn grow_drain<R: cubecl::Runtime>(client: &cubecl::prelude::ComputeClient<R>) {
    if launch_prof_enabled() && grow_drain_enabled() {
        let _ = cubecl::future::block_on(client.sync());
    }
}

/// Read-once `LGBM_ONDEVICE_BIN_HOIST != "0"` — default ON; `=0` restores the per-grow
/// re-upload for A/B comparison.
///
/// The hoist skips the driver's per-grow `upload_resident_bins` ONLY when the backend
/// reports a PINNED, geometry-matching resident-bin cache
/// ([`crate::Backend::resident_bins_pinned`]) — the pin is set exclusively by the
/// learner's once-per-train guarded upload (`resident_bins_uploaded`) and dissolved by
/// any fresh upload, so un-pinned direct-driver callers are byte-unchanged. Re-uploading
/// the immutable bin columns every grow re-crosses PCIe on every tree, so skipping the
/// re-upload once it is already resident and pinned removes a substantial, otherwise
/// avoidable, on-device cost.
fn ondevice_bin_hoist_enabled() -> bool {
    static E: OnceLock<bool> = OnceLock::new();
    *E.get_or_init(|| {
        std::env::var("LGBM_ONDEVICE_BIN_HOIST").map(|v| v != "0").unwrap_or(true)
    })
}

/// Read-once `LGBM_GRAD_RESIDENCY != "0"` — default ON; `=0` restores the per-iteration
/// label f32→f64 convert + host→device label upload + fresh grad/hess device allocs
/// (the pre-residency `get_gradients_resident_on` pattern) for same-session A/B
/// comparison. Bit-exact either way (same kernel, same launch geometry — only buffer
/// provenance differs).
#[must_use]
pub fn grad_residency_enabled() -> bool {
    static E: OnceLock<bool> = OnceLock::new();
    *E.get_or_init(|| std::env::var("LGBM_GRAD_RESIDENCY").map(|v| v != "0").unwrap_or(true))
}

/// Read-once `LGBM_SCORE_FUSED_SCATTER != "0"` — default ON; `=0` restores the
/// two-kernel `derive_leaf_map_device_handle` → `add_leaf_values_to_resident_score`
/// per-tree resident score update (single-active-warp derive + `num_data`-length
/// `-1`-map fill upload) for same-session A/B comparison. Bit-exact either way (each
/// row written exactly once with the same f64 `+=` of the same leaf value).
#[must_use]
pub fn score_fused_scatter_enabled() -> bool {
    static E: OnceLock<bool> = OnceLock::new();
    *E.get_or_init(|| {
        std::env::var("LGBM_SCORE_FUSED_SCATTER").map(|v| v != "0").unwrap_or(true)
    })
}

/// Read-once `LGBM_ONDEVICE_FUSED_PARTITION != "0"` — default ON (mirroring
/// [`ondevice_bin_hoist_enabled`]). When ON, the host arm of [`partition_resident_range`]
/// routes through [`partition_leaf_stable_fused`] (one fused pass, no `bins_sub` alloc);
/// `=0` restores the pre-change `bins_sub` gather + [`partition_leaf_stable`] path for A/B
/// comparison of the partition variant. Byte-equal either way (both fold through the same
/// `RouteFlags`/`route_left_host` machinery).
fn ondevice_fused_partition_enabled() -> bool {
    // Same-session A/B override. The env gate below is read-once (`OnceLock`), so it
    // CANNOT be toggled between arms in one process — a same-session A/B harness that
    // flips arms within one process would silently keep the first arm's frozen value.
    // This atomic lets the harness flip the partition variant per arm WITHOUT a `getenv`
    // on the per-split hot path (a `Relaxed` load is timing-neutral for the `partition`
    // bucket it measures). 0 = defer to the env; 1 = force ON; 2 = force OFF. Default 0 ⇒
    // production behaviour is byte-identical to the pre-override read-once env gate.
    match FUSED_PARTITION_OVERRIDE.load(Ordering::Relaxed) {
        1 => return true,
        2 => return false,
        _ => {}
    }
    static E: OnceLock<bool> = OnceLock::new();
    *E.get_or_init(|| {
        std::env::var("LGBM_ONDEVICE_FUSED_PARTITION").map(|v| v != "0").unwrap_or(true)
    })
}

/// Same-session A/B override for [`ondevice_fused_partition_enabled`].
/// 0 = unset (defer to `LGBM_ONDEVICE_FUSED_PARTITION`), 1 = force ON, 2 = force OFF.
static FUSED_PARTITION_OVERRIDE: AtomicU8 = AtomicU8::new(0);

/// Harness/test hook: force the fused-partition host arm ON (`Some(true)`), OFF
/// (`Some(false)`), or defer to the `LGBM_ONDEVICE_FUSED_PARTITION` env gate (`None`).
/// Exists solely so a same-session A/B harness can toggle the partition variant between
/// arms in ONE process — the env gate is read-once (`OnceLock`) and cannot be re-read
/// mid-process. Timing-neutral (an `AtomicU8` `Relaxed` store; the per-split read is a
/// `Relaxed` load, not a `getenv`). The default (never called) leaves production on the
/// pure env path.
pub fn set_fused_partition_override(v: Option<bool>) {
    let code = match v {
        None => 0,
        Some(true) => 1,
        Some(false) => 2,
    };
    FUSED_PARTITION_OVERRIDE.store(code, Ordering::Relaxed);
}

/// A/B escape hatch — read-once `LGBM_ONDEVICE_F64_FUSED=="1"`.
///
/// DEFAULT (unset/`!= "1"`): the on-device ROOT + directly-built resident histogram BUILD
/// runs on the PARALLEL u64 fixed-point kernel, which replaced the f64 single-owner
/// `CubeDim::new_1d(1)` row-fold kernel measured to be substantially slower on real
/// consumer NVIDIA hardware. `LGBM_ONDEVICE_F64_FUSED=1` restores the OLD f64 single-owner
/// fused build+scan at those two sites so a real-CUDA A/B can quantify the u64-vs-f64
/// difference side-by-side. It is NEVER the default — the hard constraint is "do NOT
/// default the on-device path to the f64 fused kernel."
fn on_device_f64_fused_build() -> bool {
    static E: OnceLock<bool> = OnceLock::new();
    *E.get_or_init(|| {
        std::env::var("LGBM_ONDEVICE_F64_FUSED").map(|v| v == "1").unwrap_or(false)
    })
}

/// One feature column's ADDITIVE grow-loop input — the faithful lgbm-compute-local
/// mirror of the fields the learner's spine feature column exposes to the device
/// kernels, using ONLY lgbm-compute-reachable types so the seam never names a
/// treelearner type (no crate cycle).
///
/// Field-for-field parity with the learner's spine column, including the categorical
/// metadata (`bin_to_category` + the five categorical config scalars) the §6.3 bitset
/// construction and §8.1 evaluator consume for the categorical grow branch. Every field
/// is a plain value / `lgbm-dataset` enum / narrow [`BinColumn`] / native primitive
/// (`Vec<i32>`/`f64`/`i32`) — nothing here reaches up into `lgbm-treelearner` (no crate
/// cycle).
#[derive(Debug, Clone)]
pub struct GrowFeature {
    /// Per-GLOBAL-ROW bin index, length `num_data`, in the narrowest unsigned type
    /// for `num_bin` (mirrors the spine column's `bins`). lgbm-compute-local.
    pub bins: BinColumn,
    /// C++ `num_bin` — this feature's bin count (histogram has `2*num_bin` cells).
    pub num_bin: u32,
    /// C++ threshold-offset descriptor (`meta_->offset`), from
    /// `offset_for_most_freq_bin` at the boundary.
    pub offset: i32,
    /// C++ `min_bin` — the feature's first bin (partition lower bound).
    pub min_bin: u32,
    /// C++ `max_bin` — the feature's last bin (partition upper bound).
    pub max_bin: u32,
    /// C++ `default_bin_` (`ValueToBin(0)`) — drives the SKIP_DEFAULT_BIN continue.
    pub default_bin: u32,
    /// C++ `most_freq_bin_` — drives `FixHistogram` + the partition default dir.
    pub most_freq_bin: u32,
    /// C++ `missing_type_` — derives `skip_default_bin` / `na_as_missing` dispatch.
    pub missing_type: MissingType,
    /// Real-value per-bin upper bounds (`bin_upper_bound_`) — the split threshold
    /// the tree records (`threshold[bin] == bin_upper_bound_[bin]`).
    pub bin_upper_bound: Vec<f64>,
    /// The ORIGINAL feature index (`real_feature_idx_`) the tree records + predict
    /// traverses.
    pub real_feature_index: i32,
    /// C++ `BinMapper::bin_type()` — numeric vs categorical dispatch flag.
    pub bin_type: BinType,
    /// C++ `BinMapper::bin_2_categorical_` — bin index → ORIGINAL category value
    /// (`BinToValue(bin)`, bin.h:138-143). Populated ONLY for categorical features;
    /// the categorical grow branch converts each winning REAL BIN to its
    /// category value to build the model-text (`cat_threshold`) bitset via
    /// SetRealThreshold. Empty (`Vec::new()`) for numeric features — inert on the
    /// numeric grow path. Native `Vec<i32>` (no crate cycle).
    pub bin_to_category: Vec<i32>,
    /// `double cat_smooth` (config default 10.0) — categorical CTR smoothing +
    /// the many-vs-many filter. Inert on the numeric path (§8.1).
    pub cat_smooth: f64,
    /// `double cat_l2` (config default 10.0) — extra l2 ADDED to lambda_l2 in the
    /// per-category gain (NOT the `gain_shift` baseline). Inert on the numeric path.
    pub cat_l2: f64,
    /// `int max_cat_threshold` (config default 32) — many-vs-many cap on the number
    /// of categories on one side. Inert on the numeric path.
    pub max_cat_threshold: i32,
    /// `int max_cat_to_onehot` (config default 4) — categorical features with
    /// `num_bin <= max_cat_to_onehot` use the one-hot (one-vs-rest) path. Inert on
    /// the numeric path.
    pub max_cat_to_onehot: i32,
    /// `int min_data_per_group` (config default 100) — many-vs-many minimum rows per
    /// accumulated category group. Inert on the numeric path.
    pub min_data_per_group: i32,
}

// =========================================================================
// data->leaf map buffer-strategy A/B harness.
//
// The per-split `UpdateDataIndexToLeafIndex` rewrite reads the row->leaf map for
// the split leaf's rows and writes the two child leaf ids. When the driver grows
// num_leaves>2 it applies this rewrite REPEATEDLY over one running map. The open
// aliasing question: does the driver read+write ONE map buffer in place (ALIAS), or
// read a source buffer and write a distinct destination then swap (DOUBLE-BUFFER)? A
// wrong alias choice silently corrupts the partition at num_leaves>2. This helper
// exposes BOTH so an A/B harness can anchor each to the cpu f64 partition and lock in
// the safe strategy (double-buffer unless alias is proven bit-identical). Each step
// drives the REAL device kernel (`update_data_index_to_leaf_on`); the strategies
// differ ONLY in how the running map buffer is carried across steps.
//
// This is a DECISION-RECORD A/B harness, NOT live driver plumbing. The shipped
// driver (`grow_tree_on_device_driver_with_cfg`) carries NO running leaf-map buffer
// and calls NEITHER strategy — it partitions each leaf into a fresh `Vec<u32>` via
// `partition_leaf_stable`. `build_leaf_map_on` / `LeafMapBufferStrategy` /
// `LeafMapStep` remain `pub` ONLY because the A/B oracle lives in a separate crate
// (`oracle-harness`) and must reach them; they are not consumed by any production
// path. Read the "LOCK" language above as "recorded the A/B conclusion", not "the
// driver applies this strategy".
// =========================================================================

/// The data->leaf map buffer strategy for the per-split rewrite.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LeafMapBufferStrategy {
    /// (A) In-place alias — one running map buffer read AND written each step.
    Alias,
    /// (B) Ping-pong double-buffer — read the source map, write a distinct
    /// destination, swap. The conservative default (no read/write aliasing).
    DoubleBuffer,
}

/// One split's row->leaf rewrite input for [`build_leaf_map_on`]: the global row
/// ids currently in the leaf being split, their `to_left` marks (1 = routes to
/// `left_leaf`, 0 = routes to `right_leaf`, aligned to `data_indices`), and the two
/// child leaf ids.
#[derive(Clone, Copy, Debug)]
pub struct LeafMapStep<'a> {
    /// Global row ids in the leaf being split.
    pub data_indices: &'a [u32],
    /// Per-`data_indices` route marks (`1` = left child, `0` = right child).
    pub to_left: &'a [u32],
    /// The left child leaf id.
    pub left_leaf: i32,
    /// The right child leaf id.
    pub right_leaf: i32,
}

/// Apply `steps` in order to build the final `num_data`-length row->leaf map,
/// starting from `init_leaf` for every row, using the chosen buffer `strategy`.
/// Each step drives the real
/// [`update_data_index_to_leaf_on`] device kernel (which writes the two child leaf
/// ids for the leaf's rows into a fresh `-1` map); the running map is then carried
/// forward either in place ([`LeafMapBufferStrategy::Alias`]) or via a swapped
/// destination copy ([`LeafMapBufferStrategy::DoubleBuffer`]). Rows a step does not
/// touch keep their prior leaf id. Both strategies MUST equal the cpu f64 partition
/// anchor — the A/B proves it and locks the safe one.
///
/// # Errors
/// [`ComputeError`] from [`update_data_index_to_leaf_on`] (length mismatch, or a
/// `data_index >= num_data`).
pub fn build_leaf_map_on<R: cubecl::Runtime>(
    client: &cubecl::prelude::ComputeClient<R>,
    num_data: usize,
    init_leaf: i32,
    steps: &[LeafMapStep<'_>],
    strategy: LeafMapBufferStrategy,
) -> Result<Vec<i32>, ComputeError> {
    let mut running = vec![init_leaf; num_data];
    for step in steps {
        // Real device kernel: a fresh -1 map with ONLY this leaf's rows set
        // to their child leaf id (rest stay -1). This is the read-side result the
        // buffer strategy then folds into the running map.
        let per_split = update_data_index_to_leaf_on(
            client,
            step.data_indices,
            step.to_left,
            num_data,
            step.left_leaf,
            step.right_leaf,
        )?;
        match strategy {
            LeafMapBufferStrategy::Alias => {
                // (A) in-place: read AND write the SAME running buffer.
                for (row, &v) in per_split.iter().enumerate() {
                    if v != -1 {
                        running[row] = v;
                    }
                }
            }
            LeafMapBufferStrategy::DoubleBuffer => {
                // (B) ping-pong: read the source `running`, write a distinct `next`,
                // then swap. No read/write aliasing of a single buffer.
                let mut next = running.clone();
                for (row, &v) in per_split.iter().enumerate() {
                    if v != -1 {
                        next[row] = v;
                    }
                }
                running = next;
            }
        }
    }
    Ok(running)
}

// =========================================================================
// The per-leaf best-first on-device grow DRIVER.
//
// This is the load-bearing STRUCTURE-bit-exact gate: it grows an ENTIRE
// continuous-feature + L2 tree by SEQUENCING the already-golden histogram
// build + subtract, best-split finder, and `DeviceCudaTree` mutation kernels into
// the C++ `SerialTreeLearner` §6/§16 best-first order — WITHOUT reusing
// lgbm-treelearner's `LeafSplits` / `HistogramPool` / `DataPartition` (those cannot
// be named from lgbm-compute: the crate wall this module exists to keep). It
// reproduces the host order with its OWN lightweight [`DriverLeaf`] bookkeeping.
//
// ## Faithfulness scope (proving slice)
// Continuous features + L2 only: NO categorical, NO L1/smoothing/max_delta_step,
// NO RenewTreeOutput refit, NO col-sampler / monotone / interaction / extra-trees /
// CEGB / forced-splits. The proving corpus is `MissingType::None` (single reverse
// scan). The L1/quantile/categorical follow-up reuses this exact ordering contract
// but adds the missing-value forward preamble + the categorical split kernel.
//
// ## No f64 per-row grow/build hot loop
// The only per-ROW device work is the histogram BUILD
// ([`construct_histograms_f64_on`]) and the row PARTITION
// ([`partition_leaf_stable`]) — both operate on integer bins / f32 grad-hess and
// keep the u64/f32 build contract inside the kernel. The f64 that appears here is
// confined to O(num_bin) per-feature histogram post-processing (FixHistogram +
// compaction), the O(num_bin) subtraction, and the reference-blessed scalar
// gain/leaf-value math — NONE of it is a per-row loop.
// =========================================================================

/// The fixed L2 proving-slice gain config — **TEST / PROVING-SLICE ONLY**.
///
/// Continuous + L2 with a PERMISSIVE `min_data_in_leaf = 1` / `min_sum_hessian_in_leaf = 0.0`
/// (admissibility effectively OFF). This is the config the parameterless
/// [`grow_tree_on_device_driver`] / [`crate::Backend::grow_tree_on_device`] seam pins so the
/// STRUCTURE/anchor gates (`on_device_integer_anchor`, tie-break/sync-count) build the cpu-f64
/// anchor with the IDENTICAL config on both arms.
///
/// # No production caller
/// The PRODUCTION on-device path (`SerialTreeLearner` → [`crate::Backend::grow_tree_on_device_with_cfg`]
/// → [`grow_tree_on_device_driver_with_cfg`]) binds the learner's REAL `GainConfig` instead —
/// NEVER this permissive config. Pinning it in production disables C++ admissibility and can let a
/// near-empty prefix win with an `inf`-magnitude gain. Do NOT re-adopt this on any production
/// grow path; it exists solely to keep the anchor fixtures' two arms identical.
#[must_use]
pub fn proving_slice_config() -> GainConfig {
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

/// The driver's OWN minimal per-leaf state — the lgbm-compute-local stand-in for
/// the learner's `LeafSplits` + `HistogramPool` slot + `best_split_per_leaf`, using
/// nothing above lgbm-compute (no crate cycle).
struct DriverLeaf {
    /// Global row ids currently in this leaf (partition order).
    rows: Vec<u32>,
    /// The leaf's seeded gradient sum (root = ordered f64 fold; child = the parent
    /// split's `left/right_sum_gradient`, kEpsilon-carrying — NOT a re-fold).
    sum_g: f64,
    /// The leaf's seeded hessian sum.
    sum_h: f64,
    /// This leaf's per-feature CONCATENATED fixed+compacted histogram (the parent
    /// buffer the subtraction trick derives the larger child from).
    hist: Vec<f64>,
    /// The leaf's best split (`gain == -inf` ⇒ no admissible split).
    best: SplitInfo,
    /// The winning feature POSITION (`-1` when no split); its real index is
    /// `features[best_fpos].real_feature_index`.
    best_fpos: i32,
    /// When the winning split is CATEGORICAL, the winning category bins
    /// (`output->cat_threshold`, each `+ offset`) the driver body stages into the
    /// pre-allocated `DeviceSplitInfo` cat slab. Empty for a numeric win.
    best_cat: Vec<u32>,
    /// The leaf's depth (root = 0), for the `max_depth` gate.
    depth: i32,
}

/// The SINGLE `+2*kEpsilon` categorical `sum_hessian` bump, mirroring the host
/// call site `learner.rs:2760`. The `best_split.rs` dispatch seam PASSES THROUGH the
/// bumped value and the §8.1 evaluator does NOT bump internally, so the bump happens
/// EXACTLY ONCE — here in the driver. A double-bump or missed-bump is a last-ULP
/// silent divergence, so this is pinned bit-exact by
/// `categorical_driver_bumps_sum_hessian_once`.
#[inline]
fn bump_sum_hessian_cat(sum_h: f64) -> f64 {
    sum_h + 2.0 * f64::from(K_EPSILON)
}

/// Overlay a categorical feature's per-feature config scalars (`cat_l2`,
/// `cat_smooth`, `max_cat_threshold`, `max_cat_to_onehot`, `min_data_per_group`)
/// from its [`GrowFeature`] onto the leaf's base [`GainConfig`] for the §8.1
/// evaluator. The numeric gain knobs (l1/l2/min_data/…) are inherited from `base`.
fn categorical_feature_config(base: &GainConfig, f: &GrowFeature) -> GainConfig {
    let mut c = *base;
    c.cat_l2 = f.cat_l2;
    c.cat_smooth = f.cat_smooth;
    c.max_cat_threshold = f.max_cat_threshold;
    c.max_cat_to_onehot = f.max_cat_to_onehot;
    c.min_data_per_group = f.min_data_per_group;
    c
}

/// `SerialTreeLearner`'s cross-feature / cross-leaf argmax tie rule
/// (`split_info.rs::split_gt`): strictly-greater gain wins; on an exact gain tie the
/// LOWER real feature index wins (`-1` ⇒ `i32::MAX`).
fn split_gt(a: &SplitInfo, a_feat: i32, b: &SplitInfo, b_feat: i32) -> bool {
    if a.gain != b.gain {
        return a.gain > b.gain;
    }
    let af = if a_feat == -1 { i32::MAX } else { a_feat };
    let bf = if b_feat == -1 { i32::MAX } else { b_feat };
    af < bf
}

/// C++ `FixHistogram` on the RAW leaf sums (`feature_histogram` — Pitfall 2). A
/// no-op for `most_freq_bin == 0` (the proving corpus). O(num_bin) scalar f64 fold
/// (ascending bin order is load-bearing — never reorder). NOT a per-row loop.
fn fix_histogram(hist: &mut [f64], most_freq_bin: u32, sum_gradient: f64, sum_hessian: f64) {
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

/// C++ compacted-histogram shift (`offset` drops the leading `offset` bins). A
/// no-op for `offset == 0`. O(num_bin) scalar f64 copy. NOT a per-row loop.
fn compact_histogram(hist: &mut [f64], offset: i32) {
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

/// Build one leaf's per-feature CONCATENATED fixed+compacted histogram by
/// DIRECTLY constructing each feature's raw histogram over the leaf's rows
/// ([`construct_histograms_f64_on`]), then FixHistogram + compacting each
/// feature region in place. `slot_off[fpos]` is the feature's start cell.
fn build_leaf_hist<R: cubecl::Runtime>(
    client: &cubecl::prelude::ComputeClient<R>,
    features: &[GrowFeature],
    gradients: &[f32],
    hessians: &[f32],
    rows: &[u32],
    sum_g: f64,
    sum_h: f64,
    slot_off: &[usize],
    hist_len: usize,
) -> Result<Vec<f64>, ComputeError> {
    let mut concat = vec![0.0f64; hist_len];
    if rows.is_empty() {
        return Ok(concat);
    }
    // Gather the leaf's ordered grad/hess ONCE (shared across features).
    let g: Vec<f32> = rows.iter().map(|&r| gradients[r as usize]).collect();
    let h: Vec<f32> = rows.iter().map(|&r| hessians[r as usize]).collect();
    // One leaf-level build launch (bumped ONCE per leaf, NOT per feature,
    // to match the host per-leaf `BUILD_RESIDENT_CNT` unit `phase_prof::dump` sums with).
    bump_launch();
    for (fpos, f) in features.iter().enumerate() {
        let binned: Vec<u32> = rows.iter().map(|&r| f.bins.bin(r as usize)).collect();
        // Device build: RAW f64 histogram (2*num_bin cells).
        let mut region = construct_histograms_f64_on(client, &binned, &g, &h, f.num_bin)?;
        // FixHistogram (RAW leaf sums) then compact — O(num_bin) f64, bit-exact to
        // the host reference fold (mfb==0 ⇒ fix is a no-op; offset==1 ⇒ drop bin 0).
        fix_histogram(&mut region, f.most_freq_bin, sum_g, sum_h);
        compact_histogram(&mut region, f.offset);
        let cells = 2 * f.num_bin as usize;
        concat[slot_off[fpos]..slot_off[fpos] + cells].copy_from_slice(&region);
    }
    Ok(concat)
}

/// Scan one leaf's concatenated compacted histogram: per-feature best-split eval +
/// the cross-feature `split_gt` argmax. NUMERIC features route into the
/// [`find_best_split_f64_on`] finder; CATEGORICAL features route into the §8.1
/// [`find_best_threshold_categorical`] evaluator. Returns the winning
/// `(SplitInfo, feature-position, cat_bins)` — `cat_bins` are the winning category
/// bins when the winner is categorical (empty otherwise); `(-inf, -1, [])` when
/// nothing is admissible.
///
/// **kEpsilon single-bump site:** for a categorical feature the driver applies
/// the ONE `+2*kEpsilon` `sum_h` bump HERE (via [`bump_sum_hessian_cat`]) before the
/// evaluator sees it, mirroring host `learner.rs:2760`. The numeric finder bumps
/// internally, so numeric passes RAW `sum_h` (unchanged).
fn scan_leaf<R: cubecl::Runtime>(
    client: &cubecl::prelude::ComputeClient<R>,
    features: &[GrowFeature],
    hist: &[f64],
    sum_g: f64,
    sum_h: f64,
    num_data_in_leaf: i32,
    slot_off: &[usize],
    cfg: &GainConfig,
) -> Result<(SplitInfo, i32, Vec<u32>), ComputeError> {
    let mut best = SplitInfo::none();
    let mut best_fpos: i32 = -1;
    let mut best_real: i32 = -1;
    let mut best_cat: Vec<u32> = Vec::new();
    #[allow(clippy::neg_cmp_op_on_partial_ord)]
    if !(sum_h > 0.0) || num_data_in_leaf <= 0 {
        return Ok((best, best_fpos, best_cat));
    }
    // One leaf-level scan launch (bumped ONCE per leaf that is actually
    // scanned, NOT per feature, to match the host per-leaf `SCAN_RESIDENT_CNT` unit).
    bump_launch();
    // The per-feature finders below each read a SplitInfo back to the host —
    // ONE blocking scan readback per SCANNED leaf (not per feature: the count must stay
    // num_features-independent, the counter-trap guard).
    bump_sync();
    for (fpos, f) in features.iter().enumerate() {
        // Proving slice: continuous MissingType::None ⇒ single REVERSE scan, no
        // default-bin skip, no NA forward preamble.
        let skip_default_bin = f.num_bin > 2 && f.missing_type == MissingType::Zero;
        let na_as_missing = f.num_bin > 2 && f.missing_type == MissingType::NaN;
        let run_forward = f.num_bin > 2 && f.missing_type == MissingType::Zero;
        if na_as_missing {
            // NA-as-missing deferred on BOTH numeric and categorical (finder rejects NA).
            continue;
        }
        let cells = 2 * f.num_bin as usize;
        let region = &hist[slot_off[fpos]..slot_off[fpos] + cells];
        let (si, cat_bins) = if f.bin_type == BinType::Categorical {
            // Categorical branch (§8.1). Bump sum_h ONCE here before the
            // evaluator; best_split passes through, the evaluator does not bump.
            let cat_cfg = categorical_feature_config(cfg, f);
            let sum_h_bumped = bump_sum_hessian_cat(sum_h);
            let cat = find_best_threshold_categorical(
                client,
                region,
                &cat_cfg,
                f.num_bin as i32,
                f.offset,
                sum_g,
                sum_h_bumped,
                num_data_in_leaf,
            )?;
            (cat.split, cat.cat_threshold)
        } else {
            let si = find_best_split_f64_on(
                client,
                region,
                cfg,
                f.num_bin,
                f.offset,
                f.default_bin,
                f.most_freq_bin,
                skip_default_bin,
                na_as_missing,
                run_forward,
                sum_g,
                sum_h,
                num_data_in_leaf,
            )?;
            (si, Vec::new())
        };
        if split_gt(&si, f.real_feature_index, &best, best_real) {
            best = si;
            best_fpos = fpos as i32;
            best_real = f.real_feature_index;
            best_cat = cat_bins;
        }
    }
    Ok((best, best_fpos, best_cat))
}

/// Grow an ENTIRE continuous-feature + L2 tree ON-DEVICE by sequencing the device
/// kernels in the `SerialTreeLearner` best-first order, using the driver's OWN
/// [`DriverLeaf`] bookkeeping. Returns the grown [`lgbm_model::Tree`] and the final
/// row→leaf [`LeafPartitionLayout`].
///
/// Thin delegator to [`grow_tree_on_device_driver_with_cfg`] pinned to the fixed
/// [`proving_slice_config`] the parameterless [`crate::Backend::grow_tree_on_device`]
/// trait seam grows under (that seam carries no `GainConfig`, so the STRUCTURE/anchor
/// gates build both arms with the IDENTICAL proving config).
///
/// # NOT the production entry
/// PRODUCTION on-device grows go through [`crate::Backend::grow_tree_on_device_with_cfg`]
/// → [`grow_tree_on_device_driver_with_cfg`], which binds the learner's REAL `GainConfig`.
/// This parameterless delegator pins the PERMISSIVE proving config and is retained ONLY for
/// the anchor fixtures — never call it from a production path (that would disable
/// admissibility checking). Tests that need a constrained gain config call `_with_cfg`.
///
/// Runs on ANY `R` (the STRUCTURE gate drives it on the cubecl-cpu runtime, anchored
/// to the cpu f64 fold — never GPU-vs-GPU). `max_depth <= 0` ⇒ no depth cap.
///
/// Takes `backend: &B` (`B: Backend<Runtime = R>`) — the driver reaches the backend
/// so it can branch on `backend.resident_pool_supported()` and dispatch the
/// resident/batched launchers `GpuBackend<R>` already implements
/// (`upload_resident_bins` / `build_fix_scan_resident` / `subtract_resident` /
/// `find_best_splits_batched` / `data_partition_native`). The driver still names NO
/// `lgbm-treelearner` type (`B` is satisfied by `CpuBackend`/`GpuBackend<R>`, both in
/// `lgbm-compute`), so no crate cycle.
///
/// # Errors
/// [`ComputeError`] from any sequenced kernel (bad num_bin / histogram length,
/// out-of-range bin, device launch), or an empty feature set / non-positive
/// `num_leaves`.
pub fn grow_tree_on_device_driver<B, R>(
    backend: &B,
    client: &cubecl::prelude::ComputeClient<R>,
    gradients: &[f32],
    hessians: &[f32],
    features: &[GrowFeature],
    num_leaves: i32,
    max_depth: i32,
) -> Result<(lgbm_model::Tree, LeafPartitionLayout), ComputeError>
where
    B: crate::Backend<Runtime = R>,
    R: cubecl::Runtime,
{
    grow_tree_on_device_driver_with_cfg(
        backend,
        client,
        gradients,
        hessians,
        features,
        num_leaves,
        max_depth,
        proving_slice_config(),
    )
}

/// Extract-parameter variant of [`grow_tree_on_device_driver`] that grows the tree
/// under an EXPLICIT `cfg: GainConfig` instead of the pinned [`proving_slice_config`].
/// The body is otherwise identical to the delegator (same length/num_leaves guards,
/// ordered-f64 root fold, build/subtract/scan/partition sequence, break path, and
/// typed `ComputeError` boundaries). Additive: the `Backend::grow_tree_on_device`
/// trait seam is untouched, so the default `LGBM_CUDA_ON_DEVICE`-unset merge gate is
/// byte-unchanged. Threading `cfg` here lets a test make a constraint (e.g.
/// `min_data_in_leaf`) observably bind through the driver without widening the seam.
///
/// # Errors
/// [`ComputeError`] from any sequenced kernel (bad num_bin / histogram length,
/// out-of-range bin, device launch), or an empty feature set / non-positive
/// `num_leaves`.
pub fn grow_tree_on_device_driver_with_cfg<B, R>(
    backend: &B,
    client: &cubecl::prelude::ComputeClient<R>,
    gradients: &[f32],
    hessians: &[f32],
    features: &[GrowFeature],
    num_leaves: i32,
    max_depth: i32,
    cfg: GainConfig,
) -> Result<(lgbm_model::Tree, LeafPartitionLayout), ComputeError>
where
    B: crate::Backend<Runtime = R>,
    R: cubecl::Runtime,
{
    if num_leaves < 1 {
        return Err(ComputeError::Runtime {
            detail: format!("grow_tree_on_device_driver_with_cfg: num_leaves must be >= 1, got {num_leaves}"),
        });
    }
    if features.is_empty() {
        return Err(ComputeError::Runtime {
            detail: "grow_tree_on_device_driver_with_cfg: at least one feature is required".to_string(),
        });
    }
    let num_data = gradients.len();
    if hessians.len() != num_data {
        return Err(ComputeError::LengthMismatch {
            expected: num_data,
            actual: hessians.len(),
        });
    }

    // ---- Capability branch. ----
    // A resident-pool-capable backend (`GpuBackend<R>` with a device-resident
    // histogram pool) takes the FAST batched+resident build/subtract/scan path for the
    // NUMERIC proving slice — dispatching `backend`'s `upload_resident_bins` (ONCE per
    // grow), `build_resident_leaf` (root + each smaller child, device-resident into a
    // pool slot), `subtract_resident` (larger child = parent − smaller, on device), and
    // `scan_resident_leaf` (each leaf's splits read from the resident slot). A
    // non-resident backend (`CpuBackend`, the f64 anchor) always folds through the
    // single-owner anchor kernels below — byte-for-byte unchanged. CATEGORICAL
    // features demote to the anchor fold (`on_device_eligible` routes categorical+quantized
    // to the host path anyway); the resident arm is the numeric proving slice only.
    if backend.resident_pool_supported()
        && !features.iter().any(|f| f.bin_type == BinType::Categorical)
    {
        return grow_tree_on_device_resident(
            backend, client, gradients, hessians, features, num_leaves, max_depth, &cfg,
        );
    }

    let min_data = cfg.min_data_in_leaf;

    // Per-feature concatenated-histogram offsets (2*num_bin cells each).
    let mut slot_off = Vec::with_capacity(features.len());
    let mut hist_len = 0usize;
    for f in features {
        slot_off.push(hist_len);
        hist_len += 2 * f.num_bin as usize;
    }

    // ---- Root init (§6.1): whole-dataset ordered f64 fold (== LeafSplits::init). ----
    let root_rows: Vec<u32> = (0..num_data as u32).collect();
    let mut root_sum_g = 0.0f64;
    let mut root_sum_h = 0.0f64;
    for &r in &root_rows {
        root_sum_g += f64::from(gradients[r as usize]);
        root_sum_h += f64::from(hessians[r as usize]);
    }
    // Loud typed failure on a NaN root seed on the anchor/non-resident arm too — never
    // a silently-truncated model via GBDT's pop-loop.
    check_root_seed_finite(root_sum_g, root_sum_h)?;
    let root_hist = build_leaf_hist(
        client, features, gradients, hessians, &root_rows, root_sum_g, root_sum_h, &slot_off,
        hist_len,
    )?;
    let (root_best, root_fpos, root_cat) = scan_leaf(
        client, features, &root_hist, root_sum_g, root_sum_h, num_data as i32, &slot_off, &cfg,
    )?;

    let mut leaves: Vec<DriverLeaf> = vec![DriverLeaf {
        rows: root_rows,
        sum_g: root_sum_g,
        sum_h: root_sum_h,
        hist: root_hist,
        best: root_best,
        best_fpos: root_fpos,
        best_cat: root_cat,
        depth: 0,
    }];

    // ---- The device flat tree, pre-allocated once. ----
    // `num_leaves >= 1` is guaranteed by the guard above, so no `.max(1)` needed.
    let mut tree = DeviceCudaTree::<R>::new(client, num_leaves as usize, num_data as i32)?;

    // ---- The pre-allocated categorical split-info slab, allocated ONCE (no
    // per-split device alloc) and ONLY when the feature set has a categorical feature
    // — so a pure-numeric grow allocates nothing new and stays byte-for-byte
    // unchanged. The runtime slab width is the max `max_cat_threshold` across the
    // categorical features (default MAX_CAT_PER_SPLIT). ----
    let has_categorical = features.iter().any(|f| f.bin_type == BinType::Categorical);
    let mut split_info: Option<DeviceSplitInfo<R>> = if has_categorical {
        let cat_width = features
            .iter()
            .filter(|f| f.bin_type == BinType::Categorical)
            .map(|f| f.max_cat_threshold.max(1) as usize)
            .max()
            .unwrap_or(MAX_CAT_PER_SPLIT);
        Some(DeviceSplitInfo::<R>::new(client, num_leaves as usize, cat_width)?)
    } else {
        None
    };
    // Seed the root leaf value so a never-split root still matches the anchor.
    let root_output =
        calculate_splitted_leaf_output(cfg.use_l1(), root_sum_g, root_sum_h, cfg.lambda_l1, cfg.lambda_l2);
    tree.add_bias(client, root_output);

    // ---- The best-first leaf-wise loop (serial_tree_learner.cpp:218-236). ----
    for _split in 0..(num_leaves - 1) {
        // best_leaf = argmax(best_split_per_leaf) via split_gt (first-max).
        let mut best_leaf = 0i32;
        for i in 1..leaves.len() {
            let a = &leaves[i];
            let a_feat = if a.best_fpos < 0 {
                -1
            } else {
                features[a.best_fpos as usize].real_feature_index
            };
            let b = &leaves[best_leaf as usize];
            let b_feat = if b.best_fpos < 0 {
                -1
            } else {
                features[b.best_fpos as usize].real_feature_index
            };
            if split_gt(&a.best, a_feat, &b.best, b_feat) {
                best_leaf = i as i32;
            }
        }
        let best = leaves[best_leaf as usize].best;
        let best_fpos = leaves[best_leaf as usize].best_fpos;
        // No positive-gain split anywhere ⇒ stop (best_leaf == -1 sentinel analog).
        if best_fpos < 0 || !(best.gain > 0.0) {
            break;
        }

        let f = &features[best_fpos as usize];
        let parent_depth = leaves[best_leaf as usize].depth;
        let parent_hist = leaves[best_leaf as usize].hist.clone();

        // ---- Partition the parent leaf's rows, BEFORE the tree
        // mutation reads the child ids. ----
        let parent_rows = leaves[best_leaf as usize].rows.clone();
        let bins_sub = BinColumn::new(
            parent_rows.iter().map(|&r| f.bins.bin(r as usize)).collect(),
            f.num_bin,
        );
        let missing_type_u8 = match f.missing_type {
            MissingType::None => 0u8,
            MissingType::Zero => 1,
            MissingType::NaN => 2,
        };
        let missing_type_code = i32::from(missing_type_u8);
        let new_left = best_leaf;

        // The categorical and numeric branches differ ONLY in the partition + tree
        // mutation (§9/§10 vs the numeric route/split); everything downstream (child
        // seeding, subtract, scan) is shared. Each branch yields the child rows/counts
        // and the new right-leaf id.
        let (left_rows, right_rows, left_count, right_count, new_right) =
            if f.bin_type == BinType::Categorical {
                // ===== Categorical grow branch (§6.3 + §9 + §10). =====
                // (1) Stage the winning thresholds into the pre-allocated DeviceSplitInfo
                //     cat slab — allocate-once, NO per-split device alloc.
                let win_bins: Vec<u32> = leaves[best_leaf as usize].best_cat.clone();
                let win_real: Vec<i32> = win_bins
                    .iter()
                    .map(|&b| f.bin_to_category.get(b as usize).copied().unwrap_or(b as i32))
                    .collect();
                // Provably unreachable (this branch is entered only when
                // `f.bin_type == Categorical`, which implies `has_categorical`, which
                // implies `split_info == Some`), but surface a typed `ComputeError`
                // rather than a raw panic to keep the grow loop's error boundary uniform.
                let dsi = split_info.as_mut().ok_or_else(|| ComputeError::Runtime {
                    detail: "grow_tree_on_device_driver: DeviceSplitInfo must be allocated for \
                             a categorical grow branch (has_categorical invariant)"
                        .to_string(),
                })?;
                dsi.set_cat_thresholds(best_leaf as usize, &win_bins, &win_real)?;
                // (2) Materialize the real + inner bitsets FROM the slab-staged thresholds
                //     (§6.3) — the host Vec<u32> bitsets are DERIVED from the slab, not a
                //     parallel per-split allocation. The REAL bitset is built
                //     by CONSUMING the `cat_threshold_real` slab (the `win_real` mapping
                //     already staged above), not by re-mapping bin→category a second time;
                //     the inner routing bitset carries the `bin - min_bin + offset`
                //     transform via `construct_inner_bitset`.
                let slab_bins: Vec<i32> = dsi
                    .cat_threshold(best_leaf as usize)
                    .iter()
                    .map(|&b| b as i32)
                    .collect();
                let real_bitset = construct_bitset(
                    &dsi.cat_threshold_real(best_leaf as usize)
                        .iter()
                        .map(|&v| v as u32)
                        .collect::<Vec<u32>>(),
                );
                let inner_bitset =
                    construct_inner_bitset(&slab_bins, f.min_bin as i32, f.offset);
                // (3) Partition parent rows by categorical membership (§9). The INNER-bin
                //     bitset is the routing key `route_to_left_categorical` expects
                //     (`bin - min_bin + offset`, offset from most_freq_bin) — isolation-tested
                //     in `categorical_partition_counts_match_host_stable`.
                let (reordered, split_point) = partition_categorical_on_device(
                    client,
                    &bins_sub,
                    &parent_rows,
                    f.num_bin,
                    f.min_bin,
                    f.max_bin,
                    f.most_freq_bin,
                    &inner_bitset,
                )?;
                let left_rows: Vec<u32> = reordered[..split_point].to_vec();
                let right_rows: Vec<u32> = reordered[split_point..].to_vec();
                let left_count = left_rows.len() as i32;
                let right_count = right_rows.len() as i32;
                // (4) Grow the categorical node via the EXISTING §10 entrypoint
                //     (`split_categorical_on_device`, tree.rs:765) — consuming BOTH the
                //     real (`cat_threshold_`) and inner (routing) bitsets. `num_cat_threshold`
                //     is the real winning count (numeric uses 0). `threshold` is unused.
                let scalars = SplitScalars {
                    is_valid: true,
                    leaf_index: best_leaf,
                    gain: best.gain + cfg.min_gain_to_split,
                    inner_feature_index: f.real_feature_index,
                    threshold: 0,
                    default_left: best.default_left,
                    left_sum_gradients: best.left_sum_gradient,
                    left_sum_hessians: best.left_sum_hessian,
                    left_sum_gh_quant: 0,
                    left_count,
                    left_gain: 0.0,
                    left_value: best.left_output,
                    right_sum_gradients: best.right_sum_gradient,
                    right_sum_hessians: best.right_sum_hessian,
                    right_sum_gh_quant: 0,
                    right_count,
                    right_gain: 0.0,
                    right_value: best.right_output,
                    num_cat_threshold: win_bins.len() as i32,
                };
                // The categorical split also reads `right_leaf_index` back —
                // ONE blocking readback per split (num_features-independent).
                bump_sync();
                let result = tree.split_categorical_on_device(
                    client,
                    best_leaf,
                    f.real_feature_index,
                    missing_type_code,
                    &scalars,
                    &real_bitset,
                    &inner_bitset,
                )?;
                (left_rows, right_rows, left_count, right_count, result.right_leaf_index)
            } else {
                // ===== Numeric grow branch (unchanged — byte-for-byte). =====
                // Single-feature-group min_bin convention: min_bin + offset.
                let partition_min_bin = f.min_bin + f.offset.max(0) as u32;
                let (reordered, split_point) = partition_leaf_stable(
                    &bins_sub,
                    &parent_rows,
                    f.num_bin,
                    partition_min_bin,
                    f.max_bin,
                    f.default_bin,
                    f.most_freq_bin,
                    missing_type_u8,
                    best.default_left,
                    best.threshold,
                )?;
                let left_rows: Vec<u32> = reordered[..split_point].to_vec();
                let right_rows: Vec<u32> = reordered[split_point..].to_vec();
                let left_count = left_rows.len() as i32;
                let right_count = right_rows.len() as i32;
                // An out-of-range `best.threshold` would silently record the raw
                // bin index cast to f64 as the tree's REAL threshold — a plausible-looking
                // wrong value that would corrupt prediction routing and mask an off-by-one
                // in the offset/compaction threshold space. Surface a typed error instead.
                let real_threshold =
                    *f.bin_upper_bound.get(best.threshold as usize).ok_or_else(|| {
                        ComputeError::Runtime {
                            detail: format!(
                                "grow_tree_on_device_driver_with_cfg: split threshold bin index {} out of \
                                 range for feature {} bin_upper_bound (len {})",
                                best.threshold,
                                f.real_feature_index,
                                f.bin_upper_bound.len()
                            ),
                        }
                    })?;
                let scalars = SplitScalars {
                    is_valid: true,
                    leaf_index: best_leaf,
                    gain: best.gain + cfg.min_gain_to_split,
                    inner_feature_index: f.real_feature_index,
                    threshold: best.threshold,
                    default_left: best.default_left,
                    left_sum_gradients: best.left_sum_gradient,
                    left_sum_hessians: best.left_sum_hessian,
                    left_sum_gh_quant: 0,
                    left_count,
                    left_gain: 0.0,
                    left_value: best.left_output,
                    right_sum_gradients: best.right_sum_gradient,
                    right_sum_hessians: best.right_sum_hessian,
                    right_sum_gh_quant: 0,
                    right_count,
                    right_gain: 0.0,
                    right_value: best.right_output,
                    num_cat_threshold: 0,
                };
                // split_on_device reads `right_leaf_index` back to the host —
                // ONE blocking readback per split (num_features-independent).
                bump_sync();
                let result = tree.split_on_device(
                    client,
                    best_leaf,
                    f.real_feature_index,
                    real_threshold,
                    missing_type_code,
                    &scalars,
                )?;
                (left_rows, right_rows, left_count, right_count, result.right_leaf_index)
            };

        // ---- Seed the two child leaves from the SplitInfo (NOT a re-fold): the
        // kEpsilon-carrying sums are load-bearing for the next split (Pitfall 2). ----
        let child_depth = parent_depth + 1;
        // Update the reused left child in place.
        {
            let l = &mut leaves[best_leaf as usize];
            l.rows = left_rows;
            l.sum_g = best.left_sum_gradient;
            l.sum_h = best.left_sum_hessian;
            l.depth = child_depth;
            l.best = SplitInfo::none();
            l.best_fpos = -1;
            l.best_cat = Vec::new();
        }
        // Append the new right child (leaf id == new_right). The kernel's
        // `right_leaf_index` (== the tree's internal `num_leaves`) and the driver's
        // `leaves.len()` are kept in lockstep only by construction. A plain
        // `debug_assert_eq!` would compile out in release, so a kernel/driver desync
        // would either panic on the later `leaves[new_right]` index or — worse —
        // silently write the derived histogram into the wrong leaf slot, corrupting
        // the partition with no typed error. Fail loudly with a typed `ComputeError`
        // in ALL build profiles, BEFORE the `push` and any `leaves[new_right]` access,
        // matching the driver's other invariant boundaries.
        if new_right as usize != leaves.len() {
            return Err(ComputeError::Runtime {
                detail: format!(
                    "grow_tree_on_device_driver: leaf-id desync — kernel right_leaf_index {} \
                     must equal the next driver leaf slot {} (tree num_leaves and driver \
                     leaves.len() out of lockstep)",
                    new_right,
                    leaves.len()
                ),
            });
        }
        leaves.push(DriverLeaf {
            rows: right_rows,
            sum_g: best.right_sum_gradient,
            sum_h: best.right_sum_hessian,
            hist: vec![0.0; hist_len],
            best: SplitInfo::none(),
            best_fpos: -1,
            best_cat: Vec::new(),
            depth: child_depth,
        });

        // ---- Build the children histograms: SMALLER directly, LARGER by
        // subtraction from the PARENT (subtract, parent-built-before-child).
        // Smaller = the fewer-row child (num_left < num_right ⇒ left, else right). ----
        let smaller_is_left = left_count < right_count;
        let (smaller_leaf, larger_leaf) = if smaller_is_left {
            (new_left, new_right)
        } else {
            (new_right, new_left)
        };
        let (s_rows, s_g, s_h) = {
            let s = &leaves[smaller_leaf as usize];
            (s.rows.clone(), s.sum_g, s.sum_h)
        };
        let smaller_hist = build_leaf_hist(
            client, features, gradients, hessians, &s_rows, s_g, s_h, &slot_off, hist_len,
        )?;
        // LARGER = parent − smaller (subtract kernel), over the whole
        // concatenated compacted buffer (zeroed tails subtract to zero).
        bump_launch(); // One real on-device subtraction-trick dispatch.
        let larger_hist = subtract_histograms_f64_on(client, &parent_hist, &smaller_hist)?;
        leaves[smaller_leaf as usize].hist = smaller_hist;
        leaves[larger_leaf as usize].hist = larger_hist;

        // ---- BeforeFindBestSplit gates + scan each child (compute its best). ----
        // Mirror C++ `BeforeFindBestSplit`'s PER-LEAF `num_data <
        // min_data_in_leaf * 2` gate — evaluate each child INDEPENDENTLY (a child
        // with fewer than `2*min_data` rows cannot yield a split with both sides
        // >= min_data, so it is skipped) rather than applying one combined
        // predicate to both children. `saturating_mul` guards the `* 2` against a
        // pathological `min_data > i32::MAX/2` overflow. Tree structure is unchanged
        // either way: the small child's splits are already rejected by
        // `scan_leaf`'s per-side `min_data_in_leaf` guards.
        let min_data_x2 = min_data.saturating_mul(2);
        for &child in &[new_left, new_right] {
            let too_small = (leaves[child as usize].rows.len() as i32) < min_data_x2;
            let depth_capped = max_depth > 0 && leaves[child as usize].depth >= max_depth;
            if depth_capped || too_small {
                leaves[child as usize].best = SplitInfo::none();
                leaves[child as usize].best_fpos = -1;
                leaves[child as usize].best_cat = Vec::new();
                continue;
            }
            let (cg, ch, cn, chist) = {
                let c = &leaves[child as usize];
                (c.sum_g, c.sum_h, c.rows.len() as i32, c.hist.clone())
            };
            let (cbest, cfpos, ccat) =
                scan_leaf(client, features, &chist, cg, ch, cn, &slot_off, &cfg)?;
            leaves[child as usize].best = cbest;
            leaves[child as usize].best_fpos = cfpos;
            leaves[child as usize].best_cat = ccat;
        }
    }

    // ---- Reconstruct the host tree (to_host_tree) + the row→leaf layout. ----
    let host_tree = tree.to_host_tree(client);
    let final_leaves = host_tree.num_leaves as usize;
    let mut indices = Vec::with_capacity(num_data);
    let mut leaf_begin = Vec::with_capacity(final_leaves);
    let mut leaf_count = Vec::with_capacity(final_leaves);
    for leaf in leaves.iter().take(final_leaves) {
        leaf_begin.push(indices.len() as i32);
        leaf_count.push(leaf.rows.len() as i32);
        indices.extend_from_slice(&leaf.rows);
    }
    let layout = LeafPartitionLayout {
        num_data: num_data as i32,
        indices,
        leaf_begin,
        leaf_count,
    };
    // Refuse to emit a tree with any non-finite leaf value (read-only).
    check_tree_leaves_finite(&host_tree)?;
    Ok((host_tree, layout))
}

/// A DEVICE-RESIDENT f64 per-row score mirror (the §11 `cuda_score_` analog) accumulated
/// ACROSS trees with NO per-tree readback.
///
/// This keeps the score buffer resident on device for the whole train instead of reading
/// the full `num_data` f64 delta back to host every tree: each grown tree's per-leaf
/// `AddScore` is applied by [`Self::add_tree_on_device`] (device leaf-map derivation via
/// [`crate::kernels::predict::derive_leaf_map_device_handle`] + resident scatter via
/// [`crate::kernels::predict::add_leaf_values_to_resident_score`], BOTH device→device, no
/// crossback), and the host reads the score back ONLY on demand via
/// [`Self::read_resident_score`] (§16 `Metric::Eval` / final).
///
/// Anchor discipline: the accumulation is an integer-indexed scatter of the SAME exact
/// post-shrinkage f64 leaf values into `score[row]`, in tree order — no float reduction,
/// no ordering dependence — so the final resident score is BIT-EXACT to the eager
/// per-tree-readback host accumulation on the cpu-f64 anchor, and within the ~1e-6 f32
/// envelope on hip (NEVER bit-exact to a second GPU f32 path).
///
/// Static geometry only — never a dynamic (device-derived) launch count.
pub struct ResidentScore<R: cubecl::Runtime> {
    /// The resident `num_data`-length f64 score accumulator (§11 `cuda_score_`), zeroed
    /// at construction and accumulated in place across trees.
    score: cubecl::server::Handle,
    /// The score buffer length (one f64 per row).
    num_data: usize,
    /// Per-train device residency for the objective's grad/hess launch
    /// ([`GradResidency`]): the f64 labels uploaded ONCE (immutable for the whole
    /// train) + the f32 grad/hess output buffers allocated ONCE and fully
    /// overwritten by every launch. Built lazily on the first grad call so the
    /// constructors stay unchanged; `OnceCell` because the per-iter caller holds
    /// `&self` (the same shared-borrow discipline as the score handle).
    grad_residency: std::cell::OnceCell<GradResidency>,
    /// Ties the mirror to its CubeCL runtime `R` without storing one.
    _runtime: std::marker::PhantomData<fn() -> R>,
}

/// Per-train device-resident objective launch state (owned by [`ResidentScore`]).
/// The labels are IMMUTABLE for the whole train, so their host f32→f64 convert +
/// 4·`num_data`-byte host→device upload runs exactly ONCE instead of every
/// iteration; the grad/hess f32 outputs are allocated once and fully overwritten by
/// each launch (the kernels write every `i < num_data` cell), so reusing them is
/// value-identical to the prior per-iter `client.empty` pair while eliminating the
/// per-iter device alloc/free churn.
#[derive(Debug)]
pub struct GradResidency {
    /// The `[num_data]` f64 label buffer (uploaded once).
    pub labels_f64: cubecl::server::Handle,
    /// The `[num_data]` f32 gradient output (reused each iteration).
    pub grad: cubecl::server::Handle,
    /// The `[num_data]` f32 hessian output (reused each iteration).
    pub hess: cubecl::server::Handle,
}

impl<R: cubecl::Runtime> ResidentScore<R> {
    /// Allocate a zeroed resident f64 score buffer of `num_data` rows on the device.
    #[must_use]
    pub fn new(client: &cubecl::prelude::ComputeClient<R>, num_data: usize) -> Self {
        use cubecl::prelude::CubeElement;
        let init = vec![0.0f64; num_data];
        let score = client.create_from_slice(f64::as_bytes(&init));
        Self {
            score,
            num_data,
            grad_residency: std::cell::OnceCell::new(),
            _runtime: std::marker::PhantomData,
        }
    }

    /// Allocate a resident f64 score buffer initialized
    /// FROM the current host `scores` snapshot (uploaded ONCE at train start, after
    /// `BoostFromAverage` has injected any per-class init constant). This is the
    /// GBDT-owned train-lifetime constructor: the buffer is created once and then
    /// accumulated in place across trees via [`Self::add_tree_on_device`], so the host
    /// score is uploaded exactly ONCE (not per iteration) — closing the per-iteration
    /// host round-trip the on-device grad/hess branch (`gbdt.rs:679`) then reads from
    /// directly. `scores` is `[num_data]` (the single-class `num_class==1` envelope).
    #[must_use]
    pub fn from_host_scores(client: &cubecl::prelude::ComputeClient<R>, scores: &[f64]) -> Self {
        use cubecl::prelude::CubeElement;
        let score = client.create_from_slice(f64::as_bytes(scores));
        Self {
            score,
            num_data: scores.len(),
            grad_residency: std::cell::OnceCell::new(),
            _runtime: std::marker::PhantomData,
        }
    }

    /// The per-train [`GradResidency`] (f64 labels uploaded once + reused f32
    /// grad/hess outputs), built lazily on the first call. `labels` must be the
    /// train's `[num_data]` label slice; it is converted/uploaded ONLY on the first
    /// call (the labels are immutable for the whole train, so later calls return the
    /// cached handles without touching the host slice).
    ///
    /// # Errors
    /// [`ComputeError::LengthMismatch`] if `labels.len() != self.num_data`.
    pub fn grad_residency(
        &self,
        client: &cubecl::prelude::ComputeClient<R>,
        labels: &[f32],
    ) -> Result<&GradResidency, ComputeError> {
        use cubecl::prelude::CubeElement;
        if labels.len() != self.num_data {
            return Err(ComputeError::LengthMismatch {
                expected: self.num_data,
                actual: labels.len(),
            });
        }
        Ok(self.grad_residency.get_or_init(|| {
            let labels_f64: Vec<f64> = labels.iter().map(|&l| f64::from(l)).collect();
            GradResidency {
                labels_f64: client.create_from_slice(f64::as_bytes(&labels_f64)),
                grad: client.empty(self.num_data * core::mem::size_of::<f32>()),
                hess: client.empty(self.num_data * core::mem::size_of::<f32>()),
            }
        }))
    }

    /// The score buffer length (one f64 per row).
    #[must_use]
    pub fn num_data(&self) -> usize {
        self.num_data
    }

    /// A borrow of the resident score device handle (the resident mirror the deferred
    /// readback crosses back).
    #[must_use]
    pub fn score_handle(&self) -> &cubecl::server::Handle {
        &self.score
    }

    /// Accumulate one ON-DEVICE-grown tree's per-leaf `AddScore` into the resident score
    /// mirror, entirely on device (NO readback): derive `data_index_to_leaf` on device
    /// from the resident partition `layout` (§9), then scatter the (post-shrinkage)
    /// `leaf_values` into the resident buffer in place (§11). `leaf_values` is
    /// `tree.leaf_value` AT SCORE TIME (post-`shrinkage`), matching the §16 Shrinkage →
    /// UpdateScore order.
    ///
    /// # Errors
    /// [`ComputeError::LengthMismatch`] if `layout.num_data != self.num_data`, if a leaf
    /// sub-range overruns `indices`, or if `layout` has more leaves than `leaf_values`
    /// (a leaf id would index past `leaf_values`); [`ComputeError::Runtime`] on a negative
    /// `leaf_begin`/`leaf_count`.
    pub fn add_tree_on_device(
        &mut self,
        client: &cubecl::prelude::ComputeClient<R>,
        layout: &LeafPartitionLayout,
        leaf_values: &[f64],
    ) -> Result<(), ComputeError> {
        let num_data = layout.num_data as usize;
        if num_data != self.num_data {
            return Err(ComputeError::LengthMismatch {
                expected: self.num_data,
                actual: num_data,
            });
        }
        // The device leaf map writes leaf ids in `[0, leaf_begin.len())`; the resident
        // scatter reads `leaf_value[leaf]`, so every leaf id must index `leaf_values`.
        if layout.leaf_begin.len() > leaf_values.len() {
            return Err(ComputeError::LengthMismatch {
                expected: layout.leaf_begin.len(),
                actual: leaf_values.len(),
            });
        }
        // FUSED derive+scatter (no readback, no intermediate `num_data`-length leaf-map
        // buffer): one lane per partition position accumulates
        // `score[indices[k]] += leaf_value[leaf_of(k)]` in place. Replaces the
        // `derive_leaf_map_device_handle` → `add_leaf_values_to_resident_score` chain,
        // whose derive kernel parallelized over LEAVES (a single active warp at
        // num_leaves=31 serially walking every row — ~O(num_data) serial device time
        // per tree that the NEXT iteration's blocking grad readback then absorbed).
        // Bit-exact: disjoint leaf ranges ⇒ each row written exactly once with the
        // same f64 `+=` of the same value (see the kernel doc). `LGBM_SCORE_FUSED_SCATTER=0`
        // restores the two-kernel chain for the same-session A/B.
        if score_fused_scatter_enabled() {
            return crate::kernels::predict::add_leaf_values_by_ranges_to_resident_score(
                client,
                &layout.indices,
                &layout.leaf_begin,
                &layout.leaf_count,
                &self.score,
                leaf_values,
                num_data,
            );
        }
        // A/B escape hatch: the pre-fusion derive→scatter chain, byte-identical result.
        let leaf_map = crate::kernels::predict::derive_leaf_map_device_handle(
            client,
            &layout.indices,
            &layout.leaf_begin,
            &layout.leaf_count,
            num_data,
        )?;
        crate::kernels::predict::add_leaf_values_to_resident_score(
            client,
            &leaf_map,
            &self.score,
            leaf_values,
            num_data,
        )
    }

    /// On-demand crossback (§16 `Metric::Eval` / final result) — read the resident score
    /// buffer back to host ONCE. The host calls this only when it genuinely needs the
    /// score, NOT every tree — the whole point of the residency.
    #[must_use]
    pub fn read_resident_score(&self, client: &cubecl::prelude::ComputeClient<R>) -> Vec<f64> {
        use cubecl::prelude::CubeElement;
        f64::from_bytes(&client.read_one_unchecked(self.score.clone())).to_vec()
    }
}

/// Read a resident `[num_data]` f32 device buffer back to
/// a host `Vec<f32>`. The on-device grad/hess branch (`gbdt.rs:679`) computes grad/hess
/// on device (`get_gradients_resident_on`, device-in/device-out Handles) from the
/// resident score, then reads them back here to feed the (host) tree learner — the SAME
/// blocking single-buffer readback the resident score uses ([`ResidentScore::read_resident_score`]),
/// only for the f32 grad/hess Handles. On a real-CUDA on-device grow the learner would
/// consume the resident grad/hess Handles directly (no readback); this host readback is
/// the correctness bridge for the host-learner path and the cpu-anchor parity gate.
#[must_use]
pub fn read_handle_f32<R: cubecl::Runtime>(
    client: &cubecl::prelude::ComputeClient<R>,
    handle: &cubecl::server::Handle,
) -> Vec<f32> {
    use cubecl::prelude::CubeElement;
    f32::from_bytes(&client.read_one_unchecked(handle.clone())).to_vec()
}

/// Apply an ON-DEVICE-grown tree's leaf-value contribution to the per-row score
/// (the §11 `AddPredictionToScore` analog) by scattering each leaf's f64 output over the
/// RESIDENT row→leaf partition the grow already produced ([`LeafPartitionLayout`]),
/// entirely on device.
///
/// This is the resident partition-SCATTER analog of the C++ training-path
/// `SerialTreeLearner::AddPredictionToScore` (`serial_tree_learner.h:100-118`): the
/// grow driver hands back the row→leaf layout, so the on-device score move needs NO
/// host re-gather of the `[num_data × num_features]` bin matrix and NO per-row tree
/// walk (previously the dominant host-scoring cost) — only the winning leaf id per row
/// (already computed on device by the grow) and the (post-shrinkage) leaf values.
/// Delegates to the golden
/// [`crate::kernels::predict::add_prediction_bagging_on_device`] partition-scatter
/// kernel in identity mode (every row walked exactly once), returning the
/// `num_data`-length per-row raw-margin delta — the host mirror is the ONLY thing
/// that crosses back (`cuda_score_` and the partition/leaf-value stay resident).
///
/// Anchor discipline: bit-exact to the host partition scatter
/// [`add_prediction_to_score`](../../../lgbm_treelearner) on the cpu-f64 anchor by
/// construction — an integer-indexed scatter of the SAME exact f64 leaf values, no
/// float reduction, no ordering dependence; the ROCm/HIP arm holds to the ~1e-6 f32
/// envelope vs that cpu anchor, NEVER bit-exact to a second GPU f32 path.
///
/// `leaf_values` is `tree.leaf_value` AT SCORE TIME (post-`shrinkage`); the
/// `boosting_on_cuda_` resident-score seam calls this AFTER shrinkage, matching the
/// §16 Shrinkage → UpdateScore order.
///
/// # Relationship to [`ResidentScore`]
/// This is the SINGLE-TREE eager-readback convenience over the resident
/// [`ResidentScore`] mirror: it allocates a fresh zeroed resident buffer, applies the one
/// tree ON DEVICE (device leaf-map derivation + resident scatter — no host `O(num_data)`
/// inversion loop), and crosses the buffer back once to return the per-tree
/// `num_data`-length delta (the fresh buffer started at 0, so the read IS this tree's
/// delta). The ACROSS-trees residency (accumulate many trees, defer the readback to
/// on-demand) is [`ResidentScore`] — the resident mirror the deferred-readback score path
/// drives; this wrapper preserves the byte-compatible `Vec<f64>` contract for the
/// existing per-tree caller.
///
/// # Errors
/// [`ComputeError::LengthMismatch`] if a leaf group (`leaf_begin + leaf_count`) overruns
/// `indices` or `layout` has more leaves than `leaf_values`; [`ComputeError::Runtime`] on
/// a negative `leaf_begin`/`leaf_count`.
pub fn add_prediction_to_score_on_device_resident<R: cubecl::Runtime>(
    client: &cubecl::prelude::ComputeClient<R>,
    layout: &LeafPartitionLayout,
    leaf_values: &[f64],
) -> Result<Vec<f64>, ComputeError> {
    let mut mirror = ResidentScore::new(client, layout.num_data as usize);
    mirror.add_tree_on_device(client, layout, leaf_values)?;
    Ok(mirror.read_resident_score(client))
}

// =========================================================================
// The RESIDENT/BATCHED numeric grow path (§7.1).
//
// The fast capability arm of `grow_tree_on_device_driver_with_cfg`. Instead of the
// anchor fold's per-feature single-thread `construct_histograms_f64_on` (`CubeDim(1)`)
// with a host re-gather + blocking readback each leaf, this arm uploads the binned
// columns to the device ONCE per grow (`upload_resident_bins`) and keeps every leaf's
// histogram DEVICE-RESIDENT in a pool slot: `build_resident_leaf` builds the root and
// each SMALLER child (build → dequant → fix → compact into a slot, using the shipped
// u64 fixed-point deterministic accumulation), and `subtract_resident` derives the
// LARGER child (parent − smaller, on device, no readback). Splits are read from the
// resident slot via `scan_resident_leaf` — the histogram never returns to host. The
// driver keeps its OWN best-first `DriverLeaf` bookkeeping and maps each leaf to a
// resident pool slot with a simple counter (SMALLER child gets a fresh slot; LARGER
// reuses the parent's slot, which holds the subtraction's parent Handle).
//
// ## Scope
// NUMERIC features + the reverse/forward split scan only (categorical demotes to the
// anchor fold in the caller). The row PARTITION still routes on the HOST
// (`partition_leaf_stable`, unchanged from the numeric anchor branch). No new kernels
// — this wires the shipped `GpuBackend` resident primitives into the driver.
//
// ## Ordering
// build-SMALLER → subtract-LARGER, ALWAYS — never subtract before the smaller build
// (`subtract_resident` reads the smaller slot). The larger child reuses the parent
// slot, so `subtract_resident(parent_slot, smaller_slot, larger_slot=parent_slot)`
// overwrites the (now-consumed) parent Handle with the derived larger child.
// =========================================================================

/// The resident-arm per-leaf state — the numeric analog of [`DriverLeaf`] that holds a
/// device pool SLOT id instead of a host `hist: Vec<f64>` (the histogram is
/// device-resident). No `best_cat` (numeric proving slice only).
struct ResidentDriverLeaf {
    /// This leaf's rows as a RESIDENT device index RANGE — the
    /// `[row_begin, row_begin + row_count)` sub-range of the single resident permutation buffer
    /// (`perm`, the `cuda_data_indices_` mirror). The row ids never leave that buffer: a split
    /// partitions the parent's sub-range IN PLACE and the two children become adjacent
    /// sub-ranges, retiring the per-leaf `rows: Vec<u32>` `.clone()`'d every split.
    row_begin: usize,
    /// The number of rows this leaf owns (the range width).
    row_count: usize,
    /// The leaf's seeded RAW gradient sum (root = ordered f64 fold; child = the
    /// parent split's `left/right_sum_gradient`).
    sum_g: f64,
    /// The leaf's seeded RAW hessian sum.
    sum_h: f64,
    /// This leaf's device histogram-pool slot id (the resident Handle mirror index).
    slot: usize,
    /// The leaf's best split (`gain == -inf` ⇒ no admissible split).
    best: SplitInfo,
    /// The winning feature POSITION (`-1` when no split).
    best_fpos: i32,
    /// The leaf's depth (root = 0), for the `max_depth` gate.
    depth: i32,
}

/// Read-once `LGBM_SIBLING_COPACK != "0"` gate — the driver-local TWIN of
/// `lgbm_treelearner::resident_pool::sibling_copack_override` (which lives ABOVE
/// `lgbm-compute` and cannot be imported here without a crate cycle). Default-ON
/// (co-pack engages whenever the correctness gate holds); `LGBM_SIBLING_COPACK=0`
/// FORCE-OFFs it (the byte-identical two-separate-scans off-path: a fused-smaller +
/// separate-larger scan). Parsed EVERY call (NOT `OnceLock`-cached) so a
/// single test process can grow with the env set and unset and observe BOTH launch
/// layouts — the two trees are identical (co-pack is bit-exact by construction), only
/// WHICH launch scans each sibling changes.
fn sibling_copack_enabled() -> bool {
    std::env::var("LGBM_SIBLING_COPACK").map(|v| v != "0").unwrap_or(true)
}

/// The driver's cross-feature `split_gt` argmax over an ORDER-PRESERVING per-feature
/// scan result (tie-break bit-exact). One [`SplitInfo`] per feature in fpos
/// order; the per-feature `na_as_missing` skip mirrors [`scan_leaf`]'s `continue` (the
/// finder rejects NA — never fires on the `MissingType::None` proving slice). Shared by
/// the fused ([`crate::Backend::build_fix_scan_resident`]), single-slot
/// ([`scan_resident_and_argmax`]), and co-packed
/// ([`crate::Backend::scan_resident_siblings`]) scan paths so all three fold their
/// results into `(SplitInfo, feature-position)` identically. Returns `(-inf, -1)` when
/// nothing is admissible.
fn argmax_over_splits(
    splits: &[SplitInfo],
    feats: &[BatchedSplitFeature],
    features: &[GrowFeature],
) -> (SplitInfo, i32) {
    let mut best = SplitInfo::none();
    let mut best_fpos: i32 = -1;
    let mut best_real: i32 = -1;
    for (fpos, si) in splits.iter().enumerate() {
        if feats[fpos].na_as_missing {
            continue;
        }
        let real = features[fpos].real_feature_index;
        if split_gt(si, real, &best, best_real) {
            best = *si;
            best_fpos = fpos as i32;
            best_real = real;
        }
    }
    (best, best_fpos)
}

/// The cross-feature best-split REDUCE (the §8.2 `SyncBestSplitForLeafKernel` analog)
/// that returns ONLY the winning `(SplitInfo, feature-position)` — the ~8-int
/// CUDASplitInfo-equivalent the on-device argmax reads back per leaf, INSTEAD of the full
/// `Vec<SplitInfo>` per feature. Byte-for-byte the same fold as [`argmax_over_splits`] (strict
/// `>` gain, lowest real-feature-index tie-break, and the
/// `na_as_missing` skip) so the winner is BIT-IDENTICAL to the host argmax; it takes
/// `real_feats[fpos]` (each feature's real index, the tie-break key) directly so it can live on
/// the [`crate::Backend`] seam in `lib.rs` without importing [`GrowFeature`]. Shared by the
/// GpuBackend on-device argmax methods
/// ([`crate::Backend::scan_resident_leaf_argmax`] / `scan_resident_siblings_argmax`).
// Consumed by the `#[cfg(feature = "gpu")]` GpuBackend argmax impls (lib.rs) + the cpu-runnable
// bit-identity unit test; unreferenced in a default non-gpu lib build.
#[cfg_attr(not(feature = "gpu"), allow(dead_code))]
pub(crate) fn argmax_over_resident_splits(
    splits: &[SplitInfo],
    feats: &[BatchedSplitFeature],
    real_feats: &[i32],
) -> (SplitInfo, i32) {
    let mut best = SplitInfo::none();
    let mut best_fpos: i32 = -1;
    let mut best_real: i32 = -1;
    for (fpos, si) in splits.iter().enumerate() {
        if feats[fpos].na_as_missing {
            continue;
        }
        let real = real_feats[fpos];
        if split_gt(si, real, &best, best_real) {
            best = *si;
            best_fpos = fpos as i32;
            best_real = real;
        }
    }
    (best, best_fpos)
}

/// The cross-LEAF best-leaf REDUCE (the §8.3 `FindBestFromAllSplitsKernel` analog) —
/// argmax over each leaf's best split via the SAME `split_gt` first-max rule the host
/// loop uses (strict `>` gain, lowest real-feature-index tie-break; a no-split leaf carries
/// `real == -1` ⇒ `i32::MAX`). Returns the winning leaf index (`0` when nothing has positive
/// gain — the caller's `best_fpos < 0 || !(gain > 0)` guard then breaks, exactly as the host
/// loop that always seeds `best_leaf = 0`). Lives here (not on the `Backend` seam consumer
/// side) so it reuses the private [`split_gt`]; the [`crate::Backend::best_leaf_reduce`] default
/// + GpuBackend impl both delegate here so the pick is bit-identical across backends.
pub(crate) fn best_leaf_argmax(leaf_best: &[SplitInfo], leaf_real_feat: &[i32]) -> i32 {
    let mut best_leaf = 0i32;
    for i in 1..leaf_best.len() {
        let a_feat = leaf_real_feat[i];
        let b_feat = leaf_real_feat[best_leaf as usize];
        if split_gt(&leaf_best[i], a_feat, &leaf_best[best_leaf as usize], b_feat) {
            best_leaf = i as i32;
        }
    }
    best_leaf
}

/// The whole-dataset ROOT grad/hess sum (the §6.1 `CUDAInitValuesKernel` analog) as an
/// ORDERED f64 fold (ascending row order is load-bearing — the bit-exact anchor;
/// `LeafSplits::init`). The [`crate::Backend::root_grad_hess_sum`] default + the GpuBackend impl
/// both delegate here so the root init is bit-exact vs the integer path and (on the GpuBackend
/// arm) held ~1e-6 vs the host-CUDA fold — NEVER a GPU-f32-vs-GPU-f32 pairing.
pub(crate) fn root_grad_hess_fold(gradients: &[f32], hessians: &[f32]) -> (f64, f64) {
    let mut sum_g = 0.0f64;
    let mut sum_h = 0.0f64;
    for (g, h) in gradients.iter().zip(hessians.iter()) {
        sum_g += f64::from(*g);
        sum_h += f64::from(*h);
    }
    (sum_g, sum_h)
}

/// The NON-FINITE TRIPWIRE on the seeded root grad/hess sums. A NaN root sum can arise
/// from an f32 root-fold bias (or any upstream corruption) accumulating to NaN;
/// `!(sum_h > 0)` then makes the root unscannable, and GBDT's C++-faithful no-split
/// POP-LOOP silently truncates the model instead of failing. Erroring HERE — the instant
/// the corruption is observable at the grow return path — surfaces the failure LOUDLY as a
/// typed [`ComputeError::NonFinite`] that the learner seam propagates (`?`), instead of
/// producing a silently truncated model. This guard lives on the on-device grow/score
/// side ONLY; the boosting pop-loop's bit-faithful semantics are NOT changed. O(1),
/// read-only — no per-row cost, no effect on healthy grows.
fn check_root_seed_finite(sum_g: f64, sum_h: f64) -> Result<(), ComputeError> {
    if !sum_g.is_finite() || !sum_h.is_finite() {
        return Err(ComputeError::NonFinite {
            detail: format!(
                "seeded root grad/hess sum is non-finite (sum_gradient = {sum_g}, \
                 sum_hessian = {sum_h}) — refusing to grow a corrupt tree that GBDT's \
                 no-split pop-loop would silently truncate (OCX-04, spike-072)"
            ),
        });
    }
    Ok(())
}

/// The NON-FINITE TRIPWIRE on the EMITTED tree's leaf values. An `is_finite()`
/// sweep over `tree.leaf_value` (O(num_leaves), read-only): a NaN/±inf leaf value
/// becomes a typed [`ComputeError::NonFinite`] naming the leaf, instead of a truncated
/// `Ok(tree)` flowing into scoring and compounding to NaN gradients. Belt-and-suspenders
/// with [`check_root_seed_finite`]: the root check catches the seed at onset, this catches
/// any non-finite that reaches an emitted leaf value on either driver arm. Healthy trees
/// pass untouched (the check never mutates the tree), so with `LGBM_CUDA_ON_DEVICE` unset
/// the whole on-device arm is dead and every backend byte is unchanged.
fn check_tree_leaves_finite(tree: &lgbm_model::Tree) -> Result<(), ComputeError> {
    for (leaf, &v) in tree.leaf_value.iter().enumerate() {
        if !v.is_finite() {
            return Err(ComputeError::NonFinite {
                detail: format!(
                    "emitted on-device leaf {leaf} value is non-finite ({v}) — refusing to \
                     emit a corrupt tree (OCX-04, spike-072 tree-4→tree-12 leaf blow-up)"
                ),
            });
        }
    }
    Ok(())
}

/// Scan one leaf's DEVICE-RESIDENT histogram (slot `slot`) for every feature's best
/// split in ONE batched launch ([`crate::Backend::scan_resident_leaf`]), then apply the
/// driver's cross-feature `split_gt` argmax over the ORDER-PRESERVING return
/// (tie-break bit-exact) via [`argmax_over_splits`]. Mirrors [`scan_leaf`]'s numeric
/// path: the same `!(sum_h > 0.0) || num_data <= 0` short-circuit and RAW `sum_h` passed
/// to the finder (it bumps `kEpsilon` internally). Returns `(SplitInfo, feature-position)`
/// — `(-inf, -1)` when nothing is admissible.
#[allow(clippy::too_many_arguments)]
fn scan_resident_and_argmax<B, R>(
    backend: &B,
    client: &cubecl::prelude::ComputeClient<R>,
    slot: usize,
    slot_len: usize,
    feats: &[BatchedSplitFeature],
    real_feats: &[i32],
    cfg: &GainConfig,
    sum_g: f64,
    sum_h: f64,
    num_data_in_leaf: i32,
) -> Result<(SplitInfo, i32), ComputeError>
where
    B: crate::Backend<Runtime = R>,
    R: cubecl::Runtime,
{
    #[allow(clippy::neg_cmp_op_on_partial_ord)]
    if !(sum_h > 0.0) || num_data_in_leaf <= 0 {
        return Ok((SplitInfo::none(), -1));
    }
    // One REAL dispatch (a single-slot `scan_resident_leaf` launch — the
    // larger child's separate scan on the co-pack-OFF path).
    bump_launch();
    // The batched scan reads ONE blocking readback per single-slot scan
    // (num_features-independent). The cross-feature argmax runs ON DEVICE
    // (`scan_resident_leaf_argmax`, §8.2) so only the winning ~8-int split crosses back — the
    // per-feature `Vec<SplitInfo>` payload collapses to a single winner; still ONE sync per scan.
    bump_sync();
    // The single-slot scan is a BLOCKING readback — in free-run mode this bucket
    // also absorbs the queued build/subtract device time draining at its sync (see the
    // ledger header; LGBM_GROW_DRAIN=1 de-aliases).
    time_phase(&GROW_SCAN_NS, || {
        backend.scan_resident_leaf_argmax(
            client, slot, slot_len, feats, real_feats, cfg, sum_g, sum_h, num_data_in_leaf,
        )
    })
}

/// Partition the parent leaf's RESIDENT index sub-range IN PLACE (§9).
///
/// `perm_range` is a mutable sub-slice of the SINGLE resident permutation buffer that mirrors
/// the reference `cuda_data_indices_` — it holds the parent leaf's GLOBAL row ids. This reorders
/// it into the stable left-then-right partition (left rows `route==0` in original relative order,
/// then right rows `route==1`) and returns `split_point` (the left-row count); the two child
/// leaves then become the adjacent sub-ranges `[0, split_point)` and `[split_point, len)` of the
/// SAME buffer. No per-leaf `Vec<u32>` is allocated and the buffer is never grown — only permuted.
///
/// Reuses the SHIPPED partition primitives (no new kernel): the DEFAULT host
/// route folds through [`partition_leaf_stable`] (the fused stable partition, which
/// already returns reordered GLOBAL row ids); the on-device route folds through
/// [`crate::Backend::data_partition_native`] (a narrow-width route kernel) and scatters
/// the global ids back into `perm_range` in place. Either way the per-leaf `rows.clone()` and the
/// host `parent_rows` map-back Vec are RETIRED — the resident buffer is updated in place. The
/// routing bin values are read from the resident index range (`perm_range`), not a separately
/// cloned `parent_rows`.
///
/// # No-HIP scope
/// The FULLY device-side gather-through-resident-index scatter (reading the split feature's bins
/// from the once-per-grow resident bin buffer via the index range, eliminating even the host bin
/// read + the on-device route's local→global map) would be a fused resident scatter kernel — a
/// NEW kernel, and one that cannot be authored/verified on the local spoofed 8-CU APU. It is a
/// real-hardware refinement deferred to a discrete-GPU test environment. What ships here is the
/// resident-buffer + in-place-partition ARCHITECTURE: one permutation buffer for the whole grow,
/// leaf ranges (not per-leaf Vecs), and the retired `rows.clone()` / `parent_rows` map-back.
///
/// # Errors
/// [`ComputeError`] from the underlying partition primitive (bad `num_bin`/threshold, an
/// out-of-range bin, or an on-device launch/readback failure).
#[allow(clippy::too_many_arguments)]
fn partition_resident_range<B, R>(
    backend: &B,
    client: &cubecl::prelude::ComputeClient<R>,
    perm_range: &mut [u32],
    f: &GrowFeature,
    best: &SplitInfo,
    partition_min_bin: u32,
    missing_type_u8: u8,
    leaf_splits: &crate::kernels::partition::DeviceLeafSplits<R>,
    leaf_id: usize,
    p_begin: i32,
    p_count: i32,
) -> Result<usize, ComputeError>
where
    B: crate::Backend<Runtime = R>,
    R: cubecl::Runtime,
{
    let (reordered, split_point) = if backend.prefers_host_partition() {
        // HOST route (fused): returns the reordered GLOBAL row ids directly.
        // Read the split feature's bins INLINE through the resident index range
        // (`partition_leaf_stable_fused`) — no per-split `bins_sub` gather / `BinColumn::new`
        // narrow, and the range check is folded into the single route pass. The env off-switch
        // (`LGBM_ONDEVICE_FUSED_PARTITION=0`) restores the pre-change gather + `partition_leaf_stable`
        // path for A/B comparison; both are byte-equal (same RouteFlags/route_left_host).
        if ondevice_fused_partition_enabled() {
            partition_leaf_stable_fused(
                &f.bins,
                perm_range,
                f.num_bin,
                partition_min_bin,
                f.max_bin,
                f.default_bin,
                f.most_freq_bin,
                missing_type_u8,
                best.default_left,
                best.threshold,
            )?
        } else {
            // Off-switch (A/B baseline): the pre-change gather + narrow + partition_leaf_stable.
            let bins_sub = BinColumn::new(
                perm_range.iter().map(|&r| f.bins.bin(r as usize)).collect(),
                f.num_bin,
            );
            partition_leaf_stable(
                &bins_sub,
                perm_range,
                f.num_bin,
                partition_min_bin,
                f.max_bin,
                f.default_bin,
                f.most_freq_bin,
                missing_type_u8,
                best.default_left,
                best.threshold,
            )?
        }
    } else {
        // Gather the split feature's bins for the parent's resident row range (read THROUGH the
        // resident index buffer — NOT a cloned `parent_rows` Vec). The gather feeds the shipped
        // route kernel; the fully device-side gather-through-index scatter is deferred (see above).
        // The host arm no longer builds this; it is confined to the DEVICE arm, whose
        // fused gather-through-index kernel is a separate real-hardware track.
        let bins_sub = BinColumn::new(
            perm_range.iter().map(|&r| f.bins.bin(r as usize)).collect(),
            f.num_bin,
        );
        // ON-DEVICE route: `data_partition_resident_no_readback`
        // runs the §9 mark→prefix-sum→scatter and writes the child left/right ranges INTO the
        // resident `DeviceLeafSplits` slot ON DEVICE — the split point no longer crosses back as
        // the host scalar the grow loop consumed. It returns the stable-ordered GLOBAL row
        // permutation DIRECTLY, so the local→global `parent_rows` map-back
        // (`locals.map(|i| perm_range[i])`) is RETIRED (no permutation map-back
        // on the resident grow path).
        // One REAL on-device partition dispatch. Counted ONLY on this arm.
        bump_launch();
        // The child ranges are resident; the host reads only the split-point SCALAR from the
        // device struct for its `ResidentDriverLeaf` row bookkeeping (the fully device-side
        // child-row consumption — so even this scalar never crosses back — is a real-hardware
        // refinement, per the no-HIP scope note above). This single scalar read replaces
        // the prior split-point + route readback (count unchanged on this arm).
        bump_sync();
        let rows = backend.data_partition_resident_no_readback(
            client,
            &bins_sub,
            perm_range,
            f.num_bin,
            partition_min_bin,
            f.max_bin,
            f.default_bin,
            f.most_freq_bin,
            missing_type_u8,
            best.default_left,
            best.threshold,
            leaf_splits,
            leaf_id,
            p_begin,
            p_count,
        )?;
        let sp = leaf_splits.read_leaf(client, leaf_id).left_count as usize;
        (rows, sp)
    };
    // Scatter the stable-ordered global row ids back into the resident buffer IN PLACE. The
    // immutable borrows (`bins_sub`, the partition input) have ended; `reordered` is owned.
    perm_range.copy_from_slice(&reordered);
    Ok(split_point)
}

/// The winner's REAL feature index (`meta_->real_feature_index`), the §8.2/§8.3 tie-break key —
/// `-1` for a no-split leaf (`fpos < 0`), so a no-split slot sorts last (`i32::MAX` in the fold).
#[cfg_attr(not(feature = "gpu"), allow(dead_code))]
fn real_feat_of(features: &[GrowFeature], fpos: i32) -> i32 {
    if fpos < 0 {
        -1
    } else {
        features[fpos as usize].real_feature_index
    }
}

/// Fold ONE leaf's winning split into the DEVICE-RESIDENT frontier slot `leaf` (§8.2) via
/// [`crate::Backend::frontier_reduce_leaf_device`] — the device-side replacement
/// for the retired host cross-leaf argmax over a `leaf_bests: Vec<SplitInfo>` collection.
///
/// The `winner` is already the cross-FEATURE reduce output of this leaf's resident scan-argmax
/// (`scan_resident_leaf_argmax` / `scan_resident_siblings_argmax`, §8.2 on device — bit-exact to
/// the host `argmax_over_resident_splits`). It is uploaded as a single-record `SplitSoa` slab
/// (length `2·num_tasks = 2`: record 0 = the winner, record 1 = an invalid pad — the reduce's
/// `[base, base+num_tasks)` read-window contract) and folded into `frontier.records[leaf]`
/// ON DEVICE (device→device, no readback). `real_feat` is the winner's real feature index (`-1`
/// for a no-split leaf), the §8.3 tie-break key. An un-set / self-invalidated slot admits the
/// first valid record, so re-seeding a re-used leaf id after the §8.3 self-invalidation is
/// correct.
///
/// # Errors
/// Propagates [`crate::Backend::frontier_reduce_leaf_device`] (device-resident frontier
/// unsupported on the CpuBackend anchor — never reached: the resident grow only runs on a
/// `resident_pool_supported()` backend; the CpuBackend structure gate falls to the anchor fold).
#[cfg_attr(not(feature = "gpu"), allow(dead_code))]
fn reduce_winner_into_frontier<B, R>(
    backend: &B,
    client: &cubecl::prelude::ComputeClient<R>,
    frontier: &crate::DeviceFrontier<R>,
    leaf: usize,
    winner: &SplitInfo,
    real_feat: i32,
) -> Result<(), ComputeError>
where
    B: crate::Backend<Runtime = R>,
    R: cubecl::Runtime,
{
    // `SplitInfo::none()` carries `gain == NEG_INFINITY` — a no-valid-split leaf reduces as
    // `is_valid = false` so the §8.3 pick skips it (exactly as the host guard `best_fpos < 0`).
    let is_valid = winner.gain > f64::NEG_INFINITY;
    let rec = SplitScalars {
        is_valid,
        leaf_index: leaf as i32,
        gain: winner.gain,
        inner_feature_index: real_feat,
        threshold: winner.threshold,
        default_left: winner.default_left,
        num_cat_threshold: 0,
        // Carry the winner's 4 child grad/hess sums into the resident
        // frontier so the §8.3 pick export can hand the driver the next build's seed sums FRESH
        // — without folding them here they would stay at the 0.0 default and the re-sourced
        // seed would be wrong → NaN-collapse.
        left_sum_gradients: winner.left_sum_gradient,
        left_sum_hessians: winner.left_sum_hessian,
        right_sum_gradients: winner.right_sum_gradient,
        right_sum_hessians: winner.right_sum_hessian,
        // Carry the winner's child leaf OUTPUTS into the resident frontier too
        // so the §8.3 pick export hands the driver the tree-record `left_output`/`right_output`
        // FRESH (retiring the host `leaves[best_leaf].best.left_output/.right_output` read). The
        // ROOT seed + the NOT-scannable `SplitInfo::none()` path both flow through here; the
        // SCANNED per-split children fold their winners device→device via the reduce launchers.
        left_value: winner.left_output,
        right_value: winner.right_output,
        ..SplitScalars::default()
    };
    let pad = SplitScalars {
        is_valid: false,
        ..SplitScalars::default()
    };
    // The winner upload + §8.2 reduce dispatch are async (no readback) — the
    // REDUCE bucket holds submission time in free-run mode; drain mode books device time here.
    time_phase(&GROW_REDUCE_NS, || {
        let in_slab = crate::kernels::best_split::SplitSoa::from_records(client, &[rec, pad]);
        // One device §8.2 reduce dispatch (device→device, NO readback — no bump_sync).
        bump_launch();
        let r = backend.frontier_reduce_leaf_device(client, frontier, &in_slab, 1, true, leaf);
        if r.is_ok() {
            grow_drain(client);
        }
        r
    })
}

/// Grow an ENTIRE continuous-feature + L2 tree on a RESIDENT-capable backend by keeping
/// every leaf's histogram DEVICE-RESIDENT (build/subtract/scan in pool slots), sequenced
/// in the `SerialTreeLearner` best-first order with the driver's OWN
/// [`ResidentDriverLeaf`] bookkeeping. Bins are uploaded ONCE per grow
/// (`upload_resident_bins`); the row partition still routes on the HOST
/// (`partition_leaf_stable`). Called from
/// [`grow_tree_on_device_driver_with_cfg`]'s resident capability arm for the NUMERIC
/// proving slice; the caller guards `features.is_empty()` / `num_leaves < 1` /
/// hessian-length and demotes categorical to the anchor fold.
///
/// # Errors
/// [`ComputeError`] from any resident launcher (empty resident cache / slot, bad
/// slot length, device launch), an out-of-range split threshold bin, or a
/// kernel/driver leaf-id desync.
#[allow(clippy::too_many_arguments)]
fn grow_tree_on_device_resident<B, R>(
    backend: &B,
    client: &cubecl::prelude::ComputeClient<R>,
    gradients: &[f32],
    hessians: &[f32],
    features: &[GrowFeature],
    num_leaves: i32,
    max_depth: i32,
    cfg: &GainConfig,
) -> Result<(lgbm_model::Tree, LeafPartitionLayout), ComputeError>
where
    B: crate::Backend<Runtime = R>,
    R: cubecl::Runtime,
{
    let num_data = gradients.len();
    let min_data = cfg.min_data_in_leaf;

    // Whole-grow wall guard (RAII — a `?` early exit still books the partial
    // span, keeping wall ≥ Σ(component buckets) invariant for the dump's host_other).
    let _grow_wall = PhaseGuard::new(&GROW_WALL_NS);
    // SETUP: host-side per-grow input shaping (slot offsets, feats/fix_feats
    // vectors) + the resident-pool mirror reset. Explicitly dropped before the uploads.
    let setup_guard = PhaseGuard::new(&GROW_SETUP_NS);

    // Per-feature concatenated-histogram offsets (2*num_bin cells each). `slot_len` is
    // the resident pool slot width (== the whole concatenated buffer length).
    let mut slot_off = Vec::with_capacity(features.len());
    let mut hist_len = 0usize;
    for f in features {
        slot_off.push(hist_len);
        hist_len += 2 * f.num_bin as usize;
    }
    let slot_len = hist_len;

    // Resident-build inputs, shared across every leaf (built ONCE):
    // - `feature_bins`: each feature's GLOBAL-row bin column (uploaded once).
    // - `num_bins` / `fix_feats`: the per-feature bin counts + FixHistogram/compaction
    //   descriptor the NON-fused resident build (`build_resident_leaf`) consumes on the
    //   CO-PACK path (smaller child built resident-only, scan deferred to the co-packed
    //   2-slot launch). The fused path reads the SAME layout out of `feats` instead.
    // - `feats`: the fpos-ORDERED `BatchedSplitFeature` layout that carries BOTH the
    //   FixHistogram+compaction descriptor (num_bin/offset/most_freq_bin) the fused
    //   `build_fix_scan_resident` applies AND the scan dispatch flags. Order-preserving
    //   returns keep the cross-feature argmax tie-break bit-exact; the
    //   dispatch flags match `scan_leaf`'s per-feature computation.
    let feature_bins: Vec<&BinColumn> = features.iter().map(|f| &f.bins).collect();
    let num_bins: Vec<u32> = features.iter().map(|f| f.num_bin).collect();
    let fix_feats: Vec<(usize, u32, i32, u32)> = features
        .iter()
        .enumerate()
        .map(|(fpos, f)| (slot_off[fpos], f.num_bin, f.offset, f.most_freq_bin))
        .collect();
    let feats: Vec<BatchedSplitFeature> = features
        .iter()
        .enumerate()
        .map(|(fpos, f)| BatchedSplitFeature {
            slot_off: slot_off[fpos],
            num_bin: f.num_bin,
            offset: f.offset,
            default_bin: f.default_bin,
            most_freq_bin: f.most_freq_bin,
            skip_default_bin: f.num_bin > 2 && f.missing_type == MissingType::Zero,
            na_as_missing: f.num_bin > 2 && f.missing_type == MissingType::NaN,
            run_forward: f.num_bin > 2 && f.missing_type == MissingType::Zero,
        })
        .collect();
    // The fused `build_fix_scan_resident` SCANS `scan_active[fpos]` features and returns
    // one SplitInfo per active feature IN ORDER. The driver scans EVERY feature (like
    // `scan_resident_and_argmax`, which passes the full `feats` and drops `na_as_missing`
    // in the argmax), so all features are scan-active ⇒ the returned Vec is one SplitInfo
    // per feature in fpos order, aligning 1:1 with `feats`/`features` for the argmax.
    let scan_active: Vec<bool> = vec![true; features.len()];
    // The fpos-ordered real-feature-index vector — the tie-break KEY the
    // on-device cross-feature argmax (`scan_resident_leaf_argmax` / `scan_resident_siblings_argmax`,
    // §8.2) folds by, so the winner is bit-identical to the host `argmax_over_splits`.
    let real_feats: Vec<i32> = features.iter().map(|f| f.real_feature_index).collect();

    // ---- ONCE per grow: size the device pool mirror (num_leaves slots) + upload the
    // binned columns to the device (no per-leaf host re-gather). ----
    backend.reset_resident_pool(num_leaves as usize, slot_len);
    drop(setup_guard); // End of SETUP.
    // SKIP the per-grow re-upload of the IMMUTABLE binned columns when the learner's
    // once-per-train guarded upload is already resident AND pinned with matching
    // geometry — this re-upload was previously a substantial fraction of the
    // remaining on-device gap on real CUDA. Un-pinned callers (anything that never
    // ran the learner's guard) take the original upload branch, byte-unchanged.
    if !(ondevice_bin_hoist_enabled() && backend.resident_bins_pinned(features.len(), num_data)) {
        // One REAL per-train device dispatch (the once-per-grow bin upload).
        // `reset_resident_pool` sizes a host-side handle mirror (no device launch) and is NOT
        // counted; `upload_resident_bins` moves the binned columns host→device, so it is.
        bump_launch();
        // UPLOAD: the once-per-grow host→device bin upload.
        time_phase(&GROW_UPLOAD_NS, || {
            backend.upload_resident_bins(client, &feature_bins);
            grow_drain(client);
        });
    }

    // ---- ONCE per grow: upload this tree's grad/hess to the device
    // (mirror of the once-per-grow `upload_resident_bins`). grad/hess are constant across the
    // whole tree grow, so the resident build gathers each leaf's grad/hess ON DEVICE from
    // these buffers via the leaf-row index (no per-build host `ord_g`/`ord_h` gather +
    // re-upload). This is a single linear statement (grow runs once per tree, with no loop
    // over it), so it needs no once-guard flag. Counted as one REAL host→device dispatch
    // (like the bin upload). ----
    bump_launch();
    time_phase(&GROW_UPLOAD_NS, || {
        backend.upload_resident_grad_hess(client, gradients, hessians);
        grow_drain(client);
    });

    // ---- Root init (§6.1): whole-dataset root grad/hess sum. Routed through
    // `backend.root_grad_hess_sum` (the §6.1 `CUDAInitValuesKernel` analog) so the GpuBackend arm
    // runs it on-device (ordered f64 reduction — bit-exact vs the integer path, ~1e-6 vs
    // host-CUDA, NEVER GPU-f32-vs-GPU-f32), retiring the host f64 root fold on that
    // arm. `root_rows == 0..num_data` in order, so this is the identical ascending fold. ----
    // The SINGLE resident permutation buffer for the WHOLE grow — the
    // `cuda_data_indices_` mirror. Identity at the root (`0..num_data` in order, so the root build
    // + root grad/hess fold below are the identical ascending order). Every split partitions a
    // leaf's sub-range of THIS buffer in place (`partition_resident_range`); the buffer is never
    // reallocated and no per-leaf `rows: Vec<u32>` is ever cloned.
    let mut perm: Vec<u32> =
        time_phase(&GROW_SETUP_NS, || (0..num_data as u32).collect());
    // ROOTFOLD: this is the bit-exact HOST anchor fold; its own bucket keeps the
    // root-fold cost visible if a device fold is ever introduced here.
    let (root_sum_g, root_sum_h) =
        time_phase(&GROW_ROOTFOLD_NS, || backend.root_grad_hess_sum(client, gradients, hessians));
    // Fail loudly HERE on a non-finite root seed rather than grow a 1-leaf tree GBDT's
    // pop-loop would silently truncate.
    check_root_seed_finite(root_sum_g, root_sum_h)?;
    // The ROOT histogram is built device-resident into slot 0 and then scanned. The
    // fixed+compacted f64 Handle is stored into slot 0 so a later split can still
    // subtract from it if the root becomes a parent. The root has no sibling, so its
    // build+scan is never co-packed (co-pack only applies to the split loop's two
    // children).
    //
    // DEFAULT: the on-device root build runs the PARALLEL u64 fixed-point resident
    // build (`build_resident_leaf`) followed by a separate resident SCAN — this
    // replaced an f64 single-owner `build_fix_scan_resident` fused kernel that measured
    // substantially slower on real NVIDIA hardware. `build_resident_leaf` stores the
    // fixed+compacted f64 Handle into slot 0 (identical to what the fused path stored),
    // so a later split can still subtract from the root. The f64 fused build is
    // reachable ONLY via the `LGBM_ONDEVICE_F64_FUSED=1` A/B escape hatch, NEVER
    // default.
    let (root_best, root_fpos) = if on_device_f64_fused_build() {
        // ESCAPE HATCH ONLY (A/B): the old f64 single-owner fused build+scan.
        bump_launch();
        bump_f64_fused();
        // The fused build+scan reads its SplitInfos back — ONE blocking readback.
        bump_sync();
        // The f64 fused build+scan is ONE launch — booked to BUILD (escape hatch
        // only, never the default arm; a nonzero f64_fused counter flags the attribution).
        let root_splits = time_phase(&GROW_BUILD_NS, || {
            backend.build_fix_scan_resident(
                client,
                0,
                &slot_off,
                slot_len,
                &perm,
                gradients,
                hessians,
                &feats,
                &scan_active,
                cfg,
                root_sum_g,
                root_sum_h,
                num_data as i32,
            )
        })?;
        argmax_over_splits(&root_splits, &feats, features)
    } else {
        // DEFAULT: parallel-u64 resident BUILD into slot 0, then resident SCAN.
        bump_launch(); // One parallel-u64 resident build dispatch.
        bump_rootbuild_u64(); // CONVERTED root site ran u64 (positive proof).
        time_phase(&GROW_BUILD_NS, || -> Result<(), ComputeError> {
            backend.build_resident_leaf(
                client,
                0,
                &feature_bins,
                &num_bins,
                &slot_off,
                slot_len,
                &perm,
                gradients,
                hessians,
                &fix_feats,
                root_sum_g,
                root_sum_h,
            )?;
            grow_drain(client);
            Ok(())
        })?;
        // `scan_resident_and_argmax` bumps its own scan dispatch + applies the
        // `!(sum_h>0)||num_data<=0` short-circuit (never fires at the root here). The
        // cross-feature argmax runs on-device via `scan_resident_leaf_argmax` (§8.2).
        scan_resident_and_argmax(
            backend,
            client,
            0,
            slot_len,
            &feats,
            &real_feats,
            cfg,
            root_sum_g,
            root_sum_h,
            num_data as i32,
        )?
    };

    let mut leaves: Vec<ResidentDriverLeaf> = vec![ResidentDriverLeaf {
        row_begin: 0,
        row_count: num_data,
        sum_g: root_sum_g,
        sum_h: root_sum_h,
        slot: 0,
        best: root_best,
        best_fpos: root_fpos,
        depth: 0,
    }];
    // Slot allocator: root owns slot 0; each split hands the SMALLER child a fresh slot
    // and the LARGER child the parent's slot, so `num_leaves` slots suffice.
    let mut next_slot = 1usize;

    // ---- The device flat tree, pre-allocated once. ----
    // Once-per-grow device-struct allocations are SETUP.
    let mut tree = time_phase(&GROW_SETUP_NS, || {
        DeviceCudaTree::<R>::new(client, num_leaves as usize, num_data as i32)
    })?;
    // Seed the root leaf value so a never-split root still matches the anchor.
    let root_output = calculate_splitted_leaf_output(
        cfg.use_l1(),
        root_sum_g,
        root_sum_h,
        cfg.lambda_l1,
        cfg.lambda_l2,
    );
    tree.add_bias(client, root_output);

    // ---- The DEVICE-RESIDENT best-first FRONTIER (§8.2/§8.3). The
    // cross-leaf `best_leaf` pick now runs ON DEVICE — the host `best_leaf_reduce` argmax over a
    // materialized `leaf_bests: Vec<SplitInfo>` collection is RETIRED. Each leaf's winning split
    // is folded into the resident frontier SoA (`reduce_winner_into_frontier` → §8.2, device→device
    // no readback); `frontier_pick_best_leaf_device` (§8.3) picks `best_leaf` into a device slot,
    // self-invalidates the chosen + freshly-created slots, and exports only the ~8-int winner
    // (the per-iteration host-visible crossing). The §8.3
    // tie-break mirrors the cpu-f64 merge-gate anchor `split_gt` EXACTLY: on an
    // exact gain tie the LOWER real feature index wins (`-1 ⇒ i32::MAX`), then the lower leaf
    // index — the resident frontier stores the winner's REAL feature index in `feat`
    // (`reduce_winner_into_frontier` folds `real_feat_of(...)` into it), so the device pick and
    // the anchor agree on ties, keeping the resident tree bit-exact to the anchor even when the
    // corpus DOES tie (the hard merge gate). ----
    let frontier =
        time_phase(&GROW_SETUP_NS, || crate::DeviceFrontier::<R>::new(client, num_leaves as usize));
    // Seed the ROOT's winning split into frontier slot 0 so the first device pick sees it.
    reduce_winner_into_frontier(
        backend,
        client,
        &frontier,
        0,
        &root_best,
        real_feat_of(features, root_fpos),
    )?;
    // The children created by the PREVIOUS split — the §8.3 self-invalidation targets + the
    // (ignored here) smaller/larger export cells. The root iteration has no prior children.
    let mut prev_smaller: i32 = 0;
    let mut prev_larger: i32 = -1;

    // ---- The DEVICE-RESIDENT child-range struct. Each split's
    // partition (on the on-device arm) writes its child left/right start/end/count into
    // `leaf_splits_dev[leaf_id]` ON DEVICE — the split point stays resident, not a host scalar
    // the grow loop consumes. Allocated once (num_leaves slots), like the frontier. ----
    let leaf_splits_dev = time_phase(&GROW_SETUP_NS, || {
        crate::kernels::partition::DeviceLeafSplits::<R>::new(client, num_leaves as usize)
    })?;

    // ---- A real_feature_index -> feature-position (`fpos`) reverse
    // lookup, built ONCE per grow from the immutable `features` slice. The §8.3 pick export
    // carries the winning leaf's REAL feature index (the resident frontier `feat` stores real,
    // via `reduce_winner_into_frontier`); the driver needs the feature POSITION to index
    // `features` / drive the partition, so it INVERTS the forward `real_feat_of` map here rather
    // than reading the host `leaves[best_leaf].best_fpos` cache. `real_feature_index` is a
    // bijection over the active `features`, so the map is unambiguous. ----
    let real_feat_to_fpos: std::collections::HashMap<i32, i32> = features
        .iter()
        .enumerate()
        .map(|(fpos, f)| (f.real_feature_index, fpos as i32))
        .collect();

    // ---- The best-first leaf-wise loop (serial_tree_learner.cpp:218-236): a FIXED
    // `num_leaves - 1` device schedule, broken early by the device stop signal
    // (`best_leaf == -1`) — no host argmax drives which leaf grows next. ----
    for _split in 0..(num_leaves - 1) {
        // (§8.3): pick `best_leaf` ON DEVICE from the resident frontier — the ONLY
        // host-visible crossing this iteration is the ~8-int export (cell [6] = best_leaf, `-1`
        // = the stop signal). `prev_smaller`/`prev_larger` are the previous split's children
        // (the §8.3 self-invalidation targets); their export cells [0..6] are unused here (the
        // driver evaluates children directly, not the CUDA-pipelined smaller/larger prep).
        // This single export readback is bumped ONCE (batched async drain);
        // it REPLACES the free host argmax with the device pick, and retires a prior
        // split readback so the per-grow blocking-sync total does not rise.
        let cur_num_leaves = leaves.len();
        // The per-iteration host crossing is ONE deferred/async drain of
        // the §8.3 8-int export (the achievable equivalent of overlapping asynchronous work
        // — cubecl 0.10 cannot express true per-op multi-stream overlap). The device-handle
        // chain build→subtract→scan→§8.2 reduce→§8.3 pick hands off by device handle within
        // the iteration; only this export reads back, drained through the batched
        // single-drain path the backend advertises.
        debug_assert!(
            backend.supports_async_device_copy(),
            "the §8.3 per-iteration best-leaf export relies on cubecl async batched device→host \
             copy (client.read(Vec<Handle>)) for a single-drain crossing (ODF-04, M4/M5)"
        );
        // The §8.3 device pick (argmax + self-invalidation + export) is one logical device
        // dispatch; its 8-int export is the ONE per-iteration blocking readback (batched drain).
        bump_launch();
        bump_sync();
        // PICK: the §8.3 device pick + its 8-int blocking export — the once-per-
        // iteration sync; in free-run mode it also absorbs queued reduce/treesplit device time.
        let export = time_phase(&GROW_PICK_NS, || {
            backend.frontier_pick_best_leaf_device(
                client,
                &frontier,
                prev_smaller,
                prev_larger,
                cur_num_leaves,
            )
        })?;
        let best_leaf = export.cells[6] as i32;
        if best_leaf < 0 {
            break;
        }
        // The device frontier chose WHICH leaf AND now hands back the picked leaf's
        // COMPLETE node record in the SAME per-iteration export (`export.winner`). Source
        // EVERY tree-record field FRESH from that export — threshold, default_left, the
        // REAL feature index (→ `fpos` via the reverse lookup), the 4 child grad/hess sums
        // (kEpsilon-carrying build seeds, Pitfall 2), AND the node-recording `gain` (NET) +
        // child leaf OUTPUTS (the frontier SoA + export carry `gain`/`left_output`/
        // `right_output` device→device). The host cache `leaves[best_leaf].best`/`.best_fpos`
        // is NO LONGER read here — the full export means the SCANNED-case cache writes are
        // dead (see the per-split arms below). The export values are bit-identical to what
        // the old scan-readback cache carried (the reduce launchers carry the SAME winning
        // feature's raw scan cells the host decode read; net gain is `(raw - min_gain_shift)
        // * penalty` bit-for-bit), so the tree grown is bit-identical.
        // `left_count`/`right_count` are recomputed from the partition `split_point` below (they
        // were never read off the cache), so 0 here is inert.
        let w = export.winner; // [feat(real), thr, dleft, lsum_g, lsum_h, rsum_g, rsum_h, gain, lout, rout]
        let best_real_feat = w[0] as i32;
        let best_fpos = *real_feat_to_fpos.get(&best_real_feat).ok_or_else(|| {
            ComputeError::Runtime {
                detail: format!(
                    "grow_tree_on_device_resident: picked real feature index {best_real_feat} \
                     has no feature-position in the grow feature set (reverse-lookup miss)"
                ),
            }
        })?;
        let best = SplitInfo {
            threshold: w[1] as u32,
            default_left: w[2] > 0.5,
            left_sum_gradient: w[3],
            left_sum_hessian: w[4],
            right_sum_gradient: w[5],
            right_sum_hessian: w[6],
            gain: w[7],
            left_output: w[8],
            right_output: w[9],
            left_count: 0,
            right_count: 0,
        };
        if best_fpos < 0 || !(best.gain > 0.0) {
            break;
        }

        let f = &features[best_fpos as usize];
        let parent_depth = leaves[best_leaf as usize].depth;
        let parent_slot = leaves[best_leaf as usize].slot;

        // ---- Partition the parent leaf's rows (§9). This is the LAST
        // per-leaf compute phase; keep the row permutation RESIDENT by routing on-device
        // via `backend.data_partition_native` (a narrow-width
        // mark→prefix-sum→scatter) UNLESS the backend prefers the host route. The arm is
        // chosen by `backend.prefers_host_partition()` (default-ON for
        // `RocmBackend` because on the shared-DDR5 APU the device round-trip is pure
        // overhead; that calculus may INVERT on real discrete CUDA over PCIe — this is an
        // unmeasured assumption). `LGBM_ROCM_HOST_PARTITION=0` forces `prefers_host_partition()`
        // false → the on-device arm, an A/B device selector. Both arms produce the SAME
        // stable (left-then-right, original relative order) partition the cpu f64 anchor
        // produces. ----
        let missing_type_u8 = match f.missing_type {
            MissingType::None => 0u8,
            MissingType::Zero => 1,
            MissingType::NaN => 2,
        };
        let missing_type_code = i32::from(missing_type_u8);
        let new_left = best_leaf;

        // Single-feature-group min_bin convention: min_bin + offset.
        let partition_min_bin = f.min_bin + f.offset.max(0) as u32;

        // (§9): partition the parent leaf's RESIDENT index sub-range IN
        // PLACE. The parent's global row ids live in `perm[p_begin..p_begin+p_count]`; the
        // partition reorders them into stable left-then-right order (left rows first in original
        // relative order, then right) and the two children become the adjacent sub-ranges
        // `[p_begin, p_begin+split_point)` and `[p_begin+split_point, p_begin+p_count)`. NO
        // `rows.clone()`, NO `bins_sub` gather off a cloned `parent_rows`, NO local→global
        // `parent_rows` map-back Vec — the single resident permutation buffer is updated in place.
        // Both arms (host / on-device) yield the SAME stable ordering the cpu f64 anchor produces
        // (`partition_native_*_matches_widened` / `partition_leaf_stable` gates). The arm is chosen
        // by `backend.prefers_host_partition()` inside `partition_resident_range`.
        let p_begin = leaves[best_leaf as usize].row_begin;
        let p_count = leaves[best_leaf as usize].row_count;
        // PARTITION: the whole per-split row-routing op (host bin gather + the
        // host-fused OR device route + the in-place perm scatter-back).
        let split_point = time_phase(&GROW_PARTITION_NS, || {
            partition_resident_range(
                backend,
                client,
                &mut perm[p_begin..p_begin + p_count],
                f,
                &best,
                partition_min_bin,
                missing_type_u8,
                &leaf_splits_dev,
                new_left as usize,
                p_begin as i32,
                p_count as i32,
            )
        })?;
        let left_count = split_point as i32;
        let right_count = (p_count - split_point) as i32;
        // An out-of-range `best.threshold` would silently record a wrong REAL
        // threshold — surface a typed error instead (mirrors the anchor branch).
        let real_threshold =
            *f.bin_upper_bound.get(best.threshold as usize).ok_or_else(|| {
                ComputeError::Runtime {
                    detail: format!(
                        "grow_tree_on_device_resident: split threshold bin index {} out of \
                         range for feature {} bin_upper_bound (len {})",
                        best.threshold,
                        f.real_feature_index,
                        f.bin_upper_bound.len()
                    ),
                }
            })?;
        let scalars = SplitScalars {
            is_valid: true,
            leaf_index: best_leaf,
            gain: best.gain + cfg.min_gain_to_split,
            inner_feature_index: f.real_feature_index,
            threshold: best.threshold,
            default_left: best.default_left,
            left_sum_gradients: best.left_sum_gradient,
            left_sum_hessians: best.left_sum_hessian,
            left_sum_gh_quant: 0,
            left_count,
            left_gain: 0.0,
            left_value: best.left_output,
            right_sum_gradients: best.right_sum_gradient,
            right_sum_hessians: best.right_sum_hessian,
            right_sum_gh_quant: 0,
            right_count,
            right_gain: 0.0,
            right_value: best.right_output,
            num_cat_threshold: 0,
        };
        // The NO-READBACK scheduled device tree split — the right
        // child leaf id is SUPPLIED by the fixed grow schedule (`leaves.len()`), NOT read back
        // from the kernel (retires a `right_leaf_index` `bump_sync`). `split_tree_scheduled`
        // asserts the desync invariant (`right == tree.num_leaves`) host-side as a
        // pre-condition WITHOUT a readback, so the driver consumes no returned value.
        let new_right = leaves.len() as i32;
        // TREESPLIT: the no-readback scheduled device tree mutation (async).
        time_phase(&GROW_TREESPLIT_NS, || -> Result<(), ComputeError> {
            backend.split_tree_scheduled(
                client,
                &mut tree,
                best_leaf,
                new_right,
                f.real_feature_index,
                real_threshold,
                missing_type_code,
                &scalars,
            )?;
            grow_drain(client);
            Ok(())
        })?;

        // ---- Assign child pool slots: SMALLER gets a FRESH slot, LARGER reuses the
        // parent slot (its resident Handle is the subtraction's parent input). ----
        let child_depth = parent_depth + 1;
        let smaller_is_left = left_count < right_count;
        let (smaller_leaf, larger_leaf) = if smaller_is_left {
            (new_left, new_right)
        } else {
            (new_right, new_left)
        };
        let smaller_slot = next_slot;
        next_slot += 1;
        let larger_slot = parent_slot;
        let left_slot = if smaller_is_left { smaller_slot } else { larger_slot };
        let right_slot = if smaller_is_left { larger_slot } else { smaller_slot };

        // Seed the two child leaves from the SplitInfo (NOT a re-fold): the
        // kEpsilon-carrying sums are load-bearing for the next split (Pitfall 2).
        // The children are the adjacent RESIDENT sub-ranges of the parent's just-partitioned span
        // (left = `[p_begin, p_begin+split_point)`, right = `[p_begin+split_point, p_begin+p_count)`)
        // — no `rows` Vec is materialized; the resident permutation buffer already holds them.
        {
            let l = &mut leaves[best_leaf as usize];
            l.row_begin = p_begin;
            l.row_count = split_point;
            l.sum_g = best.left_sum_gradient;
            l.sum_h = best.left_sum_hessian;
            l.slot = left_slot;
            l.depth = child_depth;
            l.best = SplitInfo::none();
            l.best_fpos = -1;
        }
        leaves.push(ResidentDriverLeaf {
            row_begin: p_begin + split_point,
            row_count: p_count - split_point,
            sum_g: best.right_sum_gradient,
            sum_h: best.right_sum_hessian,
            slot: right_slot,
            best: SplitInfo::none(),
            best_fpos: -1,
            depth: child_depth,
        });

        // ---- Build the children histograms device-resident and scan both. ORDER (always,
        // both arms): build-SMALLER → subtract-LARGER (Pitfall 7 —
        // `subtract_resident` reads the smaller slot; NEVER defer the smaller BUILD past
        // the subtract). `larger_slot == parent_slot`, so the subtraction overwrites the
        // now-consumed parent Handle with the derived larger child. ----
        // Read the smaller child's rows as a BORROWED slice of the resident
        // permutation buffer (no `rows.clone()`) — the build gathers grad/hess+bins THROUGH this
        // resident index range (on-device grad/hess gather via `leaf_rows[k]`).
        let (s_begin, s_count, s_g, s_h) = {
            let s = &leaves[smaller_leaf as usize];
            (s.row_begin, s.row_count, s.sum_g, s.sum_h)
        };
        let s_rows: &[u32] = &perm[s_begin..s_begin + s_count];
        let s_n = s_count as i32;
        let (l_g, l_h, l_n) = {
            let l = &leaves[larger_leaf as usize];
            (l.sum_g, l.sum_h, l.row_count as i32)
        };
        // BeforeFindBestSplit per-leaf gates: a too-small / depth-capped /
        // sum_h<=0 child is not scannable and records `none` (mirrors C++
        // `BeforeFindBestSplit` + the finder's `!(sum_h>0)||num_data<=0` short-circuit).
        let min_data_x2 = min_data.saturating_mul(2);
        #[allow(clippy::neg_cmp_op_on_partial_ord)]
        let smaller_scannable = s_n >= min_data_x2
            && !(max_depth > 0 && leaves[smaller_leaf as usize].depth >= max_depth)
            && s_h > 0.0
            && s_n > 0;
        #[allow(clippy::neg_cmp_op_on_partial_ord)]
        let larger_scannable = l_n >= min_data_x2
            && !(max_depth > 0 && leaves[larger_leaf as usize].depth >= max_depth)
            && l_h > 0.0
            && l_n > 0;

        // CO-PACK: default-ON (`LGBM_SIBLING_COPACK != "0"`) when BOTH
        // siblings are simultaneously resident AND scannable — the smaller slot survives
        // `subtract_resident`, the larger is its derived output, so ONE
        // `scan_resident_siblings` launch (+ ONE readback) scans both. Bit-exact by
        // construction: each feature's sequential scan is identical to the two single-slot
        // scans; only WHICH launch runs it changes ⇒ tree STRUCTURE is unchanged.
        // The f64-fused A/B hatch must actually revert the CHILD builds,
        // not just the root. Co-pack (default-ON) otherwise runs the u64 `build_resident_leaf`
        // whenever both siblings are scannable, so `LGBM_ONDEVICE_F64_FUSED=1` alone would
        // leave every smaller child on u64 and contaminate the u64-vs-f64 delta. Gating
        // `use_copack` on `!on_device_f64_fused_build()` forces the separate-scan f64 arm to
        // take over the children too, so the hatch does what its name/docs claim WITHOUT
        // requiring the operator to ALSO set `LGBM_SIBLING_COPACK=0`.
        let use_copack = sibling_copack_enabled()
            && !on_device_f64_fused_build()
            && smaller_scannable
            && larger_scannable;
        // Each arm folds BOTH children's winners DIRECTLY into the resident
        // frontier (`frontier.records()` slots `smaller_leaf`/`larger_leaf`). The §8.3 pick
        // self-invalidated the parent (`best_leaf`, reused by the left child) + the freshly-created
        // slot; the reduce launchers RE-SEED those slots. The host `leaves[...].best`/`.best_fpos`
        // cache is NO LONGER written for the scanned case (its only reader, the consumption site
        // above, now reads the device export) — the export fully sources the tree record. Only
        // the RESET writes (`l.best = SplitInfo::none()` at leaf creation)
        // remain. The anchor-fold branch (`grow_tree_on_device_driver_with_cfg`, a different
        // function) keeps its own `.best`/`.best_fpos` usage — untouched.
        if use_copack {
            // CO-PACK arm (default-ON): build the SMALLER child resident-ONLY (scan
            // DEFERRED to the co-pack), subtract to derive the LARGER child, then CO-SCAN both and
            // fold EACH sibling's winner DIRECTLY into its frontier slot in ONE launch —
            // device→device, NO readback (retires the co-pack per-split scan `bump_sync`,
            // Pitfall 4's MOST commonly hit arm). The precondition guarantees both siblings are
            // scannable, so there is no not-scannable branch. Bit-exact to two
            // `argmax_over_resident_splits` folds + `reduce_winner_into_frontier` on the anchor.
            bump_launch(); // One resident build dispatch (scan deferred to co-pack).
            time_phase(&GROW_BUILD_NS, || -> Result<(), ComputeError> {
                backend.build_resident_leaf(
                    client, smaller_slot, &feature_bins, &num_bins, &slot_off, slot_len,
                    s_rows, gradients, hessians, &fix_feats, s_g, s_h,
                )?;
                grow_drain(client);
                Ok(())
            })?;
            debug_assert_ne!(
                smaller_slot, larger_slot,
                "resident subtract slot aliasing (smaller must own a fresh slot)"
            );
            bump_launch(); // One on-device subtraction-trick dispatch.
            time_phase(&GROW_SUBTRACT_NS, || -> Result<(), ComputeError> {
                backend.subtract_resident(client, parent_slot, smaller_slot, larger_slot, slot_len)?;
                grow_drain(client);
                Ok(())
            })?;
            bump_launch(); // One CO-PACKED 2-slot scan + device fold (NO readback).
            time_phase(&GROW_SCAN_NS, || -> Result<(), ComputeError> {
                backend.scan_resident_siblings_into_frontier(
                    client, smaller_slot, larger_slot, slot_len, &feats, &real_feats, cfg,
                    (s_g, s_h, s_n), (l_g, l_h, l_n),
                    &frontier, smaller_leaf as usize, larger_leaf as usize,
                )?;
                grow_drain(client);
                Ok(())
            })?;
        } else if on_device_f64_fused_build() {
            // f64-fused ESCAPE HATCH (A/B, LGBM_ONDEVICE_F64_FUSED=1): build+fix+compact+scan the
            // smaller child and fold its winner DIRECTLY into the frontier (device→device, NO
            // readback). The BUILD runs UNCONDITIONALLY (required for the subtraction trick
            // even when the smaller child is unscannable); when NOT scannable, an all-false scan
            // mask makes every window decode `is_splittable=0` so the frontier slot gets the
            // no-split sentinel (histogram still built for the subtract). Subtract to derive the
            // larger child, then scan it into the frontier (or seed the sentinel).
            let smaller_active: Vec<bool> = if smaller_scannable {
                scan_active.clone()
            } else {
                vec![false; scan_active.len()]
            };
            bump_launch(); // One fused build+fix+compact+scan dispatch (device→device fold).
            bump_f64_fused();
            time_phase(&GROW_BUILD_NS, || -> Result<(), ComputeError> {
                backend.build_fix_scan_resident_into_frontier(
                    client, smaller_slot, &slot_off, slot_len, s_rows, gradients, hessians,
                    &feats, &smaller_active, &real_feats, cfg, s_g, s_h, s_n,
                    &frontier, smaller_leaf as usize,
                )?;
                grow_drain(client);
                Ok(())
            })?;
            debug_assert_ne!(
                smaller_slot, larger_slot,
                "resident subtract slot aliasing (smaller must own a fresh slot)"
            );
            bump_launch(); // One on-device subtraction-trick dispatch.
            time_phase(&GROW_SUBTRACT_NS, || -> Result<(), ComputeError> {
                backend.subtract_resident(client, parent_slot, smaller_slot, larger_slot, slot_len)?;
                grow_drain(client);
                Ok(())
            })?;
            if larger_scannable {
                bump_launch(); // One single-slot scan dispatch (device→device fold).
                time_phase(&GROW_SCAN_NS, || -> Result<(), ComputeError> {
                    backend.scan_resident_leaf_into_frontier(
                        client, larger_slot, slot_len, &feats, &real_feats, cfg, l_g, l_h, l_n,
                        &frontier, larger_leaf as usize,
                    )?;
                    grow_drain(client);
                    Ok(())
                })?;
            } else {
                reduce_winner_into_frontier(
                    backend, client, &frontier, larger_leaf as usize, &SplitInfo::none(), -1,
                )?;
            }
        } else {
            // DEFAULT: parallel-u64 resident BUILD of the smaller child,
            // subtract to derive the larger child, then fold EACH scannable child's winner DIRECTLY
            // into the frontier (device→device, NO host argmax readback — retires the
            // per-split scan `bump_sync` the two `scan_resident_and_argmax` calls issued). A
            // NOT-scannable child seeds the no-split sentinel via `reduce_winner_into_frontier`.
            // The BUILD runs UNCONDITIONALLY (required for the subtraction trick even when the
            // smaller child is unscannable — only its scan is skipped).
            bump_launch(); // One parallel-u64 resident build dispatch.
            bump_rootbuild_u64(); // CONVERTED directly-built site ran u64.
            time_phase(&GROW_BUILD_NS, || -> Result<(), ComputeError> {
                backend.build_resident_leaf(
                    client, smaller_slot, &feature_bins, &num_bins, &slot_off, slot_len,
                    s_rows, gradients, hessians, &fix_feats, s_g, s_h,
                )?;
                grow_drain(client);
                Ok(())
            })?;
            debug_assert_ne!(
                smaller_slot, larger_slot,
                "resident subtract slot aliasing (smaller must own a fresh slot)"
            );
            bump_launch(); // One on-device subtraction-trick dispatch.
            time_phase(&GROW_SUBTRACT_NS, || -> Result<(), ComputeError> {
                backend.subtract_resident(client, parent_slot, smaller_slot, larger_slot, slot_len)?;
                grow_drain(client);
                Ok(())
            })?;
            if smaller_scannable {
                bump_launch(); // One single-slot scan dispatch (device→device fold).
                time_phase(&GROW_SCAN_NS, || -> Result<(), ComputeError> {
                    backend.scan_resident_leaf_into_frontier(
                        client, smaller_slot, slot_len, &feats, &real_feats, cfg, s_g, s_h, s_n,
                        &frontier, smaller_leaf as usize,
                    )?;
                    grow_drain(client);
                    Ok(())
                })?;
            } else {
                reduce_winner_into_frontier(
                    backend, client, &frontier, smaller_leaf as usize, &SplitInfo::none(), -1,
                )?;
            }
            if larger_scannable {
                bump_launch(); // One single-slot scan dispatch (device→device fold).
                time_phase(&GROW_SCAN_NS, || -> Result<(), ComputeError> {
                    backend.scan_resident_leaf_into_frontier(
                        client, larger_slot, slot_len, &feats, &real_feats, cfg, l_g, l_h, l_n,
                        &frontier, larger_leaf as usize,
                    )?;
                    grow_drain(client);
                    Ok(())
                })?;
            } else {
                reduce_winner_into_frontier(
                    backend, client, &frontier, larger_leaf as usize, &SplitInfo::none(), -1,
                )?;
            }
        }
        // The next iteration's §8.3 self-invalidation targets (this split's children).
        prev_smaller = smaller_leaf;
        prev_larger = larger_leaf;
    }

    // ---- Reconstruct the host tree (to_host_tree) + the row→leaf layout. ----
    // TAIL: `to_host_tree` is a per-grow device READBACK and the layout rebuild is
    // O(num_data) host work — previously entirely invisible to phase_prof (a residual suspect).
    let _tail_guard = PhaseGuard::new(&GROW_TAIL_NS);
    let host_tree = tree.to_host_tree(client);
    let final_leaves = host_tree.num_leaves as usize;
    let mut indices = Vec::with_capacity(num_data);
    let mut leaf_begin = Vec::with_capacity(final_leaves);
    let mut leaf_count = Vec::with_capacity(final_leaves);
    // Each leaf's rows are the resident sub-range `perm[row_begin..+row_count]`
    // (no per-leaf `rows` Vec). Grouping them by leaf-index rebuilds the row→leaf layout.
    for leaf in leaves.iter().take(final_leaves) {
        leaf_begin.push(indices.len() as i32);
        leaf_count.push(leaf.row_count as i32);
        indices.extend_from_slice(&perm[leaf.row_begin..leaf.row_begin + leaf.row_count]);
    }
    let layout = LeafPartitionLayout {
        num_data: num_data as i32,
        indices,
        leaf_begin,
        leaf_count,
    };
    // Refuse to emit a resident-grown tree with any non-finite leaf value —
    // loud typed error over silent truncation.
    check_tree_leaves_finite(&host_tree)?;
    Ok((host_tree, layout))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::categorical_split::construct_bitset;
    use crate::kernels::data_partition::partition_categorical_stable;
    use crate::runtime::cpu_client;
    use crate::CpuBackend;

    /// Unit lane: [`check_root_seed_finite`] is the loud
    /// tripwire on the seeded root sums — NaN/±inf → typed [`ComputeError::NonFinite`]
    /// naming the sums; a healthy seed is `Ok(())`. Pinned here where the private helper is
    /// reachable (the integration NaN-injection grow lives in
    /// `oracle-harness/tests/on_device_tripwire_canary.rs`).
    #[test]
    fn check_root_seed_finite_fires_on_non_finite() {
        assert!(check_root_seed_finite(1.0, 2.0).is_ok(), "a finite seed must pass untouched");
        for (g, h) in [
            (f64::NAN, 2.0),
            (1.0, f64::NAN),
            (f64::INFINITY, 2.0),
            (1.0, f64::NEG_INFINITY),
        ] {
            let err = check_root_seed_finite(g, h).expect_err("non-finite seed must fail loudly");
            match err {
                ComputeError::NonFinite { detail } => {
                    assert!(detail.contains("root"), "detail must name the root seed: {detail}");
                }
                other => panic!("expected NonFinite, got {other:?}"),
            }
        }
    }

    /// Unit lane: [`check_tree_leaves_finite`] is the loud tripwire on the EMITTED
    /// tree's leaf values. A NaN/±inf leaf value
    /// → typed [`ComputeError::NonFinite`] naming the leaf index; an all-finite tree passes
    /// untouched (the check never mutates the tree).
    #[test]
    fn check_tree_leaves_finite_fires_on_non_finite_leaf() {
        // `as_constant` yields a valid 1-leaf tree; the check only reads `leaf_value`, so
        // widen it to a 3-leaf value vector for the assertion.
        let mut tree = lgbm_model::Tree::as_constant(0.1, 3);
        tree.num_leaves = 3;
        tree.leaf_value = vec![0.1, -0.2, 0.3];
        assert!(check_tree_leaves_finite(&tree).is_ok(), "an all-finite tree must pass");
        for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            tree.leaf_value[1] = bad;
            let err =
                check_tree_leaves_finite(&tree).expect_err("a non-finite leaf must fail loudly");
            match err {
                ComputeError::NonFinite { detail } => {
                    assert!(detail.contains("leaf 1"), "detail must name the leaf: {detail}");
                }
                other => panic!("expected NonFinite, got {other:?}"),
            }
        }
    }

    /// The sanctioned deferred-sync single drain
    /// [`crate::Backend::read_batched`] returns BYTE-IDENTICAL bytes to reading each handle
    /// separately — it is a pure call-ordering change (one `read_sync` over all handles vs
    /// N), never a numerics change. Pinned on the CpuBackend arm so the deferred-sync-batching
    /// semantics are verified WITHOUT HIP. Also asserts
    /// the documented cubecl-0.10 capability boundary the driver relies on: async batched copy
    /// available, true multi-stream overlap NOT.
    #[test]
    fn read_batched_single_drain_matches_per_handle_reads() {
        use crate::Backend;
        use cubecl::prelude::*;
        let client = cpu_client();

        // Two INDEPENDENT device handles (the shape a genuine multi-handle readback drains).
        let a: Vec<f64> = vec![1.0, -2.5, 3.75];
        let b: Vec<f32> = vec![10.0, 20.0, 30.0, 40.0];
        let h_a = client.create_from_slice(f64::as_bytes(&a));
        let h_b = client.create_from_slice(f32::as_bytes(&b));

        // Per-handle reads (the N-drain baseline).
        let ref_a = client.read_one_unchecked(h_a.clone());
        let ref_b = client.read_one_unchecked(h_b.clone());

        // The ONE deferred drain over both handles (order preserved).
        let batched = CpuBackend.read_batched(&client, vec![h_a, h_b]);
        assert_eq!(batched.len(), 2, "read_batched must return one buffer per handle, in order");
        assert_eq!(batched[0].as_slice(), &*ref_a, "batched drain byte-identical to per-handle read (handle A)");
        assert_eq!(batched[1].as_slice(), &*ref_b, "batched drain byte-identical to per-handle read (handle B)");
        assert_eq!(
            f64::from_bytes(&batched[0]),
            a.as_slice(),
            "decoded f64 payload survives the single drain unchanged"
        );
        assert_eq!(
            f32::from_bytes(&batched[1]),
            b.as_slice(),
            "decoded f32 payload survives the single drain unchanged"
        );

        // The documented cubecl-0.10 async-copy capability boundary, encoded in code.
        assert!(
            CpuBackend.supports_async_device_copy(),
            "async batched device→host copy is expressible in cubecl 0.10 (client.read(Vec<Handle>))"
        );
        assert!(
            !CpuBackend.supports_multi_stream_overlap(),
            "true per-op multi-stream overlap is NOT cleanly expressible in cubecl 0.10 \
             (set_stream pub unsafe / thread-derived StreamId / auto-merge) — documented M5 limit"
        );
    }

    /// A `BatchedSplitFeature` with the given `na_as_missing` flag (the only field the
    /// cross-feature argmax reads); every other field is a don't-care for the reduce.
    fn feat(na_as_missing: bool) -> BatchedSplitFeature {
        BatchedSplitFeature {
            slot_off: 0,
            num_bin: 8,
            offset: 1,
            default_bin: 8,
            most_freq_bin: 0,
            skip_default_bin: false,
            na_as_missing,
            run_forward: false,
        }
    }

    /// A `SplitInfo` carrying just the `gain` the argmax reduces on (other fields don't-care
    /// for the winner-SELECTION, and the whole struct is copied verbatim into the winner).
    fn split_with_gain(gain: f64) -> SplitInfo {
        SplitInfo {
            gain,
            threshold: 3,
            ..SplitInfo::none()
        }
    }

    /// The on-device cross-feature argmax reduce
    /// ([`argmax_over_resident_splits`], which the GpuBackend `scan_resident_leaf_argmax` delegates
    /// to) returns a winner BIT-IDENTICAL to the host [`argmax_over_splits`] over the SAME resident
    /// scan — including the strict-`>` gain rule, the lowest-real-feature-index tie-break (Pitfall 5),
    /// and the `na_as_missing` skip. This proves the payload-collapsed device argmax
    /// preserves the exact winner the host selected.
    #[test]
    fn on_device_argmax_reduce_matches_host_argmax_bit_for_bit() {
        // fpos order deliberately DIVERGES from real-feature-index order so the tie-break key
        // (real feature index, NOT fpos) is actually exercised.
        let feats = vec![feat(false), feat(false), feat(false), feat(false)];
        let real_feats = vec![7i32, 2, 5, 2]; // fpos 1 and fpos 3 share real index 2 (a tie candidate)

        // Case A: a clear single winner (fpos 2, gain 9.0).
        let splits_a = vec![
            split_with_gain(3.0),
            split_with_gain(5.0),
            split_with_gain(9.0),
            split_with_gain(1.0),
        ];
        // Case B: an exact gain TIE between fpos 1 (real 2) and fpos 3 (real 2) at the max gain,
        // and fpos 0 (real 7) also tied — the lowest real index (2) must win, and among equal real
        // indices the FIRST fpos (1) survives (strict `>` first-max).
        let splits_b = vec![
            split_with_gain(8.0), // real 7
            split_with_gain(8.0), // real 2  <- expected winner (lowest real, first fpos)
            split_with_gain(4.0), // real 5
            split_with_gain(8.0), // real 2
        ];
        // Case C: the NA feature carrying the highest gain must be SKIPPED by both folds.
        let feats_c = vec![feat(false), feat(true), feat(false)];
        let real_feats_c = vec![0i32, 1, 2];
        let splits_c = vec![
            split_with_gain(2.0),
            split_with_gain(99.0), // na_as_missing ⇒ skipped
            split_with_gain(3.0),
        ];

        // Build the `GrowFeature` list the host argmax reads (only `real_feature_index` matters).
        let grow = |reals: &[i32]| -> Vec<GrowFeature> {
            reals
                .iter()
                .map(|&r| GrowFeature {
                    bins: BinColumn::new(vec![0u32], 8),
                    num_bin: 8,
                    offset: 1,
                    min_bin: 0,
                    max_bin: 7,
                    default_bin: 8,
                    most_freq_bin: 0,
                    missing_type: lgbm_dataset::bin_mapper::MissingType::None,
                    bin_upper_bound: vec![0.0],
                    real_feature_index: r,
                    bin_type: lgbm_dataset::bin_mapper::BinType::Numerical,
                    bin_to_category: Vec::new(),
                    cat_smooth: 10.0,
                    cat_l2: 10.0,
                    max_cat_threshold: 32,
                    max_cat_to_onehot: 4,
                    min_data_per_group: 100,
                })
                .collect()
        };

        for (splits, feats, real_feats) in [
            (&splits_a, &feats, &real_feats),
            (&splits_b, &feats, &real_feats),
            (&splits_c, &feats_c, &real_feats_c),
        ] {
            let features = grow(real_feats);
            let host = argmax_over_splits(splits, feats, &features);
            let device = argmax_over_resident_splits(splits, feats, real_feats);
            assert_eq!(
                host.1, device.1,
                "winning feature-position must be bit-identical (host {:?} vs device {:?})",
                host, device
            );
            assert_eq!(
                host.0.gain.to_bits(),
                device.0.gain.to_bits(),
                "winning gain must be bit-identical"
            );
            assert_eq!(
                host.0.threshold, device.0.threshold,
                "winning threshold must be bit-identical"
            );
            assert_eq!(
                host.0.default_left, device.0.default_left,
                "winning default_left must be bit-identical"
            );
        }
        // Case B tie-break sanity: fpos 1 (lowest real index 2, first among equals) wins.
        assert_eq!(
            argmax_over_resident_splits(&splits_b, &feats, &real_feats).1,
            1,
            "tie must resolve to the lowest real feature index, first fpos among equals"
        );
    }

    /// The cross-leaf best-leaf reduce ([`best_leaf_argmax`], which the
    /// GpuBackend `best_leaf_reduce` delegates to) picks the SAME leaf the host best-first loop
    /// picked — strict-`>` gain, lowest-real-feature-index tie-break, seeded at leaf 0, a no-split
    /// leaf (`real == -1` ⇒ `i32::MAX`) never beating a real split.
    #[test]
    fn on_device_best_leaf_reduce_matches_host_loop() {
        // Reference host loop (the exact prior body).
        let host_pick = |bests: &[SplitInfo], reals: &[i32]| -> i32 {
            let mut best_leaf = 0i32;
            for i in 1..bests.len() {
                if split_gt(&bests[i], reals[i], &bests[best_leaf as usize], reals[best_leaf as usize]) {
                    best_leaf = i as i32;
                }
            }
            best_leaf
        };
        let cases: Vec<(Vec<SplitInfo>, Vec<i32>)> = vec![
            // Clear winner at leaf 2.
            (vec![split_with_gain(1.0), split_with_gain(2.0), split_with_gain(5.0)], vec![0, 1, 2]),
            // Gain tie leaf 0 vs leaf 2 — lower real index wins (leaf 2, real 1 < real 3).
            (vec![split_with_gain(5.0), split_with_gain(1.0), split_with_gain(5.0)], vec![3, 0, 1]),
            // A no-split leaf (gain -inf, real -1) must never win.
            (vec![SplitInfo::none(), split_with_gain(0.5), SplitInfo::none()], vec![-1, 4, -1]),
            // All no-split ⇒ leaf 0 (the seed) — caller's positive-gain guard then breaks.
            (vec![SplitInfo::none(), SplitInfo::none()], vec![-1, -1]),
        ];
        for (bests, reals) in &cases {
            assert_eq!(
                best_leaf_argmax(bests, reals),
                host_pick(bests, reals),
                "on-device best-leaf reduce must match the host loop pick"
            );
        }
    }

    /// The root grad/hess reduce (§6.1, [`root_grad_hess_fold`], which
    /// the GpuBackend `root_grad_hess_sum` delegates to) equals the ordered f64 fold the resident
    /// driver used previously — BIT-IDENTICAL (ascending row order is load-bearing).
    #[test]
    fn root_grad_hess_reduce_matches_ordered_fold() {
        let gradients: Vec<f32> = (0..257).map(|r| (r as f32 - 128.0) * 0.1).collect();
        let hessians: Vec<f32> = vec![1.0f32; 257];
        let mut sg = 0.0f64;
        let mut sh = 0.0f64;
        for (g, h) in gradients.iter().zip(hessians.iter()) {
            sg += f64::from(*g);
            sh += f64::from(*h);
        }
        let (rg, rh) = root_grad_hess_fold(&gradients, &hessians);
        assert_eq!(rg.to_bits(), sg.to_bits(), "root sum_gradient must be bit-identical to the ordered f64 fold");
        assert_eq!(rh.to_bits(), sh.to_bits(), "root sum_hessian must be bit-identical to the ordered f64 fold");
    }

    /// Double-bump / missed-bump guard. The driver applies
    /// the SINGLE `+2*kEpsilon` `sum_h` bump before the §8.1 evaluator (mirroring host
    /// `learner.rs:2760`); `best_split.rs` passes through and the evaluator does not
    /// bump. This pins the value the driver hands the evaluator to `raw + 2*kEpsilon`
    /// BIT-EXACT for BOTH committed fixtures' leaf hessian sums, independently of the
    /// end-to-end structure gate (which could pass while a last-ULP leaf value drifts).
    #[test]
    fn categorical_driver_bumps_sum_hessian_once() {
        let two_eps = 2.0 * f64::from(K_EPSILON);
        // The value the driver hands the §8.1 evaluator equals the host `learner.rs:2760`
        // bumped value `raw + 2*kEpsilon`, BIT-EXACT, for BOTH committed fixtures' leaf
        // hessian sums (cat_onehot = 40.0 = 0+10+10+10+10; cat_manyvsmany = 60.0 = 0+10*6).
        for (name, raw) in [("cat_onehot", 40.0_f64), ("cat_manyvsmany", 60.0_f64)] {
            assert_eq!(
                bump_sum_hessian_cat(raw),
                raw + two_eps,
                "{name}: driver-supplied sum_h must equal raw + 2*kEpsilon bit-exact (W-4)"
            );
        }
        // Double-bump / missed-bump guard. At the fixture magnitudes
        // (40/60) `2*kEpsilon` (2e-15) sits below the f64 ULP (~7e-15), so it is absorbed —
        // faithful to the host, but not a discriminating guard there. Exercise the guard at
        // a magnitude where `2*kEpsilon` IS representable: a single bump must differ from
        // raw (missed-bump) and from a two-bump chain (double-bump = `+4*kEpsilon`).
        let tiny = 1e-13_f64;
        let once = bump_sum_hessian_cat(tiny);
        let twice = bump_sum_hessian_cat(bump_sum_hessian_cat(tiny));
        assert_eq!(once, tiny + two_eps, "single bump == tiny + 2*kEpsilon");
        assert_ne!(once, tiny, "missed-bump guard: one bump must change a representable sum_h");
        assert_ne!(once, twice, "double-bump guard: one bump must differ from two");
    }

    /// The on-device resident §11 `AddPredictionToScore`
    /// partition scatter ([`add_prediction_to_score_on_device_resident`]) equals the host
    /// leaf-value scatter BIT-FOR-BIT on the cpu-f64 anchor. Builds a hand-rolled
    /// row→leaf [`LeafPartitionLayout`] (rows deliberately out of natural order inside
    /// each leaf, to prove the scatter is index-driven not position-driven) plus f64
    /// leaf values, scores it on device, and compares against the reference host
    /// scatter `out[row] += leaf_value[leaf]`.
    #[test]
    fn resident_add_prediction_to_score_matches_host_scatter_cpu_anchor() {
        let client = cpu_client();
        // 6 rows, 3 leaves. `indices` is leaf-grouped (leaf 0 = rows {4,1}, leaf 1 =
        // {0,5}, leaf 2 = {3,2}) — intentionally shuffled within each leaf.
        let layout = LeafPartitionLayout {
            num_data: 6,
            indices: vec![4, 1, 0, 5, 3, 2],
            leaf_begin: vec![0, 2, 4],
            leaf_count: vec![2, 2, 2],
        };
        // Distinct exact-representable f64 leaf values (no rounding — the scatter must
        // reproduce them bit-for-bit).
        let leaf_values = vec![-0.5f64, 0.25, 1.75];

        let device = add_prediction_to_score_on_device_resident(&client, &layout, &leaf_values)
            .expect("on-device resident score scatter");

        // Host reference: for every leaf, add its value to each of its rows.
        let mut host = vec![0.0f64; layout.num_data as usize];
        for (leaf, (&begin, &count)) in layout
            .leaf_begin
            .iter()
            .zip(layout.leaf_count.iter())
            .enumerate()
        {
            for &row in &layout.indices[begin as usize..(begin + count) as usize] {
                host[row as usize] += leaf_values[leaf];
            }
        }

        assert_eq!(device.len(), host.len(), "score length parity");
        for (i, (&d, &h)) in device.iter().zip(host.iter()).enumerate() {
            assert_eq!(
                d.to_bits(),
                h.to_bits(),
                "on-device resident score[{i}] = {d} != host scatter {h} (cpu-f64 anchor \
                 must be BIT-EXACT — integer-indexed f64 scatter, no reduction)"
            );
        }
    }

    /// Routing-convention isolation (Pitfall 4). The on-device
    /// categorical partition counts MUST equal the host `partition_categorical_stable`
    /// reference for both fixtures' winning bitsets, isolating the real-value-vs-inner-bin
    /// routing convention as a standalone signal BEFORE the full structure gate.
    /// The driver feeds the INNER-bin bitset (the `construct_bitset` over the winning
    /// bins) with `min_bin=0` and `most_freq_bin` supplied — both device and host derive
    /// `offset = (most_freq_bin==0)?1:0` INTERNALLY (== `offset_for_most_freq_bin`), so
    /// the same offset math is used on both sides (Pitfall 4).
    #[test]
    fn categorical_partition_counts_match_host_stable() {
        let client = cpu_client();
        // Fixture bin layout: 6 category values (bins 1..=6), 10 rows each; most_freq_bin=1
        // (=> offset 0), min_bin=0, max_bin=6, num_bin=7 (bin 0 = NaN dummy).
        let mut bins_vec: Vec<u32> = Vec::with_capacity(60);
        for cat_bin in 1..=6u32 {
            for _ in 0..10 {
                bins_vec.push(cat_bin);
            }
        }
        let bins = BinColumn::new(bins_vec, 7);
        let di: Vec<u32> = (0..60).collect();

        // cat_onehot winner = inner bin {4}; cat_manyvsmany winner = inner bins {4,5,6}.
        let cases = [
            ("cat_onehot", construct_bitset(&[4])),
            ("cat_manyvsmany", construct_bitset(&[4, 5, 6])),
        ];
        for (name, inner_bitset) in cases {
            let (dev_order, dev_split) = partition_categorical_on_device(
                &client, &bins, &di, 7, 0, 6, 1, &inner_bitset,
            )
            .unwrap_or_else(|e| panic!("{name}: device partition errored: {e:?}"));
            let (host_order, host_split) =
                partition_categorical_stable(&bins, &di, 7, 0, 6, 1, &inner_bitset)
                    .unwrap_or_else(|e| panic!("{name}: host stable partition errored: {e:?}"));
            // (left_count, right_count) must match the host stable reference exactly.
            assert_eq!(dev_split, host_split, "{name}: left count (split_point) device vs host");
            assert_eq!(
                dev_order.len() - dev_split,
                host_order.len() - host_split,
                "{name}: right count device vs host"
            );
            // Full stable order matches too (stronger than counts alone).
            assert_eq!(dev_order, host_order, "{name}: full stable partition order device vs host");
        }
    }

    /// Full-chain regression: a `most_freq_bin == 0 ⇒ offset 1`
    /// categorical feature driven through the WHOLE evaluator → `set_real_threshold`
    /// → `partition_categorical_on_device` chain, pinned to the CATEGORY-MEMBERSHIP
    /// golden (a row routes LEFT iff its raw bin is a winning category; one-hot with
    /// `most_freq_bin == 0` defaults non-members RIGHT). This is independent of the
    /// router's internal `bin - min_bin + offset` key arithmetic and therefore catches
    /// a bug class where an inner bitset that set bits at the raw winning bin (instead
    /// of the offset-adjusted bin) would miss every member when the router looks up
    /// `bin + 1` for offset 1, sending the whole winning category to the WRONG child.
    /// No committed fixture covered `offset == 1`, which is exactly how that bug hid.
    #[test]
    fn categorical_offset1_full_chain_routes_by_membership() {
        use crate::gain::GainConfig;
        use crate::kernels::categorical_split::{find_best_threshold_categorical, set_real_threshold};
        let client = cpu_client();

        // One-hot categorical config (num_bin <= max_cat_to_onehot).
        let mut cfg = GainConfig::default();
        cfg.min_data_in_leaf = 1;
        cfg.min_sum_hessian_in_leaf = 1e-3;
        cfg.lambda_l2 = 0.0;
        cfg.cat_l2 = 0.0;
        cfg.cat_smooth = 0.0;
        cfg.min_data_per_group = 1;
        cfg.max_cat_threshold = 32;
        cfg.max_cat_to_onehot = 4;

        // Feature: num_bin=4, most_freq_bin=0 (=> offset 1), min_bin=0, max_bin=3.
        // Compacted histogram (offset 1): slot 0 = dummy, then bins 1..=3. Bin with
        // raw index 3 carries the dominant gradient magnitude, so it is the unique
        // one-hot winner => cat_threshold == [3] (raw winning bin).
        let num_bin = 4i32;
        let offset = 1i32;
        let min_bin = 0i32;
        let hist: Vec<f64> = vec![
            0.0, 0.0, // slot 0 (dummy / mfb)
            5.0, 5.0, // raw bin 1
            -40.0, 5.0, // raw bin 3 winner (large |grad|) lands at compacted slot 2
            5.0, 5.0, // raw bin (unused-scan tail)
        ];
        let sum_g = 0.0 + 5.0 - 40.0 + 5.0;
        let sum_h = 15.0 + 2.0 * f64::from(K_EPSILON);
        let num_data = 15;
        let r = find_best_threshold_categorical(
            &client, &hist, &cfg, num_bin, offset, sum_g, sum_h, num_data,
        )
        .unwrap();
        assert!(r.is_splittable(), "offset==1 one-hot must find a split");
        let winning_bins: Vec<i32> = r.cat_threshold.iter().map(|&b| b as i32).collect();
        assert_eq!(winning_bins, vec![3], "unique one-hot winner is raw bin 3");

        // Build the inner routing bitset via set_real_threshold.
        let bin_to_category = [-1, 0, 1, 2]; // bins 0..=3 -> category values.
        let (_real, inner_bitset) =
            set_real_threshold(&winning_bins, &bin_to_category, min_bin, offset);

        // 5 rows in each of raw bins 1, 2, 3 (bin 0 = mfb / not present here).
        let mut bins_vec: Vec<u32> = Vec::with_capacity(15);
        for raw_bin in 1..=3u32 {
            for _ in 0..5 {
                bins_vec.push(raw_bin);
            }
        }
        let bins = BinColumn::new(bins_vec.clone(), num_bin as u32);
        let di: Vec<u32> = (0..15).collect();

        let (order, split_point) = partition_categorical_on_device(
            &client,
            &bins,
            &di,
            num_bin as u32,
            min_bin as u32,
            3, // max_bin
            0, // most_freq_bin
            &inner_bitset,
        )
        .unwrap();

        // MEMBERSHIP golden: left = exactly the rows whose raw bin is a winner (3).
        let winning_set: std::collections::HashSet<u32> =
            winning_bins.iter().map(|&b| b as u32).collect();
        let expected_left: std::collections::HashSet<u32> = di
            .iter()
            .copied()
            .filter(|&row| winning_set.contains(&bins_vec[row as usize]))
            .collect();
        let got_left: std::collections::HashSet<u32> = order[..split_point].iter().copied().collect();
        assert_eq!(
            got_left, expected_left,
            "offset==1 one-hot must route exactly the winning-category rows LEFT"
        );
        // Concretely: 5 rows (raw bin 3) LEFT, 10 rows RIGHT.
        assert_eq!(split_point, 5, "winning category (bin 3) has 5 rows -> left");
        assert_eq!(order.len() - split_point, 10, "non-winners route right");
    }

    /// The resident-index in-place partition
    /// ([`partition_resident_range`]) reorders a RESIDENT permutation sub-range into the SAME
    /// stable (left-then-right, original relative order) partition [`partition_leaf_stable`]
    /// produces over the same rows — the row permutation stays a single buffer, partitioned in
    /// place, with NO per-leaf `Vec<u32>` clone and NO local→global `parent_rows` map-back Vec.
    /// This is the cpu-runnable pin of the device-scatter semantics (the resident/rocm runtime
    /// path is compile-verified + real-hardware deferred, per the no-HIP discipline).
    #[test]
    fn resident_range_partition_matches_host_and_tiles_parent() {
        let client = cpu_client();
        // An 8-bin numeric feature; bin<=threshold routes LEFT, bin>threshold routes RIGHT.
        let col: Vec<u32> = vec![1, 5, 3, 7, 0, 4, 2, 6, 3, 5];
        let f = GrowFeature {
            bins: BinColumn::new(col.clone(), 8),
            num_bin: 8,
            offset: 0,
            min_bin: 0,
            max_bin: 7,
            default_bin: 8,
            most_freq_bin: 8, // > threshold ⇒ out-of-range defaults route right (none here)
            missing_type: MissingType::None,
            bin_upper_bound: vec![0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5],
            real_feature_index: 0,
            bin_type: BinType::Numerical,
            bin_to_category: Vec::new(),
            cat_smooth: 10.0,
            cat_l2: 10.0,
            max_cat_threshold: 32,
            max_cat_to_onehot: 4,
            min_data_per_group: 100,
        };
        let best = SplitInfo { threshold: 3, default_left: false, ..SplitInfo::none() };
        let partition_min_bin = f.min_bin + f.offset.max(0) as u32;

        // The resident permutation buffer for the whole grow (identity at root), with the
        // parent leaf occupying the sub-range [p_begin, p_begin+p_count).
        let p_begin = 0usize;
        let p_count = col.len();
        let mut perm: Vec<u32> = (0..col.len() as u32).collect();

        // Host reference: partition_leaf_stable over the same rows (the shipped anchor).
        let bins_sub = BinColumn::new(
            perm[p_begin..p_begin + p_count].iter().map(|&r| f.bins.bin(r as usize)).collect(),
            f.num_bin,
        );
        let (ref_reordered, ref_split) = partition_leaf_stable(
            &bins_sub, &perm[p_begin..p_begin + p_count], f.num_bin, partition_min_bin,
            f.max_bin, f.default_bin, f.most_freq_bin, 0, best.default_left, best.threshold,
        )
        .unwrap();

        // In-place resident partition of the parent sub-range. CpuBackend prefers the HOST
        // partition arm, so the DeviceLeafSplits struct is a required-but-unused arg here (the
        // on-device arm's resident child-range write is exercised on the rocm lane).
        let leaf_splits_dev =
            crate::kernels::partition::DeviceLeafSplits::<_>::new(&client, 2).unwrap();
        let split_point = partition_resident_range(
            &CpuBackend, &client, &mut perm[p_begin..p_begin + p_count], &f, &best,
            partition_min_bin, 0, &leaf_splits_dev, 0, p_begin as i32, p_count as i32,
        )
        .unwrap();

        // Split point + the in-place reordered globals match the host anchor bit-for-bit.
        assert_eq!(split_point, ref_split, "in-place split_point must match partition_leaf_stable");
        assert_eq!(
            &perm[p_begin..p_begin + p_count],
            ref_reordered.as_slice(),
            "in-place reordered global row ids must match the host stable partition"
        );

        // Child ranges TILE the parent range: contiguous offsets, counts sum to the parent count.
        let left_begin = p_begin;
        let left_count = split_point;
        let right_begin = p_begin + split_point;
        let right_count = p_count - split_point;
        assert_eq!(left_begin + left_count, right_begin, "child ranges must be contiguous");
        assert_eq!(left_count + right_count, p_count, "child counts must sum to the parent count");
        // Membership: left = rows with bin <= 3, right = rows with bin > 3 (original order each).
        let left: Vec<u32> = perm[left_begin..left_begin + left_count].to_vec();
        let right: Vec<u32> = perm[right_begin..right_begin + right_count].to_vec();
        assert!(left.iter().all(|&r| col[r as usize] <= 3), "left rows have bin <= threshold");
        assert!(right.iter().all(|&r| col[r as usize] > 3), "right rows have bin > threshold");
        assert_eq!(left, vec![0, 2, 4, 6, 8], "left rows in original relative order");
        assert_eq!(right, vec![1, 3, 5, 7, 9], "right rows in original relative order");

        // Edge case: a split that sends EVERY row one direction yields a zero-count child.
        let mut perm2: Vec<u32> = (0..col.len() as u32).collect();
        let best_all_left = SplitInfo { threshold: 7, default_left: false, ..SplitInfo::none() };
        let p2_count = perm2.len();
        let sp = partition_resident_range(
            &CpuBackend, &client, &mut perm2, &f, &best_all_left, partition_min_bin, 0,
            &leaf_splits_dev, 0, 0, p2_count as i32,
        )
        .unwrap();
        assert_eq!(sp, col.len(), "threshold 7 (max bin) routes all rows LEFT ⇒ empty right child");
        assert_eq!(col.len() - sp, 0, "the right child range has zero count");
    }
}
