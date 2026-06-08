//! `resident_pool` — the device-resident histogram-pool MIRROR DISCIPLINE (260608-p90).
//!
//! The device-resident histogram pool keeps a pure-numeric-spine tree's per-leaf
//! histograms DEVICE-RESIDENT from build through fix/compact/subtract/scan on
//! `RocmBackend`, eliminating the dominant per-leaf host read-back + re-upload of the
//! histogram (the L3 perf win deferred by 260608-oib).
//!
//! ## Where the state lives
//! The actual device `Handle` slot mirror is a `RefCell<Vec<Option<Handle>>>` inside
//! `RocmBackend` (in `lgbm-compute`), NOT here — Handles are device objects and the
//! CMP-01 seam keeps every `cubecl` type confined to `lgbm-compute`. This module owns
//! only the LEARNER-side discipline:
//!
//! - The learner does NOT track a separate device slot map. It REUSES the host
//!   [`HistogramPool`](crate::histogram_pool::HistogramPool)'s slot bookkeeping (the
//!   `smaller_slot` / `larger_slot` / `parent_slot` it already computes via
//!   `pool.get` / `pool.move_`) and issues the PARALLEL device-handle op
//!   (`build_resident_leaf` / `subtract_resident` / `move_resident` /
//!   `scan_resident_leaf`) at the SAME call site, with the SAME slot ids. This keeps
//!   the device mirror's slot→Handle map tracking the host pool's slot→leaf map
//!   exactly (threat T-p90-02) — no reimplemented LRU / tree-growth.
//!
//! - [`reset_resident_pool`](lgbm_compute::Backend::reset_resident_pool) is issued
//!   alongside `pool.reset_map()` per tree; [`move_resident`](lgbm_compute::Backend::move_resident)
//!   alongside `pool.move_(left, right)`.
//!
//! ## The eligibility predicate
//! The single CONSERVATIVE / FAIL-SAFE gate is [`resident_eligible`]. The resident
//! fast path is taken ONLY when the workload is a pure numeric spine; ANY non-spine
//! feature or config falls back to today's byte-unchanged host path (non-negotiable
//! #4). When in doubt, [`resident_eligible`] returns `false`.

use lgbm_compute::GainConfig;
use lgbm_dataset::bin_mapper::BinType;

use crate::learner::{FeatureColumn, LearnerConstraints};

/// 260608-s2b Lever B — the device-resident SIZE GATE. Below this row count the
/// per-leaf resident chain's extra GPU launches dominate (the workload is
/// launch-bound, not compute-bound), so the byte-unchanged host path is the net
/// win; at or above it the resident path's saved host read-back/re-upload pays off.
///
/// PROVENANCE (measured AFTER Lever A, on the local gfx1100, `bench_train` both ways
/// via `LGBM_RESIDENT_FORCE`, train_median of 5, 2 runs each):
///
/// | rows   | FORCE_HOST | FORCE_RESIDENT | winner   |
/// |--------|------------|----------------|----------|
/// | 2000   | 1.50/1.43s | 1.64/1.66s     | HOST     |
/// | 8000   | 4.33/4.18s | 4.68/4.89s     | HOST     |
/// | 20000  | 11.95/12.11s | 11.50/11.73s | RESIDENT |
///
/// Even after Lever A cut the resident chain to 2 launches/leaf, the resident path
/// LOSES at 2000 and 8000 rows (launch/overhead-bound) and only WINS at 20000 rows
/// (compute-bound, where the saved per-leaf host read-back/re-upload pays off). The
/// measured crossover lies in the (8000, 20000] bracket; the threshold is set at the
/// middle of that bracket so small+medium route to the host winner and large routes to
/// the resident winner. `LGBM_RESIDENT_FORCE` overrides it for benching either path.
pub const RESIDENT_MIN_NUM_DATA: i32 = 12_000;

/// CONSERVATIVE / FAIL-SAFE device-resident eligibility predicate (260608-p90).
///
/// Returns `true` IFF the whole workload is a pure numeric spine that the resident
/// build→fix→compact→subtract→scan chain reproduces faithfully:
///
/// - `backend_supported` — the backend's
///   [`resident_pool_supported`](lgbm_compute::Backend::resident_pool_supported)
///   (ANDed in so a cpu build NEVER takes the resident branch), AND
/// - every feature is numeric (no [`BinType::Categorical`] — the resident chain has no
///   categorical many-vs-many handling), AND
/// - no monotone constraints, no interaction constraints, no extra_trees (each routes
///   an inline non-spine split branch the resident scan does not cover), AND
/// - `!capture_snapshots` (the snapshot per-bin re-scan reads the host buffer), AND
/// - the [`GainConfig`] default-path holds (`max_delta_step == 0.0 &&
///   path_smooth == 0.0`, matching the fused launcher's own reject).
///
/// CEGB is ALLOWED: its penalty is a post-split gain adjustment (no histogram read),
/// applied to the resident scan's SplitInfo exactly as on the host path.
///
/// 260608-s2b Lever B — SIZE GATE + runtime override. After the fail-safe
/// correctness checks, `num_data < `[`RESIDENT_MIN_NUM_DATA`] routes the host path
/// (the resident chain's extra per-leaf launches lose on launch-bound tiny
/// workloads). The size gate is a PERF KNOB, not a correctness gate — both paths are
/// proven equivalent (resident == host tree, p90 / s2b T0), so routing either way
/// grows the same tree within the ~1e-6 f32 contract.
///
/// `LGBM_RESIDENT_FORCE` (read once per call) OVERRIDES the size threshold for
/// benchmarking BOTH paths from a single binary without recompiling:
/// - `LGBM_RESIDENT_FORCE=0` → force the HOST path (return `false`) even above the
///   threshold, AFTER the fail-safe correctness checks still pass.
/// - `LGBM_RESIDENT_FORCE=1` → force the RESIDENT path (skip the size threshold) even
///   below it — still gated by every correctness check above (it can never enable
///   resident on a categorical/monotone/etc. workload; it only bypasses the SIZE
///   threshold, which is purely a perf decision between two correct paths).
/// - unset / any other value → the `num_data` threshold decides.
///
/// If ANY non-spine feature/config is present → `false` → the byte-unchanged host
/// path (a mis-gated resident path that skipped categorical/monotone handling would be
/// a correctness bug — hence fail-safe).
pub fn resident_eligible(
    backend_supported: bool,
    num_data: i32,
    features: &[FeatureColumn],
    constraints: &LearnerConstraints,
    capture_snapshots: bool,
    cfg: &GainConfig,
) -> bool {
    if !backend_supported {
        return false;
    }
    if capture_snapshots {
        return false;
    }
    // GainConfig must be on the default smoothing/clamp path (the fused launcher
    // rejects non-default max_delta_step / path_smooth anyway).
    if cfg.max_delta_step != 0.0 || cfg.path_smooth != 0.0 {
        return false;
    }
    // No monotone / interaction / extra_trees (each routes a non-spine inline branch).
    if !constraints.monotone_constraints.is_empty() {
        return false;
    }
    if !constraints.interaction_constraints.is_empty() {
        return false;
    }
    if constraints.extra_trees {
        return false;
    }
    // Forced splits route a separate growth path (the spine never sets it); be
    // conservative and fall back.
    if constraints.forced_splits.is_some() {
        return false;
    }
    // Every feature must be numeric (no categorical many-vs-many).
    if features.iter().any(|f| f.bin_type == BinType::Categorical) {
        return false;
    }

    // ---- 260608-s2b Lever B: runtime override THEN the num_data size gate ----
    // The workload is correctness-eligible here (pure numeric spine). What remains is
    // purely a PERF routing decision between two equivalent paths.
    match std::env::var("LGBM_RESIDENT_FORCE").ok().as_deref() {
        // Force HOST even though eligible (bench the host path / safety override).
        Some("0") => return false,
        // Force RESIDENT, bypassing only the SIZE threshold (still correctness-gated
        // by every check above). Used to bench resident-on at small sizes.
        Some("1") => return true,
        // unset / other → fall through to the size threshold.
        _ => {}
    }
    // Below the measured crossover the launch-bound host path wins → fall back.
    if num_data < RESIDENT_MIN_NUM_DATA {
        return false;
    }
    true
}

/// 260608-t3t — the FUSED directly-built-leaf SIZE GATE. The fused
/// build+fix+compact+scan kernel (`Backend::build_fix_scan_resident`) collapses a
/// directly-built leaf's construct(1)+fix(1)+scan(1) = 3 launches into 1, attacking
/// the launch-bound regime the s2b profile found (~205 launches/tree). It targets
/// the SMALL/MEDIUM band (`num_data < `[`RESIDENT_MIN_NUM_DATA`]) — exactly where the
/// 3-launch resident chain LOSES to the host path (s2b: host wins at 2k/8k) — to ask
/// whether collapsing to ONE launch flips that.
///
/// VERDICT: **OFF by default — the fused path is FLAT-to-NEGATIVE at every band.**
///
/// PROVENANCE (measured on the local gfx1100, `bench_train` all three ways via
/// `LGBM_RESIDENT_FORCE` / `LGBM_FUSED_FORCE`, train_median of 5, 2 runs each):
///
/// | rows   | FORCE_HOST    | FORCE_RESIDENT | FORCE_FUSED   | winner |
/// |--------|---------------|----------------|---------------|--------|
/// | 2000   | 1.42 / 1.49 s | 1.62 / 1.66 s  | 1.62 / 1.66 s | HOST   |
/// | 8000   | 4.56 / 4.25 s | 4.74 / 4.88 s  | 5.03 / 5.05 s | HOST   |
/// | 20000  | 11.82 / 11.97 s | 11.58 / 11.68 s | 12.13 / 12.19 s | RESIDENT |
///
/// The fused kernel is PROVEN bit-exact (the fused==host kernel oracle + the
/// fused==host TREE-equivalence test) and DOES cut launches (3 → 1 on the directly
/// built leaves ≈ root + smaller children ≈ ~half of a num_leaves=31 tree ≈ ~16
/// leaves/tree × 2 launches saved ≈ ~32 of ~205 ≈ ~15% fewer launches/tree). But the
/// NET wall-clock is FLAT-to-WORSE: at small it ties the resident chain (~1.64 s) and
/// loses to host (~1.45 s); at medium it is the SLOWEST of the three (~5.04 s vs host
/// ~4.40 / resident ~4.81); at large it is marginally slower than both. The single-
/// owner SEQUENTIAL f64 build the fused kernel uses (to stay bit-exact) replaces the
/// resident chain's PARALLEL f32-atomic build, and on gfx1100 that sequential per-
/// feature fold costs MORE than the ~2 launches/leaf it eliminates — the launch
/// saving does not pay for the lost build parallelism. NO net win materialized.
///
/// Per the honesty mandate (non-negotiable #5): the result is reported FLAT/NEGATIVE
/// and the fused kernel is GATED OFF by default. `FUSED_MAX_NUM_DATA = -1` makes
/// `num_data <= FUSED_MAX_NUM_DATA` false for every real workload (`num_data >= 1`),
/// so the fused path never auto-engages; the proven bit-exact kernel + oracle +
/// `LGBM_FUSED_FORCE` override stay landed for future use (e.g. a future device that
/// makes a sequential f64 fold cheap, or a fused-subtract follow-up — see the SUMMARY
/// scope note).
pub const FUSED_MAX_NUM_DATA: i32 = -1;

/// CONSERVATIVE / FAIL-SAFE FUSED directly-built-leaf eligibility predicate (260608-t3t).
///
/// The fused path is a PERF KNOB layered on the SAME correctness-gated numeric spine
/// as [`resident_eligible`] (every fail-safe check below is identical — categorical /
/// monotone / interaction / extra_trees / forced-splits / non-default
/// smoothing-clamp / capture all force the host path). When eligible, directly-built
/// leaves (root + smaller children) route through the single fused launch; the
/// subtract-derived larger children KEEP subtract+scan, and large keeps the
/// atomic-parallel resident chain. ANY non-spine feature/config ⇒ `false` ⇒ the
/// byte-unchanged host path.
///
/// `LGBM_FUSED_FORCE` (read once per call) OVERRIDES the size threshold for benching:
/// - `LGBM_FUSED_FORCE=0` → force OFF (host/resident path) even when eligible.
/// - `LGBM_FUSED_FORCE=1` → force the FUSED path, bypassing only the SIZE threshold
///   (still gated by every correctness check above).
/// - unset / other → the `num_data <= `[`FUSED_MAX_NUM_DATA`] threshold decides.
///
/// NOTE: the fused gate and the resident gate are MUTUALLY EXCLUSIVE per leaf — the
/// learner consults `fused_directly_built_eligible` FIRST; if it is true the directly
/// built leaf uses the fused launch, otherwise it falls back to the existing resident
/// / host routing (`resident_eligible`). The fused gate is correctness-equivalent to
/// the resident gate (same fail-safe spine; both grow the same tree within the ~1e-6
/// f32 contract), so this is purely a launch-count perf decision.
pub fn fused_directly_built_eligible(
    backend_supported: bool,
    num_data: i32,
    features: &[FeatureColumn],
    constraints: &LearnerConstraints,
    capture_snapshots: bool,
    cfg: &GainConfig,
) -> bool {
    if !backend_supported {
        return false;
    }
    if capture_snapshots {
        return false;
    }
    if cfg.max_delta_step != 0.0 || cfg.path_smooth != 0.0 {
        return false;
    }
    if !constraints.monotone_constraints.is_empty() {
        return false;
    }
    if !constraints.interaction_constraints.is_empty() {
        return false;
    }
    if constraints.extra_trees {
        return false;
    }
    if constraints.forced_splits.is_some() {
        return false;
    }
    if features.iter().any(|f| f.bin_type == BinType::Categorical) {
        return false;
    }

    // ---- runtime override THEN the num_data size gate (perf routing only) ----
    match std::env::var("LGBM_FUSED_FORCE").ok().as_deref() {
        Some("0") => return false,
        Some("1") => return true,
        _ => {}
    }
    // Task-3 VERDICT: the fused path is FLAT-to-NEGATIVE at every measured band, so
    // `FUSED_MAX_NUM_DATA = -1` keeps this false for every real workload (the kernel
    // stays OFF by default; only `LGBM_FUSED_FORCE=1` engages it). See the
    // `FUSED_MAX_NUM_DATA` provenance block for the measured table + the why.
    num_data <= FUSED_MAX_NUM_DATA
}
