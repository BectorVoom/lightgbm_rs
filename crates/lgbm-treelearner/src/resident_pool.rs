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
