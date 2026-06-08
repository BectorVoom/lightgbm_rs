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
/// If ANY non-spine feature/config is present → `false` → the byte-unchanged host
/// path (a mis-gated resident path that skipped categorical/monotone handling would be
/// a correctness bug — hence fail-safe).
pub fn resident_eligible(
    backend_supported: bool,
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
    true
}
