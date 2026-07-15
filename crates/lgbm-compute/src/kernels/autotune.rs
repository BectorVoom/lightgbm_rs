//! Shared CubeCL-autotune plumbing for the GPU launch-config knobs.
//!
//! This is the one place the two launch-knob wirings depend on:
//!   - histogram-BUILD row-partition `P` (`histogram.rs`) and
//!   - split-SCAN `CubeDim` `W` (`split.rs`)
//! both reuse the [`LaunchKey`] AutotuneKey, the [`size_band`] log2 bucketer, the
//! [`autotune_enabled`] off-switch, and the [`cache_namespace_id`] device cache id.
//!
//! Two correctness rules apply here:
//!   - serde is MANDATORY in the crate that defines the key — the `AutotuneKey` trait
//!     alias requires `serde::{Serialize, DeserializeOwned}` under `std_io` (always on
//!     linux), so the persistent disk cache (`target/autotune/<ver>/<device>/*.json.log`)
//!     can round-trip the chosen variant across processes.
//!   - The AutotuneKey keys on the OCCUPANCY REGIME (`log2(rows)` size-band), NOT the
//!     exact row count — exact-`rows` keying would cause a per-leaf tuning STORM.
//!
//! Gated `#[cfg(feature = "rocm")]` at the declaration site (see `kernels/mod.rs`); the
//! default cpu build pulls none of this codegen (the f64 anchor is never autotuned).

use std::fmt::Display;

/// The shared [`cubecl::tune::AutotuneKey`] both launch knobs reuse.
///
/// One shape, two consumers:
///   - the histogram-BUILD tunable sets `bucket = size_band(rows)` so the chosen
///     row-partition `P` tracks the occupancy regime, and
///   - the split-SCAN tunable sets `bucket = 0` (the feature-per-lane scan is
///     bit-exact across `CubeDim` `W` and the winning `W` does not depend on row count).
///
/// `serde::{Serialize, Deserialize}` are REQUIRED: the persistent disk cache (`std_io`,
/// always active on linux) serializes the key. `Hash + Eq` key the in-memory tuner map.
#[derive(Clone, Debug, Hash, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct LaunchKey {
    /// Occupancy regime bucket. Build: `size_band(rows)`. Scan: `0`.
    pub bucket: u32,
    /// Number of features in the launch (the cube X extent).
    pub feats: u32,
    /// Histogram bins per feature (the per-feature slot width driver).
    pub bins: u32,
}

impl cubecl::tune::AutotuneKey for LaunchKey {}

impl Display for LaunchKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "LaunchKey(bucket={},feats={},bins={})",
            self.bucket, self.feats, self.bins
        )
    }
}

/// The `log2(rows)` occupancy size-band.
///
/// Returns `floor(log2(rows))`, i.e. the power-of-two decade `rows` falls in, so every
/// node within the same decade shares a key and tunes ONCE per regime instead of once
/// per distinct row count (the per-leaf tuning storm). The variant choice (e.g. P=1 for
/// small leaves vs P=16 for large) tracks the occupancy regime, not the exact count.
///
/// Never panics: `rows == 0` maps to band `0` (and so does `rows == 1`).
#[must_use]
pub fn size_band(rows: usize) -> u32 {
    if rows <= 1 {
        0
    } else {
        // floor(log2(rows)) = highest set bit index. `usize::BITS - 1 - leading_zeros`.
        usize::BITS - 1 - (rows as usize).leading_zeros()
    }
}

/// The documented autotune off-switch / fallback bound.
///
/// Default-ON: returns `true` unless `LGBM_AUTOTUNE=0` is set, in which case the rocm
/// launch paths fall back to the cold-start heuristic (`row_partition_count` /
/// `LGBM_SCAN_CUBEDIM`). Read FRESH on every call (NOT `OnceLock`-cached) so the
/// parity tests and the off-switch can toggle within a single process. Mirrors the
/// existing `LGBM_SCAN_CUBEDIM` / `LGBM_ROWPART_*` escape hatches.
///
/// Only the exact value `"0"` disables; any other value (including empty/unset/garbage)
/// keeps autotune on — `LGBM_AUTOTUNE` is a bool toggle, not a path or format string, so
/// there is no injection surface (T-13-01: accept).
#[must_use]
pub fn autotune_enabled() -> bool {
    !matches!(std::env::var("LGBM_AUTOTUNE").as_deref(), Ok("0"))
}

/// The persistent-cache namespace id — the `id` first arg of `LocalTuner::execute`.
///
/// Names the on-disk cache directory (`target/autotune/<ver>/<device>/*.json.log`) so the
/// winner persists across processes. Tied to the device ordinal bound by
/// [`crate::runtime::rocm_client`]'s `AmdDevice::new(0)`; returns the constant `"rocm:0"`
/// to match that single local device (there is no clean public ordinal accessor in
/// `runtime.rs`, and the client is hardcoded to ordinal 0).
#[must_use]
pub fn cache_namespace_id() -> String {
    // Matches `AmdDevice::new(0)` in `runtime::rocm_client`. If the runtime ever binds a
    // non-zero ordinal, derive this from that ordinal instead of the constant.
    "rocm:0".to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn size_band_never_panics_on_zero() {
        assert_eq!(size_band(0), 0);
        assert_eq!(size_band(1), 0);
    }

    #[test]
    fn size_band_is_monotonic_non_decreasing() {
        let mut prev = size_band(1);
        for rows in (2..100_000).step_by(37) {
            let b = size_band(rows);
            assert!(b >= prev, "band regressed at rows={rows}: {b} < {prev}");
            prev = b;
        }
    }

    #[test]
    fn size_band_buckets_same_decade_together() {
        // [1024, 2047] all share floor(log2) == 10; the decade boundary bumps the band.
        assert_eq!(size_band(1024), size_band(1500));
        assert_eq!(size_band(1024), size_band(2047));
        assert!(size_band(2048) > size_band(2047));
        // equal rows map to equal bands (a must_have truth).
        assert_eq!(size_band(50_000), size_band(50_000));
    }

    #[test]
    fn autotune_enabled_flips_with_env() {
        // NB: process-global env mutation — keep this serialized in one test.
        // SAFETY: single-threaded unit test, no other thread reads the env concurrently.
        unsafe { std::env::set_var("LGBM_AUTOTUNE", "0") };
        assert!(!autotune_enabled(), "LGBM_AUTOTUNE=0 must disable");
        unsafe { std::env::set_var("LGBM_AUTOTUNE", "1") };
        assert!(autotune_enabled(), "LGBM_AUTOTUNE=1 must enable");
        unsafe { std::env::remove_var("LGBM_AUTOTUNE") };
        assert!(autotune_enabled(), "unset must default-on");
    }

    #[test]
    fn launch_key_display_and_namespace() {
        // Pins the SHIPPED Display format (the persistent autotune cache logs carry
        // it) — a silent format change would orphan on-disk cache entries.
        let k = LaunchKey { bucket: 10, feats: 50, bins: 256 };
        assert_eq!(format!("{k}"), "LaunchKey(bucket=10,feats=50,bins=256)");
        assert_eq!(cache_namespace_id(), "rocm:0");
    }
}
