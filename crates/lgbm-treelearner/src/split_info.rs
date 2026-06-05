//! Split-result value struct + the C++ `SplitInfo::operator>` tie-break.
//!
//! The canonical split-result struct already exists as
//! [`lgbm_compute::gain::SplitInfo`] (the C++ `split_info.hpp:22-54` field
//! mirror, with the `gain == kMinScore` no-split sentinel). We REUSE it here via
//! a re-export rather than defining a second, divergent struct — there must be
//! exactly one `SplitInfo` in the workspace.
//!
//! This module only ADDS the comparison the learner's cross-feature argmax needs:
//! the C++ `SplitInfo::operator>` (`split_info.hpp:138-165`) tie-break, which
//! `gain.rs` does not carry.

pub use lgbm_compute::gain::SplitInfo;

/// C++ `SplitInfo::operator>` (`split_info.hpp:138-165`): order two candidate
/// splits by gain, breaking ties toward the SMALLER feature index.
///
/// Returns `true` iff split `a` (with feature `a_feat`) strictly beats split `b`
/// (with feature `b_feat`):
/// - if the gains differ, the higher gain wins (`a.gain > b.gain`);
/// - on an EQUAL gain, the smaller feature index wins, with feature `-1`
///   (the C++ "no feature" sentinel) mapped to `i32::MAX` so it loses every tie
///   to a real feature.
///
/// This matches the C++ `ArgMax`-style scan that keeps the FIRST maximum on a
/// full tie (gain AND feature equal → `split_gt` returns `false`, so the incumbent
/// is retained).
pub fn split_gt(a: &SplitInfo, a_feat: i32, b: &SplitInfo, b_feat: i32) -> bool {
    if a.gain != b.gain {
        return a.gain > b.gain;
    }
    let af = if a_feat == -1 { i32::MAX } else { a_feat };
    let bf = if b_feat == -1 { i32::MAX } else { b_feat };
    af < bf
}

#[cfg(test)]
mod tests {
    use super::*;

    fn si(gain: f64) -> SplitInfo {
        let mut s = SplitInfo::none();
        s.gain = gain;
        s
    }

    #[test]
    fn higher_gain_wins() {
        let a = si(5.0);
        let b = si(3.0);
        assert!(split_gt(&a, 0, &b, 1), "5.0 > 3.0 regardless of feature");
        assert!(!split_gt(&b, 1, &a, 0), "3.0 does not beat 5.0");
    }

    #[test]
    fn equal_gain_smaller_feature_wins() {
        let a = si(4.0);
        let b = si(4.0);
        // feature 2 vs feature 7 on equal gain: smaller (2) wins.
        assert!(split_gt(&a, 2, &b, 7));
        assert!(!split_gt(&b, 7, &a, 2));
    }

    #[test]
    fn equal_gain_and_feature_is_not_greater() {
        // Full tie -> not strictly greater (incumbent kept by the argmax scan).
        let a = si(4.0);
        let b = si(4.0);
        assert!(!split_gt(&a, 3, &b, 3));
    }

    #[test]
    fn feature_minus_one_loses_ties_to_any_real_feature() {
        let a = si(4.0);
        let b = si(4.0);
        // -1 maps to i32::MAX, so any real feature index beats it on equal gain.
        assert!(split_gt(&b, 0, &a, -1), "feature 0 beats sentinel -1");
        assert!(!split_gt(&a, -1, &b, 0), "sentinel -1 loses to feature 0");
        // -1 vs i32::MAX-real: -1 still loses (it maps to the same MAX, tie -> false).
        assert!(!split_gt(&a, -1, &b, i32::MAX));
    }
}
