//! `FixHistogram` — most-freq-bin reconstruction on RAW leaf sums (learner-side).
//!
//! Verbatim transcription of `Dataset::FixHistogram`
//! (`LightGBM/src/io/dataset.cpp:1488-1506`, commit 195c26fc, VERSION 4.6.0.99):
//!
//! ```cpp
//! const int most_freq_bin = bin_mapper->GetMostFreqBin();
//! if (most_freq_bin > 0) {
//!   GET_GRAD(data, most_freq_bin) = sum_gradient;        // leaf total
//!   GET_HESS(data, most_freq_bin) = sum_hessian;
//!   for (int i = 0; i < num_bin; ++i) {
//!     if (i != most_freq_bin) {
//!       GET_GRAD(data, most_freq_bin) -= GET_GRAD(data, i);
//!       GET_HESS(data, most_freq_bin) -= GET_HESS(data, i);
//!     }
//!   }
//! }
//! ```
//!
//! RESEARCH Open Q1 RESOLVED: this lives LEARNER-side (not behind the `Backend`
//! seam) as a plain f64 loop over the host-side histogram `Vec<f64>` returned by
//! `construct_histograms`. It runs immediately BEFORE every per-feature
//! `find_best_split` scan, for both the directly-built and the subtracted leaf.
//!
//! ## Pitfall 2 (load-bearing): use the RAW (un-bumped) sum_hessian
//! The `sum_hessian` passed here is the leaf's `leaf_splits->sum_hessians()` —
//! NOT the `+2·kEpsilon`-bumped value. That `2·kEpsilon` bump is internal to
//! `FindBestThreshold` (`feature_histogram.hpp:172`), applied by the
//! `find_best_split` host AFTER this reconstruct. Two different hessian values
//! flow through one feature's processing; this one is the raw leaf total
//! (`serial_tree_learner.cpp:531-534`). Passing the bumped value here would shift
//! the most-freq-bin cell by `2·kEpsilon` and corrupt the gain scan.

/// `FixHistogram` (`dataset.cpp:1488-1506`): reconstruct the `most_freq_bin`
/// histogram cell as `leaf_total − Σ(every other bin)`.
///
/// `hist` is the stride-2 `[g0,h0,g1,h1,…]` f64 histogram of length `2·num_bin`
/// (`GET_GRAD(i) = hist[i<<1]`, `GET_HESS(i) = hist[(i<<1)+1]`, `bin.h:45`). When
/// `most_freq_bin == 0` the C++ `if (most_freq_bin > 0)` guard is false and the
/// histogram is left untouched (bin 0 is never directly accumulated, so there is
/// nothing to reconstruct).
///
/// `sum_gradient` / `sum_hessian` are the leaf's RAW (un-bumped) totals — see the
/// module-level Pitfall-2 note.
///
/// The number of bins is inferred from `hist.len() / 2`; the caller guarantees
/// `hist.len()` is even and `most_freq_bin < num_bin` (the learner derives both
/// from the feature's `bin_mapper`). A `most_freq_bin` outside the histogram is a
/// no-op (the guarded write is skipped) rather than a panic.
pub fn fix_histogram(hist: &mut [f64], most_freq_bin: u32, sum_gradient: f64, sum_hessian: f64) {
    // C++ `if (most_freq_bin > 0)`: bin 0 is the implicit default and is never
    // directly folded, so there is nothing to reconstruct when it is the
    // most-freq bin.
    if most_freq_bin == 0 {
        return;
    }
    let num_bin = hist.len() / 2;
    let mfb = most_freq_bin as usize;
    // Defensive bound (V5 spirit): a most_freq_bin past the histogram cannot be
    // reconstructed; leave the histogram untouched rather than index OOB.
    if mfb >= num_bin {
        return;
    }

    let g_idx = mfb << 1;
    let h_idx = g_idx + 1;
    // Seed the most-freq cell with the RAW leaf totals (dataset.cpp:1490-1491).
    let mut g = sum_gradient;
    let mut h = sum_hessian;
    // Subtract every OTHER bin's cell, in ascending bin order (the C++ loop order
    // is load-bearing for the f64 fold; never reorder / parallelize).
    for i in 0..num_bin {
        if i != mfb {
            g -= hist[i << 1];
            h -= hist[(i << 1) + 1];
        }
    }
    hist[g_idx] = g;
    hist[h_idx] = h;
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Hand-computed 3-bin example using the RAW (un-bumped) sum_hessian:
    /// bins 0,1,2 with most_freq_bin = 1. The most-freq cell must become
    /// leaf_total − Σ(bin0, bin2).
    #[test]
    fn fix_histogram_reconstructs_most_freq_bin_from_raw_sums() {
        // stride-2 [g,h]: bin0=(1.0,2.0), bin1=(GARBAGE, GARBAGE), bin2=(4.0,5.0)
        let mut hist = vec![1.0, 2.0, /*bin1*/ 99.0, 88.0, 4.0, 5.0];
        // RAW leaf totals (NOT +2*kEpsilon).
        let sum_g = 10.0;
        let sum_h = 20.0;
        fix_histogram(&mut hist, 1, sum_g, sum_h);
        // bin1 grad = 10 - (1 + 4) = 5; bin1 hess = 20 - (2 + 5) = 13.
        assert_eq!(hist[2], 5.0, "most-freq grad = leaf_total - other bins");
        assert_eq!(hist[3], 13.0, "most-freq hess = leaf_total - other bins");
        // The other bins are untouched.
        assert_eq!(hist[0], 1.0);
        assert_eq!(hist[1], 2.0);
        assert_eq!(hist[4], 4.0);
        assert_eq!(hist[5], 5.0);
    }

    #[test]
    fn fix_histogram_most_freq_bin_zero_is_noop() {
        let mut hist = vec![7.0, 8.0, 1.0, 2.0];
        let before = hist.clone();
        fix_histogram(&mut hist, 0, 100.0, 200.0);
        assert_eq!(hist, before, "most_freq_bin == 0 leaves the histogram untouched");
    }

    #[test]
    fn fix_histogram_out_of_range_most_freq_bin_is_noop() {
        let mut hist = vec![1.0, 1.0, 2.0, 2.0];
        let before = hist.clone();
        // num_bin = 2; most_freq_bin = 5 is out of range -> no write, no panic.
        fix_histogram(&mut hist, 5, 10.0, 10.0);
        assert_eq!(hist, before);
    }

    /// The raw vs bumped distinction (Pitfall 2): the reconstruct must use exactly
    /// the value passed, so passing a `+2*kEpsilon`-bumped hessian would visibly
    /// differ. This guards against a caller accidentally bumping first.
    #[test]
    fn fix_histogram_uses_exactly_the_passed_hessian() {
        let mut a = vec![1.0, 1.0, 0.0, 0.0, 2.0, 2.0];
        let mut b = a.clone();
        let raw_h = 9.0_f64;
        let bumped_h = raw_h + 2.0 * f64::from(lgbm_core::types::K_EPSILON);
        fix_histogram(&mut a, 1, 5.0, raw_h);
        fix_histogram(&mut b, 1, 5.0, bumped_h);
        // bin1 hess(a) = 9 - (1 + 2) = 6; bin1 hess(b) = bumped - 3 != 6.
        assert_eq!(a[3], 6.0);
        assert_ne!(a[3], b[3], "raw vs bumped hessian produce different cells");
    }
}
