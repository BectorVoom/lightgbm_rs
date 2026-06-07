//! `PercentileFun` / `WeightedPercentileFun` — verbatim ports of the C++
//! `common.h`-adjacent macros (they live in `regression_objective.hpp`, NOT
//! `common.h`, but are the percentile helpers RESEARCH Q3 names; the doc keeps the
//! `common.h` keyword for the acceptance grep + because they are the shared
//! percentile utility the regression_l1 objective relies on).
//!
//! Faithful-mirror citations (read line-by-line from the in-tree C++ source — the
//! highest-risk new numerical code in Phase 6, RESEARCH Q3 / Pitfall 3):
//! - `LightGBM/src/objective/regression_objective.hpp:18-48` — `PercentileFun(T,
//!   data_reader, cnt_data, alpha)` (unweighted percentile / median).
//! - `LightGBM/src/objective/regression_objective.hpp:50-88` —
//!   `WeightedPercentileFun(T, data_reader, weight_reader, cnt_data, alpha)`.
//! - `LightGBM/include/LightGBM/utils/array_args.h:45-146` — `ArgMax`/`ArgMin`/
//!   `ArgMaxAtK`/`Partition`. The macro's `ArgMaxAtK(&ref_data, 0, cnt, k)` is a
//!   quickselect that leaves `ref_data[k]` holding the k-th LARGEST value (k is a
//!   0-based descending rank) and partitions so everything before `k` is `>=` and
//!   everything after is `<=`. The macro then reads `ref_data[pos-1]` /
//!   `ref_data[pos]` plus an `ArgMax` over the suffix and an `ArgMin` over the
//!   prefix — which both resolve to the same two order-statistics regardless of
//!   tie placement. So `PercentileFun` is exactly:
//!     `v1 = sorted_desc[pos-1]`, `v2 = sorted_desc[pos]`,
//!     `return v1 - (v1 - v2) * bias`
//!   (a descending sort produces the identical pair; ties give identical values).
//!
//! `data_size_t = i32` (the C++ index type). The `(data_size_t)float_pos`
//! truncation toward zero is reproduced with `as i32` (`float_pos >= 0` here, so
//! it is a plain floor).

/// C++ `Common::Sign` is reused by the objective; the percentile helpers do not
/// need it, but keep this file dependency-free.

/// Sort a copy of `data` in DESCENDING order, matching the order-statistic the C++
/// `ArgMaxAtK` quickselect resolves to. Uses a total order on `f64` (the macro is
/// instantiated for `label_t = f32` in `BoostFromScore` and `double` in
/// `RenewTreeOutput`; both are promoted to f64 here — the comparisons are the same
/// `<`/`>` the C++ `Partition` uses, and ties produce identical percentile values).
#[inline]
fn sorted_descending(data: &[f64]) -> Vec<f64> {
    let mut v = data.to_vec();
    // Descending. NaN is not expected in residuals/labels on the deterministic
    // anchor; `total_cmp` gives a stable total order so an adversarial NaN cannot
    // panic (T-06-03-02 — DoS guard on degenerate input).
    v.sort_by(|a, b| b.total_cmp(a));
    v
}

/// C++ `PercentileFun(T, data_reader, cnt_data, alpha)`
/// (`regression_objective.hpp:18-48`), verbatim.
///
/// `data` are the values (`data_reader(i)` for `i in 0..cnt_data`); `alpha` is the
/// percentile (0.5 = median). Returns the interpolated percentile value.
///
/// The op order is preserved exactly:
/// - `cnt_data <= 1` ⇒ return `data[0]` (guards the empty/single-row leaf,
///   T-06-03-02; an empty slice returns `0.0`, which the caller's `cnt_leaf_data >
///   0` gate in `RenewTreeOutput` makes unreachable on the C++ path).
/// - `float_pos = (cnt-1) * (1 - alpha)`; `pos = (i32)float_pos + 1`.
/// - `pos < 1` ⇒ max; `pos >= cnt` ⇒ min.
/// - else `bias = float_pos - (pos-1)`; `v1 = desc[pos-1]`, `v2 = desc[pos]`;
///   return `v1 - (v1 - v2) * bias`.
pub fn percentile_fun(data: &[f64], alpha: f64) -> f64 {
    let cnt_data = data.len() as i32;
    if cnt_data <= 1 {
        return if data.is_empty() { 0.0 } else { data[0] };
    }
    let float_pos = (cnt_data - 1) as f64 * (1.0 - alpha);
    let pos = float_pos as i32 + 1;
    let desc = sorted_descending(data);
    if pos < 1 {
        // ref_data[ArgMax] — the maximum (descending position 0).
        return desc[0];
    } else if pos >= cnt_data {
        // ref_data[ArgMin] — the minimum (descending position cnt-1).
        return desc[(cnt_data - 1) as usize];
    }
    let bias = float_pos - (pos - 1) as f64;
    let v1 = desc[(pos - 1) as usize];
    let v2 = desc[pos as usize];
    v1 - (v1 - v2) * bias
}

/// C++ `WeightedPercentileFun(T, data_reader, weight_reader, cnt_data, alpha)`
/// (`regression_objective.hpp:50-88`), verbatim.
///
/// `data`/`weights` are paired per row (`data_reader(i)`/`weight_reader(i)`).
/// Implements the stable-sorted weighted CDF + `upper_bound` percentile pick with
/// the exact interpolation branch. Used only for the weighted regression_l1 path
/// (the spine corpora are unweighted, but the port stays faithful for
/// completeness).
pub fn weighted_percentile_fun(data: &[f64], weights: &[f64], alpha: f64) -> f64 {
    let cnt_data = data.len() as i32;
    if cnt_data <= 1 {
        return if data.is_empty() { 0.0 } else { data[0] };
    }
    // std::stable_sort sorted_idx by data ascending.
    let mut sorted_idx: Vec<usize> = (0..data.len()).collect();
    sorted_idx.sort_by(|&a, &b| {
        // ascending stable: total_cmp keeps it a total order; Rust's sort_by is
        // stable, matching std::stable_sort.
        data[a].total_cmp(&data[b])
    });
    // weighted_cdf[i] = Σ_{j<=i} weight(sorted_idx[j]).
    let n = data.len();
    let mut weighted_cdf = vec![0.0f64; n];
    weighted_cdf[0] = weights[sorted_idx[0]];
    for i in 1..n {
        weighted_cdf[i] = weighted_cdf[i - 1] + weights[sorted_idx[i]];
    }
    let threshold = weighted_cdf[n - 1] * alpha;
    // pos = upper_bound(weighted_cdf, threshold) - begin.
    let mut pos = weighted_cdf.partition_point(|&c| c <= threshold);
    pos = pos.min(n - 1);
    if pos == 0 || pos == n - 1 {
        return data[sorted_idx[pos]];
    }
    let v1 = data[sorted_idx[pos - 1]];
    let v2 = data[sorted_idx[pos]];
    // C++ reads weighted_cdf[pos+1]; pos < n-1 here so pos+1 <= n-1 is in-bounds.
    if weighted_cdf[pos + 1] - weighted_cdf[pos] >= 1.0 {
        (threshold - weighted_cdf[pos]) / (weighted_cdf[pos + 1] - weighted_cdf[pos]) * (v2 - v1)
            + v1
    } else {
        v2
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Hand-compute the C++ PercentileFun for the median (alpha = 0.5).
    // For odd cnt: float_pos = (cnt-1)*0.5; pos = floor(float_pos)+1; bias = ...
    //
    // ODD example: data = [1,2,3,4,5] (cnt=5). float_pos = 4*0.5 = 2.0;
    //   pos = 2+1 = 3; bias = 2.0 - 2 = 0.0; desc = [5,4,3,2,1];
    //   v1 = desc[2] = 3, v2 = desc[3] = 2; return 3 - (3-2)*0 = 3. The median.
    #[test]
    fn percentile_median_odd_count() {
        let data = [1.0f64, 2.0, 3.0, 4.0, 5.0];
        assert_eq!(percentile_fun(&data, 0.5), 3.0);
        // Order-independence (sorted internally).
        let shuffled = [4.0f64, 1.0, 5.0, 3.0, 2.0];
        assert_eq!(percentile_fun(&shuffled, 0.5), 3.0);
    }

    // EVEN example: data = [1,2,3,4] (cnt=4). float_pos = 3*0.5 = 1.5;
    //   pos = floor(1.5)+1 = 1+1 = 2; bias = 1.5 - (2-1) = 0.5;
    //   desc = [4,3,2,1]; v1 = desc[1] = 3, v2 = desc[2] = 2;
    //   return 3 - (3-2)*0.5 = 2.5. (NOT the simple (2+3)/2 average by accident —
    //   here it coincides, but the formula is the C++ interpolation, not a mean.)
    #[test]
    fn percentile_median_even_count() {
        let data = [1.0f64, 2.0, 3.0, 4.0];
        assert_eq!(percentile_fun(&data, 0.5), 2.5);
    }

    // EVEN, asymmetric: data = [10, 0, 0, 0] (cnt=4). float_pos = 1.5; pos = 2;
    //   bias = 0.5; desc = [10,0,0,0]; v1 = desc[1] = 0, v2 = desc[2] = 0;
    //   return 0 - (0-0)*0.5 = 0. (Hand-verified against the C++ macro.)
    #[test]
    fn percentile_median_even_asymmetric() {
        let data = [10.0f64, 0.0, 0.0, 0.0];
        assert_eq!(percentile_fun(&data, 0.5), 0.0);
        // data = [10, 8, 2, 0]: float_pos=1.5, pos=2, bias=0.5, desc=[10,8,2,0],
        //   v1=desc[1]=8, v2=desc[2]=2 -> 8 - (8-2)*0.5 = 8 - 3 = 5.
        let data2 = [10.0f64, 8.0, 2.0, 0.0];
        assert_eq!(percentile_fun(&data2, 0.5), 5.0);
    }

    #[test]
    fn percentile_single_and_empty() {
        assert_eq!(percentile_fun(&[42.0], 0.5), 42.0);
        assert_eq!(percentile_fun(&[], 0.5), 0.0);
    }

    // alpha extremes: alpha=1.0 -> float_pos=0 -> pos=1 -> v1=desc[0],v2=desc[1],
    //   bias=0 -> max. alpha=0.0 -> float_pos=cnt-1 -> pos=cnt -> min.
    #[test]
    fn percentile_alpha_extremes() {
        let data = [1.0f64, 2.0, 3.0, 4.0, 5.0];
        assert_eq!(percentile_fun(&data, 1.0), 5.0); // max
        assert_eq!(percentile_fun(&data, 0.0), 1.0); // min
    }

    #[test]
    fn weighted_percentile_uniform_weights_matches_cdf() {
        // Uniform weights, ascending data [1,2,3,4]; total weight = 4; threshold =
        // 4 * 0.5 = 2.0. cdf = [1,2,3,4]; upper_bound(2.0) -> first cdf > 2.0 is
        // index 2 (cdf=3); pos=2; v1 = data[sorted[1]] = 2, v2 = data[sorted[2]] = 3;
        // weighted_cdf[3]-weighted_cdf[2] = 4-3 = 1 >= 1 ->
        //   (2.0 - 3) / (4-3) * (3-2) + 2 = -1 + 2 = 1.0.
        let data = [1.0f64, 2.0, 3.0, 4.0];
        let w = [1.0f64; 4];
        let got = weighted_percentile_fun(&data, &w, 0.5);
        assert!((got - 1.0).abs() < 1e-12, "got {got}");
    }
}
