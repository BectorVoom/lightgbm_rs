//! Linear-tree per-leaf model fitting (C++ `LinearTreeLearner::CalculateLinear`,
//! `src/treelearner/linear_tree_learner.cpp`).
//!
//! After the base serial learner grows the tree STRUCTURE (constant leaves), each
//! leaf is given a linear model over the RAW feature values. The algorithm is the
//! C++ one, validated against real `lib_lightgbm` 4.6 linear goldens
//! (coefficients match to < 1e-6):
//!
//! - **Feature set per leaf** = the distinct NUMERICAL `split_feature` values on
//!   the root→leaf PATH, sorted ascending (`leaf_features`).
//! - **Fit** = hessian-weighted ridge least squares. With design row
//!   `x = [feat_j …, 1]` (constant LAST, C++ `curr_row[num_feat] = 1.0`), per-row
//!   gradient `g` and hessian `h`:
//!     `A = Σ h·xxᵀ + λ·diag(1,…,1,0)`   (the constant is NOT regularized),
//!     `b = −Σ g·x`,   solve `A θ = b`.
//!   `θ[..nf]` are `leaf_coeff` (parallel to the leaf's feature list), `θ[nf]` is
//!   `leaf_const`. `λ = linear_lambda`. Raw values are read as **f32** (C++
//!   `Dataset::raw_data_` is `std::vector<float>` per feature) and promoted to
//!   f64 inside the accumulation, exactly like the C++ inner loop
//!   (`linear_tree_learner.cpp:286-298`). Only the packed UPPER TRIANGLE of `A`
//!   is accumulated (C++ `XTHX_` layout), mirrored at the solve.
//! - Coefficients with `|c| <= kZeroThreshold` are dropped together with their
//!   feature (`linear_tree_learner.cpp:365-369`).
//! - Coefficients are stored UN-shrunk; the GBDT loop's [`Tree::shrinkage`] scales
//!   `leaf_const`/`leaf_coeff` by the learning rate afterwards (exactly as it
//!   already scales `leaf_value`). The constant `leaf_value` is left intact — it is
//!   the NaN-feature fallback used at predict time.
//! - The FIRST tree of the ensemble stays constant (num_features = 0); that is the
//!   caller's decision (do not call this for `is_first_tree`).
//!
//! **Determinism:** leaves are fitted in PARALLEL (crew), but each leaf's rows are
//! accumulated serially in ascending-row order — the accumulation order per leaf
//! is fixed, so the result is independent of thread count (and identical to the
//! C++ `num_threads=1` order). The C++ per-thread-block reduction is deliberately
//! NOT copied: its result depends on `num_threads`, which would break this
//! project's thread-count-deterministic model contract.

use lgbm_core::types::K_ZERO_THRESHOLD;
use lgbm_dataset::LeafPartitionLayout;
use lgbm_model::tree::{LinearModel, Tree};

use crate::data_partition::DataPartition;

/// Column-major **f32** raw feature store for linear-tree fitting — the analog of
/// C++ `Dataset::raw_data_` (`std::vector<std::vector<float>>`, one float column
/// per feature; values are `static_cast<float>` truncations of the input). The
/// per-feature NaN flags mirror C++ `LinearTreeLearner::contains_nan_` /
/// `any_nan_` (`InitLinear`), computed once here instead of per tree.
#[derive(Debug, Clone)]
pub struct RawFeatureColumns {
    /// Column-major values: feature `f`'s column is
    /// `values[f*num_data .. (f+1)*num_data]`.
    values: Vec<f32>,
    num_data: usize,
    num_features: usize,
    /// Per-feature "column contains a NaN" flag (C++ `contains_nan_`).
    contains_nan: Vec<bool>,
    /// Any feature contains a NaN (C++ `any_nan_`).
    any_nan: bool,
}

impl RawFeatureColumns {
    /// Build from `value_at(row, col)` (f64 input, truncated to f32 exactly as
    /// C++ `static_cast<float>`). The source is visited row-outer so a row-major
    /// source streams sequentially; the per-column NaN scan runs on the
    /// contiguous columns afterwards.
    pub fn from_fn(
        num_data: usize,
        num_features: usize,
        value_at: impl Fn(usize, usize) -> f64,
    ) -> Self {
        let mut values = vec![0.0f32; num_data * num_features];
        for r in 0..num_data {
            for c in 0..num_features {
                values[c * num_data + r] = value_at(r, c) as f32;
            }
        }
        let contains_nan: Vec<bool> = (0..num_features)
            .map(|c| values[c * num_data..(c + 1) * num_data].iter().any(|v| v.is_nan()))
            .collect();
        let any_nan = contains_nan.iter().any(|&b| b);
        Self {
            values,
            num_data,
            num_features,
            contains_nan,
            any_nan,
        }
    }

    /// Feature `f`'s contiguous column (`num_data` values).
    #[inline]
    #[must_use]
    pub fn column(&self, f: usize) -> &[f32] {
        &self.values[f * self.num_data..(f + 1) * self.num_data]
    }

    #[must_use]
    pub fn num_data(&self) -> usize {
        self.num_data
    }

    #[must_use]
    pub fn num_features(&self) -> usize {
        self.num_features
    }

    /// Whether feature `f`'s column contains a NaN (C++ `contains_nan_[f]`).
    #[must_use]
    pub fn contains_nan(&self, f: usize) -> bool {
        self.contains_nan[f]
    }

    /// Whether ANY column contains a NaN (C++ `any_nan_`).
    #[must_use]
    pub fn any_nan(&self) -> bool {
        self.any_nan
    }

    /// Gather one row as f64 (for the predict-path scorers, which take a full
    /// f64 feature row). f32→f64 is exact, matching C++'s float→double
    /// promotion at its predict sites.
    #[must_use]
    pub fn row_f64(&self, row: usize) -> Vec<f64> {
        (0..self.num_features)
            .map(|c| f64::from(self.values[c * self.num_data + row]))
            .collect()
    }
}

/// Remap a bagging **subset** `DataPartition` (whose `indices_in_leaf` are
/// SUBSET-row indices `0..in_bag.len()`) into a FULL-corpus partition whose leaves
/// hold the original row ids `in_bag[subset_row]` (the C++
/// `bag_mapper[index_mapper[i]]` map). This lets [`fit_linear_leaves`] and the
/// linear score update index the full-corpus `raw`/`grad`/`hess` buffers directly.
/// Only leaves `0..num_leaves` are copied (the grown tree's leaves).
pub fn remap_partition_to_full(
    subset: &DataPartition,
    in_bag: &[i32],
    num_leaves: i32,
) -> DataPartition {
    let nl = num_leaves.max(1) as usize;
    let mut indices: Vec<u32> = Vec::with_capacity(in_bag.len());
    let mut leaf_begin = vec![0i32; nl];
    let mut leaf_count = vec![0i32; nl];
    for leaf in 0..num_leaves {
        leaf_begin[leaf as usize] = indices.len() as i32;
        let rows = subset.indices_in_leaf(leaf);
        leaf_count[leaf as usize] = rows.len() as i32;
        for &sr in rows {
            indices.push(in_bag[sr as usize] as u32);
        }
    }
    DataPartition::from_payload(LeafPartitionLayout {
        num_data: in_bag.len() as i32,
        indices,
        leaf_begin,
        leaf_count,
    })
}

/// One leaf's normal-equation accumulators (C++ `XTHX_[leaf]` / `XTg_[leaf]`).
struct LeafFit {
    /// Packed upper triangle of `A = Σ h·xxᵀ`, in C++ `XTHX_` order: index `j`
    /// walks `(f1, f2)` pairs with `f2 >= f1`, `f1` outer.
    xthx: Vec<f64>,
    /// `b = −Σ g·x` (C++ accumulates `+Σ` and negates at the solve; negating
    /// each term instead is IEEE-exact-equivalent).
    xtg: Vec<f64>,
    /// C++ `total_nonzero`: non-NaN gate for the solve. Without NaNs this is the
    /// leaf's row count; with NaNs it counts non-NaN feature READS exactly like
    /// the C++ `HAS_NAN` variant (`num_nonzero[tid][leaf] += 1` per value).
    nonzero: i64,
}

/// Fit per-leaf linear models into a freshly-grown `tree`, mutating it in place
/// (sets `is_linear` + `linear`). See the module docs for the exact algorithm.
///
/// - `raw`: the [`RawFeatureColumns`] store (column-major f32), indexed by
///   ORIGINAL feature index (the same index space as `Tree::split_feature`).
/// - `grad` / `hess`: per-row gradient / hessian (length `raw.num_data()`).
/// - `linear_lambda`: L2 penalty on coefficients (never the constant).
/// - `partition`: the tree's data partition from GROWTH — leaf membership is
///   BIN-based (`indices_in_leaf`), matching C++. This is load-bearing: routing
///   rows by the real-value thresholds instead would disagree at bin boundaries
///   and drift the fit (and the subsequent boosting scores).
pub fn fit_linear_leaves(
    tree: &mut Tree,
    raw: &RawFeatureColumns,
    grad: &[f32],
    hess: &[f32],
    linear_lambda: f64,
    partition: &DataPartition,
) {
    let num_leaves = tree.num_leaves.max(0) as usize;

    let path_feats = leaf_path_features(tree);

    // C++ `Train` (`linear_tree_learner.cpp:113-121`): the NaN-checking variant
    // runs only when some SPLIT feature's column contains a NaN.
    let has_nan = raw.any_nan()
        && path_feats
            .iter()
            .flatten()
            .any(|&f| raw.contains_nan(f as usize));

    // Per-leaf accumulators, filled in PARALLEL (one crew task per leaf; each
    // leaf's rows accumulate serially in ascending order — thread-count
    // independent, see module docs).
    let mut fits: Vec<LeafFit> = path_feats
        .iter()
        .map(|f| {
            let dim = f.len() + 1;
            LeafFit {
                xthx: vec![0.0f64; dim * (dim + 1) / 2],
                xtg: vec![0.0f64; dim],
                nonzero: 0,
            }
        })
        .collect();

    lgbm_compute::crew::for_each_mut(&mut fits, |leaf, fit| {
        let feats = &path_feats[leaf];
        let nf = feats.len();
        let dim = nf + 1;
        let cols: Vec<&[f32]> = feats.iter().map(|&fi| raw.column(fi as usize)).collect();
        let rows = partition.indices_in_leaf(leaf as i32);

        // Design row x = [feat…, 1] (C++ `curr_row`, constant LAST). f32→f64 is
        // exact, so promoting at gather equals C++'s per-multiply promotion.
        let mut x = vec![0.0f64; dim];
        x[nf] = 1.0;
        let xthx = &mut fit.xthx[..];
        let xtg = &mut fit.xtg[..];

        if has_nan {
            // C++ `CalculateLinear<true>`: skip rows with a NaN in any used
            // feature; count non-NaN feature reads (the C++ gate quantity).
            for &row in rows {
                let r = row as usize;
                let mut nan_found = false;
                for (j, col) in cols.iter().enumerate() {
                    let v = col[r];
                    if v.is_nan() {
                        nan_found = true;
                        break;
                    }
                    fit.nonzero += 1;
                    x[j] = f64::from(v);
                }
                if nan_found {
                    continue;
                }
                accumulate_row(xthx, xtg, &x, dim, grad[r], hess[r]);
            }
        } else {
            for &row in rows {
                let r = row as usize;
                for (j, col) in cols.iter().enumerate() {
                    x[j] = f64::from(col[r]);
                }
                accumulate_row(xthx, xtg, &x, dim, grad[r], hess[r]);
            }
            fit.nonzero = rows.len() as i64;
        }
    });

    // Solve per leaf (tiny O(dim³) — serial).
    let mut leaf_const = vec![0.0f64; num_leaves];
    let mut leaf_coeff: Vec<Vec<f64>> = vec![Vec::new(); num_leaves];
    let mut leaf_features: Vec<Vec<i32>> = vec![Vec::new(); num_leaves];

    for l in 0..num_leaves {
        let feats = &path_feats[l];
        let nf = feats.len();
        let dim = nf + 1;
        let fit = &fits[l];

        // C++ gate (`linear_tree_learner.cpp:330-339`): too few usable rows —
        // keep the constant leaf output (leaf_features stays empty).
        if fit.nonzero < dim as i64 {
            leaf_const[l] = tree.leaf_value[l];
            continue;
        }
        if dim == 1 {
            // No path features (e.g. root leaf of a stump): constant model —
            // C++ still emits leaf_const, fitted as −Σg / Σh (the bias-only solve).
            leaf_const[l] = if fit.xthx[0].abs() > 0.0 {
                fit.xtg[0] / fit.xthx[0]
            } else {
                0.0
            };
            continue;
        }

        // Mirror the packed upper triangle into a full matrix and ridge the
        // feature diagonal (never the constant) — C++ lines 344-355.
        let mut a = vec![0.0f64; dim * dim];
        let mut j = 0;
        for f1 in 0..dim {
            for f2 in f1..dim {
                let v = fit.xthx[j];
                a[f1 * dim + f2] = v;
                a[f2 * dim + f1] = v;
                j += 1;
            }
        }
        for f1 in 0..nf {
            a[f1 * dim + f1] += linear_lambda;
        }

        match solve_symmetric(&mut a, &fit.xtg, dim) {
            Some(theta) => {
                // C++ drops |coeff| <= kZeroThreshold together with its feature
                // (`linear_tree_learner.cpp:365-369`).
                let mut coeffs = Vec::with_capacity(nf);
                let mut kept = Vec::with_capacity(nf);
                for (i, &fi) in feats.iter().enumerate() {
                    let c = theta[i];
                    if c < -K_ZERO_THRESHOLD || c > K_ZERO_THRESHOLD {
                        coeffs.push(c);
                        kept.push(fi);
                    }
                }
                leaf_const[l] = theta[nf];
                leaf_coeff[l] = coeffs;
                leaf_features[l] = kept;
            }
            None => {
                // Singular system: fall back to the constant leaf output (drop the
                // linear part for this leaf).
                leaf_const[l] = tree.leaf_value[l];
            }
        }
    }

    tree.is_linear = true;
    tree.linear = Some(LinearModel {
        leaf_const,
        leaf_features,
        leaf_coeff,
    });
}

/// Accumulate one row into the packed normal equations — the verbatim C++ inner
/// loop (`linear_tree_learner.cpp:286-298`): `XTg[f1] += x[f1]·g` (negated here,
/// see [`LeafFit::xtg`]), `XTHX[j] += (x[f1]·h)·x[f2]` over the upper triangle.
#[inline]
fn accumulate_row(xthx: &mut [f64], xtg: &mut [f64], x: &[f64], dim: usize, g: f32, h: f32) {
    let g = f64::from(g);
    let h = f64::from(h);
    let mut j = 0;
    for f1 in 0..dim {
        let f1v = x[f1];
        xtg[f1] -= f1v * g;
        let f1h = f1v * h;
        for f2 in f1..dim {
            xthx[j] += f1h * x[f2];
            j += 1;
        }
    }
}

/// Distinct `split_feature` values on each leaf's root→leaf path, sorted ascending
/// (C++ `leaf_features`). Index = leaf id. A single-leaf tree returns one empty
/// set.
fn leaf_path_features(tree: &Tree) -> Vec<Vec<i32>> {
    let num_leaves = tree.num_leaves.max(0) as usize;
    let mut out = vec![Vec::new(); num_leaves];
    if num_leaves <= 1 {
        return out;
    }
    let mut acc: Vec<i32> = Vec::new();
    dfs_path(tree, 0, &mut acc, &mut out);
    out
}

fn dfs_path(tree: &Tree, node: i32, acc: &mut Vec<i32>, out: &mut [Vec<i32>]) {
    if node < 0 {
        let leaf = (!node) as usize;
        let mut feats = acc.clone();
        feats.sort_unstable();
        feats.dedup();
        out[leaf] = feats;
        return;
    }
    let n = node as usize;
    // C++ `LinearTreeLearner` excludes categorically-split features from the leaf
    // linear model (a categorical bin id is not an ordered numeric quantity) — only
    // push NUMERICAL split features onto the path.
    let categorical = tree.is_categorical_split(n);
    if !categorical {
        acc.push(tree.split_feature[n]);
    }
    dfs_path(tree, tree.left_child[n], acc, out);
    dfs_path(tree, tree.right_child[n], acc, out);
    if !categorical {
        acc.pop();
    }
}

/// Solve the small symmetric linear system `A θ = b` (`A` is `dim×dim` row-major,
/// consumed) via Gaussian elimination with partial pivoting. Returns `None` if the
/// matrix is (near-)singular. `dim` is tiny (path features + 1), so an O(dim³)
/// dense solve is negligible.
fn solve_symmetric(a: &mut [f64], b: &[f64], dim: usize) -> Option<Vec<f64>> {
    let mut m = vec![0.0f64; dim * (dim + 1)]; // augmented [A | b]
    for i in 0..dim {
        for j in 0..dim {
            m[i * (dim + 1) + j] = a[i * dim + j];
        }
        m[i * (dim + 1) + dim] = b[i];
    }
    let w = dim + 1;
    for col in 0..dim {
        // Partial pivot: largest |value| in this column at or below the diagonal.
        let mut piv = col;
        let mut best = m[col * w + col].abs();
        for r in (col + 1)..dim {
            let v = m[r * w + col].abs();
            if v > best {
                best = v;
                piv = r;
            }
        }
        if best < 1e-30 {
            return None;
        }
        if piv != col {
            for k in 0..w {
                m.swap(col * w + k, piv * w + k);
            }
        }
        let diag = m[col * w + col];
        for r in 0..dim {
            if r == col {
                continue;
            }
            let factor = m[r * w + col] / diag;
            if factor == 0.0 {
                continue;
            }
            for k in col..w {
                m[r * w + k] -= factor * m[col * w + k];
            }
        }
    }
    let mut theta = vec![0.0f64; dim];
    for i in 0..dim {
        theta[i] = m[i * w + dim] / m[i * w + i];
    }
    Some(theta)
}
