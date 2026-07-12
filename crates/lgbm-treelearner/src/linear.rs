//! Linear-tree per-leaf model fitting (C++ `LinearTreeLearner::CalculateLinear`,
//! `src/treelearner/linear_tree_learner.cpp`).
//!
//! After the base serial learner grows the tree STRUCTURE (constant leaves), each
//! leaf is given a linear model over the RAW feature values. The exact algorithm,
//! reverse-engineered from and validated against real `lib_lightgbm` 4.6 linear
//! goldens (coefficients match to < 1e-6):
//!
//! - **Feature set per leaf** = the distinct `split_feature` values on the
//!   root→leaf PATH, sorted ascending (`leaf_features`).
//! - **Fit** = hessian-weighted ridge least squares. With design row
//!   `x = [1, feat_j …]` (intercept first), per-row gradient `g` and hessian `h`:
//!     `A = Σ h·xxᵀ + λ·diag(0,1,1,…)`   (intercept is NOT regularized),
//!     `b = −Σ g·x`,   solve `A θ = b`.
//!   `θ[0]` is `leaf_const`, `θ[1..]` are `leaf_coeff` (parallel to the leaf's
//!   feature list). `λ = linear_lambda`.
//! - Coefficients are stored UN-shrunk; the GBDT loop's [`Tree::shrinkage`] scales
//!   `leaf_const`/`leaf_coeff` by the learning rate afterwards (exactly as it
//!   already scales `leaf_value`). The constant `leaf_value` is left intact — it is
//!   the NaN-feature fallback used at predict time.
//! - The FIRST tree of the ensemble stays constant (num_features = 0); that is the
//!   caller's decision (do not call this for `is_first_tree`).

use lgbm_dataset::LeafPartitionLayout;
use lgbm_model::tree::{LinearModel, Tree};

use crate::data_partition::DataPartition;

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

/// Fit per-leaf linear models into a freshly-grown `tree`, mutating it in place
/// (sets `is_linear` + `linear`). See the module docs for the exact algorithm.
///
/// - `raw`: row-major raw feature matrix, `num_rows * num_features` (`f64`),
///   indexed by ORIGINAL feature index (the same index space as
///   `Tree::split_feature`).
/// - `grad` / `hess`: per-row gradient / hessian (length `num_rows`).
/// - `linear_lambda`: L2 penalty on coefficients (never the intercept).
/// - `partition`: the tree's data partition from GROWTH — leaf membership is
///   BIN-based (`indices_in_leaf`), matching C++. This is load-bearing: routing
///   rows by the real-value thresholds instead would disagree at bin boundaries
///   and drift the fit (and the subsequent boosting scores).
pub fn fit_linear_leaves(
    tree: &mut Tree,
    raw: &[f64],
    num_features: usize,
    grad: &[f32],
    hess: &[f32],
    linear_lambda: f64,
    partition: &DataPartition,
) {
    let num_leaves = tree.num_leaves.max(0) as usize;

    let path_feats = leaf_path_features(tree);

    // Per-leaf normal-equation accumulators. dim = k+1 (k path features + bias).
    let dims: Vec<usize> = path_feats.iter().map(|f| f.len() + 1).collect();
    let mut a_mats: Vec<Vec<f64>> = dims.iter().map(|&d| vec![0.0f64; d * d]).collect();
    let mut b_vecs: Vec<Vec<f64>> = dims.iter().map(|&d| vec![0.0f64; d]).collect();

    // Scratch design row reused per data point.
    let max_dim = dims.iter().copied().max().unwrap_or(1);
    let mut x = vec![0.0f64; max_dim];

    // Accumulate each leaf's normal equations over its BIN-partitioned rows.
    for leaf in 0..num_leaves {
        let feats = &path_feats[leaf];
        let dim = dims[leaf];
        let a = &mut a_mats[leaf];
        let b = &mut b_vecs[leaf];
        for &row in partition.indices_in_leaf(leaf as i32) {
            let base = row as usize * num_features;
            // Build design row x = [1, feat…]; a NaN feature makes this row
            // unusable for the fit (C++ excludes it), so skip it.
            x[0] = 1.0;
            let mut usable = true;
            for (j, &fi) in feats.iter().enumerate() {
                let v = raw[base + fi as usize];
                if v.is_nan() {
                    usable = false;
                    break;
                }
                x[j + 1] = v;
            }
            if !usable {
                continue;
            }
            let g = grad[row as usize] as f64;
            let h = hess[row as usize] as f64;
            for i in 0..dim {
                let xi = x[i];
                b[i] -= g * xi;
                let hi = h * xi;
                let arow = i * dim;
                for k in 0..dim {
                    a[arow + k] += hi * x[k];
                }
            }
        }
    }

    let mut leaf_const = vec![0.0f64; num_leaves];
    let mut leaf_coeff: Vec<Vec<f64>> = vec![Vec::new(); num_leaves];
    let mut leaf_features: Vec<Vec<i32>> = path_feats;

    for l in 0..num_leaves {
        let dim = dims[l];
        if dim == 1 {
            // No path features (e.g. root leaf of a stump): constant model —
            // C++ still emits leaf_const, fitted as −Σg / Σh (the bias-only solve).
            let a = &a_mats[l];
            let b = &b_vecs[l];
            leaf_const[l] = if a[0].abs() > 0.0 { b[0] / a[0] } else { 0.0 };
            continue;
        }
        // Ridge on coefficients only (skip the intercept at index 0).
        let a = &mut a_mats[l];
        for i in 1..dim {
            a[i * dim + i] += linear_lambda;
        }
        match solve_symmetric(a, &b_vecs[l], dim) {
            Some(theta) => {
                leaf_const[l] = theta[0];
                leaf_coeff[l] = theta[1..].to_vec();
            }
            None => {
                // Singular system: fall back to the constant leaf output (drop the
                // linear part for this leaf).
                leaf_const[l] = tree.leaf_value[l];
                leaf_features[l] = Vec::new();
                leaf_coeff[l] = Vec::new();
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
