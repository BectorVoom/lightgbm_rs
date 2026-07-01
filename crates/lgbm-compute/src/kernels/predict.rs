//! On-device prediction — tree-walk `AddPredictionToScore` (§10, ODL-15).
//!
//! The Rust port of the AMD-fork `AddPredictionToScoreKernel` tree-walk
//! (`LightGBM-release-4.6.0.99/src/io/cuda/cuda_tree.cu:317-396`): a per-row walk
//! from node 0 over the §13 columnar dataset (8/16/32 native-width dispatch via the
//! `<B: Int>` monomorph), remapping each split feature's raw bin
//! (`bin ∈ [min,max] ? bin−min+offset : most_freq_bin`), routing numerically
//! (missing/`default_left` → threshold) or categorically (bitset membership), and
//! adding `leaf_value[~node]` into a `double*` score accumulator on reaching a leaf.
//!
//! ## What lives here (18-04, ODL-15)
//! - [`add_prediction_to_score_kernel`] — the `#[cube(launch)]` tree-walk over one
//!   row per unit (`ABSOLUTE_POS`), `USE_INDICES` comptime (used-index subset vs
//!   identity). The **categorical membership branch reuses the shared
//!   [`crate::kernels::data_partition::find_in_bitset`] helper** (one transcription,
//!   Pitfall 4). The numeric branch transcribes the reference predict route
//!   (`cuda_tree.cu:376-391`) — a DISTINCT, simpler decision than the dense_bin
//!   `SplitInner` fan-out (`route_to_left`): it works on the ALREADY-remapped bin
//!   with a runtime `missing_type`/`default_left` read per node, so the comptime
//!   `route_to_left` flag fan-out is neither applicable nor callable inside a
//!   runtime multi-node walk (see the 18-04 SUMMARY deviation note).
//! - [`add_prediction_bagging_kernel`] — the §9 `AddPredictionToScoreKernel<USE_BAGGING>`
//!   per-row leaf-map gather-add (D-06): `score[data_index] += leaf_value[leaf_map[data_index]]`.
//! - [`add_prediction_to_score_on_device`] / [`add_prediction_bagging_on_device`] —
//!   the host drivers (input validation at the SP-4 boundary before the confined
//!   `unsafe` launch, T-18-06).
//!
//! Additive and OFF by default behind `LGBM_CUDA_ON_DEVICE` (D-13); the score
//! accumulator + scalar leaf-value math are the ONLY f64 — bin reads are
//! native-width integer (`u32::cast_from`), no f64 per-row hot loop (D-14/SP-5).
//! The objective inverse-link (`ConvertOutput`) stays HOST-side at the readback
//! boundary this phase (Phase-19 moves it on-device). Anchored to the cubecl-cpu
//! f64 fold, never GPU-vs-GPU (D-12 / def-f8u-01).

use cubecl::prelude::*;

use crate::error::ComputeError;
use crate::kernels::data_partition::find_in_bitset;

/// `kCategoricalMask` (`tree.h:20`) — decision-type bit 0.
const K_CATEGORICAL_MASK: i32 = 1;
/// `kDefaultLeftMask` (`tree.h:21`) — decision-type bit 1.
const K_DEFAULT_LEFT_MASK: i32 = 2;

// =========================================================================
// Tree-walk kernel (AddPredictionToScoreKernel<USE_INDICES>, cuda_tree.cu:317).
// =========================================================================

/// `AddPredictionToScoreKernel<USE_INDICES>` (`cuda_tree.cu:317-396`) — one unit
/// PER ROW (`ABSOLUTE_POS`). Walks from node 0 while `node >= 0`: reads the row's
/// native-width bin (`<B: Int>`, `u32::cast_from`) for the split feature's column,
/// remaps it (`bin ∈ [min,max] ? bin−min+offset : most_freq_bin`), then routes
/// categorically (bit-0 of `decision_type` set → the shared [`find_in_bitset`]
/// membership over `bitset_inner[cat_boundaries_inner[cat_idx] ..]`) or numerically
/// (`cuda_tree.cu:376-391`: `(missing_type==1 && bin==default_bin) ||
/// (missing_type==2 && bin==max_bin) ? default_left : bin <= threshold`). On a leaf
/// (`node < 0`) adds `leaf_value[~node]` into `score[data_index]` — the f64
/// accumulator is the ONLY f64 in the loop (D-14/SP-5).
///
/// `bins` is the row-major `[num_rows × num_features]` binned matrix at native
/// width `B`; `feat_*` are the per-(inner)feature meta arrays (indexed by
/// `split_feature_inner[node]`). Bounds-guarded (`i < num_rows`, T-18-06).
#[cube(launch)]
#[allow(clippy::too_many_arguments, unused_assignments)]
pub(crate) fn add_prediction_to_score_kernel<B: Int>(
    bins: &Array<B>,
    used_indices: &Array<u32>,
    feat_min: &Array<i32>,
    feat_max: &Array<i32>,
    feat_default: &Array<i32>,
    feat_mfb: &Array<i32>,
    feat_offset: &Array<i32>,
    feat_column: &Array<i32>,
    split_feature_inner: &Array<i32>,
    threshold_in_bin: &Array<u32>,
    decision_type: &Array<i32>,
    left_child: &Array<i32>,
    right_child: &Array<i32>,
    leaf_value: &Array<f64>,
    bitset_inner: &Array<u32>,
    cat_boundaries_inner: &Array<i32>,
    score: &mut Array<f64>,
    num_features: u32,
    num_rows: u32,
    #[comptime] use_indices: bool,
) {
    let i = ABSOLUTE_POS;
    if i < num_rows as usize {
        // USE_INDICES: walk a used-index subset; else identity (inner == data index).
        let data_index = if use_indices { used_indices[i] } else { i as u32 };

        let mut node = 0i32;
        while node >= 0 {
            let nd = node as usize;
            let sf = split_feature_inner[nd] as usize;
            let column = feat_column[sf];
            let min_bin = feat_min[sf];
            let max_bin = feat_max[sf];
            let default_bin = feat_default[sf];
            let most_freq_bin = feat_mfb[sf];
            let offset = feat_offset[sf];

            // Native-width bin read (u8/u16/u32 via B), row-major [row * F + column].
            let raw = u32::cast_from(bins[data_index as usize * num_features as usize + column as usize])
                as i32;
            // remap: bin ∈ [min,max] ? bin − min + offset : most_freq_bin.
            let in_range = (raw >= min_bin) && (raw <= max_bin);
            let bin = select(in_range, raw - min_bin + offset, most_freq_bin);

            let dt = decision_type[nd];
            let is_cat = (dt & K_CATEGORICAL_MASK) != 0;

            let mut go_left = false;
            if is_cat {
                // Categorical membership (cuda_tree.cu:367-374) via the SHARED
                // find_in_bitset helper (Pitfall 4). The cat's bitset occupies pool
                // words [start, end); index it with a GLOBAL position so the shared
                // 0-based helper reads the correct pool word (word = start + bin/32).
                let cat_idx = threshold_in_bin[nd] as i32;
                let start = cat_boundaries_inner[cat_idx as usize];
                let end = cat_boundaries_inner[(cat_idx + 1) as usize];
                let member = find_in_bitset(bitset_inner, end as u32, (start * 32 + bin) as u32);
                go_left = member == 1u32;
            } else {
                // Numeric route (cuda_tree.cu:376-391) — the DISTINCT, simpler
                // predict decision over the already-remapped bin (runtime flags).
                let threshold = threshold_in_bin[nd] as i32;
                let missing_type = (dt >> 2) & 3;
                let default_left = (dt & K_DEFAULT_LEFT_MASK) != 0;
                let is_missing = ((missing_type == 1) && (bin == default_bin))
                    || ((missing_type == 2) && (bin == max_bin));
                go_left = select(is_missing, default_left, bin <= threshold);
            }
            node = select(go_left, left_child[nd], right_child[nd]);
        }

        // Leaf reached (node < 0): score += leaf_value[~node]; ~node == -node - 1.
        let leaf = (-node - 1) as usize;
        score[data_index as usize] += leaf_value[leaf];
    }
}

/// `AddPredictionToScoreKernel<USE_BAGGING>` (§9, D-06) — the per-row leaf-map
/// gather-add. Each unit adds its row's leaf output into the f64 score via the
/// `data_index → leaf` map: `score[data_index] += leaf_value[leaf_map[data_index]]`.
/// `USE_BAGGING` (== `use_indices`) walks a used-index subset; else identity.
#[cube(launch)]
pub(crate) fn add_prediction_bagging_kernel(
    used_indices: &Array<u32>,
    data_index_to_leaf: &Array<i32>,
    leaf_value: &Array<f64>,
    score: &mut Array<f64>,
    num_rows: u32,
    #[comptime] use_indices: bool,
) {
    let i = ABSOLUTE_POS;
    if i < num_rows as usize {
        let data_index = if use_indices { used_indices[i] } else { i as u32 };
        let leaf = data_index_to_leaf[data_index as usize];
        score[data_index as usize] += leaf_value[leaf as usize];
    }
}

// =========================================================================
// Host drivers (SP-4 validation → confined unsafe launch).
// =========================================================================

/// The tree + per-feature meta arrays the [`add_prediction_to_score_on_device`]
/// walk consumes. Field names mirror 18-03's flat `DeviceCudaTree` layout
/// (`split_feature_inner`/`threshold_in_bin`/`decision_type`/`left_child`/
/// `right_child`/`leaf_value` + the `cat_boundaries_inner` bitset slab) to minimise
/// Phase-21 rework, though this phase builds the arrays standalone from the parity
/// fixture rather than consuming a live `DeviceCudaTree` (plan-check W2).
#[derive(Clone, Debug)]
pub struct PredictTree<'a> {
    /// Per-(inner)feature meta, each indexed by `split_feature_inner[node]`.
    pub feat_min: &'a [i32],
    pub feat_max: &'a [i32],
    pub feat_default: &'a [i32],
    pub feat_most_freq_bin: &'a [i32],
    pub feat_offset: &'a [i32],
    pub feat_column: &'a [i32],
    /// Per-node tree arrays.
    pub split_feature_inner: &'a [i32],
    pub threshold_in_bin: &'a [u32],
    pub decision_type: &'a [i32],
    pub left_child: &'a [i32],
    pub right_child: &'a [i32],
    /// Leaf outputs (raw pre-inverse-link margins this phase).
    pub leaf_value: &'a [f64],
    /// Categorical bitset pool + boundaries (`cat_boundaries_inner[cat_idx]`).
    pub bitset_inner: &'a [u32],
    pub cat_boundaries_inner: &'a [i32],
}

const WALK_BLOCK: u32 = 256;

/// Drive the [`add_prediction_to_score_kernel`] tree-walk on runtime `R`, returning
/// the `num_data`-length f64 raw-margin score accumulator (initialised to 0). `rows`
/// is the row-major `[num_data × num_features]` raw column store (indexed by the walk's
/// `data_index`); `num_rows` is the number of units to launch (`== num_data` for the
/// identity walk, `== used_indices.len()` for a subset); `bit_type ∈ {8,16,32}`
/// selects the native column width (D-05). `used_indices == None` walks all rows
/// (identity); otherwise the used-index subset. The objective inverse-link stays
/// HOST-side at readback (Phase-19 boundary) — this emits raw margin only.
///
/// # Errors
/// [`ComputeError::LengthMismatch`] on a `rows`/meta/tree length inconsistency,
/// [`ComputeError::Runtime`] on an unsupported `bit_type` or a `num_data` too small
/// for the walked data indices.
#[allow(clippy::too_many_arguments)]
pub fn add_prediction_to_score_on_device<R: cubecl::Runtime>(
    client: &cubecl::prelude::ComputeClient<R>,
    tree: &PredictTree<'_>,
    rows: &[u32],
    num_rows: usize,
    num_features: usize,
    bit_type: u32,
    num_data: usize,
    used_indices: Option<&[u32]>,
) -> Result<Vec<f64>, ComputeError> {
    validate_walk(tree, rows, num_rows, num_features, num_data, used_indices)?;
    if num_rows == 0 {
        return Ok(vec![0.0f64; num_data]);
    }

    // Per-feature meta + per-node tree buffers (uploaded once for the walk).
    let h_feat_min = client.create_from_slice(i32::as_bytes(tree.feat_min));
    let h_feat_max = client.create_from_slice(i32::as_bytes(tree.feat_max));
    let h_feat_default = client.create_from_slice(i32::as_bytes(tree.feat_default));
    let h_feat_mfb = client.create_from_slice(i32::as_bytes(tree.feat_most_freq_bin));
    let h_feat_offset = client.create_from_slice(i32::as_bytes(tree.feat_offset));
    let h_feat_column = client.create_from_slice(i32::as_bytes(tree.feat_column));
    let nf = tree.feat_min.len();

    let h_sfi = client.create_from_slice(i32::as_bytes(tree.split_feature_inner));
    let h_thr = client.create_from_slice(u32::as_bytes(tree.threshold_in_bin));
    let h_dt = client.create_from_slice(i32::as_bytes(tree.decision_type));
    let h_lc = client.create_from_slice(i32::as_bytes(tree.left_child));
    let h_rc = client.create_from_slice(i32::as_bytes(tree.right_child));
    let nn = tree.split_feature_inner.len();
    let h_leaf = client.create_from_slice(f64::as_bytes(tree.leaf_value));
    let nl = tree.leaf_value.len();

    // Categorical slabs: upload a 1-element dummy when empty (numeric models never
    // read them — is_cat is always false — so the dummy is inert; create_from_slice
    // of an empty slice is avoided).
    let bitset_owned: Vec<u32> = if tree.bitset_inner.is_empty() {
        vec![0u32]
    } else {
        tree.bitset_inner.to_vec()
    };
    let catb_owned: Vec<i32> = if tree.cat_boundaries_inner.is_empty() {
        vec![0i32, 0i32]
    } else {
        tree.cat_boundaries_inner.to_vec()
    };
    let h_bitset = client.create_from_slice(u32::as_bytes(&bitset_owned));
    let h_catb = client.create_from_slice(i32::as_bytes(&catb_owned));
    let n_bitset = bitset_owned.len();
    let n_catb = catb_owned.len();

    // The f64 score accumulator (D-15: sized to num_data once), initialised to 0.
    let init = vec![0.0f64; num_data];
    let h_score = client.create_from_slice(f64::as_bytes(&init));

    // used_indices: a real subset or a 1-element inert dummy for the identity path.
    let (use_indices, idx_owned): (bool, Vec<u32>) = match used_indices {
        Some(idx) => (true, idx.to_vec()),
        None => (false, vec![0u32]),
    };
    let h_idx = client.create_from_slice(u32::as_bytes(&idx_owned));
    let n_idx = idx_owned.len();

    let cube_count = (num_rows as u32).div_ceil(WALK_BLOCK);
    let n_bins = rows.len();

    macro_rules! launch_walk {
        ($w:ty) => {{
            let bins_w: Vec<$w> = rows.iter().map(|&b| b as $w).collect();
            let h_bins = client.create_from_slice(<$w>::as_bytes(&bins_w));
            // SAFETY: every handle is sized exactly as declared (`h_bins` = num_rows*
            // num_features cells, meta = nf, tree = nn, leaf = nl, score = num_data);
            // all outlive the launch; the kernel bounds-guards `i < num_rows` and the
            // walk only reads valid node/feature/leaf indices from the fixture tree
            // (validated) with find_in_bitset guarding the bitset word (T-18-03/06).
            unsafe {
                add_prediction_to_score_kernel::launch::<$w, R>(
                    client,
                    CubeCount::Static(cube_count, 1, 1),
                    CubeDim::new_1d(WALK_BLOCK),
                    ArrayArg::from_raw_parts(h_bins, n_bins),
                    ArrayArg::from_raw_parts(h_idx.clone(), n_idx),
                    ArrayArg::from_raw_parts(h_feat_min.clone(), nf),
                    ArrayArg::from_raw_parts(h_feat_max.clone(), nf),
                    ArrayArg::from_raw_parts(h_feat_default.clone(), nf),
                    ArrayArg::from_raw_parts(h_feat_mfb.clone(), nf),
                    ArrayArg::from_raw_parts(h_feat_offset.clone(), nf),
                    ArrayArg::from_raw_parts(h_feat_column.clone(), nf),
                    ArrayArg::from_raw_parts(h_sfi.clone(), nn),
                    ArrayArg::from_raw_parts(h_thr.clone(), nn),
                    ArrayArg::from_raw_parts(h_dt.clone(), nn),
                    ArrayArg::from_raw_parts(h_lc.clone(), nn),
                    ArrayArg::from_raw_parts(h_rc.clone(), nn),
                    ArrayArg::from_raw_parts(h_leaf.clone(), nl),
                    ArrayArg::from_raw_parts(h_bitset.clone(), n_bitset),
                    ArrayArg::from_raw_parts(h_catb.clone(), n_catb),
                    ArrayArg::from_raw_parts(h_score.clone(), num_data),
                    num_features as u32,
                    num_rows as u32,
                    use_indices,
                );
            }
        }};
    }
    match bit_type {
        8 => launch_walk!(u8),
        16 => launch_walk!(u16),
        32 => launch_walk!(u32),
        other => {
            return Err(ComputeError::Runtime {
                detail: format!("add_prediction_to_score: unsupported bit_type {other} (expected 8/16/32)"),
            })
        }
    }

    let bytes = client.read_one_unchecked(h_score);
    Ok(f64::from_bytes(&bytes).to_vec())
}

/// Drive the §9 [`add_prediction_bagging_kernel`] leaf-map gather-add (D-06) on
/// runtime `R`, returning the `num_data`-length f64 score. `data_index_to_leaf` maps
/// each data index to its resident leaf; `leaf_value` the per-leaf output.
/// `used_indices == None` adds every row (identity); else the used-index subset.
///
/// # Errors
/// [`ComputeError::LengthMismatch`] if `data_index_to_leaf.len() != num_data`;
/// [`ComputeError::Runtime`] on a leaf/data index out of range.
pub fn add_prediction_bagging_on_device<R: cubecl::Runtime>(
    client: &cubecl::prelude::ComputeClient<R>,
    data_index_to_leaf: &[i32],
    leaf_value: &[f64],
    num_data: usize,
    used_indices: Option<&[u32]>,
) -> Result<Vec<f64>, ComputeError> {
    if data_index_to_leaf.len() != num_data {
        return Err(ComputeError::LengthMismatch {
            expected: num_data,
            actual: data_index_to_leaf.len(),
        });
    }
    for (di, &leaf) in data_index_to_leaf.iter().enumerate() {
        if leaf < 0 || leaf as usize >= leaf_value.len() {
            return Err(ComputeError::Runtime {
                detail: format!(
                    "add_prediction_bagging: data index {di} maps to leaf {leaf} out of range \
                     (num_leaves = {})",
                    leaf_value.len()
                ),
            });
        }
    }
    let (use_indices, idx_owned, num_rows): (bool, Vec<u32>, usize) = match used_indices {
        Some(idx) => {
            for &di in idx {
                if di as usize >= num_data {
                    return Err(ComputeError::Runtime {
                        detail: format!("add_prediction_bagging: used index {di} >= num_data {num_data}"),
                    });
                }
            }
            (true, idx.to_vec(), idx.len())
        }
        None => (false, vec![0u32], num_data),
    };
    if num_rows == 0 {
        return Ok(vec![0.0f64; num_data]);
    }

    let h_idx = client.create_from_slice(u32::as_bytes(&idx_owned));
    let h_map = client.create_from_slice(i32::as_bytes(data_index_to_leaf));
    let h_leaf = client.create_from_slice(f64::as_bytes(leaf_value));
    let init = vec![0.0f64; num_data];
    let h_score = client.create_from_slice(f64::as_bytes(&init));
    let cube_count = (num_rows as u32).div_ceil(WALK_BLOCK);

    // SAFETY: `h_idx` sized `idx_owned.len()`, `h_map`/`h_score` `num_data`, `h_leaf`
    // `leaf_value.len()`; all outlive the launch; the kernel guards `i < num_rows`,
    // every leaf index checked < num_leaves and every used index < num_data.
    unsafe {
        add_prediction_bagging_kernel::launch::<R>(
            client,
            CubeCount::Static(cube_count, 1, 1),
            CubeDim::new_1d(WALK_BLOCK),
            ArrayArg::from_raw_parts(h_idx, idx_owned.len()),
            ArrayArg::from_raw_parts(h_map, num_data),
            ArrayArg::from_raw_parts(h_leaf, leaf_value.len()),
            ArrayArg::from_raw_parts(h_score.clone(), num_data),
            num_rows as u32,
            use_indices,
        );
    }
    let bytes = client.read_one_unchecked(h_score);
    Ok(f64::from_bytes(&bytes).to_vec())
}

/// Validate the tree-walk inputs at the host boundary (SP-4 / T-18-06) before the
/// confined `unsafe` launch. Rejects a `rows`/`num_features` mismatch, an unequal
/// per-feature meta length, an unequal per-node tree length, and a `num_data` too
/// small for the walked data indices.
fn validate_walk(
    tree: &PredictTree<'_>,
    rows: &[u32],
    num_rows: usize,
    num_features: usize,
    num_data: usize,
    used_indices: Option<&[u32]>,
) -> Result<(), ComputeError> {
    // `rows` is the full [num_data × num_features] store indexed by the walk's
    // data_index (NOT by the unit index) — so it spans num_data rows, and num_rows
    // units select into it (identity: num_rows == num_data; subset: used.len()).
    if rows.len() != num_data * num_features {
        return Err(ComputeError::LengthMismatch {
            expected: num_data * num_features,
            actual: rows.len(),
        });
    }
    if num_rows > num_data {
        return Err(ComputeError::Runtime {
            detail: format!("add_prediction_to_score: num_rows {num_rows} > num_data {num_data}"),
        });
    }
    let nf = tree.feat_min.len();
    for (name, len) in [
        ("feat_max", tree.feat_max.len()),
        ("feat_default", tree.feat_default.len()),
        ("feat_most_freq_bin", tree.feat_most_freq_bin.len()),
        ("feat_offset", tree.feat_offset.len()),
        ("feat_column", tree.feat_column.len()),
    ] {
        if len != nf {
            return Err(ComputeError::Runtime {
                detail: format!("add_prediction_to_score: {name} len {len} != feat_min len {nf}"),
            });
        }
    }
    if nf != num_features {
        return Err(ComputeError::Runtime {
            detail: format!("add_prediction_to_score: feature meta count {nf} != num_features {num_features}"),
        });
    }
    let nn = tree.split_feature_inner.len();
    for (name, len) in [
        ("threshold_in_bin", tree.threshold_in_bin.len()),
        ("decision_type", tree.decision_type.len()),
        ("left_child", tree.left_child.len()),
        ("right_child", tree.right_child.len()),
    ] {
        if len != nn {
            return Err(ComputeError::Runtime {
                detail: format!("add_prediction_to_score: {name} len {len} != split_feature_inner len {nn}"),
            });
        }
    }
    if nn == 0 {
        return Err(ComputeError::Runtime {
            detail: "add_prediction_to_score: tree must have at least one internal node".to_string(),
        });
    }
    // A used-index subset must land inside the num_data score accumulator; the
    // identity path (None) is already covered by the num_rows <= num_data check.
    if let Some(idx) = used_indices {
        for &di in idx {
            if di as usize >= num_data {
                return Err(ComputeError::Runtime {
                    detail: format!("add_prediction_to_score: used index {di} >= num_data {num_data}"),
                });
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::cpu_client;

    /// Owned model arrays: (feat min/max/def/mfb/off/col, sfi/thr/dt/lc/rc/leaf).
    type ModelArrays = (
        Vec<i32>, Vec<i32>, Vec<i32>, Vec<i32>, Vec<i32>, Vec<i32>,
        Vec<i32>, Vec<u32>, Vec<i32>, Vec<i32>, Vec<i32>, Vec<f64>,
    );

    /// The numeric golden model from `predict.txt` (PREDICT name=numeric): a depth-2
    /// tree exercising the 8-bit column, the missing/default (Zero + default_left)
    /// branch, out-of-range → most_freq_bin, and a plain threshold branch.
    fn numeric_tree() -> ModelArrays {
        // feat0 {min1,max6,default2,mfb5,offset0,col0}; feat1 {min1,max4,default0,mfb2,offset0,col1}.
        let feat_min = vec![1, 1];
        let feat_max = vec![6, 4];
        let feat_default = vec![2, 0];
        let feat_mfb = vec![5, 2];
        let feat_offset = vec![0, 0];
        let feat_column = vec![0, 1];
        // node0 {sf0,thr3,dt6(Zero+default_left),left1,right~2}; node1 {sf1,thr2,dt0,left~0,right~1}.
        let sfi = vec![0, 1];
        let thr = vec![3u32, 2];
        let dt = vec![6, 0];
        let lc = vec![1, -1]; // node0→node1 / node1→~0
        let rc = vec![-3, -2];
        let leaf = vec![-0.5f64, 0.25, 1.75];
        (feat_min, feat_max, feat_default, feat_mfb, feat_offset, feat_column, sfi, thr, dt, lc, rc, leaf)
    }

    /// Numeric tree-walk matches the predict.txt golden raw margins for the SAME bin
    /// values presented at 8-, 16-, and 32-bit column width (a bin is a width-
    /// invariant index) — the D-05 8/16/32 dispatch coverage.
    #[test]
    fn numeric_walk_matches_golden_all_widths() {
        let client = cpu_client();
        let (fmin, fmax, fdef, fmfb, foff, fcol, sfi, thr, dt, lc, rc, leaf) = numeric_tree();
        let tree = PredictTree {
            feat_min: &fmin,
            feat_max: &fmax,
            feat_default: &fdef,
            feat_most_freq_bin: &fmfb,
            feat_offset: &foff,
            feat_column: &fcol,
            split_feature_inner: &sfi,
            threshold_in_bin: &thr,
            decision_type: &dt,
            left_child: &lc,
            right_child: &rc,
            leaf_value: &leaf,
            bitset_inner: &[],
            cat_boundaries_inner: &[],
        };
        // rows row-major [5 × 2]; expected raw margins from predict.txt PSCORE.
        let rows: Vec<u32> = vec![2, 1, 5, 4, 1, 3, 9, 2, 3, 1];
        let expected = [-0.5f64, 1.75, -0.5, 1.75, -0.5];
        for bit_type in [8u32, 16, 32] {
            let got = add_prediction_to_score_on_device(&client, &tree, &rows, 5, 2, bit_type, 5, None)
                .expect("numeric walk");
            for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
                assert_eq!(
                    g.to_bits(),
                    e.to_bits(),
                    "numeric walk width {bit_type} row {i}: {g} != {e}"
                );
            }
        }
    }

    /// Categorical membership walk matches the predict.txt golden for the one-hot
    /// (8-bit, mfb0→offset1) and many-vs-many (16-bit, members {2,3,5}) models —
    /// the shared find_in_bitset reuse (Pitfall 4).
    #[test]
    fn categorical_walk_matches_golden() {
        let client = cpu_client();
        // cat_onehot: feat{min1,max5,default0,mfb0,offset1,col0}, node{sf0,cat_idx0,dt1,left~0,right~1}.
        let tree = PredictTree {
            feat_min: &[1],
            feat_max: &[5],
            feat_default: &[0],
            feat_most_freq_bin: &[0],
            feat_offset: &[1],
            feat_column: &[0],
            split_feature_inner: &[0],
            threshold_in_bin: &[0],
            decision_type: &[1],
            left_child: &[-1],
            right_child: &[-2],
            leaf_value: &[2.5f64, -1.0],
            bitset_inner: &[0x2],
            cat_boundaries_inner: &[0, 1],
        };
        let rows: Vec<u32> = vec![1, 2, 3, 4, 0];
        let expected = [2.5f64, -1.0, -1.0, -1.0, -1.0];
        let got = add_prediction_to_score_on_device(&client, &tree, &rows, 5, 1, 8, 5, None)
            .expect("cat_onehot walk");
        for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
            assert_eq!(g.to_bits(), e.to_bits(), "cat_onehot row {i}: {g} != {e}");
        }

        // cat_manyvsmany: feat{min2,max7,default0,mfb3,offset0,col0}, 16-bit column.
        let tree2 = PredictTree {
            feat_min: &[2],
            feat_max: &[7],
            feat_default: &[0],
            feat_most_freq_bin: &[3],
            feat_offset: &[0],
            feat_column: &[0],
            split_feature_inner: &[0],
            threshold_in_bin: &[0],
            decision_type: &[1],
            left_child: &[-1],
            right_child: &[-2],
            leaf_value: &[0.75f64, -0.5],
            bitset_inner: &[44],
            cat_boundaries_inner: &[0, 1],
        };
        let rows2: Vec<u32> = vec![2, 4, 5, 7, 6];
        let expected2 = [-0.5f64, 0.75, 0.75, 0.75, -0.5];
        let got2 = add_prediction_to_score_on_device(&client, &tree2, &rows2, 5, 1, 16, 5, None)
            .expect("cat_manyvsmany walk");
        for (i, (g, e)) in got2.iter().zip(expected2.iter()).enumerate() {
            assert_eq!(g.to_bits(), e.to_bits(), "cat_manyvsmany row {i}: {g} != {e}");
        }
    }

    /// USE_INDICES=true walks a used-index subset, writing scores only at the walked
    /// data indices (the rest stay 0 in the num_data-sized accumulator).
    fn numeric_predict_tree(m: &ModelArrays) -> PredictTree<'_> {
        PredictTree {
            feat_min: &m.0,
            feat_max: &m.1,
            feat_default: &m.2,
            feat_most_freq_bin: &m.3,
            feat_offset: &m.4,
            feat_column: &m.5,
            split_feature_inner: &m.6,
            threshold_in_bin: &m.7,
            decision_type: &m.8,
            left_child: &m.9,
            right_child: &m.10,
            leaf_value: &m.11,
            bitset_inner: &[],
            cat_boundaries_inner: &[],
        }
    }

    #[test]
    fn use_indices_subset_only_writes_walked_rows() {
        let client = cpu_client();
        let m = numeric_tree();
        let tree = numeric_predict_tree(&m);
        // Full 5-row column store; walk only data indices {1, 3, 4}. The score
        // accumulator is num_data=5; indices 0 and 2 must stay 0.
        // row1[5,4]→1.75, row3[9,2]→1.75, row4[3,1]→-0.5.
        let rows: Vec<u32> = vec![2, 1, 5, 4, 1, 3, 9, 2, 3, 1];
        let used = vec![1u32, 3, 4];
        let got = add_prediction_to_score_on_device(&client, &tree, &rows, 3, 2, 8, 5, Some(&used))
            .expect("subset walk");
        let mut expected = [0.0f64; 5];
        expected[1] = 1.75;
        expected[3] = 1.75;
        expected[4] = -0.5;
        for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
            assert_eq!(g.to_bits(), e.to_bits(), "subset row {i}: {g} != {e}");
        }
    }

    /// §9 USE_BAGGING leaf-map gather-add: `score[data_index] += leaf_value[leaf_map[data_index]]`.
    #[test]
    fn bagging_leaf_map_gather_add() {
        let client = cpu_client();
        let leaf_value = vec![-0.5f64, 0.25, 1.75];
        // 4 data rows resident in leaves [0, 2, 1, 0].
        let map = vec![0i32, 2, 1, 0];
        let got = add_prediction_bagging_on_device(&client, &map, &leaf_value, 4, None)
            .expect("bagging identity");
        let expected = [-0.5f64, 1.75, 0.25, -0.5];
        assert_eq!(got.len(), 4);
        for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
            assert_eq!(g.to_bits(), e.to_bits(), "bagging row {i}: {g} != {e}");
        }

        // USE_BAGGING subset: only add data indices {1, 3}.
        let used = vec![1u32, 3];
        let got2 = add_prediction_bagging_on_device(&client, &map, &leaf_value, 4, Some(&used))
            .expect("bagging subset");
        let expected2 = [0.0f64, 1.75, 0.0, -0.5];
        for (i, (g, e)) in got2.iter().zip(expected2.iter()).enumerate() {
            assert_eq!(g.to_bits(), e.to_bits(), "bagging subset row {i}: {g} != {e}");
        }
    }

    /// Host-boundary validation rejects a rows/num_features mismatch (SP-4).
    #[test]
    fn rejects_bad_rows_length() {
        let client = cpu_client();
        let m = numeric_tree();
        let tree = numeric_predict_tree(&m);
        let rows: Vec<u32> = vec![1, 2, 3]; // 3 != 2 * 2
        assert!(matches!(
            add_prediction_to_score_on_device(&client, &tree, &rows, 2, 2, 8, 2, None),
            Err(ComputeError::LengthMismatch { .. })
        ));
    }
}
