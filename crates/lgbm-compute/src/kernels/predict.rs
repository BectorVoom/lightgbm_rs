//! On-device prediction — tree-walk `AddPredictionToScore` (§10).
//!
//! The Rust port of the AMD-fork `AddPredictionToScoreKernel` tree-walk
//! (`LightGBM-release-4.6.0.99/src/io/cuda/cuda_tree.cu:317-396`): a per-row walk
//! from node 0 over the §13 columnar dataset (8/16/32 native-width dispatch via the
//! `<B: Int>` monomorph), remapping each split feature's raw bin
//! (`bin ∈ [min,max] ? bin−min+offset : most_freq_bin`), routing numerically
//! (missing/`default_left` → threshold) or categorically (bitset membership), and
//! adding `leaf_value[~node]` into a `double*` score accumulator on reaching a leaf.
//!
//! ## What lives here
//! - [`add_prediction_to_score_kernel`] — the `#[cube(launch)]` tree-walk over one
//!   row per unit (`ABSOLUTE_POS`), `USE_INDICES` comptime (used-index subset vs
//!   identity). The **categorical membership branch reuses the shared
//!   [`crate::kernels::data_partition::find_in_bitset`] helper** (one transcription).
//!   The numeric branch transcribes the reference predict route
//!   (`cuda_tree.cu:376-391`) — a DISTINCT, simpler decision than the dense_bin
//!   `SplitInner` fan-out (`route_to_left`): it works on the ALREADY-remapped bin
//!   with a runtime `missing_type`/`default_left` read per node, so the comptime
//!   `route_to_left` flag fan-out is neither applicable nor callable inside a
//!   runtime multi-node walk.
//! - [`add_prediction_bagging_kernel`] — the §9 `AddPredictionToScoreKernel<USE_BAGGING>`
//!   per-row leaf-map gather-add: `score[data_index] += leaf_value[leaf_map[data_index]]`.
//! - [`add_prediction_to_score_on_device`] / [`add_prediction_bagging_on_device`] —
//!   the host drivers (input validation at the host boundary before the confined
//!   `unsafe` launch).
//!
//! Additive and OFF by default behind `LGBM_CUDA_ON_DEVICE`; the score
//! accumulator + scalar leaf-value math are the ONLY f64 — bin reads are
//! native-width integer (`u32::cast_from`), no f64 per-row hot loop.
//! The objective inverse-link (`ConvertOutput`) stays HOST-side at the readback
//! boundary (not yet moved on-device). Anchored to the cubecl-cpu
//! f64 fold, never GPU-vs-GPU.

use cubecl::prelude::*;
use cubecl::server::Handle;

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
/// accumulator is the ONLY f64 in the loop.
///
/// `bins` is the row-major `[num_rows × num_features]` binned matrix at native
/// width `B`; `feat_*` are the per-(inner)feature meta arrays (indexed by
/// `split_feature_inner[node]`). Bounds-guarded (`i < num_rows`).
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
                // find_in_bitset helper. The cat's bitset occupies pool
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

/// `AddPredictionToScoreKernel<USE_BAGGING>` (§9) — the per-row leaf-map
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

/// The §9 `UpdateDataIndexToLeafIndexKernel` analog —
/// derive the per-row `data_index_to_leaf[row] = leaf` map ON DEVICE from the resident
/// leaf-grouped partition layout, retiring the host `O(num_data)` inversion loop that
/// `add_prediction_to_score_on_device_resident` used to run.
///
/// One lane PER LEAF `l ∈ [0, num_leaves)`: the lane walks its own resident permutation
/// sub-range `indices[leaf_begin[l] .. leaf_begin[l] + leaf_count[l]]` and writes its
/// leaf id into `data_index_to_leaf[row]` for each of those rows. The reference derives
/// the leaf map on device during the bit-vector pass (§9); this is the resident-partition
/// analog. The output is pre-initialised to the `-1` sentinel, so any row NOT covered by
/// a leaf sub-range keeps `-1` (the scatter delegate / resident kernel rejects/skips it),
/// preserving the covered-once total-function contract without a per-row host inversion
/// loop. The sub-ranges are non-overlapping (a data partition), so no two lanes race on
/// the same output slot.
///
/// Static geometry only (`CubeCount::Static`, `num_leaves` lanes) — never a dynamic
/// (device-derived) launch count.
#[cube(launch)]
pub(crate) fn derive_leaf_map_kernel(
    indices: &Array<u32>,
    leaf_begin: &Array<u32>,
    leaf_count: &Array<u32>,
    data_index_to_leaf: &mut Array<i32>,
    num_leaves: u32,
    num_data: u32,
) {
    let l = ABSOLUTE_POS;
    if l < num_leaves as usize {
        let b = leaf_begin[l] as usize;
        let c = leaf_count[l] as usize;
        let leaf_id = i32::cast_from(l);
        for k in 0..c {
            let row = indices[b + k];
            // Defense-in-depth: bound the scattered row-id VALUE so an out-of-range
            // permutation entry cannot write past `data_index_to_leaf` (the host boundary in
            // `upload_leaf_map_inputs` rejects it before the launch; this guard makes the scatter
            // itself total — an out-of-range row simply does not write).
            if (row as usize) < num_data as usize {
                data_index_to_leaf[row as usize] = leaf_id;
            }
        }
    }
}

/// The §9 leaf-value SCATTER into an EXISTING
/// device-RESIDENT f64 score accumulator (accumulate in place, NO zero-init, NO
/// readback). One lane per row: `score[i] += leaf_value[data_index_to_leaf[i]]` guarded
/// by the `-1` sentinel (an uncovered row contributes nothing). This is the identity
/// (covered-once) resident analog of [`add_prediction_bagging_kernel`] — it never
/// zero-initialises `score`, so successive trees accumulate into the same resident
/// buffer across a whole train (§11 `cuda_score_`).
#[cube(launch)]
pub(crate) fn scatter_leaf_values_resident_kernel(
    data_index_to_leaf: &Array<i32>,
    leaf_value: &Array<f64>,
    score: &mut Array<f64>,
    num_data: u32,
) {
    let i = ABSOLUTE_POS;
    if i < num_data as usize {
        let leaf = data_index_to_leaf[i];
        if leaf >= 0 {
            score[i] += leaf_value[leaf as usize];
        }
    }
}

// =========================================================================
// Host drivers (validation → confined unsafe launch).
// =========================================================================

/// The tree + per-feature meta arrays the [`add_prediction_to_score_on_device`]
/// walk consumes. Field names mirror the flat `DeviceCudaTree` layout
/// (`split_feature_inner`/`threshold_in_bin`/`decision_type`/`left_child`/
/// `right_child`/`leaf_value` + the `cat_boundaries_inner` bitset slab); the
/// arrays here are built standalone from the parity fixture rather than
/// consuming a live `DeviceCudaTree`.
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
/// selects the native column width. `used_indices == None` walks all rows
/// (identity); otherwise the used-index subset. The objective inverse-link stays
/// HOST-side at readback — this emits raw margin only.
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

    // The f64 score accumulator, sized to num_data once, initialised to 0.
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
            // all outlive the launch; the kernel bounds-guards `i < num_rows`.
            // `validate_walk` has already range-checked every tree index the walk
            // dereferences: split_feature_inner ∈ [0, nf), each child either a
            // leaf `~child < nl` or an internal node `< nn`, categorical cat_idx within
            // cat_boundaries_inner, and cat_boundaries_inner monotone with its last word
            // bound ≤ bitset_inner.len() (the `n` passed to find_in_bitset).
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

/// Drive the §9 [`add_prediction_bagging_kernel`] leaf-map gather-add on
/// runtime `R`, returning the `num_data`-length f64 score. `data_index_to_leaf` maps
/// each data index to its resident leaf; `leaf_value` the per-leaf output.
/// `used_indices == None` adds every row (identity); else the used-index subset.
///
/// # Preconditions (debug-checked)
/// `used_indices` must be unique — the kernel does `score[data_index] += ...`,
/// so a duplicated index double-counts (and races on a real GPU backend). Reference
/// bagging indices are unique; a `debug_assert` guards it with no release-mode cost.
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
    // Validate a leaf-map entry only where the kernel actually reads it. A
    // single closure keeps the error text identical for both paths.
    let check_leaf = |di: usize, leaf: i32| -> Result<(), ComputeError> {
        if leaf < 0 || leaf as usize >= leaf_value.len() {
            return Err(ComputeError::Runtime {
                detail: format!(
                    "add_prediction_bagging: data index {di} maps to leaf {leaf} out of range \
                     (num_leaves = {})",
                    leaf_value.len()
                ),
            });
        }
        Ok(())
    };
    let (use_indices, idx_owned, num_rows): (bool, Vec<u32>, usize) = match used_indices {
        Some(idx) => {
            // Subset (USE_BAGGING) mode reads only data_index_to_leaf[idx[i]],
            // so validate ONLY the walked entries. Un-sampled rows legitimately carry
            // the `-1` sentinel that `update_data_index_to_leaf_on` writes
            // (data_partition.rs:841) and must NOT be rejected — the kernel never reads
            // them.
            for &di in idx {
                if di as usize >= num_data {
                    return Err(ComputeError::Runtime {
                        detail: format!("add_prediction_bagging: used index {di} >= num_data {num_data}"),
                    });
                }
                check_leaf(di as usize, data_index_to_leaf[di as usize])?;
            }
            // `score[data_index] += ...` requires unique used indices — a
            // duplicate double-counts (and races on a real GPU backend). Debug-only.
            debug_assert!(
                {
                    let mut seen = std::collections::HashSet::with_capacity(idx.len());
                    idx.iter().all(|&di| seen.insert(di))
                },
                "add_prediction_bagging: used_indices must be unique (a duplicate index double-counts)"
            );
            (true, idx.to_vec(), idx.len())
        }
        None => {
            // Identity mode walks every row, so every leaf-map entry is read.
            for (di, &leaf) in data_index_to_leaf.iter().enumerate() {
                check_leaf(di, leaf)?;
            }
            (false, vec![0u32], num_data)
        }
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

const LEAFMAP_BLOCK: u32 = 256;

/// Validate the leaf-grouped partition layout at the host boundary (`O(num_leaves)`,
/// NOT `O(num_data)`) and upload it, returning the uploaded `(indices, leaf_begin,
/// leaf_count)` device handles + a fresh `data_index_to_leaf` handle pre-initialised to
/// the `-1` sentinel (`num_data` long). Shared by [`derive_leaf_map_device`] (reads the
/// map back) and [`derive_leaf_map_device_handle`] (keeps it resident). Each leaf's
/// `[begin, begin + count)` must be non-negative and stay within `indices` — the
/// per-leaf bounds check that the retired host inversion loop used to fold into its
/// per-row write.
fn upload_leaf_map_inputs<R: cubecl::Runtime>(
    client: &cubecl::prelude::ComputeClient<R>,
    indices: &[u32],
    leaf_begin: &[i32],
    leaf_count: &[i32],
    num_data: usize,
) -> Result<(Handle, Handle, Handle, Handle, u32, u32), ComputeError> {
    if leaf_begin.len() != leaf_count.len() {
        return Err(ComputeError::LengthMismatch {
            expected: leaf_begin.len(),
            actual: leaf_count.len(),
        });
    }
    // O(num_positions) row-id VALUE bound: mirror the retired host inversion's
    // `if r >= num_data { return Err(..) }` guard. The per-leaf `[begin, begin + count) <=
    // indices.len()` check below bounds the READ positions in `indices`; this bounds the row-id
    // VALUES that index the `num_data`-sized `data_index_to_leaf` buffer. Without it a malformed
    // permutation launches an out-of-bounds device scatter inside `unsafe` (UB) instead of
    // returning a clean error. The data is already being uploaded, so this walk is not extra I/O.
    if let Some(&bad) = indices.iter().find(|&&r| (r as usize) >= num_data) {
        return Err(ComputeError::Runtime {
            detail: format!("derive_leaf_map: a partition row id {bad} is >= num_data {num_data}"),
        });
    }
    // O(num_leaves) per-leaf bounds validation (NOT a per-row O(num_data) loop): each
    // leaf sub-range is non-negative and stays within `indices`.
    let mut begin_u32 = Vec::with_capacity(leaf_begin.len());
    let mut count_u32 = Vec::with_capacity(leaf_count.len());
    let mut covered: usize = 0;
    for (&begin, &count) in leaf_begin.iter().zip(leaf_count.iter()) {
        let begin = usize::try_from(begin).map_err(|_| ComputeError::Runtime {
            detail: format!("derive_leaf_map: negative leaf_begin {begin}"),
        })?;
        let count = usize::try_from(count).map_err(|_| ComputeError::Runtime {
            detail: format!("derive_leaf_map: negative leaf_count {count}"),
        })?;
        let end = begin
            .checked_add(count)
            .filter(|&e| e <= indices.len())
            .ok_or_else(|| ComputeError::LengthMismatch {
                expected: indices.len(),
                actual: begin.saturating_add(count),
            })?;
        let _ = end;
        covered += count;
        begin_u32.push(begin as u32);
        count_u32.push(count as u32);
    }
    // Covered-once total-function guard (O(num_leaves)): the leaf sub-ranges partition
    // exactly the `indices` positions. Non-overlap is guaranteed by the layout
    // (a data partition); the count sum catches an uncovered/duplicated position.
    debug_assert_eq!(
        covered,
        indices.len(),
        "derive_leaf_map: leaf sub-ranges must cover every partition position exactly once"
    );

    let h_indices = client.create_from_slice(u32::as_bytes(indices));
    let h_begin = client.create_from_slice(u32::as_bytes(&begin_u32));
    let h_count = client.create_from_slice(u32::as_bytes(&count_u32));
    let init = vec![-1i32; num_data];
    let h_map = client.create_from_slice(i32::as_bytes(&init));
    Ok((
        h_indices,
        h_begin,
        h_count,
        h_map,
        indices.len() as u32,
        leaf_begin.len() as u32,
    ))
}

/// Derive `data_index_to_leaf` ON DEVICE from the
/// leaf-grouped resident partition (`indices` grouped by leaf via `leaf_begin` /
/// `leaf_count`) and read the map back to host. This is the device replacement for the
/// host `O(num_data)` inversion loop; it launches [`derive_leaf_map_kernel`] over
/// `indices.len()` lanes (static geometry) and returns the `num_data`-length leaf map
/// (`-1` for any uncovered row). Used by the parity test and by the eager
/// back-compat score path.
///
/// # Errors
/// [`ComputeError::LengthMismatch`] on a `leaf_begin`/`leaf_count` length mismatch or a
/// leaf sub-range overrunning `indices`; [`ComputeError::Runtime`] on a negative
/// `leaf_begin`/`leaf_count`.
pub fn derive_leaf_map_device<R: cubecl::Runtime>(
    client: &cubecl::prelude::ComputeClient<R>,
    indices: &[u32],
    leaf_begin: &[i32],
    leaf_count: &[i32],
    num_data: usize,
) -> Result<Vec<i32>, ComputeError> {
    let handle = derive_leaf_map_device_handle(client, indices, leaf_begin, leaf_count, num_data)?;
    let bytes = client.read_one_unchecked(handle);
    Ok(i32::from_bytes(&bytes).to_vec())
}

/// The RESIDENT variant of [`derive_leaf_map_device`]:
/// derive `data_index_to_leaf` on device and return the device [`Handle`] WITHOUT
/// crossing it back to host, so the resident score scatter can consume it in place. The
/// caller owns the returned handle (drop = free).
///
/// # Errors
/// As [`derive_leaf_map_device`].
pub fn derive_leaf_map_device_handle<R: cubecl::Runtime>(
    client: &cubecl::prelude::ComputeClient<R>,
    indices: &[u32],
    leaf_begin: &[i32],
    leaf_count: &[i32],
    num_data: usize,
) -> Result<Handle, ComputeError> {
    let (h_indices, h_begin, h_count, h_map, num_positions, num_leaves) =
        upload_leaf_map_inputs(client, indices, leaf_begin, leaf_count, num_data)?;
    if num_positions == 0 || num_leaves == 0 {
        return Ok(h_map);
    }
    let cube_count = num_leaves.div_ceil(LEAFMAP_BLOCK);
    // SAFETY: `h_indices` sized `num_positions`, `h_begin`/`h_count` sized `num_leaves`,
    // `h_map` sized `num_data`; all outlive the launch. The kernel bounds-guards
    // `l < num_leaves`, and `upload_leaf_map_inputs` has range-checked each leaf's
    // `[begin, begin + count) <= num_positions` so the READ `indices[b + k]` is in bounds.
    // The WRITTEN index `data_index_to_leaf[indices[b + k]]` is in bounds because
    // `upload_leaf_map_inputs` rejects any `indices[i] >= num_data` (the per-VALUE bound)
    // BEFORE this launch — NOT because `num_data == layout.num_data` (a COUNT equality,
    // which bounds no individual value); the kernel's `row < num_data` guard is defense-in-depth.
    // The leaf sub-ranges are non-overlapping, so no two lanes write the same slot.
    unsafe {
        derive_leaf_map_kernel::launch::<R>(
            client,
            CubeCount::Static(cube_count, 1, 1),
            CubeDim::new_1d(LEAFMAP_BLOCK),
            ArrayArg::from_raw_parts(h_indices, num_positions as usize),
            ArrayArg::from_raw_parts(h_begin, num_leaves as usize),
            ArrayArg::from_raw_parts(h_count, num_leaves as usize),
            ArrayArg::from_raw_parts(h_map.clone(), num_data),
            num_leaves,
            num_data as u32,
        );
    }
    Ok(h_map)
}

/// The §11 `AddPredictionToScore` — scatter the
/// (post-shrinkage) `leaf_values` into an EXISTING device-RESIDENT f64 score buffer
/// `score` (accumulate in place, NO zero-init, NO readback) via
/// [`scatter_leaf_values_resident_kernel`], reading the resident device leaf map
/// `leaf_map` (from [`derive_leaf_map_device_handle`]). Identity walk over all
/// `num_data` rows; the `-1` sentinel guard skips any uncovered row. This never crosses
/// the score back — the caller reads it on demand.
///
/// # Errors
/// [`ComputeError::Runtime`] never returned currently (validation is the caller's
/// `num_data`/`leaf_values` contract); the signature keeps the `Result` for symmetry
/// with the other device drivers.
pub fn add_leaf_values_to_resident_score<R: cubecl::Runtime>(
    client: &cubecl::prelude::ComputeClient<R>,
    leaf_map: &Handle,
    score: &Handle,
    leaf_values: &[f64],
    num_data: usize,
) -> Result<(), ComputeError> {
    if num_data == 0 {
        return Ok(());
    }
    let h_leaf = client.create_from_slice(f64::as_bytes(leaf_values));
    let cube_count = (num_data as u32).div_ceil(LEAFMAP_BLOCK);
    // SAFETY: `leaf_map`/`score` sized `num_data`, `h_leaf` sized `leaf_values.len()`;
    // the kernel bounds-guards `i < num_data`, skips the `-1` sentinel, and reads
    // `leaf_value[leaf]` only for `leaf >= 0` (the resident-score caller guarantees
    // `max leaf id < leaf_values.len()` — one leaf value per grown leaf).
    unsafe {
        scatter_leaf_values_resident_kernel::launch::<R>(
            client,
            CubeCount::Static(cube_count, 1, 1),
            CubeDim::new_1d(LEAFMAP_BLOCK),
            ArrayArg::from_raw_parts(leaf_map.clone(), num_data),
            ArrayArg::from_raw_parts(h_leaf, leaf_values.len()),
            ArrayArg::from_raw_parts(score.clone(), num_data),
            num_data as u32,
        );
    }
    Ok(())
}

/// The FUSED derive+scatter (§9+§11) resident `AddPredictionToScore`: one lane per
/// PARTITION POSITION `k` finds its leaf by scanning the (disjoint) per-leaf
/// `[begin, begin+count)` ranges and accumulates `score[indices[k]] +=
/// leaf_value[leaf]` in place — replacing the two-kernel
/// [`derive_leaf_map_device_handle`] → [`add_leaf_values_to_resident_score`] chain
/// AND its `num_data`-length `-1`-sentinel map buffer (one fewer `num_data×4`-byte
/// host→device fill-upload per tree). The retired derive kernel parallelized over
/// LEAVES (≤ `num_leaves` active lanes serially walking whole leaves — a single
/// active warp at `num_leaves=31`, ~O(num_data) serial device time per tree at 500k
/// rows); this kernel parallelizes over the `num_positions` rows with an
/// `O(num_leaves)` cached range scan per lane. Bit-exact by construction: the leaf
/// ranges are disjoint, so each row is written exactly ONCE with the SAME f64 `+=`
/// of the SAME leaf value — no reduction, no ordering dependence, identical to the
/// two-kernel chain's result cell-for-cell.
#[cube(launch)]
pub(crate) fn scatter_leaf_values_by_ranges_kernel(
    indices: &Array<u32>,
    leaf_begin: &Array<u32>,
    leaf_count: &Array<u32>,
    leaf_value: &Array<f64>,
    score: &mut Array<f64>,
    num_leaves: u32,
    num_positions: u32,
    num_data: u32,
) {
    let k = ABSOLUTE_POS;
    if k < num_positions as usize {
        let ku = u32::cast_from(k);
        // Disjoint ranges ⇒ at most one leaf matches; scan all (no break — the
        // uniform loop keeps lane divergence bounded by num_leaves).
        let mut leaf = i32::new(-1);
        for l in 0..num_leaves as usize {
            let b = leaf_begin[l];
            let c = leaf_count[l];
            if ku >= b && ku < b + c {
                leaf = i32::cast_from(l);
            }
        }
        if leaf >= 0 {
            let row = indices[k];
            // Defense-in-depth VALUE bound (mirrors `derive_leaf_map_kernel`): an
            // out-of-range permutation entry does not write.
            if (row as usize) < num_data as usize {
                score[row as usize] = score[row as usize] + leaf_value[leaf as usize];
            }
        }
    }
}

/// Launcher for [`scatter_leaf_values_by_ranges_kernel`] — the fused resident
/// per-tree score scatter over a leaf-grouped partition layout. Validates the
/// per-leaf ranges at `O(num_leaves)` on the host (non-negative, within `indices`,
/// leaf id within `leaf_values`), uploads `indices`/ranges/leaf values, and launches
/// one lane per position. The row-id VALUE bound is enforced in-kernel (the scatter
/// is total — an out-of-range row simply does not write), so no `O(num_positions)`
/// host pre-scan is paid.
///
/// # Errors
/// [`ComputeError::Runtime`] on a negative `leaf_begin`/`leaf_count`;
/// [`ComputeError::LengthMismatch`] if `leaf_begin`/`leaf_count` disagree, a leaf
/// range overruns `indices`, or there are more leaves than `leaf_values`.
pub fn add_leaf_values_by_ranges_to_resident_score<R: cubecl::Runtime>(
    client: &cubecl::prelude::ComputeClient<R>,
    indices: &[u32],
    leaf_begin: &[i32],
    leaf_count: &[i32],
    score: &Handle,
    leaf_values: &[f64],
    num_data: usize,
) -> Result<(), ComputeError> {
    if leaf_begin.len() != leaf_count.len() {
        return Err(ComputeError::LengthMismatch {
            expected: leaf_begin.len(),
            actual: leaf_count.len(),
        });
    }
    if leaf_begin.len() > leaf_values.len() {
        return Err(ComputeError::LengthMismatch {
            expected: leaf_begin.len(),
            actual: leaf_values.len(),
        });
    }
    let num_positions = indices.len();
    let num_leaves = leaf_begin.len();
    if num_positions == 0 || num_leaves == 0 {
        return Ok(());
    }
    let mut begin_u32: Vec<u32> = Vec::with_capacity(num_leaves);
    let mut count_u32: Vec<u32> = Vec::with_capacity(num_leaves);
    for l in 0..num_leaves {
        let (b, c) = (leaf_begin[l], leaf_count[l]);
        if b < 0 || c < 0 {
            return Err(ComputeError::Runtime {
                detail: format!(
                    "add_leaf_values_by_ranges_to_resident_score: leaf {l} has negative \
                     begin/count ({b}/{c})"
                ),
            });
        }
        let end = b as usize + c as usize;
        if end > num_positions {
            return Err(ComputeError::LengthMismatch {
                expected: num_positions,
                actual: end,
            });
        }
        begin_u32.push(b as u32);
        count_u32.push(c as u32);
    }

    let h_indices = client.create_from_slice(u32::as_bytes(indices));
    let h_begin = client.create_from_slice(u32::as_bytes(&begin_u32));
    let h_count = client.create_from_slice(u32::as_bytes(&count_u32));
    let h_leaf = client.create_from_slice(f64::as_bytes(leaf_values));
    let cube_count = (num_positions as u32).div_ceil(LEAFMAP_BLOCK);
    // SAFETY: `h_indices` sized `num_positions`, `h_begin`/`h_count` sized
    // `num_leaves`, `h_leaf` sized `leaf_values.len() >= num_leaves`, `score` sized
    // `num_data` f64 cells (the ResidentScore contract); all outlive the launch. The
    // kernel bounds-guards `k < num_positions`, matches `leaf < num_leaves` only, and
    // guards the scattered `row < num_data` — every access is in bounds.
    unsafe {
        scatter_leaf_values_by_ranges_kernel::launch::<R>(
            client,
            CubeCount::Static(cube_count, 1, 1),
            CubeDim::new_1d(LEAFMAP_BLOCK),
            ArrayArg::from_raw_parts(h_indices, num_positions),
            ArrayArg::from_raw_parts(h_begin, num_leaves),
            ArrayArg::from_raw_parts(h_count, num_leaves),
            ArrayArg::from_raw_parts(h_leaf, leaf_values.len()),
            ArrayArg::from_raw_parts(score.clone(), num_data),
            num_leaves as u32,
            num_positions as u32,
            num_data as u32,
        );
    }
    Ok(())
}

/// Validate the tree-walk inputs at the host boundary before the
/// confined `unsafe` launch. Rejects a `rows`/`num_features` mismatch, an unequal
/// per-feature meta length, an unequal per-node tree length, and a `num_data` too
/// small for the walked data indices.
///
/// # Preconditions (debug-checked)
/// - **Identity path:** with `used_indices == None` the walk is the
///   `USE_INDICES=false` semantics, which require `num_rows == num_data` — every data
///   row is walked exactly once. A `num_rows < num_data` identity call silently scores
///   only the `[0, num_rows)` prefix and leaves the tail at 0; a `debug_assert`
///   rejects it. (Release builds keep the lenient behavior — this changes
///   no hot-path numeric output.)
/// - **Uniqueness:** `used_indices` must be unique. `add_prediction_*_kernel`
///   does `score[data_index] += ...`, so a duplicated index double-counts (and races
///   on a real GPU backend). Reference bagging indices are unique; a `debug_assert`
///   guards the precondition without a release-mode cost.
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
    // The identity (`USE_INDICES=false`) path requires num_rows == num_data. A
    // shorter num_rows silently scores only a prefix. Debug-only guard (no release
    // behavior change).
    debug_assert!(
        used_indices.is_some() || num_rows == num_data,
        "add_prediction_to_score: identity walk requires num_rows ({num_rows}) == num_data ({num_data}); \
         a shorter num_rows scores only the [0, num_rows) prefix"
    );
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
    // Validate the tree indices the walk actually dereferences. The launch
    // SAFETY comment claims the walk "only reads valid node/feature/leaf indices" —
    // enforce that here so a malformed fixture tree surfaces a typed ComputeError at
    // the host boundary instead of an in-kernel out-of-bounds.
    let nl = tree.leaf_value.len();
    for n in 0..nn {
        // split_feature_inner[node] indexes the per-feature meta arrays (len nf).
        let sf = tree.split_feature_inner[n];
        if sf < 0 || sf as usize >= nf {
            return Err(ComputeError::Runtime {
                detail: format!(
                    "add_prediction_to_score: node {n} split_feature_inner {sf} out of range [0, {nf})"
                ),
            });
        }
        // left_child/right_child: negative encodes a leaf (~child == -child-1) into
        // leaf_value; non-negative is an internal-node index back into the node arrays.
        for (side, child) in [("left", tree.left_child[n]), ("right", tree.right_child[n])] {
            if child < 0 {
                let leaf = (-child - 1) as usize;
                if leaf >= nl {
                    return Err(ComputeError::Runtime {
                        detail: format!(
                            "add_prediction_to_score: node {n} {side}_child leaf {leaf} >= num_leaves {nl}"
                        ),
                    });
                }
            } else if child as usize >= nn {
                return Err(ComputeError::Runtime {
                    detail: format!(
                        "add_prediction_to_score: node {n} {side}_child node {child} >= num_nodes {nn}"
                    ),
                });
            }
        }
        // Categorical nodes read cat_boundaries_inner[cat_idx] and [cat_idx + 1].
        if (tree.decision_type[n] & K_CATEGORICAL_MASK) != 0 {
            let cat_idx = tree.threshold_in_bin[n] as usize;
            if cat_idx + 1 >= tree.cat_boundaries_inner.len() {
                return Err(ComputeError::Runtime {
                    detail: format!(
                        "add_prediction_to_score: node {n} categorical cat_idx {cat_idx} out of \
                         cat_boundaries_inner (len {})",
                        tree.cat_boundaries_inner.len()
                    ),
                });
            }
        }
    }
    // cat_boundaries_inner must be monotone non-decreasing, and its final word bound
    // must lie within the bitset pool — find_in_bitset uses `end` as the length guard,
    // so `end <= bitset_inner.len()` is what keeps the pool read in-bounds.
    if !tree.cat_boundaries_inner.is_empty() {
        for w in tree.cat_boundaries_inner.windows(2) {
            if w[1] < w[0] {
                return Err(ComputeError::Runtime {
                    detail: format!(
                        "add_prediction_to_score: cat_boundaries_inner not monotone ({} < {})",
                        w[1], w[0]
                    ),
                });
            }
        }
        let last = *tree.cat_boundaries_inner.last().unwrap();
        if last < 0 || last as usize > tree.bitset_inner.len() {
            return Err(ComputeError::Runtime {
                detail: format!(
                    "add_prediction_to_score: cat_boundaries_inner last {last} exceeds \
                     bitset_inner len {}",
                    tree.bitset_inner.len()
                ),
            });
        }
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
        // `score[data_index] += ...` requires unique used indices — a duplicate
        // double-counts (and races on a real GPU backend). Debug-only check (no
        // release-mode cost on the hot path).
        debug_assert!(
            {
                let mut seen = std::collections::HashSet::with_capacity(idx.len());
                idx.iter().all(|&di| seen.insert(di))
            },
            "add_prediction_to_score: used_indices must be unique (a duplicate index double-counts)"
        );
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
    /// invariant index) — exercising the 8/16/32 dispatch coverage.
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
    /// the shared find_in_bitset reuse.
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

    /// Host-boundary validation rejects a rows/num_features mismatch.
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

    /// The FUSED range-scatter is BIT-EXACT vs the two-kernel derive→scatter chain
    /// over the same leaf-grouped layout and the same non-zero starting score
    /// (each row written exactly once with the same f64 `+=`), including a leaf
    /// permutation whose ranges are NOT ascending in leaf id.
    #[test]
    fn fused_ranges_scatter_matches_derive_then_scatter() {
        let client = cpu_client();
        let num_data = 10usize;
        // Leaf-grouped permutation: leaf 0 -> rows {7, 2, 9}, leaf 1 -> rows {0, 4},
        // leaf 2 -> rows {1, 3, 5, 6, 8}; ranges deliberately not in leaf order.
        let indices: Vec<u32> = vec![0, 4, 7, 2, 9, 1, 3, 5, 6, 8];
        let leaf_begin: Vec<i32> = vec![2, 0, 5];
        let leaf_count: Vec<i32> = vec![3, 2, 5];
        let leaf_values = vec![-0.5f64, 0.25, 1.75];
        let start: Vec<f64> = (0..num_data).map(|i| i as f64 * 0.125 - 0.4).collect();

        // Arm A: two-kernel chain (derive map, then map-scatter).
        let score_a = client.create_from_slice(f64::as_bytes(&start));
        let leaf_map =
            derive_leaf_map_device_handle(&client, &indices, &leaf_begin, &leaf_count, num_data)
                .expect("derive leaf map");
        add_leaf_values_to_resident_score(&client, &leaf_map, &score_a, &leaf_values, num_data)
            .expect("map scatter");

        // Arm B: fused range scatter.
        let score_b = client.create_from_slice(f64::as_bytes(&start));
        add_leaf_values_by_ranges_to_resident_score(
            &client,
            &indices,
            &leaf_begin,
            &leaf_count,
            &score_b,
            &leaf_values,
            num_data,
        )
        .expect("fused range scatter");

        let a = f64::from_bytes(&client.read_one_unchecked(score_a)).to_vec();
        let b = f64::from_bytes(&client.read_one_unchecked(score_b)).to_vec();
        assert_eq!(a.len(), num_data);
        for i in 0..num_data {
            assert_eq!(a[i].to_bits(), b[i].to_bits(), "row {i}: chain={} fused={}", a[i], b[i]);
        }
        // And every covered row actually moved off its start value (all 10 covered).
        for i in 0..num_data {
            assert_ne!(b[i].to_bits(), start[i].to_bits(), "row {i} was not scattered");
        }
    }

    /// Fused-scatter host-boundary validation: negative range and overrun both reject.
    #[test]
    fn fused_ranges_scatter_rejects_bad_ranges() {
        let client = cpu_client();
        let score = client.create_from_slice(f64::as_bytes(&[0.0f64; 4]));
        let indices: Vec<u32> = vec![0, 1, 2, 3];
        // Negative begin.
        assert!(matches!(
            add_leaf_values_by_ranges_to_resident_score(
                &client, &indices, &[-1], &[2], &score, &[0.5], 4
            ),
            Err(ComputeError::Runtime { .. })
        ));
        // Range overruns indices.
        assert!(matches!(
            add_leaf_values_by_ranges_to_resident_score(
                &client, &indices, &[3], &[2], &score, &[0.5], 4
            ),
            Err(ComputeError::LengthMismatch { .. })
        ));
        // More leaves than leaf values.
        assert!(matches!(
            add_leaf_values_by_ranges_to_resident_score(
                &client, &indices, &[0, 2], &[2, 2], &score, &[0.5], 4
            ),
            Err(ComputeError::LengthMismatch { .. })
        ));
    }
}
