//! On-device tree mutation — `Split` / `SplitCategorical` / `Shrinkage` / `AddBias`
//! (§10, ODL-14).
//!
//! The device-resident flat `CUDATree` (D-07) and the scalar/elementwise kernels
//! that mutate it, the Rust port of the AMD-fork
//! `LightGBM-release-4.6.0.99/src/io/cuda/cuda_tree.cu` (`SplitKernel:48`,
//! `SplitCategoricalKernel:160`, `ShrinkageKernel:290`, `AddBiasKernel:303`).
//!
//! ## What lives here (18-03, ODL-14)
//! - [`DeviceCudaTree`] — a **Struct-of-Arrays, host pre-allocated** device flat
//!   tree: one CubeCL [`Handle`] per field (`cuda_leaf_value_`,
//!   `cuda_left_child_`/`cuda_right_child_`, `cuda_decision_type_`,
//!   thresholds/counts/depth, `cat_boundaries`/`cat_boundaries_inner`),
//!   allocated **once** in [`DeviceCudaTree::new`] via a single counted
//!   `client.empty` site (D-15) — **NO per-split device allocation**. Mirrors the
//!   [`crate::kernels::split_info::DeviceSplitInfo::new`] counted-alloc SoA idiom.
//! - [`set_decision_type`] / [`set_missing_type`] — the `int8_t` decision-type
//!   bitfield packers (`kDefaultLeftMask`/`kCategoricalMask`), branchless `select`
//!   `#[cube]` helpers (SP-2, cubecl-cpu MLIR-safe).
//! - [`split_kernel`] — the `<<<3,5>>>` 15-thread scalar fan-out that writes the 14
//!   numeric tree fields from a [`SplitScalars`] record (NaN→0 on the leaf outputs
//!   via branchless `select`), rewires the parent/child links, and advances the
//!   node/leaf bookkeeping. [`DeviceCudaTree::split_on_device`] returns the
//!   `right_leaf_index` the partition step consumes (the §1/§10 hard ordering
//!   invariant, Pitfall 3): Split runs BEFORE partition.
//! - [`split_categorical_kernel`] — the same fan-out plus `kCategoricalMask`,
//!   `num_cat`, and the `cat_boundaries`/`cat_boundaries_inner` bitset-length
//!   append (the reserved slabs).
//! - [`shrinkage_kernel`] (`leaf_value *= rate`) / [`add_bias_kernel`]
//!   (`leaf_value += val`) — elementwise `#[cube]` bodies + thin launch wrappers
//!   (SP-1); scalar math stays f64 (SP-5 / D-14 — per-leaf, not per-row).
//! - [`DeviceCudaTree::to_host_tree`] — reconstruct the host `lgbm_model::Tree`
//!   from the flat arrays for the cpu-f64-anchor compare (D-07).
//!
//! Additive and OFF by default behind `LGBM_CUDA_ON_DEVICE` (D-13); the tree
//! kernels are scalar/elementwise with f64 confined to leaf-value/gain scalar math
//! — **no f64 per-row hot loop** (D-14). Anchored to the cubecl-cpu f64 fold, never
//! GPU-vs-GPU (D-12 / def-f8u-01).

use core::marker::PhantomData;

use cubecl::prelude::*;
use cubecl::server::Handle;

use crate::error::ComputeError;
use crate::kernels::split_info::SplitScalars;

/// `kCategoricalMask` (`tree.h:20`) — decision-type bit 0: set iff categorical split.
pub const K_CATEGORICAL_MASK: i32 = 1;
/// `kDefaultLeftMask` (`tree.h:21`) — decision-type bit 1: set iff missing routes left.
pub const K_DEFAULT_LEFT_MASK: i32 = 2;

// =========================================================================
// Decision-type bitfield packers (SP-2 branchless `select`, cubecl-cpu safe).
// Port of `SetDecisionTypeCUDA`/`SetMissingTypeCUDA` (cuda_tree.cu:13-23).
// =========================================================================

/// `SetDecisionTypeCUDA` (`cuda_tree.cu:13`): `input ? (dt | mask) : (dt & (127 -
/// mask))`. Branchless via `select` (SP-2). `mask` is a compile-time constant
/// (`kCategoricalMask`/`kDefaultLeftMask`).
#[cube]
pub fn set_decision_type(dt: i32, input: bool, #[comptime] mask: i32) -> i32 {
    select(input, dt | mask, dt & (127 - mask))
}

/// `SetMissingTypeCUDA` (`cuda_tree.cu:21`): `(dt & 3) | (missing << 2)`.
#[cube]
pub fn set_missing_type(dt: i32, missing: i32) -> i32 {
    (dt & 3) | (missing << 2)
}

/// `GetDecisionTypeCUDA` (`cuda_tree.cu:26`): `(decision_type & mask) > 0`.
/// Host mirror used by the reconstruction/assertions.
#[must_use]
pub fn get_decision_type(decision_type: i8, mask: i8) -> bool {
    (decision_type & mask) > 0
}

/// `GetMissingTypeCUDA` (`cuda_tree.cu:31`): `(decision_type >> 2) & 3`.
#[must_use]
pub fn get_missing_type(decision_type: i8) -> i8 {
    (decision_type >> 2) & 3
}

// =========================================================================
// Init kernel — set the flat tree to a fresh single-leaf root (D-07).
// =========================================================================

/// Initialize every leaf/node slot to the empty state: `leaf_parent = -1`,
/// `leaf_depth/leaf_value/leaf_weight = 0`, `leaf_count = 0` (slot 0 gets
/// `root_count`), and the node arrays to 0. Runs once after [`DeviceCudaTree::new`]
/// (the reference `CUDATree::InitCUDAMemory` analog); NOT a per-split allocation.
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
fn init_tree_kernel(
    leaf_value: &mut Array<f64>,
    leaf_weight: &mut Array<f64>,
    leaf_count: &mut Array<i32>,
    leaf_parent: &mut Array<i32>,
    leaf_depth: &mut Array<i32>,
    left_child: &mut Array<i32>,
    right_child: &mut Array<i32>,
    split_feature_inner: &mut Array<i32>,
    split_feature: &mut Array<i32>,
    internal_count: &mut Array<i32>,
    decision_type: &mut Array<i32>,
    threshold_in_bin: &mut Array<u32>,
    root_count: i32,
    n: u32,
) {
    let i = ABSOLUTE_POS;
    if i < n as usize {
        leaf_value[i] = 0.0f64;
        leaf_weight[i] = 0.0f64;
        leaf_count[i] = select(i == 0, root_count, 0i32);
        leaf_parent[i] = -1i32;
        leaf_depth[i] = 0i32;
        left_child[i] = 0i32;
        right_child[i] = 0i32;
        split_feature_inner[i] = 0i32;
        split_feature[i] = 0i32;
        internal_count[i] = 0i32;
        decision_type[i] = 0i32;
        threshold_in_bin[i] = 0u32;
    }
}

// =========================================================================
// SplitKernel — one numerical split (cuda_tree.cu:48, <<<3,5>>> = 15 threads).
// =========================================================================

/// `SplitKernel` (`cuda_tree.cu:48`): the 15-thread scalar fan-out that mutates the
/// flat device tree for ONE numerical split. Each `ABSOLUTE_POS` thread writes a
/// disjoint set of cells (no cross-thread read-after-write hazard, exactly as the
/// reference `<<<3,5>>>` launch — the parent-link read `parent_index` is taken by
/// all threads at entry, before thread 0's write). NaN leaf outputs are coerced to
/// `0.0` via branchless `select` (`isnan(x) ? 0.0f : x`, `:100/:106`). The `~x`
/// two's-complement child encoding is written as `-x - 1` (portable, no cube
/// bitwise-not dependency).
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
fn split_kernel(
    leaf_value: &mut Array<f64>,
    leaf_weight: &mut Array<f64>,
    leaf_count: &mut Array<i32>,
    leaf_parent: &mut Array<i32>,
    leaf_depth: &mut Array<i32>,
    left_child: &mut Array<i32>,
    right_child: &mut Array<i32>,
    split_feature_inner: &mut Array<i32>,
    split_feature: &mut Array<i32>,
    split_gain: &mut Array<f32>,
    internal_weight: &mut Array<f64>,
    internal_value: &mut Array<f64>,
    internal_count: &mut Array<i32>,
    decision_type: &mut Array<i32>,
    threshold_in_bin: &mut Array<u32>,
    threshold: &mut Array<f64>,
    leaf_index: i32,
    num_leaves: i32,
    real_feature_index: i32,
    inner_feature_index: i32,
    real_threshold: f64,
    threshold_bin: u32,
    gain: f64,
    default_left: u32,
    missing_type: i32,
    left_sum_hessians: f64,
    right_sum_hessians: f64,
    left_value: f64,
    right_value: f64,
    left_count: i32,
    right_count: i32,
) {
    let new_node_index = (num_leaves - 1) as usize;
    let leaf = leaf_index as usize;
    let num_leaves_u = num_leaves as usize;
    let thread_index = ABSOLUTE_POS;
    // Read by all threads BEFORE any write (only thread 0 uses it) — the reference
    // relies on the same read-before-rewire ordering (:73).
    let parent_index = leaf_parent[leaf];

    if thread_index == 0 {
        // Rewire the parent's child pointer to the new internal node. Root leaf has
        // parent -1 (no rewiring). `~leaf_index == -leaf_index - 1`.
        if parent_index >= 0 {
            let p = parent_index as usize;
            let is_left = left_child[p] == (-leaf_index - 1);
            let lc = select(is_left, new_node_index as i32, left_child[p]);
            let rc = select(is_left, right_child[p], new_node_index as i32);
            left_child[p] = lc;
            right_child[p] = rc;
        }
        left_child[new_node_index] = -leaf_index - 1; // ~leaf_index
        right_child[new_node_index] = -num_leaves - 1; // ~num_leaves
        leaf_parent[leaf] = new_node_index as i32;
        leaf_parent[num_leaves_u] = new_node_index as i32;
    } else if thread_index == 1 {
        split_feature_inner[new_node_index] = inner_feature_index;
    } else if thread_index == 2 {
        split_feature[new_node_index] = real_feature_index;
    } else if thread_index == 3 {
        split_gain[new_node_index] = f32::cast_from(gain);
    } else if thread_index == 4 {
        internal_weight[new_node_index] = left_sum_hessians + right_sum_hessians;
        leaf_weight[leaf] = left_sum_hessians;
    } else if thread_index == 5 {
        internal_value[new_node_index] = leaf_value[leaf];
        // isnan(left_value) ? 0.0 : left_value (branchless — cubecl `f64::is_nan`).
        leaf_value[leaf] = select(f64::is_nan(left_value), 0.0f64, left_value);
    } else if thread_index == 6 {
        internal_count[new_node_index] = left_count + right_count;
    } else if thread_index == 7 {
        leaf_count[leaf] = left_count;
    } else if thread_index == 8 {
        leaf_value[num_leaves_u] = select(f64::is_nan(right_value), 0.0f64, right_value);
    } else if thread_index == 9 {
        leaf_weight[num_leaves_u] = right_sum_hessians;
    } else if thread_index == 10 {
        leaf_count[num_leaves_u] = right_count;
    } else if thread_index == 11 {
        leaf_depth[num_leaves_u] = leaf_depth[leaf] + 1;
        leaf_depth[leaf] = leaf_depth[leaf] + 1;
    } else if thread_index == 12 {
        // decision_type = 0; clear categorical, set default_left, set missing.
        let dt = set_missing_type(
            set_decision_type(
                set_decision_type(0i32, false, K_CATEGORICAL_MASK),
                default_left != 0,
                K_DEFAULT_LEFT_MASK,
            ),
            missing_type,
        );
        decision_type[new_node_index] = dt;
    } else if thread_index == 13 {
        threshold_in_bin[new_node_index] = threshold_bin;
    } else if thread_index == 14 {
        threshold[new_node_index] = real_threshold;
    }
}

// =========================================================================
// SplitCategoricalKernel — one categorical split (cuda_tree.cu:160, <<<3,6>>>).
// =========================================================================

/// `SplitCategoricalKernel` (`cuda_tree.cu:160`): the 17-thread fan-out. Identical
/// node/leaf bookkeeping to [`split_kernel`] (threads 0-11) but thread 12 sets
/// `kCategoricalMask` (default_left NOT encoded — categorical routes NaN/negative
/// categories right), threads 13/14 write `num_cat` as the threshold, and threads
/// 15/16 append the bitset lengths to `cat_boundaries`/`cat_boundaries_inner`
/// (`cat_boundaries[num_cat+1] = cat_boundaries[num_cat] + bitset_len`, seeded to 0
/// at `num_cat == 0`).
#[cube(launch)]
#[allow(clippy::too_many_arguments)]
fn split_categorical_kernel(
    leaf_value: &mut Array<f64>,
    leaf_weight: &mut Array<f64>,
    leaf_count: &mut Array<i32>,
    leaf_parent: &mut Array<i32>,
    leaf_depth: &mut Array<i32>,
    left_child: &mut Array<i32>,
    right_child: &mut Array<i32>,
    split_feature_inner: &mut Array<i32>,
    split_feature: &mut Array<i32>,
    split_gain: &mut Array<f32>,
    internal_weight: &mut Array<f64>,
    internal_value: &mut Array<f64>,
    internal_count: &mut Array<i32>,
    decision_type: &mut Array<i32>,
    threshold_in_bin: &mut Array<u32>,
    threshold: &mut Array<f64>,
    cat_boundaries: &mut Array<i32>,
    cat_boundaries_inner: &mut Array<i32>,
    leaf_index: i32,
    num_leaves: i32,
    real_feature_index: i32,
    inner_feature_index: i32,
    gain: f64,
    missing_type: i32,
    left_sum_hessians: f64,
    right_sum_hessians: f64,
    left_value: f64,
    right_value: f64,
    left_count: i32,
    right_count: i32,
    num_cat: i32,
    bitset_len: i32,
    bitset_inner_len: i32,
) {
    let new_node_index = (num_leaves - 1) as usize;
    let leaf = leaf_index as usize;
    let num_leaves_u = num_leaves as usize;
    let num_cat_u = num_cat as usize;
    let thread_index = ABSOLUTE_POS;
    let parent_index = leaf_parent[leaf];

    if thread_index == 0 {
        if parent_index >= 0 {
            let p = parent_index as usize;
            let is_left = left_child[p] == (-leaf_index - 1);
            let lc = select(is_left, new_node_index as i32, left_child[p]);
            let rc = select(is_left, right_child[p], new_node_index as i32);
            left_child[p] = lc;
            right_child[p] = rc;
        }
        left_child[new_node_index] = -leaf_index - 1;
        right_child[new_node_index] = -num_leaves - 1;
        leaf_parent[leaf] = new_node_index as i32;
        leaf_parent[num_leaves_u] = new_node_index as i32;
    } else if thread_index == 1 {
        split_feature_inner[new_node_index] = inner_feature_index;
    } else if thread_index == 2 {
        split_feature[new_node_index] = real_feature_index;
    } else if thread_index == 3 {
        split_gain[new_node_index] = f32::cast_from(gain);
    } else if thread_index == 4 {
        internal_weight[new_node_index] = left_sum_hessians + right_sum_hessians;
        leaf_weight[leaf] = left_sum_hessians;
    } else if thread_index == 5 {
        internal_value[new_node_index] = leaf_value[leaf];
        leaf_value[leaf] = select(f64::is_nan(left_value), 0.0f64, left_value);
    } else if thread_index == 6 {
        internal_count[new_node_index] = left_count + right_count;
    } else if thread_index == 7 {
        leaf_count[leaf] = left_count;
    } else if thread_index == 8 {
        leaf_value[num_leaves_u] = select(f64::is_nan(right_value), 0.0f64, right_value);
    } else if thread_index == 9 {
        leaf_weight[num_leaves_u] = right_sum_hessians;
    } else if thread_index == 10 {
        leaf_count[num_leaves_u] = right_count;
    } else if thread_index == 11 {
        leaf_depth[num_leaves_u] = leaf_depth[leaf] + 1;
        leaf_depth[leaf] = leaf_depth[leaf] + 1;
    } else if thread_index == 12 {
        // categorical: set kCategoricalMask, set missing; default_left NOT encoded.
        let dt = set_missing_type(
            set_decision_type(0i32, true, K_CATEGORICAL_MASK),
            missing_type,
        );
        decision_type[new_node_index] = dt;
    } else if thread_index == 13 {
        threshold_in_bin[new_node_index] = num_cat as u32;
    } else if thread_index == 14 {
        threshold[new_node_index] = f64::cast_from(num_cat);
    } else if thread_index == 15 {
        if num_cat == 0 {
            cat_boundaries[0] = 0i32;
        }
        cat_boundaries[num_cat_u + 1] = cat_boundaries[num_cat_u] + bitset_len;
    } else if thread_index == 16 {
        if num_cat == 0 {
            cat_boundaries_inner[0] = 0i32;
        }
        cat_boundaries_inner[num_cat_u + 1] = cat_boundaries_inner[num_cat_u] + bitset_inner_len;
    }
}

// =========================================================================
// ShrinkageKernel / AddBiasKernel — elementwise leaf-value math (SP-1, D-14).
// =========================================================================

/// The shared elementwise `#[cube]` body: `values[i] = values[i] OP scalar`, where
/// `is_add != 0` selects `+ scalar` (AddBias) and `is_add == 0` selects `* scalar`
/// (Shrinkage). Single source both launch wrappers call (SP-1). Scalar math is f64
/// (per-leaf, not per-row — D-14).
#[cube]
fn leaf_value_op(values: &mut Array<f64>, scalar: f64, is_add: u32, n: u32) {
    let i = ABSOLUTE_POS;
    if i < n as usize {
        let v = values[i];
        values[i] = select(is_add != 0, v + scalar, v * scalar);
    }
}

/// `ShrinkageKernel` (`cuda_tree.cu:290`): `cuda_leaf_value[i] *= rate`. Thin
/// wrapper over [`leaf_value_op`] (SP-1).
#[cube(launch)]
fn shrinkage_kernel(values: &mut Array<f64>, rate: f64, n: u32) {
    leaf_value_op(values, rate, 0u32, n);
}

/// `AddBiasKernel` (`cuda_tree.cu:303`): `cuda_leaf_value[i] += val`. Thin wrapper
/// over [`leaf_value_op`] (SP-1).
#[cube(launch)]
fn add_bias_kernel(values: &mut Array<f64>, val: f64, n: u32) {
    leaf_value_op(values, val, 1u32, n);
}

/// The elementwise-op launch dim (`cuda_tree.cu` uses `1024` threads/block).
const LEAF_OP_BLOCK: u32 = 1024;

/// Apply `leaf_value *= rate` to `values` on the runtime `R`, returning the scaled
/// vector (ShrinkageKernel, `cuda_tree.cu:290`). Standalone helper for the
/// `tree::shrinkage` unit; [`DeviceCudaTree::shrink`] runs the same kernel in place.
///
/// # Errors
/// [`ComputeError::Runtime`] if `values` is empty.
pub fn apply_shrinkage_on<R: cubecl::Runtime>(
    client: &ComputeClient<R>,
    values: &[f64],
    rate: f64,
) -> Result<Vec<f64>, ComputeError> {
    apply_leaf_op_on(client, values, rate, false)
}

/// Apply `leaf_value += val` to `values` on the runtime `R`, returning the shifted
/// vector (AddBiasKernel, `cuda_tree.cu:303`).
///
/// # Errors
/// [`ComputeError::Runtime`] if `values` is empty.
pub fn apply_add_bias_on<R: cubecl::Runtime>(
    client: &ComputeClient<R>,
    values: &[f64],
    val: f64,
) -> Result<Vec<f64>, ComputeError> {
    apply_leaf_op_on(client, values, val, true)
}

fn apply_leaf_op_on<R: cubecl::Runtime>(
    client: &ComputeClient<R>,
    values: &[f64],
    scalar: f64,
    is_add: bool,
) -> Result<Vec<f64>, ComputeError> {
    let n = values.len();
    if n == 0 {
        return Err(ComputeError::Runtime {
            detail: "apply_leaf_op: values must be non-empty".to_string(),
        });
    }
    let h = client.create_from_slice(f64::as_bytes(values));
    let blocks = (n as u32).div_ceil(LEAF_OP_BLOCK);
    // SAFETY: `h` is sized for exactly `n` f64 cells; the kernel guards `i < n`, so
    // every write stays in-bounds. All cubecl unsafe is confined here (CMP-01).
    unsafe {
        if is_add {
            add_bias_kernel::launch(
                client,
                CubeCount::Static(blocks, 1, 1),
                CubeDim::new_1d(LEAF_OP_BLOCK),
                ArrayArg::from_raw_parts(h.clone(), n),
                scalar,
                n as u32,
            );
        } else {
            shrinkage_kernel::launch(
                client,
                CubeCount::Static(blocks, 1, 1),
                CubeDim::new_1d(LEAF_OP_BLOCK),
                ArrayArg::from_raw_parts(h.clone(), n),
                scalar,
                n as u32,
            );
        }
    }
    let bytes = client.read_one_unchecked(h);
    Ok(f64::from_bytes(&bytes).to_vec())
}

// =========================================================================
// The device flat CUDATree (D-07, pre-allocated once — D-15).
// =========================================================================

/// The result of one on-device split: the new internal node index and its two
/// child leaf ids. `right_leaf_index` is the id the partition step consumes (the
/// §1/§10 Split-before-partition ordering invariant, Pitfall 3).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SplitResult {
    /// The new internal node index (`num_leaves - 1` at split time).
    pub new_node_index: i32,
    /// The left child leaf id (== the split leaf, whose output is reassigned).
    pub left_leaf_index: i32,
    /// The right child leaf id (the appended leaf) — consumed by the partition.
    pub right_leaf_index: i32,
}

/// The device-resident flat `CUDATree` SoA field buffers (allocated once).
struct TreeBuffers {
    leaf_value: Handle,
    leaf_weight: Handle,
    leaf_count: Handle,
    leaf_parent: Handle,
    leaf_depth: Handle,
    left_child: Handle,
    right_child: Handle,
    split_feature_inner: Handle,
    split_feature: Handle,
    split_gain: Handle,
    internal_weight: Handle,
    internal_value: Handle,
    internal_count: Handle,
    decision_type: Handle,
    threshold_in_bin: Handle,
    threshold: Handle,
    cat_boundaries: Handle,
    cat_boundaries_inner: Handle,
}

/// The number of device field buffers allocated by [`DeviceCudaTree::new`] — 16
/// leaf/node arrays + the two categorical boundary slabs.
pub const NUM_TREE_BUFFERS: usize = 18;

/// A CubeCL-safe, host pre-allocated flat device `CUDATree` (D-07) with **no
/// per-split device allocation** (D-15). [`Self::split_on_device`] /
/// [`Self::split_categorical_on_device`] mutate the pre-allocated handles via the
/// scalar fan-out kernels; [`Self::shrink`] / [`Self::add_bias`] apply the
/// elementwise leaf-value math; [`Self::to_host_tree`] reconstructs the host
/// `lgbm_model::Tree` for the anchor compare.
pub struct DeviceCudaTree<R: cubecl::Runtime> {
    device: TreeBuffers,
    /// Max leaves the tree is sized for.
    max_leaves: usize,
    /// Current leaf count (grows with each split; starts at 1).
    num_leaves: i32,
    /// Current categorical-split count (grows with each categorical split).
    num_cat: i32,
    /// Host-tracked serialized real-category bitset (`cat_threshold_`, appended per
    /// categorical split — the device slabs hold the inner bitset only).
    cat_threshold_host: Vec<u32>,
    /// Count of `client.empty` device allocations — equals [`NUM_TREE_BUFFERS`].
    device_allocations: usize,
    _runtime: PhantomData<R>,
}

impl<R: cubecl::Runtime> DeviceCudaTree<R> {
    /// Allocate the flat tree for `max_leaves` leaves and initialize it to a fresh
    /// single-leaf root (`leaf 0`, value 0, count `root_count`, parent -1). One
    /// counted `client.empty` per field, **exactly once** (D-15) — the SINGLE
    /// `client.empty` site in the module.
    ///
    /// # Errors
    /// [`ComputeError::Runtime`] if `max_leaves < 1` or the cat-boundary slab length
    /// overflows.
    pub fn new(
        client: &ComputeClient<R>,
        max_leaves: usize,
        root_count: i32,
    ) -> Result<Self, ComputeError> {
        if max_leaves == 0 {
            return Err(ComputeError::Runtime {
                detail: "DeviceCudaTree::new: max_leaves must be >= 1".to_string(),
            });
        }
        // Node arrays are indexed `[0, max_leaves-1)`; allocate `max_leaves` for a
        // uniform init sweep (one slot slack, harmless). cat_boundaries need
        // `num_cat + 1` cells — sized `max_leaves + 1` (a categorical split per leaf
        // at most).
        let n = max_leaves;
        let cat_len = max_leaves.checked_add(1).ok_or_else(|| ComputeError::Runtime {
            detail: "DeviceCudaTree::new: cat-boundary slab length overflows".to_string(),
        })?;

        // The ONLY `client.empty` site in this module (D-15): one alloc per field,
        // counted so "allocated exactly once" is structurally verifiable.
        let mut device_allocations = 0usize;
        let mut alloc = |elem_size: usize, len: usize| -> Handle {
            device_allocations += 1;
            client.empty(len * elem_size)
        };
        let sz_f64 = core::mem::size_of::<f64>();
        let sz_f32 = core::mem::size_of::<f32>();
        let sz_i32 = core::mem::size_of::<i32>();
        let sz_u32 = core::mem::size_of::<u32>();

        let device = TreeBuffers {
            leaf_value: alloc(sz_f64, n),
            leaf_weight: alloc(sz_f64, n),
            leaf_count: alloc(sz_i32, n),
            leaf_parent: alloc(sz_i32, n),
            leaf_depth: alloc(sz_i32, n),
            left_child: alloc(sz_i32, n),
            right_child: alloc(sz_i32, n),
            split_feature_inner: alloc(sz_i32, n),
            split_feature: alloc(sz_i32, n),
            split_gain: alloc(sz_f32, n),
            internal_weight: alloc(sz_f64, n),
            internal_value: alloc(sz_f64, n),
            internal_count: alloc(sz_i32, n),
            decision_type: alloc(sz_i32, n),
            threshold_in_bin: alloc(sz_u32, n),
            threshold: alloc(sz_f64, n),
            cat_boundaries: alloc(sz_i32, cat_len),
            cat_boundaries_inner: alloc(sz_i32, cat_len),
        };

        let tree = DeviceCudaTree {
            device,
            max_leaves,
            num_leaves: 1,
            num_cat: 0,
            cat_threshold_host: Vec::new(),
            device_allocations,
            _runtime: PhantomData,
        };
        tree.launch_init(client, root_count);
        Ok(tree)
    }

    /// Number of device buffers allocated (== [`NUM_TREE_BUFFERS`]; proves "allocated
    /// exactly once": no per-split alloc, D-15).
    #[must_use]
    pub fn device_allocations(&self) -> usize {
        self.device_allocations
    }

    /// Current leaf count.
    #[must_use]
    pub fn num_leaves(&self) -> i32 {
        self.num_leaves
    }

    /// Current categorical-split count.
    #[must_use]
    pub fn num_cat(&self) -> i32 {
        self.num_cat
    }

    fn launch_init(&self, client: &ComputeClient<R>, root_count: i32) {
        let n = self.max_leaves;
        let blocks = (n as u32).div_ceil(LEAF_OP_BLOCK);
        let d = &self.device;
        // SAFETY: every array is sized `max_leaves`; the kernel guards `i < n`, so
        // all writes stay in-bounds. cubecl unsafe confined here (CMP-01).
        unsafe {
            init_tree_kernel::launch(
                client,
                CubeCount::Static(blocks, 1, 1),
                CubeDim::new_1d(LEAF_OP_BLOCK.min(n as u32).max(1)),
                ArrayArg::from_raw_parts(d.leaf_value.clone(), n),
                ArrayArg::from_raw_parts(d.leaf_weight.clone(), n),
                ArrayArg::from_raw_parts(d.leaf_count.clone(), n),
                ArrayArg::from_raw_parts(d.leaf_parent.clone(), n),
                ArrayArg::from_raw_parts(d.leaf_depth.clone(), n),
                ArrayArg::from_raw_parts(d.left_child.clone(), n),
                ArrayArg::from_raw_parts(d.right_child.clone(), n),
                ArrayArg::from_raw_parts(d.split_feature_inner.clone(), n),
                ArrayArg::from_raw_parts(d.split_feature.clone(), n),
                ArrayArg::from_raw_parts(d.internal_count.clone(), n),
                ArrayArg::from_raw_parts(d.decision_type.clone(), n),
                ArrayArg::from_raw_parts(d.threshold_in_bin.clone(), n),
                root_count,
                n as u32,
            );
        }
    }

    /// Grow `leaf_index` into an internal NUMERICAL node with two child leaves via
    /// [`split_kernel`] (`cuda_tree.cu:48`). Returns the [`SplitResult`] whose
    /// `right_leaf_index` the partition step consumes (§1/§10 ordering, Pitfall 3):
    /// this MUST run BEFORE the partition.
    ///
    /// # Errors
    /// [`ComputeError::Runtime`] if `leaf_index` is out of range or the split would
    /// exceed `max_leaves`.
    #[allow(clippy::too_many_arguments)]
    pub fn split_on_device(
        &mut self,
        client: &ComputeClient<R>,
        leaf_index: i32,
        real_feature_index: i32,
        real_threshold: f64,
        missing_type: i32,
        scalars: &SplitScalars,
    ) -> Result<SplitResult, ComputeError> {
        self.check_split(leaf_index)?;
        let num_leaves = self.num_leaves;
        let new_node_index = num_leaves - 1;
        let right_leaf_index = num_leaves;
        let d = &self.device;
        let n = self.max_leaves;
        // SAFETY: `new_node_index` and `right_leaf_index` are `< max_leaves` (checked
        // by `check_split`); every array is sized `max_leaves`. The 15-thread fan-out
        // writes disjoint cells. cubecl unsafe confined here (CMP-01).
        unsafe {
            split_kernel::launch(
                client,
                CubeCount::Static(1, 1, 1),
                CubeDim::new_1d(15),
                ArrayArg::from_raw_parts(d.leaf_value.clone(), n),
                ArrayArg::from_raw_parts(d.leaf_weight.clone(), n),
                ArrayArg::from_raw_parts(d.leaf_count.clone(), n),
                ArrayArg::from_raw_parts(d.leaf_parent.clone(), n),
                ArrayArg::from_raw_parts(d.leaf_depth.clone(), n),
                ArrayArg::from_raw_parts(d.left_child.clone(), n),
                ArrayArg::from_raw_parts(d.right_child.clone(), n),
                ArrayArg::from_raw_parts(d.split_feature_inner.clone(), n),
                ArrayArg::from_raw_parts(d.split_feature.clone(), n),
                ArrayArg::from_raw_parts(d.split_gain.clone(), n),
                ArrayArg::from_raw_parts(d.internal_weight.clone(), n),
                ArrayArg::from_raw_parts(d.internal_value.clone(), n),
                ArrayArg::from_raw_parts(d.internal_count.clone(), n),
                ArrayArg::from_raw_parts(d.decision_type.clone(), n),
                ArrayArg::from_raw_parts(d.threshold_in_bin.clone(), n),
                ArrayArg::from_raw_parts(d.threshold.clone(), n),
                leaf_index,
                num_leaves,
                real_feature_index,
                scalars.inner_feature_index,
                real_threshold,
                scalars.threshold,
                scalars.gain,
                u32::from(scalars.default_left),
                missing_type,
                scalars.left_sum_hessians,
                scalars.right_sum_hessians,
                scalars.left_value,
                scalars.right_value,
                scalars.left_count,
                scalars.right_count,
            );
        }
        self.num_leaves += 1;
        Ok(SplitResult {
            new_node_index,
            left_leaf_index: leaf_index,
            right_leaf_index,
        })
    }

    /// Grow `leaf_index` into an internal CATEGORICAL node via
    /// [`split_categorical_kernel`] (`cuda_tree.cu:160`). `bitset` is the REAL
    /// category membership bitset (`cat_threshold_`), `bitset_inner` the inner (bin)
    /// bitset used for routing; their lengths append to `cat_boundaries` /
    /// `cat_boundaries_inner`. Returns the [`SplitResult`] (same ordering invariant).
    ///
    /// # Errors
    /// [`ComputeError::Runtime`] if `leaf_index` is out of range or the split would
    /// exceed `max_leaves`.
    #[allow(clippy::too_many_arguments)]
    pub fn split_categorical_on_device(
        &mut self,
        client: &ComputeClient<R>,
        leaf_index: i32,
        real_feature_index: i32,
        missing_type: i32,
        scalars: &SplitScalars,
        bitset: &[u32],
        bitset_inner: &[u32],
    ) -> Result<SplitResult, ComputeError> {
        self.check_split(leaf_index)?;
        let num_leaves = self.num_leaves;
        let num_cat = self.num_cat;
        let new_node_index = num_leaves - 1;
        let right_leaf_index = num_leaves;
        let d = &self.device;
        let n = self.max_leaves;
        // SAFETY: as in `split_on_device`; `num_cat + 1 <= max_leaves + 1` = the
        // cat-boundary slab length. cubecl unsafe confined here (CMP-01).
        unsafe {
            split_categorical_kernel::launch(
                client,
                CubeCount::Static(1, 1, 1),
                CubeDim::new_1d(17),
                ArrayArg::from_raw_parts(d.leaf_value.clone(), n),
                ArrayArg::from_raw_parts(d.leaf_weight.clone(), n),
                ArrayArg::from_raw_parts(d.leaf_count.clone(), n),
                ArrayArg::from_raw_parts(d.leaf_parent.clone(), n),
                ArrayArg::from_raw_parts(d.leaf_depth.clone(), n),
                ArrayArg::from_raw_parts(d.left_child.clone(), n),
                ArrayArg::from_raw_parts(d.right_child.clone(), n),
                ArrayArg::from_raw_parts(d.split_feature_inner.clone(), n),
                ArrayArg::from_raw_parts(d.split_feature.clone(), n),
                ArrayArg::from_raw_parts(d.split_gain.clone(), n),
                ArrayArg::from_raw_parts(d.internal_weight.clone(), n),
                ArrayArg::from_raw_parts(d.internal_value.clone(), n),
                ArrayArg::from_raw_parts(d.internal_count.clone(), n),
                ArrayArg::from_raw_parts(d.decision_type.clone(), n),
                ArrayArg::from_raw_parts(d.threshold_in_bin.clone(), n),
                ArrayArg::from_raw_parts(d.threshold.clone(), n),
                ArrayArg::from_raw_parts(d.cat_boundaries.clone(), n + 1),
                ArrayArg::from_raw_parts(d.cat_boundaries_inner.clone(), n + 1),
                leaf_index,
                num_leaves,
                real_feature_index,
                scalars.inner_feature_index,
                scalars.gain,
                missing_type,
                scalars.left_sum_hessians,
                scalars.right_sum_hessians,
                scalars.left_value,
                scalars.right_value,
                scalars.left_count,
                scalars.right_count,
                num_cat,
                bitset.len() as i32,
                bitset_inner.len() as i32,
            );
        }
        self.num_leaves += 1;
        self.num_cat += 1;
        self.cat_threshold_host.extend_from_slice(bitset);
        Ok(SplitResult {
            new_node_index,
            left_leaf_index: leaf_index,
            right_leaf_index,
        })
    }

    /// Apply `leaf_value *= rate` to the live leaves in place (ShrinkageKernel).
    pub fn shrink(&self, client: &ComputeClient<R>, rate: f64) {
        self.launch_leaf_op(client, rate, false);
    }

    /// Apply `leaf_value += val` to the live leaves in place (AddBiasKernel).
    pub fn add_bias(&self, client: &ComputeClient<R>, val: f64) {
        self.launch_leaf_op(client, val, true);
    }

    fn launch_leaf_op(&self, client: &ComputeClient<R>, scalar: f64, is_add: bool) {
        let n = self.num_leaves as usize;
        let blocks = (n as u32).div_ceil(LEAF_OP_BLOCK);
        let h = self.device.leaf_value.clone();
        // SAFETY: `leaf_value` is sized `max_leaves >= num_leaves`; the kernel guards
        // `i < num_leaves`. cubecl unsafe confined here (CMP-01).
        unsafe {
            if is_add {
                add_bias_kernel::launch(
                    client,
                    CubeCount::Static(blocks, 1, 1),
                    CubeDim::new_1d(LEAF_OP_BLOCK.min(n as u32).max(1)),
                    ArrayArg::from_raw_parts(h, self.max_leaves),
                    scalar,
                    n as u32,
                );
            } else {
                shrinkage_kernel::launch(
                    client,
                    CubeCount::Static(blocks, 1, 1),
                    CubeDim::new_1d(LEAF_OP_BLOCK.min(n as u32).max(1)),
                    ArrayArg::from_raw_parts(h, self.max_leaves),
                    scalar,
                    n as u32,
                );
            }
        }
    }

    fn check_split(&self, leaf_index: i32) -> Result<(), ComputeError> {
        if leaf_index < 0 || leaf_index >= self.num_leaves {
            return Err(ComputeError::Runtime {
                detail: format!(
                    "DeviceCudaTree::split: leaf_index {leaf_index} out of range \
                     (num_leaves = {})",
                    self.num_leaves
                ),
            });
        }
        if (self.num_leaves as usize) + 1 > self.max_leaves {
            return Err(ComputeError::Runtime {
                detail: format!(
                    "DeviceCudaTree::split: splitting would exceed max_leaves = {}",
                    self.max_leaves
                ),
            });
        }
        Ok(())
    }

    /// Reconstruct the host `lgbm_model::Tree` from the flat device arrays (D-07) —
    /// the cpu-f64-anchor compare target. Reads back each buffer, slicing leaf arrays
    /// to `[0, num_leaves)` and node arrays to `[0, num_leaves-1)`.
    #[must_use]
    pub fn to_host_tree(&self, client: &ComputeClient<R>) -> lgbm_model::Tree {
        let nl = self.num_leaves as usize;
        let nn = nl - 1;
        let d = &self.device;

        let read_f64 = |h: &Handle, len: usize| -> Vec<f64> {
            let bytes = client.read_one_unchecked(h.clone());
            f64::from_bytes(&bytes)[..len].to_vec()
        };
        let read_f32 = |h: &Handle, len: usize| -> Vec<f32> {
            let bytes = client.read_one_unchecked(h.clone());
            f32::from_bytes(&bytes)[..len].to_vec()
        };
        let read_i32 = |h: &Handle, len: usize| -> Vec<i32> {
            let bytes = client.read_one_unchecked(h.clone());
            i32::from_bytes(&bytes)[..len].to_vec()
        };
        let read_u32 = |h: &Handle, len: usize| -> Vec<u32> {
            let bytes = client.read_one_unchecked(h.clone());
            u32::from_bytes(&bytes)[..len].to_vec()
        };

        let decision_type_i32 = read_i32(&d.decision_type, nn);
        let decision_type: Vec<i8> = decision_type_i32.iter().map(|&x| x as i8).collect();

        let num_cat = self.num_cat;
        let cat_boundaries = if num_cat > 0 {
            read_i32(&d.cat_boundaries, (num_cat as usize) + 1)
        } else {
            Vec::new()
        };

        lgbm_model::Tree {
            num_leaves: self.num_leaves,
            num_cat,
            left_child: read_i32(&d.left_child, nn),
            right_child: read_i32(&d.right_child, nn),
            split_feature: read_i32(&d.split_feature, nn),
            threshold: read_f64(&d.threshold, nn),
            decision_type,
            split_gain: read_f32(&d.split_gain, nn),
            leaf_value: read_f64(&d.leaf_value, nl),
            leaf_weight: read_f64(&d.leaf_weight, nl),
            leaf_count: read_i32(&d.leaf_count, nl),
            internal_value: read_f64(&d.internal_value, nn),
            internal_weight: read_f64(&d.internal_weight, nn),
            internal_count: read_i32(&d.internal_count, nn),
            cat_boundaries,
            cat_threshold: self.cat_threshold_host.clone(),
            shrinkage: 1.0,
            is_linear: false,
            leaf_depth: read_i32(&d.leaf_depth, nl),
            leaf_parent: read_i32(&d.leaf_parent, nl),
            split_feature_inner: read_i32(&d.split_feature_inner, nn),
            threshold_in_bin: read_u32(&d.threshold_in_bin, nn),
        }
    }
}

#[cfg(test)]
mod shrinkage {
    use super::*;
    use crate::runtime::cpu_client;

    /// Shrinkage (`leaf_value *= rate`) matches a serial f64 reference element-for-
    /// element on the cpu f64 anchor (D-12).
    #[test]
    fn shrinkage_matches_serial_reference() {
        let client = cpu_client();
        let values = vec![1.5f64, -2.25, 0.0, 3.75, -0.125, 100.0, -1e-9];
        let rate = 0.1f64;
        let got = apply_shrinkage_on(&client, &values, rate).expect("shrinkage");
        let want: Vec<f64> = values.iter().map(|&v| v * rate).collect();
        assert_eq!(got.len(), want.len());
        for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
            assert_eq!(g.to_bits(), w.to_bits(), "shrinkage leaf {i}: {g} != {w}");
        }
    }

    /// AddBias (`leaf_value += val`) matches a serial f64 reference element-for-
    /// element on the cpu f64 anchor (D-12).
    #[test]
    fn add_bias_matches_serial_reference() {
        let client = cpu_client();
        let values = vec![1.5f64, -2.25, 0.0, 3.75, -0.125, 100.0, -1e-9];
        let bias = 0.375f64;
        let got = apply_add_bias_on(&client, &values, bias).expect("add_bias");
        let want: Vec<f64> = values.iter().map(|&v| v + bias).collect();
        assert_eq!(got.len(), want.len());
        for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
            assert_eq!(g.to_bits(), w.to_bits(), "add_bias leaf {i}: {g} != {w}");
        }
    }
}
