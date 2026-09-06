//! `#[cube]` compute kernels (CMP-05).
//!
//! Plan 04-01 shipped the minimal `construct_histograms` kernel that settled the
//! D-04a bit-determinism bet; 04-02 finalized it; 04-03 (this plan) adds
//! `find_best_split` (gain math in-kernel, D-01a), `data_partition` (stable
//! row->{left,right} routing), and `subtract_histograms` (the kernel-layer
//! histogram-subtraction math, A3 resolved in-scope).

/// Shared CubeCL-autotune plumbing (phase 13) for the GPU launch-config knobs.
/// gpu-gated (quick-260627-qxl widened from rocm): the default cpu build pulls no
/// autotune codegen (the f64 anchor is never autotuned), but every GPU backend
/// (rocm/cuda/wgpu) reuses the runtime-generic `cubecl::tune` plumbing.
#[cfg(feature = "gpu")]
pub mod autotune;

/// Shared front end for the `LGBM_*` launch-config env knobs (`LGBM_BUILD_CUBEDIM`,
/// `LGBM_AUTOTUNE_FORCE_P`, `LGBM_SCAN_CUBEDIM`, `LGBM_ROWPART_TARGET_CUBES`, ...):
/// read the var, parse as `u32`, keep only strictly-positive values. Every knob
/// layers its own clamp/default on top (the "positive" contract is the only part
/// they all share) so parse failures, `0`, and unset all fall through the same way
/// — NEVER a no-launch, only the caller's documented fallback.
#[cfg(feature = "gpu")]
pub(crate) fn parse_positive_u32_env(name: &str) -> Option<u32> {
    std::env::var(name)
        .ok()
        .and_then(|s| s.parse::<u32>().ok())
        .filter(|&w| w > 0)
}

pub mod histogram;
pub mod partition;
// Phase-14 foundation (additive, behind the OFF-by-default `LGBM_CUDA_ON_DEVICE`
// seam): shared device primitives (14-03/14-05), the SoA pre-allocated device
// split-record (14-04), and the `CUDARandom` LCG (14-04). Ungated like the other
// kernel modules (NOT `#[cfg(feature = "gpu")]` like `autotune`) so Wave-2 plans
// can each fill exactly one owned file with no `mod.rs` contention; they land as
// empty compiling stubs.
pub mod primitives;
pub mod random;
pub mod split;
pub mod split_info;
pub mod subtract;
// Phase-15 device-dataset + row-subset-gather modules (ODL-03/04), additive and
// behind the SAME OFF-by-default `LGBM_CUDA_ON_DEVICE` seam: the device columnar
// store (`column_data`), the row-wise binned matrix + feature-partition layout
// (`row_data`), and the `CopySubrow` bagging-subset gather (`copy_subrow`).
// Ungated like `random`/`split` (NOT `#[cfg(feature = "gpu")]`) so the default cpu
// f64 anchor runs them (D-08) and each Wave-1 plan fills exactly one owned file
// with no `mod.rs` contention.
pub mod column_data;
pub mod copy_subrow;
pub mod row_data;
// Phase-16 histogram-constructor support: the pre-allocated-once histogram arena +
// the `{parent, smaller, larger}` hist_t** handle-rotation contract (ODL-10 / D-02 /
// D-09). Ungated like the other Phase-14/15 kernel modules (NOT
// `#[cfg(feature = "gpu")]`) so the default cpu f64 anchor exercises the index
// bookkeeping in isolation (D-08); the whole-tree pool SWAP is deferred to Phase 18.
pub mod histogram_arena;
// Phase-17 best-split finder (ODL-11 / ODL-12): the 3-stage split-finding pipeline
// (§8) — stage-1 per-task eval / stage-2 cross-feature reduce / stage-3 cross-leaf
// argmax + the single 8-int readback (D-06). Additive and OFF by default behind
// `LGBM_CUDA_ON_DEVICE` (D-09). Ungated like the other Phase-14/15/16 kernel modules
// (NOT `#[cfg(feature = "gpu")]`) so the default cpu f64 anchor exercises the
// host task-builder + count-recovery rounding in isolation (D-08).
pub mod best_split;
// Phase-18 on-device data-partition / tree-mutation / prediction (ODL-13/14/15):
// the §9 `mark → prefix-sum → scatter` row router (`data_partition`), the flat
// device `CUDATree` mutation kernels (`tree`), and the tree-walk prediction
// kernel (`predict`). Declared here in Wave-0 (18-01) as empty compiling stubs so
// each Wave-1/2 plan (18-02/03/04) fills exactly one owned file with NO `mod.rs`
// contention. Additive and OFF by default behind `LGBM_CUDA_ON_DEVICE` (D-13);
// ungated like the other Phase-14/15/16/17 kernel modules (NOT
// `#[cfg(feature = "gpu")]`) so the default cpu f64 anchor exercises them (D-08).
pub mod data_partition;
pub mod predict;
pub mod tree;
// Phase-19 on-device objectives (ODL-05/06/07/08): the §5 grad/hess + inverse-link
// kernels — regression (§5.1), binary (§5.2), multiclass (§5.3), rank (§5.4).
// Declared here in Wave-1 (19-00) as empty compiling stubs so each Wave-2 family
// plan (19-01/02/03/04) fills exactly one owned file with NO `mod.rs` contention.
// Additive and OFF by default behind `LGBM_CUDA_ON_DEVICE` (D-06); ungated like the
// other Phase-14..18 kernel modules (NOT `#[cfg(feature = "gpu")]`, per D-08) so the
// default cpu f64 anchor exercises them.
pub mod objective_binary;
pub mod objective_multiclass;
pub mod objective_rank;
pub mod objective_regression;
// Phase-20 score-updater (§11, ODL-16) + pointwise-metric (§12, ODL-17) kernels.
// Declared here in Wave-0 (20-00) as empty compiling stubs so each Wave-2 plan
// (20-01 score updater / 20-02 metric evaluator) fills exactly one owned file with
// NO `mod.rs` contention. Additive and OFF by default behind `LGBM_CUDA_ON_DEVICE`;
// ungated like the other Phase-14..19 kernel modules (NOT `#[cfg(feature = "gpu")]`,
// per D-08) so the default cpu f64 anchor exercises them.
pub mod metric_pointwise;
pub mod score_updater;
// Phase-20 on-device tree-growth driver seam (ODL-18/ODL-19): the additive
// `GrowFeature` feature/bin metadata carrier the per-leaf grow loop consumes,
// expressed in ONLY lgbm-compute-reachable types (BinColumn + lgbm-dataset
// BinType/MissingType + primitives — NEVER an lgbm-treelearner type, which would
// form the treelearner→compute→treelearner crate cycle this replan avoids).
// The `grow_tree_on_device` driver BODY stays `Ok(None)` (20-03a plumbing slice);
// the per-leaf orchestration lands in 20-03b. Ungated like the other Phase-14..19
// kernel modules (NOT `#[cfg(feature = "gpu")]`) so the default cpu f64 anchor
// exercises the plumbing (D-08).
pub mod grow_driver;
// Phase-22 on-device categorical splits (ODL-22): the §6.3 bitset construction
// (`construct_bitset` + `set_real_threshold` two-bitset producer) and the §8.1
// categorical split evaluator (one-hot + many-vs-many, D-02) — a byte-for-byte
// transcription of the golden-proven host `feature_histogram_categorical`
// (`lgbm-treelearner`, which CANNOT be imported: crate cycle, memory
// `on-device-driver-crate-cycle-constraint`). Single-owner `CubeDim::new_1d(1)`
// f64 anchor (def-f8u-01); the many-vs-many ctr sort reuses
// `primitives::bitonic_argsort_on`. Ungated like the other Phase-14..21 kernel
// modules (NOT `#[cfg(feature = "gpu")]`) so the default cpu f64 anchor exercises
// it. This module produces the split decision + winner bitsets only; wiring into
// the grow driver is 22-04.
pub mod categorical_split;
