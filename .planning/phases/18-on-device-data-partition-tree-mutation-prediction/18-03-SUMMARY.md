---
phase: 18-on-device-data-partition-tree-mutation-prediction
plan: 03
subsystem: compute
tags: [cubecl, cuda-tree, split-kernel, shrinkage, add-bias, categorical, tree-mutation, ODL-14]

# Dependency graph
requires:
  - phase: 18-on-device-data-partition-tree-mutation-prediction
    plan: 01
    provides: tree.rs Wave-0 stub, tree_mutation_parity.rs #[ignore] scaffold, split.txt SCASE/SWIN golden
  - phase: 14-foundation-shared-device-primitives-device-structs-rng
    provides: split_info.rs SplitScalars (CUDASplitInfo field list) + DeviceSplitInfo counted-alloc SoA idiom
  - phase: 18-on-device-data-partition-tree-mutation-prediction
    plan: 02
    provides: data_partition.rs update_data_index_to_leaf_on (the partition consumer of right_leaf_index)
provides:
  - tree.rs device flat CUDATree (DeviceCudaTree) SoA + SplitKernel/SplitCategoricalKernel/ShrinkageKernel/AddBiasKernel + set/get decision-type bit packers + host lgbm_model::Tree reconstruction
  - tree_mutation_parity.rs numeric field-write + Split-before-partition ordering + categorical field-write cells un-ignored, driven live vs the golden + host mirror
  - tree::shrinkage unit (shrinkage *= rate / add_bias += val vs a serial f64 reference)
affects: [18-04]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Device flat CUDATree = counted client.empty SoA (one Handle per field, allocated once, D-15) mirroring DeviceSplitInfo::new; init via a one-shot init_tree_kernel (the InitCUDAMemory analog), never a per-split alloc"
    - "SplitKernel as an ABSOLUTE_POS 15-thread fan-out (each thread writes disjoint cells, no read-after-write hazard — faithful to the reference <<<3,5>>>); ~x child encoding written as -x-1 (no cube bitwise-not dependency)"
    - "NaN→0 leaf-output coercion via branchless select(f64::is_nan(x), 0.0, x) — cubecl `x != x` compiles to false on cubecl-cpu, so f64::is_nan is the portable NaN test"
    - "SP-1 shared elementwise body (leaf_value_op) + thin shrinkage/add_bias launch wrappers; f64 scalar-per-leaf, no f64 per-row loop (D-14)"
    - "cpu f64 anchor = a plain-Rust host mirror of the SplitKernel field writes producing the SAME lgbm_model::Tree (never GPU-vs-GPU, D-12)"

key-files:
  created: []
  modified:
    - crates/lgbm-compute/src/kernels/tree.rs
    - crates/oracle-harness/tests/tree_mutation_parity.rs

key-decisions:
  - "NaN→0 uses cubecl f64::is_nan inside the select (NOT the C-idiomatic x != x): cubecl-cpu 0.10 lowers x != x to false for NaN, so the reference isnan(x)?0.0f:x maps to select(f64::is_nan(x), 0.0, x). Bit-exact NaN→0 confirmed on the categorical right-leaf NaN case."
  - "The device flat CUDATree is initialized by a one-shot init_tree_kernel (leaf_parent=-1, depth/value/weight=0, leaf 0 count=root_count) rather than uploading init vectors — keeps the SINGLE client.empty site (D-15) and is faithful to the reference InitCUDAMemory."
  - "The categorical field-write cell is anchored to a plain-Rust host mirror (cpu f64 anchor, D-12), not a captured tree golden: split.txt is numeric-only, and no categorical tree-mutation golden exists this phase. The mirror reproduces the SplitCategoricalKernel fan-out exactly; the numeric cell IS golden-anchored (split.txt SWIN)."
  - "SplitKernel is a 15-thread ABSOLUTE_POS fan-out (faithful to the reference <<<3,5>>>) rather than a single-owner writer — honors threat T-18-04's 'scalar fan-out matches the reference thread counts' mitigation; every thread writes disjoint cells so the result is order-independent."

patterns-established:
  - "Flat device tree SoA + one-shot init kernel + per-op scalar fan-out launch, the DeviceCudaTree template the 18-04 predict walk and the future growth-loop driver reuse"
  - "Host mirror of a device scalar/structural kernel as the cpu f64 anchor when no captured golden exists for the mutated structure"

requirements-completed: [ODL-14]

# Metrics
duration: 40min
completed: 2026-07-01
status: complete
---

# Phase 18 Plan 03: On-Device Tree Mutation (ODL-14) Summary

**The §10 device-resident flat `CUDATree` (SoA, pre-allocated once) and its scalar/elementwise mutation kernels — `SplitKernel` (14 field writes from `SplitScalars`, NaN→0, returns `right_leaf_index` BEFORE the partition consumes it), `SplitCategoricalKernel` (kCategoricalMask + num_cat + cat_boundaries append), and the elementwise `ShrinkageKernel`/`AddBiasKernel` — all anchor-pinned bit-exact to the `split.txt` golden and a plain-Rust host mirror, with the merge gate green (882 passed, `LGBM_CUDA_ON_DEVICE` unset).**

## Performance
- **Duration:** ~40 min
- **Completed:** 2026-07-01
- **Tasks:** 2
- **Files modified:** 2

## Accomplishments
- **Device flat CUDATree (D-07 / D-15):** `DeviceCudaTree<R>` models the reference `CUDATree` as 18 counted-`client.empty` SoA handles (`cuda_leaf_value_`, `cuda_left_child_`/`cuda_right_child_`, `cuda_decision_type_`, thresholds/counts/depth, `cat_boundaries`/`cat_boundaries_inner`), allocated **once** in `new` (the SINGLE executable `client.empty` site) and initialized to a fresh single-leaf root by a one-shot `init_tree_kernel` (the `InitCUDAMemory` analog). `device_allocations()` proves the "allocated exactly once" invariant.
- **SplitKernel (Task 1, ODL-14):** the `<<<3,5>>>` 15-thread `ABSOLUTE_POS` fan-out that writes the 14 numeric tree fields from a `SplitScalars` record, rewires the parent/child links (`~x` encoded as `-x-1`), advances node/leaf/depth bookkeeping, and coerces NaN leaf outputs to `0.0` via branchless `select(f64::is_nan(x), 0.0, x)`. `split_on_device` returns the `SplitResult { new_node_index, left_leaf_index, right_leaf_index }` — the `right_leaf_index` the partition step consumes (§1/§10 hard ordering invariant, Pitfall 3): Split runs BEFORE partition.
- **SplitCategoricalKernel + Shrinkage + AddBias (Task 2, ODL-14):** the 17-thread categorical fan-out reuses the Task-1 bookkeeping plus sets `kCategoricalMask` (default_left NOT encoded), writes `num_cat` as the threshold, and appends the bitset lengths to `cat_boundaries`/`cat_boundaries_inner` (seeded 0 at `num_cat==0`). `ShrinkageKernel` (`leaf_value *= rate`) / `AddBiasKernel` (`leaf_value += val`) share one elementwise `#[cube]` body `leaf_value_op` behind thin launch wrappers (SP-1); scalar math is f64, no f64 per-row loop (D-14).
- **Bit packers (SP-2):** `set_decision_type` (`input ? dt|mask : dt&(127-mask)`) and `set_missing_type` (`(dt&3)|(missing<<2)`) as branchless `select` `#[cube]` helpers + host `get_decision_type`/`get_missing_type` mirrors.
- **Host reconstruction (D-07):** `to_host_tree` reads the flat arrays back into an `lgbm_model::Tree` (leaf arrays `[0,num_leaves)`, node arrays `[0,num_leaves-1)`), the cpu-f64-anchor compare target.
- **Parity green:** `tree_mutation_parity` un-ignored — the numeric field-write cell drives `split_on_device` per `split.txt` `SWIN` winner and asserts every mutated field (leaf outputs NaN→0, per-side hessian weights, threshold_in_bin, counts, child links, split_gain f32, default_left/categorical bits, depth/parent) **bit-exact vs the golden AND vs a plain-Rust host mirror**; the ordering cell confirms `split_on_device` yields `right_leaf_index=1` and feeds it to `update_data_index_to_leaf_on` (the partition consumer); the categorical cell drives `split_categorical_on_device` (incl. a NaN right-leaf) vs the host mirror. `tree::shrinkage` unit checks shrinkage/add_bias element-for-element vs a serial f64 reference.

## Task Commits
1. **Tasks 1 + 2 (tree.rs + tree_mutation_parity.rs, ODL-14)** — `91ea2e0` (feat)

*(Tasks 1 and 2 landed in one atomic commit: both tasks add cohesive sections to the same two files — the numeric `SplitKernel`/infra and the categorical/`Shrinkage`/`AddBias` additions share the `tree.rs` `DeviceCudaTree` impl and the same parity test module, so they are inseparable at the file level. Every acceptance criterion for both tasks is individually verified green — see Verification.)*

**Plan metadata:** (this commit) (docs: complete plan)

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] NaN→0 select needed `f64::is_nan`, not the C `x != x` idiom**
- **Found during:** Task 2 (categorical cell with a NaN right-leaf output)
- **Issue:** The reference `isnan(x) ? 0.0f : x` was first transcribed as `select(x != x, 0.0, x)`. cubecl-cpu 0.10 lowers the `x != x` self-inequality to **false** for a NaN operand, so the select returned the NaN unchanged — the right leaf stayed NaN (`0x7FF8…` vs expected `0x0000…`). The numeric goldens are all finite, so this only surfaced on the first NaN input (the categorical cell).
- **Fix:** Switched both kernels' coercion to `select(f64::is_nan(x), 0.0, x)` — cubecl exposes `f64::is_nan` (`Comparison::IsNan`), which is the faithful `isnan()` mapping and lowers correctly on cubecl-cpu. Still branchless `select`, no nested-if.
- **Files modified:** crates/lgbm-compute/src/kernels/tree.rs
- **Verification:** categorical cell right-leaf NaN→0 now bit-exact (`0x0000000000000000`); all 3 parity cells + 2 shrinkage units green.
- **Committed in:** 91ea2e0

### Deliberate scoping (no user permission needed — Rule 3 class)

**2. [Scoping, D-12] Categorical field-write cell anchored to a host mirror, not a captured tree golden**
- **What:** `split.txt` is numeric-only (Phase-4 `SCASE`/`SWIN`), and no categorical tree-mutation golden exists this phase. The categorical cell therefore drives `split_categorical_on_device` and asserts the kCategoricalMask/num_cat/threshold/cat_boundaries-append fields vs a plain-Rust host mirror of the exact `SplitCategoricalKernel` fan-out (the cpu f64 anchor, D-12). The **numeric** cell is fully golden-anchored (`split.txt` `SWIN` bit-exact).

**Total deviations:** 1 auto-fixed bug, 1 deliberate scoping. No architectural changes (Rule 4 not triggered).

## Verification
- `cargo test -p oracle-harness --test tree_mutation_parity` — 3/3 (numeric field writes + Split-before-partition ordering + categorical) green vs the golden + host mirror.
- `cargo test -p lgbm-compute --lib tree::shrinkage` — 2/2 (shrinkage `*= rate` + add_bias `+= val`) bit-exact vs a serial f64 reference.
- `cargo test --workspace` — **882 passed, 0 failed** with `LGBM_CUDA_ON_DEVICE` unset (ODL-19 merge gate).
- clippy clean on `tree.rs` + `tree_mutation_parity.rs` (no new warnings).
- Grep invariants: single executable `client.empty` site for the flat CUDATree (D-15); `SplitScalars` READ (imported), not redefined; the tree kernels are scalar/elementwise (`ABSOLUTE_POS`, no `for` loop in any kernel body) with f64 confined to leaf-value/gain scalar math (D-14).
- `cargo build -p lgbm-compute --features rocm` — clean (the f64 kernels lower on the hip toolchain; the hip parity gate is exercised in 18-04).
- cpu f64 fold is the anchor throughout (never GPU-vs-GPU, D-12); no `LightGBM/` changes.

## Known Stubs
None — the numeric + categorical split kernels, the shrinkage/add-bias math, and the host reconstruction are fully wired to the goldens/host mirror; no placeholder/empty-data sinks. (The `SplitScalars` `left_sum_gradients`/`*_gh_quant`/`*_gain` fields set to 0 in the test vectors are legitimate unused-by-SplitKernel inputs — the kernel consumes only hessians/values/counts — not stubs.)

## Threat Flags
None — no new network/auth/file-access surface. T-18-04 (SplitKernel index writes) is mitigated by the pre-allocated `max_leaves`-bounded flat tree + `check_split` bounds validation + NaN→0 coercion + the faithful fan-out thread counts; T-18-05 (cat_boundaries append) is bounded by the `max_leaves+1` slab with no per-split alloc (D-15).

## Next Phase Readiness
- **18-04 (Wave 2, ODL-15):** `DeviceCudaTree` + `set/get_decision_type` are available for the predict tree-walk (`AddPredictionToScoreKernel<USE_INDICES>`); the shared `route_to_left`/`find_in_bitset` from 18-02 remain the single route source. `to_host_tree` gives the anchor-compare seam.
- ODL-14 on-device tree mutation is complete: `SplitKernel` writes the flat CUDATree structure bit-exact to the anchor and returns `right_leaf_index` before the partition consumes it; `SplitCategorical`/`Shrinkage`/`AddBias` are anchor-green; the merge gate is green with the env unset.

---
*Phase: 18-on-device-data-partition-tree-mutation-prediction*
*Completed: 2026-07-01*

## Self-Check: PASSED
Both modified files exist on disk + the SUMMARY; the task commit (91ea2e0) is present in git history.
