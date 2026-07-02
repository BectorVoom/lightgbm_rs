---
phase: 20-on-device-score-updater-metrics
plan: 03b
subsystem: treelearner
tags: [on-device, tree-learner, histogram, best-split, data-partition, cubecl, bit-exact, structure-gate, cpu-anchor]

# Dependency graph
requires:
  - phase: 20-03a
    provides: "GrowFeature carrier, gated CpuBackend/GpuBackend on_device_growth_supported() flip, LeafMapBufferStrategy (DOUBLE-BUFFER locked), 5-arg grow_tree_on_device seam"
  - phase: 16
    provides: "construct_histograms_f64_on build + subtract_histograms_f64_on"
  - phase: 17
    provides: "find_best_split_f64_on split finder + FeatureMeta"
  - phase: 18
    provides: "DeviceCudaTree::split_on_device / to_host_tree, partition_leaf_stable, update_data_index_to_leaf_on"
provides:
  - "grow_tree_on_device_driver: the per-leaf best-first on-device grow orchestration sequencing the Phase-16/17/18 kernels with the driver's OWN bookkeeping (no LeafSplits/HistogramPool/DataPartition — no crate cycle)"
  - "Activated CpuBackend + GpuBackend<R> grow_tree_on_device seam returning Ok(Some((Tree, LeafPartitionLayout))) when gated (LGBM_CUDA_ON_DEVICE=1)"
  - "learner_parity_on_device_structure_gate: non-vacuous default-cpu-build STRUCTURE bit-exact gate vs the cpu f64 anchor"
affects: [phase-21-on-device-growth, on-device-tree-learner]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "On-device grow driver composes existing golden kernels with lgbm-compute-local DriverLeaf state (no treelearner types crossing the seam)"
    - "STRUCTURE-bit-exact gate anchored ALWAYS to the cubecl-cpu f64 fold (never GPU-vs-GPU, def-f8u-01); leaf values within ROCM_LEAF_VALUE_TOL, default_left tie-aware"
    - "Trait seam carries no GainConfig — the driver pins a proving_slice_config() the anchor mirrors"

key-files:
  created: []
  modified:
    - "crates/lgbm-compute/src/kernels/grow_driver.rs — grow_tree_on_device_driver + DriverLeaf + build/scan/fix/compact helpers + proving_slice_config"
    - "crates/lgbm-compute/src/lib.rs — CpuBackend + GpuBackend<R> grow_tree_on_device wired to the driver (gated)"
    - "crates/oracle-harness/tests/learner_parity.rs — activated STRUCTURE gate; shared anchor+comparator hoisted to module scope; two rocm Slice-0 cells rewritten; host_grow retired; seam_defers env-aware"

key-decisions:
  - "Used find_best_split_f64_on (the f64 sibling of the CpuBackend anchor's find_best_split_cpu_native) rather than the best_split.rs stage1/2/3 CUDA-mirror — the mirror's round_ties_even count recovery diverges from the host anchor's round-half-up; the f64_on path is the bit-exact-to-anchor choice the load-bearing STRUCTURE gate needs"
  - "FixHistogram + compaction inlined as O(num_bin) host f64 folds (matching the reference host fold) rather than the device fix_compact_f64_on (which quantizes f32 raw → ≤1/2^30 error) — MORE bit-exact and ODL-19-compliant (not per-row)"
  - "Tree leaf_count/internal_count take the ACTUAL partition counts (serial_tree_learner.cpp:788-791), child sums seeded from SplitInfo (kEpsilon-carrying, no re-fold)"
  - "Hoisted the shared anchor+comparator from mod hip to module scope so the default gate and the rocm cells share one definition; cpu_anchor_tree gained an explicit cfg param"

patterns-established:
  - "DriverLeaf bookkeeping mirrors SerialTreeLearner's best-first order (root fold -> build smaller / subtract larger -> find-best -> split_on_device BEFORE partition -> seed children) using only lgbm-compute-reachable types"
  - "Proving slice: continuous features + L2 + MissingType::None; L1/quantile/categorical follow-up reuses the same ordering contract"

requirements-completed: [ODL-18, ODL-19]

# Metrics
duration: 55min
completed: 2026-07-02
status: complete
---

# Phase 20 Plan 03b: On-device grow driver + STRUCTURE bit-exact gate Summary

**A minimal per-leaf best-first on-device grow driver that sequences the Phase-16/17/18 kernels to grow a full continuous-feature+L2 tree STRUCTURE-bit-exact (leaf values 0.000e0 diff) to the cpu f64 anchor, with the grow_tree_on_device seam activated behind the LGBM_CUDA_ON_DEVICE gate.**

## Performance

- **Duration:** ~55 min
- **Completed:** 2026-07-02
- **Tasks:** 2
- **Files modified:** 3

## Accomplishments
- `grow_tree_on_device_driver<R>` grows an entire tree on device by composing `construct_histograms_f64_on` (build) + host FixHistogram/compact + `subtract_histograms_f64_on` (larger child) + `find_best_split_f64_on` (Phase-4/17 finder) + `DeviceCudaTree::split_on_device` (BEFORE partition) + `partition_leaf_stable`, in the `SerialTreeLearner` best-first order — using the driver's OWN `DriverLeaf` state (no `LeafSplits`/`HistogramPool`/`DataPartition`, no crate cycle).
- `CpuBackend` and `GpuBackend<R>` `grow_tree_on_device` return `Ok(Some((Tree, LeafPartitionLayout)))` when `cuda_on_device_enabled()`, `Ok(None)` otherwise (byte-unchanged merge gate).
- `learner_parity_on_device_structure_gate` (DEFAULT cpu build) is non-vacuous and STRUCTURE bit-exact to `cpu_anchor_tree`: **4 leaves match cpu f64 anchor (structure bit-exact, max leaf diff 0.000e0)**.
- Both `CpuBackend` (cubecl-cpu) AND `RocmBackend` (real hip hardware, env=1) grow STRUCTURE bit-exact to the SAME cpu f64 anchor.

## Task Commits

1. **Task 1: per-leaf best-first grow driver + wire grow_tree_on_device** - `bd05c5c` (feat)
2. **Task 2: activate STRUCTURE bit-exact gate + rewrite rocm Slice-0 cells** - `e7dbcf4` (test)

## Files Created/Modified
- `crates/lgbm-compute/src/kernels/grow_driver.rs` - `grow_tree_on_device_driver`, `DriverLeaf`, `build_leaf_hist`/`scan_leaf`/`fix_histogram`/`compact_histogram`, `proving_slice_config`, `split_gt`.
- `crates/lgbm-compute/src/lib.rs` - `CpuBackend`/`GpuBackend<R>` `grow_tree_on_device` gated on `cuda_on_device_enabled()` → driver → `Ok(Some(..))`.
- `crates/oracle-harness/tests/learner_parity.rs` - module-scope anchor+comparator; `learner_parity_on_device_structure_gate`; `on_device_proving_corpus`; two rocm Slice-0 cells rewritten to the activated contract; `host_grow` retired; `seam_defers` made env-aware; `cpu_anchor_tree` gained a cfg param.

## Verification Results

- `LGBM_CUDA_ON_DEVICE=1 cargo test -p oracle-harness --test learner_parity -- --exact learner_parity_on_device_structure_gate` → `test result: ok. 1 passed; 0 failed` (non-vacuous), `on-device: 4 leaves match cpu f64 anchor (structure bit-exact, max leaf diff 0.000e0)`.
- `cargo test --workspace` (env UNSET) → all suites green, byte-unchanged (the driver is unreachable when the env gate is off). `learner_parity` suite: `32 passed`.
- `grep -rn 'lgbm_treelearner' crates/lgbm-compute/src` → NONE (no treelearner dependency edge; no crate cycle).
- `cargo test -p oracle-harness --test learner_parity --features rocm --no-run` → compiles. The two rewritten Slice-0 cells pass env-unset (`Ok(None)` branch) AND env=1 (grow on real hip hardware, STRUCTURE bit-exact vs cpu f64 anchor).

## ODL-19 no-f64 review (reviewed driver kernel list)

The driver composes ONLY existing golden kernels — no NEW per-row build kernel was written:
- `construct_histograms_f64_on` (Phase-16 histogram build — the cpu f64 deterministic anchor build)
- `subtract_histograms_f64_on` (Phase-16 larger-child derivation, O(num_bin))
- `find_best_split_f64_on` (Phase-4/17 split finder)
- `partition_leaf_stable` (Phase-18 row route — integer bins, no f64)
- `DeviceCudaTree::split_on_device` / `to_host_tree` (Phase-18 tree mutation)

Finding: **No f64 per-row grow/build HOT loop was introduced.** The only per-row f64 accumulation is the ROOT ordered fold (the reference-blessed `LeafSplits::init` analog, run ONCE); every child leaf's sums are seeded from the parent split's `SplitInfo` (kEpsilon-carrying, NO re-fold). The per-feature FixHistogram + compaction are O(num_bin) scalar f64 folds (ascending bin order, reference-blessed), not per-row. The per-row host gathers (`rows.map(|r| gradients[r])`) build f32 slices for the device kernel (no f64 accumulation).

## Decisions Made
See `key-decisions` frontmatter. Principal: `find_best_split_f64_on` over the stage1/2/3 CUDA-mirror (bit-exactness to the anchor), and host f64 FixHistogram/compact over the f32-quantizing device fix_compact (more bit-exact, ODL-19-compliant).

## Deviations from Plan

### Auto-fixed / design choices

**1. [Rule 3 - Blocking] Split-finder kernel choice: find_best_split_f64_on, not stage1/2/3**
- **Found during:** Task 1
- **Issue:** The plan's read_first pointed at best_split.rs stage1/2/3 (the CUDA best-split-finder mirror). That mirror uses `round_ties_even` count recovery, which DIVERGES from the host anchor's `round_half_up` (`find_best_split_cpu_native`) — risking non-bit-exact structure at the load-bearing STRUCTURE gate.
- **Fix:** Composed `find_best_split_f64_on` (the f64 device sibling of the anchor's native split kernel) instead. This is the kernel-parity-proven, bit-exact-to-anchor Phase-4/17 split finder. Result: max leaf diff 0.000e0.
- **Files modified:** grow_driver.rs
- **Verification:** STRUCTURE gate bit-exact (1 passed).
- **Committed in:** `bd05c5c`

**2. [Rule 3 - Blocking] FixHistogram/compaction inlined as host f64, not device fix_compact_f64_on**
- **Found during:** Task 1
- **Issue:** `fix_compact_f64_on` consumes an f32 RAW buffer and re-quantizes to u64 fixed-point (≤1/2^30 error), which would perturb bit-exactness; `construct_histograms_f64_on` already returns f64.
- **Fix:** Inlined the reference host `fix_histogram` + `compact_histogram` (O(num_bin) scalar f64, ascending order) directly in the driver — identical to the SerialTreeLearner host fold.
- **Files modified:** grow_driver.rs
- **Verification:** STRUCTURE gate bit-exact incl. leaf values.
- **Committed in:** `bd05c5c`

**3. [Rule 2 - Missing Critical] Proving-slice config pinned in the driver + test-infra hoist**
- **Found during:** Task 1 / Task 2
- **Issue:** The `grow_tree_on_device` trait seam carries no `GainConfig`, so the driver must pin one; and the anchor+comparator helpers lived inside `#[cfg(feature="rocm")] mod hip`, unreachable from the required DEFAULT-cpu-build gate.
- **Fix:** Added `proving_slice_config()` (continuous+L2, permissive min_data) which the STRUCTURE gate's anchor mirrors; hoisted the shared anchor (`cpu_anchor_tree`, now with a `cfg` param) + comparator + constants to module scope so both the default gate and the rocm cells share one definition; rewrote the two rocm Slice-0 cells to use the small proving corpus so their anchor matches the driver's config; made `seam_defers` env-aware; retired `host_grow`.
- **Files modified:** grow_driver.rs, learner_parity.rs
- **Verification:** default gate + env-unset workspace + rocm cells all green.
- **Committed in:** `bd05c5c`, `e7dbcf4`

---

**Total deviations:** 3 (2 blocking kernel-selection choices for bit-exactness, 1 missing-critical config/test-infra). **Impact:** All serve the load-bearing bit-exact STRUCTURE gate; the driver still composes the Phase-16/17/18 kernels (subtraction trick, split-before-partition, DeviceCudaTree) per the plan. Scope unchanged (continuous+L2 proving slice). No new package installs.

## Issues Encountered
None blocking. The `cuda_on_device_enabled()` OnceLock caches per-process, so the STRUCTURE gate is run in its own `LGBM_CUDA_ON_DEVICE=1` invocation (`--exact`) while `cargo test --workspace` runs env-unset — the standard merge-gate contract. Cells that must hold in both invocations read the env directly and adapt.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- The on-device grow driver + activated seam are ready for Phase 21 on-device growth. The proving slice is continuous+L2+MissingType::None; the ordering contract (root fold → build-smaller/subtract-larger → find-best → split_on_device BEFORE partition → seed children) is documented for the L1/quantile/categorical follow-up (missing-value forward preamble + categorical split kernel).
- The GPU path is validated on real hip hardware (STRUCTURE bit-exact vs the cpu f64 anchor with env=1); DOUBLE-BUFFER leaf-map strategy from 20-03a is honoured via the driver's per-leaf row bookkeeping.

---
*Phase: 20-on-device-score-updater-metrics*
*Completed: 2026-07-02*
