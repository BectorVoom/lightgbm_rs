# Roadmap: LightGBM-rs

## Overview

A pure-Rust, parity-faithful port of Microsoft LightGBM on a CubeCL CPU/ROCm backend, built bottom-up along a dependency-forced spine so numerical fidelity is provable at every layer. Data types are `f32` (single-precision) end-to-end to match the C++ reference defaults, and the oracle tolerance is ~1e-6 absolute. Milestone v1.1 ports the **full single-GPU CUDA training pipeline on-device** (all training state resident, host only orchestrates), grounded subsystem-by-subsystem in `docs/cuda-kernel-design.md`.

## Milestones

- ✅ **v1.0 Full Single-Machine Parity** — Phases 1–8 (shipped 2026-06-21)
- 🚧 **v1.1 CUDA On-Device Training Backend** — Phases 14–23 (full on-device pipeline; roadmapped 2026-06-29)

## Phases

<details>
<summary>✅ v1.0 Full Single-Machine Parity (Phases 1–8) — SHIPPED 2026-06-21</summary>

Full details archived in [`milestones/v1.0-ROADMAP.md`](milestones/v1.0-ROADMAP.md). Requirements in [`milestones/v1.0-REQUIREMENTS.md`](milestones/v1.0-REQUIREMENTS.md). Audit in [`milestones/v1.0-MILESTONE-AUDIT.md`](milestones/v1.0-MILESTONE-AUDIT.md).

- [x] Phase 1: Oracle Contract + Foundations (3/3 plans) — completed 2026-06-05
- [x] Phase 2: Dataset + Binning, determinism root (7/7 plans) — completed 2026-06-05
- [x] Phase 3: Tree Model + Model Text I/O + Predict Parity (4/4 plans) — completed 2026-06-05
- [x] Phase 4: Compute Backend, CPU-first f32 histograms → ROCm (4/4 plans) — completed 2026-06-05
- [x] Phase 5: Tree Learner + Split Finding (9/9 plans, bit-exact vs real lib_lightgbm 4.6) — completed 2026-06-06
- [x] Phase 6: GBDT Spine + Core Objectives/Metrics (6/6 plans) — completed 2026-06-07
- [x] Phase 7: Parity-Completing Variants (14/14 plans) — completed 2026-06-08
- [x] Phase 8: Python Bindings (8/8 plans) — completed 2026-06-08

</details>

### Post-v1.0 (unroadmapped — out-of-milestone work)

Phase directories that exist on disk but were never part of the v1.0 milestone scope. They ran as ad-hoc quick tasks / spikes after v1.0's phases completed. These are the **reusable shipped foundations** the v1.1 on-device pipeline builds on (resident histogram pool, u64 fixed-point build, feature-per-lane scan, sibling co-pack, autotuning):

- `09-gpu-hist-build-perf` — GPU/CPU training-speed perf campaign (CPU histogram-build wins shipped bit-exact; GPU kernel concluded ROCm-parity-not-speed). No formal VERIFICATION.
- `10-quantized-training` — opt-in approximate quantized-grad training (maps to deferred v2 requirement `QGD-01`). No formal VERIFICATION.
- `11-gpu-fixedpoint-int-atomics` — **complete (3 plans, spike-validated).** Replaced the ROCm histogram BUILD's f32 atomics with wide fixed-point u64 (S=2^30): ~1.3–1.7× faster (wide large-leaves) + ~3600× more accurate + deterministic, within the ~1e-6 gate. This **u64 fixed-point build kernel** is reused on-device in v1.1 Phase 16 (the no-f64-per-row constraint, ODL-09/19). SPEC: `phases/11-gpu-fixedpoint-int-atomics/SPEC.md`.
- `12-gpu-sibling-scan-copack` — **complete (3 plans, spike-validated).** Co-packed the two per-sibling resident scan launches+readbacks into ONE 2-slot scan launch + ONE readback per split (~59→~30 syncs/tree). The **sibling co-pack scan** is reused on-device in v1.1 Phases 16–17. Evidence: `crates/lgbm-compute/examples/spike024_sibling_scan_ab.rs`.
- `13-gpu-autotune-launch-config` — **complete (4 plans, spike-validated 037–040).** Replaced the hand-tuned/env GPU launch-config heuristics with CubeCL runtime autotuning (`cubecl::tune`), default-on for all GPU (rocm) selection (histogram-BUILD row-partition `P` + split-SCAN `CubeDim`). Self-calibrates on discrete gfx110x / NVIDIA — the portability lever the on-device CUDA path inherits. Evidence: `.claude/skills/spike-findings-lightgbm_rs/references/gpu-kernel-autotuning.md`, `.planning/spikes/037..040/`.

The `14`/`15` provisional on-device stub dirs were **CLEARED** when milestone v1.1 was rewritten 2026-06-29 from a narrow on-device-growth slice to the full single-GPU CUDA pipeline. The seam + oracle code (`Backend::grow_tree_on_device`, `on_device_growth_supported()`, `LeafPartitionLayout`, `assert_on_device_tree_matches_cpu_anchor`) **remains in git** (crates/lgbm-compute, lgbm-treelearner, oracle-harness) and is re-established/extended by the new Phase 14. The new v1.1 roadmap is below.

### 📋 Next milestone (not yet scoped)

Candidate themes deferred to v2: on-device quantized training (QGD-01..03 — the gradient discretizer §4 + integer histogram/split path), multi-GPU on-device learning, distributed/network training, linear-tree leaves, text/binary/Arrow ingestion.

## Progress (v1.0)

| Phase | Milestone | Plans Complete | Status | Completed |
|-------|-----------|----------------|--------|-----------|
| 1. Oracle Contract + Foundations | v1.0 | 3/3 | Complete | 2026-06-05 |
| 2. Dataset + Binning | v1.0 | 7/7 | Complete | 2026-06-05 |
| 3. Tree Model + Predict Parity | v1.0 | 4/4 | Complete | 2026-06-05 |
| 4. Compute Backend (CPU → ROCm) | v1.0 | 4/4 | Complete | 2026-06-05 |
| 5. Tree Learner + Split Finding | v1.0 | 9/9 | Complete | 2026-06-06 |
| 6. GBDT Spine + Core Objectives/Metrics | v1.0 | 6/6 | Complete | 2026-06-07 |
| 7. Parity-Completing Variants | v1.0 | 14/14 | Complete | 2026-06-08 |
| 8. Python Bindings | v1.0 | 8/8 | Complete | 2026-06-08 |

---

## Milestone v1.1 — CUDA On-Device Training Backend

**Goal:** Port the full LightGBM single-GPU CUDA training pipeline on-device per `docs/cuda-kernel-design.md` — all training state (gradients, histograms, split records, row permutation, tree model, cumulative scores) stays resident on device while the host only sizes grids, resolves template specializations, sequences kernel launches on streams, and reads back a handful of scalars per iteration. This mirrors `CUDASingleGPUTreeLearner` + the boosting-layer device path, replacing the host-driven per-leaf loop (~8,570 small serial launches / 100-tree train) and closing the architectural GPU training-speed gap vs official LightGBM.

**C++ port-source map:** `docs/cuda-kernel-design.md` — source-verified design reference for the full CUDA backend (58 files, 81 `__global__` kernels, 11 subsystems) being mirrored on-device. Each phase below cites the `§` it ports. See `.planning/REFERENCE_MANIFEST.md`.

**Structure:** 10 dependency-ordered, independently **anchor-gated** phases. Build order is forced by the design doc's subsystem decomposition: shared primitives → device dataset → histogram → best-split → partition/tree → boosting-layer objectives/score/metrics → end-to-end driver → categorical coverage → perf/default-on DoD. Every phase ships **additive** and **off by default** behind `LGBM_CUDA_ON_DEVICE`.

**Non-negotiables (threaded into every relevant phase's success criteria):**

- **Anchor-gated:** STRUCTURE bit-exact to the cubecl-cpu f64 fold (tie-aware `default_left`), leaf/score values within a ~1e-5 f32 envelope — **never** comparing two nondeterministic GPU f32 paths to each other (def-f8u-01).
- **Additive, off by default behind `LGBM_CUDA_ON_DEVICE`:** CPU / ROCm / existing-host-CUDA paths grow byte-identical trees with the env unset — the hard merge gate (`raw_bin_train_matches_cpp_golden`, `learner_parity`, lgbm/treelearner/compute suites) is green and unchanged on every change.
- **NO f64 per-row hot loops** in new kernels (consumer-NVIDIA f64 = 1/32 f32, measured 5.4× regression spike-052) — keep the u64 fixed-point build path; f64 only in scalar/gain math where the reference uses it.
- The **subtraction trick + most-freq-bin fix + mark→prefix-sum→scatter row order** are CORRECTNESS requirements (a different rounding/accumulation path otherwise), not just speed.
- **`CUDATree.Split` runs BEFORE `DataPartition.Split`** (returns `right_leaf_index` the partition consumes).

### Phases (summary checklist)

- [x] **Phase 14: Foundation — Shared Device Primitives + Device Structs/RNG** — The reusable CubeCL primitives + device split-record/RNG every later subsystem builds on; re-establish the on-device seam + anchor-pinned oracle. (completed 2026-06-29)
- [x] **Phase 15: On-Device Device Dataset + Row-Subset Gather** — Resident columnar binned dataset in the feature-partition layout the histogram kernel needs, + CopySubrow bagging/GOSS subset. (completed 2026-06-29)
- [x] **Phase 16: On-Device Histogram Constructor** — The hot path: build (dense/sparse × shared/global) on u64 fixed-point + the subtraction trick (FixHistogram + SubtractHistogram via `hist_t**` rotation). (completed 2026-07-01)
- [x] **Phase 17: On-Device Best-Split Finder** — Per-feature split evaluation + cross-feature/cross-leaf argmax with a single small readback; tie-aware `default_left`. (completed 2026-07-01)
- [x] **Phase 18: On-Device Data Partition, Tree Mutation & Prediction** — mark→prefix-sum→scatter row routing + pool pointer swap; Split-before-partition; tree-walk predict. (completed 2026-07-01)
- [ ] **Phase 19: On-Device Objectives** — Regression-family / binary / multiclass / ranking grad-hess + ConvertOutput/BoostFromScore/RenewTreeOutput, all anchor-pinned.
- [ ] **Phase 20: On-Device Score Updater & Metrics** — Resident cumulative `cuda_score_` + the 12 supported pointwise metrics (EvalKernel); unsupported metrics fall back to host.
- [ ] **Phase 21: End-to-End On-Device Driver Integration + Parity Gate** — The single-GPU tree-learner driver runs the full grow loop on-device; STRUCTURE bit-exact; no-f64 kernel constraint verified.
- [ ] **Phase 22: On-Device Categorical Splits (Feature Coverage)** — Bitset construction + categorical split eval + categorical partition + SplitCategorical, via the pre-allocated bitset.
- [ ] **Phase 23: Perf-Validation + Default-On Rollout (DoD)** — Kaggle A/B (`device_launches` + wall-clock ratio); flip default-ON for CUDA contingent on parity + not-slower; host fallback retained.

### Phase Details

#### Phase 14: Foundation — Shared Device Primitives + Device Structs/RNG

**Goal**: The reusable CubeCL device primitives and device structs/RNG every later subsystem builds on are ported and validated, and the existing on-device growth seam + anchor-pinned oracle are re-established/extended — all additive and off by default.
**Depends on**: v1.0 (Phases 4/5 compute + tree-learner) + post-v1.0 resident-pool / autotune work (Phases 11–13)
**Requirements**: ODL-01, ODL-02
**Success Criteria** (what must be TRUE):

  1. The shared device primitives — block + multi-kernel global **prefix-sum** (inclusive/exclusive), **shuffle reductions** (sum/max/min, dot-product), **bitonic argsort** (index-only, never moves values), and **weighted/unweighted percentile** — exist as reusable CubeCL kernels, each anchor-pinned where it carries numeric output. (§2.4; §17 "port first")
  2. A CubeCL-safe device **split-record** (a pre-allocated `CUDASplitInfo` analog, no per-split in-kernel device alloc) and a **`CUDARandom` LCG** produce a bit-identical stream to the host `Random` (verified against the host extra-trees / sampling / per-item-rand draw sequence). (§15)
  3. The additive `Backend::grow_tree_on_device` seam + default-false `on_device_growth_supported()` discriminator + the anchor-pinned tie-aware `assert_on_device_tree_matches_cpu_anchor` oracle are re-established/extended in lgbm-compute / lgbm-treelearner / oracle-harness (the seam code already in git), never comparing two GPU paths to each other.
  4. `LGBM_CUDA_ON_DEVICE` is OFF by default; CPU / ROCm / existing-host-CUDA paths are byte-unchanged; the full merge gate is green.

**Plans**: 6/6 plans complete
**Wave 1**

- [x] 14-01-PLAN.md — Plane-intrinsic smoke test (Open Q1 de-risk) + new-module scaffolding (Wave 1)
- [x] 14-02-PLAN.md — C++/HIP device-primitive fixture-capture harness + committed goldens (Wave 1)

**Wave 2** *(blocked on Wave 1 completion)*

- [x] 14-03-PLAN.md — Full-depth primitives: prefix-sum (block+global), reductions, single-block bitonic argsort (Wave 2)
- [x] 14-04-PLAN.md — SoA pre-allocated device split-record + CUDARandom LCG (Wave 2)

**Wave 3** *(blocked on Wave 2 completion)*

- [x] 14-05-PLAN.md — Anchor-pinned skeletons: percentile, multi-block argsort, per-segment items-sort (Wave 3)

**Wave 4** *(blocked on Wave 3 completion)*

- [x] 14-06-PLAN.md — No-op seam + oracle extension + primitive fixture parity + full merge gate (Wave 4)

**Notes**: The seam (`grow_tree_on_device`, `on_device_growth_supported`, `LeafPartitionLayout`, the tie-aware oracle) already exists in git from the cleared Phase-14/15 work — extend, don't rebuild. Bake the cubecl-0.10 gotcha checklist (no global barrier, `Atomic<i64>` broken, `wrapping_add` not an intrinsic, plane-sum ≤ plane width, `launch_unchecked` unsafe) into the primitives. Re-measured pre-on-device CUDA baseline (Kaggle Tesla T4, 500k×50, 100 trees): official LightGBM ~4.46× faster (3.36 s vs lgb-rs 14.98 s) — the bar later phases must beat.

#### Phase 15: On-Device Device Dataset + Row-Subset Gather

**Goal**: The binned feature matrix and the bagging/GOSS subset live resident on device in the feature-partition layout the histogram kernel is built around.
**Depends on**: Phase 14
**Requirements**: ODL-03, ODL-04
**Success Criteria** (what must be TRUE):

  1. An on-device **columnar binned dataset** (u8/16/32 bin-width dispatch; dense + sparse CSR) is resident on device, carrying the **feature-partition layout** — features grouped so one partition's histogram fits shared memory, a too-wide column becoming its own large-bin partition (→ global-memory path). (§3, §13)
  2. An on-device **row-subset gather** (a `CopySubrow` analog) builds the bagging / GOSS subset dataset on device, anchor-pinned to the host subset-selection draw sequence. (§3)
  3. The resident dataset reproduces the host binned values exactly (per-column bin parity), and bin-width + partition dispatch is validated across all three widths and the large-bin spill case.
  4. CPU / ROCm / existing-host-CUDA paths are byte-unchanged; the merge gate is green.

**Plans**: 5/5 plans complete

- [x] 15-01-PLAN.md — Wave 0: register 3 ungated kernel modules + stub API surface + author both Nyquist parity test files + D-04 sparse synthesizer (ODL-03/04)
- [x] 15-02-PLAN.md — Wave 1: §13 CUDARowData row+partition store — DivideCUDAFeatureGroups + offset tables + dense/sparse 3×3 re-lay (ODL-03)
- [x] 15-03-PLAN.md — Wave 1: §3 CUDAColumnData column store + numeric per-feature meta, upload-once, no consumer (ODL-03)
- [x] 15-04-PLAN.md — Wave 1: CopySubrow gather + on-device bagging draw anchored to host bag_data_indices (ODL-04)
- [x] 15-05-PLAN.md — Wave 2: merge gate — full workspace suite green, D-10 byte-unchanged default path (ODL-03/04)

**Notes**: `CUDARowData` (§13) is pure host-side layout infrastructure plus the `CopySubrow` kernel — the per-partition CSR re-lay (`GetSparseDataPartitioned`, subtracting `partition_hist_start`) and the `max_num_bin_per_partition = shared_hist_size/2` budget are the parity-load-bearing details. Reuse the v1.0 binning; this phase only mirrors it resident in the partition layout. Wave shape: 0 (scaffold) → 1 (row_data ∥ column_data ∥ copy_subrow — disjoint files) → 2 (merge gate). Bagging anchor test reproduces the host draw INLINE via `lgbm_core::random::Random` — lgbm-compute cannot dev-dep lgbm-boosting (crate cycle).

#### Phase 16: On-Device Histogram Constructor

**Goal**: The hot-path histogram — build, fix, and subtract — runs on-device, building only the smaller leaf and deriving the larger by subtraction via pointer rotation, anchor-pinned.
**Depends on**: Phase 15
**Requirements**: ODL-09, ODL-10
**Success Criteria** (what must be TRUE):

  1. On-device **histogram build** (dense + sparse × shared-memory + global-memory spill) runs on the f32 / **u64 fixed-point** accumulation path with two-tier atomic accumulation (block-local then cross-block merge), anchor-pinned to the cpu f64 fold — reusing the shipped Phase-11 u64 fixed-point build kernel. (§7.1–7.4)
  2. The **subtraction trick** runs on-device — build-smaller-only, **`FixHistogram`** (most-frequent-bin omission repair via leaf-total minus scanned sum), **`SubtractHistogram`** (larger = parent − smaller) via **`hist_t**` pointer rotation** (larger child inherits the parent buffer; smaller gets a fresh arena slot), no bulk histogram copy. (§7.5, §17)
  3. The build-smaller-before-subtract ordering invariant holds (parent fully built before any child subtract reads it — the 8aed100-class guard), and the most-freq-bin fix + interleaved `[2b]/[2b+1]` layout match the reference exactly — both as CORRECTNESS requirements, not speed. (§17)
  4. The derived larger-child histogram is anchor-pinned (bit-exact on the cpu f64 anchor; ROCm/CUDA f32 within ~1e-6); CPU / ROCm / host-CUDA byte-unchanged; merge gate green.

**Plans**: 5/5 plans complete

- [x] 16-01-PLAN.md
- [x] 16-02-PLAN.md
- [x] 16-03-PLAN.md
- [x] 16-04-PLAN.md
- [x] 16-05-PLAN.md

**Notes**: The single most performance-critical kernel (§7, 960 lines, 13 `__global__` variants in C++). The quantized/discretized build kernels (§7.3) are **v2 (QGD-02)** — skip. Template-flag explosion (BIN_TYPE/HIST_TYPE/shared-vs-global) maps to CubeCL comptime.

#### Phase 17: On-Device Best-Split Finder

**Goal**: Per-feature split evaluation and the cross-feature/cross-leaf argmax run on-device, returning the chosen split with a single small scalar readback and tie-aware `default_left` parity.
**Depends on**: Phase 16
**Requirements**: ODL-11, ODL-12
**Success Criteria** (what must be TRUE):

  1. On-device **per-feature split evaluation** (stage 1, one block per (leaf,feature) task) runs block prefix-sum → cumulative left/right sums, count recovery via `cnt_factor` + `__double2int_rn`, min-data / min-sum-hessian guards, gain math, forward/reverse default-bin scan, and block argmax → per-task split record. (§8.1, numerical core)
  2. On-device **cross-feature reduce (stage 2) + cross-leaf argmax (stage 3)** produce the chosen `(leaf, feature, threshold, default_left)` with a single small scalar readback per split (the 8-int buffer). (§8.2–8.3)
  3. **Tie-aware `default_left`** parity to the cpu anchor: a flip is accepted only on a verified f32 tie (same threshold + left_count + f32-equal gains); a flip on any non-tie split hard-fails; the empty / sparse-default-bin fixtures pass.
  4. The chosen split is anchor-pinned (structure bit-exact, values within ~1e-5); CPU / ROCm / host-CUDA byte-unchanged; merge gate green.

**Plans**: 5/5 plans complete

Plans:
**Wave 1**

- [x] 17-01-PLAN.md — Wave-0 test infra + host scaffolding (best_split_parity harness, best_split.txt 6-category fixtures, SplitFindTask, build_split_find_tasks task-gen table, round-ties-even helper, RNG-seed lock) [wave 1]
- [x] 17-02-PLAN.md — Gain-math USE_SMOOTHING output-blend branch + #[cube] promotion of get_leaf_gain_given_output (additive; non-smoothing gain fns byte-unchanged) [wave 1]

**Wave 2** *(blocked on Wave 1 completion)*

- [x] 17-03-PLAN.md — Stage-1 numerical core: split_eval_body cpu f64 fold (scan→complement→count-recovery→guards→gain→argmax→record) + hip f32 two-level LDS scan mirror [wave 2]

**Wave 3** *(blocked on Wave 2 completion)*

- [x] 17-04-PLAN.md — Stage-1 _GlobalMemory >256-bin spill variant + pre-allocated scratch (D-05/D-11) [wave 3]

**Wave 4** *(blocked on Wave 3 completion)*

- [x] 17-05-PLAN.md — Stage-2 cross-feature reduce + Stage-3 cross-leaf argmax + 8-int export + self-invalidation + tie-aware default_left on hip [wave 4]

**Notes**: The 256-bin within-feature scan needs a segmented LDS block-scan (plane-sum caps at plane width 32/64 ≪ 256) — net-new kernel work, not a reuse (research flag). The discretized split finder (§8.1 quantized inner) is **v2 (QGD-02)** — skip. Do NOT defer the tie-aware assert.

#### Phase 18: On-Device Data Partition, Tree Mutation & Prediction

**Goal**: Row routing, tree mutation, and prediction run on-device — Split before partition, mark→prefix-sum→scatter row permutation (never sorting), and the histogram-pool pointer swap — eliminating the host partition round-trip.
**Depends on**: Phase 17
**Requirements**: ODL-13, ODL-14, ODL-15
**Success Criteria** (what must be TRUE):

  1. On-device **data partition** routes rows by `mark → prefix-sum → scatter` (**never sorting**) into two contiguous child ranges, updates the data-index→leaf map, and performs the `SplitTreeStructure` **histogram-pool pointer swap**; the resulting row order matches the reference so per-leaf f32 accumulation order is identical. (§9, §17)
  2. On-device **tree mutation** (`Split` writing the device tree arrays) runs **BEFORE** the partition step and returns the `right_leaf_index` the partition consumes, plus `Shrinkage` / `AddBias`, anchor-pinned to the host tree structure. (§10, ordering note)
  3. On-device **prediction** — the tree-walk `AddPredictionToScore` over the device columnar dataset (numeric threshold + missing/`default_left` handling, categorical bitset membership) — is within ~1e-6 + objective inverse-link. (§10)
  4. Per-split device→host transfer is the single 16-int packet; structure anchor-pinned, leaf values within ~1e-5; CPU / ROCm / host-CUDA byte-unchanged; merge gate green.

**Plans**: 4/4 plans complete

- [x] 18-01-PLAN.md — Wave 0: u16/u32 integer scan launchers + extended kernel_capture goldens (flag fan-out, categorical, 16-int packet, predict) + #[ignore] Nyquist scaffolds (ODL-13/14/15)
- [x] 18-02-PLAN.md — Wave 1: data_partition.rs §9 mark→prefix-sum→scatter (numeric + categorical) + 16-int packet + cpu f64 stable-partition anchor + HistArena leaf-indexed pool swap (ODL-13)
- [x] 18-03-PLAN.md — Wave 1: tree.rs device flat CUDATree + SplitKernel (Split-before-partition) + SplitCategorical/Shrinkage/AddBias (ODL-14)
- [x] 18-04-PLAN.md — Wave 2: predict.rs tree-walk AddPredictionToScore (numeric 8/16/32 + categorical membership) + §9 leaf-map add + hip f32 parity gate (ODL-15)

**Notes**: Clone the shipped `LGBM_RESIDENT_FORCE` size-gate + default-off routing precedent. Partition placement nuance (spike-035): the round-trip is pure overhead on shared-memory APUs (ROCm keeps its host-partition path); the payoff is on discrete PCIe NVIDIA — measure on Kaggle. Categorical membership routing is wired here but the categorical *feature* end-to-end lands in Phase 22. D-04 order-equivalence RESOLVED CONFIRMED (cpu anchor = plain stable partition, no block-tiled escalation).

#### Phase 19: On-Device Objectives

**Goal**: All CUDA-supported objectives compute grad/hess (plus ConvertOutput, BoostFromScore, RenewTreeOutput) on-device, anchor-pinned, so the boosting layer never round-trips gradients to host.
**Depends on**: Phase 14 (primitives + RNG); parallelizable with the grow-loop chain (Phases 16–18)
**Requirements**: ODL-05, ODL-06, ODL-07, ODL-08
**Success Criteria** (what must be TRUE):

  1. On-device **regression-family** grad/hess (L2, L1, Quantile, Huber, Fair, Poisson) + `ConvertOutput` inverse-link + `BoostFromScore` (mean via reduce / median via percentile) + `RenewTreeOutput` (median/quantile leaf refit, one block per leaf), anchor-pinned. (§5.1)
  2. On-device **binary-logloss** grad/hess + `BoostFromScore` (label-prior logit init) + sigmoid `ConvertOutput`, with the one-vs-all label reset for OVA, anchor-pinned. (§5.2)
  3. On-device **multiclass** grad/hess (softmax + one-vs-all, class-major `[k·num_data+i]` layout) anchor-pinned. (§5.3)
  4. On-device **ranking** grad/hess (LambdaRank-NDCG + RankXENDCG, per-query block layout, bitonic item ranking, per-item RNG with the bit-identical stream) anchor-pinned. (§5.4)
  5. The CUDA-unsupported objectives (MAPE / Gamma / Tweedie / cross-entropy / rank-MAP) honestly fall back to host; CPU / ROCm / host-CUDA byte-unchanged; merge gate green.

**Plans**: 3/5 plans executed
**Wave 1**

- [x] 19-00-PLAN.md — Wave 1: greenfield objective module stubs + mod.rs, host-fallback support-set (SC #5), shared parity harness, lambdarank golden capture

**Wave 2** *(blocked on Wave 1 completion)*

- [x] 19-01-PLAN.md — Wave 2: ODL-05 regression family (6 grad/hess kernels + ConvertOutput + BoostFromScore + host-orchestrated RenewTreeOutput)
- [x] 19-02-PLAN.md — Wave 2: ODL-06 binary-logloss (grad/hess + two-stage BoostFromScore logit init + sigmoid ConvertOutput + OVA label reset)
- [ ] 19-03-PLAN.md — Wave 2: ODL-07 multiclass (class-major softmax grad/hess + softmax ConvertOutput + MulticlassOVA)
- [ ] 19-04-PLAN.md — Wave 2: ODL-08 ranking (LambdaRank-NDCG shared+>2048 + RankXENDCG shared+global + per-item RNG)

**Notes**: Exactly 11 CUDA-supported objectives (§5). The atomic-ordering nondeterminism in binary BoostFromScore + lambdarank is the documented f32-vs-f64 residual the ROCm gate tolerates — pin to the cpu f64 anchor, never GPU-vs-GPU. Each objective is a separable, individually-anchor-gated addition. Wave 2 plans (19-01..04) are fully parallel (disjoint owned files); each depends only on 19-00.

#### Phase 20: On-Device Score Updater & Metrics

**Goal**: The cumulative score lives resident on device and the supported pointwise metrics evaluate on-device, completing the boosting-layer device path.
**Depends on**: Phase 18 (prediction/partition) + Phase 19 (objective `ConvertOutput` inverse-link)
**Requirements**: ODL-16, ODL-17
**Success Criteria** (what must be TRUE):

  1. On-device **score update** — resident cumulative `cuda_score_`, constant add / multiply (init score / shrinkage / no-split single-leaf / DART rescale), replacing the host `add_prediction_to_score` scatter, with a host-mirror toggle for non-resident consumers. (§11)
  2. On-device **pointwise metric evaluation** — `EvalKernel` + two-stage reduction over the 12 supported regression/binary losses — anchor-pinned. (§12)
  3. The CUDA-unsupported metrics (AUC / NDCG / MAP / multiclass / cross-entropy) honestly fall back to host. (§12.1)
  4. Score + metric outputs are anchor-pinned (within the ~1e-6 / ~1e-5 envelope); CPU / ROCm / host-CUDA byte-unchanged; merge gate green.

**Plans**: TBD
**Notes**: Small subsystems (§11 = 45 lines, §12 = 78 lines) but the boosting-layer glue that lets the score stay resident across iterations (`boosting_on_cuda_`). The 12 pointwise losses are all regression/binary; everything else stays host-side per the reference's own `#ifdef USE_CUDA` branch.

#### Phase 21: End-to-End On-Device Driver Integration + Parity Gate

**Goal**: The single-GPU tree-learner driver orchestrates the full per-leaf grow loop end-to-end on device and reconstitutes into the `(Tree, DataPartition)` the boosting loop consumes — structure bit-exact to the cpu f64 anchor, with the no-f64-per-row kernel constraint verified across every new kernel.
**Depends on**: Phase 18 (grow-loop chain) + Phase 19 (objectives) + Phase 20 (score/metrics)
**Requirements**: ODL-18, ODL-19
**Success Criteria** (what must be TRUE):

  1. With `LGBM_CUDA_ON_DEVICE=1`, the on-device driver runs root init → build/subtract → best-split → tree split → partition, repeated up to `num_leaves−1` (break on `best_leaf == −1`), entirely on device, and reconstitutes into a valid `(Tree, DataPartition)` the boosting loop consumes (the continuous-feature path is the proving slice). (§6, §16)
  2. The grown tree is STRUCTURE **bit-exact** to the cpu f64 anchor (tie-aware `default_left`), leaf values within ~1e-5 — never comparing two nondeterministic GPU f32 paths to each other.
  3. Every new kernel keeps **f32 + the u64 fixed-point build with NO f64 per-row hot loops** (verified by grep + per-tree-ms, not a 6× sweep; the measured 5.4× consumer-NVIDIA f64 regression, spike-052); f64 only in scalar/gain math where the reference uses it. (§17)
  4. CPU / ROCm / existing-host-CUDA paths are **byte-unchanged** with `LGBM_CUDA_ON_DEVICE` unset — the hard merge gate is green and unchanged. (§17)

**Plans**: TBD
**Notes**: The integration phase that ties the resident loop together. Highest-uncertainty for the *magnitude* of the win (the best-first loop still serializes per split) — but parity is the gate, not speed (speed is Phase 23's DoD). Verify at plan time: cubecl 0.10 `Handle` in-place aliasing vs ping-pong double-buffering for the data→leaf map; batched `client.read(vec![h])` readback semantics on cubecl-cuda.

#### Phase 22: On-Device Categorical Splits (Feature Coverage)

**Goal**: Categorical splits work end-to-end on the proven numerical driver — bitset construction, categorical split evaluation, categorical partition membership, and SplitCategorical — via a pre-allocated bitset representation (no per-`SplitInfo` device alloc).
**Depends on**: Phase 21
**Requirements**: ODL-22
**Success Criteria** (what must be TRUE):

  1. On-device **categorical bitset construction** (`SetRealThreshold` + length + construct, §6.3) materializes the selected-category bitset via the **pre-allocated** representation — NOT the reference's per-`SplitInfo` device alloc.
  2. On-device **categorical split evaluation** (one-hot + many-vs-many bitonic-sorted, §8.1) and **categorical partition membership** (§9) route rows by `CUDAFindInBitset` membership, anchor-pinned.
  3. `SplitCategorical` **tree mutation** (§10) writes the categorical node (`num_cat`, `cat_boundaries`) and predicts correctly, anchor-pinned to the host categorical tree.
  4. The numerical spine stays byte-untouched and anchor-pinned; CPU / ROCm / host-CUDA byte-unchanged; merge gate green.

**Plans**: TBD
**Notes**: Layered on the proven Slice (Phases 14–21) numerical driver. The pre-allocated bitset slab (ODL-02 `AllocateCatVectorsKernel` analog) avoids the per-`SplitInfo` `cudaMalloc` that has no clean CubeCL analog.

#### Phase 23: Perf-Validation + Default-On Rollout (DoD)

**Goal**: Measure the on-device win on real CUDA and make the on-device learner the DEFAULT CUDA tree-learner path — contingent on parity AND not-slower — with the host path retained as an off-switch. This is the milestone Definition of Done.
**Depends on**: Phase 22 (full feature coverage) + the per-phase Kaggle checkpoints
**Requirements**: ODL-20, ODL-21
**Success Criteria** (what must be TRUE):

  1. A real-CUDA **Kaggle A/B harness** measures the on-device path's `device_launches/tree` (target well below the 8,570 / 100-trees baseline) and the lgb_rs / official wall-clock ratio at 500k×50 AND a wide shape (in-session A/B deltas, platform stated).
  2. The measured `device_launches` for 100 trees drops **BELOW** the 8,570 / 100-trees baseline — the architectural launch-collapse is confirmed, not just resident state.
  3. The on-device learner becomes the **DEFAULT** CUDA tree-learner path — contingent on anchor-pinned ~1e-6 parity AND not-slower-than-the-current-host-CUDA path on the Kaggle A/B — with `LGBM_CUDA_ON_DEVICE=0` retained as the off-switch fallback.
  4. ROCm + CPU routing stay host-driven / byte-unchanged; the CPU f64 merge gate is green.

**Plans**: TBD
**Notes**: Pure routing/perf, deferred until the win is measured — never auto-engaged before proof (the audit-before-wire value). The improvement magnitude is the genuine empirical unknown. Default-on flips ONLY where the real-CUDA A/B shows a sign-stable not-slower result (the fused-kernel default-off precedent). Multi-stream overlap is a stretch to spike only if launch-count reduction underdelivers. Kaggle CLI authenticated as `boomvector` is the only path to real discrete-CUDA numbers (local GPU is a spoofed APU).

### Progress (v1.1)

| Phase | Milestone | Plans Complete | Status | Completed |
|-------|-----------|----------------|--------|-----------|
| 14. Foundation — Shared Device Primitives + Structs/RNG | v1.1 | 6/6 | Complete   | 2026-06-29 |
| 15. On-Device Device Dataset + Row-Subset Gather | v1.1 | 5/5 | Complete    | 2026-06-29 |
| 16. On-Device Histogram Constructor | v1.1 | 5/5 | Complete    | 2026-07-01 |
| 17. On-Device Best-Split Finder | v1.1 | 5/5 | Complete    | 2026-07-01 |
| 18. On-Device Data Partition, Tree Mutation & Prediction | v1.1 | 4/4 | Complete    | 2026-07-01 |
| 19. On-Device Objectives | v1.1 | 3/5 | In Progress|  |
| 20. On-Device Score Updater & Metrics | v1.1 | 0/? | Not started | - |
| 21. End-to-End Driver Integration + Parity Gate | v1.1 | 0/? | Not started | - |
| 22. On-Device Categorical Splits | v1.1 | 0/? | Not started | - |
| 23. Perf-Validation + Default-On Rollout (DoD) | v1.1 | 0/? | Not started | - |
