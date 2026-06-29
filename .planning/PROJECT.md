# LightGBM-rs — Pure Rust LightGBM with CubeCL

## What This Is

A pure-Rust rewrite of Microsoft's LightGBM gradient-boosting library, built as a Cargo workspace and using the `cubecl` crate for compute and GPU acceleration. It targets ML practitioners and LightGBM users who want a memory-safe Rust implementation that runs on both CPU and AMD ROCm GPUs while remaining numerically faithful to the original. The Microsoft C++ implementation under `LightGBM/` is the read-only reference being ported; the deliverable is the Rust crate(s).

## Core Value

For identical inputs and configuration, the Rust implementation must reproduce the C++ LightGBM's outputs to within an absolute difference of **~1e-6 on every backend (CPU and ROCm)**, using `f32` (single-precision) data types end-to-end to match the C++ reference defaults (`score_t`/`label_t` = `float`). The CPU path — the `cubecl-cpu` f64-fold deterministic anchor — is the hard merge gate and, where the algorithm permits, achieves **bit-exact** parity with the C++ reference (e.g. binning, and the serial tree learner is bit-exact vs real `lib_lightgbm` 4.6 on both committed corpora); the ROCm path (`cubecl-hip`, f32) is held to ~1e-6 against that anchor, with residual f32-vs-f64 accumulation gaps documented per phase. Numerical fidelity at single precision is the non-negotiable contract; everything else serves it.

## Current State

**v1.0 SHIPPED (2026-06-21)** — full single-machine C++ LightGBM parity in pure Rust, exposed via a Rust-native API and Python bindings on a switchable CubeCL CPU/ROCm backend. 8 phases, 55 plans, 69/69 v1 requirements satisfied, milestone audit PASSED. ~68.7k LOC (65.5k Rust + 3.2k Python) across 11 workspace crates. The serial tree learner is bit-exact vs real `lib_lightgbm` 4.6 on both committed corpora; the cubecl-cpu f64 anchor is the hard merge gate; cubecl-hip f32 runs on real gfx1100 within ~1e-6.

Post-v1.0 work (running as phases/quick tasks, **not** part of the v1.0 scope): a GPU/CPU training-speed perf campaign (spikes 001–040 + quick tasks), an opt-in quantized-training mode (Phase 10 dir, maps to v2 `QNT-01`), the GPU fixed-point int-atomics + sibling-scan co-pack kernel work (Phases 11–12), and **GPU launch-config autotuning (Phase 13, complete 2026-06-26)** — CubeCL `cubecl::tune` runtime autotuning replaces the hand-tuned/env GPU launch heuristics, default-on for rocm, self-tuning both the histogram-build row-partition `P` and the split-scan `CubeDim` width `W`; all-PSET/all-WSET parity pinned to the CPU f64 anchor on real ROCm, CPU merge gate untouched, durable value = portability (self-calibrates on discrete gfx110x / NVIDIA).

**v1.1 Phase 14 (Scaffold + Oracle, Slice 0) — complete 2026-06-29.** The additive on-device growth seam (`Backend::grow_tree_on_device` + default-false `on_device_growth_supported()` discriminator), the `LeafPartitionLayout` payload, the decide-once `train_inner` routing fork gated on `LGBM_CUDA_ON_DEVICE` (OFF unless `=1`), and the anchor-pinned tie-aware oracle (`assert_on_device_tree_matches_cpu_anchor`) all exist with ZERO behavior change — the full bit-exact merge gate is green and byte-unchanged with the env unset. Wiring risk is now isolated from kernel risk before any Slice-1 kernel is written. The oracle's `default_left` tie-acceptance is keyed on the `split_gain` near-tie (binding quality, not scaffold). ODL-01/ODL-02 validated.

**v1.1 Phase 15 (Device Dataset + Row-Subset Gather) — complete 2026-06-30.** The resident on-device columnar binned dataset now lives in the feature-partition layout the histogram kernel is built around: `CudaColumnData` (§3, per-column native u8/16/32 upload-once store + numeric meta) and `CudaRowData` (§13, `divide_cuda_feature_groups` partition layout + offset tables + dense/sparse 3×3 CSR re-lay with large-bin spill), plus `CopySubrow` (`copy_subrow_on` row-subset gather + `bagging_draw_on` on-device draw anchored bit-for-bit to the inline host `bag_data_indices` reference — no `lgbm-boosting` dev-dep, crate-cycle held). New kernel modules are registered **ungated** so the cpu f64 anchor exercises them, but stay inert (no consumer until Phase 16). Code review caught and fixed two latent Critical parity bugs before they could reach Phase 16 (CR-01 sparse partition-local bin-width truncation; CR-02 `copy_subrow` length-invariant OOB) and the parity tests were hardened to actually exercise non-zero partition offsets + the spill case across all three widths. 5 plans, merge gate byte-unchanged with the env unset, ODL-03/ODL-04 validated.

**Port reference:** `docs/cuda-kernel-design.md` is the authoritative, source-verified map of the C++ CUDA backend being ported (58 files, 81 `__global__` kernels, 11 subsystems, verified kernel-by-kernel against `LightGBM/`). It is the per-subsystem porting reference for the on-device tree-learner slices (Phases 15–19) and records the parity-critical constraints: split-before-partition ordering (`CUDATree.Split` returns `right_leaf_index` before `DataPartition.Split`), subtraction-trick/most-freq-bin-fix as correctness (not speed) requirements, `hist_t=double` vs SP-f32 atomic nondeterminism, the integer-exact int16-packed quantized path, and the CUDA objective/metric support boundaries. Catalogued in `.planning/REFERENCE_MANIFEST.md`.

## Current Milestone: v1.1 — CUDA On-Device Training Backend

**Goal:** Port the full LightGBM single-GPU CUDA training pipeline on-device per `docs/cuda-kernel-design.md` — all training state (gradients, histograms, split records, row permutation, tree model, cumulative scores) stays resident on device while the host only sizes grids, resolves template specializations, sequences kernel launches on streams, and reads back a handful of scalars per iteration. This mirrors `CUDASingleGPUTreeLearner` + the boosting-layer device path, replacing the host-driven per-leaf loop (~8,570 small serial launches/train) and closing the architectural GPU training-speed gap vs official LightGBM. Anchor-pinned bit-exact to the cubecl-cpu f64 fold; CUDA/ROCm f32 within ~1e-6. Fully additive — CPU, ROCm, and the existing host-CUDA path stay byte-unchanged, off by default behind `LGBM_CUDA_ON_DEVICE`.

**Target subsystems (design-doc §-grounded, full on-device pipeline):**
- **Shared device primitives** (§2.4) — block/global prefix-sum, shuffle reductions, bitonic argsort, weighted percentile (the port-first foundation every subsystem builds on)
- **Device dataset + row-data partition layout** (§3, §13) — columnar binned store with u8/16/32 width dispatch, feature-partition (shared-hist-fit) layout the histogram kernel is built around, CopySubrow bagging subsets
- **On-device objectives** (§5) — regression family (L2/L1/Quantile/Huber/Fair/Poisson), binary logloss, multiclass (softmax/OVA), rank (LambdaRank-NDCG / RankXENDCG) — grad/hess + ConvertOutput + BoostFromScore + RenewTreeOutput
- **Leaf-splits + tree-learner driver** (§6) — root init, leaf-stat reduce/refit, categorical bitset construction, the per-leaf grow-loop orchestration
- **Histogram constructor** (§7) — build (dense/sparse × shared/global), FixHistogram (most-freq-bin omission repair), SubtractHistogram with `hist_t**` pointer rotation
- **Best-split finder** (§8) — 3-stage: per-feature evaluation → cross-feature reduce → cross-leaf argmax, numerical + categorical cores
- **Data partition** (§9) — mark→prefix-sum→scatter (never sort), SplitTreeStructure histogram-pool pointer swap, data-index→leaf map, AddPredictionToScore
- **Tree model I/O + prediction kernel** (§10) — device tree arrays, Split/SplitCategorical, Shrinkage/AddBias, tree-walk predict
- **Score updater** (§11) — resident `cuda_score_`, constant add/multiply, host-mirror toggle
- **On-device metrics** (§12) — EvalKernel + the 12 pointwise (regression/binary) losses
- **Device structs + RNG** (§15) — a CubeCL-safe pre-allocated `CUDASplitInfo` analog (no per-split device alloc), a `CUDARandom` LCG bit-identical to the host RNG stream
- **Real-CUDA perf-validation + default-on rollout** — Kaggle A/B `device_launches` + wall-clock ratio as the DoD; the on-device path becomes the default CUDA learner contingent on parity + not-slower

**Non-negotiables (carried from spikes 051–054 + §17 port considerations):** CPU f64 anchor stays the bit-exact merge gate; CUDA/ROCm held to ~1e-6; **NO f64 per-row hot loops** in new kernels (consumer-NVIDIA f64 = 1/32 f32 throughput; keep the u64 fixed-point build path), f64 only in scalar/gain math where the reference uses it; the **subtraction trick + most-freq-bin fix are correctness requirements** (build-smaller / subtract-larger gives a different rounding path than building larger directly); the mark→prefix-sum→scatter row permutation must match the reference so per-leaf f32 accumulation order is identical. On-device **quantized training** (the gradient discretizer §4 + discretized histogram/split paths) is **deferred to v2** (ODL-13). Canonical port reference: `docs/cuda-kernel-design.md` (source-verified against `LightGBM/src/treelearner/cuda/`).

## Requirements

### Validated

<!-- Shipped and confirmed valuable. -->

- [x] Dataset + binned columnar store (BinMapper, FeatureGroup, MultiValBin) matching C++ binning bit-for-bit — *Validated in Phase 2: dataset-binning-determinism-root (bit-exact ValueToBin, dense/CSR/CSC ingest, missing/categorical routing, EFB grouping, per-stage parity; default ingest unified onto the faithful single C++ `Dataset::Construct`)*
- [x] Model text format read/write + predict parity (Tree, GBDT model I/O, `%.17g`/`%g` formatter, raw/transformed/leaf/sub-range predict) — *Validated in Phase 3: tree-model-and-predict (PRD-01/02/03/06, DAT-08/09 all PASS within ~1e-6)*
- [x] CubeCL compute backend with switchable CPU ↔ ROCm (feature flag + runtime selection) — *Validated in Phase 4: compute-backend-cpu-first-integer-histograms-rocm. Backend trait with whole-kernel `construct_histograms`/`find_best_split`/`data_partition`/`subtract_histograms`; cubecl-cpu f64 fold is the BIT-EXACT deterministic anchor (D-04a proven across 25 launches + vs C++-order fold); cubecl-hip f32 path on real gfx1100 within a separate ~1e-6 gate vs the cpu anchor (best-effort D-03a, one documented ULP-scale gap in 04-ROCM-GAPS.md). CMP-01..05, ORA-04 satisfied. CUDA warp-level ops mapped onto CubeCL `Plane` API via the startup capability gate (asymmetric cpu/hip matrix).*
- [x] Histogram-based serial tree learner (split-gain scan, data partition, leaf splits) matching C++ — *Validated in Phase 5: tree-learner-split-finding (bit-exact vs real `lib_lightgbm` 4.6 on both committed corpora).*
- [x] GBDT training loop + core objectives/metrics + Rust-native train/predict/Dataset/Booster API, ~1e-6 (f32) oracle parity — *Validated in Phase 6: gbdt-spine-core-objectives-metrics (BST-01/02/03/07, OBJ-01/02/03, MET-01/02, API-01; 5/5 must-haves, 430 tests green). End-to-end builder→train→predict spine; 6 objectives (regression, regression_l1, binary, multiclass, multiclassova, custom) + 7 metrics; bagging RNG-replay + early stopping; the D-07 cross-product matrix validated against real `lib_lightgbm` 4.6. Scope boundary (decision-backed): `regression_l1 + bagging` typed-rejected then UN-DEFERRED + fixed in Phase 7 (07-01 min_gain_shift operand fix; DEF-06-01 closed).*
- [x] Parity-completing variants — GOSS/DART/Random Forest, categorical splits, full objective set (huber/fair/poisson/quantile/mape/gamma/tweedie, cross-entropy, lambdarank/rank_xendcg), extended + ranking metrics (ndcg/map/auc_mu/...), TreeSHAP + pred-early-stop, monotone/interaction/forced-splits/extra-trees/CEGB, refit + feature importance — *Validated in Phase 7: parity-completing-variants (BST-04/05/06, TRL-06, OBJ-04/05/06, MET-03/04, PRD-04/05, ADV-01..07; 14 plans, oracle-validated on the proven spine).*
- [x] Python bindings mirroring the official `lightgbm` package (PyO3 + numpy + sklearn wrappers, custom callbacks, refit, persistence) — *Validated in Phase 8: python-bindings (PYB-01..04; numpy/scipy/polars input, GIL-released train, LGBMClassifier/Regressor/Ranker, C++-compatible text I/O + pickle).*
- [x] Cargo workspace split into 11 loosely-coupled crates (lgbm-core/dataset/model/compute/treelearner/objective/metric/boosting/lgbm/lgbm-python/oracle-harness) under edition 2024 — *Validated across all phases.*
- [x] CubeCL CPU↔ROCm switchable backend + CUDA warp ops on the `Plane` API + standard f32 accumulations — *Validated in Phase 4 (CMP-01..05, ORA-04).*
- [x] Oracle harness comparing Rust vs C++ at ~1e-6 (f32), executed on ROCm; `thiserror` at boundaries + `anyhow` at app/test layers — *Validated in Phases 1+4 (ORA-01..04, FND-04).*

### Active

<!-- v1.0 shipped. No active v1 requirements — all 69 satisfied. Next milestone (v1.1/v2) not yet scoped. -->

**Milestone v1.1 — CUDA On-Device Training Backend (rewritten 2026-06-29, design-doc-grounded).** Active requirements are the `ODL-*` set in `.planning/REQUIREMENTS.md`, derived subsystem-by-subsystem from `docs/cuda-kernel-design.md`: shared primitives, device dataset/row-data, on-device objectives, leaf-splits/driver, histogram constructor, best-split finder, data partition, tree I/O + predict, score updater, on-device metrics, device structs/RNG, and the real-CUDA perf/rollout DoD. On-device quantized training (ODL-13) is deferred to v2. Phases renumber from 14.

### Out of Scope

<!-- Explicit boundaries with reasoning. -->

- Distributed / network (MPI-style) training — deferred beyond v1; large surface, not needed to prove the architecture
- C ABI (`c_api.h` / `LGBM_*`) parity — Rust-native + Python cover v1 consumers; revisit if C/R clients are needed
- CLI app (`lightgbm` command-line, config-file driven) — not required for v1 compatibility goals
- Raw CUDA / OpenCL backends — superseded by CubeCL per project mandate
- R bindings — out of scope
- NVIDIA-CUDA-specific tuning — ROCm is the mandated GPU test target

## Context

- **Reference codebase mapped:** `.planning/codebase/` documents the Microsoft C++ LightGBM core (ARCHITECTURE, STRUCTURE, STACK, CONVENTIONS, CONCERNS, INTEGRATIONS, TESTING). GPU-relevant subsystems are flagged there. The C++ tree is read-only reference, not a build target.
- **Current Rust crate:** v1.0 COMPLETE — all 8 phases shipped (11 crates, ~68.7k LOC). Phase 1 (oracle-contract foundations), Phase 2 (`lgbm-dataset`: bit-exact BinMapper, dense/sparse columnar bin store, missing/categorical encoding, EFB, metadata, dense/CSR/CSC ingestion), Phase 3 (`lgbm-model`: Tree + GBDT model-text I/O, `%g` formatter, predict parity), Phase 4 (`lgbm-compute`: Backend trait + histogram/split/partition/subtract kernels, bit-exact on the cubecl-cpu f64 anchor, best-effort cubecl-hip f32 on gfx1100), Phase 5 (`lgbm-treelearner`: histogram serial tree learner + split finding, bit-exact vs `lib_lightgbm` 4.6 on both corpora), Phase 6 (`lgbm-objective`/`lgbm-metric`/`lgbm-boosting`/`lgbm`: GBDT spine + 6 core objectives + 7 metrics + bagging + early stopping + Rust-native API), Phase 7 (parity-completing variants — GOSS/DART/RF, categorical, full objective/metric set, ranking, TreeSHAP, monotone/CEGB, refit/importance), and Phase 8 (`lgbm-python`: PyO3 + numpy + sklearn bindings) — all backed by the `oracle-harness` + `xtask` C++ golden-capture pipeline. **Post-v1.0 (out-of-milestone quick tasks):** a GPU/CPU train-speed perf campaign (CPU histogram-build wins shipped bit-exact; GPU kernel = ROCm-parity-not-speed) and an opt-in quantized-training mode (Phase 10 dir → v2 QNT-01).
- **Key C++ subsystems to port:** `boosting/` (GBDT/DART/RF/GOSS), `treelearner/` (histograms, split finding, gradient discretizer), `io/` (Dataset, Bin, FeatureGroup, Metadata), `objective/`, `metric/`, model text serialization.
- **Numerical fidelity is the hardest part:** even at `f32` / ~1e-6, floating-point reductions are non-associative, so matching the C++ reference on ROCm vs a C++ CPU reference still requires care in binning and accumulation — not an afterthought. The target is single-precision parity, matching the C++ `float` defaults rather than a double-precision 1e-12 bound.

## Constraints

- **Tech stack**: Pure Rust, Cargo workspace, `cubecl` for compute — no raw CUDA/OpenCL. Use latest available crate versions.
- **Compatibility**: 100% behavioral compatibility with C++ LightGBM for in-scope APIs, configs, and internal specifications (binning, split logic, model format).
- **Numerical**: `f32` (single-precision) data types end-to-end, matching C++ `score_t`/`label_t` = `float` defaults; absolute output difference ≤ ~1e-6 vs C++ reference on **both** CPU and ROCm backends.
- **Hardware**: Tests validated on a **local ROCm GPU**; CubeCL `Plane` API used for warp-level ops.
- **Backends**: CPU and ROCm must be switchable (Cargo features and/or runtime configuration).
- **Error handling**: `thiserror` for structured domain errors at library boundaries; `anyhow` for ergonomic propagation in app/high-level layers.
- **Bindings**: Python interface must mirror the official `lightgbm` package API surface.

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| v1 = full single-machine parity (no distributed) | Cover all boosting types/objectives/metrics while deferring the large distributed surface | ✓ Good — v1.0 shipped 69/69 reqs; distributed cleanly deferred |
| `f32` single-precision end-to-end + ~1e-6 oracle on **every** backend incl. ROCm | Matches the C++ reference defaults (`score_t`/`label_t` = `float`); 1e-12 is unachievable/meaningless against an f32 reference, and f32 is the most faithful baseline | ✓ Good — held across all 8 phases; cpu f64 anchor went bit-exact vs real lib_lightgbm 4.6 |
| Standard `f32` accumulations (drop integer-quantized histograms) | At f32 / ~1e-6 the integer-quantization complexity buys nothing; standard f32 reductions keep the CubeCL CPU/ROCm path simple | ✓ Good — kept the kernel path simple; opt-in quantized mode later added separately (v2 QNT-01) without disturbing the default |
| Rust-native API + Python bindings (no C ABI / CLI in v1) | Covers v1 consumers without the C-ABI/CLI surface | ✓ Good — Rust facade + PyO3/sklearn shipped (Phases 6, 8) |
| CubeCL `Plane` API for CUDA warp operations | Project mandate; portable across CPU/ROCm without raw CUDA | ✓ Good — capability-gated Plane path runs on real gfx1100 (Phase 4) |
| Cargo workspace with crate-per-responsibility | Maintainability, loose coupling, clear separation of concerns | ✓ Good — 11 crates; CMP-01 containment kept CubeCL churn behind one boundary |
| `thiserror` at boundaries, `anyhow` at app layer | Precise error matching for the library, ergonomic propagation for apps/tests | ✓ Good — typed domain errors enabled honest typed-rejects (e.g. unsupported-config deferrals) |
| GPU is ROCm-parity, not a speed win on gfx1100 | Faithful CubeCL hist kernel benchmarked ~5.4× slower than the multi-threaded CPU anchor; hist-build levers explored + closed (row-partition shipped; register-batch/packing/16-bit null) | ⚠️ Revisit — open question being scoped: locate the GPU large-data bottleneck before deciding GPU-vs-CPU routing (post-v1.0) |

## Evolution

This document evolves at phase transitions and milestone boundaries.

**After each phase transition** (via `/gsd-transition`):
1. Requirements invalidated? → Move to Out of Scope with reason
2. Requirements validated? → Move to Validated with phase reference
3. New requirements emerged? → Add to Active
4. Decisions to log? → Add to Key Decisions
5. "What This Is" still accurate? → Update if drifted

**After each milestone** (via `/gsd-complete-milestone`):
1. Full review of all sections
2. Core Value check — still the right priority?
3. Audit Out of Scope — reasons still valid?
4. Update Context with current state

---
*Last updated: 2026-06-29 — Milestone v1.1 REWRITTEN as "CUDA On-Device Training Backend", re-derived subsystem-by-subsystem from `docs/cuda-kernel-design.md` (full on-device pipeline, not just the tree learner); scope expanded from the on-device growth-loop slice to the entire single-GPU CUDA training path (shared primitives → dataset → objectives → leaf-splits → histogram → best-split → partition → tree I/O → score updater → metrics → structs/RNG → perf DoD). On-device quantized training deferred to v2 (ODL-13). Phases renumber from 14 (post-v1.0 perf phases 09–13 preserved; prior Phase 14/15 planning artifacts cleared and re-rolled into the new roadmap — the ODL-01/02 seam + oracle code remain in git). v1.0 milestone (2026-06-21): full single-machine C++ LightGBM parity shipped (8/8 phases, 55 plans, 69/69 v1 requirements, milestone audit PASSED).*
