# Roadmap: LightGBM-rs

## Overview

A pure-Rust, parity-faithful port of Microsoft LightGBM on a CubeCL CPU/ROCm backend, built bottom-up along a dependency-forced spine so numerical fidelity is provable at every layer. Data types are `f32` (single-precision) end-to-end to match the C++ reference defaults, and the oracle tolerance is ~1e-6 absolute. Milestone v1.1 ports the **full single-GPU CUDA training pipeline on-device** (all training state resident, host only orchestrates), grounded subsystem-by-subsystem in `docs/cuda-kernel-design.md`.

## Milestones

- ✅ **v1.0 Full Single-Machine Parity** — Phases 1–8 (shipped 2026-06-21)
- ✅ **v1.1 CUDA On-Device Training Backend** — Phases 14–23 (shipped 2026-07-03; audit `gaps_found` — 21/23, on-device ships opt-in)
- 📋 **v2 (not yet scoped)** — on-device quantized training + on-device perf to make the CUDA learner default-able

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

<details>
<summary>✅ v1.1 CUDA On-Device Training Backend (Phases 14–23) — SHIPPED 2026-07-03</summary>

Full details archived in [`milestones/v1.1-ROADMAP.md`](milestones/v1.1-ROADMAP.md). Requirements in [`milestones/v1.1-REQUIREMENTS.md`](milestones/v1.1-REQUIREMENTS.md). Audit in [`milestones/v1.1-MILESTONE-AUDIT.md`](milestones/v1.1-MILESTONE-AUDIT.md).

Ported the full LightGBM single-GPU CUDA training pipeline on-device (design-doc §-grounded, `docs/cuda-kernel-design.md`), anchor-pinned bit-exact to the cubecl-cpu f64 fold, additive + off by default behind `LGBM_CUDA_ON_DEVICE`. Milestone audit **`gaps_found`**: 21/23 requirements satisfied; 2 partial (ODL-20/21, both Phase 23, both intentional v2 deferrals) — the real-discrete-CUDA A/B FAILED the not-slower bar (launch-bound), so the default-on flip was correctly WITHHELD; on-device ships opt-in via `LGBM_CUDA_ON_DEVICE=1`, every backend byte-unchanged.

- [x] Phase 14: Foundation — Shared Device Primitives + Structs/RNG (6/6) — 2026-06-29
- [x] Phase 15: On-Device Device Dataset + Row-Subset Gather (5/5) — 2026-06-29
- [x] Phase 16: On-Device Histogram Constructor (5/5) — 2026-07-01
- [x] Phase 17: On-Device Best-Split Finder (5/5) — 2026-07-01
- [x] Phase 18: On-Device Data Partition, Tree Mutation & Prediction (4/4) — 2026-07-01
- [x] Phase 19: On-Device Objectives (5/5) — 2026-07-01
- [x] Phase 20: On-Device Score Updater & Metrics (+ pulled-forward driver, D-01) (6/6) — 2026-07-02
- [x] Phase 21: Harden the On-Device Driver — parity corpus + WR-01 confirmation (3/3) — 2026-07-02
- [x] Phase 22: On-Device Categorical Splits (5/5) — 2026-07-02
- [x] Phase 23: Perf-Validation + Default-On Rollout DoD — A/B FAIL → flip withheld, opt-in (4/4) — 2026-07-03

</details>

### Progress (v1.1)

| Phase | Milestone | Plans Complete | Status | Completed |
|-------|-----------|----------------|--------|-----------|
| 14. Foundation — Shared Device Primitives + Structs/RNG | v1.1 | 6/6 | Complete | 2026-06-29 |
| 15. On-Device Device Dataset + Row-Subset Gather | v1.1 | 5/5 | Complete | 2026-06-29 |
| 16. On-Device Histogram Constructor | v1.1 | 5/5 | Complete | 2026-07-01 |
| 17. On-Device Best-Split Finder | v1.1 | 5/5 | Complete | 2026-07-01 |
| 18. On-Device Data Partition, Tree Mutation & Prediction | v1.1 | 4/4 | Complete | 2026-07-01 |
| 19. On-Device Objectives | v1.1 | 5/5 | Complete | 2026-07-01 |
| 20. On-Device Score Updater & Metrics (+ driver, D-01) | v1.1 | 6/6 | Complete | 2026-07-02 |
| 21. Harden the On-Device Driver + Parity Corpus | v1.1 | 3/3 | Complete | 2026-07-02 |
| 22. On-Device Categorical Splits | v1.1 | 5/5 | Complete | 2026-07-02 |
| 23. Perf-Validation + Default-On Rollout (DoD) | v1.1 | 4/4 | Complete (A/B FAIL → default withheld; ODL-21 deferred v2) | 2026-07-03 |
