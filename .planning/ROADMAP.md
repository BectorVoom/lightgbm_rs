# Roadmap: LightGBM-rs

## Overview

A pure-Rust, parity-faithful port of Microsoft LightGBM on a CubeCL CPU/ROCm backend, built bottom-up along a dependency-forced spine so numerical fidelity is provable at every layer. Data types are `f32` (single-precision) end-to-end to match the C++ reference defaults, and the oracle tolerance is ~1e-6 absolute.

## Milestones

- ✅ **v1.0 Full Single-Machine Parity** — Phases 1–8 (shipped 2026-06-21)

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

Phase directories that exist on disk but were never part of the v1.0 milestone scope. They ran as ad-hoc quick tasks / spikes after v1.0's phases completed. Candidates for a future milestone's roadmap:

- `09-gpu-hist-build-perf` — GPU/CPU training-speed perf campaign (CPU histogram-build wins shipped bit-exact; GPU kernel concluded ROCm-parity-not-speed). No formal VERIFICATION.
- `10-quantized-training` — opt-in approximate quantized-grad training (maps to deferred v2 requirement `QNT-01`). No formal VERIFICATION.
- `11-gpu-fixedpoint-int-atomics` — **planned (3 plans, spike-validated).** Replace the ROCm histogram BUILD's f32 atomics with wide fixed-point u64 (S=2^30): ~1.3–1.7× faster (wide large-leaves) + ~3600× more accurate + deterministic, within the ~1e-6 gate. Validated by spikes 018/019 (research Q2 / finding #3). Revives the ROCm path as a speed+quality lever (does NOT change CPU routing). SPEC: `phases/11-gpu-fixedpoint-int-atomics/SPEC.md`.
  Plans:

  - [x] 11-01-PLAN.md — u64 two's-complement fixed-point resident build kernel + u64->f64 dequant at the fix-compact seam + overflow guard (wave 1)
  - [x] 11-02-PLAN.md — re-pin rocm resident parity to the CPU f64 anchor (tightened) + determinism assert (wave 2)
  - [x] 11-03-PLAN.md — device-time A/B confirming the integer build is not-slower in the wide regime (wave 2)

- `12-gpu-sibling-scan-copack` — **planned (spike-validated).** Wire spike-024's
  batch-sibling-scans co-pack: replace the TWO separate per-sibling resident scan
  launches+readbacks (one per leaf-node, ~59 syncs/tree) with ONE co-packed 2-slot scan
  launch + ONE readback per split (~30 syncs/tree). Isolated ~2.0× on the scan
  launch+readback, bit-exact (each feature's sequential scan unchanged — no spike-016
  reorder); honest e2e ~10–15% at small/medium, ~1.5% wide (per spike-023's scan-sync
  fractions). Does NOT change CPU routing or the wide build path. Validated by spikes
  023 (regime-split attribution) + 024 (isolated A/B). Evidence:
  `crates/lgbm-compute/examples/spike024_sibling_scan_ab.rs`, `.planning/spikes/024-*/`.
  Plans: 3 plans

  - [x] 12-01-PLAN.md — 2-slot co-packed scan kernel + `scan_resident_siblings` backend method + growth-loop reorder (defer smaller scan past subtract, co-pack when resident-scan-eligible) + `LGBM_SIBLING_COPACK` gate (wave 1)
  - [x] 12-02-PLAN.md — oracle `kernel_parity` co-pack cell (co-pack == two scans byte-identical + rocm within ~1e-6 of CPU f64 anchor; cubecl-cpu W=1 byte-identical) + CPU merge-gate green (wave 2)
  - [x] 12-03-PLAN.md — `bench_gpu_vs_cpu` co-pack ON/OFF A/B: `scan_resident` sync count ~halved + e2e not-slower, honest reporting (wave 2)

- `13-gpu-autotune-launch-config` — **planned (spike-validated 037–040).** Replace the
  hand-tuned/env GPU launch-config heuristics with CubeCL runtime autotuning
  (`cubecl::tune`), **default-on for all GPU (rocm) selection**. Autotunes BOTH GPU launch
  knobs: the histogram-BUILD row-partition `P` (was `row_partition_count`, which spike-040
  found under-partitions to P=1 at the production 50-feat width = ~10% slow on the 8-CU APU)
  and the split-SCAN `CubeDim` (was `LGBM_SCAN_CUBEDIM`, spike-021). Wire = a default-on
  rocm backend discriminator + a fresh-output `InputGenerator` (the accumulating build kernel
  corrupts 27× under `CloneInputGenerator`, spike-038) + a `log2(rows)` occupancy-regime
  AutotuneKey (exact-rows keying = a per-leaf tuning storm, spike-039). Selection is the
  spoof-robust axis (relative within-device); ~10% local on the APU, durable payoff =
  portability (self-calibrates on discrete gfx110x / NVIDIA). Does NOT change CPU routing or
  the f64 anchor; stays within the ~1e-6 ROCm parity gate. Validated by spikes 037
  (feasibility + 3 manual-API corrections), 038 (fresh-output correctness), 039 (key
  granularity), 040 (beats the heuristic ~10%). Evidence:
  `.claude/skills/spike-findings-lightgbm_rs/references/gpu-kernel-autotuning.md`,
  `.planning/spikes/037..040/`, `crates/lgbm-compute/examples/spike0{37,38,39,40}_*.rs`.
  Plans: 4 plans

  - [x] 13-01-PLAN.md — autotune foundation: serde→real rocm-gated dep + `kernels::autotune` module (`LaunchKey`, `size_band` log2 bucketer, `autotune_enabled` off-switch, `cache_namespace_id`) + default-on `Backend::prefers_autotune_launch_config` seam (wave 1)
  - [x] 13-02-PLAN.md — autotune the histogram-build row-partition `P`: FreshOutGenerator + `BUILD_TUNER` + PSET, wired into `resident_raw_build_into` with `LGBM_AUTOTUNE=0` heuristic fallback + `LGBM_AUTOTUNE_FORCE_P` parity seam (wave 2)
  - [x] 13-03-PLAN.md — autotune the split-scan `CubeDim` `W`: `SCAN_TUNER` + WSET (CloneInputGenerator), wired into the fused split launcher with `scan_cube_dim()`/`LGBM_SCAN_CUBEDIM` fallback (wave 2)
  - [x] 13-04-PLAN.md — all-PSET/all-WSET oracle parity pinned to the CPU f64 anchor + e2e A/B (autotune ≥ heuristic, recovers P=1) + CPU merge-gate green + honest-bound SUMMARY (wave 3)

### 📋 Next milestone (not yet scoped)

Define via `/gsd-new-milestone`. Candidate themes: GPU large-data perf (locate the bottleneck — see `notes/gpu-large-data-bottleneck-framing.md`), quantized training (QNT-01), linear-tree leaves (LIN-01), text/binary/Arrow ingestion (ING-01..03).

## Progress

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
