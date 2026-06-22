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
- `11-gpu-fixedpoint-int-atomics` — **scoping (spike-validated, ready to plan).** Replace the ROCm histogram BUILD's f32 atomics with wide fixed-point u64 (S=2^30): ~1.3–1.7× faster (wide large-leaves) + ~3600× more accurate + deterministic, within the ~1e-6 gate. Validated by spikes 018/019 (research Q2 / finding #3). Revives the ROCm path as a speed+quality lever (does NOT change CPU routing). SPEC: `phases/11-gpu-fixedpoint-int-atomics/SPEC.md`.

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
