# Roadmap: LightGBM-rs

## Overview

A pure-Rust, parity-faithful port of Microsoft LightGBM on a CubeCL CPU/ROCm backend, built bottom-up along a dependency-forced spine so numerical fidelity is provable at every layer. Data types are `f32` (single-precision) end-to-end to match the C++ reference defaults, and the oracle tolerance is ~1e-6 absolute.

## Milestones

- ✅ **v1.0 Full Single-Machine Parity** — Phases 1–8 (shipped 2026-06-21)
- 🚧 **v1.1 GPU Training-Speed: CUDA On-Device Tree Learner** — Phases 14–19 (roadmapped 2026-06-28)

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

- `14-cuda-ondevice-tree-learner` — **SUPERSEDED → scoped into milestone v1.1.** This
  provisional stub (spike-validated 051–054, milestone-sized) has been scoped into the
  **Milestone v1.1** roadmap below as Phases 14–19 (anchor-gated vertical slices). Its content
  IS the milestone goal. The phase-14 dir will be (re)used by the v1.1 Slice-0 scaffold phase.
  See [`## Milestone v1.1 — GPU Training-Speed: CUDA On-Device Tree Learner`](#milestone-v11--gpu-training-speed-cuda-on-device-tree-learner) below.

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

---

## Milestone v1.1 — GPU Training-Speed: CUDA On-Device Tree Learner

**Goal:** Close the architectural GPU training-speed gap vs official LightGBM by growing the whole
tree on-device — mirroring `CUDASingleGPUTreeLearner` — instead of the current host-driven per-leaf
loop that issues ~8,570 small serial launches per 100-tree train (~86/tree). Spikes 051–054
(real-NVIDIA, Kaggle) refuted every cheap GPU-histogram lever (occupancy/fusion/sync); the on-device
learner is the one remaining architectural lever.

**Structure:** ~6 **anchor-gated vertical slices**. Each phase grows a real tree, returns the same
`(Tree, DataPartition)` the boosting loop already consumes, passes the **anchor-pinned tie-aware**
parity gate, and ships **default-off** behind `LGBM_CUDA_ON_DEVICE`. Each GPU slice carries a Kaggle
`device_launches` checkpoint. Build order is dependency-forced (partition ← pool ← subtraction-trick
rotation; cross-leaf reduce ← per-leaf scan); Slice 0 (oracle-before-kernels) is first, the
perf/default-on rollout (the DoD) is last.

**Non-negotiables (threaded into every phase):**

- The **cubecl-cpu f64 fold stays the bit-exact merge gate** — full suite green on every change.
- CUDA/ROCm parity is **~1e-6 ANCHOR-PINNED to the cpu f64 anchor, never GPU-vs-GPU** (two
  nondeterministic f32 atomic paths compared to each other are flaky by construction, def-f8u-01).

- **NO f64 per-row hot loops** in new CUDA kernels (consumer-NVIDIA f64 = 1/32 f32, measured 5.4×
  regression spike-052) — keep the u64 fixed-point build. f64 is permitted only in scalar/storage
  gain math where the reference uses it.

- **Additive feature-gating** protects CPU + ROCm + the existing host-CUDA path (all byte-unchanged
  until `LGBM_CUDA_ON_DEVICE=1`). Backend discriminators are default-false trait methods on one
  backend, never a global switch.

### Phases (summary checklist)

- [x] **Phase 14: Scaffold + Oracle (Slice 0)** — Additive on-device seam + anchor-pinned tie-aware oracle, zero behavior change. Isolates wiring risk from kernel risk. (completed 2026-06-28)
- [ ] **Phase 15: Minimal On-Device Growth (Slice 1)** — Thinnest continuous-feature tree grown end-to-end on real CUDA via few large launches; `hist_t**` subtraction-trick rotation; u64 fixed-point / no-f64 kernel constraint.
- [ ] **Phase 16: On-Device Frontier Best-Split (Slice 2)** — Cross-leaf best-split selection on-device (removes per-leaf scan readbacks); tie-aware `default_left` assert lands here.
- [ ] **Phase 17: On-Device Data Partition (Slice 3)** — On-device row partition + leaf-index update (Split kernel); the full single-GPU learner mirror.
- [ ] **Phase 18: Feature Coverage** — Categorical splits, bagging/GOSS subsampling, and on-device score update — each anchor-pinned.
- [ ] **Phase 19: Perf-Validation + Default-On Rollout (DoD)** — Kaggle A/B (`device_launches` + ratio); flip default-ON for CUDA contingent on parity + not-slower; host fallback retained.

### Phase Details

#### Phase 14: Scaffold + Oracle (Slice 0)

**Goal**: The on-device growth seam and its anchor-pinned oracle exist with ZERO behavior change — wiring risk is isolated from kernel risk, and the merge gate is proven before any kernel is written.
**Depends on**: v1.0 (Phases 4/5 compute + tree-learner) + post-v1.0 resident-pool work (Phases 11–13)
**Requirements**: ODL-01, ODL-02
**Success Criteria** (what must be TRUE):

  1. `LGBM_CUDA_ON_DEVICE` is OFF by default; CPU, ROCm, and the existing host-CUDA path grow byte-identical trees to before — the full bit-exact merge gate (`raw_bin_train_matches_cpp_golden`, `learner_parity`, lgbm/treelearner/compute suites) is green and unchanged.
  2. An additive `Backend::grow_tree_on_device` method + default-false `on_device_growth_supported()` discriminator exist, routed by a decide-once-at-top early-return fork in `SerialTreeLearner::train_inner`; the `GpuBackend<R>` override still returns the typed error/no-op so the default path is untouched.
  3. An `assert_on_device_tree_matches_cpu_anchor` oracle scaffold exists that pins tree STRUCTURE to the cpu f64 anchor (tie-aware `default_left`) with leaf values within a ~1e-5 f32 envelope — present BEFORE any kernel, never comparing two GPU paths to each other.

**Plans**: 3/3 plans complete
**Wave 1**

- [x] 14-01-PLAN.md — LeafPartitionLayout payload + Backend grow_tree_on_device seam & discriminator (no-op)

**Wave 2** *(blocked on Wave 1 completion)*

- [x] 14-02-PLAN.md — cuda_on_device_env + on_device_eligible cache + train_inner routing fork + DataPartition::from_payload

**Wave 3** *(blocked on Wave 2 completion)*

- [x] 14-03-PLAN.md — tie-aware assert_on_device_tree_matches_cpu_anchor + live host-fallback oracle (SC#3) + seam no-op test (SC#2)

**Cross-cutting constraints:**

- The full merge gate is green AND byte-unchanged with LGBM_CUDA_ON_DEVICE unset.

**Notes**: Pure additive-discriminator wiring (the established `prefers_host_partition` / `resident_eligible` idiom); no new compute, no new Cargo feature. Bake the cubecl-0.10-gotcha checklist (no global barrier, `Atomic<i64>` broken, `wrapping_add` not an intrinsic, plane-sum ≤ plane width, `launch_unchecked` unsafe) into this slice. Merge gate is the hard gate throughout. The pre-on-device CUDA baseline re-measured 2026-06-29 on a Kaggle Tesla T4 = official LightGBM ~4.46× faster @50f cold (3.36 s vs lgb-rs 14.98 s; 500k×50, 100 trees); Phase 14 is perf-neutral (no kernel), so this is the baseline Slice 1+ must beat.

#### Phase 15: Minimal On-Device Growth (Slice 1)

**Goal**: The thinnest end-to-end proof that on-device growth works on real CUDA — a continuous-feature tree grows entirely on-device via a few large launches, the per-node launch chain collapses, and the result reconstitutes into the same `(Tree, DataPartition)` the boosting loop consumes.
**Depends on**: Phase 14
**Requirements**: ODL-03, ODL-06, ODL-07
**Success Criteria** (what must be TRUE):

  1. With `LGBM_CUDA_ON_DEVICE=1`, a small (`num_leaves` ≤ 8) continuous-feature tree grows end-to-end on-device — resident build → subtract → best-split frontier driven by a few large launches, reusing the shipped u64-fixed-point build / feature-per-lane scan / sibling co-pack kernels + host partition + host `Tree::split` replay, with ONE readback — and reconstitutes into a valid `(Tree, DataPartition)`.
  2. The on-device tree is anchor-pinned: structure bit-exact to the cpu f64 anchor (tie-aware `default_left`), leaf values within ~1e-5.
  3. The on-device histogram pool implements the subtraction trick via `hist_t**` pointer rotation (larger child inherits the parent buffer; smaller child gets a fresh arena slot) with no bulk histogram copy, and the build-smaller-before-subtract ordering invariant is preserved.
  4. New CUDA kernels keep the u64 fixed-point build with NO f64 per-row hot loops (verified by grep + per-tree-ms not 6×); f64 only in scalar/bin gain math where the reference uses it.
  5. A Kaggle measurement shows `device_launches/tree` drops materially vs the master baseline; CPU merge gate green; CPU/ROCm/host-CUDA paths byte-unchanged.

**Plans**: TBD
**Notes**: Highest-uncertainty slice — the magnitude of the win is genuinely open (the best-first loop still serializes per split). Verify at plan time: cubecl 0.10 `Handle` in-place aliasing vs ping-pong double-buffering for the data→leaf map; batched `client.read(vec![h])` readback semantics on cubecl-cuda (research flag — likely a planning spike).

#### Phase 16: On-Device Frontier Best-Split (Slice 2)

**Goal**: Best-split selection across the full leaf frontier runs on-device (cross-leaf reduce), eliminating the per-leaf scan readbacks and letting the tree grow to production `num_leaves`/`max_depth`.
**Depends on**: Phase 15
**Requirements**: ODL-04
**Success Criteria** (what must be TRUE):

  1. Cross-feature/cross-leaf argmax over the whole leaf frontier runs on-device (resident best-split-per-leaf buffer + 8-int D→H packet); per-leaf scan readbacks are eliminated and the tree grows to production `num_leaves`/`max_depth`.
  2. The tie-aware `default_left` assert lands WITH the selection kernel: a flip is accepted only on a verified f32 tie (same threshold + left_count + f32-equal gains); a flip on any non-tie split still hard-fails; the empty/sparse-default-bin fixtures pass.
  3. Structure anchor-pinned to the cpu f64 anchor, leaf values within ~1e-5; CPU merge gate green; CPU/ROCm/host-CUDA byte-unchanged.
  4. A Kaggle `device_launches` checkpoint shows a further reduction vs Phase 15.

**Plans**: TBD
**Notes**: The 256-bin within-feature scan needs a segmented LDS block-scan (plane-sum caps at plane width 32/64 ≪ 256) — net-new kernel work, not a reuse (research flag). Do NOT defer the tie-aware assert.

#### Phase 17: On-Device Data Partition (Slice 3)

**Goal**: Move row routing + leaf-index update on-device (the Split kernel + `SplitTreeStructureKernel`-equivalent), eliminating the host partition round-trip — the part that pays off on discrete PCIe NVIDIA. This is the full `CUDASingleGPUTreeLearner` mirror.
**Depends on**: Phase 16
**Requirements**: ODL-05
**Success Criteria** (what must be TRUE):

  1. Data partition + leaf-index update runs on-device (resident `row_leaf` updated per split via stable block-prefix-sum scatter + on-device child seeding / smaller-larger pick), with only a single small scalar readback per split (16-int packet) and one final `row_leaf` readback.
  2. The build-smaller-before-subtract ordering invariant is preserved and asserted (the parent histogram is fully built before any child subtract reads it — the 8aed100-class bug guard).
  3. Structure anchor-pinned to the cpu f64 anchor, leaf values within ~1e-5; CPU merge gate green; the host-partition round-trip is removed on the CUDA path while ROCm keeps its shipped host-partition path.
  4. A Kaggle `device_launches` checkpoint shows the per-split launch count at/near O(depth)/tree.

**Plans**: TBD
**Notes**: Clone the shipped `LGBM_RESIDENT_FORCE` size-gate + default-off precedent for routing. Partition placement nuance (spike-035): the round-trip is pure overhead on shared memory (ROCm stays host-side); the payoff is on discrete PCIe — measure on Kaggle.

#### Phase 18: Feature Coverage

**Goal**: Extend on-device growth beyond the pure continuous spine to the remaining table-stakes feature surface — categorical splits, bagging/GOSS subsampling, and on-device score update — each independently anchor-pinned.
**Depends on**: Phase 17
**Requirements**: ODL-08, ODL-09, ODL-10
**Success Criteria** (what must be TRUE):

  1. On-device growth handles categorical splits faithfully (anchor-pinned) via a CubeCL-compatible PRE-ALLOCATED threshold/bitset representation — NOT the reference's per-`SplitInfo` device alloc.
  2. On-device growth supports bagging / GOSS row subsampling, anchor-pinned to the host bagging RNG draw SEQUENCE.
  3. Score/prediction update runs on-device (replacing the host `add_prediction_to_score` scatter), within the ~1e-6 anchor envelope.
  4. Each feature is individually anchor-gated; the CPU f64 merge gate stays green throughout.

**Plans**: TBD
**Notes**: Each capability is a separable addition on the proven Slice-0..3 core — plannable as parallel sub-slices, each with its own anchor gate. `select_features_by_node_` (interaction constraints / feature_fraction_bynode) stays gated off by default.

#### Phase 19: Perf-Validation + Default-On Rollout (DoD)

**Goal**: Measure the win on real CUDA and make the on-device learner the DEFAULT CUDA tree-learner path — contingent on parity AND not-slower — with the host path retained as an off-switch. This is the milestone Definition of Done.
**Depends on**: Phase 18 (and the per-slice Kaggle checkpoints from Phases 15–17)
**Requirements**: ODL-11, ODL-12
**Success Criteria** (what must be TRUE):

  1. A real-CUDA Kaggle A/B harness measures the on-device path's `device_launches` and the lgb_rs/official wall-clock ratio at 500k×50 AND a wide shape (in-session A/B deltas, platform stated).
  2. The measured `device_launches` for 100 trees drops BELOW the 8,570/100-trees baseline — the architectural launch-collapse is confirmed, not just resident state.
  3. The on-device learner becomes the DEFAULT CUDA tree-learner path — contingent on anchor-pinned ~1e-6 parity AND not-slower-than-the-current-host-CUDA path on the Kaggle A/B — with `LGBM_CUDA_ON_DEVICE=0` retained as the off-switch fallback.
  4. ROCm + CPU routing stay host-driven / byte-unchanged; the CPU f64 merge gate is green.

**Plans**: TBD
**Notes**: Pure routing/perf, deferred until the win is measured — never auto-engaged before proof (the audit-before-wire value). The improvement magnitude is the genuine empirical unknown (the loop still serializes per split). Default-on flips ONLY where the real-CUDA A/B shows a sign-stable not-slower result (the fused-kernel default-off precedent). Multi-stream overlap is a stretch to spike only if launch-count reduction underdelivers.

### Progress (v1.1)

| Phase | Milestone | Plans Complete | Status | Completed |
|-------|-----------|----------------|--------|-----------|
| 14. Scaffold + Oracle (Slice 0) | v1.1 | 3/3 | Complete   | 2026-06-28 |
| 15. Minimal On-Device Growth (Slice 1) | v1.1 | 0/? | Not started | - |
| 16. On-Device Frontier Best-Split (Slice 2) | v1.1 | 0/? | Not started | - |
| 17. On-Device Data Partition (Slice 3) | v1.1 | 0/? | Not started | - |
| 18. Feature Coverage | v1.1 | 0/? | Not started | - |
| 19. Perf-Validation + Default-On Rollout (DoD) | v1.1 | 0/? | Not started | - |
