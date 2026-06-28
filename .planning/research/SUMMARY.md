# Project Research Summary

**Project:** lightgbm_rs — milestone **v1.1 GPU Training-Speed: CUDA On-Device Tree Learner**
**Domain:** On-device (whole-tree-on-GPU) histogram tree learner in CubeCL — porting LightGBM's `CUDASingleGPUTreeLearner` into the existing pure-Rust `Backend`/`SerialTreeLearner` architecture
**Researched:** 2026-06-28
**Confidence:** HIGH

> Supersedes the stale 2026-06-05 v1.0 project summary (preserved in git history). This is the v1.1 milestone summary.

## Executive Summary

This milestone is a **port of a known reference**, not new compute R&D. Spikes 051-054 (run on real NVIDIA via Kaggle) refuted every cheap GPU-histogram lever — occupancy/`P` tuning, kernel fusion, and sync reduction are all flat-to-catastrophic on real CUDA. The single remaining architectural lever is to stop driving tree growth from the host one leaf at a time (~8,570 small serial launches per 100-tree train, ~86/tree) and instead grow the whole tree on-device, mirroring official LightGBM's `CUDASingleGPUTreeLearner`. The crucial finding from the reference read: that learner is **also host-driven** — the host still runs the per-leaf best-first loop — but each step dispatches a *handful of large kernels over device-resident state* and reads back only two tiny scalar packets (an 8-int "which leaf/feature/threshold won" and a 16-int "child counts/starts/sums"). The win is **granularity, not location**: fewer/larger launches + extended resident state, not a persistent megakernel (which cubecl 0.10 does not support and the reference does not use).

The stack requires **no new crates and no new cubecl capability**: cubecl 0.10.0 (already pinned) provides every primitive needed — `sync_cube`, u64/i64/f32 atomics, `SharedMemory`, plane collectives, runtime-bound loops, and the resident-`Handle` pattern (already proven by `ResidentBins`/`resident_pool`). The work is net-new *device-resident orchestration state* (a `hist_t**` pool with pointer rotation for the subtraction trick, two persistent leaf-splits structs, a persistent per-leaf best-split buffer, a cross-leaf reduce, and an on-device `SplitTreeStructureKernel` that seeds children and picks smaller/larger on-device). The integration seam is **additive and default-off**: a new `Backend::grow_tree_on_device()` method gated by a default-false `on_device_growth_supported()` discriminator, routed by a decide-once-at-top early-return fork in `SerialTreeLearner::train_inner`. This protects the CPU f64 bit-exact anchor, the shipped ROCm host-driven path, and the existing host-CUDA path — all untouched until `LGBM_CUDA_ON_DEVICE=1`.

The non-negotiables are numerical and procedural. The **CPU f64 fold stays the bit-exact merge gate**; the CUDA/ROCm paths are held to ~1e-6 **anchor-pinned to that f64 anchor, never GPU-vs-GPU** (two nondeterministic f32 atomic paths compared to each other are flaky by construction — def-f8u-01). No f64 per-row hot loops (consumer NVIDIA f64 = 1/32 f32; spike-052 measured a **5.4x regression** from one f64-fused kernel) — keep the u64 fixed-point build. The largest open risk is **empirical, not architectural**: the best-first loop still serializes per split, so the magnitude of the win (the closing `lgb_rs/official` ratio) is genuinely unknown and **must be measured on Kaggle** — `device_launches` dropping materially below 8,570 is a first-class DoD metric, not an assumption. Build order is ~5 anchor-gated vertical slices; scope is continuous features / no bagging / no quantization, with categorical, bagging, on-device scoring, and quantization explicit P2/P3 follow-ons.

## Key Findings

### Recommended Stack

cubecl **0.10.0** (already pinned in `Cargo.lock`) is sufficient — no upgrade, no new dependency. The reference architecture is host-driven with per-step launches over device-resident state, so cubecl's lack of a persistent-megakernel / cooperative-grid API and lack of a grid-wide barrier are **non-issues**: a kernel boundary *is* the grid barrier, exactly as the reference uses it. The one capability trap is f64: on CUDA `has_f64 == true`, which would silently route the `has_f64`-keyed `ReducePath` onto the slow f64 anchor kernel — the on-device CUDA build must explicitly select the **u64 fixed-point** integer build.

**Core technologies:**
- `cubecl` 0.10.0 — the compute seam; already provides `sync_cube`, atomics, plane ops, runtime loops, resident `Handle`s — no upgrade required (churns the verified parity surface for zero capability gain).
- `cubecl-cuda` 0.10.0 (`CudaBackend = GpuBackend<CudaRuntime>`) — the real-NVIDIA target; inherits every kernel ROCm validates via the generic `GpuBackend<R>`.
- `cubecl-hip` 0.10.0 (`RocmBackend`) — the local parity gate (spoofed 8-CU APU); bit-exact-to-anchor proofs run here before Kaggle confirms speed.
- `cubecl-cpu` 0.10.0 — the **f64 bit-exact deterministic anchor / hard merge gate** — unchanged, do not touch.

Open items to verify at plan time (do not assume): multi-stream support in cubecl 0.10 (the reference uses 4 streams — treat overlap as a *stretch*, not a dependency); idiomatic batched `client.read(vec![h])` readback of the tiny scalar packets; in-place `Handle` aliasing vs ping-pong double-buffering for the data->leaf map; coarse per-tree method vs per-leaf trait granularity.

### Expected Features

The deliverable is a faithful step-ordered port of `CUDASingleGPUTreeLearner::Train()`: leaf-wise (best-first) growth, `num_leaves-1` splits, with per split a fixed small sequence — build-smaller -> subtract-larger -> per-leaf best-split scan -> cross-leaf best-leaf reduce -> on-device partition + tree-structure kernel. Steady-state per-split host<->device traffic is just **24 ints** (the two tiny packets). The compute kernel bodies (build, subtract, find-best scan, partition) already exist in `lgbm-rs`; what is new is the **on-device orchestration state**.

**Must have (table stakes, P1):**
- On-device row partition (`cuda_data_indices_` + per-leaf slices) via stable block-prefix-sum scatter — without it nothing stays resident.
- On-device histogram arena + `hist_t**` pool with subtraction-trick **pointer rotation** (larger child inherits parent's buffer) — today's pool is host-side.
- Build-smaller / subtract-larger per split (reuses existing kernels).
- Per-leaf best-split scan + **cross-leaf best-leaf reduce** (8-int D->H packet) — the reduce is net-new (host picks best leaf today).
- On-device `SplitTreeStructureKernel` equivalent: child seeding + smaller/larger pick + 16-int host mirror — the structural heart.
- Host tree-structure mirror (`CUDATree` + bin-mapper metadata stays host-resident for `tree->Split` RealThreshold/RealFeature).
- min_data_in_leaf / min_sum_hessian gates + "no positive gain" stop (already in host learner).
- Feature-gated coexistence with host-orchestrated / ROCm / CPU paths.

**Should have (P2, after the core slice proves out):**
- Categorical splits on-device (bitset gen + pre-allocated `cat_threshold`).
- Bagging subset path (inverse leaf map).
- On-device score update / boosting-on-GPU resident scores.
- `select_features_by_node_` (interaction constraints / feature_fraction_bynode).

**Defer (P3 / v2+):**
- Quantized-gradient on-device path (`use_quantized_grad`, int16/u64) — large surface; maps to v2 QNT-01 and the shipped CPU-only Phase-10 mode.
- `gpu_use_dp` f64-accumulation accuracy mode — the CPU anchor already covers the f64 reference.
- Refit / `FitByExistingTree` / leaf renew on GPU.
- Multi-stream overlap of the 4 reference streams — measure first; sync has no headroom (spike-052).

**Anti-features (do NOT port):** f64 per-row hot-loop kernels (5.4x regression), per-`SplitInfo` device heap alloc, forcing `build_fix_scan` fusion on CUDA, occupancy/`P` tuning for CUDA, replacing the host/ROCm/CPU paths, multi-stream micro-optimization first.

### Architecture Approach

The integration is an **additive Backend seam, not a new learner trait**. A new `Backend::grow_tree_on_device()` method gated by a default-false `on_device_growth_supported()` discriminator (overridden only on `GpuBackend<R>`, opt-in via an `on_device_enabled` field + `LGBM_CUDA_ON_DEVICE` env) is routed by a decide-once-at-top early-return fork in `SerialTreeLearner::train_inner` — a sibling branch of the existing resident/host fork. A parallel `TreeLearner` trait is rejected: 20+ gbdt.rs call sites name `SerialTreeLearner` concretely, so a learner-level dispatch refactor would blast-radius into the exact CPU/ROCm paths the milestone must protect. The on-device path returns a plain-data, cubecl-free `OnDeviceTreeResult` (node arrays + final `row_leaf`) that the learner reconstitutes into the same `(Tree, DataPartition)` pair the boosting loop already consumes — keeping the boosting<->treelearner boundary byte-unchanged and per-iter scores bit-comparable for free.

**Major components (net-new device-resident state):**
1. `hist_t**` histogram pool with pointer rotation — the resident frontier + subtraction-trick economy.
2. Two persistent `CUDALeafSplitsStruct` (smaller/larger frontier descriptors) seeded on-device.
3. Persistent `cuda_leaf_best_split_info_[num_leaves]` + cross-leaf argmax reduce.
4. `SplitTreeStructureKernel` — on-device child seed + smaller/larger pick + 16-int host mirror.
5. On-device `cuda_data_partition_` (row->leaf, stable scatter) — the true single-GPU partition (later slice; pays off on discrete PCIe, kept host-side on ROCm per spike-035).

Parity is **anchor-pinned, tie-aware, never bit-exact GPU-vs-GPU**: grow the corpus on `CpuBackend` (f64 anchor) and on-device, assert tree *topology* against the anchor with a mandatory tie-aware `default_left` assert (legal f32 near-tie flips accepted), leaf values within a ~1e-5 f32 envelope. The CPU merge gate suite stays the hard gate; real-CUDA validation is Kaggle-only (`boomvector`, reading the max-launches `phase_prof device_launches` dump).

### Critical Pitfalls

1. **Comparing two nondeterministic GPU f32 paths to each other** — flaky by construction (`learner_parity_resident...` failed ~4/6 runs on unchanged master, def-f8u-01). Pin **both** trees to the cpu f64 anchor; build `assert_on_device_tree_matches_cpu_anchor` in **Slice 0, before any kernel**.
2. **f32 reduction-order flips split structure (default_left / equal-gain ties)** — a naive bit-exact assert hard-fails on ~34% of fixtures with empty default bins. Use the **tie-aware assert** (commit 1832206): a flip allowed only on a verified f32 tie (same threshold + left_count + f32-equal gains), non-tie flips still hard-fail. Land it in the same slice as the selection kernel.
3. **f64 hot loops (consumer-NVIDIA 1/32 trap)** — `LGBM_FUSED_FORCE=1` (the f64 fused kernel) was **5.4x WORSE** on real CUDA (spike-052). Keep the **u64 fixed-point build**; audit every new kernel for f64 in gain math / leaf-output / score accumulation / fused build+scan. "No f64 in device hot loops" is a per-slice DoD item.
4. **Trusting spoofed-APU lever signs for real CUDA** — the local "GPU" is a spoofed 8-CU APU that mis-predicted *every* lever in 051-054. **Correctness -> local (cpu anchor + APU ~1e-6); perf/routing -> Kaggle in-session A/B only.** Design correctness to stay local so Kaggle is needed only at perf checkpoints.
5. **"On-device" but still launch-bound** — if the loop still fires a launch per frontier node, `device_launches` stays ~8,570 and the gap isn't closed. Make `device_launches` a **first-class success metric every slice**; batch the whole frontier, don't move state on-device while keeping per-node control.

Plus: cubecl-0.10 gotchas (no global barrier, `Atomic<i64>` broken -> use u64 two's-complement, `wrapping_add` not an intrinsic, `plane_inclusive_sum` capped at plane width << 256-bin -> segmented LDS block-scan, `launch_unchecked` is unsafe); monolithic big-bang port (slice vertically); and breaking the existing CPU/ROCm/host-CUDA paths (feature-gate everything, preserve build-smaller-before-subtract ordering, run the full bit-exact suite every change).

## Implications for Roadmap

Based on research, the milestone should be sequenced as **~5 anchor-gated vertical slices**, each end-to-end (grows a real tree, returns `(Tree, DataPartition)`, passes the anchor-pinned tie-aware gate, ships default-off behind `LGBM_CUDA_ON_DEVICE`), each with a Kaggle launch-count checkpoint. This is a deliberate counter to the monolithic-port pitfall: a one-shot port produces unlocalizable parity failures amid f32 tie noise.

### Phase / Slice 0: Scaffold the seam (no behavior change)
**Rationale:** Isolate wiring risk from kernel risk; build the oracle before any kernel (Pitfall 1).
**Delivers:** `on_device_growth_supported()` (default false) + `grow_tree_on_device()` (default typed error) + `OnDeviceTreeResult` + `on_device_eligible()` + the `train_inner` early-return fork + `reconstitute()` helper + `assert_on_device_tree_matches_cpu_anchor` oracle scaffold + the cubecl-0.10-gotcha checklist.
**Addresses:** the feature-gated coexistence requirement.
**Avoids:** Pitfalls 1, 5-gotchas, 7 (merge gate green; CPU/ROCm/host-CUDA all untouched — eligibility ANDs in a false discriminator).

### Phase / Slice 1: Minimal proving slice (thinnest end-to-end on-device growth)
**Rationale:** Prove the hardest uncertainty — does on-device growth + anchor-pinned parity + result reconstitution actually work on real CUDA, and does the launch chain collapse — at minimum kernel surface.
**Delivers:** `grow_tree_on_device` for the narrowest viable tree (pure numeric spine, small `num_leaves` <=8, build->subtract->best-split frontier resident via a few large launches, **host partition reused** (shipped 027 fused path) + host `Tree::split` replay, ONE readback).
**Uses:** existing u64 fixed-point build + feature-per-lane scan kernels; the resident-`Handle` pattern.
**Implements:** the resident histogram pool + frontier build/subtract/scan.
**Avoids:** Pitfalls 2 (tie-aware assert), 3 (u64 fixed-point), 6 (thin slice). **Proves Pitfall 8:** measured Kaggle `device_launches/tree` drop vs master.

### Phase / Slice 2: On-device best-split across the full frontier
**Rationale:** Remove the per-leaf scan readbacks; grow to production `num_leaves`/`max_depth`.
**Delivers:** cross-feature argmax over the whole leaf frontier on-device (resident best-split-per-leaf + 8-int packet).
**Implements:** components 2-3 (persistent leaf-splits + best-split buffer + cross-leaf reduce).
**Avoids:** Pitfall 2 (the tie-aware `default_left` assert lands here with the selection kernel); Pitfall 5 (segmented LDS block-scan, not bare plane-sum, for 256-bin scans).

### Phase / Slice 3: On-device data partition (true single-GPU learner)
**Rationale:** Eliminate the host partition round-trip — the part that pays off on discrete PCIe NVIDIA (kept host-side on ROCm per spike-035). The full `CUDASingleGPUTreeLearner` mirror.
**Delivers:** resident `row_leaf` updated per split + on-device `SplitTreeStructureKernel`; only ONE readback at end of growth.
**Implements:** components 4-5.
**Avoids:** Pitfall 7 (preserve build-smaller-before-subtract ordering invariant).

### Phase / Slice 4: Kaggle perf-validation + default-on routing + size gate
**Rationale:** Pure routing/perf, deferred until the win is measured — never auto-engaged before proof (the audit-before-wire value).
**Delivers:** Kaggle in-session A/B; `num_data` crossover/size gate from the measurement; flip `on_device_enabled` default for **CUDA only** (ROCm stays host-driven), keep the env off-switch.
**Avoids:** Pitfall 4 (every perf claim is a stated-platform Kaggle A/B). **Milestone DoD:** a measured `device_launches` reduction below 8,570 + a closing `lgb_rs/official` ratio (today 3.90x@50f -> 1.93x@500f).

### Phase Ordering Rationale
- **Dependency-driven:** the histogram pool requires the partition (leaf slices index both arena and rows); the subtraction trick requires the pool-pointer rotation; the cross-leaf reduce requires the per-leaf scan. Slice 0 -> 1 -> 2 -> 3 follows this chain while keeping each slice end-to-end.
- **Risk-front-loaded:** Slice 0 de-risks wiring; Slice 1 attacks the single biggest uncertainty (does the launch collapse + parity reconstitution work on real CUDA) before expanding the resident frontier monotonically.
- **Pitfall-driven:** oracle before kernels (P1), tie-aware assert with the selection kernel (P2), u64 fixed-point from the first build (P3), perf decisions deferred to Kaggle (P4), launch-count gated every slice (P8), feature-gate every slice (P7).

### Research Flags

Phases likely needing deeper research / a spike during planning:
- **Slice 1 (build/orchestration):** verify cubecl 0.10 `Handle` in-place aliasing vs ping-pong double-buffering for the resident data->leaf map; confirm batched `client.read(vec![h])` readback semantics on cubecl-cuda. The on-device kernel decomposition is the milestone's genuine open work (ARCHITECTURE confidence MEDIUM here).
- **Slice 2 (selection):** the 256-bin segmented LDS block-scan (plane-sum caps at plane width) is net-new kernel work, not a reuse.
- **Slice 4 (perf):** the improvement *magnitude* is the genuine empirical unknown — the best-first loop still serializes per split; the win must be measured, not modeled. Multi-stream overlap (if cubecl 0.10 exposes it) is a stretch to spike only if launch-count reduction underdelivers.

Phases with standard / well-documented patterns (skip research-phase):
- **Slice 0 (scaffold):** pure additive-discriminator wiring — the established `prefers_host_partition` / `resident_eligible` idiom, no new compute.
- **Slice 3 partition routing & Slice 4 default-on gating:** clone the shipped `LGBM_RESIDENT_FORCE` size-gate + default-off precedent.

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | HIGH | cubecl 0.10.0 version + capability API verified against `Cargo.lock` and live `runtime::probe_capabilities`; no new deps; primitives in active production use. |
| Features | HIGH | Step-ordered spec read directly from `LightGBM/src/treelearner/cuda/*` with file:line citations; a faithful read of the algorithm to port. |
| Architecture | HIGH (integration) / MEDIUM (kernel decomposition) | Seam/routing/return mechanics read from real code; the on-device kernel decomposition is the milestone's open work to be proven in Slice 1. |
| Pitfalls | HIGH | Grounded in this project's own spikes 015-054, def-f8u-01, hip-split-parity debug, and PROJECT.md non-negotiables — every pitfall already bit this codebase or follows from a committed constraint. |

**Overall confidence:** HIGH

### Gaps to Address
- **Improvement magnitude is unknown** — the loop still serializes per split; the closing `lgb_rs/official` ratio is the genuine empirical open question. Handle: make `device_launches` + the ratio a first-class Kaggle-measured DoD metric, not an assumption (Slice 4 / per-slice checkpoint).
- **cubecl 0.10 multi-stream support** — the reference uses 4 streams; unverified whether cubecl exposes them. Handle: treat overlap as a stretch, not a dependency — launch-count reduction is the proven lever.
- **`Handle` in-place mutation/aliasing rules** — whether a kernel can take the same handle as input+output for the data->leaf map, or whether double-buffering is required. Handle: verify at Slice 1 plan time against cubecl source, not the manual.
- **Per-leaf `Backend` trait vs coarse per-tree method granularity** — planner design decision; the coarse method is lower-risk for keeping the CPU anchor untouched.

## Sources

### Primary (HIGH confidence)
- `LightGBM/src/treelearner/cuda/cuda_single_gpu_tree_learner.cpp` + `cuda_data_partition.{hpp,cpp,cu}` + `cuda_best_split_finder.*` + `cuda_histogram_constructor.*` + `cuda_leaf_splits.hpp` + `cuda_split_info.hpp` — the on-device reference architecture and step-ordered `Train()` spec.
- `crates/lgbm-compute/src/lib.rs` (Backend trait :495, discriminator idioms, `GpuBackend<R>` :2037, `ResidentBins`/`resident_pool`) — the additive seam + resident-Handle pattern.
- `crates/lgbm-treelearner/src/learner.rs` + `resident_pool.rs` — the `train_inner` decide-once fork, `Tree::split` replay, `add_prediction_to_score`, the eligibility/size-gate idiom to clone.
- `crates/lgbm-boosting/src/gbdt.rs:1289` + `score_updater.rs` — the `(Tree, DataPartition)` -> per-row score scatter contract the on-device path must preserve.
- `Cargo.lock` (cubecl 0.10.0 lockstep) + `runtime.rs:108-130` (capability probe) — stack verification.
- `.claude/skills/spike-findings-lightgbm_rs/references/cuda-architectural-launch-bound.md` (spikes 051-054, real-NVIDIA Kaggle) — launch-bound mechanism, f64-5.4x penalty, occupancy/fusion/sync refuted, on-device learner is the one lever.
- `.claude/skills/spike-findings-lightgbm_rs/references/gpu-split-scan-occupancy.md` (spikes 016/021/022) — f32 reorder parity-safe, default_left tie-awareness, plane-sum vs 256-bin limit.
- `.planning/PROJECT.md` — v1.1 milestone goal + non-negotiables.

### Secondary (MEDIUM confidence)
- MEMORY: `def-f8u-01-flaky-resident-hip-test.md` (anchor-pin GPU f32 to cpu anchor, fix d82611b); `hip-split-parity-preexisting-defect.md` (tie-aware assert, commit 1832206); `kaggle-cli-cuda-bench.md`; `gpu-is-spoofed-8cu-apu`; `partition-memory-traffic.md` / spike-035 (host-vs-device partition placement).
- `.planning/spikes/052-cuda-launch-fusion/README.md` — the f64 fused-kernel 5.4x regression evidence + ~0.14ms/sync.

---
*Research completed: 2026-06-28*
*Ready for roadmap: yes*
