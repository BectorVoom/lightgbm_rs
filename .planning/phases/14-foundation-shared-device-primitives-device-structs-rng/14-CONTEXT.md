# Phase 14: Foundation — Shared Device Primitives + Device Structs/RNG - Context

**Gathered:** 2026-06-29
**Status:** Ready for planning

<domain>
## Phase Boundary

Port and validate the reusable CubeCL **device primitives** (§2.4) and the device
**split-record + RNG** (§15) that every later on-device subsystem builds on, and
**re-establish/extend** the already-in-git on-device growth seam + anchor-pinned oracle.
Everything is **additive and off by default** behind `LGBM_CUDA_ON_DEVICE`.

**Delivers (ODL-01, ODL-02):**
- Shared device primitives as reusable CubeCL kernels: block + multi-kernel global
  **prefix-sum** (inclusive/exclusive), **shuffle reductions** (sum/max/min, dot-product),
  **bitonic argsort** (index-only, never moves values), **weighted/unweighted percentile**.
- A CubeCL-safe, **pre-allocated** device split-record (`CUDASplitInfo` analog — NO
  per-split in-kernel device alloc) and a **`CUDARandom` LCG** with a bit-identical stream
  to the host `Random`.
- The seam (`grow_tree_on_device`, `on_device_growth_supported()`, `LeafPartitionLayout`,
  `assert_on_device_tree_matches_cpu_anchor`) re-established/extended.

**Explicitly NOT in this phase** (forced by the seam-boundary decision D-10):
histogram build/subtract (Phase 16), best-split finding (Phase 17), data partition / tree
mutation / prediction (Phase 18), objectives/score/metrics (19–20), the end-to-end driver
that actually grows a tree on-device (Phase 21), categorical coverage (Phase 22). No
on-device *growth* is wired this phase.

</domain>

<decisions>
## Implementation Decisions

### Primitive scope this phase (ODL-01)
- **D-01:** Build **grow-loop subset at full depth + anchor-pinned skeletons for the rest.**
  Build at full (multi-block/global where the subsystem needs it) depth the primitives the
  numerical grow-loop (Phases 15–18) actually consumes — **prefix-sum** (inclusive/exclusive,
  block + multi-kernel global) and **shuffle reductions** (sum/max/min, dot-product), plus
  **single-block bitonic argsort**.
- **D-02:** **Percentile** (weighted/unweighted), **multi-block / `…Global` argsort**, and
  **`BitonicArgSortItems` (per-query ranking sort)** are ported this phase as **anchor-pinned
  skeletons** — correct, tested, but finalized/hardened by their first real consumer
  (percentile + items → Phase 19 objectives/ranking; multi-block argsort → Phase 19/22).
  ODL-01's literal "all primitives exist" is satisfied by the skeletons; depth follows demand.
- **Rationale:** honors §17 "port these first" without front-loading YAGNI depth onto a
  foundation phase whose downstream consumers may refine the exact signatures.

### Primitive & RNG verification anchor (ODL-01, ODL-02)
- **D-03:** **Capture C++ `lib_lightgbm` golden fixtures** for the numeric primitives, via a
  **thin C++ test harness** that launches each `__device__` primitive (`ShufflePrefixSum`,
  `ShuffleReduce*`, `BitonicArgSort*`, `PercentileDevice`) on real `lib_lightgbm` and dumps
  outputs as committed fixtures. These `__device__` helpers are not host-callable standalone,
  so a small CUDA driver wrapping each kernel is required.
  - Index-only ops (argsort) assert the **permutation** bit-exact; numeric ops assert
    bit-exact on the cpu f64 anchor and ~1e-6 for ROCm/CUDA f32.
- **D-04:** The **`CUDARandom` LCG** is pinned against the **existing host `Random`**
  (`crates/lgbm-core/src/random.rs`, already C++-bit-exact `214013·x+2531011`) — NOT a new
  C++ capture. Assert the device stream reproduces `RandInt16` / `RandInt32` / `NextFloat`
  bit-for-bit against the host draw sequence. (C++ capture is reserved for the numeric
  primitives that have no existing Rust reference.)

### Device split-record representation (ODL-02)
- **D-05:** **Struct-of-Arrays (SoA), host pre-allocated, numeric fields + categorical
  reserved.** One pre-allocated buffer per field sized `[num_leaf_slots]`, allocated **once
  outside the grow loop** (`client.empty` / `empty_tensor`) and **reused/indexed by
  leaf-slot** across every split — the canonical CubeCL pre-allocation pattern (see canonical
  refs). This is the resident record the §8 argmax reductions copy between leaf slots.
- **D-06:** Numeric fields now: `is_valid`, `leaf_index`, `gain`, `inner_feature_index`,
  `threshold` (u32), `default_left`, per-side `{left,right}_sum_gradients/_sum_hessians`
  (f64), `_count` (i32), `_gain`, `_value` (f64), and the quantized
  `_sum_of_gradients_hessians` (i64). The **categorical** field buffers (`num_cat_threshold`,
  `cat_threshold`, `cat_threshold_real`) are **pre-allocated reserved** now so **Phase 22
  fills, not restructures** them.
- **D-07:** The small **8-int / 16-int packed readback packet** the design doc copies back
  per split (§8/§9) is a **separate** concern from the resident record — it is the host-read
  transfer surface, and its wiring lands with its consumer (best-split = Phase 17, partition =
  Phase 18). Phase 14 defines the resident record; it does not need the readback packet live.
- **D-08:** No per-split / per-`SplitInfo` device `cudaMalloc` analog anywhere — the C++
  `AllocateCatVectorsKernel` per-split alloc is exactly the hot-loop allocation the CubeCL
  pre-allocation guidance eliminates. Bitset/cat slabs (Phase 22) will also be pre-allocated.

### Seam wiring boundary
- **D-09:** **Strict no-op seam.** Phase 14 ships primitives + split-record + RNG + extended
  oracle ONLY. `on_device_growth_supported()` stays **false**; `grow_tree_on_device(..)`
  returns **`Ok(None)`**; no histogram/split/growth kernels. Real on-device growth begins at
  Phase 21. The Slice-0 no-op seam tests (learner_parity.rs) must remain green.
- **D-10:** Anchor discipline (carried forward, hard): anchor-pin to the **cubecl-cpu f64
  fold**, tie-aware `default_left`, **never** GPU-vs-GPU (def-f8u-01). Leaf/score numerics
  within ~1e-5 f32 envelope; structure bit-exact.
- **D-11:** `LGBM_CUDA_ON_DEVICE` OFF by default; CPU / ROCm / existing-host-CUDA paths
  **byte-unchanged**; the full merge gate (`raw_bin_train_matches_cpp_golden`,
  `learner_parity`, lgbm/treelearner/compute suites) green and unchanged.

### Claude's Discretion
- Exact CubeCL module placement of the new primitives (likely a new
  `crates/lgbm-compute/src/kernels/primitives.rs`) and whether they share `#[cube]` source
  with any CPU path — but note: cubecl-cpu lost to native on the CPU hot paths (memory:
  `unified-cpu-gpu-kernels-pref`), so the CPU anchor stays a plain serial f64 reference; these
  device primitives are GPU-side, anchored against the C++ fixtures (D-03).
- cubecl-0.10 gotcha handling baked into the primitive design (research, not discussion):
  no global barrier → multi-kernel global prefix-sum; `Atomic<i64>` broken; `wrapping_add`
  not an intrinsic; **plane-sum ≤ plane width** → 256-bin within-feature scan needs a
  segmented LDS block-scan; `launch_unchecked` unsafe.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Port-source design reference (READ FIRST)
- `docs/cuda-kernel-design.md` §2.4 — Shared device primitives (prefix-sum, reductions,
  bitonic argsort, percentile); the 16 underlying `__global__` kernels and their host
  multi-kernel wrappers; tunables (`GLOBAL_PREFIX_SUM_BLOCK_SIZE=1024`,
  `BITONIC_SORT_NUM_ELEMENTS=1024`, `BITONIC_SORT_DEPTH=11`).
- `docs/cuda-kernel-design.md` §15 — Device structs & RNG: `CUDASplitInfo` field list +
  deep-copy `operator=`; `CUDARandom` LCG (`x=214013·x+2531011`, `RandInt16`/`RandInt32`/
  `NextFloat`).
- `docs/cuda-kernel-design.md` §17 — Port considerations: "port these primitives first";
  atomic-ordering nondeterminism / f64 anchor; subtraction-trick & most-freq-bin as
  correctness; template-flag → comptime mapping.
- `.planning/REFERENCE_MANIFEST.md` — the v1.1 C++ port-source map (58 files, 81 kernels)
  and CUDA-support boundaries.

### CubeCL API
- `/home/user/Documents/workspace/cubecl_manual/manual/cubecl/13_memory_preallocation.md` —
  **Host-side pre-allocation** (`client.empty` / `empty_tensor` once, outside the hot loop,
  reused across launches) and **kernel-side `SharedMemory<T>`** — the canonical pattern the
  pre-allocated split-record (D-05/D-08) and the primitives' block scratch follow.

### Existing code to extend (already in git — DO NOT rebuild)
- `crates/lgbm-core/src/random.rs` — host `Random` LCG (C++-bit-exact); the CUDARandom parity
  reference (D-04).
- `crates/lgbm-compute/src/lib.rs` — `Backend::grow_tree_on_device` (~1272/2207),
  `on_device_growth_supported` (~1239), trait defaults (no-op seam, D-09).
- `crates/lgbm-treelearner/src/learner.rs` — `cuda_on_device_env()` (~444),
  `on_device_eligible` wiring (~488/705).
- `crates/lgbm-dataset/src/dataset.rs` — `LeafPartitionLayout` (~88).
- `crates/oracle-harness/tests/learner_parity.rs` — `assert_on_device_tree_matches_cpu_anchor`
  (~2166) + Slice-0 no-op seam tests (~2415–2479) that must stay green.

### Reused shipped kernels (foundations the primitives sit beside)
- Phase 11 u64 fixed-point build kernel, Phase 12 sibling co-pack scan, Phase 13 autotune —
  see `.claude/skills/spike-findings-lightgbm_rs/` and `.planning/ROADMAP.md` Post-v1.0.

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- **Host `Random`** (`lgbm-core/src/random.rs`): already C++-bit-exact; serves directly as the
  CUDARandom parity oracle — no new fixture build needed for RNG.
- **On-device seam** (`lgbm-compute`, `lgbm-treelearner`, `lgbm-dataset`): exists from cleared
  Phase-14/15 work; this phase extends the oracle, keeps the discriminator false.
- **Existing embedded scans** (`kernels/histogram.rs` `build_fix_scan*`, `plane_sum`): prior art
  for warp-shuffle/plane patterns and the plane-width ceiling, informing the standalone
  prefix-sum/reduction primitives (but these are fused, not the reusable standalone primitives
  ODL-01 wants).

### Established Patterns
- **Anchor-pin to cpu f64 fold, never GPU-vs-GPU** (def-f8u-01) — applies to every numeric
  primitive that carries output.
- **Additive, env-gated, byte-unchanged default path** — the merge-gate discipline from every
  v1.1 phase.

### Integration Points
- New primitives live in `lgbm-compute` (likely `src/kernels/primitives.rs`); the C++ fixture
  harness wraps real `lib_lightgbm` `__device__` helpers and dumps committed goldens consumed
  by `oracle-harness` (or a compute-crate test).

</code_context>

<specifics>
## Specific Ideas

- The C++ fixture harness must wrap each `__device__` primitive in a launchable `__global__`
  shim (they are not host-callable as-is) — call out as the first research task for the
  verify-anchor work.
- `CUDASplitInfo`'s deep-copy `operator=` (whole-record copy between leaf slots during argmax)
  must be reproducible with the SoA slab: a "copy leaf-slot a → leaf-slot b" device op across
  all field buffers.

</specifics>

<deferred>
## Deferred Ideas

- **Percentile / multi-block argsort / ranking item-sort depth-hardening** → Phase 19
  (objectives/ranking) and Phase 22 (categorical) finalize the skeletons (D-02).
- **8-int / 16-int packed split-readback packet wiring** → Phase 17 (best-split) / Phase 18
  (partition), where it has a consumer (D-07).
- **Categorical cat_threshold slab fill** → Phase 22 (buffers reserved now, D-06).
- **Quantized/discretized integer histogram & split path** (§4, §7.3, §8.1 inner) → v2
  (QGD-02), out of v1.1 scope entirely.
- **Any actual on-device tree growth** → Phase 21 (D-09).

</deferred>

---

*Phase: 14-foundation-shared-device-primitives-device-structs-rng*
*Context gathered: 2026-06-29*
