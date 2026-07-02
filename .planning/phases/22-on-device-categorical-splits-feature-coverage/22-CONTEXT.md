# Phase 22: On-Device Categorical Splits (Feature Coverage) - Context

**Gathered:** 2026-07-02
**Status:** Ready for planning

<domain>
## Phase Boundary

**This phase delivers:** categorical splits working **end-to-end on the proven
numerical on-device driver** (the Phase 14–21 slice), so a categorical feature
trains on-device with the same anchor-pinned fidelity as the continuous spine.
Concretely, filling the categorical seams already reserved in the codebase:

1. **Bitset construction (§6.3)** — `SetRealThreshold` (inner-bin → real
   category via `categorical_bin_to_value` / `categorical_bin_offsets`) + bitset
   length (`val/32+1` via shuffle-max) + `CUDAConstructBitset` (set bit
   `1<<(val%32)`), materialized into the **pre-allocated** bitset representation
   — NOT the reference's per-`SplitInfo` `cudaMalloc`.
2. **Categorical split evaluation (§8.1)** — both the **one-hot** path (each
   category as a singleton left set, `num_bin ≤ max_cat_to_onehot`) and the
   **many-vs-many** path (sort bins by `grad/(hess+cat_smooth)` via
   `BitonicArgSort`, bidirectional prefix-sweep up to
   `max_num_cat = min(max_cat_threshold, (used_bin+1)/2)`), filling the existing
   `is_categorical`/`is_one_hot` dispatch seam in `best_split.rs`.
3. **Categorical partition membership (§9)** — route rows via
   `CUDAFindInBitset(bitset, bitset_len, bin − min_bin + mfb_offset)`
   (`GenDataToLeftBitVectorKernel_Categorical` /
   `UpdateDataIndexToLeafIndexKernel_Categorical` analogs).
4. **`SplitCategorical` tree mutation (§10)** — write the categorical node
   (`kCategoricalMask` decision-type bit, `num_cat`, extend
   `cat_boundaries`/`cat_boundaries_inner`) and predict correctly through the
   bitset (`FindInBitsetCUDA`).

**Explicitly NOT in this phase (unchanged boundaries):**
- **The numerical spine** stays byte-untouched and anchor-pinned (SC #4). This
  phase is purely additive categorical coverage on top of it.
- **Perf-validation / default-ON rollout DoD** (Kaggle A/B, `device_launches`,
  wall-clock ratio, default flip) → **Phase 23** (ODL-20/ODL-21).
- **Categorical + `use_quantized_grad` on-device** — the reference itself does
  not support it (`asm("trap;")` in the discretized best-split kernel). Honest
  **host-fallback** (see D-06), not an on-device path.

Everything **additive** and gated by `LGBM_CUDA_ON_DEVICE`; CPU / ROCm /
existing-host-CUDA paths stay **byte-unchanged** with the env unset; the hard
merge gate stays green on the DEFAULT (cubecl-cpu f64) lane. Anchored to the
**cubecl-cpu f64 fold**, never GPU-vs-GPU (def-f8u-01).

</domain>

<decisions>
## Implementation Decisions

### Carried forward from Phases 14–21 (LOCKED — do not re-litigate)
- **Anchor = cubecl-cpu f64 fold.** STRUCTURE bit-exact + leaf values within
  ~1e-5 f32 envelope; **never** compare two GPU f32 paths to each other
  (def-f8u-01).
- **Additive + `LGBM_CUDA_ON_DEVICE`-gated.** Env-unset ⇒ CPU / ROCm /
  host-CUDA byte-unchanged; merge gate runs on the default cubecl-cpu lane so
  the categorical structure gate is non-vacuous without ROCm hardware.
- **Pre-allocated bitset slab, zero per-split device alloc** (ODL-02 / ODL-22).
  The C++ `AllocateCatVectorsKernel` per-`SplitInfo` `cudaMalloc` is exactly the
  anti-pattern CubeCL pre-allocation eliminates. Seams already reserved:
  `DeviceSplitInfo` cat slabs (`split_info.rs`), `best_split` categorical
  dispatch, partition/tree categorical stubs.
- **ROCm = best-effort smoke, not the gate** (D-04 from Phase 21). A real-ROCm
  (`cubecl-hip`, f32, ~1e-6) run — if attempted — is pinned to the cpu anchor
  and is informative, not blocking. Full real-hardware validation is Phase 23's
  Kaggle DoD (local GPU is a spoofed 8-CU APU — memory `rocm-gfx1100-available`).

### Parity anchor for categorical (SC #1–#3)
- **D-01: Both real 4.6 goldens AND the cubecl-cpu f64 structure gate.**
  Categorical is a fidelity UPGRADE over the numerical spine: the numerical
  on-device goldens are host re-transcriptions (memory
  `on-device-kernel-goldens-are-retranscriptions`), but **real `lib_lightgbm`
  4.6 categorical goldens already exist** (`cat_onehot`, `cat_manyvsmany` under
  `crates/oracle-harness/tests/fixtures/categorical/`, captured by
  `xtask categorical-oracle-capture`).
  1. Pin the constructed **bitset / decision-type bit / `num_cat` / chosen
     `cat_threshold` (REAL category bitset)** **bit-exact** to those real 4.6
     goldens — proves REFERENCE fidelity of the split math.
  2. Run the on-device categorical tree through the **cubecl-cpu f64 structure
     gate** (extend Phase-21 `learner_parity_on_device_structure_gate`) — proves
     the full device grow loop routes categorical rows correctly, tie-aware on
     `default_left`.
  Rationale: the goldens prove "matches C++", the structure gate proves "the
  device path matches the anchor"; together they close both risks. Leaving the
  goldens unused would waste real reference evidence.

### Categorical eval path scope (SC #2)
- **D-02: Both one-hot AND many-vs-many this phase.** ODL-22 / SC #2 explicitly
  name "one-hot + many-vs-many bitonic-sorted", and BOTH real goldens exist
  (`cat_onehot`, `cat_manyvsmany`) — deferring many-vs-many would leave a golden
  unpinned and ODL-22 partially met. The many-vs-many path **reuses the
  Phase-14 bitonic argsort primitive** (index-only, never moves values) for the
  `grad/(hess+cat_smooth)` bin sort; `cat_l2` added to `l2` in the gain math
  (§8.1).

### `max_cat_threshold` slab sizing (SC #1)
- **D-03: Size the bitset/threshold slab from `config.max_cat_threshold` at
  driver/`DeviceSplitInfo` init.** The reserved `cat_threshold` /
  `cat_threshold_real` slabs are currently a fixed `MAX_CAT_PER_SPLIT = 32` const
  (the C++ `max_cat_threshold` default). Make that width a **runtime value read
  from config once at init** (default 32) so any `max_cat_threshold > 32` config
  is faithful with **no silent truncation** and **no per-split alloc** (ODL-02
  still honored — allocate-once). `MAX_CAT_PER_SPLIT` becomes the default, not a
  hard cap. Explicitly rejected: hard-clamp-to-32 (diverges from the reference
  split → breaks the ~1e-6 parity contract).

### Categorical + quantized-grad interaction (SC #4 / boundary)
- **D-06: Honest host-fallback when categorical features AND `use_quantized_grad`
  are both set.** The CUDA reference's discretized best-split kernel does
  `asm("trap;")` for categorical — it has no on-device categorical+quantized
  path. `on_device_growth_supported()` returns **false** for that combo (routes
  to host), with a one-line log. Mirrors the reference's own non-support and the
  Phase-19 "CUDA-unsupported → host fallback" precedent (ODL bucket in
  REQUIREMENTS.md "Out of scope"). No silent wrong answers; no bespoke
  device path the reference lacks.

### Claude's Discretion
- Kernel geometry / thread-block mapping for the bitset-construction and
  categorical-eval kernels (follow §6.3 / §8.1 launch idioms; cubecl-0.10
  gotcha checklist applies — no global barrier, `Atomic<i64>` broken,
  `wrapping_add` not an intrinsic, plane-sum ≤ plane width, `launch_unchecked`
  unsafe).
- Exact bitset-construction atomic mechanics (the reference uses
  `atomicAdd_system(out + val/32, 1<<(val%32))` into a pre-zeroed `u32*`) — pick
  the CubeCL-safe equivalent that reproduces the same bitset bits.
- Parity fixture parameters (row/feature/category counts, `num_leaves`,
  `max_cat_threshold`/`max_cat_to_onehot` values) — pick the smallest configs
  that provably exercise the one-hot path, the many-vs-many path (needs
  `num_bin > max_cat_to_onehot`), the `max_num_cat` clamp, and predict-through.
- Whether the many-vs-many bitonic sort runs single-block (`BitonicArgSort_1024`
  analog) or the global-memory strided path — pick per the used-bin count of the
  fixtures; the reference's low-VRAM `_GlobalMemory` variant is optional.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Reference design (the algorithm being ported)
- `docs/cuda-kernel-design.md` §6.3 — categorical bitset construction
  (`SetRealThresholdKernel`, `CalcBitsetLenKernel`/`ReduceBlockMaxLen`,
  `CUDAConstructBitsetKernel`).
- `docs/cuda-kernel-design.md` §8.1 — categorical split evaluation
  (`FindBestSplitsForLeafKernelCategoricalInner`: one-hot + many-vs-many bitonic
  sort + bidirectional prefix-sweep; `cat_smooth`/`cat_l2`/`max_cat_threshold`/
  `min_data_per_group`), and §8.3 `AllocateCatVectorsKernel` (the pre-alloc slab
  the port replaces with allocate-once).
- `docs/cuda-kernel-design.md` §9 — categorical partition membership
  (`GenDataToLeftBitVectorKernel_Categorical` /
  `UpdateDataIndexToLeafIndexKernel_Categorical` via `CUDAFindInBitset`).
- `docs/cuda-kernel-design.md` §10 — `SplitCategoricalKernel` (`kCategoricalMask`,
  `num_cat`, `cat_boundaries`/`…_inner`) + `FindInBitsetCUDA` predict lookup.

### Requirements & roadmap
- `.planning/REQUIREMENTS.md` — **ODL-22** (the single requirement this phase
  delivers); "Out of scope" table row for per-`SplitInfo` `cudaMalloc` and for
  CUDA-unsupported combos (grounds D-06 host-fallback).
- `.planning/ROADMAP.md` §"Phase 22" — Goal, the 4 Success Criteria, Notes
  (pre-allocated slab rationale).

### Seams to fill (already in the codebase)
- `crates/lgbm-compute/src/kernels/split_info.rs` — `DeviceSplitInfo` with the
  **reserved** `num_cat_threshold` / `cat_threshold` / `cat_threshold_real`
  slabs and `MAX_CAT_PER_SPLIT = 32`; the D-03 config-sized-width change lands
  here.
- `crates/lgbm-compute/src/kernels/best_split.rs` — the `is_categorical` /
  `is_one_hot` **dispatch seam** (currently returns the `is_valid=false`
  sentinel without running eval); §8.1 categorical eval fills it.
- `crates/lgbm-compute/src/kernels/data_partition.rs` — categorical membership
  routing seam (§9).
- `crates/lgbm-compute/src/kernels/tree.rs` — `SplitCategorical` tree-mutation
  seam (§10); `crates/lgbm-compute/src/kernels/predict.rs` — bitset predict
  lookup.
- `crates/lgbm-compute/src/kernels/primitives.rs` — the Phase-14 bitonic argsort
  primitive to reuse for the many-vs-many bin sort (D-02).

### Anchor harness & goldens
- `crates/oracle-harness/tests/learner_parity.rs` — the Phase-21 STRUCTURE gate
  (`learner_parity_on_device_structure_gate`) to extend with categorical cases;
  also hosts the categorical golden-comparison harness (real 4.6 goldens,
  bit-exact bitset/decision-type/`num_cat`, ~line 1346+).
- `crates/oracle-harness/tests/fixtures/categorical/` — the REAL `lib_lightgbm`
  4.6 goldens `cat_onehot.{txt,bins.json}` and `cat_manyvsmany.{txt,bins.json}`
  (D-01 pins to these). Recapture via `cargo run -p xtask -- categorical-oracle-capture`.

### CPU host categorical path (the f64 anchor source)
- `crates/lgbm-treelearner/src/feature_histogram_categorical.rs` — host
  categorical split/gain (the cubecl-cpu f64 anchor logic for D-01 #2).
- `crates/lgbm-treelearner/src/data_partition.rs` — host categorical membership
  routing.
- `crates/lgbm-core/src/config/mod.rs` — `max_cat_threshold` (default 32),
  `max_cat_to_onehot` (default 4), `cat_smooth`, `cat_l2`, `min_data_per_group`
  defaults (grounds D-03 and the §8.1 gain math).

### Standing constraints
- `CLAUDE.md` — f32 end-to-end, ~1e-6 vs C++, cubecl-cpu f64 fold is the hard
  merge gate; `LightGBM/` is read-only reference.

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- **`DeviceSplitInfo` categorical slabs already reserved** (`split_info.rs`):
  `num_cat_threshold` (`i32[num_leaf_slots]`), `cat_threshold` (`u32`),
  `cat_threshold_real` (`i32`), sized `num_leaf_slots * MAX_CAT_PER_SPLIT`,
  allocated once — Phase 22 writes into them, never reallocating (the whole point
  of the pre-allocation D-03 extends).
- **`best_split.rs` `is_categorical`/`is_one_hot` dispatch seam** already carries
  the task flags (`is_one_hot = num_bin ≤ max_cat_to_onehot`) and returns the
  invalid sentinel — the eval math slots directly into that branch.
- **Phase-14 bitonic argsort primitive** (`primitives.rs`, index-only) — reused
  for the many-vs-many `grad/(hess+cat_smooth)` bin sort.
- **Phase-21 `learner_parity_on_device_structure_gate`** + tie-aware
  cpu-f64-anchor comparator — extend with categorical corpus cases rather than
  build a new harness.
- **Real 4.6 categorical golden harness** already scaffolded in
  `learner_parity.rs` (SKIP-passes when a golden is absent; goldens present).

### Established Patterns
- Anchor to the cubecl-cpu f64 fold; STRUCTURE bit-exact + leaf values ~1e-5;
  never GPU-vs-GPU (def-f8u-01).
- Additive + `LGBM_CUDA_ON_DEVICE`-gated; env-unset = byte-unchanged; gate runs
  on the default cubecl-cpu lane so the categorical structure gate is
  non-vacuous without ROCm hardware.
- `on_device_growth_supported()` discriminator is the honest host-fallback
  mechanism (D-06 adds the categorical+quantized false case) — Phase-19
  CUDA-unsupported precedent.
- On-device driver crate-cycle constraint (memory
  `on-device-driver-crate-cycle-constraint`): `grow_tree_on_device` lives in
  `lgbm-compute` (below `lgbm-treelearner`); categorical metadata must be
  additive native bookkeeping, no learner types imported.

### Integration Points
- The gated `CpuBackend` flip (`cuda_on_device_enabled()`) is how the categorical
  structure gate runs in the default lane — new corpus cases plug in there.
- `grow_tree_on_device_driver` (`grow_driver.rs`) sequences the grow loop; the
  categorical split/partition/mutation slot into the existing per-leaf best-first
  orchestration (no new driver).
- WR-01 `HistArena::swap` slot-aliasing was fixed in Phase 21 (memory
  `phase18-wr01-histarena-swap-aliasing`) — the categorical grow loop inherits
  the fix; no re-fix needed.

</code_context>

<specifics>
## Specific Ideas

- The **many-vs-many corpus case is the linchpin**: it must simultaneously (a)
  pin the real `cat_manyvsmany` 4.6 golden bit-exact and (b) exercise the
  bitonic-sorted eval + `max_num_cat` clamp path through the full device grow
  loop. One fixture, both purposes.
- D-01 makes categorical the FIRST on-device subsystem with a REAL reference
  anchor (not a re-transcription) — worth stating explicitly in the plan so the
  parity claim is not undersold.

</specifics>

<deferred>
## Deferred Ideas

- **Categorical + `use_quantized_grad` on-device** — host-fallback this phase
  (D-06); an actual device path would only make sense if the reference ever
  gains one (it currently traps). Not a near-term follow-up.
- **Low-VRAM global-memory categorical eval** (`_GlobalMemory` variant, §8.1) —
  optional; only needed if fixtures have more bins than threads. Claude's
  discretion whether to include; otherwise a later perf/coverage slack item.
- **Perf-validation of the categorical path** (Kaggle A/B, device_launches) →
  Phase 23 DoD, with the numerical spine.

None of the discussion drifted outside the phase domain — no scope-creep ideas
to park.

</deferred>

---

*Phase: 22-on-device-categorical-splits-feature-coverage*
*Context gathered: 2026-07-02*
