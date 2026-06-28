# Phase 15: Minimal On-Device Growth (Slice 1) - Context

**Gathered:** 2026-06-29
**Status:** Ready for planning

<domain>
## Phase Boundary

Deliver the **thinnest working on-device tree learner** on real CUDA: a small
(`num_leaves ≤ 8`) **continuous-feature** tree grows entirely on-device via a few large
launches — resident build → `hist_t**` subtraction-trick rotation → per-leaf scan — then
reconstitutes into the `(Tree, DataPartition)` the boosting loop consumes through the
Phase-14 `grow_tree_on_device` seam. This is the milestone-sized lever spikes 051–054
identified as the *only* real fix for the architectural ~2–6× CUDA gap (8570 small serial
host-driven launches gated by the best-first chain).

**Requirements:** ODL-03 (on-device continuous-feature growth, anchor-pinned), ODL-06
(subtraction trick via pointer rotation, no bulk copy), ODL-07 (u64 fixed-point, no f64
hot loops). **Scope stays ODL-03/06/07 — it does NOT absorb ODL-04/05** (D-03 below).

**Host/device boundary (DERIVED from the roadmap slicing + the reference investigation —
LOCKED, not open):** device does build → subtract(rotation) → per-leaf scan; the **host**
keeps cross-leaf best-split argmax, `Tree::split` replay, and row partition (the shipped
035 host-partition win). Per node the device returns **one best-split packet (~120 B, a
`CUDASplitInfo`-equivalent) per touched leaf**; the host picks the frontier winner, replays
the split, and routes the partition. Cross-leaf argmax on-device is **Phase 16 (ODL-04)**;
on-device partition/leaf-index update is **Phase 17 (ODL-05)**.

**Off by default** behind `LGBM_CUDA_ON_DEVICE` (Phase-14 D-05: read once at
`SerialTreeLearner::new`, ANDed with `on_device_growth_supported()`). CPU, ROCm, and the
existing host-CUDA path stay **byte-identical** to master when unset.

**In scope:** the dedicated on-device growth driver + its new compute kernels (D-01/D-02);
the `hist_t**` rotation pool (ODL-06); the per-node best-split-packet readback + host
argmax/split/partition; the narrowed eligibility gate + hard-assert (D-04); the
anchor-pinned correctness gate + local launch-count instrument (D-05).
**Out of scope (own phases):** on-device cross-leaf argmax + tie-aware assert activation →
Phase 16; on-device data partition → Phase 17; categorical / bagging / GOSS / on-device
score update → Phase 18; default-on rollout + Kaggle A/B → Phase 19.

</domain>

<decisions>
## Implementation Decisions

### Build shape — dedicated on-device growth path (not a thin orchestrator)
- **D-01:** Slice 1 builds a **dedicated on-device growth driver** — a purpose-built
  device-resident per-node loop with its own resident pool + readback discipline — rather
  than a thin orchestrator that merely relocates the existing per-leaf
  `build_resident_leaf → subtract_resident → scan_resident_siblings` calls behind the seam.
  Rationale: the shipped per-leaf resident API is shaped for the host-driven per-leaf chain;
  a dedicated driver has the freedom to collapse the per-node launch sequence that
  spikes 051/052/054 identified as the architectural long-pole.
- **D-02 (relaxes SC#1 — planner/verifier MUST honor):** the dedicated path **MAY write
  net-new compute kernels** (e.g. a growth-loop-specific build / subtract / scan), not only
  reuse the shipped u64-build / feature-per-lane scan / sibling co-pack kernels. This is a
  **deliberate relaxation of the literal SC#1 wording** ("reusing the shipped … kernels") —
  do NOT fail the phase for not literally reusing them. **Load-bearing constraint:** any new
  build kernel **keeps the u64 fixed-point accumulation with NO f64 per-row hot loops**
  (ODL-07; spike-052 measured the f64 `build_fix_scan` mega-kernel at **5.4× WORSE** on
  consumer NVIDIA — 1/32 f64 throughput). f64 is permitted only in scalar/storage gain math
  where the reference uses it. The new-kernel freedom **widens the ODL-07 no-f64 audit
  surface** — grep + per-tree-ms (not 6×) must cover every new kernel.

### Slice boundary — keep the host boundary (ODL-03/06/07 only)
- **D-03:** Despite the dedicated path, **cross-leaf argmax + `Tree::split` + partition stay
  on HOST this slice.** Phase 15 requirement scope is **unchanged (ODL-03/06/07)**; it does
  NOT absorb ODL-04 (Phase 16) or ODL-05 (Phase 17). This preserves the thin-slice
  de-risking: Slice 1 proves on-device *growth* (build/subtract/scan + the resident pool +
  the seam reconstitution) without also taking on on-device selection or partition. The
  reference investigation confirmed the readback is sufficient: mainline rebuilds the entire
  host tree from just an 8-int + 16-int packet/node, so one ~120 B best-split packet per
  touched leaf fully determines host `Tree::split` replay + host partition.

### Eligibility gate — hard-assert when forced on an unsupported shape
- **D-04:** Narrow `on_device_eligible` to the supported envelope (continuous features AND
  `num_leaves ≤ 8` AND the backend is CUDA). When `LGBM_CUDA_ON_DEVICE=1` is set AND the
  tree falls **outside** that envelope (categorical, larger trees, multiclass, etc.), the
  path **errors loudly** (a typed `NotSupported`-style failure), it does **NOT** silently
  fall through to host. Rationale: catch "I thought it was on-device but it wasn't" during
  development — the opt-in toggle is a developer/bench affordance this slice, so a forced-on
  unsupported shape is a bug to surface, not a silent no-op.
  - **Planner note:** this diverges from the `resident_eligible` fail-safe (silent
    fall-through) idiom **intentionally**. The merge gate stays green because the **default**
    (`LGBM_CUDA_ON_DEVICE` unset) keeps `on_device_eligible = false` on every backend →
    byte-unchanged host path; the hard-assert only fires under the explicit opt-in. Keep the
    assert behind the eligibility AND-gate so an un-toggled run can never hit it.

### Exit gate — anchor-pinned correctness + local launch-count instrument
- **D-05:** The **hard merge gate** is: (1) anchor-pinned correctness — tree **structure
  bit-exact** to the cpu f64 anchor (tie-aware `default_left`, the Phase-14 D-04 comparator),
  **leaf values within ~1e-5** f32 envelope (never compare two nondeterministic GPU f32 paths
  — def-f8u-01); (2) **CPU / ROCm / host-CUDA byte-unchanged** with the toggle unset; (3) a
  **local `launches/tree` instrument** proving the per-node chain collapsed vs the master
  baseline (launch COUNT is faithful on the spoofed 8-CU APU even though timing is confounded).
  The real-NVIDIA **Kaggle `device_launches` number is a runnable confirmation, NOT a blocking
  gate** — the win magnitude is genuinely open (Slice 1's host boundary adds traffic mainline
  avoids; best-first still serializes), so a paid external manual harness must not gate the
  merge.

### Tie-aware `default_left` assert (carried from Phase 14 — NOT activated here)
- The tie-aware comparator shipped dormant in Phase 14 (D-04). It is **NOT activated** in
  Slice 1 (no on-device flip source yet — argmax is on host). It **activates in Phase 16**
  against the on-device selection output. Slice 1's oracle uses the existing structure-exact
  + 1e-5-leaf comparator.

### Claude's Discretion / planner-resolved (not user decisions)
- The `hist_t**` rotation **mechanism** under cubecl-0.10 — Handle in-place aliasing vs
  ping-pong double-buffering for the data→leaf map, and batched `client.read(vec![h])`
  readback semantics on cubecl-cuda — is a **planning verification spike** (ROADMAP research
  flag). The shipped resident pool already rotates (parent buffer `Move`d to the larger
  child); the planner verifies the dedicated path's rotation meets ODL-06's "no bulk copy"
  and picks the correct cubecl mechanism.
- Exact additive parameters to extend `grow_tree_on_device` with (binned store / per-feature
  metadata / resident handles beyond the current `gradients, hessians, num_leaves, max_depth`)
  — implementation detail, planner-chosen; the Phase-14 doc-comment already says richer inputs
  arrive as additive params in Slice 1.
- Naming of the dedicated driver, the launch-count instrument env var, and the new kernels.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Phase scope & requirements
- `.planning/ROADMAP.md` § "Phase 15: Minimal On-Device Growth (Slice 1)" — goal, 5 success
  criteria, the cubecl-0.10 research flags (Handle aliasing vs ping-pong; batched
  `client.read(vec![h])` readback). NOTE: SC#1's "reusing the shipped kernels" is relaxed by
  D-02 (new kernels allowed; u64/no-f64 constraint retained).
- `.planning/ROADMAP.md` § Phase 16 / Phase 17 — the boundary D-03 holds: ODL-04 (on-device
  argmax + tie-aware assert) and ODL-05 (on-device partition) are explicitly later slices.
- `.planning/REQUIREMENTS.md` — ODL-03, ODL-06, ODL-07 (this phase); ODL-04/05 (NOT this phase).

### Phase-14 seam (the contract this slice fills in)
- `.planning/phases/14-scaffold-oracle-slice-0/14-CONTEXT.md` — the seam contract
  (D-03/Option A: `Result<Option<(Tree, LeafPartitionLayout)>>`), the `on_device_eligible`
  decide-once-at-`new` gate (D-05), the dormant tie-aware comparator (D-04).
- `crates/lgbm-compute/src/lib.rs:1219-1281` — `on_device_growth_supported` (default false) +
  `grow_tree_on_device` (default `Ok(None)`) + the cubecl-0.10 kernel checklist baked into the
  doc-comment (no global barrier; `Atomic<i64>` broken → u64 two's-complement; `wrapping_add`
  not an intrinsic; plane-sum ≤ plane width; `launch_unchecked` unsafe).
- `crates/lgbm-treelearner/src/learner.rs:693-714` — the `train_inner` on-device routing fork
  (`if self.on_device_eligible { if let Some((tree, payload)) = … }`); D-04's hard-assert lives
  inside this gate.
- `crates/lgbm-dataset/src/dataset.rs:80-88` (`LeafPartitionLayout`) +
  `crates/lgbm-treelearner/src/data_partition.rs:68-74` (`DataPartition::from_payload`) — the
  lower-crate payload `P` the seam returns and the learner reconstitutes.

### Reference implementation (the architecture being ported — READ-ONLY; never git-add LightGBM/)
- `LightGBM/src/treelearner/cuda/cuda_single_gpu_tree_learner.cpp:174-343` — the best-first
  growth loop; the host/device boundary; the 8-int + 16-int per-node readbacks; smaller/larger
  designation (`:328-329`); `tree->ToHost()` only at the end (`:343`).
- `LightGBM/src/treelearner/cuda/cuda_histogram_constructor.cu` — build (smaller leaf only),
  `SubtractHistogramKernel` (larger −= smaller in place), `FixHistogramKernel`; `HIST_TYPE`
  templated f32/f64 shared, global `hist_t` = double. Our port uses **u64 fixed-point**
  instead (ODL-07).
- `LightGBM/src/treelearner/cuda/cuda_data_partition.cu:825-905` — `SplitTreeStructureKernel`:
  the histogram-pool **pointer rotation** (larger child inherits the parent slab, smaller gets
  a fresh slab — `cuda_hist_pool_`). The ODL-06 reference.
- `LightGBM/src/treelearner/cuda/cuda_best_split_finder.cpp:324-381` — two-stage selection:
  `FindBestSplitsForLeaf` (per leaf×feature, both touched leaves) → `FindBestFromAllSplits`
  (cross-leaf argmax). Slice 1 keeps the **cross-leaf argmax on host** (Phase 16 moves it
  on-device).
- `include/LightGBM/cuda/cuda_split_info.hpp:17-39` — the `CUDASplitInfo` struct = the ~120 B
  per-leaf best-split packet shape Slice 1 reads back per touched leaf.

### Engineering memory (constraints baked into this slice)
- spike-052 (`cuda-architectural-launch-bound.md`): the f64 `build_fix_scan` mega-kernel is
  **5.4× WORSE** on consumer NVIDIA → **no f64 hot loops in new cuda kernels** (ODL-07); keep
  the separate u64-fixed-point path. The on-device multi-leaf learner is THE architectural lever.
- spike-051/054: P=1 optimal on real CUDA (APU P-sensitivity does NOT transfer); the gap is
  launch-bound (8570 small serial launches), halves with feature-width but never beats official.
- def-f8u-01 (commits 1832206 / d82611b): never compare two nondeterministic GPU f32 paths —
  always pin to the cpu f64 anchor; leaf values within the ~1e-5 f32 envelope.
- `.claude/skills/spike-findings-lightgbm_rs/SKILL.md` § cuda-architectural-launch-bound,
  cuda-discrete-gpu-bottleneck — the real-NVIDIA attribution this phase acts on.

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- **Resident pool already rotates** (`crates/lgbm-treelearner/src/resident_pool.rs`,
  `learner.rs:23` "parent buffer is `Move`d to the larger child"): `build_resident_leaf`,
  `subtract_resident`, `move_resident(src,dst)`, `scan_resident_leaf`,
  `scan_resident_siblings` (co-pack), `build_fix_scan_resident` — the on-device build → subtract
  (pointer-swap rotation) → scan chain EXISTS. Even though D-01/D-02 build a dedicated path with
  new kernels, this is the proven-correct reference for the rotation semantics (ODL-06) and the
  u64 fixed-point accumulation (ODL-07); the dedicated kernels must match its bit-exact behavior.
- `upload_resident_bins` / `wants_resident_bins` (`lib.rs:867,882`) — bin data already stays
  device-resident (spike-p9v hoist); the growth driver inherits this, no per-tree re-upload.
- `reset_resident_pool(num_slots, slot_len)` (`lib.rs:942,2507`) — the fixed histogram pool
  the rotation operates over (mirrors mainline's `cuda_hist_` slab + `cuda_hist_pool_`).
- The Phase-14 oracle (`crates/oracle-harness/tests/learner_parity.rs:2046`,
  `assert_gpu_tree_matches_cpu_anchor` + `cpu_anchor_tree`) — the structure-exact + 1e-5-leaf
  comparator the D-05 correctness gate reuses (tie-aware branch dormant until Phase 16).
- `phase_prof` (`crates/lgbm-treelearner/src/phase_prof.rs`, `LGBM_PHASE_PROF`) — the
  instrumentation home for the D-05 local `launches/tree` count.

### Established Patterns
- **Decide-once eligibility ANDed with a backend discriminator** (`on_device_eligible =
  on_device_growth_supported() && cuda_on_device_env()`, `learner.rs:488`) — D-04 narrows the
  predicate to the supported envelope and adds the hard-assert INSIDE this gate.
- **Default-false trait-method discriminator on ONE backend** (mirrors `resident_pool_supported`
  / `prefers_host_partition`) — `GpuBackend<R>` flips `on_device_growth_supported()` true only
  for the CUDA runtime + supported shape; CPU/ROCm/WGPU stay false → byte-unchanged.
- **u64 two's-complement fixed-point atomics** (spike-018/019, `Atomic<i64>` broken on
  cubecl-0.10) — the mandatory accumulation idiom for any new build kernel (ODL-07).
- **Seam returns lower-crate `LeafPartitionLayout`**, learner reconstitutes `DataPartition`
  (Phase-14 Option A) — avoids the treelearner→compute crate cycle.

### Integration Points
- `Backend::grow_tree_on_device` (`lib.rs:1272`) — the dedicated growth driver lands behind
  the `GpuBackend<R>` override (`lib.rs:2207`); extend the signature with additive feature/bin
  inputs (planner-chosen).
- `SerialTreeLearner::train_inner` fork (`learner.rs:704`) — already consumes
  `Some((tree, payload))`; D-04's hard-assert sits in/around this gate when forced-on-unsupported.
- The boosting loop consumes the reconstituted `(Tree, DataPartition)` unchanged — the seam's
  contract guarantees a valid partition for the next iteration's gradients.

</code_context>

<specifics>
## Specific Ideas

- The reference is host-orchestrated but FULLY on-device (selection + partition + tree on
  device, 2 tiny scalar readbacks/node, ~13–15 launches/node across 4 stream-overlapped
  subsystems). Slice 1 deliberately **inverts** that boundary (host argmax/partition/tree) as
  the thin first step; the full mainline mirror is the END of the slice sequence (Phases 16–17),
  not Slice 1.
- "Few large launches" is the per-node launch-collapse target; the dedicated driver's job is to
  reduce the per-node device launches (master baseline = the host-driven per-leaf chain) while
  the best-first serialization remains. Measure with the D-05 local launch-count instrument.
- The merge gate is the hard gate throughout: `raw_bin_train_matches_cpp_golden`,
  `learner_parity`, and the lgbm/treelearner/compute suites green AND byte-unchanged with
  `LGBM_CUDA_ON_DEVICE` unset.

</specifics>

<deferred>
## Deferred Ideas

- On-device cross-leaf best-split argmax + ACTIVATING the dormant tie-aware `default_left`
  comparator → **Phase 16 (Slice 2, ODL-04)**.
- On-device data partition / leaf-index update (the Split kernel, removing the host partition
  round-trip) → **Phase 17 (Slice 3, ODL-05)**.
- Categorical features / bagging / GOSS / on-device score update → **Phase 18 (ODL-08/09/10)**;
  the D-04 hard-assert fires on these in Slice 1.
- `num_leaves > 8` / production-depth on-device growth → grows across Phases 16+ (the per-node
  loop generalizes once cross-leaf argmax + partition move on-device).
- Kaggle `device_launches` A/B + default-on rollout → **Phase 19 (ODL-11/12)** (the D-05 Kaggle
  confirmation can run earlier as evidence, but it is not a Phase-15 gate).
- The cubecl-0.10 Handle-aliasing-vs-ping-pong rotation mechanism + batched `client.read(vec![h])`
  readback semantics → a **planning verification spike** within Phase 15 (research flag, not a
  separate phase).

None — discussion stayed within phase scope.

</deferred>

---

*Phase: 15-minimal-on-device-growth-slice-1*
*Context gathered: 2026-06-29*
