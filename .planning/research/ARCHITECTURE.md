# Architecture Research — On-Device CUDA Tree Learner Integration (v1.1)

**Domain:** GPU gradient-boosting tree learner (pure-Rust LightGBM port, CubeCL backends)
**Researched:** 2026-06-28
**Confidence:** HIGH (integration mechanics read directly from source); MEDIUM (on-device kernel decomposition — the milestone's open work)

> Scope: this is an **integration** architecture study for the v1.1 milestone, not the
> greenfield v1.0 one (preserved at `ARCHITECTURE.v1.0.md`). It answers "how does an
> on-device, whole-tree growth loop (mirror C++ `CUDASingleGPUTreeLearner`) slot into the
> existing `Backend` + `SerialTreeLearner` architecture WITHOUT breaking the CPU f64 anchor,
> the ROCm resident path, or the existing host-driven CUDA path." Concrete file/trait/method
> names are used throughout so the planner can wire against the real code.

---

## Standard Architecture (today, host-driven)

### System Overview

```
┌──────────────────────────────────────────────────────────────────────┐
│  lgbm-boosting :: GBDT loop  (gbdt.rs:1289)                            │
│    per iter:  grad/hess (host) ─► learner.train_returning_partition()  │
│               ◄── (Tree, DataPartition)                                │
│               score_updater.add_prediction_to_score(tree, part, score) │  ◄ host scatter
├──────────────────────────────────────────────────────────────────────┤
│  lgbm-treelearner :: SerialTreeLearner<'b, B: Backend>  (learner.rs)   │
│    train_inner():                                                      │
│      decide-once-at-top:  resident_eligible / fused_eligible           │  ◄ gating fork
│      upload_resident_bins() once/train                                 │
│      LEAF-WISE LOOP  (×(num_leaves-1)):                                │
│        before_find_best_split ─► find_best_splits ─► argmax ─► split_inner
│        find_best_splits:  build(smaller) ─► subtract(larger) ─► scan both
│        split_inner:       DataPartition::split  +  Tree::split          │
├──────────────────────────────────────────────────────────────────────┤
│  lgbm-compute :: Backend trait  (lib.rs:495)   — the CMP-01 CubeCL seam │
│    per-op methods + additive default methods + backend discriminators  │
│    impls:  CpuBackend (unit)   |   GpuBackend<R> { resident_bins,       │
│                                       resident_pool, resident_enabled } │
│    aliases:  RocmBackend=GpuBackend<RocmRuntime> (rocm feat)            │
│              CudaBackend=GpuBackend<CudaRuntime> (cuda feat)            │
└──────────────────────────────────────────────────────────────────────┘
```

**The problem this milestone attacks** (`cuda-architectural-launch-bound.md`, spikes 051–054):
the LEAF-WISE LOOP is **host-orchestrated**. Even with the device-resident histogram pool
(260608-p90), each node still issues a small `build` / `subtract` / `scan` launch gated by the
best-first dependency chain → **~8,570 small serial launches / 100-tree train**, ~86/tree.
On real NVIDIA this is **launch-latency-bound** (occupancy/fusion/sync all refuted). The one
remaining lever is to keep the whole growth frontier on-device — fewer, bigger launches —
mirroring `CUDASingleGPUTreeLearner`.

### Component Responsibilities (relevant to the seam)

| Component | What it owns | File |
|-----------|--------------|------|
| `Backend` trait | The CMP-01 CubeCL isolation seam: per-op kernels + additive default methods + discriminators | `crates/lgbm-compute/src/lib.rs:495` |
| `GpuBackend<R>` | Generic GPU backend; carries `resident_bins` + `resident_pool` device-handle mirrors behind `RefCell`; `RocmBackend`/`CudaBackend` are type aliases | `lib.rs:2037` |
| `SerialTreeLearner` | The host growth loop (`train_inner`), the decide-once-at-top eligibility fork, returns `(Tree, DataPartition)` | `crates/lgbm-treelearner/src/learner.rs:658` |
| `resident_pool` | The CONSERVATIVE fail-safe eligibility predicates + size gates + env overrides | `crates/lgbm-treelearner/src/resident_pool.rs` |
| GBDT loop | Drives one `train_returning_partition` per iter, scatters leaf values to per-row scores | `crates/lgbm-boosting/src/gbdt.rs:1289` |

---

## The Integration Decision (Question 1: the seam)

**Recommendation: a NEW additive `Backend` method `grow_tree_on_device(...)` gated by a NEW
default-false discriminator `on_device_growth_supported()`, routed by a decide-once-at-top fork
in `SerialTreeLearner::train_inner` that returns early — NOT a new trait and NOT a parallel
`TreeLearner` impl.**

Three candidates weighed against the codebase's actual idioms:

| Option | Verdict | Why |
|--------|---------|-----|
| (a) **New `Backend` method + discriminator** | **CHOSEN** | Exactly the established additive-default + discriminator idiom (`prefers_host_partition`, `resident_pool_supported`, `host_unified_fused_supported`, `data_partition_native`). Keeps the CMP-01 seam intact (all cubecl confined to lgbm-compute). Reuses the existing `train_inner` routing fork. `SerialTreeLearner` public API + the whole boosting loop are byte-unchanged. |
| (b) New `TreeLearner` impl alongside `SerialTreeLearner` (e.g. `CudaSingleGpuTreeLearner`) | Rejected (for now) | There is no `TreeLearner` *trait* in the Rust port — `SerialTreeLearner` is a concrete struct the boosting loop names directly (`SerialTreeLearner::new`, 20+ call sites in gbdt.rs). A parallel learner forces a learner-level trait + dispatch refactor across lgbm-boosting — a large blast radius into the CPU/ROCm paths the milestone must protect. The C++ `CUDASingleGPUTreeLearner` is a separate class only because C++ uses the stringly-typed `CreateTreeLearner` factory; the Rust port deliberately collapsed that into one struct + a Backend seam. Stay with the seam. |
| (c) A brand-new trait | Rejected | Redundant with `Backend`; re-creates the CMP-01 boundary the project already pays for once. |

### Concrete shape (additive, default-safe)

In `crates/lgbm-compute/src/lib.rs`, on `trait Backend`, add (all with CPU-safe defaults):

```rust
/// Whether this backend can grow a whole tree ON DEVICE in one orchestrated call
/// (mirror C++ CUDASingleGPUTreeLearner), bypassing the host per-leaf loop.
/// Default false ⇒ CpuBackend + any backend that does not opt in keep the host loop.
fn on_device_growth_supported(&self) -> bool { false }

/// Grow ONE tree fully on-device from device-resident bins + per-iter g/h, returning
/// the grown tree's node arrays + the final per-row leaf assignment in ONE readback.
/// Default: typed error (never reached — the eligibility gate ANDs in the discriminator).
#[allow(clippy::too_many_arguments)]
fn grow_tree_on_device(
    &self,
    client: &ComputeClient<Self::Runtime>,
    feats: &[BatchedSplitFeature],   // per-feature dispatch params (already exists)
    gradients: &[f32],
    hessians: &[f32],
    cfg: &GainConfig,
    num_leaves: i32,
    max_depth: i32,
    min_data_in_leaf: i32,
) -> Result<OnDeviceTreeResult, ComputeError> {
    Err(ComputeError::Runtime { detail:
        "grow_tree_on_device: on-device growth not supported on this backend".into() })
}
```

`OnDeviceTreeResult` is a new **plain-data, cubecl-free** struct in `lgbm-compute` (so it can
cross the CMP-01 boundary): the grown tree's parallel node arrays (`split_feature`,
`threshold_bin`, `left_child`, `right_child`, `leaf_value`, internal/leaf sums, `decision_type`,
`default_left`) **plus** the final `row_leaf: Vec<i32>` (per-row leaf id). The learner converts
this into a host `lgbm_model::Tree` + `DataPartition` (see Question 3).

The bins themselves do **not** travel in the signature — they ride the existing
`upload_resident_bins()` / `ResidentBins` device cache (`lib.rs:2000`) already populated once per
train. `grad`/`hess` upload once per tree inside the launcher (or via a tiny per-tree upload seam).

---

## Routing That Protects CPU + ROCm + Existing CUDA (Question 2)

**Principle (project rule, restated in `partition-memory-traffic.md`): backend discriminators are
default-false trait methods overridden on ONE backend — never a global env/flag.** The new path
must be invisible until explicitly opted in.

### Two-layer gate, mirroring `resident_eligible`

1. **Backend capability** — `on_device_growth_supported()` overridden on `GpuBackend<R>` only.
   Because `RocmBackend` and `CudaBackend` share `GpuBackend<R>`, distinguish them with a
   **field-backed opt-in** exactly like the existing `resident_enabled` bool + `with_resident()`
   constructor idiom (`lib.rs:2057`, `2097`):

   ```rust
   // GpuBackend<R> gains: on_device_enabled: bool  (Default = false)
   //   + a with_on_device(true) constructor (test/bench) and an env opt-in.
   fn on_device_growth_supported(&self) -> bool {
       // OFF by default everywhere; opt-in only. Protects ROCm (keeps the shipped
       // resident host-driven path) AND the existing host-CUDA path until proven.
       self.on_device_enabled
           && matches!(std::env::var("LGBM_CUDA_ON_DEVICE").as_deref(), Ok("1"))
   }
   ```

   - **CPU:** trait default `false` → host loop, **bit-exact anchor untouched.**
   - **ROCm:** `on_device_enabled` left `false` (its shipped path is the host-driven resident
     pool + host partition, spike-035) → **untouched.**
   - **Existing host-CUDA path:** default `false` → today's per-leaf launches still run until
     `LGBM_CUDA_ON_DEVICE=1` is set → **untouched** by default. Same ship-default-off discipline
     as the fused kernel (`FUSED_MAX_NUM_DATA = -1`, resident_pool.rs:195).

2. **Workload eligibility** — a new `on_device_eligible(...)` predicate in `resident_pool.rs`
   (or a sibling `on_device.rs`), **structurally identical to `resident_eligible`** (resident_pool.rs:99):
   ANDs in `backend.on_device_growth_supported()`, then the SAME fail-safe rejects — pure numeric
   spine only (no categorical / monotone / interaction / extra_trees / forced_splits /
   non-default `max_delta_step`/`path_smooth`), `!capture_snapshots`. When in doubt → `false` →
   byte-unchanged host path. Add a size gate (on-device wins only above some `num_data`) and an
   `LGBM_CUDA_ON_DEVICE_FORCE` three-way override (`0`/`1`/unset), mirroring `LGBM_RESIDENT_FORCE`.

### The routing fork (one place, decide-once-at-top)

In `SerialTreeLearner::train_inner` (learner.rs ~680, where `resident_eligible` / `fused_eligible`
are computed), add a THIRD decision and an **early return BEFORE the leaf-wise loop**:

```rust
let on_device = crate::on_device::on_device_eligible(
    self.backend.on_device_growth_supported(), num_data, &features,
    &self.constraints, capture_snapshots, &self.cfg);
if on_device {
    let result = self.grow_on_device(&features, gradients, hessians)?; // delegates to backend
    let (tree, data_partition) = self.reconstitute(result, &features, num_data);
    self.hist_pool = Some(pool);                 // keep the pool-reuse invariant
    return Ok((tree, Vec::new(), ColSamplerTrace::default(), data_partition));
}
// ... else: the existing host leaf-wise loop, byte-unchanged ...
```

This keeps `train` / `train_returning_partition` / `train_on_subset` and every gbdt.rs call site
**signature-identical**. The on-device path is a sibling branch of the SAME function, exactly like
the resident-vs-host split already inside `find_best_splits`.

---

## What Moves On-Device & How the Result Returns (Question 3)

### Device-resident state (mirror `CUDASingleGPUTreeLearner` members)

| C++ member | Rust home | Status |
|------------|-----------|--------|
| `cuda_histogram_constructor_` (resident histograms) | `GpuBackend.resident_pool` handle mirror | **already exists** (260608-p90) |
| binned dataset | `GpuBackend.resident_bins` (`ResidentBins`, feature-major) | **already exists** (260608-nn7) |
| `cuda_data_partition_` (row→leaf, on-device split) | NEW resident `row_leaf` + per-leaf range handles inside the launcher | new |
| `cuda_smaller/larger_leaf_splits_` (frontier sums) | NEW resident leaf-splits buffers | new |
| `cuda_best_split_finder_` (frontier argmax) | NEW resident best-split-per-leaf buffer + on-device argmax | new |
| growing `Tree` node arrays | accumulated device-side, read back ONCE | new |

The growth loop (`build → subtract → best-split → partition`) runs device-side with the frontier
resident; the host issues a **handful** of launches per tree instead of ~86.

### Result return — host-bit-comparable

The boosting layer needs exactly two host objects (gbdt.rs:1289 + score_updater.rs:123): a
`lgbm_model::Tree` and a `DataPartition`, consumed by
`add_prediction_to_score(tree, data_partition, score)` (learner.rs:3729), a pure host per-row
leaf-value scatter. So the on-device path reconstitutes both:

- **Tree:** read back the node arrays once and replay them onto `lgbm_model::Tree` via the existing
  `Tree::split` builder (learner.rs:3405) in leaf-creation order — O(num_leaves) cheap host work,
  no per-row cost. The replayed Tree is structurally identical to what a host loop would emit.
- **DataPartition:** read back the final `row_leaf` once and bucket rows per leaf into a
  `DataPartition` (its `indices_in_leaf(leaf)` is all the score scatter reads). This is the single
  device→host transfer that matters; everything else stayed resident.

**Per-iter scores stay bit-comparable for free:** the on-device path returns the SAME
`(Tree, DataPartition)` and the host `add_prediction_to_score` scatter is reused verbatim, so the
score update is identical-to-the-host-path within the f32 envelope. (Keeping the score vector on
device is explicitly OUT of the minimal slice — it would change score-update numerics and widen the
parity surface.)

> **Partition placement nuance (load-bearing).** `partition-memory-traffic.md` (spike-035) found the
> device partition round-trip is pure overhead on shared memory and routed ROCm partition on the HOST
> by default. The dominant CUDA cost is the **build→subtract→scan launch chain**, not partition. So
> the first slices may keep the shipped host fused partition (027) and still bank most of the win;
> moving partition fully on-device (true single-GPU learner) is a later slice that pays off on
> discrete PCIe NVIDIA where the round-trip crosses the bus.

---

## Parity Gating (Question 4) — anchor-pinned, NOT bit-exact

The on-device CUDA path uses **f32 atomic histogram builds**, which are ~1.9e-6 nondeterministic
run-to-run (def-f8u-01, `partition-memory-traffic.md`). It **cannot** be bit-exact, and two
nondeterministic GPU f32 paths must **never** be compared to each other at 1e-6 (def-f8u-01 MEMORY:
"never compare two nondeterministic GPU f32 paths to each other").

**Structure the oracle exactly like `learner_parity_{resident,fused}_equals_host_tree_on_hip`:**

1. **Pin STRUCTURE to the cpu f64 anchor, not to another GPU run.** Grow the corpus twice: once on
   `CpuBackend` (the bit-exact anchor) and once on-device; assert the tree TOPOLOGY
   (`split_feature`, `threshold_bin`, `left_child`/`right_child`, leaf count) matches the anchor.
2. **Leaf VALUES within an f32 envelope** (~1e-5, the def-f8u-01 bound), never bit-exact.
3. **Tie-aware structural assert (mandatory).** Cross-feature argmax and `default_left` can legally
   flip when the gain margin is within the f32 envelope — the `hip-split-parity` near-tie class,
   fixed via a tie-aware assert (commit 1832206, `hip-split-parity-preexisting-defect` MEMORY). The
   gate must accept either branch when the winning-vs-runner-up gain margin is within the envelope,
   rather than demand exact topology. **Plan for this from the start** — naive exact topology goes
   red on near-ties and wastes a debug cycle.
4. **CPU merge gate untouched.** `cargo test -p oracle-harness -p lgbm-treelearner -p lgbm` stays
   the hard gate; the on-device path is default-off so these run the CPU anchor unchanged.
5. **Real-CUDA validation surface = Kaggle** (`kaggle-cli-cuda-bench`, user `boomvector`), the only
   true discrete-NVIDIA signal (local GPU is a spoofed APU). Push to `master`, run the kernel, read
   `phase_prof` `device_launches` (select the max-launches/timed dump, not the warmup — the 051
   parse rule).

---

## Suggested Build Order (Question 5) — vertical slices

Each slice is **end-to-end** (grows a real tree, returns `(Tree, DataPartition)`, passes the
anchor-pinned gate) and ships **default-off** behind `LGBM_CUDA_ON_DEVICE`.

### Slice 0 — Scaffold the seam (no behavior change)
Add `on_device_growth_supported()` (default false) + `grow_tree_on_device()` (default typed error)
+ `OnDeviceTreeResult` + `on_device_eligible()` + the `train_inner` early-return fork + the
`reconstitute()` helper (node-array→Tree replay, `row_leaf`→DataPartition). The `GpuBackend<R>`
override still returns the typed error. **Merge gate green; CPU/ROCm/host-CUDA all untouched**
(default-off, eligibility ANDs in a false discriminator). De-risks the wiring before any kernel.

### Slice 1 — MINIMAL PROVING SLICE (proves on-device growth on real CUDA)
On `GpuBackend<R>`, implement `grow_tree_on_device` for the **narrowest viable tree**: pure numeric
spine, small `num_leaves` (e.g. ≤8), the **build→subtract→best-split frontier resident** driven by a
few large launches, **host partition reused** (the shipped 027 fused path) + **host `Tree::split`
replay** from read-back per-split decisions. ONE readback of split decisions + final `row_leaf`.
- **Proves:** the seam grows a structurally-anchor-pinned tree end-to-end on real CUDA; the per-node
  build/subtract/scan launch chain collapses to O(depth) large launches; `device_launches/tree`
  drops materially vs master (the Kaggle measurement).
- **Gate:** anchor-pinned tie-aware structure + ~1e-5 leaf values; CPU merge gate green.
- The smallest thing that attacks the 8,570-launch finding and validates parity + return
  reconstitution at once. Keep `num_leaves` small so the readback/replay is trivial to verify.

### Slice 2 — On-device best-split across the full frontier
Move the cross-feature argmax over the whole leaf frontier on-device (resident best-split-per-leaf),
removing the per-leaf scan readbacks. Grow to production `num_leaves`/`max_depth`.

### Slice 3 — On-device data partition (true single-GPU learner)
Move row routing on-device (resident `row_leaf` updated per split), eliminating the host partition
round-trip — the part that pays off on discrete PCIe NVIDIA. Now only ONE readback at end of growth
(node arrays + `row_leaf`). The full `CUDASingleGPUTreeLearner` mirror.

### Slice 4 — Default-on routing + size gate + perf-harden
Flip the size gate / `on_device_enabled` default for CUDA only (ROCm stays host-driven), set the
`num_data` crossover from Kaggle A/B, keep the env off-switch. Honesty mandate: ship default-on ONLY
where the real-CUDA A/B shows a sign-stable win (the fused-kernel default-off precedent).

**Ordering rationale:** Slice 0 isolates wiring risk from kernel risk. Slice 1 proves the hardest
uncertainty (does on-device growth + anchor-pinned parity + result reconstitution actually work on
real CUDA) at minimum kernel surface. Slices 2–3 expand the resident frontier monotonically, each
independently gated. Slice 4 is pure routing/perf, deferred until the win is measured — never
auto-engaged before proof (the project's audit-before-wire value).

---

## Anti-Patterns (specific to this integration)

### Anti-Pattern 1: A global env flag or runtime mode to switch learners
**What people do:** add `if std::env::var("USE_ON_DEVICE")` deep in the loop, or a process-global
learner mode.
**Why wrong:** the project's hard rule is backend discriminators (default-false trait methods on one
backend), never globals (`partition-memory-traffic.md` Constraints). A global would silently affect
ROCm/CPU and break the decide-once-at-top discipline.
**Do this instead:** `on_device_growth_supported()` discriminator + `on_device_eligible()` predicate
+ the single `train_inner` fork.

### Anti-Pattern 2: Bit-exact GPU-vs-GPU parity asserts
**What people do:** assert the on-device tree equals a host-GPU-grown tree cell-for-cell.
**Why wrong:** both are nondeterministic f32 atomic builds (~1.9e-6); the assert is flaky by
construction (def-f8u-01).
**Do this instead:** pin STRUCTURE to the cpu f64 anchor, leaf values within ~1e-5, tie-aware.

### Anti-Pattern 3: f64 hot loops in the new CUDA kernels
**What people do:** reuse the f64 fused `build_fix_scan_resident` math on consumer NVIDIA.
**Why wrong:** consumer NVIDIA f64 is 1/32 f32 — `LGBM_FUSED_FORCE=1` was **5.4× WORSE** on CUDA
(spike-052). The fused f64 kernel tanks.
**Do this instead:** keep the **u64 fixed-point** build path (spike-018) for the on-device histogram
build.

### Anti-Pattern 4: A parallel `TreeLearner` trait + boosting dispatch refactor
**What people do:** introduce a learner-level trait to host `CudaSingleGpuTreeLearner` alongside
`SerialTreeLearner`.
**Why wrong:** 20+ gbdt.rs call sites name `SerialTreeLearner` concretely; a trait + dispatch
refactor has a large blast radius across the CPU/ROCm paths the milestone must protect.
**Do this instead:** the Backend seam + the in-`train_inner` early-return fork.

---

## Integration Points

### Internal Boundaries

| Boundary | Communication | Notes |
|----------|---------------|-------|
| boosting ↔ treelearner | `train_returning_partition() -> (Tree, DataPartition)`; `add_prediction_to_score()` | **UNCHANGED.** On-device path returns the same pair. The seam is invisible above the learner. |
| treelearner ↔ compute | new `Backend::grow_tree_on_device()` + `on_device_growth_supported()` | Additive; all cubecl confined to lgbm-compute (CMP-01). `OnDeviceTreeResult` is plain data. |
| learner internal | `train_inner` early-return fork + `reconstitute()` | Sibling branch of the existing resident/host fork; pool-reuse + return-tuple invariants preserved. |
| compute internal | `GpuBackend<R>` reuses `resident_bins` + `resident_pool`; adds resident frontier/partition state | ROCm shares the struct but keeps `on_device_enabled=false` → its shipped path is untouched. |

### Feature-gate map
- No new Cargo feature needed. On-device impl lives under the existing `#[cfg(feature = "gpu")]`
  `GpuBackend<R>` block; CUDA-only opt-in is a runtime field/env (`on_device_enabled` +
  `LGBM_CUDA_ON_DEVICE`), not a compile gate — so a `--features cuda` build still defaults to the
  existing host-CUDA path until explicitly enabled.

---

## Scaling Considerations

| Scale (num_data) | On-device routing |
|------------------|-------------------|
| tiny / small | host loop wins (launch-bound resident chain already loses < ~12k rows, resident_pool.rs:58). Size-gate the on-device path OFF here too. |
| medium–large | the on-device frontier amortizes its few large launches; the 8,570→O(depth) launch collapse is the win. Crossover set from Kaggle A/B. |
| wide (≥500 feat) | the lgb_rs/official gap already halves with width (3.90×@50f→1.93×@500f, spike-054) because per-launch ms rises and the launch-bound fraction shrinks; on-device still helps but build dominates — measure, don't assume. |

---

## Sources

- `crates/lgbm-compute/src/lib.rs` — `Backend` trait (:495), discriminator idioms
  (`prefers_host_partition` :898, `resident_pool_supported` :935, `host_unified_fused_supported`
  :914, `data_partition_native` :637), `GpuBackend<R>` (:2037), `RocmBackend`/`CudaBackend` aliases
  (:2110/:2116), `ResidentBins` (:2000), `resident_pool` handle mirror (:2051). [HIGH]
- `crates/lgbm-treelearner/src/learner.rs` — `train_inner` decide-once-at-top fork (:680–714), the
  leaf-wise loop (:1024–1112), `find_best_splits` resident/host routing (:1404–1812), `Tree::split`
  builder (:3405), `add_prediction_to_score` (:3729). [HIGH]
- `crates/lgbm-treelearner/src/resident_pool.rs` — the conservative fail-safe eligibility predicate +
  size gate + `LGBM_*_FORCE` override pattern to clone for `on_device_eligible`. [HIGH]
- `crates/lgbm-boosting/src/gbdt.rs:1289` + `score_updater.rs:123` — the `(Tree, DataPartition)` →
  per-row score scatter contract the on-device path must preserve. [HIGH]
- `.claude/skills/spike-findings-lightgbm_rs/references/cuda-architectural-launch-bound.md` — the
  launch-bound diagnosis, "on-device learner is the one architectural lever", the f64-on-CUDA /
  u64-fixed-point / Kaggle-measurement rules. [HIGH]
- `.claude/skills/spike-findings-lightgbm_rs/references/partition-memory-traffic.md` — the
  additive-discriminator wiring idiom + def-f8u-01 anchor-pinned (not bit-exact) GPU parity approach
  + the host-vs-device partition placement finding (spike-035). [HIGH]
- MEMORY: def-f8u-01 (anchor-pin GPU f32 to cpu anchor), hip-split-parity near-tie tie-aware assert
  (commit 1832206), kaggle-cli-cuda-bench. [HIGH]
- `LightGBM/src/treelearner/cuda/cuda_single_gpu_tree_learner.cpp` — the C++ reference port target
  (resident histogram/partition/best-split-finder members, on-device `Train()` loop). [HIGH]

---
*Architecture research for: on-device CUDA tree learner integration (lightgbm_rs v1.1)*
*Researched: 2026-06-28 — prior v1.0 study preserved at ARCHITECTURE.v1.0.md*
