# Phase 14: Scaffold + Oracle (Slice 0) - Pattern Map

**Mapped:** 2026-06-28
**Files analyzed:** 4 (3 MODIFY + 1 CREATE)
**Analogs found:** 4 / 4 (all in-codebase, all line-refs re-verified against live source)

All RESEARCH.md `[VERIFIED]` line citations were re-read this session and confirmed
accurate. Excerpts below are the actual current source, not paraphrases.

## File Classification

| In-Scope File | Role | Data Flow | Closest Analog | Match Quality |
|---------------|------|-----------|----------------|---------------|
| `crates/lgbm-compute/src/lib.rs` (MODIFY) | trait / backend seam | request-response (capability discriminator + grow call) | `Backend::resident_pool_supported` (`lib.rs:935`) + `prefers_host_partition` (`lib.rs:898`) | exact (same trait, same default-false idiom) |
| `crates/lgbm-treelearner/src/learner.rs` (MODIFY) | tree-learner (routing) | event-driven (decide-once eligibility → routing fork) | `resident_eligible` cache + fork (`learner.rs:286,458,680`) | exact (same struct, mirror with intentional D-05 divergence) |
| `crates/oracle-harness/tests/learner_parity.rs` (MODIFY) | test (anchor parity) | transform / assertion (tree → bit-exact compare) | `assert_gpu_tree_matches_cpu_anchor` (`:2046`) + `cpu_anchor_tree` (`:2083`) + `kernel_parity.rs:1597` near-tie | exact (extend in place) |
| Partition payload `P` in `lgbm-core`/`lgbm-dataset` (CREATE) | model / POD data-struct | data container (raw leaf-row layout) | `EfbSamples` (`lgbm-dataset/dataset.rs:58`) for the struct *shape*; `DataPartition` (`data_partition.rs:33`) for the *fields* to carry | role-match (new lower-crate struct; no existing lower-crate partition type) |

## Pattern Assignments

### `crates/lgbm-compute/src/lib.rs` (trait / backend seam, request-response)

**Analog:** `Backend::resident_pool_supported` (`lib.rs:935`) — the default-false
discriminator precedent; `prefers_host_partition` (`lib.rs:898`) — the additive-bool
sibling. Add TWO things on `trait Backend`: a discriminator (mirror `:935` exactly) and
the `grow_tree_on_device` seam (new shape, `Ok(None)` default).

**Discriminator pattern to copy verbatim** (`lib.rs:931-937`):
```rust
/// Whether this backend supports the device-resident histogram pool (260608-p90).
/// `false` (the default, CpuBackend) means the learner's `resident_eligible` gate
/// ANDs this in and ALWAYS takes the byte-unchanged host path. RocmBackend returns
/// `true`.
fn resident_pool_supported(&self) -> bool {
    false
}
```
New `on_device_growth_supported()` is the identical shape (`-> bool { false }`). Per
RESEARCH Open Q1 / A2 + Pitfall 2 (`GpuBackend<R>` is ONE impl shared by
ROCm/CUDA/WGPU, type aliases at `lib.rs:2110,2116,2123`): **keep it `false` for
`GpuBackend<R>` in Slice 0** — do NOT override to true (no kernel exists; activation is a
Slice-1 concern). The env AND-gate keeps everything off regardless.

**Sibling default-false discriminator** (`lib.rs:898-900`) — shows the established
"one backend overrides, others inherit" idiom the seam mirrors:
```rust
fn prefers_host_partition(&self) -> bool {
    false
}
```

**Seam method shape** — return type is BLOCKED on D-03-RESOLVED Option A (see the `P`
file below). The default impl returns `Ok(None)` (NOT a typed `Err(NotSupported)` —
RESEARCH anti-pattern: keep the default path error-noise-free):
```rust
/// Grow a whole tree ON-DEVICE (ODL-01). `Ok(None)` = "I did not grow it" → the
/// learner falls through to the host path (default-path error-noise-free, D-03).
/// Default: Ok(None).
/// cubecl-0.10 (Slice 1): no global barrier; Atomic<i64> broken; wrapping_add not an
/// intrinsic; plane-sum <= plane width; launch_unchecked is unsafe.
fn grow_tree_on_device(&self, /* inputs TBD */) -> Result<Option<(Tree, P)>, ComputeError> {
    Ok(None)
}
```
**Cargo.toml edit required:** add `lgbm-model = { path = "../lgbm-model" }` to
`crates/lgbm-compute/Cargo.toml` so `Tree` is nameable. Verified acyclic — `lgbm-model`
deps are only `lgbm-core` + `lgbm-dataset` + `thiserror`, NOT `lgbm-compute`
(RESEARCH §"Standard Stack" `[VERIFIED]`). **Never** `use lgbm_treelearner::...` in
`lgbm-compute` (Pitfall 1 — the `DataPartition` crate cycle warning sign).

**`GpuBackend<R>` override** (impl at `lib.rs:2126`): also returns `Ok(None)` in Slice 0
(no-op) — this is the SC#2 proof that the default route is provably untouched.

---

### `crates/lgbm-treelearner/src/learner.rs` (tree-learner, event-driven routing)

**Analog:** the `resident_eligible` lifecycle — struct field, `new` init, `train_inner`
recompute, routing gate. D-05 mirrors the **AND-gate** but **intentionally diverges** by
reading the env ONCE at `new` (resident recomputes per-train because it size-gates on
`num_data`; on-device has no per-train size input). Do NOT normalize back into
`train_inner`.

**Struct field to add beside** (`learner.rs:286` `resident_eligible`, `:294`
`fused_eligible`):
```rust
/// 260608-p90: whether THIS train is device-resident-eligible ...
resident_eligible: bool,
```
Add `on_device_eligible: bool` here. (Doc it as "computed ONCE at `new`, NOT recomputed
in `train_inner` — D-05 intentional divergence from `resident_eligible`".)

**`new` signature already carries the backend** (`learner.rs:430-436`) — D-05 feasible:
```rust
pub fn new(
    backend: &'b B,
    client: &'b ComputeClient<B::Runtime>,
    cfg: GainConfig,
    num_leaves: i32,
    max_depth: i32,
) -> Self {
```
**`new` init site** — the `Self { .. }` literal (`learner.rs:437-467`). Existing
`resident_eligible` is `false`-defaulted at `:458`:
```rust
// Default OFF: recomputed per train in `train_inner` (260608-p90).
resident_eligible: false,
```
The new field instead computes the AND-gate INLINE at construction:
```rust
on_device_eligible: backend.on_device_growth_supported() && cuda_on_device_env(),
```

**The AND-gate precedent** — `resident_eligible` recompute at top of `train_inner`
(`learner.rs:680-687`); copy the `backend.<discriminator>() && ...` ANDing, NOT the
placement:
```rust
self.resident_eligible = crate::resident_pool::resident_eligible(
    self.backend.resident_pool_supported(),   // <- AND-gate on the discriminator
    num_data,
    &features,
    &self.constraints,
    capture_snapshots,
    &self.cfg,
);
```

**Routing fork** — goes at the TOP of `train_inner`, AHEAD of the `:680`
`resident_eligible` block. `train_inner` returns a **4-tuple**
`(Tree, Vec<SplitSnapshot>, ColSamplerTrace, DataPartition)` (`learner.rs:658-664`):
```rust
fn train_inner(
    &mut self,
    gradients: &[f32],
    hessians: &[f32],
    _is_first_tree: bool,
    capture_snapshots: bool,
) -> Result<(Tree, Vec<SplitSnapshot>, ColSamplerTrace, DataPartition), TreeLearnerError> {
```
So the fork must synthesize that 4-tuple and reconstruct `DataPartition` from the payload
`P`:
```rust
// At the TOP of train_inner, ahead of the resident_eligible block (:680):
if self.on_device_eligible {
    if let Some((tree, payload)) = self.backend.grow_tree_on_device(/* ... */)? {
        let part = DataPartition::from_payload(payload); // reconstruct in this crate
        return Ok((tree, Vec::new(), ColSamplerTrace::default(), part));
    }
    // Ok(None) → fall through to the existing host/resident path (D-02/D-03).
}
```
**D-02 critical reconciliation:** production uses `Ok(None) ⇒ fall through` ONLY. The
`unwrap_or_else(|| host_grow(..))` host fallback lives in the **oracle TEST**, never here.

**Env-parse helper** — add `cuda_on_device_env()` to this crate. INVERSE default (OFF
unless `=1`), contrasting the on-by-default `autotune_enabled`. Two verified idiom
sources:
```rust
// crates/lgbm-compute/src/kernels/autotune.rs:85-86 — on UNLESS "0":
pub fn autotune_enabled() -> bool {
    !matches!(std::env::var("LGBM_AUTOTUNE").as_deref(), Ok("0"))
}
```
```rust
// crates/lgbm-treelearner/src/resident_pool.rs:141-149 — explicit tri-state match:
match std::env::var("LGBM_RESIDENT_FORCE").ok().as_deref() {
    Some("0") => return false,
    Some("1") => return true,
    _ => {}
}
```
New helper (inverse of `autotune_enabled` — OFF unless exactly `"1"`):
```rust
fn cuda_on_device_env() -> bool {
    matches!(std::env::var("LGBM_CUDA_ON_DEVICE").as_deref(), Ok("1"))
}
```
This guarantees SC#1: var unset ⇒ `on_device_eligible == false` on every backend ⇒ fork
never taken ⇒ byte-unchanged.

---

### `crates/oracle-harness/tests/learner_parity.rs` (test, transform/assertion)

**Analog:** `assert_gpu_tree_matches_cpu_anchor` (`:2046`) + `cpu_anchor_tree` (`:2083`)
+ the `kernel_parity.rs:1597-1629` near-tie acceptance. Recommended (RESEARCH): a
tie-aware GENERALIZATION reused by both, not a duplicate.

**Tolerance const to reuse** (`learner_parity.rs:2041`):
```rust
const ROCM_LEAF_VALUE_TOL: f64 = 1e-5;
```

**Structural-assert body to generalize** (`learner_parity.rs:2046-2079`) — 8 bit-exact
structural fields + the 1e-5 leaf-value envelope:
```rust
fn assert_gpu_tree_matches_cpu_anchor(
    gpu: &lgbm_model::Tree,
    anchor: &lgbm_model::Tree,
    label: &str,
) {
    assert_eq!(gpu.num_leaves, anchor.num_leaves, "{label} vs cpu-anchor: num_leaves");
    assert_eq!(gpu.split_feature, anchor.split_feature, "{label} vs cpu-anchor: split_feature");
    assert_eq!(gpu.threshold, anchor.threshold, "{label} vs cpu-anchor: threshold");
    assert_eq!(gpu.decision_type, anchor.decision_type, "{label} vs cpu-anchor: decision_type");
    assert_eq!(gpu.left_child, anchor.left_child, "{label} vs cpu-anchor: left_child");
    assert_eq!(gpu.right_child, anchor.right_child, "{label} vs cpu-anchor: right_child");
    assert_eq!(gpu.leaf_count, anchor.leaf_count, "{label} vs cpu-anchor: leaf_count");
    assert_eq!(gpu.internal_count, anchor.internal_count, "{label} vs cpu-anchor: internal_count");
    // ... leaf_value length + per-leaf abs_diff <= ROCM_LEAF_VALUE_TOL loop (:2064-2074)
}
```
**D-04 tie-aware change:** the `decision_type` line at `:2054` is the ONE field that must
change. `default_left` is **bit1** of the packed `i8` (`DEFAULT_LEFT_MASK = 2`,
`tree.rs:47-48`; packing doc `tree.rs:24-26`). The new `assert_on_device_tree_matches_cpu_anchor`:
1. Compare `decision_type & !2` strictly per node (categorical bit0 + missing bits2-3 stay
   exact).
2. For bit1, apply the near-tie acceptance PER INTERNAL NODE (the existing logic is
   per-`SplitInfo`; lift to per-node index).

**Anchor builder to reuse as-is** (`learner_parity.rs:2083-2096`):
```rust
fn cpu_anchor_tree(
    features: &[FeatureColumn],
    g: &[f32],
    h: &[f32],
    num_leaves: i32,
    max_depth: i32,
) -> lgbm_model::Tree {
    let cpu_backend = lgbm_compute::CpuBackend;
    let cpu_client = lgbm_compute::runtime::cpu_client();
    let mut cpu_learner =
        SerialTreeLearner::new(&cpu_backend, &cpu_client, cfg(), num_leaves, max_depth)
            .with_features(features.to_vec());
    cpu_learner.train(g, h, true).expect("cpu anchor train ok")
}
```

**Near-tie acceptance to lift** (`kernel_parity.rs:1612-1629`) — accept a `default_left`
flip ONLY if same threshold AND same left_count AND net gains tie within f32; otherwise
hard-fail:
```rust
let hip_default_left = hip_raw[9] != 0.0;
if hip_default_left != si.default_left {
    let same_threshold = hip_raw[1] as u32 == si.threshold;
    let same_left_count = hip_raw[3] as i32 == si.left_count;
    let net_gain_tie =
        (hip_vals[0] - cpu_anchor_f32[0]).abs() <= HIP_SANITY_REL
            * cpu_anchor_f32[0].abs().max(1.0);
    assert!(
        same_threshold && same_left_count && net_gain_tie,
        "... default_left flip on a NON-tie split ... real wrong-direction divergence ..."
    );
}
```
At `Tree` level: per internal-node `i`, compare `threshold[i]` (exact f64) and the
left-child row count; the "net gain" tie reduces to threshold+count equality since
`split_gain` is predict-irrelevant metadata. D-01: the test obtains the tree via
`backend.grow_tree_on_device(..)?.map(|(t,_)| t).unwrap_or_else(|| host_grow(..))` then
asserts against `cpu_anchor_tree(..)` — LIVE & GREEN before any kernel exists.

**Reference test for env-window discipline** (`learner_parity.rs:2099-2141`)
`learner_parity_resident_equals_host_tree_on_hip` — the SC#2 force-eligible test mirrors
its `with_resident(true/false)` + `assert!(backend.<discriminator>())` structure (and the
`FORCE_ENV_LOCK` guard pattern if the SC#2 test sets `LGBM_CUDA_ON_DEVICE`).

---

### Partition payload `P` (CREATE — `lgbm-core` or `lgbm-dataset`)

**Analog (struct shape):** `EfbSamples` (`lgbm-dataset/src/dataset.rs:58-70`) — the
closest existing plain POD data-struct: public Vec + scalar fields, `#[derive(Debug,
Clone)]`, named after a C++ data layout, no behavior. Copy this shape:
```rust
#[derive(Debug, Clone)]
pub struct EfbSamples {
    pub sample_indices: Vec<Vec<i32>>,
    pub sample_values: Vec<Vec<f64>>,
    pub num_per_col: Vec<i32>,
    pub num_sample_col: i32,
    pub total_sample_cnt: i32,
}
```

**Analog (fields to carry):** the four fields `DataPartition` wraps
(`data_partition.rs:33-42`) — the payload IS the raw leaf-row layout:
```rust
pub struct DataPartition {
    num_data: i32,
    indices: Vec<u32>,      // row ids grouped by leaf
    leaf_begin: Vec<i32>,   // per-leaf start offset
    leaf_count: Vec<i32>,   // per-leaf row count
}
```
**Recommended `P`** (RESEARCH Open Q2): a small named struct (clearer than bare
`Vec<i32>`) in `lgbm-core` or `lgbm-dataset` — both are below `lgbm-compute`, so naming
`P` in the `Backend` seam is acyclic:
```rust
/// Raw leaf-row index layout a device-grown tree returns (ODL-01, D-03 Option A).
/// `lgbm-treelearner` reconstructs `DataPartition` from this in the train_inner fork.
#[derive(Debug, Clone)]
pub struct LeafPartitionLayout {
    pub num_data: i32,
    pub indices: Vec<u32>,
    pub leaf_begin: Vec<i32>,
    pub leaf_count: Vec<i32>,
}
```
**Reconstruction seam:** add a `DataPartition::from_payload(LeafPartitionLayout)` (or
`From`) in `lgbm-treelearner/src/data_partition.rs` (it already names both types). In
Slice 0 the seam returns `Ok(None)`, so `P` is never CONSTRUCTED — only its TYPE must be
nameable, so the reconstruct fn can be a thin field-move. **Place it in `lgbm-dataset`**
(it is the data-layout crate and `EfbSamples`/`Metadata` precedents live there);
`lgbm-core` is acceptable if a dataset dep is undesirable on `lgbm-compute` (verify
`lgbm-compute` already deps both — RESEARCH confirms `lgbm-core` + `lgbm-dataset`).

## Shared Patterns

### Default-false trait-method discriminator (additive gating, no global state)
**Source:** `crates/lgbm-compute/src/lib.rs:935` (`resident_pool_supported`), sibling
`:898` (`prefers_host_partition`).
**Apply to:** `on_device_growth_supported()` on `Backend`. One backend may override;
CPU/ROCm inherit `false` → byte-unchanged. Mirrors the SHIPPED idiom exactly.

### Decide-once eligibility AND-gate
**Source:** `crates/lgbm-treelearner/src/resident_pool.rs:99-154` (the `backend_supported`
short-circuit at `:107-109`) + `learner.rs:680` (the `backend.resident_pool_supported()`
ANDing).
**Apply to:** `on_device_eligible = backend.on_device_growth_supported() && cuda_on_device_env()`.
ANDing the discriminator means `CpuBackend` (false) can NEVER be eligible. **Divergence
(D-05):** compute at `new`, not in `train_inner` — intentional, do not normalize.

### `LGBM_*` env toggle parsing (no injection surface, V5)
**Source:** `autotune.rs:86` (`!matches!(..Ok("0"))`, on-by-default) and
`resident_pool.rs:141` (tri-state match). `LGBM_CUDA_ON_DEVICE` is the INVERSE: OFF
unless `Ok("1")`.
**Apply to:** the new `cuda_on_device_env()` helper.

### Anchor-pinned parity (never compare two GPU f32 paths)
**Source:** `learner_parity.rs:2046,2083` + `kernel_parity.rs:1597`. def-f8u-01 rule
(commit 1832206/d82611b): structure bit-exact + 1e-5 leaf envelope vs the cpu f64 anchor.
**Apply to:** `assert_on_device_tree_matches_cpu_anchor`.

### cubecl-0.10 gotcha checklist (DOC-ONLY this slice)
**Source:** ROADMAP Notes / CONTEXT canonical_refs. Bake into `grow_tree_on_device`'s
doc-comment for Slice 1: no global barrier; `Atomic<i64>` broken; `wrapping_add` not an
intrinsic; plane-sum ≤ plane width; `launch_unchecked` is `unsafe`. NO kernel in Phase 14.

## No Analog Found

None. Every in-scope file has a strong in-codebase analog. The single novelty is the
per-internal-node lift of the per-`SplitInfo` near-tie logic (`kernel_parity.rs:1597`) —
the logic exists; only its iteration granularity changes (RESEARCH: "the only non-trivial
code in the phase").

## Metadata

**Analog search scope:** `crates/lgbm-compute/src/`, `crates/lgbm-treelearner/src/`,
`crates/oracle-harness/tests/`, `crates/lgbm-model/src/`, `crates/lgbm-core/src/`,
`crates/lgbm-dataset/src/`.
**Files scanned:** 8 (lib.rs, learner.rs, learner_parity.rs, kernel_parity.rs, tree.rs,
resident_pool.rs, autotune.rs, data_partition.rs + dataset.rs struct survey).
**All RESEARCH.md line citations re-verified accurate this session.**
**Pattern extraction date:** 2026-06-28
