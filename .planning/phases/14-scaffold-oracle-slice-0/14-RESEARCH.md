# Phase 14: Scaffold + Oracle (Slice 0) - Research

**Researched:** 2026-06-28
**Domain:** Additive trait-seam wiring + anchor-pinned oracle scaffold (Rust workspace, CubeCL backends; no new kernel, no new Cargo feature)
**Confidence:** HIGH (all findings VERIFIED against the live codebase at the CONTEXT canonical refs)

## Summary

This is a **pure additive-wiring scaffold phase** with one hard constraint: zero behavior
change with `LGBM_CUDA_ON_DEVICE` unset. The CONTEXT decisions (D-01..D-05) and the
established `prefers_host_partition` / `resident_eligible` idioms are all real and present
in the codebase exactly where CONTEXT says they are. Four of the five locked decisions are
**feasible as written** against the live code; the env-read-at-`new` placement (D-05) is
confirmed feasible because `SerialTreeLearner::new` already receives `backend: &'b B`.

**One locked decision is NOT feasible as literally specified and must be resolved by the
planner: D-03's return type `Result<Option<(Tree, DataPartition)>>` cannot be placed on the
`Backend` trait.** The `Backend` trait lives in `lgbm-compute`; `DataPartition` lives in
`lgbm-treelearner`, which **already depends on `lgbm-compute`** — naming it in a `Backend`
method return type creates a **circular crate dependency** (impossible in Cargo). `Tree`
(in `lgbm-model`) is importable by `lgbm-compute` without a cycle, but `DataPartition` is
not. The planner must pick a resolution (Section "D-03 Feasibility" below) before writing
the seam signature. This is the single highest-risk finding of the phase.

**Primary recommendation:** Mirror the `resident_pool_supported()` discriminator idiom
exactly for `on_device_growth_supported()` (default-false on `Backend`, overridden on
`GpuBackend<R>`), cache `on_device_eligible` in `SerialTreeLearner::new` (D-05), and resolve
D-03 by returning a **lower-crate-friendly partition payload** (e.g. the raw leaf-row layout
that `DataPartition` already wraps, expressed in `lgbm-core`/`lgbm-dataset` terms) instead of
`DataPartition` itself — keeping the seam on `Backend` and avoiding the cycle. Build the
tie-aware oracle NOW (D-04) by decoding the `default_left` bit out of the `Tree.decision_type`
vector and lifting the `kernel_parity.rs:1597` near-tie acceptance to per-node granularity.

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| `grow_tree_on_device` seam method | Compute (`lgbm-compute` `Backend` trait) | — | Polymorphic over `&impl Backend`; the learner holds `&'b B`, so the method must live on the trait to be callable. BUT the return type pulls in higher-crate types (see D-03 feasibility). |
| `on_device_growth_supported()` discriminator | Compute (`Backend` trait) | — | Pure `-> bool`, exactly like `resident_pool_supported()`; no cross-crate type issue. |
| Decide-once eligibility caching + routing fork | Tree learner (`SerialTreeLearner`) | — | `new` caches the env∧discriminator AND-gate; `train_inner` early-returns. Mirrors `resident_eligible`. |
| Env gate (`LGBM_CUDA_ON_DEVICE`) | Tree learner (read in `new`, D-05) | — | Inverse-default bool toggle; no per-train size input (unlike resident), so reading once at `new` is correct. |
| Anchor-pinned tie-aware oracle | Test harness (`oracle-harness/tests`) | — | Tests-only; depends on every crate, so it can freely name `lgbm_model::Tree`, `CpuBackend`, `RocmBackend`, and `SerialTreeLearner`. |

## Standard Stack

No new external crates. This phase is additive wiring inside the existing workspace.

| Crate (internal) | Role in this phase | Verified |
|------------------|--------------------|----------|
| `lgbm-compute` | Defines `Backend` trait + `CpuBackend`/`GpuBackend<R>`. Add discriminator + seam method here. | `[VERIFIED: crates/lgbm-compute/src/lib.rs:495 trait Backend; :1239 CpuBackend; :2037 GpuBackend<R>]` |
| `lgbm-treelearner` | `SerialTreeLearner` (`new` + `train_inner`) + `DataPartition`. Cache `on_device_eligible`, add fork. | `[VERIFIED: crates/lgbm-treelearner/src/learner.rs:203 struct, :430 new, :658 train_inner; src/data_partition.rs:33 DataPartition]` |
| `lgbm-model` | Owns `Tree` (the seam's tree return type + oracle anchor type). | `[VERIFIED: crates/lgbm-model/src/tree.rs:72 pub struct Tree]` |
| `oracle-harness` | Test crate; holds the oracle scaffold + tie-aware comparator. | `[VERIFIED: crates/oracle-harness/tests/learner_parity.rs:2046,2083; kernel_parity.rs:1597]` |

**Installation:** none. No `cargo add`. The only Cargo.toml edit *potentially* required is adding
`lgbm-model = { path = "../lgbm-model" }` to `lgbm-compute` IF the seam returns `Tree` directly
(see D-03). `lgbm-model` depends only on `lgbm-core` + `lgbm-dataset` + `thiserror`, so this edge
is **acyclic and safe** `[VERIFIED: crates/lgbm-model/Cargo.toml — no lgbm-compute/lgbm-treelearner dep]`.

## Package Legitimacy Audit

**Not applicable.** This phase installs **zero external packages**. The only dependency change
under consideration is an *internal path dependency* (`lgbm-compute` → `lgbm-model`), which is not
a registry package and carries no slopsquatting/legitimacy surface. No `npm`/`pip`/`cargo add`
of third-party crates occurs.

## D-03 Feasibility — Seam Return Type (BLOCKING decision for the planner)

**Locked decision D-03:** `grow_tree_on_device` returns `Result<Option<(Tree, DataPartition)>>`
on the `Backend` trait, default `Ok(None)`.

**Verified crate-dependency reality:**

```
lgbm-model      → lgbm-core, lgbm-dataset            (owns Tree)
lgbm-compute    → lgbm-core, lgbm-dataset, cubecl    (owns Backend trait; does NOT dep lgbm-model today)
lgbm-treelearner→ lgbm-compute, lgbm-model, ...      (owns DataPartition)
```
`[VERIFIED: the three Cargo.toml files read this session]`

- `Tree` ∈ `lgbm-model`. `lgbm-compute` can add a dep on `lgbm-model` → **acyclic, feasible.**
- `DataPartition` ∈ `lgbm-treelearner`, which **already depends on `lgbm-compute`.** A `Backend`
  method (in `lgbm-compute`) returning `DataPartition` requires `lgbm-compute` → `lgbm-treelearner`
  → **CIRCULAR dependency. INFEASIBLE in Cargo.** `[VERIFIED]`

The `Backend` trait today names only `lgbm-core`/`lgbm-dataset` types (`SplitInfo`, `ComputeError`,
`BinColumn`) and refers to `DataPartition` *only in doc-comments*, never in a signature
`[VERIFIED: grep of lib.rs — DataPartition appears only in /// comments]`. That is precisely
because of this cycle.

**Resolution options for the planner (recommend Option A):**

| # | Approach | Cost | Keeps seam on `Backend`? | Cycle? |
|---|----------|------|--------------------------|--------|
| **A (recommended)** | Seam returns `Result<Option<(Tree, P)>>` where `P` is a **lower-crate partition payload** (the raw leaf-row index layout `DataPartition` already wraps), defined in `lgbm-core` or `lgbm-dataset` (or returned as plain `Vec<i32>` + leaf bounds). `lgbm-treelearner` reconstructs `DataPartition` from `P` in the fork. Add `lgbm-model` dep to `lgbm-compute` for `Tree`. | Low; additive | Yes | No |
| B | Define the seam on a NEW extension trait in `lgbm-treelearner` (where both `Tree` and `DataPartition` are nameable), not on `Backend`. | Med; diverges from D-01 "add to `Backend`"; can't specialize per-backend without specialization | No | No |
| C | Move `DataPartition` down into `lgbm-core`/`lgbm-dataset`. | High; not additive; touches many call sites | Yes | No |
| D | Seam returns only `Result<Option<Tree>>`; partition handled separately by the learner. | Low | Yes | No — but drops the partition half of D-03 |

Option A honors D-01 (method on `Backend`), D-02/D-03 (`Ok(None)` default, `(tree, partition)`
shape), and the "additive only" constraint, at the cost of one thin payload type. **Flag this to
the planner as a required pre-implementation decision** — the exact signature cannot be written
until it is chosen. `[VERIFIED + ASSUMED resolution]`

## Architecture Patterns

### Pattern 1: Default-false trait-method discriminator (mirror exactly)

**What:** A `-> bool` method on `Backend` defaulting to `false`; only the relevant backend
overrides it. Leaves CPU/ROCm byte-unchanged because the learner ANDs it into an eligibility gate.
**When to use:** the `on_device_growth_supported()` discriminator (ODL-01).
**Example (the precedent to copy):**
```rust
// Source: crates/lgbm-compute/src/lib.rs:935 [VERIFIED]
fn resident_pool_supported(&self) -> bool {
    false                       // default on Backend; CpuBackend inherits → never eligible
}
// CpuBackend (lib.rs:1242) inherits the default; GpuBackend<R> (lib.rs:2126) overrides.
```
New method to add, same shape:
```rust
/// Whether this backend grows whole trees ON-DEVICE (ODL-01). Default false →
/// CpuBackend/ROCm inherit and the learner's on_device gate is always off.
fn on_device_growth_supported(&self) -> bool { false }
```

### Pattern 2: Decide-once eligibility AND-gate

**What:** Compute eligibility once, ANDing the backend discriminator so an unsupported backend
can NEVER be eligible. CONTEXT D-05 deliberately reads the env at `new` (not `train_inner`).
**Example:**
```rust
// Source: crates/lgbm-treelearner/src/learner.rs:680 (train_inner) [VERIFIED]
self.resident_eligible = crate::resident_pool::resident_eligible(
    self.backend.resident_pool_supported(),   // <- AND-gate on the discriminator
    num_data, &features, &self.constraints, capture_snapshots, &self.cfg,
);
```
D-05 placement (in `new`, which already has the backend) — **feasible, verified:**
```rust
// Source: crates/lgbm-treelearner/src/learner.rs:430 — `new(backend: &'b B, ...)` [VERIFIED]
// In new(), add a cached field initialized as:
let on_device_eligible =
    backend.on_device_growth_supported() && cuda_on_device_env();   // see Pattern 3
```
Add `on_device_eligible: bool` next to `resident_eligible: bool` (struct line 286) and
`fused_eligible` (line 294). Initialize in the `Self { .. }` literal at lib `new` (lines 437-467),
NOT defaulted-then-recomputed in `train_inner` (that is the intentional divergence in D-05 —
do not "normalize" it back).

### Pattern 3: `LGBM_*` env idiom — INVERSE default for this gate

**What:** The codebase reads `LGBM_*` toggles via `matches!(env::var(..).as_deref(), Ok(..))`.
Existing toggles are mostly **on-by-default** (disabled only by `"0"`). `LGBM_CUDA_ON_DEVICE` is
the **inverse**: OFF unless exactly `"1"`.
**Examples (verified idioms):**
```rust
// Source: crates/lgbm-compute/src/kernels/autotune.rs (autotune_enabled) [VERIFIED]
!matches!(std::env::var("LGBM_AUTOTUNE").as_deref(), Ok("0"))      // on unless "0"

// Source: crates/lgbm-treelearner/src/resident_pool.rs:141 (LGBM_RESIDENT_FORCE) [VERIFIED]
match std::env::var("LGBM_RESIDENT_FORCE").ok().as_deref() {
    Some("0") => return false, Some("1") => return true, _ => {} }
```
New helper (inverse default — OFF unless `=1`):
```rust
fn cuda_on_device_env() -> bool {
    matches!(std::env::var("LGBM_CUDA_ON_DEVICE").as_deref(), Ok("1"))
}
```
This guarantees SC#1: with the var unset, `on_device_eligible` is false on every backend,
so the production fork is never taken → byte-unchanged.

### Pattern 4: Routing fork at the top of `train_inner`

**What:** A decide-once early-return ahead of the resident/host branches (D-02/D-03).
`train_inner` returns a **4-tuple** `(Tree, Vec<SplitSnapshot>, ColSamplerTrace, DataPartition)`
`[VERIFIED: learner.rs:664]`, so the fork must produce that 4-tuple. With `Ok(None)` the default
path falls through untouched.
```rust
// At the TOP of train_inner (ahead of the resident_eligible block at :680):
if self.on_device_eligible {
    if let Some((tree, part)) = self.backend.grow_tree_on_device(/* ... */)? {
        // synthesize empty snapshots + default trace (production never captures here)
        return Ok((tree, Vec::new(), ColSamplerTrace::default(), part /* from payload P */));
    }
    // Ok(None) → fall through to the existing host/resident path (D-02/D-03).
}
```
**D-02 reconciliation (critical):** the production fork uses `Ok(None) ⇒ fall through`. The
`unwrap_or_else(|| host_grow(..))` host-fallback from D-01 lives **only in the oracle test**, never
in `train_inner`. Do not route production through the fallback.

### Pattern 5: Anchor-pinned oracle (extend, never compare two GPU paths)

**What:** Structure bit-exact + `1e-5` leaf envelope vs the cpu f64 anchor (def-f8u-01 rule).
**Verified existing assets:**
```rust
// Source: crates/oracle-harness/tests/learner_parity.rs:2046 [VERIFIED]
fn assert_gpu_tree_matches_cpu_anchor(gpu: &lgbm_model::Tree, anchor: &lgbm_model::Tree, label: &str)
// :2051-2058 — assert_eq! on num_leaves, split_feature, threshold, decision_type,
//              left_child, right_child, leaf_count, internal_count  (BIT-EXACT)
// :2064-2074 — leaf_value within ROCM_LEAF_VALUE_TOL = 1e-5 (:2041)

// Source: crates/oracle-harness/tests/learner_parity.rs:2083 [VERIFIED]
fn cpu_anchor_tree(features, g, h, num_leaves, max_depth) -> lgbm_model::Tree
// builds via CpuBackend + runtime::cpu_client() + SerialTreeLearner::new(...).train(...)
```
**D-04 tie-aware extension nuance (important):** `assert_gpu_tree_matches_cpu_anchor` currently
does a single strict `assert_eq!(gpu.decision_type, anchor.decision_type)`. But `default_left`
is **bit1 of `decision_type`** (`kDefaultLeftMask = 2`), packed with categorical (bit0) and
missing_type (bits2-3) `[VERIFIED: crates/lgbm-model/src/tree.rs:24-26,53-55,135 get_decision_type]`.
The tie-aware comparator must therefore:
1. Compare `decision_type` **masking out bit1** strictly (`dt & !2` must be exactly equal).
2. For bit1 (`default_left`), apply the `kernel_parity.rs:1597-1620` near-tie acceptance
   **per internal node**, accepting a flip ONLY when that node's `threshold` AND counts
   (`left_count`/`internal_count`) match — a flip on a non-tie node still hard-fails.

The existing near-tie logic operates per `SplitInfo` (one split); at `Tree` level it must operate
per internal-node index. This is the only non-trivial code in the phase.
```rust
// Source: crates/oracle-harness/tests/kernel_parity.rs:1597-1620 [VERIFIED] — the acceptance to lift:
// default_left flip is accepted ONLY if same_threshold && same_left_count && net gains equal
// within f32 precision (documented f32-vs-f64 near-tie, D-03a/04-ROCM-GAPS.md); otherwise hard-fail.
```
**D-04 discretion (CONTEXT):** new fn vs tie-aware-extended generalization of the existing
`assert_gpu_tree_matches_cpu_anchor`, and where `assert_on_device_tree_matches_cpu_anchor` lives —
left to planning. Recommend a tie-aware generalization reused by both (avoids duplicating the
structural asserts).

### Anti-Patterns to Avoid
- **Comparing two nondeterministic GPU f32 paths to each other** (def-f8u-01, commit 1832206/d82611b):
  always pin to the cpu f64 anchor. The whole oracle exists to enforce this.
- **Naming `DataPartition` in a `Backend` method signature** — creates the crate cycle (Section D-03).
- **Re-reading `LGBM_CUDA_ON_DEVICE` in `train_inner`** — D-05 forbids it; one syscall-per-tree for
  no benefit (no per-train size input, unlike resident).
- **Returning a typed `Err(NotSupported)` from the default seam** — D-03 chose `Ok(None)` to keep the
  default path error-noise-free.
- **Flipping `on_device_growth_supported()` to true in Slice 0** — see Open Question 1; safest is
  to keep it false this slice and exercise the seam only via the oracle test's direct call.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Backend capability flag | A global `static` / config switch | A default-false `Backend` method (`on_device_growth_supported`) | Mirrors `resident_pool_supported`; leaves CPU/ROCm byte-unchanged with no global state. |
| cpu f64 anchor for the oracle | A fresh anchor builder | `cpu_anchor_tree` (`learner_parity.rs:2083`) | Already the deterministic merge-gate reference. |
| Structural tree comparison | New per-field asserts | Generalize `assert_gpu_tree_matches_cpu_anchor` (`:2046`) | Already covers all 8 structural fields + 1e-5 leaf envelope. |
| f32-vs-f64 `default_left` tie acceptance | New tie heuristic | Lift `kernel_parity.rs:1597-1620` | Battle-tested, documented (04-ROCM-GAPS.md). |
| env toggle parsing | `std::env` ad hoc | The `matches!(env::var(..).as_deref(), Ok(..))` idiom | Consistent, no injection surface (T-13-01 accept). |

**Key insight:** every primitive this slice needs already exists; the work is *wiring + one
tie-aware comparator generalization*, not new capability.

## Common Pitfalls

### Pitfall 1: The `(Tree, DataPartition)` crate cycle
**What goes wrong:** writing the literal D-03 signature on `Backend` fails to compile with a Cargo
cyclic-dependency error.
**Why:** `DataPartition` ∈ `lgbm-treelearner` which depends on `lgbm-compute`.
**How to avoid:** resolve via Section "D-03 Feasibility" (recommend Option A — lower-crate payload)
BEFORE writing the signature.
**Warning sign:** any `use lgbm_treelearner::...` appearing in `lgbm-compute`.

### Pitfall 2: `GpuBackend<R>` is generic over ROCm AND CUDA AND WGPU
**What goes wrong:** a single `impl Backend for GpuBackend<R>` (`lib.rs:2126`) means an override of
`on_device_growth_supported()` applies to **ROCm and WGPU too**, not just CUDA — risking ROCm
behavior change.
**Why:** `RocmBackend`/`CudaBackend`/`WgpuBackend` are all `type` aliases of `GpuBackend<R>`
`[VERIFIED: lib.rs:2110,2116,2123]`.
**How to avoid:** in Slice 0 keep the discriminator returning the default `false` for `GpuBackend<R>`
(no kernel exists yet), and have the GpuBackend `grow_tree_on_device` override return `Ok(None)`.
The env AND-gate keeps everything off anyway. Per-runtime (`R`) discrimination for CUDA-only
activation is a Slice-1 concern, not Slice 0. (See Open Question 1.)

### Pitfall 3: `decision_type` strict-eq hides/breaks the tie branch
**What goes wrong:** keeping the existing `assert_eq!(gpu.decision_type, anchor.decision_type)` makes
the "tie-aware" comparator non-tie-aware (a legitimate near-tie flip would hard-fail), OR loosening
the whole byte makes it accept real categorical/missing-type divergences.
**Why:** `default_left` is only **bit1** of the packed `i8`.
**How to avoid:** mask bit1 out for the strict compare; apply tie acceptance to bit1 alone, gated on
matching threshold+counts (Pattern 5).

### Pitfall 4: cubecl-0.10 gotchas baked into the seam doc-comment (no kernels yet)
**What:** ROADMAP requires baking the checklist into this slice's seam doc-comment for Slice 1:
no global grid barrier; `Atomic<i64>` broken; `wrapping_add` is not an intrinsic; plane-sum ≤ plane
width; `launch_unchecked` is `unsafe`. **No kernel is written in Phase 14** — this is documentation
only, on `grow_tree_on_device`'s doc-comment. `[CITED: ROADMAP Notes / CONTEXT canonical_refs]`

## Runtime State Inventory

> Not a rename/refactor/migration phase — this is additive code wiring. No stored data, live-service
> config, OS-registered state, or build artifacts carry a renamed string. The only new runtime input
> is the `LGBM_CUDA_ON_DEVICE` environment variable (read at `SerialTreeLearner::new`), which is a new
> read, not a migration of existing state. **Nothing to migrate — verified: no string rename, no
> datastore key change, no persisted config touched.**

## Code Examples

### Adding the cached eligibility field + `new` init (D-05)
```rust
// crates/lgbm-treelearner/src/learner.rs
// struct field, near :286 (resident_eligible) / :294 (fused_eligible):
//   on_device_eligible: bool,
// in new() Self { .. } literal (:437-467), alongside resident_eligible: false:
//   on_device_eligible: backend.on_device_growth_supported() && cuda_on_device_env(),
// `new(backend: &'b B, ...)` already has `backend` in scope [VERIFIED: :430-436] — D-05 feasible.
```

### The default seam method (shape; final return type per D-03 resolution)
```rust
// crates/lgbm-compute/src/lib.rs — on `trait Backend`, near the resident seam (:935):
/// Grow a whole tree ON-DEVICE (ODL-01). `Ok(None)` = "I did not grow it" → the learner
/// falls through to the host path (default-path error-noise-free, D-03). Default: Ok(None).
/// cubecl-0.10 (Slice 1): no global barrier; Atomic<i64> broken; wrapping_add not an
/// intrinsic; plane-sum <= plane width; launch_unchecked is unsafe.
fn grow_tree_on_device(&self, /* inputs */) -> Result<Option<(Tree, P)>, ComputeError> {
    Ok(None)                    // P = lower-crate partition payload (D-03 Option A)
}
// GpuBackend<R> override (lib.rs:2126 impl) returns Ok(None) too in Slice 0 (no-op, SC#2).
```

### Oracle test exercising the seam via host-fallback (D-01, LIVE & GREEN)
```rust
// crates/oracle-harness/tests/learner_parity.rs (new test, beside :2099)
// The seam returns Ok(None) (no kernel); the TEST supplies the host stand-in:
let tree = backend.grow_tree_on_device(/* .. */)?
    .map(|(t, _part)| t)
    .unwrap_or_else(|| host_grow(/* SerialTreeLearner::new(...).train(...) */));
let anchor = cpu_anchor_tree(&features, &g, &h, num_leaves, max_depth);   // :2083
assert_on_device_tree_matches_cpu_anchor(&tree, &anchor, "slice0-host-fallback"); // tie-aware
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Compare two GPU f32 paths at 1e-6 | Pin BOTH to cpu f64 anchor; structure bit-exact + 1e-5 leaf | def-f8u-01 (commit 1832206/d82611b) | The reason ODL-02/D-04 exist; the oracle never compares GPU-to-GPU. |
| `resident_eligible` recomputed in `train_inner` (size-gated) | `on_device_eligible` read once at `new` (D-05) | Phase 14 | Intentional divergence: on-device has no per-train size input. |
| Device round-trip partition on ROCm | `prefers_host_partition` host routing (spike-035) | quick-260626-a6t | Confirms the additive-discriminator idiom the seam mirrors. |

**Deprecated/outdated:** none relevant — the idioms to copy are current and shipped.

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | D-03 resolved via Option A (lower-crate partition payload `P`) | D-03 Feasibility | If planner picks B/C/D the seam signature + fork differ; A1 is a recommendation, not a verified decision — planner must choose. |
| A2 | Slice 0 keeps `on_device_growth_supported()` returning false on `GpuBackend<R>` (activation deferred to Slice 1) | Pitfall 2 / Open Q1 | If set true now, must verify ROCm/WGPU stay byte-unchanged via the env AND-gate (they do, but it widens the blast radius). |
| A3 | The full bit-exact merge gate runs via `cargo test --workspace` (oracle-harness suites + per-crate tests) | Validation Architecture | If a narrower command is the canonical gate, the verification step name is wrong; verify with the planner/maintainer. |

## Open Questions (RESOLVED)

1. **Should `on_device_growth_supported()` return true on `GpuBackend<R>` in Slice 0, or stay false until Slice 1?**
   - What we know: the discriminator is ANDed with `LGBM_CUDA_ON_DEVICE` (off by default), and the
     seam returns `Ok(None)`, so EITHER choice is byte-unchanged with the env unset.
   - What's unclear: `GpuBackend<R>` is generic (ROCm/CUDA/WGPU share one impl); a `true` here is not
     CUDA-specific without runtime-`R` discrimination.
   - Recommendation: **stay false in Slice 0** (no kernel to grow); exercise the seam via the oracle
     test's direct call + host-fallback. Flip to true (CUDA-only) in Slice 1 when a kernel lands.
   - **RESOLVED (planning): stay false.** Adopted in Plan 14-01 Task 2b (`GpuBackend<R>` keeps
     `on_device_growth_supported()` = false + `grow_tree_on_device` no-op override). Byte-safe under the env AND-gate.

2. **Which lower-crate type carries the partition payload `P` (D-03 Option A)?**
   - What we know: `DataPartition` (treelearner) can't appear on `Backend`; `Tree` (model) can.
   - What's unclear: whether to return the raw leaf-row layout as `Vec<i32>` + leaf bounds, or a small
     named struct in `lgbm-core`/`lgbm-dataset`.
   - Recommendation: smallest additive type that the learner can turn into `DataPartition` in the fork;
     decide at plan time. In Slice 0 the seam returns `Ok(None)` so `P` is never constructed yet — only
     its *type* must be nameable.
   - **RESOLVED (planning): named struct `LeafPartitionLayout` in `lgbm-dataset`** (fields mirror
     `DataPartition`: `num_data:i32, indices:Vec<u32>, leaf_begin:Vec<i32>, leaf_count:Vec<i32>`).
     Adopted in Plan 14-01 Task 1; `DataPartition::from_payload` reconstructs it in Plan 14-02.

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| Rust toolchain (edition 2024) | whole workspace | ✓ (assumed; repo builds) | rust-version 1.95 `[VERIFIED: Cargo.toml]` | — |
| cubecl `cpu` runtime | CPU merge gate + cpu anchor | ✓ (default feature) | cubecl 0.10 (workspace) | — |
| ROCm GPU (`--features rocm`) | the rocm-gated oracle tests in `learner_parity.rs` | ✓ but **spoofed 8-CU APU** (MEMORY: gfx1152, not gfx1100) | — | The CPU-side Slice-0 oracle (host-fallback) runs WITHOUT a GPU; only the rocm-feature tests need it. |

**Missing dependencies with no fallback:** none — the Slice-0 deliverables (seam + discriminator +
fork + tie-aware oracle on the host-fallback tree) are all exercisable on the CPU path. The ROCm GPU
is only needed for the rocm-feature subset of `learner_parity` and is present (spoofed but parity-valid).

## Validation Architecture

> nyquist_validation = true `[VERIFIED: .planning/config.json]` — section included.

### Test Framework
| Property | Value |
|----------|-------|
| Framework | Rust built-in test harness (`cargo test`) + the `oracle-harness` integration crate |
| Config file | per-crate `Cargo.toml`; tests under `crates/*/tests/` and inline `#[test]` (learner.rs has unit tests at :4363+) |
| Quick run command | `cargo test -p lgbm-treelearner` (fork + struct field) and `cargo test -p oracle-harness --test learner_parity` (oracle) |
| Full suite command | `cargo test --workspace` (the full bit-exact merge gate) `[ASSUMED A3 — confirm canonical gate]` |

### Phase Requirements → Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| ODL-01 (SC#1) | Merge gate byte-unchanged, `LGBM_CUDA_ON_DEVICE` unset: CPU/ROCm/host-CUDA grow identical trees | integration (existing gate) | `cargo test -p oracle-harness --test raw_bin_train_parity` (`raw_bin_train_matches_cpp_golden`, :119) + `--test learner_parity` (63 tests) | ✅ `[VERIFIED]` |
| ODL-01 (SC#2) | Seam + `on_device_growth_supported()` exist; fork reachable; `GpuBackend<R>` override returns `Ok(None)`/no-op; default path untouched | new test + existing gate | `cargo test -p oracle-harness --test learner_parity` (new: force-eligible → fork returns the host-fallback tree) | ❌ Wave 0 (new test) |
| ODL-02 (SC#3) | `assert_on_device_tree_matches_cpu_anchor` tie-aware oracle compiles + passes against the host-fallback tree (structure bit-exact + 1e-5 leaf + tie-aware `default_left`) | new test | `cargo test -p oracle-harness --test learner_parity` (new oracle test) | ❌ Wave 0 (new test + comparator generalization) |

### Observable / testable for each success criterion
- **SC#1 (byte-unchanged gate):** with `LGBM_CUDA_ON_DEVICE` unset, `on_device_eligible == false` on
  every backend (env AND-gate), so the fork is never entered; `raw_bin_train_matches_cpp_golden`,
  `learner_parity` (63 tests `[VERIFIED count]`), `kernel_parity`, and the lgbm/treelearner/compute
  suites stay green AND byte-identical to master. Observable: the full suite passes unchanged.
- **SC#2 (seam reachable, default untouched):** a new test forces `on_device_eligible` (constructs a
  backend whose discriminator is true, or calls `grow_tree_on_device` directly) and asserts the fork
  returns the host-fallback tree (`Ok(None)` ⇒ fall through, or `Some` ⇒ early return). The
  `GpuBackend<R>` override returning `Ok(None)` is the proof the default route is provably untouched.
  Observable: fork branch is covered AND the unforced path is identical to master.
- **SC#3 (dormant tie-aware oracle):** the new `assert_on_device_tree_matches_cpu_anchor` runs LIVE
  (D-01) against `backend.grow_tree_on_device(..)?.unwrap_or_else(host_grow)` and passes against the
  cpu f64 anchor; the tie branch is present but unexercised (no kernel produces a flip yet). Observable:
  the oracle test is GREEN and the tie-aware acceptance compiles + is reachable.

### Sampling Rate
- **Per task commit:** `cargo test -p lgbm-treelearner` + `cargo test -p oracle-harness --test learner_parity`
- **Per wave merge:** `cargo test --workspace`
- **Phase gate:** full `cargo test --workspace` green AND byte-unchanged with `LGBM_CUDA_ON_DEVICE` unset, before `/gsd-verify-work`.

### Wave 0 Gaps
- [ ] New oracle test in `crates/oracle-harness/tests/learner_parity.rs` — exercises the seam via
      host-fallback (D-01) and the tie-aware comparator (covers ODL-02 SC#3).
- [ ] New SC#2 test proving the fork is reachable + returns the host tree when forced eligible.
- [ ] Tie-aware comparator: generalize `assert_gpu_tree_matches_cpu_anchor` (decode `default_left`
      bit1, lift `kernel_parity.rs:1597-1620` to per-node) — covers ODL-02.
- Framework install: none (built-in harness present).

## Security Domain

> security_enforcement not explicitly false in config → treated as enabled. This is additive wiring
> with no new network/auth/crypto/persistence surface.

### Applicable ASVS Categories
| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | no | — (library code, no auth) |
| V3 Session Management | no | — |
| V4 Access Control | no | — |
| V5 Input Validation | yes (minimal) | New input = `LGBM_CUDA_ON_DEVICE` env var, parsed as a strict bool toggle (`Ok("1")`); no path/format/injection surface (same posture as `autotune_enabled`, T-13-01 accept). Existing V5 boundary checks in `train_inner` (:729) are unchanged. |
| V6 Cryptography | no | — (no crypto; never hand-roll, N/A here) |

### Known Threat Patterns for this stack
| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| Env-var toggle abuse (forcing on-device path) | Tampering | Toggle only selects between two correctness-gated paths; in Slice 0 the seam is a no-op (`Ok(None)`) so the worst case is "falls through to host" — no unsafe state. Behavior gated AND'd with the backend discriminator. |
| Silent behavior change leaking into CPU/ROCm | Tampering/Repudiation | Default-false discriminator + env-off default + full byte-unchanged merge gate (SC#1) is the enforcement. |

## Sources

### Primary (HIGH confidence — read this session)
- `crates/lgbm-compute/src/lib.rs` — `trait Backend` (:495), `resident_pool_supported` (:935),
  `prefers_host_partition` (:898,:1352,:2134), `CpuBackend` (:1239,:1242), `GpuBackend<R>` (:2037,:2126),
  type aliases (:2110,:2116,:2123).
- `crates/lgbm-treelearner/src/learner.rs` — `SerialTreeLearner` struct (:203), fields
  `resident_eligible`(:286)/`fused_eligible`(:294), `new` (:430), `train_inner` (:658, 4-tuple return
  :664), resident gate (:680).
- `crates/lgbm-treelearner/src/resident_pool.rs` — `resident_eligible` (:99), `LGBM_RESIDENT_FORCE`
  idiom (:141).
- `crates/oracle-harness/tests/learner_parity.rs` — `ROCM_LEAF_VALUE_TOL` (:2041),
  `assert_gpu_tree_matches_cpu_anchor` (:2046), `cpu_anchor_tree` (:2083), reference test
  `learner_parity_resident_equals_host_tree_on_hip` (:2099); 63 `#[test]`/parity fns.
- `crates/oracle-harness/tests/kernel_parity.rs` — f32-vs-f64 near-tie `default_left` acceptance (:1597-1620).
- `crates/oracle-harness/tests/raw_bin_train_parity.rs` — `raw_bin_train_matches_cpp_golden` (:119).
- `crates/lgbm-model/src/tree.rs` — `Tree` (:72), `decision_type` packing + `DEFAULT_LEFT_MASK=2`
  (:24-26,:53-55), `get_decision_type` (:135).
- `crates/lgbm-model/Cargo.toml`, `crates/lgbm-compute/Cargo.toml`, `crates/lgbm-treelearner/Cargo.toml`
  — the dependency graph proving the D-03 cycle.
- `crates/lgbm-compute/src/kernels/autotune.rs` — `autotune_enabled` env idiom.
- `.planning/config.json` — nyquist_validation true, commit_docs true.

### Secondary (MEDIUM confidence)
- `.planning/phases/14-scaffold-oracle-slice-0/14-CONTEXT.md` — D-01..D-05, canonical refs.
- `.planning/REQUIREMENTS.md` — ODL-01, ODL-02, Out-of-Scope table.

### Tertiary (LOW confidence)
- MEMORY notes (def-f8u-01, spoofed-APU) — engineering context, not re-verified this session.

## Metadata

**Confidence breakdown:**
- Standard stack / idioms: HIGH — every file:line in CONTEXT was read and confirmed.
- D-03 cycle finding: HIGH — proven from the three Cargo.toml dependency edges.
- D-03 *resolution* (Option A): MEDIUM/ASSUMED — recommendation; planner must choose.
- Oracle tie-aware extension: HIGH on the existing assets, MEDIUM on the exact generalization shape.
- Slice-0 discriminator activation (Open Q1): MEDIUM — design choice, both options byte-safe.

**Research date:** 2026-06-28
**Valid until:** ~2026-07-28 (stable internal code; re-verify file:line if learner.rs/lib.rs churn).
