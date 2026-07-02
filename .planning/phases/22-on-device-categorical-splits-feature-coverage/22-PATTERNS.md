# Phase 22: On-Device Categorical Splits (Feature Coverage) - Pattern Map

**Mapped:** 2026-07-02
**Files analyzed:** 6 (1 new, 5 modified) + 4 reuse-only wire targets
**Analogs found:** 6 / 6

## Orientation

Per RESEARCH, ~70% of this phase is **wiring existing, anchor-tested kernels** (§9 partition,
§10 `SplitCategorical`, predict-through) into the grow driver. Only two pieces are genuinely
new code: the **§8.1 categorical evaluator** and the **§6.3 bitset construction** — and both are
**byte-for-byte transcriptions** of already-golden host code in
`crates/lgbm-treelearner/src/feature_histogram_categorical.rs` (the crate-cycle forbids importing
it; it must be re-typed into `lgbm-compute`). The dominant risk is transcription fidelity
(kEpsilon bump site, l2 asymmetry, sort tie order), not novel design.

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|-------------------|------|-----------|----------------|---------------|
| `crates/lgbm-compute/src/kernels/categorical_split.rs` (NEW) | compute-kernel / transform | transform (histogram → SplitInfo + bitset) | `feature_histogram_categorical.rs` (transcription source) + `best_split.rs::find_best_splits_stage1_on` (compute-kernel host shape) | exact (transcription) |
| `crates/lgbm-compute/src/kernels/best_split.rs` (FILL) | compute-kernel / dispatch | transform | its own numeric `find_best_splits_stage1_on` eval (:632) | exact (same file, adjacent branch) |
| `crates/lgbm-compute/src/kernels/grow_driver.rs` (WIRE+EXTEND) | driver / orchestration | event-driven (per-leaf best-first) | its own numeric split path (`scan_leaf` :361, driver body :610-646) | exact (same file, numeric twin) |
| `crates/lgbm-compute/src/kernels/split_info.rs` (CHANGE D-03) | model / SoA store | CRUD (slot read/write) | its own `MAX_CAT_PER_SPLIT` const + `set_cat_thresholds`/`copy_slot` slab indexing | exact (same file) |
| `crates/lgbm-treelearner/src/learner.rs` (CHANGE D-06) | config gate / eligibility | request-response (predicate) | its own `on_device_eligible` init (:498) | exact (same file) |
| `crates/oracle-harness/tests/learner_parity.rs` (EXTEND) | test / integration | request-response (grow+assert) | `learner_parity_on_device_structure_gate` (:2329) + `run_categorical_cell` (:1486) | exact (same file, two twins to merge) |

### Reuse-only (WIRE, do not re-author — RESEARCH "Don't Hand-Roll")

| Existing entrypoint | File:line | Role |
|---------------------|-----------|------|
| `partition_categorical_on_device` | `data_partition.rs:686` | §9 categorical membership mark→scan→scatter |
| `route_to_left_categorical` / `find_in_bitset` | `data_partition.rs:153` / `:138` | §9 branchless routing + `Common::FindInBitset` |
| `split_categorical_on_device` / `split_categorical_kernel` | `tree.rs:765` / `:278` | §10 `kCategoricalMask` node write + `cat_boundaries[_inner]` |
| predict categorical branch | `predict.rs:115-122` | §10 predict-through bitset |
| `bitonic_argsort_on` | `primitives.rs:969` | many-vs-many ctr sort (index-only, single-block ≤1024) |

## Pattern Assignments

### `crates/lgbm-compute/src/kernels/categorical_split.rs` (NEW — transcribe §8.1 + §6.3)

**Analog / transcription source:** `crates/lgbm-treelearner/src/feature_histogram_categorical.rs`
(the f64 host anchor, already bit-exact to real 4.6). Cannot be imported (crate cycle,
memory `on-device-driver-crate-cycle-constraint`) — re-type it into `lgbm-compute`.

**Kernel-shape analog (host launcher convention to copy):** `best_split.rs:632-693`
`find_best_splits_stage1_on` — single-owner `CubeDim::new_1d(1)`, `client.create_from_slice`
for the histogram input, `client.empty` for the output slab, `launch_unchecked` in a confined
`unsafe` block. RESEARCH Pattern 1: cubecl-cpu has NO plane support, fixtures are tiny (≤7 bins),
so serial single-owner is both correct and the exact bit-shape of the host code.

**Signature to mirror (host anchor, feature_histogram_categorical.rs:93-101):**
```rust
pub fn find_best_threshold_categorical(
    hist: &[f64],
    cfg: &GainConfig,
    num_bin: i32,
    offset: i32,
    sum_gradient: f64,
    sum_hessian: f64,   // caller MUST pass this ALREADY +2*kEpsilon bumped (Pitfall 2)
    num_data: i32,
) -> CategoricalSplit
```

**One-hot vs many-vs-many l2 asymmetry (Pitfall 3, lines 126-130, 139, 212, 223) — copy verbatim:**
```rust
let use_onehot = num_bin <= cfg.max_cat_to_onehot;
// one-hot branch: uses cfg.lambda_l2 (ORIGINAL l2)
if use_onehot { let l2 = cfg.lambda_l2; /* scan singletons ... */ }
// many-vs-many branch: increment AFTER the one-hot return
let mut l2 = cfg.lambda_l2;
l2 += cfg.cat_l2;   // only here — never on the one-hot path
```

**ctr sort (Pitfall 1, lines 217-236) — reuse `bitonic_argsort_on`, NOT `sort_by`:**
```rust
// host anchor (f64 stable sort):
for i in bin_start..bin_end {
    if f64::from(round_int(get_hess(i) * cnt_factor)) >= cfg.cat_smooth { sorted_idx.push(i); }
}
let cat_smooth = cfg.cat_smooth;
let ctr = |t: i32| -> f64 { get_grad(t) / (get_hess(t) + cat_smooth) };
sorted_idx.sort_by(|&a,&b| ctr(a).partial_cmp(&ctr(b)).unwrap_or(Equal)); // std::stable_sort
let max_num_cat = cfg.max_cat_threshold.min((used_bin + 1) / 2);           // D-03 clamp
```
On-device: build f32 `ctr_keys`, call `primitives::bitonic_argsort_on(client, &ctr_keys, true)`
(returns `(Vec<i32> perm, Vec<f32> keys)`). **Tie hazard (Pitfall 1 / A1):** f32-bitonic tie
order ≠ f64 stable-sort tie order → different `cat_threshold` set → golden mismatch. Verify the
`cat_manyvsmany` grads (`0,-1,-2,-8,-9,-10`, `cat_smooth:0.0`, uniform hess) give strictly
distinct ctr (no ties) before locking.

**Winner-bitset build (`finalize`, lines 383-398) — copy the offset math:**
```rust
let cat_threshold: Vec<u32> = if use_onehot {
    vec![(best_threshold + offset) as u32]
} else {
    let num_cat_threshold = best_threshold + 1;
    // walk sorted_idx in best_dir, each + offset
};
```

**§6.3 bitset construction — transcribe `construct_bitset` (lines 424-435) verbatim:**
```rust
pub fn construct_bitset(vals: &[u32]) -> Vec<u32> {
    if vals.is_empty() { return Vec::new(); }
    let max_val = *vals.iter().max().unwrap();
    let n_blocks = (max_val / 32 + 1) as usize;      // shuffle-max length
    let mut bits = vec![0u32; n_blocks];
    for &v in vals { bits[(v / 32) as usize] |= 1u32 << (v % 32); }  // set 1<<(val%32)
    bits
}
```
Anti-pattern (RESEARCH): NO `Atomic<i64>` (broken on this cubecl); for tiny fixtures a
single-owner sequential OR into a pre-zeroed slab is bit-identical to this host code.

**Two-bitset producer (SetRealThreshold, learner.rs:3648-3659) — Pattern 3:** map winning inner
bins → real category values via `bin_to_category`, then `construct_bitset` the REAL values; keep
BOTH the real bitset and the inner-bin bitset consistent (see Open Q1 for routing convention).

---

### `crates/lgbm-compute/src/kernels/best_split.rs` (FILL the §8.1 dispatch seam)

**Analog:** the numeric eval in the same function, `find_best_splits_stage1_on` (:632).

**The sentinel to replace (best_split.rs:640-643):**
```rust
// Phase-22 categorical dispatch seam (D-04): the numerical core is continuous-only.
if task.is_categorical {
    return Ok(SplitScalars::default());   // ← is_valid=false sentinel; §8.1 eval replaces this
}
```
The `SplitFindTask` already carries `is_categorical` / `is_one_hot` (:98-102) and the task-gen
already sets `is_one_hot = (num_bin as i32) <= max_cat_to_onehot` (:239-243). Replace the early
return with a call into the new `categorical_split` evaluator, mapping its `CategoricalSplit`
into `SplitScalars` (mirror the numeric mapping already done downstream in this function).

---

### `crates/lgbm-compute/src/kernels/grow_driver.rs` (WIRE the branch + EXTEND `GrowFeature`)

**Analog:** the numeric split path in the same driver — `scan_leaf` (:361) and the driver body
`split_on_device` call (:610-646, note `num_cat_threshold: 0` at :637).

**`GrowFeature` extension (Pitfall 6, struct at :50, doc :45-48 explicitly omits `bin_to_category`):**
Add, as **native primitives only** (crate-cycle-safe, A3):
```rust
pub bin_to_category: Vec<i32>,   // SetRealThreshold needs it (§6.3)
pub cat_smooth: f64,
pub cat_l2: f64,
pub max_cat_threshold: i32,
pub max_cat_to_onehot: i32,
pub min_data_per_group: i32,
```

**The bail-out to replace (scan_leaf, grow_driver.rs:384-387):**
```rust
if na_as_missing || f.bin_type == BinType::Categorical {
    // Deferred (proving slice is numeric + non-NA); the finder rejects NA.
    continue;   // ← §8.1 eval + branch replaces the categorical half of this
}
```
Split the condition: keep `na_as_missing` deferred, but route `BinType::Categorical` into the
new evaluator instead of `continue`.

**Pitfall 2 (kEpsilon bump) — the driver must bump BEFORE calling the categorical evaluator:**
`scan_leaf` passes RAW `sum_h` to the numeric finder (which bumps internally); the categorical
finder expects `sum_hessian` **already +2*kEpsilon** (host does this at learner.rs:2760). Bump
`sum_h` in the categorical branch before the call.

**Driver-body SplitScalars build (analog, :618-638) — categorical variant sets `num_cat_threshold`:**
```rust
let scalars = SplitScalars { is_valid: true, leaf_index: best_leaf, /* ... */
    num_cat_threshold: 0 };   // ← numeric uses 0; categorical branch sets the real count
```
Then call the EXISTING `tree.split_categorical_on_device(...)` (tree.rs:765) and
`partition_categorical_on_device(...)` (data_partition.rs:686) instead of the numeric
`split_on_device` / `partition_leaf_stable`.

---

### `crates/lgbm-compute/src/kernels/split_info.rs` (D-03 runtime slab width)

**Analog:** the existing `MAX_CAT_PER_SPLIT` const + its three consuming sites in the same file.

**The const to demote to a default (split_info.rs:55-65):**
```rust
/// ... Phase-22-tunable ... SoA layout invariant to the cap — only the slab length changes.
pub const MAX_CAT_PER_SPLIT: usize = 32;   // ← becomes the DEFAULT, add a runtime cat_width field
```

**Sites to thread `cat_width` through (replace `MAX_CAT_PER_SPLIT` indexing):**
- `new` slab sizing + overflow check (:277 `checked_mul(MAX_CAT_PER_SPLIT)`) — read
  `config.max_cat_threshold` once here.
- `set_cat_thresholds` guard (:464 `thresholds.len() > MAX_CAT_PER_SPLIT`) — compare to runtime width.
- `set_cat_thresholds` base (:473 `slot * MAX_CAT_PER_SPLIT`), `cat_threshold`/`cat_threshold_real`
  accessors (:493, :509), `copy_slot` slab window (:551-553 `let w = MAX_CAT_PER_SPLIT`).

Keep the `checked_mul` overflow guard (V5 threat T-14-04-03). RESEARCH Open Q2: both goldens use
`max_cat_threshold: 32` (= default), so a dedicated width>32 **unit test** is required to prove
D-03 non-vacuously (assert slab length + a `set_cat_thresholds` of length 33 succeeds where the
current :464 guard rejects it).

---

### `crates/lgbm-treelearner/src/learner.rs` (D-06 host-fallback gate)

**Analog:** the `on_device_eligible` init in the same file (:498).

**The AND-gate to extend (learner.rs:498):**
```rust
on_device_eligible: backend.on_device_growth_supported() && cuda_on_device_env(),
// ← AND-in: && !(has_categorical_feature && config.use_quantized_grad)
```
RESEARCH A4: this check MUST live in the learner (where `use_quantized_grad` + `bin_type` are
visible); `on_device_growth_supported()` takes no config args. Add a one-line log on fallback,
mirroring the reference's `asm("trap;")` non-support (Phase-19 precedent). Add a learner-crate
unit test asserting `on_device_eligible == false` for the categorical+quantized combo.

---

### `crates/oracle-harness/tests/learner_parity.rs` (EXTEND — merge two existing twins)

This phase **merges two existing harnesses** rather than building new ones:
1. `learner_parity_on_device_structure_gate` (:2329) — the D-01#2 device structure gate (currently
   numeric-only via `on_device_proving_corpus` + `grow_features_of`).
2. `run_categorical_cell` (:1486) + `cat_corpus` (:1454) — the D-01#1 real-4.6-golden HOST cell.

**`grow_features_of` extension (:1982) — add the new `GrowFeature` cat fields:**
```rust
fn grow_features_of(features: &[FeatureColumn]) -> Vec<lgbm_compute::GrowFeature> {
    features.iter().map(|f| lgbm_compute::GrowFeature {
        bins: f.bins.clone(), num_bin: f.num_bin, offset: f.offset,
        /* ... existing numeric fields ... */
        bin_type: f.bin_type,
        // ADD: bin_to_category: f.bin_to_category.clone(), cat_smooth, cat_l2,
        //      max_cat_threshold, max_cat_to_onehot, min_data_per_group (from cfg)
    }).collect()
}
```
The `FeatureColumn` already carries `bin_to_category` (cat_corpus builds it, :1470).

**Structure-gate cell pattern to replicate (:2341-2349) — drive the driver, assert vs cpu anchor:**
```rust
let (driver_tree, layout) = grow_driver::grow_tree_on_device_driver_with_cfg(
    &client, &g, &h, &grow_features, num_leaves, max_depth, cfg)?;
// CpuBackend runs f64-vs-f64 → default_left bit-exact → STRICT comparator (not tie-aware):
assert_gpu_tree_matches_cpu_anchor(&driver_tree, &anchor, "on-device");
```
Add categorical corpus cases feeding `cat_corpus`-style features through this DEVICE path.

**Real-golden bit-exact asserts to reuse (:1500-1518) — now against the DEVICE tree:**
```rust
assert_eq!(tree.num_cat, golden.num_cat, ...);                    // num_cat
assert_eq!(dt & 1, golden.decision_type[i] & 1, ...);             // kCategoricalMask bit
assert_eq!(tree.cat_boundaries, golden.cat_boundaries, ...);      // cat_boundaries
assert_eq!(tree.cat_threshold, golden.cat_threshold, ...);        // REAL category bitset
```

RESEARCH Open Q1: before the full structure gate, assert the on-device partition counts equal the
host `partition_categorical_stable` (data_partition.rs:525) for both fixtures to isolate the
real-value-vs-inner-bin routing convention (Pattern 3 / Pitfall 4).

## Shared Patterns

### Single-owner f64 anchor evaluator (def-f8u-01 discipline)
**Source:** `best_split.rs:632-693` (numeric) + `primitives.rs:51` (no-plane note).
**Apply to:** `categorical_split.rs` (both eval + bitset construction).
```rust
CubeDim::new_1d(1)                 // single owner; cubecl-cpu has no planes
let h_hist = client.create_from_slice(f64::as_bytes(hist));
let h_out  = client.empty(OUT_LEN * core::mem::size_of::<f64>());
unsafe { kernel::launch(client, CubeCount::Static(1,1,1), CubeDim::new_1d(1), /* args */); }
```
Never introduce a second GPU f32 categorical path to compare against — anchor to the f64 fold.

### Pre-allocated slab, zero per-split alloc (ODL-02)
**Source:** `split_info.rs:254-293` (`new` is the ONLY `client.empty` caller).
**Apply to:** the bitset representation and the D-03 width change. Anti-pattern: the C++
`AllocateCatVectorsKernel` per-`SplitInfo` `cudaMalloc`.

### Crate-cycle-safe additive metadata
**Source:** `GrowFeature` doc (grow_driver.rs:45-48) — native `Vec`/enum/scalar only, never a
`lgbm-treelearner` type.
**Apply to:** the six new `GrowFeature` cat fields and the transcribed evaluator. `cargo build -p
lgbm-compute` breaks if any treelearner/dataset type sneaks in (A3).

### Branchless bitset membership (`Common::FindInBitset`)
**Source:** `data_partition.rs:138` `find_in_bitset` — shared by predict + partition; keep the
`pos/32 >= n → 0` bound clamp (V5 threat, bitset OOB).

## No Analog Found

None — every seam has an in-repo analog (this is a pure in-repo port; RESEARCH confidence HIGH).
The only "new" files are transcriptions of existing golden host code.

## Metadata

**Analog search scope:** `crates/lgbm-compute/src/kernels/`, `crates/lgbm-treelearner/src/`,
`crates/oracle-harness/tests/`, `crates/lgbm-core/src/config/`.
**Files scanned:** 8 (best_split, grow_driver, split_info, feature_histogram_categorical, learner,
learner_parity, tree, data_partition, primitives — read via targeted non-overlapping ranges).
**Pattern extraction date:** 2026-07-02
