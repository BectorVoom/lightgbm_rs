---
phase: 22
phase_name: "on-device-categorical-splits-feature-coverage"
project: "LightGBM-rs — Pure Rust LightGBM with CubeCL"
generated: "2026-07-02"
counts:
  decisions: 8
  lessons: 4
  patterns: 8
  surprises: 4
missing_artifacts:
  - "22-UAT.md"
---

# Phase 22 Learnings: on-device-categorical-splits-feature-coverage

Phase goal (ODL-22): categorical splits work end-to-end on the proven on-device numerical
driver — §6.3 bitset construction, §8.1 split evaluation, §9 partition membership, and §10
SplitCategorical — via a pre-allocated bitset representation (no per-`SplitInfo` device alloc).
Verified PASS (4/4 ROADMAP success criteria); categorical is now the FIRST on-device subsystem
pinned to a REAL `lib_lightgbm` 4.6 reference rather than a host re-transcription.

## Decisions

### Runtime `cat_width` slab, `MAX_CAT_PER_SPLIT` demoted to a default (D-03)
`DeviceSplitInfo::new` gained a `cat_width: usize` field + `max_cat_threshold` param read once
at construction; the old compile-time constant `MAX_CAT_PER_SPLIT` is retained as `pub const usize = 32`
but is now the DEFAULT, not a hard cap. No clamp-to-32, no silent truncation.

**Rationale:** the config-tunable device cap must honor `config.max_cat_threshold` at runtime while
preserving the D-05 allocate-once (not-per-train) property; the SoA layout is invariant to width, only
the slab length changes. `checked_mul` overflow guard kept.
**Source:** 22-01-SUMMARY.md

### D-06 host-fallback gate lives in the learner, not lgbm-compute
`on_device_eligible_gate(base, has_categorical_feature, use_quantized_grad) = base && !(cat && quantized)`
sits in `lgbm-treelearner/src/learner.rs`, wired via `refresh_on_device_eligibility()` called from
`with_features`/`with_quantized_grad`.

**Rationale:** the gate needs `use_quantized_grad` + per-feature `bin_type` visibility, which only the
learner has; `lgbm-compute`'s `on_device_growth_supported` takes no config args (RESEARCH A4). Mirrors the
CUDA reference `asm("trap;")` non-support for categorical+quantized. Computed at setup-time (compute-once),
preserving the D-05 not-per-train property.
**Source:** 22-01-SUMMARY.md

### `GrowFeature` extended with native primitives only (crate-cycle-safe)
Six additive fields (`bin_to_category: Vec<i32>`, `cat_smooth: f64`, `cat_l2: f64`, `max_cat_threshold: i32`,
`max_cat_to_onehot: i32`, `min_data_per_group: i32`) — never a `lgbm-treelearner`/`lgbm-dataset` type.

**Rationale:** `grow_tree_on_device` lives in `lgbm-compute` (below `lgbm-treelearner`); importing a learner
type would create a crate cycle. `cargo build -p lgbm-compute` green proves no cycle (T-22-04 mitigated).
**Source:** 22-02-SUMMARY.md

### Categorical evaluator is the single-owner `CubeDim::new_1d(1)` f64 anchor (def-f8u-01)
`find_best_threshold_categorical` runs single-owner f64, never a second nondeterministic GPU f32 path.

**Rationale:** parity contract — the on-device categorical result must be pinned to the deterministic f64
anchor, never compared GPU-vs-GPU at 1e-6 (def-f8u-01). The f32 mirrors in `best_split.rs` widen to f64 for
the categorical branch.
**Source:** 22-03-SUMMARY.md, 22-04-SUMMARY.md

### Dual-anchor discipline for the fidelity gate
The DEVICE-grown categorical tree is pinned to BOTH the cpu f64 fold (structure, tie-aware `default_left`
comparator) AND the real `lib_lightgbm` 4.6 golden (fidelity: `num_cat`, `kCategoricalMask`, `cat_boundaries`,
real `cat_threshold` bitset).

**Rationale:** structure-bit-exact-to-f64 proves the algorithm; golden-bit-exact proves faithfulness to the
REAL reference — categorical becomes the first on-device subsystem anchored to a real lib, not a host
re-transcription.
**Source:** 22-05-SUMMARY.md

### New fidelity cells env-gated behind `LGBM_CUDA_ON_DEVICE`
All new device parity cells skip-pass when the env var is unset.

**Rationale:** keeps the default env-unset cubecl-cpu f64 merge gate byte-green (SC #4); `cargo test --workspace`
stayed at 963/0 with the numeric spine byte-unchanged.
**Source:** 22-05-SUMMARY.md

### Predict-through asserts `predict_leaf_index`, not raw leaf values
SC #3 compares per-training-row `predict_leaf_index` (device tree vs its model-text reparse vs golden).

**Rationale:** device leaf values are the RAW Newton output while the golden carries the shrinkage'd leaf, so
raw-value compare would diverge; leaf-index routing is the correct invariant and it exercises the real
`find_in_bitset` categorical routing path.
**Source:** 22-05-SUMMARY.md

### `best_split.rs` stage-1 seam uses `GainConfig` defaults for cat knobs
The CUDA-mirror stage-1 launchers build their `GainConfig` from numeric `Stage1Scalars` + defaults for the
five categorical knobs, rather than widening `Stage1Scalars`.

**Rationale:** `Stage1Scalars`/`SplitFindTask` carry no per-feature categorical config and `GrowFeature` isn't
visible there; widening would force edits to test files outside the plan's `files_modified` contract. The live
`scan_leaf` grow path threads REAL per-feature config via `GrowFeature`, so the mirror seam's defaults never
touch the live merge-gate path.
**Source:** 22-04-SUMMARY.md

---

## Lessons

### Signature/struct changes break hidden construction & call sites that name-scoped caller scans miss
The D-03 `DeviceSplitInfo::new` param addition broke 9 call sites in `crates/lgbm-compute/tests/split_info.rs`
(the initial caller scan excluded the filename, hiding the integration-test file). Adding six non-`Default`
fields to `GrowFeature` broke a third construction site at `learner.rs:794` (the on-device fork), outside the
plan's declared `files_modified`.

**Context:** two separate Rule-3 blocking deviations across 22-01 and 22-02, both surfaced only by
`cargo test --workspace`. Scan for ALL callers/constructors (including test files and cfg-gated forks), not
just the ones sharing a directory or matching the primary source filename.
**Source:** 22-01-SUMMARY.md, 22-02-SUMMARY.md

### Unit tests on the top-level histogram miss NaN-degenerate child-leaf paths
22-03/22-04 unit tests only called the evaluator on the finite, tie-free TOP-LEVEL histogram. The full device
grow loop (`num_leaves=4`) reaches deeper child leaves where a zero-hessian categorical bin yields
`ctr = grad/(hess+cat_smooth) = 0/0 = NaN` — and only there did the f32 `bitonic_argsort_on` (a) diverge from
the f64-stable golden order and (b) leak power-of-two padding indices (6,7 for a 6→8 padded input) into the
truncated permutation, panicking OOB.

**Context:** the 22-05 fidelity gate exists precisely to drive the FULL grow loop and catch exactly this kind
of prior-wave defect. Exercise the end-to-end grow loop, not just the isolated kernel entry point.
**Source:** 22-05-SUMMARY.md

### A single-owner f64 anchor must not substitute an "optimized" f32 sort
The categorical evaluator IS the def-f8u-01 single-owner f64 anchor, so its ctr sort order MUST equal the
host's f64 `std::stable_sort` bit-exact — including NaN-Equal stable behavior on degenerate leaves. The fix
transcribed the host verbatim (`sorted_idx.sort_by(|&a,&b| ctr(a).partial_cmp(&ctr(b)).unwrap_or(Equal))`);
`client` became unused but was kept in the signature for API symmetry.

**Context:** "index-only bitonic argsort reuse" looked like a clean optimization in 22-03 but was invalid for
an anchor whose entire job is to match the reference bit-exact.
**Source:** 22-05-SUMMARY.md

### Bump guards need a representable magnitude to discriminate
At fixture leaf hessian sums (40/60), `2*kEpsilon` is below the f64 ULP and absorbed, so a bit-exact
`bump(raw) == raw + 2*kEpsilon` assertion holds trivially by rounding on both sides. The discriminating
double/missed-bump guard was therefore placed at a representable magnitude (`1e-13`).

**Context:** faithful to the host (its bump is likewise absorbed at those sums) while still catching an
accidental `+4*kEpsilon` or an omitted bump.
**Source:** 22-04-SUMMARY.md

---

## Patterns

### Pre-allocated slab, runtime width (Pattern 2 / D-03)
A compile-time cap becomes a runtime field read once at construction; the SoA layout is invariant to width,
only the slab length changes.

**When to use:** any config-tunable device cap that must stay allocate-once (not per-train).
**Source:** 22-01-SUMMARY.md

### Config-boundary host-fallback gate in the learner
The gate deciding host-vs-device lives where `use_quantized_grad` + per-feature `bin_type` are visible (the
learner), tested deterministically across the truth table with no env manipulation.

**When to use:** any device-eligibility decision that depends on config the compute crate cannot see.
**Source:** 22-01-SUMMARY.md

### Additive struct extension across a crate-cycle seam using native primitives only
Extend a shared struct that lives below the type-owning crate with `Vec<i32>`/`f64`/`i32` fields — never the
upstream domain type — to carry metadata across the seam.

**When to use:** passing metadata down to a lower crate (`lgbm-compute`) without importing an upper crate's
types.
**Source:** 22-02-SUMMARY.md

### Config-driven field population via `&GainConfig` with `default()` fallback
Thread `cfg: &GainConfig` through the populating helper and pass `GainConfig::default()` at call sites whose
corpus builder returns no cfg.

**When to use:** faithfully satisfying a "populate from cfg" must-have while keeping cfg-less call sites
compiling.
**Source:** 22-02-SUMMARY.md

### Branched driver body: categorical vs numeric differ ONLY in partition + mutation
The child-seed / histogram-subtract / scan phases are shared; only the partition and tree-mutation arms fork
on `BinType::Categorical`.

**When to use:** adding a second split type to an existing grow driver without duplicating the shared spine.
**Source:** 22-04-SUMMARY.md

### Bitsets DERIVED from the pre-allocated slab (never per-`SplitInfo` alloc)
Stage winners into the pre-allocated `DeviceSplitInfo` cat slab (`set_cat_thresholds`), then materialize both
the real category-value bitset and the inner-bin routing bitset FROM the slab (`set_real_threshold`).

**When to use:** on-device split representations where the ODL-02 allocate-once invariant forbids per-split
device allocation.
**Source:** 22-04-SUMMARY.md

### Merge two existing harness twins rather than authoring a new harness
The fidelity gate extended the existing `learner_parity_on_device_structure_gate` (cpu-f64 twin) and reused
the real-golden host-cell twin, adding cat cases to both rather than writing a fresh test harness.

**When to use:** a new fidelity gate whose two anchors already have established harness idioms.
**Source:** 22-05-SUMMARY.md

### On-device fidelity gate idiom
Drive `grow_tree_on_device_driver_with_cfg` on the `cat_corpus` fixtures, assert vs the cpu f64 anchor + the
real golden, gate behind `LGBM_CUDA_ON_DEVICE`.

**When to use:** adding a real-reference fidelity gate for any new on-device subsystem.
**Source:** 22-05-SUMMARY.md

---

## Surprises

### The `cat_onehot` fixture actually flows through the many-vs-many path
`cat_onehot` has `num_bin=5 > max_cat_to_onehot=4`, so it routes through the many-vs-many code (selecting a
single category), reproducing the golden. The fixture name reflects the one-category RESULT, not the code
path; the true one-hot branch is covered by a synthetic `num_bin=4` test.

**Impact:** clarified test coverage — no pinned golden changed, but the naming could mislead about which
branch a fixture exercises. Documented as a clarification (not a deviation).
**Source:** 22-03-SUMMARY.md

### The many-vs-many linchpin crashed on its first full-grow-loop exercise
The plan's declared linchpin path panicked OOB (`index out of bounds: the len is 6 but the index is 6`) and
diverged from the golden the moment the full device grow loop ran — a real prior-wave (22-03) defect surfaced
only by the 22-05 gate. Because `LGBM_CUDA_ON_DEVICE=1` routes `SerialTreeLearner` through the device path,
both the on-device AND the host `learner_parity_categorical_manyvsmany` cell hit the panic under env=1.

**Impact:** the test-only plan (22-05) expanded by one prerequisite SOURCE fix in `categorical_split.rs`
(commit `0566370`) — the fidelity gate did its job of catching a defect it exists to catch. Verifier logged an
`override_suggestion` to record the intentional bitonic→f64-stable-sort supersession of the 22-03 plan truth.
**Source:** 22-05-SUMMARY.md, 22-VERIFICATION.md

### Many unrelated numeric cells fail under `LGBM_CUDA_ON_DEVICE=1`
Monotone, extra_trees, col_sampler, and growth_path_subtract cells fail under env=1 because the on-device
driver does not yet implement those numeric features (e.g. monotone splits to 4 leaves where the golden is 1).
Verified pre-existing by restoring both files to `HEAD~2` and re-running under env=1 (identical failure).

**Impact:** none on this phase — the documented D-04 posture is that env=1 is best-effort on-device smoke and
the env-unset cpu-f64 lane is the hard merge gate. Not a regression. Follow-up: broaden on-device numeric
feature coverage (Phase 23).
**Source:** 22-05-SUMMARY.md, 22-VERIFICATION.md

### GBDT does not yet call `with_quantized_grad` — D-06 gate defaults false in production
The D-06 gate + builder + test all exist, but `gbdt.rs` doesn't yet call `.with_quantized_grad(...)`, so
`use_quantized_grad` defaults false in production.

**Impact:** benign — the on-device path is env-gated OFF in production (`grow_tree_on_device` returns `Ok(None)`
in Slice 0), so a categorical+quantized on-device run cannot occur regardless. This is the Phase-23 default-on
rollout seam, documented (not a gap).
**Source:** 22-01-SUMMARY.md, 22-VERIFICATION.md
