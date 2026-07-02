# Phase 22: On-Device Categorical Splits (Feature Coverage) - Research

**Researched:** 2026-07-02
**Domain:** CubeCL GPU-kernel port of LightGBM categorical splits (bitset construction, categorical split eval, categorical partition membership, `SplitCategorical` tree mutation) onto the proven numerical on-device driver, with a hard f64-fold parity contract.
**Confidence:** HIGH (all claims are grounded in current-codebase file:line evidence or the committed `docs/cuda-kernel-design.md`; no external package research needed — this is a pure in-repo port)

<user_constraints>
## User Constraints (from 22-CONTEXT.md)

### Locked Decisions

**Carried forward from Phases 14–21 (LOCKED — do not re-litigate):**
- **Anchor = cubecl-cpu f64 fold.** STRUCTURE bit-exact + leaf values within ~1e-5 f32 envelope; **never** compare two GPU f32 paths to each other (def-f8u-01).
- **Additive + `LGBM_CUDA_ON_DEVICE`-gated.** Env-unset ⇒ CPU / ROCm / host-CUDA byte-unchanged; merge gate runs on the default cubecl-cpu lane so the categorical structure gate is non-vacuous without ROCm hardware.
- **Pre-allocated bitset slab, zero per-split device alloc** (ODL-02 / ODL-22). The C++ `AllocateCatVectorsKernel` per-`SplitInfo` `cudaMalloc` is exactly the anti-pattern CubeCL pre-allocation eliminates.
- **ROCm = best-effort smoke, not the gate** (D-04). A real-ROCm run, if attempted, is pinned to the cpu anchor and is informative, not blocking. Full real-hardware validation is Phase 23's Kaggle DoD.

- **D-01: Both real 4.6 goldens AND the cubecl-cpu f64 structure gate.** (1) Pin the constructed **bitset / decision-type bit / `num_cat` / chosen `cat_threshold` (REAL category bitset)** **bit-exact** to the real 4.6 goldens (`cat_onehot`, `cat_manyvsmany`). (2) Run the on-device categorical tree through the **cubecl-cpu f64 structure gate** (extend `learner_parity_on_device_structure_gate`), tie-aware on `default_left`.
- **D-02: Both one-hot AND many-vs-many this phase.** The many-vs-many path reuses the Phase-14 bitonic argsort primitive (index-only) for the `grad/(hess+cat_smooth)` bin sort; `cat_l2` added to `l2` in the gain math (§8.1).
- **D-03: Size the bitset/threshold slab from `config.max_cat_threshold` at driver/`DeviceSplitInfo` init.** Make `MAX_CAT_PER_SPLIT` (currently fixed const 32) a runtime value read from config once at init (default 32); no silent truncation, no per-split alloc. Explicitly rejected: hard-clamp-to-32.
- **D-06: Honest host-fallback when categorical features AND `use_quantized_grad` are both set.** `on_device_growth_supported()` (effectively the learner's eligibility gate) returns **false** for that combo (routes to host), with a one-line log. Mirrors the reference's own `asm("trap;")` non-support.

### Claude's Discretion
- Kernel geometry / thread-block mapping for the bitset-construction and categorical-eval kernels (follow §6.3 / §8.1 launch idioms; cubecl-0.10 gotcha checklist applies).
- Exact bitset-construction atomic mechanics (reference uses `atomicAdd_system(out + val/32, 1<<(val%32))` into a pre-zeroed `u32*`) — pick the CubeCL-safe equivalent that reproduces the same bits.
- Parity fixture parameters — smallest configs that provably exercise one-hot, many-vs-many (`num_bin > max_cat_to_onehot`), the `max_num_cat` clamp, and predict-through.
- Whether the many-vs-many bitonic sort runs single-block or global-memory strided — pick per the used-bin count of the fixtures.

### Deferred Ideas (OUT OF SCOPE)
- **Categorical + `use_quantized_grad` on-device** — host-fallback this phase (D-06); not a near-term follow-up.
- **Low-VRAM global-memory categorical eval** (`_GlobalMemory` variant, §8.1) — optional; only needed if fixtures have more bins than threads.
- **Perf-validation of the categorical path** (Kaggle A/B, device_launches) → Phase 23 DoD.
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| ODL-22 | On-device categorical splits end-to-end — bitset construction (§6.3), categorical split evaluation (one-hot + many-vs-many bitonic-sorted, §8.1), categorical partition membership (§9), and `SplitCategorical` tree mutation (§10) — anchor-pinned, via the pre-allocated bitset representation (no per-`SplitInfo` device alloc). | §9 (`partition_categorical_on_device`) and §10 (`split_categorical_on_device`) on-device entrypoints ALREADY EXIST and are anchor-tested but UNWIRED into the grow driver; §8.1 categorical eval and §6.3 on-device bitset construction are the true unbuilt gaps; D-03 slab-sizing + grow-driver wiring + on-device structure-gate cases + D-06 host-fallback complete the requirement. See "The Exact Seams to Fill" below. |
</phase_requirements>

## Summary

Phase 22 is **narrower than it first appears** because Phases 18/21 already landed and anchor-tested most of the *consuming* categorical infrastructure. The `§9` partition membership (`partition_categorical_on_device`, `route_to_left_categorical`, `find_in_bitset`) and the `§10` `SplitCategorical` tree mutation (`split_categorical_on_device` / `split_categorical_kernel`) and the categorical predict branch **already exist, compile, and pass unit tests** — they are simply **not wired into the on-device grow driver**. The genuinely unbuilt work is the **`§8.1` categorical split evaluator** (the driver and `best_split.rs` both bail out with an `is_valid=false` sentinel / `continue` on categorical features) and the **`§6.3` on-device bitset construction** (only a *host* `construct_bitset` exists, in a crate above `lgbm-compute`).

The dominant architectural constraint is the **crate cycle**: `grow_tree_on_device_driver` lives in `lgbm-compute`, *below* `lgbm-treelearner`, so the authoritative host categorical logic (`feature_histogram_categorical::find_best_threshold_categorical`, `construct_bitset`) **cannot be imported** — it must be faithfully **transcribed into `lgbm-compute`** as additive native bookkeeping, and `GrowFeature` must gain the categorical fields it currently deliberately omits (`bin_to_category` and the categorical config scalars). The parity contract is unchanged: anchor structure-bit-exact to the cubecl-cpu f64 fold and (the fidelity upgrade this phase) bit-exact to the **real `lib_lightgbm` 4.6 goldens** that already sit in the fixtures directory.

**Primary recommendation:** Transcribe `find_best_threshold_categorical` + `construct_bitset` into `lgbm-compute` (as a single-owner `CubeDim(1)` f64 evaluator reusing `primitives::bitonic_argsort_on` for the ctr sort, byte-for-byte from the host anchor), extend `GrowFeature` with categorical metadata, make `MAX_CAT_PER_SPLIT` a runtime slab width (D-03), wire a categorical branch into `grow_driver::scan_leaf` + the driver body (calling the *existing* `partition_categorical_on_device` and `split_categorical_on_device`), and extend `learner_parity_on_device_structure_gate` with the two real-golden corpus cases. Gate D-06 (categorical + quantized) in the learner's `on_device_eligible` predicate.

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Categorical split evaluation (§8.1) | `lgbm-compute` kernels (`best_split.rs` / new categorical eval) | — | Grow driver is in `lgbm-compute`; cannot reach up to `lgbm-treelearner`'s host evaluator (crate cycle). Must transcribe. |
| Bitset construction (§6.3) | `lgbm-compute` kernels (new) | — | Consumed by `split_categorical_on_device` (already in `lgbm-compute/tree.rs`); producer must live in the same crate. |
| Categorical partition membership (§9) | `lgbm-compute/data_partition.rs` (EXISTS) | — | `partition_categorical_on_device` + `route_to_left_categorical` already built and anchor-tested. Wire only. |
| `SplitCategorical` tree mutation (§10) | `lgbm-compute/tree.rs` (EXISTS) | — | `split_categorical_on_device` + `split_categorical_kernel` already built. Wire only. |
| Predict-through bitset | `lgbm-compute/predict.rs` (EXISTS) | — | Categorical membership branch already present (predict.rs:115-122). |
| D-03 slab sizing | `lgbm-compute/split_info.rs` | driver init | Runtime width read from `config.max_cat_threshold`. |
| D-06 host-fallback gate | `lgbm-treelearner/learner.rs` (`on_device_eligible`) | `lgbm-compute` discriminator | The config-aware "has categorical && quantized" check must live where the config is visible (the learner); `on_device_growth_supported()` takes no config args. |
| Parity gate (real goldens + structure) | `oracle-harness/tests/learner_parity.rs` | fixtures/categorical | Extend existing `learner_parity_on_device_structure_gate` + `run_categorical_cell`. |

## Standard Stack

No new external packages. This is a pure in-repo port. All work uses the already-vendored `cubecl` 0.10 (compute), `rayon` (CPU parallelism — not needed here), and the existing crate graph. **No Package Legitimacy Audit required** (no `npm`/`pip`/`cargo add`).

### Core (in-repo modules the plan composes)

| Module | Location | Purpose | Status for Phase 22 |
|--------|----------|---------|---------------------|
| `split_info.rs` | `crates/lgbm-compute/src/kernels/` | Pre-allocated SoA `DeviceSplitInfo` with reserved cat slabs, `MAX_CAT_PER_SPLIT` | **CHANGE (D-03):** runtime slab width |
| `best_split.rs` | `crates/lgbm-compute/src/kernels/` | Per-feature split eval; `is_categorical`/`is_one_hot` dispatch seam | **FILL (§8.1):** categorical eval currently returns `is_valid=false` sentinel |
| `grow_driver.rs` | `crates/lgbm-compute/src/kernels/` | Per-leaf best-first on-device grow loop; `GrowFeature` metadata | **WIRE + EXTEND:** categorical branch; add cat fields to `GrowFeature` |
| `data_partition.rs` | `crates/lgbm-compute/src/kernels/` | `partition_categorical_on_device`, `route_to_left_categorical`, `find_in_bitset` | **REUSE (EXISTS):** wire into driver |
| `tree.rs` | `crates/lgbm-compute/src/kernels/` | `split_categorical_on_device`, `split_categorical_kernel` | **REUSE (EXISTS):** wire into driver |
| `predict.rs` | `crates/lgbm-compute/src/kernels/` | Categorical membership predict branch | **REUSE (EXISTS):** verify predict-through |
| `primitives.rs` | `crates/lgbm-compute/src/kernels/` | `bitonic_argsort_on` (index-only, single-block) | **REUSE (EXISTS):** many-vs-many ctr sort |
| `feature_histogram_categorical.rs` | `crates/lgbm-treelearner/src/` | Host `find_best_threshold_categorical` + `construct_bitset` (the f64 anchor logic) | **TRANSCRIBE (crate-cycle):** cannot import; port into `lgbm-compute` |
| `learner.rs` | `crates/lgbm-treelearner/src/` | `on_device_eligible` gate; host categorical grow (reference for transcription) | **CHANGE (D-06):** AND-in "not(cat && quantized)" |
| `learner_parity.rs` | `crates/oracle-harness/tests/` | Real-golden `run_categorical_cell`; `learner_parity_on_device_structure_gate` | **EXTEND (D-01):** add on-device categorical cases |

**Version verification:** N/A — no external packages installed. cubecl 0.10 is already the workspace-pinned compute runtime (confirmed by `Atomic<u64>` + `SharedMemory` usage in `histogram.rs:1274,1282` and `sync_cube()` in `primitives.rs`).

## Architecture Patterns

### System Architecture Diagram (Phase-22 categorical data flow through the on-device driver)

```
                     grow_tree_on_device_driver_with_cfg  (grow_driver.rs:465)
                                    │  per-leaf best-first loop
                                    ▼
                          ┌──── scan_leaf (grow_driver.rs:361) ────┐
                          │  for each GrowFeature f:               │
             numeric ◄────┤  if f.bin_type == Categorical  ────────┼──► NEW §8.1 categorical eval
             (existing)   │     (currently `continue`, :384)       │    (transcribe find_best_threshold_categorical)
                          └─────────────────────────────────────────┘        │
                                    │ best (SplitInfo) + cat_threshold_bins    │ reuse primitives::bitonic_argsort_on
                                    ▼                                          │ (grad/(hess+cat_smooth) ctr sort)
                    ┌─── is best feature categorical? ───┐
          numeric   │                                    │  categorical (NEW branch in driver body)
          (existing)▼                                    ▼
      partition_leaf_stable                    ① NEW §6.3 bitset construction:
      + split_on_device                           SetRealThreshold (bin→category via bin_to_category)
      (num_cat_threshold:0)                        + bitset length (val/32+1) + construct (1<<(val%32))
                                                    → real bitset (category-value) + inner bitset (bin)
                                                 ② partition_categorical_on_device  ← EXISTS (data_partition.rs:686)
                                                    (route via route_to_left_categorical / find_in_bitset)
                                                 ③ split_categorical_on_device      ← EXISTS (tree.rs:765)
                                                    (kCategoricalMask, num_cat, cat_boundaries[_inner])
                                    │
                                    ▼
                       seed children + build/subtract histograms  (unchanged)
                                    │
                                    ▼
                    lgbm_model::Tree  ──►  predict.rs categorical branch  ← EXISTS (predict.rs:115-122)
```

### Recommended Project Structure (no new files strictly required; a new module keeps the eval isolated)

```
crates/lgbm-compute/src/kernels/
├── best_split.rs           # fill the is_categorical branch (or delegate to a helper)
├── categorical_split.rs    # RECOMMENDED NEW: transcribed find_best_threshold_categorical
│                           #   + construct_bitset + SetRealThreshold, single-owner f64
├── grow_driver.rs          # GrowFeature += cat fields; scan_leaf + driver-body cat branch
├── split_info.rs           # MAX_CAT_PER_SPLIT → runtime width (D-03)
├── data_partition.rs       # (reuse partition_categorical_on_device)
├── tree.rs                 # (reuse split_categorical_on_device)
└── predict.rs              # (reuse categorical branch)
```

### Pattern 1: Single-owner f64 anchor evaluator (the def-f8u-01 discipline)
**What:** The categorical eval and bitset construction run on `CubeDim::new_1d(1)` (single owner), matching the numeric `find_best_splits_stage1_on` (best_split.rs:669-673) and the bitonic argsort anchor (`CubeDim::new_1d(1)`, primitives.rs). cubecl-cpu has NO plane support (primitives.rs:51), and the fixtures are tiny (≤7 bins), so a serial single-owner scan is both correct and the exact bit-shape of the host `find_best_threshold_categorical`.
**When to use:** All Phase-22 new kernels. Never introduce a second GPU f32 categorical path to compare against — anchor to the f64 fold.
**Example (host anchor to transcribe, feature_histogram_categorical.rs:226-236):**
```rust
// Source: crates/lgbm-treelearner/src/feature_histogram_categorical.rs:225-236
let cat_smooth = cfg.cat_smooth;
let ctr = |t: i32| -> f64 { get_grad(t) / (get_hess(t) + cat_smooth) };
sorted_idx.sort_by(|&a, &b| ctr(a).partial_cmp(&ctr(b)).unwrap_or(Equal)); // std::stable_sort
let max_num_cat = cfg.max_cat_threshold.min((used_bin + 1) / 2);           // the D-03 clamp
```
Reuse `primitives::bitonic_argsort_on(client, &ctr_keys, /*ascending=*/true)` (primitives.rs:969) for the sort — it returns `(Vec<i32> indices, Vec<f32> keys)` and replicates `BitonicArgSort_1024`'s comparator/tie order exactly (primitives.rs:876). **Tie hazard:** the host uses f64 `partial_cmp` stable-sort; the bitonic primitive is f32-keyed. On tied ctr values the sort order can differ → different `cat_threshold` set → golden mismatch. See Pitfall 1.

### Pattern 2: Pre-allocated slab, runtime width (D-03)
**What:** `MAX_CAT_PER_SPLIT` (split_info.rs:65) is a `const usize = 32`. D-03 makes the slab *width* a runtime value read once from `config.max_cat_threshold` (default 32, config/mod.rs:366) at `DeviceSplitInfo::new`. The SoA layout is invariant to the width (module docs, split_info.rs:60-64) — only the slab length changes. Keep `MAX_CAT_PER_SPLIT` as the *default* constant; add a `cat_width: usize` field threaded through `new` and the `base = slot * cat_width` indexing (split_info.rs:473, 493, 509, 551-553).
**When to use:** Only if a fixture sets `max_cat_threshold > 32`. Both committed goldens use `max_cat_threshold: 32` (fixtures), so the *default* path is exercised by them; a dedicated `>32` fixture (or a unit test) is needed to prove D-03 non-vacuously.

### Pattern 3: Two bitset conventions (real category-value vs inner-bin)
**What:** `split_categorical_on_device` (tree.rs:765) consumes BOTH `bitset` (real category-value bitset, the serialized `cat_threshold_`) AND `bitset_inner` (inner-bin bitset, the routing key). The host learner routes partition by the REAL category bitset because "routing by category value is equivalent to the C++ inner-bin bitset routing" (learner.rs:3641-3647), whereas the reference §9 (`route_to_left_categorical`, data_partition.rs:153) routes by `FindInBitset(bitset, bin − min_bin + offset)` — an INNER-bin key. The plan must produce both bitsets consistently and choose the routing convention that matches the anchor.
**When to use:** In the §6.3 construction step. Anti-pattern: producing only one bitset and reusing it for both roles without checking the offset math.

### Anti-Patterns to Avoid
- **Importing `lgbm-treelearner` into `lgbm-compute`** to reuse `find_best_threshold_categorical` — this is the crate-cycle landmine (memory `on-device-driver-crate-cycle-constraint`; GrowFeature doc grow_driver.rs:45-48). Transcribe instead.
- **Hard-clamping `cat_threshold` to 32** — explicitly rejected by D-03; diverges from the reference split → breaks the ~1e-6 parity contract.
- **Per-`SplitInfo` device alloc** for the bitset — the whole point of ODL-02/D-03 pre-allocation; the C++ `AllocateCatVectorsKernel` (§8.3) is the anti-pattern.
- **Comparing two GPU f32 categorical paths** — def-f8u-01; anchor to the f64 fold.
- **`Atomic<i64>`** in a bitset-construction kernel — broken on this cubecl (lib.rs:1276). If a parallel atomic-OR is ever needed, use `Atomic<u64>` (as histogram.rs:1274 does) — but for tiny fixtures a single-owner sequential OR into a pre-zeroed slab is simpler and bit-identical to the host `construct_bitset`.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Categorical bin sort by ctr | A new sort kernel | `primitives::bitonic_argsort_on` (primitives.rs:969) | Anchor-pinned to `BitonicArgSort_1024`; index-only; single-block ≤1024 covers all fixtures |
| Categorical partition routing | New mark/scatter kernel | `partition_categorical_on_device` (data_partition.rs:686) | Already built + anchor-tested; uses shared `route_to_left_categorical`/`find_in_bitset` |
| Categorical tree mutation | New node-writer | `split_categorical_on_device` (tree.rs:765) | Already built; writes `kCategoricalMask`, `num_cat`, `cat_boundaries[_inner]` |
| Bitset membership test | New lookup | `find_in_bitset` (data_partition.rs:138) — shared by predict + partition | One transcription of `Common::FindInBitset`; branchless |
| The categorical split math | Novel algorithm | Transcribe `find_best_threshold_categorical` (feature_histogram_categorical.rs:93) byte-for-byte | It is already bit-exact to real 4.6 on the host; a re-derivation risks the kEpsilon / l2-asymmetry / cnt_factor subtleties |
| Real-category bitset construction | New bit-packer | Transcribe `construct_bitset` (feature_histogram_categorical.rs:424) | Host version is the golden-proven `Common::ConstructBitset` |

**Key insight:** ~70% of this phase is *wiring existing, anchor-tested kernels* into the driver, not writing new kernels. The two real new pieces (§8.1 eval, §6.3 construction) are **transcriptions of already-golden host code**, not fresh algorithms — the risk is transcription fidelity (kEpsilon bump, l2 asymmetry, tie order), not design.

## Runtime State Inventory

> This is an **additive feature phase** (new categorical code path behind `LGBM_CUDA_ON_DEVICE`), not a rename/refactor/migration. No stored data, live-service config, OS-registered state, secrets, or build artifacts carry a string being renamed. All five categories: **None — verified additive**; the only persisted state affected is the model text, which gains categorical nodes only when a categorical feature is trained on-device (a new capability, not a mutation of existing serialized models).

## Common Pitfalls

### Pitfall 1: Tie-order divergence between f64 stable-sort and f32 bitonic argsort
**What goes wrong:** The host anchor sorts bins by `ctr = grad/(hess+cat_smooth)` with an f64 `partial_cmp` **stable** sort (feature_histogram_categorical.rs:228-232). The reused `bitonic_argsort_on` primitive is **f32-keyed** and its tie behavior follows `BitonicArgSort_1024` (strict `>`, index tie-break), not `std::stable_sort`. On tied or near-tied ctr values the selected `sorted_idx` order differs → a different `cat_threshold` set → the real-golden bit-exact assert fails.
**Why it happens:** The reference C++ CUDA path itself uses `BitonicArgSort_1024` (§8.1), so the *golden* was produced by a bitonic sort, not a stable sort — but the HOST anchor uses stable-sort. The on-device transcription must match the **golden** (bitonic) for D-01#1 and the **host anchor** (stable) for D-01#2 simultaneously.
**How to avoid:** Design the fixture so ctr values are strictly distinct (the committed `cat_manyvsmany` grads are `0,-1,-2,-8,-9,-10` with `cat_smooth:0.0` and uniform hess ⇒ distinct ctr, no ties — verify this holds). Document that for tied ctr the two anchors could disagree and keep fixtures tie-free. If ties are unavoidable, the on-device sort must replicate whichever tie order the golden used.
**Warning signs:** `cat_threshold != golden` on `cat_manyvsmany` but gain matches within tolerance.

### Pitfall 2: `sum_hessian + 2*kEpsilon` bump applied at the wrong site
**What goes wrong:** `find_best_threshold_categorical` expects the leaf hessian sum **already bumped** by `+2*kEpsilon` (feature_histogram_categorical.rs:86-89). The host learner applies this at the call site (learner.rs:2760-2761: `sum_h_bumped = sum_h + 2.0*eps`). The on-device driver's `scan_leaf` passes the RAW `sum_h` to the numeric finder (which bumps internally). If the categorical branch forgets the call-site bump, `gain_shift`, `cnt_factor`, and every per-category gain shift by kEpsilon → last-ULP leaf-value / gain divergence.
**Why it happens:** Numeric and categorical finders have *different* bump conventions (numeric bumps inside; categorical expects pre-bumped).
**How to avoid:** In the driver's categorical branch, bump `sum_h` before calling the transcribed evaluator, mirroring learner.rs:2760.
**Warning signs:** Structure matches but leaf values differ at ~1e-15; the DEF-07-11-01-class RAW-vs-kEpsilon bug.

### Pitfall 3: one-hot vs many-vs-many `l2` asymmetry
**What goes wrong:** One-hot uses the ORIGINAL `lambda_l2`; many-vs-many uses `lambda_l2 + cat_l2` (feature_histogram_categorical.rs:126-130, 223). Applying `cat_l2` to the one-hot path (or omitting it from many-vs-many) shifts gains and outputs.
**How to avoid:** Branch `l2` exactly as the host anchor does — increment only in the many-vs-many `else` AFTER the one-hot branch returns (feature_histogram_categorical.rs:212, 223).
**Warning signs:** `cat_onehot` leaf values off by a `cat_l2`-sized amount.

### Pitfall 4: mfb_offset / most_freq_bin routing offset
**What goes wrong:** `route_to_left_categorical` uses `offset = (most_freq_bin == 0) ? 1 : 0` and membership `FindInBitset(bitset, bin − min_bin + offset)` (data_partition.rs:161-171). The `finalize` step adds `offset = meta_->offset` to each winning bin when building `cat_threshold` (feature_histogram_categorical.rs:383-396). Both fixtures use `most_freq_bin: 1` (fixtures) so `offset = offset_for_most_freq_bin(1)`. Mismatching the offset between construction and routing sends rows the wrong way.
**How to avoid:** Use `offset_for_most_freq_bin(most_freq_bin)` consistently (learner_parity.rs:1461 shows the harness convention); the existing `partition_categorical_on_device` already encodes the routing offset — feed it a bitset built with the matching offset.
**Warning signs:** partition counts (`left_count`/`right_count`) differ from the golden even though `cat_threshold` matches.

### Pitfall 5: WR-01 HistArena::swap aliasing (already fixed, do not re-fix)
**What goes wrong:** A latent slot-aliasing bug in `HistArena::swap` (memory `phase18-wr01-histarena-swap-aliasing`) that would bite the multi-leaf grow loop. It was **fixed in Phase 21** (STATE / CONTEXT code_context:245-247). The categorical grow loop inherits the fix.
**How to avoid:** No action; the driver already uses `partition_leaf_stable` into fresh `Vec<u32>` per leaf (grow_driver.rs:96-100, WR-04 note) — no running-map aliasing. Do not reintroduce a shared buffer.

### Pitfall 6: `GrowFeature` deliberately omits `bin_to_category`
**What goes wrong:** `GrowFeature` (grow_driver.rs:50) explicitly excludes the categorical `bin_to_category` table "which the on-device numeric grow loop does not consume this milestone" (grow_driver.rs:45-46). SetRealThreshold (§6.3) NEEDS it to map inner bins → real category values.
**How to avoid:** Add `bin_to_category: Vec<i32>` and the categorical config scalars (`cat_smooth`, `cat_l2`, `max_cat_threshold`, `max_cat_to_onehot`, `min_data_per_group`) to `GrowFeature` (or thread the config) — additive, native `Vec<i32>`/scalars, no `lgbm-treelearner` types (crate-cycle-safe). The harness already builds `bin_to_category` from the sidecar (learner_parity.rs:1470).

## Code Examples

### The dispatch sentinel to replace (best_split.rs)
```rust
// Source: crates/lgbm-compute/src/kernels/best_split.rs:640-643 (and :958-960, :1814-1816)
// Phase-22 categorical dispatch seam (D-04): the numerical core is continuous-only.
if task.is_categorical {
    return Ok(SplitScalars::default());   // ← is_valid=false sentinel; §8.1 eval replaces this
}
```

### The driver bail-out to replace (grow_driver.rs)
```rust
// Source: crates/lgbm-compute/src/kernels/grow_driver.rs:384-387
if na_as_missing || f.bin_type == BinType::Categorical {
    // Deferred (proving slice is numeric + non-NA); the finder rejects NA.
    continue;   // ← categorical features are skipped; §8.1 eval + branch replaces this
}
```

### The existing §10 mutation to call (tree.rs) — already built
```rust
// Source: crates/lgbm-compute/src/kernels/tree.rs:765-773
pub fn split_categorical_on_device(
    &mut self, client: &ComputeClient<R>, leaf_index: i32,
    real_feature_index: i32, missing_type: i32, scalars: &SplitScalars,
    bitset: &[u32], bitset_inner: &[u32],   // ← BOTH real + inner bitsets (Pattern 3)
) -> Result<SplitResult, ComputeError> { ... }
```

### The host construction to transcribe (learner.rs) — the two-bitset producer
```rust
// Source: crates/lgbm-treelearner/src/learner.rs:3648-3668
let cat_values: Vec<u32> = cat_threshold_bins.iter()
    .map(|&bin| f.bin_to_category.get(bin as usize).copied().unwrap_or(bin as i32) as u32)
    .collect();
let cat_bitset_real = construct_bitset(&cat_values);   // real category-value bitset
data_partition.split_categorical(best_leaf, new_right, &f.bins, &cat_bitset_real, &f.bin_to_category);
```

## State of the Art

| Old Approach (reference CUDA) | Current Approach (this port) | When Changed | Impact |
|-------------------------------|------------------------------|--------------|--------|
| Per-`SplitInfo` `cudaMalloc` for cat vectors (`AllocateCatVectorsKernel`, §8.3) | Pre-allocated SoA slab, allocate-once (`DeviceSplitInfo`, split_info.rs) | Phase 14 | No per-split alloc; D-03 makes width runtime |
| `atomicAdd_system(out+val/32, 1<<(val%32))` bitset construct (§6.3) | Single-owner sequential OR into pre-zeroed slab (cubecl-cpu has no atomics need at fixture scale) | Phase 22 | Bit-identical to host `construct_bitset`; no `Atomic<i64>` (broken) |
| f64 categorical on host learner (`find_best_threshold_categorical`) | Transcribed into `lgbm-compute` for the driver (crate-cycle) | Phase 22 | Same math, new location; anchor-tested against host + real golden |

**Deprecated/outdated:** none — this is a forward port; the host categorical path (`feature_histogram_categorical.rs`) remains the authoritative CPU learner path and the transcription source.

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | `cat_manyvsmany` fixture ctr values are strictly distinct (no sort ties) so f32-bitonic and f64-stable sorts agree | Pitfall 1 | If ties exist, on-device sort could pick a different `cat_threshold` than the golden — planner should verify by computing ctr from the sidecar grads before locking the fixture |
| A2 | The existing `partition_categorical_on_device` (data_partition.rs:686) and `split_categorical_on_device` (tree.rs:765) are fully anchor-tested and correct (only unwired) | Summary / Don't-Hand-Roll | If they have latent bugs, wiring surfaces them — planner should have the driver's first categorical structure-gate run confirm them end-to-end |
| A3 | Adding fields to `GrowFeature` introduces no crate cycle (native `Vec<i32>`/scalars only) | Pitfall 6 | If a `lgbm-treelearner`/`lgbm-dataset` type sneaks in, `cargo build -p lgbm-compute` breaks — planner must keep fields primitive |
| A4 | D-06's "has categorical && quantized" check belongs in the learner (`on_device_eligible`, learner.rs:498), since `on_device_growth_supported()` takes no config | Arch Map / D-06 | If placed in `lgbm-compute` it can't see `use_quantized_grad`/`bin_type` — wrong layer |
| A5 | `xtask categorical-oracle-capture` regenerates the goldens and asserts `lib_lightgbm` 4.6 (main.rs:286, py/categorical_oracle_capture.py) — goldens already committed | Parity Harness | If the installed wheel version drifts, recapture would assert-fail; goldens are already present so recapture is optional |

## Open Questions

1. **Which routing bitset convention does the on-device structure gate anchor expect — real category-value or inner-bin?**
   - What we know: `split_categorical_on_device` takes both; host learner routes by real-value (learner.rs:3641-3647); reference §9 routes by inner-bin (data_partition.rs:153-171); `partition_categorical_on_device` uses the inner-bin `route_to_left_categorical`.
   - What's unclear: whether feeding `partition_categorical_on_device` the real-value bitset (as the host does) vs an inner-bin bitset yields the same partition for these fixtures.
   - Recommendation: In the plan's first categorical wiring task, assert the on-device partition counts equal the host `partition_categorical_stable` (data_partition.rs:525) for both fixtures BEFORE asserting the full structure gate — this isolates the convention.

2. **Does D-03 need a dedicated `max_cat_threshold > 32` fixture to be non-vacuous?**
   - What we know: both committed goldens use `max_cat_threshold: 32` (fixtures), which equals the default — so the runtime-width change is not observably exercised by them.
   - Recommendation: add a `lgbm-compute` unit test constructing `DeviceSplitInfo` with a config width > 32 and asserting the slab length + a `set_cat_thresholds` of length 33 succeeds (currently rejected at split_info.rs:464). Optionally a small `cat_wide` golden.

3. **Single-block vs global-memory bitonic sort for many-vs-many?**
   - What we know: `cat_manyvsmany` has `num_bin: 7` (≤ 1024) → single-block `bitonic_argsort_on` suffices; the `_GlobalMemory` variant is deferred/optional (CONTEXT Deferred).
   - Recommendation: single-block only this phase.

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| cubecl-cpu runtime | Structure gate + all new kernels (default lane) | ✓ | 0.10 (workspace-pinned) | — |
| cubecl-hip (ROCm) | Optional best-effort smoke (D-04) | ✓ (spoofed 8-CU APU) | 0.10 | Skip — not the gate |
| `lib_lightgbm` 4.6 wheel | `xtask categorical-oracle-capture` recapture | goldens already committed | 4.6 (asserted by capture, main.rs:195) | Goldens present; recapture optional |
| Committed categorical goldens | D-01 real-golden bit-exact pin | ✓ | — | SKIP-passes if absent (learner_parity.rs:1387) |

**Missing dependencies with no fallback:** none.
**Missing dependencies with fallback:** ROCm real-hardware validation is deferred to Phase 23 (local GPU is a spoofed APU — memory `rocm-gfx1100-available`).

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | Rust built-in test harness (`cargo test`) + `oracle-harness` integration crate |
| Config file | none (Cargo default) |
| Quick run command | `cargo test -p lgbm-compute --lib grow_driver` (driver unit tests) |
| Full suite command | `cargo test --workspace` (env unset) + gated on-device run |
| Merge gate | cubecl-cpu f64 lane; `cargo test --workspace` env-unset stays byte-green (Phase-21 pattern, 21-01-PLAN.md:140) |

### Phase Requirements → Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| ODL-22 | Real-golden bit-exact (host anchor logic) | integration | `cargo test -p oracle-harness --test learner_parity -- learner_parity_categorical_onehot learner_parity_categorical_manyvsmany` | ✅ (learner_parity.rs:1533,1538) — HOST learner only |
| ODL-22 | On-device categorical structure gate (D-01#2) | integration | `LGBM_CUDA_ON_DEVICE=1 cargo test -p oracle-harness --test learner_parity -- learner_parity_on_device_structure_gate` | ⚠️ Wave 0 — extend with categorical corpus cases (currently numeric-only, learner_parity.rs:2329) |
| ODL-22 | §8.1 categorical eval matches host anchor | unit | `cargo test -p lgbm-compute --lib categorical_split` | ❌ Wave 0 — new module/tests |
| ODL-22 | §6.3 bitset construction bit-identical to host `construct_bitset` | unit | `cargo test -p lgbm-compute --lib` (new) | ❌ Wave 0 |
| ODL-22 | D-03 runtime slab width > 32 | unit | `cargo test -p lgbm-compute --lib split_info` | ❌ Wave 0 |
| ODL-22 | Numeric spine byte-unchanged (SC#4) | integration | `cargo test --workspace` (env unset) + `learner_parity_categorical_no_regression_numeric_spine` | ✅ (learner_parity.rs:1553) |
| ODL-22 | D-06 categorical+quantized → host fallback | unit | learner-crate test asserting `on_device_eligible == false` for that combo | ❌ Wave 0 |
| ODL-22 | Predict-through bitset | integration | round-trip in `run_categorical_cell` (learner_parity.rs:1522) + on-device predict | ✅ host / ⚠️ extend for device |

### Sampling Rate
- **Per task commit:** `cargo test -p lgbm-compute --lib` (unit) + `cargo build -p lgbm-compute`.
- **Per wave merge:** `cargo test --workspace` (env unset, byte-green) + `LGBM_CUDA_ON_DEVICE=1 cargo test -p oracle-harness --test learner_parity` (gated on-device gate non-vacuous).
- **Phase gate:** full suite green + both real-golden categorical cases pass through the DEVICE path.

### Wave 0 Gaps
- [ ] `crates/lgbm-compute/src/kernels/categorical_split.rs` (or `best_split.rs` additions) — transcribed §8.1 eval + tests covering ODL-22 (one-hot, many-vs-many, `max_num_cat` clamp)
- [ ] §6.3 bitset construction (SetRealThreshold + length + construct) + bit-identical-to-`construct_bitset` test
- [ ] `split_info.rs` D-03 runtime-width unit test (width > 32)
- [ ] `learner_parity.rs` on-device categorical structure-gate cases (extend `learner_parity_on_device_structure_gate`) — the D-01#2 gate
- [ ] learner-crate D-06 gate test (`on_device_eligible` false when categorical && quantized)
- [ ] `GrowFeature` categorical-field extension (additive) + a `grow_features_of`-analog that carries `bin_to_category` (extend learner_parity.rs:1982 helper)

## Security Domain

`security_enforcement: true`, ASVS level 1. This is a compute-kernel port with no network/auth/session/crypto surface; the only applicable category is **V5 Input Validation** at the V5 boundaries the codebase already enforces.

### Applicable ASVS Categories

| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | no | — |
| V3 Session Management | no | — |
| V4 Access Control | no | — |
| V5 Input Validation | yes | Typed `ComputeError::Runtime`/`LengthMismatch` at every host→device boundary; overflow-checked slab sizing; slot-index bounds |
| V6 Cryptography | no | — |

### Known Threat Patterns for {cubecl categorical kernels}

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| `num_cat_threshold` slab overflow (D-03 runtime width) | Tampering / DoS | `checked_mul` on slab length (existing split_info.rs:277); reject width overflow with typed error (extend for runtime width) |
| `cat_threshold.len() > slab width` write | Tampering | Existing guard split_info.rs:464 (`> MAX_CAT_PER_SPLIT`) — update to runtime width |
| Out-of-range bin in `bin_to_category` lookup (SetRealThreshold) | Tampering | `.get(bin).unwrap_or(...)` bounds (host learner.rs:3653); replicate in transcription |
| Bitset index OOB in `find_in_bitset` | Tampering | Existing branchless clamp (data_partition.rs:135-138) |
| `cat_boundaries` slab exceed `max_leaves+1` | Tampering | Existing check in `split_categorical_on_device` (tree.rs:782, cat-boundary slab length) |
| Categorical + quantized silent-wrong-answer | Tampering (silent behavior change) | D-06 honest host-fallback (learner gate); mirrors reference `asm("trap;")` |

## Sources

### Primary (HIGH confidence)
- `docs/cuda-kernel-design.md` §6.3 (lines 528-541), §8.1 (783-836), §8.3 (853-866), §9 (870-933), §10 (937-958) — the reference algorithm being ported (registered as the v1.1 C++ port-source map, STATE deferred-items 260629-djo)
- `crates/lgbm-compute/src/kernels/split_info.rs` — `MAX_CAT_PER_SPLIT` (:65), reserved cat slabs (:228-233), `set_cat_thresholds` guard (:464), `copy_slot` slab window (:551-553)
- `crates/lgbm-compute/src/kernels/best_split.rs` — categorical dispatch sentinel (:640-643, :958-960, :1814-1816), `SplitFindTask` flags (:98-129), task-emit (:239-252)
- `crates/lgbm-compute/src/kernels/grow_driver.rs` — `GrowFeature` (:50, omits bin_to_category :45-46), `scan_leaf` categorical skip (:384), driver body numeric split (:618-646, num_cat_threshold:0 :637)
- `crates/lgbm-compute/src/kernels/data_partition.rs` — `find_in_bitset` (:138), `route_to_left_categorical` (:153), `partition_categorical_on_device` (:686), `partition_categorical_stable` (:525)
- `crates/lgbm-compute/src/kernels/tree.rs` — `split_categorical_kernel` (:278), `split_categorical_on_device` (:765), cat_boundaries slab (:805-806)
- `crates/lgbm-compute/src/kernels/predict.rs` — categorical membership branch (:115-122)
- `crates/lgbm-compute/src/kernels/primitives.rs` — `bitonic_argsort_on` (:969), comparator fidelity (:876), single-owner/no-plane note (:51)
- `crates/lgbm-treelearner/src/feature_histogram_categorical.rs` — `find_best_threshold_categorical` (:93), l2 asymmetry (:126-130,223), ctr sort (:226-232), `max_num_cat` clamp (:236), `finalize`/`cat_threshold` build (:334-398), `construct_bitset` (:424)
- `crates/lgbm-treelearner/src/learner.rs` — host categorical grow (:2745-2775), real-bitset construction (:3639-3668), `on_device_eligible` (:498)
- `crates/lgbm-compute/src/lib.rs` — `on_device_growth_supported` (:1249 default, :1358 CpuBackend), `cuda_on_device_enabled` (:1324), cubecl-0.10 checklist (:1274-1279)
- `crates/oracle-harness/tests/learner_parity.rs` — `run_categorical_cell` (:1486), `cat_corpus` (:1454), sidecar loader (:1384), on-device structure gate (:2329), anchor comparator (:2252-2286), `grow_features_of` (:1982)
- `crates/lgbm-core/src/config/mod.rs` — categorical config defaults (:136-145, :365-369)
- Fixtures: `crates/oracle-harness/tests/fixtures/categorical/{cat_onehot,cat_manyvsmany}.{txt,bins.json}` (present; sidecars read in full)
- `xtask/src/main.rs:286` + `xtask/py/categorical_oracle_capture.py` — recapture command

### Secondary (MEDIUM confidence)
- STATE.md / MEMORY.md project memories: `on-device-driver-crate-cycle-constraint`, `phase18-wr01-histarena-swap-aliasing`, `on-device-kernel-goldens-are-retranscriptions`, `def-f8u-01`, `rocm-gfx1100-available`

### Tertiary (LOW confidence)
- none — all findings verified against current source or committed reference docs

## Metadata

**Confidence breakdown:**
- Standard stack (in-repo modules): HIGH — every seam located by file:line in the current tree
- Architecture (crate-cycle, two-bitset, single-owner): HIGH — grounded in code + doc + memory
- Pitfalls: HIGH for 2/3/4/5/6 (code-grounded); MEDIUM for Pitfall 1 tie-order (depends on fixture ctr distinctness — flagged as A1/OQ1)

**Research date:** 2026-07-02
**Valid until:** 2026-08-01 (stable in-repo port; re-verify only if the grow_driver or categorical kernels change before planning)
