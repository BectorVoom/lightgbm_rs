---
title: LightGBM-rs Parity Gap Closure (G2 JSON dump · G1 if-else codegen · G4 NA-as-missing serial · G5 split-kernel gain params)
status: draft
format: markdown
spec_version: 1
updated_at: 2026-07-12T00:00:00Z
source_requirements:
  - "User request: implement unimplemented lightgbm_rs comparing with lightgbm"
  - ".planning/plans/parity-gap-closure/research.md (this milestone's audit)"
  - ".planning/PROJECT.md (Key Decisions, Validated/Out-of-scope)"
  - "CLAUDE.md / AGENTS.md (numeric contract, dependency-first rule)"
---

> **PageIndex note:** the PageIndex library for this workspace is empty (research
> §12) — PageIndex MCP is **read-only / no target document** here. This SPEC is
> therefore staged locally as the authoritative draft. **Pending PageIndex
> update:** upsert this file into the project's PageIndex collection for
> `lightgbm_rs` once a collection/document id exists. No id could be identified
> safely, so none was invented.

# 1. Context

`lightgbm_rs` is a pure-Rust port of Microsoft LightGBM 4.6 (Cargo workspace,
CubeCL compute). The numeric contract (CLAUDE.md): `f32` end-to-end; the
`cubecl-cpu` f64-fold path is the **bit-exact hard merge gate** where the
algorithm permits, `cubecl-hip`/f32 held to ~1e-6.

This milestone (v1.0 "C++ Feature-Parity Audit & Gap Closure") closes four
remaining C++→Rust gaps identified in `research.md`. The prior audit's top items
(linear-tree, quantized-grad plumbing) are already **CLOSED** on branch
`feat/linear-tree`. The four gaps in scope, per the user's decisions:

- **G2 — Model JSON serialization** (`GBDT::DumpModel` / `Tree::ToJSON`) — MISSING.
- **G1 — If-else codegen** (`convert_model` / `ModelToIfElse`) — parsed but inert.
- **G4 — `na_as_missing` forward routing, serial host path** — typed-error-gated.
- **G5 — `path_smooth` / `max_delta_step` / feature-`penalty` in the split kernel**
  — config accepts, kernel rejects.

Locked user decisions carried into every spec below:
- **DEC-1** [G2 emitter]: **hand-emit** using the existing `%g` formatters
  (`format_g17`/`format_g6`, `lgbm-model/src/format.rs`). **No `serde_json`
  dependency.** `[VERIFIED: LOCAL grep format.rs:43,52]`
- **DEC-2** [G1 API]: public entry point is **`Booster::model_to_cpp() -> String`**
  (a pure in-memory method mirroring `Booster::model_to_string()`,
  `crates/lgbm/src/booster.rs:728`). No file-writing side effect in v1.
  `[VERIFIED: LOCAL grep booster.rs:728]`

# 2. Scope and Non-goals

**In scope:** G1, G2, G4 (serial host branch only), G5 (`path_smooth`,
`max_delta_step`, feature-`penalty`) — each with an oracle-harness parity test
against a committed `lib_lightgbm` 4.6 golden.

**Non-goals (explicit):**
- G3 categorical GPU kernels (deferred — largest/riskiest, needs GPU parity infra).
- G4 GPU-resident NA scan (`histogram.rs:2482,2778`; `partition.rs:9`) — the
  resident path is coupled to the out-of-scope-by-default on-device grow loop.
- G6 stochastic-rounding oracle residual — env-blocked, already specced under
  `.planning/plans/quantized-grad-param-plumbing/`.
- C-API, distributed/MPI, OpenCL `gpu` device knobs — **OUT OF SCOPE (locked,
  PROJECT.md)**. Do not scope work there.
- No file-writing `convert_model` path side effect for G1 (DEC-2).

# 3. Dependencies

**AGENTS.md rule — dependencies confirmed FIRST:**
- **No new external crate is required for any of G1/G2/G4/G5.**
  `[VERIFIED: LOCAL research §6; grep serde_json → absent]`
- G2/G1 reuse: `lgbm_model::format::{format_g17, format_g6}`
  (`format.rs:43,52`); the `Tree` node arrays (`tree.rs:94-155`); `GbdtModel`
  ensemble fields (`ensemble.rs`); `model_text::save` structure as the layout
  template (`model_text.rs:216`). `[VERIFIED: LOCAL]`
- G5 reuses: `GainConfig` (`crates/lgbm-compute/src/gain.rs:377`) which already
  carries `max_delta_step` and `path_smooth` fields plumbed from
  `lgbm_core::Config` (`config/mod.rs:101,181`); the split-gain scan in
  `find_best_split_cpu` (`crates/lgbm-compute/src/kernels/split.rs`).
  `[VERIFIED: LOCAL gain.rs:377-430, split.rs:433-460]`
- G4 touches: `crates/lgbm-treelearner/src/learner.rs` (the NA gate at `:1113`),
  the serial histogram build + `data_partition` routing. `[VERIFIED: LOCAL
  Read learner.rs:1107-1121]`

**C++ reference dependency (BLOCKER for exactness):** the read-only `LightGBM/`
tree is **absent in this sandbox** (research §1). Exact C++ algorithm details for
G4 (NA forward routing) and G5 (`path_smooth`/`max_delta_step` gain & leaf-output
formulas) are cited from design docs and are `[UNVERIFIED against C++ source]`.
The implementer **MUST** re-verify against `LightGBM/src/treelearner/feature_histogram.hpp`
and `src/io/tree.cpp` / `src/treelearner/serial_tree_learner.cpp` before coding —
see per-spec "C++ verification required" notes. Golden generation needs
`lightgbm==4.6.0` (uv wheel, per memory `cpp-linear-tree-oracle.md`).

# 4. Typed Contracts

```rust
// ---- G2: JSON model dump (crate: lgbm-model, new module `json.rs`) ----
// Hand-emitted; floats via format_g17/format_g6 (DEC-1). No serde.
mod json {
    /// Serialize one tree's node graph to the C++ `Tree::ToJSON`
    /// "tree_structure" object (recursive split/leaf nodes).
    pub fn tree_to_json(tree: &Tree) -> String;
    /// Serialize the full ensemble to the C++ `GBDT::DumpModel` document
    /// (name, version, num_class, num_tree_per_iteration, label_index,
    /// max_feature_idx, average_output?, feature_names, monotone_constraints?,
    /// feature_infos, objective, tree_info[]).
    pub fn dump_model(model: &GbdtModel) -> String;
}
// Facade (crate: lgbm, crates/lgbm/src/booster.rs):
impl Booster { pub fn dump_model(&self) -> String; }

// ---- G1: if-else codegen (crate: lgbm-model, new module `codegen_cpp.rs`) ----
mod codegen_cpp {
    /// Emit the C++ predict-function body for one tree (nested if-else over
    /// split_feature/threshold/decision_type, leaf_value at the leaves).
    pub fn tree_to_cpp(tree: &Tree, tree_index: usize) -> String;
    /// Emit the full standalone C++ source (headers, per-tree functions,
    /// PredictRaw summation over the ensemble). Mirrors ModelToIfElse.
    pub fn model_to_cpp(model: &GbdtModel) -> String;
}
// Facade (DEC-2):
impl Booster { pub fn model_to_cpp(&self) -> String; }

// ---- G4: na_as_missing serial forward routing (crate: lgbm-treelearner) ----
// No new public type. Replaces the typed-error gate at learner.rs:1113 with
// correct forward routing in serial histogram build + partition.
// Contract: for a numerical feature with missing_type==NaN && num_bin>2,
// NA rows accumulate into / route down the tree's default branch identically
// to C++, and train() no longer returns TreeLearnerError::Compute(Runtime{..}).

// ---- G5: split-kernel gain params (crate: lgbm-compute, split.rs) ----
// No new public type. find_best_split_cpu stops returning ComputeError::Runtime
// for non-default max_delta_step / path_smooth, and applies feature `penalty`
// (currently hard-coded 1.0). GainConfig already carries the fields.
```

# 5. Failure-Isolated Behavioral Specifications

Each spec has ONE behavioral responsibility with one primary failure cause.
Status: **draft**. IDs are stable and referenced by PLAN.md tasks.

---

## G2 — Model JSON serialization

### SPEC-G2-1 — Single-tree JSON (`tree_to_json`)
- **status:** draft
- **rationale/source:** research §5 G2; C++ `Tree::ToJSON`.
- **preconditions:** a valid `Tree` (`num_leaves ≥ 1`; node arrays sized per
  `tree.rs:94-155`).
- **input:** `&Tree` (non-null; may be `is_linear`).
- **output:** `String` — a JSON `"tree_structure"` object; internal nodes carry
  `split_index`, `split_feature`, `split_gain`, `threshold`, `decision_type`,
  `default_left`, `missing_type`, `internal_value`, `internal_weight`,
  `internal_count`, `left_child`, `right_child`; leaves carry `leaf_index`,
  `leaf_value`, `leaf_weight`, `leaf_count`. Floats via `%g` (DEC-1).
- **dependencies:** `format::{format_g17,format_g6}`; `decision_type` bit decode
  (`tree.rs:24-26,45-50`: bit0 categorical, bit1 default-left, bits2-3
  missing_type).
- **behavior (G/W/T):** *Given* a tree with a numeric split at the root, *when*
  `tree_to_json` runs, *then* the emitted object nests `left_child`/`right_child`
  recursively and each float field is byte-identical to C++ `%g`.
- **invariants:** recursion order = C++ pre-order from node 0; categorical splits
  emit `cat_threshold`/`cat_boundaries`-derived `threshold` list form.
- **acceptance test:** structural + byte-exact float compare of one tree against
  a `DumpModel` golden slice (see SPEC-G2-4).
- **out of scope:** ensemble-level fields (SPEC-G2-2).
- **traceability:** `[CODEGRAPH lgbm-model/src/tree.rs:Tree]`,
  `[LOCAL tree.rs:94-155]`.
- **C++ verification required:** exact JSON key set/order and categorical
  representation from `LightGBM/src/io/tree.cpp` `Tree::NodeToJSON`.

### SPEC-G2-2 — Ensemble JSON document (`dump_model`)
- **status:** draft
- **rationale/source:** C++ `GBDT::DumpModel`.
- **preconditions:** a loaded `GbdtModel` (≥ 0 trees).
- **input:** `&GbdtModel`.
- **output:** `String` — top-level JSON object: `"name"`, `"version"`,
  `"num_class"`, `"num_tree_per_iteration"`, `"label_index"`,
  `"max_feature_idx"`, `"average_output"` (only when set), `"objective"`,
  `"feature_names"` (array), `"monotone_constraints"` (only when set),
  `"feature_infos"`, `"tree_info"` (array of `{tree_index, num_cat, shrinkage,
  tree_structure}`). Uses SPEC-G2-1 per tree.
- **dependencies:** SPEC-G2-1; `GbdtModel` fields (mirror `model_text::save`
  field order, `model_text.rs:216-260`).
- **behavior (G/W/T):** *Given* a 2-tree binary model, *when* `dump_model` runs,
  *then* the document parses as valid JSON and every scalar matches the C++
  `DumpModel` golden.
- **invariants:** field presence rules match `model_text::save` (e.g.
  `average_output` and `monotone_constraints` emitted only when present).
- **acceptance test:** SPEC-G2-4.
- **out of scope:** facade wiring (SPEC-G2-3).
- **traceability:** `[LOCAL ensemble.rs, model_text.rs:216-260]`.
- **C++ verification required:** top-level key set/order from `DumpModel`.

### SPEC-G2-3 — Public facade (`Booster::dump_model`)
- **status:** draft
- **input:** `&self` (a trained/loaded `Booster`).
- **output:** `String` (delegates to SPEC-G2-2 over the booster's `GbdtModel`).
- **behavior:** *Given* a trained booster, *when* `dump_model()` is called,
  *then* it returns the same string as `json::dump_model(&model)` — pure
  orchestration, no recomputation.
- **dependencies:** SPEC-G2-2; `Booster` model accessor (as used by
  `model_to_string`, `booster.rs:728`).
- **acceptance test:** unit test asserting facade output == module output.
- **traceability:** `[LOCAL booster.rs:728]`.

### SPEC-G2-4 — JSON parity vs `lib_lightgbm` 4.6
- **status:** draft
- **input:** committed golden JSON from `lightgbm==4.6.0` `.dump_model()` plus the
  matching captured model.
- **output/observable:** a `#[test]` in
  `crates/oracle-harness/tests/json_dump_parity.rs` that loads the committed
  model, runs `dump_model`, and asserts equality with the golden (float fields
  compared via `%g` byte-exact; SKIP-gracefully when golden absent, matching
  `predict_parity.rs:read_golden`).
- **dependencies:** SPEC-G2-2; oracle golden capture (xtask subcommand or
  `gen_golden.py`, per §7 research).
- **acceptance test:** the parity test passes with a golden present; builds+passes
  (SKIP) without one.
- **traceability:** `[LOCAL predict_parity.rs:1-40 idiom]`.

---

## G1 — If-else codegen (`Booster::model_to_cpp`)

### SPEC-G1-1 — Single-tree C++ fragment (`tree_to_cpp`)
- **status:** draft
- **input:** `&Tree`, `tree_index: usize`.
- **output:** `String` — a C++ function body computing the tree's leaf output via
  nested `if (arr[split_feature] <= threshold)` (numeric), categorical bitset
  membership for categorical splits, honoring `default_left`/missing routing;
  returns the `leaf_value` (`%g` via DEC-1).
- **dependencies:** `Tree` arrays; `format_g17`; `decision_type` decode.
- **behavior (G/W/T):** *Given* a numeric-split tree, *when* `tree_to_cpp`
  emits, *then* the generated branch structure matches C++ `Tree::ToIfElse`
  node walk order and thresholds are `%g`.
- **invariants:** decision comparison direction and missing/default-left handling
  identical to `Tree::predict` (`tree.rs:269`).
- **acceptance test:** SPEC-G1-3 golden slice.
- **C++ verification required:** `LightGBM/src/io/tree.cpp` `Tree::NodeToIfElse`
  branch comparison + missing handling.

### SPEC-G1-2 — Full C++ source (`model_to_cpp`)
- **status:** draft
- **input:** `&GbdtModel`.
- **output:** `String` — a compilable standalone C++ source: include guard/
  headers, one function per tree (SPEC-G1-1), and a `PredictRaw` that sums tree
  outputs × shrinkage per `num_tree_per_iteration` class stride, mirroring
  `ModelToIfElse`.
- **dependencies:** SPEC-G1-1; ensemble metadata.
- **behavior:** *Given* a 2-tree model, *when* `model_to_cpp` runs, *then* the
  output equals the C++ `convert_model` golden byte-for-byte.
- **acceptance test:** SPEC-G1-4.
- **C++ verification required:** overall file skeleton/prototype names from
  `gbdt_model_text.cpp` `SaveModelToIfElse`.

### SPEC-G1-3 — Public facade (`Booster::model_to_cpp`) [DEC-2]
- **status:** draft
- **input:** `&self`.
- **output:** `String` (delegates to SPEC-G1-2). No file side effect.
- **behavior:** pure orchestration mirroring `model_to_string` (`booster.rs:728`).
- **acceptance test:** unit test facade == module output.

### SPEC-G1-4 — If-else parity vs `lib_lightgbm` 4.6
- **status:** draft
- **input:** golden `.cpp` from `lightgbm==4.6.0` `convert_model` + captured model.
- **output/observable:** `crates/oracle-harness/tests/ifelse_codegen_parity.rs`
  byte-exact compare (SKIP-gracefully when absent).
- **dependencies:** SPEC-G1-2; golden capture.
- **acceptance test:** passes with golden, SKIP without.

---

## G4 — `na_as_missing` forward routing (serial host path)

> **C++ verification required (BLOCKER):** exact NA→bin accumulation and default
> branch direction from `LightGBM/src/treelearner/serial_tree_learner.cpp` +
> `feature_histogram.hpp` + `src/io/dataset.cpp` bin handling. Do NOT infer the
> algorithm; the `LightGBM/` tree must be checked out first. Marked `TBD` fields
> below resolve from that source.

### SPEC-G4-1 — NA rows accumulate into the missing bin during histogram build
- **status:** draft
- **rationale/source:** research §5 G4; `learner.rs:1113` gate.
- **preconditions:** a numerical feature with `missing_type == NaN` and
  `num_bin > 2` (`na_as_missing()` true, `tree.rs`/dataset bin metadata).
- **input:** binned feature column with the NA sentinel bin; per-row grad/hess.
- **output/observable:** the constructed node histogram places each NA row's
  (grad,hess) into the designated missing bin **exactly** as C++ (bit-exact on
  the f64-fold CPU path).
- **dependencies:** serial histogram construction (lgbm-treelearner /
  lgbm-compute CPU path).
- **behavior (G/W/T):** *Given* a corpus with genuine NaN values in a
  `use_missing=true` feature, *when* the node histogram is built, *then* the
  per-bin (Σg, Σh) equals the C++ golden bin-for-bin.
- **acceptance test:** SPEC-G4-4 histogram assertion.
- **out of scope:** split direction (SPEC-G4-2).
- **TBD:** exact missing-bin index convention — resolve from C++ source.

### SPEC-G4-2 — Split finding + application route NA down the default branch
- **status:** draft
- **input:** node histogram (incl. missing bin) + best-split decision.
- **output/observable:** during data partition, NA rows follow the split's
  default direction (`default_left` per `decision_type` bit1), and the resulting
  child leaf memberships match C++ row-for-row.
- **dependencies:** SPEC-G4-1; `data_partition` routing in the serial learner.
- **behavior (G/W/T):** *Given* the best split on the NA feature, *when* rows are
  partitioned, *then* each NA row lands in the same child as C++.
- **acceptance test:** SPEC-G4-4 prediction assertion.
- **TBD:** default-direction selection rule — resolve from C++.

### SPEC-G4-3 — Remove the typed-error gate
- **status:** draft
- **input:** a training run with an `na_as_missing` feature.
- **output/observable:** `SerialTreeLearner::train` no longer returns
  `TreeLearnerError::Compute(ComputeError::Runtime{ detail: "…NA_AS_MISSING
  forward branch not implemented" })` (`learner.rs:1113`); training completes.
- **dependencies:** SPEC-G4-1, SPEC-G4-2 (gate removed ONLY after routing works).
- **behavior:** *Given* SPEC-G4-1/2 pass, *when* the gate is removed, *then* the
  previously-rejecting corpus trains and the prior "rejects" unit test is updated
  to assert success.
- **acceptance test:** the corpus trains; SPEC-G4-4 passes.
- **traceability:** `[LOCAL learner.rs:1113]`.

### SPEC-G4-4 — NA-as-missing parity vs `lib_lightgbm` 4.6
- **status:** draft
- **input:** committed golden (bin histograms + predictions) from a
  `lightgbm==4.6.0` run on a NaN-bearing corpus with `use_missing=true`.
- **output/observable:** `crates/oracle-harness/tests/na_missing_parity.rs`:
  (1) CPU f64-fold node histograms bit-exact vs golden; (2) predictions within
  the CPU bit-exact contract. Run under `LGBM_CUDA_ON_DEVICE=0` (research §8).
- **dependencies:** SPEC-G4-1..3; golden capture.
- **acceptance test:** passes with golden, SKIP without.

---

## G5 — Split-kernel gain params (`path_smooth`, `max_delta_step`, feature penalty)

> **C++ verification required (BLOCKER):** exact formulas from
> `LightGBM/src/treelearner/feature_histogram.hpp` `GetLeafGain` /
> `CalculateSplittedLeafOutput` (`path_smooth` smoothing, `max_delta_step`
> clamp) and the `meta_->penalty` multiply (`output->gain *= penalty`). The
> `LightGBM/` tree must be checked out; do NOT infer.

### SPEC-G5-1 — Feature split penalty applied to gain
- **status:** draft
- **rationale/source:** `split.rs` doc "`penalty` defaults to 1.0 … not yet
  implemented"; C++ `output->gain *= penalty`.
- **preconditions:** a per-feature penalty vector (from `cegb_penalty_feature_*`
  / feature-contribution penalties as C++ derives `meta_->penalty`).
- **input:** `find_best_split_cpu` gains a `penalty: f64` (or reads it from
  threaded config) instead of the hard-coded `1.0`.
- **output/observable:** the returned split `gain` equals the unpenalized gain ×
  `penalty`, bit-exact vs C++.
- **dependencies:** `find_best_split_cpu` (`split.rs`); penalty source plumbing.
- **behavior (G/W/T):** *Given* a feature with `penalty = 0.5`, *when* the split
  gain is computed, *then* it is exactly half the `penalty = 1.0` gain.
- **acceptance test:** SPEC-G5-4 (penalty golden).
- **TBD:** exact source of per-feature penalty in this port — resolve from C++
  `meta_->penalty` derivation + existing CEGB plumbing
  (`cost_effective_gradient_boosting.rs`).

### SPEC-G5-2 — `max_delta_step` leaf-output clamp
- **status:** draft
- **preconditions:** `Config.max_delta_step != 0.0`
  (`config/mod.rs:101`; already in `GainConfig`, `gain.rs`).
- **input:** `find_best_split_cpu` / leaf-output calc reads `max_delta_step`.
- **output/observable:** leaf outputs (and the gains derived from them) apply the
  C++ `max_delta_step` clamp; `find_best_split_cpu` no longer returns
  `ComputeError::Runtime` for non-default `max_delta_step`.
- **dependencies:** `GainConfig.max_delta_step` (present); leaf-output formula.
- **behavior:** *Given* `max_delta_step = 0.7`, *when* a leaf output exceeds the
  clamp, *then* it is clamped exactly as C++.
- **acceptance test:** SPEC-G5-4 (max_delta_step golden).

### SPEC-G5-3 — `path_smooth` smoothing
- **status:** draft
- **preconditions:** `Config.path_smooth != 0.0` (`config/mod.rs:181`; in
  `GainConfig`, `gain.rs`). `path_smooth` is **IN_SCOPE**.
- **input:** `find_best_split_cpu` / leaf-output reads `path_smooth` and the
  parent node count/weight (path-smoothing needs the parent output).
- **output/observable:** leaf outputs blend toward the parent per the C++
  `path_smooth` formula; `find_best_split_cpu` no longer rejects non-default
  `path_smooth`.
- **dependencies:** `GainConfig.path_smooth` (present); parent-output threading.
- **behavior:** *Given* `path_smooth = 2.0`, *when* leaf output is computed,
  *then* it equals the C++ smoothed value bit-exact.
- **acceptance test:** SPEC-G5-4 (path_smooth golden).
- **TBD:** whether the parent-output term is already available at the
  `find_best_split_cpu` call site — resolve during Red step.

### SPEC-G5-4 — Gain-param parity vs `lib_lightgbm` 4.6
- **status:** draft
- **input:** three committed goldens from `lightgbm==4.6.0`, each trained with one
  of `feature penalty` / `max_delta_step=0.7` / `path_smooth=2.0` set (others
  default).
- **output/observable:** `crates/oracle-harness/tests/gain_params_parity.rs`:
  split-gain + threshold + leaf-output bit-exact (CPU f64-fold) vs each golden.
  Run under `LGBM_CUDA_ON_DEVICE=0`.
- **dependencies:** SPEC-G5-1..3; golden capture.
- **acceptance test:** passes with goldens, SKIP without.

# 6. Acceptance Scenarios (end-to-end)

- **AS-1 (G2):** `Booster::dump_model()` on a trained binary model returns JSON
  byte-equal (floats via `%g`) to `lightgbm==4.6.0` `.dump_model()`. → SPEC-G2-4.
- **AS-2 (G1):** `Booster::model_to_cpp()` returns a `.cpp` byte-equal to
  `lightgbm==4.6.0` `convert_model` output. → SPEC-G1-4.
- **AS-3 (G4):** training on a NaN-bearing `use_missing=true` corpus completes
  (no typed error) and reproduces C++ histograms/predictions on the CPU anchor.
  → SPEC-G4-4.
- **AS-4 (G5):** training with each of `penalty` / `max_delta_step` / `path_smooth`
  set reproduces C++ split gains + leaf outputs on the CPU anchor. → SPEC-G5-4.

# 7. Impact Scope

| Spec | Classification | Impacted symbols/files |
|---|---|---|
| G2-1..3 | local (lgbm-model) + cross-module (lgbm facade) | new `lgbm-model/src/json.rs`; `lgbm/src/booster.rs` (+`dump_model`) |
| G2-4 | test | new `oracle-harness/tests/json_dump_parity.rs` + fixture |
| G1-1..3 | local + cross-module | new `lgbm-model/src/codegen_cpp.rs`; `lgbm/src/booster.rs` (+`model_to_cpp`) |
| G1-4 | test | new `oracle-harness/tests/ifelse_codegen_parity.rs` + fixture |
| G4-1..3 | cross-module (hot path) | `lgbm-treelearner/src/learner.rs`; serial histogram/partition; possibly `lgbm-compute` CPU histogram |
| G4-4 | test | new `oracle-harness/tests/na_missing_parity.rs` + fixture |
| G5-1..3 | cross-module (hot split kernel) | `lgbm-compute/src/kernels/split.rs`; `gain.rs`; config threading |
| G5-4 | test | new `oracle-harness/tests/gain_params_parity.rs` + fixtures |

**Blast-radius note:** G4 and G5 touch hot histogram/split paths — schedule them
apart (research §10) so a parity regression is attributable to one gap. The
`Booster` public API grows two methods (G1/G2); Python binding exposure is
optional and **out of scope for v1** unless separately requested.

# 8. Compatibility and Migration

- G1/G2 are **additive** public methods — no breaking change.
- G4/G5 change behavior only for inputs that **currently error** (`na_as_missing`
  features; non-default `path_smooth`/`max_delta_step`) — strictly widening.
- No model-format change; no persisted-schema migration.
- Test discipline: run treelearner/on-device tests under
  **`LGBM_CUDA_ON_DEVICE=0`** until commit `42249ca`'s regression is resolved
  (research §8) — otherwise 6 learner tests mask results.

# 9. Risks and Open Questions

- **R-1 (blocking exactness):** `LightGBM/` C++ tree absent → G4/G5 algorithm
  details unverified. **Resolution owner: implementer must checkout `LightGBM/`
  before the Green step of G4/G5.**
- **R-2:** float drift if any emitter uses `ryu`/`to_string()` instead of `%g` →
  byte-diff vs golden. Mitigation: `format_g17`/`format_g6` only (DEC-1).
- **R-3:** golden generation needs `lightgbm==4.6.0` (uv wheel) — env provisioning.
  Tests SKIP-gracefully so the suite still builds without goldens.
- **R-4:** `lgbm-python` link failure (`python3.14`) — Python-level exposure of
  G1/G2 can't be validated here; kept out of v1 scope.
- **OQ-1:** exact per-feature `penalty` source for G5-1 (CEGB vs
  `meta_->penalty`) — resolve from C++ during planning-of-record for that task.
- **OQ-2:** does the `find_best_split_cpu` call site already have the parent
  output needed for `path_smooth` (G5-3)? — resolve in the Red step.

# 10. Traceability and Sources

- Research: `.planning/plans/parity-gap-closure/research.md` (§4 gap table, §5
  per-gap detail, §6 deps, §7 commands, §8 risks).
- Verified local symbols: `format.rs:43,52`; `model_text.rs:216-260`;
  `booster.rs:728`; `tree.rs:94-155,269`; `gain.rs:377-430`;
  `split.rs:433-460`; `learner.rs:1107-1121`; `config/mod.rs:101,181`;
  `oracle-harness/tests/predict_parity.rs:1-40` (golden/SKIP idiom).
- Project constraints: `CLAUDE.md`, `.planning/PROJECT.md`, `AGENTS.md`.
- Evidence labels throughout; C++-source-derived claims marked
  `[UNVERIFIED against C++ source]` pending `LightGBM/` checkout.
