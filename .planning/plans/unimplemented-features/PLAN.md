> **Planner note.** This file is the successor to
> `.planning/plans/parity-gap-closure/PLAN.md` (superseded 2026-07-16), rebuilt
> directly from `.planning/plans/unimplemented-features/SPEC.md` (spec_version 2)
> and `.../research.md` (2026-07-16 audit), with every symbol/file/line claim
> re-verified this session via CodeGraph (`codegraph_explore`) and targeted
> `grep`/`Read`. Where this session's verification **narrows or sharpens** the
> SPEC's implied scope (see "Session findings that revise scope" below), that is
> called out explicitly rather than silently assumed. No production code was
> written or modified to produce this plan.

# TDD Implementation Plan — Parity Gap Closure (G2 · G1 · G4 · G5)

Derived from `.planning/plans/unimplemented-features/SPEC.md` (draft, spec_version
2) and `research.md`. Every implementation unit is Red → Green → Refactor →
Verify. Tasks are ordered by dependency, not file layout. **No task is marked
complete during planning** — `status: pending` for all.

---

## Session findings that revise / sharpen SPEC's implied scope

These are new evidence from this session's CodeGraph + grep verification, not
present (or only partially present) in SPEC.md/research.md. They matter for
correct task sizing — recorded here so the implementer does not under-scope
G4/G5 by trusting the single file:line citations in SPEC §3 at face value.

- **F-1 (G5, penalty): no `Config.feature_contri` field exists yet.**
  `feature_contri` is listed in `IN_SCOPE_PARAMS`
  (`crates/lgbm-core/src/config/scope.rs:99`) and aliased (`feature_contrib`,
  `fc`, `fp`, `feature_penalty` → `feature_contri`,
  `crates/lgbm-core/src/config/alias.rs:105-108`), but `crates/lgbm-core/src/config/mod.rs`
  has **no `feature_contri` field** (`grep` for `feature_contri:` — zero hits;
  compare `monotone_constraints: Vec<i32>` at `config/mod.rs:154` as the sibling
  per-feature-vector pattern). **This resolves OQ-1's "source" half**: the
  C++ param name is `feature_contri` (a per-feature float list, C++ default all
  `1.0`), not the CEGB penalty (`cost_effective_gradient_boosting.rs`, which is a
  distinct, already-wired mechanism — `cegb_penalty_split`,
  `penalty_feature_coupled`, `penalty_feature_lazy`,
  `cost_effective_gradient_boosting.rs:33-39,57-65`). SPEC-G5-1's Green step MUST
  therefore include adding `Config.feature_contri: Vec<f64>` (parse + default) —
  this is materially more work than "thread an existing scalar," which is how
  SPEC's typed contract reads. `[VERIFIED: LOCAL grep scope.rs:99, alias.rs:105-108, config/mod.rs]`
- **F-2 (G5): the `max_delta_step`/`path_smooth` rejection and the hard-coded
  `penalty = 1.0` are each duplicated across ~8-9 near-identical call sites in
  `crates/lgbm-compute/src/kernels/split.rs`**, not the single site SPEC §3 cites
  (`split.rs:442-443,551-553`). Confirmed rejection-gate line numbers: 553, 4712,
  5225, 5918, 6359, 6693, 7176 (`find_best_split_cpu_native`), 7620
  (`find_best_split_cpu_native_2lane`). Confirmed `penalty = 1.0` hard-codes:
  678, 5127, 5509, 5617/5671 (`#[cube]` kernel), 5768/5812, 7334. Most of these
  belong to the fused/batched/staged/resident-reduce scan family that serves the
  **on-device grow loop** (explicitly OUT OF SCOPE per SPEC §2 non-goals: "G4
  GPU-resident NA scan... coupled to the out-of-scope-by-default on-device grow
  loop" — the same reasoning applies to G5's fused/resident kernels).
  `[VERIFIED: LOCAL grep split.rs]`
- **F-3 (G5): the production default CPU path is `CpuBackend`, NOT the cubecl
  `GpuBackend<CpuRuntime>` kernel path.** `crates/lgbm-compute/src/lib.rs:2324`
  (`pub struct CpuBackend;`) implements `Backend::find_best_split` (line 2417) by
  calling `kernels::split::find_best_split_cpu_native` (or `_2lane` under
  `LGBM_SPLIT_2LANE=1`) — a **native, non-cubecl-launched** Rust reimplementation
  documented as "bit-identical to the single-unit find_best_split_ kernel,
  without the per-(feature,leaf) cubecl launch" (`lib.rs:2433-2436`).
  `lgbm-treelearner`'s serial learner tests instantiate `CpuBackend` directly
  (`crates/lgbm-treelearner/src/learner.rs:4372,4440,4460`). The cubecl-kernel
  path (`GpuBackend<R>::find_best_split` → `find_best_split_f64_on`,
  `lib.rs:3494-3524`) is the kernel-parity / ROCm-mirror path. **Consequence for
  scope**: the mandatory G5 (and G4) Green targets for "serial host path" are
  `find_best_split_cpu_native` (`split.rs:7129`, gate `:7176`) as the primary
  production CPU target, PLUS `find_best_split_f64_on` (`split.rs:493`, gate
  `:553`) for cubecl-cpu kernel-parity/ROCm-mirror coverage. The other ~6-7
  fused/batched/resident gate sites are LEFT REJECTING (out of scope, on-device
  grow loop). `[VERIFIED: LOCAL lib.rs:2324,2417-2450,3394,3494-3524; learner.rs:4372,4440,4460]`
- **F-4 (G5-3, path_smooth): the smoothing GAIN MATH already exists and is
  already C++-verified + unit-tested** in `crates/lgbm-compute/src/gain.rs`:
  `calculate_splitted_leaf_output_smoothed` (`:183-197`, doc-cites
  `cuda_leaf_splits.hpp:74-90`) and `get_leaf_gain_smoothed` (`:208-229`, cites
  `:117-121`), plus f32 mirrors (`:306-346`) and a passing test
  (`smoothing_blend_matches_reference`, `:577-623`). **This means SPEC-G5-3's
  Green step is "wire existing, already-tested math into the host call site +
  thread `parent_output`/`num_data` + drop the rejection," NOT "derive the
  formula from scratch."** The doc-comment at `gain.rs:1-32` frames this as
  "ADDITIVE ONLY... the Wave-2 stage-1 body dispatches to these NEW fns when
  `use_smoothing` is set" — i.e. the wiring was anticipated but not finished.
  OQ-2 (is `parent_output` available at the `find_best_split_cpu_native` call
  site in `learner.rs`?) remains **genuinely open** — resolve in T-G5-3's Red
  step. `[VERIFIED: LOCAL Read gain.rs full file]`
- **F-5 (G5-2, max_delta_step): NO existing clamp formula anywhere in `gain.rs`**
  (zero occurrences of a clamp/`min`/`max` combinator keyed on `max_delta_step`
  besides the `GainConfig` field itself and doc comments). Unlike path_smooth,
  this sub-task requires **deriving and transcribing new formula code** from
  `LightGBM/src/treelearner/feature_histogram.hpp` `CalculateSplittedLeafOutput`
  (the `USE_MAX_OUTPUT=true` branch) after the P-1 checkout — genuinely blocked
  on C++ verification, not just plumbing. `[VERIFIED: LOCAL grep gain.rs]`
- **F-6 (G4): the NA_AS_MISSING forward-branch preamble is a real MISSING
  KERNEL ALGORITHM, not just a routing/gate issue.** `find_best_split_cpu_native`
  itself (the `CpuBackend` production path, `split.rs:7146-7150`) ALSO rejects
  `na_as_missing == true` unconditionally ("NA_AS_MISSING forward branch not yet
  implemented") — this rejection is duplicated at `split.rs:514-517` (`find_best_split_f64_on`),
  `:2853-2857`, `:4745-4749`, `:5270-5274`, `:7146-7150`, `:7590-7594` (≥6 sites).
  **The scan math for NA rows accumulating into / routing through the missing
  bin under `na_as_missing=true` has never been transcribed from
  `feature_histogram.hpp:945-961`.** SPEC-G4-1's Green step is therefore
  genuinely new kernel work (histogram-scan preamble), in addition to the
  `learner.rs:1113-1121` pre-check gate SPEC already names. Scope the mandatory
  Green target to `find_best_split_cpu_native` (production) +
  `find_best_split_f64_on` (kernel-parity), mirroring F-3's reasoning; leave the
  fused/batched/resident sites rejecting (on-device grow loop, out of scope).
  `[VERIFIED: LOCAL grep split.rs "na_as_missing (NA_AS_MISSING forward branch)"]`

None of F-1..F-6 contradicts a SPEC decision; they sharpen SPEC §3's
dependency list and justify why T-G4-1 and T-G5-1/2/3 below are scoped more
narrowly (two call sites, not "the kernel") and sized larger (real algorithm
work, not just gate removal) than the superseded PLAN implied.

---

## Global preconditions (do once, before any Green step)

- **P-0 Confirm dependencies (AGENTS.md rule).** No new external crate is
  required for G1/G2/G4/G5 (SPEC §3; confirmed no `serde_json` in any
  `Cargo.toml` this session's ancestor research). Do NOT add `serde_json`
  (DEC-1 — hand-emit JSON via `format_g17`/`format_g6`,
  `crates/lgbm-model/src/format.rs:43,52`). **Record this confirmation in the
  commit message** for every task below (AGENTS.md rule 2/3).
- **P-1 Checkout the C++ reference.** `LightGBM/` (4.6) is **absent in this
  sandbox** (`research.md` §1 — confirmed, `find`/`ls` return nothing). It is a
  **hard blocker** for the Green step of every G4 task and every G5 sub-task
  except the "wire existing gain.rs math" half of T-G5-3 (F-4). Verify after
  checkout: `ls LightGBM/src/treelearner/feature_histogram.hpp
  LightGBM/src/treelearner/serial_tree_learner.cpp LightGBM/src/io/tree.cpp
  LightGBM/src/io/config_auto.cpp`. Run
  `cargo test -p oracle-harness --test config_drift` FIRST — it mechanically
  diffs `IN_SCOPE_PARAMS` against `LightGBM/src/io/config_auto.cpp` and only
  passes meaningfully with the tree present (SPEC §3, research §5).
- **P-2 Oracle env.** Install `lightgbm==4.6.0` via the uv wheel (memory
  `cpp-linear-tree-oracle.md`) for golden generation
  (`.dump_model()`, `convert_model`, NaN-corpus training,
  penalty/max_delta_step/path_smooth training runs).
- **P-3 Test discipline.** Run treelearner/on-device suites with
  `LGBM_CUDA_ON_DEVICE=0` for G4/G5 (research §5/§8) to avoid the on-device path
  masking a serial-path regression. All new `*_parity.rs` tests SKIP-gracefully
  when a golden is absent (idiom: `crates/oracle-harness/tests/predict_parity.rs:36-49`
  `read_golden`).
- **P-4 (this session, F-3/F-6) Scope the Green target to the two host call
  sites.** For G4 and G5, the mandatory Green targets are
  `crates/lgbm-compute/src/kernels/split.rs::find_best_split_cpu_native`
  (`:7129`, `CpuBackend`'s production path) and `::find_best_split_f64_on`
  (`:493`, the cubecl-cpu kernel-parity/ROCm-mirror path). Do NOT attempt to
  also unwind the fused/batched/staged/resident-reduce rejection sites — those
  serve the out-of-scope on-device grow loop (SPEC §2 non-goals) and MUST keep
  rejecting until that loop is separately in scope.
  **Explicit sub-decision (checker Issue 5):** `find_best_split_cpu_native_2lane`
  (`split.rs:7573`, reached only when the opt-in `LGBM_SPLIT_2LANE=1` env toggle
  is set, dispatch at `lib.rs:2443`; gate at `:7620`) **stays rejecting by
  design** — it is an off-by-default A/B perf variant, not the default CPU path.
  A user combining `LGBM_SPLIT_2LANE=1` with `na_as_missing`/`path_smooth`/
  `max_delta_step` continues to get a typed error after G4/G5 land; this is
  intentional and must be documented in the G4/G5 commit messages, not silently
  omitted. `[VERIFIED: LOCAL PLAN-CHECK Issue 5; lib.rs:2443, split.rs:7573,7620]`

Validation commands referenced below (all repo-verified this session; run from
the workspace root):
```bash
cargo test -p lgbm-model                                    # G1/G2 unit specs
cargo test -p oracle-harness --test json_dump_parity        # G2-4 (new)
cargo test -p oracle-harness --test ifelse_codegen_parity   # G1-4 (new)
LGBM_CUDA_ON_DEVICE=0 cargo test -p lgbm-treelearner         # G4 unit
LGBM_CUDA_ON_DEVICE=0 cargo test -p oracle-harness --test na_missing_parity   # G4-4 (new)
cargo test -p lgbm-compute                                   # G5 unit (split.rs, gain.rs)
LGBM_CUDA_ON_DEVICE=0 cargo test -p oracle-harness --test gain_params_parity  # G5-4 (new)
cargo test -p oracle-harness --test config_drift             # P-1 check (needs LightGBM/)
cargo test -p lgbm                                            # facade unit tests (G1-3/G2-3)
```

---

## Wave A — G2 Model JSON dump (self-contained; do first)

### T-G2-1 — `tree_to_json` · SPEC-G2-1
- **status:** pending · **order:** 1 · **depends_on:** none
- **prereqs:** P-0.
- **files:** Create `crates/lgbm-model/src/json.rs`; Modify
  `crates/lgbm-model/src/lib.rs` (add `pub mod json;` after the existing
  `pub mod format;` at line 24, alongside the other 7 `pub mod` lines,
  `lib.rs:22-28`).
- **CodeGraph evidence:** `format_g17`/`format_g6` at
  `crates/lgbm-model/src/format.rs:43,52` (7/6 existing callers in `tree.rs`,
  no `serde`/`ryu` usage anywhere in the crate). `Tree` struct fields
  (`tree.rs:94-155`): `num_leaves: i32`, `num_cat: i32`,
  `left_child/right_child: Vec<i32>`, `split_feature: Vec<i32>`,
  `threshold: Vec<f64>`, `decision_type: Vec<i8>`, `split_gain: Vec<f32>`,
  `leaf_value/leaf_weight: Vec<f64>`, `leaf_count: Vec<i32>`,
  `internal_value/internal_weight: Vec<f64>`, `internal_count: Vec<i32>`,
  `cat_boundaries: Vec<i32>`, `cat_threshold: Vec<u32>`. `decision_type` mask
  constants (`tree.rs:46-53`): `CATEGORICAL_MASK = 1` (bit0),
  `DEFAULT_LEFT_MASK = 2` (bit1), `get_missing_type = (decision_type >> 2) & 3`
  (`tree.rs:164-168`, bits2-3), `MISSING_ZERO = 1`, `MISSING_NAN = 2`. Pre-order
  recursion precedent: `Tree::get_leaf`/`decision` (`tree.rs:189-273`) always
  descend node 0 first via `split_feature[node]`/`left_child`/`right_child`.
- **Red:** add `#[cfg(test)] fn tree_to_json_numeric_root()` in `json.rs` —
  build a minimal `Tree` (one numeric root split at node 0, two leaves; the
  `split()` helper at `tree.rs:614` or a hand-built struct literal), call
  `json::tree_to_json(&t)`, assert (a) the string is syntactically balanced
  JSON (brace/bracket count check — no serde available, DEC-1), (b) it contains
  `"threshold":<format_g17 output>` for the root's threshold value byte-exact,
  (c) it contains nested `"left_child":{...}` / `"right_child":{...}` objects,
  (d) leaf objects carry `"leaf_value"` via `format_g17`. Expected initial
  failure: **compile error** — `json` module / `tree_to_json` fn does not
  exist.
- **Green:** implement `pub fn tree_to_json(tree: &Tree) -> String` — recurse
  pre-order from node 0; for an internal node emit `split_index`,
  `split_feature`, `split_gain` (via `format_g6` — split_gain is `f32`, mirror
  the C++ `%g` precision used for gain elsewhere,
  cf. `join_f32_g6`-style helper at `tree.rs:1323`), `threshold` (via
  `format_g17`), `decision_type` (decoded per the masks above),
  `default_left`, `missing_type`, `internal_value`/`internal_weight` (via
  `format_g17`), `internal_count`, `left_child`/`right_child` (nested
  objects); for a leaf emit `leaf_index`, `leaf_value`/`leaf_weight` (via
  `format_g17`), `leaf_count`. Reconcile exact key set/order against
  `LightGBM/src/io/tree.cpp` `Tree::NodeToJSON` (P-1) before calling this
  Green step "done" — do not guess key names beyond what SPEC-G2-1 already
  lists.
- **Refactor:** extract a small `push_kv`/`push_kv_f64g17` helper to avoid
  repeating `format!("\"{k}\":{v},")` at each call site; keep `%g` the only
  float-emission path in the module (grep guard: no `to_string()`/`ryu` on any
  `f64`/`f32` in `json.rs`).
- **Verify:** `cargo test -p lgbm-model json::` (focused); then
  `cargo test -p lgbm-model` (module regression).
- **completion evidence:** the Red test is green; `SPEC-G2-1`'s acceptance
  scenario ("nests `left_child`/`right_child` recursively, floats byte-
  identical to C++ `%g`") is satisfied for a numeric-split tree; categorical
  handling is deferred to a follow-up unit test in the same task if P-1
  resolves the "unresolved" note in SPEC-G2-1 before this task lands —
  otherwise leave a `// TODO(P-1)` comment, not a silent gap.
- **rollback:** delete `json.rs` + the `pub mod json;` line — fully additive,
  no other file depends on it yet.
- **parallelization:** none (first task in the dependency chain for G2/G1).

### T-G2-2 — `dump_model` ensemble document · SPEC-G2-2
- **status:** pending · **order:** 2 · **depends_on:** T-G2-1
- **files:** Modify `crates/lgbm-model/src/json.rs`.
- **CodeGraph evidence:** `GbdtModel` fields (`crates/lgbm-model/src/ensemble.rs:35-61`):
  `trees: Vec<Tree>`, `num_class: i32`, `num_tree_per_iteration: i32`,
  `label_index: i32`, `max_feature_idx: i32`, `average_output: bool`,
  `objective_string: Option<String>`, `feature_names: String`,
  `feature_infos: String`, `monotone_constraints: Option<String>`,
  `trailer: Option<String>`. Field-presence precedent:
  `model_text::save`/`save_with_importance` (`crates/lgbm-model/src/model_text.rs:216-260`)
  is the template for "emit only when present" — confirmed at the struct level
  that `average_output`/`monotone_constraints` are `bool`/`Option<String>`
  respectively, matching SPEC-G2-2's "only when set" rule.
- **Red:** `#[cfg(test)] fn dump_model_two_tree_binary()` — build a 2-tree
  `GbdtModel` (reuse a `model()`-style test builder like
  `crates/lgbm-model/src/predict.rs:944-958`'s pattern, adapted locally), call
  `json::dump_model(&m)`, assert top-level keys `"name"`, `"version"`,
  `"num_class"`, `"num_tree_per_iteration"`, `"label_index"`,
  `"max_feature_idx"`, `"objective"`, `"feature_names"`, `"feature_infos"`,
  `"tree_info"` are present as substrings, and `"tree_info"` contains exactly 2
  `"tree_structure"` occurrences (one per tree via T-G2-1). Expected initial
  failure: `dump_model` does not exist (compile error).
- **Green:** implement `pub fn dump_model(model: &GbdtModel) -> String`
  emitting the top-level object; `"average_output"` only when `true`,
  `"monotone_constraints"` only when `Some`; `"tree_info"` is a JSON array of
  `{tree_index, num_cat, shrinkage, tree_structure}` per tree via
  `tree_to_json`. Reconcile the exact top-level key set/order against C++
  `GBDT::DumpModel` (P-1).
- **Refactor:** share the `feature_names`/`feature_infos` string-to-array
  splitting logic with `model_text.rs` ONLY if a zero-risk extraction exists;
  do not force an abstraction across crates' private helpers.
- **Verify:** `cargo test -p lgbm-model json::`.
- **completion evidence:** Red green; SPEC-G2-2 field-presence invariant
  covered by an additional assertion (`average_output=false` model's JSON does
  NOT contain `"average_output"`).
- **rollback:** revert `json.rs` to the T-G2-1 state.
- **parallelization:** none (depends on T-G2-1's `tree_to_json`).

### T-G2-3 — `Booster::dump_model` facade · SPEC-G2-3
- **status:** pending · **order:** 3 · **depends_on:** T-G2-2
- **files:** Modify `crates/lgbm/src/booster.rs` (add near
  `model_to_string`, `booster.rs:725-730`).
- **CodeGraph evidence:** `Booster` struct (`booster.rs:594-612`), private
  `model: GbdtModel` field with public accessor `pub fn model(&self) -> &GbdtModel`
  (`booster.rs:614-618`, 24 callers). `model_to_string` precedent
  (`booster.rs:728-730`): `pub fn model_to_string(&self) -> String {
  lgbm_model::model_text::save(&self.model) }` — the exact pattern to mirror.
- **Red:** `#[cfg(test)] fn dump_model_matches_module()` in `booster.rs`'s test
  module — train or load a tiny booster (reuse the existing `trained_spine()`
  helper, `booster.rs` test module), assert
  `b.dump_model() == lgbm_model::json::dump_model(b.model())`. Expected initial
  failure: `dump_model` method does not exist (compile error).
- **Green:** `pub fn dump_model(&self) -> String { lgbm_model::json::dump_model(&self.model) }`
  (or via the `self.model()` accessor — either is byte-identical since both
  reach the same private field; use `&self.model` directly since this code
  lives inside the `impl Booster` block, matching `model_to_string`'s style).
- **Refactor:** none beyond a doc comment mirroring `model_to_string`'s.
- **Verify:** `cargo test -p lgbm booster::`.
- **completion evidence:** facade test green; AS-1's `Booster::dump_model()`
  entry point exists and is pure orchestration (no recomputation — same string
  as calling the module function directly).
- **rollback:** delete the new method — additive, no breaking change (SPEC §8).
- **parallelization:** none (depends on T-G2-2).

### T-G2-4 — JSON parity vs `lib_lightgbm` 4.6 · SPEC-G2-4
- **status:** pending · **order:** 4 · **depends_on:** T-G2-3, P-2
- **files:** Create `crates/oracle-harness/tests/json_dump_parity.rs`; Create
  golden fixture(s) under
  `crates/oracle-harness/tests/fixtures/json_dump/` (new directory, following
  the `predict_modes/`, `advanced/`, `boosting/` sibling layout,
  `tests/fixtures/` listing this session); Modify `xtask/src/main.rs` to add a
  `json-dump-oracle-capture` subcommand (mirrors the ~11 existing
  `*-oracle-capture` subcommands, e.g. `predict-mode-oracle-capture`,
  `xtask/src/main.rs:279-301`).
- **CodeGraph evidence:** golden/SKIP idiom verbatim from
  `crates/oracle-harness/tests/predict_parity.rs:26-49` (`predict_modes_dir()` +
  `read_golden` returning `Option<String>`, `eprintln!` SKIP message, `CARGO_MANIFEST_DIR`-
  rooted path — NEVER the untracked `LightGBM/` tree).
- **Red:** write `json_dump_parity.rs` using the same idiom (a
  `json_dump_dir()` helper + `read_golden`); with no golden committed yet the
  test **SKIPs (still passes)** — this is the correct Red state for a
  golden-driven parity test (per SPEC-G2-4 "acceptance test: passes with a
  golden present; builds+passes (SKIP) without one"). The FALSIFYING Red step
  is: after P-2, generate the golden via the new xtask subcommand and observe
  the test FAIL if `dump_model`'s output diverges from the captured
  `.dump_model()` string.
- **Green:** run `cargo run -p xtask -- json-dump-oracle-capture` (implemented
  as part of this task) to write the golden from real `lightgbm==4.6.0`
  `.dump_model()` on a captured 2-tree binary model; reconcile any `%g`/key-
  order/field-presence diffs in `json.rs` (T-G2-1/T-G2-2) until byte-exact.
- **Refactor:** fold shared golden-loading boilerplate into
  `oracle_harness::comparator` only if an equivalent helper does not already
  exist there for string-equality goldens.
- **Verify:** `cargo test -p oracle-harness --test json_dump_parity` (present →
  passes; absent → SKIP, non-fatal).
- **completion evidence:** AS-1 met — `Booster::dump_model()` byte-equal to
  the 4.6 golden with a golden present.
- **rollback:** delete the test file + fixture dir + xtask subcommand — no
  effect on `json.rs`/`booster.rs` (those stay from T-G2-1..3).
- **parallelization:** the xtask-subcommand authoring and the parity-test
  skeleton can be written in parallel with Wave B (no shared file), but the
  golden-reconciliation half of Green strictly needs T-G2-1/2/3 complete.

---

## Wave B — G1 If-else codegen (shares tree-walk/`%g` substrate with G2)

### T-G1-1 — `tree_to_cpp` fragment · SPEC-G1-1
- **status:** pending · **order:** 5 · **depends_on:** none (soft reuse of
  T-G2-1's `decision_type` decode helper if extracted; does not block on it)
- **files:** Create `crates/lgbm-model/src/codegen_cpp.rs`; Modify
  `crates/lgbm-model/src/lib.rs` (add `pub mod codegen_cpp;`).
- **CodeGraph evidence:** `Tree::numerical_decision` (`tree.rs:190-212`) is the
  AUTHORITATIVE comparison-direction + missing/default-left source the
  generated C++ must mirror exactly: `if fval.is_nan() && missing_type !=
  MISSING_NAN { fval = 0.0 }`; then `if (missing_type==Zero && is_zero(fval)) ||
  (missing_type==NaN && fval.is_nan()) { route via DEFAULT_LEFT_MASK }`; else
  `if fval <= threshold[node] { left } else { right }`.
  `Tree::categorical_decision` (`tree.rs:214-235`) is the categorical mirror
  (`find_in_bitset` over `cat_threshold[lo..hi]`, `cat_boundaries`). `Tree::predict`
  (`tree.rs:279-290`) confirms leaf dispatch order (`is_linear` branch excluded
  — SPEC-G1 does not scope linear-tree codegen; out of scope per SPEC §2, no
  mention of linear in G1's typed contract).
- **Red:** `#[cfg(test)] fn tree_to_cpp_numeric()` — build a one-split numeric
  tree, call `codegen_cpp::tree_to_cpp(&t, 0)`, assert the output contains an
  `if (` / `<=` comparison against a `format_g17`-formatted threshold literal,
  and returns the `format_g17`-formatted `leaf_value` on both branches.
  Expected initial failure: compile error (`codegen_cpp` module absent).
- **Green:** implement `pub fn tree_to_cpp(tree: &Tree, tree_index: usize) ->
  String` — pre-order nested if-else emission matching
  `Tree::numerical_decision`'s comparison direction and missing/default-left
  handling exactly (byte-for-byte semantic mirror, not just structurally
  similar), plus a categorical branch using a bitset-membership check
  equivalent to `find_in_bitset`. Reconcile the emitted function
  signature/prototype and branch syntax against C++
  `Tree::NodeToIfElse`/`gbdt_model_text.cpp` `SaveModelToIfElse` (P-1) before
  declaring Green complete.
- **Refactor:** if T-G2-1 already extracted a `decode_decision_type` helper
  usable from both `json.rs` and `codegen_cpp.rs`, promote it to a small
  crate-private shared module (e.g. `crate::decision_decode`) — do this ONLY if
  it is a pure, zero-behavior-change extraction; otherwise duplicate the ~10
  lines rather than force a premature abstraction.
- **Verify:** `cargo test -p lgbm-model codegen_cpp::`.
- **completion evidence:** Red green; branch structure + `%g` thresholds
  verified against SPEC-G1-1's acceptance note.
- **rollback:** delete `codegen_cpp.rs` + its `pub mod` line.
- **parallelization:** can run in parallel with Wave A after T-G2-1 lands (soft
  dependency only, for helper reuse — not a hard blocker).

### T-G1-2 — `model_to_cpp` full source · SPEC-G1-2
- **status:** pending · **order:** 6 · **depends_on:** T-G1-1
- **files:** Modify `crates/lgbm-model/src/codegen_cpp.rs`.
- **CodeGraph evidence:** `GbdtModel` fields as in T-G2-2
  (`ensemble.rs:35-61`) — `num_tree_per_iteration`, `trees`, `objective_string`
  drive the per-class stride and function naming.
- **Red:** `#[cfg(test)] fn model_to_cpp_two_tree()` — build a 2-tree
  `GbdtModel`, call `codegen_cpp::model_to_cpp(&m)`, assert the output contains
  two per-tree function definitions (one per T-G1-1 emission) and a
  `PredictRaw`-equivalent summation function that references both. Expected
  initial failure: `model_to_cpp` absent (compile error).
- **Green:** emit include-guard/headers, one function per tree
  (`tree_to_cpp` per index), and a summation function over
  `num_tree_per_iteration`-strided classes × `shrinkage`, mirroring C++
  `GBDT::ModelToIfElse`/`SaveModelToIfElse` (P-1) for the overall file
  skeleton and prototype names.
- **Refactor:** none beyond dedup with T-G1-1's per-tree emission call.
- **Verify:** `cargo test -p lgbm-model codegen_cpp::`.
- **completion evidence:** Red green; SPEC-G1-2's "one function per tree + a
  `PredictRaw` summation" contract satisfied structurally (byte-exactness is
  T-G1-4's job).
- **rollback:** revert `codegen_cpp.rs` to the T-G1-1 state.
- **parallelization:** none (depends on T-G1-1).

### T-G1-3 — `Booster::model_to_cpp` facade · SPEC-G1-3 / DEC-2
- **status:** pending · **order:** 7 · **depends_on:** T-G1-2
- **files:** Modify `crates/lgbm/src/booster.rs`.
- **CodeGraph evidence:** same `model_to_string` precedent as T-G2-3
  (`booster.rs:728-730`).
- **Red:** `#[cfg(test)] fn model_to_cpp_matches_module()` — assert
  `b.model_to_cpp() == lgbm_model::codegen_cpp::model_to_cpp(b.model())`.
  Expected initial failure: method absent (compile error).
- **Green:** `pub fn model_to_cpp(&self) -> String { lgbm_model::codegen_cpp::model_to_cpp(&self.model) }`
  — **no file-writing side effect** (DEC-2 — this deliberately does NOT
  implement `Config.convert_model`'s file-path semantics in v1).
- **Refactor:** doc comment only.
- **Verify:** `cargo test -p lgbm booster::`.
- **completion evidence:** facade green; AS-2's entry point exists.
- **rollback:** delete the method — additive (SPEC §8).
- **parallelization:** none (depends on T-G1-2).

### T-G1-4 — If-else parity vs `lib_lightgbm` 4.6 · SPEC-G1-4
- **status:** pending · **order:** 8 · **depends_on:** T-G1-3, P-2
- **files:** Create `crates/oracle-harness/tests/ifelse_codegen_parity.rs` +
  golden `.cpp` fixture under
  `crates/oracle-harness/tests/fixtures/ifelse_codegen/`; Modify
  `xtask/src/main.rs` to add an `ifelse-codegen-oracle-capture` subcommand
  (same pattern as T-G2-4's `json-dump-oracle-capture`).
- **Red:** SKIP-graceful parity test (same idiom as T-G2-4); passes (SKIP)
  before a golden exists.
- **Green:** capture the golden via `lightgbm==4.6.0` `convert_model`
  (`Config.convert_model`/`convert_model_language`, `config/mod.rs:262-263`,
  default `"gbdt_prediction.cpp"`) on the same 2-tree model used for T-G2-4 (or
  a model reused from an existing fixture dir) via the new xtask subcommand;
  reconcile byte-diffs in `codegen_cpp.rs` until exact.
- **Refactor:** none beyond what T-G2-4 already established for the golden
  loader.
- **Verify:** `cargo test -p oracle-harness --test ifelse_codegen_parity`.
- **completion evidence:** AS-2 met.
- **rollback:** delete the test/fixture/xtask subcommand only.
- **parallelization:** independent of Wave A's T-G2-4 (different fixture dir,
  different xtask subcommand) — may run concurrently with it once both P-2 is
  satisfied and each wave's facade (T-G2-3 / T-G1-3) is done.

---

## Wave C — G4 NA-as-missing serial forward routing (hot path; do NOT run
## concurrently with Wave D)

> **P-1 mandatory before every Green step in this wave.** Per F-6, this is
> real missing-kernel-algorithm work (the NA_AS_MISSING forward-branch
> preamble, `feature_histogram.hpp:945-961`, has never been transcribed), not
> merely gate removal. Resolve SPEC-G4-1/2's `TBD` (missing-bin index
> convention, default-branch direction rule) from the checked-out C++ source
> FIRST.

### T-G4-1 — NA rows accumulate into the missing bin during histogram build +
### the split-kernel scan admits `na_as_missing=true` · SPEC-G4-1
- **status:** pending · **order:** 9 · **depends_on:** none (first in wave)
- **files:** Modify `crates/lgbm-compute/src/kernels/split.rs`
  (`find_best_split_cpu_native` at `:7129`, gate at `:7146-7150`; and
  `find_best_split_f64_on` at `:493`, gate at `:514-517` — per P-4/F-3/F-6, the
  two mandatory host targets). Modify
  `crates/lgbm-treelearner/src/learner.rs`: (a) confirm the serial
  histogram-build call site passes `na_as_missing=true` through to the backend
  (`FeatureColumn::na_as_missing()`, `learner.rs:172-176`, already computes the
  flag — verify in Red it reaches `find_best_split`'s `na_as_missing`
  parameter); **and (b) — checker Issue 2 (BLOCKER for correctness) — fix
  `FeatureColumn::run_forward()` (`learner.rs:186-188`).** Today
  `run_forward() == (num_bin>2 && missing_type==Zero)` returns **`false` for a
  NaN feature**, and its own doc states "the deferred NaN case is a typed error
  before this is reached." The learner passes `run_forward = f.run_forward()`
  into the scan (`learner.rs:3032`); with `run_forward=false` the FORWARD pass
  in `find_best_split_cpu_native` is skipped (`fwd_count=0` when `!run_forward`,
  `split.rs:7199-7203`). C++ NA_AS_MISSING requires forward/default-branch
  handling for the missing bin — so removing the gate WITHOUT also making
  `run_forward()` (or an NA-specific dispatch flag) true for NaN features makes
  them scan REVERSE-only and pick the wrong split. Resolve the exact
  forward-dispatch rule for NaN features from `feature_histogram.hpp` at P-1.
  `[VERIFIED: LOCAL PLAN-CHECK Issue 2; learner.rs:186-188,3032; split.rs:7199-7203]`
- **CodeGraph evidence:** `FeatureColumn::na_as_missing()` (`learner.rs:172-176`):
  `self.num_bin > 2 && self.missing_type == MissingType::NaN` — the
  authoritative flag, already computed. `BinMapper` NaN routing precedent
  (`crates/lgbm-dataset/src/bin_mapper.rs:1091-1106`, test
  `value_to_bin_nan_routing_both_missing_types`): `missing==NaN` routes NaN to
  `num_bin_ - 1` (the TOP bin) — this is the existing, already-correct BINNING
  behavior; the gap is specifically in the split-finding SCAN's handling of
  that top bin under the NA_AS_MISSING forward branch, not in binning itself.
  The rejection sites to remove/replace:
  `split.rs:514-517` (`find_best_split_f64_on`) and `split.rs:7146-7150`
  (`find_best_split_cpu_native`).
- **Red:** add a unit test in `crates/lgbm-compute/src/kernels/split.rs`'s test
  module — construct a small histogram (4 bins, top bin = the NaN sentinel
  bin) with hand-computed (Σg, Σh) per bin, call `find_best_split_cpu_native`
  with `na_as_missing=true`, and assert it returns `Ok(SplitInfo{..})` (not
  `Err`) with the missing-bin's (Σg, Σh) correctly folded per the P-1-verified
  rule (exact expected numbers computed from the C++ source, NOT guessed).
  Expected initial failure: `find_best_split_cpu_native` returns
  `Err(ComputeError::Runtime{ detail: "...NA_AS_MISSING forward branch not yet
  implemented" })` (today's behavior, `split.rs:7146-7150`) — an `unwrap()` on
  `Err` panics, which is the correct falsifying Red failure. **(checker Issue 2)
  The test MUST be constructed so the correct split comes from the
  FORWARD/default-branch handling of the missing bin** (i.e. call with
  `run_forward=true` and a bin layout where a reverse-only scan would pick a
  different, wrong threshold) — asserting merely `Ok` is insufficient to catch
  the reverse-only-scan bug; assert the returned threshold + `default_left`
  match the P-1-verified forward-branch expectation.
- **Green:** implement the NA_AS_MISSING forward-branch preamble in BOTH
  `find_best_split_cpu_native` and `find_best_split_f64_on`: fold the missing
  bin's (Σg, Σh) into the forward/reverse scan per `feature_histogram.hpp:945-961`
  (P-1-verified), matching the CPU f64-fold **bit-exact** contract (identical
  accumulation ORDER to the non-NA path — do not reorder float additions).
  Remove the two `if na_as_missing { return Err(...) }` gates at those two
  sites ONLY (leave the ~4-5 remaining fused/batched/resident sites rejecting
  per P-4).
- **Refactor:** factor the missing-bin fold into a small shared helper called
  from both `find_best_split_cpu_native` and `find_best_split_f64_on` if the
  two implementations would otherwise duplicate >10 lines identically;
  otherwise keep them independently correct (mirrors the existing
  independent-body pattern across this file's many variants).
- **Verify:** `cargo test -p lgbm-compute` (focused: the new unit test +
  regression on `find_best_split_na_as_missing_is_typed_error`
  (`split.rs:7867-7911`) — that test's NAME/assertion must be UPDATED to
  reflect the new "admits and computes" behavior instead of "must be a typed
  error," per SPEC-G4-3's "prior 'rejects' unit test is updated to assert
  success" — but do that update in T-G4-3, not here, to keep this task's Red/
  Green/Refactor cycle singly-focused on the histogram-fold behavior).
- **completion evidence:** the new histogram-fold unit test is green; the
  pre-existing "is_typed_error" test is UNCHANGED and still passing at the end
  of this task (it will be updated in T-G4-3) — i.e. this task adds a NEW
  code path without yet removing the old rejection's test coverage.
- **rollback:** revert the two call sites' gate + scan changes; the pre-check
  gate at `learner.rs:1113` (untouched by this task) continues to prevent any
  caller from reaching the new code, so this task alone is safely revertible.
- **parallelization:** none within the wave (first task).

### T-G4-2 — NA routed down default branch on partition · SPEC-G4-2
- **status:** pending · **order:** 10 · **depends_on:** T-G4-1
- **files:** Modify `crates/lgbm-treelearner/src/learner.rs` (the serial
  `data_partition`/`Split` routing — locate via CodeGraph the concrete
  partition function called after a winning `SplitInfo` is selected; SPEC §7
  cites "serial histogram/partition" generically — **[UNVERIFIED exact
  function name/line]**, resolve via `codegraph_explore "data_partition split
  apply learner.rs"` at the start of this task before editing).
- **CodeGraph evidence:** `SplitInfo::default_left` (`gain.rs:477-479`) is
  already the authoritative "which side NA/most-freq-bin rows go" bit
  (`bool default_left — true iff the winner came from the REVERSE branch`).
  `decision_type`'s `DEFAULT_LEFT_MASK` bit (`tree.rs:48`) is how the tree
  model records this. **This CONFIRMS SPEC-G4-2's `TBD` "default-direction
  selection rule" is already represented in the `SplitInfo` type** — the gap
  is specifically that the PARTITION step's row-assignment loop does not yet
  branch NA rows through it for `na_as_missing` features (needs P-1
  confirmation that the assignment rule is literally "NA row → whichever
  child `default_left` points to," not something more specific to the missing
  bin's position).
- **Red:** a partition-level test — construct a leaf with a mix of NA and
  non-NA rows on a `na_as_missing` feature, apply the winning split from
  T-G4-1's histogram, and assert every NA row lands in the
  `default_left`-selected child (row-for-row against a hand-computed
  expectation, P-1-verified). Expected initial failure: NA rows are currently
  either unreachable (gate still blocks upstream) or mis-routed if the gate is
  bypassed directly at this layer — assert `Err`/panic or wrong-child assignment
  as the concrete pre-fix behavior observed when writing the Red test.
- **Green:** implement the default-direction routing in the partition step,
  reading `default_left`/`missing_type` per the P-1-verified rule (mirroring
  `Tree::numerical_decision`'s runtime routing rule, `tree.rs:193-212`, applied
  at TRAIN time to the histogram-bin representation rather than a raw `f64`
  feature value).
- **Refactor:** none beyond removing any now-dead placeholder from T-G4-1's
  partial wiring, if any.
- **Verify:** `LGBM_CUDA_ON_DEVICE=0 cargo test -p lgbm-treelearner`.
- **completion evidence:** Red green; SPEC-G4-2's acceptance scenario ("each
  NA row lands in the same child as C++") is unit-covered (parity-level
  confirmation is T-G4-4).
- **rollback:** revert the partition-routing change; T-G4-1's scan-level fold
  is unaffected (independently revertible).
- **parallelization:** none (depends on T-G4-1's SplitInfo output).

### T-G4-3 — Remove the typed-error gate · SPEC-G4-3
- **status:** pending · **order:** 11 · **depends_on:** T-G4-1, T-G4-2
- **files:** Modify `crates/lgbm-treelearner/src/learner.rs:1110-1121`
  (the `if f.na_as_missing() { return Err(TreeLearnerError::Compute(...)) }`
  block inside the `bins_validated` pre-check loop).
- **CodeGraph evidence:** exact gate text confirmed this session
  (`learner.rs:1110-1121`): guarded by `if !self.bins_validated { for f in
  features.iter() { ... if f.na_as_missing() { return Err(...) } } }` — i.e.
  the gate is memoized per distinct feature set (only checked once). The
  existing regression test asserting this rejection:
  `learner.rs:4978-4989` (`// (d) na_as_missing feature is the deferred typed
  error` / `.expect_err("na_as_missing is deferred -> typed error")`,
  `assert!(matches!(err4, TreeLearnerError::Compute(_)))`) — this is the test
  SPEC-G4-3 says to UPDATE.
- **Red:** update `learner.rs`'s existing test at `:4978-4989` to instead
  **assert training SUCCEEDS** on the same NaN corpus (drop the
  `.expect_err(...)`, assert `.is_ok()` or a concrete trained-model shape
  instead). Run it BEFORE removing the gate — expected initial failure: the
  test still gets `Err(TreeLearnerError::Compute(...))` (gate still present),
  so the new "must be Ok" assertion fails.
  This is a case where the Red test is a MODIFICATION of an existing test, not
  a new one — the modification IS the falsifying change (SPEC-TDD rule:
  migrations/public-contract changes must be tracked as their own cycle, which
  this is).
- **Green:** remove the `if f.na_as_missing() { return Err(...) }` block
  (`learner.rs:1110-1121`); training now proceeds to the T-G4-1/T-G4-2 code
  paths for `na_as_missing` features.
- **Refactor:** grep the crate for any OTHER call site that depended on this
  error's presence (e.g. a caller matching on
  `TreeLearnerError::Compute(ComputeError::Runtime{ detail })` and checking for
  the `"na_as_missing"` substring) — none expected per SPEC §8 ("strictly
  widening"), but verify rather than assume.
- **Verify:** `LGBM_CUDA_ON_DEVICE=0 cargo test -p lgbm-treelearner` (full
  crate regression, not just the one test — this removes a defensive gate on a
  hot path).
- **completion evidence:** AS-3's "training on a NaN-bearing
  `use_missing=true` corpus completes (no typed error)" is met at the unit
  level; SPEC-G4-4 confirms parity.
- **rollback:** restore the gate block verbatim — SPEC §8 confirms this widens
  currently-erroring inputs only, so reverting is safe and non-breaking to any
  other passing input.
- **parallelization:** none (depends on both T-G4-1 and T-G4-2 being correct
  first — removing the gate before the routing is correct would let wrong
  results through silently).

### T-G4-4 — NA parity vs `lib_lightgbm` 4.6 · SPEC-G4-4
- **status:** pending · **order:** 12 · **depends_on:** T-G4-3, P-1, P-2
- **files:** Create `crates/oracle-harness/tests/na_missing_parity.rs` +
  golden (bin histograms + predictions) under
  `crates/oracle-harness/tests/fixtures/na_missing/`; Modify
  `xtask/src/main.rs` to add an `na-missing-oracle-capture` subcommand.
- **Red:** SKIP-graceful parity test (same idiom); passes (SKIP) before a
  golden exists.
- **Green:** capture a NaN-bearing `use_missing=true` corpus golden
  (histograms + predictions) from `lightgbm==4.6.0` with
  `deterministic=true force_row_wise=true num_threads=1` (research §5's
  documented capture discipline); reconcile CPU f64-fold node histograms
  bit-exact and predictions within the CPU bit-exact contract.
- **Refactor:** none beyond the shared golden-loader pattern.
- **Verify:** `LGBM_CUDA_ON_DEVICE=0 cargo test -p oracle-harness --test
  na_missing_parity`.
- **completion evidence:** AS-3 fully met (parity, not just "doesn't error").
- **rollback:** delete the test/fixture/xtask subcommand only — does not touch
  `learner.rs`/`split.rs`.
- **parallelization:** none (last task in the wave; requires the full T-G4-1..3
  chain).

---

## Wave D — G5 split-kernel gain params (hot split kernel; after Wave C, NOT
## concurrent with it)

> **P-1 mandatory** for T-G5-1 (penalty source/semantics) and T-G5-2
> (max_delta_step clamp formula, per F-5 — no existing code to reuse). T-G5-3
> (path_smooth) is PARTIALLY unblocked by F-4 (the gain math already exists and
> is tested) but OQ-2 (parent-output availability) still needs P-1 or direct
> Red-step investigation of the call site.

### T-G5-1 — Feature split penalty applied to gain · SPEC-G5-1
- **status:** pending · **order:** 13 · **depends_on:** none (independent of
  T-G5-2/3 per SPEC — but see F-1: this is the largest of the three, budget
  accordingly)
- **MECHANISM (checker Issue 1 — RESOLVED to ONE strategy): apply the penalty
  at the LEARNER per-feature loop as a post-multiply on the winning
  `split.gain`, NOT threaded into the split kernel.** The kernel scans over
  `&[BatchedSplitFeature]` (`split.rs:77`) which carries NEITHER a `penalty`
  NOR a `real_feature_index`, so a kernel-level `penalty: f64` parameter cannot
  express a *per-feature* penalty through the batched/unified wrappers without
  also widening `BatchedSplitFeature` and all three wrappers. The learner
  already post-processes per-feature `split.gain` in exactly the right place —
  CEGB (`split.gain -= delta`) and monotone (`split.gain *= penalty`) at
  `crates/lgbm-treelearner/src/learner.rs:~3113-3146`. Add `split.gain *=
  feature_contri[real_feature_index]` alongside them. This needs **NO trait
  change, NO `BatchedSplitFeature` change, NO kernel edit** — so T-G5-1 does
  NOT touch `split.rs` at all and does NOT collide with T-G5-2/T-G5-3.
- **files:** Modify `crates/lgbm-core/src/config/mod.rs` (add
  `pub feature_contri: Vec<f64>` field + parse/default, mirroring
  `monotone_constraints: Vec<i32>` at `:154,381` as the sibling per-feature
  list pattern). Modify `crates/lgbm-treelearner/src/learner.rs` at the
  per-feature post-processing loop (`:~3113-3146`, where CEGB/monotone already
  adjust `split.gain`) to multiply the winning split's gain by
  `config.feature_contri[real_feature_index]` (default `1.0` when the vector is
  empty or the index is out of range — CONFIRM the exact default/out-of-range
  rule from P-1). `real_feature_index` is available as
  `FeatureColumn::real_feature_index` (`learner.rs:125-127`).
- **CodeGraph evidence (F-1):** `feature_contri` IN_SCOPE
  (`crates/lgbm-core/src/config/scope.rs:99`), aliased from `feature_contrib`/
  `fc`/`fp`/`feature_penalty` (`crates/lgbm-core/src/config/alias.rs:105-108`),
  but **absent from `Config`** (`config/mod.rs` — zero `feature_contri:`
  hits). `FeatureColumn::real_feature_index: i32` (`learner.rs:125-127`,
  "the ORIGINAL feature index... the tree's `split_feature` records") is the
  index to key the lookup on. The hard-coded `penalty = 1.0f64` sites in
  `split.rs`: `:678, :5127, :5509, :5617/5671 (#[cube] kernel), :5768/5812,
  :7334` — per P-4, only the two feeding `find_best_split_cpu_native`
  (`:7334`-region) and `find_best_split_f64_on` (`:678`-region) are mandatory
  Green targets.
- **Red:** (a) a `lgbm-core` config test asserting `Config::default().feature_contri
  == Vec::new()` and that parsing `"feature_contri=0.5,1.0"` (or the
  P-1-verified real parse syntax — C++ list params are typically
  comma-or-space-separated; verify) yields `vec![0.5, 1.0]`. Expected initial
  failure: compile error (no such field). (b) a `lgbm-treelearner`
  integration test that trains a one-split tree on a 2-feature corpus with
  `feature_contri = [0.5, 1.0]` and asserts the recorded `split_gain` for a
  split on feature 0 is exactly half the gain obtained with default
  `feature_contri` (identical data/seed) — mirrors SPEC-G5-1's Given/When/Then
  at the learner level (NOT the kernel). Expected initial failure: with no
  `Config.feature_contri` field the test does not compile; once (a) lands, the
  post-multiply is absent so the gains are equal, not halved → assertion fails.
  Land (a) and (b) as separate Red→Green commits; do not conflate config-parse
  with the gain post-multiply.
- **Green:** (a) add the `Config` field + parser wiring (reuse the existing
  per-feature-list parse helper used for `monotone_constraints`). (b) apply
  `split.gain *= config.feature_contri[real_feature_index]` (default `1.0` when
  empty/out-of-range). **PLACEMENT (checker pass-2 MAJOR — pin precisely, do not
  say merely "beside CEGB/monotone"):**
  - **Ordering vs CEGB/monotone (non-commutative).** CEGB is an *additive*
    subtract (`split.gain -= delta`, `learner.rs:3125`) and monotone is a
    *multiply* (`split.gain *= penalty`, `:3146`). `(G·fc − delta)` ≠
    `(G − delta)·fc` whenever CEGB is active and `fc ≠ 1`, so the insert point
    is bit-exact-sensitive. C++ applies `feature_contri` at the FeatureHistogram
    gain level (before the tree-learner's CEGB/monotone post-processing), so the
    penalty multiply MUST be the FIRST per-feature gain adjustment — inserted at
    the TOP of the loop body (`~:3113`), BEFORE the CEGB subtract at `:3125`.
    **P-1 must confirm** the C++ order (feature_contri relative to CEGB and to
    `min_gain_shift`).
  - **Ordering vs `min_gain_shift`.** `split.gain` as returned by the kernel is
    already `best_gain - min_gain_shift` (`split.rs:442-443` doc). A learner-level
    post-multiply is bit-exact ONLY if C++ multiplies AFTER the `min_gain_shift`
    subtraction. **P-1 must confirm.** If C++ multiplies BEFORE, fold into the
    kernel's per-feature gain instead (the `BatchedSplitFeature`-widening path —
    recorded contingency).
  - **Categorical branch (checker pass-2 MAJOR).** The categorical split branch
    (`learner.rs:2993-3023`) `continue`s at `:3023` and NEVER reaches the numeric
    post-processing block — so the penalty must ALSO be applied inside the
    categorical branch before its `continue` (keyed on the same
    `f.real_feature_index`), OR categorical `feature_contri` must be explicitly
    scoped out with P-1 evidence that C++ does not penalize categorical gains
    (do not silently skip it). Default assumption: apply in BOTH branches.
  `[VERIFIED: LOCAL PLAN-CHECK pass-2; learner.rs:2993-3023,3113,3125,3146]`
- **Refactor:** none needed — the post-multiply sits beside the existing
  CEGB/monotone `split.gain` adjustments and reuses their structure; no trait
  or kernel change to unwind.
- **Verify:** `cargo test -p lgbm-core` (config), then
  `LGBM_CUDA_ON_DEVICE=0 cargo test -p lgbm-treelearner` (the gain
  post-multiply + full crate regression). No `lgbm-compute` change in this task.
- **completion evidence:** SPEC-G5-1's Given/When/Then ("penalty=0.5 → exactly
  half the penalty=1.0 gain") passes at the learner level; OQ-1 resolved and
  documented in the commit message (source = `feature_contri` config param, NOT
  CEGB; and the confirmed before/after-`min_gain_shift` ordering).
- **rollback:** revert the `Config` field + the single learner post-multiply
  line together — additive and default-off (empty vector ⇒ penalty 1.0 ⇒ no
  behavior change for any existing corpus). No trait/kernel surface touched.
- **parallelization:** independent of T-G5-2/T-G5-3's kernel work — T-G5-1
  touches `config/mod.rs` + `learner.rs`, and does NOT touch `split.rs`/`gain.rs`
  at all, so it never collides with the G5-2/G5-3 kernel edits. Caveat
  (checker pass-2 MINOR): T-G5-3 ALSO edits `learner.rs` (a different region —
  the `find_best_split` call site, to supply `parent_output`), so T-G5-1 and
  T-G5-3 share that file in disjoint regions (low real-conflict risk, not "no
  shared file"). Land T-G5-1 and T-G5-3 with awareness of the shared file; all
  three still precede T-G5-4.

### T-G5-2 — `max_delta_step` leaf-output clamp · SPEC-G5-2
- **status:** pending · **order:** 14 · **depends_on:** none structurally, but
  land after T-G5-1 (merge-order note above)
- **files:** Modify `crates/lgbm-compute/src/gain.rs` (add a NEW
  `calculate_splitted_leaf_output_clamped`-style function, mirroring the
  existing `calculate_splitted_leaf_output_smoothed` pattern at `:183-197`
  structurally, once the C++ formula is confirmed). Modify
  `crates/lgbm-compute/src/kernels/split.rs`
  (`find_best_split_cpu_native`/`find_best_split_f64_on`, per P-4) to call the
  new clamp function and drop `max_delta_step` from the rejection condition at
  `:553`/`:7176` (leave `path_smooth` in the condition until T-G5-3 lands, or
  land both checks' removal together if T-G5-2/T-G5-3 are sequenced back-to-
  back — either order is acceptable since SPEC marks them independent).
- **CodeGraph evidence (F-5):** confirmed **zero** existing clamp
  implementation in `gain.rs` — this is new formula work, blocked on P-1.
  `GainConfig.max_delta_step: f64` already exists and is already threaded into
  every `find_best_split_*` call (`gain.rs:382-383`, `:425`) — only the FORMULA
  application is missing.
- **Red:** a `gain.rs` unit test `leaf_output_clamped_at_max_delta_step` —
  with `max_delta_step=0.7` and inputs whose unclamped output would exceed
  `0.7` in magnitude, assert the clamped function returns exactly `±0.7`
  (sign-matching the unclamped output), per the P-1-verified C++
  `USE_MAX_OUTPUT=true` formula (`CalculateSplittedLeafOutput`,
  `feature_histogram.hpp`). Expected initial failure: the new function does
  not exist (compile error) — write this test AFTER P-1 confirms the exact
  formula (do not encode a guessed clamp rule into the test).
- **Green:** implement the clamp formula in `gain.rs`; wire it into the two
  mandatory `split.rs` call sites (compute the leaf output via the clamped
  path when `max_delta_step != 0.0`); remove `max_delta_step` from the
  rejection condition at those two sites.
- **Refactor:** keep the f32 hip mirror in step with the f64 version (mirror
  the existing `_f32` suffix convention, `gain.rs:288-321`) if the clamp needs
  to run on the no-f64 hip path too — confirm from P-1 whether `max_delta_step`
  is in scope for the hip f32 path in this milestone (SPEC does not explicitly
  scope ROCm for G5; if ambiguous, implement f64-only and flag the f32 mirror
  as a follow-up, do not silently skip a hip regression).
- **Verify:** `cargo test -p lgbm-compute`.
- **completion evidence:** SPEC-G5-2's Given/When/Then ("max_delta_step=0.7 →
  leaf output clamped exactly as C++") passes; `find_best_split_cpu_native`/
  `find_best_split_f64_on` no longer reject non-default `max_delta_step`.
- **rollback:** revert the new `gain.rs` function + the two call-site wiring
  edits; restore the rejection condition.
- **parallelization:** file-disjoint from T-G5-1 (which now touches only
  `config/mod.rs` + `learner.rs`) → may run concurrently with it. But T-G5-2
  and **T-G5-3 both edit the same two `split.rs` functions**
  (`find_best_split_cpu_native`/`find_best_split_f64_on`) → sequence G5-2 and
  G5-3 relative to each other (recommend G5-2 then G5-3), never concurrent.

### T-G5-3 — `path_smooth` smoothing · SPEC-G5-3
- **status:** pending · **order:** 15 · **depends_on:** none structurally;
  land after T-G5-1/T-G5-2 per the merge-order note
- **TRAIT BLAST RADIUS (checker Issue 3).** `parent_output: f64` is a
  per-LEAF scalar and is the ONE genuine internal-signature change in G5 (G5-1
  no longer touches the trait; G5-2 reuses the existing `GainConfig.max_delta_step`).
  Adding it to `find_best_split_cpu_native`/`find_best_split_f64_on` forces the
  same parameter onto `Backend::find_best_split` (`lib.rs:727`) — **exactly two
  implementors** (`CpuBackend` `lib.rs:2417`, `GpuBackend<R>` `lib.rs:3494`) but
  **~8 in-crate call sites of `self.find_best_split` that must all supply it**:
  the trait default `find_best_splits_batched` (`lib.rs:1029`) + CpuBackend
  override (`:2571`), `build_fix_scan_impl` (`:2871`), `subtract_scan_impl`
  (`:3014`), plus the `kernel_parity.rs` test caller. The batched/unified
  wrappers (`find_best_splits_batched`/`build_fix_scan_impl`/`subtract_scan_impl`)
  therefore ALSO gain a `parent_output` parameter (shareable across the batched
  call since it is per-leaf, not per-feature). Enumerate and update every one;
  gate the task on `cargo build --workspace` (not just `-p lgbm-compute`).
  `[VERIFIED: LOCAL PLAN-CHECK Issue 3; lib.rs:727,1029,2417,2571,2871,3014,3494]`
- **files:** Modify `crates/lgbm-compute/src/kernels/split.rs`
  (`find_best_split_cpu_native`/`find_best_split_f64_on`, per P-4) — thread
  `path_smooth`, `num_data` (already a parameter of both functions), and a NEW
  `parent_output: f64` parameter; switch to
  `gain::calculate_splitted_leaf_output_smoothed`/`get_leaf_gain_smoothed`
  (`gain.rs:183-229`, already implemented per F-4) when `cfg.path_smooth !=
  0.0`; remove `path_smooth` from the rejection condition. Modify
  `crates/lgbm-compute/src/lib.rs` — the `Backend::find_best_split` trait decl
  (`:727`) and the batched wrappers `find_best_splits_batched` (`:1029,:2571`),
  `build_fix_scan_impl` (`:2871`), `subtract_scan_impl` (`:3014`) — add the
  `parent_output` parameter to each and thread it through. Modify
  `crates/lgbm-treelearner/src/learner.rs`'s call site(s) to supply
  `parent_output` — **resolve OQ-2 here**: does the leaf-splitting call site
  already hold the parent leaf's output value (it should, since the parent
  leaf's constant/linear output is computed before it is split) — verify via
  `codegraph_explore` on the call site before writing the Red test.
- **CodeGraph evidence (F-4):** `calculate_splitted_leaf_output_smoothed`
  (`gain.rs:183-197`) and `get_leaf_gain_smoothed` (`gain.rs:208-229`) are
  ALREADY implemented, doc-cited to `cuda_leaf_splits.hpp:74-90,117-121`, and
  covered by a passing test (`smoothing_blend_matches_reference`,
  `gain.rs:577-623`, including a directional-sanity check and an f32-mirror
  consistency check). This task's Green step is wiring, not derivation.
- **Red:** a `split.rs`-level unit test `find_best_split_cpu_native_admits_path_smooth`
  — with `path_smooth=2.0` and a fixed `parent_output`, assert (a) the call
  returns `Ok` (today it returns `Err`), and (b) the returned leaf outputs
  match `gain::calculate_splitted_leaf_output_smoothed`'s output computed
  independently with the same inputs (a cross-check against the already-tested
  gain.rs function, not a fresh hand-derivation). Expected initial failure:
  `Err(ComputeError::Runtime{ detail: "...path_smooth...Phase-7+ scope..." })`
  (today's behavior).
- **Green:** thread `parent_output` into the two mandatory call sites (from
  `learner.rs`, resolving OQ-2), dispatch to the smoothed gain/output
  functions when `path_smooth != 0.0`, remove `path_smooth` from the
  rejection condition.
- **Refactor:** if OQ-2 reveals `parent_output` is NOT trivially available at
  the call site (e.g. it requires computing the parent's unconstrained output
  first, which today may be skipped when a leaf splits immediately), document
  the extra plumbing needed as a follow-up note in the commit message — do not
  silently approximate `parent_output` with a wrong value to make the test
  pass.
- **Verify:** `cargo test -p lgbm-compute`.
- **completion evidence:** SPEC-G5-3's Given/When/Then ("path_smooth=2.0 →
  smoothed leaf output bit-exact vs C++") passes at the unit level (parity
  confirmation is T-G5-4); rejection removed at the two mandatory sites.
- **rollback:** revert the call-site wiring; `gain.rs`'s smoothing functions
  are UNCHANGED by this task (they pre-existed) so no rollback needed there.
- **parallelization:** file-disjoint from T-G5-1 → may run concurrently with
  it. Shares the two `split.rs` functions with T-G5-2 → sequence relative to
  G5-2 (recommend after G5-2), never concurrent. Because T-G5-3 changes the
  `Backend::find_best_split` signature (`parent_output`), land it as one atomic
  workspace-compiling unit (checker Issue 3).

### T-G5-4 — Gain-param parity vs `lib_lightgbm` 4.6 · SPEC-G5-4
- **status:** pending · **order:** 16 · **depends_on:** T-G5-1, T-G5-2, T-G5-3,
  P-2
- **files:** Create `crates/oracle-harness/tests/gain_params_parity.rs` +
  three golden fixtures under
  `crates/oracle-harness/tests/fixtures/gain_params/{penalty,max_delta_step,path_smooth}/`;
  Modify `xtask/src/main.rs` to add a `gain-params-oracle-capture` subcommand.
- **Red:** SKIP-graceful parity test (same idiom); passes (SKIP) before
  goldens exist.
- **Green:** capture three single-param goldens from `lightgbm==4.6.0`
  (`feature_contri` set / `max_delta_step=0.7` / `path_smooth=2.0`, others at
  default, `deterministic=true force_row_wise=true num_threads=1`); reconcile
  split-gain + threshold + leaf-output to CPU f64-fold bit-exact against each.
- **Refactor:** none beyond the shared golden-loader pattern.
- **Verify:** `LGBM_CUDA_ON_DEVICE=0 cargo test -p oracle-harness --test
  gain_params_parity`.
- **completion evidence:** AS-4 met.
- **rollback:** delete the test/fixture/xtask subcommand only.
- **parallelization:** none (last task; requires T-G5-1..3 complete).

---

## Execution order & parallelism

```text
Wave A (G2):  T-G2-1 -> T-G2-2 -> T-G2-3 -> T-G2-4
Wave B (G1):  T-G1-1 -> T-G1-2 -> T-G1-3 -> T-G1-4
Wave C (G4):  T-G4-1 -> T-G4-2 -> T-G4-3 -> T-G4-4
Wave D (G5):  T-G5-1 -\
              T-G5-2 --+-> T-G5-4
              T-G5-3 -/
```

1. **Wave A (G2)** starts immediately after P-0 — no cross-wave dependency.
2. **Wave B (G1)** may start in parallel with Wave A once T-G2-1 lands (soft
   dependency, for `decision_type`-decode helper reuse only — T-G1-1 does not
   hard-block on it).
3. **Wave C (G4)** requires P-1 (checked-out `LightGBM/`) before ANY Green
   step. Do not start Wave C's Green steps until P-1 is satisfied; the Red
   steps (writing failing tests against today's rejecting behavior) may be
   drafted earlier.
4. **Wave D (G5)** requires P-1 for T-G5-1/T-G5-2 (T-G5-3's gain math is
   pre-existing per F-4, but OQ-2 and the rejection-removal still benefit from
   P-1). **Do NOT run Wave C and Wave D concurrently** — both touch
   `find_best_split_cpu_native`/`find_best_split_f64_on` in
   `crates/lgbm-compute/src/kernels/split.rs` (T-G4-1 edits the SAME two
   functions T-G5-1/2/3 edit), so a merge conflict AND a parity-regression
   attribution problem are both real risks if run in parallel. Sequence Wave C
   fully before Wave D (recommended) or fully after — never interleaved.
5. Within Wave D (post checker-revision): **T-G5-1 does not touch the split
   kernel** — with the learner-level penalty mechanism (checker Issue 1) it
   touches `config/mod.rs` + `learner.rs`, so it never collides with the
   G5-2/G5-3 kernel edits. (It shares `learner.rs` with T-G5-3 in a disjoint
   region — the `parent_output` call site — low conflict risk, checker pass-2.)
   **T-G5-2 and T-G5-3 both edit the same two `split.rs` functions**
   (`find_best_split_cpu_native`/`find_best_split_f64_on`) → sequence them
   relative to each other (recommend G5-2 then G5-3), never concurrent. T-G5-3
   additionally changes the `Backend::find_best_split` signature
   (`parent_output`) and must land as one workspace-compiling unit (Issue 3).
   All three still precede T-G5-4.

## Definition of done (per wave)

- **Wave A / Wave B:** all Red tests turned Green; Refactor left behavior
  unchanged; `json_dump_parity.rs` / `ifelse_codegen_parity.rs` pass with a
  committed golden (or SKIP cleanly with a documented "golden absent — P-2 not
  run in this environment" reason); no float in `json.rs`/`codegen_cpp.rs`
  emitted via anything but `format_g17`/`format_g6` (grep guard); commit
  messages record the P-0 dependency confirmation and cite the closed SPEC
  IDs.
- **Wave C:** T-G4-1..3's unit tests green; the CPU f64-fold accumulation
  order for NA rows is bit-identical in structure to the non-NA path (spot-
  checked by the T-G4-1 unit test's hand-computed expectations); the
  pre-existing "na_as_missing is deferred" test is UPDATED (not deleted) to
  assert success; `na_missing_parity.rs` passes with a golden or SKIPs
  cleanly; commit messages cite the P-1 source lines used to resolve each
  `TBD`.
- **Wave D:** T-G5-1..3's unit tests green independently; OQ-1 resolution
  (penalty source = `feature_contri`, applied learner-level per checker Issue 1,
  plus the confirmed before/after-`min_gain_shift` ordering) and OQ-2 resolution
  (parent-output availability) are both recorded in commit messages; the ONE
  `Backend::find_best_split` trait signature change (T-G5-3's `parent_output`)
  is applied consistently across both implementors AND all ~8 in-crate
  `self.find_best_split` call sites + batched wrappers (checker Issue 3),
  verified by a full `cargo build --workspace` (not just `-p lgbm-compute`);
  `gain_params_parity.rs` passes with goldens or SKIPs cleanly.
- **All waves:** `cargo test --workspace` (CPU default features) passes with
  no new failures outside the wave's own new tests.

## Rollback / compatibility

- **G1/G2 (Waves A/B) are purely additive** — two new `lgbm-model` modules,
  two new `Booster` methods, two new parity test files, two new xtask
  subcommands. Revert = delete those files/methods/subcommands; no other code
  depends on them (SPEC §8 — no breaking change).
- **G4 (Wave C) only widens currently-erroring inputs** (SPEC §8) — revert
  restores the `learner.rs:1110-1121` gate and the `split.rs` rejection at the
  two mandatory sites (`:514-517`, `:7146-7150`); every input that trains
  successfully TODAY is untouched by this wave (the new code paths are only
  reachable when `na_as_missing()` is true, which is exactly the set of inputs
  that error today).
- **G5 (Wave D) only widens currently-erroring inputs for `max_delta_step`/
  `path_smooth`** (SPEC §8); the `feature_contri`/penalty addition is a NEW
  default-off `Config` field (empty vector → penalty 1.0 everywhere,
  behaviorally a no-op for every existing corpus that does not set it) — also
  strictly additive. Revert paths, per task:
  - **T-G5-1 (penalty):** revert the `Config.feature_contri` field + the single
    learner-level `split.gain *=` post-multiply line — no trait/kernel surface
    touched, independently revertible.
  - **T-G5-2 (max_delta_step):** revert the new `gain.rs` clamp fn + its two
    `split.rs` call-site wirings; restore `max_delta_step` in the rejection
    condition.
  - **T-G5-3 (path_smooth):** revert the two `split.rs` call-site wirings + the
    `parent_output` parameter across the trait decl, both implementors, and the
    batched wrappers (one atomic workspace-compiling unit — not independently
    revertible per call site); restore `path_smooth` in the rejection condition.
    `gain.rs`'s smoothing fns are untouched (pre-existing).
- **No model-format change; no persisted-schema migration** in any wave.
- On landing G2, correct `.planning/PROJECT.md`'s inaccurate "Validated: text &
  JSON serialization" line (JSON was previously unimplemented — research §2.7,
  §6) — this is a documentation follow-up, not a task in this plan (out of
  this plan's file-touch scope; flag it to the user/maintainer separately).

---

## SPEC-ID -> Task coverage

| SPEC ID | Task(s) |
|---|---|
| SPEC-G2-1 | T-G2-1 |
| SPEC-G2-2 | T-G2-2 |
| SPEC-G2-3 | T-G2-3 |
| SPEC-G2-4 | T-G2-4 |
| SPEC-G1-1 | T-G1-1 |
| SPEC-G1-2 | T-G1-2 |
| SPEC-G1-3 | T-G1-3 |
| SPEC-G1-4 | T-G1-4 |
| SPEC-G4-1 | T-G4-1 |
| SPEC-G4-2 | T-G4-2 |
| SPEC-G4-3 | T-G4-3 |
| SPEC-G4-4 | T-G4-4 |
| SPEC-G5-1 | T-G5-1 |
| SPEC-G5-2 | T-G5-2 |
| SPEC-G5-3 | T-G5-3 |
| SPEC-G5-4 | T-G5-4 |

Every SPEC ID (SPEC-G2-1..4, SPEC-G1-1..4, SPEC-G4-1..4, SPEC-G5-1..4) maps to
exactly one task; every task above cites at least one SPEC ID.

---

## Outstanding blockers / unresolved items carried into implementation

- **R-1 (blocking exactness, all of Wave C/D):** `LightGBM/` 4.6 is absent in
  this sandbox (P-1). Every `[UNVERIFIED against C++ source]`-tagged claim in
  SPEC.md, and this plan's F-1..F-6 findings that reference "P-1-verified,"
  must be confirmed against the checked-out tree before the corresponding
  Green step, not inferred.
- **OQ-1 (G5-1 penalty source):** this session's evidence (F-1) narrows it to
  `Config.feature_contri` (new field, IN_SCOPE + aliased but unimplemented) as
  the almost-certain source, distinct from the already-wired CEGB penalty
  mechanism — but the exact parse syntax, default-when-empty rule, and
  `gain *= penalty` application point (before/after `min_gain_shift`
  subtraction) still need P-1 confirmation.
- **OQ-2 (G5-3 parent-output availability):** not resolved this session;
  T-G5-3's Red step is where it must be resolved (via `codegraph_explore` on
  the `learner.rs` split-call site's local variables before writing the test).
- **T-G4-2's exact partition function name/line** is marked
  `[UNVERIFIED exact symbol]` in this plan — SPEC §7 only says "serial
  histogram/partition routing in `lgbm-treelearner`" generically; the
  implementer must re-run `codegraph_explore` for the concrete function at
  task start (this plan intentionally does not guess a symbol/line it did not
  verify this session).
- **F-2/F-3/F-6 scope-narrowing (P-4)** — **CHECKER-VALIDATED as SOUND** and
  now recorded in SPEC §2 non-goals. The Plan Checker independently traced the
  full CpuBackend dispatch and confirmed every default-CPU host scan path
  (`find_best_splits_batched` `lib.rs:2571`, `build_fix_scan_impl` `:2871`,
  `subtract_scan_impl` `:3014`) funnels through `self.find_best_split` →
  `find_best_split_cpu_native`, while the left-rejecting fused/batched/resident
  sites are reachable ONLY via `grow_tree_on_device_resident` (unreachable on
  CpuBackend — `on_device/resident/fused_eligible` are always `false`). So
  targeting the two host functions covers the entire default CPU path with NO
  reachable-but-rejecting production path. One documented exception: the opt-in
  `LGBM_SPLIT_2LANE=1` variant stays rejecting by design (P-4, checker Issue 5).

**No GSD skill, command, workflow, or agent was used to produce this plan.**
This plan was authored directly per the calling agent's explicit instructions
(single `PLAN.md` deliverable, no PageIndex sync, no per-task files, no
production code changes), using CodeGraph MCP (`codegraph_explore`) plus
targeted `Read`/`Bash grep` for verification only.
