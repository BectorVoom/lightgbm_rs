> **⚠ SUPERSEDED (2026-07-16)** by `.planning/plans/unimplemented-features/PLAN.md`.
> Kept for history; implement from the successor plan.

# TDD Implementation Plan — Parity Gap Closure (G2 · G1 · G4 · G5)

Derived from `.planning/plans/parity-gap-closure/SPEC.md` (draft) and
`research.md`. Every task is Red → Green → Refactor. Tasks are ordered by
dependency, not file layout. **No task is marked complete during planning.**

## Global preconditions (do once, before any Green step)

- **P-0 Confirm dependencies (AGENTS.md rule).** No new external crate is needed
  for G1/G2/G4/G5 (SPEC §3). Do NOT add `serde_json` (DEC-1). Record the
  dependency confirmation in the eventual commit message (AGENTS.md rule 3).
- **P-1 Checkout the C++ reference.** Ensure `LightGBM/` (4.6) is present in the
  working tree — required to verify G4/G5 algorithms and to generate goldens.
  Verify: `ls LightGBM/src/treelearner/feature_histogram.hpp`.
- **P-2 Oracle env.** Install `lightgbm==4.6.0` via the uv wheel (memory
  `cpp-linear-tree-oracle.md`) for golden generation.
- **P-3 Test discipline.** Run treelearner/on-device suites with
  `LGBM_CUDA_ON_DEVICE=0` (research §8). All parity tests SKIP-gracefully when a
  golden is absent (idiom: `predict_parity.rs:read_golden`).

Validation commands referenced below:
```bash
cargo test -p lgbm-model                                   # unit specs G1/G2
cargo test -p oracle-harness --test json_dump_parity       # G2-4
cargo test -p oracle-harness --test ifelse_codegen_parity  # G1-4
LGBM_CUDA_ON_DEVICE=0 cargo test -p lgbm-treelearner       # G4 unit
cargo test -p oracle-harness --test na_missing_parity      # G4-4
cargo test -p lgbm-compute                                 # G5 unit
cargo test -p oracle-harness --test gain_params_parity     # G5-4
```

---

## Wave A — G2 Model JSON dump (self-contained; do first)

### T-G2-1 — `tree_to_json` (Red→Green→Refactor) · SPEC-G2-1
- **prereqs:** P-0. **files:** new `crates/lgbm-model/src/json.rs`; register `mod
  json;` in `crates/lgbm-model/src/lib.rs`.
- **Red:** add `#[cfg(test)]` `tree_to_json_numeric_root` in `json.rs` — build a
  small `Tree` (one numeric split, two leaves), call `json::tree_to_json(&t)`,
  assert the substring for `"threshold"` uses `%g` (`format_g17`) and node
  nesting is present. Expected initial failure: `json` module/`tree_to_json`
  does not exist (compile error).
- **Green:** implement `tree_to_json` — pre-order recursion from node 0 decoding
  `decision_type` (bit0 categorical, bit1 default-left, bits2-3 missing_type,
  per `tree.rs:24-26`), floats via `format::{format_g17,format_g6}`. Match the
  C++ key set from `LightGBM/src/io/tree.cpp` `Tree::NodeToJSON` (P-1).
- **Refactor:** extract a `push_kv` helper; keep `%g` the only float path (R-2).
- **validation:** `cargo test -p lgbm-model json::`. **completion evidence:**
  unit test green; no `to_string()`/`ryu` on floats.

### T-G2-2 — `dump_model` ensemble document · SPEC-G2-2 (dep: T-G2-1)
- **files:** `crates/lgbm-model/src/json.rs`.
- **Red:** `dump_model_two_tree_binary` — build a 2-tree `GbdtModel`, call
  `json::dump_model`, assert top-level keys (`num_class`, `num_tree_per_iteration`,
  `tree_info`) present and `tree_info` length == 2. Expected failure:
  `dump_model` absent.
- **Green:** emit the top-level object mirroring `model_text::save` field-presence
  rules (`model_text.rs:216-260`: `average_output`/`monotone_constraints` only
  when set); call `tree_to_json` per tree. Verify key set vs C++ `DumpModel` (P-1).
- **Refactor:** share the feature-name/`feature_infos` formatting with the text
  path if trivially reusable (do NOT over-abstract).
- **validation:** `cargo test -p lgbm-model json::`.

### T-G2-3 — `Booster::dump_model` facade · SPEC-G2-3 (dep: T-G2-2)
- **files:** `crates/lgbm/src/booster.rs` (near `model_to_string`, `:728`).
- **Red:** `dump_model_matches_module` in `booster.rs` tests — train a tiny
  booster, assert `b.dump_model() == json::dump_model(&model)`. Expected
  failure: method absent.
- **Green:** add `pub fn dump_model(&self) -> String` delegating to
  `lgbm_model::json::dump_model` over the booster's model (same accessor
  `model_to_string` uses).
- **Refactor:** none beyond doc comment.
- **validation:** `cargo test -p lgbm`.

### T-G2-4 — JSON parity vs 4.6 · SPEC-G2-4 (dep: T-G2-3, P-2)
- **files:** new `crates/oracle-harness/tests/json_dump_parity.rs`; golden under
  `crates/oracle-harness/tests/fixtures/predict_modes/` (or a new `json/` dir).
- **Red:** write the parity test using the `predict_parity.rs` idiom
  (`CARGO_MANIFEST_DIR` fixture root, SKIP when golden absent). With no golden
  yet, it SKIPs (passes) — then generate the golden and watch it fail if output
  diverges.
- **Green:** generate the golden (`.dump_model()` from `lightgbm==4.6.0` on a
  captured model; add an `xtask` capture subcommand or a `gen_golden.py` mirroring
  the quantized fixture). Fix any `%g`/key-order diffs until byte-exact.
- **Refactor:** fold shared golden-loading into `oracle_harness::comparator` if a
  helper doesn't already fit.
- **validation:** `cargo test -p oracle-harness --test json_dump_parity`
  (present → passes; absent → SKIP). **completion evidence:** AS-1 met.

---

## Wave B — G1 If-else codegen (shares tree-walk substrate with G2)

### T-G1-1 — `tree_to_cpp` fragment · SPEC-G1-1 (dep: none; reuse T-G2-1 patterns)
- **files:** new `crates/lgbm-model/src/codegen_cpp.rs`; `mod codegen_cpp;` in
  `lib.rs`.
- **Red:** `tree_to_cpp_numeric` — assert the emitted body contains an
  `if (... <= <%g threshold>)` and returns a `%g` leaf value. Expected failure:
  module absent.
- **Green:** implement nested if-else pre-order walk matching `Tree::predict`
  comparison direction + missing/default-left handling (`tree.rs:269`); floats via
  `format_g17`. Verify branch/missing form vs C++ `Tree::NodeToIfElse` (P-1).
- **Refactor:** reuse the `decision_type` decode helper from `json.rs` if one was
  extracted (shared substrate, SPEC §7).
- **validation:** `cargo test -p lgbm-model codegen_cpp::`.

### T-G1-2 — `model_to_cpp` full source · SPEC-G1-2 (dep: T-G1-1)
- **files:** `crates/lgbm-model/src/codegen_cpp.rs`.
- **Red:** `model_to_cpp_two_tree` — assert output has one function per tree and a
  `PredictRaw` summation. Expected failure: `model_to_cpp` absent.
- **Green:** emit headers + per-tree functions + ensemble summation × shrinkage
  with the `num_tree_per_iteration` class stride, mirroring `SaveModelToIfElse`
  (P-1).
- **Refactor:** none beyond dedup with T-G1-1.
- **validation:** `cargo test -p lgbm-model codegen_cpp::`.

### T-G1-3 — `Booster::model_to_cpp` facade · SPEC-G1-3 / DEC-2 (dep: T-G1-2)
- **files:** `crates/lgbm/src/booster.rs`.
- **Red:** `model_to_cpp_matches_module` — assert `b.model_to_cpp() ==
  codegen_cpp::model_to_cpp(&model)`. Expected failure: method absent.
- **Green:** `pub fn model_to_cpp(&self) -> String` delegating; **no file
  side-effect** (DEC-2).
- **validation:** `cargo test -p lgbm`.

### T-G1-4 — If-else parity vs 4.6 · SPEC-G1-4 (dep: T-G1-3, P-2)
- **files:** new `crates/oracle-harness/tests/ifelse_codegen_parity.rs` + golden
  `.cpp` fixture.
- **Red:** SKIP-graceful parity test; generate golden via `lightgbm==4.6.0`
  `convert_model`.
- **Green:** reconcile byte-diffs to exact.
- **validation:** `cargo test -p oracle-harness --test ifelse_codegen_parity`.
  **completion evidence:** AS-2 met.

---

## Wave C — G4 NA-as-missing serial forward routing (hot path; do NOT parallelize with G5)

> **P-1 mandatory before Green.** Resolve every SPEC-G4 `TBD` from the C++ source
> first (missing-bin index, default-branch rule).

### T-G4-1 — NA rows → missing bin in histogram build · SPEC-G4-1
- **files:** `crates/lgbm-treelearner/src/learner.rs` + serial histogram path
  (possibly `crates/lgbm-compute` CPU histogram).
- **Red:** unit/integration test building a node histogram over a tiny NaN-bearing
  `use_missing=true` feature; assert the missing-bin (Σg,Σh) equals a
  hand-computed expected (from C++ semantics verified in P-1). Expected failure:
  today the path errors at `learner.rs:1113` before building.
- **Green:** implement NA accumulation into the missing bin (leave the `:1113`
  gate in place for now so only the histogram-build unit exercises new code).
- **Refactor:** keep f64-fold accumulation order identical to the non-NA path
  (bit-exact contract).
- **validation:** `LGBM_CUDA_ON_DEVICE=0 cargo test -p lgbm-treelearner`.

### T-G4-2 — NA routed down default branch on partition · SPEC-G4-2 (dep: T-G4-1)
- **files:** serial `data_partition` routing in `lgbm-treelearner`.
- **Red:** test that partitions a node on the NA feature and asserts NA rows land
  in the `default_left`-selected child (row-for-row vs expected). Expected
  failure: routing not implemented.
- **Green:** implement default-direction routing per the P-1-verified rule.
- **validation:** `LGBM_CUDA_ON_DEVICE=0 cargo test -p lgbm-treelearner`.

### T-G4-3 — Remove the typed-error gate · SPEC-G4-3 (dep: T-G4-1, T-G4-2)
- **files:** `crates/lgbm-treelearner/src/learner.rs:1113`.
- **Red:** update the existing "rejects na_as_missing" unit test to instead
  **assert training succeeds** on the NaN corpus. Expected failure: gate still
  returns `TreeLearnerError::Compute(Runtime{..})`.
- **Green:** remove the gate block (`:1113-1121`); training completes.
- **Refactor:** ensure no other call site relied on the error.
- **validation:** `LGBM_CUDA_ON_DEVICE=0 cargo test -p lgbm-treelearner`.

### T-G4-4 — NA parity vs 4.6 · SPEC-G4-4 (dep: T-G4-3, P-1, P-2)
- **files:** new `crates/oracle-harness/tests/na_missing_parity.rs` + golden
  (bin histograms + predictions) under `tests/fixtures/`.
- **Red:** SKIP-graceful parity test; generate NaN-corpus golden from
  `lightgbm==4.6.0` (`use_missing=true`).
- **Green:** reconcile to CPU f64-fold bit-exact histograms + predictions.
- **validation:** `LGBM_CUDA_ON_DEVICE=0 cargo test -p oracle-harness --test
  na_missing_parity`. **completion evidence:** AS-3 met.

---

## Wave D — G5 split-kernel gain params (hot split kernel; after G4)

> **P-1 mandatory before Green.** Resolve `path_smooth`/`max_delta_step`/`penalty`
> formulas from `feature_histogram.hpp` first.

### T-G5-1 — Feature penalty on gain · SPEC-G5-1
- **files:** `crates/lgbm-compute/src/kernels/split.rs` (`find_best_split_cpu`,
  currently hard-codes `penalty=1.0`); penalty source threading.
- **Red:** unit test calling `find_best_split_cpu` with `penalty=0.5` asserts the
  returned gain == 0.5 × the `penalty=1.0` gain. Expected failure: penalty
  ignored (hard-coded 1.0).
- **Green:** thread `penalty` through and apply `gain *= penalty` (C++
  `output->gain *= penalty`, verified P-1). Resolve OQ-1 (penalty source).
- **validation:** `cargo test -p lgbm-compute`.

### T-G5-2 — `max_delta_step` clamp · SPEC-G5-2 (dep: independent of T-G5-1)
- **files:** `crates/lgbm-compute/src/kernels/split.rs`; leaf-output calc.
- **Red:** test with `max_delta_step=0.7` (via `GainConfig`) asserts leaf output
  clamped exactly; and that `find_best_split_cpu` no longer returns
  `ComputeError::Runtime` for non-default `max_delta_step`. Expected failure:
  current `Runtime` rejection (`split.rs` doc).
- **Green:** apply the clamp in the leaf-output/gain formula (P-1); drop the
  rejection branch for `max_delta_step`.
- **validation:** `cargo test -p lgbm-compute`.

### T-G5-3 — `path_smooth` smoothing · SPEC-G5-3 (dep: independent)
- **files:** `crates/lgbm-compute/src/kernels/split.rs`; leaf-output calc; parent
  output threading (resolve OQ-2 in Red).
- **Red:** test with `path_smooth=2.0` asserts the smoothed leaf output equals the
  C++ value; and non-default `path_smooth` no longer rejected. Expected failure:
  current `Runtime` rejection.
- **Green:** implement the smoothing formula (P-1), threading the parent output;
  drop the rejection branch for `path_smooth`.
- **validation:** `cargo test -p lgbm-compute`.

### T-G5-4 — Gain-param parity vs 4.6 · SPEC-G5-4 (dep: T-G5-1..3, P-2)
- **files:** new `crates/oracle-harness/tests/gain_params_parity.rs` + three
  goldens (`penalty`, `max_delta_step=0.7`, `path_smooth=2.0`).
- **Red:** SKIP-graceful parity test; generate the three goldens.
- **Green:** reconcile split-gain + threshold + leaf-output to CPU f64-fold
  bit-exact.
- **validation:** `LGBM_CUDA_ON_DEVICE=0 cargo test -p oracle-harness --test
  gain_params_parity`. **completion evidence:** AS-4 met.

---

## Execution order & parallelism

1. **Wave A (G2)** — no cross-deps; start immediately after P-0.
2. **Wave B (G1)** — can start in parallel with A once T-G2-1 extracts the shared
   `decision_type`/`%g` helper (soft dep for reuse only).
3. **Wave C (G4)** — requires P-1 (C++ source). Hot path — run its own regression.
4. **Wave D (G5)** — requires P-1. **Do NOT run concurrently with Wave C** (both
   touch hot histogram/split kernels; isolate parity regressions — research §10).

Within a wave, tasks are strictly sequential by the listed deps. T-G5-1/2/3 are
mutually independent (parallelizable) but all precede T-G5-4.

## Definition of done (per wave)

- All Red tests turned Green; Refactor left behavior unchanged.
- The wave's `*_parity.rs` passes with a committed golden (or SKIPs cleanly with a
  documented "golden absent" reason if P-2 could not run in the environment).
- Commit message records the dependency confirmation (AGENTS.md rule 3) and cites
  the SPEC IDs closed.
- No float emitted via anything but `%g` (grep guard for `ryu`/`to_string()` on
  floats in the new modules).

## Rollback / compatibility notes

- G1/G2 are additive — revert = drop the two new modules + two `Booster` methods.
- G4/G5 only widen currently-erroring inputs — revert restores the typed-error
  gates (`learner.rs:1113`; `split.rs` `Runtime` rejections). No model-format or
  persisted-schema change, so no migration to roll back.
