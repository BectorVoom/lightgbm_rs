---
title: Quantized-Gradient Training Param Plumbing
status: draft
format: markdown
spec_version: 1
updated_at: 2026-07-12T00:00:00Z
source_requirements:
  - User request (2026-07-12): "implement unimplemented lightgbm_rs comparing with lightgbm"
  - .planning/PROJECT.md — milestone "v1.0 C++ Feature-Parity Audit & Gap Closure", Phase 1 audit dimension "quantized gradient"
  - .planning/plans/cpp-feature-parity/research.md — Gap #3 "Quantized-gradient (`use_quantized_grad`) param-plumbing gap"
---

# Quantized-Gradient Training Param Plumbing

## 1. Context

`lightgbm_rs` already has a complete, C++-oracle-parity-tested implementation of
LightGBM's `use_quantized_grad` APPROXIMATE training mode: gradient/hessian
discretization (`GradientDiscretizer`, including stochastic rounding), the
quantize-then-train call site in `Gbdt::train_one_iter`, and leaf-output
renewal (`quant_train_renew_leaf`) are all implemented and covered by 3
passing oracle-harness tests
(`crates/oracle-harness/tests/quantized_parity.rs`)
`[VERIFIED: LOCAL crates/oracle-harness/tests/quantized_parity.rs — cargo test -p oracle-harness --test quantized_parity: 3 passed]`.

However, the four canonical C++ config keys that control this feature —
`use_quantized_grad`, `num_grad_quant_bins`, `quant_train_renew_leaf`,
`stochastic_rounding` — are NOT reachable through the string-keyed
`Config::from_params` path that every non-Rust caller (Python bindings, and
any future CLI/dict-driven caller) must use. They remain listed in
`scope::OUT_OF_SCOPE_PARAMS`
`[VERIFIED: LOCAL crates/lgbm-core/src/config/scope.rs:190-194]`, are never
parsed in `config/set.rs`
`[VERIFIED: LOCAL grep -n "quantiz\|stochastic_rounding\|num_grad_quant_bins\|quant_train_renew_leaf" crates/lgbm-core/src/config/set.rs → zero matches]`,
and are actively rejected by the Python binding's `reject_unimplemented` gate
`[VERIFIED: LOCAL crates/lgbm-python/src/params.rs:150-159]`. The only
existing callers reach the feature by constructing/mutating a `Config`
struct directly and calling `Gbdt::with_quantized_grad`/
`with_quant_renew_leaf` — a path unavailable to Python or any string-param
caller `[VERIFIED: LOCAL crates/lgbm/src/booster.rs:1195-1203, crates/oracle-harness/tests/quantized_parity.rs:81-97]`.

Several doc comments and one Python-side test are also stale as a direct
result of this half-finished state (see §5, QGP-06/QGP-07) — most notably
`Config`'s own field docs claim `quant_train_renew_leaf`/`stochastic_rounding`
are "not yet implemented"
`[VERIFIED: LOCAL crates/lgbm-core/src/config/mod.rs:116,120]` when both are
in fact implemented and (for `quant_train_renew_leaf`) oracle-tested.

This closely mirrors the already-completed precedent of moving
`linear_tree`/`linear_lambda` from `OUT_OF_SCOPE_PARAMS` to
`IN_SCOPE_PARAMS` (uncommitted diff on this branch,
`crates/lgbm-core/src/config/{scope,set}.rs`), which this spec's
acceptance criteria deliberately mirror
`[VERIFIED: LOCAL git diff crates/lgbm-core/src/config/set.rs:254-256, scope.rs:95-96]`.

Additionally, the existing oracle golden
(`crates/oracle-harness/tests/fixtures/quantized/`) only exercises
`stochastic_rounding=false` (the deterministic, parity-tractable path); no
C++ golden exists for `stochastic_rounding=true`, which is the C++
`config.h` DEFAULT for that key `[VERIFIED: LOCAL crates/lgbm-core/src/config/mod.rs:118 "config.h default: true"]`.
Per user decision (2026-07-12), this spec wires all 4 params through in one
pass, but requires a new C++ oracle golden for `stochastic_rounding=true` be
added in the same body of work, so the default-true stochastic path gets
real parity coverage before it becomes reachable from `Config::from_params`.

## 2. Scope and Non-Goals

**In scope:**
- Parsing `use_quantized_grad`, `num_grad_quant_bins`, `quant_train_renew_leaf`,
  `stochastic_rounding` in `Config::from_params` (`config/set.rs`).
- Reclassifying all 4 canonical names from `scope::OUT_OF_SCOPE_PARAMS` to
  `scope::IN_SCOPE_PARAMS`.
- Removing the Python-binding rejection of these 4 keys and fixing the
  now-stale `reject_gate` test and error-string/doc-comment wording that
  names them as excluded.
- Correcting stale "not yet implemented" doc comments on `Config` fields and
  the `GradientDiscretizer` module doc that no longer match the shipped
  implementation.
- A new C++ oracle golden + Rust test proving `stochastic_rounding=true`
  training matches real `lib_lightgbm` 4.6 within the same
  magnitude-of-quantization-effect delta-gate methodology already used by
  `rust_quantized_train_matches_cpp`
  (`crates/oracle-harness/tests/quantized_parity.rs:72-154`) — NOT a naive
  absolute-1e-6 assertion, because quantization is inherently approximate by
  construction and the project's own existing deterministic-path test
  already gates on relative delta-regime, not raw 1e-6
  `[VERIFIED: LOCAL crates/oracle-harness/tests/quantized_parity.rs:129-153]`.
- An end-to-end test that drives training through the actual
  `Config::from_params` string-param path (not direct field mutation) to
  prove the wiring closes the loop for a dict-style caller.

**Out of scope (do not implement here):**
- Any change to the underlying quantization math, `GradientDiscretizer`, or
  the `Gbdt` quantize/renew call sites — these are already implemented and
  oracle-tested; this spec only unlocks *reachability* via string params.
- A dedicated `ConfigBuilder`/`TrainingBuilder` ergonomic setter for these 4
  params — none exists today for the analogous `linear_tree`/`linear_lambda`
  pair either `[VERIFIED: LOCAL grep for these 6 names in crates/lgbm/src/builder.rs → no hits]`,
  so adding one here would be an unrequested, inconsistent enhancement.
- Fixing the pre-existing, unrelated `reject_gate` test bug for `linear_tree`
  (see §9 Risk 1) — flagged as a discovered defect, not fixed by this spec,
  since it belongs to the linear-tree slice, not this one. It is touched
  incidentally only insofar as QGP-06's Green step edits the same test
  function; the `linear_tree` sub-case is left exactly as-is unless the user
  separately asks for it.
- `crates/lgbm-python/python/tests/*.py`-level acceptance tests — their
  current content was not read this session
  `[UNVERIFIED: not read this session]`; Python-level coverage of this
  feature is deferred as an open question (§9 Q3).
- Any change to `crates/lgbm/src/builder.rs` or new public Rust API surface.

## 3. Dependencies

- `lgbm_core::config::Config` — existing struct, fields already present
  (`crates/lgbm-core/src/config/mod.rs:108-121`); this spec does not add new
  fields, only wires existing ones.
- `lgbm_core::config::set::{get_bool, get_int, check_ge, check_le}` — existing
  private helpers in `crates/lgbm-core/src/config/set.rs:535-565,809-831`.
- `lgbm_core::error::ConfigError::OutOfRange { param, value, bound }` —
  existing variant (`crates/lgbm-core/src/error.rs:19-40ish`), constructed via
  the existing `out_of_range()` helper (`set.rs:865-871`).
- `lgbm_core::config::scope::{IN_SCOPE_PARAMS, OUT_OF_SCOPE_PARAMS}` —
  existing arrays (`crates/lgbm-core/src/config/scope.rs`).
- `lgbm_python::params::reject_unimplemented` — existing function
  (`crates/lgbm-python/src/params.rs:150-179`), depends transitively on
  `OUT_OF_SCOPE_PARAMS`.
- `crates/lgbm-treelearner/src/gradient_discretizer.rs` `GradientDiscretizer`
  — existing, unmodified by this spec except its stale module doc comment.
- Oracle golden generator pattern:
  `crates/oracle-harness/tests/fixtures/quantized/gen_golden.py` — requires a
  real `lightgbm` (C++ reference, v4.6) Python installation to regenerate/add
  goldens `[VERIFIED: LOCAL gen_golden.py:12 "Run (from repo root):
  .venv/bin/python crates/oracle-harness/tests/fixtures/quantized/gen_golden.py"]`.
  No new crate dependency is introduced.

No new external crate/version is required; all work is confined to existing
crates already in the workspace `Cargo.lock`
`[VERIFIED: LOCAL .planning/plans/cpp-feature-parity/research.md §5]`.

## 4. Typed Contracts

```rust
// crates/lgbm-core/src/config/mod.rs (existing, unmodified struct shape)
pub struct Config {
    pub use_quantized_grad: bool,       // default: false
    pub num_grad_quant_bins: i32,       // default: 4, valid range 1..=254
    pub quant_train_renew_leaf: bool,   // default: false
    pub stochastic_rounding: bool,      // default: true
    // ...
}

// crates/lgbm-core/src/config/mod.rs (existing signature, unmodified)
impl Config {
    pub fn from_params(params: &HashMap<String, String>) -> Result<Config, ConfigError>;
}

// crates/lgbm-core/src/error.rs (existing variant, unmodified)
pub enum ConfigError {
    InvalidType { param: String, value: String },
    UnknownValue { param: String, value: String },
    OutOfRange { param: String, value: String, bound: String },
    // ...
}

// crates/lgbm-python/src/params.rs (existing signature, unmodified)
pub fn reject_unimplemented(map: &HashMap<String, String>) -> PyResult<()>;
```

No new types, fields, or public function signatures are introduced by this
spec — every contract above already exists; the work is exclusively about
which string keys route into which existing code path.

## 5. Failure-Isolated Behavioral Specifications

### QGP-01: `Config::from_params` parses `use_quantized_grad`

- **Status**: implemented `[VERIFIED: LOCAL crates/lgbm-core/src/config/set.rs:258 get_bool wiring; crates/lgbm-core/tests/config_validation.rs::quantized_grad_bool_params_parse_and_default — cargo test -p lgbm-core --test config_validation: 20 passed]`
- **Rationale**: currently silently ignored (unknown-key warn-only path);
  must route to the existing `cfg.use_quantized_grad: bool` field.
- **Preconditions**: none.
- **Input**: `HashMap<String, String>` containing key `"use_quantized_grad"`
  with a boolean-coercible string value (`"true"`/`"false"`/`"+"`/`"-"`,
  matching the existing `bool_coercion_matches_cpp` convention
  `[VERIFIED: LOCAL crates/lgbm-core/tests/config_validation.rs:186-197]`).
- **Output**: `Ok(Config)` with `cfg.use_quantized_grad` set to the parsed
  value; absent key → default `false`
  (`crates/lgbm-core/src/config/mod.rs:361`); malformed value → `Err(ConfigError::InvalidType)`.
- **Dependencies**: `get_bool` helper (`set.rs:561`).
- **Behavior (Given/When/Then)**:
  - Given `{"use_quantized_grad": "true"}`, when `from_params` is called,
    then `Ok(cfg)` with `cfg.use_quantized_grad == true`.
  - Given no `use_quantized_grad` key, when `from_params` is called, then
    `Ok(cfg)` with `cfg.use_quantized_grad == false` (default).
  - Given `{"use_quantized_grad": "maybe"}`, when `from_params` is called,
    then `Err(ConfigError::InvalidType { param: "use_quantized_grad", .. })`.
- **Invariants/side effects**: none beyond the field assignment; no other
  `Config` field changes as a result of this key alone.
- **Acceptance tests**: see §6, AT-01.
- **Out of scope**: alias resolution (no known C++ alias for this key;
  `[UNVERIFIED: not confirmed against config_auto.cpp — sandbox has no
  LightGBM/ tree, see Risk 2]`).
- **Traceability**: `[VERIFIED: CODEGRAPH fit... ]` n/a — direct grep/read
  evidence cited inline above.
- **Unresolved questions**: none.

### QGP-02: `Config::from_params` parses and range-validates `num_grad_quant_bins`

- **Status**: implemented `[VERIFIED: LOCAL crates/lgbm-core/src/config/set.rs:259-261 get_int + check_ge/check_le wiring; crates/lgbm-core/tests/config_validation.rs::num_grad_quant_bins_parses_and_validates_range — cargo test -p lgbm-core --test config_validation: 20 passed]`
- **Rationale**: `GradientDiscretizer::new` currently only enforces
  `1..=254` via `debug_assert!`
  (`crates/lgbm-treelearner/src/gradient_discretizer.rs:50-53`), which is
  compiled out in release builds. Per CLAUDE.md/AGENTS.md T-08-05-03 ("no
  panic ever crosses the FFI boundary"), the config-parsing layer — which
  sits directly on the Python FFI boundary — must turn out-of-range input
  into a typed `ConfigError`, not rely on a debug-only assert deep in the
  training path.
- **Preconditions**: none.
- **Input**: `HashMap<String, String>` containing key
  `"num_grad_quant_bins"` with an integer-coercible string value.
- **Output**: `Ok(Config)` with `cfg.num_grad_quant_bins` set, when
  `1 <= value <= 254`; absent key → default `4`
  (`config/mod.rs:362`); non-integer value → `Err(ConfigError::InvalidType)`;
  value `< 1` or `> 254` → `Err(ConfigError::OutOfRange)`.
- **Dependencies**: `get_int` (`set.rs:535`), `check_ge`/`check_le`
  (`set.rs:809,825`).
- **Behavior (Given/When/Then)**:
  - Given `{"num_grad_quant_bins": "128"}`, then `Ok(cfg)` with
    `cfg.num_grad_quant_bins == 128`.
  - Given `{"num_grad_quant_bins": "1"}` (lower boundary), then `Ok(cfg)`.
  - Given `{"num_grad_quant_bins": "254"}` (upper boundary), then `Ok(cfg)`.
  - Given `{"num_grad_quant_bins": "0"}`, then
    `Err(ConfigError::OutOfRange { param: "num_grad_quant_bins", .. })`.
  - Given `{"num_grad_quant_bins": "255"}`, then
    `Err(ConfigError::OutOfRange { param: "num_grad_quant_bins", .. })`.
  - Given `{"num_grad_quant_bins": "-4"}`, then `Err(ConfigError::OutOfRange)`.
- **Invariants/side effects**: none.
- **Acceptance tests**: see §6, AT-02.
- **Out of scope**: changing `GradientDiscretizer`'s `debug_assert!` itself
  — leaving it in place as defense-in-depth is acceptable since the
  config-layer check now makes it unreachable in the standard training
  entry point; removing it is not required by this spec.
- **Unresolved questions**: none — exact bound `1..=254` is directly cited
  from the existing implementation's own assertion message
  (`gradient_discretizer.rs:52`), not invented.

### QGP-03: `Config::from_params` parses `quant_train_renew_leaf`

- **Status**: implemented `[VERIFIED: LOCAL crates/lgbm-core/src/config/set.rs:262 get_bool wiring; crates/lgbm-core/tests/config_validation.rs::quantized_grad_bool_params_parse_and_default — cargo test -p lgbm-core --test config_validation: 20 passed]`
- **Rationale**: symmetric to QGP-01; field already exists and is already
  wired end-to-end into `Gbdt` (`crates/lgbm/src/booster.rs:1203`) and
  oracle-tested (`rust_quant_renew_leaf_matches_cpp_effect`,
  `quantized_parity.rs:161-204`) — only the string-param path is missing.
- **Preconditions**: none.
- **Input**: `HashMap<String, String>` with key `"quant_train_renew_leaf"`,
  boolean-coercible value.
- **Output**: `Ok(Config)` with `cfg.quant_train_renew_leaf` set; absent →
  default `false` (`config/mod.rs:363`); malformed → `Err(ConfigError::InvalidType)`.
- **Dependencies**: `get_bool`.
- **Behavior**: same Given/When/Then shape as QGP-01, substituting the key
  name.
- **Acceptance tests**: see §6, AT-03.
- **Unresolved questions**: none.

### QGP-04: `Config::from_params` parses `stochastic_rounding`

- **Status**: implemented `[VERIFIED: LOCAL crates/lgbm-core/src/config/set.rs:263 get_bool wiring; crates/lgbm-core/tests/config_validation.rs::quantized_grad_bool_params_parse_and_default — cargo test -p lgbm-core --test config_validation: 20 passed]`
- **Rationale**: symmetric to QGP-01/03, but this key's C++ default is
  `true` (`config/mod.rs:118`), meaning once wired, a caller who sets only
  `use_quantized_grad=true` and nothing else will engage the stochastic
  path by default. This is precisely why QGP-08 (a new oracle golden for
  `stochastic_rounding=true`) is a hard co-requirement of this spec, not an
  optional follow-up (see §9 for the reasoning).
- **Preconditions**: none.
- **Input**: `HashMap<String, String>` with key `"stochastic_rounding"`,
  boolean-coercible value.
- **Output**: `Ok(Config)` with `cfg.stochastic_rounding` set; absent →
  default `true` (`config/mod.rs:364`); malformed → `Err(ConfigError::InvalidType)`.
- **Dependencies**: `get_bool`.
- **Behavior**: same Given/When/Then shape as QGP-01, substituting the key
  name and default (`true` when absent, not `false`).
- **Acceptance tests**: see §6, AT-04.
- **Unresolved questions**: none.

### QGP-05: Reclassify the 4 keys from `OUT_OF_SCOPE_PARAMS` to `IN_SCOPE_PARAMS`

- **Status**: implemented `[VERIFIED: LOCAL crates/lgbm-core/src/config/scope.rs — cargo test -p lgbm-core --test scope_classification: 1 passed; cargo test -p lgbm-core (full suite): 53 passed. NOTE: config_drift.rs re-verification against a real LightGBM/ C++ checkout (SPEC.md Risk 2) was NOT performed — no LightGBM/ tree present in this sandbox; config_drift's two file-reading tests fail with ENOENT (pre-existing, environmental, unrelated to this change) and should be re-run on a machine with LightGBM/ checked out.]`
- **Rationale**: `scope.rs` is the single source of truth consumed by (a)
  the Python `reject_unimplemented` gate and (b) the `oracle-harness`
  `config_drift` mechanical checker that diffs against real C++
  `config_auto.cpp`
  `[VERIFIED: LOCAL crates/oracle-harness/tests/config_drift.rs:30,135-153]`.
  Leaving the keys in `OUT_OF_SCOPE_PARAMS` after QGP-01..04 land would make
  the Python binding actively reject params that `Config::from_params` now
  successfully parses — an inconsistency, not a mere omission.
- **Preconditions**: QGP-01..04 land first (parsing must exist before the
  scope classification claims it does) — sequencing enforced in PLAN.md.
- **Input**: none (static data change).
- **Output**: `scope::IN_SCOPE_PARAMS` contains `"use_quantized_grad"`,
  `"num_grad_quant_bins"`, `"quant_train_renew_leaf"`,
  `"stochastic_rounding"`; `scope::OUT_OF_SCOPE_PARAMS` no longer contains
  any of them.
- **Dependencies**: none beyond the two const arrays
  (`crates/lgbm-core/src/config/scope.rs:35-171,178-195`).
- **Behavior (Given/When/Then)**:
  - Given the updated `scope.rs`, when `IN_SCOPE_PARAMS.contains(&"use_quantized_grad")`
    is evaluated, then `true` (and likewise for the other 3 keys).
  - Given the updated `scope.rs`, when `OUT_OF_SCOPE_PARAMS.contains(&"use_quantized_grad")`
    is evaluated, then `false` (and likewise for the other 3 keys).
- **Invariants/side effects**: the module-level doc comment
  (`scope.rs:21-26`, and the `OUT_OF_SCOPE_PARAMS` doc at `scope.rs:173-177`)
  must be updated in the same change to stop describing quantized-grad as
  "deferred" — leaving stale prose next to a corrected array is the exact
  defect already present for `linear_tree` (§9 Risk 1) and must not be
  repeated.
- **Acceptance tests**: see §6, AT-05.
- **Unresolved questions**: none.

### QGP-06: Python `reject_unimplemented` no longer rejects the 4 keys; stale wording fixed

- **Status**: draft — BLOCKED (environment). Code + test + doc/error-string edits are complete
  (`crates/lgbm-python/src/params.rs` `reject_gate` test rewritten per AT-06; doc comment and
  error string at `params.rs:138,156-157` no longer mention quantized-grad) and statically
  verified via `cargo check -p lgbm-python --tests` (clean compile, no errors)
  `[VERIFIED: LOCAL cargo check -p lgbm-python --tests]`, but the actual test CANNOT be executed
  in this sandbox: `cargo test -p lgbm-python` / `cargo build -p lgbm-python` fail to LINK
  (`mold: fatal: library not found: python3.14`), a pre-existing environment issue unrelated to
  this change `[VERIFIED: LOCAL cargo test -p lgbm-python 2>&1 → linker error]`. Per the
  specification completion gate, this is NOT marked `implemented` until the `reject_gate` test
  actually runs and passes on a machine with a working Python dev environment.
- **Rationale**: `reject_unimplemented` derives its behavior entirely from
  `OUT_OF_SCOPE_PARAMS` (`params.rs:153`), so QGP-05 mechanically fixes the
  *behavior*; this spec item covers the *test* and *prose* that assert the
  old (now-wrong) behavior.
- **Preconditions**: QGP-05 lands first.
- **Input**: `HashMap<String, String>` containing `{"use_quantized_grad": "true"}`
  (and similarly for the other 3 keys).
- **Output**: `reject_unimplemented(&map)` returns `Ok(())` (previously
  `Err(PyValueError)`).
- **Dependencies**: `lgbm_core::config::scope::OUT_OF_SCOPE_PARAMS` (via QGP-05).
- **Behavior (Given/When/Then)**:
  - Given `{"use_quantized_grad": "true"}`, when `reject_unimplemented` is
    called, then `Ok(())`.
  - Given `{"num_grad_quant_bins": "128"}`, `{"quant_train_renew_leaf": "true"}`,
    `{"stochastic_rounding": "false"}` individually, then `Ok(())` in each case.
  - Given `{"num_machines": "2"}` (an unrelated, still-out-of-scope key),
    when `reject_unimplemented` is called, then still `Err(PyValueError)`
    — regression guard that this change doesn't over-broadly loosen the gate.
- **Invariants/side effects**: `crates/lgbm-python/src/params.rs:316-318`
  (the `use_quantized_grad` sub-case of `reject_gate`) currently asserts the
  OLD, now-incorrect behavior and must be removed or rewritten to assert the
  new `Ok(())` behavior — this is itself the acceptance test, per T-01 in
  PLAN.md. The doc comment at `params.rs:138` and the error string at
  `params.rs:156-157` (both list "quantized-grad" as an excluded group) must
  drop that mention.
- **Acceptance tests**: see §6, AT-06.
- **Out of scope**: the pre-existing, independently-broken `linear_tree`
  sub-case of the same `reject_gate` test (`params.rs:312-314`) — see §9
  Risk 1. Not modified by this spec unless the user separately requests it.
- **Unresolved questions**: none.

### QGP-07: Correct stale "not yet implemented" doc comments

- **Status**: implemented `[VERIFIED: LOCAL crates/lgbm-core/src/config/mod.rs:115-121, crates/lgbm-treelearner/src/gradient_discretizer.rs:14-20 — grep-based negative assertions pass (stale strings absent); cargo doc -p lgbm-core -p lgbm-treelearner --no-deps: clean build]`
- **Rationale**: `Config` field docs and the `GradientDiscretizer` module
  doc currently assert things that are false once QGP-01..06 land (and, for
  `quant_train_renew_leaf`/deterministic `stochastic_rounding`, were already
  false before this spec — the implementation predates the doc fix). Stale
  "not implemented" claims next to working, oracle-tested code actively
  mislead future contributors and violate the project's own "no
  spike/phase R&D-history narrative in comments" convention (commit
  `11516b6`).
- **Preconditions**: none (can land independently, but bundled here since
  it's discovered by the same investigation and touches directly-related
  lines).
- **Input**: none (doc-only change).
- **Output**: `crates/lgbm-core/src/config/mod.rs:116` no longer claims
  `quant_train_renew_leaf` is "Not yet implemented"; `mod.rs:120` no longer
  claims `stochastic_rounding` "is not yet implemented"; the module doc in
  `crates/lgbm-treelearner/src/gradient_discretizer.rs:14-15` no longer
  claims "Stochastic rounding + `quant_train_renew_leaf` are deferred...
  This module is deterministic-only."
- **Behavior**: purely textual; acceptance is a grep-based negative
  assertion (see §6, AT-07) — no runtime behavior changes.
- **Acceptance tests**: see §6, AT-07.
- **Unresolved questions**: exact replacement wording is left to the
  implementer to match house style (see PLAN.md Green step); the
  specification only fixes the assertion in each doc comment must state
  the current, true, oracle-verified capability (deterministic path is the
  parity gate; stochastic path is functionally implemented but not yet
  C++-oracle-covered until QGP-08 lands).

### QGP-08: New C++ oracle golden + Rust test for `stochastic_rounding=true`

- **Status**: draft — BLOCKED (external precondition). No real `lightgbm==4.6` Python
  installation is available in this sandbox to regenerate/add the golden: no `.venv/` directory
  exists at the repo root, and system `python3` has no `lightgbm` module
  (`ModuleNotFoundError: No module named 'lightgbm'`)
  `[VERIFIED: LOCAL .venv/bin/python -c "import lightgbm" → No such file or directory;
  python3 -c "import lightgbm" → ModuleNotFoundError]`. No golden was fabricated. This task is
  entirely unimplemented pending a real LightGBM 4.6 Python install.
- **Rationale**: per user decision (2026-07-12), wiring all 4 params
  through in one pass requires closing the coverage gap for the
  default-`true` `stochastic_rounding` path before it becomes reachable
  from `Config::from_params`/Python. Without this, QGP-04 would make a
  never-C++-verified code path reachable by default.
- **Preconditions**: none — can be developed in parallel with QGP-01..07,
  but must land before QGP-04/QGP-05 are considered complete per this
  spec's acceptance criteria (see PLAN.md sequencing).
- **Input**: same fixed corpus/params as the existing golden
  (`crates/oracle-harness/tests/fixtures/quantized/gen_golden.py`), with
  `stochastic_rounding=True` substituted for `False`.
- **Output**: new golden artifact(s) under
  `crates/oracle-harness/tests/fixtures/quantized/` (e.g.
  `quant_binary_stochastic.pred`, sibling files following the existing
  naming/format convention) generated from real `lightgbm` 4.6, plus a new
  `#[test]` in `crates/oracle-harness/tests/quantized_parity.rs` that trains
  the Rust `Booster` with `cfg.stochastic_rounding = true` via `train_raw`
  and asserts the quantization-effect delta (Rust `|stochastic_quant -
  exact|` vs C++ `|stochastic_quant - exact|`) is in the same regime,
  mirroring `rust_quantized_train_matches_cpp`'s existing delta-gate
  methodology (`quantized_parity.rs:135-153`) — NOT a raw absolute-value
  comparison, since Rust's stochastic RNG (xorshift64,
  `gradient_discretizer.rs:29-31`) is explicitly NOT bit-matched to C++'s
  mt19937 `[VERIFIED: LOCAL gradient_discretizer.rs:26-28,69-70]`.
- **Dependencies**: a real `lightgbm` (C++ reference) Python installation
  in whatever environment generates the golden — same precondition
  `gen_golden.py` already has. `[UNVERIFIED: not confirmed present in this
  sandbox this session — see §9 Risk 2]`.
- **Behavior (Given/When/Then)**:
  - Given the new golden and a Rust `Config` with `use_quantized_grad=true,
    stochastic_rounding=true` (matching golden params), when trained via
    `train_raw` and predicted, then the Rust quantization-effect delta
    (`|stochastic_pred - exact_pred|`, mean and max) is within the same
    documented multiplicative regime as C++'s own delta — the exact
    multiplier to be chosen by the implementer following the precedent in
    `quantized_parity.rs:149-153` (`2.0×`/`3.0×` regime bounds), not
    invented ad hoc.
- **Invariants/side effects**: does not modify the existing
  `stochastic_rounding=false` golden or its test.
- **Acceptance tests**: see §6, AT-08.
- **Out of scope**: bit-exact RNG matching between Rust's xorshift64 and
  C++'s mt19937 — explicitly not a goal per the existing module doc
  (`gradient_discretizer.rs:26-28`); only the magnitude-regime parity is
  asserted.
- **Unresolved questions**: §9 Q1 (is C++ `stochastic_rounding=true`
  actually deterministic/reproducible given a fixed `seed`, so the golden
  itself is stable across regenerations?) must be answered empirically
  during implementation, not assumed.

### QGP-09: End-to-end `Config::from_params`-driven training matches the existing deterministic golden

- **Status**: implemented `[VERIFIED: LOCAL crates/oracle-harness/tests/quantized_parity.rs::rust_quantized_train_from_params_matches_cpp — cargo test -p oracle-harness --test quantized_parity: 4 passed (max_delta < 1e-2 gate)]`
- **Rationale**: QGP-01..05 prove the parsing layer in isolation; this spec
  item proves the *actual user-facing path* (string params in, trained
  model out) produces the same result as the existing
  direct-field-mutation path already proven correct by
  `rust_quantized_train_matches_cpp`. This is the acceptance criterion that
  most directly answers "is quantized-grad training now usable from
  Python/dict-style callers", which is the stated goal of this spec.
- **Preconditions**: QGP-01..05 land first.
- **Input**: the existing `quant_binary.xy.csv` corpus
  (`crates/oracle-harness/tests/fixtures/quantized/`), and a
  `HashMap<String, String>` built from the same params `gen_golden.py`
  used (`objective=binary, num_leaves=7, min_data_in_leaf=5, max_bin=63,
  learning_rate=0.1, use_quantized_grad=true, num_grad_quant_bins=128,
  stochastic_rounding=false, quant_train_renew_leaf=false, deterministic=true,
  force_row_wise=true, num_threads=1, seed=1, feature_pre_filter=false`),
  constructed via `Config::from_params`, NOT via `TrainingBuilder`/direct
  field assignment.
- **Output**: `Config::from_params(&map)` returns `Ok(cfg)` with all 4
  quantized-grad fields matching the intended values; training via
  `train_raw(&cfg, &corpus)` and predicting produces per-row predictions
  matching `quant_binary.pred` (the existing golden) within the same
  absolute-delta gate already used by `rust_quantized_train_matches_cpp`
  (`qe_max < 1e-2`, `quantized_parity.rs:130`) — reusing the existing
  golden and existing gate constants, not inventing new ones.
- **Dependencies**: QGP-01..05 (parsing + scope), existing golden fixtures.
- **Behavior (Given/When/Then)**:
  - Given the params map above passed through `Config::from_params`, when
    trained and predicted, then predictions match `quant_binary.pred`
    within `1e-2` max absolute delta — identical bar to the existing
    direct-construction test, proving the two paths are equivalent.
- **Acceptance tests**: see §6, AT-09.
- **Unresolved questions**: none.

## 6. Acceptance Scenarios

| ID | Spec | Test location (new unless noted) | Assertion |
|---|---|---|---|
| AT-01 | QGP-01 | `crates/lgbm-core/tests/config_validation.rs` | `use_quantized_grad` true/false/absent/invalid roundtrip |
| AT-02 | QGP-02 | `crates/lgbm-core/tests/config_validation.rs` | `num_grad_quant_bins` boundaries (1, 254 valid; 0, 255, -4 → `OutOfRange`) |
| AT-03 | QGP-03 | `crates/lgbm-core/tests/config_validation.rs` | `quant_train_renew_leaf` true/false/absent/invalid roundtrip |
| AT-04 | QGP-04 | `crates/lgbm-core/tests/config_validation.rs` | `stochastic_rounding` true/false/absent(default true)/invalid roundtrip |
| AT-05 | QGP-05 | `crates/lgbm-core/src/config/scope.rs` (inline `#[cfg(test)]` or new `crates/lgbm-core/tests/scope_classification.rs`, per PLAN.md) | all 4 keys present in `IN_SCOPE_PARAMS`, absent from `OUT_OF_SCOPE_PARAMS` |
| AT-06 | QGP-06 | `crates/lgbm-python/src/params.rs` (`reject_gate` test, rewritten) | all 4 keys → `Ok(())`; unrelated out-of-scope key (`num_machines`) still → `Err` |
| AT-07 | QGP-07 | grep-based check (PLAN.md Refactor step, or a lightweight `#[test]` asserting `!include_str!(...).contains("not yet implemented")` scoped to the specific doc lines) | stale strings removed from `mod.rs`/`gradient_discretizer.rs` |
| AT-08 | QGP-08 | `crates/oracle-harness/tests/quantized_parity.rs` (new `#[test] fn rust_stochastic_rounding_matches_cpp_effect()`) | Rust stochastic-quant delta vs C++ stochastic-quant delta in same regime |
| AT-09 | QGP-09 | `crates/oracle-harness/tests/quantized_parity.rs` (new `#[test] fn rust_quantized_train_from_params_matches_cpp()`) | `from_params`-driven training matches existing golden within `1e-2` |

Existing regression suites that MUST still pass unmodified (compatibility
guard, not new acceptance criteria): `cargo test -p oracle-harness --test
quantized_parity` (existing 3 tests), `cargo test -p lgbm-core` (existing
`config_validation`/`config_defaults`/`alias_resolution` suites).

## 7. Impact Scope

| Area | File(s) | Classification |
|---|---|---|
| Config parsing | `crates/lgbm-core/src/config/set.rs` | local |
| Config field docs | `crates/lgbm-core/src/config/mod.rs` | local |
| Scope classification | `crates/lgbm-core/src/config/scope.rs` | local, but consumed cross-module (see below) |
| Python binding gate | `crates/lgbm-python/src/params.rs` | cross-module (consumes `scope.rs`) |
| Training-side doc | `crates/lgbm-treelearner/src/gradient_discretizer.rs` | local (doc-only) |
| Oracle parity tests | `crates/oracle-harness/tests/quantized_parity.rs` | cross-module (exercises `lgbm-core`+`lgbm-boosting`+`lgbm` together) |
| Oracle fixtures | `crates/oracle-harness/tests/fixtures/quantized/` (new files) | local, new artifacts |
| Config drift oracle | `crates/oracle-harness/tests/config_drift.rs` | cross-module, environmental precondition only (see §9 Risk 2) — not modified by this spec, but its outcome should be re-checked once `LightGBM/` is available |
| Public API surface | none | no new public functions/types; existing `Config` fields become reachable via a previously-blocked path |
| Persistence/schema | none | N/A — no serialized format changes |
| CI/build | none anticipated | new test files run under existing `cargo test` invocations already in use |

No `external/public` or `operational` impact: no C API, no distributed/
network surface, no config/feature-flag system beyond the existing
`Config`/`scope.rs` mechanism already covered above.

## 8. Compatibility and Migration

- **Backward compatible.** No existing field, signature, or default value
  changes. A caller who never sets these 4 keys observes identical
  behavior before and after (defaults unchanged: `use_quantized_grad=false`
  means the exact path is untouched, per the existing doc comment at
  `mod.rs:108-110`).
- **Behavior change for previously-rejecting callers only**: any Python
  caller who was passing `use_quantized_grad`/`num_grad_quant_bins`/
  `quant_train_renew_leaf`/`stochastic_rounding` and receiving a
  `ValueError` will, after this change, have that call succeed and actually
  train in quantized mode. This is the intended effect of "moving from
  out-of-scope to in-scope," not a regression — but it is a user-visible
  behavior change worth calling out explicitly since it turns a previously
  hard-rejected input into an accepted one.
- **No migration steps required** — no model format, no stored config
  schema, no version gate.

## 9. Risks and Open Questions

1. **Pre-existing broken test, discovered not caused by this spec**:
   `crates/lgbm-python/src/params.rs:312-314`'s `reject_gate` sub-case for
   `linear_tree` appears to already assert incorrect behavior as a residual
   of the (separate, uncommitted) linear-tree scope migration —
   `resolve_alias("linear_tree")` returns `"linear_tree"` itself, which is
   no longer in `OUT_OF_SCOPE_PARAMS`, so `reject_unimplemented` should
   return `Ok(())`, contradicting the test's `assert!(...is_err())`
   `[VERIFIED: LOCAL crates/lgbm-python/src/params.rs:312-314; crates/lgbm-core/src/config/scope.rs (no linear_tree entry); crates/lgbm-core/src/config/alias.rs:119]`.
   This could not be confirmed by actually running the test in this
   sandbox — the `lgbm-python` crate fails to link here
   (`library not found: python3.14`, an environment issue, not a code
   issue) `[VERIFIED: LOCAL cargo build --workspace --tests output]`. **Not
   fixed by this spec** (§2 non-goals) — flagged for the user/a follow-up
   task, since fixing it is a decision about the linear-tree slice, which
   this spec does not own.
2. **`config_drift.rs` cannot run in this sandbox** (missing `LightGBM/`
   checkout) `[VERIFIED: LOCAL cargo test -p oracle-harness --test
   config_drift → 2/3 fail with ENOENT]`. This spec's QGP-05 change should
   be re-verified against `config_drift` on a machine with `LightGBM/`
   present as a final check that the canonical param names used here
   (`use_quantized_grad`, `num_grad_quant_bins`, `quant_train_renew_leaf`,
   `stochastic_rounding`) exactly match `config_auto.cpp`'s spelling — they
   are reused verbatim from the existing `Config` field doc comments
   (themselves presumably drift-checked when originally written), not
   independently re-derived from C++ source this session.
3. **QGP-08's golden-generation environment**: is a real `lightgbm==4.6`
   Python package available (via `.venv` or otherwise) in whatever
   environment will execute PLAN.md's golden-generation task? `gen_golden.py`
   assumes so. `[UNVERIFIED: not confirmed present in this sandbox this
   session]` — the planner/executor should check `.venv/bin/python -c
   "import lightgbm; print(lightgbm.__version__)"` before starting QGP-08.
4. **Is C++ `stochastic_rounding=true` training actually reproducible
   given a fixed `seed`?** If C++'s internal mt19937 sequence for
   stochastic rounding draws from a source not covered by `seed=1,
   deterministic=True`, the golden itself could be non-reproducible across
   regenerations, undermining QGP-08's premise as a stable oracle. Must be
   empirically verified (regenerate twice, diff) during QGP-08's
   implementation, not assumed from this research pass.
5. **Python-level (`.py`) acceptance test coverage** for this feature was
   not scoped in this spec because `crates/lgbm-python/python/tests/*.py`
   content was not read this session `[UNVERIFIED]`. If the user wants
   Python-level test coverage (beyond the Rust-side `params.rs` unit test
   in QGP-06), that should be a follow-up decision, not silently added or
   silently omitted.
6. **`num_grad_quant_bins` alias/typo tolerance**: no known C++ alias for
   any of these 4 keys was found in `crates/lgbm-core/src/config/alias.rs`
   this session (not exhaustively re-verified against `config_auto.cpp` —
   see Risk 2). If C++ has an alias not yet in the Rust `alias.rs`, that
   would be a separate, pre-existing gap unrelated to this spec's scope.

## 10. Traceability and Sources

- `[VERIFIED: LOCAL crates/lgbm-core/src/config/mod.rs:108-121,361-364]` —
  `Config` field definitions, doc comments, defaults.
- `[VERIFIED: LOCAL crates/lgbm-core/src/config/set.rs:254-256,535-565,809-831,865-871]`
  — parsing helper patterns (`linear_tree`/`linear_lambda` precedent),
  `get_bool`/`get_int`/`check_ge`/`check_le`/`out_of_range`.
- `[VERIFIED: LOCAL crates/lgbm-core/src/config/scope.rs:21-26,35-171,173-195]`
  — `IN_SCOPE_PARAMS`/`OUT_OF_SCOPE_PARAMS` current content and stale doc.
- `[VERIFIED: LOCAL crates/lgbm-python/src/params.rs:130-179,306-330]` —
  `reject_unimplemented`, `reject_gate` test, stale doc/error strings.
- `[VERIFIED: LOCAL crates/lgbm-boosting/src/gbdt.rs:288-303,415-472,874-902,1098-1113]`
  — `Gbdt` quantized-grad fields, builder methods, call sites (unmodified
  by this spec, cited for context only).
- `[VERIFIED: LOCAL crates/lgbm-treelearner/src/gradient_discretizer.rs:1-15,21-78]`
  — `GradientDiscretizer` struct, `new`/`new_stochastic`, stale module doc.
- `[VERIFIED: LOCAL crates/lgbm/src/booster.rs:1195-1203]` — existing
  `Config` → `Gbdt` forwarding (unmodified by this spec).
- `[VERIFIED: LOCAL crates/oracle-harness/tests/quantized_parity.rs:1-213]`
  — full existing test file (3 passing tests), delta-gate methodology
  reused by QGP-08/QGP-09.
- `[VERIFIED: LOCAL crates/oracle-harness/tests/fixtures/quantized/gen_golden.py:1-100]`
  — golden-generation script, pattern to extend for QGP-08.
- `[VERIFIED: LOCAL crates/oracle-harness/tests/config_drift.rs:1-30,135-153]`
  — mechanical drift checker, environmental precondition (Risk 2).
- `[VERIFIED: LOCAL crates/lgbm-core/tests/config_validation.rs:1-320]` —
  existing test file and naming/style conventions reused for AT-01..04.
- `[VERIFIED: LOCAL crates/lgbm-core/src/error.rs:19-40]` — `ConfigError`
  enum shape.
- `[PROJECT: .planning/PROJECT.md]`, `[PROJECT: .planning/plans/cpp-feature-parity/research.md]`
  — milestone context and original gap identification (research pass,
  2026-07-12, this session).
- User decisions (this session, 2026-07-12): scope = Gap #3 (quantized-grad
  param plumbing); tolerance = same rigor as the rest of the project
  (interpreted per §1/§5 QGP-08 as "real C++ oracle coverage, delta-gated
  per the project's own existing precedent for this specific approximate
  feature," not a literal reuse of the unrelated 1e-6 exact-path number);
  all 4 params wired in one pass, with a new stochastic-rounding oracle
  golden as a co-requirement.
