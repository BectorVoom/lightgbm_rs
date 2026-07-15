# TDD Implementation Plan: Quantized-Gradient Training Param Plumbing

Derived from `SPEC.md` (same directory, status `draft`). Every task below
references at least one specification ID (QGP-01..09). No task is marked
complete during planning — this document only orders and specifies work.

Do not implement production code as part of planning this document; it is
handed to an execution agent/session separately.

## Task graph (dependency order)

```
T-01 (QGP-01) ─┐
T-02 (QGP-02) ─┤
T-03 (QGP-03) ─┼─► T-05 (QGP-05) ─► T-06 (QGP-06) ─► T-09 (QGP-09)
T-04 (QGP-04) ─┘        │
T-07 (QGP-07) [independent, parallelizable with everything]
T-08 (QGP-08) [independent, parallelizable with everything — hard co-requirement for spec completion, not a code dependency]
```

T-01..T-04 are mutually independent (parallelizable) — each touches the
same two files (`set.rs`, `config_validation.rs`) but adds a disjoint
key/test, so they may be implemented in either order or combined into one
Green commit as long as each has its own Red test recorded first. T-05
cannot start until all of T-01..T-04's Green steps exist (it asserts the
parsing already works). T-06 depends on T-05 (asserts the Python gate
reflects the new scope). T-09 depends on T-05 (needs both parsing and scope
wired for the string-param path to actually work end-to-end). T-07 and T-08
have no code dependency on the others and may run at any time, in parallel
with T-01..T-06.

---

### T-01 — Parse `use_quantized_grad` in `Config::from_params`

**Specs**: QGP-01
**Goal**: `Config::from_params` routes the `use_quantized_grad` string key
into `cfg.use_quantized_grad: bool`.
**Prerequisites**: none.
**Files**:
- Modify: `crates/lgbm-core/src/config/set.rs`
- Modify (test): `crates/lgbm-core/tests/config_validation.rs`

**Red**
- Test name: `quantized_grad_bool_params_parse_and_default` (new `#[test]`
  fn in `config_validation.rs`, placed near `bool_coercion_matches_cpp`
  at line ~186 for stylistic proximity).
- Setup: uses the file's existing `params()` helper (`config_validation.rs:17-22`).
- Input/assertions (this Red step covers only the `use_quantized_grad`
  slice of the test; QGP-03/QGP-04 add sibling assertions to the SAME
  function per T-03/T-04 below — see note under T-03):
  ```rust
  #[test]
  fn quantized_grad_bool_params_parse_and_default() {
      // use_quantized_grad: default false, roundtrip true/false, invalid -> InvalidType
      let c = Config::from_params(&params(&[])).unwrap();
      assert!(!c.use_quantized_grad, "default use_quantized_grad must be false");

      let c = Config::from_params(&params(&[("use_quantized_grad", "true")])).unwrap();
      assert!(c.use_quantized_grad);

      let c = Config::from_params(&params(&[("use_quantized_grad", "false")])).unwrap();
      assert!(!c.use_quantized_grad);

      assert!(matches!(
          Config::from_params(&params(&[("use_quantized_grad", "maybe")])),
          Err(ConfigError::InvalidType { param, .. }) if param == "use_quantized_grad"
      ));
  }
  ```
- Expected initial failure: `use_quantized_grad` is currently unparsed, so
  every value is silently dropped — `c.use_quantized_grad` stays `false`
  regardless of input, so the `"true"` assertion fails
  (`assert!(c.use_quantized_grad)` panics); the `InvalidType` assertion
  also fails because an unparsed key never reaches `get_bool`, so no error
  is ever produced (`Config::from_params` returns `Ok` for `"maybe"` too).
  Run: `cargo test -p lgbm-core --test config_validation quantized_grad_bool_params_parse_and_default`
  and confirm it fails before writing the Green step.

**Green**
- In `crates/lgbm-core/src/config/set.rs`, add one line adjacent to the
  `linear_tree`/`linear_lambda` block (`set.rs:254-256`), mirroring its
  exact shape:
  ```rust
  get_bool(&resolved, "use_quantized_grad", &mut cfg.use_quantized_grad)?;
  ```
- Minimal — no other file changes needed for this task alone.

**Refactor**
- None required (single line, already matches house style). Re-run the
  full `config_validation` suite (not just the new test) to confirm no
  regression: `cargo test -p lgbm-core --test config_validation`.

**Validation commands**
- `cargo test -p lgbm-core --test config_validation`
- `cargo build -p lgbm-core`

**Completion evidence**: paste the passing test output for
`quantized_grad_bool_params_parse_and_default` plus the full
`config_validation` suite result (0 failures).

**Rollback**: revert the single `get_bool` line and the new test function;
no other state affected.

---

### T-02 — Parse and range-validate `num_grad_quant_bins`

**Specs**: QGP-02
**Prerequisites**: none (independent of T-01).
**Files**:
- Modify: `crates/lgbm-core/src/config/set.rs`
- Modify (test): `crates/lgbm-core/tests/config_validation.rs`

**Red**
- Test name: `num_grad_quant_bins_parses_and_validates_range` (new
  `#[test]` fn).
  ```rust
  #[test]
  fn num_grad_quant_bins_parses_and_validates_range() {
      let c = Config::from_params(&params(&[])).unwrap();
      assert_eq!(c.num_grad_quant_bins, 4, "default num_grad_quant_bins");

      let c = Config::from_params(&params(&[("num_grad_quant_bins", "128")])).unwrap();
      assert_eq!(c.num_grad_quant_bins, 128);

      // boundaries: 1 and 254 are valid (GradientDiscretizer::new's own 1..=254 contract).
      let c = Config::from_params(&params(&[("num_grad_quant_bins", "1")])).unwrap();
      assert_eq!(c.num_grad_quant_bins, 1);
      let c = Config::from_params(&params(&[("num_grad_quant_bins", "254")])).unwrap();
      assert_eq!(c.num_grad_quant_bins, 254);

      assert!(matches!(
          Config::from_params(&params(&[("num_grad_quant_bins", "0")])),
          Err(ConfigError::OutOfRange { param, .. }) if param == "num_grad_quant_bins"
      ));
      assert!(matches!(
          Config::from_params(&params(&[("num_grad_quant_bins", "255")])),
          Err(ConfigError::OutOfRange { param, .. }) if param == "num_grad_quant_bins"
      ));
      assert!(matches!(
          Config::from_params(&params(&[("num_grad_quant_bins", "-4")])),
          Err(ConfigError::OutOfRange { param, .. }) if param == "num_grad_quant_bins"
      ));
  }
  ```
- Expected initial failure: the key is currently unparsed, so
  `c.num_grad_quant_bins` stays at default `4` regardless of the `"128"`/
  `"1"`/`"254"` inputs (roundtrip assertions fail), and the three
  out-of-range inputs return `Ok` instead of `Err` (range assertions fail).
  Run and confirm failure: `cargo test -p lgbm-core --test config_validation num_grad_quant_bins_parses_and_validates_range`.

**Green**
- In `crates/lgbm-core/src/config/set.rs`, add (same Stage-3 block as T-01):
  ```rust
  get_int(&resolved, "num_grad_quant_bins", &mut cfg.num_grad_quant_bins)?;
  check_ge("num_grad_quant_bins", cfg.num_grad_quant_bins, 1)?;
  check_le("num_grad_quant_bins", cfg.num_grad_quant_bins, 254)?;
  ```
  (`check_ge`/`check_le` signatures already exist at `set.rs:809,825` —
  `fn check_ge(name: &str, value: i32, bound: i32) -> Result<(), ConfigError>`.)

**Refactor**
- None required. Re-run `cargo test -p lgbm-core --test config_validation`.

**Validation commands**
- `cargo test -p lgbm-core --test config_validation`
- `cargo build -p lgbm-core`

**Completion evidence**: passing test output for the new test + full suite.

**Rollback**: revert the 3 added lines and the new test function.

---

### T-03 — Parse `quant_train_renew_leaf`

**Specs**: QGP-03
**Prerequisites**: none (independent of T-01/T-02).
**Files**:
- Modify: `crates/lgbm-core/src/config/set.rs`
- Modify (test): `crates/lgbm-core/tests/config_validation.rs`

**Red**
- Extend the SAME test function added in T-01
  (`quantized_grad_bool_params_parse_and_default`) with a
  `quant_train_renew_leaf` block, OR add a sibling function
  `quant_train_renew_leaf_parses_and_defaults` if T-01 has already landed
  and the executor prefers not to touch a merged test — either is
  acceptable; the specification requirement is the assertions below exist
  and fail before the Green step, not their exact function grouping.
  ```rust
  let c = Config::from_params(&params(&[])).unwrap();
  assert!(!c.quant_train_renew_leaf, "default quant_train_renew_leaf must be false");

  let c = Config::from_params(&params(&[("quant_train_renew_leaf", "true")])).unwrap();
  assert!(c.quant_train_renew_leaf);

  let c = Config::from_params(&params(&[("quant_train_renew_leaf", "false")])).unwrap();
  assert!(!c.quant_train_renew_leaf);

  assert!(matches!(
      Config::from_params(&params(&[("quant_train_renew_leaf", "maybe")])),
      Err(ConfigError::InvalidType { param, .. }) if param == "quant_train_renew_leaf"
  ));
  ```
- Expected initial failure: same shape as T-01 — unparsed key means the
  `"true"` roundtrip assertion fails and the `InvalidType` assertion fails
  (silently accepted instead of rejected).

**Green**
- `crates/lgbm-core/src/config/set.rs`:
  ```rust
  get_bool(&resolved, "quant_train_renew_leaf", &mut cfg.quant_train_renew_leaf)?;
  ```

**Refactor**: none required; re-run the suite.

**Validation commands**: same as T-01/T-02.

**Completion evidence**: passing test output.

**Rollback**: revert the one added line and the added assertions.

---

### T-04 — Parse `stochastic_rounding`

**Specs**: QGP-04
**Prerequisites**: none (independent of T-01/T-02/T-03).
**Files**:
- Modify: `crates/lgbm-core/src/config/set.rs`
- Modify (test): `crates/lgbm-core/tests/config_validation.rs`

**Red**
- Same grouping note as T-03 applies.
  ```rust
  let c = Config::from_params(&params(&[])).unwrap();
  assert!(c.stochastic_rounding, "default stochastic_rounding must be true (C++ config.h default)");

  let c = Config::from_params(&params(&[("stochastic_rounding", "false")])).unwrap();
  assert!(!c.stochastic_rounding);

  let c = Config::from_params(&params(&[("stochastic_rounding", "true")])).unwrap();
  assert!(c.stochastic_rounding);

  assert!(matches!(
      Config::from_params(&params(&[("stochastic_rounding", "maybe")])),
      Err(ConfigError::InvalidType { param, .. }) if param == "stochastic_rounding"
  ));
  ```
- Expected initial failure: default-true assertion actually passes
  trivially today (field default is already `true`, key is just ignored)
  — the FAILING assertion that proves the gap is the `"false"` roundtrip
  (`assert!(!c.stochastic_rounding)` panics because the unparsed key never
  changes the field away from its default `true`). Confirm this specific
  assertion is what fails, not the default-check, before writing Green.

**Green**
- `crates/lgbm-core/src/config/set.rs`:
  ```rust
  get_bool(&resolved, "stochastic_rounding", &mut cfg.stochastic_rounding)?;
  ```

**Refactor**: none required; re-run the suite.

**Validation commands**: same as T-01/T-02.

**Completion evidence**: passing test output.

**Rollback**: revert the one added line and the added assertions.

---

### T-05 — Reclassify the 4 keys from `OUT_OF_SCOPE_PARAMS` to `IN_SCOPE_PARAMS`

**Specs**: QGP-05
**Prerequisites**: T-01, T-02, T-03, T-04 (Green steps landed — parsing
must exist before scope claims it does).
**Files**:
- Modify: `crates/lgbm-core/src/config/scope.rs`

**Red**
- New test, e.g. in a new `crates/lgbm-core/tests/scope_classification.rs`
  (no existing dedicated scope test file was found this session — creating
  one keeps this assertion out of `config_validation.rs`, which tests
  `from_params` behavior, not the static classification arrays):
  ```rust
  use lgbm_core::config::scope::{IN_SCOPE_PARAMS, OUT_OF_SCOPE_PARAMS};

  #[test]
  fn quantized_grad_params_are_in_scope() {
      for key in [
          "use_quantized_grad",
          "num_grad_quant_bins",
          "quant_train_renew_leaf",
          "stochastic_rounding",
      ] {
          assert!(
              IN_SCOPE_PARAMS.contains(&key),
              "{key} must be listed in IN_SCOPE_PARAMS"
          );
          assert!(
              !OUT_OF_SCOPE_PARAMS.contains(&key),
              "{key} must NOT remain in OUT_OF_SCOPE_PARAMS"
          );
      }
  }
  ```
- Expected initial failure: all 4 `IN_SCOPE_PARAMS.contains` assertions
  fail (keys absent), confirmed by direct read of `scope.rs:35-171` this
  session (no quantized-grad entries present).
  Run: `cargo test -p lgbm-core --test scope_classification` and confirm
  failure first.

**Green**
- In `crates/lgbm-core/src/config/scope.rs`:
  - Remove the 4 lines from `OUT_OF_SCOPE_PARAMS` (currently lines
    190-194: `"use_quantized_grad", "num_grad_quant_bins",
    "quant_train_renew_leaf", "stochastic_rounding",` plus the `//
    quantized-grad` comment on line 190).
  - Add the same 4 canonical names to `IN_SCOPE_PARAMS` (near the
    `linear_tree`/`linear_lambda` entries at lines 95-96, for grouping
    continuity), e.g.:
    ```rust
    "use_quantized_grad",
    "num_grad_quant_bins",
    "quant_train_renew_leaf",
    "stochastic_rounding",
    ```
  - Update the module doc comment (lines 21-26, part of the same doc block
    already flagged stale for `linear_tree`) to stop describing
    quantized-grad as "deferred"/"later-phase" — e.g. remove or rewrite
    the "Quantized-gradient training (deferred)" bullet. Also update the
    `OUT_OF_SCOPE_PARAMS` doc header at lines 173-177 ("Grouped exactly as
    in the module docs: distributed, GPU-OpenCL, linear-tree,
    quantized-grad") to drop "quantized-grad" from that list (the
    `linear-tree` staleness there is the pre-existing, out-of-scope
    Risk 1 — leave that word alone unless the user asks, per SPEC.md §2).

**Refactor**
- None beyond the doc text above (it IS the refactor — text-only, no
  behavior change). Re-run the full `lgbm-core` test suite.

**Validation commands**
- `cargo test -p lgbm-core --test scope_classification`
- `cargo test -p lgbm-core` (full crate suite, regression guard)
- `cargo build -p lgbm-core`
- **Environment-permitting only** (see SPEC.md §9 Risk 2 — needs
  `LightGBM/` checked out, absent in this planning session's sandbox):
  `cargo test -p oracle-harness --test config_drift` — re-run this on a
  machine with the C++ tree present to confirm the 4 canonical names
  exactly match `config_auto.cpp`'s spelling before treating this task as
  fully verified.

**Completion evidence**: passing `scope_classification` test output;
passing full `lgbm-core` suite; note explicitly whether `config_drift` was
or was not run (and why) in the completion report.

**Rollback**: revert `scope.rs` to restore the 4 entries in
`OUT_OF_SCOPE_PARAMS` and remove them from `IN_SCOPE_PARAMS`; revert the
doc text and the new test file.

---

### T-06 — Python `reject_unimplemented` no longer rejects the 4 keys; fix `reject_gate` and stale wording

**Specs**: QGP-06
**Prerequisites**: T-05.
**Files**:
- Modify: `crates/lgbm-python/src/params.rs`

**Red**
- Rewrite the `reject_gate` test (`params.rs:306-330`): remove the
  `use_quantized_grad` sub-case that currently asserts rejection
  (`params.rs:316-318`, `assert!(reject_unimplemented(&m).is_err())`) and
  replace it with assertions of the NEW, correct behavior:
  ```rust
  // use_quantized_grad and its 3 siblings are now IN SCOPE — must be accepted.
  for key in [
      "use_quantized_grad",
      "num_grad_quant_bins",
      "quant_train_renew_leaf",
      "stochastic_rounding",
  ] {
      let mut m = HashMap::new();
      m.insert(key.to_string(), "true".to_string());
      assert!(
          reject_unimplemented(&m).is_ok(),
          "{key} must no longer be rejected"
      );
  }
  // Regression guard: an unrelated still-out-of-scope key is still rejected.
  let mut m = HashMap::new();
  m.insert("num_machines".to_string(), "2".to_string());
  assert!(reject_unimplemented(&m).is_err());
  ```
- Expected initial failure (BEFORE T-05's Green step, or if T-05 is
  reverted): the loop's `assert!(...is_ok())` fails for all 4 keys because
  `OUT_OF_SCOPE_PARAMS` still contains them. Since this task's
  Prerequisite is T-05 already landed, run this Red step by TEMPORARILY
  reverting T-05's `scope.rs` change (or, if executing tasks strictly in
  order, simply note that this Red step's failure was already
  demonstrated by T-05's own Red step — do not skip writing this test,
  since `reject_unimplemented`'s behavior is a distinct code path from the
  raw array containment checked in T-05, even though they're driven by the
  same array).
- Note the pre-existing `linear_tree` sub-case
  (`params.rs:312-314`) is LEFT AS-IS per SPEC.md §2/§9 Risk 1 — do not
  fix it as part of this task; if it currently fails to compile/pass for
  unrelated reasons, record that as a pre-existing observation in the
  completion evidence, not as a regression introduced here.

**Green**
- No production-code change is required beyond what T-05 already did
  (`reject_unimplemented`'s logic already correctly derives from
  `OUT_OF_SCOPE_PARAMS`, `params.rs:150-159`) — this task's "Green" is
  purely the test rewrite above becoming green once T-05 is in place, PLUS
  the following doc/string fixes:
  - `params.rs:138` doc comment: change `"(distributed / GPU-OpenCL /
    linear-tree / quantized-grad — referenced..."` to drop
    "quantized-grad" (leave "linear-tree" per Risk 1 scoping, unless a
    follow-up task also cleans that one).
  - `params.rs:156-157` error string: change `"out-of-scope for v1:
    distributed / GPU-OpenCL / linear-tree / quantized-grad"` to drop
    "quantized-grad" from the listed groups (same Risk-1 caveat for
    "linear-tree").

**Refactor**
- None beyond the doc/string edits above. Re-run the full `lgbm-python`
  test suite if the environment supports linking it (see Validation
  commands note below).

**Validation commands**
- `cargo test -p lgbm-python` — **environment caveat**: this crate failed
  to link in the planning session's sandbox (`library not found:
  python3.14`, `mold`/`cc` failure) `[VERIFIED: LOCAL cargo build
  --workspace --tests output]`; this is an environment issue unrelated to
  this change. The executor must confirm a working Python dev environment
  before treating this task's tests as verified, or explicitly document
  that they could only be verified by static/manual reasoning (as this
  planning session's research did) if the environment is unavailable.
- `cargo build -p lgbm-python` (compile-only check, may also be blocked by
  the same linker issue — note in completion evidence either way).

**Completion evidence**: passing `reject_gate` test output (or, if
blocked by the environment issue, a clear note of the blocker plus the
static verification performed instead — do not silently claim success).

**Rollback**: revert `params.rs` test and doc/string changes.

---

### T-07 — Correct stale "not yet implemented" doc comments

**Specs**: QGP-07
**Prerequisites**: none (parallelizable with everything else — these
comments are already false today, independent of this plan's other
changes).
**Files**:
- Modify: `crates/lgbm-core/src/config/mod.rs` (lines 116, 120)
- Modify: `crates/lgbm-treelearner/src/gradient_discretizer.rs` (lines 9-15)

**Red**
- This is a documentation-only change; per PLAN.md's Red/Green/Refactor
  contract, express the Red step as a mechanical grep-based check rather
  than a runtime test (there is no runtime behavior to assert):
  ```
  ! grep -n "Not yet implemented" crates/lgbm-core/src/config/mod.rs | grep -q "quant_train_renew_leaf"
  ! grep -n "not yet implemented" crates/lgbm-core/src/config/mod.rs | grep -q "stochastic_rounding"
  ! grep -n "deferred" crates/lgbm-treelearner/src/gradient_discretizer.rs | grep -q "Stochastic rounding"
  ```
  Expected initial failure: all three greps currently MATCH (the stale
  text is present), so the negated (`!`) checks currently fail — confirmed
  by direct read this session (`mod.rs:116,120`;
  `gradient_discretizer.rs:14-15`).

**Green**
- `crates/lgbm-core/src/config/mod.rs:116`: replace
  `"...Used only if `use_quantized_grad`. Not yet implemented."` with
  wording reflecting the true state, e.g. `"...Used only if
  `use_quantized_grad`. Recomputes leaf outputs from the original
  (non-quantized) gradients; C++-oracle-verified
  (see crates/oracle-harness/tests/quantized_parity.rs)."`
- `crates/lgbm-core/src/config/mod.rs:120`: replace `"...The Rust
  quantized path currently supports DETERMINISTIC rounding only
  (parity-tractable); stochastic rounding is not yet implemented."` with
  wording reflecting the true state, e.g. `"...Both deterministic and
  stochastic rounding are implemented (GradientDiscretizer); the
  deterministic path (`false`) is the C++ parity gate, stochastic
  (`true`, the default) is C++-oracle-verified via a magnitude-regime
  delta gate, not bit-exact RNG matching."` — the exact final wording is
  an implementer judgment call within house style; the specification's
  requirement is only that "not yet implemented" no longer appears next
  to a feature that is implemented.
- `crates/lgbm-treelearner/src/gradient_discretizer.rs:14-15`: replace
  `"Stochastic rounding + `quant_train_renew_leaf` are deferred — they
  need RNG-matching, a separate parity problem. This module is
  deterministic-only."` with wording noting both are implemented, the
  deterministic path is the parity gate, and stochastic rounding is
  intentionally not bit-matched to C++'s mt19937 (cite
  `new_stochastic`/`stochastic` field doc at lines 25-31 for the existing
  accurate framing to reuse/align with, avoiding duplicate/contradictory
  claims within the same file).

**Refactor**
- None. Re-run the grep checks (now expecting a clean/non-matching
  result) and `cargo doc -p lgbm-core -p lgbm-treelearner --no-deps` to
  confirm doc comments still compile (rustdoc syntax stays valid).

**Validation commands**
- The grep checks above (inverted — now expect no match).
- `cargo doc -p lgbm-core -p lgbm-treelearner --no-deps`
- `cargo test -p lgbm-core -p lgbm-treelearner` (regression guard — no
  runtime behavior should change from a doc-only edit).

**Completion evidence**: grep output showing the stale strings are gone;
clean `cargo doc` build.

**Rollback**: revert the 3 doc-comment edits.

---

### T-08 — New C++ oracle golden + Rust test for `stochastic_rounding=true`

**Specs**: QGP-08
**Prerequisites**: none (parallelizable with everything else). Requires a
real `lightgbm==4.6` Python installation in the execution environment —
**verify this first**: `.venv/bin/python -c "import lightgbm;
print(lightgbm.__version__)"` (expect `4.6.x`). If unavailable, this task
is BLOCKED and must be reported as such, not silently skipped or faked.

**Files**:
- New: `crates/oracle-harness/tests/fixtures/quantized/gen_golden_stochastic.py`
  (or extend the existing `gen_golden.py` with a second params block — the
  executor should pick whichever keeps the file most readable; either
  satisfies the spec)
- New: golden artifacts under
  `crates/oracle-harness/tests/fixtures/quantized/` (e.g.
  `quant_binary_stochastic.pred`, `quant_binary_stochastic.pred_exact` if
  a fresh exact-mode baseline is needed, following the existing
  `.pred`/`.xy.csv` plain-text convention from `gen_golden.py:65-96`)
- Modify: `crates/oracle-harness/tests/quantized_parity.rs` (new test)

**Red**
- New test `rust_stochastic_rounding_matches_cpp_effect` in
  `quantized_parity.rs`, structurally mirroring
  `rust_quantized_train_matches_cpp` (lines 72-154) and
  `rust_quant_renew_leaf_matches_cpp_effect` (lines 160-204):
  ```rust
  #[test]
  fn rust_stochastic_rounding_matches_cpp_effect() {
      use lgbm::{train_raw, Config, RawCorpus, TrainingBuilder};

      let (rows, labels) = read_xy(); // reuse existing quant_binary.xy.csv corpus
      let mut cfg: Config = TrainingBuilder::new()
          .objective("binary").num_iterations(10).learning_rate(0.1)
          .num_leaves(7).min_data_in_leaf(5).seed(1).deterministic(true)
          .build().unwrap();
      cfg.max_bin = 63;
      cfg.num_threads = 1;
      cfg.force_row_wise = true;
      cfg.feature_pre_filter = false;
      cfg.use_quantized_grad = true;
      cfg.num_grad_quant_bins = 128;
      cfg.stochastic_rounding = true; // <-- the path under test

      let pred_stoch: Vec<f32> = train_raw(&cfg, &RawCorpus::new(rows.clone(), labels.clone()))
          .unwrap().predict(&rows).iter().map(|r| r[0]).collect();

      let mut cfg_exact = cfg.clone();
      cfg_exact.use_quantized_grad = false;
      let pred_exact: Vec<f32> = train_raw(&cfg_exact, &RawCorpus::new(rows.clone(), labels))
          .unwrap().predict(&rows).iter().map(|r| r[0]).collect();

      let cpp_stoch = read_named_preds("quant_binary_stochastic.pred");
      let cpp_exact = read_named_preds("quant_binary.pred_exact"); // reuse existing exact baseline

      let delta_f32 = |a: &[f32], b: &[f32]| -> (f64, f64) { /* same pattern as existing rdelta, lines 135-138 */
          let d: Vec<f64> = a.iter().zip(b.iter()).map(|(x, y)| f64::from((*x - *y).abs())).collect();
          (d.iter().cloned().fold(0.0, f64::max), d.iter().sum::<f64>() / d.len() as f64)
      };
      let delta_f64 = |a: &[f64], b: &[f64]| -> (f64, f64) {
          let d: Vec<f64> = a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).collect();
          (d.iter().cloned().fold(0.0, f64::max), d.iter().sum::<f64>() / d.len() as f64)
      };

      let (rq_max, rq_mean) = delta_f32(&pred_stoch, &pred_exact);
      let (cq_max, cq_mean) = delta_f64(&cpp_stoch, &cpp_exact);
      eprintln!("STOCHASTIC EFFECT  Rust |stoch-exact|: max={rq_max:.3e} mean={rq_mean:.3e}");
      eprintln!("STOCHASTIC EFFECT  C++  |stoch-exact|: max={cq_max:.3e} mean={cq_mean:.3e}");

      // Same regime-check pattern as rust_quantized_train_matches_cpp (lines 149-153):
      // exact multiplier bounds are an implementer judgment call, seeded from that precedent.
      assert!(
          rq_mean < 2.0 * cq_mean.max(1e-4) && rq_max < 3.0 * cq_max.max(1e-3),
          "Rust stochastic-rounding effect (max={rq_max:.3e} mean={rq_mean:.3e}) not in \
           C++'s regime (max={cq_max:.3e} mean={cq_mean:.3e})"
      );
  }
  ```
- Expected initial failure: `quant_binary_stochastic.pred` does not exist
  yet, so `read_named_preds` returns an empty vec (per its
  `unwrap_or_default()` at `quantized_parity.rs:208`), and the zip/delta
  computation degenerates (empty `cpp_stoch` → `d` empty → `.fold` returns
  `0.0`/mean division by zero → `NaN`), causing the final assertion to
  fail. Confirm this exact failure mode when running the test before the
  golden file exists.

**Green**
1. Write and run the golden generator (new/extended Python script) against
   real `lightgbm==4.6`, using the SAME fixed corpus/seed as
   `gen_golden.py` (`rng = np.random.default_rng(20260615)`, `N=512, F=6`,
   same `logit`/`y` construction) with `stochastic_rounding=True`
   substituted for `False` in the params dict, and an unquantized
   `exact_params` companion run (or reuse the existing
   `quant_binary.pred_exact` if the exact-mode model is identical given
   the params don't change — confirm this before assuming reuse is valid,
   since `exact_params` in `gen_golden.py:60-63` already strips all 4
   quantized keys, making it independent of `stochastic_rounding`, so
   reuse of `quant_binary.pred_exact` IS valid — no need to regenerate).
2. **Empirically verify reproducibility** (SPEC.md §9 Risk 4): run the
   generator twice and diff the two `quant_binary_stochastic.pred`
   outputs. If they differ, C++'s stochastic path is not reproducible
   under `seed=1, deterministic=True` alone, and this task must be
   escalated back to the user/spec-owner before proceeding — do not ship
   a non-reproducible golden silently.
3. Add `quant_binary_stochastic.pred` (and any needed companions) to
   `crates/oracle-harness/tests/fixtures/quantized/`.
4. Add the test from the Red step to `quantized_parity.rs`.

**Refactor**
- If the delta-computation closures duplicate logic already in
  `rust_quantized_train_matches_cpp`'s inline `delta`/`rdelta`/`cdelta`
  closures (lines 112-144), consider factoring a shared helper — but only
  if it doesn't reduce clarity; the existing file currently duplicates this
  pattern per-test already (compare lines 112-120 vs 135-138), so matching
  that existing convention is acceptable and not a required refactor.

**Validation commands**
- `cargo test -p oracle-harness --test quantized_parity` (all 4 tests
  now, including the 3 pre-existing ones — regression guard)
- Golden-generation reproducibility check (§Green step 2 above) — record
  the diff result (empty = reproducible) in completion evidence.

**Completion evidence**: passing test output for the new test + all 3
pre-existing tests; reproducibility diff result; the generator script
diff/addition.

**Rollback**: remove the new fixture files, the new test, and the
generator script addition.

---

### T-09 — End-to-end `Config::from_params`-driven training matches the existing golden

**Specs**: QGP-09
**Prerequisites**: T-05 (parsing + scope both wired).
**Files**:
- Modify: `crates/oracle-harness/tests/quantized_parity.rs` (new test)

**Red**
- New test `rust_quantized_train_from_params_matches_cpp`:
  ```rust
  #[test]
  fn rust_quantized_train_from_params_matches_cpp() {
      use std::collections::HashMap;
      use lgbm::{train_raw, Config, RawCorpus};

      let (rows, labels) = read_xy();
      let golden = read_preds();

      let mut m: HashMap<String, String> = HashMap::new();
      for (k, v) in [
          ("objective", "binary"),
          ("num_leaves", "7"),
          ("min_data_in_leaf", "5"),
          ("max_bin", "63"),
          ("learning_rate", "0.1"),
          ("use_quantized_grad", "true"),
          ("num_grad_quant_bins", "128"),
          ("stochastic_rounding", "false"),
          ("quant_train_renew_leaf", "false"),
          ("deterministic", "true"),
          ("force_row_wise", "true"),
          ("num_threads", "1"),
          ("seed", "1"),
          ("feature_pre_filter", "false"),
          ("num_iterations", "10"),
      ] {
          m.insert(k.to_string(), v.to_string());
      }

      let cfg = Config::from_params(&m).expect("from_params must accept all quantized-grad keys");
      assert!(cfg.use_quantized_grad);
      assert_eq!(cfg.num_grad_quant_bins, 128);
      assert!(!cfg.stochastic_rounding);
      assert!(!cfg.quant_train_renew_leaf);

      let booster = train_raw(&cfg, &RawCorpus::new(rows.clone(), labels)).expect("train via from_params config");
      let pred: Vec<f32> = booster.predict(&rows).iter().map(|r| r[0]).collect();

      let max_delta = pred.iter().zip(golden.iter())
          .map(|(a, b)| (f64::from(*a) - b).abs())
          .fold(0.0_f64, f64::max);
      assert!(max_delta < 1e-2, "from_params-driven training diverged from C++ golden: max={max_delta:.3e}");
  }
  ```
- Expected initial failure (run BEFORE T-05 lands, or with T-05 reverted):
  `Config::from_params(&m)` returns `Err(ConfigError::...)` is NOT actually
  the failure mode (unknown-but-not-out-of-scope keys just warn, per D-06)
  — the real failure is that `cfg.use_quantized_grad`/
  `cfg.num_grad_quant_bins`/etc. all silently stay at their DEFAULTS
  (`false`/`4`/`false`/`true`) because T-01..T-04 haven't landed, so the
  `assert!(cfg.use_quantized_grad)` (and `num_grad_quant_bins == 128`)
  assertions fail immediately, before training even runs. Confirm this
  specific failure point.

**Green**
- No production-code change — this test passes automatically once T-01
  through T-05 have landed. This task's "Green" step is verifying that is
  actually true by running the test after T-05.

**Refactor**
- None required.

**Validation commands**
- `cargo test -p oracle-harness --test quantized_parity rust_quantized_train_from_params_matches_cpp`
- `cargo test -p oracle-harness --test quantized_parity` (full file, regression guard)

**Completion evidence**: passing test output.

**Rollback**: remove the new test.

---

## Cross-cutting validation (run after all tasks land)

- `cargo test -p lgbm-core` — full crate suite (T-01..T-05, T-07 touch this crate).
- `cargo test -p oracle-harness --test quantized_parity` — all tests (T-08, T-09).
- `cargo test -p oracle-harness --test config_drift` — **environment-permitting only** (needs `LightGBM/`, SPEC.md §9 Risk 2); run if available, otherwise explicitly note it was skipped and why.
- `cargo test -p lgbm-python` — **environment-permitting only** (T-06; sandbox linker issue noted in SPEC.md, unrelated to this change).
- `cargo build --workspace --tests` — confirm no unrelated regression.
- `cargo doc -p lgbm-core -p lgbm-treelearner --no-deps` — confirm T-07's doc edits compile.

## Completion criteria (mirrors SPEC.md §6 Acceptance Scenarios)

This plan is fully executed only when all of AT-01 through AT-09 pass, the
pre-existing `quantized_parity` 3-test suite still passes unmodified, and
the completion evidence for each task explicitly states whether any
environment-permitting-only command (config_drift, lgbm-python) could
actually be run — silence on that point is not acceptable per SPEC.md's
evidence-labeling requirement.
