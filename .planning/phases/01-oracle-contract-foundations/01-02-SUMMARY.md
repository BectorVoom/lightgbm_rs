---
phase: 01-oracle-contract-foundations
plan: 02
subsystem: config
tags: [rust, config, alias-table, seed-derivation, thiserror, drift-checker, d-14, d-12, d-11]

# Dependency graph
requires:
  - "01-01: lgbm-core::random (Random LCG), lgbm-core::types (K_EPSILON), lgbm-core::error (ConfigError)"
provides:
  - "lgbm-core::config::Config — flat struct mirroring C++ config.h defaults 1:1 (D-12)"
  - "lgbm-core::config::resolve_alias — verbatim alias_table() port (pass-through for unknowns)"
  - "lgbm-core::config::Config::from_params — four-stage Config::Set pipeline (alias -> seeds -> CHECK -> conflicts) returning typed Result"
  - "lgbm-core::config::scope::IN_SCOPE_PARAMS / OUT_OF_SCOPE_PARAMS — Open Question 1 resolution"
  - "lgbm-core::error::ConfigError::Conflict — multiclass mismatch variant"
  - "oracle-harness config_drift test — D-11 drift guard over config_auto.cpp"
affects: [binning, predict, treelearner, gbdt, objective, boosting]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Flat single Config struct (D-12) mirroring C++ Config 1:1; numeric widths = C++ widths (int->i32, double->f64), NOT the f32 score contract"
    - "Verbatim data-table port of alias_table() asserted byte-for-byte by the drift-checker"
    - "C++ Log::Fatal sites -> typed ConfigError (InvalidType/UnknownValue/OutOfRange/Conflict); never panic on hostile input (Security V5/T-1-01)"
    - "D-14 randomized-at-capture validation: deterministic in-test Random LCG drives in-scope/boundary/invalid coverage, asserting (params->outcome) parity / typed Err / zero panic"

key-files:
  created:
    - crates/lgbm-core/src/config/mod.rs
    - crates/lgbm-core/src/config/alias.rs
    - crates/lgbm-core/src/config/scope.rs
    - crates/lgbm-core/src/config/set.rs
    - crates/lgbm-core/tests/config_defaults.rs
    - crates/lgbm-core/tests/alias_resolution.rs
    - crates/lgbm-core/tests/seed_derivation.rs
    - crates/lgbm-core/tests/config_validation.rs
    - crates/oracle-harness/tests/config_drift.rs
  modified:
    - crates/lgbm-core/src/lib.rs
    - crates/lgbm-core/src/error.rs

key-decisions:
  - "objective parse treated as a closed known-set at the config boundary so objective=nonsense surfaces UnknownValue (C++ defers this to the objective factory's Log::Fatal); semantically equivalent (typed Err vs fatal) and required by the validation spec."
  - "num_class excluded from the randomized self-bound spec set — its validity is cross-parameter (objective vs num_class in CheckParamConflict), covered by a dedicated multiclass test instead."

# Metrics
duration: ~single session
completed: 2026-06-05
---

# Phase 1 Plan 02: Config Slice Summary

**Hand-ported flat Rust `Config` mirroring C++ `config.h` 1:1, a verbatim `alias_table()` port, and a `from_params` pipeline that derives the six sub-seeds via the ported `Random` LCG in exact C++ order, surfaces every in-scope `CHECK_*` constraint as a typed `Result` error across a deterministic randomized in-scope/boundary/invalid input set (D-14), applies the in-scope `CheckParamConflict` mutations, and is guarded by a drift-checker that parses the C++ source and proves the Rust tables are a superset.**

## Performance

- **Tasks:** 3 (all autonomous, no checkpoints)
- **Files created:** 9; **modified:** 2
- **Tests added:** config_defaults (5), alias_resolution (4), seed_derivation (4), config_validation (13), config_drift (3) = 29 new tests
- **Completed:** 2026-06-05

## Accomplishments

- **Flat `Config` struct (D-12, CFG-01):** one `pub struct Config` with ~110 in-scope public fields named identically to C++ `Config`, each `Default` value ported verbatim from `config.h` member initializers (num_iterations=100, learning_rate=0.1, num_leaves=31, max_bin=255, min_sum_hessian_in_leaf=1e-3, the six derived-seed defaults 1/3/4/2/5/6, etc.). Numeric widths mirror the C++ widths (int->i32, double->f64), distinct from the f32 score/gradient contract.
- **Verbatim alias table (CFG-02):** all 169 `(alias, canonical)` pairs from `config_auto.cpp::alias_table()` ported as a `&'static` slice with `resolve_alias` pass-through for canonical/unknown names. The drift-checker asserts this equals the C++ table byte-for-byte (same size, same pairs).
- **`from_params` four-stage pipeline (CFG-03, FND-01):** alias-resolve incoming keys -> derive the six sub-seeds from `Random::new(seed)` via six `next_short(0, 32767)` draws in the EXACT config.cpp order (data_random_seed, bagging_seed, drop_seed, feature_fraction_seed, objective_seed, extra_seed) -> extract + CHECK every in-scope member (GE/GT/LE/LT) -> apply in-scope `CheckParamConflict` mutations (max_depth->num_leaves=2^d, path_smooth->min_data_in_leaf=2, min_data/min_hessian->1, cuda->force_row_wise, goss->gbdt+strategy, bagging_by_query reset) and multiclass mismatch -> Err.
- **Typed error boundary (Security V5 / T-1-01):** every C++ `Log::Fatal` is a typed `ConfigError` (`InvalidType`/`UnknownValue`/`OutOfRange`/`Conflict`); added the `Conflict` variant. No code path panics on hostile input.
- **D-14 randomized validation:** `config_validation.rs` drives a deterministic randomized set (fixed in-test seed via the ported `Random` LCG, 2000 numeric cases + a 3000-case hostile-string fuzz) across 19 fields spanning all CHECK kinds, asserting `(params->outcome)` round-trip parity for in-bounds values, typed `Err` for out-of-bounds/malformed, and zero panics.
- **Drift-checker (D-11):** `oracle-harness/tests/config_drift.rs` parses `config_auto.cpp` `alias_table()` + `parameter_set()` (reading only workspace-relative in-repo paths — Security V12, no traversal, no Python, no C++ build) and asserts `IN_SCOPE_PARAMS` is a superset of every in-scope canonical and that in-scope aliases resolve correctly; out-of-scope groups (distributed/GPU/linear/quantized) are skipped via `OUT_OF_SCOPE_PARAMS`.
- **Open Question 1 resolved:** `scope.rs` enumerates the in-scope single-machine set explicitly, with each exclusion (distributed, GPU-OpenCL, linear-tree, quantized-grad) documented and deferral-justified.

## Task Commits

1. **Task 1: Flat Config struct + defaults + in-scope set + verbatim alias table** - `9738984` (feat)
2. **Task 2: from_params pipeline — seeds, CHECK validation, conflict mutations** - `3bedc7a` (feat)
3. **Task 3: config drift-checker** - `ab43691` (test)

## Files Created/Modified

- `crates/lgbm-core/src/config/mod.rs` - flat `Config` struct + `impl Default` mirroring config.h 1:1 (D-12)
- `crates/lgbm-core/src/config/alias.rs` - `ALIAS_TABLE` (verbatim) + `resolve_alias`
- `crates/lgbm-core/src/config/scope.rs` - `IN_SCOPE_PARAMS` + `OUT_OF_SCOPE_PARAMS` (Open Question 1)
- `crates/lgbm-core/src/config/set.rs` - `from_params` four-stage pipeline + typed getters + CHECK helpers
- `crates/lgbm-core/src/error.rs` - added `ConfigError::Conflict`
- `crates/lgbm-core/src/lib.rs` - `pub mod config;` + `pub use config::Config`
- `crates/lgbm-core/tests/{config_defaults,alias_resolution,seed_derivation,config_validation}.rs`
- `crates/oracle-harness/tests/config_drift.rs`

## Requirements Cross-Reference

- **CFG-01 (defaults match config.h):** `config_defaults.rs` (5 tests) + drift-checker param coverage.
- **CFG-02 (aliases match alias_table()):** `alias_resolution.rs` (4 tests) + `config_drift.rs` verbatim alias-table equality.
- **CFG-03 (invalid combos -> typed Result across randomized inputs, no panic):** `config_validation.rs` (13 tests incl. D-14 randomized + fuzz).
- **FND-01 (six sub-seeds in exact order via Random):** `seed_derivation.rs` (4 tests).

## Decisions Made

- **objective parse as closed known-set at the config boundary** — see frontmatter. C++ `ParseObjectiveAlias` passes unknown objectives through to a later objective-factory `Log::Fatal`; for the config boundary the spec requires `objective=nonsense -> Err(UnknownValue)`, so the known objective/alias set is enumerated here. Semantically equivalent (typed Err vs deferred fatal).
- **num_class randomized exclusion** — see frontmatter; covered by `multiclass_objective_requires_num_class`.

## Deviations from Plan

### Auto-fixed Issues

None affecting implementation. One test-authoring correction during Task 2:

**1. [test-correctness] num_class removed from the randomized self-bound field set**
- **Found during:** Task 2 (config_validation first run)
- **Issue:** The randomized harness initially treated `num_class` as a simple `> 0` self-bounded field, but `num_class != 1` with a non-multiclass objective is a `CheckParamConflict` failure (correct C++ behavior). The randomized case `num_class=31` correctly returned `Err(Conflict)`, which the test mis-asserted as expected-`Ok`.
- **Fix:** Excluded `num_class` from the randomized self-bound spec set (its validity is cross-parameter) and documented it; it remains covered by the dedicated `multiclass_objective_requires_num_class` test. No implementation change — the implementation was already faithful to C++.
- **Files modified:** `crates/lgbm-core/tests/config_validation.rs`
- **Commit:** folded into `3bedc7a`

**Total deviations:** 0 implementation deviations; 1 test-side correction. No scope creep.

## Issues Encountered

None. Full verification ran green:
- `cargo test -p lgbm-core --test config_defaults` — 5/5
- `cargo test -p lgbm-core --test alias_resolution` — 4/4
- `cargo test -p lgbm-core --test seed_derivation` — 4/4
- `cargo test -p lgbm-core --test config_validation` — 13/13 (D-14 randomized + fuzz, deterministic, panic-free)
- `cargo test -p oracle-harness --test config_drift` — 3/3
- `cargo test --workspace` — green (wave-2 merge gate)
- `cargo build -p lgbm-core` — no warnings

## Commit Hygiene

Per-task commits staged files explicitly by path (no `git add -A/.`). Out-of-scope paths (`LightGBM/`, `.serena/`, `AGENTS.md`, `.planning/config.json`) were never staged or committed. Post-commit deletion check across all three commits: no deletions. `LightGBM/` read-only reference untouched.

## Known Stubs

None. The `Config` is fully wired: every in-scope field parses, validates, and round-trips; no placeholder/empty-data paths.

## Next Phase Readiness

- The config half of the oracle contract is complete and drift-guarded. Every later crate can `use lgbm_core::Config` and `Config::from_params`.
- Seed derivation is parity-load-bearing and proven: downstream bagging/feature-sampling/GOSS/DART inherit faithful sub-seeds.
- No blockers.

## Self-Check: PASSED

Created files verified present on disk and commits verified in git log:
- `crates/lgbm-core/src/config/{mod,alias,scope,set}.rs` — FOUND
- `crates/lgbm-core/tests/{config_defaults,alias_resolution,seed_derivation,config_validation}.rs` — FOUND
- `crates/oracle-harness/tests/config_drift.rs` — FOUND
- Commits `9738984`, `3bedc7a`, `ab43691` — all FOUND in git log
- Test suites green: lgbm-core 14 unit + 26 integration, oracle-harness config_drift 3/3; `cargo test --workspace` all green
- No deletions across the three commits; `LightGBM/` untouched in git

---
*Phase: 01-oracle-contract-foundations*
*Completed: 2026-06-05*
