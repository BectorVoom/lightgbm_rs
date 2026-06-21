---
phase: 01-oracle-contract-foundations
plan: 03
subsystem: config
tags: [config, alias-resolution, KeyAliasTransform, SortAlias, validation, determinism, parity]

# Dependency graph
requires:
  - phase: 01-oracle-contract-foundations (plan 01-02)
    provides: "lgbm_core::Config + Config::from_params pipeline (alias resolve → seed derive → member extract + CHECK → conflict mutations); resolve_alias + ALIAS_TABLE; present() getter helper"
provides:
  - "Deterministic alias-collision resolution in from_params matching C++ ParameterAlias::KeyAliasTransform + Config::SortAlias (canonical beats alias; alias-vs-alias ties by (key.len(), key))"
  - "Empty-string-is-absent semantics for the seed lookup and the six enum reads (task, boosting, data_sample_strategy, objective, device_type, tree_learner) via present(), matching C++ Get* guards"
  - "Regression + determinism test coverage for both behaviors (the suite previously had no colliding-alias case and no empty-string-no-op case for these reads)"
affects: [binning, dataset, gbdt, tree-learner, python-bindings, oracle-harness]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Two-phase KeyAliasTransform: build canonical→winning-alias-key map with SortAlias tie-break, then fold aliases in only when the canonical is not directly set"
    - "C++ Get* parity: route every optional read through present() so empty == absent (no-op, default stands)"

key-files:
  created: []
  modified:
    - "crates/lgbm-core/src/config/set.rs"
    - "crates/lgbm-core/tests/alias_resolution.rs"
    - "crates/lgbm-core/tests/config_validation.rs"

key-decisions:
  - "SortAlias compares over alias KEY NAMES (not values), tracking the winning alias key in a canonical→key map, exactly mirroring the C++ tmp_map so the tie-break is faithful"
  - "Canonical/unknown keys are copied through verbatim first; winning aliases are folded in only when the canonical is absent — preserving canonical-beats-alias precedence without iteration-order dependence"

patterns-established:
  - "Deterministic-config invariant: no observable Config outcome may depend on HashMap iteration order (enforced by an N-run determinism test)"
  - "present()-routed reads: optional params treat empty string as unset, matching C++ count(name) > 0 && !empty()"

requirements-completed: [CFG-02, CFG-03]

# Metrics
duration: 3min
completed: 2026-06-05
---

# Phase 1 Plan 03: Config Gap Closure (Deterministic Aliases + Empty==Absent) Summary

**Replaced non-deterministic HashMap last-writer-wins alias-collision resolution with a faithful C++ KeyAliasTransform/SortAlias port, and routed the seed + six enum reads through present() so empty strings are treated as absent — closing the two confirmed BLOCKERS (CR-02, CR-01) behind Success Criterion #4.**

## Performance

- **Duration:** ~3 min
- **Started:** 2026-06-05T05:00:47Z
- **Completed:** 2026-06-05T05:03:27Z
- **Tasks:** 2 (both TDD: RED → GREEN)
- **Files modified:** 3

## Accomplishments

- **CR-02 closed:** `from_params` Stage-1 alias resolution is now a deterministic port of `ParameterAlias::KeyAliasTransform` + `Config::SortAlias` (config.h 1220-1261). A directly-set canonical key always beats any alias targeting that canonical; alias-vs-alias collisions break ties by `SortAlias (key.len(), key)` (shorter first, then lexicographic). No HashMap iteration order participates in any decision. New helpers: `sort_alias()`, `key_alias_transform()`.
- **CR-01 closed:** the `seed` lookup and all six enum reads (task, boosting, data_sample_strategy, objective, device_type, tree_learner) now go through the existing `present(&resolved, name)` helper, so an empty-string value is a no-op (default stands), matching the C++ `Get*` guard `count(name) > 0 && !empty()`. Non-empty invalid values still surface the typed `ConfigError` as before.
- **New regression + determinism coverage:** colliding-alias regression (canonical-beats-alias, SortAlias winner, equal-length lexicographic tie), an N=200 determinism test asserting a single distinct outcome, and empty-string-no-op tests for seed + all six enums plus a non-empty-invalid control. Suite grew 49 → 56 tests, all green.

## Task Commits

Each task was committed atomically (TDD: test → feat):

1. **Task 1 (CR-02) RED — failing colliding-alias determinism tests** - `fb399d1` (test)
2. **Task 1 (CR-02) GREEN — deterministic SortAlias alias-collision resolution** - `563f266` (feat)
3. **Task 2 (CR-01) RED — failing empty-string-is-no-op tests** - `3e9bdfb` (test)
4. **Task 2 (CR-01) GREEN — route seed + six enum reads through present()** - `1637de7` (feat)

_TDD tasks produced two commits each (RED test, then GREEN implementation). No REFACTOR commits were needed._

## Files Created/Modified

- `crates/lgbm-core/src/config/set.rs` - Stage-1 alias resolution rewritten to `key_alias_transform()` (deterministic canonical-priority + `sort_alias()` tie-break); seed lookup + six enum reads re-routed through `present()`.
- `crates/lgbm-core/tests/alias_resolution.rs` - Added `canonical_beats_alias_on_collision`, `alias_vs_alias_sortalias_winner`, `alias_vs_alias_equal_length_breaks_lexicographically`, `colliding_alias_resolution_is_deterministic`, plus a local `params()` helper.
- `crates/lgbm-core/tests/config_validation.rs` - Added `empty_seed_is_noop`, `empty_enum_values_are_noop`, `non_empty_invalid_enum_still_errors`.

## Decisions Made

- **Track the winning alias KEY (not its value) in a `canonical → key` map**, exactly mirroring the C++ `tmp_map`, so the `SortAlias` comparison is over key names — the only faithful way to reproduce the tie-break.
- **Copy canonical/unknown keys through first, then fold aliases in only when the canonical is absent.** This reproduces the C++ second KeyAliasTransform loop's canonical-priority rule without any iteration-order dependence.
- Warnings emitted by the C++ KeyAliasTransform (`Log::Warning(...)`) are intentionally omitted: they are diagnostics, not observable Config state, and the parity contract is over the resolved value, not stderr text.

## Deviations from Plan

None - plan executed exactly as written. Both tasks followed the TDD discipline specified (`tdd="true"`), the C++ reference under `LightGBM/include/LightGBM/config.h` was read-only, and all named acceptance commands (`cargo test -p lgbm-core alias_resolution`, `cargo test -p lgbm-core config_validation`, `cargo test --workspace`) exit 0.

## Issues Encountered

None. The RED phases failed exactly as predicted (the determinism test caught the non-deterministic `{10, 20}` outcome; the empty-string tests panicked on the `.unwrap()` of a parse error), and the GREEN implementations turned them green with no regression in the prior 49 tests.

## Threat Model Outcome

- **T-1-08 (Tampering / non-deterministic alias resolution) — mitigated:** the HashMap-iteration-order dependence is removed; the resolved value is now a pure deterministic function of the input map, proven by the N-run determinism test.
- **T-1-01 (DoS / parsing untrusted strings) — mitigated:** empty-string values now return Ok-with-default via `present()` rather than `Err`/panic; non-empty invalid values still return typed `ConfigError`. No new panic paths. The existing `fuzz_hostile_strings_never_panic` test still passes.
- No new security-relevant surface (network, auth, filesystem, schema) was introduced — no Threat Flags.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- Success Criterion #4 (config behavioral compatibility) is now fully closed: alias resolution is deterministic and matches C++ KeyAliasTransform, and empty==absent semantics match the C++ Get* helpers. `lgbm_core::Config` / `Config::from_params` remain the config bag every later crate inherits.
- Phase 01 plans (01-02, 01-03) are complete; the phase can be verified/closed and Phase 02 planning can begin.

## Self-Check: PASSED

- FOUND: crates/lgbm-core/src/config/set.rs (modified, `sort_alias` + `key_alias_transform` present; 7 `present(&resolved, ...)` call sites; 0 direct `resolved.get` for seed/six enums)
- FOUND: crates/lgbm-core/tests/alias_resolution.rs (8 tests pass)
- FOUND: crates/lgbm-core/tests/config_validation.rs (16 tests pass)
- FOUND commit fb399d1 (test, CR-02 RED)
- FOUND commit 563f266 (feat, CR-02 GREEN)
- FOUND commit 3e9bdfb (test, CR-01 RED)
- FOUND commit 1637de7 (feat, CR-01 GREEN)
- `cargo test --workspace` green: 56 passed, 0 failed.

---
*Phase: 01-oracle-contract-foundations*
*Completed: 2026-06-05*
