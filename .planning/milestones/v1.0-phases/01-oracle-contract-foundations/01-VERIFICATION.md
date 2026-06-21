---
phase: 01-oracle-contract-foundations
verified: 2026-06-05T05:20:00Z
status: verified
score: 5/5 success-criteria verified (9/9 in-scope requirements; ORA-03/ORA-04 out of scope for Phase 1)
overrides_applied: 0
re_verification:
  previous_status: gaps_found
  previous_score: 4/5
  gaps_closed:
    - "CR-02: alias-collision resolution is now deterministic (C++ KeyAliasTransform + SortAlias port); proven by an N=200 single-process determinism test"
    - "CR-01: seed lookup + six enum reads route through present() (empty == absent / no-op); proven by empty-string-no-op regression tests with a non-empty-invalid control"
  gaps_remaining: []
  regressions: []
gaps: []
deferred: []
---

# Phase 1: Oracle Contract Foundations Verification Report (Re-Verification)

**Phase Goal:** A falsifiable, f32 single-precision oracle contract (~1e-6 absolute) and the foundations (bit-exact RNG, f32 numerical strategy, config, workspace, pinned reference) that every later phase is validated against.

**Verified:** 2026-06-05
**Status:** verified
**Re-verification:** Yes — after gap closure (plan 01-03 closed the two SC#4 blockers CR-02 and CR-01)

## Re-Verification Summary

The prior report (status `gaps_found`, 4/5) had a single failing criterion — **SC#4 (config behavioral compatibility / CFG-02 + CFG-03)** — driven by two confirmed blockers. Gap-closure plan **01-03** has been executed. I independently re-verified both fixes in the actual source AND ran every named test myself.

- **CR-02 (alias-collision determinism) — CLOSED.** Stage-1 alias resolution in `crates/lgbm-core/src/config/set.rs` no longer uses HashMap last-writer-wins. It is now a faithful port of C++ `ParameterAlias::KeyAliasTransform` + `Config::SortAlias` (`LightGBM/include/LightGBM/config.h:1220-1261`): a directly-set canonical key always beats any alias, and alias-vs-alias collisions break ties by `SortAlias (key.len(), key)`. New helpers `sort_alias()` (set.rs:362) and `key_alias_transform()` (set.rs:374). A determinism test runs the colliding input 200× in one process and asserts a single distinct outcome.
- **CR-01 (empty == absent) — CLOSED.** The `seed` lookup (set.rs:73) and all six enum reads — task (90), boosting (93), data_sample_strategy (96), objective (99), device_type (104), tree_learner (107) — now route through the `present()` helper (empty-string filtered as absent), matching the C++ `Get*` guard `count(name) > 0 && !empty()`. `grep "resolved.get("` returns NONE; all seven sites use `present(&resolved, ...)`. Non-empty invalid values still error.

**Test evidence I ran (real exit codes):**

| Command | Result | Exit |
|---------|--------|------|
| `cargo build --workspace` | Finished, no errors | 0 |
| `cargo test -p lgbm-core --test alias_resolution` | 8 passed; 0 failed | 0 |
| `cargo test -p lgbm-core --test config_validation` | 16 passed; 0 failed | 0 |
| `cargo test --workspace` | **56 passed; 0 failed** across 11 binaries | 0 |

Workspace total grew 49 → 56 — exactly the 7 new regression tests (4 CR-02 + 3 CR-01). No regression in the previously-passing 49.

## Goal Achievement

### Observable Truths (ROADMAP Success Criteria)

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | Oracle harness compares Rust vs pinned deterministic C++ LightGBM 4.6 reference at ~1e-6, manifest checked in, goldens regenerate idempotently | ✓ VERIFIED | (No regression — re-confirmed green) `comparator.rs` tests 5/5 pass; `rng_parity.rs` 1/1 (replays 256 RNG + 256 Sample committed cases bit-for-bit); `config_drift.rs` 3/3. `REFERENCE_MANIFEST.md` pins commit `195c26fc`, VERSION 4.6.0.99, master seed 1592594996, 512 cases — git-tracked. (Live idempotent regen was verified in the prior report; unchanged this phase.) |
| 2 | User can run ported Random LCG and reproduce captured C++ sequence (RandInt16/32, NextFloat, NextInt, Sample across branch boundary) bit-for-bit, with u32 wraparound and f32 NextFloat | ✓ VERIFIED | (No regression) `lgbm_core` lib unit tests 14/14 pass; `rng_parity` replays 512 randomized cases bit-for-bit. `random.rs` is the faithful 1:1 port verified in the prior report. |
| 3 | Cargo workspace builds under edition 2024 with Cargo.lock + rust-toolchain.toml committed; thiserror at boundaries, anyhow at app/test | ✓ VERIFIED | (No regression) `cargo build --workspace` exit 0. Virtual manifest, 4 crates edition 2024, `Cargo.lock` + `rust-toolchain.toml` (1.95.0) tracked. `thiserror` only in lgbm-core; `anyhow` in oracle-harness + xtask. |
| 4 | Config struct accepts ~110 in-scope hyperparameters, resolves aliases via data table matching config_auto.cpp, rejects invalid combos with typed Result errors mirroring C++ Config::Set CHECK constraints | ✓ VERIFIED | **NOW PASSING.** The struct (111 fields), verbatim alias table (167 pairs, drift green), defaults, exact 6-seed derivation, 60+ typed CHECKs, and 2000+3000-case randomized validation all pass. The two prior divergences are closed: alias-collision resolution is now deterministic and matches C++ KeyAliasTransform/SortAlias (proven by the N=200 determinism test + canonical-beats-alias + SortAlias-winner + equal-length-lexicographic regressions, alias_resolution 8/8), and seed + six enums treat empty==absent via `present()` (empty_seed_is_noop, empty_enum_values_are_noop, non_empty_invalid_enum_still_errors, config_validation 16/16). |
| 5 | f32 single-precision data-type contract and ~1e-6 oracle tolerance documented as a Key Decision in PROJECT.md | ✓ VERIFIED | (No regression) PROJECT.md Key Decisions record the f32 end-to-end + ~1e-6 oracle contract; Core Value and Constraints restate it. |

**Score:** 5/5 success criteria verified.

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `crates/lgbm-core/src/config/set.rs` | from_params pipeline matching C++ Config::Set | ✓ VERIFIED | 785 lines; deterministic `key_alias_transform()` + `sort_alias()`; seed + six enums routed through `present()`; full CHECK pipeline + CheckParamConflict |
| `crates/lgbm-core/tests/alias_resolution.rs` | CR-02 regression + determinism | ✓ VERIFIED | 8 tests pass; includes `colliding_alias_resolution_is_deterministic` (N=200, single distinct outcome), `canonical_beats_alias_on_collision`, `alias_vs_alias_sortalias_winner`, `alias_vs_alias_equal_length_breaks_lexicographically` |
| `crates/lgbm-core/tests/config_validation.rs` | CR-01 empty==absent regression | ✓ VERIFIED | 16 tests pass; includes `empty_seed_is_noop`, `empty_enum_values_are_noop`, `non_empty_invalid_enum_still_errors` |
| `crates/lgbm-core/src/config/{mod,alias,scope}.rs` | flat Config + alias map + scope | ✓ VERIFIED | (No regression) config_defaults 5/5; drift green |
| `crates/lgbm-core/src/random.rs` | bit-exact Random LCG | ✓ VERIFIED | (No regression) 14 lib tests + rng_parity 512 cases |
| `crates/oracle-harness/src/comparator.rs` | abs-diff ~1e-6 comparator | ✓ VERIFIED | (No regression) comparator 5/5 (WR-04 NaN/inf caveat carried as Warning) |
| `crates/oracle-harness/fixtures/REFERENCE_MANIFEST.md` + `rng_sequence.txt` | pinned manifest + golden | ✓ VERIFIED | (No regression) git-tracked; 512 cases |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|----|--------|---------|
| `config/set.rs` | `config/alias.rs` | Stage-1 collision resolution reuses `resolve_alias` + `sort_alias` `(key.len(), key)` + canonical-priority | ✓ WIRED | `key_alias_transform` calls `resolve_alias` to classify keys; no HashMap iteration order in any decision |
| `config/set.rs` | `present()` | seed lookup + six enum reads call `present(&resolved, name)` | ✓ WIRED | 7 new `present(` call sites (lines 73, 90, 93, 96, 99, 104, 107); 0 `resolved.get(` for these reads |
| `config/set.rs` | `random.rs` | seed derivation uses Random::new + six next_short in C++ order | ✓ WIRED | (No regression) seed_derivation 4/4 |
| `rng_parity.rs` | `random.rs` + `fixtures/rng_sequence.txt` | replays committed golden bit-for-bit | ✓ WIRED | (No regression) 512 cases |
| `config_drift.rs` | `config_auto.cpp` | parses C++ source, asserts Rust coverage | ✓ WIRED | (No regression) 3/3 |

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| Workspace builds (edition 2024) | `cargo build --workspace` | Finished, exit 0 | ✓ PASS |
| Full test suite | `cargo test --workspace` | 56 passed, 0 failed, exit 0 | ✓ PASS |
| CR-02 alias-collision determinism | `cargo test -p lgbm-core --test alias_resolution` | 8/8; determinism test (N=200) green | ✓ PASS |
| CR-01 empty seed/enum no-op | `cargo test -p lgbm-core --test config_validation` | 16/16; empty_seed/empty_enum/non_empty_invalid green | ✓ PASS |

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|-------------|-------------|--------|----------|
| FND-01 | 01-01, 01-02 | Port Random PRNG bit-for-bit | ✓ SATISFIED | random.rs + rng_parity (512) + seed_derivation; 14 lib tests |
| FND-02 | 01-01 | Workspace under edition 2024 | ✓ SATISFIED | virtual workspace builds; 4 crates edition 2024 |
| FND-03 | 01-01 | f32 end-to-end matching C++ defaults | ✓ SATISFIED | types.rs ScoreT/LabelT=f32; documented contract |
| FND-04 | 01-01 | thiserror at boundaries, anyhow at app/test | ✓ SATISFIED | thiserror in lgbm-core; anyhow in harness/xtask |
| CFG-01 | 01-02 | Config struct, ~110 in-scope hyperparameters | ✓ SATISFIED | 111-field flat Config; config_defaults green; drift superset |
| CFG-02 | 01-02, 01-03 | Alias resolution as data table matching config_auto.cpp | ✓ SATISFIED | **NOW CLOSED** — table verbatim (drift green) AND collision precedence deterministically matches C++ KeyAliasTransform/SortAlias (CR-02 fix; alias_resolution 8/8 incl. N=200 determinism test) |
| CFG-03 | 01-02, 01-03 | Validation mirroring C++ Config::Set CHECKs as typed Result | ✓ SATISFIED | **NOW CLOSED** — 60+ CHECKs + randomized validation green; empty==absent via `present()` matches C++ Get* helpers (CR-01 fix); non-empty invalid still typed-errors (config_validation 16/16) |
| ORA-01 | 01-01 | Oracle harness comparing at ≤~1e-6 absolute (f32) | ✓ SATISFIED | comparator.rs; ORACLE_TOL=1e-6 (WR-04 non-finite caveat carried) |
| ORA-02 | 01-01 | Pinned C++ reference build/config manifest | ✓ SATISFIED | REFERENCE_MANIFEST.md committed; idempotent regen verified live (prior report) |

**Out of scope for Phase 1** (not failed here): ORA-03 (per-stage parity → Phase 2), ORA-04 (ROCm execution → Phase 4). REQUIREMENTS.md maps these to later phases; they are correctly excluded.

No orphaned requirements: all nine in-scope IDs are declared across the phase plans and satisfied.

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| (CR-02) `config/set.rs` | 47-64, 374-421 | Non-deterministic HashMap last-writer-wins | ✓ RESOLVED | Replaced by deterministic `key_alias_transform`/`sort_alias` — no longer present |
| (CR-01) `config/set.rs` | 73, 90-107 | `resolved.get()` bypassing empty==absent | ✓ RESOLVED | All seven reads now route through `present()` — no longer present |
| `comparator.rs` | ~68-94 | `(a-b).abs() <= tol` with no NaN/inf guard | ⚠️ Warning (carried) | NaN-vs-NaN / inf-inf silently pass (WR-04). Does NOT affect this phase's RNG-only exact comparison; close before later numeric phases lean on the comparator. |
| `random.rs` | 60-69 | `next_short`/`next_int` panic on empty range | ℹ️ Info (carried) | Internal callers always pass (0,32767); WR-05, not exercised this phase. |
| `config/set.rs` | objective `none`/`null` handling, std f64 parse, save_binary/metric | ℹ️ Info (carried) | WR-01/03/06/07 — deferrable, tracked for Phase 2 config-adjacent work. |

No `TODO`/`FIXME`/`TBD`/`XXX`/`HACK`/`PLACEHOLDER` debt markers in phase-modified source. No stubs — all files substantive and wired.

### Human Verification Required

None. All checkable behaviors were verified programmatically (build, full 56-test suite, targeted CR-02/CR-01 test files, source inspection against the C++ reference). The phase produces no visual/real-time/external-service surface.

### Gaps Summary

**No gaps. The single prior gap (SC#4) is fully closed.** Both blockers were independently re-verified in the actual source and proven by tests I ran myself:

1. **CR-02 closed:** `set.rs` Stage-1 is a faithful, deterministic port of C++ `KeyAliasTransform` + `SortAlias` — canonical beats alias, alias-vs-alias ties break by `(key.len(), key)`, and no HashMap iteration order participates in any decision. The N=200 single-process determinism test (previously the missing test that let the defect hide) is green and asserts exactly one distinct outcome. The Rust `sort_alias` and overwrite logic are logically equivalent to the C++ `SortAlias`/first-loop semantics (winner = SortAlias-smallest key in both).
2. **CR-01 closed:** the seed lookup and all six enum reads route through `present()`; `seed=""` and `objective=""` (and the other five enums) are no-ops keeping defaults, matching the C++ `Get*` guard, while non-empty invalid values still return typed errors (control test green).

The parity spine (bit-exact RNG, pinned reference, f32 contract, workspace, error layering) remains green with no regression. All 5 success criteria and all 9 in-scope requirements are satisfied. The carried Warning (WR-04 comparator NaN/inf blind spot) and Info items (WR-01/03/05/06/07) do not block this phase and are tracked for Phase 2; they did not regress.

**Recommendation:** Phase 01 is verified and ready to close. Phase 02 planning may begin. Track WR-04 for closure before later phases rely on the comparator for non-finite scores.

---

_Verified: 2026-06-05_
_Verifier: Claude (gsd-verifier)_
