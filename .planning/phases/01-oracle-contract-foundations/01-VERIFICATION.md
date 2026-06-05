---
phase: 01-oracle-contract-foundations
verified: 2026-06-05T00:00:00Z
status: gaps_found
score: 4/5 success-criteria verified (8/9 requirements; SC#4/CFG-02 config behavioral-compat fails)
overrides_applied: 0
gaps:
  - truth: "SC#4: A config struct accepts the ~110 in-scope hyperparameters, resolves aliases via a data table matching config_auto.cpp, and rejects invalid combos with typed Result errors mirroring C++ Config::Set CHECK constraints."
    status: partial
    reason: >-
      The struct (111 fields), verbatim alias table (167 pairs), defaults, drift-checker,
      typed errors, seed derivation, and randomized validation are all real and substantive
      and pass. BUT two confirmed behavioral divergences from C++ Config::Set break the
      "matching config_auto.cpp / mirroring C++ Config::Set" half of the criterion. CR-02 is
      empirically NON-DETERMINISTIC within a single process run, directly violating the
      project's non-negotiable bit-reproducibility contract — and this config layer is the
      foundation every later phase inherits.
    artifacts:
      - path: "crates/lgbm-core/src/config/set.rs"
        issue: >-
          CR-02 (set.rs:52-56): Stage-1 alias resolution is last-writer-wins over HashMap
          iteration order. Empirically demonstrated: from_params({"num_iterations":"10",
          "n_estimators":"20"}) returns BOTH 10 and 20 across 50 invocations in one process.
          C++ KeyAliasTransform (config.h:1225-1262) is deterministic: a directly-set canonical
          always wins over any alias, and alias-vs-alias ties break by SortAlias (shorter, then
          lexicographic). The committed unit tests never exercise a colliding-alias input, so
          the suite is green despite the defect.
      - path: "crates/lgbm-core/src/config/set.rs"
        issue: >-
          CR-01 (set.rs:62, 77-96): the `seed` lookup and all six enum parses
          (task/boosting/data_sample_strategy/objective/device_type/tree_learner) read
          resolved.get(...) directly instead of the empty-string-filtering present() helper used
          by every member getter. Empirically: seed="" -> Err(InvalidType), objective="" ->
          Err(UnknownValue), whereas C++ GetInt/GetString/etc. (config.h:1165-1219) treat an
          empty value as ABSENT (no-op, default stands). num_leaves="" correctly returns Ok.
          CLI/Python bindings routinely pass empty strings for unset params, so this rejects
          input the reference accepts.
    missing:
      - "Reproduce C++ KeyAliasTransform precedence in Stage 1: direct-canonical beats any alias; break alias-vs-alias ties by (key.len(), key) per SortAlias — not HashMap order. Add a colliding-alias regression + determinism test (assert single distinct outcome across N runs)."
      - "Route the seed lookup and the six enum parses through present() (empty == absent), matching all C++ Get* helpers. Add empty-string-is-no-op cases to config_validation.rs for seed and each enum field."
deferred: []
---

# Phase 1: Oracle Contract Foundations Verification Report

**Phase Goal:** A falsifiable, f32 single-precision oracle contract (~1e-6 absolute) and the foundations (bit-exact RNG, f32 numerical strategy, config, workspace, pinned reference) that every later phase is validated against.

**Verified:** 2026-06-05
**Status:** gaps_found
**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths (ROADMAP Success Criteria)

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | Oracle harness compares Rust vs pinned deterministic C++ LightGBM 4.6 reference at ~1e-6, manifest checked in, goldens regenerate idempotently | ✓ VERIFIED | `comparator.rs` (`abs_diff_within`, `compare_within`, `ORACLE_TOL=1e-6`) + `rng_parity.rs` replays 256 RNG + 256 Sample committed cases bit-for-bit (test printed "replayed 256 RNG cases + 256 Sample cases — all bit-for-bit"). `REFERENCE_MANIFEST.md` pins commit `195c26fc...`, VERSION 4.6.0.99, deterministic flags, master seed 1592594996, 512 cases — both git-tracked. Ran `cargo run -p xtask -- regen` live (cmake 3.28.3 + g++ 13.3.0 present): regenerated and `git diff --stat crates/oracle-harness/fixtures/` was EMPTY → idempotent. |
| 2 | User can run ported Random LCG and reproduce captured C++ sequence (RandInt16/32, NextFloat, NextInt, Sample across branch boundary) bit-for-bit, with u32 wraparound and f32 NextFloat | ✓ VERIFIED | `random.rs` is a faithful 1:1 port of `LightGBM/include/LightGBM/utils/random.h`: `wrapping_mul(214013).wrapping_add(2531011)`, `(x>>16)&0x7FFF` / `x&0x7FFFFFFF`, `NextFloat = RandInt16 as f32 / 32768.0_f32`, four-way `Sample` branch with `BTreeSet` (= std::set ordering) and `K > N/log2(K)` computed in f64, streaming compare `(next_float() as f64) < prob` mirroring C++ float→double promotion. `rng_parity` proves exact-bit equality vs the committed C++ golden over 512 randomized cases. 14 lgbm-core unit tests pass. |
| 3 | Cargo workspace builds under edition 2024 with Cargo.lock + rust-toolchain.toml committed; thiserror at boundaries, anyhow at app/test | ✓ VERIFIED | Root `Cargo.toml` is a virtual manifest (`[workspace]`, resolver "3", no `[package]`); all 4 crates use `edition.workspace = true` (= 2024). `cargo build --workspace` succeeds. `Cargo.lock` + `rust-toolchain.toml` (channel 1.95.0) git-tracked. `src/` removed. `thiserror` only in `lgbm-core/error.rs`; `anyhow` in oracle-harness + xtask. |
| 4 | Config struct accepts ~110 in-scope hyperparameters, resolves aliases via data table matching config_auto.cpp, rejects invalid combos with typed Result errors mirroring C++ Config::Set CHECK constraints | ✗ FAILED | Struct (111 fields), verbatim alias table (167 pairs, drift-checker green), defaults, seed derivation (exact 6-seed C++ order), 60+ typed CHECK constraints, and 2000+3000-case D-14 randomized validation are all real and pass. BUT two confirmed divergences from C++ `Config::Set`: CR-02 alias-collision resolution is NON-DETERMINISTIC (empirically returns both 10 and 20 for identical colliding input in one run; C++ is deterministic via SortAlias + canonical-priority), and CR-01 the seed + six enum fields reject empty-string values that C++ treats as absent/no-op. See Gaps. |
| 5 | f32 single-precision data-type contract and ~1e-6 oracle tolerance documented as a Key Decision in PROJECT.md | ✓ VERIFIED | PROJECT.md Key Decisions table lines 74-75: "`f32` single-precision end-to-end + ~1e-6 oracle on **every** backend incl. ROCm — Decided 2026-06-05" and "Standard `f32` accumulations (drop integer-quantized histograms)". Core Value (line 9) and Constraints (line 63) restate the f32 / ~1e-6 contract. |

**Score:** 4/5 success criteria verified.

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `Cargo.toml` | virtual workspace, no root package | ✓ VERIFIED | `[workspace]` resolver "3", members = crates/* + xtask; no `[package]` |
| `rust-toolchain.toml` | pinned channel for edition 2024 | ✓ VERIFIED | channel 1.95.0 + rustfmt/clippy; git-tracked |
| `crates/lgbm-core/src/random.rs` | bit-exact Random LCG (u32 wrapping) | ✓ VERIFIED | 230 lines; faithful full-class port; 10 unit tests |
| `crates/lgbm-core/src/types.rs` | f32 ScoreT/LabelT + meta.h constants | ✓ VERIFIED | ScoreT/LabelT=f32, K_EPSILON=1e-15, etc.; 2 tests |
| `crates/lgbm-core/src/error.rs` | thiserror domain enums | ✓ VERIFIED | CoreError + ConfigError (InvalidType/UnknownValue/OutOfRange/Conflict); thiserror derive |
| `crates/lgbm-compute/src/lib.rs` | CubeCL Backend trait skeleton (no kernels) | ✓ VERIFIED | `pub trait Backend { type Runtime; }`, kernel-free, CMP-01 seam documented |
| `crates/oracle-harness/src/comparator.rs` | abs-diff ~1e-6 comparator | ✓ VERIFIED (see WR-04 caveat) | `abs_diff_within`/`compare_within`/`ORACLE_TOL`/`Mismatch`; first-offending-index reported |
| `crates/oracle-harness/fixtures/rng_sequence.txt` | committed randomized C++ golden set | ✓ VERIFIED | 1.1 MB, git-tracked, 256 RNG + 256 Sample, master seed 1592594996 |
| `crates/oracle-harness/fixtures/REFERENCE_MANIFEST.md` | pinned manifest (commit+flags+seed+count) | ✓ VERIFIED | commit 195c26fc, 4.6.0.99, flags, seed, 512 cases; git-tracked |
| `crates/lgbm-core/src/config/mod.rs` | flat Config + Default mirroring C++ | ✓ VERIFIED | 111 fields; config_defaults (5 tests) green |
| `crates/lgbm-core/src/config/alias.rs` | static alias->canonical map | ✓ VERIFIED | 167 pairs; drift-checker asserts verbatim equality |
| `crates/lgbm-core/src/config/scope.rs` | explicit in-scope set | ✓ VERIFIED | IN_SCOPE_PARAMS (122) + OUT_OF_SCOPE_PARAMS |
| `crates/lgbm-core/src/config/set.rs` | from_params pipeline | ⚠️ STUB-FREE but DIVERGENT | Substantive (708 lines, full pipeline) but contains CR-01 + CR-02 behavioral breaks (see Gaps) |
| `crates/oracle-harness/tests/config_drift.rs` | drift-checker over config_auto.cpp | ✓ VERIFIED | Reads in-repo C++ source, asserts superset + verbatim alias table; 3 tests green |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|----|--------|---------|
| `rng_parity.rs` | `random.rs` | replays Random over every committed case | ✓ WIRED | `Random::new` used; 512 cases replayed bit-for-bit |
| `rng_parity.rs` | `fixtures/rng_sequence.txt` | reads committed golden at test time, no C++ toolchain | ✓ WIRED | reads fixture, asserts rng_cases>0 && sample_cases>0 (cannot pass on empty file) |
| `config/set.rs` | `random.rs` | seed derivation uses Random::new + six next_short in C++ order | ✓ WIRED | matches config.cpp:259-268 exactly; seed_derivation (4 tests) green |
| `config/set.rs` | `config/alias.rs` | resolves aliases before member extraction | ⚠️ WIRED but DIVERGENT | resolve_alias called, but collision precedence ≠ C++ (CR-02) |
| `config_drift.rs` | `LightGBM/src/io/config_auto.cpp` | parses C++ source, asserts Rust coverage | ✓ WIRED | parses alias_table()+parameter_set(); workspace-relative only |

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| Workspace builds (edition 2024) | `cargo build --workspace` | Finished, no errors | ✓ PASS |
| Full test suite | `cargo test --workspace` | 49 tests, 0 failed | ✓ PASS |
| RNG parity vs committed golden | `cargo test -p oracle-harness rng_parity -- --nocapture` | "replayed 256 RNG + 256 Sample — all bit-for-bit" | ✓ PASS |
| Idempotent golden regen | `cargo run -p xtask -- regen` then `git diff --stat fixtures/` | empty diff | ✓ PASS |
| CR-01 empty seed/enum | probe: `from_params({seed:""})` / `{objective:""}` | both Err (C++ would be Ok no-op); `{num_leaves:""}` correctly Ok | ✗ FAIL (behavioral divergence) |
| CR-02 alias-collision determinism | probe: 50x `from_params({num_iterations:10, n_estimators:20})` | distinct outcomes {10, 20} in ONE run | ✗ FAIL (non-deterministic) |

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|-------------|-------------|--------|----------|
| FND-01 | 01-01, 01-02 | Port Random PRNG bit-for-bit vs captured C++ sequence | ✓ SATISFIED | random.rs + rng_parity (512 cases) + seed_derivation |
| FND-02 | 01-01 | Workspace crate structure under edition 2024 | ✓ SATISFIED | virtual workspace builds; 4 crates edition 2024 |
| FND-03 | 01-01 | f32 end-to-end matching C++ defaults | ✓ SATISFIED | types.rs ScoreT/LabelT=f32; documented contract |
| FND-04 | 01-01 | thiserror at boundaries, anyhow at app/test | ✓ SATISFIED | thiserror in lgbm-core; anyhow in harness/xtask |
| CFG-01 | 01-02 | Config struct, ~110 in-scope hyperparameters | ✓ SATISFIED | 111-field flat Config; config_defaults green; drift superset |
| CFG-02 | 01-02 | Alias resolution as data table matching config_auto.cpp | ✗ BLOCKED | Table is verbatim (drift green) BUT collision-resolution semantics diverge & are non-deterministic (CR-02) — "matching config_auto.cpp" not fully met |
| CFG-03 | 01-02 | Validation mirroring C++ Config::Set CHECKs as typed Result | ⚠️ PARTIAL | 60+ CHECKs + D-14 randomized validation green, but CR-01 rejects empty=unset values that C++ Config::Set accepts |
| ORA-01 | 01-01 | Oracle harness comparing at ≤~1e-6 absolute (f32) | ✓ SATISFIED | comparator.rs; ORACLE_TOL=1e-6; first-offending-index (WR-04 non-finite caveat) |
| ORA-02 | 01-01 | Pinned C++ reference build/config manifest | ✓ SATISFIED | REFERENCE_MANIFEST.md committed; idempotent regen verified live |

No orphaned requirements: REQUIREMENTS.md maps exactly FND-01..04, CFG-01..03, ORA-01..02 to Phase 1, and all nine are declared across the two plans' frontmatter.

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| `config/set.rs` | 52-56 | Non-deterministic HashMap last-writer-wins for alias collisions | 🛑 Blocker | Reproducibility-contract violation (CR-02); empirically returns different Config for identical input |
| `config/set.rs` | 62, 77-96 | `resolved.get()` bypasses empty==absent filter for seed + 6 enums | 🛑 Blocker | Behavioral-compat break (CR-01); rejects valid empty=unset input |
| `comparator.rs` | 68-94 | `(a-b).abs() <= tol` with no NaN/inf guard | ⚠️ Warning | NaN-vs-NaN and inf-inf silently pass as match; scores can be ±inf/NaN (WR-04). Oracle could miss a real NaN divergence in later phases. |
| `random.rs` | 60-69 | `next_short`/`next_int` panic on `upper==lower` (`% 0`) | ℹ️ Info | Internal callers always pass (0,32767); public API could panic on empty range (WR-05). Not exercised this phase. |
| `config/set.rs` | 571-641 | objective `none`/`null`/`na` aliases dropped; unknown objective hard-rejected | ℹ️ Info | WR-03: `objective=none` (request custom) → Err instead of "custom"; documented test-driven choice for unknowns. |
| `config/set.rs` | 480-488 | f64 config parse uses std parser, not C++ deliberately-lossy Atof | ℹ️ Info | WR-01: long-decimal learning_rate/lambda could differ in low bits vs reference; matters for ≤1e-6 only on adversarial inputs; deferrable but tracked. |
| `config/set.rs` | 340-407 | save_binary task side-effect + objective/metric multiclass mismatch not ported | ℹ️ Info | WR-06/WR-07: `metric` listed in-scope but no field/parse; `task=save_binary` doesn't set save_binary. Under-validates two listed in-scope behaviors. |

No `TODO`/`FIXME`/`TBD`/`XXX`/`HACK`/`PLACEHOLDER` debt markers found in phase-modified source files. No stub artifacts — all files are substantive and wired.

### Human Verification Required

None. All checkable behaviors were verified programmatically (build, full test suite, live RNG parity, live idempotent regen, and direct empirical probes of CR-01/CR-02). The phase produces no visual/real-time/external-service surface.

### Gaps Summary

The **parity spine is genuinely real and excellent** — this is not a stub phase. The RNG port is bit-exact and proven against a real 1.1 MB committed C++ golden over 512 randomized cases; idempotent regen works live with the present toolchain; the workspace, f32 contract, error layering, and PROJECT.md Key Decision are all in place. Four of five success criteria are fully met.

The single gap is **Success Criterion 4 (config behavioral compatibility)**, and I reach this verdict independently of the prior REVIEW, confirmed by direct empirical probes against the running code and the C++ source:

1. **CR-02 (BLOCKER, primary):** Alias-collision resolution is last-writer-wins over Rust's randomized HashMap order. I demonstrated that `from_params({"num_iterations":"10","n_estimators":"20"})` returns BOTH 10 and 20 across 50 calls *in a single process*. C++ `KeyAliasTransform` is deterministic (canonical-priority + `SortAlias`). For a crate whose entire purpose is bit-reproducibility, and which is the config foundation every later phase inherits, a non-deterministic config result is a correctness defect that contradicts SC#4's "matching config_auto.cpp" and the project's non-negotiable reproducibility mandate. The test suite is green only because no test passes a colliding-alias input.

2. **CR-01 (contributing BLOCKER):** `seed` and the six enum fields read `resolved.get()` directly instead of the `present()` (empty==absent) filter every member getter uses. I confirmed `seed=""` and `objective=""` return `Err` where C++ no-ops and keeps defaults. This rejects valid input on a path CLI/Python bindings hit. The fix is mechanical (route through `present()`), but until fixed SC#4's "rejects invalid combos *mirroring* C++ Config::Set" is violated by rejecting *valid* (unset) input.

Both have narrow blast radius for the common single-alias / non-empty path (which is why downstream Phase 2 work is not blocked from *starting*), but both must be closed before the config layer can be trusted as the parity baseline the phase goal promises. WR-04 (oracle NaN/inf blind spot) is a Warning worth closing before later numeric phases lean on the comparator, though it does not affect this phase's RNG-only exact comparison.

**Recommendation:** Re-plan SC#4 closure via `/gsd-plan-phase --gaps` (deterministic SortAlias-based collision resolution + `present()` routing for seed/enums, each with a regression test). The other items (WR-01/03/05/06/07) are tracked as Info/Warning and may be scheduled into Phase 2's config-adjacent work rather than blocking.

---

_Verified: 2026-06-05_
_Verifier: Claude (gsd-verifier)_
