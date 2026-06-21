---
phase: 01-oracle-contract-foundations
reviewed: 2026-06-05T00:00:00Z
depth: standard
files_reviewed: 21
files_reviewed_list:
  - crates/lgbm-compute/src/lib.rs
  - crates/lgbm-core/src/config/alias.rs
  - crates/lgbm-core/src/config/mod.rs
  - crates/lgbm-core/src/config/scope.rs
  - crates/lgbm-core/src/config/set.rs
  - crates/lgbm-core/src/error.rs
  - crates/lgbm-core/src/lib.rs
  - crates/lgbm-core/src/random.rs
  - crates/lgbm-core/src/types.rs
  - crates/lgbm-core/tests/alias_resolution.rs
  - crates/lgbm-core/tests/config_defaults.rs
  - crates/lgbm-core/tests/config_validation.rs
  - crates/lgbm-core/tests/seed_derivation.rs
  - crates/oracle-harness/src/comparator.rs
  - crates/oracle-harness/tests/comparator.rs
  - crates/oracle-harness/tests/config_drift.rs
  - crates/oracle-harness/tests/rng_parity.rs
  - xtask/cpp/CMakeLists.txt
  - xtask/cpp/rng_capture.cpp
  - xtask/src/main.rs
findings:
  critical: 2
  warning: 7
  info: 4
  total: 13
status: issues_found
---

# Phase 1: Code Review Report

**Reviewed:** 2026-06-05
**Depth:** standard
**Files Reviewed:** 21
**Status:** issues_found

## Summary

This phase ports the parity-critical foundation: the `Random` LCG, the `Config`
struct + `from_params` pipeline (alias resolution, seed derivation, CHECK
validation, conflict mutations), the abs-diff oracle comparator, and the C++ RNG
capture harness.

The **core RNG/Sample port is bit-exact** and well-tested: the committed golden
fixture is real (`rng_sequence.txt`, 1.1 MB, git-tracked) and `rng_parity`
replays 256 RNG + 256 Sample cases against it, passing. Constants, wrapping
arithmetic, the `f32/32768.0f` divisor, the `Sample` four-way branch (including
the `BTreeSet`↔`std::set` ordering and the `K > N/log2(K)` double-division
boundary), and the six-seed derivation order all verified faithful to
`LightGBM/include/LightGBM/utils/random.h` and `config.cpp:259-268`. The alias
table is verbatim (167 == 167 pairs) and the drift-checker is genuine.

However, the **`from_params` config pipeline diverges from C++ parsing semantics
in several places** that break behavioral compatibility (CLAUDE.md: "100%
behavioral compatibility with C++ LightGBM for in-scope APIs, configs"). Two are
classified BLOCKER: an unguarded empty-string path that errors where C++ no-ops,
and the non-deterministic alias-collision resolution that contradicts C++'s
deterministic `SortAlias` tie-break. Several WARNINGs cover float-parse
bit-fidelity, dropped objective aliases, NaN/inf handling in the oracle
comparator, and `%`-by-zero panics in the public RNG API.

No structural findings block was provided.

## Critical Issues

### CR-01: Empty-string values rejected where C++ treats them as "not provided"

**File:** `crates/lgbm-core/src/config/set.rs:62`, `77-96`
**Issue:** The `seed` lookup and all six enum-typed parses read directly from
`resolved.get(...)` instead of the empty-string-filtering `present()` helper that
the rest of the pipeline uses. In C++, every `Get*` helper guards with
`params.count(name) > 0 && !params.at(name).empty()` (config.h:1165-1219) — an
**empty value is identical to an absent key** and is a no-op that leaves the
default in place.

In Rust:
- `{"seed": ""}` → `parse_int("seed", "")` → `Err(InvalidType)`. C++: no-op (no
  seed derivation, defaults stand).
- `{"task": ""}` / `{"boosting": ""}` / `{"objective": ""}` / `{"device_type":
  ""}` / `{"tree_learner": ""}` / `{"data_sample_strategy": ""}` →
  `Err(UnknownValue)`. C++: no-op, keeps the config.h default.

This is a behavioral-compatibility break on a code path users hit (CLI/Python
bindings routinely pass empty strings for unset params), and it makes the seed
path in particular reject input the reference accepts.

**Fix:** Route these through `present()` like every other field:
```rust
// Stage 2
if let Some(seed_str) = present(&resolved, "seed") {
    let seed = parse_int("seed", seed_str)?;
    // ...
}
// Stage enum parses
if let Some(v) = present(&resolved, "task") { cfg.task = parse_task(v)?; }
if let Some(v) = present(&resolved, "boosting") { cfg.boosting = parse_boosting(v)?; }
// ...and the remaining four enum fields.
```

### CR-02: Alias-collision resolution is non-deterministic and contradicts C++ `SortAlias`

**File:** `crates/lgbm-core/src/config/set.rs:52-56`
**Issue:** Stage 1 resolves every incoming key to its canonical name and inserts
into a `HashMap` with **last-writer-wins over HashMap iteration order**. When two
aliases of the same canonical (or an alias plus the canonical itself) are both
present, the winner is non-deterministic across runs — and does not match C++.

The C++ `ParameterAlias::KeyAliasTransform` (config.h) resolves collisions
deterministically via `SortAlias` (shorter key wins; ties broken
lexicographically) AND gives a **directly-set canonical name priority over any
alias**. Example: params `{"num_iterations": "10", "n_estimators": "20"}` — C++
keeps the canonical `num_iterations=10` and ignores the alias; the Rust code may
keep either value depending on hash order. Two runs of the same `from_params`
call can therefore produce different `Config`s, which is fatal for a
reproducibility-contract crate (CLAUDE.md numerical-fidelity mandate; the module
doc itself flags this as "Pitfall 4"-adjacent).

**Fix:** Reproduce the C++ precedence. Prefer a directly-provided canonical over
any alias, and break alias-vs-alias ties with the `SortAlias` rule (by
`(key.len(), key)`), not HashMap order:
```rust
let mut resolved: HashMap<String, String> = HashMap::with_capacity(params.len());
// Track the winning source key per canonical to apply SortAlias deterministically.
let mut winner_key: HashMap<String, String> = HashMap::new();
for (key, value) in params {
    let canonical = resolve_alias(key).to_string();
    let direct = key == &canonical;
    match winner_key.get(&canonical) {
        None => { winner_key.insert(canonical.clone(), key.clone()); resolved.insert(canonical, value.clone()); }
        Some(prev) => {
            let prev_direct = prev == &canonical;
            // canonical (direct) beats any alias; else shorter-then-lexicographic wins.
            let new_wins = (direct && !prev_direct)
                || (direct == prev_direct && (key.len(), key.as_str()) < (prev.len(), prev.as_str()));
            if new_wins {
                winner_key.insert(canonical.clone(), key.clone());
                resolved.insert(canonical, value.clone());
            }
        }
    }
}
```
(Confirm the exact precedence against `KeyAliasTransform` — the canonical-wins
branch corresponds to its "alias not find in params" check.)

## Warnings

### WR-01: `f64` config parse uses std parser, not C++'s deliberately-lossy `Atof`

**File:** `crates/lgbm-core/src/config/set.rs:480-488`
**Issue:** `parse_double` uses Rust's `str::parse::<f64>()` (correctly-rounded).
C++ `GetDouble` → `AtofAndCheck` → `Common::Atof`
(`LightGBM/include/LightGBM/utils/common.h`), a hand-rolled parser whose own
source comment warns it has "**less** floating point precision ... kept to
maintain bit-for-bit the legacy LightGBM behaviour in terms of precision." For
inputs like long decimal `learning_rate`/`lambda_l2` strings, the two parsers can
yield different f64 bit patterns. Those values feed tree outputs directly, so a
divergence can exceed the ≤1e-12 / ~1e-6 oracle tolerance on some inputs. The
default path uses `Atof` (not the `precise_float_parser`/`AtofPrecise` path), so
matching the legacy `Atof` rounding is the parity-correct choice.

**Fix:** Port `Common::Atof` (and `Atoi`) char-by-char rather than delegating to
the std parsers, so config float parsing is bit-identical to the reference.
Gate the precise variant behind `precise_float_parser` later.

### WR-02: Integer/float parse trim + overflow semantics differ from C++ `Atoi`/`Atof`

**File:** `crates/lgbm-core/src/config/set.rs:470-488`
**Issue:** `parse_int`/`parse_double` call `.trim()` then std `parse`. C++
`Atoi`/`Atof` only skip leading/trailing **`' '` (space)**, not `\t`/`\n`/`\r`.
So `"\t5"` → Rust Ok, C++ Fatal (rejected). Additionally:
- Overflow: `"99999999999"` → Rust `Err(InvalidType)`; C++ `Atoi` accumulates in
  `int` and silently wraps to a valid (garbage) value, returning success.
- Bare sign / no digits: `"+"` or `"-"` → C++ `Atoi` yields `0` and `*after ==
  '\0'` → **valid, out=0**; Rust → `Err(InvalidType)`.

These are edge cases but each is a behavioral divergence on the in-scope config
boundary.

**Fix:** Ports of `Atoi`/`Atof` (WR-01) resolve all three — they replicate the
space-only trim, the wrap-on-overflow accumulation, and the empty-digit→0 rule.

### WR-03: `parse_objective` drops `none`/`null`/`na` aliases and hard-rejects unknown objectives

**File:** `crates/lgbm-core/src/config/set.rs:571-641`
**Issue:** C++ `ParseObjectiveAlias` maps `none`/`null`/`custom`/`na` → `"custom"`
(config.h), and `GetObjectiveType` passes **any unrecognized objective through
unchanged** (no fatal at config time). The Rust `parse_objective`:
1. Omits `none`/`null`/`na` from `KNOWN`, so `objective=none` (a documented way to
   request a custom objective) → `Err(UnknownValue)` instead of `→ "custom"`.
2. Treats every unknown objective as a hard `UnknownValue` error, whereas C++
   only fatals later, inside the objective factory.

Item 1 is a clear compatibility regression for a real user value; item 2 is a
deliberate test-driven deviation (the validation test expects `"nonsense"` →
Err) but should be documented as an intentional early-validation choice.

**Fix:** Add `none`/`null`/`na` to the alias mapping (→ `custom`). Decide and
document whether unknown objectives should error at config time or pass through
to mirror C++.

### WR-04: Oracle comparator silently passes NaN and infinite mismatches

**File:** `crates/oracle-harness/src/comparator.rs:68-94`
**Issue:** `abs_diff_within` and `compare_within` use `(a - b).abs() <= tol` /
`> tol`. If either operand is `NaN`, `(a-b).abs()` is `NaN`; `NaN > tol` is
`false`, so the comparator reports a **match**. `inf - inf` is also `NaN` →
silent pass; `inf` vs finite is `inf > tol` → correctly flagged, but `NaN` vs
`NaN` and `NaN` vs finite both pass. Scores/gradients can legitimately be
`±inf`/`NaN` (e.g. `kMinScore`/`kMaxScore` in `types.rs`), so a real divergence
into NaN would go undetected by the oracle.

**Fix:** Treat any non-finite disagreement as a mismatch:
```rust
pub fn abs_diff_within(a: f32, b: f32, tol: f32) -> bool {
    if a.is_nan() || b.is_nan() { return a.is_nan() && b.is_nan(); }
    if a.is_infinite() || b.is_infinite() { return a == b; }
    (a - b).abs() <= tol
}
```
Mirror the same guard in `compare_within` before the `abs_diff > tol` check.

### WR-05: Public `next_short`/`next_int` panic on `upper == lower` (`% 0`)

**File:** `crates/lgbm-core/src/random.rs:60-69`
**Issue:** Both compute `rand_intN() % (upper - lower)`. When `upper == lower`
the modulus is `0` → Rust panics (C++ has UB). Internal seed-derivation calls
always pass `(0, 32767)` so the foundation is safe, but these are `pub` API. Per
CLAUDE.md "never panic on user input," a downstream caller passing an empty range
will crash. Also `upper < lower` yields a negative modulus / negative result
(matches C++ UB, but worth noting).

**Fix:** Document the precondition and/or debug-assert `upper > lower`; or define
the empty-range result explicitly. Match whatever the C++ call sites guarantee
once they are ported, but do not let library input panic.

### WR-06: `metric`/objective multiclass conflict from C++ `CheckParamConflict` is not ported

**File:** `crates/lgbm-core/src/config/set.rs:347-407`
**Issue:** C++ `CheckParamConflict` fatals on objective/metric multiclass
mismatch (`"Multiclass objective and metrics don't match"`, config.cpp:328-338).
The Rust port omits this branch. `metric` is listed in `IN_SCOPE_PARAMS`
(scope.rs:163) but there is no `Config.metric` field and it is never parsed, so
the conflict can never be detected. A user passing
`objective=multiclass, metric=l2` gets `Ok` in Rust but a fatal in C++.

**Fix:** Either add the `metric` field + parsing and port the mismatch check, or
explicitly document `metric` as not-yet-validated in this phase and remove the
mismatch-related parity claim. The current state silently under-validates a
listed in-scope param.

### WR-07: `save_binary` task side effect not ported

**File:** `crates/lgbm-core/src/config/set.rs:340-407`
**Issue:** C++ `Config::Set` forces `save_binary = true` when
`task == kSaveBinary` (config.cpp:300-303), before `CheckParamConflict`. The Rust
pipeline parses `task=save_binary` but never applies this mutation, so
`from_params({"task":"save_binary"}).save_binary` stays `false` (≠ C++ `true`).

**Fix:** After the enum parses (or in `check_param_conflict`), add:
```rust
if cfg.task == "save_binary" && !cfg.save_binary { cfg.save_binary = true; }
```

## Info

### IN-01: `feature_fraction_bynode != 1.0` monotone-method override omitted

**File:** `crates/lgbm-core/src/config/set.rs:347-407`
**Issue:** C++ resets `monotone_constraints_method` to `"basic"` when
`feature_fraction_bynode != 1.0` with intermediate/advanced method
(config.cpp:451-456). Not ported. Likely out of v1 scope (monotone constraints
are not deeply validated yet), but the `check_param_conflict` doc comment claims
to port "the in-scope `CheckParamConflict` side-effects" — narrow that claim or
add the branch.

### IN-02: `check_param_conflict` doc references a non-existent shift guard

**File:** `crates/lgbm-core/src/config/set.rs:366-368`
**Issue:** Comment says "guard the shift against overflow," but the code uses
`2f64.powi(max_depth)`, not a bit shift. Harmless, but the comment is misleading.
Behavior is otherwise correct (inf comparison short-circuits; the `as i32` cast
only runs for small values).

**Fix:** Reword the comment to describe the `powi`/double-pow path.

### IN-03: `K_EPSILON` compared in f64 after f32 widening — verify against C++ `kEpsilon`

**File:** `crates/lgbm-core/src/config/set.rs:385`, `391`; `types.rs:31`
**Issue:** `K_EPSILON` is `1e-15f32`; the conflict checks compare
`cfg.path_smooth > K_EPSILON as f64`. C++ uses `kEpsilon` (`1e-15f`, a `float`)
promoted to `double` in `path_smooth > kEpsilon`. The widening `1e-15f32 as f64`
equals the C++ `float→double` promotion bit-for-bit, so this is correct — noted
only because it is a subtle f32→f64 promotion on the parity-critical path and
deserves a one-line comment confirming intent.

### IN-04: `parse_brace_pair` in the drift-checker mis-handles canonicals containing a comma

**File:** `crates/oracle-harness/tests/config_drift.rs:92-100`
**Issue:** `parse_brace_pair` splits the brace body on `','` and takes the first
two fields. No current alias/canonical contains a comma, so this is safe today,
but if upstream ever adds one the parser would silently mis-parse and the
verbatim-table test could pass on wrong data. Low risk; add a guard or an
`assert!(parts.next().is_none())` to fail loudly on an unexpected third field.

---

_Reviewed: 2026-06-05_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
