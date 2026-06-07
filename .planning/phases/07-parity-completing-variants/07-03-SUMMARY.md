---
phase: 07-parity-completing-variants
plan: 03
subsystem: objective
tags: [poisson, gamma, tweedie, cross_entropy, cross_entropy_lambda, xentropy, gbdt, objective, parity, oracle, lightgbm-4.6, exp-log, OBJ-04, OBJ-05]

# Dependency graph
requires:
  - phase: 06-gbdt-spine-core-objectives-metrics
    provides: "GBDT spine boosting loop, objective seam (BoostObjective enum), ObjectiveKind ConvertOutput shim, layered L1-L5 oracle harness + boosting-oracle-capture xtask"
  - phase: 07-parity-completing-variants (07-02)
    provides: "OBJ-04 family-A objective seam + capture/replay pattern (write_layered, capture_family_a, family_a parity cells); DEF-07-02 deferral pattern"
provides:
  - "Objective::Poisson / Gamma / Tweedie enum arms (parse + from_config + flags + faithful C++-ported grad/hess + SafeLog BoostFromScore + check_labels >=0 guard)"
  - "Xentropy (CrossEntropy / CrossEntropyLambda) struct — grad/hess + BoostFromScore + [0,1] label guard"
  - "ObjectiveError::LabelRange typed variant (C++ LabelOutOfRange Init guard -> Result)"
  - "BoostObjective::Xentropy variant + dispatch; resolve_objective booster helper"
  - "ObjectiveKind Poisson(exp)/CrossEntropy(sigmoid)/CrossEntropyLambda(log1p(exp)) ConvertOutput"
  - "Builder setters poisson_max_delta_step / tweedie_variance_power"
  - "exp/log capped-horizon (5-iter) capture emitter + 22 capture-gated parity cells + 107 real-lib_lightgbm-4.6 goldens"
  - "OBJ-05 fully delivered GREEN (cross_entropy / cross_entropy_lambda)"
  - "DEF-07-02 extended: gamma (all) + tweedie bfa-off/axis non-constant-hessian learner knife-edge"
affects: [07-04, 07-parity-verifier, OBJ-04, OBJ-05, learner-split-gain-fp-trace-fix-plan]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "exp/log objectives: all transcendental arithmetic in f64, single f32 cast at the score_t boundary; SafeLog(label-mean) BoostFromScore for poisson/gamma/tweedie"
    - "exp-libm horizon cap (Pitfall 5): capture at EXP_LOG_NUM_ITERATIONS=5 so every tree stays bit-exact; ConvertOutput is the only ORACLE_TOL surface (mirrors multiclass 5-iter precedent)"
    - "non-constant-hessian learner split-gain knife-edge: poisson (uniform iter-0 hessian) is GREEN; gamma/tweedie (label-dependent hessian) flip a tree-0 split — same DEF-07-02 family as fair"

key-files:
  created:
    - crates/lgbm-objective/src/xentropy.rs
    - crates/oracle-harness/tests/fixtures/boosting/{poisson,gamma,tweedie,cross_entropy,cross_entropy_lambda}_*.txt (107 fixtures)
  modified:
    - crates/lgbm-objective/src/regression.rs
    - crates/lgbm-objective/src/error.rs
    - crates/lgbm-objective/src/lib.rs
    - crates/lgbm-boosting/src/objective.rs
    - crates/lgbm-model/src/objective.rs
    - crates/lgbm/src/booster.rs
    - crates/lgbm/src/builder.rs
    - crates/lgbm-core/tests/config_validation.rs
    - crates/oracle-harness/tests/boosting_parity.rs
    - xtask/src/main.rs
    - xtask/py/boosting_oracle_capture.py

key-decisions:
  - "Apply (not re-decide) the 07-02 ship-green/defer-blocked disposition: poisson/cross_entropy/cross_entropy_lambda/tweedie-spine ship faithful GREEN; gamma (all) + tweedie bfa-off/axis #[ignore]'d under DEF-07-02 (extended), as the SAME non-constant-hessian learner split-gain knife-edge as fair — g/h into each tree faithful, NOT an objective bug"
  - "exp/log horizon capped at 5 iters (Pitfall 5 exp-libm caveat) — cap the horizon, never weaken the assertion"
  - "Config params poisson_max_delta_step / tweedie_variance_power were already present + CHECK'd (07-02 scaffold); 07-03 added the poisson_max_delta_step>0 validation-test coverage"

patterns-established:
  - "exp/log objective faithful port: f64 transcendental + single f32 cast; SafeLog BoostFromScore; typed Init label-domain guard"

requirements-completed: []  # OBJ-04 still PARTIAL (gamma/tweedie-bfa-off + fair + quantile-bagged pending DEF-07-02); OBJ-05 delivered but left to the verifier to mark

# Metrics
duration: ~20min
completed: 2026-06-07
---

# Phase 07 Plan 03: OBJ-04 exp/log family (poisson/gamma/tweedie) + OBJ-05 (cross_entropy/cross_entropy_lambda) Summary

**poisson + cross_entropy + cross_entropy_lambda + the tweedie spine ship faithful real-lib_lightgbm-4.6 parity (f64 transcendental + single f32 cast, SafeLog/logit/log-expm1 BoostFromScore, typed label guards); gamma (all) + tweedie bfa-off/axis are honestly deferred under DEF-07-02 (extended) as the same non-constant-hessian learner-level f64 split-gain knife-edge as fair — g/h into each tree bit-exact, NOT an objective bug.**

## Performance

- **Duration:** ~20 min
- **Completed:** 2026-06-07
- **Tasks:** 3 (Task 1 objective math, Task 2 builder/capture/cells, Task 3 capture — wheel gate already satisfied at `/tmp/lgbm-capture-venv`)
- **Files:** 12 code/script + 107 real-binary fixtures

## Accomplishments

- **GREEN (faithful parity, committed byte-idempotent fixtures, capped 5-iter horizon):**
  - **poisson:** spine (model/pred) + scores (bit-exact f64) + g/h + full {bag×es×bfa} loop matrix + `poisson_max_delta_step` axis (0.1). Iter-0 hessian `exp(score)·exp(max_delta_step)` is UNIFORM across rows (all share the SafeLog init), so the tree-0 histogram/split is bit-exact.
  - **cross_entropy (OBJ-05):** spine + scores + g/h + full loop matrix (stable log1pexp grad/hess; logit-of-label-mean BoostFromScore; sigmoid ConvertOutput; `[0,1]` label guard).
  - **cross_entropy_lambda (OBJ-05):** spine + scores + g/h + full loop matrix (`z·(1-z)` hessian; `log(expm1(havg))` BoostFromScore; `log1p(exp)` ConvertOutput).
  - **tweedie SPINE** (default ρ=1.5, bfa-on): spine + scores + g/h faithful.
- **Objective math (Task 1):** faithful 1:1 C++ ports with all transcendental arithmetic in f64 and a single f32 cast at the `score_t` boundary; `SafeLog(label-mean)` BoostFromScore for poisson/gamma/tweedie; typed `LabelRange` Init guards (poisson/gamma/tweedie `>= 0` + non-zero-sum; xentropy `[0,1]`).
- **Predict-side:** `ObjectiveKind` extended with `Poisson` (exp), `CrossEntropy` (sigmoid), `CrossEntropyLambda` (log1p(exp)) ConvertOutput, parse + convert arms.
- **Builder:** `poisson_max_delta_step` / `tweedie_variance_power` setters route into Config (unit-tested round-trip).
- **DEFERRED (DEF-07-02, extended; honest):** gamma (all) + tweedie bfa-off loop + variance_power axis `#[ignore]`'d with reasons referencing the deferred-items doc.

## Task Commits

1. **Task 1: poisson/gamma/tweedie + cross_entropy(_lambda) objective math + label guards** — `97f1b80` (feat)
2. **Task 2: exp/log builder setters + capped-horizon capture emitter + parity cells** — `cdcc2d8` (feat)
3. **Task 3: capture exp/log real-lib_lightgbm-4.6 goldens (poisson/cross_entropy GREEN, gamma/tweedie-bfa-off DEF-07-02)** — `845b9b1` (test)

**Plan metadata:** (this SUMMARY + STATE/ROADMAP) — committed after this file.

## GREEN vs Deferred (honest OBJ-04/OBJ-05 status)

| Objective | spine | scores | g/h | loop matrix {bag×es×bfa} | param axis | status |
|-----------|-------|--------|-----|--------------------------|-----------|--------|
| poisson | GREEN | GREEN | GREEN | GREEN | GREEN (max_delta_step=0.1) | **OBJ-04: shipped faithful** |
| cross_entropy | GREEN | GREEN | GREEN | GREEN | n/a | **OBJ-05: shipped faithful** |
| cross_entropy_lambda | GREEN | GREEN | GREEN | GREEN | n/a | **OBJ-05: shipped faithful** |
| tweedie | GREEN | GREEN | GREEN | **DEF-07-02** (bfa-off tree 0) | **DEF-07-02** (ρ=1.9 tree 0) | **OBJ-04: spine shipped; loop/axis deferred** |
| gamma | **DEF-07-02** (tree 0) | **DEF-07-02** | **DEF-07-02** (iter-4 downstream) | **DEF-07-02** (tree 0) | n/a | **OBJ-04: fully deferred** |

DEF-07-02 = 07-01-class learner-level f64 split-gain knife-edge over a NON-CONSTANT hessian; g/h INTO each tree are faithful (tweedie_gradients passes iter-1 + iter-4; gamma iter-1 passes — the iter-4 diff is downstream of the diverged tree-0 split). Needs a source-built lib_lightgbm 4.6 CPU single-thread FP execution trace (the 07-01 method). `#[ignore]` is the sanctioned 05-06/CR-03 ignore-pending-fix marker — no tolerance weakened, no horizon capped to hide a flip.

**Why poisson is GREEN but gamma/tweedie are not:** at iteration 0 every row shares the SafeLog(label-mean) init score, so poisson's hessian `exp(score)·exp(max_delta_step)` is UNIFORM across rows → the tree-0 histogram/split gain matches the real binary bit-exact. gamma (`label·exp(-score)`) and tweedie (`-label·(1-ρ)·exp((1-ρ)·score)+(2-ρ)·exp((2-ρ)·score)`) have LABEL-dependent (non-uniform) hessians even at iter 0, exercising the same borderline f64 split-gain comparison that fair's tiny non-constant hessian does (DEF-07-02). The tweedie default-ρ bfa-on SPINE happens to land on the right side of the knife-edge (same as the fair/quantile spine pattern).

## Decisions Made

- **Apply (not re-decide) the 07-02 ship-green/defer-blocked disposition.** The gamma/tweedie divergence is the SAME root cause already deferred under DEF-07-02 (non-constant-hessian learner split-gain knife-edge, g/h faithful into the tree), so it was extended in `deferred-items.md` rather than raised as a new checkpoint/decision.
- **exp/log 5-iter horizon cap** (EXP_LOG_NUM_ITERATIONS) is the carried Pitfall-5 exp-libm caveat — capping the horizon keeps every tree bit-exact; the ConvertOutput predict (one exp/sigmoid/log1p per row) is the only ORACLE_TOL surface. Documented in-code in both the capture script and the parity test.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 2 - Missing critical functionality] Init label-domain guards wired into the booster**
- **Found during:** Task 1 (the threat register T-07-03-01 assigns `mitigate`; the plan specified typed guards but they must be CALLED at train time).
- **Issue:** `Objective::check_labels` / `Xentropy::check_labels` exist but the booster only constructed the objective; without calling the guard an out-of-range label would feed `exp`/`log` (NaN/-inf).
- **Fix:** Added a `resolve_objective` booster helper that constructs the objective AND calls `check_labels` for poisson/gamma/tweedie/xentropy before training (both `train` and `train_with_valid`), surfacing `LgbmError::Objective` — never a panic.
- **Files modified:** crates/lgbm/src/booster.rs
- **Verification:** lgbm-objective unit tests assert the typed errors; workspace GREEN.
- **Committed in:** 97f1b80

### Scope-boundary handling

**2. [Scope boundary] Out-of-scope re-emitted goldens moved aside (not committed)**
- **Found during:** Task 3 capture (the full capture script re-emits ALL boosting goldens).
- **Issue:** The capture re-emitted the never-tracked `regression_sqrt_*` (06-06) and `regression_mf2es_*` (06-04/CR-02) goldens, re-activating the pre-existing-failing `reg_sqrt_spine_matches_real_binary` test — a separate 06-06 gap, NOT 07-03.
- **Fix:** Moved (NOT committed, NOT deleted) to `.out-of-scope-fixtures-holding/` (untracked), exactly as 07-02 did. Also reverted the multiclass-ES tracked-fixture churn (a separate capture-script reproducibility item: the ES valid-dataset `bin_construct_sample_cnt` re-construct path is non-deterministic across re-runs in this environment) back to HEAD so it does not contaminate the 07-03 commit.
- **Verification:** 0 tracked-fixture churn; `git status` shows only the 107 new exp/log fixtures staged.

---

**Total deviations:** 2 (1 Rule-2 correctness wiring, 1 scope-boundary handling). No tolerance weakened, no horizon silently capped.

### Pre-existing clippy notes (out of scope, not fixed)

A `neg_multiply` warning at `regression.rs:962` (`-1.0f64 * weight`) and `field_reassign_with_default` warnings in pre-existing test helpers (`builder.rs:436` `from_config_escape_hatch`, `regression.rs` mape test) pre-date this plan (07-02 / earlier). My new code is clippy-clean. The `regression.rs:962` line is a load-bearing op-order mirror of the C++ `Sign(diff) * label_weight`; left unchanged to avoid altering the intentional f64-cast order.

## Issues Encountered

- gamma (all) + tweedie bfa-off/axis hit a genuine 07-01-class learner-level split-gain knife-edge (non-constant hessian; g/h faithful into each tree). Per the established (human-chosen, 07-02) DEF-07-02 disposition this was deferred-extended rather than masked. Resolution requires the dedicated source-built lib_lightgbm 4.6 FP-trace learner-fix plan (shared with fair + quantile-bagged).

## Self-Check: PASSED

- Files verified present: crates/lgbm-objective/src/xentropy.rs; crates/oracle-harness/tests/fixtures/boosting/poisson_spine_model.txt, gamma_spine_model.txt, tweedie_spine_model.txt, cross_entropy_spine_model.txt, cross_entropy_lambda_spine_model.txt, exp_log_best_iterations.txt; deferred-items.md.
- Commits verified: 97f1b80, cdcc2d8, 845b9b1.
- 107 exp/log fixtures tracked; byte-idempotent on re-run (verified); no LightGBM/ or out-of-scope (regression_sqrt/mf2es) files committed.
- Gate: `cargo test --workspace` GREEN (boosting_parity 56 passed / 13 ignored / 0 failed); `cargo build --workspace --tests` exit 0; clippy clean on edited code; spine UNREGRESSED (learner_parity 12/12, kernel_parity 4/4).

## Next Phase Readiness

- **OBJ-05 fully delivered GREEN** (cross_entropy / cross_entropy_lambda — spine + loop matrix faithful).
- **OBJ-04 still PARTIAL:** poisson fully faithful; tweedie spine faithful. gamma (all) + tweedie bfa-off/axis joined fair + quantile-bagged under DEF-07-02 — all the SAME non-constant-hessian / bagged-subset learner split-gain knife-edge family. OBJ-04 is NOT marked complete.
- **Follow-up needed:** the single 07-01-style learner-level split-gain FP-trace fix plan (DEF-07-02) now covers fair + quantile-bagged + gamma + tweedie-bfa-off; a single `find_best_split` non-constant-hessian operand fix may close several cells at once.
- 07-04+ (next waves) are unblocked — they do not depend on the deferred gamma/tweedie cells.

---
*Phase: 07-parity-completing-variants*
*Completed: 2026-06-07*
