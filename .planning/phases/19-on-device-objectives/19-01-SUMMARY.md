---
phase: 19-on-device-objectives
plan: 01
subsystem: infra
tags: [cubecl, objective, regression, grad-hess, boostfromscore, percentile, oracle-harness, cuda-parity]

# Dependency graph
requires:
  - phase: 19-on-device-objectives
    provides: ungated objective_regression kernel stub + objective_common parity harness (19-00)
  - phase: 14-foundation-shared-device-primitives-device-structs-rng
    provides: reduce_sum/reduce_min f64 primitives + bitonic_argsort_on device sort
  - phase: 18-on-device-data-partition-tree-predict
    provides: comptime-flag #[cube] fan-out precedent (data_partition route_to_left) + LGBM_CUDA_ON_DEVICE seam
provides:
  - Six regression grad/hess #[cube] kernels (L2/L1/Quantile/Huber/Fair/Poisson) via ONE generic-over-Float body + comptime objective_tag + comptime use_weight
  - ConvertOutput inverse-link kernel (passthrough / Sign(x)*x^2 / exp)
  - BoostFromScore init composition (mean via reduce_sum, median/quantile via a device-composed PercentileFun, Poisson label-check + safe_log)
  - RenewTreeOutput L1/Quantile per-leaf median (host-orchestrated loop, no percentile_device kernel)
  - objective_parity_regression parity cells bit-exact vs the f64 anchor AND the real-lib_lightgbm *_gh goldens
affects: [19-02, 19-03, 19-04, 21-on-device-growth-wiring]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "ONE generic #[cube] fn<F: Float> grad/hess body + comptime objective_tag u32 (folds the six objectives to straight-line code) + comptime use_weight bool (<USE_WEIGHT> template)"
    - "f64 cpu-anchor launcher casts the f64 body result to f32 (score_t) → reproduces the C++ f64-compute→f32-cast order bit-for-bit"
    - "Device-composed PercentileFun: bitonic_argsort_on (device sort) + PercentileFun index math as an f64 scalar host finalize"
    - "RenewTreeOutput = host loop over leaves calling the composed percentile (Discrepancy 1 — no percentile_device per-leaf kernel)"

key-files:
  created:
    - crates/oracle-harness/tests/objective_parity_regression.rs
  modified:
    - crates/lgbm-compute/src/kernels/objective_regression.rs

key-decisions:
  - "Anchor kernel is f64 (F=f64 cpu fold, D-10); the generic F: Float body lets the hip mirror instantiate F=f32 (D-07: no hardcoded f64 per-row hot loop in the production path)"
  - "comptime objective_tag if-chain (each branch assigns a seeded g/h) — the #[cube] macro restructures branches so bare uninitialized bindings fail definite-init; seed + #[allow(unused_assignments)]"
  - "Poisson grad/hess + ConvertOutput exp held to ORACLE_TOL vs the numpy-derived golden (exp-libm residual, Pitfall 5); every arithmetic-only objective is bit-exact (compare_exact_u32)"
  - "BoostFromScore/RenewTreeOutput compose the PercentileFun convention (the real-lib_lightgbm CPU reference the goldens use), NOT the percentile_unweighted_f32_on skeleton — see deviation"

patterns-established:
  - "Pattern: harden a phase-14 percentile skeleton at its Phase-19 consumer by composing the argsort primitive + the reference finalize, rather than mutating the shared primitive"
  - "Pattern: device-vs-host-anchor as the always-true D-05 gate (same f64 math + inputs) + device-vs-golden as the secondary score-derivation gate"

requirements-completed: [ODL-05]

# Metrics
duration: 16min
completed: 2026-07-01
status: complete
---

# Phase 19 Plan 01: On-Device Regression Objectives Summary

**Six regression grad/hess CubeCL kernels (L2/L1/Quantile/Huber/Fair/Poisson) from ONE generic-over-Float comptime-dispatched body, plus ConvertOutput, a primitive-composed BoostFromScore init, and a host-orchestrated RenewTreeOutput per-leaf median — all bit-exact to the `lgbm_objective::regression` f64 anchor and the real-lib_lightgbm `*_gh` goldens (poisson `exp` held to ORACLE_TOL).**

## Performance

- **Duration:** 16 min
- **Started:** 2026-07-01T15:07:49Z
- **Completed:** 2026-07-01T15:23:23Z
- **Tasks:** 3
- **Files created/modified:** 2 (1 created, 1 filled from stub)

## Accomplishments
- Filled `objective_regression.rs`: ONE `#[cube] grad_hess_body<F: Float>` dispatching the six §5.1 objectives via `#[comptime] objective_tag` + `#[comptime] use_weight`, an f64 cpu-anchor launcher that casts the f64 result to `f32` (reproducing the C++ `f64-compute → f32-cast` order), six per-objective `*_on` launchers, and a `convert_output_on` inverse-link (passthrough / `Sign(x)·x²` / `exp`).
- `boost_from_score_on` composes device primitives (D-08): L2/Huber/Fair mean via `reduce_sum_f64_on`; L1/Quantile median via a device-composed `PercentileFun`; Poisson label non-negativity/zero-sum check via `reduce_min_f64_on` + `reduce_sum_f64_on` then `safe_log(mean)`.
- `renew_tree_output_on` host-loops over leaves computing the L1/Quantile per-leaf residual percentile (Discrepancy 1 — no `percentile_device` per-leaf kernel).
- `objective_parity_regression.rs` (6 cells, all green): `regression` (six objectives × iter-1/iter-N bit-exact vs the f64 anchor AND the `*_gh` goldens), `regression_weight_branch_and_determinism` (`<USE_WEIGHT>` equivalence + twice-run determinism), `boost_from_score` + poisson label-guard, `renew_leaf`, `convert_regression`.

## Task Commits

Each task was committed atomically:

1. **Task 1: Six regression grad/hess kernels + ConvertOutput** - `c32927f` (feat)
2. **Task 2: BoostFromScore init via primitive composition** - `a0bb611` (feat)
3. **Task 3: RenewTreeOutput L1/Quantile per-leaf median (host-orchestrated)** - `620f572` (feat)

**Plan metadata:** _(final docs commit)_

## Files Created/Modified
- `crates/lgbm-compute/src/kernels/objective_regression.rs` - filled the ODL-05 stub: 6 grad/hess kernels + launchers, ConvertOutput, BoostFromScore composition, host-orchestrated RenewTreeOutput, device-composed `percentile_fun_on`.
- `crates/oracle-harness/tests/objective_parity_regression.rs` - the regression-family parity cells (regression / boost_from_score / renew_leaf / convert_regression + weight-branch/determinism/label-guard properties).

## Decisions Made
- The cpu anchor kernel is f64 (the deterministic f64-fold, D-10); the body is generic `F: Float` so the hip mirror instantiates `F = f32` (D-07 — the production per-row path is f32, not a hardcoded f64 hot loop). The launcher's single f64→f32 output cast matches the golden's `f64-compute → f32-cast` order bit-for-bit.
- The six objectives fold via a `#[comptime] objective_tag` if-chain with a seeded `g`/`h` (the `#[cube]` macro restructures branches so a bare uninitialized binding fails definite-init; seed + `#[allow(unused_assignments)]`).
- Arithmetic-only objectives (L2/L1/Quantile/Huber/Fair) are bit-exact (`compare_exact_u32`); Poisson's `exp` grad/hess and the `exp` ConvertOutput are held to `ORACLE_TOL` vs the numpy-derived golden (the documented exp-libm residual, Pitfall 5 / D-05) — device still matches the host f64 anchor.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Percentile skeleton index convention diverges from the CPU PercentileFun reference**
- **Found during:** Task 2 (BoostFromScore) — the L1 median came out 10 vs the host anchor / golden 11.
- **Issue:** The plan (must-haves + Task 2/3 actions) prescribes composing `crate::kernels::primitives::percentile_unweighted_f32_on`. That phase-14 skeleton uses a `(1-alpha)*len` index convention that DIVERGES from the CPU `PercentileFun` `(len-1)*(1-alpha)` convention the real-lib_lightgbm BoostFromScore/RenewTreeOutput goldens (and the `lgbm_objective::regression` f64 anchor) use — the spine-label median is 10 under the skeleton vs the reference 11. The skeleton's `percentile.txt` golden is absent in this checkout, so `primitive_parity_percentile` skip-passes and never caught the divergence.
- **Fix:** Added a private `percentile_fun_on` that COMPOSES the shared `bitonic_argsort_on` device primitive (descending sort) with the exact `PercentileFun` index math as an f64 scalar finalize — matching the host anchor bit-exact. Did NOT mutate the phase-14 primitive (its ROCm percentile fixture would silently drift, and `primitives.rs` is owned by phase 14 / not in this plan's `files_modified`). `boost_from_score_on` (L1/Quantile) and `renew_tree_output_on` both route through it.
- **Files modified:** crates/lgbm-compute/src/kernels/objective_regression.rs
- **Verification:** `boost_from_score` + `renew_leaf` cells now bit-exact vs the host f64 anchor for all six / both objectives; `primitive_parity` unaffected (untouched).
- **Committed in:** `a0bb611` (Task 2 commit) — the helper is reused by Task 3 (`620f572`).

---

**Total deviations:** 1 auto-fixed (1 bug — percentile-convention hardening at the designated Phase-19 consumer).
**Impact on plan:** Necessary for the hard acceptance gate ("bit-exact vs the host f64 anchor"); no scope creep and no shared-file contention (the fix is self-contained in `objective_regression.rs`). The literal must-have "via percentile_unweighted_f32_on" is honored in spirit — a device percentile is still composed from the shared argsort primitive; only the index-convention finalize differs.

## Recommendations for Follow-up
- **Phase 14 percentile skeleton reconciliation:** `percentile_unweighted_f32_on` / `percentile_weighted_f32_on` should be reconciled to the `PercentileFun` `(len-1)*(1-alpha)` convention (or their `PercentileDevice` provenance confirmed against a real CUDA golden) before any other consumer relies on them — the current `(1-alpha)*len` convention is unvalidated (no `percentile.txt` fixture) and diverges from the CPU reference.

## Issues Encountered
- **`#[cube]` definite-init:** the macro restructures the comptime `if`-chain so a bare `let g: F;` (assigned in every branch) fails rustc's definite-initialization (E0381) and a bare `let g;` fails type inference (E0282). Resolved by seeding `let mut g = F::new(0.0)` + `#[allow(unused_assignments)]` on the two `#[cube]` bodies.
- **Poisson corpus/horizon:** the poisson golden uses `exp_log_corpus("poisson")` (= the spine labels) and `EXP_LOG_LATER_ITER = 4` (score line k=3), not `LATER_ITER = 5`; the test keys the `_scores.txt` line index off `later_iter - 2` per objective.

## User Setup Required
None - no external service configuration required (the cubecl-cpu f64 anchor runs in-process; `LGBM_CUDA_ON_DEVICE` stays OFF, D-06).

## Next Phase Readiness
- ODL-05 regression objectives are STANDALONE (no GBDT wiring — D-02, Phase 21). The launchers (`get_gradients_*_on`, `boost_from_score_on`, `renew_tree_output_on`, `convert_output_on`) are ready for the Phase-21 boosting-seam wiring.
- Sibling Wave-2 plans (19-02/03/04) are unaffected — this plan touched only `objective_regression.rs` (its owned stub) + a new test file; `boosting_parity` (shared `*_gh`/`*_scores` fixtures) stays 75/75 green.
- Follow-up flagged: the phase-14 percentile skeleton convention (see Recommendations).

## Self-Check: PASSED

Both files exist on disk; all 3 task commits (`c32927f`, `a0bb611`, `620f572`) present in git history; `objective_parity_regression` 6/6 green, `lgbm-compute --lib` 100/100, `boosting_parity` 75/75, workspace tests compile.

---
*Phase: 19-on-device-objectives*
*Completed: 2026-07-01*
