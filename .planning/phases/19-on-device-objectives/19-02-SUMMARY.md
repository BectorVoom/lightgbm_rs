---
phase: 19-on-device-objectives
plan: 02
subsystem: objective
tags: [cubecl, objective, binary, sigmoid, boost-from-score, oracle-harness, cuda-parity]

# Dependency graph
requires:
  - phase: 19-on-device-objectives
    plan: 00
    provides: ungated objective_binary.rs stub + objective_common parity harness + binary_gh goldens
provides:
  - binary-logloss grad/hess #[cube] (grad_hess_body<F> + shared binary_response helper) + get_gradients_on launcher
  - sigmoid ConvertOutput (sigmoid_convert_output_on) — 1/(1+exp(-sigmoid*x)) inverse-link
  - two-stage BoostFromScore (boost_from_score_on) — device Sigma is_pos reduce COMPOSED with host logit finalize
  - OVA one-vs-all label reset kernel (reset_ova_label_on) — (label == class) ? +1 : -1
  - binary + binary_boost + convert_binary + weight/determinism parity cells
affects: [21]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Pattern 1: one generic #[cube] grad/hess body over F with two #[comptime] bool weight flags (<USE_LABEL_WEIGHT, USE_WEIGHT>) folding both branches out at expansion"
    - "Shared sigmoid #[cube] fn helper (binary_response) so the objective math exists once"
    - "Two-stage BoostFromScore: device reduce_sum_f64_on (atomicAdd analog) composed with a <<<1,1>>> host f64 scalar finalize"
    - "Per-row (no-accumulation) objective math is bit-exact vs golden; the reduce-bearing init scalar is ORACLE_TOL (documented atomic residual, D-05)"

key-files:
  created: []
  modified:
    - crates/lgbm-compute/src/kernels/objective_binary.rs
    - crates/oracle-harness/tests/objective_parity_binary.rs

key-decisions:
  - "Binary per-row grad/hess is BIT-EXACT (compare_exact_u32) vs the real binary_gh golden despite the sigmoid exp — the f64 exp ~1-ULP residual is absorbed by the score_t f32 cast (D-01), unlike the poisson exp path"
  - "The two <USE_LABEL_WEIGHT, USE_WEIGHT> templates map to two #[comptime] bool params; the balanced 1.0/1.0 label-weight default is bit-identical to the off branch"
  - "BoostFromScore init asserted with compare_within(ORACLE_TOL), NOT compare_exact — the device Sigma is_pos reduce is the documented atomicAdd-order residual (D-05/Pitfall 5)"

patterns-established:
  - "Objective family plan: fill the Wave-1 stub kernel + its owned parity test file with zero cross-plan contention, anchored to the host f64 fold AND the real *_gh golden"

requirements-completed: [ODL-06]

# Metrics
duration: 5min
completed: 2026-07-01
status: complete
---

# Phase 19 Plan 02: On-Device Binary Objective Summary

**Binary-logloss ported to standalone CubeCL kernels — a `#[cube]` sigmoid grad/hess (Pattern 1, two `#[comptime]` weight flags over a shared `binary_response` helper), the sigmoid ConvertOutput inverse-link, a two-stage device-reduce/host-finalize BoostFromScore logit init, and the one-vs-all `±1` label reset — anchor-pinned BIT-EXACT to the cpu f64 fold and the real `binary_gh` golden (init scalar within ORACLE_TOL for the documented atomic residual). Standalone per D-02 (no GBDT wiring — Phase 21).**

## Performance
- **Duration:** ~5 min
- **Started:** 2026-07-01T20:45:51Z
- **Completed:** 2026-07-01T20:50:01Z
- **Tasks:** 2
- **Files modified:** 2

## Accomplishments
- `objective_binary.rs` (stub → 410 lines): `grad_hess_body<F>` with the single-source-of-truth `binary_response` sigmoid helper and two `#[comptime] bool` weight flags (`<USE_LABEL_WEIGHT, USE_WEIGHT>`), the `grad_hess_kernel_f64` cpu-anchor wrapper, and the `get_gradients_on` launcher with a V5 length guard (the f64-compute → f32-cast reproduces the golden bit-for-bit).
- `sigmoid_convert_output_on`: the `1/(1+exp(-sigmoid*x))` elementwise inverse-link, bit-exact vs `lgbm_model::objective::convert_binary`.
- `boost_from_score_on`: the two-stage init — stage 1 = `reduce_sum_f64_on(is_pos)` on device (plus `Σ is_pos·w` + `Σ w` on the weighted path, D-08 COMPOSE), stage 2 = `pavg = clamp(Σ/N, ε, 1-ε); init = ln(pavg/(1-pavg))/σ` as an f64 host scalar (mirrors `binary.rs:102-117`).
- `reset_ova_label_on`: the `ResetOVACUDALabelKernel` analog — a per-row `(label == class) ? +1 : -1` rewrite.
- `objective_parity_binary.rs`: `binary` (grad/hess bit-exact vs host anchor AND `binary_gh_iter{1,N}` golden), `binary_boost` (init within ORACLE_TOL + OVA reset bit-exact), `convert_binary` (sigmoid bit-exact), and weight-branch/label-weight-branch/twice-run determinism cells — 4 tests, all green.

## Task Commits
1. **Task 1: binary grad/hess #[cube] + sigmoid ConvertOutput** — `9d33629` (feat)
2. **Task 2: two-stage BoostFromScore (logit init) + OVA label reset** — `79da78e` (feat)

**Plan metadata:** _(final docs commit)_

## Files Created/Modified
- `crates/lgbm-compute/src/kernels/objective_binary.rs` — filled the Wave-1 stub: grad/hess kernel + launcher, sigmoid ConvertOutput, two-stage BoostFromScore, OVA label reset.
- `crates/oracle-harness/tests/objective_parity_binary.rs` — the binary-family parity cells + weight/determinism properties.

## Decisions Made
- Binary per-row grad/hess is **bit-exact** vs the real `binary_gh` golden despite the sigmoid `exp`: the f64 `exp`'s ~1-ULP residual is absorbed by the `score_t` f32 cast (D-01) — verified empirically (`compare_exact_u32` passes on both `iter1` and `iterN`). This is the structural difference from the poisson `exp` path (regression 19-01), whose large-magnitude `exp(score)` output does NOT round-trip through f32 identically.
- The `<USE_LABEL_WEIGHT, USE_WEIGHT>` templates fold to two `#[comptime] bool` params (Pattern 1); the balanced `1.0/1.0` label-weight default is bit-identical to the off branch (proved by the label-weight-branch equivalence property).
- The BoostFromScore init is asserted within `ORACLE_TOL`, not bit-exact — the device `Σ is_pos` reduce is the documented atomicAdd-order residual (D-05 / Pitfall 5), even though on the deterministic cpu anchor it is in fact bit-stable.

## Deviations from Plan
None — plan executed exactly as written. Both tasks' acceptance criteria and the verification block pass as specified.

## Issues Encountered
- Transient: the Task-1 import of `compare_within`/`ORACLE_TOL` (needed only by Task 2) triggered an unused-import warning at the Task-1 boundary; narrowed the Task-1 import to `compare_exact_u32` and re-added the two at Task 2 so each atomic commit is warning-clean.

## Known Stubs
None — the module is fully wired to the f64 anchor; no placeholder/empty data paths. (Standalone per D-02 — GBDT boosting-seam wiring is Phase 21, by design, not a stub.)

## User Setup Required
None — the tests use the committed `binary_gh_iter{1,N}.txt` / `binary_scores.txt` goldens (present); no capture run or external service required.

## Next Phase Readiness
- ODL-06 complete: `lgbm_compute::kernels::objective_binary` exposes `get_gradients_on`, `sigmoid_convert_output_on`, `boost_from_score_on`, `reset_ova_label_on` for the Phase-21 on-device growth loop.
- `reset_ova_label_on` is shared-ready for the multiclass OVA path (19-03).
- Default host boosting path is byte-unchanged; `LGBM_CUDA_ON_DEVICE` stays OFF.

## Self-Check: PASSED
- `crates/lgbm-compute/src/kernels/objective_binary.rs` exists (410 lines, contains `get_gradients`); `crates/oracle-harness/tests/objective_parity_binary.rs` exists (4 passing cells).
- Both task commits present in git history: `9d33629`, `79da78e`.
- `cargo test -p oracle-harness --test objective_parity_binary` → 4 passed; `cargo test -p lgbm-compute --lib` → 100 passed.

---
*Phase: 19-on-device-objectives*
*Completed: 2026-07-01*
