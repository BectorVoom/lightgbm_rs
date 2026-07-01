---
phase: 19-on-device-objectives
plan: 03
subsystem: compute
tags: [cubecl, objective, multiclass, softmax, multiclassova, oracle-harness, class-major]

# Dependency graph
requires:
  - phase: 19-on-device-objectives
    plan: 00
    provides: ungated objective_multiclass.rs stub + objective_common parity harness + class-major multiclass_gh / multiclassova_gh goldens
  - phase: 19-on-device-objectives
    plan: 02
    provides: binary sigmoid response reference (transcribed locally for OVA, no cross-file dep)
provides:
  - Class-major softmax grad/hess #[cube] kernel + f64-anchor launcher (softmax_get_gradients_on)
  - Per-row softmax ConvertOutput inverse-link (softmax_convert_output_on)
  - MulticlassOVA per-class grad/hess #[cube] kernel + launcher (multiclassova_get_gradients_on)
  - objective_parity_multiclass parity cells (multiclass, multiclassova, softmax_convert)
affects: [21-on-device-growth-wiring]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Class-major [num_data*k + i] stride kernel: ONE thread per ROW, runtime loop over num_class (Pattern 3)"
    - "usize stride scalars matching the usize ABSOLUTE_POS lane index (histogram resident_bins idiom) — NOT u32"
    - "D-09 pre-allocated per-row softmax scratch: one reused client.empty handle before the launch"
    - "OVA reuses the binary-logloss response math transcribed locally per class at offset=num_data*i (self-contained, no 19-02 dep)"

key-files:
  created: []
  modified:
    - crates/lgbm-compute/src/kernels/objective_multiclass.rs
    - crates/oracle-harness/tests/objective_parity_multiclass.rs

key-decisions:
  - "Softmax buffer allocated ONCE before the (single) launch as a reused client.empty handle (D-09); the kernel writes exp into it then divides in place — identical algorithm to lgbm_model::objective::softmax, not a re-implementation"
  - "Stride scalars are usize (ABSOLUTE_POS is a usize lane index in cubecl 0.10) — the u32 form does not type-check against the num_data*k+i index arithmetic"
  - "MulticlassOVA transcribes the binary response locally rather than importing 19-02, keeping the plan self-contained (parity-neutral Discretion)"

patterns-established:
  - "Pattern: class-major per-row objective kernel with a runtime num_class loop and the load-bearing [num_data*k+i] stride"

requirements-completed: [ODL-07]

# Metrics
duration: 10min
completed: 2026-07-01
status: complete
---

# Phase 19 Plan 03: On-Device Multiclass Objectives Summary

**Class-major softmax grad/hess + softmax ConvertOutput + MulticlassOVA `#[cube]` kernels on the cubecl-cpu f64 anchor — bit-exact (`compare_exact_u32`) vs both the `lgbm-objective` multiclass anchors and the real-lib_lightgbm class-major `multiclass_gh` / `multiclassova_gh` goldens, with the load-bearing `[num_data*k+i]` stride and the held-out `Σ_k grad ≈ 0` invariant.**

## Performance
- **Duration:** ~10 min
- **Started:** 2026-07-01T21:22:59Z
- **Completed:** 2026-07-01T21:33:17Z
- **Tasks:** 2
- **Files modified:** 2

## Accomplishments
- **Task 1 — class-major softmax grad/hess + softmax ConvertOutput.** `softmax_grad_hess_body<F: Float>` runs ONE thread per ROW, gathering `rec[k] = scores[num_data*k+i]`, computing softmax with the max-subtraction algorithm into a **D-09 pre-allocated per-row scratch buffer** (a single reused `client.empty` handle allocated once before the launch), then writing `grad[num_data*k+i] = (label==k) ? p-1 : p` and `hess[num_data*k+i] = factor*p*(1-p)`, `factor = num_class/(num_class-1)`. `softmax_get_gradients_on` is the f64-anchor launcher with a V5 `num_data*num_class` shape guard. `softmax_convert_output_on` is the per-row softmax ConvertOutput inverse-link.
- **Task 2 — MulticlassOVA.** `ova_grad_hess_body<F: Float>` loops the `num_class` one-vs-all classes at `offset = num_data*i`, transcribing the binary-logloss response math locally (`is_pos = label==i`, `response = -label_val*sigmoid/(1+exp(label_val*sigmoid*score))`, `hess = |response|*(sigmoid-|response|)`) so the plan is self-contained (no cross-file dependency on 19-02). `multiclassova_get_gradients_on` is the launcher.
- **Parity cells (`objective_parity_multiclass.rs`):** `multiclass` (device == `MulticlassSoftmax` f64 anchor AND class-major `multiclass_gh_iter{1,N}` golden, bit-exact; class-major `Σ_k grad[k·N+i] ≈ 0` per-row invariant; twice-run determinism), `multiclassova` (device == `MulticlassOva` f64 anchor AND `multiclassova_gh_iter{1,N}` golden), `softmax_convert` (device == host `softmax` anchor, bit-exact). All 3 pass.

## Task Commits
1. **Task 1: class-major softmax grad/hess + softmax ConvertOutput** — `b08bc08` (feat)
2. **Task 2: MulticlassOVA per-class grad/hess** — `fe4b210` (feat)

## Files Modified
- `crates/lgbm-compute/src/kernels/objective_multiclass.rs` — filled the 19-00 stub: softmax grad/hess `#[cube]` + launcher, `softmax_convert_output`, MulticlassOVA per-class path (416 lines).
- `crates/oracle-harness/tests/objective_parity_multiclass.rs` — `multiclass` + `multiclassova` + `softmax_convert` parity cells, class-major Σgrad≈0 invariant, determinism.

## Decisions Made
- The D-09 softmax scratch is a single reused `client.empty` handle sized `num_data*num_class`, pre-allocated once before the launch; the kernel writes `exp(rec-wmax)` into it and divides in place — the SAME max-subtraction algorithm as `lgbm_model::objective::softmax`, not a numerically-different re-implementation.
- Stride scalars (`num_data`, `num_class`) are `usize`, matching the `usize` `ABSOLUTE_POS` lane index (the histogram `resident_bins[f*num_data+row]` idiom). The initial `u32` form did not type-check against the `num_data*k+i` index arithmetic.
- MulticlassOVA transcribes the binary response locally rather than importing the 19-02 kernel — parity-neutral (Discretion), keeps the plan self-contained.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Stride scalars must be `usize`, not `u32`**
- **Found during:** Task 1 (first `cargo build -p lgbm-compute`)
- **Issue:** The plan/pattern implied `u32` dimension scalars, but in cubecl 0.10 `ABSOLUTE_POS` is a `usize` lane index; `num_data*k + i` mixing a `u32` stride with the `usize` `i` fails to type-check (`cannot add usize to u32`, 37 errors).
- **Fix:** Changed `num_data`/`num_class` to `usize` in all three `#[cube]` bodies + wrappers + launch calls (the documented histogram `resident_bins` convention, `copy_subrow.rs:56`). No numerical change.
- **Files modified:** crates/lgbm-compute/src/kernels/objective_multiclass.rs
- **Verification:** `cargo build -p lgbm-compute` clean; all 3 parity cells bit-exact.
- **Committed in:** `b08bc08` / `fe4b210`

**2. [Rule 1 - Bug] iterN golden uses `MULTICLASS_LATER_ITER = 4`, not the spine's 5**
- **Found during:** Task 1 (iterN golden mismatch — device matched the host anchor exactly but not the golden)
- **Issue:** The multiclass cells cap the horizon at 5 iters and derive `*_gh_iterN` from `predict(num_iteration = 3)` (`later_iter = 4`), whereas the single-output binary spine uses `later_iter = 5`. The test initially read the wrong `*_scores.txt` line (index 3 instead of 2).
- **Fix:** Set the test's `LATER_ITER = 4` (→ scores line index `LATER_ITER-2 = 2`), matching `boosting_oracle_capture.py::MULTICLASS_LATER_ITER`.
- **Files modified:** crates/oracle-harness/tests/objective_parity_multiclass.rs
- **Verification:** iterN softmax + OVA grad/hess bit-exact vs golden.
- **Committed in:** `b08bc08` / `fe4b210`

**Total deviations:** 2 auto-fixed (1 blocking, 1 bug). No architectural changes.

## Verification
- `cargo test -p oracle-harness --test objective_parity_multiclass` — 3/3 cells pass (`multiclass`, `multiclassova`, `softmax_convert`).
- `cargo test -p lgbm-compute --lib` — 100 passed, 0 failed (no regression).
- `cargo test -p lgbm-objective multiclass` — 9 passed (host anchors unchanged).
- `LGBM_CUDA_ON_DEVICE` unset throughout; module is additive and OFF by default (D-06).

## Acceptance Criteria
- Kernel gathers `score[num_data*k+i]` and writes `grad[num_data*k+i]` (class-major `num_data*k` stride) — confirmed.
- Softmax buffer handle created once outside the launch loop (D-09) — confirmed.
- `multiclass` cell bit-exact vs the class-major golden; `Σ_k grad ≈ 0` invariant holds — confirmed.
- MulticlassOVA applies the binary response per class at `offset = num_data*i` — confirmed (grep: `let offset = num_data * i;`).
- `multiclassova` cell bit-exact vs the `multiclassova_gh` golden — confirmed.

## TDD Note
Task 1 was `tdd="true"`. Because the kernel launchers and the parity test share a compilation unit (the test references the not-yet-existing launchers), a clean compile-failing RED was impractical; the kernel + parity cell were co-developed and verified together against the pre-existing real goldens (the effective RED = the class-major golden the test asserts against). The GREEN gate is the bit-exact pass of all 3 cells.

## Next Phase Readiness
- Phase 21 (on-device growth wiring) can call `softmax_get_gradients_on` / `multiclassova_get_gradients_on` / `softmax_convert_output_on`; the D-09 per-row scratch handle would be hoisted and reused across boosting iterations there.
- All four Wave-2 objective family kernels (regression/binary/multiclass/rank) are now filled; the on-device path stays OFF by default behind `LGBM_CUDA_ON_DEVICE`.

## Self-Check: PASSED

Both modified files exist on disk; both task commits (`b08bc08`, `fe4b210`) present in git history; all 3 parity cells green.

---
*Phase: 19-on-device-objectives*
*Completed: 2026-07-01*
