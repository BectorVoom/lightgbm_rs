---
phase: 19-on-device-objectives
fixed_at: 2026-07-01T22:51:33Z
review_path: .planning/phases/19-on-device-objectives/19-REVIEW.md
iteration: 1
findings_in_scope: 8
fixed: 8
skipped: 0
status: all_fixed
---

# Phase 19: Code Review Fix Report

**Fixed at:** 2026-07-01T22:51:33Z
**Source review:** .planning/phases/19-on-device-objectives/19-REVIEW.md
**Iteration:** 1

**Summary:**
- Findings in scope: 8 (fix_scope = all — 4 warning + 4 info)
- Fixed: 8
- Skipped: 0

All in-scope findings were fixed and committed atomically. The `lgbm-compute` crate
and `xtask` crate both compile cleanly (`cargo check`), and `rank_oracle_capture.py`
parses (`ast.parse`). Removing the three `#![allow(unused_imports)]` attributes produced
no unused-import warnings, confirming the reviewer's claim that all imports are live.

## Fixed Issues

### WR-01: `lambdarank_gh_iterN.txt` golden captured/committed but never read by any test

**Files modified:** `xtask/py/rank_oracle_capture.py`, `xtask/src/main.rs`, `crates/oracle-harness/tests/fixtures/rank/lambdarank_gh_iterN.txt` (deleted)
**Commit:** 190c657
**Applied fix:** Took the review's option (a) — drop the dead iterN capture so the
committed fixtures match what tests exercise. Option (b) (emit `score_prev` into
`lambdarank_scores.txt` and add an iterN test cell) requires regenerating fixtures via
the real `lib_lightgbm` capture pipeline, which cannot be run/verified in this
environment; adding a test that reads a not-yet-emitted score line would break the
build. Changes: removed the `booster.predict(...)`-derived iterN grad/hess write from
`capture_lambdarank_gh` (the booster is now trained only as a corpus/params smoke
check; the iter-1 golden derives from a zero score vector and does not need it); removed
the now-orphaned `LATER_ITER` constant; dropped `"lambdarank_gh_iterN.txt"` from the
xtask existence assertion; and `git rm`'d the committed golden. The rank family's
real-`lib_lightgbm` parity remains proven at iter-1 (unchanged).

### WR-02: lambdarank ranks on f32-downcast keys while the kernel/anchor operate in f64

**Files modified:** `crates/lgbm-compute/src/kernels/objective_rank.rs`
**Commit:** a833e64
**Applied fix:** requires human verification (logic change). Rewrote the per-query
tie-canonicalization loop. It now groups runs by the **f32 key bitonic actually
compared** (previously grouped by f64-score equality, which never touched f32-tied /
f64-distinct rows) and sorts each such run by **descending f64 score, then ascending
original index**. This reproduces the host stable-sort order for both cases the review
identified: (1) true f64 ties → ascending index, and (2) rows distinct in f64 but
collapsing to the same f32 → descending f64 score. Distinct-f32 rows remain singleton
runs and keep bitonic's already-correct order.

### WR-03: no label-domain validation before the `launch_unchecked` lambdarank kernel

**Files modified:** `crates/lgbm-compute/src/kernels/objective_rank.rs`
**Commit:** d9679a8
**Applied fix:** Added a V5 pre-launch guard in `lambdarank_impl` that rejects any label
that is non-finite, negative, or `>= label_gain.len()` (the unchecked `label_gain[label]`
index) with `ComputeError::Runtime`. Added the parallel guard in `rank_xendcg_impl`
(labels must be finite and non-negative, since `pow2_int(l as i32)` silently returns 1.0
for negative powers — a wrong, not merely OOB, `Phi`). Consistent with the existing
`query_boundaries`/length checks.

### WR-04: `>2048` `_Sorted` / `_GlobalMemory` large-item variants never exercised in-regime

**Files modified:** `crates/lgbm-compute/src/kernels/objective_rank.rs`
**Commit:** 39114ee
**Applied fix:** Documentation clarification (the "explicitly document + gate" arm). A
genuinely large lambdarank query (`> BITONIC_SORT_NUM_ELEMENTS` items) is already
rejected before launch by the composed `bitonic_argsort_items_on` gate, so the `_Sorted`
launcher cannot silently route an untested large query — it errors. Documented this on
`lambdarank_get_gradients_sorted_on`. For `rank_xendcg_get_gradients_global_on`,
documented that the fold does no sort (size-agnostic body), so the `hess`↔`rho_buf`
aliasing is correct by same-index read-before-write construction and is validated only
on the spine corpus, not against a large-item golden. A synthetic `>1024` test was not
added because the deferred multi-block sort makes such a query error rather than exercise
the path. Ideal long-term fix (the multi-block sort + a large captured golden) remains
deferred and is now called out at the launcher boundary.

### IN-01: redundant module-level `#![allow(unused_imports)]`

**Files modified:** `crates/lgbm-compute/src/kernels/objective_rank.rs`, `crates/lgbm-compute/src/kernels/objective_multiclass.rs`, `crates/lgbm-compute/src/kernels/objective_regression.rs`
**Commit:** a94aef6
**Applied fix:** Removed `#![allow(unused_imports)]` from all three modules. Verified
every import is used (grep + clean `cargo check` with no unused-import warnings), so the
compiler can now flag future stranded imports.

### IN-02: misleading `LengthMismatch` payload for the non-divisible-scores case

**Files modified:** `crates/lgbm-compute/src/kernels/objective_multiclass.rs`
**Commit:** 409d732
**Applied fix:** In `validate_softmax`, the `scores % num_class != 0` branch now returns
`ComputeError::Runtime { detail: "multiclass scores length {scores} is not a multiple of
num_class {num_class}" }` instead of a `LengthMismatch` whose two fields were not
comparable lengths.

### IN-03: `DeviceObjectiveKind` collapses sqrt/rmse into `L2`, discarding the ConvertOutput distinction

**Files modified:** `crates/lgbm-compute/src/device_objective.rs`
**Commit:** de4f6ad
**Applied fix:** Added an enum-level doc note stating it is a support / grad-hess-kernel
classifier ONLY and must never be used as a ConvertOutput routing key — calling out the
`regression_sqrt`/`l2_root`/`rmse` → `L2` collapse and the `CONVERT_SQRT_SQUARE` vs
`CONVERT_PASSTHROUGH` divergence, and directing consumers to route inverse-link off the
original objective name.

### IN-04: `num_class == 1` divides by zero in the softmax factor/convert paths

**Files modified:** `crates/lgbm-compute/src/kernels/objective_multiclass.rs`
**Commit:** 142f750
**Applied fix:** `validate_softmax` now rejects `num_class < 2` (was `< 1`) with
`ComputeError::Runtime`, so the `factor = num_class / (num_class - 1)` divide-by-zero at
`num_class == 1` surfaces a typed error instead of propagating `inf`/`NaN` hessians.

---

_Fixed: 2026-07-01T22:51:33Z_
_Fixer: Claude (gsd-code-fixer)_
_Iteration: 1_
