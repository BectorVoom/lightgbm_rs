---
phase: 19-on-device-objectives
reviewed: 2026-07-02T00:00:00Z
depth: standard
files_reviewed: 17
files_reviewed_list:
  - crates/lgbm-compute/src/device_objective.rs
  - crates/lgbm-compute/src/kernels/mod.rs
  - crates/lgbm-compute/src/kernels/objective_binary.rs
  - crates/lgbm-compute/src/kernels/objective_multiclass.rs
  - crates/lgbm-compute/src/kernels/objective_rank.rs
  - crates/lgbm-compute/src/kernels/objective_regression.rs
  - crates/lgbm-compute/src/lib.rs
  - crates/oracle-harness/tests/fixtures/rank/lambdarank_gh_iter1.txt
  - crates/oracle-harness/tests/fixtures/rank/lambdarank_gh_iterN.txt
  - crates/oracle-harness/tests/objective_common/mod.rs
  - crates/oracle-harness/tests/objective_parity_binary.rs
  - crates/oracle-harness/tests/objective_parity_multiclass.rs
  - crates/oracle-harness/tests/objective_parity_rank.rs
  - crates/oracle-harness/tests/objective_parity_regression.rs
  - xtask/py/rank_oracle_capture.py
  - xtask/src/main.rs
findings:
  critical: 0
  warning: 4
  info: 4
  total: 8
status: issues_found
---

# Phase 19: Code Review Report

**Reviewed:** 2026-07-02
**Depth:** standard
**Files Reviewed:** 17
**Status:** issues_found

## Summary

Phase 19 ports the eleven CUDA (§5) grad/hess + inverse-link objectives to `#[cube]`
kernels on the cubecl-cpu f64 anchor, adds the `device_objective_supported`
host-fallback discriminator, and wires the ranking oracle-capture. The kernel logic is
carefully transcribed from the host anchors and is guarded by strong parity tests:
regression/binary/multiclass are asserted **bit-exact** against real `lib_lightgbm`
4.6 goldens, and the classifier + weight/label-weight/determinism properties are
covered. I traced the truncation loop, high/low selection, softmax max-subtraction, the
`_GlobalMemory` hessian-buffer aliasing (read-before-write is genuinely safe), the
gamma-draw row alignment, and the percentile index math against the host references and
the Python capture — all match on the tested corpora.

No correctness defect is provable on the tested/intended path, so there are **no
BLOCKERs**. The findings are (a) a real rank-family test-coverage gap — a captured,
committed, and xtask-verified `lambdarank_gh_iterN.txt` golden that **no test consumes**,
while every other family checks its iterN golden; (b) a latent numerical-fidelity gap in
the lambdarank sort (f32-downcast keys feeding an f64 anchor); (c) a missing
label-domain V5 guard before a `launch_unchecked` kernel that does an unchecked table
index; and (d) unverified large-item (`>2048`) code paths. Plus minor quality items.

No `<structural_findings>` block was supplied, so this report is narrative-only.

## Narrative Findings (AI reviewer)

## Warnings

### WR-01: `lambdarank_gh_iterN.txt` golden is captured, committed, and asserted-written — but no test ever reads it

**File:** `crates/oracle-harness/tests/objective_parity_rank.rs:127-166`, `xtask/py/rank_oracle_capture.py:489-498`, `xtask/src/main.rs:1896-1899`
**Issue:** `capture_lambdarank_gh` writes both `lambdarank_gh_iter1.txt` and
`lambdarank_gh_iterN.txt`, `rank_oracle_capture()` bails if either is missing
(`xtask/src/main.rs:1896-1899`), and both are committed under
`tests/fixtures/rank/`. But the `lambdarank` test only consumes the **iter1** golden
(`objective_parity_rank.rs:145`); the second call
`check_lambdarank(&client, &final_scores, …)` at line 165 compares device-vs-host anchor
only and never touches a golden. A `grep` confirms `lambdarank_gh_iterN` appears in
**zero** test files, whereas binary/multiclass/OVA/regression all assert their
`*_gh_iterN.txt` golden via `read_scores_line(LATER_ITER-2)`. The rank
`*_scores.txt` fixture stores only the single **final** raw-score line, so the iterN
intermediate score vector cannot be reconstructed — the iterN golden is structurally
unusable by the current test. Net effect: the rank family's real-`lib_lightgbm` parity
is proven at iter-1 only, a weaker guarantee than every other objective family, and a
committed golden is dead weight that will silently rot.
**Fix:** Either (a) drop the iterN capture + the `xtask` existence assertion so the
committed fixtures match what is exercised, or (b) capture the per-iter scores line
(as the boosting families do — emit `score_prev` at `num_iteration = LATER_ITER-1` into
`lambdarank_scores.txt`) and add an iterN `compare_within(ORACLE_TOL)` cell mirroring
`objective_parity_regression.rs:196-208`.

### WR-02: lambdarank ranks on f32-downcast score keys while the kernel and host f64 anchor operate in f64

**File:** `crates/lgbm-compute/src/kernels/objective_rank.rs:417-445`
**Issue:** The per-query ranking is produced by
`bitonic_argsort_items_on(client, &scores_f32, …)` over `scores as f32`
(line 417-418), but the kernel body reads the **original f64** `scores` for
`best_score`/`worst_score`/`delta_score` and the host `Lambdarank::get_gradients`
anchor sorts by f64. The tie-canonicalization loop (lines 428-445) then reorders equal
runs using **f64** equality (`scores[..] == scores[..]`). For any two rows whose scores
are distinct in f64 but collapse to the same f32, bitonic treats them as a tie and
places them in network-arbitrary order, and the f64-equality canonicalizer does **not**
repair them (it sees them as distinct, so `j` never advances past them) — yielding a
ranking that can diverge from the true f64-descending order the anchor is defined to
reproduce. This is masked today only because the committed corpus scores originate from
C++ `score_t = float` accumulation and are therefore exactly f32-representable; it
becomes a real anchor infidelity the moment this "deterministic f64 anchor" is fed
genuine f64 scores (e.g. the Phase-21 growth loop).
**Fix:** Rank on the f64 scores directly (an f64-key argsort primitive), or at minimum
extend the canonicalization to also order f32-tied / f64-distinct runs by descending
f64 score before the ascending-index tie-break, and document the f32-key precondition
at the launcher boundary if the primitive cannot be changed.

### WR-03: no label-domain validation before the `launch_unchecked` lambdarank kernel — unchecked `label_gain` table index

**File:** `crates/lgbm-compute/src/kernels/objective_rank.rs:215-216`, `391-408`
**Issue:** `lambdarank_body` indexes the caller-supplied `label_gain` table with
`label_gain[u32::cast_from(labels[high_row]) as usize]` (lines 215-216) inside a
`#[cube(launch_unchecked)]` kernel. `lambdarank_impl` validates `labels.len() == n`,
`query_boundaries`, and `inverse_max_dcgs.len()` at the V5 boundary, but never checks
that every label is `< label_gain.len()`. A label `>= label_gain.len()` (the default
gain table is 31 entries) is an out-of-bounds read in an *unchecked* kernel — UB / OOB
on device, and not bounds-checked on the cubecl-cpu anchor either. The `SAFETY` comment
(lines 465-469) asserts "label_gain by the (non-negative integer) label" as if it were
guaranteed, but nothing in this function establishes it. Every other index in the body
is provably in range; this one rests on an unstated caller precondition that no other
validated input has.
**Fix:** Add a V5 check in `lambdarank_impl` (and the xendcg `pow2_int` path, which has
the same `l as i32` assumption) that `max(labels) < label_gain.len()` and that labels
are non-negative, returning `ComputeError::Runtime`/`LengthMismatch` — consistent with
how `query_boundaries` and lengths are already guarded before the launch.

### WR-04: the `>2048` `_Sorted` / `_GlobalMemory` large-item variants are never exercised in the regime they exist for

**File:** `crates/lgbm-compute/src/kernels/objective_rank.rs:366-388`, `721-734`; `crates/oracle-harness/tests/objective_parity_rank.rs:103-112,199-207`
**Issue:** Both large-item launchers are, by their own admission
(lines 366-371, "True per-query lengths `> BITONIC_SORT_NUM_ELEMENTS` remain the
deferred multi-block-sort hardening"), validated only on the 30-row spine corpus. The
tests assert `shared == _Sorted` and `shared == _GlobalMemory` bit-exactly, which proves
the shared accumulation body is reused — but it does **not** prove the code behaves
correctly for actual `>2048`-item queries (the sort capacity the variants are named for,
and the `_GlobalMemory` hessian-buffer aliasing whose whole reason to exist is the
large-query case). The parity claim for the large-item path is therefore unproven.
**Fix:** Add a synthetic single-query corpus with `> BITONIC_SORT_NUM_ELEMENTS` items
(or explicitly document that these variants are stubs pending the deferred multi-block
sort and gate the `_Sorted`/`_GlobalMemory` public launchers behind that limitation so a
caller cannot route a genuinely large query into an untested path).

## Info

### IN-01: redundant module-level `#![allow(unused_imports)]` masks future dead imports

**File:** `crates/lgbm-compute/src/kernels/objective_multiclass.rs:39`, `objective_rank.rs:40`, `objective_regression.rs:39`
**Issue:** These modules carry `#![allow(unused_imports)]` at the top, but every import
they declare is actually used (`cubecl::prelude::*`, `ComputeError`,
`bitonic_argsort_items_on`, `draw_next_float_on`, `reduce_*`, etc.). The blanket allow
suppresses the compiler's dead-import warning going forward, so a future edit that
leaves an import stranded will not be flagged.
**Fix:** Remove the `#![allow(unused_imports)]` from the three modules where nothing is
actually unused (keep it only where a genuinely-conditional import needs it, if any).

### IN-02: misleading `LengthMismatch` payload for the non-divisible-scores case

**File:** `crates/lgbm-compute/src/kernels/objective_multiclass.rs:132-133`
**Issue:** When `scores % num_class != 0`, `validate_softmax` returns
`ComputeError::LengthMismatch { expected: scores, actual: num_class }`. The two numbers
do not describe a length mismatch (one is the score-buffer length, the other the class
count), so the error message will be confusing when it fires.
**Fix:** Use a `ComputeError::Runtime { detail: format!("multiclass scores length {scores} is not a multiple of num_class {num_class}") }`, or at least populate `expected`/`actual` with comparable quantities.

### IN-03: `DeviceObjectiveKind` collapses `regression_sqrt`/`l2_root`/`rmse` into `L2`, discarding the sqrt ConvertOutput distinction

**File:** `crates/lgbm-compute/src/device_objective.rs:70-71`
**Issue:** `regression_sqrt`, `l2_root`, `root_mean_squared_error`, and `rmse` all map to
`DeviceObjectiveKind::L2`. That is correct for the *grad/hess* kernel (same device
kernel) and harmless for the current boolean-gate use of the enum, but the sqrt variant
needs `CONVERT_SQRT_SQUARE` at inverse-link time while plain L2 needs
`CONVERT_PASSTHROUGH`. If a future consumer keys the ConvertOutput mode off
`DeviceObjectiveKind`, `regression_sqrt` will silently pick passthrough and produce
wrong predictions.
**Fix:** Either document at the enum that it is a support-classifier only (never a
convert-mode routing key), or split the sqrt variant so the distinction survives.

### IN-04: `num_class == 1` divides by zero in the softmax factor and convert paths

**File:** `crates/lgbm-compute/src/kernels/objective_multiclass.rs:174`, `238-247`
**Issue:** `validate_softmax` accepts `num_class >= 1`, but `factor = num_class / (num_class - 1)` (line 174) is `1/0 = inf` for `num_class == 1`, and the per-row softmax
convert would operate on a single class. A `num_class == 1` multiclass config is
nonsensical (and the host has the same latent issue), but it is unguarded here and would
propagate `inf`/`NaN` hessians rather than a typed error.
**Fix:** Reject `num_class < 2` for the softmax objective at the V5 boundary with a
`ComputeError::Runtime`, matching the "never a silent wrong answer" discipline used
elsewhere in the file.

---

_Reviewed: 2026-07-02_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
