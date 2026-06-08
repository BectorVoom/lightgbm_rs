---
phase: quick-260609-b1a
plan: 01
subsystem: lgbm-compute (GPU split kernel boundary)
tags: [perf, gpu-boundary, allocation, parity, landed]
verdict: LANDED (re-applied after user decision; commit 58589fb)
requires: []
provides: []
affects: []
tech-stack:
  added: []
  patterns: []
key-files:
  created:
    - .planning/quick/260609-b1a-split-outcell-empty-o1/260609-b1a-SUMMARY.md
  modified: []  # split.rs edited then git-checkout reverted to HEAD — net zero
decisions:
  - "O1 (empty() out-cells for single-feature split kernels) REVERTED — not landed."
  - "kernel_parity_split_within_tol_on_hip is a PRE-EXISTING deterministic failure on master baseline, independent of O1."
metrics:
  duration: ~10m
  completed: 2026-06-09
---

# Phase quick-260609-b1a Plan 01: O1 empty() out-cells for single-feature split kernels — Summary

**One-liner:** Applied O1 (swap `create_from_slice(zeros)` → `client.empty()` for the two
single-feature split out-cell buffers). Initially reverted by the executor on the conservative
"not worth a parity break" rule, then **RE-LANDED per explicit user decision** once it was
established the ROCm split parity cell is a PRE-EXISTING failure on master (independent of O1)
and O1 keeps the CPU f64-fold **hard merge gate bit-exact GREEN**.

## Verdict: LANDED — commit 58589fb

`crates/lgbm-compute/src/kernels/split.rs` carries O1: both single-feature split launchers
allocate `h_out` via `client.empty(out_len * size_of::<fN>())`. Re-confirmed before commit:
`client.empty(` → exactly 2 hits; CPU `kernel_parity` **6/6** bit-exact GREEN (directly
exercises the split kernel through the new uninitialized buffers — proving the kernel's
unconditional 12-cell overwrite fully covers `out`); build clean.

### Re-land rationale (supersedes the original REVERTED verdict below)

- O1 passes the project's **hard** merge gate (cubecl-cpu f64-fold, bit-exact) and is provably
  neutral to the pre-existing hip split defect: the `default_left` flip reproduces identically
  on clean HEAD without O1, and CPU bit-exactness proves every out-cell is written (so `empty()`
  garbage is never observed on any backend).
- O1 is parity-neutral with no perf claim (L3: host round-trip is not the GPU bottleneck) —
  landed as a strict micro-allocation simplification.
- The pre-existing ROCm split-parity defect (below) is being triaged separately via /gsd-debug.

---

### Original executor verdict (REVERTED) — retained for the record

The executor reverted to HEAD under the plan's gate contract; the user subsequently elected
to re-land (see above). The gate findings below were captured during that first pass and remain
accurate (the CPU gate was green with O1; the hip cell was red both with O1 and on clean HEAD).

## Safety re-confirmation (done against live code BEFORE editing)

All three invariants verified true, so the edit itself was correctly scoped and applied:

1. `split_scan_body` (split.rs:144) has **NO early `return`** — the only "return" token in
   lines 144–384 is inside a comment ("when `is_splittable == 0` the host returns ...").
2. The 12 writes execute **unconditionally** at the end of the body and use `=` (never `+=`):
   - f64: `out[ob+0..=11]` at split.rs:372–383.
   - f32 mirror: `out[0..=11]` at split.rs:605–616.
3. Both single-feature launchers use `CubeCount::Static(1,1,1)` + `CubeDim::new_1d(1)`
   (one unit) — f64 at split.rs:811–814, f32 at split.rs:1686–1689.

HARD EXCLUSIONS respected: the multi-feature fused launcher `find_best_splits_fused_kernel`
(its `vec![0.0f64; out_len]` where `out_len = n * 12`, split.rs:1244–1250) was left untouched,
as were all histogram/subtract/partition accumulate/atomic buffers.

## What was applied (then reverted)

The exact_edit from the plan: f64 launcher (~split.rs:799) and f32 launcher (~split.rs:1678)
each had `let zeros = vec![0.0fN; out_len]; let h_out = client.create_from_slice(...)` replaced
with `let h_out = client.empty(out_len * core::mem::size_of::<fN>());` + the documented comments.
Post-edit grep verification was correct:
- `client.empty(` → exactly 2 hits (the two single-feature launchers).
- `vec![0.0f64; out_len]` / `vec![0.0f32; out_len]` in single-feature launchers → 0 hits.
- Fused `vec![0.0f64; out_len]` (out_len = n*12) → still present, untouched.

## Gate results (REAL counts)

### Gate 1 — `cargo build -p lgbm-compute` (with O1 applied): CLEAN

```
   Compiling lgbm-compute v0.1.0 (.../crates/lgbm-compute)
    Finished `dev` profile [unoptimized + debuginfo] target(s) in 0.47s
```

### Gate 2 — CPU bit-exact merge gate `cargo test -p oracle-harness` (with O1 applied): ALL GREEN

| Suite                 | Result            |
| --------------------- | ----------------- |
| kernel_parity         | 6 passed; 0 failed  |
| learner_parity        | 29 passed; 0 failed |
| boosting_parity       | 75 passed; 0 failed |
| raw_bin_train_parity  | 2 passed; 0 failed  |
| rng_parity            | 1 passed; 0 failed  |
| (all other suites)    | 0 failed overall    |

The CPU f64-fold anchor — the hard merge gate — stayed **bit-exact GREEN** at every known
count with O1 applied. (Note: the CPU split kernel exercises the same `empty()` allocation;
on the cpu runtime the kernel's unconditional 12-cell write fully covers the buffer, so CPU
parity was unaffected. The hazard is hip-specific, see below.)

### Gate 3 — ROCm gfx1100 `cargo test -p oracle-harness --features rocm` (with O1 applied)

The load-bearing cell **`kernel_parity_split_within_tol_on_hip` FAILED**. Reported gaps:

```
HIP PARITY GAP `split/forward_winner`           hip=61.250004  cpu=61.25    abs_diff=3.81e-6 > TOL=1e-6
HIP PARITY GAP `split/reverse_winner`           hip=126.15001  cpu=126.15   abs_diff=7.63e-6 > TOL=1e-6
HIP PARITY GAP `split/skip_default_bin_false`   hip=18.150002  cpu=18.15    abs_diff=1.91e-6 > TOL=1e-6
assertion `left == right` failed: HIP split `skip_default_bin_false`: default_left
  left: true   right: false
```

All other hip cells passed (histogram, subtract, partition, fix/compact, resident gather,
batched-from-handle, etc. — `14 passed; 1 failed`).

### Baseline cross-check (CRITICAL finding) — the hip split cell is PRE-EXISTING red

After `git checkout -- crates/lgbm-compute/src/kernels/split.rs` (split.rs == HEAD, verified
zero diff), the **same** cell still FAILS, **deterministically 3/3 reruns**, with the **identical**
`default_left: left: true / right: false` panic at `kernel_parity.rs:1393`.

Conclusion: **`kernel_parity_split_within_tol_on_hip` is broken on the master baseline,
independent of O1.** It is a deterministic hard failure (a `default_left` boolean flip on
`skip_default_bin_false`, plus f32-vs-f64 winner gaps of 1.9e-6 / 3.8e-6 / 7.6e-6 that exceed
the 1e-6 ORACLE_TOL). This is NOT the STATE.md "pre-existingly flaky"
`learner_parity_resident_equals_host_tree_on_hip` cell, and it is NOT wobble — it is
reproducible and deterministic.

## Why REVERTED (decision rationale)

- The plan's gate contract is explicit: "If ANY parity gate REGRESSES due to this change,
  REVERT split.rs — the O1 verdict is 'not worth a parity break'. A clean revert is a valid,
  correct outcome."
- O1 is a **parity-neutral micro-allocation cleanup with no perf claim** (L3 already proved the
  host round-trip is not the GPU bottleneck). There is zero upside to landing a change that
  touches the parity-critical GPU split boundary while that boundary's load-bearing parity cell
  is red and cannot be demonstrated green. Even though the failure is pre-existing rather than
  O1-caused, keeping O1 would (a) provide no measurable benefit and (b) leave an uninitialized-
  memory `empty()` allocation on a hip code path whose split parity is currently unverifiable —
  precisely the stale-memory hazard class the safety section guards against.
- Therefore the correct outcome is a clean revert to HEAD. Repository net change: zero code.

## Follow-up (out of scope for this task — logged for the orchestrator)

`kernel_parity_split_within_tol_on_hip` fails deterministically on the current master baseline:
a `default_left` boolean flip on the `skip_default_bin_false` case plus three f32-vs-f64 split-
winner gaps (1.9e-6 / 3.8e-6 / 7.6e-6) exceeding the 1e-6 ORACLE_TOL. This is a separate,
pre-existing ROCm split-parity defect that predates and is unrelated to O1. It should be
triaged independently (candidate for a /gsd-debug investigation). O1 itself remains feasible
to revisit only after this baseline cell is green and can serve as a regression guard.

## Self-Check: PASSED

- split.rs == HEAD (no edit landed): `git diff --stat crates/lgbm-compute/src/kernels/split.rs`
  → 0 hunks; `git show HEAD:...split.rs | grep` shows the original `vec![0.0fN; out_len]` +
  `create_from_slice` at all three sites (799 f64 single, 1246 f64 fused, 1678 f32 single). FOUND.
- No code commit was made for this task (correctly — change reverted). VERIFIED.
- SUMMARY.md created at the planned path. FOUND.
