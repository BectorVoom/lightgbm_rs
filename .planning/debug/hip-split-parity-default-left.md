---
slug: hip-split-parity-default-left
status: resolved
trigger: "kernel_parity_split_within_tol_on_hip fails deterministically on master (gfx1100). default_left boolean flip (hip true vs cpu false) on skip_default_bin_false case, plus f32-vs-f64 split-winner gaps exceeding 1e-6 ROCm tol. Suspect: f32 scan path find_best_split_kernel_f32 / split_scan_body in crates/lgbm-compute/src/kernels/split.rs."
created: 2026-06-09
updated: 2026-06-09
---

# Debug Session: hip-split-parity-default-left

## Symptoms

**Expected behavior:**
`kernel_parity_split_within_tol_on_hip` passes — the hip (f32) single-feature split kernel's
winner cells match the cubecl-cpu f64 anchor within the 1e-6 ROCm tolerance, and the
`default_left` flag matches the anchor on every case (including `skip_default_bin_false`).

**Actual behavior:**
Test fails deterministically (3/3 reruns on clean HEAD). Two coupled symptoms:
1. **`default_left` boolean flip** on the `skip_default_bin_false` case: hip reports `true`,
   cpu anchor reports `false`. This is a correctness flip (a wrong split direction), not a
   tolerance gap.
2. **f32-vs-f64 split-winner magnitude gaps exceeding the 1e-6 tol:**
   - `split/forward_winner`: hip=61.250004 cpu=61.25 → abs_diff=3.81e-6
   - `split/reverse_winner`: hip=126.15001 cpu=126.15 → abs_diff=7.63e-6
   - `split/skip_default_bin_false`: hip=18.150002 cpu=18.15 → abs_diff=1.91e-6

**Error messages:**
```
assertion `left == right` failed: HIP split `skip_default_bin_false`: default_left
  left: true   right: false
```
Panic at `crates/oracle-harness/tests/kernel_parity.rs:1393`.

**Timeline:**
Pre-existing on the master baseline — reproduces 3/3 on clean HEAD with no local changes.
Discovered 2026-06-09 while landing quick task 260609-b1a (O1 `empty()` out-cells, commit
58589fb). O1 is UNRELATED and confirmed neutral: the same failure reproduces with and without
O1, and CPU bit-exact kernel_parity is 6/6 green. NOT the STATE.md known-flaky cell
`learner_parity_resident_equals_host_tree_on_hip` (that is ~1e-6 f32-atomic wobble; this is a
deterministic hard failure).

**Reproduction:**
```
cargo test -p oracle-harness --features rocm kernel_parity_split_within_tol_on_hip
```
on the local gfx1100 ROCm GPU. All other hip kernel_parity cells (histogram, subtract,
partition, fix/compact, resident gather, batched-from-handle) pass — only the single-feature
split cell fails.

## Initial suspect (from reporter — hypothesis to test, not a conclusion)

The f32 scan path: `find_best_split_kernel_f32` and the shared `split_scan_body` (run in f32
via the f32 mirror) in `crates/lgbm-compute/src/kernels/split.rs`. Candidate root-cause areas:
- The FORWARD/REVERSE winner selection (`best_default_left` encoding: REVERSE=1.0/FORWARD=0.0)
  diverging from the f64 path under f32 rounding — a near-tie gain where f32 rounding picks the
  REVERSE (default_left=true) branch while f64 picks FORWARD (default_left=false).
- The `default_bin` / `skip_default_bin` handling specific to the `skip_default_bin_false` case.
- f32 accumulation of `g²/(h+λ)` gains crossing `min_gain_shift`/`best_gain` comparison
  boundaries differently than f64 (the magnitude gaps suggest f32 gain math, which can tip a
  near-tie winner and thus flip default_left).

Note for investigator: per CLAUDE.md the ROCm f32 path is a ~1e-6 BEST-EFFORT gate, not the
hard merge gate (cubecl-cpu f64-fold is the bit-exact gate). Determine whether this is (a) a
genuine logic bug in the f32 split path, or (b) an inherent f32-vs-f64 near-tie sensitivity
that the test's fixture/tolerance should account for — the fix differs accordingly.

## Evidence

- timestamp: 2026-06-09
  source: fixture decode (crates/oracle-harness/tests/fixtures/kernels/split.txt:33-37)
  finding: |
    `skip_default_bin_false` histogram is per-bin gradient [-10,-10,-10,1], hessian
    [5,5,5,5], offset=0, num_bin=4, skip_default_bin=0 (missing_type None). Winner in
    the golden: threshold=2, default_left=0 (FORWARD), gain finalized 18.15.
    The split at threshold=2 partitions left={0,1,2} right={3} — a SINGLE physical
    split that BOTH branches can record (REVERSE as t-1+offset, FORWARD as t+offset).

- timestamp: 2026-06-09
  source: candidate-gain decode (SCAND_REV / SCAND_FWD f64 bit patterns)
  finding: |
    REVERSE candidate gains (high→low t): [60.19999999999999, 48.0999..., 44.0666...]
    FORWARD candidate gains (low→high t): [44.0666..., 48.0999..., 60.2]
    The two extremal candidates are the SAME physical split at threshold=2:
      REVERSE best = 60.19999999999999 (f64)   default_left=true
      FORWARD best = 60.2               (f64)   default_left=false
    In f64 the FORWARD gain is STRICTLY GREATER by ~1 ULP (60.2 > 60.19999999999999),
    so the kernel's `take = cand_gain > best_gain` (strict >, REVERSE scanned first,
    FORWARD second) lets FORWARD overtake → default_left=false (matches anchor).
    In f32 BOTH round to the IDENTICAL value 60.20000076293945, so FORWARD's
    `60.20000076 > 60.20000076` is FALSE — FORWARD does NOT overtake, REVERSE stays
    the winner → default_left=true. THE FLIP.

- timestamp: 2026-06-09
  source: magnitude-gap analysis (forward_winner / reverse_winner / skip_default_bin_false)
  finding: |
    The three reported net-gain gaps are pure f32 rounding of `g²/(h+λ)` (squaring
    magnifies absolute error): relative error ~6e-8..1.1e-7 (right at f32 eps), but
    the gain magnitudes (61, 126, 18) push the ABSOLUTE diff above the fixed 1e-6
    ORACLE_TOL. These are ALREADY non-blocking: assert_within (kernel_parity.rs:1157)
    surfaces a >1e-6 gap to stderr but only HARD-FAILS on the generous relative
    HIP_SANITY_REL=1e-3 bound (all three pass it, rel << 1e-3). So the magnitude gaps
    are NOT the failure.

- timestamp: 2026-06-09
  source: assertion-site read (kernel_parity.rs:1393)
  finding: |
    The ONLY hard failure is the raw `assert_eq!((hip_raw[9]!=0.0), si.default_left)`
    at line 1393 — it sits OUTSIDE the tolerance-aware assert_within helper and
    demands EXACT bool equality even when the winner is a genuine f32-vs-f64 tie
    (identical threshold, identical left_count, gains equal within f32 precision).

## Root Cause

This is category (b): an INHERENT f32-vs-f64 near-tie sensitivity, NOT a logic bug
in the f32 split path. The f32 kernel (`find_best_split_kernel_f32`) is a faithful
1:1 transcription of the f64 anchor — same gate order, same strict-`>` tie-break,
same eps placements. The `skip_default_bin_false` fixture is constructed so REVERSE
and FORWARD produce the SAME physical split at threshold=2; the anchor's
default_left=false is decided ENTIRELY by a sub-ULP f64 gain difference (60.2 vs
60.19999999999999) that vanishes under f32 rounding, leaving the f32 kernel with an
exact tie that its strict-`>` (keep-first = REVERSE) resolves to default_left=true.
No f32 code change can recover a distinction that does not exist at f32 precision
without diverging from the C++/anchor strict-`>` keep-first tie-break.

Correct fix = TEST HARNESS (fixture/comparison), per the orchestrator's category-(b)
guidance: do NOT loosen the magnitude tolerance (already correctly non-blocking) and
do NOT touch the bit-exact f64 anchor. The default_left assert must allow a flip ONLY
when the split is a verified f32 tie: same threshold AND same left_count AND the two
branches' winning net gains are equal within f32 precision. A default_left flip on a
split that is NOT a tie would still hard-fail (a real bug stays caught).

## Current Focus

- hypothesis: CONFIRMED — f32-vs-f64 near-tie sensitivity flips default_left on the
  symmetric REVERSE/FORWARD same-physical-split tie; the exact-bool assert at
  kernel_parity.rs:1393 is too strict for category-(b) f32 ties.
- next_action: DONE — fix applied and verified.

## Resolution

- root_cause: |
    Inherent f32-vs-f64 near-tie sensitivity (category b), NOT an f32 logic bug. The
    `skip_default_bin_false` fixture's REVERSE and FORWARD branches record the SAME
    physical split at threshold=2 (left={0,1,2}, right={3}) with opposite default_left.
    The f64 anchor picks FORWARD (default_left=false) ONLY because FORWARD's gain 60.2
    is ~1 f64 ULP above REVERSE's 60.19999999999999; both round to the identical f32
    60.20000076, so the kernel's strict-`>` keep-first tie-break leaves REVERSE the
    winner under f32 → default_left=true. The f32 kernel is a faithful transcription;
    the failure was the test's raw exact-bool default_left assert (kernel_parity.rs:1393)
    being too strict for a verified f32 tie. (The 1e-6 magnitude gaps were already
    correctly non-blocking via assert_within's surface-not-fail + HIP_SANITY_REL gate.)
- fix: |
    Relaxed ONLY the default_left assertion in
    crates/oracle-harness/tests/kernel_parity.rs to a tie-aware comparison: a flip is
    allowed (and surfaced to stderr for the 04-ROCM-GAPS.md ledger) ONLY when the
    winning split is a verified f32 tie — same threshold AND same left_count AND net
    gains equal within f32 precision (HIP_SANITY_REL). A default_left flip on any
    NON-tie split still hard-fails, so a real wrong-direction bug stays caught. No
    change to the f64 cpu anchor, the f32 kernel, the magnitude tolerance, or any
    production code. Verified: hip split parity 3/3 green (was 0/3); CPU bit-exact
    kernel_parity 6/6 green.
- files_touched:
  - crates/oracle-harness/tests/kernel_parity.rs (default_left tie-aware assert)
- verification: |
    cargo test -p oracle-harness --features rocm kernel_parity_split_within_tol_on_hip → ok (3/3 reruns)
    cargo test -p oracle-harness --test kernel_parity (no rocm, CPU anchor) → 6/6 ok
