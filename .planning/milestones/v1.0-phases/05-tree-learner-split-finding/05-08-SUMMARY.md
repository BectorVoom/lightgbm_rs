---
phase: 05-tree-learner-split-finding
plan: 08
subsystem: testing
tags: [tree-learner, serial-tree-learner, split-finding, leaf-splits, oracle-parity, lightgbm-4.6, kEpsilon, decision_type, cubecl]

# Dependency graph
requires:
  - phase: 05-05
    provides: "offset_for_most_freq_bin single offset rule + compacted offset==1 histogram + oracle-independent routing self-consistency gate (CR-01 closed)"
  - phase: 05-06
    provides: "REAL pip-installed lib_lightgbm 4.6 binary oracle (spine_real.txt / mfb_pos_real.txt) — the goldens that falsified the port and raised CR-03"
provides:
  - "BLOCKER CR-03 CLOSED for the spine corpus: SerialTreeLearner grows trees BIT-EXACT to real lib_lightgbm 4.6 (spine_real.txt) on every learner-authoritative field"
  - "CR-03 structural divergence CLOSED for the mfb>0 corpus: bit-exact on split_feature, threshold (incl. zero sentinel), decision_type, child topology, leaf_count (no 0-row leaf), internal_count, and 3/4 leaf values"
  - "run_forward missing_type-dispatch arg threaded learner.rs -> Backend::find_best_split -> find_best_split_cpu (faithful feature_histogram.hpp:420-429)"
  - "find_best_splits child LeafSplits slot mapping corrected to a direct pass-through (C++ smaller_leaf_splits_/larger_leaf_splits_)"
  - "spine_real gate un-#[ignore]d and passing in the default suite; mfb_pos gate narrowed (assertions unchanged), residual sub-ULP leaf value handed to 05-07"
affects: [05-07]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Per-missing_type scan-branch dispatch (run_forward) is a verbatim transcription of feature_histogram.hpp:420-429, NOT a bin-layout heuristic"
    - "Child LeafSplits are a DIRECT pass-through (smaller_leaf_splits always carries smaller_leaf), mirroring C++ carrying leaf_index_ — never key the slot off smaller==left"
    - "Newton leaf outputs pass through MaybeRoundToZero in the shrinkage finalize (normalizes IEEE -0.0 -> +0), faithful to Tree::Shrinkage"

key-files:
  created:
    - .planning/phases/05-tree-learner-split-finding/05-08-SUMMARY.md
  modified:
    - crates/lgbm-compute/src/kernels/split.rs
    - crates/lgbm-compute/src/lib.rs
    - crates/lgbm-treelearner/src/learner.rs
    - crates/oracle-harness/tests/learner_parity.rs
    - crates/oracle-harness/tests/kernel_parity.rs

key-decisions:
  - "CR-03 PRIMARY root cause was the child LeafSplits slot mapping (Fix B), not the FORWARD-dispatch gate (Fix A); the gate fixed decision_type but the slot swap was what produced structurally-wrong trees (leaf_value -17.99 vs 0.55)"
  - "mfb>0 node-2 leaf-0 residual (2.3e-16, one f64 ULP) DEFERRED to 05-07: it is a kEpsilon cascade from the not-yet-wired subtraction-trick/HistogramPool (05-07's explicit scope); ~4 orders of magnitude inside the <=1e-12 contract; NO assertion weakened"
  - "mfb_pos gate stays #[ignore]d with a narrowed honest reason rather than tolerance-wrapping or deleting its assertions"

patterns-established:
  - "Pattern: run_forward gate (TRUE only for num_bin>2 && missing_type==Zero) threads the C++ template dispatch end-to-end so the kernel runs the exact branch set LightGBM runs"
  - "Pattern: kernel-level goldens (kernel_parity split.txt) pass run_forward=true to replay BOTH branches; the learner uses the per-missing_type dispatch — the two layers test different contracts"

requirements-completed: [TRL-05, TRL-07, TRL-01, TRL-09]

# Metrics
duration: closeout
completed: 2026-06-06
---

# Phase 5 Plan 08: CR-03 Learner-Fix (real lib_lightgbm 4.6 bit-exact) Summary

**BLOCKER CR-03 closed: the Rust SerialTreeLearner grows trees bit-exact to the real lib_lightgbm 4.6 spine golden and structurally bit-exact to the mfb>0 golden, via a corrected child-LeafSplits slot mapping (primary), a missing_type==None FORWARD-dispatch gate, MaybeRoundToZero signed-zero normalization, and the bin-0 kZeroThreshold real-value mapping.**

## Performance

- **Duration:** closeout (fix implemented + diagnosed in prior executor sessions; this session committed the fix, adjusted the two gates per user decision, and closed out)
- **Completed:** 2026-06-06T11:19:55Z
- **Tasks:** 3 plan tasks (Task 1 diagnosis + Task 2 fix in prior commits 061d791/e582cf2; Task 2 fix code + Task 3 gate changes committed this session)
- **Files modified:** 5 (split.rs, lib.rs, learner.rs, learner_parity.rs, kernel_parity.rs)

## Accomplishments

- **CR-03 CLOSED (spine corpus):** `learner_parity_spine_real_binary` now PASSES bit-exact against `spine_real.txt` and runs in the default `cargo test --workspace` suite (un-`#[ignore]`d). Every learner-authoritative field matches: `split_feature=0 0 0`, `threshold=2.5000000000000004 1.5000000000000002 3.5000000000000004`, `decision_type=2 2 2`, `left_child=1 -1 -2`, `right_child=2 -3 -4`, `leaf_count=4 2 2 4`, `internal_count=12 6 6`, shrinkage(0.1)-applied `leaf_value=0.55 -0.10 0.10 -0.55`.
- **CR-03 structural divergence CLOSED (mfb>0 corpus):** the grown tree matches `mfb_pos_real.txt` bit-exact on `split_feature`, `threshold` (incl. the node-2 zero sentinel `1.0000000180025095e-35`), `decision_type=2 2 2`, `left_child=2 -2 -1`, `right_child=1 -3 -4`, `leaf_count=2 6 2 2` (NO 0-row leaf), `internal_count=12 8 4`, and 3 of the 4 leaf values. The prior structural failures (`decision_type[0]=0`, the `[4,6,0,2]` 0-row leaf, the missing zero-sentinel split) are all GONE.
- **Routing self-consistency (CR-01, 05-05) still holds** on both real-bound corpora, and **kernel_parity stays 4/4 bit-exact** (the new `run_forward` arg is passed `true` in the kernel goldens to preserve the both-branch replay).

## Where the fix landed (fix loci — required by plan frontmatter must_haves)

Four faithful C++ transcriptions, in order of impact:

1. **Fix B — child `LeafSplits` slot mapping (PRIMARY root cause).** `find_best_splits` had mapped the smaller/larger `LeafSplits` by `smaller_leaf == left_leaf`, which SWAPPED the slots whenever the smaller child was the right leaf (including the equal-count tie). That fed `smaller_leaf` its sibling's sums (leaf 1 got leaf 0's −24 → spine `leaf_value −17.99`, wrong child splits, wrong topology). Corrected to a DIRECT pass-through (`smaller_leaf_splits` always holds `smaller_leaf`), mirroring C++ `smaller_leaf_splits_`/`larger_leaf_splits_` carrying `leaf_index_` (`serial_tree_learner.cpp:851`). **This is what makes the spine fully bit-exact.**
2. **Fix A — FORWARD-branch dispatch gate.** `run_forward: bool` threaded end-to-end (`learner.rs` → `Backend::find_best_split` → `find_best_split_cpu`); FORWARD runs only for `num_bin>2 && missing_type==Zero` (a verbatim transcription of `feature_histogram.hpp:420-429`). When false, `fwd_count = 0` so only the REVERSE branch contributes and `best_default_left` stays its REVERSE/initial `1.0` (default_left=true → `decision_type==2`). Fixes mfb `decision_type[0]` 0→2.
3. **Fix C — `MaybeRoundToZero` in the shrinkage finalize.** The learner's `-sum_g/(h+l2)` Newton output with `sum_g == +0.0` is IEEE `-0.0`; C++ `Tree::Shrinkage` wraps every shrunk leaf in `MaybeRoundToZero` (`tree.h:191,255-260`, `|fval| <= kZeroThreshold ? 0 : fval`), normalizing `-0.0 → +0`. Fixes mfb leaf 1 `-0` → `0`. Faithful to the C++ finalize, not a weakening.
4. **Fix D — bin-0 `kZeroThreshold` real-value mapping.** `real_upper_bounds_mfb` maps bin 0 → `(1e-35f32 as f64)` == `1.0000000180025095e-35` (the float32 `kZeroThreshold` widened to f64, NOT the f64 literal `1.0000000000000001e-35`). Fixes the mfb node-2 threshold to the zero sentinel.

## CR-03 outcome

- **Spine corpus (most_freq_bin==0):** BIT-EXACT, CLOSED. Gate un-`#[ignore]`d, runs and passes in the default suite.
- **mfb>0 corpus (most_freq_bin==2):** structurally + 3/4 leaf-values BIT-EXACT. The single residual is the node-2 default-bin split's LEFT child (leaf 0): Rust `0.59999999999999976` vs golden `0.59999999999999953`, absolute diff **2.3e-16** (one f64 ULP). DEFERRED to 05-07 per user decision. Root cause: the REVERSE scan seeds `sum_right_hessian = kEpsilon` and FixHistogram reconstructs the most_freq_bin from the leaf's bookkept `sum_hessians_`; matching the golden's `left_h = 2 + 2·kEps` requires the kEpsilon bookkeeping the **subtraction-trick + HistogramPool** produce (`larger = parent − smaller`), which is NOT yet wired in the live growth path (`let _ = subtract_from;`). Wiring it is the explicit scope of plan 05-07. The residual is ~4 orders of magnitude inside the project's ≤1e-12 numerical contract, and **no assertion was weakened** — the mfb gate stays `#[ignore]`d with a narrowed, honest reason and its `assert_real_tree_parity` / `assert_routing_self_consistent` bodies untouched.

## Task Commits

1. **Task 1: Diagnose + localize each CR-03 divergence** — `061d791` (docs: `## CR-03 Localization` appended to 05-08-PLAN.md)
2. **Task 2 (deeper localization):** `e582cf2` (docs: `## CR-03 Task-2 Findings` — 4 fixes + mfb leaf-0 1-ULP → 05-07)
3. **Task 2 (fix) + Task 3 (gate adjustment):** `c564036` (feat: close CR-03 — bit-exact serial learner vs real lib_lightgbm 4.6; Fixes A–D + spine gate un-ignored + mfb gate narrowed)

**Plan metadata:** committed with this SUMMARY + STATE.md + ROADMAP.md.

## Files Created/Modified

- `crates/lgbm-compute/src/kernels/split.rs` — `find_best_split_cpu` (+ f32 mirror) gains the `run_forward` arg gating the FORWARD scan (fwd_count=0 when false)
- `crates/lgbm-compute/src/lib.rs` — `Backend::find_best_split` trait + `CpuBackend` impl carry the `run_forward` arg
- `crates/lgbm-treelearner/src/learner.rs` — `find_best_splits` child LeafSplits direct pass-through (Fix B); `find_best_split_for_leaf` threads `run_forward`; MaybeRoundToZero in the shrinkage path (Fix C)
- `crates/oracle-harness/tests/learner_parity.rs` — spine gate un-`#[ignore]`d (CR-03 closed doc); mfb gate narrowed reason (residual → 05-07), assertions unchanged; `real_upper_bounds_mfb` bin-0 → kZeroThreshold (Fix D)
- `crates/oracle-harness/tests/kernel_parity.rs` — passes `run_forward=true` to preserve the both-branch split.txt golden replay (kernel_parity stays 4/4)

## Decisions Made

- **CR-03 PRIMARY root cause = child LeafSplits slot mapping (Fix B), not the FORWARD gate.** The FORWARD-dispatch gate (Task-1's hypothesized single root cause) fixed `decision_type` but NOT the structurally-wrong trees; the slot swap was the dominant defect. Recorded so future readers don't over-attribute CR-03 to the scan dispatch alone.
- **Defer the mfb leaf-0 ULP to 05-07** (user decision). The residual is a kEpsilon cascade owned by 05-07's subtraction-trick/HistogramPool wiring; it is ~4 orders of magnitude inside the ≤1e-12 contract. The mfb gate stays `#[ignore]`d (assertions intact) rather than weakening it.

## Deviations from Plan

The plan (05-08) anticipated BOTH gates being un-`#[ignore]`d and passing bit-exact. In execution, the spine gate is un-ignored and passes, but the mfb>0 gate retains a one-f64-ULP (2.3e-16) leaf-value residual root-caused to the not-yet-wired subtraction-trick (05-07's explicit scope). This is a **Rule-4 architectural dependency**: the learner fixes in 05-08 are complete and correct; closing the final mfb>0 leaf-0 ULP requires 05-07's HistogramPool/subtraction wiring. Per the user decision, the mfb gate stays `#[ignore]`d with a narrowed, honest reason; no assertion was weakened, tolerance-wrapped, or deleted (T-05-08-02 mitigation upheld).

**Total deviations:** 1 (Rule-4 architectural dependency — mfb leaf-0 ULP deferred to 05-07).
**Impact on plan:** The non-negotiable contract (bit-exact / ≤1e-12) is upheld: spine is bit-exact, mfb>0 is bit-exact on all structural fields + 3/4 leaf values with a 2.3e-16 residual ~4 orders inside the contract. No scope creep; the deferred ULP is squarely 05-07's scope.

## Issues Encountered

None beyond the documented mfb>0 leaf-0 ULP residual (deferred to 05-07).

## User Setup Required

None — no external service configuration required.

## Next Phase Readiness

- **CR-03 is closed enough to unblock 05-07:** the learner reproduces the real oracle bit-exact on the spine and structurally bit-exact on mfb>0. 05-07 ("wire the subtraction-trick + HistogramPool into the live growth path") can now run — and as part of that work it should un-`#[ignore]` `learner_parity_mfb_pos_real_binary` and close the 2.3e-16 leaf-0 ULP, which is precisely the kEpsilon cascade the subtraction-trick/HistogramPool wiring produces.
- **Hand-off to 05-07:** (1) wire subtraction-trick + HistogramPool; (2) the mfb>0 node-2 leaf-0 raw `left_h` must become `2 + 2·kEps` (currently `2 + 1·kEps`); (3) un-`#[ignore]` the mfb gate and confirm bit-exact; (4) routing self-consistency + kernel_parity 4/4 must stay green.

## Verification

- `cargo test --workspace` — GREEN (0 failed). `learner_parity` 10 passed / 1 ignored (mfb); `kernel_parity` 4/4.
- `cargo test -p oracle-harness --test learner_parity` — spine_real PASSES bit-exact in the default suite; mfb_pos stays ignored.
- mfb_pos via `--ignored` — confirmed the ONLY divergence is leaf-0: `left "0.59999999999999976 0 -0.44999999999999984 0.29999999999999988"` vs `right "0.59999999999999953 0 -0.44999999999999984 0.29999999999999988"` (Δ 2.3e-16); every other field bit-exact.
- `LightGBM/` never git-added.

## Self-Check: PASSED

- `05-08-SUMMARY.md` exists on disk.
- Fix commit `c564036` present in history.
- Localization commits `061d791` (Task 1) and `e582cf2` (Task 2 findings) present in history.

---
*Phase: 05-tree-learner-split-finding*
*Completed: 2026-06-06*
