---
phase: 05-tree-learner-split-finding
plan: 07
subsystem: tree-learner
tags: [tree-learner, serial-tree-learner, subtraction-trick, histogram-pool, find-best-splits, lightgbm-4.6, oracle-parity, fixhistogram, cubecl]

# Dependency graph
requires:
  - phase: 05-05
    provides: "offset_for_most_freq_bin single offset rule + compacted offset==1 histogram + oracle-independent routing self-consistency gate (CR-01 closed)"
  - phase: 05-06
    provides: "REAL pip-installed lib_lightgbm 4.6 binary oracle (spine_real.txt / mfb_pos_real.txt) — the goldens validating the wired path bit-exact"
  - phase: 05-08
    provides: "CR-03 closed — serial learner bit-exact to real lib_lightgbm 4.6 spine + structurally bit-exact mfb>0; the corrected child LeafSplits pass-through this plan builds on"
provides:
  - "WR-01 CLOSED: HistogramPool slots are READ/reused in the live find_best_splits growth path (no orphaned _pool)"
  - "WR-02 CLOSED: the LARGER child's histogram is DERIVED by subtraction (parent − smaller via Backend::subtract_histograms) in the live growth path (no `let _ = subtract_from;`), mirroring C++ serial_tree_learner.cpp:364-378"
  - "learner_parity_growth_path_subtract gate: derived-larger-child == direct build cell-for-cell (bit-exact f64) AND the spine tree stays bit-exact after wiring"
  - "mfb>0 leaf-0 2.3e-16 ULP RE-ATTRIBUTED: 05-08's subtraction-trick framing DISPROVEN; root cause is FixHistogram-active f64 accumulation/fold order; deferred to NEW plan 05-09"
affects: [05-09, 06]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Live subtraction-trick growth path: SMALLER child built directly via construct_histograms, LARGER child = parent − smaller via Backend::subtract_histograms (never a second direct build) — mirroring C++ ConstructHistograms/use_subtract"
    - "HistogramPool slots are read in find_best_splits (parent fetched, children written) — the D-05 pool is live, not orphaned"
    - "Derived-vs-direct histogram equivalence is bit-exact for f64 cells — proven by learner_parity_growth_path_subtract, so wiring the trick changes the DERIVATION path only, not the output"

key-files:
  created:
    - .planning/phases/05-tree-learner-split-finding/05-07-SUMMARY.md
  modified:
    - crates/lgbm-treelearner/src/learner.rs
    - crates/lgbm-treelearner/src/histogram_pool.rs
    - crates/oracle-harness/tests/learner_parity.rs

key-decisions:
  - "WR-01/WR-02 CLOSED: subtraction-trick (larger = parent − smaller) + HistogramPool slot reuse wired into the live find_best_splits growth path; dead orchestration (`let _ = subtract_from;`, orphaned `_pool`) eliminated; the wired path stays bit-exact to the real lib_lightgbm 4.6 spine golden"
  - "mfb>0 node-2 leaf-0 2.3e-16 ULP did NOT close with the subtraction wiring. ROOT CAUSE CORRECTED: leaf 0 is node-2's DIRECTLY-BUILT smaller child, so the subtraction trick never touches it. The 05-08 hand-off attribution (kEpsilon cascade from the not-yet-wired subtraction-trick/HistogramPool) is DISPROVEN. The residual is a 2-ULP f64 ACCUMULATION-ORDER subtlety in the FixHistogram-active DIRECT histogram build (construct/FixHistogram/output fold). Deferred to NEW plan 05-09."
  - "Trialed the C++-faithful SplitInfo-sum LeafSplits seeding (serial_tree_learner.cpp:850-870): it did NOT change leaf 0 and REGRESSED leaf 3 — so 05-09 must target the construct/FixHistogram fold order, NOT LeafSplits seeding"
  - "mfb_pos gate stays #[ignore]d with a CORRECTED honest reason (FixHistogram f64 fold-order → 05-09); NO assertion weakened, tolerance-wrapped, or deleted; 2.3e-16 is ~4 orders of magnitude inside the ≤1e-12 contract"

patterns-established:
  - "Pattern: the subtraction trick is validated TWO ways — (1) derived-larger-child == independent direct build bit-exact (learner_parity_growth_path_subtract), (2) the grown tree still matches the real lib_lightgbm goldens — so a wiring regression fails loudly on either axis"
  - "Pattern: a residual that survives wiring the subtraction trick localizes to the DIRECT build path (FixHistogram fold order), because the subtraction trick is proven bit-exact to a direct build — the two gates jointly isolate where an FP residual lives"

requirements-completed: [TRL-01, TRL-02, TRL-05]

# Metrics
duration: closeout
completed: 2026-06-06
---

# Phase 5 Plan 07: Subtraction-Trick + HistogramPool Growth-Path Wiring Summary

**WR-01/WR-02 closed: the subtraction trick (`larger = parent − smaller` via `Backend::subtract_histograms`) and HistogramPool slot reuse are wired into the LIVE `find_best_splits` growth path (mirroring C++ `serial_tree_learner.cpp`), the derived-larger-child is bit-exact to a direct build, and the spine tree stays bit-exact to the real lib_lightgbm 4.6 golden; the lone mfb>0 leaf-0 2.3e-16 ULP is re-attributed from the subtraction trick to a FixHistogram-active f64 fold-order subtlety and deferred to new plan 05-09.**

## Performance

- **Duration:** closeout (implementation + the disproven-attribution diagnosis in prior executor sessions; this session corrected the mfb gate reason, wrote this SUMMARY, and updated STATE/ROADMAP)
- **Completed:** 2026-06-06
- **Tasks:** 2 plan tasks (Task 1 wire subtraction-trick + pool; Task 2 growth-path-subtract parity gate) — implemented + committed in `037e011`
- **Files modified:** 3 (learner.rs, histogram_pool.rs, learner_parity.rs)

## Accomplishments

- **WR-02 CLOSED — subtraction trick wired into the live growth path.** `find_best_splits` now builds only the SMALLER child directly via `construct_histograms` and DERIVES the LARGER child as `parent − smaller` via `Backend::subtract_histograms` (learner.rs:736), exactly as C++ `serial_tree_learner.cpp:364-378`. The dead `let _ = subtract_from;` discard is removed — `subtract_from` now selects the parent slot to subtract from. The root still builds the single leaf directly.
- **WR-01 CLOSED — HistogramPool is live.** The `_pool` parameter is renamed to `pool` and READ: the parent leaf's retained histogram is fetched from its pool slot, and the two children's histograms occupy pool slots (smaller built, larger derived). The D-05 pool/eviction machinery is reused unchanged.
- **`learner_parity_growth_path_subtract` added.** On a corpus with a non-root split (right_leaf ≥ 0), it asserts the larger child's derived histogram equals an independent direct `construct_histograms` of that leaf's rows CELL-FOR-CELL (bit-exact f64), AND that the grown spine tree still matches the real lib_lightgbm 4.6 golden — proving the wiring changed the DERIVATION path only, not the output.

## Validation

- `cargo test --workspace` — GREEN (0 failed). `learner_parity` 11 passed / 1 ignored (mfb); `kernel_parity` 4/4 bit-exact.
- Spine real gate (`learner_parity_spine_real_binary`) — BIT-EXACT in the default suite, unaffected by the wiring.
- Routing self-consistency (CR-01, 05-05) still holds; `kernel_parity` stays 4/4.
- `LightGBM/` never git-added.

## The Deferral — mfb>0 leaf-0 ULP re-attributed (05-08 framing DISPROVEN)

The mfb>0 corpus stays bit-exact on every structural field + 3/4 leaf values; the single residual is node-2's leaf-0 value: Rust `0.59999999999999976` vs golden `0.59999999999999953`, **Δ = 2.3e-16 (one f64 ULP)**. Wiring the subtraction trick did NOT close it.

**Root cause CORRECTED.** Leaf 0 is node-2's **directly-built smaller child**, so the subtraction trick (`larger = parent − smaller`) NEVER touches it. The 05-08 hand-off attribution — that the residual was a kEpsilon cascade owned by the not-yet-wired subtraction-trick/HistogramPool — is therefore **DISPROVEN**. The residual is a **2-ULP f64 accumulation-order subtlety in the FixHistogram-active DIRECT histogram build** (the construct / FixHistogram / output fold), not a subtraction-trick effect.

**Diagnostic that rules out LeafSplits seeding:** the prior executor trialed the C++-faithful SplitInfo-sum `LeafSplits` seeding (`serial_tree_learner.cpp:850-870`). It did NOT change leaf 0 and REGRESSED leaf 3 — so **05-09 must target the construct/FixHistogram fold order, NOT LeafSplits seeding.**

**Deferred to NEW plan 05-09** (FixHistogram f64 fold-order parity). The `mfb_pos` gate stays `#[ignore]`d with a CORRECTED, honest reason; NO assertion was weakened, tolerance-wrapped, or deleted. 2.3e-16 is ~4 orders of magnitude inside the project's ≤1e-12 contract.

## Task Commits

1. **Task 1 + Task 2 (wire subtraction-trick + HistogramPool into find_best_splits; add growth_path_subtract gate)** — `037e011` (feat: wire subtraction-trick + HistogramPool into find_best_splits growth path)
2. **mfb gate #[ignore] reason correction (re-attribute leaf-0 ULP to FixHistogram f64 fold-order → 05-09)** — `fbd4f1d` (docs)

**Plan metadata:** committed with this SUMMARY + STATE.md + ROADMAP.md.

## Files Created/Modified

- `crates/lgbm-treelearner/src/learner.rs` — `find_best_splits` derives the larger child by `subtract_histograms` in the growth path; `let _ = subtract_from;` removed; `_pool` → `pool` (read)
- `crates/lgbm-treelearner/src/histogram_pool.rs` — pool slot accessors wired for the live parent-fetch / child-write
- `crates/oracle-harness/tests/learner_parity.rs` — `learner_parity_growth_path_subtract` gate (derived==direct + spine still bit-exact); mfb gate `#[ignore]` reason corrected (FixHistogram f64 fold-order → 05-09), assertions UNCHANGED

## Decisions Made

- **WR-01/WR-02 closed by wiring the trick + pool into the live path** (the keystone D-05 orchestration now actually executes, not just passes in isolation); the wired path stays bit-exact to the real spine golden.
- **mfb>0 leaf-0 ULP root cause CORRECTED to FixHistogram f64 accumulation/fold order** (leaf 0 is directly built, untouched by subtraction); the 05-08 subtraction-trick attribution is disproven; deferred to 05-09.
- **05-09 targets the construct/FixHistogram fold order, not LeafSplits seeding** (the SplitInfo-sum seeding trial regressed leaf 3 without touching leaf 0).
- **mfb gate stays `#[ignore]`d with a corrected honest reason; no assertion weakened.**

## Deviations from Plan

The plan (05-07) anticipated the subtraction-trick wiring closing the mfb>0 leaf-0 2.3e-16 ULP (per the 05-08 hand-off framing). In execution the wiring closed WR-01/WR-02 and stayed bit-exact on the spine, but the mfb>0 leaf-0 ULP did NOT close — and diagnosis DISPROVED the subtraction-trick attribution: leaf 0 is node-2's directly-built smaller child, so the residual is a FixHistogram-active f64 fold-order subtlety, not a subtraction-trick effect. This is a **Rule-4 architectural finding**: closing the residual requires aligning the direct-build FixHistogram fold order with C++ `ConstructHistograms`/`FixHistogram`, which is squarely a new plan's scope (05-09). Per the user decision (Option A), 05-07 is accepted complete for its WR-01/WR-02 scope and the residual is deferred to 05-09; the mfb gate stays `#[ignore]`d with a corrected reason and NO assertion was weakened.

**Total deviations:** 1 (Rule-4 architectural finding — mfb leaf-0 ULP re-attributed to FixHistogram f64 fold-order, deferred to 05-09).
**Impact on plan:** The non-negotiable contract (bit-exact / ≤1e-12) is upheld — spine bit-exact, mfb>0 bit-exact on all structural fields + 3/4 leaf values with a 2.3e-16 residual ~4 orders inside the contract. WR-01/WR-02 (this plan's scope) are fully delivered and validated. No scope creep; the deferred ULP is correctly re-scoped to 05-09.

## Issues Encountered

The mfb>0 leaf-0 2.3e-16 ULP survived the subtraction-trick wiring, falsifying the 05-08 attribution. Resolved by localizing the residual to node-2's directly-built smaller child (leaf 0) — proving it lives in the FixHistogram-active direct build, not the subtraction path — and deferring it to a correctly-scoped new plan 05-09.

## User Setup Required

None — no external service configuration required.

## Next Phase Readiness

- **WR-01/WR-02 closed; the keystone subtraction-trick + pool orchestration is live and bit-exact on the spine.** TRL-01/TRL-02/TRL-05 satisfied for the wired growth path.
- **Hand-off to 05-09:** close the mfb>0 node-2 leaf-0 2.3e-16 ULP by aligning the Rust FixHistogram-active direct-build f64 accumulation/fold order with C++ `ConstructHistograms`/`FixHistogram`; then un-`#[ignore]` `learner_parity_mfb_pos_real_binary` and assert bit-exact. Do NOT re-attempt LeafSplits seeding (it regresses leaf 3).
- Phase 5 has one remaining gap-closure plan (05-09, needs planning) before the learner is fully bit-exact on both real corpora.

## Self-Check: PASSED

- `05-07-SUMMARY.md` exists on disk.
- Implementation commit `037e011` present in history.
- Gate-correction commit `fbd4f1d` present in history.

---
*Phase: 05-tree-learner-split-finding*
*Completed: 2026-06-06*
