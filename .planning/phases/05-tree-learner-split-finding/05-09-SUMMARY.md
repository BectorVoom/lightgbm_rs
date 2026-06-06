---
phase: 05-tree-learner-split-finding
plan: 09
subsystem: tree-learner
tags: [tree-learner, fix-histogram, leaf-splits, fold-order, oracle-parity, lightgbm-4.6, kEpsilon, numerical-fidelity, checkpoint-decision]

# Dependency graph
requires:
  - phase: 05-07
    provides: "subtraction-trick + HistogramPool wired live; mfb>0 leaf-0 ULP re-attributed to FixHistogram-active direct-build f64 fold (and SplitInfo-seed proven to regress leaf 3)"
  - phase: 05-08
    provides: "CR-03 closed — serial learner bit-exact to real lib_lightgbm 4.6 spine + structurally bit-exact mfb>0; the corrected child LeafSplits pass-through"
provides:
  - "DECISIVE LOCALIZATION of the mfb>0 node-2 leaf-0 2-ULP residual: it is the interplay between node-2's leaf-total sum_hessian SEED and the FixHistogram bin-2 reconstruction that consumes that same seed — NOT construct fold order, NOT FixHistogram loop order, NOT reverse-scan accumulation order (all three proven bit-exact / order-independent)"
  - "PROOF that every faithful transcription of the C++ chain (fresh-fold seed, SplitInfo-reported seed per serial_tree_learner.cpp:875-879, and subtraction-trick-derived bins) reproduces Rust's CURRENT 2.0000000000000009 — none reaches the golden-required 2.0000000000000018"
  - "checkpoint:decision raised: bit-exact is not reachable by any analytically-faithful fold-order alignment of the three named hot-path files; the user must decide (accept the 2.3e-16 residual / authorize a real-binary FP execution trace / authorize a deeper output-path investigation)"
affects: [05, 06]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Localization-before-edit gate (plan Task-1 acceptance): a 2-ULP residual is pinned to ONE decisive step via a full-precision construct→FixHistogram→reverse-scan→output trace with .to_bits() at every node BEFORE any behavior change"

key-files:
  created:
    - .planning/phases/05-tree-learner-split-finding/05-09-SUMMARY.md
  modified:
    - crates/oracle-harness/tests/learner_parity.rs

key-decisions:
  - "The mfb>0 node-2 leaf-0 residual is a 2-ULP (golden best_sum_left_hessian 2.0000000000000018 / 0x4000000000000004 vs Rust 2.0000000000000009 / 0x4000000000000002) difference in the leaf-output DENOMINATOR. The plan's prior decode (2.0000000000000013) was approximate; the exact golden-required value, back-solved from the bit-exact leaf_value 0.59999999999999953 and the integer numerator 12.0, is 2.0000000000000018."
  - "The 2-ULP origin is NOT in the construct fold (node-2's bin cells are 2-element integer sums — order-independent), NOT in the FixHistogram subtract-loop order, and NOT in the reverse-scan accumulation order. It is the leaf-total sum_hessian SEED that simultaneously (a) seeds FixHistogram's bin-2 reconstruction and (b) is bumped to feed the scan."
  - "Every FAITHFUL C++ transcription tested analytically (5+ probes) yields Rust's current 2.0000000000000009: fresh-fold seed (4.0 exact), SplitInfo-reported seed (4.0000000000000009 per serial_tree_learner.cpp:875-879, with FixHistogram bin-2 = seed-2-2 = 8.88e-16), and subtraction-trick-derived node-2 bins. The golden 2.0000000000000018 is only reachable by a NON-faithful hybrid (SplitInfo-seed scan with a FRESH-fold FixHistogram), which no single-seed C++ path produces."
  - "Per <critical_context> + plan Task-2: STOP and raise checkpoint:decision rather than (a) introduce any tolerance, (b) re-attempt the LeafSplits-SplitInfo seeding that 05-07 proved regresses leaf 3, or (c) ship a coincidental ULP flip. The mfb gate stays #[ignore]d with its assertions UNCHANGED."

requirements-completed: []

# Metrics
duration: localization+checkpoint
completed: 2026-06-06
---

# Phase 5 Plan 09: FixHistogram f64 Fold-Order Parity — Localization + checkpoint:decision

**The mfb>0 node-2 leaf-0 2.3e-16 residual was DECISIVELY localized to the leaf-total `sum_hessian` seed that simultaneously drives FixHistogram's bin-2 reconstruction and the reverse-scan denominator — NOT to construct/FixHistogram/reverse-scan fold ORDER (all three proven bit-exact). Every faithful transcription of the C++ chain reproduces Rust's current `2.0000000000000009`; the golden requires `2.0000000000000018`, reachable only by a non-faithful hybrid. Per the plan's bit-exact-or-checkpoint contract, this STOPS at a `checkpoint:decision` — no tolerance introduced, no gate weakened, no LeafSplits-seeding re-attempt.**

## Performance

- **Duration:** localization + checkpoint (Task 1 complete; Task 2 reached the bit-exact-or-stop decision gate; Task 3 not run)
- **Completed:** 2026-06-06
- **Tasks:** Task 1 (localization) COMPLETE + committed; Task 2 reached the mandated `checkpoint:decision` STOP (no faithful alignment reaches the golden); Task 3 (full-workspace regression) deferred behind the decision
- **Files modified:** 1 (`learner_parity.rs` — scratch instrumentation added; mfb gate assertions UNCHANGED, still `#[ignore]`d)

## Task 1 — Localization (COMPLETE, commit `c675d3b`)

A scratch `#[test]` (`scratch_05_09_localize_mfb_node2_leaf0`) reproduces the node-2 (root-left, bins {0,1}) leaf-0 chain at full f64 precision with `.to_bits()` at every step. Node-2's 4 rows are {0,1,10,11} (bins {0,1,1,0}, grad {-6,-3,-3,-6}, hess all 1.0); the winning leaf-0 split is the REVERSE candidate at `t == 1` (threshold = `t-1+offset = 0`, the zero sentinel).

**Bit-exact trace of the chain:**

| Step | Value | bits |
|------|-------|------|
| (a) construct bin0 | g=-12.0, h=2.0 | exact (2-element integer sums) |
| (a) construct bin1 | g=-6.0, h=2.0 | exact |
| (a) construct bin2/bin3 | empty (0.0) | exact |
| (b) FixHistogram bin-2 (Rust & C++ order) | h=0.0 | `0x0000000000000000` |
| (c) reverse-scan sum_right_hessian @ t=1 | 2.0000000000000009 | `0x4000000000000002` |
| (c) best_sum_left_hessian (Rust) | 2.0000000000000009 | `0x4000000000000002` |
| GOLDEN best_sum_left_hessian (back-solved) | 2.0000000000000018 | `0x4000000000000004` |

**The single decisive origin:** NOT the three fold ORDERS the plan hypothesized.
- Construct: node-2's non-empty bins each hold a 2-element integer sum (`-6 + -6`, `-3 + -3`, `1 + 1`) — bit-identical in any fold order. Ruled out.
- FixHistogram subtract-loop: `sum_h_raw − Σ(other bins)` in ascending order vs the hand-rolled C++ ascending order — bit-identical (`0.0`). Ruled out.
- Reverse-scan accumulation: `kEps + 0 + 0 + 2.0` is order-independent (one non-zero term). Ruled out.

The origin is the **leaf-total `sum_hessian` SEED** that flows into BOTH the FixHistogram bin-2 reconstruction (`bin2 = seed − bin0 − bin1 − bin3`) AND, bumped (`+2·kEps`), into the reverse-scan denominator. The leaf VALUE divergence (`0.59999999999999976` vs `0.59999999999999953`) is the Newton output `12.0 / (best_sum_left_hessian + λ2)`: a 2-ULP shift in that denominator is the whole defect.

## Task 2 — Faithful alignment attempted analytically; bit-exact UNREACHABLE → checkpoint:decision

The plan's Task-2 action is "align the localized step to its named C++ reference; if after a FAITHFUL alignment the value still diverges, STOP and raise a `checkpoint:decision`." Every faithful candidate was reconstructed at full f64 precision against the authoritative C++ references (`dataset.cpp:1488-1506` FixHistogram, `dense_bin.hpp:99-141` ConstructHistogramInner, `feature_histogram.hpp:854-936` FLOAT reverse scan, `serial_tree_learner.cpp:863-896` child LeafSplits Init, `leaf_splits.hpp:65-70` sum_hessians_ store):

| Candidate (faithful to C++) | node-2 best_sum_left_hessian | leaf-0 shrunk value | matches golden? |
|---|---|---|---|
| Rust current: fresh-fold seed 4.0, fresh FixHistogram (bin2=0) | 2.0000000000000009 (`…002`) | 0.59999999999999976 | NO |
| SplitInfo-reported seed 4.0000000000000009 (serial_tree_learner.cpp:875-879), FixHistogram bin2 = seed−2−2 = 8.88e-16 | 2.0000000000000009 (`…002`) | 0.59999999999999976 | NO |
| Subtraction-trick node-2 bins (parent − sibling) + SplitInfo-seed scan | 2.0000000000000009 (`…002`) | 0.59999999999999976 | NO |
| **GOLDEN (real lib_lightgbm 4.6)** | **2.0000000000000018 (`…004`)** | **0.59999999999999953** | — |
| Non-faithful hybrid: SplitInfo-seed scan + FRESH-fold FixHistogram (bin2=0) | 2.0000000000000018 (`…004`) | 0.59999999999999953 | YES (but not a real C++ path) |

**Conclusion:** the golden `2.0000000000000018` is reachable ONLY by a hybrid in which the scan's `sum_hessian` is the SplitInfo-reported value (`4.0000000000000009`) WHILE FixHistogram's bin-2 reconstruction uses the FRESH leaf total (`4.0`, → bin2 = 0). No single-seed faithful C++ chain produces this: C++ feeds the SAME `smaller_leaf_splits_->sum_hessians()` to both FixHistogram (`serial_tree_learner.cpp:533`) and the scan, and that single-seed chain produces Rust's current `…002`. The unmodeled residual likely lives in the real binary's exact `CalculateSplittedLeafOutput<true,true,USE_SMOOTHING>` / `cnt_factor` / `RoundInt` interplay or a parent-output term that cannot be reproduced without a real-binary FP execution trace.

Per the project's non-negotiable ≤1e-12 bit-exact contract and the plan's explicit "bit-exact or checkpoint, never a tolerance" mandate, this is a **`checkpoint:decision`**, not a place to ship a coincidental ULP flip or a relaxed comparison.

## checkpoint:decision — options for the user

The residual is 2.3e-16 (~4 orders of magnitude INSIDE the ≤1e-12 contract). The mfb gate stays `#[ignore]`d with its `assert_real_tree_parity` body byte-unchanged. Options, recommended first:

1. **(Recommended) Authorize a real-binary FP execution trace.** Build `lib_lightgbm` 4.6 with an instrumented `FindBestThresholdSequentially` / `CalculateSplittedLeafOutput` that dumps node-2 leaf-0's exact `sum_left_hessian`, `cnt_factor`, and `left_output` operands with `.to_bits()`, so the genuine C++ fold producing `2.0000000000000018` is captured directly (closing the one unmodeled step). This is the only path to an ATTRIBUTABLE bit-exact fix. Note: building the real binary requires the un-vendored `external_libs/*` submodules.
2. **Accept the 2.3e-16 residual as a documented, contract-internal sub-ULP** and close the mfb gate at the STRUCTURAL level (every structural field + 3/4 leaf values are already bit-exact), recording the single leaf-0 2-ULP as a known, ≤1e-12-conformant numerical-fidelity note — WITHOUT weakening `assert_real_tree_parity` (e.g. via a separately-named, explicitly-scoped structural gate, leaving the strict gate `#[ignore]`d). Requires an explicit project decision that one sub-ULP leaf-output ULP is acceptable under the f32/~1e-6 contract that STATE.md/ROADMAP currently document (note the CLAUDE.md ≤1e-12 vs STATE.md ~1e-6 contract discrepancy — see Issues).
3. **Authorize a deeper output-path investigation** (out of this plan's three-file scope): trace whether C++ seeds node-2's scan `sum_hessian` and its FixHistogram from DIFFERENT values (the only configuration that reproduces the golden), which would be a Rule-4 architectural finding about the LeafSplits/FixHistogram seam, not a fold-order fix.

Do NOT (per plan): introduce a tolerance, re-attempt the LeafSplits-SplitInfo seeding as the primary fix (05-07 proved it regresses leaf 3 and the analysis above shows it does not even close leaf 0), or delete/weaken the gate.

## Task Commits

1. **Task 1 (localize the 2-ULP origin to the leaf-total sum_hessian seed)** — `c675d3b` (test(05-09): localize node-2 leaf-0 2-ULP origin to leaf-total sum_hessian provenance)
2. **Plan metadata + checkpoint SUMMARY** — committed with this SUMMARY + STATE.md + ROADMAP.md.

## Files Created/Modified

- `crates/oracle-harness/tests/learner_parity.rs` — added `scratch_05_09_localize_mfb_node2_leaf0` (`#[ignore]`d full-precision localization trace; prints the bit patterns above). The mfb gate `learner_parity_mfb_pos_real_binary` is UNCHANGED (still `#[ignore]`d, assertions intact). `assert_real_tree_parity` byte-unchanged.

## Deviations from Plan

The plan assumed a faithful fold-order alignment of one of the three hot-path files (`fix_histogram.rs` / `histogram.rs` / `learner.rs` reverse scan) would close the 2-ULP and let the mfb gate be un-`#[ignore]`d bit-exact. Execution localized the origin to a DIFFERENT seam — the leaf-total `sum_hessian` seed feeding both FixHistogram and the scan — and proved (5+ full-precision probes against the authoritative C++ references) that NO faithful single-seed C++ transcription reaches the golden `2.0000000000000018`; they all reproduce Rust's current `2.0000000000000009`. This is the **Rule-4 / bit-exact-or-checkpoint** branch the plan and `<critical_context>` explicitly provided for: STOP and raise `checkpoint:decision` rather than introduce a tolerance, re-attempt the leaf-3-regressing LeafSplits seeding, or ship a coincidental flip.

**Total deviations:** 1 (Rule-4 / bit-exact-unreachable-by-faithful-alignment → `checkpoint:decision`).
**Impact on plan:** Task 1's localization gate is satisfied with both `.to_bits()` patterns recorded; Task 2's bit-exact target is NOT reachable by an attributable faithful fix within the three-file scope, so per the contract the mfb gate stays `#[ignore]`d (assertions UNCHANGED) and the user owns the decision. The spine gate, growth_path_subtract, kernel_parity, and the full default suite remain GREEN (the scratch test is `#[ignore]`d). No assertion weakened; `LightGBM/` never git-added.

## Issues Encountered

- **Contract discrepancy surfaced (not resolved here):** CLAUDE.md + the 05-09 plan mandate ≤1e-12 bit-exact, while STATE.md/PROJECT.md/ROADMAP currently document an f32 / ~1e-6 contract (a Phase-1 revision). The 2.3e-16 residual is INSIDE both, but the "bit-exact" framing that makes this a blocking gate comes from CLAUDE.md/the plan. The checkpoint:decision (option 2) hinges on which contract the project intends to enforce for the learner leaf output. Flagged for the user.

## Verification

- `cargo test -p oracle-harness --test learner_parity` — 11 passed / 2 ignored (mfb gate + the Task-1 scratch trace), 0 failed.
- `learner_parity_spine_real_binary` — PASSES bit-exact (no regression).
- `learner_parity_growth_path_subtract` — PASSES (subtraction-trick wiring unregressed).
- `learner_parity_mfb_pos_real_binary` — stays `#[ignore]`d, assertions UNCHANGED; under `--ignored` it still fails on exactly leaf-0 (`0.59999999999999976` vs `0.59999999999999953`).
- `LightGBM/` never git-added (`git status --porcelain LightGBM/` empty of staged entries).

## Self-Check: PASSED

- `05-09-SUMMARY.md` exists on disk.
- Task-1 commit `c675d3b` present in history.

---
*Phase: 05-tree-learner-split-finding*
*Completed: 2026-06-06 (localization + checkpoint:decision)*
