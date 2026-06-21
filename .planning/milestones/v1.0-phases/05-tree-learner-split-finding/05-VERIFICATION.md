---
phase: 05-tree-learner-split-finding
verified: 2026-06-06T13:30:21Z
status: passed
score: 5/5 must-haves verified
overrides_applied: 0
re_verification:
  previous_status: gaps_found
  previous_score: 2/5
  note: >-
    The prior 05-VERIFICATION.md (dated 2026-06-06, initial run) predates plans
    05-07/05-08/05-09 and reported CR-01/CR-02/CR-03 open. All three blockers
    plus WR-01/WR-02 have since been closed and re-validated bit-exact against
    the real lib_lightgbm 4.6 binary. This is a full re-verification overwriting
    that stale report.
  gaps_closed:
    - "CR-01: routing self-consistency (get_leaf tally == data-partition leaf_count) — now asserted in both real-binary gates and across spine/col_wise/col_sampler/real_gh corpora; offset==1 compacted convention unified in a single shared helper"
    - "CR-02: real lib_lightgbm 4.6 oracle now exists (spine_real.txt / mfb_pos_real.txt, committed 05-06); three contradictory inlined offset rules unified into offset_for_most_freq_bin"
    - "CR-03: Rust learner grows trees BIT-EXACT to the real binary on BOTH committed corpora (spine + mfb)"
    - "WR-01/WR-02: subtraction trick + HistogramPool wired into the LIVE find_best_splits growth path; dead `let _ = subtract_from` discard and orphaned `_pool` removed"
    - "mfb>0 node-2 leaf-0 2-ULP residual: closed bit-exact via LeafSplits::init_from_split parent-SplitInfo seed (05-09)"
  gaps_remaining: []
  regressions: []
findings:
  - severity: warning
    title: "Contract-doc inconsistency (non-blocking, flagged by the SUMMARYs themselves)"
    detail: >-
      CLAUDE.md states the numerical-fidelity contract is <=1e-12; REQUIREMENTS.md
      (line 4), PROJECT.md, ROADMAP.md, and STATE.md all state ~1e-6 absolute with
      f32 end-to-end (a documented Phase-1 revision, STATE.md:241). The Phase-5
      learner output is BIT-EXACT f64 vs the real golden (strict %.17g compare,
      zero ULP), which satisfies BOTH framings, so this is non-blocking for the
      phase. 05-09-SUMMARY.md:103 explicitly flags this and recommends the user
      record in PROJECT/ROADMAP that the learner leaf output is enforced bit-exact
      f64. ACTION FOR USER: reconcile the <=1e-12 (CLAUDE.md) vs ~1e-6 (planning
      docs) statements so the contract is stated consistently.
  - severity: warning
    title: "REQUIREMENTS.md TRL-02 checkbox stale (traceability lag, not a code gap)"
    detail: >-
      REQUIREMENTS.md line 38 still marks TRL-02 as `[ ]` with note "reopened:
      gaps_found ... closing in 05-07", and the traceability table line 178 says
      "In Progress (gap closure 05-07)". But 05-07 is COMPLETE (ROADMAP wave 8,
      commit 037e011): the subtraction trick is wired into the live growth path
      and learner_parity_growth_path_subtract PASSES (verified here). The TRL-02
      checkbox/table entry should be flipped to [x]/Complete. This is a stale doc
      entry, not a missing implementation. ACTION FOR USER: update REQUIREMENTS.md
      TRL-02 to complete.
  - severity: info
    title: "mfb>0 / FixHistogram-active integration coverage is via unit tests, not a committed mfb>0 reference tree"
    detail: >-
      The "mfb_pos" corpus was found (05-09, via a real-binary FP execution trace,
      [GSD-META] feature 0 most_freq_bin=0 default_bin=0 offset=1) to actually bin
      with most_freq_bin=0 (sparse-collapse, rate 0.1667 > kSparseThreshold), so
      it exercises the offset==1 path — the SAME as the spine — NOT a
      FixHistogram-active most_freq_bin>0 integration path. The FixHistogram code
      (fix_histogram.rs) is therefore NOT orphaned: it has 4 direct unit tests
      including fix_histogram_reconstructs_most_freq_bin_from_raw_sums and
      fix_histogram_most_freq_bin_zero_is_noop. Observation only: there is no
      committed real-binary REFERENCE TREE that drives the FixHistogram-active
      (most_freq_bin>0 / offset==0) scan+partition path end-to-end; that path's
      integration-level bit-exact coverage against a real golden is thin (unit
      coverage is solid). Not a Phase-5 blocker (the path is unit-validated and no
      committed corpus reaches it), but a candidate for a future most_freq_bin>0
      reference corpus.
---

# Phase 5: Tree Learner + Split Finding Verification Report

**Phase Goal:** Tree Learner + Split Finding — histogram-based serial tree learner, subtraction trick + smaller-child selection, leaf-wise (best-first) growth with num_leaves/max_depth caps, cross-feature split-gain scan with per-split parity, numeric threshold splits + missing/zero/default-bin routing, force_row_wise + force_col_wise, per-tree/per-node feature subsampling — all BIT-EXACT to the real lib_lightgbm 4.6 binary.
**Verified:** 2026-06-06T13:30:21Z
**Status:** passed
**Re-verification:** Yes — after CR-01/CR-02/CR-03 + WR-01/WR-02 + mfb 2-ULP gap closure (plans 05-05…05-09). The prior 05-VERIFICATION.md was a stale initial run (2/5) predating 05-07/08/09.

## Goal Achievement

The keystone serial tree learner is built, substantive (3,075 LOC across
learner/fix_histogram/leaf_splits/data_partition/histogram_pool/col_sampler),
fully wired, and BIT-EXACT to the real `lib_lightgbm` 4.6 binary on BOTH
committed corpora. The two real-binary parity gates — the crux of this phase —
are both un-`#[ignore]`d and PASS bit-exact via a strict `%.17g` compare with NO
tolerance/epsilon/abs_diff wrapper. The goldens were NOT altered to pass (last
changed in 05-06 / commit 6d11d35, never touched by 05-07/08/09). Every blocker
from the prior verification (CR-01 routing inconsistency, CR-02 self-referential
oracle, CR-03 structurally-wrong trees) and both warnings (WR-01/WR-02 dead
subtraction trick) are independently confirmed closed in the codebase, not just
in the SUMMARYs. Two non-blocking documentation findings are surfaced for the
user (contract-doc inconsistency; stale TRL-02 checkbox).

### Observable Truths (Success Criteria)

| # | Truth | Status | Evidence |
| - | ----- | ------ | -------- |
| 1 | Learner selects same split feature/bin/threshold/missing-direction as C++ for every split, per-split candidate-gain validated | ✓ VERIFIED | `learner_parity_spine_per_bin_gains`, `learner_parity_spine_full_tree`, `learner_parity_transcription_crosscheck` PASS; `learner_parity_spine_real_binary` + `learner_parity_mfb_pos_real_binary` assert bit-exact split_feature/decision_type/threshold (%.17g) vs real lib_lightgbm 4.6 goldens — 12 passed / 0 ignored |
| 2 | Subtraction trick reproduces smaller-child selection + derived-child histogram; default-bin-skip scan considers same candidate set | ✓ VERIFIED | `subtract_histograms(parent, smaller)` wired into LIVE `find_best_splits` (learner.rs:734); `learner_parity_growth_path_subtract` asserts derived larger child == direct build cell-for-cell (bit-exact f64) AND tree still matches real spine golden; audit asserts the trick FIRES (non-empty). `learner_parity_subtract` + `kernel_parity_subtract_bit_exact_on_cpu` PASS |
| 3 | Leaf-wise growth respects num_leaves/max_depth; gain formula matches C++ (kEpsilon, lambda_l1/l2, min_gain, min_sum_hessian, min_data, max_delta_step, path_smooth) | ✓ VERIFIED | leaf-wise loop in learner.rs; GainConfig carries all params; spine/real_gh/col goldens (which encode the gain arithmetic) replay bit-exact; leaf_splits.rs init/init_from_sums/init_from_split unit-tested |
| 4 | Numerical threshold splits route missing/zero exactly as C++; data partition (row→leaf) feeds subtraction trick correctly | ✓ VERIFIED | `assert_routing_self_consistent` (CR-01) routes every training row through `tree.get_leaf` and asserts tally == stored data-partition leaf_count — called in BOTH real-binary gates + spine/col_wise/col_sampler/real_gh. mfb>0 zero-sentinel default-bin split threshold `1.0000000180025095e-35` + decision_type=2 bit-exact vs real golden. offset==1 compacted convention unified in `offset_for_most_freq_bin` |
| 5 | Per-tree/per-node feature subsampling RNG-parity selects same features; force_row_wise == force_col_wise == C++ tree | ✓ VERIFIED | `learner_parity_col_sampler_rng` (ColSampler seeded by feature_fraction_seed via bit-exact Random LCG, reset_by_tree/get_by_node) PASS; `learner_parity_row_vs_col` asserts force_row_wise == force_col_wise PASS; both validated against col_wise.txt / col_sampler.txt goldens |

**Score:** 5/5 truths verified

### Required Artifacts

| Artifact | Expected | Status | Details |
| -------- | -------- | ------ | ------- |
| `crates/lgbm-treelearner/src/learner.rs` | SerialTreeLearner leaf-wise loop + find_best_splits + wired subtraction trick | ✓ VERIFIED | 1,598 LOC; subtract_histograms wired at :734; no dead discard |
| `crates/lgbm-treelearner/src/fix_histogram.rs` | FixHistogram most_freq_bin reconstruction | ✓ VERIFIED | 139 LOC; 4 unit tests; used in learner; not orphaned |
| `crates/lgbm-treelearner/src/leaf_splits.rs` | LeafSplits incl. init_from_split (05-09 parent-SplitInfo seed) | ✓ VERIFIED | 229 LOC; init/init_from_sums/init_from_split + 3 unit tests |
| `crates/lgbm-treelearner/src/data_partition.rs` | Row→leaf partition feeding subtraction | ✓ VERIFIED | 224 LOC; split() backend reorder; 3 unit tests |
| `crates/lgbm-treelearner/src/histogram_pool.rs` | Pool slot reuse / Move(left,right) | ✓ VERIFIED | 263 LOC; wired into growth path; LRU evict tested |
| `crates/lgbm-treelearner/src/col_sampler.rs` | feature_fraction(_bynode) RNG parity | ✓ VERIFIED | 338 LOC; reset_by_tree/get_by_node; LCG seeded |
| `crates/lgbm-treelearner/src/lib.rs` (offset_for_most_freq_bin) | Single authoritative offset rule (CR-01/CR-02) | ✓ VERIFIED | offset==1 iff most_freq_bin==0; unit-tested |
| `crates/oracle-harness/tests/learner_parity.rs` | 12 parity gates incl. 2 real-binary | ✓ VERIFIED | 0 #[ignore]; assert_real_tree_parity strict %.17g, no tolerance |
| `crates/oracle-harness/tests/fixtures/learner/spine_real.txt` | Real lib_lightgbm 4.6 spine golden | ✓ VERIFIED | Committed 05-06 (6d11d35), unaltered since |
| `crates/oracle-harness/tests/fixtures/learner/mfb_pos_real.txt` | Real lib_lightgbm 4.6 mfb golden | ✓ VERIFIED | Committed 05-06 (6d11d35), unaltered since (git log + clean working tree confirmed) |

### Key Link Verification

| From | To | Via | Status | Details |
| ---- | -- | --- | ------ | ------- |
| `find_best_splits` | `Backend::subtract_histograms` | `larger = parent − smaller` (learner.rs:734) | ✓ WIRED | Live growth path, audit-proven derived==direct cell-for-cell |
| `find_best_splits` | `HistogramPool` | slot read/move/reuse | ✓ WIRED | parent_slot drives use_subtract; pool.buffer/buffer_mut |
| learner | `offset_for_most_freq_bin` | single shared offset rule | ✓ WIRED | learner.rs:1018/1329; no inlined contradictory rules |
| `assert_real_tree_parity` | `join_g17` / `format_g17` | %.17g strict assert_eq, no tolerance | ✓ WIRED | learner_parity.rs:1099-1157; zero epsilon/abs_diff |
| `tree.get_leaf` (predict) | data-partition `leaf_count` | CR-01 self-consistency tally | ✓ WIRED | assert_routing_self_consistent in both real gates |
| `learner-oracle-capture` xtask | real lib_lightgbm 4.6 wheel | python dumper → committed goldens | ✓ WIRED | Goldens loaded by load_real_tree (fixtures present, asserts run) |

### Behavioral Spot-Checks / Probe Execution

| Behavior | Command | Result | Status |
| -------- | ------- | ------ | ------ |
| Both real-binary parity gates pass bit-exact, 0 ignored | `cargo test -p oracle-harness --test learner_parity` | 12 passed; 0 failed; 0 ignored | ✓ PASS |
| Kernel parity stays 4/4 | `cargo test -p oracle-harness --test kernel_parity` | 4 passed; 0 failed; 0 ignored | ✓ PASS |
| Full workspace green | `cargo test --workspace` | all test-result lines `ok`; 0 failed across crates | ✓ PASS |
| mfb golden not altered post-05-06 | `git log --all -- .../mfb_pos_real.txt` + `git status --short` | only commit 6d11d35; clean working tree | ✓ PASS |
| assert_real_tree_parity has no tolerance wrapper | grep abs_diff/tolerance/epsilon/approx in assert path | only C++ kEpsilon doc-comment refs; strict assert_eq on %.17g | ✓ PASS |
| No #[ignore] in harness | `grep -rn "#[ignore" crates/oracle-harness/tests/` | none | ✓ PASS |
| No debt markers in treelearner | `grep -rnE "TBD|FIXME|XXX|TODO|HACK|unimplemented!|todo!"` | none | ✓ PASS |

### Requirements Coverage

| Requirement | Source Plan(s) | Description | Status | Evidence |
| ----------- | -------------- | ----------- | ------ | -------- |
| TRL-01 | 05-02/03/05/06/07/08/09 | Histogram serial learner (Construct→FindBestSplits→Split) | ✓ SATISFIED | bit-exact vs real binary both corpora |
| TRL-02 | 05-03/07 | Subtraction trick (byte-identical FP path) | ✓ SATISFIED | wired in live path; growth_path_subtract PASS (NOTE: REQUIREMENTS.md checkbox stale — see findings) |
| TRL-03 | 05-03 | Leaf-wise growth, num_leaves/max_depth | ✓ SATISFIED | leaf-wise loop; goldens replay bit-exact |
| TRL-04 | 05-02/03 | Split-gain scan + tie-breaking | ✓ SATISFIED | GainConfig params; per-bin gain golden |
| TRL-05 | 05-01/03/05/06/07/08/09 | Numeric threshold + missing/zero routing | ✓ SATISFIED | zero-sentinel/decision_type bit-exact; CR-01 routing |
| TRL-07 | 05-03/05/08 | Data partition feeding subtraction | ✓ SATISFIED | leaf_count bit-exact, no 0-row leaf; CR-01 holds |
| TRL-08 | 05-04 | Feature subsampling per-tree/per-node | ✓ SATISFIED | col_sampler_rng PASS |
| TRL-09 | 05-04/06/08 | force_row_wise/force_col_wise | ✓ SATISFIED | row_vs_col PASS |

All 8 declared phase requirement IDs are claimed across plans and satisfied. No orphaned requirements (REQUIREMENTS.md maps only these 8 + TRL-06 which is explicitly Phase 7). TRL-06 (categorical) is correctly out of Phase-5 scope.

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
| ---- | ---- | ------- | -------- | ------ |
| — | — | None found | — | No TBD/FIXME/XXX/TODO/HACK/PLACEHOLDER/unimplemented!/todo! in treelearner src or harness; no dead `let _ = subtract_from`; no orphaned pool; no tolerance-weakened assertions; no #[ignore] |

### Human Verification Required

None. The phase's defining contract (bit-exact tree vs a REAL lib_lightgbm 4.6
binary, with train/predict routing self-consistency) is fully validated by the
committed real-binary goldens and the automated `%.17g` gates — the human
verification items in the PRIOR (stale) report (build the real binary, run the
predict round-trip) are now discharged by the committed real-binary oracle
(05-06) and the in-suite `assert_routing_self_consistent` predict round-trip.
The two warning findings above are documentation reconciliations, not test
needs, so they do NOT require human re-testing — they are surfaced for an
explicit project-doc decision.

### Gaps Summary

No gaps. All five Success Criteria verified, all eight requirements satisfied,
the two real-binary parity gates pass bit-exact (12 passed / 0 ignored) with a
strict %.17g compare and unaltered goldens, kernel_parity 4/4, and the full
workspace is green. CR-01/CR-02/CR-03 and WR-01/WR-02 from the prior verification
are confirmed closed in the codebase. Two non-blocking documentation findings
(contract-doc <=1e-12 vs ~1e-6 inconsistency; stale TRL-02 checkbox in
REQUIREMENTS.md) are recorded for the user to reconcile; neither affects goal
achievement because the delivered learner output is bit-exact f64.

---

_Verified: 2026-06-06T13:30:21Z_
_Verifier: Claude (gsd-verifier)_
