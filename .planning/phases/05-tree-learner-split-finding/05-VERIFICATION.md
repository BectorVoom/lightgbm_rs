---
phase: 05-tree-learner-split-finding
verified: 2026-06-06T00:00:00Z
status: gaps_found
score: 3/5 must-haves verified
overrides_applied: 0
gaps:
  - truth: "Numerical threshold splits route missing/zero exactly as C++; data partition (row→leaf) feeds the subtraction trick correctly (SC#4 / TRL-05, TRL-07)"
    status: failed
    reason: >-
      CR-01 (confirmed by code review AND independently re-reproduced here):
      for a `most_freq_bin == 0` feature the port encodes `offset == 0` with a
      non-compacted histogram, so the FORWARD scan records
      `threshold = t + offset = t` (split.rs:299, "left = bins <= threshold").
      But the data-partition kernel still applies the C++
      `if most_freq_bin == 0 { th -= 1 }` (partition.rs:59-61), routing
      "left = bins < threshold". The two boundaries disagree on the bin equal to
      `threshold`. Independent reproduction of the spine feature-0 layout
      (6 bins, 2 rows/bin, mfb=0, t=2, offset=0) gives data-partition
      leaf_count = [4, 8] while routing the same 12 rows through the grown
      tree's own `Tree::get_leaf` (tree.rs:170, `fval <= threshold` → left)
      gives [6, 6]. The serialized `leaf_count`/`internal_count`/leaf outputs
      are therefore computed for a DIFFERENT partition than the model predicts
      into — a silent train/predict inconsistency and a ≥1e-12 fidelity
      violation against the project's non-negotiable contract. The parity
      goldens do NOT catch it because `learner_capture.cpp::PartitionLeaf`
      hard-codes the identical `--th`-with-`offset==0` convention, so both sides
      agree with each other while both diverge from a predict-consistent
      partition; `learner_parity_*` only compares tree TEXT, never train-vs-predict
      routing.
    artifacts:
      - path: "crates/lgbm-compute/src/kernels/partition.rs"
        issue: "Lines 59-61 apply `--th` for most_freq_bin==0, but the stored threshold (offset==0, non-compacted layout) did not bake in the offset — double-counts the adjustment vs the scan."
      - path: "crates/lgbm-compute/src/kernels/split.rs"
        issue: "Line 299 records `best_threshold = t + offset` with offset==0 for mfb=0; inclusive `<=` boundary that the partition's `--th` then breaks."
      - path: "crates/lgbm-model/src/tree.rs"
        issue: "Line 170 `fval <= threshold` (predict) routes `bin <= threshold` left, inconsistent with the partition's `bin < threshold`."
      - path: "crates/oracle-harness/tests/learner_parity.rs"
        issue: "No train-vs-predict routing assertion: every spine/real_gh tree's get_leaf row-routing is never checked against the stored data-partition leaf_count, so the divergence passes the gate."
    missing:
      - "Make the partition boundary consistent with the stored threshold for the non-compacted layout: either NOT apply `--th` when the port uses offset==0 for a most_freq_bin==0 feature, OR adopt the real-LightGBM convention end-to-end (offset==1 + a compacted histogram)."
      - "Add a parity assertion in learner_parity.rs that the grown tree's `get_leaf` routing of every training row reproduces the data-partition leaf_count exactly (the regression test that fails today)."
  - truth: "Both force_row_wise and force_col_wise scan+partition the mfb>0 / offset==1 path correctly against a real C++ reference (SC#1, SC#4 coverage of the offset==1 branch / TRL-05, TRL-09)"
    status: partial
    reason: >-
      CR-02: the `FeatureColumn.offset` invariant is documented as "1 when
      most_freq_bin == 0, else 0" (learner.rs:91-94) but is used the OPPOSITE
      way in every committed corpus — the spine, col_wise, and col_sampler
      corpora set `offset: 0` with `most_freq_bin: 0` (learner_parity.rs:231,
      235, 243, 247, 614, 618), and the real_gh parser uses a third contradictory
      rule `offset: if most_freq_bin == 0 { 0 } else { 1 }` (line 768, inverted
      vs both the doc and vs LightGBM). `learner_capture.cpp` never derives
      `offset` from `most_freq_bin`; it transcribes whatever the corpus
      hard-codes. Because both Rust and C++ use the same (non-LightGBM) offset
      value, the parity gate is self-consistent but validates nothing about real
      `lib_lightgbm` fidelity for the offset==1 path. The ONLY feature layout
      exercised bit-exact against a committed reference tree is `most_freq_bin == 0`
      (the layout CR-01 shows is mis-partitioned). `learner_parity_missing_routing`
      uses `most_freq_bin: 1` but only asserts `total == 8` row conservation —
      never a C++ golden tree. There is ZERO bit-exact coverage of the
      `offset == 1` / `most_freq_bin > 0` scan+partition path.
    artifacts:
      - path: "crates/lgbm-treelearner/src/learner.rs"
        issue: "offset doc (91-94) contradicts every corpus's hard-coded offset:0; no single helper derives offset from most_freq_bin."
      - path: "crates/oracle-harness/tests/learner_parity.rs"
        issue: "Three contradictory inlined offset rules; missing_routing only asserts row conservation, no PTREE reference for mfb>0."
    missing:
      - "Pick ONE offset convention, document it as authoritative, derive offset from most_freq_bin in a single shared helper used by both learner and harness."
      - "Add a most_freq_bin > 0 corpus with a committed C++ reference tree (PTREE) so the offset==1 path is validated bit-exact; until then reject most_freq_bin > 0 with a typed error rather than silently growing an unvalidated tree."
human_verification:
  - test: "Build real lib_lightgbm 4.6 (deterministic=true, force_row_wise=true, num_threads=1, fixed seed) on the spine + real_gh corpora and dump the model text, then compare against the Rust learner's grown tree text byte-for-byte."
    expected: "Identical tree text (split feature/threshold/missing-direction, leaf_count, internal_count, leaf_value) on every node."
    why_human: "The committed goldens are a hand-transcription capture (learner_capture.cpp) that shares the port's conventions; only a real lib_lightgbm run can falsify the shared-convention errors (CR-01/CR-02). No real-binary oracle exists in the repo."
  - test: "Train the spine tree, serialize it, then predict every training row through the serialized model and tally rows per leaf."
    expected: "Per-leaf predict tallies equal the serialized leaf_count exactly."
    why_human: "This is the train/predict-consistency check that CR-01 fails; confirming the fix requires running the partition→serialize→predict round-trip end-to-end."
---

# Phase 5: Tree Learner + Split Finding Verification Report

**Phase Goal:** A histogram-based serial tree learner that grows the exact same tree as C++ — the keystone, highest-FP-risk subsystem, validated at per-split granularity.
**Verified:** 2026-06-06
**Status:** gaps_found
**Re-verification:** No — initial verification

## Goal Achievement

The keystone learner is substantially built — the leaf-wise growth loop, gain
scan, FixHistogram, DataPartition, LeafSplits, HistogramPool, ColSampler, and
force_col_wise all exist as substantive, wired code, and the full automated
suite is green (`cargo test --workspace`: all green; `learner_parity` 8/8;
`kernel_parity` 4/4). However, the phase's defining contract — that the tree the
model serializes is the tree it predicts into, routing exactly as C++ — is
**broken for the `most_freq_bin == 0` path**, which is the ONLY path with
bit-exact coverage. This is a Critical numerical-fidelity defect, not a nicety.

### Observable Truths (Success Criteria)

| # | Truth | Status | Evidence |
| - | ----- | ------ | -------- |
| 1 | Learner selects same split feature/bin/threshold/missing-direction as C++ for every split, validated against per-split candidate-gain snapshots | ✗ FAILED | Per-split gain snapshots replay bit-exact (`learner_parity_spine_per_bin_gains` ok), BUT the serialized tree is NOT predict-consistent (CR-01): leaf_count/internal_count/leaf outputs are computed for a partition `[4,8]` that the tree's own `get_leaf` routes as `[6,6]`. The "same tree as C++" is unfalsifiable here because the C++ capture shares the port's broken convention. |
| 2 | Histogram-subtraction trick reproduces C++ smaller-child selection + derived-child histogram (~1e-6); default-bin-skip scan considers same candidate set | ⚠️ PARTIAL | `learner_parity_subtract` + `kernel_parity_subtract` pass bit-exact in isolation. But in the actual growth path the subtraction trick is DEAD (WR-02): `find_best_split_for_leaf` always calls `construct_histograms` directly and `let _ = subtract_from;` (learner.rs:741) discards the sibling id; HistogramPool is passed as `_pool` and never read (WR-01, learner.rs:546). Numerically faithful (direct == subtracted for f64) but the claimed orchestration does not run. |
| 3 | Leaf-wise growth respects num_leaves/max_depth; split-gain formula matches C++ (kEpsilon, lambda_l1/l2, min_gain_to_split, min_sum_hessian, min_data, max_delta_step, path_smooth) | ✓ VERIFIED | Gain scan (split.rs) + leaf-wise arg_max loop (learner.rs) present and wired; spine/real_gh full-tree goldens replay bit-exact. (Caveat IN-04: max_depth mid-tree cap control flow differs structurally from C++ and is not exercised by any corpus — Info-level.) |
| 4 | Numerical threshold splits route missing/zero exactly as C++; data partition feeds subtraction trick correctly | ✗ FAILED | CR-01: data-partition `--th` (partition.rs:59-61) is off-by-one vs the stored threshold (split.rs:299, offset==0) for mfb=0. Independently reproduced: partition `[4,8]` vs predict `[6,6]`. The partition does NOT route as the serialized tree predicts. |
| 5 | Per-tree/per-node feature subsampling RNG parity; force_row_wise == force_col_wise produce matching trees | ✓ VERIFIED | `learner_parity_col_sampler_rng` (RNG draw-sequence parity) and `learner_parity_row_vs_col` (row==col tree equality) both replay bit-exact; ColSampler (col_sampler.rs) reproduces Random::Sample call sequence. (Self-consistent against the capture; same real-binary caveat as #1 applies but the RNG sequence parity is independently anchored on Phase-1 RNG goldens.) |

**Score:** 3/5 truths verified (SC#1 and SC#4 FAILED on CR-01; SC#2 PARTIAL on dead orchestration; SC#3 and SC#5 VERIFIED).

### Required Artifacts

| Artifact | Expected | Status | Details |
| -------- | -------- | ------ | ------- |
| `crates/lgbm-treelearner/src/learner.rs` | leaf-wise loop, BeforeFindBestSplit, FindBestSplits, SplitInner | ✓ VERIFIED (1292 lines) | Substantive + wired; but pool/subtract orchestration dead (WR-01/02) |
| `crates/lgbm-treelearner/src/histogram_pool.rs` | HistogramPool D-05 mirror | ⚠️ ORPHANED (263 lines) | Exists + unit-tested, but `_pool` never read in growth path |
| `crates/lgbm-treelearner/src/fix_histogram.rs` | FixHistogram most-freq-bin reconstruct | ✓ VERIFIED (139 lines) | Wired at learner.rs:744 |
| `crates/lgbm-treelearner/src/data_partition.rs` | DataPartition leaf_begin/leaf_count | ✓ VERIFIED (224 lines) | Wired; downstream of the broken partition kernel boundary |
| `crates/lgbm-treelearner/src/leaf_splits.rs` | LeafSplits ordered f64 fold | ✓ VERIFIED (201 lines) | Wired |
| `crates/lgbm-treelearner/src/col_sampler.rs` | ColSampler ResetByTree/GetByNode | ✓ VERIFIED (338 lines) | Wired; RNG parity green |
| `crates/lgbm-compute/src/kernels/split.rs` | find_best_split with explicit skip_default_bin/na_as_missing | ✓ VERIFIED (995 lines) | Threshold recording at :299 is the CR-01 root half |
| `crates/lgbm-compute/src/kernels/partition.rs` | data_partition_kernel | ⚠️ DEFECTIVE (221 lines) | `--th` at :59-61 is the CR-01 root half |
| fixtures: spine/col_wise/col_sampler/real_gh.txt | committed goldens | ✓ EXIST | All present + replay bit-exact, but capture shares port conventions (no real-binary oracle) |

### Key Link Verification

| From | To | Via | Status | Details |
| ---- | -- | --- | ------ | ------- |
| learner.rs | lgbm_compute::Backend | construct_histograms/find_best_split/data_partition | ✓ WIRED | Calls present (learner.rs:734, :755, :971) |
| learner.rs | lgbm_model::Tree::split | tree growth mutation | ✓ WIRED | `.split(` invoked |
| learner.rs | subtract_histograms | use_subtract path | ✗ NOT_WIRED | `let _ = subtract_from;` (line 741) — dead (WR-02) |
| learner.rs | HistogramPool buffers | pool slot reuse | ✗ NOT_WIRED | `_pool` never read (WR-01) |
| split.rs (stored threshold) | partition.rs (routing boundary) | consistent boundary for mfb=0 | ✗ BROKEN | off-by-one (CR-01) |
| col_sampler.rs | lgbm_core::Random | sample(n,k) call-sequence parity | ✓ WIRED | RNG parity green |

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
| -------- | ------- | ------ | ------ |
| Full workspace test suite | `cargo test --workspace` | all green | ✓ PASS |
| Learner parity goldens | `learner_parity` (8 tests) | 8/8 ok | ✓ PASS |
| Kernel parity goldens | `kernel_parity` (4 tests) | 4/4 ok | ✓ PASS |
| CR-01 boundary divergence (independent reproduction of partition vs predict routing on the spine mfb=0 layout) | Python re-implementation of partition.rs:58-73 vs tree.rs:170 | partition `[4,8]` vs predict `[6,6]` — DIVERGENCE | ✗ FAIL |

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
| ----------- | ----------- | ----------- | ------ | -------- |
| TRL-01 | 05-02, 05-03 | Histogram serial learner (Construct→FindBest→Split) | ⚠️ PARTIAL | Pipeline exists + wired; subtraction-trick orchestration dead (WR-02) |
| TRL-02 | 05-03 | Histogram subtraction trick, byte-identical FP path the model is defined against | ⚠️ PARTIAL | Bit-exact in isolation only; not run in growth path; AND the model the tree serializes is not predict-consistent (CR-01) |
| TRL-03 | 05-03 | Leaf-wise growth, num_leaves/max_depth caps | ✓ SATISFIED | Verified (IN-04 mid-tree depth-cap untested — Info) |
| TRL-04 | 05-02, 05-03 | Split-gain scan, exact formula + tie-break | ✓ SATISFIED | Gain formula + split_gt verified |
| TRL-05 | 05-01, 05-03 | Numerical threshold splits, C++-matching missing/zero routing | ✗ BLOCKED | CR-01: partition routing diverges from the stored threshold / predict routing for mfb=0; offset==1 path has zero bit-exact coverage (CR-02) |
| TRL-07 | 05-03 | Data partition (row→leaf) feeding subtraction | ✗ BLOCKED | CR-01: partition does not route as the serialized tree predicts |
| TRL-08 | 05-04 | Feature subsampling per-tree/per-node RNG parity | ✓ SATISFIED | col_sampler RNG parity green |
| TRL-09 | 05-04 | force_row_wise/force_col_wise both output-matching | ✓ SATISFIED | row==col tree equality green (offset==1 coverage caveat per CR-02) |

All 8 declared requirement IDs are accounted for across the 4 plan frontmatters
and present in REQUIREMENTS.md. TRL-06 (categorical) is correctly out of Phase-5
scope (deferred to Phase 7, REQUIREMENTS.md:198) — no orphaned requirements.
TRL-05 and TRL-07 are BLOCKED by CR-01; TRL-01/TRL-02 are degraded by the
dead subtraction orchestration.

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
| ---- | ---- | ------- | -------- | ------ |
| `partition.rs` | 59-61 | `--th` double-counts offset for mfb=0 | 🛑 Blocker | Train/predict routing divergence (CR-01) |
| `learner.rs` | 91-94 vs 768 | Three contradictory offset conventions | 🛑 Blocker | No faithful oracle for offset==1 (CR-02) |
| `learner.rs` | 546, 741 | `_pool` / `let _ = subtract_from;` dead orchestration | ⚠️ Warning | TRL-02 subtraction trick not exercised in growth path (WR-01/02) |
| `tree.rs` | 190-193 | unchecked `cat_boundaries[cat_idx]` indexing | ⚠️ Warning | Panic on malformed cat tree (WR-03) — Phase-7 path |
| `objective.rs` | 189-237 | softmax/convert index empty slices | ⚠️ Warning | Panic surface on `pub` helpers (WR-04) |
| `ensemble.rs` | 90-102 | `predict_raw` unchecked ntpi indexing | ⚠️ Warning | Panic on mis-sized model (WR-05) |
| `split.rs` | 194-220 | REVERSE `done`-flag monotonicity assumed | ⚠️ Warning | done==break unproven for mixed-sign hessian (WR-06) |

No unreferenced TBD/FIXME/XXX debt markers were introduced (the blockers are
logic defects, not deferred-work markers).

### Human Verification Required

1. **Real lib_lightgbm tree-text parity** — Build real LightGBM 4.6 on the spine
   + real_gh corpora and compare grown tree text byte-for-byte. Required because
   the committed goldens are a hand-transcription that shares the port's
   conventions; only a real binary can falsify CR-01/CR-02.
2. **Train/predict round-trip consistency** — Serialize the spine tree and verify
   per-leaf predict tallies equal the serialized leaf_count exactly. This is the
   check CR-01 fails.

### Gaps Summary

The keystone learner is largely complete and self-consistent, but the phase
goal — a bit-faithful, **predict-consistent** tree that routes exactly as C++ —
is NOT achieved on the `most_freq_bin == 0` path, which is the only path with
bit-exact coverage:

1. **CR-01 (BLOCKER, SC#4 + SC#1, TRL-05/TRL-07):** The data-partition `--th`
   adjustment is off-by-one relative to the stored threshold for mfb=0 features
   (`offset==0`, non-compacted histogram). The tree's leaf_count/internal_count/
   leaf outputs are computed for a partition the model does not predict into
   (`[4,8]` vs `[6,6]`, independently reproduced). This is a silent ≥1e-12
   fidelity violation of the project's non-negotiable contract. The parity
   goldens cannot catch it because the C++ capture hard-codes the same broken
   convention and only compares tree TEXT, never train-vs-predict routing.

2. **CR-02 (BLOCKER root cause, SC#1 offset==1 branch):** The `offset` invariant
   is documented one way, used the opposite way in every corpus, and inverted
   again in the real_gh parser. The only layout exercised bit-exact is the broken
   `most_freq_bin == 0` one; the `offset==1` / `most_freq_bin > 0` scan+partition
   path has zero bit-exact coverage against a real reference tree.

These share a root cause (the offset/most_freq_bin convention is not unified and
not anchored to a real lib_lightgbm oracle). The other automated checks
(`cargo test --workspace`, `learner_parity` 8/8, `kernel_parity` 4/4) genuinely
pass and the growth/gain/RNG machinery is real — but byte-level golden parity is
NOT evidence the predict-consistency criterion is met. Secondary warnings
(dead subtraction-trick + HistogramPool orchestration, WR-01/02) further weaken
the TRL-02 claim.

**Recommended next step:** `/gsd-plan-phase --gaps` to unify the offset
convention behind a single helper, fix the partition/threshold boundary, add the
train-vs-predict routing assertion, and add a `most_freq_bin > 0` corpus with a
real committed C++ reference tree.

---

_Verified: 2026-06-06_
_Verifier: Claude (gsd-verifier)_
