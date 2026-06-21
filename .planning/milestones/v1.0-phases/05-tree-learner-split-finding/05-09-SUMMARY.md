---
phase: 05-tree-learner-split-finding
plan: 09
subsystem: tree-learner
tags: [tree-learner, leaf-splits, fix-histogram, oracle-parity, lightgbm-4.6, kEpsilon, most-freq-bin, sparse-collapse, numerical-fidelity, fp-execution-trace]

# Dependency graph
requires:
  - phase: 05-07
    provides: "subtraction-trick + HistogramPool wired live; mfb>0 leaf-0 ULP isolated to the FixHistogram-active direct build"
  - phase: 05-08
    provides: "CR-03 closed for spine; child LeafSplits pass-through (smaller/larger slot mapping)"
provides:
  - "GROUND-TRUTH attribution of the mfb>0 node-2 leaf-0 2-ULP residual via a REAL lib_lightgbm 4.6 CPU-only single-thread FP execution trace: the corpus is SPARSE so the real binary collapses most_freq_bin_ = default_bin_ = ValueToBin(0) = 0 (bin.cpp:491-499) and runs the offset==1 path — the harness mislabeled it most_freq_bin=2/offset=0, which spuriously activated FixHistogram and polluted the REVERSE scan"
  - "C++-faithful child LeafSplits seeding: children are seeded DIRECTLY from the parent split's SplitInfo (best_split_info.left/right_sum_hessian carrying the parent's kEpsilon provenance), NOT a re-fold over the child's rows (serial_tree_learner.cpp:851-871)"
  - "learner_parity_mfb_pos_real_binary un-#[ignore]d and PASSING bit-exact (%.17g) in the default cargo test --workspace suite — the keystone serial learner is now bit-exact vs real lib_lightgbm 4.6 on BOTH committed corpora"
affects: [05, 06]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Ground-truth FP-trace attribution: build the real reference binary CPU-only single-thread, instrument the exact hot path with .to_bits() dumps gated on the node signature, and read the genuine operand provenance instead of hypothesizing fold orders"
    - "LeafSplits child seed from parent SplitInfo (init_from_split) — the kEpsilon provenance chain (best_sum_left_hessian - kEpsilon -> child sum_hessian -> +2*kEpsilon bump) must be preserved, not re-folded"

key-files:
  created:
    - .planning/phases/05-tree-learner-split-finding/05-09-SUMMARY.md
  modified:
    - crates/lgbm-treelearner/src/leaf_splits.rs
    - crates/lgbm-treelearner/src/learner.rs
    - crates/oracle-harness/tests/learner_parity.rs

key-decisions:
  - "The real lib_lightgbm 4.6 uses most_freq_bin=0, default_bin=0, missing_type=None, offset=1 for the mfb_pos corpus (FP trace [GSD-META]). Although raw value 2 is the modal RAW value, the feature is sparse (rate 0.1667 > kSparseThreshold) so BinMapper collapses most_freq_bin_ = default_bin_ = ValueToBin(0) = 0 (bin.cpp:491-499). The harness's most_freq_bin=2/offset=0 was WRONG and was the dominant cause of the 2-ULP leaf-0 residual: it spuriously activated FixHistogram on node-2's direct build, reconstructing a ~1e-15 bin-2 hessian that polluted the REVERSE scan."
  - "C++ seeds each child leaf's LeafSplits DIRECTLY from the parent split's SplitInfo (left/right_sum_hessian + left/right_output), NOT a re-fold over the child's rows. best_split_info.left_sum_hessian = best_sum_left_hessian - kEpsilon (feature_histogram.hpp:1042) carries the accumulated kEpsilon provenance. The Rust port's prior re-fold lost it (4.0 vs C++ 4.000000000000001). Added LeafSplits::init_from_split and switched split_inner to it (this is the authorized LeafSplits-provenance seam the prior 3-file scope excluded)."
  - "The fix is NOT in fix_histogram.rs / histogram.rs / the reverse-scan fold order (all three proven order-independent in 05-09 Task 1). The FP trace redirected the fix to (1) the corpus binning parameters and (2) the child LeafSplits seed. assert_real_tree_parity is byte-unchanged; no tolerance introduced."
  - "CONTRACT-DOC RECONCILIATION: the learner leaf output is now bit-exact f64 (1 ULP closed) vs the real golden, INSIDE both the CLAUDE.md/plan <=1e-12 contract and the STATE.md/ROADMAP ~1e-6 framing. The bit-exact result moots the gate-level discrepancy; the docs should record that the learner leaf output is enforced bit-exact f64 here. See Issues."

requirements-completed: [TRL-01, TRL-05]

# Metrics
duration: ~1 session (build real binary + FP trace + fix + regression)
completed: 2026-06-06
---

# Phase 5 Plan 09: mfb>0 Node-2 Leaf-0 Bit-Exact via Real-Binary FP Execution Trace

**The final 2.3e-16 (one f64 ULP) residual on `learner_parity_mfb_pos_real_binary` is CLOSED bit-exact. A real `lib_lightgbm` 4.6 CPU-only single-thread FP execution trace (instrumented `FindBestThresholdSequentially` / `LeafSplits::Init` / the bin meta init) gave the genuine operand provenance: (1) the corpus is SPARSE, so the real binary collapses `most_freq_bin = default_bin = ValueToBin(0) = 0` and runs the `offset==1` path — the harness had mislabeled it `most_freq_bin=2/offset=0`, which spuriously activated FixHistogram and polluted node-2's REVERSE scan by ~2 ULPs; and (2) C++ seeds each child leaf's `LeafSplits` DIRECTLY from the parent `SplitInfo` (carrying the parent's `kEpsilon` provenance), not a re-fold. Correcting both makes the gate pass bit-exact in the default suite. `assert_real_tree_parity` is byte-unchanged; no tolerance, no LightGBM/ artifacts committed.**

## Performance

- **Duration:** ~1 session (build the real binary + instrument + capture the FP trace + apply the fix + full regression)
- **Completed:** 2026-06-06
- **Tasks:** Task 1 (localization, already committed `c675d3b` by the predecessor) carried forward; Task 2 (the C++-faithful fix, this plan) COMPLETE; Task 3 (full-workspace regression) GREEN
- **Files modified:** 3 Rust-repo files (`leaf_splits.rs`, `learner.rs`, `learner_parity.rs`)

## The ground-truth FP execution trace (Option B, user-authorized)

Built `lib_lightgbm` 4.6 CPU-only, single-thread (`-DUSE_GPU=OFF -DUSE_CUDA=OFF -DUSE_OPENMP=OFF`) into `/tmp` (LightGBM/ tree kept untracked; instrumentation reverted afterward). Drove the EXACT mfb corpus through the real CLI (feature `[0,1,2,2,2,2,2,2,3,3,1,0]`, labels = `-grad`, the golden's config: `objective=regression boost_from_average=false deterministic=true force_row_wise=true num_threads=1 num_leaves=4 learning_rate=0.1 min_data_in_leaf=1 min_sum_hessian_in_leaf=0.001 lambda_l2=0`). The CLI model came out **bit-identical to `mfb_pos_real.txt`** (`leaf_value=0.59999999999999953 0 -0.44999999999999984 0.29999999999999988`), confirming the harness reproduces the golden's training.

Key instrumented operands (`.to_bits()`):

| Quantity | Real binary value | bits |
|---|---|---|
| **feature meta** | `most_freq_bin=0 default_bin=0 missing_type=0 offset=1` | — |
| root leaf-total sum_hessian seed (LeafSplits::Init) | `12.0` | `0x4028000000000000` |
| root scan `sum_hessian` (after `+2·kEpsilon` bump) | `12.0000000000000018` | `0x4028000000000001` |
| root stored `left_sum_hessian` (child-2 seed) | `4.000000000000001` | `0x4010000000000001` |
| node-2 child seed (LeafSplits::Init from parent) | `4.000000000000001` | `0x4010000000000001` |
| node-2 scan `sum_hessian` (after `+2·kEpsilon`) | `4.000000000000003` | `0x4010000000000003` |
| node-2 HIST bin0 hessian (offset==1, FixHistogram NO-OP) | `2.0` | `0x4000000000000000` |
| node-2 reverse-scan `sum_right_hessian` @ WIN (t=0) | `2.000000000000001` | `0x4000000000000002` |
| node-2 `best_sum_left_hessian` | `2.0000000000000018` | `0x4000000000000004` |
| node-2 leaf-0 `left_output` (raw) → `×0.1` shrunk | `5.999999999999995` → `0.59999999999999953` | golden |

## The single decisive origin (resolved by ground truth, not hypothesis)

The predecessor's checkpoint had narrowed the residual to "the leaf-total sum_hessian SEED" but could not reach the golden `0x...004` by any *faithful single-seed* transcription, because it assumed the corpus's `most_freq_bin=2/offset=0` (FixHistogram-active). The FP trace overturns that assumption:

1. **Binning (dominant cause).** The real `most_freq_bin == 0`, NOT 2. The feature is sparse (sparse rate `0.1667 > kSparseThreshold`), so `BinMapper` collapses `most_freq_bin_ = default_bin_ = ValueToBin(0) = 0` (`bin.cpp:491-499`). With `most_freq_bin == 0`, `offset == 1` and **FixHistogram is a NO-OP** (the same path as the spine). The harness's `most_freq_bin=2` spuriously activated FixHistogram on node-2's direct build, reconstructing a `~1e-15` bin-2 hessian (`sum_h_raw − bin0 − bin1 − bin3`); the REVERSE scan then accumulated `kEpsilon + ~1e-15 + 2.0`, 2 ULPs above the correct `kEpsilon + 2.0`, shifting `best_sum_left_hessian` from `0x...004` to `0x...002`.

2. **Child seed provenance.** C++ seeds the node-2 `LeafSplits` from the ROOT split's `best_split_info.left_sum_hessian = best_sum_left_hessian − kEpsilon = 0x4010000000000001` (`4.000000000000001`), NOT a fresh re-fold (which gives exactly `4.0`). The scan then bumps it by `+2·kEpsilon` to `0x4010000000000003`. With the clean bin0 hessian `2.0` and the correct seed, `best_sum_left_hessian = 0x4010000000000003 − 0x4000000000000002 = 0x4000000000000004` — the golden.

## The C++-faithful fix (Task 2)

- **`crates/lgbm-treelearner/src/leaf_splits.rs`** — added `LeafSplits::init_from_split(num_data, sum_g, sum_h, weight)` mirroring C++ `LeafSplits::Init(leaf, data_partition, sum_gradients, sum_hessians, weight)` (`leaf_splits.hpp:47-54`): seed a child's totals DIRECTLY from the parent split's `SplitInfo` and carry the split's already-computed `output` as `weight_` — no re-fold, no re-derivation.
- **`crates/lgbm-treelearner/src/learner.rs` (`split_inner`)** — replaced the child re-fold (`smaller/larger_leaf_splits.init(gradients, hessians, &rows, …)`) with `init_from_split(…, best.left_sum_hessian, best.left_output)` / right, selecting smaller/larger by the SplitInfo counts (`best.left_count < best.right_count`, `serial_tree_learner.cpp:851`), and using the partition leaf-count for `num_data_in_leaf` (C++ `GetIndexOnLeaf`). Removed the now-unused `gradients`/`hessians` params from `split_inner`.
- **`crates/oracle-harness/tests/learner_parity.rs`** — corrected the `mfb_pos_real` corpus to the ground-truth `most_freq_bin=0` / `offset=1` (the real sparse-collapse layout), with a doc comment recording the FP-trace `[GSD-META]` evidence; un-`#[ignore]`d `learner_parity_mfb_pos_real_binary`; removed the 05-09 Task-1 scratch instrumentation test. `assert_real_tree_parity` is byte-unchanged.

`fix_histogram.rs` and `histogram.rs` (the plan's other two named files) were NOT modified — the trace attributed the fix to the binning parameters + the LeafSplits seam, exactly the LeafSplits-provenance seam the prior 3-file scope excluded and the user authorized.

## Deviations from Plan

### [Rule 4 → resolved by user-authorized FP trace] Fix lives outside the three named hot-path files

The plan hypothesized a fold-order alignment in one of `fix_histogram.rs` / `histogram.rs` / the reverse-scan accumulation. The predecessor proved all three order-independent and raised a `checkpoint:decision`; the user authorized Option B (build the real binary, capture an attributable FP trace). The trace attributed the residual to (a) the corpus's binning parameters (`most_freq_bin` sparse-collapse) and (b) the child `LeafSplits` seed provenance — NOT a fold order. Both fixes are faithful transcriptions of the named C++ references (`bin.cpp:491-499`, `serial_tree_learner.cpp:851-871`, `feature_histogram.hpp:172/1042`), verified bit-exact against the real binary.

**Total deviations:** 1 (fix location redirected by ground-truth trace; no tolerance, no gate weakening, no LeafSplits-SplitInfo *scan* re-seeding — the leaf-3-regressing path 05-07 ruled out is NOT what was applied; the change is the child-LeafSplits *seed*, which leaves leaf 3 bit-exact).

## Issues Encountered

- **Contract-doc reconciliation (resolved at the gate level).** CLAUDE.md + the 05-09 plan mandate `≤1e-12` bit-exact; STATE.md/PROJECT.md/ROADMAP document an `f32 / ~1e-6` framing (a Phase-1 revision). The learner leaf output is now **bit-exact f64** vs the real golden, INSIDE both contracts, so the gate-level discrepancy is moot for this plan. Recommendation for the user: record in PROJECT/ROADMAP that the **learner leaf output is enforced bit-exact f64** here (consistent with the `%.17g` `assert_real_tree_parity` gate), so the contract statements are consistent. Not silently changed — flagged for an explicit project decision.
- **Harness corpus mislabel (now corrected).** The Python capture's `assert_identity_binning` checks the modal RAW value (2), which is NOT LightGBM's internal `GetMostFreqBin()` (0, after the sparse collapse). The harness inherited the raw-modal label. Corrected to ground truth; the Python capture's assertion is about raw-value identity binning and remains valid (it does not claim `GetMostFreqBin()==2`).

## Threat Flags

None. The change touches numeric fold provenance only (synthetic 12-row single-feature fixture); no new network/auth/file-access/schema surface. The threat register's `mitigate` dispositions (T-05-09-01 tampering on the fold, T-05-09-02 repudiation on the gate) are honored: the gate passes only because the value is bit-exact `%.17g` vs the real golden, `assert_real_tree_parity` is byte-unchanged, and the no-weakening audit (below) confirms no tolerance was added.

## Task Commits

1. **Task 1 (localize the 2-ULP origin)** — `c675d3b` (predecessor; carried forward).
2. **Task 2 (C++-faithful fix: ground-truth binning + child LeafSplits seed; un-ignore gate; remove scratch)** — `2ced5a2`.
3. **Plan metadata + this SUMMARY** — committed with STATE.md + ROADMAP.md + REQUIREMENTS.md.

## Verification

- `cargo test -p oracle-harness --test learner_parity` — **12 passed / 0 ignored / 0 failed** (the mfb gate moved from ignored to passed).
- `learner_parity_mfb_pos_real_binary` — PASSES bit-exact (node-2 leaf-0 `0.59999999999999953`).
- `learner_parity_spine_real_binary` — PASSES bit-exact (no spine regression).
- `learner_parity_growth_path_subtract` — PASSES (subtraction-trick wiring unregressed).
- `cargo test -p oracle-harness --test kernel_parity` — **4/4** bit-exact on cpu (histogram, split, partition, subtract).
- `cargo test --workspace` — GREEN (41 test groups ok, 0 failed, 0 ignored).
- **No-weakening audit:** `assert_real_tree_parity` byte-unchanged; no `abs_diff`/`tol`/`epsilon`/`approx`/`<= 1e-` introduced to gate the mfb leaf_value; the only `learner_parity.rs` diffs are the corpus-parameter correction + `#[ignore]` removal + scratch-test removal.
- `git status --porcelain LightGBM/` — no staged entries (LightGBM/, its submodules, and the `/tmp` build never git-added; C++ instrumentation reverted).

## Self-Check: PASSED

- `05-09-SUMMARY.md` exists on disk.
- Fix commit `2ced5a2` present in history.
- `learner_parity_mfb_pos_real_binary` runs un-ignored and passes bit-exact.

---
*Phase: 05-tree-learner-split-finding*
*Completed: 2026-06-06 (bit-exact via real-binary FP execution trace)*
