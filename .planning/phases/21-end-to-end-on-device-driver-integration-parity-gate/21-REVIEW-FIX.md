---
phase: 21-end-to-end-on-device-driver-integration-parity-gate
fixed_at: 2026-07-02T00:00:00Z
review_path: .planning/phases/21-end-to-end-on-device-driver-integration-parity-gate/21-REVIEW.md
iteration: 1
findings_in_scope: 11
fixed: 8
skipped: 3
status: partial
---

# Phase 21: Code Review Fix Report

**Fixed at:** 2026-07-02
**Source review:** .planning/phases/21-end-to-end-on-device-driver-integration-parity-gate/21-REVIEW.md
**Iteration:** 1

**Summary:**
- Findings in scope: 11 (fix_scope = all)
- Fixed: 8 (WR-01, WR-02, WR-04, WR-05, IN-01, IN-02, IN-04, IN-05)
- Skipped: 3 (CR-01 code-change, WR-03, IN-03)

All fixes were verified: re-read + `cargo check`/`cargo test` where applicable. The
full `learner_parity` suite passes (32 passed, 4 intentionally ignored) and
`cargo check --workspace --tests` is clean.

## Fixed Issues

### WR-05 / IN-01 / IN-05: grow_driver.rs cleanups
**Files modified:** `crates/lgbm-compute/src/kernels/grow_driver.rs`
**Commit:** bff71f9
**Applied fix:**
- WR-05: both `_with_cfg` guard error strings now name `grow_tree_on_device_driver_with_cfg`
  (they previously hardcoded the delegator name `grow_tree_on_device_driver`), restoring
  diagnosability at the `thiserror` domain boundary now that `_with_cfg` is called directly.
- IN-01: the `real_threshold` lookup no longer falls back to `unwrap_or(best.threshold as f64)`
  (which would silently record a raw bin index as a real threshold). It now returns a typed
  `ComputeError::Runtime` on an out-of-range threshold bin index.
- IN-05: dropped the redundant `num_leaves.max(1)` at `DeviceCudaTree::new` — `num_leaves >= 1`
  is already guaranteed by the guard above; now uses `num_leaves as usize` with a comment.

### WR-01 / IN-02: on-device structure gates now env-independent + strict f64 comparator
**Files modified:** `crates/oracle-harness/tests/learner_parity.rs`
**Commit:** 80f362a
**Applied fix:**
- WR-01: `learner_parity_on_device_structure_gate`, `_deep_multileaf_gate`, and `_nosplit_gate`
  now call `grow_tree_on_device_driver_with_cfg` DIRECTLY (env-independent, mirroring the
  mindata gate), so the broad multi-feature / deep (>2 live-leaf) / no-split growth paths get
  STRUCTURE parity coverage in the DEFAULT `cargo test` merge-gate run — not only under
  `LGBM_CUDA_ON_DEVICE=1`. The cfg-less trait seam is still asserted (grow under env=1, defer
  `Ok(None)` when unset).
- IN-02: those cubecl-cpu (f64-vs-f64) driver gates now use the STRICT
  `assert_gpu_tree_matches_cpu_anchor` (full `decision_type` equality) instead of the f32-vs-f64
  tie-aware comparator; `default_left` is bit-exact on the CPU path (verified: leaf diff 0.000e0
  across all three corpora), so the looser tie tolerance is no longer applied where it could mask
  a real direction bug. The tie-aware comparator remains reserved for the `mod hip` cells.

### WR-04: leaf-map A/B harness relabelled as a decision-record
**Files modified:** `crates/lgbm-compute/src/kernels/grow_driver.rs`
**Commit:** f697dc9
**Applied fix:** Added a doc note to the `build_leaf_map_on` / `LeafMapBufferStrategy` module
comment clarifying it is a DECISION-RECORD A/B harness, NOT live driver plumbing (the shipped
driver carries no running leaf-map buffer and partitions via `partition_leaf_stable`), that the
items stay `pub` only because the A/B oracle lives in the separate `oracle-harness` crate, and
that the "LOCK" language means "recorded the A/B conclusion", not "the driver applies it". The
`pub` surface was retained because removing it would break the cross-crate oracle test.

### IN-04: HistArena role handles documented as undefined after multi-leaf swap
**Files modified:** `crates/lgbm-compute/src/kernels/histogram_arena.rs`
**Commit:** c48c705
**Applied fix:** Documented on `parent_handle`/`smaller_handle`/`larger_handle` and at the
`swap` role-field assignment that the single-triple role fields reflect only the most recent
`rotate` (2-leaf API) and are UNDEFINED after a multi-leaf `swap` loop (only `leaf_handle(leaf)`
is meaningful there), warning any future caller mixing the two APIs.

### WR-02: vacuous self-transcription stubs marked #[ignore]
**Files modified:** `crates/oracle-harness/tests/learner_parity.rs`
**Commit:** 83d7025
**Applied fix:** `learner_parity_spine_full_tree`, `_spine_per_bin_gains`,
`_transcription_crosscheck`, and `_real_gh_full_tree` (pure `eprintln!` stubs that asserted
nothing) are now `#[ignore = "..."]` with reasons, so they report as *ignored* rather than
green-passing while covering nothing. `learner_parity_row_vs_col` was NOT ignored: it retains a
live `row==col` tree-equality assertion (only its col_wise-golden tail is a stale note). The
"make real-binary gates fail-not-skip in CI" half of the suggestion was not applied (it would
break clean checkouts lacking the committed goldens — a CI-policy change out of scope for a
source fix).

### CR-01 (constructive resolution): min_data_in_leaf STRUCTURE gate added
**Files modified:** `crates/oracle-harness/tests/learner_parity.rs`
**Commit:** 05443e1
**Applied fix:** Added `learner_parity_on_device_mindata_structure_gate`, an env-independent
STRUCTURE gate that BINDS `min_data_in_leaf` (md ∈ {2,3,4}: md=2 grows 4 leaves, md=3/4 stop at
2 leaves — the constraint observably binds) and asserts the driver is bit-exact to the cpu f64
anchor. This is the gate the earlier suite deliberately avoided; it proves the driver already
honors `min_data_in_leaf`. See the CR-01 skip entry below for why the driver GATE itself was NOT
rewritten.
**Note:** requires human verification — this commit *adds test coverage* rather than changing
driver logic; the accompanying decision to NOT change the driver gate (below) is the substantive
call and is evidence-backed.

## Skipped Issues

### CR-01 (driver gate rewrite): `both_too_small` combined-AND gate
**File:** `crates/lgbm-compute/src/kernels/grow_driver.rs:677-684`
**Reason:** SKIPPED — the review's premise is a misreading of the C++ reference; applying the
suggested rewrite would *reduce* faithfulness. Evidence:
- The driver's `both_too_small = left_count < min_data*2 && right_count < min_data*2` combined-AND
  gate is an EXACT match to C++ `SerialTreeLearner::BeforeFindBestSplit`
  (`LightGBM/src/treelearner/serial_tree_learner.cpp:356-357`), which uses the SAME `&&` over the
  two children — NOT the "each leaf independently" the review claims.
- The Rust cpu f64 anchor (the bit-exact merge-gate reference) transcribes the identical gate at
  `crates/lgbm-treelearner/src/learner.rs:1490-1502` (`min2 = min_data_in_leaf*2`, `num_right <
  min2 && num_left < min2`), and — like C++ — does NOT check `min_sum_hessian_in_leaf` in this
  pre-gate; it is enforced DOWNSTREAM per-candidate-split (`split.rs` `find_best_split_f64_on`,
  lines 259-263/323-326), exactly as the driver does.
- The combined-AND gate + downstream per-split enforcement is behaviorally equivalent to an
  independent per-child gate (a child with `< min_data*2` rows admits no downstream split anyway),
  so the review's rewrite would change nothing about the tree — while diverging the driver's gate
  STRUCTURE from the C++/anchor transcription and adding a `min_sum_hessian` term neither has.
- Empirically confirmed: a temporary driver-vs-anchor test binding `min_data_in_leaf` ∈ {2,3,4}
  showed the driver is STRUCTURE bit-exact to the cpu f64 anchor in every case (leaf diff
  0.000e0), including the leaf-count-reducing md=3/4 cases. That coverage was retained as the
  committed `learner_parity_on_device_mindata_structure_gate` (commit 05443e1).
Per the "fix the gate to match C++ per-leaf semantics rather than weakening the test / skip risky
or ambiguous fixes" guidance: the gate ALREADY matches C++, so it was left unchanged and the
proving gate was added instead.

### WR-03: `HistArena` dead code relative to the live driver
**File:** `crates/lgbm-compute/src/kernels/histogram_arena.rs:1-423`
**Reason:** SKIPPED — neither of the review's two options is a safe mechanical fix. (a) "Wire it
into the driver" is a design change (replacing the per-leaf `Vec<f64>` histograms) out of scope
for a review-fix and risks re-introducing the aliasing class this milestone closed. (b) "Drop its
public surface" is not safe: `HistArena` is referenced by `subtract.rs`
(`subtract_histograms_via_arena`), `histogram.rs`, and the cross-crate integration tests in
`crates/lgbm-compute/tests/rocm_cuda_mirror.rs`, which require the `pub` surface. The module's own
doc-comment already labels it unit-test-locked; IN-04 (committed) further documents the swap-role
trap, mitigating the "future caller wires it" risk without a risky refactor.

### IN-03: Case C `env_on` lane checks only leaf-count inequality
**File:** `crates/oracle-harness/tests/learner_parity.rs` (mindata gate `env_on` branch)
**Reason:** SKIPPED — the review's literal fix ("in the env=1 lane, also build the constrained
anchor via `cpu_anchor_tree` and assert structure") is NOT viable. Under `LGBM_CUDA_ON_DEVICE=1`
the learner's `on_device_eligible` is true and `SerialTreeLearner::train` FORKS to the cfg-less
on-device seam (`crates/lgbm-treelearner/src/learner.rs:739-773`, and the test's own doc at
~2604-2610), so `cpu_anchor_tree` under env=1 silently drops the constrained cfg and grows the
UNCONSTRAINED tree — it cannot produce a constrained reference in that lane. Building a viable
non-forking constrained reference requires new plumbing beyond a safe fix. Mitigation already in
place: the constrained `driver_tree` is grown via the env-INDEPENDENT `_with_cfg` call and is
already asserted STRUCTURE-bit-exact against the constrained anchor in the DEFAULT (env-unset)
merge-gate lane, which always runs.

---

_Fixed: 2026-07-02_
_Fixer: Claude (gsd-code-fixer)_
_Iteration: 1_
