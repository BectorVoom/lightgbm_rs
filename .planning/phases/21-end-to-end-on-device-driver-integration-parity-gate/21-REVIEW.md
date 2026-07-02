---
phase: 21-end-to-end-on-device-driver-integration-parity-gate
reviewed: 2026-07-02T00:00:00Z
depth: standard
files_reviewed: 3
files_reviewed_list:
  - crates/lgbm-compute/src/kernels/grow_driver.rs
  - crates/lgbm-compute/src/kernels/histogram_arena.rs
  - crates/oracle-harness/tests/learner_parity.rs
findings:
  critical: 0
  warning: 1
  info: 2
  total: 3
status: issues-found
---

# Phase 21: Code Review Report

**Reviewed:** 2026-07-02
**Depth:** standard
**Files Reviewed:** 3
**Status:** issues-found

## Summary

Phase 21 is an additive hardening phase with three distinct diffs, all reviewed against
base `51a4bec~1`:

1. **`grow_driver.rs`** — an extract-parameter refactor: a new
   `grow_tree_on_device_driver_with_cfg<R>(..., cfg: GainConfig)` holds the former body,
   and `grow_tree_on_device_driver<R>` becomes a thin delegator passing
   `proving_slice_config()`. I diffed the moved body: it is byte-identical except the
   removed `let cfg = proving_slice_config();` line, which is now the trailing parameter.
   The delegator preserves the original behavior exactly, so the bit-exact / byte-unchanged
   merge-gate contract holds. No correctness defect.

2. **`histogram_arena.rs`** — a doc-only 7-line block above `swap()` recording the WR-01
   closure. No executable change; the free-slot scan and its three `swap_*` repro tests are
   unchanged from the Phase-18 fix. Nothing to flag.

3. **`learner_parity.rs`** — three new env-gated STRUCTURE-parity tests (deep >2-live-leaf,
   no-split root-only, min_sum_hessian-constrained), each cloning
   `learner_parity_on_device_structure_gate` and reusing the tie-aware cpu-f64 comparator /
   anchor verbatim. I traced the corpus arithmetic for the deep and constrained cases and
   confirmed the winning split at every node is unique (symmetric ties exist only among
   non-winning candidates, so the "genuine bit-exact assert" claim holds) and that the
   constrained case genuinely binds (root splits 4/4 → hessian-mass-4 children ≥ 3.0 admit;
   further splits produce hessian-mass-2 children < 3.0 and are forbidden ⇒ 2 leaves vs 4
   unconstrained). The tests are non-vacuous.

The findings below are quality/coverage observations. No blocker: the diff neither changes
production behavior when `LGBM_CUDA_ON_DEVICE` is unset nor risks bit-exactness.

## Warnings

### WR-01: Error strings inside `grow_tree_on_device_driver_with_cfg` name the wrong function

**File:** `crates/lgbm-compute/src/kernels/grow_driver.rs:467,472`
**Issue:** The extract-parameter refactor moved the guard bodies into
`grow_tree_on_device_driver_with_cfg`, but the two `ComputeError::Runtime` detail strings
still hardcode the delegator's name `grow_tree_on_device_driver:`. Phase 21-02's case C
(`learner_parity_on_device_mindata_gate`) is the first caller to invoke `_with_cfg`
**directly** — any test or future caller that trips these guards (e.g. `num_leaves < 1`,
empty `features`) receives an error naming a function it did not call, degrading
diagnosability at a library boundary (`thiserror` domain error, per CLAUDE.md). Low impact
(the guards only fire on misuse), but it is an introduced inconsistency from the rename.
**Fix:** Reference the actual function (or a neutral operation label) in both strings, e.g.
```rust
detail: format!("grow_tree_on_device_driver_with_cfg: num_leaves must be >= 1, got {num_leaves}"),
// ...
detail: "grow_tree_on_device_driver_with_cfg: at least one feature is required".to_string(),
```

## Info

### IN-01: Cases A and B exercise the driver only under `LGBM_CUDA_ON_DEVICE=1`, not in the default merge gate

**File:** `crates/oracle-harness/tests/learner_parity.rs:2454-2489, 2531-2568`
**Issue:** In `learner_parity_on_device_deep_multileaf_gate` and
`learner_parity_on_device_nosplit_gate`, the env-unset branch asserts only that the trait
seam defers (`grown.is_none()`); the actual driver-vs-anchor STRUCTURE parity (the point of
broadening the evidence) runs solely in the `env_on` branch. The default CI/merge-gate run
(env unset) therefore validates the seam-deferral, not the newly broadened parity breadth —
that only runs when someone sets `LGBM_CUDA_ON_DEVICE=1`. This is the documented
byte-unchanged-merge-gate design (21-02 SUMMARY), so it is intentional, but the broadened
parity evidence is not continuously exercised by default.
**Fix:** No code change required if the env=1 lane is run in CI. If not, consider a CI step
that runs `LGBM_CUDA_ON_DEVICE=1 cargo test -p oracle-harness --test learner_parity on_device`
so the new parity breadth is actually gated, not just gated-on-demand.

### IN-02: Case C's `env_on` lane checks only leaf-count inequality, not structure parity

**File:** `crates/oracle-harness/tests/learner_parity.rs:2660-2678`
**Issue:** In `learner_parity_on_device_mindata_gate`, the `env_on` branch asserts only
`driver_tree.num_leaves < seam_tree.num_leaves` (constraint binds). The constrained
STRUCTURE-bit-exact comparison against `cpu_anchor_tree` runs only in the env-unset branch
(a deliberate move, because under env=1 the anchor learner forks to the cfg-less on-device
seam and drops the constraint — correctly documented in the 21-02 deviation). Consequence: a
hypothetical driver defect that still produced 2 leaves but a structurally wrong tree would
pass the env=1 lane; it is caught only by the env-unset lane. Across the two runs coverage is
complete, but neither single lane fully validates the constrained driver.
**Fix:** None required — the split coverage is sound and documented. Noted so a future reader
does not assume the env=1 lane alone proves constrained structural parity.

---

_Reviewed: 2026-07-02_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
