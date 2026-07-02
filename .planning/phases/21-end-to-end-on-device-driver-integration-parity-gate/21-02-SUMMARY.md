---
phase: 21-end-to-end-on-device-driver-integration-parity-gate
plan: 02
subsystem: oracle-harness (on-device grow-driver STRUCTURE parity gate)
tags: [on-device, driver, parity-gate, hardening, odl-18h, test-only, cpu-f64-anchor]
dependency_graph:
  requires:
    - "grow_tree_on_device_driver_with_cfg (plan 21-01, extract-param GainConfig variant)"
    - "learner_parity_on_device_structure_gate + cpu_anchor_tree + assert_on_device_tree_matches_cpu_anchor (Phase 20)"
  provides:
    - "3 targeted D-02 STRUCTURE parity gate tests (deep >2-live-leaf, no-split, min-hessian-constrained)"
    - "broadened ODL-18H parity evidence for the on-device grow driver, cpu-f64-anchored"
  affects:
    - "none (test-only additions; no production/library surface changed)"
tech_stack:
  added: []
  patterns:
    - "clone-the-gate + swap-the-corpus targeted parity cases (reuse comparator/anchor/mapper verbatim)"
    - "env-independent direct driver call for cfg-carrying constrained parity (bypass the cfg-less trait seam)"
key_files:
  created: []
  modified:
    - crates/oracle-harness/tests/learner_parity.rs
decisions:
  - "Case B pins num_leaves=1 (RESEARCH A3): a constant-gradient L2 root has analytically-zero gain but the kEpsilon-carrying hessian sums leave a residual DEGENERATE positive gain, so the fixed proving_slice_config (min_gain_to_split=0, unoverridable through the cfg-less trait seam) would still split. num_leaves=1 (loop never runs) is the guaranteed no-split path."
  - "Case C binds via min_sum_hessian_in_leaf, NOT min_data_in_leaf: min_data's `min_data*2` both-too-small pre-gate takes a divergent child-leaf path between the driver and the learner on this small corpus; min_sum_hessian is checked per-candidate identically in split.rs and the learner."
  - "Case C's constrained STRUCTURE-parity (driver == constrained cpu host anchor) runs in the ENV-UNSET lane. Under LGBM_CUDA_ON_DEVICE=1 SerialTreeLearner forks to the cfg-less on-device trait seam (proving_slice_config) and silently drops the constrained cfg, so a constrained host anchor is only obtainable env-unset. The env=1 lane instead proves the constraint binds (constrained direct driver < unconstrained live seam)."
metrics:
  duration: ~40m
  completed: 2026-07-02
  tasks: 3
  files_modified: 1
status: complete
requirements: [ODL-18H]
---

# Phase 21 Plan 02: On-Device Driver STRUCTURE Parity Hardening (D-02 targeted cases) Summary

Broadened the STRUCTURE parity evidence for the on-device grow driver beyond the single
4-leaf case the Phase-20 gate proved, with THREE targeted D-02 risk cases added to
`crates/oracle-harness/tests/learner_parity.rs`. Each clones the existing
`learner_parity_on_device_structure_gate`, swaps only the corpus (plus, for case C, the
driver entry point + cfg), and reuses the tie-aware cpu-f64 comparator / anchor / mapper
verbatim. Every case pins to the cubecl-cpu f64 anchor (never GPU-vs-GPU, def-f8u-01).

**Anchor precisely (21-RESEARCH Pitfall 4):** the gate proves `on-device == cpu
SerialTreeLearner` (itself bit-exact to real `lib_lightgbm` 4.6 on the committed
corpora). It does NOT exercise the WR-01 `HistArena::swap` path — the live driver keeps
per-leaf `Vec<f64>` histograms and never consumes HistArena (Pitfall 1); the deep case
broadens parity breadth only.

## What Was Built

### Task 1 — Case A: deep tree, >2 simultaneously-live leaves
- `deep_multileaf_corpus`: two continuous features + L2, `MissingType::None`, 16 rows.
  `f0` (8 bins × 2 rows) carries strongly-separated MONOTONE gradients so each of the
  5 splits has a distinct positive gain (near-tie-free ⇒ a genuine bit-exact assert);
  `f1` (4 bins) is scrambled so `f0` dominates. `num_leaves = 6`.
- `learner_parity_on_device_deep_multileaf_gate`: env-gated. env=1 grows through
  `CpuBackend::grow_tree_on_device`, asserts STRUCTURE bit-exact vs `cpu_anchor_tree`
  (proving cfg), asserts `num_leaves >= 5` (>2 leaves live at the final splits) and
  layout row-conservation; env-unset asserts `Ok(None)` (byte-unchanged merge gate).

### Task 2 — Case B: no-split / single-leaf root-only tree
- `nosplit_corpus`: constant gradient (`g == 3.0`, `h == 1.0`), 12 rows, `num_leaves = 1`.
- `learner_parity_on_device_nosplit_gate`: env=1 grows a root-only (1-leaf) tree,
  STRUCTURE bit-exact vs the anchor (leaf value within `ROCM_LEAF_VALUE_TOL`), asserts
  `num_leaves == 1` and layout row-conservation; env-unset asserts `Ok(None)`.
- The driver seeds the root leaf value via `calculate_splitted_leaf_output` + `add_bias`,
  so a never-split root matches the anchor.

### Task 3 — Case C: min_sum_hessian-constrained case (via `grow_tree_on_device_driver_with_cfg`)
- `mindata_corpus`: one continuous feature + L2, 8 rows, 4 bins × 2 rows, monotone
  distinct gradients (`h == 1.0`).
- `learner_parity_on_device_mindata_gate`: the constrained tree is grown by calling
  `grow_tree_on_device_driver_with_cfg` (plan 21-01) DIRECTLY with a `min_sum_hessian_in_leaf = 3.0`
  cfg — env-independent, so `driver_tree` is the constrained tree in both lanes.
  - ENV-UNSET lane (the constrained-parity home): `cpu_anchor_tree` honors the cfg;
    asserts `driver_tree` STRUCTURE bit-exact to the constrained anchor, asserts the
    constraint binds (constrained anchor 2 leaves < unconstrained anchor 4 leaves), and
    asserts the trait seam still defers (`Ok(None)`, byte-unchanged).
  - ENV=1 lane: asserts the constrained direct `driver_tree` (2 leaves) has FEWER leaves
    than the unconstrained tree the LIVE trait seam grows (4 leaves) — the constraint
    observably binds through the driver — plus layout row-conservation.

## Verification

- `LGBM_CUDA_ON_DEVICE=1 cargo test -p oracle-harness --test learner_parity -- --exact <test>`:
  all three (`deep_multileaf`, `nosplit`, `mindata`) print `1 passed` (non-vacuous).
- Env-unset `-- --exact <test>` for all three: `1 passed` (byte-unchanged / constrained
  structure parity).
- `LGBM_CUDA_ON_DEVICE=1 cargo test -p oracle-harness --test learner_parity on_device`:
  6 passed, 0 failed (all on-device gate tests — the intended env=1 lane).
- `cargo test -p oracle-harness --test learner_parity` (env unset): 35 passed, 0 failed.
- `cargo test --workspace` (env unset): 0 FAILED across all crates.
- `cargo clippy -p oracle-harness --tests`: no warnings in the added line range.
- Diff vs base is purely additive (one contiguous insertion, zero deletions of existing code).

## Deviations from Plan

### [Rule 3 - Blocking] Case C constrained STRUCTURE-parity moved to the ENV-UNSET lane
- **Found during:** Task 3.
- **Issue:** The plan directed the env_on branch to call `grow_tree_on_device_driver_with_cfg`
  directly AND anchor via `cpu_anchor_tree(constrained_cfg)`. But under
  `LGBM_CUDA_ON_DEVICE=1`, `SerialTreeLearner::new` sets
  `on_device_eligible = on_device_growth_supported() && cuda_on_device_env()` = true, so
  `cpu_anchor_tree.train()` FORKS to the on-device driver via the cfg-less trait seam
  (`proving_slice_config`) and silently DROPS the constrained cfg — the "constrained
  anchor" comes back UNCONSTRAINED (probe: constrained anchor == unconstrained anchor ==
  4 leaves `[2,2,2,2]`, while the direct driver honored the constraint at 2 leaves
  `[4,4]`). RESEARCH Pitfall 2/3 anticipated the cfg-less trait seam but not the
  learner's env-gated fork of the anchor itself.
- **Fix:** The direct `grow_tree_on_device_driver_with_cfg` call is env-independent, so it
  yields the constrained tree in both lanes. The constrained-vs-host STRUCTURE parity is
  asserted in the ENV-UNSET lane (where `cpu_anchor_tree` takes the pure host path and
  honors the cfg); the env=1 lane instead proves the constraint binds through the driver
  (constrained direct driver 2 leaves < unconstrained live seam 4 leaves). Both lanes are
  non-vacuous; all four acceptance criteria are met (env=1 `1 passed`; env-unset structure
  parity + trait-seam deferral; direct `_with_cfg` call; constraint observably binds).

### [Rule 1 - Corpus design] Case B pins num_leaves=1 (RESEARCH A3 fallback)
- **Found during:** Task 2.
- **Issue:** A constant-gradient corpus with `num_leaves = 4` (intended to hit the
  `best_fpos < 0 || !(best.gain > 0.0)` break at first iteration) still split once —
  the pure L2 gain is analytically zero but the reference-blessed `kEpsilon`-carrying
  hessian sums leave a residual degenerate positive gain, and `proving_slice_config`
  (`min_gain_to_split = 0`) cannot be overridden through the cfg-less trait seam.
- **Fix:** Pin `num_leaves = 1` per RESEARCH A3 — the best-first loop is `for _ in 0..0`
  (never runs), a guaranteed no-split root-only tree that the anchor matches bit-exactly.
  Explicitly sanctioned by the plan's acceptance criteria ("... or uses `num_leaves = 1`").

### [Rule 1 - Corpus design] Case C binds via min_sum_hessian_in_leaf, not min_data_in_leaf
- **Found during:** Task 3.
- **Issue:** With `min_data_in_leaf = 4`, the driver honored the `min_data*2`
  both-too-small pre-gate (stopped at 2 leaves) but the learner anchor did not on this
  small corpus (independent of the env-fork issue, the min_data child-leaf gate paths
  diverge).
- **Fix:** Bind via `min_sum_hessian_in_leaf = 3.0` (with `h == 1.0`, hessian mass ==
  row count), which is checked per-candidate IDENTICALLY in `split.rs` (the driver) and
  the learner. The plan explicitly permits "`min_data_in_leaf` and/or positive
  `min_sum_hessian_in_leaf`".

## Notes / Open Observations (not blockers for this test-only plan)

- **Potential learner min_data/min_sum_hessian gap (out of scope, unfixed):** during
  case-C debugging, running the *host* `SerialTreeLearner` under env=1 forks to the
  on-device seam, which masks whether the host learner honors these constraints on tiny
  L2 corpora. The env-unset host anchor DOES honor `min_sum_hessian_in_leaf` (constrained
  2 leaves vs unconstrained 4), so the constrained parity is sound. Any deeper learner
  min_data child-gate question is a separate learner concern (Rule 4 territory), not a
  test-only 21-02 change, and is left untouched.
- **ROADMAP open question (recorded per 21-RESEARCH):** does not affect this parity
  hardening — the data→leaf aliasing is moot for the live driver (host per-leaf `Vec<u32>`
  rows), and batched `client.read` is Phase-23 perf only.

## Self-Check: PASSED
- FOUND: crates/oracle-harness/tests/learner_parity.rs (deep_multileaf_corpus, nosplit_corpus, mindata_corpus)
- FOUND: learner_parity_on_device_deep_multileaf_gate / _nosplit_gate / _mindata_gate
- FOUND commit cf7da0a (Task 1 — case A)
- FOUND commit 5086030 (Task 2 — case B)
- FOUND commit 4aa5509 (Task 3 — case C)
