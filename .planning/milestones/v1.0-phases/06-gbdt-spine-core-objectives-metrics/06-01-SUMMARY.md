---
phase: 06-gbdt-spine-core-objectives-metrics
plan: 01
subsystem: infra
tags: [cargo-workspace, thiserror, gbdt, objective, metric, boosting, tree, oracle-harness, nyquist]

# Dependency graph
requires:
  - phase: 05-tree-learner-split-finding
    provides: SerialTreeLearner growth driver, DataPartition::indices_in_leaf, learner_parity goldens
  - phase: 03-model-text-predict
    provides: lgbm-model Tree (shrinkage field, leaf/internal value arrays), ObjectiveKind::convert_output, GbdtModel
  - phase: 01-oracle-contract-foundations
    provides: lgbm-core types (K_ZERO_THRESHOLD, K_EPSILON, ScoreT/LabelT f32 contract), thiserror error idiom
provides:
  - "Four net-new workspace crates: lgbm-objective, lgbm-metric, lgbm-boosting, lgbm (scaffolds + error boundaries)"
  - "ObjectiveError / MetricError / BoostingError thiserror enums (BoostingError #[from]-wraps the three upstream)"
  - "Tree::shrinkage(rate) / Tree::add_bias(val) with MaybeRoundToZero (signed-zero normalized)"
  - "SerialTreeLearner::add_prediction_to_score f64 score-scatter + renew_tree_output hook seam"
  - "Wave-0 failing boosting_parity scaffold (6 #[ignore]d Nyquist tests) + xtask boosting-oracle-capture stub + manifest section + fixtures dir"
affects: [06-02-spine-slice, 06-03-objectives, 06-04-metrics, 06-05-bagging-early-stopping, 08-pyo3-bindings]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Interface-first crate scaffolding: empty-but-compiling skeletons with thiserror boundaries before any loop math"
    - "CMP-01 containment: boosting crate names no GPU compute runtime / lgbm-compute (mirrors lgbm-treelearner)"
    - "Renew-hook seam via Option<closure> to avoid inverting crate dependency direction (treelearner stays objective-free)"

key-files:
  created:
    - crates/lgbm-objective/{Cargo.toml,src/lib.rs,src/error.rs}
    - crates/lgbm-metric/{Cargo.toml,src/lib.rs,src/error.rs}
    - crates/lgbm-boosting/{Cargo.toml,src/lib.rs,src/error.rs}
    - crates/lgbm/{Cargo.toml,src/lib.rs}
    - crates/oracle-harness/tests/boosting_parity.rs
    - crates/oracle-harness/tests/fixtures/boosting/README.md
    - crates/oracle-harness/tests/fixtures/REFERENCE_MANIFEST.md
  modified:
    - Cargo.toml
    - crates/lgbm-model/src/tree.rs
    - crates/lgbm-treelearner/src/learner.rs
    - crates/oracle-harness/Cargo.toml
    - xtask/src/main.rs

key-decisions:
  - "add_prediction_to_score / renew_tree_output take an explicit &DataPartition argument (the learner builds its partition locally in train_inner and does not retain it on self) — reproduces the C++ data_partition_ member as a parameter, keeping the scatter math identical."
  - "renew_tree_output Wave-0 seam takes Option<Fn(i32,&[u32])->f64> instead of an objective trait, so lgbm-treelearner does NOT gain an lgbm-objective dependency (which would invert the dependency direction); None == IsRenewTreeOutput()==false no-op."
  - "Phase-6 REFERENCE_MANIFEST section authored at crates/oracle-harness/tests/fixtures/REFERENCE_MANIFEST.md (co-located with boosting fixtures), NOT appended to the xtask-generated crates/oracle-harness/fixtures/REFERENCE_MANIFEST.md, to avoid clobbering the byte-idempotent generated file on the next capture."

patterns-established:
  - "Nyquist scaffold: #[ignore = \"MISSING — implemented in wave N\"] end-to-end tests sample the test surface before the implementation lands."
  - "MaybeRoundToZero (tree.h:258) as a private fn reusing the existing is_zero(K_ZERO_THRESHOLD) helper; normalizes -0.0 -> +0.0."

requirements-completed: [BST-01, BST-02, OBJ-03]

# Metrics
duration: ~6min
completed: 2026-06-07
---

# Phase 6 Plan 01: GBDT Spine Wave-0 Foundation Summary

**Four compiling net-new engine crates (lgbm-objective/metric/boosting/lgbm) with thiserror boundaries, Tree::shrinkage/add_bias (MaybeRoundToZero), learner add_prediction_to_score f64 score-scatter + renew_tree_output seam, and a 6-test #[ignore]d Nyquist boosting_parity scaffold — `cargo build --workspace` and `cargo test --workspace` green.**

## Performance

- **Duration:** ~6 min
- **Started:** 2026-06-07T08:27Z
- **Completed:** 2026-06-07T08:33Z
- **Tasks:** 3
- **Files modified:** 19 (12 created + 5 modified + 2 generated lock/members; excludes Cargo.lock churn)

## Accomplishments
- Scaffolded and workspace-registered the four Phase-6 crates; each builds and exposes its thiserror error boundary. `lgbm-boosting` is provably free of any GPU compute runtime dependency (CMP-01).
- Extended `lgbm-model::Tree` with `shrinkage(rate)` and `add_bias(val)` mirroring `tree.h:188/213`, both applying `MaybeRoundToZero` (`tree.h:258`) with `-0.0 -> +0.0` normalization.
- Extended the Phase-5 `SerialTreeLearner` with the bit-exact f64 per-leaf `add_prediction_to_score` score-scatter (`serial_tree_learner.h:100-118`) and a stable `renew_tree_output` hook seam — Phase-5 `learner_parity` unregressed (12 passed).
- Stood up the Wave-0 `boosting_parity.rs` Nyquist scaffold (6 `#[ignore]`d tests naming the spine golden + L1-L5 layers), the `boosting-oracle-capture` xtask stub, the Phase-6 manifest section, and the tracked `tests/fixtures/boosting/` dir.

## Task Commits

Each task was committed atomically:

1. **Task 1: Scaffold four crates + workspace registration + error enums** - `de99e4f` (feat)
2. **Task 2: Tree shrinkage/add_bias + learner add_prediction_to_score/renew_tree_output** - `dcf9c1c` (feat)
3. **Task 3: Wave-0 boosting_parity scaffold + capture stub + manifest section** - `7188a64` (test)

## Files Created/Modified
- `Cargo.toml` - Registered the four new crates in `members`.
- `crates/lgbm-objective/{Cargo.toml,src/lib.rs,src/error.rs}` - Training-side objective crate scaffold; `ObjectiveError` (LengthMismatch, LabelOutOfRange).
- `crates/lgbm-metric/{Cargo.toml,src/lib.rs,src/error.rs}` - Metric crate scaffold; `MetricError` (LengthMismatch).
- `crates/lgbm-boosting/{Cargo.toml,src/lib.rs,src/error.rs}` - GBDT-loop crate scaffold; `BoostingError` `#[from]`-wraps the three upstream errors; no compute-runtime dep.
- `crates/lgbm/{Cargo.toml,src/lib.rs}` - Umbrella facade crate; curated `pub use` re-exports of stable types.
- `crates/lgbm-model/src/tree.rs` - `Tree::shrinkage` / `Tree::add_bias` + `maybe_round_to_zero` helper + 4 unit tests.
- `crates/lgbm-treelearner/src/learner.rs` - `add_prediction_to_score` / `renew_tree_output` + 3 unit tests.
- `crates/oracle-harness/Cargo.toml` - Added the four crates as `[dev-dependencies]` (library stays compute-runtime-free).
- `crates/oracle-harness/tests/boosting_parity.rs` - 6 `#[ignore]`d Nyquist tests.
- `crates/oracle-harness/tests/fixtures/boosting/README.md` - Fixtures-dir placeholder + capture instructions.
- `crates/oracle-harness/tests/fixtures/REFERENCE_MANIFEST.md` - Phase-6 boosting golden-set section (L1-L5 layer table + D-07 collapse note).
- `xtask/src/main.rs` - `boosting-oracle-capture` subcommand arm + stub fn.

## Decisions Made
- See `key-decisions` frontmatter. The two load-bearing ones: (1) `add_prediction_to_score`/`renew_tree_output` take an explicit `&DataPartition` because the learner does not retain one on `self`; (2) the renew hook uses an `Option<closure>` rather than an objective trait to keep `lgbm-treelearner` free of an `lgbm-objective` dependency.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] renew_tree_output seam signature adapted to avoid a dependency-direction inversion**
- **Found during:** Task 2
- **Issue:** The plan's suggested `renew_tree_output(&self, tree, obj: &impl ...)` would require `lgbm-treelearner` to reference an objective trait, but `lgbm-treelearner` must NOT depend on `lgbm-objective` (boosting depends on both; the learner is below the objective layer). There is also no owned `DataPartition` on the learner to scatter over.
- **Fix:** Implemented the seam as `renew_tree_output<F: Fn(i32,&[u32])->f64>(&self, tree, data_partition, renew: Option<F>)` — `None` is the `IsRenewTreeOutput()==false` no-op; the real median-residual closure is supplied by the boosting layer in 06-03. `add_prediction_to_score` likewise takes `&DataPartition` explicitly.
- **Files modified:** crates/lgbm-treelearner/src/learner.rs
- **Verification:** `renew_tree_output_none_is_noop` + `add_prediction_to_score_*` tests pass; Phase-5 learner_parity unregressed (12 passed).
- **Committed in:** dcf9c1c (Task 2 commit)

**2. [Rule 3 - Blocking] Phase-6 manifest section placed at tests/fixtures/, not appended to the generated manifest**
- **Found during:** Task 3
- **Issue:** The plan's read_first + acceptance criterion reference `crates/oracle-harness/tests/fixtures/REFERENCE_MANIFEST.md`, but the actual existing manifest lives at `crates/oracle-harness/fixtures/REFERENCE_MANIFEST.md` and is GENERATED idempotently by `xtask write_manifest` — a manual edit there would be clobbered on the next capture.
- **Fix:** Authored the Phase-6 boosting section as a new tracked file at `crates/oracle-harness/tests/fixtures/REFERENCE_MANIFEST.md` (co-located with the boosting fixtures it documents), satisfying the literal `grep -q 'Boosting' ...tests/fixtures/REFERENCE_MANIFEST.md` gate while leaving the generated manifest byte-idempotent.
- **Files modified:** crates/oracle-harness/tests/fixtures/REFERENCE_MANIFEST.md (new)
- **Verification:** `grep -q 'Boosting'` succeeds; the generated manifest is untouched.
- **Committed in:** 7188a64 (Task 3 commit)

---

**Total deviations:** 2 auto-fixed (both Rule 3 - blocking/architecture-respecting adaptations)
**Impact on plan:** Both adaptations preserve the plan's intent and acceptance gates while respecting the actual crate dependency graph and the idempotent-manifest invariant. No scope creep; no objective/metric/loop math implemented (as intended for Wave 0).

## Issues Encountered
- Two unit-test assertions initially used magnitudes (1e-30) ABOVE `K_ZERO_THRESHOLD` (1e-35) and the partition split count expectation (4,4) did not match the partition's threshold convention ((2,6)). Fixed by using genuinely sub-threshold magnitudes (1e-40) for the MaybeRoundToZero tests and asserting the scatter invariants (full coverage, both leaves non-empty, per-leaf value match) rather than a hard-coded split count. All tests green.

## User Setup Required
None - no external service configuration required. (The `boosting-oracle-capture` stub does not yet require a `lightgbm` wheel; the real capture in wave 2+ will.)

## Next Phase Readiness
- 06-02 (spine slice) is UNBLOCKED: the four crates compile against stable error contracts, `Tree::shrinkage`/`add_bias` and `add_prediction_to_score`/`renew_tree_output` exist as the loop's call seams, and the `boosting_parity` scaffold names the goldens to fill in.
- No blockers. CMP-01 holds (boosting names no compute runtime); `LightGBM/` never git-added.

## Self-Check: PASSED

All 7 sampled created files exist on disk; all 3 task commit hashes (`de99e4f`, `dcf9c1c`, `7188a64`) are present in git history.
