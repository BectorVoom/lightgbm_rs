---
phase: 20-on-device-score-updater-metrics
plan: 04
subsystem: boosting
tags: [cubecl, on-device-score, resident-score, gbdt, score-updater, odl-16, ab-parity]

# Dependency graph
requires:
  - phase: 20-01
    provides: "ScoreUpdater boosting_on_cuda_ toggle + add_tree_train_path_on_device / *_on resident delegates"
  - phase: 20-02
    provides: "§12 device metric evaluator (Metric.Eval over the resident score, downstream)"
  - phase: 20-03b
    provides: "the activated on-device grow driver (grow_tree_on_device) the resident loop is fed by"
  - phase: 18
    provides: "add_prediction_to_score_on_device tree-walk kernel (the per-leaf AddScore delegate, D-02)"
provides:
  - "GBDT::TrainOneIter resident cross-iteration score loop behind LGBM_CUDA_ON_DEVICE (§16 order, L2 slice)"
  - "Gbdt::set_boosting_on_cuda / boosting_on_cuda driver+test seam"
  - "SerialTreeLearner::client() + features() read-only accessors"
  - "resident_score_ab.rs — the D-06 layer-2 resident-score A/B (resident cuda_score_ vs host score_)"
affects: [21-on-device-multi-leaf-grow-loop, on-device-l1-quantile-renew, dart-rf-on-device]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Resident score residency proven via the set_boosting_on_cuda A/B seam (both arms in one process, cpu f64 anchor bit-exact)"
    - "PredictTree reconstruction from a grown Tree: predict-remap offset is 0 (min_bin-relative), DISTINCT from FeatureColumn::offset (histogram-scan compaction)"

key-files:
  created:
    - "crates/oracle-harness/tests/resident_score_ab.rs"
  modified:
    - "crates/lgbm-boosting/src/gbdt.rs"
    - "crates/lgbm-treelearner/src/learner.rs"

key-decisions:
  - "Resident per-leaf UpdateScore routes through the Phase-18 add_prediction_to_score_on_device tree-walk delegate (D-02, no new kernel); on the identity-binned L2 corpus it is bit-exact to the host partition scatter."
  - "Predict-remap feat_offset for the walk is 0 (the grown tree's threshold_in_bin is already in the min_bin-relative predict space) — NOT FeatureColumn::offset, which is the histogram-scan compaction offset."
  - "Features for the PredictTree reconstruction are sourced from the learner (learner.features()), the authoritative holder during training, not the GBDT-spine self.features (empty on the non-bagging path)."
  - "A/B arms selected via Gbdt::set_boosting_on_cuda (the cuda_on_device_enabled() env is a process-global OnceLock — per-arm env toggling is impossible)."
  - "§16 ordering contract documented: RenewTreeOutput stays BEFORE shrinkage+UpdateScore; L2 slice applies no refit; DART/RF per-row-predict paths stay host-side (Pitfall 5)."

patterns-established:
  - "Resident-score A/B: force both arms (set_boosting_on_cuda true/false) in one process, anchor to the host/cpu f64 accumulation, compare_exact_f64_bits (never GPU-vs-GPU)."
  - "Grown-Tree → PredictTree reconstruction convention (predict offset 0, min_bin shift, empty categorical slabs for the L2 continuous slice)."

requirements-completed: [ODL-16, ODL-19]

# Metrics
duration: 25 min
completed: 2026-07-02
status: complete
---

# Phase 20 Plan 04: Resident Cross-Iteration Score Loop Summary

**GBDT::TrainOneIter keeps `cuda_score_` resident across the whole train behind `LGBM_CUDA_ON_DEVICE` — the per-leaf §11 AddScore routes through the Phase-18 device tree-walk delegate in §16 order — proven bit-exact to the host partition scatter after a full multi-iteration run (D-06 layer 2), with the env-unset path byte-unchanged.**

## Performance

- **Duration:** ~25 min
- **Started:** 2026-07-02T04:52:00Z
- **Completed:** 2026-07-02T05:17:05Z
- **Tasks:** 2
- **Files modified:** 3 (2 modified, 1 created)

## Accomplishments
- Wired the resident cross-iteration score loop into `GBDT::TrainOneIter` (full-corpus L2 continuous slice) behind the `boosting_on_cuda_` toggle: shrinkage on the tree → resident per-leaf UpdateScore via `add_prediction_to_score_on_device` (D-02) → RenewTreeOutput no-op for L2 → Metric.Eval downstream over the mirrored score. Closes the ODL-16 residency half.
- Authored `resident_score_ab.rs` (D-06 layer 2): a full multi-iteration L2 train run twice on the cpu backend (resident vs host arm) asserts the resident/mirrored `cuda_score_` equals the host `score_` **bit-for-bit** on the cpu f64 anchor; an opt-in `rocm` cell holds the hip resident arm to the ~1e-6 f32 envelope against the same cpu-anchor host accumulation (never GPU-vs-GPU).
- Env-unset path byte-unchanged (D-09/ODL-19): the resident branch is gated on `boosting_on_cuda()` (OFF by default with `LGBM_CUDA_ON_DEVICE` unset), so `cargo test --workspace` is green and unchanged.

## Task Commits

1. **Task 1: Wire the resident score loop into GBDT::TrainOneIter (§16 order, L2 slice)** — `b8a2f0a` (feat)
2. **Task 2: Resident-score A/B test — resident cuda_score_ vs host score_ (D-06 layer 2)** — `9cd1c14` (test)

## Files Created/Modified
- `crates/lgbm-boosting/src/gbdt.rs` — additive resident-loop wiring in `TrainOneIter` (gated `update_score_resident` helper reconstructing a `PredictTree` from the grown tree + learner columns), `set_boosting_on_cuda`/`boosting_on_cuda` seam, §16 ordering contract comment.
- `crates/lgbm-treelearner/src/learner.rs` — `SerialTreeLearner::client()` + `features()` read-only accessors (the only `B::Runtime` client / authoritative feature-column holder in the boosting call graph).
- `crates/oracle-harness/tests/resident_score_ab.rs` — the D-06 layer-2 resident-score A/B (new).

## Gate Results

- `cargo test --workspace` (LGBM_CUDA_ON_DEVICE unset): **green, byte-unchanged** — 81 `test result: ok` groups, 0 failed, 0 errors.
- `LGBM_CUDA_ON_DEVICE=1 cargo test -p oracle-harness --test resident_score_ab`: **`test result: ok. 2 passed; 0 failed`**.
  - Resident-score A/B cell (exact): `test resident_score_matches_host_after_full_train_cpu_anchor ... ok` → **`test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 1 filtered out`**.
- `cargo test -p lgbm-boosting`: **`test result: ok. 55 passed; 0 failed`**.
- `cargo test -p oracle-harness --test score_updater_parity` (20-01 companion, unregressed): **`test result: ok. 3 passed; 0 failed`**.

## Decisions Made
See frontmatter `key-decisions`. The load-bearing one: the walk's predict-remap `feat_offset` is **0** (the grown tree's `threshold_in_bin` already encodes the compaction/most-freq-bin shift in the `min_bin`-relative predict space), matching the canonical `lib_lightgbm` predict golden. Reusing `FeatureColumn::offset` (the histogram-scan compaction offset) double-counted the shift and mis-routed every split — see the deviation below.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Wrong predict-remap offset in the PredictTree reconstruction**
- **Found during:** Task 2 (the A/B first failed: resident row 0 = 1.02 vs host 1.72, a systematic one-bin route shift).
- **Issue:** The initial reconstruction fed `FeatureColumn::offset` (the histogram-scan compaction offset, `= 1` when `most_freq_bin == 0`) as the walk's predict-remap offset. The CUDA-tree walk remaps `bin = raw − min_bin + offset` and compares against the tree's `threshold_in_bin`, which is ALREADY recorded in the `min_bin`-relative predict space — so adding `offset` again shifted every split by one bin, routing each row to the next-higher leaf.
- **Fix:** Set the walk's `feat_offset` to `0` for all features (verified against the canonical `predict.rs::numeric_tree` golden, which uses `feat_offset = [0, 0]` even for non-zero `most_freq_bin`). Documented the distinction in a code comment.
- **Files modified:** `crates/lgbm-boosting/src/gbdt.rs`
- **Verification:** the resident A/B then matches the host `score_` bit-for-bit across a 6-iteration L2 train (`compare_exact_f64_bits`).
- **Committed in:** `b8a2f0a` (folded into the Task 1 commit).

**2. [Rule 3 - Blocking] Missing learner accessors needed to route the resident path**
- **Found during:** Task 1.
- **Issue:** The resident per-leaf delegate needs a `&ComputeClient<B::Runtime>` and the tree's feature columns, but `GBDT::TrainOneIter` had no client in scope (the learner holds the only `B::Runtime`-generic client, privately) and the GBDT-spine `self.features` is empty on the non-bagging path.
- **Fix:** Added read-only `SerialTreeLearner::client()` and `SerialTreeLearner::features()` accessors (additive, no behavior change); `update_score_resident` sources both from the learner.
- **Files modified:** `crates/lgbm-treelearner/src/learner.rs`, `crates/lgbm-boosting/src/gbdt.rs`
- **Verification:** compiles; env-unset `cargo test --workspace` unchanged; the learner's own suite unregressed.
- **Committed in:** `b8a2f0a` (Task 1 commit).

---

**Total deviations:** 2 auto-fixed (1 bug, 1 blocking). Both confined to the plan's files plus two additive read-only accessors on `learner.rs`.
**Impact on plan:** Both necessary for correctness; no scope creep. `files_modified` was `gbdt.rs` + the new test; the two `learner.rs` accessors are the minimal read-only seam the resident routing requires (the learner is the sole `B::Runtime` client / feature-column holder in the boosting call graph).

## Issues Encountered
- The predict-remap offset convention (deviation 1) was the only real snag — resolved by anchoring to the canonical `lib_lightgbm` predict golden rather than reusing the histogram-scan offset.

## Ordering Contract (L1/quantile follow-up)
The §16 order is fixed in `TrainOneIter`: `Shrinkage → UpdateScore(§11) → optional RenewTreeOutput(§5.1) → Metric.Eval(§12)`. RenewTreeOutput runs BEFORE shrinkage+UpdateScore (it reads the pre-update score), so a future device RenewTreeOutput refit (L1/quantile) slots in at the existing renew site, NOT at the UpdateScore site — do not reorder. The L2 slice applies no refit. The DART/RF per-row-predict paths (`add_tree_predict_path` / `add_tree_scaled_all`) remain host-side this phase (Pitfall 5).

## Known Stubs
None. The resident branch is fully wired and proven bit-exact; the `rocm` gpu cell is opt-in behind `--features rocm` (compiled only when the hip runtime is present), consistent with the existing oracle-harness ROCm gating.

## Next Phase Readiness
- ODL-16 residency half closed; the resident score buffer never leaves device across the train (L2 slice) and mirrors correctly for non-resident consumers.
- Phase-21 (multi-leaf on-device grow loop) can consume the same `boosting_on_cuda_` seam; note the pre-existing `HistArena::swap` aliasing caveat (project memory `phase18-wr01-histarena-swap-aliasing`) still applies to that loop.
- Follow-up: L1/quantile device RenewTreeOutput refit + DART/RF on-device per-row-predict remain out of the L2 slice (ordering contract documented above).

## Self-Check: PASSED
- `crates/oracle-harness/tests/resident_score_ab.rs` — FOUND on disk.
- Commit `b8a2f0a` (feat, Task 1) — FOUND in git log.
- Commit `9cd1c14` (test, Task 2) — FOUND in git log.
- All plan `<verification>` commands green (see Gate Results).

---
*Phase: 20-on-device-score-updater-metrics*
*Completed: 2026-07-02*
