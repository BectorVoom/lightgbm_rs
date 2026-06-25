---
phase: 12-gpu-sibling-scan-copack
plan: 01
subsystem: infra
tags: [cubecl, rocm, gpu, histogram-scan, sibling-copack, resident-pool, treelearner]

# Dependency graph
requires:
  - phase: 11-quantized-training / spike-021
    provides: feature-per-lane W=64 scan (scan_cube_dim / LGBM_SCAN_CUBEDIM), the single-slot find_best_splits_fused_kernel + split_scan_body
  - phase: 260608-p90
    provides: device-resident histogram pool (build_resident_leaf_into, subtract_resident, scan_resident_leaf, the slot mirror)
provides:
  - find_best_splits_fused_siblings_kernel (2-slot co-packed #[cube(launch)] scan kernel)
  - find_best_splits_fused_siblings_from_handles_on (two-Handle launcher -> one launch -> one readback -> two SplitInfo vecs)
  - Backend::scan_resident_siblings (trait default error + RocmBackend one-launch impl)
  - LGBM_SIBLING_COPACK env override (resident_pool::sibling_copack_override)
  - find_best_splits growth-loop reorder (defer smaller scan past subtract, co-pack both siblings)
  - scan_leaf_histogram precomputed_batched_splits param + spine_batched_feats helper
affects: [12-02-PLAN (kernel_parity co-pack cell), 12-03-PLAN (device-time + e2e A/B + SCAN_RESIDENT_CNT)]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "2-slot co-packed kernel: global lane g<n => sibling A/hist_a, n<=g<2n => sibling B/hist_b; SHARED per-feature arrays + PER-SIBLING leaf scalars; calls the SHARED split_scan_body per arm (branch on sibling because Array refs can't be selected into a binding)"
    - "Precomputed-splits injection: scan_leaf_histogram accepts Option<Vec<SplitInfo>> to skip its source dispatch and reuse the SHARED post-scan bookkeeping, keeping the co-pack and two-scan paths byte-identical"
    - "Spine-equality gate: co-pack only fires when both siblings yield byte-equal spine_batched_feats (guards against per-node col-sampler mask divergence)"

key-files:
  created: []
  modified:
    - crates/lgbm-compute/src/kernels/split.rs
    - crates/lgbm-compute/src/lib.rs
    - crates/lgbm-treelearner/src/learner.rs
    - crates/lgbm-treelearner/src/resident_pool.rs

key-decisions:
  - "CpuBackend inherits the default scan_resident_siblings error (no dead delegate): resident_pool_supported()=false, so the learner co-pack gate (ANDs in resident_eligible) never routes CpuBackend here"
  - "Co-pack eligibility adds a spine-equality check (smaller_feats == larger_feats) on top of the resident-scan-only gate: resident_eligible does NOT rule out per-node col-sampling, so the two siblings' spine membership can differ; differing spines fall back to the two-scan path"
  - "scan_leaf_histogram gains a precomputed_batched_splits param rather than duplicating the post-scan bookkeeping; this keeps records/argmax/feature_splittable byte-identical between co-pack and fall-back"

patterns-established:
  - "Co-packed device scan: one launch over 2*n feature-slots + one read_one_unchecked -> two decoded SplitInfo halves (per-sibling min_gain_shift applied to each half)"
  - "Growth-loop deferral: hoist larger build/subtract ahead of either scan so both Handles are simultaneously resident, then co-pack; build/subtract ORDER unchanged"

requirements-completed: [SC-2, SC-3]

# Metrics
duration: 35min
completed: 2026-06-25
status: complete
---

# Phase 12 Plan 01: gpu-sibling-scan-copack Summary

**Co-packs the two per-sibling resident histogram scans into ONE 2-slot scan_resident_siblings launch + ONE readback (halving per-tree scan syncs ~59->~30), gated behind LGBM_SIBLING_COPACK, bit-exact by construction with the CPU anchor byte-untouched.**

## Performance

- **Duration:** 35 min
- **Started:** 2026-06-25
- **Completed:** 2026-06-25
- **Tasks:** 3
- **Files modified:** 4

## Accomplishments
- New `find_best_splits_fused_siblings_kernel` (2-slot `#[cube(launch)]`) scans both siblings' resident Handles in one launch over `2*n_feats` lanes, calling the SHARED `split_scan_body` per arm — bit-identical to two single-slot scans by construction.
- New `find_best_splits_fused_siblings_from_handles_on` launcher: two resident Handles -> one launch (`CubeCount=ceil(2n/W)`) -> ONE `read_one_unchecked` -> two decoded `Vec<SplitInfo>`; the `2*kEpsilon` bump + `min_gain_shift` computed ONCE PER SIBLING with each sibling's raw totals.
- New `Backend::scan_resident_siblings` (trait default unsupported error + `RocmBackend` one-launch impl borrowing both resident Handles).
- Growth-loop reorder in `find_best_splits`: larger build/subtract hoisted ahead of either scan so both Handles are simultaneously resident; the smaller-child scan deferred past `subtract_resident` and co-packed when eligible; byte-unchanged two-scan fall-back otherwise.
- `LGBM_SIBLING_COPACK` env override (`0` force-off byte-identical, `1` force-on), mirroring `LGBM_RESIDENT_FORCE`.

## Task Commits

Each task was committed atomically:

1. **Task 1: 2-slot co-packed scan kernel + Handle launcher** - `1f49650` (feat)
2. **Task 2: Backend::scan_resident_siblings (default + RocmBackend impl)** - `73a56e7` (feat)
3. **Task 3: growth-loop co-pack reorder + LGBM_SIBLING_COPACK gate** - `4bb7da7` (feat)

## Files Created/Modified
- `crates/lgbm-compute/src/kernels/split.rs` - added `find_best_splits_fused_siblings_kernel` (2-slot kernel) + `find_best_splits_fused_siblings_from_handles_on` (two-Handle launcher, one readback, two decoded vecs). Single-slot kernel/inner/`split_scan_body` untouched.
- `crates/lgbm-compute/src/lib.rs` - added `Backend::scan_resident_siblings` (default error) + `RocmBackend::scan_resident_siblings` (one-launch impl, both-slot borrow, empty-slot guard). `scan_resident_leaf`/`subtract_resident` untouched.
- `crates/lgbm-treelearner/src/learner.rs` - growth-loop reorder (hoist larger build/subtract, defer smaller scan, co-pack via `scan_resident_siblings`, distribute two vecs); `scan_leaf_histogram` gained `precomputed_batched_splits` param; new `spine_batched_feats` helper.
- `crates/lgbm-treelearner/src/resident_pool.rs` - `sibling_copack_override()` reading `LGBM_SIBLING_COPACK`.

## Wiring details (per the plan's output spec)

- **New symbols:** `find_best_splits_fused_siblings_kernel`, `find_best_splits_fused_siblings_from_handles_on`, `Backend::scan_resident_siblings`, `RocmBackend::scan_resident_siblings`, `resident_pool::sibling_copack_override`, `Learner::spine_batched_feats`, `scan_leaf_histogram(... precomputed_batched_splits ...)`.
- **`sibling_copack` eligibility predicate as wired (ALL must hold):**
  `self.resident_eligible` AND `LGBM_SIBLING_COPACK != Some(false)` AND
  smaller on resident scan-only path (`smaller_resident_slot == Some(smaller_slot)` AND `!smaller_fused` AND `!smaller_unified`) AND
  smaller scannable (`sum_hessians > 0.0 && num_data_in_leaf > 0`) AND
  `larger_leaf >= 0` AND larger on resident subtract+scan-only path (`larger_resident_slot == larger_slot_id`, `larger_slot_id.is_some()`, `!larger_unified`) AND
  larger scannable (`sum_hessians > 0.0 && num_data_in_leaf > 0`) AND
  **both siblings' `spine_batched_feats` are byte-equal and non-empty** (the SHARED `feats` the co-packed kernel scans). Any false case -> byte-unchanged two separate scans.
- **Smaller scan deferred past `subtract_resident`:** the larger build/subtract block now runs FIRST (computing `larger_resident_slot`/`larger_unified`/`larger_subtract_inputs`), THEN co-pack eligibility is decided, THEN both scans run. The smaller `scan_leaf_histogram` call moved to AFTER the larger subtract, so at the co-pack call both Handles (smaller at `smaller_slot`, larger subtract-derived at `larger_slot`) are simultaneously resident. Build/subtract ORDER is unchanged (no spike-016 reorder).
- **CpuBackend routing decision:** inherits the trait DEFAULT error. CpuBackend's `resident_pool_supported()` returns `false`, so `self.resident_eligible` is false on CpuBackend and the co-pack gate never routes it to `scan_resident_siblings`. No dead CpuBackend impl was added (per CONTEXT: the simpler option, and the only correct one since CpuBackend has no resident pool).
- **`SCAN_RESIDENT_CNT` once per co-packed pair:** the co-pack branch bumps `SCAN_RESIDENT_CNT` ONCE (one readback) before the single `scan_resident_siblings` call, then feeds the two vecs via `precomputed_batched_splits` so neither `scan_leaf_histogram` re-enters the `resident_slot` dispatch (which is what bumps the counter). The two-scan fall-back still bumps twice (once per `scan_resident_leaf`). Net: ~59 -> ~30 syncs/tree on the eligible resident path (SC-3, to be measured in Plan 03).
- **Byte-untouched confirmation:** `find_best_splits_fused_kernel`, `find_best_splits_fused_inner`, `split_scan_body`, `find_best_splits_batched_fused_f64_from_handle_on`, `scan_resident_leaf`, `subtract_resident`, `build_resident_leaf_into`, and CPU/GPU routing are all git-unchanged (additive only). The CPU f64 anchor never reaches the co-pack path.

## Decisions Made
- See key-decisions in frontmatter. The notable one: the plan's read-first noted `resident_eligible` might rule out per-node col-sampling; it does NOT (it gates monotone/interaction/extra-trees/categorical/forced-splits/smoothing, but not feature_fraction). So the two siblings' spine membership can differ when per-node col-sampling is active. The co-pack gate therefore adds a `spine_batched_feats` equality check (Rule 2 — correctness requirement) so a differing-spine pair falls back to the two-scan path rather than scanning a mismatched shared `feats`.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 2 - Missing Critical] Spine-equality guard on co-pack eligibility**
- **Found during:** Task 3 (growth-loop reorder)
- **Issue:** The plan's eligibility predicate assumed both siblings share the same `batched_feats` ("they share the dataset feature layout"). That holds for the per-feature PARAM layout, but the spine MEMBERSHIP (which features are batched) also depends on the per-node col-sampler mask (`smaller_node_mask` vs `larger_node_mask`), which `resident_eligible` does NOT rule out (it gates monotone/interaction/extra-trees/categorical/forced-splits/smoothing only). With per-node col-sampling active the two spines can differ, and feeding a single shared `feats` to the 2-slot kernel would mis-map results for the sibling whose spine differs.
- **Fix:** Added `Learner::spine_batched_feats` (a faithful replica of `scan_leaf_histogram`'s Pass-1 gate-only pre-pass) and require `smaller_feats == larger_feats && !empty` before co-packing; any mismatch falls back to the byte-unchanged two-scan path. This is the SHARED `feats` the kernel scans, so when equal the co-pack is provably bit-exact to the two scans.
- **Files modified:** crates/lgbm-treelearner/src/learner.rs
- **Verification:** `learner_parity_col_sampler_rng` + all `learner_parity_*` pass; `lgbm-treelearner --lib` (76) green.
- **Committed in:** `4bb7da7` (Task 3 commit)

---

**Total deviations:** 1 auto-fixed (1 missing-critical correctness guard)
**Impact on plan:** The guard is required for correctness on col-sampled workloads and tightens (never loosens) the co-pack gate; differing-spine pairs fall back byte-identically. No scope creep.

## Issues Encountered
None. The interleaved smaller-build/larger-subtract/scan region required a careful reorder (hoist larger build/subtract ahead of both scans, then defer the smaller scan), but the build/subtract ORDER stayed unchanged and all parity gates passed first try.

## Verification Results
- `cargo build -p lgbm-compute --features rocm` + CPU-only `cargo build` — both succeed; `cargo clippy -p lgbm-compute --features rocm` 0 errors.
- `cargo build -p lgbm-treelearner` (CPU) + `--features rocm` — both succeed; `cargo clippy -p lgbm-treelearner` 0 errors.
- `cargo test -p lgbm-treelearner --lib` — 76 passed, 0 failed.
- `cargo test -p oracle-harness raw_bin_train_matches_cpp_golden` — passed (bit-exact vs lib_lightgbm 4.6).
- `cargo test -p oracle-harness learner_parity` — all `learner_parity_*` passed.
- `cargo test -p lgbm-boosting` — 55 passed, 0 failed.
- `grep -c 'fn scan_resident_siblings' lib.rs` = 2 (trait default + RocmBackend); `grep -c 'find_best_splits_fused_siblings' split.rs` >= 2.

Note: the `--features rocm` runtime co-pack path (the actual 2-slot launch on hardware) is exercised by Plan 02's `kernel_parity` co-pack cell and Plan 03's device-time/e2e A/B; this plan compiles+links it on cubecl-hip and keeps the CPU merge gate green (the co-pack path never fires on CpuBackend).

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Plan 02 can extend `kernel_parity` (`--features rocm`) with a co-pack cell asserting `scan_resident_siblings` returns SplitInfos byte-identical to two separate `scan_resident_leaf` calls (W=1 byte-identical, hip within ~1e-6 of the CPU anchor).
- Plan 03 can wire the device-time + e2e A/B (`LGBM_SIBLING_COPACK=0/1`) and assert `SCAN_RESIDENT_CNT` ~halves per tree under `LGBM_PHASE_PROF=1`.
- The co-pack is OFF-path-byte-identical (gate ANDs in `resident_eligible` + spine-equality), so it is safe to land ahead of the hardware A/B.

## Self-Check: PASSED

All modified files exist on disk; all three task commits (`1f49650`, `73a56e7`, `4bb7da7`) are present in git history.

---
*Phase: 12-gpu-sibling-scan-copack*
*Completed: 2026-06-25*
