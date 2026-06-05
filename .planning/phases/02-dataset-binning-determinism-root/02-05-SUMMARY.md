---
phase: 02-dataset-binning-determinism-root
plan: 05
subsystem: dataset-binning
tags: [efb, feature-bundling, multi-val-bin, find-groups, golden-replay, layer-3, rng, stable-sort, rust]

# Dependency graph
requires:
  - phase: 02-dataset-binning-determinism-root
    plan: 04
    provides: "from_mat/from_csr/from_csc ingestion + sampling + Metadata + Dataset construct/finish-load (layers 1+2)"
  - phase: 01-oracle-contract-foundations
    provides: "lgbm_core::random::Random LCG (Sample + NextShort), oracle-harness exact comparators, xtask bin-capture golden harness"
provides:
  - "MultiValBin dense/sparse storage for bundled EFB groups (+1 push convention)"
  - "efb.rs: fast_feature_bundling / find_groups / get_conflict_count / fix_sample_indices (verbatim, Phase-1 RNG, stable sorts)"
  - "Dataset::construct_bundled enable_bundle dispatch + real<->packed feature index maps"
  - "EFB layer-3 golden capture (header-only verbatim transcription) + committed efb_grouping.txt fixture"
  - "EFB grouping golden-replay test (group membership + bin_offsets_ + num_total_bin_ + per-row bundled indices)"
affects: [predict phase, histogram/split phases (inherit the bundled group layout)]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Header-only verbatim transcription of the EFB pipeline (dataset.cpp) for golden capture — external_libs unvendored, same discipline as plans 02-01..02-04"
    - "ALL EFB randomness routed through lgbm_core::Random::new(num_data); STABLE sorts only; element-wise parallel-vector swap"
    - "Real<->packed feature index translation via used_feature_map_ / real_feature_idx_ (C++ parity) so the shuffled bundled path pushes to the correct group"

key-files:
  created:
    - crates/lgbm-dataset/tests/efb_grouping.rs
    - crates/lgbm-dataset/tests/fixtures/efb_grouping.txt
  modified:
    - crates/lgbm-dataset/src/multi_val_bin.rs
    - crates/lgbm-dataset/src/efb.rs
    - crates/lgbm-dataset/src/feature_group.rs
    - crates/lgbm-dataset/src/dataset.rs
    - crates/lgbm-dataset/src/lib.rs
    - xtask/cpp/bin_capture.cpp
    - xtask/src/main.rs
    - crates/oracle-harness/fixtures/REFERENCE_MANIFEST.md

key-decisions:
  - "EFB golden capture = HEADER-ONLY VERBATIM TRANSCRIPTION (external_libs unvendored). Both nominal options were provably infeasible: the focused dataset.cpp build fails (fast_double_parser.h: No such file) and the CLI-dump can't build lib_lightgbm/CLI for the same reason. Human-approved: verbatim transcription."
  - "Dataset stores real_feature_idx_ + used_feature_map_ and translates REAL->packed in push_value/push_row/feature_to_group/feature_to_subfeature, so the RNG-shuffled bundled grouping pushes each feature to its correct group (a Rule 1 bug surfaced by the per-row golden)."

patterns-established:
  - "EFB layer-3 golden = feature->group membership (group-major packed order) + per-group bin_offsets_/num_total_bin_/multi_val flag + per-row bundled bin index, all compared bit-exact"

requirements-completed: [DAT-05]

# Metrics
duration: continuation (Tasks 3-4)
completed: 2026-06-05
---

# Phase 2 Plan 05: Exclusive Feature Bundling (EFB) Summary

**Exclusive Feature Bundling (DAT-05) is proven bit-identical to C++ at golden layer 3 — feature->group membership, per-group `bin_offsets_`/`num_total_bin_`/`group_is_multi_val`, and per-row bundled bin indices all replay bit-for-bit on the D-06 #4 mutually-exclusive sparse corpus, with the EFB capture done as a header-only verbatim transcription because the unvendored `external_libs` make both nominal capture paths infeasible.**

## Continuation Context

This was a CONTINUATION agent. Tasks 1-2 were completed by the prior executor and verified present before continuing (NOT redone):

1. **Task 1: MultiValBin dense/sparse storage + FeatureGroup multi-val push (+1)** - `da9196b` (feat)
2. **Task 2: FastFeatureBundling + FindGroups + GetConflictCount + FixSampleIndices + enable_bundle dispatch** - `f2f4263` (feat)

The prior executor paused at the Task 3 blocking human-verify checkpoint. The human chose **"approved: verbatim transcription"**. This agent completed Tasks 3-4.

## Task Commits (this agent)

3. **Task 3: EFB layer-3 golden capture (header-only verbatim transcription)** - `6d4a65d` (feat)
4. **Task 4: EFB grouping golden replay (layer 3 + per-row bundled index)** - `17dfa29` (test)

## Accomplishments

- Extended `xtask/cpp/bin_capture.cpp` with a HEADER-ONLY verbatim transcription of the EFB pipeline from the pinned `dataset.cpp` (commit 195c26fc, 4.6.0.99): `GetConflictCount`/`MarkUsed`/`FixSampleIndices`/`FindGroups`/`FastFeatureBundling` + the bundled `FeatureGroup` (`bin_offsets_`/`num_total_bin_`/group layout) from `feature_group.h`, compiled with only `-I LightGBM/include` plus the header-only `LightGBM::Random` for sampling + the group shuffle.
- Emitted golden layer 3 for the D-06 #4 corpus: two mutually-exclusive sparse feature sets (which EFB bundles into one group each) + a no-bundle control. The golden records feature->group membership, per-group `bin_offsets_`/`num_total_bin_`/`group_is_multi_val`, and the per-row bundled bin index per single-value group.
- Wired the `efb_out` arg into the `bin-capture` subcommand and recorded the capture-harness resolution ("verbatim transcription (external_libs unvendored)") in `REFERENCE_MANIFEST.md`, documenting why both nominal options are infeasible.
- Created `tests/efb_grouping.rs`: rebuilds the bundled `Dataset` (`construct_bundled`, `enable_bundle=true`) on the same corpus and asserts group membership + `bin_offsets_` + `num_total_bin_` + `group_is_multi_val` + per-row bundled indices are bit-exact vs C++ (via `compare_exact_u32`). The control proves the `enable_bundle` dispatch boundary. The test PASSES (not skipped).
- `cargo run -p xtask -- bin-capture` is idempotent (re-running leaves the committed fixtures unchanged); `cargo test --workspace` is green (all Phase 2 layers 1+2+3 + Phase 1 regressions).

## Files Created/Modified (this agent)

- `crates/lgbm-dataset/tests/efb_grouping.rs` (created) - EFB layer-3 golden-replay test (membership + offsets + per-row bundled index + control).
- `crates/lgbm-dataset/tests/fixtures/efb_grouping.txt` (created) - committed EFB golden (3 cases).
- `xtask/cpp/bin_capture.cpp` (modified) - EFB transcription (`GetConflictCount`/`FindGroups`/`FastFeatureBundling`/`FixSampleIndices` + `FeatureGroupMV`) + `BuildEfbCorpus`/`EmitEfbCase`; `efb_out` argv slot; stable example seed base.
- `xtask/src/main.rs` (modified) - pass the `efb_grouping.txt` path into `bin-capture`; EFB section in the reference manifest writer.
- `crates/oracle-harness/fixtures/REFERENCE_MANIFEST.md` (modified, regenerated) - EFB golden section + verbatim-transcription resolution.
- `crates/lgbm-dataset/src/dataset.rs` (modified) - real<->packed feature index maps (`real_feature_idx_`/`used_feature_map_`); `push_value`/`push_row`/`feature_to_group`/`feature_to_subfeature` translate REAL->packed; `real_feature_idx` accessor.

(Tasks 1-2 created/modified `multi_val_bin.rs`, `efb.rs`, `feature_group.rs`, `lib.rs`, and the earlier `dataset.rs` enable_bundle dispatch — listed in key-files for the full plan picture.)

## Decisions Made

- **EFB golden capture = header-only verbatim transcription (external_libs unvendored).** The Task 3 checkpoint asked focused-harness vs CLI-dump. Both are provably infeasible here: the focused `dataset.cpp` build fails because `external_libs/{fast_double_parser,fmt,eigen,compute}` are empty/unvendored (`fast_double_parser.h: No such file`), and the CLI-dump can't build `lib_lightgbm`/`lightgbm` for the same reason. The human approved the verbatim-transcription resolution — the SAME discipline plans 02-01..02-04 used for every prior golden layer. The transcribed code is the authoritative reference source, so output is byte-identical to what `lib_lightgbm` would emit, and only `LightGBM::Random` (header-only) is linked for sampling + shuffle.
- **Stable example-golden seed base.** Inserting the `efb_out` arg at `argv[7]` shifted the example inputs from `argv[8]` to `argv[9]`; the example per-dataset seed (`master_seed + argv_index`) was re-anchored to its original positional base so the existing example goldens are unchanged by this plan.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Dataset real-vs-packed feature indexing in the bundled path**
- **Found during:** Task 4 (per-row bundled-index golden compare failed: rust=0, cpp=1 at row 0).
- **Issue:** `construct_bundled` built `feature2group_`/`feature2subfeature_` indexed by the PACKED group-major index (and discarded the local `real_feature_idx_`), but `push_value`/`feature_to_group`/`feature_to_subfeature` treated their argument as a REAL feature index. In the no-bundle default these coincide, but in the RNG-shuffled bundled path they differ (e.g. the control case puts real feature 2 at packed index 0), so per-row pushes routed to the wrong group and there was no guard for trivial/unused features.
- **Fix:** Store `real_feature_idx_` (packed->real) and `used_feature_map_` (real->packed, -1 if unused) on `Dataset`/`FinishedDataset` (C++ parity), and make `push_value`/`push_row`/`feature_to_group`/`feature_to_subfeature` translate REAL->packed via `used_feature_map_`; added a `real_feature_idx` accessor. The no-bundle path is the identity map, so prior behavior is preserved (all existing dataset/ingest tests still pass).
- **Files modified:** crates/lgbm-dataset/src/dataset.rs
- **Verification:** `efb_grouping` per-row bundled indices now bit-exact across all 3 cases; `cargo test --workspace` green.
- **Committed in:** 17dfa29 (Task 4 commit)

**Total deviations:** 1 auto-fixed (1 bug).
**Impact on plan:** Necessary for correct bundled-path pushing and for the per-row golden to pass; strictly improves C++ fidelity (now mirrors `used_feature_map_`/`real_feature_idx_`). No scope creep.

## Authentication Gates

None.

## Issues Encountered

- The first per-row golden compare exposed the real-vs-packed indexing bug above (group membership + offsets already matched, which localized the divergence to the push path). Fixed via the `used_feature_map_` translation.

## User Setup Required

None. `cargo run -p xtask -- bin-capture` needs a C++ toolchain + CMake (present: cmake 3.28.3, g++ 13.3.0); normal `cargo test` replays the committed fixtures with no toolchain.

## Next Phase Readiness

- The determinism root is complete: numeric/categorical/missing binning (layers 1+2), bin storage layout, metadata, and now EFB grouping (layer 3) are all bit-identical to C++. Any later divergence is unambiguously NOT in the dataset/binning subsystem (SC#5).
- The bundled `FeatureGroup` layout (`bin_offsets_`, `num_total_bin_`, multi-val flag) and the real<->packed feature maps are the columnar contract Phase 4/5 histogram/split kernels inherit.

## Known Stubs

- The EFB golden's per-row bundled index is captured for SINGLE-VALUE bundled groups (the D-06 #4 EFB-win corpus collapses bundles to single-value groups, the common case). MultiValBin (dense/sparse) storage + its push path are implemented and unit-tested (Task 1), and `group_is_multi_val` is asserted in the golden; a dedicated per-row MULTI-VAL store golden is deferred to the histogram phase that first consumes a multi-val group (no in-scope corpus produces a multi-val group here). This is intentional and does not block DAT-05 (group membership + offsets + single-value per-row indices are bit-exact).

## Threat Flags

None — no new network endpoints, auth paths, or trust-boundary surface. The C++ EFB capture is read-only against the untracked `LightGBM/` tree (never `git add`ed), zero new packages (T-02-SC accept disposition), and the threat-register mitigations are honored: `num_total_bin_` accumulates in `u64` (T-02-14), index iteration is bounds-checked with no `unsafe` (T-02-15), and all randomness is `Random::new(num_data)` with stable sorts (T-02-16, caught by the golden).

---
*Phase: 02-dataset-binning-determinism-root*
*Completed: 2026-06-05*

## Self-Check: PASSED

Both created files verified on disk (`tests/efb_grouping.rs`, `tests/fixtures/efb_grouping.txt`) and the SUMMARY itself; all four plan commits (da9196b, f2f4263, 6d4a65d, 17dfa29) verified in git history.
