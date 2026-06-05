---
phase: 02-dataset-binning-determinism-root
plan: 07
subsystem: database
tags: [lightgbm, binning, dataset, efb, construct, determinism, parity, golden]

# Dependency graph
requires:
  - phase: 02-dataset-binning-determinism-root (02-04 ingestion, 02-05 EFB, 02-06 scaled filter_cnt)
    provides: from_mat/from_csr/from_csc ingest, construct_bundled/fast_feature_bundling, scaled_filter_cnt helper
provides:
  - "Default ingest path (from_mat/from_csr/from_csc) now routes through the single C++ Dataset::Construct (construct_bundled enable_bundle dispatch): trivial features DROPPED (used_feature_map_[real]=-1), bundling runs"
  - "Non-bundled Dataset::construct mirrors C++ trivial-feature filtering (used_features = !is_trivial_ only)"
  - "EfbSamples built on the ingest path to the exact c_api.cpp:1352-1374 sampled-set convention"
  - "default_config_ingest_parity hardened: trivial-exclusion + per-non-trivial feature_to_group/feature_to_subfeature parity vs C++-Construct golden + bit-exact stored bins; panic on missing golden"
affects: [03-predict, histogram, split, any phase reading FinishedDataset grouping]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Default ingest unifies onto the single faithful Construct (construct_bundled) instead of a divergent store-everything non-bundled path"
    - "EfbSamples sampled-set convention (positions 0..sample_cnt, non-zero/NaN filter, total_sample_cnt=sample_cnt) reusing the single create_sample_indices draw"
    - "Golden emitter computes grouping via the in-file verbatim FastFeatureBundling transcription (no LightGBM::Dataset instantiation, no lib_lightgbm link)"

key-files:
  created: []
  modified:
    - crates/lgbm-dataset/src/dataset.rs
    - crates/lgbm-dataset/src/ingest.rs
    - xtask/cpp/bin_capture.cpp
    - crates/lgbm-dataset/tests/fixtures/default_config_ingest.txt
    - crates/lgbm-dataset/tests/default_config_ingest_parity.rs

key-decisions:
  - "Route ingest through construct_bundled (the faithful single-Construct port), not a second divergent construct — C++ is a SINGLE Dataset::Construct that filters trivial features (337-343) before the enable_bundle branch (362-369)"
  - "EfbSamples on the ingest path pinned to c_api.cpp:1352-1374 sampled-set convention (NOT the efb_grouping.rs full-row convention)"
  - "Golden emitter must pass is_sparse=true (config.h default is_enable_sparse=true) to FastFeatureBundling so its grouping/shuffle matches the Rust default ingest path"

patterns-established:
  - "Determinism-root parity tests assert the GROUPING seam (feature2group_/feature2subfeature_), not just stored bins, so a wrong EfbSamples fails loudly"
  - "Committed goldens are a HARD dependency: missing golden -> panic, never a silent SKIP (WR-01)"

requirements-completed: [DAT-01, DAT-02, DAT-05, DAT-07, ORA-03]

# Metrics
duration: ~35min
completed: 2026-06-05
---

# Phase 2 Plan 07: Default-ingest Construct parity (CR-01 + WR-01 closure) Summary

**Default in-memory ingest (from_mat/from_csr/from_csc) now routes through the single faithful C++ Dataset::Construct (construct_bundled) — trivial features are dropped (used_feature_map_[real]=-1) and EFB bundling runs with an EfbSamples built to the exact c_api.cpp:1352-1374 sampled-set convention — with a parity test that asserts trivial-exclusion AND per-non-trivial group/subfeature parity vs a C++-Construct golden.**

## Performance

- **Duration:** ~35 min
- **Started:** 2026-06-05 (sequential executor)
- **Completed:** 2026-06-05
- **Tasks:** 3
- **Files modified:** 5

## Accomplishments

- **CR-01 (BLOCKER) closed:** the non-bundled `Dataset::construct` now mirrors C++ `Dataset::Construct` trivial-feature filtering (`used_features = !is_trivial_` only; `used_feature_map_[real] = -1` for trivial features; groups built over used features only). The default ingest path (`finish_from_columns`) no longer hard-codes the store-everything non-bundled construct — it dispatches on `cfg.enable_bundle` through `construct_bundled` (the faithful single-Construct port), so a trivial feature is dropped and bundling runs exactly as the single C++ `Dataset::Construct` (dataset.cpp:325-441).
- **EFB parity hole closed:** the parity test asserts per-non-trivial `feature_to_group`/`feature_to_subfeature` against the C++-Construct golden, so an incorrectly-built EfbSamples can no longer pass silently.
- **CR-01 masking closed:** the golden emitter now models C++ Construct (trivial f2 dropped — no ASSIGN, no group/subfeature; per-non-trivial group/subfeature recorded via the in-file verbatim FastFeatureBundling transcription).
- **WR-01 hardened:** a missing committed golden is now a hard `panic`, never a silent SKIP.
- `cargo test --workspace` fully green (0 failed); `ingest_equivalence` + `example_dataset_parity` unaffected (their fixtures have no trivial features).

## Task Commits

1. **Task 1: construct filters trivial features + ingest routes through enable_bundle dispatch** - `1854704` (fix)
2. **Task 2: golden emitter models C++ Construct (drop trivial, record group/subfeature) + regenerate golden** - `d5d57b6` (test)
3. **Task 3: parity test asserts trivial-exclusion + per-non-trivial group/subfeature parity; fix emitter is_sparse default** - `723c27f` (test)

## Files Created/Modified

- `crates/lgbm-dataset/src/dataset.rs` - `Dataset::construct` rewritten to mirror C++ Construct trivial-feature filtering (used_features = !is_trivial_; used_feature_map_[real] = -1; groups over used features only; num_features_ = used count). `push_value`/`finish_load`/`FinishedDataset` unchanged.
- `crates/lgbm-dataset/src/ingest.rs` - `build_mapper` now returns the sampled per-row values (same single create_sample_indices draw — no second RNG draw); new `build_efb_samples` builds the EfbSamples to the c_api.cpp:1352-1374 sampled-set convention; `finish_from_columns` dispatches through `construct_bundled`. CSR/CSC inherit automatically.
- `xtask/cpp/bin_capture.cpp` - `EmitDefaultConfigIngest` gates ASSIGN on `!bm.is_trivial_`, computes feature2group_/feature2subfeature_ via the in-file verbatim FastFeatureBundling transcription over the !is_trivial_ used-feature set (fed sampled-set-convention inputs, `is_sparse=true`), records group=/subfeature= per non-trivial feature; StoredBinSingleGroup comment re-cited to dataset.cpp:337-343.
- `crates/lgbm-dataset/tests/fixtures/default_config_ingest.txt` - regenerated C++-Construct golden via REAL capture: trivial f2 has no ASSIGN/no group fields; non-trivial f0/f1/f3 carry group=/subfeature=; f1 flip retained.
- `crates/lgbm-dataset/tests/default_config_ingest_parity.rs` - parser enforces representation agreement (trivial -> no group/ASSIGN; non-trivial -> group/subfeature + one ASSIGN); WR-01 panic on missing golden; per-feature loop asserts feature_to_group==-1 for trivial + group/subfeature parity for non-trivial before bit-exact stored-bin compare; num_groups + num_features cross-checks.

## Decisions Made

- **Construct-dispatch decision (justified against dataset.cpp:325-441):** C++ is a SINGLE `Dataset::Construct` that filters trivial features (337-343, `used_features` = only `!is_trivial()` mappers) BEFORE the `enable_bundle` branch (362-369). The in-repo `construct_bundled` is already a faithful port of that whole function. So ingest was unified onto `construct_bundled` (one path mirroring the single C++ Construct) rather than maintaining a second divergent `construct`. The non-bundled `construct` was ALSO fixed to filter trivial features so it is never a future divergence trap.
- **EfbSamples convention (c_api.cpp:1352-1374):** `total_sample_cnt = sample_cnt` (the SAMPLED-SET size, i.e. `create_sample_indices` length / clamped `bin_construct_sample_cnt`), NOT `num_rows`; `num_sample_col = num_cols`; each `sample_indices[k]` holds sampled-set POSITIONS `0..sample_cnt` (never raw row ids); each kept entry passed the `|v| > kZeroThreshold || v.is_nan()` filter (c_api.cpp:1361); `num_per_col[k] == sample_values[k].len()`. **The efb_grouping.rs FULL-ROW convention (total_sample_cnt = num_rows, raw row indices) was explicitly NOT copied** — confirmed by inspection; `build_efb_samples` documents the distinction.
- **No second RNG draw for EfbSamples:** `build_mapper` returns the sampled values from its existing single `create_sample_indices` draw; `finish_from_columns` reuses them to build the EfbSamples. No new `Random` is constructed for sampling. Confirmed by inspection.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Golden emitter passed is_sparse=false, diverging the grouping/shuffle from the Rust default ingest path**
- **Found during:** Task 3 (running the parity test against the Task-2 golden)
- **Issue:** The Task-2 emitter called `FastFeatureBundling(..., is_sparse=false)`. The default ingest Config has `is_enable_sparse=true` (config.h default; dataset.cpp:352 sets `is_sparse = io_config.is_enable_sparse`), and `Dataset::construct_bundled` passes that `true` to `fast_feature_bundling`, which runs the dense/sparse SECOND PASS in `find_groups`. With `is_sparse=false` the emitter took a different grouping branch, producing a different post-shuffle group order: the test failed with `feature 1: feature_to_group 0 != C++-Construct golden group 2`.
- **Fix:** Changed the emitter's `FastFeatureBundling` call to `is_sparse=true` (the default), matching the ingest path; regenerated the golden (only f1/f3 group ids corrected, f0 unchanged).
- **Files modified:** xtask/cpp/bin_capture.cpp, crates/lgbm-dataset/tests/fixtures/default_config_ingest.txt
- **Verification:** parity test passes after the fix; capture idempotent; only `default_config_ingest.txt` changed in fixtures.
- **Committed in:** `723c27f` (Task 3 commit)

**2. [Rule 1 - Bug] Parity test compared ds.num_features() against the TOTAL golden feature count**
- **Found during:** Task 3 (the CR-01 fix drops trivial features, so `ds.num_features()` is the USED count, 3, not the total 4)
- **Issue:** The pre-existing `assert_eq!(ds.num_features(), g.num_features)` was correct for the store-everything path but wrong after CR-01 (trivial features dropped).
- **Fix:** Assert `ds.num_features()` equals the count of NON-trivial golden features (`used_count`). This is itself a CR-01 discriminator (the unfixed path gives `num_features == 4` and fails).
- **Files modified:** crates/lgbm-dataset/tests/default_config_ingest_parity.rs
- **Verification:** passes after fix; fails-before evidence (below) fires precisely on this assertion.
- **Committed in:** `723c27f` (Task 3 commit)

---

**Total deviations:** 2 auto-fixed (2 Rule-1 bugs). **Impact:** both necessary for correctness/parity; no scope creep. The is_sparse fix is load-bearing for grouping parity at the determinism root.

## HARD fails-before / passes-after (mandatory evidence)

The parity test was run against the PRE-Task-1 construct by temporarily replacing the trivial-feature filter in `construct_bundled` with `used_features = (0..num_total_features).collect()` (reproducing the pre-fix store-everything divergence), then reverted verbatim.

**FAILS-BEFORE (verbatim):**
```
thread 'default_config_ingest_matches_cpp' panicked at crates/lgbm-dataset/tests/default_config_ingest_parity.rs:291:5:
assertion `left == right` failed: ds.num_features 4 != non-trivial golden feature count 3 (CR-01: trivial features must be dropped)
  left: 4
 right: 3
test result: FAILED. 0 passed; 1 failed; 0 ignored; 0 measured; 0 filtered out
```
This CR-01 assertion fires first; had it been bypassed, the per-feature loop's `feature_to_group(2) == -1` trivial-exclusion assertion would fire next (the unfixed construct stores f2, giving `feature_to_group(2) != -1`).

**PASSES-AFTER (verbatim):**
```
test default_config_ingest_matches_cpp ... ok
test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out
```

## Golden changes + capture

- Capture was REAL (`cargo run -p xtask -- bin-capture`, header-only `bin_capture` cmake target compiled against `-I LightGBM/include`; NOT lib_lightgbm). Idempotent (re-run produced a byte-identical golden).
- Golden line changes vs the prior (02-06) golden: `ASSIGN f=2` REMOVED (trivial f2 dropped); `group=`/`subfeature=` ADDED to each non-trivial FEATURE line (f0=group1, f1=group0, f3=group2 after the is_sparse=true correction); `FEATURE f=1 is_trivial=0` flip PRESERVED. No other committed golden changed (`example_dataset_binning.txt` `COUNTS datasets=2` byte-unchanged; `EmitExampleDataset` count unchanged at 2).

## SC re-affirmation

- **SC#2 re-affirmed:** the immutable columnar store is bit-identical to C++ Construct on the default config — trivial features dropped, grouping verified (feature2group_/feature2subfeature_ per non-trivial feature) — at the determinism root.
- **SC#5 / ORA-03 bin stage re-affirmed:** the per-stage bin parity test now localizes the trivial-feature/enable_bundle/EFB-grouping divergence against C++ Construct.
- **DAT-07 + ORA-03** move from BLOCKED to SATISFIED; **DAT-01/DAT-02/DAT-05** from PARTIAL/path-isolated to satisfied (the default ingest path now exercises the correct filter+bundle Construct).
- `ingest_equivalence` and `example_dataset_parity` (pre_filter=false) remained green — their fixtures have no trivial features, so the construct fix (which only changes which features are stored/grouped when a feature is trivial) does not affect them.

## Deferred items (explicitly out of scope this plan)

- **WR-02** (build_mapper duplication), **WR-03** (forced bins), **IN-01..IN-03** — not verification gaps; not folded in.
- **WR-04** (numeric `need_filter` `saturating_sub` symmetry) — the one-line symmetry lives in `bin_mapper.rs`, which is NOT in this plan's `files_modified` and was not otherwise touched here. Per repo constraints it is left as deferred (not applied) to avoid expanding scope into an unmodified file.

## Issues Encountered

- The EFB shuffle (`Random(num_data)`) produces a group permutation; an early test run showed a consistent group mismatch (f1/f3 swapped) that traced to the emitter's `is_sparse=false` vs the ingest default `is_enable_sparse=true` — resolved by Rule-1 deviation #1 above.

## Next Phase Readiness

- The dataset determinism root now reproduces C++ `Dataset::Construct` on the DEFAULT ingest configuration (trivial features dropped, bundling run, grouping verified). Downstream phases (predict, histogram, split) read a `FinishedDataset` whose `num_features_`/`feature2group_`/`feature2subfeature_`/stored bins are bit-identical to C++ whenever a feature is trivial.
- Re-run `/gsd-verify-phase 02` — expect the 3 remaining gaps (DAT-07, ORA-03, and the DAT-01/02/05 path-isolation) to flip to PASS.

## Self-Check: PASSED

All 5 modified files present on disk; all 3 task commits (`1854704`, `d5d57b6`, `723c27f`) present in git history.

---
*Phase: 02-dataset-binning-determinism-root*
*Completed: 2026-06-05*
