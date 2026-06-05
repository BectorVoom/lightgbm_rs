---
phase: 02-dataset-binning-determinism-root
plan: 06
subsystem: database
tags: [binning, dataset, ingest, pre-filter, determinism, parity, lightgbm]

# Dependency graph
requires:
  - phase: 02-dataset-binning-determinism-root
    provides: "BinMapper find_bin_numeric/need_filter kernel (02-01), FeatureGroup store + Dataset finish_load (02-02), from_mat/from_csr/from_csc ingest (02-04)"
provides:
  - "scaled_filter_cnt single-source-of-truth helper (exact C++ integer-truncation) shared by ingest.rs::build_mapper and bin_mapper.rs::find_bin_from_column"
  - "Default-config (feature_pre_filter=true, sample_cnt<num_rows) ingest parity golden + test — the first coverage of the default scaled-filter_cnt path"
  - "GAP-1 (CR-01/IN-02, SC#1, DAT-01) and GAP-2 (CR-02, SC#5/ORA-03 bin stage, DAT-07) closed"
affects: [predict, treelearner, histogram, gbdt, efb]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Single source-of-truth threshold helper: both former raw-forwarding sites route through scaled_filter_cnt (no divergent second copy)"
    - "Fails-before/passes-after gap-closure golden: the new parity test FAILS on the unfixed build_mapper and PASSES after the fix, proving the bug is real and reproducible"
    - "Golden carries the raw f32-representable matrix (single source of truth for the data) so both sides bin byte-identical input"

key-files:
  created:
    - crates/lgbm-dataset/tests/default_config_ingest_parity.rs
    - crates/lgbm-dataset/tests/fixtures/default_config_ingest.txt
  modified:
    - crates/lgbm-dataset/src/bin_mapper.rs
    - crates/lgbm-dataset/src/ingest.rs
    - xtask/cpp/bin_capture.cpp
    - xtask/src/main.rs

key-decisions:
  - "scaled_filter_cnt computes ((min_data_in_leaf*total_sample_cnt)/num_rows) in i64 (bit-identical to the C++ double-divide-then-truncate for in-range magnitudes); num_rows==0 returns raw min_data_in_leaf as a Rust-only, parity-unobservable empty-matrix guard with no C++ analog (cited dataset_loader.cpp:623-624)."
  - "CSR/CSC verified against c_api.cpp to use the IDENTICAL dense convention (total_sample_size=sample_cnt, num_dist_data=total_nrow); all three representations inherit the fix via finish_from_columns -> build_mapper."
  - "The golden's synthetic cells are generated as f32-representable values (cast through float) because from_mat takes &[f32] and widens to f64 at one site; otherwise the f32 round-trip drifts bin boundaries 1 ULP."
  - "Per-row STORED bins are read from feature_group().bin_data() (the bins the dataset actually stored), NOT recomputed via value_to_bin — recomputing would PASS before the fix (kernel unaffected) and silently defeat the gap closure."

patterns-established:
  - "Pattern 1: One filter_cnt derivation, two call sites — build_mapper + find_bin_from_column both route through scaled_filter_cnt."
  - "Pattern 2: New capture golden dispatched from a FIXED positional argv slot BEFORE the variadic example tail (kFirstExampleArgv bumped in lockstep) so existing goldens stay byte-identical."

requirements-completed: [DAT-01, DAT-07]

# Metrics
duration: 18min
completed: 2026-06-05
---

# Phase 2 Plan 06: Default-config scaled-filter_cnt gap closure Summary

**Fixed the default in-memory ingest path to feed the SCALED `filter_cnt = (min_data_in_leaf * sample_cnt) / num_rows` (one shared helper used by both former raw-forwarding sites) so `is_trivial_` / num_bin_ / stored per-row bins match C++ on the default `feature_pre_filter=true` configuration — proven by a new fails-before/passes-after parity golden.**

## Performance

- **Duration:** ~18 min
- **Started:** 2026-06-05T08:05:00Z
- **Completed:** 2026-06-05T08:23:00Z
- **Tasks:** 3
- **Files modified:** 6 (2 created + 4 modified)

## Accomplishments

- **GAP-1 closed (CR-01/IN-02, SC#1, DAT-01):** `ingest.rs::build_mapper` now passes the SCALED `filter_cnt` (via `scaled_filter_cnt`) to `find_bin_numeric` instead of the raw `cfg.min_data_in_leaf`; `bin_mapper.rs::find_bin_from_column` routes through the SAME helper. `from_csr`/`from_csc` inherit the fix through `finish_from_columns -> build_mapper`.
- **GAP-2 closed (CR-02, SC#5 / ORA-03 bin stage, DAT-07):** added `default_config_ingest_parity.rs` + `default_config_ingest.txt` — the first per-stage bin parity case covering the DEFAULT configuration (`feature_pre_filter=true`, `bin_construct_sample_cnt=50 < num_rows=200`, `min_data_in_leaf=20` -> `filter_cnt=5`). It FAILS before the fix and PASSES after.
- **Single source of truth:** `scaled_filter_cnt(min_data_in_leaf, total_sample_cnt, num_rows)` lives once in `bin_mapper.rs` next to `need_filter`; both divergent sites call it. No second raw-forwarding copy remains.
- **No existing golden mutated:** the new golden is dispatched from a FIXED positional argv slot (argv[8]) BEFORE the variadic example tail, with `kFirstExampleArgv` bumped 9->10 in lockstep, so every other committed fixture is byte-identical (example golden `COUNTS datasets=2` line unchanged).

## Task Commits

1. **Task 1: Capture the scaled-filter_cnt golden + add the failing parity test** - `02c279f` (test)
2. **Task 2: Fix filter_cnt derivation in a single source-of-truth helper** - `a5344f3` (fix)
3. **Task 3: Verify — parity passes, suite green, capture idempotent** - (verification only; results in this SUMMARY; metadata commit below)

## Files Created/Modified

- `crates/lgbm-dataset/tests/default_config_ingest_parity.rs` - Default `feature_pre_filter=true` ingest parity test; asserts `is_trivial_` (primary), num_bin_/bin_upper_bound_, default_bin_/most_freq_bin_ (non-trivial features), and STORED per-row bins from `feature_group().bin_data()` vs the scaled-filter_cnt C++ golden.
- `crates/lgbm-dataset/tests/fixtures/default_config_ingest.txt` - C++ golden captured through `ConstructFromSampleData` scaled-filter_cnt path; carries the raw f32-representable matrix (MATRIX block), per-feature metadata, and STORED per-row bins.
- `crates/lgbm-dataset/src/bin_mapper.rs` - NEW `scaled_filter_cnt` helper (exact C++ integer-truncation, num_rows==0 Rust-only guard); `find_bin_from_column` routes its threshold through it (param renamed `min_split_data` -> `min_data_in_leaf`); unit test `(20,50,200)==5`, `(20,200,200)==20`.
- `crates/lgbm-dataset/src/ingest.rs` - `build_mapper` computes & passes the scaled `filter_cnt` via the shared helper instead of raw `cfg.min_data_in_leaf`.
- `xtask/cpp/bin_capture.cpp` - NEW `EmitDefaultConfigIngest` emitter + `StoredBinSingleGroup` helper; dispatched from fixed argv[8]; `kFirstExampleArgv` 9->10.
- `xtask/src/main.rs` - `bin_capture()` inserts `default_config_ingest.txt` as a fixed arg before the example inputs; added to existence checks + done eprintln.

## The flip (evidence GAP-1/GAP-2 are real)

- Golden config: `num_rows=200`, `bin_construct_sample_cnt=50`, `min_data_in_leaf=20` -> `filter_cnt = (20*50)/200 = 5`.
- **Feature f1** is engineered (~75% value 1.0, ~25% value 9.0): its single split leaves 10 of the 50 sampled rows on the minority side. At `filter_cnt=5` both sides (40, 10) clear the threshold -> **non-trivial**; at the raw threshold 20 the minority side (10 < 20) fails -> **trivial**. So `is_trivial_` flips on the threshold value.
- Controls in the same golden: f0 (continuous, non-trivial both ways), f2 (near-constant, trivial both ways), f3 (balanced two-cluster, non-trivial both ways) — the golden contains BOTH trivial and non-trivial features.

### Fails-before assertion text (Task 1, against unfixed build_mapper)
```
feature 1: is_trivial_ true != golden false (scaled-filter_cnt divergence)
```
The current code (raw `min_data_in_leaf=20`) marked f1 trivial; the golden (scaled `filter_cnt=5`) says non-trivial.

### Passes-after (Task 3, after the fix)
```
test default_config_ingest_matches_cpp ... ok
test result: ok. 1 passed; 0 failed
```

## SC re-affirmation

- **SC#1 (is_trivial_ matches C++ on the default in-memory ingest path):** RE-AFFIRMED. With the scaled `filter_cnt`, `is_trivial_` / num_bin_ / per-row STORED bin assignment now match C++ for every feature in the default-config golden, including the engineered flip feature f1. DAT-01 moves PARTIAL -> SATISFIED for the default-config divergence.
- **SC#5 / ORA-03 bin stage (per-stage parity covers the DEFAULT configuration):** RE-AFFIRMED. The new `default_config_ingest_parity.rs` is the first per-stage bin parity case on the default `feature_pre_filter=true` path; combined with the existing pre_filter=false suites it localizes any divergence to the binning stage. DAT-07 moves PARTIAL -> SATISFIED for the default-config divergence.

## CSR/CSC convention finding

Verified against `LightGBM/src/c_api.cpp`: the dense (`:1368-1374`), CSR (`:1445-1452`), and CSC (`:1514-1521`) paths all call `ConstructFromSampleData(..., sample_cnt, nrow, nrow)`, i.e. `total_sample_size = sample_cnt` and `num_dist_data = total_nrow` — the IDENTICAL convention. The Rust `from_csr`/`from_csc` funnel through `finish_from_columns -> build_mapper`, so the single `scaled_filter_cnt` fix covers all three representations with no per-representation special-casing.

## Decisions Made

See `key-decisions` frontmatter. Headline: one `scaled_filter_cnt` helper (i64 integer-truncation = C++ double-divide-then-truncate for in-range magnitudes); num_rows==0 guard documented as Rust-only/parity-unobservable; STORED bins (not recomputed value_to_bin) used as the per-row discriminator.

## Deviations from Plan

None - plan executed exactly as written. (The only judgment call beyond the plan text: generating the golden's synthetic cells as f32-representable values so the `from_mat` f32->f64 widen does not drift bin boundaries — this is consistent with the plan's "EMIT THE RAW MATRIX ... both sides operate on byte-identical data" instruction and is documented in `key-decisions`.)

## Issues Encountered

- First capture used raw f64 cells; the parity test then failed on `bin_upper_bound_` (a 1-ULP drift) instead of `is_trivial_`, because `from_mat` widens f32->f64 while the C++ matrix was true f64. Resolved by generating every cell as an f32-representable value (cast through `float`) in the C++ emitter, so both sides bin byte-identical input — the test then failed on the intended `is_trivial_` signal.
- First flip-feature minority fraction (~15%) yielded a sampled minority of 4 (< 5), so f1 was trivial even at filter_cnt=5 (no flip). Raised to ~25% (sampled minority 10), which clears 5 but not 20 — a clean flip with margin on both sides.

## Known Stubs

None — no placeholder/empty-value stubs introduced.

## Threat Flags

None — no new network/auth/file-access/schema surface introduced.

## User Setup Required

None - no external service configuration required. (The `bin-capture` regeneration needs a local C++ toolchain + the untracked `LightGBM/` reference tree, which is present here; normal `cargo test` reads the committed golden.)

## Next Phase Readiness

- The determinism root's default-configuration binning contract is restored and proven; both phase-02 verification gaps are closed. Phase 02 can be re-verified.
- `cargo test --workspace` fully green (lgbm-dataset 75 lib + all integration suites; core; oracle-harness; xtask — 0 failed). `cargo run -p xtask -- bin-capture` is idempotent (committed `default_config_ingest.txt` byte-identical on re-run; no other golden mutated).

## Self-Check: PASSED

- FOUND: crates/lgbm-dataset/tests/default_config_ingest_parity.rs
- FOUND: crates/lgbm-dataset/tests/fixtures/default_config_ingest.txt
- FOUND: .planning/phases/02-dataset-binning-determinism-root/02-06-SUMMARY.md
- FOUND commit 02c279f (Task 1, test)
- FOUND commit a5344f3 (Task 2, fix)

---
*Phase: 02-dataset-binning-determinism-root*
*Completed: 2026-06-05*
