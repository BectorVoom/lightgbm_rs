---
phase: 02-dataset-binning-determinism-root
plan: 03
subsystem: dataset-binning
tags: [binning, bin-mapper, categorical, category-folding, sortforpair, missing-value, nan-routing, signed-zero, golden-replay, bit-exact, rust]

# Dependency graph
requires:
  - phase: 02-dataset-binning-determinism-root
    plan: 01
    provides: "Numeric BinMapper (find_bin_numeric/value_to_bin), BinType/MissingType, need_filter, arg_max, check_double_equal_ordered, bin-capture xtask harness + exact comparators"
  - phase: 02-dataset-binning-determinism-root
    plan: 02
    provides: "Bin storage layer (unchanged; categorical mapper feeds the same FeatureGroup/Bin seam)"
provides:
  - "Categorical BinMapper::find_bin_categorical (descending-count fold, f32 0.99 cut, min_data_in_bin fold-break, NaN dummy bin 0)"
  - "categorical_2_bin_ / bin_2_categorical_ on BinMapper + categorical value_to_bin path"
  - "Completed MissingType routing (None/Zero/NaN) proven across use_missing/zero_as_missing sweeps + signed zeros + all-missing"
  - "bin_capture.cpp categorical FindBin + missing-edge corpus emission (SortForPairDesc/RoundInt/NeedFilterCat)"
  - "Committed categorical (layers 1+3 + per-row) + missing (layer 1 + per-row) golden fixtures + replay tests"
affects: [02-04 ingestion, 02-05 EFB/MultiValBin, predict/histogram phases]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Stable descending-count fold transcribing Common::SortForPair(is_reverse=true) via slice::sort_by (ties keep ascending-value input order)"
    - "f32-literal 0.99 cut + RoundInt((x)+0.5f) — float arithmetic chain preserved (no 0.99_f64) for bit-exact categorical bin count"
    - "Plain HashMap for categorical_2_bin_ (by-key lookup; iteration order irrelevant, fold order is sort-driven)"
    - "Order-independent map golden compare (HashMap == HashMap) + sorted-key C++ dump for diff-friendly fixtures"

key-files:
  created:
    - crates/lgbm-dataset/tests/categorical_folding.rs
    - crates/lgbm-dataset/tests/missing_edge_cases.rs
    - crates/lgbm-dataset/tests/fixtures/categorical_folding.txt
    - crates/lgbm-dataset/tests/fixtures/missing_edge_cases.txt
  modified:
    - crates/lgbm-dataset/src/bin_mapper.rs
    - crates/lgbm-dataset/src/dataset.rs
    - crates/lgbm-dataset/src/feature_group.rs
    - xtask/cpp/bin_capture.cpp
    - xtask/src/main.rs

decisions:
  - "Categorical fold transcribes SortForPair(is_reverse=true) with the STABLE slice::sort_by (not sort_unstable) so equal-count ties keep ascending-value input order — bit-identical to C++ std::stable_sort"
  - "The 0.99 cut keeps the FLOAT chain: rest as f32 * 0.99_f32, then (scaled as f64 + 0.5f32 as f64) as i32 — mirroring (size_t*float)->float then RoundInt(double x)=(int)(x+0.5f); a 0.99_f64 would shift the cut count"
  - "categorical_2_bin_ uses std HashMap (by-key lookups; the fold ORDER is sort-driven so map iteration order is irrelevant) — RESEARCH State-of-the-Art; goldens compared order-independently"
  - "Categorical/missing goldens compared bit-exact (map equality + compare_exact_u32 per-row), never the ~1e-6 oracle tolerance"

metrics:
  duration: 9min
  completed: 2026-06-05
---

# Phase 2 Plan 03: Categorical Folding + Missing-Value Routing Summary

**The categorical `BinMapper::FindBin` branch (stable descending-count fold, f32 `0.99` cut, `min_data_in_bin` fold-break, NaN dummy bin 0) + categorical `value_to_bin` + completed `MissingType` routing, proven bit-identical to C++ across a 6-case categorical corpus (layers 1+3 + per-row index) and an 8-case missing-edge battery (layer 1 + per-row index).**

## Performance

- **Duration:** ~9 min
- **Started:** 2026-06-05T06:49:50Z
- **Completed:** 2026-06-05T06:58:28Z
- **Tasks:** 2
- **Files created:** 4 / **modified:** 5

## Accomplishments

- Added `categorical_2_bin_` (`HashMap<i32, u32>`) and `bin_2_categorical_` (`Vec<i32>`) to `BinMapper`, and wired the categorical `value_to_bin` path (`bin.h:638-649`): `int_value < 0 -> 0`, by-key map lookup, unknown `-> 0`.
- Transcribed `find_bin_categorical` line-for-line from `bin.cpp:410-476`: int conversion (negatives accumulate into `na_cnt`, adjacent-equal merge), the STABLE descending-count fold (`SortForPair(is_reverse=true)` via `sort_by`, ties keep ascending-value order), the `cut_cnt = RoundInt((total-na) * 0.99f)` float-chain cut, the `counts < min_data_in_bin && cur_cat_idx > 1` fold-break, the NaN dummy bin 0 (`bin_2_categorical_[0] == -1`), and the `None`-vs-`NaN` `missing_type_` resolution.
- Confirmed the completed numeric `MissingType` derivation (None/Zero/NaN, the `Zero` size-2 reset, signed-zero placement) is exact — no further numeric change needed; proven by the missing battery.
- Extended `bin_capture.cpp` with the categorical `BinMapper`/`FindBinCategorical` + `SortForPairDesc`/`RoundInt`/`NeedFilterCat` (verbatim from the pinned headers) and emitted two new corpora; wired `xtask bin-capture` to write `categorical_folding.txt` + `missing_edge_cases.txt` (5-arg argv).
- Added `categorical_folding.rs` (asserts `categorical_2_bin_` map + `bin_2_categorical_` + layer-1 internals + per-row `value_to_bin` bit-exact, with fold-break and negative coverage guards) and `missing_edge_cases.rs` (layer-1 `missing_type`/`num_bin`/`default_bin` + per-row routing across `use_missing`/`zero_as_missing`, with NaN/zero-as-missing/signed-zero/all-missing coverage guards). Both pass NOT skipped; regen is idempotent (empty `git diff`).

## Task Commits

1. **Task 1: Categorical FindBin folding + categorical ValueToBin + MissingType** — `c4bdee6` (feat)
2. **Task 2: Categorical + missing golden capture and replay (layer 1+3 + per-row)** — `850e86d` (test)

_Task 1 is TDD: implementation + 7 inline behavior tests (descending-count fold, tie-stability, 0.99 cut, min_data_in_bin fold-break, NaN/negative/unknown -> bin 0, negatives-as-NaN, all-consumed -> None) landed in one GREEN commit._

## Files Created/Modified

- `crates/lgbm-dataset/src/bin_mapper.rs` — `categorical_2_bin_`/`bin_2_categorical_` fields, `find_bin_categorical`, categorical `value_to_bin`; 7 inline categorical tests.
- `crates/lgbm-dataset/src/dataset.rs`, `src/feature_group.rs` — test-helper `BinMapper` literals updated with the two new fields.
- `crates/lgbm-dataset/tests/categorical_folding.rs` — layer 1+3 + per-row categorical golden replay (map equality + `compare_exact_u32`).
- `crates/lgbm-dataset/tests/missing_edge_cases.rs` — layer 1 + per-row missing-routing golden replay across config sweeps.
- `crates/lgbm-dataset/tests/fixtures/{categorical_folding,missing_edge_cases}.txt` — committed C++ goldens (6 categorical, 8 missing cases).
- `xtask/cpp/bin_capture.cpp` — categorical `BinMapper`/`FindBinCategorical` + `SortForPairDesc`/`RoundInt`/`NeedFilterCat` + categorical/missing corpus emitters; `main` argv extended to 5.
- `xtask/src/main.rs` — passes the two new fixture paths; existence-check loop.

## Decisions Made

- **Stable descending-count fold.** `Common::SortForPair(is_reverse=true)` is `std::stable_sort` on `(count, value)` pairs by `count` descending; equal counts keep input (ascending-value) order. Mirrored with `slice::sort_by` (stable). Using `sort_unstable` would silently re-order ties and diverge — the `cat_basic` golden (`2` and `9` tie at count 5, `2 -> bin 2`, `9 -> bin 3`) is the bit-exact witness.
- **f32 `0.99` cut chain preserved.** `cut_cnt = RoundInt((total_sample_cnt - na_cnt) * 0.99f)`: in C++ the `size_t * float` converts the integer operand to FLOAT (multiply in f32), then `RoundInt(double x) = (int)(x + 0.5f)`. Transcribed as `(rest as f32 * 0.99_f32) as f64 + 0.5_f32 as f64) as i32` — a `0.99_f64` literal would change the cut count and the folded bin set.
- **`HashMap` for `categorical_2_bin_`.** Lookups are by key and the fold ORDER is sort-driven, so map iteration order is irrelevant (RESEARCH State-of-the-Art). The golden compares the map order-independently (HashMap equality) while the C++ dump is sorted-by-key for a diff-friendly fixture.
- **Bit-exact comparison** for both corpora: HashMap/Vec equality for the maps, `compare_exact_u32` for the per-row index — never the `~1e-6` oracle tolerance.

## Deviations from Plan

None — plan executed exactly as written. The categorical branch, categorical `value_to_bin`, and `MissingType` completion are all in `bin_mapper.rs`; the harness extension and both golden replay tests match the plan's artifacts and acceptance criteria.

The plan's Task-1 hint to "finish any remaining numeric `MissingType` derivation edge ... if not already complete in Plan 01" was a no-op: Plan 01 already implemented the full numeric derivation (the `Zero` size-2 reset, signed-zero pseudo placement). The missing-edge golden battery proves it exact, so no numeric change was required.

## Capture-harness note (external_libs unavailable — continued from Plans 01/02)

Consistent with Plans 02-01/02-02: `bin.cpp` is unbuildable here (its includes transitively pull `external_libs/{fast_double_parser,fmt}`, which are empty/unvendored). `bin_capture.cpp` therefore **verbatim-transcribes** the categorical `FindBin`/`ValueToBin` + `SortForPair`/`RoundInt`/`NeedFilter` from the pinned `bin.cpp`/`bin.h`/`common.h` (commit 195c26fc, 4.6.0.99) — these depend only on `std`, so the emitted goldens are byte-identical to lib_lightgbm. Sampling is not used here (categorical/missing corpora build over the full column, sample_cnt = num_rows). Regen is idempotent.

## Issues Encountered

- `cargo clippy` flags style lints on `bin_mapper.rs` (loop-index transcription, `len() >= 1`, and a `sort_by` -> `sort_by_key(Reverse)` suggestion). These are pre-existing transcription-fidelity choices (Plans 01/02) plus one new `sort_by` kept as the clearest 1:1 mirror of C++ `SortForPair`'s `a.first > b.first` comparator (stable, functionally identical to the suggested `Reverse`). No behavior impact; the workspace builds and tests clean. Not "fixed" per the scope boundary (do not churn established patterns).

## User Setup Required

None. `cargo run -p xtask -- bin-capture` needs a C++ toolchain + CMake (already present); normal `cargo test` replays the committed fixtures with no toolchain.

## Next Phase Readiness

- Every feature type now bins bit-identically to C++: numeric (Plan 01), storage (Plan 02), and categorical + missing routing (this plan) — a later divergence is localized, not in binning (SC#3, SC#5).
- The categorical `BinMapper` plugs into the same `FeatureGroup`/`Bin` storage seam (Plan 02) unchanged; Plans 02-04 (ingestion `from_mat`/`from_csr`/`from_csc`) and 02-05 (EFB/MultiValBin) build on top.
- EFB grouping, `MultiValBin`, metadata, and the public ingestion API remain intentionally unimplemented (Plans 04/05).

## Known Stubs

None. The Plan 01/02 categorical `value_to_bin` stub is now fully wired (category map lookup), and `find_bin_categorical` is a complete, golden-proven implementation.

---
*Phase: 02-dataset-binning-determinism-root*
*Completed: 2026-06-05*

## Self-Check: PASSED

All 4 created key files + SUMMARY verified present on disk; both task commits (c4bdee6, 850e86d) verified in git history. `cargo test --workspace` green; `cargo run -p xtask -- bin-capture` idempotent.
