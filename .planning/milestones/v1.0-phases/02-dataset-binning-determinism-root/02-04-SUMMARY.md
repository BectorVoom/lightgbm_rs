---
phase: 02-dataset-binning-determinism-root
plan: 04
subsystem: dataset-binning
tags: [ingestion, from_mat, from_csr, from_csc, metadata, query-weights, cross-representation-equivalence, example-dataset-parity, sampling, boundary-validation, golden-replay, bit-exact, rust]

# Dependency graph
requires:
  - phase: 02-dataset-binning-determinism-root
    plan: 01
    provides: "Numeric BinMapper (find_bin_numeric/value_to_bin), sample_count/create_sample_indices, DatasetError, bin-capture xtask harness + exact comparators"
  - phase: 02-dataset-binning-determinism-root
    plan: 02
    provides: "Dataset::construct/push_value/finish_load immutability boundary, FeatureGroup, Bin storage layer"
  - phase: 02-dataset-binning-determinism-root
    plan: 03
    provides: "Categorical BinMapper + completed MissingType routing (unchanged; ingestion routes numeric for the MVP example data)"
provides:
  - "Metadata (label/weights/query_weights f32, init_score f64, query_boundaries i32) + finish_load query-weight derivation (CalculateQueryWeights)"
  - "from_mat / from_csr / from_csc internal ingestion API (D-05) — validated entry points wiring sample -> find_bin -> construct -> push -> finish_load"
  - "Cross-representation equivalence proof (dense == CSR == CSC, bit-identical, incl. a zero-heavy column)"
  - "End-to-end example-dataset binning parity (regression + binary_classification, layers 1+2, 28 features each) bit-identical to C++"
  - "bin_capture.cpp metadata + example-dataset golden emitters (F32Bits, TSV loader) + committed example fixtures"
affects: [02-05 EFB/MultiValBin, predict phase, histogram phase]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Validated-entry-point ingestion (config/set.rs discipline): validate ALL caller input at the boundary FIRST (Security V5), then a fixed-order pipeline; never panic on caller input"
    - "Single documented f32->f64 widen site (widen()) — caller f32 buffers cross into f64 binning arithmetic at exactly one auditable point, no scattered `as f64`"
    - "Dense-by-column sparse gather (CSR/CSC): absent entry == 0.0 (Open Q2 zero_cnt-aware), proven bin-equivalent to the dense form"
    - "Cross-representation equivalence as a tolerance-free internal invariant (dense/CSR/CSC describe identical data -> must bin to the last bit), no C++ golden needed for the equivalence itself"
    - "label_t==f32 query-weight mean (accumulate in f32, divide by int group size -> converts to f32) transcribed verbatim from CalculateQueryWeights"

key-files:
  created:
    - crates/lgbm-dataset/src/metadata.rs
    - crates/lgbm-dataset/src/ingest.rs
    - crates/lgbm-dataset/tests/metadata.rs
    - crates/lgbm-dataset/tests/ingest_equivalence.rs
    - crates/lgbm-dataset/tests/example_dataset_parity.rs
    - crates/lgbm-dataset/tests/fixtures/metadata.txt
    - crates/lgbm-dataset/tests/fixtures/example_dataset_binning.txt
    - crates/lgbm-dataset/tests/fixtures/examples/regression.train
    - crates/lgbm-dataset/tests/fixtures/examples/binary.train
  modified:
    - crates/lgbm-dataset/src/lib.rs
    - xtask/src/main.rs
    - xtask/cpp/bin_capture.cpp

decisions:
  - "Metadata.label/weights/query_weights are Vec<f32> (LabelT), init_score is Vec<f64>, query_boundaries is Vec<i32> (DataSizeT) — the C++ types.rs contract; widening labels/weights to f64 would break parity (over-precision)"
  - "Query weights derived in f32 arithmetic (accumulate in f32, divide by the int group size which converts to f32) — bit-identical to CalculateQueryWeights; the groups_frac_weights golden is the f32-rounding witness"
  - "Single f32->f64 widen site (widen()): every caller value flows through one auditable point; grep confirms no scattered `as f64` in ingest.rs"
  - "Sparse gather is dense-by-column with absent entry == 0.0 (Open Q2 zero_cnt-aware); cross-representation equivalence proves dense/CSR/CSC bin identically incl. a zero-heavy column"
  - "Example parity capped to 500 rows (kExampleMaxRows shared C++/Rust constant) for a manageable golden; the per-feature mapper is built on the f64 columns directly (matching C++), so f32-narrowing in from_mat does not pollute the bit-exact comparison"
  - "Example fixtures COPIED into the committed tests/fixtures/examples/ and read from there at test time — never the untracked LightGBM/ tree (project memory: lightgbm-ref-tree-untracked)"

metrics:
  duration: 35min
  completed: 2026-06-05
---

# Phase 2 Plan 04: from_mat/from_csr/from_csc Ingestion + Metadata Summary

**The minimal internal ingestion API (`from_mat`/`from_csr`/`from_csc`, D-05) + `Metadata` query-weight derivation, wiring the full sample -> find_bin -> construct -> push -> finish_load pipeline; proven by (a) dense vs CSR vs CSC binning bit-identically incl. a zero-heavy column, (b) metadata query weights round-tripping bit-exact with C++ `CalculateQueryWeights`, and (c) end-to-end binning of two real LightGBM example datasets (regression + binary_classification, 28 features each) bit-identical to C++ for every feature (layers 1+2).**

## Performance

- **Duration:** ~35 min
- **Started:** 2026-06-05T07:02:33Z
- **Tasks:** 3
- **Files created:** 9 / **modified:** 3

## Accomplishments

- Added `metadata.rs`: a flat `Metadata` struct (C++-typed fields, each doc-cited) with boundary validation (Security V5 — label/weight lengths, init_score multiple-of-num_rows, query_boundaries monotone + `[0, .., num_rows]`) and `finish_load` transcribing `Metadata::CalculateQueryWeights` (`metadata.cpp:742-756`) in `label_t`==f32 arithmetic.
- Added `ingest.rs`: `from_mat` (dense row-major), `from_csr`, `from_csc` — single validated public entries (`config/set.rs` discipline) that validate ALL caller input at the boundary first (typed `DatasetError`, never a panic), then run the C-API in-memory pipeline; sampling routes through the Phase-1 RNG (`create_sample_indices`, no new RNG); f32->f64 widening at one documented `widen()` site.
- CSR/CSC validate `indptr` monotonicity + `indptr[0]==0` + `last==nnz` + every `indices[k]` in range BEFORE indexing (T-02-10); `from_mat` validates `data.len()==num_rows*num_cols` (T-02-11); binning config rejects non-positive `max_bin` / negative knobs (T-02-12) — all typed errors.
- Proved cross-representation equivalence: dense, CSR, and CSC of the SAME logical matrix (incl. a zero-heavy column) produce bit-identical `bin_upper_bound_` and per-row stored bin indices (tolerance-free internal invariant, no C++ needed).
- Extended `bin_capture.cpp` + `xtask` with (a) a metadata golden emitter (`F32Bits`, `CalculateQueryWeights` verbatim, 5 cases) and (b) an example-dataset emitter (TSV loader, per-feature `FindBin` + per-row `ValueToBin`, layers 1+2). Copied `regression.train` + `binary.train` into the committed fixtures dir.
- Added three golden-replay tests: `metadata.rs` (query-weight + round-trip, bit-exact, + malformed-input typed-error coverage), `ingest_equivalence.rs` (dense/CSR/CSC bit-identical + malformed-input), `example_dataset_parity.rs` (end-to-end bit-exact `bin_upper_bound_` + per-row `value_to_bin` for all 28 features of both datasets, + a `from_mat` smoke test). All pass NOT skipped; `bin-capture` regen is idempotent (empty `git diff` on existing fixtures).

## Task Commits

1. **Task 1: Metadata struct + finish_load query-weight derivation + golden round-trip** — `e78bc6a` (feat)
2. **Task 2: from_mat/from_csr/from_csc ingestion + sampling + boundary validation** — `48f218b` (feat)
3. **Task 3: End-to-end example-dataset parity (copy fixtures, capture goldens, replay)** — `6066a0a` (test)

## Files Created/Modified

- `crates/lgbm-dataset/src/metadata.rs` — `Metadata` (f32 label/weights/query_weights, f64 init_score, i32 query_boundaries) + validated `new` + `finish_load`; 8 inline tests.
- `crates/lgbm-dataset/src/ingest.rs` — `from_mat`/`from_csr`/`from_csc` + `widen()` single site + `validate_binning_config`/`validate_indptr` + `build_mapper`/`finish_from_columns`; 6 inline tests.
- `crates/lgbm-dataset/src/lib.rs` — `pub mod metadata/ingest` + re-exports (`Metadata`, `from_mat`/`from_csr`/`from_csc`).
- `crates/lgbm-dataset/tests/metadata.rs` — metadata golden replay (bit-exact f32 query weights + f64 init_score round-trip) + 3 malformed-input boundary tests.
- `crates/lgbm-dataset/tests/ingest_equivalence.rs` — dense/CSR/CSC bit-identical (zero-heavy column) + 3 malformed-input boundary tests.
- `crates/lgbm-dataset/tests/example_dataset_parity.rs` — end-to-end layer 1+2 parity for both example datasets + a `from_mat` smoke test.
- `crates/lgbm-dataset/tests/fixtures/{metadata,example_dataset_binning}.txt` — committed C++ goldens.
- `crates/lgbm-dataset/tests/fixtures/examples/{regression,binary}.train` — copied example datasets (committed).
- `xtask/cpp/bin_capture.cpp` — `F32Bits`, metadata emitter (`CalculateQueryWeights` verbatim), example-dataset emitter (TSV loader); `main` argv extended to 7 (metadata) / >=9 (examples).
- `xtask/src/main.rs` — passes metadata + example fixture paths; example-input existence check.

## Decisions Made

- **f32 metadata contract.** `label`/`weights`/`query_weights` are `Vec<LabelT>` (f32), `init_score` is `Vec<f64>`, `query_boundaries` is `Vec<DataSizeT>` (i32) — exactly the C++ types. Query weights are computed in f32 (accumulate `+= weights[j]` in f32, divide by the int group size which converts to f32). The `groups_frac_weights` golden (mean of `0.7+0.9+1.1`/3) shows the f32 rounding (`1063675493` not the exact value) — a f64 mean would diverge.
- **Single widen site.** All caller f32 feature data crosses into f64 binning via `widen()`. `grep "as f64"` in `ingest.rs` shows only the body of `widen()` (the other two hits are doc comments).
- **Sparse gather (Open Q2).** CSR/CSC gather is dense-by-column with absent entries defaulting to `0.0` (zero_cnt-aware). The equivalence test includes a zero-heavy column so the "absent == 0.0" path is proven bit-equivalent to the dense form.
- **Example parity row cap.** `kExampleMaxRows = 500` (a shared C++/Rust constant) keeps the golden ~248 KB. The rigorous parity compare builds the mapper on the f64 columns directly (matching the C++ harness); the `from_mat` end-to-end call is a smoke test only (f32-narrowing the example data would not match the f64-binned golden, so no bin-equality is asserted there).
- **Fixture provenance.** Example datasets are COPIED into `tests/fixtures/examples/` and read from there; no test references the untracked `LightGBM/` tree (project memory).

## Deviations from Plan

None — plan executed exactly as written. The plan listed `crates/lgbm-dataset/src/dataset.rs` in Task 2's `<files>`, but the existing `Dataset::construct`/`push_value`/`finish_load` API (Plan 02) was sufficient for the ingestion pipeline, so no change to `dataset.rs` was needed (the ingestion wiring lives entirely in `ingest.rs`). This is a no-op, not a scope change.

## Capture-harness note (external_libs unavailable — continued from Plans 01-03)

Consistent with prior plans: `metadata.cpp`/`bin.cpp` are unbuildable here (their includes transitively pull the unvendored `external_libs/`). `bin_capture.cpp` therefore **verbatim-transcribes** `CalculateQueryWeights` (metadata.cpp:742-756) and reuses the already-transcribed numeric `FindBin`/`ValueToBin`, linking only the header-only reference `Random` for sampling — so the metadata + example goldens are byte-identical to lib_lightgbm. The example datasets are real (copied from `LightGBM/examples/`), so the end-to-end parity is a genuine real-data proof, not synthetic. Regen is idempotent (empty `git diff` on existing fixtures).

## Issues Encountered

- The example per-row golden compares against the mapper's `value_to_bin` (the layer-2 `ValueToBin` contract), NOT the stored bin index (which carries the most-freq skip + offset adjustments). The example harness emits `ValueToBin(col[r])` directly, and the Rust test mirrors that — keeping the comparison a clean layer-2 check independent of storage packing.

## User Setup Required

None. `cargo run -p xtask -- bin-capture` needs a C++ toolchain + CMake (already present, used here); normal `cargo test` replays the committed fixtures with no toolchain.

## Next Phase Readiness

- The only externally-callable surface this phase exposes is live: `from_mat`/`from_csr`/`from_csc` + `Metadata` ingest real data into an immutable, C++-bit-identical `FinishedDataset` (SC#2, SC#5 end-to-end).
- Cross-representation equivalence + real example-dataset parity close the binning determinism root: predict (Phase 3) and histogram (Phase 4) read an immutable Dataset whose bins are proven identical to C++.
- EFB grouping / `MultiValBin` (the multi-feature-per-group path) remains intentionally unimplemented (Plan 05); ingestion currently builds the one-feature-per-group default.

## Known Stubs

None. `from_mat`/`from_csr`/`from_csc` are complete validated entries; `Metadata::finish_load` is the full query-weight derivation. The categorical path exists (Plan 03) but the example datasets are numeric, so ingestion routes numeric `find_bin` for them — this is correct, not a stub.

## Threat Flags

None. No new network endpoints, auth paths, or trust boundaries beyond the caller->ingestion surface already enumerated in the plan's `<threat_model>` (T-02-10..13 all mitigated: indptr/indices validation, shape/length checks, sample-count clamp, degenerate-input guards via the existing `find_bin` trivial path).

---
*Phase: 02-dataset-binning-determinism-root*
*Completed: 2026-06-05*

## Self-Check: PASSED

All 9 created key files verified present on disk; all 3 task commits (e78bc6a, 48f218b, 6066a0a) verified in git history. `cargo test --workspace` green (88 dataset-crate test results across lib + integration, plus core/oracle/xtask); `cargo run -p xtask -- bin-capture` idempotent (no diff on existing tracked fixtures).
