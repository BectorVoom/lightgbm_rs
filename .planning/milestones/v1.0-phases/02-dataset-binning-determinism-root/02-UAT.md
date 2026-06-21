---
status: complete
phase: 02-dataset-binning-determinism-root
source: [02-01-SUMMARY.md, 02-02-SUMMARY.md, 02-03-SUMMARY.md, 02-04-SUMMARY.md, 02-05-SUMMARY.md, 02-06-SUMMARY.md, 02-07-SUMMARY.md]
started: 2026-06-05T09:50:47Z
updated: 2026-06-05T09:52:30Z
---

## Current Test

[testing complete]

## Tests

### 1. Cold-Start Clean Build + Full Suite
expected: From a clean tree, `cargo clean && cargo test --workspace` compiles every crate and runs the whole suite with 0 failures. Catches stale-artifact / non-idempotent build issues that warm-state runs hide.
result: pass

### 2. Numeric Binning Parity (find_bin / value_to_bin)
expected: `cargo test -p lgbm-dataset --test numeric_assignment --test bin_mapper_internals` passes. The 45-case numeric golden replays bit-for-bit: f64 bin boundaries (via next_up) and per-row value_to_bin indices are exactly equal (.to_bits()) to the C++ capture — not within 1e-6, exactly.
result: pass

### 3. Storage Layout Byte-Exactness + Immutability
expected: `cargo test -p lgbm-dataset --test bin_storage_layout` passes — DenseBin (incl. 4-bit packed even/odd path), SparseBin (delta-encode), and FeatureGroup offsets replay byte-exact across all 6 width/sparse cases. Separately, the type-state boundary holds: a `push_*` after `finish_load` is a compile error (FinishedDataset exposes no mutators).
result: pass

### 4. Categorical Folding + Missing-Value Routing
expected: `cargo test -p lgbm-dataset --test categorical_folding --test missing_edge_cases` passes. Categorical: stable descending-count fold, 0.99 f32 cut, min_data_in_bin fold-break, NaN/negative/unknown -> bin 0 all bit-exact. Missing: None/Zero/NaN derivation and per-row routing across use_missing / zero_as_missing / signed-zero sweeps all match C++.
result: pass

### 5. Ingestion Equivalence + Boundary Validation
expected: `cargo test -p lgbm-dataset --test ingest_equivalence` passes — dense, CSR, and CSC of the SAME matrix (incl. a zero-heavy column) produce bit-identical bin_upper_bound_ and per-row stored bins. Malformed input (bad indptr, length mismatch, non-positive max_bin) returns a typed DatasetError — never a panic.
result: pass

### 6. Metadata Query-Weights (f32 contract)
expected: `cargo test -p lgbm-dataset --test metadata` passes — CalculateQueryWeights computed in f32 replays bit-exact (the 0.7/0.9/1.1 group mean lands on the f32 bit pattern, not the f64 value), init_score f64 round-trips, and malformed metadata (label/weight length, query_boundaries monotonicity) yields typed errors.
result: pass

### 7. Example-Dataset End-to-End Parity
expected: `cargo test -p lgbm-dataset --test example_dataset_parity` passes — for the committed regression.train and binary.train, every feature's bin_upper_bound_ and every per-row value_to_bin index is bit-exact vs the C++ golden across all 28 features.
result: pass

### 8. EFB Bundling Parity
expected: `cargo test -p lgbm-dataset --test efb_grouping` passes — two mutually-exclusive sparse feature sets bundle into one group each (no-bundle control stays separate); feature->group membership, bin_offsets_, num_total_bin_, group_is_multi_val, and per-row bundled bin indices are all bit-exact vs C++.
result: pass

### 9. Default-Config Ingest Parity (scaled filter_cnt + trivial-feature drop)
expected: `cargo test -p lgbm-dataset --test default_config_ingest_parity` passes — under DEFAULT config (feature_pre_filter=true, scaled filter_cnt = min_data_in_leaf*sample_cnt/num_rows), the engineered f1 feature is non-trivial (flips on the scaled threshold) and trivial f2 is DROPPED (feature_to_group == -1, no stored bins) exactly like the single C++ Dataset::Construct. Non-trivial group/subfeature assignments and stored bins are bit-exact.
result: pass

### 10. Golden Capture Idempotency
expected: `cargo run -p xtask -- bin-capture` re-runs the C++ capture harness and leaves every committed fixture byte-identical — `git diff --stat crates/lgbm-dataset/tests/fixtures` is empty afterward. Proves the goldens are reproducible, not hand-edited.
result: pass

## Summary

total: 10
passed: 10
issues: 0
pending: 0
skipped: 0
blocked: 0

## Gaps

[none yet]
