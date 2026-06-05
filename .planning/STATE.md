---
gsd_state_version: 1.0
milestone: v1.0
milestone_name: milestone
status: verifying
stopped_at: Completed 02-06-PLAN.md (gap closure — GAP-1/GAP-2 closed; phase ready to re-verify)
last_updated: "2026-06-05T09:05:11.073Z"
last_activity: 2026-06-05 -- Plan 02-06 executed (default-config scaled-filter_cnt gap closure)
progress:
  total_phases: 8
  completed_phases: 1
  total_plans: 9
  completed_plans: 9
  percent: 50
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-06-05)

**Core value:** For identical inputs and config, reproduce C++ LightGBM outputs to within ~1e-6 absolute difference on every backend (CPU and ROCm), using f32 (single-precision) data types matching the C++ reference defaults.
**Current focus:** Phase 02 — dataset-binning-determinism-root

## Current Position

Phase: 02 (dataset-binning-determinism-root) — GAP CLOSURE DONE (ready to re-verify)
Plan: 6 of 6 executed (gap-closure plan)
Status: All 6 plans executed; GAP-1 + GAP-2 closed. Phase ready for re-verification.
Last activity: 2026-06-05 -- Plan 02-06 executed (default-config scaled-filter_cnt gap closure)

Progress: [██████████] 6/6 plans executed — both blocking verification gaps closed; re-verify next

### Resume

Plan 02-06 closed both blocking phase-02 verification gaps:

- **GAP-1 (CR-01/IN-02, SC#1, DAT-01) CLOSED:** `ingest.rs::build_mapper` now feeds the SCALED `filter_cnt = (min_data_in_leaf * sample_cnt) / num_rows` (via the new `bin_mapper.rs::scaled_filter_cnt` single-source-of-truth helper) to `find_bin_numeric` instead of raw `cfg.min_data_in_leaf`; `find_bin_from_column` routes through the SAME helper; CSR/CSC inherit the fix via `finish_from_columns -> build_mapper` (convention confirmed identical to dense in c_api.cpp).
- **GAP-2 (CR-02, SC#5 / ORA-03 bin stage, DAT-07) CLOSED:** new `default_config_ingest_parity.rs` + `default_config_ingest.txt` golden cover the DEFAULT `feature_pre_filter=true`, sample_cnt<num_rows path. Fails-before (`feature 1: is_trivial_ true != golden false`) / passes-after. Engineered feature f1 flips is_trivial_ between filter_cnt=5 and raw 20.

`cargo test --workspace` fully green (lgbm-dataset 75 lib + all integration; core; oracle; xtask — 0 failed); pre_filter=false suites (ingest_equivalence, example_dataset_parity) unchanged and passing; bin-capture idempotent (no existing golden mutated, example COUNTS line unchanged).

Next: `/gsd-verify-phase 02` (re-run phase verification — expect SC#1 + SC#5 to flip to PASS).

Verified PASS (prior): SC#2 (ingest + immutable store), SC#3 (missing/categorical routing), SC#4 (EFB grouping). RE-AFFIRMED this plan: SC#1 (is_trivial_ on default path), SC#5 (per-stage parity covers default path). DAT-01/DAT-07 move PARTIAL -> SATISFIED for the default-config divergence (pending formal re-verify).

- EFB (Plan 02-05): `MultiValBin` dense/sparse storage (+1 push), `efb.rs` (`fast_feature_bundling`/`find_groups`/`get_conflict_count`/`fix_sample_indices` — ALL randomness via `lgbm_core::Random::new(num_data)`, STABLE sorts, element-wise parallel-vector swap), and the `Dataset::construct_bundled` `enable_bundle` dispatch with real<->packed feature maps (`used_feature_map_`/`real_feature_idx_`). EFB layer-3 golden (feature->group membership + per-group `bin_offsets_`/`num_total_bin_`/`group_is_multi_val` + per-row bundled indices) bit-identical to C++ on the D-06 #4 mutually-exclusive sparse corpus + a no-bundle control. EFB capture = HEADER-ONLY verbatim transcription of `dataset.cpp` (external_libs unvendored → both nominal capture paths infeasible; human-approved). bin-capture idempotent.

- Ingestion (Plan 02-04): `from_mat`/`from_csr`/`from_csc` (validated entry points, Security V5 — typed `DatasetError` never a panic) wire sample → `find_bin` → `Dataset::construct` → `push` → `finish_load`; sampling routes through the Phase-1 RNG (`create_sample_indices`), f32→f64 widening at one `widen()` site. Dense vs CSR vs CSC bin bit-identically (incl. a zero-heavy column). `Metadata` (f32 label/weights/query_weights, f64 init_score, i32 query_boundaries) + `finish_load` query-weight derivation (f32 `CalculateQueryWeights`) round-trips bit-exact. End-to-end parity: regression + binary_classification example datasets (28 features each, 500 rows) bin bit-identical to C++ for every feature (layers 1+2). Example fixtures COPIED into committed `tests/fixtures/examples/`.

- `lgbm-dataset` crate exists with `BinMapper` (numeric `find_bin`/`value_to_bin`, bit-exact f64 boundary kernel), `BinType`/`MissingType`, and `DatasetError`.
- Categorical (Plan 02-03): `find_bin_categorical` (descending-count fold via stable `SortForPair`, f32 0.99 cut, `min_data_in_bin` fold-break, NaN dummy bin 0) + `categorical_2_bin_`/`bin_2_categorical_` + categorical `value_to_bin`; completed `MissingType` routing (None/Zero/NaN, signed zeros, all-missing) — 6-case categorical (layers 1+3 + per-row) + 8-case missing (layer 1 + per-row) golden replay, bit-identical, idempotent.
- Storage layer (Plan 02-02): `Bin` trait + `BinValue` (u8/u16/u32) + `create_dense_bin`/`create_sparse_bin` factories (Box<dyn Bin>, D-01); `DenseBin<T, IS_4BIT>` incl. 4-bit packing (D-02); `SparseBin<T>` delta-encode + fast-index; `FeatureGroup` offset packing (u64) + `PushData`; `Dataset::construct` + `finish_load` type-state immutability (→ `FinishedDataset`).
- `bin-capture` xtask subcommand + `xtask/cpp/bin_capture.cpp` emit numeric goldens (layers 1+2) AND storage-layout goldens (DenseBin/SparseBin/FeatureGroup bytes); oracle-harness has `compare_exact_u32`/`compare_exact_f64_bits`/`compare_exact_bytes`; later plans plug in here.
- 45-case numeric golden replay (bit-identical, SC#1/SC#5) + 6-case storage golden replay (byte-identical incl. 4-bit + sparse, SC#2); both regen idempotent.
- `lgbm_core::Config` + `Config::from_params` are the config bag for all later crates (Phase 1).

## Performance Metrics

**Velocity:**

- Total plans completed: 1 (tracked)
- Average duration: ~3 min
- Total execution time: <1 hour

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 01-oracle-contract-foundations | 3/3 | ~2 sessions | ~1 session |
| 02-dataset-binning-determinism-root | 5/5 | ~65 min + continuation | ~16 min |

**Plan 01-02:** 3 tasks, 11 files (9 created + 2 modified), 29 new tests; `cargo test --workspace` green.
**Plan 01-03:** 2 TDD tasks, 3 files modified, 7 new tests (49 → 56); deterministic alias resolution + empty==absent reads; `cargo test --workspace` green.
**Plan 02-01:** 4 tasks (~12 min), 14 files (9 created + 5 modified), numeric BinMapper kernel + bin-capture harness + 45-case golden replay (layers 1+2 bit-exact); `cargo test --workspace` green.
**Plan 02-02:** 3 tasks (~9 min), 11 files (7 created + 4 modified), bin-storage layer (Bin trait + DenseBin/4-bit + SparseBin + FeatureGroup offsets + Dataset finish_load type-state immutability) + 6-case storage golden replay (byte-exact); 25 new lib tests (10→35); `cargo test --workspace` green; bin-capture idempotent.
**Plan 02-03:** 2 tasks (~9 min), 9 files (4 created + 5 modified), categorical `find_bin_categorical` (stable descending-count fold + f32 0.99 cut + fold-break + NaN dummy bin) + categorical `value_to_bin` + completed MissingType routing; 6-case categorical (layers 1+3 + per-row) + 8-case missing (layer 1 + per-row) golden replay (bit-exact); 7 new inline + 2 integration tests (35→42 lib); `cargo test --workspace` green; bin-capture idempotent.
**Plan 02-04:** 3 tasks (~35 min), 12 files (9 created + 3 modified), `Metadata` (f32 query-weight derivation) + `from_mat`/`from_csr`/`from_csc` ingestion (validated entries, single widen site, Phase-1-RNG sampling); dense/CSR/CSC bit-identical (zero-heavy column) + metadata golden + end-to-end example-dataset parity (regression + binary, 28 features each, layers 1+2 bit-exact); 14 new inline + 3 integration test files (42→56 lib); `cargo test --workspace` green; bin-capture idempotent.
**Plan 02-05:** 4 tasks (Tasks 1-2 prior agent + Tasks 3-4 continuation), 8 files (2 created + ~6 modified), `MultiValBin` dense/sparse + `efb.rs` (FastFeatureBundling/FindGroups/GetConflictCount/FixSampleIndices, Phase-1 RNG + stable sorts) + `construct_bundled` enable_bundle dispatch with real<->packed feature maps; EFB layer-3 golden (3 cases: 2 mutually-exclusive sparse bundling + no-bundle control) bit-exact (membership + bin_offsets_ + num_total_bin_ + per-row bundled indices); EFB capture = header-only verbatim transcription of dataset.cpp; 1 Rule-1 bug fixed (real-vs-packed indexing); `cargo test --workspace` green; bin-capture idempotent.
**Plan 02-06 (gap closure):** 3 tasks (~18 min), 6 files (2 created + 4 modified), `scaled_filter_cnt` single-source-of-truth helper (exact C++ integer-truncation, num_rows==0 Rust-only guard) routed from both `ingest.rs::build_mapper` and `bin_mapper.rs::find_bin_from_column`; CSR/CSC inherit via finish_from_columns. New default-config (feature_pre_filter=true, sample_cnt=50<num_rows=200, min_data_in_leaf=20 -> filter_cnt=5) ingest parity golden + test: fails-before (`feature 1: is_trivial_ true != golden false`) / passes-after; engineered feature f1 flips is_trivial_ between filter_cnt=5 and raw 20. GAP-1 (CR-01/IN-02, SC#1, DAT-01) + GAP-2 (CR-02, SC#5/ORA-03, DAT-07) closed. New golden dispatched from fixed argv[8] before the variadic example tail (kFirstExampleArgv 9->10) so no existing golden mutated. 1 new unit test + 1 integration test (75 lib); `cargo test --workspace` green; bin-capture idempotent.

**Recent Trend:**

- Last 5 plans: -
- Trend: -

*Updated after each plan completion*

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.
Recent decisions affecting current work:

- [Roadmap]: Build order is dependency-forced — each layer must match the reference before anything above it can be validated (binning → predict → compute → learner → GBDT → variants → Python).
- [Phase 1 discuss, 2026-06-05]: Numerical contract revised to **f32 (single-precision) end-to-end, ~1e-6 absolute oracle tolerance** on every backend — matching the C++ reference defaults (`score_t`/`label_t` = `float`). Supersedes the prior strict-1e-12 / tiered-oracle direction.
- [Phase 1 discuss, 2026-06-05]: **Standard f32 histogram/score accumulations** on CPU and ROCm — the integer-quantized histogram strategy is dropped (buys nothing at f32 / ~1e-6).
- [Phase 1 exec, 2026-06-05]: **Deterministic config invariant** — `from_params` alias-collision resolution is a faithful port of C++ `ParameterAlias::KeyAliasTransform` + `Config::SortAlias` (canonical beats alias; alias-vs-alias ties by `(key.len(), key)`). No observable Config outcome may depend on HashMap iteration order; enforced by an N-run determinism test. The seed + six enum reads route through `present()` so empty == absent (C++ `Get*` parity).
- [Phase 1 exec, 2026-06-05]: **Header-only C++ RNG capture** — `rng_capture` compiles directly against `include/LightGBM/utils/random.h` instead of linking `lib_lightgbm` (external_libs submodules not vendored, so the full lib is unbuildable). Numerically identical reference source; preserves FND-01 / ORA-02 / D-14 parity contract. Master seed 1592594996, 512 cases.
- [Phase 2 exec, 2026-06-05]: **Numeric binning capture via verbatim transcription** — `xtask/cpp/bin_capture.cpp` verbatim-transcribes the numeric `BinMapper::FindBin`/`ValueToBin` family from the pinned `bin.cpp`/`bin.h` (using real `std::nextafter`) rather than compiling `bin.cpp`, because `external_libs/{fast_double_parser,fmt}` are empty/unvendored here so `bin.cpp` → `common.h` is unbuildable. Header-only reference `Random` used for sampling. Goldens byte-identical to lib_lightgbm. Binning compared **bit-exact** (`.to_bits()` / exact-u32), NOT the ~1e-6 oracle tolerance. BIN_MASTER_SEED 0x0B11BEEF, 45 cases.
- [Phase 2 exec, 2026-06-05]: **Numeric binning determinism root locked (DAT-01, ORA-03)** — `BinMapper::find_bin`/`value_to_bin` produce `bin_upper_bound_` (f64 `next_up` boundary math, asymmetric `b <= a.next_up()` dedup) and per-row bin indices (literal `(r+l-1)/2` + `<=` search) bit-identical to C++ across 45 cases / 21,475 rows.
- [Phase 2 exec, 2026-06-05]: **Bin-storage layer locked (DAT-02)** — `Bin` trait + `BinValue` width abstraction + `create_dense_bin`/`create_sparse_bin` factories (`Box<dyn Bin>`, D-01) selecting u8/u16/u32 + the 4-bit packed `DenseBin<u8,true>` (num_bin<=16, D-02) exactly per bin.cpp:613-633; `DenseBin` 4-bit even/odd `buf_` split + OR-merge, `SparseBin` 255-run-length delta + GetFastIndex, `FeatureGroup` u64 offset packing + `PushData` (most-freq skip / -1 / +offset) all byte-identical to C++ across 6 storage cases spanning every width path + sparse + odd-row 4-bit.
- [Phase 2 exec, 2026-06-05]: **Dataset immutability = type-state, not a runtime flag** — `Dataset::finish_load(self)` consumes the mutable loading state and returns `FinishedDataset` (no `push_*`/`finish_load` method), so a post-finish mutation is a COMPILE error — strictly stronger than C++ `is_finish_load_` while observably identical.
- [Phase 2 exec, 2026-06-05]: **Categorical folding + missing routing locked (DAT-03, DAT-04)** — `find_bin_categorical` transcribes the descending-count fold via the STABLE `slice::sort_by` (mirrors `SortForPair(is_reverse=true)`; equal-count ties keep ascending-value order), the f32 `0.99` cut (`RoundInt((rest as f32 * 0.99f) ...)` — NOT `0.99_f64`, which would shift the cut), the `min_data_in_bin && cur_cat_idx>1` fold-break, and the NaN dummy bin 0; `categorical_2_bin_` is a by-key `HashMap` (fold order is sort-driven). Missing routing (None/Zero/NaN, signed +0/-0 identical, all-missing single bin) proven bit-identical to C++ across a 6-case categorical (layers 1+3 + per-row) + 8-case missing (layer 1 + per-row) golden battery.
- [Phase 2 exec, 2026-06-05]: **`autobins = false` on lgbm-dataset** — the locked module path `src/bin/{mod,dense_bin,sparse_bin}.rs` collides with Cargo's binary-target convention (Cargo tried to compile the storage module files as `main`-bearing binaries). Disabling binary auto-discovery preserves the path; the crate ships no binaries.
- [Phase 2 exec, 2026-06-05]: **EFB grouping locked (DAT-05)** — `efb.rs` transcribes `FastFeatureBundling`/`FindGroups`/`GetConflictCount`/`FixSampleIndices` verbatim with ALL randomness via `lgbm_core::Random::new(num_data)` (group search `.sample` + shuffle `.next_short`), STABLE sorts only, and element-wise parallel-vector swap (`features_in_group` + `group_is_multi_val` per iter — the C++ `std::swap` on `vector<bool>` hazard). `Dataset::construct_bundled` dispatches on `cfg.enable_bundle` and stores `used_feature_map_` (real->packed) + `real_feature_idx_` (packed->real) so the shuffled bundled grouping pushes each feature to its correct group. EFB layer-3 golden (feature->group membership + per-group `bin_offsets_`/`num_total_bin_`/`group_is_multi_val` + per-row bundled bin index) bit-identical to C++ on the D-06 #4 mutually-exclusive sparse corpus + a no-bundle control. EFB capture = HEADER-ONLY verbatim transcription of `dataset.cpp` (both nominal capture paths infeasible because external_libs are unvendored — human-approved). Fixed a Rule-1 real-vs-packed indexing bug surfaced by the per-row golden.
- [Phase 2 exec, 2026-06-05]: **Scaled pre-filter threshold locked (GAP-1/GAP-2, SC#1/SC#5)** — the default in-memory ingest path now feeds the SCALED `filter_cnt = (min_data_in_leaf * total_sample_cnt) / num_rows` (i64 integer truncation, exact analog of `dataset_loader.cpp:623-624`) to `find_bin_numeric`, computed once in `bin_mapper.rs::scaled_filter_cnt` and called from BOTH `ingest.rs::build_mapper` and `bin_mapper.rs::find_bin_from_column` (no divergent raw-forwarding copy). `from_csr`/`from_csc` inherit via `finish_from_columns -> build_mapper` (CSR/CSC use the identical dense convention `total_sample_size=sample_cnt`, `num_dist_data=total_nrow`, confirmed in c_api.cpp). `num_rows==0` returns raw `min_data_in_leaf` as a Rust-only, parity-unobservable empty-matrix guard (no C++ analog — C++ never reaches FindBin with zero rows). Proven by a default-config (feature_pre_filter=true, sample_cnt=50<num_rows=200, min_data_in_leaf=20 -> filter_cnt=5) ingest parity golden that FAILS before and PASSES after, asserting `is_trivial_` + STORED per-row bins (read from `feature_group().bin_data()`, NOT recomputed). Golden cells are f32-representable (from_mat takes &[f32]) so the f32->f64 widen does not drift boundaries.
- [Phase 2 exec, 2026-06-05]: **Ingestion API locked (D-05, DAT-06/07)** — `from_mat`/`from_csr`/`from_csc` are single validated public entries (validate ALL caller input first → typed `DatasetError`, never panic; Security V5 / T-02-10..13) wiring sample→`find_bin`→`construct`→`push`→`finish_load`. f32→f64 widening at ONE `widen()` site; sparse gather is dense-by-column (absent==0.0, Open Q2). Dense/CSR/CSC of the same matrix bin bit-identically (tolerance-free internal invariant). `Metadata` query weights computed in f32 (`CalculateQueryWeights` verbatim). End-to-end real example-dataset (regression + binary) binning bit-identical to C++ for all 28 features × both datasets (layers 1+2). Example fixtures COPIED into the committed dir, never the untracked LightGBM/ tree.

### Pending Todos

None yet.

### Blockers/Concerns

- [Phase 4]: CubeCL is alpha (v0.10.0) — pin exactly, isolate behind lgbm-compute; ROCm capability gaps and CPU-runtime-vs-HIP divergence need empirical validation on the local ROCm GPU (research flag: HIGH). Now evaluated against the ~1e-6 (f32) tolerance rather than bit-exactness.
- [Phase 6]: f32 transcendental (exp/log/pow/sigmoid) parity CPU↔ROCm is unproven at ~1e-6 — needs empirical validation; if a gap appears, fallback is CPU-resident objective grad/hess.
- [Cross-cutting]: RESOLVED (2026-06-05) — the strict-1e-12-vs-tiered tension is closed by adopting the f32 / ~1e-6 contract; project docs (PROJECT/REQUIREMENTS/ROADMAP) updated to match.

## Deferred Items

Items acknowledged and carried forward from previous milestone close:

| Category | Item | Status | Deferred At |
|----------|------|--------|-------------|
| v2 | QNT-01 quantized/discretized gradient training | Deferred (v2) | Roadmap |
| v2 | LIN-01 linear-tree leaves | Deferred (v2) | Roadmap |
| v2 | ING-01/02/03 text-file / binary-cache / Arrow ingestion | Deferred (v2) | Roadmap |

## Session Continuity

Last session: 2026-06-05T08:23:00Z
Stopped at: Completed 02-06-PLAN.md (gap closure — GAP-1/GAP-2 closed; phase ready to re-verify)
Resume file: None
