---
phase: 02-dataset-binning-determinism-root
verified: 2026-06-05T00:00:00Z
status: gaps_found
score: 3/5 success-criteria verified
overrides_applied: 0
gaps:
  - truth: "BinMapper FindBin produces bin assignments / is_trivial_ matching C++ golden exactly for the in-memory ingest path under the DEFAULT configuration (SC#1)"
    status: failed
    reason: >-
      ingest.rs::build_mapper passes the RAW cfg.min_data_in_leaf as the
      min_split_data (pre-filter) argument to find_bin_numeric. The C-API
      in-memory path it claims to mirror (DatasetLoader::ConstructFromSampleData,
      dataset_loader.cpp:623-624) passes the SCALED
      filter_cnt = (min_data_in_leaf * total_sample_size) / num_dist_data
      to FindBin (dataset_loader.cpp:646). NeedFilter (bin.cpp:54-62, used at
      bin.cpp:486) compares cnt_in_bin against this threshold to set is_trivial_.
      With feature_pre_filter defaulting to true and min_data_in_leaf=20, whenever
      sampling is active (sample_cnt < num_rows) the Rust threshold (20) differs
      from C++ (< 20), so is_trivial_ — and therefore which features get a bin
      store, the EFB used_features set, and every downstream split — diverges from
      C++ on the default path. This violates the project's bit-identical (<=1e-12,
      "for identical inputs AND configuration") contract at the determinism root.
      Independently confirmed against LightGBM/src/io/dataset_loader.cpp,
      LightGBM/src/io/bin.cpp, and LightGBM/src/c_api.cpp:1368-1374
      (total_sample_size=sample_cnt, num_dist_data=total_nrow for the dense path).
    artifacts:
      - path: "crates/lgbm-dataset/src/ingest.rs"
        issue: "Line 94: passes cfg.min_data_in_leaf raw instead of computing the scaled filter_cnt = (min_data_in_leaf * total_sample_cnt) / num_rows before calling find_bin_numeric."
      - path: "crates/lgbm-dataset/src/bin_mapper.rs"
        issue: "find_bin_from_column (line 635-662) is a second copy of the same logic that also forwards min_split_data raw with no filter_cnt scaling (IN-02); a fix applied to ingest.rs would leave this divergent copy behind."
    missing:
      - "Compute filter_cnt = ((min_data_in_leaf as i64 * total_sample_cnt as i64) / num_rows as i64) as i32 in build_mapper (dense path: num_dist_data = num_rows, total_sample_size = total_sample_cnt) and pass it as min_split_data, matching dataset_loader.cpp:623-624 integer-truncation exactly."
      - "Verify total_sample_size / num_dist_data definitions against c_api.cpp for CSR/CSC before locking, since integer-division truncation is parity-load-bearing."
      - "Collapse the duplicated find_bin_from_column / build_mapper logic to a single source of truth so the fix cannot be applied to only one copy."
  - truth: "Per-stage parity tests cover bin granularity for the DEFAULT ingest configuration, localizing any divergence to binning (SC#5 / ORA-03 bin stage)"
    status: failed
    reason: >-
      Every ingest-level parity test forces feature_pre_filter=false
      (ingest_equivalence.rs:25, example_dataset_parity.rs:260), and the C++
      golden generator for the realistic example dataset ALSO uses
      /*pre_filter=*/false (bin_capture.cpp:1836). The one pre_filter=1 golden
      (numeric_binning.txt:179 'pre_filter_trivial') exercises find_bin_numeric
      directly with an explicit min_split_data=5, bypassing the ingest path's
      filter_cnt derivation entirely. Therefore NO golden — on either the Rust or
      C++ side — exercises the default feature_pre_filter=true ingest path with
      the scaled filter_cnt. The divergent default path (CR-01) ships with zero
      parity coverage; the green suite is a false positive because all tests avoid
      the divergent path. For a determinism-root phase whose sole contract is
      bit-faithfulness on the default configuration, the highest-risk default path
      having no coverage means SC#5 ("tests localize any divergence to binning")
      is not met.
    artifacts:
      - path: "crates/lgbm-dataset/src/ingest.rs"
        issue: "Test cfg() helper (line 347) sets feature_pre_filter=false; all from_mat/from_csr/from_csc tests reuse it."
      - path: "crates/lgbm-dataset/tests/ingest_equivalence.rs"
        issue: "Line 25 forces feature_pre_filter=false."
      - path: "crates/lgbm-dataset/tests/example_dataset_parity.rs"
        issue: "Line 260 forces feature_pre_filter=false."
      - path: "xtask/cpp/bin_capture.cpp"
        issue: "EmitExampleDataset (line 1836) captures the example golden with /*pre_filter=*/false, so even the realistic golden cannot catch the CR-01 divergence."
    missing:
      - "Add an ingest-level parity case with feature_pre_filter=true and min_data_in_leaf=20 (default), built against a C++ golden captured through the same scaled-filter_cnt in-memory path, asserting identical is_trivial_ / num_bin_ / per-row ASSIGN vectors. This case MUST fail before the CR-01 fix and pass after."
      - "Use a sampling regime where sample_cnt < num_rows (e.g. bin_construct_sample_cnt smaller than num_rows) so filter_cnt != min_data_in_leaf and the divergence is actually triggered."
---

# Phase 02: Dataset & Binning (Determinism Root) Verification Report

**Phase Goal:** A binned, immutable columnar dataset whose bin boundaries and bin
assignments are bit-identical to C++ — the determinism root every downstream split
inherits.

**Verified:** 2026-06-05
**Status:** gaps_found
**Re-verification:** No — initial verification

## Goal Achievement

The numeric/categorical binning *kernel*, storage byte layouts, EFB grouping,
metadata derivation, and missing-value routing are faithful, substantive, and
parity-tested against committed C++ goldens (the code-review verified these
line-for-line, and the artifacts exist with real implementations). **However, the
phase goal is bit-identical binning on identical inputs AND configuration, and the
DEFAULT-configuration ingest path diverges from C++** because it feeds the wrong
pre-filter threshold into the (correct) kernel. The green test suite does not catch
this because every parity test disables the default path. For a determinism-root
phase, a divergence on the default path that ships untested is a blocker.

### Observable Truths (Success Criteria)

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | BinMapper::ValueToBin / FindBin produces bin_upper_bound_ + edge-case bin indices matching C++ goldens exactly | ✗ FAILED | Kernel value_to_bin/find_bin_numeric verified faithful (bin_mapper.rs:201, need_filter:677 mirrors bin.cpp:54). BUT ingest.rs:94 feeds RAW min_data_in_leaf instead of scaled filter_cnt (vs dataset_loader.cpp:623-624 → FindBin:646 → NeedFilter:486), so is_trivial_ diverges on the default config. CR-01 confirmed against C++. |
| 2 | User can ingest dense + CSR/CSC + metadata into an immutable Dense/Sparse-bin store | ✓ VERIFIED | from_mat/from_csr/from_csc (ingest.rs:134/218/281) + finish_load immutability boundary (dataset.rs); ingest_equivalence.rs (4 tests) asserts dense==CSR==CSC; metadata.rs query-weight derivation tested. |
| 3 | Missing-value handling + categorical encoding route exactly as C++ | ✓ VERIFIED | categorical_2_bin_ + missing routing implemented (bin_mapper.rs); categorical_folding.rs + missing_edge_cases.rs replay committed goldens green. |
| 4 | EFB (enable_bundle) reproduces C++ feature grouping bit-for-bit | ⚠ VERIFIED (config-limited) | efb.rs find_groups/fast_feature_bundling + multi_val_bin.rs present; efb_grouping.rs golden green. Grouping logic faithful, BUT the EFB used_features/conflict inputs depend on is_trivial_, which CR-01 corrupts on the default path; the EFB golden also uses pre_filter=false. Grouping itself is correct; upstream triviality feed is the SC#1 defect. |
| 5 | Per-stage parity tests cover bin granularity, localizing divergence to binning | ✗ FAILED | Default feature_pre_filter=true ingest path has ZERO parity coverage. All ingest tests force pre_filter=false (ingest_equivalence.rs:25, example_dataset_parity.rs:260); C++ example golden also pre_filter=false (bin_capture.cpp:1836). CR-02 confirmed. |

**Score:** 3/5 success criteria verified (SC#4 verified with a documented config-coverage caveat that rolls into SC#1/SC#5).

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `crates/lgbm-dataset/src/bin_mapper.rs` | FindBin numeric+categorical, value_to_bin, need_filter | ✓ VERIFIED | 1317 lines; value_to_bin, categorical_2_bin, need_filter present and faithful. Contains the duplicated find_bin_from_column (IN-02). |
| `crates/lgbm-dataset/src/ingest.rs` | from_mat/from_csr/from_csc validated boundary | ⚠ STUB-BEHAVIOR on default path | 416 lines, present and validated, BUT passes wrong pre-filter arg (CR-01). |
| `crates/lgbm-dataset/src/dataset.rs` | construct + finish_load immutability | ✓ VERIFIED | 510 lines; finish_load present. (WR-01: doc claims empty-mapper rejection not enforced — minor.) |
| `crates/lgbm-dataset/src/bin/{dense_bin,sparse_bin,mod}.rs` | DenseBin 4-bit + SparseBin + width factory | ✓ VERIFIED | 216/286/199 lines; byte layouts golden-verified. |
| `crates/lgbm-dataset/src/feature_group.rs` | offset packing + PushData | ✓ VERIFIED | 449 lines; bin_offsets packing golden-verified. |
| `crates/lgbm-dataset/src/efb.rs` | FastFeatureBundling + FindGroups | ✓ VERIFIED | 534 lines; find_groups present, golden-verified. |
| `crates/lgbm-dataset/src/metadata.rs` | Metadata + finish_load query weights | ✓ VERIFIED | 305 lines; tested. |
| `crates/lgbm-dataset/src/multi_val_bin.rs` | MultiValBin dense/sparse | ✓ VERIFIED | 326 lines. (WR-03: pub push_one_row positional assumption — minor.) |
| `crates/oracle-harness/src/comparator.rs` | exact-equality comparators | ✓ VERIFIED | compare_exact_* present; tolerance path unused by binning (IN-04, correct). |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|----|--------|---------|
| ingest.rs / bin_mapper.rs | lgbm_core::random::Random | sampling routes through Phase-1 LCG | ✓ WIRED | create_sample_indices(data_random_seed); rng_parity green. |
| ingest.rs | dataset.rs | sample→find_bin→construct→push→finish_load | ⚠ WIRED-BUT-WRONG-ARG | Pipeline wired, but build_mapper feeds raw min_data_in_leaf as min_split_data (CR-01). |
| feature_group.rs | bin_mapper.rs | PushData → value_to_bin + most_freq_bin | ✓ WIRED | value_to_bin called; golden-verified. |
| ingest.rs | error.rs | validate caller input → DatasetError before indexing | ✓ WIRED | Typed errors for shape/sparse/config; tested. |

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| Full workspace test suite | `cargo test --workspace` | all suites pass (lgbm-dataset: 74 unit + 11 integration; core: 47; oracle: 11; 0 failed) | ✓ PASS (but see SC#5 — green because default pre-filter path is avoided) |
| Default-config ingest parity | (no such test exists) | N/A | ✗ FAIL — the determinism-root default path is uncovered (CR-02) |

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|-------------|-------------|--------|----------|
| DAT-01 | 02-01 | BinMapper FindBin bit-identical bin boundaries | ⚠ PARTIAL | Boundaries/kernel faithful; default-path is_trivial_ diverges (CR-01). |
| DAT-02 | 02-02 | DenseBin+SparseBin immutable after finish-load | ✓ SATISFIED | Byte-layout goldens green; immutability boundary present. |
| DAT-03 | 02-03 | Missing-value handling routing | ✓ SATISFIED | missing_edge_cases.rs golden green. |
| DAT-04 | 02-03 | Categorical encoding + low-freq folding | ✓ SATISFIED | categorical_folding.rs golden green. |
| DAT-05 | 02-05 | EFB feature grouping | ✓ SATISFIED (config-limited) | efb_grouping.rs golden green; inherits CR-01 triviality risk on default config. |
| DAT-06 | 02-04 | Metadata (labels/weights/init_score/query) | ✓ SATISFIED | metadata.rs tested. |
| DAT-07 | 02-04 | In-memory dense + CSR/CSC ingestion | ⚠ PARTIAL | Ingest works + cross-rep equivalence; default pre-filter arg wrong (CR-01) + sparse densification simplification (IN-01, accepted Open Q2). |
| ORA-03 | 02-01 | Per-stage parity tests (bin stage in scope) | ✗ BLOCKED | Bin-stage parity tests do NOT cover the default ingest configuration (CR-02). REQUIREMENTS.md already marks ORA-03 Pending/[ ]. |

No orphaned requirements: all 8 IDs (DAT-01..07, ORA-03) are claimed by plans and present in REQUIREMENTS.md.

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| ingest.rs | 94 | Wrong pre-filter arg (raw min_data_in_leaf vs scaled filter_cnt) | 🛑 Blocker | Determinism-root divergence on default config (CR-01). |
| ingest.rs / tests | 347, 25, 260 | Default path (pre_filter=true) untested | 🛑 Blocker | Divergence ships green (CR-02). |
| bin_mapper.rs | 635-662 | Duplicated find_bin_from_column logic | ℹ Info | Fix-divergence hazard (IN-02). |
| dataset.rs | 78-104 | Doc claims empty-mapper rejection not enforced | ⚠ Warning | Doc/code mismatch (WR-01). |
| dataset.rs / multi_val_bin.rs / sparse_bin.rs | various | pub methods panic on out-of-range caller input | ⚠ Warning | Boundary "never panic" contract gaps (WR-02/03/05). |
| ingest.rs | 259-269,322-332 | CSR/CSC densify-by-column (Open Q2) | ℹ Info | Accepted MVP simplification; sparse golden owed before "faithful" (IN-01). |

No unreferenced TBD/FIXME/XXX debt markers found in phase files.

### Human Verification Required

None. Both gaps are programmatically verifiable against the C++ reference and the
test corpus; no visual/real-time/external-service checks are needed.

### Gaps Summary

The phase delivers a substantive, mostly-faithful binning layer, but it does **not**
achieve the phase goal of *bit-identical binning on the default configuration*:

1. **CR-01 (blocker):** `ingest.rs::build_mapper` passes the raw `min_data_in_leaf`
   as the pre-filter threshold, whereas C++ (`dataset_loader.cpp:623-624`) passes the
   *scaled* `filter_cnt = (min_data_in_leaf * total_sample_size) / num_dist_data`. With
   `feature_pre_filter=true` (default) and sampling active, `is_trivial_` diverges,
   cascading into feature stores, EFB membership, and every downstream split. Confirmed
   independently against `dataset_loader.cpp`, `bin.cpp` (`NeedFilter`/`FindBin`), and
   `c_api.cpp:1368-1374`.

2. **CR-02 (blocker):** No parity test exercises the default `feature_pre_filter=true`
   ingest path — all tests (and the C++ example golden generator) force `pre_filter=false`,
   so CR-01 ships untested. The green suite is a false positive: it passes precisely
   because it avoids the divergent path.

Both gaps share one root concern (the determinism-root default-config ingest path is
wrong and uncovered) and should be closed together: fix the `filter_cnt` derivation
AND add a default-config ingest parity golden that fails before the fix and passes after.

The remaining findings (WR-01..05, IN-01..04) are robustness/maintainability concerns
that do not block the goal and can be addressed independently.

---

_Verified: 2026-06-05_
_Verifier: Claude (gsd-verifier)_
