---
phase: 02-dataset-binning-determinism-root
verified: 2026-06-05T00:00:00Z
status: gaps_found
score: 3/5 success-criteria verified
overrides_applied: 0
re_verification:
  previous_status: gaps_found
  previous_score: 3/5
  gaps_closed:
    - "GAP-1 (CR-01/IN-02, SC#1, DAT-01): build_mapper now feeds the SCALED filter_cnt via the single source-of-truth scaled_filter_cnt helper; is_trivial_ for f1 matches C++ on the default config."
    - "GAP-2 (CR-02, SC#5 bin stage, DAT-07): a default feature_pre_filter=true ingest parity test (default_config_ingest_parity.rs) now exists and discriminates the scaled-filter_cnt fix on is_trivial_."
  gaps_remaining: []
  regressions:
    - "NEW determinism-root divergence surfaced by 02-06 review (CR-01, distinct from the old CR-01): non-bundled Dataset::construct stores trivial features and ignores enable_bundle, diverging from C++ Dataset::Construct on the default path. The 02-06 golden was emitted to match the divergent Rust path, masking it."
gaps:
  - truth: "A user can ingest dense/CSR/CSC into a columnar store whose feature set, grouping, bin-offsets and stored bins are bit-identical to C++ Dataset::Construct on the default configuration (SC#2 + determinism-root goal)"
    status: failed
    reason: >-
      The public ingest path (from_mat/from_csr/from_csc ->
      finish_from_columns:122 -> Dataset::construct) uses the NON-bundled
      Dataset::construct (dataset.rs:84-94), which (a) builds a FeatureGroup for
      EVERY feature and sets used_feature_map_ = (0..num_features_), so it STORES
      bins for trivial features, and (b) ignores cfg.enable_bundle entirely. C++
      Dataset::Construct (LightGBM/src/io/dataset.cpp:337-343) ALWAYS builds
      used_features from only NON-trivial mappers (!is_trivial()), groups over just
      those, and leaves used_feature_map_[real] = -1 for trivial features — so a
      trivial feature is dropped (no FeatureGroup, no bin offset, no stored bins).
      C++ additionally runs FastFeatureBundling because enable_bundle defaults to
      true (config.h:711; Rust default also true, config/mod.rs:353), which the
      ingest path never invokes. Concretely, in the committed golden
      crates/lgbm-dataset/tests/fixtures/default_config_ingest.txt the trivial
      control feature f2 (is_trivial=1, num_bin=3) carries NON-ZERO stored bins
      for all 200 rows (ASSIGN f=2 line: 2;2;1;1;...), whereas real C++ Construct
      would store NOTHING for f2 and would not allocate it as a group. The presence
      of ANY trivial feature shifts num_features_, feature2group_,
      feature2subfeature_, and every downstream bin-offset / num_total_bin_ — the
      exact determinism-root invariants this phase exists to protect. Independently
      confirmed against dataset.cpp:325-441 (single Construct that filters trivial
      features before the enable_bundle branch) and the Rust caller chain
      (ingest.rs:122, 180, 277, 340).
    artifacts:
      - path: "crates/lgbm-dataset/src/dataset.rs"
        issue: "construct (lines 84-94): builds a group for every feature and sets used_feature_map_ = (0..num_features_); does NOT filter trivial features and does NOT honor enable_bundle. construct_bundled (130-133) DOES filter, but from_mat never calls it."
      - path: "crates/lgbm-dataset/src/ingest.rs"
        issue: "finish_from_columns (line 122) calls Dataset::construct (non-bundled) for all of from_mat/from_csr/from_csc; cfg.enable_bundle (default true) is ignored, so the default-config grouping diverges from C++."
      - path: "xtask/cpp/bin_capture.cpp"
        issue: "StoredBinSingleGroup (2140-2152) computes ValueToBin for trivial features too (never consults is_trivial_); its own comment (2137-2139) states it intentionally mirrors the Rust non-bundled path 'trivial features are NOT skipped'. The golden was therefore produced to match the divergent Rust behavior, not C++ Construct, masking the divergence."
      - path: "crates/lgbm-dataset/tests/fixtures/default_config_ingest.txt"
        issue: "Feature f2 (is_trivial=1) has a full non-zero ASSIGN/stored-bin line for 200 rows; a real C++ dataset would store nothing for f2 and would exclude it from used_feature_map_."
    missing:
      - "Make the non-bundled Dataset::construct mirror C++ Construct: build used_features from only !is_trivial_ mappers, group over just those, set used_feature_map_[real] = -1 for trivial features (push_value already early-returns on packed < 0)."
      - "Route the default ingest path through the enable_bundle dispatch (construct_bundled / single unified Construct) so the default config (enable_bundle=true) bundles AND filters trivial features as C++ does, instead of from_mat hard-coding the one-feature-per-group non-bundled path."
      - "Regenerate default_config_ingest.txt through an emitter that models C++ Dataset::Construct for trivial features (trivial feature -> dropped / no store), and have the test assert trivial features are excluded from used_feature_map_ (not stored). The is_trivial_ flip on f1 must remain so the test still discriminates the scaled-filter_cnt fix."
  - truth: "Per-stage parity tests cover bin granularity for the DEFAULT ingest configuration, localizing any divergence to binning before histograms exist (SC#5 / ORA-03 bin stage)"
    status: failed
    reason: >-
      The new default_config_ingest_parity.rs DOES correctly discriminate the
      GAP-1/GAP-2 scaled-filter_cnt fix on is_trivial_ (its primary assertion at
      line 261 compares mapper.is_trivial_ against the golden; f1 flips
      trivial->non-trivial only under the scaled threshold). However its STORED
      per-row bin assertion (line 301) compares Rust's stored bins against a golden
      that was emitted to match the SAME divergent Rust path (trivial features
      stored), so it does NOT compare against C++ Construct for the trivial case —
      it validates Rust-against-Rust there. The test also assumes one-group-per-
      feature for ALL features including trivial ones (ds.feature_group(f) at line
      257), which only holds because of the very divergence in the first gap. As a
      result no parity test localizes the trivial-feature / enable_bundle
      divergence; the green suite is a false positive on that axis. WR-01: the test
      additionally returns success (SKIP) if the committed golden is missing
      (lines 190-198), breaking the fails-before/passes-after contract.
    artifacts:
      - path: "crates/lgbm-dataset/tests/default_config_ingest_parity.rs"
        issue: "Line 257 reads ds.feature_group(f) for every feature incl. trivial (only valid under the divergent store); line 301 STORED assertion compares against a Rust-mirrored golden; lines 190-198 silently PASS when the golden is missing (WR-01)."
      - path: "xtask/cpp/bin_capture.cpp"
        issue: "EmitDefaultConfigIngest/StoredBinSingleGroup model the Rust non-bundled store, not C++ Construct, so no committed golden exercises the C++ trivial-feature-dropped representation."
    missing:
      - "Add (or fix) a default-config ingest parity case whose golden is captured through C++ Dataset::Construct (trivial features dropped, enable_bundle=true grouping), asserting identical num_features_ / feature2group_ / used_feature_map_ / per-row stored bins; it MUST fail before the construct fix and pass after."
      - "Make the missing-golden case a hard failure (panic) or #[ignore], not a silent return (WR-01)."
deferred: []
human_verification: []
---

# Phase 2: Dataset + Binning (Determinism Root) Verification Report

**Phase Goal:** A binned, immutable columnar dataset whose bin boundaries and bin
assignments are bit-identical to C++ — the determinism root every downstream split
inherits.

**Verified:** 2026-06-05
**Status:** gaps_found
**Re-verification:** Yes — after gap-closure plan 02-06 (previous status gaps_found, 3/5)
**Mode:** mvp

## Goal Achievement

Plan 02-06 successfully closed the two gaps that triggered the rollback: the
in-memory ingest path now feeds the SCALED `filter_cnt` through a single
source-of-truth helper (`scaled_filter_cnt`), so `is_trivial_` for the engineered
flip feature matches C++ on the default config, and a default `feature_pre_filter=true`
parity test now exists and discriminates that fix. **GAP-1 and GAP-2 are genuinely closed.**

However, the 02-06 code review surfaced a NEW, independently-confirmed
determinism-root divergence (review CR-01) that the gap-closure work introduced a
golden for but did not fix: the public ingest path stores trivial features and
ignores `enable_bundle`, diverging from C++ `Dataset::Construct` on the default
configuration. I verified this directly against the Rust source, the C++ reference,
and the committed golden. For a phase whose sole contract is bit-faithfulness at the
determinism root, this is a BLOCKER — the immutable columnar store (SC#2) is NOT
bit-identical to C++ whenever any feature is trivial, and no parity test localizes it
(SC#5). Status remains **gaps_found**.

### Observable Truths (Success Criteria)

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | `BinMapper::ValueToBin`/`FindBin` produces `bin_upper_bound_` + edge-case bin indices matching C++ goldens exactly (incl. is_trivial_ on the default config) | ✓ VERIFIED | Kernel verified faithful (bin_mapper.rs value_to_bin/greedy_find_bin/need_filter:724). GAP-1 closed: `scaled_filter_cnt` (bin_mapper.rs:715) is the single source of truth, called by both build_mapper (ingest.rs:95) and find_bin_from_column (bin_mapper.rs:660); matches dataset_loader.cpp:623-624 integer truncation; unit tests scaled_filter_cnt_matches_cpp_integer_truncation (1234-1244). default_config_ingest_parity asserts is_trivial_ flip for f1. |
| 2 | User can ingest dense + CSR/CSC + metadata into an IMMUTABLE Dense/Sparse-bin store (matching C++ Construct) | ✗ FAILED | from_mat/from_csr/from_csc -> finish_from_columns -> non-bundled Dataset::construct (dataset.rs:84-94) STORES trivial features and ignores enable_bundle (default true). C++ Dataset::Construct (dataset.cpp:337-343) drops trivial features (used_feature_map_=-1) and bundles. Golden f2 (is_trivial=1) has non-zero stored bins for 200 rows where C++ stores nothing. num_features_/feature2group_/bin-offsets diverge whenever any trivial feature exists. Immutability boundary (type-state finish_load) itself is correct; the STORED CONTENT diverges. |
| 3 | Missing-value handling + categorical encoding route exactly as C++ | ✓ VERIFIED | categorical_2_bin_ + missing routing implemented (bin_mapper.rs); categorical_folding.rs + missing_edge_cases.rs replay committed goldens green. Not affected by CR-01. |
| 4 | EFB (enable_bundle) reproduces C++ feature grouping bit-for-bit | ⚠ VERIFIED (path-isolated) | construct_bundled (dataset.rs:117-210) correctly filters trivial features (130-133) and bundles via efb.rs::fast_feature_bundling; efb_grouping.rs golden green. BUT this path is reachable ONLY from the EFB test, NOT from the default from_mat ingest path — so the correct bundling+filtering logic is never exercised by real ingestion (rolls into SC#2). |
| 5 | Per-stage parity tests cover bin granularity, localizing divergence to binning | ✗ FAILED | default_config_ingest_parity.rs discriminates the scaled-filter_cnt fix (is_trivial_ flip) — that part of GAP-2 is closed. BUT its STORED-bin assertion (line 301) and feature_group(f) loop (line 257) compare against a golden emitted to match the divergent Rust store (bin_capture.cpp:2137-2152 explicitly mirrors the Rust path), so no test localizes the trivial-feature/enable_bundle divergence. WR-01: test silently PASSES if the golden is missing (lines 190-198). |

**Score:** 3/5 success criteria verified (SC#1, SC#3 verified; SC#4 path-isolated; SC#2, SC#5 FAILED).

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `crates/lgbm-dataset/src/bin_mapper.rs` | FindBin numeric+categorical, value_to_bin, need_filter, scaled_filter_cnt | ✓ VERIFIED | scaled_filter_cnt:715 single source of truth; find_bin_from_column:642 routes through it; need_filter:724 mirrors bin.cpp. |
| `crates/lgbm-dataset/src/ingest.rs` | from_mat/from_csr/from_csc validated boundary, scaled filter_cnt | ⚠ WIRED-BUT-WRONG-CONSTRUCT | build_mapper:81 feeds scaled filter_cnt (GAP-1 fixed). BUT finish_from_columns:122 calls non-bundled construct, ignoring enable_bundle + storing trivial features (CR-01). |
| `crates/lgbm-dataset/src/dataset.rs` | construct mirroring C++ Construct (filter trivial, bundle) + finish_load immutability | ✗ DIVERGENT | construct:84-94 does NOT filter trivial / does NOT bundle. construct_bundled:117 does, but is uncalled by ingest. finish_load type-state boundary correct. |
| `crates/lgbm-dataset/tests/default_config_ingest_parity.rs` | default-config parity, fails-before/passes-after | ⚠ PARTIAL | Discriminates scaled-filter_cnt (is_trivial_ flip) correctly; STORED assertion compares vs Rust-mirrored golden; silent SKIP on missing golden (WR-01). |
| `crates/lgbm-dataset/tests/fixtures/default_config_ingest.txt` | C++ golden via Construct (trivial dropped) | ✗ WRONG-REFERENCE | f2 (is_trivial=1) stored with non-zero bins for 200 rows; mirrors Rust path, not C++ Construct. |
| `crates/lgbm-dataset/src/efb.rs` | FastFeatureBundling + FindGroups | ✓ VERIFIED | golden-verified via efb_grouping.rs; correct but unreachable from default ingest. |
| `crates/lgbm-dataset/src/metadata.rs` | Metadata + finish_load query weights | ✓ VERIFIED | metadata.rs tested (4 tests). |
| `crates/lgbm-dataset/src/{feature_group,bin/*}.rs` | offset packing, DenseBin/SparseBin layouts | ✓ VERIFIED | byte-layout goldens green (bin_storage_layout.rs). |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|----|--------|---------|
| ingest.rs::build_mapper | bin_mapper.rs::scaled_filter_cnt | shared helper before find_bin_numeric | ✓ WIRED | GAP-1 fix; single source of truth confirmed (ingest.rs:95, bin_mapper.rs:660 both call it). |
| default_config_ingest_parity.rs | fixtures/default_config_ingest.txt | golden load + from_mat replay | ⚠ WIRED-WRONG-REF | Wired, but golden models the Rust path not C++ Construct (CR-01). |
| ingest.rs::finish_from_columns | dataset.rs::construct | sample->find_bin->construct->push->finish_load | ✗ WRONG-TARGET | Routes to non-bundled construct, bypassing enable_bundle dispatch + trivial-feature filtering. |
| dataset.rs::construct_bundled | efb.rs::fast_feature_bundling | enable_bundle grouping + trivial filter | ✓ WIRED (but unreachable from ingest) | Correct logic, only reachable via efb_grouping.rs test. |

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| Full workspace test suite | `cargo test --workspace` | 8 unit-suites + integration: lgbm-dataset 75 unit + 22 integration; core 14+33; oracle 3+13; 0 failed | ✓ PASS (green, but masks CR-01 — see SC#2/SC#5) |
| scaled_filter_cnt integer truncation | unit `scaled_filter_cnt_matches_cpp_integer_truncation` | (20,50,200)=5, (20,200,200)=20, (20,49,200)=4 | ✓ PASS (GAP-1 verified) |
| C++ Construct trivial-feature exclusion vs Rust | inspection of dataset.cpp:337-343 vs dataset.rs:84-94 + golden f2 ASSIGN | C++ drops f2; Rust stores f2 (200 non-zero bins) | ✗ FAIL — determinism-root divergence confirmed |

### Probe Execution

| Probe | Command | Result | Status |
|-------|---------|--------|--------|
| (none declared) | — | Phase uses cargo-test parity goldens, not scripts/*/tests/probe-*.sh | ? SKIP (no probes declared) |

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|-------------|-------------|--------|----------|
| DAT-01 | 02-01, 02-06 | BinMapper FindBin bit-identical bin boundaries | ✓ SATISFIED | Kernel + scaled_filter_cnt verified; is_trivial_ matches C++ on default config. |
| DAT-02 | 02-02 | DenseBin+SparseBin immutable after finish-load | ⚠ PARTIAL | Byte layouts + type-state immutability correct, BUT stored CONTENT diverges from C++ for trivial features (CR-01). |
| DAT-03 | 02-03 | Missing-value handling routing | ✓ SATISFIED | missing_edge_cases.rs green. |
| DAT-04 | 02-03 | Categorical encoding + low-freq folding | ✓ SATISFIED | categorical_folding.rs green. |
| DAT-05 | 02-05 | EFB feature grouping | ⚠ PARTIAL | efb_grouping.rs green, but construct_bundled is unreachable from the default ingest path (from_mat ignores enable_bundle). |
| DAT-06 | 02-04 | Metadata (labels/weights/init_score/query) | ✓ SATISFIED | metadata.rs tested. |
| DAT-07 | 02-04, 02-06 | In-memory dense + CSR/CSC ingestion | ✗ BLOCKED | scaled filter_cnt fixed, but the default ingest Construct stores trivial features + ignores enable_bundle, diverging from C++ (CR-01). REQUIREMENTS.md marks this SATISFIED — that claim is contradicted by the codebase. |
| ORA-03 | 02-01 | Per-stage parity tests (bin stage in scope) | ✗ BLOCKED | Bin-stage default-config parity discriminates scaled-filter_cnt but compares the trivial-feature store against a Rust-mirrored golden, not C++ Construct (CR-01). |

No orphaned requirements: all 8 IDs (DAT-01..07, ORA-03) are claimed by plans and present in REQUIREMENTS.md.

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| dataset.rs | 84-94 | Non-bundled construct stores trivial features + ignores enable_bundle (diverges from single C++ Construct) | 🛑 Blocker | Determinism-root divergence on default config (CR-01). |
| xtask/cpp/bin_capture.cpp | 2137-2152 | Golden emitter intentionally mirrors the divergent Rust store ("trivial features are NOT skipped") | 🛑 Blocker | Golden validates Rust-against-Rust, masking the divergence (CR-01). |
| default_config_ingest_parity.rs | 190-198 | Test silently PASSES (returns) when committed golden is missing | ⚠ Warning | Breaks fails-before/passes-after contract (WR-01). |
| ingest.rs | 81-107 vs bin_mapper.rs 642-672 | build_mapper duplicates the sampling/gather/find_bin path instead of delegating | ⚠ Warning | Drift hazard; only scaled_filter_cnt is shared (WR-02). |
| ingest.rs | 96-106 | from_mat hard-codes &[] forced bounds; forcedbins_filename silently ignored | ⚠ Warning | Silent C++ divergence when forced bins configured (WR-03). |
| bin_mapper.rs | 724-746 | need_filter numeric branch indexes len()-1 guarded only by >= 1 (asymmetric w/ categorical saturating_sub) | ℹ Info | Latent empty-slice panic edge (WR-04). |

No unreferenced TBD/FIXME/XXX debt markers found in phase source files.

### Human Verification Required

None. The divergence is fully verifiable against the C++ reference
(`LightGBM/src/io/dataset.cpp:337-343`), the Rust source (`dataset.rs:84-94`,
`ingest.rs:122`), and the committed golden's f2 ASSIGN line. No visual / real-time /
external-service checks are needed.

### Gaps Summary

Plan 02-06 closed the gaps it targeted (the scaled `filter_cnt` derivation and a
default-config test that discriminates it). But the phase goal — a columnar store
whose bin assignments are **bit-identical to C++ at the determinism root** — is still
not achieved on the default configuration, because of a distinct divergence the
02-06 review surfaced and I independently confirmed:

1. **CR-01 (blocker):** The public ingest path
   (`from_mat`/`from_csr`/`from_csc` -> `finish_from_columns` -> non-bundled
   `Dataset::construct`, dataset.rs:84-94) STORES trivial features and IGNORES
   `enable_bundle` (default `true`). C++ has a single `Dataset::Construct`
   (dataset.cpp:337-343) that ALWAYS filters trivial features
   (`used_feature_map_[real] = -1`, no group, no store) before the `enable_bundle`
   bundling branch. The committed golden's trivial feature **f2** (`is_trivial=1`)
   carries non-zero stored bins for all 200 rows; real C++ would store nothing.
   Any trivial feature thus shifts `num_features_`, `feature2group_`,
   `feature2subfeature_`, and every bin-offset — the determinism-root invariants
   this phase protects. (SC#2 + goal.)

2. **CR-01 masking (blocker):** The 02-06 golden emitter `StoredBinSingleGroup`
   (bin_capture.cpp:2137-2152) was written to mirror the divergent Rust store
   ("trivial features are NOT skipped"), so the new parity test validates
   Rust-against-Rust on the trivial case and cannot localize the divergence. The
   green suite is a false positive on this axis. (SC#5.)

Both gaps share one root concern and should be closed together: make the non-bundled
`Dataset::construct` mirror C++ (drop trivial features), route the default ingest
path through the `enable_bundle` dispatch so the default config bundles+filters as
C++ does, and regenerate the golden through an emitter that models C++
`Dataset::Construct` (trivial dropped). Also harden the test to fail (not skip) on a
missing golden (WR-01).

Note: REQUIREMENTS.md currently marks DAT-01/DAT-07 as SATISFIED with "CR-01 closed";
that refers to the OLD scaled-filter_cnt CR-01 (which is closed). The NEW 02-06-REVIEW
CR-01 (trivial-feature store / enable_bundle) is a different finding and is NOT closed —
the REQUIREMENTS.md SATISFIED claim for DAT-07 is contradicted by the codebase.

---

_Verified: 2026-06-05_
_Verifier: Claude (gsd-verifier)_
