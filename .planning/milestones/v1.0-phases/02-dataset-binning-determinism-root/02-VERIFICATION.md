---
phase: 02-dataset-binning-determinism-root
verified: 2026-06-05T00:00:00Z
status: passed
score: 5/5 success-criteria verified
overrides_applied: 0
re_verification:
  previous_status: gaps_found
  previous_score: 3/5
  gaps_closed:
    - "SC#2 / determinism-root goal (CR-01): the public ingest path (from_mat/from_csr/from_csc -> finish_from_columns:181-208) now routes through construct_bundled (dataset.rs:155-248), the faithful port of the SINGLE C++ Dataset::Construct. It FILTERS trivial features (used_features = !is_trivial_, dataset.rs:169-171) and honors enable_bundle (default true, dataset.rs:178-196). Trivial features are DROPPED: used_feature_map_[real] = -1, no FeatureGroup, no stored bins. Verified against C++ dataset.cpp:337-343 (used_features) + used_feature_map_ init -1. The non-bundled construct (dataset.rs:89-142) was ALSO fixed to filter trivial features symmetrically."
    - "SC#5 / ORA-03 bin stage (CR-01 masking): the golden emitter EmitDefaultConfigIngest (bin_capture.cpp:2156-2348) was regenerated to model C++ Dataset::Construct — it builds used_features from !is_trivial_ (2274-2278), runs FastFeatureBundling, and emits group=/subfeature=/ASSIGN ONLY for non-trivial features (2327-2346); trivial features are dropped entirely. The committed golden (default_config_ingest.txt) now has 4 FEATURE lines with f2 trivial (is_trivial=1, NO group, NO ASSIGN) and ASSIGN only for f0/f1/f3 — the C++-Construct trivial-dropped representation, NOT the old Rust-mirrored store."
    - "WR-01 (silent skip): default_config_ingest_parity.rs:237-238 now hard-panics on a missing/unreadable golden (read_to_string(...).unwrap_or_else(|e| panic!(...))); the prior silent return is gone. The test asserts trivial-feature exclusion (feature_to_group == -1, line 328-334), per-non-trivial feature_to_group/feature_to_subfeature parity vs the C++-Construct golden (line 344-358), and bit-exact stored bins (compare_exact_u32, line 397)."
  gaps_remaining: []
  regressions: []
gaps: []
deferred: []
human_verification: []
---

# Phase 2: Dataset + Binning (Determinism Root) Verification Report

**Phase Goal:** A binned, immutable columnar dataset whose bin boundaries and bin
assignments are bit-identical to C++ — the determinism root every downstream split
inherits.

**Verified:** 2026-06-05
**Status:** passed
**Re-verification:** Yes — after gap-closure plan 02-07 (previous status gaps_found, 3/5)
**Mode:** mvp

## Goal Achievement

The two BLOCKER gaps the prior verification found (CR-01: the public ingest path stored
trivial features and ignored `enable_bundle`, diverging from the single C++
`Dataset::Construct` at the determinism root; and its masking golden + WR-01 silent skip)
are now **genuinely closed in source**, independently confirmed by reading the actual
files, the C++ reference, and running the full workspace test suite.

Concretely, against the codebase (not the SUMMARY):

1. **Live ingest path fixed (SC#2 / goal).** `finish_from_columns` (ingest.rs:199) now
   calls `Dataset::construct_bundled`, the faithful port of the SINGLE C++
   `Dataset::Construct`. `construct_bundled` (dataset.rs:155-248) builds
   `used_features = !is_trivial_` (dataset.rs:169-171), dispatches through
   `enable_bundle` (default true) into `fast_feature_bundling` (dataset.rs:178-196),
   and sets `used_feature_map_[real] = -1` for trivial features — they get NO group, no
   bin offset, no stored bins. This matches C++ `dataset.cpp:337-343` (`used_features`
   from `!is_trivial()`) and `used_feature_map_` init `-1` (dataset.cpp:376) verbatim.
   The non-bundled `Dataset::construct` (dataset.rs:89-142) was ALSO repaired to filter
   trivial features symmetrically.

2. **Golden regenerated to model C++ Construct (SC#5).** `EmitDefaultConfigIngest`
   (bin_capture.cpp:2156-2348) now builds `used_features` from `!is_trivial_`
   (2274-2278), runs `FastFeatureBundling`, and emits `group=`/`subfeature=`/`ASSIGN`
   ONLY for non-trivial features (2327-2346) — trivial features are dropped. The
   committed `default_config_ingest.txt` reflects this: f2 is trivial with NO
   group/subfeature and NO ASSIGN line; only f0/f1/f3 carry ASSIGN. The group IDs are
   the post-shuffle EFB order (f0->group1, f1->group0, f3->group2), so the test
   discriminates EFB grouping order, not just identity.

3. **Parity test hardened (SC#5 / WR-01).** `default_config_ingest_parity.rs` panics on a
   missing golden (237-238), asserts trivial exclusion (`feature_to_group == -1`,
   328-334), asserts `ds.num_features() == used_count` (291-298), per-non-trivial
   group/subfeature parity (344-358), and **bit-exact** stored bins/upper-bounds via
   `compare_exact_u32` / `compare_exact_f64_bits` (NOT the ~1e-6 oracle tolerance).

The bin-boundary and per-row bin goldens are compared **bit-exact** by raw IEEE bit
pattern (`compare_exact_f64_bits` -> `f64::to_bits()`, comparator.rs:150-167) and exact
integer equality (`compare_exact_u32`, 125-142) — satisfying the CLAUDE.md
determinism-root / ≤1e-12 contract. The full workspace suite is green (0 failures),
including `default_config_ingest_matches_cpp`.

The phase goal — a binned, immutable columnar dataset whose bin boundaries AND bin
assignments are bit-identical to C++ — is achieved on the default configuration.

### Observable Truths (Success Criteria)

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | `BinMapper::ValueToBin`/`FindBin` produces `bin_upper_bound_` + edge-case bin indices matching C++ goldens exactly (incl. is_trivial_ on the default config) | ✓ VERIFIED | Kernel faithful (bin_mapper.rs value_to_bin/greedy_find_bin/need_filter). GAP-1 (scaled filter_cnt) remains closed: `scaled_filter_cnt` (single source of truth) feeds both build_mapper (ingest.rs:103) and find_bin_from_column; golden FEATURE lines carry `filter_cnt=5` = (20*50)/200 integer truncation. `default_config_ingest_parity` asserts is_trivial_ flip for the engineered feature + bit-exact bin_upper_bound_ (compare_exact_f64_bits). missing_edge_cases.rs / numeric_assignment.rs green. |
| 2 | User can ingest dense + CSR/CSC + metadata into an IMMUTABLE Dense/Sparse-bin store matching C++ Construct | ✓ VERIFIED | from_mat/from_csr/from_csc -> finish_from_columns:199 -> `construct_bundled` (faithful single C++ Construct). Trivial features DROPPED (used_feature_map_=-1), enable_bundle honored. Matches dataset.cpp:337-343 + used_feature_map_ init -1. Immutability = type-state (finish_load consumes Dataset -> FinishedDataset with no mutator; compile-error on post-finish push). CSR/CSC share the same finish_from_columns tail. ingest_equivalence.rs (4 tests) green. |
| 3 | Missing-value handling + categorical encoding route exactly as C++ | ✓ VERIFIED | categorical_2_bin_ + missing routing (bin_mapper.rs); categorical_folding.rs + missing_edge_cases.rs replay committed goldens green. Unaffected by the 02-07 delta. |
| 4 | EFB (enable_bundle) reproduces C++ feature grouping bit-for-bit | ✓ VERIFIED | construct_bundled (dataset.rs:155-248) filters trivial features and bundles via efb.rs::fast_feature_bundling; efb_grouping.rs golden green. **Now reachable from the default ingest path** (ingest.rs:199), and default_config_ingest_parity asserts the post-shuffle feature_to_group/feature_to_subfeature against the C++-Construct golden — closing the prior "path-isolated" caveat. |
| 5 | Per-stage parity tests cover bin granularity, localizing divergence to binning | ✓ VERIFIED | default_config_ingest_parity.rs discriminates scaled-filter_cnt (is_trivial_ flip), trivial-feature exclusion (feature_to_group==-1), grouping parity, and bit-exact stored bins vs a C++-Construct golden (trivial dropped). Golden emitter (bin_capture.cpp:2156-2348) models C++ Construct, not the old Rust store. Hard panic on missing golden (WR-01 closed). |

**Score:** 5/5 success criteria verified.

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `crates/lgbm-dataset/src/bin_mapper.rs` | FindBin numeric+categorical, value_to_bin, need_filter, scaled_filter_cnt | ✓ VERIFIED | scaled_filter_cnt single source of truth (GAP-1 fix retained). |
| `crates/lgbm-dataset/src/ingest.rs` | from_mat/from_csr/from_csc validated boundary; routes through faithful Construct | ✓ VERIFIED | finish_from_columns:199 calls construct_bundled (enable_bundle dispatch); build_efb_samples:138-162 builds EfbSamples to the c_api.cpp:1352-1374 sampled-set convention reusing the single create_sample_indices draw. |
| `crates/lgbm-dataset/src/dataset.rs` | construct mirroring C++ Construct (filter trivial, bundle) + finish_load immutability | ✓ VERIFIED | construct_bundled:155-248 filters trivial + bundles; construct:89-142 ALSO filters trivial now. finish_load:312 type-state immutability boundary. |
| `crates/lgbm-dataset/tests/default_config_ingest_parity.rs` | default-config parity, fails-before/passes-after, bit-exact | ✓ VERIFIED | Asserts trivial exclusion + num_features()==used_count + group/subfeature parity + bit-exact stored bins; panics on missing golden. |
| `crates/lgbm-dataset/tests/fixtures/default_config_ingest.txt` | C++-Construct golden (trivial dropped) | ✓ VERIFIED | 4 features; f2 trivial with NO group/NO ASSIGN; ASSIGN only for f0/f1/f3; post-shuffle group IDs (f0->g1,f1->g0,f3->g2). |
| `xtask/cpp/bin_capture.cpp` | Emitter modeling C++ Construct | ✓ VERIFIED | EmitDefaultConfigIngest:2156-2348 builds used_features=!is_trivial_, runs FastFeatureBundling, emits group/ASSIGN only for non-trivial. (Note: in-file FastFeatureBundling transcription, not a lib_lightgbm link — see Anti-Patterns / WR-03.) |
| `crates/lgbm-dataset/src/efb.rs` | FastFeatureBundling + FindGroups | ✓ VERIFIED | one_feature_per_group + fast_feature_bundling over used_features; efb_grouping.rs green; now reachable from ingest. |
| `crates/lgbm-dataset/src/metadata.rs` | Metadata + finish_load query weights | ✓ VERIFIED | metadata.rs (4 tests) green. |
| `crates/lgbm-dataset/src/{feature_group,bin/*}.rs` | offset packing, DenseBin/SparseBin layouts | ✓ VERIFIED | byte-layout goldens green (bin_storage_layout.rs). |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|----|--------|---------|
| ingest.rs::finish_from_columns | dataset.rs::construct_bundled | enable_bundle dispatch + trivial filter | ✓ WIRED | The prior WRONG-TARGET (non-bundled construct) is fixed; ingest now routes through the faithful single Construct. |
| ingest.rs::build_efb_samples | efb.rs::fast_feature_bundling | EfbSamples (c_api.cpp:1352-1374 convention) | ✓ WIRED | Sampled-set positions 0..sample_cnt, |v|>kZeroThreshold or NaN, total_sample_cnt=sample_cnt; reuses the single create_sample_indices draw (no second RNG). |
| default_config_ingest_parity.rs | fixtures/default_config_ingest.txt | golden load + from_mat replay | ✓ WIRED | Golden now models C++ Construct (trivial dropped); test panics if missing. |
| ingest.rs::build_mapper | bin_mapper.rs::scaled_filter_cnt | shared helper | ✓ WIRED | GAP-1 fix retained (single source of truth). |

### Data-Flow Trace (Level 4)

| Artifact | Data Variable | Source | Produces Real Data | Status |
|----------|---------------|--------|--------------------|--------|
| FinishedDataset (from_mat) | feature_groups_ stored bins | columns -> build_mapper -> construct_bundled -> push_value -> finish_load | Yes (bit-exact vs C++-Construct golden) | ✓ FLOWING |
| default_config_ingest_parity | mapper.is_trivial_ / feature_to_group / stored bins | from_mat replay of the golden MATRIX | Yes (read from FeatureGroup store, not recomputed) | ✓ FLOWING |

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| Full workspace test suite | `cargo test --workspace` | lgbm-core 14u+33i; lgbm-dataset 75u+22i; oracle 3u+13i; compute/xtask 0; 0 failed | ✓ PASS |
| Default-config ingest parity (trivial-drop + grouping + bit-exact bins) | `default_config_ingest_matches_cpp` | passed | ✓ PASS |
| Trivial feature dropped from store | inspect golden f2 (is_trivial=1, NO ASSIGN) + test assert feature_to_group(2)==-1 | f2 has no ASSIGN; ds drops it | ✓ PASS |
| Goldens compared bit-exact (not 1e-6 tol) | inspect comparator.rs:150-167 (to_bits) + 125-142 | bit-exact / exact-int | ✓ PASS |
| C++ Construct trivial-filter vs Rust port | dataset.cpp:337-343 + used_feature_map_ init -1 vs dataset.rs:169-171,210 | faithful port confirmed | ✓ PASS |

### Probe Execution

| Probe | Command | Result | Status |
|-------|---------|--------|--------|
| (none declared) | — | Phase uses cargo-test parity goldens, not scripts/*/tests/probe-*.sh | ? SKIP (no probes declared) |

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|-------------|-------------|--------|----------|
| DAT-01 | 02-01, 02-06, 02-07 | BinMapper FindBin bit-identical bin boundaries | ✓ SATISFIED | Kernel + scaled_filter_cnt + faithful Construct; is_trivial_/num_bin_/upper match C++ on default config (bit-exact). |
| DAT-02 | 02-02 | DenseBin+SparseBin immutable after finish-load | ✓ SATISFIED | Byte layouts + type-state immutability; stored content now matches C++ Construct (trivial dropped). |
| DAT-03 | 02-03 | Missing-value handling routing | ✓ SATISFIED | missing_edge_cases.rs green. |
| DAT-04 | 02-03 | Categorical encoding + low-freq folding | ✓ SATISFIED | categorical_folding.rs green. |
| DAT-05 | 02-05 | EFB feature grouping | ✓ SATISFIED | efb_grouping.rs green AND construct_bundled reachable from default ingest; group/subfeature parity asserted. |
| DAT-06 | 02-04 | Metadata (labels/weights/init_score/query) | ✓ SATISFIED | metadata.rs tested. |
| DAT-07 | 02-04, 02-06, 02-07 | In-memory dense + CSR/CSC ingestion | ✓ SATISFIED | scaled filter_cnt + faithful single Construct (trivial dropped, enable_bundle honored, EfbSamples to c_api convention); default_config_ingest_parity green. REQUIREMENTS.md SATISFIED claim now matches the codebase. |
| ORA-03 | 02-01 | Per-stage parity tests (bin stage in scope) | ✓ SATISFIED (bin stage) | default-config bin-stage parity discriminates scaled-filter_cnt AND trivial-feature/grouping divergence against a C++-Construct golden; bit-exact. Remaining stages (histogram/split/leaf/predict) belong to later phases per REQUIREMENTS.md traceability. |

All 8 IDs (DAT-01..07, ORA-03) are claimed by plans and present in REQUIREMENTS.md. No orphaned requirements.

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| dataset.rs | 89-142 | Non-bundled `construct` builds groups via `new_single` (force_dense=false), vs C++ Construct force_dense=true (02-REVIEW WR-01) | ⚠ Warning (advisory) | TEST-ONLY path (callers: dataset.rs unit tests). Production ingest routes through construct_bundled (force_dense=true). Not a live determinism divergence; a public API that could mislead a future non-EFB caller. |
| dataset.rs | 127-129, 225 | `.expect()` on EFB partition can panic instead of typed error (02-REVIEW WR-02) | ⚠ Warning (advisory) | Internal-invariant guard; reachable only via an EFB grouping bug. |
| ingest.rs / bin_capture.cpp | 138-162 / 2258-2301 | EfbSamples convention self-consistent but fidelity to real lib_lightgbm unverifiable (external_libs unvendored) (02-REVIEW WR-03) | ⚠ Warning (advisory) | Golden proves Rust==in-file-transcription; on this dense fixture grouping is convention-independent. Environmental limitation, present in all prior phase-2 goldens; later plan should add a convention-observable sparse fixture. |
| default_config_ingest_parity.rs | 290,302-305 | Trivial feature "discovered" via the golden under test; per-feature is_trivial cross-check runs only in the non-trivial branch (02-REVIEW WR-04) | ⚠ Warning (advisory) | Weak independent check on over-marked-trivial; the num_features/group/branch still move in lockstep on a real divergence so the suite still discriminates the CR-01 fix. |
| dataset.rs / parity test | 474-479 / 242-248,178-188 | Misplaced doc comment; assert_eq!(bool,true); ASSIGN dead-store edge (02-REVIEW IN-01/02/03) | ℹ Info | Cosmetic / latent diagnostics; no behavioral impact. |

No unreferenced TBD/FIXME/XXX debt markers found in the phase source files.

### Human Verification Required

None. Every claim is verifiable against the C++ reference
(`LightGBM/src/io/dataset.cpp:325-411`), the Rust source (`dataset.rs`, `ingest.rs`), the
committed golden, the bit-exact comparators (`oracle-harness/src/comparator.rs`), and the
green `cargo test --workspace` run. No visual / real-time / external-service checks needed.

### Gaps Summary

None. The two BLOCKER gaps the prior verification found are independently confirmed
closed in the codebase:

1. **CR-01 (was blocker) — CLOSED.** The public ingest path now routes through
   `construct_bundled` (the faithful single C++ `Dataset::Construct`); trivial features
   are dropped (`used_feature_map_[real] = -1`, no group, no stored bins) and
   `enable_bundle` is honored. Verified against `dataset.cpp:337-343` and the Rust
   `dataset.rs:155-248` / `ingest.rs:199`. The non-bundled `construct` was also fixed to
   filter trivial features symmetrically.

2. **CR-01 masking + WR-01 (was blocker/warning) — CLOSED.** The golden emitter now
   models C++ Construct (trivial dropped); the committed golden has f2 trivial with no
   ASSIGN; the parity test asserts trivial exclusion + grouping + bit-exact stored bins
   and hard-panics on a missing golden.

The phase goal — a binned, immutable columnar dataset whose bin boundaries and bin
assignments are bit-identical to C++ on the default configuration — is achieved.
Remaining items are advisory WARNING/INFO findings from `02-REVIEW.md` (0 BLOCKER): the
test-only non-bundled `construct` force_dense divergence (WR-01), an internal-invariant
`expect()` (WR-02), the unverifiable-here EfbSamples convention fidelity due to unvendored
`external_libs` (WR-03), and a structural weakness in how the test discovers the trivial
feature (WR-04). None of these affect the live determinism path or the phase goal; they
are recommended for a later hardening plan.

---

_Verified: 2026-06-05_
_Verifier: Claude (gsd-verifier)_
