---
phase: 03-tree-model-model-text-i-o-predict-parity
verified: 2026-06-05T00:00:00Z
status: passed
score: 4/4 success criteria verified (6/6 requirement IDs satisfied)
overrides_applied: 0
re_verification:
  previous_status: none
  note: "Initial verification (no prior VERIFICATION.md)"
warnings:
  - id: CR-01
    summary: "feature_importance_split_count omits the C++ `split_gain > 0` guard (ensemble.rs:108-119)."
    disposition: out-of-corpus-latent
    evidence: "All 5 committed fixtures contain 0 split_gain entries <= 0 (scanned 1439 split-gain values across regression/binary/multiclass/categorical/subrange). The DAT-09 byte-exact round-trip therefore passes on every committed model. The divergence can only manifest on a model with a zero-gain split (forced/monotone splits) — none exist in any Phase-3 fixture. NOT a Phase-3 goal gap; recommend closing before Phase 7 (monotone/forced-split models, ADV-01/ADV-03)."
  - id: CR-02
    summary: "Leaf-index prediction ignores start_iteration/num_iteration (predict.rs predict_row_leaf:350, hard-codes full num_iteration); public predict_leaf_index_* exposes no range params."
    disposition: out-of-scope-combination
    evidence: "Phase-3 success criterion 4 + PRD-06 scope sub-range to RAW scores; the committed subrange.txt golden contains only RAW slices (4 SLICE headers, f64 values) — NO leaf-index slices. subrange/leaf.txt is full-range only. PRD-03 (pred_leaf) is verified full-range exact; PRD-06 (sub-range) is verified for raw. The PRD-03 x PRD-06 COMBINATION is neither a listed requirement nor exercised by any fixture. 03-04 SUMMARY explicitly defers leaf-index sub-range. NOT a Phase-3 goal gap; recommend threading start/num through the leaf path when a sub-range leaf golden lands."
  - id: CR-03
    summary: "average_output (RF mean) is parsed + round-tripped (model_text.rs:231) but never applied at predict time; load() accept-and-ignores it rather than rejecting."
    disposition: out-of-scope-followup
    evidence: "All 5 committed fixtures are GBDT (0 `average_output` lines present). RF / average_output is requirement BST-06, scoped to Phase 7 (ROADMAP line 183, success criterion 1). No Phase-3 requirement ID (DAT-08/09, PRD-01/02/03/06) covers RF averaging. NOT a Phase-3 goal gap. Robustness caveat: a future RF model would be silently mis-predicted by a factor of num_iteration; recommend rejecting average_output at load (typed ModelError) until Phase 7 implements it."
  - id: REQ-DRIFT
    summary: "REQUIREMENTS.md still lists PRD-06 as `[ ]` / 'Pending' (lines 80, 170) although Phase 3 delivers and verifies it."
    disposition: documentation-drift
    evidence: "predict_subrange.rs passes against subrange.txt for 4 slices (0,10)/(0,5)/(5,-1)/(1,1) within ORACLE_TOL; init_predict is a parity-asserted gbdt.h:426-435 port. PRD-06 IS satisfied in code. Recommend flipping the REQUIREMENTS.md checkbox/status to Complete."
---

# Phase 3: Tree Model + Model Text I/O + Predict Parity — Verification Report

**Phase Goal:** Load a C++-trained model and predict identically — prediction parity proven independently of (and before) any training code.
**Mode:** mvp (goal is outcome-phrased; verified against the 4 explicit ROADMAP Success Criteria, which form the concrete contract)
**Verified:** 2026-06-05
**Status:** passed
**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths (ROADMAP Success Criteria)

| # | Truth (Success Criterion) | Status | Evidence |
| - | ------------------------- | ------ | -------- |
| 1 | Load a C++-trained `.txt` model and produce raw-score predictions within ~1e-6 (f32) of the C++ reference on the deterministic CPU path. | ✓ VERIFIED | `predict_raw_parity.rs::regression_raw_parity_dense_csr_csc` PASSES against committed `regression/raw.txt` on dense + CSR + CSC. Negative control: corrupting one golden u64 → test FAILS with `abs_diff=0.615 > tol=0.000001` (assertion is load-bearing at 1e-6, fixture restored clean). Accumulation is `Vec<f64>` (ensemble.rs:92-98), f32 cast only at output boundary. Predict path is raw (no BinMapper/construct — predict.rs doc-confirmed, 0 calls). |
| 2 | Transformed predictions (ConvertOutput sigmoid/softmax) and leaf-index predictions (pred_leaf) match the C++ reference. | ✓ VERIFIED | `predict_transform.rs` (binary sigmoid, multiclass softmax w/ max-subtraction, regression identity) all PASS vs `transformed.txt` within ORACLE_TOL. `predict_leaf_parity.rs` (regression 10-col + multiclass 30-col = 10 iter × 3 class stride) PASS exact via `compare_exact_u32`. softmax uses Common::Softmax max-subtraction (objective.rs); objective params parsed from `objective=` line, not a Config. |
| 3 | Rust writer emits exact LightGBM text schema (tree/leaf/metadata) incl %.17g, and load→predict→write→reload round-trip is byte-stable. | ✓ VERIFIED | `model_text_roundtrip.rs` PASSES byte-exact (`compare_exact_bytes`) for ALL 4 model corpora (regression/binary/multiclass/categorical incl. cat_boundaries/cat_threshold lines). %.17g linchpin: `format::golden_matches_formatter` runs (NOT skipped) against committed 31-line `format_golden.txt` from authoritative C++ fmt; `tree::round_trip_parse_to_string_byte_identical` + real-block round-trip pass. All 5 model.txt begin `tree\nversion=v4`. format_g17 round-trips bit-exact. |
| 4 | Sub-range prediction (start_iteration / num_iteration) returns the C++-matching slice of the ensemble. | ✓ VERIFIED | `predict_subrange.rs::subrange_raw_parity_dense` PASSES for 4 slices (0,10)/(0,5)/(5,-1)/(1,1) — covering -1==all, bounded count, non-zero start — within ORACLE_TOL vs `subrange.txt`. `init_predict` is a parity-asserted port of gbdt.h:426-435 (ensemble.rs); inline battery covers clamp/over-range/i32::MAX/MIN without panic. |

**Score:** 4/4 truths verified

### Required Artifacts

| Artifact | Expected | Status | Details |
| -------- | -------- | ------ | ------- |
| `crates/lgbm-model/src/format.rs` | %.17g / {:g} printf-faithful formatters | ✓ VERIFIED | format_g17/format_g6 present; golden + battery tests pass; no ryu/to_string/{:.17e}. |
| `crates/lgbm-model/src/tree.rs` | Faithful parallel-array Tree + predict/get_leaf/ToString/parse | ✓ VERIFIED | 60 lib tests incl. decode, numerical_decision NaN/zero routing, single-leaf, malformed→Err, byte-exact round-trip. |
| `crates/lgbm-model/src/ensemble.rs` | GbdtModel + predict_raw f64 loop + init_predict | ✓ VERIFIED | f64 accumulation, ntpi stride, init_predict parity battery. |
| `crates/lgbm-model/src/model_text.rs` | LoadModelFromString + byte-exact SaveModelToString | ✓ VERIFIED | Verbatim metadata preserved; tree_sizes checked arithmetic; round-trip byte-exact (4 corpora). |
| `crates/lgbm-model/src/objective.rs` | Core ConvertOutput shim parsed from objective= line | ✓ VERIFIED | 4 transforms + softmax max-subtraction; non-core objective → ModelError. |
| `crates/lgbm-model/src/predict.rs` | Dense/CSR/CSC raw + transformed + leaf + sub-range driver | ✓ VERIFIED | Raw path (no BinMapper); ShapeMismatch on bad input; _range variants thread start/num for raw. |
| `crates/lgbm-model/tests/fixtures/models/` | 5 committed C++ corpora + format_golden | ✓ VERIFIED | 23 git-tracked files (5 corpora × model/raw/transformed/leaf + subrange.txt + format_golden + .gitkeep); all model.txt = `tree\nversion=v4`; byte-idempotent (empty git diff). |
| `xtask/src/main.rs model-capture` | Byte-idempotent golden capture | ✓ VERIFIED | Subcommand present; capture provenance (lightgbm 4.6.0, deterministic params) recorded + human-approved in REFERENCE_MANIFEST.md. |

### Key Link Verification

| From | To | Via | Status |
| ---- | -- | --- | ------ |
| predict.rs | tree.rs Tree::predict | ensemble loop accumulates Tree::predict | ✓ WIRED (raw parity test passes) |
| model_text.rs save | format.rs format_g17 | per-tree floats via %.17g | ✓ WIRED (byte-exact round-trip passes) |
| predict.rs predict | objective.rs convert | transformed applies ConvertOutput | ✓ WIRED (transform test passes) |
| predict_subrange.rs | ensemble init_predict | start/num threaded into slice | ✓ WIRED (subrange test passes 4 slices) |
| root Cargo.toml | crates/lgbm-model | workspace membership | ✓ WIRED (builds in workspace) |

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
| -------- | ------- | ------ | ------ |
| Full lgbm-model suite runs, non-skipping | `cargo test -p lgbm-model` | 60 lib + 4 roundtrip + 2 leaf + 1 raw + 1 subrange + 3 transform; 0 failed, 0 ignored, 0 filtered | ✓ PASS |
| Integration tests NOT silently skipping | `cargo test ... -- --nocapture \| grep SKIP` | No SKIP messages emitted | ✓ PASS |
| Parity assertions are load-bearing (1e-6) | Corrupt regression/raw.txt → run | Test FAILS `abs_diff=0.615 > tol=1e-6`; fixture restored clean | ✓ PASS |
| Workspace green (no regression) | `cargo test --workspace` | All crates pass; 0 failures | ✓ PASS |
| Capture byte-idempotent | `git status --porcelain fixtures/` | Empty | ✓ PASS |

### Probe Execution

No conventional `scripts/*/tests/probe-*.sh` declared for this phase; the D-06 layered integration tests are the runnable parity probes and all executed (above). N/A.

### Requirements Coverage

| Requirement | Source Plan(s) | Description | Status | Evidence |
| ----------- | -------------- | ----------- | ------ | -------- |
| DAT-08 | 03-01/02/03 | Model text format read | ✓ SATISFIED | load() of all 4 model corpora succeeds; round-trip tests pass. |
| DAT-09 | 03-01/02/03 | Model text format write (%.17g) | ✓ SATISFIED | Byte-exact round-trip for 4 corpora; format_golden test passes. |
| PRD-01 | 03-02/04 | Raw score prediction | ✓ SATISFIED | predict_raw_parity dense/CSR/CSC within 1e-6. |
| PRD-02 | 03-03 | Transformed prediction | ✓ SATISFIED | predict_transform binary/multiclass/regression within ORACLE_TOL. |
| PRD-03 | 03-03 | Leaf index prediction | ✓ SATISFIED | predict_leaf_parity exact (regression + multiclass stride). |
| PRD-06 | 03-04 | Sub-range prediction | ✓ SATISFIED | predict_subrange 4 slices within ORACLE_TOL. (REQUIREMENTS.md still shows stale "Pending" — see REQ-DRIFT warning.) |

All 6 declared requirement IDs accounted for and satisfied. No orphaned requirements (ROADMAP Phase 3 Requirements == plan frontmatter union == {DAT-08, DAT-09, PRD-01, PRD-02, PRD-03, PRD-06}).

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
| ---- | ---- | ------- | -------- | ------ |
| (none) | — | TBD/FIXME/XXX debt markers | — | Zero debt markers in phase source. |
| (none) | — | TODO/HACK/PLACEHOLDER | — | None found. |
| src/*.rs production | — | unwrap/expect/panic | ℹ️ Info | All in `#[cfg(test)]` blocks or format.rs internal helpers with documented invariants (Rust `{:e}` always contains 'e'). No panic on caller input in the validated predict/load path. |

### Code-Review Critical Findings — Disposition

The 03-REVIEW.md flagged 3 Critical divergences found by source comparison (not by a red test). Each was investigated against the committed corpus:

| ID | Finding | Phase-3 Gap? | Verdict |
| -- | ------- | ------------ | ------- |
| CR-01 | feature_importance_split_count missing `split_gain > 0` guard | NO | Out-of-corpus latent. 0 of 1439 split-gain values across all 5 fixtures are <= 0; DAT-09 byte-exact round-trip genuinely passes. Can only break on zero-gain-split models (forced/monotone, Phase 7). Recorded as WARNING for Phase 7. |
| CR-02 | Leaf-index ignores sub-range | NO | Out-of-scope combination. subrange golden has only RAW slices; PRD-03 (full-range leaf) and PRD-06 (sub-range raw) each verified independently; their combination is not a Phase-3 requirement nor a fixture. Explicitly deferred in 03-04 SUMMARY. Recorded as WARNING. |
| CR-03 | average_output (RF mean) never applied | NO | Out-of-scope follow-up. All fixtures GBDT (0 average_output lines); RF is BST-06 / Phase 7. Robustness caveat: load() accept-and-ignores rather than rejecting. Recorded as WARNING with recommended hardening. |

### Human Verification Required

None outstanding. The single Manual-Only Verification from 03-VALIDATION.md (golden-capture provenance: C++ `.txt` numerically identical to lib_lightgbm) was discharged at execution time — capture path B (pip lightgbm 4.6.0) was human-approved and recorded in REFERENCE_MANIFEST.md ("Model / Predict Golden Set" + "Capture-path resolution: PATH B ... human-approved").

### Gaps Summary

No gaps blocking the phase goal. All 4 ROADMAP Success Criteria are observably true in the codebase, proven by running, non-skipping, load-bearing parity tests against the committed C++-trained golden corpus (negative control confirmed the 1e-6 assertions fire). All 6 requirement IDs are satisfied.

The 3 Critical code-review findings are all latent/out-of-scope relative to the Phase-3 corpus and requirement set — verified by direct fixture inspection (zero zero-gain splits, zero average_output models, no sub-range leaf golden). They are recorded as WARNINGS with Phase-7 follow-up recommendations, not gaps. One documentation-drift note (PRD-06 stale "Pending" in REQUIREMENTS.md) is informational.

---

_Verified: 2026-06-05_
_Verifier: Claude (gsd-verifier)_
