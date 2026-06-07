---
phase: 07
slug: parity-completing-variants
status: verified
threats_open: 0
asvs_level: 1
created: 2026-06-07
---

# Phase 07 — Security

> Per-phase security contract: threat register, accepted risks, and audit trail.
> Phase 07 adds boosting variants (GOSS/DART/RF), categorical splits, the
> remaining objectives/metrics, prediction modes (SHAP, early stop), and
> tree-control constraints to an **offline numerical library**. There is no
> network, auth, session, crypto, or PII surface. The real attack surfaces are
> (1) untrusted forced-splits JSON, (2) untrusted model-text deserialization,
> (3) config-parameter range validation, (4) length/bounds gates at crate
> boundaries, and (5) the supply-chain oracle-capture step.

---

## Trust Boundaries

| Boundary | Description | Data Crossing |
|----------|-------------|---------------|
| Public train/predict facade (`lgbm`) | Developer-supplied `Config` + `DenseCorpus` cross into the engine as typed slices | Developer config + numeric feature/label data (not untrusted/external) |
| Forced-splits JSON (`forced_splits_filename`, ADV-03) | An on-disk JSON document parsed at train time | Untrusted JSON (real tampering surface) |
| Model-text deserialization (`input_model`, refit/continue, ADV-06) | A serialized model string parsed back into the in-memory ensemble | Untrusted model text (real tampering surface) |
| Oracle-capture step (xtask) | `pip install lightgbm==4.6.0` then capture goldens from the reference wheel | Third-party package (supply-chain); `LightGBM/` reference tree never git-added |

---

## Threat Register

| Threat ID | Category | Component | Disposition | Mitigation | Status |
|-----------|----------|-----------|-------------|------------|--------|
| T-07-01-SC | Tampering (supply-chain) | `pip install lightgbm==4.6.0` capture | mitigate | Version-asserted in xtask before any golden emitted (`xtask/src/main.rs:111`, `subset_determinism_capture.py`); `LightGBM/` untracked | closed |
| T-07-02-01 | Tampering→NaN | objective `get_gradients` malformed length / `alpha`/`fair_c` ≤0 | mitigate | V5 length gate `regression.rs:387-405`; quantile alpha `0<α<1` typed reject `:214-221` | closed |
| T-07-02-SC | Tampering (supply-chain) | pip capture | mitigate | `xtask/src/main.rs:98`; `boosting_oracle_capture.py:1027` version-assert | closed |
| T-07-03-01 | Tampering→NaN | poisson/gamma/tweedie/xentropy out-of-range labels into exp/log | mitigate | `regression.rs:333-363` (label≥0 + Σ≠0 `LabelRange`); `xentropy.rs:91-103` (label∈[0,1]) | closed |
| T-07-03-02 | Tampering | `tweedie_variance_power` / `poisson_max_delta_step` out-of-range | mitigate | `config/set.rs:347-352` (`>0`; `∈[1,2)`) typed Result | closed |
| T-07-03-SC | Tampering (supply-chain) | pip capture | mitigate | `xtask/src/main.rs:98`; `boosting_oracle_capture.py:1027` | closed |
| T-07-04-01 | Tampering→NaN | metric eval mismatched score/label lengths / out-of-range prob | mitigate | length gates `metric/regression.rs:204-209`, `binary.rs:82-84`; K_EPSILON clamp `binary.rs:165-173`, `xentropy.rs:196-200` | closed |
| T-07-04-SC | Tampering (supply-chain) | pip capture | mitigate | `xtask/src/main.rs:177`; `metric_oracle_capture.py:200` | closed |
| T-07-05-01 | Tampering | `top_rate`/`other_rate` out of range (top+other>1) | mitigate | `sample_strategy.rs:592-607` (top+other≤1, both>0); `config/set.rs:201-207` | closed |
| T-07-05-SC | Tampering (supply-chain) | pip capture | mitigate | `xtask/src/main.rs:126`; `goss_oracle_capture.py` | closed |
| T-07-06-01 | Tampering | `drop_rate`/`skip_drop` out of [0,1] | mitigate | `config/set.rs:187-195` (∈[0,1]) typed Result | closed |
| T-07-06-SC | Tampering (supply-chain) | pip capture | mitigate | `xtask/src/main.rs:143` | closed |
| T-07-07-01 | Tampering | RF config with no objective or no bagging/feature_fraction | mitigate | `gbdt.rs:878-895` (objective + randomization CHECKs → `RfConfig` before any tree grows) | closed |
| T-07-07-SC | Tampering (supply-chain) | pip capture | mitigate | `xtask/src/main.rs:159` | closed |
| T-07-08-01 | Tampering | malformed model-text (cat_threshold/cat_boundaries OOB) on load | mitigate | `model/tree.rs:909,920,927-948` (cat bitset + child/leaf index bounds, split_feature≥0) | closed |
| T-07-08-02 | Tampering | `cat_l2`/`max_cat_threshold`/`min_data_per_group` out of range | mitigate | `config/set.rs:209-216` typed Result | closed |
| T-07-08-SC | Tampering (supply-chain) | pip capture | mitigate | `xtask/src/main.rs:190`; `categorical_oracle_capture.py:171` | closed |
| T-07-09-01 | Tampering→NaN | non-monotonic / OOB `query_boundaries` per-query iteration | mitigate | `metric/rank.rs:38-66` + `objective/rank.rs:55-80` (`validate_query_boundaries`: start=0, monotone, ends at num_data) | closed |
| T-07-09-02 | Tampering→NaN | out-of-range rank labels feeding DCG/sigmoid | mitigate | `objective/rank.rs:86-97` (label≥0); `dcg_calculator.rs:112+` (CheckLabel non-neg int < gain size) | closed |
| T-07-09-SC | Tampering (supply-chain) | pip capture | mitigate | `xtask/src/main.rs:216` | closed |
| T-07-10-01 | Tampering | `predict_contrib` mismatched column count / OOB feature index | mitigate | `model/predict.rs:42-50` (check_cols V5 gate; called `:111,179,257,579`); CSR/CSC index bounds `:193-195,271-273` | closed |
| T-07-10-02 | DoS | TreeSHAP `unique_path` buffer sizing on a deep tree | **accept** | Buffer sized `(max_depth()+1)` `model/tree.rs:333`; `max_depth()` recomputed from load-validated node structure `:297-314`. See Accepted Risks Log. | closed |
| T-07-10-SC | Tampering (supply-chain) | pip capture | mitigate | `xtask/src/main.rs:230` | closed |
| T-07-11-01 | Tampering/DoS | untrusted forced-splits JSON (malformed schema, OOB feature/threshold, deep nesting) | mitigate | `forced_splits.rs:22` (`MAX_FORCED_DEPTH=64`), `:147-148,280-282` (depth bound), `:292-294` (feature OOB), `:300-302` (non-finite threshold); typed `ForcedSplitError`, no panics | closed |
| T-07-11-02 | Tampering (improper input validation) | `monotone_constraints` / `cegb_penalty_feature_coupled` / `cegb_penalty_feature_lazy` wrong length vs num_features | mitigate | `lgbm/error.rs:57-76` (`InvalidConstraintLength`); `lgbm/booster.rs:414-438` (validate all three before any tree grows); test `booster.rs:969-1039`. Mirrors `gbdt.cpp:58` + `cost_effective_gradient_boosting.hpp:47-60`. **Fixed in commit `5ea38c3`.** | closed |
| T-07-11-SC | Tampering (supply-chain) | a NEW JSON crate for ADV-03 | mitigate | Hand-rolled recursive-descent parser `forced_splits.rs:115-269`; **no new crate added** (workspace Cargo.toml has no serde_json; lockfile entry is transitive-only). Package-legitimacy gate moot. | closed |
| T-07-11-SC2 | Tampering (supply-chain) | pip capture | mitigate | `xtask/src/main.rs:203`; `constraints_oracle_capture.py:187` | closed |
| T-07-12-01 | Tampering | malformed `input_model` on refit/continue-load (OOB node/leaf indices, bad cat bitset) | mitigate | Reuses Phase-3 model_text load bounds-validation `model/tree.rs:764-979` (same parse/load path as T-07-08-01); typed `ModelError::MalformedModel` | closed |
| T-07-12-02 | Tampering | `refit_decay_rate` out of [0,1] | mitigate | `config/set.rs:256-258` (∈[0,1]) typed Result | closed |
| T-07-12-SC | Tampering (supply-chain) | pip capture | mitigate | `xtask/src/main.rs:243`; `advanced_oracle_capture.py:133` | closed |

*Status: open · closed*
*Disposition: mitigate (implementation required) · accept (documented risk) · transfer (third-party)*

---

## Accepted Risks Log

| Risk ID | Threat Ref | Rationale | Accepted By | Date |
|---------|------------|-----------|-------------|------|
| AR-07-01 | T-07-10-02 | TreeSHAP `unique_path` buffer is sized from `max_depth()+1`, where `max_depth()` is recomputed from the tree's own load-validated, bounded node structure (`model/tree.rs:333`, `:297-314`). Tree depth is never untrusted input — the model is produced/loaded via the Phase-3-validated path, itself bounds-checked. No unbounded allocation is reachable; verified present. Disposition `accept` per 07-10-PLAN. | appservice27@gmail.com (`/gsd-secure-phase`) | 2026-06-07 |

---

## Security Audit Trail

| Audit Date | Threats Total | Closed | Open | Run By |
|------------|---------------|--------|------|--------|
| 2026-06-07 | 28 | 27 | 1 | gsd-security-auditor (initial — T-07-11-02 open) |
| 2026-06-07 | 28 | 28 | 0 | gsd-security-auditor (re-verify after fix `5ea38c3`) |

---

## Sign-Off

- [x] All threats have a disposition (mitigate / accept / transfer)
- [x] Accepted risks documented in Accepted Risks Log
- [x] `threats_open: 0` confirmed
- [x] `status: verified` set in frontmatter

**Approval:** verified 2026-06-07
