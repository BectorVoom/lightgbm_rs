# Reference Manifest — Phase 6 (Boosting / Objective / Metric)

> The Phase 1–5 golden sets (binning, model/predict, kernel, learner) are
> documented in the generated `crates/oracle-harness/fixtures/REFERENCE_MANIFEST.md`
> (refreshed idempotently by the capture subcommands). This file documents the
> Phase-6 boosting golden set, which is co-located with the boosting fixtures dir
> and is populated as later waves (06-02..06-05) commit each golden.

## Boosting / Objective / Metric Golden Set (Phase 6, D-10..D-13 / L1–L5)

**Status (Wave 2, 06-02):** SPINE goldens CAPTURED + replaying. The L1
(`gradients`), L2 (`score_accumulation`), L5 (`spine_end_to_end`) layers are LIVE
(`boosting_parity.rs` un-`#[ignore]`d, passing); L3 (`early_stopping`), L4
(`bagging_rng`), and `custom_objective` stay `#[ignore]`d until their waves
(06-03/06-05). The committed spine cell (bagging-off / early-stopping-off /
boost-from-average-on, regression L2) is the allowed D-07 collapse.

**Committed spine goldens (06-02), trained on real `lightgbm==4.6.0` with seed
`BOOSTING_ORACLE_SEED = 0x60057000`, `deterministic=true force_row_wise=true
num_threads=1`, identity binning, 10 iters, num_leaves=4, lr=0.1, bfa=true:**

| File | Layer | Encoding |
|------|-------|----------|
| `regression_spine_model.txt` | L5 model text | `save_model()` `%.17g` |
| `regression_spine_pred.txt`  | L5 predict() | f64 bits |
| `regression_scores.txt`      | L2 per-iter raw score | f64 bits, one line per k |
| `regression_gh_iter1.txt`    | L1 iter-1 g/h | f32 bits (GRAD/HESS) |
| `regression_gh_iterN.txt`    | L1 iter-5 g/h | f32 bits (GRAD/HESS) |
| `regression_metrics.txt`     | L3 per-round l2/rmse | f64 bits |

**L2 PRECISION CONTRACT (RESOLVED — Open-Q2/A4):** the per-iter score L2 golden is
**BIT-EXACT** (`compare_exact_f64_bits`), not ~1e-6. The Rust internal `score_`
after k iters equals `predict(raw_score=True, num_iteration=k)` bit-for-bit on this
cell (verified by `lgbm::booster::predict_raw_equals_internal_score_open_q2` AND
`boosting_parity::score_accumulation`). All ~40 downstream cells (06-05) inherit
this bit-exact L2 contract; 06-05 reads it here rather than re-deciding it.

**Capture command:**

```bash
LGBM_CAPTURE_PYTHON=/path/to/venv/bin/python cargo run -p xtask -- boosting-oracle-capture
```

The capture trains a real-binary `lightgbm==4.6.0` (version asserted before
training), writes goldens under the TRACKED
`crates/oracle-harness/tests/fixtures/boosting/`, and is byte-idempotent (verified
empty `git diff` across two runs). It NEVER `git add`s the untracked `LightGBM/`
tree.

### Validation layers (RESEARCH §Validation Architecture)

| Layer | Golden | Comparator | Wave |
|-------|--------|------------|------|
| L1 `gradients` | per-row g/h from the objective | `compare_within(.., ORACLE_TOL)` (~1e-6) | 06-02 |
| L2 `score_accumulation` | per-iter `predict(raw_score=True, num_iteration=k)` | `compare_exact_f64_bits` | 06-02 |
| L3 `early_stopping` | `record_evaluation` / `evals_result` (D-12) | `compare_within(.., ORACLE_TOL)` | 06-05 |
| L4 `bagging_rng` | bagged row indices, RNG-replay (D-13 Option A) | `compare_exact_u32` | 06-05 |
| L5 `spine_end_to_end` | `save_model()` text + `predict()` — `regression_spine_model.txt` | `compare_exact_f64_bits` | 06-02 |
| — `custom_objective` | D-04 closure objective g/h + tree | `compare_within(.., ORACLE_TOL)` | 06-03 |

### Cross-product collapse (RESEARCH §Cross-Product Collapse Analysis)

The single allowed D-07 collapse: the **spine** golden is the
bagging-off / early-stopping-off / boost-from-average-on cell. The remaining
cells (bagging, early stopping, custom objective, per-class multiclass) are
captured as distinct goldens in later waves; no other cell is collapsed.
