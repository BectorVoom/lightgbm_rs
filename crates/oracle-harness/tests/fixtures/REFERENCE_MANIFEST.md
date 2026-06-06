# Reference Manifest — Phase 6 (Boosting / Objective / Metric)

> The Phase 1–5 golden sets (binning, model/predict, kernel, learner) are
> documented in the generated `crates/oracle-harness/fixtures/REFERENCE_MANIFEST.md`
> (refreshed idempotently by the capture subcommands). This file documents the
> Phase-6 boosting golden set, which is co-located with the boosting fixtures dir
> and is populated as later waves (06-02..06-05) commit each golden.

## Boosting / Objective / Metric Golden Set (Phase 6, D-10..D-13 / L1–L5)

**Status (Wave 0, 06-01):** SCAFFOLD only. The `boosting_parity.rs` tests are
`#[ignore]`d with an explicit `MISSING — implemented in wave N` reason until the
GBDT spine loop lands. Goldens below are added as each wave commits them.

**Capture command (stubbed in 06-01):**

```bash
LGBM_CAPTURE_PYTHON=/path/to/venv/bin/python cargo run -p xtask -- boosting-oracle-capture
```

The real capture (wave 2+) trains a real-binary `lightgbm==4.6.0` (version
asserted before training), writes goldens under the TRACKED
`crates/oracle-harness/tests/fixtures/boosting/`, and is byte-idempotent. It
NEVER `git add`s the untracked `LightGBM/` tree.

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
