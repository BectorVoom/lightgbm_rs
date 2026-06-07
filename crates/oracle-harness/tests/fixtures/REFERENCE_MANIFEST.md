# Reference Manifest — Phase 6 (Boosting / Objective / Metric)

> The Phase 1–5 golden sets (binning, model/predict, kernel, learner) are
> documented in the generated `crates/oracle-harness/fixtures/REFERENCE_MANIFEST.md`
> (refreshed idempotently by the capture subcommands). This file documents the
> Phase-6 boosting golden set, which is co-located with the boosting fixtures dir
> and is populated as later waves (06-02..06-05) commit each golden.

## Boosting / Objective / Metric Golden Set (Phase 6, D-10..D-13 / L1–L5)

**Status (Wave 3, 06-03):** SPINE + regression_l1 + binary + custom goldens
CAPTURED + replaying. The L1 (`*_gradients`), L2 (`*_score_accumulation`), L5
(`*_spine_end_to_end`) layers are LIVE for `regression`, `regression_l1`, `binary`,
and `custom` (`boosting_parity.rs` un-`#[ignore]`d, passing); L3 `early_stopping`
and L4 `bagging_rng` stay `#[ignore]`d until 06-05. The committed spine cell
(bagging-off / early-stopping-off / boost-from-average-on, regression L2) is the
allowed D-07 collapse.

**Committed goldens (06-02 spine + 06-03 l1/binary/custom), trained on real
`lightgbm==4.6.0` with seed `BOOSTING_ORACLE_SEED = 0x60057000`,
`deterministic=true force_row_wise=true num_threads=1`, identity binning, 10 iters,
num_leaves=4, lr=0.1:**

| File | Layer | Encoding |
|------|-------|----------|
| `regression_spine_model.txt` | L5 model text | `save_model()` `%.17g` |
| `regression_spine_pred.txt`  | L5 predict() | f64 bits |
| `regression_scores.txt`      | L2 per-iter raw score | f64 bits, one line per k |
| `regression_gh_iter1.txt`    | L1 iter-1 g/h | f32 bits (GRAD/HESS) |
| `regression_gh_iterN.txt`    | L1 iter-5 g/h | f32 bits (GRAD/HESS) |
| `regression_metrics.txt`     | L3 per-round l2/rmse | f64 bits |
| `regression_l1_spine_model.txt` | L5 model text — **leaf values are the median RESIDUAL** (RenewTreeOutput, Pitfall 2/3), NOT Newton | `%.17g` |
| `regression_l1_spine_pred.txt`  | L5 predict() | f64 bits |
| `regression_l1_scores.txt`      | L2 per-iter raw score | f64 bits |
| `regression_l1_gh_iter1.txt` / `regression_l1_gh_iterN.txt` | L1 g/h (Sign grad, unit hess) | f32 bits |
| `regression_l1_metrics.txt`     | L3 per-round l1/l2/rmse | f64 bits |
| `binary_spine_model.txt` / `binary_spine_pred.txt` | L5 model text + sigmoid predict() | `%.17g` / f64 bits |
| `binary_scores.txt`             | L2 per-iter raw score | f64 bits |
| `binary_gh_iter1.txt` / `binary_gh_iterN.txt` | L1 sigmoid g/h | f32 bits |
| `binary_metrics.txt`            | L3 per-round binary_logloss/binary_error/auc | f64 bits |
| `custom_spine_model.txt` / `custom_spine_pred.txt` | L5 custom (bfa-off) model text + raw predict() | `%.17g` / f64 bits |
| `custom_scores.txt`             | L2 per-iter raw score | f64 bits |
| `custom_gh_iter1.txt` / `custom_gh_iterN.txt` | L1 custom-closure g/h | f32 bits |
| `custom_metrics.txt`            | L3 per-round l2 (feval) | f64 bits |
| `custom_crossanchor_l2_model.txt` | OBJ-02 cross-anchor: NATIVE `regression`(L2) with **bfa OFF** — its trees bit-match `custom_spine_model.txt` | `%.17g` |

**OBJ-02 cross-anchor (RESOLVED — 06-03 Task 3):** the `custom` run uses an
L2-equivalent closure (`grad = (score - label) as f32`, `hess = 1`) chosen
DELIBERATELY so it is cross-anchorable to the native binary. Because custom forces
`boost_from_average` OFF (C++ `obj == null`), the cross-anchor reference
(`custom_crossanchor_l2_model.txt`) is the native `regression`(L2) cell captured
with `boost_from_average=false` (so init = 0, matching custom). The two model texts
are bit-identical on every tree leaf value (same g/h ⇒ same trees ⇒ same model);
they differ ONLY in the `objective=` line (`custom`/`none` vs `regression`), which
is expected and not part of the leaf-value comparison. **`preds` precision:** the
custom closure receives `&[f64]` raw preds (LightGBM 4.6 passes f64 to a Python
custom objective — verified empirically; a DEVIATION from RESEARCH D-04's "f32
preds", required for the bit-exact cross-anchor; see `custom.rs` doc).

**regression_l1 median-residual renew (RESOLVED — 06-03 Task 1, Pitfall 2/3):** the
l1 leaf values in `regression_l1_spine_model.txt` are the median RESIDUAL of each
leaf's rows (`PercentileFun` alpha=0.5), NOT the learner's Newton output. The Rust
`renew_tree_output` body reproduces them BIT-EXACT; the
`regression_l1_renew_leaf_is_median_residual` test adds a negative control (the l1
leaves DIFFER from the L2 Newton leaves on the same corpus, proving the renew is
load-bearing).

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
| L1–L5 `regression_l1_*` | l1 Sign g/h + median init + median-residual renew leaves | `compare_exact_f64_bits` (L5) / `compare_within` (L1) | 06-03 ✅ |
| L1–L5 `binary_*` | sigmoid g/h + logit init + sigmoid predict + logloss/error/auc | `compare_exact_f64_bits` (L5) / `compare_within` (L1) | 06-03 ✅ |
| `custom_objective` + cross-anchor | D-04 closure g/h + tree; bit-matches native regression(L2 bfa-off) | `compare_exact_f64_bits` | 06-03 ✅ |

### Cross-product collapse (RESEARCH §Cross-Product Collapse Analysis)

The single allowed D-07 collapse: the **spine** golden is the
bagging-off / early-stopping-off / boost-from-average-on cell. The remaining
cells (bagging, early stopping, custom objective, per-class multiclass) are
captured as distinct goldens in later waves; no other cell is collapsed.
