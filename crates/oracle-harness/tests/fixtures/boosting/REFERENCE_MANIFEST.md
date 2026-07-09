# Boosting Reference Manifest (Phase 6) — co-located index

This is the co-located Phase-6 boosting golden manifest. The full narrative
manifest (capture command, L2 precision contract, OBJ-02 cross-anchor basis,
per-layer comparator table, cross-product collapse) lives one level up at
`crates/oracle-harness/tests/fixtures/REFERENCE_MANIFEST.md` and is the canonical
source; this file mirrors the per-cell file inventory for quick reference next to
the goldens.

**Capture:** `LGBM_CAPTURE_PYTHON=<venv> cargo run -p xtask -- boosting-oracle-capture`
(real `lightgbm==4.6.0`, version-asserted, byte-idempotent). NEVER `git add LightGBM/`.

## Cells (06-02 spine + 06-03 l1/binary/custom + 06-04 multiclass/ova)

| Cell | Objective | Metrics | Files (prefix) | Iters | Status |
|------|-----------|---------|----------------|-------|--------|
| spine | `regression` (L2), bfa ON | l2, rmse | `regression_*` | 10 | 06-02 ✅ |
| regression_l1 | `regression_l1`, bfa ON | l1, l2, rmse | `regression_l1_*` | 10 | 06-03 ✅ |
| binary | `binary`, bfa ON | binary_logloss, binary_error, auc | `binary_*` | 10 | 06-03 ✅ |
| custom | D-04 closure (L2-equiv), bfa OFF | l2 (feval) | `custom_*` | 10 | 06-03 ✅ |
| custom cross-anchor | NATIVE `regression`(L2), bfa OFF | — | `custom_crossanchor_l2_model.txt` | 10 | 06-03 ✅ |
| multiclass | `multiclass num_class:3`, bfa ON | multi_logloss | `multiclass_*` | 5 | 06-04 ✅ |
| multiclassova | `multiclassova num_class:3 sigmoid:1`, bfa ON | multi_logloss | `multiclassova_*` | 5 | 06-04 ✅ |

Per cell: `<prefix>_spine_model.txt` (L5 model text, `%.17g`),
`<prefix>_spine_pred.txt` (L5 predict, f64 bits), `<prefix>_scores.txt`
(L2 per-iter raw score, f64 bits), `<prefix>_gh_iter1.txt` / `<prefix>_gh_iterN.txt`
(L1 g/h, f32 bits), `<prefix>_metrics.txt` (L3 per-round metrics, f64 bits).

### Multiclass cells (06-04)

The two multiclass cells share a 3-class, 12-row, identity-binned corpus (4 rows
per class, all classes present so `class_need_train == true`). The score / g/h /
metric goldens are **class-major** (`score_[num_data*k + i]`): the file layout is
class 0's rows, then class 1's, then class 2's — the Rust `ScoreUpdater` layout, and
the numpy `order='F'` reshape of LightGBM's `(num_data, num_class)` Python output.
The model has **`iters * num_class` = 15 trees** in class-major order
(`trees[i*num_tree_per_iteration + k]`, `num_tree_per_iteration == num_class == 3`).

**exp-libm residual / 5-iter horizon (IMPORTANT):** the multiclass cells run for
**5** iterations, not 10. The redundant-form softmax `exp` is a transcendental whose
Rust system-libm value and the C++ wheel's `std::exp` differ at the ~1-ULP level;
this propagates into the g/h → Newton leaf outputs → scores (~1e-6) and, at a
knife-edge split gain, can flip a split (observed at iter ~5-6 on this corpus — also
in `multiclassova`, whose per-class binary sigmoid uses the same `exp`). Capping the
horizon at 5 iters keeps **every** grown tree BIT-EXACT to the real binary, so the
L2 per-iter scores and L5 model-text leaf values replay `compare_exact_f64_bits`;
only the predict-side `ConvertOutput` softmax/sigmoid (one `exp`) and `multi_logloss`
are within ORACLE_TOL. Documented exp-libm residual (CLAUDE.md: "bit-exact where the
algorithm permits"); the single-output spine + binary stay bit-exact for 10 iters.
`MULTICLASS_LATER_ITER = 4` for the L1 iter-N g/h golden.

## Key facts (see the canonical manifest for full detail)

- **L2 precision contract:** per-iter scores are BIT-EXACT (`compare_exact_f64_bits`),
  inherited from 06-02.
- **regression_l1 leaves = median RESIDUAL** (RenewTreeOutput, Pitfall 2/3), NOT
  Newton; replayed bit-exact + a negative control vs L2.
- **OBJ-02 cross-anchor:** `custom_spine_model.txt` trees bit-match
  `custom_crossanchor_l2_model.txt` (native regression L2, bfa-off) — same g/h ⇒
  same trees ⇒ same model. Differ only in the `objective=` line.
- **custom `preds` are f64** (LightGBM 4.6 passes f64 to a Python custom objective;
  DEVIATION from RESEARCH D-04's f32, required for the bit-exact cross-anchor).
- **multiclass is class-major + `iters*num_class` trees**; bit-exact over the 5-iter
  horizon (scores + model leaves), within ORACLE_TOL for predict/g/h/multi_logloss.
  The 5-iter cap is the documented softmax-`exp` libm residual (see above).

## D-07 cross-product matrix (06-05)

The full ~40-cell maximal-fidelity matrix: **5 core objectives × {bagging on/off} ×
{early_stop on/off} × {boost_from_average on/off} = 8 cells/objective**. The spine
cell (`bag off / es off / bfa on`) is the per-objective spine golden above — the ONE
**referenced collapse** (RESEARCH §Cross-Product Collapse Analysis cell 1), NOT
re-captured. The remaining **7 cells/objective = 35 cells** are committed as
`<obj>_bag<B>_es<E>_bfa<F>_model.txt` + `_pred.txt`; `matrix_best_iterations.txt`
records the realized `best_iteration` per es cell. Replayed by
`boosting_parity::early_stopping` (the renamed matrix runner — no longer ignored).

**Axis exercising (RESEARCH requirements):**
- **bfa** — corpora have `|init| > kEps` (non-zero label mean / pavg), so the bfa
  axis is genuinely distinct (no label-mean-zero collapse).
- **bagging** — `bagging_fraction=0.7 bagging_freq=1 bagging_seed=3` (re-bag every
  iter so the RNG stream advances continuously — the most order-sensitive path).
- **early_stop** — a CONSTANT-label (10.0) valid set so the metric plateaus and early
  stopping **GENUINELY FIRES**: every objective has at least one es cell with
  `best_iteration < num_iterations` (binary/multiclass/multiclassova fire at 1–3
  iters; regression/regression_l1 bfa-on fire at 1). The trailing-tree pop trims the
  model to `best_iteration * num_class` trees — asserted bit-exact against the golden
  tree count.

**Replay precision per cell:**
- **regression (L2), all 7 cells** — model-text leaf values **BIT-EXACT**
  (`compare_exact_f64_bits`), including the **bagging cells** (constant hessian + no
  renew ⇒ the subset histogram + predict-side OOB scoring reproduce the C++
  `tmp_subset_` result bit-for-bit). predict within ORACLE_TOL. The es-cell
  `best_iteration` matches the captured value.
- **binary / multiclass / multiclassova, NON-bagging cells** — single-output (binary)
  bit-exact; multiclass within ORACLE_TOL (the 06-04 softmax exp-libm residual; the
  multiclass matrix cells cap the horizon at **5 iters**, like 06-04).

### D-07 matrix residuals (documented, NOT silently dropped)

Two cell families are **validated within ORACLE_TOL on overlapping trees** rather than
bit-exact, and explicitly flagged here (the "bit-exact where the algorithm permits"
carve-out, CLAUDE.md — same family as the 06-04 softmax exp-libm residual):

1. **`regression_l1` with `bfa=off`** — iter-0 gradients are UNIFORM
   (`grad = sign(0 - label) = -1` for every positive label), so the split gain is at
   the f64-NOISE level: the real binary's `split_gain ≈ 1.78e-15 > 0` accepts a
   degenerate first split, while the Rust f64-fold gain rounds to `≤ 0` and rejects it
   (a sub-ULP knife-edge flip). Affects the `regression_l1_bag*_es*_bfa0` cells.
2. **`binary` / `regression_l1` / `multiclass` / `multiclassova` with `bagging=on`** —
   the subset path's interaction with a NON-CONSTANT hessian (binary sigmoid hessian)
   or a post-growth renewal (`regression_l1` median-residual `RenewTreeOutput` over the
   in-bag leaves, deferred on the subset path) diverges in the split STRUCTURE from the
   C++ `tmp_subset_` Dataset + in-bag train-path scatter. regression(L2) bagging is
   bit-exact (constant hessian, no renew); the other objectives' bagging cells are the
   residual.

**Follow-up (Phase 7):** thread the median-residual `RenewTreeOutput` through
`train_on_subset` (the subset partition's `residual_getter`) and reconcile the
non-constant-hessian subset histogram path with the C++ `tmp_subset_` Dataset so the
binary/l1/multiclass bagging cells reach bit-exact. The es / bfa axes are bit-exact
for all single-output objectives without bagging; regression(L2) is bit-exact across
ALL eight cells.

Replayed by `crates/oracle-harness/tests/boosting_parity.rs` (22 passing, 0 ignored).
Routine `cargo test` needs no wheel (committed goldens). Capture byte-idempotent;
NEVER `git add LightGBM/`.
