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

Replayed by `crates/oracle-harness/tests/boosting_parity.rs` (20 passing / 2
ignored for 06-05). Routine `cargo test` needs no wheel (committed goldens).
