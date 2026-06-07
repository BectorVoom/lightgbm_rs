# Boosting Reference Manifest (Phase 6) — co-located index

This is the co-located Phase-6 boosting golden manifest. The full narrative
manifest (capture command, L2 precision contract, OBJ-02 cross-anchor basis,
per-layer comparator table, cross-product collapse) lives one level up at
`crates/oracle-harness/tests/fixtures/REFERENCE_MANIFEST.md` and is the canonical
source; this file mirrors the per-cell file inventory for quick reference next to
the goldens.

**Capture:** `LGBM_CAPTURE_PYTHON=<venv> cargo run -p xtask -- boosting-oracle-capture`
(real `lightgbm==4.6.0`, version-asserted, byte-idempotent). NEVER `git add LightGBM/`.

## Cells (06-02 spine + 06-03 l1/binary/custom)

| Cell | Objective | Metrics | Files (prefix) | Status |
|------|-----------|---------|----------------|--------|
| spine | `regression` (L2), bfa ON | l2, rmse | `regression_*` | 06-02 ✅ |
| regression_l1 | `regression_l1`, bfa ON | l1, l2, rmse | `regression_l1_*` | 06-03 ✅ |
| binary | `binary`, bfa ON | binary_logloss, binary_error, auc | `binary_*` | 06-03 ✅ |
| custom | D-04 closure (L2-equiv), bfa OFF | l2 (feval) | `custom_*` | 06-03 ✅ |
| custom cross-anchor | NATIVE `regression`(L2), bfa OFF | — | `custom_crossanchor_l2_model.txt` | 06-03 ✅ |

Per cell: `<prefix>_spine_model.txt` (L5 model text, `%.17g`),
`<prefix>_spine_pred.txt` (L5 predict, f64 bits), `<prefix>_scores.txt`
(L2 per-iter raw score, f64 bits), `<prefix>_gh_iter1.txt` / `<prefix>_gh_iterN.txt`
(L1 g/h, f32 bits), `<prefix>_metrics.txt` (L3 per-round metrics, f64 bits).

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

Replayed by `crates/oracle-harness/tests/boosting_parity.rs` (12 passing / 2
ignored for 06-05). Routine `cargo test` needs no wheel (committed goldens).
