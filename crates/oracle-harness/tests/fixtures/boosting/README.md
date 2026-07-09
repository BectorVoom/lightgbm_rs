# Boosting Golden Fixtures (Phase 6)

This directory holds the committed GBDT-spine / objective / metric reference
goldens replayed by `crates/oracle-harness/tests/boosting_parity.rs`.

**Populated by:** `cargo run -p xtask -- boosting-oracle-capture`
(the real-binary `lightgbm==4.6.0` capture, wave 2+ — stubbed in 06-01).

Until the boosting loop lands (06-02+), the `boosting_parity.rs` tests are
`#[ignore]`d with an explicit `MISSING — implemented in wave N` reason; this
file keeps the tracked fixtures directory present (Nyquist scaffold).

Planned layered goldens (RESEARCH §Validation Architecture, L1–L5):

- `regression_spine_model.txt` — L5 `save_model()` text + `predict()` (the spine
  end-to-end golden named by `spine_end_to_end`).
- per-iter raw scores (L2), per-row g/h (L1), eval-history (L3, D-12), bagged
  indices (L4, D-13 RNG-replay) — added as each wave commits them.

NEVER `git add LightGBM/`. All goldens live under this tracked oracle-harness
fixtures dir.
