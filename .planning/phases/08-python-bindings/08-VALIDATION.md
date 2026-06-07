---
phase: 8
slug: python-bindings
status: draft
nyquist_compliant: false
wave_0_complete: false
created: 2026-06-07
---

# Phase 8 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.
> Derived from `08-RESEARCH.md` § Validation Architecture. Discipline reuses the
> existing `oracle-harness` + `xtask` capture pattern (Python-vs-Rust-vs-C++),
> extended with **A/B parity against the side-by-side real `lightgbm` 4.6** (D-11).

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Rust framework** | `cargo test` + `crates/oracle-harness/tests/*` (existing comparator `compare_within(ORACLE_TOL)`) |
| **Python framework** | `pytest` (mirror official `tests/python_package_test/`) |
| **Config file** | none yet for Python — **Wave 0** adds the pytest tree under `crates/lgbm-python/python/tests/` + CI invocation |
| **Quick run command** | `cargo test -p lgbm` (facade bridge) · `pytest -x crates/lgbm-python/python/tests/test_smoke.py` |
| **Full suite command** | `cargo test --workspace` + `cargo test -p oracle-harness` + `pytest crates/lgbm-python/python/tests/` |
| **Estimated runtime** | ~120 seconds (Rust workspace) + ~60 seconds (Python A/B suite) |

---

## Sampling Rate

- **After every task commit:** `cargo test -p lgbm` (Rust bridge tasks) or `pytest -x <relevant test file>` (Python tasks)
- **After every plan wave:** `cargo test --workspace` + `pytest crates/lgbm-python/python/tests/`
- **Before `/gsd-verify-work`:** full Rust workspace + oracle-harness green AND the Python A/B parity suite green vs side-by-side real `lightgbm` 4.6, within the numerical contract (~1e-6 absolute; CPU f64-fold anchor bit-exact where the algorithm permits)
- **Max feedback latency:** ~120 seconds

---

## Per-Task Verification Map

> Task IDs are placeholders until plans are written; the planner must bind each
> requirement below to concrete `{N}-PP-TT` task IDs and `<automated>` verify blocks.

| Req / Decision | Behavior | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|----------------|----------|------------|-----------------|-----------|-------------------|-------------|--------|
| D-02 bridge | raw data → BinMapper → FeatureColumn → train matches identity-bin / C++ goldens | — | shape/length checks before indexing | unit + oracle | `cargo test -p lgbm raw_bin_train` · `cargo test -p oracle-harness` | ❌ W0 (Rust) | ⬜ pending |
| PYB-01 | train+predict through PyO3 mirrors official; GIL released; owned arrays | — | N/A | integration (A/B) | `pytest .../test_booster_parity.py` | ❌ W0 | ⬜ pending |
| PYB-01 GIL | `Python::detach` releases GIL during train | — | N/A | unit | `pytest .../test_gil_release.py` | ❌ W0 | ⬜ pending |
| PYB-02 | f32 AND f64 dense + CSR/CSC sparse match official either width | T-08 V5 | validate shape/dtype/contiguity; CSR/CSC indptr/indices bounds | integration (A/B) | `pytest .../test_numpy_sparse_parity.py -k 'f32 or f64 or csr or csc'` | ❌ W0 | ⬜ pending |
| PYB-02 polars | polars DataFrame (numeric + Categorical) routes per D-04, matches | T-08 V5 | dtype-routing validated; no unchecked indexing | integration | `pytest .../test_polars_input.py` | ❌ W0 | ⬜ pending |
| PYB-03 | LGBMClassifier/Regressor/Ranker semantics match official | — | N/A | integration (A/B) | `pytest .../test_sklearn_parity.py` | ❌ W0 | ⬜ pending |
| PYB-04 | custom obj/metric reproduce reference; `Booster.refit()` matches | T-08 V5 | length-validate grad/hess vs num_data*num_class | integration (A/B) | `pytest .../test_custom_refit_parity.py` | ❌ W0 | ⬜ pending |
| D-10 | save_model/model_to_string round-trip C++ format; pickle works | T-08 V5 | parse via validated `lgbm-model` text loader (typed error) | unit + parity | `pytest .../test_persistence.py` | ❌ W0 | ⬜ pending |
| D-06/07/08 | params dict coercion; unimplemented-param raises; alias resolution | — | reject malformed values; typed exception | unit | `pytest .../test_params.py` | ❌ W0 | ⬜ pending |

*Status: ⬜ pending · ✅ green · ❌ red · ⚠️ flaky*

---

## Wave 0 Requirements

- [ ] `crates/lgbm-python/` crate + `pyproject.toml` (maturin) + workspace `members` entry
- [ ] Rust bridge tests for the D-02 raw→bin→train path (reuse oracle goldens)
- [ ] facade `Booster` method coverage (batch predict / feature_importance / refit / save) — Rust tests before Python
- [ ] `crates/lgbm-python/python/tests/` pytest tree + CI invocation (pytest, `maturin develop` install)
- [ ] A/B parity fixtures: shared `(X, y, params)` feeding both `lightgbm` and `lightgbm_rs`, pinned to `crates/oracle-harness/tests/fixtures/REFERENCE_MANIFEST.md` deterministic settings (`deterministic=true`, `force_row_wise=true`, `num_threads=1`, fixed seed)

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| Plotting output (`plot_importance`/`plot_tree`/`plot_metric`) | D-09 | matplotlib/graphviz render — visual, optional deps | Generate a plot from a trained model in a notebook; confirm axes/labels render without error |
| Built abi3 wheel installs on a clean interpreter | D-13 | requires a separate venv / CI matrix leg | `pip install dist/*.whl` in a fresh venv on the target CPython floor; `import lightgbm_rs` succeeds |

*All numerical/behavioral parity has automated A/B verification.*

---

## Validation Sign-Off

- [ ] All tasks have `<automated>` verify or Wave 0 dependencies
- [ ] Sampling continuity: no 3 consecutive tasks without automated verify
- [ ] Wave 0 covers all MISSING references
- [ ] No watch-mode flags
- [ ] Feedback latency < 120s
- [ ] `nyquist_compliant: true` set in frontmatter

**Approval:** pending
