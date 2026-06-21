---
phase: 03-tree-model-model-text-i-o-predict-parity
plan: 01
subsystem: model
tags: [lgbm-model, format, model-capture, golden-corpus, DAT-09, D-05]
requires:
  - lgbm-core (Config, error idiom)
  - lgbm-dataset (example fixtures, golden-loader idiom)
  - oracle-harness (comparator idiom)
provides:
  - "crate lgbm-model (workspace member)"
  - "format::format_g17 / format::format_g6 — %.17g / {:g} printf-faithful formatters"
  - "ModelError boundary enum (MalformedModel, ShapeMismatch)"
  - "xtask model-capture subcommand"
  - "committed D-05 model/predict golden corpus (5 corpora) + format_golden.txt"
affects:
  - "all later Phase-3 parity slices (03-02+) replay these committed fixtures"
tech-stack:
  added:
    - "pip lightgbm 4.6.0 (CAPTURE-time only; not a crate dependency)"
  patterns:
    - "hand-rolled %g formatter (correctly-rounded sig digits via {:.*e} + fixed/sci selection)"
    - "header-only/prebuilt-tool golden capture, byte-idempotent (Phase 1/2 discipline)"
key-files:
  created:
    - crates/lgbm-model/Cargo.toml
    - crates/lgbm-model/src/lib.rs
    - crates/lgbm-model/src/error.rs
    - crates/lgbm-model/src/format.rs
    - crates/lgbm-model/tests/fixtures/models/.gitkeep
    - xtask/py/model_capture.py
    - "crates/lgbm-model/tests/fixtures/models/{regression,binary,multiclass,categorical,subrange}/*"
    - crates/lgbm-model/tests/fixtures/models/format_golden.txt
  modified:
    - Cargo.toml
    - Cargo.lock
    - xtask/src/main.rs
    - crates/oracle-harness/fixtures/REFERENCE_MANIFEST.md
decisions:
  - "Golden-capture path B (pip lightgbm train + dump) selected + approved (Task 3 checkpoint)"
  - "%g formatter sources correctly-rounded digits from Rust {:.*e}, then applies the %g fixed/scientific rule — NOT ryu/to_string/{:.17e}"
metrics:
  duration: ~20 min
  completed: 2026-06-05
  tasks: 4
  files: 29
---

# Phase 3 Plan 01: lgbm-model scaffold + %g formatter + model-capture golden corpus Summary

Wave-0 enabling slice for Phase 3: the `lgbm-model` crate, the byte-exact `%.17g`/`{:g}`
float formatter (DAT-09 serialization linchpin, proven FIRST), and the `xtask model-capture`
pipeline that produced the committed `version=v4` reference model `.txt` + predict-vector
goldens every later parity slice replays.

## What Was Built

- **`lgbm-model` crate** (Task 1): new workspace member depending on `lgbm-core` +
  `lgbm-dataset` (+ dev `oracle-harness`). `ModelError` `thiserror` enum with
  `MalformedModel { detail }` (missing key / inconsistent array length / OOB node index —
  mirrors `gbdt_model_text.cpp:494,514` + `tree.cpp` `Log::Fatal` sites) and
  `ShapeMismatch { detail }` (predict-input feature count vs `max_feature_idx+1`), each
  with a `///` C++-source citation and the dataset crate's Display-assertion test idiom.
- **`format.rs` `%g` formatters** (Task 2, TDD): `format_g17` (`%.17g`) for
  `threshold`/`leaf_value`/`leaf_weight` and `format_g6` (`{:g}`, precision 6) for
  `split_gain`/`internal_value`/`internal_weight` (+ `shrinkage`, golden is arbiter). The
  algorithm sources `precision` correctly-rounded significant digits from Rust's
  `format!("{:.*e}", precision-1, x)`, then applies the C/printf `%g` fixed-vs-scientific
  rule (scientific iff decimal exp `< -4` or `>= precision`), strips trailing zeros, and
  emits a C-locale exponent (lowercase `e`, explicit sign, min 2 digits) to match `fmt`.
  10-case inline battery (incl. `0.1 -> 0.10000000000000001`, subnormal `5e-324`, `1e±300`,
  signed zero, the exactly-17-digit case) verified bit-for-bit against C printf `%g`, plus a
  bit-exact round-trip property (`f64::from_str(format_g17(x)) == x`). A
  `golden_matches_formatter` test cross-checks the committed `format_golden.txt` (emitted by
  Task 4 from the authoritative `fmt`) so the literals are never a hand-guess.
- **Task 3 checkpoint (golden-capture path)**: pre-resolved by the orchestrator to
  **path-b-pip**. Recorded the decision + the exact lightgbm version (`4.6.0`) + train
  params in REFERENCE_MANIFEST.md (via `write_manifest`). No pause.
- **`xtask model-capture`** (Task 4): resolves a capture python (`$LGBM_CAPTURE_PYTHON`),
  asserts lightgbm `4.6.0`, and shells out to `xtask/py/model_capture.py`, which trains the
  5-corpus D-05 set (regression / binary / multiclass(3) / categorical / subrange) on the
  reused Phase-2 example matrices with `deterministic=true force_row_wise=true
  num_threads=1 seed=MODEL_TRAIN_SEED` and NO subsampling, then dumps each `version=v4`
  `model.txt` + `raw.txt` (PRD-01) / `transformed.txt` (PRD-02) / `leaf.txt` (PRD-03) and,
  for `subrange`, `subrange.txt` (PRD-06 `(start_iteration,num_iteration)` slices incl.
  `-1==all`). Also emits `format_golden.txt`. `write_manifest` extended with a
  "Model / Predict Golden Set" section. Recorded constants `MODEL_TRAIN_SEED` /
  `MODEL_LIGHTGBM_VERSION` added.

## Key Decisions

- **Path B (pip lightgbm), human-approved** — the only feasible source of a trained v4
  `.txt` here (no Rust trainer yet; C++ trainer unbuildable with empty `external_libs`). The
  prebuilt wheel's `save_model()` is the authoritative `%.17g` v4 format. The pip tool is
  CAPTURE-time only; `cargo test` reads the committed fixtures and needs nothing.
- **`%g` from `{:.*e}` digits, not ryu/to_string/`{:.17e}`** — Rust's scientific formatter
  gives the correctly-rounded significant digits; the `%g` fixed/scientific selection +
  trailing-zero strip + C-locale exponent are layered on top. This is the only path that
  reproduces C++ `fmt` `{:.17g}` byte-for-byte (e.g. `0.1 -> 0.10000000000000001`).

## Deviations from Plan

None — plan executed as written. The Task 3 checkpoint was pre-resolved by the orchestrator
(path-b-pip) and handled without pausing, exactly as instructed.

One environment note (not a deviation): `pip install lightgbm` into the system interpreter
is blocked (PEP-668 externally-managed). Resolved per the sanctioned venv approach — created
`/tmp/lgbm-capture-venv` and installed lightgbm 4.6.0 there; `model-capture` resolves it via
`$LGBM_CAPTURE_PYTHON`. This is capture-tooling only; no shipped-crate impact.

## Verification

- `cargo build -p lgbm-model` compiles; `cargo test -p lgbm-model error::` (3) + `format::`
  (6, incl. `golden_matches_formatter` against the committed golden) all pass.
- `cargo run -p xtask -- model-capture` exits 0, writes the 5 corpora + `format_golden.txt`,
  every `model.txt` begins `tree\nversion=v4`, no fixture references the untracked
  `LightGBM/` tree, and a re-run leaves `git diff --stat` EMPTY for the fixtures dir +
  manifest (byte-idempotent).
- `cargo test --workspace` green (0 failed; lgbm-model adds 9 lib tests; dataset/core/oracle
  unaffected).
- REFERENCE_MANIFEST.md contains the "Model / Predict Golden Set" section naming lightgbm
  `4.6.0` + the deterministic train params.

## Notes for Later Plans

- The committed corpus is the replay source for 03-02+ (model-text round-trip writer, raw /
  transformed / leaf / sub-range predict parity). Golden float vectors are `;`-separated raw
  f64 bit patterns (decimal `u64`) for bit-exact replay; leaf indices are `;`-separated `u32`.
- `feature_infos=` is preserved verbatim on a load→write round-trip (never reformatted), so
  Pitfall 2 (ostream vs `fmt` float format) is invisible to Phase 3 — flagged for Phase 6.
- `shrinkage`'s ostream-default formatting (`format_g6` covers it pending the byte-exact
  model-text round-trip golden as the final arbiter — RESEARCH Open Q4).

## Self-Check: PASSED

- Created files verified present: crate manifest/lib/error/format, `.gitkeep`,
  `model_capture.py`, `regression/model.txt`, `format_golden.txt`.
- Commits verified present: `9734bf4` (Task 1), `81d75b8` (Task 2), `2cd8706` (Task 4).
