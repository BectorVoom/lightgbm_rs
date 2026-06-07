# Reference Manifest — Phase 7 Determinism Fixtures (D-05)

This planning-side manifest records the Phase-7 Wave-0 (D-05) bagged-subset
split-gain determinism FP-trace capture command and the fixtures it produces.
It complements the crate-side manifest at
`crates/oracle-harness/fixtures/REFERENCE_MANIFEST.md` (which pins the C++ RNG /
binning / kernel / learner / boosting goldens). Normal `cargo test` reads the
committed fixtures and needs NONE of this; only the human-gated capture run does.

## Pinned Reference

- **pip `lightgbm` version:** `4.6.0` (`SUBSET_DETERMINISM_LIGHTGBM_VERSION`,
  reuses the same prebuilt-wheel binary as the Phase-3/5/6 captures)
- **Train seed:** `0x60057000` (`SUBSET_DETERMINISM_SEED`, == `BOOSTING_ORACLE_SEED`
  so the captured tree-0 bagged subset matches the D-07 matrix cells EXACTLY)

## Deterministic Capture Flags

- `deterministic=true`
- `force_row_wise=true`
- `num_threads=1`
- `bagging_fraction=0.7`
- `bagging_freq=1`
- `bagging_seed=3`
- `feature_fraction=1.0`
- identity binning (`max_bin=255 min_data_in_bin=1 bin_construct_sample_cnt=1_000_000
  feature_pre_filter=false`) so `binned_value == raw_value`
- default `float` width (`SCORE_T_USE_DOUBLE` / `LABEL_T_USE_DOUBLE` NOT defined)

The version is asserted BEFORE any training (threat T-07-01-SC); a wrong wheel
version can never silently emit a divergent trace. NEVER `git add` the
`LightGBM/` tree.

## Exact Capture Command

```bash
LGBM_CAPTURE_PYTHON=<python-with-lightgbm-4.6.0> \
  cargo run -p xtask -- subset-determinism-capture
```

which trains the two knife-edge cells and writes the tree-0 subset FP trace:

```bash
xtask/py/subset_determinism_capture.py \
  crates/oracle-harness/tests/fixtures/determinism 0x60057000 4.6.0
```

For the FINEST per-bin / per-candidate trace, build `lib_lightgbm` 4.6 CPU-only
single-thread from source with FP-trace prints (the Phase-5 05-09 technique;
`external_libs` are fetchable) and point `$LGBM_TRACE_LIB` at it before running.

## Captured Fixtures

Under `crates/oracle-harness/tests/fixtures/determinism/`:

| Fixture | Cell | Contents |
|---------|------|----------|
| `binary_bag1_es0_bfa1_subset_trace.txt` | `binary_bag1_es0_bfa1` (DEF-06-01: tree 0 rust 2 vs cpp 4 leaves) | tree-0 in-bag subset, per-bin `sum_gradient`/`sum_hessian`, `cnt_factor`, per-split `current_gain`/`min_gain_shift`, realized leaf count |
| `regression_l1_bag1_es0_bfa0_subset_trace.txt` | `regression_l1_bag1_es0_bfa0` (tree 0 rust:0.0 vs cpp:11.0) | same trace shape |

## Replay

`crates/oracle-harness/tests/boosting_parity.rs::subset_determinism_diagnostic`
drives the Rust subset histogram + per-split gain for tree 0 of each cell and
compares cell-for-cell against the captured trace, in localizing order:
per-bin subset `sum_hessian` → `cnt_factor` → `current_gain`/`min_gain_shift` →
leaf count. Until the fixtures are captured the test skip-passes (the
`read_golden` Option shape) so a fresh checkout stays GREEN.

## Trace Format (`<cell>_subset_trace.txt`)

Line-delimited text (diff-friendly). `#`-prefixed lines are comments / metadata.

```
# [GSD-META] cell=<cell> objective=<obj> bfa=<0|1> seed=<s>
# bagging_fraction=<f> bagging_freq=<n> bagging_seed=<s> num_data=<n>
LEAF_COUNT <n>
SPLIT feature=<j> threshold=<t> split_gain=<repr-f64>
# (source-build trace adds, per the Phase-5 technique:)
IN_BAG <i0;i1;...>
SUBSET_HIST feature=<j> bin=<b> sum_gradient=<repr-f64> sum_hessian=<repr-f64>
CNT_FACTOR feature=<j> value=<repr-f64>
SPLIT feature=<j> threshold=<t> current_gain=<repr-f64> min_gain_shift=<repr-f64>
```
