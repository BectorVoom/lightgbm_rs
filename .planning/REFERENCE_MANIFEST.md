# Reference Manifest — Phase 7 Determinism Fixtures (D-05)

## v1.1 CUDA On-Device Tree Learner — C++ Port-Source Map

**`docs/cuda-kernel-design.md`** — authoritative, source-verified design reference for
LightGBM's public CUDA backend (the read-only `LightGBM/` C++ tree being ported by the
v1.1 milestone, Phases 14–19). Covers **all 58 CUDA source files** and **all 81
`__global__` kernels** across **11 subsystems** (histogram constructor, best split
finder, data partition, leaf splits/driver, objectives, metrics, score updater, gradient
discretizer, tree I/O, CUDARowData, shared primitives), plus device structs, host
infrastructure, end-to-end per-iteration sequencing, and a lightgbm_rs port-considerations
section. Each kernel/device-helper/launcher is named and verified kernel-by-kernel against
source (full-doc audit: 81/81 kernels named).

Use as the port reference for Phases 15–19 (on-device growth → frontier best-split →
data partition → feature coverage), which mirror `CUDASingleGPUTreeLearner` subsystem by
subsystem. Key parity constraints captured: `CUDATree.Split` precedes `DataPartition.Split`
(returns `right_leaf_index`); the histogram subtraction-trick + most-frequent-bin-fix are
correctness (not just speed) requirements; interleaved `[2·b]/[2·b+1]` histogram layout;
`hist_t=double` is durable while SP-f32 shared atomics are non-deterministic (the f32-vs-f64
/ ROCm residual); the int16-packed quantized path is integer-exact (the natural bit-exact
GPU target). CUDA-support boundaries: 11 CUDA objectives (no MAPE/Gamma/Tweedie/xentropy),
12 CUDA pointwise metrics (no AUC/NDCG/MAP/multiclass).

_Verified against `LightGBM/` C++ source 2026-06-29 (quick task 260629-djo)._

---

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
| `binary_bag1_es0_bfa1_subset_trace.txt` | `binary_bag1_es0_bfa1` (was DEF-06-01: tree 0 rust 2 vs cpp 4 leaves — CLOSED by D-05 faithful-fix) | SOURCE-BUILT lib_lightgbm 4.6 FP trace: per-candidate `current_gain`/`min_gain_shift` (BUMPED sum_hessian) for the root + the two deeper 1-ULP-accept nodes, documented per-bin `sum_gradient`/`sum_hessian`, realized leaf count (4) |
| `regression_l1_bag1_es0_bfa0_subset_trace.txt` | `regression_l1_bag1_es0_bfa0` (was tree 0 rust:0.0 vs cpp:11.0 — UN-DEFERRED by D-05 faithful-fix) | constant no-split tree-0 (leaf_count=1, leaf_value=11.0 = label median via ObtainAutomaticInitialScore fallback) |

The committed traces are the SOURCE-BUILT capture (Phase-5 05-09 technique): a
`lib_lightgbm` 4.6 (VERSION 4.6.0.99) CPU-only single-thread CLI was instrumented
(`FindBestThresholdSequentially` `.to_bits()` dumps gated on `LGBM_FP_TRACE`) to
record the per-candidate `current_gain`/`min_gain_shift` the prebuilt wheel cannot
expose. The `LightGBM/` tree and the `/tmp` build were NEVER git-added; the C++
instrumentation was reverted after capture. See `07-D05-DECISION.md`.

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

---

# Phase 7 W4 — GOSS (BST-04) Oracle Fixtures (plan 07-05)

GOSS gradient-based one-side sampling goldens, captured on the real prebuilt
`lib_lightgbm` 4.6 pip wheel. Normal `cargo test` reads the committed fixtures and
needs none of this.

## Pinned Reference

- **pip `lightgbm` version:** `4.6.0` (`GOSS_ORACLE_LIGHTGBM_VERSION`)
- **Train seed:** `0x60057000` (`GOSS_ORACLE_SEED`, == `BOOSTING_ORACLE_SEED`)
- **Bagging (RNG) seed:** `3` (`GOSS_BAGGING_SEED`, the per-block `Random(bagging_seed+i)`
  base, goss.hpp:97)

## Deterministic Capture Flags

- `boosting=goss` (alias-expands to `gbdt` + `data_sample_strategy=goss`)
- `top_rate ∈ {0.2, 0.1}` × `other_rate ∈ {0.1, 0.05}` (top+other ≤ 0.5 ⇒ subset path)
- `deterministic=true force_row_wise=true num_threads=1`
- identity binning (as above); GOSS forbids bagging (no bag axis)

## Exact Capture Command

```bash
LGBM_CAPTURE_PYTHON=<python-with-lightgbm-4.6.0> \
  cargo run -p xtask -- goss-oracle-capture
```

which runs `xtask/py/goss_oracle_capture.py <out_dir> 0x60057000 3 4.6.0` and writes,
under `crates/oracle-harness/tests/fixtures/goss/`:

| Fixture | Kind | Contents |
|---------|------|----------|
| `goss_t{T}_o{O}_es{E}_bfa{B}_model.txt` (16 cells) | L5 model parity | real lib_lightgbm 4.6 `%.17g` model text for the top×other×{es}×{bfa} cell; GOSS sampling + grad/hess amplification reflected in the trees |
| `goss_rng_replay.txt` | RNG-replay | kept/dropped row indices the `goss.hpp` Helper produces for a fixed grad/hess input, derived over the bit-exact C++ `Random` LCG (`random.h`) + `ArgMaxAtK` (`array_args.h`); carries the input grad/hess as f32 bits |

The RNG-replay golden freezes the algorithm spec over the bit-exact LCG (the wheel
cannot expose internal bag indices — identical posture to the bagging `bag_indices_*`
golden). The capture is byte-idempotent (empty `git diff` on a re-run). NEVER
`git add` the `LightGBM/` tree.

## Replay

- `boosting_parity.rs::goss_rng_replay` — reproduces `GossSampleStrategy::bagging`
  over the recorded grad/hess and asserts the kept/dropped indices BIT-EXACT.
- `boosting_parity.rs::goss_parity_matrix` — trains each cell via `boosting=goss`
  and asserts the per-tree leaf values BIT-EXACT against the real-binary golden on
  the overlapping trees. Both skip-pass until the fixtures are captured.

# Phase 7 W5 — DART (BST-05) Oracle Fixtures (plan 07-06)

DART (Dropouts meet Multiple Additive Regression Trees) goldens, captured on the real
prebuilt `lib_lightgbm` 4.6 pip wheel. Normal `cargo test` reads the committed fixtures
and needs none of this.

## Pinned Reference

- **pip `lightgbm` version:** `4.6.0` (`DART_ORACLE_LIGHTGBM_VERSION`)
- **Train seed:** `0x60057000` (`DART_ORACLE_SEED`, == `BOOSTING_ORACLE_SEED`)
- **Drop (RNG) seed:** `4` (`DART_DROP_SEED`, the single advancing `Random(drop_seed)`,
  config.h:463, dart.hpp:45)

## Deterministic Capture Flags

- `boosting=dart`
- `drop_rate=0.1 max_drop=50 skip_drop=0.5` (cell defaults); the matrix varies
  `uniform_drop ∈ {0,1}` × `xgboost_dart_mode ∈ {0,1}` (the 4 normalize branches) × `bag ∈ {0,1}`
- `deterministic=true force_row_wise=true num_threads=1`
- identity binning (as above); 24-row corpus (shared with GOSS)

## Exact Capture Command

```bash
LGBM_CAPTURE_PYTHON=<python-with-lightgbm-4.6.0> \
  cargo run -p xtask -- dart-oracle-capture
```

which runs `xtask/py/dart_oracle_capture.py <out_dir> 0x60057000 4 4.6.0` and writes,
under `crates/oracle-harness/tests/fixtures/dart/`:

| Fixture | Kind | Contents |
|---------|------|----------|
| `dart_u{U}_x{X}_bag{B}_model.txt` (8 cells) | L5 model parity | real lib_lightgbm 4.6 `%.17g` model text for the uniform_drop×xgboost_dart_mode×{bag} cell; the normalized tree weights are baked into the stored leaf values by DART's Shrinkage sequence |
| `dart_u{U}_x{X}_bag{B}_pred.txt` (8 cells) | predict | per-row transformed `predict()` (f64 bits) for the cell |
| `dart_drop_seed4_iter12.txt` | drop RNG-replay | dropped tree indices PER ITERATION the `dart.hpp` DroppingTrees produces over the bit-exact C++ `Random` LCG (`random.h`), with the tree_weight/sum_weight evolution mirroring Normalize; carries the per-iter tree_weight history as f64 bits |

The drop RNG-replay golden freezes the algorithm spec over the bit-exact LCG (the wheel
cannot expose internal drop indices — identical posture to the GOSS `goss_rng_replay`
golden). The capture is byte-idempotent (empty `git diff` on a re-run). NEVER `git add`
the `LightGBM/` tree.

## Replay

- `boosting_parity.rs::dart_drop_rng_replay` — reproduces `DART::DroppingTrees`'s draw
  order over the recorded tree_weight history + the advancing `Random(drop_seed)` and
  asserts the dropped indices per iteration BIT-EXACT.
- `boosting_parity.rs::dart_parity_matrix` — trains each cell via `boosting=dart` and
  asserts the per-tree leaf values BIT-EXACT against the real-binary golden on the
  overlapping trees (all 4 normalize branches × {bag}) + predict() within ORACLE_TOL.
  Both skip-pass until the fixtures are captured.

# Phase 7 W6 — Random Forest (BST-06) Oracle Fixtures (plan 07-07)

## Provenance

The Random Forest (`boosting=rf`) averaged-tree variant with mandatory bagging
(`bagging_fraction=0.7 bagging_freq=1`) over the single-output (regression) and
multiclass axes, trained on the real prebuilt `lib_lightgbm` 4.6 pip wheel and dumped
as `%.17g` model text + per-row `predict()` (f64 bits). RF stores the RAW per-tree leaf
values (averaging happens at predict via `average_output`), renews leaves only when
`obj->IsRenewTreeOutput()` (a no-op for L2 — the leaf is the learner's gradient-fit
Newton output over the bagged subset, then `AddBias(init)`), and applies NO
learning-rate shrinkage (`shrinkage_rate_=1.0`). The bagged-subset leaf structure
inherits the 07-01 D-05 FAITHFUL-FIX posture.

## Capture

```bash
LGBM_CAPTURE_PYTHON=/tmp/lgbm-capture-venv/bin/python \
  cargo run -p xtask -- rf-oracle-capture
```

which runs `xtask/py/rf_oracle_capture.py <out_dir> 0x60057000 3 4.6.0` and writes,
under `crates/oracle-harness/tests/fixtures/rf/`:

| Fixture | Kind | Contents |
|---------|------|----------|
| `rf_single_bag_model.txt` | L5 model parity | real lib_lightgbm 4.6 `%.17g` model text for the single-output (regression) RF cell (12 averaged trees, mandatory bagging, `average_output`) |
| `rf_single_bag_pred.txt` | predict | per-row averaged `predict()` (f64 bits) — the per-tree sum / num_iteration |
| `rf_multi_bag_model.txt` | L5 model parity | real lib_lightgbm 4.6 `%.17g` model text for the multiclass RF cell (36 trees = 12 iters × 3 classes, class-major) |
| `rf_multi_bag_pred.txt` | predict | per-row per-class averaged `predict()` (f64 bits, class-major) |

The capture is byte-idempotent (empty `git diff` on a re-run). NEVER `git add` the
`LightGBM/` tree.

## Replay

- `boosting_parity.rs::rf_single_parity` — trains via `boosting=rf` and asserts the
  per-tree leaf values BIT-EXACT against the real-binary golden (averaged trees +
  mandatory bagging, `average_output`) + predict() within ORACLE_TOL.
- `boosting_parity.rs::rf_multi_parity` — asserts the class-major STRUCTURE exactly
  (tree count == iters × num_class, the stride) + predict() within ORACLE_TOL (the
  documented multiclass softmax exp-libm ~1-ULP residual, 06-04-SUMMARY).
  Both skip-pass until the fixtures are captured.
