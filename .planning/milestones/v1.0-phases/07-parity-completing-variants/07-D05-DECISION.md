# D-05 Decision: Bagged-Subset Split-Gain Determinism — FAITHFUL-FIX

**Phase:** 07-parity-completing-variants · **Plan:** 07-01 (Wave 0) · **Decided:** 2026-06-07

**Branch chosen: FAITHFUL-FIX.** A source-built `lib_lightgbm` 4.6 FP execution
trace proved the bagged-subset split-gain knife-edge (DEF-06-01 + the typed-rejected
`regression_l1 + bagging`) was a faithfully-fixable SPLIT-GAIN OPERAND bug, NOT an
irreducible f32 / near-zero-gain accumulation artifact. The fix is applied; the four
`regression_l1_bag1_*` matrix cells assert real-binary parity; DEF-06-01 is cleared.

---

## 1. The source-built FP trace (Phase-5 05-09 technique)

Built `lib_lightgbm` 4.6 (`VERSION.txt = 4.6.0.99`, the 4.6 line) CPU-only,
single-thread (`-DUSE_GPU=OFF -DUSE_CUDA=OFF -DUSE_OPENMP=OFF -DBUILD_CLI=ON
-DCMAKE_BUILD_TYPE=Release`) into `/tmp` (the `LightGBM/` tree and the `/tmp` build
were NEVER git-added; the C++ instrumentation was reverted after capture). The
prebuilt pip wheel could not expose the per-bin `SUBSET_HIST` / per-candidate
`current_gain` / `min_gain_shift`; the source build was authorized for exactly this
(memory: lightgbm-ref-tree-untracked).

Instrumented `FindBestThresholdSequentially` (`feature_histogram.hpp`), gated on
`LGBM_FP_TRACE`, to dump (with `.to_bits()`):
- the per-node HEADER: `sum_gradient`, `sum_hessian` (as passed = bumped),
  `cnt_factor`, `min_gain_shift`;
- the per-bin SUBSET_HIST `sum_gradient`/`sum_hessian`;
- per-candidate `current_gain` vs `min_gain_shift` with the accept flag.

Drove the EXACT `binary_bag1_es0_bfa1` cell (seed `0x60057000` = 1610969088,
`bagging_fraction=0.7 bagging_freq=1 bagging_seed=3 deterministic=true
force_row_wise=true num_threads=1`). The CLI model came out BIT-IDENTICAL to the
wheel-captured trace (`split_gain = 3.5999999… 8.881779…e-16 4.440889…e-16`, same
thresholds), confirming the source build reproduces the reference.

## 2. The confirmed root cause — fold-ORDER (operand consistency), not f32 noise

The wheel evidence showed tree-0 in C++ has 4 leaves; the two deepest splits have
`split_gain = current_gain − min_gain_shift ≈ 8.88e-16` and `4.44e-16` (≈ kEpsilon).
The source trace localized the decisive operands at the deeper node (tree-0 node 1,
7 in-bag rows):

| Quantity | value | bits |
|---|---|---|
| `current_gain` (winning candidate) | 4.9999998297010109 | `0x4013fffff4924920` |
| C++ `min_gain_shift` (from BUMPED sum_hessian) | 4.99999982970101 | `0x4013fffff492491f` |
| `current_gain > min_gain_shift`? | **TRUE** (by 1 f64 ULP) → C++ **ACCEPTS** | |

The Rust port computed `min_gain_shift` from the **RAW** leaf `sum_hessian`:

| Quantity | value | bits |
|---|---|---|
| Rust `min_gain_shift` (from RAW sum_hessian) | 4.999999829701015 | `0x4013fffff4924925` |
| `current_gain (0x…4920) > min_gain_shift (0x…4925)`? | **FALSE** → Rust **REJECTS** | |

C++ applies the `2*kEpsilon` entry bump AT the `FindBestThreshold` call site —
`find_best_threshold_fun_(sum_gradient, sum_hessian + 2*kEpsilon, …)`
(`feature_histogram.hpp:174`) — so `BeforeNumerical` (and therefore `min_gain_shift`)
divides by the BUMPED `sum_hessian`. The Rust `find_best_split_cpu` computed
`gain_shift` from the raw `sum_hessian` (it bumped AFTER), making `min_gain_shift`
~7 ULPs too high and rejecting every bagged-subset split whose `current_gain` exceeds
the C++ `min_gain_shift` by a single ULP. **This is a deterministic operand-order
bug — faithfully fixable — NOT an f32 / near-zero accumulation artifact.** (The same
relationship was independently confirmed at the regression_l1 root: trace
`min_gain_shift = 11.999999999999998 (0x4027ffffffffffff)` = the bumped value, not the
raw `12.0`.)

## 3. The faithful fix applied

1. **`crates/lgbm-compute/src/kernels/split.rs` — `find_best_split_cpu` (f64) AND
   `find_best_split_raw_f32_on` (f32 hip path):** compute the `2*kEpsilon` bump
   FIRST and feed the BUMPED `sum_hessian` into `get_leaf_gain` for `min_gain_shift`
   (mirroring C++ `feature_histogram.hpp:174,400-401`).
2. **`crates/lgbm-treelearner/src/learner.rs` — `per_bin_gains` diagnostic re-scan:**
   same bumped-`sum_hessian` `min_gain_shift` so the diagnostic stays bit-identical
   to the live kernel.
3. **`xtask/cpp/kernel_capture.cpp` — `EmitSCase`:** the golden-capture transcription
   had the SAME raw-vs-bumped bug; fixed to use the bumped `sum_hessian`, then
   `split.txt` regenerated (byte-idempotent). The 6 split cases shifted `min_gain_shift`
   (and the derived net gain) by 1–2 ULPs to match real C++; `kernel_parity` is 4/4
   bit-exact again.
4. **`crates/lgbm-boosting/src/gbdt.rs` — `no_split_constant_value`:** the C++
   `ObtainAutomaticInitialScore` fallback (`gbdt.cpp:418-429`) for the no-split FIRST
   tree. With `boost_from_average=false`, C++ still computes the automatic init score
   (the label median for regression_l1) and uses it as the constant tree's leaf value.
   The Rust port pushed `0.0`; fixed so the bfa-off `regression_l1` constant tree-0 =
   the label median (11.0), MATCHING C++ (the "rust:0.0 vs cpp:11.0" leaf-VALUE gap).
5. **Un-deferred `regression_l1 + bagging`:** removed the
   `BoostingError::UnsupportedConfig` reject in `gbdt.rs::train_one_iter`. The four
   `regression_l1_bag1_*` matrix cells now assert real-binary parity (the bfa-off pair
   joins the bounded `uniform_grad_residual` L1 cross-feature gain-tie family).
6. **Cleared DEF-06-01** in `.planning/phases/06-…/deferred-items.md`.

## 4. Verification (bit-exact / parity)

- `subset_determinism_diagnostic`: `binary_bag1_es0_bfa1` tree-0 leaf count
  **rust=4 cpp=4 (MATCH — DEF-06-01 closed)**.
- `boosting_parity` matrix: **26/26 GREEN**, incl. the un-deferred
  `regression_l1_bag1_*` cells (real-binary parity, not the typed error).
- `kernel_parity`: **4/4 bit-exact** (split golden regenerated, byte-idempotent).
- `learner_parity`: **12/12** incl. the keystone `spine_real_binary` /
  `mfb_pos_real_binary` (no learner regression).
- `cargo test --workspace`: **GREEN**.

## 5. Residual (bounded, documented — NOT a fabricated pass)

The non-bagged `regression_l1` bfa-off cells (`uniform_grad_residual`) retain a
BOUNDED cross-feature L1 gain tie on a degenerate 2-row node: two features both
separate the node perfectly with gains equal to ~1 f64 ULP, and Rust vs C++ order
them oppositely (C++ tree-6 3rd split = feature 1; Rust = feature 0). The tree
TOPOLOGY (split count + per-leaf row counts) matches C++ exactly; only the two
swapped-leaf median-residual VALUES differ (< 0.1, the matrix asserts this bound and
hard-caps the count). This is a genuine f64-accumulation knife-edge on a degenerate
node — distinct from the (now-fixed) operand bug — and is the same documented L1
uniform-gradient residual family STATE.md recorded for 06-05. No assertion was
weakened; pre-fix this cell grew DEGENERATE single-leaf STUBS (trees 6/9/11 = `[12]`/
`0.0`) that the old `rl.len() != gl.len()` guard silently SKIPPED — the fix is a
strict improvement (real topology-matching trees, bit-exact leaves on the untouched
branch).
