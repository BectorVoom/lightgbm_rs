---
phase: 07-parity-completing-variants
plan: 01
subsystem: tree-learner
tags: [tree-learner, split-gain, min-gain-shift, kEpsilon, bagging, regression-l1, fp-execution-trace, lightgbm-4.6, oracle-parity, numerical-fidelity, d-05]

# Dependency graph
requires:
  - phase: 05-09
    provides: "the real-binary FP execution-trace technique (build lib_lightgbm 4.6 CPU-only single-thread, instrument the hot path with .to_bits() dumps, read genuine operand provenance) — carried forward here to the bagged-subset split-gain"
  - phase: 06-06
    provides: "the typed-reject for regression_l1 + bagging + DEF-06-01; the subset-path median-residual RenewTreeOutput (8330cee); the D-07 matrix"
provides:
  - "DEF-06-01 CLOSED bit-exact: binary_bag1_es0_bfa1 tree-0 grows 4 leaves matching C++ (the bagged-subset split-gain knife-edge)"
  - "ROOT CAUSE (source-built FP trace): the Rust min_gain_shift was computed from the RAW leaf sum_hessian, while C++ uses the 2*kEpsilon-BUMPED value — making the Rust gain-shift ~7 ULPs too high and rejecting bagged-subset splits whose current_gain exceeds the C++ min_gain_shift by a single f64 ULP"
  - "the min_gain_shift bumped-sum_hessian fix in lgbm-compute find_best_split (f64 cpu + f32 hip) + the per_bin_gains diagnostic + the kernel_capture.cpp transcription (split.txt golden regenerated, byte-idempotent)"
  - "regression_l1 + bagging UN-DEFERRED: the BoostingError::UnsupportedConfig reject removed; the 4 regression_l1_bag1_* cells assert real-binary parity"
  - "gbdt no_split_constant_value: the C++ ObtainAutomaticInitialScore fallback (gbdt.cpp:418-429) so the bfa-off regression_l1 constant tree-0 carries the label median (11.0), not 0.0"
  - "07-D05-DECISION.md: the settled D-05 branch (faithful-fix) with source-build FP-trace evidence — every downstream bagging variant (GOSS W4, RF W6) inherits a settled answer"
affects: [07, 08]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Source-built FP execution trace for a split-gain knife-edge: build lib_lightgbm 4.6 CPU-only single-thread CLI into /tmp, instrument FindBestThresholdSequentially with .to_bits() dumps gated on an env var, drive the EXACT diverging cell, read the genuine current_gain/min_gain_shift operands — never git-add LightGBM/, revert the instrumentation after capture"
    - "min_gain_shift operand consistency: BeforeNumerical receives the SAME 2*kEpsilon-bumped sum_hessian as the scan (C++ bumps at the FindBestThreshold call site, feature_histogram.hpp:174); every downstream gain-shift consumer must use the bumped value"
    - "Bounded-known-divergence handling: for a genuine f64 cross-feature gain tie on a degenerate node, assert the trees within tolerance bit-exact, require the diverging trees to still match C++ TOPOLOGY exactly, bound the per-leaf diff, and hard-cap the count — never a blanket skip"

key-files:
  created:
    - .planning/phases/07-parity-completing-variants/07-D05-DECISION.md
    - .planning/phases/07-parity-completing-variants/07-01-SUMMARY.md
  modified:
    - crates/lgbm-compute/src/kernels/split.rs
    - crates/lgbm-treelearner/src/learner.rs
    - crates/lgbm-boosting/src/gbdt.rs
    - crates/oracle-harness/tests/boosting_parity.rs
    - crates/oracle-harness/tests/fixtures/kernels/split.txt
    - crates/oracle-harness/tests/fixtures/determinism/binary_bag1_es0_bfa1_subset_trace.txt
    - crates/oracle-harness/tests/fixtures/determinism/regression_l1_bag1_es0_bfa0_subset_trace.txt
    - xtask/cpp/kernel_capture.cpp
    - xtask/py/subset_determinism_capture.py
    - .planning/REFERENCE_MANIFEST.md
    - .planning/phases/06-gbdt-spine-core-objectives-metrics/deferred-items.md

key-decisions:
  - "D-05 = FAITHFUL-FIX. The source-built lib_lightgbm 4.6 FP trace proved the bagged-subset split-gain knife-edge (DEF-06-01 + regression_l1+bagging) was a SPLIT-GAIN OPERAND bug (raw vs 2*kEpsilon-bumped sum_hessian in min_gain_shift), NOT an irreducible f32/near-zero accumulation artifact. The deeper bagged-subset node's current_gain 0x4013fffff4924920 exceeds the C++ min_gain_shift 0x4013fffff492491f by ONE f64 ULP; the raw-sum_hessian Rust min_gain_shift (0x4013fffff4924925, 5 ULPs higher) rejected it."
  - "C++ applies the 2*kEpsilon bump at the FindBestThreshold call site (find_best_threshold_fun_(sum_gradient, sum_hessian + 2*kEpsilon, ...), feature_histogram.hpp:174), so BeforeNumerical computes gain_shift from the BUMPED sum_hessian. Fixed find_best_split_cpu (f64) + find_best_split_raw_f32_on (f32) + per_bin_gains + the kernel_capture.cpp transcription to use the bumped value."
  - "regression_l1 + bagging UN-DEFERRED (06-06 Task 2b typed-reject reversed). With the min_gain_shift fix + the no-split ObtainAutomaticInitialScore fallback (the bfa-off constant tree-0 = label median 11.0, not 0.0), the 4 regression_l1_bag1_* cells reproduce the real-binary structure and assert parity."
  - "The non-bagged regression_l1 bfa-off cells retain a BOUNDED, hard-capped cross-feature L1 gain tie on degenerate 2-row nodes (two features separate the node perfectly with gains equal to ~1 f64 ULP; Rust vs C++ order them oppositely). This is a genuine f64-accumulation knife-edge (the documented uniform_grad_residual family), distinct from the fixed operand bug; the matrix requires the topology to match C++ exactly, bounds |leaf diff| < 0.1, and caps the count. Pre-fix this cell grew degenerate single-leaf STUBS that the old rl.len()!=gl.len() guard SKIPPED — the fix is a strict improvement."

patterns-established:
  - "Source-built reference FP trace as the arbiter for a split-gain knife-edge (extends the 05-09 technique to the bagged subset / per-candidate gain layer)"
  - "min_gain_shift / cnt_factor / scan all consume the SAME bumped sum_hessian — the C++ call-site bump invariant"

requirements-completed: []

# Metrics
duration: ~1 session (build instrumented binary + FP trace + faithful fix + un-defer + regression)
completed: 2026-06-07
---

# Phase 7 Plan 01: D-05 Bagged-Subset Split-Gain Determinism — FAITHFUL-FIX

**The bagged-subset split-gain knife-edge (DEF-06-01 + the typed-rejected regression_l1 + bagging) was a faithfully-fixable SPLIT-GAIN OPERAND bug, proven by a source-built lib_lightgbm 4.6 FP execution trace: the Rust `min_gain_shift` used the RAW leaf `sum_hessian` where C++ uses the `2*kEpsilon`-BUMPED value, making the Rust gain-shift ~7 ULPs too high and rejecting bagged-subset splits whose `current_gain` exceeds the C++ `min_gain_shift` by a single f64 ULP. Fixing it (lgbm-compute `find_best_split` f64+f32, the `per_bin_gains` diagnostic, the `kernel_capture.cpp` transcription + golden) closes DEF-06-01 bit-exact, un-defers regression_l1 + bagging, and settles D-05 before any bagging-dependent wave (GOSS W4, RF W6).**

## Performance

- **Duration:** ~1 session (CMake-build the instrumented CPU-only single-thread binary into /tmp, capture the FP trace, apply the faithful fix + the no-split fallback + the un-defer, regenerate the split golden, full regression).
- **Completed:** 2026-06-07
- **Tasks:** Continuation of 07-01 — Tasks 1–2 + the automatable part of Task 3 were committed by the prior executor (`d021b9f`/`68cd7d7`/`48e6255`); this session applied the human's D-05 decision (faithful-fix) as the closing edit.

## The source-built FP execution trace (Phase-5 05-09 technique)

Built `lib_lightgbm` 4.6 (`VERSION.txt = 4.6.0.99`) CPU-only single-thread
(`-DUSE_GPU=OFF -DUSE_CUDA=OFF -DUSE_OPENMP=OFF -DBUILD_CLI=ON`) into `/tmp` against
the repo's populated `external_libs` (eigen/fmt/fast_double_parser). Instrumented
`FindBestThresholdSequentially` (env-gated on `LGBM_FP_TRACE`) to dump the per-node
HEADER (`sum_gradient`/`sum_hessian`/`cnt_factor`/`min_gain_shift`), the per-bin
`SUBSET_HIST`, and the per-candidate `current_gain` vs `min_gain_shift` with the
accept flag. Drove the EXACT `binary_bag1_es0_bfa1` cell (seed `0x60057000`,
`bagging_fraction=0.7 bagging_freq=1 bagging_seed=3`); the CLI model came out
BIT-IDENTICAL to the wheel-captured trace (`split_gain = 3.5999999… 8.881779…e-16
4.440889…e-16`), confirming the source build reproduces the reference. `LightGBM/`
and the `/tmp` build were NEVER git-added; the C++ instrumentation was reverted.

## The decisive operand (1 f64 ULP)

| node | quantity | value | bits | accept |
|---|---|---|---|---|
| tree-0 node-1 (7 rows) | `current_gain` | 4.9999998297010109 | `0x4013fffff4924920` | |
| | C++ `min_gain_shift` (BUMPED) | 4.99999982970101 | `0x4013fffff492491f` | **C++ ACCEPTS (+1 ULP)** |
| | Rust `min_gain_shift` (RAW) | 4.999999829701015 | `0x4013fffff4924925` | Rust REJECTS |
| tree-0 node-2 (4 rows) | `current_gain` | 2.8571427598291468 | `0x4006db6da9cbc144` | |
| | C++ `min_gain_shift` (BUMPED) | 2.8571427598291463 | `0x4006db6da9cbc143` | **C++ ACCEPTS (+1 ULP)** |

C++ bumps `sum_hessian` by `2*kEpsilon` at the `FindBestThreshold` call site
(`feature_histogram.hpp:174`), so `BeforeNumerical` divides by the bumped value. The
Rust port divided by the raw value → `min_gain_shift` ~7 ULPs too high → the deeper
bagged-subset splits (whose `current_gain` exceeds the C++ shift by 1 ULP) were
rejected, collapsing tree-0 from 4 leaves to 2.

## The faithful fix

1. `crates/lgbm-compute/src/kernels/split.rs` — `find_best_split_cpu` (f64) AND
   `find_best_split_raw_f32_on` (f32 hip): bump `sum_hessian` FIRST, feed the bumped
   value into `get_leaf_gain` for `min_gain_shift`.
2. `crates/lgbm-treelearner/src/learner.rs` — `per_bin_gains` diagnostic re-scan:
   same bumped-`sum_hessian` `min_gain_shift` (stay bit-identical to the live kernel).
3. `xtask/cpp/kernel_capture.cpp` — `EmitSCase`: the golden-capture transcription had
   the same raw-vs-bumped bug; fixed and `split.txt` regenerated (byte-idempotent,
   1–2 ULP shifts to match real C++). `kernel_parity` 4/4 bit-exact again.
4. `crates/lgbm-boosting/src/gbdt.rs` — `no_split_constant_value`: the C++
   `ObtainAutomaticInitialScore` fallback (`gbdt.cpp:418-429`) — the bfa-off
   `regression_l1` no-split FIRST tree carries the label median (11.0), not 0.0.
5. Un-deferred `regression_l1 + bagging` (removed the `UnsupportedConfig` reject);
   the 4 `regression_l1_bag1_*` cells assert real-binary parity.
6. Cleared DEF-06-01 in the Phase-6 deferred-items doc.

## Deviations from Plan

### [Rule 1 — Bug] kernel_capture.cpp transcription had the same min_gain_shift bug

**Found during:** running `kernel_parity` after the f64 `find_best_split` fix —
`split.txt`'s `reverse_winner` gain mismatched by 1 ULP.
**Issue:** `EmitSCase` computed `min_gain_shift` from the RAW `cs.sum_hessian` (the
golden-capture transcription inherited the same defect as the port). The golden
itself encoded the bug, so the fixed kernel diverged from the stale golden.
**Fix:** corrected `EmitSCase` to use the bumped `sum_hessian` (matching real C++),
regenerated `split.txt` via `cargo run -p xtask -- kernel-capture` (byte-idempotent,
verified by two consecutive regens).
**Files modified:** `xtask/cpp/kernel_capture.cpp`,
`crates/oracle-harness/tests/fixtures/kernels/split.txt`.
**Commit:** `7794bbc`.

### [Rule 2 — Missing critical functionality] no-split ObtainAutomaticInitialScore fallback

**Found during:** un-deferring `regression_l1 + bagging` — the bagged tree-0 came out
`rust:0.0 vs cpp:11.0`.
**Issue:** the Rust GBDT pushed `0.0` for a no-split FIRST tree under `bfa=false`,
while C++ applies the automatic init score (the label median) as the constant tree's
leaf value (`gbdt.cpp:418-429`).
**Fix:** added `GBDT::no_split_constant_value` mirroring the C++ fallback (both the
subset and full-corpus no-split branches).
**Files modified:** `crates/lgbm-boosting/src/gbdt.rs`.
**Commit:** `2db9e38`.

**Total deviations:** 2 auto-fixes (no architectural change, no user decision needed —
the D-05 branch decision was already made by the human).

## Bounded known-divergence (documented, not fabricated)

The non-bagged `regression_l1` bfa-off cells (`uniform_grad_residual`) retain a
BOUNDED cross-feature L1 gain tie on a degenerate 2-row node: two features both
separate the node perfectly with gains equal to ~1 f64 ULP, and Rust vs C++ order
them oppositely (C++ tree-6 3rd split = feature 1; Rust = feature 0). The tree
TOPOLOGY (split count + per-leaf row counts) matches C++ EXACTLY; only the two
swapped-leaf median-residual values differ (< 0.1; the matrix asserts this bound and
hard-caps the flipped-tree count). This is a genuine f64-accumulation knife-edge
distinct from the (fixed) operand bug. Pre-fix this cell grew DEGENERATE single-leaf
STUBS (trees 6/9/11 = `[12]`/`0.0`) that the old `rl.len() != gl.len()` guard SKIPPED
— the fix is a strict improvement (real topology-matching trees, bit-exact on the
untouched branch). No assertion was weakened.

## Verification

- `subset_determinism_diagnostic` — GREEN, HARD-asserts tree-0 leaf-count parity:
  `binary_bag1_es0_bfa1` rust=4 cpp=4 (DEF-06-01 closed); `regression_l1+bagging`
  rust==cpp (un-deferred).
- `boosting_parity` matrix — **26/26 GREEN**, incl. the un-deferred
  `regression_l1_bag1_*` cells (real-binary parity).
- `kernel_parity` — **4/4 bit-exact** (split golden regenerated, byte-idempotent).
- `learner_parity` — **12/12** incl. the keystone `spine_real_binary` /
  `mfb_pos_real_binary` (no learner regression).
- `cargo test --workspace` — **GREEN** (50 test binaries, 0 failed).
- `git status --porcelain LightGBM/` — empty (LightGBM/, its submodules, and the /tmp
  build never git-added; C++ instrumentation reverted).

## Task Commits

1. `7794bbc` — `fix(07-01)`: min_gain_shift from the 2*kEpsilon-bumped sum_hessian
   (f64 + f32 + per_bin_gains + kernel_capture.cpp transcription + split.txt golden).
2. `2db9e38` — `feat(07-01)`: un-defer regression_l1 + bagging; close DEF-06-01
   (gbdt no_split_constant_value + matrix re-point + deferred-items clear).
3. `bd215f7` — `docs(07-01)`: 07-D05-DECISION.md (source-build FP-trace evidence).
4. `30194f4` — `test(07-01)`: source-built FP trace fixtures + hardened
   subset-determinism diagnostic (+ manifest + capture-script cleanup).
5. (prior executor) `d021b9f`/`68cd7d7`/`48e6255` — Tasks 1–2 + wheel evidence.

## Self-Check: PASSED

- `07-01-SUMMARY.md` + `07-D05-DECISION.md` exist on disk.
- Fix commits `7794bbc` / `2db9e38` / `bd215f7` / `30194f4` present in history.
- `subset_determinism_diagnostic` runs and HARD-asserts tree-0 leaf-count parity for
  both cells.
- `cargo test --workspace` GREEN; `LightGBM/` never git-added.
