---
phase: 18-on-device-data-partition-tree-mutation-prediction
plan: 04
subsystem: compute
tags: [cubecl, predict, tree-walk, categorical, find-in-bitset, hip-parity, ODL-15]

# Dependency graph
requires:
  - phase: 18-on-device-data-partition-tree-mutation-prediction
    plan: 01
    provides: predict.rs stub + predict.txt tree-walk golden (numeric + cat_onehot + cat_manyvsmany) + predict_parity on_device/cat scaffolds
  - phase: 18-on-device-data-partition-tree-mutation-prediction
    plan: 02
    provides: shared pub(crate) find_in_bitset #[cube] helper (18-02 data_partition.rs) reused by the predict cat branch
  - phase: 15-device-dataset
    provides: ColumnFeatureMeta 8/16/32 bit_type + per-feature numeric meta (column_data.rs)
provides:
  - predict.rs AddPredictionToScoreKernel<USE_INDICES> #[cube] tree-walk (numeric 8/16/32 + categorical membership) over the §13 columnar store, f64 score accumulator only
  - §9 AddPredictionToScoreKernel<USE_BAGGING> per-row leaf-map gather-add
  - add_prediction_to_score_on_device / add_prediction_bagging_on_device host drivers + PredictTree view
  - predict_parity on_device (numeric 8/16/32) + cat (cat_onehot/cat_manyvsmany) cells un-ignored, device-driven, bit-exact + within ORACLE_TOL
  - kernel_parity.rs hip f32 predict parity cell (rocm-gated, tie-aware vs cpu f64 anchor, never GPU-vs-GPU)
affects: [19-objective-inverse-link, 21-grow-loop-wiring]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Runtime multi-node tree-walk with a `while node >= 0` cube loop (random.rs precedent) — data-dependent iteration in a #[cube(launch)] kernel"
    - "Shared find_in_bitset reuse across a sub-bitset via GLOBAL-position indexing (word = start + bin/32, n = start + len) so the 0-based helper reads the correct pool word without an Array sub-view"
    - "Native-width column dispatch via a `<B: Int>` monomorph + a bit_type launch macro (mark-kernel precedent); a bin is a width-invariant index, so the SAME numeric model is replayed at 8/16/32 for D-05 coverage"
    - "Predict hip parity: integer routing + single f64 leaf write is deterministic, so the hip walk equals the cpu f64 anchor; the tie-aware assert_within surfaces any residual as a documented gap"

key-files:
  created: []
  modified:
    - crates/lgbm-compute/src/kernels/predict.rs
    - crates/oracle-harness/tests/predict_parity.rs
    - crates/oracle-harness/tests/kernel_parity.rs

key-decisions:
  - "The numeric predict route is transcribed from cuda_tree.cu:376-391 (a DISTINCT, simpler reference than 18-02's dense_bin SplitInner `route_to_left` fan-out); route_to_left's #[comptime] flag signature is structurally incompatible with a runtime multi-node walk that reads per-node missing_type/default_left. Pitfall-4 sharing is honored where the functions actually coincide: the categorical membership genuinely CALLS the shared find_in_bitset."
  - "The predict kernel builds its tree/feature arrays from the predict.txt fixture (plan-check W2), NOT 18-03's live DeviceCudaTree; field names mirror the flat CUDATree layout to minimise Phase-21 rework. Unifying predict to walk the exact tree SplitKernel built is an explicit Phase-21 concern."
  - "The tree-walk `bins` matrix is the FULL [num_data × num_features] column store indexed by the walk's data_index (NOT the unit index); num_rows is the launched-unit count (= num_data identity, = used_indices.len() subset), matching the reference USE_INDICES semantics."

patterns-established:
  - "GLOBAL-position bitset indexing to reuse a 0-based #[cube] membership helper over a per-node sub-bitset slice"
  - "Replay one width-invariant model at 8/16/32 native column widths for D-05 dispatch coverage from a single golden"

requirements-completed: [ODL-15]

# Metrics
duration: 13min
completed: 2026-07-01
status: complete
---

# Phase 18 Plan 04: On-Device Prediction Tree-Walk (ODL-15) Summary

**The §10 `AddPredictionToScoreKernel<USE_INDICES>` device tree-walk over the §13 columnar store (numeric 8/16/32 threshold+missing/default routing and categorical bitset membership reusing the shared `find_in_bitset`), the §9 `USE_BAGGING` leaf-map gather-add, and the hip f32 parity gate — numeric + categorical predict match the `predict.txt` golden bit-exact on the cpu f64 anchor and within ~1e-6 on the local ROCm box, with the merge gate green (`LGBM_CUDA_ON_DEVICE` unset).**

## Performance
- **Duration:** ~13 min
- **Started:** 2026-07-01T12:39Z
- **Completed:** 2026-07-01T12:52Z
- **Tasks:** 2
- **Files modified:** 3

## Accomplishments
- **Tree-walk kernel (Task 1, D-05/D-14):** `add_prediction_to_score_kernel<B: Int>` walks `node = 0` while `node >= 0` (a data-dependent cube `while` loop), reads the row's native-width bin (`u32::cast_from`), remaps it (`bin ∈ [min,max] ? bin−min+offset : most_freq_bin`), routes numerically (the verbatim `cuda_tree.cu:376-391` missing/default → threshold decision) or categorically, and adds `leaf_value[~node]` into an `f64` score accumulator — the ONLY f64 in the loop (bin reads stay native-width integer, SP-5/D-14). Verified bit-exact vs the golden for all 5 numeric rows (including the missing-sentinel `remap == default_bin` and the out-of-range `→ most_freq_bin` rows) at 8-, 16-, and 32-bit column width.
- **Shared `find_in_bitset` reuse (Pitfall 4):** the categorical branch CALLS the 18-02 `pub(crate)` `find_in_bitset` helper over the node's sub-bitset (`bitset_inner[cat_boundaries_inner[cat_idx] ..]`) via GLOBAL-position indexing (`word = start + bin/32`), preserving the `pos ≥ n → 0` bound (T-18-03) — no duplicate bitset impl. Matches both cat models (one-hot 8-bit `mfb0→offset1`; many-vs-many 16-bit members `{2,3,5}`) bit-exact.
- **§9 leaf-map gather-add (D-06):** `add_prediction_bagging_kernel` adds `leaf_value[leaf_map[data_index]]` into the f64 score (identity + `USE_BAGGING` subset), unit-tested.
- **Parity cells (Tasks 1–2):** `predict_parity::on_device` (numeric, driven at 8/16/32) + `predict_parity::cat` (cat_onehot/cat_manyvsmany) un-ignored — the `predict.txt` parser extended to the full model (PFEAT/PNODE/PLEAF/PCATPOOL/PCATBOUND), each asserted bit-exact on the cpu f64 anchor AND within `ORACLE_TOL`.
- **hip f32 gate (Task 2, D-03a/def-f8u-01):** `kernel_parity_predict_within_tol_on_hip` (rocm-gated) runs the numeric + categorical walk on cubecl-hip and asserts within ~1e-6 of the cpu f64 anchor via the tie-aware `assert_within` — never GPU-vs-GPU; cross-checks the cpu anchor reproduces the golden; skips cleanly with no GPU. **Passes on the local ROCm hardware.**

## Task Commits
1. **Task 1: numeric tree-walk (8/16/32) + §9 leaf-map add + on_device cell** — `c0f45bd` (feat)
2. **Task 2: categorical membership predict cell + hip f32 parity gate** — `1f72854` (feat)

## Files Created/Modified
- `crates/lgbm-compute/src/kernels/predict.rs` - Filled the Wave-0 stub: the `AddPredictionToScoreKernel<USE_INDICES>` numeric+categorical tree-walk, the §9 `USE_BAGGING` leaf-map gather-add, the `PredictTree` view + two host drivers with SP-4 validation, and 5 in-crate cpu-anchor tests.
- `crates/oracle-harness/tests/predict_parity.rs` - Extended the `predict.txt` parser to the full synthetic model; un-ignored + device-drove the `on_device` (8/16/32) and `cat` cells.
- `crates/oracle-harness/tests/kernel_parity.rs` - Added the rocm-gated `kernel_parity_predict_within_tol_on_hip` cell (self-contained predict-block parser + tie-aware anchor compare).

## Decisions Made
- **Numeric route is a distinct reference, not `route_to_left` (see Deviations).**
- **Fixture-built tree arrays, not the live `DeviceCudaTree` (plan-check W2):** standalone parity this phase; field names mirror the flat CUDATree layout for Phase-21.
- **`bins` = full `[num_data × num_features]` store indexed by `data_index`:** the reference USE_INDICES semantics — `num_rows` is the launched-unit count, not the store height.

## Deviations from Plan

### Design clarification (no user permission needed — Rule 3 class)

**1. [Scoping] The numeric predict route is transcribed locally, not shared with `route_to_left`**
- **Found during:** Task 1 (numeric tree-walk).
- **Issue:** The plan's must-have and RESEARCH call for the numeric route to "reuse the SAME shared route `#[cube]` fn as the partition mark (Pitfall 4)". However, 18-02 built `route_to_left` as the dense_bin `SplitInner` full **comptime** flag fan-out, whereas the numeric predict decision (`cuda_tree.cu:376-391`) is a **distinct, simpler** function operating on the already-remapped bin with a **runtime** per-node `missing_type`/`default_left` (read from `decision_type`). Two blockers: (a) they are different reference functions — the predict route never uses `min_is_max`/`mfb_is_zero`/`mfb_is_na`/`ftm`; (b) `route_to_left`'s `#[comptime] bool` flags cannot be fed runtime per-node values inside a multi-node `while` walk (cubecl monomorphises per flag combination at expansion time).
- **Fix:** Transcribed the `cuda_tree.cu:376-391` numeric route **once** inline in the walk (runtime flags). Pitfall-4 sharing is honored where the functions genuinely coincide: the **categorical** membership CALLS the shared `find_in_bitset` helper (no duplicate bitset impl). Verified bit-exact vs the golden across all rows/widths.
- **Files modified:** crates/lgbm-compute/src/kernels/predict.rs
- **Verification:** `cargo test -p lgbm-compute --lib predict` + `predict_parity::on_device` green (bit-exact + within ORACLE_TOL); grep confirms no duplicate `find_in_bitset`; f64 confined to the score accumulator + scalar leaf value (D-14).
- **Committed in:** c0f45bd (Task 1 commit)

---

**Total deviations:** 1 design clarification, 0 bugs. No architectural change (Rule 4 not triggered). The numeric route is a single transcription of its own reference; the shared `find_in_bitset` is reused per the plan.

## Known Stubs
None — the numeric + categorical device walk paths are fully wired to the goldens; no placeholder/empty-data sinks. `on_device_growth_supported()` stays `false` (these kernels are parity-test-only, no live grow-loop — as required).

## Threat Flags
None — no new network/auth/file surface. The host-boundary validation (`validate_walk`, SP-4/T-18-06) bounds `rows`/meta/tree lengths, used-index range, and leaf indices before the confined `unsafe` launch; `find_in_bitset` keeps the `pos ≥ n → 0` guard (T-18-03). No packages installed (T-18-SC accept).

## Verification
- `cargo test -p oracle-harness --test predict_parity` — 7/7 (incl. `on_device` 8/16/32 + `cat` one-hot/many-vs-many) bit-exact on the cpu f64 anchor + within ORACLE_TOL.
- `cargo test -p lgbm-compute --lib predict` — 5/5 (numeric 8/16/32, categorical, USE_INDICES subset, §9 bagging).
- `cargo test -p oracle-harness --features rocm --test kernel_parity kernel_parity_predict_within_tol_on_hip` — GREEN on the local ROCm box (hip f32 within ~1e-6 of the cpu f64 anchor, tie-aware, never GPU-vs-GPU).
- `cargo test --workspace` — GREEN with `LGBM_CUDA_ON_DEVICE` unset (ODL-19 merge gate); 75 test binaries pass, no failures.
- clippy clean on all new code (`predict.rs`, the two parity cells).
- D-14 review grep: f64 in the walk appears ONLY in the score accumulator + scalar `leaf_value`; bin reads are native-width `Array<B>`; no on-device `ConvertOutput`/inverse-link.

## Next Phase Readiness
- **ODL-15 complete:** on-device prediction reproduces the reference tree-walk (numeric 8/16/32 + categorical membership) bit-for-bit on the cpu f64 anchor and within ~1e-6 on ROCm.
- **Phase 19 (objective inverse-link):** the kernel emits the RAW pre-link margin; the inverse-link boundary is host-side at readback, ready to move on-device.
- **Phase 21 (grow-loop wiring):** unifying predict to walk the exact tree `SplitKernel` built (consume the live `DeviceCudaTree` instead of fixture arrays) is the explicit follow-up; the `PredictTree` field names already mirror the flat layout.
- No blockers.

---
*Phase: 18-on-device-data-partition-tree-mutation-prediction*
*Completed: 2026-07-01*

## Self-Check: PASSED

All 3 modified source files + the SUMMARY exist on disk; both task commits (c0f45bd, 1f72854) present in git history.
