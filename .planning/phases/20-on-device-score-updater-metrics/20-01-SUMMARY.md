---
phase: 20-on-device-score-updater-metrics
plan: 01
subsystem: on-device-score-updater
tags: [score-updater, ODL-16, D-02, resident-cuda_score_, host-mirror-toggle]
status: complete
requires:
  - "kernels/score_updater.rs empty stub registered by Plan 20-00"
  - "Phase-18 add_prediction_to_score_on_device (predict.rs:212) — the per-leaf D-02 delegate"
  - "cuda_on_device_enabled() OnceLock env gate (lib.rs:1314)"
  - "objective_regression.rs convert_output_on launcher skeleton"
provides:
  - "add_score_constant_on / multiply_score_constant_on device launchers (§11 constant ops)"
  - "ScoreUpdater.boosting_on_cuda_ toggle + set_boosting_on_cuda driver seam"
  - "ScoreUpdater resident methods add_constant_on / multiply_score_on / add_tree_train_path_on_device"
affects:
  - "Plan 20-04 (device driver) — sets set_boosting_on_cuda from on_device_eligible; builds the PredictTree walk inputs for add_tree_train_path_on_device"
tech_stack:
  added: []
  patterns:
    - "elementwise #[cube] over Array<f64> at offset = num_data * tree_id (ABSOLUTE_POS is usize; scalar params cast `as usize`)"
    - "resident whole-buffer op + read-back mirror = CopyFromCUDADeviceToHost analog"
    - "boosting-layer device ops generic over B: Backend (client: &ComputeClient<B::Runtime>) — no cubecl runtime named (CMP-01 holds)"
key_files:
  created:
    - crates/oracle-harness/tests/score_updater_parity.rs
  modified:
    - crates/lgbm-compute/src/kernels/score_updater.rs
    - crates/lgbm-boosting/src/score_updater.rs
    - crates/lgbm-boosting/Cargo.toml
decisions:
  - "ABSOLUTE_POS is usize in this cubecl 0.10 build (predict.rs idiom): the #[cube] bodies compare `i < num_data as usize` and index `score[offset as usize + i]` — the initial u32-index draft failed to compile (NativeExpand<u32> vs NativeExpand<usize>)."
  - "Per the plan's explicit Task-2 action + the frontmatter key_links, the boosting ScoreUpdater names the kernel launchers (add_score_constant_on / add_prediction_to_score_on_device) DIRECTLY rather than routing through a new Backend-trait method. Stays generic over B::Runtime so CMP-01's no-runtime gate still holds; the Cargo.toml comment was widened to document this."
  - "add_tree_train_path_on_device takes the reconstructed PredictTree walk inputs (predict_tree/rows/num_features/bit_type) as arguments — the ScoreUpdater does not own the Dataset bins, so the Plan-20-04 driver reconstructs them (the L2 identity-binned scatter contract)."
metrics:
  tasks_completed: 2
  files_created: 1
  files_modified: 3
  duration_minutes: 30
  completed_date: 2026-07-02
---

# Phase 20 Plan 01: On-Device Score Updater (§11, ODL-16) Summary

Implemented the §11 on-device score updater: two whole-array scalar `#[cube]` kernels over the resident `double` score buffer (`AddScoreConstant` = `score[offset+i] += val`, `MultiplyScoreConstant` = `score[offset+i] *= val`, `offset = num_data * tree_id`), plus the boosting-layer `boosting_on_cuda_`-keyed host-mirror toggle that routes the three training-path score ops to the device (per-leaf `AddScore` delegating to the Phase-18 `add_prediction_to_score_on_device` — D-02, no new tree-walk kernel) and mirrors the resident buffer back to the host `score_` when off. Anchored bit-exact to the host `ScoreUpdater` on the cpu f64 backend.

## What Was Built

### Task 1 — §11 constant-op kernels + kernel-level A/B (commit 10bcd1e)
`crates/lgbm-compute/src/kernels/score_updater.rs` (filled the Plan-00 stub): two elementwise `#[cube]` bodies (`add_constant_body` / `multiply_constant_body`) over an `Array<f64>`, bounds-guarded `i < num_data as usize` and indexing `score[offset as usize + i]`, each wrapped in a `#[cube(launch_unchecked)]` f64 anchor kernel and a host launcher (`add_score_constant_on` / `multiply_score_constant_on`) following the `objective_regression.rs::convert_output_on` skeleton (exact-size `create_from_slice` → `cube_count = num_data.div_ceil(256)` → bounds-guarded `launch_unchecked` → `read_one_unchecked`, `unsafe` confined to the launch site). A shared `checked_slice` helper validates `num_data >= 0`, `tree_id >= 0`, overflow-safe `offset` (the `usize` arithmetic from `score_updater.rs:64`), and that `[offset, offset + num_data)` fits inside the buffer BEFORE any device alloc (T-20-01-01/02). The launchers take the whole resident buffer and return the whole updated buffer (the host-mirror shape). Five module unit tests (non-root add/multiply match a hand reference, zero-`num_data` no-op, negative-input rejection, out-of-range-slice `LengthMismatch`).

`crates/oracle-harness/tests/score_updater_parity.rs` (new), cell `constant_ops_kernel_matches_host_score_updater`: builds a 2-class × 3-row f64 score vector, applies the host `ScoreUpdater::{add_constant, multiply_score}` at a NON-root `tree_id = 1` (proving the `offset = num_data * tree_id` slice) plus a root-id add, and asserts the device kernels are bit-exact (`compare_exact_f64_bits`) on the cpu f64 anchor.

### Task 2 — boosting_on_cuda_ toggle + per-leaf AddScore delegation (commit c67ec99)
`crates/lgbm-boosting/src/score_updater.rs` extended ADDITIVELY:
- A `boosting_on_cuda: bool` field initialised in `new` from `lgbm_compute::cuda_on_device_enabled()` (OFF by default with `LGBM_CUDA_ON_DEVICE` unset — D-09), a `set_boosting_on_cuda` driver seam (Plan 20-04), and a `boosting_on_cuda()` getter.
- Resident methods `add_constant_on` / `multiply_score_on` (generic `<B: Backend>`, `client: &ComputeClient<B::Runtime>`): when the toggle is on, run the §11 kernel over the resident buffer and mirror the result back into `self.score` (`CopyFromCUDADeviceToHost` analog); when off, delegate to the byte-unchanged host method.
- `add_tree_train_path_on_device` (D-02): delegates the per-leaf training-path `AddScore` to the Phase-18 `add_prediction_to_score_on_device` tree-walk kernel — NO new tree-walk kernel — adding the returned `num_data`-length per-row raw-margin delta into class `cur_tree_id`'s slice.
- The DART/RF per-row-predict paths `add_tree_predict_path` / `add_tree_scaled_all` are explicitly left on the host path (out of the continuous proving slice), documented in a code comment.

The `Cargo.toml` dependency comment was widened to document that `lgbm-compute` is now used for the §11 launchers too — still TRAITS/TYPES/FUNCTIONS only, never a GPU runtime (CMP-01's no-runtime gate holds because the methods stay generic over `B::Runtime`).

Two parity cells: `resident_toggle_mirrors_host_accumulation` (init-add + shrinkage-multiply over two classes; forced-on resident mirror bit-exact to a pure-host `ScoreUpdater`; asserts the env-unset default reports `boosting_on_cuda == false`) and `resident_train_path_delegates_to_phase18_kernel` (the numeric Phase-18 parity tree; the device delegate's class-1 slice == base + golden margins, class 0 untouched).

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking issue] `ABSOLUTE_POS` is `usize`, not `u32`, in this cubecl build**
- **Found during:** Task 1 first compile.
- **Issue:** The RESEARCH.md code sketch (`if i < num_data { score[offset + i] += val; }`) with `num_data`/`offset` as `u32` failed to compile — 12 `NativeExpand<u32>` vs `NativeExpand<usize>` errors — because `ABSOLUTE_POS` is `usize` and `Array` indexing is `usize` in cubecl 0.10 (confirmed by the `predict.rs` idiom `if i < num_rows as usize`).
- **Fix:** The `#[cube]` bodies compare `i < num_data as usize` and index `score[offset as usize + i]`. The launcher scalar params stay `u32` (matching the `predict.rs` kernel signatures).
- **Files modified:** crates/lgbm-compute/src/kernels/score_updater.rs
- **Commit:** 10bcd1e

### Implementation choices (within plan scope)

- **Boosting names the kernel launchers directly.** The plan's Task-2 action ("route `add_constant` to `add_score_constant_on` … the per-leaf `add_tree_train_path` to `add_prediction_to_score_on_device`") and the frontmatter key_links require these exact symbols in `lgbm-boosting/src/score_updater.rs`. This is a departure from the usual Backend-trait-mediation discipline (production compute normally flows through `Backend` impls), but it is plan-directed and stays runtime-free — the methods are generic over `B::Runtime`, so no `cubecl` runtime is named and CMP-01's no-runtime gate holds.
- **`add_tree_train_path_on_device` takes the PredictTree walk inputs as arguments.** The ScoreUpdater does not own the `Dataset` bins needed to reconstruct a `PredictTree` from a grown `Tree`; that reconstruction belongs to the Plan-20-04 device driver. The method therefore accepts `predict_tree`/`rows`/`num_features`/`bit_type` and threads them to the Phase-18 kernel — the full cross-iteration residency wiring is Plan 04 (as the plan objective states).

## Verification

- `cargo test -p oracle-harness --test score_updater_parity` — 3 passed (constant-op kernel A/B, resident toggle mirror, per-leaf D-02 delegate).
- `cargo test -p lgbm-compute --lib score_updater` — 5 passed (kernel bounds/offset/validation units).
- `cargo test -p lgbm-boosting` — 55 passed with `LGBM_CUDA_ON_DEVICE` unset (host `ScoreUpdater` path byte-unchanged; the added field defaults OFF, no snapshot/PartialEq breakage).
- `cargo build --workspace` — green with the env unset.
- `cargo clippy -p lgbm-compute -p lgbm-boosting` — no findings in the changed files.
- `unsafe` appears only at the two `launch_unchecked` sites (bounds guard `i < num_data` inside each `#[cube]`).
- No f64 introduced into any per-row grow/build hot loop — the f64 here is the reference-blessed resident score buffer (D-08).

## Threat Mitigations Applied

- **T-20-01-01** (OOB device read/write): `checked_slice` proves `[offset, offset + num_data) ⊆ [0, score.len())` before launch; the `#[cube]` bodies bounds-guard `i < num_data`; buffers sized exactly with `create_from_slice`; `unsafe` confined to the `launch_unchecked` sites. Unit test `out_of_range_slice_rejected`.
- **T-20-01-02** (integer overflow in `offset = num_data * tree_id`): overflow-safe `usize` `checked_mul`/`checked_add` (the `score_updater.rs:64` pattern); `num_data >= 0` / `tree_id >= 0` validated. Unit test `negative_inputs_rejected`.
- **T-20-01-03** (env-gate misconfig mutating the default host path): the toggle keys on the OnceLock `cuda_on_device_enabled()`; `cargo test -p lgbm-boosting` green with env unset proves byte-unchanged, and the parity cell asserts the default `ScoreUpdater` reports `boosting_on_cuda == false`.
- **T-20-01-SC** (installs): no package installs this plan.

## Self-Check: PASSED

- All 3 source files (kernels/score_updater.rs, lgbm-boosting/score_updater.rs, oracle-harness/tests/score_updater_parity.rs) FOUND on disk.
- Commits 10bcd1e (Task 1) and c67ec99 (Task 2) FOUND in git history.
- `kernels/score_updater.rs` is 281 lines (>= 40 min) and contains `pub fn add_score_constant_on`.
