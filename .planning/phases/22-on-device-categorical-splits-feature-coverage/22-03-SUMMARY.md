---
phase: 22-on-device-categorical-splits-feature-coverage
plan: 03
subsystem: lgbm-compute / on-device categorical splits
tags: [categorical, bitset, split-finder, transcription, cubecl-cpu-anchor, ODL-22]
status: complete
requires:
  - "22-01: DeviceSplitInfo runtime cat_width slab (D-03), D-06 learner gate"
  - "22-02: GrowFeature native categorical fields (bin_to_category, cat_smooth, cat_l2, max_cat_threshold, max_cat_to_onehot, min_data_per_group)"
provides:
  - "lgbm_compute::kernels::categorical_split::construct_bitset (§6.3 host-faithful bit packer)"
  - "lgbm_compute::kernels::categorical_split::set_real_threshold (two-bitset producer: real category-value + inner-bin)"
  - "lgbm_compute::kernels::categorical_split::find_best_threshold_categorical (§8.1 one-hot + many-vs-many evaluator, single-owner f64 anchor)"
  - "lgbm_compute::kernels::categorical_split::CategoricalSplit (split decision + winner bins)"
affects:
  - "22-04: driver wiring consumes find_best_threshold_categorical + set_real_threshold; MUST honor the sum_hessian pre-bump caller contract"
tech-stack:
  added: []
  patterns:
    - "Single-owner CubeDim::new_1d(1) f64 anchor (def-f8u-01)"
    - "Crate-cycle-safe byte-for-byte transcription (no lgbm-treelearner import)"
    - "Index-only bitonic argsort reuse for the ctr sort (primitives::bitonic_argsort_on)"
key-files:
  created:
    - "crates/lgbm-compute/src/kernels/categorical_split.rs"
  modified:
    - "crates/lgbm-compute/src/kernels/mod.rs"
decisions:
  - "The ctr sort uses primitives::bitonic_argsort_on (f32 index-only) instead of the host f64 sort_by; committed cat_manyvsmany fixture is tie-free so both orders agree (A1)."
  - "cat_onehot fixture (num_bin=5 > max_cat_to_onehot=4) actually exercises the MANY-VS-MANY path; the true one-hot branch (num_bin <= max_cat_to_onehot) is covered by a synthetic num_bin=4 test."
metrics:
  duration: ~35m
  completed: 2026-07-02
  tasks: 2
  files: 2
  commits: 3
---

# Phase 22 Plan 03: On-Device Categorical Split Finder (§6.3 + §8.1) Summary

Transcribed the two genuinely-new categorical algorithms into a new `lgbm-compute`
module `categorical_split.rs`: the §6.3 bitset construction (`construct_bitset` +
the `set_real_threshold` two-bitset producer) and the §8.1 categorical split
evaluator (`find_best_threshold_categorical`, one-hot + many-vs-many, D-02). All
math is the single-owner `CubeDim::new_1d(1)` f64 anchor; the many-vs-many ctr sort
reuses `primitives::bitonic_argsort_on`. The module produces the split decision +
winner bitsets only — driver wiring is 22-04.

## What was built

- **§6.3 `construct_bitset`** — byte-for-byte from the host
  `feature_histogram_categorical.rs:424-435` (`n_blocks = max_val/32 + 1`; sequential
  OR `bits[v/32] |= 1<<(v%32)`; empty-input guard). No `Atomic`, no per-split alloc.
- **`set_real_threshold(cat_threshold_bins, bin_to_category, offset)`** — the
  Pattern-3 two-bitset producer (host learner.rs:3697-3721): the REAL category-value
  bitset via `bin_to_category.get(bin).copied().unwrap_or(bin)` (host-faithful
  bounds, T-22-06 — no panic, no OOB) plus the INNER-bin routing-key bitset.
- **§8.1 `find_best_threshold_categorical`** — the full one-hot + many-vs-many
  evaluator, transcribed from `feature_histogram_categorical.rs:93-326`, returning
  `CategoricalSplit { split: SplitInfo, cat_threshold: Vec<u32> }`.

## Host source lines transcribed

| Piece | Host source | Notes |
|-------|-------------|-------|
| `construct_bitset` | `feature_histogram_categorical.rs:424-435` | verbatim |
| `round_int` | `:75-77` | `(x + 0.5f32 as f64) as i32` |
| one-hot scan (lambda_l2) | `:138-208` | uses ORIGINAL `lambda_l2` |
| many-vs-many (`l2 += cat_l2`) | `:212-303` | increment AFTER one-hot return |
| ctr filter + sort | `:213-232` | `sort_by` → `bitonic_argsort_on` |
| `max_num_cat` clamp | `:236` | `min(max_cat_threshold, (used_bin+1)/2)` |
| bidirectional sweep | `:234-303` | verbatim |
| `finalize` + `+offset` cat_threshold | `:334-398` | verbatim |
| `set_real_threshold` (two-bitset) | `learner.rs:3697-3721` | real + inner |

## Fidelity landmines honored

- **Pitfall 2 (kEpsilon pre-bump):** the evaluator does NOT bump internally; the
  `sum_hessian` argument arrives ALREADY `+2*kEpsilon` bumped. This is documented on
  the fn as a **caller contract** — **22-04's driver branch MUST bump `sum_h` before
  the call** (mirroring host learner.rs:2813-2814).
- **Pitfall 3 (l2 asymmetry):** one-hot uses `cfg.lambda_l2`; many-vs-many uses
  `cfg.lambda_l2 + cfg.cat_l2`; `gain_shift` always uses the original `lambda_l2`.
  Pinned by `onehot_branch_uses_original_l2_not_cat_l2` (huge cat_l2 does not change
  the one-hot gain) and `manyvsmany_adds_cat_l2` / `manyvsmany_gain_uses_cat_l2_bit_exact`.
- **Pitfall 1 / A1 (ctr tie order):** `cat_manyvsmany` ctr values are
  `0,-1,-2,-8,-9,-10` — **strictly distinct (tie-free)**, confirmed by
  `manyvsmany_ctr_values_are_tie_free`. So the f32-bitonic order equals the f64
  stable-sort order; the winning `cat_threshold` matches the golden.

## Parity / test results

`cargo test -p lgbm-compute --lib categorical_split` → **14 passed, 0 failed.**

- `construct_bitset` cases: `[]`, `[0]`, `[5]`, `[0,31,32]`, `[0,1,33]`.
- `set_real_threshold` REAL bitset pinned to committed real-4.6 goldens:
  **cat_onehot = `8`** (bit 3), **cat_manyvsmany root = `56`** (bits 3,4,5).
- Evaluator-vs-golden (full path): cat_onehot winner bin `[4]` → real bitset `[8]`,
  net gain **250.0**; cat_manyvsmany root winner bins `[6,5,4]` → real bitset `[56]`,
  net gain **345.0** (bit-exact vs the independent `get_split_gains` `+cat_l2` anchor).
- One-hot branch (`num_bin=4 <= max_cat_to_onehot=4`) exercised by
  `onehot_branch_picks_single_category` (single category, `default_left == false`).

`cargo test --workspace` (env unset) → **green, 0 failed** (SC #4: additive module,
numeric spine byte-unchanged).

## Return shape for the 22-04 SplitScalars mapping

`find_best_threshold_categorical` returns `CategoricalSplit`:
- `split: SplitInfo` — `gain` (net of `min_gain_shift`), `left/right_count`,
  `left/right_sum_gradient`, `left/right_sum_hessian` (with kEpsilon subtracted back
  off), `left/right_output`, `default_left = false`, `threshold = 0` (unused).
- `cat_threshold: Vec<u32>` — the winning REAL BINS, each already `+ offset`
  (one-hot: one bin; many-vs-many: `num_cat_threshold` bins in the winning
  direction). 22-04 feeds these to `set_real_threshold` to get the real + inner
  bitsets for `split_categorical_on_device` / `partition_categorical_on_device`, and
  maps `split` into the numeric `SplitScalars` (`num_cat_threshold = cat_threshold.len()`).

## Deviations from Plan

None — plan executed as written. One clarification surfaced during execution: the
`cat_onehot` fixture has `num_bin=5 > max_cat_to_onehot=4`, so it flows through the
**many-vs-many** code path (selecting a single category, reproducing the golden). The
true one-hot code branch is covered by a synthetic `num_bin=4` test. This matches the
host behavior exactly (the fixture name reflects the one-category *result*, not the
code path) and does not change any pinned golden.

## Self-Check: PASSED

- FOUND: `crates/lgbm-compute/src/kernels/categorical_split.rs`
- FOUND commits: `6353ff0` (Task 1), `8fa2496` (RED), `ab0e66e` (GREEN)
