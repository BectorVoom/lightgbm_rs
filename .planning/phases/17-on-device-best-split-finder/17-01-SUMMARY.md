---
phase: 17-on-device-best-split-finder
plan: 01
subsystem: compute
tags: [cubecl, best-split, split-finder, cpu-anchor, parity, fixtures, gsd-wave-0]

# Dependency graph
requires:
  - phase: 16-on-device-histogram-constructor
    provides: "interleaved [2b]/[2b+1] hist_in_leaf histogram the stage-1 scan reads"
  - phase: 14-foundation-shared-device-primitives-device-structs-rng
    provides: "split_info::SplitScalars/DeviceSplitInfo CUDASplitInfo analog + CUDARandom LCG"
provides:
  - "SplitFindTask struct (12-field mirror of cuda_best_split_finder.hpp:28-41)"
  - "build_split_find_tasks host task-gen builder (assume_out_default_left table, cpp:137-227)"
  - "round_ties_even + branch-free fallback (__double2int_rn, DIVERGES from split.rs::round_int)"
  - "Stage1Scalars leaf-total/config record + stage-1/2/3 launcher STUB signatures"
  - "best_split_parity.rs golden-anchor harness (mirrors kernel_parity.rs)"
  - "best_split.txt 10-record D-07 fixture matrix (default/L1/smoothing/rand/empty/globalmem/tie)"
affects: [17-02, 17-03, 17-04, 17-05, on-device-best-split-finder]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Wave-0-first: pure-host scaffolding (task-gen table + count rounding) + golden harness land BEFORE the kernel; downstream waves drive best_split_parity.rs"
    - "Landmine-first: the two riskiest parity divergences (round-ties-even count recovery D-01, default_left != reverse Pitfall 3) locked as tested pure-host units"
    - "Stage launcher STUBS return is_valid=false / best_leaf=-1 sentinels so the harness RED-fails via ASSERTION not a build error (Nyquist scaffold); #[ignore]d to keep the merge gate green"
    - "Module registered UNGATED (not #[cfg(feature=gpu)]) so the cpu f64 anchor exercises it (D-08); additive + OFF behind LGBM_CUDA_ON_DEVICE (D-09)"

key-files:
  created:
    - "crates/lgbm-compute/src/kernels/best_split.rs"
    - "crates/oracle-harness/tests/best_split_parity.rs"
    - "crates/oracle-harness/tests/fixtures/kernels/best_split.txt"
  modified:
    - "crates/lgbm-compute/src/kernels/mod.rs"

key-decisions:
  - "Reused lgbm_dataset::MissingType {None,Zero,NaN} for FeatureMeta rather than defining a new enum (avoids drift from the canonical bin.h port)"
  - "Stage-1 parity test marked #[ignore = 'Wave-0 scaffold; un-ignore when 17-03 lands'] — the fixture IS committed so an un-ignored test would go red against the sentinel stub; #[ignore] is the only way to keep cargo test --workspace green (D-09/ODL-19) with the fixture present"
  - "best_split.txt SWIN goldens are Wave-0 STRUCTURAL PLACEHOLDERS with a PROVENANCE GUARD comment; 17-03 finalizes them by hand-transcription from cuda_best_split_finder.cu §8.1, NOT by copying the cpu-fold output (anti-circularity)"
  - "Open Q1 locked in the module doc: per-task RNG seed = extra_seed + task_index (cuda_best_split_finder.cu:2220-2228) — no open flag remains for the Wave-2 USE_RAND path"

patterns-established:
  - "round_ties_even provides both the f64::round_ties_even intrinsic AND a branch-free even-round identity (x>=0), proven byte-equivalent by unit test — Wave 2 picks whichever cubecl-cpu lowers inside #[cube] (A1 unverified)"
  - "default_left is decoupled from reverse: build_split_find_tasks precomputes assume_out_default_left from the missing type; the num_bin<=2 NaN feature is the genuine reverse=true && assume=false divergence case"

requirements-completed: []

# Metrics
duration: 11min
completed: 2026-07-01
status: complete
---

# Phase 17 Plan 01: On-Device Best-Split Finder — Wave 0 Scaffolding & Golden Anchor Summary

**The Phase-17 Wave-0 pure-host scaffolding — `SplitFindTask` + the `assume_out_default_left` task-gen table + the round-ties-even count-recovery helper + the 6-category `best_split.txt` golden matrix and its `best_split_parity.rs` harness — locking the two riskiest parity landmines (D-01 count rounding, Pitfall-3 `default_left != reverse`) as tested units and giving every downstream kernel wave an automated verify BEFORE the kernel is written.**

## Performance

- **Duration:** ~11 min
- **Completed:** 2026-07-01
- **Tasks:** 3/3

## What Was Built

### Task 1 — `best_split.rs` skeleton (commit 9a5da6e)
- `struct SplitFindTask` — field-for-field mirror of `cuda_best_split_finder.hpp:28-41` (12 fields, faithful C++ widths: `i32` `inner_feature_index`/`rand_threshold`, `u32` `hist_offset`/`num_bin`/`default_bin`, `u8` `mfb_offset`, six `bool` flags).
- `round_ties_even(x) -> i32` (the `f64::round_ties_even` intrinsic) + `round_ties_even_branchfree` (the `x>=0` branch-free even-round identity) — faithful to CUDA `__double2int_rn`, DIVERGING from `split.rs::round_int`'s round-half-up (the D-01 landmine).
- `Stage1Scalars` leaf-total/config record + the three stage launcher STUB signatures (`find_best_splits_stage1_on`, `sync_best_split_for_leaf_on`, `find_best_from_all_splits_on`), generic over `R: cubecl::Runtime`, returning `is_valid=false` / `best_leaf=-1` sentinels.
- Module doc locks Open Q1 (RNG seed `extra_seed + task_index`, cu:2220-2228) and the `LGBM_CUDA_ON_DEVICE` OFF-by-default seam; `pub mod best_split;` registered **ungated** in `mod.rs`.
- Unit test `count_recovery_ties_even`: 0.5→0, 1.5→2, 2.5→2 (even), 3.5→4, 2.4→2, 2.6→3, and the round-half-up-would-give-3 contrast + intrinsic/branch-free equivalence.

### Task 2 — `build_split_find_tasks` task-gen table (commit 882047a)
- `FeatureMeta` host input + `build_split_find_tasks` reproducing the C++ table (`cuda_best_split_finder.cpp:137-227`) EXACTLY: `num_bin>2` Zero/NaN emit a forward(assume=false)+reverse(assume=true) PAIR in C++ order; `num_bin<=2`/`None` a single reverse with `assume=(missing!=NaN)`; categorical a single forward one-hot seam (`is_one_hot = num_bin <= max_cat_to_onehot`), no eval math (D-04 Phase-22 dispatch seam only).
- Unit test `assume_out_default_left_table`: all four rows + `reverse=true && assume=true` (Zero num_bin>2 reverse) and the load-bearing `reverse=true && assume=false` divergence (num_bin≤2 NaN — `default_left != reverse`, Pitfall 3).

### Task 3 — harness + fixture (commit 2225e79)
- `best_split_parity.rs` mirrors `kernel_parity.rs`: `kernels_dir()` + skip-if-absent read idiom, bit-hex parse helpers, SCASE/SHIST/SWIN parse, drives `find_best_splits_stage1_on` on `cpu_client()` and compares `SplitScalars` fields via `compare_exact_f64_bits`; 7 coverage booleans assert every D-07 category present.
- `best_split.txt` — 10 records: default template (fwd+rev, smaller+larger = 4), USE_L1 (`lambda_l1=0.1`), USE_SMOOTHING (`path_smooth=2.0` + parent_output), USE_RAND (`rng_seed`), empty/skip-default-bin (`is_valid=0`), global-memory spill (`num_bin=300`), and a `default_left` tie case. PROVENANCE GUARD comment binds the Wave-2 finalization to C++ hand-transcription (anti-circularity).

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Plan Task-2 acceptance mislabeled the `default_left != reverse` divergence feature**
- **Found during:** Task 2
- **Issue:** The plan's Task-2 acceptance text said "a `MissingType::None` feature yields a single reverse task with `assume_out_default_left=false`". The C++ else-branch (`cuda_best_split_finder.cpp:216-220`) sets `assume = (missing != NaN && !categorical)`, so `None → assume=true` (= reverse here, NOT a divergence). The genuine `reverse=true && assume=false` case is a **num_bin≤2 NaN** feature (`NaN != NaN` → false).
- **Fix:** Implemented `build_split_find_tasks` faithful to the C++ source (None → assume=true) — CLAUDE.md 100% behavioral compatibility takes precedence over the plan text. The `assume_out_default_left_table` test asserts None→assume=true and moves the load-bearing `reverse && !assume` divergence assertion onto the num_bin≤2 NaN feature.
- **Files modified:** `crates/lgbm-compute/src/kernels/best_split.rs`
- **Commit:** 882047a

**2. [Rule 3 - Blocking] `cat_threshold` acceptance grep matched the `num_cat_threshold` substring**
- **Found during:** Task 2
- **Issue:** Task-2 acceptance grep forbids `cat_threshold` in `best_split.rs` (no categorical eval math), but the stage-3 export doc referenced field `[7] best_leaf.num_cat_threshold` — the forbidden token is a substring of the legitimate field name.
- **Fix:** Reworded the stage-3 export doc to "the best leaf's categorical-threshold count" — no semantics lost, the literal `cat_threshold` substring removed; the field remains the same 8-int export cell.
- **Files modified:** `crates/lgbm-compute/src/kernels/best_split.rs`
- **Commit:** 882047a

## Verification

- `cargo test -p lgbm-compute --lib best_split` — 2 unit tests green (ties-even, task-gen table). Full lib: 66 passed, 1 ignored.
- `cargo test -p oracle-harness --test best_split_parity -- --list` — `best_split_parity_stage1_bit_exact_on_cpu` discoverable.
- Default run: the stage-1 parity test is `ignored` (merge gate green); `--ignored` run RED-fails via ASSERTION `is_valid mismatch (got false, want true)` (confirmed an assertion mismatch, NOT a compile/panic — the intended Nyquist scaffold).
- `cargo build --workspace` succeeds; `on_device_growth_supported_stays_false` green; `cargo test -p oracle-harness --test kernel_parity` 10/10 unregressed.
- `grep -c 'pub mod best_split' mod.rs == 1`, ungated; all 12 SplitFindTask fields present; `extra_seed + task_index` in the doc; no `BitonicArgSort`/`cat_threshold` eval math.

## Known Stubs

The three stage launchers (`find_best_splits_stage1_on`, `sync_best_split_for_leaf_on`, `find_best_from_all_splits_on`) are intentional Wave-0 stubs returning sentinels (`is_valid=false` / `best_leaf=-1`). They are the documented scaffold seam the numerical core fills in Waves 2-4 (17-03/17-04/17-05); the `best_split_parity.rs` stage-1 test is `#[ignore]`d until 17-03 lands the core. The `best_split.txt` SWIN numeric goldens are structural placeholders finalized in 17-03 by C++ hand-transcription (PROVENANCE GUARD in the fixture header). These are the plan-designed Wave-0 outputs, not accidental stubs.

## Self-Check: PASSED
