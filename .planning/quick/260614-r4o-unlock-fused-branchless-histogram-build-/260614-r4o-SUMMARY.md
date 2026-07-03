---
status: complete
phase: quick-260614-r4o
plan: 01
subsystem: treelearner
tags: [performance, cpu, histogram, build, bit-exact, branchless, fused, treelearner, compute]

requires:
  - phase: spike-003-columnar-hist-build
    provides: "once-per-leaf ord_g/ord_h gather (shipped) + the validated 003b fused-branchless prototype + measured numbers"
provides:
  - "Fused branchless CPU build_leaf_histograms_raw: reads bins[row] inline, folds into a reused per-feature hot scratch, stream-copies into out — no ord_bins, no per-feature alloc, no per-element bin check"
  - "Relocated V5/T-04-01 mitigation: the per-element bin-range check now lives ONLY in the once-per-train upstream gate (learner.rs:700-714); a test proves the typed rejection survives there"
affects: [perf-gap-vs-cpp-40-80x, R3-histogram-build, future-columnar-bin-storage]

tech-stack:
  added: []
  patterns:
    - "Branchless hot fold + caller-guaranteed precondition (validate once upstream, trust in the kernel) mirroring C++ dense_bin.hpp"
    - "Reused per-feature hot scratch (<= 2*max_num_bin) + streaming copy into the big out buffer (NOT folding into out — p0n cache-scatter avoided)"

key-files:
  created: []
  modified:
    - crates/lgbm-compute/src/lib.rs
    - crates/lgbm-treelearner/src/learner.rs

key-decisions:
  - "Branchless fold (no per-element bin check) — 003b proved any per-element check (early-return OR clamp+OOB-flag) regresses large ~3-8%; the relocated upstream gate + debug_assert! keep V5 without serializing the loop"
  - "Fold into a reused per-feature hot scratch then stream-copy, NOT into the big multi-feature out buffer (p0n -9% large cache-scatter lesson)"
  - "V5/T-04-01 mitigation MOVED (not removed) to the once-per-train upstream gate; a typed-rejection test pins the guarantee at its new location"

patterns-established:
  - "Caller-guaranteed precondition contract: the kernel doc + the learner comment cross-reference via BinIndexOutOfRange so the two sites stay coupled"

requirements-completed: [PERF-R3-FUSED-HIST]

duration: ~30min
completed: 2026-06-14
---

# Phase quick-260614-r4o: Unlock Fused Branchless Histogram Build Summary

**Fused, branchless CPU `build_leaf_histograms_raw` — reads the bin column inline into a reused per-feature hot scratch with no per-element bin check — landing −17% small / −6.6% large train_median at zero parity cost, with the V5/T-04-01 bin-range mitigation relocated to a once-per-train upstream gate.**

## Performance

- **Duration:** ~30 min
- **Tasks:** 3/3
- **Files modified:** 2

## Accomplishments

- **Fused branchless build (Task 1):** the CPU default `Backend::build_leaf_histograms_raw` now folds each feature DIRECTLY from its bin column (`bins[row]` read inline) into a reused per-feature hot scratch sized `<= 2*max_num_bin`, then stream-copies into `out`. No `ord_bins` materialization, no per-feature alloc, no fold into the big 32KB multi-feature `out` buffer (p0n cache-scatter avoided), and — critically — NO per-element `bin < num_bin` check (only a `debug_assert!`). `ord_g`/`ord_h` are still gathered ONCE per leaf (spike-003 preserved).
- **Bit-exact preserved:** the f64 fold order is byte-identical to `construct_histograms_cpu_native` (ascending `leaf_rows`, grad at `bin<<1`, hess at `+1`, f32-read → f64-accumulate). The only change is reading `bins[row]` inline instead of materializing `ord_bins`.
- **V5/T-04-01 relocated (Task 2):** the per-element bin-range check is now the SINGLE once-per-train upstream gate in `train_inner` (learner.rs:700-714), documented as the relocated mitigation and cross-referencing the lib.rs precondition. A new test `train_rejects_out_of_range_bin_with_typed_error` proves the typed `BinIndexOutOfRange` rejection survives at its new location.
- **HARD gates GREEN (Task 3):** all four bit-exact suites pass; interleaved A/B (2 rounds) at small + large confirms the win with NO large regression.

## Task Commits

1. **Task 1: Fused branchless build_leaf_histograms_raw (CPU default impl) + relocated-validation contract** — `f48adac` (perf)
2. **Task 2: Relocate-confirm the once-per-train upstream validation + typed-rejection test** — `1a09a04` (test)
3. **Task 3: HARD bit-exact gate + interleaved A/B (measurement-only, no code change)** — covered by the gate runs below; no separate code commit (Task 3 verifies Tasks 1+2)

## Files Created/Modified

- `crates/lgbm-compute/src/lib.rs` — fused branchless `build_leaf_histograms_raw` default impl + the "Bin-range precondition (V5/T-04-01 RELOCATED)" doc + `# Errors` rewrite; `_client` is now unused on this default path (GPU override still uses it). RocmBackend untouched.
- `crates/lgbm-treelearner/src/learner.rs` — strengthened the `train_inner` bin-range loop comment to mark it the SINGLE relocated T-04-01 gate (cross-references lib.rs); added `train_rejects_out_of_range_bin_with_typed_error`.

## Bit-Exact Merge Gate — PASS

| Suite | Result |
|-------|--------|
| `cargo test -p lgbm-compute --lib` | 21 passed / 0 failed |
| `cargo test -p lgbm-treelearner` | 66 passed / 0 failed (65 existing + 1 new rejection test) |
| `cargo test -p oracle-harness --test learner_parity` | **29 passed / 0 failed — BIT-EXACT** |
| `cargo test -p oracle-harness` (full) | every suite 0-failed (boosting_parity 75/0, kernel_parity 6/0, learner_parity 29/0, metric_parity 15/0, predict_parity 5/0, rank_parity 4/0, raw_bin_train_parity 2/0, rng_parity 1/0, advanced_parity 5/0, comparator 5/0, config_drift 3/0); pre-existing DEF-07-02 cells stay green-by-ignore — no NEW failures |

clippy clean on all edited code (`build_leaf_histograms_raw` fold, the learner comment, the new test); pre-existing workspace warnings (bin_mapper, split.rs same-type casts, the already-`#[allow]`'d too_many_arguments) are untouched. `LightGBM/` never git-added (remains untracked `?? LightGBM/`).

## A/B Performance — interleaved, 2 rounds, LGBM_PHASE_PROF=1

Baseline A = spike-003 once-gather (temporarily reverted in `lib.rs` for the measurement, then fully restored — working tree matches the committed fused HEAD). Fused B = this plan's branchless HEAD.

### train_median

| Size | Baseline (003) R1 / R2 | Fused (003b) R1 / R2 | Δ (median of medians) | Target | Verdict |
|------|------------------------|----------------------|-----------------------|--------|---------|
| small 2k×12 | 32.77 / 32.36 ms | **27.00 / 26.83 ms** | **−17.0%** | ≈−17% | hit |
| large 200k×32 | 2.92 / 2.95 s | **2.73 / 2.74 s** | **−6.6%** | ≈−4.5%, no regression | **beat target, NO regression** |

### BUILD phase (`hist+split` build sub-phase, µs/iter)

| Size | Baseline build R1 / R2 (total ms) | Fused build R1 / R2 (total ms) | per-iter Δ |
|------|-----------------------------------|--------------------------------|-----------|
| small 2k×12 (900 it) | 137.575 / 135.727 ms (≈152 µs/it) | 91.797 / 91.273 ms (≈102 µs/it) | **−33%** |
| large 200k×32 (250 it) | 10515.932 / 10574.399 ms (≈42060 µs/it) | 9627.577 / 9614.082 ms (≈38483 µs/it) | **−8.5%** |

rows/s: small 61k → 74.5k; large 67.8–68.6k → 73.1–73.2k. Build phase share dropped from 51.7–51.9% → 43.0–43.1% (small) and held 89.3% (large).

**The large result does NOT regress vs the ~2.89s spike-003 baseline** — it improves to ~2.73s (−6.6%), comfortably clearing the p0n/003b hard gate. Result SHIPPED (not reverted).

## Decisions Made

- **Branchless fold, relocated check** — followed the plan and 003b finding exactly: no per-element bin check in the release fold; `debug_assert!(bin < num_bin)` defense-in-depth; the V5 mitigation lives once-per-train upstream.
- **Hot scratch, not in-place** — fold into a reused per-feature scratch then stream-copy, never into the big `out` buffer (p0n −9% lesson).

## Deviations from Plan

None — plan executed exactly as written. (Task 3 is measurement-only with no code change; the bench's temporary baseline revert in `lib.rs` was fully restored, leaving the working tree identical to the committed fused HEAD.)

## Issues Encountered

None. The A/B baseline was produced by temporarily reverting only the `build_leaf_histograms_raw` body to the spike-003 once-gather form, building, measuring both rounds/scales, then restoring the fused body from a backup (verified `git diff` clean against HEAD afterward).

## Next Phase Readiness

- R3 histogram-build headroom further reduced; build is now ≈43% of small train (was ≈52%) and ≈89% of large.
- Remaining levers (next candidates, A/B both scales first): columnar uint8 bin storage (denser bin-column read), subtraction-trick reuse on the CPU build path. The fused branchless fold is now the platform those build on.
- GPU/ROCm path untouched (RocmBackend override at lib.rs:854 is out of scope and unchanged).

## Self-Check: PASSED

- `crates/lgbm-compute/src/lib.rs` — FOUND, fused branchless fold present (commit `f48adac`)
- `crates/lgbm-treelearner/src/learner.rs` — FOUND, relocated-gate comment + `train_rejects_out_of_range_bin_with_typed_error` present (commit `1a09a04`)
- Commit `f48adac` — FOUND in git log
- Commit `1a09a04` — FOUND in git log

---
*Phase: quick-260614-r4o*
*Completed: 2026-06-14*
