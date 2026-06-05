---
phase: 04-compute-backend-cpu-first-integer-histograms-rocm
plan: 03
subsystem: infra
tags: [cubecl, cubecl-cpu, find-best-split, gain-math, data-partition, histogram-subtract, golden-capture, oracle, determinism]

# Dependency graph
requires:
  - phase: 04-compute-backend-cpu-first-integer-histograms-rocm (plan 01)
    provides: ComputeError boundary, cpu/rocm runtime + capability gate, bit-exact single-owner ordered f64 fold, CpuBackend shape
  - phase: 04-compute-backend-cpu-first-integer-histograms-rocm (plan 02)
    provides: Backend::construct_histograms, xtask kernel-capture subcommand + kernel_capture.cpp transcription pattern, kernel_parity.rs replay + f32/f64-bits golden serialization, dense-ordered-fold golden convention
  - phase: 02-dataset-binning-determinism-root
    provides: binned columnar store (FeatureGroup/Bin offset/default_bin/most_freq_bin) as the kernel inputs
  - phase: 01-oracle-contract-foundations
    provides: oracle-harness comparators (compare_exact_f64_bits, compare_exact_u32), xtask C++ golden-capture harness, REFERENCE_MANIFEST discipline
provides:
  - Backend::find_best_split (D-01 whole-kernel op, gain math IN-kernel per D-01a) + CpuBackend impl
  - gain.rs — ThresholdL1/get_leaf_gain/get_split_gains/calculate_splitted_leaf_output (#[cube] fns) + GainConfig + SplitInfo
  - Backend::data_partition (stable reordered index array + split_point, SplitInner MissingType::None routing) + CpuBackend impl
  - Backend::subtract_histograms (FeatureHistogram::Subtract math, A3 resolved in-scope) + CpuBackend impl
  - committed split.txt / partition.txt / subtract.txt goldens (kernel-capture, byte-idempotent)
  - kernel_parity.rs split/partition/subtract bit-exact replay layers (ORA-04 cpu hard gate)
affects: [04-04, phase-05-tree-learner]

# Tech tracking
tech-stack:
  added: []  # no new crates
  patterns:
    - "D-01a: the split gain formula lives INSIDE the find_best_split kernel as #[cube] fns (gain.rs), transcribed verbatim from feature_histogram.hpp:711-1057; Phase-5 consumes, never re-derives"
    - "cubecl-cpu (0.10.0) lowering recipe: loop-carried mutables MUST be literal-initialized (never from a scalar kernel arg) and conditional in-loop stores MUST use branchless select() — the C++ gate ORDER + arithmetic are preserved 1:1, only the control-flow ENCODING changes"
    - "split golden carries PER-CANDIDATE gains (REVERSE + FORWARD, NaN where gated) AND the winner, so a divergence localizes to the gain scan, not just the winner"
    - "data_partition split: a #[cube] per-row routing map (the load-bearing SplitInner decision) + a host stable two-pass gather (left rows then right rows, original order); returns (reordered index array, split_point)"

key-files:
  created:
    - crates/lgbm-compute/src/gain.rs
    - crates/lgbm-compute/src/kernels/split.rs
    - crates/lgbm-compute/src/kernels/partition.rs
    - crates/lgbm-compute/src/kernels/subtract.rs
    - crates/oracle-harness/tests/fixtures/kernels/split.txt
    - crates/oracle-harness/tests/fixtures/kernels/partition.txt
    - crates/oracle-harness/tests/fixtures/kernels/subtract.txt
  modified:
    - crates/lgbm-compute/src/lib.rs
    - crates/lgbm-compute/src/kernels/mod.rs
    - xtask/cpp/kernel_capture.cpp
    - xtask/src/main.rs
    - crates/oracle-harness/tests/kernel_parity.rs
    - crates/oracle-harness/fixtures/REFERENCE_MANIFEST.md

key-decisions:
  - "Gain sentinel inside the kernel is the LITERAL 0.0 (not C++ kMinScore = -inf): valid candidate gains are strictly > min_gain_shift >= 0, so 0.0 rejects nothing, and 'no split found' is signaled by is_splittable==0 (NOT by best_gain). This is required because cubecl-cpu rejects loop-carried mutables initialized from a scalar kernel arg (-inf would have to be an arg)."
  - "REVERSE/FORWARD scan loops are bounded RANGE loops (host-computed iteration counts passed as scalars), and C++ `break` is encoded as a sticky monotone `done` flag — equivalent to break because the gate quantities are monotone in t, so all later iterations also fail. The winner is therefore identical to the C++ break-terminated scan."
  - "data_partition returns ONLY the partition (reordered index array + split_point), not leaf_begin_/leaf_count_ — that bookkeeping is Phase-5 orchestration (resolved Open-Q3 shape)."
  - "subtract_histograms is in-scope at the kernel layer (RESEARCH A3): it is histogram-layer math; WHICH child is subtracted (the smaller sibling) is Phase-5 orchestration."
  - "Only the default CPU template instantiation is transcribed (USE_RAND/USE_MC/USE_MAX_OUTPUT/USE_SMOOTHING all false; MissingType::None for partition). max_delta_step / path_smooth / missing-NA routing are rejected (typed ComputeError) as Phase-7+ scope rather than silently mis-computed."

patterns-established:
  - "Pattern: complex sequential cube kernels on cubecl-cpu use literal-init loop-carried vars + branchless select for all conditional stores; verify each new control-flow construct compiles with a tiny isolation kernel before building the full scan."
  - "Pattern: layered split golden (per-candidate gains + winner) so a parity failure points at the gain-scan math vs the winner selection."
  - "Pattern: a kernel whose output is a dynamic reorder (partition) emits a per-row flag from the #[cube] kernel and does the stable gather on the host — keeps the kernel a flat per-row map."

requirements-completed: [CMP-04, CMP-05, ORA-04]

# Metrics
duration: 35min
completed: 2026-06-06
---

# Phase 4 Plan 03: find_best_split + data_partition + subtract_histograms Summary

**The remaining compute kernels landed bit-exact on the cubecl-cpu anchor: `find_best_split` with the gain formula transcribed VERBATIM inside the `#[cube]` kernel (D-01a, both REVERSE `t-1+offset` and FORWARD `t+offset` branches, exact kEpsilon/2*kEpsilon placement and gate order), `data_partition` (stable SplitInner row routing → reordered index array + split_point), and `subtract_histograms` (FeatureHistogram::Subtract, A3 resolved in-scope) — each wired Backend method → cubecl-cpu kernel → committed C++ golden → bit-exact `compare_exact_*` parity, with the split golden carrying per-candidate gains over both branches.**

## Performance

- **Duration:** ~35 min
- **Started:** 2026-06-05T19:02:05Z
- **Tasks:** 3
- **Files modified:** 13 (7 created, 6 modified)

## Accomplishments
- **`Backend::find_best_split` finalized with in-kernel gain math (D-01a, CMP-05 split layer):** `gain.rs` holds `ThresholdL1`/`get_leaf_gain`/`get_split_gains`/`calculate_splitted_leaf_output` as `#[cube]` fns (verbatim `feature_histogram.hpp:711-845`) plus `GainConfig` (the seven Config fields) and `SplitInfo`. `kernels/split.rs` transcribes `FindBestThresholdSequentially` (`:830-1057`) for BOTH the REVERSE branch (records `t-1+offset`, seeds `sum_right_hessian=kEpsilon`) and the FORWARD branch (records `t+offset`, seeds `sum_left_hessian=kEpsilon`), with the `2*kEpsilon` entry bump, the SKIP_DEFAULT_BIN continue, the exact min_data/min_hessian/min_gain gate order, and the kEpsilon subtracted back at finalization.
- **`Backend::data_partition` + `Backend::subtract_histograms` (A3 resolved):** partition mirrors `DataPartition::Split`/`SplitInner` (`MissingType::None`) and returns a stable reordered index array + split_point; subtract reproduces `FeatureHistogram::Subtract` (`derived[i] = parent[i] - child[i]`). Both with V5 boundary validation (typed `ComputeError`, never panic).
- **Layered C++ goldens (split/partition/subtract) + parity (ORA-04 cpu hard gate):** `kernel_capture.cpp` extended with verbatim transcriptions emitting `split.txt` (per-candidate gains REVERSE+FORWARD + winner; reverse-winner, forward-winner, default-bin-skip, L1, and no-split cases), `partition.txt` (reordered array + split_point), `subtract.txt`. `kernel_parity.rs` replays all three bit-exact on cubecl-cpu (split per-candidate gains via the public `get_split_gains` + the winner via the real kernel; partition via `compare_exact_u32`; subtract via `compare_exact_f64_bits`).
- **CMP-04 gate intact:** the capability gate still selects `ReducePath::Sequential` on cpu; all three kernels use the single-owner (`CubeDim::new_1d(1)`) launch (no Plane dependency).
- **`cargo test --workspace` green; kernel-capture byte-idempotent** across all four goldens.

## Task Commits

Each task was committed atomically:

1. **Task 1: gain.rs + find_best_split scan kernel (REVERSE + FORWARD)** - `ff24113` (feat)
2. **Task 2: data_partition + subtract_histograms kernels + Backend methods** - `d5061b9` (feat)
3. **Task 3: kernel-capture + kernel_parity for split/partition/subtract** - `7f8a5fc` (feat)

## Files Created/Modified
- `crates/lgbm-compute/src/gain.rs` - verbatim gain primitives (#[cube]) + GainConfig + SplitInfo
- `crates/lgbm-compute/src/kernels/split.rs` - find_best_split_kernel (both scan branches) + host launcher (BeforeNumerical min_gain_shift, 2*kEpsilon bump, finalization)
- `crates/lgbm-compute/src/kernels/partition.rs` - data_partition_kernel (SplitInner routing map) + host stable gather
- `crates/lgbm-compute/src/kernels/subtract.rs` - subtract_hist_kernel (element-wise parent-child)
- `crates/lgbm-compute/src/lib.rs` - three new Backend trait methods + CpuBackend impls; re-export GainConfig/SplitInfo
- `crates/lgbm-compute/src/kernels/mod.rs` - module decls for split/partition/subtract
- `xtask/cpp/kernel_capture.cpp` - verbatim FindBestThreshold/gain/SplitInner/Subtract transcription + the three emitters
- `xtask/src/main.rs` - kernel-capture passes/asserts the three new golden paths; manifest documents the layers
- `crates/oracle-harness/tests/kernel_parity.rs` - split/partition/subtract bit-exact replay layers
- `crates/oracle-harness/tests/fixtures/kernels/{split,partition,subtract}.txt` - committed goldens
- `crates/oracle-harness/fixtures/REFERENCE_MANIFEST.md` - 04-03 golden-layer documentation

## Decisions Made
- **Gain sentinel = literal `0.0` inside the kernel** (not C++ `kMinScore = -inf`): valid gains are strictly `> min_gain_shift >= 0`, so 0.0 rejects nothing; "no split" is signaled by `is_splittable == 0`. Forced by the cubecl-cpu literal-init constraint (see Deviations / Issues).
- **`break` → sticky monotone `done` flag:** the C++ `break` gates (left too small / left-hessian too small in REVERSE; right too small / right-hessian too small in FORWARD) are monotone in `t`, so gating those-and-all-later iterations off yields the IDENTICAL winner — encoded as a sticky `done` bool because cube has no `break` here.
- **`data_partition` returns only `(reordered, split_point)`** — Phase-5 owns `leaf_begin_`/`leaf_count_` (resolved Open-Q3 shape).
- **`subtract_histograms` in-scope at the kernel layer (A3)** — it is histogram-layer math; which child is subtracted is Phase-5 orchestration.
- **Phase-4 scope guard:** `max_delta_step`/`path_smooth` (find_best_split) and missing/NA routing (partition) are rejected with a typed `ComputeError` rather than silently mis-computed (Phase-7+ scope).

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] `gain.rs::threshold_l1` mis-lowered to a constant on cubecl-cpu, ZEROING every L1 gain**
- **Found during:** Task 3 (split parity — the `l1_forward` case)
- **Issue:** The branchless sign in `ThresholdL1` was written `let pos = if s > 0.0 { 1.0 } else { 0.0 }; let neg = if s < 0.0 { 1.0 } else { 0.0 };`. The `if cond { 1.0 } else { 0.0 }` value-expression mis-lowers on cubecl-cpu (0.10.0): an isolation kernel showed `threshold_l1(-19.0, 0.5)` returning `0` in-kernel vs `-18.5` on the host. This zeroed `ThresholdL1`, hence every L1 leaf/split gain → the kernel reported "no split" for any `lambda_l1 > 0` (silent, would have corrupted all L1 training in Phase 5).
- **Fix:** Replaced the two `if/else` value-expressions with branchless `select(cond, 1.0, 0.0)`. The in-kernel `threshold_l1` / `get_leaf_gain` now match the host bit-for-bit, and the `l1_forward` split parity case passes bit-exact.
- **Files modified:** crates/lgbm-compute/src/gain.rs
- **Verification:** `kernel_parity_split_bit_exact_on_cpu` (incl. the L1 case) green; isolation kernel confirms in-kernel == host for `threshold_l1`/`get_leaf_gain`.
- **Committed in:** `7f8a5fc` (Task 3 commit)

---

**Total deviations:** 1 auto-fixed (1 bug — a correctness-critical FP/codegen defect surfaced by the per-candidate-gain parity layer)
**Impact on plan:** No scope change. The bug is exactly the class the bit-exact L1 golden was designed to catch; without it, all L1-regularized splits would silently never fire.

## Issues Encountered
- **cubecl-cpu (0.10.0) MLIR lowering rejected the natural sequential-scan encoding.** The initial `find_best_split_kernel` failed to compile ("operation with block successors must terminate its parent block") and silently produced an all-zero output. Isolated via a battery of minimal kernels to TWO root causes: (1) a loop-carried mutable initialized directly from a scalar kernel argument and then reassigned in the loop produces invalid MLIR — so all loop-carried `best_*`/`is_splittable` mutables are now LITERAL-initialized (driving the `0.0` gain sentinel decision); (2) conditional in-loop stores via nested `if` mutation chains fail the same pass — so every conditional store uses branchless `select`. The C++ gate ORDER and arithmetic are preserved 1:1; only the control-flow ENCODING differs. Documented in the `split.rs` module header for future cube kernels.
- **`terminate!()` on a `CubeDim(1)` launch left `out` untouched** — the single-owner guard uses the positive `if UNIT_POS == 0 { ... }` form (mirroring the histogram kernel), not `terminate!()`.

## User Setup Required
None - no external service configuration required. (kernel-capture needs a C++ toolchain + cmake to regenerate goldens; `cargo test` reads the committed goldens and needs neither. ROCm bring-up remains deferred to 04-04.)

## Next Phase Readiness
- **The full CMP-05 Backend op set is closed on the cpu anchor:** `construct_histograms` (04-02) + `find_best_split` + `data_partition` + `subtract_histograms` are all bit-exact vs committed C++ goldens on cubecl-cpu — Phase 5 (tree learner) has the complete compute surface to orchestrate, with TRL-04's gain math already implemented (D-01a, consume-don't-rederive).
- **04-04 (ROCm)** can add a `rocm` Backend impl and a `~1e-6` parity variant of `kernel_parity.rs` against the SAME four committed goldens; the gain math's branchless-select form is already cube-portable.
- **No blockers.** CMP-04 (capability gate → Sequential on cpu), CMP-05 (full kernel set), and ORA-04 (cpu hard gate) are satisfied for all four kernels. The cubecl-cpu lowering recipe (literal-init + select) is documented for any further cube kernels.

## Self-Check: PASSED

All 7 created files verified present on disk (gain.rs, kernels/{split,partition,subtract}.rs, fixtures/kernels/{split,partition,subtract}.txt) plus the SUMMARY; all 3 task commits (`ff24113`, `d5061b9`, `7f8a5fc`) verified in git history. `cargo test --workspace` green; all four kernel goldens byte-idempotent.

---
*Phase: 04-compute-backend-cpu-first-integer-histograms-rocm*
*Completed: 2026-06-06*
