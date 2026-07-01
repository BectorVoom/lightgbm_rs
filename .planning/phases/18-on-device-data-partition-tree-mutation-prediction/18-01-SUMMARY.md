---
phase: 18-on-device-data-partition-tree-mutation-prediction
plan: 01
subsystem: testing
tags: [cubecl, prefix-sum, golden-fixtures, data-partition, tree-walk, categorical, nyquist-scaffold]

# Dependency graph
requires:
  - phase: 14-foundation-shared-device-primitives-device-structs-rng
    provides: primitives.rs block/global prefix-sum (N:Numeric bodies), validate_scan_inputs, cpu f64 anchor
  - phase: 17-on-device-best-split-finder
    provides: best_split_parity.rs golden-replay harness idioms, split_info.rs SplitScalars, best_split.txt export
provides:
  - u16/u32 integer block prefix-sum launchers (PrepareOffset inclusive + AggregateBlockOffset exclusive)
  - Empty compiling kernel stubs data_partition.rs / tree.rs / predict.rs (Waves 1-2 fill one file each, no mod.rs contention)
  - Extended partition.txt golden (D-02 flag fan-out PCASE, D-03 categorical PCAT, D-08 16-int PPACKET)
  - New predict.txt tree-walk golden (numeric + cat_onehot + cat_manyvsmany, raw margins)
  - Nyquist #[ignore] scaffolds partition_parity.rs / tree_mutation_parity.rs + predict_parity on_device/cat cells
affects: [18-02, 18-03, 18-04]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Integer-typed scan launcher = instantiation of the existing N:Numeric body (SP-1), reusing validate_scan_inputs"
    - "Empty module stubs declared in mod.rs at Wave-0 so parallel Wave-1/2 plans fill disjoint files"
    - "Verbatim C++ route/tree-walk transcription in the header-only kernel_capture (no lib_lightgbm link)"
    - "#[ignore] Wave-0 Nyquist scaffold naming eventual device entry points in UN-IGNORE comments"

key-files:
  created:
    - crates/lgbm-compute/src/kernels/data_partition.rs
    - crates/lgbm-compute/src/kernels/tree.rs
    - crates/lgbm-compute/src/kernels/predict.rs
    - crates/oracle-harness/tests/partition_parity.rs
    - crates/oracle-harness/tests/tree_mutation_parity.rs
    - crates/oracle-harness/tests/fixtures/kernels/predict.txt
  modified:
    - crates/lgbm-compute/src/kernels/primitives.rs
    - crates/lgbm-compute/src/kernels/mod.rs
    - xtask/cpp/kernel_capture.cpp
    - xtask/src/main.rs
    - crates/oracle-harness/tests/kernel_parity.rs
    - crates/oracle-harness/tests/predict_parity.rs
    - crates/oracle-harness/tests/fixtures/kernels/partition.txt

key-decisions:
  - "Open Q1/A1 RESOLVED: u16 lowers cleanly on cubecl-cpu — no u32-widen fallback needed (documented as parity-neutral if a future hip toolchain regresses)"
  - "A4/Open-Q2: 16-int PPACKET is HOST-reconstructed from the C++ tree/partition, not an instrumented device build"
  - "Predict golden uses synthetic trees/feature-meta/rows walked by the VERBATIM reference decision (cuda_tree.cu:317-396) — self-contained header-only discipline"

patterns-established:
  - "SP-1 integer scan: thin per-type wrapper over the shared N:Numeric block-scan body"
  - "Wave-0 module stubs + mod.rs declaration to keep parallel plans on disjoint files"

requirements-completed: [ODL-13, ODL-14, ODL-15]

# Metrics
duration: 75min
completed: 2026-07-01
status: complete
---

# Phase 18 Plan 01: Wave-0 Foundation (Integer Scans, Goldens, Scaffolds) Summary

**u16/u32 integer block prefix-sum launchers (bit-exact vs a serial scan on cubecl-cpu), the full-fan-out + categorical + 16-int-packet partition golden and a numeric+categorical tree-walk predict golden, and three #[ignore] Nyquist scaffolds for ODL-13/14/15 — the merge gate stays green with LGBM_CUDA_ON_DEVICE unset.**

## Performance

- **Duration:** ~75 min
- **Started:** 2026-07-01T10:31Z (approx)
- **Completed:** 2026-07-01T11:46Z
- **Tasks:** 3
- **Files modified/created:** 13 (6 created, 7 modified)

## Accomplishments
- Resolved the single MEDIUM-risk unknown (Open Q1/A1): `prefix_sum_inclusive_u16_on` + `prefix_sum_exclusive_u32_on` instantiate the existing `N: Numeric` scan bodies and are proven bit-exact (integer equality) vs a serial Rust scan on cubecl-cpu, including the 1024-block single-tile boundary and empty/zero-block-size rejection — unblocking the 18-02 scatter.
- Declared `data_partition` / `tree` / `predict` in `kernels/mod.rs` as empty compiling stubs so the three Wave-1/2 plans each fill exactly one owned file with no `mod.rs` contention.
- Extended `kernel_capture.cpp` with a verbatim `SplitInner` full flag fan-out (`SplitRouteFanout`), categorical membership routing (`SplitCategoricalRoute` + `FindInBitset`), the host-reconstructed 16-int `SplitTreeStructure` packet, and a verbatim `AddPredictionToScoreKernel` tree-walk; regenerated `partition.txt` (12 PCASE + 3 PCAT + 2 PPACKET) and new `predict.txt` byte-idempotently.
- Landed the ODL-13/14/15 `#[ignore]` Nyquist scaffolds (`partition_parity.rs` order/cat/packet, `tree_mutation_parity.rs` split-write/ordering, `predict_parity.rs` on_device/cat) — each parses + structurally validates its golden, SKIPs gracefully when absent, and names the eventual device entry point in a one-line UN-IGNORE comment.

## Task Commits

Each task was committed atomically:

1. **Task 1: u16/u32 integer scan launchers + kernel stubs** - `a915fa7` (feat)
2. **Task 2: extend kernel_capture + regenerate partition/predict goldens** - `c04d078` (feat)
3. **Task 3: Nyquist scaffold tests (ODL-13/14/15)** - `6abc54e` (test)

**Plan metadata:** (this commit) (docs: complete plan)

## Files Created/Modified
- `crates/lgbm-compute/src/kernels/primitives.rs` - Added u16 inclusive + u32 exclusive block prefix-sum launch wrappers + host drivers + `#[cfg(test)] mod int_scan`
- `crates/lgbm-compute/src/kernels/mod.rs` - Declared `pub mod data_partition/predict/tree`
- `crates/lgbm-compute/src/kernels/{data_partition,tree,predict}.rs` - Empty compiling Wave-0 stubs (module docs only)
- `xtask/cpp/kernel_capture.cpp` - `SplitRouteFanout`, `FindInBitsetHost`, `SplitCategoricalRoute`, PCAT/PPACKET emitters, predict tree-walk emitter, wired predict argv
- `xtask/src/main.rs` - Pass `predict.txt` output path to the capture driver
- `crates/oracle-harness/tests/fixtures/kernels/partition.txt` - Regenerated with flag fan-out + PCAT + PPACKET
- `crates/oracle-harness/tests/fixtures/kernels/predict.txt` - New numeric + categorical tree-walk golden
- `crates/oracle-harness/tests/kernel_parity.rs` - Guard the None-only partition test to skip `missing_type != 0` fan-out cases
- `crates/oracle-harness/tests/{partition_parity,tree_mutation_parity}.rs` - New Wave-0 scaffolds
- `crates/oracle-harness/tests/predict_parity.rs` - Added `on_device` + `cat` scaffold cells reading `predict.txt`

## Decisions Made
- **u16 vs u32-widen (Open Q1/A1):** u16 lowers cleanly on cubecl-cpu, so `prefix_sum_inclusive_u16_on` is the PrepareOffset path directly; the u32-widen fallback is documented in-code as parity-neutral (byte-identical output; every per-block partial ≤ block_size ≤ 1024 fits both widths) should a future hip toolchain ever fail to lower u16.
- **16-int packet host-reconstructed (A4/Open-Q2):** the PPACKET fields are computed from the C++ tree/partition on CPU exactly as `SplitTreeStructureKernel` packs them (including the smaller/larger branch by `num_data`), not captured from an instrumented device build.
- **Predict golden via synthetic trees:** the tree-walk decision is the VERBATIM reference (`cuda_tree.cu:317-396`); the trees/feature-meta/rows are synthetic test vectors, consistent with the existing header-only capture discipline (no `lib_lightgbm` link, external_libs unbuildable in-tree).

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Guarded the existing None-only partition parity test against the new fan-out goldens**
- **Found during:** Task 2 (regenerate partition.txt)
- **Issue:** `kernel_parity.rs::kernel_parity_partition_exact_on_cpu` drives the `MissingType::None` host-gather `data_partition` over EVERY `PCASE` and compares to the golden `PORDER`. The 8 new D-02 fan-out cases (`missing_type != 0`) have goldens computed with the missing-type-aware `SplitInner`, so the None-only replay would diverge and fail — blocking a green merge gate.
- **Fix:** Added a `missing_type` field read (defaulting to 0 for Phase-4 goldens) that skips the fan-out cases (consuming their 3 payload lines to keep the line stream aligned). The fan-out cases are now covered by the new `partition_parity.rs` scaffold, un-ignored when the device path lands in 18-02.
- **Files modified:** crates/oracle-harness/tests/kernel_parity.rs
- **Verification:** `cargo test -p oracle-harness --test kernel_parity kernel_parity_partition` passes; full workspace gate green.
- **Committed in:** c04d078 (Task 2 commit)

---

**Total deviations:** 1 auto-fixed (1 blocking)
**Impact on plan:** Necessary to keep the ODL-19 merge gate green after extending the shared golden. No scope creep — the fan-out cases remain fully covered by the new scaffold.

## Issues Encountered
- The plan's Task-3 verify command `cargo test -p oracle-harness partition_parity tree_mutation_parity predict_parity` is malformed — `cargo test` accepts only ONE test-name filter positional argument, so three at once errors (`unexpected argument`). Verified the equivalent intent via `cargo test -p oracle-harness --test partition_parity --test tree_mutation_parity --test predict_parity` (all scaffolds `ignored`) and the full `cargo test --workspace` merge gate (all green, only the 7 intended Wave-0 scaffolds ignored). Documented for future plans.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- **18-02 (Wave 1, ODL-13):** integer scan launchers, partition.txt fan-out/PCAT/PPACKET goldens, and `partition_parity.rs` (order/cat/packet) are ready — the scatter can transcribe against a concrete failing target; `data_partition.rs` stub owned.
- **18-03 (Wave 1, ODL-14):** `tree_mutation_parity.rs` (split-write + ordering) + `tree.rs` stub ready.
- **18-04 (Wave 2, ODL-15):** `predict.txt` + `predict_parity.rs` on_device/cat cells + `predict.rs` stub ready.
- No blockers. u16-lowering unknown resolved; the three plans land on disjoint files.

---
*Phase: 18-on-device-data-partition-tree-mutation-prediction*
*Completed: 2026-07-01*

## Self-Check: PASSED

All 6 created files exist on disk (3 kernel stubs, 2 scaffold tests, predict.txt fixture); all 3 task commits (a915fa7, c04d078, 6abc54e) present in git history.
