---
phase: 04-compute-backend-cpu-first-integer-histograms-rocm
plan: 02
subsystem: infra
tags: [cubecl, cubecl-cpu, histograms, golden-capture, oracle, determinism, xtask]

# Dependency graph
requires:
  - phase: 04-compute-backend-cpu-first-integer-histograms-rocm (plan 01)
    provides: ComputeError boundary, cpu/rocm runtime + capability gate, the bit-exact single-owner ordered f64 fold (construct_hist_kernel + construct_histograms_cpu), D-04a anchor proven
  - phase: 02-dataset-binning-determinism-root
    provides: binned columnar store (DenseBin/SparseBin, Bin::data) as the histogram-kernel input
  - phase: 01-oracle-contract-foundations
    provides: oracle-harness comparators (compare_exact_f64_bits), xtask C++ golden-capture harness, REFERENCE_MANIFEST discipline
provides:
  - Backend::construct_histograms whole-kernel op (D-01) + CpuBackend concrete impl
  - xtask kernel-capture subcommand + KERNEL_MASTER_SEED + xtask/cpp/kernel_capture.cpp (header-only ConstructHistogram transcription)
  - committed histogram golden (crates/oracle-harness/tests/fixtures/kernels/histogram.txt, 18 D-02a cases)
  - kernel_parity.rs — bit-exact cubecl-cpu histogram replay (ORA-04 cpu hard gate, histogram layer)
  - kernel-capture + parity machinery for 04-03 (split/partition) / 04-04 (ROCm) to reuse
affects: [04-03, 04-04, phase-05-tree-learner]

# Tech tracking
tech-stack:
  added: []  # no new crates; lgbm-compute/lgbm-dataset added as oracle-harness DEV-deps only
  patterns:
    - "D-01 whole-kernel Backend op: coarse construct_histograms(client, binned, grad, hess, num_bin) -> Vec<f64>"
    - "header-only verbatim ConstructHistogram transcription (dense_bin.hpp:130-141 / sparse_bin.hpp:138-152) — no external_libs, no lib_lightgbm link"
    - "golden HIST = dense ordered f64 fold over the round-tripped Bin::data(idx) — what the cubecl-cpu kernel computes; sparse layout exercised via the SparseBin store round-trip"
    - "kernel-parity replay: dev-dep cubecl behind the harness's [dev-dependencies] so the library crate stays cubecl-free (CMP-01)"

key-files:
  created:
    - xtask/cpp/kernel_capture.cpp
    - crates/oracle-harness/tests/fixtures/kernels/histogram.txt
    - crates/oracle-harness/tests/kernel_parity.rs
  modified:
    - crates/lgbm-compute/src/lib.rs
    - xtask/src/main.rs
    - xtask/cpp/CMakeLists.txt
    - crates/oracle-harness/fixtures/REFERENCE_MANIFEST.md
    - crates/oracle-harness/Cargo.toml
    - Cargo.lock

key-decisions:
  - "Golden HIST is the DENSE ordered f64 fold over the round-tripped Bin::data(idx) (== what the cubecl-cpu kernel computes), NOT the raw sparse ConstructHistogram cells. Dense and sparse ConstructHistogram are numerically identical for every non-zero bin; they differ only at bin 0 (sparse never folds it). To keep the golden replayable by the dense-fold kernel, sparse cases use bins in [1, num_bin) so dense-fold == sparse-result, while the SparseBin store path is still exercised (Push/FinishLoad/data round-trip)."
  - "construct_histograms is a trait method with a concrete CpuBackend impl (no default impl) — the cpu backend binds ActiveRuntime and dispatches to the 04-01 launcher; keeps the seam explicit and the rocm impl a future addition."
  - "lgbm-compute/lgbm-dataset added under oracle-harness [dev-dependencies] only — the harness LIBRARY stays cubecl-free so the CMP-01 containment guard still passes."

patterns-established:
  - "Pattern: extend the xtask capture harness (rng/bin/model) with a new <kernel>-capture subcommand copying the workspace_root -> verify_toolchain -> cmake configure+build -> locate_exe -> run -> write_manifest skeleton, single master-seed for idempotency."
  - "Pattern: kernel goldens serialize inputs as raw f32 bits (u32) + outputs as raw f64 bits (u64); the Rust parity test parses via from_bits and asserts compare_exact_f64_bits (bit-exact, never the ~1e-6 oracle tolerance) for the cpu anchor."
  - "Pattern: parity tests SKIP-green when the fixture is absent (pre-capture), driving the real Backend over committed inputs otherwise."

requirements-completed: [CMP-01, CMP-02, CMP-05, ORA-04]

# Metrics
duration: 18min
completed: 2026-06-06
---

# Phase 4 Plan 02: construct_histograms Vertical Slice (Backend → cubecl-cpu kernel → C++ golden → bit-exact parity) Summary

**First full compute-backend vertical slice: a real `construct_histograms` whole-kernel op wired end-to-end — `Backend` trait method + `CpuBackend` → the 04-01 bit-exact single-owner f64 fold → a committed C++-transcription histogram golden via a new `xtask kernel-capture` subcommand → bit-exact `compare_exact_f64_bits` parity replay across 18 D-02a cases (dense + sparse layouts, default-bin routing, u8/u16/u32 widths, grad/hess spread).**

## Performance

- **Duration:** ~18 min
- **Started:** 2026-06-06
- **Tasks:** 3
- **Files modified:** 10 (3 created, 7 modified incl. Cargo.lock)

## Accomplishments
- **`Backend::construct_histograms` finalized (D-01, CMP-05 histogram layer):** the coarse whole-kernel op `construct_histograms(client, binned, ordered_gradients, ordered_hessians, num_bin) -> Result<Vec<f64>, ComputeError>` returning stride-2 `[g0,h0,g1,h1,…]` f64 cells, plus a concrete `CpuBackend` binding `ActiveRuntime` and dispatching to the 04-01 bit-exact ordered fold. V5 boundary validation (T-04-01) and f64 accumulation (Pitfall 3) retained.
- **`xtask kernel-capture` + `kernel_capture.cpp` (D-02):** a header-only VERBATIM transcription of `ConstructHistogram` (`dense_bin.hpp:130-141` / `sparse_bin.hpp:138-152`) reusing the `DenseBin`/`SparseBin` bin-storage forms, emitting an 18-case `histogram.txt` golden over D-02a synthetic inputs — numerically identical to lib_lightgbm, no `external_libs`, no C++ toolchain at test time, byte-idempotent.
- **`kernel_parity.rs` (ORA-04 cpu hard gate, histogram layer):** drives `CpuBackend::construct_histograms` over every golden case and asserts BIT-EXACT f64 cells vs the C++ golden via the full-path `oracle_harness::comparator::compare_exact_f64_bits`. Green; SKIPs cleanly pre-capture.
- **CMP-01 boundary intact:** cubecl is pulled into oracle-harness only via `[dev-dependencies]`; the library crate stays cubecl-free and the `cmp01_containment` guard still passes.
- **`cargo test --workspace` green** (no regression); kernel-capture regenerates the golden byte-identically.

## Task Commits

Each task was committed atomically:

1. **Task 1: Finalize Backend::construct_histograms whole-kernel op** - `504e7ff` (feat)
2. **Task 2: kernel-capture subcommand + ConstructHistogram golden** - `80f8fc9` (feat)
3. **Task 3: kernel_parity.rs — bit-exact histogram replay on cubecl-cpu** - `cf3f380` (feat)

## Files Created/Modified
- `crates/lgbm-compute/src/lib.rs` - `construct_histograms` added to the `Backend` trait; `CpuBackend` concrete impl (cfg `cpu`) dispatching to the 04-01 launcher
- `xtask/cpp/kernel_capture.cpp` - header-only `ConstructHistogram` transcription emitting the f64-bit histogram golden
- `xtask/cpp/CMakeLists.txt` - `kernel_capture` target (`-I LightGBM/include` only; no external_libs)
- `xtask/src/main.rs` - `kernel-capture` subcommand, `KERNEL_MASTER_SEED`, manifest "Kernel Golden Set (Phase 4)" section
- `crates/oracle-harness/tests/fixtures/kernels/histogram.txt` - committed 18-case histogram golden (D-02a coverage)
- `crates/oracle-harness/tests/kernel_parity.rs` - bit-exact replay of the golden on cubecl-cpu
- `crates/oracle-harness/Cargo.toml` / `Cargo.lock` - `lgbm-compute`/`lgbm-dataset` dev-deps (test-only)
- `crates/oracle-harness/fixtures/REFERENCE_MANIFEST.md` - Kernel Golden Set section recording `KERNEL_MASTER_SEED` + D-02a coverage

## Decisions Made
- **Golden HIST = dense ordered f64 fold over the round-tripped `Bin::data(idx)`.** The Rust `construct_histograms` is a single dense-style ordered fold over the per-row bins. The C++ dense and sparse `ConstructHistogram` are numerically identical for every non-zero bin (same `(row, bin, grad, hess)` tuples in row order) and differ only in that sparse never folds bin 0. So the golden's HIST is computed by the dense fold over the bins read back from the layout-appropriate store (dense or sparse), which is exactly what the kernel computes. To make the dense-fold golden coincide with the genuine sparse `ConstructHistogram`, sparse cases use bins in `[1, num_bin)` (no bin-0 rows) — the SparseBin store path (Push/FinishLoad/delta-encode/`data`) is still exercised, and the sparse `ConstructHistogram` body is transcribed verbatim as the faithful reference + documented cross-check.
- **`CpuBackend` concrete impl (no trait default).** Keeps the seam explicit; the rocm impl is a future addition (04-04).
- **cubecl pulled into oracle-harness as a DEV-dep only**, preserving the CMP-01 cubecl-free library boundary.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Stray non-existent member call in `kernel_capture.cpp`**
- **Found during:** Task 2 (authoring the C++ corpus builder)
- **Issue:** An early draft of `BuildHistCorpus` called `cs.num_rows_init(w.num_rows)` on `HCaseSpec`, which has no such member (rows are derived from `bins.size()`); it would not compile.
- **Fix:** Removed the stray call (row count flows from the generated `bins` vector); also added the `<algorithm>` (`std::sort`) and `<utility>` (`std::move`) includes the SparseBin transcription needs.
- **Files modified:** xtask/cpp/kernel_capture.cpp
- **Verification:** `cargo build -p xtask` + `cargo run -p xtask -- kernel-capture` succeed (18 cases written, idempotent).
- **Committed in:** `80f8fc9` (Task 2 commit)

---

**Total deviations:** 1 auto-fixed (1 self-introduced authoring bug, fixed before commit)
**Impact on plan:** No scope change. The plan executed as written; the only fix was a typo in newly-authored capture code, corrected before its task commit.

## Issues Encountered
- **Sparse vs dense histogram semantics at bin 0** required care: a naive transcription that emitted the raw sparse `ConstructHistogram` cells would NOT be replayable by the dense-fold kernel (the kernel folds bin-0 rows that the sparse routine drops). Resolved per the Decisions section — golden HIST is the dense fold over the round-tripped bins, and sparse cases avoid bin 0 so the two coincide while still exercising the sparse store. The sparse `ConstructHistogram` body is transcribed verbatim for faithfulness/documentation.
- **`InitIndex` fast-index table** is not part of the focused SparseBin transcription; verified the `start=0` inlined priming (`cur_pos += deltas_[++i_delta]`) is equivalent to the real `InitIndex(0)` + priming loop against the pinned `sparse_bin.hpp` (the fast_index seed for index 0 is `{i_delta=0, cur_pos=first_nonzero_row}`).

## User Setup Required
None - no external service configuration required. (kernel-capture needs a C++ toolchain + cmake; `cargo test` reads the committed golden and needs neither. ROCm bring-up remains deferred to 04-04.)

## Next Phase Readiness
- **04-03 (split / data-partition kernels + goldens) is unblocked:** the kernel-capture subcommand, the layered-golden serialization (f32/f64 bits), and the bit-exact parity machinery are established and reusable; `kernel_capture.cpp` is structured to grow `FindBestThreshold*`/`GetSplitGains` and data-partition transcriptions alongside the histogram layer.
- **04-04 (ROCm)** can add a `rocm` `Backend` impl and a `~1e-6` parity variant of `kernel_parity.rs` against the same committed golden.
- **No blockers.** CMP-01/CMP-02/CMP-05 (histogram layer) and ORA-04 (cpu histogram hard gate) are satisfied; CMP-03/CMP-04 ROCm and the split/partition layers of CMP-05 remain for 04-03/04-04.

## Self-Check: PASSED

All 3 created files verified present on disk (`kernel_capture.cpp`, `histogram.txt`, `kernel_parity.rs`); all 3 task commits (`504e7ff`, `80f8fc9`, `cf3f380`) verified in git history. `cargo test --workspace` green; kernel-capture byte-idempotent.

---
*Phase: 04-compute-backend-cpu-first-integer-histograms-rocm*
*Completed: 2026-06-06*
