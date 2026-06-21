# Phase 4: Compute Backend (CPU-first f32 histograms → ROCm) - Context

**Gathered:** 2026-06-05
**Status:** Ready for planning

<domain>
## Phase Boundary

Phase 4 delivers an **isolated `lgbm-compute` backend** whose **f32 histogram-construction, best-split-finding, and data-partition kernels** run on the **cubecl-cpu** reference path and the **cubecl-hip (ROCm)** path, both meeting the project numerical contract (f32 end-to-end, ~1e-6 absolute). This is the **CubeCL-alpha-churn containment boundary** (CMP-01): all `cubecl` runtime type names live behind one `Backend` trait so the alpha API can evolve without leaking into any crate above it. The `lgbm-compute` crate already exists as a kernel-free trait skeleton (Phase 1 D-09) — this phase fills it in.

The dependency-forced build order puts the **compute kernels** (this phase) below the **tree learner** (Phase 5): Phase 4 produces the device primitives; Phase 5 orchestrates them into leaf-wise growth with constraints and leaf-splits.

In scope:
- **CMP-01** — all CubeCL usage confined behind the single `lgbm-compute` `Backend` trait; no crate above it names a CubeCL runtime; a CPU-only build needs no ROCm toolchain.
- **CMP-02** — CPU backend (cubecl-cpu) as the deterministic reference execution path.
- **CMP-03** — ROCm/HIP backend (cubecl-hip) selectable via Cargo feature and/or runtime config.
- **CMP-04** — CUDA warp-level reductions mapped onto CubeCL's `Plane` API with startup capability-gating (`Plane::Ops`, f64, atomics) and a deterministic sequential fallback when a capability is absent.
- **CMP-05** — GPU-resident **histogram construction**, **best-split finding** (gain formula inside the kernel), and **data-partition** kernels meeting the ~1e-6 (f32) contract.
- **ORA-04** — the oracle suite executes (and, for CPU, passes as a hard gate) on the backends; CPU-runtime and ROCm are **separate gates**.

Out of scope: the **tree learner** orchestration — leaf-wise (best-first) growth, leaf-splits, `num_leaves`/`max_depth` caps, histogram-subtraction *trick orchestration*, monotone/interaction constraints, feature subsampling, `force_row_wise`/`force_col_wise` strategy selection (all Phase 5, TRL-01..09); the **GBDT spine / objectives / metrics** (Phase 6); **DART/RF/GOSS** variants (Phase 7); **prediction** (already shipped Phase 3); **integer-quantized / discretized** histograms (dropped project-wide, Phase 1 D-03; v2 QNT-01); **linear-tree** kernels (v2 LIN-01); the Python surface (Phase 8).

</domain>

<decisions>
## Implementation Decisions

### Compute / Tree-Learner Boundary (CMP-05 / Phase 4↔5 cut)
- **D-01:** **Whole-kernel ops.** `lgbm-compute`'s `Backend` trait exposes **coarse, complete operations** matching the CMP-05 wording — `construct_histograms`, `find_best_split` (the split-gain math lives **inside** the kernel), and `data_partition`. The Phase-5 learner **orchestrates** these (growth, leaf-splits, constraints, subtraction-trick bookkeeping) but does not re-implement the per-bin gain scan. (Chosen over "thin device primitives," which would push all gain math up into the learner, and over the "split the difference" hybrid that kept best-split scalar.)
- **D-01a — boundary note for research:** Because `find_best_split` carries the gain formula, Phase 4 effectively implements the math of Phase-5 **TRL-04** early (`ThresholdL1`, `GetSplitGains`/`GetLeafGain`, `kEpsilon` / `2*kEpsilon` positions, `lambda_l1`/`lambda_l2`/`min_gain_to_split`/`min_sum_hessian_in_leaf`/`min_data_in_leaf`/`max_delta_step`/`path_smooth`, tie-breaking, `SKIP_DEFAULT_BIN`/missing routing). Research must define exactly **which gain parameters flow into the kernel** (the gain-config surface) and confirm the `feature_histogram.hpp` routines that move into the kernel vs stay in the Phase-5 learner. TRL-04 in Phase 5 then *consumes* this kernel rather than re-deriving gains — note this overlap in the roadmap-tracing so it isn't double-counted or re-litigated.

### Kernel Golden / Validation Strategy (ORA-04, no learner yet)
- **D-02:** **Header-only C++ transcription** for kernel goldens. There is no Phase-5 learner to drive the kernels and the full C++ treelearner is unbuildable here (`external_libs` unvendored — Phase 1/2/3 precedent). Extend the `xtask` capture harness with a histogram/split/partition capture subcommand that **verbatim-transcribes** the C++ routines (`src/treelearner/feature_histogram.hpp`, `src/io/dense_bin.hpp`/`sparse_bin.hpp`, the `ConstructHistograms`/`FindBestSplitsFromHistograms`/data-partition logic) header-only, emits goldens over **synthetic bin + grad/hess inputs**, and **commits** them. Human-approved, numerically identical to `lib_lightgbm`, replayable with no C++ toolchain at normal test time. (Chosen over standing up an independent scalar-Rust oracle, and over the "both" belt-and-suspenders option — see D-04 for why one Rust impl suffices.)
- **D-02a:** Synthetic inputs should exercise every kernel path that the contract cares about: dense + sparse bin layouts, the most-frequent-bin / default-bin skip, missing/zero routing, multiple bit widths where they affect accumulation, and grad/hess sign/magnitude spread that stresses the f32 reduction. Reuse the Phase-2 binned-store forms where possible so inputs are already bit-faithful.

### ROCm Gating Posture (CMP-03 / SC#3 / SC#5)
- **D-03:** **CPU-solid now, ROCm best-effort.** Make the **cubecl-cpu** path rock-solid and **fully oracle-gated** this phase (hard gate). **Bring up ROCm and run the oracle on the local ROCm GPU**, but if CubeCL-alpha or hardware capability gaps block ~1e-6 parity, **record them as known issues** (with specifics) rather than blocking phase completion. Rationale: CubeCL v0.10 is alpha and ROCm gaps are flagged **HIGH risk** (STATE.md Blockers); the CPU gate is the deterministic anchor and the part fully under our control. (Chosen over "ROCm is a hard blocking gate" — risks stalling on alpha issues outside our control — and over "empirically scope first," whose spike value is folded into the bring-up itself.)
- **D-03a:** ORA-04's literal "oracle passes on ROCm" remains the *target*; this decision sets the **completion bar** as "CPU gate green + ROCm executed with gaps documented," so a residual ROCm gap is a tracked follow-up, not a phase blocker. Surface any ROCm gap explicitly in verification (no silent pass).

### Determinism & Validation Anchor (CMP-02 / ~1e-6 contract)
- **D-04:** **cubecl-cpu IS the deterministic anchor — one impl per kernel.** The **single-threaded cubecl-cpu** kernel must reproduce the committed C++-transcription golden **bit-exact** (it is the deterministic anchor); **cubecl-hip** then matches cubecl-cpu **within ~1e-6** (GPU reduction order may differ, which the ~1e-6 contract tolerates per CONCERNS.md FP-ordering note). **No separate scalar-Rust reference is maintained** — there is exactly one Rust implementation of each kernel (the CubeCL kernel), relied upon to be bit-deterministic on the cubecl-cpu runtime against the golden. (This refines the discussion's "sequential reference is the anchor" answer: the *cubecl-cpu single-threaded path itself* plays the sequential-anchor role, rather than a distinct scalar port.)
- **D-04a — research/planning watch:** This decision assumes the **cubecl-cpu runtime is bit-deterministic single-threaded** (stable reduction order, no nondeterministic scheduling) so it can hit bit-exact against the C++ golden. If bring-up shows cubecl-cpu cannot be made bit-stable against the f32 golden, the fallback is to relax the cubecl-cpu anchor to ~1e-6 against the golden (and re-evaluate whether a scalar reference is needed) — flag this empirically early in the phase, before building the full kernel suite on the bit-exact assumption.

### Carried Forward (locked by prior phases — not re-litigated)
- **Faithful C++ mirror** discipline (Phase 1 D-11/D-12, Phase 2 D-01, Phase 3 D-04): kernels reproduce C++ behavior (which child is constructed vs subtracted, default-bin skip, `kEpsilon` placement), not idiomatic redesigns. **Do not "improve" the subtraction trick or reduction order.**
- **f32 end-to-end, ~1e-6 absolute, standard f32 accumulations**; integer-quantized histograms dropped (Phase 1 D-02/D-03).
- **`lgbm-compute` is the single CubeCL seam** (Phase 1 D-09, CMP-01); no crate above it names a CubeCL runtime.
- **CPU/ROCm are separate oracle gates** (Phase 1 D-02); committed-golden + idempotent-regen + header-only-transcription-fallback discipline (Phase 1/2/3).
- **Single-threaded deterministic core matching the pinned `deterministic=true force_row_wise=true num_threads=1` reference** (Phase 2 D-03), with per-row/per-feature independence as the parallel-ready seam.

### Claude's Discretion
- The exact `Backend` trait method signatures and the `Runtime` associated-type binding; the kernel buffer/launch/allocation API shape; the `Plane`-API capability-gating mechanism and the deterministic sequential-fallback structure (bounded by CMP-04 + SC#4); the precise gain-config parameter struct passed into `find_best_split` (bounded by D-01a); the synthetic-input fixture format and the histogram/split/partition golden serialization (bounded by the oracle-harness comparator seam); the cubecl-cpu vs cubecl-hip feature-flag / runtime-selection mechanism (bounded by CMP-03 "Cargo feature and/or runtime config"). When C++ behavior is the spec, the C++ source (below) is authoritative over any inferred default.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### C++ reference source (read-only port target — authoritative for all Phase-4 behavior)
- `LightGBM/src/treelearner/feature_histogram.hpp` — `ConstructHistogram` accumulation, `FindBestThreshold*` numerical/categorical split scan, `GetSplitGains`/`GetLeafGain`/`CalculateSplittedLeafOutput`, `ThresholdL1`, `Subtract` (histogram-subtraction trick), `kEpsilon`/`2*kEpsilon` placements, `SKIP_DEFAULT_BIN`/`NA_AS_MISSING` template flags — the **D-01 whole-kernel** target (histogram + best-split math).
- `LightGBM/src/treelearner/feature_histogram.cpp` — the non-inline split-finding tail co-located with the header above.
- `LightGBM/src/treelearner/serial_tree_learner.cpp` — `ConstructHistograms` (~405-470), `FindBestSplitsFromHistograms` (~474-580), `use_subtract` smaller-child selection (~398-580), and the data-partition row→leaf routing — authoritative for **what the kernels compute** (the orchestration around them is Phase 5, but the kernel inputs/outputs are defined here).
- `LightGBM/src/io/dense_bin.hpp`, `LightGBM/src/io/sparse_bin.hpp`, `LightGBM/src/io/multi_val_dense_bin.hpp`, `LightGBM/src/io/multi_val_sparse_bin.hpp` — per-bin-type `ConstructHistogram`/`Split` accumulators + iterators (the bin-storage side of the histogram kernel; reuse the Phase-2 store).
- `LightGBM/include/LightGBM/bin.h` (~180-258) — `offset`, `default_bin_`, `most_freq_bin_`, `SKIP_DEFAULT_BIN` semantics that drive which thresholds are scanned.
- `LightGBM/include/LightGBM/meta.h` — `kEpsilon = 1e-15f`, `kZeroThreshold = 1e-35f` constant definitions (load-bearing in gain/leaf-output math).
- `LightGBM/src/treelearner/ocl/histogram256.cl`, `histogram64.cl`, `histogram16.cl` — **reference GPU kernel design** (256-bins-per-workgroup) to mirror for the CubeCL histogram kernel.
- `LightGBM/src/treelearner/cuda/cuda_best_split_finder.cu` — reference design for the GPU best-split prefix-scan (largest .cu; warp/`Plane`-mapping reference for CMP-04).
- `LightGBM/src/treelearner/leaf_splits.hpp` (~98-180) — the **deterministic** ordered-summation branch (`if (... && !deterministic_)` gating) the Rust path must replicate for comparability.

### CubeCL (compute framework — alpha, pin exactly)
- Use the `find-docs` skill / `ctx7` for current CubeCL `Plane` API, runtime selection (cubecl-cpu / cubecl-hip), kernel-launch, and capability-query syntax — pinned at `cubecl = "0.10.0"` in the workspace `Cargo.toml`. CubeCL is alpha; verify APIs against current docs, do not rely on training data. (Per the user's global Context7/find-docs rule.)

### Foundations to build on (Phase 1–3 deliverables)
- `crates/lgbm-compute/src/lib.rs` — the existing kernel-free `Backend` trait skeleton (CMP-01 seam) to fill in this phase; `crates/lgbm-compute/Cargo.toml` (already declares `cubecl.workspace = true`).
- `crates/lgbm-dataset/` — the binned columnar store (`Bin` trait, Dense/Sparse/MultiVal, `FeatureGroup` offsets, `most_freq_bin_`/`default_bin_`) that feeds histogram construction; reuse as kernel inputs.
- `crates/lgbm-core/` — `Config` (gain/split hyperparameters), `src/types.rs` (f32 types), `src/error.rs` (`thiserror` boundary-error idiom), `Random` RNG.
- `crates/oracle-harness/` — comparator + committed-golden + idempotent-regen seam: `compare_exact_*` (bit-exact for the cubecl-cpu anchor) and the f32 ~1e-6 comparator (cubecl-hip). Extend `REFERENCE_MANIFEST.md` with the histogram/split/partition fixture set.
- `xtask` `bin-capture` / `model-capture` pattern + `xtask/cpp/` — the C++ golden-capture harness to extend with a histogram/split/partition capture subcommand (D-02, header-only transcription).

### Project-level contract
- `.planning/PROJECT.md` — Core Value, Constraints, Key Decisions (f32/~1e-6; standard f32 accumulations; CubeCL `Plane` mandate).
- `.planning/REQUIREMENTS.md` — CMP-01..05, ORA-04 (Phase 4 requirements).
- `.planning/ROADMAP.md` §"Phase 4" — goal + 5 success criteria.
- `.planning/STATE.md` — Blockers: CubeCL alpha pin + ROCm capability gaps (research flag HIGH); f32 transcendental CPU↔ROCm parity unproven (Phase 6, but relevant to ROCm bring-up here).
- `.planning/phases/01-oracle-contract-foundations/01-CONTEXT.md` — D-02 (~1e-6 every backend), D-03 (standard f32 accumulations, integer-quant dropped), D-08/D-09 (`lgbm-*` crates, lgbm-compute seam).
- `.planning/phases/02-dataset-binning-determinism-root/02-CONTEXT.md` — D-03 (single-threaded determinism + per-feature seams), the binned-store the histogram kernel consumes.
- `.planning/phases/03-tree-model-model-text-i-o-predict-parity/03-CONTEXT.md` — header-only-transcription capture fallback + layered-golden + committed/idempotent discipline (D-06/D-07) carried into D-02 here.

### Codebase maps (reference C++ architecture & porting concerns)
- `.planning/codebase/CONCERNS.md` §"Histogram construction + best-split finding", §"FP reduction ordering", §"Histogram subtraction trick", §"default-bin skip", §"kEpsilon" — the explicit porting-risk catalogue for this phase's hotpath.
- `.planning/codebase/ARCHITECTURE.md`, `.planning/codebase/STRUCTURE.md` — treelearner layer layout and the GPU-relevant subsystem flags.

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `crates/lgbm-compute` already exists as the `Backend` trait skeleton (`type Runtime;`, no methods) — Phase 4 binds `Runtime` to concrete CubeCL runtimes and adds the kernel methods; no new crate needed.
- `crates/lgbm-dataset` binned store (Dense/Sparse/MultiVal bins, `FeatureGroup` offsets, `most_freq_bin_`/`default_bin_`) is the bit-faithful histogram-kernel input — reuse directly, don't re-bin.
- `crates/oracle-harness` `compare_exact_*` (bit-exact anchor) + f32 ~1e-6 comparator (ROCm gate) + committed-golden/idempotent-regen seam — the histogram/split/partition goldens plug in here.
- `xtask` C++ golden-capture pipeline (`bin-capture`/`model-capture` + `xtask/cpp/`) — extend with a kernel-capture subcommand; header-only-transcription fallback carries over verbatim.
- `lgbm-core::Config` + `Random` — gain/split hyperparameters and the deterministic RNG already modeled.

### Established Patterns
- Faithful 1:1 C++ hand-port guarded by a parity test (Phase 1 D-11/D-12, Phase 2 D-01, Phase 3 D-04) — applies to the histogram accumulation, the gain scan, the subtraction trick, and the default-bin-skip logic. Reproduce *which* child is subtracted; do not "improve."
- Committed fixtures + idempotent C++-regen; **no C++ toolchain at normal test time**; header-only / verbatim transcription capture when `external_libs` are unvendored (Phase 1/2/3 precedent — needed again per D-02).
- Bit/byte-exact comparison for the deterministic anchor (cubecl-cpu vs golden) vs the ~1e-6 f32 oracle for the ROCm gate (cubecl-hip vs cubecl-cpu) — same exact-vs-tolerance split used in Phases 2/3.
- Single-threaded deterministic core matching `deterministic=true force_row_wise=true num_threads=1`; per-row/per-feature independence is the parallel/GPU seam.
- C++ constants in play: `kEpsilon = 1e-15f`, `kZeroThreshold = 1e-35f`; `2*kEpsilon` hessian bump is load-bearing in gain/leaf-output.

### Integration Points
- `lgbm-compute` depends on `lgbm-core` (Config/types/errors) and `lgbm-dataset` (binned store as kernel input). It must remain the **only** crate naming a `cubecl` runtime (CMP-01); the Phase-5 learner depends on the `Backend` trait, never on `cubecl`.
- The `Backend` trait + its kernels become the dependency root for **Phase 5** (the serial tree learner calls `construct_histograms`/`find_best_split`/`data_partition`) and downstream GBDT (Phase 6).
- ROCm bring-up runs on the **local ROCm GPU** (mandated test env); cubecl-hip selected via Cargo feature and/or runtime config (CMP-03). A CPU-only build must not require the ROCm toolchain (SC#1).
- The `LightGBM/` reference tree is **untracked** (never `git add`); kernel goldens must be C++-generated (header-only transcription) then **committed** into `tests/fixtures/`, never referenced from the untracked tree at test time (memory: lightgbm-ref-tree-untracked).

</code_context>

<specifics>
## Specific Ideas

- **Faithfulness over idiom, again:** every Phase-4 gray area resolved toward the closest C++ mirror — whole-kernel ops carrying the exact gain formula (not a redesigned split API), reproducing the subtraction trick and default-bin skip as-is, replicating the deterministic ordered-summation branch.
- **cubecl-cpu as the deterministic anchor is the key architectural bet:** one Rust impl per kernel, with the single-threaded cubecl-cpu runtime expected to be bit-deterministic against the C++ golden, and cubecl-hip matching it within ~1e-6. The risk (D-04a) is whether cubecl-cpu can be made bit-stable at f32 — validate this empirically *early*, before building the whole kernel suite on the assumption.
- **CPU-solid, ROCm-honest:** the CPU gate is the hard completion bar; ROCm is executed and its gaps documented rather than allowed to silently pass or to block the phase. CubeCL alpha + ROCm capability gaps are HIGH-risk unknowns owned outside our code.
- **Mirror the OpenCL `histogram256` workgroup design** (256 bins/workgroup) as the CubeCL histogram-kernel reference, and `cuda_best_split_finder.cu` for the prefix-scan / `Plane` warp mapping.
- **Layered kernel goldens stay maximally diagnostic** (Phase 2/3 D-06 discipline): separate histogram-accumulation / best-split / data-partition goldens so a failure points at accumulate vs gain-scan vs partition, not just "the backend is off."

</specifics>

<deferred>
## Deferred Ideas

- **Tree-learner orchestration** (leaf-wise growth, leaf-splits, `num_leaves`/`max_depth`, subtraction-trick *bookkeeping*, monotone/interaction constraints, feature subsampling, `force_row_wise`/`force_col_wise` selection) — Phase 5 (TRL-01..09). Phase 4 ships the kernels they call.
- **GBDT spine, objectives, metrics** — Phase 6; **DART/RF/GOSS** — Phase 7.
- **f32 transcendental (exp/log/pow/sigmoid) CPU↔ROCm parity** — primarily a Phase-6 objective concern, but ROCm bring-up here may surface early signal; note any findings for Phase 6.
- **Parallel (rayon) CPU histogram path** — the per-feature/per-row independence leaves the seam; a multi-threaded CPU path is a later, separately-validated optimization that must still match the deterministic anchor.
- **Integer-quantized / discretized histograms (QNT-01)** and **linear-tree kernels (LIN-01)** — v2, explicitly out of scope project-wide.
- **A residual ROCm oracle gap** (if CubeCL alpha / hardware blocks ~1e-6) — tracked as a Phase-4 follow-up per D-03a, to be closed when CubeCL/ROCm matures; ORA-04's full ROCm pass remains the standing target.

None other — discussion stayed within Phase 4 scope.

</deferred>

---

*Phase: 4-compute-backend-cpu-first-integer-histograms-rocm*
*Context gathered: 2026-06-05*
