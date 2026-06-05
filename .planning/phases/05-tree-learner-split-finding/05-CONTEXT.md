# Phase 5: Tree Learner + Split Finding - Context

**Gathered:** 2026-06-06
**Status:** Ready for planning

<domain>
## Phase Boundary

Phase 5 delivers a **histogram-based serial tree learner** that orchestrates the Phase-4 `lgbm-compute` `Backend` kernels (`construct_histograms` → `find_best_split` → `subtract_histograms` → `data_partition`) into **leaf-wise (best-first) tree growth**, growing the **exact same tree as C++ `serial_tree_learner.cpp`** — the keystone, highest-FP-risk subsystem, validated at **per-split** granularity. The Phase-4 boundary already pushed the per-feature gain math into `find_best_split` (P4 D-01/D-01a); this phase is the **orchestration** layer on top: leaf-wise queue, best-split aggregation across features, the histogram pool + subtraction-trick bookkeeping, data partition routing, feature subsampling, and the two histogram-build strategies.

In scope (TRL-01..05, 07, 08, 09):
- **TRL-01** — histogram-based serial learner: `ConstructHistograms` → `FindBestSplitsFromHistograms` → `Split`.
- **TRL-02** — histogram-subtraction trick reproducing the C++ smaller-child selection and the byte-identical FP path (within ~1e-6 f32) the model is defined against.
- **TRL-03** — leaf-wise (best-first) growth respecting `num_leaves` / `max_depth` caps.
- **TRL-04** — split-gain scan with exact gain formula + tie-breaking (`lambda_l1`, `lambda_l2`, `min_gain_to_split`, `min_sum_hessian_in_leaf`, `min_data_in_leaf`, `max_delta_step`, `path_smooth`). **Note:** the per-feature gain math already lives in the Phase-4 `find_best_split` kernel (P4 D-01a); Phase 5 *consumes* it and owns the cross-feature argmax + tie-break selection, not the per-bin gain derivation.
- **TRL-05** — numerical threshold splits with C++-matching missing/zero routing.
- **TRL-07** — data partition (row→leaf routing) feeding histogram subtraction.
- **TRL-08** — per-tree and per-node feature subsampling (`feature_fraction`, `feature_fraction_bynode`, `feature_fraction_seed`) via RNG parity.
- **TRL-09** — `force_row_wise` / `force_col_wise` histogram-build strategies, both output-matching.

Out of scope: **TRL-06 categorical splits** (`SplitCategorical`/`FindBestThresholdCategorical`) — explicitly deferred to **Phase 7**; the **GBDT spine / objectives / metrics** (Phase 6 — this phase has no boosting loop, no objective producing g/h at runtime); **DART/RF/GOSS** variants (Phase 7); **prediction** (shipped Phase 3); **monotone/interaction constraints, forced splits, CEGB, extra-trees** (Phase 7); **parallel/rayon CPU or multi-GPU** histogram paths (post-MVP optimization on the per-feature/per-row seam); **integer-quantized / linear-tree** kernels (v2, dropped project-wide).

</domain>

<decisions>
## Implementation Decisions

### Reference-tree capture & golden strategy (SC#1 per-split parity)
- **D-01:** **Extend the header-only C++ transcription harness** (P1–P4 precedent) to produce the reference tree + per-split snapshots. `external_libs` are unvendored and `LightGBM/` is untracked, so the full C++ `serial_tree_learner` cannot be built at test time; the `xtask` capture harness transcribes the learner verbatim, emits goldens over fixed g/h + binned inputs, and **commits** them. Replayable with no C++ toolchain at normal test time. (Chosen over vendoring `external_libs` to build real `lib_lightgbm`, and over a transcribe-then-cross-check-vs-real-build hybrid.)
- **D-02:** **Full end-to-end re-transcription of `serial_tree_learner`** in the golden-emitter — NOT orchestration-only. The emitter transcribes the *whole* learner including the per-feature histogram accumulation + gain scan, **independently** of the existing Phase-4 kernel transcriptions. The overlap between the two transcriptions (P4 kernel emitter vs P5 learner emitter) becomes an intentional **cross-check** of the same math.
  - **D-02a — research/planning watch:** Two independent transcriptions of the per-feature histogram/gain math now exist (P4 kernel-capture + P5 learner-capture). They MUST agree bit-for-bit where they overlap (same synthetic inputs → same per-feature histograms/gains). Plan a guard that surfaces any drift between them rather than letting them diverge silently. This redundancy is the point (belt-and-suspenders for the keystone subsystem), but it needs an explicit consistency check.

### Gradient/hessian fixture source (no objective/GBDT loop yet)
- **D-03:** **Both, layered** — the parity tests feed the learner from two g/h sources:
  1. **Synthetic deterministic g/h** vectors hand-crafted to exercise every split path (sign/magnitude spread, ties, missing/zero routing, default-bin skip, subtraction-trick edge cases).
  2. **Captured real first-iteration g/h** from a real C++ objective's iteration-1 on a real dataset (realistic distribution → "grows the same tree as C++" under realistic conditions).
  Mirrors the P2/P4 layered-golden discipline (maximally diagnostic). The captured-g/h objective(s)/dataset, `boost_from_average` on/off, and exact capture config are **Claude's discretion** (researcher decides), bounded by the faithful-mirror contract.

### Phase slice / requirement sequencing
- **D-04:** **Spine first, then parity additions.** Lock and per-split-validate the **minimal faithful tree** first — `force_row_wise` + default `feature_fraction=1.0` (no subsampling) + numeric splits + leaf-wise growth + subtraction trick + data partition (TRL-01, 02, 03, 04, 05, 07). THEN add `force_col_wise` (TRL-09) and per-node feature-subsampling RNG parity (TRL-08) as validated additions on top of the proven spine. (Chosen over building all TRL-01..09 in one pass.)

### Histogram pool faithfulness (SC#2 subtraction trick)
- **D-05:** **Mirror the full C++ histogram pool + eviction** faithfully — not just the FP-load-bearing parts. Port the `HistogramPool` sizing / eviction / reuse machinery alongside the load-bearing behavior (smaller-child selection, which child is constructed vs subtracted, parent-histogram retention, subtract math/order). Strongest-fidelity bet for the keystone subsystem; removes any risk that a pool-ordering effect is observable in the FP path. (Chosen over mirroring FP-load-bearing parts only with a simplified allocation, and over deferring the bar to Claude's discretion.)

### Validation depth
- **D-06:** **Per-split snapshot = full per-bin gain array for every candidate feature** at each split decision. The per-split parity assertion captures the entire bin-by-bin gain scan for every feature at every split — localizing any divergence to a specific (feature, bin), not just "wrong winner." (Chosen over per-feature-best-only, and over the both-layered middle option — the user wants maximum diagnostic resolution.)
- **D-07:** **Tree-match unit = full tree structure, bit-faithful.** "Grew the same tree as C++" asserts equality of the **entire grown tree**: every internal node's split feature, threshold/bin, and missing/default direction, AND every leaf's output value (`CalculateSplittedLeafOutput` with `lambda_l1`/`lambda_l2`/`path_smooth`/`max_delta_step`), compared via the **Phase-3 model-text `%.17g`** machinery. Leaf-output parity is in-scope here (it feeds Phase-6 scores), not deferred. (Chosen over split-decisions-only.)

### Carried Forward (locked by prior phases — not re-litigated)
- **Faithful C++ mirror** discipline (P1 D-11/D-12, P2 D-01, P3 D-04, P4 D-04/D-01): reproduce *which* child is constructed vs subtracted, the default-bin skip, `kEpsilon`/`2·kEpsilon` placement, and tie-break order — never an idiomatic redesign. Do not "improve" the subtraction trick, the histogram pool, or the reduction order.
- **f32 end-to-end, ~1e-6 absolute, standard f32 accumulations** into f64 histogram cells (`hist_t = double`); integer-quantized histograms dropped (P1 D-02/D-03).
- **Single-threaded deterministic core** matching the pinned `deterministic=true force_row_wise=true num_threads=1` reference (P2 D-03); per-row/per-feature independence is the parallel/GPU seam, not exercised this phase.
- **The gain math lives in the Phase-4 kernel** (P4 D-01/D-01a): the learner consumes `find_best_split`, it does NOT re-derive per-bin gains in the *runtime* path. (The *golden-emitter* re-transcribes the math per D-02 for cross-check — distinct from the runtime learner.)
- **`lgbm-compute` is the single CubeCL seam** (P1 D-09, CMP-01): the Phase-5 learner depends on the `Backend` trait, never on a `cubecl` runtime type.
- **Committed goldens + idempotent C++-regen + header-only/verbatim transcription** capture when `external_libs` unvendored (P1/2/3/4); `LightGBM/` is untracked — never `git add` it, kernel/learner goldens are committed into `tests/fixtures/`.
- **CPU is the bit-exact hard gate; ROCm is a separate ~1e-6 gate** (P4 D-03/D-04): the cubecl-cpu path is the deterministic anchor.

### Claude's Discretion
- The tree-learner crate placement/structure (new `lgbm-treelearner` crate vs an existing crate) and the learner↔`Backend` wiring; the leaf-wise priority-queue data structure and the exact leaf-split bookkeeping (`leaf_begin_`/`leaf_count_`) shape; the captured-g/h objective(s)/dataset/`boost_from_average` config (D-03); the `force_col_wise=true` and `force_row_wise=true` capture configs (D-04); the leaf-wise queue tie-break determinism mechanism (bounded by "must match C++ selection order"); the golden serialization/layering format for per-split + per-tree fixtures (bounded by the oracle-harness comparator + Phase-3 `%.17g` machinery). When C++ behavior is the spec, the C++ source below is authoritative over any inferred default.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### C++ reference source (read-only port target — authoritative for all Phase-5 behavior)
- `LightGBM/src/treelearner/serial_tree_learner.cpp` — **the primary port target.** `ConstructHistograms` (~405-470), `FindBestSplitsFromHistograms` (~474-580) cross-feature aggregation, `use_subtract` smaller-child selection (~398-580), `Split` / `SplitInner` node creation + data-partition routing, leaf-wise best-first loop, `ColSampler` feature subsampling, `force_row_wise`/`force_col_wise` strategy branch. **D-01/D-02 whole-learner transcription target.**
- `LightGBM/src/treelearner/serial_tree_learner.h` — learner member state (`data_partition_`, `histogram_pool_`, `best_split_per_leaf_`, `larger_leaf_splits_`/`smaller_leaf_splits_`, `col_sampler_`), and the leaf-split bookkeeping the Rust learner mirrors.
- `LightGBM/src/treelearner/feature_histogram.hpp` + `feature_histogram.cpp` — `ConstructHistogram`, `FindBestThreshold*` numerical scan, `GetSplitGains`/`GetLeafGain`/`CalculateSplittedLeafOutput`, `ThresholdL1`, `Subtract` (subtraction trick), `kEpsilon`/`2·kEpsilon` placements, `SKIP_DEFAULT_BIN`/`NA_AS_MISSING` template flags. Already transcribed in Phase 4 (kernel layer) — re-transcribed end-to-end here per D-02 (cross-check).
- `LightGBM/src/treelearner/leaf_splits.hpp` (~98-180) — the **deterministic** ordered-summation branch (`if (... && !deterministic_)` gating) the Rust path must replicate for comparability; `LeafSplits` sum_gradient/sum_hessian seeding.
- `LightGBM/src/treelearner/data_partition.hpp` — `DataPartition::Split` stable reorder (row→leaf), `leaf_begin_`/`leaf_count_`, `indices_` bookkeeping feeding the subtraction trick (TRL-07). The kernel-layer `data_partition` op exists (Phase 4); this is the orchestration around it.
- `LightGBM/src/treelearner/col_sampler.hpp` — `feature_fraction` / `feature_fraction_bynode` / `feature_fraction_seed` RNG-driven feature selection (TRL-08); must reproduce the C++ RNG draw sequence + call order using the Phase-1 `Random`.
- `LightGBM/src/io/tree.cpp`, `LightGBM/include/LightGBM/tree.h` — `Tree::Split`/`SplitCategorical` node-creation API (~14 args), leaf-output storage, the structure the grown tree must match (and serialize via Phase-3 `%.17g`).
- `LightGBM/src/io/dense_bin.hpp`, `sparse_bin.hpp` — per-bin `ConstructHistogram`/`Split` accumulators + iterators (histogram-kernel + partition side; reuse the Phase-2 store).
- `LightGBM/include/LightGBM/bin.h` (~180-258) — `offset`, `default_bin_`, `most_freq_bin_`, `SKIP_DEFAULT_BIN` semantics driving which thresholds are scanned + missing/zero routing (TRL-05).
- `LightGBM/include/LightGBM/meta.h` — `kEpsilon = 1e-15f`, `kZeroThreshold = 1e-35f` (load-bearing in gain/leaf-output math).

### Phase-4 compute backend (the dependency root this phase orchestrates)
- `crates/lgbm-compute/src/lib.rs` — the `Backend` trait + `CpuBackend`: `construct_histograms`, `find_best_split` (carries the gain formula, P4 D-01a), `subtract_histograms`, `data_partition`. The learner calls these; it never names a `cubecl` runtime (CMP-01).
- `.planning/phases/04-compute-backend-cpu-first-integer-histograms-rocm/04-CONTEXT.md` — D-01/D-01a (whole-kernel ops + the gain-config surface that flows into `find_best_split`), D-02 (header-only transcription capture), D-04 (cubecl-cpu deterministic anchor), the Phase 4↔5 cut.
- `crates/lgbm-compute/src/kernels/split.rs` — the in-kernel split/gain implementation (currently modified in the working tree) the learner's cross-feature selection sits on top of.

### Foundations to build on (Phase 1–3 deliverables)
- `crates/lgbm-dataset/` — the immutable binned columnar store (`Bin` trait, Dense/Sparse/MultiVal, `FeatureGroup` offsets, `most_freq_bin_`/`default_bin_`) consumed as histogram + partition input; do NOT re-bin.
- `crates/lgbm-model/` — `Tree`/`GbdtModel` + the model-text `%.17g`/`{:g}` formatter (Phase 3) — the **tree-match comparison machinery** for D-07 (full bit-faithful tree structure incl. leaf outputs).
- `crates/lgbm-core/` — `Config` (gain/split/subsampling hyperparameters), `src/types.rs` (f32 types), `src/error.rs` (`thiserror` boundary idiom), `Random` LCG (for TRL-08 feature-subsampling RNG parity).
- `crates/oracle-harness/` — `compare_exact_*` (bit-exact CPU anchor) + f32 ~1e-6 comparator + committed-golden/idempotent-regen seam; extend `REFERENCE_MANIFEST.md` with the per-split + per-tree learner fixtures.
- `xtask` `bin-capture`/`model-capture`/kernel-capture pattern + `xtask/cpp/` — extend with a learner-capture subcommand (D-01/D-02 full transcription).

### Project-level contract
- `.planning/PROJECT.md` — Core Value (f32, ~1e-6), Constraints, Key Decisions (standard f32 accumulations; faithful mirror; CubeCL `Plane` mandate).
- `.planning/REQUIREMENTS.md` — TRL-01..05, 07, 08, 09 (Phase 5); TRL-06 deferred to Phase 7.
- `.planning/ROADMAP.md` §"Phase 5" — goal + 5 success criteria.
- `.planning/STATE.md` — Blockers (CubeCL alpha pin; ROCm gaps — relevant only if ROCm parity is re-checked here).
- `.planning/phases/01-oracle-contract-foundations/01-CONTEXT.md` — `Random` RNG parity, f32 strategy, config alias table.
- `.planning/phases/02-dataset-binning-determinism-root/02-CONTEXT.md` — single-threaded determinism + per-feature seams; the binned store the learner consumes.
- `.planning/phases/03-tree-model-model-text-i-o-predict-parity/03-CONTEXT.md` — the `%.17g` formatter + tree-structure serialization reused for the D-07 tree-match.

### Codebase maps (reference C++ architecture & porting concerns)
- `.planning/codebase/CONCERNS.md` §"Histogram construction + best-split finding", §"FP reduction ordering", §"Histogram subtraction trick", §"default-bin skip", §"kEpsilon" — the porting-risk catalogue for this hotpath.
- `.planning/codebase/ARCHITECTURE.md`, `.planning/codebase/STRUCTURE.md` — treelearner layer layout, the `TreeLearner::Train(grad,hess,is_first_tree)→Tree*` factory seam, GPU-relevant subsystem flags.

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `crates/lgbm-compute` `Backend` trait — the per-feature whole-kernel ops (`construct_histograms`, `find_best_split`, `subtract_histograms`, `data_partition`) are the learner's building blocks; the learner aggregates across features and manages growth. No re-implementation of per-bin gain math in the runtime path.
- `crates/lgbm-model` `Tree` + `%.17g` formatter — directly drives the D-07 full-tree bit-faithful comparison.
- `crates/lgbm-core::Config` + `Random` — gain/subsampling hyperparameters + the deterministic RNG for TRL-08.
- `crates/lgbm-dataset` binned store — bit-faithful histogram/partition input (P2).
- `crates/oracle-harness` + `xtask` capture pipeline — committed-golden + header-only-transcription harness to extend with the learner-capture subcommand.

### Established Patterns
- Faithful 1:1 C++ hand-port guarded by a parity test (P1 D-11/D-12, P2 D-01, P3 D-04, P4 D-04) — applies to the leaf-wise loop, cross-feature argmax + tie-break, smaller-child selection, histogram pool, default-bin skip, missing/zero routing.
- Committed fixtures + idempotent C++-regen; **no C++ toolchain at normal test time**; header-only/verbatim transcription when `external_libs` unvendored.
- Layered, maximally-diagnostic goldens (P2/P3/P4 D-06 discipline): per-split (full per-bin gain arrays) + per-tree (full structure) so a failure localizes to accumulate vs gain-scan vs selection vs partition vs leaf-output.
- Bit-exact comparison for the deterministic CPU anchor; ~1e-6 for any ROCm cross-check.
- C++ constants in play: `kEpsilon = 1e-15f`, `kZeroThreshold = 1e-35f`; `2·kEpsilon` hessian bump load-bearing.

### Integration Points
- The Phase-5 learner is the dependency consumer of the Phase-4 `Backend` trait and the producer of `Tree` objects consumed by the **Phase-6 GBDT loop** (`TreeLearner::Train(grad, hess, is_first_tree) → Tree`). It must remain CubeCL-free above the `lgbm-compute` seam (CMP-01).
- Captured first-iteration g/h (D-03) couples a thin capture path to a C++ objective — kept fixture-only (committed), not a runtime dependency on Phase 6.
- The `LightGBM/` reference tree is **untracked** (never `git add`); learner goldens are C++-generated via transcription then **committed** into `tests/fixtures/` (memory: lightgbm-ref-tree-untracked).

</code_context>

<specifics>
## Specific Ideas

- **Maximal fidelity + diagnosability, deliberately:** every Phase-5 gray area resolved toward the most faithful / most diagnostic option — full histogram-pool + eviction mirror (not just the FP-load-bearing parts), full end-to-end learner re-transcription (independent cross-check vs the Phase-4 kernel transcription), full per-bin gain arrays per candidate feature at every split, and full bit-faithful tree-structure match incl. leaf-output values. This is the keystone, highest-FP-risk subsystem — the user wants no validation blind spots.
- **Spine-first sequencing:** prove the minimal faithful tree (`force_row_wise`, `feature_fraction=1.0`, numeric, leaf-wise + subtraction + partition) at per-split parity BEFORE layering `force_col_wise` and feature-subsampling RNG parity. Working parity widens outward, consistent with the project's vertical-slice spine philosophy.
- **Two-transcription cross-check is a feature, not redundancy waste:** the Phase-4 kernel emitter and the Phase-5 full-learner emitter both transcribe the per-feature histogram/gain math; they must agree bit-for-bit where they overlap, and a guard should surface any drift (D-02a).
- **Leaf outputs are in-scope now** (D-07): `CalculateSplittedLeafOutput` parity is validated here because those values feed Phase-6 scores — not deferred to "when training exists."

</specifics>

<deferred>
## Deferred Ideas

- **TRL-06 categorical splits** (`SplitCategorical`/`FindBestThresholdCategorical`: `max_cat_threshold`, `cat_smooth`, `min_data_per_group`, `max_cat_to_onehot`, `cat_l2`) — explicitly **Phase 7** per REQUIREMENTS.md, not this phase. The numeric-split learner ships here; categorical is a thin addition on the proven spine.
- **GBDT spine, objectives, metrics, bagging, early stopping** — Phase 6. This phase grows one tree from fixed g/h; the boosting loop that produces g/h iteratively is next.
- **DART / RF / GOSS** variants — Phase 7.
- **Monotone / interaction constraints, forced splits/bins, extra-trees, CEGB, refit** — Phase 7.
- **Parallel (rayon) CPU or multi-GPU histogram-build path** — post-MVP optimization on the per-feature/per-row independence seam; must still match the deterministic anchor when added.
- **Captured-g/h objective/dataset specifics + `boost_from_average` config** — left to the researcher (Claude's discretion under D-03), bounded by the layered-golden contract; noted here so it isn't lost.
- **ROCm cross-check of the full learner** — the kernels are ROCm-gated (Phase 4); whether the *orchestrated* learner is re-run on ROCm here vs deferred to a later parity sweep is a research/planning call (CPU bit-exact is the hard gate per P4 D-03).

None other — discussion stayed within Phase 5 scope.

</deferred>

---

*Phase: 5-tree-learner-split-finding*
*Context gathered: 2026-06-06*
