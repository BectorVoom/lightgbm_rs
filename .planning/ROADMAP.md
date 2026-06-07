# Roadmap: LightGBM-rs

## Overview

A pure-Rust, parity-faithful port of Microsoft LightGBM on a CubeCL CPU/ROCm backend, built bottom-up along a dependency-forced spine so numerical fidelity is provable at every layer. Data types are `f32` (single-precision) end-to-end to match the C++ reference defaults, and the oracle tolerance is ~1e-6 absolute. The journey starts by pinning the oracle contract and the foundations (bit-exact RNG, f32 numerical strategy, config) that everything downstream is validated against, then locks the binning determinism root, then proves prediction parity against a C++-trained model *before* training exists. It next builds the f32 compute backend (the CubeCL-churn containment boundary), the histogram tree learner (the keystone FP-parity subsystem), and finally the GBDT loop with core objectives/metrics — the first moment a full train→predict run hits ~1e-6 (f32) parity. The remaining boosting variants, objectives, metrics, constraints, and SHAP are thin additions on the proven spine, and Python bindings land last as a translation layer over a validated Rust facade. Each phase is a vertical, oracle-validated slice: working numerical parity widens outward from binning → prediction → training rather than being deferred to the end.

## Phases

**Phase Numbering:**

- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

Decimal phases appear between their surrounding integers in numeric order.

- [x] **Phase 1: Oracle Contract + Foundations** - f32 ~1e-6 oracle, pinned C++ reference, bit-exact RNG, config, f32 numerical strategy, workspace *(3/3 plans; UAT 10/10 passed + security 9/9 threats resolved (0 open), retroactively verified 2026-06-06)* (completed 2026-06-05)
- [x] **Phase 2: Dataset + Binning (determinism root)** - Bit-identical BinMapper, columnar bin store, missing/categorical encoding, EFB, metadata, ingestion *(7/7 plans executed incl. gap-closure 02-06 + 02-07; GAP-1/GAP-2 + CR-01 + WR-01 closed — default ingest unified onto the faithful single C++ Dataset::Construct, trivial features dropped, grouping verified; ready to re-verify)* (completed 2026-06-05)
- [x] **Phase 3: Tree Model + Model Text I/O + Predict Parity** - Load a C++-trained model and predict identically (parity before training exists) (completed 2026-06-05)
- [x] **Phase 4: Compute Backend (CPU-first f32 histograms → ROCm)** - Backend trait, f32 histogram/split/score kernels, CPU then ROCm, both at ~1e-6
- [x] **Phase 5: Tree Learner + Split Finding** - Histogram serial learner, subtraction trick, leaf-wise growth, split-gain scan with per-split parity *(9/9 plans; 05-09 closed the final mfb>0 leaf-0 2-ULP BIT-EXACT via a real lib_lightgbm 4.6 FP execution trace — the serial learner is bit-exact to the real binary on BOTH committed corpora)*
- [x] **Phase 6: GBDT Spine + Core Objectives/Metrics** - First end-to-end ~1e-6 (f32) train→predict with bagging, early stopping, Rust-native API *(6/6 plans executed; gap-closure 06-06 closed all five verification gaps A–E: CR-01 constant-tree leaf_count model-text now byte-exact, WR-01 every matrix cell asserts (no swallowed Results), CR-02 early-stop decoupled from metric_freq, GAP E reg_sqrt builder setter + golden, WR-03 subset renewal landed. Task 2b decision: regression_l1 + bagging TYPED-REJECTED (BoostingError::UnsupportedConfig) and deferred — L1 sign-gradient split-gain knife-edge over the bagged subset diverges from the C++ leaf STRUCTURE (rust:0.0 vs cpp:11.0); the related binary+bagging+bfa knife-edge is tracked as DEF-06-01. cargo test --workspace GREEN.)* (completed 2026-06-07)
- [x] **Phase 7: Parity-Completing Variants** - GOSS/DART/RF, categorical/EFB splits, remaining objectives/metrics, ranking, SHAP, monotone, refit, importance (completed 2026-06-07)
- [ ] **Phase 8: Python Bindings** - PyO3 + numpy bindings mirroring the official `lightgbm` Booster/Dataset/sklearn API

## Phase Details

### Phase 1: Oracle Contract + Foundations

**Goal**: A falsifiable, f32 single-precision oracle contract (~1e-6 absolute) and the foundations (bit-exact RNG, f32 numerical strategy, config, workspace, pinned reference) that every later phase is validated against.
**Mode:** mvp
**Depends on**: Nothing (first phase)
**Requirements**: FND-01, FND-02, FND-03, FND-04, CFG-01, CFG-02, CFG-03, ORA-01, ORA-02
**Success Criteria** (what must be TRUE):

  1. The oracle harness compares Rust output against a pinned, deterministic C++ LightGBM 4.6 reference (`deterministic=true`, `force_row_wise=true`, `num_threads=1`, fixed seed, default `float` `score_t`/`label_t` width) at ~1e-6 absolute tolerance, and the reference build/config manifest is checked in and regenerates goldens idempotently.
  2. A user can run the ported `Random` LCG and reproduce a captured 100k-draw C++ sequence (`RandInt16`/`RandInt32`/`NextFloat`/`NextInt`/`Sample(N,K)` across the branch boundary) bit-for-bit, with `u32` wraparound and `f32` `NextFloat`.
  3. The Cargo workspace (loosely-coupled crates by responsibility) builds under edition 2024 with `Cargo.lock` and `rust-toolchain.toml` committed; `thiserror` domain errors exist at crate boundaries and `anyhow` propagates at app/test layers.
  4. A config struct accepts the ~110 in-scope hyperparameters, resolves aliases (`num_iteration`/`n_estimators`/`num_boost_round`, etc.) via a data table matching `config_auto.cpp`, and rejects invalid combos with typed `Result` errors mirroring C++ `Config::Set` CHECK constraints.
  5. The f32 single-precision data-type contract and ~1e-6 oracle tolerance (standard f32 histogram/score accumulations, no integer-quantized reduction strategy) is documented as a Key Decision in PROJECT.md so no later phase targets an unfalsifiable invariant.**Plans**: 3 plans (incl. 1 gap-closure)

**Wave 1**

  - [x] 01-01-PLAN.md — Walking-skeleton spine: virtual workspace + f32 types/errors + bit-exact Random LCG + oracle comparator + pinned C++ RNG golden/manifest

**Wave 2** *(blocked on Wave 1 completion)*

  - [x] 01-02-PLAN.md — Hand-ported flat Config: struct/defaults + verbatim alias table + seed derivation + typed CHECK validation + drift-checker

**Wave 3** *(gap closure — blocked on Wave 2; closes SC#4 / CFG-02 + CFG-03)*

  - [x] 01-03-PLAN.md — Gap closure: deterministic SortAlias alias-collision resolution (CR-02) + present()-routed seed/enum empty-is-absent (CR-01), each with regression tests

### Phase 2: Dataset + Binning (determinism root)

**Goal**: A binned, immutable columnar dataset whose bin boundaries and bin assignments are bit-identical to C++ — the determinism root every downstream split inherits.
**Mode:** mvp
**Depends on**: Phase 1
**Requirements**: DAT-01, DAT-02, DAT-03, DAT-04, DAT-05, DAT-06, DAT-07, ORA-03
**Success Criteria** (what must be TRUE):

  1. `BinMapper::ValueToBin` (literal `(r+l-1)/2` + `<=` search, `max_bin`/`min_data_in_bin`/`bin_construct_sample_cnt`/`data_random_seed`) produces `bin_upper_bound_` arrays and edge-case bin indices (NaN per `MissingType`, `+0.0`/`-0.0`, on-boundary, out-of-range categorical) that match C++ golden snapshots exactly.
  2. A user can ingest dense and CSR/CSC sparse in-memory matrices and metadata (labels, weights, init_score, query/group boundaries) into a Dense/Sparse-bin columnar store that is immutable after finish-load.
  3. Missing-value handling (`use_missing`, `zero_as_missing`, `MissingType`) and categorical encoding (category→bin, low-frequency folding) route exactly as C++.
  4. Exclusive Feature Bundling (`enable_bundle`) reproduces C++ feature grouping bit-for-bit.
  5. Per-stage parity tests cover the bin granularity (bin boundaries + per-row bin assignment), localizing any divergence to binning before histograms exist.

**Plans**: 6 plans (incl. 1 gap-closure)

**Wave 1**

  - [x] 02-01-PLAN.md — Crate scaffold + golden-capture harness + numeric BinMapper (FindBin/ValueToBin) golden layers 1+2

**Wave 2** *(blocked on Wave 1)*

  - [x] 02-02-PLAN.md — Bin trait + DenseBin (incl. 4-bit) + SparseBin + FeatureGroup offsets/PushData + Dataset finish_load immutability

**Wave 3** *(blocked on Wave 2)*

  - [x] 02-03-PLAN.md — Categorical folding (category→bin) + missing-value routing golden parity (layers 1+3)

**Wave 4** *(blocked on Wave 3)*

  - [x] 02-04-PLAN.md — Metadata + from_mat/from_csr/from_csc ingestion + dense/CSR/CSC equivalence + example-dataset parity

**Wave 5** *(blocked on Wave 4; EFB sequenced last per MEDIUM-risk capture flag, has checkpoint)*

  - [x] 02-05-PLAN.md — Exclusive Feature Bundling (MultiValBin + FastFeatureBundling) group/offset golden parity (layer 3)

**Wave 6** *(gap closure — closes SC#1 / DAT-01 + SC#5 / ORA-03 bin stage / DAT-07; default-config scaled filter_cnt divergence, see 02-VERIFICATION.md)*

  - [x] 02-06-PLAN.md — Gap closure: scaled `filter_cnt = (min_data_in_leaf * total_sample_cnt) / num_rows` in a single source-of-truth helper (CR-01/IN-02) + default feature_pre_filter=true ingest parity golden that fails-before/passes-after (CR-02) — DONE; GAP-1/GAP-2 closed, workspace green, capture idempotent

**Wave 7** *(gap closure — closes CR-01 (default-ingest Construct divergence) + WR-01; restores DAT-07 / ORA-03 / DAT-01/02/05 at the determinism root, see 02-06-REVIEW.md / re-verification gaps_found)*

  - [x] 02-07-PLAN.md — Gap closure: default ingest unified onto the faithful single C++ `Dataset::Construct` (`construct_bundled`) — trivial features DROPPED (`used_feature_map_[real]=-1`), EfbSamples built to the exact c_api.cpp:1352-1374 sampled-set convention (no second RNG draw); golden emitter models C++ Construct (trivial dropped, per-non-trivial group/subfeature via in-file FastFeatureBundling, is_sparse=true); parity test asserts trivial-exclusion + per-non-trivial group/subfeature parity + bit-exact stored bins, panics on missing golden (WR-01), HARD fails-before/passes-after — DONE; CR-01 + EFB parity hole + masking + WR-01 closed, workspace green, capture idempotent

### Phase 3: Tree Model + Model Text I/O + Predict Parity

**Goal**: Load a C++-trained model and predict identically — prediction parity proven independently of (and before) any training code.
**Mode:** mvp
**Depends on**: Phase 2
**Requirements**: DAT-08, DAT-09, PRD-01, PRD-02, PRD-03, PRD-06
**Success Criteria** (what must be TRUE):

  1. A user can load a C++-trained LightGBM `.txt` model and produce raw-score predictions within ~1e-6 (f32) of the C++ reference on the deterministic CPU path.
  2. Transformed predictions (`ConvertOutput` sigmoid/softmax) and leaf-index predictions (`pred_leaf`) match the C++ reference.
  3. The Rust writer emits the exact LightGBM text schema (tree structure, leaf values, bin mappers, feature metadata) including `%.17g` float formatting, and a load→predict→write→reload round-trip is byte-stable.
  4. Sub-range prediction (`start_iteration` / `num_iteration`) returns the C++-matching slice of the ensemble.

**Plans**: 4 plans

**Wave 1** *(enabling slice — crate + %.17g formatter + golden-capture pipeline)*

  - [x] 03-01-PLAN.md — lgbm-model crate skeleton + ModelError + `%.17g`/`{:g}` formatter (the DAT-09 linchpin, built FIRST) + `xtask model-capture` committed golden corpus (capture-path decision gate)

**Wave 2** *(blocked on Wave 1)*

  - [x] 03-02-PLAN.md — Regression slice: load→raw-predict (dense/CSR/CSC, f64 accumulate)→write byte-exact→reload (DAT-08/DAT-09/PRD-01) — faithful array Tree + GbdtModel + model_text envelope + predict driver

**Wave 3** *(blocked on Wave 2)*

  - [x] 03-03-PLAN.md — Transform + leaf-index slice: core ConvertOutput (sigmoid/softmax/ova/identity) + multiclass per-class stride + pred_leaf + categorical-split parity (PRD-02/PRD-03)

**Wave 4** *(blocked on Wave 3)*

  - [x] 03-04-PLAN.md — Sub-range slice: InitPredict `start_iteration`/`num_iteration` (`-1==all`) parity (PRD-06) — full D-06 layered battery (1-5) green

**UI hint**: no

### Phase 4: Compute Backend (CPU-first integer histograms → ROCm)

**Goal**: An isolated `lgbm-compute` backend whose f32 histogram, split-scan, and data-partition kernels produce results matching CPU and ROCm within ~1e-6 — the CubeCL-churn containment boundary.
**Mode:** mvp
**Depends on**: Phase 2
**Requirements**: CMP-01, CMP-02, CMP-03, CMP-04, CMP-05, ORA-04
**Success Criteria** (what must be TRUE):

  1. All CubeCL usage lives behind one `lgbm-compute` `Backend` trait; no crate above it names a CubeCL runtime, and a CPU-only build needs no ROCm toolchain.
  2. Standard f32 histogram construction, best-split-finding, and data-partition kernels run on the cubecl-cpu reference path and produce results matching a sequential f32 CPU reference within ~1e-6.
  3. The same kernels run on the cubecl-hip (ROCm) backend, selectable by Cargo feature and/or runtime config, and produce results matching the CPU backend within ~1e-6 (f32).
  4. CUDA warp-level reductions are expressed via CubeCL's `Plane` API with startup capability-gating (`Plane::Ops`, f64, atomics) and a deterministic sequential fallback when a capability is absent.
  5. The oracle suite executes and passes on the ROCm backend for the histogram/split/partition kernels (mandated test environment), with CPU-runtime and ROCm treated as separate gates.

**Plans**: 4 plans

**Wave 1** (foundation + D-04a determinism spike FIRST)

  - [x] 04-01-PLAN.md — ComputeError + cpu/rocm runtime selection + startup capability gate + minimal histogram kernel + the D-04a bit-determinism spike (RUN FIRST) + CMP-01 containment guard — **DONE: D-04a SETTLED (cubecl-cpu fold bit-exact, 25 launches + vs C++-order sequential); CMP-01/02/04 foundation in place; cargo test --workspace green**

**Wave 2** *(blocked on 04-01)*

  - [x] 04-02-PLAN.md — First vertical slice: construct_histograms whole-kernel op + xtask kernel-capture (header-only C++ transcription) + committed histogram golden + bit-exact cubecl-cpu parity — **DONE: Backend::construct_histograms + CpuBackend wired end-to-end; 18-case D-02a golden (dense+sparse, default-bin, u8/u16/u32, grad/hess spread) replays BIT-EXACT via compare_exact_f64_bits; CMP-01/02/05(hist)/ORA-04(cpu) satisfied; cargo test --workspace green**

**Wave 3** *(blocked on 04-02)*

  - [x] 04-03-PLAN.md — find_best_split (gain math inside kernel, verbatim) + data_partition + subtract_histograms (A3 in-scope) + split/partition/subtract goldens + bit-exact cpu parity — **DONE: Backend::find_best_split (D-01a, both REVERSE t-1+offset and FORWARD t+offset branches, exact kEpsilon/2*kEpsilon + gate order) + data_partition (stable reordered index array + split_point) + subtract_histograms wired end-to-end; split.txt (per-candidate gains + winner, reverse/forward/default-bin-skip/L1/no-split) / partition.txt / subtract.txt replay BIT-EXACT; full CMP-05 kernel set closed on cpu (CMP-04/05, ORA-04 cpu); 1 Rule-1 L1-gain codegen bug fixed; cargo test --workspace green**

**Wave 4** *(blocked on 04-03; has ROCm GPU checkpoint)*

  - [x] 04-04-PLAN.md — ROCm/hip bring-up (rocm feature, f32-accumulate on no-f64 device) + capability gate on gfx1100 + ~1e-6 hip-vs-cpu separate gate + documented ROCm gaps (D-03a) — **DONE: rocm feature binds HipRuntime+AmdDevice{0}; Capabilities::accumulate_type gate (F64 cpu anchor vs F32 no-f64 hip); f32-cell MIRROR kernels + generic *_f32_on<R> launchers (histogram/split/subtract), data_partition_on<R> shared (f64-free); rocm_smoke.rs asserts gfx1100 matrix (Plane YES/f64 NO/atomic YES/plane_size 32); kernel_parity.rs hip layer = separate ~1e-6 gate (hip f32 vs cpu f64 anchor→Vec<f32> via compare_within), two-tier so the documented f32-vs-f64 accumulation gap is surfaced (no silent pass) not blocked; RAN on real gfx1100 (ROCm 7.1.1): smoke 2/2, partition bit-exact, subtract ≤1.16e-10, histogram/split within f32 ULP (max rel ≈1.1e-7); 04-ROCM-GAPS.md records all per-case max abs-diff (D-03a). cpu bit-exact gate untouched & green; CPU-only build needs no ROCm toolchain (SC#1). CMP-03/CMP-04/ORA-04 satisfied**

**Phase 4 COMPLETE** — compute backend closed on cpu (bit-exact hard gate) + hip (best-effort, documented f32 gap).

### Phase 5: Tree Learner + Split Finding

**Goal**: A histogram-based serial tree learner that grows the exact same tree as C++ — the keystone, highest-FP-risk subsystem, validated at per-split granularity.
**Mode:** mvp
**Depends on**: Phase 4
**Requirements**: TRL-01, TRL-02, TRL-03, TRL-04, TRL-05, TRL-07, TRL-08, TRL-09
**Success Criteria** (what must be TRUE):

  1. Given fixed gradients/hessians, the learner (`ConstructHistograms` → `FindBestSplitsFromHistograms` → `Split`) selects the same split feature, split bin/threshold, and missing-direction as C++ for every split, validated against per-split candidate-gain snapshots (not just the winner).
  2. The histogram-subtraction trick reproduces the C++ smaller-child selection and derived-child histogram (matching the C++ f32 path within ~1e-6), and the default-bin-skip scan considers the same candidate-threshold set.
  3. Leaf-wise (best-first) growth respects `num_leaves`/`max_depth`, and the split-gain formula matches C++ (`kEpsilon` positions, `lambda_l1`/`lambda_l2`/`min_gain_to_split`/`min_sum_hessian_in_leaf`/`min_data_in_leaf`/`max_delta_step`/`path_smooth`).
  4. Numerical threshold splits route missing/zero exactly as C++; data partition (row→leaf) feeds the subtraction trick correctly.
  5. Per-tree/per-node feature subsampling (`feature_fraction`, `feature_fraction_bynode`, `feature_fraction_seed`) selects the same features via RNG parity, and both `force_row_wise`/`force_col_wise` strategies produce matching trees.

**Plans**: 9 plans (4 original + 5 gap-closure; all COMPLETE — 05-09 closed bit-exact via a real-binary FP execution trace)

**Wave 1** *(spine prerequisites — parallel; no shared files)*

  - [x] 05-01-PLAN.md — Phase-4 boundary re-open: thread authoritative `skip_default_bin`/`na_as_missing` through `Backend::find_best_split` (replace the `cfg_skip_default_bin` heuristic) + `skip_default_bin==false` divergence golden (TRL-05 enabler)
  - [x] 05-02-PLAN.md — Enabling slice: new `lgbm-treelearner` crate + `TreeLearnerError` + reuse `SplitInfo` + `split_gt` tie-break + `Tree::split` mutation/growth arrays + `learner-capture`/`learner_parity` Wave-0 harness (failing end-to-end test in place)

**Wave 2** *(blocked on 05-01 + 05-02 — the keystone spine)*

  - [x] 05-03-PLAN.md — Faithful tree-learner spine (`force_row_wise`, `feature_fraction=1.0`, numeric, `missing_type=None`): leaf-wise loop + subtraction trick + FixHistogram + HistogramPool (D-05 full mirror) + DataPartition + LeafSplits; per-split (full per-bin gain arrays, D-06) + full-tree (`%.17g`, D-07) parity + D-02a two-transcription cross-check (TRL-01,02,03,04,05,07)

**Wave 3** *(blocked on 05-03 — parity additions on the proven spine)*

  - [x] 05-04-PLAN.md — `force_col_wise`==`force_row_wise`==C++ tree (TRL-09, Open Q2 RESOLVED: config-flag no-op over the shared construct_histograms op on the deterministic anchor) + per-tree/per-node feature subsampling RNG parity via `ColSampler` (TRL-08, ResetByTree+GetByNode draw-sequence) + captured real iter-1 g/h full-tree parity (D-03, regression-l2 + binary-logloss). **DONE: col_wise.txt / col_sampler.txt / real_gh.txt goldens + learner_parity_{row_vs_col,col_sampler_rng,real_gh_full_tree} replay bit-exact; 1 Rule-1 fix (tree leaf_count records the ACTUAL data_partition count, update_cnt=true, not the reconstructed SplitInfo count); cargo test --workspace green; byte-idempotent**

**Wave 4** *(GAP CLOSURE — blocked on 05-01..04; from 05-VERIFICATION.md gaps_found)*

  - [x] 05-05-PLAN.md — CR-01 fix (D-09): single shared `offset_for_most_freq_bin` helper unifying the three contradictory offset rules + adopt the real-LightGBM `offset==1`/compacted-histogram convention so stored threshold / partition `--th` / predict routing agree + oracle-independent `get_leaf`-tally==`leaf_count` self-consistency assertion (TRL-05, TRL-07, TRL-01)

**Wave 5** *(GAP CLOSURE — blocked on 05-05)*

  - [x] 05-06-PLAN.md — CR-02 fix (D-08): real `lib_lightgbm` 4.6 oracle — `learner-oracle-capture` xtask + python dumper trains the spine + a new `most_freq_bin>0` corpus deterministically (`deterministic=true force_row_wise=true num_threads=1` fixed seed) on the pip wheel's real binary, commits `spine_real.txt`/`mfb_pos_real.txt`, and validates the Rust learner bit-exact incl. the previously-uncovered offset==1 path (TRL-09, TRL-05, TRL-01) [human-gated capture]
    - **⚠ OUTCOME: CR-02 CLOSED (real oracle exists), but the port is FALSIFIED → new BLOCKER CR-03.** The real binary exposed that the Rust serial learner grows structurally wrong trees (wrong split points / mis-partitioned `leaf_count` / leaf outputs like `-17.99` vs `0.55`; mfb>0: 0-row leaf, missing zero-sentinel split, `decision_type` 0≠2). The two real-binary gates are committed `#[ignore]`d (live record, not weakened). **TRL-09/TRL-05/TRL-01 are NOT satisfied** — deferred to a CR-03 learner-fix plan. See 05-06-SUMMARY.md.

**Wave 7** *(GAP CLOSURE — NEW: closes BLOCKER CR-03; must run before 05-07)*

  - [x] 05-08-PLAN.md — CR-03 fix (diagnose→fix→re-validate): make the Rust serial learner reproduce the real `lib_lightgbm` 4.6 goldens bit-exact (spine + most_freq_bin>0). Faithful per-`missing_type` scan-branch dispatch — gate the FORWARD scan off for `missing_type==None` so `default_left` stays true (`decision_type=2`) and the REVERSE-only winner restores correct split points / `leaf_count` / Newton leaf outputs + the mfb>0 zero-sentinel default-bin split (no 0-row leaf; node-2 threshold `1.0000000180025095e-35`); where diagnosis proves it, correct the bin-0 real-value fixture mapping. Then un-`#[ignore]` `learner_parity_{spine,mfb_pos}_real_binary` and assert bit-exact (TRL-05, TRL-07, TRL-01, TRL-09). Plan-checker: 0 blockers.
    - **✓ OUTCOME: CR-03 CLOSED (commit c564036).** Spine corpus BIT-EXACT vs `spine_real.txt`; mfb>0 corpus structurally bit-exact vs `mfb_pos_real.txt` (split_feature, threshold incl. the `1.0000000180025095e-35` zero sentinel, decision_type=2 2 2, child topology, leaf_count with no 0-row leaf, internal_count) + 3/4 leaf values. PRIMARY fix = child LeafSplits direct pass-through (was swapping smaller/larger slots → −17.99 vs 0.55); plus the FORWARD-dispatch gate (decision_type 0→2), MaybeRoundToZero (−0.0→+0), and bin-0 kZeroThreshold mapping. `spine_real` gate un-`#[ignore]`d + passing in the default suite. The ONLY residual is the mfb>0 node-2 leaf-0 value (Δ 2.3e-16 = one f64 ULP), a kEpsilon cascade DEFERRED to 05-07 (its subtraction-trick/HistogramPool wiring); `mfb_pos` gate stays `#[ignore]`d with a narrowed reason, assertions UNCHANGED, ~4 orders inside the ≤1e-12 contract. **TRL-05/TRL-07/TRL-01/TRL-09 satisfied bit-exact vs the real binary on the spine.** routing self-consistency (CR-01) holds; kernel_parity 4/4. See 05-08-SUMMARY.md.

**Wave 8** *(GAP CLOSURE — BLOCKED on CR-03 closure: re-validates against the real goldens, only satisfiable after 05-08)*

  - [x] 05-07-PLAN.md — WR-01/WR-02 fix (D-05): wire the dead subtraction-trick + HistogramPool into the live `find_best_splits` growth path (larger child derived by `parent − smaller`, pool slots read/reused) and re-validate bit-exact against the real lib_lightgbm goldens (TRL-01, TRL-02, TRL-05) — **re-sequenced to wave 8, `depends_on` adds 05-08; its "still matches the real goldens" gate becomes satisfiable once CR-03 is closed**
    - **✓ OUTCOME: WR-01/WR-02 CLOSED (commit 037e011).** The subtraction trick (`larger = parent − smaller` via `Backend::subtract_histograms`) + HistogramPool slot reuse are wired into the LIVE `find_best_splits` growth path (mirroring C++ serial_tree_learner.cpp:364-378); the dead `let _ = subtract_from;` discard + orphaned `_pool` are gone; `learner_parity_growth_path_subtract` proves derived-larger-child == direct build cell-for-cell AND the spine stays bit-exact. `cargo test --workspace` GREEN (learner_parity 11 passed / 1 ignored; kernel_parity 4/4); routing self-consistency holds. The mfb>0 node-2 leaf-0 2.3e-16 ULP did NOT close — its 05-08 subtraction-trick attribution is **DISPROVEN** (leaf 0 is the directly-built smaller child, untouched by subtraction); RE-ATTRIBUTED to a 2-ULP f64 accumulation-order subtlety in the FixHistogram-active DIRECT histogram build and **deferred to new plan 05-09**. The mfb gate stays `#[ignore]`d with a corrected honest reason (commit fbd4f1d); NO assertion weakened (~4 orders inside ≤1e-12). **TRL-01/TRL-02/TRL-05 satisfied for the wired path.** See 05-07-SUMMARY.md.

**Wave 9** *(GAP CLOSURE — mfb>0 node-2 leaf-0 bit-exact via a real-binary FP execution trace)*

  - [x] 05-09-PLAN.md — **CLOSED BIT-EXACT (commits `c675d3b` localization + `2ced5a2` fix).** Task-1 localized the mfb>0 node-2 leaf-0 2-ULP to the leaf-total `sum_hessian` SEED. Task-2 (user-authorized Option B): built `lib_lightgbm` 4.6 CPU-only single-thread and captured an attributable FP execution trace. **GROUND TRUTH** (`[GSD-META] feature 0 most_freq_bin=0 default_bin=0 offset=1`): the mfb_pos corpus is SPARSE (rate 0.1667 > kSparseThreshold) so the real BinMapper collapses `most_freq_bin_ = default_bin_ = ValueToBin(0) = 0` (`bin.cpp:491-499`) → the real binary runs the **offset==1 path (FixHistogram NO-OP)**, the SAME as the spine — NOT a `most_freq_bin>0` path. The harness had mislabeled it `most_freq_bin=2/offset=0`, spuriously activating FixHistogram on node-2's direct build (reconstructing a `~1e-15` bin-2 hessian) and polluting the REVERSE scan by 2 ULPs. ALSO: C++ seeds child `LeafSplits` DIRECTLY from the parent `SplitInfo` (`best_split_info.left_sum_hessian = best_sum_left_hessian - kEpsilon`, `feature_histogram.hpp:1042`; `serial_tree_learner.cpp:851-871`), NOT a re-fold (which lost the kEpsilon provenance: `4.0` vs C++ `4.000000000000001`). **Fix (`2ced5a2`):** (1) corrected the corpus to the ground-truth `most_freq_bin=0/offset=1`; (2) added `LeafSplits::init_from_split` and seeded child leaves from the parent SplitInfo. `learner_parity_mfb_pos_real_binary` un-`#[ignore]`d + PASSING bit-exact (node-2 leaf-0 `0.59999999999999953`); 12 passed / 0 ignored; kernel_parity 4/4; spine_real + growth_path_subtract unregressed; `cargo test --workspace` GREEN. `assert_real_tree_parity` byte-unchanged (no tolerance); LightGBM/ never git-added. **TRL-01/TRL-05 closed bit-exact vs the real binary.** See 05-09-SUMMARY.md.

### Phase 6: GBDT Spine + Core Objectives/Metrics

**Goal**: The first end-to-end ~1e-6 (f32) train→predict run — the simplest boosting variant proves the full spine before any variant is added.
**Mode:** mvp
**Depends on**: Phase 5
**Requirements**: BST-01, BST-02, BST-03, BST-07, OBJ-01, OBJ-02, OBJ-03, MET-01, MET-02, API-01
**Success Criteria** (what must be TRUE):

  1. A user can call the Rust-native API (`Dataset`, `Booster`, `train`, `predict`) to train a GBDT model and predict, with outputs within ~1e-6 (f32) of the C++ reference and a same-tree structural match on every backend.
  2. The GBDT loop (`TrainOneIter`, `UpdateScore`, per-class trees, shrinkage, `boost_from_average`) and score updater accumulate with deterministic reduction ordering.
  3. Core objectives (`regression`, `regression_l1`, `binary`, `multiclass`, `multiclassova`, `custom`) compute grad/hess, `ConvertOutput`, `BoostFromScore`, and `reg_sqrt` to within ~1e-6 (f32) of the reference.
  4. Core metrics (`l1`, `l2`, `rmse`, `binary_logloss`, `binary_error`, `auc`, `multi_logloss`) plus multi-metric infrastructure (`metric_freq`, training-metric eval) match the reference, and early stopping (`early_stopping_round`, `first_metric_only`, `early_stopping_min_delta`) fires identically.
  5. Bagging / row subsampling (`bagging_fraction`/`bagging_freq`/`bagging_seed`, pos/neg, `bagging_by_query`) selects the same rows via RNG-matching sequence and call order.

**Plans**: 6 plans (5 spine-first vertical slices D-14→D-17 + 1 gap-closure)

**Deferral (06-06 Task 2b — decision: typed-reject):** `regression_l1 + bagging` is
TYPED-REJECTED in Phase 6 (`BoostingError::UnsupportedConfig`) and DEFERRED to a
later phase. The L1 sign-gradient split-gain is a knife-edge over the bagged subset
that diverges from the C++ reference in leaf STRUCTURE (e.g. a 2-vs-3-leaf split
count; `rust:0.0` vs `cpp:11.0` at `regression_l1_bag1_es0_bfa0` tree 0). The
faithful subset-path median-residual `RenewTreeOutput` IS implemented and retained
(commit 8330cee) and full-corpus `regression_l1` stays bit-exact, but no leaf-VALUE
renewal can fix a divergent leaf STRUCTURE — so the combination is rejected with an
honest typed error rather than shipping wrong-but-similar leaves. A related
pre-existing `binary + bagging + boost_from_average` per-tree split-count knife-edge
(`binary_bag1_es0_bfa1`) is logged in `06.../deferred-items.md` (DEF-06-01) for the
same future fix.

**Wave 1** *(Wave-0 foundation — scaffolds + extensions + failing end-to-end test)*

  - [x] 06-01-PLAN.md — Scaffold the 4 new crates (lgbm-objective/metric/boosting/lgbm) + error boundaries; Tree shrinkage/add_bias; learner add_prediction_to_score/renew_tree_output hook; failing boosting_parity scaffold + capture stub

**Wave 2** *(blocked on 06-01 — the minimal end-to-end spine, D-14/D-15)*

  - [x] 06-02-PLAN.md — regression(L2) + l2/rmse + GBDT loop + f64 ScoreUpdater + boost_from_average + builder→Config/Booster/train/predict; spine L1–L5 real-binary goldens; resolve Open-Q1/Q2

**Wave 3** *(blocked on 06-02 — objective/metric breadth, D-17 step 1)*

  - [x] 06-03-PLAN.md — regression_l1 (PercentileFun + RenewTreeOutput) + binary (sigmoid) + custom closure (OBJ-02) + binary_logloss/binary_error/auc; per-objective L1–L5 goldens

**Wave 4** *(blocked on 06-03 — per-class structural axis, D-16)*

  - [x] 06-04-PLAN.md — multiclass(softmax) + multiclassova + multi_logloss; loop generalized to num_class trees/iter (class-major layout + class_need_train); multiclass/ova L1–L5 goldens (bit-exact L2/L5 over 5-iter horizon, documented softmax exp-libm residual)

**Wave 5** *(blocked on 06-04 — bagging + early-stop axes + full matrix, D-17 steps 3+4)*

  - [x] 06-05-PLAN.md — BaggingSampleStrategy (RNG-replay D-13 golden, bit-exact) + OOB score update + early stopping (BST-07) + metric infra (MET-02); full ~40-cell D-07 cross-product replay (regression L2 bit-exact across all axes incl. bagging; documented residuals for non-L2 bagging / l1 bfa-off / multiclass es)

**Wave 6** *(gap closure — verification gaps_found 3/5 SC; closes CR-01/WR-01/WR-03/CR-02 + reg_sqrt verification gap)*

  - [x] 06-06-PLAN.md — Tighten D-07 matrix assertions (no swallowed Results, WR-01); subset-path median-residual renewal for regression_l1+bagging (WR-03; retained but regression_l1+bagging TYPED-REJECTED per Task 2b decision — see Deferral above); Tree::as_constant(count) + byte-exact constant-tree model-text assertion (CR-01); decouple early-stop eval from metric_freq + metric_freq>1+ES golden (CR-02); real-binary reg_sqrt=1 golden + .reg_sqrt(bool) setter (GAP E)

### Phase 7: Parity-Completing Variants

**Goal**: Complete full single-machine parity — every remaining boosting variant, objective, metric, constraint, and prediction mode lands as a thin, oracle-validated addition on the proven spine.
**Mode:** mvp
**Depends on**: Phase 6
**Requirements**: BST-04, BST-05, BST-06, TRL-06, OBJ-04, OBJ-05, OBJ-06, MET-03, MET-04, PRD-04, PRD-05, ADV-01, ADV-02, ADV-03, ADV-04, ADV-05, ADV-06, ADV-07
**Success Criteria** (what must be TRUE):

  1. GOSS (`top_rate`/`other_rate` with matching gradient-magnitude sort + amplification), DART (`drop_rate`/`max_drop`/`skip_drop`/`uniform_drop`/`xgboost_dart_mode`/`drop_seed`), and Random Forest (averaged trees, mandatory bagging) each train models within parity of the C++ reference.
  2. Categorical splits (`SplitCategorical`: `max_cat_threshold`/`cat_smooth`/`min_data_per_group`/`max_cat_to_onehot`/`cat_l2`) produce matching category bitsets, gains, and model-text round-trip.
  3. Remaining regression objectives (`huber`/`fair`/`poisson`/`quantile`/`mape`/`gamma`/`tweedie`), cross-entropy objectives, and ranking objectives (`lambdarank`/`rank_xendcg` with query boundaries + DCGCalculator + `objective_seed`) match the reference; extended + ranking metrics (`ndcg`/`map`/`average_precision`/`auc_mu`/...) match per-query.
  4. SHAP/feature contributions (`predict_contrib` over full node/cover structure) and prediction early stopping (`pred_early_stop`/`_freq`/`_margin`) produce C++-matching outputs.
  5. Monotone constraints (basic/intermediate/advanced + `monotone_penalty`), interaction constraints, forced splits/bins, extra trees, CEGB, refit/continue training (`Booster.refit()`), and feature importance reporting each reproduce the C++ behavior.

**Plans**: 12 plans (one phase, dependency-ordered sequential waves per D-01; Wave 0 = the D-05 bagged-subset determinism diagnostic, authored first; one end-of-phase verification gate)
Plans:
**Wave 1**

- [x] 07-01-PLAN.md — Wave 0 (D-05): bagged-subset split-gain determinism — branch = **FAITHFUL-FIX**. A source-built lib_lightgbm 4.6 FP execution trace proved the knife-edge was a `min_gain_shift` OPERAND bug (Rust used the RAW leaf sum_hessian; C++ uses the 2*kEpsilon-BUMPED value → Rust shift ~7 ULPs too high, rejecting bagged-subset splits whose current_gain exceeds C++'s shift by 1 ULP). Fixed `find_best_split` (f64+f32) + per_bin_gains + kernel_capture transcription (split.txt regenerated bit-idempotent). **DEF-06-01 CLOSED** (binary_bag1_es0_bfa1 tree-0 = 4 leaves bit-exact); **regression_l1+bagging UN-DEFERRED** (typed-reject removed + no-split ObtainAutomaticInitialScore fallback → constant tree-0 = label median 11.0); the 4 regression_l1_bag1_* cells assert real-binary parity. Bounded, hard-capped L1 cross-feature gain-tie residual documented (07-D05-DECISION.md). cargo test --workspace GREEN (boosting_parity 26/26, kernel_parity 4/4, learner_parity 12/12). (completed 2026-06-07)

**Wave 2** *(blocked on Wave 1 completion)*

- [x] 07-02-PLAN.md — Objectives breadth A (OBJ-04): huber/fair/quantile/mape

**Wave 3** *(blocked on Wave 2 completion)*

- [x] 07-03-PLAN.md — Objectives breadth B (OBJ-04/05): poisson/gamma/tweedie + cross_entropy/cross_entropy_lambda

**Wave 4** *(blocked on Wave 3 completion)*

- [x] 07-04-PLAN.md — Extended metrics (MET-03): regression/xentropy/multiclass metrics

**Wave 5** *(blocked on Wave 4 completion)*

- [x] 07-05-PLAN.md — GOSS (BST-04): sample strategy + amplification + RNG-replay

**Wave 6** *(blocked on Wave 5 completion)*

- [x] 07-06-PLAN.md — DART (BST-05): drop+normalize (4 branches) + drop RNG-replay

**Wave 7** *(blocked on Wave 6 completion)*

- [x] 07-07-PLAN.md — Random Forest (BST-06): averaged trees + mandatory bagging

**Wave 8** *(blocked on Wave 7 completion)*

- [x] 07-08-PLAN.md — Categorical splits (TRL-06, D-06/D-07): additive learner re-open + numeric-spine no-regression gate

**Wave 9** *(blocked on Wave 8 completion)*

- [x] 07-09-PLAN.md — Ranking stack (OBJ-06/MET-04/bagging_by_query): lambdarank/rank_xendcg + ndcg/map + DCGCalculator + query bagging

**Wave 10** *(blocked on Wave 9 completion)*

- [x] 07-10-PLAN.md — Prediction modes (PRD-04/05): TreeSHAP predict_contrib + pred early stop

**Wave 11** *(blocked on Wave 10 completion)*

- [x] 07-11-PLAN.md — Advanced learner constraints (ADV-01..05): monotone/interaction/forced/extra-trees/CEGB

**Wave 12** *(blocked on Wave 11 completion)*

- [x] 07-12-PLAN.md — Advanced model ops (ADV-06/07): refit/continue + feature importance

### Phase 8: Python Bindings

**Goal**: A Python interface mirroring the official `lightgbm` package, layered over the validated Rust facade.
**Mode:** mvp
**Depends on**: Phase 7
**Requirements**: PYB-01, PYB-02, PYB-03, PYB-04
**Success Criteria** (what must be TRUE):

  1. A Python user can train and predict through PyO3 + maturin bindings whose `Booster`/`Dataset` API mirrors the official `lightgbm` package, releasing the GIL (`allow_threads`) around training and returning owned arrays.
  2. NumPy interop (rust-numpy) accepts both f32 and f64 dense/sparse input and returns array outputs, with contiguity/dtype handled explicitly so results match the C++ Python package for either width.
  3. The sklearn-style wrapper API (`LGBMClassifier`/`LGBMRegressor`/`LGBMRanker`) matches the official wrappers' semantics.
  4. Python `custom` objective/metric callbacks and `Booster.refit()` work and reproduce reference outputs.

**Plans**: 8 plans (dependency-ordered sequential waves; MVP vertical slices over the validated Rust facade)Plans:
**Wave 1**

- [x] 08-01-PLAN.md — Rust facade slice: D-02 raw→bin→train bridge + new Booster methods (batch predict / feature_importance / refit / model text I/O) + custom-metric (feval) eval-history hook, oracle-tested (3/3 tasks; 41 lgbm + 2 oracle tests green)

**Wave 2** *(blocked on Wave 1 completion)*

- [x] 08-02-PLAN.md — Crate scaffold (pinned pyo3 0.27 / numpy 0.27.1 / pyo3-polars 0.26.0) + minimal PyO3 numpy-dense train→predict with GIL release + A/B parity (PYB-01)

**Wave 3** *(blocked on Wave 2 completion)*

- [x] 08-03-PLAN.md — Widen input: f32/f64 dense + scipy CSR/CSC sparse (PYB-02)

**Wave 4** *(blocked on Wave 3 completion)*

- [x] 08-04-PLAN.md — polars zero-copy via Arrow + dtype→categorical routing (PYB-02/D-03/D-04)

**Wave 5** *(blocked on Wave 4 completion)*

- [x] 08-05-PLAN.md — params dict coercion + recognized-but-unimplemented rejection (D-06/07/08)

**Wave 6** *(blocked on Wave 5 completion)*

- [x] 08-06-PLAN.md — custom obj/metric callbacks + Booster.refit() (PYB-04)

**Wave 7** *(blocked on Wave 6 completion)*

- [ ] 08-07-PLAN.md — sklearn wrappers + callbacks list + lgb.cv + plotting (PYB-03/D-09)

**Wave 8** *(blocked on Wave 7 completion)*

- [ ] 08-08-PLAN.md — persistence: C++-compatible text I/O + pickle (D-10)

**UI hint**: no

## Progress

**Execution Order:**
Phases execute in numeric order: 1 → 2 → 3 → 4 → 5 → 6 → 7 → 8

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Oracle Contract + Foundations | 3/3 | Complete    | 2026-06-05 |
| 2. Dataset + Binning | 7/7 | Complete    | 2026-06-05 |
| 3. Tree Model + Model Text I/O + Predict Parity | 4/4 | Complete    | 2026-06-05 |
| 4. Compute Backend (CPU-first → ROCm) | 4/4 | Complete    | 2026-06-05 |
| 5. Tree Learner + Split Finding | 9/9 | Complete (bit-exact vs real lib_lightgbm 4.6 on both corpora) | 2026-06-06 |
| 6. GBDT Spine + Core Objectives/Metrics | 6/6 | Complete    | 2026-06-07 |
| 7. Parity-Completing Variants | 12/12 | Complete    | 2026-06-07 |
| 8. Python Bindings | 6/8 | In Progress|  |
