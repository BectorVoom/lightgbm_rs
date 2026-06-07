---
gsd_state_version: 1.0
milestone: v1.0
milestone_name: milestone
status: executing
stopped_at: Phase 7 context gathered
last_updated: "2026-06-07T05:20:00.000Z"
last_activity: 2026-06-07 -- Phase 07 Plan 01 complete (D-05 faithful-fix)
progress:
  total_phases: 8
  completed_phases: 6
  total_plans: 45
  completed_plans: 34
  percent: 76
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-06-05)

**Core value:** For identical inputs and config, reproduce C++ LightGBM outputs to within ~1e-6 absolute difference on every backend (CPU and ROCm), using f32 (single-precision) data types matching the C++ reference defaults.
**Current focus:** Phase 07 — parity-completing-variants

## Current Position

Phase: 07 (parity-completing-variants) — EXECUTING
Plan: 2 of 12
Status: Executing Phase 07 (07-01 Wave 0 / D-05 COMPLETE)
Last activity: 2026-06-07 -- Phase 07 Plan 01 complete (D-05 faithful-fix)

### Plan 07-01 result (Wave 0 / D-05 — bagged-subset split-gain determinism FAITHFUL-FIX)

Plan 07-01 SETTLED D-05 (the bagged-subset split-gain knife-edge) BEFORE any bagging-dependent wave (GOSS W4, RF W6) — branch = FAITHFUL-FIX, proven by a source-built `lib_lightgbm` 4.6 FP execution trace.

- **ROOT CAUSE (source-built FP trace):** the Rust `min_gain_shift` was computed from the RAW leaf `sum_hessian`, while C++ uses the `2*kEpsilon`-BUMPED value (C++ bumps at the `FindBestThreshold` call site, `feature_histogram.hpp:174`, so `BeforeNumerical` divides by the bumped sum_hessian). The raw value made the Rust `min_gain_shift` ~7 ULPs too high, rejecting bagged-subset deeper splits whose `current_gain` exceeds the C++ `min_gain_shift` by a SINGLE f64 ULP (e.g. node-1 `current_gain` 0x4013fffff4924920 > C++ `min_gain_shift` 0x4013fffff492491f; Rust raw shift 0x4013fffff4924925 rejected it). NOT an irreducible f32 / near-zero artifact — a deterministic OPERAND bug.
- **FAITHFUL FIX:** `find_best_split_cpu` (f64) + `find_best_split_raw_f32_on` (f32 hip) + the `per_bin_gains` diagnostic now compute `min_gain_shift` from the bumped `sum_hessian`; the `kernel_capture.cpp` golden-capture transcription had the same bug (fixed; `split.txt` regenerated, byte-idempotent; `kernel_parity` 4/4 bit-exact).
- **DEF-06-01 CLOSED:** `binary_bag1_es0_bfa1` tree-0 now grows 4 leaves bit-exact matching C++; the matrix binary+bagging cell takes the STRICT bit-exact path; `subset_determinism_diagnostic` HARD-asserts the tree-0 leaf-count parity. Cleared in `deferred-items.md`.
- **regression_l1 + bagging UN-DEFERRED:** removed the `BoostingError::UnsupportedConfig` reject; added `GBDT::no_split_constant_value` (C++ `ObtainAutomaticInitialScore` fallback, `gbdt.cpp:418-429`) so the bfa-off `regression_l1` no-split constant tree-0 carries the label median (11.0), not 0.0. The 4 `regression_l1_bag1_*` cells assert real-binary parity.
- **BOUNDED known-divergence (documented, capped):** the non-bagged `regression_l1` bfa-off cells retain a cross-feature L1 gain tie on degenerate 2-row nodes (two features separate the node perfectly with gains equal to ~1 f64 ULP; Rust vs C++ order them oppositely). Topology matches C++ EXACTLY; |leaf diff| < 0.1; count hard-capped. Pre-fix this cell grew degenerate single-leaf STUBS that the old skip-guard hid — the fix is a strict improvement. No assertion weakened.
- **DECISION RECORD:** `07-D05-DECISION.md` (source-build FP-trace evidence + the confirmed fold-order/operand root cause + the faithful fix + the bounded residual).
- **gate:** `cargo test --workspace` GREEN (50 binaries, 0 failed; boosting_parity 26/26, kernel_parity 4/4, learner_parity 12/12 incl. spine_real/mfb_pos); `LightGBM/` never git-added; the /tmp instrumented build + C++ instrumentation reverted.

Next: `/gsd-execute-phase 7 --wave 1` — the wave-1 plans now build on a SETTLED D-05 (bagging variants inherit the faithful answer).

### Plan 06-06 result (gap-closure A–E + Task 2b — regression_l1 + bagging TYPED-REJECTED)

> **NOTE (superseded by 07-01):** the Task 2b typed-reject for regression_l1 + bagging and DEF-06-01 below were both reversed/closed by the D-05 faithful-fix in Plan 07-01 (a source-built FP trace proved the divergence was a min_gain_shift operand bug, not an irreducible leaf-structure flip). Kept for history.

Plan 06-06 closed all five Phase-6 verification gaps and resolved Task 2b by typed-rejecting regression_l1 + bagging.

- **GAP A (WR-01):** every D-07 matrix cell asserts numerically; 0 standalone `.ok();`; `MATRIX_RESIDUAL_TOL` capped `<= 1e-4` with the max observed leaf-value diff asserted in-code (6d0c84a).
- **GAP B (WR-03) / Task 2b — DECISION: typed-reject:** the faithful subset-path median-residual `RenewTreeOutput` (`train_on_subset_returning_partition`, 8330cee) is correct and RETAINED, but regression_l1 + bagging diverges from C++ in leaf STRUCTURE (an L1 sign-gradient split-gain knife-edge over the bagged subset — matrix failed at `regression_l1_bag1_es0_bfa0` tree 0 with **rust:0.0 vs cpp:11.0**, a 2-vs-3-leaf split-count divergence). No leaf-VALUE renewal can fix a divergent leaf STRUCTURE, so the combination is rejected with a NEW typed `BoostingError::UnsupportedConfig { what }` at the top of `train_one_iter` (BEFORE any tree grows: `is_renew_tree_output() && bagging.is_bagging_active()`). The 4 `regression_l1_bag1_*` matrix cells now ASSERT the typed error (teeth: panic on Ok or a different error). Deferred past Phase 6 (ROADMAP + REQUIREMENTS OBJ-01 notes). Commit 96b4700.
- **GAP C (CR-01):** `Tree::as_constant(value, count)` with `leaf_count=vec![count]`; 3 gbdt.rs sites thread `self.num_data`; byte-exact constant-tree model-text test (a6c2013).
- **GAP D (CR-02):** `early.update` guarded by `es_enabled` not `do_eval`/metric_freq; metric_freq still thins recorded history; clarifying doc line (bbec2c1, d10e3ac).
- **GAP E (reg_sqrt):** `TrainingBuilder.reg_sqrt(bool)` setter routing into `Config.reg_sqrt`; reg_sqrt/mf2es capture cells + skip-on-missing-golden tests (d10e3ac).
- **DEVIATION (Rule 1, pre-existing, out-of-scope):** a `binary + bagging + boost_from_average` per-tree split-count knife-edge (`binary_bag1_es0_bfa1` tree 0 rust 2 vs cpp 4 leaves) was unmasked once the regression_l1 panic stopped firing first. CONFIRMED pre-existing at HEAD; NOT introduced by Task 2b. Logged as DEF-06-01 (`deferred-items.md`); the matrix asserts structurally-matching trees bit-exact and tolerates <= 1 divergent tree under a hard cap (NOT typed-rejected — no user decision, valid use case).
- **TEETH PROOF (this session):** +1.0 perturbation of `regression_bag1_es0_bfa0_model.txt` leaf 4 → matrix PANICS `ExactMismatch index 3, rust:0.6666666666666665 cpp:1.6666666666666665` (boosting_parity.rs:1450); golden reverted clean.
- **6 new goldens (reg_sqrt + mf2es) STILL NEED a capture run** — the lightgbm==4.6.0 wheel is absent here, so their tests skip-pass by design until captured with the real binary.
- **gate:** `cargo test --workspace` GREEN (boosting_parity 25 passed / 0 failed); clippy clean on edited crates; `LightGBM/` never git-added.

Next: `/gsd-verify-phase 06` — re-verify the three PARTIAL SCs (#1 model text, #3 objectives, #4 ES cadence) now that gaps A–E are closed.

### Plan 06-05 result (BST-03/BST-07/MET-02 — bagging + early stopping + metric infra + D-07 matrix)

Plan 06-05 added the final two axes of the maximal-fidelity matrix and closed Phase 6.

- **bagging (BST-03 / D-13):** `BaggingSampleStrategy` mirrors bagging.hpp/threading.h verbatim over the FND-01-proven `lgbm_core::Random` (per-block `Random(bagging_seed+i)` block 1024, draw every row in order incl. OOB, in-bag left / OOB right + one-buffer reverse; balanced pos/neg). `bagging_by_query=true` rejected with a typed error (EXPLICIT Phase-7 deferral, never silent). The D-13 RNG-replay golden (`bag_indices_seed3_frac0.7.txt`) replays bit-exact (i32). OOB rows STILL scored (Pitfall 4). CRITICAL RNG-reuse fix: `bagging_rands_` built once + advanced across draws → regression(L2) bagging BIT-EXACT vs real lib_lightgbm 4.6.
- **early stopping (BST-07):** `EarlyStopping` (kMinScore init, `factor*score vs best + min_delta`, first_metric_only, trailing-tree pop = round*num_class) + per-iter facade loop with valid-set accumulation; `best_iteration` recorded + the model trimmed to it. Empty-valid-set guarded (typed error).
- **metric infra (MET-02):** metric_freq cadence, multi-metric list, is_provide_training_metric, valid-metric eval history — all wired through builder→Config→Booster.
- **D-07 cross-product matrix:** captured + replayed all ~40 cells (5 objectives × {bag} × {es} × {bfa}) vs real lib_lightgbm 4.6. regression(L2) BIT-EXACT across all 8 cells incl. bagging; single-output es/bfa bit-exact; multiclass within ORACLE_TOL. Documented residuals (manifest, NOT silently dropped): regression_l1 bfa-off uniform-grad split-gain f64-noise; non-L2 bagging cells (non-constant hessian / median-residual renew on subset); multiclass es best_iteration (softmax exp-libm). Phase-7 follow-ups recorded.
- **gate:** `cargo test --workspace` GREEN (boosting_parity 22 passed / 0 ignored — bagging_rng + early_stopping matrix now live; all 06-02..06-04 cells unregressed bit-exact). Capture byte-idempotent; LightGBM/ never git-added.

### Plan 06-04 result (OBJ-01/OBJ-03/MET-01/BST-01 — per-class structural axis complete)

Plan 06-04 made the ONE structural loop change of Phase 6: generalized the proven single-output GBDT loop to grow K=num_class trees/iteration over the class-major score/grad/hess layout (offset=num_data*cur_tree_id), completing all five core objectives end-to-end.

- **multiclass (softmax) + multiclassova:** `MulticlassSoftmax` ports `Common::Softmax` (reused verbatim) with the STRIDED class-major gather `rec[k]=score[num_data*k+i]` (Pattern 4), `factor_=num_class/(num_class-1)`, per-class `grad=p-1|p`/`hess=factor*p*(1-p)`, log-class-prob BoostFromScore, the `LabelOutOfRange` Init guard (Security V5), and class_need_train. `MulticlassOva` reuses the 06-03 Binary objective (num_class instances, is_pos=label==i) at offset=num_data*i. `MultiLogloss` (lgbm-metric) = class-major gather + ObjectiveKind::convert + -log(rec[label]) kEpsilon floor.
- **loop generalized:** `train_one_iter` grows K=objective.num_model_per_iteration() trees/iter; per-class BoostFromAverage on iter 0; `class_need_train==false` (and no-split) pushes `Tree::as_constant(init)` (new lgbm-model constructor, tree.h:232) so models_.len()==iter*K (Pitfall 6). The single-output K=1 path is byte-unchanged (regression/regression_l1/binary/custom regression-tested). Booster wires multiclass/ova dispatch + canonical_objective_string + multi_logloss + class-major predict; builder gained num_class/sigmoid setters.
- **goldens (real lib_lightgbm 4.6, class-major, L1-L5):** multiclass/ova replay BIT-EXACT for L2 per-iter scores + L5 model leaves (15 trees = 5 iters × 3 classes), within ORACLE_TOL for L1 g/h, L3 multi_logloss, L5 predict. Tree count==iters*num_class + class-major stride asserted exactly. boosting_parity: 20 passing / 2 ignored (06-05). cargo test --workspace GREEN (50 binaries, 0 failures). Capture byte-idempotent; LightGBM/ never git-added.
- **DEVIATION (Rule 1) — multiclass goldens capped at 5 iters (documented exp-libm residual):** the redundant-form softmax `exp` (Rust system libm vs the C++ wheel std::exp) differs at ~1 ULP and flips a knife-edge split at iter ~5-6 (iters 0-4 = 15 trees are bit-exact, proving the layout/gather/bfa/class_need_train are correct). Capping the horizon at 5 iters keeps every tree bit-exact rather than weakening the assertion — scores + model leaves stay compare_exact_f64_bits; only the predict-side ConvertOutput/multi_logloss (one exp/row) is within ORACLE_TOL. CLAUDE.md "bit-exact where the algorithm permits" carve-out; single-output spine + binary stay bit-exact for the full 10 iters. Commits: 0ae7eff (objectives+metric), 50dacac (loop+as_constant), b0c2686 (capture+replay).

Next: `/gsd-execute-phase 6 --wave 5` runs 06-05 (bagging + early stopping) — the final Phase-6 axis; un-#[ignore]s `early_stopping` + `bagging_rng` in boosting_parity.

### Phase 5 history (retained)

Progress: [██████████] Phase 5 COMPLETE (05-01..05-09). 05-09 CLOSED the final mfb>0 node-2 leaf-0 2.3e-16 (one f64 ULP) residual BIT-EXACT via a real lib_lightgbm 4.6 CPU-only single-thread FP execution trace (Option B, user-authorized). Ground truth from the trace ([GSD-META] feature 0 most_freq_bin=0 default_bin=0 offset=1): the corpus is SPARSE (rate 0.1667 > kSparseThreshold) so the real BinMapper collapses most_freq_bin_ = default_bin_ = ValueToBin(0) = 0 (bin.cpp:491-499) and runs the offset==1 path — the same as the spine, NOT a most_freq_bin>0/FixHistogram-active path. The harness had mislabeled it most_freq_bin=2/offset=0, which spuriously activated FixHistogram on node-2's direct build (reconstructed a ~1e-15 bin-2 hessian) and polluted the REVERSE scan by 2 ULPs. Fix: (1) corrected the mfb_pos_real corpus to the ground-truth most_freq_bin=0/offset=1; (2) added LeafSplits::init_from_split and seeded child leaves DIRECTLY from the parent SplitInfo (serial_tree_learner.cpp:851-871 — carrying best_sum_left_hessian - kEpsilon, feature_histogram.hpp:1042), not a re-fold. learner_parity_mfb_pos_real_binary un-#[ignore]d + PASSING bit-exact (12 passed / 0 ignored); kernel_parity 4/4; spine_real + growth_path_subtract unregressed; cargo test --workspace GREEN. assert_real_tree_parity byte-unchanged (no tolerance); LightGBM/ never git-added. Commit 2ced5a2. TRL-01/TRL-05 closed bit-exact vs the real binary. ---- (prior) Phase 5 plans 05-01..05-08 COMPLETE — WR-01/WR-02 CLOSED by 05-07. The subtraction trick (`larger = parent − smaller` via `Backend::subtract_histograms`) + HistogramPool slot reuse are now wired into the LIVE `find_best_splits` growth path (mirroring C++ serial_tree_learner.cpp:364-378); the dead `let _ = subtract_from;` discard + orphaned `_pool` are gone; `learner_parity_growth_path_subtract` proves derived-larger-child == direct build cell-for-cell AND the spine stays bit-exact. The mfb>0 node-2 leaf-0 2.3e-16 ULP did NOT close — its 05-08 subtraction-trick attribution is DISPROVEN (leaf 0 is the directly-built smaller child, untouched by subtraction); root cause CORRECTED to a 2-ULP f64 accumulation-order subtlety in the FixHistogram-active DIRECT histogram build, deferred to NEW plan 05-09 (FixHistogram fold-order parity). The mfb gate stays #[ignore]d with a corrected honest reason; NO assertion weakened (~4 orders inside ≤1e-12). cargo test --workspace GREEN (learner_parity 11 passed / 1 ignored; kernel_parity 4/4); routing self-consistency holds. TRL-01/TRL-02/TRL-05 satisfied for the wired path. ---- (prior) Phase 5 plans 05-01..05-06 + 05-08 — BLOCKER CR-03 CLOSED by 05-08. The Rust serial learner now grows trees BIT-EXACT to the real lib_lightgbm 4.6 spine golden (spine_real.txt) and STRUCTURALLY bit-exact to the mfb>0 golden (mfb_pos_real.txt) on every field (split_feature, threshold incl. the zero sentinel 1.0000000180025095e-35, decision_type=2 2 2, child topology, leaf_count with no 0-row leaf, internal_count) + 3/4 leaf values. The fix set (commit c564036): Fix B (PRIMARY) child LeafSplits direct pass-through (was swapping smaller/larger slots, feeding a child its sibling's sums → -17.99 vs 0.55); Fix A missing_type==None FORWARD-dispatch gate (decision_type 0→2); Fix C MaybeRoundToZero signed-zero normalize; Fix D bin-0 kZeroThreshold mapping. spine_real gate un-#[ignore]d + passing in the default suite; mfb_pos gate stays #[ignore]d (assertions UNCHANGED) with a narrowed reason — the ONLY residual is the node-2 default-bin leaf-0 value (Rust 0.59999999999999976 vs golden 0.59999999999999953, Δ 2.3e-16 = one f64 ULP), a kEpsilon cascade DEFERRED to 05-07's not-yet-wired subtraction-trick/HistogramPool (~4 orders of magnitude inside the ≤1e-12 contract; NO assertion weakened). Routing self-consistency (CR-01) still holds; kernel_parity stays 4/4. TRL-05/TRL-07/TRL-01/TRL-09 satisfied bit-exact on the spine vs the real binary. 05-07 (wave 8) is now UNBLOCKED — its subtraction-trick wiring also closes the mfb>0 leaf-0 ULP and un-#[ignore]s learner_parity_mfb_pos_real_binary.

### Plan 05-07 result (WR-01/WR-02 — subtraction-trick + HistogramPool wiring CLOSED)

Plan 05-07 closed WR-01/WR-02 by wiring the dead subtraction-trick + HistogramPool orchestration into the live growth path (commit 037e011), and re-attributed the surviving mfb>0 leaf-0 ULP:

- **WR-02 CLOSED — subtraction trick live:** `find_best_splits` now builds only the SMALLER child directly via `construct_histograms` and DERIVES the LARGER child as `parent − smaller` via `Backend::subtract_histograms` (learner.rs:736), mirroring C++ serial_tree_learner.cpp:364-378. The dead `let _ = subtract_from;` discard is removed (`subtract_from` now selects the parent slot); the root still builds directly.
- **WR-01 CLOSED — HistogramPool live:** `_pool` renamed to `pool` and READ — the parent leaf's retained histogram is fetched from its pool slot, children occupy pool slots (smaller built, larger derived); D-05 pool/eviction reused unchanged.
- **`learner_parity_growth_path_subtract` gate:** asserts the derived larger-child histogram == an independent direct build cell-for-cell (bit-exact f64) AND the grown spine tree still matches the real lib_lightgbm 4.6 golden — proving the wiring changed the DERIVATION path only, not the output.
- **mfb>0 leaf-0 2.3e-16 ULP RE-ATTRIBUTED (05-08 framing DISPROVEN):** the ULP did NOT close with the wiring. Leaf 0 is node-2's **directly-built smaller child**, so the subtraction trick NEVER touches it — the 05-08 attribution (kEpsilon cascade from the not-yet-wired subtraction-trick/HistogramPool) is disproven. Root cause CORRECTED to a **2-ULP f64 accumulation-order subtlety in the FixHistogram-active DIRECT histogram build** (construct/FixHistogram/output fold). The prior executor trialed the C++-faithful SplitInfo-sum LeafSplits seeding (serial_tree_learner.cpp:850-870) — it did NOT change leaf 0 and REGRESSED leaf 3 — so 05-09 must target the construct/FixHistogram fold order, NOT LeafSplits seeding. Deferred to NEW plan 05-09. mfb gate stays #[ignore]d with a corrected honest reason; NO assertion weakened, tolerance-wrapped, or deleted.
- `cargo test --workspace` GREEN (learner_parity 11 passed / 1 ignored; kernel_parity 4/4). Routing self-consistency (CR-01) holds. `LightGBM/` never git-added. Commits: 037e011 (Task 1+2 wiring + growth_path_subtract gate), fbd4f1d (mfb gate reason correction → 05-09).

Next: `/gsd-plan-phase 5` authors NEW plan 05-09 (FixHistogram f64 fold-order parity) — align the Rust FixHistogram-active direct-build f64 accumulation/fold order with C++ `ConstructHistograms`/`FixHistogram`, then un-#[ignore] `learner_parity_mfb_pos_real_binary` and assert bit-exact (TRL-01, TRL-05). 05-09 has no PLAN.md yet.

### Plan 05-08 result (CR-03 learner-fix — CR-03 CLOSED)

Plan 05-08 closed BLOCKER CR-03 (the 05-06 real-oracle falsification). The Rust SerialTreeLearner now reproduces the real lib_lightgbm 4.6 goldens bit-exact on the spine and structurally bit-exact on mfb>0:

- **PRIMARY root cause (Fix B) — child LeafSplits slot mapping:** `find_best_splits` mapped smaller/larger LeafSplits by `smaller_leaf == left_leaf`, SWAPPING the slots whenever the smaller child was the right leaf (incl. the equal-count tie) and feeding a child its sibling's sums (leaf 1 got leaf 0's −24 → spine leaf_value −17.99, wrong child splits/topology). Fixed to a DIRECT pass-through (smaller_leaf_splits always holds smaller_leaf), mirroring C++ smaller_leaf_splits_/larger_leaf_splits_ (serial_tree_learner.cpp:851). This makes the spine fully bit-exact. Task-1 had hypothesized the FORWARD gate as the single root cause; the slot swap was the dominant defect.
- **Fix A — FORWARD-branch dispatch gate:** `run_forward` threaded learner.rs → Backend::find_best_split → find_best_split_cpu; FORWARD runs only for num_bin>2 && missing_type==Zero (feature_histogram.hpp:420-429). Fixes mfb decision_type[0] 0→2.
- **Fix C — MaybeRoundToZero signed-zero normalize** in the shrinkage finalize (tree.h:191,255-260): −0.0 → +0. Fixes mfb leaf 1 −0 → 0.
- **Fix D — bin-0 kZeroThreshold mapping:** real_upper_bounds_mfb maps bin 0 → (1e-35f32 as f64) == 1.0000000180025095e-35 (NOT the f64 literal). Fixes the mfb node-2 zero-sentinel threshold.
- **Gates:** spine_real un-#[ignore]d and PASSES in the default `cargo test --workspace`; mfb_pos stays #[ignore]d with a NARROWED, honest reason (residual sub-ULP leaf-0 value → 05-07), assertions UNTOUCHED (T-05-08-02 upheld).
- `cargo test --workspace` GREEN (learner_parity 10 passed / 1 ignored; kernel_parity 4/4). Routing self-consistency (CR-01) holds. `LightGBM/` never git-added. Commits: 061d791 (Task 1 localization), e582cf2 (Task-2 findings), c564036 (Task 2 fix + Task 3 gate adjustment).

Next: `/gsd-execute-phase 5 --wave 8` runs 05-07 (wire subtraction-trick + HistogramPool into the live growth path) — which ALSO closes the mfb>0 leaf-0 2.3e-16 ULP and un-#[ignore]s learner_parity_mfb_pos_real_binary.

### Plan 05-06 result (gap closure — CR-02 closed; BLOCKER CR-03 raised)

Plan 05-06 replaced the self-transcription learner oracle with a REAL pip-installed lib_lightgbm 4.6 binary oracle (D-08) — and the real oracle falsified the port:

- **Real-binary oracle (CR-02 CLOSED):** `learner-oracle-capture` xtask + `xtask/py/learner_oracle_capture.py` train the spine + a NEW most_freq_bin>0 corpus on real `lightgbm==4.6.0` (`deterministic=true force_row_wise=true num_threads=1`, fixed seed), version-assert the wheel, and dump `%.17g` model text to committed `spine_real.txt` / `mfb_pos_real.txt`. Identity-binning pin (consecutive-integer raw values + assert-or-abort on realized bin count/most_freq_bin) rules out a bin-layout artifact. Byte-idempotent; routine `cargo test` needs no toolchain; `LightGBM/` never git-added.
- **Port FALSIFIED → CR-03 (new BLOCKER):** `learner_parity_spine_real_binary` + `learner_parity_mfb_pos_real_binary` compare the Rust-grown tree to the real goldens bit-exact and FAIL. The divergence is the learner's offset/compaction scan + leaf-output + default-bin (zero-sentinel) + `decision_type` path — distinct from CR-01 (routing self-consistency, closed 05-05) and CR-02 (oracle existence, closed here). The CR-01 routing test only proved internal consistency; it could not see a wrong tree because it never compared to a real oracle — exactly the self-validation gap D-08 set out to expose.
- **Gates preserved, not weakened:** both real-binary tests are `#[ignore]`d with the divergence specifics in their ignore reason + doc comment; run `cargo test -p oracle-harness --test learner_parity -- --ignored` to see the failures. Do NOT delete or relax them.
- `cargo test -p oracle-harness --test learner_parity` → 9 passed, 2 ignored. Commits: ee46f4c (Task 1), 6d11d35 (Task 2 — human-gated capture), e0c1312 (Task 3 — falsifies CR-03).

Next: this phase is BLOCKED on CR-03. Author a NEW learner-fix plan (e.g. 05-08) via `/gsd-plan-phase 5` (or `/gsd-discuss-phase 5` first) to make the serial learner reproduce the real goldens bit-exact (fix the spine + mfb>0 divergences above), then un-`#[ignore]` the two gates. Only AFTER CR-03 closes can `/gsd-execute-phase 5 --wave 6` run 05-07 (its re-validation gate then has a chance to pass).

### Plan 05-05 result (gap closure — CR-01 BLOCKER)

Plan 05-05 closed CR-01 by adopting the real-LightGBM offset==1 + compacted convention end-to-end (D-09, supersedes D-01):

- **One offset rule (Task 1):** `lgbm_treelearner::offset_for_most_freq_bin(most_freq_bin) -> i32` (1 if 0, else 0) is THE authoritative source; every corpus/feature offset (spine f0/f1, col_sampler `make`, missing_routing, real_gh parser) routes through it. The three contradictory inlined rules are deleted (grep gate: 0 inlined offset literals).
- **Compacted offset==1 histogram (Task 2):** `compact_histogram` shifts the stride-2 histogram into the C++ `data_` compacted layout (cell c = real bin c+offset, bin-0 dropped, tail zeroed to 2*num_bin) before find_best_split + the host per-bin re-scan (feature_histogram.hpp:619/943/950). No-op for offset==0; kernel_parity 4/4 stays bit-exact.
- **CR-01 root cause = single-feature-group min_bin (Rule-1 fix, surfaced by Task 3):** compaction alone did NOT fix the partition. C++ `FeatureGroup::Split(num_feature_==1)` dispatches to `DenseBin::Split(max_bin,...)` which HARD-CODES `min_bin=1` (dense_bin.hpp:423-433). Passing `partition_min_bin = f.min_bin + f.offset` collapses the verbatim `--th` to `th=threshold` for mfb==0, so partition `bin>threshold -> right` == predict `bin<=threshold -> left`. Raw `min_bin=0` left `th=threshold-1` (the [4,8] partition vs [6,6] predict off-by-one).
- **Oracle-INDEPENDENT CR-01 gate (Task 3):** `learner_parity_routing_self_consistency` routes every training row through the grown tree's `get_leaf`, tallies per leaf, asserts == `tree.leaf_count` (stored data_partition count) EXACTLY for spine/col_wise/col_sampler/real_gh. FAILED pre-fix (tally [6,2,0,4] vs stored [4,2,0,6]); PASSES after.
- **Stale-golden supersession (Rule-3):** the pre-D-09 `learner_capture.cpp` full-tree/per-bin goldens share the buggy convention (CR-02), so `spine_full_tree`/`spine_per_bin_gains`/`transcription_crosscheck`/`real_gh_full_tree` + the golden half of `row_vs_col` skip with a 05-06 re-point note; `row==col` equality (TRL-09, convention-independent) + the routing test stay live. Golden parsers kept (#[allow(dead_code)]) for 05-06 reuse.
- `cargo test --workspace` green (9 learner_parity + 32 lgbm-treelearner lib); kernel_parity 4/4 bit-exact. Never git-adds `LightGBM/`. Commits: bc3dd95 (Task 1), 2e996e0 (Task 2), ee0255f (Task 3).

Next: `/gsd-execute-phase 5` plan 05-06 — replace the self-transcription learner oracle with a REAL lib_lightgbm 4.6 binary oracle (spine_real.txt / mfb_pos_real.txt), re-point the now-skipped full-tree/per-bin tests, and give the offset==1 path its first real bit-exact coverage (CR-02).

### Plan 05-04 result

Plan 05-04 layered the two parity additions + the captured-g/h corpus onto the proven spine:

- **force_col_wise (TRL-09, Open Q2 RESOLVED):** `BuildStrategy::{RowWise,ColWise}` is a config FLAG (a no-op) over the shared `construct_histograms` Backend op on the single-thread deterministic anchor — NOT a distinct compute path (A1 confirmed). `col_wise.txt` carries the identical PTREE as `spine.txt`; `learner_parity_row_vs_col` grows the corpus under BOTH strategies and asserts `to_string()` equality to each other and to C++.
- **ColSampler RNG parity (TRL-08):** `col_sampler.rs` reproduces the C++ `Random::Sample` draw SEQUENCE (ResetByTree once per tree, GetByNode per node smaller-leaf-then-larger). `col_sampler.txt` (drawn from the genuine reference Random at `feature_fraction=1.0` / `feature_fraction_bynode=0.5`) asserts the exact selected REAL-feature indices per draw via `train_with_col_sampler_trace`.
- **Captured real iter-1 g/h (D-03):** `real_gh.txt` — regression-l2 (`grad=score-label`, `hess=1`) + binary-logloss (`boost_from_average=false`, `score_t=float`) over fixed real labels; `learner_parity_real_gh_full_tree` proves the learner grows the same tree as C++ under a realistic gradient distribution (D-07).
- **1 Rule-1 fix:** the tree's `leaf_count`/`internal_count` record the ACTUAL `data_partition` leaf_count (`serial_tree_learner.cpp:788-791`, `update_cnt=true`), not the SplitInfo `round_int(hess·cnt_factor)` reconstructed counts (which disagree by ±1 for fractional hessians) — corrected the spine counts to the faithful actual-partition values. Applied in both Rust `split_inner` and the C++ transcription.
- `cargo test --workspace` green (8 learner_parity tests + 29 lgbm-treelearner). Byte-idempotent; never git-adds `LightGBM/`. Commits: e8efc90 (Task 1, pre-committed), d06d85b (Task 2).

Next: `/gsd-execute-phase 5` is complete (4/4 plans); proceed to phase 05 review/verification or phase 06 (GBDT).

### Plan 04-04 result

Plan 04-04 brought up the cubecl-hip (ROCm) backend on the local gfx1100 GPU (best-effort, D-03a):

- **rocm runtime + capability-gated f32 accumulate (CMP-03/CMP-04):** `runtime.rs` `AccumulateType` + `Capabilities::accumulate_type()` routes f64 (cpu anchor) vs f32 (no-f64 hip). f32-cell MIRROR kernels + generic-over-`Runtime` launchers (`construct_histograms_f32_on`, `subtract_histograms_f32_on`, `find_best_split_raw_f32_on`); `data_partition_on<R>` is f64-free and shared. f32 gain primitives in `gain.rs`. The cpu f64 anchor kernels are UNTOUCHED (04-01..03 bit-exact gates still pass).
- **Separate ~1e-6 hip gate (ORA-04 rocm half):** `rocm_smoke.rs` asserts the gfx1100 matrix (Plane YES/f64 NO/atomic YES/plane_size 32); `kernel_parity.rs` hip layer compares hip f32 vs cpu f64 anchor (collected to `Vec<f32>`) via `compare_within(ORACLE_TOL)`, two-tier so the documented gap is surfaced (no silent pass) but not a blocker.
- **Real-hardware run (ROCm 7.1.1 / HIP 7.1.52802 / cubecl-hip 0.10.0):** smoke 2/2; partition bit-exact, subtract ≤1.16e-10, histogram/split within f32 ULP (max relative ≈ 1.1e-7). `04-ROCM-GAPS.md` records every per-case max abs-diff + cause (G-04-01 histogram f32 accumulation, G-04-02 split f32 gain; Phase-5+ remediation options noted). 1 Rule-3 deviation: two-tier hip gate so the anticipated f32 gap is not a hard blocker (D-03a). Commits: 6b97786, 9adedf8.

Next: `/gsd-execute-phase 5` (tree learner) — full Backend op set available on cpu (bit-exact) + hip (best-effort).

### Plan 04-03 result

Plan 04-03 closed the full compute-backend kernel suite on the cpu anchor and resolved RESEARCH open question A3:

- **`Backend::find_best_split` (D-01a — gain math IN-kernel):** `gain.rs` holds `ThresholdL1`/`get_leaf_gain`/`get_split_gains`/`calculate_splitted_leaf_output` as `#[cube]` fns (verbatim `feature_histogram.hpp:711-845`) + `GainConfig` + `SplitInfo`; `kernels/split.rs` transcribes `FindBestThresholdSequentially` (`:830-1057`) for BOTH the REVERSE (`t-1+offset`, `sum_right_hessian=kEpsilon` seed) and FORWARD (`t+offset`, `sum_left_hessian=kEpsilon` seed) branches, with the `2*kEpsilon` entry bump, SKIP_DEFAULT_BIN continue, exact gate order, and kEpsilon subtracted back at finalize.
- **`Backend::data_partition`** (stable SplitInner `MissingType::None` routing → reordered index array + split_point; Phase-5 owns leaf_begin_/leaf_count_) **and `Backend::subtract_histograms`** (FeatureHistogram::Subtract math, A3 resolved in-scope at the kernel layer).
- **Layered goldens + parity (ORA-04 cpu):** `kernel_capture.cpp` extended → `split.txt` (per-candidate gains REVERSE+FORWARD + winner; reverse/forward/default-bin-skip/L1/no-split), `partition.txt`, `subtract.txt`; `kernel_parity.rs` replays all three bit-exact (split per-candidate gains via `get_split_gains` + winner via the kernel; partition via `compare_exact_u32`; subtract via `compare_exact_f64_bits`). Byte-idempotent.
- 1 Rule-1 bug fixed: `gain.rs::threshold_l1`'s `if/else` sign form mis-lowered to a constant on cubecl-cpu, ZEROING every L1 gain (kernel returned no-split for any `lambda_l1>0`); fixed with branchless `select`, proven by the L1 split parity case. `cargo test --workspace` green. Commits: ff24113, d5061b9, 7f8a5fc.

Key decision: complex sequential cube kernels on cubecl-cpu (0.10.0) require LITERAL-initialized loop-carried mutables (never from a scalar kernel arg) + branchless `select` for all conditional in-loop stores — the C++ gate ORDER and arithmetic are preserved 1:1, only the control-flow ENCODING changes. The gain sentinel is the literal `0.0` (valid gains are strictly positive; "no split" is signaled by `is_splittable==0`).

Next: `/gsd-execute-phase 4` plan 04-04 (ROCm Backend impl + ~1e-6 parity variant against the same four committed goldens).

### Plan 04-02 result

Plan 04-02 delivered the FIRST complete compute-backend vertical slice — a real `construct_histograms` whole-kernel op wired end-to-end and proven bit-exact vs C++:

- **`Backend::construct_histograms` (D-01, CMP-05 histogram layer):** coarse whole-kernel op `(client, binned, ordered_gradients, ordered_hessians, num_bin) -> Result<Vec<f64>, ComputeError>` returning stride-2 `[g,h]` f64 cells; concrete `CpuBackend` binds `ActiveRuntime` and dispatches to the 04-01 bit-exact ordered fold. V5 validation + f64 accumulation retained.
- **`xtask kernel-capture` + `kernel_capture.cpp` (D-02):** header-only VERBATIM transcription of `ConstructHistogram` (`dense_bin.hpp:130-141` / `sparse_bin.hpp:138-152`) reusing the DenseBin/SparseBin forms; emits an 18-case `histogram.txt` golden (D-02a: dense+sparse, default-bin routing, u8/u16/u32 widths, grad/hess spread). No external_libs, byte-idempotent, `KERNEL_MASTER_SEED=0x4157F00D`.
- **`kernel_parity.rs` (ORA-04 cpu hard gate):** drives `CpuBackend::construct_histograms` over every golden case and asserts BIT-EXACT f64 cells via the full-path `oracle_harness::comparator::compare_exact_f64_bits`. Green; SKIPs cleanly pre-capture.
- **CMP-01 intact:** cubecl is an oracle-harness DEV-dep only; the library crate stays cubecl-free (containment guard green).
- 1 Rule-1 bug fixed in newly-authored capture code (stray `num_rows_init` call), corrected before its task commit. `cargo test --workspace` green. Commits: 504e7ff, 80f8fc9, cf3f380.

Key decision: the golden HIST is the DENSE ordered f64 fold over the round-tripped `Bin::data(idx)` (== what the cubecl-cpu kernel computes). Dense/sparse `ConstructHistogram` are identical for non-zero bins (differ only at bin 0, which sparse never folds), so sparse cases use bins in `[1, num_bin)` — exercising the SparseBin store path while keeping the golden replayable by the dense-fold kernel.

Next: `/gsd-execute-phase 4` plan 04-03 (find_best_split + data_partition kernels + goldens, reusing this kernel-capture/parity machinery).

### Plan 04-01 result

Plan 04-01 built the Phase-4 compute foundation and settled the load-bearing D-04a determinism bet BEFORE the kernel suite:

- **D-04a SETTLED (keystone):** the cubecl-cpu single-owner ordered f64 fold (`CubeDim::new_1d(1)`, transcribing `dense_bin.hpp:120-135`) is byte-identical across 25 launches AND bit-exact (`to_bits()`) vs a hand-computed C++-order sequential f64 fold. The bit-exact anchor assumption holds — no ~1e-6 fallback, no separate scalar reference. 04-02/04-03 may build on it.
- **ComputeError** thiserror boundary (BinIndexOutOfRange/LengthMismatch/CapabilityUnavailable/Runtime, V5/T-04-01); `lgbm-compute` now has cpu-default / rocm-opt-in features.
- **Capability gate (CMP-04)** reports the verified asymmetric cpu matrix (plane=false, f64=true, atomic=false, size=1 → Sequential).
- **CMP-01 containment guard** green (no upper crate names cubecl).
- 1 Rule-1 bug fixed: `client.empty()` returns recycled uninitialized pool memory → zero-init `out` (the spike caught it). `cargo test --workspace` green. Commits: 0021114, ef69c91, c9669f6.

Next: `/gsd-execute-phase 4` plan 04-02 (full histogram kernel + C++-transcription golden capture).

### Resume

Plan 03-04 closed PRD-06 (sub-range prediction) and completed the Phase-3 prediction surface:

- **PRD-06 CLOSED:** `init_predict` (C++ `gbdt.h:426-435`) is parity-asserted with the full `<behavior>` clamp/slice battery — `-1==all` (and `0==all`), bounded count, non-zero start, over-range/extreme clamp to empty slice (T-03-12: `i32::MAX`/`i32::MIN`/negative never panic or index OOB), and slice accumulation proven to sum ONLY the selected iterations (differs from full range). Added `predict_raw_{mat,csr,csc}_range` threading `start_iteration`/`num_iteration` through the batch driver; the full-range `predict_raw_{mat,csr,csc}` now delegate with `(0,-1)` so 03-02/03-03 callers stay green.
- **D-06 layer 5 GREEN:** `predict_subrange.rs` replays the committed `subrange.txt` golden's four slices `(0,10)`/`(0,5)`/`(5,-1)`/`(1,1)` (covering `-1==all`, bounded count, non-zero start) via `compare_within(ORACLE_TOL)` over all 7000 rows — DENSE form (slice math orthogonal to input form; CSR/CSC raw covered in 03-02).
- **Capture untouched** — the 03-01 subrange golden already recorded the needed pairs; byte-idempotency preserved (no fixture re-emission).

`cargo test --workspace` fully green (0 failed); full Phase-3 layered parity battery (layers 1-5) passes. Commits: bb29ec5 (Task 1), 642f5a5 (Task 2).

Phase 3 verification PASSED (03-VERIFICATION.md): all 4 success criteria + 6 requirement IDs satisfied; negative-control confirmed parity assertions are load-bearing. Code review (03-REVIEW.md) raised 3 Critical divergences from C++ source that are out-of-corpus latent (CR-01 feature-importance `split_gain>0` guard, CR-02 leaf-index sub-range, CR-03 RF `average_output` apply) — none affect a Phase-3 criterion; recorded as Phase 7 / RF (BST-06) follow-ups.

Next: `/gsd-plan-phase 4` — Compute Backend (CPU-first integer histograms → ROCm).

---

#### Prior phase-2 resume (retained for context)

Plan 02-07 closed the CR-01 blocker (default-ingest Construct divergence) + its masking + WR-01:

- **CR-01 (BLOCKER) CLOSED:** the non-bundled `Dataset::construct` now mirrors C++ `Dataset::Construct` trivial-feature filtering (`used_features = !is_trivial_` only; `used_feature_map_[real] = -1` for trivial features; groups over used features only). The default ingest path (`finish_from_columns`) routes through `construct_bundled` (the faithful single-Construct port) with an `EfbSamples` built to the exact c_api.cpp:1352-1374 sampled-set convention (positions 0..sample_cnt, non-zero/NaN filter, total_sample_cnt=sample_cnt; reusing the single create_sample_indices draw — NO second RNG draw). A trivial feature is now DROPPED and bundling runs, mirroring the single C++ Dataset::Construct (dataset.cpp:325-441). CSR/CSC inherit via finish_from_columns.
- **EFB parity hole CLOSED:** `default_config_ingest_parity.rs` now asserts per-non-trivial `feature_to_group`/`feature_to_subfeature` vs the C++-Construct golden, so an incorrectly-built EfbSamples fails loudly.
- **CR-01 masking CLOSED:** the golden emitter models C++ Construct (trivial f2 dropped — no ASSIGN/no group; per-non-trivial group/subfeature via the in-file FastFeatureBundling transcription, is_sparse=true); golden regenerated via real capture.
- **WR-01 hardened:** missing committed golden -> panic, never a silent SKIP.

HARD fails-before/passes-after recorded (CR-01 `ds.num_features 4 != non-trivial golden feature count 3` before; passes after). `cargo test --workspace` fully green (0 failed); ingest_equivalence + example_dataset_parity unaffected (no trivial features in their fixtures); bin-capture idempotent (only default_config_ingest.txt changed; example COUNTS line unchanged).

Next: `/gsd-verify-phase 02` (re-run phase verification — expect DAT-07, ORA-03, DAT-01/02/05 to flip to PASS/SATISFIED).

Verified PASS (prior): SC#2 (ingest + immutable store), SC#3 (missing/categorical routing), SC#4 (EFB grouping). RE-AFFIRMED 02-06: SC#1 (is_trivial_ on default path), SC#5 (per-stage parity covers default path). RE-AFFIRMED 02-07: SC#2 (store bit-identical to C++ Construct, trivial dropped, grouping verified), SC#5/ORA-03 (per-stage parity localizes the trivial/enable_bundle/EFB-grouping divergence). DAT-07 + ORA-03 BLOCKED -> SATISFIED; DAT-01/DAT-02/DAT-05 PARTIAL/path-isolated -> satisfied (pending formal re-verify).

- EFB (Plan 02-05): `MultiValBin` dense/sparse storage (+1 push), `efb.rs` (`fast_feature_bundling`/`find_groups`/`get_conflict_count`/`fix_sample_indices` — ALL randomness via `lgbm_core::Random::new(num_data)`, STABLE sorts, element-wise parallel-vector swap), and the `Dataset::construct_bundled` `enable_bundle` dispatch with real<->packed feature maps (`used_feature_map_`/`real_feature_idx_`). EFB layer-3 golden (feature->group membership + per-group `bin_offsets_`/`num_total_bin_`/`group_is_multi_val` + per-row bundled indices) bit-identical to C++ on the D-06 #4 mutually-exclusive sparse corpus + a no-bundle control. EFB capture = HEADER-ONLY verbatim transcription of `dataset.cpp` (external_libs unvendored → both nominal capture paths infeasible; human-approved). bin-capture idempotent.

- Ingestion (Plan 02-04): `from_mat`/`from_csr`/`from_csc` (validated entry points, Security V5 — typed `DatasetError` never a panic) wire sample → `find_bin` → `Dataset::construct` → `push` → `finish_load`; sampling routes through the Phase-1 RNG (`create_sample_indices`), f32→f64 widening at one `widen()` site. Dense vs CSR vs CSC bin bit-identically (incl. a zero-heavy column). `Metadata` (f32 label/weights/query_weights, f64 init_score, i32 query_boundaries) + `finish_load` query-weight derivation (f32 `CalculateQueryWeights`) round-trips bit-exact. End-to-end parity: regression + binary_classification example datasets (28 features each, 500 rows) bin bit-identical to C++ for every feature (layers 1+2). Example fixtures COPIED into committed `tests/fixtures/examples/`.

- `lgbm-dataset` crate exists with `BinMapper` (numeric `find_bin`/`value_to_bin`, bit-exact f64 boundary kernel), `BinType`/`MissingType`, and `DatasetError`.
- Categorical (Plan 02-03): `find_bin_categorical` (descending-count fold via stable `SortForPair`, f32 0.99 cut, `min_data_in_bin` fold-break, NaN dummy bin 0) + `categorical_2_bin_`/`bin_2_categorical_` + categorical `value_to_bin`; completed `MissingType` routing (None/Zero/NaN, signed zeros, all-missing) — 6-case categorical (layers 1+3 + per-row) + 8-case missing (layer 1 + per-row) golden replay, bit-identical, idempotent.
- Storage layer (Plan 02-02): `Bin` trait + `BinValue` (u8/u16/u32) + `create_dense_bin`/`create_sparse_bin` factories (Box<dyn Bin>, D-01); `DenseBin<T, IS_4BIT>` incl. 4-bit packing (D-02); `SparseBin<T>` delta-encode + fast-index; `FeatureGroup` offset packing (u64) + `PushData`; `Dataset::construct` + `finish_load` type-state immutability (→ `FinishedDataset`).
- `bin-capture` xtask subcommand + `xtask/cpp/bin_capture.cpp` emit numeric goldens (layers 1+2) AND storage-layout goldens (DenseBin/SparseBin/FeatureGroup bytes); oracle-harness has `compare_exact_u32`/`compare_exact_f64_bits`/`compare_exact_bytes`; later plans plug in here.
- 45-case numeric golden replay (bit-identical, SC#1/SC#5) + 6-case storage golden replay (byte-identical incl. 4-bit + sparse, SC#2); both regen idempotent.
- `lgbm_core::Config` + `Config::from_params` are the config bag for all later crates (Phase 1).

## Performance Metrics

**Velocity:**

- Total plans completed: 22 (tracked)
- Average duration: ~3 min
- Total execution time: <1 hour

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 01-oracle-contract-foundations | 3/3 | ~2 sessions | ~1 session |
| 02-dataset-binning-determinism-root | 5/5 | ~65 min + continuation | ~16 min |
| 02 | 7 | - | - |
| 03 | 4 | - | - |
| 04 | 4 | - | - |
| 06 | 6 | - | - |

**Plan 01-02:** 3 tasks, 11 files (9 created + 2 modified), 29 new tests; `cargo test --workspace` green.
**Plan 01-03:** 2 TDD tasks, 3 files modified, 7 new tests (49 → 56); deterministic alias resolution + empty==absent reads; `cargo test --workspace` green.
**Plan 02-01:** 4 tasks (~12 min), 14 files (9 created + 5 modified), numeric BinMapper kernel + bin-capture harness + 45-case golden replay (layers 1+2 bit-exact); `cargo test --workspace` green.
**Plan 02-02:** 3 tasks (~9 min), 11 files (7 created + 4 modified), bin-storage layer (Bin trait + DenseBin/4-bit + SparseBin + FeatureGroup offsets + Dataset finish_load type-state immutability) + 6-case storage golden replay (byte-exact); 25 new lib tests (10→35); `cargo test --workspace` green; bin-capture idempotent.
**Plan 02-03:** 2 tasks (~9 min), 9 files (4 created + 5 modified), categorical `find_bin_categorical` (stable descending-count fold + f32 0.99 cut + fold-break + NaN dummy bin) + categorical `value_to_bin` + completed MissingType routing; 6-case categorical (layers 1+3 + per-row) + 8-case missing (layer 1 + per-row) golden replay (bit-exact); 7 new inline + 2 integration tests (35→42 lib); `cargo test --workspace` green; bin-capture idempotent.
**Plan 02-04:** 3 tasks (~35 min), 12 files (9 created + 3 modified), `Metadata` (f32 query-weight derivation) + `from_mat`/`from_csr`/`from_csc` ingestion (validated entries, single widen site, Phase-1-RNG sampling); dense/CSR/CSC bit-identical (zero-heavy column) + metadata golden + end-to-end example-dataset parity (regression + binary, 28 features each, layers 1+2 bit-exact); 14 new inline + 3 integration test files (42→56 lib); `cargo test --workspace` green; bin-capture idempotent.
**Plan 02-05:** 4 tasks (Tasks 1-2 prior agent + Tasks 3-4 continuation), 8 files (2 created + ~6 modified), `MultiValBin` dense/sparse + `efb.rs` (FastFeatureBundling/FindGroups/GetConflictCount/FixSampleIndices, Phase-1 RNG + stable sorts) + `construct_bundled` enable_bundle dispatch with real<->packed feature maps; EFB layer-3 golden (3 cases: 2 mutually-exclusive sparse bundling + no-bundle control) bit-exact (membership + bin_offsets_ + num_total_bin_ + per-row bundled indices); EFB capture = header-only verbatim transcription of dataset.cpp; 1 Rule-1 bug fixed (real-vs-packed indexing); `cargo test --workspace` green; bin-capture idempotent.
**Plan 02-06 (gap closure):** 3 tasks (~18 min), 6 files (2 created + 4 modified), `scaled_filter_cnt` single-source-of-truth helper (exact C++ integer-truncation, num_rows==0 Rust-only guard) routed from both `ingest.rs::build_mapper` and `bin_mapper.rs::find_bin_from_column`; CSR/CSC inherit via finish_from_columns. New default-config (feature_pre_filter=true, sample_cnt=50<num_rows=200, min_data_in_leaf=20 -> filter_cnt=5) ingest parity golden + test: fails-before (`feature 1: is_trivial_ true != golden false`) / passes-after; engineered feature f1 flips is_trivial_ between filter_cnt=5 and raw 20. GAP-1 (CR-01/IN-02, SC#1, DAT-01) + GAP-2 (CR-02, SC#5/ORA-03, DAT-07) closed. New golden dispatched from fixed argv[8] before the variadic example tail (kFirstExampleArgv 9->10) so no existing golden mutated. 1 new unit test + 1 integration test (75 lib); `cargo test --workspace` green; bin-capture idempotent.
**Plan 02-07 (CR-01 + WR-01 closure):** 3 tasks (~35 min), 5 files modified, default ingest unified onto the faithful single C++ Dataset::Construct: `Dataset::construct` filters trivial features (used_features=!is_trivial_; used_feature_map_[real]=-1; groups over used only); `finish_from_columns` dispatches through `construct_bundled` with an EfbSamples built to the exact c_api.cpp:1352-1374 sampled-set convention (positions 0..sample_cnt, non-zero/NaN filter, total_sample_cnt=sample_cnt; reuses the single create_sample_indices draw — no second RNG). Golden emitter models C++ Construct (trivial f2 dropped, per-non-trivial group/subfeature via in-file FastFeatureBundling transcription, is_sparse=true default); golden regenerated via real capture. Parity test: trivial-exclusion (feature_to_group==-1) + per-non-trivial group/subfeature parity vs C++-Construct golden + bit-exact stored bins; WR-01 panic on missing golden; HARD fails-before (`ds.num_features 4 != non-trivial golden feature count 3`)/passes-after. 2 Rule-1 bugs fixed (emitter is_sparse=false->true grouping divergence; num_features assertion vs used count). CR-01 (BLOCKER) + EFB parity hole + masking + WR-01 closed. `cargo test --workspace` green (0 failed); bin-capture idempotent (only default_config_ingest.txt changed).

**Recent Trend:**

- Last 5 plans: -
- Trend: -

*Updated after each plan completion*
| Phase 03 P01 | ~20 min | 4 tasks | 29 files |
| Phase 03 P02 | ~7 min | 3 tasks | 8 files |
| Phase 03 P03 | ~12 min | 2 tasks | 6 files |
| Phase 04 P01 | ~8 min | 3 tasks | 10 files |
| Phase 05 P01 | 7min | 2 tasks | 5 files |
| Phase 05 P02 | 8min | 3 tasks | 17 files |
| Phase 05 P03 | 25min | 3 tasks | 14 files |
| Phase 05 P05 | 13min | 3 tasks | 3 files |
| Phase 05 P08 | closeout | 3 tasks | 5 files |
| Phase 05 P07 | closeout | 2 tasks | 3 files |
| Phase 5 P9 | checkpoint | 1 tasks | 2 files |
| Phase 06 P01 | 6min | 3 tasks | 19 files |
| Phase 06 P02 | 40min | 3 tasks | 26 files |
| Phase 06 P03 | ~50min | 3 tasks | 37 files |
| Phase 06 P06 | 60 | 6 tasks | 8 files |

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.
Recent decisions affecting current work:

- [Roadmap]: Build order is dependency-forced — each layer must match the reference before anything above it can be validated (binning → predict → compute → learner → GBDT → variants → Python).
- [Phase 1 discuss, 2026-06-05]: Numerical contract revised to **f32 (single-precision) end-to-end, ~1e-6 absolute oracle tolerance** on every backend — matching the C++ reference defaults (`score_t`/`label_t` = `float`). Supersedes the prior strict-1e-12 / tiered-oracle direction.
- [Phase 1 discuss, 2026-06-05]: **Standard f32 histogram/score accumulations** on CPU and ROCm — the integer-quantized histogram strategy is dropped (buys nothing at f32 / ~1e-6).
- [Phase 1 exec, 2026-06-05]: **Deterministic config invariant** — `from_params` alias-collision resolution is a faithful port of C++ `ParameterAlias::KeyAliasTransform` + `Config::SortAlias` (canonical beats alias; alias-vs-alias ties by `(key.len(), key)`). No observable Config outcome may depend on HashMap iteration order; enforced by an N-run determinism test. The seed + six enum reads route through `present()` so empty == absent (C++ `Get*` parity).
- [Phase 1 exec, 2026-06-05]: **Header-only C++ RNG capture** — `rng_capture` compiles directly against `include/LightGBM/utils/random.h` instead of linking `lib_lightgbm` (external_libs submodules not vendored, so the full lib is unbuildable). Numerically identical reference source; preserves FND-01 / ORA-02 / D-14 parity contract. Master seed 1592594996, 512 cases.
- [Phase 2 exec, 2026-06-05]: **Numeric binning capture via verbatim transcription** — `xtask/cpp/bin_capture.cpp` verbatim-transcribes the numeric `BinMapper::FindBin`/`ValueToBin` family from the pinned `bin.cpp`/`bin.h` (using real `std::nextafter`) rather than compiling `bin.cpp`, because `external_libs/{fast_double_parser,fmt}` are empty/unvendored here so `bin.cpp` → `common.h` is unbuildable. Header-only reference `Random` used for sampling. Goldens byte-identical to lib_lightgbm. Binning compared **bit-exact** (`.to_bits()` / exact-u32), NOT the ~1e-6 oracle tolerance. BIN_MASTER_SEED 0x0B11BEEF, 45 cases.
- [Phase 2 exec, 2026-06-05]: **Numeric binning determinism root locked (DAT-01, ORA-03)** — `BinMapper::find_bin`/`value_to_bin` produce `bin_upper_bound_` (f64 `next_up` boundary math, asymmetric `b <= a.next_up()` dedup) and per-row bin indices (literal `(r+l-1)/2` + `<=` search) bit-identical to C++ across 45 cases / 21,475 rows.
- [Phase 2 exec, 2026-06-05]: **Bin-storage layer locked (DAT-02)** — `Bin` trait + `BinValue` width abstraction + `create_dense_bin`/`create_sparse_bin` factories (`Box<dyn Bin>`, D-01) selecting u8/u16/u32 + the 4-bit packed `DenseBin<u8,true>` (num_bin<=16, D-02) exactly per bin.cpp:613-633; `DenseBin` 4-bit even/odd `buf_` split + OR-merge, `SparseBin` 255-run-length delta + GetFastIndex, `FeatureGroup` u64 offset packing + `PushData` (most-freq skip / -1 / +offset) all byte-identical to C++ across 6 storage cases spanning every width path + sparse + odd-row 4-bit.
- [Phase 2 exec, 2026-06-05]: **Dataset immutability = type-state, not a runtime flag** — `Dataset::finish_load(self)` consumes the mutable loading state and returns `FinishedDataset` (no `push_*`/`finish_load` method), so a post-finish mutation is a COMPILE error — strictly stronger than C++ `is_finish_load_` while observably identical.
- [Phase 2 exec, 2026-06-05]: **Categorical folding + missing routing locked (DAT-03, DAT-04)** — `find_bin_categorical` transcribes the descending-count fold via the STABLE `slice::sort_by` (mirrors `SortForPair(is_reverse=true)`; equal-count ties keep ascending-value order), the f32 `0.99` cut (`RoundInt((rest as f32 * 0.99f) ...)` — NOT `0.99_f64`, which would shift the cut), the `min_data_in_bin && cur_cat_idx>1` fold-break, and the NaN dummy bin 0; `categorical_2_bin_` is a by-key `HashMap` (fold order is sort-driven). Missing routing (None/Zero/NaN, signed +0/-0 identical, all-missing single bin) proven bit-identical to C++ across a 6-case categorical (layers 1+3 + per-row) + 8-case missing (layer 1 + per-row) golden battery.
- [Phase 2 exec, 2026-06-05]: **`autobins = false` on lgbm-dataset** — the locked module path `src/bin/{mod,dense_bin,sparse_bin}.rs` collides with Cargo's binary-target convention (Cargo tried to compile the storage module files as `main`-bearing binaries). Disabling binary auto-discovery preserves the path; the crate ships no binaries.
- [Phase 2 exec, 2026-06-05]: **EFB grouping locked (DAT-05)** — `efb.rs` transcribes `FastFeatureBundling`/`FindGroups`/`GetConflictCount`/`FixSampleIndices` verbatim with ALL randomness via `lgbm_core::Random::new(num_data)` (group search `.sample` + shuffle `.next_short`), STABLE sorts only, and element-wise parallel-vector swap (`features_in_group` + `group_is_multi_val` per iter — the C++ `std::swap` on `vector<bool>` hazard). `Dataset::construct_bundled` dispatches on `cfg.enable_bundle` and stores `used_feature_map_` (real->packed) + `real_feature_idx_` (packed->real) so the shuffled bundled grouping pushes each feature to its correct group. EFB layer-3 golden (feature->group membership + per-group `bin_offsets_`/`num_total_bin_`/`group_is_multi_val` + per-row bundled bin index) bit-identical to C++ on the D-06 #4 mutually-exclusive sparse corpus + a no-bundle control. EFB capture = HEADER-ONLY verbatim transcription of `dataset.cpp` (both nominal capture paths infeasible because external_libs are unvendored — human-approved). Fixed a Rule-1 real-vs-packed indexing bug surfaced by the per-row golden.
- [Phase 2 exec, 2026-06-05]: **Scaled pre-filter threshold locked (GAP-1/GAP-2, SC#1/SC#5)** — the default in-memory ingest path now feeds the SCALED `filter_cnt = (min_data_in_leaf * total_sample_cnt) / num_rows` (i64 integer truncation, exact analog of `dataset_loader.cpp:623-624`) to `find_bin_numeric`, computed once in `bin_mapper.rs::scaled_filter_cnt` and called from BOTH `ingest.rs::build_mapper` and `bin_mapper.rs::find_bin_from_column` (no divergent raw-forwarding copy). `from_csr`/`from_csc` inherit via `finish_from_columns -> build_mapper` (CSR/CSC use the identical dense convention `total_sample_size=sample_cnt`, `num_dist_data=total_nrow`, confirmed in c_api.cpp). `num_rows==0` returns raw `min_data_in_leaf` as a Rust-only, parity-unobservable empty-matrix guard (no C++ analog — C++ never reaches FindBin with zero rows). Proven by a default-config (feature_pre_filter=true, sample_cnt=50<num_rows=200, min_data_in_leaf=20 -> filter_cnt=5) ingest parity golden that FAILS before and PASSES after, asserting `is_trivial_` + STORED per-row bins (read from `feature_group().bin_data()`, NOT recomputed). Golden cells are f32-representable (from_mat takes &[f32]) so the f32->f64 widen does not drift boundaries.
- [Phase 2 exec, 2026-06-05]: **Default ingest unified onto the faithful single C++ Dataset::Construct (CR-01, SC#2, DAT-01/02/05/07, ORA-03)** — `from_mat`/`from_csr`/`from_csc` -> `finish_from_columns` now dispatch on `cfg.enable_bundle` through `Dataset::construct_bundled` (the faithful port of the single C++ `Dataset::Construct`, dataset.cpp:325-441) instead of the prior store-everything non-bundled construct. Trivial features (`is_trivial_`) are DROPPED (`used_feature_map_[real] = -1`, no FeatureGroup, no stored bins), so `num_features_`/`feature2group_`/`feature2subfeature_`/`num_total_bin_` are bit-identical to C++ whenever any feature is trivial. The `EfbSamples` it consumes is built to the EXACT c_api.cpp:1352-1374 SAMPLED-SET convention (sample-set-relative positions `0..sample_cnt`, `|v|>kZeroThreshold||isnan(v)` filter, `total_sample_cnt = sample_cnt` NOT num_rows) — DELIBERATELY NOT the efb_grouping.rs full-row convention — reusing the SAME single `create_sample_indices` draw (NO second RNG draw). The non-bundled `construct` was ALSO fixed to filter trivial features so it is never a future divergence trap. The golden emitter must pass `is_sparse=true` (config.h default `is_enable_sparse=true`, dataset.cpp:352) so its FastFeatureBundling grouping/shuffle matches the ingest path. Parity test asserts trivial-exclusion + per-non-trivial `feature_to_group`/`feature_to_subfeature` vs the C++-Construct golden (closing the EFB parity hole), bit-exact stored bins, panics on a missing golden (WR-01), HARD fails-before/passes-after.
- [Phase 2 exec, 2026-06-05]: **Ingestion API locked (D-05, DAT-06/07)** — `from_mat`/`from_csr`/`from_csc` are single validated public entries (validate ALL caller input first → typed `DatasetError`, never panic; Security V5 / T-02-10..13) wiring sample→`find_bin`→`construct`→`push`→`finish_load`. f32→f64 widening at ONE `widen()` site; sparse gather is dense-by-column (absent==0.0, Open Q2). Dense/CSR/CSC of the same matrix bin bit-identically (tolerance-free internal invariant). `Metadata` query weights computed in f32 (`CalculateQueryWeights` verbatim). End-to-end real example-dataset (regression + binary) binning bit-identical to C++ for all 28 features × both datasets (layers 1+2). Example fixtures COPIED into the committed dir, never the untracked LightGBM/ tree.
- [Phase 3 exec, 2026-06-05]: **Golden-capture path B (pip lightgbm 4.6.0 train + dump) selected + human-approved (Task 3 checkpoint)** — only feasible source of a trained `version=v4` `.txt` here (no Rust trainer yet; C++ trainer unbuildable with empty `external_libs`). The prebuilt wheel's `save_model()` is the authoritative `%.17g` v4 format. `xtask model-capture` shells out to `xtask/py/model_capture.py` (via `$LGBM_CAPTURE_PYTHON`), trains 5 corpora (regression/binary/multiclass(3)/categorical/subrange) on the reused Phase-2 example matrices with `deterministic=true force_row_wise=true num_threads=1 seed=MODEL_TRAIN_SEED` + no subsampling, dumps each `model.txt` + raw/transformed/leaf/subrange goldens + `format_golden.txt`. Byte-idempotent; pip is a CAPTURE-time tool only (fixtures committed, `cargo test` needs nothing). REFERENCE_MANIFEST.md "Model / Predict Golden Set" section pins the version + train params.
- [Phase 3 exec, 2026-06-05]: **%g formatter locked (DAT-09 linchpin, lgbm-model)** — `format::format_g17` (`%.17g`) / `format_g6` (`{:g}`) source correctly-rounded significant digits from Rust `format!("{:.*e}", precision-1, x)`, then apply the C/printf `%g` fixed-vs-scientific rule (sci iff decimal exp `< -4` or `>= precision`) + trailing-zero strip + C-locale exponent (lowercase `e`, explicit sign, min 2 digits) — NOT `ryu`/`to_string()`/`{:.17e}`. Proven bit-for-bit vs C printf `%g` on a 10-case battery (`0.1 -> 0.10000000000000001`, `5e-324`, `1e±300`, signed zero, exactly-17-digit case) + bit-exact round-trip (`f64::from_str(format_g17(x)) == x`); committed `format_golden.txt` (authoritative `fmt`) is the arbiter (`golden_matches_formatter`). Used for `threshold`/`leaf_value`/`leaf_weight` (g17) and `split_gain`/`internal_value`/`internal_weight`/`shrinkage` (g6).
- [Phase ?]: feature_importances recomputed via split-count on write
- [Phase 03]: parameters tail (incl pandas_categorical) preserved verbatim on round-trip
- [Phase 03]: Tree parser stricter than C++: validates array lengths + node indices before indexing
- [Phase ?]: 03-03: ConvertOutput parsed from objective= line (not Config); non-core objectives -> ModelError; softmax max-subtraction; leaf per-(iter x class) stride
- [Phase 4 exec, 2026-06-05]: **D-04a SETTLED — cubecl-cpu IS a bit-exact deterministic anchor.** The single-owner ordered f64 fold (`#[cube]` kernel launched with `CubeCount::Static(1,1,1)` + `CubeDim::new_1d(1)`, so exactly one cube unit owns the entire fold) is byte-identical across 25 launches AND bit-exact (`to_bits()`) vs a hand-computed sequential f64 fold in C++ `dense_bin.hpp:120-135` order (f32 read / f64 accumulate, `hist_t=double`, stride-2 `[grad,hess]` at `bin<<1`). This empirically confirms the D-04 anchor assumption despite cubecl-cpu spawning one OS worker thread per cube unit (RESEARCH Pitfall 1) — the ~1e-6 fallback and a separate scalar reference are NOT needed. The kernel MUST zero-initialize `out` explicitly (`client.create_from_slice(&zeros)`) because `client.empty()` returns recycled uninitialized pool memory (Rule-1 bug the spike caught).
- [Phase 4 exec, 2026-06-05]: **CMP-04 capability gate locked (cpu matrix).** `probe_capabilities` queries `client.features().plane.contains(Plane::Ops)` (false on cpu), `client.features().supports_type(f64)` (true), `client.properties().atomic_type_usage(f32_atomic).contains(AtomicUsage::Add)` (false), and `client.properties().hardware.plane_size_max` (1) → `ReducePath::Sequential`. The Sequential single-owner fold IS the cpu path, not a fallback. cubecl type names confined to `lgbm-compute` (`cpu` default feature, `rocm` opt-in via `cubecl/hip` behind `#[cfg]`); CMP-01 guard test enforces no upper crate names cubecl.
- [Phase 05]: 05-01: skip_default_bin==false divergence is observable in default_left (not threshold/gain) — skip=false lets FORWARD reach t=2 (default_left=0), old heuristic skip=true defers to REVERSE (default_left=1) for the same boundary; na_as_missing==true is a typed ComputeError::Runtime (deferred NA_AS_MISSING branch)
- [Phase ?]: Tree::split stores learner-computed leaf outputs (does not compute gain); owns only the C++ parallel-array rewiring + decision_type bit-packing
- [Phase ?]: Tree growth-time arrays (leaf_depth/leaf_parent/split_feature_inner/threshold_in_bin) are NOT serialized; a loaded tree carries parser-default values, so the serialized form remains the identity contract
- [Phase ?]: lgbm-treelearner re-exports the single canonical lgbm_compute::gain::SplitInfo and adds only the split_gt operator> tie-break; no duplicate struct, no compute-runtime dependency (CMP-01)
- [Phase 05]: 05-03: smaller/larger child LeafSplits seeded by DataPartition::leaf_count (not SplitInfo left/right_count) so it agrees with find_best_splits' smaller-child selection (Pitfall 3) — fixed swapped child sums
- [Phase 05]: 05-03: D-07 full-tree parity compares via the SHARED lgbm-model %.17g formatter (golden carries the reference Tree's raw field bits; Rust reconstructs + serializes), not a C++ %g reimplementation
- [Phase 05]: 05-03: FixHistogram is learner-side on RAW (un-bumped) leaf sums (Pitfall 2, Open Q1); ComputeClient re-exported from lgbm-compute so the learner names the Backend client without a cubecl dep (CMP-01)
- [Phase 05]: 05-04: Open Q2 RESOLVED — force_col_wise == force_row_wise on the single-thread deterministic anchor; it is a config FLAG (a no-op) over the shared construct_histograms op, NOT a distinct compute path (A1 confirmed). col_wise.txt carries the identical tree as spine.txt; the row==col equality gate fails loudly if a backend ever diverged.
- [Phase 05]: 05-04: the tree's leaf_count/internal_count record the ACTUAL data_partition leaf_count (serial_tree_learner.cpp:788-791, update_cnt=true), NOT the SplitInfo round_int(hess*cnt_factor) reconstructed counts (which disagree by ±1 for fractional hessians) — Rule-1 fix to the Plan-03 spine, applied in Rust split_inner + the C++ transcription (regenerated spine.txt to the faithful actual-partition counts)
- [Phase 05]: 05-04: ColSampler RNG parity is the CALL SEQUENCE (ResetByTree once per tree, GetByNode per node smaller-then-larger), not the PRNG; col_sampler.txt asserts the exact selected feature indices per draw. D-03 captured-g/h = regression-l2 (hess=1) + binary-logloss (boost_from_average=false); binary capped at num_leaves=2 to avoid the fractional-hessian degenerate-split corner (regression carries the deeper 3-leaf tree)
- [Phase ?]: [Phase 05]: 05-05: D-09 adopted end-to-end — offset_for_most_freq_bin (most_freq_bin==0 -> offset 1) is the SINGLE offset rule; offset==1 uses a COMPACTED histogram (compact_histogram: cell c = real bin c+offset). Supersedes D-01.
- [Phase ?]: [Phase 05]: 05-05: CR-01 root cause = single-feature-group min_bin — C++ FeatureGroup::Split(num_feature_==1) dispatches to DenseBin::Split(max_bin,...) hard-coding min_bin=1; passing min_bin+offset collapses the verbatim --th to th=threshold so partition (bin>th right) == predict (bin<=threshold left). Raw min_bin=0 gave the [4,8] vs [6,6] off-by-one. Closed by the oracle-INDEPENDENT routing test (get_leaf tally == data_partition leaf_count).
- [Phase ?]: [Phase 05]: 05-05: the pre-D-09 learner_capture.cpp full-tree/per-bin goldens (spine/col_wise/real_gh) share the buggy convention (CR-02) so their assertions are SUPERSEDED by 05-06's real lib_lightgbm oracle; row==col equality + routing self-consistency stay live.
- [Phase 05]: 05-08: CR-03 CLOSED — the Rust learner is bit-exact to the real lib_lightgbm 4.6 spine golden and structurally bit-exact to the mfb>0 golden. PRIMARY root cause was the child LeafSplits slot mapping (Fix B: a DIRECT pass-through mirroring C++ smaller_leaf_splits_/larger_leaf_splits_, NOT keyed off smaller==left — the prior swap fed a child its sibling's sums, −17.99 vs 0.55), not the FORWARD-dispatch gate (Fix A, which only fixed decision_type). Plus Fix C (MaybeRoundToZero −0.0→+0 in the shrinkage finalize) and Fix D (bin-0 → float32 kZeroThreshold 1.0000000180025095e-35).
- [Phase 05]: 05-08: the mfb>0 node-2 leaf-0 residual (Rust 0.59999999999999976 vs golden 0.59999999999999953, Δ 2.3e-16 = one f64 ULP) is DEFERRED to 05-07 (user decision). It is a kEpsilon cascade from the not-yet-wired subtraction-trick/HistogramPool (05-07's explicit scope), ~4 orders of magnitude inside the ≤1e-12 contract. The mfb_pos gate stays #[ignore]d with a narrowed honest reason; NO assertion weakened, tolerance-wrapped, or deleted. [SUPERSEDED by 05-07: the subtraction-trick attribution was DISPROVEN — see next entry.]
- [Phase 05]: 05-07: WR-01/WR-02 CLOSED — the subtraction trick (`larger = parent − smaller` via Backend::subtract_histograms) + HistogramPool slot reuse are wired into the LIVE find_best_splits growth path (mirroring C++ serial_tree_learner.cpp:364-378); the dead `let _ = subtract_from;` + orphaned `_pool` are gone; learner_parity_growth_path_subtract proves derived-larger-child == direct build bit-exact AND the spine stays bit-exact. Commit 037e011.
- [Phase 05]: 05-07: the mfb>0 node-2 leaf-0 2.3e-16 ULP did NOT close with the subtraction wiring. ROOT CAUSE CORRECTED — leaf 0 is node-2's DIRECTLY-BUILT smaller child, so the subtraction trick never touches it; the 05-08 subtraction-trick/kEpsilon-cascade attribution is DISPROVEN. The residual is a 2-ULP f64 ACCUMULATION-ORDER subtlety in the FixHistogram-active DIRECT histogram build (construct/FixHistogram/output fold). The SplitInfo-sum LeafSplits seeding trial (serial_tree_learner.cpp:850-870) did NOT change leaf 0 and REGRESSED leaf 3 → 05-09 targets the construct/FixHistogram FOLD ORDER, not LeafSplits seeding. Deferred to NEW plan 05-09; mfb gate stays #[ignore]d with a corrected reason, NO assertion weakened.
- [Phase ?]: Phase 6 Wave-0: four engine crates (objective/metric/boosting/lgbm) scaffolded with thiserror boundaries; lgbm-boosting names no GPU compute runtime (CMP-01).
- [Phase ?]: renew_tree_output seam uses Option<closure> not an objective trait (lgbm-treelearner stays free of lgbm-objective); add_prediction_to_score takes &DataPartition explicitly.
- [Phase ?]: 06-02: L2 score golden is BIT-EXACT (predict(raw,k)==internal score_); phase-wide L2 contract recorded for 06-05
- [Phase ?]: 06-02: ConvertOutput stays in lgbm-model (Open-Q1); lgbm-objective owns training side only, re-exports it
- [Phase ?]: 06-03: custom-objective preds are f64 not f32 (RESEARCH D-04 deviation, LightGBM 4.6 passes f64); required for the bit-exact OBJ-02 cross-anchor
- [Phase ?]: 06-03: regression_l1 leaf values are the median RESIDUAL (RenewTreeOutput) — bit-exact vs real binary + distinct from L2 Newton leaves (Pitfall 2/3)
- [Phase ?]: Task 2b: regression_l1 + bagging typed-rejected (BoostingError::UnsupportedConfig), deferred past Phase 6 — L1 sign-gradient split-gain knife-edge over the bagged subset diverges from the C++ leaf STRUCTURE (rust:0.0 vs cpp:11.0); faithful subset renewal (8330cee) retained

### Pending Todos

None yet.

### Blockers/Concerns

- [Phase 4]: CubeCL is alpha (v0.10.0) — pin exactly, isolate behind lgbm-compute; ROCm capability gaps and CPU-runtime-vs-HIP divergence need empirical validation on the local ROCm GPU (research flag: HIGH). Now evaluated against the ~1e-6 (f32) tolerance rather than bit-exactness.
- [Phase 6]: f32 transcendental (exp/log/pow/sigmoid) parity CPU↔ROCm is unproven at ~1e-6 — needs empirical validation; if a gap appears, fallback is CPU-resident objective grad/hess.
- [Cross-cutting]: RESOLVED (2026-06-05) — the strict-1e-12-vs-tiered tension is closed by adopting the f32 / ~1e-6 contract; project docs (PROJECT/REQUIREMENTS/ROADMAP) updated to match.
- [Phase 5 / 05-09 checkpoint:decision OPEN, 2026-06-06]: mfb>0 node-2 leaf-0 2-ULP (best_sum_left_hessian golden `2.0000000000000018` / `0x4000000000000004` vs Rust `2.0000000000000009` / `0x4000000000000002`; leaf_value `0.59999999999999953` vs `0.59999999999999976`, Δ 2.3e-16) is NOT reachable by any faithful C++ fold-order transcription of the three named hot-path files (fix_histogram.rs / histogram.rs / learner.rs reverse scan). 05-09 Task-1 LOCALIZED it to the leaf-total `sum_hessian` SEED that drives both FixHistogram's bin-2 reconstruction and the bumped reverse-scan denominator — construct/FixHistogram/reverse-scan ORDER are all proven bit-exact/order-independent. Every faithful C++ chain (fresh-fold seed, SplitInfo-reported seed per serial_tree_learner.cpp:875-879, subtraction-trick-derived bins) reproduces Rust's `…002`; the golden `…004` is only reachable by a non-faithful hybrid. USER DECISION required: (1) [recommended] authorize a real-binary FP execution trace of node-2 leaf-0's CalculateSplittedLeafOutput/cnt_factor operands; (2) accept the 2.3e-16 sub-ULP residual without weakening `assert_real_tree_parity`; (3) authorize a Rule-4 LeafSplits/FixHistogram-seam investigation. Do NOT introduce a tolerance or re-attempt LeafSplits seeding (05-07 proved it regresses leaf 3, and 05-09 proved it does not even close leaf 0). See 05-09-SUMMARY.md "checkpoint:decision".
- [Phase 5 / 05-09 RESOLVED, 2026-06-06]: the user chose Option (1). A real lib_lightgbm 4.6 CPU-only single-thread FP execution trace gave ground truth and the residual is now CLOSED bit-exact. The trace overturned the prior premise: the mfb_pos corpus is SPARSE so the real BinMapper collapses most_freq_bin_ = default_bin_ = ValueToBin(0) = 0 (bin.cpp:491-499) → most_freq_bin=0, offset=1, FixHistogram NO-OP (the spine path). The harness had mislabeled it most_freq_bin=2/offset=0, which spuriously activated FixHistogram on node-2's direct build and reconstructed a ~1e-15 bin-2 hessian that polluted the REVERSE scan by 2 ULPs. ALSO: C++ seeds each child LeafSplits DIRECTLY from the parent SplitInfo (best_split_info.left_sum_hessian = best_sum_left_hessian - kEpsilon, feature_histogram.hpp:1042; serial_tree_learner.cpp:851-871), NOT a re-fold — the prior re-fold lost the kEpsilon provenance (4.0 vs C++ 4.000000000000001). Fix (commit 2ced5a2): corrected the corpus to most_freq_bin=0/offset=1 + added LeafSplits::init_from_split seeded from the parent SplitInfo. learner_parity_mfb_pos_real_binary un-#[ignore]d + PASSING bit-exact (node-2 leaf-0 = 0.59999999999999953). The authorized LeafSplits-seam (excluded by the original 3-file scope) was the correct seam. NO tolerance introduced; assert_real_tree_parity byte-unchanged; LightGBM/ never git-added. Phase 5 COMPLETE.
- [Phase 5 / contract-doc reconciliation, 2026-06-06; RESOLVED 2026-06-07]: the stale CLAUDE.md `1e-12` framing has been reconciled to the authoritative ~1e-6 (f32) contract (the 2026-06-05 Phase-1 decision). CLAUDE.md Core Value + Numerical constraint now mirror PROJECT.md, and PROJECT.md's Core Value was enriched to record the two-regime reality: the `cubecl-cpu` f64-fold path is the deterministic anchor / hard merge gate and is **bit-exact** to C++ where the algorithm permits (binning; the serial tree learner is bit-exact vs real lib_lightgbm 4.6 on both committed corpora), while `cubecl-hip` (f32) is the ~1e-6 best-effort gate. All four contract docs (CLAUDE/PROJECT/REQUIREMENTS/ROADMAP) now agree on the ~1e-6-f32-with-bit-exact-CPU-anchor framing; the 2.3e-16 learner residual was closed bit-exact in 05-09 regardless. Historical phase-outcome bullets that say "inside the ≤1e-12 contract" are left as accurate past-tense records.

## Deferred Items

Items acknowledged and carried forward from previous milestone close:

| Category | Item | Status | Deferred At |
|----------|------|--------|-------------|
| v2 | QNT-01 quantized/discretized gradient training | Deferred (v2) | Roadmap |
| v2 | LIN-01 linear-tree leaves | Deferred (v2) | Roadmap |
| v2 | ING-01/02/03 text-file / binary-cache / Arrow ingestion | Deferred (v2) | Roadmap |

## Session Continuity

Last session: 2026-06-07T03:44:10.996Z
Stopped at: Phase 7 context gathered
Resume file: .planning/phases/07-parity-completing-variants/07-CONTEXT.md
