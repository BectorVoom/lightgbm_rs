# Plan Check Result — PASS 3 (final re-review)

**Verdict:** PASS
**Goal:** Close the four remaining C++→Rust parity gaps — G2 (JSON dump), G1 (if-else codegen), G4 (na_as_missing serial routing), G5 (split-kernel gain params) — via TDD, without regressing the bit-exact CPU anchor.
**Plan:** `.planning/plans/unimplemented-features/PLAN.md` (spec_version 2 SPEC.md; 16 tasks, waves A–D)

## Summary
- The pass-2 **MAJOR** (T-G5-1 penalty placement) is **fully resolved**. The plan now pins the `feature_contri` multiply at the top of the numeric post-processing block (~`learner.rs:3113`) BEFORE the CEGB subtract (`:3125`), documents the non-commutativity, covers the categorical branch, and hard-gates the exact C++ order on P-1. CodeGraph/source-verified.
- The pass-2 **MINOR** (file-disjoint claim) is **resolved and accurate**: the plan no longer claims "no shared file"; it correctly states T-G5-1 and T-G5-3 both edit `learner.rs` in disjoint regions.
- No new SPEC↔PLAN inconsistency, correctness error, or coverage gap was introduced. SPEC-ID→task coverage remains complete and 1:1 (16 tasks).
- Remaining G4/G5 bit-exact correctness is UNVERIFIABLE in this sandbox (`LightGBM/` absent) — a correctly hard-gated P-1 prerequisite, not a plan defect. Treated as such per instructions.

## Pass-2 issue verification (source/CodeGraph-confirmed)

### [pass-2 MAJOR — penalty placement vs CEGB + categorical] → RESOLVED
Verified against `crates/lgbm-treelearner/src/learner.rs` (Read + codegraph_explore):
- `:3125` = `split.gain -= delta;` — CEGB **additive** subtract, inside `if cegb_active && split.gain > K_MIN_SCORE`, `delta_gain(f.real_feature_index, …)` keyed per-feature; comment cites `serial_tree_learner.cpp:988-992`. CONFIRMED.
- `:3146` = `split.gain *= penalty;` — monotone **multiplicative** penalty. CONFIRMED.
- `:3113` = `this_leaf_splittable[fpos] = split.gain > K_MIN_SCORE;` — the raw-splittability record immediately after the split is computed (`:3066-3107`) and BEFORE the CEGB block (`:3117-3126`). The plan's "insert at ~:3113 BEFORE the CEGB subtract at :3125" lands in the correct `:3108-3116` gap. CONFIRMED.
  - Additional safety note (not a defect): the flag at `:3113` compares against `K_MIN_SCORE` (the `-inf` sentinel), so a finite `gain * fc` is invariant to whether the penalty is applied just before or just after `:3113` — the splittability flag cannot be perturbed by the multiply regardless of `fc` sign. The plan's placement is therefore safe on this axis too.
- Categorical branch: `if f.bin_type == BinType::Categorical { … continue; }` spans `:2986-3024`; `let split = cat.split;` (`:3002`), its own argmax `split_gt(&split, f.real_feature_index, …)` at `:3013-3014`, and `continue;` at **`:3023`** — it never reaches the `:3113-3173` numeric post-processing. CONFIRMED. The plan's Green (b) "Categorical branch" bullet (PLAN `:736-743`) correctly requires the multiply inside this branch before its argmax (`:3013`), or an explicit P-1-evidenced scope-out; default = apply in BOTH branches.
- The plan explicitly marks the exact C++ order (feature_contri vs CEGB, vs `min_gain_shift`, vs categorical) as P-1 resolution items in BOTH PLAN T-G5-1 Green and SPEC-G5-1 (RESOLVED OQ-1 items i–iv). Given `LightGBM/` is legitimately absent, this is a correctly-gated prerequisite — Wave D notes (PLAN `:656-661`, `:921-943`) hard-gate T-G5-1 Green on P-1. Adequate plan-level resolution.

### [pass-2 MINOR — "file-disjoint / no shared file"] → RESOLVED
- T-G5-3 does modify `learner.rs` ("Modify … learner.rs's call site(s) to supply parent_output", PLAN `:843-847`). CONFIRMED.
- The plan's parallelization note (PLAN `:757-764`) and execution-order section (PLAN `:933-937`) now state T-G5-1 and T-G5-3 share `learner.rs` in **disjoint regions** (per-feature loop ~`:3113` vs the `find_best_split` call site), low real-conflict risk — no longer "no shared file." Accurate. (T-G5-1 vs T-G5-2 remain genuinely file-disjoint.)

## New-problem scan (introduced by the pass-2 edits) — none found
- **SPEC↔PLAN consistency:** SPEC-G5-1 (`:434-443`) now carries the same four TBD items (parse syntax; before/after `min_gain_shift`; ordering vs CEGB with "penalty first, before CEGB" expectation; categorical in-both-branches-or-scope-out) that PLAN T-G5-1 Green pins. Consistent. SPEC §4 (`:184-198`) remains a looser contract sketch ("beside CEGB/monotone, ~3113-3146") but does not contradict the precise pinning in SPEC-G5-1 / T-G5-1 — acceptable, non-material.
- **Line-number accuracy:** every cited line (`:3125` CEGB, `:3146` monotone, `:3113` splittability record, `:3023` categorical continue, `:3013` categorical argmax, `real_feature_index` at `:125-127`/used at `:3008,:3014,:3017,:3124,:3161`) verified against on-disk source. No stale citation introduced.
- **Coverage:** SPEC-ID → task table (PLAN `:1013-1028`) still 16 rows, 1:1 (SPEC-G2-1..4, G1-1..4, G4-1..4, G5-1..4 → T-*). No regression.
- **Trait/scope claims unchanged and still correct:** `find_best_splits_batched` (lib.rs:2571) funnels through `self.find_best_split` (`:2606`,`:2668`); `BatchedSplitFeature` (split.rs:77-95) carries no per-feature penalty/`real_feature_index` — so the learner-level (not kernel) penalty mechanism remains the coherent choice. Unaffected by the pass-2 edits.

## Implementation Order Review (unchanged, still valid)
1. Wave A (G2) → Wave B (G1): additive, correct.
2. Wave C (G4) fully before/after Wave D (G5), never interleaved — both edit `find_best_split_cpu_native`/`find_best_split_f64_on`. Correct.
3. Within Wave D: T-G5-1 (config/mod.rs + learner.rs) is split-kernel-disjoint; shares `learner.rs` with T-G5-3 in disjoint regions (sequence or coordinate). T-G5-2 and T-G5-3 share the two split.rs functions → sequence (G5-2 then G5-3). T-G5-3 lands as one workspace-compiling unit (trait `parent_output` change). All precede T-G5-4.
4. G4/G5 Green strictly after P-1/P-2. Correct.

## Potential Bugs (all mitigated in-plan)
- feature_contri × CEGB non-commutativity → mitigated: penalty pinned before CEGB subtract, exact order P-1-gated.
- feature_contri ignored on categorical features → mitigated: categorical branch added to T-G5-1 scope (or P-1 scope-out).
- NaN feature scanned REVERSE-only (pass-1) → mitigated by T-G4-1's `run_forward` fix.

## Required Plan Revisions
- None. Both pass-2 issues are resolved; the remaining unknowns are legitimately P-1-deferred and hard-gated.

## Unverified Items (correctly P-1/P-2-gated, not blocking)
- Exact C++ application point of `output->gain *= meta_->penalty` (feature_contri) relative to CEGB, `min_gain_shift`, and whether categorical scans are penalized — `LightGBM/` absent (P-1). Evidence needed at Green: `feature_histogram.hpp` FindBestThreshold / FindBestThresholdCategoricalInner penalty point + `serial_tree_learner.cpp` ComputeBestSplitForFeature order.
- Bit-exact parity for all four gaps — requires `lightgbm==4.6.0` goldens (P-2); tests only SKIP here.
- Exact JSON / if-else key sets, order, categorical representation — P-1.
