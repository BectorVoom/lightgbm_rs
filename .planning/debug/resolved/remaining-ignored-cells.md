---
slug: remaining-ignored-cells
status: resolved
trigger: "Debug the 5 remaining #[ignore]d cells in cargo test --workspace after 07-13 — DEF-07-11 Family B (fold-order ULP + extra-trees RNG) and DEF-07-13-01 (quantile bagged-renew)"
created: 2026-06-08
updated: 2026-06-08
goal: find_and_fix
scope: "All 5 remaining ignored cells, IN ORDER: (1) DEF-07-11-01/02 fold-order ULPs, (2) DEF-07-11-03 extra-trees RNG, (3) DEF-07-13-01 quantile bagged-renew. Family A (DEF-07-02/03) is already CLOSED by 07-13 — do not touch it."
---

# Debug Session: remaining ignored parity cells (post-07-13)

## Symptoms (prefilled — `cargo test --workspace` shows 5 ignored, 0 failed)

- **Expected:** all 5 cells un-`#[ignore]`d and asserting real-lib_lightgbm-4.6 parity (or, where parity is provably irreducible, a documented bounded known-divergence with structure bit-exact + bounded leaf diff + hard-capped count — the 07-01 pattern — NOT a blind weakening).
- **Actual:** 5 cells `#[ignore]`d with honest reasons (no tolerance weakened, no horizon capped, assertions intact). 0 failed; full suite green otherwise.
- **Reproduction:** `LGBM_CAPTURE_PYTHON=/tmp/lgbm-capture-venv/bin/python cargo test --workspace` (ignored cells run with `-- --include-ignored`).

## The 5 cells (3 distinct mechanisms) — investigate IN THIS ORDER

### Group 1 — DEF-07-11-01/02: last-ULP fold-order knife-edges (structure already bit-exact)
- **`monotone_mixed`** (`learner_parity.rs:1876`, DEF-07-11-01): structure/threshold/counts bit-exact; leaf value `0.05000000000000003` vs golden `0.04999999999999989` (~1.4e-17) — a fold-order ULP in the monotone-clamped `CalculateSplittedLeafOutput`.
- **`forced_nested`** (`learner_parity.rs:1905`, DEF-07-11-02): structure + threshold + counts bit-exact; deeper continuation leaf values drift 1-2 ULP through the multi-level forced `GatherInfoForThreshold` seeding. NOTE: `forced_single` (single forced split) is GREEN bit-exact.

### Group 2 — DEF-07-11-03: extra-trees RNG draw-sequence (`learner_parity.rs:1924`/`1931`)
- **`extra_trees_seed6`** + **`extra_trees_seed9`**: the per-feature `Random(extra_seed + i)` + `NextInt(0, num_bin-2)` mechanism is wired + DETERMINISTIC per seed (unit-tested: same seed ⇒ identical tree), but the realized draw SEQUENCE diverges from lib_lightgbm's `meta_->rand` (seed6: 4 vs 3 leaves; seed9: 3 vs 4 — a SWAP ⇒ an off-by-one in the per-(feature, leaf-scan) draw timing/order vs the C++ `BeforeNumerical` call sequence).

### Group 3 — DEF-07-13-01: quantile bagged-renew structural divergence (`boosting_parity.rs:2539`, `quantile_loop_matrix`)
- **`quantile_bag1_es0_bfa0`** (only this bagged sub-cell): 12-vs-10-tree STRUCTURAL divergence. Trees 0-3 bit-exact; at iter 4 Rust's bagged subset has all-uniform gradients (0.1 → zero gain → constant 1-leaf tree) while the golden has a non-uniform 10-row subset (gain 0.4). Root cause is the bagging-draw × quantile-`RenewTreeOutput` interaction.
- **CRITICAL constraint:** the deterministic source-built CLI does NOT reproduce this golden (it stops at 1 tree: "No further splits with positive gain" for quantile+bfa-off). So the D-05 *CLI* FP-trace method is UNAVAILABLE here — this needs a different oracle (a Python-wheel-side bagging-subset + RenewTreeOutput trace via `/tmp/lgbm-capture-venv` lib_lightgbm.so), OR a proof it is an irreducible bagging-draw divergence.

## Current Focus

hypothesis: "Group 1 (fold-order ULPs): the Rust leaf-output accumulation order differs from C++ by one f64 ULP on these specific clamped/forced paths — either a faithfully-fixable fold-order/operand-order bug (like 07-01's bumped-sum_hessian) OR a genuinely irreducible f64 reduction-order artifact. Group 2 (RNG): the extra-trees per-feature draw is correct in MECHANISM but the draw is issued at a different point in the feature/leaf-scan loop than C++'s meta_->rand call site (BeforeNumerical), shifting the realized NextInt sequence by one. Group 3 (quantile bagged-renew): the bagged subset's gradient uniformity at iter 4 diverges because the bagging draw OR the quantile RenewTreeOutput on the bagged subset differs from the wheel reference."
test: "Group 1 first (cheapest, structure already bit-exact): instrument the Rust leaf-output calc vs a source-built lib_lightgbm 4.6 FP trace of CalculateSplittedLeafOutput / the forced GatherInfoForThreshold seeding for the exact monotone_mixed + forced_nested configs; compare the accumulation operand order ULP-by-ULP. Then Group 2: source-built meta_->rand draw trace for extra_trees_seed6. Then Group 3: wheel-side oracle (CLI can't reproduce)."
expecting: "Per cell: EITHER a specific faithful fold-order/RNG-call-site fix that makes it bit-exact, OR a rigorous proof it is an irreducible knife-edge → then a BOUNDED known-divergence (structure bit-exact asserted, |leaf diff| bounded, count hard-capped), never a blind weakening."
next_action: "Start Group 1 — reproduce the D-05 source-built lib_lightgbm 4.6 build (CPU-only single-thread, see resolved Family-A session) and instrument CalculateSplittedLeafOutput for the monotone_mixed config; diff the leaf-output accumulation vs Rust's monotone-clamped path (crates/lgbm-treelearner/src/{monotone_constraints.rs, leaf_splits.rs, learner.rs})."
reasoning_checkpoint: ""

## Reusable method — D-05 source-built FP/RNG trace (proven; closed Family A + DEF-06-01)

Source: `.planning/debug/resolved/split-gain-knife-edge-07-02.md` + `.planning/phases/07-parity-completing-variants/07-01-SUMMARY.md`.
- Build lib_lightgbm 4.6 (`VERSION.txt = 4.6.0.99`) CPU-only single-thread (`-DUSE_GPU=OFF -DUSE_CUDA=OFF -DUSE_OPENMP=OFF -DBUILD_CLI=ON`) into `/tmp` against the repo's populated `external_libs`. A prior build is RETAINED at `/tmp/LightGBM` (reverted/clean). NEVER `git add LightGBM/`; never touch the repo's read-only `LightGBM/` reference tree; revert C++ instrumentation after capture.
- Instrument the relevant hot path env-gated, dumping `.to_bits()` hex operands; validate the build reproduces the wheel/golden BEFORE trusting the trace.
- For Group 3 the CLI does NOT reproduce → use the wheel `lib_lightgbm.so` at `/tmp/lgbm-capture-venv/lib/python*/site-packages/lightgbm/lib/lib_lightgbm.so` as the oracle (a Python-side bagging-subset + RenewTreeOutput capture), or prove irreducibility.

## Key source locations

- Monotone leaf clamp: `crates/lgbm-treelearner/src/monotone_constraints.rs`, `leaf_splits.rs`, `learner.rs` (CalculateSplittedLeafOutput analog)
- Forced splits: `crates/lgbm-treelearner/src/forced_splits.rs` (GatherInfoForThreshold multi-level seeding)
- Extra-trees RNG: `crates/lgbm-treelearner/src/learner.rs` (`find_best_split_rand` / per-feature `Random(extra_seed+i)`+`NextInt`), `col_sampler.rs`; RNG primitive `crates/lgbm-core/src/random.rs`
- Quantile bagged-renew: `crates/lgbm-boosting/src/sample_strategy.rs` (bagging draw), `gbdt.rs` (RenewTreeOutput on bagged subset), quantile objective `crates/lgbm-objective/src/regression.rs` (renew/percentile)
- Tests: `crates/oracle-harness/tests/learner_parity.rs` (4 Family-B cells), `boosting_parity.rs` (quantile_loop_matrix)
- Reference C++ (read-only, never git-add): `LightGBM/src/treelearner/serial_tree_learner.cpp`, `src/io/tree.cpp` (CalculateSplittedLeafOutput), `src/treelearner/feature_histogram.hpp` (BeforeNumerical / extra-trees rand call site), `src/boosting/gbdt.cpp` (RenewTreeOutput), `src/boosting/sample_strategy.cpp` / `bagging.hpp`

## Hard constraints

- NEVER regress the bit-exact merge gate: kernel_parity 4/4, learner_parity keystones (spine_real_binary, mfb_pos_real_binary, growth_path_subtract, the 25 green cells), boosting_parity (73 green incl. the 12 Family-A cells closed by 07-13), goss_parity_matrix, subset_determinism_diagnostic, all *_spine/*_gradients. Full `cargo test --workspace` must stay 0-failed.
- A "fix" is EITHER bit-exact parity OR a documented bounded known-divergence (07-01 pattern: structure bit-exact asserted + bounded |leaf diff| + hard-capped count). NEVER a blanket skip, weakened tolerance, or silently capped horizon.
- If a cell is a genuine architectural change (Rule 4), STOP and surface the decision.

## Evidence

- timestamp: 2026-06-08 (Group 1 reproduction)
  finding: "Reproduced both Group-1 cells bit-for-bit. The parity assertion reaches ONLY the
  leaf_value check (all integer/threshold fields already matched ⇒ structure bit-exact):
    monotone_mixed: rust '0.050000000000000003 -0.050000000000000003' vs golden
                    '0.049999999999999989 -0.049999999999999989' (~1.4e-17, 1 ULP).
    forced_nested:  rust '0.60000000000000064 -0.53333333333333366 0.50000000000000011 -0.60000000000000009'
                    vs golden '0.60000000000000087 -0.53333333333333355 0.49999999999999983 -0.59999999999999953'
                    (1-2 ULP per deeper leaf).
    forced_single:  GREEN (single forced split, no nested re-seed)."
  source: cargo test -p oracle-harness --test learner_parity -- --include-ignored monotone_mixed forced_nested forced_single

- timestamp: 2026-06-08 (forced_nested ROOT CAUSE — re-fold loses kEpsilon provenance)
  finding: "ROOT CAUSE (faithful-fixable): `apply_forced_splits` (learner.rs:965-968) RE-FOLDS
  EACH forced leaf's sums from scratch via `LeafSplits::init(gradients, hessians, &leaf_rows, cfg)`
  for EVERY BFS level. C++ `ForceSplits` (serial_tree_learner.cpp:638-734) does NOT re-fold: at the
  top of each BFS iteration it calls `BeforeFindBestSplit` + `FindBestSplits`, then
  `GatherInfoForThreshold` consumes `left_leaf_splits->sum_gradients()/sum_hessians()/num_data_in_leaf()`
  — the leaf-splits seeded by the PRIOR `SplitInner` (serial_tree_learner.cpp:853-892) DIRECTLY from
  `best_split_info.{left,right}_sum_hessian` (= `best_sum_left_hessian - kEpsilon`,
  feature_histogram.hpp:1042), carrying the parent REVERSE-scan kEpsilon + FixHistogram fold-order
  provenance. The fresh re-fold loses that provenance ⇒ deeper-leaf output denominator drifts 1-2 ULP.
  This is the EXACT class the 05-09 mfb>0 fix + the in-code comment at leaf_splits.rs:124-138 already
  document; the spine path uses `split_inner`'s `init_from_split` (kEpsilon-bearing), but the FORCED
  BFS re-folds. `forced_single` is GREEN because the ONLY forced leaf is the root, whose fresh fold
  over ALL rows == C++ `LeafSplits::Init()` (the whole-dataset variant) bit-exact; only NESTED (level-1+)
  forced leaves trip the re-fold."
  source: learner.rs:965-968 (init re-fold) vs C++ serial_tree_learner.cpp:638-734 (ForceSplits) + 853-892 (SplitInner seed)

- timestamp: 2026-06-08 (Group 1 BOTH ROOT CAUSES — output operand uses RAW hessian, NOT -kEpsilon)
  finding: "FP trace (instrumented /tmp/LightGBM 4.6, env LGBM_FORCED_TRACE) reproduces the
  forced_nested golden BIT-EXACT (leaf_value 0.60000000000000087 -0.53333333333333355
  0.49999999999999983 -0.59999999999999953). C++ ground truth for the left-child level-1 forced
  split: INPUT sum_h=0x400ffffffffffffc (3.9999999999999982, kEpsilon-bearing from parent FT_OUT),
  OUTPUT L out=0x4018000000000009 (6.000000000000008). The Rust leaf-seed fix made the INPUT sums
  match C++ bit-exact, but the OUTPUT still drifted (0x401800000000000d=6.0000000000000115, 4 ULP).
  DECISIVE: C++ `GatherInfoForThresholdNumericalInner` (feature_histogram.hpp:580-590) AND the
  monotone/spine `FindBestThresholdSequentially` (feature_histogram.hpp:1049-1066) BOTH compute the
  child OUTPUTS via `CalculateSplittedLeafOutput(best_sum_left_gradient, best_sum_left_hessian, ...)`
  using the RAW `best_sum_left_hessian` (and `sum_hessian - best_sum_left_hessian` for the right),
  then store `{left,right}_sum_hessian = <raw> - kEpsilon`. The Rust forced gather + monotone
  build_split computed the output from the ALREADY-`-kEpsilon` hessian → ~1 ULP output drift.
  Verified: -(-12.0)/(1.9999999999999962 + kEpsilon[1.0000000036274937e-15]) == 6.000000000000008
  == 0x4018000000000009 == golden. TWO faithful fixes:
    (1) forced path: apply_forced_splits consumes the kEpsilon-bearing split_inner leaf-seeds (NOT
        a re-fold) + gather_info_for_threshold computes left/right_output from RAW hessian.
    (2) monotone path: build_split takes RAW (sum_g, sum_h, left_sum_g, left_sum_h), computes the
        clamped output from RAW hessian, stores `<raw> - kEpsilon`, and uses the C++ right operand
        order (sum_g - left_sum_g, sum_h - left_sum_h).
  RESULT: forced_nested + forced_single GREEN; all 5 monotone cells (incl. mono_mixed) GREEN;
  no monotone regression. gather is called ONLY from apply_forced_splits (isolated)."
  source: /tmp/forced_fp/forced_trace.log (FT_IN/FT_OUT) vs RST_FT trace; feature_histogram.hpp:580-590,1049-1066

## Eliminated

- "Group 1 is an IRREDUCIBLE f64 fold-order knife-edge" — ELIMINATED. It is a FAITHFULLY-FIXABLE
  operand bug: the child leaf output was computed from the `-kEpsilon`'d hessian instead of the RAW
  hessian (C++ computes output from raw, stores raw-kEpsilon). Both forced_nested and monotone_mixed
  reach bit-exact parity after the fix.

- "Group 2 is an RNG draw-ORDER / draw-TIMING off-by-one" — ELIMINATED. The Rust `Random` primitive
  + `next_int` + per-feature-per-leaf draw COUNT/ORDER all match C++ bit-exact (source-built trace:
  Random(6).NextInt(0,2)=1, Random(7)=0; draws fire in the same per-feature smaller-then-larger
  order). The actual bug was the RNG SEED-to-FEATURE mapping: C++ seeds `meta_->rand[inner_i] =
  Random(extra_seed + inner_i)` by the DATASET INNER feature index, which LightGBM's feature
  bundling REVERSES vs the real/sidecar column order (trace: CPP_MAP inner=0 real=1, inner=1 real=0).
  The harness seeded by real order → wrong LCG stream per feature → root structure flip.

- timestamp: 2026-06-08 (Group 3 ROOT CAUSE — no-split bagged iterations: C++ SKIPS the tree, Rust APPENDS a constant)
  finding: "ROOT CAUSE precisely localized via a Python-wheel oracle (CLI can't reproduce). The
  quantile_bag1_es0_bfa0 divergence is NOT a bagging-draw or renew bug — those are bit-exact:
    (1) Rust's per-iteration bagged subset MATCHES the C++ algorithm bit-for-bit (verified Rust
        BAG_TRACE == a standalone Python sim of bagging.hpp BaggingHelper for all 12 iters; e.g.
        iter4 in_bag=[0,1,4,5,6,7,8,9,10], oob=[2,3,11] in BOTH).
    (2) Rust's scores through tree 3 are BIT-EXACT to the wheel (after 4 trees both =
        [16.3164×6, 17.6374×2, 18.1274×2, 19.0081×2]); the iter-4 bagged gradients are uniformly
        +0.1 in BOTH (row 11, the only -0.9, is OOB) → a genuine NO-SPLIT iteration.
  The REAL divergence is the no-split-tree EMISSION policy. C++ `GBDT::TrainOneIter`
  (gbdt.cpp:406-447): when the grown tree has num_leaves<=1, the FIRST iteration keeps an
  AsConstantTree(init) baseline, but a LATER no-split bagged iteration POPS the would-be constant
  tree (the `!should_continue` path) and emits NO tree (wheel verbose: 3× 'Stopped training because
  there are no more leaves'; num_trees()=10 from num_boost_round=12). Rust `gbdt.rs:765,828` instead
  APPENDS a 1-leaf `Tree::as_constant(0.0)` on EVERY no-split iteration. So Rust grows 12 trees
  [1,2,3,3,1,3,3,3,1,3,2,3] while the wheel grows 10 [1,2,3,3,3,3,3,3,2,3]; the 2 extra Rust
  constants (iters 4,8) SHIFT every later tree → the loop test (which compares tree[i] vs golden
  tree[i]) sees Rust tree 4 (constant, 1 leaf) vs golden tree 4 (= Rust tree 5, 3 leaves). Confirmed
  bit-exact: wheel tree4 leaf_value [-0.18273995881080618,-0.000809943974018,0.09919005602598184]
  == Rust tree5. regression_l1_bag1 keeps all 12 (only its FIRST tree is no-split) so it is
  unaffected — the divergence is specific to objectives whose bagged subsets hit LATER no-split
  iterations (quantile alpha=0.9 here)."
  source: /tmp/quantile_oracle/{oracle.py,bag_sim.py} (wheel), Rust BAG_TRACE/tree-dump; gbdt.cpp:406-447 vs gbdt.rs:758-829

- timestamp: 2026-06-08 (Group 2 ROOT CAUSE — RNG seeded by INNER (reversed) feature index)
  finding: "ROOT CAUSE (faithful-fixable): extra-trees per-feature RNG was seeded by the REAL/sidecar
  feature position, but C++ seeds `Random(extra_seed + inner_i)` by the DATASET INNER feature index
  (feature_histogram.hpp:1450). For these dense 2-feature corpora LightGBM's feature bundling
  (dataset.cpp:387-406) REVERSES the inner order vs the columns — source-built trace shows
  CPP_MAP inner=0->real=1, inner=1->real=0, so real feature 0 draws Random(extra_seed+1). seed6:
  Random(6).NextInt(0,2)=1 (rt for the inner-0 feature), Random(7)=0 (rt for real-0); real-0's rt=0
  admits the bin-0 (offset=1 implicit-default) split → gain 96 (2/6) which beats the rt=1 4/4 split
  → 3 leaves. The harness's real-order seeding gave real-0 rt=1 → wrong root → 4 leaves. Fixing the
  seed offset to the inner (reversed) index made the STRUCTURE bit-exact for BOTH seeds; the residual
  last-ULP leaf-value diff was the SAME output-from-`-kEpsilon` bug as Group 1, also present in
  `find_best_split_rand` (REVERSE + FORWARD). Both fixes ⇒ extra_trees_seed6/9 bit-exact."
  source: /tmp/extra_fp/rand_trace6.log (CPP_MAP/CPP_RAND/CPP_BEST) vs RST_RAND trace; goldens seed6/9

## Resolution
status: "4 of 5 cells FIXED bit-exact (Groups 1+2 closed); 1 cell (Group 3) ROOT-CAUSED + deferred
pending a Rule-4 architectural GBDT-loop change (surfaced for decision)."
root_cause: "THREE distinct root causes across the 5 cells:
  (Group 1, DEF-07-11-01/02 — monotone_mixed, forced_nested) The child leaf OUTPUT was computed
  from the already-`-kEpsilon`'d hessian; C++ computes `CalculateSplittedLeafOutput` from the RAW
  hessian and only THEN stores `<raw> - kEpsilon` (feature_histogram.hpp:580-590 forced,
  1049-1066 monotone/spine). Plus: forced BFS re-folded leaf sums instead of consuming the
  kEpsilon-bearing split_inner seeds (C++ ForceSplits never re-folds). Proven by source-built FP trace.
  (Group 2, DEF-07-11-03 — extra_trees_seed6/9) The per-feature extra-trees RNG was seeded by the
  real/sidecar feature order; C++ seeds `Random(extra_seed + inner_i)` by the DATASET INNER feature
  index, which feature bundling REVERSES vs the columns for these dense corpora (trace CPP_MAP
  inner=0->real=1). Plus the same RAW-vs-`-kEpsilon` output-operand bug in find_best_split_rand.
  (Group 3, DEF-07-13-01 — quantile_bag1_es0_bfa0) NOT a bagging/renew bug (both bit-exact); the
  no-split-tree EMISSION policy differs: C++ TrainOneIter pops the would-be 1-leaf constant on a
  later no-split bagged iteration and re-bags next round (no tree, no iter advance), while Rust
  appends a 1-leaf constant and advances → 12 trees vs the wheel's 10, shifting later trees."
fix: "Groups 1+2 FIXED + un-#[ignore]'d (4 cells bit-exact). Commits 1a9e1ef (Group 1: RAW-hessian
output operand in gather_info_for_threshold + monotone build_split; forced BFS consumes kEpsilon
leaf-seeds) + 3b03f6e (Group 2: extra-trees RNG seeded by inner/reversed index + RAW-hessian operand
in find_best_split_rand). Group 3 root-caused + deferred with a sharpened reason + proposed fix
(e1449e8); its fix is a Rule-4 change to the shared GBDT boosting-loop no-split tree-emission /
iter-advance / re-bag semantics, isolated to later no-split bagged iterations (would not regress the
73 green boosting_parity cells — none has a non-first 1-leaf tree) but warranting an explicit decision."
verification: "Full `LGBM_CAPTURE_PYTHON=... cargo test --workspace` = 0 failed, 1 ignored (was 5).
Merge gate intact: kernel_parity 4/4, learner_parity 29/29 (was 27+2 ignored → all green),
boosting_parity 73 + 1 ignored, lgbm-treelearner 64, goss_parity_matrix + dart + rf + spine/gradients
all green. Groups 1+2 fixes confirmed bit-exact against source-built lib_lightgbm 4.6 FP/RNG traces.
All instrumentation reverted: /tmp/LightGBM clean (git checkout); temp Rust traces removed; repo
LightGBM/ never touched. Reproduction assets retained: /tmp/LightGBM (source build), /tmp/forced_fp,
/tmp/extra_fp, /tmp/quantile_oracle."
files_changed: "crates/lgbm-treelearner/src/learner.rs (gather_info_for_threshold RAW-hessian output;
apply_forced_splits kEpsilon leaf-seed consumption; find_best_split_rand RAW-hessian output;
extra_rng inner-index seeding), crates/lgbm-treelearner/src/monotone_constraints.rs (build_split
RAW-hessian clamped output + C++ right operand order), crates/oracle-harness/tests/learner_parity.rs
(un-ignore 4 cells), crates/oracle-harness/tests/boosting_parity.rs (DEF-07-13-01 sharpened reason),
.planning/phases/07-parity-completing-variants/deferred-items.md (DEF-07-11-* CLOSED, DEF-07-13-01
root-caused)."
