# Phase 07 — Deferred Items (out-of-scope / pending-fix discoveries during execution)

These were discovered during execution but require a dedicated follow-up fix plan.
They are logged here per the executor scope boundary and the sanctioned
05-06/CR-03 *ignore-pending-fix* pattern (`#[ignore]` with an HONEST reason that
references this doc — `#[ignore]` is a deferral marker, NOT a mask: no tolerance
is weakened and no horizon is silently capped for any deferred cell).

## DEF-07-02 — fair (all) + quantile bagged/iterated learner-level f64 split-gain knife-edge

> **STATUS: RESOLVED (2026-06-08, plan 07-13)** — except the single bagged-renew
> sub-cell re-scoped to **DEF-07-13-01** below. The original "f64 split-gain operand
> knife-edge" framing was DISPROVEN by the 07-13 source-built FP trace: the real
> defects were two learner-side count-source bugs, BOTH fixed C++-faithfully (no
> tolerance weakened, no horizon capped):
> 1. **`split_inner` seeded smaller/larger leaf-splits from SplitInfo
>    `round_int(hess·cnt_factor)` counts instead of DATA-PARTITION counts** — C++ uses
>    partition counts for both the tie-break (`serial_tree_learner.cpp:790-791/851`,
>    `update_cnt=true`) and the histogram-pool dance (`GetGlobalDataCountInLeaf`). Only
>    non-constant hessians tripped it. Fix: compare `part_left < part_right` (commit
>    `15263df`).
> 2. **Missing parent-splittability gate for subtracted children** — under GOSS
>    amplification `cnt_factor = num_data / amplified_sum_hessian` is small, so per-bin
>    `round_int(hess·cnt_factor)` rounds to 0 and a feature can fail `min_data_in_leaf`
>    at the parent yet look splittable on a subtracted child. C++ propagates the parent
>    `is_splittable_` flag (`serial_tree_learner.cpp:395-399`); Rust scanned all features
>    unconditionally. Fix: propagate the gate (commit `56c31c7`). This also closed
>    `fair_loop_matrix` (tree-5 ~2.085 — downstream of the same gate, NOT a 3rd defect).
>
> **12 of 13 Family-A cells un-ignored and asserting real-lib_lightgbm-4.6 parity**
> (commit `8a4a5af`): all fair, all gamma, both tweedie, and `quantile_alpha_axis`.
> OBJ-04 is now delivered for these. The full no-regression merge gate stayed
> bit-exact GREEN (goss_parity_matrix, kernel_parity 4/4, learner_parity keystones,
> subset_determinism_diagnostic, all `*_spine`/`*_gradients`). Root-cause evidence:
> `.planning/debug/split-gain-knife-edge-07-02.md`, plan `07-13-PLAN.md` /
> `07-13-SUMMARY.md`.
>
> _(Historical OPEN framing, superseded: "Needs an 07-01-style source-built
> lib_lightgbm 4.6 FP trace to localize the split-gain operand ... fair (all) and
> quantile bagged/iterated are deferred here.")_

- **Discovered during:** 07-02 execution (OBJ-04 family A capture + replay), the
  "ship green, defer blocked cells" disposition (human-chosen at the 07-02
  blocking-human checkpoint).
- **Family:** identical class to the 07-01 / DEF-06-01 learner-level split-gain
  knife-edge — a borderline split whose `current_gain` vs `min_gain_shift`
  comparison lands on a single-/few-ULP f64 boundary, flipping which split (or how
  many leaves) a tree grows. The gradient/hessian computed by the OBJECTIVE going
  INTO each tree are bit-exact vs the real binary (verified), so this is **NOT an
  objective bug** — the divergence is in the tree learner's f64 histogram/split
  selection, downstream of the faithful g/h.

### Affected (ignored) cells

**fair — ALL cells** (`fair_spine_end_to_end`, `fair_score_accumulation`,
`fair_gradients`, `fair_loop_matrix`, `fair_c_axis`):
- fair is the only family-A objective with a NON-constant hessian
  (`hess = c²/(|x|+c)²`). That tiny, residual-dependent hessian amplifies the
  Newton step on a borderline split, so the f64 split-gain knife-edge flips early.
- Spine cell: leaf values diverge from **tree 2** (~1.3 absolute drift, e.g.
  rust `-1.419…` vs cpp `-1.895…`).
- bfa-OFF loop cells (`fair_bag0_es0_bfa0` etc.): diverge at **tree 0**
  (~64–69 absolute drift, e.g. rust `2.048` vs cpp `0.816` once the amplification
  compounds across iters).
- `fair_score_accumulation` / `fair_gradients` failures (iter-3 score, iter-5 g/h
  ~0.068) are all DOWNSTREAM of the diverged tree-2 split — they are not
  independent objective errors.

**quantile — bagged + non-bagged iterated cells** (`quantile_loop_matrix`,
`quantile_alpha_axis`):
- `quantile_loop_matrix`:
  - BAGGED cells (`quantile_bag1_*`): diverge **tree 4** (~0.18–0.36) AND
    structurally (**12-vs-10 trees** — a bagged-renew divergence where the
    bagged-subset RenewTreeOutput / split selection produces a different grown
    tree count than the real binary).
  - non-bagged ITERATED cells (`quantile_bag0_*`): diverge at **tree 11**
    (~0.009–0.094, e.g. `quantile_bag0_es0_bfa0` tree 11).
- `quantile_alpha_axis` (alpha=0.1, 12-tree iterated param cell): diverges at
  **tree 11** (~0.009) on the same split-gain knife-edge.
- The quantile **SPINE** cell (`quantile_spine_end_to_end`,
  `quantile_score_accumulation`, `quantile_gradients`) **stays GREEN** — it is
  asserted faithfully and passes (the divergence only appears at the longer
  12-iter horizon and on bagged subsets).

### Why not fixed here

Fixing a tree-learner split-gain knife-edge is an architectural investigation
(deviation Rule 4) that needs its own decision and its own evidence base — a
source-built `lib_lightgbm` 4.6 CPU single-thread FP execution trace to localize
the exact operand/rounding divergence (the 07-01 method that un-deferred
DEF-06-01). It is NOT a leaf-VALUE renewal and NOT an objective-math fix. No
assertion was weakened to "pass" these cells; they are `#[ignore]`d with reasons
pointing here so the phase verifier and future planning surface them honestly.

### Tracked for

A future 07-01-style **learner-level split-gain FP-trace fix plan** covering:
1. the fair tiny-hessian histogram/split-gain knife-edge (early-tree flip), and
2. the quantile bagged-renew 12-vs-10-tree structural divergence + the
   non-bagged tree-11 split flip.

Both are expected to share root cause with the 07-01 `min_gain_shift` /
bagged-subset split-gain family (possibly the same `find_best_split` operand
class), so a single FP-trace fix may close several of these cells at once.

### Extended in 07-03 — gamma (all) + tweedie (bfa-off loop + variance_power axis)

> **Same root cause, same family.** 07-03 (OBJ-04 exp/log family + OBJ-05) landed
> poisson, cross_entropy, cross_entropy_lambda, and the tweedie SPINE faithfully
> GREEN. gamma (all cells) and the tweedie bfa-OFF loop + variance_power-axis cells
> hit the SAME non-constant-hessian learner-level f64 split-gain knife-edge as fair
> above — extended here rather than opened as a new DEF.

- **Why gamma/tweedie and not poisson:** at iteration 0 every row shares the same
  score (the `SafeLog(label-mean)` init), so poisson's hessian
  `exp(score)·exp(max_delta_step)` is UNIFORM across rows → its tree-0 histogram /
  split gain matches the real binary bit-exact (poisson is fully GREEN). gamma's
  hessian `label·exp(-score)` and tweedie's
  `-label·(1-ρ)·exp((1-ρ)·score) + (2-ρ)·exp((2-ρ)·score)` are LABEL-dependent
  (non-uniform even at iter 0), so they exercise the SAME borderline f64 split-gain
  comparison that fair's tiny non-constant hessian does — the knife-edge flips which
  leaf a row lands in at tree 0.
- **g/h INTO each tree are faithful (NOT an objective bug):** `tweedie_gradients`
  passes iter-1 AND iter-4 g/h within ORACLE_TOL; `gamma_gradients` passes iter-1
  g/h (the iter-4 failure is DOWNSTREAM of the diverged tree-0 split). The
  divergence is the tree learner's f64 histogram/split selection over the faithful
  non-constant-hessian g/h — identical to the fair case.

Affected (ignored) cells:
- **gamma — ALL cells** (`gamma_spine_end_to_end`, `gamma_score_accumulation`,
  `gamma_gradients`, `gamma_loop_matrix`): the spine itself diverges at **tree 0**
  (rust score 2.360… vs cpp 2.058… → the tree-0 leaf containing feature-0 rows
  differs, rust ≈ -0.375 vs cpp ≈ -3.4).
- **tweedie — bfa-OFF loop + variance_power axis** (`tweedie_loop_matrix`,
  `tweedie_variance_power_axis`): `tweedie_bag0_es0_bfa0` diverges at **tree 0**
  (rust 0.1556 vs cpp 0.0857, ~0.070); `tweedie_variance_power1p9` diverges at
  **tree 0** (rust 2.362 vs cpp 2.144, ~0.218). The tweedie **SPINE** (default
  ρ=1.5, bfa-on: `tweedie_spine_end_to_end`, `tweedie_score_accumulation`,
  `tweedie_gradients`) **stays GREEN** — same as the fair/quantile pattern where the
  default bfa-on cell happens to land on the right side of the knife-edge.

These extend the single future 07-01-style learner-level split-gain FP-trace fix
plan tracked above (the `find_best_split` / non-constant-hessian operand class).
**OBJ-04 stays PARTIAL** until that fix plan closes fair + quantile-bagged + gamma +
tweedie-bfa-off. **OBJ-05 (cross_entropy / cross_entropy_lambda) is fully delivered
GREEN.** No tolerance weakened, no horizon silently capped (the exp/log 5-iter cap is
the carried Pitfall-5 caveat, documented in-code).

### Out-of-scope untracked artifacts noted during 07-02 (not DEF-07-02)

While capturing family-A goldens, the working tree also carried untracked,
*out-of-scope* goldens from separate gaps — `regression_sqrt_*` (06-06 reg_sqrt
ConvertOutput/predict gap) and `regression_mf2es_*` (06-04/CR-02 metric_freq+ES).
Per the executor scope boundary these were **NOT committed by 07-02** and were
moved aside to
`.planning/phases/07-parity-completing-variants/.out-of-scope-fixtures-holding/`
(untracked, regeneratable by the capture xtask) so they do not contaminate the
07-02 green run. They belong to their originating phases' deferral tracking, not
DEF-07-02.

---

## DEF-07-13-01 — quantile bagged-renew structural divergence (`quantile_bag1_es0_bfa0`)

> **STATUS: OPEN (re-scoped from DEF-07-02 by plan 07-13).** The ONE Family-A cell plan
> 07-13 did NOT close. It is a DISTINCT root cause from the (now-fixed) DEF-07-02
> count-source bugs — NOT the offset/split-gain path.

- **Affected (ignored) cell:** `quantile_loop_matrix`, bagged sub-cell `quantile_bag1_es0_bfa0`
  only (`crates/oracle-harness/tests/boosting_parity.rs`). The non-bagged
  `quantile_alpha_axis` + the quantile SPINE are GREEN.
- **Symptom:** 12-vs-10-tree STRUCTURAL divergence. Trees 0–3 are bit-exact; at iter 4
  Rust's bagged subset has all-uniform gradients (0.1 → zero gain → constant 1-leaf tree)
  while the Python-captured golden has a non-uniform 10-row subset (gain 0.4). This is the
  **bagging-draw × quantile-`RenewTreeOutput`** interaction.
- **Why the D-05 FP-trace method can't localize it:** the deterministic source-built
  `lib_lightgbm` 4.6 CLI does NOT reproduce the golden — it stops at 1 tree ("No further
  splits with positive gain" for quantile + bfa-off). So the source-build arbiter that
  closed DEF-07-02/06-01 is unavailable here; this needs a DIFFERENT oracle (likely a
  Python-wheel-side bagging-subset + renew trace), not a CLI FP trace.
- **Disposition:** assertion left fully intact under `#[ignore]` with an honest reason
  (no tolerance weakened, no horizon capped). Tracks a future bagging-renew-specific
  investigation, separate from the closed DEF-07-02 split-gain/count family.

---

## DEF-07-11 (plan 07-11, W10 advanced learner constraints — ADV-01..05 axis knife-edges)

W10 added monotone (ADV-01), interaction (ADV-02), forced splits (ADV-03), extra
trees (ADV-04), and CEGB (ADV-05) as ADDITIVE split gates on the bit-exact serial
learner (D-06 HELD — spine_real/mfb_pos/growth_path_subtract + categorical goldens
stay GREEN bit-exact). Of the 14 per-axis real-binary cells, **10 are GREEN
bit-exact** vs real lib_lightgbm 4.6:

- monotone basic / intermediate / advanced / basic+penalty (`mono_basic_p0`,
  `mono_basic_p5`, `mono_intermediate_p0`, `mono_advanced_p0`)
- interaction one group / two groups (`interaction_one`, `interaction_two`)
- CEGB tradeoff=1.0 / tradeoff=0.5 / coupled (`cegb_t1_psplit`, `cegb_t0.5_psplit`,
  `cegb_coupled`)
- forced SINGLE split (`forced_single`)

**4 cells are DEFERRED** (`#[ignore]`, assertions UNCHANGED — bit-exact, never
weakened). Each grows the tree bit-exact in STRUCTURE (split_feature / topology /
threshold / counts) and diverges only in a last-f64-ULP leaf value or an RNG draw
sequence — the SAME class as DEF-07-02 (a learner-level fold-order / RNG knife-edge
that needs a source-built `lib_lightgbm` 4.6 FP / `meta_->rand` trace to align):

- **DEF-07-11-01 — monotone MIXED vector** (`mono_mixed`): structure bit-exact;
  leaf value `0.05000000000000003` vs golden `0.04999999999999989` (~1.4e-17) — a
  fold-order ULP in the monotone-clamped `CalculateSplittedLeafOutput`.
- **DEF-07-11-02 — NESTED forced split** (`forced_nested`): structure + threshold +
  counts bit-exact; deeper continuation leaf values drift 1-2 ULPs through the
  multi-level forced `GatherInfoForThreshold` seeding (`forced_single` is GREEN).
- **DEF-07-11-03 — extra-trees RNG-replay** (`extra_trees_seed6`,
  `extra_trees_seed9`): the per-feature `Random(extra_seed + i)` +
  `NextInt(0, num_bin-2)` mechanism is wired + DETERMINISTIC per seed (unit-tested:
  same seed ⇒ identical tree), but the realized draw SEQUENCE diverges from
  lib_lightgbm's `meta_->rand` (seed6: 4 vs 3 leaves; seed9: 3 vs 4 — a SWAP ⇒ an
  off-by-one in the per-(feature, leaf-scan) draw timing/order vs the C++
  `BeforeNumerical` call sequence). Needs a source-built `meta_->rand` draw trace.

**Disposition:** ship the 10 GREEN bit-exact cells; the 4 knife-edges are honestly
deferred (no tolerance weakened, no horizon capped, structure assertions intact).
The gate MECHANISMS are all proven by the GREEN real-binary cells + the unit tests
that show each gate alters split selection while inactive constraints match the
spine byte-for-byte (D-06). These 4 extend the single future 07-01-style
learner-level FP/RNG-trace fix plan (the same `find_best_split` / `meta_->rand`
operand class as DEF-07-02). ADV-01..05 are delivered (mechanisms complete,
majority bit-exact); the residual ULP/RNG cells track the FP-trace fix.
