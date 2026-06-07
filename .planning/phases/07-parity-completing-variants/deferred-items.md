# Phase 07 — Deferred Items (out-of-scope / pending-fix discoveries during execution)

These were discovered during execution but require a dedicated follow-up fix plan.
They are logged here per the executor scope boundary and the sanctioned
05-06/CR-03 *ignore-pending-fix* pattern (`#[ignore]` with an HONEST reason that
references this doc — `#[ignore]` is a deferral marker, NOT a mask: no tolerance
is weakened and no horizon is silently capped for any deferred cell).

## DEF-07-02 — fair (all) + quantile bagged/iterated learner-level f64 split-gain knife-edge

> **STATUS: OPEN.** Needs an 07-01-style source-built `lib_lightgbm` 4.6 CPU-only
> single-thread FP execution trace to localize the split-gain operand, exactly as
> 07-01 (D-05) did for the `binary`/`regression_l1` + bagging cells. OBJ-04 is
> therefore PARTIALLY delivered in 07-02: huber, mape, and the quantile SPINE
> shipped faithfully GREEN; fair (all) and quantile bagged/iterated are deferred
> here.

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
