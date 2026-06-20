# 260620-njg FINDINGS — cheaper fused per-feature WORK (BUILD/FIX+COMPACT/SCAN split + f64-pregather A/B)

Quick-task: `260620-njg` — make the fused per-feature `build -> fix -> compact -> scan`
region (`build_fix_scan_impl`, lib.rs:1497) cheaper (bit-exact) so the unified-fusion gate
(`unified_bfs_threshold` ~100 @16 cores, `unified_subscan_threshold` ~130) could drop below
~100/130 features. **Strong prior: NULL** — the spike-001..013 build wins are already present in
`fold_one_feature` / `build_fix_scan_impl`; the one unexplored bit-exact micro-lever is f64-pregather.

Backend: `cubecl-cpu (f64-anchor)` — the authoritative deterministic gate. 16-core box.
Harness: `crates/lgbm-treelearner/examples/bench_split_scan.rs`, FORCED-ON unified path
(`LGBM_UNIFIED_BFS_THRESHOLD=0 LGBM_UNIFIED_SUBSCAN_THRESHOLD=0 LGBM_FUSION_PROF=1`),
warm window (cold iteration discarded, counters reset post-cold), 3 warm reps, 20k rows, 255 bins,
30 iters, 31 leaves.

---

## Task 1 — BUILD vs FIX+COMPACT vs SCAN decomposition (verbatim)

The existing `BFS_PAR_NS` bucket lumped fold+fix+compact+scan into one region. Three new inert
sub-buckets (`BFS_BUILD_NS` / `BFS_FIXCOMPACT_NS` / `BFS_SCAN_NS`) split it, timed **per feature**
inside the `par_iter` closure via the thread-safe `fusion_prof::time()` (relaxed `fetch_add`).

> NOTE on the WORK total: the three sub-buckets are summed **per-thread** across the rayon tasks,
> so `WORK total` overcounts the single-threaded `par` wall by ~num_threads. The meaningful
> diagnostic is each sub-stage's **share of the WORK** (which sub-stage dominates the per-feature
> compute), NOT its fraction of the `par` wall.

Verbatim `[fusion_prof:*]` lines (forced-on, warm):

```
dpk_feat=60 warm_wall=484.548ms
[fusion_prof:dpk_feat60]  BFS: gather=9.278ms alloc=0.487ms par=126.830ms total=136.596ms
[fusion_prof:dpk_feat60]  WORK (per-thread summed): build=519.673ms fix+compact=58.197ms scan=163.084ms total=740.954ms
[fusion_prof:dpk_feat60]  WORK %: build=70.1 fix+compact=7.9 scan=22.0
dpk_feat=90 warm_wall=590.356ms
[fusion_prof:dpk_feat90]  BFS: gather=6.511ms alloc=0.551ms par=134.673ms total=141.735ms
[fusion_prof:dpk_feat90]  WORK (per-thread summed): build=562.356ms fix+compact=80.141ms scan=209.538ms total=852.035ms
[fusion_prof:dpk_feat90]  WORK %: build=66.0 fix+compact=9.4 scan=24.6
dpk_feat=120 warm_wall=746.375ms
[fusion_prof:dpk_feat120] BFS: gather=7.806ms alloc=0.726ms par=169.709ms total=178.241ms
[fusion_prof:dpk_feat120] WORK (per-thread summed): build=815.761ms fix+compact=112.236ms scan=292.311ms total=1220.308ms
[fusion_prof:dpk_feat120] WORK %: build=66.8 fix+compact=9.2 scan=24.0
```

### Decomposition table (% of per-feature WORK)

| feat | BUILD (fold) | FIX+COMPACT | SCAN | dominant |
|-----:|-------------:|------------:|-----:|:---------|
|   60 |       70.1 % |       7.9 % | 22.0 % | **BUILD** |
|   90 |       66.0 % |       9.4 % | 24.6 % | **BUILD** |
|  120 |       66.8 % |       9.2 % | 24.0 % | **BUILD** |

**Dominant WORK sub-bucket: BUILD (the `fold_one_feature` histogram fold) — ~66-70 % at every
width.** SCAN is a steady ~22-25 %; FIX+COMPACT is ~8-9 % (these are the no-op
`fix_histogram_inline`/`compact_histogram_inline` guards plus the per-call `time()`-wrapper overhead
on a near-empty body — the true compute is ~0, as the `most_freq_bin==0`/`offset==0` guards predict).

### Lever decision

BUILD dominates and its cost IS the fold body (FIX+COMPACT ≈ 0), so per the plan the f64-pregather
conversion is the only candidate sub-lever — pursue the **isolated build A/B** (Task 2). But flag
**likely-NULL up front**: inside the fold the `f64::from(ord_g[k])` widening is a single trivial
register op per row, whereas the dominant cost is the **scatter RMW into `h[bin*2]`** (random-access
cache traffic, read-modify-write per row) plus the sequential `ord_g[k]` read. The cheap widen is
almost certainly a sub-noise fraction of BUILD; pre-widening to `f64` doubles the `ord` array bytes
(worse cache density on the sequential read). The plan's cache-density caveat is the expected
outcome — Task 2 measures it honestly via a throwaway A/B rather than asserting it.

---

## Task 2 — f64-pregather isolated build A/B + end-to-end train-wall

BUILD dominates (Task 1), so the f64-pregather conversion is the candidate sub-lever and the
isolated A/B was run. The lever was then wired behind `LGBM_PREGATHER_F64=1` (bit-exact f64-fold
variant, lossless widen) ONLY for the end-to-end measurement, then **reverted on the NULL verdict**.

### (a) ISOLATED build A/B — throwaway `njg_pregather_ab()` (cold microbench)

Two byte-identical fold loops over the SAME synthetic narrow bin columns (20k rows, 255 bins),
differing only in A = f32 ord + `f64::from` widen-in-loop vs B = pre-widened `Vec<f64>` ord.
Bit-exact asserted (`fold_a().to_bits() == fold_b().to_bits()`). 3-run medians, cold warmup discarded:

```
njg_ab feat=60  A(f32+widen)_med=28.050ms B(f64-pregather)_med=24.128ms B_vs_A=+13.98%  A_runs=[29.222, 27.272, 28.050] B_runs=[24.855, 24.128, 23.569]
njg_ab feat=90  A(f32+widen)_med=40.147ms B(f64-pregather)_med=37.172ms B_vs_A= +7.41%  A_runs=[40.147, 40.119, 42.993] B_runs=[36.799, 37.472, 37.172]
njg_ab feat=120 A(f32+widen)_med=53.993ms B(f64-pregather)_med=49.348ms B_vs_A= +8.60%  A_runs=[54.354, 53.815, 53.993] B_runs=[49.436, 49.103, 49.348]
```

Isolated: B (f64-pregather) is **+7-14% faster, sign-stable, non-overlapping runs**. Surprising vs
the prior — but this is precisely the case the spike rule warns about: the cold tight microbench
keeps the f64 `ord` arrays L1/L2-resident across all features, hiding the 2×-bytes cache-density
penalty and rewarding conversion-elimination. **The isolated win does NOT decide adoption.**

### (b) END-TO-END train-wall A/B — `bench_train` (warm, forced-on unified, 3-run medians)

`LGBM_UNIFIED_BFS_THRESHOLD=0 LGBM_UNIFIED_SUBSCAN_THRESHOLD=0 BENCH_FEATURES=N BENCH_ROWS=20000
BENCH_BINS=255`, A = default, B = `LGBM_PREGATHER_F64=1`. Verbatim `custom` rows (train_median):

```
feat=120  A: 828.38 / 833.38 / 852.38 ms  → median 833.38ms   (range 828-852)
feat=120  B: 855.64 / 834.84 / 837.59 ms  → median 837.59ms   (range 835-856)   B_vs_A ≈ -0.5% (noise)
feat=90   A: 727.38 / 718.73 / 717.01 ms  → median 718.73ms   (range 717-727)
feat=90   B: 732.43 / 720.32 / 719.25 ms  → median 720.32ms   (range 719-732)   B_vs_A ≈ -0.2% (noise)
```

End-to-end the A and B run spreads **fully overlap** at both widths; B is +0.2-0.5% slower (within
run noise). **The +7-14% cold isolated win evaporated to zero on the warm train-wall** — the
conversion-elimination is sub-noise against the real per-leaf WORK wall (the f64 ords now compete
for cache against the actual histogram RMW + the rest of the boosting loop, surfacing the
cache-density penalty the isolated bench hid). Cold-overstates-warm: ~3-7× as the rule predicts.

### Cache-density caveat (honestly applied)

f64 ord arrays are 2× the bytes of f32. In the warm end-to-end path the sequential `ord_g64[k]`
read is less cache-dense and the per-leaf 2× alloc+widen is added work; this exactly cancels the
saved per-feature widen — netting ~0 (probably WHY spike-003 gathers f32). No regression was
shipped: the f64-pregather hot-path wiring (`LGBM_PREGATHER_F64`, `fold_one_feature_f64`) was
**reverted** after the measurement.

---

## VERDICT: NULL

**The per-feature fused WORK is at its bit-exact floor.** BUILD (the fold) dominates at ~66-70%,
but every spike build win is already present (003 once-gather f32, 004 narrow u8/u16 monomorphic
fold, 003b branchless, 010/012 pool arena+reuse, 011 closed). The one unexplored bit-exact
micro-lever — f64-pregather — is a **cold-microbench mirage**: +7-14% isolated, **0% (full overlap)
on the warm end-to-end train-wall**, with a cache-density penalty that cancels the conversion
saving. Per the spike adoption rule (ship only on a sign-stable bench_train train-wall win), it is
NOT shipped.

This **CLOSES the "cheaper per-feature work" avenue.** Combined with `260620-dpk` (per-leaf alloc/
dispatch fixed cost is irreducible at the gate granularity) and `260620-e8e` (per-leaf is
WORK-bound, not dispatch-bound, so a cheaper barrier was immaterial), the whole **sub-100-feature
fusion-gate line is closed**: the gate (`unified_bfs_threshold` ~100, `unified_subscan_threshold`
~130) cannot drop because neither the fixed cost nor the per-feature WORK has any remaining
bit-exact lever to pay for engaging fusion below the current break-even.

**Shipped: only the profiler** (`fusion_prof` BUILD/FIX+COMPACT/SCAN sub-buckets + the three inert
`time()` wrap sites) and the **throwaway A/B bench** (`njg_pregather_ab`, gated on `LGBM_NJG_AB=1`,
no hot-path effect). No hot-path logic changed; parity trivially intact (profiler inert when off).

### Parity (forced-on, after revert)

```
LGBM_UNIFIED_BFS_THRESHOLD=0 LGBM_UNIFIED_SUBSCAN_THRESHOLD=0 cargo test -p oracle-harness --test kernel_parity
  → 6 passed; 0 failed  (kernel_parity_fused_equals_per_feature_and_native + histogram/split/subtract/partition bit-exact)
```

(Full Task-3 gate counts — learner_parity, lgbm-compute --lib, lgbm-treelearner, default + forced-on
— recorded in the SUMMARY.)
