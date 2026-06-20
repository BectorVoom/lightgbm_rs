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

<!-- gsd:njg-task2 -->
