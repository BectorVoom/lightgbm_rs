# 260620-dpk — FINDINGS: lower the unified-fusion per-leaf fixed cost

Backend: cubecl-cpu (f64 anchor). Box: 16-core (rayon_cores). Warm, 3-run window,
cold iteration discarded. Forced-on via `LGBM_UNIFIED_BFS_THRESHOLD=0
LGBM_UNIFIED_SUBSCAN_THRESHOLD=0`; per-bucket counters via `LGBM_FUSION_PROF=1`
(inert otherwise — measurement-only, parity unaffected).

Instrument: `crates/lgbm-compute/src/fusion_prof.rs` — env-gated process-global
atomics timing the four per-leaf buckets of `build_fix_scan_impl` (BFS, smaller /
directly-built child) and `subtract_scan_impl` (SUB, larger / subtract-derived
child). Driven through the production `lgbm::train` path by the 260620-dpk sweep in
`crates/lgbm-treelearner/examples/bench_split_scan.rs`.

---

## Task 1 — per-leaf FIXED-COST decomposition (20/40/60/80 features, 20k rows)

Buckets:
- **GATHER** — the `ord_g`/`ord_h` per-leaf gather loop (BFS only; SUB has no build).
- **ALLOC** — the per-leaf `Vec::with_capacity` setup for `ord_g`/`ord_h`/`ranges`
  (the lever-A-reducible cost). The per-feature private `vec![0.0f64; cells]` lives
  INSIDE the par closure and is counted under PAR (it is fork/join-coupled work, not
  the once-per-leaf setup lever A would remove).
- **PAR** — the `par_iter().zip(..).map(..).collect()` region: fork/join enter/exit
  floor **plus** the actual fold/fix/compact/scan (BFS) or subtract/scan (SUB) work.

### BFS (smaller / directly-built child) — `build_fix_scan_impl`

| feat | gather ms | alloc ms | par ms | total ms | gather % | **alloc %** | par % |
|-----:|----------:|---------:|-------:|---------:|---------:|------------:|------:|
|   20 |    10.70  |   0.62   | 84.88  |  96.20   |   11.1   |   **0.6**   | 88.2  |
|   40 |     9.88  |   0.61   |108.02  | 118.51   |    8.3   |   **0.5**   | 91.1  |
|   60 |    10.37  |   0.63   |125.93  | 136.94   |    7.6   |   **0.5**   | 92.0  |
|   80 |     8.61  |   0.71   |146.00  | 155.33   |    5.5   |   **0.5**   | 94.0  |

### SUB (larger / subtract-derived child) — `subtract_scan_impl`

| feat | alloc ms | par ms | total ms | **alloc %** | par % |
|-----:|---------:|-------:|---------:|------------:|------:|
|   20 |   0.126  | 50.77  |  50.89   |   **0.2**   | 99.8  |
|   40 |   0.107  | 77.51  |  77.62   |   **0.1**   | 99.9  |
|   60 |   0.098  | 89.79  |  89.89   |   **0.1**   | 99.9  |
|   80 |   0.253  | 97.12  |  97.38   |   **0.3**   | 99.7  |

Confirmation re-run (independent warm window) reproduced the fractions within noise:
BFS alloc 0.4–0.6 %, gather 5.4–10.5 %, par 88.8–94.1 %; SUB alloc 0.1–0.2 %,
par 99.8–99.9 %.

### ALLOCATION-vs-FLOOR fraction (the lever decision)

| feat | BFS alloc-reducible | BFS fork/join+work floor | SUB alloc-reducible | SUB floor |
|-----:|--------------------:|-------------------------:|--------------------:|----------:|
|   20 |        **0.6 %**    |        99.4 %            |       **0.2 %**     |  99.8 %   |
|   40 |        **0.5 %**    |        99.5 %            |       **0.1 %**     |  99.9 %   |
|   60 |        **0.5 %**    |        99.5 %            |       **0.1 %**     |  99.9 %   |
|   80 |        **0.5 %**    |        99.5 %            |       **0.3 %**     |  99.7 %   |

(GATHER folded into "floor" above: it is a serial per-leaf cost but it is NOT
allocation, so lever A cannot touch it; lever C could, but see below.)

### Lever decision

- **Lever A (scratch-reuse of `ord_g`/`ord_h`/`ranges`/`out`): RULED OUT.** The
  allocation bucket is **≤0.6 % of BFS and ≤0.3 % of SUB** at every measured feature
  count. Even a perfect zero-alloc reuse removes <1 % of the per-leaf fixed cost —
  far below any amount that could move a ~100-feature crossover. The `with_capacity`
  calls are cheap because the allocator serves them warm from a hot arena every leaf;
  there is no per-leaf fixed cost worth reclaiming here. This is the spike-012 lesson
  inverted: pool-reuse paid off for the *histogram pool* (large, many-cell buffers),
  not for these small short-lived scratch vecs.

- **Lever C (parallelize/hoist the BFS gather): NOT PURSUED.** The gather is 5.5–11 %
  of **BFS only** (the SUB child — which carries the higher `unified_subscan`
  anchor — has **no gather at all**, it is 99.8 % par). Parallelizing a single tight
  `for &row in leaf_rows` loop trades a serial pass for *another* fork/join, which at
  these low feature counts is exactly the floor we are trying to escape — it would add
  dispatch overhead, not remove it. It also cannot help the SUB gate at all. Net: not
  a gate-moving lever.

- **Lever B (core-sized `par_chunks`/`with_min_len`): NOT THE LEVER.** Task 1 shows the
  par region is ~90–99.9 % of the cost, but that fraction is dominated by the **actual
  fold/fix/scan WORK**, not by per-task dispatch granularity that chunking would
  coalesce. At 20–80 features on a 16-core box the existing one-task-per-feature
  `par_iter` already keeps every core busy; chunking changes scheduling shape but not
  the fork/join *count* (still one region) nor the work. There is no measured
  fork/join-granularity signal to exploit.

**Verdict from Task 1: the per-leaf fixed cost is ~99 % fork/join floor + parallel
work and <1 % reducible allocation. No lever materially lowers the gate. Expected
outcome → NULL** (the plan explicitly flagged "the rayon fork/join dispatch floor may
be irreducible — NULL/PARTIAL is an acceptable, expected outcome"). Task 2 therefore
makes NO hot-path logic change (the instrument stays, gated inert); Task 3 re-measures
the crossover to confirm the floor still sets the break-even near the current anchors
and leaves the thresholds unchanged on an honest NULL.

---

## Task 3 — gate crossover re-measurement

Harness: `crates/lgbm/examples/bench_crossover.rs` (release, cubecl-cpu). A = serial
two-step (`LGBM_UNIFIED_BFS_THRESHOLD=usize::MAX LGBM_UNIFIED_SUBSCAN_THRESHOLD=usize::MAX`),
B = unified fusion on (both `=0`). 20k rows, 30 (or 20) iters, median of 3–5 reps,
A/B interleaved per feature count to suppress cross-run drift.

### Single clean wide sweep (run 1, 3 reps, monotone)

| feat | A serial (ms) | B unified (ms) | delta |
|-----:|--------------:|---------------:|------:|
|   20 |     74.2      |    120.8       | +63 % |
|   40 |    128.6      |    145.4       | +13 % |
|   60 |    169.1      |    168.3       | −0.4 %|
|   80 |    222.1      |    203.5       | −8.4 %|
|   90 |    243.3      |    220.5       | −9.4 %|
|  100 |    326.0      |    231.3       | −29 % |
|  120 |    371.7      |    273.1       | −27 % |

This single run *suggested* a crossover near 60–80 feat. But a single run cannot meet
the project's adoption rule (a48/b97/c5v): **sign-stable WIN = no run overlap across
3 independent runs**. The A column above is contaminated at high feat by background
load (the box ran load-avg 5–11 during the campaign — multiple rust-analyzer / editor
processes), which inflates A and manufactures an apparent B-win. The 3-run check below
isolates the real signal.

### 3-run sign-stability (interleaved A/B, 5 reps each)

| feat | run1 delta | run2 delta | run3 delta | sign-stable? |
|-----:|-----------:|-----------:|-----------:|:-------------|
|   70 |  −39.8 %   |   −3.2 %   |  **+1.1 %**|  **NO** — flips (run1 = load artifact, runs 2/3 straddle 0) |
|   80 |   −1.4 %   |   −1.9 %   |   −4.6 %   |  **NO** — all small, A/B distributions OVERLAP (within run spread) |
|  100 |  **+2.6 %**|  −21.1 %   |   −8.2 %   |  **NO** — flips (run1 positive); marginal even at the existing anchor |
|  120 |  −19.3 %   |  −17.6 %   |  −16.1 %   |  **YES** — all negative, tight, NO overlap → genuine WIN |

### Verdict: **NULL** — thresholds UNCHANGED

- The unified fusion's **sign-stable, no-overlap** train-wall win only materialises at
  **≥120 features** on this 16-core box. At 70–100 feat the delta is within run-to-run
  spread and **sign-flips on repetition** — not a win by the hard adoption rule.
- **No fixed-cost reduction was implemented** (Task 1: allocation ≤0.6 %; lever A ruled
  out; nothing else lowered the floor). The fused code is **byte-identical** to the c5v
  HEAD, so there is no mechanism by which the break-even could have dropped below the
  c5v-measured `unified_bfs_threshold=100` / `unified_subscan_threshold=130` anchors.
- **Measured floor:** the rayon fork/join dispatch + per-leaf serial setup + run-to-run
  scheduling variance dominate below ~120 feat, capping the gate at the current anchors.
  This is exactly the "fork/join dispatch floor may be irreducible — NULL/PARTIAL is an
  acceptable, expected outcome" the plan anticipated.

**Decision: keep `unified_bfs_threshold = core_scaled_threshold(100, cores)` and
`unified_subscan_threshold = core_scaled_threshold(130, cores)` unchanged; keep the c5v
`core_scaled_threshold` formula (slope/floor/ceiling) unchanged. No manufactured win.**
The 260620-dpk contribution is the negative result + the reusable `fusion_prof`
instrument that proves the floor, so a future lever (e.g. a fundamentally different
dispatch model) can be evaluated against a measured baseline rather than re-derived.
