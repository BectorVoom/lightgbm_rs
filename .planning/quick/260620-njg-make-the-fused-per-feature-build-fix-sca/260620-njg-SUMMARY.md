---
phase: quick-260620-njg
plan: 01
subsystem: cpu-treelearner
tags: [fusion, per-feature-work, profiling, f64-pregather, bit-exact, NULL, cold-overstates-warm]
provides:
  - "Evidence that the fused per-feature build+fix+scan WORK is at its bit-exact floor: BUILD dominates (66-70%) and already carries every shipped spike build win (narrow u8/u16 bins, once-gather, fused-branchless); the f64-pregather micro-lever is a cold-microbench mirage that vanishes warm."
  - "An extended fusion profiler splitting BFS_PAR into BUILD/FIX+COMPACT/SCAN sub-buckets (LGBM_FUSION_PROF=1, inert when off)."
affects:
  - "none (no hot-path logic change; only inert profiler wraps + a throwaway A/B bench + NULL comments)"
decisions:
  - "NULL: the per-feature WORK is at its bit-exact floor. BUILD is 66-70% of per-feature work and already uses spike-004 narrow bins / spike-003 once-gather / 003b fused-branchless; scatter-into-shared is closed (spike-011); the f64 fold + branchless scan arithmetic is fixed by parity."
  - "f64-pregather (pre-widen once-gathered ord_g/ord_h to f64 to drop per-feature f32->f64 reconversions) = NULL: cold fold microbench +7-14% sign-stable, but warm bench_train FULL OVERLAP — the 2x-byte f64 ords' cache-density penalty cancels the saved conversion (the canonical cold-overstates-warm mirage). Not shipped, prototype reverted."
  - "Closes the 'cheaper per-feature work' avenue and, with dpk (not allocation) + e8e (not dispatch), the entire sub-100-feature gate line."
metrics:
  duration: "~10 min"
  completed: "2026-06-20"
verdict: NULL
---

# Quick 260620-njg: Make the fused per-feature build+fix+scan work cheaper — VERDICT NULL

**Goal:** make the fused per-feature `build -> fix -> compact -> scan` WORK cheaper (bit-exactly)
to enable dropping the unified-fusion gate below ~100/130 feat. e8e proved the per-leaf cost is
work-bound (not dispatch); dpk proved it's not allocation. This task tested the work itself.

## What was done

### Task 1 — Profile the build-vs-scan split (the diagnostic)
Extended `fusion_prof.rs` to split the lumped `BFS_PAR_NS` work region into BUILD
(`fold_one_feature`) / FIX+COMPACT / SCAN (`find_best_split`) sub-buckets (inert when
`LGBM_FUSION_PROF` off). Forced-on, warm, real-ish data:

| feat | BUILD (fold) | FIX+COMPACT | SCAN | dominant |
|-----:|-------------:|------------:|-----:|:---------|
| 60   | 70.1% | 7.9% | 22.0% | **BUILD** |
| 90   | 66.0% | 9.4% | 24.6% | **BUILD** |
| 120  | 66.8% | 9.2% | 24.0% | **BUILD** |

BUILD dominates. And BUILD already carries every shipped spike win — `fold_one_feature`
(lib.rs:200) dispatches narrow `BinColumn::U8/U16/U32` monomorphically (spike-004, -49%), reads
the once-gathered ords (spike-003), and is fused-branchless (003b, `debug_assert` only).
spike-011 closed fold-into-shared-buf. So the dominant bucket is already at its bit-exact floor.

### Task 2 — f64-pregather micro-lever A/B (the one unexplored sub-lever) = NULL
The fold does `f64::from(ord_g[k])` once PER FEATURE (num_features x num_rows widenings).
Pre-widening the once-gathered ords to `Vec<f64>` at gather time (num_rows widenings, once per
leaf) is bit-exact (f32->f64 is lossless). A/B (throwaway `bench_split_scan.rs njg_pregather_ab`):
- **Isolated cold fold microbench:** B (f64-pregather) **+7-14%, sign-stable** (60f +13.98%,
  90f +7.41%, 120f +8.60%).
- **End-to-end `bench_train` train-wall (forced-on, 3-run medians): FULL OVERLAP, NULL** —
  120f A 833.38 vs B 837.59ms; 90f A 718.73 vs B 720.32ms (+0.2-0.5% = noise).

The cold win evaporated warm: the 2x-byte f64 ord arrays are less cache-dense in the fold's
sequential read, cancelling the saved widening — the textbook cold-overstates-warm mirage the
spike rules warn about (ship on `bench_train`, never the isolated microbench). **Not shipped;
the pregather prototype was reverted** (hot path = only the inert profiler wraps).

## Bit-exact parity gate — GREEN both modes (independently re-verified)
- DEFAULT: `kernel_parity` 6/6, `learner_parity` 29/29, `raw_bin_train_matches_cpp_golden` 1/1.
- FORCED-ON (`LGBM_UNIFIED_BFS_THRESHOLD=0 LGBM_UNIFIED_SUBSCAN_THRESHOLD=0`): identical — 6/6,
  29/29, cpp-golden 1/1.
- `lgbm-compute --lib` 43/0, `lgbm-treelearner` 76/0; `cargo check`/`--examples` clean both crates.

## NULL-path scope confirmed
Hot-path footprint = ONLY the three inert `fusion_prof::time(BFS_BUILD/FIXCOMPACT/SCAN, ...)`
wraps (run identically when off) + NULL comments. No live pregather / f64-fold code. The A/B
function `njg_pregather_ab` lives in the throwaway `bench_split_scan.rs` (kept as the
measurement artifact; lib.rs comments reference it). Verified: `cargo build --examples` clean.

## What this closes
The per-feature WORK is at its bit-exact floor. With dpk (per-leaf cost is not allocation) and
e8e (not dispatch — work-bound), this NULL closes the entire sub-100-feature gate line: the
~100/130 gate is a fundamental cache-locality/work crossover that cannot drop without breaking
bit-exact f64 parity. The shipped wins (a48/b97 fusion, c5v core-scaling) stand.

## Deviations from Plan
None in intent — the plan's measure-first NULL branch. Task 2 ran the A/B (Task 1 showed BUILD
dominant, so the conversion sub-lever was worth the cheap A/B) and confirmed NULL.

## Commits
- `6ba130f` — Task 1: split fused per-feature WORK into BUILD/FIX+COMPACT/SCAN sub-buckets
- `574eaa4` — Task 2: f64-pregather A/B = NULL (cold-microbench mirage, not shipped)

## Self-Check: PASSED
- BUILD/FIX+COMPACT/SCAN profiler sub-buckets — FOUND in `fusion_prof.rs` + inert wraps in lib.rs.
- No live pregather function definitions in lib.rs — VERIFIED (grep: comments only).
- Parity green default + forced-on incl. cpp-golden — independently re-run.
- Commits 6ba130f / 574eaa4 — FOUND in git log.
