---
quick_id: 260620-njg
type: quick
title: Make the fused per-feature build/fix/compact/scan WORK cheaper (bit-exact) so the unified-fusion gate can drop below ~100/130 features
wave: 1
autonomous: false
files_modified:
  - crates/lgbm-compute/src/fusion_prof.rs
  - crates/lgbm-compute/src/lib.rs
  - crates/lgbm-treelearner/examples/bench_split_scan.rs
  - .planning/quick/260620-njg-make-the-fused-per-feature-build-fix-sca/260620-njg-FINDINGS.md
requirements:
  - QUICK-260620-njg
must_haves:
  truths:
    - "The forced-on fused per-feature region (build_fix_scan_impl, lib.rs:1497) is decomposed into THREE sub-buckets — BUILD (fold_one_feature), FIX+COMPACT (fix_histogram_inline + compact_histogram_inline), and SCAN (find_best_split) — at a representative width (e.g. 60/90/120 feat, ~20k rows), so the dominant WORK sub-bucket is known before any code change."
    - "The f64-pregather micro-lever (widen the once-gathered ord_g/ord_h to Vec<f64> at gather time so the per-feature fold reads pre-widened f64 instead of recomputing f64::from(f32) num_features times) is tested ONLY if Task 1 shows the BUILD conversion fraction is measurable — and is bit-exact (f32->f64 widening is lossless: identical f64 values either way)."
    - "Any shipped change stays BIT-EXACT to the C++ f64 anchor: kernel_parity + learner_parity + raw_bin_train_matches_cpp_golden green on DEFAULT and with the fusion FORCED ON (LGBM_UNIFIED_BFS_THRESHOLD=0 LGBM_UNIFIED_SUBSCAN_THRESHOLD=0)."
    - "The verdict is honest: ship ONLY on a sign-stable bench_train/bench_real train-wall WIN (no 3-run overlap), with the cold-microbench-overstates-warm rule applied; otherwise NULL — the per-feature work is documented at its bit-exact floor and the 'cheaper work' avenue is CLOSED (the cache-density caveat is recorded, no regression shipped)."
  artifacts:
    - path: ".planning/quick/260620-njg-make-the-fused-per-feature-build-fix-sca/260620-njg-FINDINGS.md"
      provides: "BUILD/FIX+COMPACT/SCAN decomposition table, the f64-pregather A/B (isolated build + end-to-end train-wall) with the cache-density caveat, and the WIN/NULL verdict"
      contains: "decomposition"
    - path: "crates/lgbm-compute/src/fusion_prof.rs"
      provides: "Three new inert sub-buckets (BFS_BUILD_NS / BFS_FIXCOMPACT_NS / BFS_SCAN_NS) splitting the existing BFS_PAR_NS work region"
      contains: "BFS_BUILD_NS"
  key_links:
    - from: "crates/lgbm-compute/src/lib.rs build_fix_scan_impl par_iter closure"
      to: "crates/lgbm-compute/src/fusion_prof.rs BFS_BUILD_NS/BFS_FIXCOMPACT_NS/BFS_SCAN_NS"
      via: "fusion_prof::time() wrapping fold_one_feature / fix+compact / find_best_split inside the par closure (thread-safe fetch_add, inert when off)"
      pattern: "BFS_BUILD_NS|fold_one_feature"
---

<objective>
Make the WORK inside the fused per-feature `build -> fix -> compact -> scan` region
(`build_fix_scan_impl`, lib.rs:1497) cheaper while preserving the project's #1 non-negotiable —
bit-exact CPU f64 parity — so the unified-fusion engagement gate (`unified_bfs_threshold` ~100 @16
cores, `unified_subscan_threshold` ~130) could drop below ~100/130 features and pull more
medium-width workloads into the fusion win.

This is a fail-fast, measurement-first spike with a STRONG prior of NULL. The premise (proven in
quick 260620-e8e) is that the per-leaf cost is WORK-bound, not dispatch-bound — so a cheaper
barrier was immaterial, and the only remaining lever is to cut the per-feature WORK itself. But the
spike campaign (001-013) already extracted the build wins and they are ALREADY PRESENT in the fused
path: spike-003 once-per-leaf gather (ord_g/ord_h gathered once, lib.rs:1525); spike-004 narrow
u8/u16 bins (fold_one_feature's monomorphic fold! over BinColumn::U8/U16/U32, lib.rs:200-216 — the
-49% build win IS here); spike-003b fused-branchless (debug_assert only, no per-element bound check);
spike-010/012 pool arena+reuse. spike-011 (INVALIDATED) closed the scatter-into-shared-buf design.
fix/compact are no-ops in the common case (most_freq_bin>0 / offset>0 guards). The fold and the
branchless reverse+forward scan are bit-exact-FIXED by parity. So the obvious WORK levers are SPENT.

The ONE genuinely-unexplored bit-exact micro-lever: the fold does `f64::from(ord_g[k])` /
`f64::from(ord_h[k])` INSIDE the per-feature loop (lib.rs:206-207), so the SAME row's f32->f64
widening is recomputed once PER FEATURE (num_features x num_rows conversions). Pre-widening the
once-gathered ord_g/ord_h to Vec<f64> AT GATHER TIME (num_rows conversions, once per leaf) is
bit-exact (f32->f64 widening is lossless — identical f64 either way) and removes the per-feature
reconversions. CAVEAT to measure honestly: f64 ord arrays are 2x the bytes of f32, so the fold's
sequential `ord_g[k]` read is less cache-dense — this likely trades conversion-elimination for worse
cache density and nets ~0 (which is probably WHY spike-003 gathers f32). That trade is the test.

Purpose: cheaper per-feature work would lower the fusion break-even; faithfulness over busy-work.
Output: a BUILD/FIX+COMPACT/SCAN decomposition, an honest f64-pregather A/B (only if warranted), and
a WIN/NULL verdict. A documented NULL that closes the "cheaper work" avenue (and, with dpk + e8e,
the whole sub-100-gate line) is the expected, fully-successful outcome. Do NOT manufacture a win; do
NOT ship a cache-density regression.

NON-NEGOTIABLE (project #1): bit-exact CPU f64 parity (esp. raw_bin_train_matches_cpp_golden +
learner_parity). Proven FORCED-ON via the parity gate (Task 3).
</objective>

<execution_context>
@$HOME/.claude/gsd-core/workflows/execute-plan.md
@$HOME/.claude/gsd-core/templates/summary.md
</execution_context>

<context>
@.planning/PROJECT.md
@.planning/STATE.md

# The fused smaller-child path + the fold + the inert profiler (verified by direct reading):
#   build_fix_scan_impl  lib.rs:1497 — once-per-leaf gather of f32 ord_g/ord_h (1521-1530, spike-003);
#                                      ranges validation (1536-1558); ONE rayon fork/join over
#                                      features (1566-1606): per-feature private vec![0.0f64;cells]
#                                      -> fold_one_feature (1576) -> fix_histogram_inline (1579) ->
#                                      compact_histogram_inline (1582) -> find_best_split (1585);
#                                      serial ordered assembly into buf (~1608).
#   fold_one_feature     lib.rs:200  — monomorphic fold! over BinColumn::U8/U16/U32 (spike-004 narrow
#                                      bins). The per-feature f64::from(ord_g[k]) widening is at 206-207
#                                      (the micro-lever's target). spike-011 INVALIDATED note 240-248.
#   fix_histogram_inline lib.rs:1772 — no-op for most_freq_bin==0.
#   compact_histogram_inline lib.rs:1800 — no-op for offset==0.
#   The BFS_PAR_NS bucket (fusion_prof.rs:24) currently lumps fold+fix+compact+scan as ONE region —
#   Task 1 splits it into BUILD/FIX+COMPACT/SCAN.
@crates/lgbm-compute/src/lib.rs

# LGBM_FUSION_PROF=1 inert env-gated per-bucket counters (gather/alloc/par + sub). The time()
# helper is thread-safe (fetch_add) so it is safe to call inside the par_iter closure. Add the
# three new BUILD/FIX+COMPACT/SCAN counters here.
@crates/lgbm-compute/src/fusion_prof.rs

# The Task-1 harness: bench_split_scan.rs already drives the FORCED-ON unified path under
# LGBM_FUSION_PROF=1 and dumps the buckets per feature count (the 260620-dpk sweep at ~180-225).
# Extend the dump + sweep widths and add the f64-pregather A/B here.
@crates/lgbm-treelearner/examples/bench_split_scan.rs
</context>

<tasks>

<task type="auto">
  <name>Task 1: PROFILE the BUILD-vs-FIX+COMPACT-vs-SCAN split inside the fused per-feature region</name>
  <files>crates/lgbm-compute/src/fusion_prof.rs, crates/lgbm-compute/src/lib.rs, crates/lgbm-treelearner/examples/bench_split_scan.rs, .planning/quick/260620-njg-make-the-fused-per-feature-build-fix-sca/260620-njg-FINDINGS.md</files>
  <action>
    Decompose the WORK region of the forced-on unified path. The existing BFS_PAR_NS bucket
    (fusion_prof.rs:24) lumps fold+fix+compact+scan into one timing region; split that work into
    THREE new inert sub-buckets so the dominant sub-bucket is known: add BFS_BUILD_NS,
    BFS_FIXCOMPACT_NS, and BFS_SCAN_NS to fusion_prof.rs (same AtomicU64 + behind the existing
    enabled() gate pattern; extend dump() to print them and their % of the par total; keep them inert
    when LGBM_FUSION_PROF is unset). In build_fix_scan_impl's par_iter closure (lib.rs:1572-1604) wrap
    fold_one_feature (1576) in fusion_prof::time(&BFS_BUILD_NS, ..), wrap fix_histogram_inline +
    compact_histogram_inline (1579-1582) in fusion_prof::time(&BFS_FIXCOMPACT_NS, ..), and wrap the
    find_best_split call (1585-1599) in fusion_prof::time(&BFS_SCAN_NS, ..). The time() helper uses a
    relaxed fetch_add and is thread-safe, so calling it from inside the rayon map is correct (each
    thread accumulates into the shared atomic); it runs the closure identically when the gate is off,
    so values/order are UNTOUCHED (parity unaffected — proven forced-on in Task 3). Extend the
    bench_split_scan.rs sweep (currently 20/40/60/80 at ~180-220) to add representative widths
    60/90/120 features at ~20k rows (warm, the existing cold-warmup-then-reset-counters pattern),
    reset/dump the three new buckets alongside the existing ones. Run forced-on and write to
    260620-njg-FINDINGS.md a per-feature-count table splitting the par region into
    BUILD/FIX+COMPACT/SCAN, with each as a % of the par total. State the lever decision: if BUILD
    dominates AND its dominant cost is the fold body (with FIX+COMPACT ~0, the no-op guards) then the
    only candidate sub-lever is the f64-pregather conversion (Task 2). If the per-feature f64::from
    conversions are NOT a measurable fraction of BUILD (the fold is bound by the bin gather + h[] RMW
    cache traffic, not the cheap f32->f64 widen), flag likely-NULL up front and lean Task 2 toward a
    throwaway-bench-only A/B. Do NOT change any hot-path logic in this task — measure only.
    Cite QUICK-260620-njg in the new counter doc comments.
  </action>
  <verify>
    <automated>cargo check -p lgbm-compute && LGBM_FUSION_PROF=1 LGBM_UNIFIED_BFS_THRESHOLD=0 LGBM_UNIFIED_SUBSCAN_THRESHOLD=0 cargo run -q -p lgbm-treelearner --example bench_split_scan 2>&1 | grep -Eiv '^#' | grep -Ei 'build|fix|compact|scan|feat' | head -20</automated>
  </verify>
  <done>FINDINGS.md contains a per-feature-count (60/90/120) decomposition splitting the fused par region into BUILD/FIX+COMPACT/SCAN (with %s), the dominant sub-bucket identified, and the f64-pregather lever decision recorded (pursue isolated A/B vs flag likely-NULL). Only the profiler changed; cargo check clean.</done>
</task>

<task type="auto">
  <name>Task 2: A/B the f64-pregather micro-lever (isolated build + end-to-end), bit-exact-gated</name>
  <files>crates/lgbm-treelearner/examples/bench_split_scan.rs, crates/lgbm-compute/src/lib.rs, .planning/quick/260620-njg-make-the-fused-per-feature-build-fix-sca/260620-njg-FINDINGS.md</files>
  <action>
    ONLY pursue the live A/B if Task 1 showed the per-feature f64::from conversions are a measurable
    fraction of BUILD; otherwise record the NULL directly in FINDINGS and skip the implementation
    (the profiler-only change from Task 1 is the deliverable). The micro-lever: widen the
    once-gathered ord_g/ord_h to Vec<f64> at GATHER TIME (lib.rs:1525-1530) so each row is converted
    num_rows times per leaf (once), and have fold_one_feature read the pre-widened f64 directly
    instead of recomputing f64::from(ord_g[k]) num_features x num_rows times. This is BIT-EXACT:
    f32->f64 widening is lossless, so the pre-widened f64 values are identical to the per-feature
    widened values, and the fold order (ascending leaf_rows) is unchanged — h[bin*2] / h[bin*2+1]
    accumulate identical f64 addends in identical order. Prefer a THROWAWAY A/B in bench_split_scan.rs
    (drive both fold variants over the same gathered data) over premature hot-path plumbing; only if
    the isolated A/B shows a sign-stable BUILD win, add the production change behind a minimal
    feature-gate/env-flag (e.g. LGBM_PREGATHER_F64) so it can be A/B'd live without committing the hot
    path. Measure TWO things and record both in FINDINGS: (a) the ISOLATED build A/B (f32-gather+
    per-feature-widen vs f64-pregather), and (b) the END-TO-END train-wall via bench_train / bench_real
    (3-run medians, warm). HONESTLY apply the cache-density caveat: f64 ord arrays are 2x the bytes,
    so the fold's sequential read is less cache-dense — if the isolated A/B is sub-noise or shows a
    cache-density regression (f64-pregather slower or within run spread), it is NULL: do NOT ship, do
    NOT plumb the hot path. The cold microbench overstates warm 3-7x — ship the verdict on bench_train/
    bench_real train-wall per the spike rules, never on the cold microbench alone. If (surprisingly)
    it is a sign-stable end-to-end win, keep it bit-exact-gated and proceed to Task 3 to re-check
    whether the fusion gate can drop. Do NOT touch the serial two-step fallback, subtract_scan/resident
    paths, or the RocmBackend override. Cite QUICK-260620-njg in any shipped comment.
  </action>
  <verify>
    <automated>cargo check -p lgbm-compute -p lgbm-treelearner && LGBM_UNIFIED_BFS_THRESHOLD=0 LGBM_UNIFIED_SUBSCAN_THRESHOLD=0 cargo test -p oracle-harness --test kernel_parity -- --test-threads=1 2>&1 | tail -5</automated>
  </verify>
  <done>FINDINGS.md records the isolated build A/B AND the end-to-end bench_train/bench_real train-wall A/B (3-run medians) with the cache-density caveat applied; either the f64-pregather is shipped behind a bit-exact gate on a sign-stable WIN (kernel_parity green forced-on), or it is declared NULL (sub-noise/cache-density regression) and NOT shipped. cargo check clean.</done>
</task>

<task type="checkpoint:human-verify" gate="blocking-human">
  <name>Task 3: HARD bit-exact parity gate + real-workload A/B confirmation (or NULL confirmation)</name>
  <what-built>
    A three-way BUILD/FIX+COMPACT/SCAN decomposition of the fused per-feature region (new inert
    fusion_prof sub-buckets), an honest f64-pregather A/B (isolated build + end-to-end train-wall with
    the cache-density caveat), and a WIN/NULL verdict. On WIN: a bit-exact-gated f64-pregather change.
    On NULL (the strong prior): only the profiler changed. This checkpoint proves bit-exact parity is
    intact and — if anything shipped — the d3v high-dim win is preserved before merge.
  </what-built>
  <how-to-verify>
    1. BIT-EXACT PARITY (default thresholds):
       `cargo test -p oracle-harness --test kernel_parity --test learner_parity -- --test-threads=1`
       — MUST be green, esp. `raw_bin_train_matches_cpp_golden` (the spike bit-exact gate).
    2. BIT-EXACT PARITY (fusion FORCED ON — proves the profiler instrumentation AND any f64-pregather
       change are bit-exact):
       `LGBM_UNIFIED_BFS_THRESHOLD=0 LGBM_UNIFIED_SUBSCAN_THRESHOLD=0 cargo test -p oracle-harness --test kernel_parity --test learner_parity -- --test-threads=1`
       — MUST be green (identical results to step 1). If a pregather env-gate shipped, ALSO run with it on.
    3. UNIT/LIB GREEN: `cargo test -p lgbm-compute --lib` and `cargo test -p lgbm-treelearner` green
       on DEFAULT and forced-on.
    4. CLEAN BUILD: `cargo check --workspace` (rocm/gfx1100 cells non-blocking — do not block on them).
    5. NULL path: confirm ONLY fusion_prof.rs (+ the three time() wrap sites + bench_split_scan.rs)
       changed — parity is trivially intact (profiler is inert when off); the FINDINGS verdict states
       the per-feature work is at its bit-exact floor (all spike build wins present; the f64-pregather
       micro-lever is sub-noise/cache-density-regressive) and the "cheaper work" avenue is CLOSED.
    6. WIN path (only if Task 2 shipped): re-run the d3v real-workload A/B (bench_real on binary.train +
       MNIST-784) and confirm the high-dim (784-feat) win is preserved or IMPROVED and typical/narrow
       stay neutral or improved (no regression vs current master / the d3v baseline). Confirm the
       FINDINGS verdict matches the measured numbers and no win was manufactured / no cache-density
       regression shipped.
    Do NOT git-add the LightGBM reference trees or the gitignored bench data. No new external dep
    (rayon/cubecl 0.10 already pinned).
  </how-to-verify>
  <resume-signal>Type "approved" once kernel_parity + learner_parity + raw_bin_train_matches_cpp_golden are green on default AND forced-on, the verdict is honest (NULL closes the avenue, or WIN preserves the d3v high-dim win with no regression), and nothing manufactured shipped; otherwise describe the failing gate.</resume-signal>
</task>

</tasks>

<threat_model>
## Trust Boundaries

| Boundary | Description |
|----------|-------------|
| fusion_prof time() sites <-> per-feature fused compute | Profiler wrappers inside the par closure must NOT alter values/order; inert when the env gate is off |
| f64-pregather ord arrays <-> fold_one_feature | Pre-widened f64 must be identical to per-feature widened f64 (lossless) and consumed in identical leaf_rows order |

## STRIDE Threat Register

| Threat ID | Category | Component | Disposition | Mitigation Plan |
|-----------|----------|-----------|-------------|-----------------|
| T-njg-01 | Tampering | BFS_BUILD/FIXCOMPACT/SCAN time() wraps inside the par_iter closure | mitigate | thread-safe relaxed fetch_add; closure runs identically when gate off; values/order untouched; proven FORCED-ON kernel_parity + learner_parity (Task 3 steps 1-2) |
| T-njg-02 | Tampering | f64-pregather ord_g/ord_h widening | mitigate | f32->f64 widening is lossless -> byte-identical f64 addends; fold order (ascending leaf_rows) unchanged; FORCED-ON + raw_bin_train_matches_cpp_golden proves bit-exact (Task 3) |
| T-njg-03 | Repudiation | manufactured win / shipped cache-density regression | accept->mitigate | hard sign-stable bench_train/bench_real adoption rule + cold-overstates-warm rule + honest NULL path (Task 2); human verdict review (Task 3 steps 5-6) |
| T-njg-SC | Tampering | npm/pip/cargo installs | mitigate | no new external dep (rayon/cubecl 0.10 already pinned); blocking-human checkpoint reviews any [ASSUMED]/[SUS] additions |
</threat_model>

<verification>
- kernel_parity + learner_parity + raw_bin_train_matches_cpp_golden green on DEFAULT and FORCED-ON.
- lgbm-compute --lib + lgbm-treelearner test suites green on default and forced-on.
- cargo check --workspace clean (rocm/gfx1100 cells non-blocking).
- FINDINGS verdict (WIN/NULL) consistent with measured numbers; NULL ships only the profiler; WIN
  preserves the d3v high-dim A/B; cache-density caveat applied, no regression shipped.
</verification>

<success_criteria>
- The fused per-feature region is decomposed into BUILD/FIX+COMPACT/SCAN sub-buckets; the dominant
  WORK sub-bucket is quantified at 60/90/120 features.
- The f64-pregather micro-lever is A/B'd (isolated build + end-to-end) ONLY if Task 1 warrants it,
  with the cache-density caveat honestly applied.
- Ship ONLY on a sign-stable bench_train/bench_real train-wall WIN (no 3-run overlap; cold-overstates-
  warm rule applied); otherwise NULL — per-feature work documented at its bit-exact floor and the
  "cheaper work" avenue CLOSED (with dpk + e8e, the sub-100-gate line).
- Blocking-human bit-exact parity (+ d3v on WIN) gate approved.
</success_criteria>

<output>
Create `.planning/quick/260620-njg-make-the-fused-per-feature-build-fix-sca/260620-njg-SUMMARY.md` when done (and `260620-njg-FINDINGS.md` with the BUILD/FIX+COMPACT/SCAN decomposition table, the f64-pregather A/B + cache-density caveat, and the WIN/NULL verdict).
</output>
