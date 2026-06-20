---
quick_id: 260620-dpk
type: quick
title: Lower the unified-fusion per-leaf fixed cost so the gate threshold can drop below 100 features
wave: 1
autonomous: false
files_modified:
  - crates/lgbm-compute/src/lib.rs
  - crates/lgbm-treelearner/src/learner.rs
  - crates/lgbm-treelearner/examples/bench_split_scan.rs
  - crates/lgbm/examples/bench_crossover.rs
requirements:
  - QUICK-260620-dpk
must_haves:
  truths:
    - "The fused per-leaf fixed cost (per-leaf allocations of ord_g/ord_h/ranges/out + the serial gather) is profiled and decomposed at 20/40/60/80 features into ALLOCATION vs FORK/JOIN-FLOOR fractions, so the lever's ceiling is known before any code changes."
    - "build_fix_scan_impl and subtract_scan_impl no longer allocate ord_g/ord_h/ranges/out fresh on every leaf invocation when the chosen lever is scratch-reuse (lever A) — they reuse caller-provided or thread-local buffers sized once."
    - "Both fused paths remain BIT-EXACT to the C++ f64 anchor: cargo test kernel_parity + learner_parity green on default AND with LGBM_UNIFIED_BFS_THRESHOLD=0 LGBM_UNIFIED_SUBSCAN_THRESHOLD=0 (fusion forced on)."
    - "The gate crossover is RE-MEASURED (warm, 3-run medians, feature sweep ~20-120) and the new unified_bfs_threshold/unified_subscan_threshold anchors (plus the c5v core-scaling formula re-anchor) reflect the measured break-even — a sign-stable WIN at the setpoint with no regression below it, or an honest NULL/partial that keeps the current threshold and documents the measured floor."
    - "The d3v real-workload A/B (binary.train + MNIST-784) confirms the high-dim win is preserved or improved and the typical/narrow datasets stay neutral or improved (no regression)."
  artifacts:
    - path: "crates/lgbm-compute/src/lib.rs"
      provides: "Reduced-fixed-cost build_fix_scan_impl/subtract_scan_impl + re-anchored unified_bfs_threshold/unified_subscan_threshold/core_scaled_threshold"
      contains: "build_fix_scan_impl"
    - path: ".planning/quick/260620-dpk-lower-the-unified-fusion-per-leaf-fixed-/260620-dpk-FINDINGS.md"
      provides: "Profiling decomposition table, lever choice, crossover table, and WIN/PARTIAL/NULL verdict"
      contains: "decomposition"
  key_links:
    - from: "crates/lgbm-treelearner/src/learner.rs"
      to: "crates/lgbm-compute/src/lib.rs build_fix_scan_impl"
      via: "reusable scratch threaded through the per-leaf call (mirrors hist_pool reuse, spike-012)"
      pattern: "build_fix_scan"
---

<objective>
Lower the per-leaf FIXED COST of the unified host fusion (`build_fix_scan_impl` smaller-child
build+fix+scan at lib.rs:1487, and `subtract_scan_impl` larger-child subtract+scan delegated from
lib.rs:1455) so the engagement gate (`unified_bfs_threshold` ~100 @16 cores,
`unified_subscan_threshold` ~130) can drop below 100 features WITHOUT re-introducing the narrow-config
regressions that 9cp/a48/b97 measured (+45-84% when the fusion was forced on narrow leaves).

The d3v end-to-end A/B showed the fusion is a clear WIN at 784 features (build+scan -27%/-67% range,
train-wall -6%) but neutral below ~100 features (gated off): below ~100 features the single rayon
fork/join + per-leaf setup is not amortized. If the per-leaf fixed cost drops, the break-even drops,
and medium-width (50-90 feat) workloads start winning.

Purpose: more workloads cross into the fusion win without re-regressing narrow leaves.
Output: a profiled fixed-cost decomposition, a bit-exact fixed-cost reduction (most likely
scratch-reuse, possibly + fork/join chunking), re-anchored gate thresholds, and an honest
WIN/PARTIAL/NULL verdict — never a manufactured win.

NON-NEGOTIABLE (project #1): bit-exact CPU f64 parity. Scratch reuse must not change values or order;
any chunking must preserve per-feature result ORDER and per-feature independence. Proven FORCED-ON via
the parity gate (Task 4).
</objective>

<execution_context>
@$HOME/.claude/gsd-core/workflows/execute-plan.md
@$HOME/.claude/gsd-core/templates/summary.md
</execution_context>

<context>
@.planning/PROJECT.md
@.planning/STATE.md

# The two fused per-leaf paths and the gate helpers (verified by direct reading):
#   build_fix_scan_impl  lib.rs:1487  — per-leaf ord_g/ord_h Vec::with_capacity + serial
#                                       gather (1510-1516), ranges Vec (1521), per-feature
#                                       private `vec![0.0f64; cells]` (1555), serial assembly (1591).
#   subtract_scan_impl   (delegated from subtract_scan lib.rs:1455) — per-leaf ranges + out.
#   unified_bfs_threshold lib.rs:430   = core_scaled_threshold(100, rayon_cores())
#   unified_subscan_threshold lib.rs:473 = core_scaled_threshold(130, rayon_cores())
#   core_scaled_threshold lib.rs:377   — the c5v additive-log core-scaling formula to re-anchor.
@crates/lgbm-compute/src/lib.rs

# The learner already holds interior-mutable reusable scratch (RefCell<...> fields ~225-321)
# and a HistogramPool reused across trees (hist_pool, take/restore at 841/1082) — the
# spike-012 reuse shape that lever A mirrors. The fused calls are wired ~1608-1811.
@crates/lgbm-treelearner/src/learner.rs

# LGBM_PHASE_PROF=1 phase counters (inert otherwise) — the measurement instrument for Task 1.
@crates/lgbm-treelearner/src/phase_prof.rs
</context>

<tasks>

<task type="auto">
  <name>Task 1: PROFILE the per-leaf fixed-cost breakdown at low feature counts</name>
  <files>crates/lgbm-treelearner/examples/bench_split_scan.rs, .planning/quick/260620-dpk-lower-the-unified-fusion-per-leaf-fixed-/260620-dpk-FINDINGS.md</files>
  <action>
    Decompose the per-leaf FIXED COST of the forced-on unified path at low feature counts
    (sweep 20/40/60/80 features, ~20k rows, warm, 3-run medians). Split per-leaf wall time into four
    buckets: (1) SERIAL GATHER — the ord_g/ord_h `for &row in leaf_rows` loop at lib.rs:1513;
    (2) ALLOCATIONS — Vec::with_capacity for ord_g/ord_h/ranges plus the per-feature
    `vec![0.0f64; cells]` and output Vec (the lever-A-reducible cost); (3) FORK/JOIN DISPATCH — the
    par_iter().zip(...).map(...) enter/exit floor (the likely IRREDUCIBLE hard floor); (4) ACTUAL
    PARALLEL WORK — fold+fix+compact+scan. Instrument via LGBM_PHASE_PROF=1 counters (add narrow
    per-bucket counters around the gather, the alloc sites, and the par region in build_fix_scan_impl
    and subtract_scan_impl behind the phase_prof guard so they stay inert when off) and/or a focused
    micro-bench in bench_split_scan.rs that drives the forced-on path directly. Quantify, per feature
    count, what FRACTION of the sub-100-feat fixed cost is ALLOCATION (reducible by lever A) vs
    FORK/JOIN FLOOR (irreducible). Write the decomposition table to 260620-dpk-FINDINGS.md and state
    the lever decision: if allocation is material (>~20%) pursue lever A; if the serial gather is
    material at low feature counts consider lever C; if the fork/join floor dominates and alloc is
    negligible flag likely-NULL/partial up front. Do NOT change any hot-path logic in this task —
    measure only. Profiling-only counters added under the phase_prof guard MUST NOT alter values or
    order (parity unaffected).
  </action>
  <verify>
    <automated>LGBM_PHASE_PROF=1 cargo run -q -p lgbm-treelearner --example bench_split_scan 2>&1 | grep -Eiv '^#' | grep -Ei 'gather|alloc|fork|join|feat' | head</automated>
  </verify>
  <done>FINDINGS.md contains a per-feature-count (20/40/60/80) decomposition table splitting fixed cost into gather/alloc/forkjoin/work, with the allocation-vs-floor fraction quantified and the lever(s) chosen (A and/or B and/or C, or likely-NULL flagged).</done>
</task>

<task type="auto">
  <name>Task 2: IMPLEMENT the chosen fixed-cost reduction, bit-exact on both fused paths</name>
  <files>crates/lgbm-compute/src/lib.rs, crates/lgbm-treelearner/src/learner.rs</files>
  <action>
    Implement the Task-1 lever(s), keeping build_fix_scan_impl AND subtract_scan_impl bit-exact.
    LEVER A (most likely): replace the per-leaf Vec::with_capacity allocations (ord_g/ord_h at
    lib.rs:1511-1512, ranges at 1521, and the output/private buffers) with reusable scratch sized once
    and reused across every leaf. Prefer a learner-held reusable scratch threaded through the per-leaf
    call, mirroring the spike-012 hist_pool take/restore plumbing (learner.rs 841/1082) and the existing
    RefCell interior-mutable fields — clear() and reuse capacity, never reallocate. If threading the
    per-feature private `vec![0.0f64; cells]` through the learner is too invasive, use a thread-local
    buffer pool inside lgbm-compute for that per-task histogram. Values are pushed and consumed in the
    SAME order as today (ord_g/ord_h fill in leaf_rows order; ranges in ascending feature index; serial
    assembly in feature-index order) so the fold inputs and argmax are byte-identical — exactly the
    spike-012 reuse argument. LEVER B (only if Task 1 shows fork/join granularity is the lever): replace
    one-task-per-feature par_iter with .with_min_len(...) or core-sized par_chunks so task_count ~= cores;
    each chunk processes features in ascending index and writes results in input order, and collect
    preserves chunk order, so per-feature ORDER and independence are preserved (bit-exact). LEVER C (only
    if the serial gather is material): parallelize/hoist the ord_g/ord_h gather while keeping leaf_rows
    order. Do NOT touch the serial two-step fallback (the feats.len() < threshold branch), the
    RocmBackend fused-launch override, or the resident/subtract_resident paths. Keep the hot path minimal.
    Cite QUICK-260620-dpk in the impl comment headers.
  </action>
  <verify>
    <automated>cargo check -p lgbm-compute -p lgbm-treelearner && LGBM_UNIFIED_BFS_THRESHOLD=0 LGBM_UNIFIED_SUBSCAN_THRESHOLD=0 cargo test -p oracle-harness --test kernel_parity -- --test-threads=1 2>&1 | tail -5</automated>
  </verify>
  <done>Both fused paths reuse buffers (no per-leaf alloc) and/or chunk the fork/join per the chosen lever; cargo check clean; kernel_parity green with the fusion forced on, proving the reduction is bit-exact.</done>
</task>

<task type="auto">
  <name>Task 3: RE-MEASURE the gate crossover and set the new thresholds (or honest NULL)</name>
  <files>crates/lgbm/examples/bench_crossover.rs, crates/lgbm-compute/src/lib.rs, .planning/quick/260620-dpk-lower-the-unified-fusion-per-leaf-fixed-/260620-dpk-FINDINGS.md</files>
  <action>
    Re-measure the gate crossover with the lowered-cost fusion: warm, 3-run medians, sweep features
    ~20-120 (e.g. 20/40/60/80/90/100/120) at ~20k rows, A = serial two-step, B = lowered-cost unified,
    via bench_crossover.rs. Find the NEW break-even for BOTH children separately (smaller-child build+
    fix+scan -> unified_bfs anchor; larger-child subtract+scan -> unified_subscan anchor). HARD adoption
    rule (same as a48/b97): the new (lower) threshold must show a SIGN-STABLE train-wall WIN at its
    setpoint (no run overlap across all 3 runs) AND no regression below it. If both gates are met, set
    the new anchors: update the constant inside unified_bfs_threshold (currently core_scaled_threshold(100,..))
    and unified_subscan_threshold (currently core_scaled_threshold(130,..)) to the new measured anchors,
    and re-anchor core_scaled_threshold's c5v formula if the crossover SHAPE moved (re-derive the slope
    only if the new sweep contradicts the existing curve; otherwise keep the slope and move only the
    anchor). Update the doc comments and the threshold unit-test expectations (lib.rs ~2999-3007). If the
    fork/join floor is irreducible and the gate cannot drop materially, record an honest NULL/PARTIAL:
    KEEP the current thresholds, document the measured floor and why it caps the gate, and do NOT
    manufacture a win. Write the full crossover table and the WIN/PARTIAL/NULL verdict to FINDINGS.md.
    Verdict criterion: WIN = bfs and/or subscan anchor drops with a sign-stable setpoint win and no
    sub-threshold regression; PARTIAL = one child drops, the other holds; NULL = neither drops (floor
    dominates), thresholds unchanged.
  </action>
  <verify>
    <automated>cargo run -q -p lgbm --example bench_crossover 2>&1 | grep -Eiv '^#' | grep -Ei 'feat|cross|anchor|wall|win|null' | head -30</automated>
  </verify>
  <done>FINDINGS.md has the ~20-120 feature crossover table (A vs B, 3-run medians) and a WIN/PARTIAL/NULL verdict; thresholds + doc comments + unit-test expectations are updated to the new anchors on WIN/PARTIAL, or explicitly left unchanged with the measured floor documented on NULL.</done>
</task>

<task type="checkpoint:human-verify" gate="blocking-human">
  <name>Task 4: HARD parity gate + real-workload A/B confirmation</name>
  <what-built>
    A lowered per-leaf fixed cost on both unified fused paths (build_fix_scan_impl, subtract_scan_impl),
    re-anchored gate thresholds (or a documented NULL), and the FINDINGS verdict. This checkpoint proves
    the reduction stayed bit-exact and the d3v high-dim win is preserved before merge.
  </what-built>
  <how-to-verify>
    1. BIT-EXACT PARITY (default thresholds):
       `cargo test -p oracle-harness --test kernel_parity --test learner_parity -- --test-threads=1`
       — MUST be green.
    2. BIT-EXACT PARITY (fusion FORCED ON — proves scratch-reuse/chunking is bit-exact):
       `LGBM_UNIFIED_BFS_THRESHOLD=0 LGBM_UNIFIED_SUBSCAN_THRESHOLD=0 cargo test -p oracle-harness --test kernel_parity --test learner_parity -- --test-threads=1`
       — MUST be green (identical results to step 1).
    3. UNIT/LIB GREEN: `cargo test -p lgbm-compute --lib` and `cargo test -p lgbm-treelearner` green.
    4. CLEAN BUILD: `cargo check --workspace` (rocm cells gfx1100-only/non-blocking — do not block on them).
    5. REAL-WORKLOAD A/B (d3v): re-run bench_real.rs on binary.train + MNIST-784. Confirm the high-dim
       (784-feat) win is preserved or IMPROVED, and the typical/narrow datasets are still neutral or
       improved (no regression vs current master). Compare against the d3v baseline numbers in
       PROJECT.md / prior SUMMARY.
    6. Confirm FINDINGS.md WIN/PARTIAL/NULL verdict matches the measured numbers and that on NULL the
       thresholds were left unchanged (no manufactured win).
    Do NOT git-add the LightGBM reference trees or the gitignored bench data.
  </how-to-verify>
  <resume-signal>Type "approved" once all parity tests are green on default AND forced-on, the d3v high-dim win is preserved, and the verdict is honest; otherwise describe the failing gate.</resume-signal>
</task>

</tasks>

<threat_model>
## Trust Boundaries

| Boundary | Description |
|----------|-------------|
| reusable scratch ↔ per-leaf fused compute | Reused buffers crossing leaf invocations must be cleared/resized correctly or stale values leak into a fold |
| rayon chunk boundaries ↔ per-feature results | Re-granularized fork/join must keep per-feature order + independence |

## STRIDE Threat Register

| Threat ID | Category | Component | Disposition | Mitigation Plan |
|-----------|----------|-----------|-------------|-----------------|
| T-dpk-01 | Tampering | reusable ord_g/ord_h/ranges/out scratch | mitigate | clear() + capacity-reuse only; fill/consume in identical order; prove via FORCED-ON kernel_parity + learner_parity (Task 4 steps 1-2) |
| T-dpk-02 | Tampering | rayon chunk re-granularization (lever B) | mitigate | each chunk walks features ascending, writes in input order, collect preserves order; forced-on parity covers it |
| T-dpk-03 | Information disclosure | stale scratch across leaves of different sizes | mitigate | size/clear scratch per leaf before fill; debug_assert lengths; covered by learner_parity across multi-leaf trees |
| T-dpk-04 | Repudiation | manufactured/overstated gate-drop win | accept→mitigate | hard sign-stable adoption rule + honest NULL path (Task 3); human verdict review (Task 4 step 6) |
| T-dpk-SC | Tampering | npm/pip/cargo installs | mitigate | no new external dep (rayon/cubecl 0.10 already pinned); blocking-human checkpoint reviews any [ASSUMED]/[SUS] additions |
</threat_model>

<verification>
- kernel_parity + learner_parity green on DEFAULT and FORCED-ON (LGBM_UNIFIED_*_THRESHOLD=0).
- lgbm-compute --lib + lgbm-treelearner test suites green.
- cargo check --workspace clean (rocm/gfx1100 cells non-blocking).
- d3v real-workload A/B: high-dim (784) win preserved/improved; typical/narrow neutral/improved.
- FINDINGS verdict (WIN/PARTIAL/NULL) consistent with measured numbers; NULL leaves thresholds unchanged.
</verification>

<success_criteria>
- Per-leaf fixed cost profiled and decomposed (alloc vs fork/join floor) at 20/40/60/80 feat.
- Both fused paths' per-leaf allocations reduced (scratch-reuse) and/or fork/join re-granularized — bit-exact, proven forced-on.
- Gate crossover re-measured; new lower anchors set on a sign-stable WIN with no sub-threshold regression, OR an honest NULL/partial that keeps the current threshold and documents the measured floor.
- Blocking-human parity + d3v gate approved.
</success_criteria>

<output>
Create `.planning/quick/260620-dpk-lower-the-unified-fusion-per-leaf-fixed-/260620-dpk-SUMMARY.md` when done (and `260620-dpk-FINDINGS.md` with the decomposition table, crossover table, and verdict).
</output>
