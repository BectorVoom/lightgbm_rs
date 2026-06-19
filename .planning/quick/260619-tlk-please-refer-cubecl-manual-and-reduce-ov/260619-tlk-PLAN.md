---
phase: quick-260619-tlk
plan: 01
type: execute
wave: 1
depends_on: []
files_modified:
  - crates/lgbm-compute/examples/batched_read_audit_ab.rs
  - .planning/quick/260619-tlk-please-refer-cubecl-manual-and-reduce-ov/260619-tlk-FINDINGS.md
autonomous: true
requirements: []
must_haves:
  truths:
    - "The cubecl 0.10 manual's overhead-reduction levers (launch_unchecked, single-handle read_one, batched client.read of a Vec of handles, deferred/lazy sync) are each cited and mapped to the production GPU path's current state."
    - "An explicit, evidence-backed verdict is recorded: whether ANY non-redundant batched-read / deferred-sync / round-trip overhead lever remains in the PRODUCTION dispatch/read path."
    - "If a genuinely-new seam is found, it is A/B-confirmed on gfx1100 (sign-stable, spread-separated) BEFORE any wiring; if none is found, that is stated plainly with the audit evidence rather than a change being invented."
    - "The cubecl-cpu f64-fold bit-exact anchor and the cubecl-hip 1e-6 envelope are untouched — no production kernel, launcher, or anchor is modified."
  artifacts:
    - path: ".planning/quick/260619-tlk-please-refer-cubecl-manual-and-reduce-ov/260619-tlk-FINDINGS.md"
      provides: "The cubecl-manual-grounded audit + verdict + (if applicable) A/B result"
    - path: "crates/lgbm-compute/examples/batched_read_audit_ab.rs"
      provides: "A rocm-gated measurement-only A/B confirming the production read pattern vs the manual's batched-read idiom (or a documented confirmation harness)"
  key_links:
    - from: "260619-tlk-FINDINGS.md"
      to: "crates/lgbm-compute/src/lib.rs and src/kernels"
      via: "production read/dispatch audit citations"
      pattern: "read_one_unchecked|build_fix_scan_resident|subtract_resident"
---

<objective>
"Refer the cubecl manual and reduce overhead in the GPU kernel."

This is the latest in a long, mostly-closed GPU-overhead campaign (spikes 001-013;
quick tasks nrw/ol8/p93/ngo/mwr/j9t/q2z/sgu). The user's explicit ask has two halves:
(1) consult the cubecl manual, and (2) reduce GPU kernel overhead in production.

The hard, already-established fact (STATE.md sgu reconcile, 2026-06-19; the
gpu-lazy-dispatch-deferred-sync-win memory): the PRODUCTION per-leaf GPU histogram
path was ALREADY collapsed to ONE batched fused launch in 260608. The per-feature
submit-then-block loop and the read-per-handle loop that the prior q2z deferred-sync
spike modeled are TEST-ONLY (construct_histograms_parallel_f32_on has zero production
callers). So the cubecl-manual levers the user is pointing at have, in large part,
already been applied.

Purpose: honor the user's ask FAITHFULLY — re-read the cubecl 0.10 manual, RE-AUDIT the
production dispatch/read path against its overhead-reduction idioms, and either find a
genuinely-new non-redundant lever (A/B it before wiring) OR state plainly, with evidence,
that production already batches everything. Per the task constraints: "Faithfulness over
busy-work" — do NOT manufacture a win by re-introducing a pattern production already beats.

Output:
- 260619-tlk-FINDINGS.md: the manual-grounded audit, the per-launcher round-trip
  inventory, the verdict, and (if any candidate seam exists) the gfx1100 A/B result.
- batched_read_audit_ab.rs: a rocm-gated, measurement-only example that either confirms
  a candidate seam's win or documents that the production single-read pattern is already
  the manual's idiomatic form.
- NO change to any production kernel, launcher, learner dispatch, or the CPU f64 anchor.
</objective>

<execution_context>
@$HOME/.claude/gsd-core/workflows/execute-plan.md
@$HOME/.claude/gsd-core/templates/summary.md
</execution_context>

<context>
@.planning/STATE.md
@./CLAUDE.md
@crates/lgbm-compute/src/lib.rs
@crates/lgbm-compute/src/kernels/histogram.rs
@crates/lgbm-compute/src/kernels/split.rs
@crates/lgbm-compute/src/kernels/subtract.rs
@crates/lgbm-compute/src/kernels/partition.rs
@crates/lgbm-compute/examples/lazy_dispatch_ab.rs
@crates/lgbm-treelearner/src/learner.rs
@crates/lgbm-treelearner/src/resident_pool.rs

# cubecl 0.10 manual (the explicit user ask). Fetch via ctx7 / context7 MCP:
#   library: /tracel-ai/cubecl  (resolved; High reputation, 259 snippets)
#   docs query: "reduce kernel launch overhead, batched client.read of multiple
#                handles, deferred/lazy execution, avoid per-call host-device
#                round-trips, launch_unchecked, autotune"
# Crate is pinned to cubecl 0.10.0 — fetch version-appropriate guidance and CITE it.
</context>

<prior_art_do_not_repeat>
The following overhead levers are ALREADY RESOLVED/INVALIDATED in project memory — do NOT
re-spike or re-plan them (read CLAUDE.md "Project Skills" + the MEMORY entries):

- ROW-PARTITION histogram build: SHIPPED (P=16). Do not re-add.
- REGISTER-BATCHING, MULTI-FEATURE PACKING: spiked NULL. Do not pursue.
- 16-bit DISCRETIZED histogram: INVALIDATED for exact parity (approximate-only).
- launch_unchecked: SHIPPED across all 8 rocm histogram kernels (nrw); benched (ol8) —
  NULL for atomic-class kernels, real only for the fused f64 path (already wired).
- Plane warp-aggregated atomics (p93): NULL, kept as un-wired primitive.
- CUDA-mirror resident kernel (mwr/ngo): leave as primitive; do NOT replace the wired LDS path.
- Per-leaf round-trip elimination (nn7/oib/p90/t3t): SHIPPED — device-resident pool, fused
  build+fix+compact+scan = ONE launch/leaf, device-resident subtract, no per-leaf readback.
- Deferred-sync / lazy execution (q2z): bench WIN was vs a TEST-ONLY per-feature baseline;
  production already single-batched-launch (sgu reconcile) so WIRE is MOOT.

The ONE recorded-but-unconfirmed thread (sgu addendum): the idiomatic
client.read(Vec of handles) batch-read surfaced a sign-stable bench-only win that a
read-per-handle N-loop masks — BUT sgu also recorded that production has NO such
read-per-handle loop. This plan's job is to CONFIRM that audit at the production level,
not to manufacture a loop to beat.
</prior_art_do_not_repeat>

<tasks>

<task type="auto">
  <name>Task 1: Re-fetch the cubecl 0.10 manual + inventory every production launcher's round-trip pattern</name>
  <files>.planning/quick/260619-tlk-please-refer-cubecl-manual-and-reduce-ov/260619-tlk-FINDINGS.md</files>
  <action>
    Consult the cubecl 0.10 manual via the find-docs skill / context7 MCP (library
    /tracel-ai/cubecl, version-appropriate to the pinned 0.10.0). Query specifically for
    the overhead-reduction idioms: launch_unchecked, single-handle read_one / read_one_unchecked,
    batched client.read of a Vec of handles, deferred/lazy execution and when a sync is forced,
    avoiding per-call host-device round-trips, allocation reuse (client.empty / create_from_slice),
    and autotune. Record the exact manual recommendations with their source attribution.

    Then build a PRODUCTION round-trip inventory. Grep crates/lgbm-compute/src for every
    read_one_unchecked / client.read call (already located: histogram.rs lines
    188/373/467/693/882/1042/1131/1491/1629/2012/2222/2677, split.rs 838/1078/1291/1714,
    subtract.rs 116/228, partition.rs 266) and, for EACH production launcher (NOT the
    test-only construct_histograms_parallel_f32_on / construct_hist_kernel_atomic_f32 bench
    paths), record: how many kernel launches it issues, how many distinct out-handles it
    leaves unread, and how many read calls it makes. Cross-reference the LEARNER-side per-leaf
    sequencing (learner.rs around 1503-1620 split_inner + scan_leaf_histogram, and
    resident_pool.rs) to confirm what crosses the bus per leaf: the fused build_fix_scan_resident
    (ONE launch per leaf: build then fix then compact then scan), the device-resident
    subtract_resident (no readback), and only the small SplitInfo cells read back.

    Write the manual citations + the inventory table into 260619-tlk-FINDINGS.md. The
    inventory must make the Task 2 verdict falsifiable: if any production launcher leaves
    two or more out-handles unread before a sync, OR the learner issues N per-leaf launches
    that block sequentially, that is a candidate batched-read / deferred-sync seam; if every
    launcher is one-launch then one-consolidated-read and the per-leaf path is one fused launch,
    there is no seam and the manual's batched-read idiom is already satisfied.

    Do NOT edit any production source in this task — audit + document only.
  </action>
  <verify>
    <automated>test -f .planning/quick/260619-tlk-please-refer-cubecl-manual-and-reduce-ov/260619-tlk-FINDINGS.md && grep -qi launch_unchecked .planning/quick/260619-tlk-please-refer-cubecl-manual-and-reduce-ov/260619-tlk-FINDINGS.md && grep -qi "client.read" .planning/quick/260619-tlk-please-refer-cubecl-manual-and-reduce-ov/260619-tlk-FINDINGS.md</automated>
  </verify>
  <done>FINDINGS.md exists with the cited cubecl 0.10 manual overhead-reduction idioms AND a per-production-launcher round-trip inventory (launches issued, out-handles left unread, read calls), plus the per-leaf learner sequencing summary. The inventory makes the Task 2 verdict falsifiable.</done>
</task>

<task type="auto">
  <name>Task 2: Record the evidence-backed verdict + (if a seam exists) a rocm-gated A/B; protect the parity contract</name>
  <files>crates/lgbm-compute/examples/batched_read_audit_ab.rs, .planning/quick/260619-tlk-please-refer-cubecl-manual-and-reduce-ov/260619-tlk-FINDINGS.md</files>
  <action>
    From the Task 1 inventory, decide the verdict and act FAITHFULLY.

    CASE A — NO remaining non-redundant lever (EXPECTED, per the sgu reconcile: every
    production launcher is one-launch then one-consolidated read_one_unchecked, and the per-leaf
    path is the single fused build_fix_scan_resident launch with a device-resident subtract).
    Then state this PLAINLY in FINDINGS.md with the audit evidence, and write
    batched_read_audit_ab.rs as a measurement-only, rocm-gated CONFIRMATION harness that
    demonstrates the equivalence the manual predicts: that for the production shape (ONE
    consolidated out-handle per launcher) client.read_one_unchecked(h) and the batched
    client.read(vec![h]) are the same single drain — i.e. there is no N-handle loop for the
    batched idiom to collapse. Print the manual citation and the verdict at the top of the
    example. The example must compile under --features rocm and no-op-print under the default
    cpu feature (mirror the cfg(not(feature = rocm)) fn main guard from lazy_dispatch_ab.rs).
    Do NOT manufacture a per-handle loop just to beat it (the q2z/sgu anti-pattern). NO
    production source, kernel, launcher, learner, or CPU anchor is edited.

    CASE B — a genuinely-new candidate seam IS found (e.g. a production launcher leaves two or
    more out-handles unread, or the learner issues N sequential blocking per-leaf launches).
    Then write batched_read_audit_ab.rs as an INTERLEAVED A/B on the real gfx1100 modeled on
    lazy_dispatch_ab.rs's rigor: WARMUP-discarded, median + p25/p75 spread, arms interleaved,
    a same-input numeric guard (the batched read must not change values), a note to re-run
    across two or more process restarts, and an honest disposition rule (a delta within spread
    or sign-flipping across restarts is NULL). Record the A/B numbers + a WIRE / DO-NOT-WIRE
    verdict in FINDINGS.md. Even in Case B, this task only MEASURES + recommends — any actual
    wiring (a learner/launcher refactor + end-to-end parity re-validation) is a SEPARATE
    follow-up plan, OUT OF SCOPE here.

    In BOTH cases: the parity contract is the gate. Run the existing parity tests to prove the
    audit/example introduced no regression to the bit-exact f64 anchor or the 1e-6 hip envelope.
    Record the green test counts in FINDINGS.md.
  </action>
  <verify>
    <automated>cargo build -p lgbm-compute --example batched_read_audit_ab && cargo test -p lgbm-compute --lib && cargo test -p oracle-harness --test kernel_parity</automated>
  </verify>
  <done>FINDINGS.md records the explicit verdict (Case A: no remaining non-redundant production lever, with the per-launcher evidence + the manual citation that the single-handle read already IS the idiomatic batched form; OR Case B: the candidate seam, its gfx1100 A/B numbers, and a WIRE/DO-NOT-WIRE recommendation). batched_read_audit_ab.rs compiles (rocm + default-feature no-op guard). lgbm-compute --lib and oracle-harness kernel_parity are GREEN (counts recorded), proving the bit-exact f64 anchor and the 1e-6 hip envelope are unregressed. No production kernel/launcher/learner/anchor was modified.</done>
</task>

</tasks>

<threat_model>
## Trust Boundaries

| Boundary | Description |
|----------|-------------|
| host -> GPU (cubecl-hip) | Launch args + out-handle reads cross to gfx1100; numeric envelope is the trust contract |
| audit doc -> production code | The FINDINGS verdict must reflect the real production source, not the test-only bench paths |

## STRIDE Threat Register

| Threat ID | Category | Component | Disposition | Mitigation Plan |
|-----------|----------|-----------|-------------|-----------------|
| T-tlk-01 | Tampering | the bit-exact CPU f64 anchor + 1e-6 hip envelope | mitigate | No production kernel/launcher/anchor edited (measurement + docs only); Task 2 verify runs lgbm-compute --lib + oracle-harness kernel_parity and records green counts |
| T-tlk-02 | Information disclosure (false win) | FINDINGS verdict | mitigate | Verdict must cite the REAL production launchers (one launch + one consolidated read each; one fused launch per leaf); explicitly forbid manufacturing a per-handle loop to beat (the q2z/sgu anti-pattern) |
| T-tlk-03 | Denial of service (build break) | batched_read_audit_ab.rs | mitigate | Example mirrors lazy_dispatch_ab.rs's cfg(not(feature=rocm)) no-op main; Task 2 verify builds the example |
| T-tlk-SC | Tampering | npm/pip/cargo installs | accept | No new package-manager installs — only an example file + a docs note; cubecl 0.10.0 already pinned |
</threat_model>

<verification>
- `cargo build -p lgbm-compute --example batched_read_audit_ab` compiles (default cpu feature).
- `cargo test -p lgbm-compute --lib` GREEN (bit-exact + BinColumn guards unregressed).
- `cargo test -p oracle-harness --test kernel_parity` GREEN (the f64 bit-exact anchor + the
  1e-6 hip-envelope gate the CLAUDE.md contract names; on a non-rocm runner the hip cells are
  feature-gated out — record which cells actually ran).
- FINDINGS.md contains the cubecl-manual citations, the production round-trip inventory, and
  the explicit Case A / Case B verdict.
- `LightGBM/` is never git-added.
</verification>

<success_criteria>
- The cubecl 0.10 manual is consulted and its overhead-reduction idioms cited in FINDINGS.md.
- The production dispatch/read path is audited launcher-by-launcher and the verdict is
  evidence-backed and falsifiable.
- Either a genuinely-new lever is A/B-confirmed on gfx1100 before any wiring, OR it is stated
  plainly (with evidence) that production already batches everything — no invented change, no
  manufactured win.
- The bit-exact f64 anchor and the 1e-6 hip envelope are provably unregressed (green parity
  tests). No production kernel, launcher, learner, or CPU anchor is modified.
</success_criteria>

<output>
Create `.planning/quick/260619-tlk-please-refer-cubecl-manual-and-reduce-ov/260619-tlk-SUMMARY.md` when done.
</output>
