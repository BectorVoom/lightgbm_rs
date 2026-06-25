---
phase: quick-260625-obl
plan: 01
type: execute
wave: 1
depends_on: []
files_modified:
  - .claude/skills/spike-findings-lightgbm_rs/references/gpu-scan-roundtrip-copack.md
  - .planning/STATE.md
autonomous: true
requirements: [spike-024-wire]
must_haves:
  truths:
    - "Ground truth on spike-024 wiring is established by grep, not by trusting stale prose"
    - "The co-pack path engages by default (LGBM_SIBLING_COPACK unset) when the structural gate holds"
    - "All bit-exact gates pass with the co-pack path OFF (LGBM_SIBLING_COPACK=0) — the default path is unchanged"
    - "All bit-exact gates pass with the co-pack path ON (LGBM_SIBLING_COPACK=1) — co-pack is byte-identical"
    - "No committed oracle golden changed (a golden change would be a RED FLAG, not auto-accept)"
    - "Records (skill reference + STATE) reflect the verified ground truth"
  artifacts:
    - path: crates/lgbm-treelearner/src/learner.rs
      provides: "co-pack call site scan_resident_siblings (read-only this task — already wired ~line 1839)"
    - path: crates/lgbm-compute/src/kernels/split.rs
      provides: "find_best_splits_fused_siblings_kernel + find_best_splits_fused_siblings_from_handles_on (read-only — already present)"
  key_links:
    - from: crates/lgbm-treelearner/src/learner.rs
      to: crates/lgbm-compute/src/lib.rs
      via: "Backend::scan_resident_siblings → find_best_splits_fused_siblings_from_handles_on"
      pattern: "scan_resident_siblings"
---

<objective>
Wire spike-024's sibling-scan co-pack into the GPU tree learner so both children of a
split are scanned in ONE launch+readback instead of two (≈59→≈30 scan-readback
syncs/tree).

**Purpose:** Close the records discrepancy flagged by the task brief and confirm the
spike-024 co-pack is live + bit-exact, OR perform the wiring if ground truth shows it is
not. The brief's "fresh source read" suggested the kernel/launcher exist in split.rs but
are NOT called from learner.rs. **Step 0 (Task 1) resolves this discrepancy before any
code change.**

**Output:** A verified, default-on (or wired) co-pack path with all bit-exact gates green
in BOTH co-pack modes, and reconciled records. NO new kernel/learner code is expected
(ground truth is verify-only); if Task 1 finds it unwired, this plan escalates to a wiring
follow-up rather than silently editing a hot GPU path.

**Honest payoff (do NOT claim an e2e win on this hardware):** ~10–15% e2e small/medium,
~1.5% wide — and ONLY on a real discrete gfx110x. This box is the spoofed 8-CU APU where
the CPU anchor crushes the GPU at every size (spike-001: GPU 0.06–0.36× of CPU at
20k–100k). On THIS hardware co-pack is ROCm-parity-track maintenance, like 021/022.

**Convention:** runs on master WITHOUT worktree isolation (the repo's GPU-kernel-work
convention). The CPU f64 anchor is the hard merge gate and MUST stay byte-untouched.
</objective>

<execution_context>
@$HOME/.claude/gsd-core/workflows/execute-plan.md
@$HOME/.claude/gsd-core/templates/summary.md
</execution_context>

<context>
@.planning/STATE.md
@./CLAUDE.md
@.planning/spikes/024-batch-sibling-scans/README.md
@.claude/skills/spike-findings-lightgbm_rs/references/gpu-scan-roundtrip-copack.md

# Ground-truth source files (read-only unless Task 1 proves unwired)
@crates/lgbm-treelearner/src/learner.rs
@crates/lgbm-treelearner/src/resident_pool.rs
@crates/lgbm-compute/src/kernels/split.rs
@crates/lgbm-compute/src/lib.rs
</context>

<tasks>

<task type="auto">
  <name>Task 1: Resolve the wiring discrepancy — establish ground truth before any change</name>
  <files>crates/lgbm-treelearner/src/learner.rs, crates/lgbm-treelearner/src/resident_pool.rs, crates/lgbm-compute/src/kernels/split.rs, crates/lgbm-compute/src/lib.rs</files>
  <action>
The spike-findings SKILL reference claims spike-024 is "WIRED phase 12 behind
LGBM_SIBLING_COPACK", but the task brief's fresh read claims the kernel/launcher exist in
split.rs but are NOT called from learner.rs. RESOLVE this before touching any code.

Run the ground-truth grep and read the call site:
`grep -rn "LGBM_SIBLING_COPACK|find_best_splits_fused_siblings|scan_resident_siblings|sibling_copack" crates/`

Then read learner.rs around the co-pack eligibility block and call site (~lines 1756–1860),
resident_pool.rs `sibling_copack_override()` (~line 282), split.rs
`find_best_splits_fused_siblings_from_handles_on` (~1584) and `find_best_splits_fused_siblings_kernel`
(~1079), and lib.rs `Backend::scan_resident_siblings` (~1029 default + ~2507 Rocm override).

Branch on what grep proves (per the brief's Step-0 decision tree):
  - **(EXPECTED, per current master) Already fully wired + default-on:** learner.rs calls
    `self.backend.scan_resident_siblings(...)` inside the `copack_feats`-gated block, and
    `sibling_copack_override()` returns `None`/`Some(true)` ⇒ co-pack engages whenever the
    structural correctness gate holds (resident scan-only smaller child, resident-subtract
    larger child, both scannable, identical spine membership). `LGBM_SIBLING_COPACK=0` is
    the byte-identical OFF switch. ⇒ This task is VERIFY-ONLY: record the call-site line, the
    gate predicate, and the default-on semantics in the SUMMARY. **Make NO kernel/learner
    edit** (the brief: "task becomes verify (run gates) + confirm; no kernel change").
  - **(if grep proves unwired) Kernel/launcher exist but learner.rs never calls them:** STOP
    and escalate — write the finding into the SUMMARY, set the plan outcome to "wiring
    required, follow-up needed", and do NOT half-wire a hot GPU growth loop inside a quick
    task. The wiring recipe (2-slot `find_best_splits_fused_siblings_from_handles_on` call,
    defer-smaller-scan-past-`subtract_resident`, preserve W=64 `scan_cube_dim()`, gate behind
    `LGBM_SIBLING_COPACK`, default-OFF initially) belongs in a dedicated plan, not here.
  - **(if a partial/dead flag exists) Complete the wiring behind that same flag name** —
    again, only if the change is small + self-contained; otherwise escalate as above.

Record the resolved verdict explicitly (which branch, with the proving grep line numbers)
in the Task-1 notes so the SUMMARY can reconcile the SKILL reference's "WIRED phase 12"
claim against reality. No file is edited in this task on the EXPECTED branch.
  </action>
  <verify>
    <automated>grep -rn "scan_resident_siblings\|LGBM_SIBLING_COPACK\|find_best_splits_fused_siblings" crates/lgbm-treelearner/src/learner.rs crates/lgbm-treelearner/src/resident_pool.rs crates/lgbm-compute/src/kernels/split.rs crates/lgbm-compute/src/lib.rs | grep -v '^#'</automated>
  </verify>
  <done>
The wiring branch is RESOLVED with citing grep line numbers. On the EXPECTED branch:
learner.rs is confirmed to call `scan_resident_siblings` inside the `copack_feats` gate and
`sibling_copack_override()` is confirmed default-on (`None`⇒engage). No code edited. The
verdict (and the correction to the "NOT called from learner.rs" claim) is captured for the
SUMMARY. On an unwired/partial branch: the plan outcome is escalated and no half-wiring is
committed.
  </done>
</task>

<task type="auto">
  <name>Task 2: Run the mandatory bit-exact gates in BOTH co-pack modes (default + forced)</name>
  <files>crates/oracle-harness/tests/kernel_parity.rs, crates/oracle-harness/tests/learner_parity.rs, crates/oracle-harness/tests/raw_bin_train_parity.rs</files>
  <action>
Prove the co-pack path is bit-exact and the default (off) path is unchanged. Run each gate
TWICE — once with `LGBM_SIBLING_COPACK` UNSET (the default path) and once with
`LGBM_SIBLING_COPACK=1` (force the co-pack path) — so both the byte-unchanged fallback and
the co-pack path are exercised. The CPU f64 anchor is the hard merge gate; it must stay
byte-untouched.

CPU-anchor + facade + oracle gates (no GPU feature):
  - `cargo test -p lgbm-treelearner --lib`
  - `cargo test -p lgbm`
  - `cargo test -p oracle-harness` — especially `raw_bin_train_matches_cpp_golden`
    (committed C++ golden, RED-FLAG sentinel) and the `learner_parity_*` suite. The CPU
    co-pack parity gate `kernel_parity_sibling_copack_equals_two_scans_on_cpu` (split.rs
    cubecl-cpu W=1) asserts the 2-slot scan is byte-identical to two separate single-slot
    scans, every SplitInfo field.

ROCm split-parity gates (real GPU, `--features rocm`):
  - `cargo test -p oracle-harness --features rocm kernel_parity` — especially
    `kernel_parity_split_within_tol_on_hip` (the ~1e-6 hip split gate) and
    `kernel_parity_sibling_copack_equals_two_scans_on_hip` (the hip 2-slot==two-scans gate
    pinned to the cubecl-cpu native anchor, NOT GPU-vs-GPU — per the def-f8u-01 rule: never
    compare two nondeterministic GPU f32 paths to each other at 1e-6).

For EACH command above, run the UNSET arm then the SET arm, e.g.:
  `cargo test -p oracle-harness` then `LGBM_SIBLING_COPACK=1 cargo test -p oracle-harness`.

CRITICAL RED-FLAG rule: co-pack is bit-exact BY CONSTRUCTION (each feature's sequential
scan is unchanged — no spike-016 reorder; spike-024 proved B's two halves byte-identical to
A's two separate scans, every cell, 2 restarts). Therefore NO committed oracle golden should
change. If `raw_bin_train_matches_cpp_golden` (or any committed golden) changes, that is a
RED FLAG to INVESTIGATE — do NOT re-pin / auto-accept the golden. Stop and report the
divergence with the failing cell.

NOTE on `LightGBM/`: never `git add` the reference tree.
  </action>
  <verify>
    <automated>cargo test -p lgbm-treelearner --lib && cargo test -p lgbm && cargo test -p oracle-harness && LGBM_SIBLING_COPACK=1 cargo test -p oracle-harness raw_bin_train_matches_cpp_golden learner_parity kernel_parity_sibling_copack_equals_two_scans_on_cpu</automated>
  </verify>
  <done>
All listed gates pass in BOTH `LGBM_SIBLING_COPACK` UNSET and `=1` modes. The hip gates
(`kernel_parity_split_within_tol_on_hip`, `kernel_parity_sibling_copack_equals_two_scans_on_hip`)
pass under `--features rocm` on the local ROCm GPU. NO committed oracle golden changed
(`raw_bin_train_matches_cpp_golden` green, byte-idempotent). If any golden changed, the task
HALTS and reports it as a red flag rather than re-pinning.
  </done>
</task>

<task type="auto">
  <name>Task 3: Reconcile records to the verified ground truth</name>
  <files>.claude/skills/spike-findings-lightgbm_rs/references/gpu-scan-roundtrip-copack.md, .planning/STATE.md</files>
  <action>
Make the project records match Task 1's resolved verdict + Task 2's gate evidence, so the
discrepancy the brief flagged cannot recur.

If Task 1 confirmed EXPECTED (fully wired + default-on — current master):
  - In `gpu-scan-roundtrip-copack.md`, the "Wiring (done, phase 12)" prose already says
    WIRED behind `LGBM_SIBLING_COPACK` and that the gate "ANDs in `larger_is_resident_subtract`".
    Verify it is accurate against the resolved call site (learner.rs co-pack gate at ~1788:
    `resident_eligible && copack_override != Some(false) && smaller_resident_only &&
    smaller_scannable && larger_is_resident_subtract && ...`). Correct the one stale nuance
    if present: the env flag is **default-ON** (`None`⇒engage; `=0`=off), not "behind a flag
    that must be set ON" — state the off-switch semantics explicitly so a future reader does
    not repeat the brief's "exists but unwired / needs flag set" misread. Add a one-line
    pointer to the call site (`learner.rs` `scan_resident_siblings`) and the CPU+HIP gate
    names as the verification anchor.
  - In `STATE.md`, add a `last_activity_desc` / activity-log line recording that quick task
    260625-obl VERIFIED spike-024 co-pack is live + default-on + bit-exact in both modes
    (cite the gate result), reconciling the "is it wired?" question raised by the brief. Do
    NOT change phase/plan progress counters — this is a verification quick task, not a phase
    plan.

If Task 1 escalated (unwired/partial):
  - Record in BOTH files that the SKILL reference's "WIRED phase 12" claim was INCORRECT and
    a dedicated wiring plan is required; do not claim it is verified.

Keep edits minimal and scoped (use Edit, not Write, on STATE.md). Do not touch any source
file in this task.
  </action>
  <verify>
    <automated>grep -n "default-on\|LGBM_SIBLING_COPACK\|scan_resident_siblings\|260625-obl" .claude/skills/spike-findings-lightgbm_rs/references/gpu-scan-roundtrip-copack.md .planning/STATE.md | grep -v '^#'</automated>
  </verify>
  <done>
The skill reference and STATE.md reflect the verified ground truth: co-pack is wired,
default-on, and bit-exact in both modes (or, on the escalation branch, that wiring is still
required). The off-switch semantics (`LGBM_SIBLING_COPACK=0`) and the call-site + gate-name
anchors are recorded so the discrepancy cannot recur. No source file edited; STATE progress
counters unchanged.
  </done>
</task>

</tasks>

<threat_model>
## Trust Boundaries

| Boundary | Description |
|----------|-------------|
| host→device (ROCm) | grad/hess histograms + scalars uploaded to GPU for the 2-slot co-packed scan; results read back |
| committed golden ↔ runtime | `raw_bin_train_matches_cpp_golden` is the C++-faithful sentinel; a silent golden change would mask a parity regression |

## STRIDE Threat Register

| Threat ID | Category | Component | Disposition | Mitigation Plan |
|-----------|----------|-----------|-------------|-----------------|
| T-obl-01 | Tampering | co-pack path silently altering split results | mitigate | Task 2 runs the CPU + HIP byte-identical gates (`kernel_parity_sibling_copack_equals_two_scans_on_{cpu,hip}`) in BOTH copack modes; bit-exact by construction |
| T-obl-02 | Tampering | committed oracle golden re-pinned to hide a divergence | mitigate | RED-FLAG rule: a `raw_bin_train_matches_cpp_golden` change HALTS the task and is reported, never auto-accepted |
| T-obl-03 | Repudiation | stale "WIRED phase 12" vs "not called from learner.rs" records | mitigate | Task 1 establishes grep ground truth; Task 3 reconciles records with citing line numbers |
| T-obl-SC | Tampering | npm/pip/cargo installs | accept | no new dependencies; this task adds no package installs (verify + docs only) |
</threat_model>

<verification>
- Task 1 grep resolves the wiring branch with cited line numbers (EXPECTED = already
  wired + default-on).
- Task 2: `cargo test -p lgbm-treelearner --lib`, `cargo test -p lgbm`,
  `cargo test -p oracle-harness` (incl. `raw_bin_train_matches_cpp_golden`, `learner_parity`,
  `kernel_parity_sibling_copack_equals_two_scans_on_cpu`) all green; ROCm
  `cargo test -p oracle-harness --features rocm kernel_parity` green (incl.
  `kernel_parity_split_within_tol_on_hip`, `kernel_parity_sibling_copack_equals_two_scans_on_hip`)
  — each in BOTH `LGBM_SIBLING_COPACK` unset and `=1` modes.
- No committed oracle golden changed.
- Task 3: skill reference + STATE.md reconciled; `LightGBM/` never git-added.
</verification>

<success_criteria>
- The "is spike-024 wired?" discrepancy is resolved by grep ground truth (not prose), with
  the EXPECTED verdict: kernel + launcher + `scan_resident_siblings` backend method +
  growth-loop reorder are LIVE on master and default-ON, gated by `LGBM_SIBLING_COPACK` (off
  = `0`).
- All mandatory bit-exact gates pass in both co-pack modes; the CPU f64 anchor is untouched;
  no committed golden changed.
- Records reflect verified ground truth.
- The SUMMARY states the honest payoff (ROCm-parity-track on this APU; real value only on
  discrete gfx110x) and makes NO e2e-win claim on this hardware.
</success_criteria>

<output>
Create `.planning/quick/260625-obl-wire-spike-024-sibling-scan-co-pack-into/260625-obl-SUMMARY.md` when done.
</output>
