# Phase 21: Harden the On-Device Driver (re-cut) - Context

**Gathered:** 2026-07-02
**Status:** Ready for planning

<domain>
## Phase Boundary

**Phase 21 is RE-CUT.** Its original scope — "End-to-End On-Device Driver
Integration + Parity Gate" (ODL-18/ODL-19) — was pulled forward into **Phase 20**
per decision **D-01** and delivered there: Phase 20's VERIFICATION.md is `passed`
(6/6) and claims `requirements_verified: [ODL-16, ODL-17, ODL-18, ODL-19]`, with
crit 5 = the grow loop → `(Tree, DataPartition)` STRUCTURE-bit-exact gate and
crit 6 = the no-f64-per-row kernel constraint. The ROADMAP checklist already
instructs: *"Phase 21 reduces to hardening or folds into 22/23. Re-cut via
`/gsd-phase` before planning."* (Decided this session — user chose "harden".)

**This phase delivers:** hardening / parity-slack for the just-landed end-to-end
on-device grow loop, on the **continuous-feature slice**. Concretely:

1. Fix the **WR-01** latent slot-aliasing bug in `HistArena::swap` that the
   multi-leaf grow loop is the first live consumer of.
2. Broaden the STRUCTURE parity evidence beyond the single 4-leaf case the
   Phase-20 gate proved, using a small **targeted** set of risk cases.
3. Reconcile the ODL-18/19 requirement bookkeeping and record the ROADMAP re-cut.

**Explicitly NOT in this phase (unchanged boundaries):**
- **On-device categorical splits** (bitset / categorical eval / partition /
  `SplitCategorical`) → **Phase 22** (ODL-22). The proving slice here is
  continuous-feature only.
- **Perf-validation / default-ON rollout DoD** (Kaggle A/B, `device_launches`,
  wall-clock ratio, default flip) → **Phase 23** (ODL-20/ODL-21).
- **Wiring on-device metric eval (`EvalKernel`) into GBDT** — Phase 20 built it
  standalone/anchor-pinned but GBDT still evals host-side over the mirrored
  score. Considered and **deferred** (user chose "harden the driver", not the
  metric-wiring re-cut).

Everything **additive** and gated by `LGBM_CUDA_ON_DEVICE`; CPU / ROCm /
existing-host-CUDA paths stay **byte-unchanged** with the env unset; the hard
merge gate stays green. Anchored to the **cubecl-cpu f64 fold**, never
GPU-vs-GPU (def-f8u-01).

</domain>

<decisions>
## Implementation Decisions

### Phase identity (re-cut)
- **D-01:** Phase 21 becomes a **hardening/parity-slack** phase for the on-device
  driver. Its original ODL-18/ODL-19 driver-integration scope is accepted as
  **delivered by Phase 20** (D-01 pull-forward). No re-implementation of the
  driver — strengthen it.

### Parity corpus breadth (STRUCTURE gate)
- **D-02: Targeted risk cases, not a broad sweep.** Add a small, deliberate set
  of STRUCTURE-bit-exact cases aimed at the known correctness risks, each pinned
  to the cpu f64 anchor (tie-aware `default_left`, leaf values within ~1e-5):
  - a **deep tree with >2 simultaneously-live leaves** — the case that actually
    exercises the WR-01 swap path (the existing gate's 4 leaves may not);
  - a **no-split / single-leaf** tree (`best_leaf == −1` break path);
  - a **`min_data_in_leaf` / `min_sum_hessian_in_leaf`-constrained** case.
  Rationale: fast, high-signal; a full rows×features×leaves×constraints matrix
  is more maintenance than the risk warrants at this stage.

### WR-01 slot-aliasing fix + validation
- **D-03: Fix + dedicated repro test.** Replace the
  `(parent_slot + 1) % num_slots` heuristic in `HistArena::swap`
  (`histogram_arena.rs:365`) with a **free-slot scan against `leaf_to_slot`**
  (drop the now-internal `parent_leaf` key) per the 18-REVIEW.md WR-01 fix
  sketch. Add a **regression test that constructs a >2-live-leaf grow scenario
  proven to alias under the old heuristic** and asserts no slot corruption — so
  the bug is locked closed independently of the parity corpus.

### ROCm coverage
- **D-04: cubecl-cpu is the gate; ROCm is a best-effort smoke.** The merge gate
  stays the deterministic **cubecl-cpu f64 anchor**. A real-ROCm (`cubecl-hip`,
  f32, ~1e-6) run — if attempted — is pinned to the cpu anchor (def-f8u-01,
  never GPU-vs-GPU) and is **informative, not blocking**. Justified by the
  spoofed 8-CU APU + f32 non-determinism (see memory `rocm-gfx1100-available`);
  full real-hardware validation is Phase 23's Kaggle DoD.

### Requirement bookkeeping + ROADMAP re-cut
- **D-05: Mark done + new hardening requirement ID.** Mark **ODL-18 and ODL-19
  Complete** (delivered Phase 20) in REQUIREMENTS.md and its traceability table.
  Add a **new hardening requirement (e.g. `ODL-18H`)** covering the targeted
  corpus + WR-01 fix + repro, mapped to Phase 21. Re-cut the ROADMAP Phase 21
  **body** (Goal / Success Criteria / Notes) via `/gsd-phase` so it reflects the
  hardening scope instead of the stale driver-integration text. This bookkeeping
  reconciliation is **in-scope for the phase plan**.

### Claude's Discretion
- Exact fixture parameters for the targeted corpus (row counts, feature counts,
  `num_leaves`, threshold values) — pick the smallest configs that provably (a)
  yield >2 live leaves at once and (b) trigger each constraint/edge branch.
- Whether the WR-01 repro lives as a `lgbm-compute` unit test or an
  oracle-harness case — pick whichever most directly demonstrates the aliasing.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Re-cut evidence & bookkeeping
- `.planning/ROADMAP.md` — Phase 21 checklist line (already says "re-cut to
  hardening"); the Phase 21 **body** (Goal/Success Criteria/Notes) is STALE and
  must be re-cut via `/gsd-phase` as part of this phase.
- `.planning/REQUIREMENTS.md` — ODL-18/ODL-19 still show `[ ]` pending under
  Phase 21 (lines ~50–51, traceability ~105–106); contradicts Phase 20
  verification. Reconcile per D-05 (mark done + add `ODL-18H`).
- `.planning/phases/20-on-device-score-updater-metrics/20-VERIFICATION.md` —
  proof ODL-18/19 were delivered in Phase 20 (crit 5 STRUCTURE gate, crit 6
  no-f64-per-row); note #2 flags device metric eval NOT wired into GBDT.
- `.planning/phases/20-on-device-score-updater-metrics/20-CONTEXT.md` — the D-01
  pull-forward decision and its "Phase 21 reduces to hardening/slack" implication.

### WR-01 fix
- `crates/lgbm-compute/src/kernels/histogram_arena.rs` §`HistArena::swap` (~:365)
  — the `(parent_slot+1)%num_slots` heuristic to replace with a `leaf_to_slot`
  free-slot scan.
- `.planning/phases/18-on-device-data-partition-tree-mutation-prediction/18-REVIEW.md`
  — WR-01 finding + fix sketch.
- `.planning/phases/18-on-device-data-partition-tree-mutation-prediction/18-REVIEW-FIX.md`
  — any applied-fix notes for the surrounding review items.

### Driver + parity gate (the thing being hardened)
- `crates/lgbm-compute/src/kernels/grow_driver.rs` (~:418
  `grow_tree_on_device_driver`) — the best-first per-leaf orchestration + own
  `DriverLeaf` bookkeeping; `HistArena::swap`'s live consumer.
- `crates/oracle-harness/tests/learner_parity.rs` — hosts
  `learner_parity_on_device_structure_gate`; where the targeted corpus cases and
  the tie-aware cpu-f64-anchor comparator live.

### Standing constraints
- `CLAUDE.md` — f32 end-to-end, ~1e-6 vs C++, cubecl-cpu f64 fold is the hard
  merge gate; LightGBM/ is read-only reference.

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `grow_tree_on_device_driver` (`grow_driver.rs`) + the `CpuBackend`/`GpuBackend<R>`
  seam (`lib.rs:2302,1371`) already sequence the full grow loop — no new driver
  code, only hardening around it.
- `learner_parity_on_device_structure_gate` (`oracle-harness/tests/learner_parity.rs`)
  is the existing STRUCTURE gate; extend it with the D-02 targeted cases rather
  than build a new harness. Comparator is already tie-aware on `default_left`.
- `HistArena` + `leaf_to_slot` occupancy map already exist — the WR-01 fix is a
  localized change to the slot-selection logic, not new data structures.

### Established Patterns
- Anchor to the cubecl-cpu f64 fold; STRUCTURE bit-exact + leaf values ~1e-5;
  never compare two GPU f32 paths (def-f8u-01).
- Additive + `LGBM_CUDA_ON_DEVICE`-gated; env-unset = byte-unchanged; merge gate
  runs on the DEFAULT (cubecl-cpu) lane so the STRUCTURE gate is non-vacuous
  without ROCm hardware.

### Integration Points
- The gated CpuBackend flip (`cuda_on_device_enabled()`) is how the STRUCTURE
  gate runs in the default lane — new corpus cases plug in there.
- REQUIREMENTS.md traceability table + ROADMAP Phase 21 body are the
  bookkeeping integration points (D-05).

</code_context>

<specifics>
## Specific Ideas

- The deep >2-live-leaf corpus case is the linchpin: it must simultaneously (a)
  broaden parity evidence and (b) be the scenario that would alias under the old
  `HistArena::swap` heuristic — one case, two purposes.

</specifics>

<deferred>
## Deferred Ideas

- **Wire on-device `EvalKernel` metric eval into GBDT** (replace host-side eval
  over the mirrored score) — a real Phase-20 follow-up, but out of this
  hardening re-cut. Candidate for a later slack/follow-up phase.
- **Full rows×features×leaves×constraints parity sweep** — considered under D-02
  and rejected in favor of targeted cases; revisit only if targeted cases prove
  insufficient.

### Reviewed Todos (not folded)
Five perf-campaign todos matched on GPU/loop keywords but belong to **Phase 23**
(perf-validation DoD), not this parity-hardening phase:
- `establish-large-data-benchmark-fixture.md` — GPU profiling fixture → Phase 23.
- `profile-gpu-training-loop-large-data.md` — stage attribution → Phase 23.
- `spike-gpu-cpu-crossover.md` — GPU-vs-CPU crossover sweep → Phase 23.
- `spike-lowrow-phase-ab.md` — low-row fixed-overhead A/B → Phase 23.
- `verify-lds-atomic-lowering-gfx1100.md` — cubecl-hip LDS atomic lowering →
  perf/ROCm track, not this phase's parity gate.

</deferred>

---

*Phase: 21-harden-on-device-driver (re-cut from end-to-end-driver-integration-parity-gate)*
*Context gathered: 2026-07-02*
