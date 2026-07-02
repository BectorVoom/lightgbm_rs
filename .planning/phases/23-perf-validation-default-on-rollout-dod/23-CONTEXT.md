# Phase 23: Perf-Validation + Default-On Rollout (DoD) - Context

**Gathered:** 2026-07-02
**Status:** Ready for planning

<domain>
## Phase Boundary

Measure the on-device learner's win on **real discrete CUDA** (Kaggle, `boomvector` — the only path to real numbers; local GPU is a spoofed 8-CU APU), then make the on-device path the **DEFAULT** CUDA tree-learner — contingent on ~1e-6 parity AND a not-slower A/B result — with the host-CUDA path retained as an off-switch. This is the v1.1 milestone Definition of Done.

**In scope:** the Kaggle A/B harness (`device_launches/tree` + wall-clock ratios), the CUDA-only default-on routing flip, its off-switch, the real-CUDA parity assertion, and the DoD evidence artifact.

**Out of scope (deferred):** multi-stream overlap (stretch spike only if launch-collapse underdelivers); any new kernel/perf work beyond routing; ROCm/CPU default changes (they stay host-driven, byte-unchanged).

</domain>

<decisions>
## Implementation Decisions

### Default-On Flip Semantics
- **D-01:** `LGBM_CUDA_ON_DEVICE` becomes **tri-state**: unset ⇒ follow device default, `"0"` ⇒ force OFF (the off-switch fallback), `"1"` ⇒ force ON. This replaces the current binary parse (`cuda_on_device_enabled()` at `crates/lgbm-compute/src/lib.rs:1324`, today: unset/anything⇒off, `"1"`⇒on). The OnceLock-cached read pattern stays.
- **D-02:** The CUDA-only default lives at the **routing seam** as `device_type==cuda AND enabled`, NOT baked into the env parse. ROCm/CPU never see default-on because their device gate is false — this is how SC-4 (ROCm + CPU byte-unchanged) is upheld.
- **D-03:** "This is CUDA vs ROCm/HIP" is decided by the **compiled cubecl runtime binding (cargo feature)** — a `cubecl-cuda` binding ⇒ default-on eligible; `cubecl-hip` ⇒ default-off. No runtime device_type-string sniffing. A ROCm build literally cannot default-on. (Matches how `runtime::ActiveRuntime` is already selected by cargo feature.)

### 'Not-Slower' Pass/Fail Rule (the DoD contingency)
- **D-04:** Pass bar = on-device **median wall-clock ≤ 5% slower** than the current host-CUDA path ("within noise" — not-measurably-slower, not a strict win).
- **D-05:** Sign-stability = **3 in-session runs per config, take the median.** The default flips only if **BOTH** shapes (500k×50 AND the wide shape) pass the ≤5% bar. A regression on either shape blocks the global default flip (conservative — no shape-aware routing heuristic).

### Kaggle A/B Harness Scope
- **D-06:** The wide shape = **100k × 500** (10× the feature count of the 500k×50 point, fewer rows to fit T4 memory; stresses per-feature histogram fan-out / launch count — the thing on-device is meant to collapse). Tree count 100 (baseline convention).
- **D-07:** Harness is a **committed, reusable script** (extends the existing `benchmark.py` / `continue_benchmark.py` family) that emits a **structured results file (MD + JSON)** — `device_launches`, wall-clock medians, and ratios — committed under the phase dir as the first-class DoD evidence artifact.
- **D-08:** Comparison set to report: on-device **vs host-CUDA** (the not-slower gate, D-04/D-05), plus context ratios vs **official LightGBM** (the ~4.46× pre-on-device bar), plus `device_launches/tree` vs the **8,570 / 100-trees** baseline (SC-2 launch-collapse confirmation).

### Rollout Guardrails & Fallback
- **D-09:** Phase 23 **always** lands the committed harness + results artifact. The **default-ON flip is a SEPARATE commit gated on the pass verdict.** If the A/B fails the ≤5%/both-shapes bar, the default stays **OFF** (opt-in via `LGBM_CUDA_ON_DEVICE=1`) and the phase is still DoD-complete with documented numbers + a follow-up note. Honors the audit-before-wire / fused-kernel-default-off precedent — never auto-engage before proof.
- **D-10:** At default-ON, unsupported configs (`use_quantized_grad`, or any case where `grow_tree_on_device` returns `Ok(None)`, D-06 host-fallback gate) fall back to the host-CUDA path with the existing **silent `Ok(None)`** behavior — no log noise, results still correct.
- **D-11:** Parity proof backing the flip = the CPU f64 merge gate (on-device tree bit-exact on the cubecl-cpu anchor lane) + the already-green per-phase ~1e-6 anchor gates (14–22) **PLUS a real-CUDA end-to-end ~1e-6 parity assertion in the Kaggle harness** (on-device predictions vs host-CUDA / official on the actual datasets), committed as a hard check — the strongest real-hardware proof at the flip point.

### Claude's Discretion
- The exact mechanism for capturing `device_launches` on Kaggle (parsing the existing `[phase_prof:…] COUNTS: device_launches=…` line emitted under `LGBM_PHASE_PROF`, at `crates/lgbm-treelearner/src/phase_prof.rs:197`) is a planner/implementation detail.
- Results-file schema fields beyond the required metrics (D-07/D-08); Kaggle GPU-quota budgeting across the 3-run × 2-shape × 2-path matrix.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Phase spec & requirements
- `.planning/ROADMAP.md` — Phase 23 section (Goal, SC-1…SC-4, Notes) + the milestone Progress table.
- `.planning/REQUIREMENTS.md` — ODL-20 (Kaggle A/B harness: `device_launches/tree` + wall-clock ratio at 500k×50 and a wide shape) and ODL-21 (default CUDA path contingent on parity AND not-slower, host path as `LGBM_CUDA_ON_DEVICE=0` off-switch).

### Routing seam (the code Phase 23 changes)
- `crates/lgbm-compute/src/lib.rs` §1235–1328 — `Backend::on_device_growth_supported()` (the discriminator that ANDs in the env gate), `Backend::grow_tree_on_device()` seam, and `cuda_on_device_enabled()` (the OnceLock env parse to make tri-state, D-01).
- `crates/lgbm-boosting/src/gbdt.rs` §1006, §1485 and `crates/lgbm-boosting/src/score_updater.rs` §49, §147, §173 — the `boosting_on_cuda_` toggle that keys off the same env gate; must respect the new tri-state + CUDA-only default.
- `crates/lgbm-treelearner/src/phase_prof.rs:197` — the `device_launches` COUNTS emitter parsed by the harness.

### Perf baselines & measurement precedent
- `Skill("spike-findings-lightgbm_rs")` — spike-048 (real-CUDA attribution: metric-eval fix −26%, GPU hist = 53% long-pole; route-to-CPU & sync-floor REFUTED on real NVIDIA) and the real-discrete-CUDA Kaggle arc; the 8,570/100-trees `device_launches` baseline and the ~4.46× official-vs-lgb_rs pre-on-device bar.
- `benchmark.py`, `continue_benchmark.py`, `benchmark_cpu_gpu.py`, `Colab_Benchmark_LightGBM.ipynb` (repo root, untracked) — existing benchmark harness family the committed A/B script extends.

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `cuda_on_device_enabled()` (`lgbm-compute/src/lib.rs:1324`) — OnceLock env-cache pattern to extend to tri-state (mirror `split_2lane_enabled` at :1336).
- `phase_prof.rs` COUNTS line — already emits `device_launches` (+ build/subtract/scan/fused breakdown) under `LGBM_PHASE_PROF`; the harness parses it rather than adding new instrumentation.
- `benchmark.py` family + Kaggle CLI (`boomvector`) — the established real-discrete-CUDA measurement path.

### Established Patterns
- Env gates are OnceLock-cached, read once, default-off — the tri-state change must preserve read-once semantics and keep the env-unset merge gate deterministic.
- On-device eligibility = `on_device_growth_supported()` AND'd at the call site; the CUDA-only default rides this same AND (D-02), keeping ROCm/CPU byte-unchanged.
- "Audit-before-wire" / fused-kernel default-off precedent: perf routing is flipped only on sign-stable proof (D-09).

### Integration Points
- The default-on flip touches the boosting-layer `boosting_on_cuda_` toggle (gbdt.rs / score_updater.rs) and the compute-layer discriminator (lib.rs) — both must agree on the new tri-state + runtime-binding CUDA default.
- The Kaggle harness runs the full train loop, so it exercises the driver end-to-end (grow + score-update + metric) — the natural home for the D-11 real-CUDA parity assertion.

</code_context>

<specifics>
## Specific Ideas

- Pass bar phrased as "within noise, ≤5% slower" deliberately — the user reads "not-slower" pragmatically, not as "strictly faster".
- User upgraded the parity evidence to the STRONGER option: a dedicated real-CUDA end-to-end ~1e-6 assertion in the harness (D-11), not merge-gate-only.
- Default-on flip is intentionally a separate, verdict-gated commit so the phase ships value (harness + evidence) even if the win doesn't materialize.

</specifics>

<deferred>
## Deferred Ideas

- **Multi-stream overlap** — a stretch spike ONLY if the launch-count reduction underdelivers on wall-clock (ROADMAP Phase 23 Notes). Not in scope unless the A/B shows launch-collapse without a wall-clock win.
- **Shape-aware routing** (default-on per-shape rather than globally) — considered under D-05 and rejected for this phase (needs a routing heuristic); revisit only if one shape consistently regresses.

### Reviewed Todos (not folded)
Four GPU-profiling todos matched at score 0.6 but are largely superseded by the completed spike campaign (001–054) and prior phases; not folded into Phase 23:
- `establish-large-data-benchmark-fixture.md` — large-data fixture already exists (`LGBM_BENCH_SWEEP=wide`, 1M×500 end-to-end parity).
- `profile-gpu-training-loop-large-data.md` — stage attribution done in spike-048 (GPU hist = 53% long-pole).
- `spike-gpu-cpu-crossover.md` — crossover characterized (GPU loses to 16-core CPU on the spoofed APU; real-CUDA is the Phase-23 A/B).
- `spike-lowrow-phase-ab.md` — low-row fixed-overhead localization superseded by the real-CUDA per-phase attribution.

</deferred>

---

*Phase: 23-perf-validation-default-on-rollout-dod*
*Context gathered: 2026-07-02*
