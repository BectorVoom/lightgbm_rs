# Project Retrospective

*A living document updated after each milestone. Lessons feed forward into future planning.*

## Milestone: v1.0 — Full Single-Machine Parity

**Shipped:** 2026-06-21
**Phases:** 8 | **Plans:** 55 | **Tasks:** 129 | **Commits:** 574 | **Span:** 16 days (2026-06-05 → 2026-06-21)

### What Was Built
- A pure-Rust port of Microsoft LightGBM reaching **full single-machine parity** with C++ 4.6: GBDT/DART/RF/GOSS, all objectives + metrics, categorical/monotone/interaction/forced-split/CEGB constraints, TreeSHAP, refit, and feature importance.
- The **bit-exact spine**: binning (`BinMapper`) and the serial tree learner are bit-exact vs real `lib_lightgbm` 4.6 on both committed corpora; the `cubecl-cpu` f64-fold path is the deterministic hard merge gate.
- A **CubeCL CPU/ROCm backend** (`lgbm-compute`) containing all device churn behind one `Backend` trait — cpu bit-exact + cubecl-hip f32 within ~1e-6 on real gfx1100, with CUDA warp ops on the capability-gated `Plane` API.
- **Python bindings** (`lgbm-python`): PyO3 + numpy/scipy/polars + sklearn wrappers mirroring the official `lightgbm` package, A/B-matched to 4.6 within 1e-6.
- An **oracle harness + xtask golden-capture** pipeline (kernel/learner/boosting/predict parity) that exercises the production path, validated against a pinned deterministic C++ reference.

### What Worked
- **Dependency-forced bottom-up spine.** Each phase was a vertical, oracle-validated slice (binning → predict → backend → learner → GBDT → variants → Python). Parity widened outward instead of being deferred — divergence was always localized to the lowest unproven layer.
- **Predict-before-train (Phase 3).** Proving prediction parity against a C++-trained model *before* any training code existed isolated the model/IO surface from the learner FP surface.
- **Real-binary FP execution traces as the parity tiebreaker.** When the port was *falsified* by the real `lib_lightgbm` (Phase 5 CR-03; Phase 7 D-05), a source-built single-thread FP trace gave the genuine operand provenance (e.g. the `min_gain_shift` 2*kEpsilon-bumped-hessian bug, the child-LeafSplits seed-from-parent-SplitInfo). This beat guessing every time.
- **Honest typed-rejects over wrong-but-similar output.** Unsupported knife-edge combos were `BoostingError::UnsupportedConfig` rather than shipped with divergent leaves — and most were later un-deferred once the real root cause was traced.
- **CMP-01 containment boundary.** Keeping all CubeCL behind one crate trait let the alpha-churn GPU work proceed without destabilizing the upper crates (CPU-only build needs no ROCm toolchain).

### What Was Inefficient
- **Gap-closure churn.** Phases 1, 2, 5, 6 all went `gaps_found → re-verify` (extra closure plans 01-03, 02-06/07, 05-05..09, 06-06). The keystone learner (Phase 5) needed *five* gap-closure plans to chase a 2-ULP residual to bit-exact. High rigor, but a lot of re-entry.
- **Post-milestone bookkeeping drift.** A 28-task GPU/CPU perf campaign ran as quick tasks after the phases hit 100%, almost none flipped their completion marker — so the milestone-close artifact audit surfaced 33 "open" items that were really shipped-but-unmarked. The milestone audit + acknowledge-defer path absorbed it, but the markers should have been closed at task end.
- **Milestone never formally closed until prompted.** v1.0's phases completed 2026-06-08 but the milestone stayed `executing` for ~13 days of quick-task work before close.

### Patterns Established
- **The cubecl-cpu f64-fold is THE anchor.** Every parity claim is anchored to the deterministic single-owner ordered f64 fold; GPU f32 is a *separate* ~1e-6 gate, never compared GPU-to-GPU.
- **Out-of-milestone work = quick tasks/spikes, not roadmap phases.** Perf + quantized-training work lived outside the v1.0 roadmap (phase dirs 09/10) deliberately — parity is the gate, speed is not.
- **"Cold ceiling overstates warm" + interleaved A/B + bit-exact gate** — the measurement discipline for all perf work (see `spike-findings-lightgbm_rs`).
- **GPU is ROCm-parity, not speed, on gfx1100** — the faithful hist kernel is ~5.4× slower than the multi-threaded CPU anchor; hist-build levers explored and closed.

### Key Lessons
1. **Against an f32 reference, target f32 / ~1e-6 — not 1e-12.** The original 1e-12 framing was revised in Phase 1 discuss as unachievable/meaningless; everything downstream became falsifiable because of it.
2. **When parity fails, instrument the real binary — don't theorize.** Source-built FP traces closed every "irreducible knife-edge" that turned out to be a deterministic operand bug.
3. **A bit-exact merge gate is a force multiplier.** It made the entire perf campaign safe: every optimization had to preserve fold order, so wins shipped without re-litigating correctness.
4. **Close the marker when the task ends.** Deferred bookkeeping turned a clean milestone into a 33-item audit reconciliation.

### Cost Observations
- Model mix: not instrumented this milestone (single-developer + agent sessions).
- Notable: heavy use of parallel subagents (verification, integration check, perf spikes) under a strict bit-exact gate kept large fan-out safe.

---

## Milestone: v1.1 — CUDA On-Device Training Backend

**Shipped:** 2026-07-03
**Phases:** 10 (14–23) | **Plans:** 48 | **Audit:** `gaps_found` (21/23 satisfied)

### What Was Built
The full LightGBM single-GPU CUDA training pipeline ported on-device (design-doc §-grounded): shared device primitives + `CUDASplitInfo`/`CUDARandom` structs, a resident columnar device dataset + row-subset gather, the on-device histogram build/fix/subtract (u64 fixed-point, no f64 per-row loop), the 3-stage best-split finder, mark→prefix-sum→scatter data partition + device tree mutation + tree-walk predict, on-device objectives (regression-family/binary/multiclass/ranking), score-updater + pointwise metrics, the end-to-end grow driver, on-device categorical splits (anchored to real `lib_lightgbm` 4.6 goldens), and the real-CUDA perf-validation/rollout DoD. Fully additive behind `LGBM_CUDA_ON_DEVICE`.

### What Worked
- **Anchor-before-kernel discipline scaled to a 10-phase GPU port.** Every kernel pinned STRUCTURE bit-exact to the cubecl-cpu f64 fold (never GPU-vs-GPU, def-f8u-01), so the hard merge gate stayed green and byte-unchanged the entire milestone despite 41k LOC of new device code.
- **Audit-before-wire on the rollout DoD paid off.** Phase 23's contingent default-on flip caught a severe launch-bound regression on real discrete NVIDIA (~23 min/100-tree arm) and correctly WITHHELD the flip instead of shipping a slow default. The gate did its job.
- **Wave-0 scaffolding (goldens + task tables + Nyquist stubs before kernels)** locked the riskiest parity landmines (count-rounding, `default_left != reverse`) as tested units before any kernel was written.

### What Was Inefficient
- **7 kernels shipped anchor-pinned but unwired** (objectives, metric-eval, device dataset row-gather) — verified reusable but with no production consumer, so gradients/metric-eval/row-gather still round-trip host each iteration. The "host only orchestrates" DoD intent is only partially realized; wiring is a v2 carry-forward.
- **Phase 23 closed on the human-verify FAIL branch without a VERIFICATION.md**, surfacing as the sole audit blocker at milestone close (resolved by a verifier pass during close).
- **Kaggle's output-size cap dropped the A/B results files**, so exact per-shape numbers weren't captured on the definitive run — harness fixed to self-evidence, but a re-run is needed.

### Patterns Established
- The **tri-state env gate** (`unset⇒default / "0"⇒off / "1"⇒on`) with a compile-time `on_device_default()` stub as the single source of truth for an opt-in behavior-changing path.
- **Real-hardware verification at milestone close**: ROCm-gated tests deferred as `human_needed` during headless verification, then batch-run on the local GPU at close — parity halves are load-bearing, perf halves judged on sign only (spoofed 8-CU APU, throughput APU-confounded).

### Key Lessons
1. A *contingent* DoD (ship X only if measured-not-slower) is more honest than an unconditional one — encode the contingency in the plan (D-09 FAIL branch) so a bad result withholds the change automatically instead of forcing a judgment call under pressure.
2. "Anchor-pinned + verified" ≠ "wired": a kernel can be correct and reusable yet have no production consumer. Track wiring as a distinct requirement, or the "host only orchestrates" intent silently under-delivers.
3. Reconcile status *markers* at ship time — 30 shipped quick-tasks re-flagged the milestone audit purely because their `status:` field was never flipped, not because work was open.

### Cost Observations
- Model mix: not instrumented (single-developer + agent sessions).
- Notable: the 10-phase port ran on parallel wave-based execution under the bit-exact merge gate; hardware confirmation was batched to one GPU session at close rather than per-phase.

---

## Cross-Milestone Trends

### Process Evolution

| Milestone | Phases | Plans | Key Change |
|-----------|--------|-------|------------|
| v1.0 | 8 | 55 | Established the dependency-forced bottom-up parity spine + real-binary FP-trace tiebreak + bit-exact cpu anchor |
| v1.1 | 10 | 48 | Scaled the anchor-before-kernel discipline to a full on-device GPU port; added the contingent audit-before-wire rollout DoD (withheld a slow default) |

### Cumulative Quality

| Milestone | Requirements | Audit | LOC |
|-----------|--------------|-------|-----|
| v1.0 | 69/69 | PASSED | ~68.7k (65.5k Rust + 3.2k Py) |
| v1.1 | 21/23 | gaps_found (2 intentional v2 deferrals) | ~105.4k Rust |

### Top Lessons (Verified Across Milestones)

1. Anchor every numerical claim to one deterministic reference (the cubecl-cpu f64 fold); never compare two nondeterministic paths to each other.
2. When a faithful port is falsified, a source-built FP execution trace of the reference binary is the fastest route to the true root cause.
