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

## Cross-Milestone Trends

### Process Evolution

| Milestone | Phases | Plans | Key Change |
|-----------|--------|-------|------------|
| v1.0 | 8 | 55 | Established the dependency-forced bottom-up parity spine + real-binary FP-trace tiebreak + bit-exact cpu anchor |

### Cumulative Quality

| Milestone | Requirements | Audit | LOC |
|-----------|--------------|-------|-----|
| v1.0 | 69/69 | PASSED | ~68.7k (65.5k Rust + 3.2k Py) |

### Top Lessons (Verified Across Milestones)

1. Anchor every numerical claim to one deterministic reference (the cubecl-cpu f64 fold); never compare two nondeterministic paths to each other.
2. When a faithful port is falsified, a source-built FP execution trace of the reference binary is the fastest route to the true root cause.
