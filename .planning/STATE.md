---
gsd_state_version: 1.0
milestone: v1.0
milestone_name: milestone
status: executing
stopped_at: "Plan 01-01 paused at Task 3 human-action checkpoint (run `cargo run -p xtask -- regen` to capture the C++ RNG golden set)"
last_updated: "2026-06-05T03:46:55.000Z"
last_activity: "2026-06-05 -- Plan 01-01 Tasks 1-2 complete + Task 3 authored; awaiting human-action regen checkpoint"
progress:
  total_phases: 8
  completed_phases: 0
  total_plans: 2
  completed_plans: 0
  percent: 0
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-06-05)

**Core value:** For identical inputs and config, reproduce C++ LightGBM outputs to within ~1e-6 absolute difference on every backend (CPU and ROCm), using f32 (single-precision) data types matching the C++ reference defaults.
**Current focus:** Phase 01 — oracle-contract-foundations

## Current Position

Phase: 01 (oracle-contract-foundations) — EXECUTING
Plan: 1 of 2 — PAUSED at Task 3 human-action checkpoint
Status: Plan 01-01 Tasks 1-2 done, Task 3 code authored; awaiting human regen of the C++ RNG golden set
Last activity: 2026-06-05 -- Plan 01-01 authored through the human-action gate

Progress: [░░░░░░░░░░] 0% (plan 01-01 not yet counted complete — checkpoint open)

### Resume

To resume Plan 01-01 (Task 3 human-action gate):
1. Ensure CMake >= 3.28 and a C++ compiler are installed (`cmake --version`, `c++ --version`).
2. `cargo run -p xtask -- regen` — builds lib_lightgbm at the pinned commit, runs the capture over the master-seed-derived randomized set, writes `crates/oracle-harness/fixtures/rng_sequence.txt` and refreshes `REFERENCE_MANIFEST.md`.
3. `cargo test -p oracle-harness rng_parity` — must exit 0 (replays every committed case).
4. Re-run `cargo run -p xtask -- regen`; confirm `git diff --stat crates/oracle-harness/fixtures/` is empty (idempotent).
5. Commit the fixtures + manifest, then reply "approved" to finish Plan 01-01.

## Performance Metrics

**Velocity:**

- Total plans completed: 0
- Average duration: -
- Total execution time: 0 hours

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| - | - | - | - |

**Recent Trend:**

- Last 5 plans: -
- Trend: -

*Updated after each plan completion*

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.
Recent decisions affecting current work:

- [Roadmap]: Build order is dependency-forced — each layer must match the reference before anything above it can be validated (binning → predict → compute → learner → GBDT → variants → Python).
- [Phase 1 discuss, 2026-06-05]: Numerical contract revised to **f32 (single-precision) end-to-end, ~1e-6 absolute oracle tolerance** on every backend — matching the C++ reference defaults (`score_t`/`label_t` = `float`). Supersedes the prior strict-1e-12 / tiered-oracle direction.
- [Phase 1 discuss, 2026-06-05]: **Standard f32 histogram/score accumulations** on CPU and ROCm — the integer-quantized histogram strategy is dropped (buys nothing at f32 / ~1e-6).

### Pending Todos

None yet.

### Blockers/Concerns

- [Phase 4]: CubeCL is alpha (v0.10.0) — pin exactly, isolate behind lgbm-compute; ROCm capability gaps and CPU-runtime-vs-HIP divergence need empirical validation on the local ROCm GPU (research flag: HIGH). Now evaluated against the ~1e-6 (f32) tolerance rather than bit-exactness.
- [Phase 6]: f32 transcendental (exp/log/pow/sigmoid) parity CPU↔ROCm is unproven at ~1e-6 — needs empirical validation; if a gap appears, fallback is CPU-resident objective grad/hess.
- [Cross-cutting]: RESOLVED (2026-06-05) — the strict-1e-12-vs-tiered tension is closed by adopting the f32 / ~1e-6 contract; project docs (PROJECT/REQUIREMENTS/ROADMAP) updated to match.

## Deferred Items

Items acknowledged and carried forward from previous milestone close:

| Category | Item | Status | Deferred At |
|----------|------|--------|-------------|
| v2 | QNT-01 quantized/discretized gradient training | Deferred (v2) | Roadmap |
| v2 | LIN-01 linear-tree leaves | Deferred (v2) | Roadmap |
| v2 | ING-01/02/03 text-file / binary-cache / Arrow ingestion | Deferred (v2) | Roadmap |

## Session Continuity

Last session: 2026-06-05T02:37:56.004Z
Stopped at: Phase 1 context gathered (numerical contract revised to f32 / ~1e-6)
Resume file: .planning/phases/01-oracle-contract-foundations/01-CONTEXT.md
