# Phase 1: Oracle Contract + Foundations - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-06-05
**Phase:** 1-oracle-contract-foundations
**Areas discussed:** Oracle tier policy, C++ reference & goldens, Workspace crate split, Config modeling

---

## Oracle tier policy

### Q1 — Tier policy
| Option | Description | Selected |
|--------|-------------|----------|
| Tiered | Tier A structural bit-exact all backends / Tier B 1e-12 CPU / Tier C documented-relaxed ROCm | |
| Strict 1e-12 everywhere | ≤1e-12 absolute on CPU and ROCm, no relaxation | ✓ (initial) |
| Tiered, ROCm Tier-C = 0 | Tiered framework with ROCm tolerance initially 1e-12 | |

**User's choice:** Strict 1e-12 everywhere (initial) — later SUPERSEDED by the f32 project revision below.

### Q2 — Determinism mechanism
| Option | Description | Selected |
|--------|-------------|----------|
| Integer histograms + CPU-resident FP | Integer-quantized reductions + transcendentals stay on CPU | |
| Ordered f64 accumulation | Fixed sequential reduction order, f64, bit-reproducible | ✓ (initial) |
| Both, decide in P4 | Document both, defer binding choice | |

**User's choice:** Ordered f64 accumulation (initial) — later SUPERSEDED by standard f32 accumulation.

### Q3 — score_t width (asked, then overtaken by the revision)
Question on whether to build the C++ reference with `SCORE_T_USE_DOUBLE`. The user paused, then issued a project-wide revision: **shift from f64 to f32 to align with the C++ source defaults**, relax the oracle to ~1e-6, and drop the integer-quantized histogram strategy.

### Q4 — Propagation of the f32 / 1e-6 revision
| Option | Description | Selected |
|--------|-------------|----------|
| Update all project docs now | Edit PROJECT/REQUIREMENTS/ROADMAP then write CONTEXT | ✓ |
| CONTEXT.md only, flag the rest | Capture in CONTEXT, leave docs contradictory | |
| Pause discuss, revise docs first | Stop and treat as milestone-scope change | |

**User's choice:** Update all project docs now.
**Notes:** Final locked contract — f32 single-precision end-to-end, ~1e-6 absolute on every backend, standard f32 accumulations (no integer quantization), C++ reference built at default `float` width. PROJECT.md, REQUIREMENTS.md, ROADMAP.md, and STATE.md were updated in this session to match.

---

## C++ reference & goldens

### Q1 — Reference acquisition
| Option | Description | Selected |
|--------|-------------|----------|
| Build from the LightGBM/ submodule | CMake build of vendored source with deterministic flags | ✓ |
| Require a separately-installed lightgbm | System/pip LightGBM 4.6 provided by developer | |
| Decide in research | Researcher recommends | |

**User's choice:** Build from the LightGBM/ submodule.

### Q2 — Golden storage
| Option | Description | Selected |
|--------|-------------|----------|
| Committed fixtures + regen script | Generate once, commit, regenerate idempotently | ✓ |
| Regenerate every test run | Build C++ + produce goldens each run | |
| Decide in research | Researcher recommends | |

**User's choice:** Committed fixtures + regen script.

### Q3 — Golden emission
| Option | Description | Selected |
|--------|-------------|----------|
| Small C++ harness linking the lib | Extractor dumps RNG + per-stage intermediates; CLI for end-to-end | ✓ |
| CLI + model-text only | Only CLI train/predict + model .txt | |
| Decide in research | Researcher recommends | |

**User's choice:** Small C++ harness linking the lib.

---

## Workspace crate split

### Q1 — Granularity
| Option | Description | Selected |
|--------|-------------|----------|
| Full crate-per-responsibility now | All responsibility crates up front (many stubs) | |
| Minimal now, split as phases land | lgbm-core + lgbm-compute skeleton + oracle harness; add later | ✓ |
| Decide in planning | Planner finalizes crate list | |

**User's choice:** Minimal now, split as phases land.

### Q2 — Root structure
| Option | Description | Selected |
|--------|-------------|----------|
| Virtual workspace, crates under crates/ | Virtual manifest, members under crates/, remove hello-world | ✓ |
| Keep root package + workspace | lightgbm_rs as umbrella/facade crate | |
| Decide in planning | Planner decides virtual-vs-root | |

**User's choice:** Virtual workspace, crates under `crates/` (lgbm-* naming, hello-world removed).

---

## Config modeling

### Q1 — Generation approach
| Option | Description | Selected |
|--------|-------------|----------|
| Hand-port, verified by a checker | Hand-write struct/aliases/defaults/validation + drift-checker test | ✓ |
| Code-generate from config.h | build.rs/xtask parses config.h to emit Rust | |
| Decide in research | Researcher recommends | |

**User's choice:** Hand-port, verified by a checker.

### Q2 — Config shape
| Option | Description | Selected |
|--------|-------------|----------|
| Single flat struct (mirror C++) | One flat Config, ~110 fields, 1:1 with C++ | ✓ |
| Grouped/nested structs | Sub-structs by domain | |
| Decide in planning | Planner chooses | |

**User's choice:** Single flat struct mirroring C++ 1:1.

---

## Claude's Discretion

- Exact fixture file formats, the pinned rust-toolchain version, internal error-type taxonomy, and CMake invocation details — deferred to research/planning.

## Deferred Ideas

- Full crate-per-responsibility split (added incrementally per phase).
- CLI/example binary to replace the removed hello-world.
- Empirical CPU↔ROCm f32 transcendental parity validation (Phase 4/6).
