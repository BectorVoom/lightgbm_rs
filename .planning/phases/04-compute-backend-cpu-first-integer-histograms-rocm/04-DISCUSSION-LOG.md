# Phase 4: Compute Backend (CPU-first f32 histograms → ROCm) - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-06-05
**Phase:** 4-compute-backend-cpu-first-integer-histograms-rocm
**Areas discussed:** Compute/learner boundary, Kernel golden strategy, ROCm gating posture, Determinism & validation anchor, Anchor chain reconciliation

---

## Compute / Tree-Learner Boundary

| Option | Description | Selected |
|--------|-------------|----------|
| Thin device primitives | Low-level ops (accumulate histogram, prefix-scan/reduce, partition-by-threshold); ALL split-gain math in the Phase-5 learner | |
| Whole-kernel ops | Coarse ops matching CMP-05: construct_histograms, find_best_split (gain formula inside kernel), data_partition; Phase 5 orchestrates | ✓ |
| Split the difference | Histogram + partition as device kernels; best-split-finding stays scalar CPU in the learner, kernelized later | |

**User's choice:** Whole-kernel ops
**Notes:** `find_best_split` carries the gain formula → Phase 4 implements TRL-04 math early; research must define which gain parameters flow into the kernel and flag the Phase-5 overlap so it isn't double-counted. (D-01/D-01a)

---

## Kernel Golden / Validation Strategy

| Option | Description | Selected |
|--------|-------------|----------|
| Header-only C++ transcription | Extend xtask to transcribe C++ histogram/split/partition routines header-only, emit goldens on synthetic inputs, commit them (Phases 1-3 fallback) | ✓ |
| Scalar Rust reference as oracle | Independent sequential scalar-Rust reference treated as the ~1e-6 oracle; kernels validated against it | |
| Both: C++ golden + scalar ref | Capture C++ goldens AND keep a scalar reference; strongest, most work | |

**User's choice:** Header-only C++ transcription
**Notes:** No Phase-5 learner exists and the full C++ treelearner is unbuildable (`external_libs` unvendored). Synthetic inputs must cover dense/sparse, default-bin skip, missing/zero routing, bit widths, grad/hess spread. (D-02/D-02a)

---

## ROCm Gating Posture

| Option | Description | Selected |
|--------|-------------|----------|
| CPU-solid now, ROCm best-effort | cubecl-cpu fully oracle-gated this phase; ROCm brought up + oracle run, gaps recorded as known issues rather than blocking completion | ✓ |
| ROCm is a hard blocking gate | Phase incomplete until oracle passes on actual ROCm GPU (SC#5 literal) | |
| Empirically scope first | Spike CubeCL on the local ROCm box, then decide the gate | |

**User's choice:** CPU-solid now, ROCm best-effort
**Notes:** CubeCL v0.10 alpha + ROCm gaps flagged HIGH risk (STATE.md). Completion bar = CPU gate green + ROCm executed with gaps documented (no silent pass); residual ROCm gap is a tracked follow-up, ORA-04 full pass remains the standing target. (D-03/D-03a)

---

## Determinism & Validation Anchor

| Option | Description | Selected |
|--------|-------------|----------|
| Sequential ref = bit-exact anchor | A sequential reference is the deterministic anchor (bit-exact); cubecl-cpu and ROCm match within ~1e-6 | ✓ (refined — see Anchor chain) |
| ~1e-6 everywhere, no bit-exact tier | All paths compared at ~1e-6 uniformly | |
| You decide | Research picks the posture | |

**User's choice:** Sequential ref = bit-exact anchor — then refined in the follow-up below.
**Notes:** Refined by the Anchor-chain question: the *cubecl-cpu single-threaded path itself* is the sequential bit-exact anchor (no separate scalar port). (D-04)

---

## Anchor Chain Reconciliation (follow-up)

| Option | Description | Selected |
|--------|-------------|----------|
| Yes — scalar ref + CubeCL | Maintain a separate scalar-Rust reference (bit-exact to golden) alongside the CubeCL kernel; two impls per kernel | |
| No — cubecl-cpu IS the anchor | Single-threaded cubecl-cpu kernel reproduces the C++ golden bit-exact; cubecl-hip matches it within ~1e-6; one impl per kernel | ✓ |
| You decide | Research determines whether a scalar reference is needed | |

**User's choice:** No — cubecl-cpu IS the anchor
**Notes:** One Rust implementation per kernel. Assumes cubecl-cpu is bit-deterministic single-threaded; if bring-up shows it cannot be made bit-stable at f32, relax the cubecl-cpu anchor to ~1e-6 and re-evaluate — flag early (D-04a).

---

## Claude's Discretion

- `Backend` trait method signatures + `Runtime` associated-type binding; kernel buffer/launch/allocation API.
- `Plane`-API capability-gating mechanism + deterministic sequential-fallback structure (bounded by CMP-04 / SC#4).
- Precise gain-config parameter struct passed into `find_best_split` (bounded by D-01a).
- Synthetic-input fixture format + histogram/split/partition golden serialization.
- cubecl-cpu vs cubecl-hip feature-flag / runtime-selection mechanism (bounded by CMP-03).

## Deferred Ideas

- Tree-learner orchestration (Phase 5, TRL-01..09); GBDT spine/objectives/metrics (Phase 6); DART/RF/GOSS (Phase 7).
- f32 transcendental CPU↔ROCm parity (Phase 6; note any early ROCm-bring-up signal).
- Parallel (rayon) CPU histogram path — later, separately-validated optimization that must match the deterministic anchor.
- Integer-quantized/discretized histograms (QNT-01) + linear-tree kernels (LIN-01) — v2, out of scope.
- Residual ROCm oracle gap (if any) — tracked Phase-4 follow-up per D-03a.
