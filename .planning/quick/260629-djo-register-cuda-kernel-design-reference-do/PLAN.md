---
quick_id: 260629-djo
slug: register-cuda-kernel-design-reference-do
date: 2026-06-29
type: docs-planning
touches_code: false
---

# Quick Task: Register CUDA kernel design reference into v1.1 milestone planning

## Goal

Register the new reference document `docs/cuda-kernel-design.md` into the GSD
planning for the **v1.1 "GPU Training-Speed: CUDA On-Device Tree Learner"** milestone
(Phases 14–19), which ports the C++ `CUDASingleGPUTreeLearner` and its kernels. The doc
is the authoritative, source-verified map of the C++ CUDA backend being ported. This is
a **documentation + planning task only — no code changes.**

## What the reference doc is (facts to cite)

`docs/cuda-kernel-design.md` (~1190 lines) — comprehensive design reference for
LightGBM's public CUDA backend, written as the porting reference for the Rust/CubeCL
rewrite, **verified kernel-by-kernel against the read-only `LightGBM/` C++ source**:

- **58 CUDA source files** across 7 directories (full inventory table in the doc).
- **81 distinct `__global__` kernels** — every one named and documented; audited
  complete against source.
- **11 subsystems**, each a doc section: §2.4 shared device primitives, §3 dataset
  (CUDAColumnData), §4 gradient discretizer, §5 objectives (gradient/hessian), §6 leaf
  splits + tree-learner driver, §7 histogram constructor (13 kernels), §8 best split
  finder (3-stage pipeline), §9 data partition (mark→prefix-sum→scatter), §10 tree
  model I/O + prediction, §11 score updater, §12 metrics; plus §13 CUDARowData device
  layout, §14 host-infrastructure classes, §15 device structs (CUDASplitInfo/CUDARandom),
  §16 end-to-end per-iteration sequencing, §17 port considerations for lightgbm_rs.
- Includes a **§0 per-kernel quick index** and **host/device boundary** for every
  subsystem.
- Documents parity-critical port constraints surfaced during verification:
  - `CUDATree.Split` runs **before** `DataPartition.Split` (returns `right_leaf_index`).
  - Histogram subtraction trick + most-frequent-bin omission/fix are correctness (not
    just speed) requirements; interleaved `[2b]/[2b+1]` layout; `hist_t=double` durable,
    SP-f32 shared atomics non-deterministic (the documented f32-vs-f64 / ROCm residual).
  - Quantized (int16-packed) path is integer-exact → the natural bit-exact GPU target.
  - **CUDA-support boundaries**: CUDA *objectives* = exactly 11 (no MAPE/Gamma/Tweedie/
    xentropy); CUDA *metrics* = 12 pointwise (no AUC/NDCG/MAP/multiclass) — both
    asymmetries documented.

This directly underpins Phases 15–19 (Minimal On-Device Growth, On-Device Frontier
Best-Split, On-Device Data Partition, Feature Coverage), which mirror these exact
subsystems on-device.

## Edits (4 planning files — additive, surgical)

### 1. `.planning/REFERENCE_MANIFEST.md`
Insert a new section at the **top** of the file (immediately after the title line
`# Reference Manifest — Phase 7 Determinism Fixtures (D-05)`, before `## Pinned
Reference`), since the manifest currently only catalogs Phase-7 golden fixtures and has
no v1.1 entry. Use this content:

```markdown

---

## v1.1 CUDA On-Device Tree Learner — C++ Port-Source Map

**`docs/cuda-kernel-design.md`** — authoritative, source-verified design reference for
LightGBM's public CUDA backend (the read-only `LightGBM/` C++ tree being ported by the
v1.1 milestone, Phases 14–19). Covers **all 58 CUDA source files** and **all 81
`__global__` kernels** across **11 subsystems** (histogram constructor, best split
finder, data partition, leaf splits/driver, objectives, metrics, score updater, gradient
discretizer, tree I/O, CUDARowData, shared primitives), plus device structs, host
infrastructure, end-to-end per-iteration sequencing, and a lightgbm_rs port-considerations
section. Each kernel/device-helper/launcher named and verified kernel-by-kernel against
source (full-doc audit: 81/81 kernels named).

Use as the port reference for Phases 15–19 (on-device growth → frontier best-split →
data partition → feature coverage), which mirror `CUDASingleGPUTreeLearner` subsystem by
subsystem. Key parity constraints captured: `CUDATree.Split` precedes `DataPartition.Split`;
subtraction-trick + most-freq-bin-fix are correctness requirements; interleaved
`[2·b]/[2·b+1]` histogram layout; `hist_t=double` durable while SP-f32 shared atomics are
non-deterministic (the f32-vs-f64 / ROCm residual); the int16-packed quantized path is
integer-exact (natural bit-exact GPU target). CUDA-support boundaries: 11 CUDA objectives
(no MAPE/Gamma/Tweedie/xentropy), 12 CUDA pointwise metrics (no AUC/NDCG/MAP/multiclass).

_Verified against `LightGBM/` C++ source 2026-06-29 (quick task 260629-djo)._
```

### 2. `.planning/PROJECT.md`
In the **Current Milestone: v1.1** context (the paragraph block around the Phase-14 /
v1.1 status, ~line 17–19), append a sentence noting the reference, e.g. after the Phase-14
completion paragraph add:

```markdown

**Port reference:** `docs/cuda-kernel-design.md` is the authoritative, source-verified map
of the C++ CUDA backend being ported (58 files, 81 `__global__` kernels, 11 subsystems,
verified kernel-by-kernel against `LightGBM/`). It is the per-subsystem porting reference
for the on-device tree-learner slices (Phases 15–19) and records the parity-critical
constraints (split-before-partition ordering, subtraction-trick/most-freq-bin-fix
correctness, `hist_t=double` vs SP-f32 atomic nondeterminism, integer-exact quantized path,
and the CUDA objective/metric support boundaries).
```

### 3. `.planning/ROADMAP.md`
In the **`## Milestone v1.1 — GPU Training-Speed: CUDA On-Device Tree Learner`** section
(~line 106), add a one-line reference pointer near the milestone intro / before the phase
checklist (after the existing intro paragraph that mentions mirroring
`CUDASingleGPUTreeLearner`), e.g.:

```markdown
**C++ port-source map:** `docs/cuda-kernel-design.md` — source-verified design reference
for the full CUDA backend (58 files, 81 kernels, 11 subsystems) being mirrored on-device;
the per-subsystem porting reference for Phases 15–19. See `.planning/REFERENCE_MANIFEST.md`.
```

### 4. `.planning/STATE.md`
Add a row to the **"Quick Tasks Completed"** table for this task:

```markdown
| quick_task | 260629-djo-register-cuda-kernel-design-reference-do | complete (docs/planning only) |
```

## Acceptance
- [ ] `docs/cuda-kernel-design.md` referenced in REFERENCE_MANIFEST.md (new v1.1 section).
- [ ] PROJECT.md current-milestone context names the doc as the port reference.
- [ ] ROADMAP.md v1.1 milestone section points to the doc + manifest.
- [ ] STATE.md Quick Tasks Completed table has the 260629-djo row.
- [ ] No source/code files changed (only `.planning/*.md`; `docs/cuda-kernel-design.md`
      itself already committed/added separately).
- [ ] SUMMARY.md written; atomic commit made.

## Non-goals
- No changes to Rust/Python source, kernels, or tests.
- No new phases (Phases 14–19 already roadmapped); this only registers the reference.
