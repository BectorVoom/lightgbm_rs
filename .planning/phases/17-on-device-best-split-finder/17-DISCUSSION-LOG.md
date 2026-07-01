# Phase 17: On-Device Best-Split Finder - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-07-01
**Phase:** 17-on-device-best-split-finder
**Areas discussed:** CPU-anchor source, 256-bin within-feature scan, Categorical/spill scope, Kernel fidelity/flags

---

## CPU-anchor source — anchor core

| Option | Description | Selected |
|--------|-------------|----------|
| New CUDA-core f64 fold | Single-owner CubeDim(1) f64 fold faithful to the CUDA core's accumulation (prefix-sum order, cnt_factor/__double2int_rn, complement-from-parent). D-06 pattern. | ✓ |
| Reuse split.rs host scan | Anchor to the existing verbatim host FindBestThresholdSequentially; simpler but different rounding/accumulation path could hide a divergence. | |

**User's choice:** New CUDA-core f64 fold
**Notes:** The device kernel is anchored to a purpose-built CUDA-core-faithful fold, not the shipped host serial scan — the two take different accumulation paths.

## CPU-anchor source — gain math

| Option | Description | Selected |
|--------|-------------|----------|
| Reuse crate::gain | Reuse the shipped #[cube] gain primitives split.rs already calls in both fold and kernel. | |
| Transcribe CUDA gain helpers | Write fresh gain helpers faithful to the CUDA device functions rather than trusting host-identity. | ✓ |

**User's choice:** Transcribe CUDA gain helpers
**Notes:** Parity-conservative — transcribe GetSplitGains / CalculateSplittedLeafOutput / GetLeafGainGivenOutput faithfully; research must diff against crate::gain and document any delta before a shared #[cube] is accepted.

---

## 256-bin within-feature scan — scan reuse

| Option | Description | Selected |
|--------|-------------|----------|
| Reuse block_scan primitive | Reuse the shipped segmented block_scan (256 <= 1024, LDS + sync_cube). | |
| Purpose-built per-task scan | New within-task scan specialized to the interleaved [2b]/[2b+1] layout + forward/reverse direction. | ✓ |
| You decide (researcher) | Leave to research to pick reuse if it fits. | |

**User's choice:** Purpose-built per-task scan
**Notes:** May borrow the primitives.rs LDS/sync_cube idiom, but not the generic block_scan's segment contract/shape. Resolves the ROADMAP research flag (plane-sum caps at 32/64 << 256).

---

## Categorical/spill scope — cat scope

| Option | Description | Selected |
|--------|-------------|----------|
| Numeric core + wire cat seam | Numerical stage-1 core; wire the SplitFindTask categorical dispatch seam; cat eval math → Phase 22. | ✓ |
| Numeric core, no cat plumbing | Numerical only; Phase 22 adds both seam and core. | |
| Build categorical too | Port the categorical inner core this phase (scope creep vs Phase 22). | |

**User's choice:** Numeric core + wire cat seam

## Categorical/spill scope — spill path

| Option | Description | Selected |
|--------|-------------|----------|
| Shared-path only; defer spill | Shared/LDS stage-1 only (covers max_bin<=255); defer _GlobalMemory. | |
| Build spill too | Port the _GlobalMemory stage-1 variant now, anchored by the Phase-15 large-bin column. | ✓ |
| You decide (researcher) | Build spill only if a fixture exercises >256 bins. | |

**User's choice:** Build spill too
**Notes:** Full >256-bin coverage this phase; reuses Phase 16 D-04's synthetic large-bin/global-spill fixture. Discretized global-memory path stays out (C++ TODO + discretized skipped).

---

## Kernel fidelity/flags — stages

| Option | Description | Selected |
|--------|-------------|----------|
| 3 faithful stages | Keep stage1/stage2/stage3 separate + ReduceBestGain family + single 8-int readback. | ✓ |
| Fuse for the APU | Collapse stages on the small APU; perf-motivated but risks reduction-order + export-contract parity. | |

**User's choice:** 3 faithful stages

## Kernel fidelity/flags — comptime flag fan-out

| Option | Description | Selected |
|--------|-------------|----------|
| Default + IS_LARGER + L1 | Wire USE_L1 + IS_LARGER; keep USE_RAND=false, USE_SMOOTHING=false (matching split.rs host scope). | |
| Full fan-out now | Wire + anchor all four flags including USE_RAND/extra-trees + USE_SMOOTHING/path_smooth. | ✓ |
| You decide (researcher) | Wire+anchor only the flags a fixture needs. | |

**User's choice:** Full fan-out now
**Notes:** Expands the fixture matrix — needs extra-trees RNG-stream goldens (Phase-14 CUDARandom bit-identical stream) and path_smooth smoothing goldens beyond the default template. Flagged for research.

---

## Claude's Discretion

- Exact CubeCL module placement (new `best_split.rs` vs extend `split.rs`), reusing
  `split_info.rs` records + the 8-int export shape.
- Whether stage-2's `…AllBlocks` fold is a separate kernel or folded when
  `num_blocks_per_leaf == 1` (parity-neutral).
- Geometry tunables (256/256/1024 block dims, smaller/larger stream split) — occupancy
  knobs with no parity impact; APU-aware autotune deferred.

## Deferred Ideas

- Categorical inner core (one-hot + many-cat bitonic-argsort sweep, cat_threshold list) → Phase 22.
- Discretized / quantized split finder → v2 (QGD-02).
- Data partition / tree mutation / prediction (Split-before-partition) → Phase 18.
- APU-aware autotune of the best-split geometry → deferred perf option.
