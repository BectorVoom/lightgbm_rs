# Phase 16: On-Device Histogram Constructor - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-06-30
**Phase:** 16-on-device-histogram-constructor
**Areas discussed:** Subtract numeric domain, Arena & handle protocol, Build-kernel structure, Verification fixtures

---

## Subtract numeric domain

| Option | Description | Selected |
|--------|-------------|----------|
| De-quant after build, fix+subtract in hist_t | Confine u64 fixed-point to BUILD shared-accumulation; de-quant the two-tier merge once (cpu=f64, ROCm=f32), then Fix/Subtract in hist_t — faithful to C++ all-double, reuses subtract.rs, clean anchor | ✓ |
| Integer-domain subtract | Keep parent+smaller in u64 fixed-point, exact integer subtract, de-quant once after — exactly reproducible but diverges from C++ rounding, separate anchor argument, Fix mixes float total with int bins | |

**User's choice:** De-quant after build, fix+subtract in hist_t (D-01)
**Notes:** u64 fixed-point stays in its Phase-11 BUILD role only. Subtraction trick reproduced verbatim against the cpu f64-fold anchor.

---

## Arena & handle protocol

| Option | Description | Selected |
|--------|-------------|----------|
| Arena + explicit handle contract; swap deferred to 18 | Phase 16 owns a pre-allocated-once arena + `{parent, smaller=fresh_slot, larger=parent_alias}` contract; demonstrates pointer rotation in-place; anchor-tested in isolation; §9 SplitTreeStructure swap = Phase 18 | ✓ |
| Build the full pool rotation manager now | Implement the whole-tree §9 pointer-swap in Phase 16 — crosses into Phase 18 scope | |

**User's choice:** Arena + explicit handle contract; swap deferred to 18 (D-02)
**Notes:** Larger child derived in-place in the parent buffer; smaller in a fresh slot. Cross-tree pool management stays Phase 18.

---

## Build-kernel structure

| Option | Description | Selected |
|--------|-------------|----------|
| Net-new two-tier on §13 geometry; spill real now | NET-NEW kernel (blockIdx.x=partition, threadIdx.x=column; LDS block-local u64 atomics → cross-block atomicAdd_system), reusing Phase-11 u64 primitive + landed LDS idiom; shipped per-feature ROCm build untouched; global spill built+verified now | ✓ |
| Bend the existing ROCm per-feature build | Retrofit the one-cube-per-feature + row-partition build onto the new geometry — risks the byte-unchanged ROCm path, doesn't match §7 two-tier | |
| Net-new kernel but skeleton the global spill | Same net-new kernel but stub the NumLargeBinPartition>0 spill path | |

**User's choice:** Net-new two-tier on §13 geometry; spill real now (D-03, D-04)
**Notes:** Shipped per-feature ROCm build coexists byte-unchanged (Phase-15 D-02). Global-memory spill anchored by the Phase-15 synthetic large-bin column.

---

## Verification fixtures

| Option | Description | Selected |
|--------|-------------|----------|
| Corpora + synthetic columns + targeted Fix/ordering tests | Dense corpora + Phase-15 synthetic sparse (row_ptr 16/32/64) & large-bin columns, bit-exact to cpu f64 fold; purpose-built most_freq_bin≠0 column for FixHistogram (DEF-07-02); explicit build-before-subtract ordering test (8aed100-class) + interleaved [2b]/[2b+1] assert; never GPU-vs-GPU | ✓ |
| Corpora-only | Anchor only against committed dense corpora — leaves sparse/spill, most_freq_bin≠0 Fix, and the ordering invariant unexercised | |

**User's choice:** Corpora + synthetic columns + targeted Fix/ordering tests (D-05)
**Notes:** Targets the exact places parity historically broke.

---

## Claude's Discretion

- Geometry tunables (`NUM_DATA_PER_THREAD`, `NUM_THREADS_PER_BLOCK`, `grid_dim_y` floor 160, shared-hist sizes) — parity-neutral occupancy knobs; start from faithful C++ constants, APU-autotune deferred.
- Exact CubeCL module placement and the concrete `hist_t**` rotation handle/enum representation.
- Whether de-quant is a fused merge tail or a separate pass.

## Deferred Ideas

- §9 SplitTreeStructure pool pointer-SWAP + whole-tree pool management → Phase 18.
- Discretized/quantized build+fix+subtract (§7.3) → v2 (QGD-02).
- APU-aware autotune of the build geometry (Phase-13 reuse) → deferred perf option.
- On-device best-split (reads hist_in_leaf) → Phase 17.
