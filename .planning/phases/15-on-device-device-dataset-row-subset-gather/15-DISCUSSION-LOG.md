# Phase 15: On-Device Device Dataset + Row-Subset Gather - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-06-30
**Phase:** 15-on-device-device-dataset-row-subset-gather
**Areas discussed:** Representation scope, Feature-partition layout, Dense + sparse scope, CopySubrow selection locus

---

## Representation scope

| Option | Description | Selected |
|--------|-------------|----------|
| §13 row store + row-wise CopySubrow only | Build only the row-wise feature-partition store + its bagging gather; defer §3 column store + prediction-side CopySubrow to Phase 18 (Claude's recommendation). | |
| Both §3 column + §13 row stores | Port both stores + both CopySubrow variants this phase, matching the full C++ dual representation up front. | ✓ |

**User's choice:** Both §3 column + §13 row stores.
**Notes:** User chose to materialize the complete C++ dataset surface now rather than defer the column store/prediction gather. → CONTEXT D-01.

---

## Feature-partition layout

| Option | Description | Selected |
|--------|-------------|----------|
| Adopt C++ shared-hist grouping; ROCm path untouched | Build §13 DivideCUDAFeatureGroups + large-bin spill for the on-device route; leave the shipped per-feature ROCm kernel byte-unchanged (Claude's recommendation). | ✓ |
| Reuse existing per-feature geometry | Keep one-cube-per-feature + autotuned row-partition for the on-device path; skip the C++ grouping. | |

**User's choice:** Adopt C++ shared-hist grouping; ROCm path untouched.
**Notes:** Sets up Phase-16 §7 parity (blockIdx.x = partition). The two geometries coexist. → CONTEXT D-02.

---

## Dense + sparse scope

| Option | Description | Selected |
|--------|-------------|----------|
| Dense full (3 widths + spill); sparse CSR as skeleton | Full dense parity now; sparse CSR as an anchor-pinned skeleton hardened by its first consumer (Phase-14 D-02 pattern, Claude's recommendation). | |
| Full dense + sparse CSR parity now | Port and validate the complete 3×3 dense+sparse matrix this phase. | ✓ |

**User's choice:** Full dense + sparse CSR parity now.
**Notes:** Broader than recommended. Triggered the follow-up "Sparse anchor" question below. → CONTEXT D-03.

### Follow-up — Sparse anchor

| Option | Description | Selected |
|--------|-------------|----------|
| Add a sparse test corpus this phase | Commit a high-sparsity fixture and pin device CSR to host bins across 3 ptr widths. | |
| Anchor sparse to host Rust bins, no new corpus | Validate device CSR == host sparse binning on synthetic in-test data; widths may not all be hit. | |
| Generate sparse in-test across all 3 ptr widths | Synthesize in-test sparse columns forcing each row_ptr_type{16,32,64} (nnz crossing 2^16/2^32) + large-bin spill, anchored to host bins. | ✓ |

**User's choice:** Generate sparse in-test across all 3 ptr widths.
**Notes:** Deterministic 3×3 coverage without a committed corpus; host Rust bins are the (already C++-bit-exact) anchor. → CONTEXT D-04.

---

## CopySubrow selection locus

| Option | Description | Selected |
|--------|-------------|----------|
| Host selects indices + uploads; device gathers | Reuse host bagging/GOSS RNG to produce used_indices, upload, device gathers; on-device bagging RNG deferred (Claude's recommendation). | |
| On-device row selection via CUDARandom | Run the subset draw itself on-device via Phase-14 CUDARandom, then gather. | ✓ |

**User's choice:** On-device row selection via CUDARandom.
**Notes:** Broader than recommended. Triggered the follow-up "Selection scope" question below. → CONTEXT D-05.

### Follow-up — Selection scope

| Option | Description | Selected |
|--------|-------------|----------|
| Bagging on-device now; GOSS selection when grads land (Phase 19+) | Port block-RNG bagging draw on-device now (bit-exact vs host BAGGING_RAND_BLOCK); GOSS *selection* defers to Phase 19; GOSS gather still works from host indices. | ✓ |
| Both bagging + GOSS selection on-device now | Port both, pulling the gradient dependency (Phase 19) + percentile skeleton forward. | |

**User's choice:** Bagging on-device now; GOSS selection when grads land (Phase 19+).
**Notes:** Avoids a forward dependency on Phase-19 gradients; the CopySubrow gather works for any index set this phase. → CONTEXT D-06.

---

## Claude's Discretion

- CubeCL module placement of the column store, row/partition store, and CopySubrow.
- The runtime width-dispatch surface (C++ `void* const*` table → CubeCL enum / comptime).
- Whether the dense/CSR re-lay shares helper code with the existing host binning extraction.

## Deferred Ideas

- On-device GOSS *selection* → Phase 19+ (gather works now).
- Prediction wiring on the §3 column store → Phase 18.
- Quantized/discretized integer dataset path (§4) → v2.
- Any actual on-device tree growth → Phase 21.
