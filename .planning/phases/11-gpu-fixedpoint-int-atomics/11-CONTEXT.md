# Phase 11: gpu-fixedpoint-int-atomics - Context

**Gathered:** 2026-06-22
**Status:** Ready for planning
**Source:** Spike campaign (018 fixedpoint-int-atomics + 019 int-atomic-contention-regime). The spike process WAS the context/research gathering — this phase wires the validated lever.

<domain>
## Phase Boundary

Replace the ROCm histogram BUILD's f32 atomic accumulation with wide fixed-point integer
(`u64` two's-complement, scale `S=2^30`) accumulation, in the GPU resident build path.
Delivers (device-time proxy): ~1.3–1.7× faster build on wide large-leaves, ~3600× better
accuracy vs the f64 anchor, and a deterministic GPU histogram — within the ~1e-6 ROCm gate.
CPU f64 anchor is UNTOUCHED. Does not change CPU/GPU routing.
</domain>

<decisions>
## Implementation Decisions (LOCKED — spike-validated)

### Accumulation
- Quantize each grad/hess as `round(value * 2^30)` → i64, store its BITS as `u64`; wrapping
  `Atomic<u64>::fetch_add` == two's-complement i64 add; dequantize `(bits as i64) / 2^30` on read.
- Scale = `2^30` (spike-018a: within ~1e-6, exact in cancelling regime; never overflows i64 for ≤~1e9 rows).
- **MUST use `Atomic<u64>`, NOT `Atomic<i64>`** — cubecl-hip 0.10 lowers `Atomic<i64>::store` to
  `atomicExch(long long*)` which HIP lacks (spike-018b, hard constraint).

### Buffers
- Resident histogram + sub-hist + merge buffers become `u64`/`i64` (2× bytes vs f32). Audit the
  resident pool, subtract-trick, fix, and scan path for the wider cell type.

### Overflow guard
- i64@2^30 safe to ~1e9 rows × |g|≤8. Add a documented scale/range check or clamp for pathological
  extreme leaves.

### Scope of the kernel change
- Target the resident build path (`construct_hist_kernel_lds_f32` / the resident
  `build_resident_leaf` kernel + launcher). Reference working kernels:
  `examples/gpu_fixedpoint_i64.rs`, `examples/gpu_int_vs_f32_psweep.rs`.
- Optional (measure, don't assume): compose with spike-017 per-warp replication.

### Claude's Discretion
- Exact buffer-type plumbing through the resident pool / subtract / scan.
- Whether to keep an f32 path as a fallback/feature-flag or replace outright.
- Test/bench harness shape for the parity re-pin and the end-to-end wide-train confirmation.
</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Spike evidence (the validated design)
- `.planning/phases/11-gpu-fixedpoint-int-atomics/SPEC.md` — full phase scope.
- `.planning/spikes/018-fixedpoint-int-atomics/README.md` — parity gate + GPU feasibility/speed.
- `.planning/spikes/019-int-atomic-contention-regime/README.md` — regime confirmation (~1.3–1.7×, composes with row-partition).
- `crates/lgbm-compute/examples/gpu_fixedpoint_i64.rs` — working u64 two's-complement kernel + dequant.
- `crates/lgbm-compute/examples/gpu_int_vs_f32_psweep.rs` — row-partitioned f32/u64 twins.

### Production code to modify
- `crates/lgbm-compute/src/kernels/histogram.rs` — resident LDS build kernel + launcher.
- `crates/lgbm-compute/src/lib.rs` — `build_resident_leaf` seam / resident pool.
- `crates/lgbm-treelearner/src/resident_pool.rs` + `learner.rs` — resident buffers + subtract/scan.

### Parity gates (hard)
- oracle-harness `kernel_parity` / `learner_parity` / `boosting_parity` (rocm) — RE-PIN to CPU f64 anchor within ~1e-6.
- def-f8u-01 guardrail: never pin two nondeterministic GPU f32 paths to each other; pin to the anchor.
</canonical_refs>

<specifics>
## Specific Ideas

- Determinism is a free side-benefit (integer add is order-independent) — could later allow
  GPU-vs-GPU reproducibility and tightening def-f8u-01, but that's not required this phase.
- Measurement is the CubeCL device-time proxy (8-CU APU); discrete-gfx110x wall-clock confirmation
  is ideal but not blocking (user-accepted option-ii).
</specifics>

<deferred>
## Deferred Ideas

- Composing with spike-017 per-warp replication (measure separately if this phase's win wants more).
- Making the ROCm backend a default speed path (routing is a separate decision; GPU still loses to 16-core CPU overall).
- End-to-end discrete-GPU wall-clock validation (no hardware).
</deferred>

---

*Phase: 11-gpu-fixedpoint-int-atomics*
*Context gathered: 2026-06-22 via spike campaign 018/019*
