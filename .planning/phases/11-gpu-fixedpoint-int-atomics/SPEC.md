# Phase 11 — Fixed-point integer-atomic GPU histogram build

**Status:** scoping (spike-validated, ready to plan)
**Origin:** spikes 018 (`fixedpoint-int-atomics`) + 019 (`int-atomic-contention-regime`),
research Q2 / finding #3. Continues the `09-gpu-hist-build-perf` line.

## Goal

Replace the ROCm histogram BUILD's **f32 atomic accumulation** with **wide fixed-point
integer accumulation** (`u64` two's-complement, scale `S = 2^30`), delivering — on the
device-time proxy — **~1.3–1.7× faster build** in the wide large-leaf regime, **~3600×
better accuracy** (5.9e-9 vs f32 2.2e-5 rel-to-anchor; exact in the cancelling regime),
and a **deterministic** GPU histogram (order-independent integer add), while staying
within the ~1e-6 ROCm parity contract vs the CPU f64 anchor.

## Why (spike evidence)

- **Parity gate PASS** (spike-018a, CPU probe): i64@2^30 within ~1e-6 on all
  distributions incl. cancelling (exact); i32 overflows ⇒ i64/u64 required.
- **Speed validated** (spike-018b + 019, gfx1152 APU, 2 process runs, sign-stable):
  ~1.3–1.7× in heavy-atomic-load regimes (the wide root/large leaves that dominate
  work); composes with row-partition (survives P=16); null only at light load.
- **Mechanism:** f32 `atomicAdd` on RDNA = CAS retry loop (`ds_cmpst`) that saturates
  under contention; integer `ds_add_u64` = native single-instruction.

## Scope (what to build)

1. **u64 two's-complement fixed-point build kernel** in the resident path
   (`construct_hist_kernel_lds_f32` / the resident `build_resident_leaf` kernel) — quantize
   `round(value * 2^30)` as i64, store bits as u64, wrapping `fetch_add`, dequantize
   `(bits as i64) / 2^30` on read. Reference: `examples/gpu_fixedpoint_i64.rs`,
   `examples/gpu_int_vs_f32_psweep.rs`.
2. **i64/u64 histogram buffers** (resident pool + merge) — 2× bytes vs f32. Audit the
   resident-pool / subtract / fix / scan path for the wider cell type.
3. **Overflow guard** for extreme leaves: i64@2^30 is safe to ~1e9 rows × |g|≤8; add a
   scale/range check or clamp for pathological inputs (document the bound).
4. **cubecl-hip 0.10 constraint:** use `Atomic<u64>` (NOT `Atomic<i64>` — its `store`
   lowers to `atomicExch(long long*)` which HIP lacks; spike-018b).
5. **Optional compose:** integer atomics + spike-017 per-warp replication (both relieve
   the same CAS-retry contention) — measure if worth it.

## Hard gates / constraints

- **Oracle parity RE-PIN** on gfx1100/APU: the GPU f32→fixed-point change alters the GPU
  accumulation numerics (to MORE accurate). Re-pin `kernel_parity` / `learner_parity` /
  `boosting_parity` to the CPU f64 anchor within the ~1e-6 gate. **Never** GPU-vs-GPU
  (def-f8u-01). The new path should clear the gate MORE easily (it's closer to the anchor).
- **CPU f64 anchor UNTOUCHED** — it is the bit-exact merge gate.
- Feature-gated to `rocm`; CPU-only build emits zero fixed-point codegen.
- Measurement is the **device-time proxy** (8-CU APU; wall-clock unvalidatable here);
  discrete-gfx110x confirmation is ideal but not blocking (option-ii, user-accepted).

## Out of scope

- Changing CPU routing (GPU still loses to the 16-core CPU overall; this revives the
  ROCm path's *quality + speed* but does not by itself make GPU the default).
- Quantized-grad approximate training (phase 10, separate; this is EXACT-gradient
  fixed-point, not int16 discretization).

## Evidence / reference

`.planning/spikes/018-fixedpoint-int-atomics/README.md`,
`.planning/spikes/019-int-atomic-contention-regime/README.md`,
examples `fixedpoint_parity_probe.rs` / `gpu_fixedpoint_i64.rs` / `gpu_int_vs_f32_psweep.rs`.
