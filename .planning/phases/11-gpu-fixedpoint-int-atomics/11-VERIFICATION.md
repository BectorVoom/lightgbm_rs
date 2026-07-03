---
phase: 11-gpu-fixedpoint-int-atomics
verified: 2026-06-22T00:00:00Z
status: passed
human_verified: 2026-07-03T21:20:00Z
score: 8/8 must-haves verified (3 device-runtime truths confirmed on local ROCm 2026-07-03; see 11-UAT.md)
re_verification:
  previous_status: none
  previous_score: n/a
human_verification_completed:
  - test: "Run the re-pinned resident parity gate on the ROCm GPU"
    expected: "kernel_parity_resident_build_fix_compact_equals_host_on_hip PASSES: max_rel vs CPU f64 anchor <= FIXEDPOINT_REL_GATE (1e-7), and the 2-runs to_bits() determinism sub-assert holds"
    result: "PASS (2026-07-03, local ROCm gfx1100 spoofed 8-CU gfx1152 APU, HSA_OVERRIDE=11.0.0). Both P=1 and P>1 cells: max_rel_vs_cpu_f64_anchor=0.000e0 (bit-exact, gate 1e-7); P>1 row_partition_count(3,300000)=10 multi-cube merge also 0.000e0 — closes review WR-05. Determinism sub-assert holds."
  - test: "Run the device-time A/B example >=2x on the ROCm GPU"
    expected: "HEAVY wide 16x1M regime, u64 median ratio >= 1.0x (SEP-WIN at >=1 P), sign-stable across two process runs; LIGHT overlap acceptable"
    result: "PASS (2026-07-03, ≥2 process runs). HEAVY wide NOT-SLOWER both runs — run1 SEP-WIN at P=1 (1.64×) & P=8 (1.34×); run2 SEP-WIN every P (1.06×–1.41×). LIGHT not-regressed. SEP sign stable. Absolute Mr/s APU-confounded and disregarded per methodology."
  - test: "Confirm the unchanged non-resident f32 + bit-exact f64 subtract/construct pins stay green on the GPU"
    expected: "rocm_row_partition (2/2) and rocm_backend_parity (bit-exact) PASS at their existing tolerances after the phase-11 changes"
    result: "PASS (2026-07-03). rocm_backend_parity 5/5 green (construct/subtract/find_best_split bit-exact + data_partition + default_left_tie); rocm_row_partition 2/2 green. Integer build did not disturb the post-dequant f64 paths."
---

# Phase 11: Fixed-point integer-atomic GPU histogram build Verification Report

**Phase Goal:** Replace the ROCm resident histogram BUILD's f32 atomics with wide fixed-point u64 (S=2^30) integer LDS atomics — targeting ~1.3–1.7× faster on the wide large-leaves regime, ~3600× more accurate, and deterministic, within the ~1e-6 ROCm parity gate. Must NOT change CPU routing or the CPU f64 deterministic anchor. Re-pin the resident-build parity gate to compare the LIVE u64 GPU path against the CPU f64 anchor, add a determinism assert, and add a device-time A/B harness confirming the integer build is not-slower in the wide regime.

**Verified:** 2026-06-22 (codebase) · **Hardware-confirmed:** 2026-07-03 (local ROCm)
**Status:** passed
**Re-verification:** No — initial verification; the 3 device-runtime truths were confirmed on hardware 2026-07-03 (see 11-UAT.md), flipping human_needed → passed

## Goal Achievement

### Observable Truths

| #  | Truth | Status | Evidence |
| -- | ----- | ------ | -------- |
| 1 | ROCm resident BUILD accumulates grad/hess as u64 two's-complement fixed-point (S=2^30) via integer LDS atomics, NOT f32 atomics | ✓ VERIFIED | `construct_leaf_hist_resident_lds_kernel_u64<B: Int>` (histogram.rs:1246) uses `SharedMemory::<Atomic<u64>>` (1261), `out: &mut Array<Atomic<u64>>` (1253), quantize `u64::cast_from(i64::cast_from(f32::round(ord_g[k]*SCALE_F32)))` (1280-81), wrapping `fetch_add` + LDS→global merge (1282-91). `SCALE_F32 = 2^30` (630). |
| 2 | fix_compact_kernel widen pass dequantizes u64 bits→i64→f64/2^30; everything downstream stays f64 unchanged | ✓ VERIFIED | Dequant `hist[wbi] = f64::cast_from(i64::cast_from(h_raw[wbi]))/SCALE_F64` (histogram.rs:2003-04), `SCALE_F64 = 2^30` (1987), `h_raw: &Array<u64>`. FixHistogram fold + compact below operate on already-f64 `hist`; subtract/scan/move/upload byte-untouched (SUMMARY git-diff confirmed, no `+`/`-` for them). |
| 3 | Overflow guard documents/enforces the i64@2^30 bound (~1e9 rows x \|g\|<=8) at the resident-build boundary | ✓ VERIFIED | Guard at histogram.rs:2271-2295: one-pass `max_abs` scan over leaf rows, `worst = rows*max_abs*2^30`, returns typed `ComputeError::Runtime` (no silent clamp) when `worst >= i64::MAX`. Spike-018 bound cited inline (2264-70). |
| 4 | CPU-only build (no rocm) compiles + emits ZERO fixed-point codegen; CPU f64 anchor kernels byte-untouched | ✓ VERIFIED | u64 kernel + dispatch + example all `#[cfg(feature="rocm")]` (1243, example:52/67). `cargo build -p lgbm-compute` (CPU-only) → Finished EXIT=0. `git diff 434efb3..HEAD` shows `construct_histograms_cpu`/CPU fold untouched. No `device_type`/routing change in diff. |
| 5 | Resident-build parity test re-pinned to a CPU f64 anchor at a TIGHTENED gate (not GPU-vs-GPU) | ✓ VERIFIED (code) / ? runtime | kernel_parity.rs:1748 builds anchor from `construct_histograms_cpu` + host `fix_histogram`/`host_compact_histogram` (1789-1806), GPU arm `build_fix_compact_resident_readback_f64_on` (1813), `FIXEDPOINT_REL_GATE = 1e-7` (1841) replacing the f32 `HIP_SANITY_REL=1e-3` envelope. def-f8u-01 honored. Runtime PASS is a GPU behavior → human. |
| 6 | Determinism assert proves the u64 build is bit-equal across two runs | ✓ VERIFIED (code) / ? runtime | kernel_parity.rs:1864-1876: second `run_gpu()`, asserts `a.to_bits() == b.to_bits()` over all read-back f64 cells. |
| 7 | Device-time A/B harness measures the LIVE u64 resident build vs an f32 twin in the wide regime, spike discipline | ✓ VERIFIED (code) / ? runtime | examples/gpu_fixedpoint_resident_ab.rs: u64 arm = LIVE `construct_leaf_hist_resident_lds_kernel_u64::launch_unchecked` (199), f32 arm = `build_f32_rp` twin (69) explicitly labelled NOT production (58-66). Interleaved median[p25..p75] (252-53), SEP-WIN `u75<f25` (255), accumulate-LAUNCHES-then-one-read (218-235), 2-regime + P sweep {1,4,8,16}. |
| 8 | Integer build is NOT slower in the wide regime (~1.3-1.7x on large leaves; light-regime overlap acceptable) | ✓ VERIFIED (code+captured) / ? runtime | SUMMARY 11-03 captured 2 process runs: HEAVY wide ratio >=1.0x every P (run1 1.13-1.31x, 3/4 SEP-WIN; run2 0.99-1.66x). Harness prints the verdict; the live-hardware sign is the operator gate → human. |

**Score:** 8/8 must-haves verified in the codebase (artifacts exist, are substantive, and are wired). 3 of the 8 (truths 5/6/7-8) carry a device-runtime confirmation component that cannot execute in this verifier environment.

### Required Artifacts

| Artifact | Expected | Status | Details |
| -------- | -------- | ------ | ------- |
| `crates/lgbm-compute/src/kernels/histogram.rs` | u64 fixed-point LDS build kernel + u64 RAW alloc + dequant widen + overflow guard + `fixed_point` dispatch flag | ✓ VERIFIED | Contains `Atomic<u64>` (1253/1261), `SCALE_F32`/`SCALE_F64` (630/1987), `i64::MAX` guard (2288), `fixed_point` flag dispatch (1768/1859). Substantive, not stub. |
| `crates/lgbm-compute/src/lib.rs` | `RocmBackend::build_resident_leaf` seam doc → fixed-point | ✓ VERIFIED | Doc at lib.rs:2240-2250 describes u64 two's-complement fixed-point accumulation under the ~1e-6 contract; body unchanged. |
| `crates/oracle-harness/tests/kernel_parity.rs` | resident u64 build re-pinned to CPU f64 anchor + tightened gate + determinism | ✓ VERIFIED | `construct_histograms_cpu` anchor (1799), `FIXEDPOINT_REL_GATE=1e-7` (1841), `to_bits()` determinism (1868). No GPU-vs-GPU. |
| `crates/lgbm-compute/examples/gpu_fixedpoint_resident_ab.rs` | device-time A/B u64-live vs f32-twin, wide regime | ✓ VERIFIED | Drives LIVE u64 kernel + `build_f32_rp` twin; median/p25/p75/SEP; compiles both configs. |
| `crates/lgbm-compute/tests/rocm_row_partition.rs` | confirmation-only classification comment | ✓ VERIFIED | +5-line phase-11 classification comment only; no numeric change (git diff). |

### Key Link Verification

| From | To | Via | Status | Details |
| ---- | -- | --- | ------ | ------- |
| `resident_raw_build_into` | `construct_leaf_hist_resident_lds_kernel_u64` | `launch_lds_u64!` under `if fixed_point` | ✓ WIRED | histogram.rs:1823-1864; u8/u16/u32 monomorphizations dispatched. |
| `build_fix_compact_resident_f64_on` | `fix_compact_kernel` dequant | u64 `h_raw` buffer consumed by `(bits as i64)/2^30` widen | ✓ WIRED | u64 alloc 2306-07; dequant 2003-04. |
| re-pinned parity test | CPU f64 anchor (`construct_histograms_cpu`, NOT a 2nd GPU launcher) | `max_rel(anchor, gpu_u64) <= 1e-7` | ✓ WIRED | kernel_parity.rs:1799/1843-54. |
| determinism assert | two resident u64 build runs | `to_bits` bit-equality | ✓ WIRED | kernel_parity.rs:1864-1876. |
| A/B example | live resident u64 build path | `construct_leaf_hist_resident_lds_kernel_u64::launch_unchecked` | ✓ WIRED | example:199. |

### Data-Flow Trace (Level 4)

| Artifact | Data Variable | Source | Produces Real Data | Status |
| -------- | ------------- | ------ | ------------------ | ------ |
| u64 build kernel | `out[base+m]` (histogram cells) | LDS sub-hist scattered from `resident_bins`/`ord_g`/`ord_h` via quantize+`fetch_add` | Yes (real per-bin integer accumulation, dequantized downstream) | ✓ FLOWING |
| fix_compact_kernel | `hist[wbi]` | `h_raw` (u64) dequantized `(bits as i64)/2^30` | Yes (real u64 RAW buffer from the build) | ✓ FLOWING |
| parity anchor | `anchor[..]` | `construct_histograms_cpu` over gathered leaf `(bin,grad,hess)` + host fix/compact | Yes (bit-exact CPU f64 fold) | ✓ FLOWING |

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
| -------- | ------- | ------ | ------ |
| CPU-only build (zero fixed-point codegen leak) | `cargo build -p lgbm-compute` | Finished, EXIT=0 | ✓ PASS |
| rocm-feature compile (u64 kernel + dequant + guard lower on cubecl-hip) | `cargo check -p lgbm-compute --features rocm` | Finished, EXIT=0 | ✓ PASS |
| A/B example CPU-only stub compiles | `cargo build -p lgbm-compute --example gpu_fixedpoint_resident_ab` | Finished, EXIT=0 | ✓ PASS |
| Resident parity PASS on GPU (1e-7 + determinism) | `cargo test -p oracle-harness --features rocm kernel_parity_resident_build_fix_compact...` | max_rel=0.000e0 (P=1 & P=10), determinism holds (2026-07-03) | ✓ PASS (hardware) |
| A/B not-slower sign on GPU (>=2 runs) | `cargo run --release --features rocm --example gpu_fixedpoint_resident_ab` | HEAVY wide NOT-SLOWER, SEP sign stable across 2 runs (2026-07-03) | ✓ PASS (hardware) |
| Unchanged f64 pins on GPU | `cargo test -p lgbm-compute --features rocm --test rocm_backend_parity --test rocm_row_partition` | 5/5 + 2/2 green (2026-07-03) | ✓ PASS (hardware) |

### Probe Execution

No conventional `scripts/*/tests/probe-*.sh` probes declared for this phase; verification is via cargo tests + the device-time example. (N/A)

### Requirements Coverage

Traceability is via SPEC.md (no REQUIREMENTS.md in this project). Plan frontmatter declares SPEC-1..SPEC-4; SPEC.md enumerates a 5-item "Scope" list + "Hard gates".

| Requirement | Source Plan | Description | Status | Evidence |
| ----------- | ----------- | ----------- | ------ | -------- |
| SPEC-1 | 11-01 | u64 two's-complement fixed-point build kernel + i64/u64 buffers (SPEC scope 1+2) | ✓ SATISFIED | u64 kernel + u64 RAW merge buffer + dequant (truths 1,2). |
| SPEC-2 | 11-02 | Oracle parity RE-PIN to CPU f64 anchor + determinism (SPEC "Hard gates") | ✓ SATISFIED (code) / human (runtime) | kernel_parity re-pinned, 1e-7 gate, determinism (truths 5,6). |
| SPEC-3 | 11-03 | Device-time A/B confirming not-slower in wide regime (SPEC "speed validated" gate) | ✓ SATISFIED (code+captured) / human (runtime) | A/B harness + captured 2-run numbers (truths 7,8). |
| SPEC-4 | 11-01 | Overflow guard for extreme leaves; Atomic<u64> not Atomic<i64> (SPEC scope 3+4) | ✓ SATISFIED | Typed-error overflow guard (truth 3); `Atomic<u64>` enforced (truth 1). |
| SPEC scope item 5 | (none) | OPTIONAL compose w/ spike-017 per-warp replication | n/a — explicitly OPTIONAL | Correctly out of scope; not claimed by any plan. Not a gap. |

All 4 declared SPEC IDs accounted for. No orphaned requirements (the only un-claimed SPEC.md item is the explicitly-optional item 5).

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
| ---- | ---- | ------- | -------- | ------ |
| histogram.rs | 1877-83 | `assert!(!fixed_point, ...)` panic on >256-bin fallback across a `pub` boundary | ℹ️ Info (review WR-01) | Unreachable for max_bin<=255 (LightGBM default); a `pub`-boundary panic where siblings return typed errors. Does not undermine any phase must-have. |
| histogram.rs | 1731, 2274-83 | unchecked `slot_off` subtraction / `gradients[leaf_rows[i]]` index on caller input | ℹ️ Info (review WR-02/04) | Latent panic on malformed caller input; not a goal truth. |
| histogram.rs | 622-630 | doc overstates effective fractional precision (f32 product caps ~24 bits, not 9 frac bits) | ℹ️ Info (review WR-03) | Doc-accuracy only; measured ~5.9e-9 / 1e-7 gate still hold. |

No debt markers (TBD/FIXME/XXX) introduced by phase-11 commits. No BLOCKER anti-patterns.

### Human Verification Required

The three device-runtime items (frontmatter `human_verification`) gate the goal's runtime claims:

1. **Resident parity gate on GPU** — run `cargo test -p oracle-harness --features rocm kernel_parity_resident_build_fix_compact_equals_host_on_hip`; expect PASS (max_rel <= 1e-7 vs CPU f64 anchor + bit-equal determinism). The code is correctly re-pinned (def-f8u-01 resolved); only the hardware PASS remains to observe.
2. **A/B not-slower sign on GPU (>=2 runs)** — run `cargo run --release --features rocm --example gpu_fixedpoint_resident_ab` at least twice; expect HEAVY-wide u64 ratio >= 1.0x at every P with a stable SEP sign. Judge the SIGN + methodology, not absolute Mr/s (spoofed 8-CU APU; all throughput APU-confounded per project memory).
3. **Unchanged-path regression on GPU** — run `rocm_row_partition` (2/2) + `rocm_backend_parity` (4/4 bit-exact); expect green at existing tolerances. (cuda_mirror DEF-11-OOS-01 flake is a documented pre-existing f32-atomic nondeterminism in a different kernel — not a phase-11 regression.)

### Gaps Summary

No BLOCKER gaps. Every phase must-have is realized in the codebase as a substantive, wired artifact: the live ROCm resident BUILD now accumulates in u64 two's-complement fixed-point via `Atomic<u64>` LDS atomics, dequantized at the fix-compact seam, with a typed overflow guard; the CPU f64 anchor and CPU routing are byte-untouched; the parity gate is correctly re-pinned to the CPU f64 anchor (resolving def-f8u-01) with a determinism sub-assert; and the device-time A/B harness drives the live u64 kernel against an explicitly-labelled f32 twin using the spike-018/019 discipline. CPU-only and rocm-feature builds both compile.

Status is **passed**. The three device-runtime must-haves — parity within ~1e-6, two-runs determinism, and not-slower-in-the-wide-regime — were confirmed on the local ROCm GPU on 2026-07-03 (see 11-UAT.md), so the goal's numerical/perf contract is now hardware-certified rather than SUMMARY-claimed.

**Review-warning WR-05 — CLOSED (2026-07-03):** The most goal-relevant warning was that the live-path correctness gate only exercised a P=1 leaf, leaving the multi-cube row-partition (P>1) merge unproven at the anchor. The hardware re-run exercised the `kernel_parity_resident_build_fix_compact_p_gt_1_equals_host_on_hip` cell — `row_partition_count(3,300000)=10`, and the P=10 multi-cube read-back matches the CPU f64 anchor bit-exactly (`max_rel=0.000e0`). The order-independence of integer add is now proven at P>1, not just argued. WR-05 no longer leaves a coverage gap.

---

_Verified: 2026-06-22_
_Verifier: Claude (gsd-verifier)_
