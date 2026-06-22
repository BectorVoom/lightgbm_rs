---
status: testing
phase: 11-gpu-fixedpoint-int-atomics
source: [11-VERIFICATION.md]
started: 2026-06-22T00:00:00Z
updated: 2026-06-22T00:00:00Z
---

## Current Test

number: 1
name: Run the re-pinned resident parity gate on the ROCm GPU
expected: |
  `cargo test -p oracle-harness --features rocm kernel_parity_resident_build_fix_compact_equals_host_on_hip`
  PASSES: max_rel vs the CPU f64 anchor (construct_histograms_cpu) <= FIXEDPOINT_REL_GATE (1e-7),
  and the 2-runs to_bits() determinism sub-assert holds.
awaiting: user response

## Tests

### 1. Run the re-pinned resident parity gate on the ROCm GPU
expected: kernel_parity_resident_build_fix_compact_equals_host_on_hip PASSES — max_rel vs CPU f64 anchor <= FIXEDPOINT_REL_GATE (1e-7), and the 2-runs to_bits() determinism sub-assert holds. (Code is correctly re-pinned to the CPU f64 anchor, not GPU-vs-GPU — def-f8u-01 resolved; only the hardware PASS remains to observe.) Recommended: also force a P>1 leaf through the live u64 resident chain to close review WR-05 (multi-cube row-partition merge is currently only anchor-checked at P=1).
result: [pending]

### 2. Run the device-time A/B example >=2x on the ROCm GPU
expected: `cargo run --release --features rocm --example gpu_fixedpoint_resident_ab` prints, in the HEAVY wide 16x1M regime, u64 median ratio >= 1.0x at every P (SEP-WIN at >=1 P), sign-stable across two process runs; LIGHT regime overlap acceptable. Judge the SIGN + methodology, not absolute Mr/s (spoofed 8-CU APU — all throughput APU-confounded per project memory).
result: [pending]

### 3. Confirm the unchanged non-resident f32 + bit-exact f64 pins stay green on the GPU
expected: rocm_row_partition (2/2) and rocm_backend_parity (4/4 bit-exact) PASS at their existing tolerances after the phase-11 changes — the integer build must NOT disturb the post-dequant f64 paths. (The cuda_mirror DEF-11-OOS-01 flake is a documented pre-existing f32-atomic nondeterminism in a DIFFERENT kernel, not a phase-11 regression.)
result: [pending]

## Summary

total: 3
passed: 0
issues: 0
pending: 3
skipped: 0
blocked: 0

## Gaps
