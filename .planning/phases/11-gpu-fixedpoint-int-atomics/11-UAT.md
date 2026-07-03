---
status: complete
phase: 11-gpu-fixedpoint-int-atomics
source: [11-VERIFICATION.md]
started: 2026-06-22T00:00:00Z
updated: 2026-07-03T21:20:00Z
---

## Current Test

[testing complete]

## Tests

### 1. Run the re-pinned resident parity gate on the ROCm GPU
expected: kernel_parity_resident_build_fix_compact_equals_host_on_hip PASSES — max_rel vs CPU f64 anchor <= FIXEDPOINT_REL_GATE (1e-7), and the 2-runs to_bits() determinism sub-assert holds. (Code is correctly re-pinned to the CPU f64 anchor, not GPU-vs-GPU — def-f8u-01 resolved; only the hardware PASS remains to observe.) Recommended: also force a P>1 leaf through the live u64 resident chain to close review WR-05 (multi-cube row-partition merge is currently only anchor-checked at P=1).
result: pass
note: "Ran on local ROCm gfx1100 (spoofed 8-CU gfx1152 APU, HSA_OVERRIDE=11.0.0) 2026-07-03. Both P=1 and P>1 cells PASS: max_rel_vs_cpu_f64_anchor=0.000e0 (bit-exact, gate=1e-7); P>1 row_partition_count(3,300000)=10 multi-cube merge also 0.000e0 — closes review WR-05. Determinism sub-assert holds."

### 2. Run the device-time A/B example >=2x on the ROCm GPU
expected: `cargo run --release --features rocm --example gpu_fixedpoint_resident_ab` prints, in the HEAVY wide 16x1M regime, u64 median ratio >= 1.0x at every P (SEP-WIN at >=1 P), sign-stable across two process runs; LIGHT regime overlap acceptable. Judge the SIGN + methodology, not absolute Mr/s (spoofed 8-CU APU — all throughput APU-confounded per project memory).
result: pass
note: "Ran ≥2 process runs on the local APU 2026-07-03. HEAVY wide 16×1M NOT-SLOWER both runs — run1 SEP-WIN at P=1 (1.64×) & P=8 (1.34×); run2 SEP-WIN every P (u64 ratios 1.06×–1.41×). LIGHT not-regressed. SEP sign stable across processes. Absolute Mr/s APU-confounded and disregarded per methodology; the relative f32/u64 sign is load-bearing and positive."

### 3. Confirm the unchanged non-resident f32 + bit-exact f64 pins stay green on the GPU
expected: rocm_row_partition (2/2) and rocm_backend_parity (4/4 bit-exact) PASS at their existing tolerances after the phase-11 changes — the integer build must NOT disturb the post-dequant f64 paths. (The cuda_mirror DEF-11-OOS-01 flake is a documented pre-existing f32-atomic nondeterminism in a DIFFERENT kernel, not a phase-11 regression.)
result: pass
note: "Ran on the local APU 2026-07-03. rocm_backend_parity 5/5 green (construct_histograms/subtract/find_best_split bit-exact + data_partition + default_left_tie); rocm_row_partition 2/2 green (p1_and_p_gt_1 anchor + naive_batched_fallback). The integer build did not disturb the post-dequant f64 paths."

## Summary

total: 3
passed: 3
issues: 0
pending: 0
skipped: 0
blocked: 0

## Gaps

[none]
