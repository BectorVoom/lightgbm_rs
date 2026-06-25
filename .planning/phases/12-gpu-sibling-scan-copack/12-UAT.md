---
status: testing
phase: 12-gpu-sibling-scan-copack
source: [12-VERIFICATION.md]
started: 2026-06-25T03:10:00Z
updated: 2026-06-25T03:10:00Z
---

## Current Test

number: 1
name: ROCm bit-exact parity cell (SC-1 hardware half)
expected: |
  On a ROCm GPU:
  `cargo test -p oracle-harness --features rocm kernel_parity_sibling_copack_equals_two_scans_on_hip`
  → 1 passed — the co-packed sibling scan is byte-identical to two single-slot
  scans (assert_eq! per SplitInfo, both siblings, all 3 fixture features) AND each
  co-pack field is within ~1e-6 of the CPU f64 anchor (find_best_split_cpu_native,
  per def-f8u-01). No HIP PARITY GAP surfaced.
awaiting: user response

## Tests

### 1. ROCm bit-exact parity cell (SC-1 hardware half)
expected: |
  `cargo test -p oracle-harness --features rocm kernel_parity_sibling_copack_equals_two_scans_on_hip`
  → 1 passed: co-pack byte-identical to two scans + within ~1e-6 of the CPU f64 anchor.
  (The W=1 cubecl-cpu half of SC-1 is already runnable and PASSES on the always-available
  runtime — this confirms only the hardware ~1e-6 envelope.)
result: [pending]

### 2. Co-pack ON/OFF A/B sync-count + e2e sign (SC-3 / SC-4)
expected: |
  On a ROCm GPU, ≥2 process runs (+ a wide sweep):
  `LGBM_BENCH_COPACK_AB=1 LGBM_PHASE_PROF=1 cargo run --release --features rocm --example bench_gpu_vs_cpu`
  `LGBM_BENCH_COPACK_AB=1 LGBM_PHASE_PROF=1 LGBM_BENCH_SWEEP=wide cargo run --release --features rocm --example bench_gpu_vs_cpu`
  → SC-3: syncs_on ≈ syncs_off/2 ≈ ~30/tree on medium/large/wide (deterministic counter;
  small is not co-pack-eligible and reads 0). SC-4 (sign-only, APU-confounded): medium/large
  median train NOT-SLOWER and trends-faster; wide ~unaffected; routing unchanged. Do NOT
  read the isolated ~2× as e2e.
result: [pending]

## Summary

total: 2
passed: 0
issues: 0
pending: 2
skipped: 0
blocked: 0

## Gaps
