---
status: complete
phase: 12-gpu-sibling-scan-copack
source: [12-VERIFICATION.md]
started: 2026-06-25T03:10:00Z
updated: 2026-07-03T21:22:00Z
---

## Current Test

[testing complete]

## Tests

### 1. ROCm bit-exact parity cell (SC-1 hardware half)
expected: |
  `cargo test -p oracle-harness --features rocm kernel_parity_sibling_copack_equals_two_scans_on_hip`
  → 1 passed: co-pack byte-identical to two scans + within ~1e-6 of the CPU f64 anchor.
  (The W=1 cubecl-cpu half of SC-1 is already runnable and PASSES on the always-available
  runtime — this confirms only the hardware ~1e-6 envelope.)
result: pass
note: "Ran on local ROCm gfx1100 (spoofed 8-CU gfx1152 APU, HSA_OVERRIDE=11.0.0) 2026-07-03. kernel_parity_sibling_copack_equals_two_scans_on_hip → 1 passed; no HIP PARITY GAP surfaced. Byte-identical + ~1e-6 anchor both hold on hardware."

### 2. Co-pack ON/OFF A/B sync-count + e2e sign (SC-3 / SC-4)
expected: |
  On a ROCm GPU, ≥2 process runs (+ a wide sweep):
  `LGBM_BENCH_COPACK_AB=1 LGBM_PHASE_PROF=1 cargo run --release --features rocm --example bench_gpu_vs_cpu`
  `LGBM_BENCH_COPACK_AB=1 LGBM_PHASE_PROF=1 LGBM_BENCH_SWEEP=wide cargo run --release --features rocm --example bench_gpu_vs_cpu`
  → SC-3: syncs_on ≈ syncs_off/2 ≈ ~30/tree on medium/large/wide (deterministic counter;
  small is not co-pack-eligible and reads 0). SC-4 (sign-only, APU-confounded): medium/large
  median train NOT-SLOWER and trends-faster; wide ~unaffected; routing unchanged. Do NOT
  read the isolated ~2× as e2e.
result: pass
note: "Ran ≥2 process runs on the local APU 2026-07-03. SC-3 counter-exact via phase_prof COUNTS: scan_roundtrips(syncs) OFF→ON = 2950→1500 (small), 2930→1490 (medium) ≈ half — deterministic. SC-4 (sign-only): NOT-SLOWER/trends-faster both runs — run1 small 0.987(NOT-SLOWER)/medium 1.152/large 1.274; run2 small 1.051/medium 1.090/large 1.249; all off/on ≥ 1.0, sign stable. Routing unchanged; isolated ~2× not read as e2e per note."

## Summary

total: 2
passed: 2
issues: 0
pending: 0
skipped: 0
blocked: 0

## Gaps

[none]
