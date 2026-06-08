---
quick_id: 260608-jyl
slug: r2-native-cpu-backend
date: 2026-06-08
mode: quick (parity-gated; R2 follow-up)
---

# Quick Task 260608-jyl — R2: native CPU backend (kill cubecl-cpu launch overhead)

## Finding (probe-proven)

cubecl-cpu launch is a FIXED ~20–50µs/call wrapping a `CubeDim::new_1d(1)`
sequential loop. Native f64 fold is 5–210×/call faster and **bit-identical**
(`probe_hist`: bit_exact=true at R=300/2000/20000). ~305k construct + ~305k
find_best_split launches ≈ the whole ~8s.

## Decision (user)

Native CPU backend: replace the single-unit cubecl-cpu kernels with native Rust
f64 loops INSIDE `lgbm-compute`. cubecl-cpu stays wired + tested for the
ROCm-mirror parity; production CPU path goes native. Order: construct_histograms →
find_best_split → subtract → partition. Each its own commit + bit-exact gate.

## Tasks (atomic; bit-exact gate after each)

- **T1 — native `construct_histograms`.** `construct_histograms_cpu_native` (same V5
  validation, plain f64 fold); route `CpuBackend::construct_histograms` to it. Keep
  `construct_histograms_cpu` (cubecl) for tests/ROCm-mirror. Gate + measure (M5a).
- **T2 — native `find_best_split`.** Mirror `FindBestThresholdSequentially` in plain
  Rust, preserving EXACT operand orders (e.g. `right_g = sum_g - sum_left_g`, eps
  placements, finalization gates — see split.rs kernel doc). Gate + measure (M5b).
  HIGHEST RISK: any ULP drift fails the bit-exact gate → revert/iterate.
- **T3 — native `subtract` + `data_partition`.** Plain element-wise subtract; stable
  left/right reorder. Gate + measure (M5c).
- **T4 — report + summary + state.** Update REPORT with the R2 before/after; SUMMARY;
  STATE; memory. Remove the throwaway `probe_hist.rs`.

## Parity gate (non-negotiable)

`cargo test -p oracle-harness` GREEN (bit-exact) after EACH task — especially
kernel_parity, learner_parity, boosting_parity. Plus core unit suites.
Containment: changes stay inside `lgbm-compute` (CMP-01); ROCm path untouched.
