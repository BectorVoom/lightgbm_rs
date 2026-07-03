---
status: complete
phase: quick-260622-t4u
plan: 01
subsystem: gpu-histogram-parity
tags: [rocm, fixedpoint, row-partition, parity, WR-05, phase-11]
requires:
  - "build_fix_compact_resident_readback_f64_on (live u64 fixed-point resident build, Phase 11-01)"
  - "construct_histograms_cpu (CPU f64 anchor fold)"
provides:
  - "kernel_parity_resident_build_fix_compact_p_gt_1_equals_host_on_hip (committed P>1 multi-cube parity gate)"
  - "row_partition_count exported pub (rocm-gated) for P>1 assertion"
affects:
  - "crates/oracle-harness/tests/kernel_parity.rs"
  - "crates/lgbm-compute/src/kernels/histogram.rs"
tech-stack:
  added: []
  patterns:
    - "approach B (large real leaf) to force P>1 with zero env/global-state mutation"
    - "P>1 guard asserts row_partition_count>1 so a silent drop to P=1 fails loudly"
key-files:
  created: []
  modified:
    - "crates/lgbm-compute/src/kernels/histogram.rs (1-line pub widening, rocm-gated)"
    - "crates/oracle-harness/tests/kernel_parity.rs (+166 lines, 1 new test)"
decisions:
  - "Force P>1 via a 300k-row real leaf (approach B), NOT env mutation — the OnceLock-cached LGBM_ROWPART_TARGET_CUBES makes set_var unsafe under cargo test"
metrics:
  duration: ~6 min
  completed: 2026-06-22
---

# Quick Task 260622-t4u: Permanent P>1 Row-Partitioned Multi-Cube Resident Build Parity Test Summary

Closed WR-05 (Phase 11): added a committed `#[cfg(feature="rocm")]` parity test that exercises the live u64 fixed-point resident GPU histogram BUILD at P>1 (the multi-cube row-partitioned additive merge), pinning the read-back to the bit-exact CPU f64 anchor — the prior live-path gate only ran at P=1.

## What Was Done

**Task 1 — Export `row_partition_count` (commit 5a2b3fa):** Single-line visibility widening of `row_partition_count` from `fn` to `pub fn` in `crates/lgbm-compute/src/kernels/histogram.rs:739`. It stays `#[cfg(feature="rocm")]`-gated, so no symbol leaks into the default CPU-only build. Body, signature, and cfg untouched. This lets the new test assert the P>1 regime was genuinely reached.

**Task 2 — Add the P>1 parity test (commit f5bf726):** New test `kernel_parity_resident_build_fix_compact_p_gt_1_equals_host_on_hip` in `crates/oracle-harness/tests/kernel_parity.rs`, placed immediately after the existing P=1 test (now at line ~1879). Mirrors the P=1 test's structure exactly (CPU f64 anchor via `construct_histograms_cpu` + host `fix_histogram` + `host_compact_histogram`, the `build_fix_compact_resident_readback_f64_on` GPU arm, the `FIXEDPOINT_REL_GATE = 1e-7` gate, and the two-run `to_bits()` determinism block). Differences: 300k-row leaf (all rows in the leaf) over the same 3-feature spine (bins [4,3,5], mfb [2,0,1], offset [0,1,0]) to force the DEFAULT heuristic to P>1 with no env mutation; deterministic `row % num_bin` bins and small bounded g/h; and a `row_partition_count > 1` guard that fails loudly before the GPU build if a refactor silently drops to P=1.

## Verification (actual captured output — the real gate)

**(a) CPU-only `cargo build -p oracle-harness`:** `Finished dev profile ... in 0.07s` — clean.

**(b) `cargo build -p lgbm-compute --features rocm`:** `Finished dev profile ... in 0.84s` — clean (the pub widening lowers under rocm with no error).

**(c) NEW test on the GPU** (`cargo test -p oracle-harness --features rocm --test kernel_parity kernel_parity_resident_build_fix_compact_p_gt_1_equals_host_on_hip -- --nocapture --test-threads=1`):

```
running 1 test
test hip::kernel_parity_resident_build_fix_compact_p_gt_1_equals_host_on_hip ... resident u64 P>1 build: row_partition_count(3, 300000) = 10 (P>1 multi-cube merge)
resident u64 P>1 fixed-point build: P=10, max_rel_vs_cpu_f64_anchor=0.000e0 (gate=1e-7, spike-018 ref ~5.9e-9)
ok

test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 15 filtered out; finished in 0.82s
```

- **P>1 guard:** `row_partition_count(3, 300000) = 10` — genuinely multi-cube (10 cubes; the queried-CU target after clamp). PASS.
- **Parity:** `max_rel_vs_cpu_f64_anchor = 0.000e0` — bit-exact vs the CPU f64 anchor (well under the 1e-7 gate; integer-additive merge is order-independent). PASS.
- **Determinism:** two-run `to_bits()` assert held (no failure). PASS.

**(d) Existing P=1 test, no regression** (`...kernel_parity_resident_build_fix_compact_equals_host_on_hip...`):

```
test hip::kernel_parity_resident_build_fix_compact_equals_host_on_hip ... resident u64 fixed-point build: max_rel_vs_cpu_f64_anchor=0.000e0 (gate=1e-7, spike-018 ref ~5.9e-9)
ok
test result: ok. 1 passed; 0 failed; ...
```

The existing P=1 gate still passes at `max_rel = 0.000e0`. No regression.

## Notes

- Observed P = 10, not the 16 the plan's worst-case sizing predicted. The plan assumed the fallback target of 64 (→ clamp(64/3,1,16)=16); the live box's queried-CU-derived target instead yields 10 = clamp(target/3,1,16). Either way P>1 holds and the guard passes — the multi-cube merge is genuinely exercised. No action needed; the test asserts `P>1`, not a specific P.
- 300k rows ran fine on the APU (no OOM, ~0.8s); the plan's fallback to a smaller row count was not needed.

## Deviations from Plan

None - plan executed exactly as written. (The observed P=10 vs predicted P=16 is an environment fact, not a deviation; the test gates on P>1.)

## Self-Check: PASSED

- `crates/lgbm-compute/src/kernels/histogram.rs` — FOUND (pub row_partition_count)
- `crates/oracle-harness/tests/kernel_parity.rs` — FOUND (new test fn present)
- Commit 5a2b3fa — FOUND
- Commit f5bf726 — FOUND
- New test PASSES on GPU (P=10, max_rel=0.000e0); existing P=1 test PASSES (no regression)
- CPU f64 anchor byte-untouched; no tolerance loosened; reference trees not git-added
