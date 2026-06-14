# Phase 09 (standalone) — SUMMARY

**Shipped:** row-partitioned GPU histogram build. Branch `perf/09-gpu-rowpart-hist`
(commits 27dd0cf, 6970023, 902420c). Not yet merged to master.

## What landed

- **Lever 1 (row-partition) — SHIPPED.** `construct_leaf_hist_resident_lds_kernel` and
  `construct_leaf_hist_batched_lds_kernel` now take a `P` row dimension
  (`CubeCount = (num_features, P)`). `row_partition_count()` targets ~8 workgroups/CU
  (`clamp(768/num_features, 1, 16)`), gated to leaves ≥ `ROWPART_MIN_LEAF` (256k;
  `LGBM_ROWPART_MIN` override). **P=1 below the gate is byte-identical** to the prior
  kernel. In-tree harness reproduces the spike: **P=16 ≈ 1.25× over P=1** (1244–1267ms
  vs ~1580ms), P=16 the winner in every round.
- **Lever 2 (register-batching K=4) — NULL, not shipped.** `K4/K1 = 0.89–0.98×` at P=16.
  At saturating occupancy the bottleneck is LDS atomic contention, not load latency, so
  the K=4 extra registers slightly cut occupancy. K stays 1; the K4 kernel is retained in
  `gpu_row_partition.rs` as evidence.

## Verification

- `row_partition_count_heuristic` unit test — green (CPU, no GPU).
- `rocm_row_partition.rs` (new) — green: batched LDS build vs cpu f64 anchor, P=1 `rel=1.7e-6`
  (<1e-5 gate), P=16 `rel=2.0e-7` (<5e-5 gate). **P>1 is closer to the anchor than P=1** (tree
  summation). CPU f64 anchor + bit-exact merge gate untouched (CPU tests green).
- `rocm_parallel_histogram.rs` (single-feature LDS) — 7/7 green, unchanged.
- 04-ROCM-GAPS.md — G-09-01 residual recorded.

## Caveats / follow-ups

- **Pre-existing rocm-test bit-rot** (NOT from this work, flagged in 04-ROCM-GAPS.md):
  `rocm_backend_parity.rs` (`RocmBackend` gained fields) and `kernel_parity.rs:1548`
  (`build_leaf_histograms_raw` → `&[&BinColumn]`) no longer compile under `--features rocm`.
  They block the full rocm `kernel_parity` suite; phase-09 verified via the new test instead.
  Worth a separate cleanup pass.
- **ROI:** GPU is ROCm-parity-track — CPU (multi-threaded, spike-005) still wins at every
  tested size. This is a parity-path improvement, not overall-fastest.
- Not merged to master; no ROADMAP entry (standalone per user).

## Out of scope (seeds remain)

16-bit discretized histogram (`.planning/seeds/16bit-discretized-histogram.md`),
multi-feature-per-cube packing.
