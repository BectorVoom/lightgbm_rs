# Speed Report — lightgbm_rs optimization pass (260608-iwj)

**Instrument:** `crates/lgbm/examples/bench_train.rs` — deterministic synthetic
identity-binned corpora, regression GBDT (100 iters, 31 leaves), median wall-clock
over 5 train reps / 7 predict reps. Same binary logic across all runs.

**Machine:** local (Linux 6.17). **Measurement convention:** lower train_median is
better; numbers are medians so transient noise is suppressed.

## Per-lever measurement runs

Each run adds one lever on top of the previous (cumulative):

| Run | Levers active | small train | medium train | large train | large predict |
|-----|---------------|-------------|--------------|-------------|---------------|
| **M0** | baseline (default release, system alloc) | 1.71s | 4.75s | 8.93s | 62.75ms |
| **M1** | + `[profile.release]` lto=fat, cgu=1 | 1.68s | 4.58s | 8.68s | 62.25ms |
| **M2** | + mimalloc global allocator | 1.61s | 4.35s | 8.25s | 61.50ms |
| **M3** | + gather buffer-reuse / candidate pre-size | 1.55s | 4.21s | 8.12s | 58.25ms |

### Cumulative speedup (M0 → M3, all parity-safe levers)

| size | train M0 | train M3 | speedup | predict M0 | predict M3 |
|------|----------|----------|---------|------------|------------|
| small  | 1.71s | 1.55s | **−9.4%** | 3.00ms | 2.89ms |
| medium | 4.75s | 4.21s | **−11.4%** | 24.53ms | 23.34ms |
| large  | 8.93s | 8.12s | **−9.1%** | 62.75ms | 58.25ms |

**Parity gate after T4:** `cargo test -p oracle-harness` GREEN — boosting_parity 75,
learner_parity 29, kernel_parity 4, predict_parity 5, raw_bin_train_parity 2,
rng_parity 1, all bit-exact; core unit tests (lgbm 41, boosting 55, compute 18,
treelearner 64) GREEN. Zero numeric change.

## C++ reference point

_(pip `lightgbm==4.6`, same synthetic data — filled in at T5.)_

## Notes

- Half-precision (f16/bf16) GPU kernels were excluded by design: ~1e-3 precision
  would break the bit-exact CPU gate and the ~1e-6 ROCm gate (CLAUDE.md core value).
- bytemuck zero-copy is already used at the compute boundary, so the "zero-copy
  ingest" lever was realised as buffer-reuse / redundant-alloc elimination.
