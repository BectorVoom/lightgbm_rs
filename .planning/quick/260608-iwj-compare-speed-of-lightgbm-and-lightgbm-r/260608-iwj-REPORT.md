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
| **M3** | + smallvec / buffer-reuse | _pending_ | | | |

_(Filled in as each task lands.)_

## C++ reference point

_(pip `lightgbm==4.6`, same synthetic data — filled in at T5.)_

## Notes

- Half-precision (f16/bf16) GPU kernels were excluded by design: ~1e-3 precision
  would break the bit-exact CPU gate and the ~1e-6 ROCm gate (CLAUDE.md core value).
- bytemuck zero-copy is already used at the compute boundary, so the "zero-copy
  ingest" lever was realised as buffer-reuse / redundant-alloc elimination.
