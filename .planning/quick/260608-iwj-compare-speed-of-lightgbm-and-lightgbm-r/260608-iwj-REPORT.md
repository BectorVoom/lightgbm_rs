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

## C++ reference point — lightgbm 4.6 vs lightgbm_rs

`lightgbm==4.6` pip wheel, **num_threads=1** (matched to the Rust serial learner),
same synthetic data + config (`.planning/quick/.../bench_cpp_ref.py`):

| size | C++ 4.6 train | Rust (M3) train | **Rust / C++** |
|------|---------------|-----------------|----------------|
| small  | 19.9ms  | 1.55s | **~78× slower** |
| medium | 75.9ms  | 4.21s | **~55× slower** |
| large  | 206.6ms | 8.12s | **~39× slower** |

**This is the headline finding.** lightgbm_rs is correctness-first (bit-exact to
the C++ reference) but currently **1.5–2 orders of magnitude slower** than the C++
engine. The parity-safe levers applied here (~9–11%) are real but tiny against that
gap — closing it is an algorithmic/architectural effort, not allocator/flag tuning.

### Where the gap comes from (root-cause analysis)

1. **Always-on D-06 snapshot overhead.** `train()` → `train_with_snapshots(..).0`
   always computes the per-split D-06 golden snapshot (`per_bin_gains`, a full
   host re-scan of every feature's fixed histogram) and then discards it. This is
   pure overhead in production training — only golden-capture needs it.
2. **Per-call CubeCL-CPU dispatch.** `build_leaf_histogram_into` calls
   `backend.construct_histograms` once **per feature per leaf**, each call doing a
   host gather → device buffer create → kernel launch → readback. The CubeCL CPU
   runtime's per-launch overhead dominates; C++ runs a tight OpenMP loop straight
   over contiguous columnar bins with the histogram-subtraction trick.
3. **Non-columnar gather.** `DenseCorpus` is row-major `Vec<Vec<f64>>`; every
   per-feature histogram re-gathers scattered rows (cache-unfriendly) vs C++'s
   pre-binned columnar `Bin` storage scanned in place.

### Recommended optimization roadmap (out of scope here — needs parity care)

- **R1 (largest, low risk): make the D-06 snapshot opt-in.** A `train()` that skips
  `per_bin_gains` when snapshots aren't requested removes a whole per-feature
  per-leaf host re-scan. Behavior-preserving for the model; verify bit-exact.
- **R2 (large): batch/amortize the histogram backend.** Construct all features'
  histograms for a leaf in one dispatch (or keep bin data resident on-device)
  instead of one create/launch/readback per feature. Biggest structural win.
- **R3 (medium): columnar bin storage + subtraction trick on the CPU anchor**, so
  the larger child reuses parent−sibling instead of re-gathering.
- **R4: rayon over features** for histogram construction (C++ parallelizes here);
  must preserve the ordered f64 fold for the bit-exact gate (per-feature is
  independent, so cross-feature parallelism is safe).

These are phase-sized efforts (each needs its own plan + parity gate), not a quick
task — flagged here so the speed gap is visible and actionable.

## Notes

- Half-precision (f16/bf16) GPU kernels were excluded by design: ~1e-3 precision
  would break the bit-exact CPU gate and the ~1e-6 ROCm gate (CLAUDE.md core value).
- bytemuck zero-copy is already used at the compute boundary, so the "zero-copy
  ingest" lever was realised as buffer-reuse / redundant-alloc elimination.
