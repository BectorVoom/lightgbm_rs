---
quick_id: 260621-p9v
title: Hoist resident-bin device upload to once-per-train
status: complete
date: 2026-06-21
---

# Quick Task 260621-p9v — Summary

## What changed

Made the GPU resident-bin device upload run **once per train** instead of **once per
tree** (the spike-014b lever). The binned feature columns are immutable for the whole
train and `RocmBackend` is one instance per `train()` (its `resident_bins` cache survives
across trees — `reset_resident_pool` clears only the histogram-slot mirror, never the bin
cache), so re-uploading every tree was pure waste — measured ~31% of train wall-clock at
1M×500.

**Implementation** (`crates/lgbm-treelearner/src/learner.rs`, one small change):
- New `SerialTreeLearner` field `resident_bins_uploaded: bool` (init `false` in `new()`).
- The `wants_resident_bins()` upload block in `train_inner` is now guarded by
  `&& !self.resident_bins_uploaded`, and sets the flag `true` after `upload_resident_bins`.
- `with_features` resets the flag to `false` (a new feature set ⇒ stale device bins ⇒
  must re-upload), preserving correctness if the learner's feature set is swapped.

No backend/API change; the `phase_prof::UPLOAD_NS` timer (from spike-014b) now naturally
records the single first-tree upload.

## Verification

**Parity (bit-exact merge gate — the hard requirement):** all green.
- `cargo test -p lgbm-treelearner --lib` → 76/76
- `cargo test -p lgbm-boosting --lib` → 55/55
- `cargo test -p oracle-harness` → all suites pass (kernel histogram bit-exact, boosting
  75/75, learner parity, advanced/rng). The change reuses the *same device bytes* across
  trees, so GPU results are identical to before; the CPU f64 anchor is untouched.

**GPU speedup** (`bench_gpu_vs_cpu` `LGBM_BENCH_SWEEP=wide`, gfx1100, iters=4, median/3):

| shape | before | after | speedup | resident_bin_upload/rep (before → after) |
|-------|--------|-------|---------|------------------------------------------|
| 250k×500 | 8.55s | 6.60s | −23% | 2.35s → 0.59s |
| 500k×500 | 13.18s | 10.82s | −18% | — → 0.91s |
| 1M×500 | 29.55s | **20.03s** | **−32%** | 9.29s → 2.15s |

At 1M×500 rows/s rose 33.7k → 49.9k (**+48%**). The upload bucket dropped to exactly
~¼ of before (4 trees → 1 upload), confirming the once-per-train guard. The remaining
~2.1s is the single legitimate upload. The win **scales with rows** (redundant upload is
a bigger share at higher row counts) and, since the one upload amortizes over more trees,
**grows further at production iteration counts** (this is the conservative iters=4 figure).

## Follow-on (not done here)

- The single upload still widens narrow bins to u32 (2× 2GB host alloc at 1M×500) — a
  future lever could upload the native u8/u16 bins directly (cf. spike-006's `Array<u8>`
  on HIP) to shrink even the one-time upload.
- The ~17% boosting-loop overhead (per-iter `to_vec` clones of the 1M score buffer,
  spike-014b) remains an open, separate lever.
