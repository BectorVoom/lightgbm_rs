---
quick_id: 260608-lad
slug: abstract-backend-parallel-findsplit-batched-hist
date: 2026-06-08
mode: quick (3-part GPU rearchitecture; gated)
---

# Quick Task 260608-lad — backend abstraction + parallel find_best_split + batched histogram

Three coupled parts. Each behind the bit-exact CPU gate (default build byte-identical).

## Part 1 — backend abstraction (DONE)

Added `Backend::build_leaf_histograms_raw` (batched per-leaf, default = the per-
feature gather+construct loop → bit-exact CPU). Rewired the learner to call it once
per leaf, then host-side FixHistogram+compact. This is the seam Part 3 fills on GPU.
Gate GREEN.

## Part 2 — parallel prefix-sum find_best_split (GPU)

Replace the single-unit `find_best_split` on RocmBackend with a parallel kernel:
per-bin prefix-sum of (grad, hess, count) → per-bin gain + validity in parallel →
masked argmax reduction. Keep the C++ gate ORDER / eps / threshold semantics so it
matches the f64 anchor within ~1e-6. Validate (parity + speed) on gfx1100; CPU
untouched.

## Part 3 — device-resident bins + batched per-leaf launch (GPU)

Override `build_leaf_histograms_raw` on RocmBackend: keep the binned dataset
device-resident (upload once, reuse across leaves/iterations) and build ALL features'
histograms for a leaf in ONE kernel launch (collapse per-feature launches → 1/leaf).
~1e-6 gate; measure end-to-end GPU speedup.

## Gates

Default (CPU) build bit-exact GREEN after every part. `--features rocm`: ~1e-6 vs the
CPU anchor on real hardware; each part faster than before.
