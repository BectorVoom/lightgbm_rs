# Discrete-CUDA bottleneck attribution + the metric-eval fix (Kaggle, real NVIDIA)

The 001–040 perf campaign ran entirely on a **spoofed 8-CU APU** (ROCm). Spikes
046/048/049 are the FIRST attribution on a **real discrete NVIDIA GPU** (Kaggle,
`device_type='cuda'`, cubecl-cuda), at the user's repro shape **500k×50, 100 trees,
num_leaves=31, binary**. Headline: lightgbm_rs CUDA was ~5–6× slower than official
LightGBM CUDA — and the dominant chunk was NOT what the APU campaign predicted.

## Requirements (honored)
- Backend stays compile-time switched; CPU f64 anchor is the bit-exact gate. Perf work
  must not touch parity (every change here is env-gated and parity-verified).
- "GPU is faster" is only claimed in the regime the data supports.
- Measure on real hardware before planning — the spoofed APU mis-predicts discrete-GPU
  cost (and even mis-predicted *which* phase dominates here).

## How to measure it (the reusable harness)
1. **The shipped Python path was a profiling black box.** `phase_prof::dump()` was wired
   ONLY into the Rust bench examples, never into `booster.rs`. Spike-046 added one
   env-gated, parity-neutral line at the end of `train_inner_columns_full`:
   ```rust
   lgbm_treelearner::phase_prof::dump("train");   // inert unless LGBM_PHASE_PROF=1
   ```
   Now `LGBMClassifier.fit()` emits the full BUDGET/LOOP/COUNTS/IN_LEARNER_OTHER
   breakdown to stderr. **Any Python-path GPU profiling REQUIRES this hook.**
2. **Run on Kaggle via the authenticated CLI** (user `boomvector`, kernel
   `boomvector/lgb-rs-cuda-bench`). The local "GPU" is a spoofed APU — Kaggle is the
   only real-CUDA path. Harness: `.planning/spikes/046-*/{kaggle_bench_instrumented,
   kaggle_metric_ab,bench_runner}.py` + `048-*/kaggle_confirm_fix.py`. The kernel
   git-clones master, builds `maturin build --release -F cuda`, runs under
   `LGBM_PHASE_PROF=1`. Absolute walls are NOT cross-session comparable (Kaggle assigns
   T4/P100/T4×2 — saw 17s/20s/11s across sessions); trust **in-session deltas**.
3. **CPU components are backend-independent** — attribute them LOCALLY (the
   `spike046_validate` example takes `LGBM_VALIDATE_ROWS/ITERS`); only true GPU-device
   costs need Kaggle. This saved a full Kaggle cycle in spike-049.

## The measured map (post-metric-fix, ≈11s real CUDA, 500k×50, 100 trees)
| Chunk | % wall | Lever |
|---|---|---|
| GPU histogram phases (build+scan+partition) | **53%** | architectural — on-device monolithic learner (official's `CUDASingleGPUTreeLearner`); the 09–13 campaign already tuned the kernels |
| Python "marshalling" (actually BINNING) | **25%** | ATTRIBUTED+FIXED (spike-050): single-threaded raw→bin (624ms@500k×50), NOT numpy marshalling (43ms). Feature-parallel rayon binning shipped — 6.5×, bit-exact |
| `in_learner_other` (per-tree scaffolding) | 15% | DEAD END — diffuse, no single lever |
| grad+score+snapshot+other | 7% | low |
| metric eval | 0% | FIXED (was 26%) |

## The shipped win: the metric-eval fix (quick-260628-f57)
- **Finding:** 26% of the CUDA wall (~4.5s/100 trees, identical on CPU+GPU ⇒ host cost)
  was per-iteration training-metric eval. Root cause `booster.rs:1291`:
  `provide_train = is_provide_training_metric || valid.is_none()` — forced training-metric
  eval whenever there's no eval_set, divergent from C++ (default false ⇒ empty
  `evals_result_`).
- **Fix:** drop `|| valid.is_none()` → `provide_train = config.is_provide_training_metric`.
  Confirmed on Kaggle: metric phase 4489ms→0ms; default == metric_freq=200 workaround
  (Δ 0.08s); gap 6.1×→4.7× in-session. Parity-neutral to the trees.
- **Test-contract caveat:** the divergence was test-encoded — 4 `lgbm` unit tests +
  `oracle-harness` `multiclass_cell_builder` trained without valid yet asserted training
  `eval_history`. Realign them to the explicit opt-in `.is_provide_training_metric(true)`
  (goldens unchanged — they already held per-round training metrics).

## What to avoid (refuted hypotheses — don't chase these)
- **Per-leaf scan sync-floor is NOT the discrete-CUDA bottleneck.** 2890 scan
  round-trips/train cost only 286ms (1.7%) even on real PCIe — spike-021/024 already
  paid it off. (My going-in hypothesis; wrong.)
- **"Route narrow shapes to CPU" is REFUTED on few-core boxes.** lgb_rs CUDA (17s) BEAT
  lgb_rs CPU (22.8s) on Kaggle's ~4-vCPU box — the 16-core dev CPU edge vanishes.
  Routing is environment-dependent, not universal.
- **Redundant per-tree bin re-upload is NOT a CUDA issue** — the `qxl` work gave
  CudaBackend the full resident surface; `resident_bin_upload≈32ms` once.
- **`resident_reset` is NOT a lever** — 0.308ms/100 trees (spike-049). And
  `in_learner_other` is ~92% diffuse `residual` on BOTH CPU and GPU (child leaf-splits
  seeding, arg_max, hist-pool mgmt, tree finalization); root_fold/partition_new/
  scratch/CegbModel all negligible. **Do not scope a phase around `in_learner_other`.**

## The Python-side 25% = single-threaded binning, not marshalling (spike-050, SHIPPED)
The Python `fit()` path is `dense_any_to_rows` (numpy→`Vec<Vec<f64>>`) → `RawCorpus` →
`lgbm::train_raw` → `build_feature_columns_from_raw_with_config` (raw→bin) → boosting loop.
Measured at 500k×50 (host/CPU ⇒ local, no Kaggle):
- numpy→`Vec<Vec<f64>>` marshalling = **43ms** — a NON-ISSUE, do not chase it.
- raw→bin **serial = 624ms** — the real cost (was hidden as `binning=0`: the `BINNING_NS`
  wrap only covered the DenseCorpus path, not `train_raw`; spike-050 wrapped it).
- **FIX (shipped, bit-exact): feature-parallel binning** — the per-feature BinMapper
  build + assignment are independent (C++ bins OpenMP-over-features), so:
  ```rust
  let bin_feature = |j: usize| -> FeatureColumn { /* build BinMapper(col j) + assign */ };
  let columns = if par { (0..num_features).into_par_iter().map(bin_feature).collect() }
                else   { (0..num_features).map(bin_feature).collect() };
  ```
  Order-preserving + fixed `data_random_seed` per feature ⇒ **bit-exact** (proved by
  `raw_bin_train_matches_cpp_golden`). **6.5× (624→96ms @16 cores)**; `LGBM_PAR_BIN=0`
  serial gate. Lesson: any per-feature host loop (binning, column transforms) should be
  rayon-parallel — match C++'s OpenMP-over-features.

## Constraints
- Kaggle CLI auth = ACCESS_TOKEN at `/home/user/.kaggle` (no kaggle.json file).
- Kernel output download pulls the whole committed tree (slow, 2+ min); the log is
  `lgb-rs-cuda-bench.log` (JSON-stream); `rm -rf` the cloned `lightgbm_rs/` subdir after.
- A code change must be pushed to GitHub `master` before the kernel clones it.

## Origin
Synthesized from spikes: 046, 048, 049, 050 (047 skipped — Kaggle gave real numbers
directly). Shipped fixes: quick-260628-f57 (metric eval), spike-050 (parallel binning).
Sources in: sources/046-python-path-phase-prof/, sources/048-kaggle-cuda-confirm/,
sources/049-in-learner-other-attribution/, sources/050-python-marshalling-binning/.
