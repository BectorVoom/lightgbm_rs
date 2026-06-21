# GPU Wide-Shape Train Attribution & Host-Setup Levers

The train-speed campaign at the **wide+tall** shape (1M rows × 500 features) on the ROCm
GPU path. Spike-014 (014a coarse + 014b budget) profiled where GPU train wall-clock
*actually* goes — overturning the assumption that the histogram kernel is the
bottleneck — and four shipped follow-on levers (`p9v`, `qix`, `rdu`, `rsh`) took 1M×500
iters=4 train from **29.55 s → ~9.5 s (−68%)**, every step bit-exact.

## Requirements

- Backend stays **compile-time switched** (`--features rocm`); the CPU f64 anchor is the
  default and the **bit-exact merge gate**. Speed work must not touch parity.
- Every lever is gated bit-exact (`lgbm-treelearner --lib`, `lgbm-boosting --lib`,
  `oracle-harness` incl. model-text / per-iter / raw_bin goldens; `oracle-harness
  --features rocm` when shared code changes). The CPU f64 anchor is untouched.
- "GPU is faster" only claimed in the regime the data supports.

## How to Build It

### 1. The whole-train BUDGET profiler (the tool that found everything)

The growth-loop `phase_prof` split (before/hist+split/partition) is **insufficient** for
GPU attribution — at ≥100 features the resident fusion folds the histogram kernel into
"scan" (`build=0` is an artifact, NOT a free build), and the three phases cover **<½** of
train wall-clock. Use the **whole-train BUDGET** (env-gated `LGBM_PHASE_PROF=1`, inert
otherwise), counters in `crates/lgbm-treelearner/src/phase_prof.rs`, wrapped at the
boosting/binning seam — NOT inside the growth loop:

- `BINNING` = `build_feature_columns` (once/train) · `SETUP` = `feature_infos_from_rows`
  (once/train) · `GRAD`/`SCORE` = per-iter objective/score-update · `LEARNER` = the
  per-tree `learner.train_*` call (⊇ the growth-loop phases) · `TRAIN_ONE_ITER` = the
  whole `gbdt.train_one_iter` (⊇ grad+learner+score+snapshot) · `SNAPSHOT`/`METRIC` =
  per-iter score `to_vec` clones / metric eval.
- Derived: `loop_other = train − binning − Σtrain_one_iter`; `in_learner_other = LEARNER
  − growth-phases` (per-tree GPU upload/orchestration).
- Bench at the wide shape: `LGBM_BENCH_SWEEP=wide` in `bench_gpu_vs_cpu.rs` (feat=500 ×
  rows {250k,500k,1M}, env-overridable `LGBM_BENCH_ROWS/FEAT/ITERS`, lighter warmup/reps
  since the per-bucket RATIO is the deliverable).
- **iters A/B** (iters=1 vs 8): `per_iter = (t8−t1)/7`, `fixed_setup = t1 − per_iter` —
  separates per-train setup (amortizes) from per-iteration cost.
- **CPU-backend A/B** at the same shape proves a cost is GPU-specific (e.g.
  `in_learner_other` was **16× higher** on GPU ⇒ device transfer, not host orchestration).

### 2. Lever — upload resident bins ONCE per train (`p9v`, commit 01e405d)

`learner.rs train_inner` runs **per tree**; it re-uploaded the **immutable** binned matrix
to the device every tree (`to_u32_vec` widen + concat + `create_from_slice` = 2×~2GB host
allocs + 2GB transfer at 1M×500). Guard with a `resident_bins_uploaded: bool` learner
field (init false in `new()`, reset in `with_features` when the feature set changes); set
true after the upload. Safe because `RocmBackend` is one instance per `train()` and
`reset_resident_pool` never clears the `resident_bins` cache. **1M×500 −32%.**

### 3. Lever — upload at NATIVE bin width (`qix`, commit ff4a10b)

The once-per-train upload still widened to u32. Upload at the narrowest uniform width
(`ResidentBinWidth{U8,U16,U32}` = widest `BinColumn` variant present). The 3
resident-reading `#[cube]` kernels (`construct_leaf_hist_resident_lds_kernel`,
`construct_leaf_hist_resident_kernel`, `build_fix_scan_fused_kernel`) become generic
`<B: Int>` (read via `u32::cast_from(resident[idx])` — value-faithful index), dispatched on
the stored width at each launch site (`ArrayArg` element COUNT is width-independent).
cubecl 0.10 supports generic launch kernels (`#[cube(launch_unchecked)] fn k<B: Int>` →
`::launch_unchecked::<u8,R>`; precedent: repo `hist_fold_body<N: Numeric>`). **Upload
bucket ~5× smaller, peak host mem ~4× lower.** (spike-006's "u8 device-READ ≈0%" still
holds — the win is transfer + host memory, not kernel compute.)

### 4. Lever — cache-friendly per-train host passes (`rdu` + `rsh`, commits bf467bd, b917191)

**The general pattern:** a row-major `Vec<Vec<f64>>` corpus + a `for j { for row { row[j] }}`
loop = `num_features` **cache-hostile strided column passes** (`row[j]` jumps
`num_features*8` bytes/read = DRAM-latency-bound). Transpose to ONE contiguous row pass:

- `feature_infos_from_rows` (min/max): single row pass accumulating per-feature min/max
  arrays — same `f64::min/max` (order-independent ⇒ byte-identical). **~8×** (3929→490
  ms/rep). `rdu`.
- `build_feature_columns` (binning): single row pass **scattering** `row[j] as u32` into
  pre-sized per-feature bin vectors (~`num_features*64B` hot tails stay L2-resident;
  single-threaded ⇒ no spike-011 false-sharing). Same bins, same order ⇒ byte-identical
  FeatureColumns. **~2.3×** (−57%). `rsh`.

## What to Avoid

- **Don't trust a hypothesized hot-spot without a 60-second sanity check + a measurement.**
  The "per-iter `to_vec` score-buffer clones are the loop overhead" hypothesis was wrong:
  an 8MB f64 clone is **sub-millisecond**, so a few/iter can't be seconds. Measurement
  found the real cost was `feature_infos` (cache-hostile), not the clones (3.2 ms/rep) nor
  the boosting loop (metric 43 ms/rep). `train_one_iter ≈ learner` exactly.
- **`build=0` in the GPU phase split is a fusion-labeling artifact**, not a free build —
  the kernel is folded into "scan" by the resident `build_fix_scan_resident` path.
- **The histogram kernel is NOT the wide-shape bottleneck** — at 1M×500 it's ≤⅓ of
  wall-clock; an equal share was redundant device upload, the rest per-train host setup.
  Don't pour effort into the (already-closed) kernel levers expecting the big win there.
- **Transpose moves strided access read→write** — only a win if the write working set
  (`num_features` Vec tails ≈ 32KB) stays L2-resident; it does single-threaded, but
  spike-011 showed the PARALLEL scatter loses to false-sharing. Measure; NULL if a wash.
- The per-train host setup (binning + feature_infos) **amortizes if the binned dataset is
  reused across trains** — the bench rebuilds per `train()`, so it's partly a harness
  artifact. The cache-locality wins are real regardless, but frame steady-state honestly.

## Constraints

- gfx1100 via cubecl-hip: Plane YES, f64 ops run (despite `has_f64==false`), atomics YES,
  `Array<u8>` works. GPU f32 atomics ⇒ ~1e-6 ROCm gate (NOT bit-exact); the CPU f64 anchor
  is the bit-exact gate.
- Resident-bin readers are exactly 3 production kernels (all must handle the upload width);
  the CUDA-mirror kernel is test-only (own u32 buffer) — leave it u32.
- 1M×500 needs ~32GB host RAM (4GB f64 corpus + ~2GB u32 transpose peak + binned).

## Origin

Synthesized from spikes: 014a, 014b (+ shipped quick tasks p9v, qix, rdu, rsh).
Source files available in: sources/014a-coarse-phase-attribution/,
sources/014b-gpu-launch-vs-compute-split/ (incl. `phase_prof.rs`).
