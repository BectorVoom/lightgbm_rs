# Spike Conventions

Patterns established across spike sessions. New spikes follow these unless the question requires otherwise.

## Stack

- **Rust + cubecl 0.10**, the project's own crates. Spikes that touch the histogram/build path
  live as `crates/lgbm-compute/examples/*.rs` (not throwaway scripts) so they compile against the
  real kernels and types.
- **CPU spikes** (002–005): plain `cargo bench`/example timing, compared vs `lib_lightgbm` 4.6.
- **GPU spikes** (006, 007): `--features rocm` examples on the local gfx1100.

## GPU micro-bench harness (spikes 006, 007)

The repeatable shape for "is kernel-change X faster on the GPU?":

1. **Isolate ONE kernel access pattern** in a `#[cube(launch)]` fn that mirrors the production
   kernel (e.g. resident-column gather + f32-atomic accumulate). Vary only the one thing under
   test (bin width in 006; row-partition count P in 007). Bench the variant against a baseline
   that is **byte-identical to production** (006: u32; 007: P=1).
2. **Drive it from `lgbm_compute::runtime::rocm_client()`**, `create_from_slice` device handles
   once, loop `LAUNCHES` accumulating launches, force sync with a final `read_one_unchecked`.
3. **Report within-round ratios, not vs a fixed cold baseline.** Round-1 of the first variant is
   cold-start inflated — comparing across rounds overstates wins. Read each round's variants
   against each other, and re-run across **2–3 process restarts** to kill warmup-drift before
   declaring a verdict.
4. **Always include a correctness column** vs the production-equivalent baseline (max_abs +
   max_rel diff). f32-atomic reorder noise is expected; watch for divergence that *grows* with
   the change (007: more partitions → wider divergence — a real parity interaction, not noise).
5. **Gate findings against the MANIFEST Requirements**: the CPU f64 anchor is the bit-exact merge
   gate and must stay untouched; the GPU ~1e-6 parity contract at large shapes is separately open.

`gpu_bin_width.rs` (006) and `gpu_row_partition.rs` (007) are the reference harnesses.

## CPU data-layout micro-bench harness (spikes 010, 011)

For "is data-structure change X faster on the CPU?", the **end-to-end `bench_train`
example is too noisy** to resolve a single-function change (±10% run-to-run; at the
bench shapes the targeted op is a small slice of train time). Both spikes had to fall
back to an **isolated in-process microbench** of the one function/structure:

1. **Reproduce ONLY the structure/op under test** as `before`/`after` *local closures*
   (self-contained, not depending on which variant currently ships), in a permanent
   `#[ignore]`d in-crate test (it needs the private fold/layout). Run with
   `--ignored --nocapture`.
2. **Interleave before/after per launch** to cancel thermal/scheduler drift; report the
   **median** of N launches after a warmup; `black_box` a sink to defeat DCE.
3. **Sweep the size** — the win/regression often only appears at the shapes the code
   path actually runs on (011: scatter regressed only at threshold leaf sizes; 010: the
   alloc ceiling only matters at medium/large slot_len). A single size hides it.
4. **Re-run across 2–3 process restarts** before a verdict (warmup/allocator drift).
5. **THEN confirm end-to-end** with `bench_train` — the isolated ceiling OVERSTATES the
   real win (010: 8–22% isolated → ~4% end-to-end, allocator amortizes). Ship on the
   end-to-end number, not the microbench.
6. **Bit-exact gate** for any anchor-touching change: `cargo test -p lgbm-treelearner
   --lib` + `cargo test -p oracle-harness` (incl. `raw_bin_train_parity` vs lib_lightgbm).

`spike010_pool_alloc_ceiling` and `spike011_microbench` are the reference harnesses
(in-crate, `#[ignore]`d; sources copied to each spike dir).

## GPU whole-train attribution harness (spike 014)

For "where does GPU train wall-clock actually go at scale?", the growth-loop
`phase_prof` split (before/hist+split/partition) is **insufficient** — at ≥100 features
the resident fusion folds the histogram kernel into "scan" (`build=0` is an artifact),
and the three phases cover <½ of train wall-clock.

1. **Use the `LGBM_PHASE_PROF=1` whole-train BUDGET** (spike-014b): counters
   `BINNING/GRAD/LEARNER/SCORE` + `UPLOAD` drill-down in `phase_prof.rs`, wrapped at the
   boosting/binning seam (`gbdt.rs`, `booster.rs`) NOT inside the growth loop. `LEARNER ⊇
   growth-phases`, so `in_learner_other = LEARNER − phases` exposes per-tree GPU device
   work (upload/resident/sync) the growth guards never saw.
2. **Bench at the wide shape via `LGBM_BENCH_SWEEP=wide`** (`bench_gpu_vs_cpu.rs`):
   feat=500 × rows {250k,500k,1M}, env-overridable `LGBM_BENCH_ROWS/FEAT/ITERS`. Lighter
   warmup/reps (1/3) — the per-bucket RATIO is the deliverable, not a tight median.
3. **Decompose fixed vs per-iteration with an iters A/B** (iters=1 vs 8): per-iter =
   (t8−t1)/7; fixed setup (binning) = t1 − per-iter. Catches bench-repeated setup costs
   that amortize in real bin-once-train-many usage.
4. **A/B the CPU backend at the same shape** to prove a cost is GPU-specific
   (`in_learner_other` was 16× higher on GPU ⇒ device transfer, not host orchestration).
5. **Instrumentation must be behavior-neutral**: `phase_prof::time`/`guard` are
   passthrough when the env gate is off; verify the bit-exact gate stays green
   (`lgbm-treelearner --lib` + `lgbm-boosting --lib` + `oracle-harness`).

## Tools & Libraries

- `cubecl` 0.10 LDS API: `SharedMemory::<Atomic<f32>>::new(COMPTIME_SIZE)`, `sync_cube()`,
  per-cube atomics. `Array<u8>` compiles+runs on HIP (006). f64 ops run on gfx1100 despite
  `has_f64 == false` (used by the f64 anchor kernels).
- Reference baseline for GPU kernel parity/perf = AMD's ROCm fork `LightGBM-release-4.6.0.99/`
  (hipified CUDA), NOT mainline `LightGBM/` — see
  `.planning/notes/cubecl-vs-rocm-histogram-kernel-comparison.md`.
