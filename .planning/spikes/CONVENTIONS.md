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

**In-kernel A/B via a comptime flag/factor (p93, 017).** To isolate EXACTLY one kernel
codegen change, parameterize ONE `#[cube]` kernel by a `#[comptime]` arg whose baseline
value is **byte-identical to production** (p93: `use_plane=false`; 017: `replicas=1`),
and sweep it. cubecl JIT-specializes per distinct comptime value, so uploads / cube
count / read-back are shared across arms and the delta is pure codegen. `SharedMemory`
can be comptime-sized from the arg (`new(replicas * MAX)`, arg typed `usize`) so LDS
footprint/occupancy scales honestly with the variant. For a noisy APU, the 3-round
"speedup vs cold round-1" reading OVERSTATES (cold baseline) — use **interleaved median
+ p25/p75 over ~11 reps, ≥2 process restarts**, and require a SEP-WIN (variant p75 below
baseline p25) before claiming a robust win (017 reference: `gpu_lds_replication.rs`).

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

## Host parity probe — resolve a reorder/precision risk BEFORE a GPU kernel (spikes 008, 016, 022)

When the question is "does numerical change X (a reorder, a quantization, a parallel scan)
flip the chosen SPLIT → tree divergence?", a pure-host f64 probe answers it for a fraction of
a GPU kernel's cost. The shape:

1. **Replicate the production scan/argmax faithfully** (call the REAL `gain::*` functions, mirror
   the gate ORDER + strict-`>` tie-break of `split_scan_body`), parameterized by the ONE thing
   under test (the prefix-sum association: `seq` vs the candidate order).
2. **Use the EXACT order the backend emits, not a proxy.** cubecl-hip `plane_inclusive_sum`
   lowers to a Hillis-Steele `__shfl_up` loop (`cubecl-cpp/src/shared/warp.rs::reduce_inclusive`)
   — model THAT, not a generic pairwise tree (016 used a proxy and could only get the magnitude).
3. **Classify divergences by present-data IMPACT, not just count.** A flip is only REAL if it
   changes the partition of PRESENT data (different threshold, or a *populated* bin rerouted).
   Reroutes of missing values / empty default bins / equal-gain plateaus (gain reldiff ~1e-12)
   are COSMETIC — same tree structure + leaf values within the hip ~1e-6 gate, the tie class the
   hip split test is already tie-aware for (def-hip-split, 1832206).
4. **Add a direct mechanism demo when random generation under-covers the key case** (022: the
   random sweep never landed a populated-default near-tie, so a fixed-split mass-sweep proved the
   gain gap is *linear in mass* ⇒ only an empty bin can flip). Don't conclude from a null the
   sweep simply never exercised.
5. **A green host gate is necessary, not sufficient** — it retires the f64-reorder risk; the GPU
   f32-vs-f64 envelope is a separate, larger, already-documented residual that only *widens* the
   cosmetic tie band. The host probe never needs the GPU to answer the parity question.

`spike022_default_bin_parity_probe.rs` / `spike016_scan_reorder_probe.rs` are the references.

## Tools & Libraries

- `cubecl` 0.10 LDS API: `SharedMemory::<Atomic<f32>>::new(COMPTIME_SIZE)`, `sync_cube()`,
  per-cube atomics. `Array<u8>` compiles+runs on HIP (006). f64 ops run on gfx1100 despite
  `has_f64 == false` (used by the f64 anchor kernels).
- **`Atomic<i64>` is broken on cubecl-hip 0.10** (018): `Atomic<i64>::store` lowers to
  `atomicExch(long long*)`, which HIP lacks → compiles, fails at runtime. Use `Atomic<u64>`
  two's-complement (store the i64 bits as u64; wrapping `fetch_add` == signed add; reinterpret on
  readback). This is how the shipped u64 fixed-point build accumulates.
- **cubecl topology-constant types (021):** the *linearised* builtins `ABSOLUTE_POS`, `CUBE_POS`,
  `CUBE_COUNT` are **`usize`**; the per-axis ones (`CUBE_POS_X`, `UNIT_POS`, …) and sizes
  (`CUBE_DIM`, `PLANE_DIM`) are **`u32`**. `let f = ABSOLUTE_POS as u32;` before u32 arithmetic.
  `plane_inclusive_sum` lowers to a Hillis-Steele `__shfl_up` loop on HIP (model THAT order in
  reorder-parity probes, not a pairwise tree).
- **GPU device-time A/B discipline (017/018/019/020):** compute-throughput timing (accumulate
  ~20 launches into ONE reused buffer + a single `read_one_unchecked`), interleaved
  `median[p25..p75]` over ≥9 reps, ≥2 process restarts; require a SEP-WIN (variant p75 < baseline
  p25); judge the SIGN only (spoofed 8-CU APU ⇒ absolute Mr/s is confounded, rocprof unsupported).
- **`#[cube]` kernel authoring gotchas (024 — 3 rebuilds spent):** (1) launch SCALARS pass
  **raw** (`nb`, `n as u32`), NOT wrapped in `ScalarArg`. (2) numeric casts inside a cube body
  use `f64::cast_from(x)`, NOT `x as f64` (u32→f64 etc.). (3) the cube macro supports **neither**
  a `macro_rules!` invocation in the body (`error: Unsupported macro`) **nor**, in some
  signatures, a `#[cube]` helper fn (`f64: From<NativeExpand<f64>>`) — **inline the body
  directly** (the 022b/024 precedent). (4) loop-carried mutables MUST init from a **plain
  literal** — a scientific-notation `-1.0e30f64` sentinel trips the MLIR lowering
  (`From<NativeExpand<f64>>`); use `0.0f64` when all values are ≥0 (like `split_scan_body`).
- **Per-leaf launch/round-trip COUNTERS (023):** `phase_prof.rs` has `BUILD_RESIDENT_CNT`,
  `SUBTRACT_RESIDENT_CNT`, `SCAN_RESIDENT_CNT` (= blocking readback syncs), `FUSED_CNT`, bumped
  at the per-leaf Backend entry points, dumped as a `COUNTS:` line under `LGBM_PHASE_PROF=1`.
  Use them to make the launch/sync floor empirical (parity-neutral; inert when the gate is off).
- Reference baseline for GPU kernel parity/perf = AMD's ROCm fork `LightGBM-release-4.6.0.99/`
  (hipified CUDA), NOT mainline `LightGBM/` — see
  `.planning/notes/cubecl-vs-rocm-histogram-kernel-comparison.md`.

## CPU / host isolated-A/B harness (026–029 — the partition arc)

The repeatable shape for "is host-side change X faster?" — and unlike the GPU harness these are
LEGITIMATE wall-clock (the 16-core CPU is real hardware; only the GPU is the spoofed APU):

1. **Self-contained `crates/lgbm-compute/examples/spikeNNN_*.rs`** using the real `BinColumn` +
   kernel/op types. Deterministic data via a small inline LCG (no `rand` dep; `Math.random` is
   banned anyway). Model a SCATTERED leaf — shuffle the row indices so the `feature_bins.bin(row)`
   gather is RANDOM (a deep leaf's rows aren't contiguous); a contiguous leaf understates the cost.
2. **Sweep BOTH axes that move the regime:** size (1k→4M) AND skew (balanced vs 0.9 — serial
   branch-prediction crushes skewed data, and real trees deepen INTO skew) AND bin width (U8 the
   production narrow case vs U32). A win at one cell is not a win.
3. **Decompose the op into sub-phase timers** (`LGBM_SPIKE_PROF`) to LOCALIZE the cost before
   optimizing — 026 split marshal/count/scatter (found the scatter wall), 029 split host-gather /
   upload / kernel+readback (found the upload is NOT free on shared DDR5). Optimize the dominant
   phase, not the assumed one.
4. **median over ~21–30 reps, ≥2 process restarts**, `std::hint::black_box` the result, warmup
   discard. Report the ratio vs the production-faithful baseline (serial-native / current-path),
   and a **byte-identity parity column** every cell (partition is f64-free ⇒ bit-EXACT, no tol).
5. **Memory-bound diagnosis:** if a perf delta is FLAT across a granularity knob (chunk size in
   026) it's fixed overhead / bandwidth, not compute — parallelism won't help. Shared DDR5 means
   the 16 cores share one controller; cutting TRAFFIC (narrower types, fewer materializations)
   beats adding cores. But transfer volume is NOT free even on an APU (029).

## Wiring a spike into production — additive backend discriminator + stale-worktree integration

- **Gate a backend-specific path on a default-false trait method overridden on ONE backend**, never
  a global env/flag: `Backend::prefers_host_partition() { false }` (CpuBackend true, 027);
  `Backend::data_partition_native(&BinColumn,…)` with a widening DEFAULT that delegates to the
  existing op + a RocmBackend override (029). This is the same idiom as `wants_resident_bins` /
  `resident_pool_supported`; it keeps every other backend byte-unchanged and avoids changing
  existing signatures (low blast radius).
- **Make device kernels generic-over-`Int` + `match width`-dispatch the launch** for native-width
  uploads (the qix histogram precedent; 029 applied it to `data_partition_kernel`).
- **`/gsd-quick` executor worktrees can branch off a STALE base** (observed twice: hw2, j1l — based
  on an old commit predating recent merges). The fix: cherry-pick the executor's commits onto
  master, hand-resolve conflicts (land the change in the CORRECT current-tree branch — e.g. the
  device `else` of `prefers_host_partition`, not the pre-gate `split`), and **RE-RUN the full
  bit-exact gate on the integrated master tree** (incl. the ROCm parity test on the GPU) — the
  executor's green gate ran on the stale tree, not master.
