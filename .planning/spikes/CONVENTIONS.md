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

**"Remove-the-suspect" re-attribution + model the REAL access order (spike 030).** To find
WHICH cost dominates a kernel (not just "is X faster"), build N variants of the LIVE kernel
that each DELETE one suspected cost while keeping the rest live; whichever deletion moves the
clock IS the bottleneck. Defeat dead-code elimination by accumulating deleted-path loads into a
register written once at the end. Pair complementary deletions to disambiguate (030: `SEQ_BIN`
deletes the random gather AND the index read → ambiguous; adding `COAL_BIN` = same array/bytes
read SEQUENTIALLY isolated the uncoalesced PATTERN from bandwidth). **Re-attribute after every
build change** — 030 found the post-u64 build is uncoalesced-gather-bound, NOT the atomic-bound
that spike-015 declared pre-u64 (the bottleneck moves; the old verdict goes stale).
**Critical: model the production access ORDER, not a random permutation.** A random `leaf_rows`
gather overstated the uncoalesced penalty **5–10×** vs the real STABLE-partition monotone-
increasing subset (`(0..N).step_by(k)`), which already sits at ~70% of the coalesced ceiling.
Report **Mr/s** (reads/sec) so variants with different row counts compare fairly. Reference:
`spike030_build_roofline_ab.rs`.

**Measure divergence with a controlled identical-total-work ladder (spike 036).** Divergence
is the ONE GPU micro-arch effect that IS cleanly sign-measurable on the spoofed 8-CU APU (it's
a wavefront-SCHEDULER property — lockstep masking — not CU-count/memory-bound, the axes the spoof
confounds). To prove a divergence delta is real (vs noise) at the magnitude you care about:
build N arms that do **identical total work** (sum of per-lane loop trip counts equal across
arms) and differ ONLY in how that work is **distributed across wavefront lanes** — `UNIFORM=K`,
`DIV2=lane%2?0:2K`, `DIV4`, `DIV32` (interleaved `lane%n` keeps the imbalance within any 32/64
group ⇒ wave-width-robust). Loop body = constant ALU/iter into a register sink written once
(no memory, defeats DCE); trip count from a DEVICE ARRAY (compiler can't specialize). If
wavefronts serialize to the slowest active lane, wall-clock scales 1:2:4:32 despite constant
useful work (idle masked lanes = pure waste). Measured near-ideal (1.00/1.9/3.7/27×, 2
restarts) ⇒ resolvable; a collapse →1 ⇒ not. **But measurable ≠ worth it** — 036 also found the
production kernels are ALREADY branchless (`select`-everywhere, MLIR-forced) and the dominant
build is divergence-free by construction, so this ladder is a GATE to run BEFORE a real divergence
A/B, not a lever itself. Reference: `spike036_divergence_measurability.rs`.

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

## CubeCL autotune harness (037–040 — code from the SOURCE, not the manual)

The repeatable shape for "let cubecl autotune pick a launch config" on the rocm backend.
The `cubecl_manual/.../12_autotuning.md` is **idealized and internally inconsistent** — read
the real 0.10 API in `~/.cargo/registry/.../cubecl-runtime-0.10.0/src/tune/` first.

1. **The real 0.10 API (3 manual divergences):** (a) `TunableSet::new(key_gen, input_gen)`'s
   FIRST closure is the **KeyGenerator** `for<'a> Fn(&I::At<'a>) -> AutotuneKey` — it returns
   the **key type**, NOT a String (the manual returns `"axpy-tune"` — wrong). (b)
   `LocalTuner::execute(id, client, set, inputs)`'s first arg is the cache-namespace **ID**
   (`Display`, e.g. `"rocm:0"`), NOT the AutotuneKey — the key is generated INTERNALLY from
   inputs. (c) the `AutotuneKey` trait alias requires `serde::{Serialize, DeserializeOwned}`
   under the `std_io` cfg (always on linux ⇒ persistent cache active) — add a `serde` dep
   (dev-only if the key lives in an example). `cubecl::tune::*` is the import path
   (re-exported via cubecl-core `lib.rs:50`). `local_tuner!("name")` works; the static lives
   at module scope. Inputs type = `Vec<cubecl::server::Handle>` (blanket `TuneInputs`).
2. **Accumulating kernels need a fresh-output InputGenerator, NOT `CloneInputGenerator`**
   (037/038). `Handle::clone` is a ref-count bump (NOT a buffer copy), so `CloneInputGenerator`
   makes every benchmark rep `fetch_add` into the caller's REAL `out` ⇒ **N× corruption**
   (measured 27×; N = the whole sample budget, not +1). Fix: a struct `impl InputGenerator`
   whose `generate<'a>(&self,_k,inputs) -> <Vec<Handle> as TuneInputs>::At<'a>` returns a new
   `Vec<Handle>` with the output handle replaced by a fresh zeroed buffer (the winner's FINAL
   run uses the ORIGINAL inputs, so the real `out` is touched exactly once → `rel_err 0` by
   grad-conservation). GAT gotcha: spell the return through `…::At<'a>` or E0195. Classify
   kernels: OVERWRITE (store) = safe as-is; ACCUMULATE (build) = fresh-output; in-place RMW
   (partition) = deep-COPY generator (but partition is host-routed on rocm per 035).
3. **Key on the occupancy REGIME, not exact dims** (039). Per-leaf row counts never repeat ⇒
   keying `AutotuneKey` on exact `rows` = a tuning STORM (every node a cold ~40ms tune;
   25/25 cold = 975ms for ONE shallow tree). Key on `log2(rows)` (or size-bands) + feats +
   bins: ~one tune per size-decade, 20/25 nodes free (~3× faster than exact), AND it still
   captures the variant crossover (the choice tracks the regime, not the exact count). FIXED
   (feats-only) is cheapest but mis-applies the root's variant to small leaves.
4. **Selection is the spoof-robust axis** (037/040). Absolute Mr/s is APU-confounded, but the
   tuner's RELATIVE within-device pick is sound: it independently re-derived spike-007's P=16
   and (040) BEAT the shipped 8-CU `row_partition_count` heuristic ~10% (which under-partitions
   to P=1 at the production 50-feat width). Read the winner from the persisted
   `target/autotune/0.10.0/rocm_0/*.json.log` (`fastest_index` → PSET[idx]). Clear that dir to
   force true cold tunes when measuring tuning cost. **Measure-don't-model**: autotune ≥ the
   analytic heuristic at every cell, and self-calibrates across hardware (the portability win).
   Reference harnesses: `spike037_autotune_hip_feasibility.rs` … `spike040_autotune_vs_heuristic.rs`.

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

## Audit the SHIPPED wiring, not just the spike that validated it (spike 032)

A spike validates a design; the *production wiring* can quietly re-introduce the cost the
spike removed. Spike-027 validated a ONE-random-gather fused partition, but the wired
`split_fused_host` added a separate validation pass that does a SECOND full random gather
over the leaf (`data_partition.rs:236-246`) — re-missing cache at scale, exactly the
traffic 026→027 cut. Before spiking a *new* lever on a hot path, **re-read the live code
and diff it against the spike that "shipped" it**; a redundant pass added for parity/safety
is often a free, bit-exact reclaim (fold it into an existing pass with an early-return
*before* any mutation = same error semantics; or relocate once-per-train, the 003b/r4o
precedent). This is the host-CPU analog of the GPU "re-attribute after every build change"
rule.

**The inverse also bites: a "WIRE pending" verdict may already be SHIPPED (spike-034).** The
MANIFEST rows for 024/029 still read "WIRE = human call" / "WIREABLE", but both had been wired
in later work (024 = Phase 12; 029 = quick-260625-j1l) that never updated the verdict text.
Before acting on a "validated-but-unwired" verdict, **grep git log + read the live code** to
confirm it isn't already in production — `git log --all -i --grep=<feature>` + check the call
site. Then (a) re-point the stale verdict to SHIPPED, and (b) RE-PROFILE — a shipped lever has
already moved the bottleneck (034: post-024 the launch-bound floor moved from scan-sync to
partition). Verdict text in long campaigns lags the tree; trust the code, not the row.

## Don't refactor `.bin()` per-row matches into typed-slice loops (autovectorization trap, spike 033)

`BinColumn::bin(row)` does a per-row enum `match` (U8/U16/U32). It is tempting to hoist the
match out of a hot loop and gather off a typed `&[T]` slice. **DON'T** — on x86 the tight
typed loop auto-vectorizes into an AVX **gather** (`vpgatherdd`), which serializes
cache-missing lanes WORSE than scalar independent loads do under out-of-order MLP. Measured
**1.5–2× SLOWER** than the scalar `.bin()` path at scale, all sizes, 3 restarts
(spike-033 `p0/p1 = 0.5–0.9×`). The per-row match defeats vectorization and is the FASTER
codegen for a random gather. Keep `.bin()` in random-gather loops.

## Software prefetch only pays when the gathered array ≫ LLC (spike 033)

`_mm_prefetch`-ahead of a random gather is null-to-SLOWER until the gathered array vastly
exceeds last-level cache — then it hides miss latency (spike-033: ~2–3× whole-op at 4M×U32,
null/negative below). For this project that regime is **wide U16/U32 bin columns at
multi-million rows** (high-cardinality features), NOT the production U8 default (dense column
barely exceeds cache even at 4M rows ⇒ ~1.1× at a root split only). Optimal distance grows
with array size (D≈16 dense → 128 at 4M×U32); T0≈NTA. Gate any prefetch wire on
`width != U8 && leaf_rows ≥ ~2M`, or skip it — and weigh the x86-only-intrinsic portability
cost on the must-build-everywhere CPU anchor.

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

## SIMD vectorization with `Vector<P,N>` ("Line") on cubecl 0.10 (spikes 041–045)

**The type is `Vector<P: Scalar, N: Size>`, NOT `Line<T>`.** cubecl 0.10 has no `Line`
(it's a later rename; the burn cubecl-book + context7 `main` docs show the new name). Read
from the SOURCE — `cubecl-core-0.10.0/src/frontend/container/vector/base.rs` + the canonical
`src/runtime_tests/vector.rs`. Launch ABI:

- kernel sig `&Array<Vector<F,N>>`, `N: Size` a generic param;
- the `N` value is a RUNTIME `usize` positional arg inserted **right after `CubeDim`**,
  before the kernel's own args: `k::launch_unchecked::<F,R>(client, count, dim, vector_size, …)`;
- `ArrayArg::from_raw_parts(handle, n_elements / vector_size)` — length in **vector units**
  over the SAME byte buffer; a bare `usize` kernel param is passed RAW (cf. production `num_data,`);
- sweep widths from `client.io_optimized_vector_sizes(size_of::<F>())` (hip f32 → `[4,2,1]`;
  cpu f32 → `[16,8,4,2,1]`, f64 → `[8,4,2,1]`); element read `v[i]`, `Vector` impls `Add/Sub/…`
  element-wise so any element-wise op is **bit-exact** to scalar (no float reorder).

**THE RULE (the campaign's one durable finding): `Vector<P,N>` pays ONLY where the kernel is
memory/throughput-bound AND the vectorized op covers the bottleneck.** Evidence:

| kernel | op shape | result | why |
|--------|----------|--------|-----|
| SUBTRACT (041) | pure streaming load–sub–store | cpu **2.5–3.7×**, hip **1.06–1.29×**, bit-exact | load+store+compute all vectorize; memory-bound |
| SCAN read (042) | dependent prefix-sum + divide + argmax | **null** (0.88–1.08×), bit-exact | only the load vectorizes; dependent chain dominates |
| BUILD grad/hess (043) | gather-latency bound; grad/hess = 8–14% | null→**REGRESSION** at wide (0.83×) | load latency hidden behind gather; extract adds occupancy pressure |
| fix_compact DEQUANT (044) | streaming `u64→f64` map (memory-bound) | cpu-f64-vec8 **2.52×**, hip-f64-vec2 **~1.1× weak** | right shape + bit-exact, but a FUSED minority fraction ⇒ sub-1% e2e; cpu fix path is native |
| COALESCED-build + Vector (045) | reorder rows contiguous → vector-read grad/hess/bin → LDS scatter | **NET LOSS** (0.56–0.98× vs permuted); COAL_V **0.70–0.97× vs COAL_S** | reorder IS the same permuted gather (can't amortize, read-once); coalesced build is atomic-scatter-bound not load-bound ⇒ vector regresses |

Corollaries: (1) the win **scales with problem size** — bench at the WIDE shape, a small
op is overhead-bound (041: cpu-f32 1.2× at 25.6k → 3.7× at 256k). (2) On hip sweep to the
MAX `io_optimized` width — intermediate widths are magnitude-noisy on the APU. **hip caps f64
at vec2** (128-bit load / 64-bit) vs f32-vec4, so f64 streaming maps have a lower ceiling (044).
(3) A permuted gather (`bins[col+leaf_rows[k]]`) is **structurally un-vectorizable** (no `Vector`
read gathers arbitrary `p`); only contiguous reads vectorize. (4) Vectorizing a
NON-bottleneck isn't free — it competes for registers/occupancy and can REGRESS (043 wide).
(5) `Atomic<u64>` is **unimplemented on cubecl-cpu** (panics) — u64-atomic kernels are hip-only.
(6) `Vector` supports **cross-type casts** bit-exactly (`Vector<u64,N>→<i64,N>→<f64,N>`, 044) —
covers type-converting streaming maps; divide-by-const needs a broadcast `Vector::<f64,N>::new(S)`,
not `vec / const` (the `Div<$lit>` impl needs a literal token). (7) **The corollary that decides
WIRING:** a correctly-shaped streaming map only pays e2e when it's a MAJORITY of the kernel's work
AND on a path the backend runs through cubecl — subtract (041) wins because it IS its whole kernel;
dequant (044) is bounded because it's a fused minority fraction. The histogram-pipeline frontier is
now FULLY MAPPED: subtract WON (041, shipped), scan null (042), build immune (043), dequant bounded
(044) — no un-probed Vector lever remains in the histogram path. (8) **The coalesced-rewrite escape
hatch is also CLOSED (045).** 043's one named lever — REORDER each leaf contiguous first so the
build reads coalesced and Vector can apply — is a NET LOSS (0.56–0.98× vs the permuted build, 2
restarts × 2 shapes): the reorder IS the same permuted gather (030's bottleneck), and it can't
amortize (build reads each bin once-per-leaf, stable order changes every split — the spike-028
read-once wall). AND Vector still regresses on the coalesced layout (COAL_V 0.70–0.97× vs COAL_S),
because a coalesced build is **LDS-atomic-scatter-bound, not load-bound** — vectorizing the load
attacks a non-bottleneck + the extract adds occupancy pressure (043's wide mechanism, now confirmed
even when the gather is contiguous). Reopens only on discrete gfx110x (030: harsher permuted
penalty there may let the reorder amortize). cube-macro gotchas (045): literal `Vector<_,2>` panics
the macro (need generic `N:Size`); `N::value() as usize` in an unroll bound → runtime Vector index
→ segfault (use `#[unroll] for j in 0..N::value()`); a reorder dest-stride ≠ source-stride is an
easy OOB→segfault.

**Production fit:** the clean win (041 subtract) lands on `subtract_hist_kernel` — the verbatim
kernel the rocm RESIDENT subtract (`subtract_histograms_f64_from_handles_on`) + portable
cuda/wgpu launch; the CPU anchor subtract is NATIVE (`subtract_histograms_cpu_native`) so it's
untouched (merge gate safe). But subtract is a non-dominant phase (034: build dominates wide,
partition 30–38% launch-bound) on an APU that loses to CPU ⇒ ROCm-parity-track, bounded e2e.
Reference harnesses: `spike041_vector_subtract_ab.rs`, `spike042_…`, `spike043_…`, `spike044_vector_dequant_ab.rs`, `spike045_coalesced_build_vector_ab.rs`.

## Real-discrete-GPU profiling (Kaggle) — spikes 046/048/049

The local "GPU" is a spoofed 8-CU APU; **Kaggle is the only real-discrete-NVIDIA
measurement path** (cubecl-cuda via `maturin build --release -F cuda`). Conventions:
- **Profiling the Python path requires the spike-046 hook** — `phase_prof::dump("train")`
  in `booster.rs::train_inner_columns_full` (env-gated `LGBM_PHASE_PROF=1`, parity-neutral).
  Without it the shipped wheel emits zero attribution. The `dump()` was historically
  bench-examples-only.
- **Kaggle CLI** is authenticated as `boomvector` (ACCESS_TOKEN at `/home/user/.kaggle`,
  no kaggle.json). Kernel `boomvector/lgb-rs-cuda-bench` git-clones master, so push code
  first. Harness scripts live in `spikes/046-python-path-phase-prof/`.
- **Absolute walls are NOT cross-session comparable** (Kaggle assigns T4/P100/T4×2 — saw
  17/20/11s across sessions). Trust **in-session A/B deltas** only.
- **Attribute backend-independent (CPU) components LOCALLY** — the `spike046_validate`
  example takes `LGBM_VALIDATE_ROWS/ITERS`; only true GPU-device costs need a Kaggle cycle.

## Real-CUDA zero-code env-toggle probes + parse rules (spikes 051/052/054)

The cheapest real-CUDA spike is one with **NO code push** — sweep existing env toggles on
the *current* master and read `phase_prof`. This closed three hypotheses in three runs:
- **Zero-code occupancy/fusion probes:** `LGBM_AUTOTUNE_FORCE_P=k` pins the build
  row-partition P (unclamped, bypasses `ROWPART_P_MAX`); `LGBM_AUTOTUNE=0` forces the P=1
  heuristic; `LGBM_FUSED_FORCE=1` forces the `build_fix_scan` fusion; `LGBM_SIBLING_COPACK=0`
  disables the default-on scan co-pack. A driver that loops these per-arm (fresh subprocess
  each — `phase_prof` atomics are process-global) localizes the bottleneck with one wheel build.
- **Each arm runs ONE backend in its OWN subprocess** under `LGBM_PHASE_PROF=1`; capture
  stdout (the `RESULT … train_time_s=`) + stderr (the dump). Driver pattern in
  `spikes/051-*/spike051_kaggle.py` (reusable; inner bench inlined as a string ⇒ no git push).
- **PARSE RULE — read the absolute-ms line, NOT the `%:` line.** Each fit emits TWO
  `[phase_prof:train]` blocks: a **warmup** dump (`device_launches`≈445 — absorbs the cold
  CUDA-context + kernel-JIT, several seconds) then the **timed** 100-tree dump
  (`device_launches`=8570+). Select the **max-launches** record. And the dump has both an
  absolute `before=… hist+split=4897ms …` line AND a `%: … hist+split=73.8` percentage line —
  a naive `hist\+split=([\d.]+)` regex matches BOTH; key off the line *starting with*
  `before=` for absolutes. (This bug made spike-051's first summary read percentages as ms.)
- **Dedicated kernel per spike** (`boomvector/lgb-rs-cuda-spike0NN`) keeps runs parallel-safe
  and leaves the shared `lgb-rs-cuda-bench` kernel untouched.
- **Build official LightGBM with CUDA** for a reference ratio:
  `pip install --no-binary lightgbm lightgbm -C cmake.define.USE_CUDA=ON` (source build,
  several min; gate the import-probe first). Needed only when comparing vs official (054).
- **FINDINGS (real NVIDIA, 500k×50, 100 trees):** build occupancy is NOT a lever (FORCE_P
  flat-to-worse, P=1 optimal — the APU's P-sensitivity does not transfer); `LGBM_FUSED_FORCE=1`
  is **5.4× WORSE** (the f64 fused mega-kernel tanks on consumer-NVIDIA 1/32 f64 — keep the
  u64 fixed-point separate path; **avoid f64 hot loops in any new CUDA kernel**); readback
  **syncs are cheap** (~0.14ms; `copack=0` doubles them for +3.6%); the wall is **8570 small
  serial launches** gated by the best-first build→subtract→scan chain. The lgb_rs/official gap
  **halves with feature width** (3.90×@50f → 1.93×@500f, launches constant) but never closes —
  **the on-device multi-leaf learner is the universal architectural lever**; all cheap levers
  are refuted on real hardware.
