# Pitfalls Research

**Domain:** Pure-Rust rewrite of Microsoft LightGBM (gradient-boosted decision trees) on CubeCL (CPU + AMD ROCm), under a strict **1e-12 absolute output-parity oracle** vs the C++ reference.
**Researched:** 2026-06-05
**Confidence:** HIGH on numerical-parity and RNG pitfalls (verified against the C++ source in-repo + LightGBM issue tracker); MEDIUM on CubeCL/ROCm specifics (CubeCL is alpha v0.10.0, docs evolving); HIGH on PyO3 binding pitfalls.

---

## TL;DR — the three things that will sink this project if mishandled

1. **The 1e-12-on-ROCm contract is, taken literally, almost certainly unachievable for any code path that touches GPU reductions or transcendental functions (`exp`/`log`/`pow`/`sigmoid`).** LightGBM's *own* `deterministic=true` does **not** reproduce results across compilers, CPU instruction sets (FMA/AVX), or machines — the maintainers state this explicitly ([#6683](https://github.com/microsoft/LightGBM/issues/6683)). A GPU backend is a *more divergent* environment than "a different CPU." The oracle strategy must be defined precisely (what is compared, on which backend, against which reference build) **before** any kernel work, or the project has an unfalsifiable acceptance criterion.
2. **RNG parity is binary and unforgiving.** LightGBM uses a hand-rolled 32-bit LCG (MSVC constants) with float-precision sampling and a `std::set`-ordered reservoir sampler. One wrong cast, one `u32`-vs-`i32` sign slip, or one different sampling branch and bagging/feature-subsampling selects *different rows/features* → a completely different tree → oracle fails on iteration 1. This must be ported bit-exactly and unit-tested against the C++ `Random` before any training loop is built.
3. **The split decision is a discontinuous function of floating-point sums.** A 1-ULP difference in `sum_gradients` can flip which bin wins the best-split scan, and that divergence *cascades* into an entirely different tree. There is no "close enough" — either every split matches or the trees diverge. This is why parity must be validated at the *granularity of bin boundaries → histograms → per-split gains → leaf outputs*, not just final predictions.

---

## Critical Pitfalls

### Pitfall 1: Treating "1e-12 on every backend including ROCm" as a literally-achievable invariant

**What goes wrong:**
The project's headline contract is `|rust_output − cpp_output| ≤ 1e-12` on **both** CPU and ROCm. For large parts of the pipeline this is unachievable as stated:
- GPU floating-point **reductions are non-associative** and reorder additions relative to the C++ CPU loop; the low bits of `sum_gradients`/`sum_hessians` differ, and those feed split gains.
- GPU **transcendental intrinsics** (`exp`, `log`, `pow`, division) differ from CPU `libm` by **multiple ULP** (vendor HIP/CUDA math ≠ host `libm`; this is documented across GPU math-library studies). For `f64`, ~2-4 ULP near 1.0 is already ~1e-15–1e-16, but error *amplifies* through `sigmoid`→`logloss`→accumulated boosting scores over hundreds of iterations, and through divisions in leaf-output formulas. After 250+ boosting rounds LightGBM's *own* runs diverge to correlation 0.985 from pure FP error propagation ([#3372](https://github.com/microsoft/LightGBM/issues/3372)).
- LightGBM itself **does not** guarantee 1e-12 across machines/compilers; FMA contraction (`a*b+c` fused vs separate) alone breaks bit-parity, and the C++ build's FMA usage depends on compiler flags and target ISA ([#6683](https://github.com/microsoft/LightGBM/issues/6683)).

**Why it happens:**
The contract was written as a *product aspiration* ("numerical fidelity is the core value") and conflated with a *testable tolerance*. 1e-12 absolute is reasonable for the **integer/binning/structural** parts (bin indices, split feature, split bin, tree topology must match exactly) but is the wrong frame for floating-point accumulation and transcendental paths.

**How to avoid:**
- **Redefine the oracle as a tiered contract before Phase 1:**
  - *Tier A — exact (bit-identical, must hold on all backends):* bin boundaries, `ValueToBin` results, chosen split feature, chosen split bin/threshold, tree topology, missing-direction, RNG-selected row/feature sets, model-text structural fields.
  - *Tier B — 1e-12 absolute:* CPU-backend leaf outputs and predictions where the reference ran `deterministic=true, force_row_wise=true, num_threads=1` **and** the Rust CPU path uses the identical summation order and the identical scalar `exp`/`log` implementation (see Pitfall 8).
  - *Tier C — relaxed, documented tolerance (e.g. ~1e-6 relative, the `assert_allclose` regime the reference's own tests use):* ROCm-backend predictions and any path through GPU reductions/transcendentals. Pair Tier C with a *structural* check that the **same tree was built** (Tier A), so "numbers slightly differ" never hides "a different model was trained."
- Get this tiering **signed off as a Key Decision** in PROJECT.md; the current single-line "1e-12 on ROCm" will otherwise be cited as a failed requirement at every milestone.
- Where 1e-12 on GPU is genuinely required, the *only* way to get it is to run the numerically-sensitive math (objective gradients, leaf-output division, split-gain) in a **bit-reproducible** form: integer/fixed-point or ordered-`f64` reductions with a *shared scalar transcendental implementation compiled to both CPU and GPU* (see Pitfall 8). Treat that as expensive, scoped work, not a default.

**Warning signs:**
- Acceptance tests that only compare final predictions pass on CPU but you "can't explain" small ROCm diffs.
- A test asserts `< 1e-12` on a logloss/sigmoid path and flickers between pass/fail across GPU driver versions.
- Anyone says "we'll just tighten the GPU kernel later" — the divergence is fundamental, not a tuning bug.

**Phase to address:** **Phase 0 / Foundations (oracle design).** This is the single most important upstream decision; it shapes every later phase's success criteria.

---

### Pitfall 2: Non-bit-exact port of LightGBM's `Random` PRNG (bagging / feature sampling)

**What goes wrong:**
Bagging, feature sub-sampling (`feature_fraction`), GOSS, and DART all draw from LightGBM's custom `Random` class. If the Rust PRNG diverges by a single draw, it selects *different rows or features*, producing a *different tree* — the oracle fails immediately and the failure looks like a "numerical" bug when it is actually a selection bug.

**Why it happens:**
`Random` (in `include/LightGBM/utils/random.h`) has several easy-to-miss details:
- It is a **32-bit LCG with MSVC `rand()` constants**: `x = 214013 * x + 2531011`, with `x` declared `unsigned int` — it **relies on 32-bit unsigned wraparound**. In Rust this must be `u32` with `wrapping_mul`/`wrapping_add`; using `i32` or a wider type changes the stream.
- `RandInt16()` returns `(x >> 16) & 0x7FFF` (15 bits, range 0–32767); `RandInt32()` returns `x & 0x7FFFFFFF`.
- `NextShort`/`NextInt` use C++ `%` on a *signed* `int` result: `RandInt16() % (upper-lower) + lower`. Sign and modulo semantics must match (the values are non-negative here, but reproduce the exact expression).
- `NextFloat()` is **single-precision**: `static_cast<float>(RandInt16()) / 32768.0f`. Doing this in `f64` changes the comparison `NextFloat() < prob` and flips selections.
- `Sample(N, K)` has a **branch**: it uses a Bernoulli/selection-rate loop when `K > 1 && K > (N / log2(K))`, otherwise a **`std::set<int>`-based reservoir** whose *iteration order* (sorted) defines the output order, with a collision-reinsert rule (`if insert fails, insert r`). The branch predicate uses `std::log2(K)` — reproduce it exactly (including integer/float types) or the wrong sampling algorithm runs.
- Default seed is `123456789`; the seedless constructor uses `std::random_device` + `mt19937` (non-reproducible — irrelevant for the oracle, which always seeds).

**How to avoid:**
- Port `Random` as a standalone module **first**, and write a unit test that reproduces a *long* draw sequence (e.g. 100k of `RandInt16`, `RandInt32`, `NextFloat`, `NextInt(a,b)`, and full `Sample(N,K)` outputs for several N,K straddling the branch boundary) captured from the compiled C++ reference. Gate all sampling work behind this test passing.
- Use `u32` with `wrapping_*` arithmetic; mirror `f32` in `NextFloat`; reproduce `Sample`'s set-ordering with a `BTreeSet<i32>` (sorted iteration) and the exact collision rule.
- Verify the **call sequence**: parity of the PRNG values is necessary but not sufficient — the bagging/feature-sampling code must call the PRNG the *same number of times in the same order* (per-iteration reseeding, per-tree vs per-iteration draws). Check how `bagging.hpp`/`goss.hpp`/`sample_strategy.cpp` seed and advance `Random` each iteration.

**Warning signs:**
- Trees match with `bagging_fraction=1.0, feature_fraction=1.0` (no sampling) but diverge the moment any fraction `< 1.0` is set → PRNG or call-order bug, not numeric.
- Selected-index sets differ in *count* but not *values*, or vice versa → branch-predicate or ordering bug in `Sample`.

**Phase to address:** **Phase: core data / utilities (port `Random` standalone).** Must precede the boosting loop, bagging, GOSS, and DART phases.

---

### Pitfall 3: Floating-point summation **order** in histogram and leaf-sum reductions

**What goes wrong:**
`sum_gradients` and `sum_hessians` are reductions over many rows. FP addition is non-associative, so a different accumulation order changes the low bits, which changes split gains, which flips split choices, which forks the tree. This is the dominant "looks correct but trains a different model" failure.

**Why it happens:**
- The C++ reference has **two** summation regimes selected by the `deterministic` flag: a fast parallel `reduction(+:...)` (OpenMP, order = thread/block dependent) and a **deterministic ordered** branch (e.g. `leaf_splits.hpp:180` is the deterministic path; reductions are gated `if (... && !deterministic_)`). The block partitioning is set by LightGBM's own `Threading::For`/`BlockInfo`, **not** by a generic thread pool.
- A Rust port using `rayon` or `iter().sum()` will pick a *different* order than either C++ regime. Using CubeCL plane/tree reductions on GPU picks yet another order.

**How to avoid:**
- **Always compare against the C++ reference built and run with `deterministic=true` and `force_row_wise=true` (or `force_col_wise=true`) and `num_threads=1`.** That pins a single, reproducible summation order to target. (LightGBM docs explicitly recommend `force_*_wise=true` whenever `deterministic=true` to avoid the instability — confirmed in the Parameters docs and [#6320](https://github.com/microsoft/LightGBM/issues/6320).)
- In the Rust **CPU** path, accumulate in the **same sequential order** as the deterministic C++ branch (left-to-right over the data/bin index), in `f64`. Do **not** parallelize the reduction in a way that changes order until parity is locked; only then introduce a tree/pairwise reduction *that reproduces the same result* (e.g. fixed-shape pairwise with identical grouping).
- For the **GPU** path, accept that plane/block reductions will not match the sequential CPU order (Tier C of Pitfall 1). If 1e-12 is mandated there, the reduction must be made order-independent: accumulate gradients/hessians as **integers** (LightGBM already has a quantized-gradient integer path — exact integer `Subtract`) or use a fixed deterministic reduction tree plus `f64`, and validate that specific kernel against the CPU sequential result.

**Warning signs:**
- Parity holds for tiny datasets (few rows, sum order trivially identical) but breaks as row count grows.
- Toggling `num_threads` in the *reference* changes the reference output → you are comparing against a non-deterministic target (fix the reference invocation, not the Rust code).

**Phase to address:** **Phase: histogram + tree learner (CPU first).** Lock CPU summation order before any GPU kernelization phase.

---

### Pitfall 4: The histogram **subtraction trick** — reproducing *which* child is constructed vs derived

**What goes wrong:**
LightGBM builds the histogram for the *smaller* child directly and derives the *larger* child by subtracting from the parent histogram. The subtracted histogram differs in the low FP bits from a directly-constructed one. If the Rust port constructs both children directly (the "obvious clean" approach), every derived-child histogram differs from the reference → different gains → different splits.

**Why it happens:**
`use_subtract` logic in `serial_tree_learner.cpp` (~398-580) and `FeatureHistogram::Subtract` (`feature_histogram.hpp:96-145`) pick the smaller-by-count child to construct and subtract for the other. A faithful port must reproduce the **same choice of which child is constructed** for every split, and apply `Subtract` in the same arithmetic form. The quantized path uses *exact integer* subtraction (bit-packed), which is reproducible; the float path tolerates the accumulated error deliberately.

**How to avoid:**
- Port the subtraction trick **including the smaller-child selection criterion**, not "always construct directly." Add a fixture comparing the derived-child histogram against the C++ derived-child histogram (not against a freshly-constructed one).
- For GPU/int paths, prefer the **integer** histogram representation where the subtract is exact, which also helps GPU reproducibility (ties into Pitfall 3's GPU strategy).

**Warning signs:**
- Histograms match for the directly-constructed child but diverge for its sibling.
- Divergence appears only at deeper tree levels (subtraction error compounds with depth).

**Phase to address:** **Phase: histogram construction / tree learner.** Same phase as Pitfall 3.

---

### Pitfall 5: `BinMapper::ValueToBin`, bin-boundary search, and missing/zero/NaN placement off-by-ones

**What goes wrong:**
Binning is the *input quantization* of the whole algorithm. Any off-by-one in bin-boundary construction or `ValueToBin` puts rows in different bins → different histograms → different everything. This is a **Tier A exact-match** requirement and a frequent silent-divergence source.

**Why it happens:**
`ValueToBin` (`include/LightGBM/bin.h:612-652`) encodes subtle conventions:
- Binary search uses `m = (r + l - 1) / 2` with `value <= bin_upper_bound_[m]` — the `-1` and the `<=` (not `<`) are load-bearing; the off-by-one rounding direction matters for values exactly on a boundary.
- Missing handling differs by `MissingType`: NaN → bin 0 for categorical; → `num_bin_-1` for `MissingType::NaN` numerical; → treated as value `0.0` (then searched) for `MissingType::Zero`/`None`. `+0.0`/`-0.0` and exact-boundary values are edge cases.
- Categorical encoding has its own path (unseen/negative categories, `bin.h:640-650`).
Rust's idiomatic `slice::binary_search` uses different tie-breaking than `(r+l-1)/2` with `<=`, so it will not reproduce boundary placement.

**How to avoid:**
- Port `ValueToBin` and the bin-boundary finder **literally**, preserving `(r + l - 1) / 2` and `<=`. Do not substitute `slice::binary_search` or `partition_point` without proving identical tie-breaking.
- Build a fixture that snapshots, from the reference, the full `bin_upper_bound_` arrays and the bin index for a battery of inputs: NaN (both missing types), `+0.0`, `-0.0`, values exactly equal to each boundary, values just above/below, and out-of-range categoricals.
- Match the dataset **parser** too: text values are parsed by `AtofPrecise`/`fast_double_parser` (`common.h`). Rust's `f64::from_str` may round differently on edge cases, putting a row in a different bin. Verify with adversarial decimal strings or vendor a parity-checked parser.

**Warning signs:**
- Histograms diverge on datasets with missing values, exact-boundary values, or many categoricals, but match on clean dense data.
- A single row's bin index differs and cascades.

**Phase to address:** **Phase: dataset / binning (BinMapper).** This is foundational — must be exact before histograms.

---

### Pitfall 6: Split-gain math constants and arithmetic position (`kEpsilon`, `2*kEpsilon`, leaf-output formula)

**What goes wrong:**
Leaf-output and split-gain formulas add small epsilons (`kEpsilon`, `2*kEpsilon`) and use specific L1/L2/smoothing/max-output adjustments. Getting the *value* right but the *position* wrong (e.g. adding epsilon before vs after a division, or to the wrong hessian term) shifts outputs beyond 1e-12.

**Why it happens:**
`kEpsilon` (from `meta.h`) appears in many places in `feature_histogram.hpp` (e.g. `sum_hessian + 2 * kEpsilon` at line ~172; `sum_left_hessian = kEpsilon`; usages at 265, 489, 514, 571-662). `CalculateSplittedLeafOutput`, `GetLeafGain`, `GetSplitGains`, and `ThresholdL1` are compile-time specialized over `USE_L1`, `USE_MAX_OUTPUT`, `USE_SMOOTHING`, `USE_RAND`, `USE_MC`, etc. The exact branch and the exact arithmetic order determine the bits.

**How to avoid:**
- Copy `kEpsilon` and related constants **verbatim** from `meta.h`; apply them at the identical arithmetic position. Port `ThresholdL1`/`CalculateSplittedLeafOutput`/`GetLeafGain`/`GetSplitGains` as **scalar Rust first**, before any kernelization, and golden-test each against C++ outputs for the same `(sum_grad, sum_hess, config)` inputs.
- Reproduce the compile-time flag matrix as runtime config or const-generics, but **test each active combination** (L1 on/off, smoothing on/off, monotone constraints on/off) — the combinatorial surface is where silent divergence hides.

**Warning signs:**
- Predictions match to ~1e-6 but not 1e-12 on the CPU deterministic path → likely an epsilon-position or formula-order mismatch, not a reduction-order issue.
- Divergence appears only when `lambda_l1`, `min_sum_hessian_in_leaf`, or `path_smooth` are non-default.

**Phase to address:** **Phase: tree learner / split finding.**

---

### Pitfall 7: Default-bin / most-frequent-bin offset and `SKIP_DEFAULT_BIN` scan handling

**What goes wrong:**
Histograms store an `offset` and special-case the most-frequent bin (`default_bin_`, `most_freq_bin_`). The threshold scan **skips the default bin** (`SKIP_DEFAULT_BIN`) and has a "default left" direction. Dropping or mis-ordering this when restructuring the scan loop (especially into a GPU kernel) changes which thresholds are even *considered* → different split.

**Why it happens:**
The offset arithmetic and default-bin skip (`bin.h:180-258`, split functions in `feature_histogram.hpp` ~480-700) are easy to lose when flattening the scan into a parallel kernel that iterates bins uniformly.

**How to avoid:**
- Replicate the offset arithmetic and the default-bin-skip exactly in the scalar CPU scan first; snapshot the *set of candidate thresholds and their gains* per feature/leaf from the reference and compare.
- When kernelizing, preserve the skip as kernel logic (e.g. mask the default bin), and re-validate candidate-threshold parity, not just the winning split.

**Warning signs:**
- Winning split matches on features without a dominant frequent bin but diverges on sparse/one-hot-ish features.

**Phase to address:** **Phase: tree learner / split finding** (and re-checked in the **GPU histogram kernel** phase).

---

### Pitfall 8: Transcendental functions (`exp`, `log`, `pow`, `sigmoid`) diverging across Rust std, C++ std, and GPU intrinsics

**What goes wrong:**
Objectives/metrics (binary logloss, sigmoid, multiclass softmax, xentropy, Poisson/Gamma/Tweedie) call `exp`/`log`/`pow`. Three implementations are in play — Rust `std` (`f64::exp`), C++ `std::exp`/LightGBM's hand-rolled `Pow`, and GPU HIP intrinsics — and **all three can differ by multiple ULP**. The error feeds gradients → scores → next-iteration histograms, compounding over boosting rounds (the mechanism behind [#3372](https://github.com/microsoft/LightGBM/issues/3372)).

**Why it happens:**
There is no IEEE-mandated bit-exact `exp`/`log`; each libm/vendor library rounds differently. GPU math libraries are documented to diverge from host libm by more input-dependent ULP than CPU-to-CPU. LightGBM additionally ships a **recursive hand-rolled `Pow`** (`common.h:248`) and uses a specific `sigmoid` formulation — Rust's `powi`/`powf` won't match the recursive `Pow` bit-for-bit.

**How to avoid:**
- Port LightGBM's **`Pow` and `sigmoid` exactly** (recursive `Pow`, same sigmoid expression and any clamping like `kMinScore`/`kMaxScore`). For `exp`/`log`, on the **CPU path** Rust `f64::exp`/`f64::ln` *usually* matches glibc (often same underlying implementation) but is not guaranteed across platforms — verify per objective against the reference and, if it diverges, vendor a known scalar implementation.
- For the **GPU path**, do **not** rely on HIP intrinsic `exp`/`log` if 1e-12 is required — they will not match. Options: (a) accept Tier C tolerance on GPU objective/metric outputs (recommended default), or (b) compile a **single shared scalar `exp`/`log` implementation** (a Rust softfloat-style or polynomial implementation) into *both* CPU and GPU kernels so both backends produce identical bits. Option (b) is expensive and slow; scope it only if a stakeholder truly requires bit-parity on GPU objective math.
- Decide **where gradients/objectives compute**: if objectives run on CPU and only histograms/splits run on GPU, the transcendental divergence is confined to the CPU path and is far easier to control. The C++ CUDA backend pushes objectives to GPU; the Rust port should *not* by default.

**Warning signs:**
- Iteration-1 predictions match to ~1e-12 but drift monotonically over rounds (compounding transcendental error).
- Binary/multiclass objectives diverge while a plain L2 regression objective (no transcendentals) stays at 1e-12.

**Phase to address:** **Phase: objectives & metrics.** The CPU-vs-GPU objective residency decision should be made in **Phase 0 / architecture**.

---

### Pitfall 9: Subnormal / flush-to-zero (FTZ/DAZ) behavior differing on ROCm

**What goes wrong:**
If the GPU (or a compiler fast-math flag) **flushes subnormals to zero** while the CPU reference computes them, tiny intermediate values (common in hessians, probabilities near 0/1, small gradients) round differently, breaking parity in exactly the regimes the objective math lives in.

**Why it happens:**
GPUs historically flush denormals to zero on single-precision ops for speed; HIP/ROCm and `-ffast-math`-style flags can enable FTZ. `f64` denormal handling is usually present in hardware but is still controllable by compiler/runtime mode. The C++ CPU reference, compiled without fast-math, computes denormals normally.

**How to avoid:**
- Standardize on **`f64`** for all numerically-sensitive accumulation and objective math (the project already mandates f64 accumulation) — `f64` denormals are far less likely to be flushed than `f32`.
- Ensure neither the Rust CPU build nor CubeCL kernels enable fast-math/FTZ for the parity-critical paths. Audit CubeCL's generated HIP for FTZ flags; query/verify denormal handling.
- Add a targeted test feeding subnormal-producing inputs (probabilities extremely close to 0/1, tiny gradients) and compare CPU vs GPU.

**Warning signs:**
- GPU results match CPU except for rows with extreme probabilities or near-zero hessians.
- Enabling any "fast math" CubeCL/compiler option changes results.

**Phase to address:** **Phase: GPU backend bring-up** (ROCm kernel phase); flagged in **Phase 0 / architecture**.

---

### Pitfall 10: Oracle harness comparing against a mis-configured / non-deterministic C++ reference build

**What goes wrong:**
The whole project is validated against the C++ reference. If the reference is built or invoked with different settings than assumed (multi-threaded non-deterministic sums, different `score_t` width, FMA-enabled build, different `force_*_wise`), the "golden" outputs are themselves unstable, and the Rust port is chasing a moving target — or worse, parity "passes" against one reference build and fails against another.

**Why it happens:**
- `score_t`/`label_t` are `float` by default but become `double` under `SCORE_T_USE_DOUBLE`/`LABEL_T_USE_DOUBLE` (`meta.h`). If the reference is built `float` and the Rust port accumulates `f64`, predictions differ structurally. The build-time typedef choice must be known and matched.
- `deterministic`, `force_row_wise`/`force_col_wise`, `num_threads`, and `device_type` must be **pinned** when generating golden outputs; the reference's `auto` row/col-wise choice and multi-thread reductions are non-deterministic across runs.
- The reference's own tests use `assert_allclose` (tolerance), **not** exact equality (`tests/python_package_test/test_consistency.py`) — so "the reference passes its own tests" does not imply 1e-12 stability.
- Compiler/ISA differences (FMA, AVX) make the *same source* produce different bits on different machines ([#6683](https://github.com/microsoft/LightGBM/issues/6683)), so golden files are tied to a specific build environment.

**How to avoid:**
- **Pin and document the reference build**: exact LightGBM version (repo is `4.6.0.99`), compiler + version + flags (note FMA/`-march`), `score_t`/`label_t` width, and the fixed config (`deterministic=true`, `force_row_wise=true`, `num_threads=1`, `device_type=cpu`, fixed `seed`). Capture this as a checked-in manifest so golden files are regenerable and comparisons are reproducible.
- Generate golden snapshots at **multiple granularities** (Tier A/B from Pitfall 1): bin boundaries, per-feature histograms, per-split candidate gains, leaf outputs, final predictions, and the **model text**. Diff at the finest granularity first so a divergence is localized to a subsystem rather than only visible as a wrong final prediction.
- Consider building the reference in a **container** to pin the toolchain; otherwise golden files silently depend on the build host.
- Decide whether the reference's `score_t` is `float` or `double` and make the Rust port match the *reference's* precision for comparison (you can accumulate in f64 internally but must reproduce the reference's stored/used precision where it rounds).

**Warning signs:**
- Re-running the reference golden generator produces different `.pred` files → non-deterministic reference config.
- Parity passes on the dev machine but fails in CI on a different CPU → FMA/ISA-tied golden files.

**Phase to address:** **Phase 0 / oracle harness** (the very first deliverable, per CONCERNS.md "stand up the harness before any cubecl work").

---

### Pitfall 11: CubeCL alpha churn, Plane-API maturity, and ROCm/HIP backend gaps

**What goes wrong:**
CubeCL is **alpha (v0.10.0)** and the README explicitly warns to "expect breaking changes between minor versions" and to pin the version. The Plane API (warp/subgroup ops, the project's mandated path for CUDA warp ops) and the HIP/ROCm backend are under that same alpha umbrella. Building deep, parity-critical kernels against a moving alpha API risks repeated rewrites, and a missing/changed feature (e.g. `plane_sum`, an atomic, or `f64` support on a given runtime) can block a kernel entirely.

**Why it happens:**
- Feature availability is **runtime-queried, not guaranteed**: code must check `client.features().plane.contains(Plane::Ops)` and `feature_enabled(Feature::Type(Elem::Float(FloatKind::F64)))` etc. f64 and atomics are *optional capabilities* that may be absent or slow on a given ROCm device — and **f64 atomic-add** in particular is frequently unsupported, which matters for atomic histogram accumulation.
- `cubecl`/`cubecl-hip` versions move together (cubecl 0.10.0 ↔ cubecl-hip 0.6.0) and `cubecl-hip-sys` has had build-script breaking changes around HIP detection.
- ROCm itself is at 7.x; CubeCL's HIP backend must match the installed ROCm, and version skew breaks the build.

**How to avoid:**
- **Pin exact versions** of `cubecl` and all `cubecl-*` crates in `Cargo.toml` (and `Cargo.lock`); upgrade deliberately as a discrete task, not transitively.
- **Provide CPU and ROCm parity on the same kernels via the `cpu` runtime** so most logic can be validated without a GPU and without warp ops (plane size 1 on CPU). Develop kernels CPU-first, then enable HIP.
- **Capability-gate every kernel**: query `Plane::Ops`, `f64`, and atomic support at startup; if `f64`-atomic histogram accumulation is unsupported on the target ROCm device, fall back to a deterministic non-atomic reduction (which is *also* better for parity — see Pitfall 3).
- Keep the GPU offload boundary narrow initially (histograms only, splits/objectives on CPU), matching the OpenCL reference's "GPU histograms, CPU split decisions" boundary — this both de-risks CubeCL churn and confines numerical divergence.

**Warning signs:**
- A `cargo update` silently bumps a `cubecl-*` crate and kernels stop compiling or change results.
- A kernel works on the CPU runtime but `feature_enabled` returns false for `f64`/atomics on the ROCm device.
- HIP build fails after a ROCm system update (version skew).

**Phase to address:** **Phase: compute backend bring-up.** Version pinning and capability-gating belong in the **architecture/foundations** phase.

---

### Pitfall 12: CPU-runtime-vs-ROCm kernel result divergence within CubeCL itself

**What goes wrong:**
Even with one CubeCL kernel source, the **CPU runtime** and the **HIP runtime** can produce different FP results (different reduction lowering, FMA contraction, intrinsic selection). A kernel validated as "matches the C++ reference" on the CPU runtime may not match on ROCm, and vice versa — so "the kernel is correct" is backend-specific.

**Why it happens:**
CubeCL JIT-compiles the same IR to different targets; reduction trees, plane-op availability (plane size 1 on CPU vs 32/64 on GPU), and FMA usage differ. This is the in-toolchain version of Pitfall 1.

**How to avoid:**
- Treat "CPU runtime parity" and "ROCm parity" as **separate test gates**. Run the oracle on *both* CubeCL backends, not just one.
- For parity-critical kernels, prefer **integer/fixed-point** accumulation (order-independent, backend-independent) over float reductions; this is the most reliable route to CPU==GPU==reference.
- When a float reduction is unavoidable, fix the reduction *shape* (explicit pairwise tree of known structure) rather than relying on `plane_sum`, so CPU and GPU execute the same grouping.

**Warning signs:**
- Same kernel, same input, different output between `cpu` and `hip` runtimes.
- Plane-based reductions match reference on GPU (plane size 64) but not on CPU runtime (plane size 1 falls back to a serial loop with different order).

**Phase to address:** **Phase: GPU kernelization** — every kernel's acceptance must include both CubeCL backends.

---

### Pitfall 13: Categorical split encoding and the categorical threshold bitset

**What goes wrong:**
Categorical features use a *set-membership* split (a bitset of categories going left), not a numeric threshold, with its own gain computation, one-hot vs many-vs-many logic, `max_cat_threshold`, `cat_l2`/`cat_smooth`, and ordering of categories by gradient statistics. Mis-porting the category sort order or the bitset packing produces different splits and an unreadable/incompatible model text.

**Why it happens:**
The categorical path is a distinct branch in `feature_histogram.hpp` with its own constants and a sort of categories by `sum_gradient/sum_hessian`. The model text stores categorical splits as `cat_threshold` bitsets and `cat_boundaries`; serialization must match for model-format compatibility.

**How to avoid:**
- Port the categorical split as a separate, explicitly-tested path; snapshot the chosen category set (bitset) and gain from the reference. Reproduce the category **sort order** and tie-breaking exactly (sort stability matters).
- Validate model-text round-trip for categorical splits specifically (load a C++ categorical model, predict identically; write a model and diff the bitset fields).

**Warning signs:**
- Numeric-only datasets hit 1e-12 but anything with `categorical_feature` diverges.
- Model text differs only in `cat_threshold`/`cat_boundaries` fields.

**Phase to address:** **Phase: categorical features.** (PROJECT lists categorical + monotone constraints together.)

---

### Pitfall 14: PyO3 / numpy binding hazards (copies, contiguity, GIL, dtype, float width)

**What goes wrong:**
The Python bindings must mirror the official `lightgbm` API. Common binding bugs: silent array copies (perf + a chance to change layout), passing a non-contiguous/wrong-dtype array that errors or silently copies, returning slices into Python-owned memory (use-after-free), holding the GIL during long training (no parallelism, UI freezes), and **f32-vs-f64 mismatches** where numpy passes `float32` but the engine expects `float64` (or vice versa), silently changing results vs the C++ path.

**Why it happens:**
- `PyReadonlyArray*::as_slice()` only works on contiguous arrays and errors otherwise; careless code calls `.to_owned()`/forces contiguity, hiding a copy and possibly a dtype cast.
- LightGBM's C API is explicitly dual-dtype (`C_API_DTYPE_FLOAT32`/`FLOAT64`); the Python layer converts. A naive Rust binding that assumes one width will mismatch the reference for the other.
- `Python::allow_threads` is needed to release the GIL during training, but anything touching Python objects can't be inside it.

**How to avoid:**
- Use `rust-numpy` `PyReadonlyArray2<f64>`/`<f32>` and handle **both** dtypes explicitly, matching what the official Python package passes for `data`, `label`, `weight`, `init_score`, `group`. Validate contiguity and dtype, and make any copy/cast **explicit and intentional**, mirroring the reference's conversion semantics so results don't shift.
- Release the GIL with `Python::allow_threads` around the training loop only; never return borrowed views into numpy buffers — return owned arrays.
- Mirror the reference's `float`/`double` choices at the boundary so Python-level outputs match the C++ Python package within the same tolerance.

**Warning signs:**
- Python predictions match C++ for `float64` inputs but diverge for `float32` (or vice versa).
- Training doesn't parallelize from Python (GIL held) or segfaults on returned arrays (dangling view).

**Phase to address:** **Phase: Python bindings** (last, after the core engine + oracle are green).

---

### Pitfall 15: Edition 2024 / latest-crate churn breaking reproducibility of the build itself

**What goes wrong:**
The project pins edition 2024 and "use latest crate versions." Latest-everything on an alpha GPU stack (CubeCL) plus a fresh edition increases the chance of (a) transitive breakage on `cargo update`, (b) MSRV mismatches, and (c) the *reference golden files* and the *Rust build* drifting independently, making "did the model change or did a dependency change?" hard to answer during a parity regression.

**Why it happens:**
Edition 2024 is recent; some crates may lag. CubeCL's alpha cadence means minor bumps break APIs (Pitfall 11). "Latest versions" without a committed `Cargo.lock` makes builds non-reproducible.

**How to avoid:**
- **Commit `Cargo.lock`** (it already exists in the repo per git status) and treat dependency upgrades as discrete, tested tasks — never bundle a dep bump with a parity-sensitive change.
- Keep the GPU stack (`cubecl*`) pinned to exact versions; keep `MSRV`/toolchain pinned (e.g. `rust-toolchain.toml`) so edition-2024 + crate behavior is reproducible across dev and CI.
- When a parity test regresses, first confirm dependencies are unchanged (lockfile diff) before suspecting the algorithm.

**Warning signs:**
- Parity regresses with no source change → a transitive dependency moved.
- CI uses a different toolchain than dev and produces different bits.

**Phase to address:** **Phase 0 / foundations** (lockfile + toolchain pinning policy).

---

## Technical Debt Patterns

| Shortcut | Immediate Benefit | Long-term Cost | When Acceptable |
|----------|-------------------|----------------|-----------------|
| Compare oracle on **final predictions only** (skip bin/histogram/split snapshots) | Fast to set up | Divergence is unlocalizable; "wrong tree" hides behind "small number diff"; debugging takes weeks | Never — multi-granularity snapshots are the core debugging tool |
| Use `rayon`/`iter().sum()` for gradient/hessian reductions | Easy, fast | Different summation order than C++ deterministic branch → split divergence | Never on parity-critical sums until CPU parity is locked, then only with an order-preserving reduction |
| Construct both split children directly (skip the subtraction trick) | Simpler kernel | Derived-child histograms differ in low bits from reference → split divergence | Never if matching reference; OK only if reference is also reconfigured (it can't be) |
| Use Rust `slice::binary_search` for `ValueToBin` | Idiomatic | Different tie-breaking than `(r+l-1)/2`+`<=` → boundary rows mis-binned | Never — port the search literally |
| Use HIP intrinsic `exp`/`log` on GPU | Fast, simple | Multi-ULP divergence compounding over rounds → fails 1e-12 on GPU objective paths | Acceptable only under Tier C (relaxed GPU tolerance) with a structural same-tree check |
| Accumulate histograms with `f64` atomic-add on GPU | Simple parallel histogram | Atomic order is non-deterministic (and f64-atomic often unsupported on ROCm) → non-reproducible | Never for parity; use deterministic/integer accumulation |
| Float (`score_t=float`) accumulation to "match reference defaults" | Matches a float reference build | Loses precision needed for 1e-12; but f64 may *mismatch* a float reference | Match the reference's *actual* typedef; document it |
| Skip GOSS/DART/RF and quantized-grad in v1 | Smaller surface, faster to parity | Must revisit RNG call-order and hessian-change interactions later | Acceptable and recommended (CONCERNS.md agrees) — gate behind config errors |

## Integration Gotchas

| Integration | Common Mistake | Correct Approach |
|-------------|----------------|------------------|
| C++ reference (oracle source) | Build/run with default (non-deterministic) config; unknown `score_t` width | Pin version/compiler/flags/`score_t`; run `deterministic=true, force_row_wise=true, num_threads=1`, fixed seed; checked-in build manifest |
| CubeCL HIP runtime | Assume `f64`/atomics/`plane_sum` always available | Capability-query at startup; fall back to deterministic reductions; pin `cubecl`+`cubecl-hip` versions together |
| CubeCL CPU runtime | Treat CPU-runtime parity as proof of GPU parity | Separate test gate per backend; run oracle on both |
| ROCm system libs | CubeCL-HIP version skew vs installed ROCm (7.x) | Match `cubecl-hip(-sys)` to installed ROCm; pin; containerize if possible |
| numpy via PyO3 | Silent copies / dtype casts / non-contiguous arrays / returning borrowed views | `PyReadonlyArray` with explicit dtype+contiguity handling; return owned arrays; `allow_threads` around training only |
| Dataset text parser | Rust `f64::from_str` rounding ≠ `fast_double_parser`/`AtofPrecise` | Parity-check the parser on adversarial decimals; vendor if needed |
| Model text format | Float formatting/precision differs → model won't round-trip | Match the reference's number formatting exactly; round-trip test load↔predict↔save |

## Performance Traps

| Trap | Symptoms | Prevention | When It Breaks |
|------|----------|------------|----------------|
| Sequential (order-locked) CPU reductions kept for parity | Slow training vs reference | Keep order-locked path as the *reference oracle path*; add a separately-validated fast path that reproduces the same bits | Large datasets, once correctness is proven |
| GPU offload only of histograms, CPU split-finding | GPU underutilized; PCIe transfer dominates | Acceptable for v1 parity; widen offload (split-finding, partition) later as the C++ CUDA backend does | When datasets are large enough that transfer > compute savings |
| Per-iteration full-data histogram without subtraction trick | 2x histogram cost | Implement subtraction trick (also required for parity, Pitfall 4) | Deep trees / many leaves |
| f64 everywhere on GPU for safety | Lower GPU throughput (f64 units fewer) | f64 only on parity-critical accumulation; keep structural/index work in int | Memory-bound histogram kernels at scale |

## Security Mistakes

*(Low relevance for a numerical library, but two real items:)*

| Mistake | Risk | Prevention |
|---------|------|------------|
| Trusting model-text / dataset files as input (parsing untrusted serialized models) | Malformed model text → panic/OOM/UB in the parser | Validate lengths/indices during model-text and dataset parsing; return `thiserror` errors, never index out of bounds; fuzz the parser |
| `unsafe` type-punning of histogram buffers (replacing C++ `reinterpret_cast`) | UB / memory corruption if aliasing float and packed-int views | Use `bytemuck`/explicit audited reinterpretation; never alias one `Vec` as two live types (CONCERNS.md "Fragile Areas") |

## UX Pitfalls

| Pitfall | User Impact | Better Approach |
|---------|-------------|-----------------|
| Mirroring C++ `Log::Fatal`-aborts-process behavior in Rust | Library aborts the host app/Python kernel on a recoverable error | Convert `Log::Fatal` sites to `Result`/`thiserror` at boundaries; `anyhow` at app layer (per PROJECT) |
| Silent default-hyperparameter divergence | User gets different model than C++ for "the same" call | Replicate `config.h` defaults, aliases, and auto-extraction exactly (consider generating Rust config from `config.h`); test default-value parity |
| Python API that subtly differs from official `lightgbm` | Drop-in users get surprises | Mirror the official Python signatures/semantics; reuse the reference's own `test_consistency.py`/`test_engine.py` as conformance |

## "Looks Done But Isn't" Checklist

- [ ] **RNG port:** Often missing the `u32` wraparound, the `f32` `NextFloat`, and the `std::set`-ordered `Sample` branch — verify a 100k-draw sequence and full `Sample(N,K)` outputs match the C++ `Random` across the branch boundary.
- [ ] **Binning:** Often missing exact NaN/`+0`/`-0`/on-boundary placement and `(r+l-1)/2`+`<=` tie-breaking — verify `bin_upper_bound_` arrays and edge-case bin indices match.
- [ ] **Histogram subtraction:** Often missing the *smaller-child selection* — verify the derived child matches the reference's derived child (not a freshly-constructed one).
- [ ] **Split gain:** Often missing exact `kEpsilon` positions and L1/smoothing/monotone branches — verify per-split candidate gains, not just the winner.
- [ ] **Default-bin skip:** Often dropped when flattening the scan to a kernel — verify candidate-threshold set matches.
- [ ] **Objectives:** Often missing exact `Pow`/`sigmoid`/clamping — verify gradients/hessians match per objective; watch compounding drift over rounds.
- [ ] **Oracle reference build:** Often run with default (non-deterministic) config or unknown `score_t` width — verify the reference is `deterministic=true, force_row_wise=true, num_threads=1`, pinned compiler/flags, and the typedef widths are known.
- [ ] **Both CubeCL backends:** Often only the CPU runtime is tested — verify the oracle runs on `cpu` *and* `hip`.
- [ ] **Model text:** Often missing exact float formatting / categorical bitsets — verify load↔predict↔save round-trip against a C++-trained model.
- [ ] **Default hyperparameters:** Often missing aliases/auto-extraction — verify config defaults match `config.h`.
- [ ] **GIL/dtype in Python:** Often holds the GIL during training and assumes one float width — verify `allow_threads` and both f32/f64 inputs.

## Recovery Strategies

| Pitfall | Recovery Cost | Recovery Steps |
|---------|---------------|----------------|
| RNG mismatch discovered after building bagging/GOSS | MEDIUM | Fix `Random`; re-run the PRNG unit test; all sampling-dependent fixtures regenerate; selection-dependent trees re-validate (localized if RNG was a standalone module) |
| 1e-12-on-GPU contract is unachievable on a path | HIGH (political) | Re-tier the contract (Pitfall 1) as a Key Decision; add structural same-tree checks; document relaxed GPU tolerance — do this *early* to avoid late milestone failure |
| Summation-order divergence found late | MEDIUM-HIGH | Revert reductions to order-locked f64; re-lock CPU parity; reintroduce fast path only with proven-identical grouping |
| Reference golden files are non-deterministic | LOW-MEDIUM | Pin reference config/build; regenerate goldens from the manifest; add a "regen is idempotent" CI check |
| CubeCL minor bump broke kernels | LOW | Revert to pinned version via `Cargo.lock`; schedule the upgrade as its own task |
| Binning off-by-one found after histograms built | MEDIUM | Fix `ValueToBin`; all downstream histogram/split/tree fixtures regenerate; broad re-validation but mechanical |

## Pitfall-to-Phase Mapping

| Pitfall | Prevention Phase | Verification |
|---------|------------------|--------------|
| 1. 1e-12 contract unachievable on GPU | Phase 0 (oracle/architecture) | Tiered contract documented as Key Decision; tests assert exact (Tier A) + tolerance (Tier C) appropriately |
| 2. RNG parity | Core utilities (before boosting) | 100k-draw + `Sample(N,K)` sequence matches C++ `Random` |
| 3. FP summation order | Histogram/tree learner (CPU) | CPU sums match `deterministic=true` reference bit-for-bit |
| 4. Histogram subtraction trick | Histogram/tree learner | Derived child matches reference derived child |
| 5. Binning / `ValueToBin` | Dataset/binning (foundational) | `bin_upper_bound_` + edge-case bin indices exact |
| 6. Split-gain constants | Tree learner / split finding | Per-split candidate gains match scalar C++ |
| 7. Default-bin skip | Tree learner + GPU histogram | Candidate-threshold set matches |
| 8. Transcendentals (exp/log/pow) | Objectives & metrics (+ Phase 0 residency decision) | Per-objective gradients match; drift bounded over rounds |
| 9. Subnormal/FTZ on ROCm | GPU bring-up (+ Phase 0) | Subnormal-input CPU vs GPU comparison |
| 10. Mis-configured reference | Phase 0 (oracle harness, first deliverable) | Reference build manifest; regen idempotent; multi-granularity goldens |
| 11. CubeCL alpha churn / capability gaps | Compute backend bring-up | Pinned versions; capability queries; CPU-first kernels |
| 12. CPU-runtime vs ROCm divergence | GPU kernelization | Oracle green on both CubeCL backends |
| 13. Categorical split encoding | Categorical features | Category bitset + gain + model round-trip match |
| 14. PyO3/numpy hazards | Python bindings (last) | f32 and f64 inputs match; `allow_threads`; owned returns |
| 15. Edition 2024 / crate churn | Phase 0 (foundations) | `Cargo.lock` + toolchain pinned; lockfile-diff before suspecting algorithm |

## Sources

- LightGBM `include/LightGBM/utils/random.h` (in-repo) — exact `Random` LCG, `Sample`, `NextFloat` semantics (HIGH).
- LightGBM `include/LightGBM/bin.h`, `feature_histogram.hpp`, `meta.h` (in-repo, via `.planning/codebase/CONCERNS.md` + `CONVENTIONS.md`) — binning, subtraction trick, `kEpsilon`, `deterministic` branches, `score_t`/`label_t` typedefs (HIGH).
- LightGBM issue [#6683 — deterministic flags don't guarantee same result on different machines](https://github.com/microsoft/LightGBM/issues/6683) (HIGH; explicit maintainer statement on cross-machine/compiler non-reproducibility).
- LightGBM issue [#3372 — instability caused by floating point errors](https://github.com/microsoft/LightGBM/issues/3372) (HIGH; compounding FP error over boosting rounds).
- LightGBM issue [#6320 — how to make output deterministic](https://github.com/microsoft/LightGBM/issues/6320) and [Parameters docs](https://lightgbm.readthedocs.io/en/latest/Parameters.html) — `deterministic`, `force_row_wise`/`force_col_wise` guidance (HIGH).
- CubeCL repo/README and Context7 docs ([/tracel-ai/cubecl](https://github.com/tracel-ai/cubecl)) — alpha v0.10.0, "expect breaking changes between minor versions," Plane API, CPU runtime (plane size 1), runtime feature/capability queries (`Plane::Ops`, `Feature::Type(Elem::Float(...))`) (MEDIUM-HIGH; alpha docs evolving).
- [cubecl-hip / cubecl-hip-sys releases](https://github.com/tracel-ai/cubecl-hip-sys/releases) — HIP build-script breaking changes, version coupling (MEDIUM).
- GPU vs CPU libm precision: LLVM GPU math profiling ([blog.llvm.org GSoC 2025](https://blog.llvm.org/posts/2025-08-29-gsoc-profiling-and-testing-math-functions-on-gpus/)), [C-math-on-GPUs precision study (ACM)](https://dl.acm.org/doi/fullHtml/10.1145/3624062.3624166), [FP non-associativity & reproducibility in HPC/DL (arXiv 2408.05148)](https://arxiv.org/pdf/2408.05148) — multi-ULP transcendental divergence, reduction non-associativity (MEDIUM).
- Denormal/FTZ behavior: [NVIDIA CUDA Pro Tip: Flush Denormals](https://developer.nvidia.com/blog/cuda-pro-tip-flush-denormals-confidence/), [AMD matrix cores / ROCm blog](https://rocm.blogs.amd.com/software-tools-optimization/matrix-cores/README.html) (MEDIUM).
- PyO3 / rust-numpy: [rust-numpy](https://github.com/PyO3/rust-numpy), [rust-numpy docs](https://pyo3.github.io/rust-numpy/) — contiguity/`as_slice`, zero-copy, `allow_threads` GIL release (HIGH).
- `.planning/codebase/{CONCERNS,CONVENTIONS,TESTING}.md` (in-repo analyses) — numerical-fidelity hazards, OpenMP reduction sites, oracle strategy, fragile C++ idioms (HIGH).

---
*Pitfalls research for: pure-Rust LightGBM port on CubeCL (CPU + ROCm), 1e-12 oracle*
*Researched: 2026-06-05*
