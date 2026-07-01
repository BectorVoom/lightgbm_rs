# Phase 20: On-Device Score Updater & Metrics - Context

**Gathered:** 2026-07-02
**Status:** Ready for planning

<domain>
## Phase Boundary

Complete the boosting-layer device path by making the cumulative per-row score
**resident on device across the whole train** and evaluating the **12 CUDA-supported
pointwise regression/binary metrics on device** — and, per the user's explicit
re-scoping decision (D-01), **pulling Phase 21's on-device grow loop (ODL-18/ODL-19)
forward** so the resident score is fed by an end-to-end on-device driver this phase.

**Delivers (ODL-16, ODL-17 — plus ODL-18/ODL-19 pulled forward per D-01):**
- **Score updater (§11, ODL-16):** resident `double* cuda_score_` (per tree at
  `offset = num_data·tree_id`); whole-array scalar `AddScoreConstantKernel`
  (`score[i] += val` — init score / no-split single-leaf) and
  `MultiplyScoreConstantKernel` (`score[i] *= val` — shrinkage / DART rescale); the
  per-leaf `AddScore` delegates to the Phase-18 §9/§10 `AddPredictionToScore` kernels;
  a **host-mirror toggle** (`CopyFromCUDADeviceToHost` when `boosting_on_cuda_` is
  false) for non-resident consumers.
- **Pointwise metrics (§12, §12.1, ODL-17):** `EvalKernel<CUDA_METRIC, USE_WEIGHTS>`
  (one thread/row, `NUM_DATA_PER_EVAL_THREAD=1024`, `__shared__ double[32]`,
  `ShuffleReduceSum<double>` per block) + host global fold
  (`ShuffleReduceSumGlobal<double,double>`) → scalar `sum_loss` / `sum_weight`, over
  the **exactly 12** device-supported losses: RMSE, L2, L1, Quantile, Huber, Fair,
  Poisson, MAPE, Gamma, Gamma-deviance, Tweedie, Binary-logloss. Each supplies a
  `MetricOnPointCUDA(label, double score, double param)`; regression returns
  `AverageLoss` (RMSE applies the `sqrt`), binary returns `Σloss/Σweight`.
- **ConvertOutput composed into Eval (D-04):** the §12 Eval flow runs the Phase-19
  device `ConvertOutputCUDA` inverse-link into a `score_convert_buffer_` **then**
  `EvalKernel`, end-to-end, this phase.
- **Full on-device resident loop (D-01, pulled forward):** `cuda_score_` never leaves
  device across the train, fed by on-device grow; `on_device_growth_supported()` → **true**
  this phase (was held false through Phases 14–19); the single-GPU driver runs the full
  per-leaf grow loop and reconstitutes into `(Tree, DataPartition)`.

Everything **additive** and gated by `LGBM_CUDA_ON_DEVICE`; CPU / ROCm / existing-host-CUDA
paths stay **byte-unchanged** with the env unset, and the hard merge gate stays green.
Anchored to the **cubecl-cpu f64 fold** (never GPU-vs-GPU, def-f8u-01).

**Explicitly NOT in this phase:**
- **CUDA-unsupported metrics** (AUC / AUC-mu / NDCG / MAP / multiclass `multi_error` +
  `multi_logloss` / cross-entropy `xentropy` + `xentlambda` / KullbackLeibler) →
  **honest host fallback** via the metric-supported discriminator (D-05, SC #3).
- **On-device categorical splits** (bitset construction / categorical eval / partition /
  `SplitCategorical`) → remain **Phase 22**.
- **Perf-validation / default-ON rollout DoD** (Kaggle A/B, `device_launches` +
  wall-clock ratio, flipping the CUDA default) → remain **Phase 23**.

**Roadmap implication (see D-01):** ODL-18/ODL-19 move from Phase 21 into Phase 20;
**Phase 21 is thereby reduced to a hardening/slack phase** (or folds into 22/23). The
ROADMAP.md Phase 20/21 entries should be re-scoped (`/gsd-phase`) to reflect this before
planning — flagged, not yet applied.

</domain>

<decisions>
## Implementation Decisions

### Score-residency depth & the Phase 20/21 boundary (ODL-16, SC #1 — Decision A)
- **D-01: Pull Phase 21's on-device grow (ODL-18/ODL-19) FORWARD into Phase 20 — full
  end-to-end resident loop.** The user deliberately chose the aggressive path: the
  cumulative `cuda_score_` stays resident on device across the entire train, fed by an
  **on-device grow loop**, so `on_device_growth_supported()` flips to **true** this phase
  (it was held false through Phases 14–19). This absorbs Phase 21's single-GPU-driver
  scope and its STRUCTURE-bit-exact gate into Phase 20, making this a large, higher-risk
  phase; **Phase 21 becomes categorical/hardening slack.** *(The conservative reading —
  resident score buffer + per-tree device `AddPredictionToScore` over host-grown trees,
  `on_device_growth_supported()` staying false, keeping the roadmap boundary intact — was
  explicitly considered and declined in favor of the full loop.)*
  - **Downstream must honor the Phase-21 success criteria as part of Phase 20:** driver
    runs root init → build/subtract → best-split → tree split → partition repeated up to
    `num_leaves−1` (break on `best_leaf == −1`), continuous-feature path as the proving
    slice (§6, §16); grown tree **STRUCTURE bit-exact** to the cpu f64 anchor (tie-aware
    `default_left`), leaf values within ~1e-5; f32 + u64 fixed-point build, **no f64
    per-row hot loops** (§17).

### Score-updater kernels (§11, ODL-16 — Decision A)
- **D-02: `AddScoreConstantKernel` / `MultiplyScoreConstantKernel` are whole-array scalar
  ops over resident `double* cuda_score_`** (per tree at `offset = num_data·tree_id`); the
  per-leaf `AddScore` delegates to the Phase-18 §9/§10 `AddPredictionToScore` kernels; a
  **host-mirror toggle** (`CopyFromCUDADeviceToHost` when `boosting_on_cuda_` is false)
  keeps the host `score_` vector in sync for non-resident consumers. `double*` score
  accumulator is reference-blessed f64 (§11/§17) — not a per-row hot loop violation.

### Metric anchor strategy (§12, ODL-17 — Decision B)
- **D-03: Anchor ALL 12 device metric kernels directly to real compiled-`lib_lightgbm`
  goldens.** The captures already exist in `crates/oracle-harness/tests/fixtures/metric/`
  (huber, poisson, quantile, fair, gamma_deviance, …) with `tests/metric_parity.rs` — so
  capture cost is near-zero and this buys genuine **reference fidelity** rather than the
  weaker retranscription-agrees-with-transcription proof
  (`on-device-kernel-goldens-are-retranscriptions`). *(The Phase-19-D-01-style cpu-f64
  primary + per-family C++ cross-check was declined — real goldens are already on disk for
  metrics, so use them for all 12.)*

### ConvertOutput at the metric boundary (§12 Eval flow — Decision C)
- **D-04: Compose the Phase-19 device `ConvertOutputCUDA` inverse-link into the Eval flow
  THIS phase.** Allocate `score_convert_buffer_`, run the inverse-link (sigmoid / softmax /
  Poisson `exp` / `sign·x²` / pass-through) into it, then `EvalKernel` end-to-end,
  anchor-pinned. *(Leaving `EvalKernel` to consume pre-converted scores and deferring the
  compose to Phase 21 was declined — consistent with the D-01 decision to build the full
  path now.)*

### Unsupported-metric fallback discriminator (§12, SC #3 — Decision D)
- **D-05: Express the supported/unsupported boundary as a metric-supported discriminator**
  (mirroring `on_device_growth_supported`) that device-evaluates exactly the 12 pointwise
  losses and routes **AUC / AUC-mu / NDCG / MAP / multiclass (`multi_error`,
  `multi_logloss`) / cross-entropy (`xentropy`, `xentlambda`) / KullbackLeibler** to the
  host — matching the reference's `Metric::CreateMetric` `#ifdef USE_CUDA` branch (returns
  the CPU class even on CUDA; there is no CUDA rank/AUC/multiclass metric file).
  - **Load-bearing asymmetry:** `MAPE / Gamma / Gamma-deviance / Tweedie` **metrics ARE
    device-supported** even though their **objectives fell back to host** in Phase 19
    (SC #5 / Phase-19 deferred). Objective-support and metric-support are independent
    lists — do not conflate.

### Parity gate (SC #4 + pulled-forward SC #2 — Decision E)
- **D-06: Three-layer parity gate this phase:**
  1. **Kernel-level:** constant-op kernels + `EvalKernel`/reduction anchored to the cpu f64
     fold (tie-aware / ~1e-6–1e-5 envelope; f32-vs-f64 accumulation residual documented).
  2. **Resident-score A/B:** after a full multi-iteration run with `LGBM_CUDA_ON_DEVICE=1`,
     assert the resident `cuda_score_` matches the host `score_` vector (bit-for-bit where
     the algorithm permits, else within the f32 envelope) — proves residency correctness.
  3. **Structure gate (from pulled-forward ODL-18):** grown tree **STRUCTURE bit-exact** to
     the cpu f64 anchor (tie-aware `default_left`), leaf values within ~1e-5. Never compare
     two nondeterministic GPU f32 paths to each other.

### Carried forward from Phases 14–19 (NOT re-litigated — hard discipline)
- **D-07:** Anchor every numeric output to the **cubecl-cpu f64 fold**; **never GPU-vs-GPU**
  (def-f8u-01). Atomic-ordering nondeterminism (e.g. the two-stage reduction global fold) is
  the documented f32-vs-f64 residual the ROCm gate tolerates — pin to the f64 anchor with
  tie-aware / envelope asserts.
- **D-08:** **NO f64 per-row hot loops** in new kernels (5.4× consumer-NVIDIA f64 regression,
  spike-052). The `double* cuda_score_` accumulator and the `double point_metric` /
  `double param` in `MetricOnPointCUDA` ARE reference-blessed f64 (§11/§12/§17) — keep those;
  the prohibition is on gratuitous f64 in per-row grow/build hot loops.
- **D-09:** `LGBM_CUDA_ON_DEVICE` **OFF by default**; CPU / ROCm / existing-host-CUDA paths
  **byte-unchanged**; full merge gate green and unchanged (ODL-19 — hard merge gate). Wiring
  the `boosting_on_cuda_` seam is permitted only behind this gate.
- **D-10:** **Reuse existing primitives / kernels — do NOT rebuild:** Phase-14
  `ShuffleReduceSum` + global fold (`primitives.rs`), Phase-18 `AddPredictionToScore`
  (§9/§10, `predict.rs`) + data-partition/tree kernels, Phase-19 device objective
  `ConvertOutputCUDA` / `GetGradients` / `BoostFromScore`. The score/metric/driver code
  COMPOSES these.
- **D-11:** **Pre-allocate scratch ONCE outside the hot loop** (`reduce_block_buffer`,
  `score_convert_buffer_`, per-block reduction partials) — no per-call in-kernel device alloc
  (Phase-17 D-11 / Phase-18 D-15 / Phase-19 D-09 pattern).

### Claude's Discretion
- Whether `EvalKernel` is one comptime-generic `#[cube]` (branch on a metric enum via the
  `CUDA_METRIC` template analog) or 12 concrete kernels — parity-neutral as long as each
  `MetricOnPointCUDA` matches the §12.1 table.
- The exact code shape of the metric-supported discriminator (enum method vs match table) —
  parity-neutral; only the supported/unsupported SETS (D-05) are locked.
- Module placement — likely a `score_updater.rs` / `metric.rs` (or `metric_pointwise.rs`) in
  `crates/lgbm-compute/src/kernels/`, plus the boosting-layer `boosting_on_cuda_` wiring in
  `lgbm-boosting`.
- Block/geometry constants (`NUM_DATA_PER_EVAL_THREAD=1024`, `num_threads_per_block_`) — start
  from the faithful C++ constants; APU-aware autotune is a deferred perf option (parity-neutral).
- The on-device driver's `Handle` in-place-alias vs ping-pong double-buffer for the data→leaf
  map, and batched `client.read(vec![h])` readback semantics — verify at plan time
  (flagged in the ROADMAP Phase-21 notes, now Phase-20's concern per D-01).

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Port-source design reference (READ FIRST)
- `docs/cuda-kernel-design.md` §11 — **Score Updater** (`cuda_score_updater.cu`):
  `AddScoreConstantKernel` / `MultiplyScoreConstantKernel`, resident `double* cuda_score_`
  at `offset = num_data·tree_id`, the `boosting_on_cuda_` host-mirror `CopyFromCUDADeviceToHost`.
- `docs/cuda-kernel-design.md` §12 + §12.1 — **Metrics** (`cuda_pointwise_metric.cu`):
  `EvalKernel<CUDA_METRIC, USE_WEIGHTS>`, two-stage `ShuffleReduceSum` + global fold, the
  **12-row `MetricOnPointCUDA` table** (param / point-loss / final), the Eval flow
  (`ConvertOutputCUDA` → `LaunchEvalKernel` → `AverageLoss` / `Σloss/Σweight`), and the
  explicit CUDA-supported vs `#ifdef USE_CUDA`-falls-back-to-CPU metric split.
- `docs/cuda-kernel-design.md` §6 + §16 — **End-to-end sequencing + grow loop** (needed
  because D-01 pulls the on-device driver forward): root init → build/subtract → best-split
  → tree split → partition, and where `Score Updater` / `Metric.Eval` sit in the iteration.
- `docs/cuda-kernel-design.md` §17 — **Port considerations**: `double` score accumulator
  and `double param`/`point_metric` are load-bearing f64; no f64 per-row grow/build hot loops.
- `.planning/REFERENCE_MANIFEST.md` — C++ port-source map + the 12-supported /
  AUC-NDCG-MAP-multiclass-xentropy-unsupported metric boundary.

### Requirements & roadmap
- `.planning/REQUIREMENTS.md` — ODL-16 (score update), ODL-17 (pointwise metrics), and
  **ODL-18 / ODL-19** (on-device driver + parity gate, pulled forward per D-01).
- `.planning/ROADMAP.md` — Phase 20 + Phase 21 entries; **re-scope both (`/gsd-phase`)** to
  reflect ODL-18/19 moving into Phase 20 before planning (flagged, not yet applied).

### Existing code to reuse / anchor against (already in git — DO NOT rebuild)
- `crates/lgbm-boosting/src/score_updater.rs` — the host score-updater
  (`add_prediction_to_score` scatter) the resident device path replaces; the mirror target.
- `crates/lgbm-boosting/src/gbdt.rs` — the `boosting_on_cuda_` seam / grow-loop call sites
  the resident loop wires into (D-01).
- `crates/lgbm-metric/src/{regression,binary,lib}.rs` — host f64 metric transcription
  (the secondary anchor / behavioral reference; real C++ goldens are primary per D-03).
- `crates/oracle-harness/tests/fixtures/metric/*.txt` + `tests/metric_parity.rs` — the real
  compiled-`lib_lightgbm` metric goldens (D-03 anchor landing point).
- `crates/lgbm-compute/src/kernels/primitives.rs` — `ShuffleReduceSum` + global fold (D-10).
- `crates/lgbm-compute/src/kernels/predict.rs` — Phase-18 `AddPredictionToScore` §9/§10
  kernels the per-leaf `AddScore` delegates to (D-02, D-10).
- Phase-19 device objective kernels in `lgbm-compute` — `ConvertOutputCUDA` inverse-link
  composed into the Eval flow (D-04, D-10).
- `crates/lgbm-compute/src/lib.rs` — `Backend` seam; `on_device_growth_supported()` flips
  to **true** this phase (D-01).

### Prior-phase context (discipline carried forward)
- `.planning/phases/19-on-device-objectives/19-CONTEXT.md` — objective kernels + device
  `ConvertOutput` (composed here, D-04), the anchor / never-GPU-vs-GPU / no-f64-hot-loop /
  pre-allocate-once discipline (D-07..D-11), the objective-vs-metric support asymmetry (D-05).
- `.planning/phases/18-on-device-data-partition-tree-mutation-prediction/18-CONTEXT.md` —
  device `(Tree, DataPartition)` + `AddPredictionToScore` the driver + score updater consume.
- `.planning/phases/14-foundation-shared-device-primitives-device-structs-rng/14-CONTEXT.md`
  — the reduction primitives the metric two-stage fold composes.

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- **`crates/oracle-harness/tests/fixtures/metric/`** — real compiled-`lib_lightgbm` metric
  score captures already on disk (huber, poisson, quantile, fair, gamma_deviance, cross_entropy,
  average_precision, multi_error, kullback_leibler, …) + `metric_parity.rs`. Directly the D-03
  anchor — genuine reference fidelity at near-zero capture cost.
- **`lgbm-metric` host implementations** (`regression.rs`, `binary.rs`) — the exact
  `MetricOnPointCUDA`-equivalent per-metric math to transcribe to `#[cube]`; the secondary
  behavioral reference.
- **`primitives.rs` `ShuffleReduceSum` + global fold** — the metric two-stage reduction; the
  `BoostFromScore`-style folds. Do not rebuild (D-10).
- **`predict.rs` Phase-18 `AddPredictionToScore`** — the per-leaf `AddScore` delegate (D-02).
- **Phase-19 device `ConvertOutputCUDA`** — composed into the Eval inverse-link path (D-04).
- **`score_updater.rs` / `gbdt.rs` boosting seam** — the resident-loop wiring target (D-01).

### Established Patterns
- **Anchor to cpu f64 fold, never GPU-vs-GPU; real C++ goldens where cheap** (def-f8u-01,
  Phase-19 D-01) — D-03, D-07.
- **`double` accumulator / `double param` are reference-blessed f64; no f64 per-row grow/build
  hot loops** (spike-052, §17) — D-08.
- **Pre-allocate scratch once; env OFF by default; byte-unchanged; merge gate green** — D-09, D-11.
- **Metric-supported / growth-supported discriminator gates host fallback** (mirrors
  `on_device_growth_supported`) — D-05.
- **Template-flag → CubeCL comptime** (`<CUDA_METRIC, USE_WEIGHTS>`) — EvalKernel fan-out.

### Integration Points
- Score-updater + metric kernels live in `lgbm-compute` (`kernels/`), composing Phase-14
  reductions + Phase-18 `AddPredictionToScore` + Phase-19 `ConvertOutput`. The
  `boosting_on_cuda_` resident loop wires into `lgbm-boosting` (`gbdt.rs` / `score_updater.rs`)
  behind `LGBM_CUDA_ON_DEVICE`, flipping `on_device_growth_supported()` true (D-01). Anchored
  to `lgbm-metric` f64 + real `lib_lightgbm` metric goldens via `oracle-harness`. Default host
  path byte-unchanged with the env unset (D-09).

</code_context>

<specifics>
## Specific Ideas

- **User deliberately re-scoped the roadmap (D-01):** given the choice, the user pulled Phase
  21's full on-device grow loop (ODL-18/19) into Phase 20 rather than the conservative
  resident-score-over-host-grown-trees reading. Downstream planning must treat Phase 20 as the
  end-to-end driver phase (STRUCTURE-bit-exact gate included) and expect Phase 21 to shrink to
  categorical/hardening. Re-scope ROADMAP.md before planning.
- **Real metric goldens already exist** — D-03 leverages `oracle-harness/fixtures/metric/`
  rather than manufacturing new transcription anchors; this is the one place in the on-device
  chain where genuine reference fidelity is essentially free.
- **The objective/metric support lists are independent** — MAPE/Gamma/Gamma-deviance/Tweedie
  metrics are device-supported even though those objectives fell back to host in Phase 19. The
  discriminator (D-05) must key on the *metric*, not reuse the objective-support list.
- **`double` is not banned** — the score accumulator and `MetricOnPointCUDA` point-loss/param
  are reference-blessed f64 (§11/§12/§17); the no-f64 rule targets per-row grow/build hot loops.
- **Eval runs ConvertOutput first** — the §12 flow is inverse-link → EvalKernel; the device
  ConvertOutput (Phase 19) is composed into that path this phase (D-04).

</specifics>

<deferred>
## Deferred Ideas

- **On-device categorical splits** (bitset construction, categorical eval, categorical
  partition, `SplitCategorical`) → **Phase 22** (unchanged).
- **Perf-validation + default-ON rollout DoD** (Kaggle A/B, `device_launches` + wall-clock
  ratio, flipping the CUDA default) → **Phase 23** (unchanged).
- **CUDA-unsupported metrics** (AUC / AUC-mu / NDCG / MAP / multiclass / xentropy / KL) →
  **permanent host fallback**, never ported (SC #3, D-05).
- **APU-aware autotune of the EvalKernel / grow-loop block geometry** → deferred perf option
  (parity-neutral, Claude's Discretion).
- **Phase 21 re-scope** — with ODL-18/19 pulled into Phase 20, Phase 21 should be re-cut to
  hardening/slack (or its remaining work folded into 22/23) via `/gsd-phase`.

### Reviewed Todos (not folded)
- **Profile the GPU training loop on large data (stage attribution)** — GPU perf profiling;
  belongs to the Phase-23 perf-validation DoD, not this parity-gated phase.
- **Spike — per-phase A/B vs C++ at low rows (localize the fixed overhead)** — perf spike;
  Phase 23.
- **Spike — find the GPU-vs-CPU crossover (sweep dataset sizes upward)** — perf spike; Phase 23.
- **Establish a large-data benchmark fixture (step zero for GPU profiling)** — perf fixture;
  Phase 23. *(All four are speed-oriented; Phase 20's gate is parity, not speed.)*

</deferred>

---

*Phase: 20-on-device-score-updater-metrics*
*Context gathered: 2026-07-02*
