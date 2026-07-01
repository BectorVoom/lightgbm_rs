# Phase 20: On-Device Score Updater & Metrics - Pattern Map

**Mapped:** 2026-07-02
**Files analyzed:** 9 (3 new modules, 4 edits, 2 test edits + 1 capture-script edit)
**Analogs found:** 9 / 9 (all in-tree; this phase COMPOSES Phase 14–19 kernels)

> This is a CubeCL on-device **composition** phase, not a new-numerics phase. Every
> "new" file has a strong in-tree analog and RESEARCH.md already pinned them with
> file:line refs. Excerpts below are verified against source this session.

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|-------------------|------|-----------|----------------|---------------|
| `crates/lgbm-compute/src/kernels/score_updater.rs` (NEW) | kernel | transform (elementwise scalar op) | `kernels/objective_regression.rs` `convert_body` (elementwise `#[cube]`) + `kernels/predict.rs:212` (per-leaf delegate) | exact |
| `crates/lgbm-compute/src/kernels/metric_pointwise.rs` (NEW) | kernel | batch map-reduce (one thread/row → two-stage fold) | `objective_regression.rs:356` comptime dispatch + `primitives.rs:784` `reduce_sum_f64_on` + `lgbm-metric/regression.rs:242` loss math | exact (split across 3 analogs) |
| `crates/lgbm-compute/src/device_metric.rs` (NEW) | discriminator (utility) | classify (name → bool) | `device_objective.rs:114` `device_objective_supported` + `DeviceObjectiveKind` enum | exact |
| `crates/lgbm-compute/src/lib.rs` (EDIT) | driver / backend seam | event-driven orchestration (per-leaf grow loop) | the dormant seam itself (`lib.rs:1241,1284`) + Phase-16/17/18 kernels sequenced | role-match (self-activation) |
| `crates/lgbm-boosting/src/score_updater.rs` (EDIT) | service (host) | accumulate + mirror toggle | itself (`add_constant`/`multiply_score`/`add_tree_train_path`) | exact (self-extend) |
| `crates/lgbm-boosting/src/gbdt.rs` (EDIT) | boosting driver | orchestration (`boosting_on_cuda_` seam) | existing GBDT TrainOneIter score/eval call sites | role-match |
| `crates/oracle-harness/tests/metric_parity.rs` (EDIT) | test (parity) | replay vs golden | itself (capture-gated SKIP replay idiom) | exact (self-extend) |
| `crates/oracle-harness/tests/learner_parity.rs` (EDIT) | test (parity) | structure diff vs anchor | `assert_on_device_tree_matches_cpu_anchor:2185` | exact (activate cell) |
| `xtask/py/metric_oracle_capture.py` (EDIT) | tooling (capture) | file-I/O batch | itself (`train_and_capture`) | exact (self-extend) |

---

## Pattern Assignments

### `kernels/score_updater.rs` (kernel, elementwise transform) — §11, ODL-16

**Analog A (kernel shape):** `crates/lgbm-compute/src/kernels/objective_regression.rs:356-415` — the elementwise `#[cube]` body + `launch_unchecked` wrapper + host launcher is the exact skeleton for `AddScoreConstant` / `MultiplyScoreConstant`.

**Elementwise `#[cube]` + launch pattern** (`objective_regression.rs:356-415`, adapt to scalar-add):
```rust
#[cube]
#[allow(unused_assignments)]
fn convert_body<F: Float>(input: &Array<F>, out: &mut Array<F>, #[comptime] mode: u32) {
    let i = ABSOLUTE_POS;
    if i < input.len() { /* elementwise op */ out[i] = y; }
}

#[cube(launch_unchecked)]
fn convert_kernel_f64(input: &Array<f64>, out: &mut Array<f64>, #[comptime] mode: u32) {
    convert_body::<f64>(input, out, mode);
}

pub fn convert_output_on<R: cubecl::Runtime>(client: &ComputeClient<R>, input: &[f64], mode: u32)
  -> Result<Vec<f64>, ComputeError> {
    // create_from_slice → empty → launch_unchecked (bounds-guarded) → read_one_unchecked
    let cube_dim = 256u32;
    let cube_count = (n as u32).div_ceil(cube_dim);
    unsafe { convert_kernel_f64::launch_unchecked(client, CubeCount::Static(cube_count,1,1),
        CubeDim::new_1d(cube_dim), ArrayArg::from_raw_parts(h_in, n),
        ArrayArg::from_raw_parts(h_out.clone(), n), mode); }
    let bytes = client.read_one_unchecked(h_out);
    Ok(f64::from_bytes(&bytes).to_vec())
}
```
The two new kernels are the RESEARCH.md-provided shape — `score[offset + i] += val` / `*= val` over a resident `Array<f64>`, with `offset = num_data * tree_id`. **f64 buffer is reference-blessed** (D-02/D-08); this is NOT a per-row hot-loop violation.

**Analog B (per-leaf delegate — D-02, do NOT rebuild):** `crates/lgbm-compute/src/kernels/predict.rs:212` `add_prediction_to_score_on_device` is the per-leaf `AddScore` the training-path score update delegates to. Signature to compose against:
```rust
pub fn add_prediction_to_score_on_device<R: cubecl::Runtime>(
    client: &ComputeClient<R>, tree: &PredictTree<'_>, rows: &[u32], num_rows: usize,
    num_features: usize, bit_type: u32, num_data: usize, used_indices: Option<&[u32]>,
) -> Result<Vec<f64>, ComputeError>   // returns num_data-length f64 raw-margin accumulator (init 0)
```
Note `predict.rs:263-265` already sizes the f64 score accumulator ONCE (D-11 pre-allocate-once precedent).

**Host mirror reference (D-02 toggle target):** `crates/lgbm-boosting/src/score_updater.rs` — the resident device path mirrors into this f64 host `score_` vector when `boosting_on_cuda_` is false. See the score_updater edit section below.

---

### `kernels/metric_pointwise.rs` (kernel, map-reduce) — §12/§12.1, ODL-17

**Analog A (comptime-generic dispatch — the recommended `EvalKernel<CUDA_METRIC,USE_WEIGHTS>` shape):** `objective_regression.rs:356-372` (`convert_body` comptime mode chain) AND `objective_regression.rs:93-129` (`grad_hess_body` — the `#[comptime] objective_tag` + `#[comptime] use_weight` dual-template that is the closest twin to `<metric, weights>`):
```rust
// Source: objective_regression.rs:93-118 — dual comptime template (tag + use_weight)
#[cube]
#[allow(unused_assignments, clippy::too_many_arguments)]
fn grad_hess_body<F: Float>(
    scores: &Array<F>, labels: &Array<F>, weights: &Array<F>,
    grad: &mut Array<F>, hess: &mut Array<F>, param: F,
    #[comptime] objective_tag: u32, #[comptime] use_weight: bool,
) {
    let i = ABSOLUTE_POS;
    if i < scores.len() {
        let diff = scores[i] - labels[i];
        let mut g = F::new(0.0);
        if objective_tag == TAG_L2 { g = diff; }
        else if objective_tag == TAG_L1 { g = sign_f::<F>(diff); }
        // ... one comptime arm per objective; folds to straight-line code
    }
}
```
`metric_on_point<F>(label, score, param, #[comptime] metric: u32) -> F` follows this exactly — one comptime branch per §12.1 row (see the transcribed table below). Reuse the branchless `sign_f::<F>` helper (`objective_regression.rs:80-84`) for any sign needs.

**Existing tag/mode constant convention to mirror** (`objective_regression.rs:50-71`): `pub const TAG_L2: u32 = 0; …` and `CONVERT_PASSTHROUGH/SQRT_SQUARE/EXP`. Define a parallel `METRIC_L2 / METRIC_RMSE / METRIC_L1 / METRIC_QUANTILE / …` block.

**Analog B (two-stage reduction stage-2 — D-10, do NOT rebuild):** `primitives.rs:784-789`:
```rust
pub fn reduce_sum_f64_on<R: cubecl::Runtime>(
    client: &ComputeClient<R>, data: &[f64],
) -> Result<f64, ComputeError> {
    reduce_f64_on(client, data, ReduceOp::Sum)  // ascending matched order, bit-exact
}
```
The single-owner ordered f64 fold (`primitives.rs:748-777`, `CubeCount::Static(1,1,1)` / `CubeDim::new_1d(1)`, `read_one_unchecked`) IS the bit-exact `ShuffleReduceSumGlobal` anchor. Per-block `ShuffleReduceSum` partials fold to `sum_loss` / `sum_weight` through this.

**Analog C (§12.1 per-point loss math — transcription source):** `crates/lgbm-metric/src/regression.rs:242-310` `loss_on_point` is the VERBATIM per-metric math to transcribe into the comptime `metric_on_point` branches. The 8 device-supported regression arms, transcribed:
```rust
// Source: lgbm-metric/src/regression.rs:242-310 (host f64 reference — D-03 secondary anchor)
L2 | Rmse       => { let d = score - label; d*d }              // RMSE applies sqrt at AverageLoss
L1             => (score - label).abs()
Quantile{alpha}=> { let delta = label - score;
                    if delta < 0.0 { (alpha-1.0)*delta } else { alpha*delta } }
Huber{alpha}   => { let diff = score - label;
                    if diff.abs() <= alpha { 0.5*diff*diff }
                    else { alpha*(diff.abs() - 0.5*alpha) } }
Fair{c}        => { let x = (score-label).abs(); c*x - c*c*(x/c).ln_1p() }
Poisson        => { let eps = 1e-10_f32 as f64; let s = if score<eps {eps} else {score};
                    s - label*s.ln() }                          // score is post-exp ConvertOutput
Mape           => (label - score).abs() / (1.0_f32 as f64).max(label.abs())
Gamma          => { let theta = -1.0/score; let b = -safe_log(-theta);
                    let c = safe_log(label) ... ; -((label*theta - b) + c) }  // psi=1 deviance
GammaDeviance  => { let tmp = label/(score+1.0e-9); tmp - safe_log(tmp) - 1.0 }  // AvgLoss = sum*2
Tweedie{rho}   => { let eps=1e-10_f32 as f64; let s=if score<eps {eps} else {score};
                    -label*((1.0-rho)*s.ln()).exp()/(1.0-rho) + ((2.0-rho)*s.ln()).exp()/(2.0-rho) }
```
**AverageLoss finalizers** (`regression.rs:228-235`): `Rmse => (sum_loss/sum_weights).sqrt()`, `GammaDeviance => sum_loss*2.0`, else `sum_loss/sum_weights`.

**Binary logloss arm** (`lgbm-metric/src/binary.rs:119-130` doc): `label <= 0 → -log(1-prob)` (guard `1-prob > kEpsilon`); `label > 0 → -log(prob)` (guard `prob > kEpsilon`); else `-log(kEpsilon)`. `kEpsilon = 1e-15f` (`lgbm_core::types::K_EPSILON`). Binary returns `Σloss/Σweight`. **`prob` is the sigmoid-converted score** (see ConvertOutput compose below).

**ConvertOutput compose into Eval (D-04):** `score`/`prob` fed to `metric_on_point` is POST-ConvertOutput. The regression host reference applies it at `regression.rs:219-224` — poisson/gamma/gamma_deviance/tweedie run `convert_poisson` (exp) first. On device, run `convert_output_on` (`objective_regression.rs:386`, mode `CONVERT_EXP`) or the binary sigmoid convert (`objective_binary.rs`) into a pre-allocated `score_convert_buffer_` BEFORE `EvalKernel`. **Route the mode off the ORIGINAL metric/objective name, NOT off `DeviceObjectiveKind`** (`device_objective.rs:33-39` explicit warning: the kind enum is a support classifier, not a ConvertOutput key).

**D-11 pre-allocate-once:** `score_convert_buffer_`, per-block reduction partials, `reduce_block_buffer` allocated ONCE outside the eval loop — precedent `predict.rs:263-265`.

---

### `device_metric.rs` (discriminator, classify) — §12 SC #3, D-05

**Analog:** `crates/lgbm-compute/src/device_objective.rs:1-116` — mirror `device_objective_supported` EXACTLY (enum + `from_name → Option<Kind>` + `supported = from_name.is_some()`):
```rust
// Source: device_objective.rs:114-116 (the pattern to mirror)
#[must_use]
pub fn device_objective_supported(name: &str) -> bool {
    DeviceObjectiveKind::from_name(name).is_some()
}
```
`from_name` shape to copy (`device_objective.rs:75-100`): a `match name { "regression"|"l2"|... => Some(...), _ => None }` alias table.

**CRITICAL asymmetry (D-05 — do NOT reuse the objective list):** `metric_supported(name)` returns `true` for exactly the **12 pointwise losses** (RMSE, L2, L1, Quantile, Huber, Fair, Poisson, MAPE, Gamma, Gamma-deviance, Tweedie, Binary-logloss). **MAPE / Gamma / Gamma-deviance / Tweedie are metric-supported even though their OBJECTIVES return `None` from `device_objective_supported`** (`device_objective.rs:97-99` lists them host-only). Returns `false` for AUC / AUC-mu / NDCG / MAP / multi_error / multi_logloss / xentropy / xentlambda / KullbackLeibler. Copy the test structure from `device_objective.rs:118-159` (`unsupported_*_are_rejected` / `supported_*_are_accepted`).

---

### `lib.rs` (driver / backend seam, EDIT) — §6/§16, D-01 (ODL-18/19 pulled forward)

**Analog: the dormant seam itself** — `crates/lgbm-compute/src/lib.rs:1241-1292`. Two edits: (1) flip `on_device_growth_supported()` (gated — see Pitfall 2), (2) fill the `grow_tree_on_device` body (currently `Ok(None)`):
```rust
// Source: lib.rs:1241-1292 — the seam to activate
fn on_device_growth_supported(&self) -> bool { false }   // ← flip TRUE, gated (Pitfall 2)

fn grow_tree_on_device(
    &self, _gradients: &[f32], _hessians: &[f32], _num_leaves: i32, _max_depth: i32,
) -> Result<Option<(lgbm_model::Tree, lgbm_dataset::LeafPartitionLayout)>, ComputeError> {
    Ok(None)   // ← fill: root init → per-leaf {build/subtract → best-split → break if -1 → split → partition}
}
```
The seam's own cubecl-0.10 checklist is at `lib.rs:1264-1279` (NO cross-cube barrier; `Atomic<i64>` broken → u64 fixed-point; no `wrapping_add` in `#[cube]`; plane-sum ≤ one plane width; `launch_unchecked` invariants by hand). Compose Phase-16 `kernels/histogram.rs`, Phase-17 `kernels/best_split.rs`, Phase-18 `kernels/{tree,data_partition}.rs`, Phase-14 `kernels/{split_info,random}.rs`.

**Return-payload type (do NOT name treelearner's `DataPartition` here — crate cycle):** build `lgbm_dataset::LeafPartitionLayout` (`crates/lgbm-dataset/src/dataset.rs:88-97`):
```rust
pub struct LeafPartitionLayout {
    pub num_data: i32, pub indices: Vec<u32>, pub leaf_begin: Vec<i32>, pub leaf_count: Vec<i32>,
}
```

**Env gate (D-09):** `cuda_on_device_enabled()` (`lib.rs:1313-1317`) is the OnceLock `LGBM_CUDA_ON_DEVICE` seam. **Pitfall 2 (verified):** `GpuBackend<R>` is ONE generic impl shared by ROCm/CUDA/WGPU (`lib.rs:1238-1240`) — a bare `true` over-claims all three. Gate the flip so ROCm-with-env-unset stays `false`.

**Learner consumer is already wired (Pattern 4 — dormant plumbing):** `crates/lgbm-treelearner/src/learner.rs:714-724` — activates automatically when the discriminator flips. Production uses `Ok(None) ⇒ fall through` ONLY (NO host-fallback):
```rust
// Source: learner.rs:714-724 (already present, dead until flip)
if self.on_device_eligible {
    if let Some((tree, payload)) = self.backend.grow_tree_on_device(
        gradients, hessians, self.num_leaves, self.max_depth)? {
        let part = DataPartition::from_payload(payload);
        return Ok((tree, Vec::new(), ColSamplerTrace::default(), part));
    }
}
```
**Reconstruction (do NOT hand-roll):** `crates/lgbm-treelearner/src/data_partition.rs:74-81` `DataPartition::from_payload` — thin field-move from `LeafPartitionLayout`.

---

### `lgbm-boosting/src/score_updater.rs` (service, EDIT) — D-02 host-mirror toggle

**Analog: itself.** The existing host `ScoreUpdater` (f64 accumulator, class-major `offset = num_data * cur_tree_id` at `score_updater.rs:64-66`) is BOTH the mirror target and the behavioral reference. Enumerate every caller to map to a resident device op (Pitfall 5):
```rust
// Source: score_updater.rs — the three callers each resident op must cover
add_constant(val, cur_tree_id)   // :82  → AddScoreConstant   (init score / no-split single-leaf)
multiply_score(val, cur_tree_id) // :98  → MultiplyScoreConstant (shrinkage / DART rescale / RF avg)
add_tree_train_path(learner, tree, part, cur_tree_id) // :114 → Phase-18 add_prediction_to_score (per-leaf)
```
`add_tree_predict_path` (:139) / `add_tree_scaled_all` (:170) are per-row-predict DART/RF paths — OUT of the continuous proving slice; confirm they stay host or defer explicitly. Add a `boosting_on_cuda_`-keyed toggle: resident device path when true; `CopyFromCUDADeviceToHost` mirror into `self.score` when false so non-resident consumers read the host vector. **Additive only, gated (D-09).**

---

### `lgbm-boosting/src/gbdt.rs` (boosting driver, EDIT) — D-01/D-04 wiring

**Analog:** existing `GBDT::TrainOneIter` score-update + `Metric::Eval` call sites; the `boosting_on_cuda_` seam. Wire the resident loop (score never leaves device across the train) and the §16 sequencing (Shrinkage → UpdateScore(§11) → optional RenewTreeOutput(§5.1) → Metric.Eval(§12)) behind `LGBM_CUDA_ON_DEVICE`. **Pitfall 4:** follow §16 order exactly; scope the first driver slice to L2 (no RenewTreeOutput refit), document the ordering contract.

---

### `oracle-harness/tests/metric_parity.rs` (test, EDIT) — ODL-17

**Analog: itself** — the capture-gated SKIP-replay idiom (`metric_parity.rs:1-60`). `read_golden` returns `None` (SKIP) when a triplet is absent, so a fresh checkout stays green. Add on-device metric cells that `load_triplet(name)` and assert the device `EvalKernel` value vs the `lib_lightgbm` golden within `ORACLE_TOL`.

**⚠ Pitfall 1 (VERIFIED against `tests/fixtures/metric/`):** only **8 of 12** device-supported goldens exist on disk (quantile, huber, fair, mape, poisson, gamma, gamma_deviance, tweedie). **RMSE, L2, L1, binary_logloss have NO golden.** Wave-0 must extend the capture script (below) or those 4 cells SKIP forever, silently under-testing. (The fixtures dir also holds UNSUPPORTED-metric goldens — cross_entropy, cross_entropy_lambda, multi_error, auc_mu, average_precision, kullback_leibler — used by host-fallback replay, not device cells.)

---

### `oracle-harness/tests/learner_parity.rs` (test, EDIT) — ODL-18 structure gate

**Analog: activate the existing oracle** `assert_on_device_tree_matches_cpu_anchor` (`learner_parity.rs:2185-2241`) with a REAL on-device tree (currently runs against the `host_grow` stand-in at :2265). Structure BIT-EXACT + leaf values within `ROCM_LEAF_VALUE_TOL`; `default_left` tie-aware:
```rust
// Source: learner_parity.rs:2217-2233 — tie-aware default_left acceptance
if (od & DEFAULT_LEFT_MASK) != (an & DEFAULT_LEFT_MASK) {
    let gain_gap = (on_device.split_gain[node] as f64 - anchor.split_gain[node] as f64).abs();
    let gain_near_tie = gain_gap <= SPLIT_GAIN_TIE_TOL * gain_scale;
    let same_threshold = on_device.threshold[node] == anchor.threshold[node];
    let same_child_counts = child_row_counts(on_device, node) == child_row_counts(anchor, node);
    assert!(gain_near_tie && same_threshold && same_child_counts, /* real divergence hard-fails */);
}
```
**Anchor discipline (D-07):** `anchor` is ALWAYS `cpu_anchor_tree` (`learner_parity.rs:2245`, cubecl-cpu f64 fold). NEVER compare two GPU f32 paths (def-f8u-01). The `host_grow` stand-in (:2265) is TEST-ONLY and must not leak into production.

**Pitfall 3 (unverified — resolve at plan time):** the data→leaf map buffer alias-vs-double-buffer for the per-split partition rewrite. Prefer double-buffer; A/B against the cpu anchor before committing (latent `HistArena::swap` aliasing note bites this multi-leaf loop).

---

### `xtask/py/metric_oracle_capture.py` (tooling, EDIT) — Pitfall 1 Wave-0

**Analog: itself** (`metric_oracle_capture.py:210-238` `train_and_capture`). Add `train_and_capture(out_dir, "rmse"/"l2"/"l1", seed, "regression", <metric>, reg)` and `"binary_logloss"` (objective `"binary"`), then `cargo run -p xtask -- metric-oracle-capture` against the pinned lightgbm 4.6 uv `.venv` at repo root (project memory `phase8-python-venv`). If the venv is unavailable at plan time, anchor those 4 to the cpu-f64 fold and document the weaker proof (A1).

---

## Shared Patterns

### Comptime-mode kernel dispatch (the established fan-out idiom)
**Source:** `crates/lgbm-compute/src/kernels/objective_regression.rs:50-71` (tag constants) + `:356-372` (comptime chain) + `:93-129` (dual `<tag, use_weight>` template).
**Apply to:** `metric_pointwise.rs` (`EvalKernel<metric, weights>` + `metric_on_point`), `score_updater.rs` (add-vs-multiply if unified). One `#[cube]` body, `#[comptime]` branch folds to straight-line code — matches C++ `EvalKernel<CUDA_METRIC, USE_WEIGHTS>` exactly.

### Ordered f64 fold = the bit-exact anchor (never hand-roll a reducer)
**Source:** `crates/lgbm-compute/src/kernels/primitives.rs:784-789` `reduce_sum_f64_on` (single-owner, ascending matched order).
**Apply to:** metric two-stage global fold (stage-2), score init reductions, BoostFromScore folds. D-10.

### launch_unchecked confinement + bounds-guard (Security V5 / CMP-01)
**Source:** `objective_regression.rs:400-413` and `primitives.rs:745-777` — `create_from_slice` handles sized exactly, `i < len` guard inside the `#[cube]`, `unsafe` confined to the launch site, `read_one_unchecked` readback.
**Apply to:** every new kernel launcher (`score_updater.rs`, `metric_pointwise.rs`, the driver in `lib.rs`). Validate `num_data >= 0`, buffer lengths, `tree_id >= 0` (overflow-safe `offset` via the `usize` arithmetic at `score_updater.rs:64`).

### Host-fallback discriminator (pure classifier, no error noise)
**Source:** `crates/lgbm-compute/src/device_objective.rs:104-116`.
**Apply to:** `device_metric.rs` `metric_supported` — keys on the METRIC list, independent of the objective-support list (D-05).

### Dormant-seam activation (already-wired plumbing)
**Source:** `learner.rs:714-724` (fork) + `data_partition.rs:74` (`from_payload`) + `dataset.rs:88` (`LeafPartitionLayout`) + `lib.rs:1313` (env gate).
**Apply to:** the `lib.rs` driver — flip ONE discriminator, fill ONE body; the consumer path already exists and is exercised by Slice-0 no-op tests.

### Anchor discipline (never GPU-vs-GPU)
**Source:** `learner_parity.rs:2170-2185, 2245` — reference is always the cubecl-cpu f64 fold (`cpu_anchor_tree`); the only tolerated divergence is a `default_left` flip on a genuine f32-vs-f64 `split_gain` near-tie.
**Apply to:** all three D-06 parity layers (kernel / resident-score A/B / structure gate). def-f8u-01.

---

## No Analog Found

None. Every new/modified file has a strong in-tree analog. The two genuinely-new modules (`score_updater.rs`, `metric_pointwise.rs`) are elementwise/map-reduce compositions of existing Phase-14/18/19 kernels; the driver (`lib.rs`) fills a seam whose consumer, payload struct, reconstruction, and oracle already exist.

> RESEARCH.md §Code Examples supply the exact new-kernel skeletons; prefer the real
> in-tree analogs above (verified this session) as the primary copy source.

## Metadata

**Analog search scope:** `crates/lgbm-compute/src/{kernels,}/`, `crates/lgbm-boosting/src/`, `crates/lgbm-treelearner/src/`, `crates/lgbm-dataset/src/`, `crates/lgbm-metric/src/`, `crates/oracle-harness/tests/{fixtures/metric,}/`
**Files scanned (read this session):** 11 (objective_regression, primitives, device_objective, predict, lib, learner, data_partition, dataset, score_updater, regression/binary metric, metric_parity, learner_parity, fixtures listing)
**Pattern extraction date:** 2026-07-02
</content>
</invoke>
