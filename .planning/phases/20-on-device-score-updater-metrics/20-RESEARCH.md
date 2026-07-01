# Phase 20: On-Device Score Updater & Metrics - Research

**Researched:** 2026-07-02
**Domain:** CubeCL on-device compute port (GBDT boosting-layer device path) — score residency, pointwise metric evaluation, and the pulled-forward end-to-end on-device grow driver
**Confidence:** HIGH (all claims verified against in-tree source; no external dependencies)

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- **D-01: Pull Phase 21's on-device grow (ODL-18/ODL-19) FORWARD into Phase 20 — full end-to-end resident loop.** `cuda_score_` stays resident across the entire train, fed by an on-device grow loop; `on_device_growth_supported()` flips to **true** this phase. Absorbs Phase-21's single-GPU-driver scope + STRUCTURE-bit-exact gate. Driver runs root init → build/subtract → best-split → tree split → partition, up to `num_leaves−1` (break on `best_leaf == −1`); continuous-feature path is the proving slice (§6, §16); grown tree STRUCTURE bit-exact to the cpu f64 anchor (tie-aware `default_left`), leaf values within ~1e-5; f32 + u64 fixed-point build, **no f64 per-row hot loops** (§17). The conservative reading (resident buffer over host-grown trees, discriminator staying false) was explicitly **declined**.
- **D-02: `AddScoreConstantKernel` / `MultiplyScoreConstantKernel` are whole-array scalar ops** over resident `double* cuda_score_` (per tree at `offset = num_data·tree_id`); per-leaf `AddScore` delegates to the Phase-18 §9/§10 `AddPredictionToScore` kernels; a host-mirror toggle (`CopyFromCUDADeviceToHost` when `boosting_on_cuda_` is false) syncs host `score_`. `double*` accumulator is reference-blessed f64 — not a per-row hot-loop violation.
- **D-03: Anchor ALL 12 device metric kernels directly to real compiled-`lib_lightgbm` goldens.** Captures live in `crates/oracle-harness/tests/fixtures/metric/` with `tests/metric_parity.rs`. *(See Pitfall 1 — only 8 of 12 goldens actually exist on disk; 4 must be captured.)*
- **D-04: Compose the Phase-19 device `ConvertOutputCUDA` inverse-link into the Eval flow THIS phase.** Allocate `score_convert_buffer_`, run inverse-link (sigmoid / softmax / Poisson `exp` / `sign·x²` / pass-through) into it, then `EvalKernel`, anchor-pinned.
- **D-05: Express the supported/unsupported boundary as a metric-supported discriminator** (mirroring `on_device_growth_supported` / `device_objective_supported`) that device-evaluates exactly the 12 pointwise losses and routes AUC / AUC-mu / NDCG / MAP / multiclass (`multi_error`, `multi_logloss`) / cross-entropy (`xentropy`, `xentlambda`) / KullbackLeibler to host — matching `Metric::CreateMetric`'s `#ifdef USE_CUDA` branch. **Load-bearing asymmetry:** MAPE / Gamma / Gamma-deviance / Tweedie **metrics ARE device-supported** even though those **objectives fell back to host** in Phase 19. Do not conflate objective-support with metric-support.
- **D-06: Three-layer parity gate:** (1) Kernel-level — constant-op kernels + `EvalKernel`/reduction anchored to the cpu f64 fold (tie-aware / ~1e-6–1e-5); (2) Resident-score A/B — after a full multi-iter run with `LGBM_CUDA_ON_DEVICE=1`, resident `cuda_score_` matches host `score_` (bit-for-bit where the algorithm permits, else f32 envelope); (3) Structure gate — grown tree STRUCTURE bit-exact to the cpu f64 anchor (tie-aware `default_left`), leaf values within ~1e-5. Never compare two GPU f32 paths to each other.
- **D-07:** Anchor every numeric output to the cubecl-cpu f64 fold; **never GPU-vs-GPU** (def-f8u-01). Atomic-ordering nondeterminism is the documented f32-vs-f64 residual the ROCm gate tolerates — pin to the f64 anchor with tie-aware / envelope asserts.
- **D-08:** **NO f64 per-row hot loops** in new kernels (5.4× consumer-NVIDIA f64 regression, spike-052). The `double* cuda_score_` accumulator and `double point_metric` / `double param` in `MetricOnPointCUDA` ARE reference-blessed f64 (§11/§12/§17) — keep those; the prohibition targets gratuitous f64 in per-row grow/build hot loops.
- **D-09:** `LGBM_CUDA_ON_DEVICE` **OFF by default**; CPU / ROCm / existing-host-CUDA paths **byte-unchanged**; full merge gate green and unchanged (ODL-19 hard merge gate). Wiring the `boosting_on_cuda_` seam permitted only behind this gate.
- **D-10:** **Reuse existing primitives / kernels — do NOT rebuild.** Phase-14 `ShuffleReduceSum` + global fold (`primitives.rs`), Phase-18 `AddPredictionToScore` (`predict.rs`) + data-partition/tree kernels, Phase-19 device `ConvertOutputCUDA` / `GetGradients` / `BoostFromScore`. The score/metric/driver code COMPOSES these.
- **D-11:** **Pre-allocate scratch ONCE outside the hot loop** (`reduce_block_buffer`, `score_convert_buffer_`, per-block reduction partials) — no per-call in-kernel device alloc.

### Claude's Discretion
- Whether `EvalKernel` is one comptime-generic `#[cube]` (branch on a metric enum via the `CUDA_METRIC` template analog) or 12 concrete kernels — parity-neutral as long as each `MetricOnPointCUDA` matches the §12.1 table. **Recommendation below.**
- The exact code shape of the metric-supported discriminator (enum method vs match table) — parity-neutral; only the supported/unsupported SETS (D-05) are locked.
- Module placement — likely `score_updater.rs` / `metric.rs` (or `metric_pointwise.rs`) in `crates/lgbm-compute/src/kernels/`, plus the boosting-layer `boosting_on_cuda_` wiring in `lgbm-boosting`.
- Block/geometry constants (`NUM_DATA_PER_EVAL_THREAD=1024`, `num_threads_per_block_`) — start from the faithful C++ constants; APU-aware autotune deferred.
- The on-device driver's `Handle` in-place-alias vs ping-pong double-buffer for the data→leaf map, and batched `client.read(vec![h])` readback semantics — **verify at plan time**.

### Deferred Ideas (OUT OF SCOPE)
- **On-device categorical splits** (bitset construction / categorical eval / partition / `SplitCategorical`) → Phase 22.
- **Perf-validation / default-ON rollout DoD** (Kaggle A/B, `device_launches` + wall-clock ratio, flipping the CUDA default) → Phase 23.
- **CUDA-unsupported metrics** (AUC / AUC-mu / NDCG / MAP / multiclass / xentropy / KL) → permanent host fallback, never ported.
- **APU-aware autotune** of EvalKernel / grow-loop block geometry → deferred perf option.
- **Phase 21 re-scope** — Phase 21 shrinks to hardening/slack; re-cut ROADMAP via `/gsd-phase` before planning (flagged, not applied).
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| ODL-16 | On-device score update — resident `cuda_score_`, constant add/multiply (init/shrinkage/no-split/DART), replacing host `add_prediction_to_score` scatter, host-mirror toggle | §11 port map below; host reference is `score_updater.rs` (add_constant / multiply_score / add_tree_train_path). Delegate per-leaf to Phase-18 `add_prediction_to_score_on_device` (`predict.rs`). Constant ops are trivial elementwise `#[cube]` over the resident buffer. |
| ODL-17 | On-device pointwise metric eval — `EvalKernel` + two-stage reduction over 12 supported losses, anchor-pinned; unsupported fall back to host | §12/§12.1 port map below; host math reference is `lgbm-metric/{regression,binary}.rs`; goldens `oracle-harness/tests/fixtures/metric/*` (⚠ 4 missing — Pitfall 1); reduction primitive `primitives::reduce_sum_f64_on`; ConvertOutput compose via `objective_regression::convert_output_on` + `CONVERT_*` modes; new metric-supported discriminator mirrors `device_objective_supported`. |
| ODL-18 (pulled fwd, D-01) | On-device single-GPU tree-learner driver, per-leaf grow loop end-to-end, reconstitutes into `(Tree, DataPartition)`; STRUCTURE bit-exact | §6/§16 sequencing; the seam is `Backend::grow_tree_on_device` (`lib.rs:1284`) — currently `Ok(None)` no-op. Driver composes Phase 16/17/18 kernels. Reconstruction via `DataPartition::from_payload` (`data_partition.rs:74`) ← `LeafPartitionLayout`. Learner fork already wired (`learner.rs:714`, dead until discriminator flips). |
| ODL-19 (pulled fwd, D-01) | f32 + u64 fixed-point build, NO f64 per-row hot loops; CPU/ROCm/host-CUDA byte-unchanged with env unset; hard merge gate green | D-08/D-09 discipline; verified by grep + per-tree-ms, not a 6× sweep. `cuda_on_device_enabled()` (`lib.rs:1313`) is the OnceLock env gate; `on_device_eligible` AND-gate in `learner.rs`. |
</phase_requirements>

## Summary

This phase is a **CubeCL on-device port**, not a library-integration phase — every "dependency" is an already-golden-validated in-tree kernel from Phases 14–19 that this phase **composes**. Two genuinely new subsystems are small: (1) the §11 score updater (two trivial whole-array scalar `#[cube]` kernels + a resident `f64` buffer + a host-mirror toggle), and (2) the §12 pointwise metric evaluator (`EvalKernel` over 12 losses + a two-stage f64 reduction + a ConvertOutput pre-pass + a metric-supported discriminator). The **large, higher-risk** work is the D-01 pulled-forward driver: implementing `Backend::grow_tree_on_device` for real (currently `Ok(None)`) by sequencing the existing Phase-16/17/18 kernels into the per-leaf loop, flipping `on_device_growth_supported()` to `true`, and passing the STRUCTURE-bit-exact gate against the cpu f64 anchor.

The learner-side plumbing is **already built and dormant**: the fork in `train_inner` (`learner.rs:714`), the `on_device_eligible` AND-gate (`= backend.on_device_growth_supported() && cuda_on_device_env()`), `DataPartition::from_payload`, the `LeafPartitionLayout` payload struct, and the tie-aware oracle `assert_on_device_tree_matches_cpu_anchor` all exist and are exercised in Slice-0 no-op tests. Phase 20 activates them.

The single most important research finding that changes the plan: **only 8 of the 12 device-supported metric goldens exist on disk** (quantile, huber, fair, mape, poisson, gamma, gamma_deviance, tweedie). **RMSE, L2, L1, and binary_logloss have NO real `lib_lightgbm` golden** — the capture script (`xtask/py/metric_oracle_capture.py`) never captured them. D-03 ("captures already exist for all 12") is only 2/3 true; the plan must either extend the capture script for those 4 or anchor them to the cpu f64 fold.

**Primary recommendation:** Structure the phase as three plans — (A) §11 score updater + resident buffer + host-mirror toggle, anchored via the resident-score A/B gate; (B) §12 metric evaluator + ConvertOutput compose + metric-supported discriminator, with a Wave-0 task to capture the 4 missing goldens; (C) the `grow_tree_on_device` driver + `on_device_growth_supported()` flip + STRUCTURE-bit-exact gate. Build (A) and (B) as pure kernel composition first (they are independently testable and low-risk); (C) is the load-bearing integration and should carry the plan-time verification of the `Handle` aliasing / batched-readback semantics.

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Resident cumulative score buffer (`cuda_score_`) | Device (lgbm-compute kernels) | Boosting host (lgbm-boosting) | The buffer lives on device across the whole train (D-01); the host `ScoreUpdater` owns the mirror + toggle |
| Constant add/multiply score ops | Device kernel | — | Whole-array elementwise `#[cube]` (§11) |
| Per-leaf score scatter | Device kernel (REUSE Phase-18 `AddPredictionToScore`) | — | D-02 delegates; do not rebuild |
| Host-mirror sync toggle | Boosting host (lgbm-boosting) | Device readback | `boosting_on_cuda_`-keyed `CopyFromCUDADeviceToHost`; non-resident consumers read host `score_` |
| Pointwise metric per-row loss | Device kernel (`EvalKernel`) | — | One thread/row, §12 |
| Metric two-stage reduction | Device (REUSE Phase-14 reduction) + host global fold | — | `ShuffleReduceSum` per block → host `ShuffleReduceSumGlobal` scalar |
| ConvertOutput inverse-link at metric boundary | Device kernel (REUSE Phase-19 `convert_output_on`) | — | D-04 composes into Eval flow |
| Metric-supported / unsupported routing | Host discriminator (lgbm-compute or lgbm-metric) | — | D-05; mirrors `device_objective_supported` |
| On-device grow-loop driver | Device driver (`grow_tree_on_device`) | Boosting/treelearner host orchestration | D-01; host sequences kernel launches (kernel boundaries ARE the barrier — no megakernel) |
| `(Tree, DataPartition)` reconstruction | Treelearner host | — | `DataPartition::from_payload` ← `LeafPartitionLayout` |
| Merge-gate / env-gate | Boosting + treelearner host | — | `cuda_on_device_enabled()` OnceLock; `on_device_eligible` AND-gate |

## Standard Stack

**No new external packages.** This phase adds only in-crate Rust modules that compose existing kernels. The relevant "stack" is the internal crate graph and the CubeCL version already in use.

### Core (existing, reused — D-10)
| Component | Location | Purpose | Reuse role |
|-----------|----------|---------|-----------|
| `Backend` trait + `grow_tree_on_device` seam | `crates/lgbm-compute/src/lib.rs:1233-1292` | On-device growth discriminator + driver entry | Flip `on_device_growth_supported()`; implement the driver body |
| `cuda_on_device_enabled()` | `crates/lgbm-compute/src/lib.rs:1313` | OnceLock `LGBM_CUDA_ON_DEVICE` env gate | The D-09 off-by-default merge-gate seam |
| `reduce_sum_f64_on` (+ max/min/dot) | `crates/lgbm-compute/src/kernels/primitives.rs:784` | Ordered f64 fold (the two-stage global fold) | Metric reduction stage-2; score/BoostFromScore folds |
| `convert_output_on` + `CONVERT_PASSTHROUGH/SQRT_SQUARE/EXP` | `crates/lgbm-compute/src/kernels/objective_regression.rs:386,67-71` | ConvertOutput inverse-link (regression/Poisson) | D-04 compose (regression + poisson/gamma/tweedie exp) |
| binary sigmoid ConvertOutput (`ConvertOutputCUDAKernel_BinaryLogloss`) | `crates/lgbm-compute/src/kernels/objective_binary.rs` | Sigmoid-probability inverse-link | D-04 compose (binary_logloss metric input) |
| `device_objective_supported(name)` | `crates/lgbm-compute/src/device_objective.rs:114` | Objective host-fallback discriminator | The PATTERN the new metric-supported discriminator mirrors (D-05) |
| `add_prediction_to_score_on_device` | `crates/lgbm-compute/src/kernels/predict.rs:212` | Phase-18 tree-walk score scatter | Per-leaf `AddScore` delegate (D-02) |
| `DataPartition::from_payload` | `crates/lgbm-treelearner/src/data_partition.rs:74` | Reconstruct partition from `LeafPartitionLayout` | Driver return path |
| `LeafPartitionLayout` | `crates/lgbm-dataset/src/dataset.rs:88` | The `(num_data, indices, leaf_begin, leaf_count)` payload | Driver output; treelearner reconstructs |
| `train_inner` on-device fork | `crates/lgbm-treelearner/src/learner.rs:714-724` | Dead-until-flip driver call site | Already wired; activates when discriminator flips |
| `ScoreUpdater` (host f64) | `crates/lgbm-boosting/src/score_updater.rs` | Host score accumulator (the mirror target) | The behavioral reference + non-resident mirror |

### Supporting (existing, reused)
| Component | Location | Purpose |
|-----------|----------|---------|
| Phase-16 histogram build/fix/subtract | `crates/lgbm-compute/src/kernels/histogram.rs` | Driver step 3a (§7) |
| Phase-17 best-split finder | `crates/lgbm-compute/src/kernels/best_split.rs` | Driver step 3b (§8) |
| Phase-18 tree mutation + data partition | `crates/lgbm-compute/src/kernels/{tree,data_partition}.rs` | Driver steps 3d/3e (§10/§9) |
| Phase-14 split-info / leaf-splits structs + RNG | `crates/lgbm-compute/src/kernels/{split_info,random}.rs` | Root init (§6.1) + per-leaf state |
| host metric math | `crates/lgbm-metric/src/{regression,binary,lib}.rs` | `MetricOnPointCUDA` transcription source (secondary anchor) |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Real `lib_lightgbm` goldens for all 12 metrics (D-03) | cpu-f64-fold anchor for the 4 missing (RMSE/L2/L1/binary_logloss) | cpu-fold is a re-transcription-agrees proof (weaker), but zero capture cost + no Python toolchain dependency. **Recommend: capture the 4 real goldens** (extend the existing script — cost is one function call each) to honor D-03's fidelity intent; fall back to cpu-fold only if the capture toolchain is unavailable at plan time. |
| One comptime-generic `EvalKernel` `#[cube]` (branch on metric enum) | 12 concrete kernels | **Recommend the single comptime-generic kernel.** It matches the C++ `EvalKernel<CUDA_METRIC, USE_WEIGHTS>` template exactly, the existing codebase already uses `#[comptime] mode: u32` dispatch (`convert_body` in `objective_regression.rs:358`), and it keeps the two-stage reduction wiring in one place. `MetricOnPointCUDA` becomes a `#[cube] fn` with a `#[comptime] metric: u32` match — identical to the established `CONVERT_*` mode pattern. 12 concrete kernels duplicate the reduction plumbing 12× for no parity or perf gain on the APU. |

**Installation:** None. `cargo build` / `cargo test -p oracle-harness` in the existing workspace.

**Version verification:** N/A — no package installs. CubeCL version is pinned in the workspace `Cargo.toml` (cubecl 0.10 per project memory; kernels already compile against it). No registry lookup performed because no new crate is added.

## Architecture Patterns

### System Architecture Diagram

```
                       GBDT::TrainOneIter  (lgbm-boosting/gbdt.rs)
                                 │
        ┌────────────────────────┼───────────────────────────────────┐
        │ (1) GetGradients        │                                    │
        │  Phase-19 device grad/hess (scores → g/h, resident)          │
        ▼                                                              │
  ┌──────────────────────────────────────────────┐                    │
  │ (2-4) grow_tree_on_device  [D-01 NEW DRIVER]  │  on_device_eligible│
  │  Backend::grow_tree_on_device (lib.rs:1284)   │  = discriminator    │
  │  ┌──────────────────────────────────────────┐│    && env gate      │
  │  │ root init (§6.1)  ── Phase-14 reduce      ││                    │
  │  │   ▼                                        ││                    │
  │  │ per-leaf loop, up to num_leaves−1:         ││                    │
  │  │   build smaller + subtract larger (§7) ────┼┼─ Phase-16 kernels  │
  │  │   best-split finder (§8) ──────────────────┼┼─ Phase-17 kernels  │
  │  │   if best_leaf == −1: break                ││                    │
  │  │   CUDATree::Split (§10, before partition)──┼┼─ Phase-18 tree     │
  │  │   DataPartition::Split (§9) ───────────────┼┼─ Phase-18 partition│
  │  └──────────────────────────────────────────┘│                    │
  │  returns (Tree, LeafPartitionLayout)          │                    │
  └──────────────────────────────────────────────┘                    │
        │  DataPartition::from_payload  (treelearner)                  │
        ▼                                                              │
  ┌────────────────────────────────────────┐                          │
  │ (5) Shrinkage(rate) + ScoreUpdater     │                          │
  │     UpdateScore  [§11 NEW]              │                          │
  │   MultiplyScoreConstant (shrinkage) ────┼─ NEW whole-array #[cube] │
  │   AddScore(tree) ───────────────────────┼─ Phase-18 AddPredToScore │
  │   resident cuda_score_ (f64) STAYS on device                        │
  │   if !boosting_on_cuda_: CopyToHost mirror                          │
  └────────────────────────────────────────┘                          │
        │                                                              │
        ▼                                                              │
  ┌────────────────────────────────────────┐                          │
  │ (7) Metric::Eval  [§12 NEW]            │◄─────────────────────────┘
  │   metric_supported(name)? ── else HOST fallback (D-05)             │
  │   ConvertOutputCUDA → score_convert_buffer_ (Phase-19, D-04)       │
  │   EvalKernel<metric,weights> (one thread/row) ── NEW #[cube]       │
  │   per-block ShuffleReduceSum → host global fold (Phase-14)         │
  │   → AverageLoss (RMSE sqrt) / Σloss÷Σweight                        │
  └────────────────────────────────────────┘
```

### Recommended Project Structure
```
crates/lgbm-compute/src/kernels/
├── score_updater.rs      # NEW: AddScoreConstant / MultiplyScoreConstant #[cube]; delegates per-leaf to predict.rs
├── metric_pointwise.rs   # NEW: EvalKernel<metric,weights> comptime-generic; MetricOnPointCUDA table; two-stage fold
crates/lgbm-compute/src/
├── device_metric.rs      # NEW: metric_supported(name) discriminator, mirrors device_objective.rs
├── lib.rs                # EDIT: on_device_growth_supported()→true (gated); grow_tree_on_device driver body
crates/lgbm-boosting/src/
├── score_updater.rs      # EDIT: resident/mirror toggle (boosting_on_cuda_) additive path
├── gbdt.rs               # EDIT: wire resident loop behind LGBM_CUDA_ON_DEVICE
crates/oracle-harness/tests/
├── metric_parity.rs      # EDIT: add on-device metric cells (+ 4 new goldens)
├── learner_parity.rs     # EDIT: activate STRUCTURE gate against cpu anchor (assert_on_device_tree_matches_cpu_anchor)
xtask/py/metric_oracle_capture.py  # EDIT: capture rmse/l2/l1/binary_logloss goldens (Pitfall 1)
```

### Pattern 1: Comptime-mode kernel dispatch (the established idiom)
**What:** A single `#[cube]` body branches on a `#[comptime]` mode constant — the codebase's proven fan-out pattern.
**When to use:** `EvalKernel<CUDA_METRIC, USE_WEIGHTS>` and `MetricOnPointCUDA` (recommend over 12 concrete kernels).
**Example:**
```rust
// Source: crates/lgbm-compute/src/kernels/objective_regression.rs:356-372 (verbatim pattern)
#[cube]
#[allow(unused_assignments)]
fn convert_body<F: Float>(input: &Array<F>, out: &mut Array<F>, #[comptime] mode: u32) {
    let i = ABSOLUTE_POS;
    if i < input.len() {
        let x = input[i];
        let mut y = x;
        if mode == CONVERT_PASSTHROUGH { y = x; }
        else if mode == CONVERT_SQRT_SQUARE { y = sign_f::<F>(x) * x * x; }
        else { y = x.exp(); }
        out[i] = y;
    }
}
```
The new `metric_on_point<F>(label, score, param, #[comptime] metric: u32) -> F` follows this exactly — one branch per §12.1 row.

### Pattern 2: Ordered f64 fold as the two-stage reduction stage-2
**What:** Per-block partials are folded to a scalar by the single-owner ordered f64 fold — this IS the bit-exact anchor.
**When to use:** Metric global fold (`ShuffleReduceSumGlobal` analog); score init reductions.
**Example:**
```rust
// Source: crates/lgbm-compute/src/kernels/primitives.rs:784-789
pub fn reduce_sum_f64_on<R: cubecl::Runtime>(
    client: &ComputeClient<R>, data: &[f64],
) -> Result<f64, ComputeError> {
    reduce_f64_on(client, data, ReduceOp::Sum)  // ascending matched order, bit-exact
}
```

### Pattern 3: Host-fallback discriminator (mirror for D-05)
**What:** A pure `fn supported(name: &str) -> bool` gates the device path; unsupported names quietly take the host path (no error noise).
**Example:**
```rust
// Source: crates/lgbm-compute/src/device_objective.rs:114 (the pattern to mirror)
pub fn device_objective_supported(name: &str) -> bool { /* match on canonical names */ }
```
The new `metric_supported(name)` returns `true` for exactly the 12 pointwise losses; `false` for AUC/AUC-mu/NDCG/MAP/multi_error/multi_logloss/xentropy/xentlambda/KL. **Key trap:** it keys on the METRIC name list, NOT the objective-support list — MAPE/Gamma/Gamma-deviance/Tweedie are metric-supported though objective-unsupported (D-05).

### Pattern 4: Dormant seam activation (already-wired plumbing)
**What:** The learner fork, payload struct, and reconstruction already exist; the phase flips one discriminator and fills one method body.
**Example:**
```rust
// Source: crates/lgbm-treelearner/src/learner.rs:714-724 (already present, dead until flip)
if self.on_device_eligible {
    if let Some((tree, payload)) = self.backend.grow_tree_on_device(
        gradients, hessians, self.num_leaves, self.max_depth)? {
        let part = DataPartition::from_payload(payload);
        return Ok((tree, Vec::new(), ColSamplerTrace::default(), part));
    }
}
```
Production uses `Ok(None) ⇒ fall through` ONLY — there is NO `unwrap_or_else(host_grow)` fallback in production (that stand-in lives in the test harness only, `learner_parity.rs:2260`).

### Anti-Patterns to Avoid
- **GPU-vs-GPU comparison (def-f8u-01):** Never assert resident cuda_score_ or the on-device tree against a *second GPU f32 path*. Always pin to the cpu f64 anchor (`cpu_anchor_tree`, `learner_parity.rs:2245`).
- **f64 per-row hot loop (spike-052, 5.4× regression):** The metric per-row `point_metric` is f64 by reference blessing, but the GROW/BUILD hot loops stay f32 + u64 fixed-point. Do not "promote to f64 for accuracy" in the histogram/partition path.
- **Flipping `on_device_growth_supported()` unconditionally:** `GpuBackend<R>` is ONE generic impl shared by ROCm/CUDA/WGPU (`learner_parity.rs:2486` comment). A bare `true` wrongly claims all three support it. See Pitfall 3.
- **Conflating objective-support with metric-support (D-05):** the two lists are independent; reusing `device_objective_supported` for metric routing silently drops MAPE/Gamma/Tweedie metrics to host.
- **Per-call device alloc in the loop:** pre-allocate `score_convert_buffer_`, `reduce_block_buffer`, per-block partials ONCE (D-11).

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Metric two-stage reduction | A new block/global sum kernel | `primitives::reduce_sum_f64_on` (stage-2) + a per-block `ShuffleReduceSum` mirroring Phase-14 | The ordered f64 fold IS the bit-exact anchor; a new reducer diverges (D-10) |
| Per-leaf score scatter | A fresh tree-walk score kernel | Phase-18 `add_prediction_to_score_on_device` (`predict.rs:212`) | D-02 delegates; already golden-validated |
| Inverse-link at metric boundary | A new sigmoid/exp/sqrt kernel | Phase-19 `convert_output_on` + `CONVERT_*` modes + binary sigmoid convert | D-04 compose; already parity-pinned |
| Histogram / split / partition in the driver | Re-implementing §7/§8/§9/§10 | Phase-16/17/18 kernels sequenced by the host driver | D-01/D-10; the whole point is composition |
| `(Tree, DataPartition)` reconstruction | Manual field copies | `DataPartition::from_payload` (`data_partition.rs:74`) | Already the acyclic-safe payload path |
| STRUCTURE parity assertion | A new tree-diff | `assert_on_device_tree_matches_cpu_anchor` (`learner_parity.rs:2185`) | Tie-aware default_left, leaf-value envelope already encoded |
| Metric golden capture | A bespoke Python | Extend `xtask/py/metric_oracle_capture.py` + `cargo run -p xtask -- metric-oracle-capture` | Existing toolchain; version-pinned to lib_lightgbm 4.6 |

**Key insight:** Phase 20's new code is thin glue (2 constant kernels, 1 comptime metric kernel, 1 discriminator, 1 driver sequencer). ~80% of the parity surface is already-validated Phase-14–19 kernels. The risk is in *sequencing and residency*, not in new numeric kernels.

## Runtime State Inventory

Not a rename/refactor phase — greenfield additive kernels behind a gate. Omitted (no stored data / live service config / OS state / secret renames / build-artifact churn).

## Common Pitfalls

### Pitfall 1: Four device-supported metric goldens do not exist (D-03 gap)
**What goes wrong:** D-03 says "captures already exist in `oracle-harness/tests/fixtures/metric/`" for anchoring all 12. On disk there are goldens for only **8**: quantile, huber, fair, mape, poisson, gamma, gamma_deviance, tweedie. **RMSE, L2, L1, binary_logloss are absent** — `xtask/py/metric_oracle_capture.py:210-238` never captures them (it only captures the "extended" metrics + xentropy/multiclass/binary_precision families).
**Why it happens:** Those goldens were a Phase-7 "extended metric" capture (MET-03); the base regression/binary metrics were validated a different way and never got `lib_lightgbm` triplet captures.
**How to avoid:** Add a Wave-0 task to extend `metric_oracle_capture.py` with `train_and_capture(out_dir, "rmse"/"l2"/"l1", seed, "regression", <metric>, reg)` and `"binary_logloss"` with objective `"binary"`, then run `cargo run -p xtask -- metric-oracle-capture` (requires the version-pinned lightgbm 4.6 Python env — the uv `.venv` at repo root per project memory `phase8-python-venv`). If that env is unavailable at plan time, fall back to the cpu-f64-fold anchor for those 4 and document the weaker proof.
**Warning signs:** A plan that assumes all 12 on-device metric cells can `load_triplet(name)` immediately — 4 will `SKIP` (return early) forever, silently under-testing.

### Pitfall 2: `GpuBackend<R>` is one impl shared by ROCm / CUDA / WGPU
**What goes wrong:** Flipping `on_device_growth_supported()` to `true` on `GpuBackend<R>` claims ALL three runtimes grow on-device, but ROCm must stay byte-unchanged (D-09) and no CUDA-only path exists in the shared impl.
**Why it happens:** The single generic `GpuBackend<R>` impl (documented at `learner_parity.rs:2486`, `lib.rs:1238-1240`) cannot distinguish runtime at the trait level.
**How to avoid:** Gate the flip so it only reports `true` when (a) the env `LGBM_CUDA_ON_DEVICE=1` AND (b) the runtime is the intended target. Options to weigh at plan time: a runtime-type check inside the impl, a separate marker, or keeping the discriminator's `true` behind `cuda_on_device_enabled()` so ROCm-with-env-unset stays false. The existing `on_device_eligible = on_device_growth_supported() && cuda_on_device_env()` AND-gate (`learner.rs:498`) already provides the env half; the plan must ensure the *backend half* does not over-claim ROCm. **Flag for plan-time design.**
**Warning signs:** ROCm parity tests change output with `LGBM_CUDA_ON_DEVICE` unset — a D-09 violation.

### Pitfall 3: The `data→leaf` map buffer — alias vs double-buffer (unverified)
**What goes wrong:** The driver's partition step rewrites the row→leaf map every split. Whether the `Handle` can be aliased in-place or needs ping-pong double-buffering is an open cubecl-0.10 correctness question (flagged by the user, D-01 discretion). An in-place alias that the kernel reads-then-writes can corrupt (see the latent `HistArena::swap` aliasing note, `phase18-wr01-histarena-swap-aliasing`).
**Why it happens:** cubecl handle aliasing semantics + the mark→prefix-sum→scatter partition (§9) read and write index arrays in the same pass.
**How to avoid:** Verify at plan time with a tiny A/B (alias vs double-buffer) against the cpu anchor before committing the driver's buffer strategy. Prefer double-buffer unless proven safe. The project already documented `HistArena::swap` slot-aliasing as a latent bug that "will bite Phase-21 multi-leaf on-device grow loop" — this phase IS that loop.
**Warning signs:** Non-deterministic partition results / structure divergence that appears only at `num_leaves > 2`.

### Pitfall 4: `RenewTreeOutput` ordering for L1/quantile leaf refit (§16 step 6)
**What goes wrong:** The sequencing (§16) is Shrinkage → UpdateScore(§11) → *then* optional RenewTreeOutput(§5.1) → Metric.Eval(§12). If the resident loop applies leaf refit after the score update in the wrong order, the score and the model text diverge.
**How to avoid:** Follow §16 order exactly. For the continuous-feature proving slice, RenewTreeOutput only fires for L1/quantile/huber objectives — the plan can scope the first driver slice to L2 (no refit) and add refit as a follow-up, but must document the ordering contract.
**Warning signs:** Resident-score A/B (D-06 layer 2) drifts only for L1/quantile objectives.

### Pitfall 5: DART rescale / no-split single-leaf constant-op paths
**What goes wrong:** `MultiplyScoreConstantKernel` serves both shrinkage AND DART rescale AND the RF running-average pattern (`score_updater.rs:90-104`); `AddScoreConstantKernel` serves init-score AND the no-split single-leaf tree. Missing one caller leaves a host-only path that breaks residency.
**How to avoid:** Enumerate every host `ScoreUpdater` caller (`add_constant`, `multiply_score`, `add_tree_train_path`) and map each to a resident device op. The DART/RF `add_tree_predict_path` / `add_tree_scaled_all` per-row-predict paths (`score_updater.rs:139-185`) are out of the continuous proving slice — confirm they stay host or are explicitly deferred.

## Code Examples

### Metric per-row loss (the §12.1 table as a comptime match)
```rust
// Target shape — transcribe each row from docs/cuda-kernel-design.md §12.1 and
// cross-check against crates/lgbm-metric/src/regression.rs / binary.rs.
// param/point are f64 (reference-blessed, D-08); NOT a per-row hot-loop violation.
#[cube]
fn metric_on_point<F: Float>(label: F, score: F, param: F, #[comptime] metric: u32) -> F {
    // e.g. L2/RMSE: d = score - label; d*d
    // L1: (score-label).abs()
    // quantile: delta = label - score; if delta < 0 { (param-1)*delta } else { param*delta }
    // binary_logloss: if label <= 0 { -(1-score).ln() } else { -score.ln() }  (clamp kEpsilon)
    // ... one branch per §12.1 row
}
```

### Resident constant-op kernel (§11)
```rust
// AddScoreConstantKernel: score[i] += val  over the resident double buffer,
// offset = num_data * tree_id. MultiplyScoreConstantKernel is the *= twin.
// f64 buffer is reference-blessed (score_updater.rs already documents f64 accumulation).
#[cube]
fn add_score_constant(score: &mut Array<f64>, val: f64, offset: u32, num_data: u32) {
    let i = ABSOLUTE_POS;
    if i < num_data { score[offset + i] += val; }
}
```

### STRUCTURE-bit-exact gate (already built — reuse)
```rust
// Source: crates/oracle-harness/tests/learner_parity.rs:2185 — tie-aware oracle.
// structure fields bit-exact; leaf values within ROCM_LEAF_VALUE_TOL; default_left
// flip accepted ONLY on a genuine f32-vs-f64 split_gain near-tie (corroborated by
// identical threshold + child row-counts).
assert_on_device_tree_matches_cpu_anchor(&on_device_tree, &cpu_anchor_tree(..), "on-device");
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| `on_device_growth_supported()` frozen `false`; `grow_tree_on_device` → `Ok(None)` no-op | Discriminator flips `true` (gated); driver body sequences Phase-16/17/18 kernels | Phase 20 (this phase, D-01) | Activates the dormant learner fork; STRUCTURE gate goes live |
| Host `add_prediction_to_score` scatter into host `score_` | Resident `cuda_score_` on device across the train + host-mirror toggle | Phase 20 (§11) | Score never leaves device in the resident loop |
| Metrics evaluate only on host | 12 pointwise losses on device via `EvalKernel`; unsupported fall back to host | Phase 20 (§12) | Completes the boosting-layer device path |
| ODL-18/ODL-19 planned in Phase 21 | Pulled into Phase 20; Phase 21 → hardening/slack | 2026-07-02 discuss (D-01) | Larger, higher-risk Phase 20; ROADMAP re-scope pending |

**Deprecated/outdated:**
- The Slice-0 `host_grow` fallback stand-in (`learner_parity.rs:2260`) is TEST-ONLY and must NOT leak into production; production uses `Ok(None) ⇒ fall through`.
- The conservative "resident buffer over host-grown trees" reading of ODL-16 was explicitly declined (D-01).

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | The 4 missing goldens (rmse/l2/l1/binary_logloss) can be captured by extending the existing Python script against the repo's pinned lightgbm 4.6 venv | Pitfall 1 | If the venv/toolchain is unavailable, those 4 metrics anchor to the weaker cpu-f64 fold instead — still passes, weaker fidelity |
| A2 | The single comptime-generic `EvalKernel` compiles cleanly on cubecl 0.10 with a 12-way `#[comptime] match` | Standard Stack / Pattern 1 | If comptime match blows up compile or hits a cubecl limit, fall back to grouping (regression vs binary) or concrete kernels — parity-neutral per Claude's Discretion |
| A3 | Double-buffering (not in-place alias) is the safe default for the data→leaf map | Pitfall 3 | Verified at plan time; if alias is proven safe it saves a buffer, if not it prevents a corruption bug |
| A4 | Scoping the first driver slice to L2/continuous features (no RenewTreeOutput) is acceptable for the proving slice | Pitfall 4 | If L1/quantile must be in-slice, the plan adds the refit ordering — more work, no parity risk |
| A5 | Gating the `on_device_growth_supported()` flip so ROCm stays false with env unset satisfies D-09 without a runtime-type split | Pitfall 2 | If ROCm must be excluded even with env set, a runtime discriminator is needed — a design task, flagged |

**Not empty:** These 5 assumptions need confirmation during planning/discuss; A1 and A2 are the most plan-shaping.

## Open Questions (RESOLVED)

1. **`on_device_growth_supported()` flip on a shared `GpuBackend<R>`**
   - What we know: One generic impl serves ROCm/CUDA/WGPU; the env AND-gate provides one guard.
   - What's unclear: Whether the plan wants CUDA-only on-device growth (excluding ROCm even when env is set) or ROCm-included.
   - Recommendation: Gate the `true` behind `cuda_on_device_enabled()` for the merge gate; decide CUDA-vs-ROCm targeting at plan time (Pitfall 2). Note: the local "GPU" is a spoofed 8-CU APU (project memory `rocm-gfx1100-available`) — parity is valid, but this is a ROCm target, not real CUDA. STRUCTURE parity can be validated on ROCm; CUDA-specific behavior cannot be exercised locally.
   - **RESOLVED (20-03 Task 1):** The env-gated `on_device_growth_supported()` flip lands behind the `LGBM_CUDA_ON_DEVICE` AND-gate — ROCm/CUDA stays `false` with the env unset (byte-unchanged host path), so no runtime type-split is needed for the merge gate.

2. **Handle aliasing / batched `client.read(vec![h])` readback (D-01 discretion)**
   - What we know: The idiomatic `client.read(Vec<Handle>)` surfaced a launch-bound win in prior spikes (project memory `gpu-lazy-dispatch-deferred-sync-win`), but production launchers are already 1-launch/1-read.
   - What's unclear: In-place-alias vs ping-pong for the per-split leaf map.
   - Recommendation: Prefer double-buffer; A/B against the cpu anchor before committing (Pitfall 3).
   - **RESOLVED (20-03 Task 1):** Double-buffer is the safe execution-time default for the data→leaf `Handle` map; the in-place alias is A/B-tested against the cpu anchor at execution time, with the `num_leaves>2` STRUCTURE gate acting as the corruption catcher if the alias is unsafe.

3. **Which 12-metric goldens to capture vs cpu-fold-anchor**
   - What we know: 8 exist, 4 absent.
   - Recommendation: Capture the 4 (A1); it honors D-03's fidelity intent at near-zero cost.
   - **RESOLVED (20-00 Task 1):** Capture the 4 missing goldens (rmse/l2/l1/binary_logloss) against the pinned lightgbm 4.6 `.venv`; if the venv is unavailable the 4 anchor to the cpu-f64 fold with the reason documented in the SUMMARY (A1 fallback).

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| Rust + Cargo workspace | All build/test | ✓ | in-tree | — |
| cubecl (cpu + hip) | kernels + anchor fold | ✓ | 0.10 (pinned in workspace) | — |
| ROCm GPU (spoofed 8-CU APU) | on-device parity validation | ✓ | gfx1152 HSA-spoofed | CPU f64 anchor validates structure without GPU |
| lightgbm 4.6 Python venv (`.venv` at repo root) | capturing the 4 missing metric goldens | ✓ (per project memory) | 4.6 | cpu-f64-fold anchor for those 4 (A1) |
| Real discrete CUDA | CUDA-specific behavior | ✗ | — | ROCm APU validates parity; CUDA perf is Phase-23/Kaggle (out of scope) |

**Missing dependencies with no fallback:** None block this phase (parity is validatable on CPU + ROCm APU).
**Missing dependencies with fallback:** Real discrete CUDA — not needed for the parity gate (Phase-23 concern).

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | Rust built-in `#[test]` + `oracle-harness` integration tests |
| Config file | none (Cargo default) |
| Quick run command | `cargo test -p lgbm-compute` (kernel unit tests) |
| Full suite command | `cargo test --workspace` (the merge gate) |
| On-device gated run | `LGBM_CUDA_ON_DEVICE=1 cargo test -p oracle-harness` (resident/structure gates) |

### Phase Requirements → Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| ODL-16 | Resident score add/multiply matches host `score_` | unit + A/B | `cargo test -p oracle-harness score_updater` | ❌ Wave 0 (new cells) |
| ODL-16 | Resident-score A/B after multi-iter run | integration | `LGBM_CUDA_ON_DEVICE=1 cargo test -p oracle-harness resident_score_ab` | ❌ Wave 0 |
| ODL-17 | 12 pointwise metrics vs `lib_lightgbm` goldens | parity | `cargo test -p oracle-harness --test metric_parity` | ⚠ 8/12 goldens exist; extend + capture 4 (Pitfall 1) |
| ODL-17 | Unsupported metrics route to host (discriminator) | unit | `cargo test -p lgbm-compute metric_supported` | ❌ Wave 0 |
| ODL-18 | On-device tree STRUCTURE bit-exact to cpu anchor | parity | `LGBM_CUDA_ON_DEVICE=1 cargo test -p oracle-harness --test learner_parity on_device` | ✅ oracle exists (`assert_on_device_tree_matches_cpu_anchor`); activate cells |
| ODL-19 | No f64 per-row hot loop; env-unset byte-unchanged | grep + gate | `cargo test --workspace` (env unset) + `rg 'f64' new kernels` | ✅ merge gate exists |

### Sampling Rate
- **Per task commit:** `cargo test -p lgbm-compute` (fast kernel units)
- **Per wave merge:** `cargo test --workspace` (env unset — proves byte-unchanged) + `LGBM_CUDA_ON_DEVICE=1 cargo test -p oracle-harness`
- **Phase gate:** Full workspace green (env unset) AND all three D-06 parity layers green (env set) before `/gsd-verify-work`.

### Wave 0 Gaps
- [ ] `crates/oracle-harness/tests/metric_parity.rs` — add on-device metric cells (currently host-only replay) covering ODL-17
- [ ] `xtask/py/metric_oracle_capture.py` — capture rmse/l2/l1/binary_logloss goldens (Pitfall 1)
- [ ] `crates/oracle-harness/tests/learner_parity.rs` — activate the STRUCTURE gate cell with a real on-device tree (currently Slice-0 host-fallback stand-in at line 2445)
- [ ] New resident-score A/B test (ODL-16 D-06 layer 2) — no existing file
- [ ] New `metric_supported` discriminator unit test (mirror `device_objective.rs:141` supported/unsupported assertions)

## Security Domain

This is an internal numeric compute library with no auth/session/network surface introduced by this phase.

### Applicable ASVS Categories
| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | no | — |
| V3 Session Management | no | — |
| V4 Access Control | no | — |
| V5 Input Validation | yes (existing boundary discipline) | Existing pattern: validate `num_data >= 0`, buffer lengths, bin-type ∈ {8,16,32}, node/leaf index bounds — as `predict.rs` and `dataset.rs` already do (`DatasetError`/`ComputeError` at boundaries, never panic). New kernels' launchers must keep the same length/bounds guards before `launch_unchecked` (CMP-01 unsafe-confinement). |
| V6 Cryptography | no | — |

### Known Threat Patterns for the CubeCL kernel stack
| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| Out-of-bounds device read/write in a new kernel launcher | Tampering | Bounds-guard every launcher (`i < len`), size buffers exactly, keep `unsafe` confined to the launch site (existing `predict.rs`/`objective_regression.rs` pattern) |
| Integer overflow in `offset = num_data * tree_id` | Tampering | Use the existing `usize` offset arithmetic pattern (`score_updater.rs:64`); validate `tree_id >= 0` |
| Silent divergence from env-gate misconfiguration | (correctness, not security) | OnceLock env read (`cuda_on_device_enabled`) + AND-gate; workspace test with env unset proves byte-unchanged |

`security_block_on: high` — no high-severity security surface is introduced (no untrusted input, no auth, no crypto). The V5 discipline is a correctness/robustness carry-forward, not a new threat.

## Sources

### Primary (HIGH confidence — in-tree source, verified this session)
- `docs/cuda-kernel-design.md` §6, §11, §12/§12.1, §13, §14, §15, §16, §17 — port-source design reference (read directly)
- `crates/lgbm-boosting/src/score_updater.rs` — host score updater (add_constant/multiply_score/add_tree_train_path)
- `crates/lgbm-compute/src/lib.rs:1233-1317` — `on_device_growth_supported` / `grow_tree_on_device` seam + `cuda_on_device_enabled`
- `crates/lgbm-treelearner/src/learner.rs:280-306, 489-498, 696-724` — on-device fork + `on_device_eligible` AND-gate
- `crates/lgbm-treelearner/src/data_partition.rs:74` — `from_payload`; `crates/lgbm-dataset/src/dataset.rs:88` — `LeafPartitionLayout`
- `crates/lgbm-compute/src/kernels/primitives.rs:784` — `reduce_sum_f64_on`
- `crates/lgbm-compute/src/kernels/objective_regression.rs:356-415, 67-71` — `convert_output_on` + `CONVERT_*` modes
- `crates/lgbm-compute/src/kernels/objective_binary.rs` — sigmoid ConvertOutput
- `crates/lgbm-compute/src/device_objective.rs:114` — `device_objective_supported` (the D-05 mirror pattern)
- `crates/lgbm-compute/src/kernels/predict.rs:212` — `add_prediction_to_score_on_device`
- `crates/oracle-harness/tests/metric_parity.rs` + `tests/fixtures/metric/*` + `xtask/py/metric_oracle_capture.py:210-238` — metric goldens (verified 8/12 present)
- `crates/oracle-harness/tests/learner_parity.rs:2090-2272, 2445-2503` — STRUCTURE gate oracle + Slice-0 no-op tests
- `crates/lgbm-metric/src/{regression,binary}.rs` — host `MetricOnPointCUDA` math (Metric/BinaryMetric enums)
- Project memory: `spike-052` (5.4× f64 regression), `def-f8u-01` (never GPU-vs-GPU), `phase18-wr01-histarena-swap-aliasing`, `rocm-gfx1100-available` (spoofed APU), `phase8-python-venv`

### Secondary (MEDIUM confidence)
- `.planning/config.json` — nyquist_validation + security_enforcement toggles (both on)

### Tertiary (LOW confidence)
- None — all claims traced to in-tree source.

## Metadata

**Confidence breakdown:**
- Standard stack (reused kernels/signatures): HIGH — every signature read from source this session
- Architecture (driver sequencing, §16 order): HIGH — design doc + wired-but-dormant learner fork verified
- Pitfalls: HIGH for Pitfall 1 (verified goldens on disk) and Pitfall 2 (verified shared-impl comment); MEDIUM for Pitfall 3 (aliasing is a genuine open question, flagged for plan-time)
- Metric golden coverage: HIGH — enumerated fixtures + capture script directly

**Research date:** 2026-07-02
**Valid until:** ~2026-08-01 (stable internal codebase; re-verify if Phases 16–19 kernels are refactored)
