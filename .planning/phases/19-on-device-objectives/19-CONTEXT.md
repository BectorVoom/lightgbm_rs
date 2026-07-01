# Phase 19: On-Device Objectives - Context

**Gathered:** 2026-07-01
**Status:** Ready for planning

<domain>
## Phase Boundary

Port the **11 CUDA-supported objective functions** (§5 — `cuda_regression_objective.cu`,
`cuda_binary_objective.cu`, `cuda_multiclass_objective.cu`, `cuda_rank_objective.cu`) to
CubeCL, behind `LGBM_CUDA_ON_DEVICE`, so that **grad/hess computation, `ConvertOutput`
inverse-link, `BoostFromScore` init, and `RenewTreeOutput` leaf refit** all run on device
— eliminating the boosting layer's per-iteration gradient round-trip to host.

**Delivers (ODL-05, ODL-06, ODL-07, ODL-08):**
- **Regression-family (§5.1, ODL-05):** six `GetGradientsKernel` (L2, L1, Quantile, Huber,
  Fair, Poisson) `<bool USE_WEIGHT>`, `diff = score − label`; `ConvertOutput`
  (`sign(x)·x²` when `sqrt_`, `exp` for Poisson, else pass-through); `BoostFromScore`
  (mean via `ShuffleReduceSum`/`DotProd` for L2/Huber/Fair, median via `PercentileGlobal`
  for L1/Quantile; Poisson label-check via `ReduceSum`+`ReduceMin`); `RenewTreeOutput`
  (`RenewTreeOutputCUDAKernel_Regression{L1,Quantile}<USE_WEIGHT>`, **one block per leaf**,
  weighted/unweighted median via `PercentileDevice`).
- **Binary-logloss (§5.2, ODL-06):** `GetGradientsKernel_BinaryLogloss<USE_LABEL_WEIGHT,
  USE_WEIGHT>` (label=±1, `response = −label·σ/(1+exp(label·σ·score))`,
  `hess=|response|·(σ−|response|)`); two-stage `BoostFromScore` (`init = log(pavg/(1−pavg))/σ`,
  `pavg` clamped `[ε,1−ε]`); sigmoid `ConvertOutput`; `ResetOVACUDALabelKernel` for the
  one-vs-all label rewrite.
- **Multiclass (§5.3, ODL-07):** `GetGradientsKernel_MulticlassSoftmax<USE_WEIGHT>`
  (one thread per row, loops classes, **class-major `[k·num_data+i]`**, `SoftmaxCUDA` → p,
  `grad = p−1 if label==k else p`, `hess = ((K−1)/K)·p·(1−p)`, `double* cuda_softmax_buffer`
  scratch); per-row softmax `ConvertOutput`; `CUDAMulticlassOVA` reuses the binary path
  per class.
- **Ranking (§5.4, ODL-08):** `GetGradientsKernel_LambdarankNDCG<MAX_ITEM_GT_1024,
  NUM_RANK_LABEL>` + `_Sorted` (>2048); `GetGradientsKernel_RankXENDCG_{SharedMemory,
  GlobalMemory}`. `NUM_QUERY_PER_BLOCK=10`, block-per-query-group, `cuda_query_boundaries_`
  delimits queries, bitonic item ranking, per-item RNG (`CUDAPhi` + `cuda_item_rands_`,
  bit-identical stream). Neither rank objective provides `ConvertOutput`/`RenewTreeOutput`
  (base no-ops).

Everything **additive**; CPU / ROCm / existing-host-CUDA paths stay byte-unchanged, and the
full merge gate stays green with the env unset. **Anchored to the cubecl-cpu f64 fold**
(never GPU-vs-GPU, def-f8u-01). `on_device_growth_supported()` stays **false** this phase.

**Explicitly NOT in this phase:**
- **CUDA-unsupported objectives** (MAPE, Gamma, Gamma-deviance, Tweedie, cross-entropy
  xentropy/xentlambda, MAP/rank-MAP) → honestly **fall back to host** (SC #5, the
  discriminator). Their *metrics* may be device-supported (§12) but the *objectives* are not.
- **Wiring the device objective path into the live GBDT `train_one_iter` loop**
  (`boosting_on_cuda_` GetGradients/BoostFromScore) → **Phase 21** (end-to-end driver
  integration). This phase builds standalone, anchor-pinned kernels only (Decision B).
- **Swapping RenewTreeOutput/ConvertOutput onto the live host GBDT / Phase-18 device
  CUDATree flow** → **Phase 21** (Decision D — built as standalone device kernels here).
- **§11 score-updater constant ops + §12 metrics** → **Phase 20**.
- **Discretized/quantized objective path** (`RenewDiscretizedTreeLeavesKernel`, int16
  packed grad/hess) → **v2 (QGD)**.

</domain>

<decisions>
## Implementation Decisions

### Anchor & fixture strategy (ODL-05..08, SC #1–#5 — Decision A)
- **D-01: Two-tier anchor — cpu f64 fold is the deterministic anchor; real compiled
  `lib_lightgbm` captures cross-check fidelity for representative objectives.** The GPU
  objective kernels are anchor-pinned to the **cubecl-cpu f64 path** (the existing
  `lgbm-objective` transcription — L2, L1, Quantile, Huber, Fair, Poisson, binary,
  multiclass-softmax, lambdarank, RankXENDCG), keeping the D-12 "never GPU-vs-GPU"
  discipline. **ADDITIONALLY capture real compiled-`lib_lightgbm` grad/hess/init-score
  goldens for a representative objective per family** — **L2** (regression), **binary**,
  **multiclass-softmax**, **lambdarank** — as a *fidelity* cross-check. Rationale: objectives
  are pure elementwise math, so a real C++ capture is cheap here and it directly answers the
  `on-device-kernel-goldens-are-retranscriptions` caveat (host-transcription-agrees-with-
  kernel-transcription proves agreement, not reference fidelity). *(Host-f64-only was
  rejected as transcription-agrees-with-transcription; fresh C++ goldens for all 11 was
  rejected as disproportionate capture cost — the per-family representative gives fidelity
  coverage without 11× capture.)*

### Boosting-layer integration depth (ODL-05..08 — Decision B)
- **D-02: Build STANDALONE anchor-pinned device-objective kernels this phase; the
  `boosting_on_cuda` GetGradients/BoostFromScore wiring into the live GBDT loop is deferred
  to Phase 21.** `gbdt.rs` already calls `objective.get_gradients` / `boost_from_score` /
  RenewTreeOutput on host; `on_device_growth_supported()` stays **false** until Phase 21
  (the designated end-to-end driver phase). This phase adds the device kernels + a thin
  device-objective module/trait, unit-anchor-pinned; it does NOT touch the byte-unchanged
  default boosting path. Mirrors the Phase 14–18 chain (kernels behind the seam, wired by the
  driver phase). *(Wiring `boosting_on_cuda` now was rejected — it front-runs Phase 21 and
  touches the boosting seam ahead of the driver.)*

### Ranking objective scope (ODL-08, §5.4 — Decision C)
- **D-03: Build BOTH the shared-memory and the >2048 global-memory variants for BOTH
  LambdaRank-NDCG and RankXENDCG this phase.** Full §5.4 fidelity:
  `GetGradientsKernel_LambdarankNDCG<MAX_ITEM_GT_1024,…>` + `_Sorted`;
  `GetGradientsKernel_RankXENDCG_SharedMemory<SHARED_MEMORY_SIZE>` +
  `_GlobalMemory`. The >2048 RankXENDCG global path stashes intermediates in the hessian
  output buffer + `cuda_params_buffer` — fiddly but de-risked by the existing Phase-14
  `bitonic_argsort_global_on` / `bitonic_argsort_items_on` skeletons. *(Shared-bounded-first
  with the >2048 variant deferred was considered and declined — the global argsort skeleton
  already exists, so completing the >2048 path now avoids a host-fallback hole for large
  queries.)*

### Renew/Convert target surface (ODL-05, §5.1 — Decision D)
- **D-04: Build RenewTreeOutput (L1/quantile, one-block-per-leaf via `PercentileDevice`)
  and ConvertOutput inverse-link as STANDALONE device kernels over device buffers**
  (leaf-value array + score array), anchor-pinned in isolation. Do **NOT** swap them onto
  the live host GBDT path or the Phase-18 device `CUDATree` this phase — that integration
  lands in **Phase 21**, consistent with D-02. *(Wiring onto the Phase-18 CUDATree + resident
  score now was rejected — it couples Phase 19 to Phase 18's device tree ahead of the driver.)*

### Carried forward from Phases 14–18 (NOT re-litigated — hard discipline)
- **D-05:** Anchor every numeric output to the **cubecl-cpu f64 fold**; **never GPU-vs-GPU**
  (def-f8u-01). The atomic-ordering nondeterminism in **binary `BoostFromScore`** (`atomicAdd`
  label/weight sums) and **lambdarank** (`atomicAdd_block` per-item λ) is the **documented
  f32-vs-f64 residual** the ROCm gate tolerates — pin to the f64 anchor with **tie-aware /
  ~1e-6–1e-5 envelope** assertions, never to a second GPU run.
- **D-06:** `LGBM_CUDA_ON_DEVICE` **OFF by default**; CPU / ROCm / existing-host-CUDA paths
  **byte-unchanged**; full merge gate green and unchanged (ODL-19 — hard merge gate).
- **D-07:** **NO f64 per-row hot loops** in new kernels (5.4× consumer-NVIDIA f64 regression,
  spike-052); f64 only where the reference uses it — the `double* score` accumulator, the
  `double* cuda_softmax_buffer`, and scalar `BoostFromScore`/`RenewTreeOutput` reduction math
  (§17 — load-bearing).
- **D-08:** **Reuse the Phase-14 device primitives** — weighted/unweighted percentile
  (`percentile_{un,}weighted_f32_on`), `reduce_{sum,max,min}_f64_on`, `dot_product_f64_on`,
  single-block + global + per-segment bitonic argsort — and the **`CUDARandom` LCG**
  (`draw_next_float_on`, bit-identical stream) for the per-item ranking randoms. Do **NOT**
  rebuild these; the objective kernels COMPOSE them.
- **D-09:** **Pre-allocate scratch ONCE outside the hot loop** (softmax buffer, per-block
  reduction partials, item-rand buffer, rank-params buffer) — no per-call in-kernel device
  alloc (Phase-17 D-11 / Phase-18 D-15 pattern).

### Claude's Discretion
- Exact CubeCL module placement — likely a new `objective.rs` (or per-family
  `objective_{regression,binary,multiclass,rank}.rs`) in `crates/lgbm-compute/src/kernels/`,
  plus a device-objective trait/enum mirroring the host `lgbm-objective` surface.
- Whether the six regression grad kernels are one comptime-generic `#[cube]` (branch on an
  objective enum) or six kernels — parity-neutral as long as `diff`/hess math matches §5.1.
- Whether `CUDAMulticlassOVA` literally reuses the binary kernel per class or is a thin
  softmax-off variant — parity-neutral (reference reuses the binary path).
- Block/geometry constants (`GET_GRADIENTS_BLOCK_SIZE_*=1024`, `NUM_QUERY_PER_BLOCK=10`,
  `SHARED_MEMORY_SIZE` 1024/2048 dispatch) — start from the faithful C++ constants; APU-aware
  autotune is a deferred perf option, parity-neutral.
- Which additional objectives (beyond the four representatives) also get a real C++ capture,
  if cheap to add during fixture work — the four are the floor, not a ceiling.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Port-source design reference (READ FIRST)
- `docs/cuda-kernel-design.md` §5 — **Objective Functions**: the 11-objective inventory,
  the `CUDAObjectiveInterface<HOST_OBJECTIVE>` shape (`GetGradients`/`ConvertOutputCUDA`/
  `LaunchCalcInitScoreKernel`/`RenewTreeOutputCUDA`), and per-family kernel tables:
  §5.1 regression (six grad kernels + Convert + Renew + init-score), §5.2 binary
  (grad + two-stage BoostFromScore + sigmoid Convert + OVA label reset), §5.3 multiclass
  (softmax grad, class-major layout, softmax Convert), §5.4 rank (LambdaRank-NDCG +
  `_Sorted`, RankXENDCG shared/global, per-item RNG via `CUDAPhi`/`cuda_item_rands_`).
- `docs/cuda-kernel-design.md` §16 — **End-to-End Sequencing**: step 1
  `Objective::GetGradients(scores → grad/hess)` and step 6 (optional)
  `Objective::RenewTreeOutput` (L1/quantile leaf refit) — where the objective sits in the loop.
- `docs/cuda-kernel-design.md` §17 — **Port considerations**: `double` accumulator
  load-bearing; template-flag → CubeCL comptime; f32 accumulation-order fidelity.
- `.planning/REFERENCE_MANIFEST.md` — v1.1 C++ port-source map + CUDA-support boundaries
  (confirms the 11-supported / MAPE-Gamma-Tweedie-xentropy-MAP-unsupported split).

### CubeCL API
- `/home/user/Documents/workspace/cubecl_manual/manual/cubecl/13_memory_preallocation.md` —
  `client.empty` / reuse-once (D-09 scratch pre-allocation).
- cubecl 0.10 LDS idiom (`SharedMemory::new` / `sync_cube()` / shared atomics) as used in
  `crates/lgbm-compute/src/kernels/primitives.rs` — the reductions/percentile/bitonic the
  objective kernels compose (D-08).

### Existing code to reuse / anchor against (already in git — DO NOT rebuild)
- `crates/lgbm-objective/src/{regression,binary,multiclass,rank,percentile}.rs` — the host
  C++ transcription (L2/L1/Quantile/Huber/Fair/Poisson, binary, softmax, lambdarank,
  RankXENDCG); the **cpu f64 deterministic anchor** for D-01. `lib.rs` exposes the objective
  trait mirrored by the device module.
- `crates/lgbm-compute/src/kernels/primitives.rs` — percentile (weighted/unweighted),
  `reduce_{sum,max,min}_f64_on`, `dot_product_f64_on`, bitonic argsort (single-block +
  `bitonic_argsort_global_on` + `bitonic_argsort_items_on`), prefix-sum (D-08).
- `crates/lgbm-compute/src/kernels/random.rs` — `CUDARandom` LCG (`draw_next_float_on`,
  bit-identical stream) for §5.4 per-item ranking randoms.
- `crates/lgbm-boosting/src/gbdt.rs` — the host `get_gradients` / `boost_from_score` /
  RenewTreeOutput call sites (the seam Phase 21 wires the device path into; NOT touched here,
  D-02).
- `crates/lgbm-compute/src/lib.rs` — `Backend` seam; `on_device_growth_supported()` stays
  **false** this phase.
- `crates/oracle-harness/` — the anchor-test harness (`fixtures/`, `tests/rank_parity.rs`,
  `tests/learner_parity.rs`); the real-`lib_lightgbm` capture landing point for D-01 goldens.

### Prior-phase context (discipline carried forward)
- `.planning/phases/18-on-device-data-partition-tree-mutation-prediction/18-CONTEXT.md` —
  the "cpu f64 anchor, never GPU-vs-GPU / new-goldens cross-check" pattern (D-01 mirrors
  D-11), pre-allocate-once (D-09), byte-unchanged env gate.
- `.planning/phases/14-foundation-shared-device-primitives-device-structs-rng/14-CONTEXT.md`
  — the primitives + `CUDARandom` LCG this phase composes (D-08), the anchor conventions.

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- **`lgbm-objective` host implementations** — the exact grad/hess/Convert/BoostFromScore/
  RenewTreeOutput math per objective, already C++-validated in `learner_parity` /
  `rank_parity`. This is BOTH the transcription to port to `#[cube]` AND the cpu f64 anchor
  (D-01).
- **`primitives.rs`** — percentile, reduce sum/max/min, dot-product, bitonic argsort (all
  three depths), prefix-sum. The `BoostFromScore` mean/median, the Poisson label-check, the
  RenewTreeOutput per-leaf percentile, and the §5.4 item ranking all compose these (D-08).
- **`random.rs` `CUDARandom` LCG** — the bit-identical draw stream for §5.4 per-item randoms
  (`GenerateItemRands` analog).

### Established Patterns
- **Anchor to cpu f64 fold, never GPU-vs-GPU; new C++ goldens cross-check** (def-f8u-01 /
  Phase-18 D-11) — D-01, D-05.
- **Standalone kernels this phase, driver wiring in the designated phase** (Phase 14–18
  chain) — D-02, D-04.
- **No f64 per-row hot loops; f64 only where the reference uses it** (spike-052) — D-07.
- **Pre-allocate scratch once outside the hot loop** — D-09.
- **Template-flag `<USE_WEIGHT>` etc. → CubeCL comptime** — regression/binary kernel fan-out.

### Integration Points
- New device-objective kernels live in `lgbm-compute` (`kernels/`), composing the Phase-14
  primitives + `CUDARandom`, anchored to `lgbm-objective` f64 + real-`lib_lightgbm` goldens
  via `oracle-harness`. **Consumed by Phase 21** (the `boosting_on_cuda` driver wires
  GetGradients/BoostFromScore/RenewTreeOutput into the live GBDT loop) and **Phase 20**
  (ConvertOutput inverse-link at the metric/score boundary). Reached only when
  `LGBM_CUDA_ON_DEVICE=1`; the default host path is byte-unchanged (D-06).

</code_context>

<specifics>
## Specific Ideas

- **The retranscription caveat is the reason for D-01's real-C++ captures**: per the memory
  note `on-device-kernel-goldens-are-retranscriptions`, Phase 17/18 goldens are host
  re-transcriptions, so parity there proves transcriptions agree, not reference fidelity.
  Objectives are pure elementwise math → a real compiled-`lib_lightgbm` capture is cheap and
  buys genuine fidelity. Capture at minimum one representative per family (L2, binary,
  softmax, lambdarank).
- **Atomic-ordering nondeterminism is EXPECTED, not a bug** (§5.2 binary BoostFromScore
  `atomicAdd` sums; §5.4 lambdarank `atomicAdd_block` λ): it is the documented f32-vs-f64
  residual the ROCm gate tolerates. Pin to the f64 anchor with tie-aware / envelope asserts;
  never assert two GPU runs equal.
- **Class-major `[k·num_data+i]` layout is load-bearing** for multiclass (§5.3) — scores,
  grads, hessians all stride by class. Keep it faithful so the accumulation/prediction order
  matches the reference.
- **RenewTreeOutput is one-block-per-leaf** (§5.1) via `PercentileDevice` — reuse the Phase-14
  per-segment percentile primitive; it operates on leaf-local row subsets.
- **Rank objectives have no Convert/Renew** (base no-ops) — do not synthesize them.

</specifics>

<deferred>
## Deferred Ideas

- **Wiring the device objective path into the live GBDT loop** (`boosting_on_cuda`
  GetGradients/BoostFromScore + RenewTreeOutput/ConvertOutput on the device CUDATree/score)
  → **Phase 21** (end-to-end driver integration). D-02, D-04.
- **§11 score-updater constant ops + §12 pointwise metrics** → **Phase 20**.
- **CUDA-unsupported objectives** (MAPE / Gamma / Gamma-deviance / Tweedie / xentropy /
  xentlambda / MAP / rank-MAP) → **host fallback**, never ported (SC #5).
- **Discretized/quantized objective path** (`RenewDiscretizedTreeLeavesKernel`, int16 packed
  grad/hess de-quant) → **v2 (QGD)**.
- **APU-aware autotune of objective/rank block geometry** → deferred perf option
  (parity-neutral, Claude's Discretion).
- **Real C++ captures for the remaining 7 objectives beyond the four representatives** → add
  opportunistically if cheap during fixture work; otherwise a later hardening pass.

### Reviewed Todos (not folded)
None — no pending todos matched this phase's scope.

</deferred>

---

*Phase: 19-on-device-objectives*
*Context gathered: 2026-07-01*
