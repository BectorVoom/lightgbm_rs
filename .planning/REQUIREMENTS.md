# Requirements: LightGBM-rs — Milestone v1.1 (CUDA On-Device Training Backend)

**Defined:** 2026-06-28 · **Rewritten:** 2026-06-29 (re-derived subsystem-by-subsystem from `docs/cuda-kernel-design.md`)
**Core Value:** For identical inputs/config, reproduce C++ LightGBM within ~1e-6 on every backend (f32 end-to-end); the cubecl-cpu f64 fold is the bit-exact merge gate. v1.1 ports the **full single-GPU CUDA training pipeline on-device** — all training state resident, host only orchestrates — WITHOUT weakening that contract.

## Milestone v1.1 Requirements

Scope (locked 2026-06-29): the **entire** on-device CUDA training path mirroring `CUDASingleGPUTreeLearner` + the boosting-layer device path, grounded in `docs/cuda-kernel-design.md`. Every requirement is **anchor-gated** (structure bit-exact to the cpu f64 fold; CUDA/ROCm f32 within ~1e-6 — never two nondeterministic GPU paths compared to each other) and **additive** (CPU / ROCm / existing-host-CUDA paths stay byte-unchanged, off by default behind `LGBM_CUDA_ON_DEVICE`). `§` references are sections of the design doc.

### Foundation — shared primitives & device structs

- [ ] **ODL-01**: The shared device-primitive kernels every subsystem builds on are ported as reusable CubeCL kernels — block + multi-kernel global **prefix-sum** (inclusive/exclusive), **shuffle reductions** (sum/max/min, dot-product), **bitonic argsort** (single- and multi-block, index-only — never moves values), and **weighted/unweighted percentile** — each anchor-pinned where it carries numeric output. (§2.4; §17 "port first".)
- [ ] **ODL-02**: A CubeCL-safe device **split-record** (a pre-allocated `CUDASplitInfo` analog — NO per-split in-kernel device allocation, which has no clean CubeCL analog) and a **`CUDARandom` LCG** whose stream is bit-identical to the host `Random` (extra-trees / sampling / per-item-rand parity). (§15.)

### Device dataset

- [ ] **ODL-03**: An on-device **columnar binned dataset** (u8/16/32 bin-width dispatch; dense + sparse CSR) resident on device, carrying the **feature-partition layout** the histogram kernel is built around — features grouped so one partition's histogram fits shared memory, a too-wide column becoming its own large-bin partition (→ global-memory path). (§3, §13.)
- [ ] **ODL-04**: On-device **row-subset gather** (a `CopySubrow` analog) builds the bagging / GOSS subset dataset on device, anchor-pinned to the host subset-selection draw sequence. (§3.)

### On-device objectives (§5)

- [ ] **ODL-05**: On-device **regression-family** gradients/hessians (L2, L1, Quantile, Huber, Fair, Poisson) + `ConvertOutput` inverse-link + `BoostFromScore` (mean via reduce / median via percentile) + `RenewTreeOutput` (median/quantile leaf refit, one block per leaf), anchor-pinned. (§5.1.)
- [ ] **ODL-06**: On-device **binary-logloss** gradients/hessians + `BoostFromScore` (label-prior logit init) + sigmoid `ConvertOutput`; one-vs-all label reset for OVA, anchor-pinned. (§5.2.)
- [ ] **ODL-07**: On-device **multiclass** gradients/hessians — softmax + one-vs-all, class-major `[k·num_data+i]` layout, anchor-pinned. (§5.3.)
- [ ] **ODL-08**: On-device **ranking** gradients/hessians — LambdaRank-NDCG + RankXENDCG, per-query block layout, bitonic item ranking, per-item RNG (bit-identical stream), anchor-pinned. (§5.4.)

### Histogram constructor (§7) — the hot path

- [ ] **ODL-09**: On-device **histogram build** (dense + sparse × shared-memory + global-memory spill), two-tier atomic accumulation (block-local then cross-block merge), on the f32 / **u64 fixed-point** accumulation path (NO f64 per-row hot loop — ODL-19), anchor-pinned to the cpu f64 fold. (§7.1–7.4.)
- [ ] **ODL-10**: The **subtraction trick on device** — build-smaller-only, `FixHistogram` (most-frequent-bin omission repair via leaf-total minus scanned sum), `SubtractHistogram` (larger = parent − smaller) realized through **`hist_t**` pointer rotation** (larger child inherits the parent buffer; smaller child gets a fresh arena slot), no bulk histogram copy. Preserved as a **correctness** requirement — building the larger child directly takes a different rounding path. (§7.5, §17.)

### Best-split finder (§8)

- [ ] **ODL-11**: On-device **per-feature split evaluation** (stage 1, one block per (leaf,feature) task) — block prefix-sum → cumulative left/right sums, count recovery via `cnt_factor`, min-data / min-sum-hessian guards, gain math, forward/reverse default-bin scan, block argmax → per-task split record. (§8.1, numerical core.)
- [ ] **ODL-12**: On-device **cross-feature reduce (stage 2) + cross-leaf argmax (stage 3)** producing the chosen `(leaf, feature, threshold, default_left)` with a single small scalar readback per split (the 8-int buffer), and **tie-aware `default_left`** parity to the cpu anchor. (§8.2–8.3.)

### Data partition, tree & prediction (§9, §10)

- [ ] **ODL-13**: On-device **data partition** — `mark → prefix-sum → scatter` row permutation (**never sorting**) into two contiguous child ranges, the data-index→leaf-index map, and the `SplitTreeStructure` **histogram-pool pointer swap**; the resulting row order matches the reference so per-leaf f32 accumulation order is identical (§17). (§9.)
- [ ] **ODL-14**: On-device **tree mutation** — `Split` writing the device tree arrays, ordered **before** partition (returns `right_leaf_index` the partition consumes), plus `Shrinkage` / `AddBias`, anchor-pinned to the host tree structure. (§10, §1 ordering note.)
- [ ] **ODL-15**: On-device **prediction** — the tree-walk `AddPredictionToScore` kernel over the device columnar dataset (numeric threshold + missing/`default_left` handling, categorical bitset membership), within ~1e-6 + objective inverse-link. (§10.)

### Score updater & metrics (§11, §12)

- [ ] **ODL-16**: On-device **score update** — resident cumulative `cuda_score_`, constant add / multiply (init score / shrinkage / no-split single-leaf / DART rescale), replacing the host `add_prediction_to_score` scatter, with a host-mirror toggle for non-resident consumers. (§11.)
- [ ] **ODL-17**: On-device **pointwise metric evaluation** — `EvalKernel` + two-stage reduction over the 12 supported regression/binary losses, anchor-pinned; the CUDA-unsupported metrics (AUC / NDCG / MAP / multiclass / cross-entropy) honestly fall back to host. (§12.)

### Driver integration, feature coverage & parity

- [ ] **ODL-18**: The on-device **single-GPU tree-learner driver** orchestrates the per-leaf grow loop end-to-end on device — root init → build/subtract → best-split → tree split → partition, repeated up to `num_leaves−1` (break on `best_leaf == −1`) — and reconstitutes into the `(Tree, DataPartition)` the boosting loop consumes; STRUCTURE **bit-exact** to the cpu f64 anchor (tie-aware `default_left`), leaf values within ~1e-5. (§6, §16; the continuous-feature path is the first proving slice.)
- [ ] **ODL-19**: Every new kernel keeps **f32 + the u64 fixed-point build with NO f64 per-row hot loops** (verified by grep + per-tree-ms, not a 6× sweep; the measured 5.4× consumer-NVIDIA f64 regression, spike-052); f64 permitted only in scalar/gain math where the reference uses it. CPU / ROCm / existing-host-CUDA paths stay **byte-unchanged** with `LGBM_CUDA_ON_DEVICE` unset. (§17 — the hard merge gate.)
- [ ] **ODL-22**: On-device **categorical splits** end-to-end — bitset construction (`SetRealThreshold` + length + construct, §6.3), categorical split evaluation (one-hot + many-vs-many bitonic-sorted, §8.1), categorical partition membership (§9), and `SplitCategorical` tree mutation (§10) — anchor-pinned, via the pre-allocated bitset representation (no per-`SplitInfo` device alloc).

### Performance & rollout (the DoD)

- [ ] **ODL-20**: A real-CUDA **Kaggle A/B harness** measures the on-device path's `device_launches/tree` (target well below the 8,570 / 100-trees baseline) and the lgb_rs / official wall-clock ratio at 500k×50 and a wide shape.
- [ ] **ODL-21**: The on-device learner becomes the **DEFAULT** CUDA tree-learner path — contingent on anchor-pinned parity (~1e-6) AND not-slower than the current host-CUDA path on the Kaggle A/B — with the host path retained as the `LGBM_CUDA_ON_DEVICE=0` off-switch fallback.

## v2 Requirements

Deferred to a future milestone. Tracked but not in this roadmap.

### On-Device Quantized Training (the gradient discretizer + integer path)

- **QGD-01**: On-device **gradient discretizer** (§4) — `f32` grad/hess → `int16` packed (`ReduceMinMax` → scales → stochastic/deterministic `Discretize`), the optional quantized-training front-end.
- **QGD-02**: **Discretized histogram + split-finder + de-quant** path (§7.3, §8.1 discretized inner, §6.1 K3/K4) — integer arithmetic end-to-end (the naturally bit-exact GPU target), with per-leaf 16/32-bit histogram-width selection and the bit-width-change subtract.
- **QGD-03**: On-device integration of the opt-in quantized-grad training mode (Phase 10 `use_quantized_grad`) with the on-device learner. *(was ODL-13 in the pre-rewrite v1.1 scope.)*

## Out of Scope

Explicitly excluded. Documented to prevent scope creep.

| Feature | Reason |
|---------|--------|
| Device-side megakernel / in-kernel grow loop | cubecl 0.10 has no global grid barrier; the reference `CUDASingleGPUTreeLearner` is itself host-driven (kernel boundaries ARE the barrier). The win is fewer/larger launches over resident state, not a megakernel. |
| On-device quantized training (discretizer + integer path) | Deferred to v2 (QGD-01..03). The f32 / u64-fixed-point standard path proves the architecture first. |
| Multi-GPU on-device learning | Single-GPU parity first; distributed is a separate milestone. |
| Changing CPU or ROCm routing | v1.1 is additive and CUDA-targeted; the CPU f64 anchor and ROCm host-partition paths stay byte-unchanged. |
| Per-`SplitInfo` device `cudaMalloc` for categorical thresholds | No clean CubeCL analog; ODL-02/ODL-22 use pre-allocation instead. |
| CUDA-unsupported objectives/metrics on device (MAPE, Gamma, Tweedie, xentropy; AUC/NDCG/MAP/multiclass metrics) | The reference itself keeps these CPU-only even with `device=cuda` (§5, §12.1); they fall back to host. |

## Traceability

Which phases cover which requirements. **Filled by the roadmapper** during roadmap creation (phases renumber from 14).

| Requirement | Phase | Status |
|-------------|-------|--------|
| ODL-01 | TBD | Pending |
| ODL-02 | TBD | Pending |
| ODL-03 | TBD | Pending |
| ODL-04 | TBD | Pending |
| ODL-05 | TBD | Pending |
| ODL-06 | TBD | Pending |
| ODL-07 | TBD | Pending |
| ODL-08 | TBD | Pending |
| ODL-09 | TBD | Pending |
| ODL-10 | TBD | Pending |
| ODL-11 | TBD | Pending |
| ODL-12 | TBD | Pending |
| ODL-13 | TBD | Pending |
| ODL-14 | TBD | Pending |
| ODL-15 | TBD | Pending |
| ODL-16 | TBD | Pending |
| ODL-17 | TBD | Pending |
| ODL-18 | TBD | Pending |
| ODL-19 | TBD | Pending |
| ODL-20 | TBD | Pending |
| ODL-21 | TBD | Pending |
| ODL-22 | TBD | Pending |

**Coverage:**

- v1.1 requirements: 22 total
- Mapped to phases: 0 (roadmapper fills)
- Unmapped: 22 (roadmapper fills)

---
*Requirements defined: 2026-06-28 (milestone v1.1); rewritten 2026-06-29 — full on-device pipeline, design-doc-grounded, quantized deferred to v2.*
