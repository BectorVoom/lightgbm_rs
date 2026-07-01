# Phase 19: On-Device Objectives - Research

**Researched:** 2026-07-01
**Domain:** CubeCL port of the 11 CUDA-supported LightGBM objective functions (grad/hess, ConvertOutput, BoostFromScore, RenewTreeOutput), anchor-pinned to the cpu f64 fold
**Confidence:** HIGH (every CONTEXT claim cross-checked against repo code and `docs/cuda-kernel-design.md`)

## Summary

Phase 19 is a **kernel-port phase with an unusually settled WHAT** — the nine locked
decisions D-01..D-09 fix the anchor strategy, the integration depth (standalone, no GBDT
wiring), the ranking scope (both shared + >2048 global paths), and the primitives-reuse
discipline. The risk this research de-risks is entirely in the **HOW**: (1) how each CUDA
templated kernel maps onto the existing CubeCL primitives, (2) the real-`lib_lightgbm`
golden-capture mechanism (which turns out to be **already built and already run for 3 of
the 4 representatives**), and (3) the validation architecture the Nyquist VALIDATION.md
lifts.

The single most valuable finding: **the grad/hess goldens D-01 asks for already exist in
git** for L2, binary, and multiclass-softmax — captured from the real `lib_lightgbm` 4.6
pip wheel by the Phase-6 `xtask/py/boosting_oracle_capture.py` harness via the
*score-derivation route* (`grad/hess` computed from per-iteration raw scores using the
objective math, cross-checked against the real binary's scores/model). Only the
**lambdarank grad/hess golden is a genuine gap** and needs one new capture. This
collapses the fixture cost of D-01 dramatically.

The second most valuable finding: two CONTEXT/D-08 claims about the reusable primitives
are **subtly imprecise and the planner must not take them at face value**. The percentile
and bitonic-argsort primitives in `primitives.rs` are **host-orchestrated composites over
host `&[f32]` slices**, not the CUDA "one-block-per-leaf" / "block-per-query-group" device
kernels the reference uses. There is **no `percentile_device` per-segment primitive** —
only a whole-array percentile and a per-*segment argsort*. RenewTreeOutput's per-leaf
median and RankXENDCG's per-query softmax must therefore be composed by **host looping over
leaves/queries calling the existing whole-array primitives**, or by writing new
block-per-segment device kernels. This is a real design fork the planner must resolve
explicitly per family.

**Primary recommendation:** Slice the phase by objective family (ODL-05→08 = four plan
groups), reuse the existing `boosting/*_gh_iter*.txt` goldens + `assert_gradients` /
`parse_gh_golden` comparator harness verbatim, capture exactly one new golden
(`lambdarank_gh_*`), and treat the percentile/argsort primitives as **host-orchestrated
building blocks** (not device block-kernels) unless a plan explicitly opts to write a new
device kernel. Anchor every numeric output to the cpu f64 fold via
`compare_exact_u32` (bit-exact where determinism holds) or `compare_within(ORACLE_TOL=1e-6)`
(atomic-order / transcendental cells).

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Per-row grad/hess (all families) | Device kernel (`lgbm-compute/kernels/`) | cpu f64 anchor (`lgbm-objective`) | Elementwise one-thread-per-row; the GPU hot path §5 |
| BoostFromScore init (mean/median/logit) | Device reduction/percentile | Host scalar finalize (`<<<1,1>>>` analog) | §5.1/§5.2 — reduce on device, scalar math on host |
| ConvertOutput inverse-link | Device elementwise | `lgbm_model::ObjectiveKind::convert_output` (host reference) | Predict-side transform; standalone device kernel this phase |
| RenewTreeOutput (L1/quantile median) | Device percentile per-leaf | Host loop orchestration | One-block-per-leaf; composes percentile primitive |
| Per-item ranking RNG | Device `CUDARandom` LCG (`random.rs`) | — | Bit-identical stream is load-bearing (D-08) |
| Golden capture (fidelity cross-check) | `xtask/py/*_oracle_capture.py` + real `lib_lightgbm` 4.6 wheel | `oracle-harness` fixtures | Real-binary goldens land in `tests/fixtures/` |
| Parity assertion | `oracle-harness` comparator + tests | — | cpu f64 anchor is the hard merge gate |

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

- **D-01 (two-tier anchor):** GPU objective kernels anchor-pinned to the **cubecl-cpu f64
  path** (the `lgbm-objective` transcription); **never GPU-vs-GPU**. ADDITIONALLY capture
  real compiled-`lib_lightgbm` grad/hess/init-score goldens for **one representative per
  family — L2, binary, multiclass-softmax, lambdarank** — as a *fidelity* cross-check.
- **D-02 (integration depth):** Build **standalone** anchor-pinned device-objective kernels
  + a thin device-objective module/trait. The `boosting_on_cuda` GetGradients/BoostFromScore
  wiring into the live GBDT loop is **deferred to Phase 21**. `on_device_growth_supported()`
  stays **false**. Do NOT touch the byte-unchanged default boosting path.
- **D-03 (ranking scope):** Build BOTH shared-memory AND the >2048 global-memory variants
  for BOTH LambdaRank-NDCG and RankXENDCG this phase.
- **D-04 (renew/convert surface):** Build RenewTreeOutput (L1/quantile, one-block-per-leaf
  via PercentileDevice) and ConvertOutput inverse-link as **standalone device kernels over
  device buffers**, anchor-pinned in isolation. Do NOT swap them onto the live host GBDT or
  the Phase-18 device CUDATree — that lands in Phase 21.
- **D-05:** Anchor every numeric output to the cpu f64 fold; never GPU-vs-GPU. Atomic-order
  nondeterminism in **binary BoostFromScore** (`atomicAdd`) and **lambdarank**
  (`atomicAdd_block`) is the documented f32-vs-f64 residual — pin to the f64 anchor with
  **tie-aware / ~1e-6–1e-5 envelope** assertions, never to a second GPU run.
- **D-06:** `LGBM_CUDA_ON_DEVICE` **OFF by default**; CPU / ROCm / existing-host-CUDA paths
  byte-unchanged; full merge gate green and unchanged (ODL-19 — hard merge gate).
- **D-07:** **NO f64 per-row hot loops** in new kernels (spike-052, 5.4× consumer-NVIDIA f64
  regression); f64 only where the reference uses it — the `double* score` accumulator, the
  `double* cuda_softmax_buffer`, and scalar BoostFromScore/RenewTreeOutput reduction math.
- **D-08:** **Reuse the Phase-14 device primitives** (percentile, `reduce_{sum,max,min}_f64`,
  `dot_product_f64`, bitonic argsort single-block/global/per-segment) and the `CUDARandom`
  LCG (`draw_next_float_on`). Do NOT rebuild — the objective kernels COMPOSE them.
- **D-09:** **Pre-allocate scratch ONCE outside the hot loop** (softmax buffer, per-block
  reduction partials, item-rand buffer, rank-params buffer) — no per-call in-kernel device
  alloc.

### Claude's Discretion

- Exact CubeCL module placement — likely a new `objective.rs` or per-family
  `objective_{regression,binary,multiclass,rank}.rs` in `crates/lgbm-compute/src/kernels/`,
  plus a device-objective trait/enum mirroring the host `lgbm-objective` surface.
- Whether the six regression grad kernels are one comptime-generic `#[cube]` (branch on an
  objective enum) or six kernels — parity-neutral as long as `diff`/hess math matches §5.1.
- Whether `CUDAMulticlassOVA` literally reuses the binary kernel per class or is a thin
  softmax-off variant — parity-neutral.
- Block/geometry constants (`GET_GRADIENTS_BLOCK_SIZE_*=1024`, `NUM_QUERY_PER_BLOCK=10`,
  `SHARED_MEMORY_SIZE` 1024/2048 dispatch) — start from the faithful C++ constants;
  APU-aware autotune is a deferred perf option.
- Which additional objectives (beyond the four representatives) also get a real C++ capture
  — the four are the floor, not a ceiling.

### Deferred Ideas (OUT OF SCOPE)

- Wiring the device objective path into the live GBDT loop (`boosting_on_cuda`) → **Phase 21**.
- §11 score-updater constant ops + §12 pointwise metrics → **Phase 20**.
- CUDA-unsupported objectives (MAPE / Gamma / Gamma-deviance / Tweedie / xentropy / xentlambda
  / MAP / rank-MAP) → **host fallback**, never ported (SC #5).
- Discretized/quantized objective path (`RenewDiscretizedTreeLeavesKernel`, int16 packed
  grad/hess) → **v2 (QGD)**.
- APU-aware autotune of objective/rank block geometry → deferred perf option.
- Real C++ captures for the remaining 7 objectives beyond the four representatives → add
  opportunistically if cheap during fixture work.
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| **ODL-05** | On-device regression-family grad/hess (L2, L1, Quantile, Huber, Fair, Poisson) + ConvertOutput inverse-link + BoostFromScore (mean via reduce / median via percentile) + RenewTreeOutput (median/quantile leaf refit, one block per leaf), anchor-pinned. (§5.1) | Grad/hess math table verified §5.1 (design doc L396-401); host anchor = `lgbm-objective/src/regression.rs::get_gradients` (L380) + `boost_from_score` (L544); primitives `reduce_sum_f64_on`/`dot_product_f64_on` (mean), `percentile_{un,}weighted_f32_on` (median) confirmed; goldens `regression_gh_iter{1,N}.txt`, `regression_l1_gh_*`, `poisson_gh_*`, `huber_gh_*`, `fair_gh_*`, `quantile_gh_*` **already in git**. **Landmine:** no `percentile_device` per-leaf kernel exists — see Common Pitfalls. |
| **ODL-06** | On-device binary-logloss grad/hess + BoostFromScore (label-prior logit init) + sigmoid ConvertOutput; one-vs-all label reset, anchor-pinned. (§5.2) | Math verified §5.2 (L416-425); host anchor `binary.rs::get_gradients` (L67) + `boost_from_score` (L102); golden `binary_gh_iter{1,N}.txt` **already in git**; two-stage BoostFromScore = `reduce_sum` + scalar `log(pavg/(1-pavg))/σ`. Atomic-order residual on the sum (D-05). |
| **ODL-07** | On-device multiclass grad/hess — softmax + OVA, class-major `[k·num_data+i]` layout, anchor-pinned. (§5.3) | Math verified §5.3 (L429-434); host anchor `multiclass.rs` (`MulticlassSoftmax`/`MulticlassOva`); `double* cuda_softmax_buffer` scratch (D-07/D-09); golden `multiclass_gh_iter{1,N}.txt` + `multiclassova_gh_*` **already in git, class-major**. STATE.md confirms `factor=num_class/(num_class-1)`, strided gather `rec[k]=score[num_data*k+i]`. |
| **ODL-08** | On-device ranking grad/hess — LambdaRank-NDCG + RankXENDCG, per-query block layout, bitonic item ranking, per-item RNG (bit-identical stream), anchor-pinned. (§5.4) | Math verified §5.4 (L438-462); host anchor `rank.rs` (`Lambdarank`/`RankXendcg`, `get_gradients` L239/L441, `make_rands`, `gradients_for_one_query`); RNG `draw_next_float_on` (`random.rs` L240) + `pow2_int`/`Phi`; per-segment argsort `bitonic_argsort_items_on` (L1361). **Golden GAP: no `lambdarank_gh` exists** — must capture. |
</phase_requirements>

---

## Standard Stack

This is a pure-Rust in-repo port — no new external crates. The "stack" is the existing
workspace crates + CubeCL primitives the objective kernels compose.

### Core (existing crates — verified present)

| Component | Location | Purpose | Reuse Mode |
|-----------|----------|---------|------------|
| `lgbm-objective` | `crates/lgbm-objective/src/{regression,binary,multiclass,rank,percentile}.rs` | Host C++ transcription = the **cpu f64 anchor** (D-01) AND the math to port to `#[cube]` | Anchor + port source |
| `primitives.rs` | `crates/lgbm-compute/src/kernels/primitives.rs` | percentile, reduce, dot-product, bitonic argsort, prefix-sum | COMPOSE (D-08) |
| `random.rs` | `crates/lgbm-compute/src/kernels/random.rs` | `CUDARandom` LCG draws | COMPOSE (D-08) |
| `lgbm-compute` seam | `crates/lgbm-compute/src/lib.rs` | `Backend`; `on_device_growth_supported()` stays **false** | Extend, don't flip |
| `oracle-harness` | `crates/oracle-harness/{src,tests,fixtures}` | comparator + parity tests + goldens | Extend |
| Golden capture | `xtask/py/boosting_oracle_capture.py`, `rank_oracle_capture.py` | Real `lib_lightgbm` 4.6 capture | Extend for `lambdarank_gh` |

### Verified primitive signatures (COMPOSE — D-08)

All confirmed present in `primitives.rs` / `random.rs`. **Note the host-slice signatures** —
these are host-orchestrated composites, not device block-kernels (see Pitfall 1):

```rust
// primitives.rs
pub fn reduce_sum_f64_on<R>(client, values: &[f64]) -> Result<f64, ComputeError>   // L784
pub fn reduce_max_f64_on<R>(client, values: &[f64]) -> Result<f64, ComputeError>   // L795
pub fn reduce_min_f64_on<R>(client, values: &[f64]) -> Result<f64, ComputeError>   // L806
pub fn dot_product_f64_on<R>(client, a: &[f64], b: &[f64]) -> Result<f64, ..>      // L818
pub fn percentile_unweighted_f32_on<R>(client, values: &[f32], alpha: f64) -> Result<f32,..> // L1179
pub fn percentile_weighted_f32_on<R>(client, values: &[f32], weights: &[f64], alpha: f64) -> Result<f32,..> // L1246
pub fn bitonic_argsort_on<R>(client, keys: &[f32], ascending: bool) -> Result<(Vec<i32>, Vec<f32>),..>  // L969 single-block
pub fn bitonic_argsort_global_on<R>(...)   // L1083  >single-block
pub fn bitonic_argsort_items_on<R>(client, keys: &[f32], segment_boundaries: &[i32], ascending: bool) -> Result<Vec<i32>,..> // L1361 per-segment (HOST loop over segments)
pub fn prefix_sum_{inclusive,exclusive}_{f64,f32,u16,u32}_on<R>(...)  // L245-556

// random.rs
pub fn draw_next_float_on<R>(client, seeds: &[u32], k: u32) -> Result<Vec<f32>, ComputeError>  // L240, CubeDim(1) single-thread — bit-identical stream
```

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| Host-orchestrated percentile per leaf (loop → gather → `percentile_weighted_f32_on`) | New one-block-per-leaf device percentile kernel | Faithful to CUDA §5.1, but new kernel + parity surface; host-loop is parity-identical and far cheaper. **Recommend host-loop this phase** (perf is deferred; D-02 is standalone). |
| One comptime-generic regression grad `#[cube]` (branch on objective enum) | Six separate kernels | Parity-neutral (Claude's Discretion). One generic reduces boilerplate; six mirror the C++ 1:1. Slight lean to one generic + comptime objective tag. |
| `CUDAMulticlassOVA` = literal binary-kernel-per-class | Thin softmax-off variant | Parity-neutral; STATE.md confirms host `MulticlassOva` reuses `Binary` at `offset=num_data*i`. Mirror that. |

**Installation:** None. No new crate. (`cubecl` 0.10 already pinned; no version bump needed.)

---

## Package Legitimacy Audit

Not applicable — this phase installs **no external packages**. It ports C++ reference kernels
to CubeCL using crates already in the workspace (`cubecl`, `lgbm-*`). The only external
runtime dependency touched is the **already-vendored** `lib_lightgbm` 4.6 pip wheel used by
the existing capture harness (Phase-3/5/6 provenance, `boosting_oracle_capture.py` header).
No registry lookups required.

---

## Architecture Patterns

### System Architecture Diagram

```
                         Phase 19 (standalone, behind LGBM_CUDA_ON_DEVICE, unwired)
  ┌──────────────────────────────────────────────────────────────────────────────┐
  │                                                                                │
  │  device scores (double* / f32)  ─┐                                             │
  │  device labels (f32)             ─┤                                            │
  │  device weights (f32 | null)     ─┤──▶ [ Device Objective Kernels ]           │
  │  objective scalars (α,c,σ,...)   ─┘        (new: kernels/objective_*.rs)      │
  │                                              │                                 │
  │        ┌─────────────────────────────────────┼──────────────────────────────┐ │
  │        │ GetGradients  ──▶ grad/hess (f32, class-major for multiclass)       │ │
  │        │ BoostFromScore ─▶ reduce_sum/dot_product/percentile ─▶ init scalar  │ │
  │        │ ConvertOutput  ──▶ sigmoid/exp/sign·x²/softmax (predict-side)        │ │
  │        │ RenewTreeOutput ─▶ per-leaf percentile ─▶ leaf values (L1/quantile)  │ │
  │        └──────────────────────────┬───────────────────────────────────────── ┘ │
  │                                    │ COMPOSES                                    │
  │             ┌──────────────────────┼──────────────────────┐                     │
  │        primitives.rs          random.rs              (scratch, pre-alloc D-09)   │
  │   (reduce/dot/percentile/     (CUDARandom LCG          softmax_buf, item_rands,  │
  │    bitonic argsort/prefix)     draw_next_float)         params_buf, partials)    │
  │                                                                                  │
  └──────────────────────────────────┬───────────────────────────────────────────┘
                                      │ ANCHOR (never GPU-vs-GPU, D-01/D-05)
                                      ▼
        cpu f64 fold (lgbm-objective) ──┐        real lib_lightgbm 4.6 goldens
        get_gradients/boost_from_score  ├──▶ oracle-harness comparator ──▶ PASS/FAIL
        (deterministic reduction order) ┘   (compare_exact_u32 | compare_within 1e-6)

  NOT TOUCHED this phase: gbdt.rs train_one_iter loop, Phase-18 CUDATree, score updater.
```

### Recommended Project Structure

```
crates/lgbm-compute/src/kernels/
├── objective_regression.rs   # ODL-05: 6 grad kernels + Convert + Renew + init-score
├── objective_binary.rs       # ODL-06: grad + 2-stage BoostFromScore + sigmoid + OVA reset
├── objective_multiclass.rs   # ODL-07: softmax grad (class-major) + softmax Convert + OVA reuse
├── objective_rank.rs         # ODL-08: LambdaRank-NDCG {shared,>2048} + RankXENDCG {shared,global}
└── mod.rs                    # add `pub mod objective_*;`  (currently no objective module — greenfield)

crates/lgbm-compute/src/           # a thin device-objective trait/enum mirroring
                                   # lgbm-objective's surface (Claude's Discretion placement)

crates/oracle-harness/tests/
└── objective_parity.rs        # NEW: reuse parse_gh_golden/assert_gradients pattern from boosting_parity.rs

crates/oracle-harness/tests/fixtures/
├── boosting/*_gh_iter{1,N}.txt   # REUSE existing L2/binary/multiclass/... goldens
└── rank/lambdarank_gh_*.txt      # NEW capture (the one gap)

xtask/py/
└── rank_oracle_capture.py        # EXTEND to emit lambdarank_gh golden (score-derivation route)
```

### Pattern 1: Template-flag → CubeCL comptime (verified §17, L1197)

```rust
// Source: docs/cuda-kernel-design.md §17 + existing histogram.rs comptime usage
// C++ `GetGradientsKernel_RegressionL2<bool USE_WEIGHT>` →
#[cube(launch_unchecked)]
fn get_gradients_l2<F: Float>(
    scores: &Array<F>, labels: &Array<F>, weights: &Array<F>,
    grad: &mut Array<F>, hess: &mut Array<F>,
    #[comptime] use_weight: bool,     // ← the <USE_WEIGHT> template flag
) {
    let i = ABSOLUTE_POS;
    if i < scores.len() {
        let diff = scores[i] - labels[i];      // diff = score - label
        if use_weight {                          // comptime branch — no runtime cost
            grad[i] = diff * weights[i];
            hess[i] = weights[i];
        } else {
            grad[i] = diff;
            hess[i] = F::new(1.0);
        }
    }
}
```
`<MAX_ITEM_GT_1024>`, `<NUM_RANK_LABEL>`, `<SHARED_MEMORY_SIZE>` (rank) and
`<USE_LABEL_WEIGHT,USE_WEIGHT>` (binary) all map the same way — `#[comptime]` params.

### Pattern 2: BoostFromScore = device reduce + host scalar finalize (§5.2, L420-423)

```
// Binary two-stage (verified §5.2):
//   kernel 1: reduce_sum_f64_on(labels), reduce_sum_f64_on(weights)   [device]
//   kernel 2 (<<<1,1>>> analog): pavg = clamp(Σw·y / Σw, ε, 1-ε);
//                                init = ln(pavg/(1-pavg)) / σ           [host scalar, f64]
// Anchor: binary.rs::boost_from_score(label) L102 is the exact f64 reference.
```
The scalar stage is legitimately f64 (D-07 allows f64 in scalar BoostFromScore math). The
**atomicAdd sum** is the documented atomic-order residual (D-05) — assert the *init scalar*
within `ORACLE_TOL`, not bit-exact, when the reduction is atomic.

### Pattern 3: Class-major multiclass gather (§5.3, STATE.md L237)

```rust
// class-major [k·num_data+i] — stride by class. Verified against STATE.md:
//   rec[k] = score[num_data*k + i]  (gather all K class-scores for row i)
//   softmax → p;  grad[k·num_data+i] = (label==k ? p-1 : p);
//   hess[k·num_data+i] = factor * p * (1-p),  factor = num_class/(num_class-1)
// double* cuda_softmax_buffer is the per-row length-K scratch (pre-alloc once, D-09).
```
The golden `multiclass_gh_iter1.txt` is **stored class-major** (verified: 36 values = 12
rows × 3 classes, GRAD then HESS) — the comparator reads it in the same stride.

### Pattern 4: Per-item ranking RNG (§5.4, L441-458)

```
// RankXENDCG φ = 2^label - rand  via Phi(label, g);  Lambdarank per-item λ via atomicAdd_block
//   host anchor: rank.rs Phi = pow2_int(label) - g  (L371-372),
//                make_rands builds the per-query Random vector,
//                draws advance across iterations (rank.rs L405-435 doc).
//   device: draw_next_float_on(seeds, k) — bit-identical LCG stream (random.rs L240).
// The RNG draw ORDER is load-bearing (rank.rs L28-31) — replicate seed+q per query.
```

### Anti-Patterns to Avoid

- **f64 per-row hot loop** (D-07 / spike-052): a 5.4× consumer-NVIDIA regression. Use f32 for
  per-row grad/hess; reserve f64 for the `double* score` accumulator, `cuda_softmax_buffer`,
  and scalar reduction finalize only.
- **GPU-vs-GPU parity** (D-05 / def-f8u-01): never assert two GPU f32 runs equal. Always pin
  to the cpu f64 anchor. The atomic-order nondeterminism guarantees two GPU runs *won't* be
  bit-equal — comparing them is a flaky-test factory.
- **Assuming a `percentile_device` per-segment kernel exists** — it does not (see Pitfall 1).
- **Synthesizing Convert/Renew for rank objectives** — they are base no-ops (§5.4 L461-462,
  CONTEXT "specifics"). Do not invent them.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Median / quantile of a leaf-subset | Custom sort+select | `percentile_{un,}weighted_f32_on` | Already C++-`PercentileDevice`-faithful, tested (Phase 14) |
| Sum / max / min reductions | Custom reduce kernel | `reduce_{sum,max,min}_f64_on` | Fixed reduction order = cpu anchor's bit-exact fold |
| Mean via Σ(w·label) | Two separate reductions | `dot_product_f64_on(weights, labels)` | Single fused reduction, matches §5.1 `DotProdGlobal` |
| Per-item ranking randoms | Custom RNG | `draw_next_float_on` (`CUDARandom` LCG) | Bit-identical stream is the parity contract (D-08) |
| Item ranking within a query | Custom bitonic | `bitonic_argsort_items_on` (per-segment) | §5.4 `BitonicArgSort`; skeleton exists |
| Grad/hess golden capture | New Python from scratch | Extend `boosting_oracle_capture.py` / `rank_oracle_capture.py` | Real `lib_lightgbm` 4.6 + score-derivation route already proven |
| Parity comparison | New assert code | `compare_exact_u32`, `compare_exact_f64_bits`, `compare_within`, `parse_gh_golden`, `assert_gradients` | Existing harness (boosting_parity.rs L446, L806) |

**Key insight:** ~90% of Phase 19's machinery — the primitives, the RNG, the capture
harness, the comparator, and 3 of 4 grad/hess goldens — **already exists in git**. The
net-new work is the `#[cube]` objective kernels themselves + one lambdarank golden + one
parity test file. Treat this as *composition*, not construction.

---

## Runtime State Inventory

Not a rename/refactor/migration phase — **greenfield kernel addition**. No stored data, live
service config, OS-registered state, secrets, or build artifacts carry state that this phase
renames. The only "state" is additive: new fixture files under
`crates/oracle-harness/tests/fixtures/rank/`. **None found in any migration category —
verified: this phase only adds new kernel modules + one golden, behind an OFF-by-default env
flag, touching no existing serialized state.**

---

## Common Pitfalls

### Pitfall 1: There is NO device per-segment / per-leaf percentile kernel

**What goes wrong:** CONTEXT D-08 and §5.1 describe RenewTreeOutput as "one block per leaf via
`PercentileDevice`" and D-08 lists "per-segment bitonic argsort". A planner may assume a
`percentile_device` block-kernel and a per-segment *percentile* primitive exist.
**Reality (verified):** `primitives.rs` has only `percentile_unweighted_f32_on` /
`percentile_weighted_f32_on` — both take a **host `&[f32]` whole-array slice** and return a
**single scalar**. The *per-segment* primitive is **argsort only** (`bitonic_argsort_items_on`,
L1361, which itself **loops on the host** over segments calling the single-block argsort).
There is no block-per-leaf percentile device kernel.
**How to avoid:** For RenewTreeOutput, **host-loop over leaves**: gather each leaf's residual
subset, call `percentile_weighted_f32_on` per leaf. This is parity-identical to the CUDA
one-block-per-leaf result (same math, same sort order) and is the cheap correct move for a
standalone (unwired, perf-deferred) phase. Writing a true device block-per-leaf kernel is a
Phase-21/perf option, not a Phase-19 requirement. **Flag this discrepancy in the plan.**
**Warning signs:** a task action that says "call `percentile_device`" or "reuse the
per-segment percentile primitive" — neither exists.

### Pitfall 2: The lambdarank grad/hess golden does not exist yet

**What goes wrong:** D-01 asks for a real-`lib_lightgbm` golden for the lambdarank
representative. The `rank/` fixtures have `lambdarank_scores.txt`, `lambdarank_ndcg.txt`, and
model files — but **no `lambdarank_gh` file**. (By contrast L2/binary/multiclass grad/hess
goldens *already exist* under `boosting/*_gh_iter{1,N}.txt`.)
**How to avoid:** Extend `xtask/py/rank_oracle_capture.py` to emit a `lambdarank_gh_*.txt`
via the **score-derivation route** (compute grad/hess from the captured per-iteration raw
scores using the lambdarank math, cross-checked against the real binary's scores+model — the
identical route `boosting_oracle_capture.py` uses for the other families). This is the one
new capture the phase requires.
**Warning signs:** a fixture-read of `lambdarank_gh_iter1.txt` that returns `None`.

### Pitfall 3: Class-major stride correctness (multiclass)

**What goes wrong:** Row-major vs class-major confusion silently corrupts multiclass grad/hess.
The golden is stored **class-major** `[k·num_data+i]`.
**How to avoid:** Gather `rec[k] = score[num_data*k + i]`; write `grad[num_data*k + i]`. Verify
against `multiclass_gh_iter1.txt` (12 rows × 3 classes = 36 values, class-blocked). STATE.md
L237 confirms the exact stride and `factor=num_class/(num_class-1)`.
**Warning signs:** grad/hess correct for class 0 but wrong for classes 1..K-1.

### Pitfall 4: RankXENDCG >2048 global path buffer-stashing (§5.4, L457-458)

**What goes wrong:** The `_GlobalMemory` variant stashes intermediates in the **hessian output
buffer + `cuda_params_buffer`** when items exceed shared capacity. A naive port that treats
the hessian buffer as write-only corrupts it.
**How to avoid:** Faithfully reproduce the buffer-aliasing: hessian buffer doubles as scratch
during the reduction passes, finalized to true hessians at the end. Pre-allocate
`cuda_params_buffer` once (D-09). De-risked by the existing `bitonic_argsort_global_on` /
`bitonic_argsort_items_on` skeletons (D-03 rationale). **Note phase18-wr01 HistArena::swap
aliasing memory** — the same aliasing-discipline caution applies to any scratch reuse.
**Warning signs:** shared-path (≤2048) parity passes but global-path (>2048) hessians drift.

### Pitfall 5: Atomic-order nondeterminism mis-classified as a bug (§17, L1176-1181; D-05)

**What goes wrong:** binary BoostFromScore (`atomicAdd`) and lambdarank (`atomicAdd_block`)
f32 sums are **not bit-reproducible**. Asserting bit-exact against the anchor fails
spuriously.
**How to avoid:** For these specific outputs assert `compare_within(ORACLE_TOL=1e-6)` (or a
tie-aware envelope up to 1e-5), never `compare_exact`. Everything else (elementwise grad/hess
with no accumulation) should assert **bit-exact** (`compare_exact_u32`). Classify per output
in the Validation Architecture below.
**Warning signs:** a parity test that flakes across runs on binary/lambdarank init/λ but is
stable on L2/quantile grad.

### Pitfall 6: Poisson label-check ordering (§5.1, L410-412)

**What goes wrong:** Poisson runs `LaunchCheckLabelKernel` (label non-negativity / finiteness
via `ReduceSum`+`ReduceMin`) that other regression objectives don't. Omitting it diverges on
the init-score path.
**How to avoid:** Compose `reduce_sum_f64_on` + `reduce_min_f64_on` for the label check before
the Poisson mean init; mirror `regression.rs` Poisson BoostFromScore exactly.

---

## Code Examples

### Verified host anchor — regression get_gradients (the port source AND anchor)

```rust
// Source: crates/lgbm-objective/src/regression.rs:380 (verified present)
pub fn get_gradients(
    // ... score, gradients, hessians, weights ...
) // → the f64-deterministic reference the device kernel is pinned to
```

### Verified golden format + comparator (reuse verbatim)

```
// Source: crates/oracle-harness/tests/fixtures/boosting/binary_gh_iter1.txt
GRAD 1054168405 1054168405 ...   # per-row f32 bits (decimal u32), from_bits → f32
HESS 1048109966 1048109966 ...
```
```rust
// Source: crates/oracle-harness/tests/boosting_parity.rs:446, 806
fn parse_gh_golden(text) -> (Vec<f32>, Vec<f32>);        // GRAD/HESS lines
assert_gradients(&booster, "binary_gh_iter1.txt", "binary_gh_iterN.txt");
// comparators (oracle-harness/src/comparator.rs):
compare_exact_u32(rust, cpp)          // L125 bit-exact (no accumulation)
compare_within(rust, cpp, ORACLE_TOL) // L92  tol=1e-6 (atomic/transcendental)
// ORACLE_TOL: f32 = 1e-6  (comparator.rs:15)
```

### Verified RNG composition (rank per-item randoms)

```rust
// Source: crates/lgbm-compute/src/kernels/random.rs:240 (verified)
draw_next_float_on(client, seeds, k) -> Vec<f32>   // bit-identical LCG stream
// host anchor: crates/lgbm-objective/src/rank.rs (Phi = pow2_int(label) - g, make_rands)
```

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Host-transcription goldens only (Phase 17/18) | Real `lib_lightgbm` 4.6 per-family goldens (D-01) | Phase 19 | Answers the `on-device-kernel-goldens-are-retranscriptions` caveat — proves *reference fidelity*, not just transcription agreement |
| CUDA `__global__` templated kernels | CubeCL `#[cube]` + `#[comptime]` flags | This port | Template-flag explosion → comptime (§17 L1197) |
| f64 everywhere in reference | f32 per-row hot path, f64 only for accumulators/scalars | spike-052 / D-07 | Avoids 5.4× consumer-NVIDIA f64 regression |

**Deprecated/outdated for this phase:**
- Quantized/discretized objective path (`RenewDiscretizedTreeLeavesKernel`, int16 packed) —
  explicitly **v2 (QGD)**, not this phase (CONTEXT deferred).
- Any assumption that objectives get wired into `gbdt.rs` this phase — that is **Phase 21**.

---

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | The score-derivation route (grad/hess from per-iteration raw scores) is a valid capture for lambdarank as it was for L2/binary/multiclass | Pitfall 2 | LOW — lambdarank grad/hess is a deterministic function of scores+labels+ranks; but the *ranking* step means grad/hess depends on within-query sort, so the derivation script must replicate the rank/λ math (not just `score-label`). If the derivation is harder than expected, fall back to a custom-`fobj` interception capture. |
| A2 | Host-loop percentile-per-leaf is bit-parity-identical to the CUDA one-block-per-leaf result | Pitfall 1, Alternatives | LOW — same sort + same percentile index math; sort order is deterministic. Verify on the first RenewTreeOutput parity test. |
| A3 | No new external crate or `cubecl` version bump is needed | Standard Stack | LOW — all primitives/RNG confirmed present at cubecl 0.10; but confirm `#[comptime] bool` comptime-branch codegen on `cubecl-cpu` for the objective enum (existing histogram.rs uses comptime, so precedent holds). |
| A4 | `lib_lightgbm` 4.6 pip wheel is still available in the repo's capture venv (Phase-8 uv `.venv`) | Golden capture | MEDIUM — memory note `phase8-python-venv` says the uv `.venv` holds `lightgbm==4.6`; confirm before capture. If missing, `pip install lightgbm==4.6` reproduces it (deterministic capture). |

**These four are the only unverified points; everything else in this research was confirmed
against repo code or the design doc. The planner should gate A1/A4 behind a
`checkpoint:human-verify` or an early spike task if lambdarank capture proves fiddly.**

---

## Open Questions

1. **Does the lambdarank grad/hess golden derive cleanly from raw scores, or does it need a
   `fobj` interception?**
   - What we know: L2/binary/multiclass derive from scores via the objective math
     (`boosting_oracle_capture.py`). Lambdarank grad/hess is a function of within-query ranks
     + σ + label-gains + inv-max-dcg — all recoverable from the captured scores + config.
   - What's unclear: whether the `norm` rescale + `truncation_level` interaction is easier to
     reproduce in the capture script or to intercept via a custom objective.
   - Recommendation: attempt score-derivation first (cheapest, matches the established
     pattern); fall back to `fobj` interception if the λ accumulation is hard to reproduce.

2. **One comptime-generic regression kernel vs six kernels?**
   - What we know: parity-neutral (Claude's Discretion, D-135).
   - Recommendation: one `#[cube]` with a `#[comptime]` objective tag + `#[comptime] use_weight`
     — fewer launch sites, matches how histogram.rs already fans out on comptime. Six-kernel is
     equally acceptable if the enum-branch codegen bloats.

3. **Device-objective trait/enum placement** — `crates/lgbm-compute/src/` vs a new sub-module.
   - Recommendation: mirror `lgbm-objective`'s enum-dispatch surface (`Objective`, `Binary`,
     `MulticlassSoftmax/Ova`, `Lambdarank`, `RankXendcg`) as a device-side trait/enum so
     Phase 21 can wire it symmetrically. Non-load-bearing for parity.

---

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| `cubecl` (cpu backend) | All objective kernels + primitives | ✓ | 0.10 (workspace-pinned) | — |
| `cubecl-hip` (ROCm) | Separate ~1e-6 hip parity layer (`--features rocm`) | ✓ (spoofed 8-CU APU) | 0.10 | cpu anchor is the hard gate; hip is best-effort |
| `lib_lightgbm` 4.6 pip wheel | D-01 golden capture (lambdarank) | ✓ (per memory `phase8-python-venv`) — **confirm** | 4.6 | `pip install lightgbm==4.6` in the uv `.venv` |
| Python + numpy | `xtask/py/*_oracle_capture.py` | ✓ | — | — |
| Existing goldens L2/binary/multiclass | ODL-05/06/07 anchor | ✓ **already in git** | — | none needed |

**Missing dependencies with no fallback:** none.
**Missing dependencies with fallback:** `lambdarank_gh` golden (must be captured; capture
harness + wheel present). Confirm the uv `.venv` still has `lightgbm==4.6` before the
capture task.

---

## Validation Architecture

> **Nyquist validation is enabled** (`config.json workflow.nyquist_validation: true`). This
> section is the liftable strategy for VALIDATION.md.

### Test Framework

| Property | Value |
|----------|-------|
| Framework | Rust `cargo test` (workspace), parity tests in `crates/oracle-harness/tests/` |
| Config file | none — standard `cargo test`; ROCm layer gated by `--features rocm` |
| Quick run command | `cargo test -p oracle-harness --test objective_parity` (new file) |
| Full suite command | `cargo test --workspace` (cpu hard gate) |
| ROCm cross-check | `cargo test -p oracle-harness --features rocm` (separate ~1e-6 layer) |

### Anchor & Tolerance Policy (per D-05)

| Output class | Anchor | Assertion | Rationale |
|--------------|--------|-----------|-----------|
| Elementwise grad/hess (no accumulation): **L2, L1, Quantile, Huber, Fair, Poisson, multiclass-softmax, binary** | cpu f64 fold + real-`lib_lightgbm` golden | **bit-exact** `compare_exact_u32` on f32 bits | Pure per-row math; deterministic |
| BoostFromScore mean (L2/Huber/Fair via reduce/dot) | cpu f64 fold | bit-exact if serial fold; `compare_within` if atomic reduce | Fixed reduction order → bit-exact possible |
| BoostFromScore **binary logit** (`atomicAdd` sums) | cpu f64 fold | `compare_within(ORACLE_TOL=1e-6)` | Atomic-order residual (D-05, §17) |
| BoostFromScore median (L1/Quantile via percentile) | cpu f64 fold | bit-exact (sort is deterministic) | `percentile_*` is deterministic |
| **Lambdarank λ / hess** (`atomicAdd_block`) | cpu f64 fold + `lambdarank_gh` golden | `compare_within(ORACLE_TOL)` / tie-aware | Atomic-block accumulation residual (D-05) |
| RankXENDCG grad/hess (softmax + RNG) | cpu f64 fold | `compare_within(ORACLE_TOL)` on the transcendental cells; RNG stream **bit-exact** | exp/log-bearing (rank_parity precedent) |
| ConvertOutput (sigmoid/exp/sign·x²/softmax) | `lgbm_model::ObjectiveKind::convert_output` (host) | bit-exact where no transcendental; `compare_within` on exp/log | Predict-side transform |
| RenewTreeOutput (per-leaf median/quantile) | cpu f64 fold + `regression_l1` renewed-leaf golden | bit-exact (deterministic percentile) | boosting_parity L750 precedent (median-residual renew) |
| Per-item ranking RNG stream | `draw_next_float_on` reference | **bit-exact** `compare_exact_u32` | Bit-identical LCG stream is the contract (D-08) |

### Phase Requirements → Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| ODL-05 | 6 regression grad/hess bit-exact vs f64 anchor + L2 real golden | unit/parity | `cargo test -p oracle-harness --test objective_parity regression` | ❌ Wave 0 (test file); goldens ✅ |
| ODL-05 | BoostFromScore mean/median + Poisson label-check | parity | `... objective_parity boost_from_score` | ❌ Wave 0 |
| ODL-05 | RenewTreeOutput L1/quantile per-leaf median | parity | `... objective_parity renew_leaf` | ❌ Wave 0; `regression_l1` renewed golden ✅ |
| ODL-05 | ConvertOutput sign·x² / exp | parity | `... objective_parity convert_regression` | ❌ Wave 0 |
| ODL-06 | binary grad/hess bit-exact + real golden | parity | `... objective_parity binary` | ❌ Wave 0; `binary_gh_*` ✅ |
| ODL-06 | binary logit BoostFromScore (tol) + OVA label reset | parity | `... objective_parity binary_boost` | ❌ Wave 0 |
| ODL-07 | multiclass softmax grad/hess class-major + real golden | parity | `... objective_parity multiclass` | ❌ Wave 0; `multiclass_gh_*` ✅ |
| ODL-07 | multiclassova reuse-binary-per-class | parity | `... objective_parity multiclassova` | ❌ Wave 0; `multiclassova_gh_*` ✅ |
| ODL-08 | LambdaRank-NDCG shared + >2048 `_Sorted` grad/hess (tol) + real golden | parity | `... objective_parity lambdarank` | ❌ Wave 0; **golden ❌ capture** |
| ODL-08 | RankXENDCG shared + global grad/hess (tol) + RNG stream bit-exact | parity | `... objective_parity rank_xendcg` | ❌ Wave 0; `rank_xendcg_objseed5` ✅ |

### Property-Based / Held-Out Backstop

- **Determinism property:** run each device grad/hess kernel twice on the cpu backend →
  bit-identical (the cpu f64 fold is deterministic; guards against accidental nondeterministic
  reduction in a supposedly-deterministic kernel).
- **Weight-branch equivalence property:** `use_weight=true` with all-1.0 weights ==
  `use_weight=false` (bit-exact) — catches comptime-branch divergence.
- **Class-major invariant:** multiclass grad summed over classes per row ≈ 0 for softmax
  (Σ(p-δ) = 1 - 1 = 0 up to f32) — a cheap held-out sanity net independent of the golden.
- **RNG replay:** `rank_xendcg_objseed5` seed-replay (existing rank_parity precedent) — the
  per-item draw stream must bit-match across the host anchor and device draw.

### Sampling Rate

- **Per task commit:** `cargo test -p oracle-harness --test objective_parity <family>` (the
  family under edit — < 30 s).
- **Per wave merge:** `cargo test -p oracle-harness` (all objective + existing parity).
- **Phase gate:** `cargo test --workspace` green (cpu hard merge gate, ODL-19 / D-06) with
  `LGBM_CUDA_ON_DEVICE` unset; optional `--features rocm` for the ~1e-6 hip cross-check.

### Wave 0 Gaps

- [ ] `crates/oracle-harness/tests/objective_parity.rs` — new parity test file (reuse
      `parse_gh_golden` / `assert_gradients` / comparator pattern from `boosting_parity.rs`).
- [ ] `crates/oracle-harness/tests/fixtures/rank/lambdarank_gh_iter{1,N}.txt` — **capture**
      (extend `xtask/py/rank_oracle_capture.py`, score-derivation route).
- [ ] Confirm the uv `.venv` has `lightgbm==4.6` (A4) before the capture task.
- [ ] `crates/lgbm-compute/src/kernels/objective_*.rs` + `mod.rs` exports (greenfield — no
      objective module exists today).

*(Goldens for L2/binary/multiclass/regression_l1/poisson/huber/fair/quantile already exist —
no capture needed; only lambdarank is missing.)*

---

## Security Domain

> `security_enforcement: true`, `security_asvs_level: 1` in config. This is a **numerical
> compute-kernel phase behind an OFF-by-default env flag**, with no network, auth, session,
> or external-input surface. The relevant control is **input validation at the device
> boundary**.

### Applicable ASVS Categories

| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | no | — |
| V3 Session Management | no | — |
| V4 Access Control | no | — |
| V5 Input Validation | **yes** | Length/shape checks at the kernel entry (mirroring `percentile_*` `LengthMismatch` / empty-slice rejection at the V5 boundary — see `primitives.rs` L1188 comment `T-14-05-01`); Poisson label non-negativity/finiteness check (§5.1, a real domain guard); multiclass `LabelOutOfRange` guard (STATE.md L237 — carried from host `MulticlassSoftmax::Init`). |
| V6 Cryptography | no | RNG is a **deterministic LCG for parity**, not a security RNG — do not substitute a CSPRNG (would break the bit-identical-stream contract). |

### Known Threat Patterns for CubeCL numerical kernels

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| Out-of-bounds device index (bad num_data / class stride) | Tampering | `if i < len` guard in every `#[cube]`; length checks at host launch (existing primitive pattern) |
| Buffer aliasing corruption (RankXENDCG global path stashing in hessian buffer) | Tampering | Faithful, documented aliasing + pre-alloc once (D-09); heed phase18-wr01 swap-aliasing caution |
| NaN/Inf propagation from unchecked labels (Poisson, multiclass) | Denial (bad model) | Poisson label-check kernel + multiclass `LabelOutOfRange` guard (parity with C++ `Init`) |
| Empty leaf/query passed to percentile/argsort | Denial (panic/UB) | `percentile_*` already rejects empty at V5 boundary (`LengthMismatch`); mirror for per-leaf/per-query loops |

---

## Sources

### Primary (HIGH confidence)
- `docs/cuda-kernel-design.md` §5 (L369-462), §16 (L1145-1167), §17 (L1171-1204) — the port
  source of truth; per-family kernel tables, sequencing, port considerations.
- `crates/lgbm-compute/src/kernels/primitives.rs` (L245-1385) — verified primitive signatures
  (percentile, reduce, dot-product, bitonic argsort, prefix-sum).
- `crates/lgbm-compute/src/kernels/random.rs` (L240) — `draw_next_float_on` LCG.
- `crates/lgbm-objective/src/{regression,binary,multiclass,rank}.rs` — verified host anchor
  method signatures (`get_gradients`, `boost_from_score`, `make_rands`, `Phi`/`pow2_int`).
- `crates/oracle-harness/tests/boosting_parity.rs` (L446, L750, L806) — `parse_gh_golden` /
  `assert_gradients` reuse pattern.
- `crates/oracle-harness/src/comparator.rs` (L15, L86-172) — comparator surface + `ORACLE_TOL`.
- `crates/oracle-harness/tests/fixtures/boosting/*_gh_iter*.txt` — existing real-`lib_lightgbm`
  grad/hess goldens (L2/binary/multiclass/... class-major, f32 bits).
- `xtask/py/boosting_oracle_capture.py` (header + L28-263) — the real-binary capture mechanism
  (score-derivation route).
- `.planning/STATE.md` (L41-46, L230-237) — roadmap position + multiclass class-major stride /
  factor confirmation.

### Secondary (MEDIUM confidence)
- `.planning/phases/19-.../19-CONTEXT.md` — locked decisions (cross-checked against code).
- Memory notes: `on-device-kernel-goldens-are-retranscriptions`, `phase8-python-venv`,
  `gpu-bottleneck-now-seq-f64-scan` / spike-052 (f64-hot-loop), `def-f8u-01` (never GPU-vs-GPU),
  `phase18-wr01-histarena-swap-aliasing`.

### Tertiary (LOW confidence)
- A4 (`lib_lightgbm==4.6` still in the uv `.venv`) — asserted by memory, **not re-verified this
  session**; confirm before the capture task.

---

## Metadata

**Confidence breakdown:**
- Standard stack / primitives reuse: **HIGH** — every signature grepped and confirmed present.
- Golden mechanism: **HIGH** — capture script + existing goldens read directly; lambdarank gap
  confirmed by fixture listing.
- Architecture / kernel mapping: **HIGH** — §5/§16/§17 read in full; comptime pattern has
  in-repo precedent (histogram.rs).
- Pitfalls (percentile-not-device, lambdarank gap, class-major, global-buffer-stash): **HIGH**
  — grounded in code discrepancies, not speculation.
- A1 (lambdarank score-derivation) / A4 (wheel present): **MEDIUM** — flagged in Assumptions.

**Research date:** 2026-07-01
**Valid until:** ~2026-08-01 (stable — in-repo port against a fixed C++ reference; no
fast-moving external deps).
</content>
</invoke>
