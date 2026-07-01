# Phase 19: On-Device Objectives - Pattern Map

**Mapped:** 2026-07-01
**Files analyzed:** 7 net-new / greenfield artifacts (4 kernel modules + mod.rs export, 1 parity test, 1 capture-script extension)
**Analogs found:** 7 / 7 (all have strong in-repo analogs)

> **Consumer note (planner):** ~90% of Phase 19 machinery already exists in git (RESEARCH
> §"Don't Hand-Roll"). This map is deliberately *composition-first*: each new file COMPOSES
> existing primitives / harness rather than building new. Two RESEARCH-confirmed discrepancies
> are flagged in **## No Analog Found / Discrepancies** — read those before writing task actions.

---

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|-------------------|------|-----------|----------------|---------------|
| `crates/lgbm-compute/src/kernels/objective_regression.rs` | kernel (`#[cube]`) | transform (per-row) + reduce (BoostFromScore) + per-leaf (Renew) | `kernels/data_partition.rs` (comptime-flag `#[cube]` + launch) + `kernels/primitives.rs` reduce/percentile wrappers | role+flow exact |
| `crates/lgbm-compute/src/kernels/objective_binary.rs` | kernel (`#[cube]`) | transform + two-stage reduce→host-scalar | `kernels/random.rs` (`cube` helper fns → single-owner kernel → host launcher) + `primitives.rs::reduce_sum_f64_on` | role+flow exact |
| `crates/lgbm-compute/src/kernels/objective_multiclass.rs` | kernel (`#[cube]`) | transform (class-major strided, softmax scratch) | `primitives.rs` `#[cube]` body + launch wrapper; host anchor `multiclass.rs::get_gradients` | role+flow match |
| `crates/lgbm-compute/src/kernels/objective_rank.rs` | kernel (`#[cube]`) | per-segment (block-per-query) + RNG + bitonic argsort | `primitives.rs::bitonic_argsort_items_on` + `random.rs::draw_next_float_on` | role+flow match |
| `crates/lgbm-compute/src/kernels/mod.rs` (MODIFY) | module index | n/a (config) | existing Phase-14..18 `pub mod` block (lines 15-61) | exact |
| `crates/oracle-harness/tests/objective_parity.rs` | test (parity) | request-response (read golden → compare) | `tests/boosting_parity.rs` (`parse_gh` / `assert_gradients` / `read_golden`) + `tests/rank_parity.rs` (RNG replay) | exact |
| `xtask/py/rank_oracle_capture.py` (MODIFY) | capture script | batch (train → derive → emit) | `xtask/py/boosting_oracle_capture.py` (`*_gh_iter{1,N}` score-derivation route) | exact |

**Host anchor sources (NOT modified — the f64 fold each device kernel is pinned to, and the
port-source math to transcribe FROM):**

| Host anchor file | Functions the device kernels mirror |
|------------------|-------------------------------------|
| `crates/lgbm-objective/src/regression.rs` | `get_gradients` (L380), `boost_from_score` (L544), `renew_leaf_output` (L627), `is_renew_tree_output` (L271) |
| `crates/lgbm-objective/src/binary.rs` | `get_gradients` (L67), `boost_from_score` (L102), `class_need_train` (L124) |
| `crates/lgbm-objective/src/multiclass.rs` | `get_gradients` (L138), `boost_from_score` (L184), `class_need_train` (L193) |
| `crates/lgbm-objective/src/rank.rs` | `Lambdarank::get_gradients` (L239) / `gradients_for_one_query` (L274); `RankXendcg::get_gradients` (L441) / `make_rands` (L482) / `gradients_for_one_query` (L490); `pow2_int` (L372), `phi` (L544) |
| ConvertOutput (predict-side, re-exported, NOT re-ported) | `lgbm_model::ObjectiveKind::convert_output` (see `lgbm-objective/src/lib.rs:42`) |

---

## Pattern Assignments

### `crates/lgbm-compute/src/kernels/objective_regression.rs` (kernel, per-row transform + reduce + per-leaf)

**Primary skeleton analog:** `crates/lgbm-compute/src/kernels/data_partition.rs` (comptime-flag
`#[cube]` body) + `crates/lgbm-compute/src/kernels/primitives.rs` (launch wrapper + reduce/percentile host fns).

**Module doc + imports pattern** — copy the header shape from `primitives.rs:1-56`:
```rust
use cubecl::prelude::*;
use crate::error::ComputeError;
```
Every kernel module opens with `use cubecl::prelude::*;` + `use crate::error::ComputeError;`
(see `primitives.rs:54-56`, `random.rs:48-50`). Module docstring cites the C++ source + the
"Analog file" + the "cpu anchor stays a plain serial f64 fold (D-10)" discipline
(`primitives.rs:43-47`).

**`#[cube]` shared-body + thin launch-wrapper convention** (the SINGLE-SOURCE-OF-TRUTH idiom,
`primitives.rs:82-112` for the body, `primitives.rs:701-712` for the per-type wrappers):
```rust
// body once (generic over Float), wrappers delegate:
#[cube(launch_unchecked)]
fn dot_kernel_f64(a: &Array<f64>, b: &Array<f64>, out: &mut Array<f64>, n: u32) {
    dot_body::<f64>(a, b, out, n);
}
```

**Comptime template-flag → `#[comptime] bool`** (the `<USE_WEIGHT>` fan-out; precedent is the
6-flag `route_to_left` at `data_partition.rs:61-76`):
```rust
#[cube]
#[allow(clippy::too_many_arguments)]
pub(crate) fn route_to_left(
    bin: i32, /* ... */
    #[comptime] miss_is_zero: bool,   // ← template flag folds at comptime, no runtime cost
    #[comptime] default_left: bool,
) -> u32 { /* if miss_is_zero { ... } branches compile out */ }
```
RESEARCH Pattern 1 (§17) shows the exact `get_gradients_l2<USE_WEIGHT>` target shape; the
`diff = score - label` math is the host anchor `regression.rs:406-534` per-objective match arm
(L2 L407-411, L1 L413-423, Huber L425-440, Fair L442-454, Quantile L456-473, Poisson L489-501).
Discretion (D-135): one comptime-generic kernel branching an objective enum, OR six kernels —
parity-neutral. Lean: one `#[cube]` + `#[comptime] objective_tag` + `#[comptime] use_weight`
(matches how `data_partition.rs` fans on comptime).

**Host-orchestration launcher pattern** (`create_from_slice` → `launch_unchecked` →
`read_one_unchecked`, `primitives.rs:726-778`):
```rust
let h_in = client.create_from_slice(f64::as_bytes(data));
let h_out = client.empty(core::mem::size_of::<f64>());
// SAFETY: <sized-exactly + outlives-launch + in-range writes> (CMP-01)
unsafe {
    reduce_sum_kernel_f64::launch_unchecked(
        client, CubeCount::Static(1, 1, 1), CubeDim::new_1d(1),
        ArrayArg::from_raw_parts(h_in, n),
        ArrayArg::from_raw_parts(h_out.clone(), 1),
        n as u32,
    );
}
let bytes = client.read_one_unchecked(h_out);
Ok(f64::from_bytes(&bytes)[0])
```
Every host fn returns `Result<_, ComputeError>` and validates length at the V5 boundary
BEFORE any launch (`primitives.rs:731-740`, `random.rs:146-154`).

**BoostFromScore = device reduce + host scalar finalize** — COMPOSE, do not rebuild (D-08):
- L2/Huber/Fair mean: `primitives.rs::reduce_sum_f64_on` (L784) then `/ num_data`; or the fused
  `dot_product_f64_on` (L818) for the weighted `Σ(w·label)` (RESEARCH "Don't Hand-Roll").
  Host anchor `regression.rs:548-558`.
- L1/Quantile median: `primitives.rs::percentile_unweighted_f32_on` (L1179). Anchor
  `regression.rs:567-589`.
- Poisson label-check: `reduce_sum_f64_on` + `reduce_min_f64_on` (L784/L806) THEN `safe_log(mean)`.
  Anchor `regression.rs:595-605`. **Pitfall 6:** Poisson runs a label non-negativity check the
  other regression objectives skip — mirror it or the init-score diverges.

**RenewTreeOutput (L1/quantile, one-block-per-leaf)** — host-loop over leaves calling
`percentile_{un,}weighted_f32_on` per leaf (see Discrepancy 1). Anchor `regression.rs:627-651`
(`renew_leaf_output`: L1 = unweighted residual median α=0.5; quantile = residual percentile at
f32-rounded α).

---

### `crates/lgbm-compute/src/kernels/objective_binary.rs` (kernel, per-row transform + two-stage reduce→host-scalar)

**Analog:** `crates/lgbm-compute/src/kernels/random.rs` (the cleanest small `#[cube]`-helper →
single-owner-kernel → host-launcher module) + `primitives.rs::reduce_sum_f64_on`.

**Per-row grad/hess** — transcribe `binary.rs:67-96` verbatim into a `#[cube]` body:
`is_pos = label > 0`; `label_val = ±1`; `response = -label_val·σ/(1+exp(label_val·σ·score))`;
`grad = response·label_weight`; `hess = |response|·(σ−|response|)·label_weight`. The
`<USE_LABEL_WEIGHT, USE_WEIGHT>` templates → two `#[comptime] bool` params (same as regression).

**Two-stage BoostFromScore** (RESEARCH Pattern 2):
- kernel 1: `reduce_sum_f64_on(is_pos as f64)` [+ weight sum] — device.
- kernel 2 (`<<<1,1>>>` analog): `pavg = clamp(Σ/N, ε, 1-ε); init = ln(pavg/(1-pavg))/σ` — host
  scalar f64 (D-07 allows f64 in scalar BoostFromScore). Anchor `binary.rs:102-117` verbatim.

**Helper-fn `#[cube]` decomposition** — copy `random.rs:55-79` (small `#[cube]` fns
`cuda_rand_advance`/`cuda_rand_int16`/`cuda_next_float` composed by the kernel): factor the
sigmoid/response math into a `#[cube] fn` the grad-kernel calls, so the math exists once.

**OVA label reset (`ResetOVACUDALabelKernel`)** — a per-row elementwise `label == class ? +1 : -1`
rewrite kernel; same single-owner launcher shape as `random.rs:165-197`.

> **Assertion note (planner → VALIDATION):** the binary *init scalar* is the documented
> `atomicAdd`-order residual (D-05) → assert `compare_within(ORACLE_TOL)`, NOT `compare_exact`.
> The per-row grad/hess (no accumulation) IS bit-exact.

---

### `crates/lgbm-compute/src/kernels/objective_multiclass.rs` (kernel, class-major strided transform)

**Analog:** `primitives.rs` `#[cube]` body + launch wrapper; host anchor `multiclass.rs:138-180`.

**Class-major `[k·num_data+i]` stride is LOAD-BEARING** (RESEARCH Pitfall 3, Pattern 3). Transcribe
`multiclass.rs:156-178`:
```rust
// per row i: gather rec[k] = score[num_data*k + i]  (strided, class-major)
// softmax(rec) -> prob;
// grad[num_data*k+i] = (label[i]==k) ? (p - 1.0f) : p;
// hess[num_data*k+i] = factor * p * (1 - p),  factor = num_class/(num_class-1)
```
`double* cuda_softmax_buffer` = per-row length-K scratch, **pre-allocated once outside the loop**
(D-09 — the `client.empty` handle is created once and reused, mirroring the reused-scratch note in
`primitives.rs:10-11`). `factor` = `multiclass.rs:84`. Softmax reuses
`lgbm_model::objective::softmax` (host anchor `multiclass.rs:43` import) — the device kernel ports
the same in-place softmax math; do NOT introduce a new numerically-different softmax.

**MulticlassOVA** = reuse the binary kernel per class at `offset = num_data*i` (Discretion; anchor
`multiclass.rs` `MulticlassOva` holds K independent `Binary`). Parity-neutral.

**Held-out invariant** (RESEARCH Validation Architecture): Σ_k grad[k·N+i] ≈ 0 for softmax — a
cheap sanity net independent of the golden.

---

### `crates/lgbm-compute/src/kernels/objective_rank.rs` (kernel, block-per-query + RNG + argsort)

**Analog:** `primitives.rs::bitonic_argsort_items_on` (L1361, per-segment argsort) +
`random.rs::draw_next_float_on` (L240, bit-identical LCG) + host anchors `rank.rs`.

**LambdaRank-NDCG** — transcribe `rank.rs:274-366` (`gradients_for_one_query`): per-query
DESCENDING-score sort (`bitonic_argsort_items_on` per segment = per query), pairwise λ over
`truncation_level`, `atomicAdd_block`-style accumulation into `lambdas`/`hessians`,
`norm` rescale (L359-365). `NUM_QUERY_PER_BLOCK=10`, block-per-query-group; `cuda_query_boundaries_`
= the `query_boundaries` prefix-sum (anchor `rank.rs:253-256`). Build BOTH shared + `_Sorted`
(>2048) variants (D-03).

**RankXENDCG** — transcribe `rank.rs:490-540`: per-query `softmax(score) → rho`;
`params[i] = phi(label[i], rng.next_float())` where `phi = pow2_int(label) − g`
(`rank.rs:544-546`, `pow2_int` L372 is repeated-multiply NOT `powf`); three-order λ terms + hessian.
Build BOTH `_SharedMemory<SIZE>` and `_GlobalMemory` (D-03). **Pitfall 4:** the >2048 global path
stashes intermediates in the **hessian output buffer + `cuda_params_buffer`** — reproduce the
aliasing faithfully; pre-alloc `cuda_params_buffer` once (D-09); heed `phase18-wr01` swap-aliasing.

**Per-item RNG — COMPOSE `draw_next_float_on`** (`random.rs:240-268`), do NOT rebuild. The draw
ORDER is load-bearing: per query `q`, `Random(seed + q)` yields one `NextFloat` per row, row-major
(anchor `rank.rs:482-487` `make_rands`; the exact consumption is proven by
`rank_parity.rs:277-289`). Seed = `seed + q` per query.

> **No Convert/Renew for rank** (base no-ops, CONTEXT specifics + RESEARCH anti-pattern) — do not
> synthesize them.

---

### `crates/lgbm-compute/src/kernels/mod.rs` (MODIFY — module index)

**Analog:** the existing Phase-14..18 `pub mod` block, `mod.rs:15-61`. Append a Phase-19 block in
the identical shape (ungated, NOT `#[cfg(feature = "gpu")]`, so the default cpu f64 anchor exercises
them — D-08; behind the OFF-by-default `LGBM_CUDA_ON_DEVICE` seam at runtime — D-06):
```rust
// Phase-19 on-device objectives (ODL-05/06/07/08): per-family grad/hess +
// BoostFromScore + ConvertOutput + RenewTreeOutput device kernels, composing the
// Phase-14 primitives + CUDARandom. Additive, OFF by default; ungated like the
// other kernel modules so the cpu f64 anchor exercises them (D-08).
pub mod objective_regression;
pub mod objective_binary;
pub mod objective_multiclass;
pub mod objective_rank;
```
There is currently **NO objective module** in `kernels/` (greenfield — Discrepancy 2).

---

### `crates/oracle-harness/tests/objective_parity.rs` (test, parity)

**Analog:** `crates/oracle-harness/tests/boosting_parity.rs` (grad/hess goldens) +
`crates/oracle-harness/tests/rank_parity.rs` (RNG-replay + per-query scores).

**Fixture-dir + capture-gated skip-pass** (copy `boosting_parity.rs:34-35` + `read_golden`
L323-336, or `rank_parity.rs:39-58`):
```rust
fn boosting_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/boosting")
}
fn read_golden(name: &str) -> Option<String> {
    match std::fs::read_to_string(boosting_dir().join(name)) {
        Ok(s) => Some(s),
        Err(_) => { eprintln!("... SKIP — golden not found ..."); None }
    }
}
```
A fresh checkout without the capture still builds and skip-passes (`Option` return, early `return`
in the test).

**`parse_gh` golden parser — REUSE VERBATIM** (`boosting_parity.rs:447-459`):
```rust
fn parse_gh(text: &str) -> (Vec<f32>, Vec<f32>) {
    // "GRAD <u32 bits...>" / "HESS <u32 bits...>" lines → f32::from_bits
}
fn parse_f32_bits_line(line: &str) -> Vec<f32> {           // L315-319
    line.split_whitespace().map(|t| f32::from_bits(t.parse::<u32>().unwrap())).collect()
}
```
The `multiclass_gh_*` goldens are stored **class-major** (12 rows × 3 classes = 36 values, GRAD then
HESS) and the comparator reads them in the same stride (RESEARCH Pattern 3) — the device kernel must
emit class-major so the flat compare lines up.

**Comparator selection per output class** (`oracle-harness/src/comparator.rs`):
```rust
compare_exact_u32(rust_bits, cpp_bits)          // L125 — bit-exact (no accumulation)
compare_within(rust_f32, cpp_f32, ORACLE_TOL)   // L92  — tol=1e-6 (atomic/transcendental)
// ORACLE_TOL: f32 = 1e-6  (comparator.rs:15)
```
Assertion policy is fully specified in RESEARCH §"Anchor & Tolerance Policy": elementwise grad/hess
→ `compare_exact_u32`; binary logit init + lambdarank λ/hess (atomic-order) → `compare_within`;
RNG stream → bit-exact `compare_exact_u32` (mirror `rank_parity.rs:288`).

**RNG-replay cell for rank** — copy `rank_parity.rs:249-305` (`rank_xendcg_objseed_rng_replay`): the
per-query `Random(seed+q).next_float()` draw order asserted bit-exact via `compare_exact_u32` on
`to_bits()`. For the device path, compare `draw_next_float_on` output against the same host
`Random` stream (NEVER GPU-vs-GPU, D-05).

---

### `xtask/py/rank_oracle_capture.py` (MODIFY — emit `lambdarank_gh` golden)

**Analog:** `xtask/py/boosting_oracle_capture.py` — the **score-derivation route** for
`*_gh_iter{1,N}.txt` (L273-357). RESEARCH Pitfall 2: the lambdarank grad/hess golden is the ONE
capture gap (L2/binary/multiclass already in git).

**Score-derivation route to mirror** (`boosting_oracle_capture.py:325-357`):
```python
# gh_fn(score_prev, labels) -> (grad f32, hess f32) is the objective math;
# grad1,hess1 = gh_fn(score0, labels)  → write GRAD/HESS <f32 bits> lines
# gradN,hessN = gh_fn(score_prev, labels)  at LATER_ITER
```
For lambdarank, `gh_fn` is the within-query λ math (`rank.rs:274-366` semantics): needs the
captured per-iteration raw scores + labels + `query_boundaries` + σ + label-gains + inv-max-dcg
(RESEARCH Open Q1 / A1). The existing `rank_oracle_capture.py` already captures the raw scores +
`query_boundaries` (`{obj}_scores.txt`, see its header §(2)) and ports the C++ `Random` LCG
(L73-83) — extend it to emit `lambdarank_gh_iter{1,N}.txt` in the `GRAD/HESS <u32 bits>` format.
Bit-helpers already present: `f32_bits` (L64-65), `f64_bits` (L60-61). **A1 fallback:** if the λ
accumulation is hard to reproduce in-script, intercept via a custom `fobj`. **A4:** confirm the uv
`.venv` still has `lightgbm==4.6` before the capture task.

---

## Shared Patterns

### Kernel module skeleton (imports + docstring + single-owner determinism)
**Source:** `crates/lgbm-compute/src/kernels/random.rs:42-50` + `primitives.rs:43-56`
**Apply to:** all four `objective_*.rs`
```rust
use cubecl::prelude::*;
use crate::error::ComputeError;
```
Module docstring cites the C++ source, the "Analog file" (`histogram.rs` /`primitives.rs`), and the
"cpu anchor stays a plain serial f64 fold (D-10)". Draw/reduce kernels launch single-owner
(`CubeDim::new_1d(1)`) — one unit walks tasks in ascending order, disjoint output windows, so the
result is bit-stable on cpu AND hip and identical to a task-parallel launch (`random.rs:27-33`).

### Host launcher: create → launch_unchecked → read, with V5 length guard
**Source:** `primitives.rs:726-778`, `random.rs:165-197`
**Apply to:** every `pub fn *_on<R: cubecl::Runtime>(client, ...) -> Result<_, ComputeError>`
Validate slice lengths / `n*k` overflow at the V5 boundary BEFORE any launch
(`random.rs:146-154`, `primitives.rs:731-740`); wrap the launch in a `// SAFETY:` comment proving
sized-exactly + outlives-launch + in-range writes (CMP-01); finish with `read_one_unchecked` +
`from_bytes`.

### Pre-allocate scratch ONCE (D-09)
**Source:** `primitives.rs:8-11` (reuse-once `client.empty` scratch), CubeCL manual
`13_memory_preallocation.md`
**Apply to:** multiclass softmax buffer, rank item-rand buffer, rank params buffer, per-block
reduction partials — created once outside the hot loop, never per-call in-kernel alloc.

### Anchor discipline: cpu f64 fold, NEVER GPU-vs-GPU (D-01/D-05)
**Source:** `random.rs:8-12` ("host `Random` IS the oracle … NEVER GPU-vs-GPU, D-10"),
`def-f8u-01` memory note
**Apply to:** every parity cell. Atomic-order outputs (binary init, lambdarank λ) → envelope
(`compare_within`), never a second GPU run.

### Capture-gated skip-pass fixture read
**Source:** `boosting_parity.rs:323-336`, `rank_parity.rs:45-58`
**Apply to:** every golden read in `objective_parity.rs` — `Option<String>` + early `return` so a
fresh checkout builds green pre-capture.

---

## No Analog Found / Discrepancies

The planner MUST surface these two RESEARCH-confirmed discrepancies in the plan (they are NOT
"missing analogs" so much as **traps where a named primitive does not exist**):

| Item | Role | Data Flow | Reality (verified) |
|------|------|-----------|--------------------|
| `percentile_device` per-leaf / per-segment kernel | kernel | per-leaf reduce | **DOES NOT EXIST.** `primitives.rs` has ONLY `percentile_unweighted_f32_on` (L1179) / `percentile_weighted_f32_on` (L1246) — both take a **host `&[f32]` whole-array slice** and return a **single scalar**. The per-*segment* primitive is **argsort only** (`bitonic_argsort_items_on`, L1361, which itself loops on the host over segments). RenewTreeOutput's per-leaf median AND RankXENDCG's per-query softmax must be **host-orchestrated loops** over leaves/queries calling the whole-array primitives (parity-identical, cheap; a true device block-per-leaf kernel is a Phase-21/perf option, NOT a Phase-19 requirement). A task action that says "call `percentile_device`" or "reuse the per-segment percentile primitive" is a bug. |
| objective module in `kernels/` | kernel module | n/a | **GREENFIELD — none exists.** `mod.rs` grep confirms no `objective*` module; the only in-tree mentions of "objective" in `kernels/` are code comments in `primitives.rs`/`predict.rs`. All four `objective_*.rs` + the `mod.rs` exports are net-new. |

**Golden gap (not a discrepancy, but a Wave-0 capture task):** `lambdarank_gh_iter{1,N}.txt` does
NOT exist (L2/binary/multiclass/regression_l1/poisson/huber/fair/quantile goldens already in git).
One new capture via the `rank_oracle_capture.py` extension (score-derivation route).

---

## Metadata

**Analog search scope:** `crates/lgbm-compute/src/kernels/` (primitives, random, data_partition,
mod), `crates/lgbm-objective/src/` (regression, binary, multiclass, rank, lib), `crates/oracle-harness/`
(comparator, tests/boosting_parity, tests/rank_parity), `xtask/py/` (boosting_oracle_capture,
rank_oracle_capture).
**Files scanned:** 14
**Pattern extraction date:** 2026-07-01
