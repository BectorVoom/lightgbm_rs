# Phase 17: On-Device Best-Split Finder - Research

**Researched:** 2026-07-01
**Domain:** CubeCL device-kernel port of `cuda_best_split_finder.cu` (§8) — 3-stage split-finding pipeline, f64-fold anchor + f32 hip mirror
**Confidence:** HIGH (every numerical claim verified line-by-line against the committed C++ reference `LightGBM/src/treelearner/cuda/`)

## Summary

This phase is already tightly specified by `17-CONTEXT.md` (D-01…D-11). Research did NOT re-litigate decisions; it resolved the six open research flags the decisions delegated, by diffing the CUDA reference source (`cuda_best_split_finder.cu`, `cuda_leaf_splits.hpp`) against the existing Rust (`gain.rs`, `split.rs`, `split_info.rs`, `primitives.rs`) line-by-line.

**The single most important finding (D-02):** the CUDA gain device helpers are **bit-identical** to the existing `#[cube]` `crate::gain` functions **for the `USE_SMOOTHING=false` branch** — including the `USE_L1` path. The only structural difference (`ThresholdL1`'s sign handling) is a **no-op for `l1 >= 0`** (always true). Therefore the shared `#[cube]` gain functions **can be reused** for three of the four comptime flags. The `USE_SMOOTHING=true` (path_smooth) branch is **net-new** and must be transcribed faithfully (output-blend + `GetLeafGainGivenOutput`). See §D-02 for the exact deltas.

**The single most dangerous parity landmine (count recovery):** the CUDA core recovers counts with `__double2int_rn` = **round-to-nearest, ties-to-EVEN**. The existing host `split.rs` uses `RoundInt(x) = (int)(x + 0.5f)` = round-half-up-then-truncate. **These differ** — which is precisely why D-01 mandates a *separate* CUDA-core fold rather than anchoring the host scan. The new fold must implement round-ties-even (`f64::round_ties_even` on host; a manual even-rounding helper inside `#[cube]` if cubecl lacks the intrinsic).

**Primary recommendation:** Build one new module `crates/lgbm-compute/src/kernels/best_split.rs` with a single `#[cube]` numerical-core generic (`split_eval_body`) reused by cpu single-owner fold and hip block-parallel launch, reusing `crate::gain` for `USE_SMOOTHING=false` and adding a faithfully-transcribed smoothing branch. Wire 3 separate stage kernels (D-06) and the 8-int export. Anchor every field to a new CUDA-core f64 golden fixture set (§D-07). The gain math bit-identity means Phase 17's risk is concentrated in (a) count-recovery rounding, (b) the kEpsilon-add-then-subtract-back placement, (c) the 3-stage reduction order, and (d) the `assume_out_default_left` task-construction logic — all four documented below.

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Per-(leaf,feature) split evaluation (stage 1) | Device kernel (`lgbm-compute`) | cpu f64 fold anchor | §8.1 numerical core — one block per task, reads Phase-16 `hist_in_leaf` |
| Within-feature prefix-sum + count recovery | Device kernel | — | Interleaved `[2b]/[2b+1]` grad/hess scan; the D-03 purpose-built scan |
| Gain math (`GetSplitGains`/output/leaf-gain) | Shared `#[cube]` `crate::gain` | — | Bit-identical to CUDA for `USE_SMOOTHING=false`; smoothing branch added |
| Cross-feature reduce per leaf (stage 2) | Device kernel | — | `ReduceBestGain` over per-task records; smaller/larger duality |
| Cross-leaf argmax + 8-int export (stage 3) | Device kernel | Host (single 8-int readback) | The ONLY device→host transfer per iteration (SC#2) |
| SplitFindTask construction (`assume_out_default_left`) | Host (task builder) | — | Determined by missing-type + reverse at task-gen time, NOT by REVERSE alone |
| Task dispatch (categorical seam) | Host + device dispatch | Phase 22 fills eval | D-04: wire `is_categorical`/`is_one_hot` seam, defer eval math |
| RNG seed per task (USE_RAND) | Device (`CUDARandom` LCG) | Phase-14 `random.rs` | Extra-trees `rand_threshold` draw |

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

- **D-01: New CUDA-core-faithful f64 fold is THE bit-exact anchor — NOT the host `split.rs` serial scan.** The CUDA §8.1 core takes a different numerical path (block prefix-sum → complement-from-parent-totals → count recovery via `cnt_factor` + `__double2int_rn`). Build a single-owner `CubeDim(1)` f64 fold that faithfully mirrors the CUDA accumulation (one `#[cube]` generic; cpu = single-owner fold, hip = block-parallel).
- **D-02: Transcribe the CUDA gain device helpers verbatim — do NOT assume host-identity.** Research must diff the CUDA gain device functions against `crate::gain` and document every delta. If bit-identical, a shared `#[cube]` is fine; parity-conservative default is faithful transcription. *(Research verdict: bit-identical for USE_SMOOTHING=false; smoothing branch net-new — see §D-02.)*
- **D-03: Purpose-built per-task within-feature prefix-sum — NOT the generic `block_scan`.** Faithful to the interleaved `[2b]/[2b+1]` layout AND the forward/reverse default-bin scan direction. May borrow `primitives.rs` LDS/`SharedMemory`/`sync_cube()` idiom, not the generic `block_scan` segment contract.
- **D-04: Numerical stage-1 core only; wire the categorical dispatch seam; categorical eval math deferred to Phase 22.** Build the numerical `FindBestSplitsForLeafKernelInner` (+ REVERSE). Wire `SplitFindTask` `is_categorical`/`is_one_hot` dispatch so Phase 22 drops in the categorical core without reshaping the pipeline.
- **D-05: Build the `_GlobalMemory` stage-1 spill variant this phase**, anchored by the Phase-15 synthetic large-bin/global-spill column. Same gain math over strided global loops (`GlobalMemoryPrefixSum`, scratch `feature_hist_{grad,hess,stat}_buffer` + `feature_hist_index_buffer`) for blocks with more bins than threads.
- **D-06: Keep the 3 faithful separate stages** (stage1 per-task / stage2 cross-feature reduce / stage3 cross-leaf argmax + 8-int export) with the block-argmax reduction family. Preserves the single 8-int readback contract (SC#2) and the reduction order the anchor pins.
- **D-07: Full comptime flag fan-out — wire and anchor all four `<USE_RAND, USE_L1, USE_SMOOTHING, IS_LARGER>`.** Includes USE_RAND/extra-trees (Phase-14 `CUDARandom`) and USE_SMOOTHING/`path_smooth`. Requires extra-trees RNG-stream goldens and path_smooth smoothing goldens in addition to the default-template anchor.
- **D-08:** Anchor-pin every numeric output to the cubecl-cpu f64 fold; structure bit-exact; ROCm/CUDA f32 within ~1e-6; tie-aware where relevant; never GPU-vs-GPU (def-f8u-01). One `#[cube]` generic, comptime/runtime-split reduction order.
- **D-09:** `LGBM_CUDA_ON_DEVICE` OFF by default; CPU / ROCm / existing-host-CUDA paths byte-unchanged; full merge gate green (ODL-19 hard merge gate).
- **D-10:** NO f64 per-row hot loops in new kernels (5.4× consumer-NVIDIA f64 regression, spike-052); f64 permitted only in scalar/gain math where the reference uses it (the split-gain/count-recovery math is inherently f64/double in §8.1 — that stays).
- **D-11:** Pre-allocate the split-record + scratch buffers ONCE outside the hot loop (`split_info.rs` `DeviceSplitInfo::new` `client.empty` pattern; global-memory scratch pre-allocated once per D-05). No per-split in-kernel device alloc.

### Claude's Discretion

- Exact CubeCL module placement — likely a new `best_split.rs` (or extend `split.rs`) in `crates/lgbm-compute/src/kernels/`, reusing `split_info.rs` `SplitScalars`/`DeviceSplitInfo` for the per-task records and 8-int export buffer.
- Whether stage-2's `…AllBlocks` fold is a separate `<<<1,1>>>`-analog kernel or folded into stage-2 when `num_blocks_per_leaf == 1` (the common small case) — parity-neutral as long as the block-winner reduction order is fixed.
- Geometry tunables (`NUM_THREADS_PER_BLOCK_BEST_SPLIT_FINDER=256`, `NUM_THREADS_FIND_BEST_LEAF=256`, `NUM_TASKS_PER_SYNC_BLOCK=1024`, the smaller/larger stream split) are occupancy knobs with no parity impact — start from the faithful C++ constants; APU-aware autotune is a deferred perf option, not a parity requirement.

### Deferred Ideas (OUT OF SCOPE)

- **Categorical inner core** (one-hot singleton-left + many-cat `BitonicArgSort_1024` sweep, `cat_threshold[]` list, `cat_l2`) → **Phase 22** (D-04 wires only the dispatch seam).
- **Discretized / quantized split finder** (`FindBestSplitsDiscretizedForLeafKernel`, int32/int64 packed accumulator, `grad_scale`/`hess_scale`) → **v2 (QGD-02)**.
- **Data partition / tree mutation / prediction** (§9–10, Split-before-partition) → **Phase 18**.
- **APU-aware autotune of the best-split geometry** → deferred perf option (parity-neutral).
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| ODL-11 | On-device per-feature split evaluation (stage 1) — block prefix-sum → cumulative left/right sums, count recovery via `cnt_factor`, min-data/min-sum-hessian guards, gain math, forward/reverse default-bin scan, block argmax → per-task split record | §8.1 numerical core fully transcribed below (§"Stage 1 Numerical Core"); gain math bit-identity established (§D-02); count-recovery rounding landmine nailed (§"Count Recovery"); within-feature scan shape (§D-03) |
| ODL-12 | Cross-feature reduce (stage 2) + cross-leaf argmax (stage 3) producing chosen `(leaf, feature, threshold, default_left)` with single 8-int readback, and tie-aware `default_left` parity | §8.2/§8.3 reduction family + 8-int field layout (§"3-Stage Reduction & Export"); tie-aware default_left + `assume_out_default_left` task logic (§"Tie-Aware default_left") |
</phase_requirements>

## Standard Stack

No new external packages. Everything is already in the workspace.

| Component | Version | Purpose | Why Standard |
|-----------|---------|---------|--------------|
| `cubecl` | 0.10 (in tree) | Device kernels (`#[cube]`, `SharedMemory`, `sync_cube`, `launch_unchecked`) | Project constraint (CLAUDE.md): pure Rust, cubecl for compute |
| `cubecl-cpu` | 0.10 | f64-fold deterministic anchor backend | The hard merge gate (CLAUDE.md core value) |
| `cubecl-hip` | 0.10 | ROCm f32 mirror (~1e-6 gate) | Local ROCm APU validation |
| `crate::gain` | in tree | `threshold_l1`/`get_leaf_gain`/`get_split_gains`/`calculate_splitted_leaf_output` `#[cube]` fns | Already bit-identical to CUDA for non-smoothing (§D-02) |
| `kernels::split_info` | in tree | `SplitScalars`/`DeviceSplitInfo`/`DeviceBuffers` — pre-allocated `CUDASplitInfo` analog + 8-int export | Phase-14 storage, `new` allocates once (D-11) |
| `kernels::random` | in tree | Phase-14 `CUDARandom` LCG for USE_RAND `rand_threshold` | Bit-stream verified in Phase 14 |
| `kernels::primitives` | in tree | LDS `SharedMemory`/`sync_cube` idiom, block reductions, bitonic argsort | D-03 borrows the idiom (not the generic `block_scan`) |

**Installation:** none — all internal. No `npm view` / registry check applies (no external dependency added). Package Legitimacy Audit is therefore **N/A** (no packages installed this phase).

## Architecture Patterns

### System Architecture Diagram

```
Phase-16 hist_in_leaf (interleaved [2b]=grad, [2b+1]=hess, f64/hist_t)
Phase-15 resident dataset (feature meta: num_bin, mfb_offset, default_bin, missing_type)
CUDALeafSplitsStruct (parent_gain, sum_grad, sum_hess, num_data, parent_output)
        │
        ▼
  ┌──────────────────────── HOST: build SplitFindTask[] (once/tree) ─────────────────────────┐
  │ per inner_feature → {reverse, skip_default_bin, na_as_missing, assume_out_default_left,   │
  │   is_categorical, is_one_hot, hist_offset, mfb_offset, num_bin, default_bin, rand_thr}    │
  │ num_bin>2 & missing≠None: emit forward(assume_left=F)+reverse(assume_left=T) task PAIR    │
  │ else (num_bin≤2 or missing None): single reverse task, assume_left = (missing≠NaN)        │
  └──────────────────────────────────────────────────────────────────────────────────────────┘
        │
        ▼  STAGE 1 — FindBestSplitsForLeafKernel<USE_RAND,USE_L1,USE_SMOOTHING,IS_LARGER>
  ┌─────────────────────── one block per (leaf,feature) task, 256 threads ──────────────────┐
  │ each thread loads its bin (grad@2b,hess@2b+1) [forward|reverse read-index] │            │
  │ thread0 += kEpsilon to hess │ ShufflePrefixSum(grad), ShufflePrefixSum(hess)            │
  │ cumulative = scanned side (right if REVERSE, left if forward)                            │
  │ complement side = parent_total − cumulative   (NOT a 2nd scan)                           │
  │ count recovery: cnt = __double2int_rn(scanned_hess * cnt_factor); other = num_data−cnt   │
  │ guards: left_hess≥minH & left_cnt≥minData & right_hess≥minH & right_cnt≥minData [& RAND] │
  │ gain = GetSplitGains<L1,SMOOTH>(...); keep if gain > parent_gain+min_gain_to_split       │
  │ ReduceBestGain (warp shfl → block shared) → single winner thread writes CUDASplitInfo    │
  │   smaller-leaf task t → [t]; larger-leaf task t → [t+num_tasks]  (IS_LARGER comptime)    │
  │ dispatch: is_categorical → (Phase-22 seam, trap for now); >256 bins → _GlobalMemory core │
  └──────────────────────────────────────────────────────────────────────────────────────────┘
        │  per-task CUDASplitInfo[2·num_tasks]
        ▼  STAGE 2 — SyncBestSplitForLeafKernel (per leaf, cross-feature)
  ┌──────── block-reduce per-task (is_valid,gain) via ReduceBestGain → cuda_leaf_best_split_info ────────┐
  │ read_index = is_smaller ? task_index : task_index+num_tasks; write leaf+block·num_leaves;            │
  │ stamp inner_feature_index. …AllBlocks folds block-winners when num_blocks_per_leaf>1.               │
  │ SetInvalidLeafSplitInfoKernel marks invalid leaves.                                                  │
  └──────────────────────────────────────────────────────────────────────────────────────────────────────┘
        │  cuda_leaf_best_split_info[leaf]
        ▼  STAGE 3 — FindBestFromAllSplitsKernel (cross-leaf argmax) + PrepareLeafBestSplitInfo
  ┌── ReduceBestGainForLeaves over (gain,leaf_index) → best_leaf; SELF-INVALIDATE chosen leaf ──┐
  │ + freshly-created leaf slot (cur_num_leaves). Pack 8-int buffer, ONE device→host copy.      │
  └──────────────────────────────────────────────────────────────────────────────────────────────┘
        │
        ▼  8-int buffer → Phase 18 (CUDATree.Split → DataPartition.Split)
```

### Recommended Project Structure

```
crates/lgbm-compute/src/kernels/
├── best_split.rs        # NEW — the 3-stage pipeline (this phase)
│   ├── split_eval_body            #[cube] generic stage-1 numerical core (fwd+rev)
│   ├── split_eval_body_globalmem  #[cube] generic >256-bin spill variant (D-05)
│   ├── stage1 launch (cpu single-owner fold / hip block-parallel)
│   ├── stage2 sync-best-split-per-leaf
│   ├── stage3 find-best-from-all + prepare 8-int export
│   ├── SplitFindTask (Rust struct mirroring cuda_best_split_finder.hpp:28-41)
│   └── build_split_find_tasks (host task builder — assume_out_default_left logic)
├── split.rs             # UNCHANGED (host serial scan — not the device anchor, D-01)
├── split_info.rs        # REUSE DeviceSplitInfo/SplitScalars + 8-int buffer (D-11)
├── primitives.rs        # BORROW SharedMemory/sync_cube idiom for block scan+reduce (D-03)
├── random.rs            # REUSE CUDARandom for USE_RAND rand_threshold (D-07)
└── gain.rs              # REUSE for USE_SMOOTHING=false; ADD smoothing branch (§D-02)

crates/oracle-harness/tests/
└── best_split_parity.rs # NEW — golden anchor tests (§D-07 fixture matrix)
crates/oracle-harness/tests/fixtures/kernels/
└── best_split.txt       # NEW — CUDA-core golden fixtures (default/rand/smooth/globalmem)
```

### Pattern 1: One `#[cube]` generic, comptime/runtime-split reduction (D-01/D-06/D-08)

**What:** A single `split_eval_body` `#[cube]` fn holds the numerical core (prefix-sum → complement → count-recovery → guards → gain → argmax). The cpu launch drives it single-owner (`CubeDim(1)`, serial accumulate — the deterministic fold). The hip launch drives it block-parallel (256 threads, `ShufflePrefixSum`-analog + `ReduceBestGain`).
**When to use:** Every numeric kernel in this milestone (established Phase 14–16).
**Example (the established pattern from `split.rs` / `primitives.rs`):**
```rust
// Source: crates/lgbm-compute/src/kernels/primitives.rs:82-181 (block_scan_body + f64/f32 wrappers)
#[cube]
fn split_eval_body<N: Numeric>(/* hist, task scalars, parent totals, out */) { /* ... */ }

#[cube(launch_unchecked)]
fn split_eval_kernel_f64(/* … */) { split_eval_body::<f64>(/* … */); }
#[cube(launch_unchecked)]
fn split_eval_kernel_f32(/* … */) { split_eval_body::<f32>(/* … */); }
```

### Pattern 2: comptime flag fan-out over 4 bools (D-07)

`FindBestSplitsForLeafKernel<USE_RAND, USE_L1, USE_SMOOTHING, IS_LARGER>` — the C++ fans out via `Inner0/1/2` nested dispatch (`extra_trees_`/`lambda_l1_`/`use_smoothing_`). In CubeCL these become **comptime generics** (`#[comptime]` bool params) OR runtime `u32` flags (0|1) as `split.rs` already does with `use_l1`/`skip_default_bin`. The existing `split.rs` uses **runtime flags** (`use_l1: u32`) inside one shared body — recommend the same to avoid a 16-way monomorphization fan-out on cubecl-cpu (which is slow to compile). `IS_LARGER` is only a task-index base selector (`[t]` vs `[t+num_tasks]`) and a smaller/larger stream split; it does not change the inner math, so keep it a launch-time parameter, not a body generic.

### Anti-Patterns to Avoid

- **Anchoring the host `split.rs` serial scan (rejected by D-01).** It accumulates left-sums incrementally and rounds counts with `RoundInt` (`(int)(x+0.5f)`); the CUDA core scans-then-complements and rounds with `__double2int_rn` (ties-even). Anchoring the host path masks a real divergence.
- **A second reverse scan for the complement side (wrong).** The CUDA core derives the non-scanned side by `parent_total − cumulative`, NOT a second scan. `REVERSE` only flips the *default-bin scan direction* and the recorded *threshold offset*.
- **GPU-vs-GPU comparison (def-f8u-01).** Never compare two nondeterministic f32 paths; pin both to the cpu f64 fold (structure bit-exact, values ≤~1e-5 f32 envelope).
- **f64 per-row hot loop (D-10, spike-052).** Consumer-NVIDIA f64 is 1/32 f32 → a 5.4× regression. Keep f32/u64 accumulation; f64 only in the scalar gain/count math where §8.1 uses `double`.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Gain / leaf-output / L1 threshold math | New gain formulas | `crate::gain::{get_split_gains,calculate_splitted_leaf_output,get_leaf_gain,threshold_l1}` (+f32 mirrors) | Verified bit-identical to CUDA for USE_SMOOTHING=false (§D-02) |
| Split-record device storage / 8-int export | New SoA buffer | `split_info::DeviceSplitInfo` / `SplitScalars` / `DeviceBuffers` | Pre-allocated once (D-11), field list already mirrors `CUDASplitInfo` |
| Extra-trees per-task RNG | New LCG | `random::CUDARandom` (`NextInt(0, num_bin-2)`) | Bit-stream verified Phase 14 |
| LDS block scan / reductions idiom | New sync primitives | `primitives.rs` `SharedMemory`/`sync_cube()` idiom (borrow shape) | The D-03 scan and ReduceBestGain build on it |
| Round-to-nearest-even | `(int)(x+0.5)` | `f64::round_ties_even()` host / manual even-round in `#[cube]` (§"Count Recovery") | CUDA uses `__double2int_rn` — the `split.rs` `round_int` is the WRONG rounding for this path |

**Key insight:** The gain math is a solved problem (reuse `crate::gain`). Phase 17's real work is the *accumulation path around* the gain math — prefix-sum, complement, count-rounding, epsilon placement, and the 3-stage reduction order — where the CUDA core diverges from the host path.

## Runtime State Inventory

Not a rename/refactor/migration phase — this is additive greenfield device-kernel code behind `LGBM_CUDA_ON_DEVICE`. **None — verified: no stored data, service config, OS-registered state, secrets, or build artifacts carry a renamed string. The only persisted state touched is the pre-allocated device buffers (`DeviceSplitInfo`), created fresh per learner instance.**

## Common Pitfalls

### Pitfall 1: Count-recovery rounding mismatch (`__double2int_rn` vs `RoundInt`)
**What goes wrong:** Counts recovered with `(int)(x+0.5f)` diverge from CUDA's `__double2int_rn` at exact half-values, flipping a `left_count >= min_data_in_leaf` guard and silently changing the winning threshold.
**Why it happens:** The existing `split.rs::round_int` transcribes the **host** `Common::RoundInt` (`common.h:904`), not the CUDA `__double2int_rn`. D-01 explicitly builds a *separate* fold for exactly this reason.
**How to avoid:** Implement round-ties-to-even. Host/cpu-fold: `f64::round_ties_even(x) as i32`. Verify cubecl-cpu lowers `round_ties_even`; if not, use the branch-free even-rounding identity (§"Count Recovery"). The hip f32 mirror must round-ties-even in f32 too.
**Warning signs:** A golden case where a bin's `scanned_hess * cnt_factor` lands on `k.5`; count off by 1; threshold off by 1; a `default_left` flip on a non-tie (hard-fail).

### Pitfall 2: kEpsilon added-then-subtracted at different phases
**What goes wrong:** In the **guard/scan phase** the count is recovered from the hessian *including* the `kEpsilon` seed (thread 0 adds it before the prefix sum, lines 205-206). In the **final write phase** the code subtracts kEpsilon *before* recovering the stored count (`sum_right_hessian = local_hess_hist - kEpsilon;` line 275, then `__double2int_rn`). The reported `left_sum_hessian`/`right_sum_hessian` therefore have kEpsilon subtracted off, and the stored counts use the kEpsilon-subtracted hessian.
**Why it happens:** Faithful reproduction of `FindBestThresholdSequentially`'s epsilon contract (kEpsilon-seed for numeric safety, subtract-back for the recorded value). `split.rs` already documents this exact placement.
**How to avoid:** Two count recoveries: one in the guard (kEpsilon-included), one at write (kEpsilon-subtracted). Do NOT reuse the guard-phase count for the stored record.
**Warning signs:** Off-by-one in stored `left_count`/`right_count` vs the guard's count.

### Pitfall 3: `default_left = assume_out_default_left`, NOT `REVERSE`
**What goes wrong:** Setting `default_left = REVERSE` (as the host `split.rs` does — `output->default_left = REVERSE`) diverges from the CUDA core, which writes `default_left = task->assume_out_default_left` (lines 272, 470). These correlate but are not equal: for a MissingType::None or num_bin≤2 feature the single task is reverse=true yet `assume_out_default_left = (missing != NaN)`.
**Why it happens:** The CUDA path precomputes `default_left` at task-construction time from the missing type, not from the scan direction.
**How to avoid:** Reproduce `build_split_find_tasks` exactly (§"Tie-Aware default_left" — the full task-gen table). Carry `assume_out_default_left` in the Rust `SplitFindTask`; write it verbatim into the record.
**Warning signs:** `default_left` parity fails on features with no missing values or with only 2 bins.

### Pitfall 4: Complement side derived by a second scan
**What goes wrong:** Running a reverse prefix-sum to get the right side (when forward) produces a *different f64 accumulation order* than `parent_total − left_cumulative`, breaking bit-exactness.
**How to avoid:** Only ONE scan direction per task. The other side is `sum_gradients − scanned` / `sum_hessians − scanned`. (See lines 217-219 reverse, 241-243 forward.)

### Pitfall 5: Reduction order in `ReduceBestGain` argmax
**What goes wrong:** A different tie-break in the warp→block argmax picks a different equal-gain thread, changing the threshold.
**Why it happens:** `ReduceBestGain` compares `(gain, found, thread_index)`; ties resolve deterministically by the reduction tree order. On the cpu single-owner fold this is a serial `>` scan (first-max-wins by ascending thread index); the hip block reduction must match that tie-break.
**How to avoid:** cpu fold: strict `>` keeps the FIRST (lowest thread index) on ties, matching `split.rs`'s `take = gain > best_gain` convention. hip: the warp shuffle reduction must use the same strict-`>` so the lower `thread_index` survives a tie. Spike-022 confirms within-feature reorder tie-flips are cosmetic within ~1e-6 but the *anchor* fold must be deterministic.
**Warning signs:** threshold differs by exactly the gap between two equal-gain bins.

## Code Examples

### D-02: The CUDA gain device helpers (verbatim, the diff target)
```cpp
// Source: LightGBM/src/treelearner/cuda/cuda_leaf_splits.hpp:65-140
__device__ static double ThresholdL1(double s, double l1) {
  const double reg_s = fmax(0.0, fabs(s) - l1);
  if (s >= 0.0f) { return reg_s; } else { return -reg_s; }   // (A)
}
template <bool USE_L1, bool USE_SMOOTHING>
__device__ static double CalculateSplittedLeafOutput(double sum_gradients,
    double sum_hessians, double l1, double l2, double path_smooth,
    data_size_t num_data, double parent_output) {
  double ret;
  if (USE_L1) { ret = -ThresholdL1(sum_gradients, l1) / (sum_hessians + l2); }
  else        { ret = -sum_gradients / (sum_hessians + l2); }
  if (USE_SMOOTHING) {                                        // (B) NET-NEW
    ret = ret * (num_data / path_smooth) / (num_data / path_smooth + 1)
        + parent_output / (num_data / path_smooth + 1);
  }
  return ret;
}
template <bool USE_L1, bool USE_SMOOTHING>
__device__ static double GetLeafGain(double sum_gradients, double sum_hessians,
    double l1, double l2, double path_smooth, data_size_t num_data, double parent_output) {
  if (!USE_SMOOTHING) {                                       // (C) == crate::gain::get_leaf_gain
    if (USE_L1) { const double sg=ThresholdL1(sum_gradients,l1); return (sg*sg)/(sum_hessians+l2); }
    else        { return (sum_gradients*sum_gradients)/(sum_hessians+l2); }
  } else {                                                    // (D) NET-NEW: output-blend form
    const double output = CalculateSplittedLeafOutput<USE_L1,USE_SMOOTHING>(
        sum_gradients, sum_hessians, l1, l2, path_smooth, num_data, parent_output);
    return GetLeafGainGivenOutput<USE_L1>(sum_gradients, sum_hessians, l1, l2, output);
  }
}
```

### Stage-1 numerical core — forward branch write (the exact accumulation to fold)
```cpp
// Source: LightGBM/src/treelearner/cuda/cuda_best_split_finder.cu:296-318 (forward finalize)
const double sum_left_gradient  = local_grad_hist;                  // scanned (inclusive prefix)
const double sum_left_hessian   = local_hess_hist - kEpsilon;       // kEpsilon subtracted at WRITE
const data_size_t left_count    = __double2int_rn(sum_left_hessian * cnt_factor);   // ties-even
const double sum_right_gradient = sum_gradients - sum_left_gradient;                // complement
const double sum_right_hessian  = sum_hessians  - sum_left_hessian - kEpsilon;
const data_size_t right_count   = num_data - left_count;
const double left_output  = CUDALeafSplits::CalculateSplittedLeafOutput<USE_L1,USE_SMOOTHING>(
    sum_left_gradient, sum_left_hessian, lambda_l1, lambda_l2, path_smooth, left_count, parent_output);
// ... right_output symmetric; left_gain/right_gain via GetLeafGainGivenOutput<USE_L1>
```

### 3-stage export — the 8-int buffer (SC#2, the only device→host transfer)
```cpp
// Source: LightGBM/src/treelearner/cuda/cuda_best_split_finder.cu:2130-2158, 2181
// [0]=smaller.inner_feature_index [1]=smaller.threshold [2]=smaller.default_left
// [3]=larger.inner_feature_index  [4]=larger.threshold  [5]=larger.default_left
// [6]=best_leaf_index (cross-leaf argmax winner) [7]=best_leaf.num_cat_threshold
cuda_best_split_info_buffer[6] = best_leaf_index;
if (best_leaf_index != -1) {
  cuda_leaf_best_split_info[best_leaf_index].is_valid = false;   // SELF-INVALIDATE chosen leaf
  cuda_leaf_best_split_info[cur_num_leaves].is_valid  = false;   //  + freshly-created leaf slot
  cuda_best_split_info_buffer[7] = cuda_leaf_best_split_info[best_leaf_index].num_cat_threshold;
}
```

## State of the Art

| Old (host `split.rs`) | Current (CUDA §8.1 core, this phase) | Impact |
|-----------------------|--------------------------------------|--------|
| Incremental left-sum accumulation | Block prefix-sum → complement-from-parent | Different f64 order → separate anchor (D-01) |
| `RoundInt(x)=(int)(x+0.5f)` | `__double2int_rn` (round ties-even) | Count landmine — new even-rounding helper |
| `default_left = REVERSE` | `default_left = task->assume_out_default_left` | Task-gen logic must be ported (Pitfall 3) |
| Single-feature per-launch scan (feature-per-lane, spike-021) | One block per (leaf,feature) task, 256-thread block scan | New stage-1 launch geometry |
| USE_SMOOTHING/USE_RAND: not in host default template | Full 4-flag fan-out (D-07) | Smoothing branch + RNG stream net-new |

**Deprecated/outdated:** `FindBestSplitsDiscretizedForLeafKernel` (quantized inner) is explicitly out of scope → v2 (QGD-02). `FindBestSplitsDiscretizedForLeafKernel_GlobalMemory` does not exist even in C++ (a `TODO`) — do not port.

---

## Deep-Dive: Resolved Research Flags

### D-02 — Gain-math delta diff (VERDICT: reuse for USE_SMOOTHING=false; smoothing net-new)

Line-by-line diff of CUDA `cuda_leaf_splits.hpp` device helpers vs Rust `crate::gain`:

| Function | CUDA (device) | Rust `crate::gain` | Delta |
|----------|---------------|--------------------|-------|
| `ThresholdL1` | `reg_s=fmax(0,fabs(s)-l1); s>=0.0f ? reg_s : -reg_s` | `reg_s=max(0,abs(s)-l1); (sign_pos−sign_neg)*reg_s` | **No delta for l1≥0.** At s=0: CUDA→`+reg_s`, Rust→`0·reg_s`. reg_s(s=0)=`max(0,−l1)=0` when l1≥0 (always) ⇒ both 0. Bit-identical. |
| `CalculateSplittedLeafOutput` (SMOOTH=false) | `-Thr/(h+l2)` or `-g/(h+l2)` | identical | **Bit-identical.** |
| `CalculateSplittedLeafOutput` (SMOOTH=true) | + blend `ret·(n/ps)/(n/ps+1) + parent/(n/ps+1)` | **absent** | **NET-NEW branch.** Needs `path_smooth, num_data, parent_output` args. |
| `GetLeafGain` (SMOOTH=false) | `(sg·sg)/(h+l2)` or `(g·g)/(h+l2)` | identical (`get_leaf_gain`) | **Bit-identical.** |
| `GetLeafGain` (SMOOTH=true) | `output=CalcSplitOut(...); GetLeafGainGivenOutput(...)` | **absent** | **NET-NEW.** Uses given-output form, NOT the `sg²/(h+l2)` closed form. |
| `GetLeafGainGivenOutput` | `-(2·sg·o + (h+l2)·o·o)` | identical (`get_leaf_gain_given_output`, host fn) | **Bit-identical.** Must be promoted to `#[cube]` if the smoothing gain path runs on device. |
| `GetSplitGains` | `GetLeafGain(L)+GetLeafGain(R)` w/ counts+parent | `get_split_gains` (no counts/parent/smooth) | **Signature extension** — add `left_count,right_count,path_smooth,parent_output` (unused when SMOOTH=false). |

**Epsilon placement:** identical semantics — `split.rs` documents `2·kEpsilon` at scan entry vs the CUDA core's single `kEpsilon` at thread-0 (line 206) then subtract-back (lines 275/298). Note the CUDA best-split-finder adds `kEpsilon` **once** (thread 0), not `2·kEpsilon`; the `2·kEpsilon` in `split.rs` is the host `FindBestThresholdSequentially` convention (seeds BOTH left and right hessian). The new fold must follow the **CUDA** single-kEpsilon-at-thread-0 placement, not the host `2·kEpsilon`.

**Branch/gate order:** identical — guard order is `left_hess ≥ minH && left_cnt ≥ minData && right_hess ≥ minH && right_cnt ≥ minData [&& rand]` then `gain > min_gain_shift`. `min_gain_shift = parent_gain + min_gain_to_split`; stored `gain = current_gain − min_gain_shift`.

**Conclusion:** A shared `#[cube]` is correct for `USE_SMOOTHING=false` (3 of 4 flags). For `USE_SMOOTHING=true`, add a faithfully-transcribed smoothing branch (both `CalculateSplittedLeafOutput` and `GetLeafGain`) and promote `get_leaf_gain_given_output` to `#[cube]`. `GetSplitGains`/`get_split_gains` gains `left_count,right_count,path_smooth,parent_output` params (ignored when smoothing off). **`parent_output`** comes from the `CUDALeafSplitsStruct` (the leaf's own output); thread it into the kernel scalars.

### D-03 — Within-feature 256-bin scan (interleaved layout + LDS block scan)

**Confirmed layout:** histogram is interleaved `hist[2b]=grad`, `hist[2b+1]=hess` (Phase-16 `hist_in_leaf`, matches `GET_GRAD/GET_HESS(i)=hist[i<<1]/[(i<<1)+1]` from `split.rs:174`). Each thread `t` reads ONE bin:
- **Forward** (non-na): `bin_offset = t << 1` for `t < feature_num_bin_minus_offset && !skip_sum` (lines 180-184).
- **Forward na_as_missing & mfb_offset==1:** `bin_offset = (t-1) << 1` for `0 < t < num_bin` (lines 173-178).
- **Reverse:** `read_index = feature_num_bin_minus_offset − 1 − t`, `bin_offset = read_index << 1` (lines 187-193).

The scan operates on **two separate scalars** (grad, hess) via **two separate `ShufflePrefixSum` calls** (lines 209, 211) — the interleave is only the READ pattern, not the scan. This resolves the ROADMAP flag: it is NOT a fused interleaved scan.

**LDS block-scan shape needed (the ROADMAP plane-sum-caps-at-32/64 flag):** `ShufflePrefixSum` is a **block-wide inclusive scan over 256 elements** = warp-shuffle intra-warp prefix (32/64 lanes) + a shared-memory cross-warp carry (8 warp-sums scanned, added back). This is the classic two-level scan. The `primitives.rs` generic `block_scan` is a **single-owner serial** (`UNIT_POS==0`) per-block scan with a separate `scan_block_totals` + `add_base` multi-kernel recombination — a DIFFERENT contract (segments across CUBEs, not a within-block warp scan). **Build net-new for hip:** a within-block two-level LDS scan (borrow the `SharedMemory::new`/`sync_cube()` idiom from `primitives.rs`, not the `block_scan` body). **For the cpu fold:** a `CubeDim(1)` serial inclusive accumulate over the task's bins IS bit-exact and matches the single-owner pattern already in `block_scan_body` — reuse that shape (serial loop, forward/reverse variant).

**Directionality:** forward scan gives cumulative-LEFT (`sum_left = prefix`); reverse scan gives cumulative-RIGHT (`sum_right = prefix`). Threshold recorded:
- Reverse: `threshold = num_bin − 2 − t` (line 230); default_left from `assume_out_default_left`.
- Forward: `threshold = (na_as_missing && mfb_offset==1) ? t : t + mfb_offset` (lines 254-256).

*(Note: the CONTEXT summary "`t-1+offset` reverse vs `t+offset` forward" is the HOST `split.rs` convention; the CUDA core uses `num_bin-2-t` reverse / `t+mfb_offset` forward — port the CUDA form for this fold.)*

### D-07 — Fixture / golden matrix (the 4-flag anchor)

The existing `kernel_parity.rs` harness pattern: parse a `fixtures/kernels/*.txt` golden (C++-transcription values as f64/f32 bit-hex), drive the cubecl-cpu op, assert `compare_exact_f64_bits` for structure/f64 and `ORACLE_TOL=1e-6` for the hip f32 mirror. Follow it for `best_split.txt` / `best_split_parity.rs`.

| Golden | Flags exercised | Generated by (C++ reference config) | Anchor asserts |
|--------|-----------------|--------------------------------------|----------------|
| **default-template** | `<F,F,F,F>` + `<F,F,F,T>` (IS_LARGER) | Default config, no L1/smoothing/extra-trees, a continuous feature with missing + a clean feature; smaller & larger leaf | threshold, default_left, left/right sum grad+hess, left/right count, value, gain — `compare_exact_f64_bits` |
| **USE_L1** | `<F,T,F,·>` | `lambda_l1 = 0.1` | same fields; verifies `ThresholdL1` path bit-exact |
| **USE_SMOOTHING** | `<F,·,T,·>` | `path_smooth = 2.0` (config: `path_smooth>0`); needs `parent_output` in the leaf struct | value + gain via the output-blend form; the net-new branch |
| **USE_RAND (extra-trees)** | `<T,·,·,·>` | `extra_trees = true` (`extra_trees_` → USE_RAND); a fixed RNG seed | the `rand_threshold = CUDARandom.NextInt(0, num_bin-2)` draw sequence is **bit-identical** to Phase-14 `CUDARandom` (assert the drawn threshold index + that the chosen split is the rand-selected one, not the max-gain one) |
| **empty / sparse-default-bin** | `<F,F,F,·>` edge | a leaf with no valid split (all bins fail guards); a feature whose winning bin is the default/most-freq bin (skip_default_bin) | `is_valid=false` sentinel; `skip_sum` correctly skips the default bin |
| **global-memory spill (D-05)** | `<F,F,F,·>` + >256 bins | Phase-15 synthetic large-bin column (num_bin > 256, reuse Phase-16 D-04 fixture) | same numeric fields via the `_GlobalMemory` strided-loop core |

**RNG-stream golden detail (D-07):** `InitCUDARandomKernel` seeds a per-task `CUDARandom`; stage-1 thread 0 draws `NextInt(0, num_bin-2)` into `rand_threshold` (line 160). The extra-trees golden must verify the Rust `random::CUDARandom` produces the *same* `rand_threshold` for the same seed/task — i.e. re-assert the Phase-14 bit-identity in this kernel's context, then assert the split chosen is the one at `rand_threshold` (USE_RAND restricts the candidate to a single random threshold per feature; lines 222, 246).

**Generation approach:** These are CUDA-core goldens, but the *deterministic values* can be produced by the new cpu f64 fold itself once cross-checked against a real-`lib_lightgbm` capture at the fixture inputs — the same "C++-transcription golden" method `kernel_parity.rs` already uses for histograms. Where a real capture is impractical for a device-only kernel, the golden is the transcribed-by-hand expected record (as the histogram golden is), with the fold asserted bit-exact against it.

### Count Recovery + complement-from-parent (the exact reproduction)

**`cnt_factor`** = `num_data / sum_hessians` — an f64 division (line 147), computed ONCE per task from the leaf totals. `sum_hessians` here is the **leaf total** (from `CUDALeafSplitsStruct`), NOT the kEpsilon-seeded per-thread value.

**`__double2int_rn`** = round to nearest, ties to even (IEEE round-half-to-even). Rust equivalents:
- **cpu fold / host:** `x.round_ties_even() as i32` (stable since Rust 1.77). **Verify cubecl-cpu lowers `round_ties_even`** inside `#[cube]`; the current `split.rs::round_int` avoids it. If unsupported, use the branch-free identity:
  ```rust
  // round-half-to-even without an intrinsic (x >= 0 here: hessian·cnt_factor ≥ 0)
  let f = x.floor();
  let diff = x - f;                       // in [0,1)
  let up = diff > 0.5
      || (diff == 0.5 && ((f as i64) & 1 == 1));  // tie → round to even
  let r = if up { f + 1.0 } else { f };
  r as i32
  ```
- **hip f32 mirror:** same logic in f32 (`f32::round_ties_even`), OR rely on the hip backend lowering `__double2int_rn`-equivalent. Keep it round-ties-even, NOT `(int)(x+0.5f)`.

**Complement-from-parent:** the non-scanned side is `sum_gradients − scanned_grad`, `sum_hessians − scanned_hess`; its count is `num_data − recovered_count` (lines 217-219 reverse, 241-243 forward). Never a second scan (Pitfall 4).

**Two-phase count subtlety (Pitfall 2):** guard phase recovers count from kEpsilon-*included* hessian (line 216/240); write phase subtracts kEpsilon first (line 275/298) then recovers. Reproduce both.

### D-05 — `_GlobalMemory` spill variant (>256-bin blocks)

For features with more bins than block threads (256), the standard shared-path can't hold the whole histogram in registers/LDS. The `_GlobalMemory` core (`FindBestSplitsForLeafKernelInner_GlobalMemory`) runs the **same gain math** over **strided global loops**:
- `GlobalMemoryPrefixSum` instead of the in-block `ShufflePrefixSum` — a multi-pass scan writing to global scratch.
- Scratch buffers (pre-allocated ONCE, D-11): `feature_hist_grad_buffer`, `feature_hist_hess_buffer`, `feature_hist_stat_buffer`, `feature_hist_index_buffer` — sized to the largest feature's bin count × num concurrent blocks.
- Each thread processes bins `t, t+blockDim, t+2·blockDim, …` (strided) rather than one bin.
- The discretized global-memory path does NOT exist in C++ (a `TODO`) and is out of scope anyway.

**Anchor:** the Phase-15 synthetic large-bin / global-spill column (reuse Phase-16 D-04's fixture, num_bin > 256). The cpu fold's serial single-owner loop naturally handles >256 bins (no register/LDS cap), so the cpu anchor is the same body with a larger loop bound; the hip `_GlobalMemory` kernel is the net-new strided variant asserted against it.

### 3-Stage Reduction Order & 8-int Export (SC#2)

**Reduction family order (fixed — the anchor pins it):**
- `ReduceBestGainWarp` (`__shfl_down_sync` intra-warp over `(gain, found, thread_index)`) → `ReduceBestGainBlock` (cross-warp via `__shared__[32]`) → `ReduceBestGain` (returns the winning `thread_index`). Tie-break: strict `>` ⇒ **lowest thread index survives a tie** (match on the cpu fold, Pitfall 5).
- Stage-3 cross-leaf: `ReduceBestGainForLeavesWarp` → `…Block` → `ReduceBestGainForLeaves` over `(gain, leaf_index)`.

**Smaller/larger task duality (IS_LARGER):** stage-1 smaller-leaf task `t` writes `CUDASplitInfo[t]`; larger-leaf task `t` writes `[t+num_tasks]` (§8 header + line 788). Stage-2 reads `read_index = is_smaller ? task_index : task_index + num_tasks` (line 1943). Smaller runs on stream 0, larger on stream 1 (parity-neutral; streams are just concurrency).

**Stage-3 self-invalidation (behavioral — must preserve):** after picking `best_leaf_index`, set `cuda_leaf_best_split_info[best_leaf_index].is_valid = false` AND `[cur_num_leaves].is_valid = false` (the freshly-created leaf slot) so neither is re-picked next iteration (lines 2131-2135).

**8-int buffer field layout (the single device→host copy):**

| idx | field |
|-----|-------|
| 0 | smaller leaf `inner_feature_index` |
| 1 | smaller leaf `threshold` |
| 2 | smaller leaf `default_left` |
| 3 | larger leaf `inner_feature_index` (only if `larger_leaf_index >= 0`) |
| 4 | larger leaf `threshold` |
| 5 | larger leaf `default_left` |
| 6 | `best_leaf_index` (cross-leaf argmax winner, `-1` if none) |
| 7 | best leaf `num_cat_threshold` (0 for continuous — Phase 22 fills for categorical) |

`DeviceSplitInfo`/`DeviceBuffers` already reserves these fields; the export is a 6-thread scalar fan-out (`PrepareLeafBestSplitInfo<<<6,1>>>`) + the argmax kernel writing [6]/[7]. **No other readback per iteration** (SC#2) — the full per-side sums/counts/values stay resident for Phase 18.

### Tie-Aware `default_left` (mandatory — do NOT defer)

**The parity rule (SC#3):** a `default_left` flip vs the cpu anchor is accepted **only** on a verified f32 tie (same threshold + same left_count + f32-equal gains); a flip on any non-tie split **hard-fails**; empty/sparse-default-bin fixtures pass. This mirrors the def-f8u-01 / hip-split-parity-preexisting-defect precedent (near-tie default_left flips are f32-vs-f64, not kernel bugs). Spike-022 corroborates: within-feature reorder default_left flips are cosmetic within ~1e-6.

**`assume_out_default_left` — the full task-gen table (host, `cuda_best_split_finder.cpp:137-227`):**

| Feature condition | Tasks emitted | `assume_out_default_left` |
|-------------------|---------------|---------------------------|
| `num_bin>2 && missing==Zero && !categorical` | forward (skip_default_bin, !na) **then** reverse (skip_default_bin, !na) | forward=**false**, reverse=**true** |
| `num_bin>2 && missing==NaN && !categorical` | forward (na_as_missing) **then** reverse (na_as_missing) | forward=**false**, reverse=**true** |
| `num_bin<=2 or missing==None`, non-categorical | single reverse task | `(missing != NaN) ? **true** : **false**` |
| categorical | single forward task (is_one_hot = num_bin ≤ max_cat_to_onehot) | `(missing != NaN && !categorical) ? true : false` ⇒ **false** for categorical (Phase 22 seam) |

The Rust `SplitFindTask` must carry `assume_out_default_left` and the host builder must reproduce this table exactly. The stage-1 kernel writes it verbatim (`default_left = assume_out_default_left`), NOT `REVERSE`.

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | `cubecl-cpu` 0.10 lowers `f64::round_ties_even` inside `#[cube]` | Count Recovery | LOW — fallback branch-free even-round identity provided; verify during Wave 0 |
| A2 | Reusing `crate::gain` (runtime `use_l1` flag) for USE_SMOOTHING=false is bit-exact to the CUDA templated helper | D-02 | LOW — verified line-by-line; the only ThresholdL1 delta is a proven no-op for l1≥0 |
| A3 | The extra-trees RNG goldens can be produced by re-asserting Phase-14 `CUDARandom` bit-identity at this kernel's seeds | D-07 | MEDIUM — depends on `InitCUDARandomKernel` seeding matching Phase-14 `random.rs` seed convention; confirm the per-task seed formula against `cuda_best_split_finder`'s InitCUDARandomKernel |
| A4 | The `_GlobalMemory` cpu fold is the same serial body with a larger loop bound (no separate cpu implementation needed) | D-05 | LOW — single-owner serial fold has no register/LDS cap; hip variant is the net-new part |
| A5 | Golden values for a device-only kernel can be hand-transcribed C++ expected records (as histogram goldens are) where a real capture is impractical | D-07 | MEDIUM — if a real `lib_lightgbm` CUDA capture is required, needs a CUDA build; the histogram-golden precedent suggests transcription is accepted |

## Open Questions (RESOLVED)

1. **`InitCUDARandomKernel` per-task seed formula (A3).** — RESOLVED.
   - What we know: stage-1 draws `NextInt(0, num_bin-2)`; Phase-14 `CUDARandom` LCG is bit-stream verified.
   - What was unclear: the exact per-task seed each task's `CUDARandom` is initialized with (whether it's `config.seed + task_index` or a derived stream).
   - **Resolution:** the per-task seed is `SetSeed(extra_seed + task_index)`, read directly from `InitCUDARandomKernel` in `cuda_best_split_finder.cu:2220-2228` (`cuda_randoms[task_index].SetSeed(seed + task_index)`). Locked in 17-01 Task 1 (module doc block + the `extra_seed + task_index` citation) and consumed by the USE_RAND path in 17-03 Task 1. No open flag remains for the Wave-2 kernel.

2. **Whether a real CUDA capture is needed for the goldens or hand-transcription suffices (A5).** — RESOLVED.
   - What we know: `kernel_parity.rs` uses hand-transcribed C++-transcription goldens for histograms.
   - What was unclear: whether the planner wants a real `lib_lightgbm` (CUDA build) A/B for the best-split goldens.
   - **Resolution:** hand-transcription is chosen, following the histogram-golden precedent (A5) — transcribed C++-core expected records stored as bit-hex, cross-checked bit-exact by the cpu f64 fold. No real discrete-CUDA capture is required. The golden matrix is built in 17-01 Task 3 and turned GREEN by the 17-03 Task 1 cpu fold. Provenance guard: any Wave-2 finalization of the golden numeric values MUST be re-derived from the C++ `cuda_best_split_finder.cu` accumulation, NOT copied from the cpu-fold output, to keep the anchor independent (see 17-01 Task 3 / 17-03 Task 1).

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| `cubecl` / `cubecl-cpu` | f64 fold anchor (all tasks) | ✓ (in tree) | 0.10 | — |
| `cubecl-hip` (ROCm) | f32 mirror parity gate | ✓ (spoofed 8-CU APU, gfx1152) | 0.10 | Parity gate valid; perf numbers APU-confounded (memory: rocm-gfx1100-available) |
| `LightGBM/` C++ reference | reading `cuda_best_split_finder.cu` etc. | ✓ (read-only, untracked) | 4.6 (195c26fc) | — |
| Real discrete CUDA | golden capture (only if A5 requires it) | ✗ locally | — | Kaggle CLI harness (authenticated `boomvector`) OR hand-transcribed goldens (histogram precedent) |

**Missing dependencies with fallback:** real discrete CUDA — use hand-transcribed goldens (the established histogram method) or the Kaggle harness if a real capture is mandated.
**Missing dependencies with no fallback:** none — this phase is additive device-kernel code anchored to the always-available cubecl-cpu fold.

## Validation Architecture

*(nyquist_validation = true → section included.)*

### Test Framework
| Property | Value |
|----------|-------|
| Framework | Rust `cargo test` (`oracle-harness` integration tests + `lgbm-compute` unit tests) |
| Config file | none (Cargo built-in) |
| Quick run command | `cargo test -p lgbm-compute --lib best_split` |
| Full suite command | `cargo test -p oracle-harness --test best_split_parity && cargo test --workspace` |

### Phase Requirements → Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| ODL-11 | Stage-1 per-task record bit-exact to CUDA-core f64 fold (threshold, sums, counts, value, gain) | integration (golden) | `cargo test -p oracle-harness --test best_split_parity stage1` | ❌ Wave 0 |
| ODL-11 | Count recovery `__double2int_rn` ties-even at a `k.5` fixture | unit | `cargo test -p lgbm-compute --lib count_recovery_ties_even` | ❌ Wave 0 |
| ODL-11 | USE_L1 / USE_SMOOTHING / USE_RAND branch goldens | integration | `cargo test -p oracle-harness --test best_split_parity flags` | ❌ Wave 0 |
| ODL-11 | `_GlobalMemory` spill variant on the Phase-15 large-bin fixture | integration | `cargo test -p oracle-harness --test best_split_parity globalmem` | ❌ Wave 0 |
| ODL-12 | 8-int export field layout + best_leaf argmax + self-invalidation | integration | `cargo test -p oracle-harness --test best_split_parity stage3_export` | ❌ Wave 0 |
| ODL-12 | Tie-aware `default_left` (flip only on verified f32 tie; non-tie flip hard-fails) | integration (hip) | `cargo test -p lgbm-compute --test rocm_backend_parity default_left_tie` | ⚠️ extend existing |
| ODL-19 | CPU/ROCm/host-CUDA byte-unchanged with `LGBM_CUDA_ON_DEVICE` unset; no f64 per-row loop (grep) | gate | `cargo test --workspace` + `grep` audit | ✅ (merge gate exists) |

### Sampling Rate
- **Per task commit:** `cargo test -p lgbm-compute --lib best_split` (fast cpu-fold unit tests)
- **Per wave merge:** `cargo test -p oracle-harness --test best_split_parity`
- **Phase gate:** `cargo test --workspace` green (esp. `learner_parity`, `kernel_parity`, `raw_bin_train_matches_cpp_golden` unregressed) + hip `rocm_backend_parity` within ~1e-6 before `/gsd-verify-work`.

### Wave 0 Gaps
- [ ] `crates/oracle-harness/tests/best_split_parity.rs` — the golden anchor harness (mirror `kernel_parity.rs` parse+assert shape) — covers ODL-11/ODL-12
- [ ] `crates/oracle-harness/tests/fixtures/kernels/best_split.txt` — the 6 golden categories (§D-07)
- [ ] `crates/lgbm-compute/src/kernels/best_split.rs` unit tests — count-recovery ties-even, epsilon placement, task-gen `assume_out_default_left` table
- [ ] Extend `crates/lgbm-compute/tests/rocm_backend_parity.rs` — tie-aware default_left on hip
- [ ] Confirm `InitCUDARandomKernel` seed formula before the USE_RAND golden (Open Q1)

## Security Domain

*(security_enforcement = true, ASVS L1 → section included.)*

### Applicable ASVS Categories

| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | no | No auth surface — internal compute kernel |
| V3 Session Management | no | No sessions |
| V4 Access Control | no | No access-control surface |
| V5 Input Validation | yes | Validate host-provided kernel scalars (num_bin, mfb_offset, task counts, buffer sizes) at the launch boundary BEFORE `launch_unchecked`, mirroring `primitives.rs::validate_scan_inputs` (rejects `block_size==0`, `num_blocks>1024`). Reject `num_tasks`/`num_leaves`/`num_bin` that would overflow the pre-allocated `DeviceSplitInfo` slabs. |
| V6 Cryptography | no | `CUDARandom` is a deterministic LCG for algorithmic reproducibility, NOT security RNG — do not treat as crypto |

### Known Threat Patterns for {CubeCL device kernels}

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| Out-of-bounds device write from a bad `num_bin`/`num_tasks`/`out_base` (unchecked launch) | Tampering / DoS | Validate all sizes at the host boundary (V5); `DeviceSplitInfo` slabs pre-sized to `num_leaf_slots`; the `launch_unchecked` SAFETY contract (per `primitives.rs` convention) requires the caller to have bounds-checked |
| Integer overflow in `task_index + num_tasks` / `leaf + block·num_leaves` addressing | Tampering | Use the C++ `data_size_t=i32` / `u32` widths faithfully; assert `2·num_tasks` and `num_leaves·num_blocks_per_leaf` fit the allocated buffer before launch |
| f64 hot loop sneaking into a device kernel (perf + the 5.4× consumer-NVIDIA regression) | DoS (perf) | Grep audit (ODL-19): f64 only in scalar gain/count math, never per-row |

## Sources

### Primary (HIGH confidence)
- `LightGBM/src/treelearner/cuda/cuda_best_split_finder.cu` (lines 16-341 numerical cores + reductions; 1920-2159 stages 2/3 + export) — VERIFIED by direct read
- `LightGBM/src/treelearner/cuda/cuda_best_split_finder.hpp` (SplitFindTask struct :28-41; assume_out_default_left :33) — VERIFIED
- `LightGBM/src/treelearner/cuda/cuda_best_split_finder.cpp` (:137-227 task-gen `assume_out_default_left` table) — VERIFIED
- `LightGBM/src/treelearner/cuda/cuda_leaf_splits.hpp` (:65-140 gain device helpers) — VERIFIED (the D-02 diff target)
- `crates/lgbm-compute/src/gain.rs`, `kernels/split.rs`, `kernels/split_info.rs`, `kernels/primitives.rs` — VERIFIED by direct read (the existing reuse surface)
- `docs/cuda-kernel-design.md` §8.1-8.3, §7, §17 — the port-source design reference
- `.claude/skills/spike-findings-lightgbm_rs` — spike-022 (default_left flips cosmetic ~1e-6), spike-052 (no f64 hot loops), def-f8u-01 (never GPU-vs-GPU)

### Secondary (MEDIUM confidence)
- `17-CONTEXT.md`, `16-CONTEXT.md`, `15-CONTEXT.md`, `14-CONTEXT.md` — locked decisions carried forward
- `crates/oracle-harness/tests/kernel_parity.rs` — the golden-fixture harness pattern to mirror

### Tertiary (LOW confidence)
- None — all claims traced to a primary source read this session.

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — no new packages; all reuse surfaces read directly
- Architecture (3-stage pipeline): HIGH — every kernel + reduction order read line-by-line in the C++ reference
- Gain-math diff (D-02): HIGH — verified bit-identical for USE_SMOOTHING=false; smoothing branch located exactly
- Count-recovery / epsilon / default_left landmines: HIGH — exact source lines cited
- RNG seed formula (Open Q1): MEDIUM — draw call confirmed, per-task seed formula deferred to planning read
- Golden generation method (A5): MEDIUM — histogram precedent suggests transcription; real-capture question flagged

**Research date:** 2026-07-01
**Valid until:** 2026-07-31 (stable — pinned to `LightGBM` 4.6 commit 195c26fc, read-only reference; cubecl 0.10 in tree)
