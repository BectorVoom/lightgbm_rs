# Research: Device-Resident Grow Loop — Deferring `pick` / `read_leaf` Syncs

> Goal under investigation: "Go after the device-resident grow loop — defer the
> `pick` / `read_leaf` synchronization points to attack the ~23% sync-bound
> tail." **Research only. No production code changed.** This document is
> evidence for a downstream SPEC/TDD planner; it deliberately does **not**
> prescribe an implementation design.
>
> Every material claim carries a provenance label:
> `[VERIFIED: CODEGRAPH …]` (returned verbatim by codegraph_explore),
> `[VERIFIED: LOCAL path:line]` (read/ran directly this session),
> `[PROJECT: doc]` (project/memory doc, not re-derived from primary source),
> `[INFERRED: …]`, `[UNVERIFIED: …]`, `[ASSUMED]`.

---

## 1. Context

- **What this is.** `lightgbm_rs` has an opt-in on-device (GPU-resident) tree
  grow loop, `grow_tree_on_device_resident`
  (`crates/lgbm-compute/src/kernels/grow_driver.rs:2329`), reached only when the
  backend advertises `resident_pool_supported()==true`. The default `CpuBackend`
  f64 anchor **never runs it** — it takes an older inline non-resident loop in
  the same driver function. `[VERIFIED: CODEGRAPH grow_tree_on_device_resident
  blast radius — 1 caller only, grow_driver.rs]` `[VERIFIED: LOCAL
  grow_driver.rs:2677-2848]`
- **Why it matters.** The project's non-negotiable contract is numerical parity
  with C++ LightGBM: the `cubecl-cpu` f64-fold path is the **bit-exact hard
  merge gate**, and ROCm/CUDA f32 is a **~1e-6 best-effort** gate. Any perf
  change here must not perturb either. `[PROJECT: CLAUDE.md]`
- **The hypothesis to test.** Per-node host↔device synchronizations (reading the
  picked split back to host; reading child partition ranges back to host each
  split) serialize the loop and cost ~23% of wall clock as a "sync-bound tail."
  The lever is to defer/batch/eliminate those readbacks while preserving parity.
- **Established working method in this repo** (recurring in the last ~30
  commits): add a perf change behind an **opt-in env flag**, validate on real
  hardware, then flip the default ON only after a measured win — e.g.
  `LGBM_DESC_HOIST` (commit `1752374`/`4a1bec4`), `LGBM_PARTITION_FUSE_BC_SMEM`
  (`82a1990`/`48dbdc2`), pargain scan (`ab9739a`, backend-aware default).
  `[VERIFIED: LOCAL git log --oneline]`
- **Working-tree state at research time.** Uncommitted: `split.rs`
  (parallel-prefix scan work), `scan_pargain_parity.rs`, `phase_prof.rs`; new
  untracked `rocm_drain_profile.rs`, `docs/cubecl_cubes_per_cu.md`, `vendor/`
  (cubecl-cuda fork), and a pre-existing full `research.md`/`SPEC.md`/`PLAN.md`
  in this plan dir. This diff does **not** touch `grow_driver.rs` or
  `partition.rs` (the two files this effort would modify) but does touch
  `split.rs`. `[VERIFIED: LOCAL git status]`

---

## 2. Grow-Loop Mechanics & Sync Points (file:line)

### 2.1 Entry / dispatch
- `grow_tree_on_device_resident<B,R>` at `grow_driver.rs:2329`. Reached from the
  driver `grow_tree_on_device_driver_with_cfg` (`grow_driver.rs:~1092`) only when
  `backend.resident_pool_supported() && no categorical features`; otherwise the
  older non-resident anchor loop runs. `[VERIFIED: CODEGRAPH grow_tree_on_device_resident,
  1 caller]` `[VERIFIED: LOCAL grow_driver.rs:2677-2848 context]`
- `CpuBackend` returns `false` for `resident_pool_supported()`; only
  `GpuBackend<R>` (rocm/cuda) overrides it → the resident loop is **dead code on
  the CPU merge gate**, live only under `--features rocm` (local gfx1152) or
  `--features cuda` (Kaggle). `[VERIFIED: CODEGRAPH DeviceLeafSplits/resident
  callers in lib.rs + grow_driver.rs]` `[PROJECT: memory/local-rocm-gpu.md]`

### 2.2 Per-split loop body — `for _split in 0..(num_leaves-1)` (`grow_driver.rs:2677`)
The loop is **best-first leaf-wise**: each iteration picks the single global-best
open leaf across the whole frontier, splits it, builds/subtracts/scans its two
children, and folds their gains back into the resident frontier. Splits are
therefore inherently serial (iteration `i+1`'s winner is unknown until `i`'s
children are scanned). `[VERIFIED: LOCAL grow_driver.rs:2677-2712 comments +
control flow]`

**Every blocking device→host sync in the per-split body:**

| Phase | Line(s) | Sync? | Counter | What crosses to host |
|---|---|---|---|---|
| **PICK** — `frontier_pick_best_leaf_device` (device argmax + self-invalidate + export) | `bump_sync()` at `grow_driver.rs:2701`; timed into `GROW_PICK_NS` at `:2704` | **YES** | `bump_sync` (`grow_driver.rs:126`) | `PickExport`: `cells:[i64;8]` + `winner:[f64;10]` (feat, thr, default_left, 4 child grad/hess sums, net gain, L/R output). `cells[6]`=best_leaf (`-1`=stop). `[VERIFIED: LOCAL grow_driver.rs:2700-2753]` |
| loop-break checks (`best_leaf<0`, `!(best.gain>0.0)`) | `:2714`, `:2754` | host-only | — | genuine host control flow (growth-stop decision) |
| **PARTITION** (resident-perm arm, default) — 2–3 launches then `read_leaf` | launches `:2808-2814`; `bump_sync()` at `:2845`; `read_leaf` at `:2846`; whole block timed into `GROW_PARTITION_NS` at `:2803` | **YES** | `bump_sync` | `ChildRanges` = 6×i32 (L/R start,end,count) via `DeviceLeafSplits::read_leaf` (`partition.rs:393`). `[VERIFIED: LOCAL grow_driver.rs:2802-2848]` |
| PARTITION (legacy arm, `resident_perm==None`) | `:2850` → `partition_resident_range` | conditionally (only the on-device sub-arm) | `bump_sync` | same `ChildRanges` shape via same `read_leaf` |
| smaller/larger role assignment (`smaller_is_left = left_count < right_count`) | host branch after read_leaf | host-only | — | decides which child gets fresh vs reused histogram slot (subtraction-trick invariant) `[VERIFIED: LOCAL grow_driver.rs:2846-2848 + existing-research §5]` |
| TREESPLIT (scheduled, no-readback) | later in body | no | — | tree-mutation kernel args are host scalars already known |
| BUILD smaller / SUBTRACT / SCAN → fold into frontier | later in body | **NO** | — | device→device via `*_into_frontier`; nothing crosses back (already-retired readback from prior perf work) |

### 2.3 Per-grow (once) syncs, outside the per-split table
- **Root scan** — 1 blocking sync inside `scan_resident_and_argmax`
  (`grow_driver.rs:~2072`), once per grow (the cross-feature argmax runs on
  device; only the winning ~8-int split crosses back). `[VERIFIED: LOCAL
  existing-research §4, cross-checked against on_device_sync_count.rs]`
- **Tail perm readback** — 1 blocking sync (`rp.read_perm`), resident-perm arm
  only, reads the whole permutation buffer once to rebuild
  `LeafPartitionLayout`. `[VERIFIED: LOCAL existing-research §4]`
- `tree.to_host_tree(client)` — a further per-grow device→host transfer of the
  flat tree struct (not counted by `bump_sync`). `[VERIFIED: LOCAL existing-research §4]`

### 2.4 Machine-checked sync closed form
`crates/lgbm-compute/tests/on_device_sync_count.rs` (rocm lane) asserts exactly:
```
resident-perm arm (DEFAULT):        syncs = 2 * num_leaves
   = [1 root scan] + [(L-1) pick] + [(L-1) read_leaf] + [1 tail perm]
legacy host-partition arm:          syncs = num_leaves
   = [1 root scan] + [(L-1) pick]         (read_leaf never fires)
cpu anchor (non-resident) baseline: syncs = 1 + 3*(num_leaves-1)
```
`[VERIFIED: LOCAL existing-research §4 citing on_device_sync_count.rs:182-326;
CODEGRAPH on_device_sync_count_take → tests on_device_sync_count.rs (both crates)]`.
So the **two per-split blocking syncs are exactly `pick` and `read_leaf`** — the
precise targets named in the goal. **Caveat for the planner:** the
`crates/oracle-harness/tests/on_device_sync_count.rs` copy documents an OLDER
closed form and is flagged stale in-repo; a "closed-form drift" hazard exists in
this area — any change must re-derive both files and assert exact counts.
`[VERIFIED: LOCAL existing-research §4 note; PROJECT: memory/MEMORY.md
resident-score-host-update-gotcha]`

### 2.5 The phase ledger (instrumentation you will reuse)
`GROW_*_NS` atomics (`grow_driver.rs:214-225`): `GROW_PICK_NS:221`,
`GROW_PARTITION_NS:222`, plus setup/upload/rootfold/build/subtract/scan/
treesplit/reduce/tail. Drained via `on_device_grow_phase_take()`
(`grow_driver.rs:247`) into `GrowPhaseNs`. `time_phase` (`:266`) is a
zero-overhead passthrough unless `LGBM_PHASE_PROF=="1"`. `on_device_sync_count_take()`
(`grow_driver.rs:136`) drains the blocking-readback count. All parity-neutral
(inert unless the gate is on). `[VERIFIED: CODEGRAPH GROW_PICK_NS/on_device_sync_count_take;
LOCAL grow_driver.rs:195-262]`

---

## 3. `read_leaf` / `pick` Deep-Dive

### 3.1 `read_leaf` — `DeviceLeafSplits::read_leaf` (`partition.rs:393`)
- **Struct.** `DeviceLeafSplits<R>` (`partition.rs:345`) owns ONE device i32
  buffer of `LEAF_SPLIT_STRIDE(=6) * num_leaves` cells; leaf `L` owns `[6L,6L+6)`
  (L/R start,end,count). Constructed once per grow (`new`, `:359`). The grow loop
  reads ranges **by handle on device** (`ranges_handle`, `:383`); only the host
  bookkeeping and test goldens call `read_leaf`. `[VERIFIED: CODEGRAPH
  DeviceLeafSplits struct + read_leaf source, partition.rs:342-408]`
- **Readback mechanism.** `read_leaf` calls `client.read_one_unchecked(...)`
  (`partition.rs:396`) — a **blocking** whole-buffer read — then slices out leaf
  `leaf_id`'s 6 ints into `ChildRanges`. `[VERIFIED: CODEGRAPH partition.rs:393-407]`
- **Why the host needs it, and who consumes it** (`grow_driver.rs:2846` +
  downstream):
  1. `left_count`/`right_count` → the two new leaves' `row_begin`/`row_count`
     (needed for the NEXT split's scannability gates — real host control flow
     deciding whether the next scan even launches — and as row-range VIEW offsets
     fed to the next BUILD kernel as launch params).
  2. `smaller_is_left = left_count < right_count` → decides which child gets the
     fresh histogram-pool slot vs the reused parent slot (subtraction-trick
     invariant). Must be resolved before BUILD/SUBTRACT kernel args are chosen.
  3. counts also feed tree metadata (`leaf_count_`/`internal_count_`).
  `[VERIFIED: LOCAL grow_driver.rs:2846-2848 + existing-research §5 detailed
  consumption trace]`
- **The underlying data is already fully resident** in `DeviceLeafSplits` before
  the host reads it; the sync exists only because (a) the host loop needs
  `smaller_is_left` + scannability booleans to decide what to launch next, and
  (b) `DeviceLeafSplits` entries are **overwritten** when a leaf id is split again
  (the left child reuses the parent's leaf id) — so the read must happen before
  the next overwrite. This overwrite-reuse is the concrete constraint that makes
  the read fire immediately, every split. `[VERIFIED: CODEGRAPH DeviceLeafSplits
  doc-comment partition.rs:342-345 "indices reused"; LOCAL existing-research §5]`
- **Callers.** CodeGraph reports `read_leaf` has **5 callers, all in
  `grow_driver.rs`** (resident-perm arm + `partition_resident_range` sub-arm).
  Covering tests: `crates/oracle-harness/tests/partition_parity.rs`,
  `crates/lgbm-compute/tests/resident_perm_partition.rs`. `[VERIFIED: CODEGRAPH
  read_leaf blast radius]`

### 3.2 `pick` — `frontier_pick_best_leaf_device` (`lib.rs:1532` trait; GPU impl in `best_split.rs`)
- **Kernel work (on device).** Cross-leaf argmax over the resident `SplitSoa`
  frontier records `[0, cur_num_leaves)` with the `split_gt` tie rule (strict `>`
  gain, then lowest real-feature index); self-invalidates the chosen + freshly
  created leaf slots; packs `PickExport`. `CpuBackend`'s default trait method
  returns `Err(Runtime{…})` — the anchor uses the host fold instead
  (`lib.rs:1532-1545`). `[VERIFIED: CODEGRAPH frontier_pick_best_leaf_device
  lib.rs:1532; existing-research §5 best_split.rs:2330-2354]`
- **What crosses back.** `PickExport = { cells:[i64;8], winner:[f64;10] }`.
  `cells[6]`=best_leaf (`-1`=stop), `cells[7]`=num_cat_threshold; `winner` = real
  feat, thr, default_left, 4 kEpsilon-carrying child sums, net gain, L/R output.
  `[VERIFIED: LOCAL grow_driver.rs:2713-2753 decode; existing-research §5
  PickExport best_split.rs:2296-2313]`
- **Host consumption.** Two genuine control-flow decisions (`best_leaf<0` and
  `!(best.gain>0.0)` → loop break) plus a batch of pure launch-parameter values
  (threshold/default_left/child sums/gain/outputs flow straight into the next
  iteration's kernels). The control-flow part is the reason a **synchronous
  readback is unavoidable once per split under a host-driven for-loop**.
  `[VERIFIED: LOCAL grow_driver.rs:2713-2756]`
- **`GROW_PICK_NS` role.** Wraps the `frontier_pick_best_leaf_device` call
  (`:2704`). In **free-run** mode this bucket also absorbs queued reduce/treesplit
  **device** time that drains at this blocking point; in `LGBM_GROW_DRAIN=1` mode
  device time is forced into its own bucket (see §4). `[VERIFIED: LOCAL
  grow_driver.rs:2702-2704 comment + :203-209 timing-semantics comment]`

### 3.3 Device-side infrastructure already present (do-not-reinvent)
- `DeviceFrontier<R>` (`lib.rs:~495`) — resident per-leaf `SplitSoa` records +
  device best_leaf/stop slots; `frontier_reduce_leaf` (device→device, no
  readback) + `frontier_pick_best_leaf` (the one readback). `[VERIFIED: CODEGRAPH
  DeviceFrontier / ResidentSplitWinner lib.rs:490-625]`
- `DeviceLeafSplits<R>` — resident child-range buffer, but **per-leaf-id and
  overwritten** on leaf reuse (§3.1). `[VERIFIED: CODEGRAPH partition.rs:345]`
- **Gap:** `ResidentDriverLeaf` bookkeeping (`row_begin`, `row_count`, `sum_g`,
  `sum_h`, `slot`, `best`, `best_fpos`, `depth`; `grow_driver.rs:1860`) lives
  **entirely host-side** as `leaves: Vec<ResidentDriverLeaf>`. Only the
  split-decision (`SplitSoa`) and child-range (`DeviceLeafSplits`) pieces are
  device-resident today. This is the boundary between "defer pick/read_leaf"
  (bounded) and "fully device-resident loop" (large). `[VERIFIED: CODEGRAPH
  ResidentDriverLeaf grow_driver.rs:1860 — 1 caller; LOCAL grow_driver.rs:1857-1882]`

---

## 4. The "~23% Sync-Bound Tail" — Measurement Provenance & Reproduction

**Bottom line up front:** the 23% figure is **measured, not assumed**, but it is
(a) a **local 8-CU APU** drain-ledger number the same memory note repeatedly
warns is **NOT P100/perf-representative**, and (b) a per-phase bucket that
**bundles genuine device-compute time with the readback/sync time** — so 23% is
an **upper bound** on what deferring syncs can recover, not a pure sync cost. The
planner must not treat "23% recoverable by removing syncs" as established.

### 4.1 Where the number comes from
`[PROJECT: memory/local-rocm-gpu.md:72-82]` — a real-hardware drain profile on
the local gfx1152 (8-CU APU), 100k×50, parprefix-default-ON configuration:
- Line 72 (post-parprefix, per-tree): *"Remaining buckets (pick ~8ms/13%,
  partition ~6ms/10%) are sync/readback-bound — only the device-resident grow
  loop (defer pick-export + read_leaf syncs) would move them, a large redesign
  flagged out-of-scope."* → **13% + 10% = ~23%**, and this is verbatim the
  motivation for this phase. `[VERIFIED: LOCAL memory/local-rocm-gpu.md:72-74]`
- Lines 76-82 (a separate 100-tree profile): scan 2545ms(41%), build 1310,
  partition 862, pick 790, treesplit 328, reduce 66, subtract 17, tail 85,
  setup 76, upload 48. `[VERIFIED: LOCAL memory/local-rocm-gpu.md:76-82]`

### 4.2 What actually gets measured (and the two important caveats)
- **The buckets include device compute, not just sync.** `GROW_PICK_NS` wraps the
  whole `frontier_pick_best_leaf_device` call (device argmax kernel **+** the
  8-int readback); `GROW_PARTITION_NS` wraps the 2–3 mark/scan/scatter launches
  **+** the 6-int `read_leaf`. So "pick 13%" = pick argmax device time + readback;
  "partition 10%" = partition kernel device time + readback. The
  deferrable-by-syncs portion is only the **readback fraction** of each.
  `[VERIFIED: LOCAL grow_driver.rs:2702-2704 (GROW_PICK_NS wraps kernel+sync),
  2803-2846 (GROW_PARTITION_NS wraps launches+read_leaf)]`
- **Drain vs free-run semantics.** In free-run the async build/subtract/treesplit/
  reduce buckets hold only host **submission** time and their device time drains
  inside the next blocking bucket (scan/pick/partition). `LGBM_GROW_DRAIN=1`
  blocks the queue empty inside each phase's own timer so device time lands in
  its own bucket — "drain numbers rank phases; free-run numbers price the
  [wall]." Thus the 13%/10% attribution is a **drain-mode ranking**, and removing
  a sync can shrink a drain bucket without proportionally shrinking free-run
  **wall** if the sync was already overlapping queued work. `[VERIFIED: LOCAL
  grow_driver.rs:203-209 timing-semantics comment; 308-313 grow_drain]`
- **APU ≠ P100.** `[PROJECT: memory/local-rocm-gpu.md:38-59]` — "8-CU APU ≠
  56-SM P100; occupancy characteristics differ wildly … Perf verdicts still
  require a Kaggle P100 run." The note is emphatic that APU walls are
  "opposite-sign to P100 for scan/arithmetic parallelism." Host-enqueue/readback
  cost (~µs/launch) is roughly device-size-independent, so a local DRAIN ledger
  still shows enqueue-vs-device STRUCTURE — which is why the sync-tail is at least
  visible locally — but the 23% magnitude's transfer to P100 is unproven.
  `[VERIFIED: LOCAL memory/local-rocm-gpu.md:38-43, 49-59]`

### 4.3 How to reproduce the measurement
`crates/lgbm-treelearner/examples/rocm_drain_profile.rs` (untracked, new) is the
exact tool that produced the percentages. It trains `N_TREES` warm reps through
the resident path and dumps the per-tree-averaged `GrowPhaseNs` ledger with each
bucket's `%` of wall. `[VERIFIED: LOCAL rocm_drain_profile.rs:1-172]`
```
export ROCM_PATH=/home/user/rocm/opt/rocm-7.1.1 HIP_PATH=$ROCM_PATH ROCM_HOME=$ROCM_PATH
export LD_LIBRARY_PATH=$ROCM_PATH/lib:$LD_LIBRARY_PATH PATH=$ROCM_PATH/bin:$PATH
LGBM_CUDA_ON_DEVICE=1 LGBM_PHASE_PROF=1 LGBM_GROW_DRAIN=1 \
  cargo run --release -p lgbm-treelearner --features rocm --example rocm_drain_profile
```
Knobs: `LGBM_PROF_NDATA`(100000), `LGBM_PROF_NFEAT`(50), `LGBM_PROF_NTREES`(100),
`LGBM_PROF_WARMUP`(1), `LGBM_PROF_NBIN`(255). Default shape is `num_leaves=31`,
100k×50. `[VERIFIED: LOCAL rocm_drain_profile.rs:11-35,106-116]`
**To isolate the sync fraction from device compute**, the planner should re-run
both **free-run** (no `LGBM_GROW_DRAIN`) and **drain** modes and diff — the
existing research (§10 Rank 3) makes the same recommendation. `[VERIFIED: LOCAL
existing-research §10]`

### 4.4 Related profiling context (untracked docs)
- `docs/cubecl_cubes_per_cu.md` — the row-partition/cubes-per-CU occupancy model
  (`CUBES_PER_CU=8`, `T_cubes=N_CU*8`, `P=clamp(T/features,1,16)`), env knobs
  `LGBM_ROWPART_TARGET_CUBES`/`LGBM_ROWPART_MIN`. Relevant only as background: it
  governs the **build** bucket's occupancy, not pick/partition; the memory note
  says the build lever is **saturated** on the 8-CU APU. `[VERIFIED: LOCAL
  docs/cubecl_cubes_per_cu.md:24-91; memory/local-rocm-gpu.md:61-74]`
- `phase_prof.rs` (`crates/lgbm-treelearner/src/phase_prof.rs`, modified) — the
  `dump` that prints the `COUNTS` line (blocking_readbacks(syncs)=, launch
  counts, the various `*_CNT` tripwires including the new `scan_parprefix`) and
  folds `on_device_grow_phase_take()`/`on_device_sync_count_take()`. This is the
  aggregation surface for any new tripwire. `[VERIFIED: LOCAL git diff phase_prof.rs
  (+scan_parprefix); grow_driver.rs:132-137 doc-comment referencing phase_prof::dump]`

### 4.5 Verdict on the premise (honest)
- The **existence** of a per-split readback tail is **VERIFIED** from source:
  exactly two blocking syncs per split (`pick` + `read_leaf`), directly at
  `grow_driver.rs:2701` and `:2845`. `[VERIFIED: LOCAL]`
- The **~23% magnitude** is **VERIFIED as a measured APU drain number**
  `[PROJECT: memory/local-rocm-gpu.md:72]`, but its P100 transfer is
  **UNVERIFIED**, and the buckets **conflate device compute with sync**, so the
  recoverable share is strictly less than 23%. The planner should treat 23% as
  the phase's *motivating* figure, not a *recoverable* one, and re-measure
  free-run-vs-drain before/after any change. `[INFERRED from §4.2 + §4.3]`

---

## 5. Parity Constraints & the Opt-In-Flag Pattern

### 5.1 Parity constraints (locked)
- **CPU f64-fold anchor is the bit-exact hard merge gate; ROCm/CUDA f32 is
  ~1e-6 best-effort.** `[PROJECT: CLAUDE.md]`
- `grow_tree_on_device_resident` **never runs on the CPU anchor** (§2.1), so no
  sub-approach here can touch the merge gate's own arithmetic directly. What it
  *is* held to: the STRUCTURE-parity and envelope tests that pin the resident arm
  against the anchor (§6). `[VERIFIED: CODEGRAPH resident_pool_supported dispatch;
  PROJECT: existing-research §7]`
- **Deferring/fusing pick+read_leaf is a pure host/device-orchestration change:**
  no f64/u64 value is recomputed or reordered — only *when* the host reads it
  back changes. This carries ~zero numerics risk; the only new risk is
  orchestration correctness (stale-buffer reads, launch-geometry parity).
  `[INFERRED from §3; consistent with existing-research §7]`
- **Leaf-wise (not level-wise) growth is required for LightGBM parity** — a
  literal "batch N future splits" cannot be done without switching growth policy,
  which changes tree structure and breaks the 100%-behavioral-compat constraint.
  The achievable "batching" is fusing `read_leaf(i)`+`pick(i+1)` into one
  readback. `[VERIFIED: LOCAL grow_driver.rs:2677-2712 best-first loop; PROJECT:
  CLAUDE.md; existing-research §9]`

### 5.2 The opt-in-flag pattern (follow this exactly)
Established, uniform across recent perf work. Two co-existing mechanisms:

1. **`OnceLock`-cached env read** for the process-wide default:
   ```rust
   static E: OnceLock<bool> = OnceLock::new();
   *E.get_or_init(|| std::env::var("LGBM_PARTITION_FUSE_BC_SMEM").map(|v| v != "0").unwrap_or(true))
   ```
   `[VERIFIED: CODEGRAPH partition_fuse_bc_smem_enabled partition.rs:1269-1279]`
2. **`AtomicU8` same-session A/B override** (0=unset,1=force ON,2=force OFF) read
   *before* the OnceLock, for in-process arm switching in tests/benches.
   `[VERIFIED: CODEGRAPH partition.rs:1270-1282 PARTITION_FUSE_BC_SMEM_OVERRIDE]`
3. **Naming.** Env flags are `LGBM_<AREA>_<KNOB>` (`LGBM_DESC_HOIST`,
   `LGBM_PARTITION_FUSE_BC_SMEM`, `LGBM_PARTITION_RESIDENT`, `LGBM_SCAN_PARGAIN`,
   `LGBM_SCAN_STAGED`, `LGBM_SUBTRACT_FUSE`, `LGBM_ONDEVICE_F64_FUSED`,
   `LGBM_ONDEVICE_BIN_HOIST`, `LGBM_REDUCE_BATCH`, `LGBM_GROW_DRAIN`).
   `[VERIFIED: LOCAL git log --oneline; grow_driver.rs:305-320,417-429]`
4. **Backend-aware defaults.** A flag can default ON for one runtime and OFF for
   another — e.g. `scan_pargain_enabled(runtime_name)` at `split.rs:2238` returns
   `runtime_name == "hip"` by default (default ON for ROCm/AMD, OFF for
   `"cuda"`/cpu), gated via `<R as cubecl::Runtime>::name(client)`. The
   uncommitted parprefix flag does the same (default ON for `"hip"`).
   `[VERIFIED: LOCAL split.rs:2215-2243, 2281-2290; git commit ab9739a]`
5. **Real-device gating for kernels that "won't lower on cpu":** guard with
   `<R as cubecl::Runtime>::name(client) != "cpu"` (e.g. the SMEM BC-fusion
   tripwire at `grow_driver.rs:2819-2822`). `[VERIFIED: LOCAL grow_driver.rs:2819-2822]`

**Default policy (from commit history):** ship OFF (opt-in) → validate on real
hardware → flip default ON in a follow-up commit that cites the measured speedup
and the spike number (e.g. "default LGBM_DESC_HOIST to ON — validated 1.055x on
P100 (spike101)"). Perf verdicts that flip a P100-relevant default require a
**Kaggle P100** run; ROCm-specific defaults are flipped on **real gfx1152**.
`[VERIFIED: LOCAL git log --oneline commits 1752374, 82a1990, ab9739a,
355b9e2, 2d6267f]`

---

## 6. Test / Validation Surface & How Perf Is Measured

### 6.1 Tests that cover the resident grow loop / partition parity
- `crates/oracle-harness/tests/learner_parity.rs` —
  `learner_parity_on_device_resident_fast_path_gate` (rocm-gated): pins the
  resident fast-path grown-tree **structure** bit-exact to the cpu f64 anchor,
  across both partition routes and both copack states. **Load-bearing.**
  `[VERIFIED: LOCAL existing-research §7 citing learner_parity.rs:3053-3111]`
- `crates/lgbm-compute/tests/cuda_on_device.rs` —
  `resident_tree_bit_exact_to_u64_integer_path` (runs on the CPU structure-anchor
  arm, u64 integer reference) and `resident_score_within_envelope_of_host_cuda`
  (`#[cfg(feature="rocm")]`, ATOL=1e-5/RTOL=1e-6 envelope, "NEVER bit-exact").
  `[VERIFIED: LOCAL existing-research §7 citing cuda_on_device.rs:261,374]`
- `crates/oracle-harness/tests/partition_parity.rs` &
  `crates/lgbm-compute/tests/resident_perm_partition.rs` — cover `read_leaf`,
  `partition_leaf`, `partition_child_ranges_device`; the latter has
  `partition_bc_fusion_byte_identical_to_three_launch` (the byte-identity
  pattern for launch-geometry changes). `[VERIFIED: CODEGRAPH read_leaf/partition_leaf
  blast radius; existing-research §7]`
- `crates/lgbm-compute/tests/on_device_sync_count.rs` (authoritative) &
  `crates/oracle-harness/tests/on_device_sync_count.rs` (older/stale) — the exact
  closed-form sync counters (§2.4). Any sync-pattern change must update both and
  assert **exact** counts. `[VERIFIED: CODEGRAPH on_device_sync_count_take tests]`
- **Note:** CodeGraph reports `grow_tree_on_device_resident`, `ResidentDriverLeaf`,
  `DeviceLeafSplits`, `GROW_PICK_NS` all as "⚠️ no covering tests found" —
  i.e. **no direct unit test**; they are exercised only **transitively** via the
  learner/oracle rocm-gated tests on real hardware. `[VERIFIED: CODEGRAPH blast
  radius flags]`

### 6.2 How to run
```bash
# real-hardware (functional + numerics), gfx1152:
export ROCM_PATH=/home/user/rocm/opt/rocm-7.1.1 HIP_PATH=$ROCM_PATH ROCM_HOME=$ROCM_PATH
export LD_LIBRARY_PATH=$ROCM_PATH/lib:$LD_LIBRARY_PATH PATH=$ROCM_PATH/bin:$PATH
cargo test -p lgbm-compute   --features rocm
cargo test -p oracle-harness --features rocm
# default merge gate (cpu anchor — does NOT exercise the resident loop):
cargo test --workspace
```
Feature: `rocm = ["cubecl/hip","dep:cubecl-hip-sys","gpu"]`. `[VERIFIED: LOCAL
memory/local-rocm-gpu.md:24-29; existing-research §17]`

### 6.3 How perf is *actually* measured/validated in this repo
- **Local gfx1152 (8-CU APU): correctness + numerics ONLY, not perf.** It
  saturates at ~8 CUs and is "opposite-sign to P100" for scan/arithmetic
  parallelism. Use it to confirm a kernel stays within the ~1e-6 contract and to
  read enqueue-vs-device STRUCTURE (drain ledger), never to judge a perf verdict.
  `[PROJECT: memory/local-rocm-gpu.md:38-59]`
- **Kaggle P100 (56 SM): the perf verdict authority.** Every "default ON" commit
  cites a P100 speedup + spike number. Workflow: develop → validate
  correctness/numerics locally on rocm → one Kaggle P100 run for the perf
  verdict. `[PROJECT: memory/local-rocm-gpu.md:45-47, kaggle-bench-workflow.md;
  VERIFIED: LOCAL git log commit subjects]`
- **A/B discipline:** alternate arm order, warm-median of ≥3, and confirm via the
  `COUNTS`/tripwire ledger that the code-under-test actually ran (a bucket
  shrinking is not a wall win). `[PROJECT: memory/ondevice-perf-campaign.md;
  existing-research §16]`

---

## 7. Prior Decisions & Contradictions

### 7.1 Established pattern (opt-in → real-device validate → default ON)
Consistent across `LGBM_SCAN_STAGED` (`2d6267f`/`3cb97ff`), `LGBM_PARTITION_RESIDENT`
(`355b9e2`/`1b29e4c`, 1.25x P100), `LGBM_SUBTRACT_FUSE` (`a76e72e`),
`LGBM_DESC_HOIST` (`1752374`), `LGBM_PARTITION_FUSE_BC_SMEM` (`82a1990`), pargain
scan (`ab9739a`, backend-aware). Some spikes concluded **stays opt-in / net-negative
on P100** (`f96c389`/`8b23780` pargain; `130b074` partition BC-fusion) — i.e. the
process routinely *rejects* changes that lose on P100 even when they win locally.
`[VERIFIED: LOCAL git log --oneline]`

### 7.2 The contradiction the planner MUST weigh
- **CUDA-graph campaign conclusion:** capture/replay was PROVEN bit-identical but
  the realistic-chain magnitude was only ~1.04×; the **residual gap is
  device-compute-bound, NOT launch/enqueue-bound** on P100 (per-launch enqueue is
  ~4–11µs, not the campaign's hypothesized 91µs). CUDA graphs judged **not worth
  it**; the pivot recommendation was to the **scan/build compute lever, esp.
  ROCm**. `[PROJECT: memory/cudagraph-campaign.md via MEMORY.md index]`
- **This phase's premise:** the sync/readback tail (pick+read_leaf) is ~23% and
  worth attacking.
- **Reconciliation / honest tension:**
  - The CUDA-graph finding is about **enqueue amortization** (many small launches'
    host-submission cost), which is a **different** lever from a **blocking
    readback that stalls the queue**. So it does not directly refute the sync-tail
    premise. `[INFERRED; consistent with existing-research §1/§8]`
  - BUT it does establish that on **P100** the dominant residual is **device
    compute**, and the 23% sync-tail is measured on the **APU**, where the memory
    note says perf is "opposite-sign." The scan bucket (41% on the APU) is the
    biggest *device-compute* lever and was already pursued (pargain/parprefix,
    ROCm-default-ON). `[PROJECT: memory/local-rocm-gpu.md:76-82,cudagraph-campaign.md]`
  - **Net:** there is a real per-split readback tail (source-verified), but the
    evidence that removing it yields a proportional **P100 wall** win is **weak** —
    the strongest available data points (CUDA-graph P100 residual = device-bound;
    APU drain buckets = compute+sync bundled) both caution that the recoverable
    fraction is likely smaller than 23% and may not transfer to P100. The planner
    should require a **free-run-vs-drain A/B on real hardware** before committing,
    and consider scoping the phase **ROCm-first** (mirroring how pargain/parprefix
    ended up backend-aware). `[INFERRED from §4.2, §7.2; VERIFIED provenance above]`

### 7.3 What the C++ reference does (bounds the design space)
Per `docs/cuda-kernel-design.md` (the `LightGBM/` C++ tree is **absent** from this
sandbox, so these are `[PROJECT: docs/cuda-kernel-design.md]`, not re-verified
against source): the reference is **not** a zero-sync loop — "only a handful of
scalars cross back per iteration." It does **two per-split host crossings** (an
8-int pick copy-back and a 16-int split-info copy-back), and it **resolves
smaller/larger role assignment ON DEVICE** (`SplitTreeStructureKernel`) using
`CopyFromCUDADeviceToHostAsync` + multiple streams. cubecl 0.10 has **no
multi-stream overlap** (`supports_multi_stream_overlap()==false`) and **no
persistent-kernel/indirect-dispatch primitive** (`CubeCount` is host-specified);
it does support **batched multi-handle async read** (`client.read(Vec<Handle>)`,
`supports_async_device_copy()==true`). So a **fully device-driven zero-sync loop
is out of reach without a vendor fork**, but on-device role assignment + one
batched readback is expressible today. `[PROJECT: docs/cuda-kernel-design.md;
memory/cudagraph-campaign.md; VERIFIED: LOCAL existing-research §6/§8 citing
grow_driver.rs:3309-3358]`

---

## 8. Open Questions / Unknowns for the Planner

1. **Scope of "device-resident grow loop."** Bounded (defer/fuse pick+read_leaf,
   on-device role assignment — buildable on cubecl 0.10 today) vs the full loop
   (move all `ResidentDriverLeaf` bookkeeping on-device — needs new kernels for
   every per-split host decision, likely a vendor-fork-scale effort). These have
   very different size/risk. **Needs explicit user/planner decision before SPEC.**
   `[VERIFIED: §3.3 gap; §7.3 cubecl bound]`
2. **Is the 23% actually recoverable, and on which target?** The figure is an
   APU drain bucket that bundles device compute + sync; the deferrable share is
   the readback fraction only, and P100 transfer is unproven (CUDA-graph residual
   was device-bound). **Requires a free-run-vs-drain A/B on real hardware to
   split the sync fraction from device compute BEFORE committing.** Consider
   scoping ROCm-first / backend-aware. `[VERIFIED: §4.2, §7.2]`
3. **`DeviceLeafSplits` overwrite constraint.** Deferring `read_leaf` past the
   point a leaf id is reused would read stale data (§3.1). A deferral needs either
   non-overwriting per-split storage or a generation/consumed guard — a genuine
   data-layout design choice, not resolved here. `[VERIFIED: §3.1 overwrite doc]`
4. **The two loop-continuation host decisions cannot be removed under a
   host-driven for-loop.** `best_leaf<0` and `!(gain>0.0)` (pick) and the
   scannability gates (read_leaf) are genuine host control flow; a synchronous
   readback of *something* per split is unavoidable unless the loop itself moves
   on-device (Q1's full-scope option). So even a "defer" approach likely reduces
   `2*L` syncs toward `~L`, not to zero. `[VERIFIED: §3.1, §3.2]`
5. **Uncommitted `split.rs` (parprefix) diff.** Orthogonal to `grow_driver.rs`/
   `partition.rs` but touches `split.rs`; land or isolate it first per AGENTS.md's
   "confirm dependencies first / include dependency info in commits" rule to avoid
   two large diffs colliding. `[VERIFIED: LOCAL git status; PROJECT: AGENTS.md]`
6. **Closed-form sync-count drift.** Two `on_device_sync_count.rs` files, one
   already flagged stale; any sync change must re-derive both and assert exact
   counts. `[VERIFIED: §2.4]`
7. **No direct unit coverage of the resident loop.** It is only transitively
   tested on real GPU; a regression can pass the default `cargo test` merge gate
   undetected. `--features rocm` validation is mandatory, not optional. `[VERIFIED:
   CODEGRAPH "no covering tests found" flags]`

---

## 9. Sources

- **CodeGraph** (primary, `.codegraph/` present): `grow_tree_on_device_resident
  ResidentDriverLeaf DeviceLeafSplits GROW_PICK_NS on_device_sync_count_take`;
  `read_leaf DeviceLeafSplits partition pick best split device resident` — blast
  radii + verbatim source for `DeviceLeafSplits`(partition.rs:345),
  `read_leaf`(:393), `GROW_PICK_NS`(:221), `on_device_sync_count_take`(:136),
  `ResidentDriverLeaf`(:1860), `frontier_pick_best_leaf_device`(lib.rs:1532),
  `scan_resident_leaf_argmax`(lib.rs:3875), `partition_leaf`/`partition_leaf_stable`.
- **Local file reads (this session):** `grow_driver.rs:120-315,2677-2856`;
  `partition.rs:342-408,1266-1282` (via CodeGraph verbatim);
  `rocm_drain_profile.rs:1-177`; `docs/cubecl_cubes_per_cu.md:1-91`;
  `split.rs` grep (scan_pargain_enabled:2238); pre-existing
  `.planning/plans/device-resident-grow-loop/research.md:1-1008` (mined for
  provenance + cross-check, independently re-verified against source).
- **Memory docs:** `memory/local-rocm-gpu.md` (full — 23% provenance lines 72-82,
  APU≠P100 caveats), `memory/MEMORY.md` (index — cudagraph-campaign,
  ondevice-perf-campaign, resident-score-host-update-gotcha).
- **Command output:** `git log --oneline -30`, `git status`,
  `find .planning planning -type f`, `grep -rn "23%"`, `grep sync-bound`,
  `grep scan_pargain split.rs`.
- **Project docs:** `CLAUDE.md`, `AGENTS.md`, `docs/cuda-kernel-design.md`
  (via existing-research citations; `LightGBM/` C++ source absent from sandbox).
- **Web / Context7:** none this session (cubecl-0.10 capability facts sourced
  from prior-session Context7 pulls recorded in the pre-existing research §8 and
  cross-checked against `memory/cudagraph-campaign.md`; not independently
  re-pulled — see Confidence).

---

## 10. Confidence Assessment

**HIGH** (directly verified from source / CodeGraph / command output this session):
- The exact per-split sync inventory and line numbers: pick `bump_sync` at
  `grow_driver.rs:2701` (timed `GROW_PICK_NS`:2704), partition `read_leaf` sync at
  `:2845-2846` (timed `GROW_PARTITION_NS`:2803). The two per-split blocking syncs
  are exactly `pick` and `read_leaf`.
- What `pick`(`PickExport`) and `read_leaf`(`ChildRanges`) carry and how the host
  consumes each field; `read_leaf` uses a blocking `read_one_unchecked`.
- The resident loop never runs on the CPU merge gate; it is real-GPU-only and has
  no direct unit coverage.
- `GROW_PICK_NS`/`GROW_PARTITION_NS` wrap **kernel + sync**, so the 13%/10%
  buckets bundle device compute with readback.
- The opt-in-flag pattern (OnceLock + AtomicU8 override, backend-aware
  `scan_pargain_enabled` default-ON-for-hip, `!= "cpu"` gating) and the
  reproduction command for `rocm_drain_profile.rs`.
- The 23% figure's provenance: `memory/local-rocm-gpu.md:72` (pick 13% + partition
  10%, measured on real gfx1152 drain profile).

**MEDIUM** (multiple consistent sources; not re-exercised locally this session):
- The machine-checked sync closed forms (`2*num_leaves` etc.) — read from the
  pre-existing research's citation of `on_device_sync_count.rs`, consistent with
  the source control flow I verified, but I did not re-open that test file.
- The C++ reference behavior (two per-split crossings, on-device role assignment)
  — `[PROJECT: docs/cuda-kernel-design.md]`, not re-verified against absent C++.
- cubecl 0.10 capability bounds (no multi-stream overlap / persistent kernel;
  batched `client.read` supported) — from prior-session Context7 + repo tests, not
  re-pulled this session.

**LOW** (needs validation before planning commits):
- **Whether the ~23% is materially recoverable, and whether any win transfers to
  P100.** The magnitude is APU-measured, bundles compute+sync, and the strongest
  P100 evidence (CUDA-graph residual = device-compute-bound) cautions against a
  proportional wall win. This is the single biggest premise risk and requires a
  real-hardware free-run-vs-drain A/B before the phase is scoped/committed.
- The best `DeviceLeafSplits` deferral layout (per-split buffer vs generation
  guard) — not spiked.
- Whether a Kaggle P100 verdict is in-scope, given the phase is framed around an
  ROCm-only figure.
