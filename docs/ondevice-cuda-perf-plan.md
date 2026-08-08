# On-Device CUDA Learning-Speed: Root Cause & Improvement Plan

Date: 2026-07-15. Baseline hardware: Kaggle P100 (bench GPU), corpus 500k×50 binary,
100 trees, `num_leaves=31`, warm-median-of-3, order-alternated arms
(`crates/lgbm-treelearner/examples/phase31_ab.rs` protocol).

## 1. Where we are

| Arm | Wall (P100, 500k×50×100) |
|---|---|
| rust on-device (`LGBM_CUDA_ON_DEVICE`), campaign start | ~14.0 s |
| rust on-device, after rounds 1–11 (current master) | **~7.1 s** |
| rust host_cuda | ~7.9 s |
| official LightGBM CUDA 4.6 | **~3.4 s** |

Remaining gap ≈ **2.1×**. The 2026-07 campaign (rounds 1–11 + CUDA-graph and
sync-deferral phases) already fixed everything launch/upload/sync-structural that
was cheap to fix; the numbers below are its drained (device-time-true) ledger.

### 1.1 Current cost decomposition (drained, per 100-tree train, P100)

| Bucket | Cost | Nature |
|---|---|---|
| scan (split finding) | ~1.8 s | device compute — **biggest bucket** |
| build (histograms) | ~1.05–1.25 s | device compute |
| binning | ~0.9 s | host, per-train, serial |
| partition | ~0.47–0.56 s | device compute |
| grad | ~0.4 s | host+device residual |
| setup+upload+tail | ~0.85 s | host alloc/enqueue |
| pick + treesplit + reduce + subtract | ~0.8 s | device + 2 blocking syncs/split |

### 1.2 What is already fixed (do not re-derive)

Rounds 1–11 shipped: fused score scatter, grad residency, resident-perm partition
(1.25×), staged scan (1.05×), reduce-batching (1.035×), subtract-into-scan fusion
(1.043×), SM-count row-partition occupancy fix P=1→8 (1.061×), descriptor-upload
hoist (1.055×), SMEM partition BC-fusion (1.011×). Refuted with measurements (do
not retry as-is): pargain / parprefix intra-cube arithmetic parallelism on CUDA
(spike094/104 — both net-negative on P100; they ARE wins on ROCm and default-on
there), CUDA graphs (1.04× — chain is device-bound, enqueue is only ~4–11 µs/launch
on the current image), sync-deferral `LGBM_GROW_DEFER_SYNC` (0.836× — the deferral's
required compute changes cost more than the 2L→L+2 sync halving recovers),
partition BC-fusion without SMEM (0.978×).

## 2. Root cause of the remaining 2.1×

Three layers, in order of size:

### RC-1 Device compute in build+scan (~3.0 s) — kernel-structural vs official

**Build** (`construct_leaf_hist_resident_lds_kernel_u64`,
`crates/lgbm-compute/src/kernels/histogram.rs:683`): one cube per **feature** ×
P row-partitions; each cube gathers `leaf_rows[k]` (random), reads the 1-byte bin
(native-width resident store — already u8), and reads `ord_g[row]`/`ord_h[row]`.
Structural deficits vs official (`CUDAConstructHistogramDenseKernel`, design doc
`docs/cuda-kernel-design.md` §7):

1. **Grad/hess gather amplification.** Every feature-cube re-gathers the same
   row's g/h ⇒ ~`num_features × 8 B` = ~400 B of random-access f32 reads per row
   vs official's ~8 B/row (row-major store, `threadIdx.x` = column, the g/h read
   is one per row per block, warp-broadcast across columns). ≈ 7–9× effective
   memory traffic on the build, and it is random-gather instead of coalesced.
2. **Column-major bin gather.** Our resident bins are feature-major
   (`ResidentBins`, `lib.rs`), so child-leaf builds gather scattered rows within
   each column. Official builds from a **row-major** packed store
   (`CUDARowData`, we have the analog `kernels/row_data.rs` — currently only a
   prediction-side store) so consecutive threads read consecutive bytes.
3. **No most-frequent-bin skip.** Official's dense scatter skips the mfb and
   reconstructs it as `leaf_total − Σothers` (`FixHistogramKernel`); our kernel
   scatters every row (`histogram.rs:1660` comment documents this). On the
   random-uniform bench corpus mfb ≈ 1/255 of rows (negligible), but on real
   skewed/sparse data this is a large official advantage — worth having, low
   priority for closing the *benchmark* gap.
4. u64 fixed-point pairs (16 B/bin-pair, 2 integer atomics/row/feature) are the
   price of order-free bit-exactness — keep; official pays ~the same with
   f64 hist (or less in its quantized mode, which is off by default).

**Scan** (staged sibling kernels in `kernels/split.rs`): 100 cubes
(50 features × 2 siblings), CUBE_DIM 64, LDS stage then 2 active lanes walk
REV/FWD serially. Measured device time ≈ **350–600 µs/split**, but the
arithmetic content is only ~10–40 µs/split (255 bins × ~60 f64 ops × 2 lanes).
The 10–20× residue is **unexplained by any tested hypothesis** — arithmetic
parallelization was refuted twice (spike094/104), so it is NOT lane starvation
on the walk. Untested candidates: cubecl per-launch device-side info-buffer
memcpy on sm_60 (P100 lacks `grid_constants`), fixed-size LDS allocation
limiting occupancy, cubecl codegen overhead (bounds checks, f64 div lowering),
only-one-wave latency exposure. **No kernel-level profiler evidence exists for
either implementation — every campaign number is a wall-clock ledger.** This is
the single biggest evidence gap.

**No stream overlap.** Official runs smaller-leaf split finding on stream 0 and
larger-leaf on stream 1 and overlaps histogram fix/subtract; cubecl 0.10 gives
us one serial submission queue per device thread (upstream limitation).

### RC-2 Host per-train work (~1.9 s)

- **binning ~0.9 s**: per-train host bin-mapper + column construction, serial
  (phase seam in `crates/lgbm-treelearner/src/phase_prof.rs`). Official does the
  equivalent inside `Dataset` construction with OpenMP across features — its
  3.4 s includes it, so this ~0.9 s is pure deficit.
- **setup+upload+tail ~0.85 s**: per-grow pool allocs (perm iota, frontier,
  slot zeroing), the 2 MB tail perm readback ×100 trees.
- **grad ~0.4 s** residual after residency fix.

### RC-3 Structural residue (~0.3–0.6 s, mostly blocked upstream)

- cubecl-cuda dispatch ~4–11 µs/launch × ~30k real launches ≈ 120–330 ms
  (raw CUDA ≈ 3–5 µs). CUDA graphs proven mechanically but only 1.04×.
- 2 blocking syncs/split (pick export + read_leaf child ranges) — both feed the
  next op; the deferral rewrite was measured net-negative (T-11, 0.836×).
- Bit-exactness taxes on the CUDA path (serial scan walks, u64 build) — CLAUDE.md
  only requires ~1e-6 there, but the bit-exact anchor tests are our strongest
  validation signal; keep unless a lever demands relaxing (none currently does —
  parprefix, the one reordering lever, lost on P100 anyway).

## 3. Improvement plan

Ordering principle: evidence first (P0), then the largest structural lever with a
proven mechanism (P1 build), then evidence-gated scan work (P2), then host-side
(P3, parallelizable with P1/P2). Every lever ships behind an env hatch, is
validated bit-exactly where order-free (u64 paths) or within the documented ~1e-6
envelope otherwise, and gets a same-session order-alternated warm-median-3 Kaggle
A/B before any default flip. Counts tripwires (`COUNTS` line) must prove the new
code ran.

### Phase 0 — Re-baseline + first real kernel profile (1 Kaggle session)

The dead-toggle refactor (~3.4k lines removed) landed after the last bench; and no
`nsys`/`ncu` data has ever been collected.

- **T0.1** Re-run `phase31_ab` (rust on-device vs host_cuda vs official) on current
  master. Confirms ~7.1 s / 3.4 s still holds post-refactor.
- **T0.2** `nsys profile` both the rust wheel and official `lightgbm==4.6` CUDA on
  the same corpus. Deliverable: per-kernel table (name, count, total device time,
  avg µs) for both. Maps our drained buckets to actual kernels and gives the first
  direct kernel-vs-kernel comparison (our staged sibling scan vs
  `FindBestSplitsForLeafKernel`; our u64 build vs `CUDAConstructHistogramDenseKernel`).
- **T0.3** `ncu` (sections: Occupancy, MemoryWorkloadAnalysis, SchedulerStats,
  WarpStateStats) on our top-3 kernels: staged sibling scan, u64 resident build,
  resident scatter. Answers the RC-1 scan mystery: where do 350–600 µs/split go —
  achieved occupancy, DRAM/LDS throughput, stall reasons, launch tail.
  (If `ncu` is blocked on Kaggle by driver permissions, fall back to nsys kernel
  timings + a bisection spike: launch the scan kernel N× back-to-back standalone
  to separate per-launch fixed device cost from per-bin cost.)
- **Exit criteria:** a kernel-level gap table apportioning the ~3.0 s device
  compute; the scan residue attributed (occupancy vs fixed overhead vs memory);
  go/no-go evidence for P2 options.

### Phase 1 — Build kernel redesign: row-major tiled multi-feature build (2–4 sessions)

Target: build 1.05–1.25 s → ~0.3–0.5 s (−0.7 to −0.9 s). Mechanism is proven by
the official kernel: eliminate the per-feature g/h re-gather and the column-major
row gather.

- **T1.1** Add a **row-major resident bin store**: one packed `num_data × num_features`
  u8 (native-width) buffer uploaded once per train alongside the existing
  feature-major `ResidentBins` (+25 MB at 500k×50 — fine). Reuse/extend
  `kernels/row_data.rs` (`CudaRowData` §13 analog already models exactly this
  layout — currently prediction-only).
- **T1.2** New build kernel, grid **(feature_tile, row_block)**:
  each cube owns a tile of T features (LDS budget: T × 256 bins × 16 B ≤ 48 KB ⇒
  T ≈ 8–12; autotune T like `rowpart_target_cubes`), walks its row-block of
  `leaf_rows` once, per row reads g/h ONCE (registers), then for the T features
  reads T consecutive bytes from the row-major store (coalesced across lanes) and
  does 2·T u64 LDS atomic adds. Global merge unchanged. Traffic per (row,feature)
  drops from ~9 B random to ~1.2–2 B mostly-coalesced.
  **u64 atomics keep the order-free bit-exactness** ⇒ validation is byte-identity
  vs the current kernel (same gate style as `fixed_grid_build_byte_identical_to_exact_grid`
  in `crates/lgbm-compute/tests/resident_perm_partition.rs`), locally on gfx1151
  and on the Kaggle CUDA gate.
- **T1.3** Wire behind `LGBM_BUILD_ROWMAJOR` (default off), counts tripwire
  `build_rowmajor=`, `set_*_override` for in-process A/B; both grow-driver build
  sites (`build_smaller`, root build) + the fixed-grid deferred twin can follow later.
- **T1.4** Kaggle A/B (order-alternated, warm-median-3, drained build bucket +
  wall) → default flip on a win; keep hatch.
- **T1.5 (optional, corpus-dependent)** mfb-skip in the new kernel + reuse the
  existing fix machinery (`fix_feats` arrays + `fix_compact_from_raw_f64_on`
  tail already reconstruct cells) to write `leaf_total − Σothers` for the mfb
  cell. NOTE this changes the mfb cell from "directly accumulated" to "derived"
  ⇒ bit-exactness must be re-proven (f64 subtract of exactly-represented
  fixed-point sums is exact — verify) or gated. Skew-dependent win: ~0 on the
  uniform bench corpus, large on real data. Do after T1.4, behind its own hatch.
- **Risk:** LDS pressure at T too high hurts occupancy (autotune T);
  the row-major store doubles resident bin memory (25 MB → 50 MB total, fine);
  two stores must stay in sync (both derive from the same `BinColumn`s at upload).

### Phase 2 — Scan: evidence-gated (1–3 sessions, shaped by P0)

Target: scan 1.8 s → 0.8–1.2 s. Do NOT start before P0 — two full redesigns
(pargain, parprefix) already lost here by attacking the wrong axis.

Branch on the ncu verdict:
- **If per-launch fixed device overhead dominates** (sm_60 info-buffer memcpy,
  kernel prologue): (a) re-test on sm_70+ (T4/L4) to size the P100-specific share —
  if large, consider declaring P100-specific and re-baselining the campaign on a
  newer GPU (user decision: P100 is today's bench GPU, but the product target is
  ROCm + generic CUDA); (b) shrink the launch: comptime-specialize `num_bin=255`
  (the default) to unroll and drop bounds checks; (c) merge the co-scan's
  remaining companion launches (the frontier reduce is already batched — check
  what remains in the scan bucket per split in the nsys trace).
- **If occupancy/latency dominates** (100 cubes, one wave, serial walk latency
  exposed): raise cube count without reordering f64 — e.g. split each feature's
  REV/FWD into separate cubes (200 cubes), or co-scan MORE work per launch by
  scanning both siblings' features in one grid (already done) plus moving the
  min_data/hessian gating earlier to skip dead bins.
- **If LDS staging dominates**: skip staging and scan directly from global
  (histogram is read once per direction; L2-resident at 8 KB/feature), or stage
  only the accumulated candidate arrays (what pargain phase 1 stores).
- **Contract note:** any reordering variant is ~1e-6-legal on CUDA per CLAUDE.md,
  but both reordering attempts lost on P100 for perf reasons — prefer
  order-preserving fixes.

### Phase 3 — Host-side per-train work (1–2 sessions, parallel with P1/P2)

Target: −0.6 to −0.9 s total.
- **T3.1** Parallelize binning with rayon across features (bin-mapper find-bin +
  column fill are per-feature independent; mirrors official's
  `#pragma omp parallel for` in `Dataset::ConstructBinMappers`). CPU-side, fully
  testable locally, bit-exact by construction (per-feature outputs unchanged).
  0.9 s → ~0.2–0.3 s on Kaggle's 4-core host (validate: binning bucket in ledger).
- **T3.2** Audit setup/upload/tail (~0.85 s): reuse the perm/frontier/slot
  allocations across trees (pool them on `GpuBackend` like `GradResidency`),
  drop the per-tree 2 MB perm tail readback if the score scatter can consume the
  device perm directly (it already scatters by ranges — check whether the host
  layout readback is still needed when `LGBM_SCORE_FUSED_SCATTER` is on).
- **T3.3** grad 0.4 s residual: nsys will show whether it is the objective kernel
  or the remaining 2×2 MB D2H readbacks; if the latter, keep g/h device-resident
  end-to-end for the learner (they already are for the build — the readback only
  feeds host bookkeeping sums; compute those on device in the existing rootfold).

### Phase 4 — Deferred / blocked items (documented, not scheduled)

- cubecl multi-stream overlap (official's stream 0/1 trick) — blocked on cubecl
  0.10 single-queue architecture; revisit on a cubecl release with user streams.
- CUDA graphs — mechanism proven and archived (`vendor/cubecl-cuda/` fork +
  recipe); only worth revisiting if a future image/regression re-inflates
  per-launch enqueue ≫ 10 µs.
- Sync-deferral revisit path: co-pack the deferred scans
  (`subtract_scan_resident_siblings_into_frontier_devcount` exists) + which-aware
  staged devcount twin on CUDA, then re-run the P100 A/B — only if P1/P2 make the
  loop enqueue/sync-bound again.
- Quantized-gradient mode (official's fastest path, off by default) — a feature-
  parity project, not a benchmark lever.

## 4. Projection & decision points

| After | Expected wall (P100) |
|---|---|
| today | ~7.1 s |
| P1 (build) | ~6.2–6.4 s |
| P2 (scan, mid case) | ~5.4–5.8 s |
| P3 (host) | **~4.5–5.0 s** |

Official is 3.4 s. The last ~1.1–1.6 s is partition+pick+treesplit device time,
~30k × cubecl dispatch, the 2 syncs/split, and cubecl codegen quality — i.e.
mostly upstream/framework overhead. Reaching parity likely requires either
cubecl-level work (streams, dispatch cost, graph integration) or relaxing the
per-split sync structure beyond what the measured-net-negative deferral tried.
**Decision point after P3:** stop at ~1.4× (document), or open a cubecl-upstream
workstream.

## 5. Phase 0 RESULTS (2026-07-15, Kaggle P100, `lgb-rs-phase0-nsys-baseline`)

**T0.1 re-baseline (order-rotated ×2, median of 3 warm fits, 500k×50×100):**
official **3.32 s** · rust on-device **5.20 s** (ratio **1.57×**) · rust host_cuda
6.82 s. The gap is smaller than the campaign-era 7.1 s/2.1× (newer image + the
T-01..T-13 stack + refactor tree). rust host↔on-device predictions **bit-identical**
across all rounds; official's own predictions vary run-to-run (nondeterministic
f32 atomics) — our CUDA path is deterministic, theirs is not.

**T0.2 nsys — first kernel-level truth (one 100-tree fit each):**
total GPU device time: ours **1.348 s** vs official **0.810 s** — i.e. only ~26% of
our 5.2 s wall is device compute. Decomposition of the 0.54 s device deficit:

| Chain | ours | official | Δ |
|---|---|---|---|
| scan+reduce+pick | 737 ms (staged-subtract scan 131 µs/launch ×2385 = 313 ms; **reduce_scan_output_into_two_leaves 119 µs ×2385 = 283 ms**; single-scan 87 ms; single-reduce 55 ms; find_best_leaf 48 ms) | 95 ms (FindBestSplits 7.5 µs/launch; SyncBestSplit 7.1 µs; FindBestFromAllSplits 3.7 µs) | **+642 ms, 7.8×** |
| partition | 226 ms (mark_block_scan 40.7 µs; scatter_bc_smem 24.7 µs; split 7.3 µs) | 76 ms (5 kernels à 4–6 µs) | +150 ms, 3× |
| build+fix+subtract | 312 ms (u64 LDS build **80.5 µs**/launch) | 612 ms (ConstructHistogramDense **196.7 µs**/launch) | **−300 ms — ours 2× FASTER** |

**T0.2 cuda_api_sum (ours):** cuEventSynchronize 1.16 s (8 802 calls);
**cuMemcpyHtoDAsync 482 ms across 53 043 calls (~17.7 uploads/split)**;
cuLaunchKernel 177 ms/32 009. The host half of the wall is sync waits + the
per-launch upload storm (cubecl sm_60 info buffers + residual per-launch arg
buffers), plus binning 0.86 s, setup/upload/tail ~0.7 s, grad+score 0.34 s.

**T0.3 ncu:** blocked (`ERR_NVGPUCTRPERM`, driver counter permissions — expected
fallback). NOT needed: the kern_sum table already answers the scan mystery — the
staged scan really does cost 131 µs device time/launch, and official proves
7.5 µs is achievable for the same logical work on the same GPU.

**Plan revisions (supersede §3 priorities):**
1. **P1 (row-major build redesign) is REFUTED — deprioritize to backlog.** Our
   u64 build beats official's build 2× on device time; the grad/hess-amplification
   analysis was real but the kernel is not the bottleneck in practice.
2. **New P1 = the scan chain (~740 ms device → target ~150 ms):**
   (a) `reduce_scan_output_into_two_leaves_kernel` — 2-cube serial argmax at
   119 µs vs official's 3.7 µs analog. A parallel tree-reduce argmax under the
   total order (gain desc, feature-index asc) is order-independent ⇒ **bit-exact**,
   isolated, top ROI (−250+ ms). Same for the single-leaf reduce + find_best_leaf.
   (b) staged scan kernel 131 µs → official-shape rewrite: one thread per bin,
   block prefix-sum, per-thread gain, argmax reduce (~1e-6 f64 reorder, allowed;
   note pargain/parprefix failed by ADDING phases to the staged scaffold — the
   test is a clean full-shape rewrite, not another hybrid).
3. **P2 = host-side:** binning 0.86 s (rayon), the 53 k per-split H2D uploads
   (audit our residual per-launch buffers; the cubecl info-buffer share is
   upstream/sm_60), setup+upload+tail pools.
4. Partition kernels (~150 ms) third-tier; build backlog.

## 6. P1a RESULT (2026-07-15, Kaggle P100, `lgb-rs-p1a-reduce-par`) — SHIPPED, DEFAULT ON

Plane-parallel frontier reduce (`LGBM_REDUCE_PAR`, plane twins of the two serial
single-thread reduce kernels; no SharedMemory/barriers — `plane_max` gain +
`plane_min` feature-key argmax, bit-exact via the strict (gain desc, feat asc)
total order). **A/B: base 5.177 s → redpar 4.880 s = 1.061× (−297 ms)**, preds
bit-identical, counts proof reduce_par=2880 vs 0, CUDA parity test 3/3 on P100.
nsys: `reduce_scan_output_into_two_leaves` 119 µs → **3.5 µs**/launch (283→8.3 ms
per train); single-leaf 110 µs → 3.3 µs; total device 1.348 → 1.028 s. Local hip
gates green (byte-identity + `cuda_on_device` 7/7). Default flipped ON
(`LGBM_REDUCE_PAR=0` reverts). Gap vs official: 5.20 → **4.88 s vs 3.32 s (1.47×)**.
Remaining scan-chain targets from the §5 table: the staged scan kernel itself
(313+87 ms at 131/147 µs/launch — P1b) and `find_best_leaf_kernel` (49 ms, 16 µs,
same serial-argmax shape — a small P1a-style follow-up).

## 6b. P1b RESULT (2026-07-15, Kaggle P100, `lgb-rs-p1b-official-scan`) — SHIPPED, DEFAULT ON (CUDA)

Official-shape 256-wide scan (`LGBM_SCAN_OFFICIAL`): one lane per bin, two-level f64
block prefix-sum (`block_inclusive_scan_f64`) + block argmax (`block_max_f64` /
`block_min_u32`), stateless `active && !cont && !brk` guard. Three twins (single /
co-pack siblings / subtract-fuse), precedence over parprefix/pargain in the staged +
subtract-fuse launchers, CubeDim=256 + `plane_dim`. **A/B: base 4.604 s → official
4.233 s = 1.0876× (−371 ms)**, preds **BIT-IDENTICAL (max_abs 0.0)** — no split flipped
despite the ~1e-6-by-construction internal gain reorder, counts scan_official=2980 vs 0,
tree-count 100/100, CUDA parity gate green (official arm of `scan_pargain_parity`).
**nsys: the scan kernel drops 131 µs → 16–20 µs/launch** (subtract-fuse twin 20.1 µs
×2385 = 48 ms; single 16.1 µs ×595 = 9.6 ms; total device 0.719 s) — approaching
official LightGBM's ~7.5 µs class, and **the first lever to beat the scan wall where
pargain/parprefix both lost** (both kept CUBE_DIM=64; the 256-wide geometry was the
untested variable, confirmed). Default flipped ON for **CUDA only**; hip keeps parprefix
(gfx1151 drain: official 714 ms/scan vs parprefix 691 ms, ~3% slower — no win). Hatch
`LGBM_SCAN_OFFICIAL=0` reverts; the cpu f64 anchor never runs it (bit-exact merge gate
`resident_tree_bit_exact_to_u64_integer_path` on CpuBackend untouched). Gap vs official:
4.88 → **~4.5 s vs 3.32 s (~1.36×)**. Contract note: this is the first ~1e-6-by-
construction scan variant made a CUDA default — empirically bit-identical on the bench
corpus, but a different corpus could flip a split by ~1e-6 (allowed on the GPU path per
CLAUDE.md; the deterministic bit-exact validation lives on the cpu anchor, which is
unaffected).

## 7. Lever reassessment after P1a (2026-07-15) — where the remaining 1.47× lives

With the nsys kernel table + P1a shipped, the surviving levers were re-checked
against evidence. Most of the plan's original candidates are now refuted or spent:

| Lever | Status | Evidence |
|---|---|---|
| P1a plane reduce | **SHIPPED 1.061×** | §6 |
| P1 row-major build | **refuted** | our u64 build 2× faster than official (§5) |
| P3.1 rayon binning | **already done** | `booster.rs:546` `into_par_iter`, `LGBM_PAR_BIN` default-on; the 0.86 s is already-parallel cost on ~4 vCPU |
| P1b scan parallelization | **high-risk / likely-refuted** | parprefix (= all-lanes parallel-prefix scan) measured NET-NEGATIVE on P100 (spike104): cheap 1:2 f64 + 100-cube occupancy starvation make barriers cost more than the serial walk. Our 131 µs serial walk is near-optimal *for our kernel shape*; official's 7.5 µs comes from a 256-thread CUDA `ShufflePrefixSum` cubecl 0.10 can't match — a **primitive-quality/upstream** gap, not an algorithm we can port and win with |
| sync deferral (2L→L+2) | **refuted** | T-11 P100 A/B 0.836× (deferral's required compute > sync saved) |
| find_best_leaf argmax | **not worth it** | 49 ms but ≤31 elements/launch ⇒ launch-bound, not walk-bound; parallelizing won't help |
| CUDA graphs | **refuted** | 1.04× (device-bound; enqueue only ~4–11 µs/launch now) |

**Honest conclusion:** device compute (1.03 s) is now close to official (0.81 s)
except the scan, and the scan gap is a cubecl-primitive-quality problem that
arithmetic parallelization has already lost on P100. The wall gap (4.88 s vs
3.32 s) is dominated by **host sync waits (cuEventSynchronize 1.16 s / 8802) +
cubecl sm_60 per-launch dispatch (the 53k-upload storm, ~482 ms) + binning
0.86 s** — i.e. framework/host overhead, not our algorithms. We are at genuine
diminishing returns on safe, mechanical, P100 levers (the pre-nsys campaign
predicted this; the kernel table now confirms it).

**Remaining directions, all larger bets (need a user decision):**
1. **Smaller device micro-levers** (~100–200 ms, low risk, mechanical): partition
   `mark_block_scan` (123 ms) + `fix_compact` (62 ms) shape review vs official's
   4–12 µs analogs. Diminishing but bit-exact and local-testable.
2. **cubecl-upstream / newer-GPU** (large): the scan primitive gap + sm_60
   dispatch are upstream. A cubecl release with user streams (official's stream
   0/1 overlap) or cheaper dispatch, or re-baselining on sm_70+ (grid_constants
   removes the per-launch info buffer), could move the host half. Out of our
   codebase.
3. **Accept ~1.47× and stop** — document P1a as the last safe win; the residue is
   upstream/framework-bound.

## 8. Bench & validation protocol (invariant)

- Same-session arms only; order-alternated; warm-median-of-3; `LGBM_PHASE_PROF=1`,
  `LGBM_AUTOTUNE=0`; tree-count gate; counts tripwire proves the lever ran.
- Drain mode (`LGBM_GROW_DRAIN=1`) is the device-time source of truth; free-run
  `pick` aliases queued device work.
- Local gfx1151 = functional/byte-identity validation only, never perf verdicts
  (compute-bound APU, not P100-representative).
- Every lever: env hatch + in-process override + local byte-identity/envelope
  gate before the Kaggle A/B; default flips only on a measured same-session win.

## 9. Post-P1b root cause & the P2 host-side plan (2026-07-15)

### 9.1 Root cause, restated after P1a+P1b

State: rust on-device **~4.23 s** (P1b session; base 4.60 s) vs official **3.32 s**
⇒ gap ~0.9–1.2 s (~1.3×). nsys device totals: **ours 0.719 s vs official 0.810 s —
our device compute now BEATS official.** The scan chain, build, and reduce are no
longer root causes. The entire remaining gap is **host-side**, decomposed
(code-verified 2026-07-15, `grow_driver.rs` / `lib.rs` / `partition.rs`):

| # | Host cost | Mechanism (verified) | Size (est.) |
|---|---|---|---|
| H1 | g/h round-trip per tree | objective computes g/h ON DEVICE (`get_gradients_resident_on` → `GradResidency.grad/hess` handles), the boosting loop reads them back D2H, then `grow_tree_on_device_resident` takes `&[f32]` and `upload_resident_grad_hess` re-uploads H2D (`lib.rs:3621`, fresh `create_from_slice` per tree). 8 MB moved/tree = **800 MB/train** round-trip that shouldn't exist. Plus `grow_max_abs` host scan of both arrays/tree (`grow_driver.rs:2387`) and the host iota `Vec` alloc/tree (`grow_driver.rs:2415`, dead on the resident-perm arm). | grad 0.34 s + upload share |
| H2 | per-tree device alloc churn | `ResidentPermPartition::new` (5 buffers + iota launch), `DeviceCudaTree::new`, `DeviceFrontier::new`, `DeviceLeafSplits::new` (2 zeroed `create_from_slice` = 2 real H2D uploads) — ~10 pool allocs/frees **per tree**, ×100, each hitting cubecl's reserve/cleanup path (which takes pool-reclamation fences). | setup 0.27 s + tail share |
| H3 | 2 blocking syncs/split | pick 8-int export + `read_split` 6-int child ranges ⇒ ~6 200 blocking reads/train; the cuEventSynchronize 1.16 s/8 802 calls bucket. Deferral (2L→L+2) measured 0.836× (T-11) — but that arm was pre-P1b (legacy-kernel scan penalty) and has since been deleted in the dead-toggle refactor. | part of 1.16 s waits |
| H4 | per-launch dispatch storm | 53 043 `cuMemcpyHtoDAsync` (~17.7/split, 482 ms) — cubecl-cuda sm_60 per-launch info buffers + arg buffers. Upstream code, but we maintain `vendor/cubecl-cuda/` (CUDA-graph fork, currently unwired). | ~0.3–0.48 s |
| H5 | binning | already rayon-parallel (`LGBM_PAR_BIN` default-on) on ~4 vCPU; official also bins on CPU — **its share inside official's 3.32 s has never been measured**, so the true deficit is unknown. | ≤0.86 s, likely much less |

### 9.2 Phase P2 — host residency & pooling (safe, bit-exact, locally testable)

Ordering: evidence first, then the levers that are bit-exact-by-construction
(same bytes, same kernels — only *when/where* buffers live changes). Every lever:
env hatch + `set_*_override` + counts tripwire + local byte-identity gate
(cpu anchor + gfx1151) + same-session order-alternated warm-median-3 Kaggle A/B
before any default flip (§8 protocol). Dependencies: cubecl stays 0.10.0 (latest
published — verified, no upgrade lever); no new crates needed for P2.

- **P2.0 Evidence re-baseline (1 Kaggle session).** Post-P1b nsys rerun with
  `cuda_api_sum` (how much of the 1.16 s sync / 482 ms upload storm survives
  P1a+P1b); free-run + drain ledgers; and — new — time official's
  `Dataset` construction separately from `booster.train` to size H5 honestly.
  Exit: an updated host-side table apportioning the ~1 s gap across H1–H5.
- **P2.1 End-to-end device-resident g/h (attacks H1).** Add a device-handle grow
  entry (`Option<(&Handle, &Handle)>` alongside the host slices, or a
  `GradSource::{Host, Device}` enum like `NumDataSrc`): the boosting loop passes
  the `GradResidency` handles straight through; `upload_resident_grad_hess`
  becomes a no-op on that arm; `grow_max_abs` moves to a small device max-abs
  reduce (or is folded into the existing objective kernel launch); the root fold
  already routes through `backend.root_grad_hess_sum`. The D2H readback stays
  ONLY for consumers that genuinely need host g/h (custom objectives, bagging
  subset gather, the cpu anchor — all keep the host arm byte-unchanged).
  Bit-exact by construction (identical bytes reach identical kernels).
  Hatch `LGBM_GRAD_DEVICE_PASSTHRU`, tripwire `grad_passthru=`. Est. −0.15–0.30 s.
- **P2.2 Per-tree device-struct pool (attacks H2).** Persist
  `ResidentPermPartition` / `DeviceCudaTree` / `DeviceFrontier` /
  `DeviceLeafSplits` on `GpuBackend` keyed by geometry (the `GradResidency` /
  desc-hoist pattern, reset in `reset_resident_pool`); re-seed per tree with
  device kernels (iota relaunch; zero-fill kernels replacing the zeroed
  `create_from_slice` uploads — note the desc-hoist precedent: `client.empty` +
  full overwrite where a pass writes every cell). Hatch `LGBM_GROW_POOL`,
  tripwire `grow_pool=`. Est. −0.10–0.25 s.
- **P2.3 Tail perm readback removal (H2/H3 tail).** The once-per-grow 2 MB
  `read_perm` feeds the host `LeafPartitionLayout` rebuild, but the resident
  score update already scatters by device ranges. Make the layout LAZY: keep the
  device perm + ranges as the primary artifact; materialize the host layout only
  for consumers that need it (linear trees, `renew_leaf_output` (l1), host score
  paths, tests). Est. −0.05–0.10 s (200 MB D2H + 100 syncs/train).
- **P2.4 Trivia (with P2.2).** Skip the host iota `Vec` on the resident-perm arm;
  audit remaining per-split `create_from_slice` in the partition/treesplit
  launchers (nsys says ~17.7 uploads/split; desc-hoist covered scan+build only).
- **P2.5 Partition micro-shape (only if P2.0 re-confirms).** `mark_block_scan`
  40.7 µs and `fix_compact` 25–62 µs vs official's 4–12 µs analogs — P1a-style
  geometry review, bit-exact (u32 marks / integer scans). Est. −0.10–0.15 s device.

### 9.3 Phase P3 — framework bets (each needs a user decision)

- **P3.1 Fork lever on the dispatch storm (H4).** We already maintain
  `vendor/cubecl-cuda/` with a proven server-thread hook. Options, in escalating
  invasiveness: (a) memoize/pool the per-launch info-buffer uploads for repeated
  identical launch shapes; (b) pre-pin a staging arena to cut `reserve→cleanup`
  fences; (c) upstream an issue/PR with the nsys evidence. Est. −0.2–0.4 s but
  carries fork-maintenance cost — decide after P2.0 sizes what remains.
- **P3.2 Sync-deferral revisit (H3) — only if the post-P2 ledger shows pick/sync
  dominant.** The T-11 0.836× verdict predates P1b: the deferred arm paid for
  legacy-kernel scans + build-LEFT; a rebuild atop official-shape devcount scan
  twins + co-pack (`subtract_scan_resident_siblings_into_frontier_devcount`
  exists) could flip the sign. The arm was deleted (25a1da1) — this is a
  re-implementation, not a re-enable. High effort; evidence-gated.
- **P3.3 Multi-stream overlap** — still blocked on cubecl 0.10 single queue;
  watch upstream releases.
- **P3.4 Stop.** After P2, projected ~3.6–3.9 s vs 3.32 s (~1.1–1.2×) with
  device compute already ahead of official — a defensible place to declare the
  CUDA benchmark done and return to feature parity work.

### 9.4 Projection

| After | Expected wall (P100) | Gap vs 3.32 s |
|---|---|---|
| today (post-P1b) | ~4.2–4.5 s | ~1.3× |
| P2.1 g/h passthrough | ~4.0–4.2 s | ~1.23× |
| P2.2–P2.4 pools + tail | ~3.8–4.0 s | ~1.17× |
| P2.5 partition | ~3.7–3.9 s | ~1.14× |
| P3.1 fork lever (if taken) | ~3.4–3.6 s | **~1.05×** |

## 10. P2 v1 RESULTS (2026-08-06, Kaggle P100, `lgb-rs-p2-host-residency`)

Corpus/protocol unchanged (500k×50 regression, 100 trees, nl=31, order-rotated
warm-median-3, fresh process per run; official = lightgbm 4.6.0 pip source
build with `USE_CUDA=ON`; rs base = origin/main `8138757` + the P2.1/P2.2 diff,
hatches OFF).

| arm | warm-median | verdict |
|---|---|---|
| official | **2.836 s** | official IMPROVED vs the July image (3.32 → 2.84 s) |
| rs_base | **6.128 s** | ~1.5 s REGRESSION vs the P1b-era 4.60 s base (see below) |
| rs_pass (P2.1) | 5.949 s | **WIN 1.030×** — preds byte-identical, counts 100/100, drained `upload` 330→23 ms → **default flipped ON** |
| rs_pool (P2.2) | 6.148 s | **WASH** — drained `setup` unchanged (cubecl allocs are cheap on this image) → default stays OFF, hatch kept |
| rs_p2 | 6.006 s | passthru carries the stack |

Gates all green: 100 trees every run; rs arms byte-identical (max_abs = 0.0);
official envelope 3.07e-5.

Drained rs_base ledger (per 100-tree train): grow wall 4406 ms = build 1237 +
scan 1114 + partition 561 + pick 325 + upload 330 + setup 269 + treesplit 201 +
reduce 182 + tail 104 + rootfold 70; outside grow: binning 920, grad 371,
score 191, snapshot 94.

**Regression root cause (code-confirmed, fix validated locally):** the
max_delta_step/path_smooth port (`9cc111c`) made `get_leaf_gain_full` — called
twice per candidate bin in every scan kernel — ALWAYS compute the closed form,
the clamped-output form AND the smoothing blend (~5 f64 divides per side vs 1),
then `select` the closed form when both features are OFF. C++ compiles those
axes away as template bools (`USE_MAX_OUTPUT`/`USE_SMOOTHING`). Fix: uniform
runtime branches in `get_leaf_gain_full` / `calculate_splitted_leaf_output_full`
(+ f32 mirrors) — bit-exact on both arms (each returns exactly the value the
select form selected); gain_params/advanced parity fixtures green. Measured in
v2 (`lgb-rs-p2v2-gainfix`, two-wheel same-session A/B).

Accounting note: BOTH walls include binning/Dataset construction (official's
Dataset is lazy — it bins inside `train()`; ours bins inside `train()` too), so
the comparison is honest end-to-end.

## 11. P2 v2 RESULTS (2026-08-07, Kaggle P100, `lgb-rs-p2v2-gainfix`)

Two-wheel same-session A/B (wheel A = v1 diff / gain helpers unfixed; wheel B =
gain fix + P2.4 host-copy trivia), same corpus/protocol as v1.

| arm | warm-median | delta |
|---|---|---|
| official | 2.793 s | (2.836 in v1 — stable) |
| rs_old (wheel A) | 6.079 s | v1 rs_base twin (6.128) — reproducible |
| rs_fix (wheel B) | 5.903 s | gain fix + trivia = **1.030×** |
| rs_fix_pass (B + passthru) | **5.720 s** | cumulative **1.063×** vs rs_old |

Gates: 100 trees every run; rs_fix and rs_fix_pass preds BYTE-IDENTICAL to
rs_old (max_abs = 0.0 — the gain-fix bit-exactness claim held on hardware);
official envelope 3.07e-5.

**Mechanism finding (drain ledgers):** the drained scan bucket did NOT move
(1091 → 1085 ms) — the gain fix's win is the P2.4 host copies (snapshot 88→43,
in_iter_other 146→93) plus scan-kernel device time recovered under free-run
overlap. The drained grow (4.05 s) is dominated by dispatch: ~18 500 launches +
6 200 blocking syncs/train through cubecl 0.10, at this image's per-launch/sync
cost. Free-run wall (5.72) ≈ drain wall (5.96) − 0.24 — the loop is nearly
serialized by its 2 blocking syncs/split, so device/host overlap recovers
almost nothing.

**Post-P2 state: rs 5.72 s vs official 2.79 s (2.05×).** The July "~1.3× gap"
numbers do not transfer: Kaggle image changes moved official 3.32→2.79 s and
our dispatch-bound wall the OTHER way (the ~1.5 s "regression" vs the P1b-era
4.60 s base is at most ~0.18 s code regression — the rest is the image's
launch/sync cost profile, which hits our 185-launch/62-sync-per-tree structure
far harder than official's fused-kernel CUDA path).

### Where the remaining 2× lives (next campaign, in order)

1. **Launch-count structure** — level-batched build/scan (one launch per tree
   LEVEL, not per split) is the only lever that attacks the ~18.5k
   launches/train. Big redesign, multi-session.
2. **Sync-count** — the 2 blocking syncs/split (pick export + read_leaf). The
   T-11 sync-deferral verdict (0.836×, 2026-07-15) was measured on a
   CHEAP-launch image; on today's dispatch-cost profile the sign could flip.
   The deferred arm was deleted (dead-toggle refactor) — re-implementation.
3. **Binning 0.84–0.96 s** — rayon-parallel already; official pays a similar
   CPU cost inside its wall, so this is parity, not deficit.
4. cubecl-upstream (dispatch cost, multi-stream) — tracked, out of our tree.

## 12. P3 transport levers (2026-08-07, Kaggle P100, `lgb-rs-p3-transport`)

Attacks lever 1/2/4 of §11 at their COMMON root: the per-launch/per-sync
transport tax in the cubecl 0.10 dispatch layer, via two vendored forks wired
with `[patch.crates-io]` (commit 7f97ee5). Both are env-gated and default to
byte-for-byte upstream behavior:

- **`CUBECL_DEVICE_INLINE=1`** (vendor/cubecl-common): cubecl 0.10 runs the
  ComputeServer on a dedicated server thread — EVERY launch is a cross-thread
  channel hop and every blocking readback a 2-way thread ping-pong (~18.5k +
  6.2k per train). The lever swaps in the upstream reentrant-mutex handle:
  tasks run inline on the caller thread, zero hops.
- **`CUBECL_CUDA_INFO_ARENA=1`** (vendor/cubecl-cuda): P100 is sm_60 — no
  `grid_constants` — so every launch uploads an info/metadata+scalars buffer
  through pinned-pool reserve → staging `Bytes` → GPU-pool reserve →
  `cuMemcpyHtoDAsync` → handle drop → drop-queue. The drop-queue flushes every
  64 staged buffers with a BLOCKING fence sync — a hidden serializer that
  explains §11's "free-run ≈ drain". The arena replaces all of it with a
  persistent pinned+device ring (one async H2D per launch, one stream-sync per
  32MB wrap).
- **`CUBECL_CUDA_LAUNCH_PROF=1`**: first hard per-launch host-cost
  decomposition (command/info/resource/kernel segments + drop-flush + blocking
  fence-wait totals).

Protocol unchanged (§8): 500k×50, 100 trees, nl=31, order-rotated
warm-median-3, fresh process per run, one wheel, env-toggled arms; byte-identity
gate = all rs arms' preds identical to rs_base.

### Interim status (2026-08-07, session 1)

- Implementation SHIPPED (commits `7f97ee5`, `686e212`), local gates green:
  cpu suite 183/183 with the inline handle ON and OFF; REAL-GPU (local ROCm
  gfx1152) `rocm_backend_parity` 5/5 + `cuda_on_device` 7/7 — including
  `resident_tree_bit_exact_to_u64_integer_path` — under `CUBECL_DEVICE_INLINE=1`.
- Fix found by the local gates: upstream's reentrant handle panics when
  `utilities()` is called before the first submit (the normal client-ctor order);
  686e212 lazy-inits there, mirroring `with_lock`.
- Kaggle run 1 (`lgb-rs-p3-transport` v1, built pre-fix): the rs_inline arm hit
  that panic path and the session hung 4h+, exhausting the weekly GPU quota
  (refresh 2026-08-08T00:00Z). Hardened v2 (900s per-arm timeout) queued for the
  refresh window.

### Results (run 2, `boomvector/lgb-rs-p3-transport`, Tesla P100, 2026-08-07)

Protocol §8 (500k×50, 100 trees, nl=31, order-rotated warm-median-3, one wheel,
fresh process per run; wheel = main incl. 686e212, built in 11m45s). Gates ALL
green: 100 trees every run; rs_arena / rs_inline / rs_inline_arena preds
**byte-identical to rs_base (max_abs = 0.0)**; official envelope 3.065e-5
(identical to §10/§11 — same corpus, same baseline).

| arm | warm-median | verdict |
|---|---|---|
| official (4.6.0 pinned) | **2.886 s** | stable (2.79–2.89 across sessions) |
| rs_base | 5.889 s | post-P2 baseline reproduced |
| rs_arena | 5.995 s | **−1.8% LOSS → default stays OFF**, hatch kept |
| rs_inline | **5.635 s** | **WIN 1.045×**, byte-identical → flip candidate |
| rs_inline_arena | 5.676 s | ≈ inline alone (arena adds nothing on top) |

Drain walls: rs_base 6.149 s, rs_inline_arena 5.920 s.

**THE LOAD-BEARING FINDING — first per-launch host-cost decomposition (prof
runs, cumulative at 20 000 launches ≈ 1.08 trains):**

| segment | rs_base | per launch |
|---|---|---|
| command/stream-resolve (entry→CP1) | 12.9 ms | 0.6 µs |
| count+info upload (CP1→CP2) | 113.8 ms | 5.7 µs |
| resource resolution (CP2→CP3) | 19.1 ms | 1.0 µs |
| **kernel segment (CP3→CP4)** | **1 948 ms** | **97 µs** |
| drop-queue flush (n=420) | 4.2 ms | — |
| blocking fence waits (n=5 586) | 492.7 ms | 88 µs/wait |

The per-launch tax is NOT the info upload (5.7 µs), NOT pool churn, NOT the
drop-queue (4 ms total — its blocking-flush role was overestimated): **~97 µs
per launch sits inside CP3→CP4** — `command.kernel()`: two full `KernelId`
hash/eq lookups (module_names contains_key + execute_task get), param
marshaling, `cuLaunchKernel`, logger. ×18.5k launches ≈ **1.8 s/train — the
single biggest addressable chunk of the remaining 1.95× gap.** The arena's
hypothesized targets (info upload + drop-flush stalls) measured small, which is
exactly why rs_arena is a wash; the inline win (~0.25 s) matches the removed
client-side channel hops (which the server-side segments never contained).

### Verdicts & next levers (ranked by measured evidence)

1. **`CUBECL_DEVICE_INLINE` → flip default ON for cuda** (app-side env default
   at booster init unless user-set): 1.045×, byte-identical, locally validated
   on cpu + real ROCm GPU (bit-exact gates), P100-validated.
2. **CP3→CP4 teardown (next fork round):** add sub-checkpoints inside
   `command.kernel()` to split KernelId-hash / marshal / cuLaunchKernel /
   logger, then memoize module resolution behind a cheap key (pointer-identity
   or precomputed u64) — prize is a large share of ~1.8 s/train.
3. **Fence-wait shape (0.5 s/train, 88 µs/wait × 5.6k):** partly genuine
   device-wait; re-rank after lever 2 lands.
4. Arena: keep hatch, default OFF (falsified hypothesis, documented).

Incident notes: run 1 (v1 script) hung 8 h — the UNPINNED official install
resolved lightgbm 4.7.0 whose first CUDA worker hung on the P100 (not an rs
arm); weekly GPU quota exhausted on the yensen2 account → run 2 executed on
boomvector. v2 hardening (4.6.0 pin + 900 s/arm timeout + official-optional)
is committed as `scripts/kaggle/lgb-rs-p3-transport.py` (+ a Colab wrapper at
`scripts/colab/p3_transport_bench.ipynb`; Colab is sm_75+ — functional
validation only).


### Round 3a — CP3→CP4 teardown (2026-08-07, P100, kernel v2)

Wheel = main @ round-3 commit (inline default ON, single-lookup resolve,
funcattr-once). Walls (warm-median-3): official 2.857 | **rs 5.484** | rs_chan
(`CUBECL_DEVICE_INLINE=0`) 5.843 | rs_attr_every 5.517. Preds byte-identical
across rs arms; 100 trees every run. Inline win RE-CONFIRMED on-wheel (1.065×
here); single-lookup + trivia moved rs 5.635 → 5.484.

**Sub-segment split of the 97µs/launch (prof, 20k launches): lookup_ms=1710 of
kernel_ms=1787.** The ONE remaining full-`KernelId` map lookup — i.e. hashing
the kernel's comptime `Info` payload — IS the dispatch storm (~85µs/launch,
~1.6s/train). Everything else is noise: `cuLaunchKernel` 3.1µs, marshal 0.1µs,
attr 0.03µs, drop-flush 3ms total. Verdict: the launch-count structure (§11
lever 1) was never the problem — the per-launch HASH was. Round 3b ships the
two-level fast resolve (bucket by type-name ptr/mode/cube-dim + full-id
EQUALITY inside, no hashing; `CUBECL_CUDA_FAST_RESOLVE=0` kill switch) — if
bucket equality is cheap, projected rs ≈ 3.8–4.0s.

### Round 3b — fast resolve is a WASH; the window is content-insensitive

Kernel v3 (fast resolve default ON): official 3.02 | rs 5.93 | rs_slow
(`CUBECL_CUDA_FAST_RESOLVE=0`) 6.00 (session ~5% slower than 3a overall —
box variance). Prof: `fast_hits=19983/20000` (cache engaged) yet
`lookup_ms=1909` vs hashed `1946` — NO change. Combined with 3a/round-2 data:
the resolve window costs ~85–95µs whether it does TWO full-KernelId hashed
lookups, ONE, or a bucket-get + full-id equality. The kernels' comptime Info
is tiny (no `#[comptime]` params on the hot kernels), so neither hash nor eq
can honestly cost that. Conclusion: an EXTERNAL stall pinned to the window
(scheduler preemption / driver-internal locking), not lookup mechanics. Round
3c sub-instruments the window (key/get/find + `kernel.id()` + ctx-switch
counts) to localize it.

### Round 3c — THE UNMASKING: it was never a per-launch tax

Kernel v4 sub-timers: the resolve window's interior (id 2.1 + key 1.1 + get
2.5 + find 4.1 = **~10ms**) vs the window total (**1750ms**); ctx switches 19
vol/93 nonvol (no preemption). The missing ~1.74s sits in the window TAIL —
which on the ~17 cache-MISS launches contains `compile_kernel`: **the "97µs/
launch dispatch tax" of rounds 2–3b was ~1.7s of ONE-TIME, PER-PROCESS kernel
compilation (NVRTC source→PTX + `cuModuleLoadData` PTX→SASS), smeared across
the 20k-launch average.** The actual launch path costs ~0.5µs/launch. This
also retro-explains: content-insensitivity (3b), arena/fast-resolve washes
(they optimized noise), and the July→August "image regression" (same compile
burst, different CPU speed). The inline-handle win (1.045×) is real and
independent. Official LightGBM ships AOT-compiled kernels — it pays zero.

Round 3d: cubecl 0.10 defaults `compilation.cache = None` → the fork defaults
the PTX disk cache ON (Global root, persists across processes;
`CUBECL_CUDA_PTX_CACHE=0` restores upstream), leaving `cuModuleLoadData` as
the per-process residue (driver SASS cache applies). Compile-path prof line
added (compiles/ptxcache_hits/nvrtc_ms/modload_ms). Arms: official / rs /
rs_nocache.

### Round 3d RESULTS — PTX cache default ON (kernel v5, P100, 2026-08-07)

| arm | walls (r0/r1/r2) | warm-median |
|---|---|---|
| official 4.6.0 | 3.19 / 3.16 / 3.09 | **3.16 s** (this session's box is ~10% slower — same-session compare only) |
| rs (cache ON) | 7.23 (cold: compiles+seeds cache) / 4.06 / 4.04 | **4.05 s** |
| rs_nocache | 6.10 / 6.06 / 6.15 | 6.11 s (recompiles every process) |

Gates: 100 trees all runs; rs_nocache preds byte-identical to rs (0.0).
Compile prof: warm process = `compiles=17 ptxcache_hits=17 nvrtc_ms=0.0
modload_ms=4.2` — NVRTC eliminated, PTX→SASS module load 4ms total (driver
SASS cache). Cold/nocache process = `nvrtc_ms=1794 modload_ms=227`. Drain rs
wall 4.23 s.

**Position: rs 4.05 s vs official 3.16 s SAME-SESSION = 1.28× — from 2.05× at
the start of the P3 campaign.** Cumulative levers: inline device handle
(1.045×, default ON) + PTX cache (1.39× warm, default ON) + single-lookup
resolve/trivia. The cold-start (first process on a box) still pays ~2s NVRTC —
amortized like any JIT; an install/import-time warmup could hide even that
(official amortizes at pip-install compile time).

### Remaining gap (~0.9 s) — next campaign, in order

1. Blocking fence waits ~0.5 s/train (5 586 × ~90µs — partly genuine device
   wait; the §11 sync-deferral machinery is still in-tree).
2. Binning ~0.85 s inside our wall (official pays a similar CPU cost — parity,
   not deficit; only worth attacking if its share differs).
3. Device compute deltas (build 1.14 s + scan 1.09 s drained; official's
   equivalents unmeasured on this image).

### Round 4 (partial) — rs at 3.79s; grow-pool is still a wash

Kernel v6: rs warm-median **3.79s** (box faster this session; r0 cold 6.25
seeds the PTX cache), rs_pool 3.78 — P2.2 pooling stays a WASH even with the
compile noise gone → hatch kept, default OFF. Cache-warm drain: grow 2256ms =
build 911 + partition 392 + setup 271 + scan 180 + pick 175 + treesplit 91 +
tail 122; binning 857, grad 137, score 189. The official arm FAILED this
round: standalone `Dataset.construct()` under CUDA params segfaults (empty
worker output) — the construct/train split is reverted; round 4b answers the
binning-share question with 1-tree walls (≈ construct + fixed overhead) for
both implementations instead.

### Round 4b — the binning question answered: we BEAT official on fixed cost;
### the whole remaining gap is per-tree grow compute

Kernel v7, same-session (P100): official 2.87 / 2.83 / 2.87 → **2.87 s**;
rs 6.35 (cold) / 3.72 / 3.82 → **3.77 s** = **1.31×**. Envelope 3.05e-5, 100
trees everywhere.

**1-tree walls (≈ Dataset construct + fixed init + 1 tree): official 1.665 s,
rs 1.190 s.** Binning is NOT a deficit — our fixed pipeline is ~0.48 s FASTER
than official's. The ledger flips to per-tree marginals:

| | fixed (1-tree) | per-tree marginal ((100t − 1t)/99) |
|---|---|---|
| official | 1.665 s | **12.2 ms/tree** |
| rs | 1.190 s | **26.1 ms/tree** |

The ENTIRE remaining gap (plus our fixed-cost lead) is per-tree grow compute:
drained per tree = build 9.1 ms + partition 3.9 + setup 2.7 + scan 1.8 + pick
1.8 + treesplit 0.9 + tail 1.2 (+ grad 1.3 + score 1.9 outside grow). Next
campaign = device-kernel shape: (1) the u64 LDS build (9.1 ms/tree vs
official's f32-atomic builder — the bit-exactness premium is paid here; a
measured comparison of official's ConstructHistogram device time on this image
would size it), (2) partition micro-shape (3.9 ms/tree, P2.5), (3) per-tree
setup residue (2.7 ms/tree). The transport layer is DONE: launch path 0.5 µs,
sync waits ~0.5 s total, compile amortized.

### P3 final scorecard (all same-session P100, byte-identity gated)

| milestone | rs wall | official | gap |
|---|---|---|---|
| campaign start (§11) | 5.72 s | 2.79 s | 2.05× |
| + inline handle (default ON) | 5.48 s | 2.86 s | 1.92× |
| + PTX cache (default ON) | 4.05 s | 3.16 s | 1.28× |
| round 4b re-measure | **3.77 s** | 2.87 s | **1.31×** |

(Cross-session absolute walls vary ±10% with the Kaggle box; the same-session
gap is the metric. Cold first-process still pays ~2.6 s NVRTC once per
machine/cache — an install/import-time warmup would hide it.)

### Round 5b — device-kernel head-to-head (torch/CUPTI, 20 trees, same corpus)

nsys is not installable on the Kaggle image; torch.profiler's CUPTI activity
tracing captures ALL in-process CUDA kernels (both implementations share the
primary context). Device totals: **official 207.4ms (10.4ms/tree) vs rs
222.1ms (11.1ms/tree) — device compute is essentially PARITY.** The 14ms/tree
wall-marginal gap (26.1 vs 12.2) is ~93% host/serialization, NOT device work.

Per-kernel (ms / 20 trees):

| stage | official | rs | delta |
|---|---|---|---|
| histogram build | **144.0** (ConstructHistogramDense, 240µs×600) | **77.4** (u64 LDS, 125µs×620) | **rs WINS +67** |
| scan+subtract+fix | 24.6 | 13.9 | rs wins +11 |
| partition chain | 20.1 | **66.9** (mark_block_scan 29.4 + scatter_bc_smem 20.4 + fix_compact 12.5 + split 4.6) | **rs LOSES −47** |
| memcpys | 10.0 | **48.0** (incl. 2 283 PAGEABLE HtoD × 11.2µs = 25.5) | rs loses −38 |
| pick | 2.4 | 10.1 | rs loses −7.7 |

**The u64 bit-exact build kernel BEATS official's builder by 1.9×** — the
"bit-exactness premium" worry was unfounded; no numerics trade-off is needed.
Official launches MORE kernels/tree than we do (~390 vs 185) at ~5µs raw-CUDA
enqueue — launch count was never the problem either.

### Final-mile levers (evidence-ranked)

1. **Serialization bubbles + sync round-trips (~10–14 ms/tree):** 62 blocking
   readbacks/tree nearly serialize the loop (host decodes + re-enqueues while
   the device idles). Levers: batched pick+ranges read (ONE sync/split — the
   §11 deferral, shape-preserving devcount variant, machinery in-tree), and
   shrinking host decode/enqueue latency between dependent launches.
2. **Partition device chain (−47ms/20t = 2.3 ms/tree):** mark_block_scan at
   49µs/call and the 34µs scatter are 3.3× official's equivalents (P2.5
   micro-shape: geometry/width, u32 marks).
3. **Pageable→pinned small uploads (−25ms/20t):** 2 283 pageable HtoD/20t at
   11.2µs vs 1.8µs pinned — route stragglers through pinned staging (the
   arena's device-side benefit, revisited with correct attribution).
4. pick kernel shape (−7.7ms/20t).

Transport, compile, build-kernel, binning: all CLOSED (parity or better).

### Round 6 — parscan partition twins: WIN 1.066×, gap now 1.22×

Both resident-partition kernels had SINGLE-OWNER SERIAL scans (thread 0 walking
up to `block_size` dependent global reads with 255 units idle in the mark; per-
block serial sums over up to 1024 block totals in the fused scatter). The
parscan twins (chunk-per-unit + 8-step Hillis-Steele smem exclusive scan; strided
accumulate + smem tree reduction) are integer-exact by construction, proven
byte-identical on real GPU (`parscan_partition_byte_identical_to_serial`) and
on the P100 harness (preds 0.0 vs serial).

Kernel v10 (P100, same session): official 3.12 s | **rs (parscan ON) 3.80 s** |
rs_serscan 4.05 s → parscan = **1.066×**. CUPTI per-kernel: mark 49 → **14.1 µs**,
scatter 34 → **10.6 µs** — the partition chain now BEATS official's analog sum.
1-tree walls: official 1.78 vs rs 1.30 (fixed-cost lead holds).

**Scorecard: gap 2.05× (campaign start) → 1.22×.** Remaining ~0.7 s, ranked:
host serialization bubbles (~10 ms/tree, single-sync/split deferral), pageable
small uploads (~1.2 ms/tree), pick kernel shape (~0.4 ms/tree).

### Round 7 — THE DEFERRAL WINS: single-sync/split lands 1.045×, gap 1.17×

The shape-preserving deferred grow loop (`LGBM_GROW_DEFER_SYNC`): ONE batched
[ranges, pick-export] read per split (was 2 blocking crossings), with split
i's host bookkeeping applied at the top of iteration i+1. The per-split device
chain runs without the host knowing the split point: partition → role kernel →
FIXED-GRID build-SMALLER (which=2 — first driver wiring of the T-04 kernel) →
which-aware fused subtract+co-scan (LEFT/RIGHT scalars role-selected on
device) → device-target par reduce. Every kernel variant identical to the
eager arm ⇒ byte-identical trees (proven real-GPU + P100 preds 0.0).

Kernel v11 (P100, same session): official 2.978 | rs (eager) 3.651 |
**rs_defer 3.494** = **1.045×**, preds 0.0, 100 trees. **Gap: 1.17× — the
first time under 1.2.** Drained grow 2032ms (partition bucket 392→175 after
round-6 parscan). This REVERSES T-11's 0.836× (2026-07-15): the old deferred
arm changed compute shape (build-LEFT, split scans, legacy kernel); keeping
the variants identical was the missing ingredient.

Default FLIPPED ON behind `deferred_scan_config_applies` (cuda-default fused
config only; hip's pargain default and any non-conforming config silently
keep the byte-identical eager loop; `LGBM_GROW_DEFER_SYNC=0` restores eager).
Round 8 confirms the flipped default (official / rs / rs_eager).

### Round 8 — default CONFIRMED: gap 1.13×

Kernel v12 (P100, same session): official 3.033 | **rs (deferred default
engaged) 3.422** | rs_eager (`LGBM_GROW_DEFER_SYNC=0`) 3.560. Preds
byte-identical (rs_eager 0.0; envelope 3.06e-5); 100 trees; 1-tree fixed-cost
lead holds (rs 1.24 vs official 1.70).

### P3 final scorecard (same-session P100 gaps)

| milestone | rs | official | gap |
|---|---|---|---|
| campaign start (§11) | 5.72 | 2.79 | 2.05× |
| inline handle default ON | 5.48 | 2.86 | 1.92× |
| PTX cache default ON | 4.05 | 3.16 | 1.28× |
| parscan partition twins | 3.80 | 3.12 | 1.22× |
| single-sync deferral default ON | **3.42** | 3.03 | **1.13×** |

Every lever byte-identical to the anchor; all shipped as defaults with kill
switches. Remaining ~0.4 s, ranked: pageable→pinned small uploads
(~1.2 ms/tree), pick kernel shape (~0.4 ms/tree), then re-profile — at 1.13×
the next decomposition needs fresh CUPTI evidence (the serialization tail has
shrunk under the deferral; unknown residual shape).

### Round 9 — pin-uploads WINS 1.050×: gap 1.12×

Kernel v13 (P100, same session): official 2.817 | **rs 3.149** | rs_pageable
(`CUBECL_CUDA_PIN_UPLOADS=0`) 3.308. Preds byte-identical (0.0); envelope
3.07e-5. CUPTI: the 2 283 pageable HtoD copies collapsed to 1 (the >4MB
resident-bins upload, deliberately direct); all small uploads now pinned at
3.5µs avg. Free-run (3.15) now BEATS drain (3.42) — overlap finally works.

**Updated scorecard: gap 2.05× → 1.92× → 1.28× → 1.22× → 1.13× → 1.12×.**
Cache-warm drained ledger: binning 898 + grow 1734 (build 697, setup 265,
scan 171, pick 169, partition 153, treesplit 83, tail 87) + score 167 +
grad 80. Device kernels/20t: child build 53.3ms (88.8µs — 2.7× faster than
official's ConstructHistogram), pinned HtoD 29.7ms/8 462, fix_compact 12.5,
subtract-scan 11.9, pick 10.1. The remaining ~0.33s is flat: ~10-30ms device
items + host-side binning/score parity — no single dominant lever remains;
next session should re-rank with these tables (candidates: fix_compact
geometry 20µs/call, pick kernel 17µs/call, info-upload count via the arena
revisited on TOP of pin-uploads, host binning micro-profile).
