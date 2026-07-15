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
