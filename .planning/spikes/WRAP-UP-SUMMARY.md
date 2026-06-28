# Spike Wrap-Up Summary

**Dates:** 2026-06-17 (001–013) · 2026-06-21 (014a/b + p9v/qix/rdu/rsh) · 2026-06-25 (015–022 build/scan kernel; 023/024 + 026–029 round-trip/partition; 030/031 build re-attribution; **022b/032/033 partition+within-scan close-out**) · 2026-06-26 (**034/035 GPU partition re-attribution + route-on-host**)
**Spikes processed:** 37 (001–035 incl. 014a/b, 022b; 025 superseded/not-built) across seven wrap-up sessions
**Feature areas:** CPU histogram build · GPU histogram kernel · GPU routing & quantization · Histogram/learning memory layout · GPU wide-shape attribution · GPU build fixed-point atomics · GPU split-scan occupancy · GPU scan round-trip & co-pack · GPU build bottleneck re-attribution · Partition (row-routing) memory-traffic
**Skill output:** `./.claude/skills/spike-findings-lightgbm_rs/`

## Processed Spikes

| # | Name | Type | Verdict | Feature Area |
|---|------|------|---------|--------------|
| 001 | gpu-cpu-crossover | standard | ✅ VALIDATED | GPU routing & quantization |
| 002 | lowrow-phase-ab | standard | ✅ VALIDATED | CPU histogram build |
| 003 | columnar-hist-build | standard | ✅ VALIDATED + SHIPPED | CPU histogram build |
| 004 | columnar-u8-bins | standard | ✅ VALIDATED + SHIPPED | CPU histogram build |
| 005 | feature-parallel-build | standard | ✅ VALIDATED + SHIPPED | CPU histogram build |
| 006 | gpu-u8-bins | standard | ❌ INVALIDATED | GPU histogram kernel |
| 007 | row-partitioned-histogram-build | standard | ✅ VALIDATED | GPU histogram kernel |
| 008 | 16bit-discretized-hist | standard | ❌ INVALIDATED (exact) | GPU routing & quantization |
| 009 | multifeature-per-cube | standard | ❌ INVALIDATED | GPU histogram kernel |
| 010 | histogram-pool-arena | standard | ✅ VALIDATED + SHIPPED | Histogram memory layout |
| 011 | parallel-build-scatter | standard | ❌ INVALIDATED (load-bearing) | Histogram memory layout |
| 012 | reuse-pool-across-trees | standard | ✅ VALIDATED + SHIPPED | Histogram memory layout |
| 013 | feature-splittable-arena | standard | ❌ INVALIDATED (sub-noise) | Learning-path allocation |
| 014a | coarse-phase-attribution | standard | ⚠️ PARTIAL (overturns framing) | GPU wide-shape attribution |
| 014b | gpu-launch-vs-compute-split | standard | ✅ VALIDATED (names the cost) | GPU wide-shape attribution |
| 015 | parallel-f32-resident-build | standard | ⚠️ PARTIAL (bottleneck located) | GPU build — fixed-point atomics |
| 016 | parallel-scan-reorder-parity | standard | ⚠️ PARTIAL (→ resolved by 022) | GPU split-scan occupancy |
| 017 | perwarp-lds-replication | standard | ✅ VALIDATED modest ~1.1× (not wired) | GPU build — fixed-point atomics |
| 018 | fixedpoint-int-atomics | standard | ✅ VALIDATED strong + SHIPPED | GPU build — fixed-point atomics |
| 019 | int-atomic-contention-regime | standard | ✅ VALIDATED (corrects 018) | GPU build — fixed-point atomics |
| 020 | perwarp-replication-on-u64 | standard | ⚠️ PARTIAL/null (don't wire) | GPU build — fixed-point atomics |
| 021 | scan-feature-per-lane-occupancy | standard | ✅ VALIDATED + SHIPPED | GPU split-scan occupancy |
| 022 | within-feature-parallel-scan-parity | standard | ✅ VALIDATED (gate resolved, ROI-gated) | GPU split-scan occupancy |
| 022b | within-feature-scan-perf-ab | standard | ✅ VALIDATED experiment (confirms DON'T WIRE) | GPU split-scan occupancy |
| 023 | post-021-roundtrip-attribution | measurement | ✅ VALIDATED (regime-split) | GPU scan round-trip & co-pack |
| 024 | batch-sibling-scans | standard | ✅ VALIDATED ~2× + WIRED phase 12 | GPU scan round-trip & co-pack |
| 026 | cubecl-cpu-partition-scan-scatter | standard | ⚠️ PARTIAL/NULL (bandwidth-bound) | Partition memory-traffic |
| 027 | fused-gather-partition | standard | ✅ VALIDATED + SHIPPED (CPU, 1.3–2.7×) | Partition memory-traffic |
| 028 | doublebuffer-partition | standard | ❌ INVALIDATED/NULL | Partition memory-traffic |
| 029 | gpu-narrow-upload-fuse | standard | ✅ VALIDATED + SHIPPED (ROCm, ~1.2–1.7×) | Partition memory-traffic |
| 030 | wide-build-roofline-reattribution | measurement | ✅ VALIDATED (uncoalesced-gather) | GPU build bottleneck re-attribution |
| 031 | crossfeature-gradhess-reuse | standard | ⛔ CLOSED by 030 (not built) | GPU build bottleneck re-attribution |
| 032 | partition-validation-fold | standard | ✅ VALIDATED + SHIPPED (CPU, ~1.14–1.41× U8) | Partition memory-traffic |
| 033 | partition-gather-prefetch | standard | ⚠️ PARTIAL — DON'T WIRE (ROI-gated) | Partition memory-traffic |

## Key Findings (001–009, the perf campaign)

- **The histogram BUILD is the bottleneck** (002): 63–90% of CPU train, 5.2× slower than
  C++ at low rows; split-scan/partition are near parity. Localized via per-phase A/B vs
  `lib_lightgbm` 4.6 `-DUSE_TIMETAG`.
- **Four stacked bit-exact CPU wins shipped:** once-per-leaf gather (003, −33/−39% build),
  fused-branchless build (003b, needs validation relocated upstream), narrow u8/u16 bins
  (004, large train −49%), feature-parallel ≥16384-row leaves (005, large −26%). Cumulative
  large ≈ −67%.
- **GPU build is atomic/latency-bound, not bandwidth-bound** (006): the CPU u8 win does NOT
  transfer (~0%). The one GPU lever is row-partitioning to ~8 wkgrps/CU (007, ~1.35×);
  multi-feature packing is null at matched occupancy (009). With the CPU multi-threaded
  (005), GPU loses at every tested size → GPU work is ROCm-parity maintenance, not speed.
- **GPU wins on wall-clock ≳1M rows vs single-thread** (001, crossover ≈700k) — but moves to
  millions vs the multi-threaded anchor. **int16 quantized hist is irreducibly approximate**
  (008, ~3e-4 floor ≫ gate) → opt-in mode only, never the exact path.

## Key Findings (010–013, the Vec<Vec> thread)

- **Shipped (~7% large, bit-exact):** flatten the `HistogramPool` buffers `Vec<Vec<f64>>` →
  one flat arena (010, ~4%), then reuse the pool across trees instead of per-tree alloc
  (012, ~3% more). Both internal/storage-only ⇒ bit-exact vs the C++ golden.
- **Rejected with evidence:** the parallel build's per-feature `Vec<Vec<f64>>` accumulators
  are load-bearing — scatter regressed 13–21% via false sharing (011); the per-tree
  `feature_splittable` bool matrix is 0.005–0.25%/tree, not worth a refactor (013).
- **Method rule:** the cold isolated microbench overstates the warm end-to-end win 3–7×
  (allocator amortizes fixed-size per-iteration reallocs) — always confirm with
  `bench_train`. Captured in `.planning/spikes/CONVENTIONS.md`.
- **Decision rule:** flatten a per-iteration `vec![template; n]` only when MB-scale and
  not a per-thread private accumulator; per-leaf row lists are already flat (DataPartition).
- **Sweep status:** the learning-leaf `Vec<Vec<T>>` surface is exhausted. 001–009 remain
  for a future wrap-up.

## Key Findings (014a/014b + p9v/qix/rdu/rsh — the GPU wide-shape thread, 2026-06-21)
- **The GPU histogram kernel is NOT the 1M×500 bottleneck.** 014a: at ≥100 feat the
  resident fusion folds build+fix+scan into one launch timed under "scan" (`build=0` is an
  artifact); the growth-loop timers cover <½ of train and the blind fraction GROWS with
  rows (55→69%). The kernel is ≤⅓ of wall-clock.
- **Whole-train BUDGET profiler** (014b, `LGBM_PHASE_PROF=1`, `phase_prof.rs`) named the
  rest: a redundant per-tree resident-bin re-upload (≈ the kernel) + per-train host setup.
  Confirmed GPU-specific via a 16× CPU-vs-GPU A/B.
- **Four shipped bit-exact levers** took 1M×500 iters=4 train **29.55→~9.5s (−68%)**:
  p9v (upload once/train, −32%), qix (native-width upload, ~5× upload bucket + ~4× host
  mem, generic `<B:Int>` kernels), rdu (cache-friendly `feature_infos`, ~8×), rsh
  (cache-friendly binning via transpose-scatter, ~2.3×).
- **Method lessons:** (1) measure before "fixing" a hypothesis — the "to_vec clones"
  overhead was a mis-attribution (8MB clone = sub-ms; the real cost was cache-hostile
  `feature_infos`); (2) row-major `Vec<Vec<f64>>` + per-feature column passes are
  cache-hostile — transpose to one contiguous row pass (byte-identical); (3) single-thread
  scatter into `num_features` L2-resident tails wins where spike-011's PARALLEL scatter lost.

## Key Findings (015–022 — the GPU build/scan KERNEL campaign, 2026-06-25)
- **The wide bottleneck is the atomic-bound histogram BUILD** (015): post-014, with the
  build-drain A/B (`LGBM_SCAN_DRAIN=1`) it is 86→92% of the scan-attributed wall and GROWS
  with rows; the scan round-trip is ≤14% and shrinking; array-hoist (~0.1%) and switch-to-f32
  (already done) are dead. Tooling (`LGBM_SCAN_PROF`/`DRAIN`) kept in-tree.
- **The build win SHIPPED: f32 → u64 fixed-point integer atomics** (018/019). On RDNA, f32
  `atomicAdd` is a CAS-retry loop (`ds_cmpst`) that serializes under contention; integer
  `ds_add_u64` is native single-instruction. ~1.3–1.7× in the heavy-load regime (wide
  root/large leaves), NULL at light load; **composes** with row-partition; **+3600× accuracy +
  deterministic** (order-independent integer adds → bit-exact across runs/P). `Atomic<i64>` is
  broken in cubecl-hip 0.10 → use `Atomic<u64>` two's-complement @ S=2^30.
- **Per-warp LDS replication is a NULL** (017 f32 ~1.1× not-wired; 020 u64) — wins only at
  P=16, **regresses ~0.90× at the production P=1 wide regime** (2× LDS halves occupancy; the
  u64 switch already took the contention win). Don't wire.
- **The split SCAN win SHIPPED: feature-per-lane occupancy** (021). The scan launched
  `CubeDim(1)` = one single-threaded cube/feature (~1/32 wave ALU util). Repack to one feature
  per LANE (`ABSOLUTE_POS` index + tail guard, `CubeDim(W)`, env `LGBM_SCAN_CUBEDIM` default
  W=64) → **bit-exact** (each feature still sequential), isolated scan **~3×**, e2e **~1.27×**
  (Amdahl-capped: the readback sync is still gated by the unchanged build).
- **Within-feature parallel scan is PARITY-SAFE but ROI-gated** (016/022). Host probes resolved
  the reorder risk: threshold stable; every `default_left` flip COSMETIC (max present-data leaf
  Δ = 0.0; the gain gap is linear in default-bin mass so only empty bins flip). A tie-aware
  argmax reproduces the same splits within ~1e-6. But post-021 the scan saturates the device at
  wide, so it helps only NARROW (the GPU's weakest regime) — **don't wire**; deferred 022b = the
  perf A/B. New method: the **host parity probe** (model the exact backend reorder order +
  classify cosmetic-vs-real by present-data impact) — now a 008/016/022 convention.
- **ROI reality (unchanged):** the spoofed 8-CU gfx1152 APU loses to the multi-threaded CPU
  anchor at every shape — this whole kernel campaign is **ROCm-parity-track maintenance**, valid
  for a real discrete gfx110x where the under-utilization removed is more wasteful.

## Shipped commits
- `d9cbae4` — spike 010 (flat arena)
- `5c8fa43` — spikes 012 (pool reuse) + 013 (feature_splittable not-worth-it)
- `c490905` — spike 011 (revert + load-bearing NOTE)
- 014 thread: `01e405d` p9v, `ff4a10b` qix, `bf467bd` rdu, `b917191` rsh (spikes:
  `fe79da3` 014a, `c3ab6fd` 014b)
- 015–022 thread: 018/019 u64 fixed-point build (live, Phase-11); `eaf4094` 021
  feature-per-lane scan (SHIPPED); `acf849c` 022 within-feature parity gate; spikes 015–020
  documented (017/020 replication evidence kept rocm-gated, not wired).

---

## Session 2026-06-25 (cont.) — spikes 023/024 + 026–029 (GPU scan round-trip + PARTITION arc)

**Processed:** 6 spikes → 2 new reference files.

| # | Name | Verdict | Feature area |
|---|------|---------|--------------|
| 023 | post-021-roundtrip-attribution | VALIDATED (measurement) | gpu-scan-roundtrip-copack |
| 024 | batch-sibling-scans | VALIDATED ~2× isolated, WIRED phase 12 | gpu-scan-roundtrip-copack |
| 026 | cubecl-cpu-partition-scan-scatter | PARTIAL/NULL | partition-memory-traffic |
| 027 | fused-gather-partition | VALIDATED + SHIPPED (CPU, quick-260625-hw2) | partition-memory-traffic |
| 028 | doublebuffer-partition | INVALIDATED/NULL | partition-memory-traffic |
| 029 | gpu-narrow-upload-fuse | VALIDATED + SHIPPED (ROCm, quick-260625-j1l) | partition-memory-traffic |

### Key findings
- **Partition is memory-bandwidth-bound on shared DDR5** (026) — parallelizing it (rayon OR
  cubecl-cpu) is NULL at scale; this reframes the reverted ia0 rayon (wall = DRAM bandwidth, not
  build contention). The lever is to CUT TRAFFIC.
- **Fuse the per-leaf bin gather + ¼-width u8 route scratch** (027) — 1.3–2.7× CPU, bit-exact,
  biggest ~2.3× at U8; SHIPPED behind `prefers_host_partition()`. The ~29% tall-narrow partition
  residual was the #1 remaining CPU-vs-C++ gap.
- **Narrow the GPU per-split upload u32→native-width** (029) — ~1.2–1.7× rocm, bit-exact on the
  GPU; SHIPPED via a generic-over-Int kernel + additive `data_partition_native`. Disproved the
  "shared-DDR5 APU transfer is free" assumption — `create_from_slice` still moves the bytes.
- **Two clean NULLs with root causes:** parallelize partition (026), double-buffer to drop the
  copy-back (028 — copy-back is only 2–7% of the fused op; C++ copies back too).
- **GPU scan round-trip regime-split** (023) + **sibling-scan co-pack** (024, ~2× isolated,
  bit-exact, WIRED phase 12 behind `LGBM_SIBLING_COPACK`).

### Shipped commits
- 027: `8eb6c9e` (fused host split) + `f413e1d` (prefers_host_partition) — quick-260625-hw2.
- 029: `4fe9025` (generic kernel) + `3b79e69` (data_partition_native + wire) + `9ab8cb6` (U8/U16
  parity cells) — quick-260625-j1l.
- 024: wired in phase 12 (`LGBM_SIBLING_COPACK`).

---

## Session 2026-06-25 (cont.) — Build bottleneck RE-ATTRIBUTION (030/031)

**Spikes processed:** 2 (030 VALIDATED measurement, 031 CLOSED-by-030/not-built)
**Feature area:** GPU build — bottleneck re-attribution (new reference
`references/gpu-build-bottleneck-reattribution.md`)
**Idea:** "attack the learning speed of bottleneck in gpu" — the manifest's own rule is
"re-profile after every build change"; the wide-build attribution (015, "atomic-bound") predated
the u64 ship and was never re-run.

### Key findings
- **The wide GPU build is UNCOALESCED-BIN-GATHER-latency-bound (030, 86–95%)** — NOT atomic-bound
  (u64 made the atomic free: NOATOMIC ≈ FULL ⇒ spike-015's "atomic-bound ~820 Mr/s" is STALE) and
  NOT grad/hess-bandwidth-bound (8–14%). Proof: `COAL_BIN` reads the same 500 MB array / same
  bytes SEQUENTIALLY and runs 8–20× faster; effective 4.5–10 GB/s ≪ DDR5 peak = a latency stall.
- **The honesty caveat that capped the ROI (REAL_ORDER):** a random `leaf_rows` probe overstates
  the penalty 5–10×. LightGBM's STABLE partition gives monotone-increasing `leaf_rows`, already at
  **~70% of the coalesced ceiling** (4093/3405 vs 5636/4914 Mr/s). Residual coalescing headroom is
  only **~1.4×** — and read-once-unamortizable (same wall as 028). ⇒ the build is effectively tuned
  on the APU.
- **031 closed without building it:** original premise (grad/hess reuse) invalidated by 030's
  CONST_GH; the redirect (coalesce the bin read) is marginal and unamortizable per 030's own data.
  Reopens only on **discrete gfx110x** — re-run `examples/spike030_build_roofline_ab.rs` there.
- **New convention:** "remove-the-suspect" re-attribution (delete one cost per variant, pair
  complementary deletions, report Mr/s, model the REAL access order). Added to CONVENTIONS.md.

### Commits
- 030: `ccd285f` (VALIDATED — probe + README + MANIFEST).
- 031: `9eee58c` (CLOSED-by-030 + CONVENTIONS "remove-the-suspect" pattern).

---

## Session 2026-06-25 (cont.) — Partition + within-scan close-out (022b/032/033)

**Spikes processed:** 3 (032 VALIDATED+SHIPPED, 033 PARTIAL/don't-wire, 022b VALIDATED experiment)
**Feature areas:** Partition (row-routing) memory-traffic (032/033 → `references/partition-memory-traffic.md`);
GPU split-scan occupancy (022b → `references/gpu-split-scan-occupancy.md`)
**Idea:** "attack the learning speed of bottleneck in gpu" → user redirected to the data partition
(spike-027 follow-ons), then to the deferred prefetch lever; 022b folded in (was unprocessed).

### Key findings
- **032 — audit the shipped wiring, not just the spike that validated it (SHIPPED, quick-260625-qn9).**
  Reading the live `split_fused_host` found it did TWO random gathers (a standalone per-row
  validation loop + pass-1's route gather), not the ONE spike-027 measured — the 2nd re-misses
  cache at scale = exactly the traffic 026→027 cut. Folded the range-check into pass-1 (gather `b`
  once, check before any write ⇒ unmutated `indices` + same lowest-index error = bit-exact on
  success AND error). **~1.14–1.41× at production U8 ≥1M rows, up to ~1.8× U32**, 3 restarts parity
  OK; gate green (lgbm-treelearner 77/0 + oracle `raw_bin_train_parity` 2/0 vs lib_lightgbm 4.6).
- **033 — prefetch is ROI-gated, DON'T WIRE.** `_mm_prefetch`-ahead of the random gather hides
  miss-latency only when the bin column ≫ LLC: **~2–3× whole-op at 4M×U32 (bestD=128)**, but the
  production-default **U8** width is dense enough that even 4M rows barely exceeds cache ⇒ ~1.1× at
  a root split only, null-to-slower everywhere else. x86-only intrinsic. After 032 there's little
  gather latency left to hide at U8. **Two reusable landmines → CONVENTIONS:** (a) don't refactor
  `.bin()` per-row matches into a typed-`&[T]` loop — it auto-vectorizes to a slow AVX gather,
  **1.5–2× SLOWER** than scalar; (b) prefetch only pays when the gathered array ≫ LLC.
- **022b — the within-feature cooperative scan is perf-disproven (DON'T WIRE).** vs the SHIPPED 021
  (cd64 K1), cooperation wins only NARROW (≤256 feat, 6×→2.2×) and is WASH-to-regression at the
  WIDE F=512 production shape once the cd256 occupancy confound is removed. Confirms 022's parity
  finding on the real kernel (argmax mism=0, gainrel ≤9e-15). Closes the deferred ROI question.
- **Net: the CPU host partition is DONE.** 027 (fuse) + 032 (one-gather) shipped; 026/028/033/ia0
  all NULL/ROI-gated. No remaining positive-ROI partition lever on this hardware; the GPU-track
  on-device partition (host partition ~23% of GPU-train, never moved on-device) is the only
  un-attacked structural cost — ROCm/discrete-GPU track only.

### Commits
- 032 (spike): `905e31e`; 032 (wire): `6b6fb09` + `cc673d9` (quick-260625-qn9).
- 033 (spike): `f8dc46d`.
- CONVENTIONS: `1f3563f` (audit-the-wiring) + the 033 autovectorization/prefetch lessons (this wrap).

## Session 7 (2026-06-26) — 034/035: re-attribute-then-act, the GPU partition routing win

- **034 — re-profile after the wires; the bottleneck MOVED a 4th time (measurement).** With co-pack
  (024) + narrow-upload (029) SHIPPED since the last full attribution (023), re-ran the whole-train
  BUDGET + per-tree COUNTS + co-pack ON/OFF A/B (2 restarts, sign-stable). **Launch-bound regime
  (small/med/large): the scan-sync floor is CLOSED** — co-pack confirmed live (scan_resident 59→30
  syncs/tree, scan now 3.0% med / 7.0% large) — and the **device `data_partition_native` round-trip
  is the NEW #1 reclaimable phase, 38% medium / 30% large**. **Wide/compute-bound: UNCHANGED**,
  build-dominated ~91% (neither lever targets it, as predicted). Also corrected a tooling gap: the
  `LGBM_SCAN_DRAIN` build-drain needs `LGBM_SCAN_PROF=1` AND lived only on the single-leaf scan fn
  (Phase-12 routed the default through the co-pack siblings fn) → re-wired onto the siblings fn
  (quick-260625-tw1), GPU-verified build_drain 0%→97.5/98.6%.
- **035 — route the rocm partition on the HOST by default (VALIDATED + SHIPPED, quick-260626-a6t).**
  The fix 034 pointed at. Both partition paths land in host `indices_` and the resident build reads
  host indices either way ⇒ the device round-trip is **pure overhead on shared DDR5, no index
  re-upload** (the intuitive tradeoff is moot). Flip `RocmBackend::prefers_host_partition()` to
  default-ON (off-switch `LGBM_ROCM_HOST_PARTITION=0`) → run the SHIPPED 027 host fused path.
  **~1.18–1.23× launch-bound (medium/large), wash wide** (no regression), 2 restarts sign-stable.
  **Parity (def-f8u-01):** NOT a bit-exact swap — host-vs-device max divergence 1.907e-6 = IDENTICAL
  to device-vs-device run-to-run noise (3-arm test); valid gate = anchor-pinned hip tests, not
  GPU-vs-GPU. **The rare GPU lever that wins on the 8-CU APU itself** (most levers in this campaign
  are discrete-only deferrals); larger on discrete gfx110x (device round-trip crosses PCIe).
- **Gated by a debug fix (`8aed100`).** The anchor-pinned hip parity tests were RED on master
  (`subtract_resident: smaller slot is empty`) — a latent FUSED-path bug: Phase-12 co-pack deferred
  the smaller child's scan past `subtract_resident`, but on the fused path that scan IS the smaller
  histogram build+store, so subtract ran before it existed. Un-deferred for the fused case only
  (co-pack never touched the fused path). Method lesson reinforced: **re-profile after every wire,
  and a "validated-not-wired" spike can ship same-session once you clear the gating defect.**

### Commits (session 7)
- 034 (spike): `34671e7` + CONVENTIONS `9acac58` (the "a WIRE-pending verdict may already be shipped" inverse rule).
- 035 (spike): `73a2328`; 035 (wire): `da3032f` + docs `1815325` (quick-260626-a6t).
- debug subtract_resident: fix `8aed100` + archive `26bd150`.
- LGBM_SCAN_DRAIN re-wire: `128a4c2` + docs `872f5de` (quick-260625-tw1).

---

## Session 8 wrap-up — 2026-06-26 (spike 036, the branch-divergence gate)

**Spikes processed:** 1 (036). **Skill output:** new reference `references/gpu-branch-divergence.md`.

**Idea:** "Optimize conditional branching in GPU kernels." Run as a GATE (user chose gate-first):
is any divergent branch on a hot path AND is divergence even measurable on the spoofed APU?

- **036 — branch-divergence inventory + critical-path/measurability gate (PARTIAL).** Two
  findings of opposite sign.
  - **Measurability = PASS, and it overturns a prior.** A controlled-divergence LADDER (4 arms,
    IDENTICAL total work, only the intra-wave loop-trip-count distribution differs:
    UNIFORM/DIV2/DIV4/DIV32) scaled **1.00 / 1.89–1.95× / 3.62–3.84× / 25.6–29.3×** (near-ideal
    1:2:4:32, 2 restarts, every rung p25 ≫ UNIFORM p75). **Wavefront lockstep-masking is FAITHFUL
    on the spoofed 8-CU gfx1152 APU** — divergence is the ONE GPU micro-arch effect that survives
    the spoof and is cleanly sign-measurable (it's a scheduler property, not CU-count/memory-bound,
    the confounded axes). A reusable carve-out from the "APU numbers are unmeasurable" caveat.
  - **Critical-path = WEAK ⇒ DON'T-CHASE.** The kernels are ALREADY fully branchless
    (`select`-everywhere — cubecl-cpu's MLIR lowering rejects in-loop conditional-store `if`
    chains, so the divergence-elimination transform shipped as a side effect). The only live
    data-dependent cross-lane divergence is the split-scan **loop-trip-count** imbalance — on the
    **3–7% scan** phase (034), **zero** at the production all-256-bin cardinality, real only on
    mixed-cardinality feature sets (honest e2e ceiling ≪1%). The dominant wide **build** is
    uniform/divergence-free by construction (030); partition is branchless + host-routed (035);
    the only heavily-divergent kernel (plane-atomic) is the p93 NULL/dead path.
  - **Recommendation:** do not chase branch divergence as a general lever. 037 (scan trip-count) =
    bounded mixed-cardinality curiosity only; 038 (break-vs-select) = likely don't-build (`done`
    is intra-lane predication ⇒ no wave-max-trip reduction unless early-exit is correlated; + must
    fork hip-only off the bit-exact `split_scan_body` anchor). The 030/031/033 bounded-don't-chase
    shape.

**Method lesson reinforced:** *gate measurability AND critical-path before optimizing.* The
intuitive optimization ("convert branches to branchless") was already done; the intuitive blocker
("APU can't measure warp effects") was false. Both halves had to be checked empirically — the
controlled ladder (identical work, varied distribution) is the reusable instrument.

### Commits (session 8)
- 036 (spike): `359428d` (README inventory + ladder, harness, MANIFEST row).

---

## Session 9 wrap-up — 2026-06-26 (spikes 037–040, the GPU-kernel-AUTOTUNING arc)

**Spikes processed:** 4 (all VALIDATED). **Skill output:** new reference
`references/gpu-kernel-autotuning.md`.

**Idea:** "optimise gpu kernel by autotune" (the CubeCL `cubecl::tune` feature, per
`cubecl_manual/.../12_autotuning.md`) — replace the hand-tuned/env-var launch-config
heuristics (row-partition `P`, scan `CubeDim`) with a measured, cached, self-calibrating
runtime tuner. NOTE: these 037/038 numbers are the AUTOTUNE track, distinct from the
deferred divergence curiosities 036 happened to label "037/038".

| # | Name | Type | Verdict | Feature area |
|---|------|------|---------|--------------|
| 037 | autotune-hip-feasibility | standard | ✅ VALIDATED | GPU kernel autotuning |
| 038 | autotune-inplace-correctness | standard | ✅ VALIDATED | GPU kernel autotuning |
| 039 | autotune-key-cache-thrash | standard | ✅ VALIDATED | GPU kernel autotuning |
| 040 | autotune-vs-heuristic | comparison | ✅ VALIDATED (autotune wins) | GPU kernel autotuning |

### Key findings
- **037 — feasibility (kill Q1): autotune works END-TO-END on cubecl-hip 0.10.** Compile ✓,
  run-on-device ✓, benchmark-both ✓, pick-winner ✓, in-proc cache hit ~6µs (~78,000× vs
  490ms cold-tune), **persistent disk cache across processes** (~828µs cold-with-cache,
  `target/autotune/0.10.0/rocm_0/*.json.log`). It independently re-derived spike-007's P=16.
  **The `cubecl_manual` doc is WRONG on its 3 load-bearing points — code from the SOURCE:**
  (1) the key-gen closure returns the `AutotuneKey` (not a String), (2) `execute`'s 1st arg
  is the cache-namespace ID (not the key — the key is generated internally), (3) the key
  needs `serde::{Serialize,DeserializeOwned}` under `std_io` (always on linux). Added `serde`
  as a dev-dep (examples-only).
- **038 — correctness (kill Q2): accumulating kernels corrupt 27× under
  `CloneInputGenerator`.** `Handle::clone` is a ref-count bump, so every benchmark rep
  `fetch_add`s into the caller's REAL `out` (the 27 = the whole sample budget, not a +1
  bias). **Fix = a fresh-output `InputGenerator`** (the winner's final run uses the original
  inputs ⇒ real `out` touched once ⇒ `rel_err 0` by grad-conservation). Classify kernels:
  OVERWRITE safe-as-is / ACCUMULATE needs fresh-out / in-place-RMW needs deep-copy (partition
  is host-routed on rocm, 035). GAT gotcha: spell `generate<'a>`'s return as
  `<Vec<Handle> as TuneInputs>::At<'a>` or E0195.
- **039 — keying granularity: exact `rows` is a tuning STORM.** 25/25 tree nodes cold, ZERO
  reuse, 975ms for ONE shallow tree. `log2(rows)` bucketing → 5 keys, 20/25 free, ~3× faster,
  AND it keeps the per-regime P16↔P1 crossover (FIXED feats-only is cheaper but mis-applies
  the root's variant to small leaves). The variant choice tracks the occupancy REGIME, not
  the exact count. Surprise: EXACT's P16/P1 split is itself run-to-run noisy (small leaves sit
  near the selection tie) — a 2nd argument against over-fine keying.
- **040 — comparison: autotune BEATS the shipped heuristic ~10% (not the predicted wash).**
  `row_partition_count(50,n)` resolves `target_cubes = 8 CU × 8 = 64`, `MIN_LEAF = 256k`,
  `clamp(64/50) = 1` ⇒ **P=1 for every leaf at the production 50-feature width** (the 8-CU
  correction over-corrected from the phantom-96-CU P≈16, effectively disabling
  row-partitioning). Rigorous P-sweep {1,4,8,16,32} (3 restarts, sign-stable): **P=1 is the
  SLOWEST point at every size**; autotune picks P∈{4,8,16,32} and wins **2–16% (typ ~10%),
  never loses**. **Surfaced a latent production mis-tuning** → recalibrate `row_partition_count`
  OR adopt autotune (the robust + portability answer).

### Net signal
- Autotuning the rocm histogram kernel is **feasible and worthwhile** — all blocking risks
  resolved. Wire behind a default-false backend discriminator, key on `(log2(rows), feats,
  bins)`, use a fresh-output InputGenerator, read the winner from the persisted cache.
- **Fix or replace `row_partition_count`** (the ~10% latent under-partition at the production
  width). Autotune is the robust + portability answer (self-calibrates on discrete gfx110x /
  NVIDIA with zero re-tuning). Honest bound: the ~10% is on the spoofed-APU GPU build, which
  the 16-core CPU beats end-to-end here — the durable deliverable is the **method
  (measure-don't-model)** + portability. CPU f64 anchor untouched (example-only + dev-dep).

### Commits (session 9)
- 037: `a94c8f3`; 038: `e33a04c`; 039: `9731a17`; 040: `1e397be`; CONVENTIONS: `8803d9f`.

---

## Session 10 — 2026-06-28 (Vector<P,N> frontier + first real-discrete-CUDA attribution)

**Spikes processed:** 8 (041, 042, 043, 044, 045, 046, 048, 049; 047 skipped)
**New references:** `vector-simd-histogram-kernels.md`, `cuda-discrete-gpu-bottleneck.md`
**Shipped:** quick-260628-f57 (the metric-eval fix)

| # | Name | Verdict | Feature Area |
|---|------|---------|--------------|
| 041 | line-feasibility-subtract | ✅ WON + SHIPPED (agx) | Vector<P,N> SIMD |
| 042 | line-scan-pair-read | ❌ NULL | Vector<P,N> SIMD |
| 043 | line-build-gradhess-input | ❌ NULL + wide regression | Vector<P,N> SIMD |
| 044 | line-fixcompact-dequant | ⚠ feasible, ROI-bounded DON'T-WIRE | Vector<P,N> SIMD |
| 045 | coalesced-build-vector | ❌ INVALIDATED (closes frontier) | Vector<P,N> SIMD |
| 046 | python-path-phase-prof | ✅ VALIDATED (enabler) | Discrete-CUDA attribution |
| 048 | kaggle-cuda-confirm | ✅ VALIDATED + SHIPPED fix | Discrete-CUDA attribution |
| 049 | in-learner-other-attribution | ✅ VALIDATED (dead-end) | Discrete-CUDA attribution |

### Key findings
- **Vector<P,N> frontier CLOSED.** Rule: vectorize only memory-bound kernels where the
  vectorized op covers the bottleneck. subtract WON (shipped agx); scan/build NULL
  (dependent chain / permuted gather); dequant ROI-bounded; coalesced-rewrite invalidated.
- **First real-discrete-CUDA attribution (Kaggle, the campaign was APU-only).** lgb_rs CUDA
  ~5–6× official at 500k×50. Root cause #1 (26%, SHIPPED fix quick-260628-f57): host
  per-iter training-metric eval (`booster.rs:1291 || valid.is_none()`, divergent from C++).
  Confirmed metric 4489ms→0 on real hardware, parity-neutral.
- **Post-fix wall map:** GPU hist phases 53% (architectural on-device-learner long-pole),
  Python marshalling 25% (UNATTRIBUTED — next easy win), in_learner_other 15% (diffuse
  DEAD END), rest 7%, metric 0%.
- **Refuted on real hardware:** per-leaf sync-floor (286ms/1.7%), route-narrow-to-CPU
  (CUDA beats CPU on Kaggle's few vCPUs), resident_reset (0.3ms/100 trees).
- **Reusable:** spike-046 `phase_prof::dump("train")` hook makes the Python path
  observable; Kaggle CLI harness in `spikes/046-*/`.

### Open frontier (not yet spiked)
- Attribute the Python marshalling ~25% (numpy→corpus pyo3) — likely the next easy win.
- The architectural on-device monolithic tree-learner (the 53% GPU-phases long-pole) —
  milestone-sized.

---

## Session 11 — 2026-06-28 (Python-side binning attribution + parallel-binning fix)

**Spikes processed:** 1 (050)
**Reference updated:** `cuda-discrete-gpu-bottleneck.md` (added the Python-side binning recipe)
**Shipped:** spike-050 feature-parallel binning (in-spike, bit-exact)

| # | Name | Verdict | Feature Area |
|---|------|---------|--------------|
| 050 | python-marshalling-binning | ✅ VALIDATED + SHIPPED | Discrete-CUDA attribution |

### Key findings
- spike-049's "Python marshalling ~25%" is actually **single-threaded raw→bin BINNING**
  (624ms serial @500k×50); the numpy→`Vec<Vec<f64>>` marshalling is only **43ms** (a
  non-issue). The binning was hidden as `binning=0` because `train_raw`'s bin step was
  never `BINNING_NS`-wrapped (now fixed).
- **SHIPPED feature-parallel binning** (`into_par_iter` over features) — 6.5× (624→96ms
  @16 cores), bit-exact vs the C++ golden (`raw_bin_train_matches_cpp_golden`).
  `LGBM_PAR_BIN=0` serial gate. Reusable lesson: match C++'s OpenMP-over-features for any
  per-feature host loop.

### Campaign status
Two of the three non-kernel CUDA-wall chunks now CLOSED: metric eval 26%→0
(quick-260628-f57), binning 25%→6.5× (spike-050). `in_learner_other` 15% is a diffuse
dead-end (spike-049). **The only major lever left is the GPU histogram phases (53%) —
the architectural on-device monolithic tree-learner (milestone-sized).**
