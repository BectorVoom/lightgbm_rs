# Spike Wrap-Up Summary

**Dates:** 2026-06-17 (001–013) · 2026-06-21 (014a/b + p9v/qix/rdu/rsh) · 2026-06-25 (015–022, GPU build/scan kernel campaign)
**Spikes processed:** 22 (001–022) across three wrap-up sessions
**Feature areas:** CPU histogram build · GPU histogram kernel · GPU routing & quantization · Histogram/learning memory layout · GPU wide-shape attribution · GPU build fixed-point atomics · GPU split-scan occupancy
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
