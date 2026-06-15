================================================================================
RESOLUTION (2026-06-15) — all gaps closed. Phase-09 + spikes 007/008/009.
================================================================================

  Gap / lever                         Status
  ----------------------------------  -------------------------------------------
  ★ Row-partitioning (grid_dim_y)     ✅ SHIPPED (phase-09, spike-007). ~1.25× at
                                         P=16; gated to large leaves; P=1 below =
                                         byte-identical. On master.
  Register row-batching (NUM_DATA_…)  ⊘ NULL (phase-09). K4/K1=0.89–0.98× at P=16:
                                         at saturating occupancy the bottleneck is
                                         LDS atomic contention, not load latency.
                                         Keep K=1.
  ★ 16-bit discretized histogram      ❌ INVALIDATED for exact parity (spike-008).
                                         Even full int16 drifts ~3.2e-4 rel (~30×
                                         over the 1e-5 gate); it's use_quantized_grad
                                         (default FALSE), an APPROXIMATE mode. Only
                                         viable as a separate opt-in mode = product
                                         decision, never a drop-in. Probe:
                                         quant_parity_probe.rs.
  Multi-feature-per-cube packing      ❌ INVALIDATED (spike-009). Null/regression
   ("wastes a CU on small-bin feats")   (0.91–1.00×) at MATCHED occupancy — row-
                                         partitioning already supplies the cubes, so
                                         packing has nothing to amortize. Keep one-
                                         cube-per-feature × P.
  Global-mem fallback (>256 bins)     ◷ Already have a naive fallback; not a perf
                                         gap for the in-scope ≤256-bin features.
  Privatization / merge / subtract /  ✓ Already at parity (pre-existing).
   FixHist / precision

  Net: the ONE real lever (occupancy via row-partitioning) shipped; the other three
  were measured and correctly rejected (two perf nulls + one parity-incompatible
  approximate mode). ROI note: GPU is ROCm-parity-track — the multi-threaded CPU
  (spike-005) still wins at every tested size. See .planning/spikes/00{7,8,9}/.
================================================================================

CubeCL kernel vs LightGBM's ROCm histogram kernel (AMD fork 4.6.0.99)

  Both run the same algorithm (privatized sub-histogram → atomic-merge to global),
  but the work decomposition is fundamentally different, and that's where the perf
  gap lives.

  Launch geometry — the decisive difference

  LightGBM ROCm (CalcConstructHistogramKernelDim,
  cuda_histogram_constructor.cpp:147):
  block_dim_x = max_num_column_per_partition      // FEATURES on x
  block_dim_y = 504 / max_num_column_per_partition // ROWS on y
  (NUM_THREADS_PER_BLOCK=504)
  grid_dim_x  = num_feature_partitions             // feature groups across blocks
  grid_dim_y  = max(min_grid_dim_y, ceil(ceil(N/400)/block_dim_y))   // ← ROWS
  SPLIT ACROSS BLOCKS
  - A block holds a 2D tile: many feature-columns × many rows, building one shared
  sub-hist covering all its columns' bins at once.
  - grid_dim_y partitions the rows — many blocks each fold a row-slice of the same
  features and atomicAdd_system-merge partial sums into global.
  NUM_DATA_PER_THREAD=400 rows folded in registers first.
  - Net: a big leaf saturates the whole GPU.

  CubeCL (construct_leaf_hist_resident_lds_kernel, histogram.rs:832):
  CubeCount = num_features          // ONE cube per feature
  CubeDim   = 256                   // 256 lanes stride ALL the leaf's rows
  - One feature = one workgroup = one CU. No row-partitioning. A 1M-row leaf is
  built by 256 lanes on a single CU while the other ~95 CUs on gfx1100 sit idle for
  that feature.

  Side-by-side

  Dimension: Feature → work
  LightGBM ROCm (AMD): columns packed per block (block_dim_x)
  CubeCL (ours): one cube per feature
  Gap: ours wastes a CU on small-bin features                
  ──────────────────────────────────────── 
  Dimension: Rows → work
  LightGBM ROCm (AMD): split across grid_dim_y blocks + 400/thread in registers
  CubeCL (ours): one cube strides all rows
  Gap: ★ ours can't saturate the GPU on big leaves
  ────────────────────────────────────────
  Dimension: Privatization
  LightGBM ROCm (AMD): __shared__ sub-hist, atomicAdd_block  
  CubeCL (ours): LDS SharedMemory, fetch_add
  Gap: same idea ✓
  ────────────────────────────────────────
  Dimension: Global merge                  
  LightGBM ROCm (AMD): atomicAdd_system partial sums
  CubeCL (ours): one fetch_add per cell per cube 
  Gap: same idea ✓
  ──────────────────────────────────────── 
  Dimension: Precision
  LightGBM ROCm (AMD): gpu_use_dp → f32 or f64 hist
  CubeCL (ours): f32 (→f64 widen)          
  Gap: parallel designs ✓
  ────────────────────────────────────────
  Dimension: 16-bit discretized
  LightGBM ROCm (AMD): int16_t shared_hist + gradient discretizer
  CubeCL (ours): none
  Gap: ★ halves atomic width + LDS
  ────────────────────────────────────────
  Dimension: Register row-batching
  LightGBM ROCm (AMD): NUM_DATA_PER_THREAD=400
  CubeCL (ours): none (1 row/iter)
  Gap: ours hits LDS atomics far more often
  ────────────────────────────────────────
  Dimension: Global-mem fallback
  LightGBM ROCm (AMD): yes (when sub-hist won't fit)
  CubeCL (ours): 256-bin cap → naive path
  Gap: partial
  ────────────────────────────────────────
  Dimension: Subtraction / FixHist
  LightGBM ROCm (AMD): GPU kernels
  CubeCL (ours): subtract.rs + on-GPU fix ✓
  Gap: at parity ✓

  The three optimization levers, in priority order

  1. ★ Row-partitioning (the grid_dim_y analog). Biggest win. Split a feature's
  rows across multiple cubes, each building a partial LDS sub-hist + atomic-merging
  to global. Directly fixes the one-CU-per-feature ceiling — and it re-frames your
  spike-006 "atomic/latency-bound" finding: that latency is partly starved 
  occupancy, not just atomic cost. f32-atomic-safe, stays in the ~1e-6 gate.
  2. Register row-batching (NUM_DATA_PER_THREAD). Each lane folds K rows into a
  register before touching LDS — cuts LDS-atomic frequency ~K×. Small, low-risk
  change that compounds with #1.
  3. 16-bit discretized histogram path. Largest structural change; halves atomic
  width + LDS footprint. Interacts with the f32/f64 parity contract — needs its own
  parity analysis before committing.
