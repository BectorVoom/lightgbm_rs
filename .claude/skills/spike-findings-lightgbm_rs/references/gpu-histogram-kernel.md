# GPU (ROCm) Histogram Kernel — what moves the needle and what doesn't

Implementation blueprint from spikes 006, 007, 009 on the gfx1100. Three kernel levers
probed in isolation; only ONE is real. The throughline: the GPU build is **atomic-
contention / scattered-read-latency bound**, NOT bandwidth bound.

## Requirements

- Baseline for GPU kernel parity/perf = **AMD's ROCm fork `LightGBM-release-4.6.0.99/`**
  (hipified CUDA), NOT mainline `LightGBM/`. See
  `.planning/notes/cubecl-vs-rocm-histogram-kernel-comparison.md`.
- The CPU f64 anchor is the bit-exact merge gate and stays untouched; the GPU ~1e-6
  contract at large shapes is a separate, still-open gate.
- **Probe in an isolated micro-bench before plumbing** a multi-kernel change (this
  discipline killed two duds cheaply). Report within-round ratios across 2–3 process
  restarts; always include a correctness-vs-baseline column.

## The one lever that works

**Row-partitioning the LDS build (spike 007) — the LightGBM `grid_dim_y` analog.** The
production `construct_leaf_hist_resident_lds_kernel` launches `CubeCount=(num_features,1)`
— one workgroup per feature = ~50 cubes on a 96-CU GPU = **starved**. Split each feature's
rows across P cubes (`CubeCount=(FEATS,P)`; cube `(f,p)` strides rows into its own LDS
sub-hist, atomic-merges to the feature's global slot; P=1 is byte-identical to production):

| P | cubes | wkgrps/CU | speedup vs P=1 |
|---|-------|-----------|----------------|
| 1 | 50 | ~0.5 | 1.00× (starves the GPU) |
| **16** | **800** | **~8** | **1.30–1.39× (consistent, 6/6 rounds)** |
| 32 | 1600 | ~16 | ~1.0× (over-partitioned — regresses) |

**Tune to ~8 workgroups/CU, don't maximize:** `P ≈ clamp(8 × CU_count / num_features, 1, …)`.
Gate on leaf size + `num_features` so it only engages when features alone under-fill the GPU.
⚠️ **Parity interaction:** P≥2 raises GPU-vs-P=1 f32 divergence ~4e-7 → ~2e-5 rel (each
partition is an independent f32 partial-sum tree). CPU f64 anchor untouched; document the
GPU residual per the 04-ROCM-GAPS pattern.

## What to Avoid (two evidenced nulls)

- **u8 device bins (spike 006) — ~0%.** The CPU u8 win (L2 density) does NOT transfer:
  reading 1 byte vs 4 from a *scattered/uncoalesced* location is the same cache-line
  transaction. The kernel runs ~234 Mreads/s (≪ TB/s ceiling) → latency/atomic-bound, not
  bandwidth-bound. (Bonus proven: `Array<u8>` compiles+runs on HIP, cubecl 0.10.)
- **Multi-feature-per-cube packing (spike 009) — null at matched occupancy** (0.91–1.00×).
  Packing G features amortizes per-cube overhead but divides cube count by G, fighting the
  occupancy that row-partitioning supplies. At matched ~768 cubes there's nothing left to
  amortize, and the packed kernel's extra LDS/loops slightly regress. **Keep one-cube-per-
  feature × P.** LightGBM needs column-packing only because its 2D tile reaches occupancy a
  different way; our P lever reaches it directly.

## Constraints / ROI context

- The win is **modest (~1.35×) and occupancy-bounded** — even at P=16, ~820 Mreads/s is
  far below the bandwidth ceiling, so the kernel is *still* atomic-contention bound.
  Residual gains need attacking intra-cube LDS atomic contention directly (register
  row-batching, multiple LDS sub-hist replicas) — uncertain, larger effort.
- **Crucial ROI caveat:** with the CPU now multi-threaded (spike 005), the GPU loses to it
  at every tested size (200k: GPU 3.24s vs CPU 1.1s). GPU kernel work is **ROCm-parity-track
  maintenance, not an overall-fastest win** — weigh accordingly before a full phase.

## Harnesses (in-crate, `--features rocm`)

`gpu_row_partition.rs` (P-sweep), `gpu_bin_width.rs` (u8 A/B + Array<u8>-on-HIP proof),
`gpu_multifeature.rs` (matched-occupancy packed-vs-per-feature).

## Origin

Spikes 007 (VALIDATED), 006 + 009 (INVALIDATED). Sources in `sources/006-*`,
`sources/007-*`, `sources/009-*`.
