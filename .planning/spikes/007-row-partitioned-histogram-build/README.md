---
spike: 007
name: row-partitioned-histogram-build
type: standard
validates: "Given a large leaf where the LDS build launches only num_features cubes (one CU each), when each feature's rows are split across P cubes (each a private LDS sub-hist → atomic-merge), then wall-clock drops as occupancy rises, matching the P=1 build to f32-atomic noise"
verdict: VALIDATED
related: [006, 005, 001]
tags: [performance, gpu, rocm, histogram, occupancy, atomics, row-partition]
---

# Spike 007: Row-partitioned histogram build (the grid_dim_y analog)

## What This Validates

The wired `construct_leaf_hist_resident_lds_kernel` launches `CubeCount = (num_features, 1)`
— **one workgroup per feature**. On a 1M-row / 50-feature leaf that is only 50 workgroups on a
96-CU gfx1100, which wants several resident workgroups per CU to hide the scattered-read /
atomic latency spike 006 measured (234 Mreads/s, "latency-bound"). LightGBM's ROCm kernel
instead splits a feature's rows across `grid_dim_y` blocks. Does that row-split raise occupancy
enough to win?

**Given/When/Then:** Given the starved-occupancy regime (few features × many rows), when each
feature's rows are partitioned across P cubes (each building a private LDS sub-hist, then
atomic-merging to the feature's global slot), then wall-clock drops as P rises — while the
result still matches the P=1 build to f32-atomic reorder noise.

## Research

This is the exact "deeper work" [spike 006](../006-gpu-u8-bins/README.md) pointed to:
*"the real GPU build bottleneck is atomic contention + uncoalesced scattered reads … further
GPU gains need occupancy / atomic-contention analysis."*

Baseline = AMD's ROCm fork `LightGBM-release-4.6.0.99`. Its launch geometry
(`cuda_histogram_constructor.cpp:147`):
`grid_dim_y = max(min_grid_dim_y, ceil(ceil(N/400)/block_dim_y))` — i.e. rows ARE split across
blocks, each block's `__shared__` sub-hist accumulating a row-slice, `atomicAdd_system`-merged.
Our kernel had no such row dimension. Full diff in
`.planning/notes/cubecl-vs-rocm-histogram-kernel-comparison.md`.

CubeCL LDS API (`SharedMemory::<Atomic<f32>>`, `sync_cube`) already proven in the production
kernel — no new external research needed.

## How to Run

```
cargo run --release --features rocm --example gpu_row_partition
```
(`crates/lgbm-compute/examples/gpu_row_partition.rs`) — sweeps P ∈ {1,2,4,8,16,32} on a fixed
1M×50×256 large leaf. P=1 is byte-identical to the production kernel, so the sweep isolates the
occupancy effect. Prints correctness-vs-P=1 + three timed rounds.

## What to Expect

Wall-clock falls as P rises off 1, peaking around P=16, then regressing at P=32 (over-partition).
Throughput rises ~600 → ~800 Mreads/s but stays far below the gfx1100 bandwidth ceiling.

## Investigation Trail

1. **Parameterized the production LDS kernel with a row dimension** `CubeCount = (FEATS, P)`.
   Cube `(f, p)` strides rows `p*256+u, +P*256, …` into its own LDS sub-hist, merges to global.
   P=1 reduces exactly to `construct_leaf_hist_resident_lds_kernel`.
2. **First run** showed P=16 fastest but round-1 P=1 was cold-start inflated (2011ms vs 1629ms
   round-2) — the naïve "speedup vs P1-round1" column overstated the win. Switched to reading
   **within-round** ratios.
3. **Depth probe — two more process restarts (A, B)** to kill the warmup-drift confound. P=16
   was the lowest time in **every round of every run** (6/6): A=1290/1263/1266ms, B=1238/1240/
   1221ms vs P=1 ≈ 1600–1790ms → a stable **1.30–1.39×**. P=32 regressed to ~1.0× every time.
4. **Correctness column** surfaced a parity interaction (see Results): more partitions = more
   independent f32 partial-sum trees = larger divergence from the P=1 build.

## Results

**VERDICT: VALIDATED ✓ — row-partitioning is a real lever; ~1.3–1.4× at P=16, but modest and
occupancy-bounded.**

Within-round speedup vs P=1 (3 process runs, ×3 rounds each):

| P | cubes | wkgrps/CU | speedup vs P=1 | peak Mreads/s | note |
|---|-------|-----------|----------------|---------------|------|
| 1 | 50 | ~0.5 | 1.00× (baseline) | ~620 | = production kernel; **starves the GPU** |
| 2 | 100 | ~1 | ~1.0–1.06× | ~610 | barely helps |
| 4 | 200 | ~2 | ~1.05–1.27× | ~650 | noisy |
| 8 | 400 | ~4 | ~1.0–1.22× | ~690 | |
| **16** | **800** | **~8** | **1.30–1.39× (consistent)** | **~820** | ★ sweet spot |
| 32 | 1600 | ~16 | ~1.0–1.03× | ~620 | over-partitioned — regresses |

### Findings

1. **The occupancy hypothesis is confirmed.** 50 cubes (P=1) leaves a 96-CU GPU under-fed; ~800
   cubes (P=16) hides the scattered-read/atomic latency. So spike-006's "latency-bound" was
   **partly starved occupancy**, not pure atomic cost — and the fix is the LightGBM `grid_dim_y`
   row-split.
2. **The lever must be tuned, not maximized.** P=16 (~8 workgroups/CU) wins; P=32 over-partitions
   (per-cube work too small, merge + scheduling overhead dominates) and gives back the win.
   Heuristic: `P ≈ clamp(target_cubes / num_features, 1, …)` with `target_cubes ≈ 8 × CU_count`.
3. **The win is modest (~1.35×) and occupancy-bounded.** Even at peak P=16, throughput (~820
   Mreads/s) is *orders of magnitude* below the gfx1100 bandwidth ceiling → the kernel is STILL
   atomic-contention bound. Row-partitioning does not reduce **intra-cube LDS atomic contention**
   (each cube still has 256 units colliding on its own sub-hist). Bigger gains need attacking
   that directly — register row-batching (todo) and/or multiple LDS sub-hist replicas per cube.
4. **Parity interaction (important).** Row-partitioning raises GPU-vs-P=1 divergence from
   ~4e-7 (P=1, just atomic reorder) to **~2e-5 relative / ~0.09 absolute** (P≥2) at 1M rows,
   because each partition is an independent f32 partial-sum tree. The **CPU f64 anchor is
   untouched** (the bit-exact merge gate stays safe — MANIFEST requirement), but the GPU
   ~1e-6 best-effort gate would need this documented/absorbed, exactly as the MANIFEST already
   flags ("the ~1e-6 GPU-vs-CPU parity contract at large shapes is a separate, still-open gate").

### Surprises

- P=32 *regressing* was not predicted — naïvely "more cubes = more occupancy = faster." The
  ~8-workgroups/CU optimum is a real tuning knob, not a maximize.
- The parity divergence growth (4e-7 → 2e-5) is a structural consequence of partitioning the
  sum, not noise — it scales with P, then plateaus once each partition is small.

## Signal for the Build

- **Ship row-partitioning as a tunable `P` on the resident/batched LDS build**, gated on leaf
  size and `num_features` so it only engages when `num_features` alone under-fills the GPU
  (large leaves). Default the heuristic to ~8 workgroups/CU; never set P so high it over-partitions.
- **Expect ~1.3–1.4×, not multi-×.** Pair it with the register-row-batching todo to attack the
  residual intra-cube atomic contention — the two are complementary (occupancy vs per-cube
  contention) and the real win is likely their product.
- **Carry the parity caveat into the build:** adding P widens the GPU f32 divergence; the f64
  CPU anchor remains the gate. Document the GPU residual per the existing 04-ROCM-GAPS pattern.
- ROI context (spike-006): with the CPU now multi-threaded (spike-005), GPU loses to it at every
  tested size — so this is ROCm-parity-track performance, not an overall-fastest win. Weigh
  accordingly before a full phase.

Reusable: `gpu_row_partition.rs` (parameterized row-partition LDS build + P-sweep harness).
