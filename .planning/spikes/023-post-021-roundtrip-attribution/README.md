---
spike: 023
name: post-021-roundtrip-attribution
type: measurement
validates: "Given the post-u64-build + post-feature-per-lane-scan GPU train, when profiled across both regimes with the whole-train budget + scan-drain + new per-tree launch/round-trip counters, then the current dominant reclaimable residual is named — settling whether the per-leaf scan-readback SYNC floor (spike-001's parked launch floor) is now #1, gating 024/025"
verdict: PENDING
related: [001, 014a, 014b, 015, 021, 022b]
tags: [performance, gpu, rocm, profiling, attribution, roundtrip, launch-floor, re-profile]
---

# Spike 023: Post-021 GPU Round-Trip Attribution

## What This Validates

Given the post-u64-build (018/019) + post-feature-per-lane-scan (021) GPU train — the
bottleneck has **moved twice** since the last full attribution (015's "build 86–92%") —
when profiled across **both regimes** (small/medium launch-bound + wide 250k–1M×500) with
the whole-train BUDGET + `LGBM_SCAN_DRAIN` build-drain + **new per-tree launch/round-trip
COUNTERS**, then the current dominant **reclaimable residual** is named: build-compute vs
scan-compute vs the per-leaf scan-readback SYNC floor vs host-partition.

This is the **kill-check** for the structural-frontier campaign (024 batch-sibling-scans,
025 collapse-per-leaf-launches). The project's iron rule: *re-profile after every build
change — the bottleneck moves.*

## Why this is the frontier (prior art)

The 001–022 campaign optimized **per-kernel throughput** (u64 atomics, feature-per-lane
scan). The **one untouched dimension is the per-leaf loop STRUCTURE**:
- spike-001 parked the seed `batch-find-best-split-subtract-partition`, naming a
  **launch-bound floor** (~5–6s/30-iters regardless of rows ≤100k).
- spike-021 hit the **Amdahl wall**: isolated scan 3× but e2e only 1.27× — "readback sync
  also gated by unchanged build."
- The map (this spike's setup): builds/scans are batched across **features** (260608
  "COLLAPSE") but **not across leaves** — ~205 launches + ~31 blocking scan-readback syncs
  per tree. The histogram **subtraction trick already runs on-device** (`subtract_resident`)
  — that lever is closed.
- **build+fix+scan FUSION (3→1 launch) was already tried (260608-t3t) → FLAT-to-NEGATIVE**:
  collapsing forces a *sequential* f64 build that costs more than the launches saved. So
  any 025 launch-collapse must NOT sacrifice the parallel build.

## Method

Instrumentation added (env-gated by `LGBM_PHASE_PROF=1`, parity-neutral atomic increments):
per-tree launch/round-trip COUNTERS in `phase_prof.rs` — `BUILD_RESIDENT_CNT`,
`SUBTRACT_RESIDENT_CNT`, `SCAN_RESIDENT_CNT` (= blocking readback syncs), `FUSED_CNT`,
bumped at the per-leaf Backend entry points in `learner.rs`. Dumped as a `COUNTS:` line.

Runs (warm median, gfx1152 APU spoofed gfx1100):
```
# launch-bound regime (small 2k×12, medium 20k×30, large 200k×40)
LGBM_PHASE_PROF=1 LGBM_SCAN_PROF=1 cargo run --release --features rocm --example bench_gpu_vs_cpu
# + build-drain A/B (re-attribute async build out of the scan readback)
LGBM_SCAN_DRAIN=1 LGBM_PHASE_PROF=1 LGBM_SCAN_PROF=1 cargo run --release --features rocm --example bench_gpu_vs_cpu
# wide regime (250k/500k/1M × 500)
LGBM_BENCH_SWEEP=wide LGBM_PHASE_PROF=1 LGBM_SCAN_PROF=1 cargo run --release --features rocm --example bench_gpu_vs_cpu
```

## Investigation Trail

1. **Mapped the loop first** (read-only). Confirmed: builds/scans batched across
   FEATURES (260608 "COLLAPSE") but NOT across leaves; the histogram **subtraction trick
   already runs on-device** (`subtract_resident`, no double-rebuild) — that lever is
   closed; build+fix+scan **FUSION already tried (260608-t3t) → FLAT-to-NEGATIVE** (forces
   a sequential f64 build). So the only open structural dimension = scan-readback **SYNC
   count**.
2. **Added per-tree launch/round-trip COUNTERS** (`phase_prof.rs`, `LGBM_PHASE_PROF`-gated,
   parity-neutral) at the 4 per-leaf Backend entry points in `learner.rs`. Treelearner
   gate **76/0** + oracle golden green ⇒ instrumentation is inert/parity-neutral.
3. **Ran 3 passes** (warm median, gfx1152 APU): default sizes (non-drain + DRAIN) +
   wide sweep (DRAIN). The DRAIN A/B re-attributes the async f32/u64-atomic build out of
   the scan readback bucket (the 015 artifact).

## Results

**VERDICT: VALIDATED (measurement). The post-021 GPU bottleneck is REGIME-SPLIT, and the
launch frontier is real but narrow.**

### Per-tree launch/round-trip COUNTS (empirical, NEW — shape-INDEPENDENT, `num_leaves=31`)

| counter | per tree | meaning |
|---------|----------|---------|
| `build_resident` | **~30** | smaller child + root, directly built (1 batched launch each) |
| `subtract_resident` | **~29** | larger child, on-device `parent − smaller` |
| `scan_resident` (= **SYNCS**) | **~59** | **one blocking readback per leaf-node created — BOTH siblings of every split scanned separately** |
| device launches | **~118** | 30 + 29 + 59 |

The 59 syncs/tree corrects the "~31" estimate: every split scans its two children in **two
separate launches + two separate readbacks**. This is exactly 024's target.

### Time attribution (DRAIN run — build-compute vs genuine scan-readback SYNC)

| shape | build-compute | scan-sync (launch+readback) | **scan-sync % of round-trip** | partition % of train | ~µs/sync |
|-------|------|------|------|------|------|
| small 2k×12 | 669 ms | 651 ms | **47.7%** | 13.1% | ~44 |
| medium 20k×30 | 1964 ms | 1091 ms | **35.1%** | 13.0% | ~74 |
| large 200k×40 | 5688 ms | 2602 ms | 31.1% | 22.9% | ~178 |
| 250k×500 | 3175 ms | 433 ms | 11.9% | 7.0% | — |
| 500k×500 | 5763 ms | 371 ms | 6.0% | 7.3% | — |
| 1M×500 | 12215 ms | 410 ms | **3.2%** | 8.1% | — |

### Findings

1. **Launch-bound regime (small/medium):** the per-leaf scan-readback **SYNC is the largest
   reclaimable GPU residual** — ~48% (small) / ~35% (medium) of the scan round-trip. At
   small the compute per scan is ~0 (12 feat × 32 bins), so the ~44 µs/sync is almost
   entirely **fixed launch+sync latency**. Halving the sync count (59→~30/tree via 024)
   reclaims ~that fixed overhead × ~29 saved syncs/tree ⇒ **~10–15% e2e ceiling at
   small/medium**, and **moves the CPU-vs-GPU crossover LEFT** (spike-001's parked payoff).
2. **Compute-bound regime (large/wide):** build-compute **dominates and GROWS with rows**
   (68% @ 200k → 87.6% → 93.6% → **96.5% @ 1M×500**), exactly as 015 found, undiminished by
   the u64 win (which made each atomic cheaper, not the build smaller). Genuine scan-sync
   **collapses to 3.2%** at 1M×500. ⇒ **024 is worthless at wide**; build levers are
   exhausted (u64 shipped; per-warp replication NULL at production P=1, spike-020). Route
   wide → CPU (unchanged guidance).
3. **Host `partition` is a growing residual NEITHER GPU lever touches** — 13% (small) → 23%
   (200k×40). This is the single-threaded `DataPartition::split` (the known overall-CPU
   bottleneck). At large it is bigger than the scan-sync; but it is a HOST/CPU lever, out
   of this GPU campaign's scope. **Flagged for the CPU track.**

### Lever disposition (gates 024/025)

- **024 `batch-sibling-scans` → GREEN, targeted at small/medium.** Co-pack the two
  siblings of each split into ONE scan launch + ONE readback (59→~30 syncs/tree, ~2×).
  Bit-exact (each feature's scan unchanged — no spike-016 reorder), keeps the parallel
  build (dodges the 260608-t3t fused-build null). Both siblings are already scanned in the
  same `find_best_splits` call ⇒ a local change. Ceiling ~10–15% small/medium, ~0 wide.
- **025 `collapse-per-leaf-launches` → DOWNGRADE.** Fused build+fix+scan collapse is
  already null; beyond 024's sibling merge the leaf-wise data dependency caps further sync
  cuts. Pursue only if 024 lands AND a depth-wise frontier-batched variant looks promising.

### Caveats

- gfx1152 **8-CU APU spoofed as gfx1100** — all magnitudes are sign-only proxies; the GPU
  loses to the 16-core CPU everywhere (this is ROCm-parity / discrete-GPU-readiness work).
- DRAIN forces a pre-scan read; it slightly inflates total wall-clock but cleanly splits
  build-compute from scan-sync (the RATIO is the deliverable, stable across the 3 passes).
- One shape family per regime; the per-tree COUNTS are exact (num_leaves-derived), the time
  ratios are warm-median (default 5 reps × 50 iters; wide 3 × 8).

