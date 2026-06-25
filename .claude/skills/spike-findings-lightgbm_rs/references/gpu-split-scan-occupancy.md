# GPU split-SCAN — occupancy & within-feature parallelization

Implementation blueprint from spikes **021, 016, 022, 022b** (+ 015's decomposition tooling).
After the u64 build win (see gpu-build-fixedpoint-atomics), the per-leaf split **SCAN** became the
other ~half of the wide GPU per-leaf cost. The shipped win is a pure **occupancy repack**
(feature-per-lane, bit-exact); deeper within-feature parallelization is **parity-resolved (022)
AND perf-disproven (022b) — DON'T WIRE**.

## Requirements

- **CPU f64 anchor untouched / bit-exact**; GPU held to ~1e-6 ROCm gate. Gate scan changes with
  `cargo test -p oracle-harness --features rocm --test kernel_parity` (esp.
  `kernel_parity_split_within_tol_on_hip` + `kernel_parity_fused_equals_per_feature_and_native`).
- Spoofed 8-CU APU ⇒ judge the **SIGN** of device-time ratios; wall-clock end-to-end is noisy
  (a single cold run can show a spurious 3.8×). Use the structured `LGBM_PHASE_PROF` attribution
  + the isolated build-drained scan number, not raw train wall-clock, for the real signal.

## How to Build It

### 1. Re-profile after any build change — the bottleneck MOVES (spike-015 tooling)

Run `LGBM_PHASE_PROF=1 LGBM_SCAN_PROF=1 LGBM_SCAN_DRAIN=1 LGBM_BENCH_SWEEP=wide` on
`examples/bench_gpu_vs_cpu`. The build-drain A/B splits the per-leaf "scan" into genuine
build vs genuine scan. Post-u64-build (250k×500): the per-leaf round-trip is **~46% build /
~54% genuine scan** — the scan is now worth attacking.

### 2. The shipped win — feature-per-lane occupancy (spike-021, W=64)

The fused split-scan launched `CubeCount=(num_features,1,1) × CubeDim(1)` — **one
single-threaded cube per feature** (1 active lane of each wave32 ≈ 1/32 ALU utilization). Pack
**one feature per LANE**: index features by the global lane and guard the tail.

```rust
// kernel: feature index = global lane (ABSOLUTE_POS is usize in cubecl; cast to u32)
let f = ABSOLUTE_POS as u32;
if f < n_feats { split_scan_body(/* ... feature f, SEQUENTIAL per-feature scan ... */); }
// launch: W from env LGBM_SCAN_CUBEDIM (rocm default 64; non-rocm pinned 1)
let scan_w = scan_cube_dim();
find_best_splits_fused_kernel::launch(client,
    CubeCount::Static((n as u32).div_ceil(scan_w), 1, 1),
    CubeDim::new_1d(scan_w), /* ...args..., */ n as u32 /* n_feats */);
```

Each feature's scan stays **sequential** ⇒ **bit-identical for every W** (W=1 is byte-identical
to the original; W>1 only changes *which thread* runs each feature — NO reorder, unlike §3).
Isolated scan launch+readback (build drained, 250k×500): W=1 11.8s → **W=64 3.99s (2.96×)** →
W=128 3.33s (3.54×), monotonic. Default **W=64** = 1 cube/CU × 2 wave32 at 500 feat (the robust
knee, not over-fit to W=128's APU latency-hiding peak).

### 3. Within-feature parallel scan — PARITY-SAFE, but don't wire yet (spikes 016 + 022)

Parallelizing *inside* a feature (plane-cooperative prefix-scan over bins) **reorders the f64
prefix-sum** → not bit-exact. Resolution (host probes, no GPU kernel needed):

- **spike-016:** threshold (the present-data partition) is **stable** — 0 flips on realistic
  data; the rare flips are equal-gain plateaus (~1e-13). Only `default_left` flips (~34%).
- **spike-022 closed the deferred question:** every `default_left` flip is **COSMETIC** — it
  reroutes only MISSING values / an empty default bin; max present-data leaf-output Δ = **0.0**
  across 480k histograms, **0** flips with a populated default bin. Mechanism: the
  reverse-vs-forward gain gap is **linear in default-bin mass**, so only an (essentially empty)
  bin can be flipped by the ~1e-12 reorder noise. ⇒ a **tie-aware argmax** (reverse-first /
  lowest-`t`) reproduces the same splits within the ~1e-6 hip gate.

Feasibility: cubecl-hip 0.10 `plane_inclusive_sum` lowers to a Hillis-Steele `__shfl_up` loop
(works) — but `num_bin` reaches 256 ≫ `PLANE_DIM` (32/64), so a real kernel needs a **segmented
LDS block-scan** (substantial).

## What to Avoid

- **Don't assume the isolated scan win carries to end-to-end.** Feature-per-lane is ~3× on the
  isolated (build-drained) scan but only **~1.27× end-to-end** (phase_prof: learner −20%, total
  −8%) — the per-leaf readback **sync is also gated by the unchanged build** (Amdahl: speeding
  the scan just makes the build the bottleneck again). Report phase_prof, discard cold-run
  wall-clock outliers.
- **Don't hoist the per-leaf constant scan arrays** (the slot_off/num_bin/... 7-array rebuild) —
  spike-015 measured marshal+upload at **0.2%**; it is a dead lever.
- **Don't build the within-feature parallel scan now.** It is **parity-safe** but **ROI-gated**:
  post-021 the scan already saturates the device at wide shapes, so within-feature parallelism
  helps **only narrow** (few features under-fill the lanes) — exactly the regime where the GPU is
  least competitive vs the CPU anchor. It's also a heavy kernel (256-bin block-scan + tie-aware
  plane-argmax + default-bin handling). The deferred **022b** perf A/B (cooperative-scan vs
  feature-per-lane across feature counts) is the gate before any build.

## Constraints

- The fused scan kernel is GPU-only in production (`RocmBackend`); `CpuBackend` uses the native
  rayon per-feature scan (cubecl-cpu lost to native). The cubecl-cpu fused path exists only for
  the oracle bit-exact gate — so `scan_cube_dim()` is pinned to W=1 on non-rocm.
- ROI honesty: spoofed 8-CU APU ⇒ ROCm-parity-track maintenance. The removed `CubeDim(1)`
  under-utilization (and any future within-feature win) is **more** wasteful/valuable on a real
  discrete gfx110x (more idle lanes), where the end-to-end share would be larger.

## Harnesses

`examples/bench_gpu_vs_cpu.rs` (`LGBM_BENCH_SWEEP=wide`, env `LGBM_SCAN_CUBEDIM` for the W
sweep). Host parity probes (no GPU): `spike016_scan_reorder_probe.rs`,
`spike022_default_bin_parity_probe.rs` — see the host-parity-probe convention in CONVENTIONS.md.

## Origin

Synthesized from spikes: 015 (scan/build decomposition + `LGBM_SCAN_PROF`/`DRAIN` tooling),
016 (parallel-scan reorder parity — PARTIAL), 021 (feature-per-lane occupancy — SHIPPED),
022 (within-feature parallel-scan parity gate — resolved safe, ROI-gated), 022b (the deferred
within-feature PERF A/B — cooperation beats the shipped 021 only at NARROW ≤256 feat where the
GPU is least competitive vs CPU; at the WIDE F=512 production shape it is WASH-to-regression once
the occupancy confound is controlled [cd64 K1 = the real baseline; cd256 under-occupies]; argmax
mism=0, gainrel ≤9e-15 ⇒ confirms 022's parity finding on the real kernel — DON'T WIRE).
Source files in: sources/016-parallel-scan-reorder-parity/, sources/021-scan-feature-per-lane-occupancy/,
sources/022-within-feature-parallel-scan-parity/, sources/022b-within-feature-scan-perf-ab/
(+ 015's probes in sources/015-parallel-f32-resident-build/).
</content>
