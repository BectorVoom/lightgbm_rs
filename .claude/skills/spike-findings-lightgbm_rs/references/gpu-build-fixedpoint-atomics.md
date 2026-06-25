# GPU histogram BUILD — fixed-point atomics & LDS contention levers

Implementation blueprint from spikes **015, 017, 018, 019, 020** — the post-014 deep dive
into the wide-shape (×500 feature) GPU histogram BUILD. Throughline: the build is
**atomic-contention bound** in LDS, and the one lever that actually moves it is switching the
accumulation from **f32 atomics to u64 fixed-point integer atomics** (SHIPPED). Per-warp LDS
replication is the seductive-but-null alternative.

## Requirements

- Backend stays **compile-time switched** (`--features rocm`); the **CPU f64 anchor is the
  bit-exact merge gate and stays untouched**. GPU work is held to the ~1e-6 ROCm gate only.
- The GPU here is a **spoofed 8-CU gfx1152 APU** (shared DDR5), NOT discrete gfx1100. Absolute
  Mr/s is APU-confounded — **judge the SIGN of a device-time A/B ratio, never the magnitude**.
- Gate every build-kernel change against the resident-build parity tests:
  `cargo test -p oracle-harness --features rocm --test kernel_parity` (esp.
  `kernel_parity_resident_build_fix_compact_equals_host_on_hip` and the `p_gt_1` variant).

## How to Build It

### 1. Locate the bottleneck first (spike-015 — the decomposition)

At wide shapes the per-leaf "scan" phase label **folds the build in** (the resident
`build_fix_scan` path; `build=0` in the growth-loop split is a fusion artifact, not a free
build). Use the two env-gated, behavior-neutral probes (live in `fusion_prof.rs`):

- `LGBM_SCAN_PROF=1` → per-leaf round-trip split (marshal / upload / launch+readback).
- `LGBM_SCAN_DRAIN=1` → forces a pre-scan `read_one_unchecked` of the histogram handle so the
  **async build compute is attributed to `build_drain`** instead of hiding inside the scan's
  readback sync.

Result (1M×500): build_drain **86→92%** of the scan-attributed wall, **growing with rows**
(build scales with rows; the per-bin scan does not). The build is the wide bottleneck and it
is atomic-bound (~820 Mr/s, far below the bandwidth ceiling).

### 2. The shipped win — u64 fixed-point integer atomics (spikes 018 + 019)

Replace per-bin f32 `atomicAdd` with **u64 two's-complement fixed-point**:

```rust
const SCALE_F32: f32 = 1_073_741_824.0; // 2^30
// accumulate: quantize grad/hess to i64, store the BITS as u64, wrapping atomicAdd
let qg = u64::cast_from(i64::cast_from(f32::round(ord_g[k] * SCALE_F32)));
sub[ti].fetch_add(qg);          // SharedMemory<Atomic<u64>>; wrapping u64 add == signed i64 add
// dequantize (fix/compact kernel): u64 bits -> i64 -> f64 / 2^30
hist[wbi] = f64::cast_from(i64::cast_from(h_raw[wbi])) / 1_073_741_824.0_f64;
```

Why it wins (spike-019's regime correction): on RDNA, f32 `atomicAdd` lowers to a **CAS retry
loop** (`ds_cmpst`) that serializes under heavy contention; integer `ds_add_u64` is a **native
single-instruction** op that never retries. So the win is **proportional to atomic load**:

| regime (cubes × rows/cube) | total atomic load | u64/f32 speedup |
|----------------------------|------------------:|-----------------|
| 16×1M (wide root, heavy)   | 16M | **1.57–1.70× SEP**, holds across P=1..16 |
| 64×200k (well-occupied)    | 12.8M | 1.16–1.28× |
| 16×200k (light)            | 3.2M | ~1.0× (NULL) |

Realistic resident-kernel magnitude **~1.3–1.7×** in the heavy regime the wide root/large
leaves live in (spike-018b's single-cube 1.9× was inflated). **Bonus, unconditional:** ~3600×
more accurate than f32 (exact in the cancelling regime) and **deterministic by construction**
(integer adds are order-independent) — so the resident build is bit-exact across runs and P>1.

### 3. Tune occupancy with row-partition (spike-007 lineage, composes)

Integer atomics **compose with** row-partition (P) — splitting rows across more cubes doesn't
change device-wide atomic pressure, so the integer win survives to P=16. Keep
`row_partition_count` as the occupancy lever (≈8 wkgrps/CU); see the gpu-histogram-kernel
reference.

## What to Avoid

- **Per-warp LDS sub-histogram replication (spikes 017, 020) — NULL at production P=1.** Giving
  each wave32 its own LDS replica (R8) wins ~1.1× on the **f32** kernel (017) and ~1.17–1.20×
  on the **u64** kernel **but only at P=16** (020); at the production **P=1 wide regime it
  REGRESSES ~0.90×** — the 2× LDS halves occupancy with no row-partition to supply cubes, and
  the u64 fixed-point switch already took the contention win. Keep `gpu_u64_lds_replication_ab.rs`
  as rocm-gated evidence; **do not wire**. Revisit only if discrete gfx110x + a wide P>1 policy
  both land.
- **`Atomic<i64>` — broken in cubecl-hip 0.10.** `Atomic<i64>::store` lowers to
  `atomicExch(long long*)` which HIP lacks (compiles, fails at runtime). Use `Atomic<u64>`
  two's-complement: store the i64 value's bits as u64, wrapping-add, reinterpret on readback.
  (`fetch_add` correctly reinterpret-casts; only `store` had the gap.)
- **`i32` fixed-point overflows** the hessian-sum case — `i64` @ S=2^30 is required (never
  overflows for ≤~1e9 rows). int16 packing (W5/spike-008) is a *different, coarse* mechanism —
  it was NULL/approximate; wide i64 fixed-point is the opposite (exact + faster).
- **Switching the build to f32 / cutting the scan round-trip / hoisting the constant per-feature
  arrays** — all measured NULL by spike-015 (the build was already f32-atomic; the round-trip is
  ≤14% and *shrinking* with rows; array marshal+upload is ~0.1%).
- **rocprof counters** are impossible on this spoofed gfx1152 — use the **device-time A/B proxy**
  (interleaved median + p25/p75 over ~11 reps, ≥2 process restarts, require a SEP-WIN: variant
  p75 below baseline p25).

## Constraints / ROI context

- **The GPU loses to the multi-threaded CPU anchor at every tested shape on this APU** (250k×500:
  CPU 1.8s vs GPU 7.1s ≈ 4×). All of this is **ROCm-parity-track maintenance**, not an
  overall-fastest win. Wide-many-feature shapes should route to CPU (spike-015 routing reality).
- u64 buffers are 2× the bytes of f32 histograms (size the resident pool accordingly).
- Determinism note (def-f8u-01): never compare two nondeterministic GPU f32 paths to each other
  at 1e-6 — pin GPU trees to the CPU f64 anchor. The u64 build removes this hazard (deterministic).

## Harnesses (in-crate, `--features rocm`)

`gpu_fixedpoint_resident_ab.rs` + `gpu_int_vs_f32_psweep.rs` (018/019 — u64-vs-f32 across
occupancy×P), `gpu_lds_replication.rs` (017 f32 R-sweep), `gpu_u64_lds_replication_ab.rs`
(020 u64 R-sweep). Decomposition probes: `LGBM_SCAN_PROF` / `LGBM_SCAN_DRAIN` in `fusion_prof.rs`.

## Origin

Synthesized from spikes: 015 (bottleneck located), 017 (f32 replication, modest/not-wired),
018 (u64 fixed-point — strong, SHIPPED), 019 (contention-regime correction), 020 (u64
replication — null at P=1).
Source files in: sources/015-parallel-f32-resident-build/, sources/017-perwarp-lds-replication/,
sources/018-fixedpoint-int-atomics/, sources/019-int-atomic-contention-regime/,
sources/020-perwarp-replication-on-u64/.
</content>
