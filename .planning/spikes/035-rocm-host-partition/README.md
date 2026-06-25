---
spike: 035
name: rocm-host-partition
type: standard
validates: "Given spike-034 found the device partition round-trip is now the #1 reclaimable launch-bound phase (30–38%) after co-pack closed the scan-sync floor, when the rocm backend routes partition on the HOST via the SHIPPED spike-027 fused u8-route path (LGBM_ROCM_HOST_PARTITION) instead of the per-split device round-trip, then launch-bound train falls without regressing wide and stays within the ~1e-6 GPU parity contract"
verdict: VALIDATED
related: [034, 027, 029, 023, 024]
tags: [performance, gpu, rocm, partition, host-route, round-trip, parity, wire-candidate]
---

# Spike 035: Route the rocm partition on the HOST

## What This Validates

Given/When/Then — see frontmatter. spike-034's re-attribution surfaced this exact lever:
after co-pack (024) closed the scan-sync floor, the device `data_partition_native`
round-trip became the #1 reclaimable launch-bound phase (**38% medium / 30% large**). The
CpuBackend already routes partition on the host (the SHIPPED spike-027 fused u8-route path,
`prefers_host_partition()==true`); RocmBackend keeps the default `false` (device round-trip).
This spike flips rocm onto the host path and measures.

## Research / Prior Art

- **spike-027** (SHIPPED): the host fused u8-route partition (`split_fused_host`) — ONE random
  gather + ¼-width u8 route scratch + ONE u32 scatter, in place on `indices_`. Byte-identical
  `[left|right]` routing decision (same `SplitInner` MissingType::None as `data_partition_cpu_native`).
- **spike-034**: device partition = 30–38% of launch-bound train; the "index re-upload tradeoff"
  hypothesis — DISPROVEN here (see below).
- **def-f8u-01** (load-bearing): never compare two nondeterministic GPU f32 paths to each other
  at 1e-6; pin both to the f64 anchor. The GPU f32-atomic build is order-nondeterministic.

**Key structural finding (read the code, data_partition.rs:137–195):** BOTH partition paths
write their result into the **host** `self.indices` array (the device path reads `reordered_local`
BACK to host at :179–181 and writes it there). The resident build reads
`data_partition.indices_in_leaf(leaf)` — a **host** slice (learner.rs:1961/2053) — and uploads
it per build EITHER way. ⇒ **host partition adds NO index re-upload penalty**; the 034 tradeoff
concern is moot. The device path is pure overhead here: a host→device upload + route kernel +
blocking device→host readback, ~30 splits/tree, only to land back where the host path starts.

## Implementation (spike artifact, default-OFF)

Env-gated additive override (the spike-027 / CONVENTIONS "backend discriminator" idiom), in
`crates/lgbm-compute/src/lib.rs` `impl Backend for RocmBackend`:

```rust
fn prefers_host_partition(&self) -> bool {
    matches!(std::env::var("LGBM_ROCM_HOST_PARTITION").as_deref(), Ok("1"))
}
```

Default OFF ⇒ production byte-unchanged. `=1` flips rocm onto the existing host fused path.
**WIRE = make this return `true` unconditionally** (wide is a wash; no regime gate needed).

## How to Run

```bash
cargo build --release --features rocm --example bench_gpu_vs_cpu
# Launch-bound A/B (partition is 30–38% here):
LGBM_PHASE_PROF=1                              cargo run --release --features rocm --example bench_gpu_vs_cpu   # device (OFF)
LGBM_ROCM_HOST_PARTITION=1 LGBM_PHASE_PROF=1   cargo run --release --features rocm --example bench_gpu_vs_cpu   # host   (ON)
# Wide A/B (regression check): add LGBM_BENCH_SWEEP=wide to both.
# Parity (3-arm: device-vs-device2 isolates GPU f32 noise; device-vs-host is the question):
cargo run --release --features rocm --example spike035_host_partition_parity
```

## Investigation Trail

1. Read the partition + build code → both paths land in host `indices_`; build reads host
   indices either way ⇒ the device round-trip is pure overhead, host adds no re-upload. The
   034 "index re-upload tradeoff" is moot.
2. Added the env gate; A/B at launch-bound (2 restarts) + wide.
3. **Parity surprise:** a naive bit-exact device-vs-host compare FAILED (3/5000 rows, max
   9.5e-7). Applied def-f8u-01: added a device-vs-device2 arm. The max divergence is IDENTICAL
   (1.907e-6) for device-vs-device2 AND device-vs-host ⇒ host adds no parity class beyond the
   GPU's own f32-atomic-build run-to-run noise. Not bit-exact (the GPU never is); within contract.
4. Found the 2 hip full-train parity tests (`learner_parity_{resident,fused}_equals_host_tree_on_hip`)
   are PRE-EXISTINGLY BROKEN on master (`subtract_resident: smaller slot is empty` on the tiny
   spine corpus) — fail with the env OFF too (= master behavior). A separate defect; filed as a
   note. My bench/parity corpora (≥2k rows) train fine on both paths.

## Results

**VERDICT: ✅ VALIDATED — strong launch-bound win, wide-neutral, parity-safe within the GPU ~1e-6 contract. WIRE-CANDIDATE.**

### Perf (whole-train median, 2 process restarts, sign-stable)
| size | partition OFF→ON | speedup (r1 / r2) |
|------|------------------|-------------------|
| small (2k×12) | 13.5% → 0.8% | 1.09× / 1.04× |
| **medium (20k×30)** | 35.6% → 3.0% | **1.18× / 1.20×** |
| **large (200k×40)** | 29.9% → 11.2% | **1.23× / 1.22×** |
| wide 250k×500 | 8.8% → 2.9% | 1.03× (wash) |
| wide 500k×500 | 9.4% → 4.1% | 0.96× (wash) |
| wide 1M×500 | 8.8% → 4.7% | 1.02× (wash) |

Host partition removes the device round-trip; the residual host-gather cost scales with
leaf_rows (large 11.2%, medium 3.0%, small 0.8%). Launch-bound = clear win; wide = ±4% noise
(no regression) — partition is only ~9% there and build dominates.

### Parity (`spike035_host_partition_parity`, 5000×20)
- device-vs-device2 (inherent GPU f32-atomic noise): **max_abs = 1.907e-6**
- device-vs-host (the spike question): **max_abs = 1.907e-6** (identical ceiling; higher count
  4→20 but same magnitude)
- ⇒ host partition adds **no new parity penalty** beyond the GPU's own nondeterminism. Routing
  is byte-identical by construction; the residual is pure f32-atomic build-order sensitivity.
  CPU f64 anchor (the bit-exact hard gate) untouched (rocm-only path); `lgbm-treelearner --lib`
  77/0 green.

### Disposition: WIRE-CANDIDATE
Flip `RocmBackend::prefers_host_partition()` → `true` unconditionally. This is the rare GPU
lever with a genuine win **on this APU** (not just a "real on discrete gfx110x" deferral) —
because the device partition round-trip is pure overhead on shared DDR5. **Wiring needs an
oracle re-pin to the f64 anchor** (NOT a bit-exact swap — the GPU f32 build is ~1.9e-6
nondeterministic regardless). The pre-existing broken hip train parity tests that gate the re-pin
are now FIXED (commit `8aed100`, debug `subtract-resident-empty-hip`) → green gate available.
Route via `/gsd-quick` or `/gsd-plan-phase`.

### Caveats
Spoofed 8-CU gfx1152 APU; absolute ms APU-confounded; SIGN + fractions only; 2 restarts. On
discrete gfx110x the device round-trip crosses PCIe ⇒ the win is expected LARGER, but the
host-gather residual also grows with rows — re-measure there. Wide is a wash, not a win.

### Filed defect (separate) → ✅ FIXED (commit `8aed100`)
`learner_parity_{resident,fused}_equals_host_tree_on_hip` panicked `subtract_resident: smaller
slot is empty` on master (env-independent). Debugged in session `subtract-resident-empty-hip`:
NOT a tiny-corpus edge (my first guess) — root cause was Phase-12 co-pack DEFERRING the smaller
child's scan past `subtract_resident`, but on the FUSED path that scan IS the smaller histogram
build+store, so subtract ran before the histogram existed. Fix un-defers the smaller fused build
for the fused case only (co-pack never touched it). Resident-test breakage was a knock-on
`LGBM_FUSED_FORCE` env-leak through a panic (serialize + RAII-restore). Both tests 2/2 pass
parallel+serial; bit-exact gate green. The spike-035 wire gate is now unblocked.
