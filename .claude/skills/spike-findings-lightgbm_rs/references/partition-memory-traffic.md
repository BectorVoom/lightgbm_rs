# Partition (row-routing) — memory-traffic & narrow-upload

Implementation blueprint from spikes **026, 027, 028, 029, 032, 033, 034, 035** — the
`DataPartition::split` (C++ `DataPartition::Split`, the per-leaf stable row-routing into
`[left | right]`) optimization arc. FOUR wins SHIPPED (027 CPU fuse, 029 ROCm narrow-upload, 032
CPU one-gather fold, **035 rocm-routes-partition-on-host default-ON**), three NULLs/ROI-gated with
root causes (026 parallelize, 028 double-buffer, 033 prefetch), and one re-attribution (034). The
throughline: **partition is memory-bound; cut TRAFFIC, don't add cores.** After 032 the CPU host
partition is effectively DONE; then 034 found that — once co-pack closed the scan-sync floor — the
GPU's *device* partition round-trip became its #1 launch-bound phase (30–38%), and 035 SHIPPED the
fix (route the rocm partition on the host by default) — the rare GPU lever that wins on the APU.

## Requirements

- **CPU f64 anchor stays bit-exact**; the partition order feeds the histogram-subtraction trick,
  so any `[left|right]` reorder drift surfaces in `raw_bin_train_matches_cpp_golden`. Gate every
  change with `cargo test -p lgbm-treelearner --lib` (esp. `split_*`) + `-p oracle-harness`
  (`raw_bin_train_matches_cpp_golden`, `kernel_parity_partition_exact_on_cpu`) + `-p lgbm`.
- **ROCm partition routing is f64-free ⇒ bit-EXACT** (not ~1e-6): gate with
  `kernel_parity_partition_exact_on_hip` ON HARDWARE, and add a U8/U16 `BinColumn` cell whenever
  the narrow path changes.
- **Ship on the isolated A/B at scale** (≥500k rows, 2–3 process restarts); CPU is real hardware
  so partition wall-clock is legitimate (only the GPU is the spoofed APU).

## How to Build It

### The mechanism that works: fuse the gather, cut the materialization (027, SHIPPED — CPU)

The old `DataPartition::split` numeric branch materialized ~4 `count`-u32 buffers per split:
`leaf_rows` (to_vec) + `leaf_feature_bins` (Vec<u32>, **widened to u32 even on a U8 column**) +
`reordered` local indices + a local→row remap. The fused path (proven 1.3–2.7×, biggest ~2.3× at
U8 — `sources/027-fused-gather-partition/`):

```rust
// gate the fused host path on a backend discriminator (NOT a global env/flag) —
// add Backend::prefers_host_partition() { false }; CpuBackend overrides true.
if backend.prefers_host_partition() {
    // pass 1: ONE random gather + route + count-left, into a ¼-width u8 scratch
    let mut route = vec![0u8; count];
    let mut left = 0usize;
    for i in 0..count {
        let gr = go_right(feature_bins.bin(self.indices[begin + i] as usize)); // dense_bin.hpp:322-365
        route[i] = gr as u8; left += (!gr) as usize;
    }
    // pass 2: scatter ROW ids straight into one out buffer (stable [left|right]), then copyback
    let mut out = vec![0u32; count];
    let (mut l, mut r) = (0usize, left);
    for i in 0..count {
        let row = self.indices[begin + i];
        if route[i] == 0 { out[l] = row; l += 1; } else { out[r] = row; r += 1; }
    }
    self.indices[begin..begin + count].copy_from_slice(&out);
}
```

`go_right`: `th = threshold + min_bin` (i32; `−1` if `most_freq_bin == 0`);
`default_to_right = most_freq_bin > threshold`; `if bin < min_bin || bin > max_bin { default_to_right } else { bin > th }`.
KEEP the **u8 route scratch** — the no-scratch 2-gather variant (V2) regresses to 0.79× at U32/4M.

### Then fold the redundant validation gather into pass-1 (032, SHIPPED — CPU, quick-260625-qn9)

**Reading the SHIPPED 027 wiring** found `split_fused_host` did TWO random gathers, not the ONE
027 measured: a standalone per-row bin range-check loop (`feature_bins.bin(row)` for every leaf
row) BEFORE pass-1's route gather. At scale the 2nd gather re-misses cache = exactly the traffic
the arc fought to cut. Fold the range-check INTO pass-1 — gather `b` once, check `b >= num_bin`
**before any `route`/`indices` write** (early-return leaves `self.indices` unmutated ⇒ same
lowest-index `BinIndexOutOfRange`, bit-exact on success AND error), then `go_right(b)`:

```rust
for i in 0..count {
    let row = self.indices[begin + i];
    let b = feature_bins.bin(row as usize);
    if b >= num_bin { return Err(ComputeError::BinIndexOutOfRange { row: i, bin: b, num_bin }); }
    let gr = go_right(b);
    route[i] = gr as u8; left += (!gr) as usize;
}
```

Proven ~1.14–1.41× at production U8 ≥1M rows (up to ~1.8× U32), bit-exact (3 restarts, parity OK
every cell). Keep the `num_bin==0` / `threshold>=num_bin` pre-checks. **Lesson: audit the shipped
wiring vs the spike that "shipped" it** — a redundant pass added for parity/safety is often a free,
bit-exact reclaim (the host analog of the GPU "re-attribute after every change" rule).

### Narrow the GPU upload (029, SHIPPED — ROCm)

The device branch (`else` of `prefers_host_partition`) uploaded a u32-widened buffer per split.
Narrow it (proven ~1.2–1.7× device round-trip + ~1.5–1.7× host gather, bit-exact). Wire ADDITIVELY:

```rust
// 1. data_partition_kernel generic over <B: Int>, read u32::cast_from(bins[i]) (qix precedent)
// 2. native-width device entry: match the BinColumn variant -> launch ::<u8|u16|u32>; readback
//    (route, count u32) + host two-pass gather + (reordered, split_point) UNCHANGED.
// 3. additive trait method (do NOT change data_partition(&[u32]) signature):
fn data_partition_native(&self, client, bins: &BinColumn, ...) -> Result<(Vec<u32>, usize)> {
    self.data_partition(client, &bins.to_u32_vec(), ...)   // widening DEFAULT (cpu byte-unchanged)
}                                                          // RocmBackend OVERRIDES with native upload
// 4. device branch: let leaf_bins = feature_bins.gather(&leaf_rows); // BinColumn::gather keeps width
//    backend.data_partition_native(client, &leaf_bins, ...)
```

Bit-exact by construction: u8/u16/u32 read the same value via `u32::cast_from` → identical route.
Even on the **shared-DDR5 APU the narrow upload wins** — `create_from_slice` still moves the bytes
(see "What to Avoid": the "APU transfer is free" assumption is FALSE).

### Route the rocm partition on the HOST by default (035, SHIPPED — quick-260626-a6t)

**The biggest GPU-track partition win, and the one that wins on the APU itself.** Spike-034
re-profiled after the co-pack (024) + narrow-upload (029) wires shipped and found the bottleneck
MOVED: with the scan-sync floor closed by co-pack, the **device `data_partition_native` round-trip
is now the #1 reclaimable launch-bound phase — 38% (medium) / 30% (large)** of GPU train (scan
collapsed to 3–7%). At wide it's only ~9% (build dominates). So the device partition round-trip
(host gather → upload → route kernel → blocking readback, ~30 per-split syncs/tree) is **pure
overhead on shared DDR5**: both the device path AND the host fused path (027) land the result in
host `indices_`, and the resident build reads host indices either way — so there is **NO index
re-upload penalty** (the intuitive "tradeoff" is moot; read the code: `data_partition.rs:137-195`
+ `learner.rs` `indices_in_leaf`).

Fix = make `RocmBackend::prefers_host_partition()` default **ON** (off-switch
`LGBM_ROCM_HOST_PARTITION=0`), so the rocm path runs the SHIPPED 027 host fused partition. The
029 device narrow-upload path becomes the off-switch fallback.

```rust
// crates/lgbm-compute/src/lib.rs — RocmBackend impl
fn prefers_host_partition(&self) -> bool {
    !matches!(std::env::var("LGBM_ROCM_HOST_PARTITION").as_deref(), Ok("0"))
}
```

Measured (2 restarts, sign-stable): **~1.18–1.20× medium, ~1.22–1.23× large**; wide a WASH
(±4% noise, no regression — partition only ~9% there). **Parity (def-f8u-01, load-bearing):** this
is NOT a bit-exact swap — the GPU f32 atomic build is ~1.9e-6 nondeterministic regardless of
partition path (host-vs-device max divergence 1.907e-6 = IDENTICAL to device-vs-device run-to-run
noise; see `sources/035-rocm-host-partition/spike035_host_partition_parity.rs`, the 3-arm test).
The valid gate is the anchor-pinned `hip::learner_parity_{resident,fused}_equals_host_tree_on_hip`
(GPU tree vs cpu f64 anchor within ~1e-6), NOT a GPU-vs-GPU compare. CPU f64 anchor untouched.

**Landmine that gated this wire:** those two hip parity tests were RED on master
(`subtract_resident: smaller slot is empty`) — a latent fused-path bug from the Phase-12 co-pack
deferral (see the scan-roundtrip reference / debug `subtract-resident-empty-hip`, fix `8aed100`).
Fix it FIRST so the re-pin gate is green. ROI: rare GPU lever that genuinely wins on this 8-CU
APU; larger still on discrete gfx110x (the device round-trip crosses PCIe there).

## What to Avoid

- **Don't parallelize partition (026, NULL).** A cubecl-cpu scan+scatter (per-chunk count → host
  prefix-sum → disjoint scatter) is bit-exact but loses to serial-native except in a narrow
  cache-resident balanced window (~100k → 2×); at ≥500k it's parity-to-slower, and 0.2–0.6× on
  SKEWED leaf bins (serial wins skewed via branch-prediction; trees deepen INTO skew). Root cause:
  **at scale partition is memory-bandwidth-bound on shared DDR5** — the 16 cores share one DRAM
  controller, so parallelism buys ~nothing *even with zero build contention*. This also reframes
  the reverted rayon attempt (quick-260622-ia0): the wall was never *build contention*, it is
  *shared DRAM bandwidth*. **No host-parallelism (rayon OR cubecl-cpu) reclaims the residual at scale.**
- **Don't double-buffer to drop the copy-back (028, NULL).** The `copy_from_slice` is only 2.2–6.8%
  of the fused op (gather + scatter dominate); removing it via a persistent ping-pong `alt` buffer
  is within noise (0.86–1.04×) and costs a 2nd size-N indices buffer + cross-leaf bookkeeping under
  leaf-wise growth. **C++ LightGBM copies back for the same reason.** CPU partition opt is DONE at 027.
- **Don't assume host↔device transfer volume is free on an integrated GPU** (the 029 surprise).
  Shared DDR5 ≠ free copy; 4× fewer uploaded bytes is a real ~1.2–1.7× even on the APU.
- **Don't `cubecl-cpu`-ize a too-cheap op.** cubecl-cpu's per-launch dispatch+readback (~ms) swamps
  a few-ms op; it threads on the **CubeDim/UNIT axis** (`CubeDim(1)` runs serial — use ≥16). It lost
  to native here (corroborates 260608-mc5 / the unified-kernel-pref memory).
- **Don't software-prefetch the gather at the production U8 width (033, PARTIAL/ROI-gated).**
  `_mm_prefetch`-ahead of the random `feature_bins.bin(row)` gather is null-to-SLOWER until the bin
  column ≫ LLC. U8 (1 B/row) only crosses at ~4M rows ⇒ ~1.05–1.16× whole-op at a ROOT split only,
  null/negative for all smaller (deeper) leaves and all datasets <4M. The big 2–3× is U32-only
  (≥4M) — a high-cardinality regime the `max_bin=255`→U8 default never reaches. Plus it's an
  **x86-only intrinsic** (cfg + non-x86 fallback on a must-build-everywhere anchor). After 032 there
  is little gather latency left to hide at U8. Revisit only for U16/U32 + multi-M-row workloads
  (gate `width != U8 && leaf_rows ≥ ~2M`, tuned D ≈ 64–128).
- **Don't refactor `.bin()` per-row matches into a typed-`&[T]` loop (033 autovectorization trap).**
  The tight typed loop auto-vectorizes into a slow AVX gather (`vpgatherdd`) that serializes
  cache-missing lanes WORSE than scalar independent loads under OoO MLP — **1.5–2× SLOWER** than the
  scalar `.bin()` match at scale (all sizes, 3 restarts). The per-row enum match is the FASTER
  codegen for a random gather. Keep `.bin()` in random-gather loops.

## Constraints

- `BinColumn` is `enum {U8,U16,U32}`; `.bin(row)->u32`, `.gather(rows)->BinColumn` (preserves width),
  `.to_u32_vec()` (cold widen). The fused host path reads `.bin(row)` directly (no widen).
- Backend discriminators are default-false trait methods overridden on one backend
  (`prefers_host_partition` cpu-true; `data_partition_native` default widens, rocm overrides) — the
  same idiom as `wants_resident_bins`/`resident_pool_supported`. Never a global env/flag.
- ROI honesty: the GPU narrow-upload (029) is ROCm-parity-track on this 8-CU APU (partition is
  3–23% of GPU train per spike-023; APU loses to CPU overall) — first-order only on a discrete
  PCIe gfx110x. The CPU fuse (027) is a real end-to-end win (the ~29% tall-narrow residual = the
  #1 remaining CPU-vs-C++ gap).

## Origin

Synthesized from spikes: 026 (parallelize=NULL, the bandwidth diagnosis), 027 (fuse-gather=SHIPPED
1.3–2.7×, CPU), 028 (double-buffer=NULL), 029 (GPU narrow-upload=SHIPPED ~1.2–1.7×, bit-exact on GPU),
032 (one-gather validation fold=SHIPPED ~1.14–1.41× U8, bit-exact), 033 (prefetch=PARTIAL/ROI-gated,
don't-wire + the autovectorization landmine), 034 (post-co-pack re-attribution — device partition is
now the #1 launch-bound phase 30–38%), 035 (route rocm partition on host=SHIPPED default-ON, ~1.18–1.23×
launch-bound, wash wide, parity within ~1e-6).
Source files in: `sources/026-cubecl-cpu-partition-scan-scatter/`, `sources/027-fused-gather-partition/`,
`sources/028-doublebuffer-partition/`, `sources/029-gpu-narrow-upload-fuse/`,
`sources/032-partition-validation-fold/`, `sources/033-partition-gather-prefetch/`,
`sources/034-post-copack-narrowupload-reattribution/`, `sources/035-rocm-host-partition/`.
Shipped via quick tasks 260625-hw2 (027), 260625-j1l (029), 260625-qn9 (032), 260626-a6t (035).
