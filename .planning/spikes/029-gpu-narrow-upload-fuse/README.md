---
spike: 029
name: gpu-narrow-upload-fuse
type: standard
validates: "Given the ROCm partition path host-gathers a leaf's bins into a u32-WIDENED Vec and uploads count×u32 to the device every split, when the gather is fused to native u8 and uploaded narrow (count×u8, 4× fewer bytes) with a u8-reading route kernel, then the host gather + device upload+route+readback run faster — byte-identically (same route)"
verdict: VALIDATED
related: [027, 014b]
tags: [performance, gpu, rocm, partition, narrow-upload, transfer-volume, bit-exact, isolated-ab, roi-gated]
---

# Spike 029: GPU narrow-upload fuse (per-split partition routing)

## What This Validates

On the ROCm partition path, `DataPartition::split` host-gathers a leaf's bins into a
`Vec<u32>` (WIDENED to u32 even on a U8 column), and `RocmBackend::data_partition`
(`lib.rs:2084`, via `data_partition_on`) uploads that u32 buffer (`count × 4` bytes) to the
device, routes per-row, reads back. This spike A/Bs that against a **narrow fuse**:
host-gather to native u8, upload `count × 1` bytes (4× fewer), route with a u8-reading kernel.

Two components measured separately:
- **(A) HOST gather** — u32-widen vs u8-native (real CPU cost; the spike-027 mechanism applied
  to the rocm path's host prep).
- **(B) DEVICE round-trip** — upload + route + readback, u32 vs u8. The "GPU" here is the
  spoofed 8-CU APU on **shared DDR5**, so I *predicted* the transfer win would be APU-masked
  (same-memory copy). **It was not** — see results.

## How to Run

```
cargo run -p lgbm-compute --example spike029_gpu_narrow_upload_ab --release --features rocm
```

## Results (median, 3 process restarts; ratio = u32 / u8, >1 ⇒ narrow faster; route parity OK every cell)

| rows | host gather (u8 vs u32) | device round-trip (u8 vs u32) |
|------|--------------------------|--------------------------------|
| 100k | ~0.8–0.95× (launch/size noise) | 0.95–1.62× (noise) |
| 500k | **1.68–1.70×** | **1.10–1.22×** |
| 1M | **1.47–1.63×** | **1.71–1.73×** |
| 4M | ~1.0× (col random-read dominates) | **1.25–1.33×** |

## Verdict: VALIDATED — bit-exact, sign-stable ~1.2–1.7× at scale. Wireable. (ROI-gated to ROCm-parity-track.)

- **The narrow upload helps even on the shared-DDR5 APU** — contradicting the "transfer is
  free on an APU" prior. `create_from_slice` still moves the bytes through the memory system,
  so 4× fewer uploaded bytes (+ a u8-reading kernel that touches 4× less device memory) is a
  sign-stable **1.2–1.7×** on the device round-trip at ≥500k rows (the readback is identical —
  both return the same `count`-u32 `route`, so the win is upload + kernel-read).
- **Host gather also wins ~1.5–1.7×** at 500k–1M by not u32-widening (spike-027 mechanism).
  At 4M it washes to ~1.0× — the random `col[row]` gather over a >cache column becomes
  latency-bound and the output width stops mattering.
- **Bit-exact:** the u8 and u32 route kernels produce byte-identical `route[]` (value-identical
  routing; only the input storage width differs). Parity OK every cell, every restart.

## How to wire (extends the spike-027 fuse to the rocm path)

The op signature `data_partition(bins: &[u32], …)` forces the u32 widen. To wire:
1. Give the partition op a native-width input (pass `&BinColumn` like the CpuBackend fused path,
   or a `(bytes, ResidentBinWidth)` pair — mirror the qix `ResidentBinWidth` U8/U16/U32 pattern).
2. Add u8/u16 route-kernel variants (or make `data_partition_kernel` generic over `Int` and
   dispatch on width — exactly the qix histogram-kernel `<B: Int>` + `match width` precedent).
3. RocmBackend host-gathers native-width (no u32 widen) and uploads narrow.
Bit-exact by construction (route value-identical). The CpuBackend already routes host-side
(spike-027, `prefers_host_partition`), so this is the GPU sibling of that fuse.

## ROI honesty

ROCm-parity-track, like 021/024: on this 8-CU APU the GPU loses to the 16-core CPU overall,
and per spike-023 partition is only ~3% (wide) to ~23% (tall-narrow) of GPU train, so the e2e
share of a 1.5× round-trip win is single-digit %. The real payoff is on a **discrete gfx110x**
where the upload crosses a **PCIe bus** — there the 4× transfer-volume cut is a first-order
win, not a same-memory copy. Notably this is the one GPU-track lever in the 026→029 arc that
shows a sign-stable win on the actual APU (vs 026's cubecl-cpu null).
