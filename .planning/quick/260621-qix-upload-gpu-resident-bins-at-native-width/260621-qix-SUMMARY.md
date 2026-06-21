---
quick_id: 260621-qix
title: Upload GPU resident bins at native width (u8/u16/u32)
status: complete
date: 2026-06-21
---

# Quick Task 260621-qix — Summary

## What changed

The device-resident bin matrix (uploaded once-per-train since quick-260621-p9v) was
always widened to **u32**. It now uploads at the **narrowest uniform width** covering
every feature's `BinColumn` variant (widest variant present: any U32→U32, else any
U16→U16, else U8). Deliverable = **peak host memory** (no `to_u32_vec` 2GB widen + no
2GB u32 concat → one ~0.5GB native concat at 1M×500) + a ~4× smaller host→device
transfer.

**Implementation:**
- `lgbm-compute/src/lib.rs`: new `pub enum ResidentBinWidth {U8,U16,U32}` +
  `resident_bin_width(cols)`; `ResidentBins` carries `width`. `Backend::upload_resident_bins`
  signature `&[&[u32]]` → `&[&BinColumn]`; RocmBackend concats native bytes at the uniform
  width (narrower columns upcast) and uploads — no u32 widen.
- `kernels/histogram.rs`: the **3** resident-buffer-reading kernels
  (`construct_leaf_hist_resident_lds_kernel`, `construct_leaf_hist_resident_kernel`,
  `build_fix_scan_fused_kernel`) are now generic `<B: Int>` over the bin type, reading via
  `u32::cast_from(resident_bins[idx])` (value-faithful index — numerics unchanged). `width`
  is threaded through `resident_raw_build_into` / `build_fix_compact_resident_f64_on` /
  `build_fix_scan_resident_f64_on` / `…readback…` and each launch site `match width`-es to
  the `::<u8|u16|u32, R>` monomorphization (ArrayArg element COUNT is width-independent).
  The CUDA-mirror kernel (test-only, own u32 buffer) stays u32.
- `lgbm-treelearner/src/learner.rs`: passes native `&BinColumn` refs (drops `to_u32_vec`).
- Test/example callers of the u32 helper (`upload_resident_columns`) pass
  `ResidentBinWidth::U32`.

Feasibility precedent: cubecl 0.10 supports generic launch kernels (docs `gelu_array<F:
Float>`; repo `hist_fold_body<N: Numeric>`).

## Verification (parity — HARD gate — all GREEN)

- **GPU (~1e-6 ROCm gate, gfx1100):** `lgbm-compute --features rocm` 44/0 + every rocm
  suite (rocm_backend_parity 4, rocm_row_partition 2, rocm_parallel_histogram 7,
  rocm_cuda_mirror 4, …); `oracle-harness --features rocm` kernel_parity 15/0 (the resident
  test now uploads u8 columns **natively** and matches the host-gather path),
  learner_parity 31/0, boosting_parity 75/0. The change reads the same bin VALUES (only
  storage narrower), so device results are byte-faithful within the gate.
- **CPU f64 anchor (bit-exact merge gate, untouched):** `lgbm-treelearner --lib` 76/0,
  `lgbm-compute --lib` 43/0, `oracle-harness` (cpu) all pass. Build clean WITH and WITHOUT
  `--features rocm`.

## Measurement (gfx1100, bench_gpu_vs_cpu wide, 1M×500, all-u8 bins=128)

| metric | before (u32, p9v) | after (native u8) | delta |
|--------|-------------------|-------------------|-------|
| resident_bin_upload / rep | ~2149 ms | **~405 ms** | **~5× smaller** (iters-independent) |
| peak host alloc for upload | 2×~2GB (widen+concat) | 1×~0.5GB (native concat) | **~4× lower** |
| train @ iters=4 | ~20.0 s | **~15.9 s** (15.64/15.99/16.02, 3-run) | ~−20% |
| upload share @ iters=20 | ~5% | **~1%** | amortized away |

**Cumulative with p9v** (vs the original per-tree u32 re-upload): 1M×500 iters=4 train
**29.55s → ~15.9s (−46%)**.

**Honest framing:** the directly-attributable wins are the **upload bucket (~5×)** and the
**peak-host-memory (~4×)** reductions. The iters=4 end-to-end −20% is 3-run-stable but
larger than the upload-bucket saving alone (likely the removed 2×2GB transient allocations
easing allocator/page pressure) — not isolated in a controlled HEAD-vs-HEAD A/B, so I lean
on the bucket + memory numbers as the headline. At production iteration counts the
once-per-train upload is ~1% of wall-clock, so steady-state the value is the
memory-footprint reduction; the speed win is largest at low iteration counts. spike-006's
"u8 device-READ ≈0%" holds — the win is transfer + host memory, not kernel compute.

## Follow-on (not done)

- The ~17% boosting-loop overhead (per-iter `to_vec` clones of the 1M score buffer,
  spike-014b) remains the next non-amortizing lever.
