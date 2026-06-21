---
quick_id: 260621-qix
title: Upload GPU resident bins at native width (u8/u16/u32)
status: ready
date: 2026-06-21
---

# Quick Task 260621-qix: Native-width resident-bin upload

## Goal

Upload the device-resident bin matrix at its **native uniform width** (u8/u16/u32)
instead of always widening to u32. Primary deliverable = **peak host-memory** (1M×500
drops two ~2GB host allocs — the learner's `to_u32_vec` widen + the backend's u32
concat — to one ~0.5GB native concat) + a ~4× smaller one-time host→device transfer.
Speed is honestly ~1.5% at iters=20 (upload is once-per-train; spike-006 proved the
device-bin *read* is ~0%). Measurement-first; NULL and revert if parity breaks.

## Feasibility (confirmed)

- cubecl 0.10 supports generic launch kernels: `#[cube(launch_unchecked)] fn k<B: Int>(a:
  &Array<B>, …)` launched via `::launch_unchecked::<u8, R>(…)` (cubecl docs gelu_array;
  in-repo `hist_fold_body<N: Numeric>` + concrete wrappers). Bin is used ONLY as an index,
  so `u32::cast_from(resident_bins[idx])` (the repo's existing cast idiom) generalizes it.
- `BinColumn` is already `{U8(Vec<u8>),U16(Vec<u16>),U32(Vec<u32>)}` keyed by num_bin.

## Production resident-buffer readers (the full set — recon done)

3 kernels read the backend's resident `handle`:
1. `construct_leaf_hist_resident_lds_kernel` (histogram.rs:1075) — LDS build
2. `construct_leaf_hist_resident_kernel` (:974) — naive fallback (>256-bin feature)
   — both launched from `resident_raw_build_into` (:1570, sites :1629/:1670)
3. `build_fix_scan_fused_kernel` (~:2560) — launched from `build_fix_scan_resident_f64_on`
The CUDA-mirror kernel (`construct_hist_cuda_mirror_kernel`, :1535) is **test-only**
(rocm_cuda_mirror builds its own u32 buffer) → leave u32, do NOT route the native buffer
through it.

## Task 1 — native-width upload + width plumbing (lgbm-compute/src/lib.rs)

- **action:**
  - Add `#[derive(Clone,Copy)] enum ResidentBinWidth { U8, U16, U32 }` + a helper
    `fn resident_width(cols: &[&BinColumn]) -> ResidentBinWidth` = the widest variant
    present (any U32 → U32; else any U16 → U16; else U8). Add `width` to `ResidentBins`.
  - Change the `Backend::upload_resident_bins` trait signature from `feature_bins:
    &[&[u32]]` to `feature_bins: &[&BinColumn]` (CpuBackend impl stays a no-op match;
    `wants_resident_bins()==false` so it is never called, but must compile).
  - RocmBackend `upload_resident_bins`: pick the uniform width; concat feature-major into
    `Vec<u8>`/`Vec<u16>`/`Vec<u32>` at that width (narrower columns upcast to the uniform
    type via `u8::from`/`u16::from`/`u32::from`); `create_from_slice(<W>::as_bytes(&buf))`;
    store `width`. NO `to_u32_vec`.
  - learner.rs upload block: pass `&[&BinColumn]` (`features.iter().map(|f| &f.bins)`),
    dropping the `to_u32_vec` widen (the host-mem win). Keep the once-per-train guard
    (`resident_bins_uploaded`) and `UPLOAD_NS` timer.
- **verify:** `cargo build` + `cargo build --features rocm` clean.
- **done:** compiles both ways; learner no longer widens to u32.

## Task 2 — genericize the 3 resident kernels over the bin type (histogram.rs)

- **action:** Make `construct_leaf_hist_resident_lds_kernel`,
  `construct_leaf_hist_resident_kernel`, and `build_fix_scan_fused_kernel` generic
  `<B: Int>` with `resident_bins: &Array<B>`, reading via
  `u32::cast_from(resident_bins[idx]) as usize` (numerics-only change; scatter order /
  f32-atomic accumulation byte-unchanged). Thread a `width: ResidentBinWidth` param into
  `resident_raw_build_into` (:1570) and `build_fix_scan_resident_f64_on` (:2360) from the
  backend (read `resident.width`); at each launch site `match width` to call
  `kernel::launch_unchecked::<u8|u16|u32, R>` with `ArrayArg::from_raw_parts::<W>(handle,
  num_features*num_data)` (element COUNT unchanged across widths). Update the safety
  comments (the in-range proof is width-agnostic).
- **verify:** `cargo build --features rocm` clean; **parity gate (HARD):**
  `cargo test -p lgbm-compute --features rocm` (rocm_backend_parity, rocm_row_partition,
  rocm_parallel_histogram, rocm_cuda_mirror), `cargo test -p oracle-harness`,
  `cargo test -p lgbm-treelearner --lib`. All within the ~1e-6 ROCm gate; CPU f64 anchor
  untouched.
- **done:** all parity green; no kernel reads the resident buffer at the wrong width.

## Task 3 — measure (host memory + speed), honest verdict

- **action:** rebuild `--features rocm --example bench_gpu_vs_cpu`; run wide at 1M×500
  iters=4 and iters=20 (`LGBM_PHASE_PROF=1`). Confirm `resident_bin_upload` time drops
  (~4× smaller transfer, all-u8 path since bins=128) and report end-to-end delta honestly.
  Note the peak-host-memory reduction (no 2×2GB widen+concat → 1×~0.5GB native).
- **verify:** numbers recorded; if speed flat that's an accepted NULL-on-speed (memory win
  stands); if parity broke anywhere → revert.
- **done:** SUMMARY has before/after upload + wall-clock + memory framing.

## must_haves

- **truths:** resident buffer is shared by exactly 3 production kernels; all must read the
  chosen width; uniform width = widest variant present; numerics unchanged (index-only).
- **artifacts:** `ResidentBinWidth`, generic `<B: Int>` kernels, width-dispatched launches.
- **key_links:** lib.rs:1852/2039 (ResidentBins/upload), histogram.rs:1570/1629/1670/2360/2560
  (readers), learner.rs upload block, BinColumn lib.rs:52.
